"""Assembly seam from the current pruned checkpoint into H3 block modules.

This assembles the DiT block stack only. It does not claim an end-to-end
generator; the remaining graph and pipeline boundaries are documented.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
import json
from collections.abc import Mapping
from typing import Callable, Iterator, Protocol

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import H3CoreConfig
from .layers import (
    AttentionBackend,
    FusedQKVAttention,
    H3TransformerBlock,
    ModulationSegment,
    SwiGLUMLP,
    torch_sdpa,
)
from .lora import (
    AdaLNCurveRows,
    LowRankUpdate,
    PrunedCurveAdaLN,
    RuntimeLoRALinear,
    SlicedLowRankUpdate,
    interpolate_curve,
)


_block_cancel_check: ContextVar[Callable[[], None] | None] = ContextVar(
    "h3_block_cancel_check", default=None
)


@contextmanager
def block_cancellation(check: Callable[[], None]) -> Iterator[None]:
    """Expose cancellation inside long DiT calls at safe layer boundaries."""

    token = _block_cancel_check.set(check)
    try:
        yield
    finally:
        _block_cancel_check.reset(token)


def _raise_if_block_cancelled() -> None:
    check = _block_cancel_check.get()
    if check is not None:
        check()


from .quantization import (
    ConvRotInt8Linear,
    ConvRotW4A8Linear,
    Int8Kernel,
    W4A8Kernel,
    W4A8QuantSpec,
)


class TensorSource(Protocol):
    def tensor(self, key: str) -> torch.Tensor: ...

    def has_tensor(self, key: str) -> bool: ...

    def quantization_spec(self, prefix: str) -> Mapping[str, object] | None: ...


class MappingTensorSource:
    def __init__(
        self,
        state: Mapping[str, torch.Tensor],
        *,
        quantization_metadata: Mapping[str, Mapping[str, object]] | None = None,
    ) -> None:
        self.state = state
        self._quantization_metadata = dict(quantization_metadata or {})

    def tensor(self, key: str) -> torch.Tensor:
        return self.state[key]

    def has_tensor(self, key: str) -> bool:
        return key in self.state

    def quantization_spec(self, prefix: str) -> Mapping[str, object] | None:
        return self._quantization_metadata.get(prefix)


class SafeTensorSource:
    """Lazy tensor reader. Keep its context open while assembling modules."""

    def __init__(self, path: str) -> None:
        self.path = path
        self._reader = None
        self._keys: set[str] = set()
        self._quantization_metadata: dict[str, Mapping[str, object]] = {}

    def __enter__(self) -> "SafeTensorSource":
        try:
            from safetensors import safe_open
        except ImportError as error:
            raise RuntimeError("safetensors is required to load H3 weights") from error
        self._reader = safe_open(self.path, framework="pt", device="cpu")
        self._reader.__enter__()
        self._keys = set(self._reader.keys())
        metadata = self._reader.metadata() or {}
        raw_quantization = metadata.get("_quantization_metadata")
        if raw_quantization:
            try:
                decoded = json.loads(raw_quantization)
                layers = decoded.get("layers", {})
                if not isinstance(layers, Mapping):
                    raise TypeError("layers must be a mapping")
                self._quantization_metadata = {
                    str(name): dict(spec)
                    for name, spec in layers.items()
                    if isinstance(spec, Mapping)
                }
            except (json.JSONDecodeError, TypeError, ValueError) as error:
                raise ValueError(
                    "invalid safetensors _quantization_metadata"
                ) from error
        return self

    def __exit__(self, *args) -> None:
        assert self._reader is not None
        self._reader.__exit__(*args)
        self._reader = None
        self._keys.clear()
        self._quantization_metadata.clear()

    def tensor(self, key: str) -> torch.Tensor:
        if self._reader is None:
            raise RuntimeError("SafeTensorSource must be used as a context manager")
        return self._reader.get_tensor(key)

    def has_tensor(self, key: str) -> bool:
        if self._reader is None:
            raise RuntimeError("SafeTensorSource must be used as a context manager")
        return key in self._keys

    def quantization_spec(self, prefix: str) -> Mapping[str, object] | None:
        if self._reader is None:
            raise RuntimeError("SafeTensorSource must be used as a context manager")
        return self._quantization_metadata.get(prefix)


class FrozenLinear(nn.Module):
    def __init__(self, weight: torch.Tensor, bias: torch.Tensor | None = None) -> None:
        super().__init__()
        self.weight = nn.Parameter(weight, requires_grad=False)
        self.bias = None if bias is None else nn.Parameter(bias, requires_grad=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return F.linear(value, self.weight, self.bias)


def _quant_linear(
    source: TensorSource,
    prefix: str,
    *,
    device: torch.device | str,
    int8_kernel: Int8Kernel | None,
    w4a8_kernel: W4A8Kernel | None,
    output_dtype: torch.dtype,
) -> nn.Module:
    if source.has_tensor(f"{prefix}.weight_scale"):
        state = {
            f"{prefix}.weight": source.tensor(f"{prefix}.weight").to(device),
            f"{prefix}.weight_scale": source.tensor(
                f"{prefix}.weight_scale"
            ).to(device),
            f"{prefix}.comfy_quant": source.tensor(f"{prefix}.comfy_quant"),
        }
        return ConvRotInt8Linear.from_state_dict(
            state, prefix, kernel=int8_kernel, output_dtype=output_dtype
        )

    required = (
        f"{prefix}.weight_s_rel",
        f"{prefix}.weight_s_channel",
    )
    if not all(source.has_tensor(key) for key in required):
        raise KeyError(
            f"quantized layer {prefix!r} is neither ConvRot INT8 nor W4A8"
        )
    raw_spec = source.quantization_spec(prefix)
    if raw_spec is None:
        # Mapping-backed unit tests can omit file-level safetensors metadata;
        # physical W4 tensor names still have one unambiguous H3 layout.
        raw_spec = {
            "format": "asym_w4a8_int8",
            "group_size": 16,
            "convrot": True,
            "convrot_groupsize": 256,
        }
    spec = W4A8QuantSpec.from_mapping(raw_spec)
    return ConvRotW4A8Linear(
        source.tensor(f"{prefix}.weight").to(device),
        source.tensor(f"{prefix}.weight_s_rel").to(device),
        source.tensor(f"{prefix}.weight_s_channel").to(device),
        codebook=(
            source.tensor(f"{prefix}.weight_codebook").to(device)
            if source.has_tensor(f"{prefix}.weight_codebook")
            else None
        ),
        correction=(
            source.tensor(f"{prefix}.weight_correction").to(device)
            if source.has_tensor(f"{prefix}.weight_correction")
            else None
        ),
        bias=(
            source.tensor(f"{prefix}.bias").to(device)
            if source.has_tensor(f"{prefix}.bias")
            else None
        ),
        spec=spec,
        kernel=w4a8_kernel,
        output_dtype=output_dtype,
    )


def _plain_linear(
    source: TensorSource,
    prefix: str,
    *,
    device: torch.device | str,
    update: LowRankUpdate | None = None,
    bias: bool = True,
    dtype: torch.dtype | None = None,
    source_round_dtype: torch.dtype | None = None,
) -> nn.Module:
    to_kwargs = {"device": device}
    if dtype is not None:
        to_kwargs["dtype"] = dtype

    def convert(value: torch.Tensor) -> torch.Tensor:
        # Comfy first loads the checkpoint into the model's global storage
        # dtype, then manual-casts FP32 islands for execution.  Going directly
        # from checkpoint FP16 to FP32 preserves different low bits and is not
        # trajectory-equivalent for a distilled diffusion model.
        if source_round_dtype is not None:
            value = value.to(source_round_dtype)
        return value.to(**to_kwargs)

    bias_tensor = convert(source.tensor(f"{prefix}.bias")) if bias else None
    base = FrozenLinear(convert(source.tensor(f"{prefix}.weight")), bias_tensor)
    return base if update is None else RuntimeLoRALinear(base, update)


def assemble_pruned_block(
    index: int,
    source: TensorSource,
    *,
    config: H3CoreConfig = H3CoreConfig(),
    device: torch.device | str = "cpu",
    compute_dtype: torch.dtype = torch.bfloat16,
    int8_kernel: Int8Kernel | None = None,
    w4a8_kernel: W4A8Kernel | None = None,
    attention_backend: AttentionBackend = torch_sdpa,
    lora_updates: Mapping[str, LowRankUpdate | SlicedLowRankUpdate] | None = None,
) -> H3TransformerBlock:
    """Load one block, which is the seam needed by block-wise offload."""

    if not 0 <= index < config.num_layers:
        raise IndexError(index)
    prefix = f"blocks.{index}"
    updates = {} if lora_updates is None else lora_updates

    def quant(name: str) -> nn.Module:
        module_name = f"{prefix}.{name}"
        base = _quant_linear(
            source,
            module_name,
            device=device,
            int8_kernel=int8_kernel,
            w4a8_kernel=w4a8_kernel,
            output_dtype=compute_dtype,
        )
        update = updates.get(module_name)
        if update is None:
            return base
        return RuntimeLoRALinear(base, update)

    attention = FusedQKVAttention(
        quant("attn.qkv_proj"),
        quant("attn.out_proj"),
        num_heads=config.num_heads,
        head_dim=config.head_dim,
        qk_eps=config.qk_norm_eps,
        backend=attention_backend,
        device=device,
        dtype=compute_dtype,
    )
    with torch.no_grad():
        attention.q_norm.weight.copy_(source.tensor(f"{prefix}.attn.q_norm.weight").to(device))
        attention.k_norm.weight.copy_(source.tensor(f"{prefix}.attn.k_norm.weight").to(device))

    adaln_name = f"{prefix}.adaln_proj.linear"
    adaln_base = FrozenLinear(
        source.tensor(f"{adaln_name}.weight")
        .to(compute_dtype)
        .to(device=device, dtype=torch.float32),
        source.tensor(f"{adaln_name}.bias")
        .to(compute_dtype)
        .to(device=device, dtype=torch.float32),
    )
    adaln = PrunedCurveAdaLN(
        adaln_base,
        compressed_curve=None,
        lora_update=updates.get(adaln_name),
    )
    block = H3TransformerBlock(
        attention,
        SwiGLUMLP(quant("mlp.fc1"), quant("mlp.fc2")),
        adaln,
        hidden_size=config.hidden_size,
        norm_eps=config.norm_eps,
        device=device,
        dtype=compute_dtype,
    )
    with torch.no_grad():
        block.norm1.weight.copy_(source.tensor(f"{prefix}.norm1.weight").to(device))
        block.norm2.weight.copy_(source.tensor(f"{prefix}.norm2.weight").to(device))
    return block


class H3BlockStack(nn.Module):
    """Block stack sharing curve interpolation across all blocks."""

    def __init__(
        self,
        blocks: list[H3TransformerBlock],
        compressed_curve: torch.Tensor,
        full_silu_curve: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(blocks)
        self.register_buffer("compressed_curve", compressed_curve)
        self.register_buffer("full_silu_curve", full_silu_curve)
        self._block_executor = None
        self._block_offload_start = 0

    def configure_block_executor(self, executor, *, offload_start: int = 0) -> None:
        """Select host-source/two-device-buffer execution for this stack."""

        if not 0 <= offload_start < len(self.blocks):
            raise ValueError("offload_start must identify a block in the stack")
        self._block_executor = executor
        self._block_offload_start = int(offload_start)

    def clear_block_executor(self) -> None:
        self._block_executor = None
        self._block_offload_start = 0

    def prepare_curve_rows(self, unique_timesteps: torch.Tensor) -> AdaLNCurveRows:
        return AdaLNCurveRows(
            compressed=interpolate_curve(self.compressed_curve, unique_timesteps),
            full_silu=(
                None
                if self.full_silu_curve is None
                else interpolate_curve(self.full_silu_curve, unique_timesteps)
            ),
        )

    def forward(
        self,
        value: torch.Tensor,
        *,
        unique_timesteps: torch.Tensor,
        modulation_segments: tuple[ModulationSegment, ...],
        frequencies: torch.Tensor,
        curve_rows: AdaLNCurveRows | None = None,
        mlp_chunk_tokens: int | None = None,
    ) -> torch.Tensor:
        rows = self.prepare_curve_rows(unique_timesteps) if curve_rows is None else curve_rows
        return self.run_range(
            value,
            start=0,
            stop=len(self.blocks),
            timestep_rows=rows,
            modulation_segments=modulation_segments,
            frequencies=frequencies,
            mlp_chunk_tokens=mlp_chunk_tokens,
        )

    def run_range(
        self,
        value: torch.Tensor,
        *,
        start: int,
        stop: int,
        timestep_rows: AdaLNCurveRows,
        modulation_segments: tuple[ModulationSegment, ...],
        frequencies: torch.Tensor,
        mlp_chunk_tokens: int | None = None,
    ) -> torch.Tensor:
        if not 0 <= start <= stop <= len(self.blocks):
            raise IndexError(f"invalid H3 block range [{start}, {stop})")
        if self._block_executor is None:
            from .kernels import attention_layer

            for layer_index, block in enumerate(self.blocks[start:stop], start=start):
                _raise_if_block_cancelled()
                with attention_layer(layer_index):
                    value = block(
                        value,
                        timestep_rows=timestep_rows,
                        modulation_segments=modulation_segments,
                        frequencies=frequencies,
                        mlp_chunk_tokens=mlp_chunk_tokens,
                    )
            return value

        # A hybrid plan keeps a prefix resident and streams only the suffix.
        # This spends otherwise-idle VRAM to reduce repeated PCIe traffic
        # without changing any block math.
        resident_stop = min(stop, self._block_offload_start)
        from .kernels import attention_layer

        for layer_index, block in enumerate(
            self.blocks[start:resident_stop], start=start
        ):
            _raise_if_block_cancelled()
            with attention_layer(layer_index):
                value = block(
                    value,
                    timestep_rows=timestep_rows,
                    modulation_segments=modulation_segments,
                    frequencies=frequencies,
                    mlp_chunk_tokens=mlp_chunk_tokens,
                )
        stream_start = max(start, self._block_offload_start)
        if stream_start >= stop:
            return value
        sources = self.blocks[stream_start:stop]

        shared = {
            "timestep_rows": timestep_rows,
            "modulation_segments": modulation_segments,
            "frequencies": frequencies,
            "mlp_chunk_tokens": mlp_chunk_tokens,
        }

        def run_block(index, buffer, hidden, inputs):
            _raise_if_block_cancelled()
            with attention_layer(index):
                try:
                    return buffer.module(hidden, **inputs)
                except torch.OutOfMemoryError as error:
                    raise torch.OutOfMemoryError(
                        f"H3 block {index} exceeded the active device budget: {error}"
                    ) from error

        return self._block_executor.run(
            sources,
            value,
            run_block,
            shared,
            block_index_offset=stream_start,
        )

    def run_protected_range(
        self,
        value: torch.Tensor,
        *,
        start: int,
        stop: int,
        protected_tokens: int,
        active_video_indices: torch.Tensor | None = None,
        active_video_layer_start: int = 0,
        active_video_layer_stop: int = 50,
        timestep_rows: AdaLNCurveRows,
        modulation_segments: tuple[ModulationSegment, ...],
        frequencies: torch.Tensor,
        mlp_chunk_tokens: int | None = None,
    ) -> torch.Tensor:
        """Refresh only the packed prefix while preserving generated video rows.

        ``mlp_chunk_tokens`` is accepted for API symmetry; the protected prefix
        is small enough that its MLP is deliberately evaluated in one piece.
        """

        del mlp_chunk_tokens
        if not 0 <= start <= stop <= len(self.blocks):
            raise IndexError(f"invalid H3 protected block range [{start}, {stop})")
        if active_video_indices is not None and not (
            start
            <= active_video_layer_start
            < active_video_layer_stop
            <= stop
        ):
            raise IndexError(
                "active video layer range must lie inside the protected block range"
            )
        from .kernels import attention_layer
        from .modality_refresh import (
            refresh_protected_modalities,
            refresh_selected_video_tiles,
        )

        def run_one(layer_index, block, hidden):
            _raise_if_block_cancelled()
            with attention_layer(layer_index):
                layer_active = (
                    active_video_indices is not None
                    and active_video_layer_start
                    <= layer_index
                    < active_video_layer_stop
                )
                if not layer_active:
                    return refresh_protected_modalities(
                        block,
                        hidden,
                        protected_tokens=protected_tokens,
                        timestep_rows=timestep_rows,
                        modulation_segments=modulation_segments,
                        frequencies=frequencies,
                    )
                return refresh_selected_video_tiles(
                    block,
                    hidden,
                    protected_tokens=protected_tokens,
                    active_video_indices=active_video_indices,
                    timestep_rows=timestep_rows,
                    modulation_segments=modulation_segments,
                    frequencies=frequencies,
                )

        if self._block_executor is None:
            for layer_index, block in enumerate(self.blocks[start:stop], start=start):
                value = run_one(layer_index, block, value)
            return value

        resident_stop = min(stop, self._block_offload_start)
        for layer_index, block in enumerate(
            self.blocks[start:resident_stop], start=start
        ):
            value = run_one(layer_index, block, value)
        stream_start = max(start, self._block_offload_start)
        if stream_start >= stop:
            return value
        sources = self.blocks[stream_start:stop]
        shared = {
            "protected_tokens": protected_tokens,
            "active_video_indices": active_video_indices,
            "active_video_layer_start": active_video_layer_start,
            "active_video_layer_stop": active_video_layer_stop,
            "timestep_rows": timestep_rows,
            "modulation_segments": modulation_segments,
            "frequencies": frequencies,
        }

        def run_block(index, buffer, hidden, inputs):
            _raise_if_block_cancelled()
            with attention_layer(index):
                active = inputs["active_video_indices"]
                layer_active = (
                    active is not None
                    and inputs["active_video_layer_start"]
                    <= index
                    < inputs["active_video_layer_stop"]
                )
                refresh_inputs = {
                    key: item
                    for key, item in inputs.items()
                    if key
                    not in (
                        "active_video_indices",
                        "active_video_layer_start",
                        "active_video_layer_stop",
                    )
                }
                if not layer_active:
                    return refresh_protected_modalities(
                        buffer.module, hidden, **refresh_inputs
                    )
                return refresh_selected_video_tiles(
                    buffer.module,
                    hidden,
                    active_video_indices=active,
                    **refresh_inputs,
                )

        return self._block_executor.run(
            sources,
            value,
            run_block,
            shared,
            block_index_offset=stream_start,
        )


def assemble_pruned_block_stack(
    source: TensorSource,
    *,
    config: H3CoreConfig = H3CoreConfig(),
    device: torch.device | str = "cpu",
    compute_dtype: torch.dtype = torch.bfloat16,
    int8_kernel: Int8Kernel | None = None,
    w4a8_kernel: W4A8Kernel | None = None,
    attention_backend: AttentionBackend = torch_sdpa,
    lora_updates: Mapping[str, LowRankUpdate | SlicedLowRankUpdate] | None = None,
    full_silu_curve: torch.Tensor | None = None,
) -> H3BlockStack:
    """Assemble on CPU for 24 GB plans; direct all-CUDA assembly will OOM."""

    if full_silu_curve is None and lora_updates and any(
        name.endswith(".adaln_proj.linear") for name in lora_updates
    ):
        raise ValueError("Larry AdaLN updates require the full SiLU time-embedding curve")
    blocks = [
        assemble_pruned_block(
            index,
            source,
            config=config,
            device=device,
            compute_dtype=compute_dtype,
            int8_kernel=int8_kernel,
            w4a8_kernel=w4a8_kernel,
            attention_backend=attention_backend,
            lora_updates=lora_updates,
        )
        for index in range(config.num_layers)
    ]
    return H3BlockStack(
        blocks,
        source.tensor("adaln_t_table").to(device),
        None if full_silu_curve is None else full_silu_curve.to(device),
    )


def assemble_token_refiner(
    source: TensorSource,
    *,
    config: H3CoreConfig = H3CoreConfig(),
    device: torch.device | str = "cpu",
    compute_dtype: torch.dtype = torch.bfloat16,
    attention_backend: AttentionBackend = torch_sdpa,
    lora_updates: Mapping[str, LowRankUpdate | SlicedLowRankUpdate] | None = None,
):
    from .dit import H3TokenRefiner, H3TokenRefinerBlock
    from .layers import RMSNorm

    updates = {} if lora_updates is None else lora_updates
    blocks = []
    for index in range(config.token_refiner_layers):
        prefix = f"token_refiner.blocks.{index}"

        def linear(name: str) -> nn.Module:
            module_name = f"{prefix}.{name}"
            return _plain_linear(
                source,
                module_name,
                device=device,
                update=updates.get(module_name),
                bias=False,
            )

        attention = FusedQKVAttention(
            linear("attn.qkv_proj"),
            linear("attn.out_proj"),
            num_heads=config.num_heads,
            head_dim=config.head_dim,
            qk_eps=config.qk_norm_eps,
            backend=attention_backend,
            device=device,
            dtype=compute_dtype,
        )
        block = H3TokenRefinerBlock(
            attention,
            SwiGLUMLP(linear("mlp.fc1"), linear("mlp.fc2")),
            hidden_size=config.hidden_size,
            norm_eps=config.norm_eps,
            device=device,
            dtype=compute_dtype,
        )
        with torch.no_grad():
            attention.q_norm.weight.copy_(
                source.tensor(f"{prefix}.attn.q_norm.weight").to(device)
            )
            attention.k_norm.weight.copy_(
                source.tensor(f"{prefix}.attn.k_norm.weight").to(device)
            )
            block.norm1.weight.copy_(source.tensor(f"{prefix}.norm1.weight").to(device))
            block.norm2.weight.copy_(source.tensor(f"{prefix}.norm2.weight").to(device))
        blocks.append(block)
    final_norm = RMSNorm(
        config.hidden_size, config.norm_eps, device=device, dtype=compute_dtype
    )
    with torch.no_grad():
        final_norm.weight.copy_(source.tensor("token_refiner.final_norm.weight").to(device))
    return H3TokenRefiner(blocks, final_norm)


def assemble_full_pruned_dit(
    source: TensorSource,
    *,
    config: H3CoreConfig = H3CoreConfig(),
    device: torch.device | str = "cpu",
    compute_dtype: torch.dtype = torch.bfloat16,
    int8_kernel: Int8Kernel | None = None,
    w4a8_kernel: W4A8Kernel | None = None,
    attention_backend: AttentionBackend = torch_sdpa,
    lora_updates: Mapping[str, LowRankUpdate | SlicedLowRankUpdate] | None = None,
    full_silu_curve: torch.Tensor | None = None,
):
    """Assemble the complete pruned DiT graph, but no encoder/VAE/scheduler."""

    from .dit import FullH3DiT, H3FinalLayer
    from .layers import RMSNorm

    updates = {} if lora_updates is None else lora_updates
    stack = assemble_pruned_block_stack(
        source,
        config=config,
        device=device,
        compute_dtype=compute_dtype,
        int8_kernel=int8_kernel,
        w4a8_kernel=w4a8_kernel,
        attention_backend=attention_backend,
        lora_updates=updates,
        full_silu_curve=full_silu_curve,
    )
    refiner = assemble_token_refiner(
        source,
        config=config,
        device=device,
        compute_dtype=compute_dtype,
        attention_backend=attention_backend,
        lora_updates=updates,
    )
    final_name = "final_layer.adaln_proj.linear"
    final_adaln = PrunedCurveAdaLN(
        _plain_linear(
            source,
            final_name,
            device=device,
            dtype=torch.float32,
            source_round_dtype=compute_dtype,
        ),
        compressed_curve=None,
        lora_update=updates.get(final_name),
    )
    final_norm = RMSNorm(
        config.hidden_size, config.norm_eps, device=device, dtype=compute_dtype
    )
    with torch.no_grad():
        final_norm.weight.copy_(source.tensor("final_layer.norm.weight").to(device))
    final = H3FinalLayer(
        final_norm,
        final_adaln,
        _plain_linear(
            source,
            "final_layer.video_out",
            device=device,
            dtype=torch.float32,
            source_round_dtype=compute_dtype,
        ),
        _plain_linear(
            source,
            "final_layer.audio_out",
            device=device,
            dtype=torch.float32,
            source_round_dtype=compute_dtype,
        ),
        hidden_size=config.hidden_size,
    )
    return FullH3DiT(
        config=config,
        video_patch_proj=_plain_linear(
            source,
            "video_patch_proj",
            device=device,
            dtype=torch.float32,
            source_round_dtype=compute_dtype,
        ),
        audio_patch_proj=_plain_linear(
            source,
            "audio_patch_proj",
            device=device,
            dtype=torch.float32,
            source_round_dtype=compute_dtype,
        ),
        condition_proj=_plain_linear(source, "condition_proj", device=device),
        token_refiner=refiner,
        block_stack=stack,
        final_layer=final,
        rope_inv_freq=source.tensor("rope.inv_freq").to(device),
        compute_dtype=compute_dtype,
    )


def load_full_silu_curve(path: str, key: str = "silu_t_emb_grid") -> torch.Tensor:
    try:
        from safetensors import safe_open
    except ImportError as error:
        raise RuntimeError("safetensors is required to load the full AdaLN curve") from error
    with safe_open(path, framework="pt", device="cpu") as checkpoint:
        if key not in checkpoint.keys():
            raise KeyError(key)
        curve = checkpoint.get_tensor(key)
    if tuple(curve.shape) != (1025, 2688):
        raise ValueError(f"full SiLU curve shape {tuple(curve.shape)} != (1025, 2688)")
    return curve
