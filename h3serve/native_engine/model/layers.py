"""Lightweight single-GPU H3 transformer layers.

The fused-QKV graph, indexed AdaLN formulation and partial 3-D RoPE are adapted
from SGLang's Apache-2.0 MiniMax H3 implementation, rewritten here using only
PyTorch primitives and injected linear/attention backends. Modified for a
batch-one, no-TP, no-distributed serving target.
"""

from __future__ import annotations

import math
import os
from contextlib import nullcontext
from typing import Callable, TypeAlias

import torch
import torch.nn as nn
import torch.nn.functional as F

AttentionBackend = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]
ModulationSegment: TypeAlias = tuple[int, int, int]
_PROFILE_REGIONS_ENABLED = os.environ.get("H3_NATIVE_PROFILE_REGIONS", "0") == "1"


def _profile_region(name: str):
    """Expose opt-in Kineto regions without taxing production requests."""

    if _PROFILE_REGIONS_ENABLED:
        return torch.profiler.record_function(name)
    return nullcontext()


class RMSNorm(nn.Module):
    def __init__(self, width: int, eps: float = 1e-5, *, device=None, dtype=None) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(width, device=device, dtype=dtype))
        self.eps = eps

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        # Match the mature H3 reference path.  Besides using PyTorch's fused
        # reduction, this avoids materializing two full FP32 copies of the
        # packed [tokens, hidden] activation in every transformer block.
        return F.rms_norm(
            value,
            (self.weight.shape[0],),
            weight=self.weight.to(dtype=value.dtype),
            eps=self.eps,
        )


def rope_frequencies(
    position_ids: torch.Tensor, inv_freq: torch.Tensor
) -> torch.Tensor:
    if position_ids.ndim != 2 or position_ids.shape[-1] != 3:
        raise ValueError("position_ids must be [sequence,3]")
    per_axis = position_ids.float().unsqueeze(-1) * inv_freq.float().reshape(1, 1, -1)
    half = torch.cat(per_axis.unbind(dim=1), dim=-1)
    return torch.cat((half, half), dim=-1)


def rope_rotation_table(angles: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Pack split-half angles once for all transformer blocks."""

    half = angles.shape[-1] // 2
    active = angles[:, :half]
    cosine, sine = torch.cos(active), torch.sin(active)
    return torch.stack((cosine, -sine, sine, cosine), dim=-1).reshape(
        1, angles.shape[0], 1, half, 2, 2
    ).to(dtype)


def apply_qknorm_rope(
    query: torch.Tensor,
    key: torch.Tensor,
    *,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    frequencies: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference order: per-head RMSNorm, then split-half partial RoPE."""

    if frequencies.ndim == 6:
        rotate_width = int(frequencies.shape[-3]) * 2
        if query.is_cuda and query.dtype in (torch.float16, torch.bfloat16):
            from comfy_kitchen import rms_rope_split_half_

            query_4d, key_4d = query.unsqueeze(0), key.unsqueeze(0)
            rms_rope_split_half_(
                query_4d,
                key_4d,
                frequencies,
                q_weight.to(device=query.device, dtype=query.dtype),
                k_weight.to(device=key.device, dtype=key.dtype),
                epsilon=eps,
                rot_dim=rotate_width,
            )
            return query_4d[0], key_4d[0]

        query = F.rms_norm(
            query, (query.shape[-1],), weight=q_weight.to(query.dtype), eps=eps
        )
        key = F.rms_norm(
            key, (key.shape[-1],), weight=k_weight.to(key.dtype), eps=eps
        )

        def rotate_table(value: torch.Tensor) -> torch.Tensor:
            half_width = rotate_width // 2
            first = value[..., :half_width]
            second = value[..., half_width:rotate_width]
            table = frequencies[0, :, 0].to(value.dtype)
            out_first = (
                first * table[:, None, :, 0, 0]
                + second * table[:, None, :, 0, 1]
            )
            out_second = (
                first * table[:, None, :, 1, 0]
                + second * table[:, None, :, 1, 1]
            )
            return torch.cat(
                (out_first, out_second, value[..., rotate_width:]), dim=-1
            )

        return rotate_table(query), rotate_table(key)

    def normalize(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(
            value,
            (value.shape[-1],),
            weight=weight.to(dtype=value.dtype),
            eps=eps,
        )

    query, key = normalize(query, q_weight), normalize(key, k_weight)
    rotate_width = int(frequencies.shape[-1])
    half = rotate_width // 2
    cosine = torch.cos(frequencies[:, :half]).to(query.dtype)
    sine = torch.sin(frequencies[:, :half]).to(query.dtype)
    cosine = torch.cat((cosine, cosine), dim=-1).unsqueeze(1)
    sine = torch.cat((sine, sine), dim=-1).unsqueeze(1)

    def rotate(value: torch.Tensor) -> torch.Tensor:
        active, passthrough = value[..., :rotate_width], value[..., rotate_width:]
        first, second = active.chunk(2, dim=-1)
        perpendicular = torch.cat((-second, first), dim=-1)
        active = active * cosine + perpendicular * sine
        return torch.cat((active, passthrough), dim=-1)

    return rotate(query), rotate(key)


def _apply_single_qknorm_rope(
    value: torch.Tensor,
    *,
    weight: torch.Tensor,
    frequencies: torch.Tensor,
    eps: float,
    query_slot: bool,
) -> torch.Tensor:
    """Apply one side of QK-Norm/RoPE with the established kernel order."""

    if frequencies.ndim == 6:
        rotate_width = int(frequencies.shape[-3]) * 2
        if value.is_cuda and value.dtype in (torch.float16, torch.bfloat16):
            from .kernels import current_long_sequence_single_qknorm_rope

            if current_long_sequence_single_qknorm_rope():
                from .single_qknorm_rope import try_apply_single_qknorm_rope_

                if try_apply_single_qknorm_rope_(
                    value,
                    weight=weight.to(device=value.device, dtype=value.dtype),
                    frequencies=frequencies,
                    eps=eps,
                ):
                    return value
            from comfy_kitchen import rms_rope_split_half_

            value_4d = value.unsqueeze(0)
            scratch = torch.empty_like(value_4d)
            normalized_weight = weight.to(device=value.device, dtype=value.dtype)
            if query_slot:
                rms_rope_split_half_(
                    value_4d,
                    scratch,
                    frequencies,
                    normalized_weight,
                    normalized_weight,
                    epsilon=eps,
                    rot_dim=rotate_width,
                )
            else:
                rms_rope_split_half_(
                    scratch,
                    value_4d,
                    frequencies,
                    normalized_weight,
                    normalized_weight,
                    epsilon=eps,
                    rot_dim=rotate_width,
                )
            return value_4d[0]

        value = F.rms_norm(
            value,
            (value.shape[-1],),
            weight=weight.to(value.dtype),
            eps=eps,
        )
        half_width = rotate_width // 2
        first = value[..., :half_width]
        second = value[..., half_width:rotate_width]
        table = frequencies[0, :, 0].to(value.dtype)
        out_first = (
            first * table[:, None, :, 0, 0]
            + second * table[:, None, :, 0, 1]
        )
        out_second = (
            first * table[:, None, :, 1, 0]
            + second * table[:, None, :, 1, 1]
        )
        return torch.cat(
            (out_first, out_second, value[..., rotate_width:]), dim=-1
        )

    value = F.rms_norm(
        value,
        (value.shape[-1],),
        weight=weight.to(dtype=value.dtype),
        eps=eps,
    )
    rotate_width = int(frequencies.shape[-1])
    half = rotate_width // 2
    cosine = torch.cos(frequencies[:, :half]).to(value.dtype)
    sine = torch.sin(frequencies[:, :half]).to(value.dtype)
    cosine = torch.cat((cosine, cosine), dim=-1).unsqueeze(1)
    sine = torch.cat((sine, sine), dim=-1).unsqueeze(1)
    active, passthrough = value[..., :rotate_width], value[..., rotate_width:]
    first, second = active.chunk(2, dim=-1)
    perpendicular = torch.cat((-second, first), dim=-1)
    active = active * cosine + perpendicular * sine
    return torch.cat((active, passthrough), dim=-1)


def torch_sdpa(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    q = query.transpose(0, 1).unsqueeze(0)
    k = key.transpose(0, 1).unsqueeze(0)
    v = value.transpose(0, 1).unsqueeze(0)
    output = F.scaled_dot_product_attention(q, k, v, is_causal=False)
    return output.squeeze(0).transpose(0, 1).contiguous()


class FusedQKVAttention(nn.Module):
    def __init__(
        self,
        qkv_proj: nn.Module,
        out_proj: nn.Module,
        *,
        num_heads: int,
        head_dim: int,
        qk_eps: float = 1e-5,
        backend: AttentionBackend = torch_sdpa,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.qkv_proj = qkv_proj
        self.out_proj = out_proj
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.inner_width = num_heads * head_dim
        self.q_norm = RMSNorm(head_dim, qk_eps, device=device, dtype=dtype)
        self.k_norm = RMSNorm(head_dim, qk_eps, device=device, dtype=dtype)
        self.backend = backend

    def forward(self, value: torch.Tensor, frequencies: torch.Tensor | None) -> torch.Tensor:
        qkv = self.qkv_proj(value)
        expected = 3 * self.inner_width
        if qkv.shape[-1] != expected:
            raise ValueError(f"fused QKV output width {qkv.shape[-1]} != {expected}")
        query, key, val = qkv.split(self.inner_width, dim=-1)
        shape = (value.shape[0], self.num_heads, self.head_dim)
        query, key, val = query.view(shape), key.view(shape), val.view(shape)
        if frequencies is None:
            zero = value.new_zeros((value.shape[0], 0), dtype=torch.float32)
            frequencies = zero
        if frequencies.shape[-1]:
            query, key = apply_qknorm_rope(
                query,
                key,
                q_weight=self.q_norm.weight,
                k_weight=self.k_norm.weight,
                frequencies=frequencies,
                eps=self.q_norm.eps,
            )
        else:
            query = self.q_norm(query)
            key = self.k_norm(key)
        attended = self.backend(query, key, val).reshape(value.shape[0], self.inner_width)
        return self.out_proj(attended)

    def stream_gated_residual_(
        self,
        residual: torch.Tensor,
        hidden: torch.Tensor,
        frequencies: torch.Tensor | None,
        gate: torch.Tensor,
        modulation_segments: tuple[ModulationSegment, ...],
        *,
        query_chunk_tokens: int,
        projection_chunk_tokens: int = 8192,
    ) -> bool:
        """Run one memory-bounded fixed-TopK layer and update ``residual``.

        The normal fused projection deliberately materializes one
        ``[L, 3*attention_width]`` allocation.  At the 220k-token 1080p/15s
        shape that tensor alone is 8.8 GiB and overlaps Sparge's selector.
        This route projects modest row chunks into separately owned K/V
        tensors, drops unquantized V before K pooling, then recomputes Query in
        bounded chunks and immediately applies each output chunk's gated
        residual.  Avoiding a sequence-long Query costs one extra row-local
        QKV pass but leaves headroom for request-dependent reference prefixes.
        No sampler or model-weight math is changed; unsupported physical
        backends fail closed to the old path.
        """

        if query_chunk_tokens < 128 or query_chunk_tokens % 128:
            raise ValueError("streamed Attention chunks must be multiples of 128")
        if projection_chunk_tokens <= 0:
            raise ValueError("streamed projection chunks must be positive")
        projection_chunk_tokens = min(
            int(projection_chunk_tokens), query_chunk_tokens
        )
        from .kernels import _resolve_long_sequence_physical_backend

        physical = _resolve_long_sequence_physical_backend(
            self.backend, int(hidden.shape[0])
        )
        if physical is None:
            return False
        prepare_values = getattr(physical, "prepare_long_sequence_values", None)
        prepare_keys = getattr(physical, "prepare_long_sequence_keys", None)
        prefix_attention = getattr(
            physical, "long_sequence_prefix_queries", None
        )
        video_attention = getattr(physical, "long_sequence_video_queries", None)
        if any(
            method is None
            for method in (
                prepare_values,
                prepare_keys,
                prefix_attention,
                video_attention,
            )
        ):
            return False
        if residual.shape != hidden.shape or hidden.ndim != 2:
            raise ValueError("streamed Attention requires aligned rank-two hidden state")
        sequence = int(hidden.shape[0])
        from .kernels import (
            current_attention_protected_prefix,
            current_attention_video_layout,
        )

        segment_prefix = int(modulation_segments[-1][0])
        layout_prefix = current_attention_protected_prefix()
        protected_tokens = layout_prefix if layout_prefix > 0 else segment_prefix
        # A native continuation adds one clean-timestep generated-video run
        # before the noisy suffix.  In that case the sparse protected boundary
        # is the start of the final run, not necessarily the start of the
        # complete generated-video modality.  Requiring an actual modulation
        # boundary retains the stale-layout guard without hard-coding a single
        # video run.
        modulation_boundaries = {int(start) for start, _, _ in modulation_segments}
        if layout_prefix > 0 and layout_prefix not in modulation_boundaries:
            raise ValueError(
                "packed layout prefix does not match the generated-video "
                "modulation segment"
            )
        if not 0 < protected_tokens < sequence:
            return False
        video_layout = current_attention_video_layout()
        if video_layout is not None:
            latent_frames, frame_tokens = video_layout
            if protected_tokens + latent_frames * frame_tokens != sequence:
                raise ValueError(
                    "request-local video geometry does not cover the packed "
                    "sequence"
                )

        from .kernels import (
            current_long_sequence_compact_kv,
            current_long_sequence_direct_hnd_fp8_value,
            current_long_sequence_direct_nhd_kv,
            current_long_sequence_fused_qknorm_hnd_layout,
        )

        from .long_sequence_contract import (
            resolve_physical_long_sequence_contract,
        )

        physical_contract = resolve_physical_long_sequence_contract(
            physical,
            compact_kv=current_long_sequence_compact_kv(),
            direct_nhd_kv_requested=current_long_sequence_direct_nhd_kv(),
            fused_qknorm_hnd_requested=(
                current_long_sequence_fused_qknorm_hnd_layout()
            ),
            direct_hnd_fp8_value_requested=(
                current_long_sequence_direct_hnd_fp8_value()
            ),
        )
        kv_layout = physical_contract.kv_layout
        if kv_layout == "HND":
            kv_shape = (1, self.num_heads, sequence, self.head_dim)
        elif kv_layout == "NHD":
            kv_shape = (1, sequence, self.num_heads, self.head_dim)
        fused_qknorm_hnd_layout = (
            physical_contract.fused_qknorm_hnd_layout
        )
        expected = 3 * self.inner_width
        from .kernels import (
            current_long_sequence_shared_qkv_quantization,
            current_long_sequence_split_qkv_outputs,
        )

        output_slice = getattr(self.qkv_proj, "forward_output_slice", None)
        split_qkv = (
            current_long_sequence_split_qkv_outputs()
            and callable(output_slice)
        )
        compact_kv = current_long_sequence_compact_kv()
        if compact_kv and not split_qkv:
            raise RuntimeError(
                "compact K/V requires output-sliced Q/K/V projection"
            )
        prepare_output_slices = getattr(
            self.qkv_proj, "prepare_output_slices", None
        )
        prepared_output_slice = getattr(
            self.qkv_proj, "forward_prepared_output_slice", None
        )
        shared_qkv_quantization = bool(
            split_qkv
            and current_long_sequence_shared_qkv_quantization()
            and callable(prepare_output_slices)
            and callable(prepared_output_slice)
        )
        prepared_qkv_input = None
        if shared_qkv_quantization:
            with _profile_region("h3_long_shared_qkv_quantize"):
                prepared_qkv_input = prepare_output_slices(hidden)

        def project_kv_chunk(start: int, stop: int):
            """Project and normalize one row slab for either K/V build."""

            with _profile_region("h3_long_kv_qkv_projection"):
                if prepared_qkv_input is None:
                    kv_chunk = output_slice(
                        hidden[start:stop], self.inner_width, expected
                    )
                else:
                    kv_chunk = prepared_output_slice(
                        hidden,
                        prepared_qkv_input,
                        start,
                        stop,
                        self.inner_width,
                        expected,
                    )
            if kv_chunk.shape[-1] != 2 * self.inner_width:
                raise ValueError("split K/V projection has invalid width")
            k_chunk, v_chunk = kv_chunk.split(self.inner_width, dim=-1)
            chunk_shape = (stop - start, self.num_heads, self.head_dim)
            k_chunk = k_chunk.view(chunk_shape)
            v_chunk = v_chunk.view(chunk_shape).to(
                getattr(physical, "long_sequence_value_dtype", hidden.dtype)
            )
            if frequencies is None or not frequencies.shape[-1]:
                k_chunk = self.k_norm(k_chunk)
            else:
                chunk_frequencies = (
                    frequencies[:, start:stop]
                    if frequencies.ndim == 6
                    else frequencies[start:stop]
                )
                k_chunk = _apply_single_qknorm_rope(
                    k_chunk,
                    weight=self.k_norm.weight,
                    frequencies=chunk_frequencies,
                    eps=self.k_norm.eps,
                    query_slot=False,
                )
            return kv_chunk, k_chunk, v_chunk

        if compact_kv:
            begin_compact = getattr(
                physical, "begin_compact_long_sequence_kv", None
            )
            if not callable(begin_compact):
                return False
            key_sum = torch.zeros(
                (self.num_heads, self.head_dim),
                device=hidden.device,
                dtype=torch.float32,
            )
            value_absmax = torch.zeros(
                (self.num_heads, self.head_dim),
                device=hidden.device,
                dtype=torch.float32,
            )
            for start in range(0, sequence, projection_chunk_tokens):
                stop = min(sequence, start + projection_chunk_tokens)
                kv_chunk, k_chunk, v_chunk = project_kv_chunk(start, stop)
                with _profile_region("h3_long_compact_kv_statistics"):
                    key_sum.add_(k_chunk.detach().float().sum(dim=0))
                    torch.maximum(
                        value_absmax,
                        v_chunk.detach().float().abs().amax(dim=0),
                        out=value_absmax,
                    )
                del kv_chunk, k_chunk, v_chunk
            key_mean = key_sum.div_(sequence)
            builder = begin_compact(
                key_tokens=sequence,
                heads=self.num_heads,
                head_dim=self.head_dim,
                key_mean=key_mean,
                value_absmax=value_absmax,
                device=hidden.device,
            )
            del key_sum, key_mean, value_absmax
            for start in range(0, sequence, projection_chunk_tokens):
                stop = min(sequence, start + projection_chunk_tokens)
                kv_chunk, k_chunk, v_chunk = project_kv_chunk(start, stop)
                with _profile_region("h3_long_compact_kv_quantize"):
                    builder.add(start, k_chunk, v_chunk)
                del kv_chunk, k_chunk, v_chunk
            prepared = builder.finish()
            del builder
        else:
            key_hnd = torch.empty(
                kv_shape, device=hidden.device, dtype=torch.bfloat16
            )
            value_hnd = torch.empty(
                kv_shape,
                device=hidden.device,
                dtype=getattr(physical, "long_sequence_value_dtype", hidden.dtype),
            )
        projection_starts = (
            ()
            if compact_kv
            else range(0, sequence, projection_chunk_tokens)
        )
        for start in projection_starts:
            stop = min(sequence, start + projection_chunk_tokens)
            with _profile_region("h3_long_kv_qkv_projection"):
                if split_qkv:
                    if prepared_qkv_input is None:
                        kv_chunk = output_slice(
                            hidden[start:stop], self.inner_width, expected
                        )
                    else:
                        kv_chunk = prepared_output_slice(
                            hidden,
                            prepared_qkv_input,
                            start,
                            stop,
                            self.inner_width,
                            expected,
                        )
                else:
                    qkv = self.qkv_proj(hidden[start:stop])
            if split_qkv:
                if kv_chunk.shape[-1] != 2 * self.inner_width:
                    raise ValueError("split K/V projection has invalid width")
                k_chunk, v_chunk = kv_chunk.split(self.inner_width, dim=-1)
            else:
                if qkv.shape[-1] != expected:
                    raise ValueError(
                        f"fused QKV output width {qkv.shape[-1]} != {expected}"
                    )
                q_chunk, k_chunk, v_chunk = qkv.split(self.inner_width, dim=-1)
            chunk_shape = (stop - start, self.num_heads, self.head_dim)
            k_chunk = k_chunk.view(chunk_shape)
            v_chunk = v_chunk.view(chunk_shape)
            key_laid_out = False
            if split_qkv:
                if frequencies is None or not frequencies.shape[-1]:
                    k_chunk = self.k_norm(k_chunk)
                else:
                    chunk_frequencies = (
                        frequencies[:, start:stop]
                        if frequencies.ndim == 6
                        else frequencies[start:stop]
                    )
                    if fused_qknorm_hnd_layout:
                        from .single_qknorm_rope import (
                            try_apply_single_qknorm_rope_to_hnd,
                        )

                        key_destination = key_hnd[
                            0, :, start:stop
                        ].permute(1, 0, 2)
                        key_laid_out = try_apply_single_qknorm_rope_to_hnd(
                            k_chunk,
                            key_destination,
                            weight=self.k_norm.weight.to(
                                device=k_chunk.device,
                                dtype=k_chunk.dtype,
                            ),
                            frequencies=chunk_frequencies,
                            eps=self.k_norm.eps,
                        )
                        del key_destination
                    if not key_laid_out:
                        k_chunk = _apply_single_qknorm_rope(
                            k_chunk,
                            weight=self.k_norm.weight,
                            frequencies=chunk_frequencies,
                            eps=self.k_norm.eps,
                            query_slot=False,
                        )
            else:
                q_chunk = q_chunk.view(chunk_shape)
                if frequencies is None or not frequencies.shape[-1]:
                    q_chunk = self.q_norm(q_chunk)
                    k_chunk = self.k_norm(k_chunk)
                else:
                    chunk_frequencies = (
                        frequencies[:, start:stop]
                        if frequencies.ndim == 6
                        else frequencies[start:stop]
                    )
                    q_chunk, k_chunk = apply_qknorm_rope(
                        q_chunk,
                        k_chunk,
                        q_weight=self.q_norm.weight,
                        k_weight=self.k_norm.weight,
                        frequencies=chunk_frequencies,
                        eps=self.q_norm.eps,
                    )
            if kv_layout == "HND":
                if not key_laid_out:
                    key_hnd[0, :, start:stop].copy_(
                        k_chunk.permute(1, 0, 2)
                    )
                value_hnd[0, :, start:stop].copy_(v_chunk.permute(1, 0, 2))
            else:
                key_hnd[0, start:stop].copy_(k_chunk)
                value_hnd[0, start:stop].copy_(v_chunk)
            if split_qkv:
                del kv_chunk, k_chunk, v_chunk
            else:
                del qkv, q_chunk, k_chunk, v_chunk

        if not compact_kv:
            with _profile_region("h3_long_prepare_value"):
                if physical_contract.direct_hnd_fp8_value:
                    from .compact_fp8 import prepare_sage_fp8_hnd_direct

                    heads = int(value_hnd.shape[1])
                    key_tokens = int(value_hnd.shape[2])
                    head_dim = int(value_hnd.shape[3])
                    value_fp8, value_scale = prepare_sage_fp8_hnd_direct(
                        value_hnd
                    )
                else:
                    value_fp8, value_scale, heads, key_tokens, head_dim = (
                        prepare_values(value_hnd)
                    )
            del value_hnd
            if (
                heads != self.num_heads
                or key_tokens != sequence
                or head_dim != self.head_dim
            ):
                raise ValueError(
                    "prepared long-sequence V metadata is inconsistent"
                )
            with _profile_region("h3_long_prepare_key"):
                prepared = prepare_keys(key_hnd, value_fp8, value_scale)
            del key_hnd

        def project_queries(start: int, stop: int) -> torch.Tensor:
            """Recompute only one live Query range with exact row-local math."""

            from .kernels import current_long_sequence_fused_query_projection

            if split_qkv and current_long_sequence_fused_query_projection():
                with _profile_region("h3_long_query_qkv_projection"):
                    if prepared_qkv_input is None:
                        q_chunk = output_slice(
                            hidden[start:stop], 0, self.inner_width
                        )
                    else:
                        q_chunk = prepared_output_slice(
                            hidden,
                            prepared_qkv_input,
                            start,
                            stop,
                            0,
                            self.inner_width,
                        )
                chunk_shape = (
                    stop - start,
                    self.num_heads,
                    self.head_dim,
                )
                q_chunk = q_chunk.view(chunk_shape)
                if frequencies is None or not frequencies.shape[-1]:
                    q_chunk = self.q_norm(q_chunk)
                else:
                    chunk_frequencies = (
                        frequencies[:, start:stop]
                        if frequencies.ndim == 6
                        else frequencies[start:stop]
                    )
                    if fused_qknorm_hnd_layout:
                        from .single_qknorm_rope import (
                            try_apply_single_qknorm_rope_to_hnd,
                        )

                        query_hnd = torch.empty(
                            (
                                self.num_heads,
                                stop - start,
                                self.head_dim,
                            ),
                            device=q_chunk.device,
                            dtype=q_chunk.dtype,
                        )
                        query_nhd_view = query_hnd.permute(1, 0, 2)
                        if try_apply_single_qknorm_rope_to_hnd(
                            q_chunk,
                            query_nhd_view,
                            weight=self.q_norm.weight.to(
                                device=q_chunk.device,
                                dtype=q_chunk.dtype,
                            ),
                            frequencies=chunk_frequencies,
                            eps=self.q_norm.eps,
                        ):
                            del q_chunk, query_hnd
                            return query_nhd_view
                        del query_nhd_view, query_hnd
                    q_chunk = _apply_single_qknorm_rope(
                        q_chunk,
                        weight=self.q_norm.weight,
                        frequencies=chunk_frequencies,
                        eps=self.q_norm.eps,
                        query_slot=True,
                    )
                return q_chunk

            query = torch.empty(
                (stop - start, self.num_heads, self.head_dim),
                device=hidden.device,
                dtype=hidden.dtype,
            )
            for chunk_start in range(start, stop, projection_chunk_tokens):
                chunk_stop = min(stop, chunk_start + projection_chunk_tokens)
                with _profile_region("h3_long_query_qkv_projection"):
                    if split_qkv:
                        if prepared_qkv_input is None:
                            q_chunk = output_slice(
                                hidden[chunk_start:chunk_stop],
                                0,
                                self.inner_width,
                            )
                        else:
                            q_chunk = prepared_output_slice(
                                hidden,
                                prepared_qkv_input,
                                chunk_start,
                                chunk_stop,
                                0,
                                self.inner_width,
                            )
                    else:
                        qkv = self.qkv_proj(hidden[chunk_start:chunk_stop])
                if not split_qkv:
                    q_chunk, k_chunk, v_chunk = qkv.split(
                        self.inner_width, dim=-1
                    )
                chunk_shape = (
                    chunk_stop - chunk_start,
                    self.num_heads,
                    self.head_dim,
                )
                q_chunk = q_chunk.view(chunk_shape)
                if split_qkv:
                    if frequencies is None or not frequencies.shape[-1]:
                        q_chunk = self.q_norm(q_chunk)
                    else:
                        chunk_frequencies = (
                            frequencies[:, chunk_start:chunk_stop]
                            if frequencies.ndim == 6
                            else frequencies[chunk_start:chunk_stop]
                        )
                        q_chunk = _apply_single_qknorm_rope(
                            q_chunk,
                            weight=self.q_norm.weight,
                            frequencies=chunk_frequencies,
                            eps=self.q_norm.eps,
                            query_slot=True,
                        )
                else:
                    k_chunk = k_chunk.view(chunk_shape)
                    if frequencies is None or not frequencies.shape[-1]:
                        q_chunk = self.q_norm(q_chunk)
                        k_chunk = self.k_norm(k_chunk)
                    else:
                        chunk_frequencies = (
                            frequencies[:, chunk_start:chunk_stop]
                            if frequencies.ndim == 6
                            else frequencies[chunk_start:chunk_stop]
                        )
                        q_chunk, k_chunk = apply_qknorm_rope(
                            q_chunk,
                            k_chunk,
                            q_weight=self.q_norm.weight,
                            k_weight=self.k_norm.weight,
                            frequencies=chunk_frequencies,
                            eps=self.q_norm.eps,
                        )
                local_start = chunk_start - start
                local_stop = chunk_stop - start
                query[local_start:local_stop].copy_(q_chunk)
                if split_qkv:
                    del q_chunk
                else:
                    del qkv, q_chunk, k_chunk, v_chunk
            return query

        def commit(start: int, stop: int, attended: torch.Tensor) -> None:
            flattened = attended.reshape(stop - start, self.inner_width)
            with _profile_region("h3_long_out_projection"):
                projected = self.out_proj(flattened)
            with _profile_region("h3_long_attention_residual"):
                _gated_residual_slice_(
                    residual,
                    projected,
                    gate,
                    modulation_segments,
                    start=start,
                    stop=stop,
                )
            del flattened, projected

        all_queries = getattr(physical, "long_sequence_all_queries", None)
        if all_queries is not None:
            for start in range(0, sequence, query_chunk_tokens):
                stop = min(sequence, start + query_chunk_tokens)
                query = project_queries(start, stop)
                attended = all_queries(query, prepared)
                commit(start, stop, attended)
                del query, attended
            return True

        # Text length and the number/shape of reference image/audio items are
        # request dependent.  Stream prefix Queries as well, while retaining
        # every packed token as K/V, so no particular T2VA prompt shape is
        # baked into this path.
        for prefix_start in range(0, protected_tokens, query_chunk_tokens):
            prefix_stop = min(
                protected_tokens, prefix_start + query_chunk_tokens
            )
            query = project_queries(prefix_start, prefix_stop)
            with _profile_region("h3_long_prefix_attention"):
                prefix_output = prefix_attention(
                    query, prepared
                )
            commit(prefix_start, prefix_stop, prefix_output)
            del query, prefix_output
        video_tokens = sequence - protected_tokens
        for local_start in range(0, video_tokens, query_chunk_tokens):
            local_stop = min(video_tokens, local_start + query_chunk_tokens)
            global_start = protected_tokens + local_start
            global_stop = protected_tokens + local_stop
            indices = torch.arange(
                local_start,
                local_stop,
                device=hidden.device,
                dtype=torch.int64,
            )
            query = project_queries(global_start, global_stop)
            with _profile_region("h3_long_video_attention"):
                attended = video_attention(
                    query,
                    prepared,
                    protected_tokens=protected_tokens,
                    query_token_indices=indices,
                )
            commit(global_start, global_stop, attended)
            del query, indices, attended
        return True


class SwiGLUMLP(nn.Module):
    def __init__(self, fc1: nn.Module, fc2: nn.Module) -> None:
        super().__init__()
        self.fc1, self.fc2 = fc1, fc2

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        gate, up = self.fc1(value).chunk(2, dim=-1)
        return self.fc2(F.silu(gate).mul_(up))


def split_adaln(output: torch.Tensor, *, hidden_size: int) -> tuple[torch.Tensor, ...]:
    if output.shape[-1] != 18 * hidden_size:
        raise ValueError("block AdaLN projection must produce 18 * hidden_size")
    return output.view(-1, 6 * hidden_size).chunk(6, dim=-1)


def _scale_shift(
    value: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
    segments: tuple[ModulationSegment, ...],
) -> torch.Tensor:
    """Apply AdaLN modulation without expanding rows to every packed token.

    H3 has only a handful of contiguous modality/timestep runs.  The former
    implementation used ``index_select`` to create full [tokens, hidden]
    shift and scale tensors twice per block.  At 480p this moved hundreds of
    MiB for each call.  Segment broadcasts are the operation order used by the
    accepted Comfy implementation and preserve the BF16 materialization
    boundaries (multiply, then add).
    """

    previous = 0
    for start, stop, row in segments:
        if start != previous or stop <= start:
            raise ValueError("modulation segments must be an ordered partition")
        target = value[start:stop]
        target.mul_(1.0 + scale[row].to(value.dtype)).add_(shift[row].to(value.dtype))
        previous = stop
    if previous != value.shape[0]:
        raise ValueError("modulation segments do not cover the packed sequence")
    return value


def _gated_residual(
    residual: torch.Tensor,
    update: torch.Tensor,
    gate: torch.Tensor,
    segments: tuple[ModulationSegment, ...],
) -> torch.Tensor:
    previous = 0
    for start, stop, row in segments:
        if start != previous or stop <= start:
            raise ValueError("modulation segments must be an ordered partition")
        residual[start:stop].addcmul_(
            update[start:stop], gate[row].to(update.dtype)
        )
        previous = stop
    if previous != residual.shape[0]:
        raise ValueError("modulation segments do not cover the packed sequence")
    return residual


def _gated_residual_slice_(
    residual: torch.Tensor,
    update: torch.Tensor,
    gate: torch.Tensor,
    segments: tuple[ModulationSegment, ...],
    *,
    start: int,
    stop: int,
) -> None:
    """Apply one contiguous update slice without expanding AdaLN rows."""

    if not 0 <= start < stop <= residual.shape[0]:
        raise ValueError("gated residual slice falls outside the packed sequence")
    if update.shape != residual[start:stop].shape:
        raise ValueError("gated residual slice shape does not match its destination")
    covered = start
    for segment_start, segment_stop, row in segments:
        overlap_start = max(start, segment_start)
        overlap_stop = min(stop, segment_stop)
        if overlap_start >= overlap_stop:
            continue
        if overlap_start != covered:
            raise ValueError("modulation segments do not cover the update slice")
        update_start = overlap_start - start
        update_stop = overlap_stop - start
        residual[overlap_start:overlap_stop].addcmul_(
            update[update_start:update_stop], gate[row].to(update.dtype)
        )
        covered = overlap_stop
    if covered != stop:
        raise ValueError("modulation segments do not cover the update slice")


class H3TransformerBlock(nn.Module):
    """One full H3 block; AdaLN projector accepts unique timestep rows."""

    def __init__(
        self,
        attention: FusedQKVAttention,
        mlp: SwiGLUMLP,
        adaln_projector: nn.Module,
        *,
        hidden_size: int,
        norm_eps: float = 1e-5,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.attention = attention
        self.mlp = mlp
        self.adaln_projector = adaln_projector
        self.hidden_size = hidden_size
        self.norm1 = RMSNorm(hidden_size, norm_eps, device=device, dtype=dtype)
        self.norm2 = RMSNorm(hidden_size, norm_eps, device=device, dtype=dtype)

    def forward(
        self,
        value: torch.Tensor,
        *,
        timestep_rows: torch.Tensor,
        modulation_segments: tuple[ModulationSegment, ...],
        frequencies: torch.Tensor,
        mlp_chunk_tokens: int | None = None,
    ) -> torch.Tensor:
        from .frame_interleave import current_frame_interleave_layer
        from .kernels import current_attention_layer
        from .spatial_query_lattice import current_spatial_query_lattice_layer

        layer = current_attention_layer()
        lattice_plan = current_spatial_query_lattice_layer(layer)
        if lattice_plan is not None:
            from .modality_refresh import refresh_selected_video_tiles

            # The final generated-video run is the only suffix after the
            # protected text/condition/audio prefix.
            protected_tokens = modulation_segments[-1][0]
            active_before = value.index_select(
                0, lattice_plan.active_video_indices
            )
            value = refresh_selected_video_tiles(
                self,
                value,
                protected_tokens=protected_tokens,
                active_video_indices=lattice_plan.active_video_indices,
                timestep_rows=timestep_rows,
                modulation_segments=modulation_segments,
                frequencies=frequencies,
            )
            return lattice_plan.reconstruct_inactive_(value, active_before)

        frame_plan = current_frame_interleave_layer(layer)
        if frame_plan is not None:
            selected_input = value.index_select(0, frame_plan.selected_indices)
            selected_output = self._forward_complete(
                selected_input.clone(),
                timestep_rows=timestep_rows,
                modulation_segments=frame_plan.selected_modulation_segments(
                    modulation_segments
                ),
                frequencies=frame_plan.selected_frequencies(frequencies),
                mlp_chunk_tokens=mlp_chunk_tokens,
            )
            return frame_plan.reconstruct_(value, selected_input, selected_output)
        return self._forward_complete(
            value,
            timestep_rows=timestep_rows,
            modulation_segments=modulation_segments,
            frequencies=frequencies,
            mlp_chunk_tokens=mlp_chunk_tokens,
        )

    def _forward_complete(
        self,
        value: torch.Tensor,
        *,
        timestep_rows: torch.Tensor,
        modulation_segments: tuple[ModulationSegment, ...],
        frequencies: torch.Tensor,
        mlp_chunk_tokens: int | None,
    ) -> torch.Tensor:
        params = split_adaln(
            self.adaln_projector(timestep_rows), hidden_size=self.hidden_size
        )
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = params
        from .kernels import (
            current_attention_layer,
            current_long_sequence_projection_chunk_tokens,
            current_long_sequence_query_chunk_tokens,
            rms_adaln,
        )

        hidden = rms_adaln(
            value, self.norm1, shift_a, scale_a, modulation_segments
        )
        query_chunk_tokens = current_long_sequence_query_chunk_tokens()
        if query_chunk_tokens is None:
            value = _gated_residual(
                value,
                self.attention(hidden, frequencies),
                gate_a,
                modulation_segments,
            )
        elif not self.attention.stream_gated_residual_(
            value,
            hidden,
            frequencies,
            gate_a,
            modulation_segments,
            query_chunk_tokens=query_chunk_tokens,
            projection_chunk_tokens=(
                current_long_sequence_projection_chunk_tokens()
            ),
        ):
            from .kernels import _can_fallback_to_unstreamed_exact_attention

            if not _can_fallback_to_unstreamed_exact_attention(
                self.attention.backend, int(hidden.shape[0])
            ):
                raise RuntimeError(
                    "the selected Attention action has no validated long-sequence "
                    "streaming implementation"
                )
            value = _gated_residual(
                value,
                self.attention(hidden, frequencies),
                gate_a,
                modulation_segments,
            )
        # The norm1 output is dead once the Attention residual has been
        # accumulated into ``value``.  Keeping this long-sequence tensor alive
        # while Python evaluates the norm2 call overlaps two full hidden-state
        # allocations (about 1 GiB at 720p/15s).  That unnecessary overlap can
        # breach the intentional 7.25-GiB allocator ceiling even though the
        # Attention and MLP phases each fit independently.
        del hidden
        hidden = rms_adaln(
            value, self.norm2, shift_m, scale_m, modulation_segments
        )
        from .fused_mlp import try_fused_gated_mlp
        from .research_capture import (
            capture_kind,
            capture_target,
            persist_mlp_capture,
            record_video_mlp_gate,
        )

        record_video_mlp_gate(
            gate_m,
            row=modulation_segments[-1][2],
            layer=current_attention_layer(),
        )

        from .kernels import current_attention_step
        from .mlp_gate_budget import skip_video_mlp

        step_context = current_attention_step()
        video_modulation_row = modulation_segments[-1][2]
        if skip_video_mlp(
            layer=current_attention_layer(),
            step=None if step_context is None else step_context[0],
            step_count=None if step_context is None else step_context[1],
            gate=gate_m,
            row=video_modulation_row,
        ):
            # The generated-video segment is the final packed run.  Preserve
            # every conditioning/audio row exactly; only its local video MLP
            # residual becomes identity at the gate-selected coordinates.
            protected = modulation_segments[-1][0]
            prefix_segments = tuple(
                segment for segment in modulation_segments if segment[1] <= protected
            )
            if not prefix_segments or prefix_segments[-1][1] != protected:
                raise ValueError("video MLP budget requires a complete protected prefix")
            prefix_value = value[:protected]
            prefix_hidden = hidden[:protected]
            if not try_fused_gated_mlp(
                prefix_value,
                prefix_hidden,
                self.mlp,
                gate_m,
                prefix_segments,
                chunk_tokens=mlp_chunk_tokens,
            ):
                _gated_residual(
                    prefix_value,
                    self.mlp(prefix_hidden),
                    gate_m,
                    prefix_segments,
                )
            return value

        capture = capture_target(current_attention_layer())
        capture_protected = modulation_segments[-1][0]
        capture_delta = capture is not None and capture_kind() == "delta"
        capture_before = value[capture_protected:].clone() if capture_delta else None

        # Preserve the complete attention/motion path while routing only the
        # row-local MLP.  Prefix modalities remain exact, and omitted video
        # residuals are reconstructed strictly inside one frame/spatial row.
        from .mlp_spatial_lattice import current_mlp_spatial_lattice_layer

        mlp_plan = current_mlp_spatial_lattice_layer(
            current_attention_layer()
        )
        if mlp_plan is not None:
            protected = mlp_plan.protected_tokens
            from .mlp_spatial_lattice import current_mlp_spatial_lattice_config

            mlp_config = current_mlp_spatial_lattice_config()
            detail_fraction = 0.0 if mlp_config is None else mlp_config.detail_fraction
            detail_positions, detail_indices = mlp_plan.select_detail_positions(
                hidden, detail_fraction
            )
            prefix_indices = torch.arange(
                protected, dtype=torch.long, device=value.device
            )
            selected_indices = torch.cat(
                (prefix_indices, mlp_plan.active_video_indices, detail_indices), dim=0
            )
            selected_hidden = hidden.index_select(0, selected_indices)
            selected_value = value.index_select(0, selected_indices)
            selected_before = selected_value.clone()
            prefix_segments = tuple(
                segment for segment in modulation_segments if segment[1] <= protected
            )
            video_segments = [
                segment for segment in modulation_segments if segment[0] == protected
            ]
            if len(video_segments) != 1:
                raise ValueError("MLP spatial lattice requires one generated-video segment")
            video_row = video_segments[0][2]
            selected_segments = prefix_segments + (
                (protected, selected_value.shape[0], video_row),
            )
            if not try_fused_gated_mlp(
                selected_value,
                selected_hidden,
                self.mlp,
                gate_m,
                selected_segments,
                chunk_tokens=mlp_chunk_tokens,
            ):
                selected_value = _gated_residual(
                    selected_value,
                    self.mlp(selected_hidden),
                    gate_m,
                    selected_segments,
                )
            value[:protected].copy_(selected_value[:protected])
            selected_delta = selected_value[protected:].sub_(selected_before[protected:])
            active_count = int(mlp_plan.active_video_indices.numel())
            mlp_plan.reconstruct_(
                value,
                selected_delta[:active_count],
                detail_positions=detail_positions,
                detail_delta=selected_delta[active_count:],
            )
            return value

        with _profile_region("h3_long_mlp"):
            fused_mlp = try_fused_gated_mlp(
                value,
                hidden,
                self.mlp,
                gate_m,
                modulation_segments,
                chunk_tokens=mlp_chunk_tokens,
            )
        if fused_mlp:
            if capture is not None:
                persist_mlp_capture(
                    capture,
                    hidden_video=hidden[capture_protected:],
                    delta_video=(
                        value[capture_protected:] - capture_before
                        if capture_before is not None
                        else None
                    ),
                    protected_tokens=capture_protected,
                )
            return value
        value = _gated_residual(
            value, self.mlp(hidden), gate_m, modulation_segments
        )
        if capture is not None:
            persist_mlp_capture(
                capture,
                hidden_video=hidden[capture_protected:],
                delta_video=(
                    value[capture_protected:] - capture_before
                    if capture_before is not None
                    else None
                ),
                protected_tokens=capture_protected,
            )
        return value


def time_shift_sigma(sigma: torch.Tensor, from_shift: float, to_shift: float) -> torch.Tensor:
    base = sigma / (from_shift + sigma * (1.0 - from_shift))
    return to_shift * base / (1.0 + (to_shift - 1.0) * base)


def time_shift_slope(sigma: torch.Tensor, from_shift: float, to_shift: float) -> torch.Tensor:
    """Derivative of the target shifted schedule with respect to the source."""

    base = sigma / (from_shift + sigma * (1.0 - from_shift))
    return (
        to_shift
        * (1.0 + (from_shift - 1.0) * base).square()
        / (from_shift * (1.0 + (to_shift - 1.0) * base).square())
    )


def timestep_embedding(timestep: torch.Tensor, width: int = 256) -> torch.Tensor:
    half = width // 2
    frequencies = torch.exp(
        -math.log(10000.0)
        * torch.arange(half, dtype=torch.float32, device=timestep.device)
        / half
    )
    angles = timestep.float().reshape(-1, 1) * frequencies.reshape(1, -1)
    return torch.cat((torch.cos(angles), torch.sin(angles)), dim=-1)
