"""Learned spatial resizing for MiniMax H3 video latents.

The network definition and H3 normalization constants are adapted from
``LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler`` (Apache-2.0), revision
``6a4b191e8af583b7c097f564690325f91d18c2e2``.  The service integration is
native PyTorch and intentionally contains no ComfyUI dependency.

Unlike image interpolation, this model learns the correlations between H3's
24 latent channels.  Direct Bicubic resizing is not a valid second-sampling
initialization and is deliberately not exposed by this module.
"""

from __future__ import annotations

import re
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


LATENTS_MEAN = (
    0.858090341091156, -0.9606591463088989, 1.0661640167236328,
    -0.5090325474739075, -0.2727581858634949, -1.3675414323806763,
    -0.2553254961967468, -0.26907554268836975, -0.5376840829849243,
    -0.0464097298681736, 0.6657370328903198, 0.19690127670764923,
    -0.5460608005523682, -0.4035342037677765, -0.23683024942874908,
    0.25928452610969543, -0.30133944749832153, 0.211341992020607,
    -1.1206848621368408, 0.3581933379173279, -0.04225143790245056,
    0.2604829967021942, 0.22864092886447906, 0.7056031823158264,
)
LATENTS_STD = (
    1.2223774194717407, 1.2767263650894165, 1.6831774711608887,
    1.7549455165863037, 1.5636216402053833, 2.194143533706665,
    0.9653137922286987, 1.0569885969161987, 0.841948926448822,
    0.7729952931404114, 1.8955937623977661, 0.946841835975647,
    0.7996809482574463, 0.44988900423049927, 0.7197399735450745,
    0.6936293244361877, 2.961095094680786, 2.7694199085235596,
    3.0496184825897217, 2.1088054180145264, 3.276226282119751,
    3.1627357006073, 2.2816812992095947, 2.6127843856811523,
)


def _normalization(channels: int) -> nn.GroupNorm:
    return nn.GroupNorm(32, channels)


def _conv3d_with_static_time(
    layer: nn.Conv3d,
    value: torch.Tensor,
    *,
    static_time: bool,
) -> torch.Tensor:
    """Apply a 3D convolution with an infinite-static temporal boundary.

    The released resizer mixes neighbouring H3 latent times.  For the
    framewise research route each time slice is an independent batch item and
    has length one.  Replicate-padding only the temporal axis makes every 3D
    kernel see that slice as a static sequence, while retaining its learned
    spatial/channel mapping exactly.
    """

    if not static_time:
        return layer(value)
    if int(value.shape[2]) != 1:
        raise ValueError("static-time convolution requires one latent time slice")
    pad_t, pad_h, pad_w = layer.padding
    if pad_t:
        value = F.pad(
            value,
            (0, 0, 0, 0, pad_t, pad_t),
            mode="replicate",
        )
    return F.conv3d(
        value,
        layer.weight,
        layer.bias,
        stride=layer.stride,
        padding=(0, pad_h, pad_w),
        dilation=layer.dilation,
        groups=layer.groups,
    )


class _ResBlockEmb3D(nn.Module):
    def __init__(self, channels: int, embedding_channels: int, dropout: float) -> None:
        super().__init__()
        self.in_layers = nn.Sequential(
            _normalization(channels),
            nn.SiLU(),
            nn.Conv3d(channels, channels, 3, padding=1),
        )
        self.emb_layers = nn.Sequential(
            nn.SiLU(), nn.Linear(embedding_channels, 2 * channels)
        )
        self.out_norm = _normalization(channels)
        self.out_layers = nn.Sequential(
            nn.SiLU(),
            nn.Dropout(p=dropout),
            nn.Conv3d(channels, channels, 3, padding=1),
        )
        # This layer is zero-initialized by the upstream constructor, but the
        # checkpoint immediately replaces it.  Keeping the initialization
        # makes architecture parity explicit for strict loading.
        nn.init.zeros_(self.out_layers[-1].weight)
        nn.init.zeros_(self.out_layers[-1].bias)
        self.skip = nn.Identity()

    def forward(
        self,
        value: torch.Tensor,
        embedding: torch.Tensor,
        *,
        static_time: bool = False,
    ) -> torch.Tensor:
        hidden = self.in_layers[1](self.in_layers[0](value))
        hidden = _conv3d_with_static_time(
            self.in_layers[2], hidden, static_time=static_time
        )
        projected = self.emb_layers(embedding).to(hidden.dtype)
        while projected.ndim < hidden.ndim:
            projected = projected[..., None]
        scale, shift = projected.chunk(2, dim=1)
        hidden = self.out_norm(hidden) * (1 + scale) + shift
        hidden = self.out_layers[1](self.out_layers[0](hidden))
        hidden = _conv3d_with_static_time(
            self.out_layers[2], hidden, static_time=static_time
        )
        return self.skip(value) + hidden


class _TemporalConv(nn.Module):
    def __init__(self, channels: int, kernel_size: int) -> None:
        super().__init__()
        self.norm = _normalization(channels)
        self.dwconv = nn.Conv3d(
            channels,
            channels,
            kernel_size=(kernel_size, 1, 1),
            padding=(kernel_size // 2, 0, 0),
            groups=channels,
        )
        self.pwconv = nn.Conv3d(channels, channels, kernel_size=1)

    def forward(
        self, value: torch.Tensor, *, static_time: bool = False
    ) -> torch.Tensor:
        hidden = F.silu(self.norm(value))
        hidden = _conv3d_with_static_time(
            self.dwconv, hidden, static_time=static_time
        )
        hidden = _conv3d_with_static_time(
            self.pwconv, hidden, static_time=static_time
        )
        return value + hidden


class H3LatentResizer3D(nn.Module):
    """Exact inference graph for the released pure-3D BF16 checkpoint."""

    def __init__(
        self,
        *,
        in_channels: int = 24,
        in_blocks: int = 12,
        out_blocks: int = 12,
        channels: int = 512,
        dropout: float = 0.1,
        temporal_every: int = 2,
        temporal_kernel: int = 5,
    ) -> None:
        super().__init__()
        self.temporal_kernel = int(temporal_kernel)
        self.conv_in = nn.Conv3d(in_channels, channels, 3, padding=1)
        embedding_channels = 64
        self.embed = nn.Sequential(
            nn.Linear(1, embedding_channels),
            nn.SiLU(),
            nn.Linear(embedding_channels, embedding_channels),
        )
        self.in_blocks = self._make_blocks(
            in_blocks, channels, embedding_channels, dropout, temporal_every
        )
        self.out_blocks = self._make_blocks(
            out_blocks, channels, embedding_channels, dropout, temporal_every
        )
        self.norm_out = _normalization(channels)
        self.conv_out = nn.Conv3d(channels, in_channels, 3, padding=1)

    def _make_blocks(
        self,
        count: int,
        channels: int,
        embedding_channels: int,
        dropout: float,
        temporal_every: int,
    ) -> nn.ModuleList:
        blocks: list[nn.Module] = []
        for index in range(count):
            blocks.append(_ResBlockEmb3D(channels, embedding_channels, dropout))
            if temporal_every > 0 and index % temporal_every == 0:
                blocks.append(_TemporalConv(channels, self.temporal_kernel))
        return nn.ModuleList(blocks)

    @staticmethod
    def _run_blocks(
        blocks: nn.ModuleList,
        value: torch.Tensor,
        embedding: torch.Tensor,
        *,
        static_time: bool = False,
    ) -> torch.Tensor:
        for block in blocks:
            if isinstance(block, _ResBlockEmb3D):
                value = block(
                    value,
                    embedding.expand(value.shape[0], -1),
                    static_time=static_time,
                )
            else:
                value = block(value, static_time=static_time)
        return value

    def _forward_segment(
        self,
        value: torch.Tensor,
        *,
        effective_scale: float,
        target_size: tuple[int, int, int],
        static_time: bool = False,
    ) -> torch.Tensor:
        embedding = self.embed(
            torch.tensor(
                [[effective_scale - 1.0]],
                device=value.device,
                dtype=value.dtype,
            )
        )
        hidden = _conv3d_with_static_time(
            self.conv_in, value, static_time=static_time
        )
        hidden = self._run_blocks(
            self.in_blocks, hidden, embedding, static_time=static_time
        )
        hidden = F.interpolate(
            hidden, size=target_size, mode="trilinear", align_corners=False
        )
        hidden = self._run_blocks(
            self.out_blocks, hidden, embedding, static_time=static_time
        )
        return _conv3d_with_static_time(
            self.conv_out,
            F.silu(self.norm_out(hidden)),
            static_time=static_time,
        )

    def forward(
        self,
        value: torch.Tensor,
        *,
        effective_scale: float,
        target_size: tuple[int, int, int],
        temporal_chunk_frames: int = 24,
    ) -> torch.Tensor:
        if target_size == tuple(value.shape[-3:]):
            return value
        temporal = int(value.shape[2])
        chunk = int(temporal_chunk_frames)
        if chunk <= 0 or temporal <= chunk:
            return self._forward_segment(
                value,
                effective_scale=effective_scale,
                target_size=target_size,
            )

        overlap = self.temporal_kernel
        padded = F.pad(value, (0, 0, 0, 0, overlap, overlap), mode="replicate")
        output = torch.zeros(
            value.shape[0], value.shape[1], temporal, target_size[-2], target_size[-1],
            device=value.device, dtype=value.dtype,
        )
        weights = torch.zeros(
            1, 1, temporal, 1, 1, device=value.device, dtype=value.dtype
        )
        for start in range(0, temporal, chunk):
            end = min(temporal, start + chunk)
            output_start = max(0, start - overlap)
            output_end = min(temporal, end + overlap)
            low = max(0, output_start - overlap)
            high = min(temporal + 2 * overlap, output_end + overlap)
            segment = padded[:, :, low:high]
            segment_output = self._forward_segment(
                segment,
                effective_scale=effective_scale,
                target_size=(high - low, target_size[-2], target_size[-1]),
            )
            source_start = output_start + overlap - low
            valid = segment_output[
                :, :, source_start : source_start + output_end - output_start
            ]
            weight = torch.ones(
                output_end - output_start,
                device=value.device,
                dtype=value.dtype,
            )
            if start > output_start:
                length = start - output_start
                weight[:length] = torch.arange(
                    1, length + 1, device=value.device, dtype=value.dtype
                ) / (length + 1)
            if output_end > end:
                length = output_end - end
                weight[-length:] = torch.arange(
                    length, 0, -1, device=value.device, dtype=value.dtype
                ) / (length + 1)
            shaped = weight.view(1, 1, -1, 1, 1)
            output[:, :, output_start:output_end] += valid * shaped
            weights[:, :, output_start:output_end] += shaped
        return output / weights.clamp_min(1e-8)

    def forward_framewise_static(
        self,
        value: torch.Tensor,
        *,
        effective_scale: float,
        target_size: tuple[int, int, int],
    ) -> torch.Tensor:
        """Resize each latent time independently with static temporal kernels."""

        if int(target_size[0]) != int(value.shape[2]):
            raise ValueError("framewise-static resizing preserves latent duration")
        batch, channels, latent_times, height, width = value.shape
        frame_batch = value.permute(0, 2, 1, 3, 4).reshape(
            batch * latent_times, channels, 1, height, width
        )
        output = self._forward_segment(
            frame_batch,
            effective_scale=effective_scale,
            target_size=(1, target_size[1], target_size[2]),
            static_time=True,
        )
        return output.reshape(
            batch, latent_times, channels, 1, target_size[1], target_size[2]
        )[:, :, :, 0].permute(0, 2, 1, 3, 4).contiguous()


def _detect_architecture(state: dict[str, torch.Tensor]) -> dict[str, int]:
    try:
        conv_in = state["conv_in.weight"]
    except KeyError as error:
        raise ValueError("latent upscaler checkpoint is missing conv_in.weight") from error
    in_residuals: set[int] = set()
    out_residuals: set[int] = set()
    temporal_indices: list[int] = []
    temporal_kernel = 0
    for key, tensor in state.items():
        match = re.match(r"in_blocks\.(\d+)\.in_layers\.", key)
        if match:
            in_residuals.add(int(match.group(1)))
        match = re.match(r"out_blocks\.(\d+)\.in_layers\.", key)
        if match:
            out_residuals.add(int(match.group(1)))
        match = re.match(r"in_blocks\.(\d+)\.dwconv\.weight", key)
        if match:
            temporal_indices.append(int(match.group(1)))
            temporal_kernel = int(tensor.shape[2])
    if not in_residuals or not out_residuals or not temporal_indices:
        raise ValueError("unsupported latent upscaler architecture")
    # Released layouts alternate one residual and one temporal block every
    # ``temporal_every`` residuals.  Infer the interval from residual indexes.
    temporal_every = 2
    return {
        "in_channels": int(conv_in.shape[1]),
        "channels": int(conv_in.shape[0]),
        "in_blocks": len(in_residuals),
        "out_blocks": len(out_residuals),
        "temporal_every": temporal_every,
        "temporal_kernel": temporal_kernel,
    }


def load_h3_latent_upscaler(checkpoint: Path) -> H3LatentResizer3D:
    """Load and strictly validate the released H3 latent upscaler."""

    from safetensors.torch import load_file

    checkpoint = Path(checkpoint).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"H3 latent upscaler is missing: {checkpoint}")
    state = load_file(str(checkpoint), device="cpu")
    if any(key.startswith("upscaler.") for key in state):
        state = {
            key.removeprefix("upscaler."): value
            for key, value in state.items()
            if key.startswith("upscaler.")
        }
    model = H3LatentResizer3D(**_detect_architecture(state))
    model.load_state_dict(state, strict=True, assign=True)
    return model.to(dtype=torch.bfloat16).eval().requires_grad_(False)


def upscale_h3_video_latent(
    model: H3LatentResizer3D,
    latent: torch.Tensor,
    *,
    target_height: int,
    target_width: int,
    temporal_chunk_frames: int = 24,
    temporal_mode: str = "joint_3d",
) -> torch.Tensor:
    """Return a learned spatial upscale on CPU with source dtype preserved."""

    if latent.ndim != 5 or latent.shape[1] != 24:
        raise ValueError("H3 video latent must have shape B,24,T,H,W")
    if target_height < latent.shape[-2] or target_width < latent.shape[-1]:
        raise ValueError("learned H3 latent resizer only supports upscaling")
    if target_height == latent.shape[-2] and target_width == latent.shape[-1]:
        return latent
    source_dtype = latent.dtype
    device = next(model.parameters()).device
    compute_dtype = next(model.parameters()).dtype
    value = latent.to(device=device, dtype=compute_dtype, non_blocking=False)
    mean = torch.tensor(
        LATENTS_MEAN, device=device, dtype=compute_dtype
    ).view(1, 24, 1, 1, 1)
    std = torch.tensor(
        LATENTS_STD, device=device, dtype=compute_dtype
    ).view(1, 24, 1, 1, 1)
    effective_scale = 0.5 * (
        target_height / latent.shape[-2] + target_width / latent.shape[-1]
    )
    with torch.inference_mode():
        normalized = (value - mean) / std
        if temporal_mode == "joint_3d":
            output = model(
                normalized,
                effective_scale=float(effective_scale),
                target_size=(latent.shape[2], target_height, target_width),
                temporal_chunk_frames=temporal_chunk_frames,
            )
        elif temporal_mode == "framewise_static":
            output = model.forward_framewise_static(
                normalized,
                effective_scale=float(effective_scale),
                target_size=(latent.shape[2], target_height, target_width),
            )
        else:
            raise ValueError(f"unsupported latent-upscaler temporal mode: {temporal_mode}")
        output = output * std + mean
    return output.to(device="cpu", dtype=source_dtype)


__all__ = [
    "H3LatentResizer3D",
    "load_h3_latent_upscaler",
    "upscale_h3_video_latent",
]
