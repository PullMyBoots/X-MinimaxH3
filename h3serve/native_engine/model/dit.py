"""ComfyUI-free MiniMax H3 pruned DiT forward for T2VA and FL2VA.

This module stops at video/audio velocity tensors. It intentionally does not
own tokenization, the Qwen text encoder, schedulers, VAEs, or media muxing.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .assembly import H3BlockStack
from .config import H3CoreConfig
from .layers import (
    FusedQKVAttention,
    ModulationSegment,
    RMSNorm,
    SwiGLUMLP,
    rope_frequencies,
    rope_rotation_table,
    time_shift_sigma,
    time_shift_slope,
)
from .lora import AdaLNCurveRows, PrunedCurveAdaLN
from .kernels import attention_protected_prefix, attention_video_layout
from .frame_interleave import (
    FrameInterleavePlan,
    current_frame_interleave_config,
    frame_interleave_plan,
)
from .spatial_query_lattice import (
    SpatialQueryLatticePlan,
    current_spatial_query_lattice_config,
    spatial_query_lattice_plan,
)
from .mlp_spatial_lattice import (
    MLPSpatialLatticePlan,
    current_mlp_spatial_lattice_config,
    mlp_spatial_lattice_plan,
)
from .packed import (
    PackedLayout,
    build_fl2va_layout,
    build_hybrid_condition_layout,
    build_ref2va_layout,
    pack_audio,
    patchify_video,
    unpack_audio,
    unpatchify_video,
)


@dataclass(slots=True)
class H3DiTOutput:
    """Velocity convention consumed by the H3 flow sampler."""

    video: torch.Tensor
    audio: torch.Tensor
    layout: PackedLayout


class H3TokenRefinerBlock(nn.Module):
    def __init__(
        self,
        attention: FusedQKVAttention,
        mlp: SwiGLUMLP,
        *,
        hidden_size: int,
        norm_eps: float,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.norm1 = RMSNorm(hidden_size, norm_eps, device=device, dtype=dtype)
        self.norm2 = RMSNorm(hidden_size, norm_eps, device=device, dtype=dtype)
        self.attention = attention
        self.mlp = mlp

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = value + self.attention(self.norm1(value), frequencies=None)
        return value + self.mlp(self.norm2(value))


class H3TokenRefiner(nn.Module):
    def __init__(
        self,
        blocks: list[H3TokenRefinerBlock],
        final_norm: RMSNorm,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(blocks)
        self.final_norm = final_norm

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            value = block(value)
        return self.final_norm(value)


class H3FinalLayer(nn.Module):
    def __init__(
        self,
        norm: RMSNorm,
        adaln_projector: PrunedCurveAdaLN,
        video_out: nn.Module,
        audio_out: nn.Module,
        *,
        hidden_size: int,
    ) -> None:
        super().__init__()
        self.norm = norm
        self.adaln_projector = adaln_projector
        self.video_out = video_out
        self.audio_out = audio_out
        self.hidden_size = hidden_size

    def _stream(
        self,
        value: torch.Tensor,
        segment: slice,
        timestep_row: int,
        shift: torch.Tensor,
        scale: torch.Tensor,
        projection: nn.Module,
        *,
        chunk_tokens: int | None = None,
    ) -> torch.Tensor:
        def project(rows: torch.Tensor) -> torch.Tensor:
            hidden = self.norm(rows)
            hidden = hidden * (1.0 + scale[timestep_row].to(hidden.dtype))
            hidden = hidden + shift[timestep_row].to(hidden.dtype)
            return projection(hidden.float())

        if chunk_tokens is None:
            # Preserve the established INT8/16GB and INT8/24GB path exactly.
            return project(value[segment])
        if chunk_tokens <= 0:
            raise ValueError("final projection chunk_tokens must be positive")
        start = 0 if segment.start is None else int(segment.start)
        stop = value.shape[0] if segment.stop is None else int(segment.stop)
        if stop - start <= chunk_tokens:
            return project(value[segment])
        # Normalization, affine modulation and the output linear are all
        # row-local.  Chunking before the BF16 -> FP32 conversion prevents a
        # full [tokens, hidden_size] FP32 temporary on the 8GB backend without
        # changing the sampler trajectory or dropping any computation.
        outputs = [
            project(value[offset : min(stop, offset + chunk_tokens)])
            for offset in range(start, stop, chunk_tokens)
        ]
        return torch.cat(outputs, dim=0)

    def forward(
        self,
        value: torch.Tensor,
        *,
        curve_rows: AdaLNCurveRows,
        video_segment: slice,
        audio_segment: slice,
        video_timestep_row: int,
        audio_timestep_row: int,
        final_projection_chunk_tokens: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        params = self.adaln_projector(curve_rows)
        if params.shape[-1] != 2 * self.hidden_size:
            raise ValueError("final AdaLN projection must produce 2 * hidden_size")
        shift, scale = params.chunk(2, dim=-1)
        return (
            self._stream(
                value,
                video_segment,
                video_timestep_row,
                shift,
                scale,
                self.video_out,
                chunk_tokens=final_projection_chunk_tokens,
            ),
            self._stream(
                value,
                audio_segment,
                audio_timestep_row,
                shift,
                scale,
                self.audio_out,
                chunk_tokens=final_projection_chunk_tokens,
            ),
        )


def _pad_to_patch(value: torch.Tensor, patch_size: Sequence[int]) -> torch.Tensor:
    pt, ph, pw = (int(item) for item in patch_size)
    time_pad = (-value.shape[-3]) % pt
    height_pad = (-value.shape[-2]) % ph
    width_pad = (-value.shape[-1]) % pw
    return F.pad(value, (0, width_pad, 0, height_pad, 0, time_pad))


class FullH3DiT(nn.Module):
    """The complete pruned FL2VA DiT graph, excluding surrounding pipeline."""

    def __init__(
        self,
        *,
        config: H3CoreConfig,
        video_patch_proj: nn.Module,
        audio_patch_proj: nn.Module,
        condition_proj: nn.Module,
        token_refiner: H3TokenRefiner,
        block_stack: H3BlockStack,
        final_layer: H3FinalLayer,
        rope_inv_freq: torch.Tensor,
        compute_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.config = config
        self.video_patch_proj = video_patch_proj
        self.audio_patch_proj = audio_patch_proj
        self.condition_proj = condition_proj
        self.token_refiner = token_refiner
        self.block_stack = block_stack
        self.final_layer = final_layer
        self.register_buffer("rope_inv_freq", rope_inv_freq)
        self.compute_dtype = compute_dtype

    @staticmethod
    def _request_local_tensor(
        layout: PackedLayout,
        attribute: str,
        *,
        enabled: bool,
        device: torch.device,
        builder: Callable[[], torch.Tensor | None],
    ) -> torch.Tensor | None:
        """Return one immutable per-request tensor, optionally caching it.

        ``PackedLayout`` is created for one sampling request and discarded
        after decoding.  Reference-condition rows depend on the references,
        augmentation and seed, but none of those change between denoise steps.
        The explicit switch exists only so the former v00 behavior remains
        reproducible during optimization audits.
        """

        cached = getattr(layout, attribute) if enabled else None
        if cached is None or cached.device != device:
            cached = builder()
            if enabled:
                setattr(layout, attribute, cached)
        return cached

    def _condition_rows(
        self,
        latents: Sequence[torch.Tensor],
        *,
        device: torch.device,
        augmentation: float,
        seed: int,
        expected_rows: int,
    ) -> torch.Tensor | None:
        if not latents:
            return None
        rows: list[torch.Tensor] = []
        for latent in latents:
            latent = _pad_to_patch(latent, self.config.patch_size)
            packed = patchify_video(latent.float(), self.config.patch_size)
            if augmentation < 1.0:
                generator = torch.Generator("cpu").manual_seed(seed)
                noise = torch.randn(packed.shape, generator=generator, dtype=torch.float32)
                packed = augmentation * packed + (1.0 - augmentation) * noise.to(packed.device)
            rows.append(packed.to(device))
        result = torch.cat(rows)
        if result.shape[0] != expected_rows:
            raise ValueError(
                f"condition rows {result.shape[0]} do not match packed layout {expected_rows}"
            )
        return result

    @staticmethod
    def _condition_audio_rows(
        latents: Sequence[torch.Tensor],
        *,
        device: torch.device,
        augmentation: float,
        seed: int,
        expected_rows: int,
    ) -> torch.Tensor | None:
        if not latents:
            if expected_rows:
                raise ValueError("packed layout expects reference audio rows")
            return None
        rows = []
        for latent in latents:
            packed = pack_audio(latent.float())
            if augmentation < 1.0:
                generator = torch.Generator("cpu").manual_seed(seed)
                noise = torch.randn(packed.shape, generator=generator, dtype=torch.float32)
                packed = augmentation * packed + (1.0 - augmentation) * noise.to(packed.device)
            rows.append(packed.to(device))
        result = torch.cat(rows)
        if result.shape[0] != expected_rows:
            raise ValueError(
                f"reference audio rows {result.shape[0]} do not match packed layout {expected_rows}"
            )
        return result

    @staticmethod
    def _timestep_plan(
        sigma_video: torch.Tensor,
        layout: PackedLayout,
        *,
        sigma_shift_video: float,
        sigma_shift_audio: float,
        visual_condition_timestep: float,
        audio_condition_timestep: float = 1.0,
        text_token_tags: torch.Tensor | None,
        device: torch.device,
        masked_video_prefix_rows: int = 0,
        masked_audio_prefix_rows: int = 0,
    ) -> tuple[
        torch.Tensor,
        tuple[ModulationSegment, ...],
        dict[str, int],
    ]:
        sigma = sigma_video.float().reshape(-1)[0].clamp(min=1e-6)
        video_t = float(1.0 - sigma)
        audio_t = float(
            1.0 - time_shift_sigma(sigma, sigma_shift_video, sigma_shift_audio)
        )
        has_condition = any(segment.kind == "condition" for segment in layout.segments)
        has_audio_condition = any(segment.kind == "ref_audio" for segment in layout.segments)
        condition_t = max(video_t, visual_condition_timestep)
        audio_condition_t = max(audio_t, audio_condition_timestep)
        masked_video_t = max(video_t, visual_condition_timestep)
        masked_audio_t = max(audio_t, audio_condition_timestep)
        unique_values = sorted(
            {video_t, audio_t}
            | ({condition_t} if has_condition else set())
            | ({audio_condition_t} if has_audio_condition else set())
            | ({masked_video_t} if masked_video_prefix_rows else set())
            | ({masked_audio_t} if masked_audio_prefix_rows else set())
        )
        rows = {value: index for index, value in enumerate(unique_values)}
        kind_time = {
            "text": video_t,
            "condition": condition_t,
            "ref_audio": audio_condition_t,
            "audio": audio_t,
            "video": video_t,
        }
        kind_tag = {"text": 1, "condition": 0, "ref_audio": 2, "audio": 2, "video": 0}
        segments: list[ModulationSegment] = []
        for segment in layout.segments:
            base = rows[kind_time[segment.kind]] * 3
            masked_prefix = (
                masked_video_prefix_rows
                if segment.kind == "video"
                else masked_audio_prefix_rows
                if segment.kind == "audio"
                else 0
            )
            if masked_prefix:
                if masked_prefix >= segment.length:
                    raise ValueError(
                        f"masked {segment.kind} prefix must leave generated rows"
                    )
                pin_time = (
                    masked_video_t
                    if segment.kind == "video"
                    else masked_audio_t
                )
                tag = kind_tag[segment.kind]
                segments.append((
                    segment.start,
                    segment.start + masked_prefix,
                    rows[pin_time] * 3 + tag,
                ))
                segments.append((
                    segment.start + masked_prefix,
                    segment.stop,
                    base + tag,
                ))
            elif segment.kind == "text" and text_token_tags is not None:
                tags = text_token_tags.reshape(-1).to(dtype=torch.long)
                if tags.numel() != segment.length:
                    raise ValueError("text_token_tags length does not match text sequence")
                if torch.any((tags < 0) | (tags > 2)):
                    raise ValueError("text_token_tags must contain H3 modality tags 0..2")
                # Presentation tags normally form only one or a few runs.  A
                # short host copy avoids creating a [sequence, hidden] gather
                # for every block and denoise step.
                tag_values = tags.detach().cpu().tolist()
                run_start = 0
                for offset in range(1, len(tag_values) + 1):
                    if offset == len(tag_values) or tag_values[offset] != tag_values[run_start]:
                        segments.append(
                            (
                                segment.start + run_start,
                                segment.start + offset,
                                base + int(tag_values[run_start]),
                            )
                        )
                        run_start = offset
            else:
                segments.append(
                    (
                        segment.start,
                        segment.stop,
                        base + kind_tag[segment.kind],
                    )
                )
        return (
            torch.tensor(unique_values, device=device, dtype=torch.float32),
            tuple(segments),
            {"video": rows[video_t], "audio": rows[audio_t]},
        )

    def forward(
        self,
        video_latent: torch.Tensor,
        audio_latent: torch.Tensor,
        context: torch.Tensor,
        sigma_video: torch.Tensor,
        *,
        condition_video_latents: Sequence[torch.Tensor] = (),
        condition_audio_latents: Sequence[torch.Tensor] = (),
        keyframe_indices: Sequence[int] = (),
        reference_shapes: Sequence[Sequence[int]] = (),
        reference_kinds: Sequence[str] = (),
        reference_audio_frames: Sequence[int] = (),
        output_frame_count: int | None = None,
        text_token_tags: torch.Tensor | None = None,
        visual_condition_timestep: float = 0.999,
        audio_condition_timestep: float = 1.0,
        condition_seed: int = 0,
        cache_condition_rows: bool = True,
        cache_condition_embeddings: bool = False,
        layout: PackedLayout | None = None,
        audio_transport_scale: float | None = None,
        sigma_shift_video: float | None = None,
        sigma_shift_audio: float | None = None,
        block_stack_runner: Callable[..., torch.Tensor] | None = None,
        mlp_chunk_tokens: int | None = None,
        final_projection_chunk_tokens: int | None = None,
        masked_video_prefix_latent_frames: int = 0,
        masked_audio_prefix_latent_frames: int = 0,
        target_time_offset: float = 0.0,
    ) -> H3DiTOutput:
        if video_latent.ndim != 5 or video_latent.shape[0] != 1:
            raise ValueError("video_latent must be batch-one [1,C,T,H,W]")
        if audio_latent.ndim != 4 or audio_latent.shape[0] != 1:
            raise ValueError("audio_latent must be batch-one [1,C,2,T]")
        if context.ndim != 3 or context.shape[0] != 1:
            raise ValueError("context must be batch-one [1,L,D]")
        if reference_shapes or reference_audio_frames:
            expected_video_conditions = len(reference_shapes) + len(keyframe_indices)
            if len(condition_video_latents) != expected_video_conditions:
                raise ValueError(
                    "each reference and keyframe requires one condition latent"
                )
            if len(condition_audio_latents) != len(reference_audio_frames):
                raise ValueError("each Ref2VA audio reference requires one condition latent")
        elif len(condition_video_latents) != len(keyframe_indices):
            raise ValueError("each keyframe anchor requires one condition latent")
        if audio_transport_scale is not None and audio_transport_scale <= 0.0:
            raise ValueError("audio_transport_scale must be positive")
        if not 0 <= int(masked_video_prefix_latent_frames) < int(video_latent.shape[2]):
            raise ValueError("masked video prefix must leave generated latent frames")
        if not 0 <= int(masked_audio_prefix_latent_frames) < int(audio_latent.shape[-1]):
            raise ValueError("masked audio prefix must leave generated latent frames")
        active_video_shift = float(
            self.config.sigma_shift_video
            if sigma_shift_video is None else sigma_shift_video
        )
        active_audio_shift = float(
            self.config.sigma_shift_audio
            if sigma_shift_audio is None else sigma_shift_audio
        )
        if active_video_shift <= 0.0 or active_audio_shift <= 0.0:
            raise ValueError("sigma shifts must be positive")
        target_time_offset = float(target_time_offset)
        if not math.isfinite(target_time_offset) or target_time_offset < 0.0:
            raise ValueError("target_time_offset must be finite and non-negative")
        if target_time_offset and (reference_shapes or reference_audio_frames):
            raise ValueError("target_time_offset is not supported for Ref2VA layouts")

        # Current H3 runtimes carry the audio stream on the video sigma clock.
        # At a source step the carried latent is converted back to its own
        # audio clock before entering the network.  Keeping this transformation
        # inside the model boundary also lets the returned velocity use the
        # exact reference expression below instead of an algebraically reduced
        # dual-clock update with different floating-point rounding.
        audio_source = audio_latent
        sigma = sigma_video.float().reshape(-1)[0].clamp(min=1e-6)
        sigma_audio = time_shift_sigma(
            sigma, active_video_shift, active_audio_shift
        )
        if audio_transport_scale is not None:
            carry = (sigma_audio / sigma).to(audio_source.dtype)
            audio_latent = audio_source * carry

        # Match the accepted model wrapper boundary: the sampler carries FP32
        # state, but H3 receives both latent streams in the DiT compute dtype.
        # Patch projections are FP32 operations over values that have already
        # crossed this BF16 storage boundary.
        video_latent = video_latent.to(self.compute_dtype)
        audio_latent = audio_latent.to(self.compute_dtype)

        original_t, original_h, original_w = video_latent.shape[-3:]
        video_latent = _pad_to_patch(video_latent, self.config.patch_size)
        latent_t, latent_h, latent_w = video_latent.shape[-3:]
        audio_frames = int(audio_latent.shape[-1])
        text_length = int(context.shape[1])
        if layout is None:
            if (reference_shapes or reference_audio_frames) and keyframe_indices:
                if output_frame_count is None:
                    raise ValueError(
                        "output_frame_count is required for mixed keyframe conditioning"
                    )
                layout = build_hybrid_condition_layout(
                    text_length=text_length,
                    latent_frames=latent_t,
                    latent_height=latent_h,
                    latent_width=latent_w,
                    audio_frames=audio_frames,
                    reference_shapes=reference_shapes,
                    reference_kinds=reference_kinds,
                    reference_audio_frames=reference_audio_frames,
                    keyframe_indices=keyframe_indices,
                    output_frame_count=output_frame_count,
                )
            elif reference_shapes or reference_audio_frames:
                layout = build_ref2va_layout(
                    text_length=text_length,
                    latent_frames=latent_t,
                    latent_height=latent_h,
                    latent_width=latent_w,
                    audio_frames=audio_frames,
                    reference_shapes=reference_shapes,
                    reference_kinds=reference_kinds,
                    reference_audio_frames=reference_audio_frames,
                )
            else:
                layout = build_fl2va_layout(
                    text_length=text_length,
                    latent_frames=latent_t,
                    latent_height=latent_h,
                    latent_width=latent_w,
                    audio_frames=audio_frames,
                    keyframe_indices=keyframe_indices,
                    output_frame_count=output_frame_count,
                    target_time_offset=target_time_offset,
                )
        condition_signature = (
            (
                "hybrid_reference_keyframe_v1",
                tuple(value for shape in reference_shapes for value in map(int, shape)),
                tuple(reference_kinds) if reference_kinds else ("image",) * len(reference_shapes),
                tuple(int(value) for value in reference_audio_frames),
                tuple(int(index) for index in keyframe_indices),
                int(output_frame_count or 0),
            )
            if (reference_shapes or reference_audio_frames) and keyframe_indices
            else (
                tuple(value for shape in reference_shapes for value in map(int, shape)),
                tuple(reference_kinds) if reference_kinds else ("image",) * len(reference_shapes),
                tuple(int(value) for value in reference_audio_frames),
            )
            if reference_shapes or reference_audio_frames
            else (
                tuple(int(index) for index in keyframe_indices)
                if target_time_offset == 0.0
                else (
                    tuple(int(index) for index in keyframe_indices),
                    ("target_time_offset", target_time_offset),
                )
            )
        )
        expected_signature = (
            text_length,
            latent_t,
            latent_h,
            latent_w,
            audio_frames,
            condition_signature,
        )
        if layout.signature != expected_signature:
            raise ValueError(
                f"cached packed layout {layout.signature} does not match {expected_signature}"
            )
        device = video_latent.device

        target_video_rows = patchify_video(video_latent.float(), self.config.patch_size)
        condition_count = int((~layout.video_update_mask).sum())
        condition_rows = self._request_local_tensor(
            layout,
            "device_video_condition_rows",
            enabled=cache_condition_rows,
            device=device,
            builder=lambda: self._condition_rows(
                condition_video_latents,
                device=device,
                augmentation=visual_condition_timestep,
                seed=condition_seed,
                expected_rows=condition_count,
            ),
        )
        update_mask = layout.device_video_update_mask
        if update_mask is None or update_mask.device != device:
            update_mask = layout.video_update_mask.to(device)
            layout.device_video_update_mask = update_mask
        if cache_condition_embeddings and condition_count:
            assert condition_rows is not None
            condition_embeddings = self._request_local_tensor(
                layout,
                "device_video_condition_embeddings",
                enabled=True,
                device=device,
                builder=lambda: self.video_patch_proj(condition_rows).to(
                    self.compute_dtype
                ),
            )
            assert condition_embeddings is not None
            target_video_embeddings = self.video_patch_proj(
                target_video_rows
            ).to(self.compute_dtype)
            video_embeddings = torch.empty(
                layout.video_update_mask.numel(),
                self.config.hidden_size,
                dtype=self.compute_dtype,
                device=device,
            )
            video_embeddings[update_mask] = target_video_embeddings
            video_embeddings[~update_mask] = condition_embeddings
        else:
            all_video_rows = torch.empty(
                layout.video_update_mask.numel(),
                target_video_rows.shape[-1],
                dtype=torch.float32,
                device=device,
            )
            all_video_rows[update_mask] = target_video_rows
            if condition_count:
                assert condition_rows is not None
                all_video_rows[~update_mask] = condition_rows
            video_embeddings = self.video_patch_proj(all_video_rows).to(
                self.compute_dtype
            )
        target_audio_rows = pack_audio(audio_latent.float())
        reference_audio_count = int((~layout.audio_update_mask).sum())
        condition_audio_rows = self._request_local_tensor(
            layout,
            "device_audio_condition_rows",
            enabled=cache_condition_rows,
            device=device,
            builder=lambda: self._condition_audio_rows(
                condition_audio_latents,
                device=device,
                augmentation=audio_condition_timestep,
                seed=condition_seed + 1,
                expected_rows=reference_audio_count,
            ),
        )
        audio_update_mask = layout.device_audio_update_mask
        if audio_update_mask is None or audio_update_mask.device != device:
            audio_update_mask = layout.audio_update_mask.to(device)
            layout.device_audio_update_mask = audio_update_mask
        if cache_condition_embeddings and reference_audio_count:
            assert condition_audio_rows is not None
            condition_audio_embeddings = self._request_local_tensor(
                layout,
                "device_audio_condition_embeddings",
                enabled=True,
                device=device,
                builder=lambda: self.audio_patch_proj(condition_audio_rows).to(
                    self.compute_dtype
                ),
            )
            assert condition_audio_embeddings is not None
            target_audio_embeddings = self.audio_patch_proj(
                target_audio_rows
            ).to(self.compute_dtype)
            audio_embeddings = torch.empty(
                layout.audio_update_mask.numel(),
                self.config.hidden_size,
                dtype=self.compute_dtype,
                device=device,
            )
            audio_embeddings[audio_update_mask] = target_audio_embeddings
            audio_embeddings[~audio_update_mask] = condition_audio_embeddings
        else:
            all_audio_rows = torch.empty(
                layout.audio_update_mask.numel(), target_audio_rows.shape[-1],
                dtype=torch.float32, device=device,
            )
            all_audio_rows[audio_update_mask] = target_audio_rows
            if reference_audio_count:
                assert condition_audio_rows is not None
                all_audio_rows[~audio_update_mask] = condition_audio_rows
            audio_embeddings = self.audio_patch_proj(all_audio_rows).to(
                self.compute_dtype
            )
        text_states = context[0]
        if text_states.shape[-1] == self.config.text_dim:
            # Qwen conditions are produced in FP32 for reference parity.  The
            # H3 model wrapper casts context to the DiT compute dtype before
            # the 5120 -> 5376 projection; make that boundary explicit rather
            # than relying on the caller's text-encoder dtype.
            text_states = text_states.to(self.compute_dtype)
            text_states = self.token_refiner(self.condition_proj(text_states))
        elif text_states.shape[-1] != self.config.hidden_size:
            raise ValueError("context width must be text_dim or hidden_size")
        text_states = text_states.to(self.compute_dtype)

        packed = torch.empty(
            layout.sequence_length,
            self.config.hidden_size,
            dtype=self.compute_dtype,
            device=device,
        )
        video_offset = audio_offset = 0
        for segment in layout.segments:
            target = slice(segment.start, segment.stop)
            if segment.kind == "text":
                packed[target] = text_states
            elif segment.kind in ("condition", "video"):
                packed[target] = video_embeddings[
                    video_offset : video_offset + segment.length
                ]
                video_offset += segment.length
            else:
                packed[target] = audio_embeddings[
                    audio_offset : audio_offset + segment.length
                ]
                audio_offset += segment.length

        unique_timesteps, modulation_segments, final_rows = self._timestep_plan(
            sigma_video,
            layout,
            sigma_shift_video=active_video_shift,
            sigma_shift_audio=active_audio_shift,
            visual_condition_timestep=visual_condition_timestep,
            audio_condition_timestep=audio_condition_timestep,
            text_token_tags=text_token_tags,
            device=device,
            masked_video_prefix_rows=(
                int(masked_video_prefix_latent_frames)
                * layout.segment("video", last=True).length
                // int(video_latent.shape[2])
            ),
            masked_audio_prefix_rows=(
                int(masked_audio_prefix_latent_frames)
                * layout.segment("audio", last=True).length
                // int(audio_latent.shape[-1])
            ),
        )
        frequencies = layout.device_rope_table
        if frequencies is None or frequencies.device != device:
            frequencies = rope_rotation_table(
                rope_frequencies(
                    layout.position_ids.to(device), self.rope_inv_freq.to(device)
                ),
                self.compute_dtype,
            )
            layout.device_rope_table = frequencies
        curve_rows = self.block_stack.prepare_curve_rows(unique_timesteps)
        generated_video = layout.segment("video", last=True)
        latent_frame_count = int(video_latent.shape[2])
        if generated_video.length % latent_frame_count:
            raise ValueError("generated video rows do not form complete latent frames")
        frame_tokens = generated_video.length // latent_frame_count
        masked_video_prefix_rows = (
            int(masked_video_prefix_latent_frames) * frame_tokens
        )
        # Sparse H3 attention treats every row before ``protected_prefix`` as
        # dense context and lets each generated-video query attend all of it.
        # A continuation window's clean video prefix is contiguous with the
        # ordinary text/condition/audio prefix, so extending this boundary is
        # the exact sparse analogue of Continuum's masked joint-AV context:
        # carried frames remain full-fidelity keys and queries while only the
        # newly generated suffix is eligible for video-to-video sparsity.
        protected_prefix = generated_video.start + masked_video_prefix_rows
        sparse_video_frame_count = (
            latent_frame_count - int(masked_video_prefix_latent_frames)
        )
        patch_t, patch_h, patch_w = self.config.patch_size
        if patch_t != 1:
            raise ValueError("attention video geometry currently requires temporal patch size 1")
        grid_height = int(video_latent.shape[3]) // int(patch_h)
        grid_width = int(video_latent.shape[4]) // int(patch_w)
        if grid_height * grid_width != frame_tokens:
            raise ValueError("generated video grid does not match packed frame tokens")
        interleave_config = current_frame_interleave_config()
        interleave = (
            None
            if interleave_config is None or interleave_config.stride == 1
            else FrameInterleavePlan(layout, interleave_config, device)
        )
        lattice_config = current_spatial_query_lattice_config()
        lattice = (
            None
            if lattice_config is None or lattice_config.stride == 1
            else SpatialQueryLatticePlan(layout, lattice_config, device)
        )
        mlp_lattice_config = current_mlp_spatial_lattice_config()
        mlp_lattice = (
            None
            if mlp_lattice_config is None or mlp_lattice_config.stride == 1
            else MLPSpatialLatticePlan(layout, mlp_lattice_config, device)
        )
        with (
            attention_protected_prefix(protected_prefix),
            attention_video_layout(
                sparse_video_frame_count,
                frame_tokens,
                grid_height=grid_height,
                grid_width=grid_width,
            ),
            frame_interleave_plan(interleave),
            spatial_query_lattice_plan(lattice),
            mlp_spatial_lattice_plan(mlp_lattice),
        ):
            if block_stack_runner is None:
                packed = self.block_stack(
                    packed,
                    unique_timesteps=unique_timesteps,
                    modulation_segments=modulation_segments,
                    frequencies=frequencies,
                    curve_rows=curve_rows,
                    mlp_chunk_tokens=mlp_chunk_tokens,
                )
            else:
                packed = block_stack_runner(
                    self.block_stack,
                    packed,
                    layout=layout,
                    unique_timesteps=unique_timesteps,
                    modulation_segments=modulation_segments,
                    frequencies=frequencies,
                    curve_rows=curve_rows,
                    mlp_chunk_tokens=mlp_chunk_tokens,
                )

        video_segment = layout.segment("video", last=True)
        audio_segment = layout.segment("audio", last=True)
        video_rows, audio_rows = self.final_layer(
            packed,
            curve_rows=curve_rows,
            video_segment=slice(video_segment.start, video_segment.stop),
            audio_segment=slice(audio_segment.start, audio_segment.stop),
            video_timestep_row=final_rows["video"],
            audio_timestep_row=final_rows["audio"],
            final_projection_chunk_tokens=final_projection_chunk_tokens,
        )
        patch_t, patch_h, patch_w = self.config.patch_size
        video_output = unpatchify_video(
            video_rows,
            latent_shape=(
                latent_t // patch_t,
                latent_h // patch_h,
                latent_w // patch_w,
                self.config.video_channels,
            ),
            patch_size=self.config.patch_size,
        )[..., :original_t, :original_h, :original_w]
        audio_output = unpack_audio(audio_rows)
        raw_audio_velocity = -audio_output.to(audio_latent.dtype)
        if audio_transport_scale is None:
            audio_slope = time_shift_slope(
                sigma, active_video_shift, active_audio_shift
            ).to(raw_audio_velocity.dtype)
            audio_velocity = audio_slope * raw_audio_velocity
        else:
            scale = float(audio_transport_scale)
            audio_velocity = (
                (1.0 - scale) * audio_latent
                + (1.0 + (scale - 1.0) * sigma_audio).to(
                    raw_audio_velocity.dtype
                )
                * raw_audio_velocity
            )
        return H3DiTOutput(
            video=-video_output.to(video_latent.dtype),
            audio=audio_velocity,
            layout=layout,
        )
