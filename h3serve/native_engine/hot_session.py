"""Persistent real-weight T2AV session for one RTX 4090.

This module owns request-to-request lifecycle, not checkpoint construction.
Callers inject already prepared immutable residencies so the same implementation
can serve both the pruned INT8 base route and the Larry LoRA route.
"""

from __future__ import annotations

import gc
import ctypes
import hashlib
import json
import math
import os
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Literal

import torch
import torch.nn.functional as F

from .adapters.sampling_mux import (
    AVPrediction,
    AtomicPyAVMuxer,
    ResMultistepAVSampler,
    SASolverAVSampler,
    SamplingPlan,
    TurboAVSampler,
    TurboClockMode,
    comfy_denoise_tail_sigma_schedule,
    refinement_sigma_schedule,
    simple_sigma_schedule,
)
from .forecast import (
    DirectionalForecastController,
    ForecastErrorDebtController,
    V24_FORECAST_FEEDBACK_POLICY_ID,
)
from .model import (
    AttentionOnlineBudget,
    FrameInterleaveConfig,
    SpatialQueryLatticeConfig,
    attention_action_schedule as attention_action_schedule_context,
    attention_actual_steps,
    attention_online_budget,
    attention_sparsity,
    attention_force_dense,
    attention_step,
    block_cancellation,
    build_fl2va_layout,
    build_h3_block_executor,
    dense_qk_quantization,
    frame_interleave_config,
    spatial_query_lattice_config,
    MLPSpatialLatticeConfig,
    mlp_spatial_lattice_config,
    rms_adaln_fusion,
    long_video_attention,
    long_sequence_query_chunking,
)
from .planner import (
    ExecutionPlan,
    H3WorkloadAnalyzer,
    LONG_SEQUENCE_VALIDATED_MAX_PACKED_TOKENS,
    NoFeasibleProfile,
    RTX4090Planner,
    select_dense_safe_resident_blocks,
    select_memory_execution,
    select_long_sequence_chunks,
    select_stable_dense_qk_quantization,
)
from .runtime import ImmutablePinnedModuleResidency, OffloadMode, RuntimeConfig
from .segment_cache import (
    CoordinateAlignedSegmentCache,
    SegmentResidualCacheConfig,
)
from .terminal_latent_guard import stabilize_terminal_video_latent_
from .long_horizon import (
    H3_FRAME_ORIGIN,
    H3_FRAME_STRIDE,
    MAX_CONTINUATION_CONTEXT_FRAMES,
    audio_latent_frames,
    prepare_masked_av_prefix,
    restore_masked_av_prefix_,
    video_latent_frames,
)
from .global_co_denoise import (
    DEFAULT_WINDOW_FRAMES as GLOBAL_AV_WINDOW_FRAMES,
    DEFAULT_WINDOW_STRIDE_FRAMES as GLOBAL_AV_WINDOW_STRIDE_FRAMES,
    fuse_global_av_predictions,
    plan_balanced_global_av_windows,
    plan_global_av_windows,
    plan_prompt_owned_global_av_windows,
    stabilize_global_selflift_seams,
)


VideoDecoder = Callable[[Any, torch.Tensor, int], torch.Tensor]
AudioDecoder = Callable[[Any, torch.Tensor], torch.Tensor]


QWEN_CONDITIONING_CACHE_SCHEMA_VERSION = 1
QWEN_CONDITIONING_PREPROCESS_VERSION = (
    "h3_qwen_static_conditioning_v1"
)
VideoConditionEncoder = Callable[[Any, Any], Any]
AudioConditionEncoder = Callable[[Any, Any], Any]


# A prepared H3 QKV activation stores one INT8 value per hidden channel and
# one FP32 row scale.  Keep the admission formula next to the request planner
# rather than hiding it in a resolution/prompt branch: ``packed_tokens``
# already includes every FL2VA/Ref2VA condition token.  The extra 512-MiB
# release reserve covers allocator fragmentation and the calibrated model's
# residual error; the memory planner independently retains its ordinary
# 128-MiB guard.
_H3_SHARED_QKV_BYTES_PER_PACKED_TOKEN = 5_376 + 4
_H3_SHARED_QKV_RELEASE_RESERVE_BYTES = 512 * 1024**2
_H3_MEMORY_POLICY_GUARD_BYTES = 128 * 1024**2
_H3_W4A8_SECOND_BLOCK_BUFFER_BYTES = 218_768_128
_H3_INT8_BLOCK_BUFFER_BYTES = 387_359_520


class HotSessionCancelled(RuntimeError):
    """A queued request was cancelled between safe GPU operations."""


class HotSessionDeviceFatal(RuntimeError):
    """CUDA context is poisoned and must not be touched again in-process."""


def _is_cuda_context_fatal(error: BaseException) -> bool:
    """Classify asynchronous CUDA faults without issuing another CUDA call."""

    class_name = error.__class__.__name__.lower()
    message = str(error).lower()
    return (
        "acceleratorerror" in class_name
        or "illegal memory access" in message
        or "device-side assert" in message
        or "launch failure" in message
        or "misaligned address" in message
        or "device not ready" in message
        or "device is not ready" in message
    )


class _HotSessionCheckpointReached(RuntimeError):
    """Internal control flow after a requested sampler checkpoint."""


def resize_refinement_video_latent_spatial(
    latent: torch.Tensor,
    *,
    target_height: int,
    target_width: int,
) -> torch.Tensor:
    """Resize only the spatial axes of an H3 video latent.

    H3 stores video latents as ``B,C,T,H,W``.  Flattening ``B*T`` before the
    interpolation makes the important invariant explicit: no information is
    mixed between adjacent latent frames, so the first pass owns the motion
    trajectory while the second pass receives new spatial degrees of freedom.
    """

    if latent.ndim != 5:
        raise ValueError("refinement video latent must have shape B,C,T,H,W")
    if target_height <= 0 or target_width <= 0:
        raise ValueError("target latent height and width must be positive")
    batch, channels, latent_frames, source_height, source_width = latent.shape
    if (source_height, source_width) == (target_height, target_width):
        return latent
    frame_batch = latent.permute(0, 2, 1, 3, 4).reshape(
        batch * latent_frames,
        channels,
        source_height,
        source_width,
    )
    resized = F.interpolate(
        frame_batch.float(),
        size=(target_height, target_width),
        mode="bicubic",
        align_corners=False,
        antialias=True,
    )
    return resized.reshape(
        batch,
        latent_frames,
        channels,
        target_height,
        target_width,
    ).permute(0, 2, 1, 3, 4).contiguous()


def build_refinement_region_mask(
    *,
    height: int,
    width: int,
    regions: tuple[tuple[float, float, float, float], ...],
    feather: float,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Build one soft spatial write mask while retaining a full H3 canvas.

    Coordinates are normalized ``x0,y0,x1,y1`` boxes.  A value of one grants
    the H3 prediction full authority; zero restores the accepted source clean
    latent.  Keeping the entire canvas in the DiT preserves global context,
    unlike the earlier crop-atlas experiment.
    """

    if height <= 0 or width <= 0:
        raise ValueError("refinement mask dimensions must be positive")
    if not 0.0 <= feather <= 0.25:
        raise ValueError("refinement mask feather must lie inside [0,0.25]")
    mask = torch.zeros((1, 1, 1, height, width), device=device, dtype=torch.float32)
    if not regions:
        return mask
    x = (torch.arange(width, device=device, dtype=torch.float32) + 0.5) / width
    y = (torch.arange(height, device=device, dtype=torch.float32) + 0.5) / height

    def smoothstep(value: torch.Tensor) -> torch.Tensor:
        value = value.clamp(0.0, 1.0)
        return value.square() * (3.0 - 2.0 * value)

    for region in regions:
        if len(region) != 4:
            raise ValueError("each full-canvas region must contain x0,y0,x1,y1")
        x0, y0, x1, y1 = (float(value) for value in region)
        if not (0.0 <= x0 < x1 <= 1.0 and 0.0 <= y0 < y1 <= 1.0):
            raise ValueError("full-canvas region coordinates must lie inside [0,1]")
        if feather == 0.0:
            x_weight = ((x >= x0) & (x <= x1)).float()
            y_weight = ((y >= y0) & (y <= y1)).float()
        else:
            x_weight = smoothstep((x - (x0 - feather)) / feather) * smoothstep(
                ((x1 + feather) - x) / feather
            )
            y_weight = smoothstep((y - (y0 - feather)) / feather) * smoothstep(
                ((y1 + feather) - y) / feather
            )
        region_mask = y_weight[:, None] * x_weight[None, :]
        mask[0, 0, 0] = torch.maximum(mask[0, 0, 0], region_mask)
    return mask


def blend_terminal_refinement_detail(
    motion_latent: torch.Tensor,
    refined_latent: torch.Tensor,
    *,
    source_height: int,
    source_width: int,
    low_frequency_gain: float,
    temporal_lowpass: bool = False,
    temporal_outlier_only: bool = False,
    temporal_detail_outlier_strength: float = 0.0,
) -> torch.Tensor:
    """Keep the base trajectory's motion while retaining refined detail.

    The terminal pass is intentionally allowed to create spatial frequencies
    that did not exist on the motion canvas.  Its low-frequency residual can,
    however, also rewrite object position and coarse geometry.  Decompose that
    residual at the original latent resolution, attenuate only the low band,
    and retain the complete high band.  A gain of one is exactly the former
    behavior; zero anchors all coarse motion to ``motion_latent``.
    """

    if motion_latent.shape != refined_latent.shape or motion_latent.ndim != 5:
        raise ValueError("terminal refinement latents must share B,C,T,H,W shape")
    if not 0.0 <= low_frequency_gain <= 1.0:
        raise ValueError("terminal low-frequency gain must lie inside [0, 1]")
    if temporal_outlier_only and not temporal_lowpass:
        raise ValueError("temporal outlier filtering requires temporal lowpass")
    if not 0.0 <= temporal_detail_outlier_strength <= 1.0:
        raise ValueError(
            "temporal detail outlier strength must lie inside [0, 1]"
        )
    if (
        low_frequency_gain == 1.0
        and not temporal_lowpass
        and temporal_detail_outlier_strength == 0.0
    ):
        return refined_latent
    batch, channels, latent_frames, height, width = refined_latent.shape
    if not (0 < source_height <= height and 0 < source_width <= width):
        raise ValueError("terminal source geometry must fit the refined latent")
    delta = refined_latent.float() - motion_latent.float()
    frame_delta = delta.permute(0, 2, 1, 3, 4).reshape(
        batch * latent_frames, channels, height, width
    )
    low = F.interpolate(
        frame_delta,
        size=(source_height, source_width),
        mode="area",
    )
    low = F.interpolate(
        low,
        size=(height, width),
        mode="bicubic",
        align_corners=False,
        antialias=True,
    )
    detail = frame_delta - low
    if temporal_detail_outlier_strength > 0.0 and latent_frames > 2:
        detail_sequence = detail.reshape(
            batch, latent_frames, channels, height, width
        )
        detail_previous = torch.cat(
            (detail_sequence[:, :1], detail_sequence[:, :-1]), dim=1
        )
        detail_following = torch.cat(
            (detail_sequence[:, 1:], detail_sequence[:, -1:]), dim=1
        )
        detail_smoothed = (
            detail_previous + 2.0 * detail_sequence + detail_following
        ) * 0.25
        detail_innovation = detail_sequence - detail_smoothed

        # A real moving edge also changes between neighbouring latent frames.
        # Measure that motion on the accepted source trajectory and use it as
        # an allowance, so the filter targets newly hallucinated one-frame
        # texture rather than flattening normal subject/camera motion.
        motion_sequence = motion_latent.float().permute(0, 2, 1, 3, 4)
        motion_previous = torch.cat(
            (motion_sequence[:, :1], motion_sequence[:, :-1]), dim=1
        )
        motion_following = torch.cat(
            (motion_sequence[:, 1:], motion_sequence[:, -1:]), dim=1
        )
        motion_innovation = motion_sequence - (
            motion_previous + 2.0 * motion_sequence + motion_following
        ) * 0.25
        detail_score = detail_innovation.square().mean(dim=2).sqrt()
        motion_score = motion_innovation.square().mean(dim=2).sqrt()
        excess_score = (detail_score - 1.25 * motion_score).clamp_min(0.0)
        # Pool only spatially.  A coherent mask avoids stippled per-channel
        # corrections while leaving the temporal robust statistics intact.
        excess_score = F.avg_pool3d(
            excess_score.unsqueeze(1),
            kernel_size=(1, 3, 3),
            stride=1,
            padding=(0, 1, 1),
        ).squeeze(1)
        median = excess_score.median(dim=1, keepdim=True).values
        mad = (excess_score - median).abs().median(dim=1, keepdim=True).values
        threshold = median + 2.5 * (1.4826 * mad).clamp_min(1e-6)
        outlier_weight = (
            (excess_score - threshold).clamp_min(0.0)
            / excess_score.clamp_min(1e-6)
        ).unsqueeze(2)
        outlier_weight = outlier_weight * temporal_detail_outlier_strength
        detail_sequence = detail_sequence + outlier_weight * (
            detail_smoothed - detail_sequence
        )
        detail = detail_sequence.reshape_as(frame_delta)
    if temporal_lowpass and latent_frames > 1:
        low_sequence = low.reshape(
            batch, latent_frames, channels, height, width
        )
        previous = torch.cat(
            (low_sequence[:, :1], low_sequence[:, :-1]), dim=1
        )
        following = torch.cat(
            (low_sequence[:, 1:], low_sequence[:, -1:]), dim=1
        )
        # Binomial [1, 2, 1] filtering preserves temporally constant detail
        # exactly while suppressing a one-frame coarse correction spike.  It
        # has no threshold or content-specific hand tuning.
        smoothed = (previous + 2.0 * low_sequence + following) * 0.25
        if temporal_outlier_only and latent_frames > 2:
            innovation = low_sequence - smoothed
            score = innovation.square().mean(dim=(2, 3, 4)).sqrt()
            median = score.median(dim=1, keepdim=True).values
            mad = (score - median).abs().median(dim=1, keepdim=True).values
            robust_sigma = (1.4826 * mad).clamp_min(1e-6)
            threshold = median + 3.0 * robust_sigma
            # Preserve all in-distribution corrections exactly.  For an
            # outlier, remove only the fraction above the robust 3-sigma
            # envelope instead of replacing the frame wholesale.
            outlier_weight = (
                (score - threshold).clamp_min(0.0) / score.clamp_min(1e-6)
            ).view(batch, latent_frames, 1, 1, 1)
            low_sequence = low_sequence + outlier_weight * (
                smoothed - low_sequence
            )
            low = low_sequence.reshape_as(frame_delta)
        else:
            low = smoothed.reshape_as(frame_delta)
    blended = (
        motion_latent.float().permute(0, 2, 1, 3, 4).reshape_as(frame_delta)
        + detail
        + low_frequency_gain * low
    )
    return blended.reshape(
        batch, latent_frames, channels, height, width
    ).permute(0, 2, 1, 3, 4).contiguous().to(refined_latent.dtype)


def damp_unconverged_refinement_detail(
    motion_latent: torch.Tensor,
    previous_prediction: torch.Tensor,
    refined_latent: torch.Tensor,
    *,
    source_height: int,
    source_width: int,
    strength: float,
) -> torch.Tensor:
    """Damp high-frequency corrections that have not converged across steps.

    Low-step H3 refinement can create attractive target-grid texture that is
    still moving between its last two clean-state predictions.  That
    disagreement is a stronger confidence signal than temporal smoothing: it
    does not mistake real object or camera motion for flicker because both
    predictions describe the same complete clip.  Only spatial high-frequency
    outliers are mixed toward the penultimate prediction; coarse geometry and
    converged detail remain bit-for-bit unchanged.
    """

    if (
        motion_latent.shape != previous_prediction.shape
        or motion_latent.shape != refined_latent.shape
        or motion_latent.ndim != 5
    ):
        raise ValueError(
            "cross-step refinement latents must share B,C,T,H,W shape"
        )
    if not 0.0 <= strength <= 1.0:
        raise ValueError("cross-step detail strength must lie inside [0, 1]")
    if strength == 0.0:
        return refined_latent
    batch, channels, latent_frames, height, width = refined_latent.shape
    if not (0 < source_height <= height and 0 < source_width <= width):
        raise ValueError("cross-step source geometry must fit the refined latent")

    def high_detail(value: torch.Tensor) -> torch.Tensor:
        delta = value.float() - motion_latent.float()
        frames = delta.permute(0, 2, 1, 3, 4).reshape(
            batch * latent_frames, channels, height, width
        )
        low = F.interpolate(
            frames,
            size=(source_height, source_width),
            mode="area",
        )
        low = F.interpolate(
            low,
            size=(height, width),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )
        return (frames - low).reshape(
            batch, latent_frames, channels, height, width
        )

    previous_detail = high_detail(previous_prediction)
    final_detail = high_detail(refined_latent)
    disagreement = final_detail - previous_detail
    score = disagreement.square().mean(dim=2).sqrt()
    score = F.avg_pool3d(
        score.unsqueeze(1),
        kernel_size=(1, 3, 3),
        stride=1,
        padding=(0, 1, 1),
    ).squeeze(1)
    flat_score = score.flatten(start_dim=2)
    median = flat_score.median(dim=2, keepdim=True).values
    mad = (flat_score - median).abs().median(dim=2, keepdim=True).values
    threshold = median + 2.5 * (1.4826 * mad).clamp_min(1e-6)
    threshold = threshold.view(batch, latent_frames, 1, 1)
    outlier_weight = (
        (score - threshold).clamp_min(0.0) / score.clamp_min(1e-6)
    ).unsqueeze(2)
    # Pull an outlier halfway toward the penultimate clean prediction at full
    # strength.  This retains late-step detail while avoiding a hard switch to
    # the less-converged state.
    correction = 0.5 * strength * outlier_weight * (
        previous_detail - final_detail
    )
    correction = correction.reshape(
        batch * latent_frames, channels, height, width
    )
    refined_frames = refined_latent.float().permute(0, 2, 1, 3, 4).reshape(
        batch * latent_frames, channels, height, width
    )
    return (refined_frames + correction).reshape(
        batch, latent_frames, channels, height, width
    ).permute(0, 2, 1, 3, 4).contiguous().to(refined_latent.dtype)


def spatial_highpass_noise(
    noise: torch.Tensor,
    *,
    low_height: int,
    low_width: int,
) -> torch.Tensor:
    """Return normalized detail-band noise absent from a low-res trajectory."""

    if noise.ndim != 5:
        raise ValueError("multiscale noise must have shape B,C,T,H,W")
    batch, channels, latent_frames, height, width = noise.shape
    if not (0 < low_height <= height and 0 < low_width <= width):
        raise ValueError("low-resolution noise geometry must fit the target")
    frames = noise.permute(0, 2, 1, 3, 4).reshape(
        batch * latent_frames, channels, height, width
    ).float()
    low = F.interpolate(frames, size=(low_height, low_width), mode="area")
    reconstructed = F.interpolate(
        low,
        size=(height, width),
        mode="bicubic",
        align_corners=False,
        antialias=True,
    )
    highpass = frames - reconstructed
    scale = highpass.square().mean(dim=(-2, -1), keepdim=True).sqrt().clamp_min(1e-6)
    highpass = highpass / scale
    return highpass.reshape(
        batch, latent_frames, channels, height, width
    ).permute(0, 2, 1, 3, 4).contiguous()


def selflift_renoise_clean_endpoint(
    clean: torch.Tensor,
    noise: torch.Tensor,
    sigma: float,
) -> torch.Tensor:
    """Place a lifted rectified-flow clean endpoint at ``sigma``."""

    if clean.shape != noise.shape:
        raise ValueError("SelfLift clean latent and noise must have equal shapes")
    if not 0.0 <= float(sigma) <= 1.0:
        raise ValueError("SelfLift sigma must lie inside [0, 1]")
    return float(sigma) * noise + (1.0 - float(sigma)) * clean


def restore_selflift_target_prefix_(
    lifted: torch.Tensor,
    target_prefix: torch.Tensor | None,
) -> int:
    """Restore the prior target-grid overlap into a freshly lifted latent."""

    if target_prefix is None:
        return 0
    if lifted.ndim != 5 or target_prefix.ndim != 5:
        raise ValueError("SelfLift target prefix tensors must be five-dimensional")
    if lifted.shape[:2] != target_prefix.shape[:2] or lifted.shape[3:] != target_prefix.shape[3:]:
        raise ValueError("SelfLift target prefix geometry is incompatible")
    tokens = int(target_prefix.shape[2])
    if tokens > int(lifted.shape[2]):
        raise ValueError("SelfLift target prefix exceeds the lifted trajectory")
    lifted[:, :, :tokens].copy_(target_prefix.to(device=lifted.device, dtype=lifted.dtype))
    return tokens


@dataclass(frozen=True, slots=True)
class HotSessionRequest:
    prompt: str
    seed: int
    width: int
    height: int
    frames: int
    fps: int
    steps: int
    output_path: Path
    actual_step_indices: tuple[int, ...] | None = None
    # Exact request-local optimization: reference rows are constant across
    # denoise steps.  Keep this switch so benchmark v00 can replay the former
    # implementation and v01 can be compared in the same code revision.
    cache_condition_rows: bool = True
    cache_condition_embeddings: bool = False
    cache_reference_latents: bool = True
    mlp_chunk_tokens: int | None = None
    execution_plan: ExecutionPlan | None = None
    # Release-only mechanical optimizations that have passed a same-process,
    # full-DiT byte-equivalence gate.  Benchmarks default this off so their
    # explicit reference graph remains a real legacy comparator; the service
    # enables it after constructing the user request.
    release_byte_exact_optimizations: bool = False
    # Legacy persisted requests may still carry the historical mode field.
    # All accepted values feed the same VRAM-budget optimizer.
    memory_mode: Literal["auto", "performance", "low_vram"] = "auto"
    # Physical decisions emitted by the two-control joint optimizer.  Tuple
    # storage keeps the frozen request auditable and checkpoint-comparable.
    attention_action_schedule: tuple[tuple[int, int, str], ...] = ()
    attention_online_guard_id: str | None = None
    attention_online_budget_dense_layers: float = 0.0
    attention_online_rebate_schedule: tuple[tuple[int, int], ...] = ()
    acceleration_plan_summary: dict[str, Any] | None = None
    # In-process request-local controller produced after exact tokenisation.
    # Its auditable state is exported through the Forecast profile; it is not
    # part of the public API or prompt-dependent routing surface.
    mechanistic_runtime_controller: Any | None = None
    # When present, the exact-token V19 selector owns both the actual/forecast
    # trajectory and the per-cell Attention schedule.  Selection happens only
    # after Qwen tokenisation and reference preprocessing, so a creator prompt
    # is never routed using the former character-count approximation.
    v19_acceleration: float | None = None
    # Optional Y-trajectory acceleration for the formal tail.  The split is
    # one-based: a value of 18 assigns solver indices 0..17 to the first pass
    # and 18..N-1 to the second pass.
    v19_second_pass_acceleration: float | None = None
    acceleration_transition_step: int | None = None
    # Actual DiT evaluations required by the surrounding product protocol.
    # This is deliberately independent from preview rendering: a formal
    # checkpoint resume must reconstruct the same V19 trajectory even though
    # it must not render the already-produced checkpoint preview again.
    scheduler_required_actual_step_indices: tuple[int, ...] = ()
    first_frame: Path | None = None
    last_frame: Path | None = None
    reference_images: tuple[Path, ...] = ()
    reference_videos: tuple[Path, ...] = ()
    reference_audios: tuple[Path, ...] = ()
    # Service-side reference-media caps.  They only downscale while preserving
    # the full frame and source aspect ratio; ``original`` skips that cap.
    reference_image_resolution: str = "720p"
    reference_video_resolution: str = "360p"
    prepared_reference_images: tuple[Any, ...] = ()
    prepared_reference_videos: tuple[Any, ...] = ()
    prepared_reference_audios: tuple[Any, ...] = ()
    cancel_check: Callable[[], bool] | None = None
    progress_callback: Callable[[dict[str, Any]], None] | None = None
    use_lora: bool = False
    # Optional second-pass refinement.  ``refinement_latents_path`` points to
    # a clean (sigma=0) AV checkpoint produced by this runtime.  The checkpoint
    # is re-noised on the same rectified-flow clock and sampled over only the
    # final ``steps`` solver intervals.  This is deliberately separate from
    # forecast steps: it is a new low-noise trajectory, not an invalid
    # continuation after the first pass has already reached sigma=0.
    refinement_latents_path: Path | None = None
    # Completed pixel video used as the clean low-noise source. This is the
    # product video-repair route and does not require a retained generation
    # latent. Audio is muxed from the source after local visual repair.
    external_refinement_video_path: Path | None = None
    refinement_denoise: float | None = None
    # FaceRefine uses ComfyUI BasicScheduler's discrete simple-tail schedule.
    # Ordinary H3 second sampling keeps the continuous low-noise schedule.
    refinement_schedule_mode: Literal[
        "continuous_tail", "comfy_simple_tail"
    ] = "continuous_tail"
    # Each tuple is (x, y, width, height, pixel-frame multipliers), with the
    # spatial values normalized to the Atlas canvas. Native H3 never forwards
    # this mask as a model timestep condition; it is sampler-only, matching the
    # published FaceRefine grid-artifact fix.
    refinement_atlas_denoise_regions: tuple[
        tuple[float, float, float, float, tuple[float, ...]], ...
    ] = ()
    # A later FaceRefine window may consume the preceding refined Atlas tail as
    # a protected causal prefix.
    refinement_handoff_latents_path: Path | None = None
    refinement_handoff_context_frames: int = 0
    refinement_spatial_mode: Literal["strict", "learned_3d"] = "strict"
    refinement_sampler: Literal["sa_solver", "res_multistep"] = "sa_solver"
    preserve_refinement_audio: bool = True
    # H3-native detail regeneration.  The ordinary compatibility path keeps
    # all three controls at their identity values.  A detail-focused path may
    # start from a wider flow interval, constrain only the coarse component of
    # every predicted clean state, and retain H3's newly generated target-grid
    # high frequencies.  This avoids turning the whole result into a sharpened
    # copy of the source while stopping distant geometry from wandering.
    refinement_video_shift: float = 6.0
    refinement_sigma_power: float = 1.0
    refinement_prediction_low_frequency_gain: float = 1.0
    refinement_final_low_frequency_gain: float = 1.0
    refinement_temporal_lowpass: bool = False
    refinement_temporal_outlier_only: bool = False
    refinement_temporal_detail_outlier_strength: float = 0.0
    refinement_cross_step_detail_strength: float = 0.0
    # Full-canvas masked refinement.  The DiT still sees every source token,
    # while only these soft regions may replace the accepted source latent.
    refinement_full_canvas_regions: tuple[
        tuple[float, float, float, float], ...
    ] = ()
    refinement_full_canvas_feather: float = 0.02
    # Optional equal-canvas region-atlas pass.  Selected difficult areas are
    # magnified into a regular H3 canvas after the global refinement, sampled
    # together, then reduced and detail-merged into the accepted trajectory.
    # This is an experiment-only request surface until Human visual review.
    refinement_roi_regions: tuple[tuple[float, float, float, float], ...] = ()
    refinement_roi_auto: bool = False
    refinement_roi_max_regions: int = 4
    refinement_roi_min_side_fraction: float = 0.12
    refinement_roi_max_side_fraction: float = 0.28
    refinement_roi_steps: int = 0
    refinement_roi_denoise: float = 0.075
    refinement_roi_low_frequency_gain: float = 0.08
    refinement_roi_mid_frequency_gain: float = 0.0
    refinement_roi_coarse_scale: float = 0.25
    refinement_roi_blend: float = 0.90
    refinement_roi_temporal_outlier_strength: float = 0.65
    refinement_roi_temporal_filter: Literal[
        "outlier", "motion_gated", "residual_lowpass"
    ] = "outlier"
    refinement_roi_atlas_height: int = 0
    refinement_roi_atlas_width: int = 0
    refinement_roi_atlas_rows: int = 2
    refinement_roi_atlas_columns: int = 3
    refinement_roi_position_mode: Literal["atlas", "source_foveated"] = "atlas"
    refinement_roi_attention_mode: Literal["dense", "scheduled_sparse"] = "dense"
    save_final_latents_path: Path | None = None
    # A clean source latent may also carry the exact CPU Qwen output produced
    # by its first pass.  Second sampling reuses that immutable condition
    # instead of re-running the 32B encoder.  Legacy checkpoints simply miss
    # the optional entry and fall back to the established encode path.
    conditioning_cache_source_path: Path | None = None
    # Private semantic handoff for a true continuation. The engine owns this
    # cache and points it at the immediately preceding window's exact Qwen
    # condition. A short high-noise bootstrap lets that condition establish
    # the carried camera/world before the current local task takes authority.
    # Hard cuts deliberately leave this unset.
    continuation_text_bridge_conditioning_path: Path | None = None
    # Private long-horizon transport.  A clean AV tail from the immediately
    # preceding window is copied into a fresh sigma=1 target and clamped after
    # each solver integration.  Users still submit one prompt and duration.
    continuation_latents_path: Path | None = None
    continuation_context_frames: int = 0
    # Optional exact video prefix inside the broader AV continuation context.
    # Hard cuts use a writable visual preroll while retaining the full audio
    # context.
    continuation_video_prefix_frames: int | None = None
    # Rebase a short prefix from the source's terminal latent onto local zero.
    # This is used only to seed a continuous hidden camera relocation; it is
    # cropped before delivery and never acts as an old-view reference row.
    continuation_video_prefix_from_source_end: bool = False
    # Optional trailing part of the audio context that remains denoisable.
    # It is overlap-added with the preceding clean segment during assembly;
    # video retains the full hard prefix.
    continuation_audio_bridge_ticks: int = 0
    # Private post-trajectory repair for rare harsh long-horizon speech
    # residuals.  It never participates in DiT denoising and therefore cannot
    # perturb the already accepted video trajectory.
    audio_manifold_guard: bool = False
    # Fixed-budget clean AV latent coreset from earlier completed windows.
    # The DiT projects it into read-only reference-style packed tokens; it is
    # request-internal and never substitutes a textual/LLM summary.
    av_token_memory_path: Path | None = None
    # Training-free global temporal co-denoising.  ``frames`` continues to
    # describe the largest physical DiT workload used by the memory planner;
    # this field owns the one global AV solver state and final decode clock.
    # Every local window is a view of that state at the same timestep, never a
    # separately completed clip.
    global_co_denoise_output_frames: int | None = None
    global_co_denoise_window_frames: int = GLOBAL_AV_WINDOW_FRAMES
    global_co_denoise_stride_frames: int = GLOBAL_AV_WINDOW_STRIDE_FRAMES
    global_co_denoise_prompts: tuple[str, ...] = ()
    global_co_denoise_conditioning_paths: tuple[Path, ...] = ()
    # Contiguous creator-window frame ownership. When present, local DiT
    # views may read across a boundary but only write into their prompt's
    # interval.
    global_co_denoise_prompt_ranges: tuple[tuple[int, int], ...] = ()
    # Single-video SelfLift carries completed audio, so its final visual tail
    # can use the finer video lattice and distribute work evenly.
    global_co_denoise_balanced_windows: bool = False
    # ``window_local`` keeps every localized text/target relationship inside
    # H3's native RoPE range.  Global chronology is owned by the tensor views,
    # not by adding an out-of-distribution offset after a variable-length text
    # prefix.  ``absolute_target`` is retained only for controlled ablations.
    global_co_denoise_rotary_mode: Literal[
        "window_local", "absolute_target"
    ] = "window_local"
    # Private long-video SelfLift finalization.  The source is one already
    # connected, clean low-resolution AV x0 timeline assembled from the
    # retained window forks.  It is lifted once as a whole and the remaining
    # formal solver steps are evaluated through global overlapping views.
    # This differs from ordinary second sampling: the original sigma schedule
    # and global step indexes are preserved instead of starting a new tail.
    global_selflift_source_path: Path | None = None
    # Request-local scale for the formal high-resolution Sigma tail. One keeps
    # the calibrated trajectory; lower values constrain second-pass redraw.
    global_selflift_sigma_scale: float = 1.0
    # Internal UltimateUpscale execution contract.  Temporal windows do not
    # individually satisfy the public 17*n+5 clip rule (for example the
    # upstream 136-frame window owns exactly 40 H3 video tokens), so the
    # orchestrator supplies their exact latent clocks.  ``latent_only`` stops
    # before VAE decode; the stitched full latent is decoded once.
    internal_video_tokens: int | None = None
    internal_audio_tokens: int | None = None
    latent_only: bool = False
    retain_transformer_after_latent_only: bool = False
    # Resume a formally paused noisy sampler state without replaying its
    # prefix.  The resumed request owns a fresh, explicitly rescheduled sigma
    # tail whose length is ``steps``.  This is intended for disposable preview
    # branches, not for pretending that the shortened tail is the formal run.
    sampler_state_path: Path | None = None
    # A product checkpoint resumes the untouched suffix of the original sigma
    # trajectory.  It is intentionally distinct from sampler_state_path,
    # whose tail is rescheduled for disposable research previews.
    formal_resume_state_path: Path | None = None
    checkpoint_after_step: int | None = None
    checkpoint_state_path: Path | None = None
    # Decode one in-trajectory x0 estimate without interrupting the sampler.
    # This supports a creator-facing preview/card-selection experiment while
    # the exact original solver state continues toward the final result.
    preview_step_index: int | None = None
    preview_output_path: Path | None = None
    preview_latents_path: Path | None = None
    # ``direct_x0`` decodes the clean-sample prediction already produced by
    # the selected formal DiT evaluation.  It adds no preview DiT work.
    # ``fast_finish`` retains the research-only disposable solver branch.
    preview_decode_mode: Literal["direct_x0", "fast_finish"] = "direct_x0"
    # Optional comparison branch using only the formal controller's shallow
    # anchor and forecasted tail for N sigma transitions.
    preview_forecast_steps: int = 0
    preview_forecast_output_path: Path | None = None
    preview_branch_steps: int = 2
    preview_branch_actual_step_indices: tuple[int, ...] | None = None
    # Research controls for a more readable early preview.  Spatially reducing
    # only the disposable branch buys several stable solver evaluations for
    # roughly the cost of the former two full-canvas jumps.  The formal latent
    # trajectory always remains at the requested output geometry.
    preview_branch_spatial_scale: float = 1.0
    preview_branch_warm_history: bool = False
    preview_branch_force_dense: bool = False
    preview_branch_use_lora: bool = False
    # Optional audio-only companion branch.  The primary preview branch keeps
    # the requested/full video canvas, while this disposable LoRA branch may
    # use a cheaper video canvas because only its audio latent is retained.
    # This combines the Base branch's more faithful image with the distilled
    # route's substantially more readable early speech without touching the
    # paused formal trajectory.
    preview_audio_branch_use_lora: bool = False
    preview_audio_branch_steps: int = 4
    preview_audio_branch_spatial_scale: float = 0.65
    preview_ready_callback: Callable[[dict[str, Any]], None] | None = None
    preview_decision_wait: Callable[[], str] | None = None
    # Optional in-trajectory spatial transition. Early solver positions run
    # on a smaller canvas; after ``multiscale_resize_after_step`` the exact
    # state and, for Base, the RES clean-prediction history are lifted to the
    # requested output canvas. The missing spatial noise band is introduced
    # at the current sigma.
    multiscale_initial_width: int | None = None
    multiscale_initial_height: int | None = None
    multiscale_resize_after_step: int | None = None
    multiscale_highpass_strength: float = 1.0
    multiscale_transition_mode: Literal[
        "noisy_highpass", "selflift_learned_x0"
    ] = "noisy_highpass"
    # Finish the normal low-resolution trajectory at sigma=0, then lift its
    # clean motion latent to the requested canvas and spend a small number of
    # dense low-noise evaluations on spatial detail.  Unlike the experimental
    # in-trajectory resize above, this starts a mathematically valid new
    # rectified-flow trajectory and never interpolates a noisy solver state.
    terminal_refinement_initial_width: int | None = None
    terminal_refinement_initial_height: int | None = None
    terminal_refinement_steps: int = 0
    terminal_refinement_denoise: float = 0.0125
    terminal_refinement_dense_tail_steps: int = 1
    terminal_refinement_low_frequency_gain: float = 1.0
    terminal_refinement_temporal_lowpass: bool = False
    terminal_refinement_temporal_outlier_only: bool = False

    @property
    def num_frames(self) -> int:
        """Compatibility name consumed by the conditioning adapters."""

        return self.frames

    def validate(self) -> None:
        if not self.prompt.strip():
            raise ValueError("prompt cannot be empty")
        if self.width % 32 or self.height % 32:
            raise ValueError("width and height must be multiples of 32")
        internal_window = self.internal_video_tokens is not None
        if internal_window:
            if self.frames <= 0:
                raise ValueError("internal temporal window must contain frames")
            if self.internal_video_tokens <= 0:
                raise ValueError("internal_video_tokens must be positive")
            if self.internal_audio_tokens is None or self.internal_audio_tokens <= 0:
                raise ValueError("internal_audio_tokens must be positive")
            if not self.latent_only:
                raise ValueError("internal temporal windows must be latent-only")
        elif self.internal_audio_tokens is not None:
            raise ValueError("internal audio clock requires internal video clock")
        elif self.frames < 5 or (self.frames - 5) % 17:
            raise ValueError("frames must satisfy 17*n+5")
        if self.steps <= 0 or self.fps <= 0:
            raise ValueError("steps and fps must be positive")
        global_plan = None
        if self.global_co_denoise_output_frames is not None:
            global_plan = (
                plan_prompt_owned_global_av_windows(
                    self.global_co_denoise_output_frames,
                    self.global_co_denoise_prompt_ranges,
                    window_frames=self.global_co_denoise_window_frames,
                    stride_frames=self.global_co_denoise_stride_frames,
                    balanced=self.global_co_denoise_balanced_windows,
                )
                if self.global_co_denoise_prompt_ranges
                else (
                    plan_balanced_global_av_windows(
                        self.global_co_denoise_output_frames,
                        window_frames=self.global_co_denoise_window_frames,
                        stride_frames=self.global_co_denoise_stride_frames,
                    )
                    if self.global_co_denoise_balanced_windows
                    else plan_global_av_windows(
                        self.global_co_denoise_output_frames,
                        window_frames=self.global_co_denoise_window_frames,
                        stride_frames=self.global_co_denoise_stride_frames,
                    )
                )
            )
            if (
                self.global_co_denoise_output_frames <= self.frames
                and self.global_selflift_source_path is None
            ):
                raise ValueError(
                    "global co-denoise is private to outputs longer than one native request"
                )
            if len(self.global_co_denoise_prompts) != len(global_plan.windows):
                raise ValueError(
                    "global co-denoise requires one deterministic prompt view per window"
                )
            if any(not prompt.strip() for prompt in self.global_co_denoise_prompts):
                raise ValueError("global co-denoise window prompts cannot be empty")
            if self.global_co_denoise_prompts[0] != self.prompt:
                raise ValueError("request prompt must equal the first co-denoise prompt")
            if len(self.global_co_denoise_conditioning_paths) != len(
                global_plan.windows
            ):
                raise ValueError(
                    "global co-denoise requires one conditioning cache per window"
                )
            missing = [
                str(path)
                for path in self.global_co_denoise_conditioning_paths
                if not Path(path).is_file()
            ]
            if missing:
                raise ValueError(
                    f"global co-denoise conditioning cache does not exist: {missing[0]}"
                )
            if self.global_co_denoise_rotary_mode not in (
                "window_local",
                "absolute_target",
            ):
                raise ValueError(
                    "global co-denoise rotary mode must be window_local or "
                    "absolute_target"
                )
            incompatible = (
                self.continuation_latents_path is not None
                or self.continuation_text_bridge_conditioning_path is not None
                or self.refinement_latents_path is not None
                or self.sampler_state_path is not None
                or self.formal_resume_state_path is not None
                or self.checkpoint_after_step is not None
                or self.preview_step_index is not None
                or self.multiscale_resize_after_step is not None
                or self.terminal_refinement_steps > 0
                or self.internal_video_tokens is not None
                or self.first_frame is not None
                or self.last_frame is not None
                or bool(self.reference_videos)
            )
            if incompatible:
                raise ValueError(
                    "global co-denoise v1 supports direct T2AV/Ref-image/Ref-audio "
                    "generation only; continuation, resume, preview, spatial "
                    "transitions, FL2VA keyframes and reference video are separate routes"
                )
            if self.global_selflift_source_path is not None:
                source = Path(self.global_selflift_source_path)
                if not source.is_file():
                    raise ValueError(
                        "global SelfLift source does not exist: " f"{source}"
                    )
                if self.use_lora and self.steps < 2:
                    raise ValueError(
                        "global SelfLift must leave at least one formal tail step"
                    )
                if (
                    not math.isfinite(float(self.global_selflift_sigma_scale))
                    or not 0.25 <= float(self.global_selflift_sigma_scale) <= 1.0
                ):
                    raise ValueError(
                        "global SelfLift sigma scale must be between 0.25 and 1"
                    )
        elif self.global_selflift_source_path is not None:
            raise ValueError(
                "global SelfLift source requires global co-denoise"
            )
        if self.retain_transformer_after_latent_only and not self.latent_only:
            raise ValueError(
                "transformer retention is private to a latent-only window chain"
            )
        if self.memory_mode not in ("auto", "performance", "low_vram"):
            raise ValueError(
                "memory_mode must be auto, performance or low_vram"
            )
        if self.v19_acceleration is not None and (
            not math.isfinite(self.v19_acceleration)
            or not 0.0 <= self.v19_acceleration <= 100.0
        ):
            raise ValueError("V19 acceleration must lie in [0, 100]")
        if self.v19_second_pass_acceleration is not None and (
            not math.isfinite(self.v19_second_pass_acceleration)
            or not 0.0 <= self.v19_second_pass_acceleration <= 100.0
        ):
            raise ValueError("second-pass acceleration must lie in [0, 100]")
        if self.acceleration_transition_step is not None and not (
            1 <= self.acceleration_transition_step <= self.steps
        ):
            raise ValueError(
                "acceleration transition must lie inside the formal trajectory"
            )
        if (
            tuple(sorted(set(self.scheduler_required_actual_step_indices)))
            != self.scheduler_required_actual_step_indices
            or any(
                index < 0 or index >= self.steps
                for index in self.scheduler_required_actual_step_indices
            )
        ):
            raise ValueError(
                "scheduler-required actual steps must be sorted, unique and "
                "inside the sigma schedule"
            )
        if self.preview_step_index is None:
            if self.preview_output_path is not None:
                raise ValueError(
                    "preview output/checkpoint requires preview_step_index"
                )
            latent_only_selflift_fork = bool(
                self.preview_latents_path is not None
                and self.multiscale_transition_mode == "selflift_learned_x0"
                and self.multiscale_resize_after_step is not None
                and self.checkpoint_after_step
                == self.multiscale_resize_after_step + 1
            )
            if self.preview_latents_path is not None and not latent_only_selflift_fork:
                raise ValueError(
                    "preview latent checkpoint requires preview_step_index"
                )
        else:
            if not 0 <= self.preview_step_index < self.steps:
                raise ValueError("preview_step_index falls outside the sigma schedule")
            if (
                self.preview_output_path is None
                and self.preview_latents_path is None
            ):
                raise ValueError(
                    "preview_step_index requires a decoded output or latent output"
                )
            if self.preview_decode_mode not in ("direct_x0", "fast_finish"):
                raise ValueError(
                    "preview_decode_mode must be direct_x0 or fast_finish"
                )
            if not 0 <= self.preview_forecast_steps <= 6:
                raise ValueError("preview_forecast_steps must be between 0 and 6")
            if (
                self.preview_forecast_steps > 0
                and self.preview_forecast_output_path is None
            ):
                raise ValueError(
                    "preview_forecast_steps requires preview_forecast_output_path"
                )
            if not 1 <= self.preview_branch_steps <= 30:
                raise ValueError("preview_branch_steps must be between 1 and 30")
            if self.preview_branch_actual_step_indices is not None:
                indices = self.preview_branch_actual_step_indices
                if (
                    tuple(sorted(set(indices))) != indices
                    or any(
                        value < 0 or value >= self.preview_branch_steps
                        for value in indices
                    )
                ):
                    raise ValueError(
                        "preview branch actual steps must be sorted, unique and inside the branch"
                    )
            # The public fixed 360p checkpoint preview is about 0.3235 of an
            # aligned 1080p canvas (352 / 1088).  The old 0.4 floor made the
            # advertised 24GB 1080p checkpoint path impossible before DiT.
            if not 0.3 <= self.preview_branch_spatial_scale <= 1.0:
                raise ValueError(
                    "preview_branch_spatial_scale must be between 0.3 and 1.0"
                )
            if self.preview_audio_branch_use_lora:
                if not 1 <= self.preview_audio_branch_steps <= 6:
                    raise ValueError(
                        "preview_audio_branch_steps must be between 1 and 6"
                    )
                if not 0.4 <= self.preview_audio_branch_spatial_scale <= 1.0:
                    raise ValueError(
                        "preview_audio_branch_spatial_scale must be between 0.4 and 1.0"
                    )
            if self.preview_decision_wait is not None and self.preview_ready_callback is None:
                raise ValueError("preview decision wait requires a ready callback")
        has_external_refinement = self.external_refinement_video_path is not None
        if has_external_refinement:
            external_path = Path(self.external_refinement_video_path)
            if not external_path.is_file():
                raise ValueError(
                    f"external refinement video does not exist: {external_path}"
                )
        if self.refinement_latents_path is not None and has_external_refinement:
            raise ValueError(
                "refinement latent and external video sources are mutually exclusive"
            )
        if self.refinement_latents_path is None and not has_external_refinement:
            if self.refinement_denoise is not None:
                raise ValueError(
                    "refinement_denoise requires refinement_latents_path"
                )
            if self.refinement_spatial_mode != "strict":
                raise ValueError(
                    "refinement_spatial_mode requires refinement_latents_path"
                )
            if (
                self.refinement_video_shift != 6.0
                or self.refinement_sigma_power != 1.0
                or self.refinement_prediction_low_frequency_gain != 1.0
                or self.refinement_final_low_frequency_gain != 1.0
                or self.refinement_temporal_lowpass
                or self.refinement_temporal_outlier_only
                or self.refinement_temporal_detail_outlier_strength != 0.0
                or self.refinement_cross_step_detail_strength != 0.0
                or self.refinement_full_canvas_regions
                or self.refinement_roi_regions
                or self.refinement_roi_auto
                or self.refinement_roi_steps != 0
            ):
                raise ValueError(
                    "H3 detail-regeneration controls require refinement_latents_path"
                )
        else:
            if (
                self.refinement_latents_path is not None
                and not Path(self.refinement_latents_path).is_file()
            ):
                raise ValueError(
                    "refinement latent checkpoint does not exist: "
                    f"{self.refinement_latents_path}"
                )
            if self.refinement_denoise is None:
                raise ValueError(
                    "refinement_latents_path requires refinement_denoise"
                )
            if not 0.0 < self.refinement_denoise <= 1.0:
                raise ValueError("refinement_denoise must be in (0, 1]")
            if self.refinement_schedule_mode not in (
                "continuous_tail",
                "comfy_simple_tail",
            ):
                raise ValueError(
                    "refinement_schedule_mode must be continuous_tail or comfy_simple_tail"
                )
            if self.refinement_handoff_context_frames < 0:
                raise ValueError("refinement handoff context cannot be negative")
            if self.refinement_handoff_latents_path is None:
                if self.refinement_handoff_context_frames:
                    raise ValueError(
                        "refinement handoff context requires a latent checkpoint"
                    )
            else:
                if not Path(self.refinement_handoff_latents_path).is_file():
                    raise ValueError(
                        "refinement handoff latent checkpoint does not exist: "
                        f"{self.refinement_handoff_latents_path}"
                    )
                if self.refinement_handoff_context_frames < 1:
                    raise ValueError(
                        "refinement handoff latent checkpoint requires context frames"
                    )
            if len(self.refinement_atlas_denoise_regions) > 9:
                raise ValueError("FaceRefine Atlas supports at most nine denoise regions")
            for region in self.refinement_atlas_denoise_regions:
                if len(region) != 5:
                    raise ValueError(
                        "each Atlas denoise region requires x,y,width,height,frame multipliers"
                    )
                x, y, width, height = (float(value) for value in region[:4])
                if not (
                    0.0 <= x < 1.0
                    and 0.0 <= y < 1.0
                    and 0.0 < width <= 1.0 - x + 1.0e-8
                    and 0.0 < height <= 1.0 - y + 1.0e-8
                ):
                    raise ValueError("Atlas denoise region lies outside the canvas")
                strengths = region[4]
                if not strengths or any(
                    not math.isfinite(float(value))
                    or not 0.0 <= float(value) <= 1.0
                    for value in strengths
                ):
                    raise ValueError(
                        "Atlas denoise frame multipliers must lie inside [0,1]"
                    )
            if self.refinement_spatial_mode not in ("strict", "learned_3d"):
                raise ValueError(
                    "refinement_spatial_mode must be strict or learned_3d"
                )
            if self.refinement_sampler not in ("sa_solver", "res_multistep"):
                raise ValueError(
                    "refinement_sampler must be sa_solver or res_multistep"
                )
            if not math.isfinite(self.refinement_video_shift) or self.refinement_video_shift <= 0.0:
                raise ValueError("refinement_video_shift must be finite and positive")
            if (
                not math.isfinite(self.refinement_sigma_power)
                or not 0.5 <= self.refinement_sigma_power <= 3.0
            ):
                raise ValueError(
                    "refinement_sigma_power must lie inside [0.5, 3]"
                )
            for name, value in (
                (
                    "refinement_prediction_low_frequency_gain",
                    self.refinement_prediction_low_frequency_gain,
                ),
                (
                    "refinement_final_low_frequency_gain",
                    self.refinement_final_low_frequency_gain,
                ),
            ):
                if not 0.0 <= value <= 1.0:
                    raise ValueError(f"{name} must lie inside [0, 1]")
            if self.refinement_temporal_outlier_only and not self.refinement_temporal_lowpass:
                raise ValueError(
                    "refinement temporal outlier filtering requires temporal lowpass"
                )
            if not 0.0 <= self.refinement_temporal_detail_outlier_strength <= 1.0:
                raise ValueError(
                    "refinement temporal detail outlier strength must lie inside [0, 1]"
                )
            if not 0.0 <= self.refinement_cross_step_detail_strength <= 1.0:
                raise ValueError(
                    "refinement cross-step detail strength must lie inside [0, 1]"
                )
            if self.refinement_cross_step_detail_strength > 0.0 and self.steps < 2:
                raise ValueError(
                    "refinement cross-step detail confidence requires at least two steps"
                )
            if not 0.0 <= self.refinement_full_canvas_feather <= 0.25:
                raise ValueError(
                    "refinement full-canvas feather must lie inside [0,0.25]"
                )
            if len(self.refinement_full_canvas_regions) > 16:
                raise ValueError(
                    "full-canvas refinement supports at most sixteen regions"
                )
            for region in self.refinement_full_canvas_regions:
                if len(region) != 4:
                    raise ValueError(
                        "each full-canvas refinement region must contain x0,y0,x1,y1"
                    )
                x0, y0, x1, y1 = (float(value) for value in region)
                if not (
                    0.0 <= x0 < x1 <= 1.0
                    and 0.0 <= y0 < y1 <= 1.0
                ):
                    raise ValueError(
                        "full-canvas refinement coordinates must lie inside [0,1]"
                    )
            if self.refinement_full_canvas_regions and (
                self.refinement_roi_regions or self.refinement_roi_auto
            ):
                raise ValueError(
                    "full-canvas masked refinement cannot be combined with ROI atlas"
                )
            if not 0 <= self.refinement_roi_steps <= 8:
                raise ValueError("refinement ROI atlas supports zero to eight steps")
            roi_selector_enabled = bool(
                self.refinement_roi_regions or self.refinement_roi_auto
            )
            if roi_selector_enabled != bool(self.refinement_roi_steps):
                raise ValueError(
                    "refinement ROI selection and ROI steps must be enabled together"
                )
            if self.refinement_roi_regions and self.refinement_roi_auto:
                raise ValueError(
                    "manual and automatic refinement ROI selection are mutually exclusive"
                )
            if len(self.refinement_roi_regions) > 9:
                raise ValueError("refinement ROI atlas supports at most nine regions")
            for region in self.refinement_roi_regions:
                if len(region) != 4:
                    raise ValueError("each refinement ROI must contain x0,y0,x1,y1")
                x0, y0, x1, y1 = (float(value) for value in region)
                if not (
                    0.0 <= x0 < x1 <= 1.0
                    and 0.0 <= y0 < y1 <= 1.0
                ):
                    raise ValueError(
                        "refinement ROI coordinates must lie inside [0,1]"
                    )
            if not 1 <= self.refinement_roi_max_regions <= 9:
                raise ValueError(
                    "automatic refinement ROI selection supports one to nine regions"
                )
            if not (
                0.0
                < self.refinement_roi_min_side_fraction
                <= self.refinement_roi_max_side_fraction
                <= 1.0
            ):
                raise ValueError(
                    "automatic refinement ROI side fractions are invalid"
                )
            if self.refinement_roi_auto and self.steps < 2:
                raise ValueError(
                    "automatic refinement ROI selection requires two global predictions"
                )
            if not 0.0 < self.refinement_roi_denoise <= 0.5:
                raise ValueError("refinement ROI denoise must lie inside (0,0.5]")
            if not 0.0 <= self.refinement_roi_low_frequency_gain <= 1.0:
                raise ValueError(
                    "refinement ROI low-frequency gain must lie inside [0,1]"
                )
            if not 0.0 <= self.refinement_roi_mid_frequency_gain <= 1.0:
                raise ValueError(
                    "refinement ROI mid-frequency gain must lie inside [0,1]"
                )
            if not 0.0 < self.refinement_roi_coarse_scale <= 1.0:
                raise ValueError(
                    "refinement ROI coarse scale must lie inside (0,1]"
                )
            if not 0.0 <= self.refinement_roi_blend <= 1.0:
                raise ValueError("refinement ROI blend must lie inside [0,1]")
            if not (
                0.0
                <= self.refinement_roi_temporal_outlier_strength
                <= 1.0
            ):
                raise ValueError(
                    "refinement ROI temporal outlier strength must lie inside [0,1]"
                )
            if self.refinement_roi_temporal_filter not in (
                "outlier",
                "motion_gated",
                "residual_lowpass",
            ):
                raise ValueError(
                    "refinement ROI temporal filter must be outlier, motion_gated "
                    "or residual_lowpass"
                )
            if bool(self.refinement_roi_atlas_height) != bool(
                self.refinement_roi_atlas_width
            ):
                raise ValueError(
                    "refinement ROI atlas height and width must be enabled together"
                )
            if self.refinement_roi_atlas_height < 0 or self.refinement_roi_atlas_width < 0:
                raise ValueError("refinement ROI atlas dimensions cannot be negative")
            if self.refinement_roi_atlas_rows <= 0 or self.refinement_roi_atlas_columns <= 0:
                raise ValueError("refinement ROI atlas grid must be positive")
            if (
                self.refinement_roi_atlas_rows
                * self.refinement_roi_atlas_columns
                < len(self.refinement_roi_regions)
            ):
                raise ValueError("refinement ROI atlas grid cannot hold all regions")
            if (
                self.refinement_roi_auto
                and self.refinement_roi_atlas_rows
                * self.refinement_roi_atlas_columns
                < self.refinement_roi_max_regions
            ):
                raise ValueError(
                    "refinement ROI atlas grid cannot hold automatic regions"
                )
            if self.refinement_roi_atlas_height and (
                self.refinement_roi_atlas_height // self.refinement_roi_atlas_rows < 2
                or self.refinement_roi_atlas_width
                // self.refinement_roi_atlas_columns
                < 2
            ):
                raise ValueError("refinement ROI atlas cells are too small")
            if self.refinement_roi_position_mode not in (
                "atlas",
                "source_foveated",
            ):
                raise ValueError(
                    "refinement ROI position mode must be atlas or source_foveated"
                )
            if self.refinement_roi_attention_mode not in (
                "dense",
                "scheduled_sparse",
            ):
                raise ValueError(
                    "refinement ROI attention mode must be dense or scheduled_sparse"
                )
            if self.refinement_roi_position_mode == "source_foveated":
                if not self.refinement_roi_atlas_height:
                    raise ValueError(
                        "source-foveated ROI positions require explicit atlas dimensions"
                    )
                if (
                    self.refinement_roi_atlas_rows
                    * self.refinement_roi_atlas_columns
                    != len(self.refinement_roi_regions)
                ):
                    raise ValueError(
                        "source-foveated ROI positions require every atlas cell to be assigned"
                    )
                if (
                    self.refinement_roi_atlas_height % 2
                    or self.refinement_roi_atlas_width % 2
                ):
                    raise ValueError(
                        "source-foveated ROI atlas dimensions must align to H3 patch tokens"
                    )
                cell = min(
                    self.refinement_roi_atlas_height
                    // self.refinement_roi_atlas_rows,
                    self.refinement_roi_atlas_width
                    // self.refinement_roi_atlas_columns,
                )
                used_height = cell * self.refinement_roi_atlas_rows
                used_width = cell * self.refinement_roi_atlas_columns
                margin_y = (
                    self.refinement_roi_atlas_height - used_height
                ) // 2
                margin_x = (
                    self.refinement_roi_atlas_width - used_width
                ) // 2
                if any(value % 2 for value in (cell, margin_y, margin_x)):
                    raise ValueError(
                        "source-foveated ROI atlas cells and margins must align to H3 patch tokens"
                    )
            if self.actual_step_indices is not None and self.actual_step_indices != tuple(
                range(self.steps)
            ):
                # Forecasted x0 estimates obey the same predictor contract as
                # exact DiT estimates, so SA-Solver can integrate them. Keep
                # this capability private to the one bounded experiment that
                # was designed for it: two exact histories before the first
                # forecast, no approximate Attention, and an exact terminal
                # evaluation. Persisted or public requests cannot silently
                # opt into refinement Forecast with arbitrary actual indices.
                summary = self.acceleration_plan_summary
                forecast_profile_allowed = (
                    isinstance(summary, dict)
                    and summary.get("policy_id")
                    == "h3_second_sampling_dense_4of8_v1"
                    and self.steps == 8
                    and self.actual_step_indices == (0, 1, 4, 7)
                    and not self.attention_action_schedule
                    and self.mechanistic_runtime_controller is None
                )
                if not forecast_profile_allowed:
                    raise ValueError(
                        "second-pass refinement Forecast requires the bounded "
                        "dense_4of8_v1 execution profile"
                    )
        if (
            self.conditioning_cache_source_path is not None
            and not Path(self.conditioning_cache_source_path).is_file()
        ):
            raise ValueError(
                "conditioning cache source does not exist: "
                f"{self.conditioning_cache_source_path}"
            )
        if (
            self.continuation_text_bridge_conditioning_path is not None
            and not Path(
                self.continuation_text_bridge_conditioning_path
            ).is_file()
        ):
            raise ValueError(
                "continuation text bridge cache does not exist: "
                f"{self.continuation_text_bridge_conditioning_path}"
            )
        if self.continuation_latents_path is None:
            if self.continuation_context_frames != 0:
                raise ValueError(
                    "continuation_context_frames requires continuation_latents_path"
                )
            if self.continuation_audio_bridge_ticks != 0:
                raise ValueError(
                    "continuation_audio_bridge_ticks requires continuation_latents_path"
                )
            if self.continuation_video_prefix_frames is not None:
                raise ValueError(
                    "continuation_video_prefix_frames requires continuation_latents_path"
                )
            if self.continuation_video_prefix_from_source_end:
                raise ValueError(
                    "terminal video prefix requires continuation_latents_path"
                )
            if self.continuation_text_bridge_conditioning_path is not None:
                raise ValueError(
                    "continuation text bridge requires continuation_latents_path"
                )
        else:
            if not Path(self.continuation_latents_path).is_file():
                raise ValueError(
                    "continuation latent checkpoint does not exist: "
                    f"{self.continuation_latents_path}"
                )
            if (
                self.continuation_context_frames < H3_FRAME_ORIGIN
                or self.continuation_context_frames > MAX_CONTINUATION_CONTEXT_FRAMES
                or (
                    self.continuation_context_frames - H3_FRAME_ORIGIN
                ) % H3_FRAME_STRIDE
                or self.continuation_context_frames >= self.frames
            ):
                raise ValueError(
                    "continuation context must use H3's 5 + 17*k grid, fit the "
                    "target window, and not exceed the calibrated 90-frame bound"
                )
            context_audio_ticks = audio_latent_frames(
                self.continuation_context_frames
            )
            video_prefix_frames = (
                self.continuation_context_frames
                if self.continuation_video_prefix_frames is None
                else self.continuation_video_prefix_frames
            )
            if video_prefix_frames != 0 and (
                video_prefix_frames < H3_FRAME_ORIGIN
                or video_prefix_frames > self.continuation_context_frames
                or (video_prefix_frames - H3_FRAME_ORIGIN) % H3_FRAME_STRIDE
            ):
                raise ValueError(
                    "continuation video prefix must be zero or use H3's "
                    "5 + 17*k grid and fit inside the AV continuation context"
                )
            if (
                self.continuation_video_prefix_from_source_end
                and video_prefix_frames == 0
            ):
                raise ValueError("terminal video prefix must contain at least one token")
            if (
                self.continuation_text_bridge_conditioning_path is not None
                and video_prefix_frames == 0
            ):
                raise ValueError(
                    "continuation text bridge requires a non-empty exact video prefix"
                )
            if not 0 <= self.continuation_audio_bridge_ticks <= context_audio_ticks:
                raise ValueError(
                    "continuation audio bridge must fit inside the AV context"
                )
            if self.refinement_latents_path is not None:
                raise ValueError(
                    "continuation and second-pass refinement are mutually exclusive"
                )
            if self.sampler_state_path is not None or self.formal_resume_state_path is not None:
                raise ValueError("continuation cannot be combined with sampler resume")
            if self.terminal_refinement_steps:
                raise ValueError(
                    "continuation cannot combine with terminal spatial refinement"
                )
            if self.multiscale_resize_after_step is not None:
                selflift_fork_checkpoint = bool(
                    self.multiscale_transition_mode == "selflift_learned_x0"
                    and self.checkpoint_after_step
                    == self.multiscale_resize_after_step + 1
                    and self.preview_latents_path is not None
                    and self.preview_step_index in (
                        None, self.multiscale_resize_after_step
                    )
                )
                selflift_direct_complete = bool(
                    self.multiscale_transition_mode == "selflift_learned_x0"
                    and self.checkpoint_after_step is None
                    and self.preview_step_index is None
                    and self.preview_latents_path is None
                )
                if not (selflift_fork_checkpoint or selflift_direct_complete):
                    raise ValueError(
                        "continuation can use SelfLift only as a complete formal "
                        "trajectory or while checkpointing the exact resolution fork"
                    )
        if self.av_token_memory_path is not None:
            if not Path(self.av_token_memory_path).is_file():
                raise ValueError(
                    "AV token-memory checkpoint does not exist: "
                    f"{self.av_token_memory_path}"
                )
            if self.global_co_denoise_output_frames is not None:
                raise ValueError("AV token memory requires causal continuation")
        if (
            (self.first_frame is not None or self.last_frame is not None)
            and (
                self.reference_images
                or self.reference_videos
                or self.reference_audios
            )
        ):
            raise ValueError(
                "first/last keyframes and Ref2VA media are separate public inputs"
            )
        if self.sampler_state_path is not None:
            if self.refinement_latents_path is not None:
                raise ValueError(
                    "sampler_state_path and refinement_latents_path are mutually exclusive"
                )
            if not Path(self.sampler_state_path).is_file():
                raise ValueError(
                    "sampler state checkpoint does not exist: "
                    f"{self.sampler_state_path}"
                )
        if self.formal_resume_state_path is not None:
            if self.refinement_latents_path is not None or self.sampler_state_path is not None:
                raise ValueError(
                    "formal resume cannot be combined with refinement or preview resume"
                )
            if not Path(self.formal_resume_state_path).is_file():
                raise ValueError(
                    "formal resume checkpoint does not exist: "
                    f"{self.formal_resume_state_path}"
                )
        if self.checkpoint_after_step is None:
            if self.checkpoint_state_path is not None:
                raise ValueError(
                    "checkpoint_state_path requires checkpoint_after_step"
                )
        else:
            if not 1 <= self.checkpoint_after_step < self.steps:
                raise ValueError(
                    "checkpoint_after_step must be before the final solver step"
                )
            if self.checkpoint_state_path is None:
                raise ValueError(
                    "checkpoint_after_step requires checkpoint_state_path"
                )
            if self.formal_resume_state_path is not None:
                raise ValueError("a resumed request cannot create the same breakpoint again")
        if self.mlp_chunk_tokens is not None and self.mlp_chunk_tokens <= 0:
            raise ValueError("mlp_chunk_tokens must be positive when provided")
        if self.execution_plan is not None and self.mlp_chunk_tokens is not None:
            raise ValueError(
                "execution_plan owns MLP chunking; do not also set mlp_chunk_tokens"
            )
        if self.actual_step_indices is not None:
            if not self.actual_step_indices:
                raise ValueError("actual_step_indices cannot be empty")
            if tuple(sorted(set(self.actual_step_indices))) != self.actual_step_indices:
                raise ValueError("actual_step_indices must be sorted and unique")
            if any(index < 0 or index >= self.steps for index in self.actual_step_indices):
                raise ValueError("actual step index falls outside the requested steps")
        if self.attention_action_schedule:
            if tuple(sorted(set(self.attention_action_schedule))) != self.attention_action_schedule:
                raise ValueError(
                    "attention action schedule must be sorted and contain unique cells"
                )
            actual = (
                frozenset(range(self.steps))
                if self.actual_step_indices is None
                else frozenset(self.actual_step_indices)
            )
            forecast = frozenset(range(self.steps)) - actual
            valid_actions = {
                "dense", "sparse_topk_0.5", "sparse_topk_0.25",
                "sparse_topk_0.1", "sparse_topk_0.0625",
                "round215:sparse_topk_0.5",
                "round215:sparse_topk_0.25",
                "round215:sparse_topk_0.1",
                "round215:sparse_topk_0.0625",
                "frontier:sparse_topk_0.5",
                "frontier:sparse_topk_0.25",
                "frontier:sparse_topk_0.1",
                "frontier:sparse_topk_0.0625",
                "fastfrontier:sparse_topk_0.5",
                "fastfrontier:sparse_topk_0.25",
                "fastfrontier:sparse_topk_0.1",
                "fastfrontier:sparse_topk_0.0625",
                "forecastfrontier:sparse_topk_0.5",
                "forecastfrontier:sparse_topk_0.25",
                "forecastfrontier:sparse_topk_0.1",
                "forecastfrontier:sparse_topk_0.0625",
            }
            for step, layer, action in self.attention_action_schedule:
                forecast_anchor = (
                    step in forecast
                    and layer < 3
                    and action == "forecastfrontier:sparse_topk_0.0625"
                )
                if (
                    not 0 <= layer < 50
                    or (step not in actual and not forecast_anchor)
                ):
                    raise ValueError(
                        "attention action cell must target an actual H3 layer or "
                        "a certified forecast anchor layer"
                    )
                if action not in valid_actions:
                    raise ValueError(f"unknown attention action: {action}")
        if self.attention_online_guard_id is None:
            if self.attention_online_budget_dense_layers != 0.0:
                raise ValueError("online Attention budget requires a guard id")
            if self.attention_online_rebate_schedule:
                raise ValueError("online Attention rebate requires a guard id")
        elif (
            not math.isfinite(self.attention_online_budget_dense_layers)
            or self.attention_online_budget_dense_layers <= 0.0
        ):
            raise ValueError("online Attention guard requires a positive finite budget")
        if self.attention_online_rebate_schedule:
            if (
                tuple(sorted(set(self.attention_online_rebate_schedule)))
                != self.attention_online_rebate_schedule
            ):
                raise ValueError(
                    "online Attention rebate schedule must be sorted and unique"
                )
            actual = (
                frozenset(range(self.steps))
                if self.actual_step_indices is None
                else frozenset(self.actual_step_indices)
            )
            if any(
                step not in actual or not 0 <= layer < 50
                for step, layer in self.attention_online_rebate_schedule
            ):
                raise ValueError(
                    "online Attention rebate must target actual H3 cells"
                )
        multiscale_values = (
            self.multiscale_initial_width,
            self.multiscale_initial_height,
            self.multiscale_resize_after_step,
        )
        if any(value is not None for value in multiscale_values):
            if not all(value is not None for value in multiscale_values):
                raise ValueError("multiscale transition requires width, height and step")
            assert self.multiscale_initial_width is not None
            assert self.multiscale_initial_height is not None
            assert self.multiscale_resize_after_step is not None
            if self.multiscale_initial_width % 32 or self.multiscale_initial_height % 32:
                raise ValueError("multiscale initial canvas must be divisible by 32")
            if self.multiscale_initial_width > self.width or self.multiscale_initial_height > self.height:
                raise ValueError("multiscale initial canvas cannot exceed output canvas")
            if not 0 <= self.multiscale_resize_after_step < self.steps - 1:
                raise ValueError("multiscale transition must leave at least one solver step")
            if not 0.0 <= self.multiscale_highpass_strength <= 1.0:
                raise ValueError("multiscale highpass strength must lie inside [0, 1]")
            if self.multiscale_transition_mode not in (
                "noisy_highpass", "selflift_learned_x0"
            ):
                raise ValueError("unsupported multiscale transition mode")
            actual = (
                set(range(self.steps))
                if self.actual_step_indices is None
                else set(self.actual_step_indices)
            )
            if any(
                index not in actual
                for index in range(self.multiscale_resize_after_step, self.steps)
            ):
                raise ValueError(
                    "the SelfLift boundary and all post-transition solver steps "
                    "must be actual"
                )
            # First/last-frame anchors follow the generated canvas. The
            # runtime encodes them once at the target geometry, presents a
            # spatially reduced latent on the source grid, then restores the
            # target latent and rebuilds the packed layout at the learned-lift
            # boundary. Ref2VA references retain their independent geometry.
        terminal_values = (
            self.terminal_refinement_initial_width,
            self.terminal_refinement_initial_height,
        )
        if any(value is not None for value in terminal_values) or self.terminal_refinement_steps:
            if not all(value is not None for value in terminal_values):
                raise ValueError(
                    "terminal refinement requires initial width and height"
                )
            assert self.terminal_refinement_initial_width is not None
            assert self.terminal_refinement_initial_height is not None
            if self.terminal_refinement_steps <= 0:
                raise ValueError("terminal refinement steps must be positive")
            if self.terminal_refinement_steps > 3:
                raise ValueError("terminal refinement supports at most three steps")
            if not 1 <= self.terminal_refinement_dense_tail_steps <= self.terminal_refinement_steps:
                raise ValueError(
                    "terminal refinement dense tail must cover between one and all steps"
                )
            if not 0.0 < self.terminal_refinement_denoise <= 1.0:
                raise ValueError("terminal refinement denoise must be in (0, 1]")
            if not 0.0 <= self.terminal_refinement_low_frequency_gain <= 1.0:
                raise ValueError(
                    "terminal refinement low-frequency gain must lie inside [0, 1]"
                )
            if (
                self.terminal_refinement_temporal_outlier_only
                and not self.terminal_refinement_temporal_lowpass
            ):
                raise ValueError(
                    "terminal temporal outlier filtering requires temporal lowpass"
                )
            if (
                self.terminal_refinement_initial_width % 32
                or self.terminal_refinement_initial_height % 32
            ):
                raise ValueError(
                    "terminal refinement initial canvas must be divisible by 32"
                )
            if (
                self.terminal_refinement_initial_width > self.width
                or self.terminal_refinement_initial_height > self.height
            ):
                raise ValueError(
                    "terminal refinement initial canvas cannot exceed output canvas"
                )
            if self.refinement_latents_path is not None:
                raise ValueError(
                    "terminal refinement cannot be combined with checkpoint refinement"
                )
            if any(value is not None for value in multiscale_values):
                raise ValueError(
                    "terminal refinement cannot be combined with an in-trajectory resize"
                )
            if (
                self.first_frame is not None
                or self.last_frame is not None
                or self.reference_images
                or self.reference_videos
            ):
                raise ValueError("terminal refinement is currently T2AV-only")
        elif (
            self.terminal_refinement_low_frequency_gain != 1.0
            or self.terminal_refinement_temporal_lowpass
            or self.terminal_refinement_temporal_outlier_only
        ):
            raise ValueError(
                "terminal refinement low-frequency gain requires terminal refinement"
            )
        for role, path in (
            ("first_frame", self.first_frame),
            ("last_frame", self.last_frame),
        ):
            if path is not None and not Path(path).is_file():
                raise ValueError(f"{role} does not exist: {path}")
        if len(self.reference_images) > 9:
            raise ValueError("Ref2VA accepts at most 9 reference images")
        for path in self.reference_images:
            if not Path(path).is_file():
                raise ValueError(f"reference image does not exist: {path}")
        if len(self.reference_videos) > 3:
            raise ValueError("Ref2VA accepts at most 3 reference videos")
        if len(self.reference_images) + len(self.reference_videos) > 12:
            raise ValueError("Ref2VA accepts at most 12 total reference files")
        for path in self.reference_videos:
            if not Path(path).is_file():
                raise ValueError(f"reference video does not exist: {path}")
        if len(self.reference_audios) > 3:
            raise ValueError("Ref2VA accepts at most 3 reference audios")
        for path in self.reference_audios:
            if not Path(path).is_file():
                raise ValueError(f"reference audio does not exist: {path}")


@dataclass(frozen=True, slots=True)
class HotSessionResult:
    output_path: Path
    total_seconds: float
    phases: dict[str, float]
    step_seconds: tuple[float, ...]
    forecast_profile: dict[str, Any]
    execution_profile: dict[str, Any]
    peak_allocated_gib: float = 0.0
    peak_reserved_gib: float = 0.0


@dataclass(frozen=True, slots=True)
class HotSessionCheckpointResult:
    checkpoint_path: Path | None
    preview_path: Path | None
    completed_steps: int
    total_steps: int
    total_seconds: float
    phases: dict[str, float]
    step_seconds: tuple[float, ...]
    execution_profile: dict[str, Any]
    peak_allocated_gib: float = 0.0
    peak_reserved_gib: float = 0.0
    preview_latents_path: Path | None = None
    token_memory_path: Path | None = None


@dataclass(frozen=True, slots=True)
class _ReferenceLatentCacheEntry:
    key: tuple[Any, ...]
    video_latents: tuple[torch.Tensor, ...]
    video_shapes: tuple[tuple[int, int, int], ...]
    video_kinds: tuple[str, ...]
    audio_latents: tuple[torch.Tensor, ...]
    audio_frames: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class _ConditioningComposition:
    """One explicit merge boundary for user inputs and internal memory."""

    video_latents: tuple[torch.Tensor, ...]
    reference_shapes: tuple[tuple[int, int, int], ...]
    reference_kinds: tuple[str, ...]
    audio_latents: tuple[torch.Tensor, ...]
    reference_audio_frames: tuple[int, ...]
    profile: dict[str, Any]


# Public visual references ground the opening window.  Once clean generated
# state exists, continuation reuses that instantiated state and keeps the
# public images in Qwen's semantic context instead of re-presenting their VAE
# rows as new visual instances on every window.
_AUTHORITATIVE_REFERENCE_BOOTSTRAP_FRACTION = 0.0
# A same-scene hard cut needs the old wide frame only while global geometry is
# formed.  Three of seven Turbo steps is long enough to seed room topology and
# short enough to leave current prop/actor state in control of convergence.
_LAYOUT_MEMORY_BOOTSTRAP_FRACTION = 3.0 / 7.0
# A camera that has never appeared must first form its target projection from
# text.  One isolated canonical-scene observation at the third Turbo update
# can then supply material/layout identity, after which text-only convergence
# removes the old observation before the large terminal solver jump.
_NOVEL_CAMERA_LAYOUT_PROBE_START_STEP = 2
_NOVEL_CAMERA_LAYOUT_PROBE_STEPS = 1
# Whole-image layout references also contain obsolete actor and prop geometry.
# Give the immediately preceding state two coarse, highest-noise updates before
# an explicit camera recurrence is introduced.  One step (V20/V21) preserved
# the moved cup but still let the old reference restore the owner's former
# counter position.  The scheduler always reserves at least one later state
# update, and automatically contracts this prefix for schedules below four
# steps.
_EXPLICIT_CAMERA_STATE_SEED_STEPS = 2
# An old camera row and a recent state row are mutually incomplete
# observations.  Keep the state-conditioned prediction as the only primary
# trajectory and add a bounded camera-conditioned velocity estimate across the
# whole writable window.  Unlike the rejected V14 continuation bridge, this
# weight has no temporal support edge inside the clip and both estimates use
# the same localized text condition.
_EXPLICIT_CAMERA_LAYOUT_GUIDANCE_WEIGHT = 0.5
# Generated voice memory is weaker evidence than a user-supplied reference
# audio.  Let a native short excerpt establish coarse speaker identity only
# while the trajectory is still noise-dominated, then remove the excerpt so
# the localized dialogue and current latent state own phonetics, loudness and
# fine waveform convergence.  This schedule is prompt-content independent.
_INFERRED_VOICE_BOOTSTRAP_FRACTION = 1.0 / 3.0
# The boundary-only auxiliary produces an estimate on every solver step but may
# alter only a short temporal handoff band. It owns the hidden repaint tokens, then
# transfers authority to the current prompt over ten visible tokens with a
# zero-ended linear ramp. V8's first-step-only blend moved a cut to the end of
# its band; V11-V13 showed that current semantics alone can also cut inside a
# published repaint. Persistent, local velocity authority addresses both
# failure modes while leaving the rest of the writable window untouched. Its
# block stack must mirror the primary Actual/Forecast schedule; running an
# unconditional exact auxiliary on Forecast steps defeats V24 acceleration.
_CONTINUATION_TEXT_BRIDGE_FRACTION = 1.0
_CONTINUATION_BOUNDARY_BLEND_PEAK = 1.0
_CONTINUATION_BOUNDARY_VISIBLE_FADE_TOKENS = 10


def _continuation_text_route_name(step_index: int, steps: int) -> str:
    """Schedule the boundary auxiliary while current semantics stay active."""

    total = int(steps)
    index = int(step_index)
    if total <= 0 or not 0 <= index < total:
        raise ValueError("continuation text route step is outside the denoise schedule")
    bridge_steps = int(math.ceil(total * _CONTINUATION_TEXT_BRIDGE_FRACTION))
    return "boundary" if index < bridge_steps else "current"


def _continuation_boundary_forecast_controller(
    actual_steps: tuple[int, ...],
    steps: int,
) -> DirectionalForecastController | None:
    """Mirror the primary trajectory for the boundary-only text estimate.

    The bridge uses a different text context and therefore cannot share the
    primary controller's history. A separate controller preserves one bridge
    estimate per solver step without silently turning every V24 Forecast step
    into another complete 50-block DiT evaluation.
    """

    total = int(steps)
    actual = tuple(int(index) for index in actual_steps)
    if total <= 0 or tuple(sorted(set(actual))) != actual:
        raise ValueError("continuation boundary schedule is invalid")
    bridge_steps = int(math.ceil(total * _CONTINUATION_TEXT_BRIDGE_FRACTION))
    bridge_actual = tuple(index for index in actual if index < bridge_steps)
    if len(bridge_actual) == bridge_steps:
        return None
    return DirectionalForecastController(
        actual_steps=bridge_actual,
        segment_cache=None,
    )


def _blend_continuation_boundary_velocity(
    current: torch.Tensor,
    boundary: torch.Tensor,
    *,
    protected_prefix_tokens: int,
    hidden_repaint_tokens: int,
    visible_fade_tokens: int = _CONTINUATION_BOUNDARY_VISIBLE_FADE_TOKENS,
    peak: float = _CONTINUATION_BOUNDARY_BLEND_PEAK,
) -> torch.Tensor:
    """Isolate repaint velocity, then fade to current-window authority.

    H3 text conditioning is global: replacing the current prompt at high noise
    commits the complete suffix to the old scene. Here the boundary condition
    has full authority only on same-time hidden repaint tokens. A linear fade
    across the first visible tokens ends at exactly zero, so no discrete token
    boundary loses continuity authority and all later tokens remain bitwise
    equal to the current-prompt prediction.
    """

    if current.ndim != 5 or boundary.shape != current.shape:
        raise ValueError("continuation velocity blend requires matching 5D video tensors")
    prefix = int(protected_prefix_tokens)
    hidden = int(hidden_repaint_tokens)
    visible = int(visible_fade_tokens)
    if prefix < 0 or prefix >= current.shape[2]:
        raise ValueError("continuation velocity blend prefix must leave writable tokens")
    if hidden < 0 or visible <= 0:
        raise ValueError("continuation velocity blend bands must be non-negative and positive")
    if not 0.0 < float(peak) <= 1.0:
        raise ValueError("continuation velocity blend peak must lie inside (0, 1]")
    writable = int(current.shape[2]) - prefix
    hidden = min(hidden, writable)
    visible = min(visible, writable - hidden)
    if not hidden and not visible:
        raise ValueError("continuation velocity blend has no writable handoff tokens")
    plateau = torch.full(
        (hidden,), float(peak), device=current.device, dtype=current.dtype
    )
    fade = (
        torch.linspace(
            float(peak), 0.0, visible + 1,
            device=current.device, dtype=current.dtype,
        )[1:]
        if visible else
        torch.empty((0,), device=current.device, dtype=current.dtype)
    )
    weights = torch.cat((plateau, fade)).view(1, 1, -1, 1, 1)
    count = int(weights.shape[2])
    blended = current.clone()
    target = blended[:, :, prefix : prefix + count]
    target.mul_(1.0 - weights).add_(
        boundary[:, :, prefix : prefix + count] * weights
    )
    return blended


def _compose_conditioning(
    *,
    user_video_latents: tuple[torch.Tensor, ...],
    user_reference_shapes: tuple[tuple[int, int, int], ...],
    user_reference_kinds: tuple[str, ...],
    user_audio_latents: tuple[torch.Tensor, ...],
    user_audio_frames: tuple[int, ...],
    memory_video_latents: tuple[torch.Tensor, ...],
    memory_reference_shapes: tuple[tuple[int, int, int], ...],
    memory_reference_kinds: tuple[str, ...],
    memory_audio_latents: tuple[torch.Tensor, ...],
    memory_audio_frames: tuple[int, ...],
    keyframe_latents: tuple[torch.Tensor, ...],
    keyframe_indices: tuple[int, ...],
    user_references_requested: bool,
) -> _ConditioningComposition:
    """Compose conditions without letting bounded memory replace user media.

    The ordering is a model contract shared with ``build_hybrid_condition_layout``:
    public reference media first, bounded reference-style memory second, and
    FL2VA endpoint keyframes last.  Audio follows the same user-before-memory
    priority.  Keeping this as a pure function makes the non-overwrite
    invariant testable without loading model weights.
    """

    if not (
        len(user_video_latents)
        == len(user_reference_shapes)
        == len(user_reference_kinds)
    ):
        raise ValueError("user video conditions must have aligned latent metadata")
    if not (
        len(memory_video_latents)
        == len(memory_reference_shapes)
        == len(memory_reference_kinds)
    ):
        raise ValueError("memory video conditions must have aligned latent metadata")
    if len(user_audio_latents) != len(user_audio_frames):
        raise ValueError("user audio conditions must have aligned latent metadata")
    if len(memory_audio_latents) != len(memory_audio_frames):
        raise ValueError("memory audio conditions must have aligned latent metadata")
    if len(keyframe_latents) != len(keyframe_indices):
        raise ValueError("keyframe conditions must have aligned semantic roles")

    reference_shapes = user_reference_shapes + memory_reference_shapes
    reference_audio_frames = user_audio_frames + memory_audio_frames
    video_latents = (
        user_video_latents + memory_video_latents + keyframe_latents
    )
    audio_latents = user_audio_latents + memory_audio_latents
    if keyframe_indices and (reference_shapes or reference_audio_frames):
        mode = "hybrid_reference_keyframe"
    elif reference_shapes or reference_audio_frames:
        mode = "reference_plus_memory"
    elif keyframe_indices:
        mode = "keyframe"
    elif not user_references_requested:
        # A silent continuation can legitimately release every inferred
        # reference. Exact AV history remains in the separate masked prefix,
        # so no reference rows is a valid route, not a missing user condition.
        mode = "no_reference_rows"
    else:
        raise ValueError("conditioning composition cannot be empty")
    return _ConditioningComposition(
        video_latents=video_latents,
        reference_shapes=reference_shapes,
        reference_kinds=user_reference_kinds + memory_reference_kinds,
        audio_latents=audio_latents,
        reference_audio_frames=reference_audio_frames,
        profile={
            "mode": mode,
            "user_video_references": len(user_video_latents),
            "user_audio_references": len(user_audio_latents),
            "memory_video_references": len(memory_video_latents),
            "memory_audio_references": len(memory_audio_latents),
            "keyframe_roles": list(keyframe_indices),
            "user_conditions_retained": bool(
                not user_references_requested
                or len(user_video_latents) + len(user_audio_latents) > 0
            ),
        },
    )


def _compose_authority_routed_conditioning(
    *,
    user_video_latents: tuple[torch.Tensor, ...],
    user_reference_shapes: tuple[tuple[int, int, int], ...],
    user_reference_kinds: tuple[str, ...],
    user_audio_latents: tuple[torch.Tensor, ...],
    user_audio_frames: tuple[int, ...],
    memory_video_latents: tuple[torch.Tensor, ...],
    memory_reference_shapes: tuple[tuple[int, int, int], ...],
    memory_reference_kinds: tuple[str, ...],
    memory_audio_latents: tuple[torch.Tensor, ...],
    memory_audio_frames: tuple[int, ...],
    keyframe_latents: tuple[torch.Tensor, ...],
    keyframe_indices: tuple[int, ...],
    user_references_requested: bool,
    supports_persistent_inferred_audio: bool = False,
    memory_layout_video_latents: tuple[torch.Tensor, ...] = (),
    memory_layout_reference_shapes: tuple[tuple[int, int, int], ...] = (),
    memory_layout_reference_kinds: tuple[str, ...] = (),
    state_seeded_layout_memory: bool = False,
    novel_camera_layout_probe: bool = False,
) -> tuple[dict[str, _ConditioningComposition], dict[str, Any]]:
    """Plan mutually exclusive authority routes for inferred and public memory.

    Public reference media is authoritative evidence; bounded memory is an
    inferred state carrier.  Presenting both as independent reference items in
    the same DiT pass can turn two observations of one subject or voice into
    two subjects or two utterances.  Audio therefore uses public reference
    latents instead of inferred prototypes whenever they exist.  Visual
    continuation first uses public references to establish canonical identity,
    then uses the already instantiated history state through convergence;
    both are never present in one forward.  This order matters for H3's
    shifted schedule, whose final solver jump is structurally large rather
    than a texture-only refinement.  Qwen's multimodal context remains
    unchanged across both routes.
    """

    if not (
        len(memory_layout_video_latents)
        == len(memory_layout_reference_shapes)
        == len(memory_layout_reference_kinds)
    ):
        raise ValueError("layout memory conditions must have aligned latent metadata")

    authoritative_audio = bool(user_audio_latents)
    persistent_inferred_voice = bool(
        memory_audio_latents
        and not authoritative_audio
        and supports_persistent_inferred_audio
    )
    inferred_voice_bootstrap = bool(
        memory_audio_latents
        and not authoritative_audio
        and not persistent_inferred_voice
    )

    def compose(
        selected_user_video: tuple[torch.Tensor, ...],
        selected_user_shapes: tuple[tuple[int, int, int], ...],
        selected_user_kinds: tuple[str, ...],
        selected_memory_video: tuple[torch.Tensor, ...],
        selected_memory_shapes: tuple[tuple[int, int, int], ...],
        selected_memory_kinds: tuple[str, ...],
        *,
        include_inferred_audio: bool,
    ) -> _ConditioningComposition:
        selected_memory_audio_latents = (
            memory_audio_latents
            if include_inferred_audio and not authoritative_audio
            else ()
        )
        selected_memory_audio_frames = (
            memory_audio_frames
            if include_inferred_audio and not authoritative_audio
            else ()
        )
        return _compose_conditioning(
            user_video_latents=selected_user_video,
            user_reference_shapes=selected_user_shapes,
            user_reference_kinds=selected_user_kinds,
            user_audio_latents=user_audio_latents,
            user_audio_frames=user_audio_frames,
            memory_video_latents=selected_memory_video,
            memory_reference_shapes=selected_memory_shapes,
            memory_reference_kinds=selected_memory_kinds,
            memory_audio_latents=selected_memory_audio_latents,
            memory_audio_frames=selected_memory_audio_frames,
            keyframe_latents=keyframe_latents,
            keyframe_indices=keyframe_indices,
            user_references_requested=user_references_requested,
        )

    progressive_layout_state = bool(
        memory_layout_video_latents
        and (memory_video_latents or novel_camera_layout_probe)
    )
    state_seeded_layout_state = bool(
        state_seeded_layout_memory
        and progressive_layout_state
        and not user_video_latents
    )
    if state_seeded_layout_memory and not state_seeded_layout_state:
        raise ValueError(
            "state-seeded layout memory requires separate layout and recent-state "
            "memory routes without public visual references"
        )
    visual_schedule = bool(
        (
            memory_video_latents
            and (user_video_latents or memory_layout_video_latents)
        )
        or (novel_camera_layout_probe and memory_layout_video_latents)
    )
    if visual_schedule:
        # Public references remain the strongest coarse identity authority.
        # Without one, an internally selected canonical frame supplies only
        # the early coarse-layout route. The late route contains recent state
        # frames exclusively, so stale prop placements cannot compete through
        # convergence.
        visual_variants = {
            "history": (
                (), (), (),
                memory_video_latents,
                memory_reference_shapes,
                memory_reference_kinds,
            ),
            "reference": (
                user_video_latents,
                user_reference_shapes,
                user_reference_kinds,
                (
                    ()
                    if user_video_latents
                    else memory_layout_video_latents
                ),
                (
                    ()
                    if user_video_latents
                    else memory_layout_reference_shapes
                ),
                (
                    ()
                    if user_video_latents
                    else memory_layout_reference_kinds
                ),
            ),
        }
    else:
        visual_variants = {
            "default": (
                user_video_latents,
                user_reference_shapes,
                user_reference_kinds,
                memory_layout_video_latents + memory_video_latents,
                memory_layout_reference_shapes + memory_reference_shapes,
                memory_layout_reference_kinds + memory_reference_kinds,
            )
        }
    routes: dict[str, _ConditioningComposition] = {}
    for visual_name, visual_arguments in visual_variants.items():
        if inferred_voice_bootstrap:
            routes[f"{visual_name}_voice"] = compose(
                *visual_arguments,
                include_inferred_audio=True,
            )
            routes[f"{visual_name}_release"] = compose(
                *visual_arguments,
                include_inferred_audio=False,
            )
        else:
            routes[visual_name] = compose(
                *visual_arguments,
                include_inferred_audio=persistent_inferred_voice,
            )
    return routes, {
        "policy": "single_authority_progressive_routing_v1",
        "visual_policy": (
            "state_primary_layout_velocity_guidance"
            if state_seeded_layout_state
            else "target_camera_seed_layout_probe_text_convergence"
            if novel_camera_layout_probe and progressive_layout_state
            else
            "layout_bootstrap_state_convergence"
            if progressive_layout_state and not user_video_latents
            else
            "reference_bootstrap_history_convergence"
            if visual_schedule
            else "single_route"
        ),
        "reference_bootstrap_fraction": (
            (
                None
                if novel_camera_layout_probe else
                _LAYOUT_MEMORY_BOOTSTRAP_FRACTION
                if progressive_layout_state and not user_video_latents
                else _AUTHORITATIVE_REFERENCE_BOOTSTRAP_FRACTION
            )
            if visual_schedule
            else None
        ),
        "audio_policy": (
            "public_reference_supersedes_inferred_prototypes"
            if authoritative_audio
            else "ref2va_persistent_self_anchor_voice_v1"
            if persistent_inferred_voice
            else "inferred_short_voice_high_noise_bootstrap_v1"
            if inferred_voice_bootstrap
            else "none"
        ),
        "inferred_voice_bootstrap_fraction": (
            _INFERRED_VOICE_BOOTSTRAP_FRACTION
            if inferred_voice_bootstrap
            else None
        ),
        "inferred_voice_release_fraction": (
            1.0 - _INFERRED_VOICE_BOOTSTRAP_FRACTION
            if inferred_voice_bootstrap
            else None
        ),
        "suppressed_memory_audio_references": (
            len(memory_audio_latents) if authoritative_audio else 0
        ),
        "simultaneous_user_and_memory_visual_references": False,
        "progressive_layout_memory_references": len(
            memory_layout_video_latents
        ),
        "progressive_state_memory_references": len(memory_video_latents),
        "novel_camera_layout_probe": bool(novel_camera_layout_probe),
        "prompt_dependent": False,
    }


def _authority_conditioning_route_name(step_index: int, steps: int) -> str:
    """Bootstrap identity from public refs, then converge on one history state."""

    total = int(steps)
    index = int(step_index)
    if total <= 0 or not 0 <= index < total:
        raise ValueError("conditioning route step is outside the denoise schedule")
    reference_steps = int(math.ceil(
        total * _AUTHORITATIVE_REFERENCE_BOOTSTRAP_FRACTION
    ))
    return "reference" if index < reference_steps else "history"


def _progressive_conditioning_route_name(
    step_index: int,
    steps: int,
    *,
    visual_schedule: bool,
    inferred_voice_bootstrap: bool,
    layout_memory_bootstrap: bool = False,
    state_seeded_layout_bootstrap: bool = False,
    novel_camera_layout_probe: bool = False,
) -> str:
    """Select independent visual authority and inferred-voice exposure.

    Public audio is never scheduled here: it remains authoritative for every
    step.  Only generated long-term voice memory receives the early-bootstrap
    / late-release treatment.
    """

    total = int(steps)
    index = int(step_index)
    if total <= 0 or not 0 <= index < total:
        raise ValueError("conditioning route step is outside the denoise schedule")
    if visual_schedule and novel_camera_layout_probe:
        probe_start, probe_steps = _novel_camera_layout_probe_schedule(total)
        visual_name = (
            "reference"
            if probe_start <= index < probe_start + probe_steps
            else "history"
        )
    elif visual_schedule and layout_memory_bootstrap:
        layout_steps = int(math.ceil(
            total * _LAYOUT_MEMORY_BOOTSTRAP_FRACTION
        ))
        if state_seeded_layout_bootstrap and total >= 3:
            # V22 proved that replacing the state route with whole-image
            # layout rows can trade actor placement for a duplicated obsolete
            # prop.  State is now the primary route at every step.  A separate
            # bounded layout velocity is composed below without advancing the
            # primary forecast history.
            visual_name = "history"
        else:
            visual_name = "reference" if index < layout_steps else "history"
    else:
        visual_name = (
            _authority_conditioning_route_name(index, total)
            if visual_schedule
            else "default"
        )
    if not inferred_voice_bootstrap:
        return visual_name
    voice_steps = int(math.ceil(
        total * _INFERRED_VOICE_BOOTSTRAP_FRACTION
    ))
    audio_name = "voice" if index < voice_steps else "release"
    return f"{visual_name}_{audio_name}"


def _explicit_camera_layout_guidance_schedule(steps: int) -> tuple[int, int]:
    """Return the pure-state prefix and whole-window layout-guidance count."""

    total = int(steps)
    if total < 3:
        raise ValueError("explicit camera layout guidance requires at least three steps")
    state_steps = min(_EXPLICIT_CAMERA_STATE_SEED_STEPS, total - 2)
    layout_steps = min(
        int(math.ceil(total * _LAYOUT_MEMORY_BOOTSTRAP_FRACTION)),
        total - state_steps - 1,
    )
    return state_steps, layout_steps


def _novel_camera_layout_probe_schedule(steps: int) -> tuple[int, int]:
    """Return one middle-step scene probe after target-camera text seeding."""

    total = int(steps)
    if total < 3:
        raise ValueError("novel-camera layout probe requires at least three steps")
    start = min(_NOVEL_CAMERA_LAYOUT_PROBE_START_STEP, total - 2)
    count = min(_NOVEL_CAMERA_LAYOUT_PROBE_STEPS, total - start - 1)
    return start, count


def _layout_guidance_route_name(primary_route: str) -> str:
    """Map a state-primary authority route to its layout counterpart."""

    name = str(primary_route)
    if name == "history":
        return "reference"
    if name.startswith("history_"):
        return "reference_" + name.removeprefix("history_")
    raise ValueError("layout guidance requires a history-primary route")


def _blend_layout_guidance_velocity(
    state: torch.Tensor,
    layout: torch.Tensor,
    *,
    weight: float = _EXPLICIT_CAMERA_LAYOUT_GUIDANCE_WEIGHT,
) -> torch.Tensor:
    """Compose one full-window camera estimate into a state-primary velocity."""

    amount = float(weight)
    if state.ndim != 5 or layout.shape != state.shape:
        raise ValueError("layout guidance requires matching 5D video velocities")
    if not 0.0 < amount < 1.0:
        raise ValueError("layout guidance weight must lie inside (0, 1)")
    return torch.lerp(state, layout, amount)


class NativeT2AVHotSession:
    """Execute independent requests while retaining immutable host weights."""

    def __init__(
        self,
        *,
        engine: Literal["original", "lora", "reference", "reference_lora"],
        conditioner: Any,
        transformer: ImmutablePinnedModuleResidency,
        video_vae: ImmutablePinnedModuleResidency,
        audio_vae: ImmutablePinnedModuleResidency,
        decode_video: VideoDecoder,
        decode_audio: AudioDecoder,
        encode_video_conditioning: VideoConditionEncoder | None = None,
        encode_audio_conditioning: AudioConditionEncoder | None = None,
        output_root: Path,
        turbo_clock_mode: TurboClockMode = TurboClockMode.SHARED_VIDEO,
        debug_step_dir: Path | None = None,
        debug_final_latents_path: Path | None = None,
        runtime_config: RuntimeConfig = RuntimeConfig(),
        planner: RTX4090Planner | None = None,
        attention_backend: Any | None = None,
        v19_selector: Any | None = None,
        latent_upscaler: ImmutablePinnedModuleResidency | None = None,
        lora_video_shift: float = 12.0,
        lora_audio_shift: float = 3.0,
        lora_profile_id: str = "larry_turbo_v4_step600_ema",
        lora_recommended_steps: tuple[int, ...] = (4, 5, 6, 7, 8),
        lora_default_steps: int = 6,
        prefer_pinned_weight_prefetch: bool = False,
        resident_transformer_blocks: int = 0,
    ) -> None:
        self.engine = engine
        self.conditioner = conditioner
        self.transformer = transformer
        self.video_vae = video_vae
        self.audio_vae = audio_vae
        self.latent_upscaler = latent_upscaler
        self.decode_video = decode_video
        self.decode_audio = decode_audio
        self.encode_video_conditioning = encode_video_conditioning
        self.encode_audio_conditioning = encode_audio_conditioning
        self.output_root = output_root.resolve()
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.turbo_clock_mode = turbo_clock_mode
        self.lora_video_shift = float(lora_video_shift)
        self.lora_audio_shift = float(lora_audio_shift)
        self.lora_profile_id = str(lora_profile_id)
        self.lora_recommended_steps = tuple(int(step) for step in lora_recommended_steps)
        self.lora_default_steps = int(lora_default_steps)
        self.prefer_pinned_weight_prefetch = bool(prefer_pinned_weight_prefetch)
        self.resident_transformer_blocks = int(resident_transformer_blocks)
        if not 0 <= self.resident_transformer_blocks < 50:
            raise ValueError("resident_transformer_blocks must lie inside [0, 49]")
        block_stack = getattr(self.transformer.value, "block_stack", None)
        blocks = tuple(getattr(block_stack, "blocks", ()))
        self.transformer_block_device_bytes = max(
            (
                ImmutablePinnedModuleResidency._registered_nbytes(block)
                for block in blocks
            ),
            default=0,
        )
        if self.lora_video_shift <= 0.0 or self.lora_audio_shift <= 0.0:
            raise ValueError("LoRA sigma shifts must be positive")
        if self.lora_default_steps not in self.lora_recommended_steps:
            raise ValueError("LoRA default steps must be one of the recommended steps")
        self.debug_step_dir = (
            None if debug_step_dir is None else debug_step_dir.resolve()
        )
        self.debug_final_latents_path = (
            None
            if debug_final_latents_path is None
            else debug_final_latents_path.resolve()
        )
        self.runtime_config = runtime_config
        self.planner = planner
        self.attention_backend = attention_backend
        self.v19_selector = v19_selector
        # Research-only hook: a factory may share trajectory calibration across
        # requests while the immutable model session stays hot.  Production
        # behavior is unchanged when this remains ``None``.
        self.forecast_controller_factory: Callable[..., Any] | None = None
        # Research-only whole-DiT speculative verifier.  Production remains
        # unchanged unless a benchmark explicitly selects actual solver steps.
        self.self_speculative_verify_steps: tuple[int, ...] = ()
        self.self_speculative_verify_threshold: float = float("inf")
        self._active_block_executor = None
        # Exact one-entry cache for repeated-prompt seed/preset exploration.
        # Only immutable Qwen outputs live on pinned host memory; generated
        # state is never reused between requests.
        self._prompt_cache: tuple[str, torch.Tensor, torch.Tensor] | None = None
        # One exact multimodal cache supports the common creator workflow of
        # keeping prompt/anchors fixed while exploring seeds.  File content,
        # rather than only the path, participates in the key so replacing an
        # uploaded image can never return stale conditioning.
        self._conditioning_cache: tuple[
            str, torch.Tensor, torch.Tensor
        ] | None = None
        # Disk-backed first-pass conditioning is a one-entry host cache.  It
        # remains CPU-only so the 8/16-GB backends gain Qwen headroom without
        # reducing the DiT device budget.
        self._persisted_conditioning_cache: tuple[
            str, torch.Tensor, torch.Tensor
        ] | None = None
        self._last_conditioning_cache_payload: dict[str, Any] | None = None
        self._last_conditioning_cache_status = "not_checked"
        self._last_conditioning_cache_fallback: str | None = None
        self._media_digest_cache: dict[tuple[str, int, int], bytes] = {}
        # One exact reference-pack cache.  Ref2VA creators commonly keep the
        # same characters/props/voices while changing prompt or seed.  Those
        # VAE latents are deterministic and independent of the denoise state,
        # so retaining only the latest pack avoids repeated Video/Audio-VAE
        # encodes without allowing unbounded host-memory growth.
        self._reference_latent_cache: _ReferenceLatentCacheEntry | None = None

    @staticmethod
    def _uses_turbo_sampler(request: HotSessionRequest) -> bool:
        return bool(request.use_lora)

    @property
    def _uses_reference_layout(self) -> bool:
        return self.engine in ("reference", "reference_lora")

    @staticmethod
    def _timed(phases: dict[str, float], name: str, operation: Callable[[], Any]) -> Any:
        torch.cuda.synchronize()
        started = time.perf_counter()
        result = operation()
        torch.cuda.synchronize()
        phases[name] = time.perf_counter() - started
        return result

    @staticmethod
    def _release_device(*, collect_cycles: bool = False) -> None:
        # Normal tensor lifetimes are reference-counted. Full cyclic GC at
        # every DiT/VAE handoff adds online latency without releasing model
        # storage; reserve it for errors and service shutdown.
        if collect_cycles:
            gc.collect()
        torch.cuda.empty_cache()

    @staticmethod
    def _release_request_host_scratch() -> None:
        """Release completed task buffers without touching hot model slabs."""

        gc.collect()
        # PyTorch's host caching allocator otherwise retains multi-gigabyte
        # long-video staging blocks as /dev/zero mappings after the request.
        host_empty_cache = getattr(torch._C, "_host_emptyCache", None)
        if callable(host_empty_cache):
            host_empty_cache()
        try:
            libc = ctypes.CDLL(None)
            malloc_trim = libc.malloc_trim
            malloc_trim.argtypes = [ctypes.c_size_t]
            malloc_trim.restype = ctypes.c_int
            malloc_trim(0)
        except (AttributeError, OSError):
            pass

    def _decode_video_for_plan(
        self,
        model: Any,
        latents: torch.Tensor,
        frame_count: int,
        execution_plan: ExecutionPlan | None,
    ) -> torch.Tensor:
        """Apply one phase-local VAE transport graph without global state."""

        attribute = "_h3_temporal_host_chunk_frames"
        sentinel = object()
        previous = getattr(model, attribute, sentinel)
        host_chunk = (
            None if execution_plan is None else execution_plan.vae_temporal_tile
        )
        if host_chunk is None:
            try:
                delattr(model, attribute)
            except AttributeError:
                pass
        else:
            setattr(model, attribute, int(host_chunk))
        try:
            return self.decode_video(model, latents, frame_count)
        finally:
            if previous is sentinel:
                try:
                    delattr(model, attribute)
                except AttributeError:
                    pass
            else:
                setattr(model, attribute, previous)

    def _video_vae_block_streaming(
        self, model: Any, *, width: int, height: int
    ):
        """Return the phase-local residency context for this resource tier."""

        from .adapters.vae_block_streaming import stream_video_vae_decoder_tail

        enabled = (
            getattr(getattr(self, "runtime_config", None), "resource_profile", None)
            == "w4a8_8gb"
            and int(width) * int(height) > 1280 * 736
        )
        return stream_video_vae_decoder_tail(model, enabled=enabled)

    def generate(
        self, request: HotSessionRequest
    ) -> HotSessionResult | HotSessionCheckpointResult:
        """Generate and restore a clean CPU-resident state after any failure."""

        torch.cuda.reset_peak_memory_stats()
        telemetry_before = self._attention_telemetry()
        try:
            result = self._generate_impl(request)
            profile = dict(result.execution_profile)
            profile["attention_backend"] = self._telemetry_delta(
                telemetry_before,
                self._attention_telemetry(),
            )
            result = replace(
                result,
                execution_profile=profile,
                peak_allocated_gib=torch.cuda.max_memory_allocated() / (1024**3),
                peak_reserved_gib=torch.cuda.max_memory_reserved() / (1024**3),
            )
            self._persist_scheduler_telemetry(result)
            return result
        except BaseException as error:
            if _is_cuda_context_fatal(error):
                # CUDA reports illegal-access failures asynchronously.  Any
                # move_to(), synchronize() or empty_cache() after that point
                # can abort inside the driver; the old cleanup path did exactly
                # that and WSL then spent minutes writing a giant core while
                # port 8090 still looked alive.  Leave the poisoned context
                # untouched and let the service boundary fail closed.
                self._active_block_executor = None
                raise HotSessionDeviceFatal(
                    "CUDA context became unhealthy; restart the H3 service"
                ) from error
            # Do not strand 20+ GiB after a failed kernel/VAE invocation and
            # poison the next queued request.
            for component in (
                self.transformer,
                self.video_vae,
                self.audio_vae,
                self.latent_upscaler,
            ):
                if component is None:
                    continue
                try:
                    component.move_to("cpu", non_blocking=False)
                except Exception:
                    pass
            self._clear_block_executor()
            self._release_device(collect_cycles=True)
            raise

    def decode_latent_checkpoint(
        self,
        request: HotSessionRequest,
        checkpoint_path: Path,
        *,
        shot_video_checkpoints: tuple[Path, ...] = (),
        audio_window_checkpoints: tuple[Path, ...] = (),
        audio_window_clocks: tuple[tuple[int, int], ...] = (),
    ) -> HotSessionResult:
        """Decode an already-stitched clean AV trajectory.

        Continuation windows are decoded in one temporal domain.  When
        ``shot_video_checkpoints`` is supplied, authored hard cuts start fresh
        Video-VAE domains whose hidden writable prerolls are decoded and then
        cropped. Pixel tensors are copied into the final timeline before its
        single encode. Bounded causal audio windows are decoded in their own
        complete overlap domains and cropped only afterward, so the temporal
        Audio-VAE never sees a seam between independently completed latents.
        """

        request.validate()
        if request.latent_only or request.internal_video_tokens is not None:
            raise ValueError("stitched decode requires a normal full-clip request")
        output = request.output_path.resolve()
        if not output.is_relative_to(self.output_root):
            raise ValueError("output_path must stay inside output_root")
        output.parent.mkdir(parents=True, exist_ok=True)
        cancel_check = request.cancel_check or (lambda: False)

        def raise_if_cancelled() -> None:
            if cancel_check():
                raise HotSessionCancelled("native H3 generation cancelled")

        def progress(percent: float, stage: str, detail: str) -> None:
            if request.progress_callback is not None:
                request.progress_callback({
                    "percent": percent,
                    "stage": stage,
                    "detail": detail,
                })

        started = time.perf_counter()
        phases: dict[str, float] = {}
        torch.cuda.reset_peak_memory_stats()
        checkpoint = torch.load(
            Path(checkpoint_path), map_location="cpu", weights_only=True
        )
        expected = {
            "frames": request.frames,
            "fps": request.fps,
            "width": request.width,
            "height": request.height,
        }
        for key, value in expected.items():
            if checkpoint.get(key) != value:
                raise ValueError(
                    f"stitched latent metadata mismatch for {key}: "
                    f"expected {value!r}, got {checkpoint.get(key)!r}"
                )
        video = checkpoint.get("video")
        audio = checkpoint.get("audio")
        if not isinstance(video, torch.Tensor) or video.ndim != 5:
            raise ValueError("stitched checkpoint has invalid video latent")
        if not isinstance(audio, torch.Tensor) or audio.ndim != 4:
            raise ValueError("stitched checkpoint has invalid audio latent")
        del checkpoint
        if bool(audio_window_checkpoints) != bool(audio_window_clocks):
            raise ValueError(
                "audio window checkpoints and clocks must be supplied together"
            )
        if audio_window_checkpoints and len(audio_window_checkpoints) != len(
            audio_window_clocks
        ):
            raise ValueError("audio window checkpoints and clocks must align")
        if shot_video_checkpoints:
            # The globally stitched video remains the editable/resumable
            # trajectory, but it deliberately omits every hidden cut preroll.
            # Release it before loading the shot-local decode domains.
            del video
        raise_if_cancelled()
        execution_plan = request.execution_plan

        progress(90, "video_decode", "解码拼接后的高分辨率视频")
        self._timed(
            phases,
            "video_vae_h2d",
            lambda: self.video_vae.move_to("cuda:0", non_blocking=True),
        )
        if execution_plan is not None and execution_plan.vae_spatial_tile is not None:
            tile_height, tile_width = execution_plan.vae_spatial_tile
            if tile_height != tile_width:
                raise ValueError("the current H3 Video-VAE supports square tiles only")
            model = self.video_vae.value
            if not hasattr(model, "decoder_tile_size"):
                raise TypeError("the H3 Video-VAE does not expose decoder_tile_size")
            model.decoder_tile_size = tile_height
        from .adapters.vae_tiling import configure_vae_tile_batching
        from .adapters.vae_compile import (
            transformer_block_compile,
            transformer_block_compile_ready,
        )

        configure_vae_tile_batching(
            self.video_vae.value,
            1 if execution_plan is None else execution_plan.vae_tile_batch_size,
        )
        compile_vae_requested = bool(
            execution_plan is not None
            and execution_plan.vae_transformer_block_compile
        )
        compile_vae_block = bool(
            compile_vae_requested
            and transformer_block_compile_ready(self.video_vae.value)
        )
        vae_compile_profile = {
            "requested": compile_vae_requested,
            "enabled": compile_vae_block,
            "fallback": (
                "eager_exact_missing_prebuild_v1"
                if compile_vae_requested and not compile_vae_block
                else None
            ),
        }
        with self._video_vae_block_streaming(
            self.video_vae.value, width=request.width, height=request.height
        ):
            with transformer_block_compile(compile_vae_block):
                if shot_video_checkpoints:
                    decoded_video, shot_decode_profile = self._timed(
                        phases,
                        "video_decode",
                        lambda: self._decode_shot_video_checkpoints(
                            self.video_vae.value,
                            shot_video_checkpoints,
                            request=request,
                            execution_plan=execution_plan,
                        ),
                    )
                else:
                    decoded_video = self._timed(
                        phases,
                        "video_decode",
                        lambda: self._decode_video_for_plan(
                            self.video_vae.value,
                            video.to("cuda:0"),
                            request.frames,
                            execution_plan,
                        ),
                    )
                    shot_decode_profile = ()
                    del video
        self._timed(
            phases,
            "video_vae_evict",
            lambda: self.video_vae.move_to("cpu", non_blocking=False),
        )
        self._release_device()
        raise_if_cancelled()

        progress(96, "audio_decode", "解码原始音频")
        self._timed(
            phases,
            "audio_vae_h2d",
            lambda: self.audio_vae.move_to("cuda:0", non_blocking=True),
        )
        audio_decode_profile: dict[str, Any]
        if audio_window_checkpoints:
            del audio
            decoded_audio, audio_decode_profile = self._timed(
                phases,
                "audio_decode",
                lambda: self._decode_audio_window_checkpoints(
                    self.audio_vae.value,
                    audio_window_checkpoints,
                    audio_window_clocks,
                    request=request,
                ),
            )
        else:
            decoded_audio = self._timed(
                phases,
                "audio_decode",
                lambda: self.decode_audio(self.audio_vae.value, audio.to("cuda:0")),
            )
            del audio
            audio_decode_profile = {
                "policy": "single_stitched_audio_vae_domain_v1",
                "temporal_vae_domains": 1,
                "audio_latent_interpolation": None,
            }
        audio_manifold_profile: dict[str, Any] | None = None
        if request.audio_manifold_guard:
            from .audio_manifold_guard import apply_audio_manifold_guard

            guard_started = time.monotonic()
            decoded_audio, audio_manifold_profile = apply_audio_manifold_guard(
                self.audio_vae.value,
                decoded_audio,
                sample_rate=32_000,
            )
            phases["audio_manifold_guard"] = time.monotonic() - guard_started
        self._timed(
            phases,
            "audio_vae_evict",
            lambda: self.audio_vae.move_to("cpu", non_blocking=False),
        )
        self._release_device()
        raise_if_cancelled()

        progress(99, "mux", "封装高分辨率音视频")
        mux_receipt = self._timed(
            phases,
            "mux",
            lambda: AtomicPyAVMuxer(output_root=self.output_root).write(
                video=decoded_video,
                audio=decoded_audio,
                sample_rate=32000,
                fps=request.fps,
                output_path=output,
                cancel_check=raise_if_cancelled,
            ),
        )
        del decoded_video, decoded_audio
        self._timed(
            phases, "host_scratch_release", self._release_request_host_scratch
        )
        return HotSessionResult(
            output_path=output,
            total_seconds=time.perf_counter() - started,
            phases=phases,
            step_seconds=(),
            forecast_profile={"schema_version": 1, "mode": "decode_only"},
            execution_profile={
                "ultimate_upscale_decode": {
                    "latent_stitched": True,
                    "single_decode": not bool(shot_video_checkpoints),
                    "shot_boundary_decode": bool(shot_video_checkpoints),
                    "temporal_vae_domains": (
                        len(shot_video_checkpoints)
                        if shot_video_checkpoints else 1
                    ),
                    "shots": list(shot_decode_profile),
                },
                "video_vae_transformer_block_compile": vae_compile_profile,
                "audio_window_decode": audio_decode_profile,
                "audio_manifold_guard": audio_manifold_profile,
                "output_mux": {
                    "encoder": dict(mux_receipt.get("encoder", {})),
                    "media": dict(mux_receipt.get("media", {})),
                },
            },
            peak_allocated_gib=torch.cuda.max_memory_allocated() / (1024**3),
            peak_reserved_gib=torch.cuda.max_memory_reserved() / (1024**3),
        )

    def _decode_audio_window_checkpoints(
        self,
        model: Any,
        checkpoint_paths: tuple[Path, ...],
        window_clocks: tuple[tuple[int, int], ...],
        *,
        request: HotSessionRequest,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Decode each causal audio window before trimming hidden overlap."""

        from .audio_window_decode import assemble_window_decoded_audio

        decoded_windows: list[torch.Tensor] = []
        for index, (checkpoint_path, clock) in enumerate(
            zip(checkpoint_paths, window_clocks)
        ):
            context_frames, visible_frames = map(int, clock)
            document = torch.load(
                Path(checkpoint_path), map_location="cpu", weights_only=True
            )
            expected_frames = context_frames + visible_frames
            if int(document.get("frames", 0)) != expected_frames:
                raise ValueError(
                    f"audio window {index} frame clock does not match its checkpoint"
                )
            audio = document.get("audio")
            if not isinstance(audio, torch.Tensor) or audio.ndim != 4:
                raise ValueError(
                    f"audio window {index} has invalid latent geometry"
                )
            decoded = self.decode_audio(model, audio.to("cuda:0"))
            decoded_windows.append(decoded.detach().to("cpu").contiguous())
            del document, audio, decoded
        return assemble_window_decoded_audio(
            decoded_windows,
            window_clocks,
            output_frames=request.frames,
            fps=request.fps,
            sample_rate=32_000,
        )

    def _decode_shot_video_checkpoints(
        self,
        model: Any,
        checkpoint_paths: tuple[Path, ...],
        *,
        request: HotSessionRequest,
        execution_plan: ExecutionPlan | None,
    ) -> tuple[torch.Tensor, tuple[dict[str, int], ...]]:
        """Decode hard-cut shots independently and assemble exact RGB clocks."""

        if not checkpoint_paths:
            raise ValueError("shot-aware decode requires at least one checkpoint")
        output: torch.Tensor | None = None
        visible_cursor = 0
        receipts: list[dict[str, int]] = []
        for shot_index, checkpoint_path in enumerate(checkpoint_paths):
            document = torch.load(
                Path(checkpoint_path), map_location="cpu", weights_only=True
            )
            frames = int(document.get("frames", 0))
            lead = int(document.get("lead_context_frames", -1))
            visible = int(document.get("visible_frames", -1))
            for key, expected in (
                ("fps", request.fps),
                ("width", request.width),
                ("height", request.height),
            ):
                if document.get(key) != expected:
                    raise ValueError(
                        f"shot decode metadata mismatch for {key}: "
                        f"expected {expected!r}, got {document.get(key)!r}"
                    )
            latent = document.get("video")
            if not isinstance(latent, torch.Tensor) or latent.ndim != 5:
                raise ValueError("shot checkpoint has invalid video latent")
            if lead < 0 or visible <= 0 or lead + visible != frames:
                raise ValueError("shot checkpoint has an invalid visible clock")
            del document
            decoded = self._decode_video_for_plan(
                model,
                latent.to("cuda:0"),
                frames,
                execution_plan,
            )
            del latent
            if (
                not isinstance(decoded, torch.Tensor)
                or decoded.ndim != 5
                or decoded.shape[0] != 1
                or decoded.shape[1] != 3
                or decoded.shape[2] != frames
            ):
                raise RuntimeError(
                    "shot Video-VAE output must be [1,3,physical_frames,H,W]"
                )
            decoded = decoded.to("cpu", non_blocking=False)
            visible_stop = visible_cursor + visible
            if visible_stop > request.frames:
                raise RuntimeError("shot decode exceeds the requested output clock")
            if output is None:
                output = torch.empty(
                    (1, 3, request.frames, decoded.shape[3], decoded.shape[4]),
                    dtype=decoded.dtype,
                    device="cpu",
                )
            elif (
                decoded.dtype != output.dtype
                or decoded.shape[3:] != output.shape[3:]
            ):
                raise RuntimeError("shot Video-VAE output geometry changed at a cut")
            output[:, :, visible_cursor:visible_stop].copy_(
                decoded[:, :, lead:lead + visible]
            )
            receipts.append({
                "shot_index": shot_index,
                "physical_frames": frames,
                "hidden_preroll_frames": lead,
                "visible_frames": visible,
                "visible_start_frame": visible_cursor,
            })
            visible_cursor = visible_stop
            del decoded
        if output is None or visible_cursor != request.frames:
            raise RuntimeError("shot decode missed the requested output clock")
        return output, tuple(receipts)

    def _attention_telemetry(self) -> dict[str, Any]:
        telemetry = getattr(self.attention_backend, "telemetry", None)
        return dict(telemetry()) if callable(telemetry) else {}

    @classmethod
    def _telemetry_delta(
        cls,
        before: Any,
        after: Any,
        path: tuple[str, ...] = (),
    ) -> Any:
        """Produce one request-local view from cumulative backend counters."""

        if isinstance(before, dict) and isinstance(after, dict):
            return {
                key: cls._telemetry_delta(before.get(key), value, path + (key,))
                for key, value in after.items()
            }
        if (
            isinstance(before, (int, float))
            and not isinstance(before, bool)
            and isinstance(after, (int, float))
            and not isinstance(after, bool)
        ) and path and (
            path[-1].endswith(("calls", "_count", "_heads", "_tokens", "_pairs"))
            or (len(path) > 1 and path[-2] == "action_calls")
        ):
            return after - before
        if isinstance(before, list) and isinstance(after, list):
            return after[len(before):] if after[:len(before)] == before else after
        return after

    @staticmethod
    def _persist_scheduler_telemetry(
        result: HotSessionResult | HotSessionCheckpointResult,
    ) -> None:
        """Persist opt-in research evidence without adding release latency."""

        destination = os.environ.get("H3_NATIVE_SCHEDULER_TELEMETRY_DIR", "").strip()
        if not destination:
            return
        root = Path(destination).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        output = getattr(result, "output_path", None) or getattr(
            result, "checkpoint_path", None
        )
        stem = Path(output).stem if output is not None else f"request-{time.time_ns()}"
        target = root / f"{stem}.scheduler.json"
        temporary = target.with_suffix(target.suffix + ".tmp")
        document = {
            "schema_version": "h3_native_scheduler_runtime_v1",
            "artifact": None if output is None else str(Path(output).resolve()),
            "total_seconds": result.total_seconds,
            "peak_allocated_gib": result.peak_allocated_gib,
            "peak_reserved_gib": result.peak_reserved_gib,
            "phases": result.phases,
            "step_seconds": list(result.step_seconds),
            "execution_profile": result.execution_profile,
            "forecast_profile": getattr(result, "forecast_profile", None),
        }
        temporary.write_text(
            json.dumps(document, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        temporary.replace(target)

    def _clear_block_executor(self) -> None:
        dit = self.transformer.value
        block_stack = getattr(dit, "block_stack", None)
        if block_stack is not None:
            block_stack.clear_block_executor()
        self._active_block_executor = None

    def _analyze_request_features(
        self,
        request: HotSessionRequest,
        *,
        text_tokens: int,
    ):
        all_steps = tuple(range(request.steps))
        actual = (
            all_steps
            if request.actual_step_indices is None
            else request.actual_step_indices
        )
        forecast_count = request.steps - len(actual) if not self._uses_turbo_sampler(request) else 0
        user_reference_count = (
            len(request.reference_images)
            + len(request.reference_videos)
            + len(request.reference_audios)
        )
        keyframe_count = (
            int(request.first_frame is not None)
            + int(request.last_frame is not None)
        )
        condition_count = user_reference_count or keyframe_count
        spatial_tokens = (request.height // 32) * (request.width // 32)
        condition_tokens = keyframe_count * spatial_tokens
        if request.reference_images or request.reference_videos or request.reference_audios:
            from .adapters.conditioning_vae.preprocess import prepare_reference_audios, prepare_reference_images, prepare_reference_videos

            image_tokens = sum(
                ((image.height + 31) // 32) * ((image.width + 31) // 32)
                for image in (prepare_reference_images(request) if request.reference_images else ())
            )
            video_tokens = sum(
                (((len(item.frames) - 5) // 17) * 5 + 2)
                * ((item.frames.shape[1] + 31) // 32)
                * ((item.frames.shape[2] + 31) // 32)
                for item in (prepare_reference_videos(request) if request.reference_videos else ())
            )
            audio_tokens = sum(
                int(item.waveform.shape[-1] + 799) // 800 * 2
                for item in (prepare_reference_audios(request) if request.reference_audios else ())
            )
            condition_tokens = image_tokens + video_tokens + audio_tokens
        if request.av_token_memory_path is not None:
            from .av_token_memory import validate_av_token_memory

            memory = validate_av_token_memory(
                torch.load(
                    Path(request.av_token_memory_path),
                    map_location="cpu",
                    weights_only=True,
                )
            )
            memory_video_tokens = sum(
                int(item["latent"].shape[2])
                * ((int(item["latent"].shape[3]) + 1) // 2)
                * ((int(item["latent"].shape[4]) + 1) // 2)
                for item in memory["video_entries"]
            )
            memory_audio_tokens = sum(
                2 * int(item["latent"].shape[-1])
                for item in memory["audio_entries"]
            )
            condition_count += (
                len(memory["video_entries"]) + len(memory["audio_entries"])
            )
            condition_tokens += memory_video_tokens + memory_audio_tokens
            del memory
        # H3WorkloadAnalyzer's count is the bounded public Ref2VA category
        # (maximum 15); exact internal memory width is carried independently by
        # condition_tokens_override and must not be discarded when the merged
        # user+memory item count exceeds that public contract.
        planner_condition_count = min(15, condition_count)
        features = H3WorkloadAnalyzer(fps=request.fps).analyze(
            width=request.width,
            height=request.height,
            frames=request.frames,
            text_tokens=text_tokens,
            condition_count=planner_condition_count,
            engine=("reference" if self._uses_reference_layout else ("lora" if request.use_lora else "original")),
            actual_evaluations=len(actual),
            forecast_evaluations=forecast_count,
            condition_tokens_override=(
                condition_tokens if condition_count else None
            ),
            latent_frames_override=request.internal_video_tokens,
            audio_frames_override=request.internal_audio_tokens,
        )
        return features

    def _apply_v19_selection(
        self,
        request: HotSessionRequest,
        *,
        text_tokens: int,
    ) -> HotSessionRequest:
        if request.v19_acceleration is None:
            return request
        if self.v19_selector is None:
            # A configured V19 request cannot silently execute a legacy
            # approximate scheduler.  Preserve the request and model ability
            # with the complete Dense trajectory instead.
            dense_request = replace(
                request,
                actual_step_indices=tuple(range(request.steps)),
                attention_action_schedule=(),
                attention_online_guard_id=None,
                attention_online_budget_dense_layers=0.0,
                attention_online_rebate_schedule=(),
                acceleration_plan_summary={
                    "policy_id": "h3_v19_human_aligned_budgeted_adaptive_inference",
                    "accelerated": False,
                    "reason": "v19_release_bundle_unavailable_dense_fallback",
                    "acceleration": request.v19_acceleration,
                },
            )
            dense_request.validate()
            return dense_request
        from .planner import V19WorkloadContext

        # Initial features are used only to obtain the exact packed layout.
        # The selector then replaces the optimizer-owned actual-step schedule.
        features = self._analyze_request_features(
            request,
            text_tokens=text_tokens,
        )
        workload = V19WorkloadContext(
            model_variant="lora" if request.use_lora else "base",
            service_family=(
                "reference" if self._uses_reference_layout else "first_last"
            ),
            packed_tokens=features.packed_tokens,
            condition_count=features.condition_count,
            reference_images=len(request.reference_images),
            reference_audio=len(request.reference_audios),
            reference_videos=len(request.reference_videos),
            device_arch="sm89",
            width=request.width,
            height=request.height,
            frames=request.frames,
            steps=request.steps,
            actual_step_indices=(
                tuple(range(request.steps))
                if request.actual_step_indices is None
                else request.actual_step_indices
            ),
            sampler="turbo" if request.use_lora else "res_multistep",
            scheduler="simple",
        )
        required_actual_steps = tuple(sorted(set(
            self_index
            for self_index in (
                *request.scheduler_required_actual_step_indices,
                *(
                    ()
                    if request.preview_step_index is None
                    else (request.preview_step_index,)
                ),
            )
        )))
        selected = self.v19_selector.select(
            workload=workload,
            acceleration=request.v19_acceleration,
            required_actual_step_indices=required_actual_steps,
        )
        split = request.acceleration_transition_step
        second_acceleration = request.v19_second_pass_acceleration
        composite = bool(
            split is not None
            and split < request.steps
            and second_acceleration is not None
            and not math.isclose(
                float(second_acceleration), float(request.v19_acceleration)
            )
        )
        if composite:
            assert split is not None
            assert second_acceleration is not None
            selected_second = self.v19_selector.select(
                workload=workload,
                acceleration=second_acceleration,
                required_actual_step_indices=required_actual_steps,
            )
            selected_actual = tuple(sorted({
                *(
                    step for step in selected.actual_step_indices
                    if step < split
                ),
                *(
                    step for step in selected_second.actual_step_indices
                    if step >= split
                ),
            }))
            selected_schedule = tuple(sorted((
                *(
                    row for row in selected.attention_action_schedule
                    if row[0] < split
                ),
                *(
                    row for row in selected_second.attention_action_schedule
                    if row[0] >= split
                ),
            )))
            selected_summary = {
                "schema_version": "h3_stage_acceleration_v1",
                "policy_id": "h3_stage_acceleration_v1",
                "accelerated": bool(selected_schedule) or (
                    len(selected_actual) < request.steps
                ),
                "acceleration": request.v19_acceleration,
                "second_pass_acceleration": second_acceleration,
                "acceleration_transition_step": split,
                "actual_step_indices": list(selected_actual),
                "forecast_steps": request.steps - len(selected_actual),
                "required_actual_step_indices": list(required_actual_steps),
                "stage_plans": {
                    "first_pass": selected.summary,
                    "second_pass": selected_second.summary,
                },
            }
            runtime_controller = None
        else:
            selected_actual = selected.actual_step_indices
            selected_schedule = selected.attention_action_schedule
            selected_summary = selected.summary
            runtime_controller = getattr(
                selected, "runtime_controller", None
            )
        selected_request = replace(
            request,
            actual_step_indices=selected_actual,
            attention_action_schedule=selected_schedule,
            attention_online_guard_id=None,
            attention_online_budget_dense_layers=0.0,
            attention_online_rebate_schedule=(),
            acceleration_plan_summary=selected_summary,
            mechanistic_runtime_controller=runtime_controller,
        )
        selected_request.validate()
        return selected_request

    def _device_execution_budget_bytes(self) -> int:
        """Return service-usable capacity, accounting for external GPU users."""

        runtime_config = getattr(self, "runtime_config", None)
        configured = (
            23 * 1024**3
            if runtime_config is None
            else int(runtime_config.max_device_bytes)
        )
        if runtime_config is None or runtime_config.device == "cpu":
            return configured
        try:
            free_bytes, _ = torch.cuda.mem_get_info(runtime_config.device)
            allocated = torch.cuda.memory_allocated(runtime_config.device)
            reusable_cache = max(
                0,
                torch.cuda.memory_reserved(runtime_config.device) - allocated,
            )
        except (RuntimeError, AssertionError):
            return configured
        # The peak model is a whole-service peak, so add currently allocated
        # service tensors back. External allocations remain excluded through
        # cudaMemGetInfo and can therefore trigger the low-VRAM route.
        # cudaMemGetInfo also excludes this process's CUDA context and library
        # bookkeeping, while the calibrated peak model is expressed in Torch
        # allocations.  Add back the measured context envelope before taking
        # the configured cap; genuinely external allocations larger than that
        # still reduce the available budget and trigger the low-VRAM route.
        cuda_context_allowance = 768 * 1024**2
        return min(
            configured,
            int(
                free_bytes
                + allocated
                + reusable_cache
                + cuda_context_allowance
            ),
        )

    def _apply_memory_execution_policy(
        self,
        request: HotSessionRequest,
        features,
        plan: ExecutionPlan,
        *,
        chunk_reason: str | None,
        qk_override: str | None,
    ) -> tuple[ExecutionPlan, str | None, str | None, dict[str, object]]:
        """Project one quality plan onto the fastest budget-feasible graph."""

        decision = select_memory_execution(
            features,
            requested_mode=request.memory_mode,
            device_budget_bytes=self._device_execution_budget_bytes(),
            weight_tier=getattr(
                getattr(self, "runtime_config", None),
                "weight_tier",
                "int8",
            ),
            resource_profile=getattr(
                getattr(self, "runtime_config", None),
                "resource_profile",
                None,
            ),
            include_vae=not request.latent_only,
            existing_query_chunk_tokens=(
                plan.long_sequence_query_chunk_tokens
            ),
        )
        if (
            plan.vae_temporal_tile is not None
            and decision.vae_temporal_tile is None
            and not request.latent_only
        ):
            # An explicit product route may choose the exact temporal host
            # sink even when the clip would fit through the materialized VAE
            # graph.  Preserve that request while keeping admission and
            # telemetry tied to the smaller host-streamed working set.
            decision = replace(
                decision,
                vae_temporal_tile=int(plan.vae_temporal_tile),
                estimated_vae_selected_peak_bytes=(
                    decision.estimated_vae_host_peak_bytes
                ),
                estimated_selected_peak_bytes=max(
                    decision.estimated_dit_peak_bytes,
                    decision.estimated_vae_host_peak_bytes,
                ),
                reason=f"{decision.reason}_explicit_temporal_vae_sink",
            )
        if not decision.fits_budget:
            raise RuntimeError(
                "no full-context H3 execution graph can fit this packed "
                "workload inside the configured device budget: "
                f"packed_tokens={features.packed_tokens}, "
                f"predicted_peak_gib={decision.estimated_selected_peak_bytes / 1024**3:.3f}, "
                f"budget_gib={decision.device_budget_bytes / 1024**3:.3f}, "
                f"latent_only={request.latent_only}"
            )
        effective_qk, memory_qk_reason = select_stable_dense_qk_quantization(
            plan.dense_qk_quant_gran,
            packed_tokens=features.packed_tokens,
        )
        forced_qk_upgrade = effective_qk != plan.dense_qk_quant_gran
        memory_telemetry = decision.telemetry()
        if (
            getattr(self, "prefer_pinned_weight_prefetch", False)
            and decision.weight_tier == "w4a8"
            and decision.block_buffer_count == 1
        ):
            # The standard 8-GiB route keeps a 128-MiB planner guard and thus
            # serializes one 209-MiB W4 block slot. A 32-GiB host keeps source
            # DiT pages locked, making a second slot useful: layer N+1 copies
            # over PCIe while layer N computes. The hard CUDA allocator limit
            # remains the final fail-closed guard.
            double_buffer_peak = (
                decision.estimated_dit_peak_bytes
                + _H3_W4A8_SECOND_BLOCK_BUFFER_BYTES
            )
            if double_buffer_peak <= decision.device_budget_bytes:
                decision = replace(
                    decision,
                    block_buffer_count=2,
                    estimated_dit_peak_bytes=double_buffer_peak,
                    estimated_selected_peak_bytes=max(
                        double_buffer_peak,
                        decision.estimated_vae_selected_peak_bytes,
                    ),
                    reason="w4a8_32gb_pinned_double_buffer_prefetch",
                )
                memory_telemetry = decision.telemetry()
                memory_telemetry["uses_allocator_guard_band"] = True
        # A few planner-only callers construct a lightweight session with
        # ``__new__`` and deliberately skip the heavyweight model initializer.
        # Treat the new product residency knob as zero in that diagnostic path
        # instead of turning a planner probe into an AttributeError.
        research_resident_blocks = int(
            getattr(self, "resident_transformer_blocks", 0) or 0
        )
        research_resident_raw = os.environ.get(
            "H3_NATIVE_RESEARCH_RESIDENT_BLOCKS", ""
        ).strip()
        research_int8_curve = bool(
            decision.weight_tier == "int8"
            and os.environ.get("H3_NATIVE_RESEARCH_INT8_HOST_CURVE", "0") == "1"
        )
        research_residency_enabled = bool(
            research_resident_blocks > 0
            or (
                research_resident_raw
                and (decision.weight_tier == "w4a8" or research_int8_curve)
            )
        )
        if research_residency_enabled:
            if research_resident_raw:
                try:
                    research_resident_blocks = int(research_resident_raw)
                except ValueError as error:
                    raise ValueError(
                        "H3_NATIVE_RESEARCH_RESIDENT_BLOCKS must be an integer"
                    ) from error
            if not 0 <= research_resident_blocks < 50:
                raise ValueError(
                    "H3_NATIVE_RESEARCH_RESIDENT_BLOCKS must lie inside [0, 49]"
                )
            requested_resident_blocks = research_resident_blocks
            resident_bytes = 0
            if research_resident_blocks > 0:
                block_bytes = (
                    _H3_W4A8_SECOND_BLOCK_BUFFER_BYTES
                    if decision.weight_tier == "w4a8"
                    else _H3_INT8_BLOCK_BUFFER_BYTES
                )
                # The loaded module is authoritative: a LoRA-enabled block
                # owns its adapter tensors in addition to the quantized base
                # tensors.  The old fixed W4 estimate omitted up to ~26 MiB
                # per resident block and over-admitted the 24-GiB LoRA route.
                block_bytes = max(
                    block_bytes,
                    int(getattr(self, "transformer_block_device_bytes", 0) or 0),
                )
                # Residency is a latency optimization layered on top of the
                # request graph.  Sparse cells use compact K/V, but every Base
                # trajectory retains exact Dense anchors whose SageAttention
                # Q/K/V working set is materially larger.  Size the resident
                # prefix from that heaviest Actual cell; Forecast count and
                # sparse keep ratio must never buy additional capacity.
                (
                    research_resident_blocks,
                    safe_resident_blocks,
                    dense_actual_peak,
                ) = select_dense_safe_resident_blocks(
                    features,
                    decision,
                    requested_blocks=research_resident_blocks,
                    block_bytes=block_bytes,
                )
                resident_bytes = (
                    research_resident_blocks
                    * block_bytes
                )
                resident_dit_peak = (
                    max(
                        decision.estimated_dit_peak_bytes,
                        dense_actual_peak,
                    )
                    + resident_bytes
                )
                if (
                    resident_dit_peak
                    > decision.device_budget_bytes - _H3_MEMORY_POLICY_GUARD_BYTES
                ):
                    raise RuntimeError(
                        "requested resident prefix exceeds the research VRAM budget: "
                        f"resident_blocks={research_resident_blocks}, "
                        f"predicted_peak_gib={resident_dit_peak / 1024**3:.3f}, "
                        f"budget_gib={decision.device_budget_bytes / 1024**3:.3f}"
                    )
                decision = replace(
                    decision,
                    estimated_dit_peak_bytes=resident_dit_peak,
                    estimated_selected_peak_bytes=max(
                        resident_dit_peak,
                        decision.estimated_vae_selected_peak_bytes,
                    ),
                    reason=f"budget_{decision.weight_tier}_gpu_resident_prefix",
                )
                memory_telemetry = decision.telemetry()
                memory_telemetry.update({
                    "dense_actual_capacity_model": (
                        "exact_streaming_plus_runtime_guard_v1"
                    ),
                    "estimated_dense_actual_peak_bytes": dense_actual_peak,
                    "estimated_dense_actual_peak_gib": (
                        dense_actual_peak / 1024**3
                    ),
                    "requested_resident_block_count": (
                        requested_resident_blocks
                    ),
                    "safe_resident_block_capacity": int(
                        safe_resident_blocks
                    ),
                    "resident_prefix_clamped_for_dense_actual": bool(
                        research_resident_blocks
                        < requested_resident_blocks
                    ),
                })
            memory_telemetry.update({
                "resource_budget_residency": True,
                "resident_block_count": research_resident_blocks,
                "resident_block_bytes": resident_bytes,
                "resident_block_gib": resident_bytes / 1024**3,
            })
        if forced_qk_upgrade:
            # Full-context streaming retains every admitted KV block, but the
            # coarser per-warp INT8 Q/K scale can change sparse scores and LUT
            # ordering.  Do not advertise byte equality merely because K/V
            # are non-compact.
            memory_telemetry.update({
                "bit_exact": False,
                "numerical_contract": "full_context_per_warp_qk",
                "approximation_sources": ["per_warp_int8_qk_granularity"],
            })
        release_fused_query_projection = bool(
            request.release_byte_exact_optimizations
            and decision.selected_scheme == "exact_streaming"
        )
        # The output-layout kernel is byte-exact on both model families, but
        # its latency win was repeatable only on the FL2VA base graph.  Keep
        # Ref2VA on the fused-Query-only rail until its heavier reference
        # schedule clears the same positive-gain release gate.
        release_direct_nhd_output = bool(
            release_fused_query_projection and self.engine == "original"
        )
        # Q/K normalization and RoPE can write directly into the HND backing
        # consumed by Attention, removing the two large NHD->HND materializing
        # copies.  The full physical R-C-R/W-C-R-C gates were byte-exact and
        # positive on INT8 FL2VA and Ref2VA at both 16GB 720p15 and 24GB
        # 1080p15.  Compact K/V and direct-NHD-K/V use different storage
        # contracts and remain deliberately excluded.
        release_fused_qknorm_hnd_layout = bool(
            release_fused_query_projection
            and getattr(
                getattr(self, "runtime_config", None),
                "weight_tier",
                "int8",
            ) == "int8"
            and not decision.compact_kv
            and not plan.long_sequence_direct_nhd_kv
        )
        # Value projection already arrives as non-compact HND on this rail.
        # Quantize it directly into SageAttention's final FP8 backing instead
        # of materializing an NHD transpose and a second HND staging tensor.
        # Physical W-C-R-C gates were byte-exact and reduced mean actual-step
        # latency for INT8 FL2VA and Ref2VA at both 16GB 720p15 and 24GB
        # 1080p15.  Compact K/V and direct-NHD-K/V retain their own contracts.
        release_direct_hnd_fp8_value = bool(
            release_fused_qknorm_hnd_layout
        )
        # The protected prefix K path historically materialized a full
        # ``K - mean`` tensor and then launched SageAttention's key quantizer.
        # The fused writer performs the same BF16 boundary subtraction and
        # per-thread INT8 reduction directly into Sage's final HND buffers.  Its
        # v2 ragged-tail mask is essential: invalid rows must be zero *after*
        # centering or a non-zero key mean can change the last block scale.
        # Adversarial ragged tensors plus full-DiT W-C-R-C gates are byte-exact
        # on INT8 FL2VA/Ref2VA at 16GB 720p15 and 24GB 1080p15.  At 1080p15 it
        # also removes 1.68--2.24 GiB of peak allocation and materially reduces
        # latency, so it belongs to the exact streaming release rail.
        release_fused_prefix_k_quant = bool(
            release_fused_qknorm_hnd_layout
        )
        # The Q and K/V halves of split projection quantize the same hidden
        # rows twice.  Reusing the exact ConvRot row-INT8 activation was
        # byte-identical and repeatably positive for both FL2VA and Ref2VA
        # under the 16GB gate.  The original 24GB FL2VA gate reached 23.92 GiB
        # and remains excluded.  After fused-prefix-K and allocator work, the
        # current four-image Ref2VA 1080p15 W-C-R-C gate peaked at 20.43 GiB,
        # stayed byte-exact, and improved an Actual step by 1.011x.  Admit only
        # that measured graph, and still require request-local budget headroom.
        shared_qkv_cache_bytes = int(
            features.packed_tokens * _H3_SHARED_QKV_BYTES_PER_PACKED_TOKEN
        )
        shared_qkv_admission_budget_bytes = max(
            0,
            int(decision.device_budget_bytes)
            - _H3_MEMORY_POLICY_GUARD_BYTES,
        )
        shared_qkv_required_peak_bytes = int(
            decision.estimated_selected_peak_bytes
            + shared_qkv_cache_bytes
            + _H3_SHARED_QKV_RELEASE_RESERVE_BYTES
        )
        shared_qkv_fits_budget = bool(
            shared_qkv_required_peak_bytes
            <= shared_qkv_admission_budget_bytes
        )
        resource_profile = getattr(
            getattr(self, "runtime_config", None),
            "resource_profile",
            None,
        )
        release_shared_qkv_quantization = bool(
            release_fused_qknorm_hnd_layout
            and (
                resource_profile == "int8_16gb"
                or (
                    resource_profile == "int8_24gb"
                    and self.engine == "reference"
                )
            )
            and shared_qkv_fits_budget
        )
        # Partial Top-K avoids sorting the unused KV tail, then feeds the same
        # selected block set into the unchanged Sparge kernel.  Full-DiT
        # W-C-R-C gates were byte-identical on FL2VA at 720p15 and 1080p15;
        # the longer shape saved 0.543 s per actual step.  Ref2VA was also
        # byte-identical but latency-neutral, so its reference-heavy graph
        # deliberately keeps the established full-sort path.
        release_partial_sparse_topk = bool(
            release_fused_qknorm_hnd_layout
            and self.engine in ("original", "lora")
        )
        memory_telemetry.update({
            "release_byte_exact_optimizations": bool(
                request.release_byte_exact_optimizations
            ),
            "release_fused_query_projection": release_fused_query_projection,
            "release_fused_query_evidence": (
                "h3_fused_query_full_dit_byte_exact_16gb_720p15_"
                "24gb_1080p15_20260828"
                if release_fused_query_projection
                else None
            ),
            "release_direct_nhd_output": release_direct_nhd_output,
            "release_direct_nhd_output_evidence": (
                "h3_direct_nhd_output_full_dit_byte_exact_fl2va_"
                "16gb_720p15_24gb_1080p15_20260828"
                if release_direct_nhd_output
                else None
            ),
            "release_fused_qknorm_hnd_layout": (
                release_fused_qknorm_hnd_layout
            ),
            "release_fused_qknorm_hnd_evidence": (
                "h3_fused_qknorm_hnd_full_dit_byte_exact_fl2va_ref2va_"
                "16gb_720p15_24gb_1080p15_20260828"
                if release_fused_qknorm_hnd_layout
                else None
            ),
            "release_direct_hnd_fp8_value": release_direct_hnd_fp8_value,
            "release_direct_hnd_fp8_value_evidence": (
                "h3_direct_hnd_fp8_value_full_dit_byte_exact_fl2va_ref2va_"
                "16gb_720p15_24gb_1080p15_20260828"
                if release_direct_hnd_fp8_value
                else None
            ),
            "release_fused_prefix_k_quant": release_fused_prefix_k_quant,
            "release_fused_prefix_k_quant_evidence": (
                "h3_fused_prefix_k_quant_v2_ragged_exact_full_dit_"
                "fl2va_ref2va_16gb_720p15_24gb_1080p15_20260828"
                if release_fused_prefix_k_quant
                else None
            ),
            "release_shared_qkv_quantization": (
                release_shared_qkv_quantization
            ),
            "shared_qkv_quantization_cache_bytes": shared_qkv_cache_bytes,
            "shared_qkv_quantization_cache_gib": (
                shared_qkv_cache_bytes / 1024**3
            ),
            "shared_qkv_quantization_required_peak_bytes": (
                shared_qkv_required_peak_bytes
            ),
            "shared_qkv_quantization_required_peak_gib": (
                shared_qkv_required_peak_bytes / 1024**3
            ),
            "shared_qkv_quantization_fits_budget": (
                shared_qkv_fits_budget
            ),
            "release_shared_qkv_quantization_evidence": (
                (
                    "h3_shared_qkv_int8_activation_exact_ref2va_"
                    "24gb_1080p15_four_image_20260831"
                    if resource_profile == "int8_24gb"
                    else
                    "h3_shared_qkv_int8_activation_exact_full_dit_"
                    "fl2va_ref2va_16gb_720p15_20260828"
                )
                if release_shared_qkv_quantization
                else None
            ),
            "release_partial_sparse_topk": release_partial_sparse_topk,
            "release_partial_sparse_topk_evidence": (
                "h3_partial_topk_full_dit_byte_exact_fl2va_"
                "16gb_720p15_24gb_1080p15_20260828"
                if release_partial_sparse_topk
                else None
            ),
        })
        plan = replace(
            plan,
            resident_block_count=(
                research_resident_blocks
                if research_residency_enabled
                else plan.resident_block_count
            ),
            block_buffer_count=decision.block_buffer_count,
            prefetch_depth=1 if decision.block_buffer_count == 2 else 0,
            mlp_chunk_tokens=decision.mlp_chunk_tokens,
            vae_spatial_tile=(
                decision.vae_spatial_tile,
                decision.vae_spatial_tile,
            ),
            vae_temporal_tile=decision.vae_temporal_tile,
            dense_qk_quant_gran=effective_qk,
            long_sequence_query_chunk_tokens=decision.query_chunk_tokens,
            long_sequence_projection_chunk_tokens=(
                min(
                    plan.long_sequence_projection_chunk_tokens,
                    decision.projection_chunk_tokens,
                )
            ),
            long_sequence_split_qkv_outputs=(
                decision.query_chunk_tokens is not None
            ),
            long_sequence_shared_qkv_quantization=(
                decision.query_chunk_tokens is not None
                and not decision.compact_kv
                and shared_qkv_fits_budget
                and (
                    plan.long_sequence_shared_qkv_quantization
                    or release_shared_qkv_quantization
                )
            ),
            long_sequence_compact_kv=decision.compact_kv,
            long_sequence_exact_helper_stack=(
                plan.long_sequence_exact_helper_stack
                if decision.query_chunk_tokens is not None
                else False
            ),
            long_sequence_single_qknorm_rope=(
                decision.query_chunk_tokens is not None
            ),
            long_sequence_parallel_sparse_lut=(
                decision.query_chunk_tokens is not None
            ),
            long_sequence_partial_sparse_topk=(
                decision.query_chunk_tokens is not None
                and (
                    plan.long_sequence_partial_sparse_topk
                    or release_partial_sparse_topk
                )
            ),
            long_sequence_fused_prefix_k_quant=(
                decision.query_chunk_tokens is not None
                and not decision.compact_kv
                and not plan.long_sequence_direct_nhd_kv
                and (
                    plan.long_sequence_fused_prefix_k_quant
                    or release_fused_prefix_k_quant
                )
            ),
            long_sequence_fused_query_projection=(
                decision.query_chunk_tokens is not None
                and (
                    plan.long_sequence_fused_query_projection
                    or release_fused_query_projection
                )
            ),
            long_sequence_fused_qknorm_hnd_layout=(
                decision.query_chunk_tokens is not None
                and not decision.compact_kv
                and (
                    plan.long_sequence_fused_qknorm_hnd_layout
                    or release_fused_qknorm_hnd_layout
                )
            ),
            long_sequence_direct_nhd_output=(
                decision.query_chunk_tokens is not None
                and (
                    plan.long_sequence_direct_nhd_output
                    or release_direct_nhd_output
                )
            ),
            long_sequence_direct_nhd_kv=(
                plan.long_sequence_direct_nhd_kv
                if decision.query_chunk_tokens is not None
                and not decision.compact_kv
                else False
            ),
            long_sequence_direct_hnd_fp8_value=(
                decision.query_chunk_tokens is not None
                and not decision.compact_kv
                and not plan.long_sequence_direct_nhd_kv
                and (
                    plan.long_sequence_direct_hnd_fp8_value
                    or release_direct_hnd_fp8_value
                )
            ),
        )
        if qk_override is None and forced_qk_upgrade:
            qk_override = memory_qk_reason or "long_sequence_tail_stability"
        return (
            plan,
            (
                "isolated_resource_whole_query"
                if decision.query_chunk_tokens is None
                else chunk_reason or "isolated_resource_streaming"
            ),
            qk_override,
            memory_telemetry,
        )

    def _resolve_execution_plan(
        self,
        request: HotSessionRequest,
        *,
        text_tokens: int,
    ) -> tuple[ExecutionPlan | None, dict[str, Any]]:
        features = self._analyze_request_features(
            request,
            text_tokens=text_tokens,
        )
        feature_profile = {
            "packed_tokens": features.packed_tokens,
            "spatial_tokens": features.spatial_tokens,
            "latent_frames": features.latent_frames,
            "output_pixel_frames": features.output_pixel_frames,
            "condition_count": features.condition_count,
            "actual_evaluations": features.actual_evaluations,
            "forecast_evaluations": features.forecast_evaluations,
        }
        if request.execution_plan is not None:
            requested_qk = request.execution_plan.dense_qk_quant_gran
            effective_qk, qk_override = select_stable_dense_qk_quantization(
                requested_qk,
                packed_tokens=features.packed_tokens,
            )
            plan = (
                request.execution_plan
                if effective_qk == requested_qk
                else replace(
                    request.execution_plan,
                    dense_qk_quant_gran=effective_qk,
                )
            )
            chunk_reason = "explicit" if (
                plan.long_sequence_query_chunk_tokens is not None
            ) else None
            v24_execution_hint = (
                None
                if request.acceleration_plan_summary is None
                else request.acceleration_plan_summary.get(
                    "execution_profile_hint"
                )
            )
            if (
                plan.long_sequence_query_chunk_tokens is None
                and v24_execution_hint == "v22_medium_byte_exact_helpers"
            ):
                # Batch24 reproduced the Human-approved V22 MP4 byte for byte
                # while saving 9.49 seconds E2E at 67,535 packed tokens.  Keep
                # this helper stack bound to that exact V24 endpoint; the same
                # split/single path changed the rejected 720p15 V22 trajectory
                # and therefore must not be generalized by geometry alone.
                plan = replace(
                    plan,
                    long_sequence_query_chunk_tokens=32_768,
                    long_sequence_projection_chunk_tokens=8192,
                    long_sequence_split_qkv_outputs=True,
                    long_sequence_single_qknorm_rope=True,
                    long_sequence_parallel_sparse_lut=True,
                )
                chunk_reason = "v24_v22_medium_byte_exact_execution"
            if (
                plan.long_sequence_query_chunk_tokens is None
                and features.packed_tokens <= LONG_SEQUENCE_VALIDATED_MAX_PACKED_TOKENS
            ):
                chunk_decision = select_long_sequence_chunks(
                    video_tokens=features.video_tokens,
                    packed_tokens=features.packed_tokens,
                )
                if chunk_decision.query_chunk_tokens is not None:
                    plan = replace(
                        plan,
                        long_sequence_query_chunk_tokens=(
                            chunk_decision.query_chunk_tokens
                        ),
                        long_sequence_projection_chunk_tokens=(
                            chunk_decision.projection_chunk_tokens
                        ),
                        long_sequence_split_qkv_outputs=(
                            chunk_decision.split_qkv_outputs
                        ),
                        long_sequence_single_qknorm_rope=(
                            chunk_decision.single_qknorm_rope
                        ),
                        long_sequence_parallel_sparse_lut=(
                            chunk_decision.parallel_sparse_lut
                        ),
                    )
                    chunk_reason = chunk_decision.reason
            plan, chunk_reason, qk_override, memory_execution = (
                self._apply_memory_execution_policy(
                    request,
                    features,
                    plan,
                    chunk_reason=chunk_reason,
                    qk_override=qk_override,
                )
            )
            return plan, {
                "source": "explicit",
                "profile_id": None,
                "offload_mode": plan.offload_mode.value,
                "mlp_chunk_tokens": plan.mlp_chunk_tokens,
                "prefetch_depth": plan.prefetch_depth,
                "resident_block_count": plan.resident_block_count,
                "vae_spatial_tile": plan.vae_spatial_tile,
                "vae_transformer_block_compile": plan.vae_transformer_block_compile,
                "attention_topk": plan.attention_topk,
                "fused_rms_adaln": plan.fused_rms_adaln,
                "long_video_motion_detail_attention": (
                    plan.long_video_motion_detail_attention
                ),
                "long_sequence_query_chunk_tokens": (
                    plan.long_sequence_query_chunk_tokens
                ),
                "long_sequence_projection_chunk_tokens": (
                    plan.long_sequence_projection_chunk_tokens
                ),
                "long_sequence_split_qkv_outputs": (
                    plan.long_sequence_split_qkv_outputs
                ),
                "long_sequence_shared_qkv_quantization": (
                    plan.long_sequence_shared_qkv_quantization
                ),
                "long_sequence_compact_kv": plan.long_sequence_compact_kv,
                "long_sequence_exact_helper_stack": (
                    plan.long_sequence_exact_helper_stack
                ),
                "long_sequence_single_qknorm_rope": (
                    plan.long_sequence_single_qknorm_rope
                ),
                "long_sequence_parallel_sparse_lut": (
                    plan.long_sequence_parallel_sparse_lut
                ),
                "long_sequence_partial_sparse_topk": (
                    plan.long_sequence_partial_sparse_topk
                ),
                "long_sequence_fused_prefix_k_quant": (
                    plan.long_sequence_fused_prefix_k_quant
                ),
                "long_sequence_fused_query_projection": (
                    plan.long_sequence_fused_query_projection
                ),
                "long_sequence_fused_qknorm_hnd_layout": (
                    plan.long_sequence_fused_qknorm_hnd_layout
                ),
                "long_sequence_direct_nhd_output": (
                    plan.long_sequence_direct_nhd_output
                ),
                "long_sequence_direct_nhd_kv": plan.long_sequence_direct_nhd_kv,
                "long_sequence_direct_hnd_fp8_value": (
                    plan.long_sequence_direct_hnd_fp8_value
                ),
                "long_sequence_chunk_reason": chunk_reason,
                "dense_qk_quant_gran": plan.dense_qk_quant_gran,
                "dense_qk_quant_gran_requested": requested_qk,
                "dense_qk_stability_override": qk_override,
                "memory_execution": memory_execution,
                "frame_interleave_stride": plan.frame_interleave_stride,
                "frame_interleave_layer_start": plan.frame_interleave_layer_start,
                "frame_interleave_layer_stop": plan.frame_interleave_layer_stop,
                "frame_interleave_dense_layers": list(
                    plan.frame_interleave_dense_layers
                ),
                "frame_interleave_dense_steps": list(
                    plan.frame_interleave_dense_steps
                ),
                "spatial_query_lattice_stride": plan.spatial_query_lattice_stride,
                "spatial_query_lattice_layer_start": (
                    plan.spatial_query_lattice_layer_start
                ),
                "spatial_query_lattice_layer_stop": (
                    plan.spatial_query_lattice_layer_stop
                ),
                "spatial_query_lattice_dense_layers": list(
                    plan.spatial_query_lattice_dense_layers
                ),
                "spatial_query_lattice_dense_steps": list(
                    plan.spatial_query_lattice_dense_steps
                ),
                "mlp_spatial_lattice_stride": plan.mlp_spatial_lattice_stride,
                "mlp_spatial_lattice_layer_start": plan.mlp_spatial_lattice_layer_start,
                "mlp_spatial_lattice_layer_stop": plan.mlp_spatial_lattice_layer_stop,
                "mlp_spatial_lattice_dense_layers": list(
                    plan.mlp_spatial_lattice_dense_layers
                ),
                "mlp_spatial_lattice_dense_steps": list(
                    plan.mlp_spatial_lattice_dense_steps
                ),
                "mlp_spatial_lattice_detail_fraction": (
                    plan.mlp_spatial_lattice_detail_fraction
                ),
                "segment_cache_layer_start": plan.segment_cache_layer_start,
                "segment_cache_layer_stop": plan.segment_cache_layer_stop,
                "segment_cache_reuse_steps": list(plan.segment_cache_reuse_steps),
                "segment_cache_directional_trust": (
                    plan.segment_cache_directional_trust
                ),
                "segment_cache_directional_max_extra": (
                    plan.segment_cache_directional_max_extra
                ),
                "segment_cache_directional_min_cosine": (
                    plan.segment_cache_directional_min_cosine
                ),
                "segment_cache_protected_refresh": (
                    plan.segment_cache_protected_refresh
                ),
                "segment_cache_active_video_ratio": (
                    plan.segment_cache_active_video_ratio
                ),
                "segment_cache_dynamic_video_budget": (
                    plan.segment_cache_dynamic_video_budget
                ),
                "segment_cache_active_video_min_ratio": (
                    plan.segment_cache_active_video_min_ratio
                ),
                "segment_cache_innovation_risk_coverage": (
                    plan.segment_cache_innovation_risk_coverage
                ),
                "segment_cache_innovation_max_relative": (
                    plan.segment_cache_innovation_max_relative
                ),
                "segment_cache_active_layer_start": (
                    plan.segment_cache_active_layer_start
                ),
                "segment_cache_active_layer_stop": (
                    plan.segment_cache_active_layer_stop
                ),
                "segment_cache_sequential_layer_groups": (
                    plan.segment_cache_sequential_layer_groups
                ),
                "segment_cache_sequential_conservative_hold": (
                    plan.segment_cache_sequential_conservative_hold
                ),
                **feature_profile,
            }
        if self.planner is None:
            effective_qk, qk_override = select_stable_dense_qk_quantization(
                "per_thread",
                packed_tokens=features.packed_tokens,
            )
            return None, {
                "source": "legacy_default",
                "profile_id": None,
                "offload_mode": OffloadMode.RESIDENT.value,
                "mlp_chunk_tokens": request.mlp_chunk_tokens,
                "prefetch_depth": None,
                "resident_block_count": None,
                "attention_topk": None,
                "vae_transformer_block_compile": False,
                "fused_rms_adaln": False,
                "long_video_motion_detail_attention": False,
                "long_sequence_query_chunk_tokens": None,
                "long_sequence_projection_chunk_tokens": 8192,
                "long_sequence_split_qkv_outputs": False,
                "long_sequence_shared_qkv_quantization": False,
                "long_sequence_exact_helper_stack": False,
                "long_sequence_single_qknorm_rope": False,
                "long_sequence_parallel_sparse_lut": False,
                "long_sequence_partial_sparse_topk": False,
                "long_sequence_fused_prefix_k_quant": False,
                "long_sequence_fused_query_projection": False,
                "long_sequence_fused_qknorm_hnd_layout": False,
                "long_sequence_direct_nhd_output": False,
                "long_sequence_direct_nhd_kv": False,
                "long_sequence_direct_hnd_fp8_value": False,
                "long_sequence_chunk_reason": None,
                "dense_qk_quant_gran": effective_qk,
                "dense_qk_quant_gran_requested": "per_thread",
                "dense_qk_stability_override": qk_override,
                "frame_interleave_stride": 1,
                **feature_profile,
            }
        free_bytes, _ = torch.cuda.mem_get_info(self.runtime_config.device)
        # cudaMemGetInfo treats PyTorch's inactive cache as unavailable even
        # though subsequent model allocations can reuse it. Counting only raw
        # driver free memory makes the router unnecessarily choose Block mode
        # after Qwen. Add the reserved-but-unallocated cache back, while the
        # planner still applies its independent 1 GiB safety reserve.
        reusable_cache = max(
            0,
            torch.cuda.memory_reserved(self.runtime_config.device)
            - torch.cuda.memory_allocated(self.runtime_config.device),
        )
        effective_free = free_bytes + reusable_cache + 768 * 1024**2
        planner_source = "rtx4090_planner"
        planner_profile_id = None
        planner_predicted_seconds = None
        try:
            route_decision = self.planner.select(
                features,
                free_device_bytes=effective_free,
            )
            planner_profile_id = route_decision.profile_id
            planner_predicted_seconds = route_decision.predicted_seconds
            base_plan = route_decision.plan
        except NoFeasibleProfile:
            # The calibrated table ranks measured high-performance profiles.
            # A lack of such a profile is not a reason to skip the orthogonal
            # memory-execution policy. Start from the mature Block base, then
            # let performance/low_vram admission use the exact packed-token
            # budget below. Explicit performance still fails closed there if
            # its physical working set does not fit.
            planner_source = "memory_execution_fallback"
            base_plan = ExecutionPlan(
                offload_mode=OffloadMode.BLOCK,
                mlp_chunk_tokens=8192,
                block_buffer_count=2,
                prefetch_depth=1,
                vae_spatial_tile=(288, 288),
            )
        requested_qk = base_plan.dense_qk_quant_gran
        effective_qk, qk_override = select_stable_dense_qk_quantization(
            requested_qk,
            packed_tokens=features.packed_tokens,
        )
        plan = (
            base_plan
            if effective_qk == requested_qk
            else replace(base_plan, dense_qk_quant_gran=effective_qk)
        )
        chunk_reason = "profile_explicit" if (
            plan.long_sequence_query_chunk_tokens is not None
        ) else None
        if (
            plan.long_sequence_query_chunk_tokens is None
            and features.packed_tokens <= LONG_SEQUENCE_VALIDATED_MAX_PACKED_TOKENS
        ):
            chunk_decision = select_long_sequence_chunks(
                video_tokens=features.video_tokens,
                packed_tokens=features.packed_tokens,
            )
            if chunk_decision.query_chunk_tokens is not None:
                plan = replace(
                    plan,
                    long_sequence_query_chunk_tokens=(
                        chunk_decision.query_chunk_tokens
                    ),
                    long_sequence_projection_chunk_tokens=(
                        chunk_decision.projection_chunk_tokens
                    ),
                    long_sequence_split_qkv_outputs=(
                        chunk_decision.split_qkv_outputs
                    ),
                    long_sequence_single_qknorm_rope=(
                        chunk_decision.single_qknorm_rope
                    ),
                    long_sequence_parallel_sparse_lut=(
                        chunk_decision.parallel_sparse_lut
                    ),
                )
                chunk_reason = chunk_decision.reason
        plan, chunk_reason, qk_override, memory_execution = (
            self._apply_memory_execution_policy(
                request,
                features,
                plan,
                chunk_reason=chunk_reason,
                qk_override=qk_override,
            )
        )
        return plan, {
            "source": planner_source,
            "profile_id": planner_profile_id,
            "offload_mode": plan.offload_mode.value,
            "mlp_chunk_tokens": plan.mlp_chunk_tokens,
            "prefetch_depth": plan.prefetch_depth,
            "block_buffer_count": plan.block_buffer_count,
            "resident_block_count": plan.resident_block_count,
            "vae_spatial_tile": plan.vae_spatial_tile,
            "vae_temporal_tile": plan.vae_temporal_tile,
            "vae_transformer_block_compile": plan.vae_transformer_block_compile,
            "attention_topk": plan.attention_topk,
            "fused_rms_adaln": plan.fused_rms_adaln,
            "long_video_motion_detail_attention": (
                plan.long_video_motion_detail_attention
            ),
            "long_sequence_query_chunk_tokens": (
                plan.long_sequence_query_chunk_tokens
            ),
            "long_sequence_projection_chunk_tokens": (
                plan.long_sequence_projection_chunk_tokens
            ),
            "long_sequence_split_qkv_outputs": (
                plan.long_sequence_split_qkv_outputs
            ),
            "long_sequence_shared_qkv_quantization": (
                plan.long_sequence_shared_qkv_quantization
            ),
            "long_sequence_compact_kv": plan.long_sequence_compact_kv,
            "long_sequence_exact_helper_stack": (
                plan.long_sequence_exact_helper_stack
            ),
            "long_sequence_single_qknorm_rope": (
                plan.long_sequence_single_qknorm_rope
            ),
            "long_sequence_parallel_sparse_lut": (
                plan.long_sequence_parallel_sparse_lut
            ),
            "long_sequence_partial_sparse_topk": (
                plan.long_sequence_partial_sparse_topk
            ),
            "long_sequence_fused_prefix_k_quant": (
                plan.long_sequence_fused_prefix_k_quant
            ),
            "long_sequence_fused_query_projection": (
                plan.long_sequence_fused_query_projection
            ),
            "long_sequence_fused_qknorm_hnd_layout": (
                plan.long_sequence_fused_qknorm_hnd_layout
            ),
            "long_sequence_direct_nhd_output": (
                plan.long_sequence_direct_nhd_output
            ),
            "long_sequence_direct_nhd_kv": plan.long_sequence_direct_nhd_kv,
            "long_sequence_direct_hnd_fp8_value": (
                plan.long_sequence_direct_hnd_fp8_value
            ),
            "long_sequence_chunk_reason": chunk_reason,
            "dense_qk_quant_gran": plan.dense_qk_quant_gran,
            "dense_qk_quant_gran_requested": requested_qk,
            "dense_qk_stability_override": qk_override,
            "memory_execution": memory_execution,
            "frame_interleave_stride": plan.frame_interleave_stride,
            "frame_interleave_layer_start": plan.frame_interleave_layer_start,
            "frame_interleave_layer_stop": plan.frame_interleave_layer_stop,
            "frame_interleave_dense_layers": list(
                plan.frame_interleave_dense_layers
            ),
            "frame_interleave_dense_steps": list(
                plan.frame_interleave_dense_steps
            ),
            "spatial_query_lattice_stride": plan.spatial_query_lattice_stride,
            "spatial_query_lattice_layer_start": (
                plan.spatial_query_lattice_layer_start
            ),
            "spatial_query_lattice_layer_stop": (
                plan.spatial_query_lattice_layer_stop
            ),
            "spatial_query_lattice_dense_layers": list(
                plan.spatial_query_lattice_dense_layers
            ),
            "spatial_query_lattice_dense_steps": list(
                plan.spatial_query_lattice_dense_steps
            ),
            "mlp_spatial_lattice_stride": plan.mlp_spatial_lattice_stride,
            "mlp_spatial_lattice_layer_start": plan.mlp_spatial_lattice_layer_start,
            "mlp_spatial_lattice_layer_stop": plan.mlp_spatial_lattice_layer_stop,
            "mlp_spatial_lattice_dense_layers": list(
                plan.mlp_spatial_lattice_dense_layers
            ),
            "mlp_spatial_lattice_dense_steps": list(
                plan.mlp_spatial_lattice_dense_steps
            ),
            "mlp_spatial_lattice_detail_fraction": (
                plan.mlp_spatial_lattice_detail_fraction
            ),
            "segment_cache_layer_start": plan.segment_cache_layer_start,
            "segment_cache_layer_stop": plan.segment_cache_layer_stop,
            "segment_cache_reuse_steps": list(plan.segment_cache_reuse_steps),
            "segment_cache_directional_trust": plan.segment_cache_directional_trust,
            "segment_cache_directional_max_extra": (
                plan.segment_cache_directional_max_extra
            ),
            "segment_cache_directional_min_cosine": (
                plan.segment_cache_directional_min_cosine
            ),
            "segment_cache_protected_refresh": plan.segment_cache_protected_refresh,
            "segment_cache_active_video_ratio": (
                plan.segment_cache_active_video_ratio
            ),
            "segment_cache_dynamic_video_budget": (
                plan.segment_cache_dynamic_video_budget
            ),
            "segment_cache_active_video_min_ratio": (
                plan.segment_cache_active_video_min_ratio
            ),
            "segment_cache_innovation_risk_coverage": (
                plan.segment_cache_innovation_risk_coverage
            ),
            "segment_cache_innovation_max_relative": (
                plan.segment_cache_innovation_max_relative
            ),
            "segment_cache_active_layer_start": (
                plan.segment_cache_active_layer_start
            ),
            "segment_cache_active_layer_stop": (
                plan.segment_cache_active_layer_stop
            ),
            "segment_cache_sequential_layer_groups": (
                plan.segment_cache_sequential_layer_groups
            ),
            "segment_cache_sequential_conservative_hold": (
                plan.segment_cache_sequential_conservative_hold
            ),
            "predicted_seconds": planner_predicted_seconds,
            "predicted_peak_gib": memory_execution[
                "estimated_selected_peak_gib"
            ],
            **feature_profile,
            "driver_free_gib": free_bytes / (1024**3),
            "reusable_torch_cache_gib": reusable_cache / (1024**3),
            "effective_free_gib": effective_free / (1024**3),
        }

    def _activate_transformer(self, plan: ExecutionPlan | None) -> Any:
        mode = OffloadMode.RESIDENT if plan is None else plan.offload_mode
        self._clear_block_executor()
        if mode is OffloadMode.BLOCK:
            if plan is None or plan.block_buffer_count not in (1, 2):
                raise ValueError("H3 block offload requires a one- or two-buffer execution plan")
            block_count = len(self.transformer.value.block_stack.blocks)
            resident_count = plan.resident_block_count
            if resident_count >= block_count:
                raise ValueError(
                    "Block plan must leave at least one transformer block offloaded"
                )
            host_prefixes = tuple(
                f"block_stack.blocks.{index}"
                for index in range(resident_count, block_count)
            )
            self.transformer.move_partition_to_cuda(
                self.runtime_config.device,
                host_module_prefixes=host_prefixes,
            )
            dit = self.transformer.value
            config = replace(
                self.runtime_config,
                offload_mode=OffloadMode.BLOCK,
                block_buffer_count=plan.block_buffer_count,
            )
            executor = build_h3_block_executor(
                dit.block_stack.blocks[resident_count:],
                config,
                prefetch_depth=plan.prefetch_depth,
            )
            dit.block_stack.configure_block_executor(
                executor,
                offload_start=resident_count,
            )
            self._active_block_executor = executor
            return dit
        if mode not in (OffloadMode.RESIDENT, OffloadMode.MODEL):
            raise ValueError(f"unsupported transformer residency mode: {mode}")
        self.transformer.move_to(self.runtime_config.device, non_blocking=True)
        return self.transformer.value

    @staticmethod
    def _file_identity(path: Any) -> dict[str, Any] | None:
        if path is None:
            return None
        candidate = Path(path)
        try:
            stat = candidate.stat()
        except OSError:
            return {"name": candidate.name, "missing": True}
        return {
            "name": candidate.name,
            "size": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
        }

    def _conditioning_encoder_identity(self) -> dict[str, Any]:
        conditioner = self.conditioner
        return {
            "class": (
                f"{conditioner.__class__.__module__}."
                f"{conditioner.__class__.__qualname__}"
            ),
            "checkpoint": self._file_identity(
                getattr(conditioner, "checkpoint", None)
            ),
            "tokenizer": self._file_identity(
                Path(getattr(conditioner, "tokenizer_path", ""))
                / "tokenizer_config.json"
                if getattr(conditioner, "tokenizer_path", None)
                else None
            ),
            "layers": int(getattr(conditioner, "layers", 0)),
            "preprocess": QWEN_CONDITIONING_PREPROCESS_VERSION,
        }

    def _conditioning_fingerprint(self, request: HotSessionRequest) -> str:
        """Content-address one exact Qwen condition.

        Ref2VA images are proportionally capped independently of the output
        canvas, while standalone audio contributes stable label tokens only.
        Those static conditions therefore intentionally omit target geometry,
        allowing a 480p first pass to feed 1080p/1440p second sampling exactly.
        FL2VA keyframes and reference videos remain geometry/time addressed.
        """

        def digest(path: Path | None) -> str | None:
            return (
                None
                if path is None
                else self._cached_file_content_digest(Path(path)).hex()
            )

        geometry = None
        if request.first_frame is not None or request.last_frame is not None:
            geometry = [int(request.width), int(request.height), int(request.frames)]
        reference_video_clock = (
            int(request.frames) if request.reference_videos else None
        )
        document = {
            "schema_version": QWEN_CONDITIONING_CACHE_SCHEMA_VERSION,
            "service_family": (
                "reference" if self._uses_reference_layout else "first_last"
            ),
            "prompt_sha256": hashlib.sha256(
                request.prompt.encode("utf-8")
            ).hexdigest(),
            "first_frame": digest(request.first_frame),
            "last_frame": digest(request.last_frame),
            "reference_images": [
                digest(path) for path in request.reference_images
            ],
            "reference_videos": [
                digest(path) for path in request.reference_videos
            ],
            "reference_audios": [
                digest(path) for path in request.reference_audios
            ],
            "reference_image_resolution": request.reference_image_resolution,
            "reference_video_resolution": request.reference_video_resolution,
            "keyframe_geometry": geometry,
            "reference_video_clock": reference_video_clock,
            "encoder": self._conditioning_encoder_identity(),
        }
        canonical = json.dumps(
            document, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()

    def _host_conditioning_tensor(self, value: torch.Tensor) -> torch.Tensor:
        host = value.detach().to("cpu").contiguous()
        if (
            str(self.runtime_config.device).startswith("cuda")
            and torch.cuda.is_available()
            and not host.is_pinned()
        ):
            host = host.pin_memory()
        return host

    def _conditioning_payload(
        self,
        fingerprint: str,
        embeds: torch.Tensor,
        tags: torch.Tensor,
    ) -> dict[str, Any]:
        return {
            "schema_version": QWEN_CONDITIONING_CACHE_SCHEMA_VERSION,
            "fingerprint": fingerprint,
            "encoder": self._conditioning_encoder_identity(),
            "prompt_embeds": embeds,
            "text_token_tags": tags,
        }

    def _validated_conditioning_payload(
        self,
        payload: Any,
        *,
        fingerprint: str,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        if not isinstance(payload, dict):
            return None
        if payload.get("schema_version") != QWEN_CONDITIONING_CACHE_SCHEMA_VERSION:
            return None
        if payload.get("fingerprint") != fingerprint:
            return None
        embeds = payload.get("prompt_embeds")
        tags = payload.get("text_token_tags")
        if not isinstance(embeds, torch.Tensor) or not isinstance(tags, torch.Tensor):
            return None
        if embeds.ndim != 3 or embeds.shape[0] != 1 or embeds.shape[-1] != 5120:
            return None
        if tags.ndim != 1 or tags.shape[0] != embeds.shape[-2]:
            return None
        if not embeds.is_floating_point() or tags.dtype != torch.long:
            return None
        return (
            self._host_conditioning_tensor(embeds),
            self._host_conditioning_tensor(tags),
        )

    def _load_persisted_conditioning(
        self,
        request: HotSessionRequest,
        *,
        fingerprint: str,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        cached = self._persisted_conditioning_cache
        if cached is not None and cached[0] == fingerprint:
            self._last_conditioning_cache_status = "checkpoint_memory_hit"
            return cached[1], cached[2]
        source = request.conditioning_cache_source_path
        if source is None:
            return None
        try:
            checkpoint = torch.load(
                Path(source), map_location="cpu", weights_only=True
            )
            payload = (
                checkpoint.get("qwen_conditioning_cache")
                if isinstance(checkpoint, dict)
                else None
            )
            validated = self._validated_conditioning_payload(
                payload, fingerprint=fingerprint
            )
            del checkpoint
        except Exception as error:  # Legacy/corrupt cache must fail open.
            self._last_conditioning_cache_fallback = (
                f"checkpoint_read_{type(error).__name__}"
            )
            return None
        if validated is None:
            self._last_conditioning_cache_fallback = (
                "checkpoint_missing"
                if payload is None
                else "checkpoint_invalid_or_fingerprint_mismatch"
            )
            return None
        embeds, tags = validated
        self._persisted_conditioning_cache = (fingerprint, embeds, tags)
        self._last_conditioning_cache_status = "checkpoint_hit"
        return embeds, tags

    def _load_internal_conditioning_bridge(
        self,
        path: Path,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Load an engine-owned preceding-window Qwen condition exactly.

        The bridge path is created in the same private job directory as the
        current window cache. Its original request can have a different local
        frame count or dialogue-audio authority, so the consumer validates the
        stored encoder and tensor contract directly instead of fabricating a
        second public request merely to reproduce its fingerprint.
        """

        checkpoint = torch.load(Path(path), map_location="cpu", weights_only=True)
        payload = (
            checkpoint.get("qwen_conditioning_cache")
            if isinstance(checkpoint, dict)
            else None
        )
        if (
            not isinstance(payload, dict)
            or payload.get("encoder") != self._conditioning_encoder_identity()
            or not isinstance(payload.get("fingerprint"), str)
        ):
            raise ValueError("continuation text bridge cache is invalid")
        validated = self._validated_conditioning_payload(
            payload,
            fingerprint=payload["fingerprint"],
        )
        del checkpoint
        if validated is None:
            raise ValueError("continuation text bridge tensors are invalid")
        # Persisted conditioning is deliberately stored on the host.  The
        # ordinary request path moves a cache hit back to the runtime device
        # before DiT projection; the internal bridge must honor the same
        # contract.  Leaving only this second context on CPU fails at the
        # first continuation window when ``condition_proj`` is CUDA-resident.
        device = self.runtime_config.device
        return (
            validated[0].to(device, non_blocking=True),
            validated[1].to(device, non_blocking=True),
        )

    def _backfill_legacy_conditioning_cache(
        self,
        request: HotSessionRequest,
        payload: dict[str, Any],
    ) -> None:
        """Atomically upgrade one legacy latent after its one required encode.

        Only a truly missing cache is backfilled.  A present-but-mismatched
        payload may represent a deliberately different multimodal request and
        must never be overwritten implicitly.
        """

        source = request.conditioning_cache_source_path
        if (
            source is None
            or self._last_conditioning_cache_fallback != "checkpoint_missing"
        ):
            return
        source = Path(source).resolve()
        temporary = source.with_name(
            f".{source.name}.qwen-cache-{os.getpid()}-{time.time_ns()}.tmp"
        )
        try:
            checkpoint = torch.load(source, map_location="cpu", weights_only=True)
            if not isinstance(checkpoint, dict):
                return
            if checkpoint.get("qwen_conditioning_cache") is not None:
                return
            checkpoint["qwen_conditioning_cache"] = payload
            torch.save(checkpoint, temporary)
            os.replace(temporary, source)
        except Exception as error:  # Cache persistence cannot fail generation.
            self._last_conditioning_cache_fallback = (
                f"checkpoint_missing_backfill_{type(error).__name__}"
            )
        finally:
            temporary.unlink(missing_ok=True)

    def _encode_request(
        self, request: HotSessionRequest
    ) -> tuple[torch.Tensor, torch.Tensor]:
        has_frames = bool(
            request.reference_images
            or request.reference_videos
            or request.reference_audios
            or request.first_frame is not None
            or request.last_frame is not None
        )
        fingerprint = self._conditioning_fingerprint(request)
        self._last_conditioning_cache_payload = None
        self._last_conditioning_cache_status = "miss"
        self._last_conditioning_cache_fallback = None
        cached = self._conditioning_cache if has_frames else self._prompt_cache
        if cached is not None and cached[0] == fingerprint:
            host_embeds, host_tags = cached[1], cached[2]
            self._last_conditioning_cache_status = "hot_session_hit"
        else:
            persisted = self._load_persisted_conditioning(
                request, fingerprint=fingerprint
            )
            if persisted is not None:
                host_embeds, host_tags = persisted
            else:
                encoded = (
                    self.conditioner.encode_request(request)
                    if has_frames
                    else self.conditioner.encode_prompt(request.prompt)
                )
                embeds = encoded.prompt_embeds
                tags = encoded.text_token_tags
                host_embeds = self._host_conditioning_tensor(embeds)
                host_tags = self._host_conditioning_tensor(tags)
                self._last_conditioning_cache_status = "encoded"
                cached_entry = (fingerprint, host_embeds, host_tags)
                if has_frames:
                    self._conditioning_cache = cached_entry
                else:
                    self._prompt_cache = cached_entry
                self._last_conditioning_cache_payload = self._conditioning_payload(
                    fingerprint, host_embeds, host_tags
                )
                self._backfill_legacy_conditioning_cache(
                    request, self._last_conditioning_cache_payload
                )
                return embeds, tags
            cached_entry = (fingerprint, host_embeds, host_tags)
            if has_frames:
                self._conditioning_cache = cached_entry
            else:
                self._prompt_cache = cached_entry
        self._last_conditioning_cache_payload = self._conditioning_payload(
            fingerprint, host_embeds, host_tags
        )
        device = self.runtime_config.device
        return (
            host_embeds.to(device, non_blocking=True),
            host_tags.to(device, non_blocking=True),
        )

    @staticmethod
    def _file_content_digest(path: Path) -> bytes:
        return hashlib.sha256(Path(path).resolve().read_bytes()).digest()

    def _cached_file_content_digest(self, path: Path) -> bytes:
        resolved = Path(path).resolve()
        stat = resolved.stat()
        key = (str(resolved), int(stat.st_size), int(stat.st_mtime_ns))
        cache = getattr(self, "_media_digest_cache", None)
        if cache is None:
            cache = {}
            self._media_digest_cache = cache
        cached = cache.get(key)
        if cached is not None:
            return cached
        digest = self._file_content_digest(resolved)
        if len(cache) >= 64:
            cache.clear()
        cache[key] = digest
        return digest

    def _reference_latent_key(
        self, request: HotSessionRequest
    ) -> tuple[Any, ...]:
        """Content-address one deterministic Ref2VA VAE conditioning pack."""

        return (
            tuple(self._cached_file_content_digest(path) for path in request.reference_images),
            tuple(self._cached_file_content_digest(path) for path in request.reference_videos),
            tuple(self._cached_file_content_digest(path) for path in request.reference_audios),
            # Reference videos are capped/aligned against the requested frame
            # count.  Image and standalone-audio encodes are geometry agnostic.
            request.frames if request.reference_videos else None,
        )

    def _generate_impl(self, request: HotSessionRequest) -> HotSessionResult:
        request.validate()
        global_co_plan = (
            None
            if request.global_co_denoise_output_frames is None
            else plan_prompt_owned_global_av_windows(
                request.global_co_denoise_output_frames,
                request.global_co_denoise_prompt_ranges,
                window_frames=request.global_co_denoise_window_frames,
                stride_frames=request.global_co_denoise_stride_frames,
                balanced=request.global_co_denoise_balanced_windows,
            )
            if request.global_co_denoise_prompt_ranges
            else (
                plan_balanced_global_av_windows(
                    request.global_co_denoise_output_frames,
                    window_frames=request.global_co_denoise_window_frames,
                    stride_frames=request.global_co_denoise_stride_frames,
                )
                if request.global_co_denoise_balanced_windows
                else plan_global_av_windows(
                    request.global_co_denoise_output_frames,
                    window_frames=request.global_co_denoise_window_frames,
                    stride_frames=request.global_co_denoise_stride_frames,
                )
            )
        )
        final_output_frames = (
            request.frames
            if global_co_plan is None
            else global_co_plan.output_frames
        )
        from .model import set_lora_enabled

        def set_active_lora(enabled: bool) -> int:
            """Toggle source modules and any live block-offload device slots."""

            count = set_lora_enabled(self.transformer.value, enabled)
            executor = self._active_block_executor
            if executor is not None:
                for buffer in executor.buffers:
                    module = getattr(buffer, "module", None)
                    if module is not None:
                        count += set_lora_enabled(module, enabled)
            return count

        adapter_count = set_active_lora(request.use_lora)
        if request.use_lora and adapter_count == 0:
            raise RuntimeError("LoRA route requested but the hot family has no adapters")
        preview_lora_requested = (
            request.preview_decode_mode == "fast_finish"
            and (
                request.preview_branch_use_lora
                or request.preview_audio_branch_use_lora
            )
        )
        if preview_lora_requested and adapter_count == 0:
            raise RuntimeError("LoRA preview requested but the hot family has no adapters")
        active_lora_for_predict = bool(request.use_lora)
        primary_video_shift = (
            self.lora_video_shift if request.use_lora else 12.0
        )
        primary_audio_shift = (
            self.lora_audio_shift if request.use_lora else 3.0
        )
        request_engine = (
            "reference_lora" if self._uses_reference_layout and request.use_lora
            else "reference" if self._uses_reference_layout
            else "lora" if request.use_lora
            else "original"
        )
        cancel_check = request.cancel_check or (lambda: False)

        def progress(percent: float, stage: str, detail: str) -> None:
            if request.progress_callback is not None:
                request.progress_callback({
                    "percent": percent, "stage": stage, "detail": detail,
                })

        def raise_if_cancelled() -> None:
            if cancel_check():
                raise HotSessionCancelled("native H3 generation cancelled")

        raise_if_cancelled()
        output = request.output_path.resolve()
        if not output.is_relative_to(self.output_root):
            raise ValueError("output_path must stay inside output_root")
        output.parent.mkdir(parents=True, exist_ok=True)
        preview_output = (
            None
            if request.preview_output_path is None
            else Path(request.preview_output_path).resolve()
        )
        if preview_output is not None:
            if not preview_output.is_relative_to(self.output_root):
                raise ValueError("preview_output_path must stay inside output_root")
            preview_output.parent.mkdir(parents=True, exist_ok=True)
        preview_forecast_output = (
            None
            if request.preview_forecast_output_path is None
            else Path(request.preview_forecast_output_path).resolve()
        )
        if preview_forecast_output is not None:
            if not preview_forecast_output.is_relative_to(self.output_root):
                raise ValueError(
                    "preview_forecast_output_path must stay inside output_root"
                )
            preview_forecast_output.parent.mkdir(parents=True, exist_ok=True)
        phases: dict[str, float] = {}
        started_total = time.perf_counter()
        if request.reference_images or request.reference_videos or request.reference_audios:
            from .adapters.conditioning_vae.preprocess import prepare_reference_audios, prepare_reference_images, prepare_reference_videos

            progress(1, "reference_media", "解码参考媒体")
            prepared_images, prepared_videos, prepared_audios = self._timed(
                phases,
                "reference_media_prepare",
                lambda: (
                    prepare_reference_images(request) if request.reference_images else (),
                    prepare_reference_videos(request) if request.reference_videos else (),
                    prepare_reference_audios(request) if request.reference_audios else (),
                ),
            )
            request = replace(
                request,
                prepared_reference_images=prepared_images,
                prepared_reference_videos=prepared_videos,
                prepared_reference_audios=prepared_audios,
            )
            del prepared_images, prepared_videos, prepared_audios
        progress(3, "text", "理解提示词")
        vision_cache_hits_before = int(
            getattr(self.conditioner, "vision_cache_hits", 0)
        )

        context_5120, text_tags = self._timed(
            phases,
            "text_encode",
            lambda: self._encode_request(request),
        )
        global_context_inputs: list[tuple[torch.Tensor, torch.Tensor]] = [
            (context_5120, text_tags)
        ]
        continuation_text_bridge_index: int | None = None
        if request.continuation_text_bridge_conditioning_path is not None:
            bridge_context, bridge_tags = self._timed(
                phases,
                "text_encode_continuation_bridge",
                lambda: self._load_internal_conditioning_bridge(
                    request.continuation_text_bridge_conditioning_path
                ),
            )
            continuation_text_bridge_index = len(global_context_inputs)
            global_context_inputs.append((bridge_context, bridge_tags))
        if global_co_plan is not None:
            for index, (window, prompt, cache_path) in enumerate(
                zip(
                    global_co_plan.windows[1:],
                    request.global_co_denoise_prompts[1:],
                    request.global_co_denoise_conditioning_paths[1:],
                ),
                start=1,
            ):
                local_request = replace(
                    request,
                    prompt=prompt,
                    frames=window.frames,
                    conditioning_cache_source_path=cache_path,
                )
                local_context, local_tags = self._timed(
                    phases,
                    f"text_encode_window_{index:02d}",
                    lambda _request=local_request: self._encode_request(_request),
                )
                global_context_inputs.append((local_context, local_tags))
        progress(12, "text", "提示词编码完成")
        raise_if_cancelled()

        request = self._apply_v19_selection(
            request,
            text_tokens=max(
                int(item[0].shape[-2]) for item in global_context_inputs
            ),
        )

        execution_plan, execution_profile = self._resolve_execution_plan(
            request,
            text_tokens=max(
                int(item[0].shape[-2]) for item in global_context_inputs
            ),
        )
        execution_profile.update({
            "lora_profile": (
                {
                    "profile_id": self.lora_profile_id,
                    "recommended_steps": list(self.lora_recommended_steps),
                    "default_steps": self.lora_default_steps,
                    "requested_steps": request.steps,
                    "video_shift": self.lora_video_shift,
                    "audio_shift": self.lora_audio_shift,
                    "clock_mode": self.turbo_clock_mode.value,
                }
                if request.use_lora else None
            ),
            "qwen_pinned_weight_cache": bool(
                getattr(self.conditioner, "host_cache_ready", False)
            ),
            "qwen_layer_streaming_cache": bool(
                getattr(self.conditioner, "layer_cache_dir", None)
            ),
            "qwen_vision_feature_cache_hit": int(
                getattr(self.conditioner, "vision_cache_hits", 0)
            ) > vision_cache_hits_before,
            "qwen_vision_feature_cache_mib": round(
                int(getattr(self.conditioner, "vision_feature_cache_bytes", 0))
                / (1024**2),
                3,
            ),
            "qwen_conditioning_cache": {
                "schema_version": QWEN_CONDITIONING_CACHE_SCHEMA_VERSION,
                "status": self._last_conditioning_cache_status,
                "fallback": self._last_conditioning_cache_fallback,
                "persisted_with_latent": bool(
                    self._last_conditioning_cache_payload is not None
                    and request.save_final_latents_path is not None
                    and not request.latent_only
                ),
            },
        })
        if request.acceleration_plan_summary is not None:
            execution_profile["joint_acceleration"] = dict(
                request.acceleration_plan_summary
            )
        if global_co_plan is not None:
            execution_profile["global_co_denoise"] = {
                **global_co_plan.telemetry(),
                "prediction_space": "denoised_x0",
                "scheduler_updates_per_step": 1,
                "window_forwards_per_step": len(global_co_plan.windows),
                "forecast_enabled": False,
                "conditioning": "deterministic_window_views",
                "rotary_time": request.global_co_denoise_rotary_mode,
            }

        condition_latents_cpu: tuple[torch.Tensor, ...] = ()
        condition_audio_latents_cpu: tuple[torch.Tensor, ...] = ()
        keyframe_indices: tuple[int, ...] = ()
        reference_shapes: tuple[tuple[int, int, int], ...] = ()
        reference_kinds: tuple[str, ...] = ()
        reference_audio_frames: tuple[int, ...] = ()
        conditioning_routes_cpu: dict[str, _ConditioningComposition] = {}
        selflift_target_keyframe_latents_cpu: tuple[torch.Tensor, ...] = ()
        conditioning_authority_profile: dict[str, Any] | None = None
        external_refinement_video_cpu = None
        external_refinement_video_path = (
            Path(request.external_refinement_video_path).expanduser().resolve()
            if request.external_refinement_video_path is not None
            else None
        )
        if external_refinement_video_path is None and request.refinement_latents_path is not None:
            raw_external_refinement_video = os.environ.get(
                "H3_SECOND_SAMPLING_EXTERNAL_VIDEO_SOURCE", ""
            ).strip()
            if raw_external_refinement_video:
                external_refinement_video_path = Path(
                    raw_external_refinement_video
                ).expanduser().resolve()
        if (
            external_refinement_video_path is not None
            and not external_refinement_video_path.is_file()
        ):
            raise ValueError(
                "H3 external refinement video source does not exist: "
                f"{external_refinement_video_path}"
            )
        state_seeded_layout_memory = False
        token_memory = None
        if request.av_token_memory_path is not None:
            from .av_token_memory import validate_av_token_memory

            token_memory = validate_av_token_memory(
                torch.load(
                    Path(request.av_token_memory_path),
                    map_location="cpu",
                    weights_only=True,
                )
            )
        has_user_references = bool(
            request.reference_images
            or request.reference_videos
            or request.reference_audios
        )
        has_user_keyframes = bool(
            request.first_frame is not None or request.last_frame is not None
        )
        if has_user_references or has_user_keyframes or token_memory is not None:
            is_reference = bool(has_user_references or token_memory is not None)
            progress(13, "conditioning", "编码参考媒体条件" if is_reference else "编码首尾帧条件")
            if (request.reference_images or request.reference_videos or request.first_frame is not None or request.last_frame is not None) and self.encode_video_conditioning is None:
                raise RuntimeError(
                    "this Native H3 session has no Video-VAE condition encoder"
                )
            user_video_latents: tuple[torch.Tensor, ...] = ()
            user_reference_shapes: tuple[tuple[int, int, int], ...] = ()
            user_reference_kinds: tuple[str, ...] = ()
            user_audio_latents: tuple[torch.Tensor, ...] = ()
            user_audio_frames: tuple[int, ...] = ()
            keyframe_latents: tuple[torch.Tensor, ...] = ()
            cached_reference = None
            reference_cache_key = None
            if has_user_references and request.cache_reference_latents:
                reference_cache_key = self._reference_latent_key(request)
                candidate = self._reference_latent_cache
                if candidate is not None and candidate.key == reference_cache_key:
                    cached_reference = candidate
            execution_profile["reference_latent_cache_hit"] = bool(cached_reference)
            if cached_reference is not None:
                user_video_latents = cached_reference.video_latents
                user_reference_shapes = cached_reference.video_shapes
                user_reference_kinds = cached_reference.video_kinds
                user_audio_latents = cached_reference.audio_latents
                user_audio_frames = cached_reference.audio_frames
            elif request.reference_images or request.reference_videos:
                self._timed(phases, "condition_vae_h2d", lambda: self.video_vae.move_to("cuda:0", non_blocking=True))
                frame_conditioning = self._timed(
                    phases, "condition_video_encode",
                    lambda: self.encode_video_conditioning(self.video_vae.value, request),
                )
                user_video_latents = tuple(
                    latent.detach().to("cpu")
                    for latent in frame_conditioning.latents
                )
                user_reference_shapes = tuple(frame_conditioning.latent_shapes)
                user_reference_kinds = tuple(frame_conditioning.kinds)
                self._timed(phases, "condition_vae_evict", lambda: self.video_vae.move_to("cpu", non_blocking=False))
                self._release_device()
            elif has_user_keyframes:
                self._timed(phases, "condition_vae_h2d", lambda: self.video_vae.move_to("cuda:0", non_blocking=True))
                frame_conditioning = self._timed(
                    phases, "condition_video_encode",
                    lambda: self.encode_video_conditioning(self.video_vae.value, request),
                )
                encoded_keyframe_latents = tuple(
                    item.latent.detach().to("cpu")
                    for item in frame_conditioning.keyframes
                )
                if request.multiscale_initial_width is not None:
                    assert request.multiscale_initial_height is not None
                    selflift_target_keyframe_latents_cpu = (
                        encoded_keyframe_latents
                    )
                    keyframe_latents = tuple(
                        resize_refinement_video_latent_spatial(
                            latent,
                            target_height=(
                                request.multiscale_initial_height // 16
                            ),
                            target_width=(
                                request.multiscale_initial_width // 16
                            ),
                        )
                        for latent in encoded_keyframe_latents
                    )
                else:
                    keyframe_latents = encoded_keyframe_latents
                keyframe_indices = tuple(
                    int(item.semantic_frame_index)
                    for item in frame_conditioning.keyframes
                )
                self._timed(phases, "condition_vae_evict", lambda: self.video_vae.move_to("cpu", non_blocking=False))
                self._release_device()
            if request.reference_audios and cached_reference is None:
                if self.encode_audio_conditioning is None:
                    raise RuntimeError("this Native H3 session has no Audio-VAE condition encoder")
                self._timed(phases, "condition_audio_vae_h2d", lambda: self.audio_vae.move_to("cuda:0", non_blocking=True))
                audio_conditions = self._timed(
                    phases, "condition_audio_encode",
                    lambda: self.encode_audio_conditioning(self.audio_vae.value, request),
                )
                user_audio_latents = tuple(
                    latent.detach().to("cpu") for latent in audio_conditions
                )
                user_audio_frames = tuple(
                    int(latent.shape[-1]) for latent in user_audio_latents
                )
                self._timed(phases, "condition_audio_vae_evict", lambda: self.audio_vae.move_to("cpu", non_blocking=False))
                self._release_device()
            if (
                has_user_references
                and request.cache_reference_latents
                and cached_reference is None
            ):
                assert reference_cache_key is not None
                self._reference_latent_cache = _ReferenceLatentCacheEntry(
                    key=reference_cache_key,
                    video_latents=user_video_latents,
                    video_shapes=user_reference_shapes,
                    video_kinds=user_reference_kinds,
                    audio_latents=user_audio_latents,
                    audio_frames=user_audio_frames,
                )
            memory_video_latents: tuple[torch.Tensor, ...] = ()
            memory_reference_shapes: tuple[tuple[int, int, int], ...] = ()
            memory_reference_kinds: tuple[str, ...] = ()
            memory_layout_video_latents: tuple[torch.Tensor, ...] = ()
            memory_layout_reference_shapes: tuple[tuple[int, int, int], ...] = ()
            memory_layout_reference_kinds: tuple[str, ...] = ()
            memory_audio_latents: tuple[torch.Tensor, ...] = ()
            memory_audio_frames: tuple[int, ...] = ()
            novel_camera_layout_probe_memory = False
            if token_memory is not None:
                from .av_token_memory import (
                    memory_conditioning,
                    token_memory_telemetry,
                )

                (
                    memory_video_latents,
                    memory_reference_shapes,
                    memory_reference_kinds,
                    memory_audio_latents,
                    memory_audio_frames,
                ) = memory_conditioning(token_memory)
                visual_route = token_memory.get("visual_route", {})
                state_seeded_layout_memory = bool(
                    visual_route.get("policy")
                    == "structured_director_terminal_camera_band_state_v3"
                )
                novel_camera_layout_probe_memory = bool(
                    visual_route.get("policy")
                    == "structured_director_novel_camera_layout_probe_v1"
                )
                if bool(visual_route.get("progressive_layout_state", False)):
                    layout_positions = visual_route.get("layout_positions")
                    if layout_positions is None:
                        canonical_position = visual_route.get("canonical_position")
                        layout_positions = (
                            [] if canonical_position is None else [canonical_position]
                        )
                    layout_position_set = {
                        int(position) for position in layout_positions
                    }
                    positions = tuple(
                        int(item["position"])
                        for item in token_memory["video_entries"]
                    )
                    layout_indices = tuple(
                        index
                        for index, position in enumerate(positions)
                        if position in layout_position_set
                    )
                    state_indices = tuple(
                        index
                        for index in range(len(positions))
                        if index not in layout_indices
                    )
                    if not layout_indices or (
                        not state_indices
                        and not novel_camera_layout_probe_memory
                    ):
                        raise RuntimeError(
                            "progressive visual memory requires at least one layout "
                            "anchor and at least one recent state anchor"
                        )
                    memory_layout_video_latents = tuple(
                        memory_video_latents[index] for index in layout_indices
                    )
                    memory_layout_reference_shapes = tuple(
                        memory_reference_shapes[index] for index in layout_indices
                    )
                    memory_layout_reference_kinds = tuple(
                        memory_reference_kinds[index] for index in layout_indices
                    )
                    memory_video_latents = tuple(
                        memory_video_latents[index] for index in state_indices
                    )
                    memory_reference_shapes = tuple(
                        memory_reference_shapes[index] for index in state_indices
                    )
                    memory_reference_kinds = tuple(
                        memory_reference_kinds[index] for index in state_indices
                    )
                execution_profile["av_token_memory"] = {
                    **token_memory_telemetry(token_memory),
                    "injection": "native_reference_style_packed_tokens_v1",
                    "qwen_text_summary": False,
                }

            (
                conditioning_routes_cpu,
                conditioning_authority_profile,
            ) = _compose_authority_routed_conditioning(
                user_video_latents=user_video_latents,
                user_reference_shapes=user_reference_shapes,
                user_reference_kinds=user_reference_kinds,
                user_audio_latents=user_audio_latents,
                user_audio_frames=user_audio_frames,
                memory_video_latents=memory_video_latents,
                memory_reference_shapes=memory_reference_shapes,
                memory_reference_kinds=memory_reference_kinds,
                memory_audio_latents=memory_audio_latents,
                memory_audio_frames=memory_audio_frames,
                keyframe_latents=keyframe_latents,
                keyframe_indices=keyframe_indices,
                user_references_requested=has_user_references,
                supports_persistent_inferred_audio=(
                    self._uses_reference_layout
                ),
                memory_layout_video_latents=memory_layout_video_latents,
                memory_layout_reference_shapes=(
                    memory_layout_reference_shapes
                ),
                memory_layout_reference_kinds=memory_layout_reference_kinds,
                state_seeded_layout_memory=state_seeded_layout_memory,
                novel_camera_layout_probe=novel_camera_layout_probe_memory,
            )
            # Routes are inserted with their high-noise variant first.  This
            # initial value is replaced by the explicit per-step selector
            # below, but keeping it valid also covers setup code that inspects
            # condition geometry before the first DiT call.
            composition = next(iter(conditioning_routes_cpu.values()), None)
            assert composition is not None
            condition_latents_cpu = composition.video_latents
            reference_shapes = composition.reference_shapes
            reference_kinds = composition.reference_kinds
            condition_audio_latents_cpu = composition.audio_latents
            reference_audio_frames = composition.reference_audio_frames
            execution_profile["conditioning_composition"] = composition.profile
            execution_profile["conditioning_authority"] = (
                conditioning_authority_profile
            )
            request = replace(request, prepared_reference_images=(), prepared_reference_videos=(), prepared_reference_audios=())
            progress(17, "conditioning", "参考媒体条件完成" if is_reference else "首尾帧条件完成")
            raise_if_cancelled()
        if external_refinement_video_path is not None:
            if self.encode_video_conditioning is None:
                raise RuntimeError(
                    "this Native H3 session has no Video-VAE condition encoder"
                )
            progress(17, "conditioning", "重编码外部二采视频")
            external_request = replace(
                request,
                first_frame=None,
                last_frame=None,
                reference_images=(),
                reference_videos=(external_refinement_video_path,),
                reference_audios=(),
                reference_video_resolution="original",
                prepared_reference_images=(),
                prepared_reference_videos=(),
                prepared_reference_audios=(),
            )
            self._timed(
                phases,
                "external_refinement_vae_h2d",
                lambda: self.video_vae.move_to("cuda:0", non_blocking=True),
            )
            external_conditioning = self._timed(
                phases,
                "external_refinement_video_encode",
                lambda: self.encode_video_conditioning(
                    self.video_vae.value, external_request
                ),
            )
            external_latents = tuple(external_conditioning.latents)
            if len(external_latents) != 1:
                raise RuntimeError(
                    "external H3 refinement requires exactly one encoded video"
                )
            external_refinement_video_cpu = external_latents[0].detach().to(
                device="cpu", dtype=torch.float32
            )
            self._timed(
                phases,
                "external_refinement_vae_evict",
                lambda: self.video_vae.move_to("cpu", non_blocking=False),
            )
            self._release_device()
            execution_profile["external_refinement_video"] = {
                "mode": "full_video_h3_vae_reencode_v1",
                "path": str(external_refinement_video_path),
                "latent_shape": list(external_refinement_video_cpu.shape),
            }
            raise_if_cancelled()
        del token_memory

        if (
            getattr(
                getattr(self, "runtime_config", None),
                "resource_profile",
                None,
            )
            == "w4a8_8gb"
        ):
            # Qwen/vision conditioning and DiT use very different temporary
            # shapes.  Keeping Qwen's inactive CUDA slabs can leave 1-2 GiB
            # reserved but unusable for a long-video DiT workspace, causing a
            # false OOM under the intentional 7.25 GiB process ceiling.  The
            # projected context remains live; only allocator slack is dropped
            # at this natural phase boundary.  Larger tiers retain their hot
            # slabs for maximum throughput.
            self._timed(
                phases,
                "pre_dit_allocator_compact",
                self._release_device,
            )
        progress(18, "denoise", "载入 DiT 计算阶段")
        dit = self._timed(
            phases,
            "dit_h2d",
            lambda: self._activate_transformer(execution_plan),
        )
        projected_contexts: list[torch.Tensor] = []
        projected_text_tags: list[torch.Tensor] = []
        with torch.inference_mode():
            for index, (raw_context, raw_tags) in enumerate(global_context_inputs):
                projected_contexts.append(
                    self._timed(
                        phases,
                        (
                            "condition_projection"
                            if index == 0
                            else f"condition_projection_window_{index:02d}"
                        ),
                        lambda _raw=raw_context: dit.token_refiner(
                            dit.condition_proj(_raw[0].to(dit.compute_dtype))
                        ).unsqueeze(0),
                    )
                )
                projected_text_tags.append(raw_tags)
        context = projected_contexts[0]
        text_tags = projected_text_tags[0]
        continuation_text_bridge = (
            None
            if continuation_text_bridge_index is None
            else (
                projected_contexts[continuation_text_bridge_index],
                projected_text_tags[continuation_text_bridge_index],
            )
        )
        del context_5120, global_context_inputs
        conditioning_routes: dict[str, _ConditioningComposition] = {}
        if conditioning_routes_cpu:
            device_condition_cache: dict[int, torch.Tensor] = {}

            def move_condition(latent: torch.Tensor) -> torch.Tensor:
                key = id(latent)
                value = device_condition_cache.get(key)
                if value is None:
                    value = latent.to("cuda:0", non_blocking=False)
                    device_condition_cache[key] = value
                return value

            for route_name, route in conditioning_routes_cpu.items():
                conditioning_routes[route_name] = _ConditioningComposition(
                    video_latents=tuple(
                        move_condition(latent) for latent in route.video_latents
                    ),
                    reference_shapes=route.reference_shapes,
                    reference_kinds=route.reference_kinds,
                    audio_latents=tuple(
                        move_condition(latent) for latent in route.audio_latents
                    ),
                    reference_audio_frames=route.reference_audio_frames,
                    profile=route.profile,
                )
            initial_route_name = next(iter(conditioning_routes))
            initial_conditioning = conditioning_routes[initial_route_name]
            condition_video_latents = initial_conditioning.video_latents
            condition_audio_latents = initial_conditioning.audio_latents
            reference_shapes = initial_conditioning.reference_shapes
            reference_kinds = initial_conditioning.reference_kinds
            reference_audio_frames = initial_conditioning.reference_audio_frames
            del device_condition_cache
        else:
            condition_video_latents = tuple(
                latent.to("cuda:0", non_blocking=False)
                for latent in condition_latents_cpu
            )
            condition_audio_latents = tuple(
                latent.to("cuda:0", non_blocking=False)
                for latent in condition_audio_latents_cpu
            )
        del condition_latents_cpu, condition_audio_latents_cpu
        conditioning_routes_cpu = {}
        attention_backend = dit.block_stack.blocks[0].attention.backend
        request_attention_schedule = (
            {
                (int(step), int(layer)): str(action)
                for step, layer, action in request.attention_action_schedule
            }
            if request.attention_action_schedule
            else None
        )
        request_online_budget = (
            AttentionOnlineBudget(
                policy_id=request.attention_online_guard_id,
                limit_dense_layers=request.attention_online_budget_dense_layers,
                rebate_schedule=request.attention_online_rebate_schedule,
            )
            if request.attention_online_guard_id is not None
            else None
        )
        selected_attention_topk = (
            None if execution_plan is None else execution_plan.attention_topk
        )
        sparse_scope = (
            "full" if execution_plan is None else execution_plan.sparse_scope
        )

        def step_attention_topk(step_index: int) -> float | None:
            """Apply the user budget only inside its requested quality guard."""

            if selected_attention_topk is None:
                return None
            if sparse_scope == "full":
                return selected_attention_topk
            if sparse_scope == "guarded":
                # The first two and final two solver points retain dense
                # attention because they carry coarse layout and convergence.
                return (
                    None
                    if step_index < 2 or step_index >= request.steps - 2
                    else selected_attention_topk
                )
            # Conservative mode uses sparse attention only in the central
            # half of the trajectory (inclusive start, exclusive stop).
            start = request.steps // 4
            stop = request.steps - start
            return selected_attention_topk if start <= step_index < stop else None
        guard_approximate_math = bool(
            selected_attention_topk is not None
            if getattr(attention_backend, "request_routed", False)
            else getattr(attention_backend, "approximate", False)
        ) or bool(
            execution_plan is not None and execution_plan.fused_rms_adaln
        ) or bool(
            execution_plan is not None
            and execution_plan.frame_interleave_stride > 1
        ) or bool(
            execution_plan is not None
            and execution_plan.spatial_query_lattice_stride > 1
        ) or bool(
            execution_plan is not None
            and execution_plan.segment_cache_reuse_steps
        )

        duration = final_output_frames / float(request.fps)
        latent_video_tokens = (
            global_co_plan.video_tokens
            if global_co_plan is not None
            else int(request.internal_video_tokens)
            if request.internal_video_tokens is not None
            else ((request.frames - 5) // 17) * 5 + 2
        )
        video_shape = (
            1,
            24,
            latent_video_tokens,
            request.height // 16,
            request.width // 16,
        )
        initial_video_shape = video_shape
        if request.multiscale_initial_width is not None:
            assert request.multiscale_initial_height is not None
            initial_video_shape = (
                video_shape[0],
                video_shape[1],
                video_shape[2],
                request.multiscale_initial_height // 16,
                request.multiscale_initial_width // 16,
            )
        elif request.terminal_refinement_initial_width is not None:
            assert request.terminal_refinement_initial_height is not None
            initial_video_shape = (
                video_shape[0],
                video_shape[1],
                video_shape[2],
                request.terminal_refinement_initial_height // 16,
                request.terminal_refinement_initial_width // 16,
            )
        audio_shape = (
            1,
            32,
            2,
            global_co_plan.audio_tokens
            if global_co_plan is not None
            else int(request.internal_audio_tokens)
            if request.internal_audio_tokens is not None
            else round(duration * 40),
        )
        generator = torch.Generator("cpu").manual_seed(request.seed)
        video_noise_cpu = torch.randn(
            initial_video_shape, generator=generator, dtype=torch.float32
        )
        audio_noise_cpu = torch.randn(
            audio_shape, generator=generator, dtype=torch.float32
        )
        multiscale_highpass_cpu = None
        selflift_target_noise_cpu = None
        if request.multiscale_initial_width is not None:
            if request.multiscale_transition_mode == "selflift_learned_x0":
                # SelfLift draws a fresh target-grid noise realization for the
                # lifted branch.  Keep it independent from the low-resolution
                # video/audio draw order, matching the upstream seed+1 rule.
                selflift_generator = torch.Generator("cpu").manual_seed(
                    (int(request.seed) + 1) % (2**63 - 1)
                )
                selflift_target_noise_cpu = torch.randn(
                    video_shape,
                    generator=selflift_generator,
                    dtype=torch.float32,
                )
            else:
                target_noise = torch.randn(
                    video_shape, generator=generator, dtype=torch.float32
                )
                multiscale_highpass_cpu = spatial_highpass_noise(
                    target_noise,
                    low_height=initial_video_shape[-2],
                    low_width=initial_video_shape[-1],
                )
                del target_noise
        terminal_video_noise_cpu = None
        terminal_audio_noise_cpu = None
        if request.terminal_refinement_initial_width is not None:
            terminal_video_noise_cpu = torch.randn(
                video_shape, generator=generator, dtype=torch.float32
            )
            terminal_audio_noise_cpu = torch.randn(
                audio_shape, generator=generator, dtype=torch.float32
            )
        preserved_refinement_audio = None
        # Online SelfLift preview finalization may preserve audio that already
        # completed its disposable branch. JSON/direct generation instead
        # resumes the exact formal audio sampler state and lets the remaining
        # solver steps finish it together with the lifted video.
        preserved_global_selflift_audio = None
        global_selflift_anchor_video = None
        refinement_motion_video = None
        refinement_initialization_profile = "native_geometry"
        refinement_full_canvas_mask = None
        refinement_sampler_mask = None
        refinement_sampler_noise = None
        refinement_handoff_latent_frames = 0
        refinement_previous_video_denoised = None
        refinement_source_latent_height = None
        refinement_source_latent_width = None
        refinement_audio_sigmas = None
        refinement_first_anchor_clean = None
        refinement_first_anchor_noise = None
        refinement_first_anchor_weights = None
        resume_previous_video = None
        resume_previous_audio = None
        resume_previous_video_sigma = None
        resume_previous_audio_sigma = None
        resume_step_offset = 0
        resume_forecast_state = None
        resume_protected_video_prefix = None
        resume_protected_audio_prefix = None
        continuation_video_prefix = None
        continuation_audio_prefix = None
        # A direct SelfLift continuation has two equally important views of
        # the overlap.  The small view drives the inexpensive first-pass
        # trajectory; the previous window's target-grid values remain the
        # visual authority once the trajectory is lifted.  Keeping this CPU
        # copy avoids both a second low-resolution solve and the historical
        # target -> source -> target loss at every physical seam.
        continuation_target_video_prefix_cpu = None
        if request.formal_resume_state_path is not None:
            checkpoint = torch.load(
                Path(request.formal_resume_state_path),
                map_location="cpu",
                weights_only=True,
            )
            expected_metadata = {
                "frames": request.frames,
                "fps": request.fps,
                "width": request.width,
                "height": request.height,
                "engine": request_engine,
                "seed": request.seed,
                "steps": request.steps,
                "representation": "formal_sampler_checkpoint_v1",
            }
            if request.use_lora:
                expected_metadata["lora_profile_id"] = self.lora_profile_id
            for key, expected in expected_metadata.items():
                if checkpoint.get(key) != expected:
                    raise ValueError(
                        "formal checkpoint metadata mismatch for "
                        f"{key}: expected {expected!r}, got {checkpoint.get(key)!r}"
                    )
            if checkpoint.get("use_lora") is not request.use_lora:
                raise ValueError("formal checkpoint model variant does not match the request")
            prompt_digest = hashlib.sha256(request.prompt.encode("utf-8")).hexdigest()
            if checkpoint.get("prompt_sha256") != prompt_digest:
                raise ValueError("formal checkpoint prompt does not match the request")
            resume_step_offset = int(checkpoint.get("next_step_index", -1))
            if not 1 <= resume_step_offset < request.steps:
                raise ValueError("formal checkpoint has an invalid next step")
            full_sigmas = simple_sigma_schedule(
                request.steps, primary_video_shift
            )
            recorded_sigmas = tuple(float(value) for value in checkpoint.get("sigmas", ()))
            if recorded_sigmas != full_sigmas:
                raise ValueError("formal checkpoint sigma schedule does not match the request")
            recorded_actual = tuple(
                int(value) for value in checkpoint.get("actual_step_indices", ())
            )
            requested_actual = (
                tuple(range(request.steps))
                if request.actual_step_indices is None
                else request.actual_step_indices
            )
            if recorded_actual != requested_actual:
                raise ValueError("formal checkpoint actual-step schedule does not match")
            recorded_attention_schedule = tuple(
                (int(step), int(layer), str(action))
                for step, layer, action in checkpoint.get(
                    "attention_action_schedule", ()
                )
            )
            if recorded_attention_schedule != request.attention_action_schedule:
                raise ValueError(
                    "formal checkpoint attention schedule does not match"
                )
            if checkpoint.get("attention_online_guard_id") != request.attention_online_guard_id:
                raise ValueError("formal checkpoint online guard does not match")
            recorded_rebate_schedule = tuple(
                (int(step), int(layer))
                for step, layer in checkpoint.get(
                    "attention_online_rebate_schedule", ()
                )
            )
            if recorded_rebate_schedule != request.attention_online_rebate_schedule:
                raise ValueError(
                    "formal checkpoint online rebate schedule does not match"
                )
            if not math.isclose(
                float(checkpoint.get("attention_online_budget_dense_layers", 0.0)),
                request.attention_online_budget_dense_layers,
                rel_tol=0.0,
                abs_tol=1e-9,
            ):
                raise ValueError("formal checkpoint online budget does not match")
            recorded_online_state = checkpoint.get("attention_online_runtime_state")
            if request_online_budget is None:
                if recorded_online_state is not None:
                    raise ValueError(
                        "formal checkpoint unexpectedly contains online runtime state"
                    )
            else:
                if not isinstance(recorded_online_state, dict):
                    raise ValueError(
                        "formal checkpoint is missing online runtime state"
                    )
                request_online_budget.restore_checkpoint_state(
                    recorded_online_state.get("budget")
                )
                restore_verifier = getattr(
                    attention_backend,
                    "restore_online_checkpoint_state",
                    None,
                )
                if restore_verifier is None:
                    raise ValueError(
                        "active attention backend cannot restore online state"
                    )
                restore_verifier(
                    request_online_budget,
                    recorded_online_state.get("verifier"),
                )
            expected_resume_video_shape = initial_video_shape
            if (
                request.multiscale_resize_after_step is not None
                and resume_step_offset > request.multiscale_resize_after_step
            ):
                expected_resume_video_shape = video_shape
            for key, expected_shape in (
                ("video", expected_resume_video_shape),
                ("audio", audio_shape),
            ):
                value = checkpoint.get(key)
                if not isinstance(value, torch.Tensor) or tuple(value.shape) != expected_shape:
                    raise ValueError(f"formal checkpoint has an invalid {key} latent")
            video = checkpoint["video"].cuda()
            audio = checkpoint["audio"].cuda()
            protected_video = checkpoint.get("protected_video_prefix")
            protected_audio = checkpoint.get("protected_audio_prefix")
            if protected_video is not None or protected_audio is not None:
                if (
                    not isinstance(protected_video, torch.Tensor)
                    or protected_video.ndim != 5
                    or protected_video.shape[:2] != video.shape[:2]
                    or protected_video.shape[3:] != video.shape[3:]
                ):
                    raise ValueError(
                        "formal checkpoint has an invalid protected video prefix"
                    )
                if (
                    not isinstance(protected_audio, torch.Tensor)
                    or protected_audio.ndim != 4
                    or protected_audio.shape[:-1] != audio.shape[:-1]
                ):
                    raise ValueError(
                        "formal checkpoint has an invalid protected audio prefix"
                    )
                resume_protected_video_prefix = protected_video.cuda()
                resume_protected_audio_prefix = protected_audio.cuda()
            sigmas = full_sigmas[resume_step_offset:]
            if not request.use_lora:
                for key, expected_shape in (
                    ("previous_video", expected_resume_video_shape),
                    ("previous_audio", audio_shape),
                ):
                    value = checkpoint.get(key)
                    if not isinstance(value, torch.Tensor) or tuple(value.shape) != expected_shape:
                        raise ValueError(f"formal checkpoint has an invalid {key} latent")
                resume_previous_video = checkpoint["previous_video"].cuda()
                resume_previous_audio = checkpoint["previous_audio"].cuda()
                resume_previous_video_sigma = float(checkpoint["previous_video_sigma"])
                resume_previous_audio_sigma = float(checkpoint["previous_audio_sigma"])
                resume_forecast_state = checkpoint.get("forecast_state")
            execution_profile["formal_resume"] = {
                "checkpoint": str(Path(request.formal_resume_state_path).resolve()),
                "next_step_index": resume_step_offset,
                "remaining_steps": request.steps - resume_step_offset,
                "formal_prefix_replayed": False,
                "sigma_schedule_preserved": True,
            }
            del checkpoint
        elif request.sampler_state_path is not None:
            checkpoint = torch.load(
                Path(request.sampler_state_path),
                map_location="cpu",
                weights_only=True,
            )
            expected_metadata = {
                "frames": request.frames,
                "fps": request.fps,
                "width": request.width,
                "height": request.height,
                "engine": request_engine,
                "representation": "formal_noisy_sampler_state_after_step",
            }
            for key, expected in expected_metadata.items():
                if checkpoint.get(key) != expected:
                    raise ValueError(
                        "sampler state metadata mismatch for "
                        f"{key}: expected {expected!r}, got {checkpoint.get(key)!r}"
                    )
            for key, expected_shape in (
                ("video", video_shape),
                ("audio", audio_shape),
                ("previous_video", video_shape),
                ("previous_audio", audio_shape),
            ):
                value = checkpoint.get(key)
                if not isinstance(value, torch.Tensor) or tuple(value.shape) != expected_shape:
                    raise ValueError(f"sampler state has an invalid {key} latent")
            sigma_start = float(checkpoint["sigma_next"])
            if not 0.0 < sigma_start <= 1.0:
                raise ValueError("sampler state sigma_next must be inside (0, 1]")
            sigmas = tuple(
                sigma_start * (1.0 - offset / request.steps)
                for offset in range(request.steps + 1)
            )
            video = checkpoint["video"].cuda()
            audio = checkpoint["audio"].cuda()
            resume_previous_video = checkpoint["previous_video"].cuda()
            resume_previous_audio = checkpoint["previous_audio"].cuda()
            resume_previous_video_sigma = float(checkpoint["previous_video_sigma"])
            resume_previous_audio_sigma = float(checkpoint["previous_audio_sigma"])
            execution_profile["sampler_resume"] = {
                "checkpoint": str(Path(request.sampler_state_path).resolve()),
                "source_step_index": int(checkpoint["step_index"]),
                "source_sigma_next": sigma_start,
                "solver_steps": request.steps,
                "video_sigmas": list(sigmas),
                "formal_prefix_replayed": False,
            }
            del checkpoint
        elif request.continuation_latents_path is not None:
            checkpoint = torch.load(
                Path(request.continuation_latents_path),
                map_location="cpu",
                weights_only=True,
            )
            if checkpoint.get("fps") != request.fps:
                raise ValueError("continuation checkpoint fps does not match the request")
            continuation_width = (
                request.multiscale_initial_width
                if request.multiscale_initial_width is not None
                else request.width
            )
            continuation_height = (
                request.multiscale_initial_height
                if request.multiscale_initial_height is not None
                else request.height
            )
            source_width = checkpoint.get("width")
            source_height = checkpoint.get("height")
            valid_continuation_geometries = {
                (continuation_width, continuation_height)
            }
            if request.multiscale_resize_after_step is not None:
                # A direct JSON SelfLift window finishes on the target canvas;
                # its next window reuses only the terminal context on the
                # smaller first-pass canvas. Online fork previews remain at
                # the initial geometry and continue through the older route.
                valid_continuation_geometries.add((request.width, request.height))
            if (source_width, source_height) not in valid_continuation_geometries:
                raise ValueError(
                    "continuation checkpoint geometry does not match the request"
                )
            source_engine = checkpoint.get("engine")
            compatible_source_engines = (
                {"reference", "reference_lora"}
                if self._uses_reference_layout
                else {"original", "lora"}
            )
            if source_engine not in compatible_source_engines:
                raise ValueError(
                    "continuation checkpoint service family does not match the request"
                )
            source_video = checkpoint.get("video")
            source_audio = checkpoint.get("audio")
            if not isinstance(source_video, torch.Tensor) or source_video.ndim != 5:
                raise ValueError("continuation checkpoint has an invalid video latent")
            if not isinstance(source_audio, torch.Tensor) or source_audio.ndim != 4:
                raise ValueError("continuation checkpoint has an invalid audio latent")
            source_latent_downsampled = bool(
                (source_width, source_height) == (request.width, request.height)
                and (continuation_width, continuation_height)
                != (request.width, request.height)
            )
            source_target_video_context_cpu = None
            if source_latent_downsampled:
                context_tokens = video_latent_frames(
                    request.continuation_context_frames
                )
                source_target_video_context_cpu = (
                    source_video[:, :, -context_tokens:].contiguous()
                )
                source_video = resize_refinement_video_latent_spatial(
                    source_target_video_context_cpu,
                    target_height=video_noise_cpu.shape[-2],
                    target_width=video_noise_cpu.shape[-1],
                )
            (
                video_cpu,
                audio_cpu,
                video_prefix_cpu,
                audio_prefix_cpu,
            ) = prepare_masked_av_prefix(
                video_noise_cpu,
                audio_noise_cpu,
                source_video,
                source_audio,
                context_frames=request.continuation_context_frames,
                video_prefix_frames=request.continuation_video_prefix_frames,
                video_prefix_from_source_end=(
                    request.continuation_video_prefix_from_source_end
                ),
                audio_bridge_ticks=request.continuation_audio_bridge_ticks,
            )
            if source_target_video_context_cpu is not None:
                protected_tokens = int(video_prefix_cpu.shape[2])
                if request.continuation_video_prefix_from_source_end:
                    continuation_target_video_prefix_cpu = (
                        source_target_video_context_cpu[
                            :, :, -protected_tokens:
                        ].clone()
                        if protected_tokens
                        else source_target_video_context_cpu[:, :, :0].clone()
                    )
                else:
                    continuation_target_video_prefix_cpu = (
                        source_target_video_context_cpu[
                            :, :, :protected_tokens
                        ].clone()
                    )
            video = video_cpu.cuda()
            audio = audio_cpu.cuda()
            continuation_video_prefix = video_prefix_cpu.cuda()
            continuation_audio_prefix = audio_prefix_cpu.cuda()
            sigmas = simple_sigma_schedule(request.steps, primary_video_shift)
            execution_profile["long_horizon_continuation"] = {
                "checkpoint": str(Path(request.continuation_latents_path).resolve()),
                "context_frames": request.continuation_context_frames,
                "video_prefix_frames": (
                    request.continuation_context_frames
                    if request.continuation_video_prefix_frames is None
                    else request.continuation_video_prefix_frames
                ),
                "video_context_tokens": video_latent_frames(
                    request.continuation_context_frames
                ),
                "video_protected_tokens": int(video_prefix_cpu.shape[2]),
                "video_prefix_source": (
                    "terminal_rebased"
                    if request.continuation_video_prefix_from_source_end
                    else "same_time_context"
                ),
                "video_hidden_repaint_frames": (
                    request.continuation_context_frames
                    - (
                        request.continuation_context_frames
                        if request.continuation_video_prefix_frames is None
                        else request.continuation_video_prefix_frames
                    )
                ),
                "video_hidden_repaint_tokens": (
                    video_latent_frames(request.continuation_context_frames)
                    - int(video_prefix_cpu.shape[2])
                ),
                "video_handoff_policy": "hidden_context_repaint_v1",
                "audio_context_tokens": audio_latent_frames(
                    request.continuation_context_frames
                ),
                "audio_protected_tokens": int(audio_prefix_cpu.shape[-1]),
                "audio_bridge_ticks": request.continuation_audio_bridge_ticks,
                "audio_bridge_seconds": (
                    request.continuation_audio_bridge_ticks / 40.0
                ),
                "prefix_clamped_each_solver_step": True,
                "intermediate_decode": False,
                "source_geometry": [source_width, source_height],
                "working_geometry": [
                    continuation_width,
                    continuation_height,
                ],
                "source_latent_downsampled_for_selflift": (
                    source_latent_downsampled
                ),
                "target_resolution_prefix_retained": (
                    continuation_target_video_prefix_cpu is not None
                ),
                "target_resolution_prefix_tokens": (
                    0
                    if continuation_target_video_prefix_cpu is None
                    else int(continuation_target_video_prefix_cpu.shape[2])
                ),
            }
            del (
                checkpoint,
                source_video,
                source_audio,
                source_target_video_context_cpu,
                video_cpu,
                audio_cpu,
                video_prefix_cpu,
                audio_prefix_cpu,
            )
        elif request.global_selflift_source_path is not None:
            if global_co_plan is None:
                raise RuntimeError("global SelfLift has no temporal window plan")
            source_path = Path(request.global_selflift_source_path)
            checkpoint = torch.load(
                source_path,
                map_location="cpu",
                weights_only=True,
            )
            expected_metadata = {
                "frames": final_output_frames,
                "fps": request.fps,
                "steps": request.steps,
                "representation": "global_selflift_clean_source_x0_v2",
            }
            for key, expected in expected_metadata.items():
                if checkpoint.get(key) != expected:
                    raise ValueError(
                        "global SelfLift source metadata mismatch for "
                        f"{key}: expected {expected!r}, got {checkpoint.get(key)!r}"
                    )
            source_engine = checkpoint.get("engine")
            compatible_source_engines = (
                {"reference", "reference_lora"}
                if self._uses_reference_layout
                else {"original", "lora"}
            )
            if source_engine not in compatible_source_engines:
                raise ValueError(
                    "global SelfLift source service family does not match the request"
                )
            source_width = checkpoint.get("width")
            source_height = checkpoint.get("height")
            if not isinstance(source_width, int) or not isinstance(source_height, int):
                raise ValueError("global SelfLift source is missing its geometry")
            if source_width > request.width or source_height > request.height:
                raise ValueError("global SelfLift source cannot exceed target geometry")
            clean_low_video_cpu = checkpoint.get("video")
            clean_audio_cpu = checkpoint.get("audio")
            expected_source_video_shape = (
                1,
                24,
                global_co_plan.video_tokens,
                source_height // 16,
                source_width // 16,
            )
            if not isinstance(clean_low_video_cpu, torch.Tensor) or tuple(
                clean_low_video_cpu.shape
            ) != expected_source_video_shape:
                raise ValueError("global SelfLift source has an invalid video x0")
            if not isinstance(clean_audio_cpu, torch.Tensor) or tuple(
                clean_audio_cpu.shape
            ) != audio_shape:
                raise ValueError("global SelfLift source has an invalid audio x0")
            resume_step_offset = int(checkpoint.get("next_step_index", -1))
            if not 1 <= resume_step_offset < request.steps:
                raise ValueError("global SelfLift source has an invalid split step")
            full_sigmas = simple_sigma_schedule(request.steps, primary_video_shift)
            recorded_sigmas = tuple(float(value) for value in checkpoint.get("sigmas", ()))
            if recorded_sigmas != full_sigmas:
                raise ValueError("global SelfLift sigma schedule does not match the request")
            identity_lift = (
                source_width == request.width
                and source_height == request.height
            )
            if identity_lift:
                # A same-resolution project still uses the retained global
                # fork, but it must not pay for or alter pixels through the
                # learned spatial upscaler.
                clean_high_video_cpu = clean_low_video_cpu
            else:
                if self.latent_upscaler is None:
                    raise RuntimeError(
                        "this launcher does not provide the H3 latent upscaler"
                    )
                from .latent_upscaler import upscale_h3_video_latent

                self._clear_block_executor()
                self._timed(
                    phases,
                    "global_selflift_dit_evict",
                    lambda: self.transformer.move_to("cpu", non_blocking=False),
                )
                self._release_device()
                self._timed(
                    phases,
                    "global_selflift_upscaler_h2d",
                    lambda: self.latent_upscaler.move_to("cuda:0", non_blocking=True),
                )
                clean_high_video_cpu = self._timed(
                    phases,
                    "global_selflift_learned_x0_lift",
                    lambda: upscale_h3_video_latent(
                        self.latent_upscaler.value,
                        clean_low_video_cpu,
                        target_height=video_shape[-2],
                        target_width=video_shape[-1],
                        temporal_chunk_frames=24,
                        temporal_mode="joint_3d",
                    ),
                )
                self._timed(
                    phases,
                    "global_selflift_upscaler_evict",
                    lambda: self.latent_upscaler.move_to("cpu", non_blocking=False),
                )
                self._release_device()
                dit = self._timed(
                    phases,
                    "global_selflift_dit_h2d",
                    lambda: self._activate_transformer(execution_plan),
                )

            audio_is_final = bool(checkpoint.get("audio_final", False))
            audio_state_cpu = checkpoint.get("audio_state")
            formal_tail_sigmas = tuple(full_sigmas[resume_step_offset:])
            sigma_scale = float(request.global_selflift_sigma_scale)
            if not math.isclose(sigma_scale, 1.0) and not audio_is_final:
                raise ValueError(
                    "global SelfLift sigma scaling requires locked final audio"
                )
            sigmas = tuple(
                float(value) * sigma_scale for value in formal_tail_sigmas[:-1]
            ) + (0.0,)
            sigma_start = float(sigmas[0])
            if audio_is_final:
                preserved_global_selflift_audio = clean_audio_cpu.to(
                    device="cuda:0",
                    non_blocking=False,
                )
            elif (
                not isinstance(audio_state_cpu, torch.Tensor)
                or tuple(audio_state_cpu.shape) != audio_shape
            ):
                raise ValueError(
                    "unfinished global SelfLift audio is missing its exact "
                    "formal sampler state"
                )
            video = (
                sigma_start * video_noise_cpu
                + (1.0 - sigma_start) * clean_high_video_cpu.float()
            ).cuda()
            global_selflift_anchor_video = clean_high_video_cpu.to(
                device="cuda:0",
                dtype=video.dtype,
                non_blocking=False,
            )
            audio = (
                clean_audio_cpu.to(device="cuda:0", non_blocking=False)
                if audio_is_final
                else audio_state_cpu.to(device="cuda:0", non_blocking=False)
            )
            if not request.use_lora:
                previous_sigma = float(full_sigmas[resume_step_offset - 1])
                resume_previous_video = clean_high_video_cpu.to(
                    device="cuda:0", non_blocking=False
                )
                resume_previous_audio = clean_audio_cpu.to(
                    device="cuda:0", non_blocking=False
                )
                resume_previous_video_sigma = previous_sigma
                resume_previous_audio_sigma = previous_sigma
            execution_profile["global_selflift"] = {
                "mechanism": "global_sliding_selflift_v1",
                "source": str(source_path.resolve()),
                "source_geometry": [source_width, source_height],
                "target_geometry": [request.width, request.height],
                "spatial_handoff": (
                    "identity" if identity_lift else "learned_h3_latent_lift"
                ),
                "completed_low_resolution_steps": resume_step_offset,
                "remaining_high_resolution_steps": request.steps - resume_step_offset,
                "video_sigmas": list(sigmas),
                "formal_video_sigmas": list(formal_tail_sigmas),
                "sigma_scale": sigma_scale,
                "one_global_solver_state": True,
                "overlap_fused_before_scheduler_update": True,
                "independent_high_resolution_clips": False,
                "audio_handoff": (
                    "locked_completed_preview_x0"
                    if audio_is_final
                    else "exact_formal_sampler_state"
                ),
                "audio_rewritten_by_high_resolution_steps": not audio_is_final,
                "video_seam_stabilization": (
                    "motion_anchored_detail_residual_v1"
                ),
                "video_seam_stabilization_strength": 0.85,
                "video_seam_padding_tokens": 2,
            }
            del (
                checkpoint,
                clean_low_video_cpu,
                clean_high_video_cpu,
                clean_audio_cpu,
                audio_state_cpu,
            )
        elif (
            request.refinement_latents_path is None
            and external_refinement_video_cpu is None
        ):
            video = video_noise_cpu.cuda()
            audio = audio_noise_cpu.cuda()
            sigmas = simple_sigma_schedule(request.steps, primary_video_shift)
        else:
            target_geometry = (request.width, request.height)
            if request.refinement_latents_path is not None:
                checkpoint = torch.load(
                    Path(request.refinement_latents_path),
                    map_location="cpu",
                    weights_only=True,
                )
                expected_metadata = {
                    "frames": request.frames,
                    "fps": request.fps,
                }
                for key, expected in expected_metadata.items():
                    if checkpoint.get(key) != expected:
                        raise ValueError(
                            "refinement checkpoint metadata mismatch for "
                            f"{key}: expected {expected!r}, got {checkpoint.get(key)!r}"
                        )
                source_engine = checkpoint.get("engine")
                compatible_source_engines = (
                    {"reference", "reference_lora"}
                    if self._uses_reference_layout
                    else {"original", "lora"}
                )
                if source_engine not in compatible_source_engines:
                    raise ValueError(
                        "refinement checkpoint service family does not match the request"
                    )
                source_width = checkpoint.get("width")
                source_height = checkpoint.get("height")
                if not isinstance(source_width, int) or not isinstance(
                    source_height, int
                ):
                    raise ValueError(
                        "refinement checkpoint is missing integer source geometry"
                    )
                clean_video_cpu = checkpoint.get("video")
                clean_audio_cpu = checkpoint.get("audio")
            else:
                checkpoint = None
                source_width, source_height = target_geometry
                clean_video_cpu = external_refinement_video_cpu
                # Repair output audio is discarded and the exact source stream
                # is muxed back later. A clean zero audio anchor keeps the joint
                # H3 state on the same low-noise schedule without inventing a
                # dependency on a retained generation latent.
                clean_audio_cpu = torch.zeros_like(audio_noise_cpu)
            source_geometry = (source_width, source_height)
            if (
                source_geometry != target_geometry
                and request.refinement_spatial_mode == "strict"
            ):
                raise ValueError(
                    "refinement source metadata mismatch for geometry: "
                    f"expected {target_geometry!r}, got {source_geometry!r}"
                )
            source_video_shape = (
                1,
                24,
                video_shape[2],
                source_height // 16,
                source_width // 16,
            )
            if not isinstance(clean_video_cpu, torch.Tensor) or tuple(
                clean_video_cpu.shape
            ) != source_video_shape:
                raise ValueError("refinement checkpoint has an invalid video latent")
            if not isinstance(clean_audio_cpu, torch.Tensor) or tuple(
                clean_audio_cpu.shape
            ) != audio_shape:
                raise ValueError("refinement checkpoint has an invalid audio latent")
            if (
                external_refinement_video_cpu is not None
                and request.refinement_latents_path is not None
            ):
                if tuple(external_refinement_video_cpu.shape) != video_shape:
                    raise ValueError(
                        "external refinement video latent shape mismatch: "
                        f"expected {video_shape!r}, got "
                        f"{tuple(external_refinement_video_cpu.shape)!r}"
                    )
                clean_video_cpu = external_refinement_video_cpu
                source_width = request.width
                source_height = request.height
                source_geometry = target_geometry
                source_video_shape = video_shape
            if source_geometry != target_geometry:
                if request.refinement_spatial_mode != "learned_3d":
                    raise ValueError(
                        "cross-resolution H3 refinement requires learned_3d"
                    )
                refinement_initialization_profile = os.environ.get(
                    "H3_SECOND_SAMPLING_INIT_PROFILE", "learned_3d_v1"
                ).strip() or "learned_3d_v1"
                if refinement_initialization_profile == "bicubic_spatial_v1":
                    # Geometry-only control: every latent time slice is
                    # resized independently.  It cannot create temporal
                    # ghosts, making it a useful stable anchor for testing
                    # whether the learned 3D initializer consumes the short
                    # H3 trajectory repairing its own motion artifacts.
                    clean_video_cpu = self._timed(
                        phases,
                        "bicubic_latent_upscale",
                        lambda: resize_refinement_video_latent_spatial(
                            clean_video_cpu,
                            target_height=video_shape[-2],
                            target_width=video_shape[-1],
                        ),
                    )
                elif refinement_initialization_profile in (
                    "learned_3d_v1",
                    "learned_framewise_static_v1",
                ):
                    if self.latent_upscaler is None:
                        raise RuntimeError(
                            "this launcher does not provide H3 second sampling"
                        )
                    from .latent_upscaler import upscale_h3_video_latent

                    # Prompt projection needs the DiT once before this branch.
                    # Evict it while the 3D upscaler runs, so 16GB and 24GB use
                    # the same bounded-residency phase instead of overlapping two
                    # model slabs and relying on allocator luck.
                    self._timed(
                        phases,
                        "second_sampling_dit_evict",
                        lambda: self.transformer.move_to("cpu", non_blocking=False),
                    )
                    self._release_device()
                    self._timed(
                        phases,
                        "latent_upscaler_h2d",
                        lambda: self.latent_upscaler.move_to(
                            "cuda:0", non_blocking=True
                        ),
                    )
                    clean_video_cpu = self._timed(
                        phases,
                        "learned_latent_upscale",
                        lambda: upscale_h3_video_latent(
                            self.latent_upscaler.value,
                            clean_video_cpu,
                            target_height=video_shape[-2],
                            target_width=video_shape[-1],
                            temporal_chunk_frames=24,
                            temporal_mode=(
                                "framewise_static"
                                if refinement_initialization_profile
                                == "learned_framewise_static_v1"
                                else "joint_3d"
                            ),
                        ),
                    )
                    self._timed(
                        phases,
                        "latent_upscaler_evict",
                        lambda: self.latent_upscaler.move_to(
                            "cpu", non_blocking=False
                        ),
                    )
                    self._release_device()
                    dit = self._timed(
                        phases,
                        "second_sampling_dit_h2d",
                        lambda: self._activate_transformer(execution_plan),
                    )
                else:
                    raise ValueError(
                        "unsupported H3 second-sampling initialization profile: "
                        f"{refinement_initialization_profile}"
                    )
            refinement_source_latent_height = int(source_video_shape[-2])
            refinement_source_latent_width = int(source_video_shape[-1])
            if request.refinement_handoff_latents_path is not None:
                handoff = torch.load(
                    Path(request.refinement_handoff_latents_path),
                    map_location="cpu",
                    weights_only=True,
                )
                previous_video = handoff.get("video")
                if not isinstance(previous_video, torch.Tensor) or previous_video.ndim != 5:
                    raise ValueError("refinement handoff checkpoint has no video latent")
                if (
                    int(previous_video.shape[0]) != int(clean_video_cpu.shape[0])
                    or int(previous_video.shape[1]) != int(clean_video_cpu.shape[1])
                    or tuple(previous_video.shape[-2:]) != tuple(clean_video_cpu.shape[-2:])
                ):
                    raise ValueError("refinement handoff Atlas geometry does not match")
                requested_handoff = video_latent_frames(
                    request.refinement_handoff_context_frames
                )
                refinement_handoff_latent_frames = min(
                    requested_handoff,
                    int(previous_video.shape[2]),
                    int(clean_video_cpu.shape[2]),
                )
                clean_video_cpu = clean_video_cpu.float().clone()
                clean_video_cpu[:, :, :refinement_handoff_latent_frames].copy_(
                    previous_video[
                        :, :, -refinement_handoff_latent_frames:
                    ].to(dtype=clean_video_cpu.dtype)
                )
                del previous_video, handoff
            hard_first_anchor = os.environ.get(
                "H3_SECOND_SAMPLING_HARD_FIRST_FRAME_ANCHOR", "0"
            ).strip().lower() in ("1", "true", "yes", "on")
            if hard_first_anchor and 0 in keyframe_indices:
                anchor_index = keyframe_indices.index(0)
                anchor = keyframe_latents[anchor_index].float()
                if int(anchor.shape[2]) != 1:
                    anchor = anchor[:, :, :1]
                if tuple(anchor.shape[-2:]) != tuple(clean_video_cpu.shape[-2:]):
                    anchor = resize_refinement_video_latent_spatial(
                        anchor,
                        target_height=int(clean_video_cpu.shape[-2]),
                        target_width=int(clean_video_cpu.shape[-1]),
                    )
                fade_latent_frames = min(
                    int(clean_video_cpu.shape[2]),
                    max(
                        1,
                        int(
                            os.environ.get(
                                "H3_SECOND_SAMPLING_FIRST_ANCHOR_FADE_LATENT_FRAMES",
                                "6",
                            )
                        ),
                    ),
                )
                fade_floor = min(
                    1.0,
                    max(
                        0.0,
                        float(
                            os.environ.get(
                                "H3_SECOND_SAMPLING_FIRST_ANCHOR_FADE_FLOOR",
                                "0.05",
                            )
                        ),
                    ),
                )
                coarse_gain = min(
                    1.0,
                    max(
                        0.0,
                        float(
                            os.environ.get(
                                "H3_SECOND_SAMPLING_FIRST_ANCHOR_COARSE_GAIN",
                                "0.12",
                            )
                        ),
                    ),
                )
                source_first = clean_video_cpu[:, :, :1].float()
                anchor_delta = anchor - source_first
                flat_delta = anchor_delta[:, :, 0]
                anchor_coarse = F.avg_pool2d(
                    flat_delta,
                    kernel_size=5,
                    stride=1,
                    padding=2,
                ).unsqueeze(2)
                anchor_detail = anchor_delta - anchor_coarse
                if fade_latent_frames == 1:
                    fade_weights_cpu = torch.ones(
                        (1, 1, 1, 1, 1), dtype=torch.float32
                    )
                else:
                    fade_weights_cpu = torch.exp(
                        torch.linspace(
                            0.0,
                            math.log(max(fade_floor, 1e-6)),
                            fade_latent_frames,
                            dtype=torch.float32,
                        )
                    ).view(1, 1, fade_latent_frames, 1, 1)
                anchor_target = clean_video_cpu[
                    :, :, :fade_latent_frames
                ].float().clone()
                anchor_target[:, :, :1] = anchor
                for latent_index in range(1, fade_latent_frames):
                    source_at_time = clean_video_cpu[
                        :, :, latent_index : latent_index + 1
                    ].float()
                    motion = (source_at_time - source_first).abs().mean(
                        dim=1, keepdim=True
                    )
                    motion_scale = torch.quantile(
                        motion.reshape(-1), 0.75
                    ).clamp_min(1e-4)
                    motion_gate = torch.exp(
                        -motion / (2.0 * motion_scale)
                    ).clamp_(0.0, 1.0)
                    anchor_target[
                        :, :, latent_index : latent_index + 1
                    ] = source_at_time + motion_gate * (
                        anchor_detail + coarse_gain * anchor_coarse
                    )
                clean_prefix = clean_video_cpu[
                    :, :, :fade_latent_frames
                ].float()
                clean_video_cpu = clean_video_cpu.float().clone()
                clean_video_cpu[:, :, :fade_latent_frames] = (
                    clean_prefix * (1.0 - fade_weights_cpu)
                    + anchor_target * fade_weights_cpu
                )
                refinement_first_anchor_clean = anchor_target.to(
                    device="cuda:0", non_blocking=False
                )
                refinement_first_anchor_noise = video_noise_cpu[
                    :, :, :fade_latent_frames
                ].to(device="cuda:0", non_blocking=False)
                refinement_first_anchor_weights = fade_weights_cpu.to(
                    device="cuda:0", non_blocking=False
                )
                execution_profile["hard_first_frame_anchor"] = {
                    "mode": "latent_path_clamp_with_motion_gated_detail_fade_v1",
                    "latent_frames": fade_latent_frames,
                    "fade_floor": fade_floor,
                    "coarse_gain": coarse_gain,
                    "anchor_shape": list(anchor.shape),
                    "target_shape": list(clean_video_cpu.shape),
                }
            # Keep one clean target-grid anchor resident.  It is only a few
            # MiB for the supported clips and lets the H3 pass redraw genuine
            # target-grid detail while coarse scene geometry stays tied to the
            # accepted source trajectory.
            refinement_motion_video = clean_video_cpu.to(
                device="cuda:0", non_blocking=False
            )
            if request.refinement_atlas_denoise_regions:
                latent_t = int(clean_video_cpu.shape[2])
                latent_h = int(clean_video_cpu.shape[-2])
                latent_w = int(clean_video_cpu.shape[-1])
                atlas_mask = torch.zeros(
                    (1, 1, latent_t, latent_h, latent_w), dtype=torch.float32
                )
                for x, y, width, height, frame_strengths in (
                    request.refinement_atlas_denoise_regions
                ):
                    temporal = torch.tensor(
                        tuple(float(value) for value in frame_strengths),
                        dtype=torch.float32,
                    ).view(1, 1, -1)
                    temporal = F.interpolate(
                        temporal,
                        size=latent_t,
                        mode="linear",
                        align_corners=True,
                    ).view(latent_t, 1, 1)
                    x0 = max(0, min(latent_w - 1, int(round(float(x) * latent_w))))
                    y0 = max(0, min(latent_h - 1, int(round(float(y) * latent_h))))
                    x1 = max(x0 + 1, min(latent_w, int(round(float(x + width) * latent_w))))
                    y1 = max(y0 + 1, min(latent_h, int(round(float(y + height) * latent_h))))
                    atlas_mask[0, 0, :, y0:y1, x0:x1] = torch.maximum(
                        atlas_mask[0, 0, :, y0:y1, x0:x1], temporal
                    )
                if refinement_handoff_latent_frames:
                    atlas_mask[:, :, :refinement_handoff_latent_frames].zero_()
                refinement_sampler_mask = atlas_mask.expand(
                    int(clean_video_cpu.shape[0]),
                    int(clean_video_cpu.shape[1]),
                    -1,
                    -1,
                    -1,
                ).contiguous().to(device="cuda:0")
                refinement_sampler_noise = video_noise_cpu.to(
                    device="cuda:0", non_blocking=False
                )
            if request.refinement_full_canvas_regions:
                refinement_full_canvas_mask = build_refinement_region_mask(
                    height=int(clean_video_cpu.shape[-2]),
                    width=int(clean_video_cpu.shape[-1]),
                    regions=request.refinement_full_canvas_regions,
                    feather=request.refinement_full_canvas_feather,
                    device="cuda:0",
                )
            assert request.refinement_denoise is not None
            schedule_builder = (
                comfy_denoise_tail_sigma_schedule
                if request.refinement_schedule_mode == "comfy_simple_tail"
                else refinement_sigma_schedule
            )
            schedule_kwargs = (
                {}
                if request.refinement_schedule_mode == "comfy_simple_tail"
                else {"curve_power": request.refinement_sigma_power}
            )
            sigmas = schedule_builder(
                request.steps,
                request.refinement_denoise,
                request.refinement_video_shift,
                **schedule_kwargs,
            )
            sigma_start = float(sigmas[0])
            refinement_audio_sigmas = (
                schedule_builder(
                    request.steps,
                    request.refinement_denoise,
                    self.lora_audio_shift,
                    **schedule_kwargs,
                )
                if request.use_lora
                else sigmas
            )
            audio_sigma_start = float(refinement_audio_sigmas[0])
            initial_refinement_video = (
                sigma_start * video_noise_cpu
                + (1.0 - sigma_start) * clean_video_cpu.float()
            ).cuda()
            video = (
                initial_refinement_video
                if refinement_sampler_mask is None
                else refinement_motion_video.float()
                + refinement_sampler_mask
                * (initial_refinement_video - refinement_motion_video.float())
            )
            audio = (
                audio_sigma_start * audio_noise_cpu
                + (1.0 - audio_sigma_start) * clean_audio_cpu.float()
            ).cuda()
            if request.preserve_refinement_audio:
                preserved_refinement_audio = clean_audio_cpu.cuda()
            execution_profile["refinement"] = {
                "source": (
                    str(Path(request.refinement_latents_path).resolve())
                    if request.refinement_latents_path is not None
                    else str(external_refinement_video_path)
                ),
                "source_kind": (
                    "latent_checkpoint"
                    if request.refinement_latents_path is not None
                    else "pixel_video_vae_encode"
                ),
                "denoise": request.refinement_denoise,
                "video_sigma_shift": request.refinement_video_shift,
                "sigma_curve_power": request.refinement_sigma_power,
                "solver_steps": request.steps,
                "sampler": (
                    "turbo" if request.use_lora else request.refinement_sampler
                ),
                "schedule_total_steps": (
                    max(request.steps, int(request.steps / request.refinement_denoise))
                    if request.refinement_schedule_mode == "comfy_simple_tail"
                    else request.steps
                ),
                "schedule_semantics": (
                    "comfy_basic_scheduler_simple_tail_v1"
                    if request.refinement_schedule_mode == "comfy_simple_tail"
                    else "fixed_start_sigma_power_tail_v1"
                ),
                "video_sigmas": list(sigmas),
                "audio_sigmas": list(refinement_audio_sigmas),
                "preserve_first_pass_audio": request.preserve_refinement_audio,
                "spatial_mode": request.refinement_spatial_mode,
                "source_geometry": list(source_geometry),
                "target_geometry": list(target_geometry),
                "initialization_profile": refinement_initialization_profile,
                "atlas_sampler_only_mask": bool(refinement_sampler_mask is not None),
                "progressive_handoff_latent_frames": refinement_handoff_latent_frames,
                "detail_regeneration": {
                    "prediction_low_frequency_gain": (
                        request.refinement_prediction_low_frequency_gain
                    ),
                    "final_low_frequency_gain": (
                        request.refinement_final_low_frequency_gain
                    ),
                    "temporal_lowpass": request.refinement_temporal_lowpass,
                    "temporal_outlier_only": (
                        request.refinement_temporal_outlier_only
                    ),
                    "temporal_detail_outlier_strength": (
                        request.refinement_temporal_detail_outlier_strength
                    ),
                    "cross_step_detail_strength": (
                        request.refinement_cross_step_detail_strength
                    ),
                    "source_anchor": refinement_initialization_profile,
                    "detail_authority": "h3_target_grid_high_frequency",
                },
            }
            del checkpoint, clean_video_cpu, clean_audio_cpu
        del video_noise_cpu, audio_noise_cpu
        all_steps = tuple(range(request.steps))
        actual_steps = (
            all_steps
            if global_co_plan is not None or request.actual_step_indices is None
            else request.actual_step_indices
        )
        if self._uses_turbo_sampler(request) and actual_steps != all_steps:
            raise ValueError("the distilled LoRA route executes every requested step")
        segment_cache = None
        if (
            global_co_plan is None
            and execution_plan is not None
            and execution_plan.segment_cache_reuse_steps
        ):
            if not set(execution_plan.segment_cache_reuse_steps).issubset(actual_steps):
                raise ValueError("segment cache reuse steps must be actual DiT steps")
            segment_cache = CoordinateAlignedSegmentCache(
                SegmentResidualCacheConfig(
                    layer_start=execution_plan.segment_cache_layer_start,
                    layer_stop=execution_plan.segment_cache_layer_stop,
                    reuse_steps=execution_plan.segment_cache_reuse_steps,
                    directional_trust=(
                        execution_plan.segment_cache_directional_trust
                    ),
                    directional_max_extra=(
                        execution_plan.segment_cache_directional_max_extra
                    ),
                    directional_min_cosine=(
                        execution_plan.segment_cache_directional_min_cosine
                    ),
                    protected_refresh=(
                        execution_plan.segment_cache_protected_refresh
                    ),
                    active_video_ratio=(
                        execution_plan.segment_cache_active_video_ratio
                    ),
                    dynamic_video_budget=(
                        execution_plan.segment_cache_dynamic_video_budget
                    ),
                    active_video_min_ratio=(
                        execution_plan.segment_cache_active_video_min_ratio
                    ),
                    innovation_risk_coverage=(
                        execution_plan.segment_cache_innovation_risk_coverage
                    ),
                    innovation_max_relative=(
                        execution_plan.segment_cache_innovation_max_relative
                    ),
                    active_layer_start=(
                        execution_plan.segment_cache_active_layer_start
                    ),
                    active_layer_stop=(
                        execution_plan.segment_cache_active_layer_stop
                    ),
                    sequential_layer_groups=(
                        execution_plan.segment_cache_sequential_layer_groups
                    ),
                    sequential_conservative_hold=(
                        execution_plan.segment_cache_sequential_conservative_hold
                    ),
                )
            )
        forecast = None
        forecast_profile_override = None
        if global_co_plan is None and not self._uses_turbo_sampler(request):
            if self.forecast_controller_factory is not None:
                forecast = self.forecast_controller_factory(
                    segment_cache=segment_cache
                )
            elif (
                actual_steps != all_steps
                or request.preview_forecast_steps > 0
                or (
                    request.preview_decode_mode == "fast_finish"
                    and request.preview_branch_actual_step_indices is not None
                    and len(request.preview_branch_actual_step_indices)
                    < request.preview_branch_steps
                )
            ):
                feedback = (
                    None
                    if request.acceleration_plan_summary is None
                    else request.acceleration_plan_summary.get(
                        "runtime_feedback"
                    )
                )
                if request.mechanistic_runtime_controller is not None:
                    forecast = ForecastErrorDebtController(
                        actual_steps=actual_steps,
                        segment_cache=segment_cache,
                        risk_reserve_controller=(
                            request.mechanistic_runtime_controller
                        ),
                    )
                elif (
                    isinstance(feedback, dict)
                    and feedback.get("policy_id")
                    == V24_FORECAST_FEEDBACK_POLICY_ID
                ):
                    feedback_mode = str(feedback.get("mode", ""))
                    if feedback_mode not in (
                        "observe_only",
                        "bounded_recovery",
                    ):
                        raise ValueError(
                            "unsupported V24 forecast feedback mode"
                        )
                    forecast = ForecastErrorDebtController(
                        actual_steps=actual_steps,
                        segment_cache=segment_cache,
                        recovery_enabled=(
                            feedback_mode == "bounded_recovery"
                        ),
                        max_runtime_promotions=int(
                            feedback.get("max_runtime_promotions", 0)
                        ),
                    )
                else:
                    forecast = DirectionalForecastController(
                        actual_steps=actual_steps,
                        segment_cache=segment_cache,
                    )
        if segment_cache is not None and forecast is None:
            raise ValueError(
                "the first segment-cache prototype requires the original forecast route"
            )
        if resume_forecast_state is not None:
            if forecast is None:
                raise ValueError(
                    "formal checkpoint contains forecast history but this request has no controller"
                )
            forecast.restore_checkpoint_state(resume_forecast_state)
        remaining_actual_steps = tuple(
            index for index in actual_steps if index >= resume_step_offset
        )
        video_sigma_shift = (
            request.refinement_video_shift
            if request.refinement_latents_path is not None
            else primary_video_shift
        )
        audio_sigma_shift = (
            3.0
            if request.refinement_latents_path is not None
            else primary_audio_shift
        )
        plan = SamplingPlan(
            sampler=(
                "turbo"
                if self._uses_turbo_sampler(request)
                else request.refinement_sampler
                if request.refinement_latents_path is not None
                else "res_multistep"
            ),
            video_sigmas=sigmas,
            audio_sigmas=(
                sigmas
                if refinement_audio_sigmas is None
                else refinement_audio_sigmas
            ),
            actual_step_indices=remaining_actual_steps,
            video_shift=video_sigma_shift,
            audio_shift=audio_sigma_shift,
            step_index_offset=resume_step_offset,
            seed=request.seed,
        )
        layout = None
        visual_policy = (
            conditioning_authority_profile.get("visual_policy")
            if conditioning_authority_profile is not None
            else None
        )
        layout_velocity_guidance_scheduled = bool(
            visual_policy
            == "state_primary_layout_velocity_guidance"
        )
        novel_camera_layout_probe_scheduled = bool(
            visual_policy
            == "target_camera_seed_layout_probe_text_convergence"
        )
        layout_memory_conditioning_scheduled = bool(
            visual_policy in (
                "layout_bootstrap_state_convergence",
                "state_primary_layout_velocity_guidance",
                "target_camera_seed_layout_probe_text_convergence",
            )
        )
        visual_conditioning_scheduled = bool(
            conditioning_authority_profile is not None
            and visual_policy in (
                "reference_bootstrap_history_convergence",
                "layout_bootstrap_state_convergence",
                "state_primary_layout_velocity_guidance",
                "target_camera_seed_layout_probe_text_convergence",
            )
        )
        inferred_voice_conditioning_scheduled = bool(
            conditioning_authority_profile is not None
            and conditioning_authority_profile.get("audio_policy")
            == "inferred_short_voice_high_noise_bootstrap_v1"
        )
        conditioning_route_scheduled = bool(
            (visual_conditioning_scheduled or inferred_voice_conditioning_scheduled)
            and len(conditioning_routes) > 1
            and request.continuation_latents_path is not None
            and request.refinement_latents_path is None
            and global_co_plan is None
        )
        continuation_text_bridge_scheduled = bool(
            continuation_text_bridge is not None
            and request.continuation_latents_path is not None
            and request.refinement_latents_path is None
            and global_co_plan is None
        )
        boundary_forecast = (
            _continuation_boundary_forecast_controller(actual_steps, request.steps)
            if continuation_text_bridge_scheduled
            else None
        )
        conditioning_route_layouts: dict[tuple[str, str], Any | None] = {}
        if continuation_text_bridge_scheduled:
            bridge_steps = int(math.ceil(
                request.steps * _CONTINUATION_TEXT_BRIDGE_FRACTION
            ))
            execution_profile["continuation_text_bridge"] = {
                "policy": "protected_repaint_plateau_visible_fade_v5",
                "active": True,
                "boundary_auxiliary_steps": bridge_steps,
                "current_prompt_steps": request.steps,
                "current_prompt_start_step": 0,
                "blend_peak": _CONTINUATION_BOUNDARY_BLEND_PEAK,
                "blend_band_policy": "hidden_repaint_plateau_then_zero_ended_visible_fade",
                "blend_band_video_latent_tokens": None,
                "hidden_repaint_video_latent_tokens": None,
                "visible_fade_video_latent_tokens": None,
                "extra_dit_calls": bridge_steps,
                "auxiliary_schedule": (
                    "mirrors_primary_actual_forecast_v1"
                    if boundary_forecast is not None
                    else "all_actual"
                ),
                "planned_exact_auxiliary_steps": len(actual_steps),
                "planned_forecast_auxiliary_steps": bridge_steps - len(actual_steps),
                "nonempty_exact_video_prefix_required": True,
            }
        if conditioning_route_scheduled:
            reference_steps = int(math.ceil(
                request.steps * (
                    _LAYOUT_MEMORY_BOOTSTRAP_FRACTION
                    if layout_memory_conditioning_scheduled
                    else _AUTHORITATIVE_REFERENCE_BOOTSTRAP_FRACTION
                )
            ))
            if novel_camera_layout_probe_scheduled:
                state_seed_steps, reference_steps = (
                    _novel_camera_layout_probe_schedule(request.steps)
                )
                history_start_step = 0
                history_convergence_steps = request.steps - reference_steps
            elif (
                layout_velocity_guidance_scheduled
                and request.steps >= 3
            ):
                state_seed_steps, reference_steps = (
                    _explicit_camera_layout_guidance_schedule(request.steps)
                )
                history_start_step = 0
                history_convergence_steps = request.steps
            else:
                state_seed_steps = 0
                history_start_step = reference_steps
                history_convergence_steps = request.steps - reference_steps
            voice_steps = (
                int(math.ceil(
                    request.steps * _INFERRED_VOICE_BOOTSTRAP_FRACTION
                ))
                if inferred_voice_conditioning_scheduled
                else 0
            )
            execution_profile["conditioning_authority"] = {
                **(conditioning_authority_profile or {}),
                "active": True,
                "reference_bootstrap_steps": reference_steps,
                "reference_bootstrap_start_step": state_seed_steps,
                "state_seed_steps": state_seed_steps,
                "history_convergence_steps": history_convergence_steps,
                "history_start_step": history_start_step,
                "layout_guidance_steps": (
                    reference_steps if layout_velocity_guidance_scheduled else 0
                ),
                "layout_guidance_start_step": (
                    state_seed_steps if layout_velocity_guidance_scheduled else None
                ),
                "layout_guidance_weight": (
                    _EXPLICIT_CAMERA_LAYOUT_GUIDANCE_WEIGHT
                    if layout_velocity_guidance_scheduled else None
                ),
                "layout_guidance_extra_dit_calls": (
                    reference_steps if layout_velocity_guidance_scheduled else 0
                ),
                "inferred_voice_bootstrap_steps": voice_steps,
                "inferred_voice_release_steps": request.steps - voice_steps,
                "inferred_voice_release_start_step": (
                    voice_steps
                    if inferred_voice_conditioning_scheduled
                    else None
                ),
            }
        elif inferred_voice_conditioning_scheduled:
            # Inferred voice memory is an experimental continuation aid.  If
            # this request type cannot honor the release schedule (for example
            # a refinement or global co-denoise pass), fail closed by using a
            # route with no inferred audio instead of exposing it throughout.
            release_route_name = next(
                (
                    route_name
                    for route_name in conditioning_routes
                    if route_name.endswith("_release")
                ),
                None,
            )
            if release_route_name is not None:
                released = conditioning_routes[release_route_name]
                condition_video_latents = released.video_latents
                condition_audio_latents = released.audio_latents
                reference_shapes = released.reference_shapes
                reference_kinds = released.reference_kinds
                reference_audio_frames = released.reference_audio_frames
                execution_profile["conditioning_authority"] = {
                    **(conditioning_authority_profile or {}),
                    "active": False,
                    "fallback": "inferred_voice_released_for_unsupported_pass",
                }
        step_seconds: list[float] = []
        long_sequence_step_memory: list[dict[str, float | int | bool]] = []
        self_speculative_records: list[dict[str, Any]] = []
        last_denoised: tuple[torch.Tensor, torch.Tensor] | None = None
        preview_latents: tuple[torch.Tensor, torch.Tensor] | None = None
        preview_published = False
        checkpoint_protected_video_prefix_cpu = None
        checkpoint_protected_audio_prefix_cpu = None
        # Exact clean source-grid prediction at the SelfLift split.  Online
        # long-video creation persists this with every retained fork so final
        # rendering can assemble one continuous low-resolution timeline before
        # any spatial lift or high-resolution solver work occurs.
        selflift_source_video_x0_cpu = None
        selflift_source_audio_x0_cpu = None

        def transition_multiscale(
            index,
            clock,
            step_video,
            step_audio,
            previous_video,
            previous_audio,
        ):
            nonlocal forecast, forecast_profile_override, layout, last_denoised, multiscale_highpass_cpu
            if index != request.multiscale_resize_after_step:
                return step_video, step_audio, previous_video, previous_audio
            target_height, target_width = video_shape[-2:]
            step_video = resize_refinement_video_latent_spatial(
                step_video,
                target_height=target_height,
                target_width=target_width,
            )
            if previous_video is not None:
                previous_video = resize_refinement_video_latent_spatial(
                    previous_video,
                    target_height=target_height,
                    target_width=target_width,
                )
            if multiscale_highpass_cpu is not None:
                detail = multiscale_highpass_cpu.to("cuda:0", non_blocking=False)
                step_video = step_video + (
                    float(clock.video_sigma_next)
                    * request.multiscale_highpass_strength
                    * detail
                )
                del detail
                multiscale_highpass_cpu = None
            if last_denoised is not None:
                last_denoised = (
                    resize_refinement_video_latent_spatial(
                        last_denoised[0],
                        target_height=target_height,
                        target_width=target_width,
                    ),
                    last_denoised[1],
                )
            if forecast is not None:
                forecast_profile_override = forecast.export()
            forecast = None
            layout = None
            for route_name in conditioning_route_layouts:
                conditioning_route_layouts[route_name] = None
            execution_profile["multiscale_transition"] = {
                "after_step": index,
                "sigma_next": float(clock.video_sigma_next),
                "source_geometry": [
                    request.multiscale_initial_width,
                    request.multiscale_initial_height,
                ],
                "target_geometry": [request.width, request.height],
                "highpass_strength": request.multiscale_highpass_strength,
                "post_transition_steps": request.steps - index - 1,
            }
            return step_video, step_audio, previous_video, previous_audio

        def transition_selflift(
            index,
            clock,
            step_video,
            step_audio,
            previous_video,
            previous_audio,
        ):
            """Lift the current clean endpoint, then re-noise at sigma_next.

            The boundary owns an actual H3 prediction. Turbo resumes directly
            from the lifted state. Base additionally carries that lifted x0 as
            its second-order RES history, keeping the next high-resolution
            update shape-consistent without discarding the solver history.
            """

            nonlocal dit, forecast, forecast_profile_override, layout
            nonlocal last_denoised, selflift_target_noise_cpu
            nonlocal continuation_video_prefix
            nonlocal continuation_target_video_prefix_cpu
            nonlocal checkpoint_protected_video_prefix_cpu
            nonlocal checkpoint_protected_audio_prefix_cpu
            nonlocal selflift_source_video_x0_cpu
            nonlocal selflift_source_audio_x0_cpu
            nonlocal condition_video_latents
            if index != request.multiscale_resize_after_step:
                return step_video, step_audio, previous_video, previous_audio
            if last_denoised is None:
                raise RuntimeError("SelfLift transition has no predicted clean latent")
            if selflift_target_noise_cpu is None:
                raise RuntimeError("SelfLift transition has no target-grid noise")

            # A SelfLift preview represents the completed low-resolution
            # prefix.  Finish and decode its disposable branch before lifting
            # the formal trajectory to the target canvas.
            if request.preview_step_index == index and not preview_published:
                publish_preview(index, clock, step_video, step_audio)

            source_shape = list(last_denoised[0].shape)
            clean_low_cpu = last_denoised[0].detach().to(
                device="cpu", non_blocking=False
            )
            if continuation_video_prefix is not None:
                prefix_tokens = int(continuation_video_prefix.shape[2])
                clean_low_cpu[:, :, :prefix_tokens].copy_(
                    continuation_video_prefix.detach().to(
                        device="cpu", dtype=clean_low_cpu.dtype
                    )
                )
            clean_audio_cpu = last_denoised[1].detach().to(
                device="cpu", non_blocking=False
            )
            if continuation_audio_prefix is not None:
                audio_prefix_tokens = int(continuation_audio_prefix.shape[-1])
                clean_audio_cpu[..., :audio_prefix_tokens].copy_(
                    continuation_audio_prefix.detach().to(
                        device="cpu", dtype=clean_audio_cpu.dtype
                    )
                )
            selflift_source_video_x0_cpu = clean_low_cpu.clone()
            selflift_source_audio_x0_cpu = clean_audio_cpu.clone()
            target_height, target_width = video_shape[-2:]
            identity_lift = tuple(clean_low_cpu.shape[-2:]) == (
                target_height,
                target_width,
            )
            if identity_lift:
                # Equal resolution handles retain the same formal fork and
                # second-stage schedule, but there is no spatial work to do.
                # Keep the DiT resident and reuse the clean endpoint exactly.
                clean_high_cpu = clean_low_cpu
            else:
                if self.latent_upscaler is None:
                    raise RuntimeError(
                        "this launcher does not provide the H3 latent upscaler"
                    )
                from .latent_upscaler import upscale_h3_video_latent

                # The transformer and latent upscaler are deliberately never
                # resident together.  This keeps the mid-trajectory lift within
                # the same bounded VRAM contract as H3 second sampling.
                self._clear_block_executor()
                self._timed(
                    phases,
                    "selflift_dit_evict",
                    lambda: self.transformer.move_to("cpu", non_blocking=False),
                )
                self._release_device()
                self._timed(
                    phases,
                    "selflift_upscaler_h2d",
                    lambda: self.latent_upscaler.move_to(
                        "cuda:0", non_blocking=True
                    ),
                )
                clean_high_cpu = self._timed(
                    phases,
                    "selflift_learned_x0_lift",
                    lambda: upscale_h3_video_latent(
                        self.latent_upscaler.value,
                        clean_low_cpu,
                        target_height=target_height,
                        target_width=target_width,
                        temporal_chunk_frames=24,
                        temporal_mode="joint_3d",
                    ),
                )
            restored_target_prefix_tokens = 0
            if continuation_target_video_prefix_cpu is not None:
                restored_target_prefix_tokens = restore_selflift_target_prefix_(
                    clean_high_cpu,
                    continuation_target_video_prefix_cpu,
                )
            if continuation_video_prefix is not None:
                prefix_tokens = int(continuation_video_prefix.shape[2])
                checkpoint_protected_video_prefix_cpu = (
                    clean_high_cpu[:, :, :prefix_tokens].clone()
                )
                checkpoint_protected_audio_prefix_cpu = (
                    continuation_audio_prefix.detach().to(
                        device="cpu", dtype=step_audio.dtype
                    ).clone()
                    if continuation_audio_prefix is not None
                    else None
                )
            if not identity_lift:
                self._timed(
                    phases,
                    "selflift_upscaler_evict",
                    lambda: self.latent_upscaler.move_to(
                        "cpu", non_blocking=False
                    ),
                )
                self._release_device()
                dit = self._timed(
                    phases,
                    "selflift_dit_h2d",
                    lambda: self._activate_transformer(execution_plan),
                )

            # Endpoint keyframes are canvas-bound conditions. Early source-grid
            # steps use the reduced copies assembled above; the high-resolution
            # suffix must receive the original target-grid VAE latents. Every
            # conditioning composition appends keyframes last, so replacing
            # that suffix preserves any independent reference or memory rows.
            if selflift_target_keyframe_latents_cpu:
                target_keyframes = tuple(
                    latent.to("cuda:0", non_blocking=False)
                    for latent in selflift_target_keyframe_latents_cpu
                )
                keyframe_count = len(target_keyframes)
                for route_name, route in tuple(conditioning_routes.items()):
                    conditioning_routes[route_name] = _ConditioningComposition(
                        video_latents=(
                            route.video_latents[:-keyframe_count]
                            + target_keyframes
                        ),
                        reference_shapes=route.reference_shapes,
                        reference_kinds=route.reference_kinds,
                        audio_latents=route.audio_latents,
                        reference_audio_frames=route.reference_audio_frames,
                        profile=route.profile,
                    )
                if conditioning_routes:
                    condition_video_latents = next(
                        iter(conditioning_routes.values())
                    ).video_latents

            # Forecast and segment-cache histories are indexed in packed-token
            # coordinates.  The learned lift changes the spatial token grid,
            # so retaining the low-resolution controller would make its
            # sampled rows and residuals invalid on the first high-resolution
            # Actual observation.  Every SelfLift boundary/tail step is
            # contractually Actual, while the per-layer Attention schedule
            # remains active without this controller; retire it exactly as the
            # older multiscale transition does.
            if forecast is not None:
                forecast_profile_override = forecast.export()
            forecast = None

            clean_high = clean_high_cpu.to(
                device="cuda:0", dtype=step_video.dtype, non_blocking=False
            )
            if continuation_video_prefix is not None:
                # Subsequent target-grid steps clamp the previous window's
                # exact target-grid prefix when one was retained.  Otherwise
                # same-resolution and preview continuations keep their
                # ordinary lifted prefix behavior.
                continuation_video_prefix = clean_high[
                    :, :, : int(continuation_video_prefix.shape[2])
                ].detach().clone()
            continuation_target_video_prefix_cpu = None
            target_noise = selflift_target_noise_cpu.to(
                device="cuda:0", dtype=step_video.dtype, non_blocking=False
            )
            sigma_next = float(clock.video_sigma_next)
            step_video = selflift_renoise_clean_endpoint(
                clean_high,
                target_noise,
                sigma_next,
            )
            if not request.use_lora:
                # RES stores the current x0 for its next second-order update.
                # The incoming object still has the low-resolution shape, so
                # replace it with the exact learned lift already computed for
                # the formal SelfLift endpoint. Audio has no spatial axis and
                # keeps its uninterrupted native RES history.
                previous_video = clean_high
            last_denoised = (clean_high, last_denoised[1])
            selflift_target_noise_cpu = None
            layout = None
            for route_name in conditioning_route_layouts:
                conditioning_route_layouts[route_name] = None
            execution_profile["selflift"] = {
                "method": (
                    "identity_clean_endpoint_handoff_v1"
                    if identity_lift
                    else "learned_h3_clean_endpoint_lift_v1"
                ),
                "after_step": index,
                "completed_low_resolution_steps": index + 1,
                "sigma_next": sigma_next,
                "source_geometry": [
                    request.multiscale_initial_width,
                    request.multiscale_initial_height,
                ],
                "target_geometry": [request.width, request.height],
                "source_latent_shape": source_shape,
                "target_latent_shape": list(clean_high.shape),
                "post_transition_steps": request.steps - index - 1,
                "noise_seed": (int(request.seed) + 1) % (2**63 - 1),
                "pixel_vae_anchor_rho": 0.0,
                "sampler": "turbo" if request.use_lora else "res_multistep",
                "video_history": (
                    "not_applicable" if request.use_lora else "lifted_current_x0"
                ),
                "continuation_handoff": (
                    "dual_resolution_exact_target_prefix_v1"
                    if restored_target_prefix_tokens
                    else "single_resolution_prefix_v1"
                ),
                "restored_target_resolution_prefix_tokens": (
                    restored_target_prefix_tokens
                ),
                "keyframe_condition_transition": (
                    "source_grid_to_target_grid_v1"
                    if selflift_target_keyframe_latents_cpu
                    else "not_applicable"
                ),
            }
            del clean_low_cpu, clean_audio_cpu, clean_high_cpu, target_noise
            return step_video, step_audio, previous_video, previous_audio

        def transition_continuation(
            index,
            clock,
            step_video,
            step_audio,
            previous_video,
            previous_audio,
        ):
            del index, clock
            if continuation_video_prefix is None or continuation_audio_prefix is None:
                raise RuntimeError("continuation transition is missing its AV prefix")
            restore_masked_av_prefix_(
                step_video,
                step_audio,
                continuation_video_prefix,
                continuation_audio_prefix,
            )
            return step_video, step_audio, previous_video, previous_audio

        def transition_continuation_selflift(
            index,
            clock,
            step_video,
            step_audio,
            previous_video,
            previous_audio,
        ):
            step_video, step_audio, previous_video, previous_audio = (
                transition_continuation(
                    index,
                    clock,
                    step_video,
                    step_audio,
                    previous_video,
                    previous_audio,
                )
            )
            return transition_selflift(
                index,
                clock,
                step_video,
                step_audio,
                previous_video,
                previous_audio,
            )

        def transition_resume_protected_prefix(
            index,
            clock,
            step_video,
            step_audio,
            previous_video,
            previous_audio,
        ):
            del index, clock
            if (
                resume_protected_video_prefix is None
                or resume_protected_audio_prefix is None
            ):
                raise RuntimeError("formal resume prefix state is incomplete")
            restore_masked_av_prefix_(
                step_video,
                step_audio,
                resume_protected_video_prefix,
                resume_protected_audio_prefix,
            )
            return step_video, step_audio, previous_video, previous_audio

        def transition_refinement_first_anchor(
            index,
            clock,
            step_video,
            step_audio,
            previous_video,
            previous_audio,
        ):
            del index
            if (
                refinement_first_anchor_clean is None
                or refinement_first_anchor_noise is None
                or refinement_first_anchor_weights is None
            ):
                raise RuntimeError("refinement first-frame anchor is incomplete")
            count = int(refinement_first_anchor_clean.shape[2])
            sigma_next = float(clock.video_sigma_next)
            target = (
                sigma_next * refinement_first_anchor_noise
                + (1.0 - sigma_next) * refinement_first_anchor_clean
            )
            current = step_video[:, :, :count]
            current.mul_(1.0 - refinement_first_anchor_weights).add_(
                target * refinement_first_anchor_weights
            )
            return step_video, step_audio, previous_video, previous_audio

        current_context = context
        current_text_tags = text_tags
        active_context = current_context
        active_text_tags = current_text_tags
        active_output_frame_count = request.frames
        active_target_time_offset = 0.0
        global_window_layouts: list[Any | None] = (
            []
            if global_co_plan is None
            else [None for _ in global_co_plan.windows]
        )

        def predict_one(video_value, audio_value, clock, *, step_index, is_actual_step):
            nonlocal layout, last_denoised, active_context, active_text_tags
            nonlocal refinement_previous_video_denoised
            raise_if_cancelled()
            if refinement_sampler_mask is not None:
                # Match ComfyUI-H3-FaceRefine-Accelerated's sampler-only mask:
                # held cells are re-noised to the current flow clock before
                # each prediction, while H3 never receives the mask as a
                # duplicated timestep condition (the historical grid source).
                assert refinement_sampler_noise is not None
                assert refinement_motion_video is not None
                current_sigma = float(clock.video_sigma)
                held = (
                    current_sigma * refinement_sampler_noise
                    + (1.0 - current_sigma) * refinement_motion_video.float()
                )
                video_value = held + refinement_sampler_mask * (
                    video_value.float() - held
                )
            step_started = time.perf_counter()
            active_text_route = "current"
            boundary_velocity_blend = bool(
                continuation_text_bridge_scheduled
                and _continuation_text_route_name(
                    step_index,
                    request.steps,
                )
                == "boundary"
            )
            layout_velocity_guidance_active = False
            if layout_velocity_guidance_scheduled:
                guidance_start, guidance_steps = (
                    _explicit_camera_layout_guidance_schedule(request.steps)
                )
                layout_velocity_guidance_active = (
                    guidance_start
                    <= step_index
                    < guidance_start + guidance_steps
                )
            # Current semantics own the complete trajectory on every step.
            # The boundary condition is an auxiliary velocity estimate below,
            # never a replacement global text route.
            active_context = current_context
            active_text_tags = current_text_tags
            block_stack_runner = None
            if forecast is not None:

                def block_stack_runner(stack, value, **kwargs):
                    return forecast.run_block_stack(
                        stack,
                        value,
                        step_index=step_index,
                        requested_actual=is_actual_step,
                        **kwargs,
                    )
            interleave = None
            if (
                execution_plan is not None
                and execution_plan.frame_interleave_stride > 1
                and step_index not in execution_plan.frame_interleave_dense_steps
            ):
                interleave = FrameInterleaveConfig(
                    stride=execution_plan.frame_interleave_stride,
                    layer_start=execution_plan.frame_interleave_layer_start,
                    layer_stop=execution_plan.frame_interleave_layer_stop,
                    dense_layers=execution_plan.frame_interleave_dense_layers,
                )
            query_lattice = None
            if (
                execution_plan is not None
                and execution_plan.spatial_query_lattice_stride > 1
                and step_index
                not in execution_plan.spatial_query_lattice_dense_steps
            ):
                query_lattice = SpatialQueryLatticeConfig(
                    stride=execution_plan.spatial_query_lattice_stride,
                    layer_start=(
                        execution_plan.spatial_query_lattice_layer_start
                    ),
                    layer_stop=execution_plan.spatial_query_lattice_layer_stop,
                    dense_layers=(
                        execution_plan.spatial_query_lattice_dense_layers
                    ),
                    phase_offset=step_index,
                )
            mlp_lattice = None
            if (
                execution_plan is not None
                and execution_plan.mlp_spatial_lattice_stride > 1
                and step_index not in execution_plan.mlp_spatial_lattice_dense_steps
            ):
                mlp_lattice = MLPSpatialLatticeConfig(
                    stride=execution_plan.mlp_spatial_lattice_stride,
                    layer_start=execution_plan.mlp_spatial_lattice_layer_start,
                    layer_stop=execution_plan.mlp_spatial_lattice_layer_stop,
                    dense_layers=execution_plan.mlp_spatial_lattice_dense_layers,
                    phase_offset=step_index,
                    detail_fraction=(
                        execution_plan.mlp_spatial_lattice_detail_fraction
                    ),
                )

            active_conditioning_route: str | None = None
            active_condition_video_latents = condition_video_latents
            active_condition_audio_latents = condition_audio_latents
            active_reference_shapes = reference_shapes
            active_reference_kinds = reference_kinds
            active_reference_audio_frames = reference_audio_frames
            active_conditioning_layout = layout
            if conditioning_route_scheduled:
                active_conditioning_route = _progressive_conditioning_route_name(
                    step_index,
                    request.steps,
                    visual_schedule=visual_conditioning_scheduled,
                    inferred_voice_bootstrap=(
                        inferred_voice_conditioning_scheduled
                    ),
                    layout_memory_bootstrap=(
                        layout_memory_conditioning_scheduled
                    ),
                    state_seeded_layout_bootstrap=(
                        layout_velocity_guidance_scheduled
                    ),
                    novel_camera_layout_probe=(
                        novel_camera_layout_probe_scheduled
                    ),
                )
                routed = conditioning_routes[active_conditioning_route]
                active_condition_video_latents = routed.video_latents
                active_condition_audio_latents = routed.audio_latents
                active_reference_shapes = routed.reference_shapes
                active_reference_kinds = routed.reference_kinds
                active_reference_audio_frames = routed.reference_audio_frames
            active_layout_key: tuple[str, str] | None = None
            if continuation_text_bridge_scheduled or conditioning_route_scheduled:
                active_layout_key = (
                    active_text_route,
                    active_conditioning_route or "default",
                )
                active_conditioning_layout = conditioning_route_layouts.get(
                    active_layout_key
                )

            def run_dit_once(
                *,
                text_context=active_context,
                token_tags=active_text_tags,
                conditioning_layout=active_conditioning_layout,
                stack_runner=block_stack_runner,
                video_conditions=active_condition_video_latents,
                audio_conditions=active_condition_audio_latents,
                video_reference_shapes=active_reference_shapes,
                video_reference_kinds=active_reference_kinds,
                audio_reference_frames=active_reference_audio_frames,
            ):
                with block_cancellation(raise_if_cancelled):
                    return dit(
                        video_value,
                        audio_value,
                        text_context,
                        torch.tensor([clock.video_sigma], device="cuda"),
                        output_frame_count=active_output_frame_count,
                        text_token_tags=token_tags,
                        condition_video_latents=video_conditions,
                        condition_audio_latents=audio_conditions,
                        keyframe_indices=keyframe_indices,
                        reference_shapes=video_reference_shapes,
                        reference_kinds=video_reference_kinds,
                        reference_audio_frames=audio_reference_frames,
                        condition_seed=(
                            request.seed if self._uses_reference_layout else 42
                        ),
                        cache_condition_rows=request.cache_condition_rows,
                        cache_condition_embeddings=(
                            request.cache_condition_embeddings
                        ),
                        layout=conditioning_layout,
                        audio_transport_scale=(
                            4.0
                            if active_lora_for_predict
                            and self.turbo_clock_mode
                            is TurboClockMode.SHARED_VIDEO
                            else None
                        ),
                        sigma_shift_video=(
                            self.lora_video_shift
                            if active_lora_for_predict else 12.0
                        ),
                        sigma_shift_audio=(
                            self.lora_audio_shift
                            if active_lora_for_predict else 3.0
                        ),
                        block_stack_runner=stack_runner,
                        mlp_chunk_tokens=(
                            execution_plan.mlp_chunk_tokens
                            if execution_plan is not None
                            else request.mlp_chunk_tokens
                        ),
                        final_projection_chunk_tokens=(
                            2048
                            if self.runtime_config.weight_tier == "w4a8"
                            else None
                        ),
                        masked_video_prefix_latent_frames=(
                            0
                            if continuation_video_prefix is None
                            else int(continuation_video_prefix.shape[2])
                        ),
                        masked_audio_prefix_latent_frames=(
                            0
                            if continuation_audio_prefix is None
                            else int(continuation_audio_prefix.shape[-1])
                        ),
                        target_time_offset=active_target_time_offset,
                    )

            with (
                torch.inference_mode(),
                attention_actual_steps(actual_steps),
                attention_step(step_index, request.steps),
                attention_action_schedule_context(request_attention_schedule),
                attention_online_budget(request_online_budget),
                attention_sparsity(step_attention_topk(step_index)),
                frame_interleave_config(interleave),
                spatial_query_lattice_config(query_lattice),
                mlp_spatial_lattice_config(mlp_lattice),
                dense_qk_quantization(
                    str(execution_profile["dense_qk_quant_gran"])
                ),
                rms_adaln_fusion(
                    False
                    if execution_plan is None
                    else execution_plan.fused_rms_adaln
                ),
                long_video_attention(
                    False
                    if execution_plan is None
                    else execution_plan.long_video_motion_detail_attention
                ),
                long_sequence_query_chunking(
                    None
                    if execution_plan is None
                    else execution_plan.long_sequence_query_chunk_tokens,
                    projection_chunk_tokens=(
                        8192
                        if execution_plan is None
                        else execution_plan.long_sequence_projection_chunk_tokens
                    ),
                    split_qkv_outputs=(
                        False
                        if execution_plan is None
                        else execution_plan.long_sequence_split_qkv_outputs
                    ),
                    shared_qkv_quantization=(
                        False
                        if execution_plan is None
                        else execution_plan.long_sequence_shared_qkv_quantization
                    ),
                    compact_kv=(
                        False
                        if execution_plan is None
                        else execution_plan.long_sequence_compact_kv
                    ),
                    exact_helper_stack=(
                        False
                        if execution_plan is None
                        else execution_plan.long_sequence_exact_helper_stack
                    ),
                    single_qknorm_rope=(
                        False
                        if execution_plan is None
                        else execution_plan.long_sequence_single_qknorm_rope
                    ),
                    parallel_sparse_lut=(
                        False
                        if execution_plan is None
                        else execution_plan.long_sequence_parallel_sparse_lut
                    ),
                    partial_sparse_topk=(
                        False
                        if execution_plan is None
                        else execution_plan.long_sequence_partial_sparse_topk
                    ),
                    fused_prefix_k_quant=(
                        False
                        if execution_plan is None
                        else execution_plan.long_sequence_fused_prefix_k_quant
                    ),
                    fused_query_projection=(
                        False
                        if execution_plan is None
                        else execution_plan.long_sequence_fused_query_projection
                    ),
                    fused_qknorm_hnd_layout=(
                        False
                        if execution_plan is None
                        else execution_plan.long_sequence_fused_qknorm_hnd_layout
                    ),
                    direct_nhd_output=(
                        False
                        if execution_plan is None
                        else execution_plan.long_sequence_direct_nhd_output
                    ),
                    direct_nhd_kv=(
                        False
                        if execution_plan is None
                        else execution_plan.long_sequence_direct_nhd_kv
                    ),
                    direct_hnd_fp8_value=(
                        False
                        if execution_plan is None
                        else execution_plan.long_sequence_direct_hnd_fp8_value
                    ),
                ),
            ):
                verify_whole_dit = (
                    is_actual_step
                    and step_index in self.self_speculative_verify_steps
                )
                if verify_whole_dit and (
                    forecast is None or forecast.segment_cache is not None
                ):
                    raise RuntimeError(
                        "whole-DiT speculative verification requires the "
                        "directional forecast controller without segment cache"
                    )
                history_before = list(forecast.history) if verify_whole_dit else None
                records_before = list(forecast.records) if verify_whole_dit else None
                result = run_dit_once()
                if verify_whole_dit:
                    assert forecast is not None
                    draft_history = list(forecast.history)
                    draft_records = list(forecast.records)
                    forecast.history = history_before
                    forecast.records = records_before
                    with attention_force_dense():
                        exact_result = run_dit_once()

                    def relative_rms(reference, candidate):
                        difference = (
                            (reference.float() - candidate.float())
                            .square()
                            .mean()
                            .sqrt()
                        )
                        scale = (
                            reference.float().square().mean().sqrt().clamp_min(1e-6)
                        )
                        return float((difference / scale).item())

                    video_error = relative_rms(exact_result.video, result.video)
                    audio_error = relative_rms(exact_result.audio, result.audio)
                    rejected = max(video_error, audio_error) > float(
                        self.self_speculative_verify_threshold
                    )
                    self_speculative_records.append(
                        {
                            "step": int(step_index),
                            "video_relative_rms": video_error,
                            "audio_relative_rms": audio_error,
                            "threshold": float(
                                self.self_speculative_verify_threshold
                            ),
                            "decision": "dense_rollback" if rejected else "accept_draft",
                        }
                    )
                    if rejected:
                        result = exact_result
                    else:
                        forecast.history = draft_history
                        forecast.records = draft_records
                if layout_velocity_guidance_active:
                    if active_conditioning_route is None:
                        raise RuntimeError(
                            "layout velocity guidance is missing its state-primary route"
                        )
                    guidance_route_name = _layout_guidance_route_name(
                        active_conditioning_route
                    )
                    guidance_route = conditioning_routes[guidance_route_name]
                    guidance_layout_key = (
                        active_text_route,
                        guidance_route_name,
                    )
                    # The state-primary prediction alone advances any forecast
                    # controller. The camera estimate is exact and stateless,
                    # then contributes uniformly to every video time in this
                    # window; audio remains entirely state-primary.
                    guidance_result = run_dit_once(
                        conditioning_layout=conditioning_route_layouts.get(
                            guidance_layout_key
                        ),
                        stack_runner=None,
                        video_conditions=guidance_route.video_latents,
                        audio_conditions=guidance_route.audio_latents,
                        video_reference_shapes=guidance_route.reference_shapes,
                        video_reference_kinds=guidance_route.reference_kinds,
                        audio_reference_frames=(
                            guidance_route.reference_audio_frames
                        ),
                    )
                    conditioning_route_layouts[guidance_layout_key] = (
                        guidance_result.layout
                    )
                    result.video = _blend_layout_guidance_velocity(
                        result.video,
                        guidance_result.video,
                    )
                    del guidance_result
                if boundary_velocity_blend:
                    assert continuation_text_bridge is not None
                    if continuation_video_prefix is None:
                        raise RuntimeError(
                            "continuation boundary velocity blend is missing "
                            "its protected video prefix"
                        )
                    boundary_layout_key = (
                        "boundary",
                        active_conditioning_route or "default",
                    )
                    boundary_context, boundary_tags = (
                        continuation_text_bridge
                    )
                    boundary_block_stack_runner = None
                    if boundary_forecast is not None:
                        def boundary_block_stack_runner(stack, value, **kwargs):
                            return boundary_forecast.run_block_stack(
                                stack,
                                value,
                                step_index=step_index,
                                requested_actual=is_actual_step,
                                **kwargs,
                            )
                    # The auxiliary owns an independent Forecast history
                    # because its boundary text differs from the current
                    # prompt. It mirrors the primary Actual/Forecast schedule
                    # instead of executing an extra exact DiT on every step.
                    boundary_result = run_dit_once(
                        text_context=boundary_context,
                        token_tags=boundary_tags,
                        conditioning_layout=conditioning_route_layouts.get(
                            boundary_layout_key
                        ),
                        stack_runner=boundary_block_stack_runner,
                    )
                    conditioning_route_layouts[boundary_layout_key] = (
                        boundary_result.layout
                    )
                    context_video_tokens = video_latent_frames(
                        request.continuation_context_frames
                    )
                    hidden_repaint_tokens = max(
                        0,
                        context_video_tokens
                        - int(continuation_video_prefix.shape[2]),
                    )
                    visible_fade_tokens = min(
                        _CONTINUATION_BOUNDARY_VISIBLE_FADE_TOKENS,
                        int(result.video.shape[2]) - context_video_tokens,
                    )
                    blend_band_tokens = (
                        hidden_repaint_tokens + visible_fade_tokens
                    )
                    execution_profile["continuation_text_bridge"][
                        "blend_band_video_latent_tokens"
                    ] = blend_band_tokens
                    execution_profile["continuation_text_bridge"][
                        "hidden_repaint_video_latent_tokens"
                    ] = hidden_repaint_tokens
                    execution_profile["continuation_text_bridge"][
                        "visible_fade_video_latent_tokens"
                    ] = visible_fade_tokens
                    result.video = _blend_continuation_boundary_velocity(
                        result.video,
                        boundary_result.video,
                        protected_prefix_tokens=int(
                            continuation_video_prefix.shape[2]
                        ),
                        hidden_repaint_tokens=hidden_repaint_tokens,
                        visible_fade_tokens=visible_fade_tokens,
                    )
                    del boundary_result
            if guard_approximate_math:
                for modality, prediction in (
                    ("video", result.video),
                    ("audio", result.audio),
                ):
                    if not bool(torch.isfinite(prediction).all().item()):
                        raise FloatingPointError(
                            f"non-finite {modality} DiT prediction at sampling step "
                            f"{step_index} of {request.steps}"
                        )
            if active_layout_key is not None:
                conditioning_route_layouts[active_layout_key] = result.layout
            layout = result.layout
            torch.cuda.synchronize()
            step_seconds.append(time.perf_counter() - step_started)
            sigma = clock.video_sigma
            video_denoised = video_value - result.video * sigma
            if refinement_sampler_mask is not None:
                assert refinement_motion_video is not None
                video_denoised = refinement_motion_video.float() + (
                    refinement_sampler_mask
                    * (video_denoised.float() - refinement_motion_video.float())
                )
            if (
                refinement_motion_video is not None
                and request.refinement_prediction_low_frequency_gain < 1.0
            ):
                assert refinement_source_latent_height is not None
                assert refinement_source_latent_width is not None
                video_denoised = blend_terminal_refinement_detail(
                    refinement_motion_video,
                    video_denoised,
                    source_height=refinement_source_latent_height,
                    source_width=refinement_source_latent_width,
                    low_frequency_gain=(
                        request.refinement_prediction_low_frequency_gain
                    ),
                )
            if refinement_full_canvas_mask is not None:
                video_denoised = refinement_motion_video.float() + (
                    refinement_full_canvas_mask
                    * (
                        video_denoised.float()
                        - refinement_motion_video.float()
                    )
                )
            if (
                refinement_motion_video is not None
                and (
                    request.refinement_cross_step_detail_strength > 0.0
                    or request.refinement_roi_auto
                )
                and step_index == request.steps - 2
            ):
                refinement_previous_video_denoised = (
                    video_denoised.detach().clone()
                )
            prediction = AVPrediction(
                video_denoised=video_denoised,
                audio_denoised=audio_value - result.audio * sigma,
            )
            last_denoised = (
                prediction.video_denoised,
                prediction.audio_denoised,
            )
            return prediction

        def predict(video_value, audio_value, clock, *, step_index, is_actual_step):
            """Predict either one native clip or one global WWS solver state."""

            nonlocal active_context, active_text_tags, active_output_frame_count
            nonlocal active_target_time_offset
            nonlocal layout, last_denoised
            if global_co_plan is None:
                return predict_one(
                    video_value,
                    audio_value,
                    clock,
                    step_index=step_index,
                    is_actual_step=is_actual_step,
                )
            if forecast is not None:
                raise RuntimeError(
                    "global co-denoise requires window-local forecast histories"
                )

            saved_context = active_context
            saved_tags = active_text_tags
            saved_frames = active_output_frame_count
            saved_time_offset = active_target_time_offset
            saved_layout = layout

            def predict_window(window, local_video, local_audio):
                nonlocal active_context, active_text_tags
                nonlocal active_output_frame_count, active_target_time_offset, layout
                active_context = projected_contexts[window.index]
                active_text_tags = projected_text_tags[window.index]
                active_output_frame_count = window.frames
                # Localized prompts have different token counts.  Offsetting
                # only target tokens after those variable text prefixes makes
                # two views assign incompatible text/target RoPE relations to
                # the same overlap.  The default keeps each view in H3's
                # trained local chart; the shared tensor slices and one global
                # scheduler state, rather than RoPE extrapolation, own global
                # chronology.  The old target-only offset remains an explicit
                # research ablation.
                active_target_time_offset = (
                    float(window.audio_start)
                    if request.global_co_denoise_rotary_mode
                    == "absolute_target"
                    else 0.0
                )
                layout = global_window_layouts[window.index]
                prediction = predict_one(
                    local_video,
                    local_audio,
                    clock,
                    step_index=step_index,
                    # V1 deliberately measures the exact co-denoise algorithm
                    # before adding one independent forecast history per view.
                    is_actual_step=True,
                )
                global_window_layouts[window.index] = layout
                return prediction.video_denoised, prediction.audio_denoised

            try:
                fused_video, fused_audio = fuse_global_av_predictions(
                    video_value,
                    audio_value,
                    global_co_plan,
                    predict_window,
                )
                if global_selflift_anchor_video is not None:
                    fused_video = stabilize_global_selflift_seams(
                        global_selflift_anchor_video,
                        fused_video,
                        global_co_plan,
                        strength=0.85,
                        padding_tokens=2,
                    )
            finally:
                active_context = saved_context
                active_text_tags = saved_tags
                active_output_frame_count = saved_frames
                active_target_time_offset = saved_time_offset
                layout = saved_layout
            if preserved_global_selflift_audio is not None:
                # The high-resolution branch improves video only.  Supplying
                # the accepted source audio x0 as the denoised prediction at
                # every scheduler update also keeps the audio state used by
                # the next joint AV evaluation on the original trajectory.
                fused_audio = preserved_global_selflift_audio
            last_denoised = (fused_video, fused_audio)
            return AVPrediction(
                video_denoised=fused_video,
                audio_denoised=fused_audio,
            )

        def finish_preview_branch(
            index,
            clock,
            step_video,
            step_audio,
            *,
            branch_steps=None,
            branch_spatial_scale=None,
            branch_warm_history=None,
            branch_force_dense=None,
            branch_use_lora=None,
            branch_forecast_only=False,
            branch_actual_step_indices=None,
        ):
            """Fast-finish a disposable branch without mutating main solver state."""

            nonlocal forecast, layout, last_denoised
            nonlocal condition_video_latents, reference_shapes
            nonlocal active_lora_for_predict
            sigma_start = float(clock.video_sigma_next)
            if sigma_start <= 0.0:
                if last_denoised is None:
                    raise RuntimeError("preview branch has no denoised estimate")
                return last_denoised[0].clone(), last_denoised[1].clone()
            count = int(
                request.preview_branch_steps
                if branch_steps is None else branch_steps
            )
            # If the branch has exactly as many evaluations as the remaining
            # formal trajectory, retain the model's trained sigma suffix.
            # Replacing Larry's shifted suffix with a linear one can create a
            # regular grid across the entire preview.
            formal_tail = tuple(float(value) for value in sigmas[index + 1 :])
            use_formal_tail = (
                len(formal_tail) == count + 1
                and math.isclose(formal_tail[0], sigma_start)
            )
            branch_sigmas = (
                formal_tail
                if use_formal_tail
                else tuple(
                    sigma_start * (1.0 - offset / count)
                    for offset in range(count + 1)
                )
            )
            use_lora_override = (
                request.preview_branch_use_lora
                if branch_use_lora is None else branch_use_lora
            )
            branch_uses_lora = bool(request.use_lora or use_lora_override)
            requested_branch_actual = (
                request.preview_branch_actual_step_indices
                if branch_actual_step_indices is None
                else branch_actual_step_indices
            )
            branch_actual = (
                ()
                if branch_forecast_only
                else tuple(range(count))
                if requested_branch_actual is None
                else tuple(requested_branch_actual)
            )
            branch_plan = SamplingPlan(
                sampler="turbo" if branch_uses_lora else "res_multistep",
                video_sigmas=branch_sigmas,
                audio_sigmas=branch_sigmas,
                actual_step_indices=branch_actual,
                video_shift=(
                    self.lora_video_shift if branch_uses_lora else 12.0
                ),
                audio_shift=(
                    self.lora_audio_shift if branch_uses_lora else 3.0
                ),
            )
            saved_forecast = forecast
            saved_layout = layout
            saved_denoised = last_denoised
            saved_condition_video_latents = condition_video_latents
            saved_reference_shapes = reference_shapes
            saved_lora_mode = active_lora_for_predict
            saved_step_count = len(step_seconds)
            saved_forecast_record_count = (
                None if saved_forecast is None else len(saved_forecast.records)
            )
            branch_uses_forecast = len(branch_actual) < count
            saved_forecast_history = (
                None if saved_forecast is None else list(saved_forecast.history)
            )
            if branch_uses_forecast:
                if saved_forecast is None or len(saved_forecast.history) < 2:
                    raise RuntimeError(
                        "forecast preview requires two formal actual observations"
                    )
                forecast = saved_forecast
            else:
                forecast = None
            try:
                if branch_uses_lora != saved_lora_mode:
                    if set_active_lora(branch_uses_lora) == 0:
                        raise RuntimeError("LoRA preview adapters are unavailable")
                    active_lora_for_predict = branch_uses_lora
                branch_video = step_video.clone()
                branch_audio = step_audio.clone()
                previous_video = None
                previous_audio = None
                previous_video_sigma = None
                previous_audio_sigma = None
                warm_history = (
                    request.preview_branch_warm_history
                    if branch_warm_history is None else branch_warm_history
                )
                if warm_history and saved_denoised is not None:
                    previous_video = saved_denoised[0].clone()
                    previous_audio = saved_denoised[1].clone()
                    previous_video_sigma = float(clock.video_sigma)
                    previous_audio_sigma = float(clock.audio_sigma)

                scale = float(
                    request.preview_branch_spatial_scale
                    if branch_spatial_scale is None else branch_spatial_scale
                )
                if scale < 1.0:
                    source_height = int(branch_video.shape[-2])
                    source_width = int(branch_video.shape[-1])
                    target_height = max(
                        2, int(round(source_height * scale / 2.0)) * 2
                    )
                    target_width = max(
                        2, int(round(source_width * scale / 2.0)) * 2
                    )
                    branch_video = resize_refinement_video_latent_spatial(
                        branch_video,
                        target_height=target_height,
                        target_width=target_width,
                    )
                    if previous_video is not None:
                        previous_video = resize_refinement_video_latent_spatial(
                            previous_video,
                            target_height=target_height,
                            target_width=target_width,
                        )
                    condition_video_latents = tuple(
                        resize_refinement_video_latent_spatial(
                            latent,
                            target_height=max(
                                2,
                                int(round(int(latent.shape[-2]) * scale / 2.0))
                                * 2,
                            ),
                            target_width=max(
                                2,
                                int(round(int(latent.shape[-1]) * scale / 2.0))
                                * 2,
                            ),
                        )
                        for latent in saved_condition_video_latents
                    )
                    if saved_reference_shapes:
                        reference_shapes = tuple(
                            tuple(int(value) for value in latent.shape[-3:])
                            for latent in condition_video_latents
                        )
                    layout = None

                def branch_predict(
                    video_value,
                    audio_value,
                    branch_clock,
                    *,
                    step_index,
                    is_actual_step,
                ):
                    # Approximate-attention policies are indexed by the formal
                    # trajectory, not by this branch's local 0..N counter.
                    routed_step = min(
                        range(request.steps),
                        key=lambda candidate: abs(
                            float(sigmas[candidate])
                            - float(branch_clock.video_sigma)
                        ),
                    )
                    return predict(
                        video_value,
                        audio_value,
                        branch_clock,
                        step_index=routed_step,
                        is_actual_step=is_actual_step,
                    )

                branch_sampler = (
                    TurboAVSampler(self.turbo_clock_mode)
                    if branch_uses_lora else ResMultistepAVSampler()
                )
                force_dense = (
                    request.preview_branch_force_dense
                    if branch_force_dense is None else branch_force_dense
                )
                with attention_force_dense(force_dense):
                    if branch_uses_lora:
                        branch_video, branch_audio = branch_sampler.sample(
                            branch_video,
                            branch_audio,
                            branch_plan,
                            branch_predict,
                            cancel_check=raise_if_cancelled,
                        )
                    else:
                        branch_video, branch_audio = branch_sampler.sample(
                            branch_video,
                            branch_audio,
                            branch_plan,
                            branch_predict,
                            cancel_check=raise_if_cancelled,
                            initial_previous_video=previous_video,
                            initial_previous_audio=previous_audio,
                            initial_previous_video_sigma=previous_video_sigma,
                            initial_previous_audio_sigma=previous_audio_sigma,
                        )
                if (
                    branch_uses_lora
                    and self.turbo_clock_mode is TurboClockMode.SHARED_VIDEO
                ):
                    branch_audio.div_(4.0)
                return branch_video, branch_audio
            finally:
                forecast = saved_forecast
                layout = saved_layout
                last_denoised = saved_denoised
                condition_video_latents = saved_condition_video_latents
                reference_shapes = saved_reference_shapes
                if active_lora_for_predict != saved_lora_mode:
                    set_active_lora(saved_lora_mode)
                    active_lora_for_predict = saved_lora_mode
                if (
                    saved_forecast is not None
                    and saved_forecast_record_count is not None
                ):
                    del saved_forecast.records[saved_forecast_record_count:]
                if saved_forecast is not None and saved_forecast_history is not None:
                    saved_forecast.history = saved_forecast_history
                del step_seconds[saved_step_count:]

        def save_formal_checkpoint(index, clock, step_video, step_audio) -> Path:
            """Atomically persist the exact formal state after one solver step."""

            if request.checkpoint_state_path is None:
                raise RuntimeError("checkpoint path is unavailable")
            checkpoint_path = Path(request.checkpoint_state_path).resolve()
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
            online_runtime_state = None
            if request_online_budget is not None:
                checkpoint_verifier = getattr(
                    attention_backend,
                    "online_checkpoint_state",
                    None,
                )
                if checkpoint_verifier is None:
                    raise RuntimeError(
                        "active attention backend cannot checkpoint online state"
                    )
                online_runtime_state = {
                    "budget": request_online_budget.checkpoint_state(),
                    "verifier": checkpoint_verifier(request_online_budget),
                }
            document: dict[str, Any] = {
                "schema_version": 1,
                "representation": "formal_sampler_checkpoint_v1",
                "video": step_video.detach().cpu(),
                "audio": step_audio.detach().cpu(),
                "frames": request.frames,
                "fps": request.fps,
                "width": request.width,
                "height": request.height,
                "engine": request_engine,
                "seed": request.seed,
                "steps": request.steps,
                "step_index": index,
                "next_step_index": index + 1,
                "sigma": float(clock.video_sigma),
                "sigma_next": float(clock.video_sigma_next),
                "sigmas": list(simple_sigma_schedule(
                    request.steps, primary_video_shift
                )),
                "actual_step_indices": list(actual_steps),
                "attention_action_schedule": list(
                    request.attention_action_schedule
                ),
                "attention_online_guard_id": request.attention_online_guard_id,
                "attention_online_rebate_schedule": list(
                    request.attention_online_rebate_schedule
                ),
                "attention_online_budget_dense_layers": (
                    request.attention_online_budget_dense_layers
                ),
                "attention_online_runtime_state": online_runtime_state,
                "prompt_sha256": hashlib.sha256(
                    request.prompt.encode("utf-8")
                ).hexdigest(),
                "use_lora": request.use_lora,
                "lora_profile_id": (
                    self.lora_profile_id if request.use_lora else None
                ),
                "forecast_state": (
                    None if forecast is None else forecast.checkpoint_state()
                ),
            }
            if checkpoint_protected_video_prefix_cpu is not None:
                if checkpoint_protected_audio_prefix_cpu is None:
                    raise RuntimeError(
                        "SelfLift continuation checkpoint has no protected audio prefix"
                    )
                document["protected_video_prefix"] = (
                    checkpoint_protected_video_prefix_cpu
                )
                document["protected_audio_prefix"] = (
                    checkpoint_protected_audio_prefix_cpu
                )
            if selflift_source_video_x0_cpu is not None:
                if selflift_source_audio_x0_cpu is None:
                    raise RuntimeError(
                        "SelfLift checkpoint has no source-grid audio x0"
                    )
                document.update({
                    "selflift_source_video_x0": selflift_source_video_x0_cpu,
                    "selflift_source_audio_x0": selflift_source_audio_x0_cpu,
                    "selflift_source_width": request.multiscale_initial_width,
                    "selflift_source_height": request.multiscale_initial_height,
                    "selflift_split_step": index + 1,
                    "selflift_source_representation": (
                        "formal_clean_x0_before_learned_lift_v1"
                    ),
                })
                if (
                    request.preview_latents_path is not None
                    and not preview_published
                ):
                    source_path = Path(request.preview_latents_path).resolve()
                    source_path.parent.mkdir(parents=True, exist_ok=True)
                    source_document: dict[str, Any] = {
                        "video": selflift_source_video_x0_cpu,
                        "audio": selflift_source_audio_x0_cpu,
                        "frames": request.frames,
                        "fps": request.fps,
                        "width": request.multiscale_initial_width,
                        "height": request.multiscale_initial_height,
                        "engine": request_engine,
                        "seed": request.seed,
                        "step_index": index,
                        "sigma": 0.0,
                        "representation": "formal_selflift_source_x0_v1",
                    }
                    if self._last_conditioning_cache_payload is not None:
                        source_document["qwen_conditioning_cache"] = (
                            self._last_conditioning_cache_payload
                        )
                    temporary_source = source_path.with_suffix(
                        source_path.suffix + ".tmp"
                    )
                    torch.save(source_document, temporary_source)
                    temporary_source.replace(source_path)
            if not request.use_lora:
                if last_denoised is None:
                    raise RuntimeError("RES checkpoint has no denoised history")
                document.update({
                    "previous_video": last_denoised[0].detach().cpu(),
                    "previous_audio": last_denoised[1].detach().cpu(),
                    "previous_video_sigma": float(clock.video_sigma),
                    "previous_audio_sigma": float(clock.audio_sigma),
                })
            torch.save(document, temporary)
            temporary.replace(checkpoint_path)
            execution_profile["formal_checkpoint"] = {
                "checkpoint": str(checkpoint_path),
                "completed_steps": index + 1,
                "total_steps": request.steps,
                "sigma_next": float(clock.video_sigma_next),
                "formal_trajectory_mutated": False,
            }
            return checkpoint_path

        def publish_preview(index, clock, step_video, step_audio):
            """Decode, publish and optionally pause while the main state stays exact."""

            nonlocal dit, preview_latents, preview_published
            if request.preview_decode_mode == "direct_x0":
                if last_denoised is None:
                    raise RuntimeError(
                        "direct x0 preview requested before a DiT prediction exists"
                    )
                preview_video = last_denoised[0].clone()
                preview_audio = last_denoised[1].clone()
            else:
                preview_video, preview_audio = finish_preview_branch(
                    index, clock, step_video, step_audio
                )
            # Infinite creation carries a clean overlap from the accepted
            # preview trajectory.  The disposable finish branch has no
            # transition callback of its own, so restore that overlap once
            # before persisting/decoding the preview artifact.
            if (
                continuation_video_prefix is not None
                and continuation_audio_prefix is not None
            ):
                restore_masked_av_prefix_(
                    preview_video,
                    preview_audio,
                    continuation_video_prefix,
                    continuation_audio_prefix,
                )
            if (
                request.preview_decode_mode == "fast_finish"
                and request.preview_audio_branch_use_lora
            ):
                audio_branch_video, audio_branch_audio = finish_preview_branch(
                    index,
                    clock,
                    step_video,
                    step_audio,
                    branch_steps=request.preview_audio_branch_steps,
                    branch_spatial_scale=request.preview_audio_branch_spatial_scale,
                    branch_warm_history=False,
                    branch_force_dense=True,
                    branch_use_lora=True,
                )
                del audio_branch_video, preview_audio
                preview_audio = audio_branch_audio
            forecast_preview_latents = None
            if request.preview_forecast_steps > 0:
                forecast_video, forecast_audio = finish_preview_branch(
                    index,
                    clock,
                    step_video,
                    step_audio,
                    branch_steps=request.preview_forecast_steps,
                    branch_spatial_scale=1.0,
                    branch_warm_history=True,
                    branch_force_dense=True,
                    branch_use_lora=False,
                    branch_forecast_only=True,
                )
                forecast_preview_latents = (
                    forecast_video.detach().cpu(),
                    forecast_audio.detach().cpu(),
                )
                del forecast_video, forecast_audio
            preview_latents = (
                preview_video.detach().cpu(),
                preview_audio.detach().cpu(),
            )
            del preview_video, preview_audio
            execution_profile["intermediate_preview"] = {
                "step_index": index,
                "completed_sigma_positions": index + 1,
                "sigma": clock.video_sigma,
                "sigma_next": clock.video_sigma_next,
                "representation": (
                    "formal_step_x0_prediction"
                    if request.preview_decode_mode == "direct_x0"
                    else "isolated_fast_finish_branch"
                ),
                "decode_mode": request.preview_decode_mode,
                "branch_actual_steps": (
                    0
                    if request.preview_decode_mode == "direct_x0"
                    else request.preview_branch_steps
                    if request.preview_branch_actual_step_indices is None
                    else len(request.preview_branch_actual_step_indices)
                ),
                "branch_actual_step_indices": (
                    None
                    if request.preview_decode_mode == "direct_x0"
                    else request.preview_branch_actual_step_indices
                ),
                "branch_spatial_scale": request.preview_branch_spatial_scale,
                "branch_warm_history": request.preview_branch_warm_history,
                "branch_force_dense": request.preview_branch_force_dense,
                "branch_use_lora": request.preview_branch_use_lora,
                "audio_branch_use_lora": (
                    request.preview_decode_mode == "fast_finish"
                    and request.preview_audio_branch_use_lora
                ),
                "audio_branch_actual_steps": (
                    request.preview_audio_branch_steps
                    if request.preview_decode_mode == "fast_finish"
                    and request.preview_audio_branch_use_lora else None
                ),
                "audio_branch_spatial_scale": (
                    request.preview_audio_branch_spatial_scale
                    if request.preview_decode_mode == "fast_finish"
                    and request.preview_audio_branch_use_lora else None
                ),
                "video_source": "primary_preview_branch",
                "audio_source": (
                    "formal_step_x0_prediction"
                    if request.preview_decode_mode == "direct_x0"
                    else "lora_companion_branch"
                    if request.preview_audio_branch_use_lora
                    else "primary_preview_branch"
                ),
                "preview_width": int(preview_latents[0].shape[-1]) * 16,
                "preview_height": int(preview_latents[0].shape[-2]) * 16,
                "main_trajectory_mutated": False,
                "forecast_comparison_steps": request.preview_forecast_steps,
                "forecast_comparison_output": (
                    None
                    if preview_forecast_output is None
                    else str(preview_forecast_output)
                ),
            }
            if request.preview_latents_path is not None:
                preview_path = Path(request.preview_latents_path).resolve()
                preview_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "video": preview_latents[0], "audio": preview_latents[1],
                        # fast_finish reaches sigma=0. Global SelfLift may
                        # condition on this audio, but must never rewrite it.
                        "audio_final": (
                            request.preview_decode_mode == "fast_finish"
                        ),
                        "frames": request.frames, "fps": request.fps,
                        "width": int(preview_latents[0].shape[-1]) * 16,
                        "height": int(preview_latents[0].shape[-2]) * 16,
                        "engine": request_engine, "seed": request.seed,
                        "step_index": index, "sigma": clock.video_sigma,
                        "sigma_next": clock.video_sigma_next,
                        "representation": (
                            "formal_step_x0_prediction"
                            if request.preview_decode_mode == "direct_x0"
                            else "isolated_fast_finish_branch"
                        ),
                    },
                    preview_path,
                )

            latent_only_preview = (
                request.preview_output_path is None
                and request.preview_latents_path is not None
            )
            if latent_only_preview:
                # JSON/direct creation needs the completed low-resolution AV
                # branch but has no creator-facing preview to display. Keep
                # DiT resident and skip both VAEs plus muxing.
                execution_profile["intermediate_preview"]["decode_skipped"] = True
                forecast_preview_latents = None
                preview_latents = None
                preview_published = True
                return

            # Evict DiT, decode the disposable branch, then restore the exact
            # same graph before the blocked main sampler resumes.
            self.transformer.move_to("cpu", non_blocking=False)
            self._clear_block_executor()
            self._release_device()
            self.video_vae.move_to("cuda:0", non_blocking=True)
            preview_width = int(preview_latents[0].shape[-1]) * 16
            preview_height = int(preview_latents[0].shape[-2]) * 16
            if execution_plan is not None and execution_plan.vae_spatial_tile is not None:
                tile_height, tile_width = execution_plan.vae_spatial_tile
                if tile_height != tile_width:
                    raise ValueError(
                        "the current H3 Video-VAE supports square tiles only"
                    )
                video_vae_model = self.video_vae.value
                if not hasattr(video_vae_model, "decoder_tile_size"):
                    raise TypeError(
                        "the H3 Video-VAE does not expose decoder_tile_size"
                    )
                video_vae_model.decoder_tile_size = tile_height
            from .adapters.vae_tiling import configure_vae_tile_batching
            from .adapters.vae_compile import (
                transformer_block_compile,
                transformer_block_compile_ready,
            )

            configure_vae_tile_batching(
                self.video_vae.value,
                1 if execution_plan is None else execution_plan.vae_tile_batch_size,
            )
            compile_vae_requested = bool(
                execution_plan is not None
                and execution_plan.vae_transformer_block_compile
            )
            compile_vae_block = bool(
                compile_vae_requested
                and transformer_block_compile_ready(self.video_vae.value)
            )
            with self._video_vae_block_streaming(
                self.video_vae.value,
                width=preview_width,
                height=preview_height,
            ) as preview_block_policy:
                with transformer_block_compile(compile_vae_block):
                    decoded_video = self._decode_video_for_plan(
                        self.video_vae.value,
                        preview_latents[0].to("cuda:0"),
                        request.frames,
                        execution_plan,
                    )
                    decoded_forecast_video = (
                        None
                        if forecast_preview_latents is None
                        else self._decode_video_for_plan(
                            self.video_vae.value,
                            forecast_preview_latents[0].to("cuda:0"),
                            request.frames,
                            execution_plan,
                        )
                    )
            execution_profile["intermediate_preview"]["video_vae_decode"] = {
                "decoder_tile_size": (
                    None
                    if execution_plan is None
                    or execution_plan.vae_spatial_tile is None
                    else execution_plan.vae_spatial_tile[0]
                ),
                "tile_batch_size": (
                    1
                    if execution_plan is None
                    else execution_plan.vae_tile_batch_size
                ),
                "transformer_block_compile_requested": compile_vae_requested,
                "transformer_block_compile_enabled": compile_vae_block,
                "block_streaming": preview_block_policy,
            }
            self.video_vae.move_to("cpu", non_blocking=False)
            self._release_device()
            self.audio_vae.move_to("cuda:0", non_blocking=True)
            decoded_audio = self.decode_audio(
                self.audio_vae.value, preview_latents[1].to("cuda:0")
            )
            decoded_forecast_audio = (
                None
                if forecast_preview_latents is None
                else self.decode_audio(
                    self.audio_vae.value,
                    forecast_preview_latents[1].to("cuda:0"),
                )
            )
            self.audio_vae.move_to("cpu", non_blocking=False)
            self._release_device()
            assert preview_output is not None
            AtomicPyAVMuxer(output_root=self.output_root).write(
                video=decoded_video, audio=decoded_audio,
                sample_rate=32000, fps=request.fps,
                output_path=preview_output, cancel_check=raise_if_cancelled,
            )
            if forecast_preview_latents is not None:
                assert preview_forecast_output is not None
                assert decoded_forecast_video is not None
                assert decoded_forecast_audio is not None
                AtomicPyAVMuxer(output_root=self.output_root).write(
                    video=decoded_forecast_video,
                    audio=decoded_forecast_audio,
                    sample_rate=32000,
                    fps=request.fps,
                    output_path=preview_forecast_output,
                    cancel_check=raise_if_cancelled,
                )
                del decoded_forecast_video, decoded_forecast_audio
            del decoded_video, decoded_audio
            forecast_preview_latents = None
            preview_latents = None
            preview_published = True
            if request.preview_ready_callback is not None:
                request.preview_ready_callback({
                    "output_path": str(preview_output),
                    "step_index": index,
                    "decode_mode": request.preview_decode_mode,
                    "branch_steps": (
                        0
                        if request.preview_decode_mode == "direct_x0"
                        else request.preview_branch_steps
                    ),
                    "spatial_scale": request.preview_branch_spatial_scale,
                    "audio_branch_use_lora": request.preview_audio_branch_use_lora,
                    "audio_branch_steps": (
                        request.preview_audio_branch_steps
                        if request.preview_audio_branch_use_lora else None
                    ),
                    "width": int(execution_profile["intermediate_preview"]["preview_width"]),
                    "height": int(execution_profile["intermediate_preview"]["preview_height"]),
                    "main_trajectory_mutated": False,
                    "forecast_steps": request.preview_forecast_steps,
                    "forecast_output_path": (
                        None
                        if preview_forecast_output is None
                        else str(preview_forecast_output)
                    ),
                })
            if request.preview_decision_wait is not None:
                decision = request.preview_decision_wait()
                if decision != "continue":
                    raise HotSessionCancelled("preview branch discarded")
            raise_if_cancelled()
            if request.checkpoint_after_step is None:
                dit = self._activate_transformer(execution_plan)

        def debug_step(index, clock, step_video, step_audio):
            progress(
                20 + 58 * (index + 1) / request.steps,
                "denoise",
                f"DiT 去噪 {index + 1}/{request.steps}",
            )
            # Query streaming keeps an individual H3 block inside VRAM, but
            # long requests repeatedly allocate several differently shaped
            # Dense/Sparge workspaces.  CUDA's cache can otherwise retain
            # those slabs across solver steps until HMM silently spills well
            # beyond physical VRAM.  Reclaim only at the natural step
            # boundary and only for the opt-in long-sequence path.  This
            # policy depends on actual execution geometry, never prompt or
            # conditioning semantics.
            long_sequence_active = bool(
                execution_plan is not None
                and execution_plan.long_sequence_query_chunk_tokens is not None
            )
            if execution_plan is not None and (
                execution_plan.long_sequence_query_chunk_tokens is not None
            ):
                torch.cuda.synchronize()
                gib = float(1024**3)
                allocated_before = torch.cuda.memory_allocated()
                reserved_before = torch.cuda.memory_reserved()
                memory_budget = self._device_execution_budget_bytes()
                # Cache compaction is a pressure valve, not a mandatory tax.
                # Keep reusable slabs for 480p/720p throughput and compact only
                # when the retained allocator footprint approaches the current
                # service budget.  The 8 GiB W4A8 route is the exception: its
                # long-video KV builder needs a large contiguous allocation at
                # the beginning of every solver step, so delaying compaction
                # until 82% can cross the hard allocator ceiling inside the
                # next step.  Compact at each non-terminal long-sequence step
                # boundary for that resource tier; larger tiers keep hot slabs.
                w4a8_8gb = (
                    getattr(
                        getattr(self, "runtime_config", None),
                        "resource_profile",
                        None,
                    )
                    == "w4a8_8gb"
                )
                compact_long_sequence = bool(
                    long_sequence_active
                    and index + 1 < request.steps
                    and (
                        w4a8_8gb
                        or reserved_before >= int(memory_budget * 0.82)
                    )
                )
                if compact_long_sequence:
                    torch.cuda.empty_cache()
                long_sequence_step_memory.append(
                    {
                        "step_index": int(index),
                        "allocated_before_gib": allocated_before / gib,
                        "reserved_before_gib": reserved_before / gib,
                        "reserved_after_gib": torch.cuda.memory_reserved() / gib,
                        "cumulative_peak_allocated_gib": (
                            torch.cuda.max_memory_allocated() / gib
                        ),
                        "cumulative_peak_reserved_gib": (
                            torch.cuda.max_memory_reserved() / gib
                        ),
                        "interstep_cache_compacted": compact_long_sequence,
                        "device_budget_gib": memory_budget / gib,
                    }
                )
                execution_profile["long_sequence_step_memory"] = list(
                    long_sequence_step_memory
                )
            checkpoint_now = request.checkpoint_after_step == index + 1
            if checkpoint_now:
                save_formal_checkpoint(index, clock, step_video, step_audio)
            if request.preview_step_index == index and not preview_published:
                publish_preview(index, clock, step_video, step_audio)
            if checkpoint_now:
                raise _HotSessionCheckpointReached
            if self.debug_step_dir is None:
                return
            if last_denoised is None:
                raise RuntimeError("debug callback ran before a model prediction")
            self.debug_step_dir.mkdir(parents=True, exist_ok=True)
            denoised_video, denoised_audio = last_denoised
            torch.save(
                {
                    "x": torch.cat(
                        (step_video.flatten(1), step_audio.flatten(1)), dim=-1
                    ).detach().cpu(),
                    "denoised": torch.cat(
                        (
                            denoised_video.flatten(1),
                            denoised_audio.flatten(1),
                        ),
                        dim=-1,
                    ).detach().cpu(),
                    "sigma": clock.video_sigma,
                    "sigma_next": clock.video_sigma_next,
                },
                self.debug_step_dir / f"native_step_{index:02d}.pt",
            )

        def sample():
            second_sampling_diagnostic = os.environ.get(
                "H3_SECOND_SAMPLING_DIAGNOSTIC_PROFILE", ""
            ).strip()
            if second_sampling_diagnostic:
                if second_sampling_diagnostic not in (
                    "decode_learned_init_v1",
                    "decode_selected_init_v1",
                ):
                    raise ValueError(
                        "unsupported H3 second-sampling diagnostic profile: "
                        f"{second_sampling_diagnostic}"
                    )
                if (
                    request.refinement_latents_path is None
                    or refinement_motion_video is None
                ):
                    raise ValueError(
                        "decode_learned_init_v1 is private to H3 second sampling"
                    )
                execution_profile["second_sampling_diagnostic"] = {
                    "profile": second_sampling_diagnostic,
                    "behavior": "decode_learned_3d_initialization_without_dit",
                }
                # Research-only diagnosis: decode exactly the target-grid
                # latent produced by the learned 3D resizer.  This separates
                # initialization artifacts from those introduced by the
                # low-noise H3 trajectory.  The profile is unreachable unless
                # explicitly selected in the isolated laboratory service.
                return refinement_motion_video.clone(), audio
            if self._uses_turbo_sampler(request):
                sampler = TurboAVSampler(self.turbo_clock_mode)
            elif request.refinement_latents_path is not None:
                sampler = (
                    SASolverAVSampler()
                    if request.refinement_sampler == "sa_solver"
                    else ResMultistepAVSampler()
                )
            else:
                sampler = ResMultistepAVSampler()
            result_video, result_audio = sampler.sample(
                video,
                audio,
                plan,
                predict,
                callback=debug_step,
                transition=(
                    transition_resume_protected_prefix
                    if resume_protected_video_prefix is not None
                    else transition_continuation_selflift
                    if (
                        request.continuation_latents_path is not None
                        and request.multiscale_resize_after_step is not None
                        and request.multiscale_transition_mode
                        == "selflift_learned_x0"
                    )
                    else transition_continuation
                    if request.continuation_latents_path is not None
                    else transition_selflift
                    if (
                        request.multiscale_resize_after_step is not None
                        and request.multiscale_transition_mode
                        == "selflift_learned_x0"
                    )
                    else transition_multiscale
                    if request.multiscale_resize_after_step is not None
                    else transition_refinement_first_anchor
                    if refinement_first_anchor_clean is not None
                    else None
                ),
                initial_previous_video=resume_previous_video,
                initial_previous_audio=resume_previous_audio,
                initial_previous_video_sigma=resume_previous_video_sigma,
                initial_previous_audio_sigma=resume_previous_audio_sigma,
            )
            if (
                self._uses_turbo_sampler(request)
                and self.turbo_clock_mode is TurboClockMode.SHARED_VIDEO
            ):
                result_audio.div_(4.0)
            return result_video, result_audio

        torch.cuda.synchronize()
        denoise_started = time.perf_counter()
        try:
            video, audio = sample()
        except _HotSessionCheckpointReached:
            torch.cuda.synchronize()
            phases["denoise"] = time.perf_counter() - denoise_started
            peak_allocated_gib = torch.cuda.max_memory_allocated() / (1024**3)
            peak_reserved_gib = torch.cuda.max_memory_reserved() / (1024**3)
            if request_online_budget is not None:
                execution_profile["attention_online_guard"] = (
                    request_online_budget.telemetry()
                )
            # The formal checkpoint already owns CPU copies of every tensor
            # required for resume.  Release all request-local CUDA roots
            # *before* emptying the allocator.  Previously these locals stayed
            # alive until the function returned, so ``empty_cache`` could not
            # reclaim roughly 3 GiB of long-video latents/forecast history and
            # the next 8 GiB-tier request could fail despite an idle GPU.
            del context, text_tags, layout, dit
            del condition_video_latents, condition_audio_latents
            del conditioning_routes, conditioning_route_layouts
            del video, audio
            forecast = None
            last_denoised = None
            preview_latents = None
            self.transformer.move_to("cpu", non_blocking=False)
            self._clear_block_executor()
            self._release_device()
            self._release_request_host_scratch()
            return HotSessionCheckpointResult(
                checkpoint_path=(
                    None
                    if request.checkpoint_state_path is None
                    else Path(request.checkpoint_state_path).resolve()
                ),
                preview_path=(
                    preview_output if preview_published else None
                ),
                completed_steps=int(request.checkpoint_after_step or 0),
                total_steps=request.steps,
                total_seconds=time.perf_counter() - started_total,
                phases=phases,
                step_seconds=tuple(step_seconds),
                execution_profile=execution_profile,
                peak_allocated_gib=peak_allocated_gib,
                peak_reserved_gib=peak_reserved_gib,
                preview_latents_path=(
                    Path(request.preview_latents_path).resolve()
                    if request.preview_latents_path is not None
                    and Path(request.preview_latents_path).is_file()
                    else None
                ),
            )
        torch.cuda.synchronize()
        phases["denoise"] = time.perf_counter() - denoise_started
        if boundary_forecast is not None:
            execution_profile["continuation_text_bridge"][
                "auxiliary_forecast_profile"
            ] = boundary_forecast.export()
        if request_online_budget is not None:
            execution_profile["attention_online_guard"] = (
                request_online_budget.telemetry()
            )
        execution_profile["self_speculative_verifier"] = {
            "mode": "whole_dit_draft_verify_rollback",
            "verify_steps": list(self.self_speculative_verify_steps),
            "threshold": (
                self.self_speculative_verify_threshold
                if math.isfinite(self.self_speculative_verify_threshold)
                else None
            ),
            "records": self_speculative_records,
        }
        active_roi_regions = request.refinement_roi_regions
        roi_detection_profile: dict[str, Any] | None = None
        if request.refinement_roi_auto:
            from .roi_atlas import detect_difficult_regions

            if (
                refinement_motion_video is None
                or refinement_previous_video_denoised is None
                or refinement_source_latent_height is None
                or refinement_source_latent_width is None
            ):
                raise RuntimeError(
                    "automatic ROI refinement requires source motion and two "
                    "clean-state predictions"
                )

            def select_difficult_regions():
                return detect_difficult_regions(
                    refinement_motion_video,
                    refinement_previous_video_denoised,
                    video,
                    source_height=refinement_source_latent_height,
                    source_width=refinement_source_latent_width,
                    maximum_regions=request.refinement_roi_max_regions,
                    minimum_side_fraction=(
                        request.refinement_roi_min_side_fraction
                    ),
                    maximum_side_fraction=(
                        request.refinement_roi_max_side_fraction
                    ),
                )

            active_roi_regions, roi_detection_profile = self._timed(
                phases,
                "roi_difficulty_selection",
                select_difficult_regions,
            )
            execution_profile["roi_difficulty_selection"] = (
                roi_detection_profile
            )
        if refinement_motion_video is not None:
            if (
                refinement_previous_video_denoised is not None
                and request.refinement_cross_step_detail_strength > 0.0
            ):
                assert refinement_source_latent_height is not None
                assert refinement_source_latent_width is not None
                video = damp_unconverged_refinement_detail(
                    refinement_motion_video,
                    refinement_previous_video_denoised,
                    video,
                    source_height=refinement_source_latent_height,
                    source_width=refinement_source_latent_width,
                    strength=request.refinement_cross_step_detail_strength,
                )
            if (
                request.refinement_final_low_frequency_gain < 1.0
                or request.refinement_temporal_lowpass
            ):
                assert refinement_source_latent_height is not None
                assert refinement_source_latent_width is not None
                video = blend_terminal_refinement_detail(
                    refinement_motion_video,
                    video,
                    source_height=refinement_source_latent_height,
                    source_width=refinement_source_latent_width,
                    low_frequency_gain=(
                        request.refinement_final_low_frequency_gain
                    ),
                    temporal_lowpass=request.refinement_temporal_lowpass,
                    temporal_outlier_only=(
                        request.refinement_temporal_outlier_only
                    ),
                    temporal_detail_outlier_strength=(
                        request.refinement_temporal_detail_outlier_strength
                    ),
                )
            if refinement_full_canvas_mask is not None:
                video = refinement_motion_video.float() + (
                    refinement_full_canvas_mask
                    * (video.float() - refinement_motion_video.float())
                )
            if refinement_sampler_mask is not None:
                video = refinement_motion_video.float() + (
                    refinement_sampler_mask
                    * (video.float() - refinement_motion_video.float())
                )
            if (
                refinement_first_anchor_clean is not None
                and refinement_first_anchor_weights is not None
            ):
                count = int(refinement_first_anchor_clean.shape[2])
                video[:, :, :count].mul_(
                    1.0 - refinement_first_anchor_weights
                ).add_(
                    refinement_first_anchor_clean
                    * refinement_first_anchor_weights
                )
            del refinement_motion_video
        refinement_motion_video = None
        refinement_full_canvas_mask = None
        refinement_sampler_mask = None
        refinement_sampler_noise = None
        refinement_previous_video_denoised = None
        if request.refinement_roi_steps > 0 and active_roi_regions:
            from .roi_atlas import (
                atlas_plan_dict,
                build_region_atlas,
                merge_region_atlas,
                remap_region_atlas_positions,
            )

            if refinement_source_latent_height is None:
                raise RuntimeError(
                    "ROI atlas refinement requires the source latent geometry"
                )
            global_video = video
            clean_audio = audio
            clean_atlas, atlas_records = build_region_atlas(
                global_video,
                active_roi_regions,
                rows=request.refinement_roi_atlas_rows,
                columns=request.refinement_roi_atlas_columns,
                atlas_height=(
                    None
                    if request.refinement_roi_atlas_height == 0
                    else request.refinement_roi_atlas_height
                ),
                atlas_width=(
                    None
                    if request.refinement_roi_atlas_width == 0
                    else request.refinement_roi_atlas_width
                ),
            )
            atlas_latent_shape = (clean_atlas.shape[-2], clean_atlas.shape[-1])
            roi_sigmas = refinement_sigma_schedule(
                request.refinement_roi_steps,
                request.refinement_roi_denoise,
                request.refinement_video_shift,
            )
            roi_sigma_start = float(roi_sigmas[0])
            roi_generator = torch.Generator(device=clean_atlas.device)
            roi_generator.manual_seed(
                (int(request.seed) ^ 0x524F4941544C4153) & ((1 << 63) - 1)
            )
            roi_video_noise = torch.randn(
                clean_atlas.shape,
                dtype=torch.float32,
                device=clean_atlas.device,
                generator=roi_generator,
            )
            roi_audio_noise = torch.randn(
                clean_audio.shape,
                dtype=torch.float32,
                device=clean_audio.device,
                generator=roi_generator,
            )
            roi_video = (
                roi_sigma_start * roi_video_noise
                + (1.0 - roi_sigma_start) * clean_atlas.float()
            )
            roi_audio = (
                roi_sigma_start * roi_audio_noise
                + (1.0 - roi_sigma_start) * clean_audio.float()
            )
            del roi_video_noise, roi_audio_noise

            # A different visual chart must not inherit layout or forecast
            # history from the full-frame trajectory.  The canvas dimensions,
            # prompt projection and H3 RoPE domain stay unchanged.
            if forecast is not None:
                forecast_profile_override = forecast.export()
            forecast = None
            layout = None
            last_denoised = None
            if request.refinement_roi_position_mode == "source_foveated":
                if keyframe_indices or reference_shapes or reference_audio_frames:
                    raise RuntimeError(
                        "source-foveated ROI positions currently support text-only refinement"
                    )
                layout = build_fl2va_layout(
                    text_length=int(current_context.shape[1]),
                    latent_frames=int(clean_atlas.shape[2]),
                    latent_height=int(clean_atlas.shape[3]),
                    latent_width=int(clean_atlas.shape[4]),
                    audio_frames=int(clean_audio.shape[-1]),
                    output_frame_count=request.frames,
                )
                remap_region_atlas_positions(
                    layout,
                    atlas_records,
                    atlas_height=int(clean_atlas.shape[3]),
                    atlas_width=int(clean_atlas.shape[4]),
                    full_height=int(global_video.shape[3]),
                    full_width=int(global_video.shape[4]),
                )
            roi_plan = SamplingPlan(
                sampler="sa_solver",
                video_sigmas=roi_sigmas,
                audio_sigmas=roi_sigmas,
                actual_step_indices=tuple(
                    range(request.refinement_roi_steps)
                ),
                video_shift=request.refinement_video_shift,
                audio_shift=request.refinement_video_shift,
                seed=(int(request.seed) ^ 0x524F4941) & ((1 << 63) - 1),
            )

            def roi_predict(
                video_value,
                audio_value,
                clock,
                *,
                step_index,
                is_actual_step,
            ):
                # Reuse the calibrated global schedule on a smaller canvas.
                # Early atlas evaluations may use sparse Attention/Query/MLP
                # routes, while the final atlas evaluation inherits the dense
                # convergence point.  The atlas itself already contains only
                # high-priority regions, so no easy full-frame Query survives
                # into this pass.
                routed_index = min(
                    request.steps - 1,
                    max(
                        0,
                        request.steps
                        - request.refinement_roi_steps
                        + step_index,
                    ),
                )

                def run_prediction():
                    return predict(
                        video_value,
                        audio_value,
                        clock,
                        step_index=routed_index,
                        is_actual_step=True,
                    )

                if request.refinement_roi_attention_mode == "dense":
                    with attention_force_dense():
                        return run_prediction()
                return run_prediction()

            def refine_roi_atlas():
                return SASolverAVSampler().sample(
                    roi_video,
                    roi_audio,
                    roi_plan,
                    roi_predict,
                    cancel_check=raise_if_cancelled,
                )

            refined_atlas, discarded_audio = self._timed(
                phases,
                "roi_atlas_refinement",
                refine_roi_atlas,
            )
            del discarded_audio, roi_video, roi_audio
            source_scale = float(refinement_source_latent_height) / float(
                global_video.shape[-2]
            )
            video = merge_region_atlas(
                global_video,
                clean_atlas,
                refined_atlas,
                atlas_records,
                source_scale=source_scale,
                low_frequency_gain=(
                    request.refinement_roi_low_frequency_gain
                ),
                mid_frequency_gain=(
                    request.refinement_roi_mid_frequency_gain
                ),
                coarse_scale=request.refinement_roi_coarse_scale,
                blend=request.refinement_roi_blend,
                temporal_outlier_strength=(
                    request.refinement_roi_temporal_outlier_strength
                ),
                temporal_filter=request.refinement_roi_temporal_filter,
            )
            del refined_atlas, clean_atlas, global_video
            audio = clean_audio
            execution_profile["roi_atlas_refinement"] = {
                **atlas_plan_dict(
                    atlas_records,
                    source_scale=source_scale,
                    steps=request.refinement_roi_steps,
                    denoise=request.refinement_roi_denoise,
                    atlas_shape=atlas_latent_shape,
                ),
                "low_frequency_gain": (
                    request.refinement_roi_low_frequency_gain
                ),
                "mid_frequency_gain": (
                    request.refinement_roi_mid_frequency_gain
                ),
                "coarse_scale": request.refinement_roi_coarse_scale,
                "blend": request.refinement_roi_blend,
                "temporal_outlier_strength": (
                    request.refinement_roi_temporal_outlier_strength
                ),
                "temporal_filter": request.refinement_roi_temporal_filter,
                "position_mode": request.refinement_roi_position_mode,
                "attention_route": request.refinement_roi_attention_mode,
                "preserve_global_audio": True,
                "intermediate_decode": False,
            }
        elif request.refinement_roi_steps > 0:
            execution_profile["roi_atlas_refinement"] = {
                "policy": "magnified_region_atlas_h3_refinement_v1",
                "skipped": True,
                "reason": "automatic_selector_found_no_difficult_region",
                "steps": request.refinement_roi_steps,
            }
        if request.terminal_refinement_initial_width is not None:
            assert terminal_video_noise_cpu is not None
            assert terminal_audio_noise_cpu is not None
            clean_audio = audio
            video = resize_refinement_video_latent_spatial(
                video,
                target_height=video_shape[-2],
                target_width=video_shape[-1],
            )
            motion_video = video
            total_refinement_steps = int(
                request.terminal_refinement_steps
                / request.terminal_refinement_denoise
            )
            total_refinement_steps = max(
                total_refinement_steps, request.terminal_refinement_steps
            )
            full_refinement_sigmas = simple_sigma_schedule(
                total_refinement_steps, 12.0
            )
            refinement_sigmas = full_refinement_sigmas[
                -(request.terminal_refinement_steps + 1) :
            ]
            sigma_start = float(refinement_sigmas[0])
            video = (
                sigma_start * terminal_video_noise_cpu.to("cuda:0")
                + (1.0 - sigma_start) * video.float()
            )
            refinement_audio = (
                sigma_start * terminal_audio_noise_cpu.to("cuda:0")
                + (1.0 - sigma_start) * clean_audio.float()
            )
            del terminal_video_noise_cpu, terminal_audio_noise_cpu
            if forecast is not None:
                forecast_profile_override = forecast.export()
            forecast = None
            layout = None
            refinement_plan = SamplingPlan(
                sampler="res_multistep",
                video_sigmas=refinement_sigmas,
                audio_sigmas=refinement_sigmas,
                actual_step_indices=tuple(
                    range(request.terminal_refinement_steps)
                ),
                video_shift=12.0,
                audio_shift=3.0,
            )

            def terminal_predict(
                video_value,
                audio_value,
                clock,
                *,
                step_index,
                is_actual_step,
            ):
                # Route the correction through the protected final solver
                # positions.  For the current one-step preset this is step 19,
                # which is dense even when the motion stage uses sparse MTCR.
                routed_index = (
                    request.steps
                    - request.terminal_refinement_steps
                    + step_index
                )
                dense_start = (
                    request.terminal_refinement_steps
                    - request.terminal_refinement_dense_tail_steps
                )
                if step_index >= dense_start:
                    with attention_force_dense():
                        return predict(
                            video_value,
                            audio_value,
                            clock,
                            step_index=routed_index,
                            is_actual_step=True,
                        )
                return predict(
                    video_value,
                    audio_value,
                    clock,
                    step_index=routed_index,
                    is_actual_step=True,
                )

            def refine_terminal():
                return ResMultistepAVSampler().sample(
                    video,
                    refinement_audio,
                    refinement_plan,
                    terminal_predict,
                    cancel_check=raise_if_cancelled,
                )

            video, discarded_audio = self._timed(
                phases, "terminal_refinement", refine_terminal
            )
            del discarded_audio
            video = blend_terminal_refinement_detail(
                motion_video,
                video,
                source_height=initial_video_shape[-2],
                source_width=initial_video_shape[-1],
                low_frequency_gain=(
                    request.terminal_refinement_low_frequency_gain
                ),
                temporal_lowpass=(
                    request.terminal_refinement_temporal_lowpass
                ),
                temporal_outlier_only=(
                    request.terminal_refinement_temporal_outlier_only
                ),
            )
            del motion_video
            audio = clean_audio
            execution_profile["terminal_refinement"] = {
                "source_geometry": [
                    request.terminal_refinement_initial_width,
                    request.terminal_refinement_initial_height,
                ],
                "target_geometry": [request.width, request.height],
                "steps": request.terminal_refinement_steps,
                "dense_tail_steps": request.terminal_refinement_dense_tail_steps,
                "denoise": request.terminal_refinement_denoise,
                "low_frequency_gain": (
                    request.terminal_refinement_low_frequency_gain
                ),
                "temporal_lowpass": (
                    request.terminal_refinement_temporal_lowpass
                ),
                "temporal_outlier_only": (
                    request.terminal_refinement_temporal_outlier_only
                ),
                "video_sigmas": list(refinement_sigmas),
                "attention_route": "recovery_sparse_then_dense_tail",
                "preserve_motion_stage_audio": True,
                "intermediate_decode": False,
            }
        if preserved_refinement_audio is not None:
            del audio
            audio = preserved_refinement_audio
        if preserved_global_selflift_audio is not None:
            del audio
            audio = preserved_global_selflift_audio
        terminal_latent_guard = stabilize_terminal_video_latent_(video)
        if continuation_video_prefix is not None:
            assert continuation_audio_prefix is not None
            restore_masked_av_prefix_(
                video,
                audio,
                continuation_video_prefix,
                continuation_audio_prefix,
            )
        execution_profile["terminal_latent_guard"] = terminal_latent_guard
        final_latents_path = (
            request.save_final_latents_path or self.debug_final_latents_path
        )
        if final_latents_path is not None:
            final_latents_path = Path(final_latents_path).resolve()
            final_latents_path.parent.mkdir(parents=True, exist_ok=True)
            latent_document = {
                "video": video.detach().cpu(),
                "audio": audio.detach().cpu(),
                "frames": final_output_frames,
                "fps": request.fps,
                "width": request.width,
                "height": request.height,
                "engine": request_engine,
                "seed": request.seed,
            }
            if (
                not request.latent_only
                and self._last_conditioning_cache_payload is not None
            ):
                latent_document["qwen_conditioning_cache"] = (
                    self._last_conditioning_cache_payload
                )
            torch.save(latent_document, final_latents_path)
        del (
            context,
            text_tags,
            projected_contexts,
            projected_text_tags,
            layout,
            dit,
            condition_video_latents,
            condition_audio_latents,
            conditioning_routes,
            conditioning_route_layouts,
        )
        if request.retain_transformer_after_latent_only:
            phases["dit_retained_for_next_window"] = 0.0
        else:
            self._timed(
                phases,
                "dit_evict",
                lambda: self.transformer.move_to("cpu", non_blocking=False),
            )
        self._clear_block_executor()
        self._release_device()
        raise_if_cancelled()

        if request.latent_only:
            if final_latents_path is None:
                raise ValueError("latent-only execution requires save_final_latents_path")
            self._timed(
                phases,
                "host_scratch_release",
                self._release_request_host_scratch,
            )
            return HotSessionResult(
                output_path=output,
                total_seconds=time.perf_counter() - started_total,
                phases=phases,
                step_seconds=tuple(step_seconds),
                forecast_profile=(
                    forecast_profile_override
                    if forecast_profile_override is not None
                    else forecast.export()
                    if forecast is not None
                    else {
                        "schema_version": 1,
                        "mode": "disabled",
                        "planned_actual_steps": list(actual_steps),
                        "actual_steps": request.steps,
                        "forecast_steps": 0,
                        "records": [],
                    }
                ),
                execution_profile={
                    **execution_profile,
                    "ultimate_upscale_window": {
                        "frames": request.frames,
                        "video_tokens": request.internal_video_tokens,
                        "audio_tokens": request.internal_audio_tokens,
                        "decoded": False,
                    },
                },
            )

        progress(80, "video_decode", "视频解码")
        self._timed(
            phases,
            "video_vae_h2d",
            lambda: self.video_vae.move_to("cuda:0", non_blocking=True),
        )
        if execution_plan is not None and execution_plan.vae_spatial_tile is not None:
            tile_height, tile_width = execution_plan.vae_spatial_tile
            if tile_height != tile_width:
                raise ValueError("the current H3 Video-VAE supports square tiles only")
            video_vae_model = self.video_vae.value
            if not hasattr(video_vae_model, "decoder_tile_size"):
                raise TypeError("the H3 Video-VAE does not expose decoder_tile_size")
            video_vae_model.decoder_tile_size = tile_height
        from .adapters.vae_tiling import configure_vae_tile_batching
        from .adapters.real_vae import select_uint8_postprocess_frame_chunk
        from .adapters.vae_compile import (
            transformer_block_compile,
            transformer_block_compile_ready,
        )

        configure_vae_tile_batching(
            self.video_vae.value,
            1 if execution_plan is None else execution_plan.vae_tile_batch_size,
        )
        uint8_frame_chunk = select_uint8_postprocess_frame_chunk((
            1,
            3,
            final_output_frames,
            request.height,
            request.width,
        ))
        execution_profile["video_uint8_postprocess"] = {
            "mode": (
                "temporal_streaming_exact"
                if uint8_frame_chunk is not None
                else "single_tensor_exact"
            ),
            "frame_chunk": uint8_frame_chunk,
            "routing_inputs": "output_geometry_only",
        }
        compile_vae_requested = bool(
            execution_plan is not None
            and execution_plan.vae_transformer_block_compile
        )
        compile_vae_block = bool(
            compile_vae_requested
            and transformer_block_compile_ready(self.video_vae.value)
        )
        execution_profile["video_vae_transformer_block_compile"] = {
            "requested": compile_vae_requested,
            "enabled": compile_vae_block,
            "fallback": (
                "eager_exact_missing_prebuild_v1"
                if compile_vae_requested and not compile_vae_block
                else None
            ),
        }
        with self._video_vae_block_streaming(
            self.video_vae.value, width=request.width, height=request.height
        ) as block_policy:
            with transformer_block_compile(compile_vae_block):
                decoded_video = self._timed(
                    phases,
                    "video_decode",
                    lambda: self._decode_video_for_plan(
                        self.video_vae.value,
                        video,
                        final_output_frames,
                        execution_plan,
                    ),
                )
                decoded_preview_video = (
                    None
                    if preview_latents is None or preview_published
                    else self._timed(
                        phases,
                        "preview_video_decode",
                        lambda: self._decode_video_for_plan(
                            self.video_vae.value,
                            preview_latents[0].to("cuda:0"),
                            final_output_frames,
                            execution_plan,
                        ),
                    )
                )
        execution_profile["video_vae_block_streaming"] = block_policy
        del video
        self._timed(
            phases,
            "video_vae_evict",
            lambda: self.video_vae.move_to("cpu", non_blocking=False),
        )
        self._release_device()
        self._timed(
            phases,
            "video_host_scratch_release",
            self._release_request_host_scratch,
        )
        raise_if_cancelled()

        progress(94, "audio_decode", "音频解码")
        self._timed(
            phases,
            "audio_vae_h2d",
            lambda: self.audio_vae.move_to("cuda:0", non_blocking=True),
        )
        decoded_audio = self._timed(
            phases,
            "audio_decode",
            lambda: self.decode_audio(self.audio_vae.value, audio),
        )
        decoded_preview_audio = (
            None
            if preview_latents is None or preview_published
            else self._timed(
                phases,
                "preview_audio_decode",
                lambda: self.decode_audio(
                    self.audio_vae.value, preview_latents[1].to("cuda:0")
                ),
            )
        )
        preview_latents = None
        del audio
        self._timed(
            phases,
            "audio_vae_evict",
            lambda: self.audio_vae.move_to("cpu", non_blocking=False),
        )
        self._release_device()
        self._timed(
            phases,
            "audio_host_scratch_release",
            self._release_request_host_scratch,
        )

        progress(98, "mux", "封装音视频")
        def mux():
            return AtomicPyAVMuxer(output_root=self.output_root).write(
                video=decoded_video,
                audio=decoded_audio,
                sample_rate=32000,
                fps=request.fps,
                output_path=output,
                cancel_check=raise_if_cancelled,
            )

        mux_receipt = self._timed(phases, "mux", mux)
        execution_profile["output_mux"] = {
            "encoder": dict(mux_receipt.get("encoder", {})),
            "media": dict(mux_receipt.get("media", {})),
        }
        if preview_output is not None and not preview_published:
            if decoded_preview_video is None or decoded_preview_audio is None:
                raise RuntimeError("requested intermediate preview was not captured")

            def mux_preview():
                return AtomicPyAVMuxer(output_root=self.output_root).write(
                    video=decoded_preview_video,
                    audio=decoded_preview_audio,
                    sample_rate=32000,
                    fps=request.fps,
                    output_path=preview_output,
                    cancel_check=raise_if_cancelled,
                )

            self._timed(phases, "preview_mux", mux_preview)
            del decoded_preview_video, decoded_preview_audio
        del decoded_video, decoded_audio
        self._timed(
            phases, "host_scratch_release", self._release_request_host_scratch
        )
        return HotSessionResult(
            output_path=output,
            total_seconds=time.perf_counter() - started_total,
            phases=phases,
            step_seconds=tuple(step_seconds),
            forecast_profile=(
                forecast_profile_override
                if forecast_profile_override is not None
                else forecast.export()
                if forecast is not None
                else {
                    "schema_version": 1,
                    "mode": "disabled",
                    "planned_actual_steps": list(actual_steps),
                    "actual_steps": request.steps,
                    "forecast_steps": 0,
                    "records": [],
                }
            ),
            execution_profile=execution_profile,
        )

    def persist_conditioning_cache(
        self,
        request: HotSessionRequest,
        cache_path: Path,
    ) -> dict[str, Any]:
        """Encode one exact request condition before long-window DiT work.

        Long-horizon prompts may have a different local timeline per window.
        Encoding all unique conditions first avoids repeatedly alternating the
        large Qwen and DiT residency phases.  The small immutable embeddings
        are then consumed through the same validated checkpoint-cache path as
        H3 second sampling.
        """

        request.validate()
        fingerprint = self._conditioning_fingerprint(request)
        embeds, tags = self._encode_request(request)
        payload = self._last_conditioning_cache_payload
        if not isinstance(payload, dict):
            payload = self._conditioning_payload(
                fingerprint,
                self._host_conditioning_tensor(embeds),
                self._host_conditioning_tensor(tags),
            )
        target = Path(cache_path).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"qwen_conditioning_cache": payload}, target)
        token_count = int(payload["text_token_tags"].shape[0])
        del embeds, tags
        self._release_device()
        return {
            "path": str(target),
            "fingerprint": fingerprint,
            "token_count": token_count,
            "status": self._last_conditioning_cache_status,
        }

    def close(self) -> None:
        self._clear_block_executor()
        for component in (
            self.transformer,
            self.video_vae,
            self.audio_vae,
            self.latent_upscaler,
        ):
            if component is None:
                continue
            component.move_to("cpu", non_blocking=False)
        self._prompt_cache = None
        self._conditioning_cache = None
        self._persisted_conditioning_cache = None
        self._last_conditioning_cache_payload = None
        self._reference_latent_cache = None
        self._media_digest_cache.clear()
        self._release_device(collect_cycles=True)


__all__ = [
    "HotSessionCancelled",
    "HotSessionCheckpointResult",
    "HotSessionRequest",
    "HotSessionResult",
    "NativeT2AVHotSession",
    "selflift_renoise_clean_endpoint",
]
