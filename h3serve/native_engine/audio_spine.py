"""Low-bandwidth global audio trajectory for long H3 video generation.

Full-resolution video remains a bounded causal window chain.  Audio is much
cheaper: H3 uses only two packed rows per 40 Hz latent tick.  The first spine
prototype ran one low-resolution DiT view over the complete timeline.  That
removed joins, but a 30-second target exceeded H3's native temporal training
range and stretched late dialogue toward the end of the clip.

V2 therefore owns one global low-resolution AV *solver state* while exposing
only native-duration overlapping views to H3.  Window predictions are fused at
every sigma and the scheduler advances the global state once.  The proxy video
is discarded and the resulting single audio latent becomes the final audio
stream.  This is inference-only and never feeds back into the authoritative
video trajectory.
"""

from __future__ import annotations

from dataclasses import dataclass

from .global_co_denoise import (
    DEFAULT_WINDOW_FRAMES,
    DEFAULT_WINDOW_STRIDE_FRAMES,
    GlobalAVPlan,
    plan_global_av_windows,
)
from .long_horizon import audio_latent_frames, video_latent_frames


PROXY_MAX_SIDE = 320
PROXY_MIN_SIDE = 160
PROXY_ALIGNMENT = 32


def _aligned_dimension(value: float) -> int:
    aligned = int(round(float(value) / PROXY_ALIGNMENT)) * PROXY_ALIGNMENT
    return max(PROXY_MIN_SIDE, min(PROXY_MAX_SIDE, aligned))


@dataclass(frozen=True, slots=True)
class GlobalAudioSpinePlan:
    frames: int
    width: int
    height: int
    video_latent_frames: int
    spatial_tokens_per_frame: int
    video_tokens: int
    audio_ticks: int
    audio_tokens: int
    temporal_plan: GlobalAVPlan
    maximum_local_frames: int
    maximum_local_packed_media_tokens: int
    summed_local_packed_media_tokens: int

    def telemetry(self) -> dict[str, object]:
        return {
            "method": "low_bandwidth_windowed_global_audio_spine_v2",
            "frames": self.frames,
            "proxy_width": self.width,
            "proxy_height": self.height,
            "video_latent_frames": self.video_latent_frames,
            "spatial_tokens_per_frame": self.spatial_tokens_per_frame,
            "proxy_video_tokens": self.video_tokens,
            "audio_ticks": self.audio_ticks,
            "audio_tokens": self.audio_tokens,
            "packed_media_tokens": self.video_tokens + self.audio_tokens,
            "temporal_window_frames": self.temporal_plan.window_frames,
            "temporal_stride_frames": self.temporal_plan.stride_frames,
            "temporal_window_count": len(self.temporal_plan.windows),
            "maximum_local_frames": self.maximum_local_frames,
            "maximum_local_packed_media_tokens": (
                self.maximum_local_packed_media_tokens
            ),
            "summed_local_packed_media_tokens": (
                self.summed_local_packed_media_tokens
            ),
            "single_global_audio_trajectory": True,
            "single_global_solver_state": True,
            "global_scheduler_updates": True,
            "native_duration_dit_views": True,
            "rotary_time": "window_local_trained_range_v1",
            "internal_audio_seams": 0,
            "proxy_video_discarded": True,
            "feeds_back_into_primary_video": False,
        }


def plan_global_audio_spine(
    *,
    output_width: int,
    output_height: int,
    output_frames: int,
    window_frames: int = DEFAULT_WINDOW_FRAMES,
    stride_frames: int = DEFAULT_WINDOW_STRIDE_FRAMES,
) -> GlobalAudioSpinePlan:
    """Choose a bounded spatial proxy while preserving the output aspect."""

    width = int(output_width)
    height = int(output_height)
    frames = int(output_frames)
    if width <= 0 or height <= 0:
        raise ValueError("audio-spine output geometry must be positive")
    if frames < 5 or (frames - 5) % 17:
        raise ValueError("audio-spine frames must satisfy H3's 5 + 17*k grid")
    scale = PROXY_MAX_SIDE / float(max(width, height))
    proxy_width = _aligned_dimension(width * scale)
    proxy_height = _aligned_dimension(height * scale)
    latent_frames = video_latent_frames(frames)
    spatial = (proxy_width // 32) * (proxy_height // 32)
    audio_ticks = audio_latent_frames(frames)
    temporal_plan = plan_global_av_windows(
        frames,
        window_frames=window_frames,
        stride_frames=stride_frames,
    )
    local_packed_tokens = tuple(
        window.video_tokens * spatial + 2 * window.audio_tokens
        for window in temporal_plan.windows
    )
    return GlobalAudioSpinePlan(
        frames=frames,
        width=proxy_width,
        height=proxy_height,
        video_latent_frames=latent_frames,
        spatial_tokens_per_frame=spatial,
        video_tokens=latent_frames * spatial,
        audio_ticks=audio_ticks,
        audio_tokens=2 * audio_ticks,
        temporal_plan=temporal_plan,
        maximum_local_frames=max(
            window.frames for window in temporal_plan.windows
        ),
        maximum_local_packed_media_tokens=max(local_packed_tokens),
        summed_local_packed_media_tokens=sum(local_packed_tokens),
    )


__all__ = [
    "GlobalAudioSpinePlan",
    "PROXY_ALIGNMENT",
    "PROXY_MAX_SIDE",
    "PROXY_MIN_SIDE",
    "plan_global_audio_spine",
]
