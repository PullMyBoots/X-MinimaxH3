from __future__ import annotations

import math
import json
import secrets
from dataclasses import dataclass
from typing import Any

from .deployment_profiles import (
    LAUNCHER_DEFINITIONS,
    LEGACY_MODEL_LAUNCHERS,
    RESOURCE_BACKENDS,
)


FPS = 24
RESOLUTIONS = {
    "360p": 360,
    "480p": 480,
    "540p": 540,
    "720p": 720,
    "900p": 900,
    "1080p": 1080,
    "2k": 1440,
}
GENERATION_RESOLUTION_MIN = 360
GENERATION_RESOLUTION_MAX = 1080
GENERATION_RESOLUTION_DETENTS = (360, 480, 540, 720, 900, 1080)
PROGRESSIVE_RESOLUTION_MAX = 1440
PROGRESSIVE_RESOLUTION_DETENTS = (
    360, 480, 540, 720, 900, 1080, 1220, 1440,
)
ASPECT_RATIOS = {
    "1:1": (1, 1),
    "4:3": (4, 3),
    "3:4": (3, 4),
    "16:9": (16, 9),
    "9:16": (9, 16),
}
ENGINES = ("original", "lora", "reference", "reference_lora")
SERVICE_FAMILIES = ("first_last", "reference")
MODEL_VARIANTS = ("base", "lora")
WEIGHT_TIERS = ("int8", "w4a8")
VRAM_PROFILES = ("24gb", "16gb", "8gb")
MODEL_LAUNCHERS = tuple(LAUNCHER_DEFINITIONS)
# Persisted jobs and third-party clients may still contain the four launcher
# identifiers used before the resource backends were separated.  They resolve
# to the matching release backend but are never advertised by the new UI.
LAUNCHER_CONFIGS = {
    launcher_id: (
        definition.service_family,
        definition.weight_tier,
        definition.vram_profile,
    )
    for launcher_id, definition in LAUNCHER_DEFINITIONS.items()
}
PREVIEW_MODES = ("off", "auto", "pause")
EXECUTION_MODES = ("complete", "checkpoint")
# Product requests no longer select physical execution implementations.  Keep
# the former spellings only for persisted-job/API compatibility and normalize
# every accepted value onto the unified device-budget optimizer.
MEMORY_MODES = ("auto",)
LEGACY_MEMORY_MODES = ("auto", "performance", "low_vram")
SECOND_SAMPLING_RESOLUTIONS = ("720p", "900p", "1080p", "1220p", "2k")
SECOND_SAMPLING_STEPS = (1, 8)
SECOND_SAMPLING_DENOISE = (0.05, 0.50)
SECOND_SAMPLING_TEMPORAL_WINDOW_FRAMES = (68, 362)
SELFLIFT_SIGMA_SCALE = (0.25, 1.0)
SECOND_SAMPLING_STRENGTHS = {
    # Keep the author workflow's 0.20 operating point as the default.  The
    # upper bound stops at the community's commonly reported 0.30 repair
    # setting: beyond it, identity and motion drift rise quickly.
    "preserve": 0.10,
    "standard": 0.20,
    "enhance": 0.25,
    "strong": 0.30,
}
# Product UI policy: users choose the number of real refinement steps while
# the service selects a conservative start point for that solver budget.  The
# curve is anchored at the two common H3 community recipes (3--4 steps around
# denoise 0.20) and only widens slowly after four steps so that choosing higher
# quality cannot unexpectedly become a strong redraw.
SECOND_SAMPLING_AUTO_DENOISE_BY_STEPS = {
    1: 0.10,
    2: 0.15,
    3: 0.20,
    4: 0.20,
    5: 0.22,
    6: 0.23,
    7: 0.24,
    8: 0.25,
}
CHECKPOINT_PREVIEW_RESOLUTIONS = ("source", "360p", "480p", "720p")
REFERENCE_MEDIA_RESOLUTIONS = ("original", "360p", "480p", "720p")
DEFAULT_REFERENCE_IMAGE_RESOLUTION = "720p"
DEFAULT_REFERENCE_VIDEO_RESOLUTION = "360p"
QUALITY_LEVELS = ("fast", "balanced", "quality", "ultra")
SPARSE_SCOPES = ("middle_only", "guarded", "full")
BASE_SAMPLING_STEPS = (5, 30)
LORA_SAMPLING_STEPS = (4, 10)
ACCELERATION_RANGE = (0.0, 100.0)
MIN_CUSTOM_DIMENSION = 192
MAX_CUSTOM_DIMENSION = 2560
MAX_CUSTOM_SHORT_EDGE = 1440
MAX_CUSTOM_PIXELS = 2560 * 1440
MAX_DURATION_SECONDS = 15
MAX_LONG_HORIZON_DURATION_SECONDS = 60
MAX_HIGH_RESOLUTION_DURATION_SECONDS = 15
# The compact full-context route has completed a real 2K/15s H3 DiT checkpoint
# step at 2560x1440 on the 362-frame grid.  The public spatial-temporal contract
# therefore reaches that experimental boundary; admission remains request- and
# device-specific through the unified VRAM planner rather than being inferred
# from this geometric ceiling alone.
MAX_NATIVE_PIXEL_FRAMES = 2560 * 1440 * 362
# H3's nominal 720p preset is aligned to a 736-pixel short edge.
LONG_DURATION_SHORT_EDGE_MAX = 736
W4A8_MAX_SHORT_EDGE = 736
W4A8_MAX_PIXELS = 1280 * 736
W4A8_MAX_PIXEL_FRAMES = W4A8_MAX_PIXELS * 362
# Backward-compatible conservative hints for older clients. New clients use
# duration.max_by_preset / max_native_pixel_frames from public_options().
RESOLUTION_MAX_DURATION_SECONDS = {
    resolution: (
        MAX_HIGH_RESOLUTION_DURATION_SECONDS
        if resolution == "1080p"
        else MAX_DURATION_SECONDS
    )
    for resolution in RESOLUTIONS
}
UPSCALE_LEVELS = {
    "720p": 720, "900p": 900, "1080p": 1080,
    "1220p": 1220, "2k": 1440,
}
MIN_UPSCALE_DIMENSION = 256
MAX_UPSCALE_DIMENSION = 3840
MAX_UPSCALE_PIXELS = 3840 * 2160


def normalize_resolution_name(value: Any) -> str:
    """Map the public 1440P spelling onto the legacy internal 2k key."""

    resolution = str(value).strip().lower()
    return "2k" if resolution == "1440p" else resolution


def generation_short_edge(value: Any) -> tuple[str, int]:
    """Resolve a continuous first-generation short edge from 360P to 1080P."""

    resolution = str(value).strip().lower()
    if not resolution.endswith("p") or not resolution[:-1].isdigit():
        raise ContractError(
            "generation resolution must be between 360p and 1080p"
        )
    short_edge = int(resolution[:-1])
    if not GENERATION_RESOLUTION_MIN <= short_edge <= GENERATION_RESOLUTION_MAX:
        raise ContractError(
            "generation resolution must be between 360p and 1080p"
        )
    return f"{short_edge}p", short_edge


def progressive_short_edge(value: Any) -> tuple[str, int]:
    """Resolve a SelfLift output canvas while keeping its first pass <=1080P."""

    resolution = normalize_resolution_name(value)
    if resolution == "2k":
        return "2k", PROGRESSIVE_RESOLUTION_MAX
    if not resolution.endswith("p") or not resolution[:-1].isdigit():
        raise ContractError(
            "progressive output resolution must be between 360p and 1440p"
        )
    short_edge = int(resolution[:-1])
    if not GENERATION_RESOLUTION_MIN <= short_edge <= PROGRESSIVE_RESOLUTION_MAX:
        raise ContractError(
            "progressive output resolution must be between 360p and 1440p"
        )
    return f"{short_edge}p", short_edge


def public_resolution_name(value: str) -> str:
    return "1440p" if value == "2k" else value


def second_sampling_short_edge(value: Any) -> tuple[str, int]:
    """Normalize one continuous 720P..1440P second-pass target."""

    resolution = normalize_resolution_name(value)
    if resolution == "2k":
        return "2k", 1440
    if not resolution.endswith("p"):
        raise ContractError(
            "second-sampling resolution must be between 720p and 1440p"
        )
    try:
        short_edge = int(resolution[:-1])
    except ValueError as error:
        raise ContractError(
            "second-sampling resolution must be between 720p and 1440p"
        ) from error
    if not 720 <= short_edge <= 1440:
        raise ContractError(
            "second-sampling resolution must be between 720p and 1440p"
        )
    return f"{short_edge}p", short_edge


ORIGINAL_PRESETS: dict[str, dict[str, Any]] = {
    "fast": {
        "label": "极速",
        "description": "最快，适合预览；复杂运动建议先验片",
        "actual_step_indices": [0, 1, 2, 3, 4, 8, 13, 19],
        "actual_steps": 8,
        "forecast_steps": 12,
        "experimental": True,
    },
    "balanced": {
        "label": "均衡",
        "description": "推荐默认档，兼顾稳定性与等待时间",
        "backend_preset": "balanced",
        "actual_steps": 9,
        "forecast_steps": 11,
    },
    "quality": {
        "label": "高质量",
        "description": "增加完整计算，适合复杂运动和远景",
        "backend_preset": "quality",
        "actual_steps": 12,
        "forecast_steps": 8,
    },
    "ultra": {
        "label": "超高质量",
        "description": "全部调度点执行完整计算，速度最慢",
        "backend_preset": "full",
        "actual_steps": 20,
        "forecast_steps": 0,
    },
}


LORA_PRESETS: dict[str, dict[str, Any]] = {
    "fast": {"label": "极速", "description": "四步快速预览", "steps": 4, "strength": 1.0},
    "balanced": {"label": "均衡", "description": "五步日常生成", "steps": 5, "strength": 1.0},
    "quality": {"label": "高质量", "description": "六步稳定基线", "steps": 6, "strength": 1.0},
    "ultra": {"label": "超高质量", "description": "八步实验档；更多步不保证单调提升", "steps": 8, "strength": 1.0},
}


class ContractError(ValueError):
    pass


def _unified_memory_policy(payload: dict[str, Any]) -> str:
    legacy = str(payload.get("memory_mode", "auto")).strip().lower()
    if legacy not in LEGACY_MEMORY_MODES:
        raise ContractError(
            "memory_mode is retired; omit it to use unified VRAM planning"
        )
    return "auto"


def engine_family(engine: str) -> str:
    if engine in ("original", "lora"):
        return "first_last"
    if engine in ("reference", "reference_lora"):
        return "reference"
    raise ContractError(f"unsupported engine: {engine}")


def engine_variant(engine: str) -> str:
    if engine in ("original", "reference"):
        return "base"
    if engine in ("lora", "reference_lora"):
        return "lora"
    raise ContractError(f"unsupported engine: {engine}")


def resolve_engine(family: str, variant: str) -> str:
    if family not in SERVICE_FAMILIES:
        raise ContractError(f"unsupported service_family: {family}")
    if variant not in MODEL_VARIANTS:
        raise ContractError(f"unsupported model_variant: {variant}")
    return {
        ("first_last", "base"): "original",
        ("first_last", "lora"): "lora",
        ("reference", "base"): "reference",
        ("reference", "lora"): "reference_lora",
    }[(family, variant)]


def launcher_family(launcher: str) -> str:
    try:
        return LAUNCHER_CONFIGS[normalize_launcher(launcher)][0]
    except KeyError as error:
        raise ContractError(f"unsupported model launcher: {launcher}") from error


def launcher_weight_tier(launcher: str) -> str:
    try:
        return LAUNCHER_CONFIGS[normalize_launcher(launcher)][1]
    except KeyError as error:
        raise ContractError(f"unsupported model launcher: {launcher}") from error


def launcher_vram_profile(launcher: str) -> str:
    try:
        return LAUNCHER_CONFIGS[normalize_launcher(launcher)][2]
    except KeyError as error:
        raise ContractError(f"unsupported model launcher: {launcher}") from error


def resolve_launcher(
    family: str,
    weight_tier: str = "int8",
    vram_profile: str | None = None,
) -> str:
    if family not in SERVICE_FAMILIES:
        raise ContractError(f"unsupported service_family: {family}")
    if weight_tier not in WEIGHT_TIERS:
        raise ContractError(f"unsupported weight_tier: {weight_tier}")
    if vram_profile is None:
        vram_profile = "8gb" if weight_tier == "w4a8" else "24gb"
    if vram_profile not in VRAM_PROFILES:
        raise ContractError(f"unsupported vram_profile: {vram_profile}")
    if (weight_tier, vram_profile) not in {
        ("int8", "24gb"), ("int8", "16gb"),
        ("w4a8", "24gb"), ("w4a8", "16gb"), ("w4a8", "8gb"),
    }:
        raise ContractError(
            f"{weight_tier} weights do not support the {vram_profile} backend"
        )
    launchers_by_capability = {
        (
            definition.service_family,
            definition.weight_tier,
            definition.vram_profile,
        ): launcher_id
        for launcher_id, definition in LAUNCHER_DEFINITIONS.items()
    }
    return launchers_by_capability[(family, weight_tier, vram_profile)]


def normalize_launcher(value: str) -> str:
    """Canonicalize launcher keys while retaining old family/engine aliases."""

    value = str(value).strip()
    if value in MODEL_LAUNCHERS:
        return value
    if value in LEGACY_MODEL_LAUNCHERS:
        return LEGACY_MODEL_LAUNCHERS[value]
    if value in ENGINES:
        return resolve_launcher(engine_family(value), "int8")
    if value in SERVICE_FAMILIES:
        return resolve_launcher(value, "int8")
    raise ContractError(f"unsupported model launcher: {value}")


def default_quality(engine: str) -> str:
    """Return the calibrated default for one fixed-engine service."""

    if engine == "original":
        return "balanced"
    if engine == "reference":
        return "balanced"
    if engine in ("lora", "reference_lora"):
        # Six steps are the tested Turbo baseline. Five steps remains an
        # optional quality point, but must not be the implicit product default.
        return "quality"
    raise ContractError(f"unsupported engine: {engine}")


def _nearest_multiple(value: float, multiple: int = 32) -> int:
    return max(multiple, int(math.floor(value / multiple + 0.5)) * multiple)


def resolve_geometry(resolution: str, aspect_ratio: str) -> tuple[int, int]:
    try:
        short_edge = RESOLUTIONS[resolution]
    except KeyError:
        resolution, short_edge = generation_short_edge(resolution)
    try:
        rw, rh = ASPECT_RATIOS[aspect_ratio]
    except KeyError as error:
        raise ContractError(f"unsupported aspect ratio: {aspect_ratio}") from error

    if rw >= rh:
        raw_width = short_edge * rw / rh
        raw_height = short_edge
    else:
        raw_width = short_edge
        raw_height = short_edge * rh / rw
    return _nearest_multiple(raw_width), _nearest_multiple(raw_height)


def resolve_short_edge_geometry(
    short_edge: int,
    aspect_ratio: str,
) -> tuple[int, int]:
    try:
        rw, rh = ASPECT_RATIOS[aspect_ratio]
    except KeyError as error:
        raise ContractError(f"unsupported aspect ratio: {aspect_ratio}") from error
    if rw >= rh:
        raw_width, raw_height = short_edge * rw / rh, short_edge
    else:
        raw_width, raw_height = short_edge, short_edge * rh / rw
    return _nearest_multiple(raw_width), _nearest_multiple(raw_height)


def resolve_frames(duration_seconds: float) -> tuple[int, float]:
    if not math.isfinite(duration_seconds) or not 1.0 <= duration_seconds <= MAX_DURATION_SECONDS:
        raise ContractError("duration_seconds must be between 1 and 15")
    target = duration_seconds * FPS
    grid_index = max(0, round((target - 5) / 17))
    frames = min(362, 5 + 17 * grid_index)
    return frames, frames / FPS


def max_frames_for_geometry(
    width: int,
    height: int,
    max_native_pixel_frames: int = MAX_NATIVE_PIXEL_FRAMES,
) -> int:
    """Largest legal H3 frame count inside the native pixel-frame budget."""

    pixels = int(width) * int(height)
    if pixels <= 0:
        raise ContractError("width and height must be positive")
    raw_limit = min(362, int(max_native_pixel_frames) // pixels)
    if raw_limit < 5:
        return 0
    return 5 + 17 * ((raw_limit - 5) // 17)


def max_duration_for_geometry(
    width: int,
    height: int,
    max_native_pixel_frames: int = MAX_NATIVE_PIXEL_FRAMES,
) -> float:
    """Requested-duration ceiling for one resolved native canvas."""

    frames = max_frames_for_geometry(width, height, max_native_pixel_frames)
    if frames < 5:
        return 0.0
    return min(MAX_DURATION_SECONDS, frames / FPS)


def validate_native_spatiotemporal_budget(
    width: int,
    height: int,
    frames: int,
    max_native_pixel_frames: int = MAX_NATIVE_PIXEL_FRAMES,
) -> None:
    if int(width) * int(height) * int(frames) > int(max_native_pixel_frames):
        maximum = max_duration_for_geometry(width, height, max_native_pixel_frames)
        raise ContractError(
            "native generation exceeds the validated spatial-temporal budget "
            f"(width*height*frames <= {int(max_native_pixel_frames)}); "
            f"{width}x{height} supports at most {maximum:.3f} seconds"
        )


def _integer(payload: dict[str, Any], name: str) -> int:
    try:
        return int(payload[name])
    except (KeyError, TypeError, ValueError) as error:
        raise ContractError(f"{name} must be an integer") from error


def _boolean(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _nearest_even(value: float) -> int:
    return max(2, int(math.floor(value / 2 + 0.5)) * 2)


def resolve_upscale_geometry(
    width: int,
    height: int,
    level: str,
) -> tuple[int, int]:
    """Resolve a named delivery size while preserving the generated ratio."""

    level = normalize_resolution_name(level)
    try:
        short_edge = UPSCALE_LEVELS[level]
    except KeyError as error:
        raise ContractError(f"unsupported upscale resolution: {level}") from error
    if width >= height:
        return _nearest_even(short_edge * width / height), short_edge
    return short_edge, _nearest_even(short_edge * height / width)


@dataclass(frozen=True)
class SecondSamplingSpec:
    """Request-local model-based high-resolution refinement controls.

    This is deliberately not folded into :class:`GenerationSpec`.  The first
    pass is a completed, selectable card; a second pass is a new job whose
    input is either that card's clean AV latent or its completed video. Keeping
    the contracts separate prevents H3's one-to-eight low-noise solver steps
    from being confused with the one-step temporal video-diffusion route.
    """

    resolution: str
    width: int
    height: int
    # ``h3`` is retained for persisted jobs and API clients.  ``temporal``
    # runs the one-step video-diffusion restoration path, which is designed
    # for small faces and repeated background detail across adjacent frames.
    method: str = "h3"
    steps: int = 4
    acceleration: float = 75.0
    denoise: float = 0.20
    strength: str = "auto"
    model_variant: str = "base"
    memory_mode: str = "auto"
    spatial_mode: str = "learned_3d"
    preserve_audio: bool = True
    temporal_window_frames: int | None = None
    temporal_overlap_frames: int | None = None

    @classmethod
    def from_mapping(
        cls,
        payload: dict[str, Any],
        *,
        source: "GenerationSpec",
    ) -> "SecondSamplingSpec":
        method = str(payload.get("method", "h3")).strip().lower()
        method = {
            "flashvsr": "temporal",
            "temporal_diffusion": "temporal",
            "native": "h3",
        }.get(method, method)
        if method not in {"h3", "temporal"}:
            raise ContractError(
                "second-sampling method must be h3 or temporal"
            )

        resolution, short_edge = second_sampling_short_edge(
            payload.get("resolution", "1080p")
        )

        # Named H3 canvases use the same 32-pixel geometry as first-pass
        # generation.  Advanced/custom sources retain their actual pixel ratio
        # instead of trusting a possibly stale UI aspect-ratio label.
        if not source.advanced and source.aspect_ratio in ASPECT_RATIOS:
            width, height = resolve_short_edge_geometry(
                short_edge, source.aspect_ratio
            )
        else:
            if source.width >= source.height:
                height = _nearest_multiple(short_edge)
                width = _nearest_multiple(short_edge * source.width / source.height)
            else:
                width = _nearest_multiple(short_edge)
                height = _nearest_multiple(short_edge * source.height / source.width)
        if width <= source.width or height <= source.height:
            raise ContractError(
                "second-sampling target must be larger than the source canvas"
            )
        validate_native_spatiotemporal_budget(width, height, source.frames)

        try:
            steps = int(payload.get("steps", 4))
        except (TypeError, ValueError) as error:
            raise ContractError("second-sampling steps must be an integer") from error
        if not SECOND_SAMPLING_STEPS[0] <= steps <= SECOND_SAMPLING_STEPS[1]:
            raise ContractError("second-sampling steps must be between 1 and 8")

        try:
            acceleration = float(payload.get("acceleration", 75))
        except (TypeError, ValueError) as error:
            raise ContractError(
                "second-sampling acceleration must be numeric"
            ) from error
        if not math.isfinite(acceleration) or not (
            ACCELERATION_RANGE[0] <= acceleration <= ACCELERATION_RANGE[1]
        ):
            raise ContractError(
                "second-sampling acceleration must be between 0 and 100"
            )

        strength_value = payload.get("strength")
        if strength_value in (None, "") and payload.get("denoise") in (None, ""):
            # The product surface exposes steps only.  Keep the actual redraw
            # strength deterministic and persisted with the resulting job.
            strength = "auto"
            denoise = SECOND_SAMPLING_AUTO_DENOISE_BY_STEPS[steps]
        elif strength_value in (None, ""):
            # Preserve older API clients that supplied a continuous denoise
            # value before the automatic product policy was introduced.
            try:
                legacy_denoise = float(payload["denoise"])
            except (TypeError, ValueError) as error:
                raise ContractError("second-sampling denoise must be numeric") from error
            if not math.isfinite(legacy_denoise) or not (
                SECOND_SAMPLING_DENOISE[0]
                <= legacy_denoise
                <= SECOND_SAMPLING_DENOISE[1]
            ):
                raise ContractError(
                    "second-sampling denoise must be between 0.05 and 0.50"
                )
            strength = min(
                SECOND_SAMPLING_STRENGTHS,
                key=lambda name: abs(
                    SECOND_SAMPLING_STRENGTHS[name] - legacy_denoise
                ),
            )
            denoise = SECOND_SAMPLING_STRENGTHS[strength]
        else:
            strength = str(strength_value).strip().lower()
            if strength not in SECOND_SAMPLING_STRENGTHS:
                raise ContractError(
                    "second-sampling strength must be preserve, standard, enhance or strong"
                )
            denoise = SECOND_SAMPLING_STRENGTHS[strength]

        requested_variant = str(payload.get("model_variant", "base")).strip().lower()
        if requested_variant not in {"base", "lora"}:
            raise ContractError("second-sampling model_variant must be base or lora")
        if method == "temporal" and requested_variant != "base":
            raise ContractError(
                "temporal second sampling does not use H3 LoRA weights"
            )
        if method == "h3" and requested_variant == "lora" and steps < 4:
            raise ContractError(
                "H3 LoRA second sampling requires at least four distilled steps"
            )
        model_variant = requested_variant if method == "h3" else "base"

        raw_window = payload.get("temporal_window_frames")
        if raw_window in (None, "", "auto", 0, "0"):
            temporal_window_frames = None
        else:
            try:
                temporal_window_frames = int(raw_window)
            except (TypeError, ValueError) as error:
                raise ContractError(
                    "second-sampling temporal window must be an integer frame count"
                ) from error
            if not (
                SECOND_SAMPLING_TEMPORAL_WINDOW_FRAMES[0]
                <= temporal_window_frames
                <= SECOND_SAMPLING_TEMPORAL_WINDOW_FRAMES[1]
            ):
                raise ContractError(
                    "second-sampling temporal window must be between 68 and 362 frames"
                )
        raw_overlap = payload.get("temporal_overlap_frames")
        if raw_overlap in (None, "", "auto"):
            temporal_overlap_frames = None
        else:
            try:
                temporal_overlap_frames = int(raw_overlap)
            except (TypeError, ValueError) as error:
                raise ContractError(
                    "second-sampling temporal overlap must be an integer frame count"
                ) from error
            if temporal_overlap_frames < 0:
                raise ContractError(
                    "second-sampling temporal overlap must not be negative"
                )
            if (
                temporal_window_frames is not None
                and temporal_overlap_frames >= temporal_window_frames
            ):
                raise ContractError(
                    "second-sampling temporal overlap must be shorter than its window"
                )

        memory_mode = _unified_memory_policy(payload)
        return cls(
            resolution=resolution,
            width=width,
            height=height,
            method=method,
            steps=steps,
            acceleration=round(acceleration, 1),
            denoise=round(denoise, 3),
            strength=strength,
            model_variant=model_variant,
            memory_mode=memory_mode,
            spatial_mode=(
                "temporal_diffusion_1step"
                if method == "temporal"
                else "learned_3d"
            ),
            temporal_window_frames=temporal_window_frames,
            temporal_overlap_frames=temporal_overlap_frames,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "resolution": self.resolution,
            "width": self.width,
            "height": self.height,
            "method": self.method,
            "steps": self.steps,
            "acceleration": self.acceleration,
            "denoise": self.denoise,
            "strength": self.strength,
            "model_variant": self.model_variant,
            "memory_mode": self.memory_mode,
            "spatial_mode": self.spatial_mode,
            "preserve_audio": self.preserve_audio,
            "temporal_window_frames": self.temporal_window_frames,
            "temporal_overlap_frames": self.temporal_overlap_frames,
        }


# H3's validated custom-canvas contract starts at 192 px and requires 32-px
# alignment. Users choose one square canvas and one complete square cell grid;
# eligibility guarantees at least 1.5x enlargement inside each cell.
VIDEO_REPAIR_CANVAS_RANGE = (192, 1088)
VIDEO_REPAIR_CAPACITIES = (1, 4, 9, 16)
VIDEO_REPAIR_MINIMUM_MAGNIFICATION = 1.5
# Face restoration uses ComfyUI-H3-FaceRefine-Accelerated's dedicated
# LightX2V four-step adapter and BasicScheduler tail.
VIDEO_REPAIR_STEPS = 4


def automatic_face_repair_window_seconds(canvas_size: int) -> float:
    """Choose the internal temporal cap from the square repair canvas.

    H3 canvases are aligned to 32 pixels, so nominal 540P and 720P settings
    are represented by 544 and 736 pixels respectively.
    """

    canvas_size = int(canvas_size)
    if canvas_size <= 576:
        return 15.0
    if canvas_size <= 768:
        return 10.0
    return 6.0


def automatic_face_repair_layout(capacity: int, canvas_size: int) -> tuple[int, int]:
    """Return the exact square grid and square canvas selected by the user."""

    if capacity not in VIDEO_REPAIR_CAPACITIES:
        raise ContractError("video repair capacity must be 1, 4, 9 or 16")
    grid = int(math.isqrt(capacity))
    return grid, int(canvas_size)


@dataclass(frozen=True)
class VideoRepairSpec:
    """Pixel-video based local H3 repair submitted from a completed card.

    The source is the delivered video, so this task remains available after
    latent-cache cleanup. Difficult square crops are enlarged into one square
    atlas and processed with the dedicated LightX2V four-step Turbo adapter.
    """

    # Face repair is the only public product route. The square canvas and its
    # square cell capacity are global defaults. The four-step Turbo solver is
    # fixed; a task only chooses acceleration.
    mode: str = "face"
    max_faces: int = 4
    canvas_size: int = 768
    grid_size: int = 2
    steps: int = 4
    acceleration: float = 50.0
    minimum_window_seconds: float = 4.0
    window_seconds: float = 6.0
    overlap_seconds: float = 0.2
    model_variant: str = "lora"
    preserve_audio: bool = True

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> "VideoRepairSpec":
        # ``max_faces`` remains an API alias for older clients. New callers use
        # capacity, which must describe a complete square Atlas grid.
        legacy_grid = payload.get("grid_size")
        try:
            legacy_capacity = int(legacy_grid) ** 2 if legacy_grid is not None else 4
        except (TypeError, ValueError) as error:
            raise ContractError("video repair grid_size must be an integer") from error
        raw_max_faces = payload.get("capacity", payload.get("max_faces", legacy_capacity))
        try:
            max_faces = int(raw_max_faces)
        except (TypeError, ValueError) as error:
            raise ContractError("video repair capacity must be an integer") from error
        if max_faces not in VIDEO_REPAIR_CAPACITIES:
            raise ContractError("video repair capacity must be 1, 4, 9 or 16")
        try:
            canvas_size = int(payload.get("canvas_size", 768))
        except (TypeError, ValueError) as error:
            raise ContractError("video repair canvas_size must be an integer") from error
        if not VIDEO_REPAIR_CANVAS_RANGE[0] <= canvas_size <= VIDEO_REPAIR_CANVAS_RANGE[1]:
            raise ContractError("video repair canvas_size must be between 192 and 1088")
        if canvas_size % 32:
            raise ContractError("video repair canvas_size must be divisible by 32")
        grid_size, canvas_size = automatic_face_repair_layout(
            max_faces, canvas_size
        )
        try:
            acceleration = float(payload.get("acceleration", 50.0))
        except (TypeError, ValueError) as error:
            raise ContractError("video repair acceleration must be numeric") from error
        if not math.isfinite(acceleration) or not (
            ACCELERATION_RANGE[0] <= acceleration <= ACCELERATION_RANGE[1]
        ):
            raise ContractError("video repair acceleration must be between 0 and 100")
        # Windowing is deliberately internal. Smaller Atlases can keep more
        # temporal context without the cost and memory pressure of a 1080P
        # Atlas. The canvas tier is the single source of truth, including when
        # an older persisted job still contains the former 4--6 second values.
        minimum_window_seconds = 4.0
        window_seconds = automatic_face_repair_window_seconds(canvas_size)
        return cls(
            mode="face",
            max_faces=max_faces,
            canvas_size=canvas_size,
            grid_size=grid_size,
            steps=VIDEO_REPAIR_STEPS,
            acceleration=round(acceleration, 2),
            minimum_window_seconds=round(minimum_window_seconds, 3),
            window_seconds=round(window_seconds, 3),
        )

    @property
    def maximum_regions(self) -> int:
        return self.max_faces

    @property
    def cell_size(self) -> int:
        # Match the Atlas builder's 8-pixel cell alignment exactly so the
        # 1.5x eligibility promise also holds for 3x3 layouts.
        return (self.canvas_size // self.grid_size) // 8 * 8

    @property
    def source_crop_size(self) -> int:
        # A candidate is eligible only when its full 2.5x face crop fits this
        # bound and therefore gains at least 1.5x inside one Atlas cell.
        return max(8, int(math.floor(
            self.cell_size / VIDEO_REPAIR_MINIMUM_MAGNIFICATION
        )))

    def atlas_layout(self, detected_faces: int) -> tuple[int, int]:
        del detected_faces
        return self.grid_size, self.canvas_size

    @property
    def denoise(self) -> float:
        # Published accelerated workflow: BasicScheduler(simple), denoise .55.
        return 0.55

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "capacity": self.max_faces,
            "max_faces": self.max_faces,
            "canvas_size": self.canvas_size,
            "grid_size": self.grid_size,
            "minimum_magnification": VIDEO_REPAIR_MINIMUM_MAGNIFICATION,
            "layout_policy": "fixed_square_cells",
            "window_policy": "automatic_canvas_tier",
            "steps": self.steps,
            "acceleration": self.acceleration,
            "denoise": self.denoise,
            "minimum_window_seconds": self.minimum_window_seconds,
            "window_seconds": self.window_seconds,
            "overlap_seconds": self.overlap_seconds,
            "model_variant": self.model_variant,
            "preserve_audio": self.preserve_audio,
        }


def actual_step_schedule(count: int) -> tuple[int, ...]:
    """Build the conservative H3 20-point schedule used by advanced mode."""

    calibrated = {
        8: (0, 1, 2, 3, 4, 8, 13, 19),
        9: (0, 1, 2, 3, 4, 8, 12, 16, 19),
        12: (0, 1, 2, 3, 4, 6, 8, 11, 14, 17, 18, 19),
        20: tuple(range(20)),
    }
    if count in calibrated:
        return calibrated[count]
    if not 5 <= count <= 20:
        raise ContractError("actual_steps must be between 5 and 20")
    anchors = [0, 1, 2, 3, 4]
    remaining = count - len(anchors)
    if remaining:
        import math

        anchors.extend(
            min(19, math.ceil(4 + index * 15 / remaining))
            for index in range(1, remaining + 1)
        )
    return tuple(anchors)


@dataclass(frozen=True)
class GenerationSpec:
    prompt: str
    engine: str
    quality: str
    resolution: str
    aspect_ratio: str
    requested_duration_seconds: float
    seed: int
    width: int
    height: int
    frames: int
    actual_duration_seconds: float
    # For ordinary requests this is ``None`` and ``frames`` is the complete
    # output.  Long-horizon requests keep ``frames`` as the bounded opening
    # window while this field records the exact assembled H3 timeline.
    output_frames: int | None = None
    # Canonical immutable JSON; opt-in authored window contract, independent of env.
    long_video: str | None = None
    # Runtime provenance. W4A8 is the same H3 topology with a lower-bit base;
    # task-level Base/LoRA selection remains orthogonal to this field.
    weight_tier: str = "int8"
    # Physical resource backend selected before model loading.  Unlike the old
    # inferred device budget this identity is persisted with every job, so a
    # 16GB checkpoint can never silently resume through the 24GB executor.
    vram_profile: str = "24gb"
    advanced: bool = False
    custom_actual_steps: int | None = None
    custom_lora_steps: int | None = None
    attention_keep_ratio: float = 1.0
    sparse_scope: str = "full"
    # New two-control execution contract.  ``None`` preserves persisted jobs
    # and legacy API clients; new UI/API requests set both fields together.
    sampling_steps: int | None = None
    acceleration: float | None = None
    # Optional stage-local control for the formal tail after the green fork.
    # Older clients omit it and retain one acceleration value for the whole
    # trajectory. ``acceleration_transition_step`` is a one-based count of
    # formal steps owned by the first pass.
    second_pass_acceleration: float | None = None
    acceleration_transition_step: int | None = None
    # SelfLift progressive generation spends the early part of one Base or
    # Larry trajectory on a smaller canvas, lifts the predicted clean H3
    # latent, then completes the same trajectory on the requested canvas.
    selflift_enabled: bool = False
    selflift_initial_resolution: str = "540p"
    # Number of completed low-resolution solver steps before the lift.  This
    # is one-based at the public boundary (6 means steps 1..6 are low-res).
    selflift_transition_step: int | None = None
    # Optional bounded temporal views for the high-resolution SelfLift tail.
    # This execution tiling is independent from authored/continuation windows.
    selflift_temporal_window_enabled: bool = False
    selflift_temporal_window_seconds: float = 5.0
    selflift_temporal_overlap_seconds: float = 1.0
    # Request-local scale applied to the formal high-resolution Sigma tail.
    # 1.0 preserves the project's calibrated trajectory exactly; lower values
    # constrain how far full-film second sampling may redraw the accepted film.
    selflift_sigma_scale: float = 1.0
    # Persisted compatibility field. New requests always normalize to ``auto``
    # and the physical graph is selected only from the device VRAM budget.
    memory_mode: str = "auto"
    upscale_enabled: bool = False
    upscale_resolution: str | None = None
    upscale_target_width: int | None = None
    upscale_target_height: int | None = None
    preview_mode: str = "off"
    preview_step_index: int | None = None
    preview_branch_steps: int = 2
    preview_fast_finish: bool = False
    execution_mode: str = "complete"
    checkpoint_step: int | None = None
    checkpoint_retain: bool = True
    checkpoint_preview: bool = False
    checkpoint_preview_steps: int = 4
    checkpoint_preview_resolution: str = "source"
    # Ref2VA-only service-side pre-compression.  ``original`` skips the extra
    # user-selected cap; the model's mandatory alignment/safety normalization
    # still applies in the conditioning adapter.
    reference_image_resolution: str = DEFAULT_REFERENCE_IMAGE_RESOLUTION
    reference_video_resolution: str = DEFAULT_REFERENCE_VIDEO_RESOLUTION

    @classmethod
    def from_mapping(
        cls,
        payload: dict[str, Any],
        *,
        max_native_pixel_frames: int = MAX_NATIVE_PIXEL_FRAMES,
        max_duration_by_preset: dict[str, dict[str, float]] | None = None,
        allow_second_sampling_target: bool = False,
    ) -> "GenerationSpec":
        # ``prompt`` is the final model-facing text for REST/ComfyUI clients.
        # Validate it without rewriting it: callers may deliberately use
        # whitespace and section boundaries in an H3 prompt template.
        long_video = None
        if payload.get("long_video") is not None:
            from .long_video import normalize_long_video, story_duration, compile_window_story
            try:
                long_video = normalize_long_video(payload["long_video"])
                duration = story_duration(long_video)
                supplied = payload.get("duration_seconds", payload.get("requested_duration_seconds"))
                if supplied is not None and abs(float(supplied) - duration) > 1e-6:
                    raise ValueError("duration_seconds must equal the sum of authored segment durations")
                if payload.get("frames") not in (None, "") and payload.get("long_horizon") is not True:
                    raise ValueError("explicit frames cannot be combined with long_video")
                _, initial_preview = compile_window_story(long_video, seed=0)
                payload = dict(payload)
                payload.pop("frames", None)
                payload["duration_seconds"] = duration
                payload["prompt"] = initial_preview["windows"][0]["prompt"]
            except (ValueError, TypeError) as error:
                raise ContractError(str(error)) from error
        prompt = str(payload.get("prompt", ""))
        if not prompt.strip():
            raise ContractError("prompt is required")
        if len(prompt) > 20_000:
            raise ContractError("prompt is too long (maximum 20000 characters)")

        raw_family = payload.get("service_family")
        raw_variant = payload.get("model_variant")
        if raw_family not in (None, "") or raw_variant not in (None, ""):
            family = str(raw_family or "first_last").strip()
            variant = str(raw_variant or "base").strip()
            engine = resolve_engine(family, variant)
            legacy = payload.get("engine")
            if legacy not in (None, "", engine, family):
                raise ContractError("engine disagrees with service_family/model_variant")
        else:
            engine = str(payload.get("engine", "original"))
        quality = str(payload.get("quality", "balanced"))
        requested_selflift = _boolean(payload.get("selflift_enabled", False))
        raw_resolution = str(payload.get("resolution", "480p")).strip().lower()
        if requested_selflift:
            resolution, progressive_target_short_edge = progressive_short_edge(
                raw_resolution
            )
        else:
            resolution = normalize_resolution_name(raw_resolution)
            progressive_target_short_edge = None
        aspect_ratio = str(payload.get("aspect_ratio", "16:9"))
        if engine not in ENGINES:
            raise ContractError(f"unsupported engine: {engine}")
        if quality not in QUALITY_LEVELS:
            raise ContractError(f"unsupported quality level: {quality}")

        raw_launcher = payload.get("runtime_launcher", payload.get("launcher"))
        if raw_launcher not in (None, ""):
            launcher = normalize_launcher(str(raw_launcher))
            if launcher_family(launcher) != engine_family(engine):
                raise ContractError(
                    "runtime_launcher disagrees with service_family"
                )
            weight_tier = launcher_weight_tier(launcher)
            vram_profile = launcher_vram_profile(launcher)
            explicit_tier = payload.get("weight_tier")
            if explicit_tier not in (None, "", weight_tier):
                raise ContractError(
                    "weight_tier disagrees with runtime_launcher"
                )
            explicit_profile = payload.get("vram_profile")
            if explicit_profile not in (None, "", vram_profile):
                raise ContractError(
                    "vram_profile disagrees with runtime_launcher"
                )
        else:
            weight_tier = str(payload.get("weight_tier", "int8")).strip().lower()
            if weight_tier not in WEIGHT_TIERS:
                raise ContractError(
                    "weight_tier must be int8 or w4a8"
                )
            vram_profile = str(
                payload.get(
                    "vram_profile",
                    "8gb" if weight_tier == "w4a8" else "24gb",
                )
            ).strip().lower()
            # Resolve once to validate the quantization/resource pairing.
            launcher = resolve_launcher(
                engine_family(engine), weight_tier, vram_profile
            )

        deployment = LAUNCHER_DEFINITIONS[launcher]
        resource_backend = deployment.backend

        memory_mode = _unified_memory_policy(payload)

        reference_image_resolution = str(
            payload.get(
                "reference_image_resolution",
                DEFAULT_REFERENCE_IMAGE_RESOLUTION,
            )
        ).strip().lower()
        reference_video_resolution = str(
            payload.get(
                "reference_video_resolution",
                DEFAULT_REFERENCE_VIDEO_RESOLUTION,
            )
        ).strip().lower()
        if reference_image_resolution not in REFERENCE_MEDIA_RESOLUTIONS:
            raise ContractError(
                "reference_image_resolution must be original, 360p, 480p or 720p"
            )
        if reference_video_resolution not in REFERENCE_MEDIA_RESOLUTIONS:
            raise ContractError(
                "reference_video_resolution must be original, 360p, 480p or 720p"
            )

        preview_mode = str(payload.get("preview_mode", "off")).strip().lower()
        if preview_mode not in PREVIEW_MODES:
            raise ContractError("preview_mode must be off, auto or pause")
        preview_step_index = None
        if payload.get("preview_step_index") not in (None, "", "auto"):
            preview_step_index = _integer(payload, "preview_step_index")
            if preview_step_index < 0:
                raise ContractError("preview_step_index falls outside the supported schedule")
        try:
            preview_branch_steps = int(payload.get("preview_branch_steps", 2))
        except (TypeError, ValueError) as error:
            raise ContractError("preview_branch_steps must be an integer") from error
        if not 1 <= preview_branch_steps <= 30:
            raise ContractError("preview_branch_steps must be between 1 and 30")
        preview_fast_finish = _boolean(payload.get("preview_fast_finish", False))

        execution_mode = str(
            payload.get("execution_mode", "complete")
        ).strip().lower()
        if execution_mode not in EXECUTION_MODES:
            raise ContractError("execution_mode must be complete or checkpoint")
        checkpoint_step = None
        checkpoint_retain = _boolean(payload.get("checkpoint_retain", True))
        checkpoint_preview = _boolean(payload.get("checkpoint_preview", False))
        try:
            checkpoint_preview_steps = int(
                payload.get("checkpoint_preview_steps", 4)
            )
        except (TypeError, ValueError) as error:
            raise ContractError(
                "checkpoint_preview_steps must be an integer"
            ) from error
        if not 1 <= checkpoint_preview_steps <= 8:
            raise ContractError(
                "checkpoint_preview_steps must be between 1 and 8"
            )
        checkpoint_preview_resolution = str(
            payload.get("checkpoint_preview_resolution", "source")
        ).strip().lower()
        if checkpoint_preview_resolution not in CHECKPOINT_PREVIEW_RESOLUTIONS:
            raise ContractError(
                "checkpoint_preview_resolution must be source, 360p, 480p or 720p"
            )

        try:
            # Public requests use duration_seconds. Persisted records use the
            # explicit requested_duration_seconds name. Accept both so a
            # service restart never silently turns a 15-second job into 5s.
            requested_duration = float(payload.get(
                "duration_seconds", payload.get("requested_duration_seconds", 5)
            ))
        except (TypeError, ValueError) as error:
            raise ContractError("duration_seconds must be numeric") from error
        # Public clients use an explicit mode.  Keep the historical boolean as
        # a backwards-compatible alias for the Web console and persisted jobs.
        raw_mode = payload.get("mode")
        if raw_mode not in (None, ""):
            mode = str(raw_mode).strip().lower()
            if mode not in {"preset", "advanced"}:
                raise ContractError("mode must be preset or advanced")
            advanced = mode == "advanced"
            if "advanced" in payload and _boolean(payload["advanced"]) != advanced:
                raise ContractError("mode and advanced disagree")
        else:
            advanced = _boolean(payload.get("advanced", False))
        if not math.isfinite(requested_duration) or not 1.0 <= requested_duration <= MAX_LONG_HORIZON_DURATION_SECONDS:
            raise ContractError(
                f"duration_seconds must be between 1 and {MAX_LONG_HORIZON_DURATION_SECONDS}"
            )
        long_horizon_output_frames = None
        if advanced:
            width = _integer(payload, "width")
            height = _integer(payload, "height")
            # Exact frame control remains available, but API clients may send
            # duration_seconds and let the service choose the legal 17*n+5
            # frame grid.  This keeps advanced mode useful without requiring
            # callers to understand H3's latent temporal layout.
            explicit_frames = payload.get("frames") not in (None, "")
            if explicit_frames:
                frames = _integer(payload, "frames")
            elif long_video is not None:
                # The authored compiler selects the request-local maximum
                # physical window after geometry validation below.
                frames = 5
            elif requested_duration <= MAX_DURATION_SECONDS:
                frames, _ = resolve_frames(requested_duration)
            if not (
                MIN_CUSTOM_DIMENSION <= width <= MAX_CUSTOM_DIMENSION
                and MIN_CUSTOM_DIMENSION <= height <= MAX_CUSTOM_DIMENSION
            ):
                raise ContractError(
                    f"width and height must be between {MIN_CUSTOM_DIMENSION} "
                    f"and {MAX_CUSTOM_DIMENSION}"
                )
            if width % 32 or height % 32:
                raise ContractError("width and height must be multiples of 32")
            if min(width, height) > MAX_CUSTOM_SHORT_EDGE:
                raise ContractError(
                "custom canvas short edge exceeds the validated native 2K envelope"
                )
            if width * height > MAX_CUSTOM_PIXELS:
                raise ContractError("custom canvas exceeds the validated native 2K pixel envelope")
            if long_video is not None:
                actual_duration = requested_duration
            elif explicit_frames or requested_duration <= MAX_DURATION_SECONDS:
                if frames < 5 or frames > 362 or (frames - 5) % 17:
                    raise ContractError("frames must be 5..362 and satisfy 17*n+5")
                # Advanced mode persists the exact H3 grid as its public
                # duration, preserving historical round-trip semantics.
                requested_duration = frames / FPS
                actual_duration = frames / FPS
            else:
                from .native_engine.long_horizon import plan_long_horizon

                maximum_opening = max_frames_for_geometry(
                    width, height, max_native_pixel_frames
                )
                try:
                    long_plan = plan_long_horizon(
                        requested_duration_seconds=requested_duration,
                        prompt=prompt,
                        seed=seed if "seed" in locals() else 0,
                        maximum_opening_frames=maximum_opening,
                    )
                except ValueError as error:
                    raise ContractError(str(error)) from error
                frames = long_plan.segments[0].window_frames
                actual_duration = long_plan.actual_duration_seconds
                long_horizon_output_frames = long_plan.output_frames
            validate_native_spatiotemporal_budget(
                width, height, frames, max_native_pixel_frames
            )
        else:
            if progressive_target_short_edge is None:
                width, height = resolve_geometry(resolution, aspect_ratio)
            else:
                width, height = resolve_short_edge_geometry(
                    progressive_target_short_edge, aspect_ratio
                )
            if max_duration_by_preset is not None:
                try:
                    if resolution in max_duration_by_preset:
                        limit_resolution = resolution
                    else:
                        if progressive_target_short_edge is not None:
                            requested_short_edge = progressive_target_short_edge
                        else:
                            _, requested_short_edge = generation_short_edge(resolution)
                        # Use the next higher configured stop. This preserves an
                        # operator's conservative duration ceiling for every
                        # continuous value between two editable preset rows.
                        limit_resolution = next(
                            name for name, short_edge in RESOLUTIONS.items()
                            if short_edge >= requested_short_edge
                            and name in max_duration_by_preset
                        )
                    max_duration = float(
                        max_duration_by_preset[limit_resolution][aspect_ratio]
                    )
                except (KeyError, StopIteration, TypeError, ValueError) as error:
                    raise ContractError(
                        f"missing configured limit for {resolution} {aspect_ratio}"
                    ) from error
            else:
                max_duration = max_duration_for_geometry(
                    width, height, max_native_pixel_frames
                )
            if long_video is not None:
                # Do not run the legacy free-text splitter for an authored
                # timeline. The dedicated compiler below owns every split.
                frames = 5
                actual_duration = requested_duration
            elif requested_duration <= MAX_DURATION_SECONDS:
                if requested_duration > max_duration:
                    raise ContractError(
                        f"{resolution} {aspect_ratio} generation supports at most "
                        f"{max_duration:.3f} seconds under the configured server limit"
                    )
                frames, actual_duration = resolve_frames(requested_duration)
            else:
                from .native_engine.long_horizon import plan_long_horizon

                maximum_opening = min(
                    max_frames_for_geometry(width, height, max_native_pixel_frames),
                    resolve_frames(max_duration)[0],
                )
                try:
                    long_plan = plan_long_horizon(
                        requested_duration_seconds=requested_duration,
                        prompt=prompt,
                        seed=0,
                        maximum_opening_frames=maximum_opening,
                    )
                except ValueError as error:
                    raise ContractError(str(error)) from error
                frames = long_plan.segments[0].window_frames
                actual_duration = long_plan.actual_duration_seconds
                long_horizon_output_frames = long_plan.output_frames
            if max_duration_by_preset is None:
                validate_native_spatiotemporal_budget(
                    width, height, frames, max_native_pixel_frames
                )

        if long_video is not None:
            from .long_video import compile_window_story
            try:
                authored_plan, authored_preview = compile_window_story(
                    long_video, seed=0,
                    maximum_frames=min(
                        max_frames_for_geometry(width, height, max_native_pixel_frames),
                        resolve_frames(max_duration)[0] if not advanced else 362,
                    ),
                    service_family=engine_family(engine),
                )
            except ValueError as error:
                raise ContractError(str(error)) from error
            requested_duration = authored_plan.requested_duration_seconds
            actual_duration = authored_plan.actual_duration_seconds
            frames = round(authored_preview["effective_max_window_seconds"] * FPS)
            long_horizon_output_frames = authored_plan.output_frames
            prompt = authored_preview["windows"][0]["prompt"]
            validate_native_spatiotemporal_budget(width, height, frames, max_native_pixel_frames)

        maximum_short_edge = resource_backend.maximum_short_edge
        maximum_pixels = resource_backend.maximum_pixels
        progressive_level_values = tuple(
            UPSCALE_LEVELS[level]
            for level in resource_backend.second_sampling_levels
            if level in UPSCALE_LEVELS
        )
        progressive_target_admitted = bool(
            requested_selflift
            and progressive_target_short_edge is not None
            and progressive_level_values
            and progressive_target_short_edge <= max(progressive_level_values)
        )
        if (
            not allow_second_sampling_target
            and not progressive_target_admitted
            and (
                min(width, height) > maximum_short_edge
                or width * height > maximum_pixels
            )
        ):
            maximum_label = resource_backend.first_generation_levels[-1]
            raise ContractError(
                f"the {vram_profile} backend supports first generation up to "
                f"{maximum_label}"
            )
        if (
            resource_backend.maximum_pixel_frames is not None
            and width * height * frames
            > resource_backend.maximum_pixel_frames
        ):
            raise ContractError(
                "the 8GB W4A8 backend supports up to 720p for 15 seconds"
            )

        sampling_steps = None
        acceleration = None
        second_pass_acceleration = None
        acceleration_transition_step = None
        joint_control_fields = tuple(
            name
            for name in ("sampling_steps", "acceleration")
            if payload.get(name) not in (None, "")
        )
        joint_controls = bool(joint_control_fields)
        stage_control_fields = tuple(
            name
            for name in (
                "second_pass_acceleration",
                "acceleration_transition_step",
            )
            if payload.get(name) not in (None, "")
        )
        if stage_control_fields and not joint_controls:
            raise ContractError(
                "stage acceleration controls require sampling_steps and acceleration"
            )
        if joint_controls and len(joint_control_fields) != 2:
            missing = "acceleration" if joint_control_fields == ("sampling_steps",) else "sampling_steps"
            raise ContractError(
                f"sampling_steps and acceleration must be provided together; missing {missing}"
            )
        if joint_controls:
            mixed = tuple(
                name
                for name in (
                    "actual_steps",
                    "lora_steps",
                    "attention_keep_ratio",
                    "sparse_scope",
                )
                if payload.get(name) not in (None, "")
            )
            if mixed:
                raise ContractError(
                    "sampling_steps/acceleration cannot be mixed with legacy "
                    + ", ".join(mixed)
                )
            sampling_steps = _integer(payload, "sampling_steps")
            lower, upper = (
                BASE_SAMPLING_STEPS
                if engine_variant(engine) == "base"
                else LORA_SAMPLING_STEPS
            )
            if not lower <= sampling_steps <= upper:
                raise ContractError(
                    f"sampling_steps must be between {lower} and {upper} for this model"
                )
            try:
                acceleration = float(payload.get("acceleration"))
            except (TypeError, ValueError) as error:
                raise ContractError("acceleration must be numeric") from error
            if not math.isfinite(acceleration) or not (
                ACCELERATION_RANGE[0] <= acceleration <= ACCELERATION_RANGE[1]
            ):
                raise ContractError("acceleration must be between 0 and 100")
            acceleration = round(acceleration, 1)
            raw_second_acceleration = payload.get(
                "second_pass_acceleration", acceleration
            )
            try:
                second_pass_acceleration = float(raw_second_acceleration)
            except (TypeError, ValueError) as error:
                raise ContractError(
                    "second_pass_acceleration must be numeric"
                ) from error
            if not math.isfinite(second_pass_acceleration) or not (
                ACCELERATION_RANGE[0]
                <= second_pass_acceleration
                <= ACCELERATION_RANGE[1]
            ):
                raise ContractError(
                    "second_pass_acceleration must be between 0 and 100"
                )
            second_pass_acceleration = round(second_pass_acceleration, 1)

        total_solver_steps = (
            int(sampling_steps)
            if sampling_steps is not None
            else (
                20
                if engine_variant(engine) == "base"
                else int(LORA_PRESETS[quality]["steps"])
            )
        )
        if joint_controls:
            raw_acceleration_transition = payload.get(
                "acceleration_transition_step", total_solver_steps
            )
            try:
                acceleration_transition_step = int(
                    raw_acceleration_transition
                )
            except (TypeError, ValueError) as error:
                raise ContractError(
                    "acceleration_transition_step must be an integer"
                ) from error
            if not 1 <= acceleration_transition_step <= total_solver_steps:
                raise ContractError(
                    "acceleration_transition_step must lie inside the formal trajectory"
                )
        if (
            preview_step_index is not None
            and preview_step_index >= total_solver_steps
        ):
            raise ContractError(
                "preview_step_index falls outside the supported schedule"
            )
        selflift_enabled = requested_selflift
        selflift_initial_resolution = str(
            payload.get("selflift_initial_resolution", "540p")
        ).strip().lower()
        selflift_transition_step = None
        selflift_temporal_window_enabled = _boolean(
            payload.get("selflift_temporal_window_enabled", False)
        )
        try:
            selflift_temporal_window_seconds = float(
                payload.get("selflift_temporal_window_seconds", 5.0)
            )
        except (TypeError, ValueError) as error:
            raise ContractError(
                "SelfLift temporal window seconds must be numeric"
            ) from error
        if not math.isfinite(selflift_temporal_window_seconds) or not (
            3.0 <= selflift_temporal_window_seconds <= 15.0
        ):
            raise ContractError(
                "SelfLift temporal window seconds must be between 3 and 15"
            )
        selflift_temporal_window_seconds = round(
            selflift_temporal_window_seconds, 1
        )
        try:
            selflift_temporal_overlap_seconds = float(
                payload.get("selflift_temporal_overlap_seconds", 1.0)
            )
        except (TypeError, ValueError) as error:
            raise ContractError(
                "SelfLift temporal overlap seconds must be numeric"
            ) from error
        if not math.isfinite(selflift_temporal_overlap_seconds) or not (
            0.0 <= selflift_temporal_overlap_seconds <= 4.0
        ):
            raise ContractError(
                "SelfLift temporal overlap seconds must be between 0 and 4"
            )
        selflift_temporal_overlap_seconds = round(
            selflift_temporal_overlap_seconds, 1
        )
        try:
            selflift_sigma_scale = float(
                payload.get("selflift_sigma_scale", 1.0)
            )
        except (TypeError, ValueError) as error:
            raise ContractError("SelfLift sigma scale must be numeric") from error
        if not math.isfinite(selflift_sigma_scale) or not (
            SELFLIFT_SIGMA_SCALE[0]
            <= selflift_sigma_scale
            <= SELFLIFT_SIGMA_SCALE[1]
        ):
            raise ContractError(
                "SelfLift sigma scale must be between 0.25 and 1"
            )
        selflift_sigma_scale = round(selflift_sigma_scale, 2)
        if selflift_enabled:
            if not joint_controls:
                raise ContractError("SelfLift requires sampling_steps and acceleration")
            if long_video is not None:
                raise ContractError("SelfLift is not available for transparent long-horizon generation")
            _, initial_short_edge = generation_short_edge(
                selflift_initial_resolution
            )
            initial_width, initial_height = resolve_short_edge_geometry(
                initial_short_edge, aspect_ratio
            )
            if initial_width > width or initial_height > height:
                raise ContractError(
                    "SelfLift initial resolution cannot exceed the output resolution"
                )
            raw_transition = payload.get("selflift_transition_step")
            selflift_transition_step = (
                max(1, total_solver_steps - 2)
                if raw_transition in (None, "", "auto")
                else _integer(payload, "selflift_transition_step")
            )
            if not 1 <= selflift_transition_step < total_solver_steps:
                raise ContractError(
                    "SelfLift transition step must leave at least one high-resolution step"
                )
            if payload.get("acceleration_transition_step") in (None, ""):
                acceleration_transition_step = selflift_transition_step
            elif acceleration_transition_step != selflift_transition_step:
                raise ContractError(
                    "SelfLift and acceleration must use the same transition step"
                )
        if long_horizon_output_frames is not None:
            if execution_mode != "complete":
                raise ContractError(
                    "checkpoint execution is not available for transparent long-horizon generation"
                )
            if preview_mode != "off":
                raise ContractError(
                    "intermediate preview is not available for transparent long-horizon generation"
                )
        if (
            not joint_controls
            and advanced
            and engine_variant(engine) == "lora"
            and payload.get("lora_steps") not in (None, "")
        ):
            total_solver_steps = _integer(payload, "lora_steps")
        if execution_mode == "checkpoint":
            checkpoint_step = _integer(payload, "checkpoint_step")
            if not 1 <= checkpoint_step < total_solver_steps:
                raise ContractError(
                    "checkpoint_step must stop after at least one step and before the final step"
                )
            if not checkpoint_retain and not checkpoint_preview:
                raise ContractError(
                    "checkpoint tasks must retain state, generate a preview, or both"
                )
            if selflift_enabled:
                if checkpoint_step != selflift_transition_step:
                    raise ContractError(
                        "a SelfLift checkpoint must stop at the resolution transition"
                    )
            if checkpoint_preview_resolution != "source":
                # Product resolution labels are nominal; every public canvas
                # is aligned to the nearest 32 pixels (360p becomes 352).
                # Compare aligned sizes so a 360p task can request its fixed
                # 360p checkpoint preview instead of being rejected as an
                # accidental upscale.
                preview_short_edge = _nearest_multiple(
                    RESOLUTIONS[checkpoint_preview_resolution]
                )
                if preview_short_edge > min(width, height):
                    raise ContractError(
                        "checkpoint preview resolution cannot exceed the generated canvas"
                    )

        custom_actual_steps = None
        custom_lora_steps = None
        attention_keep_ratio = 1.0
        sparse_scope = "full"
        if not joint_controls and advanced and engine in ("original", "reference"):
            custom_actual_steps = _integer(payload, "actual_steps")
            actual_step_schedule(custom_actual_steps)
        elif not joint_controls and advanced and engine in ("lora", "reference_lora"):
            custom_lora_steps = _integer(payload, "lora_steps")
            if not 4 <= custom_lora_steps <= 8:
                raise ContractError("lora_steps must be between 4 and 8")

        if advanced and not joint_controls:
            try:
                attention_keep_ratio = float(
                    payload.get("attention_keep_ratio", 1.0)
                )
            except (TypeError, ValueError) as error:
                raise ContractError(
                    "attention_keep_ratio must be numeric"
                ) from error
            if not math.isfinite(attention_keep_ratio) or not (
                0.50 <= attention_keep_ratio <= 1.00
            ):
                raise ContractError(
                    "attention_keep_ratio must be between 0.50 and 1.00"
                )
            # Keep the public value stable and avoid insignificant JSON/HTML
            # floating-point tails from becoming distinct execution profiles.
            attention_keep_ratio = round(attention_keep_ratio, 2)
            sparse_scope = str(
                payload.get("sparse_scope", "full")
            ).strip()
            if sparse_scope not in SPARSE_SCOPES:
                raise ContractError(
                    "sparse_scope must be middle_only, guarded or full"
                )

        upscale_enabled = _boolean(payload.get("upscale_enabled", False))
        upscale_resolution = None
        upscale_target_width = None
        upscale_target_height = None
        if upscale_enabled:
            upscale_mode = str(payload.get("upscale_mode", "basic")).strip()
            if upscale_mode == "basic":
                upscale_resolution = str(
                    payload.get("upscale_resolution", "1080p")
                ).strip()
                upscale_target_width, upscale_target_height = resolve_upscale_geometry(
                    width, height, upscale_resolution
                )
            elif upscale_mode == "advanced":
                upscale_target_width = _integer(payload, "upscale_target_width")
                upscale_target_height = _integer(payload, "upscale_target_height")
            else:
                raise ContractError("upscale_mode must be basic or advanced")

            assert upscale_target_width is not None
            assert upscale_target_height is not None
            if not (
                MIN_UPSCALE_DIMENSION <= upscale_target_width <= MAX_UPSCALE_DIMENSION
                and MIN_UPSCALE_DIMENSION <= upscale_target_height <= MAX_UPSCALE_DIMENSION
            ):
                raise ContractError(
                    "upscale target dimensions must be between 256 and 3840"
                )
            if upscale_target_width * upscale_target_height > MAX_UPSCALE_PIXELS:
                raise ContractError("upscale target exceeds the 4K pixel envelope")
            if upscale_target_width % 2 or upscale_target_height % 2:
                raise ContractError("upscale target dimensions must be even")
            if upscale_target_width < width or upscale_target_height < height:
                raise ContractError("upscale target cannot be smaller than the generated video")
            if upscale_target_width == width and upscale_target_height == height:
                raise ContractError("upscale target must be larger than the generated video")
            source_ratio = width / height
            target_ratio = upscale_target_width / upscale_target_height
            if abs(target_ratio / source_ratio - 1.0) > 0.01:
                raise ContractError(
                    "upscale target must preserve the generated video aspect ratio"
                )

        raw_seed = payload.get("seed")
        if raw_seed in (None, "", "random"):
            seed = secrets.randbits(63)
        else:
            try:
                seed = int(raw_seed)
            except (TypeError, ValueError) as error:
                raise ContractError("seed must be an integer") from error
            if not 0 <= seed < 2**64:
                raise ContractError("seed must be in [0, 2^64)")

        return cls(
            prompt=prompt,
            engine=engine,
            quality=quality,
            resolution=resolution,
            aspect_ratio=aspect_ratio,
            requested_duration_seconds=requested_duration,
            seed=seed,
            width=width,
            height=height,
            frames=frames,
            actual_duration_seconds=actual_duration,
            output_frames=long_horizon_output_frames,
            long_video=long_video,
            weight_tier=weight_tier,
            vram_profile=vram_profile,
            advanced=advanced,
            custom_actual_steps=custom_actual_steps,
            custom_lora_steps=custom_lora_steps,
            attention_keep_ratio=attention_keep_ratio,
            sparse_scope=sparse_scope,
            sampling_steps=sampling_steps,
            acceleration=acceleration,
            second_pass_acceleration=second_pass_acceleration,
            acceleration_transition_step=acceleration_transition_step,
            selflift_enabled=selflift_enabled,
            selflift_initial_resolution=selflift_initial_resolution,
            selflift_transition_step=selflift_transition_step,
            selflift_temporal_window_enabled=(
                selflift_temporal_window_enabled
            ),
            selflift_temporal_window_seconds=(
                selflift_temporal_window_seconds
            ),
            selflift_temporal_overlap_seconds=(
                selflift_temporal_overlap_seconds
            ),
            selflift_sigma_scale=selflift_sigma_scale,
            memory_mode=memory_mode,
            upscale_enabled=upscale_enabled,
            upscale_resolution=upscale_resolution,
            upscale_target_width=upscale_target_width,
            upscale_target_height=upscale_target_height,
            preview_mode=preview_mode,
            preview_step_index=preview_step_index,
            preview_branch_steps=preview_branch_steps,
            preview_fast_finish=preview_fast_finish,
            execution_mode=execution_mode,
            checkpoint_step=checkpoint_step,
            checkpoint_retain=checkpoint_retain,
            checkpoint_preview=checkpoint_preview,
            checkpoint_preview_steps=checkpoint_preview_steps,
            checkpoint_preview_resolution=checkpoint_preview_resolution,
            reference_image_resolution=reference_image_resolution,
            reference_video_resolution=reference_video_resolution,
        )

    @property
    def service_family(self) -> str:
        return engine_family(self.engine)

    @property
    def model_variant(self) -> str:
        return engine_variant(self.engine)

    @property
    def runtime_launcher(self) -> str:
        return resolve_launcher(
            self.service_family, self.weight_tier, self.vram_profile
        )

    @property
    def joint_acceleration_enabled(self) -> bool:
        return self.sampling_steps is not None and self.acceleration is not None

    @property
    def preset(self) -> dict[str, Any]:
        source = ORIGINAL_PRESETS if self.engine in ("original", "reference") else LORA_PRESETS
        result = dict(source[self.quality])
        if self.custom_actual_steps is not None:
            result.update({
                "actual_step_indices": list(actual_step_schedule(self.custom_actual_steps)),
                "actual_steps": self.custom_actual_steps,
                "forecast_steps": 20 - self.custom_actual_steps,
                "advanced": True,
            })
        if self.custom_lora_steps is not None:
            result.update({"steps": self.custom_lora_steps, "advanced": True})
        if self.joint_acceleration_enabled:
            for legacy_key in (
                "actual_step_indices", "actual_steps", "forecast_steps",
                "backend_preset",
            ):
                result.pop(legacy_key, None)
            result.update({
                "steps": self.sampling_steps,
                "sampling_steps": self.sampling_steps,
                "acceleration": self.acceleration,
                "second_pass_acceleration": self.second_pass_acceleration,
                "acceleration_transition_step": self.acceleration_transition_step,
                "joint_acceleration": True,
                "advanced": True,
            })
        if self.advanced:
            if not self.joint_acceleration_enabled:
                result.update({
                    "attention_keep_ratio": self.attention_keep_ratio,
                    "sparse_scope": self.sparse_scope,
                })
        return result

    def to_dict(self, *, include_execution: bool = False) -> dict[str, Any]:
        result = {
            "prompt": self.prompt,
            "engine": self.engine,
            "service_family": self.service_family,
            "model_variant": self.model_variant,
            "runtime_launcher": self.runtime_launcher,
            "weight_tier": self.weight_tier,
            "vram_profile": self.vram_profile,
            "quality": self.quality,
            "resolution": self.resolution,
            "aspect_ratio": self.aspect_ratio,
            "requested_duration_seconds": self.requested_duration_seconds,
            "actual_duration_seconds": self.actual_duration_seconds,
            "width": self.width,
            "height": self.height,
            "frames": self.frames,
            "output_frames": self.output_frames or self.frames,
            "fps": FPS,
            "seed": self.seed,
            "experimental_duration": self.requested_duration_seconds < 5,
            "long_horizon": self.output_frames is not None,
            **({"long_video": json.loads(self.long_video)} if self.long_video is not None else {}),
            "advanced": self.advanced,
            "mode": "advanced" if self.advanced else "preset",
            "memory_mode": self.memory_mode,
            "selflift_enabled": self.selflift_enabled,
            "selflift_initial_resolution": self.selflift_initial_resolution,
            "selflift_transition_step": self.selflift_transition_step,
            "selflift_temporal_window_enabled": (
                self.selflift_temporal_window_enabled
            ),
            "selflift_temporal_window_seconds": (
                self.selflift_temporal_window_seconds
            ),
            "selflift_temporal_overlap_seconds": (
                self.selflift_temporal_overlap_seconds
            ),
            "selflift_sigma_scale": self.selflift_sigma_scale,
            "preview_mode": self.preview_mode,
            "preview_step_index": self.preview_step_index,
            "preview_branch_steps": self.preview_branch_steps,
            "preview_fast_finish": self.preview_fast_finish,
            "execution_mode": self.execution_mode,
            "checkpoint_step": self.checkpoint_step,
            "checkpoint_retain": self.checkpoint_retain,
            "checkpoint_preview": self.checkpoint_preview,
            "checkpoint_preview_steps": self.checkpoint_preview_steps,
            "checkpoint_preview_resolution": self.checkpoint_preview_resolution,
            "reference_image_resolution": self.reference_image_resolution,
            "reference_video_resolution": self.reference_video_resolution,
        }
        if self.custom_actual_steps is not None:
            result["actual_steps"] = self.custom_actual_steps
            result["forecast_steps"] = 20 - self.custom_actual_steps
        if self.custom_lora_steps is not None:
            result["lora_steps"] = self.custom_lora_steps
        if self.joint_acceleration_enabled:
            result["sampling_steps"] = self.sampling_steps
            result["acceleration"] = self.acceleration
            result["second_pass_acceleration"] = self.second_pass_acceleration
            result["acceleration_transition_step"] = (
                self.acceleration_transition_step
            )
        elif self.advanced:
            result["attention_keep_ratio"] = self.attention_keep_ratio
            result["sparse_scope"] = self.sparse_scope
        result["upscale_enabled"] = self.upscale_enabled
        if self.upscale_enabled:
            result["upscale_mode"] = (
                "basic" if self.upscale_resolution is not None else "advanced"
            )
            result["upscale_resolution"] = self.upscale_resolution
            result["upscale_target_width"] = self.upscale_target_width
            result["upscale_target_height"] = self.upscale_target_height
        if include_execution:
            result["execution"] = self.preset
        return result


def _public_presets(source: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Expose product choices without leaking launcher schedules or revisions."""
    return {
        name: {
            "label": preset["label"],
            "description": preset["description"],
            **({"experimental": True} if preset.get("experimental") else {}),
        }
        for name, preset in source.items()
    }


def public_options(
    fixed_engine: str | None = None,
    *,
    max_native_pixel_frames: int = MAX_NATIVE_PIXEL_FRAMES,
    max_duration_by_preset: dict[str, dict[str, float]] | None = None,
) -> dict[str, Any]:
    fixed_launcher = (
        None if fixed_engine is None else normalize_launcher(str(fixed_engine))
    )
    if fixed_launcher is not None:
        fixed_engine = launcher_family(fixed_launcher)
    geometry = {
        resolution: {
            ratio: {"width": resolve_geometry(resolution, ratio)[0],
                    "height": resolve_geometry(resolution, ratio)[1]}
            for ratio in ASPECT_RATIOS
        }
        for resolution in RESOLUTIONS
    }
    if max_duration_by_preset is None:
        max_duration_by_preset = {
            resolution: {
                ratio: max_duration_for_geometry(
                    geometry[resolution][ratio]["width"],
                    geometry[resolution][ratio]["height"],
                    max_native_pixel_frames,
                )
                for ratio in ASPECT_RATIOS
            }
            for resolution in RESOLUTIONS
        }
    engines = {
        "original": {
            "label": "H3 Native 高保真",
            "short_label": "高保真",
            "presets": _public_presets(ORIGINAL_PRESETS),
        },
        "lora": {
            "label": "H3 Native Turbo",
            "short_label": "Turbo",
            "presets": _public_presets(LORA_PRESETS),
        },
        "reference": {
            "label": "H3 Native 多参考",
            "short_label": "Ref2VA",
            "presets": _public_presets(ORIGINAL_PRESETS),
        },
        "reference_lora": {
            "label": "H3 Native 多参考 Turbo",
            "short_label": "Ref2VA Turbo",
            "presets": _public_presets(LORA_PRESETS),
        },
    }
    selected_engine = fixed_engine or "first_last"
    selected_variant = "base"
    families = {
        "first_last": {
            "label": "FL2VA · 文本/首尾帧生成",
            "short_label": "FL2VA",
            "description": "支持文生视频、首帧、尾帧和首尾帧约束",
            "variants": ["base", "lora"],
        },
        "reference": {
            "label": "Ref2VA · 多模态参考生成",
            "short_label": "Ref2VA",
            "description": "支持图片、视频和音频参考",
            "variants": ["base", "lora"],
        },
    }
    launchers = {
        launcher_id: {
            "label": definition.label,
            "short_label": definition.short_label,
            "description": definition.description,
            "service_family": definition.service_family,
            "weight_tier": definition.weight_tier,
            "vram_profile": definition.vram_profile,
            "variants": ["base", "lora"],
            "second_sampling": bool(
                definition.backend.second_sampling_levels
            ),
        }
        for launcher_id, definition in LAUNCHER_DEFINITIONS.items()
    }
    selected_launcher = fixed_launcher or "fl2va_int8_24gb"
    return {
        "deployment_mode": "fixed_engine" if fixed_engine else "multi_engine",
        "current_engine": selected_engine,
        "current_launcher": selected_launcher,
        "active_weight_tier": launcher_weight_tier(selected_launcher),
        "active_vram_profile": launcher_vram_profile(selected_launcher),
        "current_engine_options": families[selected_engine],
        "engines": {
            key: value for key, value in engines.items()
            if fixed_engine is None or engine_family(key) == fixed_engine
        },
        "service_families": families,
        "model_launchers": launchers,
        "model_variants": {
            "base": {"label": "原始权重", "presets": _public_presets(ORIGINAL_PRESETS)},
            "lora": {"label": "LoRA 极速", "presets": _public_presets(LORA_PRESETS)},
        },
        "resolutions": list(RESOLUTIONS),
        "resolution": {
            "min": GENERATION_RESOLUTION_MIN,
            "max": GENERATION_RESOLUTION_MAX,
            "step": 1,
            "detents": list(GENERATION_RESOLUTION_DETENTS),
        },
        "progressive_resolution": {
            "min": GENERATION_RESOLUTION_MIN,
            "max": PROGRESSIVE_RESOLUTION_MAX,
            "first_pass_max": GENERATION_RESOLUTION_MAX,
            "step": 1,
            "detents": list(PROGRESSIVE_RESOLUTION_DETENTS),
        },
        "aspect_ratios": list(ASPECT_RATIOS),
        "geometry": geometry,
        "duration": {
            "min": 1,
            "max": MAX_LONG_HORIZON_DURATION_SECONDS,
            "native_window_max": MAX_DURATION_SECONDS,
            "long_horizon_max": MAX_LONG_HORIZON_DURATION_SECONDS,
            "default": 5,
            "fps": FPS,
            "max_by_resolution": {
                resolution: max_duration_by_preset[resolution]["16:9"]
                for resolution in RESOLUTIONS
            },
            "max_by_preset": max_duration_by_preset,
            "max_native_pixel_frames": int(max_native_pixel_frames),
            "high_resolution_max": max_duration_by_preset["1080p"]["16:9"],
            "long_duration_short_edge_max": LONG_DURATION_SHORT_EDGE_MAX,
        },
        "defaults": {
            "service_family": selected_engine,
            "runtime_launcher": selected_launcher,
            "weight_tier": launcher_weight_tier(selected_launcher),
            "vram_profile": launcher_vram_profile(selected_launcher),
            "model_variant": selected_variant,
            "engine": resolve_engine(selected_engine, selected_variant),
            "quality": default_quality(resolve_engine(selected_engine, selected_variant)),
            "resolution": "480p", "aspect_ratio": "16:9", "duration_seconds": 5,
            "reference_image_resolution": DEFAULT_REFERENCE_IMAGE_RESOLUTION,
            "reference_video_resolution": DEFAULT_REFERENCE_VIDEO_RESOLUTION,
        },
        "reference_media_processing": {
            "levels": list(REFERENCE_MEDIA_RESOLUTIONS),
            "image_default": DEFAULT_REFERENCE_IMAGE_RESOLUTION,
            "video_default": DEFAULT_REFERENCE_VIDEO_RESOLUTION,
            "preserve_aspect_ratio": True,
            "preserve_composition": True,
            "preserve_duration": True,
            "crop": False,
            "stretch": False,
            "pad_user_media": False,
            "upscale_small_inputs": False,
            "internal_vae_alignment": "private_replicated_edge_padding_to_32px",
            "original_meaning": "skip_extra_service_compression",
        },
        "validated_duration_seconds": [5, 15],
        "execution_modes": list(EXECUTION_MODES),
        "device_memory_backend": {
            "policy": "startup_fixed_profile_then_minimum_latency_graph",
            "selection": "automatic_inside_selected_profile",
            "cross_profile_routing": False,
            "user_execution_modes": False,
            "weight_tier": launcher_weight_tier(selected_launcher),
            "vram_profile": launcher_vram_profile(selected_launcher),
            "profiles": {
                backend.vram_profile: {
                    "first_generation": list(
                        backend.first_generation_levels
                    ),
                    "maximum_duration_seconds": (
                        backend.maximum_duration_seconds
                    ),
                    "boundary_720p15_reference_items": (
                        backend.boundary_reference_items
                    ),
                    "second_sampling": [
                        public_resolution_name(level)
                        for level in backend.second_sampling_levels
                    ],
                }
                for backend in RESOURCE_BACKENDS.values()
            },
        },
        "advanced_limits": {
            "dimension_min": MIN_CUSTOM_DIMENSION,
            "dimension_max": MAX_CUSTOM_DIMENSION,
            "short_edge_max": MAX_CUSTOM_SHORT_EDGE,
            "max_pixels": MAX_CUSTOM_PIXELS,
            "frames_min": 5,
            "frames_max": 362,
            "frame_grid": "17*n+5",
            "sampling_steps": {
                "base": {"min": BASE_SAMPLING_STEPS[0], "max": BASE_SAMPLING_STEPS[1], "default": 20},
                "lora": {"min": LORA_SAMPLING_STEPS[0], "max": LORA_SAMPLING_STEPS[1], "default": 8},
            },
            "acceleration": {
                "min": ACCELERATION_RANGE[0], "max": ACCELERATION_RANGE[1],
                "step": 1, "default": 0,
                "meaning": (
                    "0=Dense计算；75=Human审阅质量拐点；"
                    "75–100=明确允许质量风险的激进区"
                ),
            },
            "selflift": {
                "available": True,
                "model_variants": ["base", "lora"],
                "initial_resolutions": ["360p", "480p", "540p", "720p", "900p"],
                "default_initial_resolution": "540p",
                "default_high_resolution_steps": 2,
                "sigma_scale": {
                    "min": SELFLIFT_SIGMA_SCALE[0],
                    "max": SELFLIFT_SIGMA_SCALE[1],
                    "step": 0.05,
                    "default": 1.0,
                },
                "transition_semantics": "completed_low_resolution_steps",
                "method": "learned_h3_clean_endpoint_lift_v1",
            },
            "quality_protection": "internal_non_disableable",
            "legacy_execution_controls": {
                "status": "accepted_for_persisted_clients_but_not_exposed",
                "fields": ["actual_steps", "lora_steps", "attention_keep_ratio", "sparse_scope"],
            },
            "upscaler": {
                "levels": [
                    public_resolution_name(level) for level in UPSCALE_LEVELS
                ],
                "dimension_min": MIN_UPSCALE_DIMENSION,
                "dimension_max": MAX_UPSCALE_DIMENSION,
                "max_pixels": MAX_UPSCALE_PIXELS,
                "preserve_aspect_ratio": True,
                "deprecated": True,
                "replacement": "h3_second_sampling",
            },
            "second_sampling": {
                "implementation": "dual_model_second_sampling_v3",
                "default_method": "temporal",
                "methods": {
                    "temporal": {
                        "implementation": "flashvsr_v1.1_tiny_long_1step",
                        "input": "completed_video",
                        "inference_steps": 1,
                        "temporal": True,
                    },
                    "h3": {
                        "implementation": "h3_learned_3d_second_sampling_v2",
                        "input": "clean_av_latent",
                        "inference_steps": "user_1_to_8",
                        "temporal": True,
                    },
                },
                "latent_initialization": "learned_3d_bf16",
                "sampler": "sa_solver",
                "levels": [
                    public_resolution_name(level)
                    for level in SECOND_SAMPLING_RESOLUTIONS
                ],
                "resolution": {
                    "min": 720,
                    "max": 1440,
                    "step": 1,
                    "detents": [720, 900, 1080, 1220, 1440],
                },
                "steps": {
                    "min": SECOND_SAMPLING_STEPS[0],
                    "max": SECOND_SAMPLING_STEPS[1],
                    "default": 4,
                },
                "automatic_denoise_by_steps": {
                    str(steps): denoise
                    for steps, denoise in SECOND_SAMPLING_AUTO_DENOISE_BY_STEPS.items()
                },
                "model_variants": ["base"],
                "strengths": {
                    name: {"denoise": denoise}
                    for name, denoise in SECOND_SAMPLING_STRENGTHS.items()
                },
                "denoise": {
                    "min": SECOND_SAMPLING_DENOISE[0],
                    "max": SECOND_SAMPLING_DENOISE[1],
                    "default": 0.20,
                },
                "memory_execution": "automatic_device_budget_optimizer",
                "preserve_audio": True,
                "conditioning": "reuse_source_prompt_and_reference_media",
                "full_canvas_preferred": True,
            },
            "preview": {
                "available": False,
                "deprecated": True,
                "replacement": "h3_second_sampling",
            },
        },
    }
