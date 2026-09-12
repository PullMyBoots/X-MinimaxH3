from __future__ import annotations

import argparse
import atexit
import asyncio
import dataclasses
import errno
import io
import json
import math
import os
import re
import shutil
import socket
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from aiohttp import web
from PIL import Image, UnidentifiedImageError

from . import __version__
from .backend import (
    CheckpointResult,
    GenerationResult,
    JobCancelled,
    build_native_backend,
)
from .config import ServicePaths
from .contract import (
    ContractError, DEFAULT_REFERENCE_IMAGE_RESOLUTION,
    DEFAULT_REFERENCE_VIDEO_RESOLUTION, ENGINES, REFERENCE_MEDIA_RESOLUTIONS,
    GENERATION_RESOLUTION_MAX, PROGRESSIVE_RESOLUTION_DETENTS,
    SERVICE_FAMILIES, MODEL_LAUNCHERS, LEGACY_MODEL_LAUNCHERS,
    GenerationSpec, SecondSamplingSpec, VideoRepairSpec,
    default_quality, engine_family, engine_variant, launcher_family,
    launcher_vram_profile, launcher_weight_tier, normalize_launcher,
    progressive_short_edge, resolve_engine, resolve_geometry,
    resolve_launcher, resolve_short_edge_geometry,
    second_sampling_short_edge,
    public_options,
)
from .models import model_status
from .openapi import document as openapi_document
from .memory_policy import (
    HOST_MEMORY_PROFILES,
    HostMemoryProfile,
    HostMemoryStatus,
    current_process_pss_gib,
    detect_host_memory,
    host_memory_budget_bounds,
    resolve_host_memory_budget_profile,
    resolve_host_memory_profile,
    validate_profile_for_weight_tier,
    validate_workload_for_profile,
)
from .memory_budget import (
    InMemoryBudgetController,
    LinuxCgroupMemoryBudgetController,
)
from .resources import ResourceMonitor
from .upscaler import FlashVSRUpscaler, RetiredFlashVSRUpscaler
from .workspace import WorkspaceController, WorkspaceLayout
from .generation_limits import (
    GenerationLimitPolicy,
    detect_gpu_vram_gib,
    load_generation_limit_policy,
    persist_generation_limit_policy,
)
from .deployment_profiles import LAUNCHER_DEFINITIONS, automatic_launcher
from .lora_registry import resolve_lora_profile
from .infinite_video import (
    FPS as INFINITE_FPS,
    H3_FRAME_ORIGIN,
    InfiniteContinuationSpec,
    InfiniteProjectStore,
    RETRYABLE_TAIL_STATUSES,
    compile_infinite_prompt,
    context_frames_for_seconds,
    memory_capacity,
    visible_frames_for_seconds,
)


MAX_IMAGE_BYTES = 25 * 1024 * 1024
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}
MAX_IMAGE_PIXELS = 80_000_000
MAX_REFERENCE_IMAGES = 9
MAX_REFERENCE_VIDEOS = 3
MAX_REFERENCE_AUDIOS = 3
MAX_REFERENCE_VIDEO_BYTES = 200 * 1024 * 1024
VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".webm", ".avi"}
AUDIO_SUFFIXES = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".opus"}
CHECKPOINT_PREVIEW_SETTING_RESOLUTIONS = ("360p", "480p", "720p")


def _reference_media_settings_path(data_dir: Path) -> Path:
    return data_dir / "settings" / "reference_media.json"


def _default_reference_media_settings() -> dict[str, str]:
    return {
        "image_resolution": DEFAULT_REFERENCE_IMAGE_RESOLUTION,
        "video_resolution": DEFAULT_REFERENCE_VIDEO_RESOLUTION,
    }


def _load_reference_media_settings(data_dir: Path) -> dict[str, str]:
    defaults = _default_reference_media_settings()
    try:
        document = json.loads(
            _reference_media_settings_path(data_dir).read_text(encoding="utf-8")
        )
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return defaults
    if not isinstance(document, dict):
        return defaults
    image_resolution = str(document.get("image_resolution", "")).strip().lower()
    video_resolution = str(document.get("video_resolution", "")).strip().lower()
    if image_resolution not in REFERENCE_MEDIA_RESOLUTIONS:
        image_resolution = defaults["image_resolution"]
    if video_resolution not in REFERENCE_MEDIA_RESOLUTIONS:
        video_resolution = defaults["video_resolution"]
    return {
        "image_resolution": image_resolution,
        "video_resolution": video_resolution,
    }


def _persist_reference_media_settings(
    data_dir: Path,
    settings: dict[str, str],
) -> None:
    path = _reference_media_settings_path(data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(settings, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _face_repair_settings_path(data_dir: Path) -> Path:
    return data_dir / "settings" / "face_repair.json"


def _default_face_repair_settings() -> dict[str, Any]:
    return {"canvas_size": 768, "capacity": 4}


def _load_face_repair_settings(data_dir: Path) -> dict[str, Any]:
    defaults = _default_face_repair_settings()
    try:
        document = json.loads(
            _face_repair_settings_path(data_dir).read_text(encoding="utf-8")
        )
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return defaults
    if not isinstance(document, dict):
        return defaults
    try:
        repair = VideoRepairSpec.from_mapping({
            "canvas_size": document.get("canvas_size", defaults["canvas_size"]),
            "capacity": document.get("capacity", defaults["capacity"]),
        })
    except ContractError:
        return defaults
    return {"canvas_size": repair.canvas_size, "capacity": repair.max_faces}


def _persist_face_repair_settings(
    data_dir: Path,
    settings: dict[str, Any],
) -> None:
    path = _face_repair_settings_path(data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(settings, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _checkpoint_preview_settings_path(data_dir: Path) -> Path:
    return data_dir / "settings" / "checkpoint_preview.json"


def _default_checkpoint_preview_settings() -> dict[str, Any]:
    return {"steps": 2, "resolution": "360p"}


def _load_checkpoint_preview_settings(data_dir: Path) -> dict[str, Any]:
    defaults = _default_checkpoint_preview_settings()
    try:
        document = json.loads(
            _checkpoint_preview_settings_path(data_dir).read_text(encoding="utf-8")
        )
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return defaults
    if not isinstance(document, dict):
        return defaults
    try:
        steps = int(document.get("steps", defaults["steps"]))
    except (TypeError, ValueError):
        steps = int(defaults["steps"])
    if not 1 <= steps <= 4:
        steps = int(defaults["steps"])
    resolution = str(
        document.get("resolution", defaults["resolution"])
    ).strip().lower()
    if resolution not in CHECKPOINT_PREVIEW_SETTING_RESOLUTIONS:
        resolution = str(defaults["resolution"])
    return {"steps": steps, "resolution": resolution}


def _persist_checkpoint_preview_settings(
    data_dir: Path,
    settings: dict[str, Any],
) -> None:
    path = _checkpoint_preview_settings_path(data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(settings, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _second_sampling_window_settings_path(data_dir: Path) -> Path:
    return data_dir / "settings" / "second_sampling_window.json"


def _default_second_sampling_window_settings() -> dict[str, Any]:
    return {
        "enabled": True,
        "window_seconds": 5.0,
        "overlap_seconds": 1.0,
    }


def _load_second_sampling_window_settings(data_dir: Path) -> dict[str, Any]:
    defaults = _default_second_sampling_window_settings()
    try:
        document = json.loads(
            _second_sampling_window_settings_path(data_dir).read_text(
                encoding="utf-8"
            )
        )
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return defaults
    if not isinstance(document, dict):
        return defaults
    enabled = document.get("enabled", defaults["enabled"])
    if not isinstance(enabled, bool):
        return defaults
    try:
        window_seconds = float(
            document.get("window_seconds", defaults["window_seconds"])
        )
        overlap_seconds = float(
            document.get("overlap_seconds", defaults["overlap_seconds"])
        )
    except (TypeError, ValueError):
        return defaults
    if not math.isfinite(window_seconds) or not 3.0 <= window_seconds <= 15.0:
        return defaults
    if not math.isfinite(overlap_seconds) or not 0.0 <= overlap_seconds <= 4.0:
        return defaults
    return {
        "enabled": enabled,
        "window_seconds": round(window_seconds, 1),
        "overlap_seconds": round(overlap_seconds, 1),
    }


def _persist_second_sampling_window_settings(
    data_dir: Path,
    settings: dict[str, Any],
) -> None:
    path = _second_sampling_window_settings_path(data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(settings, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


DEFAULT_LORA_CHECKPOINT = "minimax_h3_turbo_v4_step600_ema.safetensors"


def _lora_settings_path(data_dir: Path) -> Path:
    return data_dir / "settings" / "lora.json"


def _discover_lora_checkpoints(model_dir: Path) -> list[dict[str, Any]]:
    """Return safe, header-validated H3 LoRAs installed under models/loras."""

    root = (model_dir / "loras").absolute()
    if not root.is_dir():
        return []
    discovered: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*.safetensors")):
        if not path.is_file():
            continue
        relative = path.absolute().relative_to(root).as_posix()
        compatible = False
        pair_count = 0
        base_model = None
        try:
            from safetensors import safe_open

            with safe_open(str(path), framework="pt", device="cpu") as checkpoint:
                keys = set(checkpoint.keys())
                suffix = ".lora_A.weight"
                native = [
                    key for key in keys
                    if key.startswith("blocks.")
                    and key.endswith(suffix)
                    and f"{key[:-len(suffix)]}.lora_B.weight" in keys
                ]
                metadata = checkpoint.metadata() or {}
                base_model = metadata.get("base_model")
                diffusers_suffix = ".lora_A.default.weight"
                diffusers = [
                    key for key in keys
                    if key.endswith(diffusers_suffix)
                    and f"{key[:-len(diffusers_suffix)]}.lora_B.default.weight" in keys
                ]
                pair_count = len(native) if native else len(diffusers)
                compatible = bool(native) or len(diffusers) == 312
        except Exception:
            compatible = False
        profile = resolve_lora_profile(path)
        discovered.append({
            "id": relative,
            "filename": path.name,
            "bytes": path.stat().st_size,
            "compatible": compatible,
            "pair_count": pair_count,
            "base_model": base_model,
            "profile": profile.public_dict(),
        })
    return discovered


def _load_lora_selection(data_dir: Path) -> str:
    try:
        document = json.loads(
            _lora_settings_path(data_dir).read_text(encoding="utf-8")
        )
        return str(document.get("checkpoint", "")).strip()
    except (FileNotFoundError, OSError, AttributeError, json.JSONDecodeError):
        return ""


def _persist_lora_selection(data_dir: Path, checkpoint: str) -> None:
    path = _lora_settings_path(data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps({"checkpoint": checkpoint}, ensure_ascii=False, indent=2)
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _validate_image(content: bytes, role: str) -> None:
    try:
        with Image.open(io.BytesIO(content)) as image:
            width, height = image.size
            if width <= 0 or height <= 0 or width * height > MAX_IMAGE_PIXELS:
                raise ContractError(f"{role} has unsupported dimensions")
            if (image.format or "").upper() not in {"PNG", "JPEG", "WEBP"}:
                raise ContractError(f"{role} must contain PNG, JPEG or WebP data")
            image.verify()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as error:
        raise ContractError(f"{role} is not a valid image") from error


def _image_mime(content: bytes, role: str) -> str:
    _validate_image(content, role)
    with Image.open(io.BytesIO(content)) as image:
        return {
            "PNG": "image/png", "JPEG": "image/jpeg", "WEBP": "image/webp"
        }[(image.format or "").upper()]


def _validate_reference_video(content: bytes, role: str) -> float | None:
    """Probe container/video metadata without decoding the expensive payload."""

    try:
        import av

        with av.open(io.BytesIO(content)) as container:
            if not container.streams.video:
                raise ContractError(f"{role} contains no video stream")
            stream = container.streams.video[0]
            if stream.width <= 0 or stream.height <= 0:
                raise ContractError(f"{role} has invalid video dimensions")
            duration = None
            if container.duration is not None:
                duration = float(container.duration) / float(av.time_base)
            elif stream.duration is not None and stream.time_base is not None:
                duration = float(stream.duration * stream.time_base)
            elif stream.frames and (stream.average_rate or stream.guessed_rate):
                duration = float(stream.frames) / float(stream.average_rate or stream.guessed_rate)
            if duration is not None and not 1.95 <= duration <= 15.1:
                raise ContractError(f"{role} must be between 2 and 15 seconds")
            return duration
    except ContractError:
        raise
    except Exception as error:
        raise ContractError(f"{role} is not a decodable video container") from error


def _validate_reference_audio(content: bytes, role: str) -> float:
    try:
        import av

        with av.open(io.BytesIO(content)) as container:
            if not container.streams.audio:
                raise ContractError(f"{role} contains no audio stream")
            stream = container.streams.audio[0]
            if stream.rate <= 0:
                raise ContractError(f"{role} has invalid sample rate")
            if container.duration is not None:
                duration = float(container.duration) / float(av.time_base)
            elif stream.duration is not None and stream.time_base is not None:
                duration = float(stream.duration * stream.time_base)
            else:
                duration = 0.0
                for frame in container.decode(stream):
                    duration += float(frame.samples) / float(frame.sample_rate)
            if not 0.1 <= duration <= 15.1:
                raise ContractError(f"{role} must be between 0.1 and 15 seconds")
            return duration
    except ContractError:
        raise
    except Exception as error:
        raise ContractError(f"{role} is not a decodable audio container") from error


@dataclass
class JobRecord:
    id: str
    spec: GenerationSpec
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    status: str = "queued"
    first_frame: Path | None = None
    last_frame: Path | None = None
    reference_images: tuple[Path, ...] = ()
    reference_videos: tuple[Path, ...] = ()
    reference_audios: tuple[Path, ...] = ()
    runtime_key: str | None = None
    backend_prompt_id: str | None = None
    elapsed_seconds: float | None = None
    generation_elapsed_seconds: float | None = None
    upscale_elapsed_seconds: float | None = None
    upscale_peak_allocated_mib: float | None = None
    upscale_peak_reserved_mib: float | None = None
    output_path: Path | None = None
    final_latents_path: Path | None = None
    source_job_id: str | None = None
    source_latents_path: Path | None = None
    source_video_path: Path | None = None
    token_memory_path: Path | None = None
    source_token_memory_path: Path | None = None
    infinite_continuation: InfiniteContinuationSpec | None = None
    infinite_final_window_ids: tuple[str, ...] = ()
    second_sampling: SecondSamplingSpec | None = None
    video_repair: VideoRepairSpec | None = None
    preview_path: Path | None = None
    preview_decision: str | None = None
    checkpoint_path: Path | None = None
    checkpoint_completed_steps: int | None = None
    checkpoint_total_steps: int | None = None
    checkpoint_retained: bool = False
    pending_action: str = "generate"
    error: str | None = None
    progress_percent: float = 0.0
    progress_stage: str = "queued"
    progress_detail: str = "等待执行"
    estimated_total_seconds: float | None = None
    estimated_remaining_seconds: float | None = None
    started_at: float | None = None
    inference_plan: dict[str, Any] | None = None
    stage_seconds: dict[str, float] = field(default_factory=dict)

    @property
    def condition_mode(self) -> str:
        if self.reference_images or self.reference_videos or self.reference_audios:
            return "reference"
        if self.first_frame and self.last_frame:
            return "first_last"
        if self.first_frame:
            return "first"
        if self.last_frame:
            return "last"
        return "text"

    def public(
        self,
        queue_position: int | None = None,
        *,
        estimated_remaining_seconds: float | None = None,
        estimated_queue_seconds: float | None = None,
        estimated_completion_seconds: float | None = None,
    ) -> dict[str, Any]:
        result = {
            "id": self.id,
            "status": self.status,
            "queue_position": queue_position,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "request": {
                **self.spec.to_dict(),
                "condition_mode": self.condition_mode,
                "has_first_frame": self.first_frame is not None,
                "has_last_frame": self.last_frame is not None,
                "reference_image_count": len(self.reference_images),
                "reference_video_count": len(self.reference_videos),
                "reference_audio_count": len(self.reference_audios),
                "reference_video_audio_policy": "ignored",
            },
            "elapsed_seconds": self.elapsed_seconds,
            "generation_elapsed_seconds": self.generation_elapsed_seconds,
            "stage_seconds": dict(self.stage_seconds),
            "upscale_elapsed_seconds": self.upscale_elapsed_seconds,
            "upscale_peak_allocated_mib": self.upscale_peak_allocated_mib,
            "upscale_peak_reserved_mib": self.upscale_peak_reserved_mib,
            "progress": {
                "percent": round(self.progress_percent, 1),
                "stage": self.progress_stage,
                "detail": self.progress_detail,
                "estimated_total_seconds": self.estimated_total_seconds,
                "estimated_remaining_seconds": estimated_remaining_seconds,
                "estimated_queue_seconds": estimated_queue_seconds,
                "estimated_completion_seconds": estimated_completion_seconds,
            },
            "error": self.error,
        }
        if self.inference_plan is not None:
            result["inference_plan"] = dict(self.inference_plan)
        if self.output_path is not None:
            result["video_url"] = f"/api/v1/jobs/{self.id}/video"
            result["download_name"] = self.output_path.name
        launcher = LAUNCHER_DEFINITIONS.get(self.spec.runtime_launcher)
        h3_second_sampling_ready = bool(
            self.status == "succeeded"
            and self.final_latents_path is not None
            and self.final_latents_path.is_file()
            and launcher is not None
            and launcher.backend.second_sampling_levels
        )
        temporal_second_sampling_ready = bool(
            self.status == "succeeded"
            and self.output_path is not None
            and self.output_path.is_file()
        )
        result["second_sampling_methods"] = {
            "temporal": temporal_second_sampling_ready,
            "h3": h3_second_sampling_ready,
        }
        result["second_sampling_available"] = bool(
            temporal_second_sampling_ready or h3_second_sampling_ready
        )
        result["video_repair_available"] = bool(
            self.status == "succeeded"
            and self.output_path is not None
            and self.output_path.is_file()
            and self.spec.service_family == "first_last"
        )
        if self.second_sampling is not None:
            public_second_sampling = self.second_sampling.to_dict()
            if public_second_sampling["resolution"] == "2k":
                public_second_sampling["resolution"] = "1440p"
            result["second_sampling"] = {
                **public_second_sampling,
                "source_job_id": self.source_job_id,
                "source_latent_available": bool(
                    self.source_latents_path is not None
                    and self.source_latents_path.is_file()
                ),
            }
        if self.video_repair is not None:
            result["video_repair"] = {
                **self.video_repair.to_dict(),
                "source_job_id": self.source_job_id,
            }
        if self.infinite_continuation is not None:
            result["infinite_continuation"] = self.infinite_continuation.to_dict()
        if self.preview_path is not None:
            result["preview"] = {
                "ready": self.preview_path.is_file(),
                "video_url": f"/api/v1/jobs/{self.id}/preview",
                "decision_required": self.status == "awaiting_preview",
                "decision": self.preview_decision,
            }
        if self.spec.execution_mode == "checkpoint" or self.checkpoint_completed_steps:
            result["checkpoint"] = {
                "completed_steps": self.checkpoint_completed_steps,
                "total_steps": self.checkpoint_total_steps,
                "retained": self.checkpoint_retained,
                "resume_available": bool(
                    self.status in {"checkpointed", "failed"}
                    and self.checkpoint_retained
                    and self.checkpoint_path is not None
                    and self.checkpoint_path.is_file()
                ),
            }
        return result


@dataclass
class PreviewControl:
    event: threading.Event = field(default_factory=threading.Event)
    decision: str | None = None


class JobService:
    def __init__(
        self,
        data_dir: Path,
        backend: Any,
        *,
        max_queued_jobs: int = 32,
        output_root: Path | None = None,
        upscaler: Any | None = None,
        memory_profile_getter: Any | None = None,
    ) -> None:
        self.data_dir = data_dir
        self.backend = backend
        self.output_root = output_root.resolve() if output_root is not None else None
        self.max_queued_jobs = max(1, int(max_queued_jobs))
        self.upscaler = upscaler
        self.memory_profile_getter = memory_profile_getter
        self.jobs: dict[str, JobRecord] = {}
        self.pending: list[str] = []
        self.queue_changed = asyncio.Condition()
        self.cancel_events: dict[str, asyncio.Event] = {}
        self.preview_controls: dict[str, PreviewControl] = {}
        self.worker_task: asyncio.Task | None = None
        self.warmup_task: asyncio.Task | None = None
        self.fixed_engine: str | None = None
        for name in ("jobs", "uploads", "logs", "checkpoints"):
            (self.data_dir / name).mkdir(parents=True, exist_ok=True)
        self._load_jobs()

    def switch_workspace(self, layout: WorkspaceLayout) -> None:
        """Rebind persistent job state after the API has proven the GPU idle."""

        if self.pending or any(
            job.status in {"queued", "starting_backend", "running", "awaiting_preview"}
            for job in self.jobs.values()
        ):
            raise RuntimeError("cannot switch workspace while jobs are active")
        self.data_dir = layout.data_dir
        self.output_root = layout.output_dir
        self.jobs.clear()
        self.pending.clear()
        self.cancel_events.clear()
        self.preview_controls.clear()
        for name in ("jobs", "uploads", "logs", "checkpoints"):
            (self.data_dir / name).mkdir(parents=True, exist_ok=True)
        self._load_jobs()

    def _load_jobs(self) -> None:
        for path in sorted((self.data_dir / "jobs").glob("*.json")):
            try:
                document = json.loads(path.read_text(encoding="utf-8"))
                internal = document.get("_internal", {})
                request = internal.get("spec", document["request"])
                spec = GenerationSpec.from_mapping(
                    request,
                    allow_second_sampling_target=bool(
                        isinstance(internal.get("second_sampling"), dict)
                        or isinstance(internal.get("video_repair"), dict)
                    ),
                )

                def optional_path(name: str) -> Path | None:
                    value = internal.get(name)
                    return Path(value) if value else None

                status = str(document.get("status", "failed"))
                error = document.get("error")
                if status in {"queued", "starting_backend", "running", "awaiting_preview"}:
                    status = "failed"
                    error = "service restarted before the task completed"
                job = JobRecord(
                    id=str(document["id"]),
                    spec=spec,
                    created_at=float(document.get("created_at", path.stat().st_mtime)),
                    updated_at=float(document.get("updated_at", path.stat().st_mtime)),
                    status=status,
                    first_frame=optional_path("first_frame"),
                    last_frame=optional_path("last_frame"),
                    reference_images=tuple(
                        Path(value)
                        for value in internal.get("reference_images", [])
                        if value
                    ),
                    reference_videos=tuple(
                        Path(value)
                        for value in internal.get("reference_videos", [])
                        if value
                    ),
                    reference_audios=tuple(
                        Path(value)
                        for value in internal.get("reference_audios", [])
                        if value
                    ),
                    runtime_key=internal.get("runtime_key", document.get("backend_key")),
                    backend_prompt_id=internal.get("backend_prompt_id"),
                    elapsed_seconds=document.get("elapsed_seconds"),
                    generation_elapsed_seconds=document.get(
                        "generation_elapsed_seconds"
                    ),
                    upscale_elapsed_seconds=document.get("upscale_elapsed_seconds"),
                    upscale_peak_allocated_mib=document.get(
                        "upscale_peak_allocated_mib"
                    ),
                    upscale_peak_reserved_mib=document.get(
                        "upscale_peak_reserved_mib"
                    ),
                    output_path=optional_path("output_path"),
                    final_latents_path=optional_path("final_latents_path"),
                    source_job_id=internal.get("source_job_id"),
                    source_latents_path=optional_path("source_latents_path"),
                    source_video_path=optional_path("source_video_path"),
                    token_memory_path=optional_path("token_memory_path"),
                    source_token_memory_path=optional_path("source_token_memory_path"),
                    infinite_continuation=(
                        InfiniteContinuationSpec.from_dict(
                            internal["infinite_continuation"]
                        )
                        if isinstance(internal.get("infinite_continuation"), dict)
                        else None
                    ),
                    infinite_final_window_ids=tuple(
                        str(value)
                        for value in internal.get(
                            "infinite_final_window_ids", []
                        )
                    ),
                    second_sampling=(
                        SecondSamplingSpec(**internal["second_sampling"])
                        if isinstance(internal.get("second_sampling"), dict)
                        else None
                    ),
                    video_repair=(
                        VideoRepairSpec.from_mapping(internal["video_repair"])
                        if isinstance(internal.get("video_repair"), dict)
                        else None
                    ),
                    preview_path=optional_path("preview_path"),
                    preview_decision=internal.get("preview_decision"),
                    checkpoint_path=optional_path("checkpoint_path"),
                    checkpoint_completed_steps=internal.get(
                        "checkpoint_completed_steps"
                    ),
                    checkpoint_total_steps=internal.get("checkpoint_total_steps"),
                    checkpoint_retained=bool(
                        internal.get("checkpoint_retained", False)
                    ),
                    pending_action=str(internal.get("pending_action", "generate")),
                    error=error,
                    progress_percent=float(document.get("progress", {}).get("percent", 0.0)),
                    progress_stage=str(document.get("progress", {}).get("stage", status)),
                    progress_detail=str(document.get("progress", {}).get("detail", "")),
                    estimated_total_seconds=document.get("progress", {}).get(
                        "estimated_total_seconds"
                    ),
                    estimated_remaining_seconds=document.get("progress", {}).get(
                        "estimated_remaining_seconds"
                    ),
                    inference_plan=(
                        dict(document["inference_plan"])
                        if isinstance(document.get("inference_plan"), dict)
                        else None
                    ),
                )
                if job.infinite_continuation is not None:
                    job.spec = dataclasses.replace(
                        job.spec,
                        output_frames=job.infinite_continuation.output_frames,
                    )
                if job.output_path is not None and not job.output_path.is_file():
                    job.status = "failed"
                    job.error = "recorded output video is missing"
                    job.output_path = None
                if job.status == "checkpointed" and (
                    not job.checkpoint_retained
                    or job.checkpoint_path is None
                    or not job.checkpoint_path.is_file()
                ):
                    job.checkpoint_retained = False
                self.jobs[job.id] = job
            except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError):
                # One damaged history record must not prevent the service from starting.
                continue
        # A second-sampling card inherits the complete source clock. Restore
        # that derived value after all persisted source cards are available.
        for job in self.jobs.values():
            if job.second_sampling is None or job.source_job_id is None:
                continue
            source = self.jobs.get(job.source_job_id)
            if source is not None and source.spec.output_frames is not None:
                job.spec = dataclasses.replace(
                    job.spec, output_frames=source.spec.output_frames
                )

    def queue_position(self, job_id: str) -> int | None:
        try:
            return self.pending.index(job_id) + 1
        except ValueError:
            return None

    def serialize(self, job: JobRecord) -> dict[str, Any]:
        queue_position = self.queue_position(job.id)
        remaining = self._dynamic_remaining(job)
        queue_seconds = None
        completion_seconds = remaining
        if queue_position is not None:
            queue_seconds, completion_seconds = self._queued_eta(job.id)
        return job.public(
            queue_position,
            estimated_remaining_seconds=remaining,
            estimated_queue_seconds=queue_seconds,
            estimated_completion_seconds=completion_seconds,
        )

    @staticmethod
    def _dynamic_remaining(job: JobRecord, now: float | None = None) -> float | None:
        """Return a wall-clock countdown without treating stage progress as linear."""

        if job.status == "succeeded":
            return 0.0
        if job.status == "checkpointed":
            return None
        total = job.estimated_total_seconds
        if total is None:
            return None
        if job.status == "starting_backend" and job.started_at is None:
            return None
        if job.status == "awaiting_preview":
            return None
        if job.status not in {"starting_backend", "running"} or job.started_at is None:
            return round(max(0.0, total), 1)
        elapsed = max(0.0, (time.time() if now is None else now) - job.started_at)
        return round(max(0.0, total - elapsed), 1)

    def _queued_eta(self, job_id: str) -> tuple[float | None, float | None]:
        """Estimate queue wait and completion from the one-GPU FIFO schedule."""

        if job_id not in self.pending:
            return None, None
        now = time.time()
        seconds = 0.0
        known = False
        unknown_active = False
        for active in self.jobs.values():
            if active.status not in {"starting_backend", "running", "awaiting_preview"}:
                continue
            value = self._dynamic_remaining(active, now)
            if value is not None:
                seconds += value
                known = True
            else:
                unknown_active = True
        if unknown_active:
            return None, None
        target = self.jobs[job_id]
        own = target.estimated_total_seconds
        for pending_id in self.pending:
            pending = self.jobs.get(pending_id)
            if pending is None:
                continue
            if pending_id == job_id:
                queue_seconds = seconds if known or own is not None else None
                if own is None:
                    return queue_seconds, None
                return queue_seconds, round(seconds + own, 1)
            if pending.estimated_total_seconds is not None:
                seconds += pending.estimated_total_seconds
                known = True
        return None, None

    def persist(self, job: JobRecord) -> None:
        path = self.data_dir / "jobs" / f"{job.id}.json"
        temporary = path.with_suffix(".json.tmp")
        document = self.serialize(job)
        document["schema_version"] = 2
        document["_internal"] = {
            "spec": job.spec.to_dict(include_execution=True),
            "first_frame": str(job.first_frame) if job.first_frame else None,
            "last_frame": str(job.last_frame) if job.last_frame else None,
            "reference_images": [str(path) for path in job.reference_images],
            "reference_videos": [str(path) for path in job.reference_videos],
            "reference_audios": [str(path) for path in job.reference_audios],
            "backend_prompt_id": job.backend_prompt_id,
            "runtime_key": job.runtime_key,
            "output_path": str(job.output_path) if job.output_path else None,
            "final_latents_path": (
                str(job.final_latents_path) if job.final_latents_path else None
            ),
            "source_job_id": job.source_job_id,
            "source_latents_path": (
                str(job.source_latents_path) if job.source_latents_path else None
            ),
            "source_video_path": (
                str(job.source_video_path) if job.source_video_path else None
            ),
            "token_memory_path": (
                str(job.token_memory_path) if job.token_memory_path else None
            ),
            "source_token_memory_path": (
                str(job.source_token_memory_path)
                if job.source_token_memory_path else None
            ),
            "infinite_continuation": (
                job.infinite_continuation.to_dict()
                if job.infinite_continuation else None
            ),
            "infinite_final_window_ids": list(
                job.infinite_final_window_ids
            ),
            "second_sampling": (
                job.second_sampling.to_dict() if job.second_sampling else None
            ),
            "video_repair": (
                job.video_repair.to_dict() if job.video_repair else None
            ),
            "preview_path": str(job.preview_path) if job.preview_path else None,
            "preview_decision": job.preview_decision,
            "checkpoint_path": str(job.checkpoint_path) if job.checkpoint_path else None,
            "checkpoint_completed_steps": job.checkpoint_completed_steps,
            "checkpoint_total_steps": job.checkpoint_total_steps,
            "checkpoint_retained": job.checkpoint_retained,
            "pending_action": job.pending_action,
            "inference_plan": job.inference_plan,
        }
        temporary.write_text(
            json.dumps(document, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)

    async def submit(
        self,
        spec: GenerationSpec,
        uploads: dict[str, tuple[str, bytes]],
    ) -> JobRecord:
        if len(self.pending) >= self.max_queued_jobs:
            raise ContractError("generation queue is full; try again later")
        if spec.long_video is not None:
            from .long_video import validate_reference_inputs
            try:
                validate_reference_inputs(
                    spec.long_video, service_family=spec.service_family,
                    first_frame="first_frame" in uploads, last_frame="last_frame" in uploads,
                    reference_images=sum(key.startswith("reference_image_") for key in uploads),
                    reference_audios=sum(key.startswith("reference_audio_") for key in uploads),
                    reference_videos=sum(key.startswith("reference_video_") for key in uploads),
                )
            except ValueError as error:
                raise ContractError(str(error)) from error
        job_id = str(uuid.uuid4())
        reference_video_duration = 0.0
        for role, (original_name, content) in uploads.items():
            suffix = Path(original_name).suffix.lower()
            if role.startswith("reference_video_"):
                if suffix not in VIDEO_SUFFIXES:
                    raise ContractError(f"{role} must be MP4, MOV, MKV, WebM or AVI")
                duration = _validate_reference_video(content, role)
                if duration is not None:
                    reference_video_duration += duration
            elif role.startswith("reference_audio_"):
                if suffix not in AUDIO_SUFFIXES:
                    raise ContractError(f"{role} must be WAV, MP3, FLAC, M4A, OGG or Opus")
                _validate_reference_audio(content, role)
            else:
                if suffix not in IMAGE_SUFFIXES:
                    raise ContractError(f"{role} must be PNG, JPEG or WebP")
                _validate_image(content, role)
        if reference_video_duration > 15.1:
            raise ContractError("total reference video duration must not exceed 15 seconds")
        upload_dir = self.data_dir / "uploads" / job_id
        upload_dir.mkdir(parents=True, exist_ok=True)
        paths: dict[str, Path] = {}
        for role, (original_name, content) in uploads.items():
            suffix = Path(original_name).suffix.lower()
            target = upload_dir / f"{role}{suffix}"
            target.write_bytes(content)
            paths[role] = target

        job = JobRecord(
            id=job_id,
            spec=spec,
            first_frame=paths.get("first_frame"),
            last_frame=paths.get("last_frame"),
            reference_images=tuple(
                path for role, path in sorted(paths.items())
                if role.startswith("reference_image_")
            ),
            reference_videos=tuple(
                path for role, path in sorted(paths.items())
                if role.startswith("reference_video_")
            ),
            reference_audios=tuple(
                path for role, path in sorted(paths.items())
                if role.startswith("reference_audio_")
            ),
        )
        job.estimated_total_seconds = self._estimate_total(spec, job.condition_mode)
        job.estimated_remaining_seconds = job.estimated_total_seconds
        self.jobs[job_id] = job
        self.cancel_events[job_id] = asyncio.Event()
        self.preview_controls[job_id] = PreviewControl()
        async with self.queue_changed:
            self.pending.append(job_id)
            self.queue_changed.notify()
        self.persist(job)
        return job

    async def submit_second_sampling(
        self,
        source: JobRecord,
        second_sampling: SecondSamplingSpec,
        *,
        prompt_override: str | None = None,
    ) -> JobRecord:
        """Queue a model-based high-resolution pass from one completed card."""

        if len(self.pending) >= self.max_queued_jobs:
            raise ContractError("generation queue is full; try again later")
        if source.status != "succeeded":
            raise ContractError("second sampling requires a completed source job")
        if second_sampling.method == "h3":
            if (
                source.final_latents_path is None
                or not source.final_latents_path.is_file()
            ):
                raise ContractError(
                    "this card has no retained H3 latent; generate a new card first"
                )
        elif source.output_path is None or not source.output_path.is_file():
            raise ContractError(
                "temporal second sampling requires a completed source video"
            )
        if second_sampling.method == "temporal":
            if self.upscaler is None:
                raise ContractError(
                    "the temporal second-sampling model is not configured"
                )
            upscaler_status = self.upscaler.status()
            if not upscaler_status.get("ready", False):
                missing = upscaler_status.get("missing") or []
                raise ContractError(
                    "the temporal second-sampling runtime is incomplete"
                    + (f": {missing[0]}" if missing else "")
                )

        job_id = str(uuid.uuid4())
        upload_dir = self.data_dir / "uploads" / job_id
        upload_dir.mkdir(parents=True, exist_ok=True)

        def clone(path: Path | None, role: str) -> Path | None:
            if path is None:
                return None
            if not path.is_file():
                raise ContractError(f"source conditioning file is missing: {role}")
            target = upload_dir / f"{role}{path.suffix.lower()}"
            shutil.copy2(path, target)
            return target

        first_frame = clone(source.first_frame, "first_frame")
        last_frame = clone(source.last_frame, "last_frame")
        reference_images = tuple(
            clone(path, f"reference_image_{index}")
            for index, path in enumerate(source.reference_images, start=1)
        )
        reference_videos = tuple(
            clone(path, f"reference_video_{index}")
            for index, path in enumerate(source.reference_videos, start=1)
        )
        reference_audios = tuple(
            clone(path, f"reference_audio_{index}")
            for index, path in enumerate(source.reference_audios, start=1)
        )

        target_spec = dataclasses.replace(
            source.spec,
            prompt=(
                source.spec.prompt
                if prompt_override is None
                else str(prompt_override).strip()
            ),
            engine=resolve_engine(
                source.spec.service_family,
                second_sampling.model_variant,
            ),
            resolution=second_sampling.resolution,
            width=second_sampling.width,
            height=second_sampling.height,
            advanced=True,
            # Keep the persisted GenerationSpec independently valid.  The
            # second-pass solver still takes its real one-to-eight step count
            # exclusively from SecondSamplingSpec below.
            custom_actual_steps=(
                20 if second_sampling.model_variant == "base" else None
            ),
            custom_lora_steps=(
                second_sampling.steps
                if second_sampling.model_variant == "lora"
                else None
            ),
            sampling_steps=None,
            acceleration=None,
            memory_mode=second_sampling.memory_mode,
            upscale_enabled=False,
            upscale_resolution=None,
            upscale_target_width=None,
            upscale_target_height=None,
            preview_mode="off",
            preview_step_index=None,
            preview_fast_finish=False,
            execution_mode="complete",
            checkpoint_step=None,
            checkpoint_preview=False,
        )
        job = JobRecord(
            id=job_id,
            spec=target_spec,
            first_frame=first_frame,
            last_frame=last_frame,
            reference_images=reference_images,
            reference_videos=reference_videos,
            reference_audios=reference_audios,
            source_job_id=source.id,
            source_latents_path=(
                source.final_latents_path.resolve()
                if source.final_latents_path is not None
                and source.final_latents_path.is_file()
                else None
            ),
            source_video_path=(
                source.output_path.resolve()
                if source.output_path is not None
                and source.output_path.is_file()
                else None
            ),
            second_sampling=second_sampling,
            pending_action="second_sampling",
            progress_detail=(
                "时序模型二次采样已加入队列"
                if second_sampling.method == "temporal"
                else "H3二次采样已加入队列"
            ),
        )
        megapixel_frames = (
            second_sampling.width
            * second_sampling.height
            * int(source.spec.output_frames or source.spec.frames)
            / 1_000_000
        )
        if second_sampling.method == "temporal":
            # Calibrated on RTX 4090: 124 frames at 1344x768 complete in
            # roughly 17--19 seconds once the CPU-resident worker is hot.
            job.estimated_total_seconds = round(
                max(8.0, 6.0 + 0.10 * megapixel_frames), 1
            )
        else:
            job.estimated_total_seconds = round(
                max(8.0, 10.0 + 0.14 * megapixel_frames * second_sampling.steps),
                1,
            )
        job.estimated_remaining_seconds = job.estimated_total_seconds
        self.jobs[job_id] = job
        self.cancel_events[job_id] = asyncio.Event()
        self.preview_controls[job_id] = PreviewControl()
        async with self.queue_changed:
            self.pending.append(job_id)
            self.queue_changed.notify()
        self.persist(job)
        return job

    async def submit_video_repair(
        self,
        source: JobRecord,
        video_repair: VideoRepairSpec,
    ) -> JobRecord:
        """Queue local H3 repair from the pixels of one completed video."""

        if len(self.pending) >= self.max_queued_jobs:
            raise ContractError("generation queue is full; try again later")
        if source.status != "succeeded":
            raise ContractError("video repair requires a completed source job")
        if source.output_path is None or not source.output_path.is_file():
            raise ContractError("video repair requires a completed source video")
        if source.spec.service_family != "first_last":
            raise ContractError(
                "face repair is available only for completed FL2VA source jobs"
            )
        repair_video = getattr(self.backend, "video_repair", None)
        if not callable(repair_video):
            raise ContractError("the active backend does not support video repair")

        target_spec = dataclasses.replace(
            source.spec,
            engine=resolve_engine(source.spec.service_family, "lora"),
            advanced=True,
            custom_actual_steps=None,
            custom_lora_steps=video_repair.steps,
            sampling_steps=None,
            acceleration=None,
            second_pass_acceleration=None,
            acceleration_transition_step=None,
            selflift_enabled=False,
            selflift_transition_step=None,
            upscale_enabled=False,
            upscale_resolution=None,
            upscale_target_width=None,
            upscale_target_height=None,
            preview_mode="off",
            preview_step_index=None,
            preview_fast_finish=False,
            execution_mode="complete",
            checkpoint_step=None,
            checkpoint_preview=False,
        )
        job_id = str(uuid.uuid4())
        job = JobRecord(
            id=job_id,
            spec=target_spec,
            source_job_id=source.id,
            source_video_path=source.output_path.resolve(),
            video_repair=video_repair,
            pending_action="video_repair",
            progress_detail="人脸修复已加入队列",
        )
        duration = float(
            source.spec.actual_duration_seconds
            if source.spec.output_frames is None
            else source.spec.output_frames / 24.0
        )
        windows = max(1, int(math.ceil(duration / video_repair.window_seconds)))
        # This is an ordering estimate rather than a promise. Atlas count and
        # exact H3 cost are only known after difficult-region detection.
        job.estimated_total_seconds = round(
            max(
                8.0,
                windows
                * (4.0 + video_repair.steps * video_repair.canvas_size**2 / 210_000),
            ),
            1,
        )
        job.estimated_remaining_seconds = job.estimated_total_seconds
        self.jobs[job_id] = job
        self.cancel_events[job_id] = asyncio.Event()
        self.preview_controls[job_id] = PreviewControl()
        async with self.queue_changed:
            self.pending.append(job_id)
            self.queue_changed.notify()
        self.persist(job)
        return job

    async def submit_infinite_continuation(
        self,
        *,
        source: JobRecord,
        reference_source: JobRecord,
        spec: GenerationSpec,
        continuation: InfiniteContinuationSpec,
        uploads: dict[str, tuple[str, bytes]] | None = None,
        inherit_references: bool = True,
        excluded_reference_roles: frozenset[str] = frozenset(),
    ) -> JobRecord:
        """Queue one append while keeping the accepted project history immutable."""

        if len(self.pending) >= self.max_queued_jobs:
            raise ContractError("generation queue is full; try again later")
        if source.status != "succeeded":
            raise ContractError("the current project tail must finish before appending")
        if source.final_latents_path is None or not source.final_latents_path.is_file():
            raise ContractError("the current project tail has no retained H3 latent")
        if continuation.source_job_id != source.id:
            raise ContractError("continuation source does not match the current project tail")
        if spec.frames != continuation.physical_frames:
            raise ContractError("continuation physical frame count disagrees with the request")
        if spec.width != source.spec.width or spec.height != source.spec.height:
            raise ContractError("continuation geometry must match the opening window")
        if spec.service_family != source.spec.service_family:
            raise ContractError("continuation must use the opening service family")

        job_id = str(uuid.uuid4())
        upload_dir = self.data_dir / "uploads" / job_id
        upload_dir.mkdir(parents=True, exist_ok=True)

        def clone(path: Path, role: str) -> Path:
            if not path.is_file():
                raise ContractError(f"project reference is missing: {role}")
            target = upload_dir / f"{role}{path.suffix.lower()}"
            shutil.copy2(path, target)
            return target

        uploads = uploads or {}
        first_frame = (
            clone(reference_source.first_frame, "inherited_first_frame")
            if inherit_references and reference_source.first_frame is not None
            and "first_frame" not in excluded_reference_roles
            else None
        )
        last_frame = (
            clone(reference_source.last_frame, "inherited_last_frame")
            if inherit_references and reference_source.last_frame is not None
            and "last_frame" not in excluded_reference_roles
            else None
        )
        inherited_images = [
            clone(path, f"reference_image_{index}")
            for index, path in enumerate(reference_source.reference_images, start=1)
            if f"reference_image_{index}" not in excluded_reference_roles
        ] if inherit_references else []
        inherited_audios = [
            clone(path, f"reference_audio_{index}")
            for index, path in enumerate(reference_source.reference_audios, start=1)
            if f"reference_audio_{index}" not in excluded_reference_roles
        ] if inherit_references else []
        uploaded_images: list[Path] = []
        uploaded_audios: list[Path] = []
        for role, (original_name, content) in uploads.items():
            suffix = Path(original_name).suffix.lower()
            if role.startswith("reference_audio_"):
                if suffix not in AUDIO_SUFFIXES:
                    raise ContractError(f"{role} must be WAV, MP3, FLAC, M4A, OGG or Opus")
                _validate_reference_audio(content, role)
            else:
                if suffix not in IMAGE_SUFFIXES:
                    raise ContractError(f"{role} must be PNG, JPEG or WebP")
                _validate_image(content, role)
            target = upload_dir / f"uploaded_{role}{suffix}"
            target.write_bytes(content)
            if role == "first_frame":
                first_frame = target
            elif role == "last_frame":
                last_frame = target
            elif role.startswith("reference_image_"):
                uploaded_images.append(target)
            elif role.startswith("reference_audio_"):
                uploaded_audios.append(target)
            else:
                raise ContractError(f"unsupported continuation upload: {role}")
        reference_images = tuple(inherited_images + uploaded_images)
        reference_audios = tuple(inherited_audios + uploaded_audios)
        if len(reference_images) > 9:
            raise ContractError("an infinite window accepts at most 9 reference images")
        if len(reference_audios) > 3:
            raise ContractError("an infinite window accepts at most 3 reference audios")
        job = JobRecord(
            id=job_id,
            spec=spec,
            first_frame=first_frame,
            last_frame=last_frame,
            reference_images=reference_images,
            reference_audios=reference_audios,
            source_job_id=source.id,
            source_latents_path=source.final_latents_path.resolve(),
            source_token_memory_path=(
                source.token_memory_path.resolve()
                if source.token_memory_path is not None
                and source.token_memory_path.is_file()
                else None
            ),
            infinite_continuation=continuation,
            pending_action="infinite_continuation",
            progress_detail="长视频续写已加入队列",
        )
        job.estimated_total_seconds = self._estimate_total(spec, job.condition_mode)
        job.estimated_remaining_seconds = job.estimated_total_seconds
        self.jobs[job_id] = job
        self.cancel_events[job_id] = asyncio.Event()
        self.preview_controls[job_id] = PreviewControl()
        async with self.queue_changed:
            self.pending.append(job_id)
            self.queue_changed.notify()
        self.persist(job)
        return job

    async def submit_infinite_selflift_final(
        self,
        sources: tuple[JobRecord, ...],
        *,
        output_frames: int,
        final_resolution: str | None = None,
        second_pass_acceleration: float | None = None,
        sigma_scale: float | None = None,
        temporal_window_enabled: bool | None = None,
        temporal_window_seconds: float | None = None,
        temporal_overlap_seconds: float | None = None,
    ) -> JobRecord:
        """Queue one global completion of the retained source-latent track."""

        if len(self.pending) >= self.max_queued_jobs:
            raise ContractError("generation queue is full; try again later")
        if not sources:
            raise ContractError("SelfLift final generation requires preview windows")
        for source in sources:
            if source.status != "succeeded":
                raise ContractError("every SelfLift preview window must be complete")
            if (
                not source.spec.selflift_enabled
                or source.checkpoint_path is None
                or not source.checkpoint_path.is_file()
            ):
                raise ContractError(
                    "a preview window has no retained SelfLift split checkpoint"
                )
        tail = sources[-1]
        final_acceleration = (
            float(
                tail.spec.second_pass_acceleration
                if tail.spec.second_pass_acceleration is not None
                else tail.spec.acceleration or 0.0
            )
            if second_pass_acceleration is None
            else float(second_pass_acceleration)
        )
        if not math.isfinite(final_acceleration) or not 0 <= final_acceleration <= 100:
            raise ContractError("acceleration must be between 0 and 100")
        try:
            final_sigma_scale = (
                float(tail.spec.selflift_sigma_scale)
                if sigma_scale is None
                else float(sigma_scale)
            )
        except (TypeError, ValueError) as error:
            raise ContractError("sigma_scale must be numeric") from error
        if (
            not math.isfinite(final_sigma_scale)
            or not 0.25 <= final_sigma_scale <= 1.0
        ):
            raise ContractError("sigma_scale must be between 0.25 and 1")
        final_sigma_scale = round(final_sigma_scale, 2)
        requested_target_resolution = (
            tail.spec.resolution
            if final_resolution in (None, "")
            else str(final_resolution).strip().lower()
        )
        target_resolution, target_short_edge = progressive_short_edge(
            requested_target_resolution
        )
        target_width, target_height = resolve_short_edge_geometry(
            target_short_edge, tail.spec.aspect_ratio
        )
        target_spec = dataclasses.replace(
            tail.spec,
            resolution=target_resolution,
            width=target_width,
            height=target_height,
            output_frames=int(output_frames),
            requested_duration_seconds=float(output_frames) / 24.0,
            actual_duration_seconds=float(output_frames) / 24.0,
            preview_mode="off",
            preview_step_index=None,
            preview_fast_finish=False,
            execution_mode="complete",
            checkpoint_step=None,
            checkpoint_preview=False,
            second_pass_acceleration=final_acceleration,
            selflift_sigma_scale=final_sigma_scale,
            selflift_temporal_window_enabled=(
                tail.spec.selflift_temporal_window_enabled
                if temporal_window_enabled is None
                else bool(temporal_window_enabled)
            ),
            selflift_temporal_window_seconds=(
                tail.spec.selflift_temporal_window_seconds
                if temporal_window_seconds is None
                else float(temporal_window_seconds)
            ),
            selflift_temporal_overlap_seconds=(
                tail.spec.selflift_temporal_overlap_seconds
                if temporal_overlap_seconds is None
                else float(temporal_overlap_seconds)
            ),
        )
        job_id = str(uuid.uuid4())
        job = JobRecord(
            id=job_id,
            spec=target_spec,
            source_job_id=tail.id,
            infinite_final_window_ids=tuple(source.id for source in sources),
            pending_action="infinite_selflift_final",
            progress_detail="SelfLift全片重叠滑窗终采已加入队列",
        )
        # Calibrated from real RTX 4090 global-track runs at 141 and 481
        # frames. Cost follows the overlapping high-resolution view work plus
        # one whole-timeline learned lift, VAE decode and mux. The former
        # per-window estimate omitted the latter and reached zero while a
        # 20-second final still had several minutes left.
        from .native_engine.global_co_denoise import (
            plan_global_av_windows,
            window_geometry_for_seconds,
        )

        if target_spec.selflift_temporal_window_enabled:
            window_frames, stride_frames = window_geometry_for_seconds(
                target_spec.selflift_temporal_window_seconds,
                target_spec.selflift_temporal_overlap_seconds,
            )
            global_plan = plan_global_av_windows(
                int(output_frames),
                window_frames=window_frames,
                stride_frames=stride_frames,
            )
        else:
            global_plan = plan_global_av_windows(int(output_frames))
        remaining_steps = max(
            1,
            int(target_spec.sampling_steps or target_spec.preset["steps"])
            - int(target_spec.selflift_transition_step or 0),
        )
        target_pixel_scale = (
            float(target_spec.width * target_spec.height) / float(1920 * 1088)
        )
        high_resolution_frame_steps = remaining_steps * sum(
            item.frames for item in global_plan.windows
        )
        job.estimated_total_seconds = round(
            max(
                20.0,
                12.0
                + target_pixel_scale
                * (
                    0.20 * high_resolution_frame_steps
                    + 0.31 * int(output_frames)
                ),
            ),
            1,
        )
        job.estimated_remaining_seconds = job.estimated_total_seconds
        self.jobs[job_id] = job
        self.cancel_events[job_id] = asyncio.Event()
        self.preview_controls[job_id] = PreviewControl()
        async with self.queue_changed:
            self.pending.append(job_id)
            self.queue_changed.notify()
        self.persist(job)
        return job

    def _estimate_total(self, spec: GenerationSpec, condition_mode: str) -> float:
        execution = spec.preset
        actual_steps = execution.get("actual_steps")
        lora_steps = execution.get("steps")
        matches = [
            job.elapsed_seconds
            for job in sorted(self.jobs.values(), key=lambda item: item.created_at)
            if job.status == "succeeded"
            and job.elapsed_seconds is not None
            and job.spec.engine == spec.engine
            and job.spec.width == spec.width
            and job.spec.height == spec.height
            and job.spec.frames == spec.frames
            and job.spec.output_frames == spec.output_frames
            and job.condition_mode == condition_mode
            and job.spec.preset.get("actual_steps") == actual_steps
            and job.spec.preset.get("steps") == lora_steps
            and job.spec.sampling_steps == spec.sampling_steps
            and job.spec.acceleration == spec.acceleration
            and job.spec.second_pass_acceleration == spec.second_pass_acceleration
            and job.spec.acceleration_transition_step == spec.acceleration_transition_step
            and job.spec.attention_keep_ratio == spec.attention_keep_ratio
            and job.spec.sparse_scope == spec.sparse_scope
            and job.spec.upscale_enabled == spec.upscale_enabled
            and job.spec.upscale_target_width == spec.upscale_target_width
            and job.spec.upscale_target_height == spec.upscale_target_height
        ]
        if matches:
            recent = matches[-5:]
            estimate = sum(recent) / len(recent)
            return round(estimate, 1)
        compute_frames = spec.frames
        if spec.output_frames is not None and spec.output_frames > spec.frames:
            from .native_engine.long_horizon import plan_long_horizon

            long_plan = plan_long_horizon(
                requested_duration_seconds=spec.requested_duration_seconds,
                prompt=spec.prompt,
                seed=spec.seed,
                maximum_opening_frames=spec.frames,
            )
            # Include the clean continuity prefix in every continuation;
            # output duration alone would under-estimate actual DiT work.
            compute_frames = sum(
                segment.window_frames for segment in long_plan.segments
            )
        megapixel_frames = spec.width * spec.height * compute_frames / 1_000_000
        if spec.joint_acceleration_enabled:
            from .native_engine.planner import H3JointAccelerationScheduler

            joint = H3JointAccelerationScheduler().plan(
                int(spec.sampling_steps or 20),
                float(spec.acceleration or 0.0),
                allow_forecast=spec.model_variant == "base",
            )
            # Normalized compute units include full evaluations, shallow
            # forecasts and measured Attention action ratios.  This remains a
            # cold estimate; exact matching completed-job history wins above.
            denominator = 6.0 if spec.model_variant == "lora" else 9.0
            slope = 0.32 if spec.model_variant == "lora" else 0.37
            estimate = 12 + megapixel_frames * slope * (
                joint.estimated_compute_units / denominator
            )
        elif spec.engine in ("lora", "reference_lora"):
            step_factor = int(spec.preset["steps"]) / 6
            estimate = 12 + megapixel_frames * 0.32 * step_factor
        else:
            actual = int(spec.preset["actual_steps"])
            forecast = int(spec.preset["forecast_steps"])
            estimate = 16 + megapixel_frames * (0.37 * actual + 0.025 * forecast) / 9
        if condition_mode != "text":
            # Ref2VA additionally runs Qwen3-VL over the reference images and
            # Video-VAE encodes them.  On the compact memory profile Qwen is
            # intentionally read per request.  The first measured 480p/3s
            # full-schedule job spent about 108 s beyond the dense DiT/VAE
            # estimate; matching completed-job history supersedes this cold
            # fallback from the second identical workload onward.
            estimate += 4.0 if condition_mode != "reference" else 108.0
        if not spec.joint_acceleration_enabled and spec.attention_keep_ratio < 1.0:
            # Sparse attention only affects part of the DiT cost. Scope is the
            # fraction of sampling steps on which the approximation is active;
            # measured history supersedes this conservative cold estimate.
            scope_fraction = {
                "middle_only": 0.50,
                "guarded": 0.80,
                "full": 1.00,
            }[spec.sparse_scope]
            estimate *= 1.0 - (1.0 - spec.attention_keep_ratio) * scope_fraction * 0.30
        if spec.upscale_enabled:
            # Until this installation has matching history, use a conservative
            # postprocess estimate. The next completed task calibrates it.
            output_pixels = (
                int(spec.upscale_target_width or spec.width)
                * int(spec.upscale_target_height or spec.height)
            )
            estimate += 28.0 + output_pixels * compute_frames / 1_000_000 * 0.42
        return round(max(8.0, estimate), 1)

    async def cancel(self, job_id: str) -> JobRecord:
        job = self.jobs[job_id]
        if job.status in {"succeeded", "failed", "cancelled", "checkpointed"}:
            return job
        self.cancel_events[job_id].set()
        control = self.preview_controls.get(job_id)
        if control is not None and not control.event.is_set():
            control.decision = "discard"
            control.event.set()
        if job.status == "queued":
            job.status = "cancelled"
            job.updated_at = time.time()
            job.error = "cancelled before execution"
            async with self.queue_changed:
                if job_id in self.pending:
                    self.pending.remove(job_id)
                self.queue_changed.notify()
            self.persist(job)
        else:
            # CUDA kernels are not safely pre-emptible, but the native DiT
            # checks this event at every layer boundary. Reflect acceptance
            # immediately instead of leaving the UI looking unresponsive.
            job.updated_at = time.time()
            job.progress_stage = "cancelling"
            job.progress_detail = "正在取消；等待当前 GPU 算子结束"
            job.estimated_remaining_seconds = None
            self.persist(job)
        return job

    async def reorder(self, ordered_ids: list[str]) -> None:
        async with self.queue_changed:
            if len(ordered_ids) != len(set(ordered_ids)):
                raise ContractError("queue order contains duplicate job ids")
            if set(ordered_ids) != set(self.pending):
                raise ContractError("queue order must contain every queued job exactly once")
            self.pending[:] = ordered_ids
            self.queue_changed.notify()

    async def resume(self, job_id: str) -> JobRecord:
        job = self.jobs[job_id]
        # A failed resume does not mutate or consume the retained formal
        # checkpoint.  Keep it retryable after a backend fix/restart instead
        # of forcing the user to recompute the expensive prefix.
        if job.status not in {"checkpointed", "failed"}:
            raise ContractError("job is not stopped at a checkpoint")
        if (
            not job.checkpoint_retained
            or job.checkpoint_path is None
            or not job.checkpoint_path.is_file()
        ):
            raise ContractError("the checkpoint was not retained")
        if len(self.pending) >= self.max_queued_jobs:
            raise ContractError("generation queue is full; try again later")
        job.pending_action = "resume"
        job.status = "queued"
        job.error = None
        job.progress_stage = "queued_resume"
        job.progress_detail = "断点恢复已加入队列"
        job.updated_at = time.time()
        self.cancel_events[job_id] = asyncio.Event()
        async with self.queue_changed:
            self.pending.append(job_id)
            self.queue_changed.notify()
        self.persist(job)
        return job

    async def clear_latent_cache(self) -> dict[str, int]:
        """Remove only reproducible AV latent artifacts from this workspace.

        Videos, uploads and job history remain untouched.  Refuse while work
        is active so a queued or running second pass can never lose its source
        tensor mid-flight.  Formal checkpoint tensors are included because
        they back the console's disposable checkpoint-preview/resume path.
        """

        if any(
            job.status in {
                "queued", "starting_backend", "running", "awaiting_preview"
            }
            for job in self.jobs.values()
        ):
            raise ContractError(
                "wait for or cancel active jobs before clearing latent cache"
            )
        if self.output_root is None:
            raise ContractError("this workspace has no managed output root")
        latent_root = (self.output_root / ".h3-latents").resolve()
        latent_root.mkdir(parents=True, exist_ok=True)
        checkpoint_root = (self.data_dir / "checkpoints").resolve()
        checkpoint_root.mkdir(parents=True, exist_ok=True)
        removed_files = 0
        removed_bytes = 0
        removed_checkpoint_files = 0
        for root, is_checkpoint in (
            (latent_root, False),
            (checkpoint_root, True),
        ):
            for path in tuple(root.glob("*.pt")):
                candidate = path.resolve()
                if not candidate.is_relative_to(root) or not candidate.is_file():
                    continue
                removed_bytes += candidate.stat().st_size
                candidate.unlink()
                removed_files += 1
                removed_checkpoint_files += int(is_checkpoint)
        affected_jobs = 0
        for job in self.jobs.values():
            changed = False
            if job.final_latents_path is not None:
                job.final_latents_path = None
                changed = True
            if job.source_latents_path is not None:
                job.source_latents_path = None
                changed = True
            if job.token_memory_path is not None:
                job.token_memory_path = None
                changed = True
            if job.source_token_memory_path is not None:
                job.source_token_memory_path = None
                changed = True
            if job.checkpoint_path is not None:
                job.checkpoint_path = None
                job.checkpoint_retained = False
                changed = True
            if changed:
                affected_jobs += 1
                self.persist(job)
        return {
            "removed_files": removed_files,
            "removed_checkpoint_files": removed_checkpoint_files,
            "removed_bytes": removed_bytes,
            "affected_jobs": affected_jobs,
        }

    async def delete(self, job_id: str) -> dict[str, bool]:
        """Delete a job and any output owned by the configured output root.

        Persisted development jobs can legitimately point at an older smoke or
        calibration directory.  Those paths are not trusted deletion targets:
        remove the service record and uploads, but retain the external file.
        This keeps the path-containment guard without making legacy cards
        impossible to dismiss from the UI.
        """
        job = self.jobs[job_id]
        if job.status in {"starting_backend", "running", "awaiting_preview"}:
            raise ContractError("cancel the running job before deleting it")
        if job.status == "queued":
            await self.cancel(job_id)
        if any(
            child.source_job_id == job.id
            and child.status in {"queued", "starting_backend", "running"}
            for child in self.jobs.values()
        ):
            raise ContractError(
                "wait for the active child task before deleting its source"
            )
        resolved_output = None
        resolved_preview = None
        resolved_checkpoint = None
        resolved_latents = None
        resolved_token_memory = None
        output_retained = False
        if job.output_path is not None and job.output_path.is_file():
            candidate = job.output_path.resolve()
            if self.output_root is not None and candidate.is_relative_to(
                self.output_root
            ):
                resolved_output = candidate
            else:
                output_retained = True
        if job.preview_path is not None and job.preview_path.is_file():
            candidate_preview = job.preview_path.resolve()
            if self.output_root is not None and candidate_preview.is_relative_to(
                self.output_root
            ):
                resolved_preview = candidate_preview
        if job.checkpoint_path is not None and job.checkpoint_path.is_file():
            candidate_checkpoint = job.checkpoint_path.resolve()
            checkpoint_root = (self.data_dir / "checkpoints").resolve()
            if candidate_checkpoint.is_relative_to(checkpoint_root):
                resolved_checkpoint = candidate_checkpoint
        if job.final_latents_path is not None and job.final_latents_path.is_file():
            candidate_latents = job.final_latents_path.resolve()
            if self.output_root is not None and candidate_latents.is_relative_to(
                (self.output_root / ".h3-latents").resolve()
            ):
                resolved_latents = candidate_latents
        if job.token_memory_path is not None and job.token_memory_path.is_file():
            candidate_memory = job.token_memory_path.resolve()
            if self.output_root is not None and candidate_memory.is_relative_to(
                (self.output_root / ".h3-latents").resolve()
            ):
                resolved_token_memory = candidate_memory
        async with self.queue_changed:
            if job_id in self.pending:
                self.pending.remove(job_id)
        self.cancel_events.pop(job_id, None)
        self.preview_controls.pop(job_id, None)
        self.jobs.pop(job_id, None)
        (self.data_dir / "jobs" / f"{job_id}.json").unlink(missing_ok=True)
        shutil.rmtree(self.data_dir / "uploads" / job_id, ignore_errors=True)
        if resolved_output is not None:
            resolved_output.unlink()
        if resolved_preview is not None and resolved_preview != resolved_output:
            resolved_preview.unlink()
        if resolved_checkpoint is not None:
            resolved_checkpoint.unlink()
        if resolved_latents is not None:
            resolved_latents.unlink()
        if resolved_token_memory is not None:
            resolved_token_memory.unlink()
        return {
            "output_deleted": resolved_output is not None,
            "preview_deleted": resolved_preview is not None,
            "output_retained": output_retained,
            "checkpoint_deleted": resolved_checkpoint is not None,
            "latents_deleted": resolved_latents is not None,
            "token_memory_deleted": resolved_token_memory is not None,
        }

    async def decide_preview(self, job_id: str, decision: str) -> JobRecord:
        if decision not in {"continue", "discard"}:
            raise ContractError("preview decision must be continue or discard")
        job = self.jobs[job_id]
        if job.status != "awaiting_preview":
            raise ContractError("job is not waiting for a preview decision")
        control = self.preview_controls[job_id]
        if control.event.is_set():
            raise ContractError("preview decision was already submitted")
        control.decision = decision
        job.preview_decision = decision
        if decision == "continue":
            job.status = "running"
            job.progress_stage = "resuming"
            job.progress_detail = "继续正式轨迹"
        else:
            self.cancel_events[job_id].set()
            job.progress_detail = "放弃当前抽卡"
        job.updated_at = time.time()
        self.persist(job)
        control.event.set()
        return job

    def _progress_callback(self, job: JobRecord, *, scale: float = 1.0, offset: float = 0.0):
        loop = asyncio.get_running_loop()
        last_persisted = [0.0]

        def update(event: dict[str, Any]) -> None:
            def apply() -> None:
                if job.status in {"succeeded", "failed", "cancelled"}:
                    return
                mapped = offset + float(event["percent"]) * scale
                percent = min(99.0, max(job.progress_percent, mapped))
                job.progress_percent = percent
                job.progress_stage = str(event.get("stage", job.progress_stage))
                job.progress_detail = str(event.get("detail", job.progress_detail))
                now = time.time()
                elapsed = max(0.0, now - (job.started_at or now))
                # Pipeline percentages are stage-weighted, not linear time.
                # Keep the calibrated/history estimate stable. If reality has
                # already overtaken it, extend gently instead of jumping by
                # elapsed/percent (the source of the old ETA bug).
                prior = job.estimated_total_seconds
                if prior is None:
                    prior = self._estimate_total(job.spec, job.condition_mode)
                if elapsed >= prior * 0.95:
                    prior = elapsed + max(5.0, prior * 0.08)
                job.estimated_total_seconds = round(prior, 1)
                job.estimated_remaining_seconds = round(max(0.0, prior - elapsed), 1)
                job.updated_at = now
                if now - last_persisted[0] >= 1.0:
                    self.persist(job)
                    last_persisted[0] = now

            loop.call_soon_threadsafe(apply)

        return update

    def _update(self, job: JobRecord, status: str) -> None:
        job.status = status
        job.updated_at = time.time()
        self.persist(job)

    async def worker(self) -> None:
        while True:
            async with self.queue_changed:
                await self.queue_changed.wait_for(lambda: bool(self.pending))
                job_id = self.pending.pop(0)
            job = self.jobs.get(job_id)
            if job is None or job.status == "cancelled":
                continue
            await self._run(job)

    async def _run(self, job: JobRecord) -> None:
        # Persisted jobs and direct boundary tests created before fork preview
        # do not have an associated control object.  Lazily creating it keeps
        # those records executable while new submissions still allocate it at
        # submit time.
        cancel_event = self.cancel_events.setdefault(job.id, asyncio.Event())
        preview_control = self.preview_controls.setdefault(job.id, PreviewControl())
        action = job.pending_action
        try:
            job.progress_percent = 1.0
            job.progress_stage = "preparing"
            job.progress_detail = "等待模型预加载"
            self._update(job, "starting_backend")
            if self.warmup_task is not None and not self.warmup_task.done():
                await self.warmup_task
            if cancel_event.is_set():
                raise JobCancelled("cancelled before generation")
            job.started_at = time.time()
            job.progress_detail = "准备热会话"
            job.backend_prompt_id = job.id
            self._update(job, "running")
            loop = asyncio.get_running_loop()

            def preview_ready(event: dict[str, Any]) -> None:
                def apply() -> None:
                    job.preview_path = Path(str(event["output_path"])).resolve()
                    job.progress_percent = max(job.progress_percent, 60.0)
                    job.progress_stage = "preview_ready"
                    if job.spec.preview_mode == "pause":
                        job.status = "awaiting_preview"
                        job.progress_detail = "预览已就绪：继续正式生成或放弃本次抽卡"
                    else:
                        job.progress_detail = "分叉预览已保存，正式轨迹继续运行"
                    job.updated_at = time.time()
                    self.persist(job)
                loop.call_soon_threadsafe(apply)

            def wait_preview_decision() -> str:
                while not preview_control.event.wait(0.25):
                    if cancel_event.is_set():
                        return "discard"
                return preview_control.decision or "discard"

            progress_callback = self._progress_callback(
                job, scale=0.84 if job.spec.upscale_enabled else 1.0
            )
            is_infinite_continuation = (
                job.infinite_continuation is not None
                and action in {"infinite_continuation", "resume"}
            )
            if action == "infinite_selflift_final":
                sources = tuple(
                    self.jobs[source_id]
                    for source_id in job.infinite_final_window_ids
                    if source_id in self.jobs
                )
                if len(sources) != len(job.infinite_final_window_ids):
                    raise RuntimeError(
                        "a SelfLift project window is missing from job history"
                    )
                complete_selflift = getattr(
                    self.backend, "complete_infinite_selflift", None
                )
                if not callable(complete_selflift):
                    raise RuntimeError(
                        "the active backend does not support SelfLift project completion"
                    )
                job.progress_stage = "infinite_selflift_final"
                job.progress_detail = "正在沿全片 latent 轨道重叠滑窗终采"
                self.persist(job)
                result = await complete_selflift(
                    sources,
                    job.id,
                    cancel_event,
                    progress_callback,
                    final_spec=job.spec,
                )
            elif is_infinite_continuation:
                if (
                    job.infinite_continuation is None
                    or job.source_latents_path is None
                ):
                    raise RuntimeError("infinite continuation is missing its source contract")
                continue_generate = getattr(self.backend, "continue_generate", None)
                if not callable(continue_generate):
                    raise RuntimeError("the active backend does not support infinite continuation")
                job.progress_stage = "infinite_continuation"
                job.progress_detail = "正在续写并增量更新低分辨率预览"
                self.persist(job)
                preview_kwargs: dict[str, Any] = {}
                if job.spec.preview_mode != "off":
                    preview_kwargs["preview_ready_callback"] = preview_ready
                    preview_kwargs["preview_decision_wait"] = (
                        wait_preview_decision
                        if job.spec.preview_mode == "pause" else None
                    )
                if job.spec.execution_mode == "checkpoint":
                    if action == "resume":
                        preview_kwargs["resume_checkpoint_path"] = job.checkpoint_path
                    else:
                        preview_kwargs["checkpoint_path"] = (
                            self.data_dir / "checkpoints" / f"{job.id}.pt"
                        )
                source_job = (
                    self.jobs.get(job.source_job_id)
                    if job.source_job_id is not None
                    else None
                )
                preview_kwargs["source_video_path"] = (
                    source_job.output_path
                    if source_job is not None
                    and source_job.output_path is not None
                    and source_job.output_path.is_file()
                    else None
                )
                preview_kwargs["first_frame"] = job.first_frame
                preview_kwargs["last_frame"] = job.last_frame
                result = await continue_generate(
                    job.spec,
                    job.infinite_continuation,
                    job.source_latents_path,
                    job.source_token_memory_path,
                    job.id,
                    job.reference_images,
                    job.reference_audios,
                    cancel_event,
                    progress_callback,
                    **preview_kwargs,
                )
            elif action == "video_repair":
                if job.video_repair is None:
                    raise RuntimeError("video-repair job is missing its repair contract")
                source_video = job.source_video_path
                if source_video is None and job.source_job_id is not None:
                    source_job = self.jobs.get(job.source_job_id)
                    source_video = (
                        source_job.output_path if source_job is not None else None
                    )
                if source_video is None or not source_video.is_file():
                    raise RuntimeError("video-repair source video is missing")
                repair_video = getattr(self.backend, "video_repair", None)
                if not callable(repair_video):
                    raise RuntimeError(
                        "the active backend does not support video repair"
                    )
                job.progress_stage = "video_repair"
                job.progress_detail = "正在检测并修复视频难区"
                self.persist(job)
                result = await repair_video(
                    job.spec,
                    job.video_repair,
                    source_video,
                    job.id,
                    cancel_event,
                    progress_callback,
                )
            elif action == "second_sampling":
                if job.second_sampling is None:
                    raise RuntimeError("second-sampling job is missing its source contract")
                if job.second_sampling.method == "temporal":
                    source_video = job.source_video_path
                    if source_video is None and job.source_job_id is not None:
                        source_job = self.jobs.get(job.source_job_id)
                        source_video = (
                            source_job.output_path if source_job is not None else None
                        )
                    if source_video is None or not source_video.is_file():
                        raise RuntimeError(
                            "temporal second sampling source video is missing"
                        )
                    if self.upscaler is None:
                        raise RuntimeError(
                            "the temporal second-sampling model is not configured"
                        )
                    upscaler_status = self.upscaler.status()
                    if not upscaler_status.get("ready", False):
                        missing = upscaler_status.get("missing") or []
                        raise RuntimeError(
                            "the temporal second-sampling runtime is incomplete"
                            + (f": {missing[0]}" if missing else "")
                        )
                    job.progress_stage = "temporal_second_sampling"
                    job.progress_detail = "正在执行一步时序模型二次采样"
                    self.persist(job)
                    target_output = (
                        (self.output_root or source_video.parent) / f"{job.id}.mp4"
                    )
                    upscale = await self.upscaler.upscale(
                        source_video,
                        target_width=job.second_sampling.width,
                        target_height=job.second_sampling.height,
                        cancel_event=cancel_event,
                        progress_callback=progress_callback,
                        output_path=target_output,
                    )
                    job.upscale_peak_allocated_mib = upscale.peak_allocated_mib
                    job.upscale_peak_reserved_mib = upscale.peak_reserved_mib
                    result = GenerationResult(
                        runtime_key="flashvsr_v1.1_tiny_long_1step",
                        elapsed_seconds=upscale.elapsed_seconds,
                        output_path=upscale.output_path,
                        inference_plan={
                            "second_sampling_method": {
                                "method": "temporal",
                                "implementation": "FlashVSR v1.1 Tiny Long",
                                "inference_steps": 1,
                                "source": "completed H3 video",
                                "target_width": job.second_sampling.width,
                                "target_height": job.second_sampling.height,
                                "timings": dict(upscale.timings or {}),
                            }
                        },
                        final_latents_path=None,
                        stage_seconds={
                            f"temporal_second_sampling.{name}": seconds
                            for name, seconds in (upscale.timings or {}).items()
                        },
                    )
                else:
                    if job.source_latents_path is None:
                        raise RuntimeError(
                            "H3 second-sampling job is missing its source latent"
                        )
                    second_sample = getattr(self.backend, "second_sample", None)
                    if not callable(second_sample):
                        raise RuntimeError(
                            "the active backend does not support H3 second sampling"
                        )
                    job.progress_stage = "second_sampling"
                    job.progress_detail = "正在执行H3高分辨率二次采样"
                    self.persist(job)
                    result = await second_sample(
                        job.spec,
                        job.second_sampling,
                        job.source_latents_path,
                        job.id,
                        job.first_frame,
                        job.last_frame,
                        job.reference_images,
                        job.reference_videos,
                        job.reference_audios,
                        cancel_event,
                        progress_callback,
                    )
            else:
                generate_args = (
                    job.spec,
                    job.id,
                    job.first_frame,
                    job.last_frame,
                    job.reference_images,
                    job.reference_videos,
                    job.reference_audios,
                    cancel_event,
                    progress_callback,
                )
                # Do not force legacy/testing backend adapters to understand the
                # preview callbacks when the feature is disabled.
                preview_kwargs: dict[str, Any] = {}
                if job.spec.preview_mode != "off":
                    preview_kwargs["preview_ready_callback"] = preview_ready
                    preview_kwargs["preview_decision_wait"] = (
                        wait_preview_decision
                        if job.spec.preview_mode == "pause" else None
                    )
                if job.spec.execution_mode == "checkpoint":
                    if action == "resume":
                        preview_kwargs["resume_checkpoint_path"] = job.checkpoint_path
                    else:
                        preview_kwargs["checkpoint_path"] = (
                            self.data_dir / "checkpoints" / f"{job.id}.pt"
                        )
                result = await self.backend.generate(*generate_args, **preview_kwargs)
            if isinstance(result, CheckpointResult):
                job.inference_plan = result.inference_plan
                job.stage_seconds = dict(
                    getattr(result, "stage_seconds", {}) or {}
                )
                job.runtime_key = result.runtime_key
                job.generation_elapsed_seconds = round(
                    float(job.generation_elapsed_seconds or 0.0)
                    + float(result.elapsed_seconds),
                    3,
                )
                job.elapsed_seconds = job.generation_elapsed_seconds
                job.checkpoint_completed_steps = result.completed_steps
                job.checkpoint_total_steps = result.total_steps
                job.preview_path = result.preview_path
                job.checkpoint_path = result.checkpoint_path
                job.checkpoint_retained = bool(
                    job.spec.checkpoint_retain
                    and result.checkpoint_path is not None
                    and result.checkpoint_path.is_file()
                )
                if not job.spec.checkpoint_retain and result.checkpoint_path is not None:
                    result.checkpoint_path.unlink(missing_ok=True)
                    job.checkpoint_path = None
                has_selflift_source_latent = bool(
                    result.preview_latents_path is not None
                    and result.preview_latents_path.is_file()
                    and job.spec.selflift_enabled
                )
                has_decoded_selflift_preview = bool(
                    result.preview_path is not None
                    and result.preview_path.is_file()
                )
                if has_selflift_source_latent and (
                    has_decoded_selflift_preview
                    or not job.spec.checkpoint_preview
                ):
                    # Long-video creation publishes only the low-resolution
                    # preview and retains its clean source latent. The project
                    # finalizer assembles that connected source track and owns
                    # the single global learned lift and high-resolution tail.
                    if has_decoded_selflift_preview:
                        job.output_path = result.preview_path
                    job.final_latents_path = result.preview_latents_path
                    job.token_memory_path = result.token_memory_path
                    job.pending_action = "generate"
                    job.progress_percent = 100.0
                    job.progress_stage = (
                        "selflift_preview_ready"
                        if has_decoded_selflift_preview
                        else "selflift_source_ready"
                    )
                    job.progress_detail = (
                        "低清预览与全片定稿状态已就绪"
                        if has_decoded_selflift_preview
                        else "低分辨率 latent 与全片定稿状态已就绪"
                    )
                    job.estimated_remaining_seconds = 0.0
                    self._update(job, "succeeded")
                    return
                job.pending_action = (
                    "infinite_continuation"
                    if job.infinite_continuation is not None
                    else "generate"
                )
                job.progress_percent = round(
                    100.0 * result.completed_steps / result.total_steps, 1
                )
                job.progress_stage = "checkpointed"
                job.progress_detail = (
                    f"已在第 {result.completed_steps}/{result.total_steps} 步停止"
                )
                job.estimated_remaining_seconds = None
                self._update(job, "checkpointed")
                return
            job.runtime_key = result.runtime_key
            # Keep the service boundary compatible with backend adapters that
            # predate scheduler telemetry.  Production native results expose
            # the field, while a minimal backend may legitimately omit it.
            job.inference_plan = getattr(result, "inference_plan", None)
            job.stage_seconds = dict(
                getattr(result, "stage_seconds", {}) or {}
            )
            job.generation_elapsed_seconds = round(
                float(job.generation_elapsed_seconds or 0.0)
                + float(result.elapsed_seconds),
                3,
            )
            job.output_path = result.output_path
            job.final_latents_path = getattr(result, "final_latents_path", None)
            job.token_memory_path = getattr(result, "token_memory_path", None)
            if job.spec.upscale_enabled:
                if self.upscaler is None:
                    raise RuntimeError("FlashVSR upscaler is not configured")
                job.progress_percent = max(job.progress_percent, 84.0)
                job.progress_stage = "upscaling"
                job.progress_detail = "H3完成，正在加载FlashVSR"
                self.persist(job)
                profile = (
                    self.memory_profile_getter()
                    if callable(self.memory_profile_getter)
                    else None
                )
                exclusive = bool(
                    profile is not None and profile.exclusive_upscaler
                )
                upscale_cycle_started = time.monotonic()
                if exclusive:
                    job.progress_detail = "释放H3内存，准备独占超分"
                    self.persist(job)
                    await self.backend.stop()
                try:
                    upscale = await self.upscaler.upscale(
                        job.output_path,
                        target_width=int(job.spec.upscale_target_width),
                        target_height=int(job.spec.upscale_target_height),
                        cancel_event=cancel_event,
                        progress_callback=self._progress_callback(
                            job, scale=0.15, offset=84.0
                        ),
                    )
                finally:
                    if exclusive:
                        job.progress_detail = "超分完成，正在恢复H3热态"
                        self.persist(job)
                        await self.upscaler.stop()
                        await self.backend.preload(job.spec.runtime_launcher)
                        if self.backend.warm_state.get("status") != "ready":
                            raise RuntimeError("H3 failed to recover after exclusive upscaling")
                job.upscale_elapsed_seconds = (
                    time.monotonic() - upscale_cycle_started
                    if exclusive else upscale.elapsed_seconds
                )
                job.upscale_peak_allocated_mib = upscale.peak_allocated_mib
                job.upscale_peak_reserved_mib = upscale.peak_reserved_mib
                job.output_path = upscale.output_path
                if result.output_path != upscale.output_path:
                    result.output_path.unlink(missing_ok=True)
            job.elapsed_seconds = (
                float(job.generation_elapsed_seconds or 0.0)
                + float(job.upscale_elapsed_seconds or 0.0)
            )
            job.progress_percent = 100.0
            job.progress_stage = "completed"
            job.progress_detail = (
                "人脸修复完成" if action == "video_repair" else "视频生成完成"
            )
            job.estimated_remaining_seconds = 0.0
            job.pending_action = "generate"
            self._update(job, "succeeded")
        except JobCancelled as error:
            job.error = str(error)
            job.progress_stage = "cancelled"
            job.progress_detail = "任务已取消"
            self._update(job, "cancelled")
        except Exception:
            detail = self.data_dir / "logs" / f"job_{job.id}.error.log"
            detail.write_text(traceback.format_exc(), encoding="utf-8")
            job.error = f"generation failed (reference {job.id[:8]})"
            job.progress_stage = "failed"
            job.progress_detail = "生成失败"
            self._update(job, "failed")

    async def start(self, fixed_engine: str | None = None, *, preload: bool = True) -> None:
        self.fixed_engine = fixed_engine
        self.worker_task = asyncio.create_task(self.worker(), name="h3serve-worker")
        if preload and fixed_engine is not None and hasattr(self.backend, "preload"):
            self.warmup_task = asyncio.create_task(
                self.backend.preload(fixed_engine), name="h3serve-model-preload"
            )

    async def close(self) -> None:
        if self.warmup_task is not None and not self.warmup_task.done():
            self.warmup_task.cancel()
            try:
                await self.warmup_task
            except asyncio.CancelledError:
                pass
        if self.worker_task is not None:
            self.worker_task.cancel()
            try:
                await self.worker_task
            except asyncio.CancelledError:
                pass
        if self.upscaler is not None and hasattr(self.upscaler, "stop"):
            await self.upscaler.stop()
        await self.backend.stop()


async def _read_generation_request(
    request: web.Request,
) -> tuple[dict[str, Any], dict[str, tuple[str, bytes]]]:
    uploads: dict[str, tuple[str, bytes]] = {}
    if request.content_type == "application/json":
        payload = await request.json()
        if not isinstance(payload, dict):
            raise ContractError("JSON body must be an object")
        return payload, uploads

    if not request.content_type.startswith("multipart/"):
        raise ContractError("use application/json or multipart/form-data")
    payload: dict[str, Any] = {}
    reader = await request.multipart()
    async for part in reader:
        is_reference_image = bool(
            part.name and re.fullmatch(r"reference_image_[1-9]", part.name)
        )
        is_reference_video = bool(
            part.name and re.fullmatch(r"reference_video_[1-3]", part.name)
        )
        is_reference_audio = bool(
            part.name and re.fullmatch(r"reference_audio_[1-3]", part.name)
        )
        if part.name in {"first_frame", "last_frame"} or is_reference_image or is_reference_video or is_reference_audio:
            if not part.filename:
                continue
            content = bytearray()
            while True:
                chunk = await part.read_chunk(1024 * 1024)
                if not chunk:
                    break
                content.extend(chunk)
                limit = MAX_REFERENCE_VIDEO_BYTES if is_reference_video else MAX_IMAGE_BYTES
                if len(content) > limit:
                    raise ContractError(f"{part.name} exceeds {limit // (1024 * 1024)} MiB")
            uploads[part.name] = (part.filename, bytes(content))
        elif part.filename:
            raise ContractError(f"unsupported file field: {part.name}")
        else:
            payload[str(part.name)] = await part.text()
    return payload, uploads


def create_app(
    *,
    paths: ServicePaths,
    serve_dir: Path,
    api_key: str | None = None,
    backend: Any | None = None,
    max_queued_jobs: int = 32,
    fixed_engine: str | None = "original",
    preload: bool = True,
    upscaler: Any | None = None,
    memory_profile: HostMemoryProfile | None = None,
    memory_budget_controller: Any | None = None,
    host_memory_status: HostMemoryStatus | None = None,
) -> web.Application:
    fixed_launcher = (
        None if fixed_engine is None else normalize_launcher(str(fixed_engine))
    )
    default_variant = (
        engine_variant(str(fixed_engine)) if fixed_engine in ENGINES else "base"
    )
    fixed_engine = (
        None if fixed_launcher is None else launcher_family(fixed_launcher)
    )
    workspace_controller = (
        WorkspaceController(paths.release_root) if fixed_engine is None else None
    )
    runtime_paths = (
        dataclasses.replace(
            paths,
            data_dir=workspace_controller.current.data_dir,
            output_dir=workspace_controller.current.output_dir,
        )
        if workspace_controller is not None else paths
    )
    @web.middleware
    async def api_auth(request: web.Request, handler):
        if api_key and request.path.startswith("/api/v1"):
            bearer = request.headers.get("Authorization", "")
            supplied = request.headers.get("X-API-Key")
            if bearer.startswith("Bearer "):
                supplied = bearer[7:]
            if supplied != api_key:
                raise web.HTTPUnauthorized(text="missing or invalid API key")
        elif request.method not in {"GET", "HEAD", "OPTIONS"}:
            origin = request.headers.get("Origin")
            expected = f"{request.scheme}://{request.host}"
            if origin and origin.rstrip("/") != expected.rstrip("/"):
                raise web.HTTPForbidden(text="cross-origin state changes are not allowed")
        response = await handler(request)
        if request.path == "/" or request.path.startswith("/static/"):
            # index.html and its assets are one protocol surface.  A stale
            # app.js can silently submit the retired preview contract even
            # while the Python service has already been upgraded.
            response.headers["Cache-Control"] = (
                "no-store, no-cache, must-revalidate, max-age=0"
            )
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
        return response

    app = web.Application(middlewares=[api_auth], client_max_size=640 * 1024 * 1024)
    memory_profile = memory_profile or HOST_MEMORY_PROFILES["fullspeed"]
    if fixed_launcher is not None:
        validate_profile_for_weight_tier(
            memory_profile,
            launcher_weight_tier(fixed_launcher),
        )
    host_capacity_status = host_memory_status or detect_host_memory()
    budget_controller = memory_budget_controller or InMemoryBudgetController()
    memory_state = {
        "profile": memory_profile,
        "changing": False,
        "capacity": host_capacity_status,
        "controller": budget_controller,
    }
    engine_state: dict[str, Any] = {
        "active": fixed_engine,
        "launcher": fixed_launcher,
        "weight_tier": (
            None
            if fixed_launcher is None
            else launcher_weight_tier(fixed_launcher)
        ),
        "vram_profile": (
            None
            if fixed_launcher is None
            else launcher_vram_profile(fixed_launcher)
        ),
        "switching": False,
        "switchable": fixed_launcher is None,
        "default_variant": default_variant,
        "error": None,
    }
    engine_lock = asyncio.Lock()
    manager = backend if backend is not None else build_native_backend(
        runtime_paths, memory_profile=memory_profile
    )
    lora_catalog = _discover_lora_checkpoints(runtime_paths.model_dir)
    initial_family = (
        None if fixed_launcher is None else launcher_family(fixed_launcher)
    )
    compatible_loras = {
        item["id"]
        for item in lora_catalog
        if item["compatible"]
        and (
            initial_family is None
            or initial_family in item["profile"]["task_families"]
        )
    }
    saved_lora = _load_lora_selection(paths.data_dir)
    selected_lora = (
        saved_lora
        if saved_lora in compatible_loras
        else DEFAULT_LORA_CHECKPOINT
        if DEFAULT_LORA_CHECKPOINT in compatible_loras
        else next(iter(sorted(compatible_loras)), None)
    )
    lora_state: dict[str, Any] = {
        "selected": selected_lora,
        "changing": False,
    }
    configure_lora = getattr(manager, "configure_lora_checkpoint", None)
    if selected_lora is not None and callable(configure_lora):
        configure_lora(runtime_paths.model_dir / "loras" / selected_lora)
    # Keep H3 latent re-sampling and the one-step temporal video model as two
    # explicit choices.  The latter lives in an isolated Python process
    # because its pinned Torch/CUDA ABI differs from the H3 runtime.
    legacy_upscaler_injected = upscaler is not None
    if upscaler is not None:
        video_upscaler = upscaler
    else:
        temporal_upscaler = FlashVSRUpscaler(runtime_paths)
        video_upscaler = (
            temporal_upscaler
            if not temporal_upscaler.status().get("missing")
            else RetiredFlashVSRUpscaler()
        )
    service = JobService(
        runtime_paths.data_dir,
        manager,
        max_queued_jobs=max_queued_jobs,
        output_root=runtime_paths.output_dir,
        upscaler=video_upscaler,
        memory_profile_getter=lambda: memory_state["profile"],
    )
    app["job_service"] = service
    infinite_store = InfiniteProjectStore(runtime_paths.data_dir)
    app["infinite_project_store"] = infinite_store
    infinite_batch_tasks: dict[str, asyncio.Task] = {}
    app["infinite_batch_tasks"] = infinite_batch_tasks
    reference_media_state = _load_reference_media_settings(paths.data_dir)
    face_repair_state = _load_face_repair_settings(paths.data_dir)
    checkpoint_preview_state = _load_checkpoint_preview_settings(paths.data_dir)
    second_sampling_window_state = _load_second_sampling_window_settings(
        paths.data_dir
    )
    generation_limit_state = {
        "policy": load_generation_limit_policy(paths.data_dir),
        "detected_vram_gib": detect_gpu_vram_gib(),
    }
    resource_monitor = ResourceMonitor()

    def active_engine(*, required: bool = False) -> str | None:
        engine = engine_state["active"]
        if required and engine is None:
            raise ContractError("select an engine before submitting a generation")
        return engine

    def active_launcher(*, required: bool = False) -> str | None:
        launcher = engine_state["launcher"]
        if required and launcher is None:
            raise ContractError("select a model weight before submitting a generation")
        return launcher

    def engine_options() -> dict[str, Any]:
        engine = active_engine()
        document = public_options(
            fixed_launcher if fixed_launcher is not None else None,
            max_duration_by_preset=(
                generation_limit_state["policy"].preset_limits
            ),
        )
        document["deployment_mode"] = (
            "fixed_engine" if fixed_engine is not None else "unified_console"
        )
        document["current_engine"] = engine
        document["current_launcher"] = active_launcher()
        document["active_service_family"] = engine
        document["active_weight_tier"] = engine_state["weight_tier"]
        document["active_vram_profile"] = engine_state["vram_profile"]
        document["current_model_variant"] = engine_state["default_variant"]
        document["current_engine_options"] = (
            document["service_families"].get(engine) if engine is not None else None
        )
        document["defaults"]["service_family"] = engine
        document["defaults"]["runtime_launcher"] = active_launcher()
        document["defaults"]["weight_tier"] = engine_state["weight_tier"]
        document["defaults"]["vram_profile"] = engine_state["vram_profile"]
        document["device_memory_backend"]["weight_tier"] = engine_state["weight_tier"]
        document["device_memory_backend"]["vram_profile"] = engine_state["vram_profile"]
        document["defaults"]["model_variant"] = engine_state["default_variant"]
        document["defaults"]["engine"] = (
            resolve_engine(engine, engine_state["default_variant"])
            if engine is not None else None
        )
        document["defaults"]["quality"] = (
            default_quality(resolve_engine(engine, engine_state["default_variant"]))
            if engine is not None else None
        )
        document["defaults"]["reference_image_resolution"] = (
            reference_media_state["image_resolution"]
        )
        document["defaults"]["reference_video_resolution"] = (
            reference_media_state["video_resolution"]
        )
        document["reference_media_processing"]["image_default"] = (
            reference_media_state["image_resolution"]
        )
        document["reference_media_processing"]["video_default"] = (
            reference_media_state["video_resolution"]
        )
        document["reference_media_processing"]["scope"] = (
            "server_default_with_per_request_override"
        )
        document["face_repair"] = dict(face_repair_state)
        document["generation_limits"] = generation_limit_state["policy"].public(
            generation_limit_state["detected_vram_gib"]
        )
        document["checkpoint_preview"] = dict(checkpoint_preview_state)
        document["second_sampling_window"] = dict(
            second_sampling_window_state
        )
        document["engine_control"] = {
            "switchable": engine_state["switchable"],
            "switching": engine_state["switching"],
            "active": engine,
            "launcher": active_launcher(),
            "weight_tier": engine_state["weight_tier"],
            "vram_profile": engine_state["vram_profile"],
            "error": engine_state["error"],
        }
        document["model_choices"] = {
            "fl2va_w4a8": {
                "service_family": "first_last", "weight_tier": "w4a8",
                "label": "W4A8 · FL2VA",
                "description": "轻量权重；支持8GB起的显卡，显存档位自动匹配。",
            },
            "ref2va_w4a8": {
                "service_family": "reference", "weight_tier": "w4a8",
                "label": "W4A8 · Ref2VA",
                "description": "轻量多参考权重；显存档位自动匹配。",
            },
            "fl2va_int8": {
                "service_family": "first_last", "weight_tier": "int8",
                "label": "INT8 · FL2VA",
                "description": "高质量INT8权重；要求至少16GB显存。",
            },
            "ref2va_int8": {
                "service_family": "reference", "weight_tier": "int8",
                "label": "INT8 · Ref2VA",
                "description": "高质量INT8多参考权重；要求至少16GB显存。",
            },
        }
        vram_profile = engine_state["vram_profile"]
        weight_tier = engine_state["weight_tier"]
        if weight_tier == "w4a8":
            if vram_profile == "8gb":
                document["resolutions"] = ["360p", "480p", "540p", "720p"]
                document["advanced_limits"]["dimension_max"] = 1280
                document["advanced_limits"]["short_edge_max"] = 736
                document["advanced_limits"]["max_pixels"] = 1280 * 736
            else:
                document["resolutions"] = [
                    "360p", "480p", "540p", "720p", "900p", "1080p"
                ]
                document["advanced_limits"]["dimension_max"] = 1920
                document["advanced_limits"]["short_edge_max"] = 1088
                document["advanced_limits"]["max_pixels"] = 1920 * 1088
            document["advanced_limits"]["second_sampling"]["available"] = True
            document["advanced_limits"]["second_sampling"]["levels"] = [
                "720p", "900p", "1080p"
            ]
            document["advanced_limits"]["second_sampling"]["reason"] = None
        elif vram_profile == "16gb":
            # The compact full-context graph admits the complete
            # 1920x1088x362 INT8 boundary inside the 15.25-GiB planner budget.
            # Native 1080p first generation remains an experimental edge;
            # 1440p second sampling is separately admitted by the physical
            # Ref2VA image+audio 362-frame release gate from 2026-08-29.
            document["resolutions"] = [
                "360p", "480p", "540p", "720p", "900p", "1080p"
            ]
            document["advanced_limits"]["dimension_max"] = 1920
            document["advanced_limits"]["short_edge_max"] = 1088
            document["advanced_limits"]["max_pixels"] = 1920 * 1088
            document["advanced_limits"]["second_sampling"]["available"] = True
            document["advanced_limits"]["second_sampling"]["levels"] = [
                "720p", "900p", "1080p", "1220p", "1440p"
            ]
        else:
            # A single 2K15 Actual-step gate proved memory admission, not a
            # usable full clip.  Keep first-pass creation at the reviewed
            # 1080p boundary; 2K remains available through second sampling.
            document["resolutions"] = [
                "360p", "480p", "540p", "720p", "900p", "1080p"
            ]
            document["advanced_limits"]["dimension_max"] = 1920
            document["advanced_limits"]["short_edge_max"] = 1088
            document["advanced_limits"]["max_pixels"] = 1920 * 1088
            document["advanced_limits"]["second_sampling"]["available"] = True
            document["advanced_limits"]["second_sampling"]["levels"] = [
                "720p", "900p", "1080p", "1220p", "1440p"
            ]
        # The operator duration editor describes first-pass generation only.
        # Do not leak the independent 2K second-sampling target into it.
        document["generation_limits"]["resolutions"] = list(
            document["resolutions"]
        )
        progressive_levels = list(document["resolutions"])
        if weight_tier == "int8":
            for level in document["advanced_limits"]["second_sampling"]["levels"]:
                if level not in progressive_levels:
                    progressive_levels.append(level)
        document["progressive_resolutions"] = progressive_levels
        progressive_values = [
            1440 if level in {"2k", "1440p"} else int(level[:-1])
            for level in progressive_levels
        ]
        native_values = [int(level[:-1]) for level in document["resolutions"]]
        document["progressive_resolution"] = {
            "min": min(progressive_values),
            "max": max(progressive_values),
            "first_pass_max": min(
                GENERATION_RESOLUTION_MAX, max(native_values)
            ),
            "step": 1,
            "detents": [
                value for value in PROGRESSIVE_RESOLUTION_DETENTS
                if min(progressive_values) <= value <= max(progressive_values)
            ],
        }
        if workspace_controller is not None:
            document["workspace"] = workspace_controller.public(
                switchable=engine is None and not engine_state["switching"] and not service_busy()
            )
        else:
            document["workspace"] = {
                "current": {
                    "path": str(paths.release_root),
                    "name": "legacy",
                    "is_default": True,
                    "output_path": str(paths.output_dir),
                },
                "default_path": str(paths.release_root),
                "switchable": False,
            }
        return document

    def service_busy() -> bool:
        return bool(service.pending) or any(
            job.status in {"queued", "starting_backend", "running", "awaiting_preview"}
            for job in service.jobs.values()
        )

    async def select_engine(request: web.Request) -> web.Response:
        if not engine_state["switchable"]:
            raise web.HTTPConflict(text="this process uses a fixed engine launcher")
        try:
            document = await request.json()
            requested_variant = str(document.get("model_variant", "") or "base")
            raw_launcher = document.get(
                "launcher", document.get("runtime_launcher")
            )
            requested_profile = memory_state["profile"]
            if raw_launcher not in (None, ""):
                requested_launcher = normalize_launcher(str(raw_launcher))
                requested = launcher_family(requested_launcher)
            else:
                requested = str(
                    document.get("service_family", document.get("engine", ""))
                )
                if requested in ENGINES:
                    requested_variant = engine_variant(requested)
                    requested = engine_family(requested)
                if requested not in SERVICE_FAMILIES:
                    raise ContractError(
                        f"unsupported service family: {requested}"
                    )
                requested_weight = str(
                    document.get("weight_tier", "int8")
                ).strip().lower()
                detected_vram = generation_limit_state["detected_vram_gib"]
                if detected_vram is None:
                    raise ContractError(
                        "GPU VRAM could not be detected; use an explicit legacy launcher"
                    )
                try:
                    requested_launcher = automatic_launcher(
                        requested, requested_weight, detected_vram
                    )
                except ValueError as error:
                    raise ContractError(str(error)) from error
            requested_weight_tier = launcher_weight_tier(requested_launcher)
            requested_vram_profile = launcher_vram_profile(requested_launcher)
            if document.get("host_memory_limit_gib") not in (None, ""):
                requested_profile = resolve_host_memory_budget_profile(
                    requested_weight_tier,
                    document["host_memory_limit_gib"],
                    vram_profile=requested_vram_profile,
                    status=memory_state["capacity"],
                )
            elif raw_launcher in (None, ""):
                minimum, _ = host_memory_budget_bounds(
                    requested_weight_tier, memory_state["capacity"]
                )
                requested_profile = resolve_host_memory_budget_profile(
                    requested_weight_tier,
                    minimum,
                    vram_profile=requested_vram_profile,
                    status=memory_state["capacity"],
                )
            else:
                validate_profile_for_weight_tier(
                    requested_profile, requested_weight_tier,
                )
            if requested_variant not in {"base", "lora"}:
                raise ContractError(f"unsupported model variant: {requested_variant}")
        except (ContractError, json.JSONDecodeError, RuntimeError, ValueError) as error:
            raise web.HTTPBadRequest(text=str(error)) from error
        if memory_state["changing"]:
            raise web.HTTPConflict(text="host-memory profile is changing")
        selected_checkpoint = lora_state.get("selected")
        if selected_checkpoint:
            selected_profile = resolve_lora_profile(selected_checkpoint)
            if requested not in selected_profile.task_families:
                raise web.HTTPConflict(
                    text=(
                        f"selected LoRA {selected_profile.display_name!r} does not "
                        f"support {requested}; choose a compatible LoRA in settings first"
                    )
                )
        async with engine_lock:
            if service_busy():
                raise web.HTTPConflict(
                    text="cancel or finish all running and queued jobs before switching service family"
                )
            if engine_state["switching"]:
                raise web.HTTPConflict(text="engine is already switching")
            if (
                active_launcher() == requested_launcher
                and memory_state["profile"].key == requested_profile.key
                and manager.warm_state.get("status") == "ready"
            ):
                engine_state["default_variant"] = requested_variant
                return web.json_response({
                    "changed": False, "active_engine": requested,
                    "active_launcher": requested_launcher,
                    "warm_state": manager.warm_state,
                })
            engine_state.update({"switching": True, "error": None})
            previous = active_engine()
            previous_launcher = active_launcher()
            previous_profile = memory_state["profile"]
            try:
                await video_upscaler.stop()
                await manager.stop()
                engine_state["active"] = None
                engine_state["launcher"] = None
                engine_state["weight_tier"] = None
                engine_state["vram_profile"] = None
                if requested_profile.process_limit_gib is not None:
                    if budget_controller.state().enforced:
                        budget_controller.clear()
                    budget_controller.apply(requested_profile.process_limit_gib)
                configure_memory = getattr(
                    manager, "configure_memory_profile", None
                )
                if callable(configure_memory):
                    configure_memory(requested_profile)
                memory_state["profile"] = requested_profile
                await manager.preload(requested_launcher)
                if manager.warm_state.get("status") != "ready":
                    raise RuntimeError("the selected engine failed to load")
                engine_state["active"] = requested
                engine_state["launcher"] = requested_launcher
                engine_state["weight_tier"] = requested_weight_tier
                engine_state["vram_profile"] = requested_vram_profile
                engine_state["default_variant"] = requested_variant
                if memory_state["profile"].preload_upscaler:
                    try:
                        await video_upscaler.start()
                    except Exception:
                        # FlashVSR is optional and isolated; its preload must
                        # never roll back an otherwise healthy H3 engine.
                        pass
            except Exception as error:
                engine_state.update({
                    "active": None,
                    "launcher": None,
                    "weight_tier": None,
                    "vram_profile": None,
                    "error": (
                        f"failed to enter {requested_launcher}; "
                        "the service remains idle"
                    ),
                })
                try:
                    budget_controller.clear()
                except OSError:
                    pass
                configure_memory = getattr(
                    manager, "configure_memory_profile", None
                )
                if callable(configure_memory):
                    configure_memory(previous_profile)
                memory_state["profile"] = previous_profile
                raise web.HTTPInternalServerError(
                    text=f"{engine_state['error']}: {error}"
                ) from error
            finally:
                engine_state["switching"] = False
            return web.json_response({
                "changed": (
                    previous != requested
                    or previous_launcher != requested_launcher
                ),
                "active_engine": requested,
                "active_launcher": requested_launcher,
                "host_memory": requested_profile.public(),
                "warm_state": manager.warm_state,
            })

    async def exit_engine(_: web.Request) -> web.Response:
        if not engine_state["switchable"]:
            raise web.HTTPConflict(text="this process uses a fixed engine launcher")
        if memory_state["changing"]:
            raise web.HTTPConflict(text="host-memory profile is changing")
        async with engine_lock:
            if service_busy():
                raise web.HTTPConflict(
                    text="cancel or finish all running and queued jobs before exiting engine"
                )
            if engine_state["switching"]:
                raise web.HTTPConflict(text="engine is already switching")
            if active_engine() is None:
                return web.json_response({"changed": False, "active_engine": None})
            engine_state.update({"switching": True, "error": None})
            try:
                await video_upscaler.stop()
                await manager.stop()
                engine_state["active"] = None
                engine_state["launcher"] = None
                engine_state["weight_tier"] = None
                engine_state["vram_profile"] = None
                budget_controller.clear()
            finally:
                engine_state["switching"] = False
            return web.json_response({
                "changed": True, "active_engine": None,
                "warm_state": manager.warm_state,
            })

    async def browse_workspace(request: web.Request) -> web.Response:
        if workspace_controller is None:
            raise web.HTTPConflict(text="workspace selection requires the unified console")
        try:
            return web.json_response(
                workspace_controller.browse(request.query.get("path"))
            )
        except ValueError as error:
            raise web.HTTPBadRequest(text=str(error)) from error

    async def select_workspace(request: web.Request) -> web.Response:
        if workspace_controller is None:
            raise web.HTTPConflict(text="workspace selection requires the unified console")
        try:
            document = await request.json()
            layout = workspace_controller.resolve(str(document.get("path", "")))
        except (ValueError, json.JSONDecodeError) as error:
            raise web.HTTPBadRequest(text=str(error)) from error
        async with engine_lock:
            if active_engine() is not None or engine_state["switching"]:
                raise web.HTTPConflict(text="exit the current model before switching workspace")
            if service_busy():
                raise web.HTTPConflict(text="finish or cancel active jobs before switching workspace")
            try:
                await video_upscaler.stop()
                configure_output = getattr(manager, "configure_output_root", None)
                if callable(configure_output):
                    configure_output(layout.output_dir)
                configure_upscaler = getattr(video_upscaler, "configure_data_dir", None)
                if callable(configure_upscaler):
                    configure_upscaler(layout.data_dir)
                service.switch_workspace(layout)
                infinite_store.rebind(layout.data_dir)
                workspace_controller.activate(layout)
            except (OSError, RuntimeError, ValueError) as error:
                raise web.HTTPConflict(text=str(error)) from error
        return web.json_response(
            workspace_controller.public(switchable=True)
        )

    async def index(_: web.Request) -> web.StreamResponse:
        # The console and the Python backend form one protocol surface.  Never
        # let a browser reuse an index document from an older service build:
        # it can otherwise submit retired fields to a freshly updated backend
        # (or hide settings that the backend already supports).
        return web.FileResponse(
            serve_dir / "static/index.html",
            headers={
                "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
                "Pragma": "no-cache",
            },
        )

    async def openapi(_: web.Request) -> web.Response:
        return web.json_response(openapi_document(__version__))

    async def options(_: web.Request) -> web.Response:
        active_memory_profile = memory_state["profile"]
        document = engine_options()
        engine = active_engine()
        launcher = active_launcher()
        runtime = (
            manager.preflight(launcher)
            if launcher is not None
            else {"capabilities": {}}
        )
        capabilities = runtime.get("capabilities", {})
        document["advanced_limits"]["sparse_attention_available"] = bool(
            capabilities.get("sparse_attention", False)
        )
        base_scheduler = (
            (
                capabilities.get("pareto_v24_policy_id")
                or "h3_pareto_v24_human_knee_continuous_deployment_v3"
            )
            if capabilities.get("pareto_v24", False)
            else (
                "v19_certified_frontier"
                if capabilities.get("v19_certified_frontier", False)
                else "h3_int8_frozen_round229"
            )
        )
        document["advanced_limits"]["acceleration"]["scheduler"] = base_scheduler
        document["advanced_limits"]["acceleration"]["scheduler_by_variant"] = {
            "base": base_scheduler,
            "lora": "h3_lora_v1_no_forecast_round229",
        }
        document["advanced_limits"]["acceleration"]["quality_knee"] = (
            capabilities.get("pareto_v24_quality_knee")
        )
        document["advanced_limits"]["acceleration"]["release_candidate"] = (
            capabilities.get("pareto_v24_candidate_id")
        )
        document["advanced_limits"]["upscaler"]["available"] = bool(
            video_upscaler.status().get("ready", False)
        )
        temporal_status = video_upscaler.status()
        second_sampling_limits = document["advanced_limits"]["second_sampling"]
        second_sampling_limits["default_method"] = (
            "temporal" if temporal_status.get("ready", False) else "h3"
        )
        second_sampling_limits["methods"]["temporal"].update({
            "available": bool(temporal_status.get("ready", False)),
            "resident_state": temporal_status.get("resident_state"),
            "reason": (
                None
                if temporal_status.get("ready", False)
                else "FlashVSR v1.1 runtime or weights are not installed"
            ),
        })
        second_sampling_limits["methods"]["h3"].update({
            "available": bool(engine is not None),
            "reason": None if engine is not None else "load an H3 engine first",
        })
        budget_ranges: dict[str, dict[str, object]] = {}
        for weight in ("w4a8", "int8"):
            try:
                minimum, maximum = host_memory_budget_bounds(
                    weight, memory_state["capacity"]
                )
                budget_ranges[weight] = {
                    "available": True,
                    "minimum_gib": minimum,
                    "maximum_gib": maximum,
                    "recommended_gib": minimum,
                }
            except RuntimeError as error:
                budget_ranges[weight] = {
                    "available": False, "reason": str(error),
                }
        document["host_memory"] = {
            "active_profile": active_memory_profile.key,
            "profile": active_memory_profile.public(),
            "detected": memory_state["capacity"].public(),
            "budget_ranges": budget_ranges,
            "enforcement": budget_controller.state().public(),
            "selection_scope": "engine_process_limit",
            "system_reserve_gib": 6,
        }
        document["warm_state"] = manager.warm_state
        return web.json_response(document)

    async def models(_: web.Request) -> web.Response:
        state = model_status(paths.model_dir)
        engine = active_engine()
        launcher = active_launcher()
        if engine is None or launcher is None:
            return web.json_response({
                "engine": None, "ready": state["any_engine_ready"],
                "models": state["launchers"],
            })
        launcher_state = state["launchers"][launcher]
        return web.json_response({
            "engine": engine,
            "launcher": launcher,
            "weight_tier": engine_state["weight_tier"],
            "vram_profile": engine_state["vram_profile"],
            "ready": launcher_state["ready"],
            "models": {
                variant: launcher_state
                for variant in ("base", "lora")
            },
        })

    async def health(_: web.Request) -> web.Response:
        active_memory_profile = memory_state["profile"]
        active_route = (
            service.backend.key.split(":", 1)[0]
            if service.backend.key else None
        )
        return web.json_response({
            "status": "ok",
            "version": __version__,
            "queue_length": len(service.pending),
            "configured_engine": fixed_engine,
            "configured_launcher": fixed_launcher,
            "active_engine": active_engine(),
            "active_launcher": active_launcher(),
            "active_service_family": active_engine(),
            "last_model_variant": (
                engine_variant(active_route) if active_route in ENGINES else None
            ),
            "last_runtime_route": active_route,
            "engine_control": dict(engine_state),
            "warm_state": getattr(
                service.backend, "warm_state", {"status": "unsupported"}
            ),
            "gpu_concurrency": 1,
            "upscaler": video_upscaler.status(),
            "host_memory": {
                "active_profile": active_memory_profile.key,
                "profile": active_memory_profile.public(),
                "detected": memory_state["capacity"].public(),
                "enforcement": budget_controller.state().public(),
                "changing": memory_state["changing"],
            },
        })

    async def resources(_: web.Request) -> web.Response:
        """Return cached host/GPU telemetry without touching model residency."""

        document = await resource_monitor.snapshot()
        usage_reader = getattr(budget_controller, "usage", None)
        if callable(usage_reader):
            document["service_memory"] = usage_reader()
        else:
            # Compatibility for injected third-party controllers.  Never
            # present whole-machine usage as H3 process usage.
            state = budget_controller.state()
            limit = state.limit_gib
            used = float(document.get("process", {}).get("rss_gib", 0.0))
            document["service_memory"] = {
                "used_gib": round(used, 2),
                "limit_gib": limit,
                "peak_gib": None,
                "percent": round(
                    min(100.0, used * 100.0 / limit), 1
                ) if limit else 0.0,
                "enforced": bool(state.enforced),
                "scope": "service_process_fallback",
            }
        document["active_engine"] = active_engine()
        document["active_launcher"] = active_launcher()
        document["warm_state"] = getattr(
            service.backend, "warm_state", {"status": "unsupported"}
        )
        document["queue"] = {
            "running": sum(
                job.status in {"starting_backend", "running", "awaiting_preview"}
                for job in service.jobs.values()
            ),
            "queued": len(service.pending),
        }
        return web.json_response(document)

    async def readiness(_: web.Request) -> web.Response:
        models_state = model_status(paths.model_dir)
        engine = active_engine()
        launcher = active_launcher()
        if engine is None or launcher is None:
            return web.json_response({
                "status": "idle", "engines": models_state["launchers"],
                "message": "select an engine in the control console",
            })
        runtimes = {launcher: manager.preflight(launcher)}
        engines = {
            launcher: {
                "ready": models_state["launchers"][launcher]["ready"]
                and runtimes[launcher]["ready"],
                "models": {
                    variant: models_state["launchers"][launcher]
                    for variant in ("base", "lora")
                },
                "runtime": runtimes[launcher],
            }
            for launcher in (launcher,)
        }
        ready = any(value["ready"] for value in engines.values())
        return web.json_response(
            {"status": "ready" if ready else "not_ready", "engines": engines},
            status=200 if ready else 503,
        )

    async def change_memory_profile(request: web.Request) -> web.Response:
        if budget_controller.state().enforced:
            raise web.HTTPConflict(
                text=(
                    "the unified console now owns a hard process-memory budget; "
                    "exit and re-enter the model to change it"
                )
            )
        if memory_state["changing"]:
            raise web.HTTPConflict(text="host-memory profile is already changing")
        if engine_state["switching"]:
            raise web.HTTPConflict(text="engine is switching")
        try:
            document = await request.json()
            requested = str(document.get("profile", ""))
            detected = detect_host_memory()
            reconfiguration_status = HostMemoryStatus(
                detected.physical_total_gib,
                detected.effective_limit_gib,
                min(
                    detected.effective_limit_gib,
                    detected.available_gib + current_process_pss_gib(),
                ),
            )
            profile = resolve_host_memory_profile(
                requested, reconfiguration_status
            )
            if engine_state["weight_tier"] is not None:
                validate_profile_for_weight_tier(
                    profile,
                    str(engine_state["weight_tier"]),
                )
        except (json.JSONDecodeError, RuntimeError, ValueError) as error:
            raise web.HTTPBadRequest(text=str(error)) from error
        if profile.evidence not in {"validated", "review"}:
            raise web.HTTPConflict(
                text=f"{profile.label} is not release-validated on this build"
            )
        if service_busy():
            raise web.HTTPConflict(
                text="wait for the running and queued jobs before changing host memory profile"
            )
        if profile.key == memory_state["profile"].key:
            return web.json_response({
                "changed": False, "profile": profile.public(),
                "warm_state": manager.warm_state,
            })

        previous = memory_state["profile"]
        memory_state["changing"] = True
        try:
            await video_upscaler.stop()
            await manager.stop()
            manager.configure_memory_profile(profile)
            launcher = active_launcher()
            if launcher is not None:
                await manager.preload(launcher)
            if launcher is not None and manager.warm_state.get("status") != "ready":
                raise RuntimeError("H3 failed to preload under the selected host memory profile")
            if profile.preload_upscaler:
                await video_upscaler.start()
            memory_state["profile"] = profile
        except Exception as error:
            # Restore the previously working policy before returning an error.
            try:
                await video_upscaler.stop()
                await manager.stop()
                manager.configure_memory_profile(previous)
                if launcher is not None:
                    await manager.preload(launcher)
                if previous.preload_upscaler:
                    await video_upscaler.start()
            finally:
                memory_state["profile"] = previous
            raise web.HTTPInternalServerError(
                text="memory-profile change failed; the previous mode was restored"
            ) from error
        finally:
            memory_state["changing"] = False
        return web.json_response({
            "changed": True, "profile": profile.public(),
            "warm_state": manager.warm_state,
        })

    async def preview_long_video(request: web.Request) -> web.Response:
        from .long_video import compile_window_story, memory_budget, validate_reference_inputs
        try:
            payload, uploads = await _read_generation_request(request)
            if payload.get("long_video") is None:
                raise ContractError("long_video is required")
            # Preview uses the same active service constraints as submission.
            engine = active_engine(required=True)
            payload["service_family"] = engine
            payload["runtime_launcher"] = active_launcher(required=True)
            payload["weight_tier"] = engine_state["weight_tier"]
            payload["vram_profile"] = engine_state["vram_profile"]
            payload.pop("engine", None)
            spec = GenerationSpec.from_mapping(payload, max_duration_by_preset=generation_limit_state["policy"].preset_limits)
            ni = sum(key.startswith("reference_image_") for key in uploads)
            na = sum(key.startswith("reference_audio_") for key in uploads)
            nv = sum(key.startswith("reference_video_") for key in uploads)
            validate_reference_inputs(spec.long_video, service_family=spec.service_family,
                reference_images=ni, reference_audios=na, reference_videos=nv,
                first_frame="first_frame" in uploads, last_frame="last_frame" in uploads)
            plan, preview = compile_window_story(
                spec.long_video, seed=spec.seed, maximum_frames=spec.frames,
                service_family=spec.service_family,
                first_frame="first_frame" in uploads,
                last_frame="last_frame" in uploads,
            )
            preview["memory_budget"] = memory_budget(spec.long_video, service_family=spec.service_family,
                width=spec.width, height=spec.height, user_images=ni, user_audios=na)
            preview["plan"] = plan.telemetry()
            preview["request"] = spec.to_dict()
        except (ValueError, TypeError) as error:
            raise web.HTTPBadRequest(text=str(error)) from error
        return web.json_response(preview)

    def public_infinite_project(project) -> dict[str, Any]:
        return project.public(service.jobs)

    async def list_infinite_projects(_request: web.Request) -> web.Response:
        projects = sorted(
            infinite_store.projects.values(),
            key=lambda item: item.updated_at,
            reverse=True,
        )
        return web.json_response({
            "projects": [public_infinite_project(item) for item in projects]
        })

    async def create_infinite_project(request: web.Request) -> web.Response:
        try:
            payload = await request.json()
            if not isinstance(payload, dict):
                raise ContractError("request body must be a JSON object")
            if int(payload.get("workflow_version", 1)) >= 2:
                payload["service_family"] = active_engine(required=True)
            payload["preview_branch_steps"] = checkpoint_preview_state["steps"]
            payload["selflift_temporal_window_enabled"] = (
                second_sampling_window_state["enabled"]
            )
            payload["selflift_temporal_window_seconds"] = (
                second_sampling_window_state["window_seconds"]
            )
            payload["selflift_temporal_overlap_seconds"] = (
                second_sampling_window_state["overlap_seconds"]
            )
            project = infinite_store.create(payload)
        except (ContractError, ValueError, TypeError, json.JSONDecodeError) as error:
            raise web.HTTPBadRequest(text=str(error)) from error
        return web.json_response(public_infinite_project(project), status=201)

    def require_infinite_project(request: web.Request):
        try:
            return infinite_store.require(request.match_info["project_id"])
        except ContractError as error:
            raise web.HTTPNotFound(text=str(error)) from error

    async def get_infinite_project(request: web.Request) -> web.Response:
        return web.json_response(public_infinite_project(require_infinite_project(request)))

    async def delete_infinite_project(request: web.Request) -> web.Response:
        project = require_infinite_project(request)
        task = infinite_batch_tasks.pop(project.id, None)
        if task is not None and not task.done():
            task.cancel()
        deleted = infinite_store.delete(project.id)
        return web.json_response({
            "deleted": True,
            "id": deleted.id,
            "retained_job_count": len(deleted.windows),
        })

    async def submit_infinite_window(
        project,
        payload: dict[str, Any],
        uploads: dict[str, tuple[str, bytes]],
        *,
        from_batch: bool = False,
    ):
        if engine_state["switching"]:
            raise ContractError("engine is switching; wait until it is ready")
        active_final = (
            service.jobs.get(project.final_job_id)
            if project.final_job_id else None
        )
        if active_final is not None and active_final.status in {
            "queued", "starting_backend", "running", "awaiting_preview"
        }:
            raise ContractError(
                "wait for the project final sampling job to finish before changing the timeline"
            )
        engine = active_engine(required=True)
        launcher = active_launcher(required=True)
        removed_broken_tail = False
        while project.windows:
            candidate = service.jobs.get(project.windows[-1]["job_id"])
            candidate_status = (
                getattr(candidate, "status", "missing")
                if candidate is not None else "missing"
            )
            if candidate_status not in RETRYABLE_TAIL_STATUSES:
                break
            project.windows.pop()
            removed_broken_tail = True
        if removed_broken_tail:
            if not project.windows and project.workflow_version < 2:
                project.service_family = None
                project.width = None
                project.height = None
                project.resolution = None
                project.aspect_ratio = None
            infinite_store.persist(project)
        if project.service_family not in (None, engine):
            raise ContractError(
                f"this project uses the {project.service_family} service family"
            )
        if project.windows:
            tail_job = service.jobs.get(project.windows[-1]["job_id"])
            if tail_job is None:
                raise ContractError("the current accepted project tail job is missing")
            if tail_job.status != "succeeded":
                raise ContractError("wait for the current tail to finish before appending")
        else:
            tail_job = None
        inherit_references = str(
            payload.get("inherit_references", "true" if tail_job is not None else "false")
        ).strip().lower() in {"1", "true", "yes", "on"}
        excluded_reference_roles = frozenset(
            role.strip()
            for role in str(payload.get("excluded_reference_roles", "")).split(",")
            if role.strip()
        )
        if tail_job is None and excluded_reference_roles:
            raise ContractError("the opening window has no inherited references to remove")
        if tail_job is not None:
            allowed_reference_roles = {"first_frame", "last_frame"}
            allowed_reference_roles.update(
                f"reference_image_{index}"
                for index in range(1, len(tail_job.reference_images) + 1)
            )
            allowed_reference_roles.update(
                f"reference_audio_{index}"
                for index in range(1, len(tail_job.reference_audios) + 1)
            )
            unknown_roles = excluded_reference_roles - allowed_reference_roles
            if unknown_roles:
                raise ContractError(
                    "unknown inherited reference role: " + sorted(unknown_roles)[0]
                )

        reference_image_uploads = sum(
            key.startswith("reference_image_") for key in uploads
        )
        reference_audio_uploads = sum(
            key.startswith("reference_audio_") for key in uploads
        )
        reference_video_uploads = sum(
            key.startswith("reference_video_") for key in uploads
        )
        has_first_frame = "first_frame" in uploads
        has_last_frame = "last_frame" in uploads
        if engine == "reference":
            if has_first_frame or has_last_frame:
                raise ContractError(
                    "Ref2VA infinite windows accept reference images or audio, not FL2VA keyframes"
                )
            if reference_video_uploads:
                raise ContractError(
                    "Ref2VA infinite windows do not yet accept reference video"
                )
            inherited_reference_count = (
                sum(
                    f"reference_image_{index}" not in excluded_reference_roles
                    for index in range(1, len(tail_job.reference_images) + 1)
                )
                + sum(
                    f"reference_audio_{index}" not in excluded_reference_roles
                    for index in range(1, len(tail_job.reference_audios) + 1)
                )
                if tail_job is not None and inherit_references else 0
            )
            if not (reference_image_uploads or reference_audio_uploads or inherited_reference_count):
                raise ContractError(
                    "a Ref2VA window requires a reference image or audio"
                )
        else:
            if reference_image_uploads or reference_audio_uploads or reference_video_uploads:
                raise ContractError(
                    "FL2VA infinite windows accept keyframes; use Ref2VA for image or audio references"
                )

        overview = str(payload.get("overview", project.overview)).strip()
        soundscape = str(
            payload.get("overall_soundscape", project.overall_soundscape)
        ).strip() or "N/A"
        music = str(
            payload.get("non_diegetic_music", project.non_diegetic_music)
        ).strip() or "N/A"
        if project.workflow_version == 2:
            resolved_memory = memory_capacity(
                visual_capacity=project.visual_memory_capacity,
                audio_capacity=project.audio_memory_capacity,
                visual_resolution=project.visual_memory_resolution,
                service_family=engine,
            )
        elif project.workflow_version >= 3:
            resolved_memory = memory_capacity(
                visual_capacity=int(payload.get(
                    "visual_memory_capacity", project.visual_memory_capacity
                )),
                audio_capacity=int(payload.get(
                    "audio_memory_capacity", project.audio_memory_capacity
                )),
                visual_resolution=str(payload.get(
                    "visual_memory_resolution", project.visual_memory_resolution
                )),
                service_family=engine,
            )
        elif any(
            name in payload for name in (
                "visual_memory_capacity",
                "audio_memory_capacity",
                "visual_memory_resolution",
            )
        ):
            resolved_memory = memory_capacity(
                visual_capacity=int(payload.get(
                    "visual_memory_capacity", project.visual_memory_capacity
                )),
                audio_capacity=int(payload.get(
                    "audio_memory_capacity", project.audio_memory_capacity
                )),
                visual_resolution=str(payload.get(
                    "visual_memory_resolution", project.visual_memory_resolution
                )),
                service_family=engine,
            )
        else:
            # Persisted/third-party clients may still send the retired
            # coupled slider. Translate it once at the request boundary.
            legacy_memory = int(payload.get("memory", project.memory))
            resolved_memory = memory_capacity(
                legacy_memory, service_family=engine
            )
        visual_memory_capacity = int(resolved_memory["video_slots"])
        audio_memory_capacity = int(resolved_memory["audio_slots"])
        visual_memory_resolution = str(
            resolved_memory["visual_resolution"]
        )
        memory = (
            0 if not resolved_memory["enabled"]
            else max(1, int(project.memory or 60))
        )
        overlap_seconds = float(
            project.overlap_seconds
            if project.workflow_version == 2
            else payload.get("overlap_seconds", project.overlap_seconds)
        )
        context_frames = (
            0 if tail_job is None else context_frames_for_seconds(overlap_seconds)
        )
        if tail_job is None:
            prompt_reference_images = reference_image_uploads
            prompt_reference_audios = reference_audio_uploads
            prompt_first_frame = has_first_frame
        else:
            prompt_reference_images = reference_image_uploads + (
                sum(
                    f"reference_image_{index}" not in excluded_reference_roles
                    for index in range(1, len(tail_job.reference_images) + 1)
                ) if inherit_references else 0
            )
            prompt_reference_audios = reference_audio_uploads + (
                sum(
                    f"reference_audio_{index}" not in excluded_reference_roles
                    for index in range(1, len(tail_job.reference_audios) + 1)
                ) if inherit_references else 0
            )
            prompt_first_frame = has_first_frame or bool(
                inherit_references and tail_job.first_frame
                and "first_frame" not in excluded_reference_roles
            )
        prompt = compile_infinite_prompt(
            overview=overview,
            window_description=str(payload.get("window_description", "")),
            overall_soundscape=soundscape,
            non_diegetic_music=music,
            continuation=tail_job is not None and context_frames > 0,
            context_frames=context_frames,
            reference_image_count=prompt_reference_images,
            reference_audio_count=prompt_reference_audios,
            first_frame=prompt_first_frame,
        )
        try:
            requested_visible_seconds = float(
                project.window_duration_seconds
                if project.workflow_version == 2
                else payload.get("duration_seconds", project.window_duration_seconds)
            )
        except (TypeError, ValueError) as error:
            raise ContractError("duration_seconds must be numeric") from error
        model_variant = (
            project.model_variant
            if project.workflow_version >= 2
            else str(payload.get("model_variant", engine_state["default_variant"])).strip()
        )
        common = {
            "prompt": prompt,
            "service_family": engine,
            "runtime_launcher": launcher,
            "weight_tier": engine_state["weight_tier"],
            "vram_profile": engine_state["vram_profile"],
            "model_variant": model_variant,
            "sampling_steps": (
                project.sampling_steps
                if project.workflow_version >= 2
                else payload.get("sampling_steps", 7 if model_variant == "lora" else 20)
            ),
            "acceleration": (
                project.acceleration
                if project.workflow_version == 2
                else payload.get("acceleration", project.acceleration)
            ),
            "seed": payload.get("seed", "random"),
            # Infinite windows use the same explicit checkpoint contract as
            # ordinary generation. Complete generation is the default.
            "preview_mode": "off",
            "preview_fast_finish": False,
            "execution_mode": (
                "complete"
                if project.workflow_version >= 2
                else payload.get("execution_mode", "complete")
            ),
            "checkpoint_step": payload.get("checkpoint_step"),
            "checkpoint_retain": payload.get("checkpoint_retain", True),
            "checkpoint_preview": payload.get(
                "checkpoint_preview",
                str(payload.get("execution_mode", "complete")).strip().lower()
                == "checkpoint",
            ),
            "checkpoint_preview_steps": payload.get(
                "checkpoint_preview_steps", checkpoint_preview_state["steps"]
            ),
            "checkpoint_preview_resolution": (
                payload.get(
                    "checkpoint_preview_resolution",
                    checkpoint_preview_state["resolution"],
                )
            ),
            "reference_image_resolution": reference_media_state["image_resolution"],
            "reference_video_resolution": reference_media_state["video_resolution"],
            "selflift_temporal_window_enabled": (
                second_sampling_window_state["enabled"]
            ),
            "selflift_temporal_window_seconds": (
                second_sampling_window_state["window_seconds"]
            ),
            "selflift_temporal_overlap_seconds": (
                second_sampling_window_state["overlap_seconds"]
            ),
        }
        if project.workflow_version >= 3:
            transition_step = int(project.sampling_steps) - int(
                project.final_sampling_steps
            )
            direct_json = project.creation_mode == "json"
            common.update({
                "resolution": project.final_resolution,
                "selflift_enabled": True,
                "selflift_initial_resolution": project.resolution,
                "selflift_transition_step": transition_step,
                "acceleration_transition_step": transition_step,
                # One online window owns one acceleration setting. The same
                # value finishes its disposable low-resolution preview branch;
                # global final generation supplies a separate branch value.
                "second_pass_acceleration": common["acceleration"],
                # Both workflows retain the same source-grid fork. Online
                # creation also decodes a disposable preview; JSON automation
                # stores only the clean split latent and moves on immediately.
                "execution_mode": "checkpoint",
                "checkpoint_step": transition_step,
                "checkpoint_retain": True,
                "checkpoint_preview": not direct_json,
                "checkpoint_preview_steps": checkpoint_preview_state["steps"],
                "checkpoint_preview_resolution": "source",
            })
        if tail_job is None:
            if not 1.0 <= requested_visible_seconds <= 15.0:
                raise ContractError("opening window duration must be between 1 and 15 seconds")
            common.update({
                "duration_seconds": requested_visible_seconds,
                "resolution": common.get("resolution", (
                    project.resolution
                    if project.workflow_version >= 2
                    else payload.get("resolution", "480p")
                )),
                "aspect_ratio": (
                    project.aspect_ratio
                    if project.workflow_version >= 2
                    else payload.get("aspect_ratio", "16:9")
                ),
            })
            spec = GenerationSpec.from_mapping(
                common,
                max_duration_by_preset=(
                    generation_limit_state["policy"].preset_limits
                ),
            )
            job = await service.submit(spec, uploads)
            total_frames = spec.frames
            continuation = None
        else:
            maximum_visible_seconds = max(1.0, 15.0 - overlap_seconds)
            if not 1.0 <= requested_visible_seconds <= maximum_visible_seconds:
                raise ContractError(
                    "continuation duration must be between 1 and "
                    f"{maximum_visible_seconds:g} seconds for the selected overlap"
                )
            visible_frames = visible_frames_for_seconds(
                requested_visible_seconds,
                maximum_physical_frames=362,
                context_frames=context_frames,
            )
            # A hard-cut window carries no prior-tail context, but an
            # independent H3 clip still needs its five-frame temporal origin.
            # Those origin frames are generated as private preroll and trimmed
            # before the visible suffix is appended.
            physical_frames = (context_frames or H3_FRAME_ORIGIN) + visible_frames
            common.update({
                "mode": "advanced",
                "advanced": True,
                "width": project.width,
                "height": project.height,
                "frames": physical_frames,
                # A legal minimum continuation can be 22 physical frames
                # (5 context + 17 new), or 0.917 s at 24 fps. Public
                # GenerationSpec duration validation starts at one second,
                # while the explicit H3 frame count remains authoritative.
                "duration_seconds": max(1.0, physical_frames / INFINITE_FPS),
                "resolution": common.get(
                    "resolution", project.resolution or tail_job.spec.resolution
                ),
                "aspect_ratio": project.aspect_ratio or tail_job.spec.aspect_ratio,
            })
            spec = GenerationSpec.from_mapping(common)
            source_frames = int(project.windows[-1]["total_frames"])
            continuation = InfiniteContinuationSpec(
                project_id=project.id,
                window_index=len(project.windows),
                source_job_id=tail_job.id,
                source_frames=source_frames,
                context_frames=context_frames,
                visible_frames=visible_frames,
                audio_bridge_ticks=round(context_frames / INFINITE_FPS * 40),
                memory=memory,
                visual_memory_capacity=visual_memory_capacity,
                audio_memory_capacity=audio_memory_capacity,
                visual_memory_resolution=visual_memory_resolution,
                source_dialogue=(
                    "<d>" in tail_job.spec.prompt
                    and not tail_job.reference_audios
                ),
            )
            spec = dataclasses.replace(
                spec, output_frames=continuation.output_frames
            )
            job = await service.submit_infinite_continuation(
                source=tail_job,
                reference_source=tail_job,
                spec=spec,
                continuation=continuation,
                uploads=uploads,
                inherit_references=inherit_references,
                excluded_reference_roles=excluded_reference_roles,
            )
            total_frames = continuation.output_frames

        if not project.windows:
            project.service_family = spec.service_family
            project.width = spec.width
            project.height = spec.height
            if project.workflow_version < 3:
                project.resolution = spec.resolution
            project.aspect_ratio = spec.aspect_ratio
        if str(payload.get("save_shared_as_default", "false")).lower() in {
            "1", "true", "yes", "on"
        }:
            project.overview = overview
            project.overall_soundscape = soundscape
            project.non_diegetic_music = music
            if project.workflow_version != 2:
                project.overlap_seconds = overlap_seconds
                project.memory = memory
                project.visual_memory_capacity = visual_memory_capacity
                project.audio_memory_capacity = audio_memory_capacity
                project.visual_memory_resolution = visual_memory_resolution
                project.window_duration_seconds = requested_visible_seconds
                project.acceleration = float(common["acceleration"])
        project.final_job_id = None
        project.final_sampling_settings = None
        if not from_batch and project.batch_status != "running":
            project.batch_plan = []
            project.batch_cursor = 0
            project.batch_status = "idle"
            project.batch_error = None
        batch_plan_index = payload.get("_batch_plan_index") if from_batch else None
        if batch_plan_index is not None:
            batch_plan_index = int(batch_plan_index)
            project.batch_cursor = max(project.batch_cursor, batch_plan_index + 1)
        project.windows.append({
            "index": len(project.windows),
            "job_id": job.id,
            "requested_duration_seconds": requested_visible_seconds,
            "actual_duration_seconds": (
                spec.frames / INFINITE_FPS
                if continuation is None else continuation.visible_frames / INFINITE_FPS
            ),
            "context_frames": context_frames,
            "overlap_seconds": context_frames / INFINITE_FPS,
            "memory": memory,
            "visual_memory_capacity": visual_memory_capacity,
            "audio_memory_capacity": audio_memory_capacity,
            "visual_memory_resolution": visual_memory_resolution,
            "overview": overview,
            "window_description": str(payload.get("window_description", "")).strip(),
            "overall_soundscape": soundscape,
            "non_diegetic_music": music,
            "model_variant": spec.model_variant,
            "sampling_steps": spec.sampling_steps,
            "acceleration": spec.acceleration,
            "seed": spec.seed,
            "total_frames": total_frames,
            "created_at": time.time(),
            "batch_plan_index": batch_plan_index,
        })
        infinite_store.persist(project)
        return job

    async def run_infinite_batch(project_id: str) -> None:
        """Submit persisted JSON windows one at a time behind the active tail."""

        def window_uploads(
            references: dict[str, str] | None,
        ) -> dict[str, tuple[str, bytes]]:
            if references is None:
                return {}
            uploads: dict[str, tuple[str, bytes]] = {}
            for label, source in references.items():
                match = re.fullmatch(r"(Picture|Audio) ([1-9])", label)
                if match is None:
                    raise ContractError(f"invalid persisted reference id: {label}")
                kind, index = match.groups()
                path = Path(source)
                if not path.is_file():
                    raise ContractError(f"reference file is missing: {path}")
                role = (
                    f"reference_image_{index}"
                    if kind == "Picture"
                    else f"reference_audio_{index}"
                )
                uploads[role] = (path.name, path.read_bytes())
            return uploads

        try:
            while True:
                project = infinite_store.projects.get(project_id)
                if project is None or project.batch_status != "running":
                    return
                if project.windows:
                    tail = service.jobs.get(project.windows[-1]["job_id"])
                    if tail is None or tail.status in RETRYABLE_TAIL_STATUSES:
                        project.batch_status = "failed"
                        project.batch_error = (
                            "the latest window failed; edit or discard it before retrying"
                        )
                        infinite_store.persist(project)
                        return
                    if tail.status != "succeeded":
                        await asyncio.sleep(0.5)
                        continue
                if project.batch_cursor >= len(project.batch_plan):
                    if (
                        project.workflow_version >= 3
                        and project.creation_mode == "json"
                    ):
                        tail = service.jobs.get(project.windows[-1]["job_id"])
                        if tail is None or tail.status != "succeeded":
                            raise RuntimeError(
                                "the final JSON window did not complete successfully"
                            )
                        transition_step = int(project.sampling_steps) - int(
                            project.final_sampling_steps
                        )
                        sources = tuple(
                            service.jobs[window["job_id"]]
                            for window in project.windows
                        )
                        final_job = await service.submit_infinite_selflift_final(
                            sources,
                            output_frames=int(project.windows[-1]["total_frames"]),
                            final_resolution=project.final_resolution,
                            second_pass_acceleration=project.second_pass_acceleration,
                            sigma_scale=1.0,
                            temporal_window_enabled=second_sampling_window_state[
                                "enabled"
                            ],
                            temporal_window_seconds=second_sampling_window_state[
                                "window_seconds"
                            ],
                            temporal_overlap_seconds=second_sampling_window_state[
                                "overlap_seconds"
                            ],
                        )
                        project.final_job_id = final_job.id
                        project.final_sampling_settings = {
                            "method": "global_sliding_selflift",
                            "resolution": project.final_resolution,
                            "steps": int(project.final_sampling_steps),
                            "acceleration": float(
                                project.second_pass_acceleration
                            ),
                            "sigma_scale": 1.0,
                            "temporal_window_enabled": bool(
                                second_sampling_window_state["enabled"]
                            ),
                            "temporal_window_seconds": float(
                                second_sampling_window_state["window_seconds"]
                            ),
                            "temporal_overlap_seconds": float(
                                second_sampling_window_state["overlap_seconds"]
                            ),
                            "total_steps": int(project.sampling_steps),
                            "transition_step": transition_step,
                            "shared_preview_prefix": True,
                        }
                        infinite_store.persist(project)
                        while final_job.status not in {
                            "succeeded", "failed", "cancelled"
                        }:
                            await asyncio.sleep(0.5)
                        if final_job.status != "succeeded":
                            raise RuntimeError(
                                final_job.error or "global SelfLift finalization failed"
                            )
                    project.batch_status = "completed"
                    project.batch_error = None
                    infinite_store.persist(project)
                    return
                item = project.batch_plan[project.batch_cursor]
                payload = {
                    "overview": project.overview,
                    "window_description": item["window_description"],
                    "overall_soundscape": project.overall_soundscape,
                    "non_diegetic_music": project.non_diegetic_music,
                    "seed": item.get("seed", "random"),
                    "save_shared_as_default": False,
                    "_batch_plan_index": project.batch_cursor,
                }
                references = item.get("_references")
                if references is not None:
                    payload["inherit_references"] = False
                for name in (
                    "duration_seconds", "overlap_seconds", "acceleration",
                    "visual_memory_capacity", "audio_memory_capacity",
                    "visual_memory_resolution",
                ):
                    if name in item:
                        payload[name] = item[name]
                job = await submit_infinite_window(
                    project,
                    payload,
                    window_uploads(references),
                    from_batch=True,
                )
                while job.status not in {
                    "succeeded", "failed", "cancelled", "checkpointed"
                }:
                    await asyncio.sleep(0.5)
                if job.status != "succeeded":
                    project.batch_status = "failed"
                    project.batch_error = job.error or "window generation failed"
                    infinite_store.persist(project)
                    return
        except asyncio.CancelledError:
            raise
        except Exception as error:
            project = infinite_store.projects.get(project_id)
            if project is not None:
                project.batch_status = "failed"
                project.batch_error = str(error)
                infinite_store.persist(project)
        finally:
            current = infinite_batch_tasks.get(project_id)
            if current is asyncio.current_task():
                infinite_batch_tasks.pop(project_id, None)

    def launch_infinite_batch(project_id: str) -> None:
        current = infinite_batch_tasks.get(project_id)
        if current is not None and not current.done():
            return
        infinite_batch_tasks[project_id] = asyncio.create_task(
            run_infinite_batch(project_id),
            name=f"infinite-json-{project_id}",
        )

    async def start_infinite_batch(request: web.Request) -> web.Response:
        project = require_infinite_project(request)
        try:
            if project.workflow_version < 2:
                raise ContractError("JSON automation requires a locked preview-trajectory project")
            if project.batch_status == "running":
                raise ContractError("the project JSON queue is already running")
            active_final = (
                service.jobs.get(project.final_job_id)
                if project.final_job_id else None
            )
            if active_final is not None and active_final.status in {
                "queued", "starting_backend", "running", "awaiting_preview"
            }:
                raise ContractError(
                    "wait for the project final sampling job to finish before changing the timeline"
                )
            if project.windows:
                tail = service.jobs.get(project.windows[-1]["job_id"])
                if tail is None or tail.status != "succeeded":
                    raise ContractError("resolve the current tail before starting a JSON queue")
            document = await request.json()
            if not isinstance(document, dict):
                raise ContractError("request body must be a JSON object")
            def normalize_references(raw_references, field_name: str) -> dict[str, str]:
                if raw_references is None:
                    raw_references = {}
                if not isinstance(raw_references, dict):
                    raise ContractError(
                        f"{field_name} must map Picture/Audio IDs to local file paths"
                    )
                normalized_references: dict[str, str] = {}
                reference_indices: dict[str, set[int]] = {
                    "Picture": set(), "Audio": set()
                }
                for raw_label, raw_source in raw_references.items():
                    label = str(raw_label).strip().strip("<>")
                    match = re.fullmatch(r"(Picture|Audio)\s+([1-9])", label)
                    if match is None:
                        raise ContractError(
                            "reference IDs must be Picture 1..9 or Audio 1..3"
                        )
                    kind, raw_index = match.groups()
                    index = int(raw_index)
                    if kind == "Audio" and index > 3:
                        raise ContractError("Audio reference IDs cannot exceed Audio 3")
                    source_value = (
                        raw_source.get("path")
                        if isinstance(raw_source, dict)
                        else raw_source
                    )
                    source = Path(str(source_value or "")).expanduser()
                    if not source.is_absolute():
                        source = (Path.cwd() / source).resolve()
                    else:
                        source = source.resolve()
                    if not source.is_file():
                        raise ContractError(f"reference file is missing: {source}")
                    if source.stat().st_size > MAX_IMAGE_BYTES:
                        raise ContractError(
                            f"reference file exceeds {MAX_IMAGE_BYTES // (1024 * 1024)} MiB: {source.name}"
                        )
                    normalized = f"{kind} {index}"
                    normalized_references[normalized] = str(source)
                    reference_indices[kind].add(index)
                for kind, indices in reference_indices.items():
                    if indices and indices != set(range(1, max(indices) + 1)):
                        raise ContractError(
                            f"{field_name} {kind} IDs must be contiguous from 1"
                        )
                return dict(sorted(
                    normalized_references.items(),
                    key=lambda item: (
                        0 if item[0].startswith("Picture ") else 1,
                        int(item[0].split()[-1]),
                    ),
                ))

            references = normalize_references(
                document.get("references", {}), "references"
            )
            raw_windows = document.get("windows")
            if not isinstance(raw_windows, list) or not 1 <= len(raw_windows) <= 100:
                raise ContractError("windows must contain between 1 and 100 items")
            plan = []
            effective_references: dict[str, str] | None = (
                references or (None if project.windows else {})
            )
            for index, raw in enumerate(raw_windows, start=1):
                explicit_window_references = False
                if isinstance(raw, str):
                    description = raw.strip()
                    seed = "random"
                elif isinstance(raw, dict):
                    description = str(
                        raw.get("prompt", raw.get("window_description", ""))
                    ).strip()
                    seed = raw.get("seed", "random")
                else:
                    raise ContractError(f"windows[{index - 1}] must be a string or object")
                if not description:
                    raise ContractError(f"windows[{index - 1}] prompt is required")
                if len(description) > 12000:
                    raise ContractError(f"windows[{index - 1}] prompt is too long")
                item = {"window_description": description, "seed": seed}
                if isinstance(raw, dict):
                    if "references" in raw:
                        explicit_window_references = True
                        effective_references = normalize_references(
                            raw.get("references"),
                            f"windows[{index - 1}].references",
                        )
                    for name in (
                        "duration_seconds", "overlap_seconds", "acceleration",
                        "visual_memory_capacity", "audio_memory_capacity",
                        "visual_memory_resolution",
                    ):
                        if name in raw:
                            item[name] = raw[name]
                if project.service_family == "reference":
                    if effective_references == {}:
                        raise ContractError(
                            f"windows[{index - 1}] requires Picture or Audio references"
                        )
                    should_upload_references = bool(
                        explicit_window_references
                        or (index == 1 and not project.windows and references)
                    )
                    item["_references"] = (
                        dict(effective_references)
                        if should_upload_references
                        and effective_references is not None
                        else None
                    )
                    if effective_references is not None:
                        item["_reference_ids"] = sorted(effective_references)
                elif references or (
                    isinstance(raw, dict) and "references" in raw
                ):
                    raise ContractError(
                        "JSON reference mappings require the Ref2VA service family"
                    )
                plan.append(item)
            for field_name in (
                "overview", "overall_soundscape", "non_diegetic_music"
            ):
                if field_name in document:
                    value = str(document[field_name]).strip()
                    if field_name != "overview" and not value:
                        value = "N/A"
                    setattr(project, field_name, value)
            if project.workflow_version >= 3:
                final_acceleration = float(
                    document.get(
                        "final_acceleration", project.second_pass_acceleration
                    )
                )
                if (
                    not math.isfinite(final_acceleration)
                    or not 0 <= final_acceleration <= 100
                ):
                    raise ContractError(
                        "final_acceleration must be between 0 and 100"
                    )
                project.second_pass_acceleration = final_acceleration
            project.batch_plan = plan
            project.batch_references = references
            project.batch_cursor = 0
            project.batch_status = "running"
            project.batch_error = None
            project.final_job_id = None
            project.final_sampling_settings = None
            infinite_store.persist(project)
            launch_infinite_batch(project.id)
        except (ContractError, ValueError, TypeError, json.JSONDecodeError) as error:
            raise web.HTTPBadRequest(text=str(error)) from error
        return web.json_response(public_infinite_project(project), status=202)

    async def stop_infinite_batch(request: web.Request) -> web.Response:
        project = require_infinite_project(request)
        task = infinite_batch_tasks.pop(project.id, None)
        if task is not None and not task.done():
            task.cancel()
        if project.batch_status == "running":
            project.batch_status = "stopped"
            project.batch_error = None
            infinite_store.persist(project)
        return web.json_response(public_infinite_project(project))

    async def append_infinite_window(request: web.Request) -> web.Response:
        project = require_infinite_project(request)
        try:
            payload, uploads = await _read_generation_request(request)
            if project.batch_status == "running":
                raise ContractError("the project JSON queue is running")
            job = await submit_infinite_window(project, payload, uploads)
        except (ContractError, ValueError, TypeError, json.JSONDecodeError) as error:
            raise web.HTTPBadRequest(text=str(error)) from error
        return web.json_response({
            "project": public_infinite_project(project),
            "job": service.serialize(job),
        }, status=202)

    async def discard_infinite_tail(request: web.Request) -> web.Response:
        project = require_infinite_project(request)
        if project.batch_status == "running":
            raise web.HTTPConflict(text="stop the project JSON queue before discarding")
        active_final = (
            service.jobs.get(project.final_job_id)
            if project.final_job_id else None
        )
        if active_final is not None and active_final.status in {
            "queued", "starting_backend", "running", "awaiting_preview"
        }:
            raise web.HTTPConflict(
                text="wait for the project final sampling job to finish before changing the timeline"
            )
        if not project.windows:
            raise web.HTTPConflict(text="the project has no window to discard")
        tail = project.windows[-1]
        job = service.jobs.get(tail["job_id"])
        if job is not None and job.status in {"starting_backend", "running", "awaiting_preview"}:
            raise web.HTTPConflict(text="cancel the running tail before discarding it")
        if job is not None:
            try:
                await service.delete(job.id)
            except ContractError as error:
                raise web.HTTPConflict(text=str(error)) from error
        project.windows.pop()
        if not project.windows and project.workflow_version < 2:
            project.service_family = None
            project.width = project.height = None
            project.resolution = project.aspect_ratio = None
        project.final_job_id = None
        project.final_sampling_settings = None
        infinite_store.persist(project)
        return web.json_response(public_infinite_project(project))

    async def second_sample_infinite_project(request: web.Request) -> web.Response:
        project = require_infinite_project(request)
        if project.batch_status == "running":
            raise web.HTTPConflict(text="wait for the project JSON queue to finish")
        if not project.windows:
            raise web.HTTPConflict(text="the project has no completed video")
        source = service.jobs.get(project.windows[-1]["job_id"])
        if source is None or source.status != "succeeded":
            raise web.HTTPConflict(text="the current project tail is not complete")
        try:
            document = await request.json()
            if not isinstance(document, dict):
                raise ContractError("request body must be a JSON object")
            document = dict(document)
            previous_final = (
                service.jobs.get(project.final_job_id)
                if project.final_job_id else None
            )
            if previous_final is not None and previous_final.status in {
                "queued", "starting_backend", "running", "awaiting_preview"
            }:
                raise ContractError("the project final sampling job is already active")

            if project.workflow_version >= 3:
                if project.creation_mode == "json":
                    if project.batch_status != "completed":
                        raise ContractError(
                            "JSON direct generation has not completed"
                        )
                    if previous_final is None or previous_final.status != "succeeded":
                        raise ContractError(
                            "JSON generation has no completed global final job"
                        )
                    # Compatibility endpoint for clients that still call
                    # final-sampling after a JSON batch. No work is queued.
                    return web.json_response(
                        service.serialize(previous_final), status=200
                    )
                # SelfLift's target canvas and tail step count belong to the
                # formal Y trajectory. Acceleration is only a runtime attention
                # schedule, so the user may choose it when starting this branch.
                if (
                    "resolution" in document
                    and str(document["resolution"]).strip().lower()
                    != project.final_resolution
                ):
                    raise ContractError(
                        "resolution is fixed by the SelfLift project trajectory"
                    )
                if (
                    "steps" in document
                    and int(document["steps"]) != int(project.final_sampling_steps)
                ):
                    raise ContractError(
                        "steps are fixed by the SelfLift project trajectory"
                    )
                final_acceleration = float(
                    document.get("acceleration", project.second_pass_acceleration)
                )
                if (
                    not math.isfinite(final_acceleration)
                    or not 0 <= final_acceleration <= 100
                ):
                    raise ContractError("acceleration must be between 0 and 100")
                try:
                    final_sigma_scale = float(document.get("sigma_scale", 1.0))
                except (TypeError, ValueError) as error:
                    raise ContractError("sigma_scale must be numeric") from error
                if (
                    not math.isfinite(final_sigma_scale)
                    or not 0.25 <= final_sigma_scale <= 1.0
                ):
                    raise ContractError("sigma_scale must be between 0.25 and 1")
                final_sigma_scale = round(final_sigma_scale, 2)
                sources: list[JobRecord] = []
                for window in project.windows:
                    window_job = service.jobs.get(window["job_id"])
                    if window_job is None or window_job.status != "succeeded":
                        raise ContractError(
                            "every preview window must finish before final generation"
                        )
                    sources.append(window_job)
                job = await service.submit_infinite_selflift_final(
                    tuple(sources),
                    output_frames=int(project.windows[-1]["total_frames"]),
                    final_resolution=project.final_resolution,
                    second_pass_acceleration=final_acceleration,
                    sigma_scale=final_sigma_scale,
                    temporal_window_enabled=second_sampling_window_state[
                        "enabled"
                    ],
                    temporal_window_seconds=second_sampling_window_state[
                        "window_seconds"
                    ],
                    temporal_overlap_seconds=second_sampling_window_state[
                        "overlap_seconds"
                    ],
                )
                transition_step = int(project.sampling_steps) - int(
                    project.final_sampling_steps
                )
                project.final_job_id = job.id
                project.final_sampling_settings = {
                    "method": "global_sliding_selflift",
                    "resolution": project.final_resolution,
                    "steps": int(project.final_sampling_steps),
                    "acceleration": final_acceleration,
                    "sigma_scale": final_sigma_scale,
                    "temporal_window_enabled": bool(
                        second_sampling_window_state["enabled"]
                    ),
                    "temporal_window_seconds": float(
                        second_sampling_window_state["window_seconds"]
                    ),
                    "temporal_overlap_seconds": float(
                        second_sampling_window_state["overlap_seconds"]
                    ),
                    "total_steps": int(project.sampling_steps),
                    "transition_step": transition_step,
                    "shared_preview_prefix": True,
                }
                infinite_store.persist(project)
                return web.json_response(service.serialize(job), status=202)

            document["method"] = "h3"
            document.setdefault("resolution", project.final_resolution)
            document.setdefault("steps", project.final_sampling_steps)
            document.setdefault("acceleration", project.second_pass_acceleration)
            if (
                second_sampling_window_state["enabled"]
                and "temporal_window_frames" not in document
            ):
                from .native_engine.global_co_denoise import (
                    window_geometry_for_seconds,
                )

                window_frames, stride_frames = window_geometry_for_seconds(
                    second_sampling_window_state["window_seconds"],
                    second_sampling_window_state["overlap_seconds"],
                )
                document["temporal_window_frames"] = window_frames
                document.setdefault(
                    "temporal_overlap_frames", window_frames - stride_frames
                )
            second_sampling = SecondSamplingSpec.from_mapping(
                document, source=source.spec
            )
            if second_sampling.method == "h3":
                launcher_definition = LAUNCHER_DEFINITIONS[
                    active_launcher(required=True)
                ]
                maximum_second_edge = max(
                    second_sampling_short_edge(level)[1]
                    for level in launcher_definition.backend.second_sampling_levels
                )
                requested_second_edge = second_sampling_short_edge(
                    second_sampling.resolution
                )[1]
                if requested_second_edge > maximum_second_edge:
                    raise ContractError(
                        "the active backend supports H3 second sampling only up to "
                        f"{maximum_second_edge}p"
                    )
            project_prompt = compile_infinite_prompt(
                overview=project.overview,
                window_description=(
                    "[Shot 1] Preserve the complete accepted source-latent "
                    "trajectory from beginning to end: retain every existing "
                    "action, camera movement, authored cut, subject identity, "
                    "scene layout, speech timing, voice, ambient sound and music. "
                    "Add no new action, cut, object, person, speech or sound event."
                ),
                overall_soundscape=project.overall_soundscape,
                non_diegetic_music=project.non_diegetic_music,
                continuation=False,
                reference_image_count=len(source.reference_images),
                reference_audio_count=len(source.reference_audios),
                first_frame=source.first_frame is not None,
                refinement=True,
            )
            job = await service.submit_second_sampling(
                source, second_sampling, prompt_override=project_prompt
            )
            project.final_job_id = job.id
            project.final_sampling_settings = second_sampling.to_dict()
            infinite_store.persist(project)
        except (ContractError, json.JSONDecodeError) as error:
            raise web.HTTPBadRequest(text=str(error)) from error
        return web.json_response(service.serialize(job), status=202)

    async def create_generation(request: web.Request) -> web.Response:
        try:
            payload, uploads = await _read_generation_request(request)
            if engine_state["switching"]:
                raise ContractError("engine is switching; wait until it is ready")
            async with engine_lock:
                engine = active_engine(required=True)
                if engine_state["switching"]:
                    raise ContractError("engine is switching; wait until it is ready")
                requested_engine = payload.get("service_family", payload.get("engine"))
                if requested_engine in ENGINES:
                    requested_engine = engine_family(str(requested_engine))
                if requested_engine not in (None, "", engine):
                    if fixed_engine is not None:
                        raise ContractError(f"this service is fixed to the {engine} engine")
                    raise ContractError(f"the active engine is {engine}")
                payload.pop("engine", None)
                payload["service_family"] = engine
                payload["runtime_launcher"] = active_launcher(required=True)
                payload["weight_tier"] = engine_state["weight_tier"]
                payload["vram_profile"] = engine_state["vram_profile"]
                payload.setdefault("model_variant", engine_state["default_variant"])
                if payload.get("quality") in (None, ""):
                    payload["quality"] = default_quality(
                        resolve_engine(engine, str(payload["model_variant"]))
                    )
                # The active console owns one preview-step policy. It applies
                # to Web, direct API and ComfyUI jobs for this running service.
                payload.setdefault(
                    "reference_image_resolution",
                    reference_media_state["image_resolution"],
                )
                payload.setdefault(
                    "reference_video_resolution",
                    reference_media_state["video_resolution"],
                )
                payload["checkpoint_preview_steps"] = checkpoint_preview_state["steps"]
                payload["preview_branch_steps"] = checkpoint_preview_state["steps"]
                # A single-video request may explicitly turn temporal views
                # on or off. When it turns them on, the physical window and
                # overlap always inherit the global policy below. API clients
                # that omit the switch retain the configured global default.
                payload.setdefault(
                    "selflift_temporal_window_enabled",
                    second_sampling_window_state["enabled"],
                )
                payload["selflift_temporal_window_seconds"] = (
                    second_sampling_window_state["window_seconds"]
                )
                payload["selflift_temporal_overlap_seconds"] = (
                    second_sampling_window_state["overlap_seconds"]
                )
                payload.setdefault(
                    "checkpoint_preview_resolution",
                    checkpoint_preview_state["resolution"],
                )
                # Compatibility for a console page kept open across the
                # checkpoint-preview upgrade.  The immediately preceding Web
                # build either omitted the preview flag or submitted its old
                # hidden ``false`` value.  Browser FormData is always
                # multipart, even when an embedded browser strips Origin and
                # Referer (VS Code's forwarded-port webview can do that).  A
                # multipart checkpoint is therefore unambiguously a product
                # console submission and always includes its disposable
                # preview.  JSON API clients may still explicitly request a
                # checkpoint without one.
                request_origin = request.headers.get("Origin", "").rstrip("/")
                request_referer = request.headers.get("Referer", "").rstrip("/")
                expected_origin = f"{request.scheme}://{request.host}".rstrip("/")
                same_origin_console = (
                    request_origin == expected_origin
                    or request_referer == expected_origin
                    or request_referer.startswith(f"{expected_origin}/")
                )
                console_form_submission = request.content_type.startswith(
                    "multipart/"
                )
                if (
                    str(payload.get("execution_mode", "complete")).strip().lower()
                    == "checkpoint"
                    and (
                        console_form_submission
                        or same_origin_console
                        or "checkpoint_preview" not in payload
                    )
                ):
                    payload["checkpoint_preview"] = True
                spec = GenerationSpec.from_mapping(
                    payload,
                    max_duration_by_preset=(
                        generation_limit_state["policy"].preset_limits
                    ),
                )
                if spec.upscale_enabled:
                    raise ContractError(
                        "the legacy FlashVSR output upscaler was removed; "
                        "finish the source video, then submit native H3 second sampling "
                        "from its completed job"
                    )
                reference_image_roles = sorted(
                    role for role in uploads if role.startswith("reference_image_")
                )
                reference_video_roles = sorted(
                    role for role in uploads if role.startswith("reference_video_")
                )
                reference_audio_roles = sorted(
                    role for role in uploads if role.startswith("reference_audio_")
                )
                if engine == "reference":
                    if not reference_image_roles and not reference_video_roles and not reference_audio_roles:
                        raise ContractError("reference engine requires at least one reference image, video or audio")
                    if len(reference_image_roles) > MAX_REFERENCE_IMAGES:
                        raise ContractError("reference engine accepts at most 9 reference images")
                    if len(reference_video_roles) > MAX_REFERENCE_VIDEOS:
                        raise ContractError("reference engine accepts at most 3 reference videos")
                    if len(reference_audio_roles) > MAX_REFERENCE_AUDIOS:
                        raise ContractError("reference engine accepts at most 3 reference audios")
                    if "first_frame" in uploads or "last_frame" in uploads:
                        raise ContractError("reference engine does not use first/last-frame anchors")
                elif reference_image_roles or reference_video_roles or reference_audio_roles:
                    raise ContractError("reference media require the Ref2VA engine")
                try:
                    validate_workload_for_profile(
                        memory_state["profile"], width=spec.width,
                        height=spec.height, frames=spec.frames,
                    )
                except ValueError as error:
                    raise ContractError(str(error)) from error
                job = await service.submit(spec, uploads)
        except (ContractError, json.JSONDecodeError) as error:
            raise web.HTTPBadRequest(text=str(error)) from error
        return web.json_response(service.serialize(job), status=202)

    def lora_settings_payload() -> dict[str, Any]:
        available = _discover_lora_checkpoints(runtime_paths.model_dir)
        return {
            "selected": lora_state["selected"],
            "changing": lora_state["changing"],
            "available": available,
            "loaded": manager.warm_state.get("lora_checkpoint"),
        }

    async def get_lora_settings(_request: web.Request) -> web.Response:
        return web.json_response(lora_settings_payload())

    async def configure_lora_settings(request: web.Request) -> web.Response:
        try:
            document = await request.json()
        except (json.JSONDecodeError, ValueError) as error:
            raise web.HTTPBadRequest(text="request body must be JSON") from error
        if not isinstance(document, dict):
            raise web.HTTPBadRequest(text="request body must be an object")
        requested = str(document.get("checkpoint", "")).strip()
        available = {
            item["id"]: item
            for item in _discover_lora_checkpoints(runtime_paths.model_dir)
        }
        selected = available.get(requested)
        if selected is None:
            raise web.HTTPBadRequest(text="selected LoRA checkpoint is not installed")
        if not selected["compatible"]:
            raise web.HTTPBadRequest(
                text="selected file is not a supported MiniMax H3 LoRA"
            )
        launcher = active_launcher()
        if (
            launcher is not None
            and launcher_family(launcher)
            not in selected["profile"]["task_families"]
        ):
            raise web.HTTPBadRequest(
                text="selected LoRA is incompatible with the active FL2VA/Ref2VA family"
            )
        warm = manager.warm_state
        if (
            requested == lora_state["selected"]
            and (
                launcher is None
                or (
                    warm.get("status") == "ready"
                    and warm.get("lora_checkpoint") == Path(requested).name
                )
            )
        ):
            response_data = lora_settings_payload()
            response_data["changed"] = False
            return web.json_response(response_data)
        if service_busy():
            raise web.HTTPConflict(
                text="wait for the running and queued jobs before changing LoRA"
            )
        if memory_state["changing"] or engine_state["switching"]:
            raise web.HTTPConflict(text="the model engine is already changing")
        configure = getattr(manager, "configure_lora_checkpoint", None)
        if not callable(configure):
            raise web.HTTPNotImplemented(text="this backend cannot switch LoRA weights")

        previous = lora_state["selected"]
        previous_path = (
            None
            if previous is None
            else runtime_paths.model_dir / "loras" / previous
        )
        selected_path = runtime_paths.model_dir / "loras" / requested
        lora_state["changing"] = True
        engine_state["switching"] = True
        async with engine_lock:
            try:
                await manager.stop()
                configure(selected_path)
                if launcher is not None:
                    await manager.preload(launcher)
                    if manager.warm_state.get("status") != "ready":
                        raise RuntimeError("selected LoRA failed to build")
                _persist_lora_selection(paths.data_dir, requested)
                lora_state["selected"] = requested
            except Exception as error:
                try:
                    await manager.stop()
                    if previous_path is not None:
                        configure(previous_path)
                    if launcher is not None:
                        await manager.preload(launcher)
                finally:
                    lora_state["selected"] = previous
                raise web.HTTPInternalServerError(
                    text="LoRA change failed; the previous checkpoint was restored"
                ) from error
            finally:
                engine_state["switching"] = False
                lora_state["changing"] = False
        response_data = lora_settings_payload()
        response_data["changed"] = True
        return web.json_response(response_data)

    async def get_reference_media_settings(_request: web.Request) -> web.Response:
        return web.json_response({
            **reference_media_state,
            "levels": list(REFERENCE_MEDIA_RESOLUTIONS),
            "preserve_aspect_ratio": True,
            "preserve_composition": True,
            "preserve_duration": True,
            "crop": False,
            "stretch": False,
            "pad_user_media": False,
            "upscale_small_inputs": False,
            "internal_vae_alignment": "private_replicated_edge_padding_to_32px",
        })

    async def configure_reference_media_settings(
        request: web.Request,
    ) -> web.Response:
        try:
            document = await request.json()
        except (json.JSONDecodeError, ValueError) as error:
            raise web.HTTPBadRequest(text="request body must be JSON") from error
        if not isinstance(document, dict):
            raise web.HTTPBadRequest(text="request body must be an object")
        image_resolution = str(
            document.get("image_resolution", reference_media_state["image_resolution"])
        ).strip().lower()
        video_resolution = str(
            document.get("video_resolution", reference_media_state["video_resolution"])
        ).strip().lower()
        if image_resolution not in REFERENCE_MEDIA_RESOLUTIONS:
            raise web.HTTPBadRequest(
                text="image_resolution must be original, 360p, 480p or 720p"
            )
        if video_resolution not in REFERENCE_MEDIA_RESOLUTIONS:
            raise web.HTTPBadRequest(
                text="video_resolution must be original, 360p, 480p or 720p"
            )
        updated = {
            "image_resolution": image_resolution,
            "video_resolution": video_resolution,
        }
        try:
            _persist_reference_media_settings(paths.data_dir, updated)
        except OSError as error:
            raise web.HTTPInternalServerError(
                text="failed to save reference-media settings"
            ) from error
        reference_media_state.update(updated)
        return await get_reference_media_settings(request)

    async def get_face_repair_settings(_request: web.Request) -> web.Response:
        return web.json_response({
            **face_repair_state,
            "canvas_range": {"min": 192, "max": 1088, "step": 32},
            "capacities": [1, 4, 9, 16],
            "steps": 4,
            "minimum_magnification": 1.5,
            "acceleration_range": {"min": 0.0, "max": 100.0},
        })

    async def configure_face_repair_settings(
        request: web.Request,
    ) -> web.Response:
        try:
            document = await request.json()
        except (json.JSONDecodeError, ValueError) as error:
            raise web.HTTPBadRequest(text="request body must be JSON") from error
        if not isinstance(document, dict):
            raise web.HTTPBadRequest(text="request body must be an object")
        try:
            validated = VideoRepairSpec.from_mapping({
                "canvas_size": document.get(
                    "canvas_size", face_repair_state["canvas_size"]
                ),
                "capacity": document.get(
                    "capacity", face_repair_state["capacity"]
                ),
            })
        except ContractError as error:
            raise web.HTTPBadRequest(text=str(error)) from error
        updated = {
            "canvas_size": validated.canvas_size,
            "capacity": validated.max_faces,
        }
        try:
            _persist_face_repair_settings(paths.data_dir, updated)
        except OSError as error:
            raise web.HTTPInternalServerError(
                text="failed to save face-repair settings"
            ) from error
        face_repair_state.update(updated)
        return await get_face_repair_settings(request)

    async def get_checkpoint_preview_settings(
        _request: web.Request,
    ) -> web.Response:
        return web.json_response({
            **checkpoint_preview_state,
            "step_range": {"min": 1, "max": 4},
            "resolutions": list(CHECKPOINT_PREVIEW_SETTING_RESOLUTIONS),
        })

    async def configure_checkpoint_preview_settings(
        request: web.Request,
    ) -> web.Response:
        try:
            document = await request.json()
        except (json.JSONDecodeError, ValueError) as error:
            raise web.HTTPBadRequest(text="request body must be JSON") from error
        if not isinstance(document, dict):
            raise web.HTTPBadRequest(text="request body must be an object")
        try:
            steps = int(document.get("steps", checkpoint_preview_state["steps"]))
        except (TypeError, ValueError) as error:
            raise web.HTTPBadRequest(text="steps must be an integer") from error
        resolution = str(
            document.get("resolution", checkpoint_preview_state["resolution"])
        ).strip().lower()
        if not 1 <= steps <= 4:
            raise web.HTTPBadRequest(text="steps must be between 1 and 4")
        if resolution not in CHECKPOINT_PREVIEW_SETTING_RESOLUTIONS:
            raise web.HTTPBadRequest(
                text="resolution must be 360p, 480p or 720p"
            )
        updated = {"steps": steps, "resolution": resolution}
        try:
            _persist_checkpoint_preview_settings(paths.data_dir, updated)
        except OSError as error:
            raise web.HTTPInternalServerError(
                text="failed to save checkpoint-preview settings"
            ) from error
        checkpoint_preview_state.update(updated)
        return await get_checkpoint_preview_settings(request)

    async def get_second_sampling_window_settings(
        _request: web.Request,
    ) -> web.Response:
        from .native_engine.global_co_denoise import (
            window_geometry_for_seconds,
        )

        frames, stride = window_geometry_for_seconds(
            second_sampling_window_state["window_seconds"],
            second_sampling_window_state["overlap_seconds"],
        )
        return web.json_response({
            **second_sampling_window_state,
            "seconds_range": {"min": 3.0, "max": 15.0, "step": 0.5},
            "overlap_seconds_range": {"min": 0.0, "max": 4.0, "step": 0.1},
            "effective_window_frames": frames,
            "effective_window_seconds": round(frames / 24.0, 3),
            "effective_stride_frames": stride,
            "effective_overlap_frames": frames - stride,
            "effective_overlap_seconds": round((frames - stride) / 24.0, 3),
        })

    async def configure_second_sampling_window_settings(
        request: web.Request,
    ) -> web.Response:
        try:
            document = await request.json()
        except (json.JSONDecodeError, ValueError) as error:
            raise web.HTTPBadRequest(text="request body must be JSON") from error
        if not isinstance(document, dict):
            raise web.HTTPBadRequest(text="request body must be an object")
        enabled = document.get(
            "enabled", second_sampling_window_state["enabled"]
        )
        if not isinstance(enabled, bool):
            raise web.HTTPBadRequest(text="enabled must be a boolean")
        try:
            window_seconds = float(document.get(
                "window_seconds",
                second_sampling_window_state["window_seconds"],
            ))
            overlap_seconds = float(document.get(
                "overlap_seconds",
                second_sampling_window_state["overlap_seconds"],
            ))
        except (TypeError, ValueError) as error:
            raise web.HTTPBadRequest(
                text="window_seconds must be numeric"
            ) from error
        if not math.isfinite(window_seconds) or not 3.0 <= window_seconds <= 15.0:
            raise web.HTTPBadRequest(
                text="window_seconds must be between 3 and 15"
            )
        if not math.isfinite(overlap_seconds) or not 0.0 <= overlap_seconds <= 4.0:
            raise web.HTTPBadRequest(
                text="overlap_seconds must be between 0 and 4"
            )
        updated = {
            "enabled": enabled,
            "window_seconds": round(window_seconds, 1),
            "overlap_seconds": round(overlap_seconds, 1),
        }
        try:
            _persist_second_sampling_window_settings(
                paths.data_dir, updated
            )
        except OSError as error:
            raise web.HTTPInternalServerError(
                text="failed to save second-sampling window settings"
            ) from error
        second_sampling_window_state.update(updated)
        return await get_second_sampling_window_settings(request)

    async def get_generation_limit_settings(_request: web.Request) -> web.Response:
        document = generation_limit_state["policy"].public(
            generation_limit_state["detected_vram_gib"]
        )
        return web.json_response(document)

    async def configure_generation_limit_settings(
        request: web.Request,
    ) -> web.Response:
        try:
            document = await request.json()
        except (json.JSONDecodeError, ValueError) as error:
            raise web.HTTPBadRequest(text="request body must be JSON") from error
        if not isinstance(document, dict):
            raise web.HTTPBadRequest(text="request body must be an object")
        try:
            policy = GenerationLimitPolicy(
                preset_limits=document.get("preset_limits"),
            )
            persist_generation_limit_policy(paths.data_dir, policy)
        except (TypeError, ValueError) as error:
            raise web.HTTPBadRequest(text=str(error)) from error
        except OSError as error:
            raise web.HTTPInternalServerError(
                text="failed to save generation-limit settings"
            ) from error
        generation_limit_state["policy"] = policy
        # Refresh on save so a changed GPU/driver does not require restart.
        generation_limit_state["detected_vram_gib"] = detect_gpu_vram_gib()
        return await get_generation_limit_settings(request)

    async def list_jobs(request: web.Request) -> web.Response:
        try:
            limit = min(100, max(1, int(request.query.get("limit", "30"))))
        except ValueError as error:
            raise web.HTTPBadRequest(text="limit must be an integer") from error
        running = sorted(
            (
                job for job in service.jobs.values()
                if job.status in {"starting_backend", "running", "awaiting_preview"}
            ),
            key=lambda job: job.created_at,
        )
        queued = [
            service.jobs[job_id]
            for job_id in service.pending
            if job_id in service.jobs
        ]
        history = sorted(
            (
                job for job in service.jobs.values()
                if job.status not in {"queued", "starting_backend", "running", "awaiting_preview"}
            ),
            key=lambda job: job.created_at,
            reverse=True,
        )
        jobs = (running + queued + history)[:limit]
        return web.json_response({"jobs": [service.serialize(job) for job in jobs]})

    async def reorder_jobs(request: web.Request) -> web.Response:
        try:
            document = await request.json()
            ordered_ids = document.get("job_ids") if isinstance(document, dict) else None
            if not isinstance(ordered_ids, list) or not all(
                isinstance(item, str) for item in ordered_ids
            ):
                raise ContractError("job_ids must be a JSON string array")
            await service.reorder(ordered_ids)
        except (ContractError, json.JSONDecodeError) as error:
            raise web.HTTPBadRequest(text=str(error)) from error
        return web.json_response({"job_ids": list(service.pending)})

    def require_job(request: web.Request) -> JobRecord:
        job = service.jobs.get(request.match_info["job_id"])
        if job is None:
            raise web.HTTPNotFound(text="job not found")
        return job

    async def get_job(request: web.Request) -> web.Response:
        return web.json_response(service.serialize(require_job(request)))

    async def cancel_job(request: web.Request) -> web.Response:
        job = require_job(request)
        await service.cancel(job.id)
        return web.json_response(service.serialize(job))

    async def resume_job(request: web.Request) -> web.Response:
        job = require_job(request)
        engine = active_engine()
        if engine is None:
            raise web.HTTPConflict(
                text="select and finish loading a service family before resuming"
            )
        if job.spec.service_family != engine:
            raise web.HTTPConflict(
                text=(
                    "the checkpoint belongs to the "
                    f"{job.spec.service_family} service family; switch back before resuming"
                )
            )
        if job.spec.runtime_launcher != active_launcher():
            raise web.HTTPConflict(
                text=(
                    "the checkpoint belongs to the "
                    f"{job.spec.runtime_launcher} weight launcher; switch back first"
                )
            )
        try:
            await service.resume(job.id)
        except ContractError as error:
            raise web.HTTPConflict(text=str(error)) from error
        return web.json_response(service.serialize(job), status=202)

    async def second_sample_job(request: web.Request) -> web.Response:
        source = require_job(request)
        try:
            document = await request.json()
            if not isinstance(document, dict):
                raise ContractError("request body must be a JSON object")
            document = dict(document)
            if (
                second_sampling_window_state["enabled"]
                and "temporal_window_frames" not in document
            ):
                from .native_engine.global_co_denoise import (
                    window_geometry_for_seconds,
                )

                window_frames, stride_frames = window_geometry_for_seconds(
                    second_sampling_window_state["window_seconds"],
                    second_sampling_window_state["overlap_seconds"],
                )
                document["temporal_window_frames"] = window_frames
                document.setdefault(
                    "temporal_overlap_frames", window_frames - stride_frames
                )
            second_sampling = SecondSamplingSpec.from_mapping(
                document, source=source.spec
            )
            if second_sampling.method == "h3":
                engine = active_engine()
                if engine is None:
                    raise ContractError(
                        "select and finish loading a service family before H3 second sampling"
                    )
                if source.spec.service_family != engine:
                    raise ContractError(
                        "the source card belongs to the "
                        f"{source.spec.service_family} service family; switch back first"
                    )
                if source.spec.weight_tier != engine_state["weight_tier"]:
                    raise ContractError(
                        "the source card weight tier does not match the active launcher"
                    )
                launcher_definition = LAUNCHER_DEFINITIONS[
                    active_launcher(required=True)
                ]
                maximum_second_edge = max(
                    second_sampling_short_edge(level)[1]
                    for level in launcher_definition.backend.second_sampling_levels
                )
                requested_second_edge = second_sampling_short_edge(
                    second_sampling.resolution
                )[1]
                if requested_second_edge > maximum_second_edge:
                    raise ContractError(
                        f"the {engine_state['vram_profile']} backend does not support "
                        f"H3 second sampling above {maximum_second_edge}p"
                    )
            job = await service.submit_second_sampling(source, second_sampling)
        except (ContractError, json.JSONDecodeError) as error:
            raise web.HTTPBadRequest(text=str(error)) from error
        return web.json_response(service.serialize(job), status=202)

    async def video_repair_job(request: web.Request) -> web.Response:
        source = require_job(request)
        try:
            document = await request.json()
            if not isinstance(document, dict):
                raise ContractError("request body must be a JSON object")
            if source.spec.service_family != "first_last":
                raise ContractError(
                    "face repair is available only for completed FL2VA source jobs"
                )
            repair = VideoRepairSpec.from_mapping({
                **face_repair_state,
                **document,
            })
            engine = active_engine()
            if engine is None:
                raise ContractError(
                    "select and finish loading a service family before video repair"
                )
            if source.spec.service_family != engine:
                raise ContractError(
                    "the source card belongs to the "
                    f"{source.spec.service_family} service family; switch back first"
                )
            if source.spec.weight_tier != engine_state["weight_tier"]:
                raise ContractError(
                    "the source card weight tier does not match the active launcher"
                )
            job = await service.submit_video_repair(source, repair)
        except (ContractError, json.JSONDecodeError) as error:
            raise web.HTTPBadRequest(text=str(error)) from error
        return web.json_response(service.serialize(job), status=202)

    async def decide_preview(request: web.Request) -> web.Response:
        job = require_job(request)
        decision = request.match_info["decision"]
        try:
            await service.decide_preview(job.id, decision)
        except ContractError as error:
            raise web.HTTPConflict(text=str(error)) from error
        return web.json_response(service.serialize(job))

    async def get_preview(request: web.Request) -> web.StreamResponse:
        job = require_job(request)
        if job.preview_path is None or not job.preview_path.is_file():
            raise web.HTTPConflict(text="preview is not ready")
        return web.FileResponse(
            job.preview_path,
            headers={"Content-Disposition": f'inline; filename="{job.preview_path.name}"'},
        )

    async def delete_job(request: web.Request) -> web.Response:
        job = require_job(request)
        try:
            deletion = await service.delete(job.id)
        except ContractError as error:
            raise web.HTTPConflict(text=str(error)) from error
        return web.json_response({"deleted": True, "id": job.id, **deletion})

    async def delete_job_records(request: web.Request) -> web.Response:
        """Delete several history records through the same guarded path as one.

        Deletion is intentionally best-effort: a stale id or a record that became
        active does not prevent the remaining selected records from being removed.
        The response reports every failure so clients never present a partial
        deletion as complete success.
        """
        try:
            document = await request.json()
        except (json.JSONDecodeError, ValueError) as error:
            raise web.HTTPBadRequest(text="request body must be JSON") from error
        job_ids = document.get("job_ids") if isinstance(document, dict) else None
        if (
            not isinstance(job_ids, list)
            or not job_ids
            or len(job_ids) > 100
            or not all(isinstance(item, str) and item.strip() for item in job_ids)
        ):
            raise web.HTTPBadRequest(
                text="job_ids must be a non-empty JSON string array with at most 100 items"
            )
        unique_ids = list(dict.fromkeys(item.strip() for item in job_ids))
        deleted_ids: list[str] = []
        errors: list[dict[str, str]] = []
        deletion_results: dict[str, dict[str, bool]] = {}
        for job_id in unique_ids:
            if job_id not in service.jobs:
                errors.append({"id": job_id, "error": "job not found"})
                continue
            try:
                deletion_results[job_id] = await service.delete(job_id)
            except ContractError as error:
                errors.append({"id": job_id, "error": str(error)})
            else:
                deleted_ids.append(job_id)
        return web.json_response({
            "requested_count": len(unique_ids),
            "deleted_count": len(deleted_ids),
            "deleted_ids": deleted_ids,
            "errors": errors,
            "results": deletion_results,
        })

    async def clear_latent_cache(request: web.Request) -> web.Response:
        try:
            result = await service.clear_latent_cache()
        except ContractError as error:
            raise web.HTTPConflict(text=str(error)) from error
        return web.json_response({"cleared": True, **result})

    async def get_video(request: web.Request) -> web.StreamResponse:
        job = require_job(request)
        if job.status != "succeeded" or job.output_path is None:
            raise web.HTTPConflict(text="video is not ready")
        return web.FileResponse(
            job.output_path,
            headers={"Content-Disposition": f'inline; filename="{job.output_path.name}"'},
        )

    async def on_startup(_: web.Application) -> None:
        await service.start(fixed_launcher, preload=preload)
        for project in infinite_store.projects.values():
            if project.batch_status == "running":
                launch_infinite_batch(project.id)
        # On roomy hosts, load the immutable temporal weights into CPU RAM
        # after H3 is ready.  A task then pays only the measured 17--20 second
        # GPU/encode path.  Smaller hosts retain safe lazy loading.
        production_temporal_preload = bool(
            not legacy_upscaler_injected
            and isinstance(video_upscaler, FlashVSRUpscaler)
            and memory_state["capacity"].effective_limit_gib >= 72.0
        )
        if (
            hasattr(video_upscaler, "start")
            and (
                production_temporal_preload
                or (
                    preload
                    and legacy_upscaler_injected
                    and memory_state["profile"].preload_upscaler
                )
            )
        ):
            async def preload_upscaler() -> None:
                try:
                    # Prioritise the generation engine and avoid competing for
                    # disk/RAM bandwidth during its initial weight load.
                    if service.warmup_task is not None:
                        await service.warmup_task
                    await video_upscaler.start()
                except Exception:
                    # Upscaling is optional. Keep H3 generation available and
                    # expose the failure through /healthz and the daemon log.
                    pass

            app["flashvsr_preload_task"] = asyncio.create_task(
                preload_upscaler(), name="flashvsr-cpu-preload"
            )

    async def on_cleanup(_: web.Application) -> None:
        batch_tasks = list(infinite_batch_tasks.values())
        infinite_batch_tasks.clear()
        for batch_task in batch_tasks:
            if not batch_task.done():
                batch_task.cancel()
        if batch_tasks:
            await asyncio.gather(*batch_tasks, return_exceptions=True)
        task = app.get("flashvsr_preload_task")
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        await service.close()
        budget_controller.close()

    app.router.add_get("/", index)
    app.router.add_get("/openapi.json", openapi)
    app.router.add_static("/static", serve_dir / "static", show_index=False)
    app.router.add_get("/healthz", health)
    app.router.add_get("/readyz", readiness)
    app.router.add_get("/api/v1/options", options)
    app.router.add_get("/api/v1/resources", resources)
    app.router.add_get("/api/v1/models", models)
    app.router.add_get("/api/v1/workspace/browse", browse_workspace)
    app.router.add_put("/api/v1/workspace", select_workspace)
    app.router.add_put("/api/v1/engine", select_engine)
    app.router.add_delete("/api/v1/engine", exit_engine)
    app.router.add_put("/api/v1/memory-profile", change_memory_profile)
    app.router.add_post("/api/v1/generations", create_generation)
    app.router.add_post("/api/v1/long-video/preview", preview_long_video)
    app.router.add_get("/api/v1/infinite-projects", list_infinite_projects)
    app.router.add_post("/api/v1/infinite-projects", create_infinite_project)
    app.router.add_get(
        "/api/v1/infinite-projects/{project_id}", get_infinite_project
    )
    app.router.add_delete(
        "/api/v1/infinite-projects/{project_id}", delete_infinite_project
    )
    app.router.add_post(
        "/api/v1/infinite-projects/{project_id}/windows",
        append_infinite_window,
    )
    app.router.add_post(
        "/api/v1/infinite-projects/{project_id}/batch",
        start_infinite_batch,
    )
    app.router.add_delete(
        "/api/v1/infinite-projects/{project_id}/batch",
        stop_infinite_batch,
    )
    app.router.add_delete(
        "/api/v1/infinite-projects/{project_id}/windows/last",
        discard_infinite_tail,
    )
    app.router.add_post(
        "/api/v1/infinite-projects/{project_id}/second-sampling",
        second_sample_infinite_project,
    )
    app.router.add_post(
        "/api/v1/infinite-projects/{project_id}/final-sampling",
        second_sample_infinite_project,
    )
    app.router.add_get("/api/v1/settings/lora", get_lora_settings)
    app.router.add_put("/api/v1/settings/lora", configure_lora_settings)
    app.router.add_get(
        "/api/v1/settings/reference-media", get_reference_media_settings
    )
    app.router.add_put(
        "/api/v1/settings/reference-media", configure_reference_media_settings
    )
    app.router.add_get(
        "/api/v1/settings/face-repair", get_face_repair_settings
    )
    app.router.add_put(
        "/api/v1/settings/face-repair", configure_face_repair_settings
    )
    app.router.add_get(
        "/api/v1/settings/checkpoint-preview", get_checkpoint_preview_settings
    )
    app.router.add_put(
        "/api/v1/settings/checkpoint-preview",
        configure_checkpoint_preview_settings,
    )
    app.router.add_get(
        "/api/v1/settings/second-sampling-window",
        get_second_sampling_window_settings,
    )
    app.router.add_put(
        "/api/v1/settings/second-sampling-window",
        configure_second_sampling_window_settings,
    )
    app.router.add_get(
        "/api/v1/settings/generation-limits", get_generation_limit_settings
    )
    app.router.add_put(
        "/api/v1/settings/generation-limits", configure_generation_limit_settings
    )
    app.router.add_get("/api/v1/jobs", list_jobs)
    app.router.add_delete("/api/v1/cache/latents", clear_latent_cache)
    app.router.add_put("/api/v1/jobs/order", reorder_jobs)
    # Register the collection route before the variable job-id route.
    app.router.add_delete("/api/v1/jobs/records", delete_job_records)
    app.router.add_get("/api/v1/jobs/{job_id}", get_job)
    app.router.add_delete("/api/v1/jobs/{job_id}", cancel_job)
    app.router.add_post("/api/v1/jobs/{job_id}/resume", resume_job)
    app.router.add_post(
        "/api/v1/jobs/{job_id}/second-sampling", second_sample_job
    )
    app.router.add_post(
        "/api/v1/jobs/{job_id}/video-repair", video_repair_job
    )
    app.router.add_post("/api/v1/jobs/{job_id}/preview/{decision}", decide_preview)
    app.router.add_get("/api/v1/jobs/{job_id}/preview", get_preview)
    app.router.add_delete("/api/v1/jobs/{job_id}/record", delete_job)
    app.router.add_get("/api/v1/jobs/{job_id}/video", get_video)
    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)
    return app


def parse_args() -> argparse.Namespace:
    serve_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="H3 4090 accelerated video service")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument(
        "--engine",
        choices=(
            *MODEL_LAUNCHERS, *LEGACY_MODEL_LAUNCHERS,
            *SERVICE_FAMILIES, *ENGINES,
        ),
        default=os.environ.get("H3_SERVE_ENGINE", "first_last"),
        help="fix this process to one model family (legacy route aliases are accepted)",
    )
    parser.add_argument(
        "--unified-console",
        action="store_true",
        help="start idle and let the operator enter/exit one engine from the Web console",
    )
    parser.add_argument("--release-root", type=Path, default=serve_dir)
    parser.add_argument("--data-dir", type=Path, default=serve_dir / "data")
    parser.add_argument("--api-key", default=os.environ.get("H3_SERVE_API_KEY"))
    parser.add_argument("--max-queued-jobs", type=int, default=32)
    parser.add_argument(
        "--memory-profile",
        choices=("auto", *HOST_MEMORY_PROFILES),
        default=os.environ.get("H3_SERVE_MEMORY_PROFILE", "auto"),
        help="host-RAM residency policy; auto selects the fastest safe tier",
    )
    parser.add_argument(
        "--lazy-load",
        action="store_true",
        help="defer fixed-engine model construction until the first job",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    serve_dir = Path(__file__).resolve().parents[1]
    pid_file_value = os.environ.get("H3_SERVE_PID_FILE")

    def remove_own_pid_file() -> None:
        if not pid_file_value:
            return
        pid_file = Path(pid_file_value)
        try:
            if pid_file.read_text(encoding="utf-8").strip() == str(os.getpid()):
                pid_file.unlink(missing_ok=True)
        except OSError:
            pass

    atexit.register(remove_own_pid_file)
    if args.host not in {"127.0.0.1", "localhost", "::1"} and not args.api_key:
        raise SystemExit("H3_SERVE_API_KEY is required when listening beyond localhost")
    # Reserve the listening endpoint before constructing the application.
    # Application startup launches the large model preload in a worker thread;
    # discovering EADDRINUSE afterwards leaves Python waiting for that thread
    # and CUDA pinned-memory finalizers, which makes Ctrl-C appear ineffective.
    family = socket.AF_INET6 if ":" in args.host else socket.AF_INET
    listen_socket = socket.socket(family, socket.SOCK_STREAM)
    listen_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        listen_socket.bind((args.host, args.port))
        listen_socket.listen(128)
        listen_socket.setblocking(False)
    except OSError as error:
        listen_socket.close()
        if error.errno == errno.EADDRINUSE:
            raise SystemExit(
                f"Port {args.port} is already in use. Stop the existing service, "
                f"or start this one with H3_SERVE_PORT={args.port + 1}."
            ) from None
        raise
    paths = ServicePaths.defaults(args.release_root, data_dir=args.data_dir)
    memory_status = detect_host_memory()
    try:
        memory_profile = resolve_host_memory_profile(
            args.memory_profile, memory_status
        )
    except (RuntimeError, ValueError) as error:
        raise SystemExit(str(error)) from error
    print(
        f"Host memory profile: {memory_profile.label} "
        f"(effective {memory_status.effective_limit_gib:.1f} GiB, "
        f"available {memory_status.available_gib:.1f} GiB)",
        flush=True,
    )
    try:
        memory_budget_controller = LinuxCgroupMemoryBudgetController()
    except RuntimeError as error:
        raise SystemExit(
            f"Hard host-memory enforcement is unavailable: {error}"
        ) from error
    try:
        app = create_app(
            paths=paths,
            serve_dir=serve_dir,
            api_key=args.api_key,
            max_queued_jobs=args.max_queued_jobs,
            fixed_engine=None if args.unified_console else args.engine,
            preload=False if args.unified_console else not args.lazy_load,
            memory_profile=memory_profile,
            memory_budget_controller=memory_budget_controller,
            host_memory_status=memory_status,
        )
        web.run_app(app, sock=listen_socket)
    finally:
        listen_socket.close()


if __name__ == "__main__":
    main()
