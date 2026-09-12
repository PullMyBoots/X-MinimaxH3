"""Persistent tail-editable projects for incremental H3 video creation.

The product contract deliberately separates physical generation windows from
semantic shots.  Every physical boundary is a continuation.  A user-authored
camera cut, when wanted, lives inside ``window_description`` and is therefore
handled by one native H3 sample instead of by the continuation transport.
"""

from __future__ import annotations

from dataclasses import MISSING, asdict, dataclass, field
import json
import math
from pathlib import Path
import time
import uuid
from typing import Any

from .contract import (
    ASPECT_RATIOS,
    ContractError,
    generation_short_edge,
    progressive_short_edge,
    public_resolution_name,
)


FPS = 24
H3_FRAME_ORIGIN = 5
H3_FRAME_STRIDE = 17
MAX_CONTEXT_FRAMES = 90
RETRYABLE_TAIL_STATUSES = frozenset({"failed", "cancelled", "missing"})
VISUAL_MEMORY_RESOLUTIONS = ("360p", "480p", "720p", "original")
DEFAULT_VISUAL_MEMORY_CAPACITY = 6
DEFAULT_AUDIO_MEMORY_CAPACITY = 1
DEFAULT_VISUAL_MEMORY_RESOLUTION = "360p"
DEFAULT_AUDIO_MEMORY_TICKS = 80


def memory_capacity(
    amount: int | None = None,
    *,
    service_family: str = "first_last",
    visual_capacity: int | None = None,
    audio_capacity: int | None = None,
    visual_resolution: str | None = None,
) -> dict[str, int | bool | float | str]:
    """Resolve the split infinite-creation AV memory contract.

    ``amount`` remains accepted only for persisted projects and API clients
    from the former coupled 0..100 slider.  New callers provide independent
    visual/audio slot counts and a visual reference resolution.
    """

    if visual_capacity is not None or audio_capacity is not None:
        video_slots = int(
            DEFAULT_VISUAL_MEMORY_CAPACITY
            if visual_capacity is None else visual_capacity
        )
        audio_slots = int(
            DEFAULT_AUDIO_MEMORY_CAPACITY
            if audio_capacity is None else audio_capacity
        )
        resolution = str(
            visual_resolution or DEFAULT_VISUAL_MEMORY_RESOLUTION
        ).strip().lower()
        if not 0 <= video_slots <= 24:
            raise ContractError("visual_memory_capacity must be between 0 and 24")
        if not 0 <= audio_slots <= 3:
            raise ContractError("audio_memory_capacity must be between 0 and 3")
        if resolution not in VISUAL_MEMORY_RESOLUTIONS:
            raise ContractError(
                "visual_memory_resolution must be 360p, 480p, 720p or original"
            )
        return {
            "enabled": bool(video_slots or audio_slots),
            "video_slots": video_slots,
            "audio_slots": audio_slots,
            # Automatic voice memories are deliberately short. Capacity now
            # means independently addressable excerpts, not excerpt duration.
            "audio_ticks_per_clip": DEFAULT_AUDIO_MEMORY_TICKS,
            "audio_seconds": (
                DEFAULT_AUDIO_MEMORY_TICKS / 40 if audio_slots else 0.0
            ),
            "visual_resolution": resolution,
        }

    value = 60 if amount is None else int(amount)
    if not 0 <= value <= 100:
        raise ContractError("memory must be an integer from 0 to 100")
    if value == 0:
        return {
            "enabled": False,
            "video_slots": 0,
            "audio_slots": 0,
            "audio_ticks_per_clip": 0,
            "audio_seconds": 0.0,
            "visual_resolution": "original",
        }
    maximum_audio_ticks = 240 if service_family == "reference" else 120
    minimum_audio_ticks = 80 if service_family == "reference" else 20
    audio_ticks = minimum_audio_ticks + math.floor(
        (maximum_audio_ticks - minimum_audio_ticks) * (value - 1) / 99
    )
    return {
        "enabled": True,
        "video_slots": max(1, math.ceil(9 * value / 100)),
        "audio_slots": 1,
        "audio_ticks_per_clip": audio_ticks,
        "audio_seconds": audio_ticks / 40,
        "visual_resolution": "original",
    }


def align_h3_frames(frame_count: int) -> int:
    requested = max(H3_FRAME_ORIGIN, int(frame_count))
    index = max(0, round((requested - H3_FRAME_ORIGIN) / H3_FRAME_STRIDE))
    return H3_FRAME_ORIGIN + H3_FRAME_STRIDE * index


def context_frames_for_seconds(seconds: float) -> int:
    value = float(seconds)
    if not math.isfinite(value) or not 0.0 <= value <= 4.0:
        raise ContractError("overlap_seconds must be between 0 and 4")
    if value == 0:
        return 0
    return min(MAX_CONTEXT_FRAMES, align_h3_frames(round(value * FPS)))


def visible_frames_for_seconds(seconds: float, *, maximum_physical_frames: int, context_frames: int) -> int:
    """Resolve an append duration to whole H3 temporal strides.

    An opening owns ``5 + 17*k`` frames.  Every append contributes ``17*k``
    new frames because its physical target also contains a ``5 + 17*k``
    continuation prefix.  This keeps the cumulative latent on the native H3
    temporal grid after every accepted append.
    """

    value = float(seconds)
    if not math.isfinite(value) or value < H3_FRAME_STRIDE / FPS:
        raise ContractError("window duration must be at least 0.708 seconds")
    maximum = int(maximum_physical_frames) - int(context_frames)
    units = max(1, round(value * FPS / H3_FRAME_STRIDE))
    visible = units * H3_FRAME_STRIDE
    if visible > maximum:
        raise ContractError(
            f"window duration plus overlap exceeds the native window limit; "
            f"maximum new duration is {maximum / FPS:.3f} seconds"
        )
    return visible


def compile_infinite_prompt(
    *,
    overview: str,
    window_description: str,
    overall_soundscape: str,
    non_diegetic_music: str,
    continuation: bool,
    context_frames: int = 0,
    reference_image_count: int = 0,
    reference_audio_count: int = 0,
    first_frame: bool = False,
    refinement: bool = False,
) -> str:
    overview = str(overview).strip()
    action = str(window_description).strip()
    sound = str(overall_soundscape).strip() or "N/A"
    music = str(non_diegetic_music).strip() or "N/A"
    if not action:
        raise ContractError("window_description is required")
    # Workflow v3 authors one independent prompt per window.  Accept a full
    # native H3 three-section prompt directly as well as the shorter legacy
    # action-only form.  The service only injects the physical continuation
    # contract; it never requires a hidden project-level overview.
    base_markers = (
        "integrated_multimodal_description:",
        "overall_soundscape:",
        "non_diegetic_music:",
    )
    marker_offsets = tuple(action.find(marker) for marker in base_markers)
    prompt_instruction = ""
    if (
        all(offset >= 0 for offset in marker_offsets)
        and marker_offsets == tuple(sorted(marker_offsets))
    ):
        description_start = marker_offsets[0] + len(base_markers[0])
        sound_start = marker_offsets[1] + len(base_markers[1])
        music_start = marker_offsets[2] + len(base_markers[2])
        prompt_instruction = action[:marker_offsets[0]].strip()
        embedded_description = action[description_start:marker_offsets[1]].strip()
        embedded_sound = action[sound_start:marker_offsets[2]].strip()
        embedded_music = action[music_start:].strip()
        if not embedded_description:
            raise ContractError("integrated_multimodal_description is required")
        action = embedded_description
        sound = embedded_sound or "N/A"
        music = embedded_music or "N/A"
    boundary = (
        "This is a low-noise refinement of the complete accepted source AV latent. "
        "Preserve its existing temporal trajectory and audiovisual events exactly."
        if refinement else
        "This is a strict continuation from the exact carried video and audio context. "
        "Continue the visible camera position, lens, direction, velocity, character pose, "
        "scene layout and ambient sound without a reset, transition or accidental cut. "
        f"The first {context_frames / FPS:.3f} seconds are carried context and are not new story time. "
        "The current-window story clock starts at 00:00.000 immediately after that carried prefix; "
        "all authored timestamps refer to this new story time. "
        "A camera cut may occur only if the current-window instructions explicitly place one "
        "inside this window."
        if continuation else
        "This is the opening window. Establish the world directly from the complete "
        "current-window instructions. A camera cut may occur only if explicitly requested."
    )
    action = action if "[Shot 1]" in action else f"[Shot 1] {action}"
    timeline_label = (
        "Complete accepted source timeline"
        if refinement else
        "Current new-story window; its local timeline starts at 00:00.000"
    )
    continuity = f"[Overall continuity] {overview}\n" if overview else ""
    description = (
        f"{continuity}[Physical window boundary] {boundary}\n"
        f"[{timeline_label}] {action}"
    )
    image_count = int(reference_image_count)
    audio_count = int(reference_audio_count)
    if min(image_count, audio_count) < 0:
        raise ContractError("reference counts cannot be negative")
    if image_count or audio_count:
        definitions = [
            f"<Picture {index}> is a creator-supplied visual reference used "
            "where it is explicitly cited in the target description."
            for index in range(1, image_count + 1)
        ] + [
            f"<Audio {index}> is a creator-supplied voice-timbre or sound "
            "reference used where it is explicitly cited in the target description."
            for index in range(1, audio_count + 1)
        ]
        retention = [
            f"<Picture {index}> (creator-directed reference): fully_preserved - "
            "preserve the explicitly cited identity, scene layout, composition, or style."
            for index in range(1, image_count + 1)
        ] + [
            f"<Audio {index}>: reference - preserve the explicitly cited timbre "
            "or sound characteristics without copying unrelated words or events."
            for index in range(1, audio_count + 1)
        ]
        task_types = ["reference generation"]
        if audio_count:
            task_types.append("audio reference")
        summary = overview or "Creator-authored current window."
        prompt = (
            "subject_definitions:\n"
            + "\n".join(definitions)
            + "\n\n"
            "summary:\n"
            f"[{' + '.join(task_types)}] {summary}\n\n"
            "retention_analysis:\n"
            + "\n".join(retention)
            + "\n\n"
            "detailed_description:\n"
            f"{description}\n\n"
            "overall_soundscape:\n"
            f"{sound}\n\n"
            "non_diegetic_music:\n"
            f"{music}"
        )
    else:
        first_frame_instruction = (
            "For the target video, at 0.00 seconds into the target video, "
            "<Picture 1> (from [Shot 1]) is fully referenced.\n\n"
            if first_frame else ""
        )
        leading_instruction = (
            f"{prompt_instruction}\n\n" if prompt_instruction else first_frame_instruction
        )
        prompt = (
            f"{leading_instruction}"
            "integrated_multimodal_description: "
            f"{description}\n\n"
            "overall_soundscape: "
            f"{sound}\n\n"
            "non_diegetic_music: "
            f"{music}"
        )
    if len(prompt) > 20_000:
        raise ContractError("compiled window prompt is too long")
    return prompt


@dataclass(frozen=True, slots=True)
class InfiniteContinuationSpec:
    project_id: str
    window_index: int
    source_job_id: str
    source_frames: int
    context_frames: int
    visible_frames: int
    audio_bridge_ticks: int
    memory: int
    source_dialogue: bool = False
    visual_memory_capacity: int | None = None
    audio_memory_capacity: int | None = None
    visual_memory_resolution: str | None = None

    @property
    def physical_frames(self) -> int:
        return self.hidden_prefix_frames + self.visible_frames

    @property
    def hidden_prefix_frames(self) -> int:
        """Frames hidden from the appended story clock.

        A zero-context window is an authored hard cut, so it does not consume
        prior-tail latents. H3 still needs its five-frame temporal origin; that
        independently generated preroll is hidden before the new story begins.
        """

        return self.context_frames or H3_FRAME_ORIGIN

    @property
    def audio_trim_ticks(self) -> int:
        if self.context_frames:
            return self.audio_bridge_ticks
        return round(self.hidden_prefix_frames / FPS * 40)

    @property
    def output_frames(self) -> int:
        return self.source_frames + self.visible_frames

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "InfiniteContinuationSpec":
        fields = cls.__dataclass_fields__
        restored = {}
        for key, definition in fields.items():
            if key in value:
                restored[key] = value[key]
            elif definition.default is not MISSING:
                restored[key] = definition.default
            else:
                raise KeyError(key)
        return cls(**restored)


@dataclass(slots=True)
class InfiniteProject:
    id: str
    title: str
    overview: str
    overall_soundscape: str
    non_diegetic_music: str
    overlap_seconds: float = 1.625
    memory: int = 60
    visual_memory_capacity: int = DEFAULT_VISUAL_MEMORY_CAPACITY
    audio_memory_capacity: int = DEFAULT_AUDIO_MEMORY_CAPACITY
    visual_memory_resolution: str = DEFAULT_VISUAL_MEMORY_RESOLUTION
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    service_family: str | None = None
    width: int | None = None
    height: int | None = None
    resolution: str | None = None
    aspect_ratio: str | None = None
    windows: list[dict[str, Any]] = field(default_factory=list)
    workflow_version: int = 1
    creation_mode: str = "online"
    model_variant: str = "base"
    sampling_steps: int = 20
    acceleration: float = 0.0
    window_duration_seconds: float = 8.0
    final_resolution: str = "1080p"
    final_sampling_steps: int = 4
    preview_branch_steps: int = 2
    preview_enabled: bool = True
    second_pass_acceleration: float = 75.0
    final_job_id: str | None = None
    final_sampling_settings: dict[str, Any] | None = None
    batch_plan: list[dict[str, Any]] = field(default_factory=list)
    batch_references: dict[str, str] = field(default_factory=dict)
    batch_cursor: int = 0
    batch_status: str = "idle"
    batch_error: str | None = None

    def public(self, jobs: dict[str, Any]) -> dict[str, Any]:
        windows = []
        total_frames = 0
        for item in self.windows:
            row = dict(item)
            job = jobs.get(str(item.get("job_id")))
            row["status"] = getattr(job, "status", "missing")
            row["video_url"] = (
                f"/api/v1/jobs/{job.id}/video"
                if job is not None and getattr(job, "output_path", None) is not None
                else None
            )
            row["error"] = getattr(job, "error", None) if job is not None else "job record missing"
            row["progress"] = (
                {"percent": job.progress_percent, "stage": job.progress_stage, "detail": job.progress_detail}
                if job is not None else None
            )
            conditioning_roles: list[dict[str, str]] = []
            if job is not None and getattr(job, "first_frame", None):
                conditioning_roles.append({"role": "first_frame", "label": "首帧"})
            if job is not None and getattr(job, "last_frame", None):
                conditioning_roles.append({"role": "last_frame", "label": "尾帧"})
            conditioning_roles.extend(
                {"role": f"reference_image_{index}", "label": f"参考图 {index}"}
                for index, _path in enumerate(
                    getattr(job, "reference_images", ()) if job is not None else (),
                    start=1,
                )
            )
            conditioning_roles.extend(
                {"role": f"reference_audio_{index}", "label": f"参考音频 {index}"}
                for index, _path in enumerate(
                    getattr(job, "reference_audios", ()) if job is not None else (),
                    start=1,
                )
            )
            row["conditioning"] = {
                "has_first_frame": bool(
                    job is not None and getattr(job, "first_frame", None)
                ),
                "has_last_frame": bool(
                    job is not None and getattr(job, "last_frame", None)
                ),
                "reference_image_count": len(
                    getattr(job, "reference_images", ()) if job is not None else ()
                ),
                "reference_audio_count": len(
                    getattr(job, "reference_audios", ()) if job is not None else ()
                ),
                "roles": conditioning_roles,
            }
            if job is not None and job.status == "succeeded":
                total_frames = int(item.get("total_frames", total_frames))
            windows.append(row)
        tail = windows[-1] if windows else None
        tail_status = None if tail is None else tail["status"]
        final_job = jobs.get(self.final_job_id) if self.final_job_id else None
        final_status = getattr(final_job, "status", None)
        final_active = final_status in {
            "queued", "starting_backend", "running", "awaiting_preview"
        }
        legacy_tail = self.windows[-1] if self.windows else {}
        public_model_variant = (
            self.model_variant
            if self.workflow_version >= 2
            else str(legacy_tail.get("model_variant", self.model_variant))
        )
        public_sampling_steps = (
            self.sampling_steps
            if self.workflow_version >= 2
            else int(legacy_tail.get("sampling_steps", self.sampling_steps))
        )
        public_acceleration = (
            self.acceleration
            if self.workflow_version >= 2
            else float(legacy_tail.get("acceleration", self.acceleration))
        )
        public_window_duration = (
            self.window_duration_seconds
            if self.workflow_version >= 2
            else float(legacy_tail.get(
                "requested_duration_seconds", self.window_duration_seconds
            ))
        )
        final_video_url = (
            f"/api/v1/jobs/{final_job.id}/video"
            if final_job is not None
            and final_status == "succeeded"
            and getattr(final_job, "output_path", None) is not None
            else None
        )
        selflift_final_ready = bool(
            self.workflow_version >= 3
            and self.creation_mode == "online"
            and windows
            and all(
                row["status"] == "succeeded"
                and jobs.get(row["job_id"]) is not None
                and getattr(jobs[row["job_id"]], "spec", None) is not None
                and getattr(jobs[row["job_id"]].spec, "selflift_enabled", False)
                and getattr(jobs[row["job_id"]], "checkpoint_path", None) is not None
                and jobs[row["job_id"]].checkpoint_path.is_file()
                for row in windows
            )
        )
        legacy_final_ready = bool(
            self.workflow_version < 3
            and tail is not None
            and tail["status"] == "succeeded"
            and jobs.get(tail["job_id"]) is not None
            and getattr(jobs[tail["job_id"]], "final_latents_path", None) is not None
        )
        public_batch_plan = []
        batch_reference_ids = set(self.batch_references)
        for item in self.batch_plan:
            row = {
                key: value for key, value in item.items()
                if not str(key).startswith("_")
            }
            private_references = item.get("_references")
            private_reference_ids = item.get("_reference_ids")
            if isinstance(private_reference_ids, list):
                reference_ids = list(private_reference_ids)
                row["reference_ids"] = reference_ids
                batch_reference_ids.update(reference_ids)
                row["references_inherited"] = private_references is None
            elif private_references is None and self.service_family == "reference":
                row["reference_ids"] = "inherit"
            public_batch_plan.append(row)
        return {
            "id": self.id,
            "title": self.title,
            "overview": self.overview,
            "overall_soundscape": self.overall_soundscape,
            "non_diegetic_music": self.non_diegetic_music,
            "overlap_seconds": self.overlap_seconds,
            "effective_overlap_seconds": context_frames_for_seconds(self.overlap_seconds) / FPS,
            "memory": self.memory,
            "memory_capacity": memory_capacity(
                self.memory,
                service_family=self.service_family or "first_last",
                visual_capacity=self.visual_memory_capacity,
                audio_capacity=self.audio_memory_capacity,
                visual_resolution=self.visual_memory_resolution,
            ),
            "visual_memory_capacity": self.visual_memory_capacity,
            "audio_memory_capacity": self.audio_memory_capacity,
            "visual_memory_resolution": self.visual_memory_resolution,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "service_family": self.service_family,
            "width": self.width,
            "height": self.height,
            "resolution": self.resolution,
            "aspect_ratio": self.aspect_ratio,
            "workflow_version": self.workflow_version,
            "trajectory_locked": self.workflow_version >= 2,
            "window_controls_editable": (
                self.workflow_version >= 3 and self.creation_mode == "online"
            ),
            "creation_mode": self.creation_mode,
            "preview_resolution": self.resolution,
            "model_variant": public_model_variant,
            "sampling_steps": public_sampling_steps,
            "acceleration": public_acceleration,
            "window_duration_seconds": public_window_duration,
            "final_resolution": self.final_resolution,
            "final_sampling_steps": self.final_sampling_steps,
            "preview_branch_steps": self.preview_branch_steps,
            "preview_enabled": self.preview_enabled,
            "second_pass_acceleration": self.second_pass_acceleration,
            "windows": windows,
            "window_count": len(windows),
            "total_frames": total_frames,
            "duration_seconds": total_frames / FPS,
            "tail_job_id": None if tail is None else tail["job_id"],
            "tail_status": tail_status,
            "can_retry_tail": tail_status in RETRYABLE_TAIL_STATUSES,
            "can_append": (
                self.creation_mode != "json"
                and self.batch_status != "running"
                and not final_active
                and (
                    tail is None
                    or tail_status == "succeeded"
                    or tail_status in RETRYABLE_TAIL_STATUSES
                )
            ),
            "second_sampling_available": bool(
                self.batch_status != "running"
                and not final_active
                and (selflift_final_ready or legacy_final_ready)
            ),
            "final_generation_method": (
                "global_sliding_selflift"
                if self.workflow_version >= 3
                else "legacy_detached_second_sampling"
            ),
            "batch": {
                "status": self.batch_status,
                "cursor": self.batch_cursor,
                "total": len(self.batch_plan),
                "remaining": max(0, len(self.batch_plan) - self.batch_cursor),
                "error": self.batch_error,
                "plan": public_batch_plan,
                "reference_ids": sorted(batch_reference_ids),
            },
            "final_sampling": {
                "job_id": self.final_job_id,
                "status": final_status,
                "active": final_active,
                "settings": self.final_sampling_settings,
                "video_url": final_video_url,
            },
        }


class InfiniteProjectStore:
    def __init__(self, data_dir: Path) -> None:
        self.projects: dict[str, InfiniteProject] = {}
        self.rebind(data_dir)

    def rebind(self, data_dir: Path) -> None:
        self.data_dir = Path(data_dir)
        self.root = self.data_dir / "infinite_projects"
        self.root.mkdir(parents=True, exist_ok=True)
        self.projects.clear()
        for path in sorted(self.root.glob("*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if "visual_memory_capacity" not in data:
                    legacy = memory_capacity(int(data.get("memory", 60)))
                    data["visual_memory_capacity"] = int(legacy["video_slots"])
                    data["audio_memory_capacity"] = int(legacy["audio_slots"])
                    data["visual_memory_resolution"] = "original"
                for row in data.get("windows", []):
                    if not isinstance(row, dict) or "visual_memory_capacity" in row:
                        continue
                    legacy = memory_capacity(int(row.get("memory", data.get("memory", 60))))
                    row["visual_memory_capacity"] = int(legacy["video_slots"])
                    row["audio_memory_capacity"] = int(legacy["audio_slots"])
                    row["visual_memory_resolution"] = "original"
                project = InfiniteProject(**data)
                memory_capacity(
                    visual_capacity=project.visual_memory_capacity,
                    audio_capacity=project.audio_memory_capacity,
                    visual_resolution=project.visual_memory_resolution,
                )
                self.projects[project.id] = project
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                continue

    def create(self, payload: dict[str, Any]) -> InfiniteProject:
        title = str(payload.get("title", "未命名长视频")).strip() or "未命名长视频"
        overview = str(payload.get("overview", "")).strip()
        overlap = float(payload.get("overlap_seconds", 1.625))
        context_frames_for_seconds(overlap)
        memory = int(payload.get("memory", 60))
        if not 0 <= memory <= 100:
            raise ContractError("memory must be an integer from 0 to 100")
        if any(
            key in payload for key in (
                "visual_memory_capacity",
                "audio_memory_capacity",
                "visual_memory_resolution",
            )
        ):
            split = memory_capacity(
                memory,
                visual_capacity=int(payload.get(
                    "visual_memory_capacity", DEFAULT_VISUAL_MEMORY_CAPACITY
                )),
                audio_capacity=int(payload.get(
                    "audio_memory_capacity", DEFAULT_AUDIO_MEMORY_CAPACITY
                )),
                visual_resolution=str(payload.get(
                    "visual_memory_resolution", DEFAULT_VISUAL_MEMORY_RESOLUTION
                )),
            )
        elif "memory" in payload:
            split = memory_capacity(memory)
        else:
            split = memory_capacity(
                visual_capacity=DEFAULT_VISUAL_MEMORY_CAPACITY,
                audio_capacity=DEFAULT_AUDIO_MEMORY_CAPACITY,
                visual_resolution=DEFAULT_VISUAL_MEMORY_RESOLUTION,
            )
        workflow_version = int(payload.get("workflow_version", 1))
        if workflow_version not in {1, 2, 3}:
            raise ContractError("workflow_version must be 1, 2 or 3")
        resolution = None
        aspect_ratio = None
        model_variant = "base"
        sampling_steps = 20
        acceleration = 0.0
        window_duration_seconds = 8.0
        creation_mode = "online"
        final_resolution = "1080p"
        final_sampling_steps = 4
        preview_branch_steps = 2
        preview_enabled = True
        second_pass_acceleration = 75.0
        service_family = None
        if workflow_version >= 2:
            resolution, _ = generation_short_edge(
                payload.get("preview_resolution", payload.get("resolution", "540p"))
            )
            aspect_ratio = str(payload.get("aspect_ratio", "16:9")).strip()
            if aspect_ratio not in ASPECT_RATIOS:
                raise ContractError("aspect_ratio must be 1:1, 4:3, 3:4, 16:9 or 9:16")
            model_variant = str(payload.get("model_variant", "lora")).strip().lower()
            if model_variant not in {"base", "lora"}:
                raise ContractError("model_variant must be base or lora")
            sampling_steps = int(payload.get("sampling_steps", 8))
            maximum_steps = 10 if model_variant == "lora" else 30
            if not 4 <= sampling_steps <= maximum_steps:
                raise ContractError(
                    f"sampling_steps must be between 4 and {maximum_steps} for {model_variant}"
                )
            acceleration = float(payload.get("acceleration", 50))
            if not math.isfinite(acceleration) or not 0 <= acceleration <= 100:
                raise ContractError("acceleration must be between 0 and 100")
            window_duration_seconds = float(payload.get("window_duration_seconds", 5))
            if not math.isfinite(window_duration_seconds) or not 1 <= window_duration_seconds <= 15:
                raise ContractError("window_duration_seconds must be between 1 and 15")
            normalized_final_resolution, final_short_edge = progressive_short_edge(
                payload.get("final_resolution", "1080p")
            )
            final_resolution = public_resolution_name(normalized_final_resolution)
            if workflow_version == 2 and final_short_edge < 720:
                raise ContractError(
                    "final_resolution must be between 720p and 1440p for workflow_version 2"
                )
            if (
                workflow_version >= 3
                and final_short_edge < int(resolution[:-1])
            ):
                raise ContractError(
                    "final_resolution must be greater than or equal to preview_resolution"
                )
            final_sampling_steps = int(payload.get("final_sampling_steps", 4))
            if not 1 <= final_sampling_steps <= 8:
                raise ContractError("final_sampling_steps must be between 1 and 8")
            if workflow_version >= 3 and final_sampling_steps >= sampling_steps:
                raise ContractError(
                    "final_sampling_steps must leave at least one low-resolution step"
                )
            preview_branch_steps = int(payload.get("preview_branch_steps", 2))
            if not 1 <= preview_branch_steps <= 4:
                raise ContractError("preview_branch_steps must be between 1 and 4")
            raw_preview_enabled = payload.get("preview_enabled", True)
            preview_enabled = (
                raw_preview_enabled
                if isinstance(raw_preview_enabled, bool)
                else str(raw_preview_enabled).strip().lower() in {"1", "true", "yes", "on"}
            )
            second_pass_acceleration = float(
                payload.get("second_pass_acceleration", 75)
            )
            if (
                not math.isfinite(second_pass_acceleration)
                or not 0 <= second_pass_acceleration <= 100
            ):
                raise ContractError(
                    "second_pass_acceleration must be between 0 and 100"
                )
            if workflow_version >= 3:
                creation_mode = str(
                    payload.get("creation_mode", "online")
                ).strip().lower()
                if creation_mode not in {"online", "json"}:
                    raise ContractError("creation_mode must be online or json")
                preview_enabled = creation_mode == "online"
            raw_family = payload.get("service_family")
            service_family = None if raw_family in (None, "") else str(raw_family)
        project = InfiniteProject(
            id=str(uuid.uuid4()),
            title=title,
            overview=overview,
            overall_soundscape=str(payload.get("overall_soundscape", "N/A")).strip() or "N/A",
            non_diegetic_music=str(payload.get("non_diegetic_music", "N/A")).strip() or "N/A",
            overlap_seconds=overlap,
            memory=memory,
            visual_memory_capacity=int(split["video_slots"]),
            audio_memory_capacity=int(split["audio_slots"]),
            visual_memory_resolution=str(split["visual_resolution"]),
            workflow_version=workflow_version,
            creation_mode=creation_mode,
            service_family=service_family,
            resolution=resolution,
            aspect_ratio=aspect_ratio,
            model_variant=model_variant,
            sampling_steps=sampling_steps,
            acceleration=acceleration,
            window_duration_seconds=window_duration_seconds,
            final_resolution=final_resolution,
            final_sampling_steps=final_sampling_steps,
            preview_branch_steps=preview_branch_steps,
            preview_enabled=preview_enabled,
            second_pass_acceleration=second_pass_acceleration,
        )
        self.projects[project.id] = project
        self.persist(project)
        return project

    def persist(self, project: InfiniteProject) -> None:
        project.updated_at = time.time()
        target = self.root / f"{project.id}.json"
        temporary = target.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(asdict(project), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(target)

    def require(self, project_id: str) -> InfiniteProject:
        try:
            return self.projects[project_id]
        except KeyError as error:
            raise ContractError("infinite project not found") from error

    def delete(self, project_id: str) -> InfiniteProject:
        """Delete only the project container while retaining referenced jobs."""

        project = self.require(project_id)
        (self.root / f"{project.id}.json.tmp").unlink(missing_ok=True)
        (self.root / f"{project.id}.json").unlink(missing_ok=True)
        del self.projects[project.id]
        return project
