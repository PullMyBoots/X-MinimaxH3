"""Service-facing adapter for the in-process native H3 pipeline."""

from __future__ import annotations

import asyncio
import ctypes
import gc
import json
import math
import os
import tempfile
import time
import traceback
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from ..contract import (
    GenerationSpec,
    SecondSamplingSpec,
    actual_step_schedule,
    engine_variant,
    launcher_family,
    launcher_vram_profile,
    launcher_weight_tier,
    normalize_launcher,
    resolve_geometry,
)
from .pipeline import GenerationInput, NativeH3Pipeline, PipelineCancelled, SamplingConfig
from .adapters.sampling_mux import refinement_sigma_schedule


class NativeGenerationCancelled(RuntimeError):
    """A request was cancelled at a safe native-pipeline boundary."""


@dataclass(frozen=True, slots=True)
class NativeGenerationResult:
    runtime_key: str
    elapsed_seconds: float
    output_path: Path
    stage_seconds: dict[str, float]
    inference_plan: dict[str, Any] | None = None
    final_latents_path: Path | None = None


@dataclass(frozen=True, slots=True)
class NativeCheckpointResult:
    runtime_key: str
    elapsed_seconds: float
    checkpoint_path: Path | None
    preview_path: Path | None
    completed_steps: int
    total_steps: int
    stage_seconds: dict[str, float]
    inference_plan: dict[str, Any] | None = None
    preview_latents_path: Path | None = None
    token_memory_path: Path | None = None


def _public_inference_plan(execution_profile: object) -> dict[str, Any] | None:
    """Keep quality scheduling and physical memory routing auditable.

    Historically the service returned only ``joint_acceleration`` from the
    much larger native execution profile.  That made an ``auto`` memory-mode
    request impossible to inspect after completion.  Preserve the existing
    flat scheduler fields for client compatibility and attach only the small,
    stable physical-route receipt needed by the public product contract.
    """

    if not isinstance(execution_profile, dict):
        return None
    joint = execution_profile.get("joint_acceleration")
    result = dict(joint) if isinstance(joint, dict) else {}
    memory_execution = execution_profile.get("memory_execution")
    if isinstance(memory_execution, dict):
        result["memory_execution"] = dict(memory_execution)
    for key in (
        "roi_difficulty_selection",
        "roi_atlas_refinement",
        "selflift",
        "global_selflift",
        "global_co_denoise",
        "infinite_selflift_preview",
        "infinite_selflift_source",
    ):
        receipt = execution_profile.get(key)
        if isinstance(receipt, dict):
            # Automatic difficult-region refinement is meaningful only when
            # callers can audit which latent regions were selected and how
            # they shared the atlas pass.  These receipts contain scalar
            # diagnostics and normalized boxes, never latent tensors.
            result[key] = dict(receipt)
    conditioning_cache = execution_profile.get("qwen_conditioning_cache")
    if isinstance(conditioning_cache, dict):
        # This is a small receipt only; the cached tensors remain private to
        # the latent checkpoint.  Exposing the receipt makes it possible to
        # verify that second sampling did not silently run Qwen again.
        result["qwen_conditioning_cache"] = dict(conditioning_cache)
    output_mux = execution_profile.get("output_mux")
    if isinstance(output_mux, dict):
        # The encoder receipt contains the publication-time audio gain and
        # clipping counters.  It is small, contains no tensors, and is needed
        # to prove that a delivered long video used shape-preserving peak
        # normalization instead of the legacy hard clamp.
        result["output_mux"] = dict(output_mux)
    audio_manifold_guard = execution_profile.get("audio_manifold_guard")
    if isinstance(audio_manifold_guard, dict):
        # Complete compact decision receipt for the post-trajectory guard.
        # It contains acoustic scores and affected intervals only, proving
        # whether a clip was left untouched or locally projected without
        # exposing prompts, tensors, or private execution state.
        result["audio_manifold_guard"] = dict(audio_manifold_guard)
    audio_window_decode = execution_profile.get("audio_window_decode")
    if isinstance(audio_window_decode, dict):
        # This compact receipt proves whether delivery audio was decoded in
        # independent causal domains and whether any guarded PCM seam patch
        # was applied. It contains clocks and scalar diagnostics only.
        result["audio_window_decode"] = dict(audio_window_decode)
    infinite_continuation = execution_profile.get("infinite_continuation")
    if isinstance(infinite_continuation, dict):
        result["infinite_continuation"] = dict(infinite_continuation)
    continuation_text_bridge = execution_profile.get("continuation_text_bridge")
    if isinstance(continuation_text_bridge, dict):
        # Expose the compact schedule receipt so an infinite-window request
        # cannot claim V24 acceleration while its boundary auxiliary silently
        # executes a second exact trajectory.
        result["continuation_text_bridge"] = dict(continuation_text_bridge)
    infinite_window_forecast = execution_profile.get("infinite_window_forecast")
    if isinstance(infinite_window_forecast, dict):
        result["infinite_window_forecast"] = dict(infinite_window_forecast)
    long_horizon = execution_profile.get("long_horizon")
    if isinstance(long_horizon, dict):
        # Keep the research route auditable without exposing the much larger
        # per-window execution profiles.  In particular, token-memory records
        # prove that both modalities stayed within their fixed active budget.
        public_keys = (
            "single_model_session",
            "single_global_solver_state",
            "scheduler_updates_per_step",
            "intermediate_decode",
            "single_final_decode",
            "joint_av_prefix",
            "audio_seam_method",
            "audio_context_frames",
            "audio_bridge_ticks",
            "audio_repaint_overlap_discarded",
            "audio_latent_overlap_add",
            "bounded_av_token_memory",
            "inferred_audio_token_memory",
            "long_visual_memory",
            "latest_visual_only",
            "generated_voice_bank",
            "authored_voice_anchor",
            "global_audio_spine",
            "window_audio_discarded",
            "structured_speech_authority",
            "token_memory_records",
            "token_memory_route_records",
            "rotary_time",
            "unique_conditioning_encodes",
            "conditioning_preencoded_before_dit",
            "continuation_semantic_handoff",
            "continuation_video_handoff",
            "window_interface",
        )
        receipt = {
            key: long_horizon[key]
            for key in public_keys
            if key in long_horizon
        }
        if receipt:
            result["long_horizon"] = receipt
    return result or None


def _bind_generated_voice_reference(prompt: str) -> str:
    """Compile one internal Ref2VA voice-bank relation around a local prompt.

    The generated audio is presented as ``<Audio 1>`` by the packed DiT path.
    This deterministic wrapper supplies the matching Qwen text label and the
    standard Ref2VA relationship without reading or rewriting story content.
    It is used only after an earlier window has established a request-local
    S1 voice anchor.
    """

    value = str(prompt)
    description = value
    for marker in (
        "integrated_multimodal_description:",
        "detailed_description:",
    ):
        head, separator, tail = value.partition(marker)
        if separator:
            description = "\n".join(
                part for part in (head.strip(), tail.lstrip()) if part
            )
            break
    return "\n\n".join((
        "subject_definitions:\n"
        "<Audio 1> is the voice-timbre reference for the continuing generated "
        "speaker (S1).",
        "summary:\n"
        "[audio reference] The continuing target video uses <Audio 1> only "
        "to preserve the established voice identity of (S1).",
        "retention_analysis:\n"
        "<Audio 1>: reference - preserve its vocal timbre without copying "
        "its words, timing, ambience, or sound events.",
        f"detailed_description:\n{description.strip()}",
    ))


_ORIGINAL_ACTUAL_INDICES = {
    "fast": (0, 1, 2, 3, 4, 8, 13, 19),
    "balanced": (0, 1, 2, 3, 4, 8, 12, 16, 19),
    "quality": (0, 1, 2, 3, 4, 6, 8, 11, 14, 17, 18, 19),
    "ultra": tuple(range(20)),
}


def _sampling(spec: GenerationSpec) -> SamplingConfig:
    if spec.engine in ("original", "reference"):
        return SamplingConfig(
            engine="original",
            num_steps=20,
            actual_step_indices=_ORIGINAL_ACTUAL_INDICES[spec.quality],
            sampler="res_multistep",
            scheduler="simple",
        )
    return SamplingConfig(
        engine="lora",
        num_steps=int(spec.preset["steps"]),
        actual_step_indices=None,
        sampler="turbo",
        scheduler="simple",
        lora_strength=float(spec.preset["strength"]),
    )


class NativeH3Engine:
    """Map the stable service contract onto one hot in-process H3 pipeline.

    Model construction is intentionally injected.  It keeps the API/queue
    importable on CPU and lets checkpoint loading fail during readiness rather
    than while importing the web application.
    """

    def __init__(
        self,
        pipeline: NativeH3Pipeline,
        output_root: Path,
        *,
        runtime_revision: str = "native-h3-sm89-v1",
    ) -> None:
        self._pipeline = pipeline
        self._output_root = output_root.resolve()
        self._output_root.mkdir(parents=True, exist_ok=True)
        self.runtime_revision = runtime_revision

    async def generate(
        self,
        spec: GenerationSpec,
        first_frame: Path | None,
        last_frame: Path | None,
        reference_images: tuple[Path, ...],
        reference_videos: tuple[Path, ...],
        reference_audios: tuple[Path, ...],
        cancel_event: asyncio.Event,
        output_path: Path,
        progress_callback: Any | None = None,
        preview_ready_callback: Any | None = None,
        preview_decision_wait: Any | None = None,
        checkpoint_path: Path | None = None,
        resume_checkpoint_path: Path | None = None,
        final_latents_path: Path | None = None,
        second_sampling: SecondSamplingSpec | None = None,
        refinement_latents_path: Path | None = None,
        external_refinement_video_path: Path | None = None,
        refinement_schedule_mode: str = "continuous_tail",
        refinement_atlas_denoise_regions: tuple[
            tuple[float, float, float, float, tuple[float, ...]], ...
        ] = (),
        refinement_handoff_latents_path: Path | None = None,
        refinement_handoff_context_frames: int = 0,
        conditioning_cache_source_path: Path | None = None,
        continuation: Any | None = None,
        continuation_source_latents_path: Path | None = None,
        continuation_source_video_path: Path | None = None,
        continuation_source_memory_path: Path | None = None,
        continuation_output_memory_path: Path | None = None,
    ) -> NativeGenerationResult:
        output_path = output_path.resolve()
        if not output_path.is_relative_to(self._output_root):
            raise ValueError("output_path must stay inside the configured output root")
        if reference_images or reference_videos or reference_audios:
            raise RuntimeError("the compatibility pipeline does not implement Ref2VA")
        if (
            second_sampling is not None
            or refinement_latents_path is not None
            or external_refinement_video_path is not None
        ):
            raise RuntimeError("the compatibility pipeline does not implement H3 second sampling")
        if continuation is not None or continuation_source_latents_path is not None:
            raise RuntimeError("the compatibility pipeline does not implement infinite continuation")
        request = GenerationInput(
            prompt=spec.prompt,
            width=spec.width,
            height=spec.height,
            num_frames=spec.frames,
            seed=spec.seed,
            sampling=_sampling(spec),
            fps=24,
            first_frame=first_frame,
            last_frame=last_frame,
            output_path=output_path,
        )
        started = time.monotonic()
        if progress_callback is not None:
            progress_callback({"percent": 5, "stage": "generating", "detail": "开始生成"})
        try:
            state = await asyncio.to_thread(
                self._pipeline.generate,
                request,
                cancel_check=cancel_event.is_set,
            )
        except PipelineCancelled as error:
            raise NativeGenerationCancelled(str(error)) from error

        result_path = self._result_path(state.result, output_path)
        return NativeGenerationResult(
            runtime_key=f"{spec.engine}:{self.runtime_revision}",
            elapsed_seconds=round(time.monotonic() - started, 3),
            output_path=result_path,
            stage_seconds=dict(state.metrics.elapsed_seconds),
            final_latents_path=None,
        )

    def _result_path(self, result: Any, expected: Path) -> Path:
        if isinstance(result, (str, Path)):
            candidate = Path(result).resolve()
        elif isinstance(result, dict) and result.get("output_path"):
            candidate = Path(result["output_path"]).resolve()
        else:
            candidate = expected.resolve()
        if not candidate.is_relative_to(self._output_root):
            raise RuntimeError("native engine returned a path outside output_root")
        if not candidate.is_file():
            raise RuntimeError(f"native engine did not create the expected video: {candidate.name}")
        return candidate

    async def close(self) -> None:
        await asyncio.to_thread(self._pipeline.close)

    @property
    def output_root(self) -> Path:
        return self._output_root

class NativeHotH3Engine:
    """Own one hot family session with a request-level base/LoRA switch."""

    def __init__(self, factory: Any, output_root: Path) -> None:
        self._factory = factory
        self._output_root = output_root.resolve()
        self._output_root.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()
        self._built = None
        self._engine_name: str | None = None
        self._device_fatal_error: str | None = None
        self._warm_state: dict[str, Any] = {
            "status": "cold", "engine": None, "startup_seconds": None, "error": None,
            "progress_percent": 0.0,
            "progress_stage": "cold",
            "progress_detail": "尚未加载模型",
        }

    def _loading_progress(self, percent: float, stage: str, detail: str) -> None:
        if self._warm_state.get("status") != "loading":
            return
        self._warm_state.update({
            "progress_percent": round(max(0.0, min(100.0, percent)), 1),
            "progress_stage": stage,
            "progress_detail": detail,
        })

    def preflight(self, engine: str) -> dict[str, Any]:
        return self._factory.preflight(engine)

    def _ensure_session(self, engine: str):
        if self._device_fatal_error is not None:
            raise RuntimeError(
                "CUDA engine requires a service restart after a fatal device error"
            )
        launcher = normalize_launcher(engine)
        family = launcher_family(launcher)
        weight_tier = launcher_weight_tier(launcher)
        vram_profile = launcher_vram_profile(launcher)
        if self._built is not None and self._engine_name == launcher:
            return self._built
        if self._built is not None:
            self._built.session.close()
            self._built = None
            self._engine_name = None
        try:
            configure_progress = getattr(self._factory, "set_progress_callback", None)
            if callable(configure_progress):
                configure_progress(self._loading_progress)
            self._built = self._factory.build(launcher)
            self._engine_name = launcher
            self._warm_state = {
                "status": "ready",
                "engine": family,
                "launcher": launcher,
                "weight_tier": weight_tier,
                "vram_profile": vram_profile,
                "allocator_ceiling_gib": round(
                    float(getattr(self._built, "allocator_ceiling_gib", 0.0)), 3
                ),
                "startup_seconds": round(self._built.startup_seconds, 3),
                "qwen_storage": getattr(self._built, "qwen_storage", "source"),
                "qwen_layer_cache": bool(
                    getattr(self._built, "qwen_layer_cache", False)
                ),
                "host_memory_profile": getattr(
                    self._built, "host_memory_profile", None
                ),
                "dit_host_pinned": bool(
                    getattr(self._built, "dit_host_pinned", False)
                ),
                "dit_host_cache_gib": round(
                    float(getattr(self._built, "dit_host_cache_gib", 0.0)), 3
                ),
                "dit_host_pinned_gib": round(
                    float(getattr(self._built, "dit_host_pinned_gib", 0.0)), 3
                ),
                "dit_host_pinned_fraction": round(
                    float(getattr(self._built, "dit_host_pinned_fraction", 0.0)), 4
                ),
                "v19_release_bundle": getattr(
                    self._built, "v19_release_bundle", None
                ),
                "v19_release_digest": getattr(
                    self._built, "v19_release_digest", None
                ),
                "pareto_policy_id": getattr(
                    self._built, "pareto_policy_id", None
                ),
                "pareto_candidate_id": getattr(
                    self._built, "pareto_candidate_id", None
                ),
                "lora_checkpoint": Path(
                    getattr(self._built, "lora_checkpoint", "")
                ).name,
                "lora_profile_id": getattr(
                    self._built, "lora_profile_id", None
                ),
                "lora_display_name": getattr(
                    self._built, "lora_display_name", None
                ),
                "lora_recommended_steps": list(getattr(
                    self._built, "lora_recommended_steps", ()
                )),
                "lora_default_steps": getattr(
                    self._built, "lora_default_steps", None
                ),
                "error": None,
                "progress_percent": 100.0,
                "progress_stage": "ready",
                "progress_detail": "模型引擎已就绪",
            }
        except Exception as error:
            if os.environ.get("H3_NATIVE_RESEARCH_TRACE_PRELOAD_ERROR", "0") == "1":
                traceback.print_exc()
            self._warm_state = {
                "status": "failed", "engine": family,
                "launcher": launcher, "weight_tier": weight_tier,
                "vram_profile": vram_profile,
                "startup_seconds": None, "error": str(error),
                "progress_percent": 100.0,
                "progress_stage": "failed",
                "progress_detail": "模型引擎加载失败",
            }
            raise
        finally:
            configure_progress = getattr(self._factory, "set_progress_callback", None)
            if callable(configure_progress):
                configure_progress(None)
        return self._built

    async def preload(self, engine: str) -> None:
        engine = normalize_launcher(engine)
        family = launcher_family(engine)
        weight_tier = launcher_weight_tier(engine)
        vram_profile = launcher_vram_profile(engine)
        async with self._lock:
            if self._built is not None and self._engine_name == engine:
                return
            self._warm_state = {
                "status": "loading", "engine": family,
                "launcher": engine, "weight_tier": weight_tier,
                "vram_profile": vram_profile,
                "startup_seconds": None, "error": None,
                "progress_percent": 1.0,
                "progress_stage": "starting",
                "progress_detail": "开始加载模型引擎",
            }
            try:
                await asyncio.to_thread(self._ensure_session, engine)
            except Exception:
                # Readiness exposes the failure. Keep the Web/API process alive
                # so an operator can inspect it or repair files in place.
                return

    @property
    def warm_state(self) -> dict[str, Any]:
        # Health is intentionally public. Do not leak checkpoint paths or
        # loader exception details through it.
        return {
            key: self._warm_state.get(key)
            for key in (
                "status", "engine", "startup_seconds", "qwen_storage",
                "qwen_layer_cache", "host_memory_profile", "dit_host_pinned",
                "dit_host_cache_gib", "dit_host_pinned_gib",
                "dit_host_pinned_fraction", "pareto_policy_id", "pareto_candidate_id",
                "launcher", "weight_tier", "vram_profile",
                "allocator_ceiling_gib",
                "lora_checkpoint",
                "lora_profile_id", "lora_display_name",
                "lora_recommended_steps", "lora_default_steps",
                "progress_percent", "progress_stage", "progress_detail",
            )
        }

    @staticmethod
    def _request_plan(spec: GenerationSpec):
        # Six-step Larry and original 9/11 are the fully calibrated routes.
        # Other exposed presets use the same lossless mechanical plan while
        # retaining their user-selected model behavior; they are deliberately
        # excluded from latency-based routing instead of failing the request.
        approximate_attention = (
            spec.joint_acceleration_enabled and float(spec.acceleration or 0.0) > 0.0
        ) or (spec.advanced and spec.attention_keep_ratio < 1.0)
        calibrated = not spec.joint_acceleration_enabled and not approximate_attention and ((
            spec.engine in ("lora", "reference_lora") and int(spec.preset["steps"]) == 6
        ) or (
            spec.engine in ("original", "reference")
            and int(spec.preset["actual_steps"]) == 9
        ))
        if calibrated:
            return None
        from .planner import ExecutionPlan
        from .runtime import OffloadMode

        plan = ExecutionPlan(
            offload_mode=OffloadMode.BLOCK,
            mlp_chunk_tokens=8192,
            block_buffer_count=2,
            prefetch_depth=1,
            vae_spatial_tile=(288, 288),
        )
        if approximate_attention and not spec.joint_acceleration_enabled:
            plan = replace(
                plan,
                attention_topk=spec.attention_keep_ratio,
                sparse_scope=spec.sparse_scope,
            )
        return plan

    async def generate(
        self,
        spec: GenerationSpec,
        first_frame: Path | None,
        last_frame: Path | None,
        reference_images: tuple[Path, ...],
        reference_videos: tuple[Path, ...],
        reference_audios: tuple[Path, ...],
        cancel_event: asyncio.Event,
        output_path: Path,
        progress_callback: Any | None = None,
        preview_ready_callback: Any | None = None,
        preview_decision_wait: Any | None = None,
        checkpoint_path: Path | None = None,
        resume_checkpoint_path: Path | None = None,
        final_latents_path: Path | None = None,
        second_sampling: SecondSamplingSpec | None = None,
        refinement_latents_path: Path | None = None,
        external_refinement_video_path: Path | None = None,
        refinement_schedule_mode: str = "continuous_tail",
        refinement_atlas_denoise_regions: tuple[
            tuple[float, float, float, float, tuple[float, ...]], ...
        ] = (),
        refinement_handoff_latents_path: Path | None = None,
        refinement_handoff_context_frames: int = 0,
        conditioning_cache_source_path: Path | None = None,
        continuation: Any | None = None,
        continuation_source_latents_path: Path | None = None,
        continuation_source_video_path: Path | None = None,
        continuation_source_memory_path: Path | None = None,
        continuation_output_memory_path: Path | None = None,
        global_selflift_source_path: Path | None = None,
        global_selflift_prompts: tuple[str, ...] = (),
        global_selflift_prompt_ranges: tuple[tuple[int, int], ...] = (),
    ) -> NativeGenerationResult | NativeCheckpointResult:
        from .hot_session import (
            HotSessionCancelled,
            HotSessionCheckpointResult,
            HotSessionDeviceFatal,
            HotSessionRequest,
        )
        from .long_video_motion_detail import select_candidate

        is_second_sampling = second_sampling is not None
        is_incremental_continuation = continuation is not None
        is_global_selflift = global_selflift_source_path is not None
        global_selflift_plan = None
        if is_global_selflift:
            from .global_co_denoise import (
                plan_balanced_global_av_windows,
                plan_global_av_windows,
                plan_prompt_owned_global_av_windows,
                window_geometry_for_seconds,
            )

            if spec.output_frames is None:
                raise ValueError("global SelfLift requires the complete output frame clock")
            if spec.selflift_temporal_window_enabled:
                window_frames, stride_frames = window_geometry_for_seconds(
                    spec.selflift_temporal_window_seconds,
                    spec.selflift_temporal_overlap_seconds,
                )
                global_selflift_plan = (
                    plan_prompt_owned_global_av_windows(
                        int(spec.output_frames),
                        global_selflift_prompt_ranges,
                        window_frames=window_frames,
                        stride_frames=stride_frames,
                        balanced=True,
                    )
                    if global_selflift_prompt_ranges
                    else plan_balanced_global_av_windows(
                        int(spec.output_frames),
                        window_frames=window_frames,
                        stride_frames=stride_frames,
                    )
                )
            else:
                global_selflift_plan = (
                    plan_prompt_owned_global_av_windows(
                        int(spec.output_frames),
                        global_selflift_prompt_ranges,
                    )
                    if global_selflift_prompt_ranges
                    else plan_global_av_windows(int(spec.output_frames))
                )
            if len(global_selflift_prompts) != len(global_selflift_plan.windows):
                raise ValueError(
                    "global SelfLift requires one prompt for every temporal view"
                )
            if any(not prompt.strip() for prompt in global_selflift_prompts):
                raise ValueError("global SelfLift prompts cannot be empty")
            if is_second_sampling or is_incremental_continuation:
                raise ValueError(
                    "global SelfLift is a final project branch, not continuation or second sampling"
                )
        if is_second_sampling and first_frame is None:
            raw_anchor_path = os.environ.get(
                "H3_SECOND_SAMPLING_FIRST_FRAME_ANCHOR", ""
            ).strip()
            if raw_anchor_path:
                anchor_path = Path(raw_anchor_path).expanduser().resolve()
                if not anchor_path.is_file():
                    raise ValueError(
                        "H3 second-sampling first-frame anchor does not exist: "
                        f"{anchor_path}"
                    )
                first_frame = anchor_path
        if is_second_sampling and last_frame is None:
            raw_anchor_path = os.environ.get(
                "H3_SECOND_SAMPLING_LAST_FRAME_ANCHOR", ""
            ).strip()
            if raw_anchor_path:
                anchor_path = Path(raw_anchor_path).expanduser().resolve()
                if not anchor_path.is_file():
                    raise ValueError(
                        "H3 second-sampling last-frame anchor does not exist: "
                        f"{anchor_path}"
                    )
                last_frame = anchor_path
        if is_incremental_continuation != bool(
            continuation_source_latents_path is not None
            and continuation_output_memory_path is not None
        ):
            raise ValueError(
                "infinite continuation requires its contract, source latent and output memory path"
            )
        if not is_incremental_continuation and continuation_source_memory_path is not None:
            raise ValueError("infinite continuation memory requires a continuation contract")
        if not is_incremental_continuation and continuation_source_video_path is not None:
            raise ValueError("infinite continuation preview requires a continuation contract")
        if is_incremental_continuation and is_second_sampling:
            raise ValueError("infinite continuation and second sampling are separate tasks")
        if is_incremental_continuation:
            from ..infinite_video import InfiniteContinuationSpec

            if not isinstance(continuation, InfiniteContinuationSpec):
                raise TypeError("invalid infinite continuation contract")
            if spec.frames != continuation.physical_frames:
                raise ValueError("infinite continuation frame clocks disagree")
            if reference_videos:
                raise ValueError("infinite continuation does not accept reference videos")
        is_long_horizon = bool(
            not is_second_sampling
            and not is_incremental_continuation
            and not is_global_selflift
            and
            (spec.long_video is not None or (spec.output_frames is not None and spec.output_frames > spec.frames))
        )
        runtime_frames = int(
            max(window.frames for window in global_selflift_plan.windows)
            if global_selflift_plan is not None
            else spec.output_frames
            if is_second_sampling and spec.output_frames is not None
            else spec.frames
        )
        has_refinement_source = bool(
            refinement_latents_path is not None
            or external_refinement_video_path is not None
        )
        if is_second_sampling != has_refinement_source:
            raise ValueError(
                "second_sampling requires one refinement latent or external video source"
            )
        if (
            refinement_latents_path is not None
            and external_refinement_video_path is not None
        ):
            raise ValueError(
                "refinement latent and external video sources are mutually exclusive"
            )
        if (
            second_sampling is not None
            and second_sampling.model_variant == "lora"
            and second_sampling.steps < 4
        ):
            raise ValueError("H3 LoRA second sampling requires at least four steps")
        long_plan = None  # Legacy sequential continuation; retained as fallback only.
        global_co_plan = None
        global_co_prompts: tuple[str, ...] = ()
        if global_selflift_plan is not None:
            global_co_plan = global_selflift_plan
            global_co_prompts = tuple(global_selflift_prompts)
        bounded_token_memory = bool(
            is_long_horizon
            and os.environ.get("H3_LONG_AV_TOKEN_MEMORY", "0").strip() == "1"
        )
        inferred_audio_token_memory = bool(
            bounded_token_memory
            and os.environ.get("H3_LONG_AUDIO_TOKEN_MEMORY", "0").strip()
            != "0"
        )
        long_visual_memory = bool(
            bounded_token_memory
            and os.environ.get("H3_LONG_VISUAL_TOKEN_MEMORY", "1").strip() != "0"
        )
        latest_visual_only = bool(
            long_visual_memory
            and os.environ.get("H3_LONG_LATEST_VISUAL_ONLY", "0").strip() == "1"
        )
        generated_voice_bank = bool(
            inferred_audio_token_memory
            and os.environ.get(
                "H3_LONG_GENERATED_VOICE_BANK", ""
            ).strip() == "ref2va_self_anchor_v1"
        )
        global_audio_spine = bool(
            bounded_token_memory
            and os.environ.get("H3_LONG_GLOBAL_AUDIO_SPINE", "0").strip()
            == "1"
        )
        structured_director = bool(
            is_long_horizon
            and os.environ.get("H3_LONG_STRUCTURED_DIRECTOR", "0").strip()
            == "1"
        )
        structured_speech_gate = bool(
            structured_director
            and os.environ.get(
                "H3_LONG_STRUCTURED_SPEECH_GATE", "0"
            ).strip()
            == "1"
        )
        authored_preview = None
        authored_memory = None
        if spec.long_video is not None and not is_second_sampling:
            from ..long_video import memory_budget, validate_reference_inputs
            validate_reference_inputs(
                spec.long_video, first_frame=first_frame is not None,
                last_frame=last_frame is not None, reference_images=len(reference_images),
                reference_audios=len(reference_audios), reference_videos=len(reference_videos),
                service_family=spec.service_family,
            )
            authored_memory = memory_budget(
                spec.long_video, service_family=spec.service_family,
                width=spec.width, height=spec.height,
                user_images=len(reference_images), user_audios=len(reference_audios),
            )
            # New requests have immutable per-task policy. Legacy env flags
            # cannot change window prompts, routing or audio postprocessing.
            bounded_token_memory = True  # retain the causal AV solver at memory=0
            long_visual_memory = authored_memory["video_frames"] > 0
            inferred_audio_token_memory = authored_memory["audio_clips"] > 0
            structured_director = True
            latest_visual_only = generated_voice_bank = global_audio_spine = structured_speech_gate = False
        memory_collection_enabled = bool(
            bounded_token_memory and (authored_memory is None or
                authored_memory["video_frames"] > 0 or authored_memory["audio_clips"] > 0)
        )
        if generated_voice_bank:
            if not structured_director:
                raise ValueError(
                    "generated voice bank requires structured-director dialogue clocks"
                )
            if spec.service_family != "reference":
                raise ValueError(
                    "generated voice bank requires the trained Ref2VA reference layout"
                )
            if reference_audios:
                raise ValueError(
                    "generated voice bank is only used when no public voice reference exists"
                )
        if is_long_horizon:
            conditional_video = bool(
                first_frame is not None
                or last_frame is not None
                or reference_videos
            )
            if bounded_token_memory and reference_videos:
                raise ValueError(
                    "bounded AV token memory does not yet support reference videos"
                )
            if conditional_video or bounded_token_memory:
                # Causal continuation owns FL2VA endpoint anchors and bounded
                # reference-style AV memory.  Temporal reference-video rows
                # remain excluded until their clock can be mapped into every
                # continuation window without changing user semantics.
                from .long_horizon import (
                    DEFAULT_AUDIO_BRIDGE_CONTEXT_FRAMES,
                    plan_long_horizon,
                )

                if spec.long_video is not None:
                    from ..long_video import compile_window_story
                    long_plan, authored_preview = compile_window_story(
                        spec.long_video, seed=spec.seed, maximum_frames=spec.frames,
                        service_family=spec.service_family,
                        first_frame=first_frame is not None,
                        last_frame=last_frame is not None,
                    )
                else:
                    long_plan = plan_long_horizon(
                        requested_duration_seconds=spec.requested_duration_seconds,
                        prompt=spec.prompt,
                        seed=spec.seed,
                        maximum_opening_frames=spec.frames,
                        context_frames=(
                            DEFAULT_AUDIO_BRIDGE_CONTEXT_FRAMES
                            if bounded_token_memory
                            else 39
                        ),
                        structured_director=structured_director,
                    )
            else:
                from .global_co_denoise import plan_global_av_windows
                from .long_horizon import localize_h3_prompt

                global_co_plan = plan_global_av_windows(
                    int(spec.output_frames),
                )
                global_co_prompts = tuple(
                    localize_h3_prompt(
                        spec.prompt,
                        context_start_frame=window.start_frame,
                        visible_start_frame=window.start_frame,
                        visible_stop_frame=window.stop_frame,
                        segment_index=window.index,
                        timeline_stop_frame=global_co_plan.output_frames,
                        structured_director=structured_director,
                    )
                    for window in global_co_plan.windows
                )
            planned_output = (
                global_co_plan.output_frames
                if global_co_plan is not None
                else long_plan.output_frames
            )
            if planned_output != spec.output_frames:
                raise RuntimeError(
                    "long-horizon contract and native planner disagree on output frames"
                )
        joint_plan = None
        joint_plan_second = None
        stage_joint_actual_steps: tuple[int, ...] | None = None
        stage_joint_attention_schedule: tuple[tuple[int, int, str], ...] | None = None
        stage_joint_plan_summary: dict[str, Any] | None = None
        second_attention_schedule: tuple[tuple[int, int, str], ...] = ()
        second_plan_summary: dict[str, Any] | None = None
        second_forecast_actual_steps: tuple[int, ...] = ()
        second_sampling_sampler = "sa_solver"
        # INT8 V19 and distilled LoRA deliberately have separate scheduling
        # domains.  A Base release bundle (or isolated Base research overlay)
        # must never make LoRA borrow a forecast trajectory calibrated for the
        # 20-step model.
        use_v19 = bool(
            not is_second_sampling
            and
            spec.model_variant == "base"
            and
            spec.joint_acceleration_enabled
            and getattr(
                self._factory,
                "v19_scheduler_enabled",
                getattr(self._factory, "v19_release_enabled", False),
            )
        )
        joint_scheduler_id = None
        if (spec.joint_acceleration_enabled or is_second_sampling) and not use_v19:
            from .planner import (
                FROZEN_INT8_JOINT_POLICY,
                H3JointAccelerationScheduler,
                H3LoraAccelerationScheduler,
                H3WorkloadAnalyzer,
                JointWorkloadContext,
                LORA_NO_FORECAST_SCHEDULER_ID,
            )

            latent_frames = H3WorkloadAnalyzer.video_latent_frames(runtime_frames)
            spatial_tokens = (spec.height // 32) * (spec.width // 32)
            visual_condition_count = (
                int(first_frame is not None)
                + int(last_frame is not None)
                + len(reference_images)
                + len(reference_videos)
            )
            # Exact Qwen tokenisation happens later in the hot session.  Text
            # is <2% of the calibrated packed sequence, so this bounded
            # estimate selects/interpolates the correct shape cost model
            # without performing text encoding twice.
            estimated_text_tokens = max(
                128, min(1024, int(math.ceil(len(spec.prompt) * 0.55)))
            )
            audio_tokens = 2 * round((runtime_frames / 24.0) * 40.0)
            workload = JointWorkloadContext(
                packed_tokens=(
                    latent_frames * spatial_tokens
                    + visual_condition_count * spatial_tokens
                    + audio_tokens
                    + estimated_text_tokens
                ),
                condition_count=visual_condition_count,
                service_family=spec.service_family,
                model_variant=spec.model_variant,
            )

            # Research runs can pin a version without changing the two-field
            # public request contract or silently moving the release default.
            # The selected id remains serialized in request telemetry.
            requested_steps = (
                int(second_sampling.steps)
                if second_sampling is not None
                else int(
                    spec.sampling_steps
                    or (
                        self._built.lora_default_steps
                        if spec.model_variant == "lora"
                        else 20
                    )
                )
            )
            requested_acceleration = (
                float(second_sampling.acceleration)
                if second_sampling is not None
                else float(spec.acceleration or 0.0)
            )
            requested_second_acceleration = (
                requested_acceleration
                if second_sampling is not None
                else float(
                    spec.second_pass_acceleration
                    if spec.second_pass_acceleration is not None
                    else requested_acceleration
                )
            )
            second_sampling_sampler_profile = os.environ.get(
                "H3_SECOND_SAMPLING_SAMPLER_PROFILE", ""
            ).strip().lower()
            if second_sampling_sampler_profile:
                if not is_second_sampling:
                    raise ValueError(
                        "H3_SECOND_SAMPLING_SAMPLER_PROFILE is only valid for "
                        "second sampling"
                    )
                if second_sampling_sampler_profile != "res_multistep_v1":
                    raise ValueError(
                        "unsupported H3 second-sampling sampler profile: "
                        f"{second_sampling_sampler_profile}"
                    )
                if spec.model_variant != "base":
                    raise ValueError(
                        "res_multistep_v1 requires the original H3 Base weights"
                    )
                second_sampling_sampler = "res_multistep"
            if spec.model_variant == "lora":
                policy_id = (
                    os.environ.get(
                        "H3_NATIVE_RESEARCH_LORA_JOINT_POLICY", ""
                    ).strip()
                    or FROZEN_INT8_JOINT_POLICY
                )
                scheduler = H3LoraAccelerationScheduler(policy_id=policy_id)
                joint_plan = scheduler.plan(
                    max(4, requested_steps) if is_second_sampling else requested_steps,
                    requested_acceleration,
                    workload=workload,
                )
                if (
                    not is_second_sampling
                    and spec.acceleration_transition_step is not None
                    and spec.acceleration_transition_step < requested_steps
                    and not math.isclose(
                        requested_acceleration, requested_second_acceleration
                    )
                ):
                    joint_plan_second = scheduler.plan(
                        requested_steps,
                        requested_second_acceleration,
                        workload=workload,
                    )
                joint_scheduler_id = LORA_NO_FORECAST_SCHEDULER_ID
            else:
                policy_id = (
                    os.environ.get(
                        "H3_NATIVE_RESEARCH_JOINT_POLICY", ""
                    ).strip()
                    or FROZEN_INT8_JOINT_POLICY
                )
                scheduler = H3JointAccelerationScheduler(policy_id=policy_id)
                joint_plan = scheduler.plan(
                    max(4, requested_steps) if is_second_sampling else requested_steps,
                    requested_acceleration,
                    # UltimateUpscale is a short, low-noise trajectory.  All
                    # solver positions remain real DiT evaluations; the same
                    # acceleration control is projected only onto Attention.
                    allow_forecast=not is_second_sampling,
                    workload=workload,
                )
                if (
                    not is_second_sampling
                    and spec.acceleration_transition_step is not None
                    and spec.acceleration_transition_step < requested_steps
                    and not math.isclose(
                        requested_acceleration, requested_second_acceleration
                    )
                ):
                    joint_plan_second = scheduler.plan(
                        requested_steps,
                        requested_second_acceleration,
                        allow_forecast=True,
                        workload=workload,
                    )
                joint_scheduler_id = (
                    "h3_second_sampling_exact_attention_v1"
                    if is_second_sampling
                    else "h3_int8_frozen_round229"
                )
            if joint_plan_second is not None:
                split = int(spec.acceleration_transition_step or requested_steps)
                first_schedule = joint_plan.runtime_action_schedule()
                second_schedule = joint_plan_second.runtime_action_schedule()
                actual = {
                    step for step in joint_plan.actual_step_indices
                    if step < split
                }
                actual.update(
                    step for step in joint_plan_second.actual_step_indices
                    if step >= split
                )
                forced_actual: set[int] = set()
                if spec.selflift_enabled:
                    forced_actual.update(range(max(0, split - 1), requested_steps))
                    converted_to_actual = forced_actual - actual
                    actual.update(forced_actual)
                else:
                    converted_to_actual = set()
                stage_joint_actual_steps = tuple(sorted(actual))
                stage_rows = {
                    (step, layer): action
                    for (step, layer), action in first_schedule.items()
                    if (
                        step < split
                        and step not in converted_to_actual
                        and action != "dense"
                    )
                }
                stage_rows.update({
                    (step, layer): action
                    for (step, layer), action in second_schedule.items()
                    if (
                        step >= split
                        and step not in converted_to_actual
                        and action != "dense"
                    )
                })
                stage_joint_attention_schedule = tuple(
                    (step, layer, action)
                    for (step, layer), action in sorted(stage_rows.items())
                )
                stage_joint_plan_summary = {
                    "schema_version": "h3_stage_acceleration_v1",
                    "policy_id": "h3_stage_acceleration_v1",
                    "scheduler_family": joint_scheduler_id,
                    "model_variant": spec.model_variant,
                    "acceleration": requested_acceleration,
                    "second_pass_acceleration": requested_second_acceleration,
                    "acceleration_transition_step": split,
                    "actual_step_indices": list(stage_joint_actual_steps),
                    "forecast_step_indices": [
                        step for step in range(requested_steps)
                        if step not in actual
                    ],
                    "actual_evaluations": len(stage_joint_actual_steps),
                    "forecast_evaluations": (
                        requested_steps - len(stage_joint_actual_steps)
                    ),
                    "stage_plans": {
                        "first_pass": {
                            key: value for key, value in joint_plan.to_dict().items()
                            if key != "attention_decisions"
                        },
                        "second_pass": {
                            key: value for key, value in joint_plan_second.to_dict().items()
                            if key != "attention_decisions"
                        },
                    },
                }
            if is_second_sampling and requested_steps < 4:
                # The first-pass optimizer's certified domain starts at four
                # trajectory points.  UltimateUpscale commonly uses one real
                # low-noise evaluation, so project the terminal N rows of the
                # four-step exact-only Attention policy onto N refinement
                # rows.  This never invents Forecast steps and preserves the
                # terminal layer protection learned by the optimizer.
                provisional = joint_plan
                terminal_start = provisional.total_steps - requested_steps
                remapped = []
                for (step, layer), action in sorted(
                    provisional.runtime_action_schedule().items()
                ):
                    if step >= terminal_start:
                        remapped.append((step - terminal_start, layer, action))
                second_attention_schedule = tuple(remapped)
                second_plan_summary = {
                    **{
                        key: value
                        for key, value in provisional.to_dict().items()
                        if key != "attention_decisions"
                    },
                    "schema_version": "h3_second_sampling_attention_projection_v1",
                    "total_steps": requested_steps,
                    "actual_step_indices": list(range(requested_steps)),
                    "forecast_step_indices": [],
                    "actual_evaluations": requested_steps,
                    "forecast_evaluations": 0,
                    "projection_source_steps": provisional.total_steps,
                    "projection_source_terminal_start": terminal_start,
                    "scheduler_family": joint_scheduler_id,
                    "model_variant": spec.model_variant,
                }
                joint_plan = None
            second_sampling_attention_profile = os.environ.get(
                "H3_SECOND_SAMPLING_ATTENTION_PROFILE", ""
            ).strip().lower()
            if second_sampling_attention_profile:
                if not is_second_sampling:
                    raise ValueError(
                        "H3_SECOND_SAMPLING_ATTENTION_PROFILE is only valid for "
                        "second sampling"
                    )
                if second_sampling_attention_profile != "dense_tail_v1":
                    raise ValueError(
                        "unsupported H3 second-sampling attention profile: "
                        f"{second_sampling_attention_profile}"
                    )
                if requested_steps < 5:
                    raise ValueError(
                        "dense_tail_v1 requires at least five H3 evaluations"
                    )

                # The opening evaluation establishes the target-grid scene and
                # the last two low-noise evaluations remove structured errors.
                # Only the intervening trajectory is approximated.  In
                # particular, no sparse cell is allowed to write the terminal
                # clean-state prediction: motion-dependent lattice artefacts
                # then receive two complete H3 corrections before decoding.
                sparse_middle_steps = tuple(range(1, requested_steps - 2))
                sparse_action = "forecastfrontier:sparse_topk_0.25"
                second_attention_schedule = tuple(
                    (step, layer, sparse_action)
                    for step in sparse_middle_steps
                    for layer in range(50)
                )
                dense_steps = tuple(
                    step
                    for step in range(requested_steps)
                    if step not in sparse_middle_steps
                )
                # Round229's measured 25% action costs 0.403 of complete
                # Attention. Non-Attention work is unchanged at 0.286 units.
                sparse_step_units = 0.286 + (1.0 - 0.286) * 0.403
                estimated_compute_units = (
                    len(dense_steps)
                    + len(sparse_middle_steps) * sparse_step_units
                )
                second_plan_summary = {
                    "schema_version": "h3_second_sampling_dense_tail_v1",
                    "total_steps": requested_steps,
                    "acceleration": requested_acceleration,
                    "actual_step_indices": list(range(requested_steps)),
                    "forecast_step_indices": [],
                    "actual_evaluations": requested_steps,
                    "forecast_evaluations": 0,
                    "dense_step_indices": list(dense_steps),
                    "sparse_middle_step_indices": list(sparse_middle_steps),
                    "sparse_action": sparse_action,
                    "estimated_compute_units": round(
                        estimated_compute_units, 6
                    ),
                    "dense_compute_units": float(requested_steps),
                    "estimated_compute_ratio": round(
                        estimated_compute_units / requested_steps, 6
                    ),
                    "policy_id": "h3_second_sampling_dense_tail_v1",
                    "safety_envelope": (
                        "opening_dense_plus_two_terminal_dense_evaluations"
                    ),
                    "scheduler_family": "h3_second_sampling_exact_steps_v1",
                    "model_variant": spec.model_variant,
                }
                joint_plan = None
            second_sampling_forecast_profile = os.environ.get(
                "H3_SECOND_SAMPLING_FORECAST_PROFILE", ""
            ).strip().lower()
            if second_sampling_forecast_profile:
                if not is_second_sampling:
                    raise ValueError(
                        "H3_SECOND_SAMPLING_FORECAST_PROFILE is only valid for "
                        "second sampling"
                    )
                if second_sampling_forecast_profile != "dense_4of8_v1":
                    raise ValueError(
                        "unsupported H3 second-sampling Forecast profile: "
                        f"{second_sampling_forecast_profile}"
                    )
                if spec.model_variant != "base":
                    raise ValueError(
                        "dense_4of8_v1 requires the original H3 Base weights"
                    )
                if requested_steps != 8:
                    raise ValueError(
                        "dense_4of8_v1 requires exactly eight solver positions"
                    )
                if requested_acceleration != 0.0:
                    raise ValueError(
                        "dense_4of8_v1 owns its compute schedule; acceleration "
                        "must remain zero"
                    )
                if second_sampling_attention_profile:
                    raise ValueError(
                        "dense_4of8_v1 cannot be combined with approximate "
                        "Attention"
                    )

                # Four complete H3 evaluations match the dense four-step
                # comparator.  The four intervening solver positions execute
                # only the first three blocks and forecast the expensive tail
                # from dense histories.  Every real Attention cell remains
                # complete, so this route cannot imprint Sparge's spatial
                # selection lattice into moving objects.  A final complete
                # evaluation owns the decoded low-noise state.
                second_forecast_actual_steps = (0, 1, 4, 7)
                second_attention_schedule = ()
                second_plan_summary = {
                    "schema_version": "h3_second_sampling_dense_forecast_v1",
                    "total_steps": requested_steps,
                    "acceleration": requested_acceleration,
                    "actual_step_indices": list(second_forecast_actual_steps),
                    "forecast_step_indices": [2, 3, 5, 6],
                    "actual_evaluations": len(second_forecast_actual_steps),
                    "forecast_evaluations": (
                        requested_steps - len(second_forecast_actual_steps)
                    ),
                    "actual_attention_cells": {
                        "dense": len(second_forecast_actual_steps) * 50,
                    },
                    "forecast_anchor_attention_cells": {
                        "dense": (
                            requested_steps - len(second_forecast_actual_steps)
                        ) * 3,
                    },
                    "forecast_anchor_depth": 3,
                    "policy_id": "h3_second_sampling_dense_4of8_v1",
                    "safety_envelope": (
                        "all_actual_cells_dense_plus_terminal_dense_evaluation"
                    ),
                    "scheduler_family": "h3_second_sampling_sa_forecast_v1",
                    "model_variant": spec.model_variant,
                }
                joint_plan = None
        sparse_requested = (
            use_v19 and max(
                float(spec.acceleration or 0.0),
                float(
                    spec.second_pass_acceleration
                    if spec.second_pass_acceleration is not None
                    else spec.acceleration or 0.0
                ),
            ) > 0.0
        ) or (
            joint_plan is not None and joint_plan.uses_sparse_attention
        ) or (
            stage_joint_attention_schedule is not None
            and any(
                action != "dense"
                for _step, _layer, action in stage_joint_attention_schedule
            )
        ) or (
            any(action != "dense" for _step, _layer, action in second_attention_schedule)
        ) or (
            spec.advanced
            and not spec.joint_acceleration_enabled
            and spec.attention_keep_ratio < 1.0
        )
        if sparse_requested and not self._factory.sparse_attention_available:
            raise RuntimeError(
                "sparse attention is not installed for this service; "
                "use acceleration=0 or install the optional SM89 runtime"
            )

        output_path = output_path.resolve()
        if not output_path.is_relative_to(self._output_root):
            raise ValueError("output_path must stay inside the configured output root")
        async with self._lock:
            if cancel_event.is_set():
                raise NativeGenerationCancelled("native H3 generation cancelled")
            built = await asyncio.to_thread(
                self._ensure_session, spec.runtime_launcher
            )
            if (
                second_sampling is not None
                and second_sampling.model_variant == "lora"
                and second_sampling.steps not in built.lora_recommended_steps
            ):
                raise ValueError(
                    "the selected H3 LoRA does not support "
                    f"{second_sampling.steps} steps; supported values are "
                    + ", ".join(str(value) for value in built.lora_recommended_steps)
                )
            runtime_config = getattr(built.session, "runtime_config", None)
            allocator_ceiling_gib = (
                float(runtime_config.max_device_bytes) / 1024**3
                if runtime_config is not None
                else {"24gb": 23.25, "16gb": 15.25, "8gb": 7.25}[
                    spec.vram_profile
                ]
            )
            capacity_attention_schedule = None
            capacity_plan_summary = None
            if is_second_sampling:
                assert second_sampling is not None
                total_steps = second_sampling.steps
                actual_steps = (
                    second_forecast_actual_steps
                    if second_forecast_actual_steps
                    else tuple(range(total_steps))
                )
            elif use_v19:
                total_steps = int(spec.sampling_steps or 20)
                actual_steps = tuple(range(total_steps))
            elif joint_plan is not None:
                actual_steps = (
                    stage_joint_actual_steps
                    if stage_joint_actual_steps is not None
                    else joint_plan.actual_step_indices
                )
                total_steps = joint_plan.total_steps
                # Forecast and the sparse-action backend both retain extra
                # preparation state before block zero.  On the 8-GiB backend
                # a >=512-MiB hidden state leaves insufficient room for the
                # first 1-GiB RMS workspace even though compact Dense fits.
                # At this hard capacity boundary prefer the exact Dense
                # trajectory; smaller requests and larger backends retain the
                # full Forecast+sparsity acceleration surface.
                hidden_bytes = int(workload.packed_tokens) * 5376 * 2
                approximate_attention_cells = any(
                    action != "dense"
                    for action in (
                        (
                            row[2] for row in stage_joint_attention_schedule
                        )
                        if stage_joint_attention_schedule is not None
                        else joint_plan.runtime_action_schedule().values()
                    )
                )
                if (
                    runtime_config is not None
                    and runtime_config.resource_profile == "w4a8_8gb"
                    and hidden_bytes >= 512 * 1024**2
                    and (
                        len(actual_steps) < total_steps
                        or approximate_attention_cells
                        or (
                            stage_joint_plan_summary is None
                            and joint_plan.online_guard_id is not None
                        )
                    )
                ):
                    actual_steps = tuple(range(total_steps))
                    capacity_attention_schedule = ()
                    capacity_plan_summary = {
                        **{
                            key: value
                            for key, value in (
                                stage_joint_plan_summary
                                if stage_joint_plan_summary is not None
                                else joint_plan.to_dict()
                            ).items()
                            if key != "attention_decisions"
                        },
                        "actual_step_indices": list(actual_steps),
                        "forecast_step_indices": [],
                        "actual_evaluations": total_steps,
                        "forecast_evaluations": 0,
                        "reason": "w4a8_8gb_long_hidden_approximation_capacity_guard",
                        "capacity_guard": {
                            "policy_id": "w4a8_8gb_long_hidden_dense_capacity_v1",
                            "hidden_bytes": hidden_bytes,
                            "hidden_gib": hidden_bytes / 1024**3,
                            "threshold_bytes": 512 * 1024**2,
                            "attention_projection": "dense_capacity_fallback",
                        },
                        "technique_mix": {
                            "actual_dit_evaluations": total_steps,
                            "forecast_evaluations": 0,
                            "actual_attention_cells": {"dense": total_steps * 50},
                            "forecast_anchor_attention_cells": {},
                            "coupled_techniques": [
                                "exact_runtime",
                                "capacity_guard",
                            ],
                        },
                    }
            else:
                actual_steps = (
                    actual_step_schedule(int(spec.preset["actual_steps"]))
                    if spec.engine in ("original", "reference")
                    else tuple(range(int(spec.preset["steps"])))
                )
                total_steps = (
                    20
                    if spec.engine in ("original", "reference")
                    else int(spec.preset["steps"])
                )
            candidate = (
                None
                if joint_plan is not None or use_v19 or is_second_sampling or is_incremental_continuation
                else select_candidate(
                    spec,
                    first_frame=first_frame,
                    last_frame=last_frame,
                    reference_images=reference_images,
                    reference_videos=reference_videos,
                    reference_audios=reference_audios,
                )
            )
            execution_plan = self._request_plan(spec)
            if (is_second_sampling or is_long_horizon or is_incremental_continuation) and execution_plan is None:
                from .planner import ExecutionPlan
                from .runtime import OffloadMode

                execution_plan = ExecutionPlan(
                    offload_mode=OffloadMode.BLOCK,
                    mlp_chunk_tokens=8192,
                    block_buffer_count=2,
                    prefetch_depth=1,
                    vae_spatial_tile=(288, 288),
                )
            if joint_plan is not None or use_v19 or is_second_sampling:
                if execution_plan is None:
                    raise RuntimeError(
                        "joint acceleration requires an explicit RTX 4090 execution plan"
                    )
                # The scheduler is allowed to trade only sampler/Attention
                # compute.  Keep the mature Round86/143 mechanical baseline
                # underneath every versioned joint policy so a new control-plane
                # version cannot accidentally benchmark against slower,
                # unfused runtime defaults.
                execution_plan = replace(
                    execution_plan,
                    fused_rms_adaln=True,
                    vae_transformer_block_compile=True,
                )
            if (
                global_selflift_plan is not None
                and spec.selflift_temporal_window_enabled
            ):
                if execution_plan is None:
                    raise RuntimeError(
                        "windowed global SelfLift requires an explicit execution plan"
                    )
                # A short DiT view alone cannot lower the end-to-end peak when
                # the final Video-VAE later materialises every decoded frame on
                # CUDA.  The native VAE sink consumes each already blended
                # causal 17-frame piece, performs the exact uint8 pixel
                # transform and copies it to the final CPU timeline.  This is
                # an output-transport change only: VAE weights, temporal
                # receptive fields and encoded MP4 bytes remain unchanged.
                execution_plan = replace(
                    execution_plan,
                    vae_temporal_tile=17,
                )
            if candidate is not None:
                if not self._factory.sparse_attention_available:
                    raise RuntimeError(
                        "the reviewed long-video route requires the pinned "
                        "SM89 sparse-attention runtime"
                    )
                if execution_plan is None:
                    raise RuntimeError(
                        "the reviewed long-video route has no explicit "
                        "RTX 4090 execution plan"
                    )
                execution_plan = replace(
                    execution_plan,
                    fused_rms_adaln=True,
                    dense_qk_quant_gran="per_warp",
                    vae_transformer_block_compile=True,
                    long_video_motion_detail_attention=True,
                )
            preview_step = None
            preview_output = None
            preview_latents_output = None
            checkpoint_after_step = None
            latent_only_selflift_finish = False
            if spec.execution_mode == "checkpoint" and resume_checkpoint_path is None:
                checkpoint_after_step = spec.checkpoint_step
                if checkpoint_path is None:
                    raise RuntimeError("checkpoint task has no persistence path")
                # Every SelfLift fork persists a clean source-grid x0 for the
                # next continuation and eventual global finalization.  Online
                # creation may additionally decode a disposable preview;
                # one-click JSON creation stores the latent only.
                if spec.selflift_enabled:
                    preview_latents_output = final_latents_path
                if spec.checkpoint_preview:
                    preview_step = int(spec.checkpoint_step or 1) - 1
                    preview_output = output_path.with_name(
                        output_path.stem + ".checkpoint-preview.mp4"
                    )
                    # The preview branch is also the clean low-resolution
                    # trajectory consumed by the next infinite window.  Keep
                    # it separately from the lifted formal checkpoint.
                    preview_latents_output = final_latents_path
                elif spec.selflift_enabled:
                    # One-click/JSON creation does not decode a disposable
                    # preview, but its audio must still finish the exact
                    # low-resolution sigma suffix once. The later spatial
                    # windows refine video while reusing this locked audio x0.
                    latent_only_selflift_finish = True
                    preview_step = int(spec.checkpoint_step or 1) - 1
            elif spec.preview_mode != "off":
                if spec.preview_step_index is not None:
                    preview_step = spec.preview_step_index
                elif spec.model_variant == "lora":
                    preview_step = max(1, min(int(spec.preset["steps"]) - 2, int(spec.preset["steps"]) // 2))
                elif use_v19:
                    # V19 owns the actual/forecast schedule only after exact
                    # tokenisation.  Use the standard three-quarter protected
                    # anchor and let the certified selector require it as an
                    # actual evaluation; if no admitted trajectory contains
                    # it, routing fails closed to Dense rather than decoding a
                    # low-quality forecast x0 estimate.
                    preview_step = max(
                        1,
                        min(total_steps - 2, round(total_steps * 0.75)),
                    )
                else:
                    # Select a real-compute anchor around two thirds of the
                    # calibrated actual evaluations, never a forecast point.
                    preview_step = actual_steps[min(len(actual_steps) - 2, (2 * len(actual_steps)) // 3)]
                preview_output = output_path.with_name(output_path.stem + ".preview.mp4")
            preview_scale = 1.0
            fast_preview = (
                (spec.execution_mode == "checkpoint" and spec.checkpoint_preview)
                or latent_only_selflift_finish
                or (spec.preview_mode != "off" and spec.preview_fast_finish)
            )
            if (
                fast_preview
                and not latent_only_selflift_finish
                and spec.checkpoint_preview_resolution != "source"
            ):
                preview_scale = min(
                    1.0,
                    int(spec.checkpoint_preview_resolution[:-1])
                    / float(min(spec.width, spec.height)),
                )
            ultimate_plan = None
            if second_sampling is not None:
                from .ultimate_upscale import plan_ultimate_upscale

                ultimate_plan = plan_ultimate_upscale(
                    target_width=second_sampling.width,
                    target_height=second_sampling.height,
                    frames=runtime_frames,
                    device_budget_bytes=built.session._device_execution_budget_bytes(),
                    text_tokens=max(
                        128, min(1024, int(math.ceil(len(spec.prompt) * 0.55)))
                    ),
                    condition_count=(
                        int(first_frame is not None)
                        + int(last_frame is not None)
                        + len(reference_images)
                        + len(reference_videos)
                        + len(reference_audios)
                    ),
                    engine=spec.engine,
                    actual_evaluations=second_sampling.steps,
                    requested_mode=second_sampling.memory_mode,
                    weight_tier=built.weight_tier,
                    resource_profile=built.session.runtime_config.resource_profile,
                    allow_spatial_tiles=False,
                    temporal_window_frames=(
                        second_sampling.temporal_window_frames
                    ),
                    temporal_overlap_frames=(
                        second_sampling.temporal_overlap_frames
                    ),
                )
                if not ultimate_plan.full_canvas and len(ultimate_plan.spatial) != 1:
                    raise RuntimeError(
                        "this second-sampling target requires spatial tiles; "
                        "the release executor currently enables the faster full-spatial "
                        "temporal-window executor only"
                    )

            h3_detail_regeneration = bool(
                second_sampling is not None
                and os.environ.get(
                    "H3_SECOND_SAMPLING_DETAIL_REGENERATION", "0"
                ).strip().lower()
                in {"1", "true", "yes", "on"}
            )
            refinement_video_shift = (
                float(built.session.lora_video_shift)
                if second_sampling is not None
                and second_sampling.model_variant == "lora"
                else float(os.environ.get("H3_SECOND_SAMPLING_VIDEO_SHIFT", "12.0"))
                if h3_detail_regeneration
                else 6.0
            )
            refinement_sigma_power = (
                float(
                    os.environ.get(
                        "H3_SECOND_SAMPLING_SIGMA_POWER", "1.0"
                    )
                )
                if second_sampling is not None
                and second_sampling.model_variant == "base"
                else 1.0
            )
            refinement_prediction_low_frequency_gain = (
                float(
                    os.environ.get(
                        "H3_SECOND_SAMPLING_PREDICTION_LOW_FREQUENCY_GAIN",
                        "0.60",
                    )
                )
                if h3_detail_regeneration
                else 1.0
            )
            refinement_final_low_frequency_gain = (
                float(
                    os.environ.get(
                        "H3_SECOND_SAMPLING_FINAL_LOW_FREQUENCY_GAIN",
                        "0.10",
                    )
                )
                if h3_detail_regeneration
                else 1.0
            )
            refinement_temporal_lowpass = bool(
                h3_detail_regeneration
                and os.environ.get(
                    "H3_SECOND_SAMPLING_TEMPORAL_LOWPASS", "1"
                ).strip().lower()
                in {"1", "true", "yes", "on"}
            )
            refinement_temporal_outlier_only = bool(
                refinement_temporal_lowpass
                and os.environ.get(
                    "H3_SECOND_SAMPLING_TEMPORAL_OUTLIER_ONLY", "1"
                ).strip().lower()
                in {"1", "true", "yes", "on"}
            )
            refinement_temporal_detail_outlier_strength = (
                float(
                    os.environ.get(
                        "H3_SECOND_SAMPLING_TEMPORAL_DETAIL_OUTLIER_STRENGTH",
                        "0.0",
                    )
                )
                if h3_detail_regeneration
                else 0.0
            )
            refinement_cross_step_detail_strength = (
                float(
                    os.environ.get(
                        "H3_SECOND_SAMPLING_CROSS_STEP_DETAIL_STRENGTH",
                        "0.35",
                    )
                )
                if h3_detail_regeneration
                and second_sampling is not None
                and second_sampling.steps >= 2
                else 0.0
            )
            full_canvas_regions: tuple[
                tuple[float, float, float, float], ...
            ] = ()
            raw_full_canvas_regions = os.environ.get(
                "H3_SECOND_SAMPLING_FULL_CANVAS_REGIONS", ""
            ).strip()
            if second_sampling is not None and raw_full_canvas_regions:
                try:
                    parsed_full_canvas_regions = json.loads(
                        raw_full_canvas_regions
                    )
                    full_canvas_regions = tuple(
                        tuple(float(value) for value in region)
                        for region in parsed_full_canvas_regions
                    )
                except (TypeError, ValueError, json.JSONDecodeError) as error:
                    raise ValueError(
                        "H3_SECOND_SAMPLING_FULL_CANVAS_REGIONS must be a JSON "
                        "array of normalized [x0,y0,x1,y1] boxes"
                    ) from error
            roi_regions: tuple[tuple[float, float, float, float], ...] = ()
            raw_roi_regions = os.environ.get(
                "H3_SECOND_SAMPLING_ROI_REGIONS", ""
            ).strip()
            if second_sampling is not None and raw_roi_regions:
                try:
                    parsed_roi_regions = json.loads(raw_roi_regions)
                    roi_regions = tuple(
                        tuple(float(value) for value in region)
                        for region in parsed_roi_regions
                    )
                except (TypeError, ValueError, json.JSONDecodeError) as error:
                    raise ValueError(
                        "H3_SECOND_SAMPLING_ROI_REGIONS must be a JSON array "
                        "of normalized [x0,y0,x1,y1] boxes"
                    ) from error
            roi_auto = bool(
                second_sampling is not None
                and os.environ.get(
                    "H3_SECOND_SAMPLING_ROI_AUTO", "0"
                ).strip().lower()
                in {"1", "true", "yes", "on"}
            )
            if roi_regions and roi_auto:
                raise ValueError(
                    "manual H3 second-sampling ROI regions cannot be combined "
                    "with automatic ROI selection"
                )
            roi_enabled = bool(roi_regions or roi_auto)
            roi_steps = (
                int(
                    os.environ.get(
                        "H3_SECOND_SAMPLING_ROI_STEPS",
                        "6" if roi_auto else "1",
                    )
                )
                if roi_enabled
                else 0
            )

            refinement_prompt = spec.prompt
            raw_refinement_prompt = os.environ.get(
                "H3_SECOND_SAMPLING_PROMPT_OVERRIDE", ""
            ).strip()
            if second_sampling is not None and raw_refinement_prompt:
                # Lab-only bridge for testing tightly cropped scene regions.
                # A crop that fills the H3 canvas no longer has the semantics
                # of the full source shot, so retaining the original global
                # prompt can actively fight the local reconstruction.
                refinement_prompt = raw_refinement_prompt

            selflift_initial_geometry = (
                resolve_geometry(
                    spec.selflift_initial_resolution,
                    spec.aspect_ratio,
                )
                if spec.selflift_enabled and not is_global_selflift
                else (None, None)
            )
            scheduler_required_actual_steps: set[int] = set()
            if use_v19 and spec.execution_mode == "checkpoint":
                scheduler_required_actual_steps.add(
                    int(spec.checkpoint_step or 1) - 1
                )
            if use_v19 and spec.selflift_enabled:
                # SelfLift must lift an actual clean prediction, and Base RES
                # must then rebuild every target-grid history entry from real
                # H3 evaluations. V19 may still accelerate those evaluations
                # through its per-layer sparse Attention schedule.
                scheduler_required_actual_steps.update(range(
                    int(spec.selflift_transition_step or 1) - 1,
                    total_steps,
                ))

            request = HotSessionRequest(
                prompt=refinement_prompt,
                seed=spec.seed,
                width=spec.width,
                height=spec.height,
                frames=runtime_frames,
                fps=24,
                steps=total_steps,
                output_path=output_path,
                actual_step_indices=actual_steps,
                execution_plan=execution_plan,
                release_byte_exact_optimizations=True,
                memory_mode=spec.memory_mode,
                attention_action_schedule=(
                    second_attention_schedule
                    if second_attention_schedule
                    else capacity_attention_schedule
                    if capacity_attention_schedule is not None
                    else stage_joint_attention_schedule
                    if stage_joint_attention_schedule is not None
                    else ()
                    if joint_plan is None
                    else tuple(
                        (step, layer, action)
                        for (step, layer), action in sorted(
                            joint_plan.runtime_action_schedule().items()
                        )
                    )
                ),
                attention_online_guard_id=(
                    None
                    if (
                        joint_plan is None
                        or capacity_plan_summary is not None
                        or stage_joint_plan_summary is not None
                    )
                    else joint_plan.online_guard_id
                ),
                attention_online_budget_dense_layers=(
                    0.0
                    if (
                        joint_plan is None
                        or capacity_plan_summary is not None
                        or stage_joint_plan_summary is not None
                        or joint_plan.online_guard_id is None
                    )
                    else joint_plan.online_recovery_reserve_units * 50.0
                ),
                attention_online_rebate_schedule=(
                    ()
                    if (
                        joint_plan is None
                        or capacity_plan_summary is not None
                        or stage_joint_plan_summary is not None
                    )
                    else joint_plan.online_rebate_schedule
                ),
                acceleration_plan_summary=(
                    second_plan_summary
                    if second_plan_summary is not None
                    else {
                        **capacity_plan_summary,
                        "scheduler_family": joint_scheduler_id,
                        "model_variant": spec.model_variant,
                    }
                    if capacity_plan_summary is not None
                    else stage_joint_plan_summary
                    if stage_joint_plan_summary is not None
                    else None
                    if joint_plan is None
                    else {
                        **{
                            key: value
                            for key, value in joint_plan.to_dict().items()
                            if key != "attention_decisions"
                        },
                        "scheduler_family": joint_scheduler_id,
                        "model_variant": spec.model_variant,
                    }
                ),
                v19_acceleration=(
                    float(spec.acceleration or 0.0) if use_v19 else None
                ),
                v19_second_pass_acceleration=(
                    float(
                        spec.second_pass_acceleration
                        if spec.second_pass_acceleration is not None
                        else spec.acceleration or 0.0
                    )
                    if use_v19 else None
                ),
                acceleration_transition_step=(
                    spec.acceleration_transition_step
                    if spec.joint_acceleration_enabled else None
                ),
                scheduler_required_actual_step_indices=tuple(sorted(
                    scheduler_required_actual_steps
                )),
                first_frame=first_frame,
                last_frame=last_frame,
                reference_images=reference_images,
                reference_videos=reference_videos,
                reference_audios=reference_audios,
                reference_image_resolution=spec.reference_image_resolution,
                reference_video_resolution=spec.reference_video_resolution,
                cancel_check=cancel_event.is_set,
                progress_callback=progress_callback,
                use_lora=engine_variant(spec.engine) == "lora",
                global_selflift_source_path=(
                    Path(global_selflift_source_path).resolve()
                    if global_selflift_source_path is not None
                    else None
                ),
                global_selflift_sigma_scale=float(spec.selflift_sigma_scale),
                refinement_latents_path=refinement_latents_path,
                external_refinement_video_path=(
                    Path(external_refinement_video_path).resolve()
                    if external_refinement_video_path is not None
                    else None
                ),
                refinement_denoise=(
                    None if second_sampling is None else second_sampling.denoise
                ),
                refinement_schedule_mode=refinement_schedule_mode,
                refinement_atlas_denoise_regions=(
                    refinement_atlas_denoise_regions
                ),
                refinement_handoff_latents_path=(
                    Path(refinement_handoff_latents_path).resolve()
                    if refinement_handoff_latents_path is not None
                    else None
                ),
                refinement_handoff_context_frames=(
                    int(refinement_handoff_context_frames)
                ),
                refinement_spatial_mode=(
                    "strict" if second_sampling is None else second_sampling.spatial_mode
                ),
                refinement_sampler=second_sampling_sampler,
                preserve_refinement_audio=(
                    True if second_sampling is None else second_sampling.preserve_audio
                ),
                refinement_video_shift=refinement_video_shift,
                refinement_sigma_power=refinement_sigma_power,
                refinement_prediction_low_frequency_gain=(
                    refinement_prediction_low_frequency_gain
                ),
                refinement_final_low_frequency_gain=(
                    refinement_final_low_frequency_gain
                ),
                refinement_temporal_lowpass=refinement_temporal_lowpass,
                refinement_temporal_outlier_only=(
                    refinement_temporal_outlier_only
                ),
                refinement_temporal_detail_outlier_strength=(
                    refinement_temporal_detail_outlier_strength
                ),
                refinement_cross_step_detail_strength=(
                    refinement_cross_step_detail_strength
                ),
                refinement_full_canvas_regions=full_canvas_regions,
                refinement_full_canvas_feather=(
                    float(
                        os.environ.get(
                            "H3_SECOND_SAMPLING_FULL_CANVAS_FEATHER", "0.02"
                        )
                    )
                    if full_canvas_regions
                    else 0.02
                ),
                refinement_roi_regions=roi_regions,
                refinement_roi_auto=roi_auto,
                refinement_roi_max_regions=(
                    int(
                        os.environ.get(
                            "H3_SECOND_SAMPLING_ROI_MAX_REGIONS", "4"
                        )
                    )
                    if roi_auto
                    else 4
                ),
                refinement_roi_min_side_fraction=(
                    float(
                        os.environ.get(
                            "H3_SECOND_SAMPLING_ROI_MIN_SIDE_FRACTION", "0.12"
                        )
                    )
                    if roi_auto
                    else 0.12
                ),
                refinement_roi_max_side_fraction=(
                    float(
                        os.environ.get(
                            "H3_SECOND_SAMPLING_ROI_MAX_SIDE_FRACTION", "0.28"
                        )
                    )
                    if roi_auto
                    else 0.28
                ),
                refinement_roi_steps=roi_steps,
                refinement_roi_denoise=(
                    float(
                        os.environ.get(
                            "H3_SECOND_SAMPLING_ROI_DENOISE", "0.075"
                        )
                    )
                    if roi_enabled
                    else 0.075
                ),
                refinement_roi_low_frequency_gain=(
                    float(
                        os.environ.get(
                            "H3_SECOND_SAMPLING_ROI_LOW_FREQUENCY_GAIN",
                            "0.08",
                        )
                    )
                    if roi_enabled
                    else 0.08
                ),
                refinement_roi_mid_frequency_gain=(
                    float(
                        os.environ.get(
                            "H3_SECOND_SAMPLING_ROI_MID_FREQUENCY_GAIN",
                            "0.0",
                        )
                    )
                    if roi_enabled
                    else 0.0
                ),
                refinement_roi_coarse_scale=(
                    float(
                        os.environ.get(
                            "H3_SECOND_SAMPLING_ROI_COARSE_SCALE", "0.25"
                        )
                    )
                    if roi_enabled
                    else 0.25
                ),
                refinement_roi_blend=(
                    float(
                        os.environ.get(
                            "H3_SECOND_SAMPLING_ROI_BLEND", "0.90"
                        )
                    )
                    if roi_enabled
                    else 0.90
                ),
                refinement_roi_temporal_outlier_strength=(
                    float(
                        os.environ.get(
                            "H3_SECOND_SAMPLING_ROI_TEMPORAL_OUTLIER_STRENGTH",
                            "0.65",
                        )
                    )
                    if roi_enabled
                    else 0.65
                ),
                refinement_roi_temporal_filter=(
                    os.environ.get(
                        "H3_SECOND_SAMPLING_ROI_TEMPORAL_FILTER", "outlier"
                    ).strip()
                    if roi_enabled
                    else "outlier"
                ),
                refinement_roi_atlas_height=(
                    int(
                        os.environ.get(
                            "H3_SECOND_SAMPLING_ROI_ATLAS_HEIGHT", "0"
                        )
                    )
                    if roi_enabled
                    else 0
                ),
                refinement_roi_atlas_width=(
                    int(
                        os.environ.get(
                            "H3_SECOND_SAMPLING_ROI_ATLAS_WIDTH", "0"
                        )
                    )
                    if roi_enabled
                    else 0
                ),
                refinement_roi_atlas_rows=(
                    int(
                        os.environ.get(
                            "H3_SECOND_SAMPLING_ROI_ATLAS_ROWS", "2"
                        )
                    )
                    if roi_enabled
                    else 2
                ),
                refinement_roi_atlas_columns=(
                    int(
                        os.environ.get(
                            "H3_SECOND_SAMPLING_ROI_ATLAS_COLUMNS", "3"
                        )
                    )
                    if roi_enabled
                    else 3
                ),
                refinement_roi_position_mode=(
                    os.environ.get(
                        "H3_SECOND_SAMPLING_ROI_POSITION_MODE", "atlas"
                    ).strip()
                    if roi_enabled
                    else "atlas"
                ),
                refinement_roi_attention_mode=(
                    os.environ.get(
                        "H3_SECOND_SAMPLING_ROI_ATTENTION_MODE",
                        "scheduled_sparse" if roi_auto else "dense",
                    ).strip()
                    if roi_enabled
                    else "dense"
                ),
                save_final_latents_path=final_latents_path,
                conditioning_cache_source_path=(
                    Path(conditioning_cache_source_path).resolve()
                    if conditioning_cache_source_path is not None
                    else Path(refinement_latents_path).resolve()
                    if second_sampling is not None and refinement_latents_path is not None
                    else None
                ),
                formal_resume_state_path=resume_checkpoint_path,
                checkpoint_after_step=checkpoint_after_step,
                checkpoint_state_path=(
                    checkpoint_path if checkpoint_after_step is not None else None
                ),
                preview_step_index=preview_step,
                preview_output_path=preview_output,
                preview_latents_path=preview_latents_output,
                preview_decode_mode=(
                    "fast_finish"
                    if fast_preview
                    else "direct_x0"
                ),
                preview_branch_steps=(
                    int(spec.sampling_steps or 0)
                    - int(spec.checkpoint_step or 0)
                    if latent_only_selflift_finish
                    else spec.checkpoint_preview_steps
                    if (
                        spec.execution_mode == "checkpoint"
                        and spec.checkpoint_preview
                    )
                    else spec.preview_branch_steps
                ),
                # The ordinary API preview keeps its historical direct-x0
                # path.  Explicit fast-finish clients (ComfyUI/checkpoints)
                # may instead request a disposable LoRA branch and a smaller
                # preview canvas without mutating the formal trajectory.
                preview_branch_spatial_scale=preview_scale,
                preview_branch_warm_history=(
                    latent_only_selflift_finish
                    and spec.model_variant == "base"
                ),
                preview_branch_force_dense=True,
                preview_branch_use_lora=(
                    fast_preview
                    and (
                        spec.model_variant == "lora"
                        or (
                            spec.execution_mode == "checkpoint"
                            and spec.checkpoint_preview
                        )
                    )
                ),
                preview_audio_branch_use_lora=False,
                preview_audio_branch_steps=4,
                preview_audio_branch_spatial_scale=0.65,
                preview_ready_callback=preview_ready_callback,
                preview_decision_wait=(
                    preview_decision_wait if spec.preview_mode == "pause" else None
                ),
                multiscale_initial_width=selflift_initial_geometry[0],
                multiscale_initial_height=selflift_initial_geometry[1],
                multiscale_resize_after_step=(
                    int(spec.selflift_transition_step or 1) - 1
                    if spec.selflift_enabled
                    else None
                ),
                multiscale_transition_mode=(
                    "selflift_learned_x0"
                    if spec.selflift_enabled
                    else "noisy_highpass"
                ),
                terminal_refinement_initial_width=(
                    candidate.initial_width if candidate is not None else None
                ),
                terminal_refinement_initial_height=(
                    candidate.initial_height if candidate is not None else None
                ),
                terminal_refinement_steps=(
                    candidate.refinement_steps if candidate is not None else 0
                ),
                terminal_refinement_denoise=(
                    candidate.refinement_denoise if candidate is not None else 0.0125
                ),
                terminal_refinement_dense_tail_steps=(
                    candidate.dense_tail_steps if candidate is not None else 1
                ),
            )
            started = time.monotonic()
            try:
                if global_co_plan is not None:
                    def run_global_co_denoise():
                        persist_conditioning = getattr(
                            built.session, "persist_conditioning_cache", None
                        )
                        if not callable(persist_conditioning):
                            raise RuntimeError(
                                "global co-denoise requires persisted window conditioning"
                            )
                        with tempfile.TemporaryDirectory(
                            prefix=".h3-global-co-denoise-",
                        ) as temporary_root:
                            temporary = Path(temporary_root)
                            conditioning_paths: list[Path] = []
                            receipts: list[dict[str, Any]] = []
                            cache_by_prompt: dict[str, Path] = {}
                            for window, prompt in zip(
                                global_co_plan.windows, global_co_prompts
                            ):
                                if cancel_event.is_set():
                                    raise HotSessionCancelled(
                                        "native H3 generation cancelled"
                                    )
                                cached_path = cache_by_prompt.get(prompt)
                                if cached_path is None:
                                    cached_path = temporary / (
                                        f"condition-{len(cache_by_prompt):02d}.pt"
                                    )
                                    condition_request = replace(
                                        request,
                                        prompt=prompt,
                                        frames=window.frames,
                                        output_path=output_path,
                                        first_frame=None,
                                        last_frame=None,
                                        reference_videos=(),
                                        refinement_latents_path=None,
                                        refinement_denoise=None,
                                        global_selflift_source_path=None,
                                        continuation_latents_path=None,
                                        continuation_context_frames=0,
                                        conditioning_cache_source_path=None,
                                        save_final_latents_path=None,
                                        latent_only=False,
                                        sampler_state_path=None,
                                        formal_resume_state_path=None,
                                        checkpoint_after_step=None,
                                        checkpoint_state_path=None,
                                        preview_step_index=None,
                                        preview_output_path=None,
                                        preview_latents_path=None,
                                        preview_ready_callback=None,
                                        preview_decision_wait=None,
                                        multiscale_initial_width=None,
                                        multiscale_initial_height=None,
                                        multiscale_resize_after_step=None,
                                        terminal_refinement_initial_width=None,
                                        terminal_refinement_initial_height=None,
                                        terminal_refinement_steps=0,
                                    )
                                    receipt = persist_conditioning(
                                        condition_request, cached_path
                                    )
                                    receipts.append(dict(receipt))
                                    cache_by_prompt[prompt] = cached_path
                                conditioning_paths.append(cached_path)

                            co_request = replace(
                                request,
                                prompt=global_co_prompts[0],
                                actual_step_indices=tuple(range(total_steps)),
                                scheduler_required_actual_step_indices=(),
                                attention_online_guard_id=None,
                                attention_online_budget_dense_layers=0.0,
                                attention_online_rebate_schedule=(),
                                first_frame=None,
                                last_frame=None,
                                reference_videos=(),
                                refinement_latents_path=None,
                                refinement_denoise=None,
                                continuation_latents_path=None,
                                continuation_context_frames=0,
                                conditioning_cache_source_path=conditioning_paths[0],
                                global_co_denoise_output_frames=(
                                    global_co_plan.output_frames
                                ),
                                global_co_denoise_window_frames=(
                                    global_co_plan.window_frames
                                ),
                                global_co_denoise_stride_frames=(
                                    global_co_plan.stride_frames
                                ),
                                global_co_denoise_prompts=global_co_prompts,
                                global_co_denoise_prompt_ranges=(
                                    tuple(global_selflift_prompt_ranges)
                                ),
                                global_co_denoise_balanced_windows=(
                                    spec.selflift_temporal_window_enabled
                                ),
                                global_co_denoise_conditioning_paths=tuple(
                                    conditioning_paths
                                ),
                                save_final_latents_path=final_latents_path,
                                latent_only=False,
                                formal_resume_state_path=None,
                                checkpoint_after_step=None,
                                checkpoint_state_path=None,
                                preview_step_index=None,
                                preview_output_path=None,
                                preview_latents_path=None,
                                preview_ready_callback=None,
                                preview_decision_wait=None,
                                multiscale_initial_width=None,
                                multiscale_initial_height=None,
                                multiscale_resize_after_step=None,
                                terminal_refinement_initial_width=None,
                                terminal_refinement_initial_height=None,
                                terminal_refinement_steps=0,
                                execution_plan=replace(
                                    request.execution_plan,
                                    vae_temporal_tile=6,
                                    vae_tile_batch_size=(
                                        8 if built.vram_profile == "8gb" else 1
                                    ),
                                    # This research path may start from a cold
                                    # service without the release-only compiled
                                    # decoder graph.  Exact eager VAE decoding
                                    # is slower only in the final phase and must
                                    # not invalidate completed global denoising.
                                    vae_transformer_block_compile=False,
                                ),
                            )
                            generated = built.session.generate(co_request)
                            return replace(
                                generated,
                                execution_profile={
                                    **generated.execution_profile,
                                    "long_horizon": {
                                        **global_co_plan.telemetry(),
                                        "single_global_solver_state": True,
                                        "scheduler_updates_per_step": 1,
                                        "intermediate_decode": False,
                                        "single_final_decode": True,
                                        "rotary_time": (
                                            co_request.global_co_denoise_rotary_mode
                                        ),
                                        "conditioning_preencoded_before_dit": True,
                                        "unique_conditioning_encodes": len(receipts),
                                    },
                                },
                            )

                    result = await asyncio.to_thread(run_global_co_denoise)
                elif long_plan is not None:
                    from .long_horizon import (
                        DEFAULT_AUDIO_BRIDGE_TICKS,
                        plan_shot_decode_groups,
                        stitch_clean_av_segment_files,
                    )
                    import torch

                    def run_long_horizon():
                        from .av_token_memory import (
                            DEFAULT_AUDIO_RECENT_GUARD_TICKS,
                            empty_av_token_memory,
                            route_audio_memory_authority,
                            route_visual_memory_authority,
                            route_visual_memory_interval,
                            token_memory_telemetry,
                            update_av_token_memory,
                        )
                        from .long_horizon import (
                            AUDIO_LATENT_HZ,
                            CONTINUATION_CANDIDATE_MAX_ATTEMPTS,
                            FPS,
                            audio_latent_frames,
                            continuation_retry_seed,
                            evaluate_video_repaint_overlap_files,
                            evaluate_visible_video_trajectory_file,
                        )

                        all_phases: dict[str, float] = {}
                        all_steps: list[float] = []
                        window_profiles: list[dict[str, Any]] = []
                        segment_paths: list[Path] = []
                        peak_allocated = 0.0
                        peak_reserved = 0.0
                        opening_conditioning: dict[str, Any] | None = None
                        generated_voice_ticks = 120
                        token_memory: dict[str, Any] | None = (
                            empty_av_token_memory(
                                audio_slots=1,
                                audio_block_ticks=generated_voice_ticks,
                            )
                            if generated_voice_bank
                            else None
                        )
                        if authored_memory is not None and memory_collection_enabled:
                            token_memory = empty_av_token_memory(
                                video_slots=max(1, authored_memory["video_frames"]),
                                audio_slots=1,
                                audio_block_ticks=max(1, authored_memory["audio_ticks_per_clip"]),
                            )
                        token_memory_path: Path | None = None
                        token_memory_records: list[dict[str, Any]] = []
                        token_memory_route_records: list[dict[str, Any]] = []
                        audio_spine_telemetry: dict[str, Any] | None = None
                        audio_spine_profile: dict[str, Any] | None = None
                        audio_spine_path: Path | None = None
                        segment_count = len(long_plan.segments)
                        audio_bridge_schedule = tuple(
                            0
                            if index == 0 or not bounded_token_memory
                            else (
                                audio_latent_frames(
                                    long_plan.segments[index].context_frames
                                )
                                if authored_preview is not None
                                else min(
                                    DEFAULT_AUDIO_BRIDGE_TICKS,
                                    audio_latent_frames(
                                        long_plan.segments[index].context_frames
                                    ),
                                )
                            )
                            for index in range(segment_count)
                        )
                        with tempfile.TemporaryDirectory(
                            prefix=".h3-long-horizon-",
                        ) as temporary_root:
                            temporary = Path(temporary_root)
                            conditioning_paths: list[Path | None] = [
                                None for _ in long_plan.segments
                            ]
                            boundary_conditioning_paths: list[Path | None] = [
                                None for _ in long_plan.segments
                            ]
                            segment_prompts: list[str] = []
                            voice_anchor_available = False
                            voice_bank_binding_records: list[dict[str, Any]] = []
                            for segment in long_plan.segments:
                                binds_voice = bool(
                                    generated_voice_bank
                                    and voice_anchor_available
                                    and segment.audio_memory_active
                                )
                                segment_prompts.append(
                                    _bind_generated_voice_reference(segment.prompt)
                                    if binds_voice
                                    else segment.prompt
                                )
                                voice_bank_binding_records.append({
                                    "segment_index": segment.index,
                                    "bound": binds_voice,
                                })
                                visible_audio_ticks = (
                                    audio_latent_frames(segment.window_frames)
                                    - (
                                        audio_latent_frames(segment.context_frames)
                                        if segment.context_frames
                                        else 0
                                    )
                                )
                                if (
                                    generated_voice_bank
                                    and not voice_anchor_available
                                    and segment.authorized_dialogue_count is not None
                                    and segment.authorized_dialogue_count > 0
                                    and visible_audio_ticks
                                    - DEFAULT_AUDIO_RECENT_GUARD_TICKS
                                    >= generated_voice_ticks
                                    and any(
                                        0
                                        <= round(
                                            (
                                                frame
                                                - segment.visible_start_frame
                                            )
                                            / FPS
                                            * AUDIO_LATENT_HZ
                                        )
                                        < visible_audio_ticks
                                        - DEFAULT_AUDIO_RECENT_GUARD_TICKS
                                        for frame in segment.authorized_dialogue_frames
                                    )
                                ):
                                    voice_anchor_available = True
                            conditioning_receipts: list[dict[str, Any]] = []
                            persist_conditioning = getattr(
                                built.session, "persist_conditioning_cache", None
                            )
                            fingerprint_conditioning = getattr(
                                built.session, "_conditioning_fingerprint", None
                            )
                            cache_by_fingerprint: dict[str, Path] = {}
                            boundary_conditioning_encode_count = 0
                            boundary_segment_indices = [
                                segment.index
                                for segment in long_plan.segments
                                if segment.transition == "continue"
                                and segment.continuation_bridge_prompt is not None
                            ]
                            if boundary_segment_indices and not (
                                callable(persist_conditioning)
                                and callable(fingerprint_conditioning)
                            ):
                                raise RuntimeError(
                                    "continuation boundary conditioning requires "
                                    "persisted Qwen conditioning support"
                                )
                            if callable(persist_conditioning) and callable(
                                fingerprint_conditioning
                            ):
                                for index, segment in enumerate(long_plan.segments):
                                    if cancel_event.is_set():
                                        raise HotSessionCancelled(
                                            "native H3 generation cancelled"
                                        )
                                    condition_request = replace(
                                        request,
                                        prompt=segment_prompts[index],
                                        seed=segment.seed,
                                        frames=segment.window_frames,
                                        # The hot session enforces that public
                                        # RGB outputs remain below output_root.
                                        # Conditioning-only requests never write
                                        # RGB, so retain the authorized final
                                        # path while their tensor cache stays in
                                        # the Linux-native temporary directory.
                                        output_path=output_path,
                                        first_frame=(
                                            request.first_frame if index == 0 else None
                                        ),
                                        last_frame=(
                                            request.last_frame
                                            if index == segment_count - 1
                                            else None
                                        ),
                                        reference_audios=(
                                            request.reference_audios
                                            if segment.reference_audio_active
                                            else ()
                                        ),
                                        prepared_reference_audios=(),
                                        refinement_latents_path=None,
                                        refinement_denoise=None,
                                        continuation_latents_path=None,
                                        continuation_context_frames=0,
                                        continuation_audio_bridge_ticks=0,
                                        av_token_memory_path=None,
                                        conditioning_cache_source_path=None,
                                        save_final_latents_path=None,
                                        latent_only=False,
                                        sampler_state_path=None,
                                        formal_resume_state_path=None,
                                        checkpoint_after_step=None,
                                        checkpoint_state_path=None,
                                        preview_step_index=None,
                                        preview_output_path=None,
                                        preview_latents_path=None,
                                        preview_ready_callback=None,
                                        preview_decision_wait=None,
                                        multiscale_initial_width=None,
                                        multiscale_initial_height=None,
                                        multiscale_resize_after_step=None,
                                        terminal_refinement_initial_width=None,
                                        terminal_refinement_initial_height=None,
                                        terminal_refinement_steps=0,
                                    )
                                    fingerprint = fingerprint_conditioning(
                                        condition_request
                                    )
                                    cached_path = cache_by_fingerprint.get(fingerprint)
                                    if cached_path is None:
                                        cached_path = temporary / (
                                            f"condition-{len(cache_by_fingerprint):02d}.pt"
                                        )
                                        if progress_callback is not None:
                                            progress_callback({
                                                "percent": 0.5 + 1.5 * (
                                                    index / max(1, segment_count)
                                                ),
                                                "stage": "long_horizon_conditioning",
                                                "detail": (
                                                    f"预编码长视频语义 {index + 1}/{segment_count}"
                                                ),
                                            })
                                        receipt = persist_conditioning(
                                            condition_request, cached_path
                                        )
                                        conditioning_receipts.append(dict(receipt))
                                        cache_by_fingerprint[fingerprint] = cached_path
                                    conditioning_paths[index] = cached_path
                                    if segment.continuation_bridge_prompt is not None:
                                        boundary_request = replace(
                                            condition_request,
                                            prompt=segment.continuation_bridge_prompt,
                                            # A boundary condition preserves the
                                            # exact carried AV state. It must not
                                            # introduce a public voice reference or
                                            # a future endpoint into its text rows.
                                            reference_audios=(),
                                            prepared_reference_audios=(),
                                            last_frame=None,
                                        )
                                        boundary_fingerprint = (
                                            fingerprint_conditioning(
                                                boundary_request
                                            )
                                        )
                                        boundary_path = cache_by_fingerprint.get(
                                            boundary_fingerprint
                                        )
                                        if boundary_path is None:
                                            boundary_path = temporary / (
                                                "boundary-condition-"
                                                f"{len(cache_by_fingerprint):02d}.pt"
                                            )
                                            if progress_callback is not None:
                                                progress_callback({
                                                    "percent": 0.5 + 1.5 * (
                                                        (index + 0.5)
                                                        / max(1, segment_count)
                                                    ),
                                                    "stage": "long_horizon_conditioning",
                                                    "detail": (
                                                        "预编码续写边界语义 "
                                                        f"{index + 1}/{segment_count}"
                                                    ),
                                                })
                                            receipt = persist_conditioning(
                                                boundary_request,
                                                boundary_path,
                                            )
                                            boundary_conditioning_encode_count += 1
                                            conditioning_receipts.append(
                                                dict(receipt)
                                            )
                                            cache_by_fingerprint[
                                                boundary_fingerprint
                                            ] = boundary_path
                                        boundary_conditioning_paths[index] = (
                                            boundary_path
                                        )
                            previous_path: Path | None = None
                            for index, segment in enumerate(long_plan.segments):
                                if cancel_event.is_set():
                                    raise HotSessionCancelled(
                                        "native H3 generation cancelled"
                                    )
                                piece_output = temporary / f"segment-{index:02d}-clean.pt"

                                def piece_progress(event, *, _index=index):
                                    if progress_callback is None:
                                        return
                                    local = float(event.get("percent", 0.0)) / 100.0
                                    progress_callback({
                                        "percent": 2.0 + 82.0 * ((_index + local) / segment_count),
                                        "stage": "long_horizon_segment",
                                        "detail": (
                                            f"长视频窗口 {_index + 1}/{segment_count} · "
                                            f"{event.get('detail', event.get('stage', '执行中'))}"
                                        ),
                                    })

                                active_token_memory_path = token_memory_path
                                if memory_collection_enabled and index > 0 and (
                                    segment.visual_memory_floor_frame is not None
                                    or latest_visual_only
                                    or not long_visual_memory
                                    or not segment.audio_memory_active
                                    or not inferred_audio_token_memory
                                ):
                                    if token_memory is None:
                                        raise RuntimeError(
                                            "director memory routing requires initialized token memory"
                                        )
                                    routed_memory = token_memory
                                    visual_route_applied = False
                                    if not long_visual_memory:
                                        routed_memory = route_visual_memory_authority(
                                            routed_memory, active=False,
                                        )
                                        visual_route_applied = True
                                    elif latest_visual_only:
                                        routed_memory = route_visual_memory_authority(
                                            routed_memory, active=True, latest_only=True,
                                        )
                                        if (
                                            len(routed_memory["video_entries"]) != 1
                                            or int(routed_memory["video_entries"][0]["position"])
                                            != segment.visible_start_frame - 1
                                        ):
                                            raise RuntimeError("latest-only reference must be the immediately preceding generated state")
                                        visual_route_applied = True
                                    elif segment.visual_memory_floor_frame is not None:
                                        candidate_memory = route_visual_memory_interval(
                                            routed_memory,
                                            minimum_position=(
                                                segment.visual_memory_floor_frame
                                            ),
                                            maximum_position=(
                                                segment.visible_start_frame
                                            ),
                                            include_canonical=(
                                                segment.visual_memory_include_canonical
                                            ),
                                            progressive_layout_state=(
                                                segment.visual_memory_progressive_layout_state
                                            ),
                                            layout_minimum_position=(
                                                segment.visual_memory_layout_anchor_start_frame
                                            ),
                                            layout_maximum_position=(
                                                segment.visual_memory_layout_anchor_stop_frame
                                            ),
                                            novel_camera_cut=(
                                                segment.novel_camera_cut
                                            ),
                                            novel_camera_layout_probe=(
                                                segment.novel_camera_layout_probe
                                            ),
                                        )
                                        # The transition window normally supplies
                                        # at least one active-shot anchor. Retain the
                                        # global visual route as a safe compatibility
                                        # fallback if an unusual coreset has none.
                                        if (
                                            candidate_memory["video_entries"]
                                            or segment.novel_camera_cut
                                        ):
                                            routed_memory = candidate_memory
                                            visual_route_applied = True
                                    if (
                                        not segment.audio_memory_active
                                        or not inferred_audio_token_memory
                                    ):
                                        routed_memory = route_audio_memory_authority(
                                            routed_memory,
                                            active=False,
                                        )
                                    # The transition window normally supplies
                                    # at least one active-shot anchor.  Audio-only
                                    # routing is still persisted when visual routing
                                    # falls back to the complete coreset.
                                    if (
                                        visual_route_applied
                                        or not segment.audio_memory_active
                                        or not inferred_audio_token_memory
                                    ):
                                        active_token_memory_path = temporary / (
                                            f"token-memory-route-{index:02d}.pt"
                                        )
                                        torch.save(
                                            routed_memory,
                                            active_token_memory_path,
                                        )
                                        token_memory_route_records.append(
                                            token_memory_telemetry(routed_memory)
                                        )

                                piece_request = replace(
                                    request,
                                    prompt=segment_prompts[index],
                                    seed=segment.seed,
                                    frames=segment.window_frames,
                                    # Latent-only windows do not emit an MP4;
                                    # using the authorized final path preserves
                                    # the output-root security invariant without
                                    # moving latent staging back to /mnt/c.
                                    output_path=output_path,
                                    first_frame=(request.first_frame if index == 0 else None),
                                    last_frame=(
                                        request.last_frame
                                        if index == segment_count - 1
                                        else None
                                    ),
                                    reference_audios=(
                                        request.reference_audios
                                        if segment.reference_audio_active
                                        else ()
                                    ),
                                    prepared_reference_audios=(),
                                    progress_callback=piece_progress,
                                    refinement_latents_path=None,
                                    refinement_denoise=None,
                                    refinement_spatial_mode="strict",
                                    continuation_latents_path=previous_path,
                                    continuation_context_frames=segment.context_frames,
                                    continuation_video_prefix_frames=(
                                        segment.video_prefix_frames
                                    ),
                                    continuation_video_prefix_from_source_end=(
                                        segment.terminal_video_seed
                                    ),
                                    continuation_audio_bridge_ticks=(
                                        audio_bridge_schedule[index]
                                    ),
                                    av_token_memory_path=(
                                        active_token_memory_path
                                        if memory_collection_enabled and index > 0
                                        else None
                                    ),
                                    conditioning_cache_source_path=(
                                        conditioning_paths[index]
                                    ),
                                    continuation_text_bridge_conditioning_path=(
                                        boundary_conditioning_paths[index]
                                        if segment.transition == "continue"
                                        else None
                                    ),
                                    save_final_latents_path=piece_output,
                                    internal_video_tokens=None,
                                    internal_audio_tokens=None,
                                    latent_only=True,
                                    retain_transformer_after_latent_only=(
                                        global_audio_spine
                                        or (
                                            index < segment_count - 1
                                            and not (
                                                index + 1 == segment_count - 1
                                                and request.last_frame is not None
                                            )
                                            and not (
                                                bool(request.reference_videos)
                                                and long_plan.segments[
                                                    index + 1
                                                ].window_frames
                                                != segment.window_frames
                                            )
                                            and not (
                                                bool(
                                                    request.reference_images
                                                    or request.reference_videos
                                                    or request.reference_audios
                                                )
                                                and not request.cache_reference_latents
                                            )
                                        )
                                    ),
                                    sampler_state_path=None,
                                    formal_resume_state_path=None,
                                    checkpoint_after_step=None,
                                    checkpoint_state_path=None,
                                    preview_step_index=None,
                                    preview_output_path=None,
                                    preview_latents_path=None,
                                    preview_ready_callback=None,
                                    preview_decision_wait=None,
                                    multiscale_initial_width=None,
                                    multiscale_initial_height=None,
                                    multiscale_resize_after_step=None,
                                    terminal_refinement_initial_width=None,
                                    terminal_refinement_initial_height=None,
                                    terminal_refinement_steps=0,
                                )
                                repaint_frames = (
                                    segment.context_frames
                                    - int(
                                        segment.video_prefix_frames
                                        if segment.video_prefix_frames is not None
                                        else segment.context_frames
                                    )
                                    if segment.transition == "continue"
                                    else 0
                                )
                                gate_active = bool(
                                    previous_path is not None
                                    and segment.transition == "continue"
                                    and repaint_frames > 0
                                )
                                maximum_attempts = (
                                    CONTINUATION_CANDIDATE_MAX_ATTEMPTS
                                    if gate_active else 1
                                )
                                candidates = []
                                for attempt in range(maximum_attempts):
                                    candidate_output = temporary / (
                                        f"segment-{index:02d}-clean-attempt-"
                                        f"{attempt:02d}.pt"
                                    )

                                    def retry_progress(
                                        event,
                                        *,
                                        _index=index,
                                        _attempt=attempt,
                                    ):
                                        if _attempt == 0:
                                            piece_progress(event)
                                            return
                                        if progress_callback is None:
                                            return
                                        progress_callback({
                                            "percent": 2.0 + 82.0 * (
                                                (_index + 1) / segment_count
                                            ),
                                            "stage": "long_horizon_continuity_retry",
                                            "detail": (
                                                f"窗口 {_index + 1}/{segment_count} "
                                                f"连续性重采样 {_attempt + 1}/"
                                                f"{maximum_attempts}"
                                            ),
                                        })

                                    candidate_request = replace(
                                        piece_request,
                                        seed=continuation_retry_seed(
                                            segment.seed, attempt
                                        ),
                                        progress_callback=retry_progress,
                                        save_final_latents_path=candidate_output,
                                    )
                                    candidate_result = built.session.generate(
                                        candidate_request
                                    )
                                    agreement = (
                                        evaluate_video_repaint_overlap_files(
                                            previous_path,
                                            candidate_output,
                                            context_frames=segment.context_frames,
                                            repaint_frames=repaint_frames,
                                        )
                                        if gate_active else
                                        {
                                            "policy": "not_applicable",
                                            "accepted": True,
                                        }
                                    )
                                    trajectory = (
                                        evaluate_visible_video_trajectory_file(
                                            candidate_output,
                                            context_frames=segment.context_frames,
                                        )
                                        if gate_active else
                                        {
                                            "policy": "not_applicable",
                                            "accepted": True,
                                        }
                                    )
                                    candidate_accepted = bool(
                                        agreement["accepted"]
                                        and trajectory["accepted"]
                                    )
                                    candidate = {
                                        "attempt": attempt,
                                        "seed": int(candidate_request.seed),
                                        "path": candidate_output,
                                        "result": candidate_result,
                                        "accepted": candidate_accepted,
                                        "agreement": agreement,
                                        "trajectory": trajectory,
                                    }
                                    candidates.append(candidate)
                                    all_steps.extend(
                                        candidate_result.step_seconds
                                    )
                                    peak_allocated = max(
                                        peak_allocated,
                                        candidate_result.peak_allocated_gib,
                                    )
                                    peak_reserved = max(
                                        peak_reserved,
                                        candidate_result.peak_reserved_gib,
                                    )
                                    if candidate_accepted:
                                        break

                                accepted_candidate = next(
                                    (
                                        candidate
                                        for candidate in candidates
                                        if candidate["accepted"]
                                    ),
                                    None,
                                )
                                selected_candidate = (
                                    accepted_candidate
                                    if accepted_candidate is not None else
                                    min(
                                        candidates,
                                        key=lambda candidate: (
                                            0
                                            if candidate["agreement"]["accepted"]
                                            else 1,
                                            float(candidate["trajectory"].get(
                                                "maximum_joint_risk",
                                                float("inf"),
                                            )),
                                            float(candidate["agreement"].get(
                                                "maximum_low_frequency_relative_rms",
                                                float("inf"),
                                            )),
                                        ),
                                    )
                                )
                                piece_output = selected_candidate["path"]
                                piece_result = selected_candidate["result"]
                                for candidate in candidates:
                                    phase_prefix = (
                                        f"segment_{index + 1:02d}"
                                        if candidate is selected_candidate else
                                        f"segment_{index + 1:02d}.rejected_attempt_"
                                        f"{candidate['attempt'] + 1:02d}"
                                    )
                                    for name, seconds in candidate[
                                        "result"
                                    ].phases.items():
                                        all_phases[
                                            f"{phase_prefix}.{name}"
                                        ] = seconds
                                if gate_active:
                                    piece_result.execution_profile[
                                        "continuation_candidate_gate"
                                    ] = {
                                        "policy": (
                                            "same_time_agreement_and_visible_"
                                            "trajectory_retry_v2"
                                        ),
                                        "active": True,
                                        "accepted": bool(
                                            accepted_candidate is not None
                                        ),
                                        "attempt_count": len(candidates),
                                        "maximum_attempts": maximum_attempts,
                                        "selected_attempt": int(
                                            selected_candidate["attempt"]
                                        ),
                                        "selected_seed": int(
                                            selected_candidate["seed"]
                                        ),
                                        "attempts": [
                                            {
                                                "attempt": int(
                                                    candidate["attempt"]
                                                ),
                                                "seed": int(candidate["seed"]),
                                                "accepted": bool(
                                                    candidate["accepted"]
                                                ),
                                                "overlap_agreement": (
                                                    candidate["agreement"]
                                                ),
                                                "visible_trajectory": (
                                                    candidate["trajectory"]
                                                ),
                                            }
                                            for candidate in candidates
                                        ],
                                    }
                                if index == 0 and segment.prompt == spec.prompt:
                                    payload = getattr(
                                        built.session,
                                        "_last_conditioning_cache_payload",
                                        None,
                                    )
                                    if isinstance(payload, dict):
                                        opening_conditioning = payload
                                segment_paths.append(piece_output)
                                if memory_collection_enabled:
                                    clean_document = torch.load(
                                        piece_output,
                                        map_location="cpu",
                                        weights_only=True,
                                    )
                                    # A single take needs the newest state because
                                    # its scene evolves causally.  In a multi-shot
                                    # timeline, forcing that state beside canonical
                                    # and diverse views can make H3 interpret two
                                    # views of one actor as two actor instances at
                                    # the next cut.  Keep the established coreset
                                    # policy for cuts and reserve newest-state
                                    # routing only for continuous takes.
                                    token_memory = update_av_token_memory(
                                        token_memory,
                                        clean_document,
                                        context_frames=segment.context_frames,
                                        visible_start_frame=segment.visible_start_frame,
                                        visible_frames=segment.visible_frames,
                                        preserve_latest_visual=(
                                            segment.preserve_latest_visual
                                            or latest_visual_only
                                        ),
                                        collect_audio=(
                                            inferred_audio_token_memory
                                            and structured_director
                                            and segment.authorized_dialogue_count is not None
                                            and segment.authorized_dialogue_count > 0
                                            and not request.reference_audios
                                            and (
                                                not token_memory["audio_entries"]
                                                if authored_memory is not None
                                                else (
                                                    not generated_voice_bank
                                                    or not token_memory["audio_entries"]
                                                )
                                            )
                                        ),
                                        audio_focus_frames=(
                                            segment.authorized_dialogue_frames
                                        ),
                                        leading_preroll_frames=(
                                            segment.opening_preroll_frames
                                        ),
                                    )
                                    token_memory_path = temporary / (
                                        f"token-memory-{index:02d}.pt"
                                    )
                                    torch.save(token_memory, token_memory_path)
                                    token_memory_records.append(
                                        token_memory_telemetry(token_memory)
                                    )
                                    del clean_document
                                previous_path = piece_output
                                window_profiles.append(piece_result.execution_profile)

                            continuation_handoff_records: list[dict[str, Any]] = []
                            for segment, profile in zip(
                                long_plan.segments, window_profiles
                            ):
                                bridge_profile = profile.get(
                                    "continuation_text_bridge"
                                )
                                if not (
                                    isinstance(bridge_profile, dict)
                                    and bridge_profile.get("active") is True
                                ):
                                    continue
                                continuation_handoff_records.append({
                                    "window_index": segment.index,
                                    "boundary_auxiliary_steps": int(
                                        bridge_profile[
                                            "boundary_auxiliary_steps"
                                        ]
                                    ),
                                    "current_prompt_steps": int(
                                        bridge_profile["current_prompt_steps"]
                                    ),
                                    "blend_peak": float(
                                        bridge_profile["blend_peak"]
                                    ),
                                    "blend_band_policy": str(
                                        bridge_profile["blend_band_policy"]
                                    ),
                                    "blend_band_video_latent_tokens": int(
                                        bridge_profile[
                                            "blend_band_video_latent_tokens"
                                        ]
                                    ),
                                    "hidden_repaint_video_latent_tokens": int(
                                        bridge_profile[
                                            "hidden_repaint_video_latent_tokens"
                                        ]
                                    ),
                                    "visible_fade_video_latent_tokens": int(
                                        bridge_profile[
                                            "visible_fade_video_latent_tokens"
                                        ]
                                    ),
                                    "extra_dit_calls": int(
                                        bridge_profile["extra_dit_calls"]
                                    ),
                                })
                            active_boundary_indices = [
                                record["window_index"]
                                for record in continuation_handoff_records
                            ]
                            if active_boundary_indices != boundary_segment_indices:
                                raise RuntimeError(
                                    "continuation boundary conditioning did not execute "
                                    "for every compiled continuation window"
                                )
                            continuation_semantic_handoff = {
                                "policy": (
                                    "protected_repaint_plateau_visible_fade_v5"
                                    if continuation_handoff_records
                                    else "retired_after_temporal_mask_edge_cuts_v14"
                                ),
                                "active": bool(continuation_handoff_records),
                                "active_window_indices": active_boundary_indices,
                                "active_window_count": len(
                                    continuation_handoff_records
                                ),
                                "window_records": continuation_handoff_records,
                                "extra_qwen_conditioning_encodes": (
                                    boundary_conditioning_encode_count
                                ),
                                "extra_dit_calls": sum(
                                    record["extra_dit_calls"]
                                    for record in continuation_handoff_records
                                ),
                            }
                            continuation_video_records: list[dict[str, Any]] = []
                            continuation_candidate_gate_records: list[
                                dict[str, Any]
                            ] = []
                            for segment, profile in zip(
                                long_plan.segments, window_profiles
                            ):
                                if segment.transition != "continue":
                                    continue
                                transport = profile.get(
                                    "long_horizon_continuation"
                                )
                                if not isinstance(transport, dict):
                                    raise RuntimeError(
                                        "continuation window is missing its latent "
                                        "transport receipt"
                                    )
                                continuation_video_records.append({
                                    "window_index": segment.index,
                                    "context_frames": int(
                                        transport["context_frames"]
                                    ),
                                    "protected_video_prefix_frames": int(
                                        transport["video_prefix_frames"]
                                    ),
                                    "hidden_video_repaint_frames": int(
                                        transport["video_hidden_repaint_frames"]
                                    ),
                                    "context_video_latent_tokens": int(
                                        transport["video_context_tokens"]
                                    ),
                                    "protected_video_latent_tokens": int(
                                        transport["video_protected_tokens"]
                                    ),
                                    "hidden_video_repaint_latent_tokens": int(
                                        transport["video_hidden_repaint_tokens"]
                                    ),
                                })
                                if int(
                                    transport["video_hidden_repaint_frames"]
                                ) > 0:
                                    candidate_gate = profile.get(
                                        "continuation_candidate_gate"
                                    )
                                    if not isinstance(candidate_gate, dict):
                                        raise RuntimeError(
                                            "repaint continuation is missing its "
                                            "continuity candidate gate receipt"
                                        )
                                    continuation_candidate_gate_records.append({
                                        "window_index": segment.index,
                                        **candidate_gate,
                                    })
                            continuation_video_handoff = {
                                "policy": (
                                    "agreement_and_trajectory_gated_same_time_"
                                    "repaint_v2"
                                ),
                                "active": any(
                                    record["hidden_video_repaint_frames"] > 0
                                    for record in continuation_video_records
                                ),
                                "active_window_indices": [
                                    record["window_index"]
                                    for record in continuation_video_records
                                    if record["hidden_video_repaint_frames"] > 0
                                ],
                                "window_records": continuation_video_records,
                                "repaint_replaces_same_time_predecessor_tail": True,
                                "candidate_quality_gate": {
                                    "policy": (
                                        "same_time_agreement_and_visible_"
                                        "trajectory_retry_v2"
                                    ),
                                    "active": bool(
                                        continuation_candidate_gate_records
                                    ),
                                    "accepted_all": all(
                                        bool(record["accepted"])
                                        for record in continuation_candidate_gate_records
                                    ),
                                    "window_records": (
                                        continuation_candidate_gate_records
                                    ),
                                    "total_attempts": sum(
                                        int(record["attempt_count"])
                                        for record in continuation_candidate_gate_records
                                    ),
                                    "total_retries": sum(
                                        int(record["attempt_count"]) - 1
                                        for record in continuation_candidate_gate_records
                                    ),
                                },
                                "extra_qwen_conditioning_encodes": 0,
                                "extra_dit_calls": sum(
                                    max(0, int(record["attempt_count"]) - 1)
                                    * request.steps
                                    for record in continuation_candidate_gate_records
                                ),
                            }

                            if global_audio_spine:
                                from .audio_spine import plan_global_audio_spine
                                from .long_horizon import localize_h3_prompt

                                if not (
                                    callable(persist_conditioning)
                                    and callable(fingerprint_conditioning)
                                ):
                                    raise RuntimeError(
                                        "windowed global audio spine requires "
                                        "persisted window conditioning"
                                    )

                                spine_plan = plan_global_audio_spine(
                                    output_width=request.width,
                                    output_height=request.height,
                                    output_frames=long_plan.output_frames,
                                )
                                audio_spine_path = temporary / "global-audio-spine.pt"
                                spine_global_plan = spine_plan.temporal_plan
                                spine_prompts = tuple(
                                    localize_h3_prompt(
                                        spec.prompt,
                                        context_start_frame=window.start_frame,
                                        visible_start_frame=window.start_frame,
                                        visible_stop_frame=window.stop_frame,
                                        segment_index=window.index,
                                        timeline_stop_frame=(
                                            spine_global_plan.output_frames
                                        ),
                                    )
                                    for window in spine_global_plan.windows
                                )
                                spine_conditioning_paths: list[Path] = []
                                for window, prompt in zip(
                                    spine_global_plan.windows,
                                    spine_prompts,
                                ):
                                    condition_request = replace(
                                        request,
                                        prompt=prompt,
                                        seed=spec.seed,
                                        width=spine_plan.width,
                                        height=spine_plan.height,
                                        frames=window.frames,
                                        output_path=output_path,
                                        first_frame=None,
                                        last_frame=None,
                                        reference_images=(),
                                        reference_videos=(),
                                        reference_audios=(),
                                        refinement_latents_path=None,
                                        refinement_denoise=None,
                                        continuation_latents_path=None,
                                        continuation_context_frames=0,
                                        continuation_audio_bridge_ticks=0,
                                        av_token_memory_path=None,
                                        conditioning_cache_source_path=None,
                                        save_final_latents_path=None,
                                        global_co_denoise_output_frames=None,
                                        global_co_denoise_prompts=(),
                                        global_co_denoise_conditioning_paths=(),
                                        internal_video_tokens=None,
                                        internal_audio_tokens=None,
                                        latent_only=False,
                                        retain_transformer_after_latent_only=False,
                                        sampler_state_path=None,
                                        formal_resume_state_path=None,
                                        checkpoint_after_step=None,
                                        checkpoint_state_path=None,
                                        preview_step_index=None,
                                        preview_output_path=None,
                                        preview_latents_path=None,
                                        preview_ready_callback=None,
                                        preview_decision_wait=None,
                                        multiscale_initial_width=None,
                                        multiscale_initial_height=None,
                                        multiscale_resize_after_step=None,
                                        terminal_refinement_initial_width=None,
                                        terminal_refinement_initial_height=None,
                                        terminal_refinement_steps=0,
                                    )
                                    fingerprint = fingerprint_conditioning(
                                        condition_request
                                    )
                                    cached_path = cache_by_fingerprint.get(
                                        fingerprint
                                    )
                                    if cached_path is None:
                                        cached_path = temporary / (
                                            "audio-spine-condition-"
                                            f"{len(cache_by_fingerprint):02d}.pt"
                                        )
                                        receipt = persist_conditioning(
                                            condition_request,
                                            cached_path,
                                        )
                                        conditioning_receipts.append(dict(receipt))
                                        cache_by_fingerprint[fingerprint] = cached_path
                                    spine_conditioning_paths.append(cached_path)

                                def audio_spine_progress(event):
                                    if progress_callback is None:
                                        return
                                    local = float(event.get("percent", 0.0)) / 100.0
                                    progress_callback({
                                        "percent": 84.0 + 6.0 * local,
                                        "stage": "long_horizon_audio_spine",
                                        "detail": (
                                            "生成无接缝全局音频轨 · "
                                            f"{event.get('detail', event.get('stage', '执行中'))}"
                                        ),
                                    })

                                spine_request = replace(
                                    request,
                                    prompt=spine_prompts[0],
                                    seed=spec.seed,
                                    width=spine_plan.width,
                                    height=spine_plan.height,
                                    # The planner/memory contract describes
                                    # the largest physical DiT view.  The
                                    # sampler itself owns the complete global
                                    # latent clock below.
                                    frames=spine_plan.maximum_local_frames,
                                    output_path=output_path,
                                    first_frame=None,
                                    last_frame=None,
                                    reference_images=(),
                                    reference_videos=(),
                                    reference_audios=(),
                                    progress_callback=audio_spine_progress,
                                    refinement_latents_path=None,
                                    refinement_denoise=None,
                                    refinement_spatial_mode="strict",
                                    continuation_latents_path=None,
                                    continuation_context_frames=0,
                                    continuation_audio_bridge_ticks=0,
                                    av_token_memory_path=None,
                                    conditioning_cache_source_path=(
                                        spine_conditioning_paths[0]
                                    ),
                                    global_co_denoise_output_frames=(
                                        spine_global_plan.output_frames
                                    ),
                                    global_co_denoise_window_frames=(
                                        spine_global_plan.window_frames
                                    ),
                                    global_co_denoise_stride_frames=(
                                        spine_global_plan.stride_frames
                                    ),
                                    global_co_denoise_prompts=spine_prompts,
                                    global_co_denoise_conditioning_paths=tuple(
                                        spine_conditioning_paths
                                    ),
                                    global_co_denoise_rotary_mode="window_local",
                                    save_final_latents_path=audio_spine_path,
                                    internal_video_tokens=None,
                                    internal_audio_tokens=None,
                                    latent_only=True,
                                    retain_transformer_after_latent_only=False,
                                    sampler_state_path=None,
                                    formal_resume_state_path=None,
                                    checkpoint_after_step=None,
                                    checkpoint_state_path=None,
                                    preview_step_index=None,
                                    preview_output_path=None,
                                    preview_latents_path=None,
                                    preview_ready_callback=None,
                                    preview_decision_wait=None,
                                    multiscale_initial_width=None,
                                    multiscale_initial_height=None,
                                    multiscale_resize_after_step=None,
                                    terminal_refinement_initial_width=None,
                                    terminal_refinement_initial_height=None,
                                    terminal_refinement_steps=0,
                                )
                                spine_result = built.session.generate(spine_request)
                                for name, seconds in spine_result.phases.items():
                                    all_phases[f"audio_spine.{name}"] = seconds
                                all_steps.extend(spine_result.step_seconds)
                                audio_spine_profile = spine_result.execution_profile
                                peak_allocated = max(
                                    peak_allocated,
                                    spine_result.peak_allocated_gib,
                                )
                                peak_reserved = max(
                                    peak_reserved,
                                    spine_result.peak_reserved_gib,
                                )
                                audio_spine_telemetry = spine_plan.telemetry()

                            (
                                stitched_video,
                                stitched_audio,
                                stitched_frames,
                                stitched_engine,
                            ) = stitch_clean_av_segment_files(
                                segment_paths,
                                (segment.context_frames for segment in long_plan.segments),
                                expected_frames=long_plan.output_frames,
                                audio_bridge_ticks=audio_bridge_schedule,
                                video_repaint_frames=(
                                    (
                                        segment.context_frames
                                        - (
                                            segment.context_frames
                                            if segment.video_prefix_frames is None
                                            else int(segment.video_prefix_frames)
                                        )
                                    )
                                    if segment.transition == "continue"
                                    else 0
                                    for segment in long_plan.segments
                                ),
                                leading_preroll_frames=(
                                    long_plan.segments[0].opening_preroll_frames
                                ),
                            )
                            if audio_spine_path is not None:
                                spine_document = torch.load(
                                    audio_spine_path,
                                    map_location="cpu",
                                    weights_only=True,
                                )
                                spine_audio = spine_document.get("audio")
                                if (
                                    not isinstance(spine_audio, torch.Tensor)
                                    or spine_audio.ndim != 4
                                    or int(spine_audio.shape[-1])
                                    != int(stitched_audio.shape[-1])
                                ):
                                    raise RuntimeError(
                                        "global audio spine missed the output audio clock"
                                    )
                                del stitched_audio
                                stitched_audio = spine_audio.contiguous()
                                del spine_document, spine_audio
                            if stitched_frames != long_plan.output_frames:
                                raise RuntimeError(
                                    "long-horizon assembly missed the requested output clock"
                                )
                            stitched_path = (
                                Path(final_latents_path)
                                if final_latents_path is not None
                                else temporary / "stitched-final.pt"
                            )
                            stitched_path.parent.mkdir(parents=True, exist_ok=True)
                            stitched_document = {
                                "video": stitched_video,
                                "audio": stitched_audio,
                                "frames": stitched_frames,
                                "fps": request.fps,
                                # A checkpoint-only SelfLift continuation is
                                # still on the source grid.  Record the tensor's
                                # real geometry so the next window does not
                                # mistake this clean 540p track for a 1080p
                                # target-grid prefix.
                                "width": int(stitched_video.shape[-1]) * 16,
                                "height": int(stitched_video.shape[-2]) * 16,
                                "engine": stitched_engine,
                                "seed": spec.seed,
                                "long_horizon": long_plan.telemetry(),
                            }
                            if opening_conditioning is not None:
                                stitched_document["qwen_conditioning_cache"] = (
                                    opening_conditioning
                                )
                            torch.save(stitched_document, stitched_path)
                            del stitched_video, stitched_audio

                            # A temporal VAE must never straddle an authored
                            # hard cut.  Materialize one latent domain per shot,
                            # retaining the cut window's hidden writable visual
                            # preroll.  The decoder crops that preroll in RGB and
                            # copies all shots into one tensor before the sole
                            # final encode.  Audio continues to use the globally
                            # stitched trajectory above.
                            shot_video_checkpoints: tuple[Path, ...] = ()
                            shot_groups = plan_shot_decode_groups(
                                long_plan.segments
                            )
                            if len(shot_groups) > 1:
                                decodable_cut_prerolls = all(
                                    cut_segment.video_prefix_frames == 0
                                    or bool(cut_segment.terminal_video_seed)
                                    for cut_segment in (
                                        long_plan.segments[group.segment_indices[0]]
                                        for group in shot_groups[1:]
                                    )
                                )
                                if (
                                    long_plan.mechanism.endswith("_v2")
                                    and not decodable_cut_prerolls
                                ):
                                    raise RuntimeError(
                                        "v2 camera changes require a writable preroll or "
                                        "one cropped terminal seed"
                                    )
                                if decodable_cut_prerolls:
                                    staged_shots: list[Path] = []
                                    for group in shot_groups:
                                        group_segments = tuple(
                                            long_plan.segments[index]
                                            for index in group.segment_indices
                                        )
                                        group_paths = tuple(
                                            segment_paths[index]
                                            for index in group.segment_indices
                                        )
                                        group_contexts = (
                                            0,
                                            *(
                                                segment.context_frames
                                                for segment in group_segments[1:]
                                            ),
                                        )
                                        group_video_repaints = (
                                            0,
                                            *(
                                                (
                                                    segment.context_frames
                                                    - (
                                                        segment.context_frames
                                                        if segment.video_prefix_frames is None
                                                        else int(
                                                            segment.video_prefix_frames
                                                        )
                                                    )
                                                )
                                                if segment.transition == "continue"
                                                else 0
                                                for segment in group_segments[1:]
                                            ),
                                        )
                                        (
                                            shot_video,
                                            shot_audio,
                                            shot_frames,
                                            shot_engine,
                                        ) = stitch_clean_av_segment_files(
                                            group_paths,
                                            group_contexts,
                                            expected_frames=group.physical_frames,
                                            video_repaint_frames=(
                                                group_video_repaints
                                            ),
                                        )
                                        if shot_frames != group.physical_frames:
                                            raise RuntimeError(
                                                "shot decode staging missed its physical clock"
                                            )
                                        shot_path = temporary / (
                                            f"shot-decode-{group.index:02d}.pt"
                                        )
                                        torch.save({
                                            "video": shot_video,
                                            "frames": shot_frames,
                                            "visible_frames": group.visible_frames,
                                            "lead_context_frames": (
                                                group.lead_context_frames
                                            ),
                                            "fps": request.fps,
                                            "width": request.width,
                                            "height": request.height,
                                            "engine": shot_engine,
                                        }, shot_path)
                                        del shot_video, shot_audio
                                        staged_shots.append(shot_path)
                                    shot_video_checkpoints = tuple(staged_shots)

                            def decode_progress(event):
                                if progress_callback is None:
                                    return
                                local = float(event.get("percent", 0.0)) / 100.0
                                decode_start = 90.0 if global_audio_spine else 84.0
                                decode_span = 10.0 if global_audio_spine else 16.0
                                progress_callback({
                                    "percent": decode_start + decode_span * local,
                                    "stage": "long_horizon_decode",
                                    "detail": event.get(
                                        "detail", event.get("stage", "解码长视频")
                                    ),
                                })

                            if request.execution_plan is None:
                                raise RuntimeError(
                                    "long-horizon decode requires an explicit execution plan"
                                )
                            decode_request = replace(
                                request,
                                prompt=spec.prompt,
                                seed=spec.seed,
                                frames=stitched_frames,
                                output_path=output_path,
                                first_frame=None,
                                last_frame=None,
                                reference_images=(),
                                reference_videos=(),
                                reference_audios=(),
                                progress_callback=decode_progress,
                                refinement_latents_path=None,
                                refinement_denoise=None,
                                continuation_latents_path=None,
                                continuation_context_frames=0,
                                continuation_audio_bridge_ticks=0,
                                audio_manifold_guard=bool(bounded_token_memory),
                                av_token_memory_path=None,
                                save_final_latents_path=None,
                                conditioning_cache_source_path=None,
                                internal_video_tokens=None,
                                internal_audio_tokens=None,
                                latent_only=False,
                                checkpoint_after_step=None,
                                checkpoint_state_path=None,
                                preview_step_index=None,
                                preview_output_path=None,
                                preview_latents_path=None,
                                terminal_refinement_initial_width=None,
                                terminal_refinement_initial_height=None,
                                terminal_refinement_steps=0,
                                execution_plan=replace(
                                    request.execution_plan,
                                    vae_temporal_tile=6,
                                    vae_tile_batch_size=(
                                        8 if built.vram_profile == "8gb" else 1
                                    ),
                                    # A research worker may start cold without
                                    # the release-only prebuilt decoder graph.
                                    # Eager VAE math is exact and must not
                                    # invalidate a completed long AV trajectory.
                                    vae_transformer_block_compile=False,
                                ),
                            )
                            decoded = built.session.decode_latent_checkpoint(
                                decode_request,
                                stitched_path,
                                shot_video_checkpoints=shot_video_checkpoints,
                                audio_window_checkpoints=(
                                    ()
                                    if global_audio_spine
                                    else tuple(segment_paths)
                                ),
                                audio_window_clocks=(
                                    ()
                                    if global_audio_spine
                                    else tuple(
                                        (
                                            int(segment.context_frames)
                                            + int(segment.opening_preroll_frames),
                                            int(segment.visible_frames),
                                        )
                                        for segment in long_plan.segments
                                    )
                                ),
                            )
                            all_phases.update({
                                f"final_decode.{name}": seconds
                                for name, seconds in decoded.phases.items()
                            })
                            speech_authority_profile: dict[str, Any] | None = None
                            if structured_speech_gate:
                                from .speech_authority import (
                                    SpeechAuthorityGateConfig,
                                    apply_vocal_residual_gate,
                                    forbidden_speech_intervals,
                                )

                                denied_intervals = forbidden_speech_intervals(
                                    long_plan
                                )
                                gate_started = time.monotonic()
                                speech_authority_profile = (
                                    apply_vocal_residual_gate(
                                        decoded.output_path,
                                        denied_intervals,
                                        config=(
                                            SpeechAuthorityGateConfig
                                            .from_environment()
                                        ),
                                    )
                                )
                                all_phases[
                                    "final_decode.speech_authority_gate"
                                ] = time.monotonic() - gate_started
                            return replace(
                                decoded,
                                total_seconds=time.monotonic() - started,
                                phases=all_phases,
                                step_seconds=tuple(all_steps),
                                execution_profile={
                                    **decoded.execution_profile,
                                    "long_horizon": {
                                        **long_plan.telemetry(),
                                        **({"window_interface": {
                                            **authored_preview,
                                            "memory_budget": authored_memory,
                                            "effective_window_prompts": segment_prompts,
                                            "audio_bridge_ticks_by_window": list(audio_bridge_schedule),
                                        }} if authored_preview is not None else {}),
                                        "single_model_session": True,
                                        "intermediate_decode": False,
                                        "single_final_decode": bool(
                                            global_audio_spine
                                        ),
                                        "single_final_video_encode": True,
                                        "joint_av_prefix": True,
                                        "audio_seam_method": (
                                            "windowed_global_audio_spine_v2"
                                            if global_audio_spine
                                            else (
                                                "window_local_audio_vae_"
                                                "pcm_overlap_save_v1"
                                            )
                                            if bounded_token_memory
                                            else "hard_prefix_cut_v1"
                                        ),
                                        "audio_context_frames": long_plan.context_frames,
                                        "audio_bridge_ticks": (
                                            max(audio_bridge_schedule, default=0)
                                        ),
                                        "audio_repaint_overlap_discarded": bool(
                                            bounded_token_memory
                                            and not global_audio_spine
                                        ),
                                        "audio_latent_overlap_add": False,
                                        "bounded_av_token_memory": (
                                            bounded_token_memory
                                        ),
                                        "inferred_audio_token_memory": (
                                            inferred_audio_token_memory
                                        ),
                                        "long_visual_memory": long_visual_memory,
                                        "latest_visual_only": latest_visual_only,
                                        "generated_voice_bank": (
                                            {
                                                "method": "ref2va_frozen_self_anchor_v1",
                                                "audio_slots": 1,
                                                "audio_ticks_per_slot": generated_voice_ticks,
                                                "refresh_policy": "write_once",
                                                "denoise_exposure": "all_steps",
                                                "prompt_binding": "h3_audio1_subject_s1_v1",
                                                "segment_bindings": voice_bank_binding_records,
                                            }
                                            if generated_voice_bank
                                            else None
                                        ),
                                        "authored_voice_anchor": (
                                            {
                                                "method": (
                                                    "first_eligible_dialogue_"
                                                    "native_latent_v1"
                                                ),
                                                "audio_slots": 1,
                                                "audio_ticks_per_slot": (
                                                    authored_memory[
                                                        "audio_ticks_per_clip"
                                                    ]
                                                ),
                                                "refresh_policy": "write_once",
                                                "collection_authority": (
                                                    "structured_dialogue_clock"
                                                ),
                                                "injection_authority": (
                                                    "dialogue_windows_only"
                                                ),
                                            }
                                            if (
                                                authored_memory is not None
                                                and inferred_audio_token_memory
                                                and not request.reference_audios
                                            )
                                            else None
                                        ),
                                        "global_audio_spine": audio_spine_telemetry,
                                        "global_audio_spine_execution_profile": (
                                            audio_spine_profile
                                        ),
                                        "window_audio_discarded": global_audio_spine,
                                        "structured_speech_authority": (
                                            speech_authority_profile
                                        ),
                                        "token_memory_records": (
                                            token_memory_records
                                        ),
                                        "token_memory_route_records": (
                                            token_memory_route_records
                                        ),
                                        "unique_conditioning_encodes": len(
                                            conditioning_receipts
                                        ),
                                        "conditioning_preencoded_before_dit": bool(
                                            conditioning_receipts
                                        ),
                                        "continuation_semantic_handoff": (
                                            continuation_semantic_handoff
                                        ),
                                        "continuation_video_handoff": (
                                            continuation_video_handoff
                                        ),
                                        "system_temp_staging": True,
                                        "streaming_latent_stitch": True,
                                        "window_profiles": window_profiles,
                                    },
                                },
                                peak_allocated_gib=max(
                                    peak_allocated, decoded.peak_allocated_gib
                                ),
                                peak_reserved_gib=max(
                                    peak_reserved, decoded.peak_reserved_gib
                                ),
                            )

                    result = await asyncio.to_thread(run_long_horizon)
                elif is_incremental_continuation:
                    from ..infinite_video import memory_capacity
                    from .av_token_memory import (
                        empty_av_token_memory,
                        route_audio_memory_authority,
                        token_memory_telemetry,
                        update_av_token_memory,
                        validate_av_token_memory,
                    )
                    from .long_horizon import stitch_clean_av_segment_files
                    import torch

                    def run_incremental_continuation():
                        if continuation is None:
                            raise RuntimeError("missing infinite continuation contract")
                        source_path = Path(continuation_source_latents_path)
                        source = torch.load(
                            source_path, map_location="cpu", weights_only=True
                        )
                        expected_source = {
                            "frames": continuation.source_frames,
                            "fps": request.fps,
                            "width": (
                                request.multiscale_initial_width
                                if request.multiscale_initial_width is not None
                                else request.width
                            ),
                            "height": (
                                request.multiscale_initial_height
                                if request.multiscale_initial_height is not None
                                else request.height
                            ),
                        }
                        for key in ("frames", "fps"):
                            value = expected_source[key]
                            if source.get(key) != value:
                                raise ValueError(
                                    f"infinite continuation source mismatch for {key}: "
                                    f"expected {value!r}, got {source.get(key)!r}"
                                )
                        expected_geometries = {
                            (
                                expected_source["width"],
                                expected_source["height"],
                            )
                        }
                        if request.multiscale_resize_after_step is not None:
                            # JSON direct mode carries the prior completed
                            # high-resolution window. The hot session reduces
                            # only its terminal context for the next low-res
                            # prefix, then SelfLift returns the new piece to
                            # the target canvas before stitching.
                            expected_geometries.add(
                                (request.width, request.height)
                            )
                        source_geometry = (
                            source.get("width"), source.get("height")
                        )
                        if source_geometry not in expected_geometries:
                            raise ValueError(
                                "infinite continuation source mismatch for geometry: "
                                f"expected one of {sorted(expected_geometries)!r}, "
                                f"got {source_geometry!r}"
                            )

                        capacity = memory_capacity(
                            continuation.memory,
                            service_family=spec.service_family,
                            visual_capacity=continuation.visual_memory_capacity,
                            audio_capacity=continuation.audio_memory_capacity,
                            visual_resolution=continuation.visual_memory_resolution,
                        )
                        token_memory = None
                        routed_memory_path = None
                        memory_reused = False
                        if capacity["enabled"]:
                            if continuation_source_memory_path is not None:
                                candidate_memory = torch.load(
                                    Path(continuation_source_memory_path),
                                    map_location="cpu",
                                    weights_only=True,
                                )
                                candidate_memory = validate_av_token_memory(
                                    candidate_memory
                                )
                                if (
                                    candidate_memory["video_slots"]
                                    == capacity["video_slots"]
                                    and candidate_memory["audio_slots"]
                                    == capacity["audio_slots"]
                                    and candidate_memory["audio_block_ticks"]
                                    == capacity["audio_ticks_per_clip"]
                                    and candidate_memory["visual_resolution"]
                                    == capacity["visual_resolution"]
                                ):
                                    token_memory = candidate_memory
                                    memory_reused = True
                            if token_memory is None:
                                token_memory = empty_av_token_memory(
                                    video_slots=int(capacity["video_slots"]),
                                    audio_slots=int(capacity["audio_slots"]),
                                    audio_block_ticks=int(
                                        capacity["audio_ticks_per_clip"]
                                    ),
                                    visual_resolution=str(
                                        capacity["visual_resolution"]
                                    ),
                                )
                                token_memory = update_av_token_memory(
                                    token_memory,
                                    source,
                                    context_frames=0,
                                    visible_start_frame=0,
                                    visible_frames=continuation.source_frames,
                                    preserve_latest_visual=True,
                                    collect_audio=(
                                        int(capacity["audio_slots"]) > 0
                                        and continuation.source_dialogue
                                        and not request.reference_audios
                                    ),
                                )
                            # Generated speech memory is injected only into an
                            # explicitly authored dialogue window. This avoids
                            # treating remembered words as ambient audio and
                            # recreating the historical reverse/replay defect.
                            routed_memory = route_audio_memory_authority(
                                token_memory,
                                active=(
                                    "<d>" in request.prompt
                                    and not request.reference_audios
                                ),
                            )

                        all_phases: dict[str, float] = {}
                        all_steps: list[float] = []
                        with tempfile.TemporaryDirectory(
                            prefix=".h3-infinite-continuation-"
                        ) as temporary_root:
                            temporary = Path(temporary_root)
                            if capacity["enabled"]:
                                routed_memory_path = temporary / "routed-memory.pt"
                                torch.save(routed_memory, routed_memory_path)
                            piece_path = temporary / "continuation-window.pt"

                            def piece_progress(event):
                                if progress_callback is None:
                                    return
                                local = float(event.get("percent", 0.0))
                                progress_callback({
                                    "percent": 4.0 + 0.78 * local,
                                    "stage": "infinite_continuation_window",
                                    "detail": event.get(
                                        "detail", "正在生成严格续写窗口"
                                    ),
                                })

                            piece_request = replace(
                                request,
                                frames=continuation.physical_frames,
                                output_path=output_path,
                                progress_callback=piece_progress,
                                first_frame=None,
                                last_frame=None,
                                refinement_latents_path=None,
                                refinement_denoise=None,
                                continuation_latents_path=(
                                    source_path
                                    if continuation.context_frames > 0
                                    else None
                                ),
                                continuation_context_frames=(
                                    continuation.context_frames
                                ),
                                continuation_video_prefix_frames=(
                                    continuation.context_frames
                                    if continuation.context_frames > 0
                                    else None
                                ),
                                continuation_video_prefix_from_source_end=False,
                                continuation_audio_bridge_ticks=(
                                    continuation.audio_bridge_ticks
                                ),
                                # Window-interface V14 falsified mixing an old
                                # and current text-conditioned velocity field:
                                # it doubled DiT work and moved camera cuts to
                                # the blend-band edges.  Incremental creation
                                # uses the same accepted V15+ single-semantic
                                # trajectory as batch long-horizon generation.
                                continuation_text_bridge_conditioning_path=None,
                                av_token_memory_path=routed_memory_path,
                                conditioning_cache_source_path=None,
                                save_final_latents_path=piece_path,
                                internal_video_tokens=None,
                                internal_audio_tokens=None,
                                latent_only=True,
                                retain_transformer_after_latent_only=False,
                                sampler_state_path=None,
                                formal_resume_state_path=None,
                                checkpoint_after_step=request.checkpoint_after_step,
                                checkpoint_state_path=request.checkpoint_state_path,
                                preview_step_index=request.preview_step_index,
                                preview_output_path=request.preview_output_path,
                                preview_latents_path=(
                                    piece_path
                                    if request.checkpoint_after_step is not None
                                    else None
                                ),
                                preview_ready_callback=None,
                                preview_decision_wait=None,
                                terminal_refinement_initial_width=None,
                                terminal_refinement_initial_height=None,
                                terminal_refinement_steps=0,
                            )
                            piece_result = built.session.generate(piece_request)
                            for name, seconds in getattr(
                                piece_result, "phases", {}
                            ).items():
                                all_phases[f"window.{name}"] = seconds
                            all_steps.extend(
                                getattr(piece_result, "step_seconds", ())
                            )
                            piece = torch.load(
                                piece_path, map_location="cpu", weights_only=True
                            )

                            memory_receipt = None
                            if capacity["enabled"]:
                                token_memory = update_av_token_memory(
                                    token_memory,
                                    piece,
                                    context_frames=continuation.hidden_prefix_frames,
                                    visible_start_frame=continuation.source_frames,
                                    visible_frames=continuation.visible_frames,
                                    preserve_latest_visual=True,
                                    collect_audio=(
                                        int(capacity["audio_slots"]) > 0
                                        and "<d>" in request.prompt
                                        and not request.reference_audios
                                    ),
                                )
                                memory_target = Path(
                                    continuation_output_memory_path
                                )
                                memory_target.parent.mkdir(
                                    parents=True, exist_ok=True
                                )
                                torch.save(token_memory, memory_target)
                                memory_receipt = token_memory_telemetry(
                                    token_memory
                                )

                            (
                                stitched_video,
                                stitched_audio,
                                stitched_frames,
                                stitched_engine,
                            ) = stitch_clean_av_segment_files(
                                (source_path, piece_path),
                                (0, continuation.hidden_prefix_frames),
                                expected_frames=continuation.output_frames,
                                audio_bridge_ticks=(
                                    0, continuation.audio_trim_ticks
                                ),
                            )
                            stitched_path = Path(final_latents_path)
                            stitched_path.parent.mkdir(parents=True, exist_ok=True)
                            stitched_document = {
                                "video": stitched_video,
                                "audio": stitched_audio,
                                "audio_final": bool(
                                    source.get("audio_final", False)
                                    and piece.get("audio_final", False)
                                ),
                                "frames": stitched_frames,
                                "fps": request.fps,
                                # Incremental SelfLift checkpoints store the
                                # completed source-grid x0 track.  The request
                                # itself targets the later high-resolution
                                # branch, so copying request.width/height here
                                # lies about the tensor geometry and makes the
                                # next continuation treat a 540p latent as a
                                # 1080p target prefix.  Derive the only valid
                                # metadata directly from the saved tensor.
                                "width": int(stitched_video.shape[-1]) * 16,
                                "height": int(stitched_video.shape[-2]) * 16,
                                "engine": stitched_engine,
                                "seed": request.seed,
                                "infinite_project_id": continuation.project_id,
                                "infinite_window_index": continuation.window_index,
                            }
                            latest_conditioning = getattr(
                                built.session,
                                "_last_conditioning_cache_payload",
                                None,
                            )
                            if isinstance(latest_conditioning, dict):
                                stitched_document["qwen_conditioning_cache"] = (
                                    latest_conditioning
                                )
                            torch.save(stitched_document, stitched_path)
                            del stitched_video, stitched_audio, piece, source

                            if (
                                isinstance(piece_result, HotSessionCheckpointResult)
                                and request.preview_output_path is None
                            ):
                                # JSON one-click creation retains the connected
                                # source-grid x0 track and the formal fork only.
                                # Decoding the growing cumulative preview here
                                # would add quadratic VAE work that the user
                                # never requested; the project-level global
                                # SelfLift pass owns the sole final decode.
                                return replace(
                                    piece_result,
                                    preview_path=None,
                                    preview_latents_path=stitched_path,
                                    token_memory_path=(
                                        Path(continuation_output_memory_path)
                                        if memory_receipt is not None
                                        else None
                                    ),
                                    total_seconds=time.monotonic() - started,
                                    phases=all_phases,
                                    execution_profile={
                                        **getattr(
                                            piece_result,
                                            "execution_profile",
                                            {},
                                        ),
                                        "infinite_selflift_source": {
                                            "project_id": continuation.project_id,
                                            "window_index": continuation.window_index,
                                            "source_frames": continuation.source_frames,
                                            "context_frames": continuation.context_frames,
                                            "visible_frames": continuation.visible_frames,
                                            "output_frames": continuation.output_frames,
                                            "decoded_preview": False,
                                            "shared_prefix": True,
                                        },
                                    },
                                )

                            if (
                                isinstance(piece_result, HotSessionCheckpointResult)
                                and request.preview_output_path is not None
                                and piece_result.preview_path is not None
                                and continuation_source_video_path is not None
                            ):
                                # Online creation already decoded this physical
                                # low-resolution window for the user. Reuse it:
                                # trim the hidden continuation context, append
                                # only the visible suffix, and packet-copy every
                                # previously accepted frame. The old path below
                                # remains a safe fallback if external ffmpeg
                                # cannot assemble the preview.
                                from .incremental_preview import (
                                    append_cumulative_preview,
                                )

                                preview_started = time.perf_counter()
                                try:
                                    preview_receipt = append_cumulative_preview(
                                        continuation_source_video_path,
                                        piece_result.preview_path,
                                        piece_result.preview_path,
                                        source_frames=continuation.source_frames,
                                        context_frames=continuation.hidden_prefix_frames,
                                        physical_frames=continuation.physical_frames,
                                        output_frames=continuation.output_frames,
                                        fps=request.fps,
                                    )
                                except Exception as error:
                                    all_phases[
                                        "incremental_preview_append_failed"
                                    ] = time.perf_counter() - preview_started
                                    # Keep the timing receipt numeric. The
                                    # established cumulative latent decode
                                    # below is the correctness fallback.
                                    del error
                                else:
                                    all_phases[
                                        "incremental_preview_append"
                                    ] = time.perf_counter() - preview_started
                                    return replace(
                                        piece_result,
                                        preview_path=piece_result.preview_path,
                                        preview_latents_path=stitched_path,
                                        token_memory_path=(
                                            Path(continuation_output_memory_path)
                                            if memory_receipt is not None
                                            else None
                                        ),
                                        total_seconds=time.monotonic() - started,
                                        phases=all_phases,
                                        execution_profile={
                                            **getattr(
                                                piece_result,
                                                "execution_profile",
                                                {},
                                            ),
                                            "infinite_selflift_preview": {
                                                "project_id": continuation.project_id,
                                                "window_index": continuation.window_index,
                                                "source_frames": continuation.source_frames,
                                                "context_frames": continuation.context_frames,
                                                "visible_frames": continuation.visible_frames,
                                                "output_frames": continuation.output_frames,
                                                "preview_latent_geometry": [
                                                    expected_source["width"],
                                                    expected_source["height"],
                                                ],
                                                "formal_checkpoint_geometry": [
                                                    request.width,
                                                    request.height,
                                                ],
                                                "shared_prefix": True,
                                                "preview_assembly": preview_receipt,
                                            },
                                        },
                                    )

                            if request.execution_plan is None:
                                raise RuntimeError(
                                    "infinite continuation decode requires an execution plan"
                                )
                            decode_request = replace(
                                request,
                                frames=continuation.output_frames,
                                progress_callback=progress_callback,
                                first_frame=None,
                                last_frame=None,
                                reference_images=(),
                                reference_videos=(),
                                reference_audios=(),
                                prepared_reference_images=(),
                                prepared_reference_videos=(),
                                prepared_reference_audios=(),
                                continuation_latents_path=None,
                                continuation_context_frames=0,
                                continuation_video_prefix_frames=None,
                                continuation_video_prefix_from_source_end=False,
                                continuation_audio_bridge_ticks=0,
                                continuation_text_bridge_conditioning_path=None,
                                av_token_memory_path=None,
                                save_final_latents_path=None,
                                internal_video_tokens=None,
                                internal_audio_tokens=None,
                                latent_only=False,
                                execution_plan=replace(
                                    request.execution_plan,
                                    vae_temporal_tile=6,
                                    vae_tile_batch_size=(
                                        8 if built.vram_profile == "8gb" else 1
                                    ),
                                    vae_transformer_block_compile=False,
                                ),
                            )
                            decoded = built.session.decode_latent_checkpoint(
                                decode_request,
                                stitched_path,
                                audio_window_checkpoints=(
                                    source_path, piece_path
                                ),
                                audio_window_clocks=(
                                    (0, continuation.source_frames),
                                    (
                                        continuation.hidden_prefix_frames,
                                        continuation.visible_frames,
                                    ),
                                ),
                            )
                            for name, seconds in getattr(
                                decoded, "phases", {}
                            ).items():
                                all_phases[f"final_decode.{name}"] = seconds
                            if isinstance(piece_result, HotSessionCheckpointResult):
                                return replace(
                                    piece_result,
                                    preview_path=decoded.output_path,
                                    preview_latents_path=stitched_path,
                                    token_memory_path=(
                                        Path(continuation_output_memory_path)
                                        if memory_receipt is not None
                                        else None
                                    ),
                                    total_seconds=time.monotonic() - started,
                                    phases=all_phases,
                                    execution_profile={
                                        **getattr(
                                            piece_result,
                                            "execution_profile",
                                            {},
                                        ),
                                        **getattr(decoded, "execution_profile", {}),
                                        "infinite_selflift_preview": {
                                            "project_id": continuation.project_id,
                                            "window_index": continuation.window_index,
                                            "source_frames": continuation.source_frames,
                                            "context_frames": continuation.context_frames,
                                            "visible_frames": continuation.visible_frames,
                                            "output_frames": continuation.output_frames,
                                            "preview_latent_geometry": [
                                                expected_source["width"],
                                                expected_source["height"],
                                            ],
                                            "formal_checkpoint_geometry": [
                                                request.width,
                                                request.height,
                                            ],
                                            "shared_prefix": True,
                                        },
                                    },
                                    peak_allocated_gib=max(
                                        float(getattr(
                                            piece_result,
                                            "peak_allocated_gib",
                                            0.0,
                                        )),
                                        float(getattr(
                                            decoded,
                                            "peak_allocated_gib",
                                            0.0,
                                        )),
                                    ),
                                    peak_reserved_gib=max(
                                        float(getattr(
                                            piece_result,
                                            "peak_reserved_gib",
                                            0.0,
                                        )),
                                        float(getattr(
                                            decoded,
                                            "peak_reserved_gib",
                                            0.0,
                                        )),
                                    ),
                                )
                            return replace(
                                decoded,
                                total_seconds=time.monotonic() - started,
                                phases=all_phases,
                                step_seconds=tuple(all_steps),
                                execution_profile={
                                    **getattr(piece_result, "execution_profile", {}),
                                    **getattr(decoded, "execution_profile", {}),
                                    "infinite_window_forecast": getattr(
                                        piece_result, "forecast_profile", {}
                                    ),
                                    "infinite_continuation": {
                                        "project_id": continuation.project_id,
                                        "window_index": continuation.window_index,
                                        "source_frames": continuation.source_frames,
                                        "context_frames": continuation.context_frames,
                                        "visible_frames": continuation.visible_frames,
                                        "output_frames": continuation.output_frames,
                                        "physical_boundary": "strict_continuation",
                                        "semantic_trajectory": (
                                            "single_current_prompt_v15"
                                        ),
                                        "camera_cuts": "current_window_prompt_only",
                                        "audio_decode": "window_local_overlap_save",
                                        "memory": {
                                            "amount": continuation.memory,
                                            "visual_capacity": continuation.visual_memory_capacity,
                                            "audio_capacity": continuation.audio_memory_capacity,
                                            "visual_resolution": continuation.visual_memory_resolution,
                                            "capacity": capacity,
                                            "reused_previous_checkpoint": memory_reused,
                                            "receipt": memory_receipt,
                                        },
                                    },
                                },
                                peak_allocated_gib=max(
                                    float(getattr(piece_result, "peak_allocated_gib", 0.0)),
                                    float(getattr(decoded, "peak_allocated_gib", 0.0)),
                                ),
                                peak_reserved_gib=max(
                                    float(getattr(piece_result, "peak_reserved_gib", 0.0)),
                                    float(getattr(decoded, "peak_reserved_gib", 0.0)),
                                ),
                            )

                    result = await asyncio.to_thread(
                        run_incremental_continuation
                    )
                elif ultimate_plan is not None and not ultimate_plan.full_canvas:
                    from .ultimate_upscale import (
                        append_av_temporal_piece,
                        slice_av_temporal_piece,
                    )
                    import torch

                    def run_temporal_windows():
                        source = torch.load(
                            Path(refinement_latents_path),
                            map_location="cpu",
                            weights_only=True,
                        )
                        source_video = source.get("video")
                        source_audio = source.get("audio")
                        if not isinstance(source_video, torch.Tensor) or source_video.ndim != 5:
                            raise ValueError("second-sampling source has invalid video latent")
                        if not isinstance(source_audio, torch.Tensor) or source_audio.ndim != 4:
                            raise ValueError("second-sampling source has invalid audio latent")
                        if source_video.shape[2] != ultimate_plan.temporal[-1].video_token_stop:
                            raise ValueError("UltimateUpscale plan does not cover source video clock")
                        if source_audio.shape[-1] < ultimate_plan.temporal[-1].audio_token_stop:
                            raise ValueError("UltimateUpscale plan does not cover source audio clock")

                        accumulated_video = None
                        accumulated_audio = None
                        all_phases: dict[str, float] = {}
                        all_steps: list[float] = []
                        window_profiles: list[dict[str, Any]] = []
                        rebuilt_conditioning: dict[str, Any] | None = None
                        peak_allocated = 0.0
                        peak_reserved = 0.0
                        window_count = len(ultimate_plan.temporal)
                        with tempfile.TemporaryDirectory(
                            prefix=".h3-ultimate-",
                        ) as temporary_root:
                            temporary = Path(temporary_root)
                            for index, piece in enumerate(ultimate_plan.temporal):
                                if cancel_event.is_set():
                                    raise HotSessionCancelled(
                                        "native H3 generation cancelled"
                                    )
                                piece_video, piece_audio = slice_av_temporal_piece(
                                    source_video, source_audio, piece
                                )
                                piece_input = temporary / f"window-{index:02d}-source.pt"
                                piece_output = temporary / f"window-{index:02d}-sampled.pt"
                                torch.save(
                                    {
                                        "video": piece_video,
                                        "audio": piece_audio,
                                        "frames": piece.frames,
                                        "fps": request.fps,
                                        "width": source.get("width"),
                                        "height": source.get("height"),
                                        "engine": source.get("engine"),
                                        "seed": request.seed,
                                    },
                                    piece_input,
                                )

                                def piece_progress(event, *, _index=index):
                                    if progress_callback is None:
                                        return
                                    local = float(event.get("percent", 0.0)) / 100.0
                                    progress_callback({
                                        "percent": 8.0 + 76.0 * ((_index + local) / window_count),
                                        "stage": "second_sampling_window",
                                        "detail": (
                                            f"{second_sampling.resolution} 时间窗口 "
                                            f"{_index + 1}/{window_count} · "
                                            f"{event.get('detail', event.get('stage', '执行中'))}"
                                        ),
                                    })

                                piece_request = replace(
                                    request,
                                    frames=piece.frames,
                                    # This temporal piece is latent-only. Keep
                                    # its public output placeholder authorized
                                    # while source/sampled tensors use TMPDIR.
                                    output_path=output_path,
                                    first_frame=(request.first_frame if index == 0 else None),
                                    last_frame=(
                                        request.last_frame
                                        if index == window_count - 1
                                        else None
                                    ),
                                    progress_callback=piece_progress,
                                    refinement_latents_path=piece_input,
                                    save_final_latents_path=piece_output,
                                    internal_video_tokens=(
                                        piece.video_token_stop - piece.video_token_start
                                    ),
                                    internal_audio_tokens=(
                                        piece.audio_token_stop - piece.audio_token_start
                                    ),
                                    latent_only=True,
                                    formal_resume_state_path=None,
                                    checkpoint_after_step=None,
                                    checkpoint_state_path=None,
                                    preview_step_index=None,
                                    preview_output_path=None,
                                    terminal_refinement_initial_width=None,
                                    terminal_refinement_initial_height=None,
                                    terminal_refinement_steps=0,
                                )
                                piece_result = built.session.generate(piece_request)
                                sampled = torch.load(
                                    piece_output, map_location="cpu", weights_only=True
                                )
                                accumulated_video, accumulated_audio = append_av_temporal_piece(
                                    accumulated_video,
                                    accumulated_audio,
                                    sampled["video"],
                                    sampled["audio"],
                                    piece,
                                )
                                for name, seconds in piece_result.phases.items():
                                    all_phases[
                                        f"window_{index + 1:02d}.{name}"
                                    ] = seconds
                                all_steps.extend(piece_result.step_seconds)
                                window_profiles.append(piece_result.execution_profile)
                                latest_conditioning = getattr(
                                    built.session,
                                    "_last_conditioning_cache_payload",
                                    None,
                                )
                                if (
                                    rebuilt_conditioning is None
                                    and isinstance(latest_conditioning, dict)
                                ):
                                    # A legacy source latent has no persisted
                                    # Qwen cache.  The first temporal piece
                                    # necessarily rebuilds it; retain that exact
                                    # host payload so the stitched checkpoint is
                                    # automatically upgraded for future runs.
                                    rebuilt_conditioning = latest_conditioning
                                peak_allocated = max(
                                    peak_allocated, piece_result.peak_allocated_gib
                                )
                                peak_reserved = max(
                                    peak_reserved, piece_result.peak_reserved_gib
                                )
                                del sampled, piece_video, piece_audio

                            if accumulated_video is None or accumulated_audio is None:
                                raise RuntimeError("UltimateUpscale produced no windows")
                            # The upstream algorithm never re-samples audio.  Use
                            # the original full clock byte-for-byte instead of a
                            # numerically equivalent overlap blend.
                            del accumulated_audio
                            accumulated_audio = source_audio
                            stitched_path = (
                                Path(final_latents_path)
                                if final_latents_path is not None
                                else temporary / "stitched-final.pt"
                            )
                            stitched_path.parent.mkdir(parents=True, exist_ok=True)
                            stitched_document = {
                                "video": accumulated_video,
                                "audio": accumulated_audio,
                                "frames": request.frames,
                                "fps": request.fps,
                                "width": request.width,
                                "height": request.height,
                                "engine": source.get("engine"),
                                "seed": request.seed,
                            }
                            source_conditioning = source.get(
                                "qwen_conditioning_cache"
                            )
                            if not isinstance(source_conditioning, dict):
                                source_conditioning = rebuilt_conditioning
                            if isinstance(source_conditioning, dict):
                                stitched_document["qwen_conditioning_cache"] = (
                                    source_conditioning
                                )
                            torch.save(stitched_document, stitched_path)
                            decode_request = replace(
                                request,
                                refinement_latents_path=None,
                                refinement_denoise=None,
                                refinement_spatial_mode="strict",
                                save_final_latents_path=None,
                                internal_video_tokens=None,
                                internal_audio_tokens=None,
                                latent_only=False,
                                # The stitched high-resolution latent is intentionally
                                # decoded through the already validated exact
                                # host-temporal Video-VAE graph.  Reusing the
                                # per-window DiT plan here would materialize a
                                # full FP32 2K clip on GPU and consume the last
                                # ~0.6 GiB of the card for no speed benefit.
                                execution_plan=replace(
                                    request.execution_plan,
                                    vae_spatial_tile=request.execution_plan.vae_spatial_tile,
                                    vae_temporal_tile=6,
                                    vae_tile_batch_size=(
                                        8
                                        if built.vram_profile == "8gb"
                                        else 1
                                    ),
                                ),
                            )
                            decoded = built.session.decode_latent_checkpoint(
                                decode_request, stitched_path
                            )
                            all_phases.update({
                                f"final_decode.{name}": seconds
                                for name, seconds in decoded.phases.items()
                            })
                            return replace(
                                decoded,
                                total_seconds=time.monotonic() - started,
                                phases=all_phases,
                                step_seconds=tuple(all_steps),
                                execution_profile={
                                    **decoded.execution_profile,
                                    **(
                                        {
                                            "qwen_conditioning_cache": dict(
                                                window_profiles[0][
                                                    "qwen_conditioning_cache"
                                                ]
                                            )
                                        }
                                        if window_profiles
                                        and isinstance(
                                            window_profiles[0].get(
                                                "qwen_conditioning_cache"
                                            ),
                                            dict,
                                        )
                                        else {}
                                    ),
                                    "ultimate_upscale_windows": {
                                        "provenance": ultimate_plan.provenance,
                                        "count": window_count,
                                        "full_spatial_canvas": True,
                                        "audio_resampled": False,
                                        "latent_crossfade": True,
                                        "single_final_decode": True,
                                        "window_profiles": window_profiles,
                                    },
                                },
                                peak_allocated_gib=max(
                                    peak_allocated, decoded.peak_allocated_gib
                                ),
                                peak_reserved_gib=max(
                                    peak_reserved, decoded.peak_reserved_gib
                                ),
                            )

                    result = await asyncio.to_thread(run_temporal_windows)
                else:
                    profile_path_raw = os.environ.get(
                        "H3_NATIVE_RESEARCH_TORCH_PROFILE_PATH", ""
                    ).strip()

                    def run_request():
                        if not profile_path_raw:
                            return built.session.generate(request)
                        import torch

                        profile_path = Path(profile_path_raw)
                        profile_path.parent.mkdir(parents=True, exist_ok=True)
                        with torch.profiler.profile(
                            activities=(
                                torch.profiler.ProfilerActivity.CPU,
                                torch.profiler.ProfilerActivity.CUDA,
                            ),
                            record_shapes=False,
                            profile_memory=False,
                        ) as profiler:
                            try:
                                profiled_result = built.session.generate(request)
                            except Exception:
                                traceback.print_exc()
                                raise
                        if os.environ.get(
                            "H3_NATIVE_RESEARCH_TORCH_EXPORT_TRACE", "0"
                        ) == "1":
                            profiler.export_chrome_trace(str(profile_path))
                        aggregate_rows = []
                        for item in profiler.key_averages():
                            aggregate_rows.append({
                                "key": str(item.key),
                                "count": int(item.count),
                                "cpu_time_total_us": float(item.cpu_time_total),
                                "self_cpu_time_total_us": float(
                                    item.self_cpu_time_total
                                ),
                                "device_time_total_us": float(
                                    getattr(item, "device_time_total", 0.0)
                                ),
                                "self_device_time_total_us": float(
                                    getattr(item, "self_device_time_total", 0.0)
                                ),
                            })
                        profile_path.write_text(
                            json.dumps(
                                {"schema_version": 1, "events": aggregate_rows},
                                indent=2,
                            ),
                            encoding="utf-8",
                        )
                        profile_path.with_suffix(".kernels.txt").write_text(
                            profiler.key_averages().table(
                                sort_by="cuda_time_total", row_limit=250
                            ),
                            encoding="utf-8",
                        )
                        return profiled_result

                    result = await asyncio.to_thread(run_request)
            except HotSessionCancelled as error:
                raise NativeGenerationCancelled(str(error)) from error
            except HotSessionDeviceFatal as error:
                # Do not rebuild or close a session after an asynchronous CUDA
                # illegal-access fault.  Further CUDA API calls can block or
                # abort the process.  Keep HTTP responsive, fail subsequent
                # jobs immediately, and require a clean process restart.
                self._device_fatal_error = str(error)
                self._warm_state = {
                    "status": "failed",
                    "engine": launcher_family(spec.runtime_launcher),
                    "launcher": spec.runtime_launcher,
                    "weight_tier": spec.weight_tier,
                    "vram_profile": spec.vram_profile,
                    "startup_seconds": None,
                    "error": "fatal CUDA device error",
                    "progress_percent": 100.0,
                    "progress_stage": "restart_required",
                    "progress_detail": "CUDA引擎异常，需要重启服务",
                }
                raise RuntimeError(
                    "fatal CUDA device error; restart the H3 service"
                ) from error
            if isinstance(result, HotSessionCheckpointResult):
                public_plan = _public_inference_plan(
                    getattr(result, "execution_profile", None)
                )
                if result.peak_allocated_gib > 0.0 or result.peak_reserved_gib > 0.0:
                    public_plan = dict(public_plan or {})
                    public_plan["runtime_memory"] = {
                        "peak_allocated_gib": round(result.peak_allocated_gib, 4),
                        "peak_reserved_gib": round(result.peak_reserved_gib, 4),
                        "allocator_ceiling_gib": round(allocator_ceiling_gib, 4),
                        "vram_profile": spec.vram_profile,
                    }
                return NativeCheckpointResult(
                    runtime_key=(
                        f"{spec.engine}:{spec.weight_tier}:{spec.vram_profile}:native-sm89"
                    ),
                    elapsed_seconds=round(time.monotonic() - started, 3),
                    checkpoint_path=result.checkpoint_path,
                    preview_path=result.preview_path,
                    completed_steps=result.completed_steps,
                    total_steps=result.total_steps,
                    stage_seconds=dict(result.phases),
                    inference_plan=public_plan,
                    preview_latents_path=result.preview_latents_path,
                    token_memory_path=result.token_memory_path,
                )
            phases = dict(result.phases)
            if candidate is not None:
                from .detail_restore import (
                    DetailRestoreCancelled,
                    restore_intrame_detail,
                )

                if progress_callback is not None:
                    progress_callback({
                        "percent": 98,
                        "stage": "detail_restore",
                        "detail": "恢复画面细节",
                    })
                try:
                    restored = await asyncio.to_thread(
                        restore_intrame_detail,
                        result.output_path,
                        expected_width=spec.width,
                        expected_height=spec.height,
                        expected_frames=spec.frames,
                        cancel_check=cancel_event.is_set,
                        preserve_raw=True,
                        parallel_shards=4,
                        fps=24,
                    )
                except DetailRestoreCancelled as error:
                    raise NativeGenerationCancelled(str(error)) from error
                phases["intrame_detail_restore"] = restored.elapsed_seconds
            public_plan = _public_inference_plan(
                getattr(result, "execution_profile", None)
            )
            peak_allocated_gib = float(getattr(result, "peak_allocated_gib", 0.0))
            peak_reserved_gib = float(getattr(result, "peak_reserved_gib", 0.0))
            if peak_allocated_gib > 0.0 or peak_reserved_gib > 0.0:
                public_plan = dict(public_plan or {})
                public_plan["runtime_memory"] = {
                    "peak_allocated_gib": round(peak_allocated_gib, 4),
                    "peak_reserved_gib": round(peak_reserved_gib, 4),
                    "allocator_ceiling_gib": round(allocator_ceiling_gib, 4),
                    "vram_profile": spec.vram_profile,
                }
            if ultimate_plan is not None:
                public_plan = dict(public_plan or {})
                public_plan["ultimate_upscale"] = ultimate_plan.telemetry()
            if long_plan is not None:
                public_plan = dict(public_plan or {})
                route_receipt = public_plan.get("long_horizon")
                if not isinstance(route_receipt, dict):
                    route_receipt = {}
                public_plan["long_horizon"] = {
                    **long_plan.telemetry(),
                    **route_receipt,
                    "system_temp_staging": True,
                    "streaming_latent_stitch": True,
                }
            if global_co_plan is not None:
                public_plan = dict(public_plan or {})
                public_plan["long_horizon"] = {
                    **global_co_plan.telemetry(),
                    "single_global_solver_state": True,
                    "scheduler_updates_per_step": 1,
                    "intermediate_decode": False,
                    "single_final_decode": True,
                    "rotary_time": request.global_co_denoise_rotary_mode,
                }
            if second_sampling is not None:
                public_plan = dict(public_plan or {})
                public_plan["second_sampling_solver"] = {
                    "model_variant": second_sampling.model_variant,
                    "sampler": (
                        "turbo"
                        if second_sampling.model_variant == "lora"
                        else request.refinement_sampler
                    ),
                    "scheduler": "simple",
                    "video_shift": request.refinement_video_shift,
                    "sigma_curve_power": request.refinement_sigma_power,
                    "audio_shift": 3.0,
                    "steps": second_sampling.steps,
                    "strength": second_sampling.strength,
                    "denoise": second_sampling.denoise,
                    "start_sigma": refinement_sigma_schedule(
                        second_sampling.steps,
                        second_sampling.denoise,
                        request.refinement_video_shift,
                        curve_power=request.refinement_sigma_power,
                    )[0],
                    "forecast_enabled": (
                        request.actual_step_indices is not None
                        and len(request.actual_step_indices) < request.steps
                    ),
                    "detail_regeneration": {
                        "enabled": h3_detail_regeneration,
                        "prediction_low_frequency_gain": (
                            request.refinement_prediction_low_frequency_gain
                        ),
                        "final_low_frequency_gain": (
                            request.refinement_final_low_frequency_gain
                        ),
                        "temporal_lowpass": (
                            request.refinement_temporal_lowpass
                        ),
                        "temporal_outlier_only": (
                            request.refinement_temporal_outlier_only
                        ),
                        "temporal_detail_outlier_strength": (
                            request.refinement_temporal_detail_outlier_strength
                        ),
                        "cross_step_detail_strength": (
                            request.refinement_cross_step_detail_strength
                        ),
                        "model_authority": "h3_target_grid",
                    },
                }
            return NativeGenerationResult(
                runtime_key=(
                    f"{spec.engine}:{spec.weight_tier}:"
                    f"{spec.vram_profile}:native-sm89"
                ),
                elapsed_seconds=round(time.monotonic() - started, 3),
                output_path=result.output_path,
                stage_seconds=phases,
                inference_plan=public_plan,
                final_latents_path=(
                    final_latents_path
                    if final_latents_path is not None and final_latents_path.is_file()
                    else None
                ),
            )

    async def close(self) -> None:
        async with self._lock:
            if self._built is not None and self._device_fatal_error is None:
                await asyncio.to_thread(self._built.session.close)
            # A poisoned CUDA context must not be traversed by model teardown.
            # Keep its Python object alive until process exit instead of
            # triggering driver calls from tensor destruction.
            if self._device_fatal_error is None:
                self._built = None
                self._engine_name = None
            def release_host_session_pages() -> None:
                gc.collect()
                try:
                    ctypes.CDLL(None).malloc_trim(0)
                except (AttributeError, OSError):
                    pass
            await asyncio.to_thread(release_host_session_pages)
            if self._device_fatal_error is None:
                self._warm_state = {
                    "status": "cold", "engine": None,
                    "launcher": None, "weight_tier": None, "vram_profile": None,
                    "startup_seconds": None, "error": None,
                }

    @property
    def output_root(self) -> Path:
        return self._output_root

    def set_output_root(self, output_root: Path) -> None:
        if self._built is not None:
            raise RuntimeError("cannot switch workspace while an engine is loaded")
        resolved = output_root.resolve()
        resolved.mkdir(parents=True, exist_ok=True)
        self._factory.set_output_root(resolved)
        self._output_root = resolved
