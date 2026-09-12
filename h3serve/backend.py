"""Backend-neutral Web/API boundary for the in-process Native H3 engine."""

from __future__ import annotations

import asyncio
import dataclasses
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import ServicePaths
from .contract import (
    GenerationSpec,
    SecondSamplingSpec,
    VideoRepairSpec,
    resolve_geometry,
    resolve_engine,
    resolve_launcher,
)
from .infinite_video import InfiniteContinuationSpec


class BackendError(RuntimeError):
    pass


class JobCancelled(BackendError):
    pass


@dataclass(frozen=True)
class GenerationResult:
    runtime_key: str
    elapsed_seconds: float
    output_path: Path
    inference_plan: dict[str, Any] | None = None
    final_latents_path: Path | None = None
    token_memory_path: Path | None = None
    stage_seconds: dict[str, float] | None = None


@dataclass(frozen=True)
class CheckpointResult:
    runtime_key: str
    elapsed_seconds: float
    checkpoint_path: Path | None
    preview_path: Path | None
    completed_steps: int
    total_steps: int
    inference_plan: dict[str, Any] | None = None
    stage_seconds: dict[str, float] | None = None
    # A SelfLift checkpoint owns two products at the fork: the disposable
    # low-resolution preview branch used by the editor/next continuation and
    # the lifted formal checkpoint used by the eventual high-resolution tail.
    preview_latents_path: Path | None = None
    token_memory_path: Path | None = None


class NativeBackendManager:
    """Keep persistence/job identifiers outside the model runtime."""

    def __init__(self, engine: Any) -> None:
        self.engine = engine
        self.key: str | None = None

    def preflight(self, engine: str) -> dict[str, Any]:
        check = getattr(self.engine, "preflight", None)
        if callable(check):
            return check(engine)
        return {"ready": True, "checks": {"native_engine": True}}

    async def preload(self, engine: str) -> None:
        preload = getattr(self.engine, "preload", None)
        if callable(preload):
            await preload(engine)

    @property
    def warm_state(self) -> dict[str, Any]:
        return dict(getattr(self.engine, "warm_state", {"status": "unsupported"}))

    async def generate(
        self,
        spec: GenerationSpec,
        job_id: str,
        first_frame: Path | None,
        last_frame: Path | None,
        reference_images: tuple[Path, ...],
        reference_videos: tuple[Path, ...],
        reference_audios: tuple[Path, ...],
        cancel_event: asyncio.Event,
        progress_callback: Any | None = None,
        preview_ready_callback: Any | None = None,
        preview_decision_wait: Any | None = None,
        checkpoint_path: Path | None = None,
        resume_checkpoint_path: Path | None = None,
        source_video_path: Path | None = None,
    ) -> GenerationResult | CheckpointResult:
        from .native_engine.engine import (
            NativeCheckpointResult,
            NativeGenerationCancelled,
        )

        safe_name = re.sub(r"[^A-Za-z0-9_.-]", "_", job_id).strip("._")
        if not safe_name:
            raise BackendError("job id does not contain a safe output name")
        output_path = self.engine.output_root / f"{safe_name[:128]}.mp4"
        latent_root = self.engine.output_root / ".h3-latents"
        latent_root.mkdir(parents=True, exist_ok=True)
        final_latents_path = latent_root / f"{safe_name[:128]}.pt"
        windowed_selflift_required = False
        if spec.selflift_enabled and spec.selflift_temporal_window_enabled:
            from .native_engine.global_co_denoise import (
                window_geometry_for_seconds,
            )

            preferred_window_frames, _ = window_geometry_for_seconds(
                spec.selflift_temporal_window_seconds,
                spec.selflift_temporal_overlap_seconds,
            )
            windowed_selflift_required = spec.frames > preferred_window_frames
        if (
            spec.selflift_enabled
            and spec.selflift_temporal_window_enabled
            and windowed_selflift_required
            and spec.execution_mode == "complete"
            and resume_checkpoint_path is None
        ):
            return await self._generate_windowed_selflift(
                spec=spec,
                safe_name=safe_name,
                output_path=output_path,
                final_latents_path=final_latents_path,
                first_frame=first_frame,
                last_frame=last_frame,
                reference_images=reference_images,
                reference_videos=reference_videos,
                reference_audios=reference_audios,
                cancel_event=cancel_event,
                progress_callback=progress_callback,
            )
        try:
            result = await self.engine.generate(
                spec, first_frame, last_frame, reference_images, reference_videos, reference_audios, cancel_event, output_path,
                progress_callback=progress_callback,
                preview_ready_callback=preview_ready_callback,
                preview_decision_wait=preview_decision_wait,
                checkpoint_path=checkpoint_path,
                resume_checkpoint_path=resume_checkpoint_path,
                final_latents_path=final_latents_path,
            )
        except NativeGenerationCancelled as error:
            raise JobCancelled(str(error)) from error
        self.key = result.runtime_key
        if isinstance(result, NativeCheckpointResult):
            return CheckpointResult(
                runtime_key=result.runtime_key,
                elapsed_seconds=result.elapsed_seconds,
                checkpoint_path=result.checkpoint_path,
                preview_path=result.preview_path,
                completed_steps=result.completed_steps,
                total_steps=result.total_steps,
                inference_plan=getattr(result, "inference_plan", None),
                stage_seconds=dict(getattr(result, "stage_seconds", {})),
                preview_latents_path=getattr(
                    result, "preview_latents_path", None
                ),
                token_memory_path=getattr(result, "token_memory_path", None),
            )
        return GenerationResult(
            runtime_key=result.runtime_key,
            elapsed_seconds=result.elapsed_seconds,
            output_path=result.output_path,
            inference_plan=getattr(result, "inference_plan", None),
            final_latents_path=getattr(result, "final_latents_path", None),
            stage_seconds=dict(getattr(result, "stage_seconds", {})),
        )

    async def _generate_windowed_selflift(
        self,
        *,
        spec: GenerationSpec,
        safe_name: str,
        output_path: Path,
        final_latents_path: Path,
        first_frame: Path | None,
        last_frame: Path | None,
        reference_images: tuple[Path, ...],
        reference_videos: tuple[Path, ...],
        reference_audios: tuple[Path, ...],
        cancel_event: asyncio.Event,
        progress_callback: Any | None,
    ) -> GenerationResult:
        """Finish a single SelfLift request through global temporal views."""

        from .native_engine.engine import (
            NativeCheckpointResult,
            NativeGenerationCancelled,
        )
        from .native_engine.global_co_denoise import (
            plan_balanced_global_av_windows,
            window_geometry_for_seconds,
        )
        import torch

        started = time.monotonic()
        work_dir = (
            self.engine.output_root
            / ".h3-selflift-single"
            / safe_name[:128]
        )
        shutil.rmtree(work_dir, ignore_errors=True)
        work_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = work_dir / "split-checkpoint.pt"
        source_path = work_dir / "source-x0.pt"
        global_source_path = work_dir / "global-source-x0.pt"
        prefix_output = work_dir / "prefix-placeholder.mp4"
        split_step = int(spec.selflift_transition_step or 0)
        if not 1 <= split_step < int(spec.sampling_steps or 0):
            raise BackendError("windowed SelfLift requires a valid split step")

        def mapped_progress(offset: float, span: float):
            if progress_callback is None:
                return None

            def report(event: dict[str, Any]) -> None:
                local = float(event.get("percent", 0.0)) / 100.0
                progress_callback({
                    **event,
                    "percent": offset + span * local,
                })

            return report

        try:
            prefix_width, prefix_height = resolve_geometry(
                spec.selflift_initial_resolution,
                spec.aspect_ratio,
            )
            prefix_spec = dataclasses.replace(
                spec,
                resolution=spec.selflift_initial_resolution,
                width=prefix_width,
                height=prefix_height,
                execution_mode="checkpoint",
                checkpoint_step=split_step,
                checkpoint_retain=True,
                checkpoint_preview=False,
                preview_mode="off",
                preview_step_index=None,
                preview_fast_finish=False,
                selflift_temporal_window_enabled=False,
            )
            prefix = await self.engine.generate(
                prefix_spec,
                first_frame,
                last_frame,
                reference_images,
                reference_videos,
                reference_audios,
                cancel_event,
                prefix_output,
                progress_callback=mapped_progress(0.0, 42.0),
                checkpoint_path=checkpoint_path,
                final_latents_path=source_path,
            )
            if not isinstance(prefix, NativeCheckpointResult):
                raise BackendError("SelfLift prefix did not produce a checkpoint")
            if not checkpoint_path.is_file() or not source_path.is_file():
                raise BackendError("SelfLift prefix artifacts are missing")

            checkpoint = await asyncio.to_thread(
                torch.load, checkpoint_path, map_location="cpu", weights_only=True
            )
            source = await asyncio.to_thread(
                torch.load, source_path, map_location="cpu", weights_only=True
            )
            video = checkpoint.get("selflift_source_video_x0")
            audio = source.get("audio")
            if (
                not isinstance(video, torch.Tensor)
                or video.ndim != 5
                or not isinstance(audio, torch.Tensor)
                or audio.ndim != 4
                or not bool(source.get("audio_final", False))
            ):
                raise BackendError(
                    "SelfLift split is missing its completed low-resolution AV latents"
                )
            sigmas = tuple(float(value) for value in checkpoint.get("sigmas", ()))
            total_steps = int(checkpoint.get("steps", -1))
            if total_steps != int(spec.sampling_steps or 0) or len(sigmas) != total_steps + 1:
                raise BackendError("SelfLift split schedule is incomplete")
            global_document = {
                "video": video,
                "audio": audio,
                # Audio completed the low-resolution suffix once. Global
                # spatial windows retain it verbatim while refining video.
                "audio_final": True,
                "frames": spec.frames,
                "fps": 24,
                "width": int(checkpoint.get(
                    "selflift_source_width", video.shape[-1] * 16
                )),
                "height": int(checkpoint.get(
                    "selflift_source_height", video.shape[-2] * 16
                )),
                "engine": source.get("engine"),
                "seed": spec.seed,
                "steps": total_steps,
                "next_step_index": split_step,
                "sigmas": list(sigmas),
                "representation": "global_selflift_clean_source_x0_v2",
                "audio_representation": "completed_low_resolution_tail_x0_v1",
            }
            await asyncio.to_thread(torch.save, global_document, global_source_path)
            window_frames, stride_frames = window_geometry_for_seconds(
                spec.selflift_temporal_window_seconds,
                spec.selflift_temporal_overlap_seconds,
            )
            global_plan = plan_balanced_global_av_windows(
                spec.frames,
                window_frames=window_frames,
                stride_frames=stride_frames,
            )
            final_spec = dataclasses.replace(
                spec,
                output_frames=spec.frames,
                execution_mode="complete",
                checkpoint_step=None,
                checkpoint_preview=False,
                preview_mode="off",
                preview_step_index=None,
            )
            generated = await self.engine.generate(
                final_spec,
                first_frame,
                last_frame,
                reference_images,
                reference_videos,
                reference_audios,
                cancel_event,
                output_path,
                progress_callback=mapped_progress(42.0, 58.0),
                final_latents_path=final_latents_path,
                global_selflift_source_path=global_source_path,
                global_selflift_prompts=tuple(
                    spec.prompt for _ in global_plan.windows
                ),
            )
        except NativeGenerationCancelled as error:
            raise JobCancelled(str(error)) from error
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)

        stage_seconds = {
            **{
                f"selflift_prefix.{name}": seconds
                for name, seconds in prefix.stage_seconds.items()
            },
            **{
                f"selflift_windowed_tail.{name}": seconds
                for name, seconds in generated.stage_seconds.items()
            },
        }
        inference_plan = dict(generated.inference_plan or {})
        inference_plan["single_selflift_temporal_window"] = {
            "enabled": True,
            "requested_seconds": spec.selflift_temporal_window_seconds,
            "requested_overlap_seconds": spec.selflift_temporal_overlap_seconds,
            "effective_overlap_seconds": (
                global_plan.window_frames - global_plan.stride_frames
            ) / 24.0,
            "shared_low_resolution_prefix": True,
            "intermediate_decode": False,
            "single_final_decode": True,
            "global_temporal_views": global_plan.telemetry(),
            "prefix": prefix.inference_plan,
        }
        self.key = generated.runtime_key
        return GenerationResult(
            runtime_key=generated.runtime_key,
            elapsed_seconds=round(time.monotonic() - started, 3),
            output_path=generated.output_path,
            inference_plan=inference_plan,
            final_latents_path=generated.final_latents_path,
            stage_seconds=stage_seconds,
        )

    async def second_sample(
        self,
        spec: GenerationSpec,
        second_sampling: SecondSamplingSpec,
        source_latents_path: Path,
        job_id: str,
        first_frame: Path | None,
        last_frame: Path | None,
        reference_images: tuple[Path, ...],
        reference_videos: tuple[Path, ...],
        reference_audios: tuple[Path, ...],
        cancel_event: asyncio.Event,
        progress_callback: Any | None = None,
    ) -> GenerationResult:
        from .native_engine.engine import NativeGenerationCancelled

        safe_name = re.sub(r"[^A-Za-z0-9_.-]", "_", job_id).strip("._")
        if not safe_name:
            raise BackendError("job id does not contain a safe output name")
        source_latents_path = source_latents_path.resolve()
        latent_root = (self.engine.output_root / ".h3-latents").resolve()
        if not source_latents_path.is_relative_to(latent_root):
            raise BackendError("second-sampling source is outside the latent store")
        if not source_latents_path.is_file():
            raise BackendError("second-sampling source latent is missing")
        output_path = self.engine.output_root / f"{safe_name[:128]}.mp4"
        final_latents_path = latent_root / f"{safe_name[:128]}.pt"
        try:
            result = await self.engine.generate(
                spec,
                first_frame,
                last_frame,
                reference_images,
                reference_videos,
                reference_audios,
                cancel_event,
                output_path,
                progress_callback=progress_callback,
                final_latents_path=final_latents_path,
                second_sampling=second_sampling,
                refinement_latents_path=source_latents_path,
            )
        except NativeGenerationCancelled as error:
            raise JobCancelled(str(error)) from error
        self.key = result.runtime_key
        return GenerationResult(
            runtime_key=result.runtime_key,
            elapsed_seconds=result.elapsed_seconds,
            output_path=result.output_path,
            inference_plan=getattr(result, "inference_plan", None),
            final_latents_path=getattr(result, "final_latents_path", None),
            stage_seconds=dict(getattr(result, "stage_seconds", {})),
        )

    async def video_repair(
        self,
        spec: GenerationSpec,
        repair: VideoRepairSpec,
        source_video_path: Path,
        job_id: str,
        cancel_event: asyncio.Event,
        progress_callback: Any | None = None,
    ) -> GenerationResult:
        """Repair several pixel-video regions per H3 Turbo atlas pass."""

        if spec.service_family != "first_last":
            raise BackendError(
                "face repair is available only for FL2VA source jobs"
            )

        from .native_engine.engine import NativeGenerationCancelled
        from .video_repair import (
            atlas_denoise_regions,
            assemble_repaired_video,
            cleanup_repair_workdir,
            merge_refined_atlas,
            prepare_repair_windows,
        )

        safe_name = re.sub(r"[^A-Za-z0-9_.-]", "_", job_id).strip("._")
        if not safe_name:
            raise BackendError("job id does not contain a safe output name")
        source_video_path = Path(source_video_path).resolve()
        if not source_video_path.is_file():
            raise BackendError("video-repair source video is missing")
        output_path = self.engine.output_root / f"{safe_name[:128]}.mp4"
        work_dir = self.engine.output_root / ".video-repair-work" / safe_name[:128]
        started = time.monotonic()
        stage_seconds: dict[str, float] = {}
        window_receipts: list[dict[str, Any]] = []
        factory = getattr(self.engine, "_factory", None)
        previous_lora = getattr(factory, "lora_checkpoint", None)
        previous_launcher = getattr(self.engine, "_engine_name", None)
        repair_launcher = resolve_launcher(
            "first_last", spec.weight_tier, spec.vram_profile
        )
        repair_lora = (
            Path(factory.paths.model_root)
            / "loras/lightx2v/minimax_h3_fl2v_turbo_4step_v1.1_768p_bf16.safetensors"
            if factory is not None and hasattr(factory, "paths")
            else None
        )
        switched_lora = bool(
            repair_lora is not None
            and repair_lora.is_file()
            and previous_lora is not None
            and Path(previous_lora).resolve() != repair_lora.resolve()
        )
        switched_launcher = bool(
            previous_launcher is not None
            and previous_launcher != repair_launcher
        )

        def progress(percent: float, stage: str, detail: str) -> None:
            if progress_callback is not None:
                progress_callback({
                    "percent": percent, "stage": stage, "detail": detail,
                })

        try:
            if repair_lora is None or not repair_lora.is_file():
                raise BackendError(
                    "ComfyUI-H3-FaceRefine-Accelerated requires the installed "
                    "LightX2V H3 FL2V 4-step LoRA"
                )
            if switched_lora or switched_launcher:
                progress(1, "video_repair_model", "切换到 LightX2V H3 四步修复模型")
                await self.engine.close()
            if switched_lora:
                factory.set_lora_checkpoint(repair_lora)
            stage = time.monotonic()
            meta, prepared = await asyncio.to_thread(
                prepare_repair_windows,
                source_video_path,
                work_dir,
                repair,
                progress=progress,
            )
            stage_seconds["video_repair.detect_and_pack"] = round(
                time.monotonic() - stage, 3
            )
            last_runtime_key = (
                f"{resolve_engine(spec.service_family, 'lora')}:"
                f"{spec.weight_tier}:{spec.vram_profile}:native-sm89"
            )
            last_latent_by_chain: dict[str, Path] = {}
            for index, item in enumerate(prepared):
                if cancel_event.is_set():
                    raise JobCancelled("cancelled during video repair")
                progress(
                    24.0 + 62.0 * index / len(prepared),
                    "video_repair_h3",
                    f"H3修复窗口 {index + 1}/{len(prepared)}",
                )
                atlas_spec = dataclasses.replace(
                    spec,
                    prompt=(
                        "Live-action source-preserving local video restoration. "
                        "Keep identity, geometry, motion, lighting and color exactly; "
                        "recover only plausible fine detail. No text or new objects."
                    ),
                    engine=resolve_engine("first_last", "lora"),
                    resolution=f"{item.canvas_size}p",
                    width=item.canvas_size,
                    height=item.canvas_size,
                    frames=item.window.padded_frames,
                    output_frames=None,
                    requested_duration_seconds=item.window.padded_frames / 24.0,
                    actual_duration_seconds=item.window.padded_frames / 24.0,
                    advanced=True,
                    custom_actual_steps=None,
                    custom_lora_steps=repair.steps,
                    sampling_steps=None,
                    acceleration=None,
                    second_pass_acceleration=None,
                    acceleration_transition_step=None,
                    selflift_enabled=False,
                    selflift_transition_step=None,
                    preview_mode="off",
                    preview_step_index=None,
                    execution_mode="complete",
                )
                second = SecondSamplingSpec(
                    resolution=f"{item.canvas_size}p",
                    width=item.canvas_size,
                    height=item.canvas_size,
                    method="h3",
                    steps=repair.steps,
                    acceleration=repair.acceleration,
                    denoise=repair.denoise,
                    strength="auto",
                    model_variant="lora",
                    memory_mode="auto",
                    spatial_mode="strict",
                    preserve_audio=False,
                    # This source is already a bounded repair window. Avoid a
                    # second nested temporal split inside the Atlas pass.
                    temporal_window_frames=None,
                )
                stage = time.monotonic()
                prior_latent = last_latent_by_chain.get(item.chain_id)
                try:
                    generated = await self.engine.generate(
                        atlas_spec,
                        None,
                        None,
                        (),
                        (),
                        (),
                        cancel_event,
                        item.refined_atlas_path,
                        progress_callback=None,
                        final_latents_path=item.refined_latents_path,
                        second_sampling=second,
                        refinement_latents_path=None,
                        external_refinement_video_path=item.atlas_path,
                        refinement_schedule_mode="comfy_simple_tail",
                        refinement_atlas_denoise_regions=(
                            atlas_denoise_regions(item)
                        ),
                        refinement_handoff_latents_path=(
                            prior_latent if item.window.context_frames else None
                        ),
                        refinement_handoff_context_frames=(
                            item.window.context_frames if prior_latent is not None else 0
                        ),
                        conditioning_cache_source_path=prior_latent,
                    )
                except NativeGenerationCancelled as error:
                    raise JobCancelled(str(error)) from error
                h3_seconds = time.monotonic() - stage
                stage_seconds[f"video_repair.window_{index:03d}.h3"] = round(
                    h3_seconds, 3
                )
                last_runtime_key = generated.runtime_key
                if item.refined_latents_path is not None:
                    last_latent_by_chain[item.chain_id] = item.refined_latents_path
                stage = time.monotonic()
                gate = await asyncio.to_thread(merge_refined_atlas, item)
                stage_seconds[f"video_repair.window_{index:03d}.merge"] = round(
                    time.monotonic() - stage, 3
                )
                window_receipts.append({
                    "atlas_pass_index": index,
                    "temporal_window_index": item.window.index,
                    "atlas_batch_index": item.batch_index,
                    "start_frame": item.window.start,
                    "end_frame": item.window.end,
                    "padded_frames": item.window.padded_frames,
                    "atlas_canvas_size": item.canvas_size,
                    "atlas_grid_size": item.grid_size,
                    **gate,
                })
            progress(90, "video_repair_merge", "融合时间窗口并恢复源音频")
            stage = time.monotonic()
            await asyncio.to_thread(
                assemble_repaired_video,
                source_video_path,
                output_path,
                prepared,
                meta,
            )
            stage_seconds["video_repair.final_mux"] = round(
                time.monotonic() - stage, 3
            )
            elapsed = time.monotonic() - started
            return GenerationResult(
                runtime_key=last_runtime_key,
                elapsed_seconds=round(elapsed, 3),
                output_path=output_path,
                inference_plan={
                    "video_repair": {
                        **repair.to_dict(),
                        "implementation": "comfyui_h3_facerefine_accelerated_native_v1",
                        "upstream_project": "ComfyUI-H3-FaceRefine-Accelerated",
                        "repair_lora": "lightx2v/minimax_h3_fl2v_turbo_4step_v1.1_768p_bf16.safetensors",
                        "source_service_family": spec.service_family,
                        "repair_service_family": "first_last",
                        "schedule": "BasicScheduler simple, 4-step tail, denoise 0.55",
                        "crop_factor": 2.5,
                        "pixel_handoff": "FFV1 lossless",
                        "progressive_latent_handoff": True,
                        "source": str(source_video_path),
                        "output_resolution": [meta["width"], meta["height"]],
                        "selected_face_count": meta.get("selected_face_count", 0),
                        "atlas_capacity": meta.get("face_repair_capacity", repair.max_faces),
                        "atlas_cell_size": meta.get("face_repair_cell_size", repair.cell_size),
                        "temporal_window_count": len({
                            item.window.index for item in prepared
                        }),
                        "atlas_pass_count": len(prepared),
                        "windows": window_receipts,
                    }
                },
                final_latents_path=None,
                stage_seconds=stage_seconds,
            )
        finally:
            cleanup_repair_workdir(work_dir)
            if switched_lora or switched_launcher:
                await self.engine.close()
                if switched_lora and previous_lora is not None:
                    factory.set_lora_checkpoint(Path(previous_lora))
                if previous_launcher:
                    await self.engine.preload(previous_launcher)

    async def continue_generate(
        self,
        spec: GenerationSpec,
        continuation: InfiniteContinuationSpec,
        source_latents_path: Path,
        source_memory_path: Path | None,
        job_id: str,
        reference_images: tuple[Path, ...],
        reference_audios: tuple[Path, ...],
        cancel_event: asyncio.Event,
        progress_callback: Any | None = None,
        preview_ready_callback: Any | None = None,
        preview_decision_wait: Any | None = None,
        checkpoint_path: Path | None = None,
        resume_checkpoint_path: Path | None = None,
        source_video_path: Path | None = None,
        first_frame: Path | None = None,
        last_frame: Path | None = None,
    ) -> GenerationResult | CheckpointResult:
        """Append one physical window to a completed infinite project tail."""

        from .native_engine.engine import (
            NativeCheckpointResult,
            NativeGenerationCancelled,
        )

        safe_name = re.sub(r"[^A-Za-z0-9_.-]", "_", job_id).strip("._")
        if not safe_name:
            raise BackendError("job id does not contain a safe output name")
        latent_root = (self.engine.output_root / ".h3-latents").resolve()
        source_latents_path = Path(source_latents_path).resolve()
        if not source_latents_path.is_relative_to(latent_root) or not source_latents_path.is_file():
            raise BackendError("infinite continuation source latent is unavailable")
        if source_memory_path is not None:
            source_memory_path = Path(source_memory_path).resolve()
            if not source_memory_path.is_relative_to(latent_root) or not source_memory_path.is_file():
                raise BackendError("infinite continuation memory is unavailable")
        output_path = self.engine.output_root / f"{safe_name[:128]}.mp4"
        final_latents_path = latent_root / f"{safe_name[:128]}.pt"
        token_memory_path = latent_root / f"{safe_name[:128]}.memory.pt"
        try:
            result = await self.engine.generate(
                spec,
                first_frame,
                last_frame,
                reference_images,
                (),
                reference_audios,
                cancel_event,
                output_path,
                progress_callback=progress_callback,
                preview_ready_callback=preview_ready_callback,
                preview_decision_wait=preview_decision_wait,
                checkpoint_path=checkpoint_path,
                resume_checkpoint_path=resume_checkpoint_path,
                final_latents_path=final_latents_path,
                continuation=continuation,
                continuation_source_latents_path=source_latents_path,
                continuation_source_video_path=source_video_path,
                continuation_source_memory_path=source_memory_path,
                continuation_output_memory_path=token_memory_path,
            )
        except NativeGenerationCancelled as error:
            raise JobCancelled(str(error)) from error
        self.key = result.runtime_key
        if isinstance(result, NativeCheckpointResult):
            return CheckpointResult(
                runtime_key=result.runtime_key,
                elapsed_seconds=result.elapsed_seconds,
                checkpoint_path=result.checkpoint_path,
                preview_path=result.preview_path,
                completed_steps=result.completed_steps,
                total_steps=result.total_steps,
                inference_plan=getattr(result, "inference_plan", None),
                stage_seconds=dict(getattr(result, "stage_seconds", {})),
                preview_latents_path=getattr(
                    result, "preview_latents_path", None
                ),
                token_memory_path=getattr(result, "token_memory_path", None),
            )
        return GenerationResult(
            runtime_key=result.runtime_key,
            elapsed_seconds=result.elapsed_seconds,
            output_path=result.output_path,
            inference_plan=getattr(result, "inference_plan", None),
            final_latents_path=getattr(result, "final_latents_path", None),
            token_memory_path=(token_memory_path if token_memory_path.is_file() else None),
            stage_seconds=dict(getattr(result, "stage_seconds", {})),
        )

    async def complete_infinite_selflift(
        self,
        sources: tuple[Any, ...],
        job_id: str,
        cancel_event: asyncio.Event,
        progress_callback: Any | None = None,
        final_spec: GenerationSpec | None = None,
    ) -> GenerationResult:
        """Finish one connected low-res project through global SelfLift views.

        Every retained fork contributes its exact clean source-grid x0 at the
        split.  Those pieces are first assembled into one timeline.  Learned
        spatial lifting then happens once over that complete timeline and the
        remaining formal H3 steps share one global solver state through
        overlapping temporal views.  No independently completed 1080p clips
        are concatenated at the end.
        """

        from .native_engine.engine import NativeGenerationCancelled
        from .native_engine.global_co_denoise import (
            plan_prompt_owned_global_av_windows,
        )
        from .native_engine.long_horizon import (
            stitch_audio_segment_files,
            stitch_clean_av_segment_files,
        )
        import torch

        safe_name = re.sub(r"[^A-Za-z0-9_.-]", "_", job_id).strip("._")
        if not safe_name:
            raise BackendError("job id does not contain a safe output name")
        if not sources:
            raise BackendError("SelfLift final generation has no source windows")
        output_root = self.engine.output_root.resolve()
        latent_root = (output_root / ".h3-latents").resolve()
        latent_root.mkdir(parents=True, exist_ok=True)
        work_dir = output_root / ".h3-selflift-final" / safe_name[:128]
        shutil.rmtree(work_dir, ignore_errors=True)
        work_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_root / f"{safe_name[:128]}.mp4"
        final_latents_path = latent_root / f"{safe_name[:128]}.pt"
        source_x0_segments: list[Path] = []
        contexts: list[int] = []
        bridge_ticks: list[int] = []
        stage_seconds: dict[str, float] = {}
        started = time.monotonic()

        try:
            for index, source in enumerate(sources):
                if cancel_event.is_set():
                    raise JobCancelled("cancelled during SelfLift final generation")
                checkpoint = source.checkpoint_path
                if checkpoint is None or not Path(checkpoint).is_file():
                    raise BackendError(
                        f"SelfLift fork checkpoint is missing for window {index + 1}"
                    )
                document = await asyncio.to_thread(
                    torch.load,
                    Path(checkpoint),
                    map_location="cpu",
                    weights_only=True,
                )
                clean_video = document.get("selflift_source_video_x0")
                clean_audio = document.get("selflift_source_audio_x0")
                formal_audio_state = document.get("audio")
                source_width = document.get("selflift_source_width")
                source_height = document.get("selflift_source_height")
                if (
                    not isinstance(clean_video, torch.Tensor)
                    or clean_video.ndim != 5
                    or not isinstance(clean_audio, torch.Tensor)
                    or clean_audio.ndim != 4
                    or not isinstance(formal_audio_state, torch.Tensor)
                    or tuple(formal_audio_state.shape) != tuple(clean_audio.shape)
                    or not isinstance(source_width, int)
                    or not isinstance(source_height, int)
                ):
                    raise BackendError(
                        "SelfLift fork predates global source-grid finalization; "
                        f"regenerate preview window {index + 1}"
                    )
                segment_path = work_dir / f"source-x0-{index:03d}.pt"
                await asyncio.to_thread(
                    torch.save,
                    {
                        "video": clean_video,
                        "audio": clean_audio,
                        "audio_state": formal_audio_state,
                        "frames": source.spec.frames,
                        "fps": 24,
                        "width": source_width,
                        "height": source_height,
                        "engine": document.get("engine"),
                        "seed": source.spec.seed,
                        "representation": "selflift_window_source_x0_v1",
                    },
                    segment_path,
                )
                source_x0_segments.append(segment_path)
                continuation = source.infinite_continuation
                contexts.append(
                    0 if continuation is None else continuation.hidden_prefix_frames
                )
                bridge_ticks.append(
                    0 if continuation is None else continuation.audio_trim_ticks
                )

            tail = sources[-1]
            expected_frames = int(
                tail.infinite_continuation.output_frames
                if tail.infinite_continuation is not None
                else tail.spec.frames
            )
            stitch_started = time.monotonic()
            (
                stitched_video,
                stitched_audio,
                stitched_frames,
                stitched_engine,
            ) = await asyncio.to_thread(
                stitch_clean_av_segment_files,
                tuple(source_x0_segments),
                tuple(contexts),
                expected_frames=expected_frames,
                audio_bridge_ticks=tuple(bridge_ticks),
            )
            # Every new SelfLift checkpoint finishes its low-resolution audio
            # suffix exactly once. Online creation also decodes that branch;
            # JSON creation stores only the latent. The global spatial pass
            # uses the completed cumulative audio in both modes.
            accepted_audio_path = getattr(tail, "final_latents_path", None)
            accepted_document = None
            if accepted_audio_path is not None and Path(accepted_audio_path).is_file():
                accepted_document = await asyncio.to_thread(
                    torch.load,
                    Path(accepted_audio_path),
                    map_location="cpu",
                    weights_only=True,
                )
            completed_source_audio = bool(
                isinstance(accepted_document, dict)
                and accepted_document.get("audio_final", False)
            )
            stitched_audio_state = None
            if completed_source_audio:
                accepted_audio = accepted_document.get("audio")
                if (
                    accepted_document.get("frames") != stitched_frames
                    or accepted_document.get("fps") != 24
                    or not isinstance(accepted_audio, torch.Tensor)
                    or tuple(accepted_audio.shape) != tuple(stitched_audio.shape)
                ):
                    raise BackendError(
                        "SelfLift completed audio does not match the final timeline"
                    )
                stitched_audio = accepted_audio
            else:
                stitched_audio_state = await asyncio.to_thread(
                    stitch_audio_segment_files,
                    tuple(source_x0_segments),
                    tuple(contexts),
                    expected_frames=expected_frames,
                    audio_key="audio_state",
                    audio_bridge_ticks=tuple(bridge_ticks),
                )
            first_checkpoint = await asyncio.to_thread(
                torch.load,
                Path(sources[0].checkpoint_path),
                map_location="cpu",
                weights_only=True,
            )
            split_step = int(first_checkpoint.get("selflift_split_step", -1))
            total_steps = int(first_checkpoint.get("steps", -1))
            sigmas = tuple(float(value) for value in first_checkpoint.get("sigmas", ()))
            if not 1 <= split_step < total_steps or len(sigmas) != total_steps + 1:
                raise BackendError("SelfLift fork has an invalid formal split schedule")
            source_width = int(first_checkpoint["selflift_source_width"])
            source_height = int(first_checkpoint["selflift_source_height"])
            for index, source in enumerate(sources[1:], start=2):
                checkpoint_document = await asyncio.to_thread(
                    torch.load,
                    Path(source.checkpoint_path),
                    map_location="cpu",
                    weights_only=True,
                )
                schedule = tuple(
                    float(value) for value in checkpoint_document.get("sigmas", ())
                )
                if (
                    int(checkpoint_document.get("selflift_split_step", -1)) != split_step
                    or int(checkpoint_document.get("steps", -1)) != total_steps
                    or schedule != sigmas
                    or int(checkpoint_document.get("selflift_source_width", -1))
                    != source_width
                    or int(checkpoint_document.get("selflift_source_height", -1))
                    != source_height
                ):
                    raise BackendError(
                        f"SelfLift fork schedule or source geometry changed at window {index}"
                    )
            global_source_path = work_dir / "global-source-x0.pt"
            global_document = {
                "video": stitched_video,
                "audio": stitched_audio,
                "audio_final": completed_source_audio,
                "frames": stitched_frames,
                "fps": 24,
                "width": source_width,
                "height": source_height,
                "engine": stitched_engine,
                "seed": sources[0].spec.seed,
                "steps": total_steps,
                "next_step_index": split_step,
                "sigmas": list(sigmas),
                "representation": "global_selflift_clean_source_x0_v2",
                "audio_representation": (
                    "completed_low_resolution_tail_x0_v1"
                    if completed_source_audio
                    else "formal_sampler_state_at_split_v1"
                ),
            }
            if stitched_audio_state is not None:
                global_document["audio_state"] = stitched_audio_state
            await asyncio.to_thread(torch.save, global_document, global_source_path)
            stage_seconds["global_selflift.source_x0_stitch"] = round(
                time.monotonic() - stitch_started, 3
            )

            visible_ranges: list[tuple[int, int, str]] = []
            for source in sources:
                continuation = source.infinite_continuation
                if continuation is None:
                    start_frame = 0
                    stop_frame = source.spec.frames
                else:
                    start_frame = continuation.source_frames
                    stop_frame = continuation.output_frames
                visible_ranges.append(
                    (start_frame, stop_frame, source.spec.prompt)
                )
            prompt_ranges = tuple(
                (start, stop) for start, stop, _prompt in visible_ranges
            )
            if tail.spec.selflift_temporal_window_enabled:
                from .native_engine.global_co_denoise import (
                    window_geometry_for_seconds,
                )

                window_frames, stride_frames = window_geometry_for_seconds(
                    tail.spec.selflift_temporal_window_seconds,
                    tail.spec.selflift_temporal_overlap_seconds,
                )
                global_plan = plan_prompt_owned_global_av_windows(
                    stitched_frames,
                    prompt_ranges,
                    window_frames=window_frames,
                    stride_frames=stride_frames,
                    balanced=True,
                )
            else:
                global_plan = plan_prompt_owned_global_av_windows(
                    stitched_frames,
                    prompt_ranges,
                )
            global_prompts = [
                visible_ranges[window.prompt_index][2]
                for window in global_plan.windows
            ]

            requested_final_spec = tail.spec if final_spec is None else final_spec
            if (
                requested_final_spec.sampling_steps != total_steps
                or requested_final_spec.selflift_transition_step != split_step
            ):
                raise BackendError(
                    "SelfLift final sampling schedule does not match its retained forks"
                )
            target_spec = dataclasses.replace(
                requested_final_spec,
                prompt=global_prompts[0],
                output_frames=stitched_frames,
                requested_duration_seconds=stitched_frames / 24.0,
                actual_duration_seconds=stitched_frames / 24.0,
                preview_mode="off",
                preview_step_index=None,
                preview_fast_finish=False,
                execution_mode="complete",
                checkpoint_step=None,
                checkpoint_preview=False,
            )
            if progress_callback is not None:
                progress_callback({
                    "percent": 4.0,
                    "stage": "global_selflift",
                    "detail": "正在沿全片低分辨率 latent 轨道滑窗终采",
                })
            try:
                result = await self.engine.generate(
                    target_spec,
                    None,
                    None,
                    sources[0].reference_images,
                    (),
                    sources[0].reference_audios,
                    cancel_event,
                    output_path,
                    progress_callback=progress_callback,
                    final_latents_path=final_latents_path,
                    global_selflift_source_path=global_source_path,
                    global_selflift_prompts=tuple(global_prompts),
                    global_selflift_prompt_ranges=prompt_ranges,
                )
            except NativeGenerationCancelled as error:
                raise JobCancelled(str(error)) from error
            if not output_path.is_file() or not final_latents_path.is_file():
                raise BackendError("global SelfLift final artifacts were not created")
            stage_seconds.update(dict(result.stage_seconds))
            balance_started = time.monotonic()
            from .native_engine.audio_window_balance import (
                balance_encoded_creator_windows,
            )

            audio_balance_profile = await asyncio.to_thread(
                balance_encoded_creator_windows,
                output_path,
                prompt_ranges,
                fps=24,
            )
            stage_seconds["global_selflift.audio_window_balance"] = round(
                time.monotonic() - balance_started, 3
            )
            return GenerationResult(
                runtime_key=result.runtime_key,
                elapsed_seconds=round(time.monotonic() - started, 3),
                output_path=output_path,
                final_latents_path=final_latents_path,
                stage_seconds=stage_seconds,
                inference_plan={
                    "infinite_selflift": {
                        "schema_version": "global_sliding_selflift_v1",
                        "window_count": len(sources),
                        "shared_low_resolution_prefix": True,
                        "preview_branch": "isolated_low_resolution_finish",
                        "formal_branch": "global_source_x0_lift_then_overlapping_co_denoise",
                        "hidden_context_frames": contexts,
                        "output_frames": stitched_frames,
                        "completed_low_resolution_steps": split_step,
                        "remaining_high_resolution_steps": total_steps - split_step,
                        "creator_window_audio_balance": audio_balance_profile,
                        "global_temporal_views": global_plan.telemetry(),
                        "temporal_window_setting": {
                            "enabled": bool(
                                tail.spec.selflift_temporal_window_enabled
                            ),
                            "requested_seconds": float(
                                tail.spec.selflift_temporal_window_seconds
                            ),
                            "requested_overlap_seconds": float(
                                tail.spec.selflift_temporal_overlap_seconds
                            ),
                            "effective_overlap_seconds": (
                                global_plan.window_frames
                                - global_plan.stride_frames
                            ) / 24.0,
                        },
                        "runtime": result.inference_plan,
                    }
                },
            )
        finally:
            # The stitched latent and final MP4 are durable products. Window
            # pieces are private resumable-work artifacts.
            shutil.rmtree(work_dir, ignore_errors=True)

    async def stop(self) -> None:
        await self.engine.close()
        self.key = None

    def configure_memory_profile(self, profile: Any) -> None:
        factory = getattr(self.engine, "_factory", None)
        configure = getattr(factory, "set_memory_profile", None)
        if not callable(configure):
            raise BackendError("native backend does not support memory-profile changes")
        configure(profile)

    def configure_lora_checkpoint(self, checkpoint: Path) -> None:
        factory = getattr(self.engine, "_factory", None)
        configure = getattr(factory, "set_lora_checkpoint", None)
        if not callable(configure):
            raise BackendError("native backend does not support LoRA changes")
        configure(checkpoint)

    def configure_output_root(self, output_root: Path) -> None:
        configure = getattr(self.engine, "set_output_root", None)
        if not callable(configure):
            raise BackendError("native backend does not support workspace changes")
        configure(output_root)


def build_native_backend(paths: ServicePaths, *, memory_profile: Any = None) -> NativeBackendManager:
    """Construct the production backend without loading large weights yet."""

    from .native_engine import NativeHotH3Engine
    from .native_engine.session_factory import NativeSessionFactory, NativeSessionPaths

    factory = NativeSessionFactory(
        NativeSessionPaths(
            model_root=paths.model_dir,
            minimax_source=paths.minimax_source_dir,
            lightx_source=paths.lightx_source_dir,
            turbo_curve=paths.turbo_curve_path,
            output_root=paths.output_dir,
        ),
        memory_profile=memory_profile,
    )
    return NativeBackendManager(
        NativeHotH3Engine(factory, output_root=paths.output_dir)
    )


__all__ = [
    "BackendError",
    "CheckpointResult",
    "GenerationResult",
    "JobCancelled",
    "NativeBackendManager",
    "build_native_backend",
]
