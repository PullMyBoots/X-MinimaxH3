from __future__ import annotations

import asyncio
import json
import shutil
import tempfile
import time
import unittest
import io
import wave
from pathlib import Path
import aiohttp
import av
import numpy as np
from aiohttp.test_utils import AioHTTPTestCase
from PIL import Image

from h3serve.app import (
    JobRecord, JobService, create_app,
)
from h3serve.backend import CheckpointResult, GenerationResult, JobCancelled
from h3serve.config import ServicePaths
from h3serve.contract import GenerationSpec, SecondSamplingSpec, VideoRepairSpec
from h3serve.memory_policy import HOST_MEMORY_PROFILES


class FakeBackend:
    def __init__(self, video_path: Path) -> None:
        self.video_path = video_path
        self.key: str | None = None
        self.preloaded: str | None = None
        self.warm_state = {"status": "cold", "engine": None}
        self.reference_images: tuple[Path, ...] = ()
        self.reference_videos: tuple[Path, ...] = ()
        self.reference_audios: tuple[Path, ...] = ()
        self.last_spec = None
        self.lora_checkpoint: Path | None = None
        self.fail_lora_checkpoint: str | None = None

    async def preload(self, engine: str) -> None:
        self.preloaded = engine
        if (
            self.lora_checkpoint is not None
            and self.lora_checkpoint.name == self.fail_lora_checkpoint
        ):
            self.warm_state = {"status": "failed", "engine": engine}
            return
        self.warm_state = {
            "status": "ready", "engine": engine,
            "lora_checkpoint": (
                self.lora_checkpoint.name if self.lora_checkpoint else None
            ),
        }

    def configure_lora_checkpoint(self, checkpoint: Path) -> None:
        self.lora_checkpoint = Path(checkpoint)

    async def generate(
        self, spec, _job_id: str, _first_frame: Path | None,
        _last_frame: Path | None, _reference_images: tuple[Path, ...],
        _reference_videos: tuple[Path, ...], _reference_audios: tuple[Path, ...], cancel_event: asyncio.Event,
        progress_callback=None, **preview_callbacks,
    ) -> GenerationResult:
        self.last_spec = spec
        self.reference_images = _reference_images
        self.reference_videos = _reference_videos
        self.reference_audios = _reference_audios
        self.key = (
            "reference:native-sm89" if spec.engine == "reference" else
            "original:balanced" if spec.engine == "original" else "turbo:shared"
        )
        if cancel_event.is_set():
            raise JobCancelled("cancelled")
        if progress_callback is not None:
            progress_callback({"percent": 50, "stage": "denoise", "detail": "1/2"})
        checkpoint_path = preview_callbacks.get("checkpoint_path")
        if checkpoint_path is not None:
            checkpoint_path = Path(checkpoint_path)
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            checkpoint_path.write_bytes(b"formal-checkpoint")
            preview_path = None
            if spec.checkpoint_preview:
                preview_path = self.video_path.with_name("checkpoint-preview.mp4")
                preview_path.write_bytes(b"checkpoint-preview")
            preview_latents_path = None
            if spec.selflift_enabled:
                preview_latents_path = checkpoint_path.with_name(
                    checkpoint_path.stem + ".preview.pt"
                )
                preview_latents_path.write_bytes(b"selflift-low-resolution-preview")
            return CheckpointResult(
                runtime_key=self.key,
                elapsed_seconds=0.5,
                checkpoint_path=checkpoint_path,
                preview_path=preview_path,
                completed_steps=int(spec.checkpoint_step),
                total_steps=20 if spec.model_variant == "base" else int(spec.preset["steps"]),
                preview_latents_path=preview_latents_path,
            )
        preview_ready = preview_callbacks.get("preview_ready_callback")
        if preview_ready is not None:
            preview_path = self.video_path.with_name("preview.mp4")
            preview_path.write_bytes(b"preview-video")
            preview_ready({"output_path": str(preview_path)})
            wait_decision = preview_callbacks.get("preview_decision_wait")
            if wait_decision is not None:
                decision = await asyncio.to_thread(wait_decision)
                if decision != "continue":
                    raise JobCancelled("preview discarded")
        latent_path = self.video_path.with_suffix(".pt")
        latent_path.write_bytes(b"clean-h3-av-latent")
        return GenerationResult(
            runtime_key=self.key,
            elapsed_seconds=1.25,
            output_path=self.video_path,
            final_latents_path=latent_path,
        )

    async def second_sample(
        self, spec, second_sampling: SecondSamplingSpec, source_latents_path,
        job_id, _first_frame, _last_frame, _reference_images,
        _reference_videos, _reference_audios, cancel_event,
        progress_callback=None,
    ) -> GenerationResult:
        if cancel_event.is_set():
            raise JobCancelled("cancelled")
        self.last_spec = spec
        self.last_second_sampling = second_sampling
        output = self.video_path.with_name(f"{job_id}.mp4")
        output.write_bytes(b"h3-second-sampled-video")
        latent_path = output.with_suffix(".pt")
        latent_path.write_bytes(b"second-pass-clean-latent")
        if progress_callback:
            progress_callback({
                "percent": 70, "stage": "second_sampling", "detail": "1/1",
            })
        return GenerationResult(
            runtime_key="original:native-sm89",
            elapsed_seconds=2.0,
            output_path=output,
            inference_plan={"ultimate_upscale": {"full_canvas": True}},
            final_latents_path=latent_path,
        )

    async def video_repair(
        self, spec, video_repair: VideoRepairSpec, source_video_path,
        job_id, cancel_event, progress_callback=None,
    ) -> GenerationResult:
        if cancel_event.is_set():
            raise JobCancelled("cancelled")
        self.last_spec = spec
        self.last_video_repair = video_repair
        self.last_video_repair_source = Path(source_video_path)
        output = self.video_path.with_name(f"{job_id}.mp4")
        output.write_bytes(b"video-repaired")
        if progress_callback:
            progress_callback({
                "percent": 78,
                "stage": "video_repair_h3",
                "detail": "1/1",
            })
        return GenerationResult(
            runtime_key="original:native-sm89",
            elapsed_seconds=1.75,
            output_path=output,
            inference_plan={
                "video_repair": {
                    **video_repair.to_dict(),
                    "implementation": "tracked_face_atlas_h3_turbo_v3",
                },
            },
        )

    async def continue_generate(
        self, spec, continuation, source_latents_path, source_memory_path,
        job_id, reference_images, reference_audios, cancel_event,
        progress_callback=None, **preview_callbacks,
    ) -> GenerationResult:
        if cancel_event.is_set():
            raise JobCancelled("cancelled")
        self.last_spec = spec
        self.last_continuation = continuation
        self.last_continuation_source = Path(source_latents_path)
        self.reference_images = reference_images
        self.reference_audios = reference_audios
        output = self.video_path.with_name(f"{job_id}.mp4")
        checkpoint_path = preview_callbacks.get("checkpoint_path")
        resume_checkpoint_path = preview_callbacks.get("resume_checkpoint_path")
        self.last_continuation_resume_checkpoint = (
            Path(resume_checkpoint_path) if resume_checkpoint_path else None
        )
        if checkpoint_path is not None:
            checkpoint_path = Path(checkpoint_path)
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            checkpoint_path.write_bytes(b"infinite-formal-checkpoint")
            preview_path = None
            if spec.checkpoint_preview:
                preview_path = output.with_name(f"{job_id}.checkpoint-preview.mp4")
                preview_path.write_bytes(b"infinite-checkpoint-preview")
            preview_latents_path = None
            if spec.selflift_enabled:
                preview_latents_path = output.with_name(
                    f"{job_id}.checkpoint-preview.pt"
                )
                preview_latents_path.write_bytes(
                    b"cumulative-selflift-low-resolution-preview"
                )
            return CheckpointResult(
                runtime_key="original:native-sm89",
                elapsed_seconds=0.75,
                checkpoint_path=checkpoint_path,
                preview_path=preview_path,
                completed_steps=int(spec.checkpoint_step),
                total_steps=int(spec.sampling_steps),
                preview_latents_path=preview_latents_path,
            )
        output.write_bytes(b"cumulative-infinite-video")
        latent_path = output.with_suffix(".pt")
        latent_path.write_bytes(b"cumulative-clean-av-latent")
        memory_path = output.with_suffix(".memory.pt")
        if continuation.memory > 0:
            memory_path.write_bytes(b"bounded-av-memory")
        if progress_callback:
            progress_callback({
                "percent": 75, "stage": "infinite_continuation_window",
                "detail": "strict continuation",
            })
        preview_ready = preview_callbacks.get("preview_ready_callback")
        if preview_ready is not None:
            preview_path = output.with_name(f"{job_id}.preview.mp4")
            preview_path.write_bytes(b"infinite-tail-intermediate-preview")
            preview_ready({"output_path": str(preview_path)})
        return GenerationResult(
            runtime_key="original:native-sm89",
            elapsed_seconds=2.0,
            output_path=output,
            inference_plan={
                "infinite_continuation": {
                    "physical_boundary": "strict_continuation",
                },
            },
            final_latents_path=latent_path,
            token_memory_path=memory_path if memory_path.is_file() else None,
        )

    async def complete_infinite_selflift(
        self, sources, job_id, cancel_event, progress_callback=None, final_spec=None,
    ) -> GenerationResult:
        if cancel_event.is_set():
            raise JobCancelled("cancelled")
        self.last_infinite_selflift_sources = tuple(sources)
        self.last_infinite_selflift_final_spec = final_spec
        output = self.video_path.with_name(f"{job_id}.mp4")
        output.write_bytes(b"selflift-final-film")
        latent_path = output.with_suffix(".pt")
        latent_path.write_bytes(b"selflift-final-clean-av-latent")
        if progress_callback:
            progress_callback({
                "percent": 90,
                "stage": "infinite_selflift_assemble",
                "detail": "fake SelfLift final",
            })
        return GenerationResult(
            runtime_key="original:native-sm89",
            elapsed_seconds=2.25,
            output_path=output,
            final_latents_path=latent_path,
            inference_plan={
                "infinite_selflift": {
                    "schema_version": "global_sliding_selflift_v1",
                    "window_count": len(sources),
                },
            },
        )

    async def stop(self) -> None:
        self.key = None
        self.preloaded = None
        self.warm_state = {"status": "cold", "engine": None}

    def preflight(self, _engine: str) -> dict:
        return {"ready": True, "checks": {"fake": True}}


class FakeUpscaler:
    def __init__(self):
        self.stop_calls = 0

    def status(self):
        return {"ready": True, "implementation": "fake", "missing": []}

    async def upscale(
        self, source, *, target_width, target_height, cancel_event,
        progress_callback=None, output_path=None,
    ):
        from h3serve.upscaler import UpscaleResult

        if progress_callback:
            progress_callback({
                "percent": 50, "stage": "upscaling", "detail": "fake upscale"
            })
        output = Path(output_path) if output_path is not None else source.with_name("upscaled.mp4")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(source.read_bytes() + b"-upscaled")
        return UpscaleResult(
            output, 2.5, target_width, target_height, 9000.0, 9800.0,
            {"inference": 1.5, "encode": 0.5},
        )

    async def stop(self):
        self.stop_calls += 1


class ApiTest(AioHTTPTestCase):
    async def get_application(self):
        self.temporary = Path(tempfile.mkdtemp(prefix="h3serve-test-"))
        self.video = self.temporary / "result.mp4"
        self.video.write_bytes(b"test-video")
        serve_dir = Path(__file__).resolve().parents[1]
        paths = ServicePaths.defaults(self.temporary, data_dir=self.temporary / "data")
        return create_app(
            paths=paths,
            serve_dir=serve_dir,
            api_key="secret",
            backend=FakeBackend(self.video),
        )

    async def asyncTearDown(self) -> None:
        await super().asyncTearDown()
        shutil.rmtree(self.temporary, ignore_errors=True)

    async def test_long_video_preview_compiles_without_queuing(self) -> None:
        headers = {"X-API-Key": "secret"}
        example_path = (
            Path(__file__).resolve().parents[1]
            / "static" / "long-video-examples" / "cafe30.json"
        )
        payload = json.loads(example_path.read_text(encoding="utf-8"))
        response = await self.client.post(
            "/api/v1/long-video/preview", headers=headers, json=payload,
        )
        self.assertEqual(response.status, 200, await response.text())
        preview = await response.json()
        self.assertEqual(len(preview["windows"]), 3)
        self.assertEqual(
            [item["transition"] for item in preview["windows"]],
            ["opening", "cut", "cut"],
        )
        self.assertEqual(preview["memory_budget"]["video_frames"], 6)
        self.assertEqual(preview["memory_budget"]["audio_ticks_per_clip"], 79)
        self.assertEqual(self.app["job_service"].jobs, {})

    async def test_auth_contract_queue_and_video(self) -> None:
        response = await self.client.get("/api/v1/options")
        self.assertEqual(response.status, 401)

        headers = {"X-API-Key": "secret"}
        response = await self.client.get("/api/v1/options", headers=headers)
        self.assertEqual(response.status, 200)
        options = await response.json()
        self.assertEqual(options["deployment_mode"], "fixed_engine")
        self.assertEqual(options["current_engine"], "first_last")
        self.assertEqual(set(options["engines"]), {"original", "lora"})
        self.assertEqual(options["defaults"]["quality"], "balanced")
        self.assertIn("1080p", options["resolutions"])
        self.assertNotIn("2k", options["resolutions"])
        self.assertIn("1440p", options["progressive_resolutions"])
        self.assertEqual(options["progressive_resolution"]["max"], 1440)
        self.assertEqual(options["progressive_resolution"]["first_pass_max"], 1080)
        self.assertIn(
            "1440p", options["advanced_limits"]["second_sampling"]["levels"]
        )
        self.assertEqual(
            options["advanced_limits"]["second_sampling"]["default_method"],
            "h3",
        )
        self.assertFalse(
            options["advanced_limits"]["second_sampling"]["methods"]["temporal"]["available"]
        )
        self.assertTrue(
            options["advanced_limits"]["second_sampling"]["methods"]["h3"]["available"]
        )
        self.assertEqual(options["duration"]["max_by_resolution"]["1080p"], 15)
        self.assertEqual(options["duration"]["max_by_preset"]["1080p"]["4:3"], 15.0)
        self.assertIn("1440p", options["advanced_limits"]["upscaler"]["levels"])
        self.assertFalse(options["advanced_limits"]["sparse_attention_available"])
        self.assertEqual(
            options["advanced_limits"]["acceleration"]["scheduler"],
            "h3_int8_frozen_round229",
        )
        self.assertEqual(
            options["advanced_limits"]["acceleration"]["scheduler_by_variant"],
            {
                "base": "h3_int8_frozen_round229",
                "lora": "h3_lora_v1_no_forecast_round229",
            },
        )
        self.assertEqual(self.app["job_service"].backend.preloaded, "fl2va_int8_24gb")

        health = await (await self.client.get("/healthz")).json()
        self.assertEqual(health["warm_state"]["status"], "ready")

        response = await self.client.post("/api/v1/generations", headers=headers, json={
            "prompt": "A short stable scene.",
            "resolution": "480p",
            "aspect_ratio": "16:9",
            "duration_seconds": 5,
            "seed": 4404,
        })
        self.assertEqual(response.status, 202)
        job_id = (await response.json())["id"]

        job = None
        for _ in range(50):
            response = await self.client.get(f"/api/v1/jobs/{job_id}", headers=headers)
            job = await response.json()
            if job["status"] in {"succeeded", "failed"}:
                break
            await asyncio.sleep(0.01)
        self.assertIsNotNone(job)
        self.assertEqual(job["status"], "succeeded")
        self.assertEqual(job["request"]["engine"], "original")
        self.assertEqual(job["request"]["quality"], "balanced")
        self.assertEqual(job["progress"]["percent"], 100.0)
        self.assertEqual(job["progress"]["stage"], "completed")
        self.assertEqual(job["progress"]["estimated_remaining_seconds"], 0.0)
        self.assertEqual(job["elapsed_seconds"], 1.25)
        self.assertNotIn("execution", job["request"])
        self.assertNotIn("runtime_key", job)

        response = await self.client.get(f"/api/v1/jobs/{job_id}/video", headers=headers)
        self.assertEqual(response.status, 200)
        self.assertEqual(await response.read(), b"test-video")

    async def test_settings_can_discover_and_switch_native_h3_lora(self) -> None:
        from safetensors.numpy import save_file

        lora_root = self.temporary / "models" / "loras"
        lora_root.mkdir(parents=True)
        compatible = lora_root / "release-v2.safetensors"
        save_file({
            "blocks.0.attn.to_q.lora_A.weight": np.zeros((1, 1), dtype=np.float16),
            "blocks.0.attn.to_q.lora_B.weight": np.zeros((1, 1), dtype=np.float16),
        }, compatible, metadata={"base_model": "MiniMax-H3"})
        save_file({
            "diffusion_model.other.weight": np.zeros((1, 1), dtype=np.float16),
        }, lora_root / "foreign.safetensors")
        lightx = lora_root / "minimax_h3_fl2v_turbo_4step_v1.1_768p_bf16.safetensors"
        lightx_state = {}
        for index in range(312):
            prefix = f"synthetic.{index}"
            lightx_state[f"{prefix}.lora_A.default.weight"] = np.zeros(
                (1, 1), dtype=np.float16
            )
            lightx_state[f"{prefix}.lora_B.default.weight"] = np.zeros(
                (1, 1), dtype=np.float16
            )
        save_file(
            lightx_state,
            lightx,
            metadata={"key_format": "minimax-h3-diffusers", "alpha": "128"},
        )

        headers = {"X-API-Key": "secret"}
        response = await self.client.get("/api/v1/settings/lora", headers=headers)
        self.assertEqual(response.status, 200)
        catalog = await response.json()
        by_id = {item["id"]: item for item in catalog["available"]}
        self.assertTrue(by_id["release-v2.safetensors"]["compatible"])
        self.assertEqual(by_id["release-v2.safetensors"]["pair_count"], 1)
        self.assertFalse(by_id["foreign.safetensors"]["compatible"])
        lightx_item = by_id[lightx.name]
        self.assertTrue(lightx_item["compatible"])
        self.assertEqual(lightx_item["pair_count"], 312)
        self.assertEqual(
            lightx_item["profile"]["profile_id"],
            "lightx2v_fl2v_4step_v1_1_768p",
        )
        self.assertEqual(lightx_item["profile"]["default_steps"], 4)

        response = await self.client.put(
            "/api/v1/settings/lora", headers=headers,
            json={"checkpoint": "release-v2.safetensors"},
        )
        self.assertEqual(response.status, 200)
        changed = await response.json()
        self.assertTrue(changed["changed"])
        self.assertEqual(changed["selected"], "release-v2.safetensors")
        self.assertEqual(changed["loaded"], "release-v2.safetensors")
        self.assertEqual(
            json.loads(
                (self.temporary / "data/settings/lora.json").read_text(
                    encoding="utf-8"
                )
            )["checkpoint"],
            "release-v2.safetensors",
        )

        response = await self.client.put(
            "/api/v1/settings/lora", headers=headers,
            json={"checkpoint": "foreign.safetensors"},
        )
        self.assertEqual(response.status, 400)

        failing = lora_root / "broken-build.safetensors"
        save_file({
            "blocks.0.attn.to_q.lora_A.weight": np.zeros((1, 1), dtype=np.float16),
            "blocks.0.attn.to_q.lora_B.weight": np.zeros((1, 1), dtype=np.float16),
        }, failing, metadata={"base_model": "MiniMax-H3"})
        backend = self.app["job_service"].backend
        backend.fail_lora_checkpoint = failing.name
        response = await self.client.put(
            "/api/v1/settings/lora", headers=headers,
            json={"checkpoint": failing.name},
        )
        self.assertEqual(response.status, 500)
        self.assertEqual(backend.lora_checkpoint.name, "release-v2.safetensors")
        self.assertEqual(backend.warm_state["status"], "ready")
        persisted = json.loads(
            (self.temporary / "data/settings/lora.json").read_text(encoding="utf-8")
        )
        self.assertEqual(persisted["checkpoint"], "release-v2.safetensors")

    async def test_completed_card_can_queue_native_h3_second_sampling(self) -> None:
        headers = {"X-API-Key": "secret"}
        response = await self.client.post(
            "/api/v1/generations", headers=headers, json={
                "prompt": "low resolution selectable card",
                "resolution": "480p", "aspect_ratio": "16:9",
                "duration_seconds": 15,
                "model_variant": "lora",
            },
        )
        source_id = (await response.json())["id"]
        for _ in range(80):
            source = await (
                await self.client.get(
                    f"/api/v1/jobs/{source_id}", headers=headers
                )
            ).json()
            if source["status"] in {"succeeded", "failed"}:
                break
            await asyncio.sleep(0.01)
        self.assertEqual(source["status"], "succeeded")
        self.assertEqual(source["request"]["model_variant"], "lora")
        self.assertTrue(source["second_sampling_available"])

        response = await self.client.post(
            f"/api/v1/jobs/{source_id}/second-sampling",
            headers=headers,
            json={
                "resolution": "1080p", "steps": 1,
                "acceleration": 75, "strength": "enhance",
                "memory_mode": "auto",
                "temporal_window_frames": 119,
            },
        )
        self.assertEqual(response.status, 202, await response.text())
        child_id = (await response.json())["id"]
        for _ in range(80):
            child = await (
                await self.client.get(
                    f"/api/v1/jobs/{child_id}", headers=headers
                )
            ).json()
            if child["status"] in {"succeeded", "failed"}:
                break
            await asyncio.sleep(0.01)
        self.assertEqual(child["status"], "succeeded")
        self.assertEqual(child["second_sampling"]["source_job_id"], source_id)
        self.assertEqual(child["request"]["model_variant"], "base")
        self.assertEqual(child["second_sampling"]["model_variant"], "base")
        self.assertEqual(child["second_sampling"]["strength"], "enhance")
        self.assertEqual(child["second_sampling"]["denoise"], 0.25)
        self.assertEqual(
            child["second_sampling"]["temporal_window_frames"], 119
        )
        self.assertEqual(
            (child["second_sampling"]["width"], child["second_sampling"]["height"]),
            (1920, 1088),
        )
        self.assertTrue(child["inference_plan"]["ultimate_upscale"]["full_canvas"])
        video = await self.client.get(
            f"/api/v1/jobs/{child_id}/video", headers=headers
        )
        self.assertEqual(await video.read(), b"h3-second-sampled-video")

        response = await self.client.post(
            f"/api/v1/jobs/{source_id}/second-sampling",
            headers=headers,
            json={
                "resolution": "1080p", "steps": 4,
                "model_variant": "lora", "acceleration": 75,
                "strength": "standard",
            },
        )
        self.assertEqual(response.status, 202, await response.text())
        lora_child_id = (await response.json())["id"]
        for _ in range(80):
            lora_child = await (
                await self.client.get(
                    f"/api/v1/jobs/{lora_child_id}", headers=headers
                )
            ).json()
            if lora_child["status"] in {"succeeded", "failed"}:
                break
            await asyncio.sleep(0.01)
        self.assertEqual(lora_child["status"], "succeeded")
        self.assertEqual(lora_child["request"]["model_variant"], "lora")
        self.assertEqual(lora_child["request"]["engine"], "lora")
        self.assertEqual(lora_child["second_sampling"]["model_variant"], "lora")
        self.assertEqual(lora_child["second_sampling"]["steps"], 4)

    async def test_completed_card_can_queue_pixel_video_repair(self) -> None:
        headers = {"X-API-Key": "secret"}
        settings_response = await self.client.put(
            "/api/v1/settings/face-repair", headers=headers,
            json={"canvas_size": 1088, "capacity": 16},
        )
        self.assertEqual(settings_response.status, 200)
        settings = await settings_response.json()
        self.assertEqual(settings["steps"], 4)
        self.assertEqual(settings["canvas_size"], 1088)
        self.assertEqual(settings["capacity"], 16)
        persisted_settings = json.loads(
            (self.temporary / "data/settings/face_repair.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(
            persisted_settings, {"canvas_size": 1088, "capacity": 16}
        )
        response = await self.client.post(
            "/api/v1/generations", headers=headers, json={
                "prompt": "Several distant faces in a station.",
                "resolution": "480p",
                "aspect_ratio": "16:9",
                "duration_seconds": 5,
                "model_variant": "lora",
            },
        )
        source_id = (await response.json())["id"]
        for _ in range(80):
            source = await (
                await self.client.get(
                    f"/api/v1/jobs/{source_id}", headers=headers
                )
            ).json()
            if source["status"] in {"succeeded", "failed"}:
                break
            await asyncio.sleep(0.01)
        self.assertEqual(source["status"], "succeeded")
        self.assertTrue(source["video_repair_available"])

        response = await self.client.post(
            f"/api/v1/jobs/{source_id}/video-repair",
            headers=headers,
            json={
                "acceleration": 72,
            },
        )
        self.assertEqual(response.status, 202, await response.text())
        queued = await response.json()
        self.assertEqual(queued["video_repair"]["source_job_id"], source_id)
        self.assertEqual(queued["video_repair"]["max_faces"], 16)
        self.assertEqual(
            queued["video_repair"]["layout_policy"],
            "fixed_square_cells",
        )
        self.assertEqual(queued["video_repair"]["canvas_size"], 1088)
        self.assertEqual(queued["video_repair"]["steps"], 4)
        self.assertEqual(queued["video_repair"]["acceleration"], 72)
        self.assertEqual(queued["video_repair"]["minimum_window_seconds"], 4.0)
        self.assertEqual(queued["video_repair"]["window_seconds"], 6.0)
        self.assertEqual(
            queued["video_repair"]["window_policy"],
            "automatic_canvas_tier",
        )
        self.assertEqual(queued["request"]["model_variant"], "lora")

        child_id = queued["id"]
        for _ in range(80):
            child = await (
                await self.client.get(
                    f"/api/v1/jobs/{child_id}", headers=headers
                )
            ).json()
            if child["status"] in {"succeeded", "failed"}:
                break
            await asyncio.sleep(0.01)
        self.assertEqual(child["status"], "succeeded", child.get("error"))
        self.assertEqual(child["progress"]["detail"], "人脸修复完成")
        self.assertEqual(
            child["inference_plan"]["video_repair"]["implementation"],
            "tracked_face_atlas_h3_turbo_v3",
        )
        backend = self.app["job_service"].backend
        self.assertEqual(backend.last_video_repair.mode, "face")
        self.assertEqual(backend.last_video_repair_source, self.video)
        persisted = json.loads(
            (self.temporary / "data" / "jobs" / f"{child_id}.json").read_text()
        )
        # Completed jobs normalize pending_action back to "generate" because
        # there is no queued operation left.  The durable task identity and
        # parameters live in the immutable video_repair contract.
        self.assertEqual(persisted["_internal"]["pending_action"], "generate")
        self.assertEqual(persisted["_internal"]["source_job_id"], source_id)
        self.assertEqual(persisted["_internal"]["video_repair"]["max_faces"], 16)
        video = await self.client.get(
            f"/api/v1/jobs/{child_id}/video", headers=headers
        )
        self.assertEqual(await video.read(), b"video-repaired")

    async def test_reference_card_does_not_offer_or_accept_face_repair(self) -> None:
        headers = {"X-API-Key": "secret"}
        service = self.app["job_service"]
        source = JobRecord(
            id="reference-face-repair-source",
            spec=GenerationSpec.from_mapping({
                "prompt": "A reference-conditioned portrait video.",
                "runtime_launcher": "ref2va_int8_24gb",
                "service_family": "reference",
                "resolution": "480p",
                "duration_seconds": 5,
            }),
            status="succeeded",
            output_path=self.video,
        )
        service.jobs[source.id] = source

        card_response = await self.client.get(
            f"/api/v1/jobs/{source.id}", headers=headers
        )
        self.assertEqual(card_response.status, 200)
        card = await card_response.json()
        self.assertFalse(card["video_repair_available"])

        repair_response = await self.client.post(
            f"/api/v1/jobs/{source.id}/video-repair",
            headers=headers,
            json={"acceleration": 50},
        )
        self.assertEqual(repair_response.status, 400)
        self.assertIn(
            "only for completed FL2VA source jobs",
            await repair_response.text(),
        )

    async def test_completed_card_can_queue_temporal_video_second_sampling(self) -> None:
        headers = {"X-API-Key": "secret"}
        response = await self.client.post(
            "/api/v1/generations", headers=headers, json={
                "prompt": "distant faces and detailed shelves",
                "resolution": "480p", "aspect_ratio": "16:9",
                "duration_seconds": 5,
            },
        )
        source_id = (await response.json())["id"]
        for _ in range(80):
            source = await (
                await self.client.get(f"/api/v1/jobs/{source_id}", headers=headers)
            ).json()
            if source["status"] in {"succeeded", "failed"}:
                break
            await asyncio.sleep(0.01)
        self.assertEqual(source["status"], "succeeded")
        self.assertEqual(
            source["second_sampling_methods"], {"temporal": True, "h3": True}
        )

        service = self.app["job_service"]
        service.upscaler = FakeUpscaler()
        response = await self.client.post(
            f"/api/v1/jobs/{source_id}/second-sampling",
            headers=headers,
            json={"method": "temporal", "resolution": "720p"},
        )
        self.assertEqual(response.status, 202, await response.text())
        child_id = (await response.json())["id"]
        for _ in range(80):
            child = await (
                await self.client.get(f"/api/v1/jobs/{child_id}", headers=headers)
            ).json()
            if child["status"] in {"succeeded", "failed"}:
                break
            await asyncio.sleep(0.01)
        self.assertEqual(child["status"], "succeeded", child.get("error"))
        self.assertEqual(child["second_sampling"]["method"], "temporal")
        self.assertEqual(child["request"]["width"], 1280)
        self.assertEqual(child["request"]["height"], 736)
        self.assertEqual(
            child["inference_plan"]["second_sampling_method"]["inference_steps"],
            1,
        )
        self.assertEqual(
            child["stage_seconds"],
            {
                "temporal_second_sampling.inference": 1.5,
                "temporal_second_sampling.encode": 0.5,
            },
        )
        video = await self.client.get(
            f"/api/v1/jobs/{child_id}/video", headers=headers
        )
        self.assertEqual(await video.read(), b"test-video-upscaled")

    async def test_infinite_project_appends_and_replaces_only_the_tail(self) -> None:
        headers = {"X-API-Key": "secret"}
        response = await self.client.post(
            "/api/v1/infinite-projects",
            headers=headers,
            json={
                "title": "Cafe infinite",
                "overview": "The same owner remains inside one fixed cafe.",
                "overall_soundscape": "Continuous quiet cafe room tone.",
                "non_diegetic_music": "N/A",
                "overlap_seconds": 1.625,
                "memory": 60,
            },
        )
        self.assertEqual(response.status, 201, await response.text())
        project = await response.json()
        project_id = project["id"]

        async def append(description: str, **overrides):
            payload = {
                "window_description": description,
                "duration_seconds": 5,
                "model_variant": "base",
                "sampling_steps": 20,
                "acceleration": 0,
                "seed": 100,
                "resolution": "650p",
                "aspect_ratio": "16:9",
                "visual_memory_capacity": 12,
                "audio_memory_capacity": 2,
                "visual_memory_resolution": "480p",
                "overlap_seconds": 1.625,
                "save_shared_as_default": True,
            }
            payload.update(overrides)
            response = await self.client.post(
                f"/api/v1/infinite-projects/{project_id}/windows",
                headers=headers,
                json=payload,
            )
            self.assertEqual(response.status, 202, await response.text())
            return (await response.json())["job"]["id"]

        first_id = await append("A continuous counter shot establishes the owner.")
        for _ in range(60):
            project = await (
                await self.client.get(
                    f"/api/v1/infinite-projects/{project_id}", headers=headers
                )
            ).json()
            if project["tail_status"] in {"succeeded", "failed"}:
                break
            await asyncio.sleep(0.01)
        self.assertEqual(project["tail_status"], "succeeded")
        self.assertEqual(project["window_count"], 1)
        opening_job = await (
            await self.client.get(
                f"/api/v1/jobs/{first_id}", headers=headers
            )
        ).json()
        self.assertNotIn("preview", opening_job)
        self.assertEqual(opening_job["request"]["execution_mode"], "complete")

        second_id = await append(
            "Continue without a boundary cut. At 3 seconds, cut inside this "
            "window to the reverse angle of the same counter.",
            execution_mode="checkpoint",
            checkpoint_step=9,
            checkpoint_retain=True,
            checkpoint_preview=True,
            checkpoint_preview_steps=4,
            checkpoint_preview_resolution="360p",
        )
        for _ in range(60):
            project = await (
                await self.client.get(
                    f"/api/v1/infinite-projects/{project_id}", headers=headers
                )
            ).json()
            if project["tail_status"] in {"checkpointed", "failed"}:
                break
            await asyncio.sleep(0.01)
        self.assertEqual(project["tail_status"], "checkpointed")
        self.assertEqual(project["window_count"], 2)
        self.assertEqual(project["total_frames"], project["windows"][0]["total_frames"])
        self.assertFalse(project["can_append"])
        self.assertEqual(project["memory_capacity"]["video_slots"], 12)
        self.assertEqual(project["memory_capacity"]["audio_slots"], 2)
        self.assertEqual(project["memory_capacity"]["visual_resolution"], "480p")
        self.assertEqual((project["width"], project["height"]), (1152, 640))
        backend = self.app["job_service"].backend
        self.assertEqual(backend.last_continuation.source_job_id, first_id)
        self.assertEqual(backend.last_continuation.context_frames, 39)
        self.assertEqual(backend.last_continuation.visual_memory_capacity, 12)
        self.assertEqual(backend.last_continuation.audio_memory_capacity, 2)
        self.assertEqual(backend.last_continuation.visual_memory_resolution, "480p")
        self.assertEqual(
            backend.last_spec.output_frames,
            project["windows"][1]["total_frames"],
        )
        self.assertIn("strict continuation", backend.last_spec.prompt)
        self.assertIn("cut inside this window", backend.last_spec.prompt)
        self.assertEqual(backend.last_spec.preview_mode, "off")
        self.assertFalse(backend.last_spec.preview_fast_finish)
        self.assertEqual(backend.last_spec.execution_mode, "checkpoint")
        self.assertEqual(backend.last_spec.checkpoint_step, 9)
        second_job = await (
            await self.client.get(
                f"/api/v1/jobs/{second_id}", headers=headers
            )
        ).json()
        self.assertTrue(second_job["preview"]["ready"])
        self.assertTrue(second_job["checkpoint"]["resume_available"])
        response = await self.client.get(
            f"/api/v1/jobs/{second_id}/preview", headers=headers
        )
        self.assertEqual(response.status, 200, await response.text())
        self.assertEqual(
            await response.read(), b"infinite-checkpoint-preview"
        )

        response = await self.client.post(
            f"/api/v1/jobs/{second_id}/resume", headers=headers
        )
        self.assertEqual(response.status, 202, await response.text())
        for _ in range(60):
            project = await (
                await self.client.get(
                    f"/api/v1/infinite-projects/{project_id}", headers=headers
                )
            ).json()
            if project["tail_status"] in {"succeeded", "failed"}:
                break
            await asyncio.sleep(0.01)
        self.assertEqual(project["tail_status"], "succeeded")
        self.assertGreater(project["total_frames"], project["windows"][0]["total_frames"])
        self.assertIsNotNone(backend.last_continuation_resume_checkpoint)

        response = await self.client.post(
            f"/api/v1/infinite-projects/{project_id}/second-sampling",
            headers=headers,
            json={
                "resolution": "1080p",
                "steps": 2,
                "acceleration": 75,
                "strength": "preserve",
                "temporal_window_frames": 136,
            },
        )
        self.assertEqual(response.status, 202, await response.text())
        second_sample_id = (await response.json())["id"]
        for _ in range(60):
            sampled = await (
                await self.client.get(
                    f"/api/v1/jobs/{second_sample_id}", headers=headers
                )
            ).json()
            if sampled["status"] in {"succeeded", "failed"}:
                break
            await asyncio.sleep(0.01)
        self.assertEqual(sampled["status"], "succeeded")
        self.assertEqual(sampled["request"]["output_frames"], project["total_frames"])
        self.assertIn("Preserve the complete accepted source-latent", backend.last_spec.prompt)
        self.assertIn("Add no new action, cut, object", backend.last_spec.prompt)

        response = await self.client.delete(
            f"/api/v1/infinite-projects/{project_id}/windows/last",
            headers=headers,
        )
        self.assertEqual(response.status, 200, await response.text())
        rolled_back = await response.json()
        self.assertEqual(rolled_back["window_count"], 1)
        self.assertEqual(rolled_back["tail_job_id"], first_id)
        self.assertNotIn(second_id, self.app["job_service"].jobs)

        response = await self.client.delete(
            f"/api/v1/infinite-projects/{project_id}", headers=headers
        )
        self.assertEqual(response.status, 200, await response.text())
        deleted = await response.json()
        self.assertTrue(deleted["deleted"])
        self.assertEqual(deleted["retained_job_count"], 1)
        self.assertEqual(
            (await self.client.get(
                f"/api/v1/infinite-projects/{project_id}", headers=headers
            )).status,
            404,
        )
        self.assertEqual(
            (await self.client.get(
                f"/api/v1/jobs/{first_id}", headers=headers
            )).status,
            200,
        )

    async def test_locked_project_runs_json_previews_then_waits_for_final_sampling(self) -> None:
        headers = {"X-API-Key": "secret"}
        response = await self.client.post(
            "/api/v1/infinite-projects",
            headers=headers,
            json={
                "workflow_version": 2,
                "title": "JSON preview film",
                "overview": "One fixed station concourse.",
                "overall_soundscape": "Continuous station room tone.",
                "non_diegetic_music": "N/A",
                "preview_resolution": "540p",
                "aspect_ratio": "16:9",
                "model_variant": "lora",
                "sampling_steps": 8,
                "acceleration": 50,
                "window_duration_seconds": 5,
                "overlap_seconds": 0.75,
                "visual_memory_capacity": 8,
                "audio_memory_capacity": 1,
                "visual_memory_resolution": "360p",
            },
        )
        self.assertEqual(response.status, 201, await response.text())
        project = await response.json()
        project_id = project["id"]
        self.assertTrue(project["trajectory_locked"])
        self.assertEqual(project["preview_resolution"], "540p")

        response = await self.client.post(
            f"/api/v1/infinite-projects/{project_id}/batch",
            headers=headers,
            json={
                "overview": "The same fixed station with four people.",
                "windows": [
                    {"prompt": "[Shot 1] Establish the station.", "seed": 101},
                    "[Shot 1] Continue exactly; the group crosses the concourse.",
                ],
            },
        )
        self.assertEqual(response.status, 202, await response.text())

        for _ in range(120):
            response = await self.client.get(
                f"/api/v1/infinite-projects/{project_id}", headers=headers
            )
            project = await response.json()
            if project["batch"]["status"] in {"completed", "failed"}:
                break
            await asyncio.sleep(0.02)
        self.assertEqual(
            project["batch"]["status"],
            "completed",
            project["batch"].get("error"),
        )
        self.assertEqual(project["batch"]["cursor"], 2)
        self.assertEqual(project["window_count"], 2)
        self.assertEqual(project["tail_status"], "succeeded")
        self.assertIsNone(project["final_sampling"]["job_id"])
        self.assertTrue(project["second_sampling_available"])
        self.assertEqual(
            [item["batch_plan_index"] for item in project["windows"]],
            [0, 1],
        )
        for item in project["windows"]:
            job = self.app["job_service"].jobs[item["job_id"]]
            self.assertEqual(job.spec.resolution, "540p")
            self.assertEqual(job.spec.model_variant, "lora")
            self.assertEqual(job.spec.sampling_steps, 8)
            self.assertEqual(job.spec.acceleration, 50)
            self.assertEqual(job.spec.execution_mode, "complete")

        response = await self.client.post(
            f"/api/v1/infinite-projects/{project_id}/final-sampling",
            headers=headers,
            json={"method": "temporal", "resolution": "1080p", "steps": 4, "acceleration": 75},
        )
        self.assertEqual(response.status, 202, await response.text())
        final_id = (await response.json())["id"]
        for _ in range(60):
            project = await (
                await self.client.get(
                    f"/api/v1/infinite-projects/{project_id}", headers=headers
                )
            ).json()
            if project["final_sampling"]["status"] in {"succeeded", "failed"}:
                break
            await asyncio.sleep(0.01)
        self.assertEqual(project["final_sampling"]["job_id"], final_id)
        self.assertEqual(project["final_sampling"]["status"], "succeeded")
        self.assertEqual(project["final_sampling"]["settings"]["method"], "h3")
        self.assertIsNotNone(project["final_sampling"]["video_url"])

    async def test_v3_online_windows_keep_locked_trajectory_and_allow_window_timing(self) -> None:
        headers = {"X-API-Key": "secret"}
        response = await self.client.post(
            "/api/v1/infinite-projects",
            headers=headers,
            json={
                "workflow_version": 3,
                "creation_mode": "online",
                "title": "Editable online trajectory",
                "overview": "One continuous workshop.",
                "overall_soundscape": "Quiet machinery.",
                "non_diegetic_music": "N/A",
                "preview_resolution": "540p",
                "final_resolution": "1080p",
                "model_variant": "lora",
                "sampling_steps": 8,
                "final_sampling_steps": 5,
                "preview_branch_steps": 3,
                "acceleration": 50,
                "second_pass_acceleration": 67,
                "window_duration_seconds": 5,
                "overlap_seconds": 0.75,
            },
        )
        self.assertEqual(response.status, 201, await response.text())
        project = await response.json()
        project_id = project["id"]
        self.assertTrue(project["window_controls_editable"])

        for index, controls in enumerate((
            {
                "duration_seconds": 4,
                "overlap_seconds": 0.5,
                "acceleration": 25,
                "visual_memory_capacity": 3,
                "audio_memory_capacity": 0,
                "visual_memory_resolution": "480p",
            },
            {
                "duration_seconds": 1,
                "overlap_seconds": 0,
                "acceleration": 35,
                "visual_memory_capacity": 5,
                "audio_memory_capacity": 2,
                "visual_memory_resolution": "360p",
            },
        )):
            response = await self.client.post(
                f"/api/v1/infinite-projects/{project_id}/windows",
                headers=headers,
                json={
                    "window_description": f"[Shot 1] Window {index + 1}.",
                    "save_shared_as_default": True,
                    **controls,
                },
            )
            self.assertEqual(response.status, 202, await response.text())
            for _ in range(80):
                project = await (
                    await self.client.get(
                        f"/api/v1/infinite-projects/{project_id}", headers=headers
                    )
                ).json()
                if project["tail_status"] in {"succeeded", "failed"}:
                    break
                await asyncio.sleep(0.01)
            self.assertEqual(project["tail_status"], "succeeded")

        self.assertEqual(project["window_count"], 2)
        self.assertEqual(project["window_duration_seconds"], 1)
        self.assertEqual(project["overlap_seconds"], 0)
        self.assertEqual(project["acceleration"], 35)
        self.assertEqual(project["visual_memory_capacity"], 5)
        self.assertEqual(project["audio_memory_capacity"], 2)
        second = self.app["job_service"].jobs[project["windows"][1]["job_id"]]
        self.assertEqual(second.spec.model_variant, "lora")
        self.assertEqual(second.spec.sampling_steps, 8)
        self.assertEqual(second.spec.acceleration, 35)
        self.assertEqual(second.spec.second_pass_acceleration, 35)
        self.assertEqual(second.infinite_continuation.context_frames, 0)
        self.assertEqual(second.infinite_continuation.hidden_prefix_frames, 5)
        self.assertTrue(second.spec.selflift_enabled)
        self.assertEqual(second.spec.resolution, "1080p")
        self.assertEqual(second.spec.selflift_initial_resolution, "540p")
        self.assertEqual(second.spec.selflift_transition_step, 3)
        self.assertEqual(second.spec.checkpoint_step, 3)
        self.assertEqual(second.spec.checkpoint_preview_steps, 2)
        self.assertIsNotNone(second.checkpoint_path)
        self.assertTrue(second.checkpoint_path.is_file())
        self.assertIsNotNone(second.final_latents_path)
        self.assertTrue(second.final_latents_path.is_file())
        self.assertEqual(project["windows"][1]["visual_memory_capacity"], 5)
        self.assertEqual(project["final_generation_method"], "global_sliding_selflift")

        response = await self.client.post(
            f"/api/v1/infinite-projects/{project_id}/final-sampling",
            headers=headers,
            json={"acceleration": 82, "sigma_scale": 0.75},
        )
        self.assertEqual(response.status, 202, await response.text())
        final_id = (await response.json())["id"]
        for _ in range(80):
            state = await (
                await self.client.get(f"/api/v1/jobs/{final_id}", headers=headers)
            ).json()
            if state["status"] in {"succeeded", "failed"}:
                break
            await asyncio.sleep(0.01)
        self.assertEqual(state["status"], "succeeded")
        backend = self.app["job_service"].backend
        self.assertEqual(len(backend.last_infinite_selflift_sources), 2)
        self.assertEqual(
            state["inference_plan"]["infinite_selflift"]["schema_version"],
            "global_sliding_selflift_v1",
        )
        project = await (
            await self.client.get(
                f"/api/v1/infinite-projects/{project_id}", headers=headers
            )
        ).json()
        self.assertEqual(
            project["final_sampling"]["settings"]["method"],
            "global_sliding_selflift",
        )
        self.assertEqual(project["final_sampling"]["settings"]["resolution"], "1080p")
        self.assertEqual(project["final_sampling"]["settings"]["steps"], 5)
        self.assertEqual(project["final_sampling"]["settings"]["acceleration"], 82)
        self.assertEqual(project["final_sampling"]["settings"]["sigma_scale"], 0.75)
        final_job = self.app["job_service"].jobs[final_id]
        self.assertEqual(final_job.spec.second_pass_acceleration, 82)
        self.assertEqual(final_job.spec.selflift_sigma_scale, 0.75)
        self.assertIsNotNone(backend.last_infinite_selflift_final_spec)
        self.assertEqual(
            backend.last_infinite_selflift_final_spec.second_pass_acceleration,
            82,
        )
        self.assertEqual(
            backend.last_infinite_selflift_final_spec.selflift_sigma_scale,
            0.75,
        )

    async def test_v3_online_full_film_selflift_accepts_1440p_target(self) -> None:
        headers = {"X-API-Key": "secret"}
        response = await self.client.post(
            "/api/v1/infinite-projects",
            headers=headers,
            json={
                "workflow_version": 3,
                "creation_mode": "online",
                "title": "1440P full-film SelfLift",
                "preview_resolution": "540p",
                "final_resolution": "1440p",
                "aspect_ratio": "16:9",
                "model_variant": "lora",
                "sampling_steps": 8,
                "final_sampling_steps": 2,
                "preview_branch_steps": 2,
                "acceleration": 50,
                "second_pass_acceleration": 70,
                "window_duration_seconds": 1,
                "overlap_seconds": 0,
            },
        )
        self.assertEqual(response.status, 201, await response.text())
        project_id = (await response.json())["id"]

        response = await self.client.post(
            f"/api/v1/infinite-projects/{project_id}/windows",
            headers=headers,
            json={
                "window_description": "[Shot 1] One continuous test shot.",
                "duration_seconds": 1,
                "overlap_seconds": 0,
                "acceleration": 50,
            },
        )
        self.assertEqual(response.status, 202, await response.text())
        for _ in range(80):
            project = await (
                await self.client.get(
                    f"/api/v1/infinite-projects/{project_id}", headers=headers
                )
            ).json()
            if project["tail_status"] in {"succeeded", "failed"}:
                break
            await asyncio.sleep(0.01)
        self.assertEqual(project["tail_status"], "succeeded")

        response = await self.client.post(
            f"/api/v1/infinite-projects/{project_id}/final-sampling",
            headers=headers,
            json={"acceleration": 70},
        )
        self.assertEqual(response.status, 202, await response.text())
        final_id = (await response.json())["id"]
        final_job = self.app["job_service"].jobs[final_id]
        self.assertEqual(final_job.spec.resolution, "2k")
        self.assertEqual((final_job.spec.width, final_job.spec.height), (2560, 1440))

    async def test_infinite_window_can_inherit_replace_and_remove_keyframes(self) -> None:
        headers = {"X-API-Key": "secret"}
        response = await self.client.post(
            "/api/v1/infinite-projects",
            headers=headers,
            json={
                "workflow_version": 3,
                "creation_mode": "online",
                "title": "Per-window keyframes",
                "overview": "One continuous scene with stable subjects.",
                "preview_resolution": "540p",
                "final_resolution": "1080p",
                "model_variant": "lora",
                "sampling_steps": 8,
                "final_sampling_steps": 2,
            },
        )
        self.assertEqual(response.status, 201, await response.text())
        project_id = (await response.json())["id"]

        image_buffer = io.BytesIO()
        Image.new("RGB", (32, 32), (40, 80, 120)).save(image_buffer, format="PNG")
        image_bytes = image_buffer.getvalue()
        opening = aiohttp.FormData(default_to_multipart=True)
        opening.add_field("window_description", "Opening window")
        opening.add_field("duration_seconds", "4")
        opening.add_field("first_frame", image_bytes, filename="first.png", content_type="image/png")
        opening.add_field("last_frame", image_bytes, filename="last.png", content_type="image/png")
        response = await self.client.post(
            f"/api/v1/infinite-projects/{project_id}/windows",
            headers=headers,
            data=opening,
        )
        self.assertEqual(response.status, 202, await response.text())
        for _ in range(80):
            project = await (
                await self.client.get(
                    f"/api/v1/infinite-projects/{project_id}", headers=headers
                )
            ).json()
            if project["tail_status"] in {"succeeded", "failed"}:
                break
            await asyncio.sleep(0.01)
        self.assertEqual(project["tail_status"], "succeeded")
        self.assertTrue(project["windows"][0]["conditioning"]["has_first_frame"])
        self.assertTrue(project["windows"][0]["conditioning"]["has_last_frame"])

        continuation = aiohttp.FormData(default_to_multipart=True)
        continuation.add_field("window_description", "Continuation window")
        continuation.add_field("duration_seconds", "4")
        continuation.add_field("overlap_seconds", "1")
        continuation.add_field("inherit_references", "true")
        continuation.add_field("excluded_reference_roles", "first_frame")
        continuation.add_field("last_frame", image_bytes, filename="new-last.png", content_type="image/png")
        response = await self.client.post(
            f"/api/v1/infinite-projects/{project_id}/windows",
            headers=headers,
            data=continuation,
        )
        self.assertEqual(response.status, 202, await response.text())
        continuation_id = (await response.json())["job"]["id"]
        job = self.app["job_service"].jobs[continuation_id]
        self.assertIsNone(job.first_frame)
        self.assertIsNotNone(job.last_frame)
        self.assertEqual(job.last_frame.name, "uploaded_last_frame.png")
        for _ in range(80):
            project = await (
                await self.client.get(
                    f"/api/v1/infinite-projects/{project_id}", headers=headers
                )
            ).json()
            if project["tail_status"] in {"succeeded", "failed"}:
                break
            await asyncio.sleep(0.01)
        self.assertEqual(project["tail_status"], "succeeded")

        response = await self.client.post(
            f"/api/v1/infinite-projects/{project_id}/windows",
            headers=headers,
            json={
                "window_description": "Too long after overlap",
                "duration_seconds": 14.25,
                "overlap_seconds": 1,
            },
        )
        self.assertEqual(response.status, 400)
        self.assertIn("continuation duration", await response.text())

    async def test_v3_equal_resolution_keeps_fork_without_spatial_lift(self) -> None:
        headers = {"X-API-Key": "secret"}
        response = await self.client.post(
            "/api/v1/infinite-projects",
            headers=headers,
            json={
                "workflow_version": 3,
                "creation_mode": "online",
                "title": "Identity handoff film",
                "overview": "One continuous same-resolution room.",
                "overall_soundscape": "Quiet room tone.",
                "non_diegetic_music": "N/A",
                "preview_resolution": "720p",
                "final_resolution": "720p",
                "model_variant": "lora",
                "sampling_steps": 8,
                "final_sampling_steps": 2,
            },
        )
        self.assertEqual(response.status, 201, await response.text())
        project_id = (await response.json())["id"]
        response = await self.client.post(
            f"/api/v1/infinite-projects/{project_id}/windows",
            headers=headers,
            json={"window_description": "[Shot 1] One same-resolution window."},
        )
        self.assertEqual(response.status, 202, await response.text())
        for _ in range(80):
            project = await (
                await self.client.get(
                    f"/api/v1/infinite-projects/{project_id}", headers=headers
                )
            ).json()
            if project["tail_status"] in {"succeeded", "failed"}:
                break
            await asyncio.sleep(0.01)
        self.assertEqual(project["tail_status"], "succeeded")
        job_id = project["windows"][0]["job_id"]
        job = self.app["job_service"].jobs[job_id]
        self.assertTrue(job.spec.selflift_enabled)
        self.assertEqual(job.spec.resolution, "720p")
        self.assertEqual(job.spec.selflift_initial_resolution, "720p")
        self.assertEqual(job.spec.selflift_transition_step, 6)
        self.assertIsNotNone(job.checkpoint_path)

    async def test_infinite_project_creation_requires_only_a_name(self) -> None:
        headers = {"X-API-Key": "secret"}
        response = await self.client.post(
            "/api/v1/infinite-projects",
            headers=headers,
            json={"title": "只创建项目容器"},
        )
        self.assertEqual(response.status, 201, await response.text())
        project = await response.json()
        self.assertEqual(project["title"], "只创建项目容器")
        self.assertEqual(project["overview"], "")
        self.assertEqual(project["overall_soundscape"], "N/A")
        self.assertEqual(project["non_diegetic_music"], "N/A")
        self.assertEqual(project["window_count"], 0)

    async def test_infinite_fl2va_rejects_reference_only_media(self) -> None:
        headers = {"X-API-Key": "secret"}
        response = await self.client.post(
            "/api/v1/infinite-projects",
            headers=headers,
            json={
                "title": "FL2VA text project",
                "overview": "One continuous text-generated scene.",
                "overall_soundscape": "Quiet room tone.",
                "non_diegetic_music": "N/A",
            },
        )
        self.assertEqual(response.status, 201, await response.text())
        project_id = (await response.json())["id"]
        form = aiohttp.FormData(default_to_multipart=True)
        for name, value in {
            "window_description": "Establish the opening shot.",
            "duration_seconds": "5",
            "resolution": "480p",
            "aspect_ratio": "16:9",
        }.items():
            form.add_field(name, value)
        form.add_field(
            "reference_image_1",
            b"not-a-real-image",
            filename="reference.png",
            content_type="image/png",
        )
        response = await self.client.post(
            f"/api/v1/infinite-projects/{project_id}/windows",
            headers=headers,
            data=form,
        )
        self.assertEqual(response.status, 400, await response.text())
        self.assertIn("use Ref2VA", await response.text())
        project = await (
            await self.client.get(
                f"/api/v1/infinite-projects/{project_id}", headers=headers
            )
        ).json()
        self.assertEqual(project["window_count"], 0)

    async def test_infinite_missing_tail_is_directly_retryable(self) -> None:
        headers = {"X-API-Key": "secret"}
        response = await self.client.post(
            "/api/v1/infinite-projects",
            headers=headers,
            json={
                "title": "Retry missing tail",
                "overview": "One stable room and one stable actor.",
                "overall_soundscape": "Continuous room tone.",
                "non_diegetic_music": "N/A",
            },
        )
        project_id = (await response.json())["id"]

        async def append(description: str) -> str:
            response = await self.client.post(
                f"/api/v1/infinite-projects/{project_id}/windows",
                headers=headers,
                json={
                    "window_description": description,
                    "duration_seconds": 5,
                    "model_variant": "base",
                    "sampling_steps": 20,
                    "acceleration": 0,
                    "seed": 7,
                    "resolution": "480p",
                    "aspect_ratio": "16:9",
                    "memory": 60,
                    "overlap_seconds": 1.625,
                },
            )
            self.assertEqual(response.status, 202, await response.text())
            return (await response.json())["job"]["id"]

        opening_id = await append("Establish one continuous opening shot.")
        for _ in range(60):
            project = await (
                await self.client.get(
                    f"/api/v1/infinite-projects/{project_id}", headers=headers
                )
            ).json()
            if project["tail_status"] == "succeeded":
                break
            await asyncio.sleep(0.01)

        store = self.app["infinite_project_store"]
        internal = store.require(project_id)
        internal.windows.append({
            "index": 1,
            "job_id": "deleted-failed-job",
            "requested_duration_seconds": 5,
            "actual_duration_seconds": 4.958333333,
            "context_frames": 39,
            "overlap_seconds": 1.625,
            "memory": 60,
            "overview": internal.overview,
            "window_description": "The failed tail description.",
            "overall_soundscape": internal.overall_soundscape,
            "non_diegetic_music": internal.non_diegetic_music,
            "model_variant": "base",
            "sampling_steps": 14,
            "acceleration": 65,
            "total_frames": 243,
        })
        store.persist(internal)
        project = await (
            await self.client.get(
                f"/api/v1/infinite-projects/{project_id}", headers=headers
            )
        ).json()
        self.assertEqual(project["tail_status"], "missing")
        self.assertTrue(project["can_retry_tail"])
        self.assertTrue(project["can_append"])

        replacement_id = await append("Retry this tail from the accepted opening.")
        self.assertNotEqual(replacement_id, "deleted-failed-job")
        for _ in range(60):
            project = await (
                await self.client.get(
                    f"/api/v1/infinite-projects/{project_id}", headers=headers
                )
            ).json()
            if project["tail_status"] == "succeeded":
                break
            await asyncio.sleep(0.01)
        self.assertEqual(project["window_count"], 2)
        self.assertEqual(project["windows"][0]["job_id"], opening_id)
        self.assertEqual(project["windows"][1]["job_id"], replacement_id)
        self.assertEqual(
            self.app["job_service"].backend.last_continuation.source_job_id,
            opening_id,
        )

    async def test_latent_cache_clear_keeps_history_and_videos(self) -> None:
        service = self.app["job_service"]
        latent_root = service.output_root / ".h3-latents"
        latent_root.mkdir(parents=True, exist_ok=True)
        latent = latent_root / "cache-test.pt"
        latent.write_bytes(b"reproducible-clean-av-latent")
        checkpoint = service.data_dir / "checkpoints" / "cache-test.pt"
        checkpoint.write_bytes(b"formal-checkpoint-tensor")
        spec = GenerationSpec.from_mapping({
            "prompt": "cache cleanup card", "resolution": "480p",
        })
        job = JobRecord(
            id="cache-test", spec=spec, status="succeeded",
            output_path=self.video, final_latents_path=latent,
            checkpoint_path=checkpoint, checkpoint_retained=True,
        )
        service.jobs[job.id] = job
        service.persist(job)

        response = await self.client.delete(
            "/api/v1/cache/latents", headers={"X-API-Key": "secret"}
        )
        self.assertEqual(response.status, 200, await response.text())
        result = await response.json()
        self.assertEqual(result["removed_files"], 2)
        self.assertEqual(result["removed_checkpoint_files"], 1)
        self.assertFalse(latent.exists())
        self.assertFalse(checkpoint.exists())
        self.assertTrue(self.video.exists())
        self.assertIn(job.id, service.jobs)
        self.assertIsNone(service.jobs[job.id].final_latents_path)
        self.assertIsNone(service.jobs[job.id].checkpoint_path)
        public = service.serialize(service.jobs[job.id])
        self.assertTrue(public["second_sampling_available"])
        self.assertEqual(
            public["second_sampling_methods"], {"temporal": True, "h3": False}
        )

    async def test_w4a8_completed_card_exposes_second_sampling_entry(self) -> None:
        service = self.app["job_service"]
        latent_root = service.output_root / ".h3-latents"
        latent_root.mkdir(parents=True, exist_ok=True)
        latent = latent_root / "w4a8-source.pt"
        latent.write_bytes(b"retained-w4a8-clean-av-latent")
        spec = GenerationSpec.from_mapping({
            "prompt": "8GB selectable card",
            "runtime_launcher": "fl2va_w4a8_8gb",
            "resolution": "480p",
        })
        job = JobRecord(
            id="w4a8-source",
            spec=spec,
            status="succeeded",
            output_path=self.video,
            final_latents_path=latent,
        )
        self.assertTrue(service.serialize(job)["second_sampling_available"])

    async def test_public_second_sampling_uses_1440p_name(self) -> None:
        service = self.app["job_service"]
        source = GenerationSpec.from_mapping({
            "prompt": "public resolution name",
            "resolution": "480p",
        })
        second = SecondSamplingSpec.from_mapping({
            "resolution": "1440p",
            "steps": 1,
        }, source=source)
        job = JobRecord(
            id="public-1440p",
            spec=source,
            status="queued",
            second_sampling=second,
        )
        public = service.serialize(job)
        self.assertEqual(second.resolution, "2k")
        self.assertEqual(public["second_sampling"]["resolution"], "1440p")

    async def test_generation_limits_are_saved_and_drive_options_and_validation(self) -> None:
        headers = {"X-API-Key": "secret"}
        current = await (
            await self.client.get(
                "/api/v1/settings/generation-limits", headers=headers
            )
        ).json()
        limits = current["preset_limits"]
        limits["720p"]["1:1"] = 9
        limits["720p"]["16:9"] = 12
        limits["1080p"]["16:9"] = 10
        response = await self.client.put(
            "/api/v1/settings/generation-limits",
            headers=headers,
            json={"preset_limits": limits},
        )
        self.assertEqual(response.status, 200)
        policy = await response.json()
        self.assertEqual(policy["preset_limits"]["720p"]["1:1"], 9)
        self.assertEqual(policy["preset_limits"]["720p"]["16:9"], 12)
        self.assertEqual(policy["preset_limits"]["1080p"]["16:9"], 10)
        self.assertIn("detected_vram_gib", policy)

        options = await (
            await self.client.get("/api/v1/options", headers=headers)
        ).json()
        self.assertEqual(options["duration"]["max_by_preset"]["720p"]["1:1"], 9)
        self.assertEqual(options["duration"]["max_by_preset"]["720p"]["16:9"], 12)
        self.assertEqual(options["duration"]["max_by_preset"]["1080p"]["16:9"], 10)
        rejected = await self.client.post(
            "/api/v1/generations",
            headers=headers,
            json={
                "prompt": "too long for the configured preset ceiling",
                "resolution": "1080p",
                "aspect_ratio": "16:9",
                "duration_seconds": 10.5,
            },
        )
        self.assertEqual(rejected.status, 400)
        self.assertIn("configured server limit", await rejected.text())
        self.assertTrue(
            (self.temporary / "data/settings/generation_limits.json").is_file()
        )

    async def test_fl2va_generation_forwards_one_raw_prompt_without_enhancement(self) -> None:
        prompt = "  自由文本第一段。\n\noverall_soundscape: 保持这一行原样。  "
        headers = {"X-API-Key": "secret"}
        response = await self.client.post(
            "/api/v1/generations", headers=headers,
            json={"prompt": prompt, "duration_seconds": 5},
        )
        self.assertEqual(response.status, 202, await response.text())
        job_id = (await response.json())["id"]
        for _ in range(50):
            state = await (
                await self.client.get(f"/api/v1/jobs/{job_id}", headers=headers)
            ).json()
            if state["status"] in {"succeeded", "failed"}:
                break
            await asyncio.sleep(0.01)
        self.assertEqual(state["status"], "succeeded")
        self.assertEqual(self.app["job_service"].backend.last_spec.prompt, prompt)

    async def test_1080p_duration_transparently_crosses_the_native_window(self) -> None:
        headers = {"X-API-Key": "secret"}
        long_request = await self.client.post(
            "/api/v1/generations", headers=headers, json={
                "prompt": "transparent long 1080p", "resolution": "1080p",
                "aspect_ratio": "16:9", "duration_seconds": 15.5,
            },
        )
        self.assertEqual(long_request.status, 202, await long_request.text())
        long_spec = (await long_request.json())["request"]
        self.assertTrue(long_spec["long_horizon"])
        self.assertEqual(long_spec["output_frames"], 379)
        self.assertLessEqual(long_spec["frames"], 362)

        accepted = await self.client.post(
            "/api/v1/generations", headers=headers, json={
                "prompt": "validated 1080p", "resolution": "1080p",
                "aspect_ratio": "16:9", "duration_seconds": 15,
            },
        )
        self.assertEqual(accepted.status, 202, await accepted.text())
        request = (await accepted.json())["request"]
        self.assertEqual((request["width"], request["height"]), (1920, 1088))
        self.assertEqual(request["frames"], 362)

        accepted_four_three = await self.client.post(
            "/api/v1/generations", headers=headers, json={
                "prompt": "validated longer 1080p 4:3", "resolution": "1080p",
                "aspect_ratio": "4:3", "duration_seconds": 15,
            },
        )
        self.assertEqual(accepted_four_three.status, 202, await accepted_four_three.text())
        request = (await accepted_four_three.json())["request"]
        self.assertEqual((request["width"], request["height"], request["frames"]),
                         (1440, 1088, 362))

        accepted_square = await self.client.post(
            "/api/v1/generations", headers=headers, json={
                "prompt": "validated longer 1080p square", "resolution": "1080p",
                "aspect_ratio": "1:1", "duration_seconds": 15,
            },
        )
        self.assertEqual(accepted_square.status, 202, await accepted_square.text())

    async def test_2k_first_pass_is_rejected_and_reserved_for_second_sampling(self) -> None:
        response = await self.client.post(
            "/api/v1/generations",
            headers={"X-API-Key": "secret"},
            json={
                "prompt": "experimental full-context 2k route",
                "resolution": "2k",
                "aspect_ratio": "16:9",
                "duration_seconds": 15,
                "memory_mode": "auto",
            },
        )
        self.assertEqual(response.status, 400, await response.text())
        self.assertIn("1080p", await response.text())

    async def test_request_can_hot_switch_variant_inside_fixed_family(self) -> None:
        response = await self.client.post(
            "/api/v1/generations",
            headers={"X-API-Key": "secret"},
            json={"prompt": "hot lora", "model_variant": "lora"},
        )
        self.assertEqual(response.status, 202, await response.text())
        submitted = await response.json()
        self.assertEqual(submitted["request"]["engine"], "lora")

    async def test_preset_request_accepts_direct_steps_and_acceleration(self) -> None:
        response = await self.client.post(
            "/api/v1/generations",
            headers={"X-API-Key": "secret"},
            json={
                "prompt": "preset geometry with direct execution controls",
                "resolution": "480p",
                "aspect_ratio": "16:9",
                "duration_seconds": 5,
                "sampling_steps": 15,
                "acceleration": 60,
            },
        )
        self.assertEqual(response.status, 202, await response.text())
        request = (await response.json())["request"]
        self.assertFalse(request["advanced"])
        self.assertEqual(request["sampling_steps"], 15)
        self.assertEqual(request["acceleration"], 60.0)

    async def test_scheduled_lora_checkpoint_releases_worker_and_can_resume(self) -> None:
        headers = {"X-API-Key": "secret"}
        response = await self.client.post(
            "/api/v1/generations", headers=headers, json={
                "prompt": "scheduled LoRA checkpoint task",
                "service_family": "first_last",
                "model_variant": "lora",
                "mode": "advanced",
                "width": 864,
                "height": 480,
                "duration_seconds": 5,
                "sampling_steps": 8,
                "acceleration": 50,
                "execution_mode": "checkpoint",
                "checkpoint_step": 3,
                "checkpoint_retain": True,
                "checkpoint_preview": False,
            },
        )
        self.assertEqual(response.status, 202, await response.text())
        job_id = (await response.json())["id"]
        for _ in range(100):
            state = await (
                await self.client.get(f"/api/v1/jobs/{job_id}", headers=headers)
            ).json()
            if state["status"] == "checkpointed":
                break
            await asyncio.sleep(0.01)
        self.assertEqual(state["status"], "checkpointed")
        self.assertEqual(state["checkpoint"]["completed_steps"], 3)
        self.assertEqual(state["checkpoint"]["total_steps"], 8)
        self.assertTrue(state["checkpoint"]["resume_available"])
        self.assertNotIn("preview", state)
        self.assertEqual(state["request"]["model_variant"], "lora")
        self.assertEqual(state["request"]["sampling_steps"], 8)
        self.assertEqual(state["request"]["acceleration"], 50.0)

        # A second job is not blocked by the stopped checkpoint task.
        second = await self.client.post(
            "/api/v1/generations", headers=headers, json={"prompt": "next job"}
        )
        second_id = (await second.json())["id"]
        for _ in range(100):
            second_state = await (
                await self.client.get(f"/api/v1/jobs/{second_id}", headers=headers)
            ).json()
            if second_state["status"] == "succeeded":
                break
            await asyncio.sleep(0.01)
        self.assertEqual(second_state["status"], "succeeded")

        resumed = await self.client.post(
            f"/api/v1/jobs/{job_id}/resume", headers=headers
        )
        self.assertEqual(resumed.status, 202, await resumed.text())
        for _ in range(100):
            state = await (
                await self.client.get(f"/api/v1/jobs/{job_id}", headers=headers)
            ).json()
            if state["status"] == "succeeded":
                break
            await asyncio.sleep(0.01)
        self.assertEqual(state["status"], "succeeded")
        self.assertEqual(state["generation_elapsed_seconds"], 1.75)

    async def test_failed_resume_with_retained_checkpoint_can_be_retried(self) -> None:
        headers = {"X-API-Key": "secret"}
        response = await self.client.post(
            "/api/v1/generations", headers=headers, json={
                "prompt": "retry retained formal checkpoint",
                "execution_mode": "checkpoint",
                "checkpoint_step": 3,
                "checkpoint_retain": True,
            },
        )
        job_id = (await response.json())["id"]
        for _ in range(100):
            state = await (
                await self.client.get(f"/api/v1/jobs/{job_id}", headers=headers)
            ).json()
            if state["status"] == "checkpointed":
                break
            await asyncio.sleep(0.01)
        service = self.app["job_service"]
        job = service.jobs[job_id]
        job.status = "failed"
        job.error = "simulated resume failure"
        service.persist(job)
        failed_state = await (
            await self.client.get(f"/api/v1/jobs/{job_id}", headers=headers)
        ).json()
        self.assertTrue(failed_state["checkpoint"]["resume_available"])

        retried = await self.client.post(
            f"/api/v1/jobs/{job_id}/resume", headers=headers
        )
        self.assertEqual(retried.status, 202, await retried.text())

    async def test_single_video_pause_preview_can_continue(self) -> None:
        headers = {"X-API-Key": "secret"}
        response = await self.client.post(
            "/api/v1/generations", headers=headers, json={
                "prompt": "preview branch",
                "preview_mode": "pause",
                "preview_step_index": 5,
                "preview_branch_steps": 2,
                "preview_fast_finish": True,
            },
        )
        self.assertEqual(response.status, 202, await response.text())
        job_id = (await response.json())["id"]
        for _ in range(100):
            state = await (
                await self.client.get(f"/api/v1/jobs/{job_id}", headers=headers)
            ).json()
            if state["status"] == "awaiting_preview":
                break
            await asyncio.sleep(0.01)
        self.assertEqual(state["status"], "awaiting_preview")
        self.assertTrue(state["preview"]["ready"])
        continued = await self.client.post(
            f"/api/v1/jobs/{job_id}/preview/continue", headers=headers
        )
        self.assertEqual(continued.status, 200, await continued.text())
        for _ in range(100):
            state = await (
                await self.client.get(f"/api/v1/jobs/{job_id}", headers=headers)
            ).json()
            if state["status"] == "succeeded":
                break
            await asyncio.sleep(0.01)
        self.assertEqual(state["status"], "succeeded")

    async def test_checkpoint_preview_is_generated_and_resumable(self) -> None:
        headers = {"X-API-Key": "secret"}
        response = await self.client.post(
            "/api/v1/generations", headers=headers, json={
                "prompt": "retired checkpoint preview",
                "execution_mode": "checkpoint",
                "checkpoint_step": 3,
                "checkpoint_preview": True,
            },
        )
        self.assertEqual(response.status, 202, await response.text())
        job_id = (await response.json())["id"]
        for _ in range(100):
            state = await (
                await self.client.get(
                    f"/api/v1/jobs/{job_id}", headers=headers
                )
            ).json()
            if state["status"] == "checkpointed":
                break
            await asyncio.sleep(0.01)
        self.assertEqual(state["status"], "checkpointed")
        self.assertTrue(state["preview"]["ready"])
        self.assertTrue(state["checkpoint"]["resume_available"])

    async def test_checkpoint_preview_defaults_on_for_legacy_console(self) -> None:
        headers = {"X-API-Key": "secret"}
        response = await self.client.post(
            "/api/v1/generations", headers=headers, json={
                "prompt": "legacy console checkpoint",
                "execution_mode": "checkpoint",
                "checkpoint_step": 3,
            },
        )
        self.assertEqual(response.status, 202, await response.text())
        job_id = (await response.json())["id"]
        for _ in range(100):
            state = await (
                await self.client.get(
                    f"/api/v1/jobs/{job_id}", headers=headers
                )
            ).json()
            if state["status"] == "checkpointed":
                break
            await asyncio.sleep(0.01)
        self.assertEqual(state["status"], "checkpointed")
        self.assertTrue(state["request"]["checkpoint_preview"])
        self.assertTrue(state["preview"]["ready"])

    async def test_stale_same_origin_console_false_still_gets_preview(self) -> None:
        origin = str(self.client.make_url("/")).rstrip("/")
        headers = {"X-API-Key": "secret", "Origin": origin}
        response = await self.client.post(
            "/api/v1/generations", headers=headers, json={
                "prompt": "stale console checkpoint",
                "execution_mode": "checkpoint",
                "checkpoint_step": 3,
                "checkpoint_preview": False,
            },
        )
        self.assertEqual(response.status, 202, await response.text())
        job_id = (await response.json())["id"]
        for _ in range(100):
            state = await (
                await self.client.get(
                    f"/api/v1/jobs/{job_id}",
                    headers={"X-API-Key": "secret"},
                )
            ).json()
            if state["status"] == "checkpointed":
                break
            await asyncio.sleep(0.01)
        self.assertEqual(state["status"], "checkpointed")
        self.assertTrue(state["request"]["checkpoint_preview"])
        self.assertTrue(state["preview"]["ready"])

    async def test_multipart_console_without_origin_cannot_disable_preview(self) -> None:
        form = aiohttp.FormData(default_to_multipart=True)
        for name, value in {
            "prompt": "embedded browser checkpoint",
            "execution_mode": "checkpoint",
            "checkpoint_step": "3",
            "checkpoint_preview": "false",
            "checkpoint_preview_steps": "4",
            "checkpoint_preview_resolution": "360p",
        }.items():
            form.add_field(name, value)
        response = await self.client.post(
            "/api/v1/generations",
            headers={"X-API-Key": "secret"},
            data=form,
        )
        self.assertEqual(response.status, 202, await response.text())
        job_id = (await response.json())["id"]
        for _ in range(100):
            state = await (
                await self.client.get(
                    f"/api/v1/jobs/{job_id}",
                    headers={"X-API-Key": "secret"},
                )
            ).json()
            if state["status"] == "checkpointed":
                break
            await asyncio.sleep(0.01)
        self.assertEqual(state["status"], "checkpointed")
        self.assertTrue(state["request"]["checkpoint_preview"])
        self.assertEqual(state["request"]["checkpoint_preview_steps"], 2)
        self.assertEqual(
            state["request"]["checkpoint_preview_resolution"], "360p"
        )
        self.assertTrue(state["preview"]["ready"])

    async def test_checkpoint_preview_global_settings_round_trip(self) -> None:
        headers = {"X-API-Key": "secret"}
        response = await self.client.put(
            "/api/v1/settings/checkpoint-preview",
            headers=headers,
            json={"steps": 4, "resolution": "480p"},
        )
        self.assertEqual(response.status, 200, await response.text())
        self.assertEqual(
            await response.json(),
            {
                "steps": 4,
                "resolution": "480p",
                "step_range": {"min": 1, "max": 4},
                "resolutions": ["360p", "480p", "720p"],
            },
        )
        response = await self.client.post(
            "/api/v1/generations",
            headers=headers,
            json={
                "prompt": "global checkpoint preview defaults",
                "execution_mode": "checkpoint",
                "checkpoint_step": 3,
                "checkpoint_preview": True,
            },
        )
        self.assertEqual(response.status, 202, await response.text())
        self.assertEqual(
            self.app["job_service"].backend.last_spec.checkpoint_preview_steps,
            4,
        )
        self.assertEqual(
            self.app["job_service"].backend.last_spec.checkpoint_preview_resolution,
            "480p",
        )

    async def test_second_sampling_temporal_window_settings_are_shared(self) -> None:
        headers = {"X-API-Key": "secret"}
        response = await self.client.put(
            "/api/v1/settings/second-sampling-window",
            headers=headers,
            json={
                "enabled": True,
                "window_seconds": 5.0,
                "overlap_seconds": 2.0,
            },
        )
        self.assertEqual(response.status, 200, await response.text())
        policy = await response.json()
        self.assertTrue(policy["enabled"])
        self.assertEqual(policy["window_seconds"], 5.0)
        self.assertEqual(policy["overlap_seconds"], 2.0)
        self.assertEqual(policy["effective_window_frames"], 124)
        self.assertEqual(policy["effective_stride_frames"], 51)
        self.assertEqual(policy["effective_overlap_frames"], 73)
        self.assertEqual(policy["effective_overlap_seconds"], 3.042)
        self.assertEqual(
            json.loads(
                (self.temporary / "data/settings/second_sampling_window.json")
                .read_text(encoding="utf-8")
            ),
            {
                "enabled": True,
                "window_seconds": 5.0,
                "overlap_seconds": 2.0,
            },
        )

        response = await self.client.post(
            "/api/v1/generations",
            headers=headers,
            json={
                "prompt": "windowed progressive generation",
                "resolution": "720p",
                "aspect_ratio": "1:1",
                "duration_seconds": 5,
                "model_variant": "lora",
                "sampling_steps": 8,
                "acceleration": 60,
                "second_pass_acceleration": 70,
                "selflift_enabled": True,
                "selflift_initial_resolution": "540p",
                "selflift_transition_step": 6,
            },
        )
        self.assertEqual(response.status, 202, await response.text())
        request = (await response.json())["request"]
        self.assertTrue(request["selflift_temporal_window_enabled"])
        self.assertEqual(request["selflift_temporal_window_seconds"], 5.0)
        self.assertEqual(request["selflift_temporal_overlap_seconds"], 2.0)

        response = await self.client.post(
            "/api/v1/generations",
            headers=headers,
            json={
                "prompt": "full-timeline progressive generation",
                "resolution": "720p",
                "aspect_ratio": "1:1",
                "duration_seconds": 5,
                "model_variant": "lora",
                "sampling_steps": 8,
                "acceleration": 60,
                "second_pass_acceleration": 70,
                "selflift_enabled": True,
                "selflift_initial_resolution": "540p",
                "selflift_transition_step": 6,
                "selflift_temporal_window_enabled": False,
            },
        )
        self.assertEqual(response.status, 202, await response.text())
        unwindowed = (await response.json())["request"]
        self.assertFalse(unwindowed["selflift_temporal_window_enabled"])

        response = await self.client.post(
            "/api/v1/generations",
            headers=headers,
            json={
                "prompt": "windowed low-drift progressive generation",
                "resolution": "720p",
                "aspect_ratio": "1:1",
                "duration_seconds": 5,
                "model_variant": "lora",
                "sampling_steps": 8,
                "acceleration": 60,
                "second_pass_acceleration": 70,
                "selflift_enabled": True,
                "selflift_initial_resolution": "540p",
                "selflift_transition_step": 6,
                "selflift_temporal_window_enabled": True,
                "selflift_temporal_window_seconds": 8.0,
                "selflift_temporal_overlap_seconds": 0.0,
                "selflift_sigma_scale": 0.65,
            },
        )
        self.assertEqual(response.status, 202, await response.text())
        windowed = (await response.json())["request"]
        self.assertTrue(windowed["selflift_temporal_window_enabled"])
        # Physical geometry comes from Service settings, not the task body.
        self.assertEqual(windowed["selflift_temporal_window_seconds"], 5.0)
        self.assertEqual(windowed["selflift_temporal_overlap_seconds"], 2.0)
        self.assertEqual(windowed["selflift_sigma_scale"], 0.65)

        maximum = await self.client.put(
            "/api/v1/settings/second-sampling-window",
            headers=headers,
            json={"enabled": True, "window_seconds": 15},
        )
        self.assertEqual(maximum.status, 200, await maximum.text())
        maximum_policy = await maximum.json()
        self.assertEqual(maximum_policy["window_seconds"], 15.0)
        self.assertEqual(maximum_policy["effective_window_frames"], 362)

        invalid = await self.client.put(
            "/api/v1/settings/second-sampling-window",
            headers=headers,
            json={"enabled": True, "window_seconds": 15.5},
        )
        self.assertEqual(invalid.status, 400)
        invalid_overlap = await self.client.put(
            "/api/v1/settings/second-sampling-window",
            headers=headers,
            json={"enabled": True, "window_seconds": 5, "overlap_seconds": 4.1},
        )
        self.assertEqual(invalid_overlap.status, 400)

    async def test_legacy_upscale_request_points_to_native_second_sampling(self) -> None:
        response = await self.client.post(
            "/api/v1/generations", headers={"X-API-Key": "secret"}, json={
                "prompt": "upscaled scene", "seed": 7,
                "upscale_enabled": True, "upscale_mode": "basic",
                "upscale_resolution": "1080p",
            },
        )
        self.assertEqual(response.status, 400)
        self.assertIn("native H3 second sampling", await response.text())

    async def test_default_service_reports_missing_temporal_runtime(self) -> None:
        status = self.app["job_service"].upscaler.status()
        self.assertFalse(status["ready"])
        self.assertEqual(status["resident_state"], "unavailable")
        self.assertIn("scripts/install.sh", status["remediation"])

    async def test_prompt_polishing_routes_are_removed(self) -> None:
        studio = await self.client.post(
            "/studio/prompt-enhancements",
            headers={"X-API-Key": "secret"},
            data={"storyboard": "{}"},
        )
        settings = await self.client.get(
            "/api/v1/settings/mimo-key", headers={"X-API-Key": "secret"}
        )
        self.assertEqual(studio.status, 404)
        self.assertEqual(settings.status, 404)

    async def test_resource_snapshot_exposes_host_and_gpu_contract(self) -> None:
        response = await self.client.get(
            "/api/v1/resources", headers={"X-API-Key": "secret"}
        )
        self.assertEqual(response.status, 200)
        document = await response.json()
        self.assertIn("cpu", document)
        self.assertIn("memory", document)
        self.assertIn("service_memory", document)
        self.assertIn("gpu", document)
        self.assertIn("queue", document)
        self.assertGreater(document["memory"]["total_gib"], 0)
        self.assertIn("available_gib", document["memory"])
        self.assertIn("occupied_gib", document["memory"])
        self.assertIn("reclaimable_gib", document["memory"])
        self.assertGreaterEqual(
            document["memory"]["occupied_gib"], document["memory"]["used_gib"]
        )
        self.assertEqual(document["memory"]["scope"], "linux_host")
        self.assertIn("used_gib", document["service_memory"])
        self.assertIn("limit_gib", document["service_memory"])
        self.assertIn("resident_gib", document["service_memory"])
        self.assertEqual(document["service_memory"]["resident_metric"], "pss")
        self.assertNotEqual(
            document["service_memory"].get("scope"), "whole_machine"
        )

    async def test_queue_reorder_and_record_delete_contract(self) -> None:
        headers = {"X-API-Key": "secret"}
        # Stop the worker so both jobs remain reorderable.
        service = self.app["job_service"]
        service.worker_task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await service.worker_task
        service.worker_task = None

        ids = []
        for seed in (1, 2):
            response = await self.client.post(
                "/api/v1/generations", headers=headers,
                json={"prompt": f"queued {seed}", "seed": seed},
            )
            self.assertEqual(response.status, 202)
            ids.append((await response.json())["id"])
        response = await self.client.put(
            "/api/v1/jobs/order", headers={**headers, "Content-Type": "application/json"},
            json={"job_ids": list(reversed(ids))},
        )
        self.assertEqual(response.status, 200)
        self.assertEqual((await response.json())["job_ids"], list(reversed(ids)))

        response = await self.client.delete(
            f"/api/v1/jobs/{ids[0]}/record", headers=headers
        )
        self.assertEqual(response.status, 200)
        response = await self.client.get(f"/api/v1/jobs/{ids[0]}", headers=headers)
        self.assertEqual(response.status, 404)

    async def test_record_delete_retains_untrusted_legacy_output(self) -> None:
        """A legacy result outside output/ must not make its card undeletable."""

        headers = {"X-API-Key": "secret"}
        service = self.app["job_service"]
        legacy_output = self.temporary / "runtime" / "old-smoke" / "result.mp4"
        legacy_output.parent.mkdir(parents=True)
        legacy_output.write_bytes(b"legacy-result")
        job = JobRecord(
            id="legacy-output-job",
            spec=GenerationSpec.from_mapping({"prompt": "legacy", "seed": 7}),
            status="succeeded",
            output_path=legacy_output,
        )
        service.jobs[job.id] = job
        service.cancel_events[job.id] = asyncio.Event()
        service.persist(job)

        response = await self.client.delete(
            f"/api/v1/jobs/{job.id}/record", headers=headers
        )
        self.assertEqual(response.status, 200)
        document = await response.json()
        self.assertTrue(document["deleted"])
        self.assertFalse(document["output_deleted"])
        self.assertTrue(document["output_retained"])
        self.assertTrue(legacy_output.is_file())
        self.assertNotIn(job.id, service.jobs)
        self.assertFalse((service.data_dir / "jobs" / f"{job.id}.json").exists())

    async def test_batch_record_delete_is_deduplicated_and_best_effort(self) -> None:
        headers = {"X-API-Key": "secret"}
        service = self.app["job_service"]
        service.output_root.mkdir(parents=True, exist_ok=True)

        deleted_jobs = []
        deleted_outputs = []
        for index in range(2):
            output = service.output_root / f"batch-{index}.mp4"
            output.write_bytes(f"batch-{index}".encode())
            job = JobRecord(
                id=f"batch-delete-{index}",
                spec=GenerationSpec.from_mapping({
                    "prompt": f"batch {index}", "seed": index + 20,
                }),
                status="succeeded",
                output_path=output,
            )
            service.jobs[job.id] = job
            service.cancel_events[job.id] = asyncio.Event()
            service.persist(job)
            deleted_jobs.append(job)
            deleted_outputs.append(output)

        active = JobRecord(
            id="batch-active",
            spec=GenerationSpec.from_mapping({"prompt": "active", "seed": 30}),
            status="running",
        )
        service.jobs[active.id] = active
        service.cancel_events[active.id] = asyncio.Event()
        service.persist(active)

        response = await self.client.delete(
            "/api/v1/jobs/records",
            headers=headers,
            json={"job_ids": [
                deleted_jobs[0].id,
                deleted_jobs[1].id,
                deleted_jobs[0].id,
                "missing-batch-job",
                active.id,
            ]},
        )
        self.assertEqual(response.status, 200, await response.text())
        document = await response.json()
        self.assertEqual(document["requested_count"], 4)
        self.assertEqual(document["deleted_count"], 2)
        self.assertEqual(document["deleted_ids"], [job.id for job in deleted_jobs])
        self.assertEqual(
            {item["id"] for item in document["errors"]},
            {"missing-batch-job", active.id},
        )
        self.assertTrue(all(not output.exists() for output in deleted_outputs))
        self.assertTrue(all(job.id not in service.jobs for job in deleted_jobs))
        self.assertIn(active.id, service.jobs)

    async def test_batch_record_delete_rejects_empty_or_oversized_selection(self) -> None:
        headers = {"X-API-Key": "secret"}
        for job_ids in ([], [f"job-{index}" for index in range(101)]):
            response = await self.client.delete(
                "/api/v1/jobs/records", headers=headers, json={"job_ids": job_ids}
            )
            self.assertEqual(response.status, 400)

    async def test_persisted_fifteen_second_job_round_trips(self) -> None:
        data = self.temporary / "roundtrip"
        backend = FakeBackend(self.video)
        service = JobService(data, backend)
        original = JobRecord(
            id="roundtrip-job",
            spec=GenerationSpec.from_mapping({
                "prompt": "fifteen seconds", "duration_seconds": 15, "seed": 1
            }),
            inference_plan={
                "policy_id": "h3_v19_human_aligned_budgeted_adaptive_inference",
                "accelerated": True,
                "execution_digest": "a" * 64,
            },
        )
        service.jobs[original.id] = original
        service.persist(original)
        restored = JobService(data, backend).jobs[original.id]
        self.assertEqual(restored.spec.requested_duration_seconds, 15)
        self.assertEqual(restored.spec.frames, 362)
        self.assertEqual(restored.inference_plan, original.inference_plan)
        self.assertEqual(
            service.serialize(original)["inference_plan"],
            original.inference_plan,
        )

    async def test_eta_counts_down_and_includes_jobs_ahead(self) -> None:
        service = self.app["job_service"]
        if service.worker_task is not None:
            service.worker_task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await service.worker_task
            service.worker_task = None

        spec = GenerationSpec.from_mapping({"prompt": "eta", "seed": 1})
        running = JobRecord(
            id="eta-running", spec=spec, status="running",
            estimated_total_seconds=40.0, started_at=time.time() - 10.0,
        )
        first = JobRecord(
            id="eta-first", spec=spec, estimated_total_seconds=20.0,
            estimated_remaining_seconds=20.0,
        )
        second = JobRecord(
            id="eta-second", spec=spec, estimated_total_seconds=25.0,
            estimated_remaining_seconds=25.0,
        )
        service.jobs.update({job.id: job for job in (running, first, second)})
        service.pending[:] = [first.id, second.id]

        running_progress = service.serialize(running)["progress"]
        second_progress = service.serialize(second)["progress"]
        self.assertAlmostEqual(running_progress["estimated_remaining_seconds"], 30, delta=1)
        self.assertAlmostEqual(second_progress["estimated_queue_seconds"], 50, delta=1)
        self.assertAlmostEqual(second_progress["estimated_completion_seconds"], 75, delta=1)

    async def test_eta_history_matches_execution_parameters(self) -> None:
        service = self.app["job_service"]
        sparse = GenerationSpec.from_mapping({
            "prompt": "sparse history", "advanced": True, "width": 864,
            "height": 480, "frames": 124, "actual_steps": 9,
            "attention_keep_ratio": 0.75, "sparse_scope": "middle_only", "seed": 1,
        })
        dense = GenerationSpec.from_mapping({
            "prompt": "dense request", "advanced": True, "width": 864,
            "height": 480, "frames": 124, "actual_steps": 9,
            "attention_keep_ratio": 1.0, "sparse_scope": "middle_only", "seed": 2,
        })
        service.jobs["sparse-history"] = JobRecord(
            id="sparse-history", spec=sparse, status="succeeded", elapsed_seconds=9.0,
        )
        self.assertNotEqual(service._estimate_total(dense, "text"), 9.0)
        self.assertEqual(service._estimate_total(sparse, "text"), 9.0)

    async def test_starting_backend_eta_waits_for_model_preload(self) -> None:
        spec = GenerationSpec.from_mapping({"prompt": "wait for preload", "seed": 3})
        job = JobRecord(
            id="preload-eta", spec=spec, status="starting_backend",
            estimated_total_seconds=35.0, estimated_remaining_seconds=35.0,
            started_at=None,
        )
        progress = self.app["job_service"].serialize(job)["progress"]
        self.assertIsNone(progress["estimated_remaining_seconds"])
        self.assertIsNone(progress["estimated_completion_seconds"])


class UnifiedConsoleApiTest(AioHTTPTestCase):
    async def get_application(self):
        self.temporary = Path(tempfile.mkdtemp(prefix="h3serve-unified-test-"))
        self.video = self.temporary / "unified.mp4"
        self.video.write_bytes(b"unified-video")
        paths = ServicePaths.defaults(self.temporary, data_dir=self.temporary / "data")
        self.backend = FakeBackend(self.video)
        return create_app(
            paths=paths, serve_dir=Path(__file__).resolve().parents[1],
            backend=self.backend, fixed_engine=None, preload=False,
        )

    async def asyncTearDown(self) -> None:
        await super().asyncTearDown()
        shutil.rmtree(self.temporary, ignore_errors=True)

    async def test_enter_exit_and_switch_engine(self) -> None:
        options = await (await self.client.get("/api/v1/options")).json()
        self.assertEqual(options["deployment_mode"], "unified_console")
        self.assertIsNone(options["current_engine"])
        response = await self.client.post(
            "/api/v1/generations", json={"prompt": "must select first"}
        )
        self.assertEqual(response.status, 400)

        response = await self.client.put("/api/v1/engine", json={"engine": "lora"})
        self.assertEqual(response.status, 200, await response.text())
        self.assertEqual(self.backend.preloaded, "fl2va_int8_24gb")
        options = await (await self.client.get("/api/v1/options")).json()
        self.assertEqual(options["current_engine"], "first_last")
        self.assertEqual(options["current_model_variant"], "lora")
        self.assertEqual(options["defaults"]["quality"], "quality")

        response = await self.client.delete("/api/v1/engine")
        self.assertEqual(response.status, 200, await response.text())
        self.assertIsNone(self.backend.preloaded)
        options = await (await self.client.get("/api/v1/options")).json()
        self.assertIsNone(options["current_engine"])

        response = await self.client.put("/api/v1/engine", json={"engine": "reference"})
        self.assertEqual(response.status, 200, await response.text())
        self.assertEqual(self.backend.preloaded, "ref2va_int8_24gb")

        response = await self.client.delete("/api/v1/engine")
        self.assertEqual(response.status, 200, await response.text())
        response = await self.client.put("/api/v1/engine", json={"engine": "reference_lora"})
        self.assertEqual(response.status, 200, await response.text())
        self.assertEqual(self.backend.preloaded, "ref2va_int8_24gb")

        response = await self.client.delete("/api/v1/engine")
        self.assertEqual(response.status, 200, await response.text())
        response = await self.client.put(
            "/api/v1/engine",
            json={"launcher": "fl2va_w4a8_8gb", "model_variant": "lora"},
        )
        self.assertEqual(response.status, 200, await response.text())
        self.assertEqual(self.backend.preloaded, "fl2va_w4a8_8gb")
        options = await (await self.client.get("/api/v1/options")).json()
        self.assertEqual(options["current_launcher"], "fl2va_w4a8_8gb")
        self.assertEqual(options["active_weight_tier"], "w4a8")
        self.assertEqual(options["resolutions"], ["360p", "480p", "540p", "720p"])
        self.assertTrue(
            options["advanced_limits"]["second_sampling"]["available"]
        )
        self.assertEqual(
            options["advanced_limits"]["second_sampling"]["levels"],
            ["720p", "900p", "1080p"],
        )

        response = await self.client.delete("/api/v1/engine")
        self.assertEqual(response.status, 200, await response.text())
        response = await self.client.put(
            "/api/v1/engine",
            json={"launcher": "ref2va_int8_16gb"},
        )
        self.assertEqual(response.status, 200, await response.text())
        options = await (await self.client.get("/api/v1/options")).json()
        self.assertEqual(options["active_vram_profile"], "16gb")
        self.assertEqual(
            options["resolutions"],
            ["360p", "480p", "540p", "720p", "900p", "1080p"],
        )
        self.assertEqual(
            options["advanced_limits"]["second_sampling"]["levels"],
            ["720p", "900p", "1080p", "1220p", "1440p"],
        )

    async def test_four_product_choices_auto_route_vram_and_compile_ram_budget(self) -> None:
        options = await (await self.client.get("/api/v1/options")).json()
        self.assertEqual(
            set(options["model_choices"]),
            {"fl2va_w4a8", "ref2va_w4a8", "fl2va_int8", "ref2va_int8"},
        )
        self.assertEqual(
            options["host_memory"]["budget_ranges"]["w4a8"]["minimum_gib"],
            12,
        )
        self.assertEqual(
            options["host_memory"]["budget_ranges"]["int8"]["minimum_gib"],
            24,
        )
        response = await self.client.put(
            "/api/v1/engine",
            json={
                "service_family": "first_last",
                "weight_tier": "w4a8",
                "host_memory_limit_gib": 12,
            },
        )
        self.assertEqual(response.status, 200, await response.text())
        # The development 4090 is detected as 24GB, but that tier is internal.
        self.assertEqual(self.backend.preloaded, "fl2va_w4a8_24gb")
        options = await (await self.client.get("/api/v1/options")).json()
        self.assertEqual(options["active_vram_profile"], "24gb")
        self.assertEqual(options["active_weight_tier"], "w4a8")
        self.assertEqual(
            options["resolutions"],
            ["360p", "480p", "540p", "720p", "900p", "1080p"],
        )
        self.assertEqual(options["host_memory"]["profile"]["process_limit_gib"], 12)
        self.assertEqual(
            options["host_memory"]["profile"]["evidence"],
            "experimental_low_memory",
        )

        response = await self.client.delete("/api/v1/engine")
        self.assertEqual(response.status, 200, await response.text())
        response = await self.client.put(
            "/api/v1/engine",
            json={
                "service_family": "reference",
                "weight_tier": "int8",
                "host_memory_limit_gib": 24,
            },
        )
        self.assertEqual(response.status, 200, await response.text())
        # INT8 uses exactly the same product contract: VRAM is internal and
        # the selected RAM value becomes the service-process ceiling.
        self.assertEqual(self.backend.preloaded, "ref2va_int8_24gb")
        options = await (await self.client.get("/api/v1/options")).json()
        self.assertEqual(options["active_vram_profile"], "24gb")
        self.assertEqual(options["active_weight_tier"], "int8")
        self.assertEqual(options["host_memory"]["profile"]["process_limit_gib"], 24)
        self.assertEqual(options["host_memory"]["profile"]["evidence"], "experimental_low_memory")
        self.assertEqual(options["host_memory"]["enforcement"]["limit_gib"], 24)
        resources = await (await self.client.get("/api/v1/resources")).json()
        self.assertEqual(resources["service_memory"]["limit_gib"], 24)
        self.assertNotEqual(resources["service_memory"]["scope"], "whole_machine")

    async def test_16gb_completed_card_can_queue_1440p_second_sampling(self) -> None:
        response = await self.client.put(
            "/api/v1/engine", json={"launcher": "ref2va_int8_16gb"}
        )
        self.assertEqual(response.status, 200, await response.text())
        service = self.app["job_service"]
        latent = service.output_root / ".h3-latents" / "ref16-source.pt"
        latent.parent.mkdir(parents=True, exist_ok=True)
        latent.write_bytes(b"ref16-clean-av-latent")
        source = JobRecord(
            id="ref16-source",
            spec=GenerationSpec.from_mapping({
                "prompt": "16GB 1440P second-sampling release boundary",
                "runtime_launcher": "ref2va_int8_16gb",
                "service_family": "reference",
                "resolution": "480p",
                "duration_seconds": 15,
            }),
            status="succeeded",
            output_path=self.video,
            final_latents_path=latent,
        )
        service.jobs[source.id] = source
        response = await self.client.post(
            f"/api/v1/jobs/{source.id}/second-sampling",
            json={
                "resolution": "1440p",
                "steps": 1,
                "acceleration": 100,
                "strength": "preserve",
            },
        )
        self.assertEqual(response.status, 202, await response.text())
        child = await response.json()
        self.assertEqual(child["second_sampling"]["resolution"], "1440p")
        self.assertEqual(
            (
                child["second_sampling"]["width"],
                child["second_sampling"]["height"],
            ),
            (2560, 1440),
        )

    async def test_busy_queue_blocks_engine_exit(self) -> None:
        await self.client.put("/api/v1/engine", json={"engine": "original"})
        service = self.app["job_service"]
        spec = GenerationSpec.from_mapping({"prompt": "queued", "seed": 7})
        service.jobs["queued"] = JobRecord(id="queued", spec=spec, status="queued")
        service.pending.append("queued")
        response = await self.client.delete("/api/v1/engine")
        self.assertEqual(response.status, 409)
        self.assertIn("queued", await response.text())

    async def test_workspace_switch_isolates_history_and_storage(self) -> None:
        options = await (await self.client.get("/api/v1/options")).json()
        default_root = self.temporary / "workspace" / "default"
        self.assertEqual(Path(options["workspace"]["current"]["path"]), default_root)
        self.assertTrue(options["workspace"]["switchable"])

        project_a = self.temporary / "creative-project-a"
        response = await self.client.put(
            "/api/v1/workspace", json={"path": str(project_a)}
        )
        self.assertEqual(response.status, 200, await response.text())
        service = self.app["job_service"]
        self.assertEqual(service.data_dir, project_a / ".x-minimax-h3")
        self.assertEqual(service.output_root, project_a / "outputs")
        spec = GenerationSpec.from_mapping({"prompt": "workspace A", "seed": 17})
        service.jobs["workspace-a-job"] = JobRecord(
            id="workspace-a-job", spec=spec, status="failed", error="fixture"
        )
        service.persist(service.jobs["workspace-a-job"])

        project_b = self.temporary / "creative-project-b"
        response = await self.client.put(
            "/api/v1/workspace", json={"path": str(project_b)}
        )
        self.assertEqual(response.status, 200, await response.text())
        self.assertNotIn("workspace-a-job", service.jobs)
        self.assertTrue((project_b / "outputs").is_dir())
        self.assertTrue((project_b / ".x-minimax-h3/checkpoints").is_dir())

        response = await self.client.put(
            "/api/v1/workspace", json={"path": str(project_a)}
        )
        self.assertEqual(response.status, 200, await response.text())
        self.assertIn("workspace-a-job", service.jobs)

    async def test_workspace_cannot_switch_with_loaded_engine(self) -> None:
        await self.client.put("/api/v1/engine", json={"engine": "original"})
        response = await self.client.put(
            "/api/v1/workspace", json={"path": str(self.temporary / "blocked")}
        )
        self.assertEqual(response.status, 409)
        self.assertIn("exit", await response.text())


class TurboApiTest(AioHTTPTestCase):
    async def get_application(self):
        self.temporary = Path(tempfile.mkdtemp(prefix="h3serve-turbo-test-"))
        self.video = self.temporary / "turbo.mp4"
        self.video.write_bytes(b"turbo-video")
        serve_dir = Path(__file__).resolve().parents[1]
        paths = ServicePaths.defaults(self.temporary, data_dir=self.temporary / "data")
        return create_app(
            paths=paths,
            serve_dir=serve_dir,
            backend=FakeBackend(self.video),
            fixed_engine="lora",
        )

    async def asyncTearDown(self) -> None:
        await super().asyncTearDown()
        shutil.rmtree(self.temporary, ignore_errors=True)

    async def test_turbo_process_injects_lora_and_six_step_default(self) -> None:
        response = await self.client.get("/api/v1/options")
        self.assertEqual(response.status, 200)
        options = await response.json()
        self.assertEqual(options["current_engine"], "first_last")
        self.assertEqual(options["current_model_variant"], "lora")
        self.assertEqual(options["defaults"]["quality"], "quality")

        response = await self.client.post(
            "/api/v1/generations", json={"prompt": "Turbo six-step default", "seed": 9}
        )
        self.assertEqual(response.status, 202)
        job = await response.json()
        self.assertEqual(job["request"]["engine"], "lora")
        self.assertEqual(job["request"]["quality"], "quality")


class ReferenceApiTest(AioHTTPTestCase):
    async def get_application(self):
        self.temporary = Path(tempfile.mkdtemp(prefix="h3serve-reference-test-"))
        self.video = self.temporary / "reference.mp4"
        self.video.write_bytes(b"reference-video")
        serve_dir = Path(__file__).resolve().parents[1]
        paths = ServicePaths.defaults(self.temporary, data_dir=self.temporary / "data")
        self.backend = FakeBackend(self.video)
        return create_app(
            paths=paths,
            serve_dir=serve_dir,
            backend=self.backend,
            fixed_engine="reference",
        )

    async def asyncTearDown(self) -> None:
        await super().asyncTearDown()
        shutil.rmtree(self.temporary, ignore_errors=True)

    async def test_reference_process_requires_and_forwards_reference_images(self) -> None:
        response = await self.client.post(
            "/api/v1/generations", json={"prompt": "missing reference", "seed": 1}
        )
        self.assertEqual(response.status, 400)
        self.assertIn("requires at least one", await response.text())

        form = aiohttp.FormData()
        image_buffer = io.BytesIO()
        Image.new("RGB", (32, 32), (120, 60, 30)).save(image_buffer, format="PNG")
        form.add_field("prompt", "keep the reference identity")
        form.add_field("seed", "2")
        form.add_field("reference_image_resolution", "original")
        form.add_field("reference_video_resolution", "480p")
        form.add_field(
            "reference_image_1", image_buffer.getvalue(),
            filename="identity.png", content_type="image/png",
        )
        response = await self.client.post("/api/v1/generations", data=form)
        self.assertEqual(response.status, 202)
        job = await response.json()
        self.assertEqual(job["request"]["engine"], "reference")
        self.assertEqual(job["request"]["quality"], "balanced")
        self.assertEqual(job["request"]["condition_mode"], "reference")
        self.assertEqual(job["request"]["reference_image_count"], 1)
        self.assertEqual(job["request"]["reference_image_resolution"], "original")
        self.assertEqual(job["request"]["reference_video_resolution"], "480p")

        for _ in range(50):
            state = await (
                await self.client.get(f"/api/v1/jobs/{job['id']}")
            ).json()
            if state["status"] in {"succeeded", "failed"}:
                break
            await asyncio.sleep(0.01)
        self.assertEqual(state["status"], "succeeded")
        self.assertEqual(len(self.backend.reference_images), 1)
        self.assertEqual(self.backend.reference_images[0].name, "reference_image_1.png")

    async def test_v3_json_project_loads_opening_reference_id_path_mapping(self) -> None:
        reference = self.temporary / "panorama.png"
        Image.new("RGB", (64, 40), (70, 90, 120)).save(reference, format="PNG")
        response = await self.client.post(
            "/api/v1/infinite-projects",
            json={
                "workflow_version": 3,
                "creation_mode": "json",
                "title": "Reference one-click film",
                "overview": "Keep <Picture 1> as the scene and character anchor.",
                "overall_soundscape": "A continuous room tone.",
                "non_diegetic_music": "N/A",
                "preview_resolution": "540p",
                "final_resolution": "1080p",
                "model_variant": "base",
                "sampling_steps": 5,
            },
        )
        self.assertEqual(response.status, 201, await response.text())
        project_id = (await response.json())["id"]
        response = await self.client.post(
            f"/api/v1/infinite-projects/{project_id}/batch",
            json={
                "references": {"Picture 1": {"path": str(reference)}},
                "final_acceleration": 64,
                "windows": [
                    {
                        "prompt": "[Shot 1] Establish <Picture 1> exactly.",
                        "duration_seconds": 2,
                        "acceleration": 30,
                    },
                    "[Shot 1] Continue the exact same shot and motion.",
                ],
            },
        )
        self.assertEqual(response.status, 202, await response.text())
        for _ in range(180):
            project = await (
                await self.client.get(
                    f"/api/v1/infinite-projects/{project_id}"
                )
            ).json()
            if project["batch"]["status"] in {"completed", "failed"}:
                break
            await asyncio.sleep(0.02)
        self.assertEqual(
            project["batch"]["status"],
            "completed",
            project["batch"].get("error"),
        )
        self.assertEqual(project["batch"]["reference_ids"], ["Picture 1"])
        self.assertNotIn(str(reference), json.dumps(project))
        self.assertEqual(project["window_count"], 2)
        self.assertEqual(len(self.backend.reference_images), 1)
        self.assertEqual(self.backend.reference_images[0].name, "reference_image_1.png")
        first = self.app["job_service"].jobs[project["windows"][0]["job_id"]]
        self.assertEqual(first.spec.acceleration, 30)
        self.assertEqual(first.spec.second_pass_acceleration, 30)
        self.assertIn("<Picture 1>", first.spec.prompt)
        jobs = [
            self.app["job_service"].jobs[item["job_id"]]
            for item in project["windows"]
        ]
        for job in jobs:
            self.assertTrue(job.spec.selflift_enabled)
            self.assertEqual(job.spec.resolution, "1080p")
            self.assertEqual(job.spec.selflift_initial_resolution, "540p")
            self.assertEqual(job.spec.execution_mode, "checkpoint")
            self.assertIsNotNone(job.spec.checkpoint_step)
            self.assertFalse(job.spec.checkpoint_preview)
            self.assertIsNotNone(job.checkpoint_path)
        self.assertFalse(project["window_controls_editable"])
        self.assertFalse(project["second_sampling_available"])
        self.assertEqual(
            project["final_generation_method"], "global_sliding_selflift"
        )
        self.assertNotEqual(project["final_sampling"]["job_id"], jobs[-1].id)
        self.assertEqual(project["final_sampling"]["status"], "succeeded")
        self.assertEqual(
            project["final_sampling"]["settings"]["method"],
            "global_sliding_selflift",
        )
        self.assertTrue(
            project["final_sampling"]["settings"]["shared_preview_prefix"]
        )
        self.assertEqual(project["final_sampling"]["settings"]["acceleration"], 64)
        final_job = self.app["job_service"].jobs[
            project["final_sampling"]["job_id"]
        ]
        self.assertEqual(final_job.spec.second_pass_acceleration, 64)
        self.assertIsNotNone(project["final_sampling"]["video_url"])

        # Older clients may still call final-sampling after a JSON batch. It
        # returns the existing tail and must not enqueue the online fork path.
        response = await self.client.post(
            f"/api/v1/infinite-projects/{project_id}/final-sampling",
            json={},
        )
        self.assertEqual(response.status, 200, await response.text())
        self.assertEqual(
            (await response.json())["id"],
            project["final_sampling"]["job_id"],
        )
        self.assertEqual(len(self.backend.last_infinite_selflift_sources), 2)

    async def test_v3_ref_json_supports_independent_references_per_window(self) -> None:
        references = []
        for index, colour in enumerate(((170, 20, 30), (20, 170, 40), (30, 50, 190)), start=1):
            path = self.temporary / f"window-{index}.png"
            Image.new("RGB", (64, 40), colour).save(path, format="PNG")
            references.append(path)
        response = await self.client.post(
            "/api/v1/infinite-projects",
            json={
                "workflow_version": 3,
                "creation_mode": "json",
                "service_family": "reference",
                "title": "Per-window Ref2VA references",
                "preview_resolution": "540p",
                "final_resolution": "720p",
                "model_variant": "base",
                "sampling_steps": 5,
            },
        )
        self.assertEqual(response.status, 201, await response.text())
        project_id = (await response.json())["id"]
        response = await self.client.post(
            f"/api/v1/infinite-projects/{project_id}/batch",
            json={
                "windows": [
                    {
                        "prompt": "[Shot 1] Establish <Picture 1>.",
                        "duration_seconds": 2,
                        "references": {
                            "Picture 1": {"path": str(references[0])},
                        },
                    },
                    {
                        "prompt": (
                            "[Shot 1] Continue using this window's "
                            "<Picture 1> and <Picture 2>."
                        ),
                        "duration_seconds": 2,
                        "references": {
                            "Picture 1": {"path": str(references[1])},
                            "Picture 2": {"path": str(references[2])},
                        },
                    },
                    {
                        "prompt": (
                            "[Shot 1] Continue with the inherited "
                            "<Picture 1> and <Picture 2>."
                        ),
                        "duration_seconds": 2,
                    },
                ],
            },
        )
        self.assertEqual(response.status, 202, await response.text())
        for _ in range(240):
            project = await (
                await self.client.get(
                    f"/api/v1/infinite-projects/{project_id}"
                )
            ).json()
            if project["batch"]["status"] in {"completed", "failed"}:
                break
            await asyncio.sleep(0.02)
        self.assertEqual(
            project["batch"]["status"],
            "completed",
            project["batch"].get("error"),
        )
        jobs = [
            self.app["job_service"].jobs[item["job_id"]]
            for item in project["windows"]
        ]
        self.assertEqual(
            [len(job.reference_images) for job in jobs],
            [1, 2, 2],
        )
        self.assertEqual(
            [Image.open(path).getpixel((0, 0)) for path in jobs[0].reference_images],
            [(170, 20, 30)],
        )
        expected_replacement = [(20, 170, 40), (30, 50, 190)]
        for job in jobs[1:]:
            self.assertEqual(
                [Image.open(path).getpixel((0, 0)) for path in job.reference_images],
                expected_replacement,
            )
        self.assertEqual(
            [item["reference_ids"] for item in project["batch"]["plan"]],
            [
                ["Picture 1"],
                ["Picture 1", "Picture 2"],
                ["Picture 1", "Picture 2"],
            ],
        )
        serialized = json.dumps(project)
        for path in references:
            self.assertNotIn(str(path), serialized)

    async def test_v3_ref_json_rejects_invalid_window_reference_sets(self) -> None:
        reference = self.temporary / "reference.png"
        Image.new("RGB", (64, 40), (60, 80, 100)).save(reference, format="PNG")
        invalid_sets = (
            ({}, "requires Picture or Audio references"),
            (
                {"Picture 2": {"path": str(reference)}},
                "Picture IDs must be contiguous from 1",
            ),
        )
        for invalid_references, expected_error in invalid_sets:
            with self.subTest(references=invalid_references):
                response = await self.client.post(
                    "/api/v1/infinite-projects",
                    json={
                        "workflow_version": 3,
                        "creation_mode": "json",
                        "service_family": "reference",
                        "title": "Invalid per-window references",
                        "preview_resolution": "540p",
                        "final_resolution": "720p",
                        "model_variant": "base",
                        "sampling_steps": 5,
                    },
                )
                self.assertEqual(response.status, 201, await response.text())
                project_id = (await response.json())["id"]
                response = await self.client.post(
                    f"/api/v1/infinite-projects/{project_id}/batch",
                    json={
                        "windows": [{
                            "prompt": "[Shot 1] Establish the opening.",
                            "references": invalid_references,
                        }],
                    },
                )
                self.assertEqual(response.status, 400)
                self.assertIn(expected_error, await response.text())

    async def test_reference_generation_forwards_one_raw_prompt_without_enhancement(self) -> None:
        prompt = "  subject_definitions:\n<Subject 1> from <Picture 1>.\n  "
        image_buffer = io.BytesIO()
        Image.new("RGB", (32, 32), (10, 20, 30)).save(image_buffer, format="PNG")
        form = aiohttp.FormData()
        form.add_field("prompt", prompt)
        form.add_field(
            "reference_image_1", image_buffer.getvalue(),
            filename="identity.png", content_type="image/png",
        )
        response = await self.client.post("/api/v1/generations", data=form)
        self.assertEqual(response.status, 202)
        job = await response.json()
        for _ in range(50):
            state = await (
                await self.client.get(f"/api/v1/jobs/{job['id']}")
            ).json()
            if state["status"] in {"succeeded", "failed"}:
                break
            await asyncio.sleep(0.01)
        self.assertEqual(state["status"], "succeeded")
        self.assertEqual(self.backend.last_spec.prompt, prompt)

    async def test_reference_media_console_policy_is_shared_default(self) -> None:
        response = await self.client.put(
            "/api/v1/settings/reference-media",
            json={"image_resolution": "480p", "video_resolution": "720p"},
        )
        self.assertEqual(response.status, 200)
        policy = await response.json()
        self.assertEqual(policy["image_resolution"], "480p")
        self.assertEqual(policy["video_resolution"], "720p")
        self.assertTrue(policy["preserve_aspect_ratio"])
        self.assertFalse(policy["crop"])
        self.assertFalse(policy["stretch"])
        persisted = json.loads(
            (self.temporary / "data/settings/reference_media.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(persisted, {
            "image_resolution": "480p",
            "video_resolution": "720p",
        })

        options = await (await self.client.get("/api/v1/options")).json()
        self.assertEqual(options["defaults"]["reference_image_resolution"], "480p")
        self.assertEqual(options["defaults"]["reference_video_resolution"], "720p")

        image_buffer = io.BytesIO()
        Image.new("RGB", (48, 32), (30, 90, 120)).save(
            image_buffer, format="PNG"
        )
        form = aiohttp.FormData()
        form.add_field("prompt", "inherit the global reference policy")
        form.add_field(
            "reference_image_1", image_buffer.getvalue(),
            filename="identity.png", content_type="image/png",
        )
        response = await self.client.post("/api/v1/generations", data=form)
        self.assertEqual(response.status, 202)
        job = await response.json()
        self.assertEqual(job["request"]["reference_image_resolution"], "480p")
        self.assertEqual(job["request"]["reference_video_resolution"], "720p")

        bad = await self.client.put(
            "/api/v1/settings/reference-media",
            json={"image_resolution": "1080p"},
        )
        self.assertEqual(bad.status, 400)

    async def test_reference_video_is_persisted_and_forwarded(self) -> None:
        buffer = io.BytesIO()
        with av.open(buffer, "w", format="mp4") as container:
            stream = container.add_stream("mpeg4", rate=24)
            stream.width, stream.height, stream.pix_fmt = 64, 48, "yuv420p"
            for index in range(48):
                pixels = np.full((48, 64, 3), index, dtype=np.uint8)
                for packet in stream.encode(av.VideoFrame.from_ndarray(pixels, format="rgb24")):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        form = aiohttp.FormData()
        form.add_field("prompt", "continue <Video 1>")
        form.add_field("seed", "3")
        form.add_field("reference_video_1", buffer.getvalue(), filename="motion.mp4", content_type="video/mp4")
        response = await self.client.post("/api/v1/generations", data=form)
        self.assertEqual(response.status, 202, await response.text())
        self.assertEqual(len(self.backend.reference_videos), 1)
        self.assertEqual(self.backend.reference_videos[0].name, "reference_video_1.mp4")

    async def test_reference_audio_is_persisted_and_forwarded(self) -> None:
        buffer = io.BytesIO()
        with av.open(buffer, "w", format="wav") as container:
            stream = container.add_stream("pcm_s16le", rate=48000)
            stream.layout = "mono"
            frame = av.AudioFrame.from_ndarray(
                np.zeros((1, 4800), dtype=np.float32), format="flt", layout="mono"
            )
            frame.sample_rate = 48000
            for packet in stream.encode(frame):
                container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        form = aiohttp.FormData()
        form.add_field("prompt", "use <Audio 1> as the speaker voice")
        form.add_field("seed", "4")
        form.add_field(
            "reference_audio_1", buffer.getvalue(), filename="voice.wav",
            content_type="audio/wav",
        )
        response = await self.client.post("/api/v1/generations", data=form)
        self.assertEqual(response.status, 202, await response.text())
        job = await response.json()
        self.assertEqual(job["request"]["reference_audio_count"], 1)
        for _ in range(50):
            state = await (await self.client.get(f"/api/v1/jobs/{job['id']}")).json()
            if state["status"] in {"succeeded", "failed"}:
                break
            await asyncio.sleep(0.01)
        self.assertEqual(state["status"], "succeeded")
        self.assertEqual(self.backend.reference_audios[0].name, "reference_audio_1.wav")


if __name__ == "__main__":
    unittest.main()
