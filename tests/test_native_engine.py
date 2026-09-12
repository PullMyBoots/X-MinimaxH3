from __future__ import annotations

import asyncio
import dataclasses
import json
import shutil
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
from types import SimpleNamespace

from h3serve.backend import NativeBackendManager
from h3serve.contract import GenerationSpec, SecondSamplingSpec
from h3serve.infinite_video import InfiniteContinuationSpec
from h3serve.native_engine import NativeH3Engine, NativeHotH3Engine
from h3serve.native_engine.engine import (
    _bind_generated_voice_reference,
    _public_inference_plan,
)
from h3serve.native_engine.hot_session import (
    HotSessionRequest,
    NativeT2AVHotSession,
    _continuation_boundary_forecast_controller,
)


class FakePipeline:
    def __init__(self) -> None:
        self.last_request = None
        self.closed = False

    def generate(self, request, *, cancel_check):
        if cancel_check():
            raise AssertionError("test request was unexpectedly cancelled")
        self.last_request = request
        request.output_path.write_bytes(b"native-video")
        return SimpleNamespace(
            result=request.output_path,
            metrics=SimpleNamespace(elapsed_seconds={"denoise": 0.5}),
        )

    def close(self) -> None:
        self.closed = True


class FakeHotSession:
    def __init__(self) -> None:
        self.requests = []
        self.closed = False
        self.runtime_config = SimpleNamespace(
            resource_profile="int8_24gb",
            max_device_bytes=int(23.25 * 1024**3),
        )

    def generate(self, request):
        self.requests.append(request)
        request.output_path.write_bytes(b"hot-native-video")
        if request.save_final_latents_path is not None:
            request.save_final_latents_path.parent.mkdir(parents=True, exist_ok=True)
            request.save_final_latents_path.write_bytes(b"clean-av-latent")
        return SimpleNamespace(
            output_path=request.output_path,
            phases={"denoise": 0.1},
            execution_profile={
                "joint_acceleration": request.acceleration_plan_summary,
                "memory_execution": {
                    "requested_mode": request.memory_mode,
                    "selected_scheme": "low_vram",
                },
            },
        )

    def _device_execution_budget_bytes(self):
        return 23 * 1024**3

    def close(self):
        self.closed = True


class FakeCheckpointHotSession(FakeHotSession):
    def generate(self, request):
        from h3serve.native_engine.hot_session import HotSessionCheckpointResult

        self.requests.append(request)
        if request.checkpoint_after_step is not None:
            checkpoint_path = Path(request.checkpoint_state_path)
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            checkpoint_path.write_bytes(b"formal-lora-checkpoint")
            preview_path = request.preview_output_path
            if preview_path is not None:
                preview_path.write_bytes(b"lora-preview")
            preview_latents_path = request.preview_latents_path
            if preview_latents_path is not None:
                preview_latents_path.parent.mkdir(parents=True, exist_ok=True)
                preview_latents_path.write_bytes(b"low-resolution-clean-av")
            return HotSessionCheckpointResult(
                checkpoint_path=checkpoint_path,
                preview_path=preview_path,
                completed_steps=request.checkpoint_after_step,
                total_steps=request.steps,
                total_seconds=0.1,
                phases={"denoise": 0.1},
                step_seconds=(0.03,) * request.checkpoint_after_step,
                execution_profile={
                    "joint_acceleration": request.acceleration_plan_summary,
                    "formal_checkpoint": {
                        "completed_steps": request.checkpoint_after_step,
                        "formal_trajectory_mutated": False,
                    },
                },
                preview_latents_path=preview_latents_path,
            )
        request.output_path.write_bytes(b"resumed-lora-video")
        return SimpleNamespace(
            output_path=request.output_path,
            phases={"denoise": 0.1},
            execution_profile={
                "joint_acceleration": request.acceleration_plan_summary,
                "formal_resume": {
                    "formal_prefix_replayed": False,
                },
            },
        )


class FakeHotFactory:
    sparse_attention_available = False

    def __init__(self) -> None:
        self.builds = []
        self.sessions = []

    def build(self, family):
        self.builds.append(family)
        session = FakeHotSession()
        self.sessions.append(session)
        return SimpleNamespace(
            session=session, startup_seconds=0.1, qwen_storage="source",
            weight_tier="int8", vram_profile="24gb",
        )

    def preflight(self, _family):
        return {"ready": True, "checks": {"fake": True}}


class FakeProgressHotFactory(FakeHotFactory):
    def __init__(self) -> None:
        super().__init__()
        self.progress_callback = None
        self.callback_history = []

    def set_progress_callback(self, callback) -> None:
        self.progress_callback = callback
        self.callback_history.append(callback)

    def build(self, family):
        if self.progress_callback is not None:
            self.progress_callback(42, "model_graphs", "模型组件已准备 2/5")
        return super().build(family)


class FakeSparseHotFactory(FakeHotFactory):
    sparse_attention_available = True


class FakeV19HotFactory(FakeSparseHotFactory):
    v19_release_enabled = True


class FakeW4A8EightGibFactory(FakeSparseHotFactory):
    def build(self, family):
        self.builds.append(family)
        session = FakeHotSession()
        session.runtime_config = SimpleNamespace(
            resource_profile="w4a8_8gb",
            max_device_bytes=int(7.25 * 1024**3),
        )
        self.sessions.append(session)
        return SimpleNamespace(
            session=session, startup_seconds=0.1, qwen_storage="source",
            weight_tier="w4a8", vram_profile="8gb",
        )


class FakeCheckpointHotFactory(FakeSparseHotFactory):
    def build(self, family):
        self.builds.append(family)
        session = FakeCheckpointHotSession()
        self.sessions.append(session)
        return SimpleNamespace(
            session=session, startup_seconds=0.1, qwen_storage="source",
            weight_tier="int8", vram_profile="24gb",
        )


class NativeEngineBoundaryTest(unittest.IsolatedAsyncioTestCase):
    def test_exact_token_selector_composes_two_acceleration_stages(self) -> None:
        class Selector:
            def select(self, *, workload, acceleration, required_actual_step_indices=()):
                action = (
                    "frontier:sparse_topk_0.5"
                    if acceleration == 25
                    else "frontier:sparse_topk_0.25"
                )
                return SimpleNamespace(
                    actual_step_indices=tuple(range(int(workload.steps))),
                    attention_action_schedule=tuple(
                        (step, 0, action) for step in range(int(workload.steps))
                    ),
                    summary={"acceleration": acceleration},
                )

        harness = SimpleNamespace(
            v19_selector=Selector(),
            _uses_reference_layout=False,
            _analyze_request_features=lambda request, text_tokens: SimpleNamespace(
                packed_tokens=4096,
                condition_count=0,
            ),
        )
        request = HotSessionRequest(
            prompt="two stage selector",
            seed=1,
            width=864,
            height=480,
            frames=124,
            fps=24,
            steps=8,
            output_path=self.temporary / "two-stage.mp4",
            v19_acceleration=25,
            v19_second_pass_acceleration=75,
            acceleration_transition_step=6,
        )
        selected = NativeT2AVHotSession._apply_v19_selection(
            harness, request, text_tokens=128
        )
        actions = {
            step: action
            for step, _layer, action in selected.attention_action_schedule
        }
        self.assertEqual(
            [actions[index] for index in range(6)],
            ["frontier:sparse_topk_0.5"] * 6,
        )
        self.assertEqual(
            [actions[index] for index in range(6, 8)],
            ["frontier:sparse_topk_0.25"] * 2,
        )
        self.assertEqual(
            selected.acceleration_plan_summary["acceleration_transition_step"],
            6,
        )

    def test_continuation_boundary_auxiliary_mirrors_accelerated_steps(self) -> None:
        actual = (0, 1, 2, 3, 4, 8, 11, 15, 18, 19)
        controller = _continuation_boundary_forecast_controller(actual, 20)
        self.assertIsNotNone(controller)
        assert controller is not None
        self.assertEqual(controller.actual_steps, frozenset(actual))
        self.assertIsNone(
            _continuation_boundary_forecast_controller(tuple(range(20)), 20)
        )

    def test_generated_voice_reference_binding_preserves_story_and_sound_sections(self) -> None:
        source = (
            "integrated_multimodal_description: [Shot 1] At 00:02.000, "
            "(S1) says: <d>[Chinese] 你好。</d>\n\n"
            "overall_soundscape: Quiet room tone.\n\n"
            "non_diegetic_music: N/A"
        )
        bound = _bind_generated_voice_reference(source)
        self.assertIn(
            "<Audio 1> is the voice-timbre reference for the continuing "
            "generated speaker (S1).",
            bound,
        )
        self.assertIn("<d>[Chinese] 你好。</d>", bound)
        self.assertIn("overall_soundscape: Quiet room tone.", bound)
        self.assertIn("non_diegetic_music: N/A", bound)
        self.assertIn("without copying its words, timing, ambience", bound)

    async def asyncSetUp(self) -> None:
        self.temporary = Path(tempfile.mkdtemp(prefix="native-engine-test-"))
        self.pipeline = FakePipeline()
        self.engine = NativeH3Engine(self.pipeline, self.temporary)
        self.manager = NativeBackendManager(self.engine)

    async def asyncTearDown(self) -> None:
        await self.manager.stop()
        shutil.rmtree(self.temporary, ignore_errors=True)

    def test_public_inference_receipt_keeps_memory_route_with_joint_plan(self) -> None:
        receipt = _public_inference_plan({
            "joint_acceleration": {"policy_id": "v24", "accelerated": True},
            "memory_execution": {
                "requested_mode": "auto",
                "selected_scheme": "low_vram",
                "reason": "whole_query_exceeds_device_budget",
            },
            "roi_difficulty_selection": {
                "policy": "latent_difficulty_selector_v1",
                "selected_regions": [
                    {"top": 0.2, "left": 0.4, "height": 0.2, "width": 0.2}
                ],
            },
            "roi_atlas_refinement": {
                "policy": "magnified_region_atlas_h3_refinement_v1",
                "region_count": 1,
                "attention_route": "scheduled_sparse",
            },
            "qwen_conditioning_cache": {
                "schema_version": 1,
                "status": "checkpoint_hit",
                "fallback": None,
                "persisted_with_latent": True,
            },
            "output_mux": {
                "encoder": {
                    "audio_normalization": {
                        "policy": "std5_peak0p9_shape_preserving_v2",
                        "hard_clipped_samples": 0,
                    }
                },
                "media": {"audio_codec": "aac"},
            },
            "audio_manifold_guard": {
                "policy": "speech_local_audio_vae_manifold_guard_v1",
                "applied": False,
                "acoustic_risk_blocks": 0,
                "prompt_or_timeline_inspected": False,
            },
            "audio_window_decode": {
                "policy": "window_local_audio_vae_pcm_overlap_save_v1",
                "temporal_vae_domains": 3,
                "audio_latent_interpolation": False,
                "window_records": [],
            },
            "long_horizon": {
                "continuation_video_handoff": {
                    "policy": "agreement_gated_same_time_repaint_v1",
                    "active": True,
                    "active_window_indices": [1, 2],
                    "window_records": [
                        {
                            "window_index": 1,
                            "context_frames": 90,
                            "protected_video_prefix_frames": 73,
                            "hidden_video_repaint_frames": 17,
                        },
                        {
                            "window_index": 2,
                            "context_frames": 90,
                            "protected_video_prefix_frames": 73,
                            "hidden_video_repaint_frames": 17,
                        },
                    ],
                    "repaint_replaces_same_time_predecessor_tail": True,
                    "extra_qwen_conditioning_encodes": 0,
                    "extra_dit_calls": 0,
                },
                "window_profiles": [{"private": True}],
            },
            "private_debug_payload": {"large": "not public"},
        })
        self.assertEqual(receipt["policy_id"], "v24")
        self.assertEqual(
            receipt["memory_execution"]["selected_scheme"], "low_vram"
        )
        self.assertEqual(
            receipt["roi_difficulty_selection"]["selected_regions"][0]["top"],
            0.2,
        )
        self.assertEqual(
            receipt["roi_atlas_refinement"]["attention_route"],
            "scheduled_sparse",
        )
        self.assertEqual(
            receipt["qwen_conditioning_cache"]["status"], "checkpoint_hit"
        )
        self.assertEqual(
            receipt["output_mux"]["encoder"]["audio_normalization"]["policy"],
            "std5_peak0p9_shape_preserving_v2",
        )
        self.assertEqual(
            receipt["output_mux"]["encoder"]["audio_normalization"]
            ["hard_clipped_samples"],
            0,
        )
        self.assertFalse(receipt["audio_manifold_guard"]["applied"])
        self.assertFalse(
            receipt["audio_manifold_guard"]["prompt_or_timeline_inspected"]
        )
        self.assertEqual(
            receipt["audio_window_decode"]["temporal_vae_domains"], 3
        )
        self.assertFalse(
            receipt["audio_window_decode"]["audio_latent_interpolation"]
        )
        handoff = receipt["long_horizon"]["continuation_video_handoff"]
        self.assertEqual(handoff["active_window_indices"], [1, 2])
        self.assertEqual(handoff["extra_qwen_conditioning_encodes"], 0)
        self.assertEqual(handoff["extra_dit_calls"], 0)
        self.assertNotIn("window_profiles", receipt["long_horizon"])
        self.assertNotIn("private_debug_payload", receipt)

    async def test_hot_engine_reports_real_loading_progress_and_finishes_ready(self) -> None:
        factory = FakeProgressHotFactory()
        engine = NativeHotH3Engine(factory, output_root=self.temporary)
        try:
            await engine.preload("fl2va_int8_24gb")
            self.assertEqual(engine.warm_state["status"], "ready")
            self.assertEqual(engine.warm_state["progress_percent"], 100.0)
            self.assertEqual(engine.warm_state["progress_stage"], "ready")
            self.assertEqual(engine.warm_state["progress_detail"], "模型引擎已就绪")
            self.assertEqual(len(factory.callback_history), 2)
            self.assertTrue(callable(factory.callback_history[0]))
            self.assertIsNone(factory.callback_history[1])
        finally:
            await engine.close()

    def test_public_inference_receipt_can_report_memory_route_without_v24(self) -> None:
        receipt = _public_inference_plan({
            "memory_execution": {
                "requested_mode": "low_vram",
                "selected_scheme": "low_vram",
            }
        })
        self.assertEqual(
            receipt,
            {"memory_execution": {
                "requested_mode": "low_vram",
                "selected_scheme": "low_vram",
            }},
        )

    async def test_original_quality_maps_to_explicit_schedule(self) -> None:
        spec = GenerationSpec.from_mapping({
            "prompt": "fixed native contract",
            "engine": "original",
            "quality": "balanced",
            "duration_seconds": 5,
            "seed": 4404,
        })
        result = await self.manager.generate(
            spec, "job-1", None, None, (), (), (), asyncio.Event()
        )
        self.assertEqual(result.output_path.read_bytes(), b"native-video")
        self.assertEqual(
            self.pipeline.last_request.sampling.actual_step_indices,
            (0, 1, 2, 3, 4, 8, 12, 16, 19),
        )
        self.assertEqual(self.pipeline.last_request.sampling.engine, "original")

    async def test_lora_route_preserves_distilled_step_count(self) -> None:
        spec = GenerationSpec.from_mapping({
            "prompt": "fixed native turbo contract",
            "engine": "lora",
            "quality": "quality",
            "duration_seconds": 5,
            "seed": 8833,
        })
        await self.manager.generate(spec, "job-2", None, None, (), (), (), asyncio.Event())
        sampling = self.pipeline.last_request.sampling
        self.assertEqual(sampling.engine, "lora")
        self.assertEqual(sampling.num_steps, 6)
        self.assertEqual(sampling.sampler, "turbo")
        self.assertEqual(sampling.lora_strength, 1.0)

    def test_hot_engine_only_latency_routes_calibrated_step_presets(self) -> None:
        original_balanced = GenerationSpec.from_mapping({
            "prompt": "balanced",
            "engine": "original",
            "quality": "balanced",
            "seed": 1,
        })
        original_quality = GenerationSpec.from_mapping({
            "prompt": "quality",
            "engine": "original",
            "quality": "quality",
            "seed": 2,
        })
        lora_quality = GenerationSpec.from_mapping({
            "prompt": "lora six",
            "engine": "lora",
            "quality": "quality",
            "seed": 3,
        })
        lora_fast = GenerationSpec.from_mapping({
            "prompt": "lora four",
            "engine": "lora",
            "quality": "fast",
            "seed": 4,
        })
        original_advanced_balanced = GenerationSpec.from_mapping({
            "prompt": "advanced balanced", "engine": "original", "advanced": True,
            "width": 864, "height": 480, "frames": 124,
            "actual_steps": 9, "seed": 5,
        })
        lora_advanced_six = GenerationSpec.from_mapping({
            "prompt": "advanced six", "engine": "lora", "advanced": True,
            "width": 864, "height": 480, "frames": 124,
            "lora_steps": 6, "seed": 6,
        })
        self.assertIsNone(NativeHotH3Engine._request_plan(original_balanced))
        self.assertIsNotNone(NativeHotH3Engine._request_plan(original_quality))
        self.assertIsNone(NativeHotH3Engine._request_plan(lora_quality))
        self.assertIsNotNone(NativeHotH3Engine._request_plan(lora_fast))
        self.assertIsNone(NativeHotH3Engine._request_plan(original_advanced_balanced))
        self.assertIsNone(NativeHotH3Engine._request_plan(lora_advanced_six))

    async def test_hot_engine_reuses_family_session_across_base_lora_base(self) -> None:
        factory = FakeHotFactory()
        engine = NativeHotH3Engine(factory, output_root=self.temporary)
        try:
            specs = [
                GenerationSpec.from_mapping({
                    "prompt": "base one", "service_family": "first_last",
                    "model_variant": "base", "seed": 1,
                }),
                GenerationSpec.from_mapping({
                    "prompt": "lora", "service_family": "first_last",
                    "model_variant": "lora", "quality": "quality", "seed": 2,
                }),
                GenerationSpec.from_mapping({
                    "prompt": "base two", "service_family": "first_last",
                    "model_variant": "base", "seed": 3,
                }),
            ]
            for index, spec in enumerate(specs):
                await engine.generate(
                    spec, None, None, (), (), (), asyncio.Event(),
                    self.temporary / f"hot-{index}.mp4",
                )
            self.assertEqual(factory.builds, ["fl2va_int8_24gb"])
            self.assertEqual(
                [request.use_lora for request in factory.sessions[0].requests],
                [False, True, False],
            )
        finally:
            await engine.close()

    async def test_hot_engine_applies_two_control_joint_plan_without_rebuild(self) -> None:
        factory = FakeSparseHotFactory()
        engine = NativeHotH3Engine(factory, output_root=self.temporary)
        try:
            spec = GenerationSpec.from_mapping({
                "prompt": "joint schedule",
                "engine": "original",
                "mode": "advanced",
                "width": 1280,
                "height": 736,
                "duration_seconds": 15,
                "sampling_steps": 20,
                "acceleration": 100,
                "seed": 82303,
            })
            await engine.generate(
                spec, None, None, (), (), (), asyncio.Event(),
                self.temporary / "joint.mp4",
            )
            request = factory.sessions[0].requests[-1]
            self.assertEqual(request.steps, 20)
            self.assertEqual(
                request.actual_step_indices,
                (0, 1, 2, 3, 4, 8, 12, 15, 18, 19),
            )
            self.assertEqual(len(request.attention_action_schedule), 10 * 50 + 10 * 3)
            self.assertEqual(
                request.acceleration_plan_summary["acceleration"], 100.0
            )
            self.assertEqual(
                request.acceleration_plan_summary["scheduler_family"],
                "h3_int8_frozen_round229",
            )
            self.assertTrue(request.execution_plan.fused_rms_adaln)
            self.assertTrue(request.execution_plan.vae_transformer_block_compile)
            self.assertEqual(factory.builds, ["fl2va_int8_24gb"])
        finally:
            await engine.close()

    async def test_8gb_long_hidden_capacity_guard_disables_all_sparse_runtime_state(self) -> None:
        factory = FakeW4A8EightGibFactory()
        engine = NativeHotH3Engine(factory, output_root=self.temporary)
        try:
            spec = GenerationSpec.from_mapping({
                "prompt": "8GB long hidden capacity boundary",
                "runtime_launcher": "fl2va_w4a8_8gb",
                "resolution": "720p",
                "aspect_ratio": "16:9",
                "duration_seconds": 15,
                "sampling_steps": 5,
                "acceleration": 95,
                "seed": 82303,
            })
            await engine.generate(
                spec, None, None, (), (), (), asyncio.Event(),
                self.temporary / "8gb-capacity.mp4",
            )
            request = factory.sessions[0].requests[-1]
            self.assertEqual(request.actual_step_indices, tuple(range(5)))
            self.assertEqual(request.attention_action_schedule, ())
            self.assertIsNone(request.attention_online_guard_id)
            self.assertEqual(request.attention_online_budget_dense_layers, 0.0)
            self.assertEqual(request.attention_online_rebate_schedule, ())
            self.assertEqual(
                request.acceleration_plan_summary["reason"],
                "w4a8_8gb_long_hidden_approximation_capacity_guard",
            )
        finally:
            await engine.close()

    async def test_hot_engine_routes_lora_to_no_forecast_sparse_scheduler(self) -> None:
        # Even when a Base-only V19 bundle is installed, LoRA remains in its
        # own no-forecast scheduling domain.
        factory = FakeV19HotFactory()
        engine = NativeHotH3Engine(factory, output_root=self.temporary)
        try:
            spec = GenerationSpec.from_mapping({
                "prompt": "LoRA scheduled trajectory",
                "service_family": "first_last",
                "model_variant": "lora",
                "mode": "advanced",
                "width": 864,
                "height": 480,
                "duration_seconds": 5,
                "sampling_steps": 8,
                "acceleration": 50,
                "seed": 82416,
            })
            await engine.generate(
                spec, None, None, (), (), (), asyncio.Event(),
                self.temporary / "lora-joint.mp4",
            )
            request = factory.sessions[0].requests[-1]
            self.assertTrue(request.use_lora)
            self.assertEqual(request.steps, 8)
            self.assertEqual(request.actual_step_indices, tuple(range(8)))
            self.assertEqual(len(request.attention_action_schedule), 8 * 50)
            self.assertIsNone(request.v19_acceleration)
            self.assertEqual(
                request.acceleration_plan_summary["forecast_evaluations"], 0
            )
            self.assertFalse(
                request.acceleration_plan_summary["forecast_allowed"]
            )
            self.assertEqual(
                request.acceleration_plan_summary["scheduler_family"],
                "h3_lora_v1_no_forecast_round229",
            )
            self.assertEqual(
                request.acceleration_plan_summary["model_variant"], "lora"
            )
        finally:
            await engine.close()

    async def test_lora_composes_first_and_second_pass_acceleration(self) -> None:
        factory = FakeSparseHotFactory()
        engine = NativeHotH3Engine(factory, output_root=self.temporary)
        try:
            spec = GenerationSpec.from_mapping({
                "prompt": "LoRA two-stage acceleration",
                "service_family": "first_last",
                "model_variant": "lora",
                "width": 864,
                "height": 480,
                "duration_seconds": 5,
                "sampling_steps": 8,
                "acceleration": 0,
                "second_pass_acceleration": 100,
                "acceleration_transition_step": 6,
                "seed": 82417,
            })
            await engine.generate(
                spec, None, None, (), (), (), asyncio.Event(),
                self.temporary / "lora-two-stage.mp4",
            )
            request = factory.sessions[0].requests[-1]
            summary = request.acceleration_plan_summary
            self.assertEqual(summary["policy_id"], "h3_stage_acceleration_v1")
            self.assertEqual(summary["acceleration"], 0.0)
            self.assertEqual(summary["second_pass_acceleration"], 100.0)
            self.assertEqual(summary["acceleration_transition_step"], 6)
            self.assertTrue(all(
                step >= 6 for step, _layer, _action
                in request.attention_action_schedule
            ))
        finally:
            await engine.close()

    async def test_hot_engine_second_sampling_is_exact_step_and_preserves_audio(self) -> None:
        factory = FakeV19HotFactory()
        engine = NativeHotH3Engine(factory, output_root=self.temporary)
        source_latent = self.temporary / "source.pt"
        source_latent.write_bytes(b"source-clean-av")
        final_latent = self.temporary / "second.pt"
        try:
            target = GenerationSpec.from_mapping({
                "prompt": "same H3 conditioning",
                "engine": "original",
                "mode": "advanced",
                "width": 1920,
                "height": 1088,
                "frames": 124,
                "duration_seconds": 124 / 24,
                "actual_steps": 20,
                "seed": 12,
                "memory_mode": "auto",
            })
            second = SecondSamplingSpec(
                resolution="1080p", width=1920, height=1088,
                steps=1, acceleration=75.0, denoise=0.2,
                memory_mode="auto",
            )
            result = await engine.generate(
                target, None, None, (), (), (), asyncio.Event(),
                self.temporary / "second.mp4",
                final_latents_path=final_latent,
                second_sampling=second,
                refinement_latents_path=source_latent,
            )
            request = factory.sessions[0].requests[-1]
            self.assertEqual(request.steps, 1)
            self.assertEqual(request.actual_step_indices, (0,))
            self.assertIsNone(request.v19_acceleration)
            self.assertEqual(request.refinement_latents_path, source_latent)
            self.assertEqual(
                request.conditioning_cache_source_path, source_latent.resolve()
            )
            self.assertEqual(request.refinement_denoise, 0.2)
            self.assertEqual(request.refinement_spatial_mode, "learned_3d")
            self.assertTrue(request.preserve_refinement_audio)
            self.assertEqual(result.final_latents_path, final_latent)
            self.assertTrue(result.inference_plan["ultimate_upscale"]["full_canvas"])
            self.assertEqual(
                result.inference_plan["ultimate_upscale"]["redundancy_ratio"],
                1.0,
            )
            solver = result.inference_plan["second_sampling_solver"]
            self.assertEqual(solver["model_variant"], "base")
            self.assertEqual(solver["sampler"], "sa_solver")
            self.assertEqual(solver["scheduler"], "simple")
            self.assertAlmostEqual(solver["start_sigma"], 0.6)
            self.assertFalse(solver["forecast_enabled"])
        finally:
            await engine.close()

    async def test_hot_engine_accepts_completed_pixel_video_as_repair_source(self) -> None:
        factory = FakeV19HotFactory()
        engine = NativeHotH3Engine(factory, output_root=self.temporary)
        source_video = self.temporary / "repair-atlas.mp4"
        source_video.write_bytes(b"completed-pixel-atlas")
        try:
            target = GenerationSpec.from_mapping({
                "prompt": "source-preserving local restoration",
                "engine": "original",
                "mode": "advanced",
                "width": 768,
                "height": 768,
                "frames": 22,
                "duration_seconds": 1,
                "actual_steps": 20,
                "seed": 14,
            })
            second = SecondSamplingSpec(
                resolution="768p",
                width=768,
                height=768,
                steps=4,
                acceleration=50.0,
                denoise=0.22,
                memory_mode="auto",
                spatial_mode="strict",
                preserve_audio=False,
            )
            await engine.generate(
                target,
                None,
                None,
                (),
                (),
                (),
                asyncio.Event(),
                self.temporary / "repair-output.mp4",
                second_sampling=second,
                external_refinement_video_path=source_video,
            )
            request = factory.sessions[0].requests[-1]
            self.assertIsNone(request.refinement_latents_path)
            self.assertEqual(
                request.external_refinement_video_path, source_video.resolve()
            )
            self.assertEqual(request.refinement_spatial_mode, "strict")
            self.assertFalse(request.preserve_refinement_audio)
            self.assertEqual(request.steps, 4)
        finally:
            await engine.close()

    async def test_infinite_continuation_builds_cumulative_latent_and_window_audio_decode(self) -> None:
        import torch
        from h3serve.native_engine.hot_session import HotSessionResult
        from h3serve.native_engine.long_horizon import (
            audio_latent_frames,
            video_latent_frames,
        )

        class IncrementalSession:
            def __init__(self):
                self.requests = []
                self.decode_request = None
                self.audio_window_clocks = None
                self._last_conditioning_cache_payload = None
                self.runtime_config = SimpleNamespace(
                    resource_profile="int8_24gb",
                    max_device_bytes=int(23.25 * 1024**3),
                )

            def generate(self, request):
                self.requests.append(request)
                self._last_conditioning_cache_payload = {"schema_version": 1}
                torch.save({
                    "video": torch.full(
                        (1, 4, video_latent_frames(request.frames), 2, 2),
                        2.0,
                    ),
                    "audio": torch.full(
                        (1, 8, 2, audio_latent_frames(request.frames)), 2.0
                    ),
                    "frames": request.frames,
                    "fps": request.fps,
                    "width": request.width,
                    "height": request.height,
                    "engine": "original",
                    "seed": request.seed,
                }, request.save_final_latents_path)
                return HotSessionResult(
                    output_path=request.output_path,
                    total_seconds=0.2,
                    phases={"denoise": 0.2},
                    step_seconds=(0.02,),
                    forecast_profile={"actual_steps": 10, "forecast_steps": 10},
                    execution_profile={
                        "joint_acceleration": {
                            "accelerated": True,
                            "actual_dit_evaluations": 10,
                        },
                    },
                )

            def decode_latent_checkpoint(
                self, request, checkpoint_path, *, shot_video_checkpoints=(),
                audio_window_checkpoints=(), audio_window_clocks=(),
            ):
                self.decode_request = request
                self.audio_window_clocks = audio_window_clocks
                document = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
                assert document["frames"] == request.frames
                request.output_path.write_bytes(b"cumulative-video")
                return HotSessionResult(
                    output_path=request.output_path,
                    total_seconds=0.1,
                    phases={"decode": 0.1},
                    step_seconds=(),
                    forecast_profile={},
                    execution_profile={
                        "audio_window_decode": {"temporal_vae_domains": 2}
                    },
                )

            def _device_execution_budget_bytes(self):
                return 23 * 1024**3

            def close(self):
                pass

        class IncrementalFactory(FakeSparseHotFactory):
            def build(self, family):
                self.builds.append(family)
                session = IncrementalSession()
                self.sessions.append(session)
                return SimpleNamespace(
                    session=session, startup_seconds=0.1, qwen_storage="source",
                    weight_tier="int8", vram_profile="24gb",
                )

        factory = IncrementalFactory()
        engine = NativeHotH3Engine(factory, output_root=self.temporary)
        source_path = self.temporary / "source.pt"
        final_path = self.temporary / "cumulative.pt"
        memory_path = self.temporary / "cumulative.memory.pt"
        source_frames = 124
        context_frames = 39
        visible_frames = 85
        torch.save({
            "video": torch.ones(
                (1, 4, video_latent_frames(source_frames), 2, 2)
            ),
            "audio": torch.ones(
                (1, 8, 2, audio_latent_frames(source_frames))
            ),
            "frames": source_frames,
            "fps": 24,
            "width": 512,
            "height": 288,
            "engine": "original",
            "seed": 10,
            "qwen_conditioning_cache": {"schema_version": 1},
        }, source_path)
        spec = GenerationSpec.from_mapping({
            "prompt": "Strictly continue the same moving camera.",
            "engine": "original",
            "mode": "advanced",
            "width": 512,
            "height": 288,
            "frames": context_frames + visible_frames,
            "duration_seconds": (context_frames + visible_frames) / 24,
            "actual_steps": 20,
            "seed": 11,
        })
        continuation = InfiniteContinuationSpec(
            project_id="project-1",
            window_index=1,
            source_job_id="source-job",
            source_frames=source_frames,
            context_frames=context_frames,
            visible_frames=visible_frames,
            audio_bridge_ticks=65,
            memory=60,
        )
        spec = dataclasses.replace(spec, output_frames=continuation.output_frames)
        try:
            result = await engine.generate(
                spec, None, None, (), (), (), asyncio.Event(),
                self.temporary / "cumulative.mp4",
                final_latents_path=final_path,
                continuation=continuation,
                continuation_source_latents_path=source_path,
                continuation_output_memory_path=memory_path,
            )
            session = factory.sessions[0]
            request = session.requests[0]
            self.assertEqual(request.continuation_latents_path, source_path)
            self.assertEqual(request.continuation_context_frames, context_frames)
            self.assertEqual(request.continuation_video_prefix_frames, context_frames)
            self.assertIsNone(
                request.continuation_text_bridge_conditioning_path
            )
            self.assertTrue(request.latent_only)
            self.assertEqual(session.decode_request.frames, continuation.output_frames)
            self.assertEqual(
                session.audio_window_clocks,
                ((0, source_frames), (context_frames, visible_frames)),
            )
            document = torch.load(final_path, map_location="cpu", weights_only=True)
            self.assertEqual(document["frames"], continuation.output_frames)
            self.assertTrue(memory_path.is_file())
            self.assertEqual(
                result.inference_plan["infinite_continuation"]["physical_boundary"],
                "strict_continuation",
            )
            self.assertTrue(result.inference_plan["accelerated"])
            self.assertEqual(result.inference_plan["actual_dit_evaluations"], 10)
            self.assertNotIn("continuation_text_bridge", result.inference_plan)
            self.assertEqual(
                result.inference_plan["infinite_continuation"][
                    "semantic_trajectory"
                ],
                "single_current_prompt_v15",
            )
            self.assertEqual(
                result.inference_plan["infinite_window_forecast"][
                    "forecast_steps"
                ],
                10,
            )
        finally:
            await engine.close()

    async def test_2k15_second_sampling_uses_three_native_temporal_windows(self) -> None:
        import torch
        from h3serve.native_engine.hot_session import HotSessionResult

        class WindowSession:
            def __init__(self):
                self.requests = []
                self.decode_request = None
                self.runtime_config = SimpleNamespace(
                    resource_profile="int8_24gb",
                    max_device_bytes=int(23.25 * 1024**3),
                )

            def _device_execution_budget_bytes(self):
                return 23 * 1024**3

            def generate(self, request):
                self.requests.append(request)
                self._last_conditioning_cache_payload = cached_conditioning
                source = torch.load(
                    request.refinement_latents_path,
                    map_location="cpu",
                    weights_only=True,
                )
                torch.save(
                    {
                        **source,
                        "width": request.width,
                        "height": request.height,
                    },
                    request.save_final_latents_path,
                )
                return HotSessionResult(
                    output_path=request.output_path,
                    total_seconds=0.1,
                    phases={"denoise": 0.1},
                    step_seconds=(0.1,),
                    forecast_profile={"mode": "disabled"},
                    execution_profile={
                        "window": request.frames,
                        "qwen_conditioning_cache": {
                            "schema_version": 1,
                            "status": "hot_session_hit",
                            "fallback": None,
                            "persisted_with_latent": False,
                        },
                    },
                    peak_allocated_gib=8.0,
                    peak_reserved_gib=9.0,
                )

            def decode_latent_checkpoint(self, request, checkpoint_path, **kwargs):
                self.decode_request = request
                request.output_path.write_bytes(b"windowed-2k-video")
                return HotSessionResult(
                    output_path=request.output_path,
                    total_seconds=0.1,
                    phases={"decode": 0.1},
                    step_seconds=(),
                    forecast_profile={"mode": "decode_only"},
                    execution_profile={"decode": True},
                    peak_allocated_gib=7.0,
                    peak_reserved_gib=8.0,
                )

            def close(self):
                pass

        class WindowFactory(FakeV19HotFactory):
            def build(self, family):
                session = WindowSession()
                self.sessions.append(session)
                return SimpleNamespace(
                    session=session, startup_seconds=0.1, qwen_storage="source",
                    weight_tier="int8", vram_profile="24gb",
                )

        factory = WindowFactory()
        engine = NativeHotH3Engine(factory, output_root=self.temporary)
        source_latent = self.temporary / "source-480p15.pt"
        original_audio = torch.arange(603.0).view(1, 1, 1, 603)
        cached_embeds = torch.zeros((1, 2, 5120), dtype=torch.bfloat16)
        cached_tags = torch.ones((2,), dtype=torch.long)
        cached_conditioning = {
            "schema_version": 1,
            "fingerprint": "test-fingerprint",
            "prompt_embeds": cached_embeds,
            "text_token_tags": cached_tags,
        }
        torch.save(
            {
                "video": torch.zeros((1, 1, 107, 1, 1)),
                "audio": original_audio,
                "frames": 362,
                "fps": 24,
                "width": 864,
                "height": 480,
                "engine": "original",
                "seed": 12,
            },
            source_latent,
        )
        final_latent = self.temporary / "second-2k.pt"
        try:
            target = GenerationSpec.from_mapping(
                {
                    "prompt": "same H3 conditioning",
                    "engine": "original",
                    "mode": "advanced",
                    "width": 2560,
                    "height": 1440,
                    "frames": 362,
                    "duration_seconds": 362 / 24,
                    "actual_steps": 20,
                    "seed": 12,
                },
                allow_second_sampling_target=True,
            )
            second = SecondSamplingSpec(
                resolution="2k",
                width=2560,
                height=1440,
                steps=1,
                acceleration=75.0,
                denoise=0.2,
                memory_mode="auto",
            )
            result = await engine.generate(
                target,
                None,
                None,
                (),
                (),
                (),
                asyncio.Event(),
                self.temporary / "second-2k.mp4",
                final_latents_path=final_latent,
                second_sampling=second,
                refinement_latents_path=source_latent,
            )
            session = factory.sessions[0]
            self.assertEqual(
                [request.frames for request in session.requests],
                [136, 136, 124],
            )
            self.assertTrue(all(request.latent_only for request in session.requests))
            self.assertTrue(all(
                request.conditioning_cache_source_path == source_latent.resolve()
                for request in session.requests
            ))
            self.assertEqual(session.decode_request.execution_plan.vae_temporal_tile, 6)
            stitched = torch.load(final_latent, map_location="cpu", weights_only=True)
            self.assertEqual(stitched["video"].shape[2], 107)
            self.assertTrue(torch.equal(stitched["audio"], original_audio))
            self.assertTrue(torch.equal(
                stitched["qwen_conditioning_cache"]["prompt_embeds"],
                cached_embeds,
            ))
            self.assertTrue(torch.equal(
                stitched["qwen_conditioning_cache"]["text_token_tags"],
                cached_tags,
            ))
            self.assertTrue(result.output_path.is_file())
            self.assertFalse(result.inference_plan["ultimate_upscale"]["full_canvas"])
            self.assertEqual(
                result.inference_plan["qwen_conditioning_cache"]["status"],
                "hot_session_hit",
            )
            self.assertEqual(result.inference_plan["ultimate_upscale"]["temporal"][0]["frame_stop"], 136)
        finally:
            await engine.close()

    async def test_long_horizon_generation_co_denoises_one_global_av_state(self) -> None:
        import torch
        from h3serve.native_engine.hot_session import HotSessionResult
        from h3serve.native_engine.long_horizon import prepare_masked_av_prefix

        class LongSession:
            def __init__(self):
                self.requests = []
                self.decode_requests = []
                self.runtime_config = SimpleNamespace(
                    resource_profile="int8_24gb",
                    max_device_bytes=int(23.25 * 1024**3),
                )
                self._last_conditioning_cache_payload = None

            def persist_conditioning_cache(self, request, cache_path):
                torch.save({"fake_condition": request.prompt}, cache_path)
                return {
                    "path": str(cache_path),
                    "fingerprint": request.prompt,
                    "token_count": 32,
                    "status": "fake",
                }

            def generate(self, request):
                self.requests.append(request)
                output_frames = (
                    request.frames
                    if request.global_co_denoise_output_frames is None
                    else request.global_co_denoise_output_frames
                )
                video_t = 2 + 5 * ((output_frames - 5) // 17)
                audio_t = round(output_frames / 24 * 40)
                video = torch.full((1, 1, video_t, 1, 1), float(len(self.requests)))
                audio = torch.full((1, 1, 1, audio_t), float(len(self.requests)))
                if request.continuation_latents_path is not None:
                    source = torch.load(
                        request.continuation_latents_path,
                        map_location="cpu",
                        weights_only=True,
                    )
                    video, audio, _, _ = prepare_masked_av_prefix(
                        video,
                        audio,
                        source["video"],
                        source["audio"],
                        context_frames=request.continuation_context_frames,
                        video_prefix_frames=request.continuation_video_prefix_frames,
                        audio_bridge_ticks=request.continuation_audio_bridge_ticks,
                    )
                torch.save(
                    {
                        "video": video,
                        "audio": audio,
                        "frames": output_frames,
                        "fps": request.fps,
                        "width": request.width,
                        "height": request.height,
                        "engine": "original",
                        "seed": request.seed,
                    },
                    request.save_final_latents_path,
                )
                if request.global_co_denoise_output_frames is not None:
                    request.output_path.write_bytes(b"one-global-co-denoise")
                execution_profile = {"segment": len(self.requests)}
                if request.continuation_latents_path is not None:
                    context = int(request.continuation_context_frames)
                    protected = (
                        context
                        if request.continuation_video_prefix_frames is None
                        else int(request.continuation_video_prefix_frames)
                    )
                    context_tokens = video_latent_frames(context)
                    protected_tokens = (
                        0 if protected == 0 else video_latent_frames(protected)
                    )
                    execution_profile["long_horizon_continuation"] = {
                        "context_frames": context,
                        "video_prefix_frames": protected,
                        "video_hidden_repaint_frames": context - protected,
                        "video_context_tokens": context_tokens,
                        "video_protected_tokens": protected_tokens,
                        "video_hidden_repaint_tokens": context_tokens - protected_tokens,
                    }
                    if request.continuation_text_bridge_conditioning_path is not None:
                        hidden = context_tokens - protected_tokens
                        execution_profile["continuation_text_bridge"] = {
                            "policy": "protected_repaint_plateau_visible_fade_v5",
                            "active": True,
                            "boundary_auxiliary_steps": request.steps,
                            "current_prompt_steps": request.steps,
                            "blend_peak": 1.0,
                            "blend_band_policy": "hidden_repaint_plateau_then_zero_ended_visible_fade",
                            "blend_band_video_latent_tokens": hidden + 10,
                            "hidden_repaint_video_latent_tokens": hidden,
                            "visible_fade_video_latent_tokens": 10,
                            "extra_dit_calls": request.steps,
                        }
                return HotSessionResult(
                    output_path=request.output_path,
                    total_seconds=0.1,
                    phases={"denoise": 0.1},
                    step_seconds=(0.1,),
                    forecast_profile={"mode": "fake"},
                    execution_profile=execution_profile,
                    peak_allocated_gib=8.0,
                    peak_reserved_gib=9.0,
                )

            def decode_latent_checkpoint(self, request, checkpoint_path, **kwargs):
                self.decode_requests.append((request, checkpoint_path))
                request.output_path.write_bytes(b"one-final-long-decode")
                return HotSessionResult(
                    output_path=request.output_path,
                    total_seconds=0.1,
                    phases={"decode": 0.1},
                    step_seconds=(),
                    forecast_profile={"mode": "decode_only"},
                    execution_profile={"decode": True},
                    peak_allocated_gib=7.0,
                    peak_reserved_gib=8.0,
                )

            def close(self):
                pass

        class LongFactory(FakeV19HotFactory):
            def build(self, family):
                session = LongSession()
                self.sessions.append(session)
                return SimpleNamespace(
                    session=session,
                    startup_seconds=0.1,
                    qwen_storage="source",
                    weight_tier="int8",
                    vram_profile="24gb",
                )

        factory = LongFactory()
        engine = NativeHotH3Engine(factory, output_root=self.temporary)
        final_latent = self.temporary / "long-final.pt"
        prompt = (
            "integrated_multimodal_description: [Shot 1] A woman enters. "
            "[Shot 2] At 00:10.000, she lights a candle. "
            "[Shot 3] At 00:20.000, a man laughs.\n\n"
            "overall_soundscape: Quiet room tone.\n\n"
            "non_diegetic_music: N/A"
        )
        try:
            spec = GenerationSpec.from_mapping({
                "prompt": prompt,
                "duration_seconds": 30,
                "resolution": "480p",
                "aspect_ratio": "16:9",
                "sampling_steps": 10,
                "acceleration": 75,
                "seed": 82341,
            })
            result = await engine.generate(
                spec,
                None,
                None,
                (),
                (),
                (),
                asyncio.Event(),
                self.temporary / "long.mp4",
                final_latents_path=final_latent,
            )
            session = factory.sessions[0]
            self.assertEqual(len(session.requests), 1)
            co_request = session.requests[0]
            self.assertEqual(co_request.global_co_denoise_output_frames, 719)
            self.assertEqual(co_request.global_co_denoise_window_frames, 277)
            self.assertEqual(co_request.global_co_denoise_stride_frames, 204)
            self.assertEqual(len(co_request.global_co_denoise_prompts), 3)
            self.assertEqual(
                co_request.actual_step_indices,
                tuple(range(spec.sampling_steps)),
            )
            self.assertEqual(co_request.continuation_context_frames, 0)
            self.assertFalse(co_request.latent_only)
            self.assertEqual(len(session.decode_requests), 0)
            stitched = torch.load(final_latent, map_location="cpu", weights_only=True)
            self.assertEqual(stitched["frames"], 719)
            self.assertEqual(stitched["video"].shape[2], 212)
            self.assertEqual(stitched["audio"].shape[-1], 1198)
            self.assertEqual(result.output_path.read_bytes(), b"one-global-co-denoise")
            self.assertEqual(result.inference_plan["long_horizon"]["window_count"], 3)
            self.assertTrue(
                result.inference_plan["long_horizon"]["single_global_solver_state"]
            )
        finally:
            await engine.close()

    async def test_global_selflift_routes_one_connected_source_through_overlapping_views(self) -> None:
        import torch
        from h3serve.native_engine.hot_session import HotSessionResult

        class GlobalSelfLiftSession:
            def __init__(self):
                self.requests = []
                self.runtime_config = SimpleNamespace(
                    resource_profile="int8_24gb",
                    max_device_bytes=int(23.25 * 1024**3),
                )
                self._last_conditioning_cache_payload = None

            def persist_conditioning_cache(self, request, cache_path):
                torch.save({"fake_condition": request.prompt}, cache_path)
                return {"path": str(cache_path), "status": "fake"}

            def generate(self, request):
                self.requests.append(request)
                request.output_path.write_bytes(b"global-selflift")
                output_frames = request.global_co_denoise_output_frames
                video_t = 2 + 5 * ((output_frames - 5) // 17)
                audio_t = round(output_frames / 24 * 40)
                torch.save({
                    "video": torch.zeros((1, 24, video_t, 68, 120)),
                    "audio": torch.zeros((1, 32, 2, audio_t)),
                    "frames": output_frames,
                    "fps": 24,
                    "width": request.width,
                    "height": request.height,
                    "engine": "lora",
                    "seed": request.seed,
                }, request.save_final_latents_path)
                return HotSessionResult(
                    output_path=request.output_path,
                    total_seconds=0.1,
                    phases={"global_tail": 0.1},
                    step_seconds=(0.05, 0.05),
                    forecast_profile={"mode": "disabled"},
                    execution_profile={
                        "global_selflift": {
                            "mechanism": "global_sliding_selflift_v1",
                            "remaining_high_resolution_steps": 2,
                        }
                    },
                )

            def close(self):
                pass

        class GlobalSelfLiftFactory(FakeSparseHotFactory):
            def build(self, family):
                session = GlobalSelfLiftSession()
                self.sessions.append(session)
                return SimpleNamespace(
                    session=session,
                    startup_seconds=0.1,
                    qwen_storage="source",
                    weight_tier="int8",
                    vram_profile="24gb",
                    lora_recommended_steps=(4, 5, 6, 8),
                    lora_default_steps=6,
                )

        factory = GlobalSelfLiftFactory()
        engine = NativeHotH3Engine(factory, output_root=self.temporary)
        source = self.temporary / "global-source-x0.pt"
        source.write_bytes(b"runtime-owned source fixture")
        spec = GenerationSpec.from_mapping({
            "prompt": "window zero",
            "service_family": "first_last",
            "model_variant": "lora",
            "resolution": "1080p",
            "aspect_ratio": "16:9",
            "duration_seconds": 5,
            "sampling_steps": 8,
            "acceleration": 60,
            "second_pass_acceleration": 70,
            "acceleration_transition_step": 6,
            "selflift_enabled": True,
            "selflift_initial_resolution": "540p",
            "selflift_transition_step": 6,
            "selflift_sigma_scale": 0.65,
            "seed": 44,
        })
        spec = dataclasses.replace(
            spec,
            output_frames=719,
            requested_duration_seconds=719 / 24,
            actual_duration_seconds=719 / 24,
        )
        try:
            result = await engine.generate(
                spec,
                None,
                None,
                (),
                (),
                (),
                asyncio.Event(),
                self.temporary / "global-selflift.mp4",
                final_latents_path=self.temporary / "global-selflift.pt",
                global_selflift_source_path=source,
                global_selflift_prompts=("window zero", "window one", "window two"),
            )
            request = factory.sessions[0].requests[-1]
            self.assertEqual(request.global_co_denoise_output_frames, 719)
            self.assertEqual(request.global_selflift_source_path, source.resolve())
            self.assertEqual(request.global_selflift_sigma_scale, 0.65)
            self.assertIsNone(request.multiscale_resize_after_step)
            self.assertEqual(request.steps, 8)
            self.assertEqual(
                request.acceleration_plan_summary["schema_version"],
                "h3_stage_acceleration_v1",
            )
            self.assertEqual(
                request.acceleration_plan_summary["second_pass_acceleration"],
                70.0,
            )
            self.assertTrue(any(
                step >= 6 and action != "dense"
                for step, _layer, action in request.attention_action_schedule
            ))
            self.assertEqual(
                result.inference_plan["global_selflift"]["mechanism"],
                "global_sliding_selflift_v1",
            )
        finally:
            await engine.close()

    async def test_infinite_selflift_final_stitches_source_x0_before_one_global_call(self) -> None:
        import torch
        from h3serve.native_engine.engine import NativeGenerationResult

        class FinalEngine:
            def __init__(self, output_root):
                self.output_root = output_root
                self.calls = []

            async def generate(
                self,
                spec,
                first_frame,
                last_frame,
                reference_images,
                reference_videos,
                reference_audios,
                cancel_event,
                output_path,
                **kwargs,
            ):
                source = torch.load(
                    kwargs["global_selflift_source_path"],
                    map_location="cpu",
                    weights_only=True,
                )
                self.calls.append((spec, kwargs, source))
                output_path.write_bytes(b"one-global-selflift-film")
                final_path = kwargs["final_latents_path"]
                torch.save({
                    "video": source["video"],
                    "audio": source["audio"],
                    "frames": source["frames"],
                }, final_path)
                return NativeGenerationResult(
                    runtime_key="lora:int8:24gb:native-sm89",
                    elapsed_seconds=0.1,
                    output_path=output_path,
                    stage_seconds={"global_tail": 0.1},
                    inference_plan={"global_selflift": {"mechanism": "global_sliding_selflift_v1"}},
                    final_latents_path=final_path,
                )

        base = GenerationSpec.from_mapping({
            "prompt": "window zero",
            "service_family": "first_last",
            "model_variant": "lora",
            "resolution": "1080p",
            "aspect_ratio": "16:9",
            "duration_seconds": 5,
            "sampling_steps": 8,
            "acceleration": 60,
            "second_pass_acceleration": 70,
            "acceleration_transition_step": 6,
            "selflift_enabled": True,
            "selflift_initial_resolution": "540p",
            "selflift_transition_step": 6,
            "seed": 55,
        })
        frames = (311, 243, 243)
        prompts = ("window zero", "window one", "window two")
        source_jobs = []
        total_before = 0
        for index, (frame_count, prompt) in enumerate(zip(frames, prompts)):
            spec = dataclasses.replace(
                base,
                prompt=prompt,
                frames=frame_count,
                requested_duration_seconds=frame_count / 24,
                actual_duration_seconds=frame_count / 24,
                checkpoint_preview=True,
            )
            video_t = 2 + 5 * ((frame_count - 5) // 17)
            audio_t = round(frame_count / 24 * 40)
            checkpoint = self.temporary / f"fork-{index}.pt"
            torch.save({
                "selflift_source_video_x0": torch.full(
                    (1, 24, video_t, 34, 60), float(index + 1), dtype=torch.bfloat16
                ),
                "selflift_source_audio_x0": torch.full(
                    (1, 32, 2, audio_t), float(index + 1), dtype=torch.bfloat16
                ),
                "audio": torch.full(
                    (1, 32, 2, audio_t), float(index + 11), dtype=torch.bfloat16
                ),
                "selflift_source_width": 960,
                "selflift_source_height": 544,
                "selflift_split_step": 6,
                "steps": 8,
                "sigmas": [1.0, 0.9722222089767456, 0.9375, 0.8928571343421936,
                           0.8333333134651184, 0.75, 0.625, 0.4166666567325592, 0.0],
                "engine": "lora",
            }, checkpoint)
            continuation = None
            if index:
                continuation = InfiniteContinuationSpec(
                    project_id="p",
                    window_index=index,
                    source_job_id=str(index - 1),
                    source_frames=total_before,
                    context_frames=39,
                    visible_frames=204,
                    audio_bridge_ticks=0,
                    memory=0,
                )
            total_before = frame_count if index == 0 else total_before + 204
            source_jobs.append(SimpleNamespace(
                checkpoint_path=checkpoint,
                spec=spec,
                infinite_continuation=continuation,
                reference_images=(),
                reference_audios=(),
            ))

        # Online preview latents are cumulative.  Their audio, rather than the
        # earlier formal split estimate, is what the user has accepted.
        accepted_preview = self.temporary / "accepted-cumulative-preview.pt"
        torch.save({
            "video": torch.zeros((1, 24, 212, 34, 60), dtype=torch.bfloat16),
            "audio": torch.full(
                (1, 32, 2, 1198), 9.0, dtype=torch.bfloat16
            ),
            "audio_final": True,
            "frames": 719,
            "fps": 24,
            "width": 960,
            "height": 544,
            "engine": "lora",
        }, accepted_preview)
        source_jobs[-1].final_latents_path = accepted_preview

        fake_engine = FinalEngine(self.temporary)
        backend = NativeBackendManager(fake_engine)
        with patch(
            "h3serve.native_engine.audio_window_balance."
            "balance_encoded_creator_windows",
            return_value={"policy": "test", "boundary_count": 2},
        ):
            result = await backend.complete_infinite_selflift(
                tuple(source_jobs),
                "global-final",
                asyncio.Event(),
                final_spec=dataclasses.replace(
                    base,
                    acceleration=0,
                    second_pass_acceleration=76,
                    output_frames=719,
                    requested_duration_seconds=719 / 24,
                    actual_duration_seconds=719 / 24,
                ),
            )

        self.assertEqual(len(fake_engine.calls), 1)
        called_spec, called_kwargs, source = fake_engine.calls[0]
        self.assertEqual(source["frames"], 719)
        self.assertEqual(source["video"].shape[2], 212)
        self.assertEqual(source["audio"].shape[-1], 1198)
        self.assertTrue(torch.equal(
            source["audio"],
            torch.full((1, 32, 2, 1198), 9.0, dtype=torch.bfloat16),
        ))
        self.assertEqual(
            source["audio_representation"],
            "completed_low_resolution_tail_x0_v1",
        )
        self.assertTrue(source["audio_final"])
        self.assertNotIn("audio_state", source)
        self.assertEqual(called_spec.output_frames, 719)
        self.assertEqual(called_spec.acceleration, 0)
        self.assertEqual(called_spec.second_pass_acceleration, 76)
        self.assertEqual(
            called_kwargs["global_selflift_prompt_ranges"],
            ((0, 311), (311, 515), (515, 719)),
        )
        self.assertGreaterEqual(len(called_kwargs["global_selflift_prompts"]), 3)
        self.assertEqual(
            result.inference_plan["infinite_selflift"]["schema_version"],
            "global_sliding_selflift_v1",
        )

        # JSON one-click generation decodes no preview, but it retains the
        # same completed low-resolution audio latent.  The spatial suffix must
        # lock it exactly as online creation does.
        for source_job in source_jobs:
            source_job.spec = dataclasses.replace(
                source_job.spec,
                checkpoint_preview=False,
            )
        with patch(
            "h3serve.native_engine.audio_window_balance."
            "balance_encoded_creator_windows",
            return_value={"policy": "test", "boundary_count": 2},
        ):
            await backend.complete_infinite_selflift(
                tuple(source_jobs),
                "global-final-json",
                asyncio.Event(),
                final_spec=dataclasses.replace(
                    base,
                    output_frames=719,
                    requested_duration_seconds=719 / 24,
                    actual_duration_seconds=719 / 24,
                ),
            )
        self.assertEqual(len(fake_engine.calls), 2)
        _json_spec, _json_kwargs, json_source = fake_engine.calls[1]
        self.assertTrue(json_source["audio_final"])
        self.assertEqual(
            json_source["audio_representation"],
            "completed_low_resolution_tail_x0_v1",
        )
        self.assertNotIn("audio_state", json_source)
        self.assertTrue(torch.equal(
            json_source["audio"],
            torch.full((1, 32, 2, 1198), 9.0, dtype=torch.bfloat16),
        ))

    async def test_long_horizon_bounded_memory_and_global_audio_spine(self) -> None:
        import torch
        from h3serve.native_engine.av_token_memory import token_memory_telemetry
        from h3serve.native_engine.hot_session import HotSessionResult
        from h3serve.native_engine.long_horizon import (
            audio_latent_frames,
            prepare_masked_av_prefix,
            video_latent_frames,
        )

        class TokenMemorySession:
            def __init__(self):
                self.requests = []
                self.received_memories = []
                self.decode_requests = []
                self.runtime_config = SimpleNamespace(
                    resource_profile="int8_24gb",
                    max_device_bytes=int(23.25 * 1024**3),
                )
                self._last_conditioning_cache_payload = None

            def _conditioning_fingerprint(self, request):
                return "|".join((request.prompt, str(request.frames)))

            def persist_conditioning_cache(self, request, destination):
                Path(destination).write_bytes(b"fake-persisted-conditioning")
                return {
                    "status": "fake_persisted",
                    "prompt": request.prompt,
                    "frames": request.frames,
                }

            def generate(self, request):
                self.requests.append(request)
                if request.av_token_memory_path is None:
                    self.received_memories.append(None)
                else:
                    memory = torch.load(
                        request.av_token_memory_path,
                        map_location="cpu",
                        weights_only=True,
                    )
                    self.received_memories.append(token_memory_telemetry(memory))
                call = float(len(self.requests))
                generated_frames = int(
                    request.global_co_denoise_output_frames or request.frames
                )
                video = torch.full(
                    (1, 24, video_latent_frames(generated_frames), 2, 2),
                    call,
                )
                audio = torch.full(
                    (1, 32, 2, audio_latent_frames(generated_frames)),
                    call,
                )
                if request.continuation_latents_path is not None:
                    source = torch.load(
                        request.continuation_latents_path,
                        map_location="cpu",
                        weights_only=True,
                    )
                    video, audio, _, _ = prepare_masked_av_prefix(
                        video,
                        audio,
                        source["video"],
                        source["audio"],
                        context_frames=request.continuation_context_frames,
                        video_prefix_frames=request.continuation_video_prefix_frames,
                        audio_bridge_ticks=request.continuation_audio_bridge_ticks,
                    )
                torch.save(
                    {
                        "video": video,
                        "audio": audio,
                        "frames": generated_frames,
                        "fps": request.fps,
                        "width": request.width,
                        "height": request.height,
                        "engine": "lora",
                        "seed": request.seed,
                    },
                    request.save_final_latents_path,
                )
                execution_profile = {"segment": len(self.requests)}
                if request.continuation_latents_path is not None:
                    context = int(request.continuation_context_frames)
                    protected = (
                        context
                        if request.continuation_video_prefix_frames is None
                        else int(request.continuation_video_prefix_frames)
                    )
                    context_tokens = video_latent_frames(context)
                    protected_tokens = (
                        0 if protected == 0 else video_latent_frames(protected)
                    )
                    execution_profile["long_horizon_continuation"] = {
                        "context_frames": context,
                        "video_prefix_frames": protected,
                        "video_hidden_repaint_frames": context - protected,
                        "video_context_tokens": context_tokens,
                        "video_protected_tokens": protected_tokens,
                        "video_hidden_repaint_tokens": context_tokens - protected_tokens,
                    }
                    if request.continuation_text_bridge_conditioning_path is not None:
                        hidden = context_tokens - protected_tokens
                        execution_profile["continuation_text_bridge"] = {
                            "policy": "protected_repaint_plateau_visible_fade_v5",
                            "active": True,
                            "boundary_auxiliary_steps": request.steps,
                            "current_prompt_steps": request.steps,
                            "blend_peak": 1.0,
                            "blend_band_policy": "hidden_repaint_plateau_then_zero_ended_visible_fade",
                            "blend_band_video_latent_tokens": hidden + 10,
                            "hidden_repaint_video_latent_tokens": hidden,
                            "visible_fade_video_latent_tokens": 10,
                            "extra_dit_calls": request.steps,
                        }
                return HotSessionResult(
                    output_path=request.output_path,
                    total_seconds=0.1,
                    phases={"denoise": 0.1},
                    step_seconds=(0.01,) * request.steps,
                    forecast_profile={"mode": "fake"},
                    execution_profile=execution_profile,
                    peak_allocated_gib=8.0,
                    peak_reserved_gib=9.0,
                )

            def decode_latent_checkpoint(self, request, checkpoint_path, **kwargs):
                self.decode_requests.append((request, checkpoint_path))
                request.output_path.write_bytes(b"bounded-token-memory-video")
                return HotSessionResult(
                    output_path=request.output_path,
                    total_seconds=0.1,
                    phases={"decode": 0.1},
                    step_seconds=(),
                    forecast_profile={"mode": "decode_only"},
                    execution_profile={"decode": True},
                    peak_allocated_gib=7.0,
                    peak_reserved_gib=8.0,
                )

            def close(self):
                pass

        class TokenMemoryFactory(FakeV19HotFactory):
            def build(self, family):
                session = TokenMemorySession()
                self.sessions.append(session)
                return SimpleNamespace(
                    session=session,
                    startup_seconds=0.1,
                    qwen_storage="source",
                    weight_tier="int8",
                    vram_profile="24gb",
                )

        factory = TokenMemoryFactory()
        engine = NativeHotH3Engine(factory, output_root=self.temporary)
        final_latent = self.temporary / "bounded-memory-final.pt"
        try:
            spec = GenerationSpec.from_mapping({
                "prompt": "A subject leaves and later returns to the original scene.",
                "engine": "lora",
                "duration_seconds": 30,
                "resolution": "480p",
                "aspect_ratio": "16:9",
                "sampling_steps": 7,
                "acceleration": 0,
                "seed": 82901,
            })
            with patch.dict(
                "os.environ",
                {
                    "H3_LONG_AV_TOKEN_MEMORY": "1",
                    "H3_LONG_GLOBAL_AUDIO_SPINE": "1",
                },
            ):
                result = await engine.generate(
                    spec,
                    None,
                    None,
                    (),
                    (),
                    (),
                    asyncio.Event(),
                    self.temporary / "bounded-memory.mp4",
                    final_latents_path=final_latent,
                )
            session = factory.sessions[0]
            self.assertGreaterEqual(len(session.requests), 4)
            spine_request = session.requests[-1]
            continuation_requests = session.requests[:-1]
            self.assertEqual(
                spine_request.global_co_denoise_output_frames,
                spec.output_frames,
            )
            self.assertLess(spine_request.frames, spec.output_frames)
            self.assertEqual((spine_request.width, spine_request.height), (320, 192))
            self.assertTrue(spine_request.latent_only)
            self.assertIsNone(spine_request.continuation_latents_path)
            self.assertIsNone(spine_request.av_token_memory_path)
            self.assertIsNone(session.received_memories[0])
            self.assertIsNone(session.received_memories[-1])
            for index, receipt in enumerate(
                session.received_memories[1:-1], start=1
            ):
                self.assertIsNotNone(receipt)
                self.assertEqual(receipt["updates"], index)
                self.assertGreater(receipt["video_entries"], 0)
                # Free prose has no structural dialogue authority, so inferred
                # contiguous voice excerpts are never collected or replayed.
                self.assertEqual(receipt["audio_entries"], 0)
                self.assertLessEqual(receipt["video_entries"], receipt["video_slots"])
                self.assertLessEqual(receipt["audio_entries"], receipt["audio_slots"])
                self.assertFalse(receipt["text_summary"])
            self.assertTrue(all(
                request.continuation_context_frames == 39
                for request in continuation_requests[1:]
            ))
            self.assertEqual(
                [
                    request.continuation_audio_bridge_ticks
                    for request in continuation_requests
                ],
                [0] + [65] * (len(continuation_requests) - 1),
            )
            self.assertEqual(len(session.decode_requests), 1)
            route = result.inference_plan["long_horizon"]
            self.assertTrue(route["bounded_av_token_memory"])
            self.assertEqual(
                route["audio_seam_method"],
                "windowed_global_audio_spine_v2",
            )
            self.assertEqual(route["audio_context_frames"], 39)
            self.assertEqual(route["audio_bridge_ticks"], 65)
            self.assertEqual(
                len(route["token_memory_records"]), len(continuation_requests)
            )
            self.assertTrue(
                route["global_audio_spine"]["single_global_audio_trajectory"]
            )
            self.assertEqual(route["global_audio_spine"]["internal_audio_seams"], 0)
            self.assertTrue(route["window_audio_discarded"])
            self.assertTrue(all(
                receipt["bounded_active_context"]
                for receipt in route["token_memory_records"]
            ))
            stitched = torch.load(final_latent, map_location="cpu", weights_only=True)
            self.assertEqual(stitched["frames"], spec.output_frames)
            self.assertEqual(
                stitched["audio"].shape[-1], audio_latent_frames(spec.output_frames)
            )
            self.assertEqual(
                stitched["video"].shape[2], video_latent_frames(spec.output_frames)
            )
            self.assertEqual(
                float(stitched["audio"].mean()), float(len(session.requests))
            )

            # The new authored contract owns its memory policy per request;
            # legacy environment switches must not be needed to activate it.
            authored_payload = json.loads((
                Path(__file__).resolve().parents[1]
                / "static" / "long-video-examples" / "cafe30.json"
            ).read_text(encoding="utf-8"))
            # Exercise the public 3.75-second overlap used by the long-video
            # experiments.  Ninety video frames map to 150 Audio-VAE ticks;
            # keeping the old fixed 65-tick bridge caused an audible generated
            # transient at every delivered window boundary.
            authored_payload["long_video"]["overlap_seconds"] = 3.75
            authored_spec = GenerationSpec.from_mapping(authored_payload)
            offset = len(session.requests)
            with patch.dict("os.environ", {
                "H3_LONG_AV_TOKEN_MEMORY": "0",
                "H3_LONG_AUDIO_TOKEN_MEMORY": "0",
                "H3_LONG_VISUAL_TOKEN_MEMORY": "0",
                "H3_LONG_GLOBAL_AUDIO_SPINE": "1",
                "H3_LONG_STRUCTURED_DIRECTOR": "0",
            }):
                authored_result = await engine.generate(
                    authored_spec,
                    None,
                    None,
                    (),
                    (),
                    (),
                    asyncio.Event(),
                    self.temporary / "authored-window-interface.mp4",
                )
            authored_requests = session.requests[offset:]
            authored_memories = session.received_memories[offset:]
            self.assertEqual(len(authored_requests), 3)
            self.assertEqual(
                [item.continuation_context_frames for item in authored_requests],
                [0, 90, 90],
            )
            self.assertEqual(
                [item.continuation_video_prefix_frames for item in authored_requests],
                [None, 0, 0],
            )
            self.assertEqual(
                [
                    item.continuation_video_prefix_from_source_end
                    for item in authored_requests
                ],
                [False, False, False],
            )
            self.assertIsNone(authored_memories[0])
            self.assertTrue(all(item is not None for item in authored_memories[1:]))
            self.assertIn(
                "complete local target is one uninterrupted shot",
                authored_requests[1].prompt,
            )
            self.assertNotIn(
                "looks toward the rainy window", authored_requests[1].prompt,
            )
            authored_route = authored_result.inference_plan["long_horizon"]
            self.assertFalse(authored_route["global_audio_spine"])
            self.assertTrue(authored_route["bounded_av_token_memory"])
            self.assertEqual(
                authored_route["window_interface"]["memory_budget"]["video_frames"],
                6,
            )
            self.assertEqual(
                authored_route["window_interface"]["audio_bridge_ticks_by_window"],
                [0, 150, 150],
            )
            # The first eligible authored dialogue becomes the canonical voice
            # excerpt. Later dialogue windows may consume it but cannot replace
            # it with a voice that has already drifted.
            authored_audio_positions = [
                item["audio_positions"]
                for item in authored_route["token_memory_records"]
            ]
            self.assertTrue(authored_audio_positions[0])
            self.assertTrue(all(
                positions == authored_audio_positions[0]
                for positions in authored_audio_positions[1:]
            ))
            self.assertEqual(
                authored_route["authored_voice_anchor"]["refresh_policy"],
                "write_once",
            )
        finally:
            await engine.close()

    async def test_bounded_memory_routes_first_last_and_image_audio_inputs_without_overwrite(self) -> None:
        import torch
        from h3serve.native_engine.hot_session import HotSessionResult
        from h3serve.native_engine.long_horizon import (
            audio_latent_frames,
            prepare_masked_av_prefix,
            video_latent_frames,
        )

        class MultimodalMemorySession:
            def __init__(self):
                self.requests = []
                self.received_memories = []
                self.decode_requests = []
                self.runtime_config = SimpleNamespace(
                    resource_profile="int8_24gb",
                    max_device_bytes=int(23.25 * 1024**3),
                )
                self._last_conditioning_cache_payload = None

            def generate(self, request):
                self.requests.append(request)
                if request.av_token_memory_path is None:
                    self.received_memories.append(None)
                else:
                    from h3serve.native_engine.av_token_memory import (
                        token_memory_telemetry,
                    )

                    self.received_memories.append(token_memory_telemetry(
                        torch.load(
                            request.av_token_memory_path,
                            map_location="cpu",
                            weights_only=True,
                        )
                    ))
                call = float(len(self.requests))
                video = torch.full(
                    (1, 24, video_latent_frames(request.frames), 2, 2),
                    call,
                )
                audio = torch.full(
                    (1, 32, 2, audio_latent_frames(request.frames)),
                    call,
                )
                if request.continuation_latents_path is not None:
                    source = torch.load(
                        request.continuation_latents_path,
                        map_location="cpu",
                        weights_only=True,
                    )
                    video, audio, _, _ = prepare_masked_av_prefix(
                        video,
                        audio,
                        source["video"],
                        source["audio"],
                        context_frames=request.continuation_context_frames,
                        video_prefix_frames=request.continuation_video_prefix_frames,
                        audio_bridge_ticks=request.continuation_audio_bridge_ticks,
                    )
                torch.save(
                    {
                        "video": video,
                        "audio": audio,
                        "frames": request.frames,
                        "fps": request.fps,
                        "width": request.width,
                        "height": request.height,
                        "engine": "lora",
                        "seed": request.seed,
                    },
                    request.save_final_latents_path,
                )
                execution_profile = {"segment": len(self.requests)}
                if request.continuation_latents_path is not None:
                    context = int(request.continuation_context_frames)
                    protected = (
                        context
                        if request.continuation_video_prefix_frames is None
                        else int(request.continuation_video_prefix_frames)
                    )
                    context_tokens = video_latent_frames(context)
                    protected_tokens = (
                        0 if protected == 0 else video_latent_frames(protected)
                    )
                    execution_profile["long_horizon_continuation"] = {
                        "context_frames": context,
                        "video_prefix_frames": protected,
                        "video_hidden_repaint_frames": context - protected,
                        "video_context_tokens": context_tokens,
                        "video_protected_tokens": protected_tokens,
                        "video_hidden_repaint_tokens": context_tokens - protected_tokens,
                    }
                    if request.continuation_text_bridge_conditioning_path is not None:
                        hidden = context_tokens - protected_tokens
                        execution_profile["continuation_text_bridge"] = {
                            "policy": "protected_repaint_plateau_visible_fade_v5",
                            "active": True,
                            "boundary_auxiliary_steps": request.steps,
                            "current_prompt_steps": request.steps,
                            "blend_peak": 1.0,
                            "blend_band_policy": "hidden_repaint_plateau_then_zero_ended_visible_fade",
                            "blend_band_video_latent_tokens": hidden + 10,
                            "hidden_repaint_video_latent_tokens": hidden,
                            "visible_fade_video_latent_tokens": 10,
                            "extra_dit_calls": request.steps,
                        }
                return HotSessionResult(
                    output_path=request.output_path,
                    total_seconds=0.1,
                    phases={"denoise": 0.1},
                    step_seconds=(0.01,) * request.steps,
                    forecast_profile={"mode": "fake"},
                    execution_profile=execution_profile,
                    peak_allocated_gib=8.0,
                    peak_reserved_gib=9.0,
                )

            def decode_latent_checkpoint(self, request, checkpoint_path, **kwargs):
                self.decode_requests.append((request, checkpoint_path))
                request.output_path.write_bytes(b"multimodal-long-video")
                return HotSessionResult(
                    output_path=request.output_path,
                    total_seconds=0.1,
                    phases={"decode": 0.1},
                    step_seconds=(),
                    forecast_profile={"mode": "decode_only"},
                    execution_profile={"decode": True},
                    peak_allocated_gib=7.0,
                    peak_reserved_gib=8.0,
                )

            def close(self):
                pass

        class MultimodalMemoryFactory(FakeV19HotFactory):
            def build(self, family):
                self.builds.append(family)
                session = MultimodalMemorySession()
                self.sessions.append(session)
                return SimpleNamespace(
                    session=session,
                    startup_seconds=0.1,
                    qwen_storage="source",
                    weight_tier="int8",
                    vram_profile="24gb",
                )

        def spec(service_family):
            return GenerationSpec.from_mapping({
                "prompt": "Multimodal bounded-memory routing contract.",
                "service_family": service_family,
                "model_variant": "lora",
                "duration_seconds": 30,
                "resolution": "480p",
                "aspect_ratio": "16:9",
                "sampling_steps": 7,
                "acceleration": 0,
                "seed": 82901,
            })

        first = self.temporary / "first.png"
        last = self.temporary / "last.png"
        image = self.temporary / "reference.png"
        audio = self.temporary / "reference.wav"
        video_reference = self.temporary / "reference.mp4"
        for path in (first, last, image, audio, video_reference):
            path.write_bytes(b"routing fixture")

        fl_factory = MultimodalMemoryFactory()
        fl_engine = NativeHotH3Engine(fl_factory, output_root=self.temporary)
        try:
            with patch.dict(
                "os.environ",
                {
                    "H3_LONG_AV_TOKEN_MEMORY": "1",
                    "H3_LONG_GLOBAL_AUDIO_SPINE": "0",
                },
            ):
                await fl_engine.generate(
                    spec("first_last"),
                    first,
                    last,
                    (),
                    (),
                    (),
                    asyncio.Event(),
                    self.temporary / "first-last-long.mp4",
                )
            requests = fl_factory.sessions[0].requests
            self.assertGreaterEqual(len(requests), 3)
            self.assertEqual(fl_factory.builds, ["fl2va_int8_24gb"])
            self.assertEqual(requests[0].first_frame, first)
            self.assertTrue(all(item.first_frame is None for item in requests[1:]))
            self.assertEqual(requests[-1].last_frame, last)
            self.assertTrue(all(item.last_frame is None for item in requests[:-1]))
            self.assertIsNone(requests[0].av_token_memory_path)
            self.assertTrue(all(
                item.av_token_memory_path is not None for item in requests[1:]
            ))
            self.assertTrue(all(
                not item.reference_images
                and not item.reference_videos
                and not item.reference_audios
                for item in requests
            ))
        finally:
            await fl_engine.close()

        ref_factory = MultimodalMemoryFactory()
        ref_engine = NativeHotH3Engine(ref_factory, output_root=self.temporary)
        try:
            with patch.dict(
                "os.environ",
                {
                    "H3_LONG_AV_TOKEN_MEMORY": "1",
                    "H3_LONG_GLOBAL_AUDIO_SPINE": "0",
                },
            ):
                await ref_engine.generate(
                    spec("reference"),
                    None,
                    None,
                    (image,),
                    (),
                    (audio,),
                    asyncio.Event(),
                    self.temporary / "reference-long.mp4",
                )
            requests = ref_factory.sessions[0].requests
            self.assertGreaterEqual(len(requests), 3)
            self.assertEqual(ref_factory.builds, ["ref2va_int8_24gb"])
            self.assertTrue(all(item.reference_images == (image,) for item in requests))
            self.assertTrue(all(item.reference_audios == (audio,) for item in requests))
            self.assertTrue(all(not item.reference_videos for item in requests))
            self.assertTrue(all(
                item.first_frame is None and item.last_frame is None
                for item in requests
            ))
            self.assertIsNone(requests[0].av_token_memory_path)
            self.assertTrue(all(
                item.av_token_memory_path is not None for item in requests[1:]
            ))

            session = ref_factory.sessions[0]
            offset = len(session.requests)
            with patch.dict("os.environ", {
                "H3_LONG_AV_TOKEN_MEMORY": "1",
                "H3_LONG_GLOBAL_AUDIO_SPINE": "0",
                "H3_LONG_VISUAL_TOKEN_MEMORY": "0",
            }):
                await ref_engine.generate(
                    spec("reference"), None, None, (image,), (), (audio,),
                    asyncio.Event(), self.temporary / "audio-only-memory.mp4",
                )
            ablated = session.requests[offset:]
            for request, memory in zip(
                ablated[1:], session.received_memories[offset + 1:]
            ):
                self.assertEqual(memory["video_entries"], 0)
                self.assertIsNotNone(request.continuation_latents_path)
                self.assertGreater(request.continuation_context_frames, 0)
                self.assertEqual(request.continuation_audio_bridge_ticks, 65)
                self.assertEqual(request.reference_images, (image,))
                self.assertEqual(request.reference_audios, (audio,))

            with patch.dict(
                "os.environ",
                {"H3_LONG_AV_TOKEN_MEMORY": "1"},
            ), self.assertRaisesRegex(ValueError, "reference videos"):
                await ref_engine.generate(
                    spec("reference"),
                    None,
                    None,
                    (),
                    (video_reference,),
                    (),
                    asyncio.Event(),
                    self.temporary / "excluded-reference-video.mp4",
                )
        finally:
            await ref_engine.close()

        director_factory = MultimodalMemoryFactory()
        director_engine = NativeHotH3Engine(
            director_factory, output_root=self.temporary
        )
        director_prompt = """integrated_multimodal_description:
[Shot 1] A speaker talks once: <d>[English] opening.</d>
[Shot 2] At 00:08.000, the camera cuts to silent work.
[Shot 3] At 00:16.000, the silent work continues from above.
[Shot 4] At 00:24.000, the camera returns. At 00:27.000, the speaker says: <d>[Japanese] 完了。</d>

overall_soundscape: Preserve room tone; no unlisted utterance.
non_diegetic_music: N/A
"""
        try:
            director_spec = GenerationSpec.from_mapping({
                "prompt": director_prompt,
                "service_family": "reference",
                "model_variant": "lora",
                "duration_seconds": 30,
                "resolution": "480p",
                "aspect_ratio": "16:9",
                "sampling_steps": 7,
                "acceleration": 0,
                "seed": 91,
            })
            with patch.dict(
                "os.environ",
                {
                    "H3_LONG_AV_TOKEN_MEMORY": "1",
                    "H3_LONG_GLOBAL_AUDIO_SPINE": "0",
                    "H3_LONG_STRUCTURED_DIRECTOR": "1",
                },
            ):
                result = await director_engine.generate(
                    director_spec,
                    None,
                    None,
                    (image,),
                    (),
                    (audio,),
                    asyncio.Event(),
                    self.temporary / "reference-director-long.mp4",
                )
            requests = director_factory.sessions[0].requests
            memories = director_factory.sessions[0].received_memories
            self.assertEqual(len(requests), 4)
            self.assertEqual(
                [bool(item.reference_audios) for item in requests],
                [True, False, False, True],
            )
            self.assertIsNone(memories[0])
            for receipt in memories[1:3]:
                self.assertIsNotNone(receipt)
                self.assertEqual(receipt["audio_entries"], 0)
                self.assertFalse(receipt["audio_route"]["active"])
                self.assertGreater(receipt["video_entries"], 0)
            self.assertIsNotNone(memories[3])
            # The user's reference audio remains authoritative; adding a
            # self-generated voice excerpt would duplicate that condition.
            self.assertEqual(memories[3]["audio_entries"], 0)
            segment_receipts = result.inference_plan["long_horizon"]["segments"]
            self.assertEqual(
                [item["authorized_dialogue_count"] for item in segment_receipts],
                [1, 0, 0, 1],
            )
            self.assertEqual(
                [item["audio_memory_active"] for item in segment_receipts],
                [True, False, False, True],
            )
        finally:
            await director_engine.close()

    async def test_lora_joint_checkpoint_preview_and_resume_keep_one_schedule(self) -> None:
        factory = FakeCheckpointHotFactory()
        engine = NativeHotH3Engine(factory, output_root=self.temporary)
        checkpoint_path = self.temporary / "checkpoints" / "lora.pt"
        output_path = self.temporary / "lora-checkpoint.mp4"
        spec = GenerationSpec.from_mapping({
            "prompt": "LoRA resumable scheduled trajectory",
            "service_family": "reference",
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
            "checkpoint_preview": True,
            "checkpoint_preview_steps": 4,
            "seed": 82416,
        })
        try:
            stopped = await engine.generate(
                spec, None, None, (), (), (), asyncio.Event(), output_path,
                checkpoint_path=checkpoint_path,
            )
            self.assertEqual(stopped.completed_steps, 3)
            self.assertEqual(stopped.total_steps, 8)
            self.assertEqual(stopped.checkpoint_path, checkpoint_path)
            first = factory.sessions[0].requests[-1]
            self.assertEqual(first.actual_step_indices, tuple(range(8)))
            self.assertEqual(len(first.attention_action_schedule), 8 * 50)
            self.assertEqual(first.checkpoint_after_step, 3)
            self.assertTrue(first.preview_branch_use_lora)
            self.assertTrue(first.preview_branch_force_dense)
            self.assertEqual(first.preview_branch_steps, 4)

            resumed = await engine.generate(
                spec, None, None, (), (), (), asyncio.Event(), output_path,
                resume_checkpoint_path=checkpoint_path,
            )
            second = factory.sessions[0].requests[-1]
            self.assertEqual(second.formal_resume_state_path, checkpoint_path)
            self.assertIsNone(second.checkpoint_after_step)
            self.assertEqual(
                second.attention_action_schedule,
                first.attention_action_schedule,
            )
            self.assertEqual(second.actual_step_indices, first.actual_step_indices)
            self.assertEqual(
                second.acceleration_plan_summary["scheduler_family"],
                "h3_lora_v1_no_forecast_round229",
            )
            self.assertEqual(resumed.output_path.read_bytes(), b"resumed-lora-video")
        finally:
            await engine.close()

    async def test_hot_engine_defers_v19_routing_until_exact_tokenisation(self) -> None:
        factory = FakeV19HotFactory()
        engine = NativeHotH3Engine(factory, output_root=self.temporary)
        try:
            spec = GenerationSpec.from_mapping({
                "prompt": "V19 exact token routing",
                "engine": "original",
                "mode": "advanced",
                "width": 1280,
                "height": 736,
                "duration_seconds": 15,
                "sampling_steps": 20,
                "acceleration": 100,
                "seed": 82303,
            })
            await engine.generate(
                spec, None, None, (), (), (), asyncio.Event(),
                self.temporary / "v19.mp4",
            )
            request = factory.sessions[0].requests[-1]
            self.assertEqual(request.actual_step_indices, tuple(range(20)))
            self.assertEqual(request.attention_action_schedule, ())
            self.assertIsNone(request.acceleration_plan_summary)
            self.assertEqual(request.v19_acceleration, 100.0)
            self.assertTrue(request.execution_plan.fused_rms_adaln)
            self.assertTrue(request.execution_plan.vae_transformer_block_compile)
        finally:
            await engine.close()

    async def test_v19_auto_preview_requests_the_standard_actual_anchor(self) -> None:
        factory = FakeV19HotFactory()
        engine = NativeHotH3Engine(factory, output_root=self.temporary)
        try:
            spec = GenerationSpec.from_mapping({
                "prompt": "V19 preview anchor",
                "engine": "original",
                "mode": "advanced",
                "width": 1280,
                "height": 736,
                "duration_seconds": 15,
                "sampling_steps": 20,
                "acceleration": 100,
                "preview_mode": "auto",
                "preview_step_index": 5,
                "preview_branch_steps": 2,
                "preview_fast_finish": True,
                "checkpoint_preview_steps": 4,
                "checkpoint_preview_resolution": "360p",
                "seed": 82303,
            })
            await engine.generate(
                spec, None, None, (), (), (), asyncio.Event(),
                self.temporary / "v19-preview.mp4",
            )
            request = factory.sessions[0].requests[-1]
            self.assertEqual(request.preview_step_index, 5)
            self.assertIsNotNone(request.preview_output_path)
            self.assertEqual(request.preview_decode_mode, "fast_finish")
            self.assertFalse(request.preview_branch_use_lora)
            self.assertEqual(request.preview_branch_steps, 2)
            self.assertAlmostEqual(request.preview_branch_spatial_scale, 360 / 736)
        finally:
            await engine.close()

    async def test_selflift_fast_preview_uses_preview_branch_steps(self) -> None:
        factory = FakeSparseHotFactory()
        engine = NativeHotH3Engine(factory, output_root=self.temporary)
        try:
            spec = GenerationSpec.from_mapping({
                "prompt": "SelfLift preview keeps the trained sigma tail",
                "service_family": "first_last",
                "model_variant": "lora",
                "mode": "advanced",
                "width": 1920,
                "height": 1088,
                "duration_seconds": 5,
                "sampling_steps": 8,
                "acceleration": 0,
                "selflift_enabled": True,
                "selflift_initial_resolution": "540p",
                "selflift_transition_step": 6,
                "preview_mode": "auto",
                "preview_step_index": 5,
                "preview_branch_steps": 2,
                "preview_fast_finish": True,
                "checkpoint_preview_steps": 4,
                "seed": 9090954,
            })
            await engine.generate(
                spec, None, None, (), (), (), asyncio.Event(),
                self.temporary / "selflift-preview.mp4",
            )
            request = factory.sessions[0].requests[-1]
            self.assertEqual(request.preview_step_index, 5)
            self.assertEqual(request.preview_branch_steps, 2)
            self.assertTrue(request.preview_branch_use_lora)
            self.assertEqual(request.multiscale_resize_after_step, 5)
            self.assertEqual(
                request.multiscale_transition_mode,
                "selflift_learned_x0",
            )
        finally:
            await engine.close()

    async def test_selflift_checkpoint_returns_preview_latent_and_lifted_fork(self) -> None:
        factory = FakeCheckpointHotFactory()
        engine = NativeHotH3Engine(factory, output_root=self.temporary)
        checkpoint_path = self.temporary / "checkpoints" / "selflift-fork.pt"
        preview_latents = self.temporary / "latents" / "selflift-preview.pt"
        try:
            spec = GenerationSpec.from_mapping({
                "prompt": "SelfLift long-video fork",
                "service_family": "first_last",
                "model_variant": "lora",
                "mode": "advanced",
                "width": 1280,
                "height": 736,
                "duration_seconds": 1,
                "sampling_steps": 8,
                "acceleration": 40,
                "second_pass_acceleration": 70,
                "selflift_enabled": True,
                "selflift_initial_resolution": "480p",
                "selflift_transition_step": 6,
                "acceleration_transition_step": 6,
                "execution_mode": "checkpoint",
                "checkpoint_step": 6,
                "checkpoint_retain": True,
                "checkpoint_preview": True,
                "checkpoint_preview_steps": 2,
                "checkpoint_preview_resolution": "source",
                "seed": 9090956,
            })
            stopped = await engine.generate(
                spec, None, None, (), (), (), asyncio.Event(),
                self.temporary / "selflift-fork-preview.mp4",
                checkpoint_path=checkpoint_path,
                final_latents_path=preview_latents,
            )
            request = factory.sessions[0].requests[-1]
            self.assertEqual(request.preview_step_index, 5)
            self.assertEqual(request.checkpoint_after_step, 6)
            self.assertEqual(request.preview_branch_steps, 2)
            self.assertEqual(request.preview_latents_path, preview_latents)
            self.assertEqual(stopped.preview_latents_path, preview_latents)
            self.assertEqual(stopped.checkpoint_path, checkpoint_path)
        finally:
            await engine.close()

    async def test_selflift_json_checkpoint_finishes_audio_without_preview_decode(self) -> None:
        factory = FakeCheckpointHotFactory()
        engine = NativeHotH3Engine(factory, output_root=self.temporary)
        checkpoint_path = self.temporary / "checkpoints" / "selflift-json.pt"
        completed_latents = self.temporary / "latents" / "selflift-json-source.pt"
        try:
            spec = GenerationSpec.from_mapping({
                "prompt": "SelfLift JSON direct source",
                "service_family": "first_last",
                "model_variant": "lora",
                "mode": "advanced",
                "width": 1280,
                "height": 736,
                "duration_seconds": 1,
                "sampling_steps": 8,
                "acceleration": 40,
                "second_pass_acceleration": 70,
                "selflift_enabled": True,
                "selflift_initial_resolution": "480p",
                "selflift_transition_step": 6,
                "acceleration_transition_step": 6,
                "execution_mode": "checkpoint",
                "checkpoint_step": 6,
                "checkpoint_retain": True,
                "checkpoint_preview": False,
                "seed": 9090957,
            })
            stopped = await engine.generate(
                spec, None, None, (), (), (), asyncio.Event(),
                self.temporary / "selflift-json-final.mp4",
                checkpoint_path=checkpoint_path,
                final_latents_path=completed_latents,
            )
            request = factory.sessions[0].requests[-1]
            self.assertEqual(request.preview_step_index, 5)
            self.assertIsNone(request.preview_output_path)
            self.assertEqual(request.preview_latents_path, completed_latents)
            self.assertEqual(request.preview_decode_mode, "fast_finish")
            self.assertEqual(request.preview_branch_steps, 2)
            self.assertEqual(request.preview_branch_spatial_scale, 1.0)
            self.assertTrue(request.preview_branch_use_lora)
            self.assertFalse(request.preview_branch_warm_history)
            self.assertEqual(stopped.preview_latents_path, completed_latents)
        finally:
            await engine.close()

    async def test_selflift_base_keeps_res_boundary_and_tail_actual(self) -> None:
        factory = FakeV19HotFactory()
        engine = NativeHotH3Engine(factory, output_root=self.temporary)
        try:
            spec = GenerationSpec.from_mapping({
                "prompt": "Base SelfLift keeps an exact RES transition",
                "service_family": "first_last",
                "model_variant": "base",
                "mode": "advanced",
                "width": 1280,
                "height": 736,
                "duration_seconds": 5,
                "sampling_steps": 20,
                "acceleration": 35,
                "second_pass_acceleration": 70,
                "acceleration_transition_step": 18,
                "selflift_enabled": True,
                "selflift_initial_resolution": "480p",
                "selflift_transition_step": 18,
                "seed": 9090955,
            })
            await engine.generate(
                spec, None, None, (), (), (), asyncio.Event(),
                self.temporary / "selflift-base.mp4",
            )
            request = factory.sessions[0].requests[-1]
            self.assertFalse(request.use_lora)
            self.assertEqual(request.v19_acceleration, 35.0)
            self.assertEqual(request.v19_second_pass_acceleration, 70.0)
            self.assertEqual(request.acceleration_transition_step, 18)
            self.assertEqual(request.multiscale_resize_after_step, 17)
            self.assertEqual(
                request.multiscale_transition_mode,
                "selflift_learned_x0",
            )
            self.assertEqual(
                request.scheduler_required_actual_step_indices,
                (17, 18, 19),
            )
        finally:
            await engine.close()

    async def test_v19_checkpoint_and_resume_keep_scheduler_anchor(self) -> None:
        factory = FakeV19HotFactory()
        engine = NativeHotH3Engine(factory, output_root=self.temporary)
        checkpoint_path = self.temporary / "checkpoints" / "v19.pt"
        output_path = self.temporary / "v19-checkpoint.mp4"
        spec = GenerationSpec.from_mapping({
            "prompt": "V19 invariant checkpoint route",
            "engine": "original",
            "mode": "advanced",
            "width": 1280,
            "height": 736,
            "duration_seconds": 5,
            "sampling_steps": 20,
            "acceleration": 50,
            "execution_mode": "checkpoint",
            "checkpoint_step": 10,
            "checkpoint_retain": True,
            "checkpoint_preview": True,
            "seed": 4404,
        })
        try:
            await engine.generate(
                spec, None, None, (), (), (), asyncio.Event(), output_path,
                checkpoint_path=checkpoint_path,
            )
            first = factory.sessions[0].requests[-1]
            await engine.generate(
                spec, None, None, (), (), (), asyncio.Event(), output_path,
                resume_checkpoint_path=checkpoint_path,
            )
            resumed = factory.sessions[0].requests[-1]
            self.assertEqual(first.scheduler_required_actual_step_indices, (9,))
            self.assertEqual(resumed.scheduler_required_actual_step_indices, (9,))
            self.assertEqual(first.preview_step_index, 9)
            self.assertIsNone(resumed.preview_step_index)
        finally:
            await engine.close()

    async def test_long_video_candidate_is_exactly_routed_and_does_not_leak(self) -> None:
        factory = FakeSparseHotFactory()
        engine = NativeHotH3Engine(factory, output_root=self.temporary)
        try:
            with (
                patch.dict(
                    "os.environ", {"H3_NATIVE_LONG_VIDEO_REVIEW": "1"}
                ),
                patch(
                    "h3serve.native_engine.detail_restore.restore_intrame_detail",
                    return_value=SimpleNamespace(elapsed_seconds=8.78),
                ) as detail_restore,
            ):
                eligible = GenerationSpec.from_mapping({
                    "prompt": "eligible long video",
                    "engine": "original",
                    "quality": "quality",
                    "resolution": "720p",
                    "aspect_ratio": "16:9",
                    "duration_seconds": 15,
                    "seed": 82303,
                })
                eligible_result = await engine.generate(
                    eligible, None, None, (), (), (), asyncio.Event(),
                    self.temporary / "eligible.mp4",
                )
                selected = factory.sessions[-1].requests[-1]
                self.assertEqual(
                    (
                        selected.terminal_refinement_initial_width,
                        selected.terminal_refinement_initial_height,
                        selected.terminal_refinement_steps,
                        selected.terminal_refinement_denoise,
                        selected.terminal_refinement_dense_tail_steps,
                    ),
                    (864, 480, 2, 0.025, 1),
                )
                self.assertTrue(
                    selected.execution_plan.long_video_motion_detail_attention
                )
                self.assertTrue(selected.execution_plan.fused_rms_adaln)
                self.assertTrue(
                    selected.execution_plan.vae_transformer_block_compile
                )
                self.assertEqual(
                    selected.execution_plan.dense_qk_quant_gran, "per_warp"
                )
                detail_restore.assert_called_once()
                self.assertEqual(
                    eligible_result.stage_seconds["intrame_detail_restore"], 8.78
                )

                short = GenerationSpec.from_mapping({
                    "prompt": "short request",
                    "engine": "original",
                    "quality": "quality",
                    "resolution": "720p",
                    "aspect_ratio": "16:9",
                    "duration_seconds": 5,
                    "seed": 82304,
                })
                await engine.generate(
                    short, None, None, (), (), (), asyncio.Event(),
                    self.temporary / "short.mp4",
                )
                excluded = factory.sessions[-1].requests[-1]
                self.assertIsNone(excluded.terminal_refinement_initial_width)
                self.assertEqual(excluded.terminal_refinement_steps, 0)
                self.assertFalse(
                    excluded.execution_plan.long_video_motion_detail_attention
                )
        finally:
            await engine.close()


if __name__ == "__main__":
    unittest.main()
