from __future__ import annotations

import dataclasses
import json
import math
import tempfile
import unittest
from pathlib import Path, PurePosixPath

import torch

from h3serve.app import JobRecord, JobService
from h3serve.native_engine.hot_session import (
    HotSessionRequest,
    NativeT2AVHotSession,
    restore_selflift_target_prefix_,
    selflift_renoise_clean_endpoint,
)
from h3serve.contract import (
    FPS,
    LORA_PRESETS,
    ORIGINAL_PRESETS,
    GenerationSpec,
    resolve_frames,
    resolve_geometry,
)


SERVE_ROOT = Path(__file__).resolve().parents[2]


def latent_shape(spec: GenerationSpec) -> dict[str, tuple[int, ...]]:
    video_t = 2 if spec.frames <= 5 else ((spec.frames - 5) // 17) * 5 + 2
    audio_t = round(spec.actual_duration_seconds * 40)
    return {
        "video": (1, 24, video_t, spec.height // 16, spec.width // 16),
        "audio": (1, 32, 2, audio_t),
    }


class GenerationPlanningContractTest(unittest.TestCase):
    def test_latent_only_finished_preview_is_a_valid_selflift_checkpoint_sink(self) -> None:
        request = HotSessionRequest(
            prompt="SelfLift latent-only audio finish",
            seed=8,
            width=1920,
            height=1088,
            frames=124,
            fps=24,
            steps=8,
            output_path=Path("unused.mp4"),
            use_lora=True,
            checkpoint_after_step=6,
            checkpoint_state_path=Path("checkpoint.pt"),
            preview_step_index=5,
            preview_output_path=None,
            preview_latents_path=Path("completed-low-resolution.pt"),
            preview_decode_mode="fast_finish",
            preview_branch_steps=2,
            multiscale_initial_width=960,
            multiscale_initial_height=544,
            multiscale_resize_after_step=5,
            multiscale_transition_mode="selflift_learned_x0",
        )

        request.validate()

    def test_selflift_base_and_larry_transition_contract_and_flow_state(self) -> None:
        request = HotSessionRequest(
            prompt="SelfLift",
            seed=9,
            width=1920,
            height=1088,
            frames=124,
            fps=24,
            steps=8,
            output_path=Path("unused.mp4"),
            use_lora=True,
            multiscale_initial_width=960,
            multiscale_initial_height=544,
            multiscale_resize_after_step=5,
            multiscale_transition_mode="selflift_learned_x0",
        )
        request.validate()
        clean = torch.full((1, 2, 1, 2, 2), 2.0)
        noise = torch.full_like(clean, -2.0)
        self.assertTrue(torch.equal(
            selflift_renoise_clean_endpoint(clean, noise, 0.25),
            torch.ones_like(clean),
        ))
        dataclasses.replace(request, use_lora=False).validate()

    def test_selflift_accepts_ref2va_media_and_canvas_keyframes(self) -> None:
        with tempfile.TemporaryDirectory(prefix="h3-selflift-ref2va-") as directory:
            reference = Path(directory) / "reference.png"
            reference.write_bytes(b"existence-only contract fixture")
            request = HotSessionRequest(
                prompt="SelfLift with stable packed reference rows",
                seed=10,
                width=1920,
                height=1088,
                frames=124,
                fps=24,
                steps=8,
                output_path=Path(directory) / "unused.mp4",
                use_lora=True,
                reference_images=(reference,),
                multiscale_initial_width=960,
                multiscale_initial_height=544,
                multiscale_resize_after_step=5,
                multiscale_transition_mode="selflift_learned_x0",
            )

            request.validate()

            dataclasses.replace(
                request,
                reference_images=(),
                first_frame=reference,
            ).validate()

    def test_selflift_dual_resolution_handoff_restores_exact_prefix(self) -> None:
        lifted = torch.zeros((1, 2, 5, 3, 4), dtype=torch.float32)
        prefix = torch.arange(48, dtype=torch.float32).reshape(1, 2, 2, 3, 4)

        restored = restore_selflift_target_prefix_(lifted, prefix)

        self.assertEqual(restored, 2)
        self.assertTrue(torch.equal(lifted[:, :, :2], prefix))
        self.assertTrue(torch.equal(lifted[:, :, 2:], torch.zeros_like(lifted[:, :, 2:])))
        self.assertEqual(restore_selflift_target_prefix_(lifted, None), 0)

    def spec(self, **overrides) -> GenerationSpec:
        request = {
            "prompt": "Contract fixture.",
            "engine": "original",
            "quality": "balanced",
            "resolution": "480p",
            "aspect_ratio": "16:9",
            "duration_seconds": 5,
            "seed": 4404,
        }
        request.update(overrides)
        return GenerationSpec.from_mapping(request)

    def test_geometry_is_the_resolved_canvas_not_the_marketing_label(self) -> None:
        self.assertEqual(resolve_geometry("360p", "16:9"), (640, 352))
        self.assertEqual(resolve_geometry("480p", "16:9"), (864, 480))
        self.assertEqual(resolve_geometry("720p", "16:9"), (1280, 736))
        self.assertEqual(resolve_geometry("1080p", "16:9"), (1920, 1088))
        self.assertEqual(resolve_geometry("2k", "16:9"), (2560, 1440))
        self.assertEqual(resolve_geometry("480p", "9:16"), (480, 864))
        for resolution in ("360p", "480p", "540p", "720p", "900p", "1080p", "2k"):
            for ratio in ("1:1", "4:3", "3:4", "16:9", "9:16"):
                width, height = resolve_geometry(resolution, ratio)
                self.assertEqual(width % 32, 0)
                self.assertEqual(height % 32, 0)

    def test_fixed_360p_preview_is_admitted_for_1080p_checkpoint(self) -> None:
        request = HotSessionRequest(
            prompt="fixed checkpoint preview",
            seed=1,
            width=1920,
            height=1088,
            frames=362,
            fps=24,
            steps=5,
            output_path=Path("unused.mp4"),
            preview_step_index=0,
            preview_output_path=Path("unused.preview.mp4"),
            preview_decode_mode="fast_finish",
            preview_branch_steps=4,
            preview_branch_spatial_scale=352 / 1088,
        )

        request.validate()

    def test_automatic_roi_refinement_admits_one_batched_sparse_atlas(self) -> None:
        with tempfile.TemporaryDirectory(prefix="h3-auto-roi-contract-") as directory:
            source = Path(directory) / "source.pt"
            source.write_bytes(b"runtime validates the latent payload")
            request = HotSessionRequest(
                prompt="automatic difficult-region refinement",
                seed=1,
                width=1920,
                height=1088,
                frames=124,
                fps=24,
                steps=4,
                output_path=Path(directory) / "unused.mp4",
                refinement_latents_path=source,
                refinement_denoise=0.20,
                refinement_spatial_mode="learned_3d",
                refinement_roi_auto=True,
                refinement_roi_max_regions=4,
                refinement_roi_steps=6,
                refinement_roi_atlas_height=48,
                refinement_roi_atlas_width=48,
                refinement_roi_atlas_rows=2,
                refinement_roi_atlas_columns=2,
                refinement_roi_attention_mode="scheduled_sparse",
            )
            request.validate()

            with self.assertRaisesRegex(ValueError, "mutually?.*exclusive"):
                dataclasses.replace(
                    request,
                    refinement_roi_regions=((0.1, 0.1, 0.2, 0.2),),
                ).validate()

    def test_pixel_video_can_supply_a_low_noise_repair_source(self) -> None:
        with tempfile.TemporaryDirectory(prefix="h3-video-repair-contract-") as directory:
            source = Path(directory) / "atlas.mp4"
            source.write_bytes(b"pixel-video placeholder")
            request = HotSessionRequest(
                prompt="source-preserving local restoration",
                seed=1,
                width=768,
                height=768,
                frames=22,
                fps=24,
                steps=4,
                output_path=Path(directory) / "unused.mp4",
                external_refinement_video_path=source,
                refinement_denoise=0.22,
                preserve_refinement_audio=False,
            )
            request.validate()

            latent = Path(directory) / "source.pt"
            latent.write_bytes(b"latent placeholder")
            with self.assertRaisesRegex(ValueError, "mutually exclusive"):
                dataclasses.replace(
                    request, refinement_latents_path=latent
                ).validate()

    def test_long_horizon_continuation_is_a_private_masked_av_transport(self) -> None:
        with tempfile.TemporaryDirectory(prefix="h3-continuation-contract-") as directory:
            source = Path(directory) / "source.pt"
            bridge = Path(directory) / "previous-condition.pt"
            source.write_bytes(b"placeholder validated by the sampler at runtime")
            bridge.write_bytes(b"placeholder validated by the sampler at runtime")
            request = HotSessionRequest(
                prompt="continue one scene",
                seed=2,
                width=864,
                height=480,
                frames=141,
                fps=24,
                steps=4,
                output_path=Path(directory) / "unused.mp4",
                continuation_latents_path=source,
                continuation_context_frames=22,
                save_final_latents_path=Path(directory) / "clean.pt",
                latent_only=True,
            )
            request.validate()

            dataclasses.replace(
                request,
                continuation_text_bridge_conditioning_path=bridge,
            ).validate()

            dataclasses.replace(
                request,
                frames=260,
                continuation_context_frames=90,
                continuation_video_prefix_frames=73,
                continuation_text_bridge_conditioning_path=bridge,
            ).validate()

            with self.assertRaisesRegex(ValueError, "non-empty exact video prefix"):
                dataclasses.replace(
                    request,
                    frames=260,
                    continuation_context_frames=90,
                    continuation_video_prefix_frames=0,
                    continuation_text_bridge_conditioning_path=bridge,
                ).validate()

            with self.assertRaisesRegex(ValueError, "requires continuation"):
                dataclasses.replace(
                    request,
                    continuation_latents_path=None,
                    continuation_context_frames=0,
                    continuation_text_bridge_conditioning_path=bridge,
                ).validate()

            dataclasses.replace(
                request,
                frames=260,
                continuation_context_frames=90,
                continuation_video_prefix_frames=0,
                continuation_audio_bridge_ticks=65,
            ).validate()

            with self.assertRaisesRegex(ValueError, "video prefix"):
                dataclasses.replace(
                    request,
                    continuation_video_prefix_frames=39,
                ).validate()

            with self.assertRaisesRegex(ValueError, "calibrated 90-frame bound"):
                dataclasses.replace(
                    request,
                    frames=260,
                    continuation_context_frames=107,
                ).validate()

            with self.assertRaisesRegex(ValueError, "requires continuation"):
                dataclasses.replace(
                    request,
                    continuation_latents_path=None,
                ).validate()

    def test_selflift_continuation_accepts_a_complete_formal_trajectory(self) -> None:
        with tempfile.TemporaryDirectory(prefix="h3-selflift-direct-") as directory:
            source = Path(directory) / "previous-high-resolution.pt"
            source.write_bytes(b"runtime validates the latent payload")
            request = HotSessionRequest(
                prompt="continue directly through the formal SelfLift tail",
                seed=3,
                width=1280,
                height=736,
                frames=141,
                fps=24,
                steps=8,
                output_path=Path(directory) / "unused.mp4",
                continuation_latents_path=source,
                continuation_context_frames=22,
                save_final_latents_path=Path(directory) / "clean.pt",
                latent_only=True,
                multiscale_initial_width=960,
                multiscale_initial_height=544,
                multiscale_resize_after_step=5,
                multiscale_transition_mode="selflift_learned_x0",
            )

            request.validate()

            with self.assertRaisesRegex(
                ValueError, "complete formal trajectory or while checkpointing"
            ):
                dataclasses.replace(
                    request,
                    checkpoint_after_step=6,
                    checkpoint_state_path=Path(directory) / "fork.pt",
                ).validate()

    def test_bounded_memory_composes_with_either_public_conditioning_entry(self) -> None:
        with tempfile.TemporaryDirectory(prefix="h3-memory-condition-contract-") as directory:
            root = Path(directory)
            memory = root / "memory.pt"
            image = root / "reference.png"
            audio = root / "reference.wav"
            endpoint = root / "last.png"
            for path in (memory, image, audio, endpoint):
                path.write_bytes(b"existence-only contract fixture")

            base = HotSessionRequest(
                prompt="bounded memory composition",
                seed=2,
                width=864,
                height=480,
                frames=141,
                fps=24,
                steps=7,
                output_path=root / "unused.mp4",
                av_token_memory_path=memory,
                save_final_latents_path=root / "clean.pt",
                latent_only=True,
            )
            dataclasses.replace(base, last_frame=endpoint).validate()
            dataclasses.replace(
                base,
                reference_images=(image,),
                reference_audios=(audio,),
            ).validate()

            with self.assertRaisesRegex(ValueError, "separate public inputs"):
                dataclasses.replace(
                    base,
                    last_frame=endpoint,
                    reference_images=(image,),
                ).validate()

    def test_global_co_denoise_can_return_one_latent_only_solver_state(self) -> None:
        with tempfile.TemporaryDirectory(prefix="h3-global-latent-contract-") as directory:
            root = Path(directory)
            conditions = []
            for index in range(3):
                path = root / f"condition-{index}.pt"
                path.write_bytes(b"validated by conditioning loader at runtime")
                conditions.append(path)
            request = HotSessionRequest(
                prompt="window zero",
                seed=2,
                width=320,
                height=192,
                frames=311,
                fps=24,
                steps=7,
                output_path=root / "unused.mp4",
                conditioning_cache_source_path=conditions[0],
                global_co_denoise_output_frames=719,
                global_co_denoise_prompts=(
                    "window zero",
                    "window one",
                    "window two",
                ),
                global_co_denoise_conditioning_paths=tuple(conditions),
                global_co_denoise_rotary_mode="window_local",
                save_final_latents_path=root / "audio-spine.pt",
                latent_only=True,
            )

            request.validate()

            with self.assertRaisesRegex(ValueError, "rotary mode"):
                dataclasses.replace(
                    request,
                    global_co_denoise_rotary_mode="invalid",
                ).validate()

    def test_global_selflift_requires_and_accepts_one_global_solver_state(self) -> None:
        with tempfile.TemporaryDirectory(prefix="h3-global-selflift-contract-") as directory:
            root = Path(directory)
            source = root / "global-source-x0.pt"
            source.write_bytes(b"payload validated when the runtime loads it")
            conditions = []
            for index in range(3):
                path = root / f"condition-{index}.pt"
                path.write_bytes(b"conditioning fixture")
                conditions.append(path)
            request = HotSessionRequest(
                prompt="window zero",
                seed=2,
                width=1920,
                height=1088,
                frames=311,
                fps=24,
                steps=8,
                output_path=root / "unused.mp4",
                use_lora=True,
                global_co_denoise_output_frames=719,
                global_co_denoise_prompts=(
                    "window zero",
                    "window one",
                    "window two",
                ),
                global_co_denoise_conditioning_paths=tuple(conditions),
                global_selflift_source_path=source,
            )

            request.validate()

            with self.assertRaisesRegex(ValueError, "sigma scale"):
                dataclasses.replace(
                    request,
                    global_selflift_sigma_scale=0.2,
                ).validate()

            with self.assertRaisesRegex(ValueError, "requires global co-denoise"):
                dataclasses.replace(
                    request,
                    global_co_denoise_output_frames=None,
                    global_co_denoise_prompts=(),
                    global_co_denoise_conditioning_paths=(),
                ).validate()

    def test_time_and_latent_grids_are_fully_resolved_before_engine_entry(self) -> None:
        five = self.spec(duration_seconds=5)
        self.assertEqual(resolve_frames(5), (124, 124 / FPS))
        self.assertEqual(latent_shape(five), {
            "video": (1, 24, 37, 30, 54),
            "audio": (1, 32, 2, 207),
        })
        fifteen = self.spec(duration_seconds=15)
        self.assertEqual(resolve_frames(15), (362, 362 / FPS))
        self.assertEqual(latent_shape(fifteen), {
            "video": (1, 24, 107, 30, 54),
            "audio": (1, 32, 2, 603),
        })
        for seconds in (1, 2.5, 3, 5, 10, 15):
            spec = self.spec(duration_seconds=seconds)
            self.assertEqual((spec.frames - 5) % 17, 0)
            self.assertTrue(math.isclose(
                spec.actual_duration_seconds, spec.frames / FPS
            ))

    def test_presets_keep_exact_algorithms_not_only_step_counts(self) -> None:
        self.assertEqual(
            ORIGINAL_PRESETS["fast"]["actual_step_indices"],
            [0, 1, 2, 3, 4, 8, 13, 19],
        )
        self.assertEqual(ORIGINAL_PRESETS["balanced"]["actual_steps"], 9)
        self.assertEqual(ORIGINAL_PRESETS["balanced"]["forecast_steps"], 11)
        self.assertEqual(ORIGINAL_PRESETS["quality"]["actual_steps"], 12)
        self.assertEqual(ORIGINAL_PRESETS["ultra"]["actual_steps"], 20)
        self.assertEqual(ORIGINAL_PRESETS["ultra"]["forecast_steps"], 0)
        self.assertEqual(
            {name: preset["steps"] for name, preset in LORA_PRESETS.items()},
            {"fast": 4, "balanced": 5, "quality": 6, "ultra": 8},
        )
        self.assertTrue(all(preset["strength"] == 1.0 for preset in LORA_PRESETS.values()))

    def test_request_round_trip_preserves_seed_and_resolved_plan(self) -> None:
        original = self.spec(
            engine="lora", quality="quality", duration_seconds=15,
            resolution="720p", aspect_ratio="9:16", seed=2**64 - 1,
        )
        document = json.loads(json.dumps(
            original.to_dict(include_execution=True), ensure_ascii=False
        ))
        restored = GenerationSpec.from_mapping(document)
        self.assertEqual(restored, original)
        self.assertEqual(document["execution"]["steps"], 6)


class ArtifactPathContractTest(unittest.TestCase):
    def test_uploaded_media_cache_identity_follows_content_not_job_path(self) -> None:
        with tempfile.TemporaryDirectory(prefix="h3-content-cache-") as directory:
            root = Path(directory)
            first = root / "job-a" / "first.png"
            second = root / "job-b" / "first.png"
            first.parent.mkdir()
            second.parent.mkdir()
            first.write_bytes(b"same uploaded image bytes")
            second.write_bytes(first.read_bytes())
            self.assertNotEqual(first.resolve(), second.resolve())
            self.assertEqual(
                NativeT2AVHotSession._file_content_digest(first),
                NativeT2AVHotSession._file_content_digest(second),
            )

    def test_manifest_has_one_safe_pinned_artifact_per_required_role(self) -> None:
        manifest = json.loads(
            (SERVE_ROOT / "models/manifest.json").read_text(encoding="utf-8")
        )
        roles = [artifact["role"] for artifact in manifest["artifacts"]]
        required_roles = {
            "diffusion_model", "reference_diffusion_model",
            "diffusion_model_w4a8", "reference_diffusion_model_w4a8",
            "text_encoder", "video_vae", "audio_vae", "turbo_lora",
            "latent_upscaler",
        }
        optional_release_roles = {
            "lightx2v_fl2va_4step_lora",
            "lightx2v_fl2va_8step_lora",
            "lightx2v_ref2va_4step_lora",
            "temporal_second_sampling_dit",
            "temporal_second_sampling_lq_projection",
            "temporal_second_sampling_decoder",
            "temporal_second_sampling_prompt",
        }
        self.assertTrue(required_roles.issubset(roles))
        self.assertTrue(set(roles).issubset(required_roles | optional_release_roles))
        self.assertTrue(all(roles.count(role) == 1 for role in set(roles)))
        seen_paths = set()
        for artifact in manifest["artifacts"]:
            install = PurePosixPath(artifact["install_path"])
            self.assertFalse(install.is_absolute())
            self.assertNotIn("..", install.parts)
            self.assertNotIn(str(install), seen_paths)
            seen_paths.add(str(install))
            self.assertGreater(artifact["bytes"], 0)
            self.assertRegex(artifact["sha256"], r"^[0-9a-f]{64}$")
            self.assertRegex(artifact["revision"], r"^[0-9a-f]{40}$")

    def test_persisted_job_does_not_recompute_duration_or_seed(self) -> None:
        class NoopBackend:
            key = None

        with tempfile.TemporaryDirectory(prefix="h3-persist-contract-") as directory:
            data = Path(directory)
            service = JobService(data, NoopBackend())
            job = JobRecord(
                id="round-trip",
                spec=GenerationSpec.from_mapping({
                    "prompt": "fifteen seconds",
                    "duration_seconds": 15,
                    "seed": 18446744073709551615,
                }),
            )
            service.jobs[job.id] = job
            service.persist(job)
            restored = JobService(data, NoopBackend()).jobs[job.id]
            self.assertEqual(restored.spec.frames, 362)
            self.assertEqual(restored.spec.requested_duration_seconds, 15)
            self.assertEqual(restored.spec.seed, 2**64 - 1)


class AcceptancePolicyContractTest(unittest.TestCase):
    def test_contract_makes_visual_gate_prior_and_ssim_non_blocking(self) -> None:
        contract = (SERVE_ROOT / "docs/COMFY_MIGRATION_CONTRACT.md").read_text(
            encoding="utf-8"
        )
        visual = contract.index("先**通过多帧视觉门控")
        numeric = contract.index("通过视觉门控后才记录")
        self.assertLess(visual, numeric)
        self.assertIn("SSIM 只作同 seed 数值差异诊断", contract)
        self.assertIn("不得**作为质量硬门", contract)


if __name__ == "__main__":
    unittest.main()
