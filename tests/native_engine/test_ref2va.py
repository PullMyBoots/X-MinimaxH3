from __future__ import annotations

import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace

import av
import numpy as np
import torch

from h3serve.contract import GenerationSpec, default_quality, public_options
from h3serve.native_engine.model.packed import (
    build_hybrid_condition_layout,
    build_ref2va_layout,
)
from h3serve.native_engine.adapters.conditioning_vae.preprocess import (
    _reference_geometry,
    prepare_reference_audios,
    prepare_reference_images,
    prepare_reference_videos,
)
from h3serve.native_engine.adapters.conditioning_vae.contracts import PreparedReferenceVideo
from h3serve.native_engine.adapters.conditioning_vae.qwen_quantized import PackedQwen3VLT2AVConditioner
from h3serve.native_engine.adapters.conditioning_vae.video_vae import H3VideoVAEAdapter
from h3serve.native_engine.hot_session import (
    HotSessionRequest,
    NativeT2AVHotSession,
    _authority_conditioning_route_name,
    _blend_continuation_boundary_velocity,
    _blend_layout_guidance_velocity,
    _compose_authority_routed_conditioning,
    _compose_conditioning,
    _continuation_text_route_name,
    _explicit_camera_layout_guidance_schedule,
    _layout_guidance_route_name,
    _progressive_conditioning_route_name,
)


class Ref2VAContractTest(unittest.TestCase):
    def test_continuation_boundary_auxiliary_runs_at_every_solver_step(self) -> None:
        self.assertEqual(
            [_continuation_text_route_name(index, 7) for index in range(7)],
            ["boundary"] * 7,
        )

    def test_continuation_boundary_velocity_is_local_to_writable_band(self) -> None:
        current = torch.zeros((1, 2, 20, 1, 1), dtype=torch.float32)
        boundary = torch.full_like(current, 10.0)
        blended = _blend_continuation_boundary_velocity(
            current,
            boundary,
            protected_prefix_tokens=5,
            hidden_repaint_tokens=3,
            visible_fade_tokens=4,
            peak=1.0,
        )
        self.assertTrue(torch.equal(blended[:, :, :5], current[:, :, :5]))
        self.assertTrue(torch.equal(blended[:, :, 12:], current[:, :, 12:]))
        self.assertTrue(torch.allclose(
            blended[:, :, 5:8], torch.full_like(blended[:, :, 5:8], 10.0)
        ))
        self.assertTrue(torch.allclose(
            blended[:, :, 8], torch.full_like(blended[:, :, 8], 7.5)
        ))
        self.assertTrue(torch.allclose(
            blended[:, :, 9], torch.full_like(blended[:, :, 9], 5.0)
        ))
        self.assertTrue(torch.allclose(
            blended[:, :, 10], torch.full_like(blended[:, :, 10], 2.5)
        ))
        self.assertTrue(torch.equal(blended[:, :, 11], current[:, :, 11]))

    def test_explicit_layout_anchor_keeps_state_primary_and_guides_middle_steps(self) -> None:
        layout = torch.full((1, 24, 1, 4, 6), 10.0)
        layout_late = torch.full((1, 24, 1, 4, 6), 11.0)
        state = torch.full((1, 24, 1, 4, 6), 20.0)
        routes, profile = _compose_authority_routed_conditioning(
            user_video_latents=(),
            user_reference_shapes=(),
            user_reference_kinds=(),
            user_audio_latents=(),
            user_audio_frames=(),
            memory_video_latents=(state,),
            memory_reference_shapes=((1, 4, 6),),
            memory_reference_kinds=("image",),
            memory_audio_latents=(),
            memory_audio_frames=(),
            keyframe_latents=(),
            keyframe_indices=(),
            user_references_requested=False,
            memory_layout_video_latents=(layout, layout_late),
            memory_layout_reference_shapes=((1, 4, 6), (1, 4, 6)),
            memory_layout_reference_kinds=("image", "image"),
            state_seeded_layout_memory=True,
        )
        self.assertEqual(set(routes), {"reference", "history"})
        self.assertIs(routes["reference"].video_latents[0], layout)
        self.assertIs(routes["reference"].video_latents[1], layout_late)
        self.assertIs(routes["history"].video_latents[0], state)
        self.assertEqual(
            profile["visual_policy"],
            "state_primary_layout_velocity_guidance",
        )
        self.assertEqual(
            [
                _progressive_conditioning_route_name(
                    index,
                    7,
                    visual_schedule=True,
                    inferred_voice_bootstrap=False,
                    layout_memory_bootstrap=True,
                    state_seeded_layout_bootstrap=True,
                )
                for index in range(7)
            ],
            ["history"] * 7,
        )
        self.assertEqual(_explicit_camera_layout_guidance_schedule(7), (2, 3))
        self.assertEqual(_explicit_camera_layout_guidance_schedule(3), (1, 1))
        self.assertEqual(_layout_guidance_route_name("history"), "reference")
        self.assertEqual(
            _layout_guidance_route_name("history_release"),
            "reference_release",
        )
        self.assertEqual(
            [
                _progressive_conditioning_route_name(
                    index,
                    7,
                    visual_schedule=True,
                    inferred_voice_bootstrap=False,
                    layout_memory_bootstrap=True,
                )
                for index in range(7)
            ],
            ["reference"] * 3 + ["history"] * 4,
        )

    def test_layout_guidance_blends_one_full_video_velocity_only(self) -> None:
        state = torch.zeros((1, 2, 3, 2, 2), dtype=torch.float32)
        layout = torch.full_like(state, 8.0)
        blended = _blend_layout_guidance_velocity(
            state,
            layout,
            weight=0.25,
        )
        self.assertTrue(torch.equal(blended, torch.full_like(state, 2.0)))

    def test_novel_camera_uses_one_layout_probe_between_text_only_steps(self) -> None:
        layout = torch.full((1, 24, 1, 4, 6), 8.0)
        routes, profile = _compose_authority_routed_conditioning(
            user_video_latents=(),
            user_reference_shapes=(),
            user_reference_kinds=(),
            user_audio_latents=(),
            user_audio_frames=(),
            memory_video_latents=(),
            memory_reference_shapes=(),
            memory_reference_kinds=(),
            memory_audio_latents=(),
            memory_audio_frames=(),
            keyframe_latents=(),
            keyframe_indices=(),
            user_references_requested=False,
            memory_layout_video_latents=(layout,),
            memory_layout_reference_shapes=((1, 4, 6),),
            memory_layout_reference_kinds=("image",),
            novel_camera_layout_probe=True,
        )
        self.assertEqual(set(routes), {"reference", "history"})
        self.assertIs(routes["reference"].video_latents[0], layout)
        self.assertEqual(routes["history"].video_latents, ())
        self.assertEqual(
            profile["visual_policy"],
            "target_camera_seed_layout_probe_text_convergence",
        )
        self.assertEqual(
            [
                _progressive_conditioning_route_name(
                    index,
                    7,
                    visual_schedule=True,
                    inferred_voice_bootstrap=False,
                    layout_memory_bootstrap=True,
                    novel_camera_layout_probe=True,
                )
                for index in range(7)
            ],
            [
                "history",
                "history",
                "reference",
                "history",
                "history",
                "history",
                "history",
            ],
        )

    def test_authority_routing_never_presents_user_and_memory_as_duplicate_evidence(self) -> None:
        user_image = torch.full((1, 24, 1, 4, 6), 11.0)
        memory_image = torch.full((1, 24, 1, 4, 6), 22.0)
        user_audio = torch.full((1, 32, 2, 3), 33.0)
        memory_audio = torch.full((1, 32, 2, 1), 44.0)
        routes, profile = _compose_authority_routed_conditioning(
            user_video_latents=(user_image,),
            user_reference_shapes=((1, 4, 6),),
            user_reference_kinds=("image",),
            user_audio_latents=(user_audio,),
            user_audio_frames=(3,),
            memory_video_latents=(memory_image,),
            memory_reference_shapes=((1, 4, 6),),
            memory_reference_kinds=("image",),
            memory_audio_latents=(memory_audio,),
            memory_audio_frames=(1,),
            keyframe_latents=(),
            keyframe_indices=(),
            user_references_requested=True,
        )
        self.assertEqual(set(routes), {"history", "reference"})
        self.assertIs(routes["history"].video_latents[0], memory_image)
        self.assertIs(routes["reference"].video_latents[0], user_image)
        self.assertIs(routes["history"].audio_latents[0], user_audio)
        self.assertIs(routes["reference"].audio_latents[0], user_audio)
        self.assertTrue(all(
            latent is not memory_audio
            for latent in routes["history"].audio_latents
        ))
        self.assertEqual(profile["suppressed_memory_audio_references"], 1)
        self.assertFalse(
            profile["simultaneous_user_and_memory_visual_references"]
        )
        self.assertEqual(
            [_authority_conditioning_route_name(index, 7) for index in range(7)],
            ["history"] * 7,
        )

    def test_user_reference_media_precedes_and_survives_bounded_memory_merge(self) -> None:
        user_image = torch.full((1, 24, 1, 4, 6), 11.0)
        memory_image = torch.full((1, 24, 1, 4, 6), 22.0)
        last_frame = torch.full((1, 24, 1, 4, 6), 33.0)
        user_audio = torch.full((1, 32, 2, 3), 44.0)
        memory_audio = torch.full((1, 32, 2, 1), 55.0)
        composition = _compose_conditioning(
            user_video_latents=(user_image,),
            user_reference_shapes=((1, 4, 6),),
            user_reference_kinds=("image",),
            user_audio_latents=(user_audio,),
            user_audio_frames=(3,),
            memory_video_latents=(memory_image,),
            memory_reference_shapes=((1, 4, 6),),
            memory_reference_kinds=("image",),
            memory_audio_latents=(memory_audio,),
            memory_audio_frames=(1,),
            keyframe_latents=(last_frame,),
            keyframe_indices=(-1,),
            user_references_requested=True,
        )
        self.assertIs(composition.video_latents[0], user_image)
        self.assertIs(composition.video_latents[1], memory_image)
        self.assertIs(composition.video_latents[2], last_frame)
        self.assertIs(composition.audio_latents[0], user_audio)
        self.assertIs(composition.audio_latents[1], memory_audio)
        self.assertEqual(composition.reference_shapes, ((1, 4, 6), (1, 4, 6)))
        self.assertEqual(composition.reference_audio_frames, (3, 1))
        self.assertEqual(composition.profile["mode"], "hybrid_reference_keyframe")
        self.assertTrue(composition.profile["user_conditions_retained"])

    def test_inferred_voice_is_present_only_on_bootstrap_route(self) -> None:
        memory_image = torch.full((1, 24, 1, 4, 6), 22.0)
        memory_audio = torch.full((1, 32, 2, 20), 55.0)
        routes, profile = _compose_authority_routed_conditioning(
            user_video_latents=(),
            user_reference_shapes=(),
            user_reference_kinds=(),
            user_audio_latents=(),
            user_audio_frames=(),
            memory_video_latents=(memory_image,),
            memory_reference_shapes=((1, 4, 6),),
            memory_reference_kinds=("image",),
            memory_audio_latents=(memory_audio,),
            memory_audio_frames=(20,),
            keyframe_latents=(),
            keyframe_indices=(),
            user_references_requested=False,
        )
        self.assertEqual(set(routes), {"default_voice", "default_release"})
        self.assertIs(routes["default_voice"].audio_latents[0], memory_audio)
        self.assertEqual(routes["default_release"].audio_latents, ())
        self.assertIs(routes["default_voice"].video_latents[0], memory_image)
        self.assertIs(routes["default_release"].video_latents[0], memory_image)
        self.assertEqual(
            profile["audio_policy"],
            "inferred_short_voice_high_noise_bootstrap_v1",
        )
        self.assertEqual(
            [
                _progressive_conditioning_route_name(
                    index,
                    12,
                    visual_schedule=False,
                    inferred_voice_bootstrap=True,
                )
                for index in range(12)
            ],
            ["default_voice"] * 4 + ["default_release"] * 8,
        )

    def test_ref2va_keeps_frozen_inferred_voice_on_every_denoise_route(self) -> None:
        memory_image = torch.full((1, 24, 1, 4, 6), 22.0)
        memory_audio = torch.full((1, 32, 2, 120), 55.0)
        routes, profile = _compose_authority_routed_conditioning(
            user_video_latents=(),
            user_reference_shapes=(),
            user_reference_kinds=(),
            user_audio_latents=(),
            user_audio_frames=(),
            memory_video_latents=(memory_image,),
            memory_reference_shapes=((1, 4, 6),),
            memory_reference_kinds=("image",),
            memory_audio_latents=(memory_audio,),
            memory_audio_frames=(120,),
            keyframe_latents=(),
            keyframe_indices=(),
            user_references_requested=False,
            supports_persistent_inferred_audio=True,
        )
        self.assertEqual(set(routes), {"default"})
        self.assertIs(routes["default"].audio_latents[0], memory_audio)
        self.assertEqual(
            profile["audio_policy"],
            "ref2va_persistent_self_anchor_voice_v1",
        )

    def test_audio_only_and_silent_memory_routes_need_no_visual_references(self) -> None:
        voice = torch.full((1, 32, 2, 120), 55.0)
        arguments = dict(
            user_video_latents=(), user_reference_shapes=(),
            user_reference_kinds=(), user_audio_latents=(), user_audio_frames=(),
            memory_video_latents=(), memory_reference_shapes=(),
            memory_reference_kinds=(), keyframe_latents=(), keyframe_indices=(),
            user_references_requested=False, supports_persistent_inferred_audio=True,
        )
        for audio, frames in (((), ()), ((voice,), (120,))):
            routes, profile = _compose_authority_routed_conditioning(
                **arguments, memory_audio_latents=audio, memory_audio_frames=frames,
            )
            self.assertEqual(set(routes), {"default"})
            self.assertEqual(routes["default"].video_latents, ())
            self.assertEqual(routes["default"].reference_shapes, ())
            self.assertEqual(routes["default"].reference_audio_frames, frames)
            if audio:
                self.assertIs(routes["default"].audio_latents[0], voice)
                self.assertEqual(profile["audio_policy"], "ref2va_persistent_self_anchor_voice_v1")
            else:
                self.assertEqual(routes["default"].profile["mode"], "no_reference_rows")
        arguments["user_references_requested"] = True
        with self.assertRaisesRegex(ValueError, "composition cannot be empty"):
            _compose_authority_routed_conditioning(
                **arguments, memory_audio_latents=(), memory_audio_frames=(),
            )

    def test_public_audio_remains_authoritative_across_visual_routes(self) -> None:
        self.assertEqual(
            [
                _progressive_conditioning_route_name(
                    index,
                    7,
                    visual_schedule=True,
                    inferred_voice_bootstrap=False,
                )
                for index in range(7)
            ],
            ["history"] * 7,
        )

    def test_request_planner_counts_keyframe_and_memory_tokens_together(self) -> None:
        from h3serve.native_engine.av_token_memory import empty_av_token_memory

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            memory_path = root / "memory.pt"
            memory = empty_av_token_memory()
            memory["video_entries"] = [
                {
                    "position": index,
                    "latent": torch.zeros((1, 24, 1, 30, 54)),
                }
                for index in (0, 100)
            ]
            memory["audio_entries"] = [{
                "position": 0,
                "latent": torch.zeros((1, 32, 2, 20)),
            }]
            torch.save(memory, memory_path)
            session = NativeT2AVHotSession.__new__(NativeT2AVHotSession)
            session.engine = "lora"
            features = session._analyze_request_features(
                HotSessionRequest(
                    prompt="planner contract",
                    seed=1,
                    width=864,
                    height=480,
                    frames=362,
                    fps=24,
                    steps=7,
                    output_path=root / "unused.mp4",
                    last_frame=root / "last.png",
                    av_token_memory_path=memory_path,
                    use_lora=True,
                ),
                text_tokens=100,
            )

        spatial_tokens = (864 // 32) * (480 // 32)
        self.assertEqual(features.condition_count, 4)
        self.assertEqual(
            features.condition_tokens,
            spatial_tokens + 2 * spatial_tokens + 40,
        )

    def test_qwen_vision_cache_key_is_content_and_geometry_addressed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "person.png"
            path.write_bytes(b"first")
            conditioner = PackedQwen3VLT2AVConditioner.__new__(
                PackedQwen3VLT2AVConditioner
            )
            request = SimpleNamespace(
                reference_images=(path,), reference_videos=(),
                first_frame=None, last_frame=None,
                width=864, height=480, num_frames=192,
                reference_image_resolution="720p",
                reference_video_resolution="360p",
            )
            first = conditioner._vision_cache_key(request)
            path.write_bytes(b"second")
            second = conditioner._vision_cache_key(request)
            request.width = 896
            third = conditioner._vision_cache_key(request)
            request.reference_image_resolution = "480p"
            fourth = conditioner._vision_cache_key(request)
            self.assertNotEqual(first, second)
            self.assertNotEqual(second, third)
            self.assertNotEqual(third, fourth)

    def test_reference_resolution_changes_pixels_not_aspect_or_composition(self) -> None:
        geometry = _reference_geometry(1536, 1024, "720p")
        self.assertEqual(geometry["content_size"], (1080, 720))
        self.assertEqual(geometry["canvas_size"], (1080, 720))
        self.assertEqual(geometry["padding"], (0, 0, 0, 0))
        self.assertAlmostEqual(
            geometry["content_size"][0] / geometry["content_size"][1],
            1536 / 1024,
        )

        # Smaller media is not enlarged and its public canvas is untouched.
        original = _reference_geometry(641, 359, "720p")
        self.assertEqual(original["content_size"], (641, 359))
        self.assertEqual(original["canvas_size"], (641, 359))

        # The P level caps the short edge; it never forces landscape media or
        # rewrites a portrait/panoramic asset onto a preset canvas.
        portrait = _reference_geometry(1024, 1536, "720p")
        self.assertEqual(portrait["content_size"], (720, 1080))
        self.assertAlmostEqual(720 / 1080, 1024 / 1536)
        panoramic = _reference_geometry(2048, 512, "360p")
        self.assertEqual(panoramic["content_size"], (1440, 360))
        self.assertAlmostEqual(1440 / 360, 2048 / 512)

    def test_reference_image_policy_preserves_exact_resized_canvas(self) -> None:
        from PIL import Image

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "wide.png"
            pixels = np.zeros((1024, 1536, 3), dtype=np.uint8)
            pixels[:, :32] = (255, 0, 0)
            pixels[:, -32:] = (0, 0, 255)
            Image.fromarray(pixels).save(path)
            request = SimpleNamespace(
                reference_images=(path,),
                reference_image_resolution="720p",
            )
            image = prepare_reference_images(request)[0]
            self.assertEqual(image.size, (1080, 720))
            output = np.asarray(image)
            # Both source edges remain present without crop, stretch or pad.
            self.assertGreater(output[:, 0, 0].mean(), 240)
            self.assertGreater(output[:, -1, 2].mean(), 240)

    def test_qwen_video_mrope_expands_temporal_blocks(self) -> None:
        ids = torch.arange(11)
        token_types = torch.tensor([0, 0] + [2] * 4 + [0] + [2] * 4)
        positions = PackedQwen3VLT2AVConditioner._mrope_position_ids(
            ids, token_types, 2, video_grid_thw=torch.tensor([[2, 4, 4]])
        )
        self.assertEqual(tuple(positions.shape), (3, 11))
        self.assertTrue(torch.equal(positions[:, :2], torch.tensor([[0, 1], [0, 1], [0, 1]])))

    def test_video_vae_reference_adapter_uses_video_encoder(self) -> None:
        class FakeVAE(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.marker = torch.nn.Parameter(torch.zeros(()))
                self.calls = 0
            def decode(self, value):
                return value
            def encode_videos(self, videos, use_fp16_latent=False):
                self.calls += 1
                self.assertion = (videos[0].shape, use_fp16_latent)
                return [torch.ones((24, 2, 4, 4), dtype=torch.float32)]
        model = FakeVAE()
        adapter = H3VideoVAEAdapter(model, latents_mean=[0.0] * 24, latents_std=[1.0] * 24)
        prepared = PreparedReferenceVideo(
            frames=np.zeros((5, 48, 64, 3), dtype=np.uint8),
            qwen_frames=np.zeros((2, 48, 64, 3), dtype=np.uint8),
            qwen_block_timestamps=(0.25,), source_fps=24.0, source_duration_seconds=2.0,
        )
        request = SimpleNamespace(
            reference_images=(), reference_videos=(Path("placeholder.mp4"),),
            prepared_reference_videos=(prepared,),
        )
        result = adapter.encode_references(request)
        self.assertEqual(model.calls, 1)
        self.assertEqual(model.assertion, ((5, 64, 64, 3), True))
        self.assertEqual(result.kinds, ("video",))
        self.assertEqual(result.latent_shapes, ((2, 4, 4),))
        self.assertTrue(torch.equal(result.latents[0], torch.ones((1, 24, 2, 4, 4))))

    def test_low_vram_reference_encode_keeps_fp16_vae_weights(self) -> None:
        class FakeVAE(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.marker = torch.nn.Parameter(
                    torch.zeros((), dtype=torch.float16)
                )
                self.dtype_transitions = []

            def decode(self, value):
                return value

            def to(self, *args, **kwargs):
                dtype = kwargs.get("dtype")
                if dtype is None and args and isinstance(args[0], torch.dtype):
                    dtype = args[0]
                if dtype is not None:
                    self.dtype_transitions.append(dtype)
                return super().to(*args, **kwargs)

            def encode_images(self, images, use_fp16_latent=False):
                self.encode_observation = (
                    self.marker.dtype,
                    use_fp16_latent,
                )
                return [torch.ones((24, 1, 4, 4), dtype=torch.float16)]

        from PIL import Image

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reference.png"
            Image.new("RGB", (64, 64), (32, 64, 96)).save(path)
            model = FakeVAE()
            adapter = H3VideoVAEAdapter(
                model,
                latents_mean=[0.0] * 24,
                latents_std=[1.0] * 24,
                encode_precision="fp16_weights_fp32_posterior",
            )
            result = adapter.encode_references(
                SimpleNamespace(
                    reference_images=(path,),
                    reference_videos=(),
                    reference_image_resolution="720p",
                )
            )

        self.assertEqual(model.encode_observation, (torch.float16, True))
        self.assertEqual(model.dtype_transitions, [])
        self.assertEqual(model.marker.dtype, torch.float16)
        self.assertEqual(result.kinds, ("image",))
        self.assertTrue(torch.isfinite(result.latents[0]).all())

    def test_video_vae_encode_precision_is_validated(self) -> None:
        class FakeVAE(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.marker = torch.nn.Parameter(torch.zeros(()))

            def decode(self, value):
                return value

        with self.assertRaisesRegex(ValueError, "encode precision"):
            H3VideoVAEAdapter(
                FakeVAE(),
                latents_mean=[0.0] * 24,
                latents_std=[1.0] * 24,
                encode_precision="unsupported",
            )

    def test_reference_video_decode_resample_and_temporal_alignment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reference.mp4"
            with av.open(str(path), "w") as container:
                stream = container.add_stream("mpeg4", rate=24)
                stream.width, stream.height, stream.pix_fmt = 640, 480, "yuv420p"
                for index in range(48):
                    pixels = np.zeros((480, 640, 3), dtype=np.uint8)
                    pixels[:, :, 0] = index
                    pixels[:, :24, 1] = 255
                    pixels[:, -24:, 2] = 255
                    frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
                    for packet in stream.encode(frame):
                        container.mux(packet)
                for packet in stream.encode():
                    container.mux(packet)
            request = SimpleNamespace(
                reference_videos=(path,), num_frames=39,
                reference_video_resolution="360p",
            )
            prepared = prepare_reference_videos(request)
            # 640x480 becomes exactly 480x360. It is never forced into 16:9
            # and the user-facing preprocessing stage adds no padding.
            self.assertEqual(prepared[0].frames.shape, (39, 360, 480, 3))
            self.assertGreater(prepared[0].frames[:, :, 0, 1].mean(), 180)
            self.assertGreater(prepared[0].frames[:, :, -1, 2].mean(), 180)
            self.assertEqual(prepared[0].qwen_frames.shape[0] % 2, 0)
            self.assertEqual(len(prepared[0].qwen_block_timestamps), prepared[0].qwen_frames.shape[0] // 2)
            self.assertAlmostEqual(prepared[0].source_duration_seconds, 2.0, places=2)

    def test_reference_service_defaults_to_balanced_original_weight_schedule(self) -> None:
        spec = GenerationSpec.from_mapping({
            "prompt": "<Picture 1> walks into the market",
            "engine": "reference",
            "quality": default_quality("reference"),
            "duration_seconds": 5,
            "seed": 9,
        })
        self.assertEqual(spec.preset["actual_steps"], 9)
        self.assertEqual(spec.preset["forecast_steps"], 11)
        options = public_options("reference")
        self.assertEqual(options["current_engine_options"]["short_label"], "Ref2VA")

    def test_reference_layout_places_conditions_before_target_streams(self) -> None:
        layout = build_ref2va_layout(
            text_length=12,
            latent_frames=7,
            latent_height=30,
            latent_width=54,
            audio_frames=207,
            reference_shapes=((1, 30, 54), (1, 20, 30)),
        )
        self.assertEqual(
            [segment.kind for segment in layout.segments],
            ["text", "condition", "condition", "audio", "video"],
        )
        condition_rows = (30 // 2) * (54 // 2) + (20 // 2) * (30 // 2)
        self.assertEqual(int((~layout.video_update_mask).sum()), condition_rows)
        self.assertTrue(torch.all(layout.video_update_mask[-7 * 15 * 27 :]))
        self.assertEqual(layout.segment("video", last=True).stop, layout.sequence_length)

    def test_reference_images_advance_one_rotary_time_unit_each(self) -> None:
        layout = build_ref2va_layout(
            text_length=12,
            latent_frames=7,
            latent_height=30,
            latent_width=54,
            audio_frames=207,
            reference_shapes=((1, 30, 54), (1, 20, 30)),
        )
        first, second = [
            segment for segment in layout.segments if segment.kind == "condition"
        ]
        self.assertTrue(torch.all(layout.position_ids[first.start:first.stop, 0] == 12))
        self.assertTrue(torch.all(layout.position_ids[second.start:second.stop, 0] == 13))

    def test_reference_video_uses_temporal_rotary_positions(self) -> None:
        layout = build_ref2va_layout(
            text_length=12, latent_frames=7, latent_height=30, latent_width=54,
            audio_frames=207, reference_shapes=((7, 30, 54),),
            reference_kinds=("video",),
        )
        segment = layout.segment("condition")
        times = torch.unique(layout.position_ids[segment.start:segment.stop, 0])
        self.assertGreater(len(times), 1)
        self.assertEqual(float(times[0]), 12.0)
        audio = layout.segment("audio")
        self.assertAlmostEqual(float(layout.position_ids[audio.start, 0]), 12.0 + 110.0 / 3.0)

    def test_reference_kind_shape_alignment_is_enforced(self) -> None:
        with self.assertRaisesRegex(ValueError, "reference_kinds"):
            build_ref2va_layout(
                text_length=12, latent_frames=7, latent_height=30, latent_width=54,
                audio_frames=207, reference_shapes=((1, 30, 54),),
                reference_kinds=("image", "video"),
            )

    def test_reference_audio_decode_resamples_mono_to_stereo_32khz(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "voice.wav"
            with av.open(str(path), "w") as container:
                stream = container.add_stream("pcm_s16le", rate=48000)
                stream.layout = "mono"
                samples = np.linspace(-0.25, 0.25, 4800, dtype=np.float32)[None]
                frame = av.AudioFrame.from_ndarray(samples, format="flt", layout="mono")
                frame.sample_rate = 48000
                for packet in stream.encode(frame):
                    container.mux(packet)
                for packet in stream.encode():
                    container.mux(packet)
            prepared = prepare_reference_audios(
                SimpleNamespace(reference_audios=(path,))
            )
            self.assertEqual(prepared[0].sample_rate, 32000)
            self.assertEqual(tuple(prepared[0].waveform.shape[:2]), (1, 2))
            self.assertAlmostEqual(prepared[0].source_duration_seconds, 0.1, places=2)

    def test_reference_audio_layout_precedes_target_audio_and_is_immutable(self) -> None:
        layout = build_ref2va_layout(
            text_length=12, latent_frames=7, latent_height=30, latent_width=54,
            audio_frames=207, reference_shapes=((1, 30, 54),),
            reference_audio_frames=(90, 41),
        )
        self.assertEqual(
            [segment.kind for segment in layout.segments],
            ["text", "condition", "ref_audio", "ref_audio", "audio", "video"],
        )
        self.assertEqual(int((~layout.audio_update_mask).sum()), 2 * (90 + 41))
        self.assertTrue(torch.all(layout.audio_update_mask[-414:]))

    def test_hybrid_memory_and_last_frame_keep_distinct_temporal_roles(self) -> None:
        layout = build_hybrid_condition_layout(
            text_length=4,
            latent_frames=3,
            latent_height=4,
            latent_width=6,
            audio_frames=5,
            reference_shapes=((1, 4, 6), (1, 2, 4)),
            reference_kinds=("image", "image"),
            reference_audio_frames=(2,),
            keyframe_indices=(-1,),
            output_frame_count=39,
        )
        self.assertEqual(
            [segment.kind for segment in layout.segments],
            [
                "text",
                "condition",
                "condition",
                "ref_audio",
                "condition",
                "audio",
                "video",
            ],
        )
        conditions = [
            segment for segment in layout.segments if segment.kind == "condition"
        ]
        memory_conditions, keyframe_condition = conditions[:2], conditions[-1]
        target_video = layout.segment("video")
        target_last_time = layout.position_ids[target_video.stop - 1, 0]

        # Reference-style memory owns its own leading rotary chart, while the
        # FL2VA endpoint retains the exact last target-video time coordinate.
        self.assertTrue(torch.all(
            layout.position_ids[
                memory_conditions[0].start:memory_conditions[0].stop, 0
            ] == 4
        ))
        self.assertTrue(torch.all(
            layout.position_ids[
                memory_conditions[1].start:memory_conditions[1].stop, 0
            ] == 5
        ))
        self.assertTrue(torch.all(
            layout.position_ids[
                keyframe_condition.start:keyframe_condition.stop, 0
            ] == target_last_time
        ))
        self.assertEqual(
            int((~layout.video_update_mask).sum()),
            (4 // 2) * (6 // 2) + (2 // 2) * (4 // 2) + (4 // 2) * (6 // 2),
        )
        self.assertEqual(int((~layout.audio_update_mask).sum()), 4)
        self.assertTrue(torch.all(layout.video_update_mask[-3 * 2 * 3:]))
        self.assertTrue(torch.all(layout.audio_update_mask[-10:]))

    def test_hybrid_audio_memory_can_coexist_with_first_frame_without_visual_memory(self) -> None:
        layout = build_hybrid_condition_layout(
            text_length=4,
            latent_frames=3,
            latent_height=4,
            latent_width=6,
            audio_frames=5,
            reference_shapes=(),
            reference_audio_frames=(1, 1),
            keyframe_indices=(0,),
            output_frame_count=39,
        )
        condition = layout.segment("condition")
        target_video = layout.segment("video")
        self.assertTrue(torch.all(
            layout.position_ids[condition.start:condition.stop, 0]
            == layout.position_ids[target_video.start, 0]
        ))
        self.assertEqual(int((~layout.audio_update_mask).sum()), 4)

    def test_reference_audio_timestep_is_clean_and_target_audio_follows_own_clock(self) -> None:
        from h3serve.native_engine.model.dit import FullH3DiT

        layout = build_ref2va_layout(
            text_length=4, latent_frames=2, latent_height=4, latent_width=4,
            audio_frames=3, reference_shapes=(), reference_audio_frames=(2,),
        )
        values, segments, rows = FullH3DiT._timestep_plan(
            torch.tensor([0.5]), layout,
            sigma_shift_video=12.0, sigma_shift_audio=3.0,
            visual_condition_timestep=0.999, audio_condition_timestep=1.0,
            text_token_tags=None, device=torch.device("cpu"),
        )
        ref = next(item for item in layout.segments if item.kind == "ref_audio")
        target = layout.segment("audio", last=True)
        ref_row = next(row for start, stop, row in segments if start == ref.start and stop == ref.stop)
        target_row = next(row for start, stop, row in segments if start == target.start and stop == target.stop)
        self.assertAlmostEqual(float(values[ref_row // 3]), 1.0)
        self.assertEqual(target_row // 3, rows["audio"])

    def test_reference_condition_row_cache_keeps_v00_replayable(self) -> None:
        from h3serve.native_engine.model.dit import FullH3DiT

        layout = build_ref2va_layout(
            text_length=4, latent_frames=2, latent_height=4, latent_width=4,
            audio_frames=3, reference_shapes=((1, 4, 4),),
            reference_audio_frames=(2,),
        )
        calls = 0

        def build() -> torch.Tensor:
            nonlocal calls
            calls += 1
            return torch.tensor([float(calls)])

        first = FullH3DiT._request_local_tensor(
            layout, "device_video_condition_rows", enabled=True,
            device=torch.device("cpu"), builder=build,
        )
        second = FullH3DiT._request_local_tensor(
            layout, "device_video_condition_rows", enabled=True,
            device=torch.device("cpu"), builder=build,
        )
        self.assertIs(first, second)
        self.assertEqual(calls, 1)

        FullH3DiT._request_local_tensor(
            layout, "device_video_condition_rows", enabled=False,
            device=torch.device("cpu"), builder=build,
        )
        FullH3DiT._request_local_tensor(
            layout, "device_video_condition_rows", enabled=False,
            device=torch.device("cpu"), builder=build,
        )
        self.assertEqual(calls, 3)

    def test_split_condition_projection_matches_combined_linear_math(self) -> None:
        generator = torch.Generator().manual_seed(82416)
        projection = torch.nn.Linear(12, 19)
        target = torch.randn(12, 12, generator=generator)
        condition = torch.randn(12, 12, generator=generator)
        mask = torch.tensor([True, False, True, False] * 6)
        all_rows = torch.empty(24, 12)
        all_rows[mask] = target
        all_rows[~mask] = condition
        combined = projection(all_rows)
        split = torch.empty_like(combined)
        split[mask] = projection(target)
        split[~mask] = projection(condition)
        # Separating M is row-wise equivalent.  Production FP32 GEMMs may
        # choose a different tile shape, so the candidate is not called
        # bit-exact even though this CPU reference happens to match exactly.
        torch.testing.assert_close(split, combined, rtol=1e-6, atol=1e-6)

    def test_reference_latent_cache_key_is_content_order_and_frame_sensitive(self) -> None:
        session = NativeT2AVHotSession.__new__(NativeT2AVHotSession)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first.bin"
            second = root / "second.bin"
            first.write_bytes(b"first")
            second.write_bytes(b"second")

            def request(images=(), videos=(), frames=124):
                return HotSessionRequest(
                    prompt="test", seed=1, width=864, height=480,
                    frames=frames, fps=24, steps=20,
                    output_path=root / "out.mp4",
                    reference_images=tuple(images),
                    reference_videos=tuple(videos),
                )

            base = session._reference_latent_key(request((first, second)))
            self.assertNotEqual(
                base,
                session._reference_latent_key(request((second, first))),
            )
            self.assertEqual(
                base,
                session._reference_latent_key(
                    request((first, second), frames=192)
                ),
            )
            video_base = session._reference_latent_key(
                request(videos=(first,), frames=124)
            )
            self.assertNotEqual(
                video_base,
                session._reference_latent_key(
                    request(videos=(first,), frames=192)
                ),
            )
            first.write_bytes(b"changed")
            self.assertNotEqual(
                base,
                session._reference_latent_key(request((first, second))),
            )


if __name__ == "__main__":
    unittest.main()
