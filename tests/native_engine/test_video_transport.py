from __future__ import annotations

import unittest


class VideoTransportTests(unittest.TestCase):
    def test_long_uint8_transport_selects_bounded_geometry_only_chunk(self):
        from h3serve.native_engine.adapters.real_vae import (
            select_uint8_postprocess_frame_chunk,
        )

        self.assertIsNone(
            select_uint8_postprocess_frame_chunk((1, 3, 124, 736, 1280))
        )
        self.assertEqual(
            select_uint8_postprocess_frame_chunk((1, 3, 362, 1088, 1920)),
            10,
        )

    def test_720p15_streams_when_runtime_cuda_headroom_is_too_small(self):
        from h3serve.native_engine.adapters.real_vae import (
            select_uint8_postprocess_frame_chunk,
        )

        shape = (1, 3, 362, 736, 1280)
        # Geometry alone stays on the established single-tensor path.
        self.assertIsNone(select_uint8_postprocess_frame_chunk(shape))
        # A hard 16GB launcher can have less than the 3.81-GiB output tensor
        # available after the Video-VAE and decoded clip are resident.
        chunk = select_uint8_postprocess_frame_chunk(
            shape, workspace_budget_bytes=2 * 1024**3
        )
        self.assertIsNotNone(chunk)
        self.assertGreater(chunk, 0)
        self.assertLess(chunk, shape[2])

    def test_streaming_uint8_transform_is_byte_exact(self):
        import torch

        from h3serve.native_engine.adapters import real_vae

        generator = torch.Generator("cpu").manual_seed(1303)
        decoded = torch.randn(
            (1, 3, 9, 16, 32), generator=generator, dtype=torch.float16
        )
        original_threshold = real_vae.VIDEO_UINT8_STREAMING_MIN_FP32_BYTES
        original_working_set = real_vae.VIDEO_UINT8_STREAMING_WORKING_SET_BYTES
        try:
            real_vae.VIDEO_UINT8_STREAMING_MIN_FP32_BYTES = 0
            real_vae.VIDEO_UINT8_STREAMING_WORKING_SET_BYTES = 3 * 16 * 32 * 4 * 2
            streamed = real_vae.postprocess_native_video(
                decoded, output_dtype="uint8"
            )
        finally:
            real_vae.VIDEO_UINT8_STREAMING_MIN_FP32_BYTES = original_threshold
            real_vae.VIDEO_UINT8_STREAMING_WORKING_SET_BYTES = original_working_set

        pixel_mean = decoded.new_tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1, 1)
        pixel_std = decoded.new_tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1, 1)
        reference = torch.round(
            (decoded.float() * pixel_std + pixel_mean).clamp_(0, 1).mul_(255.0)
        ).to(torch.uint8)
        self.assertTrue(torch.equal(streamed, reference))

    def test_temporal_host_sink_is_byte_exact_across_piece_boundaries(self):
        import torch

        from h3serve.native_engine.adapters.real_vae import (
            _TemporalUint8HostSink,
            postprocess_native_video,
        )

        generator = torch.Generator("cpu").manual_seed(82417)
        decoded = torch.randn(
            (1, 3, 17, 16, 32), generator=generator, dtype=torch.float32
        )
        output = torch.empty(decoded.shape, dtype=torch.uint8, device="cpu")
        sink = _TemporalUint8HostSink(output=output, frame_chunk=3)
        sink.begin(tuple(decoded.shape), decoded[:, :, :9])
        sink.write(decoded[:, :, :9], 0)
        sink.write(decoded[:, :, 9:], 9)
        streamed = sink.finish()
        reference = postprocess_native_video(decoded.clone(), output_dtype="uint8")

        self.assertIs(streamed, output)
        self.assertTrue(torch.equal(streamed, reference))
        self.assertFalse(streamed.is_inference())

    def test_decode_native_video_can_return_exact_temporal_host_output(self):
        import torch

        from h3serve.native_engine.adapters.real_vae import (
            decode_native_video,
            postprocess_native_video,
        )

        class FakeTemporalModel:
            def __init__(self, decoded):
                self.decoded = decoded

            def decode_base(self, _latent, frame_num=None):
                self.assert_frame_num = frame_num
                sink = self._h3_temporal_output_sink
                sink.begin(tuple(self.decoded.shape), self.decoded[:, :, :7])
                sink.write(self.decoded[:, :, :7], 0)
                sink.write(self.decoded[:, :, 7:], 7)
                return sink.finish()

        generator = torch.Generator("cpu").manual_seed(82418)
        decoded = torch.randn(
            (1, 3, 22, 16, 32), generator=generator, dtype=torch.float32
        )
        model = FakeTemporalModel(decoded)
        normalized = torch.zeros((1, 24, 7, 1, 2), dtype=torch.float32)
        mean = torch.zeros(24, dtype=torch.float32)
        std = torch.ones(24, dtype=torch.float32)
        actual = decode_native_video(
            model,
            normalized,
            mean,
            std,
            22,
            output_dtype="uint8",
            temporal_host_chunk_frames=4,
        )
        expected = postprocess_native_video(decoded.clone(), output_dtype="uint8")

        self.assertEqual(model.assert_frame_num, 22)
        self.assertTrue(torch.equal(actual, expected))
        self.assertFalse(hasattr(model, "_h3_temporal_output_sink"))

    def test_hot_session_applies_temporal_host_route_phase_locally(self):
        import torch

        from h3serve.native_engine.hot_session import NativeT2AVHotSession
        from h3serve.native_engine.planner import ExecutionPlan
        from h3serve.native_engine.runtime import OffloadMode

        class Model:
            pass

        model = Model()
        observed = []
        session = object.__new__(NativeT2AVHotSession)

        def decoder(active_model, _latents, frame_count):
            observed.append(
                (
                    getattr(active_model, "_h3_temporal_host_chunk_frames", None),
                    frame_count,
                )
            )
            return torch.zeros((1, 3, frame_count, 1, 1), dtype=torch.uint8)

        session.decode_video = decoder
        plan = ExecutionPlan(
            offload_mode=OffloadMode.BLOCK,
            mlp_chunk_tokens=4096,
            vae_temporal_tile=6,
        )
        result = session._decode_video_for_plan(
            model,
            torch.zeros((1, 24, 7, 1, 1)),
            22,
            plan,
        )

        self.assertEqual(observed, [(6, 22)])
        self.assertEqual(tuple(result.shape), (1, 3, 22, 1, 1))
        self.assertFalse(hasattr(model, "_h3_temporal_host_chunk_frames"))

    def test_streaming_accepts_real_vae_inference_tensor(self):
        import torch

        from h3serve.native_engine.adapters import real_vae

        with torch.inference_mode():
            decoded = torch.zeros((1, 3, 3, 8, 8), dtype=torch.float16)
        original_threshold = real_vae.VIDEO_UINT8_STREAMING_MIN_FP32_BYTES
        try:
            real_vae.VIDEO_UINT8_STREAMING_MIN_FP32_BYTES = 0
            output = real_vae.postprocess_native_video(
                decoded, output_dtype="uint8"
            )
        finally:
            real_vae.VIDEO_UINT8_STREAMING_MIN_FP32_BYTES = original_threshold
        self.assertEqual(output.device.type, "cpu")
        self.assertEqual(output.dtype, torch.uint8)
        self.assertFalse(output.is_inference())

    def test_uint8_transport_matches_existing_mux_rounding(self):
        import numpy as np
        import torch

        from h3serve.native_engine.adapters.real_vae import (
            postprocess_native_video,
        )
        from h3serve.native_engine.adapters.sampling_mux.mux import _video_uint8

        generator = torch.Generator("cpu").manual_seed(4090)
        decoded = torch.randn(
            (1, 3, 7, 16, 32), generator=generator, dtype=torch.float16
        )
        # Include exact codec-boundary and tie-adjacent values in the audit.
        decoded.flatten()[:8] = torch.tensor(
            [-8.0, 8.0, -2.117, 2.117, -0.5, 0.0, 0.5, 1.0],
            dtype=torch.float16,
        )

        reference = postprocess_native_video(decoded, output_dtype="float32")
        expected = _video_uint8(reference)
        transported = postprocess_native_video(decoded, output_dtype="uint8")
        actual = _video_uint8(transported)

        self.assertEqual(transported.dtype, torch.uint8)
        self.assertEqual(transported.numel(), reference.numel())
        self.assertEqual(
            transported.numel() * transported.element_size() * 4,
            reference.numel() * reference.element_size(),
        )
        np.testing.assert_array_equal(actual, expected)

    def test_rejects_unknown_transport_dtype(self):
        import torch

        from h3serve.native_engine.adapters.real_vae import (
            postprocess_native_video,
        )

        with self.assertRaisesRegex(ValueError, "float32 or uint8"):
            postprocess_native_video(torch.zeros(1, 3, 1, 1, 1), output_dtype="fp8")


if __name__ == "__main__":
    unittest.main()
