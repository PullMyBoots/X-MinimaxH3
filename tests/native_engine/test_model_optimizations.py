from __future__ import annotations

import unittest
import sys
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import torch

from h3serve.native_engine.adapters.conditioning_vae.qwen_quantized import (
    PackedQwen3VLT2AVConditioner,
)
from h3serve.native_engine.adapters.sampling_mux import simple_sigma_schedule
from h3serve.native_engine.forecast import (
    DirectionalForecastController,
    forecast_history_storage_mode,
    target_layout,
)
from h3serve.native_engine.model.assembly import MappingTensorSource, _plain_linear
from h3serve.native_engine.model.dit import FullH3DiT
from h3serve.native_engine.model.layers import (
    FusedQKVAttention,
    SwiGLUMLP,
    _gated_residual,
    _scale_shift,
    apply_qknorm_rope,
    rope_rotation_table,
)
from h3serve.native_engine.model.fused_mlp import (
    _chunked_module_gated_mlp,
    _segment_row_map,
)
from h3serve.native_engine.model.kernels import (
    ActionScheduledAttentionBackend,
    RequestRoutedSpargeAttentionBackend,
    StepScheduledAttentionBackend,
    attention_layer,
    attention_video_layout,
    current_long_sequence_compact_kv,
    current_long_sequence_shared_qkv_quantization,
    current_long_sequence_fused_query_projection,
    current_long_sequence_fused_qknorm_hnd_layout,
    current_long_sequence_direct_nhd_output,
    current_long_sequence_direct_nhd_kv,
    current_long_sequence_query_chunk_tokens,
    dense_qk_quantization,
    long_sequence_query_chunking,
    attention_protected_prefix,
    attention_sparsity,
    attention_step,
    _write_compact_value_fp8,
)
from h3serve.native_engine.model.lora import (
    PrunedCurveAdaLN,
    LowRankUpdate,
    SlicedLowRankUpdate,
    RuntimeLoRALinear,
    set_lora_enabled,
)
from h3serve.native_engine.model.quantization import (
    ComfyQuantSpec,
    ConvRotInt8Linear,
    ConvRotW4A8Linear,
    W4A8QuantSpec,
    comfy_kitchen_int8_kernel,
)
from h3serve.native_engine.model.packed import build_fl2va_layout, build_ref2va_layout


class SegmentedModulationTests(unittest.TestCase):
    def test_compact_value_writer_bounds_noncontiguous_split_copy(self) -> None:
        fused_kv = torch.randn(64, 2, 256)
        value_view = fused_kv[..., 128:]
        self.assertFalse(value_view.is_contiguous())
        captured: dict[str, torch.Tensor] = {}

        def fake_write(target, source, value_absmax, **kwargs):
            del target, value_absmax, kwargs
            captured["source"] = source

        with patch(
            "h3serve.native_engine.model.compact_fp8.write_sage_fp8_slab",
            side_effect=fake_write,
        ):
            _write_compact_value_fp8(
                torch.empty(1),
                value_view,
                value_view.float().abs().amax(dim=0),
                start=0,
                layout="NHD",
            )

        self.assertTrue(captured["source"].is_contiguous())
        torch.testing.assert_close(captured["source"], value_view)

    def test_forecast_history_storage_routes_long_geometry_only(self) -> None:
        self.assertEqual(
            forecast_history_storage_mode(
                rows=100_141,
                channels=5376,
                element_size=2,
                device_type="cuda",
            ),
            "pinned_whole",
        )
        self.assertEqual(
            forecast_history_storage_mode(
                rows=219_659,
                channels=5376,
                element_size=2,
                device_type="cuda",
            ),
            "pageable_chunked",
        )
        self.assertEqual(
            forecast_history_storage_mode(
                rows=219_659,
                channels=5376,
                element_size=2,
                device_type="cpu",
            ),
            "pageable_whole",
        )

    def test_long_sequence_chunk_scope_is_explicit_and_restored(self) -> None:
        self.assertIsNone(current_long_sequence_query_chunk_tokens())
        self.assertFalse(current_long_sequence_compact_kv())
        self.assertFalse(current_long_sequence_shared_qkv_quantization())
        self.assertFalse(current_long_sequence_fused_query_projection())
        self.assertFalse(current_long_sequence_fused_qknorm_hnd_layout())
        self.assertFalse(current_long_sequence_direct_nhd_output())
        self.assertFalse(current_long_sequence_direct_nhd_kv())
        with long_sequence_query_chunking(256):
            self.assertEqual(current_long_sequence_query_chunk_tokens(), 256)
        self.assertIsNone(current_long_sequence_query_chunk_tokens())
        with long_sequence_query_chunking(
            256, split_qkv_outputs=True, compact_kv=True
        ):
            self.assertTrue(current_long_sequence_compact_kv())
        self.assertFalse(current_long_sequence_compact_kv())
        with long_sequence_query_chunking(
            256,
            split_qkv_outputs=True,
            shared_qkv_quantization=True,
        ):
            self.assertTrue(current_long_sequence_shared_qkv_quantization())
        self.assertFalse(current_long_sequence_shared_qkv_quantization())
        with self.assertRaises(ValueError):
            with long_sequence_query_chunking(
                256, shared_qkv_quantization=True
            ):
                pass
        with long_sequence_query_chunking(
            256, fused_query_projection=True
        ):
            self.assertTrue(current_long_sequence_fused_query_projection())
        self.assertFalse(current_long_sequence_fused_query_projection())
        with long_sequence_query_chunking(
            256,
            split_qkv_outputs=True,
            single_qknorm_rope=True,
            fused_query_projection=True,
            fused_qknorm_hnd_layout=True,
        ):
            self.assertTrue(current_long_sequence_fused_qknorm_hnd_layout())
        self.assertFalse(current_long_sequence_fused_qknorm_hnd_layout())
        with long_sequence_query_chunking(256, direct_nhd_output=True):
            self.assertTrue(current_long_sequence_direct_nhd_output())
        self.assertFalse(current_long_sequence_direct_nhd_output())
        with long_sequence_query_chunking(
            256, split_qkv_outputs=True, direct_nhd_kv=True
        ):
            self.assertTrue(current_long_sequence_direct_nhd_kv())
        self.assertFalse(current_long_sequence_direct_nhd_kv())
        with self.assertRaises(ValueError):
            with long_sequence_query_chunking(
                256,
                split_qkv_outputs=True,
                compact_kv=True,
                direct_nhd_kv=True,
            ):
                pass
        with self.assertRaises(ValueError):
            with long_sequence_query_chunking(256, compact_kv=True):
                pass
        for invalid in (0, 127, 129):
            with self.assertRaises(ValueError):
                with long_sequence_query_chunking(invalid):
                    pass

    def test_quantized_output_slice_matches_full_projection(self) -> None:
        generator = torch.Generator().manual_seed(20260825)
        qweight = torch.randint(
            -127, 128, (12, 8), dtype=torch.int8, generator=generator
        )
        scale = torch.rand(12, 1, generator=generator)
        value = torch.randn(17, 8, generator=generator)
        linear = ConvRotInt8Linear(
            qweight,
            scale,
            spec=ComfyQuantSpec(
                format="int8_tensorwise",
                convrot=True,
                convrot_groupsize=4,
            ),
            output_dtype=torch.float32,
        )
        full = linear(value)
        torch.testing.assert_close(
            linear.forward_output_slice(value, 0, 4),
            full[:, :4],
            rtol=2e-6,
            atol=4e-6,
        )
        torch.testing.assert_close(
            linear.forward_output_slice(value, 4, 12),
            full[:, 4:],
            rtol=2e-6,
            atol=4e-6,
        )
        with self.assertRaisesRegex(ValueError, "outside"):
            linear.forward_output_slice(value, 4, 13)

    @unittest.skipUnless(
        torch.cuda.is_available()
        and torch.cuda.get_device_capability() == (8, 9),
        "prepared QKV quantization requires the release SM89 CUDA backend",
    )
    def test_prepared_int8_output_slices_are_byte_exact(self) -> None:
        torch.manual_seed(20260828)
        value = torch.randn(
            (257, 256), device="cuda", dtype=torch.bfloat16
        )
        qweight = torch.randint(
            -127,
            128,
            (384, 256),
            device="cuda",
            dtype=torch.int8,
        )
        scale = torch.rand((384, 1), device="cuda", dtype=torch.float32)
        bias = torch.randn((384,), device="cuda", dtype=torch.bfloat16)
        base = ConvRotInt8Linear(
            qweight,
            scale,
            bias,
            spec=ComfyQuantSpec(
                format="int8_tensorwise",
                convrot=True,
                convrot_groupsize=256,
            ),
            kernel=comfy_kitchen_int8_kernel,
            output_dtype=torch.bfloat16,
        )
        prepared = base.prepare_output_slices(value)
        for row_start, row_stop, output_start, output_stop in (
            (0, 257, 0, 128),
            (0, 129, 128, 384),
            (129, 257, 64, 320),
        ):
            expected = base.forward_output_slice(
                value[row_start:row_stop], output_start, output_stop
            )
            actual = base.forward_prepared_output_slice(
                value,
                prepared,
                row_start,
                row_stop,
                output_start,
                output_stop,
            )
            self.assertTrue(torch.equal(expected, actual))

        update = LowRankUpdate(
            down=torch.randn((8, 256), device="cuda", dtype=torch.bfloat16),
            up=torch.randn((384, 8), device="cuda", dtype=torch.bfloat16),
        )
        lora = RuntimeLoRALinear(base, update)
        lora_prepared = lora.prepare_output_slices(value)
        expected = lora.forward_output_slice(value, 128, 384)
        actual = lora.forward_prepared_output_slice(
            value, lora_prepared, 0, 257, 128, 384
        )
        self.assertTrue(torch.equal(expected, actual))

    def test_w4a8_output_slice_and_lora_match_full_projection(self) -> None:
        generator = torch.Generator().manual_seed(20260827)
        qdata = torch.randint(
            -128, 128, (12, 8), dtype=torch.int8, generator=generator
        )
        s_rel = torch.rand(12, 4, generator=generator) * 0.5
        s_channel = torch.rand(12, generator=generator) * 0.05
        codebook = torch.linspace(-7.5, 7.5, 16)
        value = torch.randn(17, 16, generator=generator)
        base = ConvRotW4A8Linear(
            qdata,
            s_rel,
            s_channel,
            codebook=codebook,
            spec=W4A8QuantSpec(
                format="asym_w4a8_int8",
                group_size=4,
                convrot=True,
                convrot_groupsize=4,
            ),
            output_dtype=torch.float32,
        )
        update = LowRankUpdate(
            down=torch.randn(3, 16, generator=generator),
            up=torch.randn(12, 3, generator=generator),
            strength=0.75,
            alpha=3.0,
        )
        module = RuntimeLoRALinear(base, update)

        base_full = base(value)
        torch.testing.assert_close(
            torch.cat(
                [base.forward_output_slice(value, 0, 4),
                 base.forward_output_slice(value, 4, 12)],
                dim=-1,
            ),
            base_full,
            rtol=2e-6,
            atol=4e-6,
        )
        lora_full = module(value)
        torch.testing.assert_close(
            torch.cat(
                [module.forward_output_slice(value, 0, 4),
                 module.forward_output_slice(value, 4, 12)],
                dim=-1,
            ),
            lora_full,
            rtol=2e-6,
            atol=4e-6,
        )
        module.set_lora_enabled(False)
        torch.testing.assert_close(
            module.forward_output_slice(value, 4, 12),
            base.forward_output_slice(value, 4, 12),
            rtol=0,
            atol=0,
        )

    def test_w4a8_chunked_mlp_matches_unbounded_module_math(self) -> None:
        generator = torch.Generator().manual_seed(82727)

        def linear(out_features: int, in_features: int):
            return ConvRotW4A8Linear(
                torch.randint(
                    -128, 128,
                    (out_features, in_features // 2),
                    dtype=torch.int8,
                    generator=generator,
                ),
                torch.rand(
                    out_features, in_features // 4, generator=generator
                ) * 0.5,
                torch.rand(out_features, generator=generator) * 0.05,
                codebook=torch.linspace(-7.5, 7.5, 16),
                spec=W4A8QuantSpec(
                    format="asym_w4a8_int8",
                    group_size=4,
                    convrot=True,
                    convrot_groupsize=4,
                ),
                output_dtype=torch.float32,
            )

        mlp = SwiGLUMLP(linear(32, 16), linear(16, 16))
        hidden = torch.randn(23, 16, generator=generator)
        residual = torch.randn(23, 16, generator=generator)
        gate = torch.randn(3, 16, generator=generator)
        segments = ((0, 5, 0), (5, 11, 1), (11, 23, 2))
        expected = _gated_residual(
            residual.clone(), mlp(hidden), gate, segments
        )
        actual = residual.clone()
        _chunked_module_gated_mlp(
            actual,
            hidden,
            mlp,
            gate,
            segments,
            chunk_tokens=7,
        )
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)

    def test_streamed_attention_applies_the_same_segment_gates(self) -> None:
        prefix_query_sizes: list[int] = []
        projection_ranges: list[tuple[int, int]] = []

        class SliceLinear(torch.nn.Linear):
            def forward_output_slice(self, value, start, stop):
                projection_ranges.append((start, stop))
                bias = None if self.bias is None else self.bias[start:stop]
                return torch.nn.functional.linear(
                    value, self.weight[start:stop], bias
                )

        class IdentityPreparedBackend:
            def __call__(self, query, key, value):
                del key, value
                return query

            def resolve_long_sequence_backend(self, query_tokens):
                return self if query_tokens >= 128 else None

            @staticmethod
            def prepare_long_sequence_values(value_hnd):
                _, heads, tokens, head_dim = value_hnd.shape
                return value_hnd, torch.ones(1), heads, tokens, head_dim

            @staticmethod
            def prepare_long_sequence_keys(key_hnd, value_fp8, value_scale):
                del value_fp8, value_scale
                return key_hnd

            @staticmethod
            def long_sequence_prefix_queries(query, prepared):
                del prepared
                prefix_query_sizes.append(int(query.shape[0]))
                return query

            @staticmethod
            def long_sequence_video_queries(
                query,
                prepared,
                *,
                protected_tokens,
                query_token_indices,
            ):
                del prepared, protected_tokens, query_token_indices
                return query

        torch.manual_seed(7)
        backend = IdentityPreparedBackend()
        attention = FusedQKVAttention(
            SliceLinear(8, 24, bias=False),
            torch.nn.Linear(8, 8, bias=False),
            num_heads=2,
            head_dim=4,
            backend=backend,
            dtype=torch.float32,
        )
        attention.q_norm.weight.data.fill_(1.0)
        attention.k_norm.weight.data.fill_(1.0)
        # Model a variable Ref2VA-style
        # [text | image refs | audio refs | target audio | clean continuation
        # video | noisy target video].  The extended protected prefix is
        # intentionally neither calibrated nor aligned to the Query chunk.
        hidden = torch.randn(901, 8)
        residual = torch.randn_like(hidden)
        gate = torch.randn(5, 8)
        segments = (
            (0, 97, 0),
            (97, 257, 1),
            (257, 389, 2),
            (389, 517, 3),
            (517, 901, 4),
        )
        expected = _gated_residual(
            residual.clone(), attention(hidden, None), gate, segments
        )
        actual = residual.clone()
        with (
            attention_protected_prefix(517),
            attention_video_layout(3, 128),
            long_sequence_query_chunking(128, split_qkv_outputs=True),
        ):
            self.assertTrue(
                attention.stream_gated_residual_(
                    actual,
                    hidden,
                    None,
                    gate,
                    segments,
                    query_chunk_tokens=128,
                )
            )
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
        self.assertEqual(prefix_query_sizes, [128, 128, 128, 128, 5])
        self.assertEqual(projection_ranges[:8], [(8, 24)] * 8)
        self.assertEqual(projection_ranges[8:], [(0, 8)] * 8)

    def test_streamed_attention_rejects_stale_fixed_prefix_geometry(self) -> None:
        class IdentityPreparedBackend:
            def resolve_long_sequence_backend(self, query_tokens):
                del query_tokens
                return self

            @staticmethod
            def prepare_long_sequence_values(value_hnd):
                _, heads, tokens, head_dim = value_hnd.shape
                return value_hnd, torch.ones(1), heads, tokens, head_dim

            @staticmethod
            def prepare_long_sequence_keys(key_hnd, value_fp8, value_scale):
                del value_fp8, value_scale
                return key_hnd

            @staticmethod
            def long_sequence_prefix_queries(query, prepared):
                del prepared
                return query

            long_sequence_video_queries = long_sequence_prefix_queries

        attention = FusedQKVAttention(
            torch.nn.Linear(8, 24, bias=False),
            torch.nn.Linear(8, 8, bias=False),
            num_heads=2,
            head_dim=4,
            backend=IdentityPreparedBackend(),
            dtype=torch.float32,
        )
        hidden = torch.randn(384, 8)
        with attention_protected_prefix(129), self.assertRaisesRegex(
            ValueError, "layout prefix"
        ):
            attention.stream_gated_residual_(
                hidden.clone(),
                hidden,
                None,
                torch.ones(2, 8),
                ((0, 128, 0), (128, 384, 1)),
                query_chunk_tokens=128,
            )

    def test_request_routed_attention_switches_without_module_mutation(self) -> None:
        calls: list[tuple[str, float | None]] = []

        def dense(query, key, value):
            del key, value
            calls.append(("dense", None))
            return query

        def sparse(query, key, value, **kwargs):
            del key, value
            calls.append(("sparse", kwargs["topk"]))
            return query.unsqueeze(0)

        module = SimpleNamespace(
            spas_sage2_attn_meansim_topk_cuda=sparse
        )
        backend = RequestRoutedSpargeAttentionBackend(
            minimum_sparse_tokens=128
        )
        short = torch.zeros(64, 2, 4)
        long = torch.zeros(256, 2, 4)
        with (
            patch(
                "h3serve.native_engine.model.kernels.sage_attention_sm89",
                side_effect=dense,
            ),
            patch.dict(sys.modules, {"spas_sage_attn": module}),
        ):
            backend(long, long, long)
            with attention_sparsity(0.5):
                backend(short, short, short)
                backend(long, long, long)
            backend(long, long, long)
        self.assertEqual(
            calls,
            [("dense", None), ("dense", None), ("sparse", 0.5), ("dense", None)],
        )

    def test_request_routed_dense_fallback_has_bounded_long_sequence_backend(self) -> None:
        backend = RequestRoutedSpargeAttentionBackend(minimum_sparse_tokens=128)
        self.assertIsNone(backend.resolve_long_sequence_backend(256))
        with dense_qk_quantization("per_warp"):
            self.assertIsNotNone(backend.resolve_long_sequence_backend(256))
            with attention_sparsity(0.5):
                self.assertIsNone(backend.resolve_long_sequence_backend(256))

    def test_step_scheduled_attention_keeps_short_and_anchor_steps_dense(self) -> None:
        calls: list[str] = []

        def dense(query, key, value):
            del key, value
            calls.append("dense")
            return query

        def sparse(query, key, value):
            del key, value
            calls.append("sparse")
            return query

        backend = StepScheduledAttentionBackend(
            dense,
            sparse,
            dense_step_indices=(0, 5),
            minimum_sparse_tokens=128,
        )
        short = torch.zeros(64, 2, 4)
        long = torch.zeros(256, 2, 4)
        backend(short, short, short)
        backend(long, long, long)
        with attention_step(0, 6):
            backend(long, long, long)
        with attention_step(2, 6):
            backend(long, long, long)
        with attention_step(5, 6):
            backend(long, long, long)
        self.assertEqual(calls, ["dense", "dense", "dense", "sparse", "dense"])

    def test_unscheduled_sparse_policy_preserves_fixed_sparse_behavior(self) -> None:
        calls: list[str] = []

        def record(name):
            def call(query, key, value):
                del key, value
                calls.append(name)
                return query

            return call

        backend = StepScheduledAttentionBackend(record("dense"), record("sparse"))
        value = torch.zeros(256, 2, 4)
        backend(value, value, value)
        self.assertEqual(calls, ["sparse"])

    def test_action_schedule_routes_physical_step_layer_and_fails_closed(self) -> None:
        calls: list[str] = []

        def record(name):
            def call(query, key, value):
                del key, value
                calls.append(name)
                return query

            return call

        backend = ActionScheduledAttentionBackend(
            {"dense": record("dense"), "draft": record("draft")},
            {(2, 7): "draft"},
            expected_sequence_tokens=256,
        )
        value = torch.zeros(256, 2, 4)
        backend(value, value, value)
        with attention_step(2, 6), attention_layer(7):
            backend(value, value, value)
        with attention_step(2, 6), attention_layer(8):
            backend(value, value, value)
        self.assertEqual(calls, ["dense", "draft", "dense"])
        self.assertEqual(backend.telemetry()["exact_fallback_calls"], 2)

    def test_action_schedule_adapts_to_uncalibrated_long_shape(self) -> None:
        calls: list[str] = []

        def record(name):
            def call(query, key, value):
                del key, value
                calls.append(name)
                return query

            return call

        backend = ActionScheduledAttentionBackend(
            {"dense": record("dense"), "draft": record("draft")},
            {(0, 0): "draft"},
            expected_sequence_tokens=256,
        )
        value = torch.zeros(512, 2, 4)
        with attention_step(0, 1), attention_layer(0):
            backend(value, value, value)
        self.assertEqual(calls, ["draft"])
        telemetry = backend.telemetry()
        self.assertEqual(telemetry["shape_adapted_calls"], 1)
        self.assertEqual(telemetry["observed_sequence_tokens"], {512: 1})
        self.assertEqual(telemetry["observed_to_calibration_ratio_min"], 2.0)

    def test_action_schedule_accepts_full_ref2va_media_layout(self) -> None:
        """Nine images, three videos and three audios remain legal inputs."""

        calls: list[str] = []

        def record(name):
            def call(query, key, value):
                del key, value
                calls.append(name)
                return query

            return call

        reference_shapes = ((1, 30, 54),) * 9 + ((5, 30, 54),) * 3
        layout = build_ref2va_layout(
            text_length=2048,
            latent_frames=7,
            latent_height=30,
            latent_width=54,
            audio_frames=90,
            reference_shapes=reference_shapes,
            reference_kinds=("image",) * 9 + ("video",) * 3,
            reference_audio_frames=(90, 80, 70),
        )
        total = layout.segments[-1].stop
        protected = layout.segment("video", last=True).start
        value = torch.zeros(total, 1, 1)
        backend = ActionScheduledAttentionBackend(
            {"dense": record("dense"), "draft": record("draft")},
            {(3, 17): "draft"},
            expected_sequence_tokens=15_702,
        )
        with (
            attention_step(3, 20),
            attention_layer(17),
            attention_protected_prefix(protected),
        ):
            backend(value, value, value)
        self.assertEqual(calls, ["draft"])
        self.assertEqual(backend.telemetry()["shape_adapted_calls"], 1)

    def test_forecast_target_layout_matches_final_audio_video_segments(self) -> None:
        layout = build_fl2va_layout(
            text_length=5,
            latent_frames=2,
            latent_height=4,
            latent_width=4,
            audio_frames=3,
        )
        info = target_layout(layout)
        self.assertEqual(info.audio_rows, 6)
        self.assertEqual(info.video_rows, 8)
        self.assertEqual(info.start, layout.segment("audio", last=True).start)
        self.assertEqual(info.stop, layout.segment("video", last=True).stop)
        self.assertIsNone(layout.device_video_update_mask)
        self.assertIsNone(layout.device_rope_table)

    def test_depth3_forecast_runs_only_anchor_blocks_after_two_refreshes(self) -> None:
        layout = build_fl2va_layout(
            text_length=5,
            latent_frames=2,
            latent_height=4,
            latent_width=4,
            audio_frames=3,
        )
        calls: list[int] = []

        class AddBlock(torch.nn.Module):
            def __init__(self, index: int) -> None:
                super().__init__()
                self.index = index

            def forward(self, value: torch.Tensor, **_: object) -> torch.Tensor:
                calls.append(self.index)
                return value + float(self.index + 1)

        stack = SimpleNamespace(
            blocks=torch.nn.ModuleList(AddBlock(i) for i in range(5))
        )
        controller = DirectionalForecastController(actual_steps=(0, 1))
        kwargs = {
            "layout": layout,
            "unique_timesteps": torch.ones(1),
            "modulation_segments": (),
            "frequencies": torch.ones(1),
            "curve_rows": None,
        }
        values = [
            torch.zeros(layout.sequence_length, 8),
            torch.ones(layout.sequence_length, 8),
            torch.full((layout.sequence_length, 8), 2.0),
        ]
        first = controller.run_block_stack(
            stack, values[0].clone(), step_index=0, requested_actual=True, **kwargs
        )
        second = controller.run_block_stack(
            stack, values[1].clone(), step_index=1, requested_actual=True, **kwargs
        )
        predicted = controller.run_block_stack(
            stack, values[2].clone(), step_index=2, requested_actual=False, **kwargs
        )
        target = target_layout(layout)
        expected = values[2][target.start : target.stop] + sum(range(1, 6))
        torch.testing.assert_close(
            predicted[target.start : target.stop], expected, rtol=0, atol=0
        )
        self.assertEqual(calls, [0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2])
        self.assertEqual(controller.export()["actual_steps"], 2)
        self.assertEqual(controller.export()["forecast_steps"], 1)
        self.assertEqual(first.shape, second.shape)

    def test_long_forecast_history_streaming_is_elementwise_equivalent(self) -> None:
        layout = build_fl2va_layout(
            text_length=5,
            latent_frames=5,
            latent_height=80,
            latent_width=80,
            audio_frames=3,
        )

        class AddBlock(torch.nn.Module):
            def __init__(self, index: int) -> None:
                super().__init__()
                self.index = index

            def forward(self, value: torch.Tensor, **_: object) -> torch.Tensor:
                return value + float(self.index + 1)

        kwargs = {
            "layout": layout,
            "unique_timesteps": torch.ones(1),
            "modulation_segments": (),
            "frequencies": torch.ones(1),
            "curve_rows": None,
        }
        values = [
            torch.zeros(layout.sequence_length, 8),
            torch.ones(layout.sequence_length, 8),
            torch.full((layout.sequence_length, 8), 2.0),
        ]

        def run(*, streamed: bool):
            stack = SimpleNamespace(
                blocks=torch.nn.ModuleList(AddBlock(i) for i in range(5))
            )
            controller = DirectionalForecastController(actual_steps=(0, 1))
            with patch(
                "h3serve.native_engine.forecast._stream_forecast_history",
                return_value=streamed,
            ):
                controller.run_block_stack(
                    stack,
                    values[0].clone(),
                    step_index=0,
                    requested_actual=True,
                    **kwargs,
                )
                controller.run_block_stack(
                    stack,
                    values[1].clone(),
                    step_index=1,
                    requested_actual=True,
                    **kwargs,
                )
                result = controller.run_block_stack(
                    stack,
                    values[2].clone(),
                    step_index=2,
                    requested_actual=False,
                    **kwargs,
                )
            return result, controller.export()

        whole, whole_report = run(streamed=False)
        chunked, chunked_report = run(streamed=True)
        torch.testing.assert_close(chunked, whole, rtol=0, atol=0)
        self.assertEqual(
            [row["history_storage"] for row in chunked_report["records"][:2]],
            ["pageable_chunked", "pageable_chunked"],
        )
        self.assertEqual(
            chunked_report["records"][2]["history_transfer"],
            "pageable_chunked",
        )
        self.assertEqual(
            whole_report["records"][2]["history_transfer"],
            "pinned_whole",
        )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_long_forecast_history_streaming_crosses_cuda_in_chunks(self) -> None:
        layout = build_fl2va_layout(
            text_length=5,
            latent_frames=2,
            latent_height=8,
            latent_width=8,
            audio_frames=3,
        )

        class AddBlock(torch.nn.Module):
            def __init__(self, index: int) -> None:
                super().__init__()
                self.index = index

            def forward(self, value: torch.Tensor, **_: object) -> torch.Tensor:
                return value + float(self.index + 1)

        stack = SimpleNamespace(
            blocks=torch.nn.ModuleList(AddBlock(i) for i in range(5))
        )
        controller = DirectionalForecastController(actual_steps=(0, 1))
        kwargs = {
            "layout": layout,
            "unique_timesteps": torch.ones(1, device="cuda"),
            "modulation_segments": (),
            "frequencies": torch.ones(1, device="cuda"),
            "curve_rows": None,
        }
        values = [
            torch.zeros(layout.sequence_length, 8, device="cuda"),
            torch.ones(layout.sequence_length, 8, device="cuda"),
            torch.full((layout.sequence_length, 8), 2.0, device="cuda"),
        ]
        with patch(
            "h3serve.native_engine.forecast._stream_forecast_history",
            return_value=True,
        ):
            controller.run_block_stack(
                stack,
                values[0].clone(),
                step_index=0,
                requested_actual=True,
                **kwargs,
            )
            controller.run_block_stack(
                stack,
                values[1].clone(),
                step_index=1,
                requested_actual=True,
                **kwargs,
            )
            predicted = controller.run_block_stack(
                stack,
                values[2].clone(),
                step_index=2,
                requested_actual=False,
                **kwargs,
            )
        target = target_layout(layout)
        expected = values[2][target.start : target.stop] + sum(range(1, 6))
        torch.testing.assert_close(
            predicted[target.start : target.stop],
            expected,
            rtol=0,
            atol=0,
        )
        self.assertEqual(
            controller.export()["records"][2]["history_transfer"],
            "pageable_chunked",
        )

    def test_forecast_controller_rejects_unsorted_actual_steps(self) -> None:
        with self.assertRaisesRegex(ValueError, "sorted and unique"):
            DirectionalForecastController(actual_steps=(1, 0, 1))

    def test_fp32_island_preserves_comfy_bf16_storage_boundary(self) -> None:
        weight = torch.tensor(
            [[1.0037, -0.4979], [0.2521, 0.7491]], dtype=torch.float32
        )
        bias = torch.tensor([0.1003, -0.2007], dtype=torch.float32)
        source = MappingTensorSource(
            {"island.weight": weight, "island.bias": bias}
        )
        module = _plain_linear(
            source,
            "island",
            device="cpu",
            dtype=torch.float32,
            source_round_dtype=torch.bfloat16,
        )
        self.assertEqual(module.weight.dtype, torch.float32)
        torch.testing.assert_close(
            module.weight,
            weight.bfloat16().float(),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            module.bias,
            bias.bfloat16().float(),
            rtol=0,
            atol=0,
        )
        self.assertFalse(torch.equal(module.weight, weight))

    def test_simple_schedule_matches_comfy_discrete_h3_table(self) -> None:
        self.assertEqual(
            simple_sigma_schedule(6, 12.0),
            (
                1.0,
                0.9836838841438293,
                0.9600575566291809,
                0.9230769276618958,
                0.8575096726417542,
                0.7063800096511841,
                0.0,
            ),
        )

    def test_qwen3vl_rope_base_matches_h3_checkpoint_contract(self) -> None:
        self.assertEqual(PackedQwen3VLT2AVConditioner.rope_theta, 5_000_000.0)

    def test_qwen_pinned_cache_uses_the_same_tensor_access_contract(self) -> None:
        cached = {"weight": torch.arange(4, dtype=torch.float32)}
        self.assertIs(
            PackedQwen3VLT2AVConditioner._get_tensor(cached, "weight"),
            cached["weight"],
        )
        self.assertTrue(
            PackedQwen3VLT2AVConditioner._contains(cached, "weight")
        )

    def test_qwen_embedding_preserves_checkpoint_bf16_store_before_fp32_compute(self) -> None:
        conditioner = PackedQwen3VLT2AVConditioner.__new__(
            PackedQwen3VLT2AVConditioner
        )
        conditioner.device = torch.device("cpu")
        conditioner.dtype = torch.float32
        tensors = {
            "model.embed_tokens.weight": torch.tensor(
                [[127, -83, 29], [17, 91, -119]], dtype=torch.int8
            ),
            "model.embed_tokens.weight_scale": torch.tensor(
                [[0.00371], [0.00293]], dtype=torch.float32
            ),
        }

        class Checkpoint:
            @staticmethod
            def get_tensor(key):
                return tensors[key]

        ids = torch.tensor([1, 0], dtype=torch.long)
        actual = conditioner._embedding(Checkpoint(), ids)
        expected = (
            tensors["model.embed_tokens.weight"][ids].float()
            * tensors["model.embed_tokens.weight_scale"][ids]
        ).bfloat16().float()
        raw_fp32 = (
            tensors["model.embed_tokens.weight"][ids].float()
            * tensors["model.embed_tokens.weight_scale"][ids]
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertFalse(torch.equal(actual, raw_fp32))

    def test_qwen_runtime_does_not_mutate_global_kernel_backend(self) -> None:
        calls: list[str] = []

        @contextmanager
        def use_backend(name: str):
            calls.append(f"enter:{name}")
            try:
                yield
            finally:
                calls.append(f"exit:{name}")

        def dequantize_nvfp4(qweight, tensor_scale, block_scale, dtype):
            calls.append("dequantize")
            return qweight.to(dtype)

        fake_kitchen = SimpleNamespace(
            use_backend=use_backend,
            dequantize_nvfp4=dequantize_nvfp4,
        )
        conditioner = PackedQwen3VLT2AVConditioner.__new__(
            PackedQwen3VLT2AVConditioner
        )
        conditioner._kitchen = None
        conditioner.device = torch.device("cpu")
        conditioner.dtype = torch.float32

        tensors = {
            "linear.weight": torch.eye(3),
            "linear.weight_scale": torch.ones(1),
            "linear.weight_scale_2": torch.ones(1),
        }

        class Checkpoint:
            @staticmethod
            def keys():
                return tensors.keys()

            @staticmethod
            def get_tensor(key):
                return tensors[key]

        with patch.dict("sys.modules", {"comfy_kitchen": fake_kitchen}):
            self.assertIs(conditioner._load_runtime(), fake_kitchen)
            result = conditioner._linear(
                Checkpoint(), "linear", torch.tensor([[1.0, 2.0, 3.0]])
            )
        torch.testing.assert_close(result, torch.tensor([[1.0, 2.0, 3.0]]))
        self.assertEqual(calls, ["enter:triton", "dequantize", "exit:triton"])

    def test_packed_rope_table_matches_angle_reference_on_cpu(self) -> None:
        generator = torch.Generator().manual_seed(9)
        query = torch.randn(7, 3, 128, generator=generator)
        key = torch.randn(7, 3, 128, generator=generator)
        q_weight = torch.randn(128, generator=generator)
        k_weight = torch.randn(128, generator=generator)
        angles = torch.randn(7, 96, generator=generator)
        table = rope_rotation_table(angles, torch.float32)
        expected = apply_qknorm_rope(
            query.clone(),
            key.clone(),
            q_weight=q_weight,
            k_weight=k_weight,
            frequencies=angles,
            eps=1e-5,
        )
        actual = apply_qknorm_rope(
            query.clone(),
            key.clone(),
            q_weight=q_weight,
            k_weight=k_weight,
            frequencies=table,
            eps=1e-5,
        )
        torch.testing.assert_close(actual[0], expected[0], rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(actual[1], expected[1], rtol=1e-6, atol=1e-6)

    def test_segment_row_map_matches_segments(self) -> None:
        segments = ((0, 3, 1), (3, 7, 5), (7, 11, 0))
        actual = _segment_row_map(11, segments, torch.device("cpu"))
        expected = torch.tensor([1, 1, 1, 5, 5, 5, 5, 0, 0, 0, 0], dtype=torch.int32)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_runtime_lora_buffers_follow_module_residency_and_dtype(self) -> None:
        generator = torch.Generator().manual_seed(41)
        base = torch.nn.Linear(5, 4, bias=False)
        update = LowRankUpdate(
            down=torch.randn(2, 5, generator=generator),
            up=torch.randn(4, 2, generator=generator),
            strength=0.75,
            alpha=2.0,
        )
        module = RuntimeLoRALinear(base, update).to(dtype=torch.float64)
        self.assertEqual(module.update.down.dtype, torch.float64)
        self.assertEqual(module.update.up.dtype, torch.float64)
        registered = dict(module.named_buffers())
        self.assertIn("update.down", registered)
        self.assertIn("update.up", registered)

        value = torch.randn(3, 5, generator=generator, dtype=torch.float64)
        expected = base(value)
        expected = expected + 0.75 * (value @ module.update.down.T) @ module.update.up.T
        torch.testing.assert_close(module(value), expected)

    def test_runtime_lora_fails_closed_instead_of_copying_weights_per_forward(self) -> None:
        base = torch.nn.Linear(5, 4, bias=False, dtype=torch.float64)
        update = LowRankUpdate(
            down=torch.randn(2, 5, dtype=torch.float32),
            up=torch.randn(4, 2, dtype=torch.float32),
        )
        module = RuntimeLoRALinear(base, update)
        value = torch.randn(3, 5, dtype=torch.float64)
        with self.assertRaisesRegex(RuntimeError, "dtype mismatch"):
            module(value)

    def test_runtime_lora_hot_switch_preserves_exact_base_path(self) -> None:
        generator = torch.Generator().manual_seed(91)
        base = torch.nn.Linear(5, 4, bias=False)
        update = LowRankUpdate(
            down=torch.randn(2, 5, generator=generator),
            up=torch.randn(4, 2, generator=generator),
        )
        module = RuntimeLoRALinear(base, update)
        value = torch.randn(3, 5, generator=generator)
        base_expected = base(value).clone()

        self.assertEqual(set_lora_enabled(module, False), 1)
        torch.testing.assert_close(module(value), base_expected, rtol=0, atol=0)
        self.assertEqual(set_lora_enabled(module, True), 1)
        self.assertFalse(torch.equal(module(value), base_expected))
        # The switch is execution policy; it must not duplicate or modify the
        # resident checkpoint tensors.
        self.assertEqual(
            set(module.state_dict()), {"base.weight", "update.down", "update.up"}
        )

    def test_diffusers_split_qkv_lora_matches_three_independent_updates(self) -> None:
        generator = torch.Generator().manual_seed(20260829)

        class SliceLinear(torch.nn.Linear):
            def forward_output_slice(self, value, start, stop):
                return torch.nn.functional.linear(
                    value, self.weight[start:stop], None
                )

        base = SliceLinear(5, 12, bias=False)
        slices = []
        for start, stop in ((0, 4), (4, 8), (8, 12)):
            update = LowRankUpdate(
                down=torch.randn(2, 5, generator=generator),
                up=torch.randn(4, 2, generator=generator),
                alpha=2.0,
            )
            slices.append((start, stop, update))
        module = RuntimeLoRALinear(base, SlicedLowRankUpdate(tuple(slices)))
        value = torch.randn(7, 5, generator=generator)
        expected = base(value)
        for start, stop, update in slices:
            expected[..., start:stop] += (value @ update.down.T) @ update.up.T
        torch.testing.assert_close(module(value), expected)
        torch.testing.assert_close(
            torch.cat(
                [module.forward_output_slice(value, start, stop)
                 for start, stop, _ in slices],
                dim=-1,
            ),
            expected,
        )
        self.assertEqual(set_lora_enabled(module, False), 3)
        torch.testing.assert_close(module(value), base(value), rtol=0, atol=0)

    def test_adaln_lora_keeps_bf16_residency_and_allows_fp32_island(self) -> None:
        base = torch.nn.Linear(8, 12, dtype=torch.float32)
        update = LowRankUpdate(
            down=torch.randn(2, 6, dtype=torch.bfloat16),
            up=torch.randn(12, 2, dtype=torch.bfloat16),
        )
        module = PrunedCurveAdaLN(base, None, lora_update=update)
        self.assertEqual(module.lora_update.down.dtype, torch.bfloat16)
        self.assertEqual(module.lora_update.up.dtype, torch.bfloat16)
        result = module.lora_update.apply(
            torch.randn(3, 6, dtype=torch.float32),
            torch.randn(3, 12, dtype=torch.float32),
        )
        self.assertEqual(result.dtype, torch.float32)

    def test_segmented_scale_shift_matches_reference_operation_order(self) -> None:
        generator = torch.Generator().manual_seed(7)
        value = torch.randn(11, 8, generator=generator)
        shift = torch.randn(6, 8, generator=generator)
        scale = torch.randn(6, 8, generator=generator)
        segments = ((0, 3, 1), (3, 7, 5), (7, 11, 0))

        expected = value.clone()
        for start, stop, row in segments:
            expected[start:stop].mul_(1.0 + scale[row]).add_(shift[row])

        actual = _scale_shift(value.clone(), shift, scale, segments)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_segmented_gate_matches_reference_operation_order(self) -> None:
        generator = torch.Generator().manual_seed(8)
        residual = torch.randn(11, 8, generator=generator)
        update = torch.randn(11, 8, generator=generator)
        gate = torch.randn(6, 8, generator=generator)
        segments = ((0, 3, 1), (3, 7, 5), (7, 11, 0))

        expected = residual.clone()
        for start, stop, row in segments:
            expected[start:stop].addcmul_(update[start:stop], gate[row])

        actual = _gated_residual(residual.clone(), update, gate, segments)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_timestep_plan_splits_text_tag_runs_and_covers_layout(self) -> None:
        layout = build_fl2va_layout(
            text_length=5,
            latent_frames=2,
            latent_height=4,
            latent_width=4,
            audio_frames=3,
        )
        _, segments, _ = FullH3DiT._timestep_plan(
            torch.tensor([0.75]),
            layout,
            sigma_shift_video=12.0,
            sigma_shift_audio=3.0,
            visual_condition_timestep=0.999,
            text_token_tags=torch.tensor([1, 1, 0, 0, 1]),
            device=torch.device("cpu"),
        )

        self.assertEqual([(a, b) for a, b, _ in segments[:3]], [(0, 2), (2, 4), (4, 5)])
        self.assertEqual(segments[0][0], 0)
        self.assertEqual(segments[-1][1], layout.sequence_length)
        self.assertTrue(all(left[1] == right[0] for left, right in zip(segments, segments[1:])))

    def test_timestep_plan_pins_masked_av_prefix_to_clean_clock(self) -> None:
        layout = build_fl2va_layout(
            text_length=5,
            latent_frames=4,
            latent_height=4,
            latent_width=4,
            audio_frames=10,
        )
        values, segments, rows = FullH3DiT._timestep_plan(
            torch.tensor([0.75]),
            layout,
            sigma_shift_video=12.0,
            sigma_shift_audio=3.0,
            visual_condition_timestep=0.999,
            audio_condition_timestep=1.0,
            text_token_tags=None,
            device=torch.device("cpu"),
            masked_video_prefix_rows=8,
            masked_audio_prefix_rows=6,
        )

        video = layout.segment("video", last=True)
        audio = layout.segment("audio", last=True)
        video_parts = [
            item for item in segments
            if video.start <= item[0] and item[1] <= video.stop
        ]
        audio_parts = [
            item for item in segments
            if audio.start <= item[0] and item[1] <= audio.stop
        ]
        self.assertEqual([(a, b) for a, b, _ in video_parts], [
            (video.start, video.start + 8),
            (video.start + 8, video.stop),
        ])
        self.assertEqual([(a, b) for a, b, _ in audio_parts], [
            (audio.start, audio.start + 6),
            (audio.start + 6, audio.stop),
        ])
        self.assertAlmostEqual(float(values[video_parts[0][2] // 3]), 0.999)
        self.assertAlmostEqual(float(values[audio_parts[0][2] // 3]), 1.0)
        self.assertEqual(video_parts[1][2] // 3, rows["video"])
        self.assertEqual(audio_parts[1][2] // 3, rows["audio"])


if __name__ == "__main__":
    unittest.main()
