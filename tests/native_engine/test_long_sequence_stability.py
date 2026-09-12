from __future__ import annotations

import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from h3serve.native_engine.hot_session import HotSessionRequest, NativeT2AVHotSession
from h3serve.native_engine.planner import (
    ExecutionPlan,
    LONG_SEQUENCE_EXTENDED_PREFIX_MIN_PACKED_TOKENS,
    LONG_SEQUENCE_STABLE_QK_MIN_TOKENS,
    LONG_SEQUENCE_VALIDATED_MAX_PACKED_TOKENS,
    NoFeasibleProfile,
    select_long_sequence_chunks,
    select_stable_dense_qk_quantization,
)
from h3serve.native_engine.runtime import OffloadMode


class LongSequenceStabilityTests(unittest.TestCase):
    def test_unmeasured_2k_request_reaches_compact_budget_fallback(self) -> None:
        class NoRoutePlanner:
            @staticmethod
            def select(*_args, **_kwargs):
                raise NoFeasibleProfile("no calibrated performance profile")

        session = NativeT2AVHotSession.__new__(NativeT2AVHotSession)
        session.engine = "original"
        session.planner = NoRoutePlanner()
        session.runtime_config = SimpleNamespace(
            device="cuda:0", max_device_bytes=23 * 1024**3
        )
        request = HotSessionRequest(
            prompt="2k low-vram routing is content-independent",
            seed=1,
            width=2560,
            height=1440,
            frames=362,
            fps=24,
            steps=20,
            output_path=Path("unused.mp4"),
            actual_step_indices=(0, 1, 2, 3, 4, 8, 12, 15, 18, 19),
            memory_mode="auto",
            release_byte_exact_optimizations=True,
        )
        with (
            patch("torch.cuda.mem_get_info", return_value=(23 * 1024**3, 24 * 1024**3)),
            patch("torch.cuda.memory_reserved", return_value=0),
            patch("torch.cuda.memory_allocated", return_value=0),
        ):
            plan, profile = session._resolve_execution_plan(request, text_tokens=512)

        assert plan is not None
        self.assertEqual(profile["source"], "memory_execution_fallback")
        self.assertEqual(profile["predicted_seconds"], None)
        self.assertEqual(
            profile["predicted_peak_gib"],
            profile["memory_execution"]["estimated_selected_peak_gib"],
        )
        self.assertEqual(
            profile["memory_execution"]["selected_scheme"], "compact_streaming"
        )
        self.assertTrue(plan.long_sequence_compact_kv)
        self.assertEqual(plan.long_sequence_query_chunk_tokens, 8192)
        self.assertFalse(plan.long_sequence_fused_query_projection)
        self.assertFalse(plan.long_sequence_fused_qknorm_hnd_layout)
        self.assertFalse(plan.long_sequence_fused_prefix_k_quant)
        self.assertFalse(plan.long_sequence_direct_nhd_output)
        self.assertFalse(plan.long_sequence_direct_hnd_fp8_value)
        self.assertFalse(plan.long_sequence_shared_qkv_quantization)
        self.assertFalse(
            profile["memory_execution"]["release_fused_query_projection"]
        )
        self.assertFalse(
            profile["memory_execution"][
                "release_fused_qknorm_hnd_layout"
            ]
        )
        self.assertFalse(
            profile["memory_execution"]["release_direct_hnd_fp8_value"]
        )
        self.assertFalse(
            profile["memory_execution"]["release_fused_prefix_k_quant"]
        )
        self.assertFalse(
            profile["memory_execution"]["release_shared_qkv_quantization"]
        )

    def test_legacy_mode_names_resolve_to_same_unified_graph(self) -> None:
        class NoRoutePlanner:
            @staticmethod
            def select(*_args, **_kwargs):
                raise NoFeasibleProfile("no calibrated performance profile")

        session = NativeT2AVHotSession.__new__(NativeT2AVHotSession)
        session.engine = "original"
        session.planner = NoRoutePlanner()
        session.runtime_config = SimpleNamespace(
            device="cuda:0", max_device_bytes=23 * 1024**3
        )
        request = HotSessionRequest(
            prompt="small explicit performance route",
            seed=2,
            width=864,
            height=480,
            frames=124,
            fps=24,
            steps=20,
            output_path=Path("unused.mp4"),
            actual_step_indices=(0, 1, 2, 3, 4, 8, 12, 15, 18, 19),
            memory_mode="performance",
        )
        with (
            patch("torch.cuda.mem_get_info", return_value=(23 * 1024**3, 24 * 1024**3)),
            patch("torch.cuda.memory_reserved", return_value=0),
            patch("torch.cuda.memory_allocated", return_value=0),
        ):
            plan, profile = session._resolve_execution_plan(request, text_tokens=512)

        assert plan is not None
        self.assertEqual(profile["source"], "memory_execution_fallback")
        self.assertEqual(
            profile["memory_execution"]["selected_scheme"], "exact_streaming"
        )
        self.assertTrue(profile["memory_execution"]["fits_budget"])
        self.assertFalse(plan.long_sequence_compact_kv)

        low_request = replace(request, memory_mode="low_vram")
        with (
            patch("torch.cuda.mem_get_info", return_value=(23 * 1024**3, 24 * 1024**3)),
            patch("torch.cuda.memory_reserved", return_value=0),
            patch("torch.cuda.memory_allocated", return_value=0),
        ):
            low_plan, low_profile = session._resolve_execution_plan(
                low_request, text_tokens=512
            )

        assert low_plan is not None
        self.assertEqual(
            low_profile["memory_execution"]["selected_scheme"],
            "exact_streaming",
        )
        self.assertTrue(low_profile["memory_execution"]["bit_exact"])
        self.assertEqual(
            low_plan.long_sequence_query_chunk_tokens,
            plan.long_sequence_query_chunk_tokens,
        )

    def test_short_video_geometry_does_not_enable_streaming_route(self) -> None:
        for video_tokens, packed_tokens in (
            (98_440, 100_163),  # 1280x736x362 (720p15)
            (75_480, 76_859),  # 1920x1088x124 (1080p5)
        ):
            with self.subTest(video_tokens=video_tokens):
                decision = select_long_sequence_chunks(
                    video_tokens=video_tokens,
                    packed_tokens=packed_tokens,
                )
                self.assertIsNone(decision.query_chunk_tokens)
                self.assertFalse(decision.split_qkv_outputs)
                self.assertFalse(decision.single_qknorm_rope)
                self.assertFalse(decision.parallel_sparse_lut)
                self.assertIsNone(decision.reason)

    def test_measured_1080p15_geometry_selects_fast_streaming_bucket(self) -> None:
        decision = select_long_sequence_chunks(
            video_tokens=218_280,
            packed_tokens=219_659,
        )
        self.assertEqual(decision.query_chunk_tokens, 32_768)
        self.assertEqual(decision.projection_chunk_tokens, 8192)
        self.assertTrue(decision.split_qkv_outputs)
        self.assertTrue(decision.single_qknorm_rope)
        self.assertTrue(decision.parallel_sparse_lut)
        self.assertEqual(decision.reason, "request_geometry_1080p15")

    def test_extended_prefix_selects_lower_peak_streaming_bucket(self) -> None:
        decision = select_long_sequence_chunks(
            video_tokens=218_280,
            packed_tokens=LONG_SEQUENCE_EXTENDED_PREFIX_MIN_PACKED_TOKENS,
        )
        self.assertEqual(decision.query_chunk_tokens, 16_384)
        self.assertTrue(decision.split_qkv_outputs)
        self.assertTrue(decision.single_qknorm_rope)
        self.assertTrue(decision.parallel_sparse_lut)
        self.assertEqual(decision.reason, "request_geometry_extended_prefix")

    def test_unvalidated_packed_geometry_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "validated long-sequence"):
            select_long_sequence_chunks(
                video_tokens=218_280,
                packed_tokens=LONG_SEQUENCE_VALIDATED_MAX_PACKED_TOKENS + 1,
            )

    def test_short_per_thread_request_is_unchanged(self) -> None:
        effective, reason = select_stable_dense_qk_quantization(
            "per_thread", packed_tokens=34_780
        )
        self.assertEqual(effective, "per_thread")
        self.assertIsNone(reason)

    def test_measured_long_boundary_uses_stable_per_warp_path(self) -> None:
        effective, reason = select_stable_dense_qk_quantization(
            "per_thread",
            packed_tokens=LONG_SEQUENCE_STABLE_QK_MIN_TOKENS,
        )
        self.assertEqual(effective, "per_warp")
        self.assertEqual(reason, "long_sequence_tail_stability")

    def test_explicit_per_warp_request_is_not_reported_as_override(self) -> None:
        effective, reason = select_stable_dense_qk_quantization(
            "per_warp", packed_tokens=100_163
        )
        self.assertEqual(effective, "per_warp")
        self.assertIsNone(reason)

    def test_invalid_token_count_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "packed_tokens"):
            select_stable_dense_qk_quantization("per_thread", packed_tokens=0)

    def test_hot_session_overrides_explicit_long_per_thread_plan(self) -> None:
        session = NativeT2AVHotSession.__new__(NativeT2AVHotSession)
        session.engine = "original"
        session.planner = None
        requested_plan = ExecutionPlan(
            offload_mode=OffloadMode.BLOCK,
            mlp_chunk_tokens=8192,
            dense_qk_quant_gran="per_thread",
            long_sequence_query_chunk_tokens=32768,
            long_sequence_projection_chunk_tokens=8192,
        )
        request = HotSessionRequest(
            prompt="test",
            seed=1,
            width=1280,
            height=736,
            frames=362,
            fps=24,
            steps=20,
            output_path=Path("unused.mp4"),
            actual_step_indices=(0, 1, 2, 3, 4, 8, 13, 17, 18, 19),
            execution_plan=requested_plan,
        )

        effective_plan, profile = session._resolve_execution_plan(
            request,
            text_tokens=517,
        )

        self.assertIsNotNone(effective_plan)
        assert effective_plan is not None
        self.assertEqual(effective_plan.dense_qk_quant_gran, "per_warp")
        self.assertEqual(profile["packed_tokens"], 100_163)
        self.assertEqual(profile["dense_qk_quant_gran_requested"], "per_thread")
        self.assertEqual(profile["long_sequence_query_chunk_tokens"], 32768)
        self.assertEqual(profile["long_sequence_projection_chunk_tokens"], 8192)
        self.assertEqual(
            profile["dense_qk_stability_override"],
            "long_sequence_tail_stability",
        )

    def test_hot_session_auto_selects_chunks_from_1080p15_geometry(self) -> None:
        session = NativeT2AVHotSession.__new__(NativeT2AVHotSession)
        session.engine = "original"
        session.planner = None
        requested_plan = ExecutionPlan(
            offload_mode=OffloadMode.BLOCK,
            mlp_chunk_tokens=8192,
            dense_qk_quant_gran="per_thread",
        )
        request = HotSessionRequest(
            prompt="content must not affect the geometry route",
            seed=1,
            width=1920,
            height=1088,
            frames=362,
            fps=24,
            steps=4,
            output_path=Path("unused.mp4"),
            execution_plan=requested_plan,
            release_byte_exact_optimizations=True,
        )

        effective_plan, profile = session._resolve_execution_plan(
            request,
            text_tokens=173,
        )

        self.assertIsNotNone(effective_plan)
        assert effective_plan is not None
        self.assertEqual(effective_plan.long_sequence_query_chunk_tokens, 32_768)
        self.assertEqual(
            effective_plan.long_sequence_projection_chunk_tokens,
            8192,
        )
        self.assertTrue(effective_plan.long_sequence_split_qkv_outputs)
        self.assertTrue(profile["long_sequence_split_qkv_outputs"])
        self.assertTrue(effective_plan.long_sequence_single_qknorm_rope)
        self.assertTrue(profile["long_sequence_single_qknorm_rope"])
        self.assertTrue(effective_plan.long_sequence_parallel_sparse_lut)
        self.assertTrue(profile["long_sequence_parallel_sparse_lut"])
        self.assertTrue(effective_plan.long_sequence_partial_sparse_topk)
        self.assertTrue(profile["long_sequence_partial_sparse_topk"])
        self.assertTrue(effective_plan.long_sequence_fused_query_projection)
        self.assertTrue(profile["long_sequence_fused_query_projection"])
        self.assertTrue(
            effective_plan.long_sequence_fused_qknorm_hnd_layout
        )
        self.assertTrue(profile["long_sequence_fused_qknorm_hnd_layout"])
        self.assertTrue(effective_plan.long_sequence_direct_nhd_output)
        self.assertTrue(profile["long_sequence_direct_nhd_output"])
        self.assertTrue(effective_plan.long_sequence_direct_hnd_fp8_value)
        self.assertTrue(profile["long_sequence_direct_hnd_fp8_value"])
        self.assertTrue(effective_plan.long_sequence_fused_prefix_k_quant)
        self.assertTrue(profile["long_sequence_fused_prefix_k_quant"])
        self.assertFalse(
            effective_plan.long_sequence_shared_qkv_quantization
        )
        self.assertFalse(
            profile["memory_execution"][
                "release_shared_qkv_quantization"
            ]
        )
        self.assertTrue(
            profile["memory_execution"]["release_fused_query_projection"]
        )
        self.assertIsNotNone(
            profile["memory_execution"]["release_fused_query_evidence"]
        )
        self.assertTrue(
            profile["memory_execution"][
                "release_fused_qknorm_hnd_layout"
            ]
        )
        self.assertIsNotNone(
            profile["memory_execution"][
                "release_fused_qknorm_hnd_evidence"
            ]
        )
        self.assertTrue(
            profile["memory_execution"]["release_direct_hnd_fp8_value"]
        )
        self.assertIsNotNone(
            profile["memory_execution"][
                "release_direct_hnd_fp8_value_evidence"
            ]
        )
        self.assertTrue(
            profile["memory_execution"]["release_fused_prefix_k_quant"]
        )
        self.assertIsNotNone(
            profile["memory_execution"][
                "release_fused_prefix_k_quant_evidence"
            ]
        )
        self.assertTrue(
            profile["memory_execution"]["release_partial_sparse_topk"]
        )
        self.assertIsNotNone(
            profile["memory_execution"][
                "release_partial_sparse_topk_evidence"
            ]
        )
        self.assertTrue(
            profile["memory_execution"]["release_direct_nhd_output"]
        )
        self.assertIsNotNone(
            profile["memory_execution"]["release_direct_nhd_output_evidence"]
        )
        self.assertEqual(profile["packed_tokens"], 219_659)
        self.assertEqual(
            profile["long_sequence_chunk_reason"],
            "request_geometry_1080p15",
        )

        session.engine = "reference"
        reference_plan, reference_profile = session._resolve_execution_plan(
            request,
            text_tokens=173,
        )
        assert reference_plan is not None
        self.assertTrue(reference_plan.long_sequence_fused_query_projection)
        self.assertTrue(
            reference_plan.long_sequence_fused_qknorm_hnd_layout
        )
        self.assertTrue(reference_plan.long_sequence_direct_hnd_fp8_value)
        self.assertTrue(reference_plan.long_sequence_fused_prefix_k_quant)
        self.assertTrue(
            reference_profile["long_sequence_direct_hnd_fp8_value"]
        )
        self.assertTrue(
            reference_profile["memory_execution"][
                "release_fused_qknorm_hnd_layout"
            ]
        )
        self.assertTrue(
            reference_profile["memory_execution"][
                "release_direct_hnd_fp8_value"
            ]
        )
        self.assertTrue(
            reference_profile["memory_execution"][
                "release_fused_prefix_k_quant"
            ]
        )
        self.assertFalse(reference_plan.long_sequence_partial_sparse_topk)
        self.assertFalse(
            reference_profile["memory_execution"][
                "release_partial_sparse_topk"
            ]
        )
        self.assertFalse(reference_plan.long_sequence_direct_nhd_output)
        self.assertFalse(
            reference_profile["memory_execution"]["release_direct_nhd_output"]
        )

    def test_16gb_release_reuses_qkv_quantization_only_with_budget_headroom(
        self,
    ) -> None:
        request = HotSessionRequest(
            prompt="packed-token admission cannot depend on prompt semantics",
            seed=1,
            width=1280,
            height=736,
            frames=362,
            fps=24,
            steps=5,
            output_path=Path("unused.mp4"),
            actual_step_indices=(0, 1, 2, 3, 4),
            execution_plan=ExecutionPlan(
                offload_mode=OffloadMode.BLOCK,
                mlp_chunk_tokens=8192,
            ),
            release_byte_exact_optimizations=True,
            reference_images=(Path("reference.png"),),
            reference_audios=(Path("reference.wav"),),
            prepared_reference_images=(
                SimpleNamespace(width=1280, height=720),
            ),
            prepared_reference_audios=(
                SimpleNamespace(
                    waveform=SimpleNamespace(shape=(1, 2, 32000))
                ),
            ),
        )
        for engine, text_tokens in (("original", 512), ("reference", 1200)):
            with self.subTest(engine=engine):
                session = NativeT2AVHotSession.__new__(
                    NativeT2AVHotSession
                )
                session.engine = engine
                session.planner = None
                session.runtime_config = SimpleNamespace(
                    device="cpu",
                    max_device_bytes=int(15.25 * 1024**3),
                    resource_profile="int8_16gb",
                    weight_tier="int8",
                )
                plan, profile = session._resolve_execution_plan(
                    request,
                    text_tokens=text_tokens,
                )
                assert plan is not None
                memory = profile["memory_execution"]
                self.assertEqual(
                    memory["selected_scheme"], "exact_streaming"
                )
                self.assertTrue(plan.long_sequence_split_qkv_outputs)
                self.assertTrue(
                    plan.long_sequence_shared_qkv_quantization
                )
                self.assertTrue(
                    profile["long_sequence_shared_qkv_quantization"]
                )
                self.assertTrue(
                    memory["release_shared_qkv_quantization"]
                )
                self.assertTrue(
                    memory["shared_qkv_quantization_fits_budget"]
                )
                self.assertEqual(
                    memory["shared_qkv_quantization_cache_bytes"],
                    profile["packed_tokens"] * (5_376 + 4),
                )
                self.assertLessEqual(
                    memory["shared_qkv_quantization_required_peak_bytes"],
                    memory["device_budget_bytes"] - 128 * 1024**2,
                )

        # Simulate external pressure on the same startup-fixed 16GB backend.
        # The base exact-streaming graph still fits, but its optional cache
        # plus release reserve does not, so planning must retain recomputation.
        session.runtime_config = SimpleNamespace(
            device="cpu",
            max_device_bytes=13 * 1024**3,
            resource_profile="int8_16gb",
            weight_tier="int8",
        )
        constrained_plan, constrained_profile = (
            session._resolve_execution_plan(request, text_tokens=512)
        )
        assert constrained_plan is not None
        constrained_memory = constrained_profile["memory_execution"]
        self.assertEqual(
            constrained_memory["selected_scheme"], "exact_streaming"
        )
        self.assertTrue(constrained_plan.long_sequence_fused_query_projection)
        self.assertFalse(
            constrained_plan.long_sequence_shared_qkv_quantization
        )
        self.assertFalse(
            constrained_memory["release_shared_qkv_quantization"]
        )
        self.assertFalse(
            constrained_memory["shared_qkv_quantization_fits_budget"]
        )

    def test_24gb_release_reuses_qkv_quantization_only_for_ref2va(self) -> None:
        request = HotSessionRequest(
            prompt="four-image resource admission is content independent",
            seed=1,
            width=1920,
            height=1088,
            frames=362,
            fps=24,
            steps=5,
            output_path=Path("unused.mp4"),
            actual_step_indices=(0, 1, 4),
            execution_plan=ExecutionPlan(
                offload_mode=OffloadMode.BLOCK,
                mlp_chunk_tokens=4096,
            ),
            release_byte_exact_optimizations=True,
            reference_images=tuple(
                Path(f"reference-{index}.png") for index in range(4)
            ),
            prepared_reference_images=tuple(
                SimpleNamespace(width=1280, height=720) for _ in range(4)
            ),
        )
        for engine, expected in (("reference", True), ("original", False)):
            with self.subTest(engine=engine):
                session = NativeT2AVHotSession.__new__(
                    NativeT2AVHotSession
                )
                session.engine = engine
                session.planner = None
                session.runtime_config = SimpleNamespace(
                    device="cpu",
                    max_device_bytes=int(23.25 * 1024**3),
                    resource_profile="int8_24gb",
                    weight_tier="int8",
                )
                plan, profile = session._resolve_execution_plan(
                    request,
                    text_tokens=1200,
                )
                assert plan is not None
                memory = profile["memory_execution"]
                self.assertEqual(
                    plan.long_sequence_shared_qkv_quantization,
                    expected,
                )
                self.assertEqual(
                    memory["release_shared_qkv_quantization"],
                    expected,
                )
                self.assertTrue(
                    memory["shared_qkv_quantization_fits_budget"]
                )
                if expected:
                    self.assertEqual(
                        memory[
                            "release_shared_qkv_quantization_evidence"
                        ],
                        "h3_shared_qkv_int8_activation_exact_ref2va_"
                        "24gb_1080p15_four_image_20260831",
                    )

    def test_v24_medium_anchor_enables_only_its_byte_exact_helpers(self) -> None:
        session = NativeT2AVHotSession.__new__(NativeT2AVHotSession)
        session.engine = "original"
        session.planner = None
        requested_plan = ExecutionPlan(
            offload_mode=OffloadMode.BLOCK,
            mlp_chunk_tokens=8192,
            dense_qk_quant_gran="per_thread",
        )
        request = HotSessionRequest(
            prompt="the words of the prompt cannot select this route",
            seed=1,
            width=1280,
            height=736,
            frames=243,
            fps=24,
            steps=20,
            output_path=Path("unused.mp4"),
            actual_step_indices=(0, 1, 2, 3, 4, 8, 12, 15, 18, 19),
            execution_plan=requested_plan,
            acceleration_plan_summary={
                "policy_id": "h3_pareto_v24_human_calibrated_deployment_v1",
                "execution_profile_hint": "v22_medium_byte_exact_helpers",
            },
        )

        effective_plan, profile = session._resolve_execution_plan(
            request,
            text_tokens=485,
        )

        assert effective_plan is not None
        self.assertEqual(effective_plan.long_sequence_query_chunk_tokens, 32_768)
        self.assertEqual(
            effective_plan.long_sequence_projection_chunk_tokens,
            8192,
        )
        self.assertTrue(effective_plan.long_sequence_split_qkv_outputs)
        self.assertTrue(effective_plan.long_sequence_single_qknorm_rope)
        self.assertTrue(effective_plan.long_sequence_parallel_sparse_lut)
        self.assertFalse(effective_plan.long_sequence_partial_sparse_topk)
        self.assertFalse(effective_plan.long_sequence_fused_prefix_k_quant)
        self.assertEqual(
            profile["long_sequence_chunk_reason"],
            "v24_v22_medium_byte_exact_execution",
        )

    def test_unhinted_requests_receive_unified_exact_streaming(self) -> None:
        session = NativeT2AVHotSession.__new__(NativeT2AVHotSession)
        session.engine = "original"
        session.planner = None
        requested_plan = ExecutionPlan(
            offload_mode=OffloadMode.BLOCK,
            mlp_chunk_tokens=8192,
            dense_qk_quant_gran="per_thread",
        )
        request = HotSessionRequest(
            prompt="same geometry, no reviewed endpoint hint",
            seed=1,
            width=1280,
            height=736,
            frames=243,
            fps=24,
            steps=20,
            output_path=Path("unused.mp4"),
            execution_plan=requested_plan,
        )

        effective_plan, profile = session._resolve_execution_plan(
            request,
            text_tokens=485,
        )

        assert effective_plan is not None
        self.assertEqual(effective_plan.long_sequence_query_chunk_tokens, 49_152)
        self.assertTrue(effective_plan.long_sequence_split_qkv_outputs)
        self.assertTrue(effective_plan.long_sequence_single_qknorm_rope)
        self.assertTrue(effective_plan.long_sequence_parallel_sparse_lut)
        self.assertFalse(effective_plan.long_sequence_fused_query_projection)
        self.assertFalse(
            effective_plan.long_sequence_fused_qknorm_hnd_layout
        )
        self.assertFalse(effective_plan.long_sequence_direct_nhd_output)
        self.assertFalse(effective_plan.long_sequence_fused_prefix_k_quant)
        self.assertFalse(
            profile["memory_execution"]["release_fused_query_projection"]
        )
        self.assertEqual(
            profile["long_sequence_chunk_reason"],
            "isolated_resource_streaming",
        )

    def test_hot_session_preserves_explicit_long_sequence_chunks(self) -> None:
        session = NativeT2AVHotSession.__new__(NativeT2AVHotSession)
        session.engine = "original"
        session.planner = None
        requested_plan = ExecutionPlan(
            offload_mode=OffloadMode.BLOCK,
            mlp_chunk_tokens=8192,
            dense_qk_quant_gran="per_warp",
            long_sequence_query_chunk_tokens=16_384,
            long_sequence_projection_chunk_tokens=4096,
            long_sequence_split_qkv_outputs=True,
            long_sequence_parallel_sparse_lut=True,
            long_sequence_partial_sparse_topk=True,
            long_sequence_fused_prefix_k_quant=True,
            long_sequence_direct_hnd_fp8_value=True,
        )
        request = HotSessionRequest(
            prompt="test",
            seed=1,
            width=1920,
            height=1088,
            frames=362,
            fps=24,
            steps=4,
            output_path=Path("unused.mp4"),
            execution_plan=requested_plan,
        )

        effective_plan, profile = session._resolve_execution_plan(
            request,
            text_tokens=173,
        )

        assert effective_plan is not None
        self.assertEqual(effective_plan.long_sequence_query_chunk_tokens, 16_384)
        self.assertEqual(effective_plan.long_sequence_projection_chunk_tokens, 4096)
        self.assertTrue(effective_plan.long_sequence_split_qkv_outputs)
        self.assertTrue(profile["long_sequence_split_qkv_outputs"])
        self.assertTrue(profile["long_sequence_parallel_sparse_lut"])
        self.assertTrue(profile["long_sequence_partial_sparse_topk"])
        self.assertTrue(profile["long_sequence_fused_prefix_k_quant"])
        self.assertTrue(effective_plan.long_sequence_direct_hnd_fp8_value)
        self.assertTrue(profile["long_sequence_direct_hnd_fp8_value"])
        self.assertEqual(profile["long_sequence_chunk_reason"], "explicit")


if __name__ == "__main__":
    unittest.main()
