from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from h3serve.native_engine.hot_session import (
    HotSessionRequest,
    blend_terminal_refinement_detail,
    build_refinement_region_mask,
    damp_unconverged_refinement_detail,
)
from scripts.benchmark_native_hot_session import load_scenarios, parse_args


class TerminalRefinementContractTests(unittest.TestCase):
    def _request(self, **overrides) -> HotSessionRequest:
        values = {
            "prompt": "test",
            "seed": 1,
            "width": 1280,
            "height": 736,
            "frames": 362,
            "fps": 24,
            "steps": 20,
            "output_path": Path("out.mp4"),
            "actual_step_indices": (0, 1, 2, 3, 4, 6, 8, 11, 14, 17, 18, 19),
            "terminal_refinement_initial_width": 864,
            "terminal_refinement_initial_height": 480,
            "terminal_refinement_steps": 1,
            "terminal_refinement_denoise": 0.0125,
            "terminal_refinement_dense_tail_steps": 1,
            "terminal_refinement_low_frequency_gain": 1.0,
            "terminal_refinement_temporal_lowpass": False,
            "terminal_refinement_temporal_outlier_only": False,
        }
        values.update(overrides)
        return HotSessionRequest(**values)

    def test_valid_original_12_8_terminal_refinement(self) -> None:
        self._request().validate()

    def test_partial_geometry_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "initial width and height"):
            self._request(terminal_refinement_initial_height=None).validate()

    def test_dense_tail_cannot_exceed_refinement_steps(self) -> None:
        with self.assertRaisesRegex(ValueError, "dense tail"):
            self._request(terminal_refinement_dense_tail_steps=2).validate()

    def test_low_frequency_gain_requires_terminal_refinement(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires terminal refinement"):
            self._request(
                terminal_refinement_initial_width=None,
                terminal_refinement_initial_height=None,
                terminal_refinement_steps=0,
                terminal_refinement_low_frequency_gain=0.25,
            ).validate()

    def test_motion_detail_blend_anchors_coarse_residual(self) -> None:
        motion = torch.zeros((1, 1, 1, 4, 4), dtype=torch.float32)
        refined = torch.ones_like(motion)
        anchored = blend_terminal_refinement_detail(
            motion,
            refined,
            source_height=2,
            source_width=2,
            low_frequency_gain=0.0,
        )
        self.assertTrue(torch.allclose(anchored, motion, atol=1e-6))
        unchanged = blend_terminal_refinement_detail(
            motion,
            refined,
            source_height=2,
            source_width=2,
            low_frequency_gain=1.0,
        )
        self.assertIs(unchanged, refined)

    def test_full_canvas_region_mask_preserves_context_and_feathers_edges(self) -> None:
        mask = build_refinement_region_mask(
            height=20,
            width=40,
            regions=((0.25, 0.25, 0.75, 0.75),),
            feather=0.10,
        )
        self.assertEqual(tuple(mask.shape), (1, 1, 1, 20, 40))
        self.assertEqual(float(mask[0, 0, 0, 0, 0]), 0.0)
        self.assertGreater(float(mask[0, 0, 0, 10, 20]), 0.99)
        self.assertGreater(float(mask[0, 0, 0, 5, 8]), 0.0)
        self.assertLess(float(mask[0, 0, 0, 5, 8]), 1.0)

    def test_full_canvas_regions_require_refinement_source(self) -> None:
        request = HotSessionRequest(
            prompt="test",
            seed=1,
            width=1280,
            height=736,
            frames=124,
            fps=24,
            steps=4,
            output_path=Path("out.mp4"),
            refinement_full_canvas_regions=((0.1, 0.1, 0.2, 0.2),),
        )
        with self.assertRaisesRegex(ValueError, "detail-regeneration controls"):
            request.validate()

    def test_temporal_lowpass_preserves_constant_and_suppresses_spike(self) -> None:
        motion = torch.zeros((1, 1, 3, 4, 4), dtype=torch.float32)
        constant = torch.ones_like(motion)
        preserved = blend_terminal_refinement_detail(
            motion,
            constant,
            source_height=2,
            source_width=2,
            low_frequency_gain=1.0,
            temporal_lowpass=True,
        )
        self.assertTrue(torch.allclose(preserved, constant, atol=1e-6))
        spike = torch.zeros_like(motion)
        spike[:, :, 1] = 1.0
        smoothed = blend_terminal_refinement_detail(
            motion,
            spike,
            source_height=2,
            source_width=2,
            low_frequency_gain=1.0,
            temporal_lowpass=True,
        )
        self.assertTrue(torch.allclose(smoothed[:, :, 1], torch.full((1, 1, 4, 4), 0.5)))
        self.assertTrue(torch.allclose(smoothed[:, :, 0], torch.full((1, 1, 4, 4), 0.25)))

    def test_robust_temporal_filter_changes_only_outlier_excess(self) -> None:
        motion = torch.zeros((1, 1, 9, 4, 4), dtype=torch.float32)
        refined = torch.zeros_like(motion)
        refined[:, :, 4] = 10.0
        filtered = blend_terminal_refinement_detail(
            motion,
            refined,
            source_height=2,
            source_width=2,
            low_frequency_gain=1.0,
            temporal_lowpass=True,
            temporal_outlier_only=True,
        )
        self.assertLess(float(filtered[:, :, 4].mean()), 10.0)
        self.assertGreater(float(filtered[:, :, 4].mean()), 0.0)
        self.assertTrue(torch.allclose(filtered[:, :, 0], refined[:, :, 0]))

    def test_motion_aware_detail_filter_suppresses_only_new_texture_spike(self) -> None:
        motion = torch.zeros((1, 1, 9, 8, 8), dtype=torch.float32)
        refined = torch.zeros_like(motion)
        checker = torch.tensor(
            [[1.0, -1.0] * 4, [-1.0, 1.0] * 4] * 4,
            dtype=torch.float32,
        )
        refined[:, :, 4] = checker * 10.0
        filtered = blend_terminal_refinement_detail(
            motion,
            refined,
            source_height=4,
            source_width=4,
            low_frequency_gain=1.0,
            temporal_detail_outlier_strength=0.5,
        )
        self.assertLess(
            float(filtered[:, :, 4].abs().mean()),
            float(refined[:, :, 4].abs().mean()),
        )
        self.assertGreater(float(filtered[:, :, 4].abs().mean()), 0.0)
        self.assertTrue(torch.allclose(filtered[:, :, 0], refined[:, :, 0]))

        constant = checker.view(1, 1, 1, 8, 8).repeat(1, 1, 9, 1, 1)
        preserved = blend_terminal_refinement_detail(
            motion,
            constant,
            source_height=4,
            source_width=4,
            low_frequency_gain=1.0,
            temporal_detail_outlier_strength=0.5,
        )
        self.assertTrue(torch.allclose(preserved, constant, atol=1e-6))

    def test_cross_step_confidence_damps_only_unconverged_detail(self) -> None:
        motion = torch.zeros((1, 1, 3, 8, 8), dtype=torch.float32)
        checker = torch.tensor(
            [[1.0, -1.0] * 4, [-1.0, 1.0] * 4] * 4,
            dtype=torch.float32,
        )
        previous = checker.view(1, 1, 1, 8, 8).repeat(1, 1, 3, 1, 1)
        refined = previous.clone()
        refined[:, :, 1, 2:6, 2:6] *= 4.0
        filtered = damp_unconverged_refinement_detail(
            motion,
            previous,
            refined,
            source_height=4,
            source_width=4,
            strength=1.0,
        )
        error_before = (refined - previous).abs().mean()
        error_after = (filtered - previous).abs().mean()
        self.assertLess(float(error_after), float(error_before))
        self.assertTrue(torch.allclose(filtered[:, :, 0], refined[:, :, 0]))
        preserved = damp_unconverged_refinement_detail(
            motion,
            previous,
            previous,
            source_height=4,
            source_width=4,
            strength=1.0,
        )
        self.assertTrue(torch.allclose(preserved, previous, atol=1e-6))

    def test_registry_preserves_terminal_refinement_fields(self) -> None:
        root = Path(__file__).resolve().parents[2]
        registry = root / "tests/fixtures/motorcycle_terminal_refine_candidate.json"
        with patch.object(sys, "argv", [
            "benchmark_native_hot_session.py",
            "--engine", "original",
            "--candidate-registry", str(registry),
            "--repeat", "1",
        ]):
            args = parse_args()
        scenario = load_scenarios(args)[0]
        self.assertEqual(scenario["width"], 1280)
        self.assertEqual(scenario["height"], 736)
        self.assertEqual(scenario["terminal_refinement_initial_width"], 864)
        self.assertEqual(scenario["terminal_refinement_initial_height"], 480)
        self.assertEqual(scenario["terminal_refinement_steps"], 1)
        self.assertEqual(scenario["terminal_refinement_denoise"], 0.0125)


if __name__ == "__main__":
    unittest.main()
