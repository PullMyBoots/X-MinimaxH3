from __future__ import annotations

import unittest

import torch

from h3serve.native_engine.roi_atlas import (
    _temporal_motion_gated_filter,
    _temporal_residual_lowpass,
    build_region_atlas,
    detect_difficult_regions,
    merge_region_atlas,
    remap_region_atlas_positions,
)
from h3serve.native_engine.model.packed import build_fl2va_layout


class RegionAtlasTests(unittest.TestCase):
    def setUp(self) -> None:
        generator = torch.Generator(device="cpu").manual_seed(17)
        self.video = torch.randn(
            (1, 4, 7, 48, 84), generator=generator, dtype=torch.float32
        )
        self.regions = (
            (0.39, 0.12, 0.54, 0.39),
            (0.53, 0.18, 0.68, 0.45),
            (0.77, 0.16, 0.92, 0.43),
        )

    def test_atlas_preserves_canvas_and_magnifies_regions(self) -> None:
        atlas, records = build_region_atlas(self.video, self.regions)
        self.assertEqual(atlas.shape, self.video.shape)
        self.assertEqual(len(records), 3)
        for record in records:
            self.assertEqual(record.atlas_y1 - record.atlas_y0, 24)
            self.assertEqual(record.atlas_x1 - record.atlas_x0, 24)
            self.assertGreater(record.atlas_x1 - record.atlas_x0, record.source_width)

    def test_noop_refinement_is_noop_merge(self) -> None:
        atlas, records = build_region_atlas(self.video, self.regions)
        merged = merge_region_atlas(
            self.video,
            atlas,
            atlas.clone(),
            records,
            source_scale=480 / 768,
        )
        torch.testing.assert_close(merged, self.video, rtol=0.0, atol=1e-6)

    def test_square_budget_atlas_can_differ_from_full_frame_shape(self) -> None:
        atlas, records = build_region_atlas(
            self.video,
            self.regions,
            rows=2,
            columns=2,
            atlas_height=36,
            atlas_width=36,
        )
        self.assertEqual(atlas.shape, (1, 4, 7, 36, 36))
        self.assertEqual(len(records), 3)
        merged = merge_region_atlas(
            self.video,
            atlas,
            atlas.clone(),
            records,
            source_scale=480 / 768,
            mid_frequency_gain=0.65,
        )
        torch.testing.assert_close(merged, self.video, rtol=0.0, atol=1e-6)

    def test_merge_changes_only_requested_spatial_support(self) -> None:
        atlas, records = build_region_atlas(self.video, self.regions[:1])
        refined = atlas.clone()
        record = records[0]
        refined[
            ...,
            record.atlas_y0 : record.atlas_y1,
            record.atlas_x0 : record.atlas_x1,
        ] += torch.randn_like(
            refined[
                ...,
                record.atlas_y0 : record.atlas_y1,
                record.atlas_x0 : record.atlas_x1,
            ]
        ) * 0.1
        merged = merge_region_atlas(
            self.video,
            atlas,
            refined,
            records,
            source_scale=480 / 768,
        )
        outside = torch.ones((48, 84), dtype=torch.bool)
        outside[
            record.source_y0 : record.source_y1,
            record.source_x0 : record.source_x1,
        ] = False
        difference = (merged - self.video).abs()[0, :, :, outside]
        self.assertEqual(float(difference.max()), 0.0)
        inside = (merged - self.video).abs()[
            ...,
            record.source_y0 : record.source_y1,
            record.source_x0 : record.source_x1,
        ]
        self.assertGreater(float(inside.max()), 0.0)

    def test_source_foveated_positions_preserve_layout_and_temporal_axis(self) -> None:
        regions = (
            (0.39, 0.12, 0.54, 0.39),
            (0.53, 0.18, 0.68, 0.45),
            (0.77, 0.16, 0.92, 0.43),
            (0.64, 0.12, 0.76, 0.39),
        )
        atlas, records = build_region_atlas(
            self.video,
            regions,
            rows=2,
            columns=2,
            atlas_height=36,
            atlas_width=36,
        )
        layout = build_fl2va_layout(
            text_length=8,
            latent_frames=atlas.shape[2],
            latent_height=atlas.shape[3],
            latent_width=atlas.shape[4],
            audio_frames=11,
        )
        signature = layout.signature
        video_segment = layout.segment("video", last=True)
        audio_segment = layout.segment("audio", last=True)
        before_video = layout.position_ids[
            video_segment.start : video_segment.stop
        ].clone()
        before_temporal = before_video[:, 0].clone()
        before_audio = layout.position_ids[
            audio_segment.start : audio_segment.stop
        ].clone()
        layout.device_rope_table = torch.ones(1)

        remap_region_atlas_positions(
            layout,
            records,
            atlas_height=atlas.shape[3],
            atlas_width=atlas.shape[4],
            full_height=self.video.shape[3],
            full_width=self.video.shape[4],
        )

        after_video = layout.position_ids[
            video_segment.start : video_segment.stop
        ]
        after_audio = layout.position_ids[
            audio_segment.start : audio_segment.stop
        ]
        self.assertEqual(layout.signature, signature)
        torch.testing.assert_close(after_video[:, 0], before_temporal)
        self.assertFalse(torch.equal(after_video[:, 1:], before_video[:, 1:]))
        self.assertFalse(torch.equal(after_audio[:, 2], before_audio[:, 2]))
        self.assertIsNone(layout.device_rope_table)

        frame = after_video[: 18 * 18].reshape(18, 18, 3)
        first = records[0]
        second = records[1]
        first_x = frame[
            first.atlas_y0 // 2 : first.atlas_y1 // 2,
            first.atlas_x0 // 2 : first.atlas_x1 // 2,
            2,
        ]
        second_x = frame[
            second.atlas_y0 // 2 : second.atlas_y1 // 2,
            second.atlas_x0 // 2 : second.atlas_x1 // 2,
            2,
        ]
        self.assertLess(float(first_x.max()), float(second_x.min()))

    def test_motion_gate_suppresses_static_flicker_and_preserves_constant_detail(
        self,
    ) -> None:
        motion = torch.zeros((1, 4, 7, 6, 6), dtype=torch.float32)
        alternating = torch.tensor(
            [0.0, 1.0, -1.0, 1.0, -1.0, 1.0, 0.0],
            dtype=torch.float32,
        ).view(1, 1, 7, 1, 1).expand_as(motion)
        filtered = _temporal_motion_gated_filter(
            alternating,
            motion,
            strength=0.9,
        )
        self.assertLess(float(filtered.abs().mean()), float(alternating.abs().mean()))

        constant = torch.full_like(motion, 0.25)
        constant_filtered = _temporal_motion_gated_filter(
            constant,
            motion,
            strength=0.9,
        )
        torch.testing.assert_close(constant_filtered, constant)

    def test_residual_lowpass_reduces_flicker_without_changing_constant_detail(
        self,
    ) -> None:
        alternating = torch.tensor(
            [0.0, 1.0, -1.0, 1.0, -1.0, 1.0, 0.0],
            dtype=torch.float32,
        ).view(1, 1, 7, 1, 1).expand(1, 4, 7, 6, 6)
        filtered = _temporal_residual_lowpass(alternating, strength=0.65)
        self.assertLess(float(filtered.abs().mean()), float(alternating.abs().mean()))
        constant = torch.full_like(alternating, 0.25)
        torch.testing.assert_close(
            _temporal_residual_lowpass(constant, strength=0.65),
            constant,
        )

    def test_motion_gated_filter_reduces_static_temporal_flicker(self) -> None:
        video = torch.zeros((1, 4, 7, 24, 24), dtype=torch.float32)
        regions = ((0.25, 0.25, 0.75, 0.75),)
        atlas, records = build_region_atlas(
            video,
            regions,
            rows=1,
            columns=1,
            atlas_height=24,
            atlas_width=24,
        )
        refined = atlas.clone()
        alternating = torch.tensor(
            [0.0, 0.2, -0.2, 0.2, -0.2, 0.2, 0.0],
            dtype=torch.float32,
        ).view(1, 1, 7, 1, 1)
        refined += alternating
        unfiltered = merge_region_atlas(
            video,
            atlas,
            refined,
            records,
            source_scale=0.625,
            low_frequency_gain=1.0,
            mid_frequency_gain=1.0,
            blend=1.0,
            temporal_outlier_strength=0.0,
            temporal_filter="motion_gated",
        )
        filtered = merge_region_atlas(
            video,
            atlas,
            refined,
            records,
            source_scale=0.625,
            low_frequency_gain=1.0,
            mid_frequency_gain=1.0,
            blend=1.0,
            temporal_outlier_strength=1.0,
            temporal_filter="motion_gated",
        )
        unfiltered_second = torch.diff(unfiltered, n=2, dim=2).abs().mean()
        filtered_second = torch.diff(filtered, n=2, dim=2).abs().mean()
        self.assertLess(float(filtered_second), float(unfiltered_second) * 0.8)

    def test_auto_selector_finds_cross_step_unconverged_patch(self) -> None:
        motion = torch.zeros((1, 8, 7, 48, 84), dtype=torch.float32)
        previous = motion.clone()
        refined = motion.clone()
        pattern = torch.tensor(
            [[1.0, -1.0], [-1.0, 1.0]], dtype=torch.float32
        ).repeat(5, 5)
        for frame in range(7):
            refined[:, :, frame, 15:25, 54:64] = pattern * (
                0.35 if frame % 2 == 0 else -0.35
            )
        regions, profile = detect_difficult_regions(
            motion,
            previous,
            refined,
            source_height=24,
            source_width=42,
            maximum_regions=1,
        )
        self.assertEqual(len(regions), 1)
        x0, y0, x1, y1 = regions[0]
        self.assertLessEqual(x0, 59 / 84)
        self.assertGreaterEqual(x1, 59 / 84)
        self.assertLessEqual(y0, 20 / 48)
        self.assertGreaterEqual(y1, 20 / 48)
        self.assertEqual(
            profile["policy"], "h3_cross_step_structural_difficulty_v1"
        )

    def test_auto_selector_finds_structured_region_with_no_new_detail(self) -> None:
        motion = torch.zeros((1, 8, 7, 48, 84), dtype=torch.float32)
        pattern = torch.tensor(
            [[0.7, 0.7, -0.7, -0.7], [0.7, 0.7, -0.7, -0.7]],
            dtype=torch.float32,
        ).repeat(5, 3)
        motion[:, :, :, 28:38, 18:30] = pattern
        regions, _ = detect_difficult_regions(
            motion,
            motion.clone(),
            motion.clone(),
            source_height=24,
            source_width=42,
            maximum_regions=1,
        )
        self.assertEqual(len(regions), 1)
        x0, y0, x1, y1 = regions[0]
        self.assertLessEqual(x0, 24 / 84)
        self.assertGreaterEqual(x1, 24 / 84)
        self.assertLessEqual(y0, 33 / 48)
        self.assertGreaterEqual(y1, 33 / 48)

    def test_auto_selector_skips_featureless_converged_video(self) -> None:
        video = torch.zeros((1, 8, 7, 48, 84), dtype=torch.float32)
        regions, profile = detect_difficult_regions(
            video,
            video.clone(),
            video.clone(),
            source_height=24,
            source_width=42,
            maximum_regions=4,
        )
        self.assertEqual(regions, ())
        self.assertEqual(profile["selected_regions"], [])


if __name__ == "__main__":
    unittest.main()
