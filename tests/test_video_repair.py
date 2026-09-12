from __future__ import annotations

import unittest
from unittest.mock import patch

import cv2
import numpy as np

from h3serve.video_repair import (
    _grid_periodicity,
    RepairRegion,
    detect_regions,
    plan_face_atlas_batches,
    plan_repair_windows,
)


class VideoRepairPipelineTest(unittest.TestCase):
    def test_temporal_windows_cover_every_frame_with_normalized_overlap(self) -> None:
        windows = plan_repair_windows(
            360, 24.0, minimum_seconds=4.0,
            maximum_seconds=6.0, overlap_seconds=0.2
        )
        self.assertGreater(len(windows), 1)
        self.assertTrue(all(96 <= window.padded_frames <= 144 for window in windows))
        self.assertEqual(len({window.padded_frames for window in windows}), 1)
        self.assertTrue(all((window.padded_frames - 5) % 17 == 0 for window in windows))
        self.assertTrue(all(window.padded_frames >= 56 for window in windows))

        coverage = np.zeros(360, dtype=np.float64)
        for window in windows:
            coverage[window.start:window.end] += np.asarray(window.weights)
        np.testing.assert_allclose(coverage, 1.0, atol=1.0e-8)

    def test_five_second_clip_stays_in_one_canonical_window(self) -> None:
        windows = plan_repair_windows(124, 24.0)
        self.assertEqual(len(windows), 1)
        self.assertEqual((windows[0].start, windows[0].end), (0, 124))
        self.assertEqual(windows[0].padded_frames, 124)

    def test_awkward_tail_is_padded_into_four_to_six_second_band(self) -> None:
        windows = plan_repair_windows(150, 24.0)
        self.assertEqual(len(windows), 2)
        self.assertTrue(all(96 <= item.padded_frames <= 144 for item in windows))
        self.assertEqual({item.padded_frames for item in windows}, {107})

    def test_face_detector_tracks_people_and_prefers_under_resolved_face(self) -> None:
        frames = []
        for index in range(9):
            frame = np.full((240, 320, 3), 110, dtype=np.uint8)
            # A small, flat, moving face is a stronger repair candidate.
            cv2.circle(frame, (60 + index, 40), 10, (118, 118, 118), -1)
            # A large face with resolved high-frequency structure should rank lower.
            for y in range(60, 130, 4):
                level = 40 if y % 8 else 220
                cv2.line(frame, (150, y), (220, y), (level, level, level), 2)
            frames.append(frame)
        detections = [
            [(50 + index, 30, 20, 20), (150, 60, 70, 70)]
            for index in range(9)
        ]
        with patch("h3serve.video_repair._face_detections", side_effect=detections):
            regions = detect_regions(
                frames, mode="face", crop_size=80, maximum_regions=1
            )
        self.assertEqual(len(regions), 1)
        self.assertLess(regions[0].face_size, 30)
        self.assertEqual(len(regions[0].positions), len(frames))
        self.assertGreater(regions[0].positions[-1][0], regions[0].positions[0][0])

    def test_face_detector_respects_user_face_limit(self) -> None:
        frames = [np.full((240, 320, 3), 96, dtype=np.uint8) for _ in range(5)]
        detections = [
            [(10 + column * 45, 30 + row * 70, 20, 20) for row in range(2) for column in range(3)]
            for _ in frames
        ]
        with patch("h3serve.video_repair._face_detections", side_effect=detections):
            regions = detect_regions(
                frames, mode="face", crop_size=64, maximum_regions=4
            )
        self.assertEqual(len(regions), 4)
        self.assertEqual(len({region.track_id for region in regions}), 4)

    def test_transient_tiny_false_positive_cannot_outrank_real_face(self) -> None:
        frames = [np.full((240, 320, 3), 96, dtype=np.uint8) for _ in range(20)]
        detections = []
        for index in range(len(frames)):
            row = [(120 + index // 4, 50, 20, 20, 0.82)]
            if index == 8:
                # A tiny one-frame detection has high repair need and high
                # confidence, but it is not a persistent face track.
                row.append((155, 130, 10, 10, 0.99))
            detections.append(row)
        with patch("h3serve.video_repair._face_detections", side_effect=detections):
            regions = detect_regions(
                frames, mode="face", crop_size=80, maximum_regions=1
            )
        self.assertEqual(len(regions), 1)
        self.assertEqual(regions[0].face_size, 20.0)
        self.assertGreater(regions[0].visibility, 0.9)

    def test_face_atlas_uses_minimum_canvas_and_only_splits_when_required(self) -> None:
        small = tuple(
            RepairRegion(0, 0, 32, score=1.0, track_id=index)
            for index in range(6)
        )
        batches = plan_face_atlas_batches(small, 3.0)
        self.assertEqual(len(batches), 1)
        self.assertEqual((batches[0][1], batches[0][2]), (288, 3))

        medium = tuple(
            RepairRegion(0, 0, 128, score=1.0, track_id=index)
            for index in range(6)
        )
        batches = plan_face_atlas_batches(medium, 3.0)
        self.assertEqual([len(item[0]) for item in batches], [4, 2])
        self.assertTrue(all(item[1] == 768 for item in batches))
        self.assertTrue(all(item[1] % 32 == 0 for item in batches))

        distant = (
            RepairRegion(0, 0, 36, score=1.0, track_id=1),
            RepairRegion(0, 0, 32, score=1.0, track_id=2),
        )
        batches = plan_face_atlas_batches(distant, 10.0)
        self.assertEqual(len(batches), 1)
        self.assertEqual((batches[0][1], batches[0][2]), (736, 2))

    def test_periodic_grid_gate_distinguishes_grid_from_random_residual(self) -> None:
        rng = np.random.default_rng(7)
        random = rng.normal(0, 1, (128, 128)).astype(np.float32)
        grid = random.copy()
        grid[:, ::8] += 6.0
        self.assertGreater(
            _grid_periodicity(grid), _grid_periodicity(random) * 1.25
        )


if __name__ == "__main__":
    unittest.main()
