from __future__ import annotations

import unittest

import torch

from h3serve.native_engine.ultimate_upscale import (
    TemporalPiece,
    append_av_temporal_piece,
    frames_for_video_tokens,
    plan_ultimate_upscale,
    spatial_tiles,
    slice_av_temporal_piece,
    temporal_pieces,
    video_tokens_for_frames,
)


class UltimateUpscaleNativePlanningTests(unittest.TestCase):
    def test_h3_frame_token_grid_matches_15_second_latent(self) -> None:
        self.assertEqual(frames_for_video_tokens(107), 362)
        self.assertEqual(video_tokens_for_frames(362), 107)

    def test_temporal_pieces_are_phase_aligned_and_cover_the_tail(self) -> None:
        pieces = temporal_pieces(107, chunk_frames=124, overlap_frames=17)
        self.assertEqual(pieces[0].frame_start, 0)
        self.assertEqual(pieces[-1].frame_stop, 362)
        self.assertTrue(all(piece.video_token_start % 5 == 0 for piece in pieces))
        self.assertGreater(sum(piece.frames for piece in pieces), 362)

    def test_spatial_grid_reports_real_edge_overlap(self) -> None:
        tiles = spatial_tiles(
            736, 1280,
            tile_height=640,
            tile_width=640,
            overlap_height=128,
            overlap_width=128,
            minimum_tile=256,
        )
        self.assertGreater(len(tiles), 1)
        self.assertTrue(any(tile.left_overlap for tile in tiles))
        self.assertTrue(any(tile.top_overlap for tile in tiles))

    def test_native_av_window_slice_uses_independent_audio_clock(self) -> None:
        video = torch.arange(10.0).view(1, 1, 10, 1, 1)
        audio = torch.arange(20.0).view(1, 1, 1, 20)
        piece = TemporalPiece(3, 10, 7, 24, 8, 16)
        piece_video, piece_audio = slice_av_temporal_piece(video, audio, piece)
        self.assertEqual(tuple(piece_video.shape), (1, 1, 4, 1, 1))
        self.assertEqual(tuple(piece_audio.shape), (1, 1, 1, 8))
        self.assertEqual(piece_video.flatten().tolist(), [3.0, 4.0, 5.0, 6.0])
        self.assertEqual(piece_audio.flatten().tolist(), list(range(8, 16)))

    def test_native_temporal_append_crossfades_overlap(self) -> None:
        first = TemporalPiece(0, 0, 5, 17, 0, 5)
        second = TemporalPiece(3, 10, 8, 27, 3, 8)
        video_a = torch.zeros((1, 1, 5, 1, 1))
        audio_a = torch.zeros((1, 1, 1, 5))
        video_b = torch.ones((1, 1, 5, 1, 1))
        audio_b = torch.ones((1, 1, 1, 5))
        acc_v, acc_a = append_av_temporal_piece(None, None, video_a, audio_a, first)
        acc_v, acc_a = append_av_temporal_piece(
            acc_v, acc_a, video_b, audio_b, second
        )
        self.assertEqual(tuple(acc_v.shape), (1, 1, 8, 1, 1))
        self.assertEqual(tuple(acc_a.shape), (1, 1, 1, 8))
        self.assertEqual(acc_v.flatten().tolist(), [0, 0, 0, 0, 1, 1, 1, 1])
        self.assertEqual(acc_a.flatten().tolist(), [0, 0, 0, 0, 1, 1, 1, 1])

    def test_24gib_2k15_uses_temporal_windows_after_real_thrash(self) -> None:
        plan = plan_ultimate_upscale(
            target_width=2560,
            target_height=1440,
            frames=362,
            device_budget_bytes=23 * 1024**3,
            text_tokens=512,
            condition_count=0,
        )
        self.assertFalse(plan.full_canvas)
        self.assertGreater(plan.redundancy_ratio, 1.0)
        self.assertEqual(len(plan.temporal), 3)
        self.assertEqual(len(plan.spatial), 1)
        self.assertEqual(
            plan.memory_execution["admission_reason"],
            "2k_long_pcie_thrash_guard",
        )

    def test_user_window_forces_full_spatial_temporal_pieces(self) -> None:
        plan = plan_ultimate_upscale(
            target_width=1920,
            target_height=1088,
            frames=362,
            device_budget_bytes=23 * 1024**3,
            text_tokens=512,
            condition_count=0,
            temporal_window_frames=119,
            temporal_overlap_frames=39,
        )
        self.assertFalse(plan.full_canvas)
        self.assertEqual(len(plan.spatial), 1)
        self.assertGreater(len(plan.temporal), 1)
        self.assertLessEqual(max(piece.frames for piece in plan.temporal), 136)
        self.assertEqual(
            plan.memory_execution["admission_reason"], "user_temporal_window"
        )
        self.assertEqual(
            plan.memory_execution["requested_temporal_window_frames"], 119
        )
        self.assertEqual(
            plan.memory_execution["requested_temporal_overlap_frames"], 39
        )

    def test_24gib_720p45_auto_stays_inside_native_temporal_horizon(self) -> None:
        plan = plan_ultimate_upscale(
            target_width=1280,
            target_height=736,
            frames=1076,
            device_budget_bytes=23 * 1024**3,
            text_tokens=512,
            condition_count=0,
            actual_evaluations=2,
        )
        self.assertFalse(plan.full_canvas)
        self.assertEqual(len(plan.temporal), 4)
        self.assertEqual(len(plan.spatial), 1)
        self.assertLessEqual(max(piece.frames for piece in plan.temporal), 362)
        self.assertEqual(
            plan.memory_execution["admission_reason"], "native_horizon_guard"
        )
        self.assertEqual(
            plan.memory_execution["automatic_temporal_window_frames"], 328
        )

    def test_720p45_fallback_cannot_re_admit_full_duration(self) -> None:
        plan = plan_ultimate_upscale(
            target_width=1280,
            target_height=736,
            frames=1076,
            device_budget_bytes=10 * 1024**3,
            text_tokens=512,
            condition_count=0,
            actual_evaluations=2,
        )
        self.assertFalse(plan.full_canvas)
        self.assertGreater(len(plan.temporal), 1)
        self.assertLessEqual(max(piece.frames for piece in plan.temporal), 362)

    def test_user_full_context_can_override_throughput_window(self) -> None:
        plan = plan_ultimate_upscale(
            target_width=2560,
            target_height=1440,
            frames=362,
            device_budget_bytes=23 * 1024**3,
            text_tokens=512,
            condition_count=0,
            temporal_window_frames=362,
        )
        self.assertTrue(plan.full_canvas)
        self.assertEqual(plan.redundancy_ratio, 1.0)

    def test_8gib_720p15_selects_bounded_pieces(self) -> None:
        plan = plan_ultimate_upscale(
            target_width=1280,
            target_height=736,
            frames=362,
            device_budget_bytes=7 * 1024**3,
            text_tokens=512,
            condition_count=0,
        )
        self.assertFalse(plan.full_canvas)
        self.assertGreater(len(plan.temporal) * len(plan.spatial), 1)
        self.assertTrue(plan.memory_execution["fits_budget"])
        self.assertGreater(plan.redundancy_ratio, 1.0)

    def test_w4a8_8gib_1080p15_uses_full_spatial_temporal_windows(self) -> None:
        plan = plan_ultimate_upscale(
            target_width=1920,
            target_height=1088,
            frames=362,
            device_budget_bytes=int(7.25 * 1024**3),
            text_tokens=512,
            condition_count=1,
            weight_tier="w4a8",
            resource_profile="w4a8_8gb",
        )
        self.assertFalse(plan.full_canvas)
        self.assertGreater(len(plan.temporal), 1)
        self.assertEqual(len(plan.spatial), 1)
        self.assertTrue(plan.memory_execution["fits_budget"])
        self.assertEqual(
            plan.memory_execution["vae_output_strategy"], "not_executed"
        )
        self.assertLessEqual(
            plan.memory_execution["estimated_selected_peak_gib"], 7.125
        )


if __name__ == "__main__":
    unittest.main()
