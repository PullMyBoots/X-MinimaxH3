from __future__ import annotations

import unittest

import torch

from h3serve.native_engine.global_co_denoise import (
    fuse_global_av_predictions,
    plan_balanced_global_av_windows,
    plan_global_av_windows,
    plan_prompt_owned_global_av_windows,
    stabilize_global_selflift_seams,
    window_geometry_for_seconds,
)
from h3serve.native_engine.model.packed import build_fl2va_layout


class GlobalCoDenoiseTests(unittest.TestCase):
    def test_completed_audio_tail_balances_visual_views_without_extra_overlap(self) -> None:
        plan = plan_balanced_global_av_windows(
            243,
            window_frames=107,
            stride_frames=102,
        )
        self.assertEqual(plan.mechanism, "global_joint_av_balanced_views_v2")
        self.assertEqual([item.frames for item in plan.windows], [90, 90, 73])
        self.assertEqual([item.start_frame for item in plan.windows], [0, 85, 170])
        self.assertEqual(sum(item.frames for item in plan.windows), 253)
        self.assertEqual(plan.windows[-1].stop_frame, 243)
        self.assertTrue(
            all(
                item.audio_stop - item.audio_start == round(item.frames / 24 * 40)
                for item in plan.windows
            )
        )

    def test_creator_window_duration_is_h3_aligned_and_bounds_long_views(self) -> None:
        window_frames, stride_frames = window_geometry_for_seconds(5.0)
        self.assertEqual((window_frames, stride_frames), (124, 102))
        fifteen = plan_global_av_windows(
            362,
            window_frames=window_frames,
            stride_frames=stride_frames,
        )
        thirty = plan_global_av_windows(
            719,
            window_frames=window_frames,
            stride_frames=stride_frames,
        )
        self.assertEqual(len(fifteen.windows), 4)
        self.assertEqual(len(thirty.windows), 7)
        self.assertTrue(all(item.frames <= 124 for item in fifteen.windows))
        self.assertTrue(all(item.frames <= 124 for item in thirty.windows))

    def test_requested_overlap_snaps_to_legal_joint_av_geometry(self) -> None:
        self.assertEqual(window_geometry_for_seconds(6.0, 0.0), (141, 102))
        self.assertEqual(window_geometry_for_seconds(6.0, 4.0), (141, 51))
        self.assertEqual(window_geometry_for_seconds(5.0, 1.0), (124, 102))
        self.assertEqual(window_geometry_for_seconds(15.0, 1.0), (362, 357))
        with self.assertRaisesRegex(ValueError, "inside \\[0, 4\\]"):
            window_geometry_for_seconds(5.0, 4.1)

    def test_short_high_resolution_views_align_to_creator_boundaries(self) -> None:
        """Prompt writes stay inside their creator range without duplicate views."""

        # Three connected creator regions span roughly ten seconds apiece.
        # The final high-resolution pass is deliberately repartitioned into
        # five-second read views. A later prompt may read a short earlier halo,
        # but no identical view is evaluated under two prompts.
        plan = plan_prompt_owned_global_av_windows(
            719,
            ((0, 243), (243, 481), (481, 719)),
            window_frames=124,
            stride_frames=102,
        )
        self.assertEqual(plan.mechanism, "global_creator_aligned_co_denoise_v2")
        self.assertEqual(plan.window_frames, 124)
        self.assertEqual(len(plan.windows), 8)
        self.assertTrue(all(73 <= item.frames <= 141 for item in plan.windows))
        self.assertEqual(
            [item.prompt_index for item in plan.windows],
            [0, 0, 1, 1, 1, 2, 2, 2],
        )
        self.assertEqual(
            [item.start_frame for item in plan.windows],
            [0, 102, 204, 306, 408, 459, 561, 612],
        )
        self.assertEqual(len({item.start_frame for item in plan.windows}), 8)
        crossing_204 = [item for item in plan.windows if item.start_frame == 204]
        crossing_459 = [item for item in plan.windows if item.start_frame == 459]
        self.assertEqual(len(crossing_204), 1)
        self.assertEqual(len(crossing_459), 1)
        self.assertEqual(
            [(item.writable_video_start, item.writable_video_stop)
             for item in crossing_204],
            [(72, 97)],
        )
        self.assertEqual(
            [(item.writable_video_start, item.writable_video_stop)
             for item in crossing_459],
            [(142, 172)],
        )

    def test_prompt_owned_views_balance_within_each_creator_range(self) -> None:
        plan = plan_prompt_owned_global_av_windows(
            719,
            ((0, 243), (243, 481), (481, 719)),
            window_frames=175,
            stride_frames=153,
            balanced=True,
        )
        self.assertEqual(
            plan.mechanism,
            "global_creator_aligned_balanced_co_denoise_v3",
        )
        self.assertEqual(
            [(item.prompt_index, item.start_frame, item.frames)
             for item in plan.windows],
            [
                (0, 0, 141),
                (0, 119, 124),
                (1, 204, 158),
                (1, 340, 141),
                (2, 459, 141),
                (2, 578, 141),
            ],
        )
        self.assertTrue(all(item.frames <= 175 for item in plan.windows))
        self.assertEqual(plan.windows[-1].stop_frame, 719)

    def test_prompt_owned_fusion_never_blends_instructions_across_boundaries(self) -> None:
        plan = plan_prompt_owned_global_av_windows(
            719,
            ((0, 243), (243, 481), (481, 719)),
            window_frames=124,
            stride_frames=102,
        )
        video = torch.zeros(1, 1, plan.video_tokens, 1, 1)
        audio = torch.zeros(1, 1, 1, plan.audio_tokens)

        def predict(window, local_video, local_audio):
            value = float(window.prompt_index + 1)
            return (
                torch.full_like(local_video, value),
                torch.full_like(local_audio, value),
            )

        fused_video, fused_audio = fuse_global_av_predictions(
            video, audio, plan, predict
        )
        self.assertTrue(torch.allclose(fused_video[:, :, :72],
                                       torch.ones_like(fused_video[:, :, :72])))
        self.assertTrue(torch.allclose(fused_video[:, :, 72:142],
                                       torch.full_like(fused_video[:, :, 72:142], 2.0)))
        self.assertTrue(torch.allclose(fused_video[:, :, 142:],
                                       torch.full_like(fused_video[:, :, 142:], 3.0)))
        self.assertTrue(torch.allclose(fused_audio[..., :405],
                                       torch.ones_like(fused_audio[..., :405])))
        self.assertTrue(torch.allclose(fused_audio[..., 405:802],
                                       torch.full_like(fused_audio[..., 405:802], 2.0)))
        self.assertTrue(torch.allclose(fused_audio[..., 802:],
                                       torch.full_like(fused_audio[..., 802:], 3.0)))

    def test_internal_view_entries_start_with_zero_new_prediction_weight(self) -> None:
        plan = plan_prompt_owned_global_av_windows(
            719,
            ((0, 243), (243, 481), (481, 719)),
            window_frames=124,
            stride_frames=102,
        )
        video = torch.zeros(1, 1, plan.video_tokens, 1, 1)
        audio = torch.zeros(1, 1, 1, plan.audio_tokens)

        def predict(window, local_video, local_audio):
            value = float(window.index + 1)
            return (
                torch.full_like(local_video, value),
                torch.full_like(local_audio, value),
            )

        fused_video, _ = fuse_global_av_predictions(video, audio, plan, predict)
        # Window 6 enters at token 165 and window 7 enters at token 180. At
        # those exact positions the preceding view remains authoritative.
        self.assertEqual(float(fused_video[0, 0, 165, 0, 0]), 6.0)
        self.assertEqual(float(fused_video[0, 0, 180, 0, 0]), 7.0)
        # At each overlap's far edge, ownership has smoothly completed.
        self.assertEqual(float(fused_video[0, 0, 171, 0, 0]), 7.0)
        self.assertEqual(float(fused_video[0, 0, 201, 0, 0]), 8.0)

    def test_selflift_seam_guard_reduces_only_overlap_residual_pulse(self) -> None:
        plan = plan_prompt_owned_global_av_windows(
            719,
            ((0, 243), (243, 481), (481, 719)),
            window_frames=124,
            stride_frames=102,
        )
        anchor = torch.zeros(1, 2, plan.video_tokens, 3, 3)
        prediction = anchor.clone()
        prediction[:, :, 165] = 4.0
        prediction[:, :, 50] = 4.0
        guarded = stabilize_global_selflift_seams(
            anchor, prediction, plan, strength=1.0, padding_tokens=2
        )
        self.assertLess(
            float(guarded[:, :, 165].abs().mean()),
            float(prediction[:, :, 165].abs().mean()),
        )
        # The same pulse away from every overlap is outside the correction.
        self.assertTrue(torch.equal(guarded[:, :, 50], prediction[:, :, 50]))

    def test_selflift_seam_guard_preserves_constant_detail(self) -> None:
        plan = plan_prompt_owned_global_av_windows(
            719,
            ((0, 243), (243, 481), (481, 719)),
            window_frames=124,
            stride_frames=102,
        )
        anchor = torch.randn(1, 2, plan.video_tokens, 3, 3)
        prediction = anchor + 0.25
        guarded = stabilize_global_selflift_seams(anchor, prediction, plan)
        self.assertTrue(torch.allclose(guarded, prediction, atol=1e-6, rtol=0.0))

    def test_overlaps_keep_identical_absolute_h3_rotary_positions(self) -> None:
        plan = plan_global_av_windows(719)
        first, second = plan.windows[:2]
        first_layout = build_fl2va_layout(
            text_length=16,
            latent_frames=first.video_tokens,
            latent_height=4,
            latent_width=4,
            audio_frames=first.audio_tokens,
            target_time_offset=first.audio_start,
        )
        second_layout = build_fl2va_layout(
            text_length=16,
            latent_frames=second.video_tokens,
            latent_height=4,
            latent_width=4,
            audio_frames=second.audio_tokens,
            target_time_offset=second.audio_start,
        )

        first_audio = first_layout.position_ids[first_layout.segment("audio").start:
                                                 first_layout.segment("audio").stop]
        second_audio = second_layout.position_ids[second_layout.segment("audio").start:
                                                   second_layout.segment("audio").stop]
        audio_overlap = first.audio_stop - second.audio_start
        self.assertTrue(torch.equal(
            first_audio.reshape(2, first.audio_tokens, 3)[:, -audio_overlap:],
            second_audio.reshape(2, second.audio_tokens, 3)[:, :audio_overlap],
        ))

        rows_per_frame = 4
        first_video = first_layout.position_ids[first_layout.segment("video").start:
                                                 first_layout.segment("video").stop]
        second_video = second_layout.position_ids[second_layout.segment("video").start:
                                                   second_layout.segment("video").stop]
        video_overlap = first.video_stop - second.video_start
        self.assertTrue(torch.allclose(
            first_video.reshape(first.video_tokens, rows_per_frame, 3)[-video_overlap:],
            second_video.reshape(second.video_tokens, rows_per_frame, 3)[:video_overlap],
            rtol=0.0,
            atol=1e-12,
        ))

    def test_thirty_seconds_has_exact_prompt_agnostic_av_windows(self) -> None:
        plan = plan_global_av_windows(719)
        self.assertEqual(
            [(item.start_frame, item.frames) for item in plan.windows],
            [(0, 277), (204, 277), (408, 311)],
        )
        self.assertEqual(
            [(item.video_start, item.video_stop) for item in plan.windows],
            [(0, 82), (60, 142), (120, 212)],
        )
        self.assertEqual(
            [(item.audio_start, item.audio_stop) for item in plan.windows],
            [(0, 462), (340, 802), (680, 1198)],
        )
        self.assertFalse(plan.telemetry()["independent_completed_clips"])
        self.assertTrue(plan.telemetry()["global_scheduler_updates"])

    def test_sixty_seconds_keeps_bounded_windows_and_first_last_coverage(self) -> None:
        plan = plan_global_av_windows(1433)
        self.assertEqual(plan.windows[0].start_frame, 0)
        self.assertEqual(plan.windows[-1].stop_frame, 1433)
        self.assertEqual(plan.video_tokens, 422)
        self.assertEqual(plan.audio_tokens, 2388)
        self.assertTrue(all(item.frames <= 362 for item in plan.windows))
        self.assertTrue(all(item.frames >= 124 for item in plan.windows))
        self.assertTrue(all(
            left.stop_frame > right.start_frame
            for left, right in zip(plan.windows, plan.windows[1:])
        ))

    def test_weighted_prediction_fusion_is_continuous_and_global(self) -> None:
        plan = plan_global_av_windows(719)
        video = torch.zeros(1, 1, plan.video_tokens, 1, 1)
        audio = torch.zeros(1, 1, 1, plan.audio_tokens)

        def predict(window, local_video, local_audio):
            value = float(window.index + 1)
            return (
                torch.full_like(local_video, value),
                torch.full_like(local_audio, value),
            )

        fused_video, fused_audio = fuse_global_av_predictions(
            video, audio, plan, predict
        )
        self.assertEqual(float(fused_video[0, 0, 0, 0, 0]), 1.0)
        self.assertEqual(float(fused_video[0, 0, -1, 0, 0]), 3.0)
        self.assertEqual(float(fused_audio[0, 0, 0, 0]), 1.0)
        self.assertEqual(float(fused_audio[0, 0, 0, -1]), 3.0)
        # Overlap predictions are a normalized convex consensus, not a hard
        # handover to either independently completed clip.
        self.assertTrue(bool(((fused_video[:, :, 61:81] > 1.0) &
                              (fused_video[:, :, 61:81] < 2.0)).all()))
        self.assertTrue(bool(((fused_audio[..., 341:461] > 1.0) &
                              (fused_audio[..., 341:461] < 2.0)).all()))

    def test_identical_window_predictions_are_partition_invariant(self) -> None:
        plan = plan_global_av_windows(1433)
        video = torch.randn(1, 2, plan.video_tokens, 2, 2)
        audio = torch.randn(1, 2, 2, plan.audio_tokens)

        def predict(_window, local_video, local_audio):
            return local_video * 0.25 + 7.0, local_audio * 0.5 - 3.0

        fused_video, fused_audio = fuse_global_av_predictions(
            video, audio, plan, predict
        )
        self.assertTrue(torch.allclose(fused_video, video * 0.25 + 7.0))
        self.assertTrue(torch.allclose(fused_audio, audio * 0.5 - 3.0))


if __name__ == "__main__":
    unittest.main()
