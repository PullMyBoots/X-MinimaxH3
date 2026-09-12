from __future__ import annotations

from pathlib import Path
import tempfile
import time
import unittest

import torch

from h3serve.native_engine.long_horizon import (
    continuation_retry_seed,
    evaluate_video_repaint_overlap_files,
    evaluate_visible_video_trajectory_file,
    localize_h3_prompt,
    plan_long_horizon,
    plan_shot_decode_groups,
    prepare_masked_av_prefix,
    restore_masked_av_prefix_,
    scale_h3_prompt_timeline,
    stitch_audio_segment_files,
    stitch_clean_av_segment_files,
    stitch_clean_av_segments,
)


PROMPT = """integrated_multimodal_description: [Shot 1] A woman enters a radio room and finds a brass key. [Shot 2] At 00:10.000, she gives the key to a man. [Shot 3] At 00:20.000, he unlocks a cabinet and removes a red envelope.

overall_soundscape: Quiet radio static and natural footsteps.

non_diegetic_music: N/A"""


class LongHorizonTests(unittest.TestCase):
    def test_continuation_overlap_gate_accepts_same_layout_and_rejects_mode_jump(self) -> None:
        previous = torch.randn(1, 24, 52, 16, 24)
        incoming = torch.randn(1, 24, 82, 16, 24)
        incoming[:, :, :22].copy_(previous[:, :, -27:-5])
        incoming[:, :, 22:27].copy_(previous[:, :, -5:])
        with tempfile.TemporaryDirectory(prefix="h3-overlap-gate-") as root:
            old_path = Path(root) / "old.pt"
            good_path = Path(root) / "good.pt"
            bad_path = Path(root) / "bad.pt"
            torch.save({"video": previous}, old_path)
            torch.save({"video": incoming}, good_path)
            bad = incoming.clone()
            bad[:, :, 24:26].add_(4.0)
            torch.save({"video": bad}, bad_path)
            good = evaluate_video_repaint_overlap_files(
                old_path,
                good_path,
                context_frames=90,
                repaint_frames=17,
            )
            rejected = evaluate_video_repaint_overlap_files(
                old_path,
                bad_path,
                context_frames=90,
                repaint_frames=17,
            )
        self.assertTrue(good["accepted"])
        self.assertFalse(rejected["accepted"])
        self.assertLess(good["maximum_low_frequency_relative_rms"], 1e-6)
        self.assertGreater(rejected["maximum_low_frequency_relative_rms"], 0.3)

    def test_continuation_retry_seeds_are_stable_and_distinct(self) -> None:
        seeds = [continuation_retry_seed(1234, index) for index in range(4)]
        self.assertEqual(seeds[0], 1234)
        self.assertEqual(seeds, [continuation_retry_seed(1234, index) for index in range(4)])
        self.assertEqual(len(set(seeds)), 4)

    def test_visible_trajectory_gate_rejects_isolated_latent_mode_jump(self) -> None:
        temporal_groups = 16
        latent = torch.empty(1, 4, 2 + 5 * temporal_groups, 8, 8)
        latent[:, :, :2].zero_()
        base = torch.linspace(-1.0, 1.0, steps=4 * 8 * 8).reshape(1, 4, 8, 8)
        for group in range(temporal_groups):
            for phase in range(5):
                latent[:, :, 2 + 5 * group + phase].copy_(
                    base + 0.01 * group + 0.002 * phase
                )
        jumped = latent.clone()
        jumped[:, :, 2 + 5 * 9 :].add_(4.0)
        with tempfile.TemporaryDirectory(prefix="h3-trajectory-gate-") as root:
            smooth_path = Path(root) / "smooth.pt"
            jumped_path = Path(root) / "jumped.pt"
            torch.save({"video": latent}, smooth_path)
            torch.save({"video": jumped}, jumped_path)
            smooth = evaluate_visible_video_trajectory_file(
                smooth_path,
                context_frames=90,
            )
            rejected = evaluate_visible_video_trajectory_file(
                jumped_path,
                context_frames=90,
            )
        self.assertTrue(smooth["accepted"])
        self.assertFalse(rejected["accepted"])
        self.assertEqual(rejected["rejected_transition_video_frames"], [158])
        culprit = next(
            item for item in rejected["transitions"]
            if item["unauthorized_cut"]
        )
        self.assertGreaterEqual(culprit["low_frequency_relative_rms"], 0.60)
        self.assertGreaterEqual(culprit["local_outlier_ratio"], 1.50)

    def test_tes_prompt_clock_is_scaled_without_semantic_rules(self) -> None:
        scaled = scale_h3_prompt_timeline(
            PROMPT,
            source_frames=719,
            target_frames=277,
        )
        self.assertIn("At 00:03.853", scaled)
        self.assertIn("At 00:07.705", scaled)
        self.assertIn("she gives the key", scaled)
        self.assertIn("he unlocks a cabinet", scaled)

    def test_thirty_seconds_uses_event_aware_cost_optimal_bounded_windows(self) -> None:
        plan = plan_long_horizon(
            requested_duration_seconds=30,
            prompt=PROMPT,
            seed=82341,
        )
        self.assertEqual(plan.output_frames, 719)
        self.assertEqual([item.window_frames for item in plan.segments], [328, 277, 192])
        self.assertEqual([item.context_frames for item in plan.segments], [0, 39, 39])
        self.assertEqual([item.visible_frames for item in plan.segments], [328, 238, 153])
        self.assertEqual(plan.segments[-1].visible_stop_frame, 719)
        self.assertEqual(len({item.seed for item in plan.segments}), 3)
        self.assertEqual(plan.planning_policy, "event_aware_continuous_cost_v1")
        # Neither internal boundary is allowed to collide with the authored
        # 10s/20s shot transitions.
        self.assertGreater(min(abs(328 - 240), abs(328 - 480)), 24)
        self.assertGreater(min(abs(566 - 240), abs(566 - 480)), 24)

    def test_sixteen_seconds_avoids_a_tiny_tail_pass(self) -> None:
        plan = plan_long_horizon(
            requested_duration_seconds=16,
            prompt="one continuous shot",
            seed=1,
        )
        self.assertEqual(plan.output_frames, 379)
        self.assertEqual([item.window_frames for item in plan.segments], [209, 209])

    def test_public_sixty_second_limit_is_planned_in_bounded_time(self) -> None:
        started = time.perf_counter()
        plan = plan_long_horizon(
            requested_duration_seconds=60,
            prompt=PROMPT,
            seed=7,
        )
        elapsed = time.perf_counter() - started
        self.assertEqual(plan.output_frames, 1433)
        self.assertEqual(plan.segments[-1].visible_stop_frame, plan.output_frames)
        self.assertTrue(all(item.window_frames <= 362 for item in plan.segments))
        # This is an algorithmic guard, not a hardware benchmark.  The DAG has
        # fewer than ten thousand legal edges at the 60 s API limit.
        self.assertLess(elapsed, 1.0)

    def test_shot_decode_groups_keep_cut_preroll_outside_visible_clock(self) -> None:
        from dataclasses import replace

        plan = plan_long_horizon(
            requested_duration_seconds=30,
            prompt=PROMPT,
            seed=7,
            context_frames=39,
        )
        segments = tuple(
            replace(
                segment,
                transition=(
                    "opening" if index == 0 else "cut" if index == 1 else "continue"
                ),
                video_prefix_frames=(None if index == 0 else 0 if index == 1 else 39),
            )
            for index, segment in enumerate(plan.segments)
        )
        groups = plan_shot_decode_groups(segments)
        self.assertEqual([group.segment_indices for group in groups], [(0,), (1, 2)])
        self.assertEqual([group.lead_context_frames for group in groups], [0, 39])
        self.assertEqual(
            [group.visible_frames for group in groups],
            [segments[0].visible_frames, segments[1].visible_frames + segments[2].visible_frames],
        )
        self.assertEqual(
            groups[1].physical_frames,
            39 + segments[1].visible_frames + segments[2].visible_frames,
        )
        self.assertEqual(
            sum(group.visible_frames for group in groups),
            plan.output_frames,
        )

    def test_shot_decode_groups_keep_opening_preroll_outside_visible_clock(self) -> None:
        from dataclasses import replace

        plan = plan_long_horizon(
            requested_duration_seconds=15,
            prompt=PROMPT,
            seed=7,
            context_frames=39,
        )
        opening = replace(
            plan.segments[0],
            window_frames=plan.segments[0].window_frames + 17,
            opening_preroll_frames=17,
        )
        groups = plan_shot_decode_groups((opening,))
        self.assertEqual(groups[0].lead_context_frames, 17)
        self.assertEqual(groups[0].physical_frames, opening.window_frames)
        self.assertEqual(groups[0].visible_frames, opening.visible_frames)

    def test_timed_h3_prompt_is_localized_without_user_rewrite(self) -> None:
        localized = localize_h3_prompt(
            PROMPT,
            context_start_frame=340,
            visible_start_frame=362,
            visible_stop_frame=481,
            segment_index=1,
        )
        self.assertIn("Continue directly", localized)
        self.assertIn("she gives the key", localized)
        # The 481-frame H3 grid reaches 20.041 s, so the event beginning at
        # global 20.000 s is correctly admitted at the very end of this window.
        self.assertIn("At 00:05.833", localized)
        self.assertIn("overall_soundscape", localized)
        self.assertIn("Continuation already in progress", localized)
        self.assertNotIn("camera cuts", localized)

    def test_continuation_drops_completed_actions_instead_of_replaying_them(self) -> None:
        prompt = """integrated_multimodal_description: [Shot 1] A woman leaves the room. [Shot 2] At 00:10.000, the camera cuts to a robot beside a table. The robot approaches the table. Its brush pulls a napkin. A doughnut falls onto the robot. [Shot 3] At 00:20.000, the camera cuts back to the room. The robot crosses the floor. The woman returns through the door. She notices the doughnut and speaks. The robot stops. The final frame holds.

overall_soundscape: Quiet room tone.

non_diegetic_music: N/A"""
        plan = plan_long_horizon(
            requested_duration_seconds=30,
            prompt=prompt,
            seed=82901,
        )
        ending = plan.segments[-1].prompt
        self.assertIn("earlier clauses of this shot are already completed", ending)
        self.assertNotIn("The robot crosses the floor", ending)
        self.assertNotIn("The woman returns through the door", ending)
        self.assertNotIn("She notices the doughnut and speaks", ending)
        self.assertIn("The robot stops", ending)
        self.assertIn("The final frame holds", ending)

    def test_plain_language_time_ranges_are_windowed_and_retimed(self) -> None:
        prompt = (
            "integrated_multimodal_description: "
            "第1秒到第10秒，女人摆好杯子。"
            "第10秒到第20秒，男人点燃蜡烛。"
            "第20秒到第30秒，两人一起大笑。\n\n"
            "overall_soundscape: Quiet room tone.\n\n"
            "non_diegetic_music: N/A"
        )
        plan = plan_long_horizon(
            requested_duration_seconds=30,
            prompt=prompt,
            seed=9,
        )
        opening = plan.segments[0].prompt
        middle = plan.segments[1].prompt
        ending = plan.segments[-1].prompt
        self.assertIn("女人摆好杯子", opening)
        self.assertIn("男人点燃蜡烛", opening)
        self.assertNotIn("两人一起大笑", opening)
        self.assertNotIn("女人摆好杯子", middle)
        self.assertIn("男人点燃蜡烛", middle)
        self.assertIn("两人一起大笑", middle)
        self.assertNotIn("女人摆好杯子", ending)
        self.assertIn("两人一起大笑", ending)
        self.assertIn("[Local interval", ending)
        self.assertIn("overall_soundscape", ending)

    def test_ref2va_six_section_prompt_keeps_reference_contract_while_windowing_shots(self) -> None:
        prompt = """subject_definitions: <Subject 1> is the navy backpack from <Picture 1>.\n<Audio 1> is the voice-timbre reference for <Subject 2> (S1).\n\nsummary: [reference generation + audio reference] The target keeps <Subject 1> visible and uses <Audio 1> only as a timbre reference.\n\nretention_analysis: <Subject 1> (appears in [Shot 1], [Shot 3]): fully_preserved - its navy fabric and black trim remain unchanged.\n<Audio 1>: reference - S1 follows its timbre without copying the source signal.\n\ndetailed_description: Live-action with steady natural lighting. [Shot 1] A girl places the navy backpack on a bench and speaks. [Shot 2] At 00:10.000, the camera cuts to a close-up of her hands closing the zipper. [Shot 3] At 00:20.000, the camera cuts back to the room and the same girl lifts the same backpack.\n\noverall_soundscape: Quiet room tone and a zipper.\n\nnon_diegetic_music: N/A"""
        plan = plan_long_horizon(
            requested_duration_seconds=30,
            prompt=prompt,
            seed=82901,
        )
        ending = plan.segments[-1].prompt
        self.assertIn("subject_definitions:", ending)
        self.assertIn("summary: [reference generation + audio reference]", ending)
        self.assertIn("retention_analysis:", ending)
        self.assertIn("detailed_description:", ending)
        self.assertIn("<Subject 1>", ending)
        self.assertIn("<Audio 1>", ending)
        self.assertIn("the same girl lifts the same backpack", ending)
        self.assertNotIn("A girl places the navy backpack", ending)
        self.assertIn("overall_soundscape:", ending)
        self.assertIn("non_diegetic_music:", ending)

    def test_masked_joint_av_prefix_is_exactly_restored(self) -> None:
        source_video = torch.arange(1 * 24 * 37 * 2 * 3, dtype=torch.float32).reshape(1, 24, 37, 2, 3)
        source_audio = torch.arange(1 * 32 * 2 * 207, dtype=torch.float32).reshape(1, 32, 2, 207)
        noise_video = torch.randn(1, 24, 47, 2, 3)
        noise_audio = torch.randn(1, 32, 2, 263)
        video, audio, video_prefix, audio_prefix = prepare_masked_av_prefix(
            noise_video, noise_audio, source_video, source_audio
        )
        self.assertTrue(torch.equal(video[:, :, :12], source_video[:, :, -12:]))
        self.assertTrue(torch.equal(audio[..., :65], source_audio[..., -65:]))
        video[:, :, :12].zero_()
        audio[..., :65].zero_()
        restore_masked_av_prefix_(video, audio, video_prefix, audio_prefix)
        self.assertTrue(torch.equal(video[:, :, :12], video_prefix))
        self.assertTrue(torch.equal(audio[..., :65], audio_prefix))

    def test_partial_video_prefix_keeps_its_true_position_in_full_context(self) -> None:
        source_video = torch.arange(
            1 * 2 * 37 * 1 * 1, dtype=torch.float32
        ).reshape(1, 2, 37, 1, 1)
        source_audio = torch.zeros(1, 2, 1, 207)
        noise_video = torch.full((1, 2, 47, 1, 1), -1.0)
        noise_audio = torch.full((1, 2, 1, 263), -2.0)

        video, _, video_prefix, _ = prepare_masked_av_prefix(
            noise_video,
            noise_audio,
            source_video,
            source_audio,
            context_frames=90,
            video_prefix_frames=73,
        )

        # Ninety source frames occupy 27 video tokens. The exact 73-frame
        # anchor occupies their leading 22 tokens; the trailing five context
        # tokens remain noise for hidden repaint. Taking source[-22:] here
        # would shift recent history 17 frames earlier in target time.
        expected = source_video[:, :, -27:-5]
        self.assertTrue(torch.equal(video_prefix, expected))
        self.assertTrue(torch.equal(video[:, :, :22], expected))
        self.assertTrue(torch.equal(video[:, :, 22:27], noise_video[:, :, 22:27]))

    def test_terminal_video_seed_is_rebased_to_local_zero(self) -> None:
        source_video = torch.arange(
            1 * 2 * 37 * 1 * 1, dtype=torch.float32
        ).reshape(1, 2, 37, 1, 1)
        source_audio = torch.zeros(1, 2, 1, 207)
        noise_video = torch.full((1, 2, 47, 1, 1), -1.0)
        noise_audio = torch.full((1, 2, 1, 263), -2.0)

        video, _, video_prefix, _ = prepare_masked_av_prefix(
            noise_video,
            noise_audio,
            source_video,
            source_audio,
            context_frames=90,
            video_prefix_frames=5,
            video_prefix_from_source_end=True,
            audio_bridge_ticks=150,
        )

        self.assertTrue(torch.equal(video_prefix, source_video[:, :, -2:]))
        self.assertTrue(torch.equal(video[:, :, :2], source_video[:, :, -2:]))
        self.assertTrue(torch.equal(video[:, :, 2:27], noise_video[:, :, 2:27]))

    def test_audio_bridge_repaints_only_the_trailing_context_band(self) -> None:
        source_video = torch.full((1, 24, 37, 2, 3), 3.0)
        source_audio = torch.arange(
            1 * 32 * 2 * 207, dtype=torch.float32
        ).reshape(1, 32, 2, 207)
        noise_video = torch.full((1, 24, 47, 2, 3), -1.0)
        noise_audio = torch.full((1, 32, 2, 263), -2.0)
        video, audio, video_prefix, audio_prefix = prepare_masked_av_prefix(
            noise_video,
            noise_audio,
            source_video,
            source_audio,
            context_frames=56,
            audio_bridge_ticks=65,
        )
        source_context = source_audio[..., -93:]
        self.assertTrue(torch.equal(video[:, :, :17], source_video[:, :, -17:]))
        self.assertEqual(audio_prefix.shape[-1], 28)
        self.assertTrue(torch.equal(audio[..., :28], source_context[..., :28]))
        self.assertTrue(torch.equal(audio[..., 28:93], noise_audio[..., 28:93]))
        video.zero_()
        audio.zero_()
        restore_masked_av_prefix_(video, audio, video_prefix, audio_prefix)
        self.assertTrue(torch.equal(video[:, :, :17], video_prefix))
        self.assertTrue(torch.equal(audio[..., :28], audio_prefix))
        self.assertTrue(torch.equal(
            audio[..., 28:93], torch.zeros_like(audio[..., 28:93])
        ))

    def test_hard_cut_keeps_audio_context_without_a_video_prefix(self) -> None:
        source_video = torch.full((1, 24, 37, 2, 3), 3.0)
        source_audio = torch.full((1, 32, 2, 207), 4.0)
        noise_video = torch.full((1, 24, 47, 2, 3), -1.0)
        noise_audio = torch.full((1, 32, 2, 263), -2.0)
        video, audio, video_prefix, audio_prefix = prepare_masked_av_prefix(
            noise_video,
            noise_audio,
            source_video,
            source_audio,
            context_frames=39,
            video_prefix_frames=0,
            audio_bridge_ticks=0,
        )
        self.assertEqual(video_prefix.shape[2], 0)
        self.assertTrue(torch.equal(video[:, :, :12], noise_video[:, :, :12]))
        self.assertEqual(audio_prefix.shape[-1], 65)
        self.assertTrue(torch.equal(audio[..., :65], source_audio[..., -65:]))

    def test_full_context_audio_overlap_save_preserves_video_geometry(self) -> None:
        source_video = torch.full((1, 24, 37, 2, 3), 3.0)
        source_audio = torch.full((1, 32, 2, 207), 4.0)
        noise_video = torch.full((1, 24, 47, 2, 3), -1.0)
        noise_audio = torch.full((1, 32, 2, 263), -2.0)
        video, audio, video_prefix, audio_prefix = prepare_masked_av_prefix(
            noise_video,
            noise_audio,
            source_video,
            source_audio,
            context_frames=39,
            audio_bridge_ticks=65,
        )
        self.assertEqual(video_prefix.shape[2], 12)
        self.assertTrue(torch.equal(video[:, :, :12], source_video[:, :, -12:]))
        self.assertEqual(audio_prefix.shape[-1], 0)
        self.assertTrue(torch.equal(audio[..., :65], noise_audio[..., :65]))
        video.zero_()
        audio.zero_()
        restore_masked_av_prefix_(video, audio, video_prefix, audio_prefix)
        self.assertTrue(torch.equal(video[:, :, :12], video_prefix))
        self.assertTrue(torch.equal(audio[..., :65], torch.zeros_like(audio[..., :65])))

    def test_stitch_concatenates_once_and_repairs_only_terminal_audio_rounding(self) -> None:
        documents = [
            {
                "video": torch.zeros(1, 24, 107, 2, 3),
                "audio": torch.zeros(1, 32, 2, 603),
                "frames": 362,
            }
        ]
        for value in (1.0, 2.0, 3.0):
            documents.append({
                "video": torch.full((1, 24, 47, 2, 3), value),
                "audio": torch.full((1, 32, 2, 263), value),
                "frames": 158,
            })
        video, audio, frames = stitch_clean_av_segments(documents, (0, 39, 39, 39))
        self.assertEqual(frames, 719)
        self.assertEqual(video.shape[2], 212)
        self.assertEqual(audio.shape[-1], 1198)
        self.assertEqual(float(video[:, :, 107:142].mean()), 1.0)
        self.assertEqual(float(video[:, :, 142:177].mean()), 2.0)
        self.assertEqual(float(video[:, :, 177:].mean()), 3.0)
        with tempfile.TemporaryDirectory() as temporary_root:
            paths = []
            for index, document in enumerate(documents):
                path = Path(temporary_root) / f"segment-{index}.pt"
                torch.save({**document, "engine": "test-engine"}, path)
                paths.append(path)
            streamed_video, streamed_audio, streamed_frames, engine = (
                stitch_clean_av_segment_files(
                    paths,
                    (0, 39, 39, 39),
                    expected_frames=719,
                )
            )
        self.assertTrue(torch.equal(streamed_video, video))
        self.assertTrue(torch.equal(streamed_audio, audio))
        self.assertEqual(streamed_frames, frames)
        self.assertEqual(engine, "test-engine")

    def test_opening_preroll_is_cropped_from_global_latent_clock(self) -> None:
        first = {
            "video": torch.arange(57, dtype=torch.float32).reshape(1, 1, 57, 1, 1),
            "audio": torch.arange(320, dtype=torch.float32).reshape(1, 1, 1, 320),
            "frames": 192,
        }
        second = {
            "video": (1000 + torch.arange(82, dtype=torch.float32)).reshape(1, 1, 82, 1, 1),
            "audio": torch.ones(1, 1, 1, 462),
            "frames": 277,
        }
        video, audio, frames = stitch_clean_av_segments(
            (first, second),
            (0, 90),
            leading_preroll_frames=17,
        )
        self.assertEqual(frames, 362)
        self.assertEqual(video.shape[2], 107)
        self.assertEqual(audio.shape[-1], 603)
        self.assertTrue(torch.equal(video[:, :, :52], first["video"][:, :, -52:]))

        with tempfile.TemporaryDirectory(prefix="h3-opening-preroll-") as root:
            paths = []
            for index, document in enumerate((first, second)):
                path = Path(root) / f"segment-{index}.pt"
                torch.save({**document, "engine": "test-engine"}, path)
                paths.append(path)
            streamed, streamed_audio, streamed_frames, engine = (
                stitch_clean_av_segment_files(
                    paths,
                    (0, 90),
                    expected_frames=362,
                    leading_preroll_frames=17,
                )
            )
        self.assertTrue(torch.equal(streamed, video))
        self.assertTrue(torch.equal(streamed_audio, audio))
        self.assertEqual(streamed_frames, frames)
        self.assertEqual(engine, "test-engine")

    def test_video_repaint_replaces_accepted_same_time_tail_and_matches_streaming(self) -> None:
        first_video = torch.arange(52, dtype=torch.float32).reshape(
            1, 1, 52, 1, 1
        )
        second_video = (
            1000
            + torch.arange(82, dtype=torch.float32)
        ).reshape(1, 1, 82, 1, 1)
        first = {
            "video": first_video,
            "audio": torch.zeros(1, 1, 1, 292),
            "frames": 175,
        }
        second = {
            "video": second_video,
            "audio": torch.ones(1, 1, 1, 462),
            "frames": 277,
        }

        video, _, frames = stitch_clean_av_segments(
            (first, second),
            (0, 90),
            video_repaint_frames=(0, 17),
        )
        self.assertEqual(frames, 362)
        self.assertEqual(video.shape[2], 107)
        self.assertTrue(torch.equal(video[:, :, :47], first_video[:, :, :47]))
        self.assertTrue(torch.equal(video[:, :, 47:], second_video[:, :, 22:]))

        with tempfile.TemporaryDirectory(prefix="h3-video-repaint-") as root:
            paths = []
            for index, document in enumerate((first, second)):
                path = Path(root) / f"segment-{index}.pt"
                torch.save({**document, "engine": "test-engine"}, path)
                paths.append(path)
            streamed, _, streamed_frames, engine = stitch_clean_av_segment_files(
                paths,
                (0, 90),
                expected_frames=362,
                video_repaint_frames=(0, 17),
            )
        self.assertTrue(torch.equal(streamed, video))
        self.assertEqual(streamed_frames, frames)
        self.assertEqual(engine, "test-engine")

    def test_audio_bridge_overlap_save_discards_repaint_and_matches_streaming(self) -> None:
        first = {
            "video": torch.zeros(1, 24, 37, 2, 3),
            "audio": torch.full((1, 32, 2, 207), 3.0),
            "frames": 124,
        }
        second = {
            "video": torch.ones(1, 24, 37, 2, 3),
            "audio": torch.full((1, 32, 2, 207), 2.0),
            "frames": 124,
        }
        video, audio, frames = stitch_clean_av_segments(
            (first, second),
            (0, 39),
            (0, 20),
        )
        self.assertEqual(frames, 209)
        self.assertEqual(video.shape[2], 62)
        self.assertEqual(audio.shape[-1], 348)
        bridge = audio[..., 187:207]
        # The incoming repaint is useful inside its DiT window, but it must
        # never replace or mix with the already accepted predecessor latent.
        self.assertTrue(torch.equal(bridge, torch.full_like(bridge, 3.0)))
        self.assertEqual(float(audio[..., 207:].mean()), 2.0)
        with tempfile.TemporaryDirectory() as temporary_root:
            paths = []
            for index, document in enumerate((first, second)):
                path = Path(temporary_root) / f"bridge-segment-{index}.pt"
                torch.save({**document, "engine": "test-engine"}, path)
                paths.append(path)
            streamed_video, streamed_audio, streamed_frames, _ = (
                stitch_clean_av_segment_files(
                    paths,
                    (0, 39),
                    expected_frames=209,
                    audio_bridge_ticks=(0, 20),
                )
            )
        self.assertTrue(torch.equal(streamed_video, video))
        self.assertTrue(torch.equal(streamed_audio, audio))
        self.assertEqual(streamed_frames, frames)

    def test_audio_only_stitch_preserves_named_formal_sampler_state(self) -> None:
        first = {
            "video": torch.zeros(1, 1, 37, 1, 1),
            "audio": torch.zeros(1, 1, 1, 207),
            "audio_state": torch.full((1, 1, 1, 207), 6.0),
            "frames": 124,
        }
        second = {
            "video": torch.ones(1, 1, 37, 1, 1),
            "audio": torch.ones(1, 1, 1, 207),
            "audio_state": torch.full((1, 1, 1, 207), 7.0),
            "frames": 124,
        }
        with tempfile.TemporaryDirectory(prefix="h3-audio-state-stitch-") as root:
            paths = []
            for index, document in enumerate((first, second)):
                path = Path(root) / f"segment-{index}.pt"
                torch.save(document, path)
                paths.append(path)
            stitched = stitch_audio_segment_files(
                paths,
                (0, 39),
                expected_frames=209,
                audio_key="audio_state",
            )
        self.assertEqual(stitched.shape[-1], 348)
        self.assertTrue(torch.equal(
            stitched[..., :207],
            torch.full_like(stitched[..., :207], 6.0),
        ))
        self.assertTrue(torch.equal(
            stitched[..., 207:],
            torch.full_like(stitched[..., 207:], 7.0),
        ))


if __name__ == "__main__":
    unittest.main()
