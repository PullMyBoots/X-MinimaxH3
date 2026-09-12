from __future__ import annotations

import unittest

from h3serve.native_engine.long_horizon import (
    _first_director_cut_bridge_frame,
    _prompt_event_frames,
    has_structured_timeline,
    localize_h3_prompt,
    plan_long_horizon,
    scale_h3_prompt_timeline,
)


class LongHorizonPromptLocalizationTests(unittest.TestCase):
    PROMPT = """subject_definitions: <Subject 1> is the person in <Picture 1>.

summary: [reference generation + audio reference] The person starts in a wide view, disappears during a close-up, and returns to the wide view at the end.

retention_analysis: <Subject 1> (appears in [Shot 1], [Shot 2], [Shot 3]): fully_preserved - identity remains unchanged.

detailed_description: One stable room remains throughout.
[Shot 1] A wide view establishes the person and table. At about 00:04.500, <Subject 1> (S1) says: <d>[Chinese] 第一行。</d>
[Shot 2] At 00:16.000, the camera cuts to a steady overhead close-up of the same object and two hands; the face remains outside the frame. At about 00:19.500, <Subject 1> (S1) says: <d>[Chinese] 已完成台词。</d> The right hand turns the key once. The same object remains centered and still.
[Shot 3] At 00:24.000, the camera cuts back to the established wide view. The same person remains at the same table. At about 00:27.000, <Subject 1> (S1) says: <d>[Chinese] 最后一行。</d>

overall_soundscape: Quiet room tone.
non_diegetic_music: N/A
"""

    def test_continuation_keeps_active_framing_anchor_and_retimes_clocks(self) -> None:
        localized = localize_h3_prompt(
            self.PROMPT,
            context_start_frame=442,
            visible_start_frame=481,
            visible_stop_frame=719,
            segment_index=2,
            timeline_stop_frame=719,
        )

        self.assertIn("Ongoing-shot state anchor", localized)
        self.assertIn(
            "the established camera framing remains a steady overhead close-up",
            localized,
        )
        self.assertNotIn("已完成台词", localized)
        self.assertNotIn("turns the key once", localized)
        self.assertIn("At 00:05.583, the camera cuts back", localized)
        self.assertIn("At 00:08.583, <Subject 1> (S1) says", localized)
        self.assertIn("retention_analysis:", localized)
        self.assertIn("Generate only the window-local continuation", localized)
        self.assertNotIn("returns to the wide view at the end", localized)
        self.assertNotIn("appears in [Shot 1]", localized)

    def test_partition_events_are_shot_starts_not_dialogue_cues(self) -> None:
        self.assertEqual(
            _prompt_event_frames(self.PROMPT, 719),
            (384, 576),
        )

    def test_timed_dialogue_has_exactly_one_visible_window_owner(self) -> None:
        prompt = """integrated_multimodal_description:
[Shot 1] A girl stands beside a table. At 00:06.000, she says: <d>[Chinese] 第一行。</d>
[Shot 2] At 00:10.000, the camera cuts to an overhead view. Her hands pack a notebook.
[Shot 3] At 00:20.000, the camera cuts back to the same girl. She lifts the bag. Her father adjusts one strap. At about 00:26.000, the same girl says: <d>[Chinese] 我会记得把东西都带回来的。</d> No other voice occurs. The final composition holds.

overall_soundscape: Quiet room tone and only the specified dialogue.
non_diegetic_music: N/A
"""
        plan = plan_long_horizon(
            requested_duration_seconds=30,
            prompt=prompt,
            seed=82901,
        )
        self.assertEqual(
            [segment.visible_start_frame for segment in plan.segments],
            [0, 328, 566],
        )
        prompts = [segment.prompt for segment in plan.segments]
        first_line = "第一行"
        final_line = "我会记得把东西都带回来的"
        self.assertEqual(sum(item.count(first_line) for item in prompts), 1)
        self.assertEqual(sum(item.count(final_line) for item in prompts), 1)
        self.assertNotIn(final_line, prompts[1])
        self.assertIn(final_line, prompts[2])
        self.assertIn("At 00:04.042, the same girl says", prompts[2])
        self.assertIn("No other voice occurs", prompts[2])
        self.assertIn("The final composition holds", prompts[2])

    def test_minute_clock_is_not_counted_twice(self) -> None:
        prompt = (
            "integrated_multimodal_description: [Shot 1] A clock ticks. "
            "At 01:02.000, a bell rings."
        )
        scaled = scale_h3_prompt_timeline(
            prompt,
            source_frames=1488,
            target_frames=744,
        )
        self.assertIn("At 00:31.000", scaled)

    def test_first_cut_bridge_uses_only_clocks_and_crosses_the_cut(self) -> None:
        prompt = """integrated_multimodal_description:
[Shot 1] One stable view.
[Shot 2] At 00:07.500, the camera cuts to one side view. The cup moves. At 00:12.000, S1 says: <d>[Chinese] 不会泄漏。</d>
[Shot 3] At 00:15.500, the camera cuts to a close view.
"""
        self.assertEqual(_first_director_cut_bridge_frame(prompt, 719), 226)

    def test_structured_director_aligns_windows_and_compiles_local_contract(self) -> None:
        prompt = """integrated_multimodal_description:
[Shot 1] A stable wide shot. At 00:04.500, the girl speaks once: <d>[Chinese] 第一行。</d>
[Shot 2] At 00:08.000, the camera cuts to a medium shot. The girl turns one gear.
[Shot 3] At 00:16.000, the camera cuts to an overhead shot. The hands turn one key.
[Shot 4] At 00:24.000, the camera cuts back to the stable wide shot. At 00:27.000, the girl speaks once: <d>[Chinese] 最后一行。</d>

overall_soundscape: Quiet room tone.
non_diegetic_music: N/A
"""
        self.assertTrue(has_structured_timeline(prompt))
        plan = plan_long_horizon(
            requested_duration_seconds=30,
            prompt=prompt,
            seed=82901,
            maximum_opening_frames=277,
            structured_director=True,
        )
        self.assertTrue(plan.structured_director)
        self.assertEqual(
            [segment.visible_start_frame for segment in plan.segments],
            [0, 158, 345, 532],
        )
        self.assertEqual(
            [segment.visible_frames for segment in plan.segments],
            [158, 187, 187, 187],
        )
        opening = plan.segments[0].prompt
        second = plan.segments[1].prompt
        closing = plan.segments[-1].prompt
        self.assertIn("structured_director_contract:", opening)
        self.assertIn("No new camera cut is permitted in the writable suffix", opening)
        self.assertIn(
            "only new camera cut(s) permitted in the writable suffix are at local 00:03.042",
            second,
        )
        self.assertIn(
            "only new camera cut(s) permitted in the writable suffix are at local 00:03.458",
            closing,
        )
        self.assertIn("Shot 4", closing)
        self.assertNotIn("the camera cuts to an overhead shot", closing)
        self.assertIn(
            "A timestamp inside a shot changes only its stated action or sound",
            closing,
        )
        self.assertIn("At 00:06.458, the girl speaks once", closing)
        self.assertEqual(
            plan.planning_policy,
            "structured_director_first_cut_bridge_causal_state_v7",
        )
        self.assertTrue(all(
            segment.visual_memory_floor_frame is None
            for segment in plan.segments
        ))
        self.assertFalse(plan.segments[-2].reference_audio_active)
        self.assertTrue(plan.segments[-1].reference_audio_active)
        self.assertEqual(
            [segment.authorized_dialogue_count for segment in plan.segments],
            [1, 0, 0, 1],
        )
        self.assertEqual(
            [segment.authorized_dialogue_frames for segment in plan.segments],
            [(108,), (), (), (648,)],
        )
        self.assertEqual(
            [segment.audio_memory_active for segment in plan.segments],
            [True, False, False, True],
        )
        self.assertNotIn("bootstrap_lookahead_contract:", opening)
        self.assertNotIn("The girl turns one gear", opening)
        self.assertNotIn("bootstrap_lookahead_contract:", second)
        self.assertIn("Do not render dialogue as subtitles", opening)
        silent_transition = plan.segments[-2].prompt
        self.assertIn(
            "no authorized H3 dialogue event",
            silent_transition,
        )
        self.assertNotIn("<d>", silent_transition)
        self.assertIn(
            "overall_soundscape: Quiet room tone.",
            silent_transition,
        )
        self.assertIn(
            "exactly 1 authorized H3 dialogue event",
            closing,
        )

    def test_structured_director_compiles_one_declared_shot_as_single_take(self) -> None:
        prompt = """integrated_multimodal_description:
[Shot 1] One unbroken tracking shot follows a stage manager through one connected theatre. At 00:04.500, S1 says: <d>[Chinese] 检查开始。</d> At 00:16.500, she walks onto the stage without a cut. At 00:27.000, S1 says: <d>[Chinese] 检查完成。</d>

overall_soundscape: Quiet theatre room tone.
non_diegetic_music: N/A
"""
        self.assertTrue(has_structured_timeline(prompt))
        self.assertEqual(_prompt_event_frames(prompt, 719), ())
        plan = plan_long_horizon(
            requested_duration_seconds=30,
            prompt=prompt,
            seed=95001,
            maximum_opening_frames=209,
            structured_director=True,
        )
        self.assertTrue(plan.structured_director)
        self.assertEqual(
            plan.planning_policy,
            "structured_director_single_take_dual_timescale_isolated_latch_v9_56f",
        )
        self.assertEqual(plan.context_frames, 56)
        self.assertEqual(
            [segment.visible_start_frame for segment in plan.segments],
            [0, 209, 345, 481, 600],
        )
        for segment in plan.segments:
            self.assertIn("structured_director_contract:", segment.prompt)
            self.assertIn(
                "No new camera cut is permitted in the writable suffix",
                segment.prompt,
            )
            self.assertIn("Shot 1", segment.prompt)
        self.assertEqual(
            [segment.visual_memory_floor_frame for segment in plan.segments],
            [None, 153, 289, 425, 544],
        )
        self.assertEqual(
            [segment.visual_memory_include_canonical for segment in plan.segments],
            [False, True, True, True, True],
        )
        self.assertTrue(all(
            segment.preserve_latest_visual for segment in plan.segments
        ))
        self.assertIn(
            "Ongoing timed-event continuation",
            plan.segments[-1].prompt,
        )
        self.assertIn(
            "she walks onto the stage without a cut",
            plan.segments[-1].prompt,
        )
        self.assertEqual(
            [segment.authorized_dialogue_count for segment in plan.segments],
            [1, 0, 0, 0, 1],
        )
        self.assertEqual(
            sum(segment.prompt.count("检查开始") for segment in plan.segments),
            1,
        )
        self.assertEqual(
            sum(segment.prompt.count("检查完成") for segment in plan.segments),
            1,
        )

    def test_single_take_carries_an_action_that_straddles_a_window(self) -> None:
        prompt = """integrated_multimodal_description:
[Shot 1] One unbroken tracking shot follows one potter. At 00:04.500, S1 says: <d>[Chinese] 开始检查。</d> He closes his lips. At 00:11.500, he carries one bowl down the aisle while the camera tracks beside him. At 00:17.000, he places that bowl on the shelf without a cut. At 00:27.000, S1 says: <d>[Chinese] 已经放好。</d>

overall_soundscape: Quiet room tone.
non_diegetic_music: N/A
"""
        plan = plan_long_horizon(
            requested_duration_seconds=30,
            prompt=prompt,
            seed=95003,
            maximum_opening_frames=209,
            structured_director=True,
        )
        crossing = plan.segments[2]
        self.assertEqual(crossing.visible_start_frame, 345)
        self.assertIn("Ongoing timed-event continuation", crossing.prompt)
        self.assertIn("carries one bowl down the aisle", crossing.prompt)
        self.assertIn("camera tracks beside him", crossing.prompt)
        self.assertIn("At 00:04.958, he places that bowl", crossing.prompt)
        self.assertNotIn("开始检查", crossing.prompt)
        self.assertEqual(crossing.authorized_dialogue_count, 0)

    def test_event_onset_inside_exact_prefix_is_latched_not_replayed(self) -> None:
        prompt = """integrated_multimodal_description:
[Shot 1] One unbroken tracking shot follows one craftsperson. At 00:19.000, he presses one fixed switch once; one warm pool of light appears across the shelf while the camera continues one shallow arc. At 00:25.500, he takes two steps backward with empty hands.

overall_soundscape: Quiet room tone.
non_diegetic_music: N/A
"""
        localized = localize_h3_prompt(
            prompt,
            context_start_frame=425,
            visible_start_frame=481,
            visible_stop_frame=719,
            segment_index=3,
            timeline_stop_frame=719,
            structured_director=True,
        )
        self.assertIn("Context-latched timed event", localized)
        self.assertNotIn("presses one fixed switch", localized)
        self.assertIn("one warm pool of light appears", localized)
        self.assertIn("camera continues one shallow arc", localized)
        self.assertIn("At 00:07.792, he takes two steps backward", localized)

    def test_indivisible_motion_inside_prefix_is_retained_conservatively(self) -> None:
        prompt = """integrated_multimodal_description:
[Shot 1] One continuous tracking shot. At 00:19.000, he carries one box down the connected aisle while the camera tracks beside him. At 00:25.500, he sets the box on one table.

overall_soundscape: Quiet footsteps.
non_diegetic_music: N/A
"""
        localized = localize_h3_prompt(
            prompt,
            context_start_frame=425,
            visible_start_frame=481,
            visible_stop_frame=719,
            segment_index=3,
            timeline_stop_frame=719,
            structured_director=True,
        )
        self.assertIn("Ongoing timed-event continuation", localized)
        self.assertIn(
            "this event began in the exact carried prefix and is still underway",
            localized,
        )
        self.assertIn("carries one box down the connected aisle", localized)
        self.assertNotIn("Context-latched timed event", localized)

    def test_active_dialogue_event_carries_only_its_post_speech_motion(self) -> None:
        prompt = """integrated_multimodal_description:
[Shot 1] One continuous tracking shot. At 00:06.000, S1 says: <d>[Chinese] 沿着走廊走。</d> He closes his lips and keeps walking forward while the camera tracks beside him. At 00:16.500, he opens one door without a cut. At 00:27.000, S1 says: <d>[Chinese] 到了。</d>

overall_soundscape: Quiet footsteps.
non_diegetic_music: N/A
"""
        plan = plan_long_horizon(
            requested_duration_seconds=30,
            prompt=prompt,
            seed=95005,
            maximum_opening_frames=209,
            structured_director=True,
        )
        crossing = plan.segments[1]
        self.assertIn("Ongoing timed-event continuation", crossing.prompt)
        self.assertIn("keeps walking forward", crossing.prompt)
        self.assertNotIn("沿着走廊走", crossing.prompt)
        self.assertNotIn("<d>", crossing.prompt)
        self.assertEqual(crossing.authorized_dialogue_count, 0)

    def test_inline_clock_keeps_its_shared_subject_after_localization(self) -> None:
        prompt = """integrated_multimodal_description:
[Shot 1] A stable medium shot. She places one cup on the table and at about 00:04.500 says: <d>[Chinese] 已经放好。</d> She closes her lips.

overall_soundscape: Quiet room tone.
non_diegetic_music: N/A
"""
        localized = localize_h3_prompt(
            prompt,
            context_start_frame=0,
            visible_start_frame=0,
            visible_stop_frame=209,
            segment_index=0,
            timeline_stop_frame=209,
            structured_director=True,
        )
        self.assertIn(
            "She places one cup on the table and at 00:04.500 says",
            localized,
        )
        self.assertNotIn("and At", localized)

    def test_dialogue_authority_uses_tags_not_story_words_or_language(self) -> None:
        prompt = """integrated_multimodal_description:
[Shot 1] A narrator discusses voice, speech, and dialogue metadata without an H3 dialogue span.
[Shot 2] At 00:08.000, the camera cuts to another room. At 00:11.000, S1 says: <d>[English] ready.</d>
[Shot 3] At 00:16.000, the camera cuts outside. The word voice appears again but nobody has an authorized utterance.
[Shot 4] At 00:24.000, the camera returns. At 00:27.000, S1 says: <d>[Japanese] 帰りました。</d>

overall_soundscape: The prose words speech, voice, and dialogue are metadata.
non_diegetic_music: N/A
"""
        plan = plan_long_horizon(
            requested_duration_seconds=30,
            prompt=prompt,
            seed=17,
            maximum_opening_frames=277,
            structured_director=True,
        )
        counts = [
            segment.authorized_dialogue_count for segment in plan.segments
        ]
        self.assertEqual(counts, [0, 1, 0, 1])
        self.assertEqual(
            [segment.authorized_dialogue_frames for segment in plan.segments],
            [(), (264,), (), (648,)],
        )
        self.assertEqual(
            [segment.reference_audio_active for segment in plan.segments],
            [False, True, False, True],
        )
        self.assertEqual(
            [segment.audio_memory_active for segment in plan.segments],
            [False, True, False, True],
        )
        self.assertIn("voice, speech, and dialogue", plan.segments[0].prompt)
        self.assertNotIn("<d>", plan.segments[0].prompt)

    def test_zero_dialogue_window_preserves_authored_sound_source_identity(self) -> None:
        prompt = """subject_definitions: <Subject 1> is the person in <Picture 1>.
<Audio 1> is the voice reference for <Subject 1>.

summary: [reference generation + audio reference] A complete timeline.

retention_analysis:
<Subject 1>: fully_preserved - identity remains unchanged.
<Audio 1>: reference - voice remains unchanged.

detailed_description: One room remains stable.
[Shot 1] A person works silently. At 00:04.000, S1 says: <d>[English] first.</d>
[Shot 2] At 00:08.000, the camera cuts to an object-only close-up.
[Shot 3] At 00:16.000, the camera cuts overhead. The person works silently.
[Shot 4] At 00:24.000, the camera returns. At 00:27.000, S1 says: <d>[Chinese] 三。</d>

overall_soundscape: A delicate dry mechanical music-box melody continues beneath the authored lines.
non_diegetic_music: N/A
"""
        plan = plan_long_horizon(
            requested_duration_seconds=30,
            prompt=prompt,
            seed=31,
            maximum_opening_frames=277,
            structured_director=True,
        )
        denied = next(
            segment.prompt
            for segment in plan.segments
            if segment.authorized_dialogue_count == 0
            and "camera cuts overhead" in segment.prompt
        )
        # Text localization must not replace a concrete acoustic source with a
        # generic "instrumental" abstraction. Hard vocal enforcement happens
        # after decoding and therefore need not destroy soundscape identity.
        self.assertIn("<Audio 1>", denied)
        self.assertIn("delicate dry mechanical music-box", denied)
        self.assertIn("camera cuts overhead", denied)
        authorized = next(
            segment.prompt
            for segment in plan.segments
            if segment.authorized_dialogue_count == 1
        )
        self.assertIn("<Audio 1>", authorized)
        self.assertIn("delicate dry mechanical music-box", authorized)

    def test_localization_preserves_arbitrary_authored_acoustic_sources(self) -> None:
        soundscapes = (
            "A tiny clockwork music box plays dry metallic notes on the desk.",
            "A mono tabletop radio at camera left plays muffled late-night jazz.",
            "A cellist in the room performs a close, resonant acoustic phrase.",
            "Rain strikes the window while an old refrigerator hums at camera right.",
        )
        for soundscape in soundscapes:
            with self.subTest(soundscape=soundscape):
                prompt = f"""integrated_multimodal_description:
[Shot 1] A stable wide shot. At 00:04.000, S1 says: <d>[English] first.</d>
[Shot 2] At 00:08.000, the camera cuts to a silent object close-up.
[Shot 3] At 00:16.000, the camera cuts overhead. At 00:19.000, S1 says: <d>[Chinese] 第二句。</d>
[Shot 4] At 00:24.000, the camera returns to the wide shot.

overall_soundscape: {soundscape}
non_diegetic_music: N/A
"""
                plan = plan_long_horizon(
                    requested_duration_seconds=30,
                    prompt=prompt,
                    seed=37,
                    maximum_opening_frames=277,
                    structured_director=True,
                )
                self.assertTrue(any(
                    segment.authorized_dialogue_count == 0
                    for segment in plan.segments
                ))
                for segment in plan.segments:
                    self.assertIn(
                        f"overall_soundscape: {soundscape}",
                        segment.prompt,
                    )
                    self.assertNotIn(
                        "Preserve only an instrumental bed",
                        segment.prompt,
                    )

    def test_structured_director_does_not_change_free_prompt_path(self) -> None:
        prompt = "A person works at a table and later smiles."
        self.assertFalse(has_structured_timeline(prompt))
        baseline = plan_long_horizon(
            requested_duration_seconds=30,
            prompt=prompt,
            seed=82901,
            maximum_opening_frames=277,
        )
        requested = plan_long_horizon(
            requested_duration_seconds=30,
            prompt=prompt,
            seed=82901,
            maximum_opening_frames=277,
            structured_director=True,
        )
        self.assertFalse(requested.structured_director)
        self.assertTrue(all(
            segment.authorized_dialogue_count is None
            and segment.reference_audio_active
            and segment.audio_memory_active
            for segment in requested.segments
        ))
        self.assertEqual(baseline.telemetry(), requested.telemetry())
        self.assertEqual(
            [segment.prompt for segment in baseline.segments],
            [segment.prompt for segment in requested.segments],
        )


if __name__ == "__main__":
    unittest.main()
