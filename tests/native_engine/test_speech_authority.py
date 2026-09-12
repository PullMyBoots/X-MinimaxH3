from __future__ import annotations

import unittest

from h3serve.native_engine.long_horizon import plan_long_horizon
from h3serve.native_engine.speech_authority import (
    forbidden_speech_intervals,
    vocal_gate_expression,
)


class StructuredSpeechAuthorityTests(unittest.TestCase):
    PROMPT = """integrated_multimodal_description:
[Shot 1] A stable wide shot. At 00:04.500, S1 says: <d>[English] one.</d>
[Shot 2] At 00:08.000, the camera cuts to a close shot. Only object sounds occur.
[Shot 3] At 00:16.000, the camera cuts overhead. At 00:19.500, S1 says: <d>[Japanese] 二。</d>
[Shot 4] At 00:24.000, the camera cuts back. At 00:27.000, S1 says: <d>[Chinese] 三。</d>

overall_soundscape: Quiet room tone.
non_diegetic_music: N/A
"""

    def test_only_compiled_zero_dialogue_windows_are_denied(self) -> None:
        plan = plan_long_horizon(
            requested_duration_seconds=30,
            prompt=self.PROMPT,
            seed=17,
            maximum_opening_frames=277,
            structured_director=True,
        )
        expected = tuple(
            (
                segment.visible_start_frame / 24,
                segment.visible_stop_frame / 24,
            )
            for segment in plan.segments
            if segment.authorized_dialogue_count == 0
        )
        self.assertEqual(forbidden_speech_intervals(plan), expected)
        self.assertTrue(expected)

    def test_free_prompt_never_creates_a_denial_interval(self) -> None:
        plan = plan_long_horizon(
            requested_duration_seconds=30,
            prompt="A person tells a story while walking through a room.",
            seed=17,
            maximum_opening_frames=277,
            structured_director=True,
        )
        self.assertFalse(plan.structured_director)
        self.assertEqual(forbidden_speech_intervals(plan), ())

    def test_envelope_merges_overlaps_and_has_crossfade(self) -> None:
        expression = vocal_gate_expression(
            ((2.0, 4.0), (3.5, 5.0), (8.0, 9.0)),
            fade_seconds=0.1,
        )
        self.assertIn("1.900000", expression)
        self.assertIn("5.100000", expression)
        self.assertIn("7.900000", expression)
        self.assertIn("9.100000", expression)
        self.assertIn("max(", expression)


if __name__ == "__main__":
    unittest.main()
