from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from h3serve.native_engine.long_horizon import (
    _expand_shot_state_intervals, localize_h3_prompt,
)


class LocalStateIntervalTests(unittest.TestCase):
    PROMPT = '''integrated_multimodal_description: [Shot 1] One continuous fixed shot. At 00:27.000, a cyclist leaves the room. The camera remains fixed. From 00:30.000 through 00:42.000, the room is empty and a helmet stays on the shelf. At 00:43.000, the cyclist returns empty-handed.

overall_soundscape: Quiet ambience.
non_diegetic_music: N/A'''

    def local(self):
        return localize_h3_prompt(self.PROMPT, context_start_frame=918,
            visible_start_frame=974, visible_stop_frame=1229, segment_index=5,
            timeline_stop_frame=1433, structured_director=True)

    def test_hold_owns_current_interval_instead_of_replaying_departure(self):
        with patch.dict(os.environ, {"H3_LONG_LOCAL_STATE_INTERVALS": "1"}):
            value = self.local()
        self.assertNotIn("cyclist leaves", value)
        self.assertIn("room is empty", value)
        self.assertIn("through local 00:03.750", value)
        self.assertIn("At 00:04.750, the cyclist returns", value)
        self.assertNotIn("From 00:30.000", value)

    def test_disabled_path_is_unchanged(self):
        with patch.dict(os.environ, {"H3_LONG_LOCAL_STATE_INTERVALS": "0"}):
            value = self.local()
        self.assertIn("cyclist leaves", value)
        self.assertIn("From 00:30.000 through 00:42.000", value)

    def test_literal_dialogue_and_mid_sentence_range_are_not_promoted(self):
        value = '<d>[English] From 00:30.000 through 00:42.000, wait.</d> She waits from 00:30.000 through 00:42.000.'
        self.assertEqual(_expand_shot_state_intervals(value, 20), value)

    def test_future_interval_does_not_leak_into_earlier_window(self):
        with patch.dict(os.environ, {"H3_LONG_LOCAL_STATE_INTERVALS": "1"}):
            value = localize_h3_prompt(self.PROMPT, context_start_frame=500,
                visible_start_frame=556, visible_stop_frame=700, segment_index=3,
                timeline_stop_frame=1433, structured_director=True)
        self.assertIn("cyclist leaves", value)
        self.assertNotIn("helmet stays", value)


if __name__ == '__main__':
    unittest.main()
