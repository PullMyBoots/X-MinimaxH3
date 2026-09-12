from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from h3serve.native_engine.long_horizon import localize_h3_prompt


class StateFirstContinuationTests(unittest.TestCase):
    PROMPT = '''integrated_multimodal_description: [Shot 1] A continuous fixed wide shot with a gray-haired woman wearing a violet jacket. At 00:20.000, she opens the door; the door remains open and the ring stays on the desk. At 00:27.000, she leaves through the door; the camera keeps recording. From 00:30.000 through 00:42.000, the room is empty; the ring remains on the desk. At 00:43.000, the same woman returns empty-handed. At 00:54.000, she says: <d>[Chinese] 还在这里。</d>

overall_soundscape: Quiet ambience.
non_diegetic_music: N/A'''

    def local(self, start, stop, enabled='1'):
        with patch.dict(os.environ, {'H3_LONG_STATE_FIRST_CONTINUATION': enabled,
                                    'H3_LONG_LOCAL_STATE_INTERVALS': '1',
                                    'H3_LONG_RETAIN_SHOT_SETUP': '0',
                                    'H3_LONG_CONCISE_LOCAL_PROMPT': '0'}):
            return localize_h3_prompt(self.PROMPT, context_start_frame=start-56,
                visible_start_frame=start, visible_stop_frame=stop, segment_index=3,
                timeline_stop_frame=1433, structured_director=True)

    def test_old_atomic_onset_is_not_reissued(self):
        value = self.local(600, 787)
        self.assertNotIn('she opens the door', value)
        self.assertIn('door remains open', value)
        self.assertIn('At 00:04.333, she leaves', value)
        self.assertIn('she opens the door', self.local(600, 787, '0'))

    def test_current_hold_does_not_repopulate_opening_cast(self):
        value = self.local(787, 974)
        self.assertIn('room is empty', value)
        self.assertIn('ring remains on the desk', value)
        self.assertNotIn('violet jacket', value)
        self.assertNotIn('she leaves', value)
        self.assertNotIn('she opens', value)

    def test_return_gets_identity_after_its_own_clock_not_during_hold(self):
        value = self.local(974, 1229)
        self.assertIn('room is empty', value)
        self.assertIn('At 00:04.750, the same woman returns', value)
        self.assertGreater(value.index('violet jacket'), value.index('At 00:04.750'))
        self.assertNotIn('还在这里', value)

    def test_disabled_path_keeps_setup_during_hold(self):
        value = self.local(787, 974, '0')
        self.assertIn('violet jacket', value)

    def test_multishot_carries_latest_result_instead_of_stale_opening(self):
        prompt = '''integrated_multimodal_description: [Shot 1] A wide view.
[Shot 2] At 00:15.000, the camera cuts to a side view with a latched cabinet. At 00:18.000, she opens the cabinet; the cabinet stays open and her hands withdraw. At 00:23.000, she steps back.
[Shot 3] At 00:30.000, the camera cuts back to a wide view.
overall_soundscape: Quiet ambience.
non_diegetic_music: N/A'''
        for enabled in ('0', '1'):
            with patch.dict(os.environ, {'H3_LONG_STATE_FIRST_CONTINUATION': enabled}):
                value = localize_h3_prompt(prompt, context_start_frame=442,
                    visible_start_frame=481, visible_stop_frame=685, segment_index=2,
                    timeline_stop_frame=1076, structured_director=True)
            if enabled == '1':
                self.assertIn('cabinet stays open', value)
                self.assertNotIn('latched cabinet', value)
                self.assertNotIn('she opens the cabinet', value)
                self.assertIn('At 00:04.583, she steps back', value)
                self.assertNotIn('camera cuts back', value)
            else:
                self.assertIn('latched cabinet', value)
                self.assertNotIn('cabinet stays open', value)


if __name__ == '__main__':
    unittest.main()
