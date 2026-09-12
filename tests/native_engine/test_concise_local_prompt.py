from __future__ import annotations

import os
import re
import unittest
from unittest.mock import patch

from h3serve.native_engine.long_horizon import _compact_local_prompt, localize_h3_prompt, plan_long_horizon


class ConciseLocalPromptTests(unittest.TestCase):
    PROMPT = '''integrated_multimodal_description: [Shot 1] One continuous fixed wide shot of a room and a woman wearing a violet coat. At 00:04.000, she says: <d>[Chinese] 第一句。</d> At 00:12.000, she places a book on the table; her hands are empty. At 00:25.000, she says: <d>[Chinese] 最后一句。</d>

overall_soundscape: Quiet room tone.
non_diegetic_music: N/A'''

    def test_geometry_and_literal_dialogue_ownership_do_not_change(self):
        plans = []
        for enabled in ('0', '1'):
            with patch.dict(os.environ, {'H3_LONG_CONCISE_LOCAL_PROMPT': enabled}):
                plans.append(plan_long_horizon(requested_duration_seconds=30,
                    prompt=self.PROMPT, seed=913, structured_director=True))
        self.assertEqual(len(plans[0].segments), len(plans[1].segments))
        for old, new in zip(plans[0].segments, plans[1].segments):
            self.assertEqual((old.window_frames, old.context_frames, old.seed, old.visible_start_frame),
                             (new.window_frames, new.context_frames, new.seed, new.visible_start_frame))
            self.assertEqual(old.authorized_dialogue_count, new.authorized_dialogue_count)
            self.assertEqual(re.findall(r'<d>.*?</d>', old.prompt), re.findall(r'<d>.*?</d>', new.prompt))
            self.assertIn('violet coat', new.prompt)
            self.assertNotIn('structured_director_contract:', new.prompt)
            self.assertLess(len(new.prompt), len(old.prompt))

    def test_authored_cut_clocks_are_not_removed(self):
        prompt = self.PROMPT.replace('At 00:12.000,', '[Shot 2] At 00:12.000, the camera cuts to a side view.')
        with patch.dict(os.environ, {'H3_LONG_CONCISE_LOCAL_PROMPT': '1'}):
            value = localize_h3_prompt(prompt, context_start_frame=180, visible_start_frame=219,
                visible_stop_frame=447, segment_index=1, timeline_stop_frame=719, structured_director=True)
        self.assertIn('At 00:04.500, the camera cuts to a side view', value)
        self.assertIn('local 00:04.500', value)

    def test_compactor_preserves_literal_dialogue_even_if_it_looks_like_metadata(self):
        value = '<d>[English] [Ongoing-shot state anchor: literal quoted words]</d>'
        with patch.dict(os.environ, {'H3_LONG_CONCISE_LOCAL_PROMPT': '1'}):
            self.assertEqual(_compact_local_prompt(value), value)

    def test_free_prompt_is_not_rewritten(self):
        prompt = 'A quiet landscape and distant birds.'
        versions = []
        for enabled in ('0', '1'):
            with patch.dict(os.environ, {'H3_LONG_CONCISE_LOCAL_PROMPT': enabled}):
                versions.append(localize_h3_prompt(prompt, context_start_frame=100,
                    visible_start_frame=139, visible_stop_frame=311, segment_index=1, structured_director=True))
        self.assertEqual(*versions)


if __name__ == '__main__':
    unittest.main()
