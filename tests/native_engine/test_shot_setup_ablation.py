from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from h3serve.native_engine.long_horizon import localize_h3_prompt, plan_long_horizon


class ShotSetupAblationTests(unittest.TestCase):
    PROMPT = '''integrated_multimodal_description: [Shot 1] Realistic live action in one continuous take. A locked-off wide camera shows a room with a desk left and a blue door right. One elderly woman wearing round glasses and a purple jacket holds a brass ring. At 00:03.000, she says: <d>[Chinese] 放在这里。</d> At 00:08.000, she places the ring on the desk. At 00:18.000, she leaves through the door. At 00:25.000, the same woman returns empty-handed.

overall_soundscape: Quiet room tone.
non_diegetic_music: N/A'''

    def local(self):
        return localize_h3_prompt(self.PROMPT, context_start_frame=350,
            visible_start_frame=406, visible_stop_frame=610, segment_index=2,
            timeline_stop_frame=719, structured_director=True)

    def test_disabled_path_keeps_existing_first_clause_policy(self):
        with patch.dict(os.environ, {"H3_LONG_RETAIN_SHOT_SETUP": "0"}):
            value = self.local()
        self.assertIn("Ongoing-shot state anchor", value)
        self.assertNotIn("purple jacket", value)

    def test_setup_preserves_details_without_replaying_dialogue(self):
        with patch.dict(os.environ, {"H3_LONG_RETAIN_SHOT_SETUP": "1"}):
            value = self.local()
        self.assertIn("locked-off wide camera", value)
        self.assertIn("purple jacket", value)
        self.assertIn("superseded by the exact carried", value)
        self.assertNotIn("放在这里", value)
        self.assertIn("At 00:03.417, she leaves", value)

    def test_setup_does_not_change_geometry_seeds_or_opening(self):
        plans = []
        for enabled in ("0", "1"):
            with patch.dict(os.environ, {"H3_LONG_RETAIN_SHOT_SETUP": enabled}):
                plans.append(plan_long_horizon(requested_duration_seconds=30,
                    prompt=self.PROMPT, seed=781, structured_director=True))
        self.assertEqual(plans[0].segments[0].prompt, plans[1].segments[0].prompt)
        for left, right in zip(plans[0].segments, plans[1].segments):
            self.assertEqual(left.visible_start_frame, right.visible_start_frame)
            self.assertEqual(left.visible_stop_frame, right.visible_stop_frame)
            self.assertEqual(left.seed, right.seed)


if __name__ == '__main__':
    unittest.main()
