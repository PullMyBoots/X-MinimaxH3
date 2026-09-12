from __future__ import annotations

import unittest

from h3serve.native_engine.audio_spine import plan_global_audio_spine


class GlobalAudioSpineTests(unittest.TestCase):
    def test_landscape_proxy_is_small_and_keeps_one_audio_clock(self) -> None:
        plan = plan_global_audio_spine(
            output_width=864,
            output_height=480,
            output_frames=719,
        )
        self.assertEqual((plan.width, plan.height), (320, 192))
        self.assertEqual(plan.audio_ticks, 1198)
        self.assertEqual(plan.audio_tokens, 2396)
        self.assertEqual(plan.video_latent_frames, 212)
        self.assertEqual(plan.spatial_tokens_per_frame, 60)
        self.assertEqual(plan.maximum_local_frames, 311)
        self.assertEqual(len(plan.temporal_plan.windows), 3)
        self.assertLess(
            plan.maximum_local_packed_media_tokens,
            plan.video_tokens + plan.audio_tokens,
        )
        receipt = plan.telemetry()
        self.assertTrue(receipt["single_global_audio_trajectory"])
        self.assertEqual(receipt["internal_audio_seams"], 0)
        self.assertTrue(receipt["proxy_video_discarded"])
        self.assertFalse(receipt["feeds_back_into_primary_video"])
        self.assertTrue(receipt["native_duration_dit_views"])
        self.assertTrue(receipt["single_global_solver_state"])
        self.assertEqual(
            receipt["rotary_time"], "window_local_trained_range_v1"
        )

    def test_portrait_proxy_preserves_orientation(self) -> None:
        plan = plan_global_audio_spine(
            output_width=480,
            output_height=864,
            output_frames=719,
        )
        self.assertEqual((plan.width, plan.height), (192, 320))


if __name__ == "__main__":
    unittest.main()
