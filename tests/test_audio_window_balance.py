from __future__ import annotations

import unittest

import numpy as np

from h3serve.native_engine.audio_window_balance import balance_creator_window_pcm


class CreatorWindowAudioBalanceTests(unittest.TestCase):
    def test_reduces_bounded_window_level_mismatch(self) -> None:
        rate = 4000
        seconds = 6
        timeline = np.arange(rate * seconds, dtype=np.float32) / rate
        source = np.sin(2 * np.pi * 180 * timeline).astype(np.float32)
        stereo = np.stack((source, source), axis=1)
        stereo[: rate * 2] *= 0.20
        stereo[rate * 2 : rate * 4] *= 0.10
        stereo[rate * 4 :] *= 0.14

        balanced, profile = balance_creator_window_pcm(
            stereo, (rate * 2, rate * 4), sample_rate=rate
        )

        self.assertEqual(balanced.shape, stereo.shape)
        self.assertTrue(np.isfinite(balanced).all())
        self.assertEqual(profile["boundary_count"], 2)
        self.assertAlmostEqual(profile["relative_gain_db"][1], 3.0, places=3)
        self.assertAlmostEqual(max(profile["target_gain_db"]), 3.0, places=6)
        self.assertLessEqual(float(np.max(np.abs(balanced))), 0.98)
        before_ratio = np.sqrt(np.mean(np.square(stereo[rate : rate * 2]))) / np.sqrt(
            np.mean(np.square(stereo[rate * 2 + 1000 : rate * 3]))
        )
        after_ratio = np.sqrt(np.mean(np.square(balanced[rate : rate * 2]))) / np.sqrt(
            np.mean(np.square(balanced[rate * 2 + 1000 : rate * 3]))
        )
        self.assertLess(after_ratio, before_ratio)

    def test_does_not_amplify_a_deliberately_silent_window(self) -> None:
        rate = 4000
        tone = np.sin(
            2 * np.pi * 120 * np.arange(rate * 4, dtype=np.float32) / rate
        )
        stereo = np.stack((tone, tone), axis=1) * 0.15
        stereo[rate * 2 :] = 0.0

        balanced, profile = balance_creator_window_pcm(
            stereo, (rate * 2,), sample_rate=rate
        )

        self.assertEqual(profile["target_gain_db"], [0.0, 0.0])
        self.assertEqual(
            profile["records"][0]["reason"], "insufficient_persistent_audio"
        )
        self.assertTrue(np.allclose(balanced[rate * 2 + rate :], 0.0))

    def test_large_but_plausible_boundary_uses_bounded_correction(self) -> None:
        rate = 4000
        timeline = np.arange(rate * 4, dtype=np.float32) / rate
        tone = np.sin(2 * np.pi * 150 * timeline).astype(np.float32)
        stereo = np.stack((tone, tone), axis=1)
        stereo[: rate * 2] *= 0.20
        stereo[rate * 2 :] *= 0.07

        _, profile = balance_creator_window_pcm(
            stereo, (rate * 2,), sample_rate=rate
        )

        self.assertIsNone(profile["records"][0]["reason"])
        self.assertAlmostEqual(profile["relative_gain_db"][1], 3.0, places=3)


if __name__ == "__main__":
    unittest.main()
