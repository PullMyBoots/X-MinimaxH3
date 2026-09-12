import unittest

import torch

from h3serve.native_engine.audio_window_decode import assemble_window_decoded_audio


class AudioWindowDecodeTests(unittest.TestCase):
    def test_crops_after_decode_on_one_cumulative_pcm_clock(self) -> None:
        rate = 2_400
        first = torch.arange(500, dtype=torch.float32).reshape(1, 1, -1).repeat(1, 2, 1)
        second = (1_000 + torch.arange(600, dtype=torch.float32)).reshape(1, 1, -1).repeat(1, 2, 1)
        audio, profile = assemble_window_decoded_audio(
            (first, second),
            ((0, 3), (2, 4)),
            output_frames=7,
            fps=24,
            sample_rate=rate,
        )
        self.assertEqual(audio.shape, (1, 2, 700))
        torch.testing.assert_close(audio[..., :300], first[..., :300])
        torch.testing.assert_close(audio[..., 300:], second[..., 200:600])
        self.assertFalse(profile["audio_latent_interpolation"])
        self.assertEqual(profile["temporal_vae_domains"], 2)

    def test_correlated_hidden_overlap_gets_only_short_preboundary_patch(self) -> None:
        rate = 8_000
        time = torch.arange(8_000, dtype=torch.float32) / rate
        source = torch.sin(2.0 * torch.pi * 220.0 * time)
        first = source[:4_000].reshape(1, 1, -1).repeat(1, 2, 1)
        hidden = torch.roll(first[..., -800:], shifts=25, dims=-1)
        continuation = source[4_000:5_600].reshape(1, 1, -1).repeat(1, 2, 1)
        second = torch.cat((hidden, continuation), dim=-1)
        audio, profile = assemble_window_decoded_audio(
            (first, second),
            ((0, 10), (2, 4)),
            output_frames=14,
            fps=20,
            sample_rate=rate,
        )
        seam = profile["window_records"][1]["seam"]
        self.assertTrue(seam["applied"])
        self.assertLessEqual(seam["crossfade_samples"], round(rate * 0.060))
        torch.testing.assert_close(audio[..., 4_000:], continuation)

    def test_fractional_av_clock_deficit_is_taken_from_hidden_preroll(self) -> None:
        # At 24 fps / 40-Hz Audio-VAE geometry, 260 video frames decode to 433
        # audio ticks. At this reduced sample rate that is 25,980 samples,
        # while the exact video clock asks for 26,000. The 20-sample deficit
        # must come from hidden preroll, never waveform stretching or tail pad.
        rate = 2_400
        first = torch.arange(17_520, dtype=torch.float32).reshape(
            1, 1, -1
        ).repeat(1, 2, 1)
        second = (100_000 + torch.arange(25_980, dtype=torch.float32)).reshape(
            1, 1, -1
        ).repeat(1, 2, 1)
        audio, profile = assemble_window_decoded_audio(
            (first, second),
            ((0, 175), (90, 170)),
            output_frames=345,
            fps=24,
            sample_rate=rate,
        )
        self.assertEqual(audio.shape, (1, 2, 34_500))
        torch.testing.assert_close(audio[..., 17_500:], second[..., 8_980:25_980])
        record = profile["window_records"][1]
        self.assertEqual(record["nominal_trim_samples"], 9_000)
        self.assertEqual(record["trim_samples"], 8_980)
        self.assertEqual(record["hidden_clock_compensation_samples"], 20)
        self.assertEqual(
            record["clock_compensation_policy"],
            "hidden_preroll_crop_shift_v1",
        )
        self.assertFalse(profile["audio_latent_interpolation"])

    def test_cumulative_source_clock_deficit_uses_next_hidden_overlap(self) -> None:
        # A cumulative 719-frame source has no own preroll and is 20 samples
        # short at this reduced 24-fps/2.4-kHz clock. The next continuation
        # window carries 56 frames of aligned hidden overlap, so the tiny
        # deficit can be filled without stretching or terminal padding.
        rate = 2_400
        first = torch.zeros(1, 2, 71_880)
        second = torch.ones(1, 2, 29_400)
        audio, profile = assemble_window_decoded_audio(
            (first, second),
            ((0, 719), (56, 238)),
            output_frames=957,
            fps=24,
            sample_rate=rate,
        )
        self.assertEqual(audio.shape, (1, 2, 95_700))
        record = profile["window_records"][0]
        self.assertEqual(record["decoded_samples"], 71_880)
        self.assertEqual(record["hidden_clock_compensation_samples"], 20)
        self.assertEqual(record["borrowed_next_overlap_samples"], 20)
        self.assertEqual(
            record["clock_compensation_policy"],
            "next_window_hidden_overlap_fill_v1",
        )

    def test_rejects_a_clock_that_does_not_cover_the_output(self) -> None:
        with self.assertRaisesRegex(ValueError, "missed the output frame count"):
            assemble_window_decoded_audio(
                (torch.zeros(1, 2, 100),),
                ((0, 1),),
                output_frames=2,
                fps=24,
                sample_rate=2_400,
            )


if __name__ == "__main__":
    unittest.main()
