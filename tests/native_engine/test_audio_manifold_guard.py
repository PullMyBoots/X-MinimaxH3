from __future__ import annotations

import unittest
from unittest import mock

import numpy as np
import torch

from h3serve.native_engine.audio_manifold_guard import (
    AudioManifoldGuardConfig,
    apply_audio_manifold_guard,
    detect_audio_manifold_risks,
)


class _IdentityScaleAudioVAE:
    def encode(self, value, *, return_cpu):
        self.assert_cuda_contract(return_cpu)
        return value[:, 0, :]

    def decode(self, value, *, stereo_batch, return_cpu):
        self.assert_cuda_contract(return_cpu)
        if not stereo_batch:
            raise AssertionError("guard must retain H3 stereo-batch decoding")
        return value * 0.5

    @staticmethod
    def assert_cuda_contract(return_cpu):
        if return_cpu:
            raise AssertionError("projection must remain on device until complete")


class AudioManifoldGuardTest(unittest.TestCase):
    def test_silence_returns_before_vad(self) -> None:
        audio = np.zeros((2, 32_000), dtype=np.float32)
        with mock.patch(
            "h3serve.native_engine.audio_manifold_guard._vad_flags"
        ) as vad:
            risks, profile = detect_audio_manifold_risks(audio)
        self.assertEqual(risks, [])
        self.assertFalse(profile["vad_evaluated"])
        vad.assert_not_called()

    def test_speech_vote_accepts_overlapping_acoustic_blocks(self) -> None:
        risks = [
            {"start_seconds": 1.0, "stop_seconds": 1.5},
            {"start_seconds": 1.25, "stop_seconds": 1.75},
        ]
        flags = np.zeros(100, dtype=bool)
        flags[33:50] = True
        with (
            mock.patch(
                "h3serve.native_engine.audio_manifold_guard._acoustic_risk_blocks",
                return_value=risks,
            ),
            mock.patch(
                "h3serve.native_engine.audio_manifold_guard._vad_flags",
                return_value=flags,
            ),
        ):
            accepted, profile = detect_audio_manifold_risks(
                np.zeros((2, 96_000), dtype=np.float32),
                config=AudioManifoldGuardConfig(
                    minimum_local_speech_fraction=0.7,
                ),
            )
        self.assertEqual(len(accepted), 2)
        self.assertEqual(profile["accepted_risk_blocks"], 2)

    def test_continuous_noise_and_sparse_transient_fail_closed(self) -> None:
        risk = [{"start_seconds": 1.0, "stop_seconds": 1.5}]
        continuous = np.ones(100, dtype=bool)
        sparse = np.zeros(100, dtype=bool)
        sparse[33:38] = True
        for flags in (continuous, sparse):
            with (
                mock.patch(
                    "h3serve.native_engine.audio_manifold_guard._acoustic_risk_blocks",
                    return_value=risk,
                ),
                mock.patch(
                    "h3serve.native_engine.audio_manifold_guard._vad_flags",
                    return_value=flags,
                ),
            ):
                accepted, _ = detect_audio_manifold_risks(
                    np.zeros((2, 96_000), dtype=np.float32)
                )
            self.assertEqual(accepted, [])

    def test_no_risk_preserves_original_tensor_object(self) -> None:
        audio = torch.zeros((2, 32_000), dtype=torch.float32)
        output, profile = apply_audio_manifold_guard(object(), audio)
        self.assertIs(output, audio)
        self.assertFalse(profile["applied"])

    @unittest.skipUnless(torch.cuda.is_available(), "Audio-VAE guard runs on CUDA")
    def test_repair_changes_only_feathered_interval(self) -> None:
        audio = torch.full((2, 64_000), 0.2, dtype=torch.float32)
        risks = [{"start_seconds": 0.75, "stop_seconds": 1.25}]
        config = AudioManifoldGuardConfig(
            projection_rounds=1,
            feather_seconds=0.1,
            projection_context_seconds=0.2,
        )
        with mock.patch(
            "h3serve.native_engine.audio_manifold_guard.detect_audio_manifold_risks",
            return_value=(risks, {"accepted_risk_blocks": 1}),
        ):
            output, profile = apply_audio_manifold_guard(
                _IdentityScaleAudioVAE(), audio, config=config
            )
        self.assertTrue(profile["applied"])
        self.assertTrue(torch.equal(output[:, :20_000], audio[:, :20_000]))
        self.assertTrue(torch.equal(output[:, 45_000:], audio[:, 45_000:]))
        self.assertLess(float(output[:, 32_000].mean()), 0.2)


if __name__ == "__main__":
    unittest.main()
