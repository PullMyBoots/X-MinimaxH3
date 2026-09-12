from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

from h3serve.native_engine.hot_session import (
    HotSessionRequest,
    NativeT2AVHotSession,
    QWEN_CONDITIONING_CACHE_SCHEMA_VERSION,
)
from h3serve.native_engine.runtime import RuntimeConfig


class _FakeConditioner:
    layers = 50

    def __init__(self, root: Path) -> None:
        self.calls = 0
        self.checkpoint = root / "qwen.safetensors"
        self.checkpoint.write_bytes(b"fixed-qwen-weights")
        self.tokenizer_path = root / "tokenizer"
        self.tokenizer_path.mkdir()
        (self.tokenizer_path / "tokenizer_config.json").write_text(
            "{}", encoding="utf-8"
        )

    def _encoded(self):
        self.calls += 1
        return SimpleNamespace(
            prompt_embeds=torch.arange(3 * 5120, dtype=torch.float32).view(
                1, 3, 5120
            ),
            text_token_tags=torch.ones(3, dtype=torch.long),
        )

    def encode_prompt(self, _prompt: str):
        return self._encoded()

    def encode_request(self, _request: HotSessionRequest):
        return self._encoded()


class QwenConditioningCacheTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.conditioner = _FakeConditioner(self.root)
        self.session = object.__new__(NativeT2AVHotSession)
        self.session.engine = "original"
        self.session.conditioner = self.conditioner
        self.session.runtime_config = RuntimeConfig.cpu_test()
        self.session._prompt_cache = None
        self.session._conditioning_cache = None
        self.session._persisted_conditioning_cache = None
        self.session._last_conditioning_cache_payload = None
        self.session._last_conditioning_cache_status = "not_checked"
        self.session._last_conditioning_cache_fallback = None

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _request(self, **overrides) -> HotSessionRequest:
        values = {
            "prompt": "one immutable H3 condition",
            "seed": 7,
            "width": 864,
            "height": 480,
            "frames": 362,
            "fps": 24,
            "steps": 1,
            "output_path": self.root / "output.mp4",
        }
        values.update(overrides)
        return HotSessionRequest(**values)

    def test_persisted_prompt_condition_bypasses_qwen_exactly(self) -> None:
        request = self._request()
        fingerprint = self.session._conditioning_fingerprint(request)
        embeds = torch.arange(3 * 5120, dtype=torch.float32).view(1, 3, 5120)
        tags = torch.tensor([1, 0, 1], dtype=torch.long)
        source = self.root / "source.pt"
        torch.save(
            {
                "video": torch.zeros(1),
                "audio": torch.zeros(1),
                "qwen_conditioning_cache": {
                    "schema_version": QWEN_CONDITIONING_CACHE_SCHEMA_VERSION,
                    "fingerprint": fingerprint,
                    "encoder": self.session._conditioning_encoder_identity(),
                    "prompt_embeds": embeds,
                    "text_token_tags": tags,
                },
            },
            source,
        )
        request = replace(request, conditioning_cache_source_path=source)
        actual_embeds, actual_tags = self.session._encode_request(request)
        self.assertEqual(self.conditioner.calls, 0)
        self.assertTrue(torch.equal(actual_embeds, embeds))
        self.assertTrue(torch.equal(actual_tags, tags))
        self.assertEqual(
            self.session._last_conditioning_cache_status, "checkpoint_hit"
        )

        # Output resolution and duration do not affect a prompt-only Qwen
        # condition, so a subsequent temporal piece stays a hot-session hit.
        resized = replace(request, width=2560, height=1440, frames=136)
        second_embeds, second_tags = self.session._encode_request(resized)
        self.assertEqual(self.conditioner.calls, 0)
        self.assertTrue(torch.equal(second_embeds, embeds))
        self.assertTrue(torch.equal(second_tags, tags))
        self.assertEqual(
            self.session._last_conditioning_cache_status, "hot_session_hit"
        )

    def test_legacy_checkpoint_falls_back_to_qwen_and_builds_cache(self) -> None:
        source = self.root / "legacy.pt"
        torch.save({"video": torch.zeros(1), "audio": torch.zeros(1)}, source)
        request = self._request(conditioning_cache_source_path=source)
        embeds, tags = self.session._encode_request(request)
        self.assertEqual(self.conditioner.calls, 1)
        self.assertEqual(tuple(embeds.shape), (1, 3, 5120))
        self.assertEqual(tuple(tags.shape), (3,))
        self.assertEqual(self.session._last_conditioning_cache_status, "encoded")
        self.assertEqual(
            self.session._last_conditioning_cache_fallback,
            "checkpoint_missing",
        )
        self.assertIsNotNone(self.session._last_conditioning_cache_payload)
        upgraded = torch.load(source, map_location="cpu", weights_only=True)
        self.assertIn("qwen_conditioning_cache", upgraded)
        self.assertTrue(torch.equal(
            upgraded["qwen_conditioning_cache"]["prompt_embeds"], embeds
        ))

    def test_internal_bridge_moves_persisted_tensors_to_runtime_device(self) -> None:
        embeds = torch.arange(3 * 5120, dtype=torch.float32).view(1, 3, 5120)
        tags = torch.tensor([1, 0, 1], dtype=torch.long)
        source = self.root / "bridge.pt"
        torch.save(
            {
                "qwen_conditioning_cache": {
                    "schema_version": QWEN_CONDITIONING_CACHE_SCHEMA_VERSION,
                    "fingerprint": "preceding-window-condition",
                    "encoder": self.session._conditioning_encoder_identity(),
                    "prompt_embeds": embeds,
                    "text_token_tags": tags,
                }
            },
            source,
        )
        # Intercept the CUDA move so this remains runnable on CPU-only CI. The
        # production failure this guards was a host tensor reaching a CUDA
        # ``condition_proj`` only for the second long-video window.
        self.session.runtime_config = replace(
            RuntimeConfig.cpu_test(), device="cuda:0"
        )
        original_to = torch.Tensor.to
        requested_cuda_moves: list[str] = []

        def observe_to(tensor, *args, **kwargs):
            destination = args[0] if args else kwargs.get("device")
            if str(destination) == "cuda:0":
                requested_cuda_moves.append(str(destination))
                return tensor
            return original_to(tensor, *args, **kwargs)

        with mock.patch.object(torch.Tensor, "to", observe_to):
            self.session._load_internal_conditioning_bridge(source)
        self.assertEqual(requested_cuda_moves, ["cuda:0", "cuda:0"])

    def test_ref_images_and_audio_are_static_across_target_geometry(self) -> None:
        self.session.engine = "reference"
        image = self.root / "reference.png"
        audio = self.root / "reference.wav"
        video = self.root / "reference.mp4"
        first = self.root / "first.png"
        image.write_bytes(b"image")
        audio.write_bytes(b"audio")
        video.write_bytes(b"video")
        first.write_bytes(b"first")

        static = self._request(
            reference_images=(image,), reference_audios=(audio,)
        )
        high = replace(static, width=2560, height=1440, frames=136)
        self.assertEqual(
            self.session._conditioning_fingerprint(static),
            self.session._conditioning_fingerprint(high),
        )

        video_condition = replace(static, reference_videos=(video,))
        video_piece = replace(video_condition, frames=136)
        self.assertNotEqual(
            self.session._conditioning_fingerprint(video_condition),
            self.session._conditioning_fingerprint(video_piece),
        )

        self.session.engine = "original"
        keyframe = self._request(first_frame=first)
        keyframe_high = replace(keyframe, width=2560, height=1440)
        self.assertNotEqual(
            self.session._conditioning_fingerprint(keyframe),
            self.session._conditioning_fingerprint(keyframe_high),
        )


if __name__ == "__main__":
    unittest.main()
