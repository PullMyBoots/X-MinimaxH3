from __future__ import annotations

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from h3serve.native_engine.incremental_preview import append_cumulative_preview


FFMPEG = shutil.which("ffmpeg")
FFPROBE = shutil.which("ffprobe")


def _run(*arguments: str) -> bytes:
    return subprocess.run(
        list(arguments),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    ).stdout


@unittest.skipUnless(FFMPEG and FFPROBE, "ffmpeg and ffprobe are required")
class IncrementalPreviewTests(unittest.TestCase):
    def test_packet_copy_preserves_history_and_appends_only_visible_suffix(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root)
            source = directory / "source.mp4"
            piece = directory / "piece.mp4"
            _run(
                FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
                "-f", "lavfi", "-i", "testsrc2=size=160x96:rate=24",
                "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=32000",
                "-frames:v", "24", "-c:v", "libx264", "-crf", "14",
                "-preset", "superfast", "-pix_fmt", "yuv420p", "-c:a", "aac",
                "-b:a", "192k", "-ar", "32000", "-ac", "2", "-shortest",
                str(source),
            )
            _run(
                FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
                "-f", "lavfi", "-i", "testsrc=size=160x96:rate=24",
                "-f", "lavfi", "-i", "sine=frequency=880:sample_rate=32000",
                "-frames:v", "24", "-c:v", "libx264", "-crf", "14",
                "-preset", "superfast", "-pix_fmt", "yuv420p", "-c:a", "aac",
                "-b:a", "192k", "-ar", "32000", "-ac", "2", "-shortest",
                str(piece),
            )

            receipt = append_cumulative_preview(
                source,
                piece,
                piece,
                source_frames=24,
                context_frames=8,
                physical_frames=24,
                output_frames=40,
                fps=24,
            )

            source_pixels = _run(
                FFMPEG, "-hide_banner", "-loglevel", "error", "-i", str(source),
                "-map", "0:v:0", "-frames:v", "24", "-f", "rawvideo",
                "-pix_fmt", "yuv420p", "-",
            )
            cumulative_prefix = _run(
                FFMPEG, "-hide_banner", "-loglevel", "error", "-i", str(piece),
                "-map", "0:v:0", "-frames:v", "24", "-f", "rawvideo",
                "-pix_fmt", "yuv420p", "-",
            )
            self.assertEqual(cumulative_prefix, source_pixels)
            self.assertEqual(receipt["output_frames"], 40)
            self.assertEqual(receipt["visible_frames_appended"], 16)
            self.assertFalse(receipt["prior_preview_reencoded"])
            self.assertFalse(receipt["cumulative_vae_decode"])

    def test_rejects_inconsistent_output_clock(self) -> None:
        with self.assertRaisesRegex(ValueError, "output clock"):
            append_cumulative_preview(
                __file__,
                __file__,
                Path(__file__).with_suffix(".mp4"),
                source_frames=24,
                context_frames=8,
                physical_frames=24,
                output_frames=39,
                fps=24,
            )


if __name__ == "__main__":
    unittest.main()
