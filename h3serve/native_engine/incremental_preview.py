"""Incrementally publish cumulative low-resolution long-video previews."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any


def _run(command: list[str]) -> None:
    completed = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    if completed.returncode:
        detail = completed.stdout.strip()[-4000:]
        raise RuntimeError(f"incremental preview ffmpeg failed: {detail}")


def _probe(path: Path) -> dict[str, Any]:
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        raise RuntimeError("incremental preview requires ffprobe")
    completed = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,nb_read_frames,r_frame_rate,start_time",
            "-of",
            "json",
            str(path),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    if completed.returncode:
        raise RuntimeError(
            "incremental preview probe failed: " + completed.stdout.strip()[-4000:]
        )
    streams = json.loads(completed.stdout).get("streams", [])
    if len(streams) != 1:
        raise RuntimeError("incremental preview has no video stream")
    return dict(streams[0])


def _concat_entry(path: Path) -> str:
    # Runtime outputs are UUID-named, but keep the helper safe for ordinary
    # operator paths as well. FFmpeg concat files escape a quote as '\''.
    return "file '" + str(path.resolve()).replace("'", "'\\''") + "'\n"


def append_cumulative_preview(
    source: str | Path,
    physical_piece: str | Path,
    destination: str | Path,
    *,
    source_frames: int,
    context_frames: int,
    physical_frames: int,
    output_frames: int,
    fps: int,
) -> dict[str, Any]:
    """Append only the new visible frames from a decoded continuation.

    The previous cumulative preview is packet-copied byte for byte. Only the
    new physical window is decoded once so its hidden context can be trimmed,
    then that visible suffix is encoded with the service's regular delivery
    settings. This removes the old growing full-timeline VAE decode and avoids
    generation loss on already accepted preview frames.
    """

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("incremental preview requires ffmpeg")
    source_path = Path(source).resolve()
    piece_path = Path(physical_piece).resolve()
    destination_path = Path(destination).resolve()
    if not source_path.is_file() or not piece_path.is_file():
        raise RuntimeError("incremental preview source media is unavailable")
    source_count = int(source_frames)
    context_count = int(context_frames)
    physical_count = int(physical_frames)
    output_count = int(output_frames)
    frame_rate = int(fps)
    visible_count = physical_count - context_count
    if min(source_count, visible_count, frame_rate) <= 0:
        raise ValueError("incremental preview has an invalid frame clock")
    if output_count != source_count + visible_count:
        raise ValueError("incremental preview output clock is inconsistent")

    destination_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{destination_path.stem}.append-",
        dir=destination_path.parent,
    ) as temporary_root:
        temporary = Path(temporary_root)
        visible_piece = temporary / "visible-piece.mp4"
        concat_list = temporary / "concat.txt"
        combined = temporary / "cumulative.mp4"
        start_seconds = context_count / frame_rate
        duration_seconds = visible_count / frame_rate
        _run([
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(piece_path),
            "-vf",
            (
                f"trim=start_frame={context_count}:"
                f"end_frame={physical_count},setpts=PTS-STARTPTS"
            ),
            "-af",
            (
                f"atrim=start={start_seconds:.12f}:"
                f"duration={duration_seconds:.12f},asetpts=PTS-STARTPTS"
            ),
            "-frames:v",
            str(visible_count),
            "-r",
            str(frame_rate),
            "-c:v",
            "libx264",
            "-crf",
            "14",
            "-preset",
            "superfast",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-ar",
            "32000",
            "-ac",
            "2",
            "-movflags",
            "+faststart",
            str(visible_piece),
        ])
        concat_list.write_text(
            _concat_entry(source_path) + _concat_entry(visible_piece),
            encoding="utf-8",
        )
        _run([
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            # Preserve the source track's zero video timestamp. Without
            # -copyts, AAC encoder priming makes the MP4 muxer shift video by
            # roughly one frame; players then hold/duplicate the opening frame
            # even though every H.264 packet itself was copied unchanged.
            "-copyts",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_list),
            "-map",
            "0:v:0",
            "-map",
            "0:a:0?",
            "-c",
            "copy",
            "-movflags",
            "+faststart",
            str(combined),
        ])
        probe = _probe(combined)
        if int(probe.get("nb_read_frames", -1)) != output_count:
            raise RuntimeError(
                "incremental preview frame count mismatch: "
                f"expected {output_count}, got {probe.get('nb_read_frames')!r}"
            )
        if abs(float(probe.get("start_time", 0.0))) > 0.5 / frame_rate:
            raise RuntimeError(
                "incremental preview video timeline does not start at zero: "
                f"{probe.get('start_time')!r}"
            )
        with combined.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(combined, destination_path)

    return {
        "mechanism": "packet_preserving_visible_suffix_append_v1",
        "source_frames": source_count,
        "context_frames_trimmed": context_count,
        "visible_frames_appended": visible_count,
        "output_frames": output_count,
        "prior_preview_reencoded": False,
        "cumulative_vae_decode": False,
        "width": int(probe["width"]),
        "height": int(probe["height"]),
        "fps": frame_rate,
    }
