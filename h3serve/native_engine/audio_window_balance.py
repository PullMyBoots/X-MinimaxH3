"""Conservative PCM level matching across creator-window boundaries."""

from __future__ import annotations

import json
import math
import os
import subprocess
from pathlib import Path
from typing import Any, Iterable


def _window_level_db(
    audio: Any,
    start: int,
    stop: int,
    *,
    sample_rate: int,
) -> float | None:
    """Estimate the persistent local bed while ignoring speech transients."""

    import numpy as np

    value = np.asarray(audio[start:stop], dtype=np.float32)
    block = max(64, int(round(sample_rate * 0.025)))
    block_count = int(value.shape[0]) // block
    if block_count < 4:
        return None
    value = value[: block_count * block].reshape(block_count, block, -1)
    rms = np.sqrt(np.mean(np.square(value, dtype=np.float64), axis=(1, 2)))
    finite = rms[np.isfinite(rms) & (rms >= 10.0 ** (-65.0 / 20.0))]
    if finite.size < 4:
        return None
    # The 35th percentile follows the continuous room/vehicle bed instead of
    # a nearby spoken syllable or one isolated impact.
    level = float(np.percentile(finite, 35.0))
    return 20.0 * math.log10(max(level, 1.0e-12))


def balance_creator_window_pcm(
    audio: Any,
    boundary_samples: Iterable[int],
    *,
    sample_rate: int,
    probe_seconds: float = 0.9,
    guard_seconds: float = 0.08,
    transition_seconds: float = 0.30,
    maximum_adjustment_db: float = 3.0,
    maximum_match_delta_db: float = 12.0,
    maximum_absolute_gain_db: float = 4.0,
) -> tuple[Any, dict[str, Any]]:
    """Match local levels with a bounded smooth gain envelope.

    Boundaries are the visible creator-window cuts, not the later SelfLift
    denoise views. Large changes and near-silence are treated as authored
    dynamics and left untouched.
    """

    import numpy as np

    value = np.asarray(audio, dtype=np.float32)
    if value.ndim != 2 or value.shape[1] not in (1, 2):
        raise ValueError("decoded PCM must have shape [samples, channels]")
    rate = int(sample_rate)
    if rate <= 0 or value.shape[0] <= 0 or not bool(np.isfinite(value).all()):
        raise ValueError("decoded PCM must be finite and non-empty")
    boundaries = sorted({int(item) for item in boundary_samples})
    if any(item <= 0 or item >= value.shape[0] for item in boundaries):
        raise ValueError("creator-window boundary falls outside the PCM clock")

    probe = max(1, int(round(rate * float(probe_seconds))))
    guard = max(0, int(round(rate * float(guard_seconds))))
    relative_gain_db = [0.0]
    records: list[dict[str, Any]] = []
    for index, boundary in enumerate(boundaries, start=1):
        previous_level = _window_level_db(
            value,
            max(0, boundary - guard - probe),
            max(0, boundary - guard),
            sample_rate=rate,
        )
        incoming_level = _window_level_db(
            value,
            min(value.shape[0], boundary + guard),
            min(value.shape[0], boundary + guard + probe),
            sample_rate=rate,
        )
        requested = None
        applied = 0.0
        reason = None
        if previous_level is None or incoming_level is None:
            reason = "insufficient_persistent_audio"
        else:
            requested = previous_level + relative_gain_db[-1] - incoming_level
            if abs(requested) > float(maximum_match_delta_db):
                reason = "authored_or_large_level_change"
            else:
                applied = max(
                    -float(maximum_adjustment_db),
                    min(float(maximum_adjustment_db), requested),
                )
                applied = max(
                    -float(maximum_absolute_gain_db),
                    min(float(maximum_absolute_gain_db), applied),
                )
        relative_gain_db.append(float(applied))
        records.append({
            "boundary_index": index,
            "boundary_sample": boundary,
            "boundary_seconds": boundary / rate,
            "previous_level_dbfs": previous_level,
            "incoming_level_dbfs": incoming_level,
            "requested_gain_db": requested,
            "relative_gain_db": float(applied),
            "reason": reason,
        })

    # Apply only the bounded relative correction. Preserve the existing level
    # of unaffected windows; a whole-film peak guard below handles the rare
    # case where the quieter window did not actually have enough headroom.
    center_db = 0.0
    target_gain_db = [
        max(-float(maximum_absolute_gain_db),
            min(float(maximum_absolute_gain_db), gain))
        for gain in relative_gain_db
    ]

    segment_starts = [0, *boundaries]
    segment_stops = [*boundaries, int(value.shape[0])]
    envelope_db = np.empty(value.shape[0], dtype=np.float32)
    for start, stop, gain_db in zip(
        segment_starts, segment_stops, target_gain_db
    ):
        envelope_db[start:stop] = float(gain_db)

    transition = max(2, int(round(rate * float(transition_seconds))))
    half = transition // 2
    for index, boundary in enumerate(boundaries, start=1):
        start = max(0, boundary - half)
        stop = min(value.shape[0], boundary + half)
        if stop - start < 2:
            continue
        progress = np.linspace(0.0, 1.0, stop - start, dtype=np.float32)
        smooth = progress * progress * (3.0 - 2.0 * progress)
        envelope_db[start:stop] = (
            target_gain_db[index - 1] * (1.0 - smooth)
            + target_gain_db[index] * smooth
        )
    envelope = np.power(10.0, envelope_db / 20.0, dtype=np.float32)
    balanced = value * envelope[:, None]
    peak_before_guard = float(np.max(np.abs(balanced)))
    peak_guard_gain = 1.0
    if peak_before_guard > 0.98:
        peak_guard_gain = 0.98 / peak_before_guard
        balanced *= peak_guard_gain
    peak_guard_db = 20.0 * math.log10(max(peak_guard_gain, 1.0e-12))
    for index, record in enumerate(records, start=1):
        record["applied_gain_db"] = float(target_gain_db[index] + peak_guard_db)
    return np.ascontiguousarray(balanced, dtype=np.float32), {
        "policy": "creator_window_local_bed_gain_v1",
        "sample_rate": rate,
        "window_count": len(target_gain_db),
        "boundary_count": len(boundaries),
        "relative_gain_db": relative_gain_db,
        "centering_db": float(center_db),
        "target_gain_db": target_gain_db,
        "probe_seconds": float(probe_seconds),
        "guard_seconds": float(guard_seconds),
        "transition_seconds": float(transition_seconds),
        "maximum_adjustment_db": float(maximum_adjustment_db),
        "maximum_match_delta_db": float(maximum_match_delta_db),
        "peak_guard_gain": float(peak_guard_gain),
        "peak_guard_db": float(peak_guard_db),
        "records": records,
    }


def balance_encoded_creator_windows(
    video_path: Path,
    frame_ranges: Iterable[tuple[int, int]],
    *,
    fps: int,
    sample_rate: int = 32_000,
) -> dict[str, Any]:
    """Balance creator windows while stream-copying the encoded video track."""

    import numpy as np

    target = Path(video_path).resolve()
    ranges = [(int(start), int(stop)) for start, stop in frame_ranges]
    if len(ranges) <= 1:
        return {"policy": "creator_window_local_bed_gain_v1", "applied": False,
                "reason": "single_window"}
    if not target.is_file() or int(fps) <= 0:
        raise ValueError("encoded video and positive fps are required")
    for index, (start, stop) in enumerate(ranges):
        if start < 0 or stop <= start:
            raise ValueError("creator-window frame range is invalid")
        if index and start != ranges[index - 1][1]:
            raise ValueError("creator-window visible ranges must be contiguous")

    decoded = subprocess.run(
        [
            "ffmpeg", "-v", "error", "-i", str(target), "-map", "0:a:0",
            "-f", "f32le", "-acodec", "pcm_f32le", "-ar", str(sample_rate),
            "-ac", "2", "pipe:1",
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=600,
    )
    pcm = np.frombuffer(decoded.stdout, dtype="<f4")
    if pcm.size % 2:
        raise RuntimeError("decoded H3 audio has an incomplete stereo sample")
    pcm = pcm.reshape(-1, 2).copy()
    boundary_samples = [
        int(round(stop / int(fps) * sample_rate)) for _, stop in ranges[:-1]
    ]
    balanced, profile = balance_creator_window_pcm(
        pcm,
        boundary_samples,
        sample_rate=sample_rate,
    )
    temporary = target.with_name(target.stem + ".audio-balance.tmp.mp4")
    command = [
        "ffmpeg", "-y", "-v", "error", "-i", str(target),
        "-f", "f32le", "-ar", str(sample_rate), "-ac", "2", "-i", "pipe:0",
        "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac",
        "-b:a", "192k", "-shortest", "-movflags", "+faststart", str(temporary),
    ]
    try:
        encoded = subprocess.run(
            command,
            input=balanced.astype("<f4", copy=False).tobytes(),
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=600,
        )
        if not temporary.is_file() or temporary.stat().st_size <= 0:
            raise RuntimeError("window-balanced mux did not create an output")
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    profile.update({
        "applied": True,
        "video_stream": "packet_copy",
        "audio_codec": "aac",
        "audio_bit_rate": 192_000,
        "frame_ranges": [list(item) for item in ranges],
        "ffmpeg_stderr": encoded.stderr.decode("utf-8", "replace") if 'encoded' in locals() else "",
    })
    return profile


__all__ = [
    "balance_creator_window_pcm",
    "balance_encoded_creator_windows",
]
