"""Hard temporal speech authority for structured long-video generation.

MiniMax H3 generates audio and video jointly.  Removing reference-audio rows
from a continuation window therefore prevents voice *copying*, but it cannot
prevent the model from synthesising a fresh voice from visual/text semantics.
This module supplies the missing output-side contract: generated windows that
own no explicit ``<d>...</d>`` event may keep ambience, music and effects, but
their separated vocal stem is not allowed into the published soundtrack.

The policy is deliberately structural.  It consumes only the window planner's
dialogue ownership and global clocks; it never searches for a person, language,
word, action or story-specific phrase.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
from typing import Any, Iterable

from .long_horizon import FPS, LongHorizonPlan


@dataclass(frozen=True, slots=True)
class SpeechAuthorityGateConfig:
    """Runtime configuration for the optional neural vocal residual gate."""

    python_executable: Path
    model: str = "htdemucs"
    device: str = "cuda"
    fade_seconds: float = 0.100
    timeout_seconds: float = 300.0

    @classmethod
    def from_environment(cls) -> "SpeechAuthorityGateConfig":
        executable = os.environ.get("H3_SPEECH_GATE_PYTHON", "").strip()
        if not executable:
            raise RuntimeError(
                "H3_LONG_STRUCTURED_SPEECH_GATE requires "
                "H3_SPEECH_GATE_PYTHON to point to a Python environment "
                "containing Demucs"
            )
        return cls(
            python_executable=Path(executable),
            model=(
                os.environ.get("H3_SPEECH_GATE_MODEL", "htdemucs").strip()
                or "htdemucs"
            ),
            device=(
                os.environ.get("H3_SPEECH_GATE_DEVICE", "cuda").strip()
                or "cuda"
            ),
            fade_seconds=float(
                os.environ.get("H3_SPEECH_GATE_FADE_SECONDS", "0.100")
            ),
            timeout_seconds=float(
                os.environ.get("H3_SPEECH_GATE_TIMEOUT_SECONDS", "300")
            ),
        )

    def validate(self) -> None:
        if not self.python_executable.is_file():
            raise RuntimeError(
                f"speech-gate Python does not exist: {self.python_executable}"
            )
        if not self.model.strip():
            raise ValueError("speech-gate model cannot be empty")
        if not self.device.strip():
            raise ValueError("speech-gate device cannot be empty")
        if not 0.0 <= float(self.fade_seconds) <= 1.0:
            raise ValueError("speech-gate fade must be between zero and one second")
        if float(self.timeout_seconds) <= 0.0:
            raise ValueError("speech-gate timeout must be positive")
        if shutil.which("ffmpeg") is None:
            raise RuntimeError("speech authority requires ffmpeg")


def _merge_intervals(
    intervals: Iterable[tuple[float, float]],
) -> tuple[tuple[float, float], ...]:
    ordered = sorted(
        (max(0.0, float(start)), max(0.0, float(stop)))
        for start, stop in intervals
        if float(stop) > float(start)
    )
    merged: list[list[float]] = []
    for start, stop in ordered:
        if not merged or start > merged[-1][1] + 1e-9:
            merged.append([start, stop])
        else:
            merged[-1][1] = max(merged[-1][1], stop)
    return tuple((start, stop) for start, stop in merged)


def forbidden_speech_intervals(
    plan: LongHorizonPlan,
) -> tuple[tuple[float, float], ...]:
    """Return global output intervals owned by zero-dialogue windows.

    ``None`` means the user supplied a free prompt without an explicit speech
    contract and is therefore never gated.  Only a compiled numeric zero is a
    denial of vocal authority.
    """

    if not plan.structured_director:
        return ()
    return _merge_intervals(
        (
            segment.visible_start_frame / FPS,
            segment.visible_stop_frame / FPS,
        )
        for segment in plan.segments
        if segment.authorized_dialogue_count == 0
    )


def _interval_envelope(
    start: float,
    stop: float,
    *,
    fade_seconds: float,
) -> str:
    """Build one FFmpeg expression that is one over the denied interval."""

    fade = max(0.0, float(fade_seconds))
    start = max(0.0, float(start))
    stop = max(start, float(stop))
    if fade <= 1e-9:
        return f"between(t,{start:.6f},{stop:.6f})"
    fade_in = max(0.0, start - fade)
    fade_out = stop + fade
    if start - fade_in <= 1e-9:
        opening = "1"
    else:
        opening = (
            f"(t-{fade_in:.6f})/{(start - fade_in):.6f}"
        )
    return (
        f"if(lt(t,{fade_in:.6f}),0,"
        f"if(lt(t,{start:.6f}),{opening},"
        f"if(lt(t,{stop:.6f}),1,"
        f"if(lt(t,{fade_out:.6f}),"
        f"({fade_out:.6f}-t)/{fade:.6f},0))))"
    )


def vocal_gate_expression(
    intervals: Iterable[tuple[float, float]],
    *,
    fade_seconds: float,
) -> str:
    """Build a bounded union envelope for one or more denied intervals."""

    items = _merge_intervals(intervals)
    if not items:
        return "0"
    expressions = [
        _interval_envelope(start, stop, fade_seconds=fade_seconds)
        for start, stop in items
    ]
    value = expressions[0]
    for expression in expressions[1:]:
        value = f"max({value},{expression})"
    return value


def _run_checked(
    command: list[str],
    *,
    timeout_seconds: float,
) -> None:
    completed = subprocess.run(
        command,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout_seconds,
    )
    if completed.returncode:
        tail = completed.stdout[-4000:]
        raise RuntimeError(
            "structured speech-authority subprocess failed "
            f"({completed.returncode}):\n{tail}"
        )


def apply_vocal_residual_gate(
    media_path: Path,
    intervals: Iterable[tuple[float, float]],
    *,
    config: SpeechAuthorityGateConfig,
) -> dict[str, Any]:
    """Atomically suppress separated vocals only in denied time intervals."""

    destination = Path(media_path).resolve()
    if not destination.is_file():
        raise ValueError(f"speech-gate source does not exist: {destination}")
    denied = _merge_intervals(intervals)
    if not denied:
        return {
            "policy": "structured_zero_dialogue_vocal_residual_gate_v1",
            "applied": False,
            "forbidden_intervals_seconds": [],
        }
    config.validate()
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:  # Kept after validate for type narrowing.
        raise RuntimeError("speech authority requires ffmpeg")

    started = time.monotonic()
    with tempfile.TemporaryDirectory(
        prefix=f".{destination.stem}.speech-authority-",
        dir=destination.parent,
    ) as temporary_root:
        temporary = Path(temporary_root)
        source_audio = temporary / "source.wav"
        separated_root = temporary / "separated"
        gated_audio = temporary / "gated.wav"
        remuxed = temporary / "published.mp4"

        _run_checked(
            [
                ffmpeg,
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(destination),
                "-vn",
                "-ac",
                "2",
                "-ar",
                "32000",
                "-c:a",
                "pcm_f32le",
                str(source_audio),
            ],
            timeout_seconds=config.timeout_seconds,
        )
        _run_checked(
            [
                str(config.python_executable),
                "-m",
                "demucs.separate",
                "-n",
                config.model,
                "--two-stems",
                "vocals",
                "--shifts",
                "0",
                "--overlap",
                "0.1",
                "-d",
                config.device,
                "-o",
                str(separated_root),
                str(source_audio),
            ],
            timeout_seconds=config.timeout_seconds,
        )
        non_vocal = (
            separated_root
            / config.model
            / source_audio.stem
            / "no_vocals.wav"
        )
        if not non_vocal.is_file():
            raise RuntimeError(
                f"speech separator did not produce a non-vocal stem: {non_vocal}"
            )

        envelope = vocal_gate_expression(
            denied,
            fade_seconds=config.fade_seconds,
        )
        filter_graph = (
            f"[0:a]aresample=32000,volume='1-({envelope})':eval=frame[a0];"
            f"[1:a]aresample=32000,volume='{envelope}':eval=frame[a1];"
            "[a0][a1]amix=inputs=2:normalize=0:dropout_transition=0,"
            "aformat=sample_fmts=flt:sample_rates=32000:channel_layouts=stereo[a]"
        )
        _run_checked(
            [
                ffmpeg,
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(source_audio),
                "-i",
                str(non_vocal),
                "-filter_complex",
                filter_graph,
                "-map",
                "[a]",
                "-c:a",
                "pcm_f32le",
                str(gated_audio),
            ],
            timeout_seconds=config.timeout_seconds,
        )
        _run_checked(
            [
                ffmpeg,
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(destination),
                "-i",
                str(gated_audio),
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                "-ar",
                "32000",
                "-ac",
                "2",
                "-shortest",
                "-movflags",
                "+faststart",
                str(remuxed),
            ],
            timeout_seconds=config.timeout_seconds,
        )
        if not remuxed.is_file() or remuxed.stat().st_size <= 0:
            raise RuntimeError("speech authority produced no publishable media")
        os.replace(remuxed, destination)

    return {
        "policy": "structured_zero_dialogue_vocal_residual_gate_v1",
        "applied": True,
        "separator": "demucs",
        "separator_model": config.model,
        "device": config.device,
        "fade_seconds": config.fade_seconds,
        "forbidden_intervals_seconds": [list(item) for item in denied],
        "prompt_content_inspected": False,
        "elapsed_seconds": time.monotonic() - started,
    }


__all__ = [
    "SpeechAuthorityGateConfig",
    "apply_vocal_residual_gate",
    "forbidden_speech_intervals",
    "vocal_gate_expression",
]
