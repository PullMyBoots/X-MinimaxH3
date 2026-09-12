"""Prompt-agnostic repair for rare harsh H3 speech residuals.

Long-horizon AV windows can occasionally leave a short, high-band residual in
otherwise intelligible speech.  Changing the joint continuation context also
changes the video trajectory, so this guard runs only after the final video is
fixed.  It detects a narrow acoustic failure signature, verifies that the
event belongs to a speech island, and projects only a small waveform excerpt
through H3's own Audio-VAE posterior mean.  No prompt text, dialogue string,
window index, seam time or story-specific rule participates in the decision.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Any

import numpy as np
from scipy.signal import resample_poly
import torch

from .adapters.sampling_mux import normalize_h3_audio_loudness


@dataclass(frozen=True, slots=True)
class AudioManifoldGuardConfig:
    """Conservative fixed policy for post-trajectory audio repair."""

    projection_rounds: int = 8
    analysis_window_seconds: float = 0.5
    analysis_hop_seconds: float = 0.25
    feather_seconds: float = 0.15
    projection_context_seconds: float = 2.0
    minimum_rms_dbfs: float = -38.0
    minimum_high_band_fraction: float = 0.65
    maximum_spectral_flatness: float = 0.15
    minimum_normalized_derivative: float = 3.0
    vad_mode: int = 3
    minimum_local_speech_fraction: float = 0.70
    maximum_global_speech_fraction: float = 0.75

    def validate(self) -> None:
        if self.projection_rounds < 1:
            raise ValueError("audio projection rounds must be positive")
        if self.analysis_window_seconds <= 0.0 or self.analysis_hop_seconds <= 0.0:
            raise ValueError("audio analysis windows must be positive")
        if not 0.0 <= self.feather_seconds <= 1.0:
            raise ValueError("audio repair feather must be between zero and one")
        if self.projection_context_seconds < self.feather_seconds:
            raise ValueError("audio projection context must cover the repair feather")
        if self.vad_mode not in (0, 1, 2, 3):
            raise ValueError("WebRTC VAD mode must lie inside [0, 3]")
        for name, value in (
            ("local speech fraction", self.minimum_local_speech_fraction),
            ("global speech fraction", self.maximum_global_speech_fraction),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"audio {name} must lie inside [0, 1]")


def _stereo_numpy(audio: Any) -> np.ndarray:
    value = np.asarray(
        audio.detach().float().cpu().numpy()
        if isinstance(audio, torch.Tensor)
        else audio,
        dtype=np.float32,
    )
    if value.ndim == 3:
        if value.shape[0] != 1:
            raise ValueError("audio manifold guard supports one batch")
        value = value[0]
    if value.ndim != 2:
        raise ValueError("audio manifold guard expects a stereo waveform")
    if value.shape[0] == 2:
        pass
    elif value.shape[1] == 2:
        value = value.T
    else:
        raise ValueError("audio manifold guard expects exactly two channels")
    if not np.isfinite(value).all():
        raise ValueError("audio manifold guard received non-finite samples")
    return np.ascontiguousarray(value)


def _acoustic_risk_blocks(
    stereo: np.ndarray,
    sample_rate: int,
    config: AudioManifoldGuardConfig,
) -> list[dict[str, float]]:
    mono = np.mean(stereo, axis=0, dtype=np.float64)
    window_samples = max(256, round(sample_rate * config.analysis_window_seconds))
    hop_samples = max(128, round(sample_rate * config.analysis_hop_seconds))
    taper = np.hanning(window_samples)
    frequencies = np.fft.rfftfreq(window_samples, d=1.0 / sample_rate)
    audible = (frequencies >= 80.0) & (frequencies <= 15_000.0)
    high = (frequencies >= 4_000.0) & (frequencies <= 12_000.0)
    flatness_band = (frequencies >= 300.0) & (frequencies <= 12_000.0)
    risks: list[dict[str, float]] = []
    for start in range(0, max(1, len(mono) - window_samples + 1), hop_samples):
        block = mono[start : start + window_samples]
        if len(block) < window_samples:
            block = np.pad(block, (0, window_samples - len(block)))
        block = block - float(np.mean(block))
        rms = float(np.sqrt(np.mean(np.square(block))))
        spectrum = np.fft.rfft(block * taper)
        power = np.square(np.abs(spectrum)) + 1e-18
        audible_power = float(np.sum(power[audible]))
        high_fraction = float(np.sum(power[high]) / max(audible_power, 1e-18))
        flat_power = power[flatness_band]
        flatness = float(
            np.exp(np.mean(np.log(flat_power)))
            / max(float(np.mean(flat_power)), 1e-18)
        )
        derivative = np.abs(np.diff(block))
        normalized_derivative = (
            float(np.percentile(derivative, 99)) / max(rms, 1e-9)
            if derivative.size
            else 0.0
        )
        rms_dbfs = 20.0 * math.log10(max(rms, 1e-12))
        if (
            rms_dbfs >= config.minimum_rms_dbfs
            and high_fraction >= config.minimum_high_band_fraction
            and flatness <= config.maximum_spectral_flatness
            and normalized_derivative >= config.minimum_normalized_derivative
        ):
            risks.append({
                "start_seconds": start / sample_rate,
                "stop_seconds": min(len(mono), start + window_samples) / sample_rate,
                "rms_dbfs": rms_dbfs,
                "high_band_fraction": high_fraction,
                "spectral_flatness": flatness,
                "normalized_derivative_p99": normalized_derivative,
            })
    return risks


def _vad_flags(stereo: np.ndarray, sample_rate: int, mode: int) -> np.ndarray:
    try:
        import webrtcvad
    except ImportError as error:  # pragma: no cover - installation contract
        raise RuntimeError(
            "audio manifold guard requires the webrtcvad-wheels dependency"
        ) from error

    mono = np.mean(stereo, axis=0, dtype=np.float64)
    if sample_rate != 16_000:
        common = math.gcd(int(sample_rate), 16_000)
        mono = resample_poly(mono, 16_000 // common, sample_rate // common)
    pcm = np.asarray(
        np.clip(mono, -1.0, 1.0) * 32767.0,
        dtype="<i2",
    )
    frame_samples = 480  # 30 ms at 16 kHz, accepted by WebRTC VAD.
    vad = webrtcvad.Vad(mode)
    return np.asarray(
        [
            vad.is_speech(pcm[start : start + frame_samples].tobytes(), 16_000)
            for start in range(0, len(pcm) - frame_samples + 1, frame_samples)
        ],
        dtype=bool,
    )


def _speech_fraction(
    flags: np.ndarray,
    start_seconds: float,
    stop_seconds: float,
) -> float:
    if not len(flags):
        return 0.0
    start = max(0, math.floor(start_seconds / 0.03))
    stop = min(len(flags), max(start + 1, math.ceil(stop_seconds / 0.03)))
    return float(np.mean(flags[start:stop])) if stop > start else 0.0


def detect_audio_manifold_risks(
    audio: Any,
    *,
    sample_rate: int = 32_000,
    config: AudioManifoldGuardConfig = AudioManifoldGuardConfig(),
) -> tuple[list[dict[str, float]], dict[str, Any]]:
    """Return only acoustic risks that belong to a credible speech island."""

    config.validate()
    stereo = _stereo_numpy(audio)
    acoustic = _acoustic_risk_blocks(stereo, sample_rate, config)
    if not acoustic:
        return [], {
            "acoustic_risk_blocks": 0,
            "vad_evaluated": False,
            "accepted_risk_blocks": 0,
        }
    flags = _vad_flags(stereo, sample_rate, config.vad_mode)
    global_fraction = float(np.mean(flags)) if len(flags) else 0.0
    accepted: list[dict[str, float]] = []
    for risk in acoustic:
        local_fraction = _speech_fraction(
            flags,
            risk["start_seconds"],
            risk["stop_seconds"],
        )
        enriched = {**risk, "local_speech_fraction": local_fraction}
        if (
            local_fraction >= config.minimum_local_speech_fraction
            and global_fraction <= config.maximum_global_speech_fraction
        ):
            accepted.append(enriched)
    # Keep directly overlapping acoustic blocks together.  A single speech
    # island may straddle two 250 ms analysis hops even if only one block has
    # enough VAD votes on its own.
    if accepted:
        expanded: list[dict[str, float]] = []
        for risk in acoustic:
            if any(
                risk["start_seconds"] <= item["stop_seconds"] + 1e-9
                and risk["stop_seconds"] >= item["start_seconds"] - 1e-9
                for item in accepted
            ):
                local_fraction = _speech_fraction(
                    flags,
                    risk["start_seconds"],
                    risk["stop_seconds"],
                )
                expanded.append({**risk, "local_speech_fraction": local_fraction})
        accepted = expanded
    return accepted, {
        "acoustic_risk_blocks": len(acoustic),
        "vad_evaluated": True,
        "vad": "webrtcvad_mode_3_30ms",
        "global_speech_fraction": global_fraction,
        "accepted_risk_blocks": len(accepted),
    }


def _merge_intervals(
    risks: list[dict[str, float]],
) -> list[tuple[float, float]]:
    intervals: list[list[float]] = []
    for risk in sorted(risks, key=lambda item: item["start_seconds"]):
        start = float(risk["start_seconds"])
        stop = float(risk["stop_seconds"])
        if not intervals or start > intervals[-1][1] + 1e-9:
            intervals.append([start, stop])
        else:
            intervals[-1][1] = max(intervals[-1][1], stop)
    return [(start, stop) for start, stop in intervals]


def _projection_excerpt(
    model: Any,
    waveform: torch.Tensor,
    *,
    rounds: int,
) -> torch.Tensor:
    target_samples = int(waveform.shape[-1])
    projected = waveform
    with torch.inference_mode():
        for _ in range(rounds):
            latent = model.encode(
                projected.reshape(2, 1, projected.shape[-1]),
                return_cpu=False,
            )
            projected = model.decode(
                latent,
                stereo_batch=True,
                return_cpu=False,
            )
            if projected.shape[-1] > target_samples:
                projected = projected[..., :target_samples]
            elif projected.shape[-1] < target_samples:
                projected = torch.nn.functional.pad(
                    projected, (0, target_samples - projected.shape[-1])
                )
    return projected


def _feathered_mask(
    samples: int,
    sample_rate: int,
    intervals: list[tuple[float, float]],
    feather_seconds: float,
) -> np.ndarray:
    mask = np.zeros(samples, dtype=np.float32)
    feather = max(1, round(sample_rate * feather_seconds))
    for start_seconds, stop_seconds in intervals:
        start = max(0, round(start_seconds * sample_rate))
        stop = min(samples, round(stop_seconds * sample_rate))
        left = max(0, start - feather)
        right = min(samples, stop + feather)
        mask[start:stop] = 1.0
        if start > left:
            phase = np.linspace(0.0, math.pi, start - left, endpoint=False)
            mask[left:start] = np.maximum(
                mask[left:start], (1.0 - np.cos(phase)) * 0.5
            )
        if right > stop:
            phase = np.linspace(0.0, math.pi, right - stop, endpoint=False)
            mask[stop:right] = np.maximum(
                mask[stop:right], (1.0 + np.cos(phase)) * 0.5
            )
    return mask


def apply_audio_manifold_guard(
    model: Any,
    decoded_audio: Any,
    *,
    sample_rate: int = 32_000,
    config: AudioManifoldGuardConfig = AudioManifoldGuardConfig(),
) -> tuple[Any, dict[str, Any]]:
    """Repair accepted residuals while leaving every other sample untouched."""

    started = time.monotonic()
    normalized = normalize_h3_audio_loudness(decoded_audio)
    stereo = _stereo_numpy(normalized)
    risks, detection = detect_audio_manifold_risks(
        stereo, sample_rate=sample_rate, config=config
    )
    if not risks:
        return decoded_audio, {
            "policy": "speech_local_audio_vae_manifold_guard_v1",
            "applied": False,
            **detection,
            "elapsed_seconds": time.monotonic() - started,
            "prompt_or_timeline_inspected": False,
        }

    intervals = _merge_intervals(risks)
    mask = _feathered_mask(
        stereo.shape[-1],
        sample_rate,
        intervals,
        config.feather_seconds,
    )
    repaired = stereo.copy()
    for start_seconds, stop_seconds in intervals:
        crop_start = max(
            0,
            round((start_seconds - config.projection_context_seconds) * sample_rate),
        )
        crop_stop = min(
            stereo.shape[-1],
            round((stop_seconds + config.projection_context_seconds) * sample_rate),
        )
        source = torch.from_numpy(
            np.ascontiguousarray(stereo[:, crop_start:crop_stop])
        ).to("cuda:0")
        projected = _projection_excerpt(
            model, source, rounds=config.projection_rounds
        ).detach().float().cpu().numpy()
        local_mask = mask[crop_start:crop_stop][None, :]
        repaired[:, crop_start:crop_stop] = (
            stereo[:, crop_start:crop_stop] * (1.0 - local_mask)
            + projected * local_mask
        )
    output: Any = torch.from_numpy(np.ascontiguousarray(repaired))
    if isinstance(decoded_audio, torch.Tensor):
        output = output.to(dtype=decoded_audio.dtype)
        if decoded_audio.ndim == 3:
            output = output.unsqueeze(0)
    return output, {
        "policy": "speech_local_audio_vae_manifold_guard_v1",
        "applied": True,
        **detection,
        "risk_blocks": risks,
        "repair_intervals_seconds": [list(item) for item in intervals],
        "projection_rounds": config.projection_rounds,
        "projection_context_seconds": config.projection_context_seconds,
        "feather_seconds": config.feather_seconds,
        "affected_seconds": float(np.sum(mask > 1e-6) / sample_rate),
        "fully_projected_seconds": float(np.sum(mask >= 1.0) / sample_rate),
        "unchanged_sample_fraction": float(np.mean(mask == 0.0)),
        "waveform_mean_absolute_delta": float(np.mean(np.abs(repaired - stereo))),
        "elapsed_seconds": time.monotonic() - started,
        "prompt_or_timeline_inspected": False,
    }


__all__ = [
    "AudioManifoldGuardConfig",
    "apply_audio_manifold_guard",
    "detect_audio_manifold_risks",
]
