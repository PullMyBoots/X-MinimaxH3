"""Window-local Audio-VAE output assembly for bounded long video.

Every causal H3 window owns a complete Audio-VAE decode domain, including its
hidden overlap. Cropping after waveform decode preserves that domain; joining
independently completed audio latents before the temporal decoder does not.
"""

from __future__ import annotations

import math
from typing import Any, Iterable


def _as_cpu_stereo(waveform: Any) -> Any:
    import torch

    if not isinstance(waveform, torch.Tensor) or waveform.ndim != 3:
        raise ValueError("decoded audio window must be [B,C,S]")
    if waveform.shape[0] != 1 or waveform.shape[1] != 2:
        raise ValueError("decoded audio window must contain one stereo batch")
    value = waveform.detach().to(device="cpu").contiguous()
    if value.shape[-1] <= 0 or not bool(torch.isfinite(value).all().item()):
        raise ValueError("decoded audio window must be finite and non-empty")
    return value


def _mid(waveform: Any) -> Any:
    return waveform.detach().float().mean(dim=(0, 1))


def _correlation(left: Any, right: Any) -> float:
    import torch

    length = min(int(left.numel()), int(right.numel()))
    if length < 8:
        return 0.0
    left = left[:length] - left[:length].mean()
    right = right[:length] - right[:length].mean()
    denominator = torch.sqrt(
        torch.sum(left.square()) * torch.sum(right.square())
    ).clamp_min(1.0e-12)
    value = float((torch.sum(left * right) / denominator).item())
    return max(-1.0, min(1.0, value)) if math.isfinite(value) else 0.0


def _fade_milliseconds(previous: Any, incoming: Any, *, cut: int, rate: int) -> int:
    import torch

    probe = max(8, int(round(rate * 0.020)))
    if previous.shape[-1] < probe or cut < probe:
        return 10
    old = _mid(previous)[-probe:]
    new = _mid(incoming)[cut - probe : cut]
    rms = torch.sqrt(torch.mean(torch.cat((old, new)).square())).clamp_min(1.0e-6)
    transient = max(
        float(torch.mean(torch.abs(torch.diff(old))).item()),
        float(torch.mean(torch.abs(torch.diff(new))).item()),
    ) / float(rms.item())
    if transient > 0.80:
        return 10
    if transient > 0.35:
        return 20
    if transient > 0.15:
        return 40
    return 60


def _guarded_seam_patch(
    previous: Any,
    incoming: Any,
    *,
    cut_sample: int,
    sample_rate: int,
) -> tuple[Any | None, dict[str, Any]]:
    """Build a short pre-boundary patch only when overlap audio correlates."""

    import torch

    cut = int(cut_sample)
    rate = int(sample_rate)
    maximum_lag = max(1, int(round(rate * 0.020)))
    compare = min(
        max(8, int(round(rate * 0.080))),
        int(previous.shape[-1]),
        max(0, cut - maximum_lag),
    )
    jump_before = float(
        torch.mean(torch.abs(previous[..., -1] - incoming[..., cut])).item()
    )
    profile: dict[str, Any] = {
        "policy": "correlated_hidden_overlap_equal_power_v1",
        "applied": False,
        "correlation_before": 0.0,
        "correlation_after": 0.0,
        "offset_samples": 0,
        "crossfade_samples": 0,
        "boundary_jump_before": jump_before,
        "boundary_jump_after": jump_before,
    }
    if rate <= 0 or compare < 8:
        profile["reason"] = "insufficient_overlap"
        return None, profile

    previous_tail = _mid(previous)[-compare:]
    incoming_mid = _mid(incoming)

    def correlation_at(lag: int) -> float | None:
        stop = cut + int(lag)
        start = stop - compare
        if start < 0 or stop > incoming_mid.shape[-1]:
            return None
        return _correlation(previous_tail, incoming_mid[start:stop])

    baseline = correlation_at(0)
    if baseline is None:
        profile["reason"] = "cut_outside_waveform"
        return None, profile
    best_lag = 0
    best = baseline
    for lag in range(-maximum_lag, maximum_lag + 1):
        candidate = correlation_at(lag)
        if candidate is not None and candidate > best:
            best_lag, best = lag, candidate
    profile["correlation_before"] = float(baseline)
    profile["correlation_after"] = float(best)
    if best < 0.20 or best - baseline < 0.02:
        profile["reason"] = "overlap_not_correlated"
        return None, profile

    center = max(0, min(int(incoming.shape[-1]) - 1, cut + best_lag))
    radius = max(1, int(round(rate * 0.001)))
    search_start = max(0, center - radius)
    search_stop = min(int(incoming.shape[-1]), center + radius + 1)
    if search_stop > search_start:
        best_lag = (
            int(torch.argmin(torch.abs(incoming_mid[search_start:search_stop])).item())
            + search_start
            - cut
        )

    fade_samples = min(
        int(round(rate * _fade_milliseconds(
            previous, incoming, cut=cut, rate=rate
        ) / 1000.0)),
        int(previous.shape[-1]),
        cut,
    )
    if fade_samples < 2:
        profile["reason"] = "fade_too_short"
        return None, profile

    progress = torch.linspace(0.0, 1.0, fade_samples, dtype=torch.float32)
    base = torch.arange(cut - fade_samples, cut, dtype=torch.long)
    offsets = torch.round(float(best_lag) * (1.0 - progress)).to(torch.long)
    indices = (base + offsets).clamp(0, int(incoming.shape[-1]) - 1)
    current = incoming.index_select(-1, indices).float()
    prior = previous[..., -fade_samples:].float()

    prior_rms = torch.sqrt(torch.mean(prior.square())).clamp_min(1.0e-6)
    current_rms = torch.sqrt(torch.mean(current.square())).clamp_min(1.0e-6)
    requested_db = 20.0 * math.log10(float((prior_rms / current_rms).item()))
    level_db = max(-1.5, min(1.5, requested_db)) if abs(requested_db) <= 6.0 else 0.0
    level_gain = 10.0 ** (level_db / 20.0)
    requested_dc = float((prior.mean() - current.mean()).item())
    dc_bias = max(-0.02, min(0.02, requested_dc)) if abs(requested_dc) <= 0.10 else 0.0
    current = current * level_gain + dc_bias

    shape = [1] * (prior.ndim - 1) + [fade_samples]
    fade_in = torch.sin(progress * (math.pi / 2.0)).reshape(shape)
    fade_out = torch.cos(progress * (math.pi / 2.0)).reshape(shape)
    patch = prior * fade_out + current * fade_in
    input_peak = max(
        float(prior.abs().max().item()),
        float(current.abs().max().item()),
        1.0e-6,
    )
    patch_peak = float(patch.abs().max().item())
    if patch_peak > input_peak * 1.05:
        patch.mul_((input_peak * 1.05) / patch_peak)
    patch = patch.to(dtype=previous.dtype).contiguous()
    jump_after = float(
        torch.mean(torch.abs(patch[..., -1] - incoming[..., cut])).item()
    )
    profile.update({
        "applied": True,
        "offset_samples": int(best_lag),
        "crossfade_samples": int(fade_samples),
        "boundary_jump_after": jump_after,
        "level_gain": float(level_gain),
        "dc_bias": float(dc_bias),
        "reason": None,
    })
    return patch, profile


def assemble_window_decoded_audio(
    decoded_windows: Iterable[Any],
    window_clocks: Iterable[tuple[int, int]],
    *,
    output_frames: int,
    fps: int,
    sample_rate: int,
) -> tuple[Any, dict[str, Any]]:
    """Crop decoded overlap and assemble one exact cumulative PCM clock.

    ``window_clocks`` contains ``(context_frames, visible_frames)`` pairs. No
    Audio-VAE latent is interpolated, mixed, or decoded across a window seam.
    """

    import torch

    windows = [_as_cpu_stereo(item) for item in decoded_windows]
    clocks = [(int(context), int(visible)) for context, visible in window_clocks]
    if not windows or len(windows) != len(clocks):
        raise ValueError("decoded audio windows and clocks must align")
    rate = int(sample_rate)
    frame_rate = int(fps)
    target_frames = int(output_frames)
    if min(rate, frame_rate, target_frames) <= 0:
        raise ValueError("audio output clock must be positive")
    if any(context < 0 or visible <= 0 for context, visible in clocks):
        raise ValueError("audio window clocks must be positive")
    if sum(visible for _, visible in clocks) != target_frames:
        raise ValueError("audio window visible clocks missed the output frame count")
    if any(item.dtype != windows[0].dtype for item in windows[1:]):
        raise ValueError("decoded audio window dtype changed")

    total_samples = int(round(target_frames / frame_rate * rate))
    output = torch.empty(
        (1, 2, total_samples), dtype=windows[0].dtype, device="cpu"
    )
    frame_cursor = 0
    records: list[dict[str, Any]] = []
    latent_hop_samples = max(1, int(math.ceil(rate / 40.0)))
    for index, (waveform, (context_frames, visible_frames)) in enumerate(
        zip(windows, clocks)
    ):
        decoded_samples = int(waveform.shape[-1])
        sample_start = int(round(frame_cursor / frame_rate * rate))
        frame_cursor += visible_frames
        sample_stop = int(round(frame_cursor / frame_rate * rate))
        nominal_trim_samples = int(round(context_frames / frame_rate * rate))
        trim_samples = nominal_trim_samples
        copy_samples = sample_stop - sample_start
        source_stop = trim_samples + copy_samples
        hidden_clock_compensation_samples = max(
            0, source_stop - int(waveform.shape[-1])
        )
        borrowed_next_overlap_samples = 0
        clock_compensation_policy = None
        # Video frames live on a 24-Hz clock while native audio latents live on
        # a 40-Hz clock. Some legal window lengths therefore decode up to one
        # audio hop shorter than ``context + visible`` after independent
        # rounding (260 video frames are short by 267 samples at 32 kHz). Use
        # that tiny amount from the already-generated hidden preroll instead
        # of stretching the visible waveform or padding its terminal edge.
        if hidden_clock_compensation_samples:
            if hidden_clock_compensation_samples > latent_hop_samples:
                raise ValueError(
                    "decoded audio window is shorter than its visible frame "
                    f"clock: window={index}, decoded={waveform.shape[-1]}, "
                    f"required={source_stop}, shortfall="
                    f"{hidden_clock_compensation_samples}"
                )
            if trim_samples >= hidden_clock_compensation_samples:
                trim_samples -= hidden_clock_compensation_samples
                source_stop = trim_samples + copy_samples
                clock_compensation_policy = "hidden_preroll_crop_shift_v1"
            elif index + 1 < len(windows) and clocks[index + 1][0] > 0:
                # The cumulative source window has no hidden preroll of its
                # own. Its successor does: the next window's carried overlap
                # is the same story time as this short terminal interval. Fill
                # the sub-hop rounding deficit from that aligned overlap. The
                # existing guarded seam patch then owns the audible boundary.
                next_trim = int(round(
                    clocks[index + 1][0] / frame_rate * rate
                ))
                next_waveform = windows[index + 1]
                if (
                    next_trim < hidden_clock_compensation_samples
                    or int(next_waveform.shape[-1]) < next_trim
                ):
                    raise ValueError(
                        "next decoded audio overlap cannot cover the prior "
                        f"clock deficit: window={index}, shortfall="
                        f"{hidden_clock_compensation_samples}"
                    )
                borrowed_next_overlap_samples = hidden_clock_compensation_samples
                waveform = torch.cat((
                    waveform,
                    next_waveform[..., next_trim - borrowed_next_overlap_samples:next_trim],
                ), dim=-1).contiguous()
                clock_compensation_policy = "next_window_hidden_overlap_fill_v1"
            else:
                raise ValueError(
                    "decoded audio window is shorter than its visible frame "
                    f"clock: window={index}, decoded={waveform.shape[-1]}, "
                    f"required={source_stop}, shortfall="
                    f"{hidden_clock_compensation_samples}"
                )
        if source_stop > waveform.shape[-1]:
            raise ValueError(
                "decoded audio window is shorter than its visible frame "
                f"clock after compensation: window={index}, "
                f"decoded={waveform.shape[-1]}, required={source_stop}"
            )

        seam_profile: dict[str, Any] | None = None
        if index:
            patch, seam_profile = _guarded_seam_patch(
                output[..., :sample_start],
                waveform,
                cut_sample=trim_samples,
                sample_rate=rate,
            )
            if patch is not None:
                patch_start = sample_start - int(patch.shape[-1])
                output[..., patch_start:sample_start].copy_(patch)
        output[..., sample_start:sample_stop].copy_(
            waveform[..., trim_samples:source_stop]
        )
        records.append({
            "window_index": index,
            "context_frames": context_frames,
            "visible_frames": visible_frames,
            "decoded_samples": decoded_samples,
            "nominal_trim_samples": nominal_trim_samples,
            "trim_samples": trim_samples,
            "hidden_clock_compensation_samples": (
                hidden_clock_compensation_samples
            ),
            "borrowed_next_overlap_samples": borrowed_next_overlap_samples,
            "clock_compensation_policy": clock_compensation_policy,
            "copied_samples": copy_samples,
            "sample_start": sample_start,
            "sample_stop": sample_stop,
            "latent_interpolation": False,
            "seam": seam_profile,
        })

    return output.contiguous(), {
        "policy": "window_local_audio_vae_pcm_overlap_save_v1",
        "temporal_vae_domains": len(windows),
        "sample_rate": rate,
        "output_samples": total_samples,
        "cumulative_frame_clock": True,
        "audio_latent_interpolation": False,
        "audio_latent_overlap_add": False,
        "window_records": records,
    }


__all__ = ["assemble_window_decoded_audio"]
