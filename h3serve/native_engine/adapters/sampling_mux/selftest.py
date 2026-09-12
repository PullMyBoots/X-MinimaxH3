"""CPU synthetic AV test for scheduler, samplers, mux, and probe."""

from __future__ import annotations

import tempfile
from pathlib import Path
from types import SimpleNamespace

from ...pipeline import GenerationInput, PipelineState, SamplingConfig
from .mux import AtomicPyAVMuxer, MuxConfig, normalize_h3_audio_loudness, probe_media
from .samplers import (
    AVPrediction,
    ResMultistepAVSampler,
    SASolverAVSampler,
    TurboAVSampler,
)
from .scheduler import H3SimpleScheduler, SamplingPlan, refinement_sigma_schedule


def run() -> None:
    import numpy as np
    import torch

    default_mux = MuxConfig()
    assert default_mux.video_codec == "libx264"
    assert default_mux.video_crf == 14
    assert default_mux.video_preset == "superfast"

    sampling = SamplingConfig(
        engine="original",
        num_steps=4,
        actual_step_indices=(0, 1, 2, 3),
        sampler="res_multistep",
    )
    request = GenerationInput(
        prompt="synthetic AV",
        width=64,
        height=64,
        num_frames=22,
        seed=4090,
        sampling=sampling,
        fps=24,
    )
    scheduler = H3SimpleScheduler(execution_device="cpu")
    first_state = PipelineState(request=request)
    second_state = PipelineState(request=request)
    first = scheduler.prepare(first_state)
    second = scheduler.prepare(second_state)
    assert torch.equal(first["video_latents"], second["video_latents"])
    assert torch.equal(first["audio_latents"], second["audio_latents"])
    assert first["video_latents"].shape == (1, 24, 7, 4, 4)
    assert first["audio_latents"].shape == (1, 32, 2, 37)
    plan = first_state.metadata["sampling_plan"]
    assert plan.step_count == 4
    assert plan.video_sigmas[0] == plan.audio_sigmas[0] == 1.0
    assert plan.video_sigmas[-1] == plan.audio_sigmas[-1] == 0.0

    def predict(video, audio, clock, *, step_index, is_actual_step):
        assert is_actual_step
        return AVPrediction(torch.zeros_like(video), torch.zeros_like(audio))

    video, audio = ResMultistepAVSampler().sample(
        first["video_latents"], first["audio_latents"], plan, predict
    )
    assert torch.isfinite(video).all() and torch.isfinite(audio).all()
    assert float(video.abs().max()) < 1e-5
    assert float(audio.abs().max()) < 1e-5
    branch_plan = SamplingPlan(
        sampler="res_multistep",
        video_sigmas=(0.75, 0.35),
        audio_sigmas=(0.75, 0.35),
        actual_step_indices=(0,),
        video_shift=12.0,
        audio_shift=3.0,
    )
    branch_input_video = torch.ones_like(first["video_latents"])
    branch_input_audio = torch.ones_like(first["audio_latents"])
    cold_video, cold_audio = ResMultistepAVSampler().sample(
        branch_input_video,
        branch_input_audio,
        branch_plan,
        predict,
    )
    warm_video, warm_audio = ResMultistepAVSampler().sample(
        branch_input_video,
        branch_input_audio,
        branch_plan,
        predict,
        initial_previous_video=torch.full_like(branch_input_video, 0.25),
        initial_previous_audio=torch.full_like(branch_input_audio, 0.25),
        initial_previous_video_sigma=0.9,
        initial_previous_audio_sigma=0.9,
    )
    assert torch.isfinite(warm_video).all() and torch.isfinite(warm_audio).all()
    assert not torch.equal(cold_video, warm_video)
    assert not torch.equal(cold_audio, warm_audio)
    try:
        ResMultistepAVSampler().sample(
            branch_input_video,
            branch_input_audio,
            branch_plan,
            predict,
            initial_previous_video=torch.zeros_like(branch_input_video),
        )
    except ValueError as error:
        assert "initial RES history" in str(error)
    else:
        raise AssertionError("partial RES warm history was accepted")
    TurboAVSampler().sample(
        first["video_latents"], first["audio_latents"], plan, predict
    )

    # Strength owns the same start sigma at every solver length; steps only
    # refine the numerical path from that point to zero.
    one_sigma = refinement_sigma_schedule(1, 0.30, 6.0)
    eight_sigmas = refinement_sigma_schedule(8, 0.30, 6.0)
    assert abs(one_sigma[0] - 0.72) < 1e-12
    assert abs(eight_sigmas[0] - one_sigma[0]) < 1e-12

    one_step_sa = SamplingPlan(
        sampler="sa_solver",
        video_sigmas=refinement_sigma_schedule(1, 0.20, 6.0),
        audio_sigmas=refinement_sigma_schedule(1, 0.20, 6.0),
        actual_step_indices=(0,),
        video_shift=6.0,
        audio_shift=3.0,
        seed=82303,
    )

    def constant_predict(video, audio, clock, *, step_index, is_actual_step):
        assert is_actual_step
        return AVPrediction(
            torch.full_like(video, 0.25), torch.full_like(audio, -0.125)
        )

    sa_video, sa_audio = SASolverAVSampler().sample(
        first["video_latents"].clone(),
        first["audio_latents"].clone(),
        one_step_sa,
        constant_predict,
    )
    assert torch.equal(sa_video, torch.full_like(sa_video, 0.25))
    assert torch.equal(sa_audio, torch.full_like(sa_audio, -0.125))

    strong_sigmas = refinement_sigma_schedule(4, 0.30, 6.0)
    multi_step_sa = SamplingPlan(
        sampler="sa_solver",
        video_sigmas=strong_sigmas,
        audio_sigmas=strong_sigmas,
        actual_step_indices=(0, 1, 2, 3),
        video_shift=6.0,
        audio_shift=3.0,
        seed=82303,
    )
    first_sa = SASolverAVSampler().sample(
        first["video_latents"].clone(),
        first["audio_latents"].clone(),
        multi_step_sa,
        constant_predict,
    )
    second_sa = SASolverAVSampler().sample(
        first["video_latents"].clone(),
        first["audio_latents"].clone(),
        multi_step_sa,
        constant_predict,
    )
    assert torch.equal(first_sa[0], second_sa[0])
    assert torch.equal(first_sa[1], second_sa[1])
    assert torch.isfinite(first_sa[0]).all() and torch.isfinite(first_sa[1]).all()

    forecast_flags = []
    forecast_sa_plan = SamplingPlan(
        sampler="sa_solver",
        video_sigmas=strong_sigmas,
        audio_sigmas=strong_sigmas,
        actual_step_indices=(0, 1, 3),
        video_shift=6.0,
        audio_shift=3.0,
        seed=82303,
    )

    def forecast_contract_predict(
        video, audio, clock, *, step_index, is_actual_step
    ):
        forecast_flags.append((step_index, is_actual_step))
        return AVPrediction(
            torch.full_like(video, 0.25), torch.full_like(audio, -0.125)
        )

    forecast_sa = SASolverAVSampler().sample(
        first["video_latents"].clone(),
        first["audio_latents"].clone(),
        forecast_sa_plan,
        forecast_contract_predict,
    )
    assert forecast_flags == [(0, True), (1, True), (2, False), (3, True)]
    assert torch.isfinite(forecast_sa[0]).all()
    assert torch.isfinite(forecast_sa[1]).all()

    # A persisted prefix followed by the untouched sigma suffix must be
    # numerically identical to one uninterrupted RES run.  The predictor uses
    # the global step index so this also guards the resume offset contract.
    def indexed_predict(video, audio, clock, *, step_index, is_actual_step):
        value = float(step_index + 1) / 10.0
        return AVPrediction(
            torch.full_like(video, value), torch.full_like(audio, -value)
        )

    full_video_in = first["video_latents"].clone()
    full_audio_in = first["audio_latents"].clone()
    full_video, full_audio = ResMultistepAVSampler().sample(
        full_video_in.clone(), full_audio_in.clone(), plan, indexed_predict
    )
    prefix_history = {}

    def capture_prefix(index, clock, next_video, next_audio):
        prefix_history.update({
            "index": index,
            "video": next_video.clone(),
            "audio": next_audio.clone(),
            "previous_video": torch.full_like(next_video, float(index + 1) / 10.0),
            "previous_audio": torch.full_like(next_audio, -float(index + 1) / 10.0),
            "sigma": clock.video_sigma,
        })

    prefix_plan = SamplingPlan(
        sampler="res_multistep",
        video_sigmas=plan.video_sigmas[:3],
        audio_sigmas=plan.audio_sigmas[:3],
        actual_step_indices=(0, 1),
        video_shift=12.0,
        audio_shift=3.0,
    )
    ResMultistepAVSampler().sample(
        full_video_in.clone(), full_audio_in.clone(), prefix_plan,
        indexed_predict, callback=capture_prefix,
    )
    assert prefix_history["index"] == 1
    suffix_plan = SamplingPlan(
        sampler="res_multistep",
        video_sigmas=plan.video_sigmas[2:],
        audio_sigmas=plan.audio_sigmas[2:],
        actual_step_indices=(2, 3),
        video_shift=12.0,
        audio_shift=3.0,
        step_index_offset=2,
    )
    resumed_video, resumed_audio = ResMultistepAVSampler().sample(
        prefix_history["video"], prefix_history["audio"], suffix_plan,
        indexed_predict,
        initial_previous_video=prefix_history["previous_video"],
        initial_previous_audio=prefix_history["previous_audio"],
        initial_previous_video_sigma=prefix_history["sigma"],
        initial_previous_audio_sigma=prefix_history["sigma"],
    )
    assert torch.equal(resumed_video, full_video)
    assert torch.equal(resumed_audio, full_audio)

    frame_count, height, width, fps, sample_rate = 12, 64, 64, 24, 32_000
    frames = np.zeros((frame_count, height, width, 3), dtype=np.float32)
    for index in range(frame_count):
        frames[index, :, :, 0] = index / max(frame_count - 1, 1)
        frames[index, :, :, 1] = np.linspace(0.0, 1.0, width)[None, :]
    sample_count = round(frame_count / fps * sample_rate)
    time = np.arange(sample_count, dtype=np.float32) / sample_rate
    waveform = np.stack(
        [0.25 * np.sin(2 * np.pi * 440 * time), 0.25 * np.sin(2 * np.pi * 660 * time)]
    )
    normalized = normalize_h3_audio_loudness(waveform)
    assert normalized.shape == waveform.shape
    assert np.isfinite(normalized).all()
    loud = normalize_h3_audio_loudness(waveform * 4.0)
    assert float(loud.std(ddof=1)) < float((waveform * 4.0).std(ddof=1))
    sparse_peak = waveform.copy()
    sparse_peak[0, sample_rate // 4] = 1.8
    peak_safe = normalize_h3_audio_loudness(sparse_peak)
    assert float(np.max(np.abs(peak_safe))) <= 0.900001
    assert np.allclose(
        peak_safe / peak_safe[0, sample_rate // 4],
        sparse_peak / sparse_peak[0, sample_rate // 4],
        atol=1e-6,
    )

    with tempfile.TemporaryDirectory(prefix="h3-native-mux-") as directory:
        root = Path(directory)
        destination = root / "synthetic.mp4"
        result = AtomicPyAVMuxer(output_root=root).write(
            video=frames,
            audio=waveform,
            sample_rate=sample_rate,
            fps=fps,
            output_path=destination,
        )
        assert Path(result["output_path"]).is_file()
        probe = probe_media(destination)
        assert probe.frame_count == frame_count
        assert (probe.width, probe.height) == (width, height)
        assert probe.audio_channels == 2
        assert probe.audio_sample_rate == sample_rate
        assert (
            result["encoder"]["audio_normalization"]["policy"]
            == "std5_peak0p9_shape_preserving_v2"
        )
        assert result["encoder"]["audio_normalization"]["hard_clipped_samples"] == 0
        assert not list(root.glob("*.tmp.mp4"))

        cancelled_destination = root / "cancelled.mp4"
        cancel_calls = 0

        def cancel_during_video() -> None:
            nonlocal cancel_calls
            cancel_calls += 1
            if cancel_calls >= 5:
                raise RuntimeError("synthetic cancellation")

        try:
            AtomicPyAVMuxer(output_root=root).write(
                video=frames,
                audio=waveform,
                sample_rate=sample_rate,
                fps=fps,
                output_path=cancelled_destination,
                cancel_check=cancel_during_video,
            )
        except RuntimeError as error:
            assert str(error) == "synthetic cancellation"
        else:
            raise AssertionError("mux cancellation did not propagate")
        assert not cancelled_destination.exists()
        assert not list(root.glob("*.tmp.mp4"))


if __name__ == "__main__":
    run()
    print("native scheduler/sampler/PyAV mux CPU self-test: PASS")
