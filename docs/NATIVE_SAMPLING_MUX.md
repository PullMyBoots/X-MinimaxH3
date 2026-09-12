# Native H3 sampling and media mux

This adapter package replaces the remaining scheduler/sampler/media behavior
that previously happened behind ComfyUI nodes. It does not implement the DiT,
VAE, text encoder, forecast controller, or LoRA loader.

## Sampling contract

`H3SimpleScheduler.prepare(state)` is directly compatible with the native
pipeline's scheduler stage. It:

- treats `num_steps` as model evaluations and creates `num_steps + 1` sigma
  endpoints from 1 to 0;
- uses separate rectified-flow shifts for video (12) and audio (3);
- uses one CPU RNG stream, drawing complete float32 video noise before complete
  channel-major stereo audio noise;
- derives video latent time with `5n+2`, spatial size with VAE scale 16, and
  audio latent time at 40 Hz;
- stores `SamplingPlan` and `H3LatentGeometry` in `PipelineState.metadata`;
- accepts an injected `layout_builder`; packed H3 layout remains the model-core
  adapter's responsibility.

`ResMultistepAVSampler` implements an eta-zero, second-order RES update on both
clocks. Its predictor receives `is_actual_step`; the forecast controller can
therefore replace a DiT evaluation without changing sampler mathematics.

`TurboAVSampler` offers two explicit modes:

- `DUAL_SHIFT` is the default native contract and converts the prediction from
  video sigma to the shifted audio clock using the analytic slope;
- `SHARED_VIDEO` is only for a model adapter that has already represented audio
  on the video clock. Selecting it and also applying audio shift in the model
  would be a correctness error.

Both samplers call an injected cancellation hook before every step. The model
adapter must return `AVPrediction(video_denoised, audio_denoised)` and must not
return raw velocity under that interface.

## Mux contract

`AtomicPyAVMuxer.write(...)` matches the native pipeline mux stage and:

1. accepts host RGB video in `[F,H,W,3]`, `[3,F,H,W]`, or `[F,3,H,W]`;
2. requires stereo audio and applies the established
   `audio /= max(std(channel,time) * 5, 1)` rule;
3. trims or zero-pads audio to the video duration;
4. writes H.264 (`libx264`, yuv420p) and AAC into a same-directory temporary
   MP4 using explicit video and audio time bases;
5. fsyncs, probes geometry/frame count/FPS/stereo/sample rate/duration, then
   atomically publishes with `os.replace` and fsyncs the parent directory;
6. deletes the temporary output on encode, probe, cancellation, or publication
   failure.

PyAV, NumPy and PyTorch are imported only by this adapter at execution time.
They must be declared by the final native-runtime installation; the lightweight
Web/API process can still import without initializing CUDA or PyAV.

## Provenance

The implementation is a focused rewrite informed by Apache-2.0 sources:

- LightX2V `231e2307f15b9eb60fe3f877f7eed945c8d8d717`:
  `models/schedulers/minimax_h3/scheduler.py` for CPU RNG order, latent geometry,
  per-modality shift schedules, and `utils/ltx2_media_io.py` for PyAV stream
  construction.
- SGLang Diffusion `c2fbe2f6d88692fa7756ed1be73ef9e85bd6b7cf`:
  MiniMax H3 time-request/timestep stages for fixed float32 shifted schedules.
- The released Apache-2.0 Turbo adapter behavior supplies the explicit
  dual-clock slope contract and the warning against applying audio shift twice.

The original-route RES implementation is written from its numerical update and
the already-established product behavior, without importing or copying the
GPL-licensed ComfyUI runtime. No source framework becomes a runtime dependency.

## Tests and remaining proof

Run the synthetic CPU AV test in an environment with PyAV/libx264/AAC:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m \
  h3serve.native_engine.adapters.sampling_mux.selftest
```

The test checks deterministic noise, latent shapes, both sampler interfaces,
loudness processing, real H.264/AAC encode, atomic publication, and decoded
media probe. It does not prove H3 parity. Integration must still compare:

- exact initial latent tensors against the current correct workflow;
- all 20 video/audio sigma endpoints;
- every full-compute RES step and every Turbo clock update;
- final 24 FPS / 32 kHz stereo media duration;
- full generated videos through the mandatory multi-frame visual gate.
