# Native original-weight forecast execution

The original-weight engine supports a 20-point RES sampling schedule with an
explicit set of full-compute points. A point outside that set is still a real
sampler update: it executes the first three H3 transformer blocks, predicts the
remaining block-stack residual, runs the original final heads, and then applies
the unchanged RES update. “Forecast” therefore does not mean adding empty
steps after generation.

## Predictor

`DirectionalForecastController` retains the two most recent full-compute tail
residuals in pinned host memory. At a forecast point it:

1. executes blocks 0 through 2;
2. measures the current depth-3 movement using 32 sampled channels;
3. computes one directional confidence for audio and a smooth local field for
   the video latent grid;
4. blends/extrapolates the two retained tail residuals in `[1.0, 1.35]`; and
5. returns the predicted audio/video target rows to the unchanged final layer.

Low-confidence video regions fall back toward the newest full-compute tail.
The predictor never changes checkpoint tensors, LoRA weights, scheduler sigmas,
sampler equations, final heads, VAE decoding, or muxing.

The distilled LoRA engine does not use this mechanism. Its four to eight
requested steps are all real distilled model evaluations.

## Controls

`HotSessionRequest.actual_step_indices` and the benchmark runner's
`--actual-steps` option accept sorted, unique, zero-based full-compute points.
Omitting the option executes every requested point and bypasses the predictor.
The service-facing presets resolve to explicit schedules before engine entry;
the recommended middle setting is:

```text
0,1,2,3,4,8,12,16,19
```

That setting is 9 full-compute points plus 11 forecast points. It remains a
quality/speed choice, not a claim of mathematical equivalence to 20/0.

## Validation evidence

The fixed four-scene 864×480, 124-frame, 24 fps run is stored under the ignored
runtime evidence directory:

```text
runtime/comparisons/four_scene/native_original9a11f/
```

All four outputs completed with exactly 9 actual and 11 forecast records. A
12-frame contact-sheet review per output passed the corruption gate before any
same-seed numerical comparison. The three hot requests averaged 38.357 seconds
end to end; their denoise phase averaged 25.388 seconds. Against the Native
20/0 hot reference, that is 1.686× end-to-end and 2.033× denoise speedup.

SSIM, PSNR, and waveform correlation are retained only as same-seed trajectory
diagnostics. They are not visual-quality acceptance gates.
