# Transparent MiniMax H3 long-horizon generation

Status: release candidate validated on 2026-09-01.

## User contract

The Web console, REST API and normal ComfyUI generation nodes accept one
ordinary H3 prompt and a total duration from 1 to 60 seconds. A request above
the physical native-window limit stays one job, one progress stream and one
final MP4. Users do not select chunks, overlaps, local prompts or stitch modes.

Intermediate checkpoint previews are intentionally rejected above 15 seconds;
silently checkpointing only one internal window would violate the public
trajectory contract.

## Evaluated upstream routes

| Route | Pinned revision | License | Decision |
|---|---|---|---|
| [ComfyUI-H3-Continuum](https://github.com/ukr8b3g-cmyk/ComfyUI-H3-Continuum) | `0a7c6703b7715689c7aab4177e44e5fcd318fe5b` | MIT | Selected mechanism: masked clean joint audio/video prefix |
| [ComfyUI-MiniMax-H3-LongMedia](https://github.com/vizart-vj/ComfyUI-MiniMax-H3-LongMedia) | `4a31874addc3d1b632589ac0454eb990a2736547` | Apache-2.0 | Did not embed its broad ComfyUI monkey-patch runtime; independently adopted one-session lifecycle, deferred decode and localized prompt timelines |

Continuum has the smaller semantic surface and the stronger invariant: prior
clean video and audio are both copied into the next target and protected from
denoising. LongMedia's complete runtime would duplicate X-MinimaxH3's V24
scheduler, sparse/dense Attention, offload and memory governor.

The checked-out analysis copies and the comparison are under
`experiments/h3_long_video_lab/upstreams/` and
`experiments/h3_long_video_lab/UPSTREAMS.md`. They are research inputs, not
bundled public-runtime dependencies.

## Native mechanism

Implementation: `h3serve/native_engine/long_horizon.py`, integrated by
`h3serve/native_engine/engine.py` and `h3serve/native_engine/hot_session.py`.

1. Convert the requested movie to H3's exact `5 + 17*k` frame grid.
2. Solve a bounded integer window partition. The objective combines a
   calibrated fixed window cost, a quadratic long-sequence cost and a smooth
   Gaussian penalty around authored prompt events. It is not a duration lookup
   table or prompt-specific `if/else` schedule.
3. Carry 39 clean frames across every internal join. This equals 12 Video-VAE
   latent frames and 65 Audio-VAE ticks.
4. Assign the clean video prefix a near-clean timestep and the clean audio
   prefix its clean endpoint while the new suffix follows the ordinary noisy
   trajectory. Restore both prefixes after every solver update.
5. Keep prefix Attention dense and allow V24 sparsity only on the new suffix.
6. Stage clean segment latents in the Linux-native process `TMPDIR`, remove each
   duplicated prefix exactly once, and stream one window at a time into a
   preallocated final joint-AV tensor. Only then perform one Video-VAE/Audio-VAE
   decode and one MP4 mux. This avoids both Windows-mounted intermediate I/O and
   the old all-windows-plus-`torch.cat` host-memory peak.

For the 30-second acceptance prompt, the event-aware planner selected physical
windows `[328, 277, 192]`, visible contributions `[328, 238, 153]`, and hidden
joins at frames 328 and 566 (13.667 and 23.583 seconds). Those joins stay away
from the authored 10- and 20-second shot changes.

## Prompt localization without a second user syntax

The backend recognizes ordinary H3 `[Shot N] At MM:SS` timestamps as well as
plain Chinese or English time ranges. Each physical window receives a local
clock while soundtrack sections remain attached.

If a shot started before the new suffix, the carried clean prefix defines the
current physical state. Repeating all earlier action clauses caused camera
resets and repeated dialogue in real A/B videos. The release compiler therefore
uses chronological sentence order and a front-loaded `sqrt(progress)` action
clock to remove clauses already represented by the carried prefix. Explicit
later shot timestamps remain separate events and are never removed.

This improves adherence but does not turn H3 timing into a frame-exact editor.
Exact second boundaries remain probabilistic model instructions.

## Real-video evidence

Acceptance prompt:
`experiments/h3_long_video_lab/prompts/timeline_30s.txt`.

Release artifact:
`experiments/h3_long_video_lab/results/20260901_timeline30_frontloaded_progress_base20_a75/timeline30_frontloaded_progress_base20_a75_seed82901.mp4`.
The machine-readable settings, hashes, timings and review observations are in
the adjacent `acceptance_report.json`.

Configuration: INT8 FL2VA Base, 864×480, 719 frames at 24 fps (29.959 s),
20 sampling steps, acceleration 75, seed 82901, RTX 4090.

| Evidence | Result |
|---|---|
| First-process run | 261.526 s end to end |
| Same hot-session rerun | 254.398 s end to end |
| Current release replay after Linux staging + streaming stitch | 224.742 s end to end; 1.132× faster than the prior hot rerun |
| Determinism | identical MP4 SHA-256 `88aedede881587e23a01027896fc939af367c5021877ae498c2b26a57e27fac5` |
| DiT windows, first run | 74.940 / 69.175 / 44.332 s |
| Final Video-VAE decode | 29.809 s |
| Final Audio-VAE decode + mux | 3.566 s |
| Visual join/frame 328 | frame MAE 1.270× its local temporal median |
| Visual join/frame 566 | frame MAE 1.242× its local temporal median |
| Audio join/frame 328 | sample jump 0.003414; below local P99 0.138644 |
| Audio join/frame 566 | sample jump 0.048082; below local P99 0.253155 |
| WhisperX transcript | `large-v3`, CUDA FP16, Silero VAD and English forced alignment: `Don't even think about it.` at 4.098–5.059 s; `I knew it.` at 21.745–22.546 s; no repeated dialogue |

The contact sheet and exact join frames are saved beside the MP4 as
`contact_2s.jpg` and `seams.jpg`. They show continuous identity, wardrobe,
camera framing, table, robot vacuum and doughnut across both hidden joins. The
movie follows the intended three-part story: warning, theft, discovery.
The aligned transcript is retained under `whisperx_large_v3/` beside the
movie; its JSON SHA-256 is
`6d956882d4012e7d6a126f5922e6cfedc605881624c20ad3c67d574eec4fe548`.

The final post-optimization replay used the current service rather than a fake
session. Its three DiT windows took 73.404 / 67.548 / 42.615 s; final video
decode, audio decode and mux took 29.808 / 0.510 / 3.144 s. Runtime telemetry
reported both `system_temp_staging=true` and `streaming_latent_stitch=true`, and
the temporary directory was observed under `/root/h3-new-serve-runtime/tmp`
through every window then disappeared after completion. Its 19,228,436-byte
MP4 is byte-identical to both previously reviewed movies, so the memory/I/O
optimization introduces no latent, pixel or audio change. The full receipt is
`current_streaming_replay.json` beside the movie.

Earlier rejected iterations are retained in sibling result directories. They
demonstrate why clean timesteps alone were insufficient: one version created a
90× local visual jump at the second hidden join; another repeated `I knew it`
after the join. Both failures drove the progress-aware prompt compiler.

## Performance decisions

The event-aware three-window route took 130.768 s in the five-step smoke test,
versus 142.153 s for the same 39-frame context with the earlier fixed four-window
layout: 1.087× faster while removing one redundant DiT window. It was also
1.046× faster than the original four-window baseline (136.731 s), despite using
the longer, more stable context.

The partition solver is a single shortest-path pass over H3 temporal-grid
units rather than a duration/window-count enumeration. On the release host it
planned the public 60-second limit in less than 0.0005 s; the accepted 30-second layout
remained byte-for-byte identical after this complexity fix.

The final 864×480×719 Video-VAE decode was profiled independently:

| Decode | Time | Peak allocated | Output hash |
|---|---:|---:|---|
| temporal host chunks = 6 | 29.411 s | 5.649 GiB | `cb3273...ddd23` |
| no temporal host chunks | 29.435 s | 15.756 GiB | `cb3273...ddd23` |

Removing temporal chunks provides no speed gain and consumes 10.107 GiB more
VRAM, so the low-memory decode remains the release route. There is no
intermediate RGB decode, repeated Qwen encode for identical localized prompts,
second H3 model, or full latent clone in the hot path.

Final latent assembly was also profiled in isolated processes with eight
362-frame BF16 windows on a 1080p-class latent canvas (2623 output frames,
109.292 s). The previous load-all-then-`cat` path peaked at 1117.352 MiB RSS;
the release streaming path peaked at 832.746 MiB, saving 284.605 MiB (25.47%).
Assembly time changed only from 0.141 to 0.154 s, and the output tensors are
element-identical in regression tests. The machine-readable result is
`experiments/h3_long_video_lab/results/20260901_streaming_stitch_profile.json`.

## Evidence boundary

The executor and contract tests cover arbitrary duration, FL2VA/Ref2VA request
plumbing, English/Chinese timeline localization, exact H3 grids, prefix masks,
stitching, sparse-prefix protection and cancellation. The real release-quality
movie gate above covers FL2VA 480p×30s on one RTX 4090. Higher resolutions,
longer durations, reference-heavy Ref2VA and physical 8/16GB cards remain
deployment-specific quality/performance validation boundaries; the service
must not describe those combinations as visually proven by this single movie.
