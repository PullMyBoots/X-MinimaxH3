# Ref2VA resolution matrix — RTX 4090

Date: 2026-08-13

## Fixed contract

- Standalone Native Ref2VA Pruned INT8 ConvRot
- 3 reference images and 2 standalone reference audios; no reference video
- 8 seconds, 192 frames, 24 fps, seed 82416
- 20-point scheduler with 9 actual and 11 forecast evaluations
- Dense SageAttention and exact INT8 path; no sparse attention or fused RMS/AdaLN
- Block offload, prefetch depth 1, MLP chunk 8192, Video-VAE tile 288
- Compact host-memory profile
- Same structured complex-dialogue prompt at every resolution

Manifest: `benchmarks/ref2va_extreme/ref2va_resolution_matrix_8s_9_11.json`

## Direct measurements

Each geometry ran twice in one persistent process. The first request includes
geometry-specific multimodal Qwen conditioning; the second is the reported
same-prompt/reference hot request.

| Geometry | First request | Hot total | Hot denoise | Video VAE | Peak allocated | Packed tokens |
|---|---:|---:|---:|---:|---:|---:|
| 640x352 (360p) | 87.361 s | 35.361 s | 26.652 s | 6.696 s | 6.453 GiB | 15,743 |
| 864x480 (480p) | 112.778 s | 66.530 s | 55.104 s | 8.975 s | 7.786 GiB | 27,392 |
| 1280x736 (720p compute canvas) | 237.600 s | 192.060 s | 167.995 s | 20.336 s | 11.497 GiB | 59,897 |

All three requests completed without OOM. Relative to 480p, 720p has 2.27x
as many pixels, 2.19x as many packed tokens and took 2.89x as long end to end.
The nonlinear gap comes mainly from long-sequence DiT attention/GEMM rather
than model loading: 720p denoise alone was 167.995 seconds.

The first-request premium under the compact memory profile was 45–52 seconds
because the Qwen multimodal embedding depends on target geometry. Same-shape,
same-prompt/reference requests hit the exact conditioning and reference-latent
caches. This is distinct from model startup, which took 63.146 seconds once
for the persistent process.

## Quality gates

Eight uniformly spaced frames from every measured video were inspected before
numeric/audio checks. None showed colored blotches, glare-like double images,
VAE seams, structural collapse or obvious identity loss. 720p preserved the
finest backpack, hair, face-edge and building detail. 360p remained usable but
was visibly softer. Different resolutions regenerated their own latent
trajectory and composition; 720p is not a deterministic upscale of 480p.

Formal WhisperX used the project-mandated checkout, large-v3, CUDA FP16,
Pyannote VAD and Chinese alignment. All three complete transcripts exactly
matched the expected combined dialogue (global CER 0). 720p also split both
sentences exactly inside their expected windows. 360p/480p placed a few
boundary characters in the adjacent sentence window but omitted no dialogue.
Whisper does not prove speaker identity, voice naturalness or lip sync.

## Human review

Review continuous playback for reference identity, voice assignment, lip sync,
acting, motion continuity, backpack consistency, background music and audio
artifacts:

1. `runtime/outputs/ref2va_extreme/resolution_matrix_8s_9_11_attempt1/ref2va_resolution_matrix_8s_9_11_360p_measured_640x352_192f_seed82416.mp4`
2. `runtime/outputs/ref2va_extreme/resolution_matrix_8s_9_11_attempt1/ref2va_resolution_matrix_8s_9_11_480p_measured_864x480_192f_seed82416.mp4`
3. `runtime/outputs/ref2va_extreme/resolution_matrix_8s_9_11_attempt1/ref2va_resolution_matrix_8s_9_11_720p_measured_1280x736_192f_seed82416.mp4`

The immutable timing report, contact sheets and Whisper JSON live beside the
videos.

## Reference-video product surface

Reference-video conditioning now uses the same Ref2VA surface as images and
standalone audio: Web Studio accepts up to three 2–15-second videos, the API
uses `reference_video_1` through `reference_video_3`, and ComfyUI exposes
`Video 1` through `Video 3`. Total video duration is capped at 15 seconds.
The embedded soundtrack is intentionally ignored; voice or sound conditioning
still requires standalone `reference_audio_N` inputs.

## Direct FL2VA original-INT8 comparison

FL2VA was measured on the same RTX 4090 with the same prompt, seed, canvas,
192 frames and 9/11 schedule. Its first and last inputs were extracted from the
Ref2VA dense 20/0 anchor; every geometry was run twice and the second request
is reported. Ref2VA retained its three reference images and two standalone
reference audios. This is therefore a same-output-workload throughput
comparison, not an equal-conditioning-capability comparison.

| Geometry | FL2VA INT8 hot | Ref2VA INT8 hot | Ref2VA gap |
|---|---:|---:|---:|
| 640x352 | 32.137 s | 35.361 s | +3.224 s / 10.03% |
| 864x480 | 64.937 s | 66.530 s | +1.593 s / 2.45% |
| 1280x736 | 191.463 s | 192.060 s | +0.597 s / 0.31% |

The relative Ref2VA overhead shrinks with resolution because the fixed
reference-conditioning prefix is increasingly amortized by the target video
sequence. At 720p, target DiT work dominates and the two engines are effectively
equal in throughput. Eight-frame inspection found no colored blotches,
double-image artifacts, VAE seams or structural collapse in either engine.
Continuous Human review remains necessary for motion, acting, lip sync and
sound. FL2VA did not receive the two reference voices, so voice-reference
fidelity cannot be compared as if the capabilities were equal.
