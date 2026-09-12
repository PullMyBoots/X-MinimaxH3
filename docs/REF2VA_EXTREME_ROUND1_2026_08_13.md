# Ref2VA extreme optimization — Round 1

Date: 2026-08-13

## Contract

- Engine: standalone Native Ref2VA Pruned INT8 ConvRot
- Input: 3 reference images and 2 reference audios from `ref-test/`
- Canvas: 864x480, 192 frames, 24 fps (8 seconds)
- Seed: 82416
- Scheduler: original-weight 20-point RES multistep schedule
- Prompt: locally rewritten with the product's H3 rules; no external API call
- Prompt manifest: `benchmarks/ref2va_extreme/ref2va_multiref_8s_enhanced_v1.json`
- Main quality gate: 8 sampled frames, WhisperX large-v3 CUDA/FP16, then Human playback

The prompt binds Picture 1 to the school bag, Picture 2/Audio 1 to the girl,
Picture 3/Audio 2 to the young man, assigns two timed Mandarin lines, colocates
actions with diegetic sound, forbids extra speech, and sets
`non_diegetic_music: N/A`.

## Measured hot results

The fair hot anchor is the cached-prompt 20 actual / 0 forecast request at
127.503 seconds. Cold Qwen and model construction are not included in the
speedup claim.

| Candidate | Actual / forecast | Total | Denoise | Peak allocated | Hot speedup | Automatic gate |
|---|---:|---:|---:|---:|---:|---|
| Dense anchor | 20 / 0 | 127.503 s | 114.731 s | 10.118 GiB | 1.000x | visual + exact full transcript |
| Conservative | 16 / 4 | 105.938 s | 94.424 s | 7.786 GiB | 1.204x | visual + exact full transcript |
| Balanced | 14 / 6 | 95.149 s | 83.632 s | 7.786 GiB | 1.340x | visual + exact full transcript |
| Fast | 12 / 8 | 84.286 s | 72.822 s | 7.786 GiB | 1.513x | visual + exact full transcript |
| Extreme candidate | 10 / 10 | 73.441 s | 61.898 s | 7.786 GiB | 1.736x | visual + exact full transcript |
| Extreme candidate | 9 / 11 | 68.045 s | 56.605 s | 7.787 GiB | 1.874x | visual + exact full transcript |
| 2x candidate | 8 / 12 | 62.602 s | 51.173 s | 7.786 GiB | 2.037x | visual + exact full transcript |
| Frontier candidate | 7 / 13 | 57.186 s | 45.729 s | 7.787 GiB | 2.230x | visual + exact full transcript |
| Frontier candidate | 6 / 14 | 51.954 s | 40.403 s | 7.786 GiB | 2.454x | visual pass; Whisper transcript semantically exact but traditional characters inflate CER |

All speedups are single-process same-machine measurements. Only the dense
anchor is a quality reference. Forecast candidates change the denoising
trajectory and remain Human-review-only until continuous playback accepts
motion, acting, lip sync, reference voice identity and non-speech artifacts.

## Online prompt and reference conditioning

The table above is deliberately a same-prompt hot comparison. A new Ref2VA
request also has to encode a long multimodal prompt, three reference images
and two reference audios. Under the compact memory profile this conditioning
path originally took about 79 seconds: about 8.2 seconds for multimodal vision
and about 70.8 seconds for the 50-layer quantized Qwen encoder.

The full-speed service path now keeps the quantized Qwen weights in pinned
host memory and caches only reference-image features by file content and
geometry. Direct isolated measurements are:

| Conditioning case | Vision | Qwen | Additional hot request cost |
|---|---:|---:|---:|
| First request for a new reference set | 5.282 s | 3.972 s | about 9.25 s |
| New prompt, same reference images | 0.029 s | 3.549 s | about 3.58 s |

The pinned Qwen cache occupies 13.50 GiB of host RAM and the visual feature
cache only 46.875 MiB. File-mapped and pinned executions produced identical
embedding checksums. Changing only the prompt produced a different embedding,
which verifies that the cache does not incorrectly reuse final prompt
conditioning. Therefore a 7/13 request is approximately 60.7 seconds for a
new prompt with the same references, or 66.4 seconds for the first use of a
new reference set. These two totals are component-derived estimates pending a
single-process full-speed end-to-end remeasurement; 57.186 seconds remains the
direct same-prompt hot measurement.

Evidence:

- `runtime/profile/ref2va_qwen/compact_filemapped.json`
- `runtime/profile/ref2va_qwen/fullspeed_pinned.json`
- `runtime/profile/ref2va_qwen/fullspeed_same_refs_different_prompt.json`

## Fair FL2VA comparison

The closest existing FL2VA measurement uses exactly the same output geometry,
864x480 by 192 frames. Its LoRA engine ran six real steps in 44.100 seconds in
a hot session. Ref2VA has no trained six-step LoRA in this release: it uses the
original 20-step Ref2VA checkpoint plus forecast, and must preserve three image
and two voice references.

| Engine and schedule | Direct hot total | Gap to FL2VA LoRA6 |
|---|---:|---:|
| FL2VA LoRA, 6 real steps | 44.100 s | reference |
| Ref2VA 8/12 | 62.602 s | +18.502 s / 1.420x slower |
| Ref2VA 7/13 | 57.186 s | +13.086 s / 1.297x slower |
| Ref2VA 6/14 | 51.954 s | +7.854 s / 1.178x slower |

This is an equal-canvas and equal-duration throughput comparison, not an
equal-model-quality claim. The FL2VA result comes from a distilled six-step
LoRA, while the Ref2VA candidates approximate a 20-step original-weight path.
The 6/14 Ref2VA candidate is numerically closest, but it is not publishable
until Human review accepts reference identity, voice, motion and acting.

### Original INT8 FL2VA correction

An API-level comparison exposed and fixed a hot-cache identity bug. Uploaded
first/last frames were previously keyed by both pathname and content hash;
because every job stores uploads in a new directory, identical images missed
the cache. The cache now uses content hash, ordered role and geometry. The
obsolete cache-miss measurements (87.704 s for 5 s 9/11, 111.884 s for 8 s
9/11 and 164.125 s for 8 s 20/0) must not be used as hot results.

Corrected same-service measurements are:

| Workload | FL2VA original INT8 | Ref2VA original INT8 | Ref2VA gap |
|---|---:|---:|---:|
| 864x480, 192 frames, 9/11 | 64.937 s | 68.045 s | +3.108 s / 4.79% |
| 864x480, 192 frames, 20/0 | 119.418 s | 127.503 s | +8.085 s / 6.77% |

For context, the historical 34.107-second result is 864x480 by 124 frames,
9/11, **T2VA without keyframes**. Repeating that geometry with first and last
frames took 38.496 seconds, an additional 4.389 seconds or 12.87%. Thus the
user's remembered 30-second-class 480p/5-second performance remains valid.

## Kernel and memory findings

One profiled two-actual-step request recorded about 30.1 seconds of aggregate
CUDA work. The largest groups were long-sequence SageAttention (~6.42 s),
FP16/CUTLASS GEMM (~6.34 s), pinned H2D (~3.46 s), followed by RMSNorm,
casts and elementwise kernels. Attention is the largest single family, not the
whole request.

Each of the 50 DiT blocks occupies about 369.4 MiB; the block stack is about
18.04 GiB. Keeping 0, 16, 32 or 40 blocks resident raised peak allocation from
10.12 to 20.05 GiB but improved a real step by less than one percent. Double
buffering already hides most weight transport, so consuming all 24 GiB is not
a useful default optimization.

Keeping forecast tail history on the GPU instead of the host produced
byte-identical MP4 output but no measurable speedup. The implementation was
reverted; the A/B evidence remains under `tail_cache_ab_attempt1/`.

Fused RMS/AdaLN and top-k 0.75 sparse attention were also measured. The best
combination completed in 60.440 seconds, only 3.6% faster than dense-kernel
8/12, while sparse attention visibly changed composition. It is retained as a
review artifact, not selected as the default.

## Human review files

Review in this order:

1. Dense 20/0 reference:
   `runtime/outputs/ref2va_extreme/enhanced_v1/hot_batch_attempt1/enhanced_v1_hot_batch_e00_hot_dense20_864x480_192f_seed82416.mp4`
2. 8/12 2x candidate:
   `runtime/outputs/ref2va_extreme/enhanced_v1/extreme_batch_attempt1/enhanced_v1_extreme_batch_x08_forecast8_12_864x480_192f_seed82416.mp4`
3. 7/13 frontier:
   `runtime/outputs/ref2va_extreme/enhanced_v1/frontier_attempt1/enhanced_v1_frontier_f07_dense_forecast7_13_review_864x480_192f_seed82416.mp4`
4. 6/14 frontier:
   `runtime/outputs/ref2va_extreme/enhanced_v1/frontier_attempt1/enhanced_v1_frontier_f06_dense_forecast6_14_review_864x480_192f_seed82416.mp4`
5. Optional sparse/fused combination:
   `runtime/outputs/ref2va_extreme/enhanced_v1/approx_ab_attempt1/enhanced_v1_approx_ab_a30_sparse075_fusedrms_8_12_review_864x480_192f_seed82416.mp4`

The immutable JSON timing reports, contact sheets and Whisper reports live
beside those videos. No earlier candidate was overwritten.
