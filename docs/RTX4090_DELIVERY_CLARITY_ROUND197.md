# RTX 4090 delivery-clarity frontier (Round197)

Status: implemented production-safe improvement; no DiT/VAE/trajectory change.

## Question

Can the service preserve more of H3's decoded detail without increasing total
generation time or changing motion/audio?

The previous delivery default was x264 `CRF18/veryfast`. A direct post-VAE CAS
sharpening prototype was also tested, but it increased 720p5 mux time from
about 1.32 to 1.89 seconds and did not produce a stable FasterVQA gain. It was
removed rather than exposed as a quality feature.

## Controlled experiment

The accepted Round144 1280x736x362-frame output was decoded once to identical
RGB frames and audio. Every candidate encoded that exact array through the
release `AtomicPyAVMuxer`; therefore prompt, seed, latent, VAE output, frame
timing, motion, and audio were identical. Five repetitions were used for the
selected candidate and three for the former default.

| x264 delivery | Median mux | PSNR vs decoded source | SSIM | Output size |
|---|---:|---:|---:|---:|
| former CRF18/veryfast | 2.346 s | 45.074 dB | 0.990396 | 6.89 MiB |
| selected CRF14/superfast | 2.284 s | 46.442 dB | 0.993200 | 16.91 MiB |
| rejected CRF12/superfast | 2.372 s | 46.880 dB | 0.994252 | 22.27 MiB |

The selected point improves PSNR by 1.37 dB and SSIM by 0.002804 while reducing
median mux latency by 2.6%. Its cost is a 2.45x larger MP4 than the former
default. This is a deliberate local-product trade-off: preserve expensive H3
detail while keeping generation latency flat or lower.

FasterVQA did not rank these codec-only transcodes monotonically with exact
reconstruction fidelity (`0.7593` former versus `0.7497` selected). It remains
a useful broad perceptual signal for same-content generation A/B tests, but is
not used as the gate for codec fidelity when an exact decoded reference exists.

## Artifacts

- former delivery A/B:
  `runtime/calibration/clarity_round197/encode_720p15/former_crf18_veryfast_r3.mp4`
- selected delivery A/B:
  `runtime/calibration/clarity_round197/encode_720p15/release_crf14_superfast_r5.mp4`
- FasterVQA report:
  `runtime/quality/fastvqa/round197_encode_720p15.json`

## Production-path verification (Round198)

The selected default was then exercised through the complete native H3 path,
not only the offline mux benchmark. Round192 and Round198 used the same prompt,
seed, 1280x736x124 workload, 12 actual / 8 forecast schedule, and the same
MTCR/fused-RMS inference configuration. Round198 changed only the release mux
default from CRF14/veryfast to CRF14/superfast.

| End-to-end run | Total | Denoise | Video decode | Mux |
|---|---:|---:|---:|---:|
| Round192, former preset | 90.119 s | 73.680 s | 11.661 s | 1.325 s |
| Round198, release preset | 90.041 s | 73.788 s | 11.631 s | 1.212 s |

The release candidate therefore stayed within ordinary run-to-run noise at
end-to-end level and reduced the measured delivery stage by 8.5%. The produced
file is H.264/AAC, 1280x736, 124 frames at 24 fps, 5.167 seconds. Its decoded
video differs from the former encoded output by PSNR 46.805 dB / SSIM 0.989755;
the stronger exact-reference comparison remains the controlled Round197 test
above because it starts from identical decoded source frames.

- production-path output:
  `runtime/calibration/clarity_round198/release_delivery_720p5/round198_release_delivery_720p5_round86_mtcr_fused_rms_720p5_1280x736_124f_seed82303.mp4`
- production-path timing:
  `runtime/calibration/clarity_round198/release_delivery_720p5/round198_release_delivery_720p5_hot_session.json`
- supporting FasterVQA report:
  `runtime/quality/fastvqa/round198_release_e2e_720p5.json`

FasterVQA was 0.84725 for Round192 and 0.84660 for Round198. This tiny reversed
delta reinforces the earlier boundary: FasterVQA is not a codec-fidelity gate;
for delivery-only changes the exact decoded-reference PSNR/SSIM test is the
relevant automatic evidence, followed by Human playback.

## Separate numerical experiment

Changing the same 10%-budget sparse Attention map from FP16 to FP32
accumulation was also tested on a real 100,163-token H3 block. FP32 was 4.3%
slower (`379.7` versus `364.1` ms) and did not reduce full-block error against
the dense teacher (cosine `0.59358` versus `0.59359`). Sparse selection error
dominates accumulation error at this budget, so this route was rejected and
was not added to production.
