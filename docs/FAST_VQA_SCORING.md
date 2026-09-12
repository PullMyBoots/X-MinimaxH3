# FasterVQA scoring

This release keeps FasterVQA as an **offline comparison tool**, separate from
the hot H3 generation service. It does not change generation output or latency.

## Install once

```bash
./scripts/setup_fast_vqa.sh
```

The installer reuses the H3 PyTorch/CUDA runtime, downloads the pinned official
source and checkpoint into `runtime/quality/fastvqa`, and installs only isolated
`decord` and `timm` packages. It verifies the checkpoint size and SHA-256.

The upstream S-Lab license permits non-commercial use. Commercial distribution
or use requires contacting its contributors; review
`third_party_licenses/FasterVQA-LICENSE` before use.

## Score videos

```bash
./scripts/score-fast-vqa.sh video-a.mp4 video-b.mp4 \
  --json-out runtime/quality/fastvqa/comparison.json
```

Useful options:

- `--device auto` uses CUDA when available and otherwise CPU.
- `--device cpu` avoids GPU contention with H3, but is slower.
- `--seed 42` fixes temporal and spatial fragment sampling. Keep the same seed
  for every A/B comparison.

The model is loaded once per invocation, so pass all comparison videos in one
command. Each result records scoring time and peak CUDA allocation/reservation.

## Measured footprint on this RTX 4090 host

Validation on 1280x736 videos, using the locked service PyTorch 2.8 runtime:

- isolated tool directory: about 222 MiB total (122 MiB checkpoint, 58 MiB
  shallow source checkout, 43 MiB isolated decoder/model helpers);
- cold one-video command: 5.8 seconds wall time, including Python and PyTorch
  imports; model construction and checkpoint loading itself was 0.94 seconds;
- 720p 5-second score stage: 0.61 seconds cold and about 0.20 seconds warm;
- 720p 15-second score stage: 0.79 seconds in a cold command and 0.40 seconds
  after another video in the same process;
- worst observed CUDA footprint in the multi-video run: 2.62 GiB allocated and
  3.08 GiB reserved; maximum host RSS in the cold command: 1.34 GiB.

The near-constant 5/15-second cost is expected: this FasterVQA configuration
samples 32 frames from the full timeline. On a 24 GiB card, schedule scoring
after H3 generation or use `--device cpu`; do not run it concurrently with a
generation process that already fills VRAM.

## Interpretation boundary

`quality_score` is the official FasterVQA logistic-rescaled score in `[0, 1]`;
higher means better predicted general perceptual quality. It is suitable as a
supporting relative signal for same-content A/B videos. It is **not** a pure
motion-blur score, and it does not validate physical causality, dialogue,
identity, prompt following, or audiovisual synchronization. Final release
acceptance remains native-speed human review.
