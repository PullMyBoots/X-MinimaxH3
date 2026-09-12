# X-MinimaxH3

**English** · [简体中文](README.zh-CN.md)

X-MinimaxH3 is a local MiniMax H3 video-generation service optimized for a
single NVIDIA SM89 GPU. Its bilingual Web console and REST API cover single
video creation, long-video creation and task management, with FL2VA/Ref2VA,
Base/LoRA execution, SelfLift progressive generation, retained-latent final
sampling and automatic face repair.

> Model weights, user uploads, latent states and generated videos are not
> distributed in this repository.

## Features

- A unified two-handle trajectory for first-pass/final steps and resolutions,
  with independently tunable acceleration on both sampling segments.
- Joint Base scheduler for actual DiT evaluations, forecast evaluations and
  per-step/per-layer attention budgets.
- Four public model choices: W4A8 or INT8, each with FL2VA and Ref2VA. The
  private 8GB/16GB/24GB VRAM executor is selected automatically.
- Automatic resource routing and execution-plan compilation outside the DiT
  hot loop.
- SelfLift generation: run the first trajectory segment at a smaller canvas,
  lift its clean latent with the H3 learned 3D upscaler, then finish at up to
  1440p on admitted INT8 profiles.
- Long-video online creation with low-resolution cumulative previews and a
  retained clean final branch, plus JSON one-click creation without redundant
  intermediate preview decoding. Every Ref2VA JSON window may define its own
  complete Picture/Audio reference set; omitted sets inherit the previous
  window, and server-local paths remain private.
- Full-film final sampling over one continuous low-resolution latent timeline,
  with optional 3–8 second temporal windows, overlap, audio-token authority,
  adjustable final sigma scale and one final decode.
- Automatic FL2VA face repair that ranks under-resolved face tracks, packs the
  selected regions into a square atlas and runs four-step H3 Turbo repair.
- Text-only, first-frame, last-frame and first+last-frame FL2VA generation.
- Multi-reference Ref2VA with images, videos and independent audio references.
- Larry Turbo and three task-aware LightX2V LoRA profiles.
- Configurable 1–4 step fork previews that do not modify the retained formal
  sampling state.
- Serial GPU queue, cancellation, task history and one-second hardware
  telemetry.
- Optional ComfyUI HTTP connector that does not load a second H3 model.
- English and Simplified Chinese console and documentation.

## AI prompt-writing guides

The [`ai-prompt-guides/`](ai-prompt-guides/) directory contains six standalone
Chinese instruction files that can be uploaded to ChatGPT or another AI before
describing a scene. They cover FL2VA and Ref2VA across single-video creation,
online window-by-window long-video creation and one-click long-video JSON. Each
file defines the required questions, H3 writing rules and exact paste-ready
output format; use only the file matching the current task.

## Video tutorial

<p align="center">
  <a href="https://youtu.be/KYkMspNGEh4">
    <img src="assets/tutorial/x-minimaxh3-tutorial-en.png" width="860" alt="MiniMax H3 unlimited-length video generation — X-MinimaxH3 tutorial">
  </a>
</p>

<p align="center">
  <strong>▶ MiniMax H3 unlimited-length video generation</strong><br>
  <a href="https://youtu.be/KYkMspNGEh4">Watch the complete English tutorial on YouTube</a>
</p>

## Measured effect comparison

The following local comparison shows the 720p generation and 1440p native H3
second-sampling result side by side. The divider sweeps across the same source
footage for 5-, 10- and 15-second examples. Measurements were recorded on a
14th Gen Intel Core i9, 128GB RAM and an RTX 4090 24GB using INT8 FL2VA.

<p align="center">
  <video controls muted loop playsinline width="860" src="assets/demos/effect-comparison-en.mp4">
    Your browser does not support embedded video.
  </video>
</p>

<p align="center">
  <a href="assets/demos/effect-comparison-en.mp4">▶ Watch or download the English comparison video</a>
</p>

## Community and feedback

Use [GitHub Discussions](https://github.com/PullMyBoots/X-MinimaxH3/discussions)
for installation help, hardware compatibility reports, benchmarks, API questions
and generated-video showcases. For real-time chat, join the public
[Telegram group](https://t.me/XMinimaxH3Community). You can also contact the
author directly on WeChat; please include `X-MinimaxH3` in your friend request.

| Join the Telegram community | Contact the author | Join the WeChat group |
|:---:|:---:|:---:|
| <a href="https://t.me/XMinimaxH3Community"><img src="assets/community/telegram-community.png" width="260" alt="X-MinimaxH3 Telegram community QR code"></a> | <img src="assets/community/wechat-contact.jpg" width="260" alt="Author WeChat QR code"> | <img src="assets/community/wechat-group.jpg" width="260" alt="X-MinimaxH3 WeChat group QR code"> |
| [Open the public group](https://t.me/XMinimaxH3Community) | Add `X-MinimaxH3` to the request | An updated QR code will be posted here after the current one expires |

For reproducible bugs and feature requests, please use
[GitHub Issues](https://github.com/PullMyBoots/X-MinimaxH3/issues) so that the
discussion and resolution remain searchable.

## Validated platform

| Component | Validated configuration |
|---|---|
| GPU | NVIDIA GeForce RTX 4090, SM89 |
| OS | Linux x86-64 / WSL2 |
| Python | 3.10.20 |
| PyTorch | 2.13.0+cu130 |
| PyTorch CUDA runtime | 13.0 |
| Service build toolkit | CUDA 13.3 |
| Host memory | 64GB or more recommended; runtime residency is selected automatically |

Other GPU architectures have not been release-validated. The logical 8GB and
16GB routes were tested with hard allocator limits on SM89; a physical card of
the same capacity still requires device-specific validation.

## Quick start

### Fresh installation

This creates the runtime, checks out pinned upstream sources and downloads all
weights declared by `models/manifest.json`:

```bash
git clone https://github.com/PullMyBoots/X-MinimaxH3.git
cd X-MinimaxH3
./setup.sh --download-models --accept-model-license
./run.sh
```

`--accept-model-license` confirms that you have reviewed and accepted the
publishers' model licenses. It does not alter or replace those licenses.

### Reuse an existing installation

```bash
./setup.sh \
  --reuse-env /path/to/python-env \
  --model-dir /path/to/h3-model-store \
  --vendor-dir /path/to/vendor \
  --sparse-build-dir /path/to/compiled/sparge
./run.sh
```

The vendor directory must contain `MiniMax-H3/` and `LightX2V/`. The sparse
build argument can be omitted when the compatible extension is in the standard
sibling `extensions/` directory.

Open <http://127.0.0.1:8090>. Stop the service with:

```bash
./stop.sh
```

On WSL2, `./run.sh` automatically mirrors the hot source tree and caches to the
Linux filesystem, avoiding repeated imports and metadata access through
`/mnt/c`.

## Validation

Run a quick installation check, full model/revision preflight and regression
suite with:

```bash
./doctor.sh
./doctor.sh --full
./test.sh
```

The current source, Web UI, API contracts, long-video/SelfLift path, face
repair path, packaging boundary and ComfyUI connector are covered by the
release regression suite. Exact counts and the clean-archive verification are
recorded in [VALIDATION.md](VALIDATION.md). Model hashes and the earlier real
RTX 4090 generation matrix remain recorded as historical hardware evidence.

## Automatic resource execution

The console exposes only four choices: `W4A8 · FL2VA`, `W4A8 · Ref2VA`,
`INT8 · FL2VA` and `INT8 · Ref2VA`. It detects the GPU before loading weights:

| Detected VRAM | Private execution route | Available weights | Native first generation | Native H3 second sampling |
|---|---|---|---|---|
| 8–15GB | 8GB | W4A8 | native windows up to 720p × 15s; transparent long-horizon requests | up to 1080p |
| 16–23GB | 16GB | W4A8 or INT8 | experimental native windows up to 1080p for both weight tiers; transparent long-horizon requests | W4A8 up to 1080p; INT8 up to 1440p |
| 24GB+ | 24GB | W4A8 or INT8 | native windows up to 1080p for both weight tiers; transparent long-horizon requests | W4A8 up to 1080p; INT8 up to 1440p |

The Web console, REST API and generation nodes accept 1–300 seconds. Requests
above the physical native-window limit are planned automatically with a clean
39-frame joint A/V prefix and one final decode. The current real-video release
gate covers 480p × 30s, Base 20 steps, acceleration 75; higher resolutions and
longer durations should be validated on the target deployment. See the
[mechanism and acceptance evidence](docs/TRANSPARENT_LONG_HORIZON_2026_09_01.md).

The runtime manages H3, Qwen, VAE and subprocess residency internally. It
compiles the selected resource plan at model load and keeps per-step routing
out of the DiT hot path.

The September 1 resource gate completed 23/23 short end-to-end rows: every
FL2VA resolution in the 8/16/24GB matrix plus one real Ref2VA image-conditioned
boundary row per backend. See [VALIDATION.md](VALIDATION.md).

Out-of-envelope jobs are rejected instead of silently switching to another
backend. Resolution, duration and media limits exposed by the active service
are authoritative.

The Settings page can enable 3–8 second temporal windows for final sampling.
Shorter windows reduce per-window latency and peak memory; longer windows keep
more motion context. Phase alignment, overlap, latent blending, audio-token
authority and VRAM-safe shortening remain automatic.

## LoRA profiles

| Profile | Task family | Calibrated steps |
|---|---|---:|
| Larry Turbo v4-600 EMA | FL2VA / Ref2VA | 4–8, default 6 |
| LightX2V FL2VA Turbo v1.1 768p | FL2VA | 4 |
| LightX2V FL2VA Turbo v1.0 768p | FL2VA | 8 |
| LightX2V Ref2VA Turbo v0.1 | Ref2VA | 4 |

FL2VA and Ref2VA LightX2V adapters are task-specific and cannot be
interchanged. The settings page scans compatible files recursively under the
configured model store's `loras/` directory.

## ComfyUI

See the [English ComfyUI guide](integrations/comfyui/README.en.md) or the
[Chinese ComfyUI guide](integrations/comfyui/README.md).

Start X-MinimaxH3 first, select one of the four model choices in its console,
and then run:

```bash
./integrations/comfyui/start_comfyui.sh
```

Open <http://127.0.0.1:8188>. Example workflows are provided in
`integrations/comfyui/example_workflows/` in both English and Simplified
Chinese. The connector calls the same 8090
HTTP service and does not allocate another copy of H3 inside ComfyUI.

## Repository layout

```text
h3serve/                 Web/API, queue, scheduler and native H3 runtime
backends/                SM89 kernels and audited narrow binary runtime
static/                  bilingual Web console
ai-prompt-guides/        six AI-facing prompt-writing and output contracts
integrations/comfyui/    optional connector and example workflows
models/manifest.json     weight provenance, sizes and SHA-256 contract
scripts/                 setup, launch, validation and research utilities
tests/                   unit, contract and runtime regression tests
docs/                    user, deployment and architecture documentation
```

## Documentation

- [English user guide](docs/USER_GUIDE.en.md)
- [English deployment guide](docs/DEPLOYMENT.en.md)
- [中文用户指南](docs/USER_GUIDE.zh-CN.md)
- [中文部署指南](docs/DEPLOYMENT.zh-CN.md)
- [Native engine architecture](docs/NATIVE_ENGINE_ARCHITECTURE.md)
- [SelfLift progressive generation](docs/SELFLIFT_PROGRESSIVE_GENERATION.md)
- [Long-video creation studio v3](docs/INFINITE_CREATION_STUDIO_V3.md)
- [Automatic VRAM routing and hard host-RAM budget](docs/AUTOMATIC_RESOURCE_BUDGET_2026_08_31.md)
- [Third-party notices](THIRD_PARTY_NOTICES.md)
- [Release validation](VALIDATION.md)

## Security

The default server listens only on `127.0.0.1`. Set a strong
`H3_SERVE_API_KEY` before binding to a non-loopback address. The service does
not provide TLS or multi-tenant isolation; use a trusted reverse proxy for
network deployments. See [SECURITY.md](SECURITY.md).

## Acknowledgements

X-MinimaxH3 builds on important work from the MiniMax H3 community. In
particular, we thank:

- [Comfyui-MMH3-UltimateUpscale](https://github.com/bbaudio-2025/Comfyui-MMH3-UltimateUpscale)
  for the temporal/spatial chunking, overlap and stitching design underlying
  our native second-sampling planner. We adapted this design into a
  ComfyUI-independent runtime, added full-canvas admission, automatic resource
  routing, H3 phase alignment and condition-cache reuse.
- [Comfyui Minimax H3 Latent Upscaler](https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler)
  for the learned 3D latent-upscaling architecture and released H3 latent
  upscaler weights used to initialize second sampling.
- [comfyui-SelfLift](https://github.com/facok/comfyui-SelfLift) for the
  progressive clean-endpoint lifting mechanism that informed our native H3
  two-resolution trajectory. No upstream runtime source is embedded.
- [SageAttention](https://github.com/thu-ml/SageAttention) for the quantized
  dense-attention kernels and implementation foundation used by our SM89
  dense Attention path. X-MinimaxH3 adds H3-specific layout, quantization,
  long-sequence stability and scheduler integration around that foundation.
- [ComfyUI-H3-Continuum](https://github.com/ukr8b3g-cmyk/ComfyUI-H3-Continuum)
  for publishing and validating the masked joint audio/video prefix mechanism
  that informed our native transparent long-horizon executor.
- [ComfyUI-MiniMax-H3-LongMedia](https://github.com/vizart-vj/ComfyUI-MiniMax-H3-LongMedia)
  for its long-media systems work. X-MinimaxH3 independently adapted the
  compatible principles of one model lifecycle, deferred decode and localized
  prompt timelines without embedding its ComfyUI monkey-patch runtime.

The upstream projects are not affiliated with or responsible for
X-MinimaxH3. Their original licenses and notices remain in force; see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## License

This is a **public-source** release, not an open-source license grant. Original
project code is currently all rights reserved. Third-party software and model
artifacts retain their own licenses. Review [LICENSE](LICENSE),
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and
[models/manifest.json](models/manifest.json) before use or redistribution.
