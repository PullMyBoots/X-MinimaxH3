# X-MinimaxH3 1.1.0

Release date: 2026-09-12

## Highlights

- Three focused workspaces: Single Video Creation, Long Video Creation and
  Task Center, with complete English and Simplified Chinese UI coverage.
- One two-handle trajectory for first-pass/final sampling steps and
  resolutions. First-pass and final acceleration are controlled independently.
- SelfLift progressive H3 generation with a learned 3D latent lift between the
  low- and high-resolution trajectory segments.
- Long-video online creation keeps cumulative low-resolution previews and clean
  formal checkpoints. JSON one-click creation skips intermediate preview
  decoding and directly completes the final high-resolution branch.
- Ref2VA JSON one-click creation supports a complete independent Picture/Audio
  reference set on every window. Omitting the set inherits the previous window;
  explicitly supplying it replaces the previous set without exposing local paths.
- Six standalone AI-facing TXT guides cover FL2VA and Ref2VA prompt authoring
  for single video, online long-video windows and one-click long-video JSON.
- Full-film final sampling uses a continuous low-resolution latent timeline,
  prompt-owned temporal views, optional 3–8 second windows, overlap fusion,
  stable audio-token authority and one final decode.
- Enabled single-video temporal final sampling now chooses the minimum window
  count permitted by the configured maximum and distributes the timeline
  evenly across those windows. This avoids an under-filled terminal view while
  retaining the existing overlap, Sigma schedule, audio authority and fusion.
- Long-video full-film final sampling now applies the same minimum-count,
  balanced allocation inside each prompt-owned creator range. It preserves
  prompt boundaries and read-only temporal halos while avoiding a maximum-size
  view followed by an inefficient short tail.
- Automatic FL2VA face repair detects and ranks under-resolved face tracks,
  packs selected regions into a square atlas and uses a four-step Turbo pass.
- Long-video projects can be removed from the creation workspace; task history
  and completed media remain independently manageable.

## Compatibility and boundaries

- Validated release platform: Linux x86-64 or WSL2 with an NVIDIA RTX 4090
  (SM89), Python 3.10 and the pinned CUDA/PyTorch stack.
- Face repair is exposed for completed FL2VA jobs. Ref2VA jobs keep the button
  disabled because the four-step Turbo repair route is not compatible with the
  active Ref2VA execution contract.
- FlashVSR temporal restoration uses its isolated Python 3.11 / Torch 2.6
  runtime. Native H3 generation and SelfLift remain in the Python 3.10 runtime.
- Model weights, user uploads, generated task output and runtime caches are not
  distributed. The two MP4 files under `assets/demos/` are public documentation
  examples.

## Validation

- Main release suite: 959 tests passed with 4 intentional skips.
- Clean extracted archive: 959 tests passed with 4 intentional skips.
- ComfyUI connector: 25 tests passed.
- Shell, Python, JSON and browser JavaScript syntax checks passed.
- A standalone `x_minimaxh3-1.1.0-py3-none-any.whl` metadata build passed.

See [VALIDATION.md](VALIDATION.md) for the evidence ledger and historical RTX
4090 real-generation results.
