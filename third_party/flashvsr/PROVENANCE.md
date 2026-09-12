# FlashVSR provenance

- Upstream: `https://github.com/OpenImagingLab/FlashVSR`
- Vendored commit: `b527c6f285fb30df530f5febc8b45764a789c961`
- License: Apache License 2.0 (see `LICENSE` in this directory)
- Vendored scope: `diffsynth/` plus the two small model-construction modules in
  `utils/`. The service owns the process wrapper and audio mux code.
- Local modification: the unused ModelScope downloader import is lazy. This
  keeps the pinned local-weight inference path independent of ModelScope.
- Local modification: package exports are narrowed to `ModelManager` and
  `FlashVSRTinyLongPipeline`; unrelated trainers, prompters and model families
  are not imported into the online inference worker.
- Weight source: `JunhaoZhuang/FlashVSR-v1.1`; exact byte sizes and SHA-256
  hashes are recorded in `../../models/manifest.json`.

No runtime import or executable path points to the developer's external
`toolkit/tools/FlashVSR` checkout.
