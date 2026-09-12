# Third-party notices

X-MinimaxH3 combines or interoperates with third-party software and model
artifacts. This file is a provenance guide, not a substitute for the upstream
license text. Model weights are downloaded separately and are never relicensed
by this repository.

| Component | Purpose | Upstream / license location |
|---|---|---|
| MiniMax H3 | Base model architecture and inference source | `MiniMaxAI/MiniMax-H3`, pinned by `scripts/install.sh`; review the upstream model/source terms |
| LightX2V | H3 runtime integration and LoRA conversion reference | `ModelTC/LightX2V`, Apache-2.0 upstream |
| MiniMax H3 INT8 weights | FL2VA/Ref2VA diffusion, Qwen encoder and VAEs | `Comfy-Org/MiniMax-H3`; exact revisions and hashes in `models/manifest.json` |
| MiniMax H3 W4A8 weights | 8GB FL2VA/Ref2VA profiles | `starsfriday/MiniMax-H3-w4a8`; exact revision and hashes in the manifest |
| Larry Turbo LoRA | Optional accelerated FL2VA profile | `larryvrh/MiniMax-H3-Turbo-Lora`; exact revision and hash in the manifest |
| LightX2V Turbo LoRAs | Optional FL2VA 4/8-step and Ref2VA 4-step profiles | `lightx2v/Minimax-h3-Turbo`; exact revision and hashes in the manifest |
| MMH3 UltimateUpscale | Temporal/spatial second-sampling pieces, overlap and stitching design | [`bbaudio-2025/Comfyui-MMH3-UltimateUpscale`](https://github.com/bbaudio-2025/Comfyui-MMH3-UltimateUpscale), MIT; pinned evaluation revision `6db8fa5a4e4ca0718d2ea8d08002ea899fe27721`; license copy in `third_party_licenses/MMH3-UltimateUpscale-LICENSE` |
| H3 latent upscaler | Learned 3D latent resize used by native second sampling | [`LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler`](https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler), Apache-2.0; exact weight revision and hash in the manifest |
| comfyui-SelfLift | Design reference for progressive clean-endpoint lifting; no upstream runtime code is bundled | [`facok/comfyui-SelfLift`](https://github.com/facok/comfyui-SelfLift), inspected at `b143ed2731438e606c7a6c70ee580a1f86e7bf72`; review the upstream repository terms |
| FlashVSR | Optional isolated temporal video restoration backend | [`OpenImagingLab/FlashVSR`](https://github.com/OpenImagingLab/FlashVSR), Apache-2.0; pinned at `b527c6f285fb30df530f5febc8b45764a789c961`; vendored license in `third_party/flashvsr/LICENSE` and exact weight hashes in the manifest |
| Block Sparse Attention | CUDA attention extension for the isolated FlashVSR environment | [`mit-han-lab/Block-Sparse-Attention`](https://github.com/mit-han-lab/Block-Sparse-Attention), pinned at `49d6c39e4dc0303442cda3bb758b3925d4399c49`; license in `third_party/block_sparse_attention/LICENSE` |
| OpenCV Zoo YuNet | Lightweight CPU face detector used by automatic face repair | `face_detection_yunet_2023mar.onnx`, SHA-256 `8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4`; license in `third_party_licenses/YUNET_LICENSE` |
| SageAttention | Quantized dense-attention acceleration and SM89 implementation foundation | [`thu-ml/SageAttention`](https://github.com/thu-ml/SageAttention), Apache-2.0; license copy in `third_party_licenses/SageAttention-LICENSE` |
| SpargeAttention | Sparse attention integration | Review the pinned upstream source installed by `scripts/install.sh` |
| Comfy Kitchen 0.2.28 | SM89 INT8/W4A8 kernels; a narrow vendored runtime is included | License and notice in `third_party_licenses/Comfy-Kitchen-*` |
| H3 SiLU/temb grid | 5.3MB deterministic kernel-calibration lookup asset, not a model checkpoint | `backends/turbo/custom_node/h3_silu_temb_grid.safetensors`, SHA-256 recorded in `RELEASE_MANIFEST.json` |
| FastVQA | Optional validation tooling | `third_party_licenses/FasterVQA-LICENSE` |
| ComfyUI connector | Optional HTTP workflow integration | Connector code in `integrations/comfyui`; ComfyUI remains separately licensed upstream |
| ComfyUI-H3-Continuum | Design reference for masked joint A/V prefix continuation; no upstream runtime code is bundled | [`ukr8b3g-cmyk/ComfyUI-H3-Continuum`](https://github.com/ukr8b3g-cmyk/ComfyUI-H3-Continuum), MIT; evaluated at `0a7c6703b7715689c7aab4177e44e5fcd318fe5b` |
| ComfyUI-MiniMax-H3-LongMedia | Design reference for one-session long-media execution, deferred decode and prompt localization; no upstream runtime code is bundled | [`vizart-vj/ComfyUI-MiniMax-H3-LongMedia`](https://github.com/vizart-vj/ComfyUI-MiniMax-H3-LongMedia), Apache-2.0; evaluated at `4a31874addc3d1b632589ac0454eb990a2736547` |

Additional Python and system dependencies retain their own licenses. Run
`pip-licenses` in the configured environment if a deployment needs a complete
environment-specific software bill of materials.

The MiniMax H3 model license and each weight publisher's repository terms may
restrict commercial use or redistribution. Review them before downloading,
using or redistributing any weight.
