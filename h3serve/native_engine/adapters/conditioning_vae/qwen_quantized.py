"""Standalone multimodal Qwen3-VL prefix for the local packed H3 checkpoint.

It consumes the Comfy-Org packed NVFP4/AWQ safetensors layout through the Apache-2.0
``comfy-kitchen`` tensor primitives, but does not import or initialize
ComfyUI.  Only decoder layers 0..49 are evaluated because MiniMax-H3 consumes
the layer-50 hidden representation.
"""

from __future__ import annotations

import gc
import hashlib
import itertools
import json
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open

from ...local_checkpoint_cache import drop_file_page_cache
from ...runtime.pinned_pool import pack_pinned_tensors
from .preprocess import prepare_keyframes


@dataclass(frozen=True, slots=True)
class TextEncodingResult:
    prompt_embeds: torch.Tensor
    text_token_tags: torch.Tensor
    token_count: int
    elapsed_seconds: float
    peak_allocated_gib: float
    input_ids: torch.Tensor | None = None
    keyframe_count: int = 0


@dataclass(frozen=True, slots=True)
class _VisionFeatureCacheEntry:
    """One content-addressed Ref2VA/keyframe presentation on pinned host RAM."""

    key: tuple
    image_grid_thw: torch.Tensor | None
    video_grid_thw: torch.Tensor | None
    video_block_timestamps: tuple[tuple[float, ...], ...]
    vision_embeds: torch.Tensor
    deepstack: tuple[torch.Tensor, ...]


@dataclass(frozen=True, slots=True)
class _PinnedLayer:
    tensors: dict[str, torch.Tensor]
    slabs: tuple[torch.Tensor, ...]
    allocated_bytes: int


def _rms_norm(value: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    # Keep the same fused reduction/rounding boundary as ComfyUI's Qwen3-VL
    # reference.  A hand-written FP32 variance is mathematically close but the
    # per-layer difference compounds across the truncated 50-layer encoder.
    return F.rms_norm(
        value,
        (value.shape[-1],),
        weight=weight.to(device=value.device, dtype=value.dtype),
        eps=eps,
    )


class PackedQwen3VLT2AVConditioner:
    """Block-streamed text and keyframe conditioner for one RTX 4090."""

    hidden_size = 5120
    intermediate_size = 25600
    num_heads = 64
    num_kv_heads = 8
    head_dim = 128
    num_layers = 50
    # Qwen3-VL uses a ten-times larger base than the plain Qwen variants.
    # This must match the H3 Qwen3VL_32B checkpoint definition exactly; a
    # 500_000 base still produces finite embeddings but changes every prompt
    # trajectory and therefore defeats same-seed parity with the reference.
    rope_theta = 5_000_000.0
    mrope_section = (24, 20, 20)

    def __init__(
        self,
        checkpoint: str | Path,
        tokenizer_path: str | Path,
        *,
        device: str | torch.device = "cuda",
        dtype: torch.dtype = torch.float32,
        layers: int = 50,
        cache_pinned_weights: bool = False,
        processor_path: str | Path | None = None,
        model_config_path: str | Path | None = None,
        cache_vision_features: bool = True,
        layer_cache_dir: str | Path | None = None,
    ) -> None:
        self.checkpoint = Path(checkpoint)
        self.tokenizer_path = Path(tokenizer_path)
        self.device = torch.device(device)
        self.dtype = dtype
        self.layers = int(layers)
        self.cache_pinned_weights = bool(cache_pinned_weights)
        self.cache_vision_features = bool(cache_vision_features)
        self.layer_cache_dir = (
            None if layer_cache_dir is None else Path(layer_cache_dir)
        )
        model_root = self.tokenizer_path.parent
        if processor_path is not None:
            self.processor_path = Path(processor_path)
        else:
            processor_candidate = model_root / "processor"
            # The source release de-duplicates Qwen tokenizer and processor
            # assets into one offline directory. Development checkouts keep a
            # separate processor/ directory, so prefer it when present and
            # otherwise use the tokenizer directory as the processor source.
            self.processor_path = (
                processor_candidate
                if processor_candidate.is_dir()
                else self.tokenizer_path
            )
        self.model_config_path = Path(
            model_config_path or model_root / "text_encoder" / "config.json"
        )
        if not self.checkpoint.is_file():
            raise FileNotFoundError(self.checkpoint)
        if not self.tokenizer_path.is_dir():
            raise FileNotFoundError(self.tokenizer_path)
        if not 1 <= self.layers <= self.num_layers:
            raise ValueError("layers must be between 1 and 50")
        if self.device.type != "cuda":
            raise ValueError("packed NVFP4 conditioner currently requires CUDA")
        self._tokenizer = None
        self._kitchen = None
        self._host_cache: dict[str, torch.Tensor] | None = None
        self._processor = None
        self._vision_encoder = None
        self._vision_feature_cache: _VisionFeatureCacheEntry | None = None
        self.vision_cache_hits = 0
        self.vision_cache_misses = 0
        if self.layer_cache_dir is not None and not self.layer_cache_dir.is_dir():
            raise FileNotFoundError(self.layer_cache_dir)

    @staticmethod
    def _content_key(path: str | Path) -> tuple[str, bytes]:
        resolved = Path(path).resolve()
        return str(resolved), hashlib.sha256(resolved.read_bytes()).digest()

    def _vision_cache_key(self, request) -> tuple | None:
        reference_images = tuple(getattr(request, "reference_images", ()) or ())
        reference_videos = tuple(getattr(request, "reference_videos", ()) or ())
        first_frame = getattr(request, "first_frame", None)
        last_frame = getattr(request, "last_frame", None)
        if not reference_images and not reference_videos and first_frame is None and last_frame is None:
            return None
        return (
            "ref2va" if reference_images or reference_videos else "keyframes",
            tuple(self._content_key(path) for path in reference_images),
            tuple(self._content_key(path) for path in reference_videos),
            None if first_frame is None else self._content_key(first_frame),
            None if last_frame is None else self._content_key(last_frame),
            int(request.width),
            int(request.height),
            int(request.num_frames),
            str(getattr(request, "reference_image_resolution", "720p")),
            str(getattr(request, "reference_video_resolution", "360p")),
        )

    @staticmethod
    def _pin_feature(value: torch.Tensor) -> torch.Tensor:
        host = value.detach().to("cpu")
        return host if host.is_pinned() else host.pin_memory()

    def _load_runtime(self):
        if self._kitchen is None:
            import comfy_kitchen as kitchen

            # Qwen's packed NVFP4 dequantizer is validated through Triton on
            # this SM89 host.  Do *not* change the process-wide backend order
            # here: the H3 DiT runs afterwards and its INT8/QK primitives are
            # substantially faster in Comfy-Kitchen's CUDA extension.  The
            # individual NVFP4 call below uses a thread-local backend override
            # which is restored as soon as dequantization returns.
            self._kitchen = kitchen
        return self._kitchen

    def _load_tokenizer(self):
        if self._tokenizer is None:
            try:
                # Transformers 5 no longer re-exports every model class from
                # the package root.  Keep the old import as a compatibility
                # fallback for the Transformers 4 release environment.
                from transformers.models.qwen2 import Qwen2TokenizerFast
            except ImportError:  # pragma: no cover - Transformers 4 fallback
                from transformers import Qwen2TokenizerFast

            self._tokenizer = Qwen2TokenizerFast.from_pretrained(
                self.tokenizer_path,
                local_files_only=True,
            )
        return self._tokenizer

    def _load_processor(self):
        if self._processor is None:
            if not self.processor_path.is_dir():
                raise FileNotFoundError(self.processor_path)
            try:
                # Transformers 5 exposes Qwen3-VL through its model package,
                # while Transformers 4 also exported it at the package root.
                from transformers.models.qwen3_vl import Qwen3VLProcessor
            except ImportError:  # pragma: no cover - Transformers 4 fallback
                from transformers import Qwen3VLProcessor

            self._processor = Qwen3VLProcessor.from_pretrained(
                self.processor_path,
                local_files_only=True,
            )
        return self._processor

    def _load_vision_encoder(self):
        if self._vision_encoder is None:
            if not self.model_config_path.is_file():
                raise FileNotFoundError(self.model_config_path)
            from .qwen_vision import MiniMaxH3Qwen3VLVisionTower

            config = json.loads(self.model_config_path.read_text(encoding="utf-8"))
            self._vision_encoder = MiniMaxH3Qwen3VLVisionTower.from_single_file(
                self.checkpoint,
                config["vision_config"],
            )
        return self._vision_encoder

    @staticmethod
    def _get_tensor(checkpoint, key: str) -> torch.Tensor:
        if isinstance(checkpoint, dict):
            return checkpoint[key]
        return checkpoint.get_tensor(key)

    @staticmethod
    def _contains(checkpoint, key: str) -> bool:
        return key in checkpoint if isinstance(checkpoint, dict) else key in checkpoint.keys()

    def _plain(self, checkpoint, key: str) -> torch.Tensor:
        return self._get_tensor(checkpoint, key).to(
            self.device,
            dtype=self.dtype,
            non_blocking=isinstance(checkpoint, dict),
        )

    def _linear(self, checkpoint, prefix: str, value: torch.Tensor) -> torch.Tensor:
        kitchen = self._load_runtime()
        pre_key = f"{prefix}.pre_quant_scale"
        if self._contains(checkpoint, pre_key):
            value = value * self._get_tensor(checkpoint, pre_key).to(
                self.device,
                dtype=value.dtype,
                non_blocking=isinstance(checkpoint, dict),
            )
        qweight = self._get_tensor(checkpoint, f"{prefix}.weight").to(
            self.device, non_blocking=isinstance(checkpoint, dict)
        )
        block_scale = self._get_tensor(checkpoint, f"{prefix}.weight_scale").to(
            self.device, non_blocking=isinstance(checkpoint, dict)
        )
        tensor_scale = self._get_tensor(checkpoint, f"{prefix}.weight_scale_2").to(
            self.device, non_blocking=isinstance(checkpoint, dict)
        )
        with kitchen.use_backend("triton"):
            weight = kitchen.dequantize_nvfp4(
                qweight,
                tensor_scale,
                block_scale,
                self.dtype,
            )
        output = F.linear(value, weight)
        del qweight, block_scale, tensor_scale, weight
        return output

    def _embedding(self, checkpoint, input_ids: torch.Tensor) -> torch.Tensor:
        qweight = self._get_tensor(checkpoint, "model.embed_tokens.weight")
        scale = self._get_tensor(checkpoint, "model.embed_tokens.weight_scale")
        # Gather on CPU before dequantization; copying the full 1.55 GiB BF16
        # embedding table to the accelerator is unnecessary for a short prompt.
        rows = qweight[input_ids.cpu()].to(self.device, dtype=torch.float32)
        row_scales = scale[input_ids.cpu()].to(self.device, dtype=torch.float32)
        # The checkpoint declares a BF16 embedding table.  Comfy's optimized
        # row gather dequantizes into that storage dtype first and only then
        # promotes the selected rows to the Qwen FP32 compute dtype.  Keeping
        # the intermediate BF16 store is important: multiplying INT8 rows and
        # scales directly into FP32 changes layer-zero inputs even though the
        # same checkpoint and token ids are used.
        return (rows * row_scales).to(torch.bfloat16).to(self.dtype)

    def prepare_host_cache(self) -> None:
        """Prepare text and vision weights once on the service startup path.

        Vision construction used to happen in the first I2AV request and cost
        several seconds even though it can overlap DiT/VAE startup.  Keeping
        it CPU-resident here changes no model math and makes the first
        multimodal request representative of normal service latency.
        """

        if self._host_cache is None:
            selected_prefixes = tuple(
                f"model.layers.{index}." for index in range(self.layers)
            )
            keys: list[str] = []
            sources: list[torch.Tensor] = []
            with safe_open(self.checkpoint, framework="pt", device="cpu") as checkpoint:
                for key in checkpoint.keys():
                    if not (
                        key.startswith(selected_prefixes)
                        or key in {
                            "model.embed_tokens.weight",
                            "model.embed_tokens.weight_scale",
                        }
                    ):
                        continue
                    keys.append(key)
                    sources.append(checkpoint.get_tensor(key))
            packed = pack_pinned_tensors(sources)
            self._host_cache = dict(zip(keys, packed.tensors))
            # Retain explicit slab owners for clarity and for reporting. Tensor
            # views also retain their storage, so this does not duplicate RAM.
            self._host_cache_slabs = packed.slabs
            self._host_cache_allocated_bytes = packed.allocated_bytes
        self._load_processor()
        self._load_vision_encoder()

    @property
    def host_cache_bytes(self) -> int:
        if self._host_cache is None:
            return 0
        return sum(
            tensor.numel() * tensor.element_size()
            for tensor in self._host_cache.values()
        )

    @property
    def host_cache_ready(self) -> bool:
        return self._host_cache is not None

    @property
    def vision_feature_cache_bytes(self) -> int:
        entry = self._vision_feature_cache
        if entry is None:
            return 0
        return sum(
            int(value.numel()) * int(value.element_size())
            for value in (entry.vision_embeds, *entry.deepstack)
        )

    def _position_embeddings(
        self,
        token_count: int,
        position_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inv_freq = 1.0 / (
            self.rope_theta
            ** (
                torch.arange(0, self.head_dim, 2, device=self.device, dtype=torch.float32)
                / self.head_dim
            )
        )
        if position_ids is None:
            position_ids = torch.arange(token_count, device=self.device)[None].expand(
                3, -1
            )
        else:
            position_ids = position_ids.to(self.device)
        if tuple(position_ids.shape) != (3, token_count):
            raise ValueError(
                "Qwen3-VL position_ids must have shape "
                f"{(3, token_count)}, got {tuple(position_ids.shape)}"
            )
        frequencies_3d = position_ids.to(torch.float32)[..., None] * inv_freq[
            None, None, :
        ]
        frequencies = frequencies_3d[0].clone()
        for dimension, offset in enumerate((1, 2), start=1):
            target = slice(offset, self.mrope_section[dimension] * 3, 3)
            frequencies[..., target] = frequencies_3d[dimension, ..., target]
        embeddings = torch.cat((frequencies, frequencies), dim=-1)
        return embeddings.cos().to(self.dtype), embeddings.sin().to(self.dtype)

    @staticmethod
    def _mrope_position_ids(
        input_ids: torch.Tensor,
        token_types: torch.Tensor,
        spatial_merge_size: int,
        image_grid_thw: torch.Tensor | None = None,
        video_grid_thw: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Unbatched Qwen3-VL M-RoPE indices for image/video presentations."""

        groups = []
        for modality, values in itertools.groupby(
            enumerate(token_types.tolist()), key=lambda item: item[1]
        ):
            values = list(values)
            groups.append((int(modality), values[0][0], values[-1][0] + 1))
        if video_grid_thw is not None:
            video_grid_thw = torch.repeat_interleave(
                video_grid_thw, video_grid_thw[:, 0], dim=0
            ).clone()
            video_grid_thw[:, 0] = 1
        grid_iters = {
            1: iter(image_grid_thw) if image_grid_thw is not None else None,
            2: iter(video_grid_thw) if video_grid_thw is not None else None,
        }
        current_position = 0
        positions = []
        for modality, start, stop in groups:
            if modality == 0:
                length = stop - start
                positions.append(
                    torch.arange(length, device=input_ids.device)
                    .view(1, -1)
                    .expand(3, -1)
                    + current_position
                )
                current_position += length
                continue
            if modality not in (1, 2) or grid_iters[modality] is None:
                raise ValueError(f"conditioner received unsupported modality {modality}")
            grid = next(grid_iters[modality])
            grid_t = int(grid[0])
            grid_h = int(grid[1]) // spatial_merge_size
            grid_w = int(grid[2]) // spatial_merge_size
            temporal = torch.arange(grid_t, device=input_ids.device)
            height = torch.arange(grid_h, device=input_ids.device) + current_position
            width = torch.arange(grid_w, device=input_ids.device) + current_position
            t_grid, h_grid, w_grid = torch.meshgrid(
                temporal, height, width, indexing="ij"
            )
            block = torch.stack((t_grid, h_grid, w_grid), dim=0).reshape(3, -1)
            block[0] += current_position
            positions.append(block)
            current_position += max(int(grid[1]), int(grid[2])) // spatial_merge_size
        result = torch.cat(positions, dim=1)
        if result.shape[1] != input_ids.shape[0]:
            raise RuntimeError(
                f"Qwen3-VL M-RoPE produced {result.shape[1]} positions for "
                f"{input_ids.shape[0]} tokens"
            )
        return result

    def _prepare_multimodal(self, request):
        reference_paths = tuple(getattr(request, "reference_images", ()) or ())
        reference_video_paths = tuple(getattr(request, "reference_videos", ()) or ())
        reference_audio_paths = tuple(getattr(request, "reference_audios", ()) or ())
        cache_key = self._vision_cache_key(request)
        cached_vision = (
            self._vision_feature_cache
            if self.cache_vision_features and cache_key is not None
            else None
        )
        if cached_vision is not None and cached_vision.key == cache_key:
            self.vision_cache_hits += 1
            image_grid_thw = cached_vision.image_grid_thw
            video_grid_thw = cached_vision.video_grid_thw
            video_block_timestamps = cached_vision.video_block_timestamps
            vision_embeds = cached_vision.vision_embeds.to(
                self.device, non_blocking=True
            )
            deepstack = tuple(
                value.to(self.device, non_blocking=True)
                for value in cached_vision.deepstack
            )
            images = videos = ()
            image_vision = video_vision = None
        else:
            if cache_key is not None:
                self.vision_cache_misses += 1
            videos = []
            images = []
            if reference_paths or reference_video_paths or reference_audio_paths:
                from .preprocess import prepare_reference_images, prepare_reference_videos

                images = list(prepare_reference_images(request)) if reference_paths else []
                videos = list(prepare_reference_videos(request)) if reference_video_paths else []
            else:
                keyframes = prepare_keyframes(request)
                images = [item.image for item in keyframes]
            image_vision = video_vision = None
            image_grid_thw = video_grid_thw = None
            video_block_timestamps = tuple(
                tuple(float(value) for value in item.qwen_block_timestamps)
                for item in videos
            )
            vision_embeds = None
            deepstack = ()
        if cache_key is None and not reference_audio_paths:
            return None
        tokenizer = self._load_tokenizer()
        processor = self._load_processor()
        if cached_vision is None or cached_vision.key != cache_key:
            image_vision = (
                processor.image_processor(images=images, return_tensors="pt")
                if images else None
            )
            image_grid_thw = None if image_vision is None else image_vision["image_grid_thw"]
            video_vision = (
                processor.video_processor(
                    videos=[item.qwen_frames for item in videos],
                    do_sample_frames=False,
                    return_tensors="pt",
                ) if videos else None
            )
            video_grid_thw = None if video_vision is None else video_vision["video_grid_thw"]
        merge_area = int(processor.image_processor.merge_size) ** 2
        token_ids: list[int] = []
        token_tags: list[int] = []
        vision_start = int(tokenizer.convert_tokens_to_ids("<|vision_start|>"))
        image_pad = int(tokenizer.convert_tokens_to_ids("<|image_pad|>"))
        vision_end = int(tokenizer.convert_tokens_to_ids("<|vision_end|>"))
        for index, grid in enumerate(() if image_grid_thw is None else image_grid_thw, start=1):
            count = int(grid.prod().item()) // merge_area
            label = tokenizer(
                f"<Picture {index}>: ", add_special_tokens=False
            )["input_ids"]
            block = [vision_start] + [image_pad] * count + [vision_end]
            token_ids.extend(int(value) for value in label)
            token_tags.extend([1] * len(label))
            token_ids.extend(block)
            token_tags.extend([0] * len(block))
        video_pad = int(tokenizer.convert_tokens_to_ids("<|video_pad|>"))
        for index, (timestamps, grid) in enumerate(
            zip(video_block_timestamps, () if video_grid_thw is None else video_grid_thw), start=1
        ):
            count = int(grid[1]) * int(grid[2]) // merge_area
            label = tokenizer(f"<Video {index}>: ", add_special_tokens=False)["input_ids"]
            token_ids.extend(int(value) for value in label)
            token_tags.extend([1] * len(label))
            for timestamp in timestamps:
                label = tokenizer(f"<{timestamp:.1f} seconds>", add_special_tokens=False)["input_ids"]
                block = [vision_start] + [video_pad] * count + [vision_end]
                token_ids.extend(int(value) for value in label)
                token_tags.extend([1] * len(label))
                token_ids.extend(block)
                token_tags.extend([0] * len(block))
        # Standalone reference audio is represented in Qwen's presentation by
        # a stable label only. Its waveform is encoded by Audio-VAE and enters
        # the DiT packed sequence; Qwen never consumes raw audio samples.
        for index, _path in enumerate(reference_audio_paths, start=1):
            label = tokenizer(f"<Audio {index}>: ", add_special_tokens=False)["input_ids"]
            token_ids.extend(int(value) for value in label)
            token_tags.extend([1] * len(label))
        prompt_ids = tokenizer(request.prompt, add_special_tokens=False)["input_ids"]
        if not prompt_ids:
            raise ValueError("prompt produced no tokens")
        token_ids.extend(int(value) for value in prompt_ids)
        token_tags.extend([1] * len(prompt_ids))
        ids = torch.tensor(token_ids, dtype=torch.long)
        tags = torch.tensor(token_tags, dtype=torch.long)
        token_types = torch.tensor(
            processor.create_mm_token_type_ids([token_ids])[0], dtype=torch.long
        )
        position_ids = self._mrope_position_ids(
            ids,
            token_types,
            int(processor.image_processor.merge_size),
            image_grid_thw,
            video_grid_thw,
        )

        if vision_embeds is None and image_vision is None and video_vision is None:
            return (
                ids,
                tags,
                position_ids,
                None,
                None,
                (),
                len(reference_audio_paths),
            )

        if vision_embeds is not None:
            return (
                ids,
                tags,
                position_ids,
                ids == image_pad if video_grid_thw is None else (ids == image_pad) | (ids == video_pad),
                vision_embeds,
                deepstack,
                int(0 if image_grid_thw is None else image_grid_thw.shape[0])
                + int(0 if video_grid_thw is None else video_grid_thw.shape[0])
                + len(reference_audio_paths),
            )

        encoder = self._load_vision_encoder().to(self.device)
        parameter = next(encoder.parameters())
        try:
            with torch.inference_mode():
                image_embeds = image_deepstack = video_embeds = video_deepstack = None
                if image_vision is not None:
                    image_embeds, image_deepstack = encoder(
                        image_vision["pixel_values"].to(self.device, parameter.dtype),
                        image_grid_thw.to(self.device),
                    )
                if video_vision is not None:
                    video_embeds, video_deepstack = encoder(
                        video_vision["pixel_values_videos"].to(self.device, parameter.dtype),
                        video_grid_thw.to(self.device),
                    )
            image_mask, video_mask = ids == image_pad, ids == video_pad
            vision_mask = image_mask | video_mask
            feature_dim = (image_embeds if image_embeds is not None else video_embeds).shape[-1]
            vision_embeds = torch.empty(
                (int(vision_mask.sum()), feature_dim), device=self.device, dtype=parameter.dtype
            )
            image_joint, video_joint = image_mask[vision_mask], video_mask[vision_mask]
            if image_embeds is not None:
                vision_embeds[image_joint] = image_embeds
            if video_embeds is not None:
                vision_embeds[video_joint] = video_embeds
            source_stack = image_deepstack if image_deepstack is not None else video_deepstack
            deepstack = []
            for layer in range(len(source_stack or ())):
                rows = torch.empty_like(vision_embeds)
                if image_deepstack is not None:
                    rows[image_joint] = image_deepstack[layer]
                if video_deepstack is not None:
                    rows[video_joint] = video_deepstack[layer]
                deepstack.append(rows)
            if int(vision_mask.sum()) != int(vision_embeds.shape[0]):
                raise RuntimeError(
                    "Qwen vision placeholder count does not match encoded rows"
                )
            if self.cache_vision_features and cache_key is not None:
                self._vision_feature_cache = _VisionFeatureCacheEntry(
                    key=cache_key,
                    image_grid_thw=(
                        None if image_grid_thw is None else image_grid_thw.clone()
                    ),
                    video_grid_thw=(
                        None if video_grid_thw is None else video_grid_thw.clone()
                    ),
                    video_block_timestamps=video_block_timestamps,
                    vision_embeds=self._pin_feature(vision_embeds),
                    deepstack=tuple(self._pin_feature(value) for value in deepstack),
                )
            return (
                ids,
                tags,
                position_ids,
                vision_mask,
                vision_embeds.detach(),
                tuple(value.detach() for value in deepstack),
                int(0 if image_grid_thw is None else image_grid_thw.shape[0])
                + int(0 if video_grid_thw is None else video_grid_thw.shape[0])
                + len(reference_audio_paths),
            )
        finally:
            encoder.to("cpu")
            gc.collect()
            torch.cuda.empty_cache()

    def _attention(
        self,
        checkpoint,
        prefix: str,
        value: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        tokens = value.shape[0]
        q = self._linear(checkpoint, f"{prefix}.q_proj", value).view(
            tokens, self.num_heads, self.head_dim
        )
        k = self._linear(checkpoint, f"{prefix}.k_proj", value).view(
            tokens, self.num_kv_heads, self.head_dim
        )
        v = self._linear(checkpoint, f"{prefix}.v_proj", value).view(
            tokens, self.num_kv_heads, self.head_dim
        )
        q = _rms_norm(
            q,
            self._plain(checkpoint, f"{prefix}.q_norm.weight"),
        )
        k = _rms_norm(
            k,
            self._plain(checkpoint, f"{prefix}.k_norm.weight"),
        )
        cos, sin = position_embeddings
        # Match the reference's split-half mul -> in-place addcmul stores.
        # Expressing this as ``q*cos + rotate(q)*sin`` may select a different
        # fusion and changes BF16 rounding at every encoder layer.
        def apply_rope(value: torch.Tensor) -> torch.Tensor:
            cosine = cos[:, None, :]
            sine = sin[:, None, :]
            rotated = value * cosine
            midpoint = value.shape[-1] // 2
            rotated[..., :midpoint].addcmul_(
                value[..., midpoint:], -sine[..., :midpoint]
            )
            rotated[..., midpoint:].addcmul_(
                value[..., :midpoint], sine[..., :midpoint]
            )
            return rotated

        q, k = apply_rope(q), apply_rope(k)
        causal_mask = torch.empty(
            (tokens, tokens), device=value.device, dtype=value.dtype
        ).fill_(torch.finfo(value.dtype).min / 4).triu_(1)
        output = F.scaled_dot_product_attention(
            q.transpose(0, 1).unsqueeze(0),
            k.transpose(0, 1).unsqueeze(0),
            v.transpose(0, 1).unsqueeze(0),
            attn_mask=causal_mask,
            is_causal=False,
            enable_gqa=True,
        )
        output = output.squeeze(0).transpose(0, 1).reshape(tokens, -1)
        return self._linear(checkpoint, f"{prefix}.o_proj", output)

    def _apply_decoder_layer(
        self,
        checkpoint,
        index: int,
        hidden: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        *,
        device_vision_mask: torch.Tensor | None,
        deepstack: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        prefix = f"model.layers.{index}"
        residual = hidden
        normed = _rms_norm(
            hidden,
            self._plain(checkpoint, f"{prefix}.input_layernorm.weight"),
        )
        hidden = residual + self._attention(
            checkpoint,
            f"{prefix}.self_attn",
            normed,
            position_embeddings,
        )
        residual = hidden
        normed = _rms_norm(
            hidden,
            self._plain(checkpoint, f"{prefix}.post_attention_layernorm.weight"),
        )
        gate = self._linear(checkpoint, f"{prefix}.mlp.gate_proj", normed)
        up = self._linear(checkpoint, f"{prefix}.mlp.up_proj", normed)
        hidden = residual + self._linear(
            checkpoint,
            f"{prefix}.mlp.down_proj",
            F.silu(gate) * up,
        )
        del residual, normed, gate, up
        if index < len(deepstack):
            if device_vision_mask is None:
                raise ValueError("deepstack rows require a vision mask")
            hidden = hidden.clone()
            hidden[device_vision_mask] += deepstack[index].to(
                self.device, hidden.dtype
            )
        return hidden

    def _load_pinned_layer(self, index: int) -> _PinnedLayer:
        """Read one execution-ordered layer into short-lived pinned slabs."""

        if self.layer_cache_dir is None:
            raise RuntimeError("Qwen layer cache is not configured")
        path = self.layer_cache_dir / f"layer-{index:02d}.safetensors"
        with safe_open(path, framework="pt", device="cpu") as checkpoint:
            keys = tuple(checkpoint.keys())
            sources = tuple(checkpoint.get_tensor(key) for key in keys)
        # Small slabs keep the rolling two-layer window close to the logical
        # ~262 MiB per layer rather than rounding it to two 256 MiB slabs.
        packed = pack_pinned_tensors(sources, slab_bytes=32 * 1024**2)
        # ``pack_pinned_tensors`` has copied every byte needed by the CUDA
        # consumer.  The source shard will not be reused during this encode,
        # so retaining its clean mmap pages only makes a 16/32 GiB host look
        # full and forces unrelated model pages through reclaim later.  Drop
        # each execution-ordered shard immediately; the two-layer pinned
        # rolling window remains available for read/compute overlap.
        del sources
        drop_file_page_cache(path)
        return _PinnedLayer(
            tensors=dict(zip(keys, packed.tensors)),
            slabs=packed.slabs,
            allocated_bytes=packed.allocated_bytes,
        )

    def _encode_streamed_layers(
        self,
        hidden: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        *,
        device_vision_mask: torch.Tensor | None,
        deepstack: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        """Overlap layer N compute with the sequential read of layer N+1."""

        retained: deque[tuple[torch.cuda.Event, _PinnedLayer]] = deque()
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="h3-qwen-read") as pool:
            future = pool.submit(self._load_pinned_layer, 0)
            for index in range(self.layers):
                layer = future.result()
                future = (
                    pool.submit(self._load_pinned_layer, index + 1)
                    if index + 1 < self.layers else None
                )
                hidden = self._apply_decoder_layer(
                    layer.tensors,
                    index,
                    hidden,
                    position_embeddings,
                    device_vision_mask=device_vision_mask,
                    deepstack=deepstack,
                )
                complete = torch.cuda.Event()
                complete.record()
                retained.append((complete, layer))
                # Keep at most two completed/in-flight pinned layers. Waiting
                # here is safe and still leaves disk preparation overlapped.
                if len(retained) > 2:
                    event, _old = retained.popleft()
                    event.synchronize()
            while retained:
                event, _layer = retained.popleft()
                event.synchronize()
        return hidden

    def _encode_tokens(
        self,
        ids: torch.Tensor,
        tags: torch.Tensor,
        *,
        position_ids: torch.Tensor | None = None,
        vision_mask: torch.Tensor | None = None,
        vision_embeds: torch.Tensor | None = None,
        deepstack: tuple[torch.Tensor, ...] = (),
        keyframe_count: int = 0,
    ) -> TextEncodingResult:
        torch.cuda.reset_peak_memory_stats(self.device)
        started = time.perf_counter()
        if self.cache_pinned_weights and self._host_cache is None:
            self.prepare_host_cache()
        checkpoint_context = (
            nullcontext(self._host_cache)
            if self._host_cache is not None
            else safe_open(self.checkpoint, framework="pt", device="cpu")
        )
        with torch.inference_mode(), checkpoint_context as checkpoint:
            hidden = self._embedding(checkpoint, ids)
            device_vision_mask = None
            if vision_embeds is not None:
                if vision_mask is None or int(vision_mask.sum()) != vision_embeds.shape[0]:
                    raise ValueError(
                        "Qwen vision placeholder count does not match vision embeddings"
                    )
                device_vision_mask = vision_mask.to(self.device)
                hidden = hidden.clone()
                hidden[device_vision_mask] = vision_embeds.to(
                    self.device, hidden.dtype
                )
            position_embeddings = self._position_embeddings(
                hidden.shape[0], position_ids
            )
            if self._host_cache is None and self.layer_cache_dir is not None:
                # The embedding rows above still come from the canonical file;
                # decoder shards contain the exact same tensors in execution
                # order and change only I/O scheduling.
                hidden = self._encode_streamed_layers(
                    hidden,
                    position_embeddings,
                    device_vision_mask=device_vision_mask,
                    deepstack=deepstack,
                )
            else:
                for index in range(self.layers):
                    hidden = self._apply_decoder_layer(
                        checkpoint,
                        index,
                        hidden,
                        position_embeddings,
                        device_vision_mask=device_vision_mask,
                        deepstack=deepstack,
                    )
        if self._host_cache is None:
            # The compact 64GB profile streams the packed file per cache miss.
            # Do not let its ~15GB clean page cache compete with long-video
            # host scratch after encoding; the next miss can reread the fast
            # native-disk copy while exact prompt/reference hits bypass Qwen.
            drop_file_page_cache(self.checkpoint)
        torch.cuda.synchronize(self.device)
        elapsed = time.perf_counter() - started
        result = hidden.unsqueeze(0).contiguous()
        tags = tags.to(self.device)
        peak = torch.cuda.max_memory_allocated(self.device) / (1024**3)
        gc.collect()
        torch.cuda.empty_cache()
        return TextEncodingResult(
            prompt_embeds=result,
            text_token_tags=tags,
            token_count=int(ids.shape[0]),
            elapsed_seconds=elapsed,
            peak_allocated_gib=peak,
            input_ids=ids,
            keyframe_count=keyframe_count,
        )

    def encode_prompt(self, prompt: str) -> TextEncodingResult:
        if not prompt.strip():
            raise ValueError("prompt cannot be empty")
        tokenizer = self._load_tokenizer()
        token_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        if not token_ids:
            raise ValueError("prompt produced no tokens")
        ids = torch.tensor(token_ids, dtype=torch.long)
        tags = torch.ones((len(token_ids),), dtype=torch.long)
        return self._encode_tokens(ids, tags)

    def encode_request(self, request) -> TextEncodingResult:
        if not request.prompt.strip():
            raise ValueError("prompt cannot be empty")
        prepared = self._prepare_multimodal(request)
        if prepared is None:
            return self.encode_prompt(request.prompt)
        (
            ids,
            tags,
            position_ids,
            vision_mask,
            vision_embeds,
            deepstack,
            keyframe_count,
        ) = prepared
        return self._encode_tokens(
            ids,
            tags,
            position_ids=position_ids,
            vision_mask=vision_mask,
            vision_embeds=vision_embeds,
            deepstack=deepstack,
            keyframe_count=keyframe_count,
        )

    def encode(self, request) -> dict[str, torch.Tensor | float | int]:
        result = self.encode_request(request)
        return {
            "prompt_embeds": result.prompt_embeds,
            "text_token_tags": result.text_token_tags,
            "token_count": result.token_count,
            "elapsed_seconds": result.elapsed_seconds,
            "peak_allocated_gib": result.peak_allocated_gib,
        }


__all__ = ["PackedQwen3VLT2AVConditioner", "TextEncodingResult"]
