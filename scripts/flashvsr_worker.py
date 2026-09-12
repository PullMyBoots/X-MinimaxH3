#!/usr/bin/env python3
"""Isolated FlashVSR v1.1 postprocessing worker for the H3 service."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import shutil
import subprocess
import sys
import time
import traceback
from fractions import Fraction
from pathlib import Path


PREFIX = "H3_UPSCALE_PROGRESS "
READY_PREFIX = "H3_UPSCALE_READY "
RESPONSE_PREFIX = "H3_UPSCALE_RESPONSE "
_REQUEST_ID: str | None = None


def progress(percent: float, detail: str) -> None:
    event = {
        "percent": percent,
        "stage": "upscaling",
        "detail": detail,
    }
    if _REQUEST_ID is not None:
        event["request_id"] = _REQUEST_ID
    print(PREFIX + json.dumps(event, ensure_ascii=False), flush=True)


def aligned(value: int, multiple: int = 128) -> int:
    return int(math.ceil(value / multiple) * multiple)


def fit_geometry(
    source_width: int, source_height: int, canvas_width: int, canvas_height: int
) -> tuple[int, int, int, int, int, int]:
    """Return resized geometry and the exact content box inside the canvas."""
    scale = min(canvas_width / source_width, canvas_height / source_height)
    resized_width = max(2, int(round(source_width * scale)))
    resized_height = max(2, int(round(source_height * scale)))
    pad_left = (canvas_width - resized_width) // 2
    pad_top = (canvas_height - resized_height) // 2
    return (
        resized_width,
        resized_height,
        pad_left,
        pad_top,
        pad_left + resized_width,
        pad_top + resized_height,
    )


def load_input(path: Path, width: int, height: int):
    import av
    import numpy as np
    import torch
    from PIL import Image

    frames = []
    content_box = None
    with av.open(str(path), mode="r") as container:
        stream = container.streams.video[0]
        rate = stream.average_rate or stream.guessed_rate or Fraction(24, 1)
        for frame in container.decode(stream):
            image = Image.fromarray(frame.to_ndarray(format="rgb24"), mode="RGB")
            source_width, source_height = image.size
            geometry = fit_geometry(source_width, source_height, width, height)
            resized_width, resized_height, pad_left, pad_top, right, bottom = geometry
            frame_box = (pad_left, pad_top, right, bottom)
            if content_box is None:
                content_box = frame_box
            elif content_box != frame_box:
                raise RuntimeError("input video changes frame dimensions mid-stream")
            image = image.resize(
                (resized_width, resized_height), Image.Resampling.LANCZOS
            )
            array = np.asarray(image, dtype=np.uint8).copy()
            pad_right = width - resized_width - pad_left
            pad_bottom = height - resized_height - pad_top
            # Edge extension keeps the generated frame geometry untouched and
            # avoids a black border influencing the restoration network.
            array = np.pad(
                array,
                ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
                mode="edge",
            )
            tensor = torch.from_numpy(array).permute(2, 0, 1).float()
            frames.append((tensor / 127.5 - 1.0).to(torch.bfloat16))
    if not frames:
        raise RuntimeError("input video contains no decodable frames")
    original_count = len(frames)
    # Tiny Long emits 13 frames for its first process window, then 8 frames per
    # additional window: output_count = F - 12 for valid F=8n+1. Pad to the
    # smallest valid F that covers the source, then trim back exactly.
    model_frames = 8 * math.ceil((original_count + 11) / 8) + 1
    frames.extend([frames[-1]] * (model_frames - original_count))
    video = torch.stack(frames, dim=1).unsqueeze(0)
    assert content_box is not None
    return video, original_count, model_frames, Fraction(rate), content_box


def tensor_to_frames(video, count: int, width: int, height: int, content_box):
    import numpy as np
    from PIL import Image

    arrays = ((video[:, :count].float() + 1.0) * 127.5).clamp(0, 255)
    arrays = arrays.permute(1, 2, 3, 0).cpu().numpy().astype(np.uint8)
    result = []
    for array in arrays:
        image = Image.fromarray(array, mode="RGB")
        image = image.crop(content_box)
        if image.size != (width, height):
            image = image.resize((width, height), Image.Resampling.LANCZOS)
        result.append(np.asarray(image, dtype=np.uint8))
    while len(result) < count:
        result.append(result[-1].copy())
    return result[:count]


def encode_video(frames, rate: Fraction, path: Path, audio_source: Path) -> None:
    import av

    path.parent.mkdir(parents=True, exist_ok=True)
    silent = path.with_name(f".{path.stem}.video.mp4")
    silent.unlink(missing_ok=True)
    with av.open(str(silent), mode="w", format="mp4") as container:
        stream = container.add_stream("libx264", rate=rate)
        stream.width = frames[0].shape[1]
        stream.height = frames[0].shape[0]
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "16", "preset": "medium"}
        for array in frames:
            frame = av.VideoFrame.from_ndarray(array, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        silent.replace(path)
        return
    command = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(silent), "-i", str(audio_source),
        "-map", "0:v:0", "-map", "1:a:0?", "-c:v", "copy",
        "-c:a", "aac", "-b:a", "192k", str(path),
    ]
    try:
        subprocess.run(command, check=True)
    finally:
        silent.unlink(missing_ok=True)


class FlashVSRRuntime:
    """CPU-resident FlashVSR model with exclusive, on-demand GPU ownership."""

    def __init__(self, source_root: Path, model_root: Path) -> None:
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        sys.path.insert(0, str(source_root))

        import torch
        from diffsynth import FlashVSRTinyLongPipeline, ModelManager
        from utils.TCDecoder import build_tcdecoder
        from utils.utils import Causal_LQ4x_Proj

        self.torch = torch
        self.model_root = model_root
        started = time.perf_counter()
        dtype = torch.bfloat16
        manager = ModelManager(torch_dtype=dtype, device="cpu")
        manager.load_models([
            str(model_root / "diffusion_pytorch_model_streaming_dmd.safetensors")
        ])
        pipe = FlashVSRTinyLongPipeline.from_model_manager(
            manager, device="cuda", torch_dtype=dtype
        )
        # Construct on CPU. GPU ownership begins only when a request reaches
        # run(), so the H3 engine can keep using the card while this daemon is
        # idle or preloading.
        pipe.denoising_model().LQ_proj_in = Causal_LQ4x_Proj(
            in_dim=3, out_dim=1536, layer_num=1
        ).to("cpu", dtype=dtype)
        pipe.denoising_model().LQ_proj_in.load_state_dict(
            torch.load(model_root / "LQ_proj_in.ckpt", map_location="cpu"),
            strict=True,
        )
        pipe.TCDecoder = build_tcdecoder(
            new_channels=[512, 256, 128, 128],
            new_latent_channels=16 + 768,
            device="cpu",
            dtype=dtype,
        )
        pipe.TCDecoder.load_state_dict(
            torch.load(model_root / "TCDecoder.ckpt", map_location="cpu"),
            strict=False,
        )
        pipe.enable_vram_management(num_persistent_param_in_dit=None)
        self.context = torch.load(model_root / "posi_prompt.pth", map_location="cpu")
        self.pipe = pipe
        self.load_seconds = time.perf_counter() - started
        self.request_count = 0

    def _release_gpu(self) -> None:
        """Return task-local CUDA state while retaining immutable CPU weights."""
        torch = self.torch
        pipe = self.pipe
        if not torch.cuda.is_available():
            raise RuntimeError("FlashVSR requires a CUDA GPU")
        try:
            if hasattr(pipe.denoising_model(), "clear_cross_kv"):
                pipe.denoising_model().clear_cross_kv()
            if hasattr(pipe.denoising_model(), "LQ_proj_in"):
                pipe.denoising_model().LQ_proj_in.clear_cache()
            for module in pipe.denoising_model().modules():
                if hasattr(module, "local_attn_mask"):
                    module.local_attn_mask = None
            pipe.load_models_to_device([])
            # VRAM wrappers cover the large Linear/Conv modules; modulation
            # parameters and positional buffers are ordinary tensors. Move the
            # complete graph back as the final symmetry/safety boundary.
            pipe.denoising_model().to("cpu")
            pipe.TCDecoder.clean_mem()
            pipe.TCDecoder.to("cpu")
            pipe.prompt_emb_posi = None
            pipe.timestep = None
            pipe.t = None
            pipe.t_mod = None
        finally:
            gc.collect()
            torch.cuda.empty_cache()

    def run_one(
        self,
        *,
        input_path: Path,
        output_path: Path,
        target_width: int,
        target_height: int,
        gpu_input_limit_mib: int,
    ) -> dict:
        torch = self.torch
        pipe = self.pipe
        timings: dict[str, float] = {}
        started = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        internal_width = aligned(target_width)
        internal_height = aligned(target_height)
        lq_video = video = frames = None
        try:
            stage = time.perf_counter()
            progress(2, "读取并对齐输入视频")
            lq_video, original_count, model_frames, rate, content_box = load_input(
                input_path, internal_width, internal_height
            )
            timings["decode_preprocess_seconds"] = time.perf_counter() - stage

            # For ordinary 360p/480p/720p clips, one H2D transfer is both faster
            # and mathematically identical to the old per-window .to(cuda).
            # Long 1080p clips retain the CPU-streaming fallback for VRAM safety.
            input_mib = lq_video.numel() * lq_video.element_size() / (1024 * 1024)
            gpu_resident_input = input_mib <= max(0, gpu_input_limit_mib)

            stage = time.perf_counter()
            progress(7, "复用内存热态权重并准备GPU")
            # The vendor wrapper manages large Linear/Conv modules but leaves a
            # few modulation parameters and buffers unmanaged. A full graph
            # move first keeps those tensors device-consistent; the following
            # wrapper onload calls maintain their state bookkeeping.
            pipe.denoising_model().to("cuda")
            pipe.TCDecoder.to("cuda")
            pipe.init_cross_kv(context_tensor=self.context)
            pipe.load_models_to_device(["dit", "vae"])
            if gpu_resident_input:
                lq_video = lq_video.to("cuda")
            torch.cuda.synchronize()
            timings["gpu_prepare_seconds"] = time.perf_counter() - stage

            stage = time.perf_counter()
            progress(15, "FlashVSR 单步流式增强")
            video = pipe(
                prompt="",
                negative_prompt="",
                cfg_scale=1.0,
                num_inference_steps=1,
                seed=0,
                LQ_video=lq_video,
                num_frames=model_frames,
                height=internal_height,
                width=internal_width,
                is_full_block=False,
                if_buffer=True,
                topk_ratio=1.5 * 768 * 1280 / (internal_height * internal_width),
                kv_ratio=2.5,
                local_range=9,
                color_fix=True,
            )
            torch.cuda.synchronize()
            timings["inference_seconds"] = time.perf_counter() - stage

            stage = time.perf_counter()
            progress(92, "编码视频并保留原始音轨")
            frames = tensor_to_frames(
                video, original_count, target_width, target_height, content_box
            )
            encode_video(frames, rate, output_path, input_path)
            timings["postprocess_encode_seconds"] = time.perf_counter() - stage
            peak_allocated_mib = torch.cuda.max_memory_allocated() / (1024 * 1024)
            peak_reserved_mib = torch.cuda.max_memory_reserved() / (1024 * 1024)
            self.request_count += 1
            timings["task_seconds"] = time.perf_counter() - started
            result = {
                "peak_allocated_mib": round(peak_allocated_mib, 1),
                "peak_reserved_mib": round(peak_reserved_mib, 1),
                "input_mib": round(input_mib, 1),
                "gpu_resident_input": gpu_resident_input,
                "model_load_seconds": round(self.load_seconds, 3),
                "request_count": self.request_count,
                "warm": self.request_count > 1,
                "timings": {key: round(value, 3) for key, value in timings.items()},
            }
            print(PREFIX + json.dumps({
                "percent": 100,
                "stage": "upscaling",
                "detail": "FlashVSR 超分完成",
                **result,
                **({"request_id": _REQUEST_ID} if _REQUEST_ID else {}),
            }, ensure_ascii=False), flush=True)
            return result
        finally:
            del frames, video, lq_video
            self._release_gpu()


def run(args) -> None:
    """One-shot compatibility entry used by manual diagnostics."""
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    runtime = FlashVSRRuntime(args.source_root, args.model_root)
    runtime.run_one(
        input_path=args.input,
        output_path=args.output,
        target_width=args.target_width,
        target_height=args.target_height,
        gpu_input_limit_mib=args.gpu_input_limit_mib,
    )


def serve(args) -> None:
    """Line-delimited JSON protocol for the persistent service-side daemon."""
    global _REQUEST_ID
    runtime = FlashVSRRuntime(args.source_root, args.model_root)
    print(READY_PREFIX + json.dumps({
        "ready": True,
        "model_load_seconds": round(runtime.load_seconds, 3),
        "pid": os.getpid(),
    }), flush=True)
    for raw in sys.stdin:
        raw = raw.strip()
        if not raw:
            continue
        request_id = None
        try:
            request = json.loads(raw)
            request_id = str(request["request_id"])
            _REQUEST_ID = request_id
            if request.get("command") == "shutdown":
                print(RESPONSE_PREFIX + json.dumps({
                    "request_id": request_id, "ok": True, "shutdown": True
                }), flush=True)
                return
            if request.get("command") != "upscale":
                raise ValueError("unsupported FlashVSR daemon command")
            result = runtime.run_one(
                input_path=Path(request["input"]),
                output_path=Path(request["output"]),
                target_width=int(request["target_width"]),
                target_height=int(request["target_height"]),
                gpu_input_limit_mib=int(request.get("gpu_input_limit_mib", 1536)),
            )
            print(RESPONSE_PREFIX + json.dumps({
                "request_id": request_id, "ok": True, **result
            }), flush=True)
        except Exception as error:
            traceback.print_exc(file=sys.stderr)
            sys.stderr.flush()
            print(RESPONSE_PREFIX + json.dumps({
                "request_id": request_id,
                "ok": False,
                "error": f"{type(error).__name__}: {error}",
            }, ensure_ascii=False), flush=True)
        finally:
            _REQUEST_ID = None


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--target-width", type=int)
    parser.add_argument("--target-height", type=int)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--gpu-input-limit-mib", type=int, default=1536)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    started = time.perf_counter()
    if args.serve:
        serve(args)
    else:
        required = (args.input, args.output, args.target_width, args.target_height)
        if any(value is None for value in required):
            raise SystemExit(
                "--input, --output, --target-width and --target-height are required"
            )
        run(args)
        print(json.dumps({"elapsed_seconds": time.perf_counter() - started}), flush=True)
