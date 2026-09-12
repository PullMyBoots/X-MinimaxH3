#!/usr/bin/env python3
"""Sweep exact Sparge SM89 CTA/WARP-Q geometries at one H3 long-video chunk.

The selector is deliberately excluded.  Every candidate receives the same
already-quantized Q/K/FP8-V and the same delta-encoded sparse LUT.  CTA64 gets
the exact duplicated LUT and Q scale required to represent each original
CTA128 row as two halves.  This makes any output drift or speed change an
executor property instead of a different sparsity policy.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
from pathlib import Path

import torch
from torch.utils.cpp_extension import load


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query-tokens", type=int, default=32768)
    parser.add_argument("--key-tokens", type=int, default=220003)
    parser.add_argument("--heads", type=int, default=56)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--selected-key-blocks", type=int, default=230)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=9)
    parser.add_argument("--seed", type=int, default=4090)
    parser.add_argument(
        "--max-registers",
        type=int,
        default=0,
        help=(
            "Optional nvcc --maxrregcount value for an isolated occupancy/spill "
            "experiment. Zero preserves the production compiler decision."
        ),
    )
    parser.add_argument(
        "--sparge-source",
        type=Path,
        default=Path(
            os.environ.get(
                "H3_NATIVE_SPARGE_SOURCE",
                "/root/h3-new-serve-runtime/vendor/SpargeAttn",
            )
        ),
    )
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=Path(
            "/root/h3-new-serve-runtime/extensions/"
            "sparge-sm89-tile-sweep-torch213-cu133"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "runtime/calibration/v19_long_video_20260825/"
            "sparge_sm89_tile_sweep_32k_220k_r1.json"
        ),
    )
    return parser.parse_args()


def measure(operation, *, warmup: int, repeat: int) -> tuple[list[float], torch.Tensor]:
    result = None
    for _ in range(warmup):
        result = operation()
    torch.cuda.synchronize()
    samples: list[float] = []
    for _ in range(repeat):
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        result = operation()
        stop.record()
        stop.synchronize()
        samples.append(float(start.elapsed_time(stop)))
    assert result is not None
    return samples, result


def compare(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float | bool]:
    exact = bool(torch.equal(reference, candidate))
    delta = (reference.float() - candidate.float()).abs()
    return {
        "exact_equal": exact,
        "max_abs": float(delta.max().cpu()),
        "mean_abs": float(delta.mean().cpu()),
        "cosine": float(
            torch.nn.functional.cosine_similarity(
                reference.float().reshape(1, -1),
                candidate.float().reshape(1, -1),
            ).cpu()
        ),
    }


@torch.inference_mode()
def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (8, 9):
        raise SystemExit("this sweep requires one RTX 4090 / SM89 GPU")
    if args.head_dim != 128:
        raise SystemExit("the experimental extension is specialized for head_dim=128")
    if args.query_tokens <= 0 or args.query_tokens % 128:
        raise SystemExit("query tokens must be a positive multiple of 128")
    if args.key_tokens <= 0:
        raise SystemExit("key tokens must be positive")
    if args.warmup < 0 or args.repeat <= 0:
        raise SystemExit("warmup must be non-negative and repeat positive")
    if args.max_registers and not 64 <= args.max_registers <= 255:
        raise SystemExit("max registers must be zero or lie inside [64, 255]")

    key_blocks = math.ceil(args.key_tokens / 64)
    if not 1 <= args.selected_key_blocks <= key_blocks:
        raise SystemExit("selected key block count is outside the KV block range")
    source = (
        Path(__file__).resolve().parents[1]
        / "experiments/sparge_sm89_tile_sweep/sparge_tile_sweep.cu"
    )
    args.build_dir.mkdir(parents=True, exist_ok=True)
    cuda_flags = [
        "-O3",
        "-std=c++20",
        "--use_fast_math",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-gencode=arch=compute_89,code=sm_89",
    ]
    if args.max_registers:
        cuda_flags.append(f"--maxrregcount={args.max_registers}")
    module = load(
        name="h3_sparge_sm89_tile_sweep",
        sources=[str(source)],
        extra_include_paths=[str((args.sparge_source / "csrc").resolve())],
        extra_cflags=["-O3", "-std=c++20"],
        extra_cuda_cflags=cuda_flags,
        build_directory=str(args.build_dir),
        verbose=True,
    )

    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    q = torch.randint(
        -32,
        33,
        (1, args.heads, args.query_tokens, args.head_dim),
        device=device,
        dtype=torch.int8,
    )
    k = torch.randint(
        -32,
        33,
        (1, args.heads, args.key_tokens, args.head_dim),
        device=device,
        dtype=torch.int8,
    )
    padded_key_tokens = key_blocks * 64
    v = torch.full(
        (1, args.heads, args.head_dim, padded_key_tokens),
        0.25,
        device=device,
        dtype=torch.float8_e4m3fn,
    )
    query_blocks_128 = args.query_tokens // 128
    lut128 = torch.zeros(
        (1, args.heads, query_blocks_128, key_blocks),
        device=device,
        dtype=torch.int32,
    )
    # Contiguous selected keys are represented as cumulative increments:
    # first block 0, then +1 for every following block.
    lut128[..., 1 : args.selected_key_blocks] = 1
    valid128 = torch.full(
        (1, args.heads, query_blocks_128),
        args.selected_key_blocks,
        device=device,
        dtype=torch.int32,
    )
    qscale128 = torch.rand(
        (1, args.heads, query_blocks_128), device=device, dtype=torch.float32
    ).mul_(0.01).add_(0.001)
    kscale = torch.rand(
        (1, args.heads, key_blocks), device=device, dtype=torch.float32
    ).mul_(0.01).add_(0.001)
    vscale = torch.rand(
        (1, args.heads, args.head_dim), device=device, dtype=torch.float32
    ).mul_(0.01).add_(0.001)
    pv_threshold = torch.full(
        (args.heads,), 50.0, device=device, dtype=torch.float32
    )
    scale = args.head_dim**-0.5

    common128 = (
        q,
        k,
        v,
        lut128,
        valid128,
        pv_threshold,
        qscale128,
        kscale,
        vscale,
        scale,
    )
    operations = {
        "cta128_warpq32_reference": lambda: module.run_cta128_warpq32(*common128),
        "cta128_warpq16": lambda: module.run_cta128_warpq16(*common128),
    }

    # CTA64 represents every original 128-query sparse row as two identical
    # 64-query rows.  Q is unchanged; only its scale/LUT row metadata doubles.
    lut64 = lut128.repeat_interleave(2, dim=2).contiguous()
    valid64 = valid128.repeat_interleave(2, dim=2).contiguous()
    qscale64 = qscale128.repeat_interleave(2, dim=2).contiguous()
    common64 = (
        q,
        k,
        v,
        lut64,
        valid64,
        pv_threshold,
        qscale64,
        kscale,
        vscale,
        scale,
    )
    operations["cta64_warpq16"] = lambda: module.run_cta64_warpq16(*common64)

    # CTA256 consumes the union of each adjacent pair of CTA128 masks.  The
    # synthetic sweep uses identical masks, so this is the maximum-reuse case.
    # Per-warp Q scales reproduce the original per-128-block scales exactly:
    # four 32-row warps receive the first scale and four the second.
    query_blocks_256 = query_blocks_128 // 2
    lut256 = lut128[:, :, 0::2].contiguous()
    valid256 = valid128[:, :, 0::2].contiguous()
    qscale256 = (
        qscale128.view(1, args.heads, query_blocks_256, 2, 1)
        .expand(-1, -1, -1, -1, 4)
        .reshape(1, args.heads, query_blocks_256 * 8)
        .contiguous()
    )
    common256 = (
        q,
        k,
        v,
        lut256,
        valid256,
        pv_threshold,
        qscale256,
        kscale,
        vscale,
        scale,
    )
    operations["cta256_warpq32_perwarp"] = (
        lambda: module.run_cta256_warpq32_perwarp(*common256)
    )

    report: dict[str, object] = {
        "schema_version": "h3_sparge_sm89_tile_sweep_v1",
        "warning": "Executor-only synthetic quantized tensors; no quality claim.",
        "runtime": {
            "device": torch.cuda.get_device_name(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "nvcc": os.environ.get("CUDA_HOME", ""),
            "source": str(source),
            "max_registers": args.max_registers or None,
            "cuda_flags": cuda_flags,
        },
        "shape": {
            "query_tokens": args.query_tokens,
            "key_tokens": args.key_tokens,
            "heads": args.heads,
            "head_dim": args.head_dim,
            "key_blocks": key_blocks,
            "selected_key_blocks": args.selected_key_blocks,
            "selected_fraction": args.selected_key_blocks / key_blocks,
        },
        "variants": {},
    }
    outputs: dict[str, torch.Tensor] = {}
    for name, operation in operations.items():
        samples, output = measure(
            operation, warmup=args.warmup, repeat=args.repeat
        )
        outputs[name] = output
        report["variants"][name] = {
            "samples_ms": samples,
            "median_ms": statistics.median(samples),
            "min_ms": min(samples),
            "max_ms": max(samples),
        }

    reference_name = "cta128_warpq32_reference"
    reference = outputs[reference_name]
    reference_ms = report["variants"][reference_name]["median_ms"]
    for name, output in outputs.items():
        row = report["variants"][name]
        row["speedup_vs_reference"] = reference_ms / row["median_ms"]
        row["output_vs_reference"] = compare(reference, output)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
