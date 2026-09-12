#!/usr/bin/env python3
"""Screen Sol/CuTe SM89 against SageAttention at the real 720p/5s shape."""

from __future__ import annotations

import argparse
import json
import site
import statistics
import sys
import time
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    workspace = root.parents[1]
    old_serve = workspace / "subprojects-main/main/release/serve"
    sol_source = (
        workspace
        / "subprojects-main/main/DIT-knowledge/sources/projects/sana_sol"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", type=int, default=34871)
    parser.add_argument("--protected", type=int, default=831)
    parser.add_argument("--taus", default="0.8,1.0,1.2,1.3,1.4,1.5")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--seed", type=int, default=82303)
    parser.add_argument(
        "--sol-source",
        type=Path,
        default=sol_source / "techniques/sparse_backends",
    )
    parser.add_argument(
        "--sol-deps",
        type=Path,
        default=old_serve / "runtime/extensions/sol-attn-deps",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root
        / "runtime/calibration/cuda13_optimization/sol_attention_720p5_sm89.json",
    )
    args = parser.parse_args()
    try:
        args.taus = tuple(float(value) for value in args.taus.split(","))
    except ValueError as error:
        parser.error(f"--taus must be comma-separated floats: {error}")
    if (
        args.sequence <= 0
        or not 0 <= args.protected <= args.sequence
        or args.warmup < 1
        or args.repeat < 3
        or any(tau < 0 for tau in args.taus)
    ):
        parser.error("invalid shape, repeat count or tau")
    return args


def measure(function, *, warmup: int, repeat: int) -> tuple[torch.Tensor, list[float], int]:
    output = None
    for _ in range(warmup):
        output = function()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    samples: list[float] = []
    for _ in range(repeat):
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        output = function()
        stop.record()
        stop.synchronize()
        samples.append(float(start.elapsed_time(stop)))
    assert output is not None
    return output, samples, int(torch.cuda.max_memory_allocated())


def error_stats(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float]:
    delta = candidate.float().sub_(reference.float())
    return {
        "mean_abs": float(delta.abs().mean()),
        "max_abs": float(delta.abs().max()),
        "rmse": float(delta.square().mean().sqrt()),
        "cosine": float(
            torch.nn.functional.cosine_similarity(
                reference.float().reshape(1, -1),
                candidate.float().reshape(1, -1),
            )
        ),
    }


@torch.inference_mode()
def main() -> int:
    args = parse_args()
    # CuTe DSL's wheel exposes its nested package through a .pth file, which
    # must be processed by addsitedir rather than a raw PYTHONPATH entry.
    site.addsitedir(str(args.sol_deps.resolve()))
    sys.path.insert(0, str(args.sol_source.resolve()))

    from sol_attn import get_sol_attn_backend, sol_attn

    from h3serve.native_engine.model import sage_attention_sm89
    from h3serve.native_engine.sm89_policy import configure_sm89_runtime

    runtime = configure_sm89_runtime(quant_backend="cuda", smoke_test=True)
    generator = torch.Generator("cuda:0").manual_seed(args.seed)
    shape = (args.sequence, 56, 128)
    query = torch.randn(shape, device="cuda:0", dtype=torch.bfloat16, generator=generator)
    key = torch.randn(shape, device="cuda:0", dtype=torch.bfloat16, generator=generator)
    value = torch.randn(shape, device="cuda:0", dtype=torch.bfloat16, generator=generator)

    reference, dense_samples, dense_peak = measure(
        lambda: sage_attention_sm89(query, key, value),
        warmup=args.warmup,
        repeat=args.repeat,
    )
    dense_median = statistics.median(dense_samples)
    rows = []
    for tau in args.taus:
        compile_start = time.perf_counter()

        def candidate() -> torch.Tensor:
            output = sol_attn(
                query.unsqueeze(0),
                key.unsqueeze(0),
                value.unsqueeze(0),
                tau=tau,
                thresh_type="diag",
                sink_start=0,
                sink_tokens=args.protected,
            ).squeeze(0)
            if args.protected:
                output[: args.protected].copy_(
                    sage_attention_sm89(query[: args.protected], key, value)
                )
            return output

        output, samples, peak = measure(
            candidate, warmup=args.warmup, repeat=args.repeat
        )
        median = statistics.median(samples)
        rows.append(
            {
                "tau": tau,
                "samples_ms": samples,
                "median_ms": median,
                "speedup_vs_dense": dense_median / median,
                "peak_allocated_gib": peak / (1024**3),
                "first_candidate_wall_seconds_including_compile": (
                    time.perf_counter() - compile_start
                ),
                "full_vs_dense": error_stats(reference, output),
                "prefix_vs_dense": error_stats(
                    reference[: args.protected], output[: args.protected]
                )
                if args.protected
                else None,
                "video_vs_dense": error_stats(
                    reference[args.protected :], output[args.protected :]
                ),
            }
        )
        del output
    report = {
        "schema_version": 1,
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "shape": [1, args.sequence, 56, 128],
        "protected_tokens": args.protected,
        "sol_backend": get_sol_attn_backend(0),
        "sol_source": str(args.sol_source.resolve()),
        "sol_deps": str(args.sol_deps.resolve()),
        "runtime": runtime.to_dict(),
        "dense": {
            "backend": "sageattention-sm89",
            "samples_ms": dense_samples,
            "median_ms": dense_median,
            "peak_allocated_gib": dense_peak / (1024**3),
        },
        "candidates": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
