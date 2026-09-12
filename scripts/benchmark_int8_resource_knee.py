#!/usr/bin/env python3
"""Locate INT8 host-RAM latency knees under hard 16/24-GiB VRAM tiers.

The public INT8 profiles currently pin broad model residencies and therefore
cannot reveal the true minimum host envelope.  This research harness reuses
the validated low-RAM sequential build, fixes the public INT8 CUDA launcher,
and varies only exact residency mechanics: cgroup RAM, a GPU-resident DiT
prefix, and complete streamed Block groups copied into pinned host slabs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmark_w4a8_resource_knee import run_case


def capacity_cases() -> list[dict]:
    result: list[dict] = []
    for vram_gib, launcher in (
        (16, "fl2va_int8_16gb"),
        (24, "fl2va_int8_24gb"),
    ):
        for memory_max_gib in (8, 10, 11, 12, 14, 16, 20, 24, 28, 32, 40, 48):
            result.append({
                "id": f"v{vram_gib}_m{memory_max_gib}_r0_p0",
                "vram_gib": vram_gib,
                "resident_blocks": 0,
                "pin_gib": 0.0,
                "memory_max_gib": memory_max_gib,
                "memory_profile": "w4a8_16gb",
                "launcher": launcher,
                "research_int8_curve": True,
                "cold_cache": True,
                # One full Actual DiT evaluation is sufficient to expose the
                # request peak and reject infeasible host envelopes quickly.
                # Base contracts accept 5--30 declared steps.  The checkpoint
                # stops after the first completed step, so this still pays for
                # only one full Actual DiT evaluation.
                "sampling_steps": 5,
                "checkpoint_step": 1,
                "acceleration": 95,
                "phase": "capacity",
            })
    return result


def full_product_cases() -> list[dict]:
    """Short solver, full 720p5 product path including AV decode."""
    result: list[dict] = []
    for vram_gib, launcher in (
        (16, "fl2va_int8_16gb"),
        (24, "fl2va_int8_24gb"),
    ):
        for memory_max_gib in (11, 12, 14, 16, 24, 32):
            result.append({
                "id": f"v{vram_gib}_720p5_full_m{memory_max_gib}_r0_p0",
                "vram_gib": vram_gib,
                "resident_blocks": 0,
                "pin_gib": 0.0,
                "memory_max_gib": memory_max_gib,
                "memory_profile": "w4a8_16gb",
                "launcher": launcher,
                "research_int8_curve": True,
                "cold_cache": True,
                "sampling_steps": 5,
                "acceleration": 95,
                "duration_seconds": 5,
                "execution_mode": "complete",
                "phase": "full_product",
            })
    for vram_gib, launcher in (
        (16, "fl2va_int8_16gb"),
        (24, "fl2va_int8_24gb"),
    ):
        result.append({
            "id": f"v{vram_gib}_720p15_full_m11_r0_p0",
            "vram_gib": vram_gib,
            "resident_blocks": 0,
            "pin_gib": 0.0,
            "memory_max_gib": 11,
            "memory_profile": "w4a8_16gb",
            "launcher": launcher,
            "research_int8_curve": True,
            "cold_cache": True,
            "sampling_steps": 5,
            "acceleration": 95,
            "duration_seconds": 15,
            "execution_mode": "complete",
            "phase": "full_product",
        })
    return result


def residency_smoke_cases() -> list[dict]:
    """Verify the largest useful exact GPU-resident prefixes first."""
    cases = [
        {
            "id": "v16_m24_r7_p0",
            "vram_gib": 16,
            "resident_blocks": 7,
            "pin_gib": 0.0,
            "memory_max_gib": 24,
            "memory_profile": "w4a8_16gb",
            "launcher": "fl2va_int8_16gb",
            "research_int8_curve": True,
            "cold_cache": True,
            "sampling_steps": 5,
            "checkpoint_step": 1,
            "acceleration": 95,
            "phase": "residency_smoke",
        },
        {
            "id": "v24_m24_r28_p0",
            "vram_gib": 24,
            "resident_blocks": 28,
            "pin_gib": 0.0,
            "memory_max_gib": 24,
            "memory_profile": "w4a8_16gb",
            "launcher": "fl2va_int8_24gb",
            "research_int8_curve": True,
            "cold_cache": True,
            "sampling_steps": 5,
            "checkpoint_step": 1,
            "acceleration": 95,
            "phase": "residency_smoke",
        },
    ]
    for case_id, vram_gib, launcher, pin_gib in (
        ("v24_m24_r0_p0", 24, "fl2va_int8_24gb", 0.0),
        ("v16_m40_r0_p16", 16, "fl2va_int8_16gb", 16.0),
        ("v24_m40_r0_p18", 24, "fl2va_int8_24gb", 18.0),
    ):
        cases.append({
            "id": case_id,
            "vram_gib": vram_gib,
            "resident_blocks": 0,
            "pin_gib": pin_gib,
            "memory_max_gib": 24 if pin_gib == 0 else 40,
            "memory_profile": "w4a8_16gb",
            "launcher": launcher,
            "research_int8_curve": True,
            "cold_cache": True,
            "sampling_steps": 5,
            "checkpoint_step": 1,
            "acceleration": 95,
            "phase": "residency_smoke",
        })
    return cases


def pin_curve_cases() -> list[dict]:
    """Convert increasing host headroom into exact pinned Block coverage."""
    result: list[dict] = []
    for vram_gib, launcher, points in (
        (
            16,
            "fl2va_int8_16gb",
            ((16, 4.0), (24, 12.0), (28, 16.0), (32, 18.0), (36, 21.0)),
        ),
        (
            24,
            "fl2va_int8_24gb",
            ((16, 4.0), (24, 12.0), (32, 18.0), (36, 21.0)),
        ),
    ):
        for memory_max_gib, pin_gib in points:
            result.append({
                "id": f"v{vram_gib}_m{memory_max_gib}_r0_p{pin_gib:g}",
                "vram_gib": vram_gib,
                "resident_blocks": 0,
                "pin_gib": pin_gib,
                "memory_max_gib": memory_max_gib,
                "memory_profile": "w4a8_16gb",
                "launcher": launcher,
                "research_int8_curve": True,
                "cold_cache": True,
                "sampling_steps": 5,
                "checkpoint_step": 1,
                "acceleration": 95,
                "phase": "pin_curve",
            })
    return result


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "runtime/validation/int8_resource_knee_20260830/coarse",
    )
    parser.add_argument("--port", type=int, default=8094)
    parser.add_argument("--only", nargs="*", default=())
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--ready-timeout", type=float, default=240.0)
    parser.add_argument("--job-timeout", type=float, default=600.0)
    parser.add_argument(
        "--phase",
        choices=("capacity", "full", "residency", "pin_curve", "all"),
        default="capacity",
    )
    # Attributes consumed by the shared exact-residency runner.
    parser.set_defaults(nsys=False, torch_profile=False)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "int8_resource_knee_raw.json"
    report = {
        "schema_version": "h3_int8_resource_knee_v1",
        "numerical_contract": (
            "same INT8 weights, 720p15 latent shape, V24 acceleration; only "
            "exact host/GPU residency mechanics vary"
        ),
        "cases": [],
    }
    if report_path.is_file() and not args.rerun:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    completed = {item["case"]["id"] for item in report.get("cases", [])}
    selected = set(args.only)
    cases = capacity_cases()
    if args.phase == "full":
        cases = full_product_cases()
    elif args.phase == "residency":
        cases = residency_smoke_cases()
    elif args.phase == "pin_curve":
        cases = pin_curve_cases()
    elif args.phase == "all":
        cases += (
            full_product_cases() + residency_smoke_cases() + pin_curve_cases()
        )
    for case in cases:
        if selected and case["id"] not in selected:
            continue
        if case["id"] in completed and not args.rerun:
            print(f"skip {case['id']}: already recorded", flush=True)
            continue
        print(f"run {case['id']}: {case}", flush=True)
        value = run_case(args, case)
        report["cases"] = [
            item for item in report.get("cases", [])
            if item["case"]["id"] != case["id"]
        ]
        report["cases"].append(value)
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(
            f"done {case['id']}: status={value.get('status')} "
            f"peak={value.get('cgroup_memory_peak_gib', 0):.3f}GiB",
            flush=True,
        )
        if value.get("error"):
            print(value["error"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
