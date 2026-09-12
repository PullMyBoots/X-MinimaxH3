#!/usr/bin/env python3
"""Validate the public 8/16/24-GiB resource matrix through the real HTTP API.

The validator intentionally uses a short complete generation (five declared
steps, acceleration 95) instead of a checkpoint.  A successful row therefore
proves text conditioning, DiT execution, both VAEs and muxing under the selected
CUDA allocator ceiling and service-process cgroup limit.  The report is
resumable and retains the generated MP4 for every passing row.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import aiohttp


TERMINAL = {"succeeded", "failed", "cancelled", "checkpointed"}
PROMPT = """integrated_multimodal_description: [Shot 1]
A single continuous locked-off product shot shows a red ceramic cup on a wooden table beside a small green plant. A hand enters slowly, rotates the same cup once, releases it, and leaves. Keep the cup, table, plant, hand anatomy, lighting, shadows, and background physically consistent. No cuts, no camera movement, no text.

overall_soundscape: Quiet room tone, one soft ceramic scrape synchronized with the visible rotation, and subtle sleeve movement. No speech and no music.

non_diegetic_music: N/A
"""


@dataclass(frozen=True, slots=True)
class MatrixCase:
    case_id: str
    launcher: str
    vram_gib: int
    weight_tier: str
    host_memory_limit_gib: int
    resolutions: tuple[str, ...]


MINIMUM_CASES = (
    MatrixCase(
        "w4a8_v8_ram12", "fl2va_w4a8_8gb", 8, "w4a8", 12,
        ("480p", "720p"),
    ),
    MatrixCase(
        "w4a8_v16_ram12", "fl2va_w4a8_16gb", 16, "w4a8", 12,
        ("480p", "720p", "1080p"),
    ),
    MatrixCase(
        "w4a8_v24_ram12", "fl2va_w4a8_24gb", 24, "w4a8", 12,
        ("480p", "720p", "1080p"),
    ),
    MatrixCase(
        "int8_v16_ram24", "fl2va_int8_16gb", 16, "int8", 24,
        ("480p", "720p", "1080p"),
    ),
    MatrixCase(
        "int8_v24_ram24", "fl2va_int8_24gb", 24, "int8", 24,
        ("480p", "720p", "1080p"),
    ),
    # Ref2VA shares the same DiT resource backend but adds real reference
    # conditioning.  One boundary row per backend proves that the extra input
    # path remains inside the same RAM/VRAM contract.
    MatrixCase(
        "w4a8_ref_v8_ram12", "ref2va_w4a8_8gb", 8, "w4a8", 12,
        ("720p",),
    ),
    MatrixCase(
        "w4a8_ref_v16_ram12", "ref2va_w4a8_16gb", 16, "w4a8", 12,
        ("1080p",),
    ),
    MatrixCase(
        "w4a8_ref_v24_ram12", "ref2va_w4a8_24gb", 24, "w4a8", 12,
        ("1080p",),
    ),
    MatrixCase(
        "int8_ref_v16_ram24", "ref2va_int8_16gb", 16, "int8", 24,
        ("1080p",),
    ),
    MatrixCase(
        "int8_ref_v24_ram24", "ref2va_int8_24gb", 24, "int8", 24,
        ("1080p",),
    ),
)

SATURATED_CASES = (
    MatrixCase(
        "w4a8_v8_ram22", "fl2va_w4a8_8gb", 8, "w4a8", 22,
        ("480p",),
    ),
    MatrixCase(
        "w4a8_v16_ram17", "fl2va_w4a8_16gb", 16, "w4a8", 17,
        ("480p",),
    ),
    MatrixCase(
        "int8_v16_ram32", "fl2va_int8_16gb", 16, "int8", 32,
        ("480p",),
    ),
    MatrixCase(
        "int8_v24_ram32", "fl2va_int8_24gb", 24, "int8", 32,
        ("480p",),
    ),
)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8090")
    parser.add_argument(
        "--output-dir", type=Path,
        default=root / "runtime/validation/release_resource_matrix_20260901",
    )
    parser.add_argument(
        "--ram-mode", choices=("minimum", "saturated", "both"),
        default="both",
    )
    parser.add_argument("--duration-seconds", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--acceleration", type=float, default=95.0)
    parser.add_argument("--poll-seconds", type=float, default=0.5)
    parser.add_argument("--timeout-seconds", type=float, default=900.0)
    parser.add_argument(
        "--reference-image", type=Path,
        default=root.parents[1] / "test/ref2va/01_luna_courier.png",
    )
    parser.add_argument("--only", nargs="*", default=())
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument(
        "--keep-engine", action="store_true",
        help="leave the final engine loaded instead of returning the service to idle",
    )
    return parser.parse_args()


async def json_request(
    session: aiohttp.ClientSession,
    method: str,
    url: str,
    **kwargs: Any,
) -> dict[str, Any]:
    async with session.request(method, url, **kwargs) as response:
        body = await response.text()
        if response.status >= 400:
            raise RuntimeError(
                f"{method} {url}: HTTP {response.status}: {body}"
            )
        return json.loads(body) if body else {}


async def wait_job(
    session: aiohttp.ClientSession,
    base_url: str,
    job_id: str,
    *,
    poll_seconds: float,
    timeout_seconds: float,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    while True:
        job = await json_request(
            session, "GET", f"{base_url}/api/v1/jobs/{job_id}"
        )
        if job.get("status") in TERMINAL:
            return job
        if time.monotonic() >= deadline:
            raise TimeoutError(f"job {job_id} exceeded {timeout_seconds}s")
        await asyncio.sleep(poll_seconds)


async def download_video(
    session: aiohttp.ClientSession,
    base_url: str,
    job: dict[str, Any],
    destination: Path,
) -> int:
    video_url = job.get("video_url")
    if not video_url:
        return 0
    async with session.get(f"{base_url}{video_url}") as response:
        response.raise_for_status()
        payload = await response.read()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(payload)
    return len(payload)


def selected_cases(mode: str) -> tuple[MatrixCase, ...]:
    if mode == "minimum":
        return MINIMUM_CASES
    if mode == "saturated":
        return SATURATED_CASES
    return MINIMUM_CASES + SATURATED_CASES


def row_ids(cases: tuple[MatrixCase, ...]) -> set[str]:
    return {
        f"{case.case_id}_{resolution}"
        for case in cases
        for resolution in case.resolutions
    }


def selected_row_ids(args: argparse.Namespace) -> set[str]:
    only = set(args.only)
    return {
        f"{case.case_id}_{resolution}"
        for case in selected_cases(args.ram_mode)
        if not only or case.case_id in only
        for resolution in case.resolutions
    }


def completed_row_ids(
    report: dict[str, Any], output_dir: Path | None = None
) -> set[str]:
    completed: set[str] = set()
    for row in report.get("rows", []):
        row_id = row.get("row_id")
        if (
            not row_id
            or row.get("status") != "passed"
            or row.get("job", {}).get("status") != "succeeded"
        ):
            continue
        if output_dir is not None:
            try:
                expected_bytes = int(row.get("download_bytes") or 0)
            except (TypeError, ValueError):
                continue
            media = output_dir / f"{row_id}.mp4"
            if (
                expected_bytes <= 0
                or not media.is_file()
                or media.stat().st_size != expected_bytes
            ):
                continue
        completed.add(row_id)
    return completed


def finalize_report(
    report: dict[str, Any], args: argparse.Namespace,
    *, output_dir: Path | None = None,
) -> tuple[bool, bool]:
    """Seal both this invocation and the complete resumable matrix.

    ``--only`` is deliberately a targeted repair/recheck surface.  It must not
    shrink the merged report's definition of the complete release matrix.
    ``selected_rows_passed`` therefore controls this invocation's exit status,
    while ``matrix_passed`` and the backwards-compatible top-level ``passed``
    describe all minimum and saturation rows.
    """

    expected = row_ids(selected_cases("both"))
    selected = selected_row_ids(args)
    passing = completed_row_ids(report, output_dir)
    selected_passed = selected <= passing
    matrix_passed = expected <= passing
    report.update({
        "schema_version": "h3_release_resource_matrix_v2",
        "finished_at": time.time(),
        "expected_rows": sorted(expected),
        "selected_rows": sorted(selected),
        "selected_rows_passed": selected_passed,
        "matrix_passed": matrix_passed,
        "passed": matrix_passed,
    })
    return selected_passed, matrix_passed


def save_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )


async def main_async(args: argparse.Namespace) -> int:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "release_resource_matrix.json"
    report: dict[str, Any] = {
        "schema_version": "h3_release_resource_matrix_v1",
        "base_url": args.base_url,
        "policy": {
            "duration_seconds": args.duration_seconds,
            "steps": args.steps,
            "acceleration": args.acceleration,
            "execution": "complete_end_to_end",
            "ram_mode": args.ram_mode,
        },
        "started_at": time.time(),
        "rows": [],
    }
    if report_path.is_file() and not args.rerun:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    completed = completed_row_ids(report, args.output_dir)
    only = set(args.only)
    timeout = aiohttp.ClientTimeout(total=None, connect=30)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        try:
            await json_request(session, "GET", f"{args.base_url}/healthz")
            for case in selected_cases(args.ram_mode):
                if only and case.case_id not in only:
                    continue
                switch_started = time.monotonic()
                switched = await json_request(
                    session,
                    "PUT",
                    f"{args.base_url}/api/v1/engine",
                    json={
                        "launcher": case.launcher,
                        "model_variant": "base",
                        "host_memory_limit_gib": case.host_memory_limit_gib,
                    },
                )
                switch_seconds = time.monotonic() - switch_started
                options = await json_request(
                    session, "GET", f"{args.base_url}/api/v1/options"
                )
                if options.get("current_launcher") != case.launcher:
                    raise RuntimeError(
                        f"launcher mismatch: expected {case.launcher}, "
                        f"got {options.get('current_launcher')}"
                    )
                enforcement = options.get("host_memory", {}).get(
                    "enforcement", {}
                )
                if (
                    not enforcement.get("enforced")
                    or float(enforcement.get("limit_gib") or 0)
                    != case.host_memory_limit_gib
                ):
                    raise RuntimeError(
                        f"host-memory limit is not enforced for {case.case_id}: "
                        f"{enforcement}"
                    )
                for resolution in case.resolutions:
                    row_id = f"{case.case_id}_{resolution}"
                    if row_id in completed and not args.rerun:
                        print(f"skip {row_id}: already passed", flush=True)
                        continue
                    print(f"run {row_id}", flush=True)
                    row: dict[str, Any] = {
                        "row_id": row_id,
                        "case": {
                            "launcher": case.launcher,
                            "vram_gib": case.vram_gib,
                            "weight_tier": case.weight_tier,
                            "host_memory_limit_gib": case.host_memory_limit_gib,
                            "resolution": resolution,
                        },
                        "engine_switch_seconds": round(switch_seconds, 3),
                        "engine_switch": switched,
                        "started_at": time.time(),
                    }
                    try:
                        before = await json_request(
                            session, "GET", f"{args.base_url}/api/v1/resources"
                        )
                        generation = {
                            "prompt": PROMPT,
                            "seed": str(20260901),
                            "resolution": resolution,
                            "aspect_ratio": "16:9",
                            "duration_seconds": str(args.duration_seconds),
                            "model_variant": "base",
                            "sampling_steps": str(args.steps),
                            "acceleration": str(args.acceleration),
                            "preview_mode": "off",
                        }
                        if case.launcher.startswith("ref2va_"):
                            form = aiohttp.FormData()
                            for key, value in generation.items():
                                form.add_field(key, value)
                            form.add_field(
                                "reference_image_1",
                                args.reference_image.read_bytes(),
                                filename=args.reference_image.name,
                                content_type="image/png",
                            )
                            submitted = await json_request(
                                session,
                                "POST",
                                f"{args.base_url}/api/v1/generations",
                                data=form,
                            )
                        else:
                            submitted = await json_request(
                                session,
                                "POST",
                                f"{args.base_url}/api/v1/generations",
                                json=generation,
                            )
                        job = await wait_job(
                            session,
                            args.base_url,
                            submitted["id"],
                            poll_seconds=args.poll_seconds,
                            timeout_seconds=args.timeout_seconds,
                        )
                        after = await json_request(
                            session, "GET", f"{args.base_url}/api/v1/resources"
                        )
                        row.update({
                            "status": (
                                "passed"
                                if job.get("status") == "succeeded"
                                else "failed"
                            ),
                            "job": job,
                            "resources_before": before,
                            "resources_after": after,
                        })
                        if row["status"] == "passed":
                            row["download_bytes"] = await download_video(
                                session,
                                args.base_url,
                                job,
                                args.output_dir / f"{row_id}.mp4",
                            )
                    except Exception as error:
                        row.update({
                            "status": "failed",
                            "error": f"{type(error).__name__}: {error}",
                        })
                    report["rows"] = [
                        item for item in report.get("rows", [])
                        if item.get("row_id") != row_id
                    ]
                    report["rows"].append(row)
                    save_report(report_path, report)
        finally:
            if not args.keep_engine:
                try:
                    await json_request(
                        session, "DELETE", f"{args.base_url}/api/v1/engine"
                    )
                except Exception:
                    pass

    selected_passed, _ = finalize_report(
        report, args, output_dir=args.output_dir
    )
    save_report(report_path, report)
    print(report_path, flush=True)
    return 0 if selected_passed else 1


def main() -> int:
    args = parse_args()
    if any(
        case.launcher.startswith("ref2va_")
        and (not args.only or case.case_id in set(args.only))
        for case in selected_cases(args.ram_mode)
    ) and not args.reference_image.is_file():
        raise SystemExit(f"missing reference image: {args.reference_image}")
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
