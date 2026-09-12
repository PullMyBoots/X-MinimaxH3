#!/usr/bin/env python3
"""Run a resumable, low-cost real-GPU release matrix through the public API."""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

import aiohttp

from h3serve.deployment_profiles import LAUNCHER_DEFINITIONS


TERMINAL = {"succeeded", "failed", "cancelled", "checkpointed"}
PROMPT = """integrated_multimodal_description: [Shot 1]
A single continuous locked-off documentary shot of a red ceramic cup on a wooden table beside a small green plant. Soft daylight remains stable. A hand enters slowly, rotates the same cup once, releases it, and leaves. The cup, table, plant, hand anatomy, shadows, and background remain physically consistent. No cuts, no camera movement, no text.

overall_soundscape: quiet room tone, one soft ceramic scrape synchronized with the visible rotation, and subtle sleeve movement. No speech and no music.

non_diegetic_music: N/A
"""
REFERENCE_PROMPT = """integrated_multimodal_description: [Shot 1]
Use <Picture 1> only as the identity and appearance reference. In one continuous locked-off documentary shot, the same subject turns their head gently toward a red ceramic cup on a wooden table and then remains still. Keep facial identity, clothing, anatomy, cup geometry, lighting, shadows, and background stable. No cuts, no camera movement, no text.

overall_soundscape: quiet room tone and subtle clothing movement. No speech and no music.

non_diegetic_music: N/A
"""


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8091")
    parser.add_argument(
        "--launchers",
        nargs="+",
        choices=tuple(LAUNCHER_DEFINITIONS),
        default=tuple(LAUNCHER_DEFINITIONS),
    )
    parser.add_argument(
        "--reference-image",
        type=Path,
        default=root.parent / "pud/compare/ref2va/girl.png",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "runtime/validation/tiered_backend_20260828",
    )
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    parser.add_argument("--timeout-seconds", type=float, default=1800.0)
    parser.add_argument(
        "--complete-resolution",
        choices=("360p", "480p", "720p", "1080p"),
        default="360p",
        help="geometry used by the complete generation stages",
    )
    parser.add_argument(
        "--complete-duration-seconds",
        type=float,
        default=1.0,
        help="requested duration used by the complete generation stages",
    )
    parser.add_argument("--base-steps", type=int, default=5)
    parser.add_argument("--lora-steps", type=int, default=4)
    parser.add_argument("--acceleration", type=float, default=95.0)
    parser.add_argument("--skip-lora", action="store_true")
    parser.add_argument("--skip-second-sampling", action="store_true")
    parser.add_argument(
        "--boundary-checkpoint-only",
        action="store_true",
        help="run one accelerated Actual step at each tier's maximum geometry",
    )
    parser.add_argument(
        "--second-sampling-target",
        choices=("720p", "1080p", "2k"),
        default="720p",
    )
    return parser.parse_args()


async def json_request(
    session: aiohttp.ClientSession,
    method: str,
    url: str,
    **kwargs,
) -> dict:
    async with session.request(method, url, **kwargs) as response:
        text = await response.text()
        if response.status >= 400:
            raise RuntimeError(f"{method} {url}: HTTP {response.status}: {text}")
        return json.loads(text)


async def wait_job(
    session: aiohttp.ClientSession,
    base_url: str,
    job_id: str,
    *,
    poll_seconds: float,
    timeout_seconds: float,
) -> dict:
    started = time.monotonic()
    while True:
        job = await json_request(
            session, "GET", f"{base_url}/api/v1/jobs/{job_id}"
        )
        if job["status"] in TERMINAL:
            return job
        if time.monotonic() - started > timeout_seconds:
            raise TimeoutError(f"job {job_id} exceeded {timeout_seconds}s")
        await asyncio.sleep(poll_seconds)


async def submit_generation(
    session: aiohttp.ClientSession,
    base_url: str,
    launcher: str,
    variant: str,
    reference_image: Path,
    *,
    resolution: str,
    duration_seconds: float,
    base_steps: int,
    lora_steps: int,
    acceleration: float,
) -> dict:
    definition = LAUNCHER_DEFINITIONS[launcher]
    fields = {
        "prompt": (
            REFERENCE_PROMPT
            if definition.service_family == "reference"
            else PROMPT
        ),
        "seed": "20260828",
        "resolution": resolution,
        "aspect_ratio": "16:9",
        "duration_seconds": str(duration_seconds),
        "model_variant": variant,
        "sampling_steps": str(lora_steps if variant == "lora" else base_steps),
        "acceleration": str(acceleration),
        "preview_mode": "off",
    }
    if definition.service_family == "reference":
        form = aiohttp.FormData()
        for key, value in fields.items():
            form.add_field(key, value)
        form.add_field(
            "reference_image_1",
            reference_image.read_bytes(),
            filename=reference_image.name,
            content_type="image/png",
        )
        return await json_request(
            session,
            "POST",
            f"{base_url}/api/v1/generations",
            data=form,
        )
    return await json_request(
        session,
        "POST",
        f"{base_url}/api/v1/generations",
        json=fields,
    )


async def submit_boundary_checkpoint(
    session: aiohttp.ClientSession,
    base_url: str,
    launcher: str,
    reference_image: Path,
) -> dict:
    definition = LAUNCHER_DEFINITIONS[launcher]
    resolution = definition.backend.first_generation_levels[-1]
    fields = {
        "prompt": (
            REFERENCE_PROMPT
            if definition.service_family == "reference"
            else PROMPT
        ),
        "seed": "20260829",
        "resolution": resolution,
        "aspect_ratio": "16:9",
        "duration_seconds": "15",
        "model_variant": "base",
        "sampling_steps": "5",
        "acceleration": "95",
        "execution_mode": "checkpoint",
        "checkpoint_step": "1",
        "checkpoint_retain": "true",
        "checkpoint_preview": "false",
        "preview_mode": "off",
    }
    if definition.service_family == "reference":
        form = aiohttp.FormData()
        for key, value in fields.items():
            form.add_field(key, value)
        form.add_field(
            "reference_image_1",
            reference_image.read_bytes(),
            filename=reference_image.name,
            content_type="image/png",
        )
        return await json_request(
            session, "POST", f"{base_url}/api/v1/generations", data=form
        )
    return await json_request(
        session, "POST", f"{base_url}/api/v1/generations", json=fields
    )


async def save_video(
    session: aiohttp.ClientSession,
    base_url: str,
    job: dict,
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


async def main_async(args: argparse.Namespace) -> int:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "tiered_backend_gpu_matrix.json"
    report = {
        "schema_version": "h3_tiered_backend_gpu_matrix_v1",
        "base_url": args.base_url,
        "test_policy": {
            "base_steps": args.base_steps,
            "lora_steps": args.lora_steps,
            "acceleration": args.acceleration,
            "complete_geometry": (
                f"{args.complete_resolution}_"
                f"{args.complete_duration_seconds:g}s"
            ),
            "second_sampling_steps": 1,
            "second_sampling_target": args.second_sampling_target,
        },
        "started_at": time.time(),
        "cases": [],
    }
    timeout = aiohttp.ClientTimeout(total=None, connect=30)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        for launcher in args.launchers:
            definition = LAUNCHER_DEFINITIONS[launcher]
            case = {"launcher": launcher, "stages": []}
            report["cases"].append(case)
            try:
                switched = await json_request(
                    session,
                    "PUT",
                    f"{args.base_url}/api/v1/engine",
                    json={"launcher": launcher, "model_variant": "base"},
                )
                case["stages"].append({"stage": "preload", "status": "passed", "result": switched})
                if args.boundary_checkpoint_only:
                    submitted = await submit_boundary_checkpoint(
                        session,
                        args.base_url,
                        launcher,
                        args.reference_image,
                    )
                    job = await wait_job(
                        session,
                        args.base_url,
                        submitted["id"],
                        poll_seconds=args.poll_seconds,
                        timeout_seconds=args.timeout_seconds,
                    )
                    case["stages"].append({
                        "stage": "boundary_actual_step_1",
                        "status": (
                            "passed"
                            if job["status"] == "checkpointed"
                            else "failed"
                        ),
                        "job": job,
                    })
                    report_path.write_text(
                        json.dumps(report, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                    continue
                source_job = None
                variants = ("base",) if args.skip_lora else ("base", "lora")
                for variant in variants:
                    submitted = await submit_generation(
                        session,
                        args.base_url,
                        launcher,
                        variant,
                        args.reference_image,
                        resolution=args.complete_resolution,
                        duration_seconds=args.complete_duration_seconds,
                        base_steps=args.base_steps,
                        lora_steps=args.lora_steps,
                        acceleration=args.acceleration,
                    )
                    job = await wait_job(
                        session,
                        args.base_url,
                        submitted["id"],
                        poll_seconds=args.poll_seconds,
                        timeout_seconds=args.timeout_seconds,
                    )
                    stage = {
                        "stage": f"complete_{variant}",
                        "status": "passed" if job["status"] == "succeeded" else "failed",
                        "job": job,
                    }
                    if job["status"] == "succeeded":
                        stage["download_bytes"] = await save_video(
                            session,
                            args.base_url,
                            job,
                            args.output_dir
                            / (
                                f"{launcher}_{variant}_"
                                f"{args.complete_resolution}_"
                                f"{args.complete_duration_seconds:g}s.mp4"
                            ),
                        )
                        if variant == "base":
                            source_job = job
                    case["stages"].append(stage)
                    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

                if (
                    source_job is not None
                    and definition.backend.second_sampling_levels
                    and not args.skip_second_sampling
                ):
                    submitted = await json_request(
                        session,
                        "POST",
                        f"{args.base_url}/api/v1/jobs/{source_job['id']}/second-sampling",
                        json={
                            "resolution": args.second_sampling_target,
                            "steps": 1,
                            "acceleration": 95,
                            "strength": "preserve",
                            "model_variant": "base",
                        },
                    )
                    job = await wait_job(
                        session,
                        args.base_url,
                        submitted["id"],
                        poll_seconds=args.poll_seconds,
                        timeout_seconds=args.timeout_seconds,
                    )
                    stage = {
                        "stage": (
                            "second_sampling_base_"
                            f"{args.second_sampling_target}"
                        ),
                        "status": "passed" if job["status"] == "succeeded" else "failed",
                        "job": job,
                    }
                    if job["status"] == "succeeded":
                        stage["download_bytes"] = await save_video(
                            session,
                            args.base_url,
                            job,
                            args.output_dir
                            / (
                                f"{launcher}_base_second_"
                                f"{args.second_sampling_target}.mp4"
                            ),
                        )
                    case["stages"].append(stage)
            except Exception as error:
                case["stages"].append({
                    "stage": "orchestration",
                    "status": "failed",
                    "error": f"{type(error).__name__}: {error}",
                })
            report_path.write_text(
                json.dumps(report, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    report["finished_at"] = time.time()
    report["passed"] = all(
        stage["status"] == "passed"
        for case in report["cases"]
        for stage in case["stages"]
    )
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(report_path)
    return 0 if report["passed"] else 1


def main() -> int:
    args = parse_args()
    if not args.reference_image.is_file():
        raise SystemExit(f"missing reference image: {args.reference_image}")
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
