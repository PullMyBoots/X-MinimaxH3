#!/usr/bin/env python3
"""Measure W4A8 latency/host-RAM knees under 8/16/24-GiB VRAM caps.

This is a research harness, not a product launcher.  Every case keeps the
checkpoint, prompt, scheduler and V24 acceleration controls fixed.  It changes
only three numerically exact residency mechanics: the CUDA allocator ceiling,
the number of transformer blocks kept on the GPU, and the amount of complete
offloaded blocks copied into pinned host slabs.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import statistics
import subprocess
import sys
import threading
import time
import urllib.request
import urllib.error


GIB = 1024**3
TERMINAL = {"succeeded", "failed", "cancelled", "checkpointed"}
PROMPT = """integrated_multimodal_description: [Shot 1]
A single continuous locked-off documentary shot of a red ceramic cup on a wooden table beside a small green plant. Soft daylight remains stable. A hand enters slowly, rotates the same cup once, releases it, and leaves. The cup, table, plant, hand anatomy, shadows, and background remain physically consistent. No cuts, no camera movement, no text.

overall_soundscape: quiet room tone, one soft ceramic scrape synchronized with the visible rotation, and subtle sleeve movement. No speech and no music.

non_diegetic_music: N/A
"""


def default_cases() -> list[dict[str, float | int | str]]:
    # Resident ladders locate the best VRAM use.  Pin ladders then trace the
    # host-memory curve at the largest conservative resident prefix.  Duplicate
    # zero-pin endpoints are intentionally omitted.
    return [
        {"id": "v8_r0_p0", "vram_gib": 8, "resident_blocks": 0, "pin_gib": 0.0, "memory_max_gib": 15},
        {"id": "v8_r0_p3p5", "vram_gib": 8, "resident_blocks": 0, "pin_gib": 3.5, "memory_max_gib": 18},
        {"id": "v8_r0_p7", "vram_gib": 8, "resident_blocks": 0, "pin_gib": 7.0, "memory_max_gib": 22},
        {"id": "v8_r0_p11", "vram_gib": 8, "resident_blocks": 0, "pin_gib": 11.0, "memory_max_gib": 27},
        {"id": "v16_r0_p0", "vram_gib": 16, "resident_blocks": 0, "pin_gib": 0.0, "memory_max_gib": 15},
        {"id": "v16_r18_p0", "vram_gib": 16, "resident_blocks": 18, "pin_gib": 0.0, "memory_max_gib": 15},
        {"id": "v16_r36_p0", "vram_gib": 16, "resident_blocks": 36, "pin_gib": 0.0, "memory_max_gib": 15},
        {"id": "v16_r36_p1p5", "vram_gib": 16, "resident_blocks": 36, "pin_gib": 1.5, "memory_max_gib": 17},
        {"id": "v16_r36_p3p2", "vram_gib": 16, "resident_blocks": 36, "pin_gib": 3.2, "memory_max_gib": 19},
        {"id": "v24_r0_p0", "vram_gib": 24, "resident_blocks": 0, "pin_gib": 0.0, "memory_max_gib": 15},
        {"id": "v24_r24_p0", "vram_gib": 24, "resident_blocks": 24, "pin_gib": 0.0, "memory_max_gib": 15},
        {"id": "v24_r49_p0", "vram_gib": 24, "resident_blocks": 49, "pin_gib": 0.0, "memory_max_gib": 15},
        {"id": "v24_r49_p0p3", "vram_gib": 24, "resident_blocks": 49, "pin_gib": 0.3, "memory_max_gib": 15},
        # Knee refinement / repeat gates. These are kept in the same harness
        # so an interrupted study can resume without a second ad-hoc script.
        {"id": "v8_r0_p0_repeat", "vram_gib": 8, "resident_blocks": 0, "pin_gib": 0.0, "memory_max_gib": 15},
        {"id": "v8_r0_p11_repeat", "vram_gib": 8, "resident_blocks": 0, "pin_gib": 11.0, "memory_max_gib": 27},
        {"id": "v16_r12_p0", "vram_gib": 16, "resident_blocks": 12, "pin_gib": 0.0, "memory_max_gib": 15},
        {"id": "v16_r18_p0_repeat", "vram_gib": 16, "resident_blocks": 18, "pin_gib": 0.0, "memory_max_gib": 15},
        {"id": "v16_r18_p3p5", "vram_gib": 16, "resident_blocks": 18, "pin_gib": 3.5, "memory_max_gib": 18},
        {"id": "v16_r18_p7", "vram_gib": 16, "resident_blocks": 18, "pin_gib": 7.0, "memory_max_gib": 22},
        {"id": "v16_r24_p0", "vram_gib": 16, "resident_blocks": 24, "pin_gib": 0.0, "memory_max_gib": 15},
        {"id": "v24_r49_p0_repeat", "vram_gib": 24, "resident_blocks": 49, "pin_gib": 0.0, "memory_max_gib": 15},
        {"id": "v16_r24_p6", "vram_gib": 16, "resident_blocks": 24, "pin_gib": 6.0, "memory_max_gib": 21},
        {"id": "v16_r27_p0", "vram_gib": 16, "resident_blocks": 27, "pin_gib": 0.0, "memory_max_gib": 15},
        {"id": "v16_r30_p0", "vram_gib": 16, "resident_blocks": 30, "pin_gib": 0.0, "memory_max_gib": 15},
        {"id": "v16_r33_p0", "vram_gib": 16, "resident_blocks": 33, "pin_gib": 0.0, "memory_max_gib": 15},
    ]


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "runtime/validation/w4a8_resource_knee_20260830",
    )
    parser.add_argument("--port", type=int, default=8094)
    parser.add_argument("--only", nargs="*", default=())
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--ready-timeout", type=float, default=180.0)
    parser.add_argument("--job-timeout", type=float, default=600.0)
    parser.add_argument(
        "--nsys",
        action="store_true",
        help=(
            "wrap each selected server in Nsight Systems so CUDA memcpy and "
            "kernel timelines can be audited; intended only for targeted cases"
        ),
    )
    parser.add_argument(
        "--torch-profile",
        action="store_true",
        help="capture one full request with the gated in-process Kineto profiler",
    )
    return parser.parse_args()


def request_json(url: str, *, payload: dict | None = None) -> dict:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={} if body is None else {"Content-Type": "application/json"},
        method="GET" if body is None else "POST",
    )
    # Kineto can briefly hold the worker GIL while reducing a large event set.
    # The service remains healthy; allow the diagnostic client to outlive that
    # reduction instead of falsely marking the generation as timed out.
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        # Preserve the service's contract explanation in benchmark artifacts;
        # an HTTP 400 must never be mistaken for a resource-capacity failure.
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {error.code}: {detail}") from error


def read_key_values(path: Path) -> dict[str, int]:
    result: dict[str, int] = {}
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            key, value = line.split()
            result[key] = int(value)
    except (OSError, ValueError):
        pass
    return result


def evict_clean_checkpoint_pages(paths: list[Path]) -> float:
    """Best-effort cold-cache normalization for isolated research cases.

    POSIX_FADV_DONTNEED only discards clean file-cache pages.  It neither
    changes nor removes checkpoints, and avoids globally dropping unrelated
    Linux caches.  Return the logical file GiB covered by the hint so every
    benchmark artifact records the normalization scope.
    """
    advised_bytes = 0
    for path in paths:
        if not path.is_file():
            continue
        try:
            with path.open("rb", buffering=0) as stream:
                size = os.fstat(stream.fileno()).st_size
                os.posix_fadvise(
                    stream.fileno(), 0, 0, os.POSIX_FADV_DONTNEED
                )
                advised_bytes += size
        except (AttributeError, OSError):
            continue
    return advised_bytes / GIB


def gpu_sample() -> tuple[float, float, float] | None:
    try:
        value = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,power.draw,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=3,
        ).strip().splitlines()[0]
        memory_mib, power_w, utilization = (
            float(item.strip()) for item in value.split(",")
        )
        return memory_mib, power_w, utilization
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


class Sampler:
    def __init__(self, cgroup: Path) -> None:
        self.cgroup = cgroup
        self.samples: list[dict[str, float]] = []
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self.stop.is_set():
            sample = gpu_sample()
            if sample is not None:
                memory_mib, power_w, utilization = sample
                try:
                    host_bytes = int((self.cgroup / "memory.current").read_text())
                except (OSError, ValueError):
                    host_bytes = 0
                self.samples.append({
                    "time": time.time(),
                    "gpu_memory_mib": memory_mib,
                    "gpu_power_w": power_w,
                    "gpu_utilization_percent": utilization,
                    "host_memory_gib": host_bytes / GIB,
                })
            self.stop.wait(0.5)

    def start(self) -> None:
        self.thread.start()

    def finish(self) -> dict[str, float | int]:
        self.stop.set()
        self.thread.join(timeout=5)
        active = [item for item in self.samples if item["gpu_utilization_percent"] > 0]
        power = [item["gpu_power_w"] for item in active]
        utilization = [item["gpu_utilization_percent"] for item in active]
        return {
            "sample_count": len(self.samples),
            "active_sample_count": len(active),
            "gpu_memory_peak_mib": max(
                (item["gpu_memory_mib"] for item in self.samples), default=0.0
            ),
            "gpu_power_mean_active_w": statistics.fmean(power) if power else 0.0,
            "gpu_power_peak_w": max(power, default=0.0),
            "gpu_utilization_mean_active_percent": (
                statistics.fmean(utilization) if utilization else 0.0
            ),
            "host_memory_sampled_peak_gib": max(
                (item["host_memory_gib"] for item in self.samples), default=0.0
            ),
        }


def wait_ready(base_url: str, process: subprocess.Popen, timeout: float) -> dict:
    deadline = time.monotonic() + timeout
    last_error = "service did not answer"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"server exited with code {process.returncode}")
        try:
            health = request_json(f"{base_url}/healthz")
            state = health.get("warm_state", {})
            if state.get("status") == "ready":
                return health
            if state.get("status") == "failed":
                raise RuntimeError(f"engine preload failed: {state}")
            last_error = str(state)
        except (OSError, urllib.error.URLError, json.JSONDecodeError) as error:
            last_error = str(error)
        time.sleep(0.5)
    raise TimeoutError(f"service readiness timeout: {last_error}")


def wait_job(base_url: str, job_id: str, timeout: float) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = request_json(f"{base_url}/api/v1/jobs/{job_id}")
        if job.get("status") in TERMINAL:
            return job
        time.sleep(0.5)
    raise TimeoutError(f"job {job_id} exceeded {timeout}s")


def terminate(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    process.send_signal(signal.SIGINT)
    try:
        process.wait(timeout=15)
        return
    except subprocess.TimeoutExpired:
        process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


def run_case(args: argparse.Namespace, case: dict) -> dict:
    project_root = Path(__file__).resolve().parents[1]
    memory_profile = str(case.get("memory_profile", "w4a8_16gb"))
    launcher = str(case.get("launcher", "fl2va_w4a8_8gb"))
    research_int8_curve = bool(case.get("research_int8_curve", False))
    case_dir = args.output_dir / str(case["id"])
    if case_dir.exists():
        shutil.rmtree(case_dir)
    case_dir.mkdir(parents=True)
    data_dir = case_dir / "data"
    cgroup = Path("/sys/fs/cgroup") / f"h3-w4-knee-{os.getpid()}-{case['id']}"
    cgroup.mkdir()
    # Nsight's injected collector and trace buffers are charged to the same
    # cgroup but are not part of the H3 execution graph. Give the profiler
    # explicit headroom so it cannot manufacture memory pressure in the case
    # being measured. The engine's pin/residency controls remain unchanged.
    profiler_headroom_gib = 8.0 if (args.nsys or args.torch_profile) else 0.0
    (cgroup / "memory.max").write_text(
        str(int((float(case["memory_max_gib"]) + profiler_headroom_gib) * GIB)),
        encoding="utf-8",
    )
    (cgroup / "memory.swap.max").write_text("0", encoding="utf-8")
    (cgroup / "memory.oom.group").write_text("1", encoding="utf-8")

    cold_cache_advised_gib = 0.0
    if case.get("cold_cache", False):
        model_root = Path("/root/h3-model-store")
        local_qwen_root = Path("/root/.cache/h3serve/checkpoints")
        cold_paths = [
            model_root / (
                "diffusion_models/"
                "minimax_h3_fl2va_pruned_int8_convrot.safetensors"
            ),
            model_root / (
                "text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
            ),
            local_qwen_root / "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
            model_root / "vae/minimax_h3_video_vae_fp16.safetensors",
            model_root / "vae/minimax_h3_audio_vae_fp32.safetensors",
        ]
        cold_paths.extend(
            sorted(local_qwen_root.glob("*.layers-v1-*/*.safetensors"))
        )
        cold_cache_advised_gib = evict_clean_checkpoint_pages(cold_paths)

    env = os.environ.copy()
    env.update({
        "H3_SERVE_MEMORY_PROFILE": memory_profile,
        # Avoid comparing a symlink spelling against the resolved model root
        # in the LoRA containment guard when this research tree reuses the
        # canonical release weight store.
        "H3_SERVE_MODEL_DIR": str(
            Path("/root/h3-model-store")
            if Path("/root/h3-model-store/.ready").is_file()
            else (project_root / "models").resolve()
        ),
        "H3_SERVE_MINIMAX_SOURCE": "/root/h3-new-serve-runtime/vendor/MiniMax-H3",
        "H3_SERVE_LIGHTX_SOURCE": "/root/h3-new-serve-runtime/vendor/LightX2V",
        "H3_NATIVE_SPARGE_BUILD_DIR": (
            "/root/h3-new-serve-runtime/worktree/runtime/extensions/"
            "sparge-sm89-py310-torch213-cu133"
        ),
        "H3_NATIVE_ENABLE_SPARSE": "1",
        "H3_NATIVE_RESEARCH_W4_VRAM_GIB": (
            "" if research_int8_curve else str(case["vram_gib"])
        ),
        "H3_NATIVE_RESEARCH_INT8_HOST_CURVE": (
            "1" if research_int8_curve else "0"
        ),
        "H3_NATIVE_RESEARCH_RESIDENT_BLOCKS": str(case["resident_blocks"]),
        "H3_NATIVE_RESEARCH_PIN_TRANSFORMER_GIB": str(case["pin_gib"]),
        "H3_NATIVE_RESEARCH_TRACE_PRELOAD_ERROR": "1" if args.nsys else "0",
        "H3_NATIVE_PROFILE_REGIONS": "1" if args.torch_profile else "0",
        "H3_NATIVE_RESEARCH_TORCH_PROFILE_PATH": (
            str(case_dir / "torch_trace.json") if args.torch_profile else ""
        ),
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    })
    server_command = [
        sys.executable,
        "server.py",
        "--host", "127.0.0.1",
        "--port", str(args.port),
        "--data-dir", str(data_dir),
        "--memory-profile", memory_profile,
        "--engine", launcher,
    ]
    if args.nsys:
        command = [
            "nsys",
            "profile",
            "--trace=cuda,nvtx,osrt",
            "--sample=none",
            "--cpuctxsw=none",
            "--force-overwrite=true",
            f"--output={case_dir / 'cuda_timeline'}",
            *server_command,
        ]
    else:
        command = server_command
    log_path = case_dir / "server.log"
    log = log_path.open("w", encoding="utf-8")
    process = subprocess.Popen(
        command,
        cwd=project_root,
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
        text=True,
    )
    (cgroup / "cgroup.procs").write_text(str(process.pid), encoding="utf-8")
    base_url = f"http://127.0.0.1:{args.port}"
    started = time.time()
    sampler = Sampler(cgroup)
    result: dict = {
        "case": case,
        "started_at": started,
        "profiler_cgroup_headroom_gib": profiler_headroom_gib,
        "cold_cache_advised_gib": cold_cache_advised_gib,
    }
    try:
        health = wait_ready(base_url, process, args.ready_timeout)
        result["warm_state"] = health.get("warm_state")
        execution_mode = str(case.get("execution_mode", "checkpoint"))
        payload = {
            "prompt": PROMPT,
            "seed": "20260829",
            "resolution": str(case.get("resolution", "720p")),
            "aspect_ratio": str(case.get("aspect_ratio", "16:9")),
            "duration_seconds": str(case.get("duration_seconds", 15)),
            "model_variant": "base",
            # Five completed solver steps capture two recurring Actual DiT
            # evaluations without paying the final Video/Audio-VAE decode.
            "sampling_steps": str(case.get("sampling_steps", 6)),
            "acceleration": str(case.get("acceleration", 95)),
            "execution_mode": execution_mode,
            "preview_mode": "off",
        }
        if execution_mode == "checkpoint":
            payload.update({
                "checkpoint_step": str(case.get("checkpoint_step", 5)),
                "checkpoint_retain": "true",
                "checkpoint_preview": "false",
            })
        sampler.start()
        submitted = request_json(
            f"{base_url}/api/v1/generations", payload=payload
        )
        result["job"] = wait_job(base_url, submitted["id"], args.job_timeout)
        result["status"] = result["job"].get("status")
    except Exception as error:
        result["status"] = "failed"
        result["error"] = f"{type(error).__name__}: {error}"
    finally:
        if sampler.thread.is_alive():
            result["samples"] = sampler.finish()
        result["cgroup_memory_peak_gib"] = (
            int((cgroup / "memory.peak").read_text()) / GIB
        )
        result["cgroup_memory_events"] = read_key_values(
            cgroup / "memory.events"
        )
        result["elapsed_wall_seconds"] = time.time() - started
        terminate(process)
        log.close()
        # Retained sampler checkpoints are multi-GiB scratch artifacts. Keep
        # the auditable job JSON/log and remove only this harness's latent data.
        shutil.rmtree(data_dir / "checkpoints", ignore_errors=True)
        try:
            cgroup.rmdir()
        except OSError:
            pass
    return result


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "resource_knee_raw.json"
    report = {
        "schema_version": "h3_w4a8_resource_knee_v1",
        "numerical_contract": (
            "same W4A8 weights, prompt, 6-step solver, acceleration 95; "
            "only exact residency/offload mechanics vary"
        ),
        "cases": [],
    }
    if report_path.is_file() and not args.rerun:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    completed = {item["case"]["id"] for item in report.get("cases", [])}
    selected = set(args.only)
    for case in default_cases():
        if selected and case["id"] not in selected:
            continue
        if case["id"] in completed and not args.rerun:
            print(f"skip {case['id']} (already recorded)", flush=True)
            continue
        print(f"run {case['id']}: {case}", flush=True)
        value = run_case(args, case)
        report.setdefault("cases", []).append(value)
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(
            f"done {case['id']}: status={value.get('status')} "
            f"peak={value.get('cgroup_memory_peak_gib', 0):.3f}GiB",
            flush=True,
        )
        if value.get("status") == "failed":
            print(value.get("error", "job failed"), flush=True)
    return 0 if all(
        item.get("status") == "checkpointed" for item in report.get("cases", [])
    ) else 1


if __name__ == "__main__":
    raise SystemExit(main())
