from __future__ import annotations

import asyncio
import os
import subprocess
import time
from pathlib import Path
from typing import Any

def _read_cpu_counters() -> tuple[int, int]:
    fields = Path("/proc/stat").read_text(encoding="utf-8").splitlines()[0].split()
    values = [int(value) for value in fields[1:]]
    idle = values[3] + (values[4] if len(values) > 4 else 0)
    return sum(values), idle


def _read_memory() -> dict[str, float]:
    values: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        key, raw = line.split(":", 1)
        values[key] = int(raw.strip().split()[0]) * 1024
    # /proc/meminfo describes the Linux/WSL host as a whole even when this
    # service lives inside a smaller cgroup.  The cgroup-scoped H3 value is
    # reported separately as `service_memory` by the API layer.
    total = values.get("MemTotal", 0)
    free = values.get("MemFree", 0)
    available = values.get("MemAvailable", values.get("MemFree", 0))
    used = max(0, total - available)
    # Keep the conventional Linux pressure-oriented `used_gib` for API
    # compatibility, but also expose physical occupancy.  A service cgroup's
    # memory.current includes charged file cache, whereas MemAvailable treats
    # reclaimable cache as available.  Comparing those two values directly can
    # otherwise make a child cgroup appear larger than the whole host.
    occupied = max(0, total - free)
    reclaimable = max(0, available - free)
    return {
        "used_gib": round(used / 2**30, 2),
        "occupied_gib": round(occupied / 2**30, 2),
        "total_gib": round(total / 2**30, 2),
        "free_gib": round(free / 2**30, 2),
        "available_gib": round(available / 2**30, 2),
        "reclaimable_gib": round(reclaimable / 2**30, 2),
        "percent": round(used * 100 / total, 1) if total else 0.0,
        "occupied_percent": round(occupied * 100 / total, 1) if total else 0.0,
        "physical_total_gib": round(total / 2**30, 2),
        "scope": "linux_host",
    }


def _read_process_rss() -> float:
    for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
        if line.startswith("VmRSS:"):
            return round(int(line.split()[1]) / 2**20, 2)
    return 0.0


def _read_gpu() -> dict[str, Any] | None:
    command = [
        "nvidia-smi",
        "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
        "--format=csv,noheader,nounits",
        "--id=0",
    ]
    try:
        result = subprocess.run(
            command, check=True, capture_output=True, text=True, timeout=2,
        )
        fields = [field.strip() for field in result.stdout.splitlines()[0].split(",")]
        name, utilization, used, total, temperature, power = fields[:6]
        used_mib, total_mib = float(used), float(total)
        return {
            "name": name,
            "utilization_percent": round(float(utilization), 1),
            "memory_used_gib": round(used_mib / 1024, 2),
            "memory_total_gib": round(total_mib / 1024, 2),
            "memory_percent": round(used_mib * 100 / total_mib, 1) if total_mib else 0.0,
            "temperature_c": round(float(temperature), 1),
            "power_w": round(float(power), 1),
        }
    except (FileNotFoundError, subprocess.SubprocessError, ValueError, IndexError):
        return None


class ResourceMonitor:
    """Low-overhead host/GPU telemetry with a short shared cache."""

    def __init__(self, *, cache_seconds: float = 1.0) -> None:
        self.cache_seconds = max(0.2, float(cache_seconds))
        self._cached_at = 0.0
        self._cached: dict[str, Any] | None = None
        self._cpu = _read_cpu_counters()
        self._lock = asyncio.Lock()

    def _sample(self) -> dict[str, Any]:
        total, idle = _read_cpu_counters()
        prior_total, prior_idle = self._cpu
        total_delta = max(0, total - prior_total)
        idle_delta = max(0, idle - prior_idle)
        utilization = (
            (total_delta - idle_delta) * 100 / total_delta if total_delta else 0.0
        )
        self._cpu = (total, idle)
        return {
            "timestamp": time.time(),
            "cpu": {
                "utilization_percent": round(max(0.0, min(100.0, utilization)), 1),
                "logical_cores": os.cpu_count() or 1,
                "load_1m": round(os.getloadavg()[0], 2),
            },
            "memory": _read_memory(),
            "process": {"rss_gib": _read_process_rss()},
            "gpu": _read_gpu(),
        }

    async def snapshot(self) -> dict[str, Any]:
        async with self._lock:
            now = time.monotonic()
            if self._cached is not None and now - self._cached_at < self.cache_seconds:
                return self._cached
            self._cached = await asyncio.to_thread(self._sample)
            self._cached_at = time.monotonic()
            return self._cached
