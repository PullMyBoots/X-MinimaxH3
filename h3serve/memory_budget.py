"""Service-process host-memory enforcement for the unified console."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path


GIB = 1024**3


def _process_rss_gib(pid: int) -> float:
    """Best-effort RSS fallback when a private cgroup is not attached."""

    try:
        for line in Path(f"/proc/{int(pid)}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024 / GIB
    except (FileNotFoundError, OSError, ValueError):
        pass
    return 0.0


def _process_pss_gib(pid: int) -> float:
    """Return proportional resident RAM, including mapped model file pages."""

    try:
        for line in Path(f"/proc/{int(pid)}/smaps_rollup").read_text().splitlines():
            if line.startswith("Pss:"):
                return int(line.split()[1]) * 1024 / GIB
    except (FileNotFoundError, OSError, ValueError):
        pass
    return _process_rss_gib(pid)


def _service_resident_gib(directory: Path, fallback_pid: int) -> tuple[float, int]:
    """Sum PSS for every service process currently inside one cgroup."""

    try:
        members = {
            int(value)
            for value in (directory / "cgroup.procs").read_text().split()
            if value.isdigit()
        }
    except (FileNotFoundError, OSError, ValueError):
        members = {int(fallback_pid)}
    values = [_process_pss_gib(pid) for pid in members]
    return sum(values), len(values)


def _usage_document(
    *,
    used_gib: float,
    limit_gib: float | None,
    peak_gib: float | None,
    enforced: bool,
    scope: str,
    anonymous_gib: float | None = None,
    file_cache_gib: float | None = None,
    oom_count: int | None = None,
    oom_kill_count: int | None = None,
    resident_gib: float | None = None,
    resident_process_count: int | None = None,
) -> dict[str, object]:
    percent = (
        max(0.0, min(100.0, used_gib * 100.0 / limit_gib))
        if limit_gib and limit_gib > 0
        else 0.0
    )
    document: dict[str, object] = {
        "used_gib": round(max(0.0, used_gib), 2),
        "limit_gib": None if limit_gib is None else round(limit_gib, 2),
        "peak_gib": None if peak_gib is None else round(max(0.0, peak_gib), 2),
        "percent": round(percent, 1),
        "enforced": bool(enforced),
        "scope": scope,
    }
    if anonymous_gib is not None:
        document["anonymous_gib"] = round(max(0.0, anonymous_gib), 2)
    if file_cache_gib is not None:
        document["file_cache_gib"] = round(max(0.0, file_cache_gib), 2)
    if oom_count is not None:
        document["oom_count"] = max(0, int(oom_count))
    if oom_kill_count is not None:
        document["oom_kill_count"] = max(0, int(oom_kill_count))
    if resident_gib is not None:
        document["resident_gib"] = round(max(0.0, resident_gib), 2)
        document["resident_metric"] = "pss"
    if resident_process_count is not None:
        document["resident_process_count"] = max(0, int(resident_process_count))
    return document


@dataclass(frozen=True, slots=True)
class MemoryBudgetState:
    supported: bool
    enforced: bool
    limit_gib: float | None
    mechanism: str
    detail: str

    def public(self) -> dict[str, object]:
        return {
            "supported": self.supported,
            "enforced": self.enforced,
            "limit_gib": self.limit_gib,
            "mechanism": self.mechanism,
            "detail": self.detail,
        }


class InMemoryBudgetController:
    """Non-enforcing controller for unit tests and injected backends."""

    def __init__(self) -> None:
        self.limit_gib: float | None = None

    def apply(self, limit_gib: float) -> MemoryBudgetState:
        self.limit_gib = float(limit_gib)
        return self.state()

    def clear(self) -> MemoryBudgetState:
        self.limit_gib = None
        return self.state()

    def state(self) -> MemoryBudgetState:
        return MemoryBudgetState(
            supported=False,
            enforced=False,
            limit_gib=self.limit_gib,
            mechanism="test_planner_only",
            detail="process limit is not kernel-enforced in this injected test app",
        )

    def usage(self) -> dict[str, object]:
        """Expose the test process RSS using the same public telemetry shape."""

        return _usage_document(
            used_gib=_process_rss_gib(os.getpid()),
            limit_gib=self.limit_gib,
            peak_gib=None,
            enforced=False,
            scope="service_process_fallback",
            resident_gib=_process_pss_gib(os.getpid()),
            resident_process_count=1,
        )

    def close(self) -> None:
        self.clear()


class LinuxCgroupMemoryBudgetController:
    """Put this service and its children in a private cgroup-v2 memory cap.

    GPU VRAM is not charged to ``memory.max``.  Qwen helpers, ffmpeg and any
    future child processes inherit the cgroup, so the UI value is a genuine
    service-wide CPU-RAM ceiling instead of an advisory Python cache size.
    """

    def __init__(self, *, pid: int | None = None, root: Path = Path("/sys/fs/cgroup")) -> None:
        self.pid = int(os.getpid() if pid is None else pid)
        self.root = Path(root)
        self.parent = self._current_directory()
        # Root-owned WSL services use the unified root: a systemd session scope
        # may permit mkdir while still denying ``memory.max`` because its
        # memory controller was not enabled in ``cgroup.subtree_control``.
        # Non-root Linux services instead use their explicitly delegated
        # systemd subtree.
        self.group_parent = (
            self.root
            if os.geteuid() == 0 and os.access(self.root, os.W_OK)
            else self.parent
        )
        self.directory = self.group_parent / f"x-minimaxh3-{self.pid}"
        self.limit_gib: float | None = None
        self._attached = False
        self._validate_support()

    def _current_directory(self) -> Path:
        for line in Path(f"/proc/{self.pid}/cgroup").read_text().splitlines():
            hierarchy, controllers, relative = line.split(":", 2)
            if hierarchy == "0" and controllers == "":
                return self.root / relative.lstrip("/")
        raise RuntimeError("cgroup v2 membership is unavailable")

    def _validate_support(self) -> None:
        if not (self.root / "cgroup.controllers").is_file():
            raise RuntimeError("cgroup v2 is unavailable; cannot enforce a hard host-RAM limit")
        if not os.access(self.group_parent, os.W_OK):
            raise RuntimeError(
                "the current cgroup v2 scope is not delegated for writes; "
                "cannot enforce a hard host-RAM limit"
            )

    def _write(self, name: str, value: str) -> None:
        (self.directory / name).write_text(value, encoding="ascii")

    def apply(self, limit_gib: float) -> MemoryBudgetState:
        limit = float(limit_gib)
        if limit <= 0 or not limit.is_integer():
            raise ValueError("host memory limit must be a positive whole GiB value")
        self.directory.mkdir(mode=0o755, exist_ok=True)
        limit_bytes = int(limit * GIB)
        # Set the ceiling before attaching a cold/idle service.  When changing
        # to a lower value Linux rejects the write if the stopped process still
        # owns too many pages, yielding a clean UI error instead of pretending
        # that the requested cap is active.
        self._write("memory.max", str(limit_bytes))
        swap_file = self.directory / "memory.swap.max"
        if swap_file.exists():
            swap_file.write_text("0", encoding="ascii")
        oom_group = self.directory / "memory.oom.group"
        if oom_group.exists():
            oom_group.write_text("1", encoding="ascii")
        self._write("cgroup.procs", str(self.pid))
        self._attached = True
        self.limit_gib = limit
        return self.state()

    def clear(self) -> MemoryBudgetState:
        if self._attached:
            # TorchInductor keeps a compile-worker pool alive across model
            # reloads.  Child processes inherit the H3 cgroup, so moving only
            # the web-server PID leaves the directory busy and makes the UI's
            # "switch model" action fail after an otherwise successful run.
            # Move every inherited service child back to the original scope.
            for _ in range(4):
                try:
                    members = [
                        value for value in
                        (self.directory / "cgroup.procs").read_text().split()
                        if value.isdigit()
                    ]
                except FileNotFoundError:
                    members = []
                if not members:
                    break
                for member in members:
                    (self.parent / "cgroup.procs").write_text(
                        member, encoding="ascii"
                    )
            self._attached = False
        self.limit_gib = None
        try:
            self.directory.rmdir()
        except FileNotFoundError:
            pass
        return self.state()

    def state(self) -> MemoryBudgetState:
        enforced = self._attached and self.limit_gib is not None
        return MemoryBudgetState(
            supported=True,
            enforced=enforced,
            limit_gib=self.limit_gib,
            mechanism="cgroup_v2_memory.max",
            detail=(
                "H3 service process and child processes are kernel-limited"
                if enforced
                else "hard limit is ready and will be applied when an engine is selected"
            ),
        )

    def usage(self) -> dict[str, object]:
        """Return current RAM charged to the complete H3 service cgroup.

        ``memory.current`` includes the web process, DiT/Qwen workers, ffmpeg,
        allocator pages and charged file cache.  This is deliberately broader
        than the web process RSS and matches the scope enforced by
        ``memory.max``.
        """

        if self._attached:
            try:
                used = int((self.directory / "memory.current").read_text().strip())
                raw_limit = (self.directory / "memory.max").read_text().strip()
                limit = self.limit_gib if raw_limit == "max" else int(raw_limit) / GIB
                peak_path = self.directory / "memory.peak"
                peak = (
                    int(peak_path.read_text().strip()) / GIB
                    if peak_path.is_file()
                    else None
                )
                memory_stat: dict[str, int] = {}
                stat_path = self.directory / "memory.stat"
                if stat_path.is_file():
                    for line in stat_path.read_text().splitlines():
                        key, value = line.split(None, 1)
                        if value.isdigit():
                            memory_stat[key] = int(value)
                memory_events: dict[str, int] = {}
                events_path = self.directory / "memory.events"
                if events_path.is_file():
                    for line in events_path.read_text().splitlines():
                        key, value = line.split(None, 1)
                        if value.isdigit():
                            memory_events[key] = int(value)
                resident, process_count = _service_resident_gib(
                    self.directory, self.pid
                )
                return _usage_document(
                    used_gib=used / GIB,
                    limit_gib=limit,
                    peak_gib=peak,
                    enforced=self.limit_gib is not None,
                    scope="h3_service_cgroup",
                    anonymous_gib=memory_stat.get("anon", 0) / GIB,
                    file_cache_gib=memory_stat.get("file", 0) / GIB,
                    oom_count=memory_events.get("oom", 0),
                    oom_kill_count=memory_events.get("oom_kill", 0),
                    resident_gib=resident,
                    resident_process_count=process_count,
                )
            except (FileNotFoundError, OSError, ValueError):
                # A model switch briefly detaches and recreates the cgroup.
                # Returning the server RSS keeps the dashboard responsive
                # without ever substituting whole-machine RAM.
                pass
        return _usage_document(
            used_gib=_process_rss_gib(self.pid),
            limit_gib=self.limit_gib,
            peak_gib=None,
            enforced=False,
            scope="service_process_fallback",
            resident_gib=_process_pss_gib(self.pid),
            resident_process_count=1,
        )

    def close(self) -> None:
        try:
            self.clear()
        except OSError:
            # Interpreter shutdown must not mask the original service exit.
            pass


__all__ = [
    "InMemoryBudgetController",
    "LinuxCgroupMemoryBudgetController",
    "MemoryBudgetState",
]
