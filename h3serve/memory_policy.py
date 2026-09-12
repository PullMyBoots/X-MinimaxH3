"""Host-RAM policies for the fixed single-RTX4090 service."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import os
from pathlib import Path

from .contract import (
    MAX_CUSTOM_DIMENSION,
    MAX_CUSTOM_PIXELS,
    MAX_CUSTOM_SHORT_EDGE,
    MAX_NATIVE_PIXEL_FRAMES,
)


GIB = 1024**3


@dataclass(frozen=True, slots=True)
class HostMemoryProfile:
    key: str
    label: str
    minimum_ram_gib: int
    description: str
    cache_qwen_weights: bool
    pin_model_weights: bool
    copy_model_weights: bool
    preload_upscaler: bool
    exclusive_upscaler: bool
    parallel_model_build: bool
    evidence: str
    # The 8 GiB W4A8 route cannot afford a second device block buffer, but a
    # 32 GiB host can still remove pageable-H2D stalls by pinning only the DiT
    # masters.  Keeping this separate from ``pin_model_weights`` avoids also
    # pinning the 5+ GiB VAE and the 15+ GiB Qwen checkpoint, which would spend
    # RAM on components that are transferred only once per request.
    pin_transformer_weights: bool = False
    copy_transformer_weights: bool = False
    # Continuous product budget.  Legacy named profiles leave these unset;
    # the unified console fills them from the operator's process-RAM slider.
    process_limit_gib: float | None = None
    pin_transformer_budget_gib: float | None = None
    resident_transformer_blocks: int = 0

    def public(self) -> dict[str, object]:
        value = asdict(self)
        for private in (
            "cache_qwen_weights", "pin_model_weights", "copy_model_weights",
            "preload_upscaler",
            "exclusive_upscaler",
            "parallel_model_build",
            "pin_transformer_weights", "copy_transformer_weights",
            "pin_transformer_budget_gib", "resident_transformer_blocks",
        ):
            value.pop(private)
        return value


# A profile exists only when residency mechanics change. Larger machines use
# the same fastest eligible profile; there are intentionally no 128/96 aliases.
HOST_MEMORY_PROFILES: dict[str, HostMemoryProfile] = {
    "fullspeed": HostMemoryProfile(
        key="fullspeed",
        label="自动内存配置",
        minimum_ram_gib=128,
        description=(
            "统一控制台启动时的内部兼容配置；选择模型后立即由用户设置的"
            "H3进程内存硬上限替代。"
        ),
        cache_qwen_weights=True,
        pin_model_weights=True,
        copy_model_weights=True,
        preload_upscaler=True,
        exclusive_upscaler=False,
        parallel_model_build=True,
        evidence="validated",
    ),
    "generation_hot": HostMemoryProfile(
        key="generation_hot",
        label="96GB 生成优先",
        minimum_ram_gib=96,
        description="Qwen与H3保持热态；生成与原生H3二次采样共享同一热引擎。",
        cache_qwen_weights=True,
        pin_model_weights=True,
        copy_model_weights=True,
        preload_upscaler=False,
        exclusive_upscaler=True,
        parallel_model_build=True,
        evidence="validated",
    ),
    "compact": HostMemoryProfile(
        key="compact",
        label="64GB 高效兼容",
        minimum_ram_gib=64,
        description="Qwen按执行层流水读取；H3按需驻留并支持原生二次采样。",
        cache_qwen_weights=False,
        pin_model_weights=True,
        copy_model_weights=True,
        preload_upscaler=False,
        exclusive_upscaler=True,
        parallel_model_build=False,
        evidence="validated",
    ),
    "w4a8_32gb": HostMemoryProfile(
        key="w4a8_32gb",
        label="32GB W4A8 均衡",
        minimum_ram_gib=32,
        description=(
            "W4A8 DiT保持锁页主存并加速逐层搬运；Qwen按层流水读取，"
            "VAE不保留低收益整模型Pinned副本。"
        ),
        cache_qwen_weights=False,
        pin_model_weights=False,
        copy_model_weights=False,
        preload_upscaler=False,
        exclusive_upscaler=True,
        parallel_model_build=False,
        evidence="validated",
        pin_transformer_weights=True,
        copy_transformer_weights=False,
    ),
    "w4a8_16gb": HostMemoryProfile(
        key="w4a8_16gb",
        label="16GB W4A8 极限兼容",
        minimum_ram_gib=16,
        description=(
            "W4A8权重按阶段从Linux本地NVMe重读；只保留当前执行窗口，"
            "优先守住主机内存硬上限。"
        ),
        cache_qwen_weights=False,
        pin_model_weights=False,
        copy_model_weights=False,
        preload_upscaler=False,
        exclusive_upscaler=True,
        parallel_model_build=False,
        evidence="validated",
    ),
}


@dataclass(frozen=True, slots=True)
class HostMemoryStatus:
    physical_total_gib: float
    effective_limit_gib: float
    available_gib: float

    def public(self) -> dict[str, float]:
        return {
            "physical_total_gib": round(self.physical_total_gib, 2),
            "effective_limit_gib": round(self.effective_limit_gib, 2),
            "available_gib": round(self.available_gib, 2),
        }


def _meminfo() -> dict[str, int]:
    values: dict[str, int] = {}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, raw = line.split(":", 1)
            value = raw.strip().split()
            if value and value[0].isdigit():
                values[key] = int(value[0]) * 1024
    except (OSError, ValueError):
        pass
    return values


def _current_cgroup_memory_files() -> tuple[tuple[Path, Path | None], ...]:
    """Locate the calling process's cgroup limit, not only the root limit."""

    candidates: list[tuple[Path, Path | None]] = []
    try:
        memberships = Path("/proc/self/cgroup").read_text().splitlines()
    except OSError:
        memberships = []
    for line in memberships:
        try:
            _hierarchy, controllers, relative = line.split(":", 2)
        except ValueError:
            continue
        relative = relative.lstrip("/")
        if controllers == "":  # unified cgroup v2
            root = Path("/sys/fs/cgroup")
            directory = root / relative
            while True:
                candidates.append(
                    (directory / "memory.max", directory / "memory.current")
                )
                if directory == root:
                    break
                directory = directory.parent
        elif "memory" in controllers.split(","):  # cgroup v1
            root = Path("/sys/fs/cgroup/memory")
            directory = root / relative
            while True:
                candidates.append(
                    (
                        directory / "memory.limit_in_bytes",
                        directory / "memory.usage_in_bytes",
                    )
                )
                if directory == root:
                    break
                directory = directory.parent
    # Root files retain compatibility with containers that hide membership.
    candidates.extend((
        (Path("/sys/fs/cgroup/memory.max"), Path("/sys/fs/cgroup/memory.current")),
        (
            Path("/sys/fs/cgroup/memory/memory.limit_in_bytes"),
            Path("/sys/fs/cgroup/memory/memory.usage_in_bytes"),
        ),
    ))
    return tuple(dict.fromkeys(candidates))


def detect_host_memory() -> HostMemoryStatus:
    info = _meminfo()
    total = int(info.get("MemTotal", 0))
    available = int(info.get("MemAvailable", total))
    limits = [total] if total else []
    cgroup_available: list[int] = []
    for candidate, usage_path in _current_cgroup_memory_files():
        try:
            text = candidate.read_text().strip()
            if text != "max" and text.isdigit():
                value = int(text)
                # Some cgroup-v1 hosts publish a sentinel near LONG_MAX.
                if value > 0 and (not total or value < total * 16):
                    limits.append(value)
                    if usage_path is not None:
                        usage_text = usage_path.read_text().strip()
                        if usage_text.isdigit():
                            cgroup_available.append(max(0, value - int(usage_text)))
        except OSError:
            pass
    effective = min(limits) if limits else total
    if cgroup_available:
        available = min(available, *cgroup_available)
    return HostMemoryStatus(total / GIB, effective / GIB, available / GIB)


def current_process_pss_gib() -> float:
    """Physical pages reclaimed when the current hot session is rebuilt."""

    try:
        for line in Path("/proc/self/smaps_rollup").read_text().splitlines():
            if line.startswith("Pss:"):
                return int(line.split()[1]) * 1024 / GIB
    except (OSError, ValueError, IndexError):
        pass
    return 0.0


def resolve_host_memory_profile(
    requested: str,
    status: HostMemoryStatus | None = None,
) -> HostMemoryProfile:
    status = detect_host_memory() if status is None else status
    # A retail 48/64 GiB installation reports less usable RAM to Linux/WSL
    # after firmware and host reservations. Five percent is the conventional
    # capacity-label tolerance; operational headroom is enforced by each
    # profile's measured resident target, not by pretending MemTotal is exact.
    measured_peak_targets = {
        "fullspeed": 88.835,
        "generation_hot": 73.141,
        "compact": 49.931,
        # The accelerated 32 GiB W4A8 path is admitted only when the process
        # can use roughly 23 GiB while preserving six GiB for the OS/desktop.
        # Retail 32 GiB machines commonly expose about 29--30 GiB to Linux.
        "w4a8_32gb": 23.0,
        "w4a8_16gb": 12.0,
    }
    cold_build_targets = {
        "fullspeed": 88.835,
        "generation_hot": 73.141,
        "compact": 39.390,
        # Cold construction touches source mmap pages while the final pinned
        # DiT slabs are filled. Clean source pages are reclaimable, so the
        # resident build target is expected to stay below the 23 GiB hard
        # process envelope. Keep six GiB genuinely available to the host.
        "w4a8_32gb": 19.0,
        "w4a8_16gb": 10.0,
    }

    def fits(profile: HostMemoryProfile) -> bool:
        # Use measured full-product peaks plus 8GiB rather than assuming that
        # Linux/WSL exposes the retail DIMM label verbatim.
        reserve_gib = 8.0 if profile.minimum_ram_gib >= 64 else (
            6.0 if profile.minimum_ram_gib >= 32 else 2.0
        )
        effective_capacity_floor = measured_peak_targets[profile.key] + reserve_gib
        capacity_ok = status.effective_limit_gib >= effective_capacity_floor
        # At startup the process itself is still small, so MemAvailable must
        # cover the measured construction peak plus an 8GiB OS reserve.
        available_ok = status.available_gib >= cold_build_targets[profile.key] + reserve_gib
        return capacity_ok and available_ok

    if requested == "auto":
        eligible = [
            profile for profile in HOST_MEMORY_PROFILES.values()
            if fits(profile)
        ]
        if not eligible:
            raise RuntimeError(
                f"effective host RAM {status.effective_limit_gib:.1f} GiB is below "
                "the validated 16 GiB-class minimum"
            )
        return max(eligible, key=lambda profile: profile.minimum_ram_gib)
    try:
        profile = HOST_MEMORY_PROFILES[requested]
    except KeyError as error:
        raise ValueError(f"unknown host-memory profile: {requested}") from error
    research_int8_host_curve = (
        os.environ.get("H3_NATIVE_RESEARCH_INT8_HOST_CURVE", "0") == "1"
        and profile.key in {"w4a8_16gb", "w4a8_32gb"}
    )
    if not fits(profile) and not research_int8_host_curve:
        raise RuntimeError(
            f"{profile.label} requires a {profile.minimum_ram_gib} GiB class host "
            f"and enough free RAM; effective limit is {status.effective_limit_gib:.1f} "
            f"GiB and currently available is {status.available_gib:.1f} GiB"
        )
    return profile


PRODUCT_MEMORY_MINIMUM_GIB = {"w4a8": 12, "int8": 24}
PRODUCT_MEMORY_VALIDATED_GIB = {"w4a8": 16, "int8": 32}
PRODUCT_HOST_RESERVE_GIB = 6


def host_memory_budget_bounds(
    weight_tier: str,
    status: HostMemoryStatus | None = None,
) -> tuple[int, int]:
    """Return the public service-process slider bounds.

    The value is an application/cgroup allowance, not the machine's DIMM
    label.  Six GiB remain outside the H3 cgroup for the OS and desktop.
    """

    if weight_tier not in PRODUCT_MEMORY_MINIMUM_GIB:
        raise ValueError(f"unknown weight tier: {weight_tier}")
    status = detect_host_memory() if status is None else status
    minimum = PRODUCT_MEMORY_MINIMUM_GIB[weight_tier]
    host_ceiling = min(status.physical_total_gib, status.effective_limit_gib)
    maximum = int(host_ceiling - PRODUCT_HOST_RESERVE_GIB)
    if maximum < minimum:
        raise RuntimeError(
            f"{weight_tier.upper()} requires at least {minimum} GiB for the H3 "
            f"service plus {PRODUCT_HOST_RESERVE_GIB} GiB reserved for the system"
        )
    return minimum, maximum


def _int8_pin_budget(limit_gib: float) -> float:
    """Piecewise-linear projection of the measured 2026-08-31 INT8 curve."""

    points = ((16.0, 3.765), (24.0, 11.670), (28.0, 15.811), (32.0, 17.693))
    if limit_gib <= points[0][0]:
        return points[0][1]
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        if limit_gib <= x1:
            ratio = (limit_gib - x0) / (x1 - x0)
            return y0 + ratio * (y1 - y0)
    # More than 32 GiB application RAM did not improve hot DiT latency.
    return points[-1][1]


def resolve_host_memory_budget_profile(
    weight_tier: str,
    requested_limit_gib: float,
    *,
    vram_profile: str,
    status: HostMemoryStatus | None = None,
) -> HostMemoryProfile:
    """Compile one user RAM allowance into the fastest measured residency.

    This function runs only when entering/reloading an engine.  It never sits
    in the per-step DiT path.  All selected strategies are byte-equivalent:
    the budget changes residency/copy scheduling, not weights or arithmetic.
    """

    minimum, maximum = host_memory_budget_bounds(weight_tier, status)
    try:
        limit = float(requested_limit_gib)
    except (TypeError, ValueError) as error:
        raise ValueError("host_memory_limit_gib must be numeric") from error
    if not limit.is_integer():
        raise ValueError("host_memory_limit_gib must be a whole GiB value")
    if not minimum <= limit <= maximum:
        raise ValueError(
            f"{weight_tier.upper()} host_memory_limit_gib must be between "
            f"{minimum} and {maximum} GiB on this host"
        )
    if vram_profile not in {"8gb", "16gb", "24gb"}:
        raise ValueError(f"unknown VRAM profile: {vram_profile}")

    resident_blocks = 0
    if weight_tier == "int8":
        pin_budget = _int8_pin_budget(limit)
        description = (
            "INT8权重按层流式执行，并按主机预算连续扩大锁页DiT缓存；"
            "32GiB应用额度达到实测热态速度拐点。"
        )
    elif vram_profile == "8gb":
        pin_budget = min(10.972, max(0.0, limit - 10.75))
        description = "8GB显存采用流式DiT；额外主机预算只用于可收益的锁页Block缓存。"
    elif vram_profile == "16gb":
        if limit >= 17.0:
            resident_blocks = 24
            pin_budget = min(5.705, max(0.0, limit - 10.8))
        else:
            resident_blocks = 30
            pin_budget = 0.0
        description = "16GB显存自动平衡GPU常驻Block与主机锁页缓存。"
    else:
        # This is an upper bound.  The request-local dense-Actual admission
        # model may lower it for long/high-resolution contexts so SageAttention
        # always retains its transient Q/K/V workspace.
        resident_blocks = 49
        pin_budget = min(0.219, max(0.0, limit - 10.6))
        description = (
            "24GB显存按任务的稠密Actual峰值动态选择最快安全Block前缀；"
            "多余主机缓存不再无效扩张。"
        )

    return HostMemoryProfile(
        key=f"budget_{weight_tier}_{int(limit)}g_{vram_profile}",
        label=f"{weight_tier.upper()} · H3进程上限 {int(limit)}GiB",
        minimum_ram_gib=minimum,
        description=description,
        cache_qwen_weights=False,
        pin_model_weights=False,
        copy_model_weights=False,
        preload_upscaler=False,
        exclusive_upscaler=True,
        parallel_model_build=False,
        evidence=(
            "validated"
            if limit >= PRODUCT_MEMORY_VALIDATED_GIB[weight_tier]
            else "experimental_low_memory"
        ),
        pin_transformer_weights=pin_budget > 0.0,
        copy_transformer_weights=False,
        process_limit_gib=limit,
        pin_transformer_budget_gib=round(pin_budget, 3),
        resident_transformer_blocks=resident_blocks,
    )


def validate_workload_for_profile(
    profile: HostMemoryProfile,
    *,
    width: int,
    height: int,
    frames: int,
) -> None:
    """Fail before queueing workloads outside a measured host-RAM envelope."""

    del profile
    width = int(width)
    height = int(height)
    frames = int(frames)
    pixels = width * height
    if (
        width > MAX_CUSTOM_DIMENSION
        or height > MAX_CUSTOM_DIMENSION
        or min(width, height) > MAX_CUSTOM_SHORT_EDGE
        or pixels > MAX_CUSTOM_PIXELS
        or frames > 362
        or pixels * frames > MAX_NATIVE_PIXEL_FRAMES
    ):
        raise ValueError(
            "workload exceeds the validated native spatial-temporal envelope "
            f"(width*height*frames <= {MAX_NATIVE_PIXEL_FRAMES})"
        )


def validate_profile_for_weight_tier(
    profile: HostMemoryProfile,
    weight_tier: str,
) -> None:
    """Keep W4A8-only host envelopes away from the larger INT8 weights."""

    if (
        os.environ.get("H3_NATIVE_RESEARCH_INT8_HOST_CURVE", "0") == "1"
        and weight_tier == "int8"
        and profile.key in {"w4a8_16gb", "w4a8_32gb"}
    ):
        # Calibration-only: reuse the low-RAM residency mechanics while the
        # fixed INT8 launcher still owns the actual weights and CUDA backend.
        # Public launchers never set this flag.
        return

    if profile.key in {"w4a8_16gb", "w4a8_32gb"} and weight_tier != "w4a8":
        raise ValueError(
            f"{profile.label} is validated only for the W4A8 8GB launchers; "
            "select a 64GB-or-higher host profile for INT8"
        )


__all__ = [
    "HOST_MEMORY_PROFILES",
    "HostMemoryProfile",
    "HostMemoryStatus",
    "detect_host_memory",
    "host_memory_budget_bounds",
    "current_process_pss_gib",
    "_current_cgroup_memory_files",
    "resolve_host_memory_profile",
    "resolve_host_memory_budget_profile",
    "PRODUCT_HOST_RESERVE_GIB",
    "PRODUCT_MEMORY_MINIMUM_GIB",
    "PRODUCT_MEMORY_VALIDATED_GIB",
    "validate_profile_for_weight_tier",
    "validate_workload_for_profile",
]
