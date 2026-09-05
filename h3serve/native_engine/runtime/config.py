"""Typed runtime policy for the supported RTX 4090 deployment."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import os
from typing import Literal

from ..resource_backends import get_resource_backend


class OffloadMode(str, Enum):
    """Transformer residency strategy.

    ``BLOCK`` is the expected 24 GiB production mode. ``MODEL`` is useful for
    smaller future checkpoints, while ``RESIDENT`` is primarily a diagnostic
    mode and is not expected to fit the current H3 deployment checkpoint.
    """

    BLOCK = "block"
    MODEL = "model"
    RESIDENT = "resident"


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    """Hardware and memory policy fixed at service startup.

    Per-request creative controls deliberately do not live here.  The native
    service targets one RTX 4090, one generation at a time, and batch size one.
    """

    device: str = "cuda:0"
    expected_compute_capability: tuple[int, int] = (8, 9)
    max_device_bytes: int = 23 * 1024**3
    batch_size: int = 1
    offload_mode: OffloadMode = OffloadMode.BLOCK
    block_buffer_count: int = 2
    pin_host_weights: bool = True
    copy_stream_priority: int = 0
    compute_stream_priority: int = -1
    clear_cache_on_phase_transition: bool = False
    retain_block_buffers_between_requests: bool = True
    weight_tier: Literal["int8", "w4a8"] = "int8"
    # Production launchers always set this explicitly.  ``None`` is retained
    # only for low-level tests and legacy callers, where the old capacity
    # inference remains a compatibility fallback.
    backend_profile: Literal[
        "int8_24gb", "int8_16gb",
        "w4a8_24gb", "w4a8_16gb", "w4a8_8gb",
    ] | None = None

    @property
    def provisioned_device_gib(self) -> float:
        """Physical/configured tier before the fixed 768-MiB service reserve."""

        return self.max_device_bytes / 1024**3 + 0.75

    @property
    def resource_profile(self) -> str:
        if self.backend_profile is not None:
            return self.backend_profile
        provisioned = self.provisioned_device_gib
        if self.weight_tier == "w4a8":
            if provisioned >= 20.0:
                return "w4a8_24gb"
            if provisioned >= 12.0:
                return "w4a8_16gb"
            return "w4a8_8gb"
        if provisioned >= 20.0:
            return "int8_24gb"
        if provisioned >= 12.0:
            return "int8_16gb"
        return "future_compact_8gb"

    def __post_init__(self) -> None:
        if self.batch_size != 1:
            raise ValueError("the RTX 4090 runtime supports batch_size=1 only")
        if self.block_buffer_count not in (1, 2):
            raise ValueError("block offload supports one or two device buffers")
        if self.max_device_bytes <= 0:
            raise ValueError("max_device_bytes must be positive")
        if self.weight_tier not in ("int8", "w4a8"):
            raise ValueError("weight_tier must be int8 or w4a8")
        if self.backend_profile is not None:
            backend = get_resource_backend(
                self.backend_profile,
                weight_tier=self.weight_tier,
            )
            if self.provisioned_device_gib > backend.provisioned_gib + 1.0e-6:
                raise ValueError(
                    "resource backend allocator exceeds its provisioned tier"
                )
        if self.device != "cpu" and not self.device.startswith("cuda:"):
            raise ValueError("device must be 'cpu' or an explicit CUDA device such as 'cuda:0'")

    @classmethod
    def for_cuda_device(
        cls,
        device: str = "cuda:0",
        *,
        weight_tier: Literal["int8", "w4a8"] = "int8",
        provisioned_limit_gib: float | None = None,
        backend_profile: Literal[
            "int8_24gb", "int8_16gb",
            "w4a8_24gb", "w4a8_16gb", "w4a8_8gb",
        ] | None = None,
    ) -> "RuntimeConfig":
        """Build one device-sized budget while retaining a 768-MiB reserve.

        The optional override is useful for validating 8/16 GiB policies on a
        larger development GPU.  It can only reduce the physical budget.
        """

        import torch

        if backend_profile is not None:
            backend = get_resource_backend(
                backend_profile,
                weight_tier=weight_tier,
            )
            if provisioned_limit_gib is None:
                provisioned_limit_gib = backend.provisioned_gib
            elif provisioned_limit_gib > backend.provisioned_gib:
                raise ValueError(
                    "provisioned limit cannot exceed the fixed resource backend"
                )

        total_bytes = int(torch.cuda.get_device_properties(device).total_memory)
        if provisioned_limit_gib is not None:
            if provisioned_limit_gib <= 0:
                raise ValueError("provisioned_limit_gib must be positive")
            total_bytes = min(
                total_bytes, int(float(provisioned_limit_gib) * 1024**3)
            )
        override = os.environ.get("H3_NATIVE_MAX_VRAM_GIB", "").strip()
        if override:
            try:
                override_bytes = int(float(override) * 1024**3)
            except ValueError as error:
                raise ValueError("H3_NATIVE_MAX_VRAM_GIB must be numeric") from error
            if override_bytes <= 0:
                raise ValueError("H3_NATIVE_MAX_VRAM_GIB must be positive")
            total_bytes = min(total_bytes, override_bytes)
        max_device_bytes = total_bytes - 768 * 1024**2
        if max_device_bytes <= 0:
            raise ValueError("CUDA device needs more than 1 GiB total VRAM")
        return cls(
            device=device,
            max_device_bytes=max_device_bytes,
            weight_tier=weight_tier,
            backend_profile=backend_profile,
        )

    @classmethod
    def cpu_test(cls) -> "RuntimeConfig":
        """Return a no-CUDA configuration for unit tests and adapter bring-up."""

        return cls(
            device="cpu",
            expected_compute_capability=(0, 0),
            max_device_bytes=8 * 1024**3,
            pin_host_weights=False,
            clear_cache_on_phase_transition=False,
        )
