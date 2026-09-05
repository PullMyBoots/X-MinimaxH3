"""Single-device runtime primitives for the native H3 engine.

Importing this package never initializes CUDA.  CUDA objects are created only
when :func:`create_stream_coordinator` is called with a CUDA configuration.
"""

from .config import OffloadMode, RuntimeConfig
from .offload import BlockBuffer, DoubleBufferBlockExecutor
from .pinned_pool import PackedPinnedTensors, pack_pinned_tensors
from .residency import (
    ComponentResidency,
    HostComponentResidency,
    ImmutablePinnedModuleResidency,
    ResidencyBudgetError,
    ResidencyManager,
    TorchModuleResidency,
)
from .streams import StreamCoordinator, create_stream_coordinator

__all__ = [
    "BlockBuffer",
    "ComponentResidency",
    "DoubleBufferBlockExecutor",
    "HostComponentResidency",
    "ImmutablePinnedModuleResidency",
    "OffloadMode",
    "PackedPinnedTensors",
    "ResidencyBudgetError",
    "ResidencyManager",
    "RuntimeConfig",
    "StreamCoordinator",
    "TorchModuleResidency",
    "create_stream_coordinator",
    "pack_pinned_tensors",
]
