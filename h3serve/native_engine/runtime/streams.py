"""Copy/compute stream coordination without import-time CUDA side effects."""

from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import AbstractContextManager, nullcontext
from importlib import import_module
from typing import Any

from .config import RuntimeConfig


class StreamCoordinator(ABC):
    """Small synchronization surface consumed by block offload.

    The CPU implementation makes the scheduling logic deterministic in unit
    tests.  The CUDA implementation uses per-buffer events, avoiding a global
    synchronization at every transformer block.
    """

    @abstractmethod
    def reset(self, buffer_count: int) -> None:
        """Begin one block traversal."""

    @abstractmethod
    def copy_scope(self) -> AbstractContextManager[Any]:
        """Return the stream context used for host-to-device copies."""

    @abstractmethod
    def compute_scope(self) -> AbstractContextManager[Any]:
        """Return the stream context used for block execution."""

    @abstractmethod
    def record_copy_ready(self, slot: int) -> None:
        """Record that ``slot`` has received its source block."""

    @abstractmethod
    def wait_copy_ready(self, slot: int) -> None:
        """Make compute wait until ``slot`` is ready."""

    @abstractmethod
    def record_compute_done(self, slot: int) -> None:
        """Record that compute no longer reads ``slot``."""

    @abstractmethod
    def wait_compute_done_before_copy(self, slot: int) -> None:
        """Prevent the copy stream from overwriting an in-use slot."""

    @abstractmethod
    def handoff_to_caller(self) -> None:
        """Make the stream that entered the traversal wait for its result."""

    @abstractmethod
    def synchronize(self) -> None:
        """Synchronize both streams for shutdown, tests, or error recovery."""


class CpuStreamCoordinator(StreamCoordinator):
    """Synchronous coordinator used by CPU tests."""

    def reset(self, buffer_count: int) -> None:
        self._copy_ready = [False] * buffer_count
        self._compute_done = [True] * buffer_count

    def copy_scope(self) -> AbstractContextManager[Any]:
        return nullcontext()

    def compute_scope(self) -> AbstractContextManager[Any]:
        return nullcontext()

    def record_copy_ready(self, slot: int) -> None:
        self._copy_ready[slot] = True
        self._compute_done[slot] = False

    def wait_copy_ready(self, slot: int) -> None:
        if not self._copy_ready[slot]:
            raise RuntimeError(f"block buffer {slot} was consumed before its copy completed")

    def record_compute_done(self, slot: int) -> None:
        self._compute_done[slot] = True

    def wait_compute_done_before_copy(self, slot: int) -> None:
        if not self._compute_done[slot]:
            raise RuntimeError(f"block buffer {slot} would be overwritten during compute")

    def handoff_to_caller(self) -> None:
        return None

    def synchronize(self) -> None:
        return None


class TorchCudaStreamCoordinator(StreamCoordinator):
    """CUDA stream implementation created lazily at engine startup."""

    def __init__(self, config: RuntimeConfig) -> None:
        torch = import_module("torch")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA runtime requested but torch.cuda.is_available() is false")

        self._torch = torch
        self._device = torch.device(config.device)
        with torch.cuda.device(self._device):
            capability = tuple(torch.cuda.get_device_capability(self._device))
            if capability != config.expected_compute_capability:
                raise RuntimeError(
                    "native runtime is calibrated for compute capability "
                    f"{config.expected_compute_capability}, found {capability}"
                )
            self._copy_stream = torch.cuda.Stream(
                device=self._device, priority=config.copy_stream_priority
            )
            self._compute_stream = torch.cuda.Stream(
                device=self._device, priority=config.compute_stream_priority
            )
        self._caller_stream = None
        self._copy_ready: list[Any] = []
        self._compute_done: list[Any | None] = []

    def reset(self, buffer_count: int) -> None:
        torch = self._torch
        with torch.cuda.device(self._device):
            self._caller_stream = torch.cuda.current_stream(self._device)
            # Packed inputs, RoPE tables and AdaLN rows are produced by the
            # caller stream immediately before block traversal. Non-default
            # streams do not inherit that dependency. Without this handoff the
            # first block can observe partially-written activations even when
            # its own weight-copy event is correct.
            # A forecasted H3 step can call the block stack in multiple
            # ranges. Before reusing slot zero for the next traversal, make
            # the copy stream wait for all caller-stream block reads already
            # queued by the previous range. This is stream-ordered and does
            # not introduce a device-wide or CPU synchronization.
            self._copy_stream.wait_stream(self._caller_stream)
            self._copy_ready = [torch.cuda.Event(enable_timing=False) for _ in range(buffer_count)]
            self._compute_done = [None] * buffer_count

    def copy_scope(self) -> AbstractContextManager[Any]:
        return self._torch.cuda.stream(self._copy_stream)

    def compute_scope(self) -> AbstractContextManager[Any]:
        if self._caller_stream is None:
            raise RuntimeError("stream traversal was not initialized")
        # Execute model kernels on the stream that entered the traversal.
        # Some third-party SM89 extensions used by H3 do not reliably follow a
        # newly-created PyTorch compute stream. Keeping compute on the caller
        # stream preserves their ordering while the dedicated copy stream can
        # still prefetch the next block over PCIe.
        return self._torch.cuda.stream(self._caller_stream)

    def record_copy_ready(self, slot: int) -> None:
        self._copy_ready[slot].record(self._copy_stream)

    def wait_copy_ready(self, slot: int) -> None:
        if self._caller_stream is None:
            raise RuntimeError("stream traversal was not initialized")
        self._caller_stream.wait_event(self._copy_ready[slot])

    def record_compute_done(self, slot: int) -> None:
        if self._caller_stream is None:
            raise RuntimeError("stream traversal was not initialized")
        event = self._torch.cuda.Event(enable_timing=False)
        event.record(self._caller_stream)
        self._compute_done[slot] = event

    def wait_compute_done_before_copy(self, slot: int) -> None:
        event = self._compute_done[slot]
        if event is not None:
            self._copy_stream.wait_event(event)

    def handoff_to_caller(self) -> None:
        if self._caller_stream is None:
            raise RuntimeError("stream traversal was not initialized")
        # Computation already ran on the caller stream.
        return None

    def synchronize(self) -> None:
        self._copy_stream.synchronize()
        if self._caller_stream is not None:
            self._caller_stream.synchronize()


def create_stream_coordinator(config: RuntimeConfig) -> StreamCoordinator:
    """Construct the configured stream backend without importing CUDA on CPU."""

    if config.device == "cpu":
        coordinator = CpuStreamCoordinator()
        coordinator.reset(config.block_buffer_count)
        return coordinator
    return TorchCudaStreamCoordinator(config)
