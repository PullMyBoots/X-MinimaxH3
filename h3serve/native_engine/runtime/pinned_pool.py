"""Compact CUDA-pinned storage for immutable inference tensors.

PyTorch's pinned allocator rounds many individual allocations up to large size
classes.  H3 has more than a thousand immutable tensors, so pinning each tensor
separately can retain substantially more physical RAM than the checkpoint
payload.  This module packs same-dtype tensors into moderately sized pinned
slabs while returning normal strided tensor views.  The views remain eligible
for non-blocking H2D copies and preserve the model's values and strides.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import mmap
from typing import Any, Callable, Iterable
import weakref


DEFAULT_PINNED_SLAB_BYTES = 256 * 1024 * 1024
_ALIGNMENT_BYTES = 256


def _unregister_host_memory(pointer: int) -> None:
    """Best-effort cleanup for a slab registered with the CUDA driver."""

    try:
        import torch

        torch.cuda.cudart().cudaHostUnregister(pointer)
    except Exception:
        # CUDA may already be shutting down during interpreter finalization.
        pass


def _register_exact_pinned_tensor(tensor: Any) -> None:
    """Pin an existing exact-size CPU allocation without allocator rounding."""

    import torch

    error = torch.cuda.cudart().cudaHostRegister(
        int(tensor.data_ptr()),
        int(tensor.numel()) * int(tensor.element_size()),
        1,  # cudaHostRegisterPortable
    )
    if error != torch.cuda.cudart().cudaError.success:
        raise RuntimeError(f"cudaHostRegister failed: {error}")
    weakref.finalize(tensor, _unregister_host_memory, int(tensor.data_ptr()))


@dataclass(frozen=True, slots=True)
class PackedPinnedTensors:
    tensors: tuple[Any, ...]
    slabs: tuple[Any, ...]
    logical_bytes: int
    allocated_bytes: int


@dataclass(slots=True)
class RegisteredPinnedMemory:
    """CUDA host registrations over existing CPU tensor storage."""

    ranges: tuple[tuple[int, int], ...]
    registered_bytes: int
    _closed: bool = False

    def close(self) -> None:
        if self._closed:
            return
        try:
            import torch

            for pointer, _size in reversed(self.ranges):
                torch.cuda.cudart().cudaHostUnregister(pointer)
        except Exception:
            pass
        self._closed = True

    def __del__(self) -> None:
        self.close()


def register_pinned_tensors_in_place(
    sources: Iterable[Any],
) -> RegisteredPinnedMemory:
    """Pin existing immutable tensor storage without making a second copy.

    CUDA registration is page-granular. Tensor allocations that share a host
    page are merged before registration, preventing overlapping-register
    errors while retaining the exact module storage and aliases.
    """

    import torch

    page = int(mmap.PAGESIZE)
    intervals: set[tuple[int, int]] = set()
    for source in tuple(source.detach() for source in sources):
        if source.device.type != "cpu":
            raise ValueError("pinned tensor sources must be on CPU")
        storage = source.untyped_storage()
        pointer = int(storage.data_ptr())
        size = int(storage.nbytes())
        if pointer == 0 or size == 0:
            continue
        start = pointer - pointer % page
        end = ((pointer + size + page - 1) // page) * page
        intervals.add((start, end))

    merged: list[list[int]] = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])

    registered: list[tuple[int, int]] = []
    try:
        for start, end in merged:
            size = end - start
            error = torch.cuda.cudart().cudaHostRegister(
                start,
                size,
                1,  # cudaHostRegisterPortable
            )
            if error != torch.cuda.cudart().cudaError.success:
                raise RuntimeError(f"cudaHostRegister failed: {error}")
            registered.append((start, size))
    except Exception:
        for pointer, _size in reversed(registered):
            try:
                torch.cuda.cudart().cudaHostUnregister(pointer)
            except Exception:
                pass
        raise
    return RegisteredPinnedMemory(
        ranges=tuple(registered),
        registered_bytes=sum(size for _pointer, size in registered),
    )


def _storage_elements(tensor: Any) -> int:
    if tensor.numel() == 0:
        return 0
    if any(int(stride) < 0 for stride in tensor.stride()):
        raise ValueError("negative-stride tensors cannot be packed")
    return 1 + sum(
        (int(size) - 1) * int(stride)
        for size, stride in zip(tensor.size(), tensor.stride())
        if int(size) > 0
    )


def pack_pinned_tensors(
    sources: Iterable[Any],
    *,
    slab_bytes: int = DEFAULT_PINNED_SLAB_BYTES,
    source_copied: Callable[[int], None] | None = None,
) -> PackedPinnedTensors:
    """Copy CPU tensors into compact pinned slabs without changing values.

    Slabs are separated by dtype because a tensor view cannot reinterpret a
    typed storage safely. Tensor starts are 256-byte aligned. The backing CPU
    allocations are registered in-place with CUDA rather than created through
    PyTorch's caching pinned allocator; this preserves exact slab capacities
    instead of rounding each large allocation to a power-of-two size class.
    """

    import torch

    source_list = tuple(source.detach() for source in sources)
    if not source_list:
        return PackedPinnedTensors((), (), 0, 0)
    if slab_bytes < _ALIGNMENT_BYTES or slab_bytes % _ALIGNMENT_BYTES:
        raise ValueError("slab_bytes must be a positive multiple of 256")
    for source in source_list:
        if source.device.type != "cpu":
            raise ValueError("pinned tensor sources must be on CPU")

    grouped: dict[Any, list[tuple[int, Any]]] = defaultdict(list)
    for index, source in enumerate(source_list):
        grouped[source.dtype].append((index, source))

    result: list[Any | None] = [None] * len(source_list)
    slabs: list[Any] = []
    logical_bytes = 0
    allocated_bytes = 0

    for dtype, entries in grouped.items():
        element_bytes = torch.empty((), dtype=dtype).element_size()
        alignment_elements = max(1, _ALIGNMENT_BYTES // element_bytes)
        slab_elements = max(alignment_elements, slab_bytes // element_bytes)
        current = None
        cursor = 0

        for index, source in entries:
            span = _storage_elements(source)
            logical_bytes += int(source.numel()) * int(source.element_size())
            aligned = (
                (cursor + alignment_elements - 1) // alignment_elements
            ) * alignment_elements
            if current is None or aligned + span > current.numel():
                capacity = (
                    ((span + alignment_elements - 1) // alignment_elements)
                    * alignment_elements
                    if span > slab_elements
                    else slab_elements
                )
                current = torch.empty(
                    capacity,
                    dtype=dtype,
                    device="cpu",
                )
                _register_exact_pinned_tensor(current)
                slabs.append(current)
                allocated_bytes += int(current.numel()) * element_bytes
                cursor = 0
                aligned = 0

            base = current.narrow(0, aligned, span)
            view = torch.as_strided(base, source.size(), source.stride())
            view.copy_(source)
            result[index] = view
            if source_copied is not None:
                source_copied(
                    int(source.numel()) * int(source.element_size())
                )
            cursor = aligned + span

    if any(tensor is None for tensor in result):
        raise RuntimeError("internal pinned-pool packing failure")
    return PackedPinnedTensors(
        tensors=tuple(result),
        slabs=tuple(slabs),
        logical_bytes=logical_bytes,
        allocated_bytes=allocated_bytes,
    )


__all__ = [
    "DEFAULT_PINNED_SLAB_BYTES",
    "PackedPinnedTensors",
    "pack_pinned_tensors",
    "RegisteredPinnedMemory",
    "register_pinned_tensors_in_place",
]
