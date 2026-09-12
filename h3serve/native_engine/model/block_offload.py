"""Production model adapters for H3 double-buffered block execution."""

from __future__ import annotations

import copy
from collections.abc import Sequence
from typing import Any

import torch
import torch.nn as nn

from ..runtime import DoubleBufferBlockExecutor, RuntimeConfig, create_stream_coordinator


def _state_tensors(module: nn.Module) -> dict[str, torch.Tensor]:
    # keep_vars exposes the registered storage that execution actually reads.
    return dict(module.state_dict(keep_vars=True))


class TorchModuleBlockBuffer:
    """One preallocated device block whose tensors are overwritten in-place."""

    def __init__(self, module: nn.Module, *, device: str) -> None:
        self.module = module
        self.device = str(device)
        self._targets = _state_tensors(module)
        if not self._targets:
            raise ValueError("H3 block buffer cannot be empty")
        self._target_values = tuple(self._targets.values())
        self._source_cache: dict[int, tuple[torch.Tensor, ...]] = {}

    @classmethod
    def from_source(cls, source: nn.Module, *, device: str) -> "TorchModuleBlockBuffer":
        # All H3 blocks share one graph/schema. deepcopy preserves Python-side
        # quant specs and injected kernel callables while allocating independent
        # registered tensor storage for the device slot.
        module = copy.deepcopy(source).eval().requires_grad_(False)
        module.to(device)
        return cls(module, device=device)

    @property
    def registered_bytes(self) -> int:
        seen: set[int] = set()
        total = 0
        for tensor in self._targets.values():
            if id(tensor) in seen:
                continue
            seen.add(id(tensor))
            total += int(tensor.numel()) * int(tensor.element_size())
        return total

    def validate_source(self, source_block: Any) -> tuple[torch.Tensor, ...]:
        if not isinstance(source_block, nn.Module):
            raise TypeError("H3 block source must be a torch.nn.Module")
        source = _state_tensors(source_block)
        if source.keys() != self._targets.keys():
            missing = sorted(self._targets.keys() - source.keys())
            extra = sorted(source.keys() - self._targets.keys())
            raise ValueError(
                f"H3 block schema mismatch; missing={missing[:4]}, extra={extra[:4]}"
            )
        for name, target in self._targets.items():
            value = source[name]
            if target.shape != value.shape or target.dtype != value.dtype:
                raise ValueError(
                    f"H3 block tensor mismatch for {name}: "
                    f"{tuple(value.shape)}/{value.dtype} != "
                    f"{tuple(target.shape)}/{target.dtype}"
                )
        values = tuple(source[name] for name in self._targets)
        self._source_cache[id(source_block)] = values
        return values

    def load_from(
        self,
        source_block: Any,
        *,
        block_index: int,
        non_blocking: bool,
    ) -> None:
        del block_index
        source_values = self._source_cache.get(id(source_block))
        if source_values is None:
            source_values = self.validate_source(source_block)
        with torch.no_grad():
            # Schema validation and state-dict traversal are startup work.
            # Keep individual DMA submissions: the mixed-dtype foreach path
            # regressed the physical 720p15 gate despite lower Python work.
            for target, source in zip(self._target_values, source_values):
                target.copy_(source, non_blocking=non_blocking)


def build_h3_block_executor(
    source_blocks: Sequence[nn.Module],
    config: RuntimeConfig,
    *,
    prefetch_depth: int = 1,
) -> DoubleBufferBlockExecutor:
    if not source_blocks:
        raise ValueError("cannot build H3 block executor without source blocks")
    first = source_blocks[0]
    buffers = tuple(
        TorchModuleBlockBuffer.from_source(first, device=config.device)
        for _ in range(config.block_buffer_count)
    )
    for source in source_blocks:
        for buffer in buffers:
            buffer.validate_source(source)

    between_block_hook = None
    if config.resource_profile == "w4a8_8gb":
        # Long sparse cells allocate differently-shaped lookup/KV slabs in
        # every H3 block.  Under the deliberate 7.25-GiB allocator ceiling,
        # released slabs can occupy enough CUDA cache that the next block's
        # full hidden-state RMS output cannot be admitted.  Preserve reusable
        # cache normally and compact only when the next unavoidable hidden
        # allocation would cross the budget.  This executes at a natural
        # single-buffer boundary, never inside an Attention kernel.
        guard_bytes = 128 * 1024**2
        long_hidden_bytes = 512 * 1024**2

        def compact_allocator_under_pressure(hidden: torch.Tensor) -> None:
            next_hidden_bytes = int(hidden.numel()) * int(hidden.element_size())
            reserved = int(torch.cuda.memory_reserved(config.device))
            if (
                next_hidden_bytes >= long_hidden_bytes
                or
                reserved + next_hidden_bytes + guard_bytes
                > config.max_device_bytes
            ):
                # Kernel launches are asynchronous: at the immediate block
                # boundary a dead slab may still be reported as allocated and
                # only become reclaimable before the next RMS allocation.  The
                # single-buffer 8-GiB path has no useful copy/compute overlap
                # to preserve here, so synchronize only after the predictive
                # pressure test fires, then release the now-dead cache.
                torch.cuda.synchronize(config.device)
                torch.cuda.empty_cache()

        between_block_hook = compact_allocator_under_pressure
    return DoubleBufferBlockExecutor(
        buffers,
        create_stream_coordinator(config),
        overlap_copy_compute=prefetch_depth == 1,
        between_block_hook=between_block_hook,
    )


__all__ = ["TorchModuleBlockBuffer", "build_h3_block_executor"]
