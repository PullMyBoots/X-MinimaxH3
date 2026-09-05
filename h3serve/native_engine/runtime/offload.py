"""One- or two-buffer transformer block execution for one accelerator."""

from __future__ import annotations

from typing import Any, Callable, Protocol, Sequence, TypeVar

from .streams import StreamCoordinator


class BlockBuffer(Protocol):
    """Adapter implemented by the model/weight layer for one device buffer."""

    def load_from(
        self,
        source_block: Any,
        *,
        block_index: int,
        non_blocking: bool,
    ) -> None:
        """Copy one host-resident block into this preallocated device buffer."""


HiddenState = TypeVar("HiddenState")
BlockRunner = Callable[[int, BlockBuffer, HiddenState, Any], HiddenState]
BetweenBlockHook = Callable[[HiddenState], None]


class DoubleBufferBlockExecutor:
    """Execute with one serial buffer or two copy/compute-overlapped buffers.

    Source blocks remain on the host. The model adapter owns exactly two
    shape-compatible device buffers and implements :class:`BlockBuffer`.
    Unlike a cyclic prefetch loop, the executor never copies block zero after
    the final block, and it uses per-slot events rather than synchronizing the
    whole device between blocks.
    """

    def __init__(
        self,
        buffers: Sequence[BlockBuffer],
        streams: StreamCoordinator,
        *,
        overlap_copy_compute: bool = True,
        between_block_hook: BetweenBlockHook | None = None,
    ) -> None:
        if len(buffers) not in (1, 2):
            raise ValueError("block executor requires one or two buffers")
        self._buffers = tuple(buffers)
        self._streams = streams
        # A single slot cannot be overwritten until its kernels finish.  Use
        # the established synchronous correctness path instead of inserting a
        # device-wide barrier between every layer.
        self._overlap_copy_compute = bool(overlap_copy_compute and len(buffers) == 2)
        self._between_block_hook = between_block_hook

    @property
    def buffers(self) -> tuple[BlockBuffer, ...]:
        return self._buffers

    def _enqueue_copy(self, source: Any, block_index: int, slot: int) -> None:
        self._streams.wait_compute_done_before_copy(slot)
        with self._streams.copy_scope():
            self._buffers[slot].load_from(
                source,
                block_index=block_index,
                non_blocking=True,
            )
            self._streams.record_copy_ready(slot)

    def run(
        self,
        source_blocks: Sequence[Any],
        hidden_state: HiddenState,
        run_block: BlockRunner[HiddenState],
        shared_inputs: Any = None,
        *,
        block_index_offset: int = 0,
    ) -> HiddenState:
        if not source_blocks:
            return hidden_state

        # The final request-local hidden state (including Forecast/controller
        # setup) exists only now.  Let constrained single-buffer backends
        # release setup slabs at this exact phase boundary before block zero.
        if self._between_block_hook is not None:
            self._between_block_hook(hidden_state)

        if not self._overlap_copy_compute:
            # Correctness/reference path: copy and execute on the caller
            # stream. It deliberately forfeits overlap so an A/B can separate
            # model-buffer errors from cross-stream scheduling errors.
            buffer = self._buffers[0]
            for local_index, source in enumerate(source_blocks):
                block_index = block_index_offset + local_index
                buffer.load_from(
                    source,
                    block_index=block_index,
                    non_blocking=False,
                )
                hidden_state = run_block(
                    block_index,
                    buffer,
                    hidden_state,
                    shared_inputs,
                )
                if (
                    self._between_block_hook is not None
                    and local_index + 1 < len(source_blocks)
                ):
                    self._between_block_hook(hidden_state)
            return hidden_state

        self._streams.reset(len(self._buffers))
        self._enqueue_copy(source_blocks[0], block_index=0, slot=0)

        try:
            for local_index in range(len(source_blocks)):
                block_index = block_index_offset + local_index
                slot = local_index % len(self._buffers)
                self._streams.wait_copy_ready(slot)

                next_local_index = local_index + 1
                if next_local_index < len(source_blocks):
                    next_index = block_index_offset + next_local_index
                    next_slot = next_local_index % len(self._buffers)
                    self._enqueue_copy(
                        source_blocks[next_local_index],
                        block_index=next_index,
                        slot=next_slot,
                    )

                with self._streams.compute_scope():
                    hidden_state = run_block(
                        block_index,
                        self._buffers[slot],
                        hidden_state,
                        shared_inputs,
                    )
                    self._streams.record_compute_done(slot)
                if (
                    self._between_block_hook is not None
                    and local_index + 1 < len(source_blocks)
                ):
                    self._between_block_hook(hidden_state)

            self._streams.handoff_to_caller()
            return hidden_state
        except Exception:
            # Weight copies may still be in flight. Error recovery must not
            # release their buffers until both streams are quiescent.
            self._streams.synchronize()
            raise
