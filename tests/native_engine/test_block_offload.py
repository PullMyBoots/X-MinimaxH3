from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn as nn

from h3serve.native_engine.model import (
    H3BlockStack,
    block_cancellation,
    build_h3_block_executor,
)
from h3serve.native_engine.model.lora import AdaLNCurveRows
from h3serve.native_engine.runtime import RuntimeConfig


class _TinyBlock(nn.Module):
    def __init__(self, scale: float, bias: float) -> None:
        super().__init__()
        self.register_buffer("scale", torch.tensor(scale))
        self.register_buffer("bias", torch.tensor(bias))

    def forward(self, value, **kwargs):
        del kwargs
        return value * self.scale + self.bias


class H3BlockOffloadTest(unittest.TestCase):
    def _stack(self) -> H3BlockStack:
        return H3BlockStack(
            [_TinyBlock(2.0, 1.0), _TinyBlock(3.0, -2.0), _TinyBlock(0.5, 4.0)],
            compressed_curve=torch.zeros(1, 1),
        )

    @staticmethod
    def _run(stack: H3BlockStack, start: int = 0, stop: int = 3):
        return stack.run_range(
            torch.tensor([2.0]),
            start=start,
            stop=stop,
            timestep_rows=AdaLNCurveRows(compressed=torch.zeros(1, 1)),
            modulation_segments=(),
            frequencies=torch.zeros(1),
        )

    def test_double_buffer_matches_resident_stack(self) -> None:
        stack = self._stack()
        expected = self._run(stack)
        executor = build_h3_block_executor(stack.blocks, RuntimeConfig.cpu_test())
        stack.configure_block_executor(executor)
        actual = self._run(stack)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_single_buffer_matches_resident_stack(self) -> None:
        stack = self._stack()
        expected = self._run(stack)
        config = RuntimeConfig.cpu_test()
        config = RuntimeConfig(
            device=config.device,
            expected_compute_capability=config.expected_compute_capability,
            max_device_bytes=config.max_device_bytes,
            block_buffer_count=1,
            pin_host_weights=False,
        )
        executor = build_h3_block_executor(
            stack.blocks, config, prefetch_depth=0
        )
        self.assertEqual(len(executor.buffers), 1)
        stack.configure_block_executor(executor)
        actual = self._run(stack)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_8gb_single_buffer_compacts_only_under_next_block_pressure(self) -> None:
        gib = 1024**3
        config = RuntimeConfig(
            device="cpu",
            max_device_bytes=int(7.25 * gib),
            block_buffer_count=1,
            pin_host_weights=False,
            weight_tier="w4a8",
            backend_profile="w4a8_8gb",
        )
        executor = build_h3_block_executor(
            self._stack().blocks, config, prefetch_depth=0
        )
        hook = executor._between_block_hook
        self.assertIsNotNone(hook)
        one_gib_hidden = SimpleNamespace(
            numel=lambda: gib // 2,
            element_size=lambda: 2,
        )
        with (
            patch("torch.cuda.memory_reserved", return_value=int(6.5 * gib)),
            patch("torch.cuda.synchronize") as synchronize,
            patch("torch.cuda.empty_cache") as empty_cache,
        ):
            hook(one_gib_hidden)
        synchronize.assert_called_once_with("cpu")
        empty_cache.assert_called_once_with()

    def test_range_offset_loads_the_requested_sources(self) -> None:
        stack = self._stack()
        expected = self._run(stack, start=1, stop=3)
        executor = build_h3_block_executor(stack.blocks, RuntimeConfig.cpu_test())
        stack.configure_block_executor(executor)
        actual = self._run(stack, start=1, stop=3)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_hybrid_resident_prefix_matches_fully_resident_stack(self) -> None:
        stack = self._stack()
        expected = self._run(stack)
        executor = build_h3_block_executor(
            stack.blocks[1:], RuntimeConfig.cpu_test()
        )
        stack.configure_block_executor(executor, offload_start=1)
        actual = self._run(stack)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_cancellation_is_checked_between_resident_blocks(self) -> None:
        checks = 0

        def cancel_after_first_block() -> None:
            nonlocal checks
            checks += 1
            if checks == 2:
                raise RuntimeError("cancelled")

        with block_cancellation(cancel_after_first_block):
            with self.assertRaisesRegex(RuntimeError, "cancelled"):
                self._run(self._stack())
        self.assertEqual(checks, 2)


if __name__ == "__main__":
    unittest.main()
