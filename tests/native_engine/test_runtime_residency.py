from __future__ import annotations

import unittest

import torch
import torch.nn as nn

from h3serve.native_engine.runtime import ImmutablePinnedModuleResidency
from h3serve.native_engine.runtime.pinned_pool import pack_pinned_tensors
from h3serve.native_engine.hot_session import _is_cuda_context_fatal


class _SharedWeightModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        shared = nn.Parameter(torch.arange(12, dtype=torch.float32).reshape(3, 4))
        self.weight = shared
        self.alias = shared
        self.register_buffer("scale", torch.tensor([2.0]))


class _TwoBlockModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([
            nn.Linear(512, 512, bias=False),
            nn.Linear(512, 512, bias=False),
        ])


class ImmutablePinnedModuleResidencyTest(unittest.TestCase):
    def test_device_not_ready_is_a_fatal_cuda_context_error(self) -> None:
        self.assertTrue(
            _is_cuda_context_fatal(
                RuntimeError("CUDA driver error: device not ready")
            )
        )

    def test_cpu_round_trip_preserves_values_and_aliases(self) -> None:
        module = _SharedWeightModule()
        residency = ImmutablePinnedModuleResidency(
            "tiny", module, pin_host_weights=False
        )
        residency.prepare_host()

        self.assertTrue(residency.prepared)
        self.assertEqual(residency.host_bytes, 52)
        self.assertIs(module.weight, module.alias)
        torch.testing.assert_close(
            module.weight,
            torch.arange(12, dtype=torch.float32).reshape(3, 4),
        )
        residency.move_to("cpu", non_blocking=True)
        self.assertIs(module.weight, module.alias)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_cuda_evict_rebinds_original_host_storage_without_d2h(self) -> None:
        module = _SharedWeightModule()
        residency = ImmutablePinnedModuleResidency(
            "tiny", module, pin_host_weights=True
        )
        residency.prepare_host()
        host_pointer = module.weight.data_ptr()
        self.assertTrue(residency.host_is_pinned)

        residency.move_to("cuda:0", non_blocking=True)
        torch.cuda.synchronize()
        self.assertEqual(module.weight.device.type, "cuda")
        self.assertIs(module.weight, module.alias)

        # Mutating the disposable device copy must not mutate the immutable
        # authoritative host master used by inference phase transitions.
        module.weight.data.zero_()
        residency.move_to("cpu", non_blocking=False)
        self.assertEqual(module.weight.device.type, "cpu")
        self.assertEqual(module.weight.data_ptr(), host_pointer)
        self.assertGreater(float(module.weight.detach().sum()), 0.0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_pinned_pool_preserves_values_strides_and_async_copy(self) -> None:
        contiguous = torch.arange(60, dtype=torch.float32).reshape(5, 12)
        strided = torch.arange(48, dtype=torch.float16).reshape(6, 8).t()
        packed = pack_pinned_tensors((contiguous, strided), slab_bytes=4096)

        self.assertEqual(len(packed.slabs), 2)  # one slab per dtype
        self.assertTrue(all(tensor.is_pinned() for tensor in packed.tensors))
        self.assertEqual(packed.tensors[1].stride(), strided.stride())
        torch.testing.assert_close(packed.tensors[0], contiguous)
        torch.testing.assert_close(packed.tensors[1], strided)
        copied = packed.tensors[0].to("cuda:0", non_blocking=True)
        torch.cuda.synchronize()
        torch.testing.assert_close(copied.cpu(), contiguous)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_pinned_preparation_reports_reclaimable_source_bytes(self) -> None:
        module = _SharedWeightModule()
        copied: list[int] = []
        residency = ImmutablePinnedModuleResidency(
            "tiny",
            module,
            pin_host_weights=True,
            copy_host_weights=True,
            source_copied=copied.append,
        )
        residency.prepare_host()
        self.assertTrue(copied)
        self.assertEqual(sum(copied), residency.host_bytes)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_in_place_pin_keeps_original_storage_without_second_master(self) -> None:
        module = _SharedWeightModule()
        original_pointer = module.weight.data_ptr()
        residency = ImmutablePinnedModuleResidency(
            "tiny",
            module,
            pin_host_weights=True,
            copy_host_weights=False,
        )
        residency.prepare_host()
        self.assertEqual(module.weight.data_ptr(), original_pointer)
        self.assertTrue(residency.host_is_pinned)
        self.assertGreaterEqual(residency.host_allocated_bytes, residency.host_bytes)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_repeated_partition_move_reuses_identical_device_storage(self) -> None:
        module = _TwoBlockModule()
        residency = ImmutablePinnedModuleResidency(
            "partitioned", module, pin_host_weights=False
        )
        residency.prepare_host()
        prefixes = ("blocks.1",)

        residency.move_partition_to_cuda(
            "cuda:0", host_module_prefixes=prefixes
        )
        torch.cuda.synchronize()
        first_pointer = module.blocks[0].weight.data_ptr()
        self.assertEqual(module.blocks[0].weight.device.type, "cuda")
        self.assertEqual(module.blocks[1].weight.device.type, "cpu")

        residency.move_partition_to_cuda(
            "cuda:0", host_module_prefixes=prefixes
        )
        torch.cuda.synchronize()
        self.assertEqual(module.blocks[0].weight.data_ptr(), first_pointer)
        residency.move_to("cpu", non_blocking=False)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_partial_pin_budget_registers_complete_module_prefix(self) -> None:
        module = _TwoBlockModule()
        one_block_bytes = module.blocks[0].weight.numel() * 4
        residency = ImmutablePinnedModuleResidency(
            "partial",
            module,
            pin_host_weights=True,
            copy_host_weights=False,
            pin_host_budget_bytes=one_block_bytes,
            pin_host_module_prefixes=("blocks.0", "blocks.1"),
        )
        residency.prepare_host()

        self.assertFalse(residency.host_is_pinned)
        self.assertEqual(residency.host_pinned_bytes, one_block_bytes)
        self.assertAlmostEqual(residency.host_pinned_fraction, 0.5)


if __name__ == "__main__":
    unittest.main()
