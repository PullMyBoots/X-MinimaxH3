from __future__ import annotations

import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

from h3serve.memory_policy import (
    HOST_MEMORY_PROFILES,
    HostMemoryStatus,
    _current_cgroup_memory_files,
    host_memory_budget_bounds,
    resolve_host_memory_budget_profile,
    resolve_host_memory_profile,
    validate_profile_for_weight_tier,
    validate_workload_for_profile,
)


class HostMemoryPolicyTests(unittest.TestCase):
    def test_internal_auto_profile_has_no_full_power_product_label(self) -> None:
        self.assertEqual(HOST_MEMORY_PROFILES["fullspeed"].label, "自动内存配置")

    def status(self, total: float) -> HostMemoryStatus:
        return HostMemoryStatus(total, total, total)

    def test_auto_uses_only_meaningful_strategy_boundaries(self) -> None:
        self.assertEqual(resolve_host_memory_profile("auto", self.status(140)).key, "fullspeed")
        # A 128GB Windows host commonly exposes about 110GiB inside WSL.
        self.assertEqual(resolve_host_memory_profile("auto", self.status(110)).key, "fullspeed")
        self.assertEqual(resolve_host_memory_profile("auto", self.status(90)).key, "generation_hot")
        self.assertEqual(resolve_host_memory_profile("auto", self.status(70)).key, "compact")
        self.assertEqual(resolve_host_memory_profile("auto", self.status(48)).key, "w4a8_32gb")
        self.assertEqual(resolve_host_memory_profile("auto", self.status(30)).key, "w4a8_32gb")
        self.assertEqual(resolve_host_memory_profile("auto", self.status(15)).key, "w4a8_16gb")
        with self.assertRaisesRegex(RuntimeError, "validated 16 GiB-class minimum"):
            resolve_host_memory_profile("auto", self.status(11))

    def test_product_budget_uses_weight_specific_floor_and_measured_knee(self) -> None:
        status = self.status(110)
        self.assertEqual(host_memory_budget_bounds("w4a8", status), (12, 104))
        self.assertEqual(host_memory_budget_bounds("int8", status), (24, 104))
        int8 = resolve_host_memory_budget_profile(
            "int8", 24, vram_profile="24gb", status=status
        )
        self.assertEqual(int8.process_limit_gib, 24)
        self.assertAlmostEqual(int8.pin_transformer_budget_gib, 11.67)
        self.assertEqual(int8.evidence, "experimental_low_memory")
        w4 = resolve_host_memory_budget_profile(
            "w4a8", 12, vram_profile="24gb", status=status
        )
        self.assertEqual(w4.resident_transformer_blocks, 49)
        self.assertLessEqual(w4.pin_transformer_budget_gib, 0.219)
        self.assertEqual(w4.evidence, "experimental_low_memory")

    def test_w4a8_budget_compiles_to_each_vram_knee_without_overallocating(self) -> None:
        status = self.status(110)
        v8_min = resolve_host_memory_budget_profile(
            "w4a8", 16, vram_profile="8gb", status=status
        )
        v8_knee = resolve_host_memory_budget_profile(
            "w4a8", 22, vram_profile="8gb", status=status
        )
        v8_excess = resolve_host_memory_budget_profile(
            "w4a8", 40, vram_profile="8gb", status=status
        )
        self.assertLess(
            v8_min.pin_transformer_budget_gib,
            v8_knee.pin_transformer_budget_gib,
        )
        self.assertEqual(
            v8_knee.pin_transformer_budget_gib,
            v8_excess.pin_transformer_budget_gib,
        )

        v16_min = resolve_host_memory_budget_profile(
            "w4a8", 16, vram_profile="16gb", status=status
        )
        v16_knee = resolve_host_memory_budget_profile(
            "w4a8", 17, vram_profile="16gb", status=status
        )
        v16_excess = resolve_host_memory_budget_profile(
            "w4a8", 40, vram_profile="16gb", status=status
        )
        self.assertEqual(v16_min.resident_transformer_blocks, 30)
        self.assertEqual(v16_min.pin_transformer_budget_gib, 0.0)
        self.assertEqual(v16_knee.resident_transformer_blocks, 24)
        self.assertEqual(
            v16_knee.pin_transformer_budget_gib,
            v16_excess.pin_transformer_budget_gib,
        )

        v24_min = resolve_host_memory_budget_profile(
            "w4a8", 16, vram_profile="24gb", status=status
        )
        v24_excess = resolve_host_memory_budget_profile(
            "w4a8", 40, vram_profile="24gb", status=status
        )
        self.assertEqual(v24_min.resident_transformer_blocks, 49)
        self.assertEqual(v24_min.pin_transformer_budget_gib, 0.219)
        self.assertEqual(
            v24_min.pin_transformer_budget_gib,
            v24_excess.pin_transformer_budget_gib,
        )

    def test_product_budget_rejects_values_outside_process_contract(self) -> None:
        status = self.status(40)
        with self.assertRaisesRegex(ValueError, "between 24 and 34"):
            resolve_host_memory_budget_profile(
                "int8", 23, vram_profile="16gb", status=status
            )
        with self.assertRaisesRegex(ValueError, "whole GiB"):
            resolve_host_memory_budget_profile(
                "w4a8", 20.5, vram_profile="8gb", status=status
            )

    def test_explicit_profile_checks_effective_wsl_limit(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "effective limit"):
            resolve_host_memory_profile("generation_hot", self.status(48))

    def test_32gb_w4a8_keeps_six_gib_host_reserve(self) -> None:
        profile = HOST_MEMORY_PROFILES["w4a8_32gb"]
        self.assertEqual(
            resolve_host_memory_profile(
                profile.key,
                HostMemoryStatus(30, 30, 25),
            ).key,
            profile.key,
        )
        with self.assertRaisesRegex(RuntimeError, "enough free RAM"):
            resolve_host_memory_profile(
                profile.key,
                HostMemoryStatus(30, 30, 24.9),
            )

    def test_auto_downgrades_when_other_processes_consume_ram(self) -> None:
        busy = HostMemoryStatus(140, 140, 50)
        self.assertEqual(resolve_host_memory_profile("auto", busy).key, "compact")

    def test_profiles_fail_close_above_native_generation_envelopes(self) -> None:
        compact = HOST_MEMORY_PROFILES["compact"]
        validate_workload_for_profile(compact, width=864, height=480, frames=362)
        validate_workload_for_profile(compact, width=1280, height=736, frames=362)
        validate_workload_for_profile(compact, width=1920, height=1088, frames=362)
        validate_workload_for_profile(compact, width=1440, height=1088, frames=362)
        validate_workload_for_profile(compact, width=1088, height=1088, frames=362)
        with self.assertRaisesRegex(ValueError, "spatial-temporal"):
            validate_workload_for_profile(compact, width=1920, height=1088, frames=379)
        with self.assertRaisesRegex(ValueError, "spatial-temporal"):
            validate_workload_for_profile(compact, width=2560, height=1472, frames=192)
        validate_workload_for_profile(
            HOST_MEMORY_PROFILES["generation_hot"],
            width=1280, height=736, frames=362,
        )

    def test_w4a8_low_ram_tiers_are_release_validated(self) -> None:
        self.assertEqual(
            {profile.minimum_ram_gib for profile in HOST_MEMORY_PROFILES.values()},
            {16, 32, 64, 96, 128},
        )
        self.assertEqual(HOST_MEMORY_PROFILES["w4a8_32gb"].evidence, "validated")
        self.assertEqual(HOST_MEMORY_PROFILES["w4a8_16gb"].evidence, "validated")
        self.assertFalse(HOST_MEMORY_PROFILES["w4a8_32gb"].pin_model_weights)
        self.assertTrue(HOST_MEMORY_PROFILES["w4a8_32gb"].pin_transformer_weights)
        self.assertFalse(HOST_MEMORY_PROFILES["w4a8_32gb"].copy_transformer_weights)
        self.assertFalse(HOST_MEMORY_PROFILES["w4a8_16gb"].copy_model_weights)
        self.assertFalse(HOST_MEMORY_PROFILES["w4a8_16gb"].pin_transformer_weights)

    def test_low_ram_profiles_are_isolated_to_w4a8_weights(self) -> None:
        for key in ("w4a8_16gb", "w4a8_32gb"):
            profile = HOST_MEMORY_PROFILES[key]
            validate_profile_for_weight_tier(profile, "w4a8")
            with self.assertRaisesRegex(ValueError, "only for the W4A8"):
                validate_profile_for_weight_tier(profile, "int8")
        validate_profile_for_weight_tier(
            HOST_MEMORY_PROFILES["compact"],
            "int8",
        )

    def test_only_128gb_profile_allows_h3_and_upscaler_to_overlap(self) -> None:
        self.assertFalse(HOST_MEMORY_PROFILES["fullspeed"].exclusive_upscaler)
        self.assertTrue(HOST_MEMORY_PROFILES["generation_hot"].exclusive_upscaler)
        self.assertTrue(HOST_MEMORY_PROFILES["compact"].exclusive_upscaler)

    def test_current_cgroup_membership_is_checked_before_root(self) -> None:
        paths = _current_cgroup_memory_files()
        self.assertTrue(paths)
        # The active process path must be represented on cgroup-v2 systems;
        # checking only /sys/fs/cgroup/memory.max misses systemd/container caps.
        membership = Path("/proc/self/cgroup").read_text(encoding="utf-8")
        if "0::" in membership:
            relative = membership.split("0::", 1)[1].splitlines()[0].lstrip("/")
            expected = Path("/sys/fs/cgroup") / relative / "memory.max"
            self.assertEqual(paths[0][0], expected)

    def test_detected_capacity_honors_leaf_cgroup_limit_and_usage(self) -> None:
        from h3serve.memory_policy import GIB, detect_host_memory

        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            limit = root / "memory.max"
            usage = root / "memory.current"
            limit.write_text(str(58 * GIB))
            usage.write_text(str(10 * GIB))
            with patch(
                "h3serve.memory_policy._meminfo",
                return_value={"MemTotal": 128 * GIB, "MemAvailable": 100 * GIB},
            ), patch(
                "h3serve.memory_policy._current_cgroup_memory_files",
                return_value=((limit, usage),),
            ):
                status = detect_host_memory()
        self.assertEqual(status.physical_total_gib, 128)
        self.assertEqual(status.effective_limit_gib, 58)
        self.assertEqual(status.available_gib, 48)
        self.assertEqual(resolve_host_memory_profile("auto", status).key, "compact")


if __name__ == "__main__":
    unittest.main()
