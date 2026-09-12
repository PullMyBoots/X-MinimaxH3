from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest

from h3serve.memory_budget import (
    GIB,
    InMemoryBudgetController,
    LinuxCgroupMemoryBudgetController,
)


class MemoryBudgetTelemetryTests(unittest.TestCase):
    def test_injected_controller_reports_process_scope_and_selected_limit(self) -> None:
        controller = InMemoryBudgetController()
        controller.apply(16)
        usage = controller.usage()
        self.assertEqual(usage["limit_gib"], 16.0)
        self.assertGreaterEqual(usage["used_gib"], 0.0)
        self.assertGreaterEqual(usage["resident_gib"], 0.0)
        self.assertEqual(usage["resident_metric"], "pss")
        self.assertEqual(usage["scope"], "service_process_fallback")
        self.assertFalse(usage["enforced"])

    def test_linux_controller_reports_complete_cgroup_current_and_peak(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            directory = Path(name)
            (directory / "memory.current").write_text(str(7 * GIB))
            (directory / "memory.max").write_text(str(16 * GIB))
            (directory / "memory.peak").write_text(str(11 * GIB))
            (directory / "memory.stat").write_text(
                f"anon {5 * GIB}\nfile {2 * GIB}\n"
            )
            (directory / "memory.events").write_text(
                "low 0\nhigh 3\nmax 8\noom 0\noom_kill 0\n"
            )

            # Bypass cgroup attachment: this unit test exercises only the
            # read-only telemetry contract against kernel-shaped files.
            controller = object.__new__(LinuxCgroupMemoryBudgetController)
            controller.pid = os.getpid()
            controller.directory = directory
            controller.limit_gib = 16.0
            controller._attached = True

            usage = controller.usage()

        self.assertEqual(usage["used_gib"], 7.0)
        self.assertEqual(usage["limit_gib"], 16.0)
        self.assertEqual(usage["peak_gib"], 11.0)
        self.assertEqual(usage["percent"], 43.8)
        self.assertTrue(usage["enforced"])
        self.assertEqual(usage["scope"], "h3_service_cgroup")
        self.assertEqual(usage["anonymous_gib"], 5.0)
        self.assertEqual(usage["file_cache_gib"], 2.0)
        self.assertEqual(usage["oom_count"], 0)
        self.assertEqual(usage["oom_kill_count"], 0)
        self.assertGreaterEqual(usage["resident_gib"], 0.0)
        self.assertGreaterEqual(usage["resident_process_count"], 1)


if __name__ == "__main__":
    unittest.main()
