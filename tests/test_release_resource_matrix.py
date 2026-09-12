from __future__ import annotations

from argparse import Namespace
from pathlib import Path
import tempfile
import unittest

from scripts.validate_release_resource_matrix import (
    completed_row_ids,
    finalize_report,
    row_ids,
    selected_cases,
)


class ReleaseResourceMatrixReportTests(unittest.TestCase):
    def test_only_recheck_cannot_shrink_complete_matrix_contract(self) -> None:
        args = Namespace(
            ram_mode="minimum",
            only=("w4a8_v8_ram12",),
        )
        selected = {
            "w4a8_v8_ram12_480p",
            "w4a8_v8_ram12_720p",
        }
        report = {
            "rows": [
                {
                    "row_id": row_id,
                    "status": "passed",
                    "job": {"status": "succeeded"},
                }
                for row_id in selected
            ]
        }

        selected_passed, matrix_passed = finalize_report(report, args)

        self.assertTrue(selected_passed)
        self.assertFalse(matrix_passed)
        self.assertEqual(set(report["selected_rows"]), selected)
        self.assertEqual(
            set(report["expected_rows"]), row_ids(selected_cases("both"))
        )
        self.assertFalse(report["passed"])

        receipt = {
            "rows": [{
                "row_id": "one",
                "status": "passed",
                "job": {"status": "succeeded"},
                "download_bytes": 4,
            }]
        }
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            self.assertEqual(completed_row_ids(receipt, output_dir), set())
            (output_dir / "one.mp4").write_bytes(b"bad")
            self.assertEqual(completed_row_ids(receipt, output_dir), set())
            (output_dir / "one.mp4").write_bytes(b"good")
            self.assertEqual(completed_row_ids(receipt, output_dir), {"one"})

    def test_complete_merged_report_is_sealed_as_passed(self) -> None:
        args = Namespace(ram_mode="minimum", only=())
        expected = row_ids(selected_cases("both"))
        report = {
            "rows": [
                {
                    "row_id": row_id,
                    "status": "passed",
                    "job": {"status": "succeeded"},
                }
                for row_id in expected
            ]
        }

        selected_passed, matrix_passed = finalize_report(report, args)

        self.assertTrue(selected_passed)
        self.assertTrue(matrix_passed)
        self.assertTrue(report["matrix_passed"])
        self.assertTrue(report["passed"])

if __name__ == "__main__":
    unittest.main()
