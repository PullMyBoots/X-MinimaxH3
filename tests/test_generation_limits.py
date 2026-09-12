from __future__ import annotations

import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

from h3serve.contract import ContractError, GenerationSpec, public_options
from h3serve.generation_limits import (
    GenerationLimitPolicy,
    default_preset_limits,
    load_generation_limit_policy,
    persist_generation_limit_policy,
)


class GenerationLimitPolicyTest(unittest.TestCase):
    def test_each_resolution_and_ratio_is_independently_configurable(self) -> None:
        limits = default_preset_limits()
        limits["720p"]["1:1"] = 9
        limits["720p"]["16:9"] = 12
        limits["1080p"]["16:9"] = 15
        policy = GenerationLimitPolicy(limits)
        self.assertEqual(policy.preset_limits["720p"]["1:1"], 9)
        self.assertEqual(policy.preset_limits["720p"]["16:9"], 12)
        self.assertEqual(policy.preset_limits["1080p"]["16:9"], 15)
        self.assertEqual(policy.public(32)["detected_vram_gib"], 32)

    def test_explicit_matrix_drives_options_and_submission_validation(self) -> None:
        limits = default_preset_limits()
        limits["1080p"]["16:9"] = 10
        options = public_options(max_duration_by_preset=limits)
        self.assertEqual(options["duration"]["max_by_preset"]["1080p"]["16:9"], 10)
        GenerationSpec.from_mapping({
            "prompt": "accepted by operator limit",
            "resolution": "1080p",
            "aspect_ratio": "16:9",
            "duration_seconds": 10,
        }, max_duration_by_preset=limits)
        with self.assertRaisesRegex(ContractError, "configured server limit"):
            GenerationSpec.from_mapping({
                "prompt": "too long for operator limit",
                "resolution": "1080p",
                "aspect_ratio": "16:9",
                "duration_seconds": 10.5,
            }, max_duration_by_preset=limits)

    def test_policy_requires_a_complete_valid_matrix(self) -> None:
        incomplete = deepcopy(default_preset_limits())
        del incomplete["720p"]["1:1"]
        with self.assertRaisesRegex(ValueError, "720p.1:1"):
            GenerationLimitPolicy(incomplete)
        invalid = default_preset_limits()
        invalid["480p"]["4:3"] = 16
        with self.assertRaisesRegex(ValueError, "between 1 and 15"):
            GenerationLimitPolicy(invalid)

    def test_policy_persists_and_old_intermediate_format_migrates(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            data_dir = Path(temporary)
            limits = default_preset_limits()
            limits["1080p"]["16:9"] = 12
            policy = GenerationLimitPolicy(limits)
            persist_generation_limit_policy(data_dir, policy)
            self.assertEqual(load_generation_limit_policy(data_dir), policy)
            document = json.loads(
                (data_dir / "settings/generation_limits.json").read_text("utf-8")
            )
            self.assertEqual(document["preset_limits"]["1080p"]["16:9"], 12)

            (data_dir / "settings/generation_limits.json").write_text(
                '{"mode":"manual","manual_vram_gib":32}', encoding="utf-8"
            )
            migrated = load_generation_limit_policy(data_dir)
            self.assertEqual(migrated, GenerationLimitPolicy())

    def test_legacy_matrix_preserves_operator_values_and_seeds_new_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            data_dir = Path(temporary)
            limits = default_preset_limits()
            for resolution in ("540p", "900p", "2k"):
                del limits[resolution]
            limits["1080p"]["16:9"] = 10
            path = data_dir / "settings/generation_limits.json"
            path.parent.mkdir(parents=True)
            path.write_text(
                json.dumps({"preset_limits": limits}), encoding="utf-8"
            )

            migrated = load_generation_limit_policy(data_dir)
            self.assertEqual(migrated.preset_limits["1080p"]["16:9"], 10)
            self.assertEqual(migrated.preset_limits["540p"]["16:9"], 15)
            self.assertEqual(migrated.preset_limits["900p"]["16:9"], 15)
            self.assertEqual(migrated.preset_limits["2k"]["16:9"], 15)


if __name__ == "__main__":
    unittest.main()
