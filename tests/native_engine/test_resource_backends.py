from __future__ import annotations

import unittest

from h3serve.contract import LAUNCHER_CONFIGS, public_options
from h3serve.deployment_profiles import (
    LAUNCHER_DEFINITIONS,
    automatic_launcher,
    automatic_vram_profile,
)
from h3serve.models import LAUNCHER_ROLES
from h3serve.native_engine.resource_backends import (
    RESOURCE_BACKENDS,
    get_resource_backend,
)
from h3serve.native_engine.runtime import RuntimeConfig


class ResourceBackendIsolationTests(unittest.TestCase):
    def test_all_public_surfaces_share_the_canonical_launcher_definition(self) -> None:
        options = public_options()
        self.assertEqual(
            set(options["model_launchers"]), set(LAUNCHER_DEFINITIONS)
        )
        self.assertEqual(set(LAUNCHER_ROLES), set(LAUNCHER_DEFINITIONS))
        for launcher_id, definition in LAUNCHER_DEFINITIONS.items():
            with self.subTest(launcher=launcher_id):
                self.assertEqual(
                    LAUNCHER_CONFIGS[launcher_id],
                    (
                        definition.service_family,
                        definition.weight_tier,
                        definition.vram_profile,
                    ),
                )
                advertised = options["model_launchers"][launcher_id]
                self.assertEqual(
                    advertised["second_sampling"],
                    bool(definition.backend.second_sampling_levels),
                )
                self.assertEqual(
                    LAUNCHER_ROLES[launcher_id],
                    definition.required_model_roles,
                )

    def test_five_backends_have_independent_capacity_and_graph_contracts(self) -> None:
        self.assertEqual(
            set(RESOURCE_BACKENDS),
            {
                "int8_24gb", "int8_16gb", "w4a8_24gb",
                "w4a8_16gb", "w4a8_8gb",
            },
        )
        self.assertEqual(
            get_resource_backend("int8_24gb").maximum_first_generation,
            "1080p_15s",
        )
        self.assertEqual(
            get_resource_backend("int8_24gb").second_sampling_levels,
            ("720p", "900p", "1080p", "1220p", "2k"),
        )
        self.assertEqual(
            get_resource_backend("int8_16gb").maximum_first_generation,
            "1080p_15s_experimental",
        )
        self.assertEqual(
            get_resource_backend("int8_16gb").first_generation_levels,
            ("360p", "480p", "540p", "720p", "900p", "1080p"),
        )
        self.assertEqual(
            get_resource_backend("int8_16gb").second_sampling_levels,
            ("720p", "900p", "1080p", "1220p", "2k"),
        )
        compact = get_resource_backend("w4a8_8gb")
        self.assertEqual(compact.maximum_first_generation, "720p_15s")
        self.assertEqual(
            compact.second_sampling_levels, ("720p", "900p", "1080p")
        )
        self.assertEqual(compact.execution_preference, ("compact_streaming",))
        w4_24 = get_resource_backend("w4a8_24gb")
        self.assertEqual(
            w4_24.first_generation_levels,
            ("360p", "480p", "540p", "720p", "900p", "1080p"),
        )
        self.assertEqual(w4_24.maximum_first_generation, "1080p_15s")
        self.assertEqual(w4_24.maximum_short_edge, 1088)

    def test_weight_formats_cannot_cross_resource_backends(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires int8"):
            get_resource_backend("int8_16gb", weight_tier="w4a8")
        with self.assertRaisesRegex(ValueError, "requires w4a8"):
            get_resource_backend("w4a8_8gb", weight_tier="int8")

    def test_physical_vram_is_hidden_behind_automatic_model_route(self) -> None:
        self.assertEqual(automatic_vram_profile("w4a8", 8), "8gb")
        self.assertEqual(automatic_vram_profile("w4a8", 16), "16gb")
        self.assertEqual(automatic_vram_profile("w4a8", 24), "24gb")
        self.assertEqual(
            automatic_launcher("reference", "int8", 24),
            "ref2va_int8_24gb",
        )
        with self.assertRaisesRegex(ValueError, "at least 16GB"):
            automatic_vram_profile("int8", 12)

    def test_runtime_uses_explicit_backend_identity_without_capacity_inference(self) -> None:
        config = RuntimeConfig(
            device="cpu",
            expected_compute_capability=(0, 0),
            max_device_bytes=int(15.25 * 1024**3),
            pin_host_weights=False,
            weight_tier="int8",
            backend_profile="int8_16gb",
        )
        self.assertEqual(config.resource_profile, "int8_16gb")

    def test_backend_rejects_an_allocator_from_a_larger_tier(self) -> None:
        with self.assertRaisesRegex(ValueError, "exceeds"):
            RuntimeConfig(
                max_device_bytes=int(23.25 * 1024**3),
                weight_tier="int8",
                backend_profile="int8_16gb",
            )


if __name__ == "__main__":
    unittest.main()
