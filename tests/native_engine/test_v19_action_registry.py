from __future__ import annotations

from dataclasses import replace
import hashlib
import unittest

from h3serve.native_engine.planner.action_registry import (
    ActionEvidenceBinding,
    ActionKind,
    ActionRegistry,
    ActionRegistryError,
    EvidenceStatus,
    RegisteredAction,
    build_v19_bootstrap_registry,
)
from h3serve.native_engine.planner.joint_global_dp import (
    FIXED_TOPK_ACTION_IMPLEMENTATION,
    ROUND188_HEAD_RAIL_ACTION_IMPLEMENTATION,
    ROUND215_ACTION_IMPLEMENTATION,
    ROUND228_FAST_HEAD_RAIL_ACTION_IMPLEMENTATION,
    ROUND229_FORECAST_AWARE_HEAD_RAIL_ACTION_IMPLEMENTATION,
)
from h3serve.native_engine.planner.v19_contracts import (
    V19HumanRiskVector,
    V19TrajectoryDebt,
    V19_INPUT_CAPABILITY,
)
from h3serve.native_engine.planner.v19_planner import (
    V19ActionUse,
    V19CandidatePlan,
    V19ParetoPlanner,
    V19PlanningError,
    V19PlanningRequest,
    V19WorkloadContext,
    verify_v19_plan_certificate,
)
from h3serve.native_engine.planner.v19_evidence import load_v19_human_evidence


def _implementation_ids() -> dict[str, str]:
    return {
        "fixed_topk": FIXED_TOPK_ACTION_IMPLEMENTATION,
        "round215": ROUND215_ACTION_IMPLEMENTATION,
        "round188": ROUND188_HEAD_RAIL_ACTION_IMPLEMENTATION,
        "round228": ROUND228_FAST_HEAD_RAIL_ACTION_IMPLEMENTATION,
        "round229": ROUND229_FORECAST_AWARE_HEAD_RAIL_ACTION_IMPLEMENTATION,
    }


class V19ActionRegistryTests(unittest.TestCase):
    def test_bootstrap_registers_runtime_implementations_without_false_parity(self) -> None:
        registry = build_v19_bootstrap_registry(
            implementation_ids=_implementation_ids()
        )
        round215 = registry.resolve_implementation(ROUND215_ACTION_IMPLEMENTATION)
        round229 = registry.resolve_implementation(
            ROUND229_FORECAST_AWARE_HEAD_RAIL_ACTION_IMPLEMENTATION
        )
        cache = registry.resolve("h3.cache.coordinate_segment_residual.v1")
        self.assertTrue(round215.planner_eligible)
        self.assertFalse(round229.planner_eligible)
        self.assertEqual(cache.evidence_status, EvidenceStatus.REJECTED)
        self.assertFalse(cache.planner_eligible)

    def test_registry_digest_is_registration_order_independent(self) -> None:
        source = build_v19_bootstrap_registry(
            implementation_ids=_implementation_ids()
        )
        reverse = ActionRegistry(reversed(source.actions))
        self.assertEqual(source.digest, reverse.digest)

    def test_duplicate_physical_implementation_is_rejected(self) -> None:
        action = RegisteredAction(
            action_id="one",
            implementation_id="physical_v1",
            kind=ActionKind.DENSE_ATTENTION,
            executor_id="dense",
            canonical_actions=("dense",),
            exact=True,
            evidence_status=EvidenceStatus.CALIBRATED,
            calibration_ids=("cost_v1",),
            planner_eligible=True,
        )
        with self.assertRaises(ActionRegistryError):
            ActionRegistry((action, replace(action, action_id="two")))

    def test_evidence_requires_exact_registry_and_implementation_identity(self) -> None:
        registry = build_v19_bootstrap_registry(
            implementation_ids=_implementation_ids()
        )
        action = registry.resolve_implementation(ROUND215_ACTION_IMPLEMENTATION)
        binding = ActionEvidenceBinding(
            action_id=action.action_id,
            implementation_id=action.implementation_id,
            registry_digest=registry.digest,
            evidence_id=action.calibration_ids[0],
            evidence_sha256=hashlib.sha256(b"calibration").hexdigest(),
        )
        self.assertEqual(registry.verify_evidence_binding(binding), action)
        with self.assertRaises(ActionRegistryError):
            registry.verify_evidence_binding(
                replace(binding, implementation_id=FIXED_TOPK_ACTION_IMPLEMENTATION)
            )
        with self.assertRaises(ActionRegistryError):
            registry.verify_evidence_binding(
                replace(binding, registry_digest="0" * 64)
            )

    def test_ood_action_filter_does_not_change_input_capability(self) -> None:
        registry = build_v19_bootstrap_registry(
            implementation_ids=_implementation_ids()
        )
        actions = registry.planner_actions_for(
            model_variant="base",
            service_family="reference",
            packed_tokens=100_163,
            condition_count=15,
        )
        self.assertTrue(V19_INPUT_CAPABILITY.accepts(
            service_family="reference",
            model_variant="base",
            reference_images=9,
            reference_audio=3,
            reference_videos=3,
        ))
        self.assertTrue(actions)


class V19QualityContractTests(unittest.TestCase):
    def test_human_risk_dimensions_do_not_compensate_each_other(self) -> None:
        limits = V19HumanRiskVector(
            prompt_adherence=0.1,
            contact_causality=0.1,
            trajectory_continuity=0.1,
            temporal_clarity=0.1,
            identity_binding=0.1,
            audio_integrity=0.1,
            anomaly=0.1,
        )
        failure = V19HumanRiskVector(contact_causality=0.11)
        self.assertFalse(failure.within(limits))

    def test_dense_refresh_does_not_implicitly_zero_trajectory_debt(self) -> None:
        debt = V19TrajectoryDebt(
            consecutive_forecasts=0,
            forecast_debt=1.25,
            sparse_mass_deficit=0.5,
            last_refresh_step=8,
        )
        self.assertGreater(debt.forecast_debt, 0.0)
        self.assertEqual(debt.last_refresh_step, 8)
        self.assertFalse(debt.within(V19TrajectoryDebt(
            consecutive_forecasts=0,
            forecast_debt=1.0,
            sparse_mass_deficit=1.0,
        )))


class V19ParetoPlannerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = build_v19_bootstrap_registry(
            implementation_ids=_implementation_ids()
        )
        # These tests isolate Pareto/certificate mechanics.  Physical
        # calibration enforcement is covered by test_v19_calibration.
        self.planner = V19ParetoPlanner(
            self.registry, require_physical_calibration=False
        )
        self.workload = V19WorkloadContext(
            model_variant="base",
            service_family="first_last",
            packed_tokens=100_163,
            condition_count=2,
        )
        self.request = V19PlanningRequest(
            workload=self.workload,
            maximum_cost_p90_ms=100.0,
            risk_limits=V19HumanRiskVector(
                prompt_adherence=0.1,
                contact_causality=0.1,
                trajectory_continuity=0.1,
                temporal_clarity=0.1,
                identity_binding=0.1,
                audio_integrity=0.1,
                anomaly=0.1,
            ),
        )
        self.dense_use = V19ActionUse(
            action_id="h3.attention.dense.sage_per_warp.sm89.v1",
            canonical_action="dense",
            step_indices=(0, 1),
        )

    def _candidate(
        self,
        name: str,
        *,
        cost: float,
        risk: V19HumanRiskVector = V19HumanRiskVector(),
        peak_vram_gib: float = 0.0,
        terminal_debt: V19TrajectoryDebt = V19TrajectoryDebt(),
        maximum_debt: V19TrajectoryDebt = V19TrajectoryDebt(),
    ) -> V19CandidatePlan:
        return V19CandidatePlan(
            candidate_id=name,
            action_uses=(self.dense_use,),
            predicted_cost_p50_ms=cost - 1.0,
            predicted_cost_p90_ms=cost,
            predicted_peak_vram_gib=peak_vram_gib,
            risk_ucb=risk,
            terminal_debt=terminal_debt,
            maximum_debt=maximum_debt,
        )

    def test_frontier_removes_plan_dominated_in_cost_and_every_risk(self) -> None:
        better = self._candidate("better", cost=80.0)
        worse = self._candidate(
            "worse", cost=90.0, risk=V19HumanRiskVector(temporal_clarity=0.01)
        )
        self.assertEqual(
            self.planner.feasible_frontier(self.request, (worse, better)),
            (better,),
        )

    def test_frontier_prunes_equal_quality_plan_with_more_vram_and_debt(self) -> None:
        better = self._candidate(
            "better_resources",
            cost=80.0,
            peak_vram_gib=8.0,
            terminal_debt=V19TrajectoryDebt(forecast_debt=0.1),
            maximum_debt=V19TrajectoryDebt(
                consecutive_forecasts=1,
                forecast_debt=0.2,
            ),
        )
        worse = self._candidate(
            "worse_resources",
            cost=80.0,
            peak_vram_gib=9.0,
            terminal_debt=V19TrajectoryDebt(forecast_debt=0.2),
            maximum_debt=V19TrajectoryDebt(
                consecutive_forecasts=2,
                forecast_debt=0.3,
            ),
        )
        self.assertEqual(
            self.planner.feasible_frontier(self.request, (worse, better)),
            (better,),
        )

    def test_unbound_forecast_and_rejected_cache_cannot_enter_v19(self) -> None:
        for action_id, canonical in (
            ("h3.forecast.directional.anchor3.round229.v1", "forecast"),
            ("h3.cache.coordinate_segment_residual.v1", "reuse"),
        ):
            candidate = V19CandidatePlan(
                candidate_id=action_id,
                action_uses=(V19ActionUse(
                    action_id=action_id,
                    canonical_action=canonical,
                    step_indices=(1,),
                ),),
                predicted_cost_p50_ms=10.0,
                predicted_cost_p90_ms=11.0,
                risk_ucb=V19HumanRiskVector(),
            )
            with self.assertRaises(V19PlanningError):
                self.planner.certify(self.request, candidate)

    def test_certificate_binds_registry_workload_candidate_and_actions(self) -> None:
        candidate = self._candidate("certified", cost=80.0)
        certificate = self.planner.certify(self.request, candidate)
        self.assertTrue(
            verify_v19_plan_certificate(
                self.registry, self.request, candidate, certificate
            ).valid
        )
        corrupted = replace(certificate, candidate_digest="0" * 64)
        verification = verify_v19_plan_certificate(
            self.registry, self.request, candidate, corrupted
        )
        self.assertFalse(verification.valid)
        self.assertIn("candidate digest mismatch", verification.reasons)


class V19HumanEvidenceTests(unittest.TestCase):
    def test_shared_dense_failure_is_not_an_acceleration_negative(self) -> None:
        source = (
            __import__("pathlib").Path(__file__).resolve().parents[2]
            / "h3serve/native_engine/planner/evidence/human_reviews_v19_seed.json"
        )
        # Public source packages intentionally omit generated MP4s.  The
        # checked-in evidence surface carries their immutable SHA-256 values,
        # so semantic attribution remains verifiable without bundling media.
        evidence = load_v19_human_evidence(source, require_artifacts=False)
        shared = next(
            row for row in evidence.records if row.evidence_id == "H19-SHARED-219-220"
        )
        regression = next(
            row for row in evidence.records if row.evidence_id == "H19-BUDGET-217"
        )
        self.assertEqual(shared.attribution, "shared_failure")
        self.assertFalse(shared.acceleration_negative)
        self.assertEqual(regression.attribution, "candidate_regression")
        self.assertTrue(regression.acceleration_negative)
        self.assertGreater(len(evidence.attributable_negatives), 0)
        self.assertEqual(len(shared.artifact_sha256), 64)
        self.assertEqual(len(shared.comparator_artifact_sha256 or ""), 64)


if __name__ == "__main__":
    unittest.main()
