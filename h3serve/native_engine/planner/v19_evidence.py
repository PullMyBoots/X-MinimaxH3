"""Human mechanism evidence for V19 risk calibration.

The key distinction is attribution.  A candidate failure is acceleration
evidence only when a matched comparator does not share that failure.  Shared
Dense/candidate failures stay in the dataset but cannot be used to penalise a
sparse action.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Literal


V19_HUMAN_EVIDENCE_SCHEMA = "h3_v19_human_mechanism_evidence_v1"
ReviewLabel = Literal["accept", "reject", "not_reported"]
Attribution = Literal[
    "candidate_positive",
    "candidate_regression",
    "shared_failure",
    "unattributed",
]


class V19EvidenceError(ValueError):
    pass


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class V19DimensionLabels:
    prompt_adherence: ReviewLabel = "not_reported"
    contact_causality: ReviewLabel = "not_reported"
    trajectory_continuity: ReviewLabel = "not_reported"
    temporal_clarity: ReviewLabel = "not_reported"
    identity_binding: ReviewLabel = "not_reported"
    audio_integrity: ReviewLabel = "not_reported"
    anomaly: ReviewLabel = "not_reported"

    def __post_init__(self) -> None:
        allowed = {"accept", "reject", "not_reported"}
        if any(value not in allowed for value in self.as_tuple()):
            raise V19EvidenceError("invalid Human dimension label")

    def as_tuple(self) -> tuple[ReviewLabel, ...]:
        return (
            self.prompt_adherence,
            self.contact_causality,
            self.trajectory_continuity,
            self.temporal_clarity,
            self.identity_binding,
            self.audio_integrity,
            self.anomaly,
        )


@dataclass(frozen=True, slots=True)
class V19HumanEvidenceRecord:
    evidence_id: str
    mechanism: str
    candidate_id: str
    comparator_id: str
    same_prompt_seed_shape_steps: bool
    candidate_outcome: Literal["accept", "reject"]
    comparator_outcome: ReviewLabel
    dimensions: V19DimensionLabels
    artifact: str
    artifact_sha256: str
    comparator_artifact: str | None = None
    comparator_artifact_sha256: str | None = None
    note: str = ""

    def __post_init__(self) -> None:
        if not all((self.evidence_id, self.mechanism, self.candidate_id, self.comparator_id)):
            raise V19EvidenceError("Human evidence requires stable identities")
        if self.candidate_outcome not in ("accept", "reject"):
            raise V19EvidenceError("invalid candidate outcome")
        if self.comparator_outcome not in ("accept", "reject", "not_reported"):
            raise V19EvidenceError("invalid comparator outcome")
        if not self.artifact:
            raise V19EvidenceError("Human evidence requires a candidate artifact")
        for label, value in (
            ("candidate", self.artifact_sha256),
            ("comparator", self.comparator_artifact_sha256),
        ):
            if value is None:
                continue
            if len(value) != 64:
                raise V19EvidenceError(
                    f"Human evidence {label} artifact digest is not SHA256"
                )
            try:
                int(value, 16)
            except ValueError as error:
                raise V19EvidenceError(
                    f"Human evidence {label} artifact digest is not hexadecimal"
                ) from error
        if (self.comparator_artifact is None) != (
            self.comparator_artifact_sha256 is None
        ):
            raise V19EvidenceError(
                "Human evidence comparator path/digest must be present together"
            )

    @property
    def attribution(self) -> Attribution:
        if self.candidate_outcome == "accept":
            return "candidate_positive"
        if not self.same_prompt_seed_shape_steps or self.comparator_outcome == "not_reported":
            return "unattributed"
        if self.comparator_outcome == "reject":
            return "shared_failure"
        return "candidate_regression"

    @property
    def acceleration_negative(self) -> bool:
        return self.attribution == "candidate_regression"


@dataclass(frozen=True, slots=True)
class V19HumanEvidenceSet:
    source: Path
    records: tuple[V19HumanEvidenceRecord, ...]
    sha256: str

    @property
    def attributable_negatives(self) -> tuple[V19HumanEvidenceRecord, ...]:
        return tuple(row for row in self.records if row.acceleration_negative)

    @property
    def shared_failures(self) -> tuple[V19HumanEvidenceRecord, ...]:
        return tuple(row for row in self.records if row.attribution == "shared_failure")


def load_v19_human_evidence(
    path: str | Path,
    *,
    serve_root: str | Path | None = None,
    require_artifacts: bool = True,
) -> V19HumanEvidenceSet:
    source = Path(path).resolve()
    try:
        raw = source.read_bytes()
        document = json.loads(raw)
    except (OSError, json.JSONDecodeError) as error:
        raise V19EvidenceError(f"invalid V19 Human evidence: {source}") from error
    if document.get("schema_version") != V19_HUMAN_EVIDENCE_SCHEMA:
        raise V19EvidenceError("unexpected V19 Human evidence schema")
    rows = document.get("records")
    if not isinstance(rows, list):
        raise V19EvidenceError("V19 Human evidence records must be a list")
    records: list[V19HumanEvidenceRecord] = []
    ids: set[str] = set()
    root = Path(serve_root).resolve() if serve_root is not None else source.parents[4]
    for row in rows:
        try:
            dimensions = V19DimensionLabels(**row["dimensions"])
            artifact_digests: dict[str, str | None] = {
                "candidate": None,
                "comparator": None,
            }
            for label, relative in (
                ("candidate", row["artifact"]),
                ("comparator", row.get("comparator_artifact")),
            ):
                if relative is None:
                    continue
                if require_artifacts:
                    artifact = (root / relative).resolve()
                    try:
                        artifact.relative_to(root)
                    except ValueError as error:
                        raise V19EvidenceError(
                            f"{row['evidence_id']}: {label} artifact escapes serve root"
                        ) from error
                    if not artifact.is_file():
                        raise V19EvidenceError(
                            f"{row['evidence_id']}: missing {label} artifact: {relative}"
                        )
                    artifact_digests[label] = _sha256_file(artifact)
                else:
                    digest_key = (
                        "artifact_sha256"
                        if label == "candidate"
                        else "comparator_artifact_sha256"
                    )
                    digest = row.get(digest_key)
                    if not isinstance(digest, str) or not re.fullmatch(
                        r"[0-9a-f]{64}", digest
                    ):
                        raise V19EvidenceError(
                            f"{row['evidence_id']}: missing portable {label} digest"
                        )
                    artifact_digests[label] = digest
            record = V19HumanEvidenceRecord(
                evidence_id=row["evidence_id"],
                mechanism=row["mechanism"],
                candidate_id=row["candidate_id"],
                comparator_id=row["comparator_id"],
                same_prompt_seed_shape_steps=bool(row["same_prompt_seed_shape_steps"]),
                candidate_outcome=row["candidate_outcome"],
                comparator_outcome=row["comparator_outcome"],
                dimensions=dimensions,
                artifact=row["artifact"],
                artifact_sha256=str(artifact_digests["candidate"] or ""),
                comparator_artifact=row.get("comparator_artifact"),
                comparator_artifact_sha256=(
                    None
                    if artifact_digests["comparator"] is None
                    else str(artifact_digests["comparator"])
                ),
                note=row.get("note", ""),
            )
        except (KeyError, TypeError) as error:
            raise V19EvidenceError("malformed V19 Human evidence record") from error
        if record.evidence_id in ids:
            raise V19EvidenceError(f"duplicate Human evidence id: {record.evidence_id}")
        ids.add(record.evidence_id)
        records.append(record)
    return V19HumanEvidenceSet(
        source=source,
        records=tuple(records),
        sha256=hashlib.sha256(raw).hexdigest(),
    )


__all__ = [
    "V19_HUMAN_EVIDENCE_SCHEMA",
    "V19DimensionLabels",
    "V19EvidenceError",
    "V19HumanEvidenceRecord",
    "V19HumanEvidenceSet",
    "load_v19_human_evidence",
]
