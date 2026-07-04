"""Cross-repo typed schemas.

Per RFC §"Branch Model", promoted/candidate/shadow state is derived from
which Git branch a manifest lives on. Therefore no ``promotion_status``
field exists on :class:`ArtifactManifest`; representing it as data would
recreate the drift the branch model removes.

Every schema here uses Pydantic v2 ``frozen=True`` so consumers cannot
mutate shared instances by accident.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Literal, Optional
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .regime import RegimeLabel


class Tier(str, Enum):
    """Acceptance tier per ``doc/research/promotion-methodology.md``."""

    REJECT = "REJECT"
    SCREEN = "SCREEN"
    LIVE_PROMOTABLE = "LIVE_PROMOTABLE"


class OOSEvidence(BaseModel):
    """Out-of-sample evidence summary attached to an :class:`ArtifactManifest`."""

    model_config = ConfigDict(frozen=True)

    mean_ic: float
    std_ic: float
    per_fold_ic: tuple[float, ...]
    cv_method: str
    embargo_days: int = Field(ge=0)


class ArtifactManifest(BaseModel):
    """Cross-repo handshake for a trained-model artifact.

    Producers (renquant-model subdirs) write instances; consumers
    (renquant-pipeline, renquant-backtesting) load them via
    :func:`renquant_common.load_scorer`. The actual model bytes live at
    ``artifact_uri`` (file://, s3://, dvc://); the manifest is the small,
    git-tracked, branch-able pointer.
    """

    model_config = ConfigDict(frozen=True)

    kind: str
    family: str
    artifact_uri: str
    feature_fingerprint: str
    config_fingerprint: str
    training_data_fingerprint: str
    trained_at: datetime
    lookahead_days: int = Field(ge=1)
    oos_evidence: OOSEvidence
    calibrator_uri: Optional[str] = None
    owner_repo: str


def validate_manifest_for_leakage_triad(manifest: ArtifactManifest) -> ArtifactManifest:
    """Validate manifest invariants required by the leakage triad sidecar."""

    for field_name in (
        "feature_fingerprint",
        "config_fingerprint",
        "training_data_fingerprint",
    ):
        value = getattr(manifest, field_name)
        if not value.startswith("sha256:"):
            raise ValueError(f"{field_name} must use a sha256: fingerprint")

    artifact_scheme = urlparse(manifest.artifact_uri).scheme
    if artifact_scheme not in {"dvc", "file", "s3"}:
        raise ValueError("artifact_uri must use file://, s3://, or dvc://")

    if manifest.oos_evidence.embargo_days < manifest.lookahead_days:
        raise ValueError("oos_evidence.embargo_days must cover lookahead_days")

    return manifest


class VerifiedArtifact(BaseModel):
    """Leakage sidecar record for a manifest after contract validation."""

    model_config = ConfigDict(frozen=True)

    role: Literal["candidate", "baseline", "shadow"]
    manifest: ArtifactManifest
    manifest_fingerprint: str
    verified_at: datetime
    checks: tuple[str, ...] = Field(min_length=1)
    leakage_safe: bool = True
    evidence_uri: Optional[str] = None

    @model_validator(mode="after")
    def _validate_manifest_contract(self) -> "VerifiedArtifact":
        validate_manifest_for_leakage_triad(self.manifest)
        if not self.manifest_fingerprint.startswith("sha256:"):
            raise ValueError("manifest_fingerprint must use a sha256: fingerprint")
        return self


class TriadReport(BaseModel):
    """Candidate/baseline/shadow leakage verification summary."""

    model_config = ConfigDict(frozen=True)

    candidate: VerifiedArtifact
    baseline: VerifiedArtifact
    shadow: VerifiedArtifact
    generated_at: datetime
    leakage_safe: bool
    rationale: str

    @model_validator(mode="after")
    def _validate_triad_contract(self) -> "TriadReport":
        expected_roles = {
            "candidate": self.candidate,
            "baseline": self.baseline,
            "shadow": self.shadow,
        }
        for expected, artifact in expected_roles.items():
            if artifact.role != expected:
                raise ValueError(
                    f"{expected} artifact must have role={expected!r}"
                )

        lookaheads = {
            artifact.manifest.lookahead_days
            for artifact in expected_roles.values()
        }
        if len(lookaheads) != 1:
            raise ValueError("triad artifacts must share lookahead_days")

        if self.leakage_safe and any(
            not artifact.leakage_safe for artifact in expected_roles.values()
        ):
            raise ValueError(
                "leakage_safe report cannot include unsafe artifacts"
            )

        return self


class DecisionTraceRow(BaseModel):
    """Per-ticker per-bar forensics row.

    The starter field set covers the cross-repo invariants needed for
    live/sim parity tests. Field-set finalization is the responsibility of
    RFC Open Question #4 ("DecisionTraceRow field set ratification") and
    will land as an additive minor bump.
    """

    model_config = ConfigDict(frozen=True)

    run_id: str
    bar_ts: datetime
    ticker: str
    regime: RegimeLabel
    raw_score: Optional[float] = None
    calibrated_score: Optional[float] = None
    decision: str  # "buy" | "sell" | "hold" | "reject" | "skip"
    gate_history: tuple[str, ...] = ()
    artifact_fingerprint: str


class LiveRunBundle(BaseModel):
    """Readonly live-run bundle for native-vs-bridge offboard parity."""

    model_config = ConfigDict(frozen=True)

    schema_version: int = 1
    source: str
    decision_trace: tuple[dict[str, object], ...]
    order_intents: tuple[dict[str, object], ...]
    state_mutations: tuple[dict[str, object], ...] = ()
    execution_audit: tuple[dict[str, object], ...] = ()
    submitted_orders: tuple[dict[str, object], ...] = ()

    @model_validator(mode="after")
    def _validate_live_bundle_contract(self) -> "LiveRunBundle":
        if self.schema_version != 1:
            raise ValueError("LiveRunBundle schema_version must be 1")
        if not self.source:
            raise ValueError("LiveRunBundle source is required")
        if not (
            self.state_mutations
            or self.execution_audit
            or self.submitted_orders
        ):
            raise ValueError(
                "LiveRunBundle requires at least one state source: "
                "state_mutations, execution_audit, or submitted_orders"
            )
        return self


def validate_live_run_bundle(bundle: LiveRunBundle | dict[str, object]) -> LiveRunBundle:
    """Validate and return a :class:`LiveRunBundle` instance."""
    if isinstance(bundle, LiveRunBundle):
        return bundle
    return LiveRunBundle.model_validate(bundle)


class RegimeMetric(BaseModel):
    """Aggregate of a metric over a single regime's bars."""

    model_config = ConfigDict(frozen=True)

    regime: RegimeLabel
    n: int = Field(ge=0)
    mean: float
    std: float


class PooledMetric(BaseModel):
    """Pooled (cross-regime) aggregate for diagnostic comparison.

    Per PRIME DIRECTIVE, pooled aggregates are diagnostic only — the
    promotion decision uses per-regime evidence (see
    :class:`AcceptanceReport.per_regime`).
    """

    model_config = ConfigDict(frozen=True)

    n: int = Field(ge=0)
    mean: float
    std: float
    t: float
    hac_t: float


class AcceptanceReport(BaseModel):
    """Output of renquant-backtesting promotion gating.

    Consumed by renquant-orchestrator to decide whether to merge a
    candidate manifest into renquant-artifacts/main per RFC §"Branch
    Model" promotion workflow.
    """

    model_config = ConfigDict(frozen=True)

    candidate: ArtifactManifest
    baseline: ArtifactManifest
    n_seeds: int = Field(ge=1)
    n_windows: int = Field(ge=1)
    per_regime: dict[RegimeLabel, RegimeMetric]
    overall: PooledMetric
    dsr: float = Field(ge=0.0, le=1.0)
    pbo: float = Field(ge=0.0, le=1.0)
    wilcoxon_p: float = Field(ge=0.0, le=1.0)
    tier: Tier
    rationale: str
