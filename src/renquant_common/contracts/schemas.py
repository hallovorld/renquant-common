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

from pydantic import BaseModel, ConfigDict, Field

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
    calibrator_uri: str | None = None
    owner_repo: str


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
    raw_score: float | None = None
    calibrated_score: float | None = None
    decision: str  # "buy" | "sell" | "hold" | "reject" | "skip"
    gate_history: tuple[str, ...] = ()
    artifact_fingerprint: str


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
