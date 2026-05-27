from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from renquant_common import (
    AcceptanceReport,
    ArtifactManifest,
    DecisionTraceRow,
    OOSEvidence,
    PooledMetric,
    RegimeLabel,
    RegimeMetric,
    Tier,
)


def _manifest(**overrides) -> ArtifactManifest:
    base = dict(
        kind="panel_ltr_xgboost",
        family="gbdt",
        artifact_uri="file:///tmp/booster.json",
        feature_fingerprint="sha256:abc",
        config_fingerprint="sha256:def",
        training_data_fingerprint="sha256:ghi",
        trained_at=datetime(2026, 5, 20, tzinfo=timezone.utc),
        lookahead_days=5,
        oos_evidence=OOSEvidence(
            mean_ic=0.05,
            std_ic=0.01,
            per_fold_ic=(0.04, 0.05, 0.06),
            cv_method="purged_kfold",
            embargo_days=5,
        ),
        owner_repo="renquant-model",
    )
    base.update(overrides)
    return ArtifactManifest(**base)


def test_manifest_minimal_construction() -> None:
    m = _manifest()
    assert m.kind == "panel_ltr_xgboost"
    assert m.family == "gbdt"
    assert m.calibrator_uri is None


def test_manifest_has_no_promotion_status_field() -> None:
    # Per RFC §"Branch Model": promotion status is derived from branch,
    # not stored on the manifest. Guard against accidental reintroduction.
    assert "promotion_status" not in ArtifactManifest.model_fields


def test_manifest_is_frozen() -> None:
    m = _manifest()
    with pytest.raises(ValidationError):
        m.kind = "patchtst_panel"  # type: ignore[misc]


def test_oos_evidence_embargo_nonnegative() -> None:
    with pytest.raises(ValidationError):
        OOSEvidence(
            mean_ic=0.0,
            std_ic=0.0,
            per_fold_ic=(0.0,),
            cv_method="purged_kfold",
            embargo_days=-1,
        )


def test_manifest_lookahead_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        _manifest(lookahead_days=0)


def test_decision_trace_row_requires_regime_enum() -> None:
    row = DecisionTraceRow(
        run_id="run-1",
        bar_ts=datetime(2026, 5, 27, tzinfo=timezone.utc),
        ticker="AAPL",
        regime=RegimeLabel.BULL_CALM,
        raw_score=0.42,
        calibrated_score=0.05,
        decision="buy",
        gate_history=("regime_ok", "score_above_floor"),
        artifact_fingerprint="sha256:abc",
    )
    assert row.regime is RegimeLabel.BULL_CALM
    # Raw string also coerces via the str-enum base.
    row2 = DecisionTraceRow(
        **{**row.model_dump(), "regime": "BEAR"}
    )
    assert row2.regime is RegimeLabel.BEAR


def test_decision_trace_row_rejects_unknown_regime() -> None:
    with pytest.raises(ValidationError):
        DecisionTraceRow(
            run_id="run-1",
            bar_ts=datetime(2026, 5, 27, tzinfo=timezone.utc),
            ticker="AAPL",
            regime="RANGE_BOUND",  # type: ignore[arg-type]
            decision="buy",
            artifact_fingerprint="sha256:abc",
        )


def test_acceptance_report_full_roundtrip() -> None:
    per_regime = {
        label: RegimeMetric(regime=label, n=20, mean=0.01, std=0.005)
        for label in RegimeLabel
    }
    report = AcceptanceReport(
        candidate=_manifest(),
        baseline=_manifest(feature_fingerprint="sha256:base"),
        n_seeds=5,
        n_windows=16,
        per_regime=per_regime,
        overall=PooledMetric(n=80, mean=0.012, std=0.01, t=2.0, hac_t=1.8),
        dsr=0.72,
        pbo=0.35,
        wilcoxon_p=0.04,
        tier=Tier.LIVE_PROMOTABLE,
        rationale="DSR>0.5 AND per-regime min > 0",
    )
    payload = report.model_dump_json()
    parsed = AcceptanceReport.model_validate_json(payload)
    assert parsed == report
    assert parsed.tier is Tier.LIVE_PROMOTABLE


def test_acceptance_report_probability_bounds() -> None:
    per_regime = {
        label: RegimeMetric(regime=label, n=1, mean=0.0, std=0.0)
        for label in RegimeLabel
    }
    common = dict(
        candidate=_manifest(),
        baseline=_manifest(feature_fingerprint="sha256:base"),
        n_seeds=1,
        n_windows=1,
        per_regime=per_regime,
        overall=PooledMetric(n=5, mean=0.0, std=0.0, t=0.0, hac_t=0.0),
        tier=Tier.REJECT,
        rationale="",
    )
    with pytest.raises(ValidationError):
        AcceptanceReport(**common, dsr=1.5, pbo=0.0, wilcoxon_p=0.0)
    with pytest.raises(ValidationError):
        AcceptanceReport(**common, dsr=0.5, pbo=-0.01, wilcoxon_p=0.0)
