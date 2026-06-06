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
    TriadReport,
    VerifiedArtifact,
    validate_manifest_for_leakage_triad,
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


def _verified_artifact(
    role: str = "candidate", **overrides
) -> VerifiedArtifact:
    base = dict(
        role=role,
        manifest=_manifest(),
        manifest_fingerprint=f"sha256:{role}",
        verified_at=datetime(2026, 6, 1, tzinfo=timezone.utc),
        checks=("manifest_contract", "embargo_covers_lookahead"),
    )
    base.update(overrides)
    return VerifiedArtifact(**base)


def test_leakage_manifest_validation_accepts_contract_manifest() -> None:
    manifest = _manifest()
    assert validate_manifest_for_leakage_triad(manifest) == manifest


def test_leakage_manifest_validation_requires_sha256_fingerprints() -> None:
    with pytest.raises(ValueError, match="feature_fingerprint"):
        validate_manifest_for_leakage_triad(
            _manifest(feature_fingerprint="blake3:abc")
        )


def test_verified_artifact_requires_embargo_to_cover_lookahead() -> None:
    with pytest.raises(ValidationError, match="embargo_days"):
        _verified_artifact(
            manifest=_manifest(
                lookahead_days=5,
                oos_evidence=OOSEvidence(
                    mean_ic=0.05,
                    std_ic=0.01,
                    per_fold_ic=(0.04, 0.05),
                    cv_method="purged_kfold",
                    embargo_days=4,
                ),
            )
        )


def test_triad_report_roundtrip() -> None:
    report = TriadReport(
        candidate=_verified_artifact("candidate"),
        baseline=_verified_artifact("baseline"),
        shadow=_verified_artifact("shadow"),
        generated_at=datetime(2026, 6, 1, tzinfo=timezone.utc),
        leakage_safe=True,
        rationale="all triad manifests pass the leakage MVP contract",
    )
    parsed = TriadReport.model_validate_json(report.model_dump_json())
    assert parsed == report


def test_triad_report_requires_matching_roles() -> None:
    with pytest.raises(ValidationError, match="baseline artifact"):
        TriadReport(
            candidate=_verified_artifact("candidate"),
            baseline=_verified_artifact("shadow"),
            shadow=_verified_artifact("shadow"),
            generated_at=datetime(2026, 6, 1, tzinfo=timezone.utc),
            leakage_safe=True,
            rationale="invalid roles",
        )


def test_triad_report_requires_aligned_lookahead() -> None:
    with pytest.raises(ValidationError, match="lookahead_days"):
        TriadReport(
            candidate=_verified_artifact("candidate"),
            baseline=_verified_artifact(
                "baseline", manifest=_manifest(lookahead_days=6)
            ),
            shadow=_verified_artifact("shadow"),
            generated_at=datetime(2026, 6, 1, tzinfo=timezone.utc),
            leakage_safe=True,
            rationale="invalid target alignment",
        )


def test_triad_report_cannot_mark_unsafe_artifact_safe() -> None:
    with pytest.raises(ValidationError, match="unsafe artifacts"):
        TriadReport(
            candidate=_verified_artifact("candidate"),
            baseline=_verified_artifact("baseline", leakage_safe=False),
            shadow=_verified_artifact("shadow"),
            generated_at=datetime(2026, 6, 1, tzinfo=timezone.utc),
            leakage_safe=True,
            rationale="invalid safe summary",
        )


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
