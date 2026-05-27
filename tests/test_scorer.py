from __future__ import annotations

from datetime import datetime, timezone
from importlib.metadata import EntryPoint
from unittest.mock import patch

import pytest

from renquant_common import (
    ArtifactManifest,
    OOSEvidence,
    Scorer,
    ScorerKindNotRegistered,
    load_scorer,
)
from renquant_common.contracts.scorer import SCORER_ENTRY_POINT_GROUP


class _FakeScorer:
    """Minimal Scorer impl used as the entry-point target."""

    def __init__(self, feature_cols: list[str], fingerprint: str) -> None:
        self.feature_cols = feature_cols
        self._fingerprint = fingerprint

    def feature_fingerprint(self) -> str:
        return self._fingerprint

    def predict_rows(self, rows):
        return {ticker: 0.0 for ticker in rows}

    def predict_variance(self, rows):
        return None


def _fake_loader(manifest: ArtifactManifest) -> _FakeScorer:
    return _FakeScorer(
        feature_cols=["a", "b"], fingerprint=manifest.feature_fingerprint
    )


def _broken_loader(manifest: ArtifactManifest) -> object:
    return object()  # does NOT satisfy the Scorer Protocol


def _manifest(kind: str) -> ArtifactManifest:
    return ArtifactManifest(
        kind=kind,
        family="gbdt",
        artifact_uri="file:///tmp/x.json",
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


def _stub_ep(name: str, loader_callable) -> EntryPoint:
    # Constructed entry-point bypasses package metadata; we patch the
    # loader returned by `.load()` directly.
    ep = EntryPoint(name=name, value="dummy:dummy", group=SCORER_ENTRY_POINT_GROUP)
    ep.load = lambda: loader_callable  # type: ignore[assignment]
    return ep


def test_fake_scorer_satisfies_protocol_runtime_check() -> None:
    fake = _FakeScorer(feature_cols=["a"], fingerprint="x")
    assert isinstance(fake, Scorer)


def test_load_scorer_dispatches_on_kind() -> None:
    manifest = _manifest("panel_ltr_xgboost")
    eps = [_stub_ep("panel_ltr_xgboost", _fake_loader)]
    with patch(
        "renquant_common.contracts.scorer.metadata.entry_points",
        return_value=eps,
    ):
        scorer = load_scorer(manifest)
    assert isinstance(scorer, Scorer)
    assert scorer.feature_cols == ["a", "b"]
    assert scorer.feature_fingerprint() == "sha256:abc"


def test_load_scorer_raises_for_unknown_kind() -> None:
    manifest = _manifest("nonexistent_kind")
    eps = [_stub_ep("panel_ltr_xgboost", _fake_loader)]
    with patch(
        "renquant_common.contracts.scorer.metadata.entry_points",
        return_value=eps,
    ):
        with pytest.raises(ScorerKindNotRegistered, match="nonexistent_kind"):
            load_scorer(manifest)


def test_load_scorer_rejects_non_protocol_return() -> None:
    manifest = _manifest("broken_kind")
    eps = [_stub_ep("broken_kind", _broken_loader)]
    with patch(
        "renquant_common.contracts.scorer.metadata.entry_points",
        return_value=eps,
    ):
        with pytest.raises(TypeError, match="does not satisfy"):
            load_scorer(manifest)
