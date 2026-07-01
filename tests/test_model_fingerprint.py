"""Tests for ``renquant_common.model_fingerprint`` — the single source of
truth for the calibrator/scorer binding fingerprint.

Extracted 2026-07-01 after a recurring production incident (2026-05-27,
2026-06-22, 2026-07-01): renquant-model and renquant-pipeline each
hand-copied ``model_content_sha256`` with DIFFERENT included/excluded
field sets, so a calibrator fit by one could never match the runtime
check by another. This module is now the ONLY place the algorithm is
defined; both repos import it.
"""
from __future__ import annotations

import json

import pytest

from renquant_common.model_fingerprint import (
    MUTABLE_ARTIFACT_KEYS,
    PREDICTIVE_CONTENT_HINTS,
    artifact_sha256,
    model_content_sha256,
    model_content_sha256_from_path,
    stamp_artifact_metadata,
)


def _payload(**overrides) -> dict:
    base = {
        "kind": "panel_ltr_xgboost",
        "version": 3,
        "feature_cols": ["a", "b", "c"],
        "params": {"objective": "rank:pairwise", "max_depth": 4},
        "booster_raw_json": '{"fake": "booster"}',
        "label_col": "fwd_60d_excess",
        "trained_date": "2026-06-01",
        "metadata": {"note": "irrelevant"},
    }
    base.update(overrides)
    return base


def test_determinism_same_payload_same_hash() -> None:
    payload = _payload()
    assert model_content_sha256(payload) == model_content_sha256(json.loads(json.dumps(payload)))


def test_mutable_metadata_edits_do_not_change_fingerprint() -> None:
    """The whole point of the denylist: post-training bookkeeping edits
    (WF gate stamps, promotion status, CV bookkeeping, docstring-adjacent
    fields) must not flip the fingerprint — that's what caused 3 calibrator
    rebinds in one day before the original 2026-05-30 fix."""
    base = model_content_sha256(_payload())
    mutated = model_content_sha256(_payload(
        trained_date="2026-07-01",
        wf_gate_metadata={"tier": 3},
        promotion_status="promoted",
        promotion_gating_reason="ok",
        cv_method="purged_kfold",
        cv_embargo_days=30,
        train_run_id="run-123",
        oos_mean_ic=0.05,
        version=4,
    ))
    assert base == mutated


def test_predictive_content_change_changes_fingerprint() -> None:
    base = model_content_sha256(_payload())
    changed = model_content_sha256(_payload(booster_raw_json='{"fake": "different-booster"}'))
    assert base != changed


def test_no_predictive_content_raises() -> None:
    with pytest.raises(ValueError):
        model_content_sha256({"trained_date": "2026-06-01", "metadata": {}})


def test_label_col_excluded_from_fingerprint() -> None:
    """label_col is metadata about what the model was trained to predict,
    not part of the predictor itself — must be excluded (this was one of
    the concrete divergences between the two hand-copied implementations:
    renquant-model's allowlist INCLUDED label_col, renquant-pipeline's
    denylist EXCLUDED it)."""
    base = model_content_sha256(_payload())
    relabeled = model_content_sha256(_payload(label_col="fwd_20d_excess"))
    assert base == relabeled


def test_mutable_artifact_keys_and_predictive_hints_disjoint() -> None:
    assert MUTABLE_ARTIFACT_KEYS.isdisjoint(PREDICTIVE_CONTENT_HINTS)


def test_model_content_sha256_from_path(tmp_path) -> None:
    payload = _payload()
    p = tmp_path / "artifact.json"
    p.write_text(json.dumps(payload))
    assert model_content_sha256_from_path(p) == model_content_sha256(payload)


def test_model_content_sha256_from_path_falls_back_on_non_dict(tmp_path) -> None:
    p = tmp_path / "not_a_dict.json"
    p.write_text(json.dumps([1, 2, 3]))
    assert model_content_sha256_from_path(p) == artifact_sha256(p)


def test_stamp_artifact_metadata_sets_expected_fields(tmp_path) -> None:
    payload = _payload()
    p = tmp_path / "artifact.json"
    p.write_text(json.dumps(payload))
    meta = stamp_artifact_metadata({}, p, payload=payload)
    assert meta["artifact_path"] == str(p)
    assert meta["artifact_sha256"] == artifact_sha256(p)
    assert meta["artifact_fingerprint"] == artifact_sha256(p)
    assert meta["model_content_fingerprint"] == model_content_sha256(payload)


def test_stamp_artifact_metadata_preserves_existing_values(tmp_path) -> None:
    payload = _payload()
    p = tmp_path / "artifact.json"
    p.write_text(json.dumps(payload))
    meta = stamp_artifact_metadata({"artifact_path": "custom"}, p, payload=payload)
    assert meta["artifact_path"] == "custom"


def test_stamp_artifact_metadata_flattens_nested_metadata(tmp_path) -> None:
    payload = _payload()
    p = tmp_path / "artifact.json"
    p.write_text(json.dumps(payload))
    meta = stamp_artifact_metadata({"metadata": {"extra_field": 1}}, p, payload=payload)
    assert meta["extra_field"] == 1
