"""Tests for ``renquant_common.model_fingerprint`` — schema v1 of the
total-classification content-fingerprint contract (renquant-orchestrator
``doc/design/2026-07-02-m6-fingerprint-unification.md``).

Includes the cross-repo REGRESSION fixtures: frozen copies of the two
divergent implementations' field sets as read on 2026-07-02, asserting the
new tables cover them — in particular that no field the model repo's
allowlist hashes today is silently dropped in migration (the false-MATCH
class the design exists to remove).
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from renquant_common.model_fingerprint import (
    FINGERPRINT_SCHEMA_VERSION,
    OPERATIONAL_KEYS,
    PREDICTIVE_KEYS,
    FingerprintError,
    MismatchError,
    NoPredictiveContentError,
    NonFiniteValueError,
    UnclassifiedKeyError,
    VersionGapError,
    artifact_sha256,
    model_content_sha256,
    predictive_field_digests,
    stamp,
    verify,
)

# ---------------------------------------------------------------------------
# Cross-repo regression fixtures: the two divergent implementations' field
# sets, frozen as read on 2026-07-02. If either upstream file changes, these
# fixtures are updated deliberately in the migration PRs — never silently.
# ---------------------------------------------------------------------------

# renquant-model src/renquant_model_gbdt/fit_calibrator_alpha158_fund.py:35
# (ADDITIVE allowlist — every field it hashes today).
MODEL_REPO_ALLOWLIST_2026_07_02 = frozenset({
    "params",
    "feature_cols",
    "feature_columns",
    "feature_means",
    "feature_stds",
    "feature_norm_kind",
    "feature_norm_kinds",
    "feature_raw_clip_low",
    "feature_raw_clip_high",
    "label_col",
    "booster_raw_json",
})

# renquant-pipeline src/renquant_pipeline/kernel/panel_pipeline/
# panel_scorer.py — _PREDICTIVE_CONTENT_HINTS (fields it treats as
# prediction content).
PIPELINE_PREDICTIVE_HINTS_2026_07_02 = frozenset({
    "booster_raw_json",
    "feature_cols",
    "feature_columns",
    "feature_means",
    "feature_stds",
    "feature_norm_kind",
    "feature_norm_kinds",
    "feature_raw_clip_low",
    "feature_raw_clip_high",
    "coef",
    "intercept",
    "clip_sigma",
    "state_dict",
    "config_dict",
    "model_bytes",
    "model_bytes_b64",
})

# renquant-pipeline panel_scorer.py — _MUTABLE_ARTIFACT_KEYS (SUBTRACTIVE
# denylist: fields it excludes from the hash today).
PIPELINE_DENYLIST_2026_07_02 = frozenset({
    "metadata",
    "wf_gate_metadata",
    "artifact_path",
    "artifact_sha256",
    "artifact_fingerprint",
    "model_content_fingerprint",
    "config_fingerprint",
    "config_fingerprint_fields",
    "trained_date",
    "training_notes",
    "label",
    "label_col",
    "lookahead_days",
    "panel_shape",
    "n_train_rows",
    "training_train_ic",
    "val_mean_ic",
    "val_median_ic",
    "test_mean_ic",
    "test_median_ic",
    "oos_mean_ic",
    "cv_method",
    "cv_embargo_days",
    "cv_folds",
    "cv_n_splits",
    "oos_std_ic",
    "oos_per_fold_ic",
    "eval_ic",
    "train_run_id",
    "sentiment_runtime_gate_contract",
    "sentiment_runtime_gate_trained",
    "promotion_status",
    "promotion_gating_reason",
    "version",
    "side_label",
})


def test_predictive_set_covers_model_repo_allowlist() -> None:
    """No predictive field the model repo hashes today is silently dropped
    in migration — the headline cross-repo regression fixture. A field the
    old allowlist hashed but the new tables ignored would reintroduce the
    false-MATCH class (predictive change, unchanged fingerprint)."""
    missing = MODEL_REPO_ALLOWLIST_2026_07_02 - PREDICTIVE_KEYS
    assert not missing, (
        f"model repo's allowlist fields dropped from PREDICTIVE_KEYS: "
        f"{sorted(missing)}"
    )


def test_predictive_set_covers_pipeline_predictive_hints() -> None:
    missing = PIPELINE_PREDICTIVE_HINTS_2026_07_02 - PREDICTIVE_KEYS
    assert not missing, (
        f"pipeline predictive-hint fields dropped from PREDICTIVE_KEYS: "
        f"{sorted(missing)}"
    )


def test_operational_set_covers_pipeline_denylist_except_label_col() -> None:
    """Every field the pipeline denylist excludes today stays OPERATIONAL,
    with ONE deliberate exception: ``label_col`` was the direct conflict
    between the two impls (model repo hashed it, pipeline excluded it) and
    is resolved PREDICTIVE — the calibrator fit is bound to the label
    horizon, and the model repo owns modeling-contract judgment."""
    expected_operational = PIPELINE_DENYLIST_2026_07_02 - {"label_col"}
    missing = expected_operational - OPERATIONAL_KEYS
    assert not missing, (
        f"pipeline denylist fields dropped from OPERATIONAL_KEYS: "
        f"{sorted(missing)}"
    )
    assert "label_col" in PREDICTIVE_KEYS


def test_tables_are_disjoint() -> None:
    """Total classification: exactly one class per key."""
    overlap = PREDICTIVE_KEYS & OPERATIONAL_KEYS
    assert not overlap, f"keys classified in BOTH tables: {sorted(overlap)}"


# ---------------------------------------------------------------------------
# Hashing behavior.
# ---------------------------------------------------------------------------


def _payload(**overrides) -> dict:
    base = {
        "kind": "panel_ltr_xgboost",
        "version": 3,
        "feature_cols": ["a", "b", "c"],
        "feature_means": [0.1, -1.5, 2.25],
        "params": {"objective": "rank:pairwise", "max_depth": 4},
        "booster_raw_json": '{"fake": "booster"}',
        "label_col": "fwd_60d_excess",
        "trained_date": "2026-06-01",
        "metadata": {"note": "irrelevant"},
    }
    base.update(overrides)
    return base


def test_identical_payloads_hash_identically() -> None:
    payload = _payload()
    assert model_content_sha256(payload) == model_content_sha256(
        json.loads(json.dumps(payload))
    )


def test_determinism_across_dict_orderings() -> None:
    """Insertion order of both top-level keys and nested dict keys must
    not affect the fingerprint (canonical sorted-key serialization)."""
    forward = _payload()
    reversed_top = dict(reversed(list(forward.items())))
    reversed_params = _payload(
        params={"max_depth": 4, "objective": "rank:pairwise"}
    )
    assert model_content_sha256(forward) == model_content_sha256(reversed_top)
    assert model_content_sha256(forward) == model_content_sha256(
        reversed_params
    )


def test_operational_mutations_do_not_change_fingerprint() -> None:
    """Post-training bookkeeping edits (WF gate stamps, promotion status,
    CV bookkeeping, our own stamp fields) must not flip the fingerprint —
    what caused 3 calibrator rebinds in one day before the 2026-05-30
    fix, and the 05-27/06-22/07-01 fail-closed incidents."""
    base = model_content_sha256(_payload())
    mutated = model_content_sha256(_payload(
        trained_date="2026-07-02",
        wf_gate_metadata={"tier": 3},
        promotion_status="promoted",
        promotion_gating_reason="ok",
        cv_method="purged_kfold",
        cv_embargo_days=30,
        train_run_id="run-123",
        oos_mean_ic=0.05,
        version=4,
        artifact_path="/somewhere/else.json",
        model_content_fingerprint="sha256:deadbeef",
        fingerprint_schema_version=FINGERPRINT_SCHEMA_VERSION,
    ))
    assert base == mutated


@pytest.mark.parametrize(
    "mutation",
    [
        {"booster_raw_json": '{"fake": "different-booster"}'},
        {"params": {"objective": "rank:pairwise", "max_depth": 6}},
        {"feature_cols": ["a", "b", "c", "d"]},
        {"feature_means": [0.1, -1.5, 2.26]},
        {"label_col": "fwd_20d_excess"},
        {"kind": "panel_ltr_lightgbm"},
    ],
)
def test_predictive_mutations_change_fingerprint(mutation: dict) -> None:
    assert model_content_sha256(_payload()) != model_content_sha256(
        _payload(**mutation)
    )


def test_label_col_change_rebinds_calibrator() -> None:
    """The resolved cross-impl conflict, pinned as behavior: re-labeling an
    artifact to a different horizon MUST change the pairing identity (the
    model repo's semantics win; the pipeline previously excluded it)."""
    assert model_content_sha256(_payload()) != model_content_sha256(
        _payload(label_col="fwd_20d_excess")
    )


def test_numpy_values_hash_like_plain_python() -> None:
    """Same IEEE-754 values must hash identically whether produced by
    numpy or plain Python (design §2b exact canonicalization)."""
    plain = _payload()
    numpied = _payload(
        feature_means=np.array([0.1, -1.5, 2.25]),
        params={
            "objective": "rank:pairwise",
            "max_depth": np.int64(4),
        },
    )
    assert model_content_sha256(plain) == model_content_sha256(numpied)


def test_floats_are_exact_not_rounded() -> None:
    """No lossy rounding (design §2b, r3 correction): two floats differing
    only in the last IEEE-754 bits must hash differently."""
    a = _payload(feature_means=[0.1 + 0.2])
    b = _payload(feature_means=[0.3])
    assert (0.1 + 0.2) != 0.3
    assert model_content_sha256(a) != model_content_sha256(b)


# ---------------------------------------------------------------------------
# Fail-loud classification.
# ---------------------------------------------------------------------------


def test_unclassified_key_raises_listing_the_key() -> None:
    with pytest.raises(UnclassifiedKeyError) as exc_info:
        model_content_sha256(_payload(brand_new_field=1))
    assert "brand_new_field" in str(exc_info.value)
    assert exc_info.value.keys == ("brand_new_field",)


def test_unclassified_keys_all_listed() -> None:
    """EVERY unclassified key is reported, not just the first."""
    with pytest.raises(UnclassifiedKeyError) as exc_info:
        model_content_sha256(
            _payload(zeta_field=1, alpha_field=2, mid_field=3)
        )
    assert exc_info.value.keys == ("alpha_field", "mid_field", "zeta_field")
    for key in ("alpha_field", "mid_field", "zeta_field"):
        assert key in str(exc_info.value)


def test_unclassified_key_raises_at_verify_time_too() -> None:
    """Design §2b verify-time rule: fails closed, never silently ignored
    or treated as OPERATIONAL-by-default."""
    stamped = stamp(_payload())
    with pytest.raises(UnclassifiedKeyError):
        verify(
            _payload(brand_new_field=1),
            stamped["model_content_fingerprint"],
            expected_version=stamped["fingerprint_schema_version"],
        )


def test_errors_are_valueerror_compatible() -> None:
    """Legacy call sites wrap the old impls in ``except ValueError``; the
    new contract's errors must keep failing closed through them."""
    assert issubclass(FingerprintError, ValueError)
    assert issubclass(UnclassifiedKeyError, FingerprintError)


def test_no_predictive_content_raises() -> None:
    with pytest.raises(NoPredictiveContentError):
        model_content_sha256(
            {"trained_date": "2026-06-01", "metadata": {}}
        )


def test_non_finite_predictive_value_rejected() -> None:
    """Design §2b: NaN/Inf in the PREDICTIVE subset is an invalid model
    state — hard error, never canonicalized into a stable hash."""
    with pytest.raises(NonFiniteValueError):
        model_content_sha256(_payload(feature_means=[0.1, float("nan")]))
    with pytest.raises(NonFiniteValueError):
        model_content_sha256(_payload(clip_sigma=float("inf")))


def test_non_finite_operational_value_is_ignored() -> None:
    """Non-finite values OUTSIDE the predictive subset are not serialized
    at all, so they neither error nor perturb the hash."""
    base = model_content_sha256(_payload())
    with_nan_metric = model_content_sha256(
        _payload(oos_mean_ic=float("nan"))
    )
    assert base == with_nan_metric


def test_unsupported_type_raises_not_default_str() -> None:
    """No lossy ``default=str`` fallback (the old impls' silent hazard)."""
    with pytest.raises(FingerprintError):
        model_content_sha256(_payload(config_dict={"when": object()}))


# ---------------------------------------------------------------------------
# stamp / verify round-trip and error taxonomy.
# ---------------------------------------------------------------------------


def test_stamp_returns_fingerprint_and_schema_version() -> None:
    payload = _payload()
    stamped = stamp(payload)
    assert stamped == {
        "model_content_fingerprint": model_content_sha256(payload),
        "fingerprint_schema_version": FINGERPRINT_SCHEMA_VERSION,
    }
    assert FINGERPRINT_SCHEMA_VERSION == 1


def test_verify_roundtrip_passes() -> None:
    payload = _payload()
    stamped = stamp(payload)
    verify(
        payload,
        stamped["model_content_fingerprint"],
        expected_version=stamped["fingerprint_schema_version"],
    )


def test_stamped_payload_reverifies_with_stamp_fields_merged() -> None:
    """Real artifacts carry the stamp fields inside the payload itself;
    merging them in must not perturb the fingerprint (both stamp fields
    are OPERATIONAL)."""
    payload = _payload()
    stamped = stamp(payload)
    merged = {**payload, **stamped}
    verify(
        merged,
        stamped["model_content_fingerprint"],
        expected_version=stamped["fingerprint_schema_version"],
    )


def test_version_gap_is_its_own_error_not_a_mismatch() -> None:
    """Design §2.3: a schema-version gap means "re-stamp under vN", never
    "content differs" — the two must be distinct exception types."""
    payload = _payload()
    stamped = stamp(payload)
    with pytest.raises(VersionGapError) as exc_info:
        verify(
            payload,
            stamped["model_content_fingerprint"],
            expected_version=FINGERPRINT_SCHEMA_VERSION + 1,
        )
    assert not isinstance(exc_info.value, MismatchError)
    assert exc_info.value.stamped_version == FINGERPRINT_SCHEMA_VERSION + 1
    assert exc_info.value.supported_version == FINGERPRINT_SCHEMA_VERSION


def test_version_gap_checked_before_content() -> None:
    """A version gap is not evidence about content: it must be raised
    even when the content hash ALSO differs."""
    with pytest.raises(VersionGapError):
        verify(
            _payload(),
            "sha256:" + "0" * 64,
            expected_version=FINGERPRINT_SCHEMA_VERSION + 1,
        )


def test_mismatch_error_carries_diff_hint() -> None:
    payload = _payload()
    wrong = "sha256:" + "0" * 64
    with pytest.raises(MismatchError) as exc_info:
        verify(payload, wrong, expected_version=FINGERPRINT_SCHEMA_VERSION)
    err = exc_info.value
    assert not isinstance(err, VersionGapError)
    assert err.expected == wrong
    assert err.actual == model_content_sha256(payload)
    assert err.field_digests == predictive_field_digests(payload)
    assert set(err.field_digests) == {
        k for k in payload if k in PREDICTIVE_KEYS
    }


def test_mismatch_field_digests_localize_the_divergent_field() -> None:
    """The stage-2 triage property: comparing per-field digests between the
    stamping and verifying side identifies exactly which field diverged."""
    base = predictive_field_digests(_payload())
    mutated = predictive_field_digests(
        _payload(booster_raw_json='{"fake": "different-booster"}')
    )
    differing = {k for k in base if base[k] != mutated[k]}
    assert differing == {"booster_raw_json"}


def test_verify_with_no_version_still_checks_content() -> None:
    """``expected_version=None`` supports pre-schema-stamp artifacts during
    migration: content is still checked."""
    payload = _payload()
    verify(payload, model_content_sha256(payload))
    with pytest.raises(MismatchError):
        verify(payload, "sha256:" + "0" * 64)


# ---------------------------------------------------------------------------
# artifact_sha256 (whole-file audit hash, unchanged semantics).
# ---------------------------------------------------------------------------


def test_artifact_sha256_hashes_file_bytes(tmp_path) -> None:
    p = tmp_path / "artifact.json"
    p.write_text(json.dumps(_payload()))
    first = artifact_sha256(p)
    assert first.startswith("sha256:")
    p.write_text(json.dumps(_payload(trained_date="2026-07-02")))
    assert artifact_sha256(p) != first
