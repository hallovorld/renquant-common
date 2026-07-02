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

from renquant_common.contracts.regime import RegimeLabel
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


# REAL-ARTIFACT CENSUS (r2 review directive "inspect real artifact
# families"). Top-level key sets read on 2026-07-02 from the live umbrella
# tree:
#
# * the production scorer artifact
#   ``data/panel-ltr-prod-alpha158-fund-fwd60d.json`` (trained 2026-05-18);
# * the shadow-lane artifacts ``data/shadow_analyst/*.json`` (which carry
#   the full CV/clip/preprocess field surface);
# * every field ``RenQuant/scripts/train_production_model.py::
#   build_artifact`` can write, including the conditional cutoff/window/
#   side-label/sentiment/addendum branches.
#
# These pin that schema v1's tables are TOTAL over the real XGB/GBDT JSON
# family: a real artifact must stamp without UnclassifiedKeyError, or
# stage-1 dual-write (design §2c) would crash training instead of
# shadowing it. (HF/PatchTST checkpoints are a separate, whole-file-bound
# family per design §2a and are out of scope for these tables.)

REAL_PROD_XGB_ARTIFACT_KEYS_2026_07_02 = frozenset({
    "best_iter", "booster_raw_json", "config_fingerprint",
    "config_fingerprint_fields", "feature_cols", "feature_means",
    "feature_stds", "kind", "label_col", "lookahead_days", "panel_shape",
    "params", "trained_date", "training_notes", "version",
})

REAL_SHADOW_XGB_ARTIFACT_KEYS_2026_07_02 = frozenset({
    "best_iter", "booster_raw_json", "config_fingerprint",
    "cv_embargo_days", "cv_folds", "cv_method", "cv_n_splits", "eval_ic",
    "feature_cols", "feature_means", "feature_norm_kind",
    "feature_preprocess_version", "feature_raw_clip_fit_split",
    "feature_raw_clip_high", "feature_raw_clip_low",
    "feature_source_contract", "feature_stds", "kind", "label_col",
    "lookahead_days", "metadata", "oos_mean_ic", "oos_per_fold_ic",
    "oos_std_ic", "panel_shape", "params", "train_run_id", "trained_date",
    "training_notes", "training_train_ic", "version",
})

# Conditional writer fields from build_artifact not present in the two
# snapshots above (cutoff/window provenance, side label, sentiment gate
# attestations, Track B addendum).
TRAIN_PRODUCTION_MODEL_CONDITIONAL_FIELDS_2026_07_02 = frozenset({
    "cutoff_date", "cutoff_embargo_days", "effective_train_cutoff_date",
    "train_start_date", "effective_train_start_date", "train_window",
    "side_label", "feature_addendum_v1",
    "sentiment_runtime_gate_contract",
    "sentiment_runtime_gate_feature_cols",
    "sentiment_runtime_gate_disabled_regimes",
    "sentiment_runtime_gate_zeroed_rows",
    "sentiment_runtime_gate_warmup_zeroed_rows",
    "sentiment_runtime_gate_missing_regime_policy",
    "sentiment_runtime_gate_policy",
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


def test_operational_set_covers_pipeline_denylist_except_reclassified_fields() -> None:
    """Every field the pipeline denylist excludes today stays OPERATIONAL,
    with THREE deliberate exceptions resolved PREDICTIVE: ``label_col`` (the
    direct conflict between the two impls — model repo hashed it, pipeline
    excluded it), ``lookahead_days`` (r2 correction: traced to
    ``global_calibrator.expected_return``'s horizon-rescaling logic, which
    reads this field to scale calibrated output — a mechanical, real
    predictive effect, not inert bookkeeping), and ``label`` (r2 correction:
    multiple artifact-producing scripts use it as the same
    target-definition descriptor label_col serves for other artifact
    families). Preserving the OLD denylist's classification is not evidence
    it was correct — that denylist is one of the conflicting
    implementations this module replaces; each exception here required
    tracing a real call site, not inheriting the old table."""
    expected_operational = PIPELINE_DENYLIST_2026_07_02 - {
        "label_col", "label", "lookahead_days",
    }
    missing = expected_operational - OPERATIONAL_KEYS
    assert not missing, (
        f"pipeline denylist fields dropped from OPERATIONAL_KEYS: "
        f"{sorted(missing)}"
    )
    assert "label_col" in PREDICTIVE_KEYS
    assert "label" in PREDICTIVE_KEYS
    assert "lookahead_days" in PREDICTIVE_KEYS


def test_tables_are_disjoint() -> None:
    """Total classification: exactly one class per key."""
    overlap = PREDICTIVE_KEYS & OPERATIONAL_KEYS
    assert not overlap, f"keys classified in BOTH tables: {sorted(overlap)}"


def test_tables_are_total_over_real_artifact_census() -> None:
    """Every top-level key observed in the REAL production/shadow XGB
    artifacts and every field the production trainer can write is
    classified — a real artifact stamps without UnclassifiedKeyError, so
    stage-1 dual-write shadows training instead of crashing it."""
    census = (
        REAL_PROD_XGB_ARTIFACT_KEYS_2026_07_02
        | REAL_SHADOW_XGB_ARTIFACT_KEYS_2026_07_02
        | TRAIN_PRODUCTION_MODEL_CONDITIONAL_FIELDS_2026_07_02
    )
    unclassified = census - PREDICTIVE_KEYS - OPERATIONAL_KEYS
    assert not unclassified, (
        f"real-artifact fields unclassified in schema v1: "
        f"{sorted(unclassified)}"
    )


def test_real_prod_artifact_shape_stamps_and_verifies() -> None:
    """Functional form of the census: a payload with the production
    artifact's full key surface (plus the conditional writer fields)
    round-trips stamp() → verify()."""
    payload = _payload(
        best_iter=100,
        config_fingerprint="sha256:9333f7bf91d10cc4",
        config_fingerprint_fields={"objective": "rank:pairwise"},
        panel_shape={"rows": 100, "tickers": 10, "dates": 10},
        training_train_ic=0.12,
        train_run_id="run-123",
        cv_method="purged_kfold",
        cv_embargo_days=30,
        cv_folds=[[0, 1], [2, 3]],
        cv_n_splits=2,
        eval_ic=0.05,
        oos_mean_ic=0.04,
        oos_std_ic=0.01,
        oos_per_fold_ic=[0.03, 0.05],
        feature_norm_kind=["legacy_full_z"] * 3,
        feature_raw_clip_low=[-3.0, -3.0, -3.0],
        feature_raw_clip_high=[3.0, 3.0, 3.0],
        feature_raw_clip_fit_split="train",
        feature_preprocess_version=2,
        feature_source_contract={"raw": "apply clips then z-score"},
        cutoff_date="2026-05-18",
        cutoff_embargo_days=60,
        effective_train_cutoff_date="2026-02-20",
        train_start_date="2016-01-01",
        effective_train_start_date="2016-01-01",
        train_window={"start": "2016-01-01", "end": "2026-02-20"},
        feature_addendum_v1={"track_b_features_active": ["pead_sue"]},
        sentiment_runtime_gate_contract="trained_zeroing",
        sentiment_runtime_gate_feature_cols=["sent_1"],
        sentiment_runtime_gate_disabled_regimes=[RegimeLabel.BEAR.value],
        sentiment_runtime_gate_zeroed_rows=42,
        sentiment_runtime_gate_warmup_zeroed_rows=7,
        sentiment_runtime_gate_missing_regime_policy="warmup_zero_only",
        sentiment_runtime_gate_policy={RegimeLabel.BULL_CALM.value: True},
    )
    stamped = stamp(payload)
    verify(
        {**payload, **stamped},
        stamped["model_content_fingerprint"],
        stamped["fingerprint_schema_version"],
    )


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
        "label": "fwd_60d_excess",
        "lookahead_days": 60,
        "trained_date": "2026-06-01",
        "metadata": {"note": "irrelevant"},
        "side_label": "specialist_wf_bull_2026-06-01",
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
        side_label="specialist_wf_bear_2026-07-02",
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
        {"label": "fwd_20d_excess"},
        {"lookahead_days": 20},
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


def test_lookahead_days_change_rebinds_calibrator() -> None:
    """r2 correction, pinned as behavior: renquant_model_common.
    global_calibrator.expected_return()/expected_return_vec() read a
    stamped ``lookahead_days`` directly to RESCALE the calibrated output by
    ``horizon_days / native`` whenever they differ — a drifted
    ``lookahead_days`` mechanically changes the calibrator's numeric
    output, so it must change the pairing identity like ``label_col``."""
    assert model_content_sha256(_payload()) != model_content_sha256(
        _payload(lookahead_days=20)
    )


def test_label_change_rebinds_calibrator() -> None:
    """r2 correction: artifact families that stamp a free-text ``label``
    instead of ``label_col`` (e.g. multi-horizon ensembles) must get the
    same pairing-identity protection ``label_col`` already has."""
    assert model_content_sha256(_payload()) != model_content_sha256(
        _payload(label="multi_horizon_ensemble (fwd_5d + fwd_20d + fwd_60d)")
    )


def test_side_label_change_does_not_change_fingerprint() -> None:
    """side_label identifies which experimental/side training config
    produced the artifact (lineage/provenance bookkeeping used to prevent a
    side-experiment's output from being mistaken for a production
    artifact) — it does not select how predictions are interpreted, so its
    value must not change the fingerprint."""
    assert model_content_sha256(_payload()) == model_content_sha256(
        _payload(side_label="specialist_wf_choppy_2026-07-02")
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


def test_verify_requires_expected_version_argument() -> None:
    """r2 correction: expected_version is now a MANDATORY positional
    argument, not an optional one defaulting to None. Omitting it must
    fail loudly at the call site (Python's own argument-binding error) —
    the prior optional form was exactly the "one forgotten argument
    bypasses the whole migration contract" hole the review required
    closed. There is no default/omitted-version form of verify() left."""
    payload = _payload()
    with pytest.raises(TypeError):
        verify(payload, model_content_sha256(payload))


def test_verify_rejects_older_version_as_gap_not_mismatch() -> None:
    """An OLDER stamped version than currently supported is a version
    gap (re-stamp under vN), not silently accepted and not reported as a
    content mismatch — same treatment as a newer version."""
    payload = _payload()
    stamped = stamp(payload)
    with pytest.raises(VersionGapError) as exc_info:
        verify(
            payload,
            stamped["model_content_fingerprint"],
            expected_version=FINGERPRINT_SCHEMA_VERSION - 1,
        )
    assert not isinstance(exc_info.value, MismatchError)
    assert exc_info.value.stamped_version == FINGERPRINT_SCHEMA_VERSION - 1


@pytest.mark.parametrize(
    "bad_version",
    [1.0, "1", None, True, False, [1], {}],
)
def test_verify_rejects_non_integer_version(bad_version) -> None:
    """A malformed version value (float, string, None, bool, or any other
    non-plain-int) must fail closed as a version gap rather than silently
    coercing through an equality check (the classic Python trap where
    ``1.0 == 1`` and ``True == 1``) or raising an unrelated TypeError deep
    inside comparison logic."""
    payload = _payload()
    stamped = stamp(payload)
    with pytest.raises(VersionGapError):
        verify(
            payload,
            stamped["model_content_fingerprint"],
            expected_version=bad_version,
        )


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
