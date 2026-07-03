"""Byte-identity guard for the deprecated 0.8.1 back-compat shims (0.9.1).

renquant-common 0.9.0 (the M6 schema-v1 rewrite, PRs #19/#20) removed four
names that renquant-pipeline main actively imports
(``src/renquant_pipeline/kernel/panel_pipeline/panel_scorer.py:46``, pinned
by pipeline's ``tests/test_model_content_sha256_shared.py``):
``model_content_sha256_from_path``, ``stamp_artifact_metadata``,
``MUTABLE_ARTIFACT_KEYS``, ``PREDICTIVE_CONTENT_HINTS``. 0.9.1 restores
them as deprecated shims with VERBATIM 0.8.1 semantics for the M6
migration window (removed again in the M6 stage-2 pipeline migration PR).

The ``sha256:...`` constants below are the GROUND TRUTH: they were computed
by executing the actual 0.8.1 implementation (``git show
b96d190:src/renquant_common/model_fingerprint.py``) against these exact
payloads. They pin the shims to the 0.8.1 output byte-for-byte — live
artifacts and calibrators carry stamps produced by those semantics
(incident-hot: 2026-05-27 / 06-22 / 07-01 fail-closed no-trade). Do NOT
"fix" a failing hash here by updating the constant; a divergence means the
shim broke 0.8.1 fidelity.
"""
from __future__ import annotations

import datetime
import json
import warnings

import pytest

from renquant_common.model_fingerprint import (
    MUTABLE_ARTIFACT_KEYS,
    PREDICTIVE_CONTENT_HINTS,
    OPERATIONAL_KEYS,
    PREDICTIVE_KEYS,
    UnclassifiedKeyError,
    artifact_sha256,
    model_content_sha256,
    model_content_sha256_from_path,
    stamp_artifact_metadata,
)

# ---------------------------------------------------------------------------
# 0.8.1 reference outputs (computed from commit b96d190 — see module
# docstring). Ground truth; never regenerate from the shim itself.
# ---------------------------------------------------------------------------
REF_P1_CONTENT = "sha256:a64b282442ceeb767846f29251305467dd727a91344a30e74cb0cb8ba4a87322"
REF_P2_UNCLASSIFIED_CONTENT = "sha256:fdc37806446858dc9ae8c00bb1ae1c2329f74657320238ef6ae6832db9ae191f"
REF_P4_DEFAULT_STR_CONTENT = "sha256:f99016bc817bb79c2feccac003c3d0557ff528818de1322a6f632d5acb305d57"
REF_P1_FILE_SHA = "sha256:23f12e51bee8c8650867f0b46a4ef3b6a5e9c308f7e855e42506c11662246dc7"


def _payload(**overrides) -> dict:
    """The 0.8.1 test suite's canonical panel-LTR fixture payload."""
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


def _write_payload(tmp_path, payload: dict, name: str = "artifact.json"):
    p = tmp_path / name
    # Exact bytes matter for the pinned artifact_sha256 fixture.
    p.write_text(json.dumps(payload, sort_keys=True))
    return p


# ---------------------------------------------------------------------------
# Byte-identity to 0.8.1
# ---------------------------------------------------------------------------

def test_from_path_matches_0_8_1_reference_hash(tmp_path) -> None:
    p = _write_payload(tmp_path, _payload())
    with pytest.warns(DeprecationWarning):
        assert model_content_sha256_from_path(p) == REF_P1_CONTENT


def test_file_hash_matches_0_8_1_reference(tmp_path) -> None:
    p = _write_payload(tmp_path, _payload())
    assert artifact_sha256(p) == REF_P1_FILE_SHA


def test_stamp_artifact_metadata_matches_0_8_1_reference(tmp_path) -> None:
    payload = _payload()
    p = _write_payload(tmp_path, payload)
    with pytest.warns(DeprecationWarning):
        meta = stamp_artifact_metadata({}, p, payload=payload)
    assert meta["artifact_path"] == str(p)
    assert meta["artifact_sha256"] == REF_P1_FILE_SHA
    assert meta["artifact_fingerprint"] == REF_P1_FILE_SHA
    assert meta["model_content_fingerprint"] == REF_P1_CONTENT


def test_unclassified_key_is_hashed_not_raised_exactly_like_0_8_1(tmp_path) -> None:
    """The load-bearing isolation property: schema v1 raises
    UnclassifiedKeyError on an unknown key, but the 0.8.1 denylist HASHED
    it (silent default). The shim must reproduce the 0.8.1 hash — proving
    it does NOT route through the total-classification hasher."""
    payload = _payload(brand_new_operational_key="surprise")
    with pytest.raises(UnclassifiedKeyError):
        model_content_sha256(payload)  # v1 API: hard error
    p = _write_payload(tmp_path, payload)
    with pytest.warns(DeprecationWarning):
        assert model_content_sha256_from_path(p) == REF_P2_UNCLASSIFIED_CONTENT


def test_label_col_relabel_is_invariant_exactly_like_0_8_1(tmp_path) -> None:
    """0.8.1 EXCLUDED label_col (denylist); schema v1 classifies it
    PREDICTIVE. The shim must keep the 0.8.1 exclusion: relabeled payload
    hashes identically."""
    p = _write_payload(tmp_path, _payload(label_col="fwd_20d_excess"))
    with pytest.warns(DeprecationWarning):
        assert model_content_sha256_from_path(p) == REF_P1_CONTENT


def test_default_str_fallback_is_preserved(tmp_path) -> None:
    """0.8.1 serialized non-JSON values via ``default=str`` (lossy, silent);
    schema v1 refuses them. The shim keeps the 0.8.1 behavior — fidelity
    over ideology for the migration window."""
    payload = _payload(
        params={"objective": "rank:pairwise", "cutoff": datetime.date(2026, 1, 15)},
    )
    p = tmp_path / "artifact.json"
    p.write_text("{}")  # file bytes only feed artifact_sha256; payload= wins
    with pytest.warns(DeprecationWarning):
        meta = stamp_artifact_metadata({}, p, payload=payload)
    assert meta["model_content_fingerprint"] == REF_P4_DEFAULT_STR_CONTENT


def test_shim_output_differs_from_v1_hasher() -> None:
    """If these ever agree on the canonical fixture, the isolation property
    is moot and someone probably rerouted the shim through v1 — re-verify
    against b96d190 before touching this."""
    assert model_content_sha256(_payload()) != REF_P1_CONTENT


# ---------------------------------------------------------------------------
# 0.8.1 silent-fallback semantics (ported from the 0.8.1 test suite)
# ---------------------------------------------------------------------------

def test_from_path_falls_back_on_non_dict(tmp_path) -> None:
    p = tmp_path / "not_a_dict.json"
    p.write_text(json.dumps([1, 2, 3]))
    with pytest.warns(DeprecationWarning):
        assert model_content_sha256_from_path(p) == artifact_sha256(p)


def test_from_path_falls_back_on_unparseable_bytes(tmp_path) -> None:
    p = tmp_path / "model.bin"
    p.write_bytes(b"\x00\x01\x02not-json")
    with pytest.warns(DeprecationWarning):
        assert model_content_sha256_from_path(p) == artifact_sha256(p)


def test_from_path_falls_back_when_no_predictive_content(tmp_path) -> None:
    p = tmp_path / "meta_only.json"
    p.write_text(json.dumps({"trained_date": "2026-06-01", "metadata": {}}))
    with pytest.warns(DeprecationWarning):
        assert model_content_sha256_from_path(p) == artifact_sha256(p)


def test_stamp_preserves_existing_values(tmp_path) -> None:
    payload = _payload()
    p = _write_payload(tmp_path, payload)
    with pytest.warns(DeprecationWarning):
        meta = stamp_artifact_metadata({"artifact_path": "custom"}, p, payload=payload)
    assert meta["artifact_path"] == "custom"


def test_stamp_flattens_nested_metadata(tmp_path) -> None:
    payload = _payload()
    p = _write_payload(tmp_path, payload)
    with pytest.warns(DeprecationWarning):
        meta = stamp_artifact_metadata({"metadata": {"extra_field": 1}}, p, payload=payload)
    assert meta["extra_field"] == 1


def test_stamp_without_payload_uses_from_path(tmp_path) -> None:
    payload = _payload()
    p = _write_payload(tmp_path, payload)
    with pytest.warns(DeprecationWarning):
        meta = stamp_artifact_metadata({}, p, payload=None)
    assert meta["model_content_fingerprint"] == REF_P1_CONTENT


def test_stamp_falls_back_to_file_hash_when_no_predictive_content(tmp_path) -> None:
    payload = {"trained_date": "2026-06-01", "metadata": {}}
    p = _write_payload(tmp_path, payload)
    with pytest.warns(DeprecationWarning):
        meta = stamp_artifact_metadata({}, p, payload=payload)
    assert meta["model_content_fingerprint"] == artifact_sha256(p)


# ---------------------------------------------------------------------------
# Deprecation + legacy tables
# ---------------------------------------------------------------------------

def test_both_shims_emit_deprecation_warning_naming_the_removal_point(tmp_path) -> None:
    payload = _payload()
    p = _write_payload(tmp_path, payload)
    with pytest.warns(DeprecationWarning, match="M6 stage-2"):
        model_content_sha256_from_path(p)
    with pytest.warns(DeprecationWarning, match="M6 stage-2"):
        stamp_artifact_metadata({}, p, payload=payload)


def test_legacy_tables_restored_verbatim_and_disjoint() -> None:
    assert MUTABLE_ARTIFACT_KEYS.isdisjoint(PREDICTIVE_CONTENT_HINTS)
    # Sentinels for the deliberate 0.8.1-vs-v1 divergences: 0.8.1 excluded
    # label_col/label/lookahead_days; schema v1 classifies them PREDICTIVE.
    for moved in ("label_col", "label", "lookahead_days"):
        assert moved in MUTABLE_ARTIFACT_KEYS
        assert moved in PREDICTIVE_KEYS
    # 0.8.1 tables know nothing of the #20 real-artifact census fields.
    assert "best_iter" not in MUTABLE_ARTIFACT_KEYS
    assert "best_iter" in OPERATIONAL_KEYS
    assert "feature_source_contract" not in PREDICTIVE_CONTENT_HINTS
    assert "feature_source_contract" in PREDICTIVE_KEYS


def test_v1_api_surface_untouched_by_shims() -> None:
    """The shims are additive: the v1 names still exist and the v1 hasher
    still enforces total classification on the canonical fixture."""
    from renquant_common import model_fingerprint as mf

    for name in (
        "FINGERPRINT_SCHEMA_VERSION",
        "PREDICTIVE_KEYS",
        "OPERATIONAL_KEYS",
        "model_content_sha256",
        "predictive_field_digests",
        "stamp",
        "verify",
        "artifact_sha256",
    ):
        assert hasattr(mf, name)
    stamped = mf.stamp(_payload())
    mf.verify(
        _payload(),
        stamped["model_content_fingerprint"],
        stamped["fingerprint_schema_version"],
    )


def test_importing_the_legacy_names_is_silent() -> None:
    """Pipeline imports these at module-import time (panel_scorer.py:46);
    the warning fires on CALL, not on import, so consumers running with
    ``-W error`` can still import the pinned pipeline main."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        from renquant_common.model_fingerprint import (  # noqa: F401
            MUTABLE_ARTIFACT_KEYS as _a,
            PREDICTIVE_CONTENT_HINTS as _b,
            model_content_sha256_from_path as _c,
            stamp_artifact_metadata as _d,
        )
