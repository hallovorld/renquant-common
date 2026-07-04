"""0.9.2: OPERATIONAL classification of the M6 stage-2 migration fields.

The stage-2 re-stamp plan (renquant-orchestrator
``doc/design/2026-07-03-m6-stage2-fingerprint-migration.md`` §3 step 2)
writes two new top-level fields into every re-stamped scorer artifact:

* ``model_content_fingerprint_legacy_081`` — the prior legacy (0.8.1)
  hash, preserved as audit/rollback metadata (never read by a verifier);
* ``restamp_provenance`` — the re-stamp run's provenance record.

Design §3 step 1 names this 0.9.2 prerequisite explicitly: without
classifying these keys, the first v1 ``stamp()`` of a dual-stamped
artifact raises ``UnclassifiedKeyError`` on the very fields the migration
added.

These tests pin the three properties the migration depends on:

1. a step-2-shaped payload stamps and verifies cleanly;
2. the addition is HASH-PRESERVING (same payload with/without the new
   fields hashes identically) — which is why ``FINGERPRINT_SCHEMA_VERSION``
   stays 1;
3. the frozen 0.8.1 legacy shim tables are untouched (adding the keys to
   ``MUTABLE_ARTIFACT_KEYS`` would silently CHANGE legacy hashes of
   re-stamped files — 0.8.1 fidelity is byte-frozen by
   ``test_model_fingerprint_legacy_shims.py``).
"""
from __future__ import annotations

import pytest

from renquant_common.model_fingerprint import (
    FINGERPRINT_SCHEMA_VERSION,
    MUTABLE_ARTIFACT_KEYS,
    OPERATIONAL_KEYS,
    PREDICTIVE_CONTENT_HINTS,
    PREDICTIVE_KEYS,
    UnclassifiedKeyError,
    model_content_sha256,
    stamp,
    verify,
)

STAGE2_FIELDS = ("model_content_fingerprint_legacy_081", "restamp_provenance")


def _payload() -> dict:
    return {
        "kind": "panel_ltr_xgboost",
        "version": 3,
        "feature_cols": ["a", "b", "c"],
        "params": {"objective": "rank:pairwise", "max_depth": 4},
        "booster_raw_json": '{"fake": "booster"}',
        "label_col": "fwd_60d_excess",
        "trained_date": "2026-06-01",
        "metadata": {"note": "irrelevant"},
    }


def _stage2_payload() -> dict:
    """A payload shaped exactly like a step-2 dual-stamped artifact."""
    base = _payload()
    stamp_fields = stamp(base)
    base.update(stamp_fields)
    base["model_content_fingerprint_legacy_081"] = "sha256:" + "0" * 64
    base["restamp_provenance"] = {
        "stamped_by": "restamp tool",
        "design": "2026-07-03-m6-stage2-fingerprint-migration.md §3 step 2",
        "operator_grant": "grant note",
    }
    return base


def test_stage2_fields_are_classified_operational() -> None:
    for key in STAGE2_FIELDS:
        assert key in OPERATIONAL_KEYS
        assert key not in PREDICTIVE_KEYS


def test_stage2_dual_stamped_payload_stamps_and_verifies() -> None:
    payload = _stage2_payload()
    # Stamping must not raise UnclassifiedKeyError on the migration fields.
    fields = stamp(payload)
    assert fields["fingerprint_schema_version"] == FINGERPRINT_SCHEMA_VERSION
    # And the artifact's own stamp verifies against its content.
    verify(
        payload,
        payload["model_content_fingerprint"],
        payload["fingerprint_schema_version"],
    )


def test_stage2_fields_are_hash_preserving() -> None:
    """Adding the migration fields must not change the v1 hash.

    This is the exact property that lets ``FINGERPRINT_SCHEMA_VERSION``
    stay 1 for this table change: no payload stampable under the 0.9.1
    tables hashes differently under 0.9.2 (payloads carrying these keys
    previously hard-errored, so none was ever stamped).
    """
    assert model_content_sha256(_stage2_payload()) == model_content_sha256(_payload())


def test_pre_092_behavior_was_a_hard_error() -> None:
    """The keys route through UnclassifiedKeyError when unclassified.

    Sanity check of the "hash-preserving because previously unstampable"
    argument: an actually-unclassified sibling key still hard-errors, so
    the classification tables (not a silent default) are what admit the
    stage-2 fields.
    """
    payload = _payload()
    payload["restamp_provenance_v2_unclassified"] = {"x": 1}
    with pytest.raises(UnclassifiedKeyError):
        model_content_sha256(payload)


def test_legacy_shim_tables_untouched_by_stage2_fields() -> None:
    """0.8.1 fidelity: the frozen legacy tables must NOT learn these keys.

    If ``MUTABLE_ARTIFACT_KEYS`` grew them, the legacy engine would start
    EXCLUDING them and the legacy hash of a re-stamped file would change
    out from under every 0.8.1-semantics reader — breaking the verbatim
    shim contract. The stage-2 audit path instead recomputes the legacy
    hash over the payload minus the migration-added top-level fields.
    """
    for key in STAGE2_FIELDS:
        assert key not in MUTABLE_ARTIFACT_KEYS
        assert key not in PREDICTIVE_CONTENT_HINTS
