"""Pinning test for the bundle-contract fixture vectors (GOAL-5 AC4).

RFC RenQuant#492 §2.5: "a contract fixture in renquant-common pins the
verdict semantics both sides test against." These tests are the pin —
they freeze the vectors' structural contract (contract version, member
names, serialization rule, the exact five case names, and every case's
expected ``PairVerdict`` fields) plus the byte-level provenance of the
promoted file. renquant-pipeline (``bundle_contract.validate_pair``, the
reader side) and renquant-artifacts (``create_default_store``, the
publisher side) consume these vectors; a change here is a contract
change and requires a ``contract_version`` bump with coordinated
consumer PRs — never a silent edit.
"""
from __future__ import annotations

from hashlib import sha256
from importlib import resources

from renquant_common.contracts.bundle_contract import (
    BUNDLE_CONTRACT_VECTORS_RESOURCE,
    bundle_contract_vector_cases,
    load_bundle_contract_vectors,
)

#: Byte-for-byte provenance: sha256 of the promoted file == sha256 of the
#: phase-2 source it was copied from —
#: renquant-pipeline@ad3520f80e571aa91403b20ca395e025d7cdd35d
#: tests/fixtures/bundle_contract/vectors.json (renquant-pipeline#206).
#: If this fails, the file was edited in place: that is a contract change
#: — bump contract_version and update BOTH consumers instead.
PINNED_SHA256 = "710d5977c9b9960e089c31038ff8af5d7e49d813e9518b4fb23db9cdf47a6f7e"

#: The five contract-v1 cases and their expected verdicts, frozen.
EXPECTED_VERDICTS = {
    "matching_pair_legacy_schema": {
        "ok": True, "matched_schema": "legacy", "reason_codes": [],
    },
    "matching_pair_v1_schema": {
        "ok": True, "matched_schema": "v1", "reason_codes": [],
    },
    "mismatched_pair": {
        "ok": False, "matched_schema": None,
        "reason_codes": ["fingerprint_mismatch"],
    },
    "missing_binding": {
        "ok": False, "matched_schema": None,
        "reason_codes": ["missing_binding"],
    },
    "cross_schema_comparison_refused": {
        "ok": False, "matched_schema": None,
        "reason_codes": ["cross_schema_refused"],
    },
}


def test_vectors_byte_provenance_pinned() -> None:
    raw = (
        resources.files("renquant_common.contracts")
        .joinpath(BUNDLE_CONTRACT_VECTORS_RESOURCE)
        .read_bytes()
    )
    assert sha256(raw).hexdigest() == PINNED_SHA256, (
        "bundle_contract_vectors.json content drifted from the promoted "
        "phase-2 file; vector edits are contract changes — bump "
        "contract_version and update renquant-pipeline + renquant-artifacts "
        "in coordinated PRs, then re-pin this digest"
    )


def test_vectors_pin_contract_identity() -> None:
    vectors = load_bundle_contract_vectors()
    assert vectors["contract"] == "renquant_pipeline.bundle_contract.validate_pair"
    assert vectors["contract_version"] == 1
    assert vectors["scorer_member"] == "panel-ltr.alpha158_fund.json"
    assert vectors["calibrator_member"] == "panel-rank-calibration.json"
    assert vectors["member_serialization"] == (
        "json.dumps(payload, sort_keys=True, indent=2) + '\\n' (utf-8)"
    )


def test_vectors_pin_exact_case_set_and_verdicts() -> None:
    cases = bundle_contract_vector_cases()
    assert sorted(cases) == sorted(EXPECTED_VERDICTS)
    for name, expected in EXPECTED_VERDICTS.items():
        case = cases[name]
        assert case["expected"] == expected, name
        # Every case is materially complete: both member payloads, a
        # schema-v1 manifest naming exactly the two members, and the
        # migration-window flag the verdict was computed under.
        assert case["manifest"]["schema_version"] == 1
        assert sorted(case["manifest"]["members"]) == [
            "panel-ltr.alpha158_fund.json",
            "panel-rank-calibration.json",
        ]
        assert isinstance(case["accept_legacy_stamps"], bool)
        assert case["scorer_payload"]["kind"] == "panel_ltr_xgboost"
        assert case["calibrator_payload"]["kind"] == "global_panel_calibration"


def test_loader_returns_fresh_documents() -> None:
    a = load_bundle_contract_vectors()
    b = load_bundle_contract_vectors()
    assert a == b
    assert a is not b
    a["cases"].clear()
    assert load_bundle_contract_vectors()["cases"], (
        "loader must not share mutable state across calls"
    )
