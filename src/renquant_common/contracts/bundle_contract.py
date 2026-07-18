"""Serving-pair bundle-contract fixture vectors (GOAL-5 AC4, RFC RenQuant#492 §2.5).

RFC §2.5: "A contract fixture in renquant-common pins the verdict
semantics both sides test against." This module is that fixture's
canonical home — phase 3 PR-B of the bundle-transactionality work
promotes the five phase-2 vectors (authored in renquant-pipeline#206,
``tests/fixtures/bundle_contract/vectors.json``, copied byte-for-byte)
so the pipeline validator (``renquant_pipeline.bundle_contract.
validate_pair``, the READER-side check) and the artifacts store binding
(``renquant_artifacts.create_default_store``, the PUBLISHER-side writer
step 6) test against ONE set of expected verdicts instead of two copies
that can drift.

The vectors file — ``bundle_contract_vectors.json``, packaged next to
this module — declares for each case: the scorer + calibrator member
payloads, the member serialization they are materialized with, a
schema-v1 manifest, the ``accept_legacy_stamps`` flag, and the expected
``PairVerdict`` fields (``ok``, ``matched_schema``, ``reason_codes``).
Five cases (contract v1): ``matching_pair_legacy_schema``,
``matching_pair_v1_schema``, ``mismatched_pair`` (the
2026-05-27/06-22/07-01/07-14→16 orphaned-binding incident shape),
``missing_binding``, ``cross_schema_comparison_refused``.

This module is data + a loader ONLY. It defines no verdict semantics of
its own: the semantics live in the phase-2 API; consumers assert their
implementation reproduces the expected verdicts recorded here. Any
semantic change requires bumping ``contract_version`` in the vectors and
updating both consumers in coordinated PRs.

Loader is stdlib-only (``json`` + ``importlib.resources``) so importing
it adds nothing beyond what ``renquant_common.contracts`` already loads.
"""
from __future__ import annotations

import json
from importlib import resources
from typing import Any

#: Packaged resource name of the vectors file (next to this module).
BUNDLE_CONTRACT_VECTORS_RESOURCE = "bundle_contract_vectors.json"


def load_bundle_contract_vectors() -> dict[str, Any]:
    """Load the pinned vectors document (parsed JSON, new dict per call).

    Top-level keys: ``contract`` (the API the vectors pin),
    ``contract_version``, ``rfc``, ``member_serialization``,
    ``scorer_member``, ``calibrator_member``, ``cases``.
    """
    payload = (
        resources.files(__package__)
        .joinpath(BUNDLE_CONTRACT_VECTORS_RESOURCE)
        .read_text(encoding="utf-8")
    )
    return json.loads(payload)


def bundle_contract_vector_cases() -> dict[str, dict[str, Any]]:
    """The vectors' cases keyed by case name (convenience for consumers
    that parametrize over them)."""
    return {case["name"]: case for case in load_bundle_contract_vectors()["cases"]}


__all__ = [
    "BUNDLE_CONTRACT_VECTORS_RESOURCE",
    "bundle_contract_vector_cases",
    "load_bundle_contract_vectors",
]
