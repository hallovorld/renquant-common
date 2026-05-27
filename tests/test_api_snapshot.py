"""Public-API snapshot test.

Any change to the public surface of ``renquant_common`` must update
``tests/api_snapshot/public_api.json`` in the same PR. The diff is the
review surface for breaking vs additive change classification per RFC
§"Schema Versioning §3".

The snapshot covers:

* Package version (bumped per semver).
* Sorted list of public names exported from ``renquant_common``.
* The regime taxonomy (closed set per PRIME DIRECTIVE).
* The entry-point group used by ``load_scorer``.
* Schemas declared frozen.
* Fields that ``ArtifactManifest`` MUST NOT have (e.g. ``promotion_status``
  is derived from branch per §"Branch Model").
* The :class:`Tier` enum members.
"""
from __future__ import annotations

import json
from importlib import metadata
from pathlib import Path

import pytest

import renquant_common as common
from renquant_common import (
    ArtifactManifest,
    RegimeLabel,
    Tier,
)
from renquant_common.contracts.scorer import SCORER_ENTRY_POINT_GROUP

SNAPSHOT_PATH = Path(__file__).parent / "api_snapshot" / "public_api.json"


@pytest.fixture(scope="module")
def snapshot() -> dict:
    return json.loads(SNAPSHOT_PATH.read_text())["renquant_common"]


def test_version_matches_snapshot(snapshot: dict) -> None:
    declared = metadata.version("renquant-common")
    assert declared == snapshot["version"], (
        "renquant-common version drifted from snapshot; bump per semver "
        "and update tests/api_snapshot/public_api.json"
    )


def test_public_names_match_snapshot(snapshot: dict) -> None:
    actual = sorted(common.__all__)
    expected = sorted(snapshot["public_names"])
    assert actual == expected, (
        "renquant_common public __all__ drifted from snapshot; update "
        "tests/api_snapshot/public_api.json in the same PR and classify "
        "the change as additive (minor) or breaking (major) per "
        "RFC §'Schema Versioning'"
    )


def test_regime_labels_match_snapshot(snapshot: dict) -> None:
    actual = sorted(RegimeLabel.values())
    expected = sorted(snapshot["regime_labels"])
    assert actual == expected, (
        "RegimeLabel taxonomy drifted; adding or renaming a regime is a "
        "MAJOR semver bump per RFC §'Schema Versioning §6' and requires "
        "coordinated consumer migration"
    )


def test_scorer_entry_point_group_stable(snapshot: dict) -> None:
    assert SCORER_ENTRY_POINT_GROUP == snapshot["scorer_entry_point_group"]


def test_artifact_manifest_does_not_have_forbidden_fields(snapshot: dict) -> None:
    fields = set(ArtifactManifest.model_fields)
    for forbidden in snapshot["artifact_manifest_must_not_have"]:
        assert forbidden not in fields, (
            f"ArtifactManifest grew a forbidden field {forbidden!r}; per "
            f"RFC §'Branch Model', promotion state is derived from the "
            f"branch (main vs candidate/<id> vs shadow) not a manifest field"
        )


def test_tier_members_match_snapshot(snapshot: dict) -> None:
    actual = sorted(t.value for t in Tier)
    expected = sorted(snapshot["tiers"])
    assert actual == expected
