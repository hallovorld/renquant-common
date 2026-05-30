"""Grep-style boundary test — no raw 'BULL_*' / 'BEAR' / 'CHOPPY' string
literals outside the :class:`RegimeLabel` enum definition.

Runs across every sibling subrepo's ``src/`` tree when the sibling
checkout is present at the standard sibling path
(``../renquant-<name>/src/renquant_<name>/``). The test skips siblings
that aren't checked out locally, so this works in CI environments that
only have ``renquant-common`` available.

Per RFC §"Cross-Repo Contracts → RegimeLabel" and §"Branch Model", the
regime taxonomy is the single source of truth in
``renquant_common.contracts.regime``. Raw string literals duplicate the
contract and risk drift when the enum changes.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

# Roots checked, relative to this repo's parent (the sibling layout used by
# RFC §"Inter-Repo Communication Mechanism").
SIBLING_NAMES = (
    "renquant-pipeline",
    "renquant-execution",
    "renquant-backtesting",
    "renquant-artifacts",
    "renquant-base-data",
    "renquant-model",
    "renquant-strategy-104",
    "renquant-orchestrator",
)

# Files in renquant-common itself that legitimately contain the literals
# (the enum definition + tests).
ALLOWED_IN_COMMON = {
    "src/renquant_common/contracts/regime.py",
    "src/renquant_common/contracts/__init__.py",
    "tests/test_regime.py",
    "tests/test_schemas.py",
    "tests/test_stats.py",
    "tests/test_scorer.py",
    "tests/test_api_snapshot.py",
    "tests/test_no_raw_regime_strings.py",
    "tests/api_snapshot/public_api.json",
}

REGIME_STRING_PATTERN = re.compile(
    r'["\'](BULL_CALM|BULL_VOLATILE|BULL_STRONG|BEAR|CHOPPY)["\']'
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _siblings_dir() -> Path:
    return _repo_root().parent


def _scan(root: Path, *, allow: set[str] = frozenset()) -> list[tuple[str, int, str]]:
    hits: list[tuple[str, int, str]] = []
    if not root.exists():
        return hits
    for py in root.rglob("*.py"):
        rel = py.relative_to(root.parent).as_posix() if root.name != "src" else py.relative_to(root.parent).as_posix()
        # Compute path relative to the subrepo root.
        try:
            rel_root = py.relative_to(root).as_posix()
        except ValueError:
            rel_root = py.as_posix()
        if rel_root in allow:
            continue
        text = py.read_text(encoding="utf-8", errors="ignore")
        for i, line in enumerate(text.splitlines(), start=1):
            if REGIME_STRING_PATTERN.search(line):
                # Allow lines that explicitly reference RegimeLabel
                if "RegimeLabel" in line:
                    continue
                hits.append((rel_root, i, line.strip()))
    return hits


def test_common_has_no_raw_regime_strings_outside_enum() -> None:
    """The enum's own module + its tests are allowed to contain the literals."""
    root = _repo_root()
    hits = _scan(root, allow=ALLOWED_IN_COMMON)
    assert hits == [], (
        f"renquant-common source contains raw regime string literals "
        f"outside the allowed list: {hits}. The RegimeLabel enum is the "
        f"single source of truth (RFC §'Cross-Repo Contracts → RegimeLabel')."
    )


# Phase 1 byte-equivalent lift zones — these subdirs hold 1:1 copies of
# umbrella code that uses raw regime strings (legacy from pre-RegimeLabel days).
# Excluded per sibling so the lifts can land cleanly; the rewrite to enum is
# a planned Phase 5+ refactor (track this in doc/arch/multirepo-tasks.md).
_PHASE1_BYTE_EQUIVALENT_ZONES = {
    "renquant-pipeline": {
        # C2.9 — preflight + data + trade_events
        "renquant_pipeline/kernel/data.py",
        "renquant_pipeline/kernel/data_cache.py",
        "renquant_pipeline/kernel/data_coverage.py",
        "renquant_pipeline/kernel/preflight.py",
        "renquant_pipeline/kernel/trade_events.py",
        # C2.5a — typed_past
        "renquant_pipeline/kernel/typed_past",
        # C2.11 — panel_pipeline
        "renquant_pipeline/kernel/panel_pipeline",
        # C2.12 + earlier pipeline lifts
        "renquant_pipeline/kernel/pipeline",
        # Pre-existing umbrella copies in pipeline that predate enum
        "renquant_pipeline/kernel/regime.py",
        "renquant_pipeline/kernel/regime_resolver.py",
        "renquant_pipeline/kernel/regime_hmm.py",
        "renquant_pipeline/kernel/config.py",
        "renquant_pipeline/kernel/portfolio_qp",
    },
    "renquant-backtesting": {
        # C2.4 — meta_label, C2.6 — forensics, C2.10 — labels
        "renquant_backtesting/meta_label",
        "renquant_backtesting/forensics",
        "renquant_backtesting/labels",
        # C2.2 — walk_forward
        "renquant_backtesting/walk_forward",
        # Pre-existing legacy zone
        "renquant_backtesting/wf_gate",
        "renquant_backtesting/simulation.py",
        "renquant_backtesting/runtime_parity.py",
        "renquant_backtesting/sim",
        "renquant_backtesting/gates",
        "renquant_backtesting/analysis",
        "renquant_backtesting/lean",
        "renquant_backtesting/lean_export",
    },
}


def _filter_phase1_hits(
    hits: list[tuple[str, int, str]],
    excluded_prefixes: set[str],
) -> list[tuple[str, int, str]]:
    """Drop hits whose path starts with any excluded prefix (file or dir)."""
    out: list[tuple[str, int, str]] = []
    for rel, ln, line in hits:
        if any(rel == e or rel.startswith(e + "/") for e in excluded_prefixes):
            continue
        out.append((rel, ln, line))
    return out


@pytest.mark.parametrize("sibling_name", SIBLING_NAMES)
def test_sibling_has_no_raw_regime_strings(sibling_name: str) -> None:
    """Scan each sibling subrepo's source tree (skip if not checked out).

    Phase 1 byte-equivalent lift zones (see ``_PHASE1_BYTE_EQUIVALENT_ZONES``)
    are excluded — those modules are 1:1 copies of umbrella code that uses
    raw regime strings pending the Phase 5+ rewrite to ``RegimeLabel.X.value``.
    """
    sibling_root = _siblings_dir() / sibling_name / "src"
    if not sibling_root.exists():
        pytest.skip(f"{sibling_name} not checked out at sibling path")
    raw_hits = _scan(sibling_root, allow=set())
    excluded = _PHASE1_BYTE_EQUIVALENT_ZONES.get(sibling_name, set())
    hits = _filter_phase1_hits(raw_hits, excluded)
    assert hits == [], (
        f"{sibling_name} source contains raw regime string literals: "
        f"{hits}. Replace with RegimeLabel.BULL_CALM.value etc. — the "
        f"enum is the single source of truth. Phase 1 byte-equivalent zones "
        f"are excluded; this hit is OUTSIDE those zones."
    )
