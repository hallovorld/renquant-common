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
    "renquant-model-gbdt",
    "renquant-model-patchtst",
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


@pytest.mark.parametrize("sibling_name", SIBLING_NAMES)
def test_sibling_has_no_raw_regime_strings(sibling_name: str) -> None:
    """Scan each sibling subrepo's source tree (skip if not checked out)."""
    sibling_root = _siblings_dir() / sibling_name / "src"
    if not sibling_root.exists():
        pytest.skip(f"{sibling_name} not checked out at sibling path")
    # Sibling sources are allowed to contain the literals only inside
    # docstrings / comments. We enforce string-literal-free code by scanning
    # for the literal inside a quoted string; if a sibling needs the literal
    # in a string for some legacy reason, it should switch to
    # RegimeLabel.X.value.
    hits = _scan(sibling_root, allow=set())
    assert hits == [], (
        f"{sibling_name} source contains raw regime string literals: "
        f"{hits}. Replace with RegimeLabel.BULL_CALM.value etc. — the "
        f"enum is the single source of truth."
    )
