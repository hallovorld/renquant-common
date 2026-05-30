"""Smoke test for the C2.7 shared-primitives lift.

Two more top-level kernel/*.py → renquant_common (Shared C per inventory):
  * calibrator_quality.py — calibrator health diagnostics (pool IC, n_unique)
  * row_coverage.py       — row-coverage gate primitives

Phase 1 invariant: byte-equivalent + clean import (no kernel.* deps).
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest


_BT_PKG = Path(__file__).resolve().parents[1] / "src" / "renquant_common"
_UMBRELLA = Path(__file__).resolve().parents[2] / "RenQuant" / "backtesting" / \
            "renquant_104" / "kernel"

_LIFTED = ("calibrator_quality.py", "row_coverage.py")


def test_byte_equivalent_to_umbrella() -> None:
    if not _UMBRELLA.exists():
        pytest.skip(f"umbrella not at {_UMBRELLA}")
    for name in _LIFTED:
        bt = _BT_PKG / name
        um = _UMBRELLA / name
        assert bt.exists(), f"missing in subrepo: {name}"
        assert um.exists(), f"missing in umbrella: {name}"
        assert hashlib.md5(bt.read_bytes()).hexdigest() == hashlib.md5(um.read_bytes()).hexdigest(), \
            f"byte-mismatch: {name}"


@pytest.mark.parametrize("name", [
    "renquant_common.calibrator_quality",
    "renquant_common.row_coverage",
])
def test_imports_cleanly(name: str) -> None:
    """These have zero kernel.* deps and should import cleanly in the subrepo."""
    __import__(name)
