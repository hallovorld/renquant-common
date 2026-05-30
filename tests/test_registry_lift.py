"""Smoke test for the registry/ lift (Track C2.5b).

`kernel/registry/` carries the optional MLflow registry integration. Per
kernel-inventory.md it's a Shared C primitive — lifted to renquant-common.

Phase 1 invariant: byte-equivalent + file-presence; soft-skip if mlflow not installed.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest


_BT_PKG = Path(__file__).resolve().parents[1] / "src" / "renquant_common" / "registry"
_UMBRELLA = Path(__file__).resolve().parents[2] / "RenQuant" / "backtesting" / \
            "renquant_104" / "kernel" / "registry"


def test_byte_equivalent_to_umbrella() -> None:
    if not _UMBRELLA.exists():
        pytest.skip(f"umbrella not at {_UMBRELLA}")
    seen = 0
    for f in sorted(_BT_PKG.glob("*.py")):
        u = _UMBRELLA / f.name
        if not u.exists():
            continue
        assert hashlib.md5(f.read_bytes()).hexdigest() == hashlib.md5(u.read_bytes()).hexdigest(), \
            f"byte-mismatch: {f.name}"
        seen += 1
    assert seen >= 1


def test_expected_files_present() -> None:
    expected = {"__init__.py", "mlflow_registry.py"}
    present = {f.name for f in _BT_PKG.glob("*.py")}
    missing = expected - present
    assert not missing, f"missing: {missing}"


def test_mlflow_registry_imports_or_soft_skip() -> None:
    """mlflow is an optional dep; either the module imports OR raises ImportError('mlflow')."""
    try:
        import renquant_common.registry.mlflow_registry  # noqa: F401
    except ImportError as exc:
        if "mlflow" not in str(exc).lower():
            raise
