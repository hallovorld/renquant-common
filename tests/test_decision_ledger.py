"""Tests for the ``decision_ledger.connect()`` fail-closed pytest guard.

Background (2026-07 incident): a test elsewhere in the multi-repo system had
mocking that failed to intercept a ``connect()`` call, so the real, unmocked
call fired against the real production DB (``~/renquant-data/decision_ledger.db``)
and wrote fixture rows into it. This module is the single point of truth for
opening that DB, so the guard lives here, in ``connect()`` itself, rather
than being re-implemented per call site.

These tests never touch the real ``~/renquant-data/decision_ledger.db`` file:
the "points at production" cases raise before any filesystem access (the guard
runs before ``mkdir``/``sqlite3.connect``).  There is no escape hatch — no
pytest process may open the production ledger.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from renquant_common import decision_ledger
from renquant_common.decision_ledger import (
    DEFAULT_DB,
    connect,
    write_verdicts,
)


# ---------------------------------------------------------------------------
# (a) explicit non-production db_path works fine under pytest
# ---------------------------------------------------------------------------

def test_connect_with_explicit_tmp_path_works_under_pytest(tmp_path):
    assert "PYTEST_CURRENT_TEST" in os.environ  # sanity: we are under pytest
    conn = connect(tmp_path / "test.db")
    try:
        cols = [r[1] for r in conn.execute("PRAGMA table_info(decision_ledger)")]
        assert cols == ["run_id", "as_of", "scope", "gate", "verdict", "reason",
                         "inputs_json"]
    finally:
        conn.close()


def test_connect_with_memory_db_works_under_pytest():
    conn = connect(":memory:")
    try:
        n = write_verdicts(conn, "run-1", "2026-07-13", [
            {"scope": "book", "gate": "g", "verdict": "allow", "reason": "r"},
        ])
        assert n == 1
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# (b) no db_path (defaults to the real production path) RAISES under pytest
# ---------------------------------------------------------------------------

def test_connect_with_no_path_raises_under_pytest():
    """The dangerous default-fallthrough case: a test that forgot to mock or
    pass a path must fail loudly, never silently touch the real DB."""
    with pytest.raises(RuntimeError, match="REAL production decision ledger"):
        connect()


def test_connect_with_explicit_production_path_raises_under_pytest():
    """Guard applies whether the production path was defaulted OR passed
    explicitly — a test that (accidentally or not) spells out the real path
    literally is exactly as dangerous as relying on the default."""
    with pytest.raises(RuntimeError, match="REAL production decision ledger"):
        connect(DEFAULT_DB)


def test_connect_with_spelling_variant_of_production_path_still_raises():
    """Path *spelling* differences (trailing slash, redundant '.', a
    round-trip through str()) must not bypass the check — both sides are
    canonicalised before comparison. This never touches disk: the guard
    raises before any mkdir/sqlite3.connect happens."""
    home = str(Path.home())
    variant = f"{home}/./renquant-data/../renquant-data/decision_ledger.db"
    with pytest.raises(RuntimeError, match="REAL production decision ledger"):
        connect(variant)


def test_error_message_explains_the_fix():
    with pytest.raises(RuntimeError) as excinfo:
        connect()
    msg = str(excinfo.value)
    assert "tmp_path" in msg
    assert "DEFAULT_DB" in msg
    assert "no escape hatch" in msg.lower()


# ---------------------------------------------------------------------------
# (c) guard is inert outside of pytest (production behaviour unchanged)
# ---------------------------------------------------------------------------

def test_guard_is_a_noop_when_pytest_current_test_is_absent(monkeypatch, tmp_path):
    """Simulate "not running under pytest" by clearing the signal pytest
    sets for the duration of a test node. Outside of pytest, connect()
    against the (here, redirected-for-safety) default must behave exactly
    as it did before this guard existed."""
    fake_prod = tmp_path / "renquant-data" / "decision_ledger.db"
    monkeypatch.setattr(decision_ledger, "DEFAULT_DB", fake_prod)
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    conn = connect()
    try:
        assert fake_prod.exists()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# (d) regression: monkeypatching DEFAULT_DB does NOT redefine guard boundary
# ---------------------------------------------------------------------------

def test_monkeypatching_default_db_does_not_disable_guard(monkeypatch, tmp_path):
    """The guard derives the production identity from Path.home() inside the
    function — monkeypatching DEFAULT_DB to a safe tmp_path cannot move the
    protection boundary. An explicit connect() with the real production path
    must still raise."""
    monkeypatch.setattr(decision_ledger, "DEFAULT_DB", tmp_path / "safe.db")
    prod_path = Path.home() / "renquant-data/decision_ledger.db"
    with pytest.raises(RuntimeError, match="REAL production decision ledger"):
        connect(prod_path)


def test_relpath_is_inlined_not_a_module_attribute(monkeypatch, tmp_path):
    """The production relpath is a string literal inside the guard function,
    not a module attribute. Setting a module attribute of any plausible name
    cannot shift the guard boundary."""
    monkeypatch.setattr(decision_ledger, "DEFAULT_DB", tmp_path / "safe.db")
    monkeypatch.setattr(
        decision_ledger, "_PRODUCTION_LEDGER_RELPATH", "fake/path.db",
        raising=False,
    )
    prod_path = Path.home() / "renquant-data/decision_ledger.db"
    with pytest.raises(RuntimeError, match="REAL production decision ledger"):
        connect(prod_path)
