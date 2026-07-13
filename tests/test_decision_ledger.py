"""Tests for the ``decision_ledger.connect()`` fail-closed pytest guard.

Background (2026-07 incident): a test elsewhere in the multi-repo system had
mocking that failed to intercept a ``connect()`` call, so the real, unmocked
call fired against the real production DB (``~/renquant-data/decision_ledger.db``)
and wrote fixture rows into it. This module is the single point of truth for
opening that DB, so the guard lives here, in ``connect()`` itself, rather
than being re-implemented per call site.

These tests never touch the real ``~/renquant-data/decision_ledger.db`` file:
the "points at production" cases either raise before any filesystem access
(the guard runs before ``mkdir``/``sqlite3.connect``), or use a monkeypatched
``DEFAULT_DB`` pointed at a ``tmp_path``-backed stand-in so "the production
path" for the test is itself a throwaway file.
"""
from __future__ import annotations

import os

import pytest

from renquant_common import decision_ledger
from renquant_common.decision_ledger import (
    ALLOW_LIVE_LEDGER_IN_TESTS_ENV,
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
    home = str(DEFAULT_DB.parent.parent)  # .../<home>
    variant = f"{home}/./renquant-data/../renquant-data/decision_ledger.db"
    with pytest.raises(RuntimeError, match="REAL production decision ledger"):
        connect(variant)


def test_error_message_explains_the_fix(tmp_path):
    with pytest.raises(RuntimeError) as excinfo:
        connect()
    msg = str(excinfo.value)
    assert "tmp_path" in msg
    assert "DEFAULT_DB" in msg
    assert ALLOW_LIVE_LEDGER_IN_TESTS_ENV in msg


# ---------------------------------------------------------------------------
# (c) the explicit escape hatch, proven without ever touching the real file
# ---------------------------------------------------------------------------

def test_escape_hatch_allows_the_configured_production_path_through(
    monkeypatch, tmp_path,
):
    """Point DEFAULT_DB (the thing the guard treats as "the production
    path") at a tmp_path-backed stand-in, so we can prove the escape-hatch
    logic end to end without the test ever opening the *real*
    ~/renquant-data/decision_ledger.db."""
    fake_prod = tmp_path / "renquant-data" / "decision_ledger.db"
    monkeypatch.setattr(decision_ledger, "DEFAULT_DB", fake_prod)

    # Without the escape hatch: still raises, even though it's a tmp_path
    # under the hood — from the guard's point of view this IS "the
    # production path" for the duration of this test.
    with pytest.raises(RuntimeError, match="REAL production decision ledger"):
        connect()

    # With the escape hatch set: allowed through, real connection returned.
    monkeypatch.setenv(ALLOW_LIVE_LEDGER_IN_TESTS_ENV, "1")
    conn = connect()
    try:
        assert fake_prod.exists()
        cols = [r[1] for r in conn.execute("PRAGMA table_info(decision_ledger)")]
        assert cols == ["run_id", "as_of", "scope", "gate", "verdict", "reason",
                         "inputs_json"]
    finally:
        conn.close()


def test_escape_hatch_value_other_than_1_does_not_bypass(monkeypatch, tmp_path):
    """Not a blanket bypass: the env var must be exactly '1', not merely set
    (guards against e.g. an accidentally-inherited empty-string env var)."""
    fake_prod = tmp_path / "renquant-data" / "decision_ledger.db"
    monkeypatch.setattr(decision_ledger, "DEFAULT_DB", fake_prod)
    monkeypatch.setenv(ALLOW_LIVE_LEDGER_IN_TESTS_ENV, "0")
    with pytest.raises(RuntimeError, match="REAL production decision ledger"):
        connect()


# ---------------------------------------------------------------------------
# guard is inert outside of pytest (production behaviour unchanged)
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
