"""Decision ledger persistence — append-only gate-verdict event store.

Moved from renquant-orchestrator to renquant-common (V-003 remediation)
so that both orchestrator and pipeline can import without a reverse
dependency.

One row per (run_id, scope, gate): the verdict a gate returned, its
reason, and the inputs it saw.  Append-only, WAL mode, busy timeout.
"""
from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from typing import Any, Iterable, Mapping

DEFAULT_DB = Path.home() / "renquant-data/decision_ledger.db"

_VALID_VERDICTS = ("allow", "halve", "block")

# Relative sub-path that identifies the production ledger under any
# home directory.  The guard computes the full path from Path.home()
# at call time — a function-local derivation that cannot be
# monkeypatched away.
_PRODUCTION_LEDGER_RELPATH = "renquant-data/decision_ledger.db"

DDL = """
CREATE TABLE IF NOT EXISTS decision_ledger (
  run_id TEXT NOT NULL, as_of DATE NOT NULL, scope TEXT NOT NULL,
  gate TEXT NOT NULL, verdict TEXT NOT NULL CHECK(verdict IN ('allow','halve','block')),
  reason TEXT NOT NULL, inputs_json TEXT NOT NULL DEFAULT '{}',
  PRIMARY KEY (run_id, scope, gate)
) WITHOUT ROWID;
"""


def _running_under_pytest() -> bool:
    """Reliable "a test is executing right now" signal.

    ``PYTEST_CURRENT_TEST`` is set by pytest for the duration of each test
    and unset otherwise. This is deliberately NOT ``"pytest" in
    sys.modules``: that can be true just because *something* imported
    pytest (e.g. a conftest, a plugin, a REPL) with no test actually
    running, which would make the guard fire outside of tests too.
    """
    return "PYTEST_CURRENT_TEST" in os.environ


def _canonical_path(db_path: str | Path) -> Path | None:
    """Canonicalise a DB path for equality comparisons so spelling
    differences (``~``, trailing slashes, symlinks, relative segments)
    can't be used to accidentally — or deliberately — dodge the
    production-path check below. Returns ``None`` for the sqlite
    in-memory pseudo-path, which can never refer to a real file."""
    if str(db_path) == ":memory:":
        return None
    expanded = Path(db_path).expanduser()
    try:
        return expanded.resolve()
    except (OSError, RuntimeError):
        # resolve() can raise on pathological inputs (e.g. symlink loops);
        # fail toward still comparing something rather than skipping the
        # check silently.
        return expanded


def _guard_against_live_ledger_in_tests(db_path: str | Path) -> None:
    """Raise loudly if ``db_path`` resolves to the real production ledger
    while a pytest test is running.  There is no escape hatch: a pytest
    process must never open the production ledger.

    The production identity is derived from ``Path.home()`` and the
    hard-coded relative path at call time — a function-local computation
    that cannot be defeated by monkeypatching a module attribute."""
    if not _running_under_pytest():
        return
    resolved = _canonical_path(db_path)
    if resolved is None:
        return
    production_path = _canonical_path(
        Path.home() / _PRODUCTION_LEDGER_RELPATH
    )
    if production_path is None or resolved != production_path:
        return
    raise RuntimeError(
        f"refusing to open the REAL production decision ledger "
        f"({production_path}) from inside a pytest test run.\n\n"
        "This is almost always a bug: a test's mocking failed to "
        "intercept a decision_ledger.connect() call, and the real, "
        "unmocked call is about to read/write the LIVE trading system's "
        "decision ledger. A 2026-07 incident hit exactly this path and "
        "wrote fixture rows into the production DB.\n\n"
        "Fix the test instead of bypassing this guard:\n"
        "  - pass an explicit non-production db_path, e.g. "
        "connect(tmp_path / 'test.db')\n"
        "  - or monkeypatch the default: monkeypatch.setattr("
        "'renquant_common.decision_ledger.DEFAULT_DB', tmp_path / 'x.db')\n"
        "  - or mock connect()/write_verdicts() so this code never runs\n\n"
        "There is no escape hatch.  If you think a test needs the real "
        "production path, that test is wrong — redesign it to use a "
        "tmp_path-backed stand-in."
    )


def connect(db_path: str | Path | None = None) -> sqlite3.Connection:
    """Open (and create) the ledger DB.

    Fail-closed safety net: under pytest, refuses to open the real
    production DB (``DEFAULT_DB``) — see ``_guard_against_live_ledger_in_tests``.
    """
    if db_path is None:
        db_path = DEFAULT_DB
    _guard_against_live_ledger_in_tests(db_path)
    if str(db_path) != ":memory:":
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(str(db_path), timeout=10)
    c.execute("PRAGMA journal_mode=WAL")
    c.execute("PRAGMA busy_timeout=5000")
    c.executescript(DDL)
    return c


def write_verdicts(
    conn: sqlite3.Connection,
    run_id: str,
    as_of: str,
    verdicts: Iterable[Mapping[str, Any]],
) -> int:
    """Append verdicts for a run.  INSERT OR IGNORE keeps the table
    append-only and idempotent.  Returns the number of new rows."""
    rows = []
    for v in verdicts:
        verdict = v["verdict"]
        if verdict not in _VALID_VERDICTS:
            raise ValueError(
                f"invalid verdict {verdict!r} for gate {v.get('gate')!r} "
                f"(scope {v.get('scope')!r}); must be one of {_VALID_VERDICTS}"
            )
        rows.append(
            (run_id, as_of, v["scope"], v["gate"], verdict, v["reason"],
             json.dumps(dict(v.get("inputs", {})), sort_keys=True))
        )
    before = conn.total_changes
    conn.executemany(
        "INSERT OR IGNORE INTO decision_ledger VALUES (?,?,?,?,?,?,?)", rows)
    conn.commit()
    return conn.total_changes - before
