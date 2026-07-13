"""Decision ledger persistence — append-only gate-verdict event store.

Moved from renquant-orchestrator to renquant-common (V-003 remediation)
so that both orchestrator and pipeline can import without a reverse
dependency.

One row per (run_id, scope, gate): the verdict a gate returned, its
reason, and the inputs it saw.  Append-only, WAL mode, busy timeout.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Iterable, Mapping

DEFAULT_DB = Path.home() / "renquant-data/decision_ledger.db"

_VALID_VERDICTS = ("allow", "halve", "block")

DDL = """
CREATE TABLE IF NOT EXISTS decision_ledger (
  run_id TEXT NOT NULL, as_of DATE NOT NULL, scope TEXT NOT NULL,
  gate TEXT NOT NULL, verdict TEXT NOT NULL CHECK(verdict IN ('allow','halve','block')),
  reason TEXT NOT NULL, inputs_json TEXT NOT NULL DEFAULT '{}',
  PRIMARY KEY (run_id, scope, gate)
) WITHOUT ROWID;
"""


def connect(db_path: str | Path | None = None) -> sqlite3.Connection:
    """Open (and create) the ledger DB."""
    if db_path is None:
        db_path = DEFAULT_DB
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
