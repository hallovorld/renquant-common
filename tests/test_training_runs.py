from __future__ import annotations

import datetime as dt
import json
import sqlite3
from pathlib import Path

from renquant_common import record_training_run


def test_record_training_run_writes_sqlite_row_and_jsonl(tmp_path: Path) -> None:
    db = tmp_path / "data" / "sim_runs.db"
    db.parent.mkdir()
    conn = sqlite3.connect(str(db))
    _create_training_runs(conn)

    run_id = record_training_run(
        conn,
        run_date=dt.datetime(2026, 5, 30, 12, 0, 0),
        strategy="renquant_104",
        artifact_type="hf_patchtst",
        config_snapshot={"config_fingerprint": "sha256:config"},
        oos_mean_ic=0.12,
        train_ic=0.34,
        n_rows=100,
        feature_cols=["alpha_1", "alpha_2"],
        artifact_path="/tmp/model.pt",
        elapsed_sec=12.5,
        trigger="unit",
        n_tickers=2,
        n_dates=50,
        n_features=2,
        device="cpu",
        deterministic=False,
        training_window_years=5.0,
        notes="smoke",
        commit_sha="abc1234",
    )

    row = conn.execute(
        """SELECT run_id, run_date, strategy, artifact_type, config_json,
                  oos_mean_ic, train_ic, n_rows, feature_cols, artifact_path,
                  commit_sha, elapsed_sec, trigger, n_tickers, n_dates,
                  n_features, device, deterministic, training_window_years,
                  notes
           FROM training_runs"""
    ).fetchone()
    conn.close()

    assert row is not None
    assert row[0] == run_id
    assert row[1] == "2026-05-30T12:00:00"
    assert row[2:4] == ("renquant_104", "hf_patchtst")
    assert json.loads(row[4]) == {"config_fingerprint": "sha256:config"}
    assert row[5:8] == (0.12, 0.34, 100)
    assert json.loads(row[8]) == ["alpha_1", "alpha_2"]
    assert row[9:17] == ("/tmp/model.pt", "abc1234", 12.5, "unit", 2, 50, 2, "cpu")
    assert row[17:] == (0, 5.0, "smoke")

    log_path = tmp_path / "logs" / "training" / "2026-05-30.jsonl"
    payload = json.loads(log_path.read_text(encoding="utf-8"))
    assert payload["run_id"] == run_id
    assert payload["feature_cols"] == json.dumps(["alpha_1", "alpha_2"])
    assert payload["commit_sha"] == "abc1234"


def test_record_training_run_can_skip_outputs(tmp_path: Path) -> None:
    assert record_training_run(
        None,
        run_date=dt.datetime(2026, 5, 30),
        artifact_type="noop",
        also_log_jsonl=False,
    ).startswith("20260530000000-noop-")


def _create_training_runs(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE training_runs (
            run_id TEXT PRIMARY KEY,
            run_date TIMESTAMP NOT NULL,
            strategy TEXT,
            artifact_type TEXT,
            config_json TEXT,
            oos_mean_ic REAL,
            train_ic REAL,
            n_rows INTEGER,
            feature_cols TEXT,
            artifact_path TEXT,
            commit_sha TEXT,
            elapsed_sec REAL,
            trigger TEXT,
            n_tickers INTEGER,
            n_dates INTEGER,
            n_features INTEGER,
            device TEXT,
            deterministic INTEGER,
            training_window_years REAL,
            notes TEXT
        )
    """)
    conn.commit()
