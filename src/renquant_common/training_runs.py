"""Shared training-run persistence helpers."""
from __future__ import annotations

import datetime
import json
import sqlite3
import subprocess
import uuid
from pathlib import Path
from typing import Any


def record_training_run(
    conn: sqlite3.Connection | None,
    *,
    run_date: datetime.datetime | None = None,
    strategy: str = "",
    artifact_type: str = "",
    config_snapshot: dict[str, Any] | None = None,
    oos_mean_ic: float | None = None,
    train_ic: float | None = None,
    n_rows: int | None = None,
    feature_cols: list[str] | None = None,
    artifact_path: str | None = None,
    elapsed_sec: float | None = None,
    trigger: str | None = None,
    n_tickers: int | None = None,
    n_dates: int | None = None,
    n_features: int | None = None,
    device: str | None = None,
    deterministic: bool | None = None,
    training_window_years: float | None = None,
    notes: str | None = None,
    also_log_jsonl: bool = True,
    jsonl_dir: Path | None = None,
    commit_sha: str | None = None,
    repo_dir: Path | str | None = None,
) -> str | None:
    """Record one model-training run to SQLite and optional JSONL audit log.

    The SQL column set mirrors the canonical RenQuant ``training_runs`` table.
    Unknown/missing columns are intentionally not tolerated here: callers should
    run their owning schema migration before recording training evidence.
    """
    rd = run_date or datetime.datetime.utcnow()
    run_id = f"{rd.strftime('%Y%m%d%H%M%S')}-{artifact_type}-{uuid.uuid4().hex[:6]}"
    sha = commit_sha if commit_sha is not None else _commit_sha(repo_dir)
    config_json = json.dumps(config_snapshot, default=str) if config_snapshot else None
    feature_cols_json = json.dumps(feature_cols) if feature_cols is not None else None
    deterministic_int = int(deterministic) if deterministic is not None else None

    row = {
        "run_id": run_id,
        "run_date": rd.isoformat(),
        "strategy": strategy,
        "artifact_type": artifact_type,
        "config_json": config_json,
        "oos_mean_ic": oos_mean_ic,
        "train_ic": train_ic,
        "n_rows": n_rows,
        "feature_cols": feature_cols_json,
        "artifact_path": artifact_path,
        "commit_sha": sha,
        "elapsed_sec": elapsed_sec,
        "trigger": trigger,
        "n_tickers": n_tickers,
        "n_dates": n_dates,
        "n_features": n_features,
        "device": device,
        "deterministic": deterministic_int,
        "training_window_years": training_window_years,
        "notes": notes,
    }

    if conn is not None:
        conn.execute(
            """INSERT INTO training_runs
                  (run_id, run_date, strategy, artifact_type, config_json,
                   oos_mean_ic, train_ic, n_rows, feature_cols, artifact_path,
                   commit_sha, elapsed_sec, trigger, n_tickers, n_dates,
                   n_features, device, deterministic, training_window_years,
                   notes)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            tuple(row[col] for col in (
                "run_id",
                "run_date",
                "strategy",
                "artifact_type",
                "config_json",
                "oos_mean_ic",
                "train_ic",
                "n_rows",
                "feature_cols",
                "artifact_path",
                "commit_sha",
                "elapsed_sec",
                "trigger",
                "n_tickers",
                "n_dates",
                "n_features",
                "device",
                "deterministic",
                "training_window_years",
                "notes",
            )),
        )
        conn.commit()

    if also_log_jsonl:
        log_dir = jsonl_dir or _default_training_jsonl_dir(conn)
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"{rd.strftime('%Y-%m-%d')}.jsonl"
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, default=str) + "\n")

    return run_id


def _commit_sha(repo_dir: Path | str | None = None) -> str | None:
    cmd = ["git"]
    if repo_dir is not None:
        cmd.extend(["-C", str(repo_dir)])
    cmd.extend(["rev-parse", "--short", "HEAD"])
    try:
        sha = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=2,
        ).stdout.strip()
    except Exception:
        return None
    return sha or None


def _default_training_jsonl_dir(conn: sqlite3.Connection | None) -> Path:
    if conn is None:
        return Path("logs/training")
    try:
        db_rows = conn.execute("PRAGMA database_list").fetchall()
        main_path = next((row[2] for row in db_rows if row[1] == "main"), "")
    except Exception:
        main_path = ""
    if not main_path:
        return Path("logs/training")
    db_path = Path(main_path)
    root = db_path.parent.parent if db_path.parent.name == "data" else db_path.parent
    return root / "logs" / "training"


__all__ = ["record_training_run"]
