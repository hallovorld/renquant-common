"""Training-panel freshness + coverage contract (GOAL-5 AC7).

ONE canonical, PURE freshness gate imported by BOTH walk-forward trainers
(PatchTST today; XGB is the AC7 follow-up) so they apply identical
discipline — the same anti-drift rationale as the shared artifact resolver
and :mod:`renquant_common.row_coverage`. A drift here can't be "fixed in one
trainer and not the other".

Problem it closes
-----------------
Today both WF trainers only reject an *empty* post-cutoff window; a
stale-but-nonempty parquet trains **silently**. Each fold slices the panel
with ``date < data_end`` (``data_end = cutoff − 60 business days``), so a
parquet that stops short of a fold's ``data_end`` simply yields fewer rows —
no error, a quietly truncated training set. Callers therefore compute::

    required_through_date = max(fold.data_end for fold in cutoffs)

and call this gate ONCE, BEFORE dispatching any fold. A panel that fails is
a fail-closed abort, never a silent proceed.

Checks (all deterministic, evaluated over the training window
``date < required_through_date`` — the union of every fold's slice):

* **COVERAGE (load-bearing):** ``max(panel.date) >= required_through_date``
  and at least one row lands in the window. A panel that stops short would
  silently truncate the most-recent folds.
* **RECENCY (optional):** if ``max_staleness_days`` is given,
  ``max(panel.date) >= today − max_staleness_days``. Left OFF by default —
  WF corpora legitimately train on historical windows, so calendar-recency
  is NOT the load-bearing check.
* **FLOORS:** ``min_tickers_per_day`` (``PerDayDataset`` silently drops
  <5-ticker days), ``min_rows`` (total), ``max_gap_days`` (largest
  intra-window gap between consecutive trading dates — catches a whole
  chunk dropped during a data rebuild). Each floor is disabled by passing 0
  / ``None``.

Purity: no I/O beyond reading the ``date`` (and, when a floor needs it,
``ticker``) columns of the parquet when a path is passed. Given a DataFrame,
it does zero I/O. Deterministic for a given input.
"""
from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class FreshnessVerdict:
    """Result of :func:`assess_training_panel_freshness`.

    ``ok`` is ``True`` iff ``reasons`` is empty. Every other field is a
    diagnostic the caller can log or persist to a run bundle (the reasons
    are already human-readable and prefixed by which check tripped).
    """

    ok: bool
    reasons: list[str]
    required_through_date: pd.Timestamp | None
    max_date: pd.Timestamp | None
    min_date: pd.Timestamp | None
    n_days: int
    n_rows: int
    n_rows_in_window: int
    min_tickers_per_day_observed: int | None
    max_gap_days_observed: int | None
    n_days_below_ticker_floor: int

    def as_dict(self) -> dict[str, Any]:
        """JSON-able projection (dates → iso strings) for run-bundle records."""
        def _iso(v: pd.Timestamp | None) -> str | None:
            return None if v is None else pd.Timestamp(v).date().isoformat()

        return {
            "ok": self.ok,
            "reasons": list(self.reasons),
            "required_through_date": _iso(self.required_through_date),
            "max_date": _iso(self.max_date),
            "min_date": _iso(self.min_date),
            "n_days": self.n_days,
            "n_rows": self.n_rows,
            "n_rows_in_window": self.n_rows_in_window,
            "min_tickers_per_day_observed": self.min_tickers_per_day_observed,
            "max_gap_days_observed": self.max_gap_days_observed,
            "n_days_below_ticker_floor": self.n_days_below_ticker_floor,
        }


def _load_date_ticker(
    path: str | Path, date_col: str, ticker_col: str
) -> pd.DataFrame:
    """Read ONLY the date (and ticker, if present) columns of a parquet.

    Reading a projected column set keeps the gate cheap even on a
    multi-hundred-MB feature panel — the load-bearing coverage check needs
    only the date column. Falls back to date-only if the ticker column is
    absent so the per-day floor degrades gracefully rather than crashing.
    """
    try:
        return pd.read_parquet(path, columns=[date_col, ticker_col])
    except (ValueError, KeyError):
        return pd.read_parquet(path, columns=[date_col])


def assess_training_panel_freshness(
    panel_or_path: pd.DataFrame | str | Path,
    *,
    required_through_date: Any,
    min_tickers_per_day: int = 0,
    min_rows: int = 0,
    max_gap_days: int | None = None,
    max_staleness_days: int | None = None,
    today: Any = None,
    date_col: str = "date",
    ticker_col: str = "ticker",
) -> FreshnessVerdict:
    """Assess whether a training panel covers + is dense enough to train.

    Parameters
    ----------
    panel_or_path
        A long-form panel DataFrame (one row per ticker/date) OR a path to a
        parquet holding one. When a path is passed only the ``date_col`` and
        ``ticker_col`` columns are read.
    required_through_date
        The latest ``data_end`` any fold needs — callers pass
        ``max(fold.data_end)``. The load-bearing COVERAGE check verifies the
        panel actually reaches this date.
    min_tickers_per_day
        Minimum distinct tickers required on EVERY trading day inside the
        window. 0 disables the check. Days below this would be thin (or, at
        <5, silently dropped by ``PerDayDataset``).
    min_rows
        Minimum TOTAL rows in the panel. 0 disables.
    max_gap_days
        Maximum allowed calendar-day gap between consecutive trading dates in
        the window (weekends/holidays are ≤4d, so e.g. 5 flags a real hole).
        ``None`` disables.
    max_staleness_days
        If given, require ``max(date) >= today − max_staleness_days``. OFF
        (``None``) by default: historical WF windows are legitimate.
    today
        Override for "now" (defaults to today's date). Only used by the
        recency check; making it explicit keeps the function deterministic
        and testable.
    date_col, ticker_col
        Column names (defaults ``"date"`` / ``"ticker"``).

    Returns
    -------
    FreshnessVerdict
        ``ok`` is ``True`` iff no check tripped; ``reasons`` lists every
        breach, each prefixed by the check name.
    """
    required_ts = pd.Timestamp(required_through_date).normalize()

    if isinstance(panel_or_path, (str, Path)):
        df = _load_date_ticker(panel_or_path, date_col, ticker_col)
    else:
        cols = [c for c in (date_col, ticker_col) if c in panel_or_path.columns]
        df = panel_or_path.loc[:, cols].copy()

    if date_col not in df.columns:
        raise KeyError(
            f"assess_training_panel_freshness: panel has no {date_col!r} "
            f"column (columns={list(df.columns)}) — cannot assess coverage"
        )

    reasons: list[str] = []
    dates = pd.to_datetime(df[date_col]).dt.normalize()
    n_rows = int(len(dates))

    if n_rows == 0:
        reasons.append("COVERAGE: panel is empty (0 rows)")
        return FreshnessVerdict(
            ok=False, reasons=reasons, required_through_date=required_ts,
            max_date=None, min_date=None, n_days=0, n_rows=0,
            n_rows_in_window=0, min_tickers_per_day_observed=None,
            max_gap_days_observed=None, n_days_below_ticker_floor=0,
        )

    max_date = dates.max()
    min_date = dates.min()
    unique_dates = pd.DatetimeIndex(dates.unique()).sort_values()
    n_days = int(len(unique_dates))

    in_window = dates < required_ts
    n_rows_in_window = int(in_window.sum())
    has_ticker = ticker_col in df.columns

    # 1. COVERAGE (load-bearing) --------------------------------------------
    if max_date < required_ts:
        reasons.append(
            f"COVERAGE: panel ends {max_date.date()} but training needs data "
            f"through required_through_date {required_ts.date()} — the most "
            f"recent folds would be silently truncated"
        )
    if n_rows_in_window == 0:
        reasons.append(
            f"COVERAGE: no rows before required_through_date "
            f"{required_ts.date()} (0 training rows in window; panel spans "
            f"{min_date.date()}..{max_date.date()})"
        )

    # 2. RECENCY (optional) --------------------------------------------------
    if max_staleness_days is not None:
        today_ts = (
            pd.Timestamp(today).normalize() if today is not None
            else pd.Timestamp(_dt.date.today())
        )
        stale_before = today_ts - pd.Timedelta(days=int(max_staleness_days))
        if max_date < stale_before:
            reasons.append(
                f"RECENCY: panel ends {max_date.date()}, older than "
                f"today {today_ts.date()} − {int(max_staleness_days)}d "
                f"= {stale_before.date()}"
            )

    # 3a. total-row floor ----------------------------------------------------
    if min_rows and n_rows < int(min_rows):
        reasons.append(
            f"FLOOR: total rows {n_rows} < min_rows {int(min_rows)}"
        )

    window_dates = unique_dates[unique_dates < required_ts]

    # 3b. per-day ticker floor (within window) ------------------------------
    min_tickers_observed: int | None = None
    n_days_below = 0
    if min_tickers_per_day and has_ticker and n_rows_in_window > 0:
        wdf = df.loc[in_window, [date_col, ticker_col]].copy()
        wdf[date_col] = dates[in_window].values
        per_day = wdf.groupby(date_col)[ticker_col].nunique()
        min_tickers_observed = int(per_day.min())
        below = per_day[per_day < int(min_tickers_per_day)]
        n_days_below = int(len(below))
        if n_days_below > 0:
            worst = below.sort_values().head(3)
            worst_str = ", ".join(
                f"{pd.Timestamp(d).date()}:{int(c)}" for d, c in worst.items()
            )
            reasons.append(
                f"FLOOR: {n_days_below} day(s) in window below "
                f"min_tickers_per_day {int(min_tickers_per_day)} "
                f"(worst {worst_str}; min observed {min_tickers_observed})"
            )
    elif min_tickers_per_day and not has_ticker:
        reasons.append(
            f"FLOOR: min_tickers_per_day {int(min_tickers_per_day)} requested "
            f"but panel has no {ticker_col!r} column to verify per-day coverage"
        )

    # 3c. max intra-window date gap -----------------------------------------
    # Include the first date on/after required_through_date so a hole right
    # before the boundary (which truncates the last fold even when
    # max(date) >= required) is measured, not just interior gaps.
    max_gap_observed: int | None = None
    if max_gap_days is not None:
        after = unique_dates[unique_dates >= required_ts]
        gap_dates = window_dates
        if len(after):
            gap_dates = window_dates.append(after[:1])
        if len(gap_dates) >= 2:
            gaps = gap_dates.to_series().diff().dropna().dt.days
            max_gap_observed = int(gaps.max())
            if max_gap_observed > int(max_gap_days):
                worst_end = gaps.idxmax()
                reasons.append(
                    f"FLOOR: max intra-window date gap {max_gap_observed}d > "
                    f"max_gap_days {int(max_gap_days)} (largest gap ends "
                    f"{pd.Timestamp(worst_end).date()})"
                )

    return FreshnessVerdict(
        ok=(len(reasons) == 0),
        reasons=reasons,
        required_through_date=required_ts,
        max_date=max_date,
        min_date=min_date,
        n_days=n_days,
        n_rows=n_rows,
        n_rows_in_window=n_rows_in_window,
        min_tickers_per_day_observed=min_tickers_observed,
        max_gap_days_observed=max_gap_observed,
        n_days_below_ticker_floor=n_days_below,
    )
