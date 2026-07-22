"""Unit tests for the canonical training-panel freshness gate (GOAL-5 AC7).

One PURE contract, imported by both WF trainers, must fail-closed on the
four breach classes (short coverage / gappy window / thin-ticker day /
too-few rows) and the optional recency check, and pass a healthy panel.
The path-input branch is exercised against a real tmp parquet so the
"read only date+ticker columns" loader is covered too.
"""
from __future__ import annotations

import pandas as pd
import pytest

from renquant_common.training_freshness import (
    FreshnessVerdict,
    assess_training_panel_freshness,
)

REQUIRED = pd.Timestamp("2025-06-02")  # a Monday


def _panel(dates: list[pd.Timestamp], n_tickers: int = 25) -> pd.DataFrame:
    """Long-form (date, ticker) panel with ``n_tickers`` names on each date."""
    rows = []
    for d in dates:
        for i in range(n_tickers):
            rows.append({"date": d, "ticker": f"T{i:03d}"})
    return pd.DataFrame(rows)


def _bdays(start: str, end: str) -> list[pd.Timestamp]:
    return list(pd.bdate_range(start, end))


# ── PASS: a healthy panel that covers the window, dense, gapless ────────────

def test_good_panel_passes() -> None:
    # Covers through 2025-06-06 (> REQUIRED), 25 tickers/day, business-day
    # cadence (≤3d weekend gaps), well above every floor.
    dates = _bdays("2025-01-01", "2025-06-06")
    verdict = assess_training_panel_freshness(
        _panel(dates, n_tickers=25),
        required_through_date=REQUIRED,
        min_tickers_per_day=20,
        min_rows=100,
        max_gap_days=5,
    )
    assert isinstance(verdict, FreshnessVerdict)
    assert verdict.ok, verdict.reasons
    assert verdict.reasons == []
    assert verdict.max_date >= REQUIRED
    assert verdict.n_rows_in_window > 0
    assert verdict.min_tickers_per_day_observed == 25


# ── FAIL: short coverage (load-bearing check) ──────────────────────────────

def test_short_coverage_fails() -> None:
    # Panel stops 2025-05-15, well short of REQUIRED (2025-06-02).
    dates = _bdays("2025-01-01", "2025-05-15")
    verdict = assess_training_panel_freshness(
        _panel(dates),
        required_through_date=REQUIRED,
        min_tickers_per_day=20,
        max_gap_days=5,
    )
    assert not verdict.ok
    assert any(r.startswith("COVERAGE") for r in verdict.reasons), verdict.reasons
    assert verdict.max_date < REQUIRED


def test_window_entirely_after_required_fails_coverage() -> None:
    # max(date) > REQUIRED (so the naive max-date check passes) but every row
    # is on/after REQUIRED → zero training rows in the window.
    dates = _bdays("2025-06-02", "2025-07-01")
    verdict = assess_training_panel_freshness(
        _panel(dates),
        required_through_date=REQUIRED,
        max_gap_days=5,
    )
    assert not verdict.ok
    assert verdict.n_rows_in_window == 0
    assert any("no rows before" in r for r in verdict.reasons), verdict.reasons


# ── FAIL: a gap (missing chunk) inside the window ──────────────────────────

def test_gappy_window_fails() -> None:
    # Two dense stretches with a ~1-month hole between them, all before
    # REQUIRED and extending past it so coverage itself is satisfied.
    dates = _bdays("2025-01-01", "2025-02-01") + _bdays("2025-03-05", "2025-06-06")
    verdict = assess_training_panel_freshness(
        _panel(dates),
        required_through_date=REQUIRED,
        min_tickers_per_day=20,
        max_gap_days=5,
    )
    assert not verdict.ok
    assert verdict.max_date >= REQUIRED  # coverage OK; the GAP is what fails
    assert any("gap" in r for r in verdict.reasons), verdict.reasons
    assert verdict.max_gap_days_observed > 5


def test_boundary_gap_before_required_fails() -> None:
    # Dense until mid-April then nothing until AFTER required → coverage
    # (max_date) passes but the last fold is truncated; the boundary gap
    # inclusion must catch it.
    dates = _bdays("2025-01-01", "2025-04-15") + _bdays("2025-06-03", "2025-06-10")
    verdict = assess_training_panel_freshness(
        _panel(dates),
        required_through_date=REQUIRED,
        max_gap_days=5,
    )
    assert not verdict.ok
    assert verdict.max_date >= REQUIRED
    assert any("gap" in r for r in verdict.reasons), verdict.reasons


# ── FAIL: a thin-ticker day ────────────────────────────────────────────────

def test_thin_ticker_day_fails() -> None:
    dates = _bdays("2025-01-01", "2025-06-06")
    panel = _panel(dates, n_tickers=25)
    # Starve one in-window day down to 3 tickers.
    thin_day = pd.Timestamp("2025-03-03")
    keep = ~((panel["date"] == thin_day) & (panel["ticker"] >= "T003"))
    panel = panel[keep].reset_index(drop=True)
    verdict = assess_training_panel_freshness(
        panel,
        required_through_date=REQUIRED,
        min_tickers_per_day=20,
        max_gap_days=5,
    )
    assert not verdict.ok
    assert verdict.min_tickers_per_day_observed == 3
    assert verdict.n_days_below_ticker_floor == 1
    assert any("min_tickers_per_day" in r for r in verdict.reasons), verdict.reasons


def test_thin_ticker_floor_disabled_when_zero() -> None:
    dates = _bdays("2025-01-01", "2025-06-06")
    panel = _panel(dates, n_tickers=3)  # only 3 tickers/day
    verdict = assess_training_panel_freshness(
        panel,
        required_through_date=REQUIRED,
        min_tickers_per_day=0,  # disabled
        max_gap_days=5,
    )
    assert verdict.ok, verdict.reasons


# ── FAIL: too few total rows ───────────────────────────────────────────────

def test_min_rows_floor_fails() -> None:
    dates = _bdays("2025-01-01", "2025-06-06")
    panel = _panel(dates, n_tickers=25)
    verdict = assess_training_panel_freshness(
        panel,
        required_through_date=REQUIRED,
        min_rows=10_000_000,  # unreachable
    )
    assert not verdict.ok
    assert any("min_rows" in r for r in verdict.reasons), verdict.reasons


# ── recency: off by default, fires only when max_staleness_days is set ──────

def test_recency_off_by_default_passes_historical_window() -> None:
    dates = _bdays("2025-01-01", "2025-06-06")
    verdict = assess_training_panel_freshness(
        _panel(dates),
        required_through_date=REQUIRED,
        # no max_staleness_days → historical window is fine
    )
    assert verdict.ok, verdict.reasons


def test_recency_fires_when_stale() -> None:
    dates = _bdays("2025-01-01", "2025-06-06")
    verdict = assess_training_panel_freshness(
        _panel(dates),
        required_through_date=REQUIRED,
        max_staleness_days=30,
        today=pd.Timestamp("2025-12-01"),  # panel is ~6 months old
    )
    assert not verdict.ok
    assert any(r.startswith("RECENCY") for r in verdict.reasons), verdict.reasons


def test_recency_passes_when_fresh() -> None:
    dates = _bdays("2025-01-01", "2025-06-06")
    verdict = assess_training_panel_freshness(
        _panel(dates),
        required_through_date=REQUIRED,
        max_staleness_days=30,
        today=pd.Timestamp("2025-06-10"),  # within 30d of max_date
    )
    assert verdict.ok, verdict.reasons


# ── empty panel ─────────────────────────────────────────────────────────────

def test_empty_panel_fails() -> None:
    empty = pd.DataFrame({"date": pd.Series([], dtype="datetime64[ns]"),
                          "ticker": pd.Series([], dtype=object)})
    verdict = assess_training_panel_freshness(
        empty, required_through_date=REQUIRED, min_tickers_per_day=20,
    )
    assert not verdict.ok
    assert verdict.n_rows == 0
    assert any("empty" in r for r in verdict.reasons), verdict.reasons


# ── path input: read only date+ticker columns from a real parquet ──────────

def test_path_input_matches_dataframe(tmp_path) -> None:
    dates = _bdays("2025-01-01", "2025-06-06")
    panel = _panel(dates, n_tickers=25)
    # Add a wide feature column the gate must NOT need to read.
    panel["some_feature"] = 1.23
    pq = tmp_path / "panel.parquet"
    panel.to_parquet(pq)

    from_df = assess_training_panel_freshness(
        panel, required_through_date=REQUIRED,
        min_tickers_per_day=20, max_gap_days=5,
    )
    from_path = assess_training_panel_freshness(
        str(pq), required_through_date=REQUIRED,
        min_tickers_per_day=20, max_gap_days=5,
    )
    assert from_df.ok and from_path.ok
    assert from_df.as_dict() == from_path.as_dict()


def test_path_input_short_coverage_fails(tmp_path) -> None:
    dates = _bdays("2025-01-01", "2025-05-15")
    pq = tmp_path / "short.parquet"
    _panel(dates).to_parquet(pq)
    verdict = assess_training_panel_freshness(
        pq, required_through_date=REQUIRED, min_tickers_per_day=20,
    )
    assert not verdict.ok
    assert any(r.startswith("COVERAGE") for r in verdict.reasons)


# ── missing ticker column: date-only assessment still works, floor degrades ─

def test_missing_ticker_column_degrades_gracefully() -> None:
    dates = _bdays("2025-01-01", "2025-06-06")
    panel = pd.DataFrame({"date": dates})  # no ticker column
    # Coverage still assessable; requesting a ticker floor reports it can't.
    verdict = assess_training_panel_freshness(
        panel, required_through_date=REQUIRED, min_tickers_per_day=20,
    )
    assert not verdict.ok
    assert any("no 'ticker' column" in r for r in verdict.reasons), verdict.reasons

    # Without a ticker floor, a ticker-less date panel passes coverage.
    ok_verdict = assess_training_panel_freshness(
        panel, required_through_date=REQUIRED,
    )
    assert ok_verdict.ok, ok_verdict.reasons
