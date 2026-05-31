"""Regression guard for the HMM regime detector on known golden windows.

Per umbrella ``CLAUDE.md`` §1.4 PRIME DIRECTIVE: detector quality is P0.
Mislabeling propagates into every regime-conditional knob and produces
false NEITHER verdicts on pooled-mean analysis.

This test pins ground truth for 7 historically distinct SPY windows. If
any one of them flips majority regime, the test fails with a diff that
shows which window changed — caller must decide whether the new label
is correct (and update the pin) or fix the detector.

2026-05-31 baseline: vol-based BULL_CALM path added (see hmm_regime_labels
docstring). Pre-fix the hurst-alone gate mislabeled calm_2017 / 2019 H2 /
2021 as BULL_VOLATILE — codex research harness's RegimeDetectorContractTask
flagged calm_2017 specifically.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from renquant_common.contracts.regime import RegimeLabel
from renquant_common.hmm_regime_labels import (
    BULL_CALM_DRIFT_THR,
    BULL_CALM_VOL_THR,
    HURST_TREND_THR,
    compute_hmm_regime_labels,
)

# Local aliases for readability — use the enum's canonical values so the
# repo-wide ``test_no_raw_regime_strings`` guard stays green. The enum is the
# single source of truth (RFC §"Cross-Repo Contracts → RegimeLabel").
_BC = RegimeLabel.BULL_CALM.value
_BV = RegimeLabel.BULL_VOLATILE.value
_BE = RegimeLabel.BEAR.value


# Adjust this path if the test is run outside the standard umbrella layout.
_SPY_CANDIDATE_PATHS = [
    Path("/Users/renhao/git/github/RenQuant/data/ohlcv/SPY/1d.parquet"),
    Path(__file__).resolve().parents[2] / "RenQuant" / "data" / "ohlcv" / "SPY" / "1d.parquet",
]


def _spy_path() -> Path | None:
    for p in _SPY_CANDIDATE_PATHS:
        if p.exists():
            return p
    return None


# (window_name, start, end, expected_majority).
# ``expected_majority`` is one of:
#   * a single regime literal → that regime must be the strict majority
#   * a 2-tuple → either regime accepted (windows with legitimately mixed
#     labels, e.g. 2018 Q4 sell-off had both BEAR and BULL_VOLATILE days)
GOLDEN_WINDOWS: list[tuple[str, str, str, str | tuple[str, ...]]] = [
    ("calm_2017",     "2017-01-01", "2017-12-31", _BC),
    ("2018_q4_sell", "2018-10-01", "2018-12-31", (_BE, _BV)),
    ("2019_h2_calm", "2019-07-01", "2019-12-31", _BC),
    ("covid_crash",  "2020-02-20", "2020-04-30", _BE),
    ("2021_calm",    "2021-04-01", "2021-12-31", _BC),
    ("q2_2022_bear", "2022-04-01", "2022-06-30", _BE),
    ("2023_h2_recov","2023-07-01", "2023-12-31", _BC),
]


def _majority_regime(labels: pd.DataFrame, start: str, end: str) -> str | None:
    mask = (labels["date"] >= pd.Timestamp(start)) & (labels["date"] <= pd.Timestamp(end))
    sub = labels[mask]
    if sub.empty:
        return None
    return str(sub["regime"].value_counts().idxmax())


@pytest.fixture(scope="module")
def hmm_labels() -> pd.DataFrame:
    p = _spy_path()
    if p is None:
        pytest.skip(f"SPY parquet not found at any of {_SPY_CANDIDATE_PATHS}")
    return compute_hmm_regime_labels(p)


@pytest.mark.parametrize("name,start,end,expected", GOLDEN_WINDOWS,
                         ids=[w[0] for w in GOLDEN_WINDOWS])
def test_golden_window_majority_regime(hmm_labels, name, start, end, expected) -> None:
    """Each golden window's majority regime is correctly identified."""
    majority = _majority_regime(hmm_labels, start, end)
    assert majority is not None, f"{name}: no SPY rows in window {start}..{end}"
    if isinstance(expected, tuple):
        assert majority in expected, (
            f"{name}: majority={majority}, expected one of {expected} "
            f"(window {start}..{end})"
        )
    else:
        assert majority == expected, (
            f"{name}: majority={majority}, expected {expected} "
            f"(window {start}..{end})"
        )


def test_calm_2017_bull_calm_dominance(hmm_labels) -> None:
    """calm_2017 must label BULL_CALM ≥ 50% of trading days.

    This is the specific window codex's research harness gates against
    (RegimeDetectorContractTask). Pre-2026-05-31 fix this was 6.8%; the
    vol-based BULL_CALM path lifts it above 50%.
    """
    mask = ((hmm_labels["date"] >= pd.Timestamp("2017-01-01"))
            & (hmm_labels["date"] <= pd.Timestamp("2017-12-31")))
    sub = hmm_labels[mask]
    assert not sub.empty, "no 2017 SPY coverage"
    n = len(sub)
    bull_calm = int((sub["regime"] == _BC).sum())
    pct = 100 * bull_calm / n
    assert pct >= 50.0, (
        f"calm_2017 BULL_CALM coverage = {pct:.1f}% (n={n}); expected ≥ 50%. "
        f"If hurst threshold or BULL_CALM_VOL_THR was tuned, verify the "
        f"detector docstring's empirical contract table still holds."
    )


def test_bear_windows_not_mislabeled_as_bull_calm(hmm_labels) -> None:
    """covid_crash and q2_2022_bear must label < 15% BULL_CALM.

    Guards that the vol-based BULL_CALM path doesn't accidentally admit
    crash windows — drift_20d > 0 should filter, but pin it explicitly.
    """
    for name, start, end in [
        ("covid_crash",  "2020-02-20", "2020-04-30"),
        ("q2_2022_bear", "2022-04-01", "2022-06-30"),
    ]:
        mask = ((hmm_labels["date"] >= pd.Timestamp(start))
                & (hmm_labels["date"] <= pd.Timestamp(end)))
        sub = hmm_labels[mask]
        if sub.empty:
            continue
        n = len(sub)
        bull_calm = int((sub["regime"] == _BC).sum())
        pct = 100 * bull_calm / n
        assert pct < 15.0, (
            f"{name} BULL_CALM = {pct:.1f}% (n={n}); expected < 15%. "
            f"Vol-based BULL_CALM path may be admitting crash windows; "
            f"check BULL_CALM_DRIFT_THR (current: {BULL_CALM_DRIFT_THR})."
        )


def test_threshold_constants_are_finite_and_ordered() -> None:
    """Sanity: thresholds are real numbers and ordered as the docstring claims."""
    assert 0.0 < BULL_CALM_VOL_THR < 0.35  # below BEAR_VOL_20D_THR
    assert -0.05 < BULL_CALM_DRIFT_THR <= 0.05  # near zero
    assert 0.5 < HURST_TREND_THR < 1.0
