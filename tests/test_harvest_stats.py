"""Unit tests for metrics.harvest_stats — each pins a bug this session minted."""
import numpy as np
import pandas as pd
from renquant_common.metrics.harvest_stats import (
    per_date_rank_ic, top_n_spread, shuffle_labels_within_date,
    moving_block_ci, paired_clean_series)


def _panel(n_dates=8, n_names=40, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for d in pd.date_range("2024-01-02", periods=n_dates, freq="B"):
        s = rng.normal(size=n_names)
        rows.append(pd.DataFrame({
            "date": d, "ticker": [f"T{i}" for i in range(n_names)],
            "score": s, "label": 0.5 * s + rng.normal(size=n_names)}))
    return pd.concat(rows, ignore_index=True)


def test_rank_ic_positive_on_correlated_and_skips_degenerate():
    df = _panel()
    ic = per_date_rank_ic(df, "score", "label")
    assert len(ic) == 8 and ic.mean() > 0.2
    # degenerate cross-section (all-tied score) must yield NO observation,
    # not a spurious number — the zero-vol screen bug
    dg = df.copy()
    dg.loc[dg.date == dg.date.min(), "score"] = 1.0
    assert len(per_date_rank_ic(dg, "score", "label")) == 7


def test_top_n_spread_sign_and_winsorize_caps_tail():
    df = _panel()
    sp = top_n_spread(df, "score", "label", n=5)
    assert sp.mean() > 0
    big = df.copy()
    first = big[big.date == big.date.min()].nlargest(1, "score").index
    big.loc[first, "label"] = 100.0
    raw = top_n_spread(big, "score", "label", n=5)
    w = top_n_spread(big, "score", "label", n=5, winsorize=0.5)
    assert raw.iloc[0] > w.iloc[0]  # winsorized read caps the lottery point


def test_placebo_preserves_marginals_destroys_alignment():
    df = _panel(n_dates=20)
    pl = shuffle_labels_within_date(df, "label", seed=7)
    d0 = df.date.min()
    assert np.allclose(
        np.sort(df[df.date == d0].label.values),
        np.sort(pl[pl.date == d0].label.values))  # same per-date distribution
    assert per_date_rank_ic(pl, "score", "label").mean() < \
        per_date_rank_ic(df, "score", "label").mean()


def test_moving_block_ci_widens_with_dependence():
    # SAME autocorrelated series: naive block=1 understates the mean's
    # uncertainty; block=60 must widen the CI (the sqrt-h overstatement fix)
    rng = np.random.default_rng(1)
    dep = pd.Series(rng.normal(size=600)).rolling(60, min_periods=1).mean()
    lo1, hi1 = moving_block_ci(dep, block=1, n_boot=2000)
    lo2, hi2 = moving_block_ci(dep, block=60, n_boot=2000)
    assert (hi2 - lo2) > 1.5 * (hi1 - lo1)


def test_paired_clean_uses_common_dates_only():
    a = pd.Series([1.0, 2.0, 3.0],
                  index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]))
    b = pd.Series([0.5, 0.5],
                  index=pd.to_datetime(["2024-01-03", "2024-01-04"]))
    d = paired_clean_series(a, b)
    assert len(d) == 2 and d.iloc[0] == 1.5  # the coverage-cliff guard
