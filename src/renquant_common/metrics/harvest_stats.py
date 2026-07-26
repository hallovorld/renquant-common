"""Harvest-statistic primitives — the single source for research screens.

Extracted from the 2026-07-24/25 research line after five separate
hand-rolled copies each minted its own bug (guard surrogate, anchor
eval-horizon, frozen price denominator, null merge keys, degenerate
cross-sections). Standing rule: screens IMPORT these; they do not re-derive
them. See orchestrator `doc/research/2026-07-24-capacity-and-power-
reconciliation.md` §7 for why the top-N spread (not whole-cross-section IC)
is the harvest-relevant statistic for a top-N book: same data, same blocks —
IC t=1.15 vs DGTW top-10 spread t=2.92.

Complements (does not duplicate) `metrics.block_bootstrap`, which owns
Sharpe/mean CIs via the stationary bootstrap. The moving-block variant here
exists because the research preregs froze MOVING-block inference with a
horizon-matched block length; both are exposed.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

__all__ = [
    "per_date_rank_ic",
    "top_n_spread",
    "shuffle_labels_within_date",
    "moving_block_ci",
    "paired_clean_series",
]


def per_date_rank_ic(df: pd.DataFrame, score_col: str, label_col: str,
                     date_col: str = "date", min_names: int = 5) -> pd.Series:
    """Per-date cross-sectional Spearman IC.

    Degenerate cross-sections (fewer than ``min_names`` rows, or zero
    variance on either side — ties everywhere) yield NO observation rather
    than a spurious value; this is the guard the zero-vol screen lacked.
    """
    out = {}
    for d, g in df[[date_col, score_col, label_col]].dropna().groupby(date_col):
        if (len(g) >= min_names and g[score_col].std() > 0
                and g[label_col].std() > 0):
            out[d] = float(g[score_col].corr(g[label_col], method="spearman"))
    return pd.Series(out, dtype=float).sort_index()


def top_n_spread(df: pd.DataFrame, score_col: str, label_col: str,
                 n: int = 10, date_col: str = "date", min_names: int = 30,
                 winsorize: Optional[float] = None) -> pd.Series:
    """Per-date mean label of the top-``n`` by score, minus the cross mean.

    ``winsorize`` clips the label at ±that value FIRST — the anti-lottery
    read every prereg in this line carries alongside the raw spread.
    """
    out = {}
    for d, g in df[[date_col, score_col, label_col]].dropna().groupby(date_col):
        if len(g) < min_names:
            continue
        v = g[label_col] if winsorize is None else g[label_col].clip(
            -winsorize, winsorize)
        top = g.nlargest(n, score_col).index
        out[d] = float(v.loc[top].mean() - v.mean())
    return pd.Series(out, dtype=float).sort_index()


def shuffle_labels_within_date(df: pd.DataFrame, label_col: str, seed: int,
                               date_col: str = "date") -> pd.DataFrame:
    """Matched placebo: permute the label WITHIN each date (training side only).

    Preserves every marginal (per-date label distribution, feature matrix)
    while destroying the cross-sectional alignment — the placebo convention
    of the 07-24/25 preregs.
    """
    rng = np.random.default_rng(seed)
    out = df.copy()
    out[label_col] = out.groupby(date_col)[label_col].transform(
        lambda s: rng.permutation(s.values))
    return out


def moving_block_ci(x: np.ndarray | pd.Series, block: int,
                    alpha: float = 0.10, n_boot: int = 10_000,
                    seed: int = 20260725) -> tuple[float, float]:
    """Percentile CI on the mean of a serially dependent daily series.

    ``block`` should match the label's overlap horizon (60 for fwd_60d):
    consecutive per-date statistics share up to (h−1)/h of their label
    window, and a naive t-test overstates significance by roughly √h.
    """
    x = np.asarray(pd.Series(x).dropna(), dtype=float)
    n = len(x)
    if n <= block or n == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    starts = np.arange(n - block + 1)
    k = int(np.ceil(n / block))
    means = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.choice(starts, size=k, replace=True)
        means[b] = np.concatenate([x[i:i + block] for i in idx])[:n].mean()
    return (float(np.percentile(means, 100 * alpha / 2)),
            float(np.percentile(means, 100 * (1 - alpha / 2))))


def paired_clean_series(real: pd.Series, placebo: pd.Series) -> pd.Series:
    """clean(d) = real(d) − placebo(d) on common dates (each side per-date).

    The subtraction is per-date so the clean series remains block-bootstrap
    compatible; means of differences over mismatched date sets are NOT
    admissible (that was the coverage-cliff confound in the famA column test).
    """
    c = real.index.intersection(placebo.index)
    return (real[c] - placebo[c]).sort_index()
