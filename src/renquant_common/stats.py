"""Statistical primitives for promotion gating.

Pure functions, no I/O, no model assumptions. Consumed by
``renquant-backtesting`` to produce :class:`AcceptanceReport` instances
(see RFC §"Cross-Repo Contracts" and §"Repository Set" → backtesting).

References:
    Bailey, D.H. & López de Prado, M. (2014). "The Deflated Sharpe Ratio:
        Correcting for Selection Bias, Backtest Overfitting, and
        Non-Normality." *Journal of Portfolio Management*, 40(5), 94-107.
    Bailey, D.H., Borwein, J.M., López de Prado, M., & Zhu, Q.J. (2015).
        "The Probability of Backtest Overfitting." *Journal of
        Computational Finance*, 14(1).
    Newey, W.K. & West, K.D. (1987). "A Simple, Positive Semi-Definite,
        Heteroskedasticity and Autocorrelation Consistent Covariance
        Matrix." *Econometrica*, 55(3), 703-708.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations
from typing import Any

import numpy as np
from scipy import stats as _scipy_stats

from .contracts.regime import RegimeLabel

EULER_MASCHERONI = 0.5772156649015329


@dataclass(frozen=True)
class WilcoxonResult:
    statistic: float
    p_value: float
    n: int


@dataclass(frozen=True)
class HACResult:
    mean: float
    se: float
    t: float
    n: int
    lag: int


def deflated_sharpe(
    returns: Sequence[float],
    n_trials: int,
    *,
    annualization: float = 1.0,
) -> float:
    """Deflated Sharpe Ratio per Bailey-López de Prado (2014).

    Args:
        returns: per-period returns of the *single* best candidate.
        n_trials: number of independent strategy variants considered (how
            many other strategies could have produced this SR by chance).
            ``n_trials == 1`` disables the deflation.
        annualization: factor applied to the raw SR before deflation
            (e.g. ``math.sqrt(252)`` for daily returns).

    Returns:
        Probability ∈ [0, 1] that the candidate's SR exceeds the
        expected-max-of-N-trials benchmark under the null. A common
        actionable threshold is ``DSR > 0.95``.
    """
    x = np.asarray(returns, dtype=float)
    if x.size < 2:
        return float("nan")
    mean = float(x.mean())
    sd = float(x.std(ddof=1))
    if sd == 0.0:
        return float("nan")
    sr = (mean / sd) * annualization
    skew = float(_scipy_stats.skew(x, bias=False))
    kurt = float(_scipy_stats.kurtosis(x, fisher=False, bias=False))
    t = x.size
    if n_trials < 1:
        n_trials = 1
    e_max = _expected_max_sr(n_trials)
    denom_sq = 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr * sr
    denom = math.sqrt(max(1e-12, denom_sq))
    z = (sr - e_max) * math.sqrt(max(0.0, t - 1)) / denom
    return float(_scipy_stats.norm.cdf(z))


def _expected_max_sr(n_trials: int) -> float:
    """E[max of N IID standard normals] — BLP 2014 closed-form approximation."""
    if n_trials <= 1:
        return 0.0
    inv_n = 1.0 / n_trials
    inv_ne = 1.0 / (n_trials * math.e)
    return (1.0 - EULER_MASCHERONI) * float(_scipy_stats.norm.ppf(1.0 - inv_n)) + \
        EULER_MASCHERONI * float(_scipy_stats.norm.ppf(1.0 - inv_ne))


def pbo_cscv(
    returns_matrix: Any,
    *,
    n_partitions: int | None = None,
    max_partitions: int = 16,
) -> float:
    """Probability of Backtest Overfitting via CSCV (Bailey-Borwein-LdP-Zhu 2015).

    Args:
        returns_matrix: 2D array shape ``(n_strategies, n_observations)``.
        n_partitions: even number of equal-sized observation chunks to
            partition into. Defaults to ``min(max_partitions, n_obs)``
            rounded down to even.
        max_partitions: cap on ``n_partitions`` to bound the combinatorial
            cost ``C(n_partitions, n_partitions/2)``.

    Returns:
        PBO ∈ [0, 1]. ``>= 0.5`` → the best in-sample configuration
        typically ranks below median out-of-sample (overfit); ``<= 0.5`` →
        robust.
    """
    R = np.asarray(returns_matrix, dtype=float)
    if R.ndim != 2:
        raise ValueError("returns_matrix must be 2D (n_strategies, n_obs)")
    n_strats, n_obs = R.shape
    if n_strats < 2 or n_obs < 4:
        return float("nan")
    if n_partitions is None:
        np_pick = min(max_partitions, n_obs)
        n_partitions = np_pick if np_pick % 2 == 0 else np_pick - 1
    if n_partitions < 2 or n_partitions % 2 != 0:
        raise ValueError("n_partitions must be an even integer >= 2")
    chunk_size = n_obs // n_partitions
    if chunk_size < 1:
        return float("nan")
    chunk_slices = [
        (i * chunk_size, (i + 1) * chunk_size) for i in range(n_partitions)
    ]
    half = n_partitions // 2
    indices = list(range(n_partitions))
    logits: list[float] = []
    for is_chunks in combinations(indices, half):
        is_set = set(is_chunks)
        oos_chunks = [i for i in indices if i not in is_set]
        is_idx = np.concatenate(
            [np.arange(chunk_slices[i][0], chunk_slices[i][1]) for i in is_chunks]
        )
        oos_idx = np.concatenate(
            [np.arange(chunk_slices[i][0], chunk_slices[i][1]) for i in oos_chunks]
        )
        is_sr = _row_sharpe(R[:, is_idx])
        oos_sr = _row_sharpe(R[:, oos_idx])
        if np.all(np.isnan(is_sr)) or np.all(np.isnan(oos_sr)):
            continue
        best_is = int(np.nanargmax(is_sr))
        # Relative rank in (0, 1]: fraction of OOS SRs <= best-IS's OOS SR.
        ranks = (np.argsort(np.argsort(oos_sr)) + 1) / float(len(oos_sr))
        omega = float(ranks[best_is])
        omega = min(max(omega, 1e-9), 1.0 - 1e-9)
        logits.append(math.log(omega / (1.0 - omega)))
    if not logits:
        return float("nan")
    return float(np.mean(np.asarray(logits) <= 0.0))


def _row_sharpe(returns: np.ndarray) -> np.ndarray:
    mean = returns.mean(axis=1)
    sd = returns.std(axis=1, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        sr = np.where(sd > 0, mean / sd, np.nan)
    return sr


def wilcoxon_signed_rank(
    paired: Sequence[tuple[float, float]],
) -> WilcoxonResult:
    """Wilcoxon signed-rank test for paired observations (two-sided)."""
    arr = np.asarray(paired, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError("paired must be shape (n, 2)")
    diffs = arr[:, 0] - arr[:, 1]
    diffs = diffs[diffs != 0.0]
    if diffs.size < 1:
        return WilcoxonResult(statistic=float("nan"), p_value=float("nan"), n=0)
    stat, p = _scipy_stats.wilcoxon(
        diffs, zero_method="wilcox", alternative="two-sided"
    )
    return WilcoxonResult(
        statistic=float(stat), p_value=float(p), n=int(diffs.size)
    )


def hac_mean(series: Sequence[float], *, lag: int | None = None) -> HACResult:
    """HAC (Newey-West 1987) corrected mean and t-statistic."""
    x = np.asarray(series, dtype=float)
    n = int(x.size)
    if n < 2:
        return HACResult(
            mean=float(x.mean()) if n else float("nan"),
            se=float("nan"),
            t=float("nan"),
            n=n,
            lag=0,
        )
    if lag is None:
        lag = max(1, int(round(n ** (1.0 / 3.0))))
    lag = max(0, min(lag, n - 1))
    mean = float(x.mean())
    centered = x - mean
    gamma0 = float(np.dot(centered, centered) / n)
    var = gamma0
    for l in range(1, lag + 1):
        weight = 1.0 - l / (lag + 1.0)
        gamma_l = float(np.dot(centered[l:], centered[:-l]) / n)
        var += 2.0 * weight * gamma_l
    var = max(var, 1e-18)
    se = math.sqrt(var / n)
    t = mean / se if se > 0 else float("nan")
    return HACResult(mean=mean, se=se, t=t, n=n, lag=lag)


def regime_stratified(
    per_regime: Mapping[RegimeLabel, float],
    *,
    weight: str = "min",
) -> float:
    """Aggregate a per-regime metric into a single number.

    Per PRIME DIRECTIVE, default is ``"min"`` (worst regime) — never use
    pooled-mean as an objective because pooling buries the asymmetric
    catastrophe regime.

    Args:
        per_regime: ``{RegimeLabel: scalar_metric}``.
        weight: ``"min"`` (default), ``"mean"`` (diagnostic only), or
            ``"median"`` (robust).
    """
    if not per_regime:
        return float("nan")
    values = list(per_regime.values())
    if weight == "min":
        return float(min(values))
    if weight == "mean":
        return float(sum(values) / len(values))
    if weight == "median":
        return float(np.median(np.asarray(values, dtype=float)))
    raise ValueError(
        f"unknown weight {weight!r}; valid: 'min', 'mean', 'median'"
    )
