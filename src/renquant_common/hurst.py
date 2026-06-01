"""Hurst exponent — rescaled-range (R/S) memory statistic for time series.

Extracted from ``RenQuant/backtesting/renquant_104/kernel/regime.py`` (lines
97-145 of the canonical 557-line module) into renquant-common per
CLAUDE.md §3.5: shared statistical primitives belong in renquant-common,
not in strategy-specific kernel modules.

Why a focused lift (not full kernel.regime): the umbrella's regime module
imports ``from .config import BULL_CALM, BULL_VOLATILE, CHOPPY, BEAR, REGIMES``
— strategy-specific config that doesn't belong in renquant-common.
``compute_hurst`` and ``rolling_hurst`` are the ONLY functions consumed
by sibling subrepos (renquant-base-data: 2 sites; renquant-pipeline:
indirect via base-data) and they're pure stdlib + numpy/pandas with
zero coupling to the regime detector's config. The remaining 500
lines of kernel.regime (RegimeState, CUSUM, ADX, GMM, detect_regime)
stay in the umbrella where the strategy config lives.

Distinct from ``hmm_regime_labels`` (HMM-based regime detection at the
labelling layer): that module classifies bars into BULL_CALM /
BULL_VOLATILE / BEAR / CHOPPY. ``compute_hurst`` is the underlying
persistence statistic the legacy detector once relied on
(superseded by the vol-based admission rule in detector v2026-05-31).

Reference: Hurst, H.E. (1951). "Long-term storage capacity of reservoirs."
*Transactions of the American Society of Civil Engineers* 116:770-808.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_hurst(returns: np.ndarray, window: int | None = None) -> float:
    """Rescaled-range (R/S) Hurst exponent. Returns H ∈ [0, 1].

    2026-04-24 fixes (carried over from umbrella):
      - chunk loop was off-by-one (``range(0, n - lag, lag)`` skipped the
        trailing ``arr[n-lag:n]``) — now uses ``range(0, n - lag + 1, lag)``.
      - ``lags_used`` was regenerated from ``range(2, 2+len(rs_vals))``,
        misaligning when a particular lag produced no chunks. Now we
        pair ``(lag, rs)`` explicitly.
    """
    arr = returns if window is None else returns[-window:]
    n = len(arr)
    if n < 10:
        return 0.5
    max_lag = min(n // 2, 40)
    lags_used: list[int] = []
    rs_vals: list[float] = []
    for lag in range(2, max_lag):
        chunks = [arr[i:i + lag] for i in range(0, n - lag + 1, lag)]
        rs_chunk: list[float] = []
        for chunk in chunks:
            if len(chunk) < 2:
                continue
            mean = chunk.mean()
            devs = np.cumsum(chunk - mean)
            R = devs.max() - devs.min()
            S = chunk.std(ddof=1)
            if S > 0:
                rs_chunk.append(R / S)
        if rs_chunk:
            lags_used.append(lag)
            rs_vals.append(float(np.mean(rs_chunk)))
    if len(rs_vals) < 2:
        return 0.5
    try:
        poly = np.polyfit(np.log(lags_used), np.log(rs_vals), 1)
        return float(np.clip(poly[0], 0.0, 1.0))
    except Exception:
        return 0.5


def rolling_hurst(returns: pd.Series, window: int = 63) -> pd.Series:
    """Rolling Hurst exponent on a return series."""
    result = pd.Series(index=returns.index, dtype=float)
    arr = returns.values
    for i in range(window, len(arr) + 1):
        result.iloc[i - 1] = compute_hurst(arr[i - window:i])
    return result


__all__ = ["compute_hurst", "rolling_hurst"]
