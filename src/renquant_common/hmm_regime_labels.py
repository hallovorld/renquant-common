"""HMM-style per-date regime labels — stateless approximation for IC eval.

Per user 2026-05-18 23:10 mandate: bull_regime_IC uses internal HMM
categorical labels {BULL_CALM, BULL_VOLATILE, BEAR, CHOPPY}, not the
SPY-derived 9-grid. The full stateful detector in
`backtesting/renquant_104/kernel/regime.py::detect_regime` is too heavy
to replay per-date (needs GMM artifact + state + config). This module
provides a stateless approximation reusing the same thresholds.

NOTE: BULL_STRONG appears in some golden config entries but is NOT
emitted by the production detector (only BULL_CALM, BULL_VOLATILE,
CHOPPY, BEAR per kernel/config.py::REGIMES). So `bull_regime_ic`
aggregates {BULL_CALM, BULL_VOLATILE} — the 2 bull labels the detector
actually emits.

Categorical labels (matching kernel/config.py::REGIMES) — version-gated
on the ``detector_version`` parameter:

  - BEAR (all versions): vol_20d > 0.35 OR ret_20d < -0.08
                         OR vol_5d > 0.25 OR ret_5d < -0.04
  - CHOPPY (all versions): vol_5d > vol_60d × 1.5
                           AND |drift_20d| < 0.02 AND not BEAR
  - BULL_CALM
      * "v2026-05-31" (default since 2026-06-01):
        (vol_20d < 0.18 AND drift_20d > 0) OR hurst > 0.65
      * "legacy" (opt-in for pre-flip back-compat):
        hurst > 0.65 only
  - BULL_VOLATILE: everything else

The default flipped 2026-06-01 (RenQuant task #28). Downstream consumers
who haven't migrated their per-regime configs yet pin
``DETECTOR_VERSION_LEGACY`` explicitly to receive the pre-flip behavior.
Sim-parity verified by ``test_default_flip_resolves_calm_to_bull_calm``
per §1.5 promotion methodology.

BULL_CALM detection — 2026-05-31 §1.4 fix
------------------------------------------
Pre-fix the gate was ``hurst > 0.65`` alone. The rescaled-range Hurst statistic
is a memory/persistence signal, NOT a trending-up signal, and for SPY's normal
grind-up regime it hovers around 0.5 regardless of how calm the year is.
Empirical contract check showed even 2017 — the calmest SPY year in modern
history (realized vol = 6.8%) — labeled as BULL_VOLATILE 84% of days under
the hurst-alone rule. Two other historically calm windows (2019 H2 and 2021)
labeled at 1-0% BULL_CALM. The RegimeDetectorContractTask in
``renquant_model_patchtst.research_pipeline`` hard-gated against this.

Fix: add a vol-based BULL_CALM path. ``vol_20d < BULL_CALM_VOL_THR=0.18``
AND positive 20-day drift admits the canonically calm regime. Hurst path
preserved as an OR (rarely fires, but capture any strong-trend windows it
catches). Bear / Choppy overrides keep priority, so the new path can't
mislabel a vol-23% choppy quarter or a vol-67% crash.

Threshold selection (2026-05-31 empirical):

  window           realized_vol_20d   verdict_under_new_rule
  ─────────────────────────────────────────────────────────────────────
  calm_2017        0.07               75.7% BULL_CALM (expected ✓)
  2019 H2 calm     0.12               67.5% BULL_CALM (expected ✓)
  2021 calm        0.12               73.3% BULL_CALM (expected ✓)
  2023 recovery    0.12               67.0% BULL_CALM (mixed ✓)
  2018 Q4 choppy   0.24               9.5%  BULL_CALM (mostly BEAR/CHOPPY ✓)
  covid_crash      0.67               6.0%  BULL_CALM (mostly BEAR ✓)
  q2_2022_bear     0.29               6.5%  BULL_CALM (mostly BEAR ✓)

Thresholds source: kernel/regime.py::detect_regime defaults (2026-05-17
post-detector-fix versions). Vol-based BULL_CALM threshold = 0.18 chosen
from the gap between calm (≤0.13) and choppy (≥0.24) realized vols.
"""
from __future__ import annotations
import math
from pathlib import Path

import numpy as np
import pandas as pd

from .contracts.regime import RegimeLabel


# Thresholds — keep in sync with kernel/regime.py defaults
BEAR_VOL_20D_THR = 0.35
BEAR_RET_20D_THR = -0.08
BEAR_VOL_5D_THR  = 0.25
BEAR_RET_5D_THR  = -0.04
CHOPPY_VOL_RATIO = 1.5
CHOPPY_DRIFT_TH  = 0.02
HURST_TREND_THR  = 0.65
# BULL_CALM vol-based gate (2026-05-31 detector fix — see module docstring).
# Realized 20-day vol below this AND positive 20-day drift admits BULL_CALM
# alongside the legacy hurst path. Calibrated to separate historically calm
# bull windows (≤0.13 vol) from choppy (≥0.24) without mislabeling crashes.
BULL_CALM_VOL_THR   = 0.18
BULL_CALM_DRIFT_THR = 0.0


def _compute_hurst(returns: np.ndarray, window: int = 63) -> float:
    """Simplified rescaled-range Hurst exponent. Returns 0.5 if insufficient."""
    if len(returns) < window:
        return 0.5
    r = returns[-window:]
    mean = r.mean()
    cum = np.cumsum(r - mean)
    R = cum.max() - cum.min()
    S = r.std(ddof=1)
    if S < 1e-9:
        return 0.5
    # Single-window RS heuristic; full Hurst uses multi-window log-log.
    # For BULL_CALM detection, this monotonic transform is sufficient.
    rs = R / (S * math.sqrt(window))
    # Map RS to Hurst-ish in [0, 1]
    return float(np.clip(0.5 + (rs - 1.0) * 0.3, 0.0, 1.0))


DETECTOR_VERSION_LEGACY = "legacy"
DETECTOR_VERSION_V20260531 = "v2026-05-31"
# Default flipped 2026-06-01 (RenQuant task #28): the corrected detector
# v2026-05-31 fixes the calm_2017 mislabel (calm windows being classified
# as BULL_VOLATILE because hurst > 0.65 admission required a trending-up
# pattern that SPY's calm grind doesn't always produce). Library
# consumers who haven't migrated yet pin DETECTOR_VERSION_LEGACY
# explicitly. Production cron parity verified by the sim-parity smoke
# below + the renquant-model research harness's RegimeDetectorContractTask
# which now passes under the default without a --detector-version override.
DETECTOR_VERSION_DEFAULT = DETECTOR_VERSION_V20260531

_KNOWN_VERSIONS = frozenset({DETECTOR_VERSION_LEGACY, DETECTOR_VERSION_V20260531})


def compute_hmm_regime_labels(
    spy_path: Path,
    lookback_days: int = 252,
    *,
    detector_version: str = DETECTOR_VERSION_DEFAULT,
) -> pd.DataFrame:
    """Per-date HMM-style regime label from SPY OHLCV.

    ``detector_version`` selects the BULL_CALM admission rule:

      * ``"v2026-05-31"`` (DEFAULT as of 2026-06-01) — adds a vol-based
        admission alongside the hurst path: BULL_CALM when
        ``hurst > HURST_TREND_THR`` OR
        ``(vol_20d < BULL_CALM_VOL_THR AND drift_20d > BULL_CALM_DRIFT_THR)``.
        Correctly labels 2017/2019/2021 as BULL_CALM majority. Fixes the
        calm_2017 false-negative the legacy detector exhibited.
      * ``"legacy"`` (opt-in, for back-compat) — BULL_CALM only when
        ``hurst > HURST_TREND_THR``. Has the known calm_2017 mislabel
        problem; consumers who haven't migrated their per-regime configs
        yet pin this version explicitly.

    The default flipped per §1.5 promotion methodology: sim-parity
    verified by ``tests/test_detector_default_v2026.py``; production
    cron parity by the renquant-model research harness's
    RegimeDetectorContractTask passing under the new default without
    a ``--detector-version`` CLI override.

    Returns: DataFrame with columns [date, regime] where regime ∈
    {BULL_CALM, BULL_VOLATILE, BEAR, CHOPPY}.
    """
    if detector_version not in _KNOWN_VERSIONS:
        raise ValueError(
            f"unknown detector_version={detector_version!r}; "
            f"expected one of {sorted(_KNOWN_VERSIONS)}"
        )

    spy = pd.read_parquet(spy_path)
    spy.index = pd.to_datetime(spy.index)
    spy = spy.sort_index()
    spy["ret"] = np.log(spy["close"] / spy["close"].shift(1))
    spy = spy.dropna(subset=["ret"])

    out = []
    rets = spy["ret"].values
    dates = spy.index
    for i in range(len(rets)):
        if i < 30:
            out.append({"date": dates[i], "regime": RegimeLabel.BULL_CALM.value})
            continue
        # 20-day stats
        w20 = rets[max(0, i - 20):i]
        vol_20d = float(np.std(w20, ddof=1) * math.sqrt(252)) if len(w20) >= 5 else 0.0
        ret_20d = float(np.prod(1.0 + w20) - 1.0)
        # 5-day stats
        w5 = rets[max(0, i - 5):i]
        vol_5d = float(np.std(w5, ddof=1) * math.sqrt(252)) if len(w5) >= 3 else 0.0
        ret_5d = float(np.prod(1.0 + w5) - 1.0)
        # 60-day vol baseline
        w60 = rets[max(0, i - 60):i]
        vol_60d = float(np.std(w60, ddof=1) * math.sqrt(252)) if len(w60) >= 30 else vol_20d
        # 20-day drift
        drift_20d = ret_20d  # cumulative 20d return as drift signal
        # Hurst
        hurst = _compute_hurst(rets[:i + 1], window=63)

        # BEAR override (highest priority — same across all detector versions)
        if (vol_20d > BEAR_VOL_20D_THR or ret_20d < BEAR_RET_20D_THR or
            vol_5d > BEAR_VOL_5D_THR or ret_5d < BEAR_RET_5D_THR):
            regime = RegimeLabel.BEAR.value
        # CHOPPY: vol cluster + low drift (same across all detector versions)
        elif (vol_60d > 1e-6 and vol_5d > vol_60d * CHOPPY_VOL_RATIO
              and abs(drift_20d) < CHOPPY_DRIFT_TH):
            regime = RegimeLabel.CHOPPY.value
        # BULL_CALM admission — version-gated. ``legacy`` keeps the hurst-only
        # rule for byte-for-byte parity with pre-2026-05-31 consumers; the
        # v2026-05-31 path adds the vol-based fix without removing the hurst
        # branch (so any window that hurst would have admitted still admits).
        elif (
            (
                detector_version == DETECTOR_VERSION_V20260531
                and vol_20d < BULL_CALM_VOL_THR
                and drift_20d > BULL_CALM_DRIFT_THR
            )
            or hurst > HURST_TREND_THR
        ):
            regime = RegimeLabel.BULL_CALM.value
        else:
            regime = RegimeLabel.BULL_VOLATILE.value
        out.append({"date": dates[i], "regime": regime})

    return pd.DataFrame(out)


def per_hmm_regime_ic(preds_df: pd.DataFrame, hmm_labels: pd.DataFrame,
                      min_samples_per_day: int = 5,
                      min_days_per_regime: int = 5) -> dict[str, float]:
    """Compute per-HMM-regime mean cross-sectional Spearman IC.

    Args:
      preds_df: columns date, pred, label (one row per (date, ticker))
      hmm_labels: from compute_hmm_regime_labels()
      min_samples_per_day: skip days with fewer tickers
      min_days_per_regime: regimes with fewer days excluded

    Returns:
      dict regime → mean IC across that regime's days
    """
    from scipy.stats import spearmanr

    preds_df = preds_df.copy()
    preds_df["date"] = pd.to_datetime(preds_df["date"])
    hmm_labels = hmm_labels.copy()
    hmm_labels["date"] = pd.to_datetime(hmm_labels["date"])
    merged = preds_df.merge(hmm_labels, on="date", how="left")

    by_regime: dict[str, list[float]] = {}
    for _, g in merged.groupby("date"):
        if len(g) < min_samples_per_day:
            continue
        x = g["pred"].values
        y = g["label"].values
        if np.std(x) < 1e-8 or np.std(y) < 1e-8:
            continue
        r, _ = spearmanr(x, y)
        if np.isnan(r):
            continue
        regime = g["regime"].iloc[0]
        if pd.isna(regime):
            continue
        by_regime.setdefault(str(regime), []).append(float(r))

    return {regime: float(np.mean(ics))
            for regime, ics in by_regime.items()
            if len(ics) >= min_days_per_regime}


def bull_regime_ic(per_regime: dict[str, float]) -> float:
    """Aggregate {BULL_CALM, BULL_VOLATILE} ICs into single criterion.

    User mandate 2026-05-18 23:10: PatchTST↔XGB swap decision uses
    bull_regime_IC. Returns nan if neither bull regime present.
    """
    bull_ics = [per_regime[r] for r in (RegimeLabel.BULL_CALM.value, RegimeLabel.BULL_VOLATILE.value)
                if r in per_regime]
    if not bull_ics:
        return float("nan")
    return float(np.mean(bull_ics))


__all__ = ["compute_hmm_regime_labels", "per_hmm_regime_ic",
           "bull_regime_ic"]
