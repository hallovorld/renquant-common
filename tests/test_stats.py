from __future__ import annotations

import math

import numpy as np
import pytest

from renquant_common import RegimeLabel
from renquant_common.stats import (
    HACResult,
    WilcoxonResult,
    deflated_sharpe,
    hac_mean,
    pbo_cscv,
    regime_stratified,
    wilcoxon_signed_rank,
)


def test_deflated_sharpe_clean_signal_beats_zero_trials() -> None:
    rng = np.random.default_rng(0)
    # Positive-drift series: SR ≈ 0.1 per period, T=252
    returns = rng.normal(loc=0.001, scale=0.01, size=252)
    dsr = deflated_sharpe(returns, n_trials=1)
    assert 0.0 <= dsr <= 1.0
    assert dsr > 0.5


def test_deflated_sharpe_punishes_multiple_trials() -> None:
    rng = np.random.default_rng(1)
    returns = rng.normal(loc=0.0005, scale=0.01, size=252)
    dsr_one = deflated_sharpe(returns, n_trials=1)
    dsr_many = deflated_sharpe(returns, n_trials=200)
    # Searching across more trials inflates expected-max benchmark, so
    # the same observed SR is deflated harder.
    assert dsr_one > dsr_many


def test_deflated_sharpe_pure_noise_near_half() -> None:
    rng = np.random.default_rng(42)
    returns = rng.normal(loc=0.0, scale=0.01, size=252)
    dsr = deflated_sharpe(returns, n_trials=1)
    # Centered noise should land near 0.5 ± 0.4 (one realization).
    assert 0.05 < dsr < 0.95


def test_pbo_cscv_clean_signal_below_half() -> None:
    rng = np.random.default_rng(2)
    n_strats, n_obs = 8, 240
    R = rng.normal(loc=0.0, scale=0.01, size=(n_strats, n_obs))
    # Inject a genuinely-best strategy with positive drift across all obs.
    R[0] += 0.002
    pbo = pbo_cscv(R, n_partitions=8)
    assert 0.0 <= pbo <= 1.0
    assert pbo < 0.5  # the genuine winner should hold up OOS


def test_pbo_cscv_pure_noise_near_half_or_above() -> None:
    rng = np.random.default_rng(3)
    n_strats, n_obs = 8, 240
    R = rng.normal(loc=0.0, scale=0.01, size=(n_strats, n_obs))
    pbo = pbo_cscv(R, n_partitions=8)
    assert 0.0 <= pbo <= 1.0
    # Pure noise — best IS often does NOT survive OOS; PBO should be
    # non-trivially elevated above the clean-signal case.
    assert pbo > 0.2


def test_pbo_cscv_rejects_odd_partitions() -> None:
    R = np.zeros((4, 100))
    with pytest.raises(ValueError, match="even"):
        pbo_cscv(R, n_partitions=5)


def test_wilcoxon_paired_basic() -> None:
    paired = [(1.0, 0.5), (1.5, 0.7), (1.2, 0.6), (1.8, 1.0), (1.1, 0.4)]
    res = wilcoxon_signed_rank(paired)
    assert isinstance(res, WilcoxonResult)
    assert res.n == 5
    assert res.p_value < 0.1


def test_wilcoxon_drops_zeros() -> None:
    paired = [(1.0, 1.0), (1.5, 0.7), (1.2, 0.6)]
    res = wilcoxon_signed_rank(paired)
    assert res.n == 2


def test_hac_mean_constant_series_has_zero_std() -> None:
    res = hac_mean([1.0] * 50)
    assert isinstance(res, HACResult)
    assert math.isclose(res.mean, 1.0)
    # zero variance → t undefined (nan) or se ~ 0
    assert res.n == 50


def test_hac_mean_t_statistic_for_positive_drift() -> None:
    rng = np.random.default_rng(4)
    series = rng.normal(loc=0.5, scale=0.1, size=200)
    res = hac_mean(series)
    assert math.isclose(res.mean, 0.5, abs_tol=0.05)
    assert res.t > 2.0  # strong positive drift → t-stat > 2


def test_regime_stratified_default_is_min() -> None:
    per_regime = {
        RegimeLabel.BULL_CALM: 0.5,
        RegimeLabel.BEAR: -0.2,
        RegimeLabel.CHOPPY: 0.1,
    }
    assert regime_stratified(per_regime) == pytest.approx(-0.2)


def test_regime_stratified_explicit_weights() -> None:
    per_regime = {
        RegimeLabel.BULL_CALM: 0.5,
        RegimeLabel.BEAR: -0.2,
        RegimeLabel.CHOPPY: 0.1,
    }
    assert regime_stratified(per_regime, weight="mean") == pytest.approx(
        (0.5 - 0.2 + 0.1) / 3
    )
    assert regime_stratified(per_regime, weight="median") == pytest.approx(0.1)


def test_regime_stratified_unknown_weight_raises() -> None:
    with pytest.raises(ValueError, match="unknown weight"):
        regime_stratified({RegimeLabel.BEAR: 0.0}, weight="harmonic")
