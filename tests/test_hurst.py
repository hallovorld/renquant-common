"""Tests for ``renquant_common.hurst`` — extracted from umbrella
``kernel.regime.compute_hurst`` / ``rolling_hurst``.

Parity invariant: same input MUST produce same output as the umbrella's
original. Any divergence is either an unintentional change in the
extraction (this PR fixes) or umbrella drift ahead of this copy
(consumers upgrade).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from renquant_common.hurst import compute_hurst, rolling_hurst


# ---- Structural tests --------------------------------------------------


def test_compute_hurst_short_input_returns_neutral() -> None:
    """Fewer than 10 observations → return 0.5 (random-walk default)."""
    assert compute_hurst(np.array([0.01, -0.02, 0.005])) == 0.5
    assert compute_hurst(np.array([])) == 0.5


def test_compute_hurst_random_walk_in_unit_interval() -> None:
    """White-noise input produces a valid Hurst exponent in [0, 1].

    Note: this R/S estimator has known finite-sample positive bias (it
    can return values around 0.6-0.7 for true random walks at this
    sample size and max_lag=40). The parity test against the umbrella
    is the authoritative guard — this just checks the function returns
    a valid number, not strict statistical unbiasedness.
    """
    rng = np.random.default_rng(20260601)
    rets = rng.standard_normal(2000) * 0.01
    h = compute_hurst(rets)
    assert 0.0 <= h <= 1.0
    assert h != 0.5, "random-walk input shouldn't hit the short-input fallback"


def test_compute_hurst_trending_series_above_half() -> None:
    """Persistent trend (positive autocorrelation) → Hurst > 0.5.

    Construct an AR(1) with high positive autocorrelation, which is
    persistent in the rescaled-range sense.
    """
    rng = np.random.default_rng(20260601)
    n = 2000
    rets = np.zeros(n)
    eps = rng.standard_normal(n) * 0.01
    phi = 0.7  # AR(1) coef — high positive autocorrelation
    rets[0] = eps[0]
    for i in range(1, n):
        rets[i] = phi * rets[i - 1] + eps[i]
    h = compute_hurst(rets)
    assert h > 0.5, f"trending AR(1) Hurst should be > 0.5; got {h}"


def test_compute_hurst_bounded_in_unit_interval() -> None:
    """Output is clipped to [0, 1]."""
    rng = np.random.default_rng(20260601)
    for _ in range(5):
        rets = rng.standard_normal(500) * 0.02
        h = compute_hurst(rets)
        assert 0.0 <= h <= 1.0


def test_compute_hurst_window_kwarg_subsets_input() -> None:
    """``window=N`` uses only the last N returns."""
    rng = np.random.default_rng(20260601)
    rets = rng.standard_normal(2000) * 0.01
    h_full = compute_hurst(rets)
    h_tail = compute_hurst(rets, window=500)
    # The tail-only Hurst MAY differ from the full-sample Hurst (smaller
    # window → noisier estimate). We're testing the SLICING behaviour:
    # the result must match a fresh call on just the tail.
    h_tail_direct = compute_hurst(rets[-500:])
    assert h_tail == h_tail_direct


def test_rolling_hurst_returns_series_of_correct_length() -> None:
    """Output Series has same index as input; first window-1 values are NaN."""
    rng = np.random.default_rng(20260601)
    n = 200
    rets = pd.Series(rng.standard_normal(n) * 0.01,
                     index=pd.bdate_range("2024-01-02", periods=n))
    out = rolling_hurst(rets, window=63)
    assert len(out) == n
    # First (window-1) values stay NaN — only fills from index window-1 onward.
    assert out.iloc[:62].isna().all()
    # Values from index window-1 onward are populated.
    assert out.iloc[62:].notna().all()


def test_rolling_hurst_warmup_then_populated() -> None:
    """All filled values in [0, 1] (clipped by compute_hurst)."""
    rng = np.random.default_rng(20260601)
    rets = pd.Series(rng.standard_normal(400) * 0.01)
    out = rolling_hurst(rets, window=63)
    filled = out.dropna()
    assert (filled >= 0.0).all() and (filled <= 1.0).all()


# ---- Parity with umbrella ---------------------------------------------


@pytest.fixture
def umbrella_module():
    """Import the umbrella's kernel.regime for parity checks.

    Skips (not fails) if the umbrella tree isn't on disk — CI checks
    out renquant-common alone where the umbrella isn't reachable.
    """
    umbrella_kernel = (Path(__file__).resolve().parents[2]
                       / "RenQuant" / "backtesting" / "renquant_104" / "kernel")
    if not umbrella_kernel.exists():
        pytest.skip("umbrella kernel not on disk; skipping parity check")
    sys_path_entry = str(umbrella_kernel.parent)
    sys.path.insert(0, sys_path_entry)
    cached_keys = [k for k in sys.modules if k == "kernel" or k.startswith("kernel.")]
    cached = {k: sys.modules.pop(k) for k in cached_keys}
    try:
        from kernel import regime as umbrella_reg  # noqa: PLC0415
        yield umbrella_reg
    finally:
        if sys_path_entry in sys.path:
            sys.path.remove(sys_path_entry)
        for k in [m for m in sys.modules if m == "kernel" or m.startswith("kernel.")]:
            sys.modules.pop(k, None)
        for k, mod in cached.items():
            sys.modules.setdefault(k, mod)


def test_compute_hurst_matches_umbrella_on_random_walk(umbrella_module) -> None:
    rng = np.random.default_rng(20260601)
    rets = rng.standard_normal(2000) * 0.01
    assert compute_hurst(rets) == pytest.approx(umbrella_module.compute_hurst(rets))


def test_compute_hurst_matches_umbrella_on_short_input(umbrella_module) -> None:
    short = np.array([0.01, -0.02, 0.005])
    assert compute_hurst(short) == umbrella_module.compute_hurst(short) == 0.5


def test_rolling_hurst_matches_umbrella(umbrella_module) -> None:
    rng = np.random.default_rng(20260601)
    rets = pd.Series(rng.standard_normal(300) * 0.01)
    lifted = rolling_hurst(rets, window=63)
    umbrella = umbrella_module.rolling_hurst(rets, window=63)
    # Compare only the filled portion; both should have identical NaN
    # mask + identical filled values.
    pd.testing.assert_series_equal(lifted, umbrella, check_dtype=True)
