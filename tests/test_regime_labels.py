"""Test `renquant_common.regime_labels` — the lift from
`RenQuant/kernel/regime_labels.py`.

Beyond the standard unit tests, this file pins the **byte-equivalence
invariant**: the lifted module's behaviour on the same SPY input MUST
match the umbrella's original implementation (modulo docstring updates
that explain the lift). Any divergence is either:

  * the umbrella copy is out of sync (consumers should upgrade)
  * the lift introduced a regression (this PR must fix)
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from renquant_common.regime_labels import (
    compute_spy_regime_labels,
    min_across_regimes,
    per_regime_cs_ic,
)


@pytest.fixture
def synthetic_spy(tmp_path: Path) -> Path:
    """Deterministic SPY parquet with enough history for the regime
    rolling windows (60d trend + 252d vol percentile) to populate."""
    rng = np.random.default_rng(20260601)
    n = 400
    dates = pd.bdate_range("2022-01-03", periods=n)
    # Mix three vol regimes so the labeller sees variation: calm
    # first 130 days, normal middle, spiked tail.
    rets = np.concatenate([
        rng.normal(0.0005, 0.005, 130),
        rng.normal(0.0005, 0.012, 130),
        rng.normal(-0.001, 0.025, 140),
    ])
    close = 400.0 * np.exp(np.cumsum(rets))
    spy = pd.DataFrame({"close": close}, index=dates)
    out = tmp_path / "spy.parquet"
    spy.to_parquet(out)
    return out


# ---- Structural tests ---------------------------------------------------


def test_compute_spy_regime_labels_returns_date_regime_frame(synthetic_spy: Path) -> None:
    df = compute_spy_regime_labels(synthetic_spy)
    assert set(df.columns) == {"date", "regime"}
    # Date column is datetime-typed (downstream merges rely on this).
    assert pd.api.types.is_datetime64_any_dtype(df["date"])


def test_regime_labels_use_9_grid_buckets(synthetic_spy: Path) -> None:
    df = compute_spy_regime_labels(synthetic_spy)
    valid = {
        f"{trend}_{vol}"
        for trend in ("LOW", "MED", "HIGH")
        for vol in ("CALM", "NORMAL", "SPIKED")
    }
    # Plus nan_* during the warm-up window before the rolling
    # statistics have enough data.
    seen = set(df["regime"].dropna().unique()) - {"nan_nan"}
    nan_prefixed = {r for r in seen if r.startswith("nan_") or r.endswith("_nan")}
    real = seen - nan_prefixed
    assert real <= valid, f"unexpected regime labels: {real - valid}"
    # The synthetic input spans calm/normal/spiked vol → must see
    # multiple distinct buckets, not collapse to one.
    assert len(real) >= 2


def test_per_regime_cs_ic_filters_undersampled(synthetic_spy: Path) -> None:
    regimes = compute_spy_regime_labels(synthetic_spy)
    # Synthetic preds: 5 tickers × all dates, IC ≈ 0 by construction.
    rng = np.random.default_rng(42)
    rows = []
    for d in regimes["date"]:
        for t in range(5):
            rows.append({
                "date": d,
                "pred": rng.standard_normal(),
                "label": rng.standard_normal(),
            })
    preds = pd.DataFrame(rows)
    ic = per_regime_cs_ic(preds, regimes, min_days_per_regime=20)
    # Only regimes with ≥ 20 days appear (the under-sampled ones are filtered).
    # ICs should be near zero in expectation (random preds vs random labels).
    # Small samples (~20-50 days per regime) can drift to ±0.25 by chance —
    # we're testing the FILTER + the output SHAPE, not statistical purity.
    for regime, v in ic.items():
        assert abs(v) < 0.30, f"regime {regime} IC {v} too far from 0 — broken filter or test"
    assert len(ic) >= 1, "filter must admit at least one regime from the synthetic input"


def test_min_across_regimes_returns_worst() -> None:
    assert min_across_regimes({"A": 0.05, "B": -0.02, "C": 0.10}) == pytest.approx(-0.02)
    assert math.isnan(min_across_regimes({}))


def test_per_regime_cs_ic_filters_warmup_artifacts() -> None:
    """PR #5 reviewer (codex) caught: prior filter only rejected
    ``nan_nan``, letting partial-warmup labels like ``HIGH_nan`` or
    ``nan_CALM`` through. With custom windows or low
    ``min_days_per_regime``, those partial labels become objective
    regimes — Optuna would happily optimize against the warm-up
    window. Filter now rejects ANY regime with a ``nan`` component."""
    dates = pd.bdate_range("2024-01-02", periods=60)
    # Mix 3 valid regimes with 3 warm-up artifact labels — all with
    # enough days to clear the 10-day filter individually.
    regimes_per_date = (
        ["HIGH_CALM"] * 12 + ["MED_NORMAL"] * 12 + ["LOW_SPIKED"] * 12
        + ["nan_nan"] * 8 + ["HIGH_nan"] * 8 + ["nan_CALM"] * 8
    )
    regime_df = pd.DataFrame({"date": dates, "regime": regimes_per_date})

    rng = np.random.default_rng(42)
    rows = []
    for d in dates:
        for t in range(5):
            rows.append({
                "date": d,
                "pred": rng.standard_normal(),
                "label": rng.standard_normal(),
            })
    preds = pd.DataFrame(rows)

    ic = per_regime_cs_ic(preds, regime_df, min_days_per_regime=5)
    # Only the 3 VALID regimes — warm-up artifacts excluded.
    valid_only = {"HIGH_CALM", "MED_NORMAL", "LOW_SPIKED"}
    assert set(ic.keys()) == valid_only, (
        f"warm-up artifacts leaked into per_regime_cs_ic output: "
        f"{set(ic.keys()) - valid_only}")


def test_is_warmup_regime_truth_table() -> None:
    """Pin the warm-up filter contract directly."""
    from renquant_common.regime_labels import _is_warmup_regime
    # Valid 9-grid labels — not warm-up.
    for trend in ("LOW", "MED", "HIGH"):
        for vol in ("CALM", "NORMAL", "SPIKED"):
            assert _is_warmup_regime(f"{trend}_{vol}") is False
    # Warm-up artifacts — must be filtered.
    for w in ("nan_nan", "HIGH_nan", "nan_CALM", "nan_nan", "LOW_nan"):
        assert _is_warmup_regime(w) is True, f"{w} should be warm-up"
    # Pandas NaN-typed regime cell.
    assert _is_warmup_regime(float("nan")) is True
    assert _is_warmup_regime(None) is True
    # Malformed (e.g. extra underscores) — treat as warm-up
    # (fail-closed: an unrecognized shape shouldn't enter objectives).
    assert _is_warmup_regime("HIGH_CALM_extra") is True


# ---- Lift parity: behaviour matches umbrella ---------------------------


@pytest.fixture
def umbrella_module():
    """Import the umbrella's kernel.regime_labels for parity checks.

    Skips the test (not failing) if the umbrella isn't on PYTHONPATH —
    not every CI checkout has access to the umbrella tree.
    """
    umbrella_kernel = Path(__file__).resolve().parents[2] / "RenQuant" / "kernel"
    # CI checks out renquant-common alone; umbrella isn't present then.
    if not umbrella_kernel.exists():
        pytest.skip("umbrella kernel not on disk; skipping parity check")
    sys.path.insert(0, str(umbrella_kernel.parent))
    try:
        from kernel import regime_labels as umbrella_rl  # noqa: PLC0415
        yield umbrella_rl
    finally:
        # Don't pollute other tests' sys.path.
        if str(umbrella_kernel.parent) in sys.path:
            sys.path.remove(str(umbrella_kernel.parent))


def test_compute_spy_regime_labels_matches_umbrella(
    synthetic_spy: Path, umbrella_module
) -> None:
    """The labeller (``compute_spy_regime_labels``) MUST be byte-equivalent
    to the umbrella's original on the same SPY input — including warm-up
    rows where vol percentile hasn't accumulated yet (those are part of
    the function's contract output). Any divergence flags either:

      * an unintentional change in the lift (this PR must fix), OR
      * the umbrella drifted ahead of this copy (consumers must upgrade)

    NB: ``per_regime_cs_ic``'s filter behaviour INTENTIONALLY diverges
    from the umbrella (PR #5 reviewer follow-up — warm-up artifacts
    must not enter objectives). That divergence is tested separately
    in ``test_per_regime_cs_ic_filters_warmup_artifacts``.
    """
    lifted = compute_spy_regime_labels(synthetic_spy)
    umbrella = umbrella_module.compute_spy_regime_labels(synthetic_spy)
    pd.testing.assert_frame_equal(
        lifted.reset_index(drop=True),
        umbrella.reset_index(drop=True),
        check_dtype=True,
    )


def test_min_across_regimes_matches_umbrella(umbrella_module) -> None:
    cases = [
        {"A": 0.05, "B": -0.02, "C": 0.10},
        {},
        {"only": 0.0},
        {"x": float("nan")},
    ]
    for c in cases:
        a = min_across_regimes(c)
        b = umbrella_module.min_across_regimes(c)
        # NaN equality safe-handler.
        if math.isnan(a) and math.isnan(b):
            continue
        assert a == b, f"parity violation on {c!r}: lifted={a} umbrella={b}"
