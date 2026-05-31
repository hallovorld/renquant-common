#!/usr/bin/env python
"""Regenerate ``spy_synth.parquet`` — the synthetic OHLCV fixture used by
``tests/test_hmm_regime_golden_windows.py``.

The fixture is committed so tests run in CI without external data. This
script lets you regenerate it after intentional changes (different segment
lengths, different vol/drift parameters) — keep the seed stable so the
detector contract pins stay deterministic.

Run from the repo root::

    python tests/data/_regen_spy_synth.py

Segments produced:
  * dates[0:200]    — 200d calm uptrend  (drift +6 bp/day, vol ~8% ann)
  * dates[200:260]  — 60d  crash         (drift -1.2%/day, vol ~63% ann)
  * dates[260:350]  — 90d  choppy        (vol cluster, near-zero drift)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# Deterministic seed pinned by the regression tests. Change only when the
# fixture's purpose changes — otherwise the BULL_CALM% thresholds shift
# and the test pins drift.
SEED = 20260531
N_CALM = 200
N_CRASH = 60
N_CHOPPY = 90


def _calm_uptrend(rng: np.random.Generator, n: int) -> np.ndarray:
    """drift ~6 bp/day, vol ~0.5%/day → annualized vol ~8%."""
    return rng.normal(0.0006, 0.005, n)


def _crash_period(rng: np.random.Generator, n: int) -> np.ndarray:
    """drift -1.2%/day, vol 4%/day → annualized vol ~63%."""
    return rng.normal(-0.012, 0.04, n)


def _choppy_period(rng: np.random.Generator, n: int) -> np.ndarray:
    """Vol cluster: alternating low-vol and high-vol windows around zero drift."""
    out = []
    for i in range(n):
        vol = 0.015 if (i // 5) % 2 == 0 else 0.005
        out.append(rng.normal(0.0, vol))
    return np.asarray(out)


def regen(out_path: Path) -> Path:
    rng = np.random.default_rng(SEED)
    dates = pd.bdate_range("2025-01-01", periods=N_CALM + N_CRASH + N_CHOPPY)
    rets = np.concatenate([
        _calm_uptrend(rng, N_CALM),
        _crash_period(rng, N_CRASH),
        _choppy_period(rng, N_CHOPPY),
    ])
    prices = 100.0 * np.exp(np.cumsum(rets))
    df = pd.DataFrame({
        "open":   prices * 0.999,
        "high":   prices * 1.003,
        "low":    prices * 0.997,
        "close":  prices,
        "volume": rng.integers(8_000_000, 12_000_000, len(dates)),
    }, index=dates)
    df.index.name = "date"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path)
    return out_path


def main() -> int:
    out = Path(__file__).resolve().parent / "spy_synth.parquet"
    written = regen(out)
    size = written.stat().st_size
    print(f"wrote {written} ({size} bytes)")
    print(f"  calm   : dates[0:{N_CALM}]                  drift=+6bp/day  vol~8%")
    print(f"  crash  : dates[{N_CALM}:{N_CALM + N_CRASH}]         drift=-1.2%/day vol~63%")
    print(f"  choppy : dates[{N_CALM + N_CRASH}:{N_CALM + N_CRASH + N_CHOPPY}]         vol cluster (5d high / 5d low)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
