"""Regression guard for the HMM regime detector on known golden windows.

Per umbrella ``CLAUDE.md`` §1.4 PRIME DIRECTIVE: detector quality is P0.
Mislabeling propagates into every regime-conditional knob and produces
false NEITHER verdicts on pooled-mean analysis.

This test runs two complementary suites:

1. **Synthetic OHLCV fixture** (committed at ``tests/data/spy_synth.parquet``,
   ~22 KB, deterministic seed). Executes in CI without needing the
   umbrella's full SPY history. Validates BOTH detector versions on a
   stylized 350-day series: 200d calm uptrend → 60d crash → 90d choppy.

2. **Real SPY golden windows** (skipped if ``RenQuant/data/ohlcv/SPY/1d.parquet``
   absent). 7 windows from 2017–2023 with known regime majorities; runs
   in the umbrella's local dev environment, skipped in renquant-common CI.

2026-05-31 fix added a vol-based BULL_CALM admission. Default
``detector_version="legacy"`` preserves pre-fix behavior — flipping to
``"v2026-05-31"`` is the operator opt-in.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from renquant_common.contracts.regime import RegimeLabel
from renquant_common.hmm_regime_labels import (
    BULL_CALM_DRIFT_THR,
    BULL_CALM_VOL_THR,
    DETECTOR_VERSION_DEFAULT,
    DETECTOR_VERSION_LEGACY,
    DETECTOR_VERSION_V20260531,
    HURST_TREND_THR,
    compute_hmm_regime_labels,
)


# Aliases for readability — use the enum's canonical values so the
# repo-wide ``test_no_raw_regime_strings`` guard stays green.
_BC = RegimeLabel.BULL_CALM.value
_BV = RegimeLabel.BULL_VOLATILE.value
_BE = RegimeLabel.BEAR.value
_CH = RegimeLabel.CHOPPY.value


# ---- Synthetic fixture (always available; runs in CI) -------------------

_SYNTH = Path(__file__).resolve().parent / "data" / "spy_synth.parquet"

# Segment boundaries baked into tests/data/spy_synth.parquet:
#   dates[0:200]   = 200-day calm uptrend (drift +6bp/day, vol ~8% ann)
#   dates[200:260] = 60-day crash         (drift -1.2%/day, vol ~63% ann)
#   dates[260:350] = 90-day choppy        (vol cluster, near-zero drift)
# We skip the first 30 rows of any segment query because the detector
# bootstraps with BULL_CALM until enough history accumulates.


def _majority(labels: pd.DataFrame, lo: int, hi: int) -> str | None:
    sub = labels.iloc[lo:hi]
    if sub.empty:
        return None
    return str(sub["regime"].value_counts().idxmax())


def test_synth_fixture_exists() -> None:
    assert _SYNTH.exists(), (
        f"missing test fixture at {_SYNTH}. Regenerate via "
        f"`python tests/data/_regen_spy_synth.py` or restore from git."
    )


def test_regen_script_reproduces_committed_fixture(tmp_path: Path) -> None:
    """The regen script must produce the same DATA as the committed fixture,
    so future maintainers can confidently rebuild it after intentional
    seed/segment changes.

    Pre-fix this asserted byte-identical md5. That failed in CI because
    parquet bytes are not deterministic across pyarrow versions /
    platforms (compression flavor, dict encoding, metadata blobs). The
    detector contract only cares about DATA equality, so check data here
    too — what we actually need is that the same numbers come back out,
    not that the same bytes hit the disk.
    """
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent / "data"))
    try:
        import _regen_spy_synth as regen_mod  # type: ignore[import-not-found]
    finally:
        sys.path.pop(0)

    regenerated_path = regen_mod.regen(tmp_path / "spy_synth.parquet")
    committed_df = pd.read_parquet(_SYNTH)
    regen_df = pd.read_parquet(regenerated_path)

    # Index name + column set + dtypes
    assert regen_df.index.name == committed_df.index.name, (
        f"index name drift: committed={committed_df.index.name!r} "
        f"regen={regen_df.index.name!r}"
    )
    assert list(regen_df.columns) == list(committed_df.columns), (
        f"column drift: committed={list(committed_df.columns)} "
        f"regen={list(regen_df.columns)}"
    )

    # Numeric data equality within float-precision tolerance. Across
    # platforms / pyarrow versions, parquet's float64 serialization can
    # introduce last-bit perturbations in a small fraction of rows even
    # when the source numpy arrays were bit-identical. The detector
    # contract only needs values close enough that all derived statistics
    # (vol_20d, drift_20d, hurst) match to detector precision — last-bit
    # noise is well below the BEAR_VOL_20D_THR / BULL_CALM_VOL_THR gates.
    # Keep dtype + column-order + index-name strict; use tight numeric
    # tolerance.
    pd.testing.assert_frame_equal(
        regen_df, committed_df,
        check_dtype=True,
        check_exact=False,
        rtol=1e-9,
        atol=1e-12,
        check_like=False,  # column order matters — it's a fixture, not a table
    )


@pytest.mark.parametrize("version", [DETECTOR_VERSION_LEGACY, DETECTOR_VERSION_V20260531])
def test_synth_crash_segment_labels_bear(version: str) -> None:
    """Both detector versions must label the 60-day synthetic crash as BEAR."""
    labels = compute_hmm_regime_labels(_SYNTH, detector_version=version)
    # Skip the first 10 days of the crash to let 5/20-day windows refresh.
    assert _majority(labels, 210, 260) == _BE, (
        f"detector_version={version}: crash segment must label BEAR; "
        f"check BEAR_VOL_20D_THR / BEAR_RET_*_THR constants"
    )


def test_synth_calm_segment_legacy_underdetects_bull_calm() -> None:
    """Pin the known false-negative: legacy hurst-only rule mislabels even
    a synthetic clean calm uptrend (drift +6bp/day, vol ~8% ann) as
    majority BULL_VOLATILE. Test exists to document the bug — if the
    legacy path ever starts working correctly without the v2026-05-31 fix,
    something else in the detector changed and we want to know."""
    labels = compute_hmm_regime_labels(_SYNTH, detector_version=DETECTOR_VERSION_LEGACY)
    sub = labels.iloc[30:200]
    n = len(sub)
    bull_calm_pct = 100 * int((sub["regime"] == _BC).sum()) / n
    assert bull_calm_pct < 30.0, (
        f"legacy detector unexpectedly labels {bull_calm_pct:.1f}% of "
        f"synthetic calm uptrend as BULL_CALM (n={n}); expected < 30% "
        f"because hurst-only is the known-broken path"
    )


def test_synth_calm_segment_v20260531_labels_bull_calm_majority() -> None:
    """The fix: vol-based path admits the calm segment correctly."""
    labels = compute_hmm_regime_labels(_SYNTH, detector_version=DETECTOR_VERSION_V20260531)
    sub = labels.iloc[30:200]
    n = len(sub)
    bull_calm_pct = 100 * int((sub["regime"] == _BC).sum()) / n
    assert bull_calm_pct >= 50.0, (
        f"v2026-05-31 detector only labels {bull_calm_pct:.1f}% of "
        f"synthetic calm uptrend as BULL_CALM (n={n}); expected ≥ 50%. "
        f"If BULL_CALM_VOL_THR ({BULL_CALM_VOL_THR}) was tightened, "
        f"verify against the synthetic fixture's annualized vol ~8%."
    )


def test_synth_crash_not_admitted_to_bull_calm_by_v20260531() -> None:
    """Drift>0 gate must filter the crash even though the new path is enabled."""
    labels = compute_hmm_regime_labels(_SYNTH, detector_version=DETECTOR_VERSION_V20260531)
    sub = labels.iloc[210:260]
    n = len(sub)
    bull_calm_pct = 100 * int((sub["regime"] == _BC).sum()) / n
    assert bull_calm_pct < 15.0, (
        f"v2026-05-31 detector labels {bull_calm_pct:.1f}% of synthetic "
        f"crash as BULL_CALM; vol+drift gate should filter (drift > "
        f"{BULL_CALM_DRIFT_THR}). Possible BEAR-override regression."
    )


def test_detector_version_default_is_legacy() -> None:
    """Backward-compat invariant: omitting the version arg keeps pre-fix
    behavior, so downstream consumers don't suddenly get re-labeled."""
    labels_explicit = compute_hmm_regime_labels(
        _SYNTH, detector_version=DETECTOR_VERSION_LEGACY,
    )
    labels_default = compute_hmm_regime_labels(_SYNTH)
    pd.testing.assert_frame_equal(labels_explicit, labels_default)
    assert DETECTOR_VERSION_DEFAULT == DETECTOR_VERSION_LEGACY


def test_unknown_detector_version_rejected() -> None:
    """Typos in the opt-in flag must fail loud, not silently fall back."""
    with pytest.raises(ValueError, match="unknown detector_version"):
        compute_hmm_regime_labels(_SYNTH, detector_version="latest")


def test_threshold_constants_are_finite_and_ordered() -> None:
    """Sanity: thresholds are real numbers and ordered as the docstring claims."""
    assert 0.0 < BULL_CALM_VOL_THR < 0.35  # below BEAR_VOL_20D_THR
    assert -0.05 < BULL_CALM_DRIFT_THR <= 0.05  # near zero
    assert 0.5 < HURST_TREND_THR < 1.0


# ---- Real SPY golden windows (skipped if data file absent) --------------

_REAL_SPY_CANDIDATES = [
    Path("/Users/renhao/git/github/RenQuant/data/ohlcv/SPY/1d.parquet"),
    Path(__file__).resolve().parents[2] / "RenQuant" / "data" / "ohlcv" / "SPY" / "1d.parquet",
]

GOLDEN_WINDOWS: list[tuple[str, str, str, str | tuple[str, ...]]] = [
    ("calm_2017",     "2017-01-01", "2017-12-31", _BC),
    ("2018_q4_sell", "2018-10-01", "2018-12-31", (_BE, _BV)),
    ("2019_h2_calm", "2019-07-01", "2019-12-31", _BC),
    ("covid_crash",  "2020-02-20", "2020-04-30", _BE),
    ("2021_calm",    "2021-04-01", "2021-12-31", _BC),
    ("q2_2022_bear", "2022-04-01", "2022-06-30", _BE),
    ("2023_h2_recov","2023-07-01", "2023-12-31", _BC),
]


def _real_spy_path() -> Path | None:
    for p in _REAL_SPY_CANDIDATES:
        if p.exists():
            return p
    return None


@pytest.fixture(scope="module")
def real_hmm_labels_v20260531() -> pd.DataFrame:
    p = _real_spy_path()
    if p is None:
        pytest.skip(f"real SPY parquet not found at any of {_REAL_SPY_CANDIDATES}")
    return compute_hmm_regime_labels(p, detector_version=DETECTOR_VERSION_V20260531)


def _real_window_majority(labels: pd.DataFrame, start: str, end: str) -> str | None:
    mask = (labels["date"] >= pd.Timestamp(start)) & (labels["date"] <= pd.Timestamp(end))
    sub = labels[mask]
    if sub.empty:
        return None
    return str(sub["regime"].value_counts().idxmax())


@pytest.mark.parametrize("name,start,end,expected", GOLDEN_WINDOWS,
                         ids=[w[0] for w in GOLDEN_WINDOWS])
def test_real_spy_golden_window_majority_v20260531(
    real_hmm_labels_v20260531,
    name: str, start: str, end: str,
    expected: str | tuple[str, ...],
) -> None:
    """Under v2026-05-31, each historical golden window labels correctly."""
    majority = _real_window_majority(real_hmm_labels_v20260531, start, end)
    assert majority is not None, f"{name}: no SPY rows in window {start}..{end}"
    if isinstance(expected, tuple):
        assert majority in expected, (
            f"{name}: majority={majority}, expected one of {expected}"
        )
    else:
        assert majority == expected, (
            f"{name}: majority={majority}, expected {expected}"
        )


def test_real_calm_2017_v20260531_bull_calm_dominance(real_hmm_labels_v20260531) -> None:
    """calm_2017 (codex harness's specific gate) must be ≥ 50% BULL_CALM
    under v2026-05-31. Pre-fix this was 6.8%."""
    labels = real_hmm_labels_v20260531
    mask = ((labels["date"] >= pd.Timestamp("2017-01-01"))
            & (labels["date"] <= pd.Timestamp("2017-12-31")))
    sub = labels[mask]
    assert not sub.empty
    n = len(sub)
    bull_calm = int((sub["regime"] == _BC).sum())
    pct = 100 * bull_calm / n
    assert pct >= 50.0, (
        f"calm_2017 BULL_CALM coverage = {pct:.1f}% (n={n}); expected ≥ 50%"
    )
