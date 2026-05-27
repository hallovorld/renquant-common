from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from renquant_common import CombinatorialPurgedCV, PurgedKFold


def _make_panel(n_dates: int = 50, tickers_per_date: int = 4) -> pd.DataFrame:
    dates = pd.date_range("2026-01-01", periods=n_dates, freq="B")
    rows = []
    rng = np.random.default_rng(0)
    for d in dates:
        for i in range(tickers_per_date):
            rows.append({
                "date": d,
                "ticker": f"T{i}",
                "feature": rng.normal(),
                "label": rng.normal(),
            })
    return pd.DataFrame(rows)


def test_purged_kfold_train_and_test_disjoint() -> None:
    panel = _make_panel(n_dates=40)
    cv = PurgedKFold(n_splits=4, embargo_days=2, lookahead_days=2)
    for train_idx, test_idx in cv.split(panel):
        assert set(train_idx).isdisjoint(set(test_idx))


def test_purged_kfold_embargo_invariant() -> None:
    """RFC §5.13.16 invariant: max(train_date) + lookahead < min(test_date),
    AND min(train_date_after_test) > max(test_date) + embargo."""
    panel = _make_panel(n_dates=40)
    lookahead, embargo = 3, 4
    cv = PurgedKFold(n_splits=4, embargo_days=embargo, lookahead_days=lookahead)
    dates_col = pd.to_datetime(panel["date"]).values
    for train_idx, test_idx in cv.split(panel):
        train_dates = dates_col[train_idx]
        test_dates = dates_col[test_idx]
        train_before = train_dates[train_dates < test_dates.min()]
        train_after = train_dates[train_dates > test_dates.max()]
        if len(train_before):
            # max train-before-test + lookahead < min test
            gap_days = (test_dates.min() - train_before.max()).astype("timedelta64[D]").astype(int)
            assert gap_days > lookahead, (
                f"purge invariant violated: gap={gap_days} <= lookahead={lookahead}"
            )
        if len(train_after):
            gap_days = (train_after.min() - test_dates.max()).astype("timedelta64[D]").astype(int)
            assert gap_days > embargo, (
                f"embargo invariant violated: gap={gap_days} <= embargo={embargo}"
            )


def test_purged_kfold_requires_min_splits() -> None:
    panel = _make_panel(n_dates=20)
    with pytest.raises(ValueError, match="n_splits must be >= 2"):
        list(PurgedKFold(n_splits=1).split(panel))


def test_purged_kfold_requires_date_col() -> None:
    panel = _make_panel(n_dates=20).rename(columns={"date": "ts"})
    with pytest.raises(ValueError, match="missing date column"):
        list(PurgedKFold().split(panel))


def test_combinatorial_yields_expected_count() -> None:
    panel = _make_panel(n_dates=60)
    cv = CombinatorialPurgedCV(
        n_splits=6, n_test_groups=2, embargo_days=2, lookahead_days=2
    )
    splits = list(cv.split(panel))
    # C(6, 2) = 15 combinations.
    assert len(splits) == 15


def test_combinatorial_rejects_bad_group_count() -> None:
    panel = _make_panel(n_dates=30)
    with pytest.raises(ValueError, match="n_test_groups must be in"):
        list(CombinatorialPurgedCV(n_splits=5, n_test_groups=0).split(panel))
    with pytest.raises(ValueError, match="n_test_groups must be in"):
        list(CombinatorialPurgedCV(n_splits=5, n_test_groups=5).split(panel))
