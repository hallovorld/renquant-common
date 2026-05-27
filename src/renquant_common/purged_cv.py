"""Purged k-fold and combinatorial-purged cross-validation.

Canonical location per RFC §"Cross-Repo Contracts → PurgedKFold". Every
model family in ``renquant-model`` (and any future model repo) imports
:class:`PurgedKFold` from here; parallel implementations are forbidden
under §5.13.5 (one business decision = one function) and §5.13.16
(walk-forward leakage invariant).

The pinned invariant — ``max(train_date) + lookahead_days < min(val_date)``
for every fold — is verified by ``tests/test_purged_cv.py::test_embargo_invariant``
in this repo's CI.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Callable, Iterator

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


@dataclass
class PurgedKFold:
    """Time-series k-fold with purge + embargo around each test window.

    Attributes:
        n_splits: number of folds (>= 2).
        embargo_days: trading-day count of training rows AFTER each test
            window that get masked out.
        lookahead_days: label-horizon of the training target; training
            rows whose label window reaches into the test window are
            purged.
    """

    n_splits: int = 5
    embargo_days: int = 5
    lookahead_days: int = 5

    def split(
        self,
        panel: pd.DataFrame,
        date_col: str = "date",
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        if self.n_splits < 2:
            raise ValueError("n_splits must be >= 2")
        if date_col not in panel.columns:
            raise ValueError(f"panel missing date column {date_col!r}")
        dates = pd.to_datetime(panel[date_col]).values
        unique_dates = np.array(sorted(set(dates)))
        if len(unique_dates) < self.n_splits:
            raise ValueError(
                f"Not enough unique dates ({len(unique_dates)}) "
                f"for {self.n_splits}-fold CV"
            )
        fold_size = len(unique_dates) // self.n_splits
        fold_edges = [idx * fold_size for idx in range(self.n_splits + 1)]
        fold_edges[-1] = len(unique_dates)
        all_idx = np.arange(len(panel), dtype=np.int64)
        for fold_idx in range(self.n_splits):
            lo, hi = fold_edges[fold_idx], fold_edges[fold_idx + 1]
            test_dates = unique_dates[lo:hi]
            test_mask = np.isin(dates, test_dates)
            train_mask = ~test_mask
            purge_dates = unique_dates[
                max(0, lo - int(self.lookahead_days)):lo
            ]
            embargo_dates = unique_dates[
                hi:min(len(unique_dates), hi + int(self.embargo_days))
            ]
            if len(purge_dates):
                train_mask &= ~np.isin(dates, purge_dates)
            if len(embargo_dates):
                train_mask &= ~np.isin(dates, embargo_dates)
            yield all_idx[train_mask], all_idx[test_mask]


def evaluate_fold_ic(
    model,
    panel: pd.DataFrame,
    feature_cols: list[str],
    label_col: str,
    test_idx: np.ndarray,
    *,
    date_col: str = "date",
) -> pd.Series:
    """Per-date Spearman IC for a fitted model over a test fold."""
    sub = panel.iloc[test_idx]
    preds = model.predict(sub[feature_cols])
    df = pd.DataFrame({
        "date": sub[date_col].values,
        "pred": preds,
        "label": sub[label_col].values,
    })
    out: dict = {}
    for d, group in df.groupby("date", sort=True):
        y = group["label"].to_numpy()
        p = group["pred"].to_numpy()
        if len(y) < 2 or np.allclose(y, y[0]) or np.allclose(p, p[0]):
            continue
        rho, _ = spearmanr(p, y)
        if not np.isnan(rho):
            out[d] = float(rho)
    return pd.Series(out).sort_index()


def cross_validated_ic(
    model_factory: Callable,
    panel: pd.DataFrame,
    feature_cols: list[str],
    label_col: str,
    cv: "PurgedKFold | CombinatorialPurgedCV",
    weight_col: str | None = "weight",
) -> dict:
    """Train+evaluate a model factory across every fold of the splitter."""
    per_fold_mean: list[float] = []
    per_fold_series: list[pd.Series] = []
    for train_idx, test_idx in cv.split(panel):
        model = model_factory()
        train = panel.iloc[train_idx]
        X_train = train[feature_cols]
        y_train = train[label_col].to_numpy(dtype=float)
        weights = (
            train[weight_col].to_numpy(dtype=float)
            if weight_col and weight_col in train
            else None
        )
        finite = np.isfinite(y_train)
        if not finite.all():
            X_train = X_train[finite]
            y_train = y_train[finite]
            if weights is not None:
                weights = weights[finite]
        try:
            model.fit(X_train, y_train, sample_weight=weights)
        except TypeError:
            model.fit(X_train, y_train)
        ic_series = evaluate_fold_ic(
            model, panel, feature_cols, label_col, test_idx
        )
        per_fold_series.append(ic_series)
        if len(ic_series):
            per_fold_mean.append(float(ic_series.mean()))
    arr = np.asarray(per_fold_mean, dtype=float)
    return {
        "mean_ic": float(arr.mean()) if len(arr) else float("nan"),
        "std_ic": float(arr.std(ddof=1)) if len(arr) > 1 else float("nan"),
        "per_fold_ic": per_fold_mean,
        "per_fold_ic_series": per_fold_series,
    }


@dataclass
class CombinatorialPurgedCV:
    """Combinatorial purged CV (López de Prado 2018) — multi-window OOS."""

    n_splits: int = 6
    n_test_groups: int = 2
    embargo_days: int = 5
    lookahead_days: int = 5

    def split(
        self,
        panel: pd.DataFrame,
        date_col: str = "date",
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        if self.n_splits < 2:
            raise ValueError("n_splits must be >= 2")
        if self.n_test_groups < 1 or self.n_test_groups >= self.n_splits:
            raise ValueError("n_test_groups must be in [1, n_splits-1]")
        dates = pd.to_datetime(panel[date_col]).values
        unique_dates = np.array(sorted(set(dates)))
        if len(unique_dates) < self.n_splits:
            raise ValueError(
                f"Not enough unique dates ({len(unique_dates)}) "
                f"for {self.n_splits}-fold CV"
            )
        fold_size = len(unique_dates) // self.n_splits
        fold_edges = [idx * fold_size for idx in range(self.n_splits + 1)]
        fold_edges[-1] = len(unique_dates)
        groups = [
            unique_dates[fold_edges[idx]:fold_edges[idx + 1]]
            for idx in range(self.n_splits)
        ]
        all_idx = np.arange(len(panel), dtype=np.int64)
        for combo in combinations(range(self.n_splits), self.n_test_groups):
            test_dates = np.concatenate([groups[idx] for idx in combo])
            test_mask = np.isin(dates, test_dates)
            train_mask = ~test_mask
            for idx in combo:
                lo, hi = fold_edges[idx], fold_edges[idx + 1]
                purge_dates = unique_dates[
                    max(0, lo - int(self.lookahead_days)):lo
                ]
                embargo_dates = unique_dates[
                    hi:min(len(unique_dates), hi + int(self.embargo_days))
                ]
                if len(purge_dates):
                    train_mask &= ~np.isin(dates, purge_dates)
                if len(embargo_dates):
                    train_mask &= ~np.isin(dates, embargo_dates)
            yield all_idx[train_mask], all_idx[test_mask]


def cross_validated_ic_cpcv(
    model_factory: Callable,
    panel: pd.DataFrame,
    feature_cols: list[str],
    label_col: str,
    cv: CombinatorialPurgedCV,
    weight_col: str | None = "weight",
) -> dict:
    """CombinatorialPurgedCV variant of :func:`cross_validated_ic` with quantiles."""
    result = cross_validated_ic(
        model_factory, panel, feature_cols, label_col, cv, weight_col
    )
    fold_ics = np.asarray(result["per_fold_ic"], dtype=float)
    quantiles = (
        np.quantile(fold_ics, [0.05, 0.25, 0.5, 0.75, 0.95])
        if len(fold_ics)
        else np.full(5, np.nan)
    )
    result["quantiles"] = {
        "q05": float(quantiles[0]),
        "q25": float(quantiles[1]),
        "q50": float(quantiles[2]),
        "q75": float(quantiles[3]),
        "q95": float(quantiles[4]),
    }
    return result
