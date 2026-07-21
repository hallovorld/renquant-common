"""Canonical point-in-time walk-forward fold ELIGIBILITY/SELECTION semantics.

This is the single source of truth for "which retrain fold is it PIT-legal to
use for prediction date ``today``" — the exact question
``kernel.walk_forward.loader.WalkForwardModelLoader.entry_as_of`` (RenQuant,
``backtesting/renquant_104/kernel/walk_forward/loader.py``) answers for the
live/sim scoring path, and the exact question a walkforward-sim-DB evidence
extraction pipeline must answer identically to admit historical sim dates as
G4 (multi-model ensemble) evidence.

Why this module exists (Codex review, renquant-model PR #64, G4 walkforward
admissibility): the PR's first cut reimplemented this selection inline in
``experiments/ensemble_phase0/walkforward_admissibility.py`` using
``cutoff_date + datetime.timedelta(lookahead_days)`` — calendar days, and it
ignored ``effective_train_cutoff_date`` entirely. The real loader uses
``effective_train_cutoff_date or cutoff_date`` plus
``pandas.tseries.offsets.BDay(lookahead_days)`` (BUSINESS days). The two
disagree whenever a weekend falls inside the lookahead window (calendar-day
math reaches its boundary date earlier than business-day math), and whenever
a fold declares a pre-embargoed ``effective_train_cutoff_date`` (calendar-day
math ignores it and uses the plain ``cutoff_date`` instead, incorrectly
excluding a fold that was already leakage-safe, or worse: producing WRONG
different eligibility depending on ``cutoff_date`` vs
``effective_train_cutoff_date`` when the two differ). A reimplementation
that "looks equivalent" silently drifts from the runtime contract it is
supposed to mirror; per this repo's CLAUDE.md ("do not add strategy-specific
trading logic here" is the only boundary — the DATE ARITHMETIC contract
itself is domain-neutral), the fix is to have exactly ONE implementation,
imported by both the loader (RenQuant, its own follow-up) and any extraction
harness (renquant-model), not two copies kept manually in sync.

Contract mirrored exactly from ``WalkForwardModelLoader``:

    feature_cutoff_date(cutoff_date, effective_train_cutoff_date) =
        effective_train_cutoff_date or cutoff_date

    safe_last_label_date(cutoff_date, lookahead_days, effective_train_cutoff_date) =
        feature_cutoff_date(...) + BDay(lookahead_days) if lookahead_days > 0
        else feature_cutoff_date(...)

    is_fold_eligible(today, ...) = safe_last_label_date(...) < today   # STRICT

``select_latest_eligible_fold`` additionally mirrors ``entry_as_of``'s
selection rule: among entries eligible for ``today``, the one with the
LATEST ``cutoff_date`` wins (ties are not expected — manifests are built
with monotonically increasing, non-repeating cutoff dates).

All date/timestamp inputs accept anything ``pandas.Timestamp`` can coerce
(ISO date strings, ``datetime.date``, ``datetime.datetime``,
``pandas.Timestamp`` itself).
"""
from __future__ import annotations

from typing import Any, Protocol, Sequence, runtime_checkable

import pandas as pd


@runtime_checkable
class WalkForwardFoldLike(Protocol):
    """Structural type any per-fold record must satisfy for selection.

    Both ``RenQuant``'s ``RetrainEntry`` and ``renquant-model``'s
    ``WalkforwardFold`` already expose these three attributes under these
    exact names — no adapter/wrapper class is required to reuse this module
    from either repo.
    """

    cutoff_date: Any
    lookahead_days: int
    effective_train_cutoff_date: Any | None


def feature_cutoff_date(
    cutoff_date: Any,
    effective_train_cutoff_date: Any | None = None,
) -> pd.Timestamp:
    """The upper-exclusive feature-row cutoff: ``effective_train_cutoff_date
    or cutoff_date``.

    A present ``effective_train_cutoff_date`` means the artifact already
    pre-embargoed labels before the plain selection ``cutoff_date`` — in that
    case it, not ``cutoff_date``, is the correct upper bound on what the
    model has "seen". Empty string is treated the same as ``None`` (the
    manifest JSON round-trip can emit either for "absent").
    """
    if effective_train_cutoff_date is not None and effective_train_cutoff_date != "":
        return pd.Timestamp(effective_train_cutoff_date)
    return pd.Timestamp(cutoff_date)


def safe_last_label_date(
    cutoff_date: Any,
    lookahead_days: int = 0,
    effective_train_cutoff_date: Any | None = None,
) -> pd.Timestamp:
    """The first date this fold is safe to use to score a bar.

    ``lookahead_days`` is a count of BUSINESS days (``pandas.tseries.offsets
    .BDay``), not calendar days: the fold's labels are the forward return
    realised ``lookahead_days`` TRADING sessions after the feature cutoff,
    and trading sessions skip weekends (and, at the exchange-calendar level,
    holidays — BDay does not know about holidays, matching the real loader's
    own approximation exactly, so this module stays a faithful mirror rather
    than a stricter-than-reality "fix").
    """
    base = feature_cutoff_date(cutoff_date, effective_train_cutoff_date)
    lookahead = int(lookahead_days or 0)
    if lookahead > 0:
        return base + pd.tseries.offsets.BDay(lookahead)
    return base


def is_fold_eligible(
    today: Any,
    cutoff_date: Any,
    lookahead_days: int = 0,
    effective_train_cutoff_date: Any | None = None,
) -> bool:
    """True iff this fold may legally score prediction date ``today``.

    STRICT ``<`` — a fold whose safe-last-label-date falls exactly on
    ``today`` is NOT yet safe (mirrors ``WalkForwardModelLoader.entry_as_of``,
    which the original PR's boundary test already got right; kept identical
    here).
    """
    return safe_last_label_date(
        cutoff_date, lookahead_days, effective_train_cutoff_date,
    ) < pd.Timestamp(today)


def select_latest_eligible_fold(
    entries: Sequence[WalkForwardFoldLike],
    today: Any,
) -> WalkForwardFoldLike | None:
    """Return the fold ``entry_as_of(today)``/``model_as_of(today)`` would
    pick, or ``None`` if no fold is eligible yet.

    Mirrors ``WalkForwardModelLoader.entry_as_of``: among entries whose
    ``safe_last_label_date`` is strictly before ``today``, pick the one with
    the latest ``cutoff_date``. Entries need not be pre-sorted; ties on
    ``cutoff_date`` are not expected from a well-formed manifest and are
    resolved by list order (last-in-max wins), matching the real loader's
    ascending-sort-then-take-last behaviour for a manifest with unique
    cutoff dates.
    """
    today_ts = pd.Timestamp(today)
    eligible = [
        e for e in entries
        if safe_last_label_date(
            e.cutoff_date,
            getattr(e, "lookahead_days", 0),
            getattr(e, "effective_train_cutoff_date", None),
        ) < today_ts
    ]
    if not eligible:
        return None
    return max(eligible, key=lambda e: pd.Timestamp(e.cutoff_date))


__all__ = [
    "WalkForwardFoldLike",
    "feature_cutoff_date",
    "safe_last_label_date",
    "is_fold_eligible",
    "select_latest_eligible_fold",
]
