"""Tests for ``renquant_common.walk_forward_fold_selection`` — the canonical
point-in-time fold eligibility/selection contract mirrored from
``WalkForwardModelLoader.entry_as_of`` (RenQuant
``backtesting/renquant_104/kernel/walk_forward/loader.py``).

Boundary coverage per Codex review (renquant-model PR #64): a Friday/weekend
prediction date must select using BUSINESS-day lookahead (not calendar-day
arithmetic, which lands on a different date whenever a weekend falls inside
the lookahead window), and ``effective_train_cutoff_date`` must be preferred
over ``cutoff_date`` when both are present and differ.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import pytest

from renquant_common.walk_forward_fold_selection import (
    feature_cutoff_date,
    is_fold_eligible,
    safe_last_label_date,
    select_latest_eligible_fold,
)


@dataclass(frozen=True)
class _Fold:
    cutoff_date: str
    lookahead_days: int = 0
    effective_train_cutoff_date: str | None = None


class TestFeatureCutoffDate:
    def test_falls_back_to_cutoff_date_when_effective_absent(self):
        assert feature_cutoff_date("2024-01-02") == pd.Timestamp("2024-01-02")

    def test_prefers_effective_train_cutoff_date_when_present(self):
        # effective_train_cutoff_date pre-embargoes labels BEFORE the plain
        # selection cutoff -- it, not cutoff_date, is the true upper bound.
        assert feature_cutoff_date(
            "2024-01-02", "2023-11-01",
        ) == pd.Timestamp("2023-11-01")

    def test_empty_string_treated_as_absent(self):
        assert feature_cutoff_date("2024-01-02", "") == pd.Timestamp("2024-01-02")


class TestSafeLastLabelDateBusinessDays:
    def test_zero_lookahead_is_the_feature_cutoff_itself(self):
        assert safe_last_label_date("2024-01-02", 0) == pd.Timestamp("2024-01-02")

    def test_business_day_offset_not_calendar_day(self):
        # 2023-12-01 is a FRIDAY. +1 business day = 2023-12-04 (Monday), NOT
        # 2023-12-02 (the calendar-day answer, a Saturday). This is exactly
        # the bug Codex found: the PR's first cut used
        # `cutoff_date + datetime.timedelta(lookahead_days)`.
        friday = "2023-12-01"
        assert pd.Timestamp(friday).day_name() == "Friday"
        bday_result = safe_last_label_date(friday, 1)
        calendar_day_result = pd.Timestamp(friday) + pd.Timedelta(days=1)
        assert bday_result == pd.Timestamp("2023-12-04")
        assert calendar_day_result == pd.Timestamp("2023-12-02")  # a Saturday
        assert bday_result != calendar_day_result

    def test_multi_business_day_lookahead_skips_two_weekends(self):
        # 60 business days from a Friday should land ~84 calendar days later
        # (60 + 2*(60//5) weekend days, roughly) -- strictly more than 60
        # calendar days out, so a calendar-day mirror is systematically TOO
        # EARLY (over-admits) relative to the real business-day contract.
        friday = "2023-10-06"
        bday_result = safe_last_label_date(friday, 60)
        calendar_day_result = pd.Timestamp(friday) + pd.Timedelta(days=60)
        assert bday_result > calendar_day_result

    def test_effective_train_cutoff_date_used_as_lookahead_base(self):
        # lookahead is measured from the EFFECTIVE cutoff when present, not
        # the plain cutoff_date.
        result = safe_last_label_date(
            "2024-01-02", 1, effective_train_cutoff_date="2023-12-01",
        )
        assert result == pd.Timestamp("2023-12-04")  # Fri + 1 BDay = Mon


class TestIsFoldEligible:
    def test_strict_less_than_excludes_equal_date(self):
        # usable_from for a Friday cutoff + 1 BDay lookahead is 2023-12-04.
        # A prediction exactly ON that date is NOT eligible (strict <,
        # mirrors entry_as_of).
        assert is_fold_eligible("2023-12-04", "2023-12-01", 1) is False
        assert is_fold_eligible("2023-12-05", "2023-12-01", 1) is True

    def test_weekend_prediction_date_boundary(self):
        # Prediction date itself falls on a Saturday (not a trading day --
        # this module does not validate session calendars, only date
        # arithmetic, so it must still answer correctly for an arbitrary
        # calendar date).
        saturday = "2023-12-02"
        assert is_fold_eligible(saturday, "2023-12-01", 1) is False  # usable 12-04

    def test_rejects_when_effective_cutoff_makes_fold_too_new(self):
        # effective_train_cutoff_date governs eligibility even when it is
        # LATER than cutoff_date would suggest on its own -- the function
        # must not silently fall back to the (more permissive) cutoff_date.
        # Here effective_train_cutoff_date=2024-01-01 is far later than
        # cutoff_date=2023-01-01; a prediction date exactly on it is not yet
        # eligible (strict <), one day later is.
        assert is_fold_eligible(
            "2024-01-01", "2023-01-01", 0,
            effective_train_cutoff_date="2024-01-01",
        ) is False
        assert is_fold_eligible(
            "2024-01-02", "2023-01-01", 0,
            effective_train_cutoff_date="2024-01-01",
        ) is True


class TestSelectLatestEligibleFold:
    def setup_method(self):
        self.folds = [
            _Fold("2023-10-02", 60),  # usable 2023-12-01 (Fri + 60 BDay)
            _Fold("2023-10-23", 60),
            _Fold("2024-01-15", 60),
        ]

    def test_picks_latest_eligible_cutoff(self):
        chosen = select_latest_eligible_fold(self.folds, "2024-06-01")
        assert chosen.cutoff_date == "2024-01-15"

    def test_none_when_before_all_coverage(self):
        assert select_latest_eligible_fold(self.folds, "2023-01-01") is None

    def test_prefers_effective_train_cutoff_date_entry_over_plain_cutoff(self):
        # Two folds with the SAME cutoff_date-adjacent ordering, but one
        # declares an effective_train_cutoff_date that differs from
        # cutoff_date -- selection must key off the correct (effective)
        # date, proving the loader's preference is honoured end-to-end, not
        # just in the single-value helper.
        folds = [
            _Fold("2023-06-01", lookahead_days=0),
            _Fold(
                "2023-09-01", lookahead_days=0,
                effective_train_cutoff_date="2023-07-01",
            ),
        ]
        # 2023-08-15 is after fold[0]'s cutoff (06-01) and after fold[1]'s
        # EFFECTIVE cutoff (07-01, since lookahead=0) -- both eligible, latest
        # cutoff_date (09-01) wins, proving it was admitted via the
        # effective date rather than being wrongly excluded by the plain
        # cutoff_date (09-01) which would still be in the future relative to
        # 2023-08-15 under a (bugged) cutoff_date-only comparison.
        chosen = select_latest_eligible_fold(folds, "2023-08-15")
        assert chosen.cutoff_date == "2023-09-01"

    def test_uneligible_effective_cutoff_correctly_excluded(self):
        # Without effective_train_cutoff_date, a fold whose cutoff_date is
        # in the future relative to today must be excluded -- confirms the
        # previous test isn't passing for an unrelated reason.
        folds = [_Fold("2023-09-01", lookahead_days=0)]
        assert select_latest_eligible_fold(folds, "2023-08-15") is None


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
