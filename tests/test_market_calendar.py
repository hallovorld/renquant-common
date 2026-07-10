"""Tests for the canonical NYSE market calendar (campaign B5).

Golden vectors are REAL calendar facts (holidays, half days, weekends) —
they pin the semantics every consumer repo re-points onto, so a
pandas_market_calendars dataset regression fails loudly here rather than
silently drifting six call-sites."""
from __future__ import annotations

import datetime as dt
import sys

import pandas as pd
import pytest

from renquant_common.market_calendar import (
    ET,
    DEFAULT_LOOKBACK_DAYS,
    CalendarUnavailableError,
    NyseSessionCalendar,
    SessionBounds,
    _calendar,
    default_session_calendar,
    is_session,
    last_completed_session,
    previous_session,
    previous_session_from_calendar,
    session_bounds,
    session_key,
    session_keys,
    sessions_between,
)


# ---------------------------------------------------------------------------
# is_session — holidays / weekends / half days
# ---------------------------------------------------------------------------
def test_is_session_regular_weekday() -> None:
    assert is_session("2026-06-30") is True  # Tuesday


def test_is_session_weekend() -> None:
    assert is_session("2026-06-27") is False  # Saturday
    assert is_session("2026-06-28") is False  # Sunday


def test_is_session_holidays() -> None:
    assert is_session("2026-07-03") is False  # Independence Day observed (Fri)
    assert is_session("2025-11-27") is False  # Thanksgiving
    assert is_session("2016-03-25") is False  # Good Friday
    assert is_session("2016-01-01") is False  # New Year's Day


def test_half_day_is_a_session() -> None:
    assert is_session("2025-11-28") is True  # Friday after Thanksgiving


def test_accepts_date_datetime_and_timestamp() -> None:
    assert is_session(dt.date(2026, 6, 30)) is True
    assert is_session(dt.datetime(2026, 6, 30, 12, 0)) is True
    assert is_session(pd.Timestamp("2026-06-30")) is True


# ---------------------------------------------------------------------------
# session_bounds — RTH bounds, early close, DST awareness
# ---------------------------------------------------------------------------
def test_session_bounds_regular_day() -> None:
    b = session_bounds("2025-11-25")
    assert b is not None
    assert b.open.astimezone(ET).time() == dt.time(9, 30)
    assert b.close.astimezone(ET).time() == dt.time(16, 0)
    assert b.open.tzinfo is not None and b.close.tzinfo is not None


def test_session_bounds_half_day_early_close() -> None:
    b = session_bounds("2025-11-28")
    assert b is not None
    assert b.close.astimezone(ET).time() == dt.time(13, 0)


def test_session_bounds_non_session_is_none() -> None:
    assert session_bounds("2026-07-03") is None
    assert session_bounds("2026-06-28") is None


def test_session_bounds_contains_naive_treated_as_et() -> None:
    b = session_bounds("2025-11-25")
    assert b is not None
    assert b.contains(dt.datetime(2025, 11, 25, 12, 0)) is True  # naive == ET
    assert b.contains(dt.datetime(2025, 11, 25, 16, 0)) is False  # close exclusive


def test_nyse_session_calendar_caches_per_date() -> None:
    cal = NyseSessionCalendar()
    first = cal.session_bounds(dt.date(2025, 11, 25))
    again = cal.session_bounds(dt.date(2025, 11, 25))
    assert first == again
    assert "2025-11-25" in cal._cache


def test_default_session_calendar_is_singleton() -> None:
    assert default_session_calendar() is default_session_calendar()


# ---------------------------------------------------------------------------
# previous_session — strictly-before semantics over weekends/holidays
# ---------------------------------------------------------------------------
def test_previous_session_over_holiday_weekend() -> None:
    # Mon 2026-07-06 ← (Sun, Sat, Fri=Jul-4-observed) ← Thu 2026-07-02
    assert previous_session("2026-07-06") == dt.date(2026, 7, 2)


def test_previous_session_regular() -> None:
    assert previous_session("2026-07-01") == dt.date(2026, 6, 30)


def test_previous_session_of_non_session_day() -> None:
    assert previous_session("2026-06-28") == dt.date(2026, 6, 26)  # Sun → Fri


def test_previous_session_fail_closed_on_empty_window() -> None:
    with pytest.raises(ValueError, match="fail-closed"):
        previous_session("2026-07-06", lookback_days=1)  # only Sunday in window


def test_previous_session_from_calendar_matches_default() -> None:
    cal = default_session_calendar()
    for day in ("2026-07-06", "2026-07-01", "2026-06-28"):
        assert previous_session_from_calendar(cal, day) == previous_session(day)


def test_previous_session_from_calendar_fail_closed() -> None:
    class NeverOpen:
        name = "FAKE"

        def session_bounds(self, day):  # noqa: ANN001, ANN202
            return None

    with pytest.raises(ValueError, match="no exchange session"):
        previous_session_from_calendar(NeverOpen(), "2026-07-06", max_lookback_days=5)


# ---------------------------------------------------------------------------
# last_completed_session — close-passed semantics incl. half days
# ---------------------------------------------------------------------------
def test_last_completed_session_after_regular_close() -> None:
    # Tue 2025-11-25 16:30 ET — today's 16:00 close has passed.
    assert last_completed_session("2025-11-25 16:30") == dt.date(2025, 11, 25)


def test_last_completed_session_before_close_is_prior_session() -> None:
    assert last_completed_session("2025-11-25 14:00") == dt.date(2025, 11, 24)


def test_last_completed_session_half_day_close() -> None:
    # Fri 2025-11-28 closes 13:00 ET: 14:00 → today, 12:00 → Wed (Thu holiday).
    assert last_completed_session("2025-11-28 14:00") == dt.date(2025, 11, 28)
    assert last_completed_session("2025-11-28 12:00") == dt.date(2025, 11, 26)


def test_last_completed_session_at_exact_close_counts() -> None:
    # now >= close ⇒ today (>= not >) — the shared convention of the old
    # orchestrator and base-data copies.
    assert last_completed_session("2025-11-25 16:00") == dt.date(2025, 11, 25)


def test_last_completed_session_weekend() -> None:
    assert last_completed_session("2026-06-28 09:00") == dt.date(2026, 6, 26)


def test_last_completed_session_aware_input() -> None:
    aware = dt.datetime(2025, 11, 25, 21, 30, tzinfo=dt.timezone.utc)  # 16:30 ET
    assert last_completed_session(aware) == dt.date(2025, 11, 25)


def test_last_completed_session_fail_closed() -> None:
    with pytest.raises(ValueError, match="fail-closed"):
        last_completed_session("2026-06-28 09:00", lookback_days=1)


# ---------------------------------------------------------------------------
# sessions_between — inclusive range, tz-naive normalized index
# ---------------------------------------------------------------------------
def test_sessions_between_inclusive_and_holiday_aware() -> None:
    days = sessions_between("2026-06-29", "2026-07-03")
    assert [d.isoformat() for d in days.date] == [
        "2026-06-29", "2026-06-30", "2026-07-01", "2026-07-02",
    ]
    assert days.tz is None


def test_sessions_between_empty_when_inverted_or_no_sessions() -> None:
    assert len(sessions_between("2026-07-02", "2026-07-01")) == 0
    assert len(sessions_between("2026-07-04", "2026-07-05")) == 0  # Sat+Sun


# ---------------------------------------------------------------------------
# session_key / session_keys — last session at or before
# ---------------------------------------------------------------------------
def test_session_key_identity_on_session_day() -> None:
    assert session_key("2026-06-30") == dt.date(2026, 6, 30)


def test_session_key_weekend_rolls_to_friday() -> None:
    assert session_key("2026-06-28") == dt.date(2026, 6, 26)


def test_session_key_holiday_rolls_back() -> None:
    assert session_key("2026-07-03") == dt.date(2026, 7, 2)


def test_session_key_with_prefetched_sessions() -> None:
    sessions = sessions_between("2026-06-01", "2026-07-31")
    assert session_key("2026-06-28", sessions) == dt.date(2026, 6, 26)


def test_session_key_fail_closed_outside_window() -> None:
    sessions = sessions_between("2026-06-01", "2026-06-30")
    with pytest.raises(ValueError, match="widen"):
        session_key("2026-05-01", sessions)


def test_session_keys_vectorized_matches_scalar() -> None:
    dates = pd.Series(["2026-06-26", "2026-06-27", "2026-06-28", "2026-07-03"])
    keys = session_keys(dates)
    assert [k.date() for k in keys] == [session_key(d) for d in dates]
    assert list(keys.index) == list(dates.index)


def test_session_keys_empty_input() -> None:
    assert session_keys(pd.Series([], dtype="object")).empty


# ---------------------------------------------------------------------------
# Fail-closed on missing backend
# ---------------------------------------------------------------------------
def test_calendar_unavailable_raises_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    _calendar.cache_clear()
    monkeypatch.setitem(sys.modules, "pandas_market_calendars", None)
    try:
        with pytest.raises(CalendarUnavailableError):
            sessions_between("2026-06-01", "2026-06-30", calendar_name="NYSE")
        with pytest.raises(CalendarUnavailableError):
            NyseSessionCalendar().session_bounds(dt.date(2026, 6, 30))
    finally:
        _calendar.cache_clear()


def test_default_lookback_wider_than_all_historical_copies() -> None:
    # The old hand-copies used 14 (base-data, rq105 bundle) and 16
    # (retrain freshness). The canonical default must dominate both so the
    # divergence class is dead by construction.
    assert DEFAULT_LOOKBACK_DAYS >= 16


def test_session_bounds_dst_transition_days() -> None:
    # 2026-03-09 is the first session after the US spring-forward (2026-03-08);
    # 2025-11-03 the first after fall-back (2025-11-02). Bounds must still be
    # 09:30 / 16:00 ET with correct UTC offsets (-4 vs -5).
    spring = session_bounds("2026-03-09")
    fall = session_bounds("2025-11-03")
    assert spring is not None and fall is not None
    assert spring.open.astimezone(ET).time() == dt.time(9, 30)
    assert fall.open.astimezone(ET).time() == dt.time(9, 30)
    assert spring.open.utcoffset() == dt.timedelta(hours=-4)
    assert fall.open.utcoffset() == dt.timedelta(hours=-5)


def test_session_bounds_dataclass_frozen() -> None:
    b = SessionBounds(
        open=dt.datetime(2026, 6, 30, 9, 30, tzinfo=ET),
        close=dt.datetime(2026, 6, 30, 16, 0, tzinfo=ET),
    )
    with pytest.raises(Exception):
        b.open = b.close  # type: ignore[misc]


# ---------------------------------------------------------------------------
# ALWAYS_OPEN mode (crypto RFC 2026-07-10, M2/P1) — sessions = UTC days
# ---------------------------------------------------------------------------
class TestAlwaysOpenMode:
    def test_every_day_is_a_session_including_weekend_and_holiday(self) -> None:
        from renquant_common.market_calendar import AlwaysOpenSessionCalendar

        cal = AlwaysOpenSessionCalendar()
        # Sunday, Christmas, and a regular Tuesday all have sessions.
        for day in (dt.date(2026, 7, 5), dt.date(2025, 12, 25), dt.date(2026, 6, 30)):
            assert is_session(day, calendar=cal)

    def test_session_bounds_are_utc_calendar_day(self) -> None:
        from renquant_common.market_calendar import AlwaysOpenSessionCalendar

        b = AlwaysOpenSessionCalendar().session_bounds(dt.date(2026, 7, 5))
        assert b is not None
        assert b.open == dt.datetime(2026, 7, 5, tzinfo=dt.timezone.utc)
        assert b.close == dt.datetime(2026, 7, 6, tzinfo=dt.timezone.utc)
        assert b.contains(dt.datetime(2026, 7, 5, 23, 59, tzinfo=dt.timezone.utc))
        assert not b.contains(dt.datetime(2026, 7, 6, 0, 0, tzinfo=dt.timezone.utc))

    def test_previous_session_is_prior_calendar_day(self) -> None:
        # Monday's previous session is SUNDAY, not Friday.
        assert previous_session(
            dt.date(2026, 7, 6), calendar_name="ALWAYS_OPEN"
        ) == dt.date(2026, 7, 5)

    def test_last_completed_session_is_yesterday_utc(self) -> None:
        # Mid-day UTC Sunday → Saturday's UTC bar is the last completed one.
        now = pd.Timestamp("2026-07-05 14:00:00", tz="UTC")
        assert last_completed_session(
            now, calendar_name="ALWAYS_OPEN"
        ) == dt.date(2026, 7, 4)

    def test_last_completed_session_at_exact_midnight_counts_just_ended_day(self) -> None:
        now = pd.Timestamp("2026-07-06 00:00:00", tz="UTC")
        assert last_completed_session(
            now, calendar_name="ALWAYS_OPEN"
        ) == dt.date(2026, 7, 5)

    def test_last_completed_session_naive_now_is_utc(self) -> None:
        # 2026-07-05 01:00 naive == 01:00 UTC (NOT ET) → last completed 07-04.
        assert last_completed_session(
            pd.Timestamp("2026-07-05 01:00:00"), calendar_name="ALWAYS_OPEN"
        ) == dt.date(2026, 7, 4)

    def test_sessions_between_covers_every_calendar_day(self) -> None:
        days = sessions_between(
            "2026-07-03", "2026-07-06", calendar_name="ALWAYS_OPEN"
        )
        assert list(days) == list(pd.to_datetime(
            ["2026-07-03", "2026-07-04", "2026-07-05", "2026-07-06"]
        ))

    def test_session_key_is_identity_on_any_day(self) -> None:
        # Weekend day maps to itself (NYSE mode would roll back to Friday).
        assert session_key(
            dt.date(2026, 7, 5), calendar_name="ALWAYS_OPEN"
        ) == dt.date(2026, 7, 5)

    def test_session_keys_vectorized_identity(self) -> None:
        out = session_keys(
            ["2026-07-04", "2026-07-05"], calendar_name="ALWAYS_OPEN"
        )
        assert list(out) == list(pd.to_datetime(["2026-07-04", "2026-07-05"]))

    def test_mode_works_without_pandas_market_calendars(self, monkeypatch) -> None:
        # The always-open branch must not touch the mcal backend at all.
        import renquant_common.market_calendar as mc

        monkeypatch.setitem(sys.modules, "pandas_market_calendars", None)
        _calendar.cache_clear()
        try:
            assert previous_session(
                dt.date(2026, 7, 6), calendar_name="ALWAYS_OPEN"
            ) == dt.date(2026, 7, 5)
            assert last_completed_session(
                pd.Timestamp("2026-07-05 14:00:00", tz="UTC"),
                calendar_name="ALWAYS_OPEN",
            ) == dt.date(2026, 7, 4)
            assert len(sessions_between(
                "2026-07-03", "2026-07-05", calendar_name="ALWAYS_OPEN"
            )) == 3
            # NYSE mode must still fail closed with the backend gone.
            with pytest.raises(CalendarUnavailableError):
                mc.sessions_between("2026-07-01", "2026-07-06")
        finally:
            _calendar.cache_clear()

    def test_nyse_default_unaffected_by_always_open_addition(self) -> None:
        # Equity byte-identity pin: default calendar_name still rolls a
        # Sunday back to Friday and skips July-4 observance.
        assert previous_session(dt.date(2026, 7, 6)) == dt.date(2026, 7, 2)

    # -- round-1 fix (Codex review of #27): UTC semantics, not local/ET --

    def test_contains_naive_moment_is_utc_not_et(self) -> None:
        from renquant_common.market_calendar import AlwaysOpenSessionCalendar

        b = AlwaysOpenSessionCalendar().session_bounds(dt.date(2026, 7, 10))
        # Naive 23:00 is INSIDE the July-10 UTC session (23:00 UTC). A
        # hardcoded naive-as-ET reading would treat this as 23:00 EDT ==
        # 2026-07-11 03:00 UTC, landing OUTSIDE [D 00:00, D+1 00:00) UTC.
        assert b.contains(dt.datetime(2026, 7, 10, 23, 0))
        assert not b.contains(dt.datetime(2026, 7, 11, 0, 0))

    def test_contains_aware_offset_crossing_utc_midnight(self) -> None:
        from renquant_common.market_calendar import AlwaysOpenSessionCalendar

        b = AlwaysOpenSessionCalendar().session_bounds(dt.date(2026, 7, 10))
        # 2026-07-10 23:00 EDT (-04:00) == 2026-07-11 03:00 UTC — outside
        # July 10's UTC session, inside July 11's.
        et_late = dt.datetime(2026, 7, 10, 23, 0, tzinfo=dt.timezone(dt.timedelta(hours=-4)))
        assert not b.contains(et_late)
        b_next = AlwaysOpenSessionCalendar().session_bounds(dt.date(2026, 7, 11))
        assert b_next.contains(et_late)

    def test_nyse_contains_naive_moment_still_et(self) -> None:
        # The fix must not regress NYSE-mode bounds' own naive convention.
        b = SessionBounds(
            open=dt.datetime(2026, 7, 10, 9, 30, tzinfo=ET),
            close=dt.datetime(2026, 7, 10, 16, 0, tzinfo=ET),
        )
        assert b.contains(dt.datetime(2026, 7, 10, 12, 0))
        assert not b.contains(dt.datetime(2026, 7, 10, 20, 0))

    def test_previous_session_aware_offset_crossing_utc_midnight(self) -> None:
        # 2026-07-10 23:00 EDT == 2026-07-11 03:00 UTC -> UTC day is the
        # 11th, so the previous UTC session is the 10th, not the 9th (which
        # a local-date .date() on the ET-aware input would wrongly give).
        et_late = dt.datetime(2026, 7, 10, 23, 0, tzinfo=dt.timezone(dt.timedelta(hours=-4)))
        assert previous_session(
            et_late, calendar_name="ALWAYS_OPEN"
        ) == dt.date(2026, 7, 10)

    def test_session_bounds_scalar_helper_aware_offset_crossing_utc_midnight(self) -> None:
        from renquant_common.market_calendar import AlwaysOpenSessionCalendar

        et_late = dt.datetime(2026, 7, 10, 23, 0, tzinfo=dt.timezone(dt.timedelta(hours=-4)))
        b = session_bounds(et_late, calendar=AlwaysOpenSessionCalendar())
        assert b.open == dt.datetime(2026, 7, 11, tzinfo=dt.timezone.utc)

    def test_sessions_between_aware_offset_crossing_utc_midnight(self) -> None:
        et_late = dt.datetime(2026, 7, 10, 23, 0, tzinfo=dt.timezone(dt.timedelta(hours=-4)))
        days = sessions_between(et_late, et_late, calendar_name="ALWAYS_OPEN")
        assert list(days) == [pd.Timestamp("2026-07-11")]

    def test_session_key_aware_offset_crossing_utc_midnight(self) -> None:
        et_late = dt.datetime(2026, 7, 10, 23, 0, tzinfo=dt.timezone(dt.timedelta(hours=-4)))
        assert session_key(
            et_late, calendar_name="ALWAYS_OPEN"
        ) == dt.date(2026, 7, 11)

    def test_session_keys_vectorized_aware_offset_crossing_utc_midnight(self) -> None:
        et_late = dt.datetime(2026, 7, 10, 23, 0, tzinfo=dt.timezone(dt.timedelta(hours=-4)))
        out = session_keys([et_late], calendar_name="ALWAYS_OPEN")
        assert list(out) == [pd.Timestamp("2026-07-11")]

    def test_previous_session_from_calendar_aware_offset_crossing_utc_midnight(self) -> None:
        from renquant_common.market_calendar import AlwaysOpenSessionCalendar

        et_late = dt.datetime(2026, 7, 10, 23, 0, tzinfo=dt.timezone(dt.timedelta(hours=-4)))
        assert previous_session_from_calendar(
            AlwaysOpenSessionCalendar(), et_late
        ) == dt.date(2026, 7, 10)
