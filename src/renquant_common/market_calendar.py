"""Canonical NYSE market-session calendar — the ONE shared implementation.

Campaign B5 (orchestrator audit #296 §4.1 / finding XC-2): six independent
"previous / last-completed NYSE session" implementations had accumulated
across renquant-orchestrator, renquant-base-data, renquant-backtesting and
the umbrella scripts (plus two research-only variants), with real
divergences — 16-day vs 14-day lookback windows, fail-closed raise vs
swallow-to-``None``, and docstring-admitted hand-copies. This module is the
single canonical implementation; every repo imports it.

Backend: ``pandas_market_calendars`` — the same holiday / half-day dataset
the whole stack already prices against. Holidays are skipped by the
calendar; half days count as sessions with an early close; DST is resolved
via tz-aware timestamps.

Fail-mode convention (house rule): **fail closed**.

* Backend unavailable (``pandas_market_calendars`` not importable) →
  :class:`CalendarUnavailableError` from every entry point.
* No session found inside a query window → :class:`ValueError`.

Callers that want lenient behavior (return ``None``, weekday fallback, …)
must wrap EXPLICITLY at the call-site, so a loosened posture is visible
where it is chosen — never silently inside the shared primitive.

Lookback windows: scalar lookups bound their calendar query with
``lookback_days`` (default 30). The historical hand-copies used 14 or 16;
NYSE's longest non-trading stretch in the modern era is ~6 calendar days
(post-9/11 2001), so any window >= 14 yields identical results on real
dates. The default is deliberately wider than every historical copy so the
14-vs-16 divergence class is dead by construction.
"""
from __future__ import annotations

import datetime as dt
import functools
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

try:  # py>=3.9
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover - py<3.9 not supported
    raise

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd

ET = ZoneInfo("America/New_York")
UTC = dt.timezone.utc

#: Default calendar-day query window for scalar session lookups. Wider than
#: every historical hand-copy (14 / 16) — see module docstring.
DEFAULT_LOOKBACK_DAYS = 30


class CalendarUnavailableError(RuntimeError):
    """The exchange-calendar backend (``pandas_market_calendars``) cannot be
    imported. Raised fail-closed by every entry point in this module; callers
    that can legitimately degrade (weekday fallback, ``None``) must catch this
    explicitly at the call-site."""


@functools.lru_cache(maxsize=8)
def _calendar(calendar_name: str) -> Any:
    """The shared, cached ``pandas_market_calendars`` calendar object."""
    try:
        import pandas_market_calendars as mcal  # noqa: PLC0415
    except Exception as exc:  # pragma: no cover - exercised via sys.modules stub
        raise CalendarUnavailableError(
            f"pandas_market_calendars unavailable — cannot resolve {calendar_name!r} "
            f"sessions (fail-closed): {exc}"
        ) from exc
    return mcal.get_calendar(calendar_name)


def _as_date(day: "dt.date | dt.datetime | str | pd.Timestamp") -> dt.date:
    """Coerce a date-like (date / datetime / ISO string / pandas Timestamp)
    to a plain ``datetime.date``."""
    if isinstance(day, dt.datetime):
        return day.date()
    if isinstance(day, dt.date):
        return day
    if isinstance(day, str):
        return dt.date.fromisoformat(day)
    to_pydatetime = getattr(day, "to_pydatetime", None)
    if callable(to_pydatetime):
        return to_pydatetime().date()
    raise TypeError(f"cannot interpret {day!r} as a date")


def _as_aware(moment: dt.datetime) -> dt.datetime:
    """Treat a naive datetime as ET; leave an aware one alone."""
    return moment if moment.tzinfo is not None else moment.replace(tzinfo=ET)


def _pandas_ts_to_et(ts: Any) -> dt.datetime:
    """Convert a (tz-aware, typically UTC) pandas Timestamp to an aware ET
    ``datetime``. DST is resolved by the tz conversion."""
    to_pydatetime = getattr(ts, "to_pydatetime", None)
    out = to_pydatetime() if callable(to_pydatetime) else ts
    if out.tzinfo is None:
        out = out.replace(tzinfo=UTC)
    return out.astimezone(ET)


# ---------------------------------------------------------------------------
# Session-bounds primitive (dependency-injectable)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SessionBounds:
    """The regular-trading-hours boundaries of ONE exchange session, as aware
    ET datetimes. ``open`` is inclusive, ``close`` is exclusive. Early closes
    yield an earlier ``close``; DST is already resolved (the datetimes are
    tz-aware)."""

    open: dt.datetime
    close: dt.datetime

    def contains(self, moment: dt.datetime) -> bool:
        """True when ``moment`` (aware; naive treated as ET) is within
        [open, close)."""
        m = _as_aware(moment)
        return self.open <= m < self.close


class SessionCalendar(Protocol):
    """Pluggable exchange session calendar. The real impl is NYSE via
    ``pandas_market_calendars``; tests inject a deterministic fake.
    ``session_bounds`` returns ``None`` for a non-trading day
    (weekend/holiday)."""

    name: str

    def session_bounds(self, day: dt.date) -> "SessionBounds | None":
        ...


class NyseSessionCalendar:
    """Real NYSE session calendar backed by ``pandas_market_calendars``.
    Handles holidays (no session), half days / early closes (earlier
    ``market_close``) and DST (tz-aware timestamps). Schedules are cached per
    date so tick-frequency callers pay the pandas cost once per session, not
    once per tick."""

    name = "NYSE"

    def __init__(self, calendar_name: str = "NYSE") -> None:
        self.name = calendar_name
        self._cache: "dict[str, SessionBounds | None]" = {}

    def session_bounds(self, day: dt.date) -> "SessionBounds | None":
        key = day.isoformat()
        if key in self._cache:
            return self._cache[key]
        cal = _calendar(self.name)
        sched = cal.schedule(key, key)
        bounds: "SessionBounds | None"
        if sched.empty:
            bounds = None
        else:
            open_ts = sched["market_open"].iloc[0]
            close_ts = sched["market_close"].iloc[0]
            bounds = SessionBounds(
                open=_pandas_ts_to_et(open_ts), close=_pandas_ts_to_et(close_ts)
            )
        self._cache[key] = bounds
        return bounds


_DEFAULT_CALENDAR: "SessionCalendar | None" = None


def default_session_calendar() -> SessionCalendar:
    """Lazily-constructed shared NYSE calendar for real runs. Tests inject a
    fake and never reach this."""
    global _DEFAULT_CALENDAR
    if _DEFAULT_CALENDAR is None:
        _DEFAULT_CALENDAR = NyseSessionCalendar()
    return _DEFAULT_CALENDAR


# ---------------------------------------------------------------------------
# Scalar session lookups (fail-closed)
# ---------------------------------------------------------------------------
def session_bounds(
    day: "dt.date | dt.datetime | str | pd.Timestamp",
    *,
    calendar: "SessionCalendar | None" = None,
) -> "SessionBounds | None":
    """RTH bounds of ``day``'s session as aware ET datetimes, or ``None`` when
    ``day`` is not a session (weekend/holiday). Raises
    :class:`CalendarUnavailableError` when the backend is missing."""
    cal = calendar or default_session_calendar()
    return cal.session_bounds(_as_date(day))


def is_session(
    day: "dt.date | dt.datetime | str | pd.Timestamp",
    *,
    calendar: "SessionCalendar | None" = None,
) -> bool:
    """True when ``day`` is an exchange session (holiday/half-day aware —
    half days ARE sessions, merely early-close)."""
    return session_bounds(day, calendar=calendar) is not None


def previous_session(
    day: "dt.date | dt.datetime | str | pd.Timestamp",
    *,
    calendar_name: str = "NYSE",
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
) -> dt.date:
    """The most recent session STRICTLY before ``day`` (weekend / holiday /
    half-day aware). Raises :class:`ValueError` when no session exists in the
    ``lookback_days`` window (fail-closed)."""
    d = _as_date(day)
    days = sessions_between(
        d - dt.timedelta(days=lookback_days),
        d - dt.timedelta(days=1),
        calendar_name=calendar_name,
    )
    if len(days) == 0:
        raise ValueError(
            f"no {calendar_name} session found in the {lookback_days}-day "
            f"window before {d.isoformat()} (fail-closed)"
        )
    return days[-1].date()


def previous_session_from_calendar(
    calendar: SessionCalendar,
    day: "dt.date | dt.datetime | str | pd.Timestamp",
    *,
    max_lookback_days: int = DEFAULT_LOOKBACK_DAYS,
) -> dt.date:
    """:func:`previous_session` generalized over an injected
    :class:`SessionCalendar` (day-walk — used where tests inject deterministic
    fakes). Raises :class:`ValueError` when no session exists within
    ``max_lookback_days`` (fail-closed)."""
    start = _as_date(day)
    probe = start
    for _ in range(max_lookback_days):
        probe = probe - dt.timedelta(days=1)
        if calendar.session_bounds(probe) is not None:
            return probe
    raise ValueError(
        f"no exchange session found in the {max_lookback_days} days before "
        f"{start.isoformat()} (calendar {getattr(calendar, 'name', '?')!r})"
    )


def last_completed_session(
    now: "dt.datetime | str | pd.Timestamp | None" = None,
    *,
    calendar_name: str = "NYSE",
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
) -> dt.date:
    """The most recent COMPLETED session as of ``now`` (default: wall clock).

    Today counts only once its (possibly early, half-day) close has passed —
    ``now >= close`` — otherwise the prior session is returned. A naive
    ``now`` is treated as ET. Raises :class:`ValueError` when no completed
    session exists in the ``lookback_days`` window (fail-closed) and
    :class:`CalendarUnavailableError` when the backend is missing."""
    import pandas as pd  # noqa: PLC0415

    ts = pd.Timestamp.now(tz=ET) if now is None else pd.Timestamp(now)
    if ts.tzinfo is None:
        ts = ts.tz_localize(ET)
    ref_date = ts.date()
    cal = _calendar(calendar_name)
    sched = cal.schedule(
        start_date=(ref_date - dt.timedelta(days=lookback_days)).isoformat(),
        end_date=ref_date.isoformat(),
    )
    if not sched.empty:
        todays = sched[sched.index.date == ref_date]
        if not todays.empty:
            close = pd.Timestamp(todays["market_close"].iloc[-1])
            close = close.tz_localize("UTC") if close.tzinfo is None else close
            if ts >= close.tz_convert(ts.tz):
                return ref_date
        before = sched[sched.index.date < ref_date]
        if not before.empty:
            return before.index[-1].date()
    raise ValueError(
        f"no completed {calendar_name} session found in the {lookback_days}-day "
        f"window ending {ref_date.isoformat()} as of {ts} (fail-closed)"
    )


# ---------------------------------------------------------------------------
# Range / vectorized helpers
# ---------------------------------------------------------------------------
def sessions_between(
    start: "dt.date | dt.datetime | str | pd.Timestamp",
    end: "dt.date | dt.datetime | str | pd.Timestamp",
    *,
    calendar_name: str = "NYSE",
) -> "pd.DatetimeIndex":
    """All sessions in ``[start, end]`` (both inclusive) as a tz-naive,
    normalized ``pd.DatetimeIndex``. Empty when ``start > end`` or the range
    holds no session. Raises :class:`CalendarUnavailableError` when the
    backend is missing."""
    import pandas as pd  # noqa: PLC0415

    s, e = _as_date(start), _as_date(end)
    if s > e:
        return pd.DatetimeIndex([])
    cal = _calendar(calendar_name)
    days = pd.DatetimeIndex(cal.valid_days(start_date=s.isoformat(), end_date=e.isoformat()))
    if days.tz is not None:
        days = days.tz_localize(None)
    return days.normalize()


def session_key(
    day: "dt.date | dt.datetime | str | pd.Timestamp",
    sessions: "pd.DatetimeIndex | None" = None,
    *,
    calendar_name: str = "NYSE",
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
) -> dt.date:
    """Canonical decision-session key: the last session AT OR BEFORE ``day``
    (a session day maps to itself; weekend/holiday dates roll back). A pure
    function of the date against the calendar. Pass a pre-fetched
    ``sessions`` index (from :func:`sessions_between`) to amortize the
    calendar query; it must cover ``day``'s lookback. Raises
    :class:`ValueError` when no session exists at or before ``day`` inside
    the window (fail-closed — widen the window)."""
    import pandas as pd  # noqa: PLC0415

    d = _as_date(day)
    ts = pd.Timestamp(d)
    if sessions is None:
        sessions = sessions_between(
            d - dt.timedelta(days=lookback_days), d, calendar_name=calendar_name
        )
    idx = int(sessions.searchsorted(ts, side="right")) - 1
    if idx < 0:
        raise ValueError(
            f"no session at or before {d.isoformat()} in the provided window "
            f"(fail-closed; widen the sessions window)"
        )
    return sessions[idx].date()


def session_keys(
    dates: Any,
    sessions: "pd.DatetimeIndex | None" = None,
    *,
    calendar_name: str = "NYSE",
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
) -> "pd.Series":
    """Vectorized :func:`session_key`: map a series of dates to each one's
    last session at or before it. Returns a ``pd.Series`` of normalized
    ``datetime64`` values, index-aligned with the input. Raises
    :class:`ValueError` if any date precedes the sessions window
    (fail-closed)."""
    import pandas as pd  # noqa: PLC0415

    d = pd.to_datetime(pd.Series(dates)).dt.normalize()
    if d.empty:
        return pd.Series([], dtype="datetime64[ns]")
    if sessions is None:
        sessions = sessions_between(
            d.min().date() - dt.timedelta(days=lookback_days),
            d.max().date(),
            calendar_name=calendar_name,
        )
    idx = sessions.searchsorted(d.values, side="right") - 1
    if (idx < 0).any():
        bad = d[idx < 0].iloc[0]
        raise ValueError(
            f"no session at or before {bad.date().isoformat()} in the provided "
            f"window (fail-closed; widen the sessions window)"
        )
    return pd.Series(sessions.values[idx], index=d.index)
