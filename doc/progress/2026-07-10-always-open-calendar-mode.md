# ALWAYS_OPEN calendar mode (crypto RFC M2 / pipeline P1)

Date: 2026-07-10
PR: feat(calendar): ALWAYS_OPEN session mode for 24/7 asset classes

## What

Adds the canonical always-open (24/7) session model to
`renquant_common.market_calendar`, per the merged crypto trading RFC
(orchestrator `doc/design/2026-07-10-crypto-trading-rfc.md` §2.5 gap M2 /
§3.4): ONE shared ALWAYS_OPEN calendar in common, consumed by
base-data / pipeline / orchestrator, instead of three local hacks.

Semantics (frozen):

- A "session" is one **UTC calendar day**: `[D 00:00, D+1 00:00) UTC`.
  Every day is a session — no weekends, no holidays.
- Session `D` **completes at `D+1 00:00:00 UTC`** → the last completed
  session as of any instant inside UTC day `X` is `X - 1` (at exactly
  midnight the just-ended day counts, mirroring the NYSE `now >= close`
  rule).
- Naive datetimes are interpreted as **UTC** in this mode (NYSE mode keeps
  its naive-as-ET convention).
- The mode never touches `pandas_market_calendars`, so it never raises
  `CalendarUnavailableError`.

## API

- `ALWAYS_OPEN_CALENDAR_NAME = "ALWAYS_OPEN"` — sentinel accepted by every
  helper that takes a `calendar_name` (`previous_session`,
  `last_completed_session`, `sessions_between`, and therefore
  `session_key`/`session_keys` which route through `sessions_between`).
- `AlwaysOpenSessionCalendar` — `SessionCalendar`-protocol implementation
  (UTC-day `SessionBounds`), injectable wherever `NyseSessionCalendar` is.
- Both exported from the package root; API snapshot + version bumped
  (0.10.0 → 0.11.0, minor: additive).

## Equity byte-identity

Default `calendar_name="NYSE"` paths are untouched (branch-on-sentinel
only); the existing 44 calendar tests pass unchanged, plus an explicit pin
that a Sunday still rolls back to the prior NYSE session under the default.

## Tests

`tests/test_market_calendar.py::TestAlwaysOpenMode` — 11 new tests:
session-every-day (weekend/holiday), UTC bounds, previous-session = day-1,
last-completed = yesterday-UTC (incl. exact-midnight edge + naive-as-UTC
pin), sessions_between covers all days, session_key identity on weekends,
works with `pandas_market_calendars` absent while NYSE mode still fails
closed, NYSE default unaffected.

## Consumers

First consumer: renquant-pipeline's asset-class execution-policy PR
(crypto P1 freshness clock) soft-consumes this mode via
`kernel/asset_class.py` and degrades to identical local UTC-day arithmetic
when the installed common predates 0.11.0 — so the pipeline PR does not
hard-depend on this one merging first. Base-data (B3) and the orchestrator
24/7 scheduler (D-C11) consume it next per the RFC deliverables table.
