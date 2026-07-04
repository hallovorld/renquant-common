# Canonical NYSE market calendar (campaign B5)

Date: 2026-07-04
PR: feat(calendar): canonical NYSE market calendar (campaign B5)

## What

New public module `renquant_common.market_calendar` — the ONE
pandas_market_calendars-backed implementation of the session primitives that
audit #296 (orchestrator `doc/arch/2026-07-04-orchestrator-data-backtesting-
compliance-audit.md` §4.1, finding XC-2) found hand-copied SIX times across
renquant-orchestrator, renquant-base-data, renquant-backtesting and the
umbrella scripts, with real divergences (16d vs 14d lookback; fail-closed
raise vs swallow-to-None; docstring-admitted copies).

API (superset of what the six call-sites need):

- `SessionBounds` / `SessionCalendar` (protocol) / `NyseSessionCalendar` /
  `default_session_calendar()` — lifted from the de-facto orchestrator
  canonical (`intraday_quote_logger`), per-date schedule cache included.
- `session_bounds(day)` / `is_session(day)`
- `previous_session(day)` — strictly-before; `previous_session_from_calendar`
  for injected-calendar (fake-in-tests) callers.
- `last_completed_session(now)` — today counts once its (possibly half-day)
  close has passed (`now >= close`).
- `sessions_between(start, end)` — inclusive, tz-naive normalized index.
- `session_key(day)` / `session_keys(dates)` — last session at or before
  (scalar + vectorized), the backtesting `session_resolution` semantics.

## Conventions

- FAIL-CLOSED: backend unavailable ⇒ `CalendarUnavailableError`; empty query
  window ⇒ `ValueError`. Lenient callers (None / weekday fallback) wrap
  explicitly at the call-site.
- `DEFAULT_LOOKBACK_DAYS = 30` dominates every historical copy (14/16), so
  the lookback divergence class is dead by construction.
- Version 0.9.2 → 0.10.0 (additive minor); API snapshot updated;
  `pandas_market_calendars>=4` added as a declared dependency.

## Evidence

- 37 new tests (real-calendar golden vectors: holidays, half days, DST
  transitions, exact-close boundary, fail-closed paths); full suite green.
- 10-year equivalence proof (2016-01-01..2026-12-31, every calendar date,
  4 intraday probes) old-impls-vs-canonical ran in the orchestrator campaign
  workspace before any consumer re-point; results recorded in the consumer
  re-point PRs.

## Merge order

This PR merges FIRST; the per-repo re-point PRs (orchestrator, base-data,
backtesting, then umbrella scripts last) depend on it and stay red until it
lands on main.
