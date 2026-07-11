# ALWAYS_OPEN calendar: fix UTC semantics (round-1, Codex review of #27)

Date: 2026-07-10
PR: fix(calendar): ALWAYS_OPEN mode UTC semantics (round-1 fix)

## What

Codex's review of the ALWAYS_OPEN calendar mode PR found two real bugs where
the naive-as-ET / local-date NYSE-mode conventions leaked into the new
naive-as-UTC mode:

1. `SessionBounds.contains()` always treated a naive `moment` as ET via the
   module-level `_as_aware()` helper, regardless of which calendar mode
   produced the bounds. A naive instant intended as UTC (e.g. `23:00` inside
   a UTC session) could be misclassified as outside the session once
   (incorrectly) read as ET and converted. Fixed by having `contains()`
   infer the naive convention from the bounds' OWN `open.tzinfo` (ET for
   NYSE-mode bounds, UTC for ALWAYS_OPEN-mode bounds) instead of a hardcoded
   constant — this also makes the dataclass correct for any future calendar
   mode with its own naive convention, not just these two.
2. The ALWAYS_OPEN scalar/range helpers (`session_bounds`,
   `previous_session`, `previous_session_from_calendar`, `sessions_between`,
   `session_key`, `session_keys`) extracted a date via the shared `_as_date`
   / `pd.Timestamp.normalize()` path, which returns an aware datetime's date
   in ITS OWN attached timezone rather than converting to UTC first. An
   aware non-UTC-offset instant near midnight (e.g. `2026-07-10 23:00 EDT`
   == `2026-07-11 03:00 UTC`) landed on the wrong UTC calendar day. Fixed by
   adding `_as_utc_date()` (naive treated as already-UTC, aware converted to
   UTC before taking `.date()`) and routing every ALWAYS_OPEN branch through
   it instead of the NYSE-mode `_as_date()`; the vectorized `session_keys`
   uses the equivalent `pd.to_datetime(..., utc=True)` idiom.

## Tests

8 new cases in `TestAlwaysOpenMode` reproduce Codex's exact scenarios:
naive-23:00-is-inside-the-UTC-session, an NYSE-mode regression pin (naive
still ET for NYSE bounds), and one aware-EDT-instant-crossing-UTC-midnight
case per affected function (`session_bounds`, `previous_session`,
`sessions_between`, `session_key`, `session_keys`,
`previous_session_from_calendar`). Verified meaningful via stash-revert:
reverting only the source fix failed 7 of 8 (the 8th, `session_keys`, was
not affected in the specific case tested — see PR for detail).

Full suite: 320 passed, 18 skipped, plus 2 known-pre-existing failures in
`test_api_snapshot.py` (installed package metadata is stale relative to the
source tree in this shared venv — reproduces identically on the pre-fix
head, unrelated to this change).

## Scope note

Does not address renquant-pipeline#183's separate finding (local
ALWAYS_OPEN duplication instead of depending on this package) — that PR
stays blocked until this one merges and versions, at which point it can
drop its local fallback and hard-depend on `renquant-common>=0.11.0`.
