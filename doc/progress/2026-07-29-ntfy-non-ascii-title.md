# 2026-07-29 — a non-ASCII title silently dropped the ENTIRE alert, body included   (PR #37)

STATUS:    delivered
WHAT:      `src/renquant_common/notify.py` adds `encode_header()`, which
           RFC-2047-encodes a header value (`=?UTF-8?B?...?=`) when it is not
           pure ASCII, and leaves ASCII values untouched. `send()` now routes
           `Title` and `Tags` through it before building the `urllib.request`
           headers dict; `Priority` stays numeric-only, unaffected.
WHY/DIR:   Live fleet log showed `ntfy send failed ... title='rq104 blend
           假想前10 — 2026-07-28': 'latin-1' codec can't encode characters in
           position 12-14`. HTTP header values go out latin-1 and `urllib`
           builds the request eagerly, so `headers = {"Title": str(title)}`
           raised before the request was ever sent — the exception was
           caught, counted, and logged, and the WHOLE notification (body
           included) was discarded. Any operator-natural text in a title —
           a Chinese noun, an em dash, a curly quote, a degree sign, an
           emoji — silently disabled the alert it was attached to. The
           GOAL-5 sentinels, the drift scan, and the degradation watchers
           all terminate in this one function, so this closed a hole in the
           safety net, not a cosmetic formatting nit.
EVIDENCE:
  artifact:      tests/test_notify_non_ascii_title.py (13 new tests) +
                 tests/test_notify.py (existing suite).
  prod or exp:   prod path (`src/renquant_common/notify.py::send`, the sole
                 ntfy send function all fleet alerts route through).
  existing data: `pytest -q tests/test_notify.py
                 tests/test_notify_non_ascii_title.py` on this PR's head
                 (3b345ec) -> 50 passed. Load-bearing case:
                 `test_the_body_is_not_lost_to_a_title_problem` asserts the
                 payload (not just the title) survives a non-ASCII title.
                 Full-suite diff vs stashed change on the same worktree:
                 baseline 10 failures, with the change 10 failures — zero
                 introduced; the 10 are pre-existing on untouched
                 `origin/main` (20442b6), unrelated to this file.
  best-known?:   only implementation of header-safe encoding in this
                 function; no prior variant to compare against.
  scope:         this is `src/renquant_common/notify.py::encode_header` +
                 `send`, prod, vs the prior behavior of raising and
                 silently dropping the whole notification on any non-ASCII
                 `Title`/`Tags` value.
NEXT:      merge -> no follow-up code change; the fix is self-contained to
           the ntfy send path. Progress doc + `fixed by claude` audit
           comment added in this pass to satisfy the reviewer's MED finding
           (missing C5 doc) — no further findings open on this PR.
