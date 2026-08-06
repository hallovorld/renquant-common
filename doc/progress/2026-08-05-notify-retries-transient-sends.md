# A timed-out alarm was simply lost

STATUS: complete. `notify.send` gains a bounded retry for transient failures. The
function still never raises, and every existing caller is untouched.

WHAT: `send()` now attempts up to `SEND_ATTEMPTS = 3` times with
`SEND_BACKOFF_SECONDS = (1.0, 2.0)` between them, retrying **only** failures that
another attempt could plausibly fix. A send that recovers says so in the log; a send
that exhausts its attempts fails soft exactly as before.

WHY/DIR: operator report, 2026-08-05 — *"failure should be retried, this should be in
pipeline design!"* That is correct here, and it cost real alarms: two pages were lost
to `ntfy send failed … read operation timed out`, from `run-surface-drift` and
`rq105-liveness`. A monitoring path that drops a page on a transient network error is
strictly worse than one that never sent it, because the silence reads as health.

## What is retried, and what is deliberately not

```python
def _is_transient(exc: BaseException) -> bool:
    if isinstance(exc, urllib.error.HTTPError):
        return 500 <= int(exc.code) < 600
    if isinstance(exc, urllib.error.URLError):
        return True                      # DNS / refused / unreachable
    return isinstance(exc, (TimeoutError, OSError))
```

5xx, DNS/refused/unreachable, timeouts and OS-level errors are worth another attempt.
**4xx is not**: a malformed request, a bad topic or a rejected header is wrong in a way
repetition cannot fix, and retrying it turns one useless send into three. Retrying
everything is the easy version of this change and it is the wrong one — it would add
load precisely when the far side is telling us the request itself is bad.

The same reasoning bounds the loop. Three attempts and a 1s/2s backoff keep a
scheduled job's worst case near three seconds; an unbounded retry inside an alerting
path can hold a monitor open long enough to miss its own next firing.

## Six tests, one fixture

- a timeout is retried and the alarm survives
- retries are BOUNDED (no unbounded loop)
- a 5xx is transient and retried
- **a 4xx is NOT retried** — the anti-over-retry control
- the happy path adds NO attempts
- a recovered send SAYS it recovered (`ntfy send succeeded on attempt N/M`)

An autouse fixture patches the backoff so the suite does not actually sleep: without it
these six tests would spend real seconds waiting. Suite: **40 passed in 0.71s**.

## Scope discipline

This does **not** add retry anywhere else. The operator's report was general, but retry
is wrong for a deterministic verdict — the merge-audit gate fails identically on every
attempt, so retrying it would multiply noise rather than recover an alarm. The
distinction that matters is whether a second attempt could plausibly differ, and only
the transport layer qualifies.

EVIDENCE:

| claim | value | provenance |
|---|---|---|
| alarms were actually lost to this | 2 (`run-surface-drift`, `rq105-liveness`) | [VERIFIED — `ntfy send failed … read operation timed out` in their logs] |
| retry is bounded | `SEND_ATTEMPTS = 3`, backoff `(1.0, 2.0)` | [VERIFIED — `src/renquant_common/notify.py:56,61`] |
| 4xx is excluded | `_is_transient` returns False below 500 | [VERIFIED — same file, `_is_transient`] |
| suite | **40 passed in 0.71s** (6 new) | [VERIFIED — `pytest -q tests/test_notify.py` on this branch] |
| `send()` still never raises | unchanged fail-soft path, `_send_failures` still incremented once per failed send | [VERIFIED — diff is additive around the existing except block] |

artifact: none. A library function; no artifact is produced, staged or promoted.
prod or exp: production library path — `renquant_common.notify.send` is the canonical
  sender every alerting caller uses. Behaviour changes only on the failure branch: a
  send that used to fail once now fails after three transient attempts, and one that
  used to be lost to a timeout can now arrive.
existing data: yes — the failure this fixes was read from the two jobs' own logs, which
  already carried the `read operation timed out` lines. Nothing was generated.
best-known?: yes for the stated problem. Bounded retry with a transient/permanent split
  is the standard remedy, and the alternative (retry everything) is measurably worse on
  4xx. A durable outbox would survive process death too, but that is a much larger
  change and this path is fail-soft by design.
scope: one function plus its two module constants and a helper, and one test file. No
  caller signature changes; no other repo is touched.

NEXT: the callers that page on every run still page on every run — repetition of an
unchanged verdict is a separate problem from a lost one, and it is not addressed here.
`ops/liveness_common.py` has no suppression primitive; an opt-in `dedup_key` with a
recorded (not silent) skip is the shape, and it belongs in its own PR.
