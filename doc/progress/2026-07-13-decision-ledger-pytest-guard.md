# Fail-closed pytest guard for `decision_ledger.connect()`

Date: 2026-07-13
PR: fix(decision_ledger): fail-closed guard against real-DB writes from pytest

## Incident

While fixing an unrelated stale-mock bug in renquant-pipeline (`fix/v003-import-from-common`,
PR hallovorld/renquant-pipeline#195 — `tests/test_task_decision_ledger.py`'s
autouse fixture still faked `renquant_orchestrator.decision_ledger` after
production code (`task_decision_ledger.py`) was repointed to import
`renquant_common.decision_ledger` per this repo's V-003 fix (#30)), the
unmodified pre-fix test suite hit a code path where mocking failed to
intercept a `connect()` call. The REAL, unmocked `connect()`/`write_verdicts()`
ran against the REAL `~/renquant-data/decision_ledger.db` — the production
decision-ledger DB that real daily trading runs write real gate verdicts
into — and wrote fixture rows into it. The rows were identified (uniquely
fixture-shaped `run_id`, no match to any live-run naming convention) and
deleted; independently verified clean afterward.

That specific test bug is already fixed on the pipeline side (PR #195). But
the class of bug is systemic: **any** call site across renquant-common,
renquant-orchestrator, and renquant-pipeline that reaches
`decision_ledger.connect()` with no `db_path` (or a mis-mocked one) is
exposed to the same risk, forever, as long as the only thing standing
between a test and the live DB is "did this particular test's mock actually
intercept the call." A per-test-file fix doesn't close that gap — it only
patches the one file where the bug happened to manifest this time. Since
`connect()` was just consolidated into renquant-common as the single point
of truth (V-003), that's also the single point where a structural guard can
actually cover every caller in every repo at once.

## Fix

Added a fail-closed check directly inside `renquant_common.decision_ledger.connect()`:

- Detects "a pytest test is executing right now" via
  `"PYTEST_CURRENT_TEST" in os.environ` — set by pytest for the duration of
  each test, unlike `"pytest" in sys.modules` (true merely because pytest
  was imported, with no test running).
- When that signal is present AND the resolved `db_path` (explicit or
  defaulted from `DEFAULT_DB`) canonicalizes (`expanduser().resolve()`) to
  the real production path (`~/renquant-data/decision_ledger.db`), raises
  `RuntimeError` with a message explaining what happened and how to fix the
  test (explicit `tmp_path`, monkeypatch `DEFAULT_DB`, or mock
  `connect`/`write_verdicts` entirely). It does **not** silently redirect
  to a temp path — a silent redirect would just mask the same bug in a
  different, harder-to-notice way.
- **No escape hatch.** There is no env var, no module attribute, no
  mechanism to bypass the guard. If a pytest process needs the real
  production path, the test is wrong — redesign it with a tmp_path.
- The production path identity is a string literal (`"renquant-data/decision_ledger.db"`)
  inlined inside the guard function body, joined with `Path.home()` at call
  time. There is no module-level attribute for the relpath — monkeypatching
  `DEFAULT_DB`, or setting any other module attribute, cannot shift the
  protection boundary.
- `write_verdicts()` takes an already-open `sqlite3.Connection` and never
  calls `connect()` itself, so it is unaffected by this guard; traced every
  caller of `connect()` in common/orchestrator/pipeline (see below) and
  confirmed none breaks.

## Verification across the other two repos

- **renquant-orchestrator**: `decision_ledger.py` re-exports `connect` from
  common (same function object), so the guard covers it automatically.
  Checked every test file importing `decision_ledger.connect` (`test_decision_ledger.py`,
  `test_cli.py`, `test_daily_trading_health.py`, `test_outcome_observer.py`,
  `test_ledger_attribution.py`) — all already pass an explicit `tmp_path`/`:memory:`
  path or a `conn=` fixture. `tests/test_attribution_engine.py`'s
  `lg.connect(REAL_DB)` is a *different* `connect` (`attribution/ledger.py`'s
  own, against a different DB, `runs.alpaca.db`, and read-only/skip-if-absent)
  — unrelated. No changes needed.
- **renquant-pipeline**: the only production call site is
  `task_decision_ledger.py` (function-local import of
  `renquant_common.decision_ledger.connect`/`write_verdicts`). Its test
  (`tests/test_task_decision_ledger.py`) is the file PR #195 already fixed
  to mock `renquant_common.decision_ledger.connect`/`write_verdicts`
  directly (verified against this branch's actual module shape:
  `connect`, `write_verdicts`, `DDL`, `DEFAULT_DB`, `_VALID_VERDICTS`) — no
  further changes needed. Nothing else in pipeline calls
  `decision_ledger.connect`/`write_verdicts`.

Related-but-out-of-scope observation: `renquant_orchestrator.ledger_attribution.connect_attribution()`
has its own, separate sqlite3-connect-with-`DEFAULT_DB`-fallback
implementation (does not call `decision_ledger.connect`), so this guard does
not cover it. All current tests for it pass `:memory:` explicitly, so it is
not exposed today, but it is the same class of risk if a future test forgets
to. Flagged, not fixed here (out of this task's scope — the task specifically
targets `decision_ledger.connect()`).

## Tests

`tests/test_decision_ledger.py` (new, common had no test file for this module
before — it moved from orchestrator without one). 9 cases: explicit
`tmp_path`/`:memory:` paths work under pytest; no-path and explicit-real-path
both raise; a canonicalization-robustness case (path spelling variant of the
real path still raises, safely, since the guard fires before any
`mkdir`/`sqlite3.connect`); the error message names the fixes; guard is a
no-op when `PYTEST_CURRENT_TEST` is absent (production behaviour unchanged);
monkeypatching `DEFAULT_DB` does not disable the guard; monkeypatching a
module attribute named `_PRODUCTION_LEDGER_RELPATH` cannot shift the guard
boundary (the relpath is an inlined string literal, not an attribute).

Full suite: 404 passed, 10 failed, 11 skipped. Confirmed via `git stash -u`
that the same 10 failures (Newey-West/bootstrap numeric mismatches under this
interpreter's scipy/statsmodels, an `importlib.metadata.PackageNotFoundError`
because `renquant-common` isn't `pip install -e`'d in this bare interpreter,
and a pre-existing raw-regime-string lint hit in the `renquant-strategy-104`
sibling) already exist on `origin/main` before this change — pre-existing/
environmental, unrelated.

Verified the real `~/renquant-data/decision_ledger.db` was untouched
(identical md5/mtime/size) before and after running the full suite twice.
