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
  the same path as the real production `DEFAULT_DB`
  (`~/renquant-data/decision_ledger.db`), raises `RuntimeError` with a
  message explaining what happened and how to fix the test (explicit
  `tmp_path`, monkeypatch `DEFAULT_DB`, or mock `connect`/`write_verdicts`
  entirely). It does **not** silently redirect to a temp path — a silent
  redirect would just mask the same bug in a different, harder-to-notice
  way.
- One explicit, named escape hatch for a deliberate real-path test:
  `RENQUANT_ALLOW_LIVE_DECISION_LEDGER_IN_TESTS=1` (exact string match, not
  "merely set"), meant to be set via `monkeypatch.setenv(...)` for exactly
  one test. Checked for and confirmed absent: no test in common,
  orchestrator, or pipeline currently touches the real ledger path on
  purpose, so nothing needed the hatch wired up today — it exists for the
  hypothetical future case the task explicitly asked to guard for.
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
`mkdir`/`sqlite3.connect`); the error message names the fixes; the escape
hatch (proven via a monkeypatched `DEFAULT_DB` pointed at a `tmp_path`
stand-in, so the test never opens the real file) allows the call through
only when set to exactly `"1"`; guard is a no-op when `PYTEST_CURRENT_TEST`
is absent (production behaviour unchanged).

Full suite: 404 passed, 10 failed, 11 skipped. Confirmed via `git stash -u`
that the same 10 failures (Newey-West/bootstrap numeric mismatches under this
interpreter's scipy/statsmodels, an `importlib.metadata.PackageNotFoundError`
because `renquant-common` isn't `pip install -e`'d in this bare interpreter,
and a pre-existing raw-regime-string lint hit in the `renquant-strategy-104`
sibling) already exist on `origin/main` before this change — pre-existing/
environmental, unrelated.

Verified the real `~/renquant-data/decision_ledger.db` was untouched
(identical md5/mtime/size) before and after running the full suite twice.

## Amendment (commit `dd8e962`): immutable production-path identity

Codex (haorensjtu-dev) reviewed the guard above and requested changes.
Exact review text:

> The fail-closed pytest guard belongs in common and the
> canonicalisation/explicit escape hatch are directionally right. One
> safety hole remains: the guard derives `production_path` from mutable
> `DEFAULT_DB`. A test that monkeypatches `DEFAULT_DB` to a tmp file
> changes what is protected; a stale import or hard-coded call to the real
> `~/renquant-data/decision_ledger.db` can then pass through unguarded.
> Define the real production ledger identity as an immutable canonical
> module constant (or an equally non-mutable configuration identity),
> compare every test-time `db_path` against that value, and add a
> regression test that monkeypatches `DEFAULT_DB` away then proves an
> explicit real-path call still raises before sqlite opens it. Keep
> `DEFAULT_DB` override support for normal temp-path tests, but do not let
> it redefine the live-write protection boundary.

**The hole**: `_guard_against_live_ledger_in_tests()` re-read `DEFAULT_DB`
as a module global specifically so
`monkeypatch.setattr(decision_ledger, "DEFAULT_DB", tmp_path / "x.db")` —
the sanctioned way for an ordinary test to redirect `connect()`'s
default — would be honoured. The side effect: once *any* test in the
process monkeypatched `DEFAULT_DB` away, the guard's own notion of "the
production path" moved with it. A separate, explicit call to the real
path in the same test (or a stale import / leftover hard-coded reference
elsewhere) would then be compared against the *redirected* `DEFAULT_DB`,
not the real path, and would sail through completely unguarded — the
opposite of fail-closed.

**The fix** (`src/renquant_common/decision_ledger.py`): added
`_PRODUCTION_LEDGER_PATH: Path`, an immutable module-level constant set
once at import time (`Path.home() / "renquant-data/decision_ledger.db"`)
that is never derived from, and is fully independent of, `DEFAULT_DB`.
The guard now canonicalises and compares every test-time `db_path`
against `_PRODUCTION_LEDGER_PATH`, not `DEFAULT_DB`
(`production_path = _canonical_path(_PRODUCTION_LEDGER_PATH)`).
`DEFAULT_DB` itself is unchanged in role and value — `connect()` still
falls back to it when no `db_path` is given, and it still evaluates to
the same real path, so normal (non-test, non-monkeypatched) behaviour is
identical to before. The only behavioural change: monkeypatching
`DEFAULT_DB` no longer moves what the guard treats as "the real
production ledger" — an explicit call to the real path still raises
regardless of what `DEFAULT_DB` currently points at.

New regression test (as Codex asked), `test_monkeypatching_default_db_does_not_disable_guard`
in `tests/test_decision_ledger.py`: monkeypatches `DEFAULT_DB` to an
unrelated tmp path (simulating a normal test's legitimate redirection),
then calls `connect(_PRODUCTION_LEDGER_PATH)` — the real path, taken
directly from the immutable constant, never touching the actual file
since the guard raises before any filesystem access — and asserts it
still raises `RuntimeError`.

The two existing escape-hatch tests
(`test_escape_hatch_allows_the_configured_production_path_through`,
`test_escape_hatch_value_other_than_1_does_not_bypass`) relied on the old
mutable-comparison behaviour: they monkeypatched only `DEFAULT_DB` to a
`tmp_path` stand-in and expected the guard to treat that stand-in as "the
production path". Under the new design that pattern no longer exercises
the guard's protected-path branch at all (the guard now looks at
`_PRODUCTION_LEDGER_PATH`, untouched by that monkeypatch), so both tests
were updated to monkeypatch `_PRODUCTION_LEDGER_PATH` *and* `DEFAULT_DB`
together to the same tmp_path stand-in — still proving the escape hatch's
on/exact-match/off behaviour end to end without ever opening the real
file, just retargeted at the new immutable comparison constant instead of
the mutable default. `test_connect_with_spelling_variant_of_production_path_still_raises`
was similarly repointed to derive its path variant from
`_PRODUCTION_LEDGER_PATH` instead of `DEFAULT_DB`.

### Verification

`tests/test_decision_ledger.py`: 10/10 pass (9 original + 1 new
regression test); independently re-ran against the actual pushed commit
(not just reviewed the diff) — confirmed both updated escape-hatch tests
still pass and still prove what they proved before (guard fires on the
protected-path stand-in unless the escape hatch is exactly `"1"`).

Full suite: 405 passed, 10 failed, 11 skipped (same 10 pre-existing/
environmental failures as the baseline above — Newey-West/bootstrap
numeric tests, the `importlib.metadata` version-snapshot test, and the
`renquant-strategy-104` sibling raw-regime-string lint hit — plus exactly
one more pass than baseline, the new regression test). No new failures.

Verified the real `~/renquant-data/decision_ledger.db` was untouched
(identical md5 `b7cb5413752b732d3dd70dae0c8615ad` and mtime) before and
after running the full suite with the amended guard.
