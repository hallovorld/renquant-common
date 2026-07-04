# 2026-07-04 — Canonical ntfy sender (campaign B6, audit XC-4)

## What

`renquant_common/notify.py`: the fleet's single Python ntfy sender —
`send(title, body, topic=None, *, priority=None, tags=None, timeout=5.0,
env_file=None) -> bool`. Version 0.9.2 → 0.10.0 (additive minor; API
snapshot updated).

## Why

Audit #296 (orchestrator repo) finding XC-4: ~10 Python + 8 shell ntfy sender
copies across renquant-orchestrator / renquant-base-data / renquant-backtesting
with semantic drift — priority none/3/4/5, timeout 5 vs 10, and
`RENQUANT_NO_NOTIFY` honored only OUTSIDE the orchestrator (its monitors could
not be muted via the documented env). Campaign PR #297 (orchestrator) Group B.

## Contract

- Topic resolution: explicit arg > `NTFY_TOPIC` env > `NTFY_TOPIC=` line in an
  env file (explicit `env_file` arg, else `$RQ_ROOT/.env` when `RQ_ROOT` set)
  > fleet default `"renquant"`.
- `RENQUANT_NO_NOTIFY` truthy (1/true/yes/on) suppresses ALWAYS.
- Timeout standardized at 5 s.
- Never raises into the caller: every failure swallowed, counted
  (`send_failure_count()`), logged as a warning.
- `topic` stays third-positional so `post_ntfy(title, body, topic)` call sites
  and `Callable[[str, str, str], bool]` poster seams re-point via import alias.

Shell twin: `RenQuant/scripts/notify.sh` (`rq_notify`), same contract.

## Tests

`tests/test_notify.py` — 30 tests: resolution order, suppression truthy-set,
never-raise across exception types (counted + warned), priority/tags header
mapping, timeout, POST shape, poster-seam positional compatibility.

## Consumers

Re-point PRs in renquant-orchestrator (8 Python sites + 8 shell wrappers),
renquant-base-data (1), renquant-backtesting (1); each bumps its
renquant-common floor to 0.10. Merge order: this PR first.
