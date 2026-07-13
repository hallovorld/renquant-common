# fix(V-003): move decision_ledger persistence to renquant-common

**Date**: 2026-07-13
**PR**: common fix/v003-decision-ledger-to-common

## Change

Move `connect`, `write_verdicts`, `DDL`, and related constants from
`renquant_orchestrator.decision_ledger` to `renquant_common.decision_ledger`.

This fixes architecture violation V-003: pipeline was importing persistence
functions directly from orchestrator (`from renquant_orchestrator.decision_ledger
import connect, write_verdicts`), creating a reverse dependency that violates
the subrepo operating model's dependency direction (pipeline should not depend
on orchestrator).

With this change, both orchestrator and pipeline import from common,
restoring a one-directional dependency graph.

## Companion PRs

- **renquant-pipeline**: update import in `task_decision_ledger.py`
- **renquant-orchestrator**: re-export from common for backward compat
