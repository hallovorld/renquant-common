# renquant-common

Shared contracts and small framework primitives for RenQuant.

Operating model: https://github.com/hallovorld/RenQuant/blob/main/doc/arch/subrepo-operating-model.md

Repository map: [RENQUANT_REPOS.md](RENQUANT_REPOS.md)

Local automation:

```bash
make test
make doctor
```

This repo is intentionally boring. It contains domain-neutral building blocks
that every other RenQuant repo can depend on without pulling in model training,
broker execution, LEAN, Torch, XGBoost, data files, or artifacts.

## Owned Surface

- Generic pipeline primitives:
  - `Task`
  - `Job`
  - `Pipeline`
  - `PipelineResult`
  - `run_parallel`
- Small shared contract helpers and fixtures.

## Boundary Rules

`renquant-common` must not import:

- broker/live modules
- model training frameworks such as `xgboost` or `torch`
- LEAN/backtesting runtime modules
- strategy-specific RenQuant repos
- local data or artifact paths

All model, inference, backtesting, and execution repos should express their
workflows as `Task` / `Job` / `Pipeline` chains from this package.

## Local Test

```bash
python -m pytest -q
```

## Source

Initial split source: `hallovorld/RenQuant` commit
`8f3e08d8d1ae1e402a78f4815efb59e3c7c66aa8`.
