# CLAUDE.md

Canonical operating model:
https://github.com/hallovorld/RenQuant/blob/main/doc/arch/subrepo-operating-model.md

Local repo map: `RENQUANT_REPOS.md`.

Branch policy: `main` is the stable interface consumed by other repos and
automation. Experiments, optimizations, and large upgrades happen on feature
branches, then merge back only after tests and integration checks pass.

## Repo Role

`renquant-common` owns shared, domain-neutral RenQuant contracts and pipeline
primitives. Other RenQuant repos may depend on this repo; this repo must not
depend on other RenQuant repos.

## Hard Boundaries

- Keep this package small and boring: `Task`, `Job`, `Pipeline`, schemas, and
  contract helpers.
- Do not import broker/live modules, LEAN/backtesting modules, model training
  frameworks, strategy configs, data files, artifact stores, or local paths.
- Do not add strategy-specific trading logic here.
- Do not delete or empty the source umbrella repo at
  `/Users/renhao/git/github/RenQuant`.

## Workflow

- Express new shared orchestration behavior as tested pipeline primitives.
- Large changes use a feature branch; `main` must stay runnable.
- Every behavior change needs focused tests.
- Run before commit:

```bash
make test
make doctor
```
