# Walk-forward fold-eligibility/selection contract (renquant-model PR #64 fix)

Date: 2026-07-21
Trigger: Codex review on `hallovorld/renquant-model#64` ("feat(g4):
walkforward-sim admissibility by model vintage, not created_at") — a P0
finding that the PR's extraction-layer admissibility module had
reimplemented `WalkForwardModelLoader.entry_as_of`'s fold-selection date
arithmetic (RenQuant `backtesting/renquant_104/kernel/walk_forward/loader.py`)
using `cutoff_date + datetime.timedelta(lookahead_days)` (calendar days) and
without `effective_train_cutoff_date` support — a second implementation of
the PIT eligibility contract that silently disagrees with the real loader
whenever a weekend falls inside the lookahead window, or whenever a fold
declares a pre-embargoed `effective_train_cutoff_date`.

## Delivered

- `src/renquant_common/walk_forward_fold_selection.py` — the pure,
  domain-neutral date-arithmetic contract mirrored exactly from the real
  loader: `feature_cutoff_date` (`effective_train_cutoff_date or
  cutoff_date`), `safe_last_label_date` (+ `pandas.tseries.offsets
  .BDay(lookahead_days)`, business days), `is_fold_eligible` (strict `<`),
  and `select_latest_eligible_fold` (the `entry_as_of` selection rule: latest
  `cutoff_date` among eligible entries). Accepts any structural record
  exposing `.cutoff_date` / `.lookahead_days` /
  `.effective_train_cutoff_date` (duck-typed via `WalkForwardFoldLike`) —
  both RenQuant's `RetrainEntry` and renquant-model's `WalkforwardFold`
  already satisfy this without an adapter class.
- `tests/test_walk_forward_fold_selection.py` — 14 tests: effective-cutoff
  preference (present/absent/empty-string), business-day vs calendar-day
  boundaries (a Friday cutoff's +1/+60 business-day answer differs from the
  calendar-day answer whenever a weekend intervenes), strict-`<` boundary
  behavior, and end-to-end fold selection preferring the effective cutoff.
- Not re-exported from the package root (submodule import only, same
  convention as `model_fingerprint` and `bundle_contract`) — root `__all__`
  and the public-API snapshot's `public_names` list are unchanged.
- Packaging: version 0.14.0 → 0.15.0 (additive), snapshot
  `tests/api_snapshot/public_api.json` version updated in the same commit.

## Consumer switch

`renquant-model` PR #64 imports this module instead of reimplementing the
date math (companion commit on that PR, `renquant-common>=0.15.0` added as a
structural pyproject dependency).

**Not done here, flagged honestly:** the real umbrella loader
(`RenQuant/backtesting/renquant_104/kernel/walk_forward/loader.py`,
`WalkForwardModelLoader._feature_cutoff_date` /
`_safe_last_label_date`) still carries its OWN inline copy of this exact
date arithmetic — it was not refactored to import
`renquant_common.walk_forward_fold_selection` in this change, since that is
a different repo's live scoring/sim kernel and out of scope for a
renquant-model PR-review fix. Until that follow-up lands, two
implementations of the same contract still exist (the loader's inline one —
verified byte-logic-identical by inspection and by this module's tests — and
this canonical one). Recommended follow-up: a small refactor PR against
RenQuant importing `feature_cutoff_date`/`safe_last_label_date` here instead
of the loader's local staticmethods, which is what fully closes the F-2
finding in `doc/arch/2026-07-04-umbrella-compliance-audit.md` ("the WF-stamp
contract is verified by THREE forked `WalkForwardModelLoader`
implementations").

## Verification

- New tests green (14/14). Full suite: 435 passed, 7 skipped, 1 pre-existing
  unrelated failure (`test_registry_lift.py::test_byte_equivalent_to_umbrella`
  — an MLflow-registry byte-diff check against a sibling `RenQuant` checkout
  whose `main` has already drifted from this repo's `main` for an unrelated
  lifted module; present before this change too). `[VERIFIED]`
- Version triple-agreement (pyproject == installed metadata == snapshot)
  verified in a fresh venv with this tree `pip install -e .`-ed. `[VERIFIED]`
