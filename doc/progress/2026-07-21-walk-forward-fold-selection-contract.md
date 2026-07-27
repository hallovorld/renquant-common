# Walk-forward fold-eligibility/selection contract (renquant-model PR #64 fix)

## STATUS
Shared canonical module + tests DONE on this branch (v0.15.0, additive);
runtime-consumer refactor DONE in the paired renquant-pipeline#214
(codex-APPROVED, merges immediately after this PR). Awaiting #33 clearance.

## WHAT
- `src/renquant_common/walk_forward_fold_selection.py` — the pure,
  domain-neutral PIT date-arithmetic contract mirrored exactly from the real
  loader: `feature_cutoff_date` (`effective_train_cutoff_date or
  cutoff_date`), `safe_last_label_date` (BDay lookahead arithmetic),
  `is_fold_eligible` (strict `<`), `select_latest_eligible_fold` (latest
  `cutoff_date` among eligible; Python `max` keeps the FIRST maximal element
  in iteration order — callers reproducing the pipeline loader's historical
  last-among-ties rule feed entries descending, pinned by
  renquant-pipeline#214's parity test). Duck-typed `WalkForwardFoldLike`
  (both `RetrainEntry` and renquant-model's `WalkforwardFold` satisfy it).
- `tests/test_walk_forward_fold_selection.py` — 14 tests: effective-cutoff
  preference (present/absent/empty-string), business-day vs calendar-day
  boundary divergence, strict-`<` boundaries, end-to-end selection.
- Not re-exported from the package root (submodule import only, same
  convention as `model_fingerprint`/`bundle_contract`); public-API snapshot
  `public_names` unchanged; version 0.14.0 → 0.15.0 additive with snapshot
  updated in the same commit.

## WHY/DIR
Codex P0 on renquant-model#64: the extraction-layer admissibility module had
reimplemented `WalkForwardModelLoader.entry_as_of` date arithmetic with
CALENDAR-day lookahead and no `effective_train_cutoff_date` support — a
silently divergent second implementation whenever a weekend falls inside the
lookahead window or a fold declares a pre-embargoed effective cutoff. One
canonical implementation in renquant-common, consumed by both the live
loader (pipeline#214) and extraction (model#64), removes the fork class
(umbrella audit F-2: "verified by THREE forked WalkForwardModelLoader
implementations").

## EVIDENCE
- This branch: `pytest tests/test_walk_forward_fold_selection.py` 14/14.
- Paired consumer: renquant-pipeline#214 imports this module at loader top,
  delegates `_feature_cutoff_date`/`_safe_last_label_date`/selection, and its
  `tests/test_wf_fold_selection_parity.py` (29 cases) drives the LIVE loader
  path and this selector on the same manifest/boundary fixtures — full
  pipeline suite with this branch on PYTHONPATH: 2007 passed, 8 skipped, 0
  failed; codex's independent focused validation on the queued heads: 50/50.
- Local `test_api_snapshot` version-assertion failures in a dev venv are
  install-version artifacts (venv carries the released common; branch
  pyproject pins 0.15.0); green on CI which installs the branch.

## NEXT
Codex clears #33 → merge → release 0.15.0 on main → merge pipeline#214
immediately after (its CI turns green against released common) →
renquant-model#64 consumes `renquant-common>=0.15.0` → sim-time provenance
design (pipeline#215) proceeds on the single canonical selector.
