# 2026-07-26 — metrics.harvest_stats: shared research primitives

STATUS:    additive module + 5 bug-pinning unit tests
WHAT:      metrics/harvest_stats.py — per_date_rank_ic, top_n_spread (winsorize-aware),
           shuffle_labels_within_date (matched placebo), moving_block_ci
           (horizon-matched block), paired_clean_series (common-dates guard).
WHY/DIR:   Five hand-rolled copies of these primitives in the 07-24/25 research line
           each minted a distinct bug (guard surrogate; anchor eval-horizon; frozen
           price denominator; null merge keys; degenerate cross-sections). Each unit
           test pins one class. Standing rule: screens IMPORT these, never re-derive.
EVIDENCE:
  artifact:      tests/test_harvest_stats.py — 9 passed (4 added post-review:
                 moving_block_ci fail-closed params; top_n_spread tie-break
                 row-order invariance; top_n_spread fails closed on a
                 duplicated tie_col within a date; top_n_spread does not
                 leak rows through a duplicated DataFrame index)
  prod or exp:   library code; no runtime consumer changed in this PR
  existing data: complements metrics.block_bootstrap (stationary CI) — no overlap;
                 suite 447+6: the 6 failures VERIFIED PRE-EXISTING on main
                 (reproduced with this PR's files removed: 443+6)
  best-known?:   moving-block kept because the frozen preregs specify it;
                 sampling delegates to arch.bootstrap.MovingBlockBootstrap
                 (repo convention) instead of a hand-rolled loop.
                 top_n_spread now requires tie_col to be unique within each
                 date (raises ValueError otherwise — a residual tie in
                 (score_col, tie_col) is ambiguous and a stable sort alone
                 does not resolve it) and reads the sorted label
                 positionally instead of via `.loc[sorted_index]`, so a
                 duplicated DataFrame index can no longer pull extra rows
                 into the top-N (post second codex review, 2026-07-26)
  scope:         additive; executor adoption is follow-up work in their repos
NEXT:      migrate model-repo executors to import these (mechanical, separate PRs).
