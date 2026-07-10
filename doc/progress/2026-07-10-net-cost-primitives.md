# Generic net-of-cost accounting primitives (crypto RFC D-C8a)

Date: 2026-07-10
PR: feat(cost): generic net-of-cost accounting primitives (crypto RFC D-C8a)

## What

New module `renquant_common.cost_model` — the ONE authoritative
cost-accounting primitive required by the merged crypto trading RFC
(renquant-orchestrator `doc/design/2026-07-10-crypto-trading-rfc.md` §4.4
CORRECTED, gap M1, deliverable D-C8a): fees / spread / slippage /
increment-rounding bps over realized turnover, consumed IDENTICALLY by
WF-gate replay evaluation and live runtime accounting so the two sides can
never drift (the fingerprint-unification lesson, M6, applied to costs).

API (submodule-scoped, like `model_fingerprint` at introduction — not in the
package `__init__`):

- `CostModelSpec(fee_bps, spread_bps, slippage_bps, increment_rounding_bps)`
  — frozen, validated (finite, >= 0), every component defaulting to 0 so a
  fee-only model is explicit.
- `per_side_cost_bps(spec)` = fee + spread/2 + slippage + rounding;
  `round_trip_cost_bps(spec)` = 2x.
- `turnover_breakdown(prev_weights, next_weights)` → buy/sell/traded
  fractions with BOTH conventions named (`traded_fraction` = Σ|Δw| — the
  quantity costs apply to; `one_sided` = half — the statistic), so no
  consumer can silently halve or double costs.
- `rebalance_cost_fraction(prev, next, spec)` = per-side rate × traded
  fraction.
- `realized_traded_fraction(intended, fill_ratio)` — the RFC §4.4
  rejected/unfilled/resting-order convention pinned as a named primitive: a
  fill that never happened costs nothing (and, caller obligation, earns
  nothing).
- `apply_costs_to_period_returns(gross, traded, spec)` — net_t = gross_t −
  rate × traded_t; length mismatch and non-finite inputs are hard errors.

## Boundaries (D-C8 split, RFC §9.3 resolved)

Asset-agnostic by contract and by test: no crypto/BTC/venue-specific symbol
or default exists here. The crypto taker-fee default, the BTC-baseline
comparison and the crypto promotion decision live in renquant-model
(D-C8b), consuming this primitive.

## Versioning

NO version bump in this PR: renquant-common #27 (ALWAYS_OPEN calendar,
open) already claims 0.11.0, and this environment's installed metadata is
stale (0.8.1) so the api-snapshot triple-agreement tests cannot verify a
bump anyway (pre-existing env failure, also red on main). The module is
submodule-scoped (no `__init__`/api-snapshot surface change); consumers
feature-detect by import with an identical local fallback (the
pipeline#183 soft-consume pattern), so merge order stays free with no pin
bump. Version rolls up with the next release PR.

## Evidence

- 33 new tests (`tests/test_cost_model.py`): hand-computed per-side /
  round-trip / rebalance-cost / net-return cases (numbers written out in
  comments), turnover conventions, unfilled-order semantics, validation
  hard-errors, asset-agnostic boundary check.
- Full suite: 348 passed / 5 pre-existing environment failures, identical
  on clean origin/main in this environment (2 api-snapshot version tests —
  stale installed metadata 0.8.1; 2 sibling raw-regime-string scans + 1
  umbrella byte-equivalence — sibling-checkout state, fail from the primary
  checkout on main too). Zero new failures.

## Merge order

Free: the consumer (renquant-model crypto fee gate, D-C8b) soft-consumes
with an identical fallback. This PR SHOULD still merge first so the
canonical exists before the fallback ever runs in anger.
