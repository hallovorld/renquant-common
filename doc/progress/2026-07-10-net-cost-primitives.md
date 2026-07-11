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

## Versioning (r2 — Codex review)

**0.10.0 -> 0.12.0** (version-ADDRESSABLE, r2 correction): consumers that
REQUIRE the cost primitive pin `renquant-common>=0.12.0` and fail closed
below it — the r1 "feature-detect by import with a local fallback" posture
is withdrawn on the consumer side (model#43 r2 removes its fallback).
0.11.0 stays claimed by the open #27 (ALWAYS_OPEN calendar); whichever of
#27/#28 lands second rebases its version above the other (documented in
pyproject.toml). api-snapshot version updated to 0.12.0; no `__init__`
surface change (submodule-scoped API, the `model_fingerprint` precedent).
The two api-snapshot version tests remain red in THIS environment for the
pre-existing reason only (installed metadata stale at 0.8.1 — also red on
main); they pass in any env whose editable install is current.

## Canonical serialization + content identity (r2 — Codex review; unified with the round-1 push)

A parallel round-1 commit on this branch already added `to_dict` /
`cost_model_spec_from_dict` / `cost_model_content_sha256` /
`COST_MODEL_FINGERPRINT_SCHEMA_VERSION`; r2 keeps THOSE names as the one
public API and folds in strictness + golden digests:

- `CostModelSpec.to_dict()` — canonical serialization: always all four
  components, declaration order.
- `cost_model_spec_from_dict()` — strict inverse: unknown components are a
  hard error (two different intended specs must never round-trip to the
  same identity); missing components take the documented zero defaults.
- `cost_model_content_sha256(spec)` — `sha256:<hex>` over the canonical
  sorted-keys JSON (same construction as `model_fingerprint._digest`).
  **Contract: every downstream evidence artifact reporting a net-of-cost
  number (WF-gate results, run bundles, model cards) MUST stamp this
  identity next to the number**; verifiers recompute via
  `cost_model_content_sha256(cost_model_spec_from_dict(stamped))`.
- Golden digests frozen in tests (changing them = breaking identity
  change): fee-only 25bps ->
  `sha256:00db27bb…`, full 25/10/5/2 -> `sha256:bd752be7…`.

## Evidence

- 41 new tests (`tests/test_cost_model.py`): hand-computed per-side /
  round-trip / rebalance-cost / net-return cases (numbers written out in
  comments), turnover conventions, unfilled-order semantics, validation
  hard-errors, asset-agnostic boundary check.
- Serialization/identity: round-trip, strict unknown-key rejection,
  frozen golden digests, int/float canonicalization, stamped-payload
  verifier flow.
- Full suite: 356 passed / 5 pre-existing environment failures, identical
  in class and count on clean origin/main in this environment (2
  api-snapshot version tests — stale installed metadata 0.8.1; 2 sibling
  raw-regime-string scans + 1 umbrella byte-equivalence — sibling-checkout
  state, fail from the primary checkout on main too). Zero new failures.

## Merge order (r2)

**This PR must merge FIRST.** model#43 r2 (per the same Codex review)
removes its local fallback and hard-requires `renquant-common>=0.12.0`,
failing closed when `cost_model` is absent.
