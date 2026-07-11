# Canonical crypto pair/slug symbol helper (D-C1, other half)

Date: 2026-07-11
PR: feat(calendar): pair/slug symbol helper (D-C1 completion)

## What

Adds `renquant_common.pair_slug` — `pair_slug`/`slug_pair`/`as_pair`/`as_slug` — the
other half of D-C1 alongside #27's ALWAYS_OPEN calendar mode (already merged,
0.11.0). `renquant-base-data#41`'s crypto bars ingestion (D-C2) carries a local,
explicitly-labeled stand-in for exactly this ("local stand-in for renquant-common
D-C1") pending this primitive's existence; base-data's own most recent commit
(cf3f5930) already repointed its calendar half onto #27 and explicitly noted "no
pair_slug/slug_pair helper exists in renquant-common main" as the remaining gap —
this PR closes it.

Semantics (mirrors base-data's local implementation exactly, so a repoint is a
behavior-identity no-op): `pair_slug("BTC/USD") == "BTC-USD"`, `slug_pair`
its exact inverse, both strict (malformed input raises rather than silently
producing a colliding cache key or ambiguous path — a bare `/` in a pair used as
a path component would create a nested directory). `as_pair`/`as_slug` accept
either canonical form and round-trip through the strict helpers.

Submodule-scoped (not re-exported at the package root) — same convention as
`cost_model`/`model_fingerprint` at their introduction.

## Tests

`tests/test_pair_slug.py` (27 cases): round-trip across 4 representative pairs,
case/whitespace normalization, the exact malformed-input battery base-data's own
tests already pin (mirrored verbatim so a future repoint's tests need zero
changes), plus `as_pair`/`as_slug` acceptance and rejection cases. Full suite: 351
passed, 18 skipped, 2 pre-existing unrelated failures (installed-package-version
staleness in the shared venv, reproduces identically on main).

## Version

0.11.0 -> 0.12.0. `#28` (net-cost primitives) also claims 0.12.0 concurrently on
a separate branch; whichever of #28/this PR merges second rebases its version
above the other (same collision-handling convention #27/#28 already used and
documented in each PR's own version-bump comment).

## Next

Once merged, `base-data#41` can repoint its local `pair_slug`/`slug_pair`/`_as_pair`/
`_as_slug` stand-in onto this module and drop the local copy — completing D-C1
end to end (calendar + symbol helper both canonical).
