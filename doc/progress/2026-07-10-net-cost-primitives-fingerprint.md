# cost_model fingerprint round-1 (Codex review of #28)

Date: 2026-07-10
PR: fix(cost): add to_dict/content-fingerprint helpers (D-C8a round-1)

## What

Codex's review of `renquant_common.cost_model` (D-C8a) raised two findings. This
round addresses the one that isn't sequencing-dependent:

- Added `CostModelSpec.to_dict()`, `cost_model_spec_from_dict()`, and
  `cost_model_content_sha256()` — the same "M6 fingerprint" pattern already used for
  `model_content_sha256` (sorted-keys canonical JSON, `sha256:`-prefixed digest,
  `allow_nan=False`). WF-gate replay evaluation and live runtime accounting can now
  stamp/verify the SAME hash of the SAME fee/spread/slippage/rounding numbers they
  actually used — sharing formulas was not sufficient, since the two sides could
  still silently apply different parameter VALUES.
- `COST_MODEL_FINGERPRINT_SCHEMA_VERSION` travels with a stamped fingerprint so a
  verifier can distinguish "different cost parameters" from "different contract"
  (same convention as `model_fingerprint.FINGERPRINT_SCHEMA_VERSION`).

## Not addressed in this round (sequencing-dependent)

Codex's OTHER finding — "this adds a new imported module with no release/version
change while downstream model code is expected to consume it... coordinate an
explicit common release version after the calendar and cost APIs are settled" — is
real but can't be resolved from inside this PR alone: `renquant-common#27` (the
ALWAYS_OPEN calendar, also unmerged, also based on 0.10.0) already claims the
0.10.0 → 0.11.0 bump for ITS OWN addition. Bumping to 0.11.0 in THIS branch too would
just create a second, unrelated PR claiming the identical version number for
different new content — not a real fix, just a coincidental collision waiting to
surface when whichever PR merges second needs to re-bump anyway. The correct
sequence: merge #27 first, then bump this branch to whatever version follows it
(0.12.0 if additive-only), and only then can `renquant-model`'s D-C8b PR (#43)
require that version and drop its own local fallback — the exact same dependency
chain already documented for `renquant-pipeline#183`.

## Tests

`tests/test_cost_model.py::TestFingerprint` (6 new cases): round-trip through
`to_dict`/`from_dict` preserves the fingerprint, missing dict keys default like the
constructor, different params produce different fingerprints, identical params
produce identical fingerprints, fingerprint format is a stable 64-hex-char
`sha256:`-prefixed string, `to_dict()` has every field explicit. Verified meaningful:
reverting only the source change makes the test module fail to IMPORT (the new
names don't exist), confirming the tests are not vacuous. Full suite: 339 passed, 18
skipped, 2 pre-existing unrelated failures (installed-package-version staleness in
this shared venv — reproduces identically on the pre-fix branch).

## Next

None for this PR beyond the fingerprint helpers. Version-bump sequencing tracked
above; revisit once #27 merges.
