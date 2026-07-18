# Bundle-contract fixture vectors — phase 3 PR-B (GOAL-5 AC4)

Date: 2026-07-18
Spec: RFC "transactional artifact bundles for the 104 serving pair"
(RenQuant#492, `doc/design/2026-07-17-artifact-bundle-transactionality.md`
§2.5): "a contract fixture in renquant-common pins the verdict semantics
both sides test against." Phase 2 (renquant-pipeline#206) shipped the
five vectors inside the pipeline repo with an explicit note that the
renquant-common move is the phase-3 binding step; this PR is that move.

## Delivered

- `src/renquant_common/contracts/bundle_contract_vectors.json` — the five
  phase-2 vectors, copied BYTE-FOR-BYTE from
  renquant-pipeline@ad3520f8 `tests/fixtures/bundle_contract/vectors.json`
  (sha256 `710d5977…6a7f7e` pinned in the new test): matching-legacy,
  matching-v1, mismatched (the 05-27/06-22/07-01/07-14→16
  orphaned-binding incident shape), missing-binding,
  cross-schema-comparison-refused. Each case carries member payloads, the
  pinned member serialization, a schema-v1 manifest, the
  `accept_legacy_stamps` flag, and the expected `PairVerdict` fields.
- `src/renquant_common/contracts/bundle_contract.py` — stdlib-only loader
  (`load_bundle_contract_vectors()`, `bundle_contract_vector_cases()`,
  `BUNDLE_CONTRACT_VECTORS_RESOURCE`). Data + loader ONLY: verdict
  semantics stay defined by the phase-2 API; this repo pins what both
  sides must reproduce, it does not reimplement it. Not re-exported from
  the package root (submodule import, same convention as
  `model_fingerprint`) — root `__all__` unchanged.
- `tests/test_bundle_contract_vectors.py` — the pinning test: byte
  provenance (sha256), contract identity (`contract`,
  `contract_version == 1`, member names, serialization rule), the EXACT
  five case names, every case's expected verdict frozen in the test
  itself, structural completeness per case, loader freshness.
- Packaging: version 0.13.0 → 0.14.0 (additive), snapshot
  `tests/api_snapshot/public_api.json` version updated in the same PR
  (public names unchanged); `[tool.setuptools.package-data]` added so
  wheels ship the vectors (and `py.typed`).

## Consumers switch in a FOLLOW-UP (explicitly not here)

renquant-pipeline (`tests/test_bundle_contract.py`, currently reading its
local `tests/fixtures/bundle_contract/vectors.json`) and
renquant-artifacts (phase-3 PR-A `tests/test_bundle_contract_binding.py`,
currently reading the pipeline sibling checkout's copy) will repoint to
`renquant_common.contracts.bundle_contract` in follow-up PRs once this
lands (consumer floor: renquant-common>=0.14). This PR deliberately does
NOT modify renquant-pipeline (per the phase-3 task scoping); until the
follow-ups land, the byte-provenance pin here guarantees the copies are
identical, so nothing can drift silently in the interim.

## Verification

- New pinning tests green; full suite: 415 passed, 10 skipped — the only
  failures on this machine are the 4 pre-existing
  `test_no_raw_regime_strings` sibling-scan hits (raw regime literals in
  OTHER repos' current mains; those tests skip in CI where siblings are
  not checked out) — identical before and after this change. [VERIFIED]
- Version triple-agreement (pyproject == installed metadata == snapshot)
  verified in a fresh venv with this tree `pip install -e .`-ed.
  [VERIFIED]
