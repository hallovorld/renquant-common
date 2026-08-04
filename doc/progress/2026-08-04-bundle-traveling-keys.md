# 2026-08-04 — LiveRunBundle: the traveling keys are typed, not silently dropped

STATUS:    schema + regressions + semver bump (0.15.1 → 0.16.0, additive)
WHAT:      orch#769 item 11 measured that both parity producers write keys
           the shared schema silently drops from validation (pydantic
           extra="ignore"): `metadata` (both producers) and
           `smalln_ledger` (bridge; a non-empty dict or the pipeline#207
           literal "absent"). Same failure class the wf_gate_provenance
           field note documents from measurement (2026-07-31: a malformed
           value VALIDATED CLEAN by being dropped). Both keys are now
           typed optional fields with model-validator checks: metadata
           non-empty-dict-or-None; smalln_ledger dict(non-empty) |
           "absent" | None, any other form refuses. 4 new tests incl. the
           measured malformed-clean class as a refusal regression.
WHY/DIR:   GOAL-5 AC6 R4 family — the bundle contract must cover what
           actually travels. Additive, backward-compatible: absent keys
           remain valid (None defaults), so every existing producer
           validates unchanged; only previously-silently-dropped
           MALFORMED values now refuse.
EVIDENCE:  tests/test_schemas.py 23 passed; full suite 495 passed with
           the SAME 5 machine-local failures as clean main ef7726d
           (version-snapshot drift + sibling-checkout scans + umbrella
           byte-equivalence — control run performed; none mine).
NEXT:      merge → umbrella pin advance for renquant-common in a future
           batch → orch's ops/bundle_producer_key_audit.py reads 0 unread
           keys; item 11 then leaves orch#769 via that measurement.
