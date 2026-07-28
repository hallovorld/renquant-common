# 2026-07-27 — classify recipe-provenance stamp keys as OPERATIONAL (round-8 contract)

STATUS:    delivered
WHAT:      model_fingerprint.py classifies `provenance_schema_version`,
           `recipe_id`, `required_axis_fields` as OPERATIONAL (with a
           documenting bullet in the section docstring). Version 0.15.0 ->
           0.15.1 (patch, classification-only; api snapshot version bumped in
           the same PR; no root-export change). FINGERPRINT_SCHEMA_VERSION
           stays 1 — see hash-preservation argument below.
WHY/DIR:   tonight's e2e probe marked both shadows' top_picks NOT ACTIONABLE:
           "artifact does not stamp provenance_schema_version/recipe_id
           (fail-closed, required-recipe-schema, round 8)". The round-8
           contract (renquant-pipeline #426; umbrella
           backtesting/renquant_104/kernel/panel_pipeline/panel_scorer.py
           `stamp_provenance_schema` / `RECIPE_REQUIRED_AXES`) requires the
           artifact itself to stamp the three fields TOP-LEVEL. But this
           repo's v1 total-classification hasher REFUSED all three
           (`UnclassifiedKeyError` — verified empirically on a copy of the
           deployed clf shadow artifact), so no v1-fingerprinted artifact
           could carry the stamp: stamping before `config_fingerprint`
           computation hard-errors, stamping after makes the artifact
           permanently unfingerprintable. Consumer chain unblocked by this
           PR: model-repo trainer stamps the three fields -> shadow round-8
           admission resolves `walkforward_only_v1` -> top_picks actionable.
           OPERATIONAL is the correct class: recipe-provenance ATTESTATIONS
           consumed by admission/staleness gates only, never by scoring —
           same category as `effective_train_cutoff_date` (already
           OPERATIONAL, training-window provenance); the trained function is
           bound via the booster bytes. The admission gate re-verifies the
           claimed recipe against actually-present axis fields, so the stamp
           cannot smuggle predictive content.
           Hash-preserving, no schema bump (0.9.2 precedent, quoted in the
           table's own NOTE): these keys previously hard-errored at stamp AND
           verify time, so NO existing artifact carrying them was ever
           v1-stampable; excluding them from the hash therefore changes no
           existing payload's hash — every payload that hashed under 0.15.0
           hashes identically under 0.15.1.
EVIDENCE:
  artifact:      tests/test_model_fingerprint.py — 51 passed (3 new: the
                 three keys stamp+verify without UnclassifiedKeyError and
                 sit in OPERATIONAL_KEYS not PREDICTIVE_KEYS; hash
                 preservation PROVEN — same payload with/without the stamp
                 (and each key alone) hashes identically; totality not
                 loosened — a lookalike unknown key `recipe_id_v2` still
                 raises UnclassifiedKeyError listing exactly that key).
  prod or exp:   full suite 453 passed / 15 skipped / 2 failed — the 2
                 failures are PRE-EXISTING at origin/main in this
                 environment (verified on a pristine origin/main worktree:
                 identical 2 failures): api-snapshot version-agreement tests
                 compare against the venv's stale editable-install metadata
                 (0.8.1); CI installs from source, so they are green there.
  existing data: census of ALL 68 prod artifacts
                 (RenQuant backtesting/renquant_104/artifacts/prod/*.json):
                 NONE carry the three keys — consistent with "previously
                 refused", so the hash-preservation argument covers every
                 real artifact on the machine, including the deployed
                 clf shadow artifact (config_fingerprint sha256:1d8f167f...,
                 which must NOT move when the trainer re-stamps).
  best-known?:   mirrors the 0.9.2 M6 stage-2 classification (same file,
                 same argument structure) — the only prior instance of
                 classifying previously-refused keys.
  scope:         classification tables + version bump + tests only; no
                 hashing-algorithm change, no root-export change, no
                 consumer code. The model-repo trainer PR (stamp at source)
                 and the deploy batch (runtime common pin advance) follow
                 separately.
NEXT:      merge -> model-repo trainer PR stamps the three fields (vendored
           constants, KEEP IN SYNC with the umbrella/pipeline panel_scorer
           contract) -> artifact regen + verification -> coordinator deploy
           batch advances the runtime common pin so the pinned verifier
           accepts the stamped artifact.
