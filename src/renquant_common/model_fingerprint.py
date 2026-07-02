"""Shared model-content fingerprint — total classification, schema-versioned.

Implements schema v1 of the contract designed in renquant-orchestrator
``doc/design/2026-07-02-m6-fingerprint-unification.md`` (M6/R2 of the
unified plan). Three recurring fail-closed no-trade incidents (2026-05-27,
2026-06-22, 2026-07-01) traced to ``model_content_sha256`` disagreeing
across repos:

* renquant-pipeline ``panel_scorer.py`` — SUBTRACTIVE: hash everything not
  on a mutable-key denylist, so any NEW key is hashed by default
  (false MISMATCH when an operational field appears — the observed
  incident class).
* renquant-model ``fit_calibrator_alpha158_fund.py`` — ADDITIVE: hash an
  explicit allowlist, so any NEW key is ignored by default (false MATCH
  when a predictive field appears — the silent, more dangerous class).

Both silent defaults are removed here by TOTAL CLASSIFICATION: every
top-level payload key MUST appear in exactly one of :data:`PREDICTIVE_KEYS`
or :data:`OPERATIONAL_KEYS`. An unclassified key is a hard error at stamp
time and at verify time (:class:`UnclassifiedKeyError`) — never a silent
default in either direction.

Schema v1 scope (per the design's staged rollout — migrations of call
sites are separate PRs):

* Classification is over TOP-LEVEL keys; a nested value (e.g. ``params``,
  ``metadata``) inherits its top-level key's classification as one atomic
  unit — the design §2a's "explicitly frozen, named sub-structure treated
  as an atomic PREDICTIVE/OPERATIONAL unit". Finer-grained key-path tables
  and artifact-family scoping land with a ``FINGERPRINT_SCHEMA_VERSION``
  bump per design §2b.
* Floats are canonicalized EXACTLY via the shortest round-tripping decimal
  representation (``repr(float)``, the ``json`` module's float encoding) —
  never lossy rounding (design §2b, r3 correction).
* Non-finite floats (``NaN``/``Infinity``) in the PREDICTIVE subset are
  REJECTED at stamp time (:class:`NonFiniteValueError`) — an invalid model
  state must not be blessed with a stable fingerprint (design §2b).
* A schema-version gap at verify time is its own explicit error
  (:class:`VersionGapError` — "re-stamp under vN", an auditable
  operation), never conflated with a content mismatch
  (:class:`MismatchError`).

Ownership split (design §2b): the classification TABLES are a modeling
contract — the model repo reviews changes to them; the MECHANISM
(canonicalization, hashing, version stamping) is shared infrastructure
owned here. Any table change requires a ``FINGERPRINT_SCHEMA_VERSION``
bump.

This supersedes the 2026-07-01 hot-fix extraction (renquant-common PR
#18), which copied the pipeline's subtractive denylist verbatim and so
retained its silent default. Do not re-fork this logic into other repos;
import it from here.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

FINGERPRINT_SCHEMA_VERSION = 1

#: Fields whose value changes what (or how) the artifact predicts, or the
#: pairing identity downstream consumers (calibrators) are bound to.
#: Union of the two divergent implementations' predictive knowledge as of
#: 2026-07-02: renquant-model's explicit allowlist + renquant-pipeline's
#: ``_PREDICTIVE_CONTENT_HINTS``.
#:
#: Classification judgment calls (see the tests' cross-repo fixtures):
#:
#: * ``label_col`` — the one direct CONFLICT between the two impls: the
#:   model repo's allowlist HASHES it, the pipeline's denylist EXCLUDES
#:   it. Resolved PREDICTIVE: the calibrator fit is bound to the label
#:   horizon (``fit_calibrator_alpha158_fund._infer_raw_er_label`` derives
#:   the raw-ER column from ``label_col``), so an artifact re-labeled to a
#:   different horizon must NOT silently keep pairing with the old
#:   calibrator; and the model repo owns modeling-contract judgment per
#:   design §2b.
#: * ``kind`` — known to neither table but hashed today by the pipeline's
#:   subtractive impl (absent from its denylist) and present in real
#:   artifacts; it selects which scorer interprets the artifact, so it is
#:   content, not bookkeeping.
#: * ``params`` — in the model repo's allowlist only, but the pipeline's
#:   subtractive impl also hashes it by default; uncontroversial.
#: * ``lookahead_days`` — MOVED here from OPERATIONAL (r2 correction).
#:   Traced to real usage: ``renquant_model_common.global_calibrator``'s
#:   ``_native_lookahead_days()`` reads this field directly from artifact
#:   metadata, and ``expected_return``/``expected_return_vec`` use it to
#:   RESCALE the calibrated expected-return output by the ratio
#:   ``horizon_days / native`` whenever they differ. A stamped
#:   ``lookahead_days`` that drifts from the artifact's true label horizon
#:   therefore mechanically changes the calibrator's numeric output at
#:   inference time — the definition of predictive content, not inert
#:   bookkeeping. Inheriting the old pipeline denylist's classification
#:   without checking this call site is exactly the failure mode this
#:   fingerprint module exists to close. Additionally, at calibrator-FIT
#:   time the stamped value is read WITH PRECEDENCE over label-derived
#:   inference (``renquant_model_patchtst/fit_calibrator.py:365-367``:
#:   ``_metadata_value(..., "lookahead_days", "lookahead_days_used") or
#:   _infer_label_lookahead_days(...)``), and the serving μ horizon for
#:   QP/rotation resolves through it
#:   (``job_panel_scoring.py::_calibrator_native_horizon_days``).
#: * ``label`` — MOVED here from OPERATIONAL (r2 correction). Multiple
#:   artifact-producing scripts (``RenQuant/scripts/qlib_linear_baseline.py``,
#:   ``backfill_doe_to_mlflow.py``, ``portfolio_simulation_multihorizon.py``)
#:   stamp this field with the same role as ``label_col`` — a description
#:   of the target/horizon the artifact was trained against (e.g.
#:   ``"multi_horizon_ensemble (fwd_5d + fwd_20d + fwd_60d, ...)"``) — for
#:   artifact families that use a free-text ``label`` instead of a strict
#:   ``label_col`` column name. Since ``label_col`` is already PREDICTIVE
#:   for the same reason (target-definition identity), ``label`` must be
#:   too, for consistency across artifact families that use either field.
#:   Confirmed by a LIVE runtime reader, not writers alone: the PatchTST
#:   calibrator fit resolves its label column as ``label_col or
#:   _metadata_value(checkpoint, sidecar, "label_col", "label")``
#:   (``renquant_model_patchtst/fit_calibrator.py:363``) — when
#:   ``label_col`` is absent, ``label`` ALONE selects the fitted label
#:   column, the inferred raw-ER column, and the lookahead inference.
#: * ``side_label`` — investigated and CONFIRMED OPERATIONAL: it identifies
#:   which experimental/side training CONFIG produced the artifact
#:   (``renquant_model_gbdt/pipeline.py``'s ``side_label`` field,
#:   ``renquant_pipeline/pp_training_full.py``'s side-config-path safety
#:   check) — lineage/provenance bookkeeping used to prevent a side
#:   experiment's output from being mistaken for a production artifact. No
#:   call site was found where its VALUE changes how predictions are
#:   interpreted; it stays in OPERATIONAL_KEYS below.
#: * ``version`` — investigated: only known reference is the pipeline's
#:   own denylist comment ("artifact-format version, not a model
#:   parameter"); no call site found where it selects output
#:   interpretation (scorer selection branches on ``kind``,
#:   ``panel_scorer.py:235``). Stays in OPERATIONAL_KEYS below pending
#:   contrary evidence.
#:
#: REAL-ARTIFACT CENSUS additions (r2 review directive "inspect real
#: artifact families", applied to the live umbrella tree 2026-07-02: the
#: production artifact ``data/panel-ltr-prod-alpha158-fund-fwd60d.json``,
#: the shadow-lane artifacts ``data/shadow_analyst/*.json``, and every
#: field ``RenQuant/scripts/train_production_model.py::build_artifact``
#: writes). These fields exist in REAL artifacts but were known to
#: neither legacy table; without classifying them, stamping a real
#: production artifact raises UnclassifiedKeyError, so stage-1 dual-write
#: (design §2c) would CRASH real training runs instead of shadowing them.
#: Tie-break rule for the judgment calls below: a field goes OPERATIONAL
#: only when (a) no scoring/calibration code path reads it AND (b) it is
#: post-training-mutable bookkeeping or gate/evaluation evidence. A
#: train-time-stamped, never-mutated field in doubt goes PREDICTIVE: a
#: redundant PREDICTIVE classification of an immutable field cannot
#: create a false MATCH (the dangerous class), and cannot create a false
#: MISMATCH unless a post-training mutation occurs — which stage-1
#: shadow telemetry would surface loudly.
#:
#: * ``feature_source_contract`` / ``feature_preprocess_version`` /
#:   ``feature_raw_clip_fit_split`` — written by ``build_artifact``
#:   alongside the clip/norm fields (``train_production_model.py:819-822``;
#:   present in the live shadow artifacts): they declare/version the
#:   serving-side preprocessing semantics of the PREDICTIVE normalization
#:   fields. Train-time-stamped, never post-training-mutated ⇒ PREDICTIVE
#:   per the tie-break rule.
#: * ``feature_addendum_v1`` — recipe-variant identity stamp
#:   (``build_artifact``, Track B): mirrors PREDICTIVE ``feature_cols``
#:   membership and pins the variant for WF-gate recipe matching;
#:   train-time-immutable ⇒ PREDICTIVE per the tie-break rule.
PREDICTIVE_KEYS = frozenset({
    "booster_raw_json",
    "params",
    "kind",
    "feature_cols",
    "feature_columns",
    "feature_means",
    "feature_stds",
    "feature_norm_kind",
    "feature_norm_kinds",
    "feature_raw_clip_low",
    "feature_raw_clip_high",
    "feature_raw_clip_fit_split",
    "feature_preprocess_version",
    "feature_source_contract",
    "feature_addendum_v1",
    "label_col",
    "label",
    "lookahead_days",
    "coef",
    "intercept",
    "clip_sigma",
    "state_dict",
    "config_dict",
    "model_bytes",
    "model_bytes_b64",
})

#: Fields that never change predictions: paths, file hashes, gate results,
#: stamps, CV/OOS bookkeeping, promotion state, audit IDs. Union of the
#: pipeline's ``_MUTABLE_ARTIFACT_KEYS`` denylist (minus ``label_col``,
#: ``label``, and ``lookahead_days`` — all three moved to PREDICTIVE
#: above, see that set's docstring for the evidence) plus this module's
#: own stamp fields, the legacy stamp-field aliases the model repo reads
#: as fallbacks (``model_fingerprint``, ``fingerprint``), and the
#: real-artifact census additions (see the PREDICTIVE set's docstring for
#: the census sources):
#:
#: * ``best_iter`` — training-quality evidence read ONLY by the preflight
#:   admission gate P-BEST-ITER (``preflight.py:698-772``, with an
#:   ``eval_ic`` escape clause — the same gate that reads ``eval_ic``,
#:   which both legacy impls classify operational). Scoring loads the
#:   booster exclusively from ``booster_raw_json``
#:   (``panel_scorer.py:251``), so ``best_iter`` cannot alter scores.
#:   Present in the LIVE production artifact — unclassified, it would
#:   make the real artifact unstampable.
#: * ``cutoff_date`` / ``cutoff_embargo_days`` /
#:   ``effective_train_cutoff_date`` / ``train_start_date`` /
#:   ``effective_train_start_date`` / ``train_window`` — training-window
#:   provenance consumed by leakage/staleness audit gates, not by scoring
#:   or calibration; the trained function itself is already bound via the
#:   booster bytes.
#: * ``sentiment_runtime_gate_*`` / ``sentiment_gate_contract`` —
#:   contract ATTESTATIONS: the admission gate only checks presence/kind
#:   (``artifact_contract.py::has_sentiment_runtime_gate_contract``);
#:   actual serving behavior derives from ``feature_cols`` (PREDICTIVE)
#:   plus runtime config (``sentiment_runtime_gate_requirement``), never
#:   from these stamped values.
OPERATIONAL_KEYS = frozenset({
    # Paths, file hashes, fingerprint stamps.
    "metadata",
    "artifact_path",
    "artifact_sha256",
    "artifact_fingerprint",
    "model_content_fingerprint",
    "fingerprint_schema_version",
    "model_fingerprint",
    "fingerprint",
    "config_fingerprint",
    "config_fingerprint_fields",
    # Gate results / promotion state / contract attestations.
    "wf_gate_metadata",
    "promotion_status",
    "promotion_gating_reason",
    "sentiment_runtime_gate_contract",
    "sentiment_runtime_gate_trained",
    "sentiment_runtime_gate_feature_cols",
    "sentiment_runtime_gate_disabled_regimes",
    "sentiment_runtime_gate_zeroed_rows",
    "sentiment_runtime_gate_warmup_zeroed_rows",
    "sentiment_runtime_gate_missing_regime_policy",
    "sentiment_runtime_gate_policy",
    "sentiment_gate_contract",
    # Training bookkeeping and evaluation evidence (post-training metadata:
    # stamping these changes the JSON bytes but not the predictions —
    # previously caused 3 calibrator rebinds in one day).
    "trained_date",
    "training_notes",
    "panel_shape",
    "n_train_rows",
    "training_train_ic",
    "val_mean_ic",
    "val_median_ic",
    "test_mean_ic",
    "test_median_ic",
    "oos_mean_ic",
    "oos_std_ic",
    "oos_per_fold_ic",
    "eval_ic",
    "best_iter",
    "cv_method",
    "cv_embargo_days",
    "cv_folds",
    "cv_n_splits",
    "train_run_id",
    # Training-window provenance (audit/leakage-gate inputs).
    "cutoff_date",
    "cutoff_embargo_days",
    "effective_train_cutoff_date",
    "train_start_date",
    "effective_train_start_date",
    "train_window",
    "version",  # artifact-format version, not a model parameter
    "side_label",
})


class FingerprintError(ValueError):
    """Base error for the fingerprint contract.

    Subclasses ``ValueError`` so legacy call sites wrapping the old
    implementations in ``except ValueError`` keep failing closed during
    migration.
    """


class UnclassifiedKeyError(FingerprintError):
    """A payload key is in neither PREDICTIVE_KEYS nor OPERATIONAL_KEYS.

    Raised at stamp time AND at verify time — the total-classification
    contract's core property (design §2b): never silently hash a new key
    (the pipeline's false-MISMATCH default), never silently ignore one
    (the model repo's false-MATCH default).
    """

    def __init__(self, keys: list[str]) -> None:
        self.keys = tuple(sorted(str(k) for k in keys))
        super().__init__(
            "unclassified payload key(s) "
            f"{list(self.keys)}: every key must be classified in "
            "renquant_common.model_fingerprint.PREDICTIVE_KEYS or "
            "OPERATIONAL_KEYS (a modeling-contract change reviewed by the "
            "model repo, with a FINGERPRINT_SCHEMA_VERSION bump). Refusing "
            "to guess — silent defaults are the root cause of the "
            "2026-05-27/06-22/07-01 incidents."
        )


class NoPredictiveContentError(FingerprintError):
    """The payload contains no PREDICTIVE-classified field at all."""


class NonFiniteValueError(FingerprintError):
    """A PREDICTIVE field contains NaN/Infinity — an invalid model state.

    Design §2b: rejected at stamp time (the artifact is never written),
    never canonicalized into a stable, hashable representation.
    """

    def __init__(self, key_path: str, value: float) -> None:
        self.key_path = key_path
        super().__init__(
            f"non-finite value {value!r} in PREDICTIVE field {key_path!r}: "
            "this indicates an invalid model state (training failure or "
            "numerical instability) and must not be fingerprinted"
        )


class VersionGapError(FingerprintError):
    """The artifact was stamped under a different fingerprint schema.

    Its own explicit error, deliberately NOT a :class:`MismatchError`
    (design §2.3): the remedy is "re-stamp under v{supported}" — an
    auditable operation — not a content investigation.
    """

    def __init__(self, stamped_version: Any, supported_version: int) -> None:
        self.stamped_version = stamped_version
        self.supported_version = supported_version
        super().__init__(
            f"fingerprint schema version gap: artifact stamped under "
            f"{stamped_version!r} but this module implements "
            f"v{supported_version}; re-stamp the artifact under "
            f"v{supported_version} (auditable operation) — do not treat "
            "this as a content mismatch"
        )


class MismatchError(FingerprintError):
    """The recomputed content fingerprint differs from the stamped one."""

    def __init__(
        self,
        expected: str,
        actual: str,
        field_digests: dict[str, str],
    ) -> None:
        self.expected = expected
        self.actual = actual
        self.field_digests = dict(field_digests)
        super().__init__(
            f"model content fingerprint mismatch: stamped {expected} but "
            f"payload hashes to {actual}. Diff hint — per-field digests of "
            f"this payload's PREDICTIVE content: {self.field_digests}; "
            "recompute predictive_field_digests() on the stamping side and "
            "compare to localize the divergent field(s)."
        )


def _canonical(value: Any, key_path: str) -> Any:
    """Return a pure-JSON structure with the design §2b guarantees.

    * Exact float canonicalization (``repr``-based shortest round-trip via
      the ``json`` encoder) — no lossy rounding.
    * Non-finite floats rejected (:class:`NonFiniteValueError`).
    * numpy scalars/arrays accepted via duck-typed ``tolist()`` so the
      same IEEE-754 values hash identically whether produced by numpy or
      plain Python.
    * Anything unrepresentable is a hard error — never ``default=str``
      (the old impls' lossy, silent fallback).
    """
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise NonFiniteValueError(key_path, value)
        return value
    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            if not isinstance(k, str):
                raise FingerprintError(
                    f"non-string dict key {k!r} at {key_path!r}: canonical "
                    "JSON serialization requires string keys"
                )
            out[k] = _canonical(v, f"{key_path}.{k}")
        return out
    if isinstance(value, (list, tuple)):
        return [
            _canonical(v, f"{key_path}[{i}]") for i, v in enumerate(value)
        ]
    tolist = getattr(value, "tolist", None)
    if callable(tolist):  # numpy scalars and arrays, without importing numpy
        return _canonical(tolist(), key_path)
    raise FingerprintError(
        f"unsupported type {type(value).__name__!r} at {key_path!r}: "
        "refusing lossy default=str serialization; convert to plain "
        "JSON-compatible types before stamping"
    )


def _classified_predictive_content(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise FingerprintError(
            f"payload must be a dict, got {type(payload).__name__!r}"
        )
    unclassified = [
        k for k in payload
        if k not in PREDICTIVE_KEYS and k not in OPERATIONAL_KEYS
    ]
    if unclassified:
        raise UnclassifiedKeyError(unclassified)
    content = {k: payload[k] for k in payload if k in PREDICTIVE_KEYS}
    if not content:
        raise NoPredictiveContentError(
            "payload has no recognizable scorer prediction content: no "
            "PREDICTIVE-classified field present"
        )
    return content


def _digest(canonical_value: Any) -> str:
    blob = json.dumps(
        canonical_value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return "sha256:" + hashlib.sha256(blob.encode("utf-8")).hexdigest()


def model_content_sha256(payload: dict[str, Any]) -> str:
    """Stable scorer identity over the PREDICTIVE-classified content.

    Hashes the canonical JSON (sorted keys, exact float representation) of
    the PREDICTIVE subset of ``payload``. Every payload key must be
    classified; see :class:`UnclassifiedKeyError`.

    Both the fit-time (calibrator training) side and the runtime
    (scorer-binding check) side MUST call this same function — importing
    it from renquant-common is what makes the two sides structurally
    guaranteed to agree.
    """
    content = _classified_predictive_content(payload)
    canonical = {k: _canonical(v, k) for k, v in content.items()}
    return _digest(canonical)


def predictive_field_digests(payload: dict[str, Any]) -> dict[str, str]:
    """Per-field digests of the PREDICTIVE content, for mismatch triage.

    Comparing the stamping side's and the verifying side's per-field
    digests localizes exactly which field diverged (stage-2 triage
    tooling, design §2c).
    """
    content = _classified_predictive_content(payload)
    return {k: _digest(_canonical(v, k)) for k, v in content.items()}


def stamp(payload: dict[str, Any]) -> dict[str, Any]:
    """Return the stamp fields to persist alongside the artifact payload.

    ``fingerprint_schema_version`` travels WITH the hash so a verifier can
    distinguish "different content" from "different contract"
    (:class:`VersionGapError` vs :class:`MismatchError`).
    """
    return {
        "model_content_fingerprint": model_content_sha256(payload),
        "fingerprint_schema_version": FINGERPRINT_SCHEMA_VERSION,
    }


def verify(
    payload: dict[str, Any],
    expected: str,
    expected_version: int,
) -> None:
    """Check ``payload`` against a stamped fingerprint; raise on failure.

    :param payload: the artifact payload under verification.
    :param expected: the stamped ``model_content_fingerprint``.
    :param expected_version: the stamped ``fingerprint_schema_version``.
        MANDATORY — a caller must extract this from the artifact's own
        stamp (see :func:`stamp`) and pass it explicitly. There is no
        optional/omitted form: an omitted or malformed version is exactly
        the "one forgotten argument bypasses the whole migration contract"
        hole this function must never reopen. A gap against
        :data:`FINGERPRINT_SCHEMA_VERSION` raises :class:`VersionGapError`
        BEFORE any content comparison — a version gap is not evidence
        about content. Verifying an artifact with NO stamped version at
        all (a pre-schema-v1 artifact) is out of scope for this function;
        use an explicitly named legacy path if that migration case is
        needed — never silently accept a versionless stamp here.
    :raises VersionGapError: schema-version gap, or a malformed
        (non-integer, or bool masquerading as int) version value
        (re-stamp, auditable).
    :raises MismatchError: content fingerprint differs (with per-field
        diff hint).
    :raises UnclassifiedKeyError: the payload contains a key the current
        tables do not classify — fails closed, never ignored (design §2b
        verify-time rule).
    """
    if not isinstance(expected_version, int) or isinstance(expected_version, bool):
        # Reject non-integer versions (including the classic Python trap
        # where 1.0 == 1 and True == 1 would otherwise silently coerce
        # through an equality check) rather than let a malformed stamp
        # value pass or fail unpredictably.
        raise VersionGapError(expected_version, FINGERPRINT_SCHEMA_VERSION)
    if expected_version != FINGERPRINT_SCHEMA_VERSION:
        raise VersionGapError(expected_version, FINGERPRINT_SCHEMA_VERSION)
    actual = model_content_sha256(payload)
    if actual != expected:
        raise MismatchError(
            expected, actual, predictive_field_digests(payload)
        )


def artifact_sha256(path: str | Path) -> str:
    """Full-file artifact hash for tamper/audit checks.

    Do not use this as the scorer/calibrator pairing identity: acceptance
    tools append mutable metadata such as ``wf_gate_metadata`` after
    training, which changes the file bytes without changing the model.
    """
    return "sha256:" + hashlib.sha256(Path(path).read_bytes()).hexdigest()
