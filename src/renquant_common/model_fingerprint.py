"""Canonical model-content fingerprint — single source of truth.

Extracted 2026-07-01 after a recurring production incident: the calibrator
fit-time stamp (``renquant-model``'s ``fit_calibrator_alpha158_fund.py``)
and the runtime scorer-binding check (``renquant-pipeline``'s
``panel_scorer.py``, called from ``job_panel_scoring.py``'s
``_assert_calibrator_matches_scorer``) independently hand-copied
``model_content_sha256()`` — an ALLOWLIST-style implementation in
renquant-model vs. a DENYLIST-style implementation in renquant-pipeline —
hashing DIFFERENT field sets for the same logical concept. A calibrator
fit by one could never match the runtime check by another, by
construction. This fail-closed monthly whenever
``monthly_calibrator_refresh.sh`` re-fit the calibrator (hit 2026-05-27,
2026-06-22, 2026-07-01).

Both ``renquant-model`` and ``renquant-pipeline`` depend on
``renquant-common`` already, so this module is the natural shared home:
importing the SAME function from both repos makes the fit-time stamp and
the runtime check structurally guaranteed to agree, forever — not just
coincidentally aligned today. Do not re-fork a local copy of this logic;
import it from here.

Panel-LTR artifacts are JSON files that later acquire operational
metadata (WF gate results, file hashes, paths, promotion state).
Calibrators are fitted to the model's SCORE DISTRIBUTION, not to that
mutable metadata — so the fingerprint must hash only the content that
changes the scorer's predictions, and must be invariant to metadata-only
edits (stamping ``cv_method``, ``promotion_status``, docstrings, etc.).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def artifact_sha256(path: str | Path) -> str:
    """Full-file artifact hash for tamper/audit checks.

    Do not use this as the scorer/calibrator pairing identity: acceptance
    tools append mutable metadata such as ``wf_gate_metadata`` after training,
    which changes the file bytes without changing the model.
    """
    return "sha256:" + hashlib.sha256(Path(path).read_bytes()).hexdigest()


MUTABLE_ARTIFACT_KEYS = {
    "metadata",
    "wf_gate_metadata",
    "artifact_path",
    "artifact_sha256",
    "artifact_fingerprint",
    "model_content_fingerprint",
    "config_fingerprint",
    "config_fingerprint_fields",
    "trained_date",
    "training_notes",
    "label",
    "label_col",
    "lookahead_days",
    "panel_shape",
    "n_train_rows",
    "training_train_ic",
    "val_mean_ic",
    "val_median_ic",
    "test_mean_ic",
    "test_median_ic",
    "oos_mean_ic",
    # P-PANEL-CONTRACT acceptance fields (2026-05-30 Bug D fix).
    # These are pure post-training metadata: CV bookkeeping, OOS evidence,
    # promotion gates, sentiment-contract markers, audit IDs. Stamping any
    # of these changes the JSON bytes but does NOT change the model's
    # predictions — must be excluded from model_content_fingerprint so the
    # calibrator binding survives metadata edits (previously caused 3
    # calibrator rebinds in one day).
    "cv_method",
    "cv_embargo_days",
    "cv_folds",
    "cv_n_splits",
    "oos_std_ic",
    "oos_per_fold_ic",
    "eval_ic",
    "train_run_id",
    "sentiment_runtime_gate_contract",
    "sentiment_runtime_gate_trained",
    "promotion_status",
    "promotion_gating_reason",
    "version",  # artifact-format version, not a model parameter
    "side_label",
}

PREDICTIVE_CONTENT_HINTS = {
    "booster_raw_json",
    "feature_cols",
    "feature_columns",
    "feature_means",
    "feature_stds",
    "feature_norm_kind",
    "feature_norm_kinds",
    "feature_raw_clip_low",
    "feature_raw_clip_high",
    "coef",
    "intercept",
    "clip_sigma",
    "state_dict",
    "config_dict",
    "model_bytes",
    "model_bytes_b64",
}


def model_content_sha256(payload: dict[str, Any]) -> str:
    """Stable scorer identity over immutable model content.

    Panel artifacts are JSON files that later acquire operational metadata
    (WF gate results, file hashes, paths). Calibrators are fitted to the model
    score distribution, not to that mutable metadata. Hash only the content
    that changes the scorer's predictions.

    Callers on both the fit-time (calibrator training) side and the
    runtime (scorer-binding check) side MUST call this same function —
    that is the whole point of it living in ``renquant-common`` instead of
    being hand-copied per repo.
    """
    content = {
        k: v for k, v in payload.items()
        if k not in MUTABLE_ARTIFACT_KEYS
    }
    if not any(k in content for k in PREDICTIVE_CONTENT_HINTS):
        raise ValueError("payload has no recognizable scorer prediction content")
    blob = json.dumps(content, sort_keys=True, separators=(",", ":"), default=str)
    return "sha256:" + hashlib.sha256(blob.encode("utf-8")).hexdigest()


def model_content_sha256_from_path(path: str | Path) -> str:
    """Return model-content hash for JSON artifacts, full hash otherwise."""
    p = Path(path)
    try:
        payload = json.loads(p.read_text())
    except Exception:
        return artifact_sha256(p)
    if not isinstance(payload, dict):
        return artifact_sha256(p)
    try:
        return model_content_sha256(payload)
    except ValueError:
        return artifact_sha256(p)


def stamp_artifact_metadata(
    metadata: dict | None,
    path: str | Path,
    payload: dict[str, Any] | None = None,
) -> dict:
    """Return metadata with path + fingerprint fields for runtime contracts."""
    meta = dict(metadata or {})
    nested = meta.get("metadata")
    if isinstance(nested, dict):
        for key, value in nested.items():
            meta.setdefault(key, value)
    sha = artifact_sha256(path)
    try:
        content_sha = (
            model_content_sha256(payload)
            if isinstance(payload, dict)
            else model_content_sha256_from_path(path)
        )
    except ValueError:
        content_sha = sha
    meta.setdefault("artifact_path", str(Path(path)))
    meta.setdefault("artifact_sha256", sha)
    meta.setdefault("artifact_fingerprint", sha)
    meta.setdefault("model_content_fingerprint", content_sha)
    return meta
