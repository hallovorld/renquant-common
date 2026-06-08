"""Typed cross-repo contracts.

Per RFC §"Cross-Repo Contracts", every data flow across subrepo boundaries
must be one of the types defined here. Subpackages:

* ``regime``  — :class:`RegimeLabel` enum and config validator.
* ``schemas`` — Pydantic v2 schemas for artifacts, decisions, and
  acceptance reports.
* ``scorer``  — :class:`Scorer` Protocol and ``load_scorer`` registry.
"""
from __future__ import annotations

from .regime import RegimeLabel, validate_regime_params
from .scorer import (
    SCORER_ENTRY_POINT_GROUP,
    Scorer,
    ScorerKindNotRegistered,
    load_scorer,
)
from .schemas import (
    AcceptanceReport,
    ArtifactManifest,
    DecisionTraceRow,
    LiveRunBundle,
    OOSEvidence,
    PooledMetric,
    RegimeMetric,
    Tier,
    TriadReport,
    VerifiedArtifact,
    validate_live_run_bundle,
    validate_manifest_for_leakage_triad,
)

__all__ = [
    "AcceptanceReport",
    "ArtifactManifest",
    "DecisionTraceRow",
    "LiveRunBundle",
    "OOSEvidence",
    "PooledMetric",
    "RegimeLabel",
    "RegimeMetric",
    "SCORER_ENTRY_POINT_GROUP",
    "Scorer",
    "ScorerKindNotRegistered",
    "Tier",
    "TriadReport",
    "VerifiedArtifact",
    "load_scorer",
    "validate_live_run_bundle",
    "validate_manifest_for_leakage_triad",
    "validate_regime_params",
]
