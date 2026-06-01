"""Shared RenQuant contracts and pipeline primitives."""

from .contracts import (
    AcceptanceReport,
    ArtifactManifest,
    DecisionTraceRow,
    OOSEvidence,
    PooledMetric,
    RegimeLabel,
    RegimeMetric,
    SCORER_ENTRY_POINT_GROUP,
    Scorer,
    ScorerKindNotRegistered,
    Tier,
    load_scorer,
    validate_regime_params,
)
from .pipeline import (
    Job,
    ParallelTimeoutError,
    Pipeline,
    PipelineResult,
    PipelineStepRecord,
    Task,
    run_parallel,
)
from .purged_cv import (
    CombinatorialPurgedCV,
    PurgedKFold,
    cross_validated_ic,
    cross_validated_ic_cpcv,
    evaluate_fold_ic,
)
from .net_safety import FetchBudget, call_with_timeout
from .regime_labels import (
    compute_spy_regime_labels,
    min_across_regimes,
    per_regime_cs_ic,
)
from .training_runs import record_training_run

__all__ = [
    "AcceptanceReport",
    "ArtifactManifest",
    "CombinatorialPurgedCV",
    "DecisionTraceRow",
    "Job",
    "OOSEvidence",
    "ParallelTimeoutError",
    "Pipeline",
    "PipelineResult",
    "PipelineStepRecord",
    "PooledMetric",
    "PurgedKFold",
    "RegimeLabel",
    "RegimeMetric",
    "SCORER_ENTRY_POINT_GROUP",
    "Scorer",
    "ScorerKindNotRegistered",
    "Task",
    "FetchBudget",
    "Tier",
    "call_with_timeout",
    "compute_spy_regime_labels",
    "cross_validated_ic",
    "cross_validated_ic_cpcv",
    "evaluate_fold_ic",
    "load_scorer",
    "min_across_regimes",
    "per_regime_cs_ic",
    "record_training_run",
    "run_parallel",
    "validate_regime_params",
]
