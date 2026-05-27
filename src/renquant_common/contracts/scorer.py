"""Scorer Protocol and loader registry.

Every model backend (XGBoost, PatchTST, future families) implements the
:class:`Scorer` Protocol and registers a loader via an entry point in its
``pyproject.toml``::

    [project.entry-points."renquant_common.scorers"]
    panel_ltr_xgboost = "renquant_model.gbdt.scorer:load"
    patchtst_panel    = "renquant_model.patchtst.scorer:load"

Consumers (renquant-pipeline, renquant-backtesting) call
:func:`load_scorer` with an :class:`ArtifactManifest`; the registry
dispatches on ``manifest.kind`` to the right loader. Consumers must NOT
``import`` any concrete backend module — the entry-point indirection is
what keeps the import boundary clean (RFC §"Bootstrap Drift Audit" item 1).
"""
from __future__ import annotations

from importlib import metadata
from typing import Protocol, runtime_checkable

from .schemas import ArtifactManifest

SCORER_ENTRY_POINT_GROUP = "renquant_common.scorers"


@runtime_checkable
class Scorer(Protocol):
    """Stateless inference adapter for a trained model artifact.

    Backends implement these methods; consumers use only this Protocol.
    The Protocol is ``runtime_checkable`` so ``isinstance(x, Scorer)``
    works at load time.
    """

    #: Feature column names the model expects, in order.
    feature_cols: list[str]

    def feature_fingerprint(self) -> str:
        """Return a stable hash of ``(feature_cols, transform_version)``.

        Used by consumers to verify they are passing the same feature
        shape the artifact was trained on. A mismatch is fail-closed.
        """
        ...

    def predict_rows(
        self, rows: dict[str, dict[str, float]]
    ) -> dict[str, float]:
        """Score a batch of tickers.

        Args:
            rows: ``{ticker: {feature_name: value, ...}, ...}``.

        Returns:
            ``{ticker: raw_score, ...}``. Order is not significant.
        """
        ...

    def predict_variance(
        self, rows: dict[str, dict[str, float]]
    ) -> dict[str, float] | None:
        """Optional variance prediction (e.g. for NGBoost-style heads).

        Backends that do not produce variance return ``None``. Consumers
        must treat ``None`` as "no variance available" and not as an error.
        """
        ...


class ScorerKindNotRegistered(LookupError):
    """Raised when no entry point matches ``manifest.kind``."""


def load_scorer(manifest: ArtifactManifest) -> Scorer:
    """Resolve and load a :class:`Scorer` for the given manifest.

    Discovery walks the ``renquant_common.scorers`` entry-point group; the
    entry-point whose ``name`` equals ``manifest.kind`` is selected. The
    target must be a callable that accepts the manifest and returns an
    object satisfying the :class:`Scorer` Protocol.

    Raises:
        ScorerKindNotRegistered: no entry point matches ``manifest.kind``.
        TypeError: the returned object does not satisfy :class:`Scorer`.
    """
    eps = metadata.entry_points(group=SCORER_ENTRY_POINT_GROUP)
    matches = [ep for ep in eps if ep.name == manifest.kind]
    if not matches:
        available = sorted({ep.name for ep in eps})
        raise ScorerKindNotRegistered(
            f"no Scorer registered for kind={manifest.kind!r}; "
            f"available kinds: {available}"
        )
    loader = matches[0].load()
    scorer = loader(manifest)
    if not isinstance(scorer, Scorer):
        raise TypeError(
            f"loader for kind={manifest.kind!r} returned "
            f"{type(scorer).__name__} which does not satisfy the Scorer "
            f"Protocol (missing feature_cols / predict_rows / "
            f"feature_fingerprint)"
        )
    return scorer
