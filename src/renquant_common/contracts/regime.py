"""Regime taxonomy — closed set of regime labels used across all subrepos.

Per RFC PRIME DIRECTIVE, RenQuant is a regime-conditional strategy. The
regime labels referenced by config (``regime_params``), the regime detector,
the runtime pipeline, artifacts, and acceptance reports MUST agree on the
same set. ``RegimeLabel`` is the single source of truth.

Adding or renaming a member is a breaking change to ``renquant-common``
(semver major bump). See RFC §"Schema Versioning §6" for the migration
protocol.
"""
from __future__ import annotations

from enum import Enum
from typing import Any


class RegimeLabel(str, Enum):
    """The closed set of regime identifiers."""

    BULL_CALM = "BULL_CALM"
    BULL_VOLATILE = "BULL_VOLATILE"
    BULL_STRONG = "BULL_STRONG"
    BEAR = "BEAR"
    CHOPPY = "CHOPPY"

    @classmethod
    def values(cls) -> tuple[str, ...]:
        return tuple(member.value for member in cls)

    @classmethod
    def from_str(cls, value: str) -> "RegimeLabel":
        """Parse a string into a :class:`RegimeLabel`, raising on unknown."""
        try:
            return cls(value)
        except ValueError as exc:
            valid = ", ".join(cls.values())
            raise ValueError(
                f"unknown regime label {value!r}; valid: {valid}"
            ) from exc


def validate_regime_params(cfg: dict[str, Any], *, strict: bool = True) -> None:
    """Validate a strategy config's ``regime_params`` block.

    Args:
        cfg: a loaded strategy config dict.
        strict: when True (default), reject any regime keys not in
            :class:`RegimeLabel`; when False, allow extras (intended only
            for backward-compat during migration). Missing required regimes
            always fails.

    Raises:
        ValueError: if ``regime_params`` is missing, is not a dict, lacks a
            required ``RegimeLabel`` member, or (when strict) carries an
            unknown extra key.
    """
    regime_params = cfg.get("regime_params")
    if not isinstance(regime_params, dict):
        raise ValueError("strategy config missing required 'regime_params' dict")
    expected = set(RegimeLabel.values())
    present = set(regime_params.keys())
    missing = expected - present
    if missing:
        raise ValueError(
            f"regime_params missing required regimes: {sorted(missing)}"
        )
    if strict:
        extras = present - expected
        if extras:
            raise ValueError(
                f"regime_params has unknown regime keys: {sorted(extras)}; "
                f"valid set: {sorted(expected)}"
            )
