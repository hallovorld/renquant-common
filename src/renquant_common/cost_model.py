"""Generic net-of-cost accounting primitives (crypto RFC D-C8a, gap M1).

ONE authoritative cost model, per the merged crypto trading RFC
(renquant-orchestrator ``doc/design/2026-07-10-crypto-trading-rfc.md`` §4.4,
CORRECTED per Codex review): the same cost-accounting math must be used
IDENTICALLY by walk-forward-gate replay evaluation AND live runtime
accounting (paper P&L, reservation sizing, QP/rotation cost-kappa) — one
number, every consumer. A model that passes a gross gate and fails net of
costs is a FAIL; two hand-copied cost formulas that drift apart are exactly
the class of bug the fingerprint unification (M6) existed to kill.

Scope split (RFC D-C8a/D-C8b, resolved question §9.3): this module is the
GENERIC, asset-agnostic PRIMITIVE — fee/spread/slippage/increment-rounding
bps accounting over realized turnover, plus the rejected/unfilled-order
convention. Asset-specific numbers (e.g. the crypto taker-fee default) and
asset-specific promotion decisions (the BTC-baseline bar) live in the
consuming repos (renquant-model), NEVER here.

Conventions (frozen; consumers and tests pin these):

* All rates are in **basis points per side** unless named otherwise. A
  "side" is one fill: a buy pays the per-side cost once, a sell pays it
  once; a full round trip pays it twice.
* ``spread_bps`` is the FULL quoted bid-ask spread; a marketable order
  crossing the spread pays HALF of it per side (mid-to-touch).
* Portfolio weights are fractions of NAV. ``traded_fraction`` between two
  weight vectors is ``sum(|w_new - w_old|)`` — buys plus sells, each unit
  of which pays the per-side cost once. The conventional "one-sided
  turnover" is half of that; both are exposed, explicitly named, so no
  consumer can silently halve or double costs.
* Rejected/unfilled/resting-never-filled orders (RFC §4.4): a fill that
  never happened costs nothing AND earns nothing. Consumers must account
  costs over **realized** (filled) turnover only — and must equally not
  credit the phantom fill's return on the earning side (that obligation
  sits with the replay evaluator; :func:`realized_traded_fraction` pins
  the cost-side convention).
* Calibration provenance is the CALLER's problem: this module does not
  know whether a spread number came from Stage-0 ex-ante quote percentiles
  or a live-canary re-estimate. It only guarantees that whatever numbers
  are supplied are applied identically everywhere.

No I/O, no network, no pandas dependency — pure float/mapping arithmetic,
deterministic, so replay and runtime cannot diverge on environment.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

__all__ = [
    "CostModelSpec",
    "TurnoverBreakdown",
    "apply_costs_to_period_returns",
    "per_side_cost_bps",
    "realized_traded_fraction",
    "rebalance_cost_fraction",
    "round_trip_cost_bps",
    "turnover_breakdown",
]

_BPS = 1e4


def _require_finite_nonneg(name: str, value: float) -> float:
    v = float(value)
    if not math.isfinite(v) or v < 0.0:
        raise ValueError(f"CostModelSpec.{name} must be finite and >= 0, got {value!r}")
    return v


@dataclass(frozen=True)
class CostModelSpec:
    """Per-side cost components, all in basis points.

    :param fee_bps: venue fee per side (taker or maker — the caller picks
        which schedule applies to the flow being modeled).
    :param spread_bps: FULL quoted bid-ask spread; half is paid per side.
    :param slippage_bps: per-side impact/slippage allowance BEYOND the
        half-spread (e.g. a Stage-0 ex-ante bound from quote-depth-vs-size).
    :param increment_rounding_bps: per-side loss bound from quantity
        truncation to a venue's minimum trade increment.

    Every component defaults to 0 so a fee-only model is spelled
    ``CostModelSpec(fee_bps=...)`` with nothing implicit.
    """

    fee_bps: float = 0.0
    spread_bps: float = 0.0
    slippage_bps: float = 0.0
    increment_rounding_bps: float = 0.0

    def __post_init__(self) -> None:
        for name in ("fee_bps", "spread_bps", "slippage_bps", "increment_rounding_bps"):
            object.__setattr__(self, name, _require_finite_nonneg(name, getattr(self, name)))


def per_side_cost_bps(spec: CostModelSpec) -> float:
    """Total cost of ONE side (one fill), in bps of the traded notional.

    ``fee + spread/2 + slippage + increment_rounding``.
    """
    return spec.fee_bps + spec.spread_bps / 2.0 + spec.slippage_bps + spec.increment_rounding_bps


def round_trip_cost_bps(spec: CostModelSpec) -> float:
    """Cost of a full round trip (buy then sell), in bps: 2 x per-side."""
    return 2.0 * per_side_cost_bps(spec)


@dataclass(frozen=True)
class TurnoverBreakdown:
    """Turnover between two weight vectors, every convention named.

    * ``buy_fraction``: sum of positive weight increases (NAV fraction bought).
    * ``sell_fraction``: sum of weight decreases, as a positive number.
    * ``traded_fraction``: ``buy_fraction + sell_fraction`` — the quantity
      per-side costs apply to.
    * ``one_sided``: ``traded_fraction / 2`` — the conventional turnover
      statistic; NEVER multiply costs by this (that silently halves them).
    """

    buy_fraction: float
    sell_fraction: float

    @property
    def traded_fraction(self) -> float:
        return self.buy_fraction + self.sell_fraction

    @property
    def one_sided(self) -> float:
        return self.traded_fraction / 2.0


def _validated_weights(name: str, weights: Mapping[str, float]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, value in weights.items():
        v = float(value)
        if not math.isfinite(v):
            raise ValueError(f"{name}[{key!r}] must be finite, got {value!r}")
        out[str(key)] = v
    return out


def turnover_breakdown(
    prev_weights: Mapping[str, float],
    next_weights: Mapping[str, float],
) -> TurnoverBreakdown:
    """Buy/sell/traded fractions to move ``prev_weights`` -> ``next_weights``.

    Symbols absent from a mapping have weight 0 (a new position is all buy;
    a dropped position is all sell). Weights may be any finite floats; the
    caller owns whether they sum to 1.
    """
    prev = _validated_weights("prev_weights", prev_weights)
    nxt = _validated_weights("next_weights", next_weights)
    buys = 0.0
    sells = 0.0
    for key in set(prev) | set(nxt):
        delta = nxt.get(key, 0.0) - prev.get(key, 0.0)
        if delta > 0:
            buys += delta
        else:
            sells += -delta
    return TurnoverBreakdown(buy_fraction=buys, sell_fraction=sells)


def rebalance_cost_fraction(
    prev_weights: Mapping[str, float],
    next_weights: Mapping[str, float],
    spec: CostModelSpec,
) -> float:
    """Cost of one rebalance as a fraction of NAV.

    ``per_side_cost_bps(spec)/1e4 * traded_fraction`` — every traded unit
    (buy or sell) pays the per-side cost exactly once.
    """
    traded = turnover_breakdown(prev_weights, next_weights).traded_fraction
    return per_side_cost_bps(spec) / _BPS * traded


def realized_traded_fraction(intended_traded_fraction: float, fill_ratio: float) -> float:
    """REALIZED traded fraction after partial/zero fills (RFC §4.4).

    An order that rests and never fills is a zero-fee, zero-fill outcome: it
    contributes nothing to costs (this function) and its would-be return
    must equally not be credited by the replay evaluator (the caller's
    obligation, documented at module level). ``fill_ratio`` must lie in
    [0, 1]; anything else is a hard error, never clipped silently.
    """
    intended = float(intended_traded_fraction)
    ratio = float(fill_ratio)
    if not math.isfinite(intended) or intended < 0.0:
        raise ValueError(f"intended_traded_fraction must be finite and >= 0, got {intended_traded_fraction!r}")
    if not math.isfinite(ratio) or not (0.0 <= ratio <= 1.0):
        raise ValueError(f"fill_ratio must lie in [0, 1], got {fill_ratio!r}")
    return intended * ratio


def apply_costs_to_period_returns(
    gross_returns: Sequence[float] | Iterable[float],
    traded_fractions: Sequence[float] | Iterable[float],
    spec: CostModelSpec,
) -> list[float]:
    """Net-of-cost period returns: ``net_t = gross_t - rate * traded_t``.

    ``rate`` is the per-side cost as a fraction (``per_side_cost_bps/1e4``);
    ``traded_fractions[t]`` is the REALIZED traded fraction charged to
    period ``t`` (0 for periods with no rebalance). Lengths must match —
    a silent zip-truncation would drop cost periods, so mismatch is a hard
    error.
    """
    gross = [float(g) for g in gross_returns]
    traded = [float(t) for t in traded_fractions]
    if len(gross) != len(traded):
        raise ValueError(
            f"gross_returns (n={len(gross)}) and traded_fractions (n={len(traded)}) "
            "must have the same length"
        )
    rate = per_side_cost_bps(spec) / _BPS
    net: list[float] = []
    for g, t in zip(gross, traded):
        if not math.isfinite(g):
            raise ValueError(f"gross_returns contains non-finite value {g!r}")
        if not math.isfinite(t) or t < 0.0:
            raise ValueError(f"traded_fractions contains invalid value {t!r} (must be finite, >= 0)")
        net.append(g - rate * t)
    return net
