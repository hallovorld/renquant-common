"""Net-of-cost accounting primitives (crypto RFC D-C8a) — arithmetic pins.

Every number below is hand-computed and written out in the comments; these
tests are the frozen semantics consumers (renquant-model's fee-aware WF
gate, runtime accounting) rely on being identical everywhere.
"""
from __future__ import annotations

import pytest

from renquant_common.cost_model import (
    CostModelSpec,
    apply_costs_to_period_returns,
    cost_model_content_sha256,
    cost_model_spec_from_dict,
    per_side_cost_bps,
    realized_traded_fraction,
    rebalance_cost_fraction,
    round_trip_cost_bps,
    turnover_breakdown,
)


class TestPerSideCost:
    def test_hand_computed_components(self) -> None:
        # fee 25 + spread 10/2 + slippage 5 + rounding 2 = 37 bps per side.
        spec = CostModelSpec(fee_bps=25.0, spread_bps=10.0, slippage_bps=5.0,
                             increment_rounding_bps=2.0)
        assert per_side_cost_bps(spec) == pytest.approx(37.0)
        assert round_trip_cost_bps(spec) == pytest.approx(74.0)

    def test_fee_only_model_is_explicit(self) -> None:
        spec = CostModelSpec(fee_bps=25.0)
        assert per_side_cost_bps(spec) == pytest.approx(25.0)
        assert round_trip_cost_bps(spec) == pytest.approx(50.0)

    def test_zero_spec_costs_nothing(self) -> None:
        assert per_side_cost_bps(CostModelSpec()) == 0.0

    @pytest.mark.parametrize("field", ["fee_bps", "spread_bps", "slippage_bps",
                                       "increment_rounding_bps"])
    @pytest.mark.parametrize("bad", [-1.0, float("nan"), float("inf")])
    def test_invalid_components_rejected(self, field: str, bad: float) -> None:
        with pytest.raises(ValueError, match=field):
            CostModelSpec(**{field: bad})


class TestTurnover:
    def test_full_rotation(self) -> None:
        # Sell 100% A, buy 100% B: buys=1, sells=1, traded=2, one-sided=1.
        t = turnover_breakdown({"A": 1.0}, {"B": 1.0})
        assert t.buy_fraction == pytest.approx(1.0)
        assert t.sell_fraction == pytest.approx(1.0)
        assert t.traded_fraction == pytest.approx(2.0)
        assert t.one_sided == pytest.approx(1.0)

    def test_partial_rebalance(self) -> None:
        # A 0.6->0.4 (sell 0.2), B 0.4->0.5 (buy 0.1), C 0->0.1 (buy 0.1).
        t = turnover_breakdown({"A": 0.6, "B": 0.4}, {"A": 0.4, "B": 0.5, "C": 0.1})
        assert t.buy_fraction == pytest.approx(0.2)
        assert t.sell_fraction == pytest.approx(0.2)
        assert t.traded_fraction == pytest.approx(0.4)

    def test_no_change_is_zero(self) -> None:
        t = turnover_breakdown({"A": 0.5, "B": 0.5}, {"B": 0.5, "A": 0.5})
        assert t.traded_fraction == 0.0

    def test_entry_from_cash_and_exit_to_cash(self) -> None:
        assert turnover_breakdown({}, {"A": 1.0}).traded_fraction == pytest.approx(1.0)
        assert turnover_breakdown({"A": 1.0}, {}).traded_fraction == pytest.approx(1.0)

    def test_non_finite_weight_rejected(self) -> None:
        with pytest.raises(ValueError, match="finite"):
            turnover_breakdown({"A": float("nan")}, {})
        with pytest.raises(ValueError, match="finite"):
            turnover_breakdown({}, {"A": float("inf")})


class TestRebalanceCost:
    def test_hand_computed_full_rotation(self) -> None:
        # 25 bps/side, traded 2.0 -> 2 x 25 bps = 50 bps = 0.005 of NAV.
        spec = CostModelSpec(fee_bps=25.0)
        cost = rebalance_cost_fraction({"A": 1.0}, {"B": 1.0}, spec)
        assert cost == pytest.approx(0.005)

    def test_hand_computed_entry(self) -> None:
        # Enter 50% from cash at 37 bps/side: 0.5 x 0.0037 = 0.00185.
        spec = CostModelSpec(fee_bps=25.0, spread_bps=10.0, slippage_bps=5.0,
                             increment_rounding_bps=2.0)
        assert rebalance_cost_fraction({}, {"A": 0.5}, spec) == pytest.approx(0.00185)


class TestRealizedTradedFraction:
    def test_unfilled_order_costs_nothing(self) -> None:
        assert realized_traded_fraction(0.8, 0.0) == 0.0

    def test_partial_fill(self) -> None:
        assert realized_traded_fraction(0.8, 0.25) == pytest.approx(0.2)

    @pytest.mark.parametrize("bad", [-0.1, 1.1, float("nan")])
    def test_fill_ratio_outside_unit_interval_rejected(self, bad: float) -> None:
        with pytest.raises(ValueError, match="fill_ratio"):
            realized_traded_fraction(1.0, bad)

    def test_negative_intended_rejected(self) -> None:
        with pytest.raises(ValueError, match="intended_traded_fraction"):
            realized_traded_fraction(-0.5, 0.5)


class TestApplyCosts:
    def test_hand_computed_net_series(self) -> None:
        # 25 bps/side. Period 0: gross 1%, traded 0.5 -> net 0.01 - 0.5*0.0025
        # = 0.00875. Period 1: gross -0.4%, no trade -> unchanged. Period 2:
        # gross 2%, traded 2.0 (full rotation) -> 0.02 - 0.005 = 0.015.
        spec = CostModelSpec(fee_bps=25.0)
        net = apply_costs_to_period_returns([0.01, -0.004, 0.02], [0.5, 0.0, 2.0], spec)
        assert net == pytest.approx([0.00875, -0.004, 0.015])

    def test_length_mismatch_is_hard_error(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            apply_costs_to_period_returns([0.01, 0.02], [0.5], CostModelSpec())

    def test_invalid_values_rejected(self) -> None:
        with pytest.raises(ValueError, match="gross_returns"):
            apply_costs_to_period_returns([float("nan")], [0.0], CostModelSpec())
        with pytest.raises(ValueError, match="traded_fractions"):
            apply_costs_to_period_returns([0.0], [-0.1], CostModelSpec())

    def test_gross_equals_net_when_costless(self) -> None:
        gross = [0.01, 0.02, -0.03]
        assert apply_costs_to_period_returns(gross, [1.0, 1.0, 1.0], CostModelSpec()) == gross


class TestFingerprint:
    """Codex review (D-C8a round-1): 'sharing formulas is insufficient if WF
    evaluation and runtime can silently use different fee, spread, slippage
    or rounding values' — a stable to_dict/sha256 identity so the two sides
    can PROVE they used the same numbers, not just the same code path."""

    def test_round_trip_through_dict(self) -> None:
        spec = CostModelSpec(fee_bps=25.0, spread_bps=10.0, slippage_bps=5.0,
                              increment_rounding_bps=2.0)
        restored = cost_model_spec_from_dict(spec.to_dict())
        assert restored == spec
        assert cost_model_content_sha256(restored) == cost_model_content_sha256(spec)

    def test_missing_dict_keys_default_like_the_constructor(self) -> None:
        assert cost_model_spec_from_dict({"fee_bps": 25.0}) == CostModelSpec(fee_bps=25.0)

    def test_different_params_produce_different_fingerprint(self) -> None:
        a = CostModelSpec(fee_bps=25.0)
        b = CostModelSpec(fee_bps=25.001)
        assert cost_model_content_sha256(a) != cost_model_content_sha256(b)

    def test_identical_params_produce_identical_fingerprint(self) -> None:
        a = CostModelSpec(fee_bps=25.0, spread_bps=10.0)
        b = CostModelSpec(fee_bps=25.0, spread_bps=10.0)
        assert cost_model_content_sha256(a) == cost_model_content_sha256(b)

    def test_fingerprint_is_stable_sha256_format(self) -> None:
        fp = cost_model_content_sha256(CostModelSpec(fee_bps=25.0))
        assert fp.startswith("sha256:")
        assert len(fp) == len("sha256:") + 64

    def test_to_dict_has_every_field_explicit(self) -> None:
        assert CostModelSpec(fee_bps=25.0).to_dict() == {
            "fee_bps": 25.0, "spread_bps": 0.0, "slippage_bps": 0.0,
            "increment_rounding_bps": 0.0,
        }


def test_asset_agnostic_no_crypto_symbols() -> None:
    """D-C8a boundary: the shared primitive carries NO asset-specific logic.

    Asset-specific defaults (e.g. the crypto taker fee) and promotion
    decisions live in the consuming repos (renquant-model), never here.
    The check is over the public surface + defaults, not docstring prose
    (the module docstring legitimately NAMES what is out of scope).
    """
    import renquant_common.cost_model as cm

    for name in cm.__all__:
        lowered = name.lower()
        for token in ("btc", "crypto", "equity", "taker", "maker"):
            assert token not in lowered, (
                f"asset-specific name {name!r} leaked into the shared primitive"
            )
    # Every cost component defaults to zero — no baked-in venue schedule.
    zero = cm.CostModelSpec()
    assert cm.per_side_cost_bps(zero) == 0.0
