from __future__ import annotations

import pytest

from renquant_common import RegimeLabel, validate_regime_params


def test_regime_label_is_closed_set() -> None:
    expected = {"BULL_CALM", "BULL_VOLATILE", "BULL_STRONG", "BEAR", "CHOPPY"}
    assert set(RegimeLabel.values()) == expected


def test_regime_label_string_interop() -> None:
    assert RegimeLabel.BEAR == "BEAR"
    assert RegimeLabel("BULL_CALM") is RegimeLabel.BULL_CALM


def test_from_str_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="unknown regime label"):
        RegimeLabel.from_str("RANGE_BOUND")


def test_validate_requires_every_regime() -> None:
    cfg = {"regime_params": {"BULL_CALM": {}, "BULL_VOLATILE": {}}}
    with pytest.raises(ValueError, match="missing required regimes"):
        validate_regime_params(cfg)


def test_validate_strict_rejects_extras() -> None:
    cfg = {
        "regime_params": {
            "BULL_CALM": {},
            "BULL_VOLATILE": {},
            "BULL_STRONG": {},
            "BEAR": {},
            "CHOPPY": {},
            "RANGE_BOUND": {},
        }
    }
    with pytest.raises(ValueError, match="unknown regime keys"):
        validate_regime_params(cfg)


def test_validate_non_strict_allows_extras_but_not_missing() -> None:
    full = {label: {} for label in RegimeLabel.values()}
    cfg = {"regime_params": {**full, "RANGE_BOUND": {}}}
    validate_regime_params(cfg, strict=False)

    partial = {label: {} for label in list(RegimeLabel.values())[:-1]}
    with pytest.raises(ValueError, match="missing required regimes"):
        validate_regime_params({"regime_params": partial}, strict=False)


def test_validate_rejects_missing_block() -> None:
    with pytest.raises(ValueError, match="missing required 'regime_params'"):
        validate_regime_params({})
