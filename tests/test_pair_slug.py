"""Canonical crypto pair/slug symbol helpers (crypto RFC D-C1, gap M1).

Test cases mirror renquant-base-data's local stand-in exactly (the one
this module exists to let that repo repoint onto) — same semantics, same
malformed-input battery, so a repoint is a behavior-identity no-op.
"""
from __future__ import annotations

import pytest

from renquant_common.pair_slug import as_pair, as_slug, pair_slug, slug_pair


@pytest.mark.parametrize(
    ("pair", "slug"),
    [
        ("BTC/USD", "BTC-USD"),
        ("ETH/USD", "ETH-USD"),
        ("USDT/USD", "USDT-USD"),
        ("DOGE/USD", "DOGE-USD"),
    ],
)
def test_pair_slug_round_trip(pair: str, slug: str) -> None:
    assert pair_slug(pair) == slug
    assert slug_pair(slug) == pair
    assert slug_pair(pair_slug(pair)) == pair
    assert pair_slug(slug_pair(slug)) == slug


def test_pair_slug_normalizes_case_and_whitespace() -> None:
    assert pair_slug(" btc/usd ") == "BTC-USD"
    assert slug_pair(" eth-usd ") == "ETH/USD"


@pytest.mark.parametrize("bad", ["BTCUSD", "BTC/USD/X", "BTC-USD", "", "/USD", "BTC/"])
def test_pair_slug_rejects_malformed(bad: str) -> None:
    with pytest.raises(ValueError):
        pair_slug(bad)


@pytest.mark.parametrize("bad", ["BTC/USD", "BTCUSD", "BTC-USD-X", "", "-USD", "BTC-"])
def test_slug_pair_rejects_malformed(bad: str) -> None:
    with pytest.raises(ValueError):
        slug_pair(bad)


class TestAsPairAsSlug:
    def test_accepts_either_canonical_form(self) -> None:
        assert as_pair("BTC/USD") == "BTC/USD"
        assert as_pair("BTC-USD") == "BTC/USD"
        assert as_slug("BTC/USD") == "BTC-USD"
        assert as_slug("BTC-USD") == "BTC-USD"

    def test_normalizes_case_and_whitespace(self) -> None:
        assert as_pair(" btc-usd ") == "BTC/USD"
        assert as_slug(" btc/usd ") == "BTC-USD"

    @pytest.mark.parametrize("bad", ["BTCUSD", "BTC/USD/X", "", "BTC/USD-X"])
    def test_as_pair_rejects_malformed(self, bad: str) -> None:
        with pytest.raises(ValueError):
            as_pair(bad)

    @pytest.mark.parametrize("bad", ["BTCUSD", "BTC-USD-X", "", "BTC-USD/X"])
    def test_as_slug_rejects_malformed(self, bad: str) -> None:
        with pytest.raises(ValueError):
            as_slug(bad)
