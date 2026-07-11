"""Canonical crypto pair/slug symbol helpers (crypto RFC D-C1, gap M1).

The other half of D-C1 alongside :mod:`renquant_common.market_calendar`'s
``ALWAYS_OPEN`` mode (common#27): ONE shared symbol-normalization primitive
for the two canonical crypto symbol forms every consuming repo needs —
"pair" form (``"BTC/USD"``, the broker/API convention) and "slug" form
(``"BTC-USD"``, the filesystem/path-safe convention: a bare ``"/"`` would
create a nested directory). renquant-base-data's D-C2 ingestion (#41)
carried a local, explicitly-labeled stand-in for exactly this module
("local stand-in for renquant-common D-C1") — this is the canonical
primitive that stand-in repoints onto.

Strict by design: every function raises on a malformed symbol rather than
silently producing a colliding cache key or an ambiguous path — a base/quote
side must be non-empty, and the input must unambiguously be ONE of the two
canonical forms (not both delimiters, not neither).
"""
from __future__ import annotations

__all__ = [
    "as_pair",
    "as_slug",
    "pair_slug",
    "slug_pair",
]


def pair_slug(pair: str) -> str:
    """Canonical pair form -> canonical slug form: ``"BTC/USD"`` -> ``"BTC-USD"``.

    Strict by design: the input must be pair form (exactly one ``/``, both
    sides non-empty, no ``-``). Malformed symbols raise instead of silently
    producing a colliding cache key (e.g. ``"BTC/USD"`` used as a path
    component creates a nested ``BTC/USD/`` directory).
    """
    p = str(pair).strip().upper()
    if p.count("/") != 1 or "-" in p:
        raise ValueError(f"not a canonical crypto pair (expected 'BASE/QUOTE'): {pair!r}")
    base, _, quote = p.partition("/")
    if not base or not quote:
        raise ValueError(f"not a canonical crypto pair (expected 'BASE/QUOTE'): {pair!r}")
    return f"{base}-{quote}"


def slug_pair(slug: str) -> str:
    """Canonical slug form -> canonical pair form: ``"BTC-USD"`` -> ``"BTC/USD"``.

    Exact inverse of :func:`pair_slug`; round-trip is pinned by tests.
    """
    s = str(slug).strip().upper()
    if s.count("-") != 1 or "/" in s:
        raise ValueError(f"not a canonical crypto slug (expected 'BASE-QUOTE'): {slug!r}")
    base, _, quote = s.partition("-")
    if not base or not quote:
        raise ValueError(f"not a canonical crypto slug (expected 'BASE-QUOTE'): {slug!r}")
    return f"{base}/{quote}"


def as_pair(symbol: str) -> str:
    """Accept either canonical form, return validated pair form.

    Round-trips through the strict helpers so a malformed symbol (e.g.
    ``"BTC/USD/X"``) is rejected here, not deep in a caller's store layer.
    """
    s = str(symbol).strip().upper()
    return slug_pair(pair_slug(s)) if "/" in s else slug_pair(s)


def as_slug(symbol: str) -> str:
    """Accept either canonical form, return validated slug form."""
    s = str(symbol).strip().upper()
    return pair_slug(s) if "/" in s else pair_slug(slug_pair(s))
