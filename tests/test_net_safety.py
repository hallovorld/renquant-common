"""Tests for `renquant_common.net_safety` — the lift from
`RenQuant/backtesting/renquant_104/kernel/net_safety.py`.

Parity invariant: the lifted module's behaviour on the same call MUST
match the umbrella's original. Any divergence flags either an
unintentional change in the lift (this PR must fix) or umbrella
drift ahead of this copy (consumers should upgrade).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

from renquant_common.net_safety import FetchBudget, call_with_timeout


# ---- Structural tests (no umbrella dependency) -------------------------


def test_call_with_timeout_returns_value_when_callable_succeeds() -> None:
    result = call_with_timeout(lambda: 42, timeout_sec=1.0, label="ok")
    assert result == 42


def test_call_with_timeout_returns_none_on_exception() -> None:
    def boom() -> int:
        raise RuntimeError("nope")
    result = call_with_timeout(boom, timeout_sec=1.0, label="raises")
    # Exception is caught + logged, function returns None.
    assert result is None


def test_call_with_timeout_returns_none_on_actual_timeout() -> None:
    def slow() -> int:
        time.sleep(2.0)
        return 99
    start = time.monotonic()
    result = call_with_timeout(slow, timeout_sec=0.3, label="slow")
    elapsed = time.monotonic() - start
    assert result is None
    # MUST return well before the slow function's full sleep — otherwise
    # the timeout is theatrical.
    assert elapsed < 1.5, f"timeout took {elapsed:.2f}s; expected < 1.5s"


def test_fetch_budget_starts_unexhausted() -> None:
    budget = FetchBudget(total_sec=10.0, label="test")
    assert budget.exhausted() is False
    assert budget.remaining() > 0


def test_fetch_budget_exhausts_after_charge() -> None:
    budget = FetchBudget(total_sec=0.5, label="test")
    assert budget.exhausted() is False
    budget.charge(0.6)  # over budget
    # Budget exhausts via explicit .charge() calls, not wall-clock time.
    # call_with_timeout charges automatically; manual charge here pins
    # the contract.
    assert budget.exhausted() is True


def test_call_with_timeout_short_circuits_when_budget_exhausted() -> None:
    budget = FetchBudget(total_sec=0.5, label="test-budget")
    budget.charge(1.0)  # exhaust explicitly
    assert budget.exhausted() is True

    counter = {"calls": 0}
    def counted() -> int:
        counter["calls"] += 1
        return 7
    result = call_with_timeout(counted, timeout_sec=1.0, label="counted", budget=budget)
    assert result is None
    # The function should NEVER be invoked when budget is exhausted —
    # that's the whole point of the budget short-circuit.
    assert counter["calls"] == 0, (
        "budget-exhausted call MUST short-circuit; counted() was called "
        f"{counter['calls']} times")


def test_call_with_timeout_charges_budget_on_success() -> None:
    """The budget is charged by elapsed time even when the call
    succeeds — that's how subsequent calls see the budget shrink."""
    budget = FetchBudget(total_sec=10.0, label="charge-test")
    assert budget.consumed == 0.0
    call_with_timeout(lambda: time.sleep(0.05) or 1, timeout_sec=1.0,
                     label="quick", budget=budget)
    assert budget.consumed > 0.0, "successful call did not charge budget"
    assert budget.consumed < 1.0, "charged too aggressively"


# ---- Lift parity test (skip when umbrella isn't on disk) ---------------


@pytest.fixture
def umbrella_module():
    """Import the umbrella's kernel.net_safety for parity checks.

    Skips (not fails) if the umbrella tree isn't on disk — CI checks
    out renquant-common alone where the umbrella isn't reachable.

    Cleans up sys.path AND sys.modules so the cached `kernel` import
    doesn't leak to other tests (test_regime_labels.py's parity
    fixture also imports `kernel.regime_labels` from a DIFFERENT
    umbrella subdir — leaking sys.modules['kernel'] would break it).
    """
    umbrella_kernel = (Path(__file__).resolve().parents[2]
                       / "RenQuant" / "backtesting" / "renquant_104" / "kernel")
    if not umbrella_kernel.exists():
        pytest.skip("umbrella kernel not on disk; skipping parity check")
    sys_path_entry = str(umbrella_kernel.parent)
    sys.path.insert(0, sys_path_entry)
    # Drop any cached `kernel` package from a prior test run.
    cached_kernel_keys = [k for k in sys.modules if k == "kernel" or k.startswith("kernel.")]
    cached = {k: sys.modules.pop(k) for k in cached_kernel_keys}
    try:
        from kernel import net_safety as umbrella_ns  # noqa: PLC0415
        yield umbrella_ns
    finally:
        # Remove our sys.path entry.
        if sys_path_entry in sys.path:
            sys.path.remove(sys_path_entry)
        # Drop the `kernel` we loaded so the next test can re-import
        # from its own umbrella subdir without colliding on cache.
        for k in [m for m in sys.modules if m == "kernel" or m.startswith("kernel.")]:
            sys.modules.pop(k, None)
        # Restore any pre-existing cache (if a sibling test had its
        # own kernel loaded first, leave that alone — best-effort).
        for k, mod in cached.items():
            sys.modules.setdefault(k, mod)


def test_call_with_timeout_matches_umbrella_on_success(umbrella_module) -> None:
    lifted = call_with_timeout(lambda: "ok", timeout_sec=1.0, label="lift")
    umbrella = umbrella_module.call_with_timeout(lambda: "ok", timeout_sec=1.0, label="umb")
    assert lifted == umbrella == "ok"


def test_call_with_timeout_matches_umbrella_on_exception(umbrella_module) -> None:
    def boom() -> int:
        raise ValueError("err")
    assert call_with_timeout(boom, timeout_sec=1.0, label="lift") is None
    assert umbrella_module.call_with_timeout(boom, timeout_sec=1.0, label="umb") is None


def test_fetch_budget_matches_umbrella(umbrella_module) -> None:
    lifted = FetchBudget(total_sec=5.0, label="lift")
    umbrella = umbrella_module.FetchBudget(total_sec=5.0, label="umb")
    # Both should report unexhausted initially.
    assert lifted.exhausted() is False
    assert umbrella.exhausted() is False
    # Both should report positive remaining time.
    assert lifted.remaining() > 0
    assert umbrella.remaining() > 0
