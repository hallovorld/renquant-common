from __future__ import annotations

import importlib
import sys


def test_common_import_does_not_pull_heavy_or_runtime_modules() -> None:
    importlib.import_module("renquant_common")

    forbidden_prefixes = (
        "alpaca",
        "backtesting",
        "ib_insync",
        "live",
        "torch",
        "xgboost",
    )
    loaded = set(sys.modules)
    offenders = sorted(
        name for name in loaded
        if name == forbidden_prefixes or name.startswith(forbidden_prefixes)
    )
    assert offenders == []
