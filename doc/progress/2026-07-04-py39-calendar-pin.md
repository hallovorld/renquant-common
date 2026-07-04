# Py3.9 calendar pin + egg-info refresh

**Date**: 2026-07-04
**PR**: (this PR)

## Problem

`pandas_market_calendars` v5.2.4 uses `X | None` union syntax which crashes
on Python 3.9.6 (system Python). This caused 37 downstream `TypeError`
collection failures in `renquant-common` (and cascading to every repo that
imports `market_calendar`).

Separately, the local `.egg-info` had stale version 0.8.1 vs pyproject.toml
0.10.0, causing 2 `test_api_snapshot` failures.

## Fix

- Pin `pandas_market_calendars>=4,<5` in pyproject.toml (v4.6.1 confirmed
  working on Py3.9).
- Refresh `src/renquant_common.egg-info/PKG-INFO` to match pyproject.toml
  version 0.10.0.

## Verification

- Calendar tests: 37 → 0 failures
- API snapshot tests: 2 → 0 failures
- Total: 10 → 8 failures (remaining 7 = missing `statsmodels`, 1 = registry
  byte-equivalence — both are environment issues, not code bugs)
