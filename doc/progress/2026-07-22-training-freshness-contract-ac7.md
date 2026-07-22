# Canonical training-panel freshness/coverage contract (GOAL-5 AC7)   (PR #34)

STATUS:    delivered
WHAT:      Adds `assess_training_panel_freshness` in
           `src/renquant_common/training_freshness.py` — ONE pure,
           deterministic contract for both WF trainers to import (same
           anti-drift discipline as the artifact resolver / row_coverage).
           Returns `FreshnessVerdict{ok, reasons, max_date, n_days, ...}`.
           Checks over the training window (`date < required_through_date`):
           coverage (load-bearing: `max(date) >= required_through_date` +
           >=1 in-window row), recency (optional, off by default), and
           floors (min tickers/day, min rows, max intra-window date gap).
           Pure/deterministic: reads only date(+ticker) columns when a path
           is passed; zero I/O for an in-memory DataFrame.
WHY/DIR:   GOAL-5 AC7 (training-pipeline reliability track). Today both WF
           trainers reject only an EMPTY post-cutoff window; a
           stale-but-nonempty parquet that stops short of a fold's
           `data_end` silently trains on a truncated window (each fold
           slices `date < data_end`). This contract catches that at
           TRAINING time instead of silently shipping a truncated model.
EVIDENCE:  n/a (contract module + unit tests only, no model/data claim).
           `python3 -m pytest tests/test_training_freshness.py` -> 15 passed
           (covers: pass / each breach type / recency / empty / path vs
           DataFrame input / missing-ticker degrade).
NEXT:      Wiring into the PatchTST WF driver + Modal executor pre-dispatch
           lands in the paired renquant-backtesting PR (#77). XGB wiring
           (AC7 part 2), AC8 (auto-correct), AC9 (retry/self-heal) are
           follow-ups.
