from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import pytest

from renquant_common import Job, ParallelTimeoutError, Pipeline, Task, run_parallel


@dataclass
class Ctx:
    ticker: str = "AAA"
    values: list[str] | None = None
    skip: bool = False


class AppendTask(Task):
    def __init__(self, value: str, result: bool | None = None) -> None:
        self.value = value
        self.result = result

    def run(self, ctx: Ctx) -> bool | None:
        if ctx.values is None:
            ctx.values = []
        ctx.values.append(self.value)
        return self.result


class ChainJob(Job):
    def __init__(self, *tasks: Task, skip: bool = False) -> None:
        self._tasks = list(tasks)
        self._skip = skip

    @property
    def tasks(self) -> list[Task]:
        return self._tasks

    def should_skip(self, ctx: Ctx) -> bool:
        return self._skip or ctx.skip


def test_task_false_short_circuits_job() -> None:
    ctx = Ctx(values=[])
    ChainJob(AppendTask("a"), AppendTask("b", False), AppendTask("c")).run(ctx)
    assert ctx.values == ["a", "b"]


def test_pipeline_records_steps_and_skips() -> None:
    ctx = Ctx(values=[])
    result = Pipeline(
        [
            ChainJob(AppendTask("a")),
            ChainJob(AppendTask("skip"), skip=True),
            ChainJob(AppendTask("b")),
        ],
        name="unit",
    ).run(ctx)

    assert result.ok is True
    assert result.name == "unit"
    assert ctx.values == ["a", "b"]
    assert [s.job_name for s in result.steps] == ["ChainJob", "ChainJob", "ChainJob"]
    assert [s.skipped for s in result.steps] == [False, True, False]


class SleepJob(Job):
    def __init__(self, delays: dict[str, float]) -> None:
        self.delays = delays

    def run(self, ctx: Ctx) -> None:
        time.sleep(self.delays.get(ctx.ticker, 0.0))
        ctx.values = [ctx.ticker]


def test_run_parallel_success_preserves_context_mutations() -> None:
    ctxs = [Ctx("A"), Ctx("B")]
    run_parallel(ctxs, SleepJob({"A": 0.0, "B": 0.0}), max_workers=2, timeout_seconds=1.0)
    assert [ctx.values for ctx in ctxs] == [["A"], ["B"]]


def test_run_parallel_timeout_is_hard_failure(caplog: pytest.LogCaptureFixture) -> None:
    ctxs = [Ctx("FAST"), Ctx("SLOW")]
    caplog.set_level(logging.INFO, logger="renquant_common.pipeline")

    with pytest.raises(ParallelTimeoutError) as excinfo:
        run_parallel(
            ctxs,
            SleepJob({"FAST": 0.0, "SLOW": 0.20}),
            max_workers=2,
            timeout_seconds=0.03,
            progress_log_seconds=0.01,
        )

    assert excinfo.value.job_name == "SleepJob"
    assert "SLOW" in excinfo.value.pending_labels
    assert any("TIMEOUT" in record.message for record in caplog.records)
