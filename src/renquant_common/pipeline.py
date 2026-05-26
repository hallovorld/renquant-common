"""Domain-neutral Task/Job/Pipeline primitives.

These classes are intentionally stdlib-only. Model training, inference,
backtesting, and execution repos should compose these primitives instead of
creating repo-local orchestration frameworks.
"""
from __future__ import annotations

import logging
import os
import time
from abc import ABC, abstractmethod
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from threading import current_thread
from typing import Any, Iterable

log = logging.getLogger("renquant_common.pipeline")


class ParallelTimeoutError(RuntimeError):
    """Raised when a parallel phase exceeds its wall-clock budget."""

    def __init__(self, job_name: str, elapsed: float, pending_labels: list[str]) -> None:
        self.job_name = job_name
        self.elapsed = float(elapsed)
        self.pending_labels = list(pending_labels)
        super().__init__(
            f"{job_name} timed out after {elapsed:.2f}s with "
            f"{len(pending_labels)} pending item(s): {', '.join(pending_labels[:20])}"
        )


@dataclass(frozen=True)
class PipelineStepRecord:
    """Audit record for one Job inside a Pipeline run."""

    job_name: str
    skipped: bool
    elapsed_sec: float


@dataclass(frozen=True)
class PipelineResult:
    """Structured result returned by Pipeline.run()."""

    name: str
    ok: bool
    elapsed_sec: float
    steps: tuple[PipelineStepRecord, ...] = field(default_factory=tuple)


def resolve_workers(config_value: int | None, item_count: int) -> int:
    """Resolve worker count: explicit positive config wins; auto uses cpu_count-2."""
    if item_count <= 0:
        return 0
    if config_value is not None and config_value > 0:
        n = int(config_value)
    else:
        n = max(1, (os.cpu_count() or 4) - 2)
    return min(n, item_count)


class Task(ABC):
    """Atomic step within a Job.

    ``run`` returns True or None to continue the chain, and False to stop the
    current Job early. Gate tasks use False to short-circuit downstream work.
    """

    @abstractmethod
    def run(self, ctx: Any) -> bool | None:
        """Execute the task against a context object."""

    @property
    def name(self) -> str:
        return type(self).__name__


class Job(ABC):
    """Sequential chain of Tasks."""

    @property
    def tasks(self) -> list[Task]:
        return []

    def should_skip(self, ctx: Any) -> bool:
        return False

    def run(self, ctx: Any) -> None:
        for task in self.tasks:
            if task.run(ctx) is False:
                log.debug("[%s] chain stopped by %s", type(self).__name__, task.name)
                return


class Pipeline:
    """Sequential Job pipeline with structured audit records."""

    def __init__(self, jobs: Iterable[Job], *, name: str | None = None) -> None:
        self.jobs = list(jobs)
        self.name = name or type(self).__name__

    def run(self, ctx: Any) -> PipelineResult:
        started = time.monotonic()
        records: list[PipelineStepRecord] = []
        for job in self.jobs:
            job_name = type(job).__name__
            step_started = time.monotonic()
            skipped = bool(job.should_skip(ctx))
            if not skipped:
                job.run(ctx)
            records.append(
                PipelineStepRecord(
                    job_name=job_name,
                    skipped=skipped,
                    elapsed_sec=time.monotonic() - step_started,
                )
            )
        return PipelineResult(
            name=self.name,
            ok=True,
            elapsed_sec=time.monotonic() - started,
            steps=tuple(records),
        )


def run_parallel(
    item_contexts: list[Any],
    job: Job,
    max_workers: int | None = None,
    timeout_seconds: float | None = None,
    progress_log_seconds: float | None = None,
) -> None:
    """Run ``job.run(ctx)`` for each context in parallel.

    Worker exceptions are logged and do not stop sibling items. A phase timeout
    is a hard failure because downstream steps must not consume a partial set.
    """
    if not item_contexts:
        return
    if progress_log_seconds is None:
        progress_log_seconds = 30.0
    job_name = type(job).__name__
    n_workers = resolve_workers(max_workers, len(item_contexts))
    log.info(
        "run_parallel: %s  %d items  %d workers  timeout=%s",
        job_name,
        len(item_contexts),
        n_workers,
        timeout_seconds,
    )
    started = time.monotonic()
    executor = ThreadPoolExecutor(max_workers=n_workers, thread_name_prefix="rq")
    futures = {executor.submit(_wrapped_run, job, ctx): _ctx_label(ctx) for ctx in item_contexts}
    pending = set(futures)
    completed = 0
    progress_interval = max(0.01, float(progress_log_seconds or 0.0))
    next_progress = started + progress_interval
    abandon_executor = False
    try:
        while pending:
            now = time.monotonic()
            elapsed = now - started
            if timeout_seconds is not None and elapsed >= float(timeout_seconds):
                pending_labels = sorted(futures[f] for f in pending)
                for future in pending:
                    future.cancel()
                log.error(
                    "run_parallel: %s TIMEOUT after %.2fs; done=%d/%d "
                    "pending=%d labels=%s; worker may still be running",
                    job_name,
                    elapsed,
                    completed,
                    len(futures),
                    len(pending_labels),
                    pending_labels[:20],
                )
                executor.shutdown(wait=False, cancel_futures=True)
                abandon_executor = True
                raise ParallelTimeoutError(job_name, elapsed, pending_labels)

            wait_timeout = max(0.0, next_progress - now)
            if timeout_seconds is not None:
                wait_timeout = min(wait_timeout, max(0.0, float(timeout_seconds) - elapsed))
            done, pending = wait(pending, timeout=wait_timeout, return_when=FIRST_COMPLETED)

            for future in done:
                label = futures[future]
                completed += 1
                try:
                    future.result()
                except Exception as exc:  # noqa: BLE001
                    log.error(
                        "run_parallel [%s] %s ERROR - %s: %s",
                        label,
                        job_name,
                        type(exc).__name__,
                        exc,
                    )

            now = time.monotonic()
            if pending and now >= next_progress:
                pending_labels = sorted(futures[f] for f in pending)
                log.info(
                    "run_parallel: %s progress done=%d/%d pending=%d "
                    "elapsed=%.2fs pending_labels=%s",
                    job_name,
                    completed,
                    len(futures),
                    len(pending_labels),
                    now - started,
                    pending_labels[:10],
                )
                next_progress = now + progress_interval
    finally:
        if not abandon_executor:
            executor.shutdown(wait=True)
    log.info("run_parallel: %s DONE  %.2fs", job_name, time.monotonic() - started)


def _wrapped_run(job: Job, ctx: Any) -> None:
    label = _ctx_label(ctx)
    log.debug("[%s|%s] %s START", label, current_thread().name, type(job).__name__)
    started = time.monotonic()
    job.run(ctx)
    log.debug(
        "[%s|%s] %s DONE  %.2fs",
        label,
        current_thread().name,
        type(job).__name__,
        time.monotonic() - started,
    )


def _ctx_label(ctx: Any) -> str:
    for attr in ("ticker", "symbol", "name", "id"):
        value = getattr(ctx, attr, None)
        if value is not None:
            return str(value)
    if isinstance(ctx, dict):
        for key in ("ticker", "symbol", "name", "id"):
            if key in ctx:
                return str(ctx[key])
    return type(ctx).__name__
