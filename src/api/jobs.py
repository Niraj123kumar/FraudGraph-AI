"""Small execution boundary for background analysis jobs."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from fastapi import BackgroundTasks


class JobExecutor(Protocol):
    def submit(self, background_tasks: BackgroundTasks, job: Callable[[], None]) -> None:
        """Schedule a job for execution."""


class InProcessJobExecutor:
    """FastAPI-backed executor that can later be replaced by RQ or Arq."""

    def submit(self, background_tasks: BackgroundTasks, job: Callable[[], None]) -> None:
        background_tasks.add_task(job)
