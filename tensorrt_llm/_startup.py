# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host-only startup timing; never adds device or distributed synchronization."""

import os
import time
from contextlib import contextmanager
from typing import Iterator

from tensorrt_llm.logger import logger


class _StartupTimer:
    """Measure one process-local interval and its non-overlapping phases.

    Nested phases are logged but excluded from the summary to avoid counting
    their time twice. Different timers/processes must not be summed together.
    """

    def __init__(self, name: str):
        """Initialize an empty timer for the named startup interval."""
        self.name = name
        self.timings: dict[str, float] = {}
        self.depth = 0
        self.started = 0.0

    def __enter__(self):
        """Start the interval and emit its start marker."""
        self.started = time.perf_counter()
        logger.info(f"[startup][pid={os.getpid()}] {self.name}: start")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Log the complete or failed interval without suppressing exceptions."""
        total = time.perf_counter() - self.started
        measured = sum(self.timings.values())
        parts = ", ".join(f"{name}={seconds:.3f}s" for name, seconds in self.timings.items())
        status = "failed" if exc_type is not None else "done"
        logger.info(
            f"[startup][pid={os.getpid()}] {self.name}: {status}, total={total:.3f}s, "
            f"unattributed={max(0.0, total - measured):.3f}s | {parts}"
        )

    def mark_initialization(self, name: str) -> None:
        """Account for the prefix before any timed phase begins."""
        elapsed = time.perf_counter() - self.started
        self.timings[name] = elapsed
        logger.info(f"[startup][pid={os.getpid()}] {self.name}/{name}: done in {elapsed:.3f}s")

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        """Record outermost elapsed time, including failures, and log status.

        Failed work still contributes to startup time; its exception propagates
        and its completion marker is explicitly labelled ``failed``.
        """
        logger.info(f"[startup][pid={os.getpid()}] {self.name}/{name}: start")
        start = time.perf_counter()
        outermost = self.depth == 0
        self.depth += 1
        completed = False
        try:
            yield
            completed = True
        finally:
            elapsed = time.perf_counter() - start
            self.depth -= 1
            if outermost:
                self.timings[name] = self.timings.get(name, 0.0) + elapsed
            status = "done" if completed else "failed"
            logger.info(
                f"[startup][pid={os.getpid()}] {self.name}/{name}: {status} in {elapsed:.3f}s"
            )
