# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host-only warmup timing; never adds device or distributed synchronization."""

import os
import time
from collections.abc import Iterator
from contextlib import contextmanager

from tensorrt_llm.logger import logger


class _WarmupTimer:
    """Record process-local warmup phases without adding synchronization.

    Shape intervals use ``record=False`` because their enclosing phase already
    accounts for their duration. Exceptions propagate after partial timings are
    logged. Reusing the timer starts a fresh breakdown with a new pass number.
    """

    def __init__(self, rank: int):
        self.rank = rank
        self.purpose = "unspecified"
        self.pass_index = 0
        self.timings: dict[str, float] = {}
        self.started = 0.0

    def __enter__(self):
        self.timings = {}
        self.pass_index += 1
        self.started = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.summary(time.perf_counter() - self.started)

    @property
    def prefix(self) -> str:
        return (
            f"[warmup][pid={os.getpid()}][rank={self.rank}]"
            f"[purpose={self.purpose}][pass={self.pass_index}]"
        )

    @contextmanager
    def phase(self, name: str, *, record: bool = True, log_start: bool = True) -> Iterator[None]:
        """Log start/end of one warmup phase and record its wall-clock time.

        ``name`` should match the ``log_mem_snapshot`` tag of the same phase
        (``warmup/after_<name>``) so timing and memory logs line up.
        """
        if log_start:
            logger.info(f"{self.prefix} {name}: start")
        start = time.perf_counter()
        completed = False
        try:
            yield
            completed = True
        finally:
            elapsed = time.perf_counter() - start
            if record:
                self.timings[name] = self.timings.get(name, 0.0) + elapsed
            if completed:
                logger.info(f"{self.prefix} {name}: done in {elapsed:.1f}s")
            else:
                logger.warning(f"{self.prefix} {name}: failed after {elapsed:.1f}s")

    # A single phase taking longer than this is reported at WARNING level in
    # the summary, so a slow JIT/autotune shows up without grepping.
    _WARMUP_SLOW_PHASE_SEC = 60.0

    def summary(self, total_sec: float) -> None:
        """Report local timings subject to the configured logger rank filtering."""
        if not self.timings:
            logger.info(f"{self.prefix} summary: total={total_sec:.1f}s (no phases ran)")
            return
        parts = []
        for name, sec in self.timings.items():
            pct = 100.0 * sec / total_sec if total_sec > 0 else 0.0
            parts.append(f"{name}={sec:.1f}s ({pct:.0f}%)")
        logger.info(f"{self.prefix} summary: total={total_sec:.1f}s | " + ", ".join(parts))
        slow = {
            name: sec for name, sec in self.timings.items() if sec > self._WARMUP_SLOW_PHASE_SEC
        }
        if slow:
            logger.warning(
                f"{self.prefix} slow phases (>"
                + f"{self._WARMUP_SLOW_PHASE_SEC:.0f}s): "
                + ", ".join(f"{name}={sec:.1f}s" for name, sec in slow.items())
            )
