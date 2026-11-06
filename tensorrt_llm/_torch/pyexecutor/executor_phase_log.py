# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Per-rank host phase timing of overlap-executor iterations (diagnostics).

``TRTLLM_EXECUTOR_PHASE_LOG=<ms>`` makes every rank log, for each iteration of
the overlap executor loop that admitted a request, ran a request's first
generation step, finished a request, or kept the host busy for more than
``<ms>`` milliseconds (the iteration's host time minus the wait for the
previous step, ``wait_previous``), one line with the host time spent between
consecutive phase marks (scheduling, first-step preparation, resource
preparation, first-token response, forward launch, wait for the previous step,
pending-response flush, sampling, previous-step processing) and the
wall-clock time of the iteration start. The device step time of rank 0's iteration log shows *that* an iteration
was slow; these lines show on which rank and in which host phase, and whether
the host waited for the GPU (a long "wait") or the GPU waited for the host (a
short "wait" after long earlier phases). Unset or empty: off, and the executor
only pays an attribute check per mark.

``wait_previous`` is the phase that contains the wait for the previous step to finish, and also
the host work that follows it in the same phase (per-request token processing of the previous
step, the acceptance-rate gate, the KV send); ``busy`` is the iteration's host time minus that
phase. The GPU waits of ``_wait_for_model_engine_input_copy`` inside the schedule, sample and
process_previous phases count as busy time.

Line format::

    [phase] rank=<r> iter=<n> t0=<unix s> total=<ms> busy=<ms> <mark>=<ms> ... [admitted=<k>] [first=<k>] [finished=<k>]

``iter`` is the executor's iteration counter during the iteration; the
``print_iter_log`` line that reports this iteration carries ``iter = <n + 1>``.
"""

import os
import time
from typing import Callable, Dict, List, Optional

PHASE_LOG_ENV = "TRTLLM_EXECUTOR_PHASE_LOG"


def phase_log_threshold_ms() -> Optional[float]:
    """The ``TRTLLM_EXECUTOR_PHASE_LOG`` threshold in ms, or None when off.

    A value that is not a non-negative number is reported and ignored (the log stays off): a
    diagnostic knob must not stop the executor from starting.
    """
    value = os.environ.get(PHASE_LOG_ENV, "")
    if not value.strip():
        return None
    try:
        threshold = float(value)
    except ValueError:
        threshold = -1.0
    if not threshold >= 0:  # also rejects NaN
        from tensorrt_llm.logger import logger

        logger.warning(
            f"{PHASE_LOG_ENV}={value!r} is not a non-negative number of milliseconds; "
            "the executor phase log stays off"
        )
        return None
    return threshold


class ExecutorPhaseLog:
    """Collects the phase marks of one executor iteration and decides whether to log it.

    Args:
        threshold_ms: Iterations whose busy host time (total minus the
            ``wait_mark`` phase) exceeds this are logged even without a request
            event.
        rank: Global rank written into every line.
        emit: Sink for a finished line (e.g. ``logger.info``).
        clock: Monotonic clock in seconds (injectable for tests).
        wall_clock: Wall clock in seconds since the epoch, for aligning ranks.
        wait_mark: The mark whose phase is the wait for the previous step.
    """

    def __init__(
        self,
        threshold_ms: float,
        rank: int,
        emit: Callable[[str], None],
        clock: Callable[[], float] = time.perf_counter,
        wall_clock: Callable[[], float] = time.time,
        wait_mark: str = "wait_previous",
    ):
        self.threshold_ms = threshold_ms
        self.rank = rank
        self._emit = emit
        self._clock = clock
        self._wall_clock = wall_clock
        self._wait_mark = wait_mark
        self._start: Optional[float] = None
        self._wall_start = 0.0
        self._last = 0.0
        self._marks: List[tuple] = []
        self._events: Dict[str, int] = {}

    def begin(self) -> None:
        """Start a new iteration (drops an unfinished previous one)."""
        self._start = self._clock()
        self._wall_start = self._wall_clock()
        self._last = self._start
        self._marks = []
        self._events = {}

    def mark(self, name: str) -> None:
        """Record the host time since the previous mark under ``name``."""
        if self._start is None:
            return
        now = self._clock()
        self._marks.append((name, (now - self._last) * 1e3))
        self._last = now

    def count(self, event: str, num: int) -> None:
        """Add ``num`` occurrences of a request event (admitted, first, finished)."""
        if num > 0:
            self._events[event] = self._events.get(event, 0) + num

    def end(self, iteration: int) -> Optional[str]:
        """Finish the iteration; emit and return its line if it qualifies."""
        if self._start is None:
            return None
        total_ms = (self._clock() - self._start) * 1e3
        self._start = None
        busy_ms = total_ms - sum(ms for name, ms in self._marks if name == self._wait_mark)
        if not self._events and busy_ms <= self.threshold_ms:
            return None
        parts = [
            f"[phase] rank={self.rank} iter={iteration} t0={self._wall_start:.6f} "
            f"total={total_ms:.3f} busy={busy_ms:.3f}"
        ]
        parts += [f"{name}={ms:.3f}" for name, ms in self._marks]
        parts += [f"{event}={num}" for event, num in self._events.items()]
        line = " ".join(parts)
        self._emit(line)
        return line
