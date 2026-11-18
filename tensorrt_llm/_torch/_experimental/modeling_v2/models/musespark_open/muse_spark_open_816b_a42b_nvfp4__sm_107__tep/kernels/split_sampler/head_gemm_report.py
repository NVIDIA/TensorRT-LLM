# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""When a rank reports which GEMM its LM head runs (``target_head.TargetHead.produce``).

The built-in low-M dispatcher takes the LM head's GEMM through a timed autotuner race, per rank
and per process. Its choice is final only after the autotuner's warmup: a forward before it (KV
cache sizing, a dummy request) finds no cache entry and runs the fallback runner, which is not
what the rank serves. ``HeadGemmReport`` decides, per row count, when the choice is worth a log
line:

* never while the autotuner tunes;
* on a cache hit (a tuned entry, or one loaded from a cache file);
* on a cache miss only after a tuning pass has been seen: the fallback is then what serves;
* again whenever the reported choice changes.

The cache is looked up only when the autotuner's cache has changed size, or a tuning pass has
been seen, since that row count's last lookup, so a steady call costs a tuple comparison. The
module imports nothing, so the decision runs on any host.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

#: ``lookup()``'s result: whether the autotuner's cache holds the shape, and the choice in words.
Lookup = Callable[[], Tuple[bool, str]]


class HeadGemmReport:
    """Which LM-head GEMM choices to log, per row count (module docstring)."""

    def __init__(self) -> None:
        #: Whether a call has been seen while the autotuner tuned.
        self.tuning_seen = False
        #: Row count -> (cache size, tuning seen) at its last lookup.
        self._looked_up: Dict[int, Tuple[int, bool]] = {}
        #: Row count -> the choice last reported.
        self._reported: Dict[int, str] = {}

    def observe(self, rows: int, tuning: bool, cache_size: int, lookup: Lookup) -> Optional[str]:
        """The text to log for this call of ``rows`` rows, or None.

        Args:
            rows: the call's row count.
            tuning: whether the autotuner is tuning now (``AutoTuner.is_tuning_mode``).
            cache_size: the number of entries in the autotuner's cache now.
            lookup: reads the cached choice for this call (no profiling, no launch).
        """
        if tuning:
            self.tuning_seen = True
            return None
        state = (cache_size, self.tuning_seen)
        if self._looked_up.get(rows) == state:
            return None
        self._looked_up[rows] = state
        hit, choice = lookup()
        if not hit and not self.tuning_seen:
            # Before the autotuner's warmup: the fallback this call runs is not the rank's choice.
            return None
        previous = self._reported.get(rows)
        if previous == choice:
            return None
        self._reported[rows] = choice
        return choice if previous is None else f"{choice} (was {previous})"
