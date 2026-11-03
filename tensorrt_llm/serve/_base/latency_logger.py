# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Periodic latency percentiles for the disaggregated coordinator and its clients."""

from typing import List

from tensorrt_llm.logger import logger


class PeriodicLatencyLogger:
    """Periodically log latency percentiles for a named coordinator API.

    This lock-free, self-resetting logger runs on one asyncio loop and profiles
    the in-process owner and HTTP client without per-call log spam.
    """

    def __init__(self, name: str, window: int = 500):
        self._name = name
        self._window = window
        self._samples: List[float] = []
        self._n = 0

    def record(self, dt_s: float) -> None:
        self._samples.append(dt_s * 1000.0)  # ms
        self._n += 1
        if self._n % self._window == 0:
            s = sorted(self._samples)
            m = len(s)

            def percentile(q):
                return s[min(int(q * m), m - 1)]

            logger.info(f"[coord_api] {self._name} n={self._n} ms: "
                        f"mean={sum(s)/m:.2f} p50={percentile(0.5):.2f} "
                        f"p90={percentile(0.9):.2f} "
                        f"p99={percentile(0.99):.2f} max={s[-1]:.2f}")
            self._samples = []
