# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``GET /metrics``: iteration statistics, and the collector loop that feeds it."""

import asyncio
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional

from tensorrt_llm.metrics.collector import MetricsCollector
from tensorrt_llm.serve._base.serving import BaseServing


class MetricsServing(BaseServing):
    """Iteration statistics for every role but EMBEDDING.

    Owns the stats machinery: the tee buffer the collector loop fills, the
    read state of the direct queue path and the thread the blocking reads
    run on. The composition root's lifespan starts and stops the collector
    loop and keeps the task.
    """

    def __init__(self, *, model: str, generator: Any, is_visual_gen: bool,
                 metrics_collector: Optional[MetricsCollector],
                 iteration_stats_wakeup_event: asyncio.Event,
                 iteration_stats_poll_interval: Optional[float]) -> None:
        self.model = model
        self.generator = generator
        self._is_visual_gen = is_visual_gen
        self.metrics_collector = metrics_collector
        self._iteration_stats_wakeup_event = iteration_stats_wakeup_event
        self._iteration_stats_poll_interval = iteration_stats_poll_interval
        # /metrics reads iteration stats with a blocking call that waits up to
        # 2 s. A dedicated thread keeps those waits off the default executor,
        # which request handling also uses; get_iteration_stats runs at most
        # one read on it at a time.
        self._metrics_stats_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="trtllm_metrics_stats",
        )
        self._metrics_stats_read: Optional[asyncio.Future] = None
        # Whether a scrape is awaiting the last read, and whether a scrape has
        # returned its stats.
        self._metrics_stats_read_waiting = False
        self._metrics_stats_read_consumed = True
        # Bounded snapshot of iteration stats for the GET /metrics handler.
        # When the background Prometheus collector loop is active, it is the
        # sole consumer of the engine's stats queue and appends each drained
        # stat here; /metrics then serves from this buffer instead of racing
        # the loop for the queue. Created lazily when the loop starts.
        # See nvbug 6102381.
        self._iteration_stats_buffer: Optional[deque] = None
