# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``GET /metrics``: iteration statistics, and the collector loop that feeds it."""

import asyncio
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional

from fastapi.responses import JSONResponse

from tensorrt_llm.executor.rpc import RPCError
from tensorrt_llm.logger import logger
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

    @staticmethod
    def _iteration_stats_buffer_maxlen(
            iter_stats_max_iterations: Optional[int]) -> Optional[int]:
        if iter_stats_max_iterations is None or iter_stats_max_iterations == 0:
            return 1000
        if iter_stats_max_iterations < 0:
            return None
        return iter_stats_max_iterations

    async def get_iteration_stats(self) -> JSONResponse:
        # Visual-gen deployments use a separate iteration-stats producer
        # (see ``DiffusionRemoteClient._iter_stats``).  The LLM PyExecutor
        # stats queue is empty in visual-gen mode, so route /metrics to the
        # visual-gen-shaped producer instead. Each snapshot mirrors LLM
        # field names where it makes sense (numActiveRequests,
        # numQueuedRequests) so downstream consumers that already parse the
        # LLM /metrics shape work for visual-gen with minimal changes.
        if self._is_visual_gen:
            stats = []
            async for stat in self.generator.get_stats_async(timeout=None):
                stats.append(stat)
            return JSONResponse(content=stats)

        # Without iteration stats the engine produces none, so there is
        # nothing to wait for.
        if not getattr(self.generator.args, "enable_iter_perf_stats", True):
            return JSONResponse(content=[])

        # When the background collector loop is active it is the sole
        # consumer of the engine stats queue; serve /metrics from the tee
        # buffer it populates so we do not race it for queue items. Racing
        # caused >80% iteration loss and ~2s per-call latency (nvbug 6102381).
        # The caller receives the stats accumulated since the previous /metrics
        # call (up to iter_stats_max_iterations) and the buffer is cleared.
        if self._iteration_stats_buffer is not None:
            stats = list(self._iteration_stats_buffer)
            self._iteration_stats_buffer.clear()
            return JSONResponse(content=stats)

        # Legacy path: no background collector -> read the queue directly.
        # The read waits up to 2 s for stats inside a blocking executor call,
        # so it runs on the metrics thread; on the event loop it would stall
        # every response being streamed for as long as it waits.
        # One read at a time: a scrape that arrives while a read is in flight
        # returns no stats instead of queueing behind it, so scraping an idle
        # server faster than one read per 2 s cannot build a backlog. A read
        # drains the queue and hands everything to its own scrape, so when two
        # scrapers poll at once, one of them regularly gets an empty list.
        read = self._metrics_stats_read
        if read is not None and (not read.done()
                                 or self._metrics_stats_read_waiting):
            # In flight, or finished with its scrape about to take the stats.
            return JSONResponse(content=[])
        if (read is None or self._metrics_stats_read_consumed
                or read.cancelled()):
            read = asyncio.get_running_loop().run_in_executor(
                self._metrics_stats_executor, self.generator.get_stats, 2)
            self._metrics_stats_read = read
            self._metrics_stats_read_consumed = False
        # Otherwise the last read finished after its scrape went away, and
        # this scrape takes the stats it drained rather than losing them.
        self._metrics_stats_read_waiting = True
        try:
            # Shielded so a scrape that goes away leaves the read running
            # until its thread finishes, and its stats for the next scrape.
            stats = await asyncio.shield(read)
        except RPCError as e:
            # The executor is unreachable, e.g. while it shuts down.
            logger.debug(f"Failed to fetch iteration stats: {e}")
            stats = []
        except Exception:
            # The read failed; the next scrape starts a new one.
            self._metrics_stats_read_consumed = True
            raise
        finally:
            self._metrics_stats_read_waiting = False
        self._metrics_stats_read_consumed = True
        return JSONResponse(content=stats)

    async def _iteration_stats_collector_loop(self):
        """Background task that continuously collects iteration statistics from the LLM engine.

        This task runs in the background for the lifetime of the server and drains iteration
        stats from the engine's stats queue, logging every stat to Prometheus.  Gauges
        (kv_cache_hit_rate, kv_cache_utilization, kv_cache_iter_reuse_rate) are naturally
        overwritten with the latest value, while counters (missed_blocks_total,
        gen_alloc_blocks_total, etc.) must be incremented by *every* per-iteration delta
        to remain accurate.  Logging only the latest stat would drop counter deltas from
        earlier iterations and could leave gauges unset if the latest iteration had no
        context-phase activity.

        The task sleeps when idle and is woken up via _iteration_stats_wakeup_event when
        requests complete.

        The loop will continue until the task is cancelled during server shutdown.
        """
        try:
            logger.info("Iteration stats collector loop started")
            while True:
                # Wait for signal that requests have completed and stats may
                # be available (a multi-frontend launcher also polls).
                woken_by_timer = False
                try:
                    await asyncio.wait_for(
                        self._iteration_stats_wakeup_event.wait(),
                        timeout=self._iteration_stats_poll_interval)
                except asyncio.TimeoutError:
                    woken_by_timer = True

                # Clear the event for next wakeup
                self._iteration_stats_wakeup_event.clear()

                # Drain all available iteration stats and log each one to Prometheus.
                # The fetch blocks for up to its timeout, so it runs on the
                # metrics thread, which /metrics leaves idle while this loop runs.
                try:
                    # A timer wakeup (multi-frontend launcher poll) reads
                    # without waiting.
                    for llm_stat in await asyncio.get_running_loop(
                    ).run_in_executor(self._metrics_stats_executor,
                                      self.generator.get_stats,
                                      0 if woken_by_timer else 0.5):
                        self._record_iteration_stat(llm_stat)
                except Exception as e:
                    # Log errors but continue collecting stats
                    logger.error(f"Error collecting iteration stats: {e}",
                                 exc_info=True)
                    # Brief sleep to avoid tight loop on persistent errors
                    await asyncio.sleep(0.1)
        except asyncio.CancelledError:
            logger.info("Iteration stats collector loop cancelled")
            raise

    def _record_iteration_stat(self, llm_stat: dict) -> None:
        self.metrics_collector.log_iteration_stats(llm_stat)
        # Tee into the /metrics snapshot buffer so the HTTP handler can serve
        # without competing for the engine queue (nvbug 6102381).
        if self._iteration_stats_buffer is not None:
            self._iteration_stats_buffer.append(llm_stat)
