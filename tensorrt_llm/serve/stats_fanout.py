# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Single-consumer TRT-LLM iteration-stat pump with shared snapshots."""

import asyncio
from collections import deque
from collections.abc import Callable
from typing import Any

from tensorrt_llm.logger import logger


class StatsFanout:
    """Own the engine stats queue and tee it to HTTP, metrics, and OpenEngine."""

    def __init__(self, llm: object, buffer_size: int | None = 1000) -> None:
        self._llm = llm
        self._buffer: deque[dict[str, Any]] = deque(maxlen=buffer_size)
        self._latest_by_rank: dict[int, dict[str, Any]] = {}
        self._consumer: Callable[[dict[str, Any]], None] | None = None
        self._wake_event: asyncio.Event | None = None
        self._task: asyncio.Task[None] | None = None

    def start(self, consumer: Callable[[dict[str, Any]], None] | None = None) -> None:
        self._consumer = consumer
        if self._task is None:
            self._wake_event = asyncio.Event()
            self._task = asyncio.create_task(self._pump())

    def wake(self) -> None:
        if self._wake_event is not None:
            self._wake_event.set()

    async def stop(self) -> None:
        if self._task is None:
            return
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        self._task = None
        self._wake_event = None

    def drain_http_buffer(self) -> list[dict[str, Any]]:
        stats = list(self._buffer)
        self._buffer.clear()
        return stats

    def latest_by_rank(self) -> dict[int, dict[str, Any]]:
        return {rank: dict(stat) for rank, stat in self._latest_by_rank.items()}

    def _publish(self, stat: dict[str, Any]) -> None:
        rank = int(stat.get("attentionDpRank", stat.get("attention_dp_rank", 0)))
        self._latest_by_rank[rank] = stat
        self._buffer.append(stat)
        if self._consumer is not None:
            try:
                self._consumer(stat)
            except (RuntimeError, ValueError, TypeError) as error:
                logger.error("Iteration-stat consumer failed: %s", error)

    async def _pump(self) -> None:
        assert self._wake_event is not None
        while True:
            await self._wake_event.wait()
            self._wake_event.clear()
            try:
                async for stat in self._llm.get_stats_async(timeout=0.5):
                    self._publish(stat)
            except (
                RuntimeError,
                AttributeError,
                IndexError,
                TypeError,
                ValueError,
                asyncio.QueueEmpty,
            ) as error:
                logger.error("Error collecting iteration stats: %s", error)
                await asyncio.sleep(0.1)
