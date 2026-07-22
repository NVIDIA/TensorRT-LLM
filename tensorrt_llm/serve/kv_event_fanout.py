# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Single-consumer TRT-LLM KV-event pump with in-process fanout."""

import asyncio
from collections import deque
from collections.abc import AsyncGenerator
from typing import Any, cast

from tensorrt_llm.logger import logger


class KvEventSubscriberOverflow(RuntimeError):
    """Raised when a subscriber can no longer receive a lossless event stream."""


_SUBSCRIBER_OVERFLOW = object()


class KvEventFanout:
    """Own TRT-LLM's event queue and copy events to every protocol."""

    def __init__(self, llm: object, buffer_size: int = 1024) -> None:
        if buffer_size <= 0:
            raise ValueError("buffer_size must be positive")
        self._llm = llm
        self._buffer: deque[dict[str, Any]] = deque(maxlen=buffer_size)
        self._subscribers: dict[asyncio.Queue[object], frozenset[int]] = {}
        self._subscriber_overflow_count = 0
        self._task: asyncio.Task[None] | None = None
        self._sequence_numbers: dict[int, int] = {}
        self._last_engine_event_ids: dict[int, int] = {}
        self._max_window_size: int | None = None
        self._processing_initial_created_events = True

    @property
    def buffer_size(self) -> int:
        return self._buffer.maxlen or 0

    @property
    def subscriber_overflow_count(self) -> int:
        return self._subscriber_overflow_count

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._pump())

    async def stop(self) -> None:
        if self._task is None:
            return
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        self._task = None

    def drain_http_buffer(self) -> list[dict[str, Any]]:
        events = list(self._buffer)
        self._buffer.clear()
        return events

    async def subscribe(
        self, data_parallel_ranks: set[int] | None = None
    ) -> AsyncGenerator[tuple[int, dict[str, Any]], None]:
        self.start()
        queue: asyncio.Queue[object] = asyncio.Queue(maxsize=self.buffer_size)
        self._subscribers[queue] = frozenset(data_parallel_ranks or ())
        try:
            while True:
                item = await queue.get()
                if item is _SUBSCRIBER_OVERFLOW:
                    raise KvEventSubscriberOverflow(
                        "OpenEngine KV event subscriber queue overflowed"
                    )
                yield cast(tuple[int, dict[str, Any]], item)
        finally:
            self._subscribers.pop(queue, None)

    def _is_routable_event(self, event: dict[str, Any]) -> bool:
        event_type = event.get("data", event).get("type")
        if event_type == "created" and self._processing_initial_created_events:
            window_size = event.get("window_size")
            if window_size is not None:
                self._max_window_size = max(self._max_window_size or 0, int(window_size))
            return False
        self._processing_initial_created_events = False
        if event_type not in ("stored", "removed"):
            return False
        window_size = event.get("window_size")
        return (
            window_size is None
            or self._max_window_size is None
            or int(window_size) == self._max_window_size
        )

    def _publish(self, event: dict[str, Any]) -> None:
        rank = int(event.get("attention_dp_rank", 0))
        event_id = event.get("event_id")
        if isinstance(event_id, int) and not isinstance(event_id, bool):
            previous_event_id = self._last_engine_event_ids.get(rank)
            if (previous_event_id is None and event_id != 0) or (
                previous_event_id is not None and event_id != previous_event_id + 1
            ):
                logger.warning(
                    "TRT-LLM KV event gap on DP rank %d: expected %d, received %d; "
                    "clearing the advertised KV index",
                    rank,
                    0 if previous_event_id is None else previous_event_id + 1,
                    event_id,
                )
                self._publish_routable(
                    {
                        "attention_dp_rank": rank,
                        "data": {"type": "all_cleared"},
                    },
                    include_http_raw_event=False,
                )
            self._last_engine_event_ids[rank] = event_id
        if not self._is_routable_event(event):
            return
        self._publish_routable(event)

    def _publish_routable(
        self, event: dict[str, Any], *, include_http_raw_event: bool = True
    ) -> None:
        rank = int(event.get("attention_dp_rank", 0))
        sequence = self._sequence_numbers.get(rank, 0) + 1
        self._sequence_numbers[rank] = sequence
        sequenced_event = (sequence, event)
        if include_http_raw_event:
            self._buffer.append(event)
        for queue, selected_ranks in tuple(self._subscribers.items()):
            if selected_ranks and rank not in selected_ranks:
                continue
            try:
                queue.put_nowait(sequenced_event)
            except asyncio.QueueFull:
                self._subscriber_overflow_count += 1
                self._subscribers.pop(queue, None)
                while not queue.empty():
                    queue.get_nowait()
                queue.put_nowait(_SUBSCRIBER_OVERFLOW)
                logger.error(
                    "Closing slow OpenEngine KV subscriber after queue overflow; "
                    "lossless routing can no longer be guaranteed"
                )

    async def _pump(self) -> None:
        while True:
            try:
                async for event in self._llm.get_kv_cache_events_async(1):
                    self._publish(event)
            except (IndexError, asyncio.QueueEmpty):
                await asyncio.sleep(0)
            except (RuntimeError, AttributeError) as error:
                logger.debug("KV event pump unavailable: %s", error)
                await asyncio.sleep(1)
