# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import datetime
import queue
import threading
import time
from itertools import repeat
from typing import Iterable, List, Optional, Tuple

from tensorrt_llm.llmapi.disagg_utils import get_local_request_id

from ..disaggregation.diagnostics import (DisaggLifecycleEmitter,
                                          DisaggLifecycleEmitterName,
                                          DisaggLifecycleEvent)
from ..distributed import Distributed
from .llm_request import ExecutorRequest
from .request_utils import get_num_child_requests

SHUTDOWN_REQUEST_ID = -1
CONTROL_REQUEST_ID = -2


@dataclasses.dataclass
class RequestQueueItem:
    id: int
    request: Optional[ExecutorRequest] = None
    _ = dataclasses.KW_ONLY
    child_req_ids: Optional[list] = None
    is_canceled_request: bool = False
    query: Optional[list] = None  # only used in `StarAttention`
    # Only meaningful for control requests. True = drain active/waiting
    # queues before firing the action. False = fire at the next scheduler
    # step boundary with in-flight requests still in the engine.
    # See ``PyExecutor.control_action``.
    control_requires_drain: bool = True
    control_id: Optional[str] = None

    @property
    def is_shutdown_request(self):
        return self.id == SHUTDOWN_REQUEST_ID

    @property
    def is_normal_request(self):
        return not (self.is_shutdown_request or self.is_canceled_request
                    or self.is_control_request)

    @property
    def is_control_request(self):
        return self.id == CONTROL_REQUEST_ID


class ExecutorRequestQueue:
    """Handles basic queue operations for executor requests."""

    def __init__(
        self,
        dist: Distributed,
        max_batch_size: int,
        enable_iter_perf_stats: bool,
        batch_wait_timeout_ms: float,
        disagg_lifecycle_emitter: Optional[DisaggLifecycleEmitter] = None,
    ):
        self.dist = dist
        self.request_queue: queue.Queue[RequestQueueItem] = queue.Queue()
        self.max_batch_size = max_batch_size
        self.enqueue_lock = threading.Lock()
        self.next_request_id = max_batch_size
        self.enable_iter_perf_stats = enable_iter_perf_stats
        self.start_times = {}
        self.active = True
        self.batch_wait_timeout_ms = batch_wait_timeout_ms
        self._disagg_lifecycle_emitter = disagg_lifecycle_emitter

    def _get_request_id(self, request: Optional[ExecutorRequest] = None):
        # if request has a disagg_request_id, use it as request id so that
        # corresponding context and generation requests have the same request id
        if request and request.disagg_request_id and isinstance(
                request.disagg_request_id, int):
            return request.disagg_request_id

        current_id = self.next_request_id
        self.next_request_id = get_local_request_id(current_id)
        return current_id

    def _generate_child_request_ids(
            self, request: ExecutorRequest) -> List[int] | None:
        """ Generate child request IDs if needed. """
        child_req_ids = None
        num_children = get_num_child_requests(request)
        if num_children > 0:
            child_req_ids = []
            for _ in range(num_children):
                child_req_id = self._get_request_id()
                if self.enable_iter_perf_stats:
                    self.start_times[child_req_id] = time.time()
                child_req_ids.append(child_req_id)

        return child_req_ids

    def _enqueue_impl(
        self, requests_and_queries: Iterable[Tuple[ExecutorRequest,
                                                   Optional[List]]]
    ) -> List[int]:
        req_ids = []
        lifecycle_arrivals = []
        with self.enqueue_lock:
            assert self.active, "PyExecutor has already been shutdown."
            start_time = time.time()
            for request, query in requests_and_queries:
                req_id = self._get_request_id(request)
                if self.enable_iter_perf_stats:
                    self.start_times[req_id] = start_time
                child_req_ids = self._generate_child_request_ids(request)

                if (self._disagg_lifecycle_emitter is not None
                        and self._disagg_lifecycle_emitter.enabled):
                    lifecycle_arrivals.append(
                        (self._disagg_lifecycle_emitter.capture_stamp(),
                         request, req_id))
                self.request_queue.put(
                    RequestQueueItem(req_id,
                                     request,
                                     child_req_ids=child_req_ids,
                                     query=query))
                req_ids.append(req_id)
        if self._disagg_lifecycle_emitter is not None:
            for stamp, request, req_id in lifecycle_arrivals:
                if stamp is None:
                    self._disagg_lifecycle_emitter.publish(None)
                    continue
                self._disagg_lifecycle_emitter.emit_for_role(
                    ctx_event=DisaggLifecycleEvent.CTX_ARRIVED,
                    gen_event=DisaggLifecycleEvent.GEN_ARRIVED,
                    request=request,
                    emitter=DisaggLifecycleEmitterName.EXECUTOR_QUEUE,
                    local_request_id=req_id,
                    stamp=stamp,
                )
        return req_ids

    def emit_disagg_dequeue_events(
            self, request_items: Iterable[RequestQueueItem]) -> None:
        """Record requests removed from the rank-zero engine queue."""

        self._emit_disagg_role_events(
            request_items,
            ctx_event=DisaggLifecycleEvent.CTX_DEQUEUED,
            gen_event=DisaggLifecycleEvent.GEN_DEQUEUED,
        )

    def emit_disagg_dispatch_events(
            self, request_items: Iterable[RequestQueueItem]) -> None:
        """Record requests accepted for rank-zero waiting-queue dispatch."""

        self._emit_disagg_role_events(
            request_items,
            ctx_event=DisaggLifecycleEvent.CTX_DISPATCHED,
            gen_event=DisaggLifecycleEvent.GEN_DISPATCHED,
        )

    def emit_disagg_waiting_release_events(
            self, request_items: Iterable[RequestQueueItem]) -> None:
        """Record rank-zero release from the replicated waiting queue."""

        self._emit_disagg_role_events(
            request_items,
            ctx_event=DisaggLifecycleEvent.CTX_WAITING_RELEASED,
            gen_event=DisaggLifecycleEvent.GEN_WAITING_RELEASED,
        )

    def _emit_disagg_role_events(
        self,
        request_items: Iterable[RequestQueueItem],
        *,
        ctx_event: DisaggLifecycleEvent,
        gen_event: DisaggLifecycleEvent,
    ) -> None:
        emitter = self._disagg_lifecycle_emitter
        if emitter is None or not emitter.enabled:
            return
        for item in request_items:
            if not item.is_normal_request or item.request is None:
                continue
            emitter.emit_for_role(
                ctx_event=ctx_event,
                gen_event=gen_event,
                request=item.request,
                emitter=DisaggLifecycleEmitterName.EXECUTOR_QUEUE,
                local_request_id=item.id,
            )

    def enqueue_requests(self, requests: List[ExecutorRequest]) -> List[int]:
        """
        Enqueue new requests
        """
        return self._enqueue_impl(zip(requests, repeat(None)))

    def enqueue_request(self,
                        request: ExecutorRequest,
                        query: Optional[List] = None) -> int:
        """
        Enqueue a new request, query is only used in `StarAttention`.
        """
        return self._enqueue_impl([(request, query)])[0]

    def enqueue_cancel_request(self, req_id: int):
        with self.enqueue_lock:
            self.request_queue.put(
                RequestQueueItem(req_id, is_canceled_request=True))

    def enqueue_control_request(self,
                                drain: bool = True,
                                control_id: Optional[str] = None):
        """Enqueue a control-action sentinel. See ``PyExecutor.control_action``."""
        with self.enqueue_lock:
            self.request_queue.put(
                RequestQueueItem(id=CONTROL_REQUEST_ID,
                                 control_requires_drain=drain,
                                 control_id=control_id))

    def enqueue_shutdown_request(self):
        with self.enqueue_lock:
            self.request_queue.put(RequestQueueItem(SHUTDOWN_REQUEST_ID))
            self.active = False

    def can_enqueue_request(self) -> bool:
        with self.enqueue_lock:
            return self.active and self.dist.rank == 0

    def get_from_request_queue(
            self,
            timeout: Optional[datetime.timedelta]) -> List[RequestQueueItem]:
        """Fetch requests from the queue with optional timeout.

        Args:
            timeout: Optional timeout for waiting on queue.

        Returns:
            List of RequestQueueItem fetched from the queue.
        """
        items = []
        timeout_secs = timeout.total_seconds() if timeout is not None else None

        try:
            if self.request_queue.empty() and (timeout_secs is None
                                               or timeout_secs > 0):
                # if queue is empty and want to wait, wait
                items.append(self.request_queue.get(timeout=timeout_secs))
            else:
                # if not empty or don't want to wait, just return all items in queue
                while True:
                    queue_item = self.request_queue.get_nowait()
                    items.append(queue_item)
        except queue.Empty:
            pass

        if self.batch_wait_timeout_ms == 0:
            return items

        if len(items) >= self.max_batch_size:
            return items

        deadline = time.monotonic() + self.batch_wait_timeout_ms / 1000.0
        while len(items) < self.max_batch_size:
            remaining_timeout = deadline - time.monotonic()

            if remaining_timeout <= 0:
                break

            try:
                item = self.request_queue.get(timeout=remaining_timeout)
                items.append(item)
            except queue.Empty:
                break

        return items

    def get_request_queue_size(self) -> int:
        return self.request_queue.qsize()

    def get_request_queue(self) -> queue.Queue[RequestQueueItem]:
        return self.request_queue

    def calculate_queue_latency(self, request_items: List[RequestQueueItem],
                                now: float) -> float:
        if not self.enable_iter_perf_stats:
            return 0.0

        total_latency = 0.0

        for req_item in request_items:
            # Handle parent request
            if req_item.id in self.start_times:
                total_latency += now - self.start_times.pop(req_item.id)

            # Handle child requests
            if req_item.child_req_ids:
                for child_id in req_item.child_req_ids:
                    if child_id in self.start_times:
                        total_latency += now - self.start_times.pop(child_id)

        return total_latency
