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

import asyncio
import time
from queue import Queue
from threading import Event
from typing import AsyncGenerator, Optional

from .._utils import nvtx_range_debug
from ..llmapi.utils import logger_debug
from ..logger import logger
from .request import GenerationRequest
from .rpc import RPCServer


class RpcWorkerMixin:
    """Mixin for workers that serve RPC requests.

    Provides:
    - RPC server initialization
    - Response queue management
    - Async response fetching methods
    - Shutdown logic for RPC components

    The inheriting class should call init_rpc_worker() in its __init__.
    """

    # Default number of RPC server workers
    # This can be overridden by setting num_workers in the inheriting class
    NUM_WORKERS = 6

    def init_rpc_worker(self, rank: int, rpc_addr: Optional[str], hmac_key: bytes):
        if rpc_addr is None:
            raise RuntimeError("RPC mode enabled but no rpc_addr provided to worker")

        self.hmac_key = hmac_key
        self.rank = rank
        self.shutdown_event = Event()
        self._response_queue = Queue()
        self.set_result_queue(self._response_queue)

        self.rpc_server = None
        self.rpc_addr = rpc_addr
        self._postproc_pool = None
        self._postproc_input_queues = None
        self._postproc_collector = None
        self._postproc_collector_thread = None
        self._postproc_futures = []

    def init_postproc_workers(self) -> None:
        """Spawn local PostprocWorker processes feeding the RPC response stream.

        The classic (MPI proxy) path gives each PostprocWorker a dedicated
        push lane straight to the frontend process. Under RPC/Ray
        orchestration no such lane exists: every record must travel the single
        RPC response stream. Instead of teaching PostprocWorker a new
        transport, its push pipe is pointed at a local collector socket whose
        consumer thread enqueues the already-final ``PostprocWorker.Output``
        batches into ``_response_queue`` — the same queue ``fetch_responses``
        drains — so the stream, the proxy demux, and the client stay on the
        code they already run for the classic postproc path.

        Call on the response-producing rank only (rank 0), after
        ``init_rpc_worker``.
        """
        import threading
        from concurrent.futures import ProcessPoolExecutor

        import zmq

        from .ipc import IpcQueue
        from .postproc_worker import PostprocWorker, postproc_worker_main

        num = self.postproc_config.num_postprocess_workers
        assert num > 0

        self._postproc_input_queues = [
            IpcQueue(is_server=True, name=f"rpc_worker_postproc_input_{i}") for i in range(num)
        ]
        self._postproc_collector = IpcQueue(
            is_server=True, socket_type=zmq.PULL, name="rpc_worker_postproc_collector"
        )
        # Both the result_queue (RPC stream feed) and the postproc input
        # queues are live on this worker — see set_postproc_queues docstring.
        self.set_postproc_queues(self._postproc_input_queues, coexist_with_result_queue=True)

        # fork (default), matching the classic path. spawn is NOT usable
        # here: the spawn bootstrap re-imports the Ray worker's __main__,
        # which deadlocks inside a Ray actor (verified empirically).
        self._postproc_pool = ProcessPoolExecutor(max_workers=num)

        def _on_postproc_worker_done(fut) -> None:
            # ProcessPoolExecutor stores task exceptions on the Future and
            # never raises them in the parent. A postproc child that dies
            # outside per-request handling would otherwise fail silently and
            # leave its requests pending forever; surface the exception on the
            # worker's background-error path so the next response turns into
            # an ErrorResponse for the client.
            exc = fut.exception()
            if exc is not None and not self.shutdown_event.is_set():
                logger.error(f"PostprocWorker process died: {exc}")
                self._error_queue.put(exc)

        for i in range(num):
            fut = self._postproc_pool.submit(
                postproc_worker_main,
                self._postproc_input_queues[i].address,
                [self._postproc_collector.address],
                self.postproc_config.postprocess_tokenizer_dir,
                PostprocWorker.default_record_creator,
                self.postproc_config.post_processor_hook,
            )
            fut.add_done_callback(_on_postproc_worker_done)
            self._postproc_futures.append(fut)

        def _collect_postproc_outputs() -> None:
            while not self.shutdown_event.is_set():
                batch = self._postproc_collector.get()
                if batch is None:
                    break
                # fetch_responses drains _response_queue batch-wise; Output
                # batches ride the RPC stream exactly like final responses.
                self._response_queue.put(batch)

        self._postproc_collector_thread = threading.Thread(
            target=_collect_postproc_outputs, name="rpc_worker_postproc_collector", daemon=True
        )
        self._postproc_collector_thread.start()

    def shutdown_postproc_workers(self) -> None:
        """Best-effort teardown of the local postproc pool (idempotent)."""
        if self._postproc_input_queues:
            for q in self._postproc_input_queues:
                try:
                    q.put(None)  # PostprocWorker mainloop exits on None
                except Exception:
                    pass
        if self._postproc_pool is not None:
            self._postproc_pool.shutdown(wait=False)
            self._postproc_pool = None

    def start_rpc_server(self):
        if self.rank == 0:
            # Use num_workers if set on the instance, otherwise use class default
            num_workers = getattr(self, "num_workers", RpcWorkerMixin.NUM_WORKERS)
            self.rpc_server = RPCServer(self, num_workers=num_workers, hmac_key=self.hmac_key)
            self.rpc_server.bind(self.rpc_addr)
            self.rpc_server.start()

    def submit(self, request: GenerationRequest):
        """Submits a request to the worker."""
        with nvtx_range_debug("RpcWorker.submit", color="blue", category="Worker"):
            logger_debug(f"[worker] Submitting request {request.id}", color="green")
            result = super().submit(request)
            logger_debug(f"[worker] Submitted request {request.id}", color="green")
            return result

    def fetch_responses(self, timeout: Optional[float] = None) -> list:
        """Fetch responses from the response queue (blocking)."""
        logger_debug(f"[worker] RpcWorker {self.rank} is fetching responses", color="yellow")
        with nvtx_range_debug("RpcWorker.fetch_responses", color="orange", category="Worker"):
            # NOTE: This is a blocking call, it will wait for the responses to be available.
            # Use the configured fetch timeout if no timeout is provided
            actual_timeout = (
                timeout if timeout is not None else getattr(self, "_fetch_timeout", 0.1)
            )
            responses = super().await_responses(timeout=actual_timeout)
            responses = self._await_response_helper.process_and_handle_responses(responses)
            logger_debug(f"[worker] Fetched {len(responses)} responses", color="green")

        qsize = self._response_queue.qsize()
        logger_debug(f"[worker] RpcWorker returning {qsize} responses", color="yellow")

        all_responses = []
        for _ in range(qsize):
            # The queue contains batches of responses, so extend the list
            all_responses.extend(self._response_queue.get())
        return all_responses

    async def fetch_responses_async(self, timeout: Optional[float] = None) -> list:
        """Async version of fetch_responses using asyncio.to_thread."""
        # Use asyncio.to_thread to avoid blocking the event loop
        # This is similar to fetch_stats_async and fetch_kv_cache_events_async
        responses = await asyncio.to_thread(self.fetch_responses, timeout=timeout)
        return responses

    async def fetch_responses_loop_async(self) -> AsyncGenerator[list, None]:
        """Stream responses in a loop until shutdown."""
        while not self.shutdown_event.is_set():
            responses = await self.fetch_responses_async()
            if responses:  # Only yield if there are actual responses
                logger_debug(
                    f"[worker] RpcWorker {self.rank} is yielding responses: {responses}",
                    color="yellow",
                )
                yield responses  # batching the responses to opt IPC performance
            else:
                # Small delay to prevent busy waiting when no responses
                await asyncio.sleep(0)
        logger_debug(
            f"[worker] RpcWorker {self.rank} quitting fetch_responses_loop_async", color="yellow"
        )

    async def fetch_stats_wait_async(self, timeout: Optional[float] = None) -> list:
        """Poll for stats until available or timeout.

        Args:
            timeout: Max wait time in seconds. If None, fetch once without waiting.
        """
        logger_debug(
            f"[worker] RpcWorker {self.rank} is fetching stats with timeout {timeout}",
            color="yellow",
        )
        start = time.time()
        while True:
            stats = await asyncio.to_thread(self.fetch_stats)
            if stats or timeout is None:
                break
            if (time.time() - start) >= timeout:
                break
            await asyncio.sleep(0.1)
        return [self._stats_serializer(s) for s in stats]

    async def fetch_kv_cache_events_wait_async(self, timeout: Optional[float] = None) -> list:
        """Poll for KV cache events until available or timeout.

        Args:
            timeout: Max wait time in seconds. If None, fetch once without waiting.
        """
        start = time.time()
        while True:
            events = await asyncio.to_thread(self.fetch_kv_cache_events)
            if events or timeout is None:
                break
            if (time.time() - start) >= timeout:
                break
            await asyncio.sleep(0.1)
        return [self._kv_cache_events_serializer(e) for e in events]

    async def fetch_stats_async(self, timeout: Optional[float] = None) -> list:
        """Async version of fetch_stats using asyncio.to_thread.

        This method is exposed via RPC and can be called directly by the proxy.
        Returns serialized stats (JSON strings) that can be sent over RPC.
        """
        stats = await asyncio.to_thread(self.fetch_stats)
        # Serialize stats before sending over RPC (IterationStats objects are not picklable)
        return [self._stats_serializer(s) for s in stats]

    async def fetch_kv_cache_capacity_async(self) -> str:
        """Async version of fetch_kv_cache_capacity using asyncio.to_thread."""
        capacity = await asyncio.to_thread(self.fetch_kv_cache_capacity)
        return self._kv_cache_capacity_serializer(capacity)

    async def fetch_kv_cache_events_async(self, timeout: Optional[float] = None) -> list:
        """Async version of fetch_kv_cache_events using asyncio.to_thread.

        This method is exposed via RPC and can be called directly by the proxy.
        Returns serialized events (JSON strings) that can be sent over RPC.
        """
        events = await asyncio.to_thread(self.fetch_kv_cache_events)
        # Serialize events before sending over RPC
        return [self._kv_cache_events_serializer(e) for e in events]
