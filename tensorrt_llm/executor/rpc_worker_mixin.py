import asyncio
import time
from queue import Queue
from threading import Event
from typing import AsyncGenerator, Optional

from .._utils import nvtx_range_debug
from ..llmapi.utils import logger_debug
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

    def init_rpc_worker(self, rank: int, rpc_addr: Optional[str], hmac_key: Optional[bytes] = None):
        if rpc_addr is None:
            raise RuntimeError("RPC mode enabled but no rpc_addr provided to worker")

        self.hmac_key = hmac_key
        self.rank = rank
        self.shutdown_event = Event()
        self._response_queue = Queue()
        self.set_result_queue(self._response_queue)

        self.rpc_server = None
        self.rpc_addr = rpc_addr

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
            self._await_response_helper.responses_handler(responses)
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

    async def fetch_kv_cache_events_async(self, timeout: Optional[float] = None) -> list:
        """Async version of fetch_kv_cache_events using asyncio.to_thread.

        This method is exposed via RPC and can be called directly by the proxy.
        Returns serialized events (JSON strings) that can be sent over RPC.
        """
        events = await asyncio.to_thread(self.fetch_kv_cache_events)
        # Serialize events before sending over RPC
        return [self._kv_cache_events_serializer(e) for e in events]
