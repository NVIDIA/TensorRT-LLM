import asyncio
import atexit
import json
import threading
from typing import Optional

from ..llmapi.mpi_session import MpiPoolSession, MpiSession
from ..llmapi.tracer import global_tracer
from ..llmapi.utils import AsyncQueue, _SyncQueue, logger_debug
from ..logger import logger
from .executor import GenerationExecutor
from .postproc_worker import PostprocWorkerConfig
from .request import GenerationRequest
from .result import GenerationResult
from .rpc import RPCClient
from .rpc.rpc_common import get_unique_ipc_addr
from .rpc_worker import RpcWorker
from .utils import (ErrorResponse, create_mpi_comm_session,
                    get_spawn_proxy_process_env, is_llm_response)


class GenerationExecutorRpcProxy(GenerationExecutor):
    # NOTE: this is a global counter for the number of instances of this class
    INSTANCE_COUNTER = 0

    def __init__(
        self,
        worker_kwargs: dict,
        model_world_size: int = 1,
        mpi_session: Optional[MpiSession] = None,
        *,
        postproc_worker_config: Optional[PostprocWorkerConfig] = None,
        is_llm_executor: Optional[bool] = None,
    ):
        """
        Args:
            worker_kwargs: kwargs for the rpc worker
            model_world_size: the world size of the model
            mpi_session: the mpi session to use
            postproc_worker_config: the postproc worker config
            is_llm_executor: whether this is an llm executor
        """
        GenerationExecutorRpcProxy.INSTANCE_COUNTER += 1
        self.rpc_addr = get_unique_ipc_addr()
        self.rpc_client = RPCClient(self.rpc_addr)

        postproc_worker_config = postproc_worker_config or PostprocWorkerConfig(
        )

        super().__init__(
            num_postprocess_workers=postproc_worker_config.
            num_postprocess_workers,
            postprocess_tokenizer_dir=postproc_worker_config.
            postprocess_tokenizer_dir,
            is_llm_executor=is_llm_executor,
        )

        self._results = {}

        self._create_mpi_session(model_world_size, mpi_session)

        self._shutdown_event = threading.Event()
        self.worker_kwargs = worker_kwargs

        self.main_loop_task_obj = None
        self.main_loop = None

        self.launch_workers()

        # Invoke model creation on the remote
        # TBD: Move model creation to the mpi task, or left in RPC?
        self.setup_engine_remote()

        # Setup main loop after engine is ready
        self.setup_mainloop()

    def launch_workers(self):
        logger.debug(f"Launching workers")
        assert self.mpi_session is not None
        self.mpi_session.submit(RpcWorker.main_task,
                                rpc_addr=self.rpc_addr,
                                **self.worker_kwargs)

    async def _generic_fetch_loop_async(self, fetch_method_name: str,
                                        handler_method, method_name: str):
        """Generic method for fetching data in a loop from RPC worker.

        Args:
            fetch_method_name: Name of the RPC client method to call
            handler_method: The handler method to call with the fetched data
            method_name: Name of the method for logging
        """
        try:
            fetch_method = getattr(self.rpc_client, fetch_method_name)
            async for data in fetch_method().remote_streaming():
                if self._shutdown_event.is_set():
                    return
                handler_method(data)
        except asyncio.CancelledError:
            logger.debug(f"{method_name} task cancelled")
        except Exception as e:
            logger.error(f"Error in {method_name}: {e}")
            raise

    async def _fetch_responses_loop_async(self):
        await self._generic_fetch_loop_async(
            fetch_method_name="fetch_responses_loop_async",
            handler_method=self.handle_responses,
            method_name="_fetch_responses_loop_async")

    async def _fetch_stats_loop_async(self):
        await self._generic_fetch_loop_async(
            fetch_method_name="fetch_stats_loop_async",
            handler_method=self.handle_stats,
            method_name="_fetch_stats_loop_async")

    async def _fetch_kv_cache_events_loop_async(self):
        await self._generic_fetch_loop_async(
            fetch_method_name="fetch_kv_cache_events_loop_async",
            handler_method=self.handle_kv_cache_events,
            method_name="_fetch_kv_cache_events_loop_async")

    def setup_mainloop(self):

        async def main_loop_task():
            tasks = [
                self._fetch_responses_loop_async(),
                self._fetch_stats_loop_async(),
                self._fetch_kv_cache_events_loop_async(),
            ]
            # Only add kv_cache_events loop if it's enabled
            if self._iter_kv_events_result:
                tasks.append(self._fetch_kv_cache_events_loop_async())
            await asyncio.gather(*tasks)

        def _run_main_loop_task():
            """Local method to run the main loop task."""
            self.main_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.main_loop)

            self.main_loop_task_obj = self.main_loop.create_task(
                main_loop_task())
            try:
                self.main_loop.run_until_complete(self.main_loop_task_obj)
            except asyncio.CancelledError:
                pass  # Task cancellation is expected during shutdown
            finally:
                self.main_loop.close()

        self.main_loop_thread = threading.Thread(target=_run_main_loop_task,
                                                 daemon=True)
        self.main_loop_thread.start()
        atexit.register(self.shutdown)

    def handle_responses(self, responses: list[GenerationResult]) -> bool:
        async_queues = []
        event_loop = None

        def process_res(res: list):
            for r in res:
                client_id = r.client_id
                nonlocal event_loop
                nonlocal async_queues

                if client_id not in self._results:
                    logger.warning(
                        f"Received response for unknown client_id: {client_id}")
                    continue

                queue = self._results[client_id].queue
                if isinstance(queue, _SyncQueue):
                    queue.put_nowait(r)
                    async_queues.append(queue)
                    # all the loops are identical
                    event_loop = event_loop or queue.loop
                else:
                    queue.put(r)

                if (is_llm_response(r) and r.result.is_final) or isinstance(
                        r, ErrorResponse):
                    self._results.pop(client_id)

        # Handle the case where responses might not be a list of lists
        if responses and not isinstance(responses[0], list):
            # If responses is a flat list, wrap it
            responses = [responses]

        for res in responses:
            global_tracer().log_instant("RPC.get")
            process_res(res)

        if async_queues:
            _SyncQueue.notify_many(event_loop, async_queues)

    def _handle_iteration_data(self, data, result_singleton, data_type: str):
        """Generic method to handle iteration data received from RPC worker.

        Args:
            data: Data from the RPC worker (can be dict, str, or list)
            result_singleton: The iteration result singleton to put data into
            data_type: Type of data for logging (e.g., "stats", "kv_cache_events")
        """
        # Make sure we have initialized the iteration results
        self._maybe_initialize_iteration_results()

        if not result_singleton:
            logger.debug(
                f"Skipping {data_type} handling while result_singleton=None")
            return

        # Get the queue from the result singleton
        queue = result_singleton.queue
        async_queues = []

        # Clear old data if queue is full (similar to _iteration_result_task)
        while queue.full():
            queue.get()

        try:
            # Handle different types of data
            if isinstance(data, str):
                # Already JSON serialized
                data_json = data
            elif isinstance(data, list):
                # Skip empty lists to avoid putting nothing in the queue
                if not data:
                    logger.debug(
                        f"rpc_proxy.py: Skipping empty {data_type} list")
                    return

                # Handle list of data (multiple iterations)
                for item in data:
                    if isinstance(item, str):
                        item_json = item
                    else:
                        item_json = json.dumps(item)

                    if isinstance(queue, _SyncQueue):
                        queue.put_nowait(item_json)
                        async_queues.append(queue)
                    else:
                        queue.put(item_json)

                if async_queues:
                    _SyncQueue.notify_many(queue.loop, async_queues)
                return
            else:
                # Convert dict/other to JSON string as expected by IterationResult
                data_json = json.dumps(data)

            if isinstance(queue, _SyncQueue):
                queue.put_nowait(data_json)
                async_queues.append(queue)
            else:
                queue.put(data_json)

            if async_queues:
                _SyncQueue.notify_many(queue.loop, async_queues)

        except AsyncQueue.EventLoopShutdownError:
            # This happens when the event loop is already closed
            logger.debug(
                f"rpc_proxy.py: EventLoopShutdownError in handle_{data_type}")
        except Exception as e:
            logger.error(f"rpc_proxy.py: Error in handle_{data_type}: {e}")
            raise e

    def handle_stats(self, stats):
        """Handle stats received from RPC worker and put them into the stats result queue.

        Args:
            stats: Statistics data from the RPC worker (can be dict, str, or list)
        """
        self._handle_iteration_data(stats, self._iter_stats_result, "stats")

    def handle_kv_cache_events(self, events):
        """Handle KV cache events received from RPC worker and put them into the events result queue.

        Args:
            events: KV cache events data from the RPC worker (can be dict, str, or list)
        """
        self._handle_iteration_data(events, self._iter_kv_events_result,
                                    "kv_cache_events")

    def submit(self, request: GenerationRequest) -> GenerationResult:
        request.set_id(self._get_next_client_id())
        logprob_params = self._get_logprob_params(request)

        # submit is a fire-and-forget operation, don't need to wait for response
        self.rpc_client.submit(request).remote(need_response=False)

        result = GenerationResult(
            request,
            background_error_handler=self._handle_background_error,
            executor=self,
            disaggregated_params=request.disaggregated_params,
            logprob_params=logprob_params)
        self._results[request.id] = result

        return result

    def fetch_stats_remote(self):
        return self.rpc_client.fetch_stats().remote()

    def setup_engine_remote(self):
        return self.rpc_client.setup_engine().remote(need_response=True)

    def shutdown_remote(self):
        logger_debug(f"Shutting down rpc remote", color="yellow")
        self.rpc_client.shutdown().remote()

    def abort_request(self, request_id: int) -> None:
        return self.rpc_client.abort_request(request_id).remote()

    def shutdown(self):
        if self._shutdown_event.is_set():
            return
        self._shutdown_event.set()
        logger_debug(f"Shutting down GenerationExecutorRpcProxy",
                     color="yellow")

        # 1. shutdown the rpc server (PyExecutor Rank 0 + RPC server)
        self.shutdown_remote()

        # 2. stop the main loop, so that no new rpc requests
        if self.main_loop and self.main_loop_task_obj:
            logger_debug("Cancelling main loop task.", color="yellow")
            # The cancel() is thread-safe
            try:
                self.main_loop.call_soon_threadsafe(
                    self.main_loop_task_obj.cancel)
            except Exception as e:
                logger_debug(f"Error cancelling main loop task: {e}",
                             color="yellow")

        self.main_loop_thread.join()

        # 3. shutdown the mpi session, this should wait until all the PyExecutor
        # processes are shutdown
        if self.mpi_session is not None:
            logger_debug(f"Shutting down mpi session", color="yellow")
            self.mpi_session.shutdown()
            logger_debug(f"Mpi session shutdown", color="yellow")
            self.mpi_session = None

        self.rpc_client.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.shutdown()

    def _create_mpi_session(self, model_world_size: int,
                            mpi_session: Optional[MpiSession]):
        mpi_process_pre_spawned: bool = get_spawn_proxy_process_env()
        if mpi_session is None:
            if mpi_process_pre_spawned:
                logger_debug('create comm session ...\n', "yellow")
                self.mpi_session = create_mpi_comm_session(model_world_size)
            else:
                logger_debug('create pool session ...\n', "yellow")
                self.mpi_session = MpiPoolSession(n_workers=model_world_size)
        else:
            logger_debug('using external mpi session ...\n', "yellow")
            self.mpi_session = mpi_session
