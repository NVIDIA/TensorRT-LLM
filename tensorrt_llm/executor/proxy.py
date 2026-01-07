import atexit
import concurrent.futures
import json
import os
import threading
import weakref
from typing import Dict, List, Optional

import torch
import zmq
import zmq.asyncio

from tensorrt_llm.logger import logger

from .._utils import customized_gc_thresholds, mpi_rank, nvtx_range_debug
from ..llmapi.mpi_session import (MpiCommSession, MpiPoolSession, MpiSession,
                                  RemoteMpiCommSessionClient)
from ..llmapi.tracer import enable_llm_tracer, get_tracer, global_tracer
from ..llmapi.utils import (AsyncQueue, ManagedThread, _SyncQueue,
                            enable_llm_debug, logger_debug, print_colored)
from .executor import GenerationExecutor
from .ipc import FusedIpcQueue, IpcQueue
from .postproc_worker import PostprocWorker, PostprocWorkerConfig
from .request import CancellingRequest, GenerationRequest
from .result import GenerationResult, IterationResult
from .rpc import RPCClient
from .rpc.rpc_common import get_unique_ipc_addr
from .utils import (ErrorResponse, WorkerCommIpcAddrs, create_mpi_comm_session,
                    get_spawn_proxy_process_env, is_llm_response,
                    print_alive_threads)
from .worker import GenerationExecutorWorker, worker_main

__all__ = [
    "GenerationExecutorProxy",
]


class GenerationExecutorProxy(GenerationExecutor):
    READY_SIGNAL = b"READY"

    def __init__(
        self,
        worker_kwargs: dict,
        model_world_size: int = 1,
        mpi_session: Optional[MpiSession] = None,
        *,
        worker_cls: type = GenerationExecutorWorker,
        postproc_worker_config: Optional[PostprocWorkerConfig] = None,
        is_llm_executor: Optional[bool] = None,
    ) -> None:
        postproc_worker_config = postproc_worker_config or PostprocWorkerConfig(
        )
        super().__init__(
            num_postprocess_workers=postproc_worker_config.
            num_postprocess_workers,
            postprocess_tokenizer_dir=postproc_worker_config.
            postprocess_tokenizer_dir,
            is_llm_executor=is_llm_executor,
        )

        self.workers_started = False
        self.worker_cls = worker_cls

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

        if isinstance(self.mpi_session,
                      (MpiCommSession, RemoteMpiCommSessionClient)):
            print_colored(
                f"rank {mpi_rank()} using MpiCommSession to bind to external MPI processes\n",
                "yellow")
        else:
            print_colored(
                f"rank {mpi_rank()} using MpiPoolSession to spawn MPI processes\n",
                "yellow")

        self._results: Dict[int, GenerationResult] = {}

        self.model_world_size = model_world_size

        self.garbage_collection_gen0_threshold = worker_kwargs[
            "llm_args"].garbage_collection_gen0_threshold if worker_kwargs.get(
                "llm_args", None) is not None else None

        # Generate RPC address and key for stats RPC
        self.rpc_addr = get_unique_ipc_addr()
        self.hmac_key = os.urandom(32)

        worker_kwargs = dict(**worker_kwargs,
                             worker_queues=self._setup_queues(),
                             postproc_worker_config=postproc_worker_config,
                             is_llm_executor=False,
                             rpc_addr=self.rpc_addr,
                             hmac_key=self.hmac_key)

        if "log_level" not in worker_kwargs:
            worker_kwargs["log_level"] = logger.level

        self.dispatch_result_thread: Optional[ManagedThread] = None
        self.rpc_client: Optional[RPCClient] = None
        self._start_executor_workers(worker_kwargs)

        # Create RPC client after workers are started (worker starts RPC server)
        self.rpc_client = RPCClient(self.rpc_addr, hmac_key=self.hmac_key)

        # MPI registers its joiner using threading._register_atexit if possible.
        # These functions run before atexit.register, so to avoid deadlock,
        # we have to notify workers to exit before MPI starts to wait them.
        try:
            threading._register_atexit(  # type: ignore[attr-defined]
                self.pre_shutdown)
        except AttributeError:
            atexit.register(self.pre_shutdown)

    def _setup_queues(self) -> WorkerCommIpcAddrs:

        self.request_queue = IpcQueue(is_server=True,
                                      name="proxy_request_queue")
        self.worker_init_status_queue = IpcQueue(
            is_server=True,
            socket_type=zmq.ROUTER,
            name="worker_init_status_queue")
        # TODO[chunweiy]: Unify IpcQueue and FusedIpcQueue
        # Use PULL mode when enable_postprocess_parallel as there are
        # multiple senders from multiple processes.
        self.result_queue = FusedIpcQueue(
            is_server=True,
            fuse_message=False,
            socket_type=zmq.PULL
            if self.enable_postprocess_parallel else zmq.PAIR,
            name="proxy_result_queue")
        # Stats and KV events are now fetched via RPC, not IPC queues.
        return WorkerCommIpcAddrs(
            request_queue_addr=self.request_queue.address,
            worker_init_status_queue_addr=self.worker_init_status_queue.address,
            result_queue_addr=self.result_queue.address,
        )

    def abort_request(self, request_id: int) -> None:
        ''' Abort a request by sending a cancelling request to the request queue.

        Args:
            request_id (int): The id of the request to abort.
        '''
        # NOTE, it just sends a cancelling request to the request queue, but it
        # may take a while for the request to be cancelled in the worker and
        # send back a finished result.
        self.request_queue.put(CancellingRequest(request_id))

    def dispatch_result_task(self) -> bool:
        # TODO[chunweiy]: convert the dispatch_result_task to async, that should
        # benefit from zmq.asyncio.Context
        with customized_gc_thresholds(self.garbage_collection_gen0_threshold):
            if (res := self.result_queue.get()) is None:
                return False  # shutdown the thread

        async_queues = []
        event_loop = None

        def process_res(res):
            client_id = res.client_id
            nonlocal event_loop
            nonlocal async_queues

            queue = self._results[client_id].queue
            if isinstance(queue, _SyncQueue):
                queue.put_nowait(res)
                async_queues.append(queue)
                # all the loops are identical
                event_loop = event_loop or queue.loop
            else:
                queue.put(res)

            # FIXME: Add type annotations and make 'res' type more homogeneous (e.g.
            #        include PostprocWorker.Output in is_llm_response and unify is_final APIs).
            if (is_llm_response(res) and res.result.is_final) or isinstance(
                    res,
                    ErrorResponse) or (isinstance(res, PostprocWorker.Output)
                                       and res.is_final):
                self._results.pop(client_id)

        res = res if isinstance(res, list) else [res]

        for i in res:
            global_tracer().log_instant("IPC.get")
            if i is None:
                return False
            process_res(i)

        if async_queues:
            try:
                _SyncQueue.notify_many(event_loop, async_queues)
            except AsyncQueue.EventLoopShutdownError:
                logger.warning(
                    "proxy.py: EventLoopShutdownError because event loop is not running"
                )

        return True  # success

    # NOTE: _iteration_result_task, dispatch_stats_task, and dispatch_kv_cache_events_task
    # have been removed as stats and kv_events are now fetched via RPC directly.

    def _start_dispatch_threads(self):
        if self.dispatch_result_thread is None:

            self.dispatch_result_thread = ManagedThread(
                weakref.WeakMethod(self.dispatch_result_task),
                error_queue=self._error_queue,
                name="proxy_dispatch_result_thread")

            self.dispatch_result_thread.start()

        self._handle_background_error()

    def _start_executor_workers(self, worker_kwargs):

        self_ref = weakref.ref(self)

        def mpi_done_callback(future: concurrent.futures.Future):
            # This is called when the MPI worker is done, so future.exception()
            # will not block.
            if future.exception() is not None:
                if self_ := self_ref():
                    self_._error_queue.put_nowait(future.exception())

        tracer_init_kwargs = get_tracer().init_kwargs if enable_llm_tracer(
        ) else None
        from tensorrt_llm._torch.models.modeling_auto import MODEL_CLASS_MAPPING
        torch.cuda.Stream()
        self.mpi_futures = self.mpi_session.submit(
            worker_main,
            **worker_kwargs,
            worker_cls=self.worker_cls,
            tracer_init_kwargs=tracer_init_kwargs,
            _torch_model_class_mapping=MODEL_CLASS_MAPPING,
            ready_signal=GenerationExecutorProxy.READY_SIGNAL,
        )
        for fut in self.mpi_futures:
            fut.add_done_callback(mpi_done_callback)

        self.workers_started = True

        while True:
            if self.worker_init_status_queue.poll(1):
                ready_signal, error_trace = self.worker_init_status_queue.get()
                # Send ACK to the worker
                self.worker_init_status_queue.put("ACK")
                logger.info("get signal from executor worker")
                break
            if any(fut.done() for fut in self.mpi_futures):
                logger.error("Executor worker died during initialization.")
                raise RuntimeError("Executor worker died during initialization")
            self._handle_background_error()

        if ready_signal != GenerationExecutorProxy.READY_SIGNAL:
            logger.error(f"Executor worker initialization error: {error_trace}")
            self.mpi_session.shutdown_abort(reason=ready_signal)
            raise RuntimeError(
                "Executor worker returned error") from ready_signal

    def _abort_all_requests(self):
        # The results can be finished during this loop, so self._results may be changed.
        for result in list(self._results.values()):
            result.abort()

    def pre_shutdown(self):
        if not self.workers_started:
            return
        logger_debug('Proxy.pre_shutdown...\n', "yellow")

        if self.doing_shutdown:
            return
        else:
            self.doing_shutdown = True

        self._abort_all_requests()

        # notify the workers to quit
        if all(not f.done() for f in self.mpi_futures):
            self.request_queue.put_noblock(None, retry=4)

    def shutdown(self):
        if not self.workers_started:
            return

        if not self.doing_shutdown:
            self.pre_shutdown()

        logger_debug('Proxy.shutdown...\n', "yellow")

        for f in self.mpi_futures:
            try:
                f.result()
            except:
                # The errors are already captured in mpi_done_callback, ignored
                # here
                pass

        # step2: notify the background threads to quit
        if self.dispatch_result_thread is not None and self.dispatch_result_thread.is_alive(
        ):
            self.dispatch_result_thread.stop()
            self.dispatch_result_thread.join()

        # step3: finish all remaining work

        # close the RPC client
        if self.rpc_client is not None:
            self.rpc_client.close()
            self.rpc_client = None

        # close all the sockets
        self.request_queue.close()
        self.worker_init_status_queue.close()
        self.result_queue.close()

        self.workers_started = False
        self.mpi_session.shutdown()

        # Process the errors in-case error during shutting down the threads
        self._handle_background_error()

        if enable_llm_debug():
            print_alive_threads()

    def submit(self, request: GenerationRequest) -> GenerationResult:
        """
            Low-level API to the executor. Return a "future" GenerationResult
            which can be waited.
            Forwards the request to the workers through the request queue.
        """

        self._start_dispatch_threads()

        request.set_id(self._get_next_client_id())
        logprob_params = self._get_logprob_params(request)

        result = GenerationResult(
            request,
            background_error_handler=self._handle_background_error,
            executor=self,
            disaggregated_params=request.disaggregated_params,
            logprob_params=logprob_params)
        self._results[request.id] = result

        with nvtx_range_debug("request_queue.put"):
            self.request_queue.put(request)

        self._handle_background_error()

        return result

    def get_stats(self, timeout: float) -> List[dict]:
        """Get iteration statistics from the runtime via RPC.

        Args:
            timeout (float): Max wait time in seconds for the RPC call.

        Returns:
            List[dict]: A list of runtime stats as dict.
        """
        if self.rpc_client is None:
            logger.warning("RPC client not initialized, cannot get stats")
            return []

        stats = self.rpc_client.fetch_stats_wait_async(timeout=timeout).remote()
        return [json.loads(s) if isinstance(s, str) else s for s in stats]

    def aget_stats(self, timeout: float) -> IterationResult:
        """Get iteration statistics from the runtime via RPC (async).

        Args:
            timeout (float): Max wait time in seconds for the RPC call.

        Returns:
            IterationResult: An async iterable object containing runtime stats.
        """
        # Initialize iteration result if needed
        self._maybe_initialize_iteration_results()

        if self._iter_stats_result is None:
            logger.warning("Iteration statistics are not available yet.")
            from .executor import empty_async_iterable
            return empty_async_iterable()

        # Fetch stats via RPC and populate the result
        try:
            stats = self.rpc_client.fetch_stats_wait_async(
                timeout=timeout).remote()
        except Exception as e:
            logger.debug(f"Error fetching stats via RPC: {e}")
            stats = []

        for stat in stats:
            self._iter_stats_result.queue.put(stat)

        self._iter_stats_result.set_timeout(timeout)
        return self._iter_stats_result

    def get_kv_events(self, timeout: float) -> List[dict]:
        """Get iteration KV events from the runtime via RPC.

        Args:
            timeout (float): Max wait time in seconds for the RPC call.

        Returns:
            List[dict]: A list of runtime events as dict.
        """
        if self.rpc_client is None:
            logger.warning("RPC client not initialized, cannot get kv events")
            return []

        try:
            events = self.rpc_client.fetch_kv_cache_events_wait_async(
                timeout=timeout).remote()
            return [json.loads(e) if isinstance(e, str) else e for e in events]
        except Exception as e:
            logger.error(f"Error fetching kv events via RPC: {e}")
            return []

    def aget_kv_events(self, timeout: float) -> IterationResult:
        """Get iteration KV events from the runtime via RPC (async).

        Args:
            timeout (float): Max wait time in seconds for the RPC call.

        Returns:
            IterationResult: An async iterable object containing runtime events.
        """
        # Initialize iteration result if needed
        self._maybe_initialize_iteration_results()

        if self._iter_kv_events_result is None:
            from .executor import empty_async_iterable
            return empty_async_iterable()

        # Fetch kv events via RPC and populate the result
        try:
            events = self.rpc_client.fetch_kv_cache_events_wait_async(
                timeout=timeout).remote()
        except Exception as e:
            logger.debug(f"Error fetching kv events via RPC: {e}")
            events = []

        for event in events:
            self._iter_kv_events_result.queue.put(event)

        self._iter_kv_events_result.set_timeout(timeout)
        return self._iter_kv_events_result

    def __del__(self):
        self.shutdown()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.shutdown()
        return False  # propagate the exception
