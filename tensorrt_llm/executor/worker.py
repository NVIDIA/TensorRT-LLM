import gc
import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from queue import Queue
from typing import Callable, List, Optional, Union

import zmq

from tensorrt_llm.logger import logger

from .._utils import mpi_comm, mpi_rank
from ..bindings import executor as tllm
from ..builder import Engine
from ..llmapi.llm_args import BaseLlmArgs
from ..llmapi.mpi_session import set_mpi_session_cpp
from ..llmapi.tokenizer import TokenizerBase
from ..llmapi.tracer import VizTracer, set_global_tracer
from ..llmapi.utils import (AsyncQueue, ManagedThread, _SyncQueue,
                            clear_sched_affinity, logger_debug,
                            print_traceback_on_error)
from ..sampling_params import BatchedLogitsProcessor
from .base_worker import BaseWorker
from .executor import IterationResultQueue
from .ipc import FusedIpcQueue, IpcQueue
from .postproc_worker import (PostprocWorker, PostprocWorkerConfig,
                              postproc_worker_main)
from .request import CancellingRequest, GenerationRequest
from .result import IterationResult
from .utils import (ErrorResponse, RequestError, WorkerCommIpcAddrs,
                    has_event_loop)

__all__ = [
    "GenerationExecutorWorker",
]


class GenerationExecutorWorker(BaseWorker):

    def __init__(
        self,
        engine: Union[Path, Engine],
        executor_config: Optional[tllm.ExecutorConfig] = None,
        batched_logits_processor: Optional[BatchedLogitsProcessor] = None,
        postproc_worker_config: Optional[PostprocWorkerConfig] = None,
        is_llm_executor: Optional[bool] = None,
        hf_model_dir: Optional[Path] = None,
        tokenizer: Optional[TokenizerBase] = None,
        llm_args: Optional[BaseLlmArgs] = None,
    ) -> None:
        super().__init__(
            engine=engine,
            executor_config=executor_config,
            batched_logits_processor=batched_logits_processor,
            postproc_worker_config=postproc_worker_config,
            is_llm_executor=is_llm_executor,
            hf_model_dir=hf_model_dir,
            tokenizer=tokenizer,
            llm_args=llm_args,
        )

        self.setup_engine()

        self.await_response_thread = ManagedThread(
            self.await_response_task,
            error_queue=self._error_queue,
            name="await_response_thread")

        self.dispatch_stats_thread = ManagedThread(
            self.dispatch_stats_task,
            error_queue=self._error_queue,
            name="dispatch_stats_thread")

        self.dispatch_kv_cache_events_thread = ManagedThread(
            self.dispatch_kv_cache_events_task,
            error_queue=self._error_queue,
            name="dispatch_kv_cache_events_thread")

    def _create_iteration_result_queue(self,
                                       it_result_queue: IterationResultQueue):
        if not it_result_queue.is_initialized:
            # not yet initialized
            it_result_queue.is_initialized = True
            if has_event_loop():
                _queue = AsyncQueue()
                it_result_queue.queue = _queue.sync_q
                it_result_queue.aqueue = _queue
            else:
                _queue = Queue()
                it_result_queue.queue = _queue
                it_result_queue.aqueue = None

    def start_thread(self, thread: ManagedThread):
        if self.engine.can_enqueue_requests() and not thread.is_alive():
            thread.start()

    def await_response_task(self) -> bool:
        return self._await_response_helper()

    def _iteration_result_task(self, it_result_queue: IterationResultQueue,
                               engine_get_result_api: Callable,
                               result_singleton: IterationResult,
                               result_serializer: Callable) -> bool:
        time.sleep(0.2)
        async_queues = []
        queue = result_singleton.queue if self._is_llm_executor and result_singleton else it_result_queue.queue
        try:
            for results in engine_get_result_api():
                res = result_serializer(results)
                if self._is_llm_executor and result_singleton:
                    # In this case, there's no ExecutorBindingProxy.
                    # Worker needs to take care of putting to result queue.
                    while queue.full():
                        queue.get()
                    if isinstance(queue, _SyncQueue):
                        queue.put_nowait(res)
                        async_queues.append(queue)
                    else:
                        queue.put(res)
                else:
                    # Send to ExecutorBindingProxy via IPC
                    queue.put(res)

            if async_queues:
                _SyncQueue.notify_many(queue.loop, async_queues)
        except AsyncQueue.EventLoopShutdownError:
            # This happens in the last results loop while the generate workflow is stopped.
            logger.debug("worker.py: EventLoopShutdownError")
        except Exception as e:
            logger.error(f"worker.py: Error in _iteration_result_task: {e}")
            raise e

        return True  # success

    def dispatch_stats_task(self) -> bool:
        return self._iteration_result_task(self.stats_queues, self.fetch_stats,
                                           self._iter_stats_result,
                                           self._stats_serializer)

    def dispatch_kv_cache_events_task(self) -> bool:
        if isinstance(self.engine, tllm.Executor):
            # Check if the engine has a kv cache event manager
            # If not, return an empty list for the events which will cause the thread to exit early.
            event_manager = self.engine.get_kv_cache_event_manager()
            if event_manager is None:
                events_api = lambda: [None]
            else:
                events_api = event_manager.get_latest_events
            return self._iteration_result_task(self.kv_events_queues,
                                               events_api,
                                               self._iter_kv_events_result,
                                               self._kv_cache_events_serializer)
        else:
            return self._iteration_result_task(
                self.kv_events_queues, self.engine.get_latest_kv_cache_events,
                self._iter_kv_events_result, self._kv_cache_events_serializer)

    def start(self):
        # create iteration result queues
        self._create_iteration_result_queue(self.stats_queues)
        self._create_iteration_result_queue(self.kv_events_queues)

        # start threads
        self.start_thread(self.await_response_thread)
        self.start_thread(self.dispatch_kv_cache_events_thread)
        if mpi_rank() == 0:
            self.start_thread(self.dispatch_stats_thread)

    def shutdown(self):

        if self.doing_shutdown:
            return
        else:
            self.doing_shutdown = True

        logger_debug(f'Worker {mpi_rank()} shutdown...\n', "yellow")

        if self.engine is not None:
            if self.engine.can_enqueue_requests():

                if self.await_response_thread.is_alive():
                    self.await_response_thread.stop()
                    self.await_response_thread.join()
                if self.dispatch_stats_thread.is_alive():
                    self.dispatch_stats_thread.stop()
                    self.dispatch_stats_thread.join()
                if self.dispatch_kv_cache_events_thread.is_alive():
                    self.dispatch_kv_cache_events_thread.stop()
                    self.dispatch_kv_cache_events_thread.join()

            self.engine.shutdown()
            self.engine = None

            if self.llm_args is not None:
                assert self._executor_config is None, "An empty executor_config is expected in shutdown when LLM arguments are defined."
                if (self.llm_args.backend == "pytorch"
                        and hasattr(self, "checkpoint_loader")
                        and self.checkpoint_loader is not None):
                    self.checkpoint_loader.cleanup()
                    self.checkpoint_loader = None
            else:
                if hasattr(
                        self._executor_config, "checkpoint_loader"
                ) and self._executor_config.checkpoint_loader is not None:
                    self._executor_config.checkpoint_loader.cleanup()
                    self._executor_config.checkpoint_loader = None

        # Check if there are any errors from the threads before shutdown.
        self._handle_background_error()

        logger_debug(f"Worker {mpi_rank()} shutdown done.\n", "yellow")

    def block_subordinates(self):
        if self.rank != 0:
            if isinstance(self.engine, tllm.Executor):
                self.shutdown()
                raise self.WorkerExit(
                    "block_subordinates() should be used in a `with GenerationExecutorWorker() as ...:` block"
                )
            from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor
            if isinstance(self.engine, PyExecutor):
                self.engine.wait_shutdown()


@print_traceback_on_error
def worker_main(
    engine: Path | Engine,
    worker_queues: WorkerCommIpcAddrs,
    log_level: str,
    executor_config: Optional[tllm.ExecutorConfig] = None,
    batched_logits_processor: Optional[BatchedLogitsProcessor] = None,
    worker_cls: type = GenerationExecutorWorker,
    tracer_init_kwargs: Optional[dict] = None,
    _torch_model_class_mapping: Optional[dict] = None,
    postproc_worker_config: Optional[PostprocWorkerConfig] = None,
    ready_signal: Optional[str] = None,
    is_llm_executor: Optional[
        bool] = True,  # whether it's the main executor instance
    hf_model_dir: Optional[Path] = None,
    tokenizer: Optional[TokenizerBase] = None,
    llm_args: Optional[BaseLlmArgs] = None,
) -> None:
    mpi_comm().barrier()
    logger_debug(f"Worker {mpi_rank()} entering worker_main...\n", "green")

    pid = os.getpid()
    cpus = os.sched_getaffinity(pid)
    if cpus:
        logger.warning(
            f"Found worker process {pid} was bound to {cpus}, this may harm "
            "performance.", )
        logger.warning(f"Will clear the cpu affinity")
        clear_sched_affinity(pid)

    result_queue: Optional[IpcQueue] = None
    result_queues: Optional[List[IpcQueue]] = None

    postproc_worker_config = postproc_worker_config or PostprocWorkerConfig()

    is_leader: bool = mpi_rank() == 0
    if tracer_init_kwargs is not None and is_leader:
        tracer = VizTracer(**tracer_init_kwargs)
        tracer.register_exit()
        tracer.start()
        set_global_tracer(tracer)

    if _torch_model_class_mapping is not None:
        from tensorrt_llm._torch.models.modeling_auto import MODEL_CLASS_MAPPING
        MODEL_CLASS_MAPPING.update(**_torch_model_class_mapping)

    set_mpi_session_cpp(mpi_comm())

    if is_leader:
        # Only set the log level for the leader process, the other processes will
        # inherit the log level from "TLLM_LOG_LEVEL" environment variable
        logger.set_level(log_level)
        request_queue = IpcQueue(worker_queues.request_queue_addr,
                                 is_server=False,
                                 name="worker_request_queue")
        worker_init_status_queue = IpcQueue(
            worker_queues.worker_init_status_queue_addr,
            is_server=False,
            socket_type=zmq.DEALER,
            name="worker_init_status_queue")
        mp_stats_queue = FusedIpcQueue(worker_queues.stats_queue_addr,
                                       is_server=False,
                                       fuse_message=True,
                                       name="worker_stats_queue")
        kv_cache_events_queue = FusedIpcQueue(
            worker_queues.kv_cache_events_queue_addr,
            is_server=False,
            fuse_message=False,
            name="worker_kv_cache_events_queue")

        if postproc_worker_config.enabled:
            # IPC queues for sending inputs to the postprocess parallel
            # processes, each one is a PAIR zmq socket
            result_queues = [
                FusedIpcQueue(is_server=True,
                              fuse_message=False,
                              name=f"postprocess_{i}_feedin_queue")
                for i in range(postproc_worker_config.num_postprocess_workers)
            ]
        else:
            # IPC queue for sending results back to the proxy, and let the
            # Proxy process to handle the postprocess
            result_queue = FusedIpcQueue(worker_queues.result_queue_addr,
                                         is_server=False,
                                         fuse_message=False,
                                         name="worker_result_queue")

    def notify_proxy_threads_to_quit():
        # Signal the dispatcher thread in the proxy to quit
        if result_queue is not None:
            result_queue.put(None)
        else:
            assert result_queues is not None
            for q in result_queues:
                q.put(None)
        # Signal the stats thread in the proxy to quit
        mp_stats_queue.put(None)
        kv_cache_events_queue.put(None)

    postprocess_worker_futures = []
    if is_leader and postproc_worker_config.enabled:
        logger_debug(f"initiate postprocess workers...", "yellow")

        proxy_result_queue: tuple[
            str, Optional[bytes]] = worker_queues.result_queue_addr

        assert result_queues is not None
        postproc_worker_pool = ProcessPoolExecutor(
            max_workers=postproc_worker_config.num_postprocess_workers)
        assert isinstance(proxy_result_queue, tuple)
        for i in range(postproc_worker_config.num_postprocess_workers):
            fut = postproc_worker_pool.submit(
                postproc_worker_main,
                result_queues[i].address,
                proxy_result_queue,
                postproc_worker_config.postprocess_tokenizer_dir,
                PostprocWorker.default_record_creator,
            )
            postprocess_worker_futures.append(fut)

    # Error handling in the Worker/MPI process
    #   1. During Executor initialization, the errors will be captured and
    #      send back via request_error_queue.
    #   2. During execution, the errors will be captured by ManagedThreads
    #      a) For per-request error, the error will be send back via
    #         result_queue, and eventually raised in handle_response() in
    #         the main thread.
    #      b) For system error, the error will be raised in the MPI process
    #         and handled by future.done_callback, that will propagate the
    #         error to the error_queue in the main thread.

    mpi_comm().barrier()
    logger_debug(f"Worker {mpi_rank()} ready to setup backend...\n", "green")

    try:
        worker: GenerationExecutorWorker = worker_cls(
            engine,
            executor_config,
            batched_logits_processor,
            postproc_worker_config=postproc_worker_config,
            is_llm_executor=is_llm_executor,
            hf_model_dir=hf_model_dir,
            tokenizer=tokenizer,
            llm_args=llm_args)
    except Exception as e:
        logger.error(f"Failed to initialize executor on rank {mpi_rank()}: {e}")
        logger.error(traceback.format_exc())
        logger_debug(f"error: {traceback.format_exc()}", "red")
        if is_leader:
            # Send error message with confirmation
            error_msg = (e, traceback.format_exc())
            if not worker_init_status_queue.notify_with_retry(error_msg):
                logger.error("Failed to deliver error message to proxy")
        return

    # Optionally disable GC (default: not disabled)
    if os.getenv("TRTLLM_WORKER_DISABLE_GC", "0") == "1":
        gc.disable()

    with worker:
        try:
            worker.block_subordinates()

            if is_leader:
                if postproc_worker_config.enabled:
                    worker.set_postproc_queues(result_queues)
                else:
                    worker.set_result_queue(result_queue)

                # initialize the iteration result queues
                worker._set_iteration_result_queue(worker.stats_queues,
                                                   mp_stats_queue)
                worker._set_iteration_result_queue(worker.kv_events_queues,
                                                   kv_cache_events_queue)
                # Send ready signal with confirmation
                ready_msg = (ready_signal, None)
                if not worker_init_status_queue.notify_with_retry(ready_msg):
                    logger.warning(
                        "Failed to deliver ready signal to proxy, continuing anyway"
                    )
                while (req := request_queue.get()) is not None:
                    if isinstance(req, CancellingRequest):
                        worker.abort_request(req.id)
                    elif isinstance(req, GenerationRequest):
                        try:
                            worker.submit(req)
                        except RequestError as e:
                            logger.error(f"submit request failed: {e}")
                            logger.error(traceback.format_exc())
                            worker._await_response_helper.temp_error_responses.put(
                                ErrorResponse(req.id, e, req.id))
                    else:
                        raise ValueError(f"Unknown request type: {type(req)}")

                notify_proxy_threads_to_quit()

        except GenerationExecutorWorker.WorkerExit as e:
            # This will capture by the with-statement and exit normally.
            raise e

        except Exception as e:  # other critical errors
            if is_leader:
                notify_proxy_threads_to_quit()
            logger.error(traceback.format_exc())
            # This will be captured by mpi4py and handled by future.done_callback
            raise e
