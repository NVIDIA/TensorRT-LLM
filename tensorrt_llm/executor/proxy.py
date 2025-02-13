import concurrent.futures
import os
import time
import traceback
import weakref
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Callable, Dict, List, Optional

import zmq
import zmq.asyncio

from tensorrt_llm.logger import logger

from .._utils import mpi_rank, mpi_world_size
from ..bindings import executor as tllm
from ..builder import Engine
from ..llmapi.mpi_session import MpiCommSession, MpiPoolSession, MpiSession
from ..llmapi.tracer import (VizTracer, enable_llm_tracer, get_tracer,
                             global_tracer, set_global_tracer)
from ..llmapi.utils import (AsyncQueue, ManagedThread, _SyncQueue,
                            clear_sched_affinity, enable_llm_debug,
                            print_colored, print_colored_debug,
                            print_traceback_on_error)
from .executor import GenerationExecutor
from .ipc import FusedIpcQueue, IpcQueue
from .postproc_worker import PostprocWorker, PostprocWorkerConfig
from .request import GenerationRequest
from .result import GenerationResult
from .utils import (BATCH_RESP_IN_AWAIT, IntraProcessQueue,
                    ProcessPoolExecutorSession, RequestError, WorkerIPCAddrs,
                    WorkerQueues)
from .worker import ExecutorBindingsWorker

__all__ = [
    "ExecutorBindingsProxy",
]


class ExecutorBindingsProxy(GenerationExecutor):
    READY_SIGNAL = b"READY"

    def __init__(
            self,
            workers_kwargs: dict,
            model_world_size: int = 1,
            mpi_session: Optional[MpiSession] = None,
            *,
            worker_cls: type = ExecutorBindingsWorker,
            postproc_worker_config: Optional[PostprocWorkerConfig] = None
    ) -> None:
        postproc_worker_config = postproc_worker_config or PostprocWorkerConfig(
        )
        super().__init__(
            num_postprocess_workers=postproc_worker_config.
            num_postprocess_workers,
            postprocess_tokenizer_dir=postproc_worker_config.
            postprocess_tokenizer_dir,
        )

        self.workers_started = False
        self.worker_cls = worker_cls

        if mpi_session is None:
            if model_world_size == mpi_world_size() and model_world_size > 1:
                self.mpi_session = MpiCommSession(model_world_size)
            else:
                self.mpi_session = MpiPoolSession(n_workers=model_world_size)
        else:
            self.mpi_session = mpi_session

        if isinstance(self.mpi_session, MpiCommSession):
            print_colored(
                "Using MpiCommSession to bind to external MPI processes\n",
                "yellow")
        else:
            print_colored("Using MpiPoolSession to spawn MPI processes\n",
                          "yellow")

        self._results: Dict[int, GenerationResult] = {}

        self.model_world_size = model_world_size

        intra_node = isinstance(self.mpi_session,
                                (MpiPoolSession, ProcessPoolExecutorSession))
        if (not intra_node) and postproc_worker_config.enabled:
            raise NotImplementedError(
                "Postprocess parallel is not supported in inter-node mode")

        self.workers_kwargs = dict(
            **workers_kwargs,
            worker_queues=self._setup_queues(intra_node),
            postproc_worker_config=postproc_worker_config,
        )

        if "log_level" not in self.workers_kwargs:
            self.workers_kwargs["log_level"] = logger.level

        self.dispatch_result_thread: Optional[ManagedThread] = None
        self.dispatch_stats_thread: Optional[ManagedThread] = None

        self._start_executor_workers()

    def _setup_queues(self, intra_node: bool) -> WorkerIPCAddrs | WorkerQueues:
        # For intra-node communication, we use IPC queues. While for inter-node
        # communication, we use Queue instead as the MPI process is the Python
        # main process in rank 0.
        # TODO: In inter-node mode, it may necessary to spawn a separate process
        # for the MPI process for higher streaming generation performance.
        # TODO: Support postproc in the inter-node mode, since the postproc
        # workers need IPC queues.

        if intra_node:
            self.request_queue = IpcQueue(is_server=True,
                                          name="proxy_request_queue")
            self.request_error_queue = IpcQueue(
                is_server=True, name="proxy_request_error_queue")
            # TODO[chunweiy]: Unify IpcQueue and FusedIpcQueue
            # Use PULL mode when enable_postprocess_parallel as there are
            # multiple senders from multiple processes.
            self.result_queue = FusedIpcQueue(
                is_server=True,
                fuse_message=False,
                socket_type=zmq.PULL
                if self.enable_postprocess_parallel else zmq.PAIR,
                name="proxy_result_queue")
            self.mp_stats_queue = FusedIpcQueue(is_server=True,
                                                fuse_message=False,
                                                name="proxy_stats_queue")
            return WorkerIPCAddrs(
                request_queue_addr=self.request_queue.address,
                request_error_queue_addr=self.request_error_queue.address,
                result_queue_addr=self.result_queue.address,
                stats_queue_addr=self.mp_stats_queue.address,
            )
        else:
            self.request_queue = IntraProcessQueue()
            self.request_error_queue = IntraProcessQueue()
            self.result_queue = IntraProcessQueue()
            self.mp_stats_queue = IntraProcessQueue()

            return WorkerQueues(
                request_queue=self.request_queue,
                request_error_queue=self.request_error_queue,
                result_queue=self.result_queue,
                stats_queue=self.mp_stats_queue,
            )

    @print_traceback_on_error
    @staticmethod
    def postprocess_workers_main(feedin_ipc_addr: str, feedout_ipc_addr: str,
                                 tokenizer_dir: str, record_creator: Callable,
                                 result_handler: Callable):
        worker = PostprocWorker(feedin_ipc_addr,
                                feedout_ipc_addr,
                                tokenizer_dir=tokenizer_dir,
                                record_creator=record_creator,
                                result_handler=result_handler)
        worker.start()

    @print_traceback_on_error
    @staticmethod
    def workers_main(
            engine: Path | Engine,
            worker_queues: WorkerIPCAddrs | WorkerQueues,
            log_level: str,
            executor_config: Optional[tllm.ExecutorConfig] = None,
            logits_post_processor_map: Optional[Dict[str, Callable]] = None,
            worker_cls: type = ExecutorBindingsWorker,
            tracer_init_kwargs: Optional[dict] = None,
            _torch_model_class_mapping: Optional[dict] = None,
            postproc_worker_config: Optional[PostprocWorkerConfig] = None,
            rank0_extra_kwargs: Optional[
                dict] = None,  # a placeholder for multi-node
    ) -> None:
        pid = os.getpid()
        cpus = os.sched_getaffinity(pid)
        if cpus:
            logger.warning(
                f"Found worker process {pid} was bound to {cpus}, this may harm"
                "performance.", )
            logger.warning(f"Will clear the cpu affinity")
            clear_sched_affinity(pid)

        result_queue: Optional[IpcQueue] = None
        result_queues: Optional[List[IpcQueue]] = None

        postproc_worker_config = postproc_worker_config or PostprocWorkerConfig(
        )

        if tracer_init_kwargs is not None and mpi_rank() == 0:
            tracer = VizTracer(**tracer_init_kwargs)
            tracer.register_exit()
            tracer.start()
            set_global_tracer(tracer)

        if _torch_model_class_mapping is not None:
            from tensorrt_llm._torch.models.modeling_auto import \
                MODEL_CLASS_MAPPING
            MODEL_CLASS_MAPPING.update(**_torch_model_class_mapping)

        is_leader: bool = mpi_rank() == 0

        if is_leader:
            # Only set the log level for the leader process, the other processes will inherit the log level from "TLLM_LOG_LEVEL" environment variable
            logger.set_level(log_level)
            if isinstance(worker_queues, WorkerIPCAddrs):
                request_queue = IpcQueue(worker_queues.request_queue_addr,
                                         is_server=False,
                                         name="worker_request_queue")
                request_error_queue = IpcQueue(
                    worker_queues.request_error_queue_addr,
                    is_server=False,
                    name="worker_request_error_queue")

                if postproc_worker_config.enabled:
                    # IPC queues for sending inputs to the postprocess parallel
                    # processes, each one is a PAIR zmq socket
                    result_queues = [
                        FusedIpcQueue(is_server=True,
                                      fuse_message=True,
                                      name=f"postprocess_{i}_feedin_queue")
                        for i in range(
                            postproc_worker_config.num_postprocess_workers)
                    ]
                else:
                    # IPC queue for sending results back to the proxy, and let the
                    # Proxy process to handle the postprocess
                    result_queue = FusedIpcQueue(
                        worker_queues.result_queue_addr,
                        is_server=False,
                        fuse_message=not BATCH_RESP_IN_AWAIT,
                        name="worker_result_queue")

                mp_stats_queue = FusedIpcQueue(worker_queues.stats_queue_addr,
                                               is_server=False,
                                               fuse_message=False,
                                               name="worker_stats_queue")
            else:
                request_queue = worker_queues.request_queue
                request_error_queue = worker_queues.request_error_queue
                if postproc_worker_config.enabled:
                    raise NotImplementedError("Postprocess parallel is not "
                                              "supported in intra-node mode")
                else:
                    result_queue = worker_queues.result_queue
                mp_stats_queue = worker_queues.stats_queue

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

        postprocess_worker_futures = []
        if is_leader and postproc_worker_config.enabled:
            print_colored_debug(f"initiate postprocess workers...", "yellow")
            assert result_queues is not None
            assert postproc_worker_config.postprocess_tokenizer_dir is not None
            postprocess_worker_pool = ProcessPoolExecutor(
                max_workers=postproc_worker_config.num_postprocess_workers)
            for i in range(postproc_worker_config.num_postprocess_workers):
                fut = postprocess_worker_pool.submit(
                    ExecutorBindingsProxy.postprocess_workers_main,
                    result_queues[i].address,
                    worker_queues.result_queue_addr,
                    postproc_worker_config.postprocess_tokenizer_dir,
                    PostprocWorker.default_record_creator,
                    result_handler=postproc_worker_config.
                    postprocess_result_handler)
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

        try:
            executor: ExecutorBindingsWorker = worker_cls(
                engine,
                executor_config,
                logits_post_processor_map,
                postproc_worker_config=postproc_worker_config)
        except Exception as e:
            logger.error(
                f"Failed to initialize executor on rank {mpi_rank()}: {e}")
            logger.error(traceback.format_exc())
            if mpi_rank() == 0:
                request_error_queue.put(e)
            return

        with executor:
            try:
                executor.block_subordinates()

                if mpi_rank() == 0:
                    if postproc_worker_config.enabled:
                        executor.set_postprocess_queues(result_queues)
                    else:
                        executor.set_result_queue(result_queue)

                    executor.set_stats_queue(mp_stats_queue)
                    request_error_queue.put(ExecutorBindingsProxy.READY_SIGNAL)
                    while (req := request_queue.get()) is not None:
                        try:
                            result = executor.submit(req)
                            request_error_queue.put(None)  # None means success
                        except RequestError as e:
                            request_error_queue.put(e)

                    notify_proxy_threads_to_quit()

            except ExecutorBindingsWorker.WorkerExit as e:
                # This will capture by the with-statement and exit normally.
                raise e

            except Exception as e:  # other critical errors
                if mpi_rank() == 0:
                    notify_proxy_threads_to_quit()
                err = Exception(f"Failed during generation: {e}")
                if mpi_rank() == 0:
                    request_error_queue.put(err)

    def dispatch_result_task(self) -> bool:
        # TODO[chunweiy]: convert the dispatch_result_task to async, that should
        # benefit from zmq.asyncio.Context
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

            if res.is_final:
                self._results.pop(client_id)

        res = res if isinstance(res, list) else [res]

        for i in res:
            global_tracer().log_instant("IPC.get")
            if i is None:
                return False
            process_res(i)

        if async_queues:
            _SyncQueue.notify_many(event_loop, async_queues)

        return True  # success

    def dispatch_stats_task(self) -> bool:
        # get-stats is not urgent, so we can sleep a bit

        time.sleep(0.1)

        try:
            stats = self.mp_stats_queue.get()
        except:
            return False

        if stats is None:
            return False

        stats = stats if isinstance(stats, list) else [stats]

        while self.stats_queue.full():
            self.stats_queue.get()

        try:
            for s in stats:
                if s is None:
                    return False
                self.stats_queue.put(s)
        except AsyncQueue.EventLoopShutdownError:
            # This happens in the last stats loop while the generate workflow is
            # stopped.
            pass
        except Exception as e:
            raise e

        return True  # success

    def _start_dispatch_threads(self):
        if self.dispatch_result_thread is None:

            self.dispatch_result_thread = ManagedThread(
                weakref.WeakMethod(self.dispatch_result_task),
                error_queue=self._error_queue,
                name="proxy_dispatch_result_thread")
            self.dispatch_stats_thread = ManagedThread(
                weakref.WeakMethod(self.dispatch_stats_task),
                error_queue=self._error_queue,
                name="proxy_dispatch_stats_thread")

            self.dispatch_result_thread.start()
            self.create_stats_queue()
            # TODO: clean up the stats thread, and replace with a decent
            # get_stats API
            #self.dispatch_stats_thread.start()

        self._handle_background_error()

    def _start_executor_workers(self):

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

        rank0_extra_kwargs = {}
        if worker_queues := self.workers_kwargs["worker_queues"]:
            if isinstance(worker_queues, WorkerQueues):
                rank0_extra_kwargs = {"worker_queues": worker_queues}
                self.workers_kwargs["worker_queues"] = None
        self.mpi_futures = self.mpi_session.submit(
            ExecutorBindingsProxy.workers_main,
            rank0_extra_kwargs=rank0_extra_kwargs,
            **self.workers_kwargs,
            worker_cls=self.worker_cls,
            tracer_init_kwargs=tracer_init_kwargs,
            _torch_model_class_mapping=MODEL_CLASS_MAPPING,
        )
        for fut in self.mpi_futures:
            fut.add_done_callback(mpi_done_callback)

        self.workers_started = True

        while not self.request_error_queue.poll(1):
            self._handle_background_error()

        ready_signal = self.request_error_queue.get()
        if ready_signal != ExecutorBindingsProxy.READY_SIGNAL:
            raise ready_signal

    def shutdown(self):
        if enable_llm_debug():
            try:
                print_colored('Proxy.shutdown...\n', "yellow")
                print_colored(str(traceback.format_exc()) + "\n", "yellow")
            except ValueError:
                pass
        if not self.workers_started:
            return

        if self.doing_shutdown:
            return
        else:
            self.doing_shutdown = True

        # step1: notify the workers to quit
        if all(not f.done() for f in self.mpi_futures):
            self.request_queue.put(None)

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
        if self.dispatch_stats_thread is not None and self.dispatch_stats_thread.is_alive(
        ):
            self.dispatch_stats_thread.stop()
            self.dispatch_stats_thread.join()

        # step3: finish all remaining work

        # close all the sockets
        self.request_queue.close()
        self.request_error_queue.close()
        self.result_queue.close()
        self.mp_stats_queue.close()

        self.workers_started = False
        self.mpi_session.shutdown()

        # Process the errors in-case error during shutting down the threads
        self._handle_background_error()

    def submit(self, request: GenerationRequest) -> GenerationResult:
        """
            Low-level API to the executor. Return a "future" GenerationResult
            which can be waited.
            Forwards the request to the workers through the request queue.
        """
        self._start_dispatch_threads()

        request.set_id(self._get_next_client_id())

        result = GenerationResult(
            request, background_error_handler=self._handle_background_error)
        self._results[request.id] = result

        self.request_queue.put(request)

        error = self.request_error_queue.get()
        if isinstance(error, Exception):
            raise error

        self._handle_background_error()

        return result

    def __del__(self):
        self.shutdown()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.shutdown()
        return False  # propagate the exception
