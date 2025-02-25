import concurrent.futures
import time
import traceback
import weakref
from typing import Dict, Optional

import zmq
import zmq.asyncio

from tensorrt_llm.logger import logger

from .._utils import mpi_world_size
from ..llmapi.mpi_session import MpiCommSession, MpiPoolSession, MpiSession
from ..llmapi.tracer import enable_llm_tracer, get_tracer, global_tracer
from ..llmapi.utils import (AsyncQueue, ManagedThread, _SyncQueue,
                            enable_llm_debug, print_colored)
from .executor import GenerationExecutor
from .ipc import FusedIpcQueue, IpcQueue
from .postproc_worker import PostprocWorkerConfig
from .request import CancellingRequest, GenerationRequest
from .result import GenerationResult
from .utils import (IntraProcessQueue, ProcessPoolExecutorSession,
                    WorkerCommIpcAddrs, WorkerCommQueues)
from .worker import ExecutorBindingsWorker, worker_main

__all__ = [
    "ExecutorBindingsProxy",
]


class ExecutorBindingsProxy(GenerationExecutor):
    READY_SIGNAL = b"READY"

    def __init__(
        self,
        worker_kwargs: dict,
        model_world_size: int = 1,
        mpi_session: Optional[MpiSession] = None,
        *,
        worker_cls: type = ExecutorBindingsWorker,
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

        if mpi_session is None:
            if model_world_size == mpi_world_size() and model_world_size > 1:
                self.mpi_session = MpiCommSession(n_workers=model_world_size)
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

        worker_kwargs = dict(**worker_kwargs,
                             worker_queues=self._setup_queues(intra_node),
                             postproc_worker_config=postproc_worker_config,
                             is_llm_executor=False)

        if "log_level" not in worker_kwargs:
            worker_kwargs["log_level"] = logger.level

        self.dispatch_result_thread: Optional[ManagedThread] = None
        self.dispatch_stats_thread: Optional[ManagedThread] = None

        self._start_executor_workers(worker_kwargs)

    def _setup_queues(
            self, intra_node: bool) -> WorkerCommIpcAddrs | WorkerCommQueues:
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
            return WorkerCommIpcAddrs(
                request_queue_addr=self.request_queue.address,
                request_error_queue_addr=self.request_error_queue.address,
                result_queue_addr=self.result_queue.address,
                stats_queue_addr=self.mp_stats_queue.address,
            )
        else:
            self.request_queue = IntraProcessQueue()
            self.request_error_queue = IntraProcessQueue()
            self.mp_stats_queue = IntraProcessQueue()

            if self.enable_postprocess_parallel:
                self.result_queue = FusedIpcQueue(
                    is_server=True,
                    fuse_message=False,
                    socket_type=zmq.PULL
                    if self.enable_postprocess_parallel else zmq.PAIR,
                    name="proxy_result_queue")

                res = WorkerCommQueues(
                    request_queue=self.request_queue,
                    request_error_queue=self.request_error_queue,
                    result_queue=self.result_queue.address,
                    stats_queue=self.mp_stats_queue,
                )

            else:
                self.result_queue = IntraProcessQueue()
                res = WorkerCommQueues(
                    request_queue=self.request_queue,
                    request_error_queue=self.request_error_queue,
                    result_queue=self.result_queue,
                    stats_queue=self.mp_stats_queue,
                )

            return res

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
            return False  # shutdown the thread

        stats = stats if isinstance(stats, list) else [stats]
        queue = self._iter_stats_result.queue
        async_queues = []

        while queue.full():
            queue.get()

        try:
            for s in stats:
                if s is None:
                    return False

                if isinstance(queue, _SyncQueue):
                    queue.put_nowait(s)
                    async_queues.append(queue)
                else:
                    queue.put(s)

            if async_queues:
                _SyncQueue.notify_many(queue.loop, async_queues)

        except AsyncQueue.EventLoopShutdownError:
            # This happens in the last stats loop while the generate workflow is
            # stopped, or when get_stats() or aget_stats() are not called by users
            # and therefore event loop can already be closed.
            return False
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

            # Only collect stats when submission
            # is via LLM API
            if self._iter_stats_result:
                self.dispatch_stats_thread.start()

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

        rank0_extra_kwargs = {}
        if worker_queues := worker_kwargs["worker_queues"]:
            if isinstance(worker_queues, WorkerCommQueues):
                rank0_extra_kwargs = {"worker_queues": worker_queues}
                worker_kwargs["worker_queues"] = None
        self.mpi_futures = self.mpi_session.submit(
            worker_main,
            **worker_kwargs,
            worker_cls=self.worker_cls,
            tracer_init_kwargs=tracer_init_kwargs,
            _torch_model_class_mapping=MODEL_CLASS_MAPPING,
            rank0_extra_kwargs=rank0_extra_kwargs,
            ready_signal=ExecutorBindingsProxy.READY_SIGNAL,
        )
        for fut in self.mpi_futures:
            fut.add_done_callback(mpi_done_callback)

        self.workers_started = True

        while not self.request_error_queue.poll(1):
            self._handle_background_error()

        ready_signal = self.request_error_queue.get()
        if ready_signal != ExecutorBindingsProxy.READY_SIGNAL:
            raise ready_signal

    def _abort_all_requests(self):
        for result in self._results.values():
            result.abort()

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

        self._abort_all_requests()

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
            request,
            background_error_handler=self._handle_background_error,
            executor=self)
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
