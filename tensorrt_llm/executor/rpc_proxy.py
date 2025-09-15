import asyncio
import atexit
import os
import threading
import time
from typing import Optional

from ..llmapi.mpi_session import MpiPoolSession, MpiSession
from ..llmapi.tracer import global_tracer
from ..llmapi.utils import (_SyncQueue, logger_debug, print_colored_debug,
                            print_traceback_on_error)
from ..logger import logger
from .executor import GenerationExecutor
from .postproc_worker import PostprocWorkerConfig
from .request import GenerationRequest
from .result import GenerationResult
from .rpc import RPCClient
from .rpc_worker import RpcWorker
from .utils import (ErrorResponse, create_mpi_comm_session,
                    get_spawn_proxy_process_env, is_llm_response)


class GenerationExecutorRpcProxy(GenerationExecutor):
    # NOTE: this is a global counter for the number of instances of this class
    INSTANCE_COUNTER = 0

    def __init__(self,
                 worker_kwargs: dict,
                 model_world_size: int = 1,
                 mpi_session: Optional[MpiSession] = None,
                 *,
                 postproc_worker_config: Optional[PostprocWorkerConfig] = None,
                 is_llm_executor: Optional[bool] = None,
                 garbage_collection_gen0_threshold: Optional[int] = None,
                 clock_unit: int = 1):
        """
        Args:
            worker_kwargs: kwargs for the rpc worker
            model_world_size: the world size of the model
            mpi_session: the mpi session to use
            postproc_worker_config: the postproc worker config
            is_llm_executor: whether this is an llm executor
            garbage_collection_gen0_threshold: the garbage collection gen0 threshold
            clock_unit: the unit of the clock, 1 means 1 second
        """
        self.clock_unit = clock_unit

        GenerationExecutorRpcProxy.INSTANCE_COUNTER += 1
        self.rpc_addr = self.gen_uniq_rpc_addr()
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

        self.launch_workers()
        time.sleep(1)  # wait for the workers to launch

        # Invoke model creation on the remote
        # TBD: Move model creation to the mpi task, or left in RPC?
        self.setup_engine_remote()

        self.setup_mainloop()

    def launch_workers(self):
        logger.debug(f"Launching workers")
        assert self.mpi_session is not None
        self.mpi_session.submit(RpcWorker.main_task,
                                rpc_addr=self.rpc_addr,
                                **self.worker_kwargs)

    @print_traceback_on_error
    async def main_loop_task(self):
        """
        Main loop of the proxy, it will invoke the actions periodically.
        """
        async for responses in self.rpc_client.fetch_responses_loop_async(
        ).remote_streaming():
            if self._shutdown_event.is_set():
                return
            self.handle_responses(responses)

    def setup_mainloop(self):

        def _run_main_loop_task():
            """Local method to run the main loop task."""
            asyncio.run(self.main_loop_task())

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

        for res in responses:
            global_tracer().log_instant("RPC.get")
            process_res(res)

        if async_queues:
            _SyncQueue.notify_many(event_loop, async_queues)

    def handle_stats(self, stats: dict):
        # raise NotImplementedError
        pass

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

    def fetch_responses_remote(self):
        return self.rpc_client.fetch_responses().remote(timeout=20)

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
        logger_debug(f"Shutting down GenerationExecutorRpcProxy",
                     color="yellow")

        # 1. shutdown the rpc server (PyExecutor Rank 0 + RPC server)
        self.shutdown_remote()

        # 2. stop the main loop, so that no new rpc requests
        self._shutdown_event.set()
        self.main_loop_thread.join()

        # 3. shutdown the mpi session, this should wait until all the PyExecutor
        # processes are shutdown
        if self.mpi_session is not None:
            self.mpi_session.shutdown()

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
                print_colored_debug('create comm session ...\n', "yellow")
                self.mpi_session = create_mpi_comm_session(model_world_size)
            else:
                print_colored_debug('create pool session ...\n', "yellow")
                self.mpi_session = MpiPoolSession(n_workers=model_world_size)
        else:
            print_colored_debug('using external mpi session ...\n', "yellow")
            self.mpi_session = mpi_session

    @staticmethod
    def gen_uniq_rpc_addr() -> str:
        process_id = os.getpid()
        return f"ipc:///tmp/rpc-proxy-{process_id}-{GenerationExecutorRpcProxy.INSTANCE_COUNTER}"
