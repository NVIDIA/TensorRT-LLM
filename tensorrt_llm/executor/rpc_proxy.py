import atexit
import os
import threading
import time
from typing import Optional

from ..llmapi.mpi_session import MpiPoolSession, MpiSession
from ..llmapi.tracer import global_tracer
from ..llmapi.utils import _SyncQueue, print_colored_debug
from .executor import GenerationExecutor
from .postproc_worker import PostprocWorkerConfig
from .request import GenerationRequest
from .result import GenerationResult
from .rpc import RPCClient
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

        self.mpi_session = self._create_mpi_session(model_world_size,
                                                    mpi_session)

        self._shutdown_event = threading.Event()

        self.launch_workers()
        time.sleep(1)  # wait for the workers to launch

        # Invoke model creation on the remote
        # TBD: Move model creation to the mpi task, or left in RPC?
        self.create_engine_remote()

        self.setup_mainloop()

    def launch_workers(self):
        assert self.mpi_session is not None
        self.mpi_session.submit(rpc_worker_main,
                                rpc_addr=self.rpc_addr,
                                **self.worker_kwargs)

    def main_loop_task(self):
        """
        Main loop of the proxy, it will invoke the actions periodically.
        """
        clock = 0
        while not self._shutdown_event.is_set():
            if clock % 1 == 0:
                responses = self.await_responses_remote()
                self.handle_responses(responses)
            if clock % 10 == 0:
                stats = self.get_stats_remote()  # TODO
                self.handle_stats(stats)

            clock += 1
            time.sleep(self.clock_unit)

    def setup_mainloop(self):
        self.main_loop_thread = threading.Thread(target=self.main_loop_task,
                                                 daemon=True)
        self.main_loop_thread.start()
        atexit.register(self.shutdown)

    def handle_responses(self, responses: list[GenerationResult]) -> bool:
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

            if (is_llm_response(res) and res.result.is_final) or isinstance(
                    res, ErrorResponse):
                self._results.pop(client_id)

        for res in responses:
            global_tracer().log_instant("RPC.get")
            process_res(res)

        if async_queues:
            _SyncQueue.notify_many(event_loop, async_queues)

    def handle_stats(self, stats: dict):
        raise NotImplementedError

    def submit(self, request: GenerationRequest) -> GenerationResult:
        # submit is a fire-and-forget operation, don't need to wait for response
        return self.rpc_client.submit(request, need_response=False)

    def await_responses_remote(self):
        return self.rpc_client.await_responses()

    def create_engine_remote(self):
        return self.rpc_client.create_engine()  # TODO

    def shutdown_remote(self):
        self.rpc_client.shutdown()

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
