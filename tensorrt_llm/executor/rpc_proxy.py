import threading
from typing import Optional

from ..llmapi.mpi_session import MpiPoolSession, MpiSession
from ..llmapi.utils import logger_debug
from ..logger import logger
from .executor import GenerationExecutor
from .postproc_worker import PostprocWorkerConfig
from .rpc_proxy_mixin import RpcExecutorMixin
from .rpc_worker import RpcWorker
from .utils import create_mpi_comm_session, get_spawn_proxy_process_env


class GenerationExecutorRpcProxy(RpcExecutorMixin, GenerationExecutor):
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
        self.init_rpc_executor()

        postproc_worker_config = postproc_worker_config or PostprocWorkerConfig(
        )

        super().__init__(
            num_postprocess_workers=postproc_worker_config.
            num_postprocess_workers,
            postprocess_tokenizer_dir=postproc_worker_config.
            postprocess_tokenizer_dir,
            is_llm_executor=is_llm_executor,
        )

        self._create_mpi_session(model_world_size, mpi_session)

        # Inject the generated HMAC key into worker_kwargs for workers
        worker_kwargs['hmac_key'] = self.hmac_key
        self.worker_kwargs = worker_kwargs

        self.launch_workers()

        # Invoke model creation on the remote
        # TBD: Move model creation to the mpi task, or left in RPC?
        self.setup_engine_remote()

        # Setup main loop after engine is ready
        self._setup_mainloop_with_tasks()

    def launch_workers(self):
        logger.debug(f"Launching workers")
        assert self.mpi_session is not None
        self.mpi_session.submit(RpcWorker.main_task,
                                rpc_addr=self.rpc_addr,
                                **self.worker_kwargs)

    def _setup_mainloop_with_tasks(self):
        """Setup mainloop with all tasks needed for RpcProxy."""
        tasks = [
            self._fetch_responses_loop_async,
            self._fetch_stats_loop_async,
        ]
        # Only add kv_cache_events loop if it's enabled
        if self._iter_kv_events_result:
            tasks.append(self._fetch_kv_cache_events_loop_async)

        # Call mixin's setup_mainloop with custom tasks
        self.setup_mainloop(tasks=tasks, thread_name="rpc_proxy_main_loop")

    def fetch_stats_remote(self):
        return self.rpc_client.fetch_stats().remote()

    def setup_engine_remote(self):
        return self.rpc_client.setup_engine().remote(need_response=True)

    def shutdown_remote(self):
        logger_debug(f"Shutting down rpc remote", color="yellow")
        self.rpc_client.shutdown().remote(need_response=False)

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

        # Only join if we're not calling from the main_loop_thread itself
        # (e.g., during garbage collection in that thread)
        if self.main_loop_thread and threading.current_thread(
        ) != self.main_loop_thread:
            self.main_loop_thread.join(timeout=2.0)
            if self.main_loop_thread.is_alive():
                logger.warning("Main loop thread did not exit gracefully")

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
                logger_debug('[proxy] create comm session ...\n', "yellow")
                self.mpi_session = create_mpi_comm_session(model_world_size)
            else:
                logger_debug('[proxy] create pool session ...\n', "yellow")
                self.mpi_session = MpiPoolSession(n_workers=model_world_size)
        else:
            logger_debug('[proxy] using external mpi session ...\n', "yellow")
            self.mpi_session = mpi_session
