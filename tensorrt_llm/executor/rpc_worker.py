from pathlib import Path
from queue import Queue
from threading import Event
from typing import Optional, Union

import nvtx

from tensorrt_llm._utils import mpi_comm
from tensorrt_llm.llmapi.utils import enable_llm_debug, logger_debug

from .._utils import mpi_rank
from ..bindings import executor as tllm
from ..builder import Engine
from ..llmapi.llm_args import BaseLlmArgs
from ..llmapi.tokenizer import TokenizerBase
from ..logger import set_level
from ..sampling_params import BatchedLogitsProcessor
from .base_worker import BaseWorker
from .postproc_worker import PostprocWorkerConfig
from .rpc import RPCServer
from .rpc_worker_mixin import RpcWorkerMixin


class RpcWorker(RpcWorkerMixin, BaseWorker):
    """
    A RPC wrapper for the BaseWorker class.

    Actions:
        - `setup_engine`: Setup the engine.
        - `submit`: Submit a request to the worker.
        - `fetch_responses`: Fetch the latest responses from engine.
        - `fetch_stats`: Fetch the latest stats from engine.
        - `fetch_kv_cache_events`: Fetch the latest kv cache events from engine.
        - `shutdown`: Shutdown the worker.
    """

    # Default number of RPC server workers
    # Increased to handle concurrent requests and prevent thread pool exhaustion
    # Need enough workers for: submit requests + fetch_responses + other operations
    # Can be overridden via constructor parameter
    DEFAULT_NUM_WORKERS = 32

    # Default timeout for fetch_responses in seconds
    # This is a short timeout to prevent blocking the event loop while still allowing
    # responses to be fetched efficiently. The value is tuned to balance responsiveness
    # and CPU usage. Can be overridden via constructor parameter.
    DEFAULT_FETCH_TIMEOUT = 0.1

    def __init__(
        self,
        engine: Union[Path, Engine],
        executor_config: Optional[tllm.ExecutorConfig] = None,
        is_llm_executor: Optional[bool] = None,
        batched_logits_processor: Optional[BatchedLogitsProcessor] = None,
        postproc_worker_config: Optional[PostprocWorkerConfig] = None,
        hf_model_dir: Optional[Path] = None,
        tokenizer: Optional[TokenizerBase] = None,
        llm_args: Optional[BaseLlmArgs] = None,
        num_workers: Optional[int] = None,
        fetch_timeout: Optional[float] = None,
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

        # Configure number of RPC workers
        self.num_workers = num_workers if num_workers is not None else self.DEFAULT_NUM_WORKERS

        # Configure fetch timeout
        self._fetch_timeout = fetch_timeout if fetch_timeout is not None else self.DEFAULT_FETCH_TIMEOUT

        # Extract garbage_collection_gen0_threshold from llm_args if available
        self.garbage_collection_gen0_threshold = (
            llm_args.garbage_collection_gen0_threshold if llm_args is not None
            and hasattr(llm_args, 'garbage_collection_gen0_threshold') else
            None)
        self.shutdown_event = Event()

        self._response_queue = Queue()
        self.set_result_queue(self._response_queue)

        # Note: We don't create a persistent ThreadPoolExecutor anymore
        # to avoid thread leaks. Instead, we use asyncio.to_thread() which
        # manages threads internally.

    def setup_engine(self):
        # Force all the ranks to wait here, and start creating the executor simultaneously.
        # Only call barrier if we have multiple ranks to avoid hanging in single-process tests
        if mpi_comm().Get_size() > 1:
            mpi_comm().barrier()

        super().setup_engine()

    def shutdown(self):
        logger_debug(f"[worker] RpcWorker #{mpi_rank()} is shutting down",
                     color="yellow")
        self.shutdown_event.set()
        super().shutdown()
        logger_debug(f"[worker] RpcWorker #{mpi_rank()} is shutdown",
                     color="yellow")

    def start(self):
        pass

    @staticmethod
    def main_task(
        engine: Union[Path, Engine],
        rpc_addr: str,
        *,
        executor_config: Optional[tllm.ExecutorConfig] = None,
        batched_logits_processor: Optional[BatchedLogitsProcessor] = None,
        postproc_worker_config: Optional[PostprocWorkerConfig] = None,
        is_llm_executor: Optional[bool] = None,
        llm_args: Optional[BaseLlmArgs] = None,
        hf_model_dir: Optional[Path] = None,
        tokenizer: Optional[TokenizerBase] = None,
        **kwargs,
    ) -> None:
        nvtx.push_range(f"RpcWorker.main_task_{mpi_rank()}", color="pink")

        if enable_llm_debug():
            set_level("debug")

        # Step 1: Create the worker instance
        worker = RpcWorker(
            engine=engine,
            executor_config=executor_config,
            is_llm_executor=is_llm_executor,
            llm_args=llm_args,
            batched_logits_processor=batched_logits_processor,
            postproc_worker_config=postproc_worker_config,
            hf_model_dir=hf_model_dir,
            tokenizer=tokenizer,
        )

        if mpi_rank() != 0:
            # The non-leader worker will setup the engine immediately.
            # The leader worker will wait for the RPC call to propagate the
            # potential error.
            logger_debug(
                f"[worker] Worker {mpi_rank()} is setting up the engine",
                color="yellow")
            worker.setup_engine()

        else:
            logger_debug(
                f"[worker] Worker {mpi_rank()} is creating the RPC service with {worker.num_workers} workers",
                color="yellow")
            # Step 2: Create the RPC service, it will expose all the APIs of the worker as remote call to the client
            # Set num_workers to larger than 1 since there are some streaming tasks runs infinitely, such as await_responses_async.
            hmac_key = kwargs.get("hmac_key")
            rpc_server = RPCServer(worker,
                                   num_workers=worker.num_workers,
                                   hmac_key=hmac_key)
            rpc_server.bind(rpc_addr)
            rpc_server.start()
            logger_debug(f"[worker] RPC server {mpi_rank()} is started",
                         color="yellow")

            # Step 3: Wait for the worker to shutdown
            logger_debug(
                f"[worker] Worker {mpi_rank()} is waiting for shutdown event",
                color="yellow")
            worker.shutdown_event.wait()
            rpc_server.shutdown()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.shutdown()
        return True
