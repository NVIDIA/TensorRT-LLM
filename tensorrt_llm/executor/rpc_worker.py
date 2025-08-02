from pathlib import Path
from queue import Queue
from threading import Event
from typing import Optional, Union

from tensorrt_llm.llmapi.utils import enable_llm_debug

from .._utils import mpi_rank
from ..bindings import executor as tllm
from ..builder import Engine
from ..logger import logger, set_level
from ..lora_manager import LoraConfig
from ..sampling_params import BatchedLogitsProcessor
from .postproc_worker import PostprocWorkerConfig
from .rpc import RPCServer
from .worker_base import WorkerBase


class RpcWorker(WorkerBase):
    """
    A RPC wrapper for the WorkerBase class.

    Actions:
        - `setup_engine`: Setup the engine.
        - `fetch_responses`: Fetch the latest responses from engine.
        - `fetch_stats`: Fetch the latest stats from engine.
        - `fetch_kv_cache_events`: Fetch the latest kv cache events from engine.
        - `shutdown`: Shutdown the worker.
    """

    def __init__(
        self,
        engine: Union[Path, Engine],
        executor_config: Optional[tllm.ExecutorConfig] = None,
        is_llm_executor: Optional[bool] = None,
        lora_config: Optional[LoraConfig] = None,
        garbage_collection_gen0_threshold: Optional[int] = None,
    ) -> None:
        super().__init__(
            engine=engine,
            executor_config=executor_config,
            is_llm_executor=is_llm_executor,
            lora_config=lora_config,
            garbage_collection_gen0_threshold=garbage_collection_gen0_threshold)
        self.shutdown_event = Event()

        self._response_queue = Queue()
        self.set_result_queue(self._response_queue)

    def fetch_stats(self) -> list:
        return super().fetch_stats()

    def fetch_responses(self) -> list:
        logger.debug(f"RpcWorker {mpi_rank()} is fetching responses")
        # NOTE: This is a blocking call, it will wait for the responses to be available.
        super().await_responses()
        logger.debug(f"RpcWorker returning responses")
        qsize = self._response_queue.qsize()
        return [self._response_queue.get() for _ in range(qsize)]

    def shutdown(self):
        logger.debug(f"RPC worker {mpi_rank()} is shutting down")
        self.shutdown_event.set()
        super().shutdown()

    @staticmethod
    def main_task(
        engine: Union[Path, Engine],
        rpc_addr: str,
        *,
        executor_config: Optional[tllm.ExecutorConfig] = None,
        batched_logits_processor: Optional[BatchedLogitsProcessor] = None,
        postproc_worker_config: Optional[PostprocWorkerConfig] = None,
        is_llm_executor: Optional[bool] = None,
        lora_config: Optional[LoraConfig] = None,
        garbage_collection_gen0_threshold: Optional[int] = None,
        **kwargs,
    ) -> None:
        if enable_llm_debug():
            set_level("debug")

        # Step 1: Create the worker instance
        worker = RpcWorker(
            engine=engine,
            executor_config=executor_config,
            is_llm_executor=is_llm_executor,
            lora_config=lora_config,
            garbage_collection_gen0_threshold=garbage_collection_gen0_threshold)

        if mpi_rank() != 0:
            logger.debug(f"Worker {mpi_rank()} is setting up the engine")
            # The non-leader worker will setup the engine immediately.
            # The leader worker will wait for the RPC call to propagate the
            # potential error.
            logger.debug(f"Worker {mpi_rank()} is setting up the engine")
            worker.setup_engine()

        if mpi_rank() == 0:
            logger.debug(f"Worker {mpi_rank()} is creating the RPC service")
            # Step 2: Create the RPC service, it will expose all the APIs of the worker as remote call to the client
            rpc_server = RPCServer(worker)
            rpc_server.bind(rpc_addr)
            rpc_server.start()

            # Step 3: Wait for the worker to shutdown
            logger.debug(
                f"Worker {mpi_rank()} is waiting for the worker to shutdown")
            worker.shutdown_event.wait()
            rpc_server.shutdown()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.shutdown()
        return True
