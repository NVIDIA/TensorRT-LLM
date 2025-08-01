from pathlib import Path
from queue import Queue
from threading import Event
from typing import Optional, Union

from .._utils import mpi_rank
from ..bindings import executor as tllm
from ..builder import Engine
from ..logger import logger
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
    ) -> None:
        super().__init__(engine=engine,
                         executor_config=executor_config,
                         is_llm_executor=is_llm_executor)
        self.shutdown_event = Event()

        self._response_queue = Queue()
        self.set_result_queue(self._response_queue)

    def fetch_responses(self) -> list:
        logger.debug(f"RPC worker {mpi_rank()} is fetching responses")
        super().await_responses()
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
    ) -> None:
        # Step 1: Create the worker instance
        worker = RpcWorker(engine=engine, executor_config=executor_config)

        if mpi_rank() != 0:
            logger.debug(f"Worker {mpi_rank()} is setting up the engine")
            # The non-leader worker will setup the engine immediately.
            # The leader worker will wait for the RPC call to propagate the
            # potential error.
            worker.setup_engine(
                engine=engine,
                executor_config=executor_config,
                batched_logits_processor=batched_logits_processor,
                postproc_worker_config=postproc_worker_config,
                is_llm_executor=is_llm_executor,
                lora_config=lora_config,
                garbage_collection_gen0_threshold=
                garbage_collection_gen0_threshold)

        if mpi_rank() == 0:
            # Step 2: Create the RPC service, it will expose all the APIs of the worker as remote call to the client
            rpc_server = RPCServer(worker)
            rpc_server.bind(rpc_addr)
            rpc_server.start()

            # Step 3: Wait for the worker to shutdown
            worker.shutdown_event.wait()
            rpc_server.shutdown()
