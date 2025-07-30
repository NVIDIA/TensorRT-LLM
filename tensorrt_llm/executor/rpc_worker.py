from pathlib import Path
from threading import Event
from typing import Optional, Union

from ..bindings import executor as tllm
from ..builder import Engine
from ..lora_manager import LoraConfig
from ..sampling_params import BatchedLogitsProcessor
from .postproc_worker import PostprocWorkerConfig
from .rpc import RpcService
from .worker_base import WorkerBase


class RpcWorker(WorkerBase):

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

    def shutdown(self):
        self.shutdown_event.set()
        super().shutdown()


def rpc_worker_main(
    engine: Union[Path, Engine],
    rpc_addr: str,
    executor_config: Optional[tllm.ExecutorConfig] = None,
    batched_logits_processor: Optional[BatchedLogitsProcessor] = None,
    postproc_worker_config: Optional[PostprocWorkerConfig] = None,
    is_llm_executor: Optional[bool] = None,
    lora_config: Optional[LoraConfig] = None,
    garbage_collection_gen0_threshold: Optional[int] = None,
) -> None:
    # Step 1: Create the worker instance
    worker = RpcWorker(engine=engine, executor_config=executor_config)
    worker.create_engine(
        engine=engine,
        executor_config=executor_config,
        batched_logits_processor=batched_logits_processor,
        postproc_worker_config=postproc_worker_config,
        is_llm_executor=is_llm_executor,
        lora_config=lora_config,
        garbage_collection_gen0_threshold=garbage_collection_gen0_threshold)

    # Step 2: Create the RPC service, it will expose all the APIs of the worker as remote call to the client
    rpc_service = RpcService(worker)
    rpc_service.bind(rpc_addr)
    rpc_service.start()

    # Step 3: Wait for the worker to shutdown
    worker.shutdown_event.wait()
    rpc_service.shutdown()
