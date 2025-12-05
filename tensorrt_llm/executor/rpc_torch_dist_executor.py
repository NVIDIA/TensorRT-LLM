import multiprocessing
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from tensorrt_llm._utils import get_free_port
from tensorrt_llm.bindings import executor as tllm
from tensorrt_llm.builder import Engine
from tensorrt_llm.executor.executor import GenerationExecutor
from tensorrt_llm.executor.postproc_worker import PostprocWorkerConfig
from tensorrt_llm.executor.rpc_proxy_mixin import RpcExecutorMixin
from tensorrt_llm.executor.rpc_torch_dist_worker import RpcTorchDistWorker
from tensorrt_llm.llmapi.llm_args import BaseLlmArgs
from tensorrt_llm.llmapi.tokenizer import TokenizerBase
from tensorrt_llm.logger import logger
from tensorrt_llm.sampling_params import BatchedLogitsProcessor


class RpcTorchDistExecutor(RpcExecutorMixin, GenerationExecutor):
    def __init__(
        self,
        worker_kwargs: Dict,
        model_world_size: int,
        postproc_worker_config: PostprocWorkerConfig,
        is_llm_executor: bool,
    ):
        # Initialize GenerationExecutor
        super().__init__(
            num_postprocess_workers=postproc_worker_config.num_postprocess_workers,
            postprocess_tokenizer_dir=postproc_worker_config.postprocess_tokenizer_dir,
            is_llm_executor=is_llm_executor,
        )

        self.world_size = model_world_size
        self.processes: List[multiprocessing.Process] = []

        # Setup RPC
        self.init_rpc_executor()

        # Determine Master Addr/Port for torch.distributed
        self.master_addr = "127.0.0.1"
        self.master_port = str(get_free_port())

        logger.info(
            f"RpcTorchDistExecutor starting with {model_world_size} workers."
            f"Master: {self.master_addr}:{self.master_port}"
        )

        # Spawn workers
        self.start_workers(worker_kwargs)

        # Setup engine (remote)
        # This will trigger setup_engine on rank 0 via RPC, which broadcasts to other ranks
        try:
            logger.info("Setting up remote engine...")
            self.setup_engine_remote()
        except Exception as e:
            logger.error(f"Failed to setup remote engine: {e}")
            self.shutdown()
            raise e

        # Setup main loop for receiving results from RPC
        self.setup_mainloop()

    def start_workers(self, worker_kwargs: Dict):
        ctx = multiprocessing.get_context("spawn")

        for rank in range(self.world_size):
            p = ctx.Process(
                target=RpcTorchDistWorker.worker_main,
                args=(
                    rank,
                    self.world_size,
                    self.master_addr,
                    self.master_port,
                    self.rpc_addr,  # Passed to all, but only used by rank 0
                    worker_kwargs,
                ),
                name=f"RpcTorchDistWorker-{rank}",
            )
            p.start()
            self.processes.append(p)

    def setup_engine_remote(self):
        # Call setup_engine on Rank 0 via RPC
        # We wait for the result to ensure everything is initialized
        self.rpc_client.setup_engine().remote()

    def shutdown(self):
        if self.doing_shutdown:
            return
        self.doing_shutdown = True

        logger.info("Shutting down RpcTorchDistExecutor...")

        # RPC shutdown to Rank 0
        try:
            if hasattr(self, "rpc_client") and self.rpc_client:
                # This tells Rank 0 to shutdown, which broadcasts shutdown to others
                self.rpc_client.shutdown().remote(need_response=False)
        except Exception as e:
            logger.warning(f"Error during RPC shutdown: {e}")

        # Cleanup RPC client
        if hasattr(self, "rpc_client") and self.rpc_client:
            self.rpc_client.close()

        # Join processes
        for p in self.processes:
            if p.is_alive():
                p.join(timeout=5)
                if p.is_alive():
                    logger.warning(f"Process {p.name} did not exit, terminating...")
                    p.terminate()

        super().shutdown()

    @classmethod
    def create(
        cls,
        engine: Union[Path, Engine],
        executor_config: Optional[tllm.ExecutorConfig] = None,
        batched_logits_processor: Optional[BatchedLogitsProcessor] = None,
        model_world_size: int = 1,
        mpi_session: Optional[Any] = None,
        reuse_mpi_comm: bool = False,
        return_logits: bool = False,
        postproc_worker_config: Optional[PostprocWorkerConfig] = None,
        is_llm_executor: Optional[bool] = None,
        hf_model_dir: Optional[Path] = None,
        tokenizer: Optional[TokenizerBase] = None,
        llm_args: Optional[BaseLlmArgs] = None,
        **kwargs,
    ):
        postproc_worker_config = postproc_worker_config or PostprocWorkerConfig()

        worker_kwargs = {
            "engine": engine,
            "executor_config": executor_config,
            "batched_logits_processor": batched_logits_processor,
            "hf_model_dir": hf_model_dir,
            "tokenizer": tokenizer,
            "llm_args": llm_args,
        }

        return cls(
            worker_kwargs=worker_kwargs,
            model_world_size=model_world_size,
            postproc_worker_config=postproc_worker_config,
            is_llm_executor=is_llm_executor or False,
        )

    # Implement abstract methods from GenerationExecutor
    def submit(self, request):
        return super().submit(request)  # RpcExecutorMixin.submit

    def abort_request(self, request_id: int):
        # Forward to Rank 0
        self.rpc_client.abort_request(request_id).remote(need_response=False)
