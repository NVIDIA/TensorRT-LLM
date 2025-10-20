import asyncio
from pathlib import Path
from queue import Queue
from threading import Event
from typing import AsyncGenerator, Optional, Union

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
from .request import GenerationRequest
from .rpc import RPCServer


class RpcWorker(BaseWorker):
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

    # Number of RPC server workers
    NUM_WORKERS = 6

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
    ) -> None:
        super().__init__(
            engine=engine,
            executor_config=executor_config,
            is_llm_executor=is_llm_executor,
            llm_args=llm_args,
            batched_logits_processor=batched_logits_processor,
            postproc_worker_config=postproc_worker_config,
            hf_model_dir=hf_model_dir,
            tokenizer=tokenizer,
        )
        # Extract garbage_collection_gen0_threshold from llm_args if available
        self.garbage_collection_gen0_threshold = (
            llm_args.garbage_collection_gen0_threshold if llm_args is not None
            and hasattr(llm_args, 'garbage_collection_gen0_threshold') else
            None)
        self.shutdown_event = Event()

        self._response_queue = Queue()
        self.set_result_queue(self._response_queue)

    def submit(self, request: GenerationRequest):
        """ Submits a request to the worker. """
        super().submit(request)

    def fetch_responses(self, timeout: Optional[float] = None) -> list:
        logger_debug(f"RpcWorker {mpi_rank()} is fetching responses",
                     color="yellow")
        # NOTE: This is a blocking call, it will wait for the responses to be available.
        responses = super().await_responses(timeout)
        self._await_response_helper.responses_handler(responses)

        qsize = self._response_queue.qsize()
        logger_debug(f"RpcWorker returning {qsize} responses", color="yellow")

        all_responses = []
        for _ in range(qsize):
            # The queue contains batches of responses, so extend the list
            all_responses.extend(self._response_queue.get())
        return all_responses

    async def fetch_responses_async(self,
                                    timeout: Optional[float] = None) -> list:
        # A really async version of fetch_responses
        logger_debug(f"RpcWorker {mpi_rank()} is fetching responses async",
                     color="yellow")

        # First, await any pending responses without blocking the event loop
        responses = await asyncio.to_thread(self.fetch_responses,
                                            timeout=timeout)
        return responses

    async def fetch_stats_async(self, timeout: Optional[float] = None) -> list:
        return await asyncio.to_thread(self.fetch_stats)

    async def fetch_kv_cache_events_async(self,
                                          timeout: Optional[float] = None
                                          ) -> list:
        return await asyncio.to_thread(self.fetch_kv_cache_events)

    # for streaming performance
    async def fetch_responses_loop_async(self) -> AsyncGenerator[list, None]:
        while not self.shutdown_event.is_set():
            responses = await self.fetch_responses_async()
            if responses:  # Only yield if there are actual responses
                logger_debug(
                    f"RpcWorker {mpi_rank()} is yielding responses: {responses}",
                    color="yellow")
                yield responses  # batching the responses to opt IPC performance
            else:
                # Small delay to prevent busy waiting when no responses
                await asyncio.sleep(0)
        logger_debug(
            f"RpcWorker {mpi_rank()} quitting fetch_responses_loop_async",
            color="yellow")

    async def _generic_fetch_loop_async(
            self,
            fetch_method,
            serializer,
            method_name: str,
            timeout: Optional[float] = None) -> AsyncGenerator[list, None]:
        """Generic method for fetching data in a loop.

        Args:
            fetch_method: The async method to call for fetching data
            serializer: The serializer function to apply to each item
            method_name: Name of the method for logging
            timeout: Optional timeout between fetches
        """
        while not self.shutdown_event.is_set():
            timeout = timeout or 0.1
            await asyncio.sleep(timeout)
            data = await fetch_method()
            # Always yield data, even if empty, to prevent the client looks like hanging
            # TODO: Remove the empty data to reduce the IPC overhead
            yield [serializer(item) for item in data]
        logger_debug(f"RpcWorker {mpi_rank()} quitting {method_name}",
                     color="yellow")

    async def fetch_stats_loop_async(
            self,
            timeout: Optional[float] = None) -> AsyncGenerator[list, None]:
        async for data in self._generic_fetch_loop_async(
                fetch_method=self.fetch_stats_async,
                serializer=self._stats_serializer,
                method_name="fetch_stats_loop_async",
                timeout=timeout):
            yield data

    async def fetch_kv_cache_events_loop_async(
            self,
            timeout: Optional[float] = None) -> AsyncGenerator[list, None]:
        async for data in self._generic_fetch_loop_async(
                fetch_method=self.fetch_kv_cache_events_async,
                serializer=self._kv_cache_events_serializer,
                method_name="fetch_kv_cache_events_loop_async",
                timeout=timeout):
            yield data

    def setup_engine(self):
        # Force all the ranks to wait here, and start creating the executor simultaneously.
        # Only call barrier if we have multiple ranks to avoid hanging in single-process tests
        if mpi_comm().Get_size() > 1:
            mpi_comm().barrier()

        super().setup_engine()

    def shutdown(self):
        logger_debug(f"RPC worker {mpi_rank()} is shutting down",
                     color="yellow")
        self.shutdown_event.set()
        super().shutdown()
        logger_debug(f"RPC worker {mpi_rank()} is shutdown", color="yellow")

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
            logger_debug(f"Worker {mpi_rank()} is setting up the engine",
                         color="yellow")
            worker.setup_engine()

        else:
            logger_debug(f"Worker {mpi_rank()} is creating the RPC service",
                         color="yellow")
            # Step 2: Create the RPC service, it will expose all the APIs of the worker as remote call to the client
            # Set num_workers to larger than 1 since there are some streaming tasks runs infinitely, such as await_responses_async.
            rpc_server = RPCServer(worker, num_workers=RpcWorker.NUM_WORKERS)
            rpc_server.bind(rpc_addr)
            rpc_server.start()

            # Step 3: Wait for the worker to shutdown
            logger_debug(
                f"Worker {mpi_rank()} is waiting for the worker to shutdown")
            worker.shutdown_event.wait()
            rpc_server.shutdown()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.shutdown()
        return True
