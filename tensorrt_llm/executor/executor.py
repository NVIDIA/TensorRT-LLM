import atexit
import faulthandler
import multiprocessing
import platform
import signal
import traceback
from abc import ABC, abstractmethod
from pathlib import Path
from queue import Queue
from typing import (TYPE_CHECKING, AsyncIterable, Dict, Generator, List,
                    Optional, Union)

import numpy as np
import torch

from tensorrt_llm.inputs.multimodal import MultimodalParams
from tensorrt_llm.logger import logger, set_level

from .._utils import mpi_world_size
from ..bindings import executor as tllm
from ..builder import Engine
from ..disaggregated_params import DisaggregatedParams
from ..llmapi.llm_args import BaseLlmArgs, TorchLlmArgs
from ..llmapi.llm_utils import KvCacheRetentionConfig
from ..llmapi.mpi_session import (MpiSession, external_mpi_comm_available,
                                  need_spawn_mpi_workers)
from ..llmapi.tokenizer import TokenizerBase
from ..llmapi.utils import (AsyncQueue, enable_llm_debug,
                            enable_worker_single_process_for_tp1, logger_debug,
                            print_colored)
from ..sampling_params import (BatchedLogitsProcessor, LogprobParams,
                               SamplingParams)
from ..scheduling_params import SchedulingParams
from .ipc import FusedIpcQueue
from .postproc_worker import PostprocParams, PostprocWorkerConfig
from .request import GenerationRequest, LoRARequest, PromptAdapterRequest
from .result import GenerationResult, IterationResult
from .utils import IntraProcessQueue, ProcessPoolExecutorSession, RequestError

if TYPE_CHECKING:
    from .proxy import GenerationExecutorProxy
    from .worker import GenerationExecutorWorker

__all__ = [
    "GenerationExecutor",
    "CppExecutorError",
]

if enable_llm_debug():
    # Mainly enable more detailed logging from cpp runtime.
    set_level("info")


async def empty_async_iterable() -> AsyncIterable:
    if False:  # ensures the function remains an async generator
        yield


class CppExecutorError(RuntimeError):

    def __init__(self, message: Optional[str] = None):
        self.message = message
        self.stack_trace = traceback.format_exc()
        super().__init__(message)

    def __str__(self):
        return f"{self.message}\nStack trace:\n{self.stack_trace}"


class IterationResultQueue:
    is_initialized: bool = False
    # FusedIpcQueue or IntraProcessQueue is used to communicate results from workers to proxy
    queue: Optional[Union[Queue, FusedIpcQueue, IntraProcessQueue]] = None
    aqueue: Optional[AsyncQueue] = None


class GenerationExecutor(ABC):

    def __init__(self,
                 num_postprocess_workers: int = 0,
                 postprocess_tokenizer_dir: Optional[str] = None,
                 is_llm_executor: Optional[bool] = None):
        self.postproc_config = PostprocWorkerConfig(
            num_postprocess_workers=num_postprocess_workers,
            postprocess_tokenizer_dir=postprocess_tokenizer_dir)

        self.kv_events_queues = IterationResultQueue()
        self.stats_queues = IterationResultQueue()

        atexit.register(self.shutdown)

        # This is used to capture the exceptions from the threads.
        self._error_queue = Queue()

        # A flag to avoid calling shutdown() recursively. This happens when the background threads raise errors.
        self.doing_shutdown = False

        self._last_client_id: int = 1

        # whether it's the executor instance of LLM API
        self._is_llm_executor = is_llm_executor
        self._iter_kv_events_result: IterationResult | None = None
        self._iter_stats_result: IterationResult | None = None

    def use_ray_queue(self) -> bool:
        return False

    @abstractmethod
    def submit(self, request: GenerationRequest) -> GenerationResult:
        pass

    @abstractmethod
    def abort_request(self, request_id: int) -> None:
        pass

    def generate_async(
        self,
        prompt_token_ids: List[int],
        sampling_params: SamplingParams,
        query_token_ids: Optional[Union[torch.Tensor, np.ndarray, list]] = None,
        lora_request: Optional[LoRARequest] = None,
        prompt_adapter_request: Optional[PromptAdapterRequest] = None,
        streaming: bool = False,
        kv_cache_retention_config: Optional[KvCacheRetentionConfig] = None,
        disaggregated_params: Optional[DisaggregatedParams] = None,
        postproc_params: Optional[PostprocParams] = None,
        multimodal_params: Optional[MultimodalParams] = None,
        scheduling_params: Optional[SchedulingParams] = None,
        cache_salt_id: Optional[int] = None,
        arrival_time: Optional[float] = None,
    ) -> GenerationResult:
        """Generate output for the given prompt token ids in the asynchronous mode.
        Asynchronous generation accepts single prompt only.
        """
        assert isinstance(prompt_token_ids[0], int)
        assert isinstance(sampling_params, SamplingParams)

        self._maybe_initialize_iteration_results()

        if postproc_params:
            postproc_params.postproc_args.num_prompt_tokens = len(
                prompt_token_ids)
        request = GenerationRequest(
            prompt_token_ids,
            sampling_params=sampling_params,
            postproc_params=postproc_params,
            query_token_ids=query_token_ids,
            lora_request=lora_request,
            prompt_adapter_request=prompt_adapter_request,
            streaming=streaming,
            kv_cache_retention_config=kv_cache_retention_config,
            disaggregated_params=disaggregated_params,
            multimodal_params=multimodal_params,
            scheduling_params=scheduling_params,
            cache_salt_id=cache_salt_id,
            arrival_time=arrival_time)
        result = self.submit(request)
        # release memory in time
        if hasattr(request, "multimodal_params"):
            del request.multimodal_params
        return result

    def generate(
        self,
        prompt_token_ids: Union[List[int], List[List[int]]],
        sampling_params: Union[SamplingParams, List[SamplingParams]],
        query_token_ids: Optional[Union[torch.Tensor, np.ndarray, list]] = None,
        lora_request: Optional[Union[LoRARequest, List[LoRARequest]]] = None,
        prompt_adapter_request: Optional[Union[
            PromptAdapterRequest, List[PromptAdapterRequest]]] = None,
        disaggregated_params: Optional[DisaggregatedParams] = None,
    ) -> Union[GenerationResult, List[GenerationResult]]:
        """Generate output for the given prompt token ids in the synchronous mode.
        Synchronous generation accepts either single prompt or batched prompts.
        """
        unbatched = isinstance(prompt_token_ids[0], int)

        if unbatched:
            prompt_token_ids = [prompt_token_ids]
            if query_token_ids:
                query_token_ids = [query_token_ids]

        futures = []
        for i, p in enumerate(prompt_token_ids):
            if isinstance(sampling_params, list):
                sp = sampling_params[i]
            else:
                sp = sampling_params
            if isinstance(lora_request, list):
                lora_req = lora_request[i]
            else:
                lora_req = lora_request
            if isinstance(prompt_adapter_request, list):
                pa_req = prompt_adapter_request[i]
            else:
                pa_req = prompt_adapter_request
            future = self.generate_async(
                p,
                sampling_params=sp,
                query_token_ids=query_token_ids,
                lora_request=lora_req,
                prompt_adapter_request=pa_req,
                streaming=False,
                disaggregated_params=disaggregated_params)
            futures.append(future)

        for future in futures:
            future.result()

        if unbatched:
            futures = futures[0]

        return futures

    def _get_next_client_id(self):
        # (self._last_client_id + 1) % UINT64_MAX
        self._last_client_id = (self._last_client_id + 1) & ((1 << 64) - 1)
        return self._last_client_id

    def _get_logprob_params(
            self, request: GenerationRequest) -> Optional[LogprobParams]:
        """Store logprobs-related fields from request for the later logprob calculation."""
        logprob_params = None
        if request.sampling_params.logprobs or request.sampling_params.prompt_logprobs:
            logprob_params = LogprobParams(
                logprobs=request.sampling_params.logprobs,
                prompt_logprobs=request.sampling_params.prompt_logprobs,
                # drop logits if users didn't explicitly ask for it, or if it's using PostProcess flow
                drop_context_logits=(
                    not request.sampling_params._need_return_context_logits)
                or self.postproc_config.num_postprocess_workers > 0,
                drop_generation_logits=(
                    not request.sampling_params._need_return_generation_logits)
                or self.postproc_config.num_postprocess_workers > 0)

        return logprob_params

    def _maybe_initialize_iteration_results(self):
        if self._is_llm_executor:
            if self._iter_stats_result is None:
                # singleton to store cpp runtime stats
                self._iter_stats_result = IterationResult()
            else:
                # expect more engine stats whenever new prompts are submitted
                self._iter_stats_result.mark_undone()

            if self._iter_kv_events_result is None:
                self._iter_kv_events_result = IterationResult()
            else:
                self._iter_kv_events_result.mark_undone()

    def _handle_background_error(self, error: Optional[Exception | str] = None):
        """ Process the errors from the threads or processes.
        NOTE: This should be called in the main thread.
        """
        if error is not None:
            # For details please refer to the comment of `GenerationResult.error`

            if isinstance(error, RequestError):
                # A per-request error, can be captured and ignored
                if enable_llm_debug():
                    print_colored(f"Got per-request error: {repr(error)}\n",
                                  "red")
            elif isinstance(error, str):
                # A per-request error, can be captured and ignored
                if enable_llm_debug():
                    print_colored(f"Got per-request error: {repr(error)}\n",
                                  "red")
                    print_colored(str(traceback.extract_stack()) + "\n", "red")
                error = RequestError(error)
            else:
                # Serious error from background thread or process
                if not isinstance(error, BaseException):
                    error = RuntimeError(repr(error))
                if enable_llm_debug():
                    print_colored(
                        f"Got background error: {repr(error)}, will shutdown the LLM instance\n",
                        "red")
                self.shutdown()
            raise error

        # Here we raise the first error in the queue. This method will be called repeatedly and user can choose to catch
        # more than one error.
        if not self._error_queue.empty():
            e = self._error_queue.get()
            self._error_queue.task_done()
            self.shutdown()
            # We can catch some exceptions here.
            raise e

    def is_shutdown(self) -> bool:
        return self.doing_shutdown

    @abstractmethod
    def shutdown(self):
        pass

    @property
    def enable_postprocess_parallel(self) -> bool:
        return self.postproc_config.enabled

    def get_stats(self, timeout: float) -> List[dict]:
        """
        Get iteration statistics from the runtime.
        Args:
            timeout (float): Max wait time in seconds when retrieving stats from queue.
        Returns:
            List[dict]: A list of runtime stats as dict.
        """
        if self._iter_stats_result is None:
            print_colored(
                "Iteration statistics are not available yet. To collect runtime statistics, please call get_stats() AFTER prompts have been submitted.\n",
                "yellow")
            return []

        self._iter_stats_result.set_timeout(timeout)
        return self._iter_stats_result.get_results()

    def aget_stats(self, timeout: float) -> IterationResult:
        """
        Get iteration statistics from the runtime.
        Returns:
            IterationResult: An async iterable object containing runtime stats.
        """
        if self._iter_stats_result is None:
            print_colored(
                "Iteration statistics are not available yet. To collect runtime statistics, please call get_stats_async() in async coroutine or the /metrics endpoint (if you're using trtllm-serve) AFTER prompts have been submitted.\n",
                "yellow")
            return empty_async_iterable()

        self._iter_stats_result.set_timeout(timeout)
        return self._iter_stats_result

    def get_kv_events(self, timeout: float) -> List[dict]:
        """
        Get iteration kv events from the runtime.
        Args:
            timeout (float): Max wait time in seconds when retrieving stats from queue.
        Returns:
            List[dict]: A list of runtime events as dict.
        """
        assert self._iter_kv_events_result is not None, "KV Event IterationResult is not properly instantiated."

        self._iter_kv_events_result.set_timeout(timeout)
        return self._iter_kv_events_result.get_results()

    def aget_kv_events(self, timeout=None) -> IterationResult:
        """
        Get iteration kv events from the runtime.
        Args:
            timeout (float): Max wait time in seconds when retrieving stats from queue.
        Returns:
            IterationResult: An async iterable object containing runtime events.
        """
        assert self._iter_kv_events_result is not None, "KV Event IterationResult is not properly instantiated."

        self._iter_kv_events_result.set_timeout(timeout)
        return self._iter_kv_events_result

    @staticmethod
    def _create_ray_executor(
        worker_kwargs: Dict,
        model_world_size: int,
        postproc_worker_config: PostprocWorkerConfig,
        is_llm_executor: bool,
        tp_size: int,
    ):
        from .ray_executor import RayExecutor

        return RayExecutor(worker_kwargs,
                           model_world_size=model_world_size,
                           postproc_worker_config=postproc_worker_config,
                           is_llm_executor=is_llm_executor,
                           tp_size=tp_size)

    @staticmethod
    def _create_rpc_executor(
        worker_kwargs: Dict,
        model_world_size: int,
        mpi_session: Optional[MpiSession],
        postproc_worker_config: PostprocWorkerConfig,
        is_llm_executor: bool,
    ):
        """Create RPC-based executor (GenerationExecutorRpcProxy)."""
        from .rpc_proxy import GenerationExecutorRpcProxy
        return GenerationExecutorRpcProxy(
            worker_kwargs,
            model_world_size=model_world_size,
            mpi_session=mpi_session,
            postproc_worker_config=postproc_worker_config,
            is_llm_executor=is_llm_executor)

    @staticmethod
    def _create_ipc_executor(
        worker_kwargs: Dict,
        model_world_size: int,
        mpi_session: Optional[MpiSession],
        postproc_worker_config: PostprocWorkerConfig,
        is_llm_executor: bool,
        use_worker: bool = False,
    ):
        """Create IPC-based executor (GenerationExecutorProxy or GenerationExecutorWorker).

        Args:
            use_worker: If True, creates GenerationExecutorWorker (single process).
                       If False, creates GenerationExecutorProxy (multi-process with IPC).
        """
        if use_worker:
            from .worker import GenerationExecutorWorker
            return GenerationExecutorWorker(**worker_kwargs,
                                            is_llm_executor=is_llm_executor)
        else:
            from .proxy import GenerationExecutorProxy
            return GenerationExecutorProxy(
                worker_kwargs,
                model_world_size=model_world_size,
                mpi_session=mpi_session,
                postproc_worker_config=postproc_worker_config,
                is_llm_executor=is_llm_executor)

    @staticmethod
    def create(
        engine: Union[Path, Engine],
        executor_config: Optional[tllm.ExecutorConfig] = None,
        batched_logits_processor: Optional[BatchedLogitsProcessor] = None,
        model_world_size: int = 1,
        world_size: int = 0,
        mpi_session: Optional[MpiSession] = None,
        reuse_mpi_comm: bool = False,
        return_logits: bool = False,
        postproc_worker_config: Optional[PostprocWorkerConfig] = None,
        is_llm_executor: Optional[bool] = None,
        hf_model_dir: Optional[Path] = None,
        tokenizer: Optional[TokenizerBase] = None,
        llm_args: Optional[BaseLlmArgs] = None,
        **args,
    ) -> Union["GenerationExecutorProxy", "GenerationExecutorWorker"]:
        if world_size == 0:
            world_size = mpi_world_size()

        if world_size > 1 and world_size < model_world_size:
            raise RuntimeError(
                "Cannot instantiate Generator for engine built "
                f"for {model_world_size} ranks, while currently running "
                f"on {world_size} ranks.")

        postproc_worker_config = postproc_worker_config or PostprocWorkerConfig(
        )

        if postproc_worker_config.enabled:
            logger_debug(
                f"Using {postproc_worker_config.num_postprocess_workers} postprocess parallel processes.\n",
                "green")

        worker_kwargs = {
            "engine": engine,
            "executor_config": executor_config,
            "batched_logits_processor": batched_logits_processor,
            "hf_model_dir": hf_model_dir,
            "tokenizer": tokenizer,
            "llm_args": llm_args,
        }

        orchestrator_type = None if not isinstance(
            llm_args, TorchLlmArgs) else llm_args.orchestrator_type
        if orchestrator_type == "ray":
            if llm_args and hasattr(llm_args, 'ray_worker_extension_cls'):
                worker_kwargs[
                    "ray_worker_extension_cls"] = llm_args.ray_worker_extension_cls
            return GenerationExecutor._create_ray_executor(
                worker_kwargs,
                model_world_size,
                postproc_worker_config,
                is_llm_executor=is_llm_executor,
                tp_size=args.get("tp_size", 1))
        elif orchestrator_type is not None and orchestrator_type != "rpc":
            raise ValueError(
                f"Unsupported orchestrator_type: {orchestrator_type}")

        # The case where the Python main process is launched by mpirun
        mpirun_launch = external_mpi_comm_available(model_world_size)
        # The case where the Python main process utilizes mpi4py to spawn MPI workers
        spawn_workers = need_spawn_mpi_workers(model_world_size)
        orchestrator_is_rpc = llm_args and llm_args.orchestrator_type == "rpc"

        if spawn_workers or (mpirun_launch and reuse_mpi_comm):
            if reuse_mpi_comm:
                assert mpi_session is not None, "reuse_mpi_comm requires an external MPI session"

            if orchestrator_is_rpc:
                return GenerationExecutor._create_rpc_executor(
                    worker_kwargs,
                    model_world_size=model_world_size,
                    mpi_session=mpi_session,
                    postproc_worker_config=postproc_worker_config,
                    is_llm_executor=is_llm_executor)

            return GenerationExecutor._create_ipc_executor(
                worker_kwargs,
                model_world_size=model_world_size,
                mpi_session=mpi_session,
                postproc_worker_config=postproc_worker_config,
                is_llm_executor=is_llm_executor,
                use_worker=False)

        # WAR: For the performance of gathering logits, we use single process worker
        # for TP1 to avoid the large overhead of IPC.
        # WAR: Developers can enable this manually, this will be easier for TP1
        # debugging. We will introduce a better solution in the future.
        if return_logits or enable_worker_single_process_for_tp1():
            logger.warning(
                "Using single process worker for TP1, this may hurt streaming generation performance."
            )
            if orchestrator_is_rpc:
                return GenerationExecutor._create_rpc_executor(
                    worker_kwargs,
                    model_world_size=model_world_size,
                    mpi_session=mpi_session,
                    postproc_worker_config=postproc_worker_config,
                    is_llm_executor=is_llm_executor)

            return GenerationExecutor._create_ipc_executor(
                worker_kwargs,
                model_world_size=model_world_size,
                mpi_session=mpi_session,
                postproc_worker_config=postproc_worker_config,
                is_llm_executor=is_llm_executor,
                use_worker=True)

        # For single-gpu case:
        # Partition the workload to multiple process for streaming performance.
        # While this requires uses to protect their entrypoint to
        # `if __name__ == "__main__":`.
        if not platform.system() == 'Windows':
            if orchestrator_is_rpc:
                return GenerationExecutor._create_rpc_executor(
                    worker_kwargs,
                    model_world_size=model_world_size,
                    mpi_session=mpi_session,
                    postproc_worker_config=postproc_worker_config,
                    is_llm_executor=is_llm_executor)

            return GenerationExecutor._create_ipc_executor(
                worker_kwargs,
                model_world_size=model_world_size,
                mpi_session=None,  # use mpi4py
                postproc_worker_config=postproc_worker_config,
                is_llm_executor=is_llm_executor,
                use_worker=False)
        else:
            ctx = multiprocessing.get_context("spawn")
            # The ProcessPoolExecutorSession is used to support Windows, as mpi4py cannot.
            mpi_session = ProcessPoolExecutorSession(n_workers=1,
                                                     mp_context=ctx)
            # TODO: add rpc worker here
            return GenerationExecutor._create_ipc_executor(
                worker_kwargs,
                model_world_size=model_world_size,
                mpi_session=mpi_session,
                postproc_worker_config=postproc_worker_config,
                is_llm_executor=is_llm_executor,
                use_worker=False)

    def wait_first_completed(
        self, futures: List[GenerationResult]
    ) -> Generator[GenerationResult, None, None]:
        wait_set = set(futures)

        # clear already-finished requests
        for f in futures:
            if f._done:
                wait_set.pop(f)
                yield f

        # wait remaining active requests
        while len(wait_set) > 0:
            fut = wait_set.pop()
            if fut.request_id not in self._results:
                yield fut
            else:
                wait_set.add(fut)


if enable_llm_debug():
    print_colored("LLM debug mode enabled.\n", "yellow")
    # This will dump all the alive threads when the process is interrupted by SIGINT.
    faulthandler.register(signal.SIGINT, all_threads=True)
