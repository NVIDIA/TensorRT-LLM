import asyncio
import atexit
import faulthandler
import multiprocessing
import platform
import signal
import traceback
from abc import ABC, abstractmethod
from pathlib import Path
from queue import Empty, Queue
from typing import (TYPE_CHECKING, Callable, Dict, Generator, List, Optional,
                    Union)

import numpy as np
import torch

from tensorrt_llm.logger import logger, set_level

from .._utils import mpi_world_size
from ..bindings import executor as tllm
from ..builder import Engine
from ..llmapi.llm_utils import KvCacheRetentionConfig
from ..llmapi.mpi_session import (MpiSession, external_mpi_comm_available,
                                  need_spawn_mpi_workers)
from ..llmapi.utils import (AsyncQueue, enable_llm_debug,
                            enable_worker_single_process_for_tp1, print_colored,
                            print_colored_debug)
from ..sampling_params import SamplingParams
from .postproc_worker import PostprocWorkerConfig
from .request import GenerationRequest, LoRARequest, PromptAdapterRequest
from .result import GenerationResult
from .utils import ProcessPoolExecutorSession, RequestError, has_event_loop

if TYPE_CHECKING:
    from .proxy import ExecutorBindingsProxy
    from .worker import ExecutorBindingsWorker

__all__ = [
    "GenerationExecutor",
    "NoStatsAvailable",
    "CppExecutorError",
]

if enable_llm_debug():
    # Mainly enable more detailed logging from cpp runtime.
    set_level("info")


class CppExecutorError(RuntimeError):

    def __init__(self, message: Optional[str] = None):
        self.message = message
        self.stack_trace = traceback.format_exc()
        super().__init__(message)

    def __str__(self):
        return f"{self.message}\nStack trace:\n{self.stack_trace}"


class NoStatsAvailable(Exception):
    pass


class GenerationExecutor(ABC):

    def __init__(self,
                 num_postprocess_workers: int = 0,
                 postprocess_tokenizer_dir: Optional[str] = None):
        self.postproc_config = PostprocWorkerConfig(
            num_postprocess_workers=num_postprocess_workers,
            postprocess_tokenizer_dir=postprocess_tokenizer_dir)

        self._stats = None
        self.stats_queue = None

        atexit.register(self.shutdown)

        # This is used to capture the exceptions from the threads.
        self._error_queue = Queue()

        # A flag to avoid calling shutdown() recursively. This happens when the background threads raise errors.
        self.doing_shutdown = False

        self._last_client_id: int = 1

    @abstractmethod
    def submit(self, request: GenerationRequest) -> GenerationResult:
        pass

    def generate_async(
        self,
        prompt_token_ids: List[int],
        sampling_params: SamplingParams,
        query_token_ids: Optional[Union[torch.Tensor, np.ndarray, list]] = None,
        lora_request: Optional[LoRARequest] = None,
        prompt_adapter_request: Optional[PromptAdapterRequest] = None,
        streaming: bool = False,
        prompt_tuning_config: Optional[list] = None,
        kv_cache_retention_config: Optional[KvCacheRetentionConfig] = None
    ) -> GenerationResult:
        """Generate output for the given prompt token ids in the asynchronous mode.
        Asynchronous generation accepts single prompt only.
        """
        assert isinstance(prompt_token_ids[0], int)
        assert isinstance(sampling_params, SamplingParams)
        result = self.submit(
            GenerationRequest(
                prompt_token_ids,
                sampling_params=sampling_params,
                query_token_ids=query_token_ids,
                lora_request=lora_request,
                prompt_adapter_request=prompt_adapter_request,
                streaming=streaming,
                prompt_tuning_config=prompt_tuning_config,
                kv_cache_retention_config=kv_cache_retention_config))
        return result

    def generate(
        self,
        prompt_token_ids: Union[List[int], List[List[int]]],
        sampling_params: Union[SamplingParams, List[SamplingParams]],
        query_token_ids: Optional[Union[torch.Tensor, np.ndarray, list]] = None,
        lora_request: Optional[Union[LoRARequest, List[LoRARequest]]] = None,
        prompt_adapter_request: Optional[Union[
            PromptAdapterRequest, List[PromptAdapterRequest]]] = None,
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
            future = self.generate_async(p,
                                         sampling_params=sp,
                                         query_token_ids=query_token_ids,
                                         lora_request=lora_req,
                                         prompt_adapter_request=pa_req,
                                         streaming=False)
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

    def _handle_background_error(self, error: Optional[Exception | str] = None):
        """ Process the errors from the threads or processes.
        NOTE: This should be called in the main thread.
        """
        if error is not None:
            # For details please refer to the comment of `GenerationResult.error`
            if isinstance(error, Exception):
                # Serious error from background thread or process
                if enable_llm_debug():
                    print_colored(
                        f"Got background error: {repr(error)}, will shutdown the LLM instance\n",
                        "red")
                self.shutdown()
                raise error
            elif isinstance(error, str):
                if enable_llm_debug():
                    print_colored(f"Got per-request error: {repr(error)}\n",
                                  "red")
                    print_colored(str(traceback.extract_stack()) + "\n", "red")
                # A per-request error, can be captured and ignored
                raise RequestError(error)

        # Here we raise the first error in the queue. This method will be called repeatedly and user can choose to catch
        # more than one error.
        if not self._error_queue.empty():
            e = self._error_queue.get()
            self._error_queue.task_done()
            self.shutdown()
            # We can catch some exceptions here.
            raise e

    @abstractmethod
    def shutdown(self):
        pass

    @property
    def enable_postprocess_parallel(self) -> bool:
        return self.postproc_config.enabled

    def create_stats_queue(self):
        # Stats queue is created during first submission to ensure event loop exists if it is needed.
        if not self._stats:
            if has_event_loop():
                self._stats = AsyncQueue()
                self.stats_queue = self._stats.sync_q
                self.stats_aqueue = self._stats
            else:
                self._stats = Queue()
                self.stats_queue = self._stats
                self.stats_aqueue = None

    def get_stats(self, timeout=None) -> str:
        ''' Get the stats from the runtime.

        Exceptions:
            NoStatsAvailable: If the stats are not available.

        Returns:
            str: The stats in JSON format.

        Known issue:
            The `get_stats` cannot mix with `aget_stats` in the same Executor instance.
        '''
        assert self.stats_queue, "The stats queue is not created. It is likely that `get_stats` and `aget_stats` methods" \
            " are mixed."
        try:
            res = self.stats_queue.get(timeout=timeout)
        except Empty:
            raise NoStatsAvailable
        return res

    async def aget_stats(self, timeout=None) -> Optional[str]:
        ''' Get the stats from the runtime.

        Exceptions:
            NoStatsAvailable: If the stats are not available.

        Returns:
            str: The stats in JSON format.

        Known issue:
            The `aget_stats` cannot mix with `get_stats` in the same Executor instance.
        '''
        self.create_stats_queue()
        assert self.stats_aqueue is not None

        if not has_event_loop():
            raise NoStatsAvailable

        try:
            res = await self.stats_aqueue.get(timeout=timeout)
        except asyncio.TimeoutError:
            raise NoStatsAvailable
        return res

    @staticmethod
    def create(
        engine: Union[Path, Engine],
        executor_config: Optional[tllm.ExecutorConfig] = None,
        logits_post_processor_map: Optional[Dict[str, Callable]] = None,
        model_world_size: int = 1,
        world_size: int = 0,
        mpi_session: Optional[MpiSession] = None,
        reuse_mpi_comm: bool = False,
        return_logits: bool = False,
        postproc_worker_config: Optional[PostprocWorkerConfig] = None,
    ) -> Union["ExecutorBindingsProxy", "ExecutorBindingsWorker"]:
        # local imports to avoid cyclic importing
        from .proxy import ExecutorBindingsProxy
        from .worker import ExecutorBindingsWorker

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
            print_colored_debug(
                f"Using {postproc_worker_config.num_postprocess_workers} postprocess parallel processes.\n",
                "green")

        worker_kwargs = {
            "engine": engine,
            "executor_config": executor_config,
            "logits_post_processor_map": logits_post_processor_map,
        }

        # The case where the Python main process is launched by mpirun
        mpirun_launch = external_mpi_comm_available(model_world_size)
        # The case where the Python main process utilizes mpi4py to spawn MPI workers
        spawn_workers = need_spawn_mpi_workers(model_world_size)
        if spawn_workers or (mpirun_launch and reuse_mpi_comm):
            if reuse_mpi_comm:
                assert mpi_session is not None, "reuse_mpi_comm requires an external MPI session"
            return ExecutorBindingsProxy(
                worker_kwargs,
                model_world_size=model_world_size,
                mpi_session=mpi_session,
                postproc_worker_config=postproc_worker_config)

        # WAR: For the performance of gathering logits, we use single process worker
        # for TP1 to avoid the large overhead of IPC.
        # WAR: Developers can enable this manually, this will be easier for TP1
        # debugging. We will introduce a better solution in the future.
        if return_logits or enable_worker_single_process_for_tp1():
            logger.warning(
                "Using single process worker for TP1, this may hurt streaming generation performance."
            )
            return ExecutorBindingsWorker(**worker_kwargs)

        # For single-gpu case:
        # Partition the workload to multiple process for streaming performance.
        # While this requires uses to protect their entrypoint to
        # `if __name__ == "__main__":`.
        if not platform.system() == 'Windows':
            return ExecutorBindingsProxy(
                worker_kwargs,
                model_world_size=model_world_size,
                mpi_session=None,  # use mpi4py
                postproc_worker_config=postproc_worker_config,
            )
        else:
            ctx = multiprocessing.get_context("spawn")
            # The ProcessPoolExecutorSession is used to support Windows, as mpi4py cannot.
            mpi_session = ProcessPoolExecutorSession(n_workers=1,
                                                     mp_context=ctx)
            return ExecutorBindingsProxy(
                worker_kwargs,
                model_world_size=model_world_size,
                mpi_session=mpi_session,
                postproc_worker_config=postproc_worker_config,
            )

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
