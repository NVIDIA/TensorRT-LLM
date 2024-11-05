import asyncio
import atexit
import concurrent.futures
import copy
import datetime
import faulthandler
import io
import json
import os
import pickle  # nosec B403
import secrets
import signal
import time
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path
from queue import Empty, Queue
from typing import (Any, Callable, Dict, Generator, List, Literal, NamedTuple,
                    Optional, Tuple, Union)
from weakref import WeakMethod

import numpy as np
import torch
import zmq

from ._utils import mpi_rank, mpi_world_size
from .bindings import executor as tllm
from .builder import ConfigEncoder, Engine, EngineConfig
from .llmapi.mpi_session import (MpiPoolSession, MpiSession,
                                 external_mpi_comm_available, find_free_port,
                                 need_spawn_mpi_workers)
from .llmapi.tracer import (VizTracer, enable_llm_tracer, get_tracer,
                            global_tracer, set_global_tracer)
from .llmapi.utils import (AsyncQueue, ManagedThread, SamplingParams,
                           _SyncQueue, enable_llm_debug, print_colored)
from .lora_manager import LoraManager
from .prompt_adapter_manager import PromptAdapterManager
from .runtime.model_runner import _engine_config_to_model_config


@dataclass(slots=True)
class LoRARequest:
    """ Request for a LoRA adapter. """
    lora_name: str
    lora_int_id: int
    lora_path: str = ""

    def __post_init__(self):
        if not os.path.exists(self.lora_path):
            raise ValueError(f"lora_path ({self.lora_path}) does not exist.")

    @property
    def adapter_id(self):
        return self.lora_int_id

    @property
    def name(self):
        return self.lora_name

    @property
    def path(self):
        return self.lora_path


@dataclass(slots=True)
class PromptAdapterRequest:
    """
    Request for a Prompt adapter.
    """
    prompt_adapter_name: str
    prompt_adapter_id: int
    prompt_adapter_local_path: str = ""

    def __post_init__(self):
        if not os.path.exists(self.prompt_adapter_local_path):
            raise RuntimeError(
                f"prompt_adapter_local_path ({self.prompt_adapter_local_path}) does not exist."
            )

    @property
    def adapter_id(self):
        return self.prompt_adapter_id

    @property
    def name(self):
        return self.prompt_adapter_name

    @property
    def local_path(self):
        return self.prompt_adapter_local_path


class GenerationRequest:

    def __init__(
        self,
        prompt_token_ids: Union[torch.Tensor, np.ndarray,
                                Union[List[int], List[List[int]]]],
        sampling_params: SamplingParams,
        lora_request: Optional[LoRARequest] = None,
        prompt_adapter_request: Optional[PromptAdapterRequest] = None,
        streaming: bool = False,
    ):
        if isinstance(prompt_token_ids, list):
            self.prompt_token_ids = prompt_token_ids
        elif isinstance(prompt_token_ids, (torch.Tensor, np.ndarray)):
            self.prompt_token_ids = prompt_token_ids.tolist()
        else:
            raise TypeError(
                f"prompt_token_ids ({prompt_token_ids}) should be an instance of torch.Tensor, np.ndarray or list"
            )

        self.sampling_params = sampling_params
        self.lora_request = lora_request
        self.prompt_adapter_request = prompt_adapter_request
        self.streaming = streaming
        self.id = -1

    def set_id(self, id):
        self.id = id
        return self


@dataclass(slots=True)
class CompletionOutput:
    """The output data of one completion output of a request.

    Args:
        index (int): The index of the output in the request.
        text (str): The generated output text. Defaults to "".
        token_ids (List[int]): The token ids of the generated output text. Defaults to [].
        cumulative_logprob (float, optional): The cumulative log probability of the generated output text. Defaults to None.
        logprobs (List[float]): The log probabilities of the top probability words at each position if the logprobs are requested. Defaults to [].
        finish_reason (Literal['stop', 'length'], optional): The reason why the sequence is finished. Defaults to None.
        stop_reason (int, str, optional): The stop string or token id that caused the completion to stop, None if the completion finished for some other reason. Defaults to None.
        generation_logits (torch.Tensor, optional): The logits on the generated output token ids. Defaults to None.

    Properties:
        length (int): The number of generated tokens.
        token_ids_diff (List[int]): Newly generated token ids.
        logprobs_diff (List[float]): Logprobs of newly generated tokens.
        text_diff (str): Newly generated tokens.
    """
    index: int
    text: str = ""
    token_ids: List[int] = field(default_factory=list)
    cumulative_logprob: Optional[float] = None
    logprobs: List[float] = field(default_factory=list)
    finish_reason: Optional[Literal['stop', 'length']] = None
    stop_reason: Optional[Union[int, str]] = None
    generation_logits: Optional[torch.Tensor] = None

    # hidden fields for tracking the diffs
    _last_text_len: int = field(default=0, init=False, repr=False)
    _last_token_ids_len: int = field(default=0, init=False, repr=False)
    _last_logprobs_len: int = field(default=0, init=False, repr=False)
    _incremental_states: Optional[dict] = field(default=None,
                                                init=False,
                                                repr=False)

    @property
    def length(self):
        return len(self.token_ids)

    @property
    def text_diff(self) -> str:
        return self.text[self._last_text_len:]

    @property
    def token_ids_diff(self) -> List[int]:
        return self.token_ids[self._last_token_ids_len:]

    @property
    def logprobs_diff(self) -> List[float]:
        return self.logprobs[self._last_logprobs_len:]


class CppExecutorError(RuntimeError):

    def __init__(self, message: Optional[str] = None):
        self.message = message
        self.stack_trace = traceback.format_exc()
        super().__init__(message)

    def __str__(self):
        return f"{self.message}\nStack trace:\n{self.stack_trace}"


class RequestError(RuntimeError):
    ''' The error raised when the request is failed. '''


class GenerationResult:
    '''
    The result of a generation request. It can be used to wait for the completion of the request.

    Args:
        generation_request (GenerationRequest): The generation request object.
        background_error_handler (Callable, optional): The error handler to process the errors from the background threads/processes. Defaults to None.
    '''

    def __init__(self,
                 generation_request: GenerationRequest,
                 background_error_handler: Optional[Callable] = None) -> None:
        self._done = False
        self._cancelled = False
        self._generation_request = generation_request

        if has_event_loop():
            aqueue = AsyncQueue()
            self.queue = aqueue.sync_q
            self.aqueue = aqueue.async_q
        else:
            self.queue = Queue()
            self.aqueue = None

        # In Sampling mode, the Executor runtime will return best_of sequences
        # in total, which the LLM API will select the n-best sequences among
        # them based on their cumulative log probabilities.
        self._outputs: List[CompletionOutput] = [
            CompletionOutput(i)
            for i in range(self._generation_request.sampling_params.best_of)
        ]
        self.context_logits: Optional[torch.Tensor] = None

        self._background_error_handler = None
        if background_error_handler is not None:
            self._background_error_handler = WeakMethod(
                background_error_handler)

    @property
    def request_id(self) -> int:
        return self._generation_request.id

    @property
    def prompt_token_ids(self) -> List[int]:
        return self._generation_request.prompt_token_ids

    @property
    def finished(self) -> bool:
        return self._done

    @property
    def streaming(self):
        return self._generation_request.streaming

    @property
    def outputs(self) -> List[CompletionOutput]:
        sampling_param = self._generation_request.sampling_params
        if (sampling_param.use_beam_search
                or sampling_param.n == sampling_param.best_of):
            return self._outputs[:sampling_param.n]
        # Pick the top-n outputs, sorted by cumulative log probs.
        sorted_outputs = sorted(
            self._outputs,
            key=lambda x:
            (x.cumulative_logprob
             if x.cumulative_logprob is not None else float('-inf')),
            reverse=True)
        # Reindex the sequence.
        for i, sorted_out in enumerate(sorted_outputs):
            sorted_out.index = i
        return sorted_outputs[:sampling_param.n]

    def handle_sequence(self, response: "GenerationExecutor.Response",
                        sequence_index: int):
        """ Handle a single sequence in the response. """

        tensors = response.tensors
        assert tensors is not None

        beam_search = self._generation_request.sampling_params.use_beam_search
        seq_idx = sequence_index
        src_idx = sequence_index if beam_search else 0

        output = self._outputs[seq_idx]

        output._last_token_ids_len = len(output.token_ids)
        output.token_ids.extend(tensors.output_token_ids[src_idx])
        if tensors.cum_log_probs is not None:
            output.cumulative_logprob = tensors.cum_log_probs[src_idx]
        if tensors.log_probs is not None:
            output._last_logprobs_len = len(output.logprobs)
            output.logprobs = tensors.log_probs[src_idx]
            assert len(output.logprobs) == output.length
        if tensors.generation_logits is not None:
            output.generation_logits = tensors.generation_logits[
                src_idx, :output.length]

        if self.finished:
            if response.finish_reasons[src_idx] == tllm.FinishReason.END_ID:
                output.finish_reason = 'stop'
            elif response.finish_reasons[
                    src_idx] == tllm.FinishReason.STOP_WORDS:
                output.finish_reason = 'stop'
                sampling_params = self._generation_request.sampling_params
                for stop_reason, stop_ids in sampling_params._get_stop_reasons_and_words(
                ):
                    if output.token_ids[-len(stop_ids):] == stop_ids:
                        output.stop_reason = stop_reason
                        if not sampling_params.include_stop_str_in_output:
                            output.token_ids = output.token_ids[:-len(stop_ids)]
                        break
            elif response.finish_reasons[src_idx] == tllm.FinishReason.LENGTH:
                output.finish_reason = 'length'

    def handle_response(self, response: "GenerationExecutor.Response"):

        self._done = response.is_final

        if response.error:
            if handler := self._background_error_handler():
                handler(response.error)

        tensors = response.tensors

        # output_token_ids = (beams, tokens)
        if self._generation_request.sampling_params.use_beam_search:
            for beam_idx, _ in enumerate(tensors.output_token_ids):
                self.handle_sequence(response, beam_idx)
        else:
            self.handle_sequence(response, response.sequence_index)

        if tensors.context_logits is not None:
            self.context_logits = tensors.context_logits

        # Processing background errors here ASAF during generation.
        if self._background_error_handler and (
                handler := self._background_error_handler()):
            handler()

    def result_step(self, timeout: Optional[float] = None):
        response = self.queue.get(timeout=timeout)
        self.handle_response(response)

    async def aresult_step(self):
        assert self.aqueue is not None, "The asyncio event loop was not present during initialization, so async operations are not available."
        response = await self.aqueue.get()
        global_tracer().log_instant("result_step.get")
        self.handle_response(response)

    def result(self, timeout: Optional[float] = None) -> "GenerationResult":
        while not self._done:
            self.result_step(timeout)
        return self

    async def aresult(self) -> "GenerationResult":
        while not self._done:
            await self.aresult_step()
        return self

    def __await__(self):
        return self.aresult().__await__()

    def __iter__(self):
        return self

    def __next__(self):
        if self._done:
            raise StopIteration

        self.result_step()
        return self

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._done:
            raise StopAsyncIteration

        await self.aresult_step()
        return self

    def running(self) -> bool:
        return not self._done

    def cancelled(self) -> bool:
        return self._cancelled

    def cancel(self):
        raise NotImplementedError

    def done(self) -> bool:
        return self._done

    def exception(self, timeout: Optional[float] = None):
        try:
            self.result(timeout)
        except RuntimeError as e:
            return e

    def _repr_fields(self):
        return [
            'request_id', 'prompt_token_ids', 'outputs', 'finished',
            "context_logits"
        ]

    def __repr__(self) -> str:
        repr = []
        for field in self._repr_fields():
            value = getattr(self, field)
            if isinstance(value, str):
                repr.append(f"{field}={value!r}")
            else:
                repr.append(f"{field}={value}")
        repr = ", ".join(repr)
        repr = f"{self.__class__.__name__}({repr})"
        return repr

    def __hash__(self):
        return hash(self.request_id)


class NoStatsAvailable(Exception):
    pass


class GenerationExecutor(ABC):

    PENDING_REQ_ID_TIMEOUT = 2  # second

    class ResponseTensors(NamedTuple):
        output_token_ids: List[List[int]]
        # context_logits is a tensor or a string denoting the path to the shared memory.
        context_logits: Optional[torch.Tensor | str]
        # generation_logits is a tensor or a string denoting the path to the shared memory.
        generation_logits: Optional[torch.Tensor | str]
        log_probs: Optional[list]
        cum_log_probs: Optional[list]

    class Response(NamedTuple):
        """ The response from the cpp-executor to the Python main thread. """
        request_id: int
        tensors: Optional["GenerationExecutor.ResponseTensors"]
        finish_reasons: Optional[List[tllm.FinishReason]]
        is_final: Optional[bool]
        sequence_index: Optional[int]
        # There are two types of errors:
        # 1. str for the errors from the cpp-executor.await_responses, this will be dispatched to the user's
        #    generate_async as a per-request error, and won't stop the whole service.
        # 2. Exception for the errors from the background threads/processes, this will be processed in the main thread,
        #    and stop the whole service.
        error: Optional[str | Exception]

    @dataclass(slots=True)
    class PendingResponse:
        response: "GenerationExecutor.Response"
        start_time: float  # this is used to track the latency before the response is dispatched.

    def __init__(self):
        self._stats = None
        self.stats_queue = None

        atexit.register(self.shutdown)

        # This is used to capture the exceptions from the threads.
        self._error_queue = Queue()

        # mapping of pending request_id -> response
        self._pending_responses: Dict[
            int, List[GenerationExecutor.PendingResponse]] = {}

        # A flag to avoid calling shutdown() recursively. This happens when the background threads raise errors.
        self.doing_shutdown = False

    @abstractmethod
    def submit(self, request: GenerationRequest) -> GenerationResult:
        pass

    def generate_async(
        self,
        prompt_token_ids: List[int],
        sampling_params: SamplingParams,
        lora_request: Optional[LoRARequest] = None,
        prompt_adapter_request: Optional[PromptAdapterRequest] = None,
        streaming: bool = False,
    ) -> GenerationResult:
        """Generate output for the given prompt token ids in the asynchronous mode.
        Asynchronous generation accepts single prompt only.
        """
        assert isinstance(prompt_token_ids[0], int)
        assert isinstance(sampling_params, SamplingParams)
        result = self.submit(
            GenerationRequest(prompt_token_ids,
                              sampling_params=sampling_params,
                              lora_request=lora_request,
                              prompt_adapter_request=prompt_adapter_request,
                              streaming=streaming))
        return result

    def generate(
        self,
        prompt_token_ids: Union[List[int], List[List[int]]],
        sampling_params: Union[SamplingParams, List[SamplingParams]],
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
                                         lora_request=lora_req,
                                         prompt_adapter_request=pa_req,
                                         streaming=False)
            futures.append(future)

        for future in futures:
            future.result()

        if unbatched:
            futures = futures[0]

        return futures

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

    def _to_delay_response(self,
                           response: "GenerationExecutor.Response") -> bool:
        ''' the engine.enqueue_request may not be finished in another thread, so we need to postpone it. '''
        req_id = response.request_id
        if req_id not in self._results:
            self._pending_responses.setdefault(req_id, []).append(
                self.PendingResponse(response, time.perf_counter()))
            if time.perf_counter() - self._pending_responses[req_id][
                    0].start_time > self.PENDING_REQ_ID_TIMEOUT:
                raise TimeoutError(
                    f"Request ID {req_id} not found in the results queue.")
            return True

        return False

    def _cleanup_pending_responses(self, nowait=False) -> bool:
        ''' Process the pending responses that are not found in the results. '''

        def cleanup():
            done_req_ids = set()
            for req_id, responses in self._pending_responses.items():
                if req_id not in self._results:
                    if time.perf_counter(
                    ) - responses[0].start_time > self.PENDING_REQ_ID_TIMEOUT:
                        raise TimeoutError(
                            f"Request ID {req_id} not found in the results queue."
                        )
                else:
                    is_done = False
                    for response in responses:
                        is_done = is_done or response.response.is_final
                        self._results[req_id].queue.put(
                            response.response)  # dispatch
                    if is_done:
                        done_req_ids.add(req_id)

            for req_id in done_req_ids:
                self._pending_responses.pop(req_id, None)

            return not bool(self._pending_responses)

        if nowait:
            cleanup()
        else:
            # It is possible that some requests are still pending in the workers, we need to process them before shutdown
            for _ in range(int(self.PENDING_REQ_ID_TIMEOUT / 0.1) + 1):
                if cleanup(): break
                time.sleep(0.1)
                # It will raise TimeoutError if the pending responses are not processed in time.

        return not bool(self._pending_responses)

    @abstractmethod
    def shutdown(self):
        pass

    def create_stats_queue(self):
        # Stats queue is created during first submission to ensure event loop exists if it is needed.
        if not self._stats:
            if has_event_loop():
                self._stats = AsyncQueue()
                self.stats_queue = self._stats.sync_q
                self.stats_aqueue = self._stats.async_q
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
        executor_config: tllm.ExecutorConfig = tllm.ExecutorConfig(1),
        model_world_size: int = 1,
        world_size: int = 0,
        mpi_session: Optional[MpiSession] = None,
        reuse_mpi_comm: bool = False,
    ) -> Union["ExecutorBindingsProxy", "ExecutorBindingsWorker"]:

        if world_size == 0:
            world_size = mpi_world_size()

        if world_size > 1 and world_size < model_world_size:
            raise RuntimeError(
                "Cannot instantiate Generator for engine built "
                f"for {model_world_size} ranks, while currently running "
                f"on {world_size} ranks.")

        worker_kwargs = {
            "engine": engine,
            "executor_config": executor_config,
        }

        # The case where the Python main process is launched by mpirun
        mpirun_launch = external_mpi_comm_available(model_world_size)
        # The case where the Python main process utilizes mpi4py to spawn MPI workers
        spawn_workers = need_spawn_mpi_workers(model_world_size)
        if spawn_workers or (mpirun_launch and reuse_mpi_comm):
            if reuse_mpi_comm:
                assert mpi_session is not None, "reuse_mpi_comm requires an external MPI session"
            return ExecutorBindingsProxy(worker_kwargs,
                                         model_world_size=model_world_size,
                                         mpi_session=mpi_session)

        return ExecutorBindingsWorker(**worker_kwargs)


class ExecutorBindingsWorker(GenerationExecutor):

    class WorkerExit(GeneratorExit):
        pass

    def __init__(
        self,
        engine: Union[Path, Engine],
        executor_config: tllm.ExecutorConfig = tllm.ExecutorConfig(1)
    ) -> None:
        super().__init__()

        self.engine = None
        self.result_queue = None
        self.rank = mpi_rank()
        self._results: Dict[int, GenerationResult] = {}

        if isinstance(engine, list):
            engine = engine[self.rank]

        if isinstance(engine, Engine):
            self.engine = tllm.Executor(engine.engine,
                                        json.dumps(engine.config.to_dict(),
                                                   cls=ConfigEncoder),
                                        tllm.ModelType.DECODER_ONLY,
                                        executor_config=executor_config,
                                        managed_weights=engine.managed_weights)
        else:
            self.engine = tllm.Executor(engine,
                                        tllm.ModelType.DECODER_ONLY,
                                        executor_config=executor_config)

        self._lora_manager: Optional[LoraManager] = None
        self._prompt_adapter_manager: Optional[PromptAdapterManager] = None
        self._runtime_model_config: Optional[ModelConfig] = None
        if self.rank == 0:
            if isinstance(engine, Engine):
                engine_config = engine.config
            else:
                engine_config = EngineConfig.from_json_file(
                    f"{engine}/config.json")
            self._runtime_model_config = _engine_config_to_model_config(
                engine_config)
            if engine_config.build_config.plugin_config.lora_plugin:
                self._lora_manager = LoraManager()
            if engine_config.build_config.max_prompt_embedding_table_size > 0:
                self._prompt_adapter_manager = PromptAdapterManager()

        self.await_response_thread = ManagedThread(
            self.await_response_task,
            error_queue=self._error_queue,
            name="await_response_thread")

        self.dispatch_stats_thread = ManagedThread(
            self.dispatch_stats_task,
            error_queue=self._error_queue,
            name="dispatch_stats_thread")

    def create_stats_queue(self):
        # Stats queue is created during first submission to ensure event loop exists if it is needed.
        if not self._stats:
            if has_event_loop():
                self._stats = AsyncQueue()
                self.stats_queue = self._stats.sync_q
                self.stats_aqueue = self._stats.async_q
            else:
                self._stats = Queue()
                self.stats_queue = self._stats
                self.stats_aqueue = None

    def set_result_queue(self, queue):
        """In multi-gpu mode, result_queue will be set here to communicate between the proxy and the worker 0 process."""
        self.result_queue = queue

    def set_stats_queue(self, queue):
        """In multi-gpu mode, stats_queue will be set here to communicate between the proxy and the worker 0 process."""
        self._stats = queue
        self.stats_queue = self._stats
        self.stats_aqueue = None

    def return_queue(self, req_id: int):
        """ If a centralized result queue is registered (used for communication with the proxy)
            send the message there.
            Otherwise, push the result directly in the GenerationResult queue.
        """
        if self.result_queue is not None:
            return self.result_queue
        return self._results[req_id].queue

    def start_awaiter_thread(self):
        if self.engine.can_enqueue_requests(
        ) and not self.await_response_thread.is_alive():
            self.await_response_thread.start()

    def start_stats_thread(self):
        if self.engine.can_enqueue_requests(
        ) and not self.dispatch_stats_thread.is_alive():
            self.dispatch_stats_thread.start()

    def _engine_response_callback(self, response: tllm.Response):
        return response

    def await_response_task(self) -> bool:
        # Get responses and place in queue.

        async_events = []
        event_loop = None
        for response in self.engine.await_responses(timeout=datetime.timedelta(
                milliseconds=100)):
            response = self._engine_response_callback(response)

            req_id = response.request_id
            if response.has_error():
                # This error will be dispatched to the user's generate_async for the corresponding request. It won't
                # stop the whole service.
                rsp = self.Response(
                    req_id,
                    tensors=None,
                    # Note: error Response only has one finish reason.
                    # Since the error will be raised in the main thread, so the finish reason is not actually used.
                    finish_reasons=[tllm.FinishReason.NOT_FINISHED],
                    is_final=True,
                    sequence_index=None,
                    error=response.error_msg)

            else:
                tensors = self.ResponseTensors(
                    output_token_ids=response.result.output_token_ids,
                    context_logits=response.result.context_logits,
                    generation_logits=response.result.generation_logits,
                    log_probs=response.result.log_probs,
                    cum_log_probs=response.result.cum_log_probs,
                )

                rsp = self.Response(
                    req_id,
                    tensors,
                    finish_reasons=response.result.finish_reasons,
                    is_final=response.result.is_final,
                    sequence_index=response.result.sequence_index,
                    error=None)

            if self._to_delay_response(rsp):
                continue
            self._cleanup_pending_responses(nowait=True)

            queue = self.return_queue(req_id)

            if self._has_background_error():
                rsp = self._create_error_response(req_id)

            # For AsyncQueue.sync_q, we will batch the events to avoid too many event notifications, thus put without
            # wait here.
            if isinstance(queue, _SyncQueue):
                global_tracer().log_instant("worker-rsp.put")
                queue.put_nowait(rsp)
                async_events.append(queue.event)
                event_loop = queue.loop if event_loop is None else event_loop  # all the loops are identical
            else:
                global_tracer().log_instant("worker-rsp.put")
                queue.put(rsp)  # This could be IPC

            # Eliminate the finished GenerationRequest instances timely, which may take considerable memory.
            if rsp.is_final:
                self._results.pop(req_id)

        if async_events:
            _SyncQueue.notify_events(event_loop, async_events)

        return True  # success

    def _has_background_error(self) -> bool:
        return not self._error_queue.empty()

    def _create_error_response(self, req_id) -> GenerationExecutor.Response:
        bck_error = self._error_queue.get_nowait()
        assert isinstance(bck_error, Exception)
        return GenerationExecutor.Response(req_id,
                                           tensors=None,
                                           finish_reasons=None,
                                           is_final=None,
                                           sequence_index=None,
                                           error=bck_error)

    stats_count = 0

    def dispatch_stats_task(self) -> bool:
        time.sleep(0.1)
        # Get stats and place in queue.
        for stats in self.engine.get_latest_iteration_stats():
            self.stats_count += 1
            while hasattr(self.stats_queue, "full") and self.stats_queue.full():
                self.stats_queue.get()

            try:
                self.stats_queue.put(stats.to_json_str())
            except AsyncQueue.EventLoopShutdownError:
                # This happens in the last stats loop while the generate workflow is stopped.
                pass
            except Exception as e:
                raise e

        return True  # success

    def start(self):
        self.create_stats_queue()
        self.start_awaiter_thread()
        self.start_stats_thread()

    def _load_lora_adapter(self, lora_request: LoRARequest):
        self._lora_manager.load_from_ckpt(
            [lora_request.path],
            model_config=self._runtime_model_config,
            runtime_mapping=None,
            uids=[str(lora_request.adapter_id)])

    def _load_prompt_adapter(self,
                             prompt_adapter_request: PromptAdapterRequest):
        self._prompt_adapter_manager.load_from_ckpt(
            [prompt_adapter_request.local_path],
            model_config=self._runtime_model_config,
            uids=[str(prompt_adapter_request.adapter_id)])

    def _enqueue_request(self, request: GenerationRequest) -> int:
        if self._lora_manager is not None and request.lora_request is not None:
            self._load_lora_adapter(request.lora_request)
            uid = str(request.lora_request.adapter_id)
            lora_config = tllm.LoraConfig(
                task_id=request.lora_request.adapter_id,
                weights=self._lora_manager.cpp_lora_weights[uid],
                config=self._lora_manager.cpp_lora_config[uid])
        else:
            lora_config = None

        prompt_token_ids = copy.deepcopy(request.prompt_token_ids)
        if request.prompt_adapter_request is not None:
            self._load_prompt_adapter(request.prompt_adapter_request)
            uid = str(request.prompt_adapter_request.adapter_id)
            prompt_tuning_config = tllm.PromptTuningConfig(
                self._prompt_adapter_manager.uid_to_weights[uid])
            vocab_size = self._runtime_model_config.vocab_size
            pa_length = prompt_tuning_config.embedding_table.size(0)
            prompt_token_ids = list(range(
                vocab_size, vocab_size + pa_length)) + prompt_token_ids
        else:
            prompt_tuning_config = None

        try:
            executor_request = tllm.Request(
                input_token_ids=prompt_token_ids,
                max_tokens=request.sampling_params.max_tokens,
                max_new_tokens=request.sampling_params.max_new_tokens,
                streaming=request.streaming,
                sampling_config=request.sampling_params._get_sampling_config(),
                end_id=-1 if request.sampling_params.ignore_eos else
                request.sampling_params.end_id,
                pad_id=request.sampling_params.pad_id,
                output_config=request.sampling_params._get_output_config(),
                bad_words=request.sampling_params._get_bad_words(),
                stop_words=request.sampling_params._get_stop_words(),
                embedding_bias=request.sampling_params.embedding_bias,
                external_draft_tokens_config=request.sampling_params.
                external_draft_tokens_config,
                lora_config=lora_config,
                prompt_tuning_config=prompt_tuning_config,
                logits_post_processor_name=request.sampling_params.
                logits_post_processor_name,
            )
            req_id = self.engine.enqueue_request(executor_request)
            return req_id
        except Exception as e:
            raise RequestError(str(e))

    def submit(self, request: GenerationRequest) -> GenerationResult:
        """ Low-level API to the executor. Return a "future" GenerationResult which can be waited. """
        self.start()

        if self.rank != 0:
            raise RuntimeError(
                "Only rank 0 can submit requests.\n"
                "To fix this, ensure that the llm.generate(...) method is "
                "guarded with the `if __name__ == '__main__':` block.")
        req_id = self._enqueue_request(request)

        request.set_id(req_id)

        result = GenerationResult(
            request, background_error_handler=self._handle_background_error)
        self._results[req_id] = result

        self._handle_background_error()

        return result

    def shutdown(self):
        if enable_llm_debug():
            try:
                print_colored('Proxy.shutdown...\n', "yellow")
                print(traceback.extract_stack())
            except ValueError:
                pass

        if self.doing_shutdown:
            return
        else:
            self.doing_shutdown = True

        if self.engine is not None:
            if self.engine.can_enqueue_requests():

                if self.await_response_thread.is_alive():
                    self.await_response_thread.stop()
                    self.await_response_thread.join()
                if self.dispatch_stats_thread.is_alive():
                    self.dispatch_stats_thread.stop()
                    self.dispatch_stats_thread.join()

                self.engine.shutdown()

            self.engine = None

        # Check if there are any errors from the threads before shutdown.
        self._handle_background_error()

    def block_subordinates(self):
        if self.rank != 0:
            self.shutdown()
            raise self.WorkerExit(
                "block_subordinates() should be used in a `with ExecutorBindingsWorker() as ...:` block"
            )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        self.shutdown()
        return exc_type is None or exc_type == ExecutorBindingsWorker.WorkerExit

    def __del__(self):
        self.shutdown()

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


class ZeroMqQueue:
    ''' A Queue-like container for IPC using ZeroMQ. '''

    def __init__(self,
                 address: Optional[Tuple[str, int, str]] = None,
                 *,
                 is_server: bool):
        # NOTE: The port could be occupied by other processes if run in parallel.
        address = address or ('localhost', find_free_port(),
                              secrets.token_bytes(512))

        self.host_port, self.authkey = (address[0], address[1]), address[2]
        self.is_server = is_server
        self.context = zmq.Context()
        self.socket = None

    @property
    def address(self):
        return (self.host_port[0], self.host_port[1], self.authkey)

    def setup(self):
        self.socket = self.context.socket(
            zmq.PAIR)  # PAIR for bidir communication
        if self.is_server:
            self.socket.bind(f'tcp://{self.host_port[0]}:{self.host_port[1]}')
        else:
            self.socket.connect(
                f'tcp://{self.host_port[0]}:{self.host_port[1]}')

    def put(self, obj: Any):
        if self.socket is None:
            self.setup()

        if isinstance(obj, GenerationExecutor.Response):
            tensors = self._store_tensors_in_shmm(obj.tensors)
            obj = GenerationExecutor.Response(request_id=obj.request_id,
                                              tensors=tensors,
                                              finish_reasons=obj.finish_reasons,
                                              is_final=obj.is_final,
                                              error=obj.error)

        message = pickle.dumps(obj)  # nosec B301
        self.socket.send(message)

    def get(self) -> Any:
        if self.socket is None:
            self.setup()

        message = self.socket.recv()
        obj = pickle.loads(message)  # nosec B301

        if isinstance(obj, GenerationExecutor.Response):
            tensors = self._load_tensors_from_shmm(obj.tensors)
            obj = GenerationExecutor.Response(request_id=obj.request_id,
                                              tensors=tensors,
                                              finish_reasons=obj.finish_reasons,
                                              is_final=obj.is_final,
                                              error=obj.error)

        return obj

    def close(self):
        if self.socket:
            self.socket.close()
            self.socket = None
        if self.context:
            self.context.term()
            self.context = None

    def _store_tensors_in_shmm(
        self, tensors: GenerationExecutor.ResponseTensors
    ) -> GenerationExecutor.ResponseTensors:
        if tensors is None:
            return tensors

        # The tensors are huge and cannot be transferred through socket directly. We need to store them in shared memory,
        # and replace the tensors with the shared memory path.
        def store_tensor(tensor: Optional[torch.Tensor]) -> Optional[str]:
            if tensor is None:
                return None
            # NOTE: We create random shmm here rather than two specific shmm for context and generation logit, since the
            # shmm may not be read timely by the IpcQueue.get() in the other side, so there might be multiple alive shmm
            # for logits.
            # A known issue: the shmm instance may leak if the IpcQueue.get() thread is stopped before the IpcQueue.put()
            # thread. This is not a big issue since the shmm will be automatically cleaned up when the process exits.
            shm = SharedMemory(create=True, size=tensor.nbytes + 2048)
            torch.save(tensor, shm._mmap)
            shm.close()
            return shm.name

        return GenerationExecutor.ResponseTensors(
            output_token_ids=tensors.output_token_ids,
            context_logits=store_tensor(tensors.context_logits),
            generation_logits=store_tensor(tensors.generation_logits),
            log_probs=tensors.log_probs,
            cum_log_probs=tensors.cum_log_probs,
        )

    def _load_tensors_from_shmm(
        self, tensors: GenerationExecutor.ResponseTensors
    ) -> GenerationExecutor.ResponseTensors:
        if tensors is None:
            return tensors

        def load_tensor(tensor: Optional[str]) -> Optional[torch.Tensor]:
            if tensor is None or isinstance(tensor, torch.Tensor):
                return tensor

            shm = SharedMemory(name=tensor, create=False)
            tensor = torch.load(io.BytesIO(shm.buf))
            shm.close()
            shm.unlink()
            return tensor

        return GenerationExecutor.ResponseTensors(
            output_token_ids=tensors.output_token_ids,
            context_logits=load_tensor(tensors.context_logits),
            generation_logits=load_tensor(tensors.generation_logits),
            log_probs=tensors.log_probs,
            cum_log_probs=tensors.cum_log_probs,
        )

    def __del__(self):
        self.close()


IpcQueue = ZeroMqQueue


class FusedIpcQueue:
    ''' A Queue-like container for IPC with optional message batched. '''

    def __init__(self,
                 address: Optional[Tuple[str, int, str]] = None,
                 *,
                 is_server: bool,
                 fuse_message=False,
                 fuse_size=100000,
                 error_queue=None,
                 queue_cls=ZeroMqQueue):

        self.queue = queue_cls(address=address, is_server=is_server)
        self.fuse_message = fuse_message
        self.error_queue = error_queue
        self.fuse_size = fuse_size
        self._message_counter = 0
        self._obj_counter = 0
        self._send_thread = None
        self.sending_queue = Queue() if fuse_message else None

    def setup_sender(self):
        if not self.fuse_message or self._send_thread is not None:
            return

        def send_task():
            while True:
                qsize = self.sending_queue.qsize()
                if qsize > 0:
                    qsize = min(self.fuse_size, qsize)
                    self._obj_counter += qsize
                    message = [
                        self.sending_queue.get_nowait() for _ in range(qsize)
                    ]
                    self.queue.put(message)
                    self._message_counter += 1
                else:
                    time.sleep(0.001)

        self._send_thread = ManagedThread(send_task,
                                          name="fused_send_thread",
                                          error_queue=self.error_queue)
        self._send_thread.start()

    def put(self, obj: Any):
        self.setup_sender()
        if self.fuse_message:
            self.sending_queue.put_nowait(self._prepare_message(obj))
        else:
            self.queue.put(self._prepare_message(obj))

    def get(self) -> Any:
        obj = self.queue.get()
        if isinstance(obj, list):
            return [self._process_message(o) for o in obj]
        return self._process_message(obj)

    def _prepare_message(self, obj: Any) -> Any:
        if isinstance(obj, GenerationExecutor.Response):
            tensors = self.queue._store_tensors_in_shmm(obj.tensors)
            return GenerationExecutor.Response(
                request_id=obj.request_id,
                tensors=tensors,
                finish_reasons=obj.finish_reasons,
                is_final=obj.is_final,
                sequence_index=obj.sequence_index,
                error=obj.error)
        return obj

    def _process_message(self, obj: Any) -> Any:
        if isinstance(obj, GenerationExecutor.Response):
            tensors = self.queue._load_tensors_from_shmm(obj.tensors)
            return GenerationExecutor.Response(
                request_id=obj.request_id,
                tensors=tensors,
                finish_reasons=obj.finish_reasons,
                is_final=obj.is_final,
                sequence_index=obj.sequence_index,
                error=obj.error)
        return obj

    @property
    def address(self) -> Tuple[str, int, bytes]:
        return self.queue.address

    def __del__(self):
        self.close()

    def print_fuse_stats(self):
        if self._message_counter > 0:
            print_colored(
                f"IPCQueue: {self._message_counter} messages, {self._obj_counter} objects sent, average: {self._obj_counter/self._message_counter}.\n",
                "green")

    def close(self):
        self.queue.close()

        if self._send_thread is not None:
            self._send_thread.stop()
            self._send_thread.join()
            self._send_thread = None

        if enable_llm_debug():
            self.print_fuse_stats()


class ExecutorBindingsProxy(GenerationExecutor):

    def __init__(self,
                 workers_kwargs,
                 model_world_size: int = 1,
                 mpi_session: Optional[MpiSession] = None,
                 *,
                 worker_cls: type = ExecutorBindingsWorker) -> None:
        super().__init__()

        self.workers_started = False
        self.worker_cls = worker_cls

        self.request_queue = IpcQueue(is_server=True)
        # Return request id back to dispatcher
        self.rid_or_err_queue = IpcQueue(is_server=True)
        self.result_queue = FusedIpcQueue(is_server=True, fuse_message=True)
        self.mp_stats_queue = FusedIpcQueue(is_server=True, fuse_message=True)

        self._results: Dict[int, GenerationResult] = {}

        if mpi_session is None:
            self.mpi_session = MpiPoolSession(n_workers=model_world_size)
        else:
            self.mpi_session = mpi_session

        self.model_world_size = model_world_size

        self.workers_kwargs = workers_kwargs
        self.workers_kwargs.update({
            "request_queue_addr":
            self.request_queue.address,
            "rid_or_err_queue_addr":
            self.rid_or_err_queue.address,
            "result_queue_addr":
            self.result_queue.address,
            "stats_queue_addr":
            self.mp_stats_queue.address,
        })

        self.dispatch_result_thread = ManagedThread(
            self.dispatch_result_task,
            error_queue=self._error_queue,
            name="proxy_dispatch_result_thread")
        self.dispatch_stats_thread = ManagedThread(
            self.dispatch_stats_task,
            error_queue=self._error_queue,
            name="proxy_dispatch_stats_thread")

    @staticmethod
    def workers_main(engine: Union[Path, Engine],
                     request_queue_addr: Tuple[str, int, bytes],
                     rid_or_err_queue_addr: Tuple[str, int, bytes],
                     result_queue_addr: Tuple[str, int, bytes],
                     stats_queue_addr: Tuple[str, int, bytes],
                     executor_config: tllm.ExecutorConfig = tllm.ExecutorConfig(
                         1),
                     worker_cls: type = ExecutorBindingsWorker,
                     tracer_init_kwargs=None) -> None:
        result_queue = None

        if tracer_init_kwargs is not None and mpi_rank() == 0:
            tracer = VizTracer(**tracer_init_kwargs)
            tracer.register_exit()
            tracer.start()
            set_global_tracer(tracer)

        if mpi_rank() == 0:
            request_queue = IpcQueue(request_queue_addr, is_server=False)
            rid_or_err_queue = IpcQueue(rid_or_err_queue_addr, is_server=False)
            result_queue = FusedIpcQueue(result_queue_addr,
                                         is_server=False,
                                         fuse_message=True)
            mp_stats_queue = FusedIpcQueue(stats_queue_addr,
                                           is_server=False,
                                           fuse_message=True)

        def notify_proxy_threads_to_quit():
            # Signal the dispatcher thread in the proxy to quit
            result_queue.put(None)
            # Signal the stats thread in the proxy to quit
            mp_stats_queue.put(None)

        # Error handling in the Worker/MPI process
        #   1. During Executor initialization, the errors will be captured by MPIPoolExecutor, and propagate to the
        #      future.done_callback in the Python main process.
        #   2. After Executor initialization, the errors will be captured by ManagedThreads, and propagate to the
        #      error_queue in the Python main process by IPC queue of result_queue in await_response_task.

        try:
            executor = worker_cls(engine, executor_config)
        except Exception as e:
            raise CppExecutorError(f"Failed to initialize executor: {e}") from e

        with executor:
            try:
                executor.block_subordinates()

                if mpi_rank() == 0:
                    executor.set_result_queue(result_queue)
                    executor.set_stats_queue(mp_stats_queue)
                    while (req := request_queue.get()) is not None:
                        try:
                            result = executor.submit(req)
                            rid_or_err_queue.put(result.request_id)
                        except RequestError as e:
                            rid_or_err_queue.put(e)
                    notify_proxy_threads_to_quit()

            except ExecutorBindingsWorker.WorkerExit as e:
                raise e  # This will capture by the with-statement and exit normally.

            except Exception as e:  # other critical errors
                if mpi_rank() == 0:
                    notify_proxy_threads_to_quit()
                err = CppExecutorError(f"Failed during generation: {e}")
                if mpi_rank() == 0:
                    rid_or_err_queue.put(err)

    def dispatch_result_task(self) -> bool:
        # process the remaining pending req_ids before getting the next response, since the queue.get will block, we'd
        # better to process the pending req_ids before queue.get.
        self._cleanup_pending_responses(nowait=True)

        if (res := self.result_queue.get()) is None:
            return False  # shutdown the thread

        async_events = []
        event_loop = None

        def process_res(res):
            req_id = res.request_id
            nonlocal event_loop
            nonlocal async_events

            if not self._to_delay_response(res):
                queue = self._results[req_id].queue
                if isinstance(queue, _SyncQueue):
                    queue.put_nowait(res)
                    async_events.append(queue.event)
                    event_loop = queue.loop if event_loop is None else event_loop  # all the loops are identical
                else:
                    queue.put(res)

                if res.is_final:
                    self._results.pop(req_id)
            else:
                self._pending_responses.setdefault(req_id, []).append(
                    self.PendingResponse(res, time.perf_counter()))

        res = res if isinstance(res, list) else [res]

        for i in res:
            global_tracer().log_instant("IPC.get")
            if i is None:
                return False
            process_res(i)

        if async_events:
            _SyncQueue.notify_events(event_loop, async_events)

        return True  # success

    def dispatch_stats_task(self) -> bool:
        # get-stats is not urgent, so we can sleep a bit

        time.sleep(0.1)

        try:
            stats = self.mp_stats_queue.get()
        except:
            return False

        if stats is None:
            return False

        stats = stats if isinstance(stats, list) else [stats]

        while self.stats_queue.full():
            self.stats_queue.get()

        try:
            for s in stats:
                if s is None: return False
                self.stats_queue.put(s)
        except AsyncQueue.EventLoopShutdownError:
            # This happens in the last stats loop while the generate workflow is stopped.
            pass
        except Exception as e:
            raise e

        return True  # success

    def start(self):

        def mpi_done_callback(future: concurrent.futures.Future):
            # This is called when the MPI worker is done, so future.exception() will not block.
            if future.exception() is not None:
                self._error_queue.put_nowait(future.exception())

        tracer_init_kwargs = get_tracer().init_kwargs if enable_llm_tracer(
        ) else None
        self.mpi_futures = self.mpi_session.submit(
            ExecutorBindingsProxy.workers_main,
            **self.workers_kwargs,
            worker_cls=self.worker_cls,
            tracer_init_kwargs=tracer_init_kwargs)
        for fut in self.mpi_futures:
            fut.add_done_callback(mpi_done_callback)

        self.workers_started = True

        self.dispatch_result_thread.start()
        self.create_stats_queue()
        self.dispatch_stats_thread.start()

        self._handle_background_error()

    def shutdown(self):
        if enable_llm_debug():
            try:
                print_colored('Proxy.shutdown...\n', "yellow")
                print_colored(str(traceback.format_exc()) + "\n", "yellow")
            except ValueError:
                pass
        if not self.workers_started:
            return

        if self.doing_shutdown:
            return
        else:
            self.doing_shutdown = True

        # step1: notify the workers to quit
        self.request_queue.put(None)

        for f in self.mpi_futures:
            try:
                f.result()
            except:
                # The errors are already captured in mpi_done_callback, ignored here
                pass

        # step2: notify the background threads to quit
        if self.dispatch_result_thread.is_alive():
            self.dispatch_result_thread.stop()
            self.dispatch_result_thread.join()
        if self.dispatch_stats_thread.is_alive():
            self.dispatch_stats_thread.stop()
            self.dispatch_stats_thread.join()

        # step3: finish all remaining work

        # It is possible that some requests are still pending in the workers, we need to process them before shutdown
        self._cleanup_pending_responses(nowait=False)

        # close all the sockets
        self.request_queue.close()
        self.rid_or_err_queue.close()
        self.result_queue.close()
        self.mp_stats_queue.close()

        self.workers_started = False

        # Process the errors in-case error during shutting down the threads
        self._handle_background_error()

    def submit(self, request: GenerationRequest) -> GenerationResult:
        """
            Low-level API to the executor. Return a "future" GenerationResult which can be waited.
            Forwards the request to the workers through the request queue.
        """
        if not self.workers_started:
            self.start()

        self.request_queue.put(request)

        rid_or_err = self.rid_or_err_queue.get()
        if isinstance(rid_or_err, Exception):
            raise rid_or_err
        request.set_id(rid_or_err)

        result = GenerationResult(
            request, background_error_handler=self._handle_background_error)
        self._results[rid_or_err] = result

        self._handle_background_error()

        return result

    def __del__(self):
        self.shutdown()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.shutdown()
        return False  # propagate the exception


if enable_llm_debug():
    print_colored("LLM debug mode enabled.\n", "yellow")
    # This will dump all the alive threads when the process is interrupted by SIGINT.
    faulthandler.register(signal.SIGINT, all_threads=True)


def has_event_loop() -> bool:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return False
    return True
