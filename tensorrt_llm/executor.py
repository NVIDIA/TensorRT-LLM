import asyncio
import atexit
import concurrent.futures
import datetime
import json
import secrets
import time
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from multiprocessing.connection import Client, Listener
from pathlib import Path
from queue import Queue
from typing import (Any, Dict, Generator, List, NamedTuple, Optional, Tuple,
                    Union)

import numpy as np
import torch
from janus import Queue as AsyncQueue

from ._utils import mpi_rank, mpi_world_size
from .bindings import executor as tllm
from .builder import ConfigEncoder, Engine, EngineConfig
from .hlapi.mpi_session import (MpiPoolSession, MpiSession,
                                external_mpi_comm_available, find_free_port,
                                need_spawn_mpi_workers)
from .hlapi.utils import ManagedThread, SamplingParams, exception_handler
from .lora_manager import LoraManager
from .runtime import ModelConfig
from .runtime.model_runner import _engine_config_to_model_config


def has_event_loop() -> bool:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return False
    return True


@dataclass(slots=True)
class LoRARequest:
    lora_name: str
    lora_int_id: int
    lora_path: str = ""

    def __post_init__(self):
        assert self.lora_path, "lora_path cannot be empty"

    @property
    def adapter_id(self):
        return self.lora_int_id

    @property
    def name(self):
        return self.lora_name

    @property
    def path(self):
        return self.lora_path


class GenerationRequest:

    def __init__(
        self,
        prompt_token_ids: Union[torch.Tensor, np.ndarray, list],
        sampling_params: SamplingParams,
        lora_request: Optional[LoRARequest] = None,
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
        text (str): The generated output text.
        token_ids (List[int]): The token ids of the generated output text.
        cumulative_logprob (float): The cumulative log probability of the generated output text.
        logprobs (List[float]): The log probabilities of the top probability words at each position if the logprobs are requested.
        generation_logits (torch.Tensor): The logits on the generated output token ids.
    """
    index: int
    text: str = ""
    token_ids: List[int] = field(default_factory=list)
    cumulative_logprob: Optional[float] = None
    logprobs: List[float] = field(default_factory=list)
    generation_logits: Optional[torch.Tensor] = field(default=None, repr=False)
    _last_text: str = field(default="", init=False, repr=False)

    @property
    def length(self):
        return len(self.token_ids)

    @property
    def text_diff(self) -> str:
        diff = self.text[len(self._last_text):]
        self._last_text = self.text
        return diff


class CppExecutorError(RuntimeError):

    def __init__(self, message: Optional[str] = None):
        self.message = message
        self.stack_trace = traceback.format_exc()
        super().__init__(message)

    def __str__(self):
        return f"{self.message}\nStack trace:\n{self.stack_trace}"


class GenerationResult:
    '''
    The result of a generation request. It can be used to wait for the completion of the request.

    Args:
        generation_request (GenerationRequest): The generation request object.
        background_error_handler (Optional[callable]): The error handler to process the errors from the background threads/processes.
    '''

    def __init__(self,
                 generation_request: GenerationRequest,
                 background_error_handler: Optional[callable] = None) -> None:
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

        self.outputs: List[CompletionOutput] = [
            CompletionOutput(i) for i in range(self.beam_width)
        ]
        self.context_logits: Optional[torch.Tensor] = None

        self._background_error_handler = background_error_handler

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
    def beam_width(self):
        return self._generation_request.sampling_params.beam_width

    def handle_response(self, response: "GenerationExecutor.Response"):

        if response.error:
            if isinstance(response.error, Exception):
                raise response.error
            else:
                raise CppExecutorError(response.error)

        self._done = response.is_final

        tensors = response.tensors

        for i, beam_ids in enumerate(tensors.output_token_ids):
            self.outputs[i].token_ids.extend(beam_ids)
            if tensors.cum_log_probs is not None:
                self.outputs[i].cumulative_logprob = tensors.cum_log_probs[i]
            if tensors.log_probs is not None:
                self.outputs[i].logprobs = tensors.log_probs[i]
                assert len(self.outputs[i].logprobs) == self.outputs[i].length
            if tensors.generation_logits is not None:
                self.outputs[i].generation_logits = tensors.generation_logits[
                    i, :self.outputs[i].length]

        if self.finished and not self._generation_request.sampling_params.include_stop_str_in_output:
            for beam_output in self.outputs:
                for stop_ids in self._generation_request.sampling_params._get_stop_words(
                ):
                    if beam_output.token_ids[-len(stop_ids):] == stop_ids:
                        beam_output.token_ids = beam_output.token_ids[:-len(
                            stop_ids)]
                        break

        if tensors.context_logits is not None:
            self.context_logits = tensors.context_logits

        # Processing background errors here ASAF during generation.
        if self._background_error_handler:
            self._background_error_handler()

    def result_step(self, timeout: Optional[float] = None):
        response = self.queue.get(timeout=timeout)
        self.handle_response(response)

    async def aresult_step(self):
        assert self.aqueue is not None, "The asyncio event loop was not present during initialization, so async operations are not available."
        response = await self.aqueue.get()
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
        return ['request_id', 'prompt_token_ids', 'outputs', 'finished']

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


class GenerationExecutor(ABC):

    PENDING_REQ_ID_TIMEOUT = 2  # second

    class ResponseTensors(NamedTuple):
        output_token_ids: list
        context_logits: Optional[torch.Tensor]
        generation_logits: Optional[torch.Tensor]
        log_probs: Optional[list]
        cum_log_probs: Optional[list]

    class Response(NamedTuple):
        """ The response from the cpp-executor to the Python main thread. """
        request_id: int
        tensors: Optional["GenerationExecutor.ResponseTensors"]
        is_final: Optional[bool]
        # error is either str from cpp-executor or a Exception from Python threads/processes
        error: Optional[str | Exception]

    @dataclass(slots=True)
    class PendingResponse:
        response: "GenerationExecutor.Response"
        start_time: float  # this is used to track the latency before the response is dispatched.

    def __init__(self):
        self._stats = None
        self.stats_queue = None

        # This is used to capture the exceptions from the threads.
        self._error_queue = Queue()

        # mapping of pending request_id -> response
        self._pending_responses: Dict[
            int, List[GenerationExecutor.PendingResponse]] = {}

        exception_handler.register(self, 'shutdown')
        atexit.register(self.shutdown)

    @abstractmethod
    def submit(self, request: GenerationRequest) -> GenerationResult:
        pass

    def generate_async(
        self,
        prompt_token_ids: List[int],
        sampling_params: SamplingParams,
        lora_request: Optional[LoRARequest] = None,
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
                              streaming=streaming))
        return result

    def generate(
        self,
        prompt_token_ids: Union[List[int], List[List[int]]],
        sampling_params: Union[SamplingParams, List[SamplingParams]],
        lora_request: Optional[Union[LoRARequest, List[LoRARequest]]] = None,
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
            future = self.generate_async(p,
                                         sampling_params=sp,
                                         lora_request=lora_req,
                                         streaming=False)
            futures.append(future)

        for future in futures:
            future.result()

        if unbatched:
            futures = futures[0]

        return futures

    def _handle_background_error(self):
        """ Process the errors from the threads or processes.
        NOTE: This should be called in the main thread.
        """
        # Here we raise the first error in the queue. This method will be called repeatedly and user can choose to catch
        # more than one error.
        if not self._error_queue.empty():
            e = self._error_queue.get()
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
                    for response in responses:
                        self._results[req_id].queue.put(
                            response.response)  # dispatch
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

    def get_stats(self):
        return self.stats_queue.get()

    async def aget_stats(self):
        assert self.stats_aqueue is not None, "The asyncio event loop was not present during initialization, so async operations are not available."
        return await self.stats_aqueue.get()

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
            engine.regularize_managed_weights()
            self.engine = tllm.Executor(engine.engine,
                                        json.dumps(engine.config.to_dict(),
                                                   cls=ConfigEncoder),
                                        tllm.ModelType.DECODER_ONLY,
                                        executor_config=executor_config,
                                        managed_weights=engine.managed_weights
                                        or {})
        else:
            self.engine = tllm.Executor(engine,
                                        tllm.ModelType.DECODER_ONLY,
                                        executor_config=executor_config)

        self._lora_manager: Optional[LoraManager] = None
        self._runtime_model_config: Optional[ModelConfig] = None
        if self.rank == 0:
            if isinstance(engine, Engine):
                engine_config = engine.config
            else:
                engine_config = EngineConfig.from_json_file(
                    f"{engine}/config.json")
            if engine_config.build_config.plugin_config.lora_plugin:
                self._runtime_model_config = _engine_config_to_model_config(
                    engine_config)
                self._lora_manager = LoraManager()

        self.await_response_thread = ManagedThread(
            self.await_response_task, error_queue=self._error_queue)
        self.dispatch_stats_thread = ManagedThread(
            self.dispatch_stats_task, error_queue=self._error_queue)

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

    def await_response_task(self) -> bool:
        # Get responses and place in queue.

        for response in self.engine.await_responses(timeout=datetime.timedelta(
                milliseconds=100)):
            req_id = response.request_id
            if response.has_error():
                rsp = self.Response(req_id,
                                    tensors=None,
                                    is_final=None,
                                    error=response.error_msg)
            else:
                tensors = self.ResponseTensors(
                    response.result.output_token_ids,
                    response.result.context_logits,
                    response.result.generation_logits,
                    response.result.log_probs, response.result.cum_log_probs)

                rsp = self.Response(req_id,
                                    tensors,
                                    is_final=response.result.is_final,
                                    error=None)

            if self._to_delay_response(rsp):
                continue

            self._cleanup_pending_responses(nowait=True)

            queue = self.return_queue(req_id)
            bck_error = self._error_queue.get_nowait(
            ) if not self._error_queue.empty() else None

            if bck_error is not None:
                rsp = self.Response(req_id,
                                    tensors=None,
                                    is_final=None,
                                    error=bck_error)

            queue.put(rsp)

            if response.result.is_final:
                self._results.pop(req_id)

        return True  # success

    def dispatch_stats_task(self) -> bool:
        time.sleep(0.1)
        # Get stats and place in queue.
        for stats in self.engine.get_latest_iteration_stats():
            while hasattr(self.stats_queue, "full") and self.stats_queue.full():
                self.stats_queue.get()
            self.stats_queue.put(stats.to_json_str())

        return True  # success

    def start(self):
        self.create_stats_queue()
        self.start_awaiter_thread()
        self.start_stats_thread()

    def _load_lora_adapter(self, lora_request: LoRARequest):
        self._lora_manager.load_from_ckpt(
            [lora_request.lora_path],
            model_config=self._runtime_model_config,
            runtime_mapping=None,
            uids=[str(lora_request.adapter_id)])

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

        executor_request = tllm.Request(
            input_token_ids=request.prompt_token_ids,
            max_tokens=request.sampling_params.max_tokens,
            max_new_tokens=request.sampling_params.max_new_tokens,
            streaming=request.streaming,
            sampling_config=request.sampling_params._get_sampling_config(),
            end_id=request.sampling_params.end_id,
            pad_id=request.sampling_params.pad_id,
            output_config=request.sampling_params._get_output_config(),
            bad_words=request.sampling_params._get_bad_words(),
            stop_words=request.sampling_params._get_stop_words(),
            embedding_bias=request.sampling_params.embedding_bias,
            external_draft_tokens_config=request.sampling_params.
            external_draft_tokens_config,
            prompt_tuning_config=request.sampling_params.prompt_tuning_config,
            lora_config=lora_config,
            logits_post_processor_name=request.sampling_params.
            logits_post_processor_name,
        )
        req_id = self.engine.enqueue_request(executor_request)
        return req_id

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
        if self.engine is not None:
            self.await_response_thread.stop()
            self.dispatch_stats_thread.stop()

            if self.engine.can_enqueue_requests():
                if self.await_response_thread.is_alive():
                    self.await_response_thread.join()
                if self.dispatch_stats_thread.is_alive():
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


class IpcQueue:
    ''' A Queue-like container for IPC. '''

    def __init__(self,
                 address: Optional[Tuple[str, int, str]] = None,
                 *,
                 is_server: bool):

        # NOTE: The port could be occupied by other processes if run in parallel.
        address = address or ('localhost', find_free_port(),
                              secrets.token_bytes(512))

        self.host_port, self.authkey = (address[0], address[1]), address[2]
        self.is_server = is_server
        self.conn = None
        if is_server:
            self.listener = Listener(self.host_port,
                                     'AF_INET',
                                     authkey=self.authkey)

    def setup(self):
        if self.is_server:
            self.conn = self.listener.accept()
        else:
            self.conn = Client(self.host_port, authkey=self.authkey)

    def put(self, obj: Any):
        if self.conn is None:
            self.setup()
        self.conn.send(obj)

    def get(self) -> Any:
        if self.conn is None:
            self.setup()
        return self.conn.recv()

    @property
    def address(self) -> Tuple[str, int, bytes]:
        return (self.host_port[0], self.host_port[1], self.authkey)


class ExecutorBindingsProxy(GenerationExecutor):

    def __init__(
        self,
        workers_kwargs,
        model_world_size: int = 1,
        mpi_session: Optional[MpiSession] = None,
    ) -> None:
        super().__init__()

        self.workers_started = False

        self.request_queue = IpcQueue(is_server=True)
        # Return request id back to dispatcher
        self.request_id_queue = IpcQueue(is_server=True)
        self.result_queue = IpcQueue(is_server=True)
        self.mp_stats_queue = IpcQueue(is_server=True)

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
            "request_id_queue_addr":
            self.request_id_queue.address,
            "result_queue_addr":
            self.result_queue.address,
            "stats_queue_addr":
            self.mp_stats_queue.address,
        })

        self.dispatch_result_thread = ManagedThread(
            self.dispatch_result_task, error_queue=self._error_queue)
        self.dispatch_stats_thread = ManagedThread(
            self.dispatch_stats_task, error_queue=self._error_queue)

        exception_handler.register(self, 'shutdown')
        atexit.register(self.shutdown)

    @staticmethod
    def workers_main(
        engine: Union[Path, Engine],
        request_queue_addr: Tuple[str, int, bytes],
        request_id_queue_addr: Tuple[str, int, bytes],
        result_queue_addr: Tuple[str, int, bytes],
        stats_queue_addr: Tuple[str, int, bytes],
        executor_config: tllm.ExecutorConfig = tllm.ExecutorConfig(1)
    ) -> None:
        result_queue = None

        if mpi_rank() == 0:
            request_queue = IpcQueue(request_queue_addr, is_server=False)
            request_id_queue = IpcQueue(request_id_queue_addr, is_server=False)
            result_queue = IpcQueue(result_queue_addr, is_server=False)
            mp_stats_queue = IpcQueue(stats_queue_addr, is_server=False)

        def notify_proxy_threads_to_quit():
            # Signal the dispatcher thread in the proxy to quit
            result_queue.put(None)
            # Signal the stats thread in the proxy to quit
            mp_stats_queue.put(None)

        try:
            executor = ExecutorBindingsWorker(engine, executor_config)
        except Exception as e:
            raise CppExecutorError(f"Failed to initialize executor: {e}") from e

        with executor:
            try:
                executor.block_subordinates()

                if mpi_rank() == 0:
                    executor.set_result_queue(result_queue)
                    executor.set_stats_queue(mp_stats_queue)
                    while (req := request_queue.get()) is not None:
                        result = executor.submit(req)
                        request_id_queue.put(result.request_id)

                    notify_proxy_threads_to_quit()

            except ExecutorBindingsWorker.WorkerExit as e:
                raise e

            except Exception as e:  # other critical errors
                if mpi_rank() == 0:
                    notify_proxy_threads_to_quit()

                raise CppExecutorError(f"Failed during generation: {e}") from e

    def dispatch_result_task(self) -> bool:
        # process the remaining pending req_ids before getting the next response, since the queue.get will block, we'd
        # better to process the pending req_ids before queue.get.
        self._cleanup_pending_responses(nowait=True)

        if (res := self.result_queue.get()) is None:
            return False  # shutdown the thread

        req_id = res.request_id

        if not self._to_delay_response(res):
            self._results[req_id].queue.put(res)

            if res.is_final:
                self._results.pop(req_id)
        else:
            self._pending_responses.setdefault(req_id, []).append(
                self.PendingResponse(res, time.perf_counter()))

        return True  # success

    def dispatch_stats_task(self) -> bool:
        if (stats := self.mp_stats_queue.get()) is None:
            return False  # shutdown the thread

        # get-stats is not urgent, so we can sleep a bit
        time.sleep(0.1)
        while self.stats_queue.full():
            self.stats_queue.get()
        self.stats_queue.put(stats)
        return True  # success

    def start(self):

        def mpi_done_callback(future: concurrent.futures.Future):
            try:
                future.result()
            except:
                self._error_queue.put_nowait(future.exception())

        self.mpi_futures = self.mpi_session.submit(
            ExecutorBindingsProxy.workers_main, **self.workers_kwargs)
        for fut in self.mpi_futures:
            fut.add_done_callback(mpi_done_callback)

        self.workers_started = True

        self.dispatch_result_thread.start()
        self.create_stats_queue()
        self.dispatch_stats_thread.start()

        self._handle_background_error()

    def shutdown(self):
        if not self.workers_started:
            return

        self.request_queue.put(None)  # Tell the rank0 worker to quit

        for f in self.mpi_futures:
            f.result()

        if self.dispatch_result_thread.is_alive():
            self.dispatcher.join()
        if self.dispatch_stats_thread.is_alive():
            self.dispatch_stats_thread.join()

        # It is possible that some requests are still pending in the workers, we need to process them before shutdown
        self._cleanup_pending_responses(nowait=False)

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

        req_id = self.request_id_queue.get()
        request.set_id(req_id)

        result = GenerationResult(
            request, background_error_handler=self._handle_background_error)
        self._results[req_id] = result

        self._handle_background_error()

        return result

    def __del__(self):
        self.shutdown()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.shutdown()
        return False
