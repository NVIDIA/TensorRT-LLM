import asyncio
import atexit
import datetime
import secrets
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from multiprocessing.connection import Client, Listener
from pathlib import Path
from queue import Queue
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

import numpy as np
import torch
from janus import Queue as AsyncQueue

from ._utils import mpi_rank, mpi_world_size
from .bindings import executor as tllm
from .hlapi.mpi_session import (MpiPoolSession, MpiSession,
                                external_mpi_comm_available, find_free_port,
                                need_spawn_mpi_workers)
from .hlapi.utils import (ContextManager, SamplingParams, exception_handler,
                          print_traceback_on_error)


def has_event_loop() -> bool:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return False
    return True


class GenerationRequest:

    def __init__(
        self,
        prompt_token_ids: Union[torch.Tensor, np.ndarray, list],
        sampling_params: SamplingParams,
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
        self.streaming = streaming
        self.id = -1

    def set_id(self, id):
        self.id = id
        return self

    def as_executor_request(self) -> tllm.Request:
        request_kwargs = {
            "input_token_ids":
            self.prompt_token_ids,
            "max_new_tokens":
            self.sampling_params.max_new_tokens,
            "streaming":
            self.streaming,
            "sampling_config":
            self.sampling_params._get_sampling_config(),
            "end_id":
            self.sampling_params.end_id,
            "pad_id":
            self.sampling_params.pad_id,
            "output_config":
            self.sampling_params._get_output_config(),
            # The following options in the Executor API are not yet exposed by the HLAPI:
            # https://jirasw.nvidia.com/browse/TRTLLM-489
            "bad_words":
            self.sampling_params.bad_words or [],
            "stop_words":
            self.sampling_params.stop_words or [],
            "embedding_bias":
            self.sampling_params.embedding_bias,
            "external_draft_tokens_config":
            self.sampling_params.external_draft_tokens_config,
            "prompt_tuning_config":
            self.sampling_params.prompt_tuning_config,
            "lora_config":
            self.sampling_params.lora_config,
            "logits_post_processor_name":
            self.sampling_params.logits_post_processor_name,
        }
        request = tllm.Request(**request_kwargs)
        return request


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


class GenerationResult:

    def __init__(self, generation_request: GenerationRequest) -> None:
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

    def handle_generation_msg(self, tensors: tuple, error: str):
        if error:
            raise RuntimeError(error)

        output_token_ids, context_logits, generation_logits, log_probs, cum_log_probs = tensors

        for i, beam_ids in enumerate(output_token_ids):
            self.outputs[i].token_ids.extend(beam_ids)
            if cum_log_probs is not None:
                self.outputs[i].cumulative_logprob = cum_log_probs[i]
            if log_probs is not None:
                self.outputs[i].logprobs = log_probs[i]
                assert len(self.outputs[i].logprobs) == self.outputs[i].length
            if generation_logits is not None:
                self.outputs[i].generation_logits = generation_logits[
                    i, :self.outputs[i].length]

        if context_logits is not None:
            self.context_logits = context_logits

    def result_step(self, timeout: Optional[float] = None):
        _, tensors, self._done, error = self.queue.get(timeout=timeout)
        self.handle_generation_msg(tensors, error)

    async def aresult_step(self):
        assert self.aqueue is not None, "The asyncio event loop was not present during initialization, so async operations are not available."
        _, tensors, self._done, error = await self.aqueue.get()
        self.handle_generation_msg(tensors, error)

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


class GenerationExecutor(ABC):
    TERMINATE_REQUEST_ID = 0

    def __init__(self):
        self.id_counter = GenerationExecutor.TERMINATE_REQUEST_ID + 1
        self._stats = None
        self.stats_queue = None

        exception_handler.register(self)
        atexit.register(self.shutdown)

    def generate_id(self) -> int:
        gen_id = self.id_counter

        # underlying C type is uint64
        uint64_max = 2**64 - 1
        self.id_counter = (self.id_counter + 1) % uint64_max

        if self.id_counter == GenerationExecutor.TERMINATE_REQUEST_ID:
            self.id_counter += 1

        return gen_id

    @abstractmethod
    def submit(self, request: GenerationRequest) -> GenerationResult:
        pass

    def generate_async(
        self,
        prompt_token_ids: List[int],
        sampling_params: SamplingParams,
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
                              streaming=streaming))
        return result

    def generate(
        self, prompt_token_ids: Union[List[int], List[List[int]]],
        sampling_params: Union[SamplingParams, List[SamplingParams]]
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
            future = self.generate_async(p, sampling_params=sp, streaming=False)
            futures.append(future)

        for future in futures:
            future.result()

        if unbatched:
            futures = futures[0]

        return futures

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
        engine_dir: Path,
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
            "engine_dir": engine_dir,
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
        engine_dir: Path,
        executor_config: tllm.ExecutorConfig = tllm.ExecutorConfig(1),
    ) -> None:
        super().__init__()

        self.engine = None
        self._results: Dict[int, GenerationResult] = {}
        self._pending: set = set()
        self.result_queue = None
        self.rank = mpi_rank()

        self.engine = tllm.Executor(engine_dir,
                                    tllm.ModelType.DECODER_ONLY,
                                    executor_config=executor_config)
        self.awaiter_stop_event = threading.Event()
        self.awaiter_thread = threading.Thread(target=self.awaiter_loop,
                                               daemon=True)
        self.stats_thread = threading.Thread(target=self.stats_loop,
                                             daemon=True)

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
        ) and not self.awaiter_thread.is_alive():
            self.awaiter_thread.start()

    def start_stats_thread(self):
        if self.engine.can_enqueue_requests(
        ) and not self.stats_thread.is_alive():
            self.stats_thread.start()

    def awaiter_loop(self):
        """ Gets responses from executor and places in the return queue."""
        while not self.awaiter_stop_event.is_set():
            # Get responses and place in queue.
            for response in self.engine.await_responses(
                    timeout=datetime.timedelta(milliseconds=100)):
                req_id = response.request_id
                if response.has_error():
                    self.return_queue(req_id).put(
                        (req_id, None, None, response.error_msg))
                else:
                    tensors = (
                        response.result.output_token_ids,
                        response.result.context_logits,
                        response.result.generation_logits,
                        response.result.log_probs,
                        response.result.cum_log_probs,
                    )
                    self.return_queue(req_id).put(
                        (response.request_id, tensors, response.result.is_final,
                         None))
                    if response.result.is_final:
                        self._pending.remove(req_id)

    def stats_loop(self):
        while not self.awaiter_stop_event.is_set():
            time.sleep(0.1)
            # Get stats and place in queue.
            for stats in self.engine.get_latest_iteration_stats():
                while hasattr(self.stats_queue,
                              "full") and self.stats_queue.full():
                    self.stats_queue.get()
                self.stats_queue.put(stats.to_json_str())

    def start(self):
        self.create_stats_queue()
        self.start_awaiter_thread()
        self.start_stats_thread()

    def submit(self, request: GenerationRequest) -> GenerationResult:
        """
            Low-level API to the executor. Return a "future" GenerationResult which can be waited.
        """
        self.start()

        if self.rank != 0:
            raise NotImplementedError("Only rank 0 can submit requests.")
        req_id = self.engine.enqueue_request(request.as_executor_request())
        request.set_id(req_id)

        result = GenerationResult(request)
        self._results[req_id] = result
        self._pending.add(req_id)
        return result

    def shutdown(self):
        if self.engine is not None:
            self.awaiter_stop_event.set()
            if self.engine.can_enqueue_requests():
                if self.awaiter_thread.is_alive():
                    self.awaiter_thread.join()
                if self.stats_thread.is_alive():
                    self.stats_thread.join()
            self.engine.shutdown()
            self.engine = None

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
        wait_set = set(f.request_id for f in futures)

        # clear already-finished requests
        for f in futures:
            if f._done:
                wait_set.remove(f.request_id)
                yield f

        # wait remaining active requests
        while len(wait_set) > 0:
            req_id = wait_set.pop()

            if req_id not in self._pending:
                yield self._results[req_id]
            else:
                wait_set.add(req_id)


class Fifo:

    def __init__(self, address: Tuple[str, int, bytes], *, is_server: bool):
        self.address, self.authkey = (address[0], address[1]), address[2]
        self.is_server = is_server
        self.conn = None
        if is_server:
            self.listener = Listener(self.address,
                                     'AF_INET',
                                     authkey=self.authkey)

    def setup(self):
        if self.is_server:
            self.conn = self.listener.accept()
        else:
            self.conn = Client(self.address, authkey=self.authkey)

    def put(self, obj: Any):
        if self.conn is None:
            self.setup()
        self.conn.send(obj)

    def get(self) -> Any:
        if self.conn is None:
            self.setup()
        return self.conn.recv()


class ExecutorBindingsProxy(GenerationExecutor):

    def __init__(
        self,
        workers_kwargs,
        model_world_size: int = 1,
        mpi_session: Optional[MpiSession] = None,
    ) -> None:
        super().__init__()

        self.workers_started = False

        request_queue_addr = ("127.0.0.1", find_free_port(),
                              secrets.token_bytes(512))
        self.request_queue = Fifo(request_queue_addr, is_server=True)

        # Return request id back to dispatcher
        request_id_queue_addr = ("127.0.0.1", find_free_port(),
                                 secrets.token_bytes(512))
        self.request_id_queue = Fifo(request_id_queue_addr, is_server=True)

        result_queue_addr = ("127.0.0.1", find_free_port(),
                             secrets.token_bytes(512))
        self.result_queue = Fifo(result_queue_addr, is_server=True)

        stats_queue_addr = ("127.0.0.1", find_free_port(),
                            secrets.token_bytes(512))
        self.mp_stats_queue = Fifo(stats_queue_addr, is_server=True)

        self._results: Dict[int, GenerationResult] = {}
        self._request_id_dispatcher_queue = Queue()

        if mpi_session is None:
            self.mpi_session = MpiPoolSession(n_workers=model_world_size)
        else:
            self.mpi_session = mpi_session
        self.model_world_size = model_world_size

        self.workers_kwargs = workers_kwargs
        self.workers_kwargs.update({
            "request_queue_addr": request_queue_addr,
            "request_id_queue_addr": request_id_queue_addr,
            "result_queue_addr": result_queue_addr,
            "stats_queue_addr": stats_queue_addr,
        })
        self.workers_init_ok = False
        self.dispatcher = threading.Thread(target=self.dispatcher_thread,
                                           daemon=True)
        self.stats_thread = threading.Thread(target=self.stats_main,
                                             daemon=True)

    @print_traceback_on_error
    @staticmethod
    def workers_main(
        engine_dir: Path,
        request_queue_addr: Tuple[str, int, bytes],
        request_id_queue_addr: Tuple[str, int, bytes],
        result_queue_addr: Tuple[str, int, bytes],
        stats_queue_addr: Tuple[str, int, bytes],
        executor_config: tllm.ExecutorConfig = tllm.ExecutorConfig(1)
    ) -> None:
        result_queue = None

        if mpi_rank() == 0:
            request_queue = Fifo(request_queue_addr, is_server=False)
            request_id_queue = Fifo(request_id_queue_addr, is_server=False)
            result_queue = Fifo(result_queue_addr, is_server=False)
            mp_stats_queue = Fifo(stats_queue_addr, is_server=False)

        # Only the failure on rank0 can be captured here. All the non-rank0 process will hang once the executor runtime
        # is successfully initialized, that is controlled within cpp runtime.
        # To capture the failure on all the ranks, more work should be done in the cpp runtime.
        # TODO[chunweiy]: fix the non-rank0 process failure
        init_ok = True
        try:
            executor = ExecutorBindingsWorker(engine_dir, executor_config)
        except Exception as e:
            init_ok = False
            raise e
        finally:
            if mpi_rank() == 0:
                result_queue.put(init_ok)

        with ContextManager(executor) as executor:
            if mpi_rank() == 0:
                executor.set_result_queue(result_queue)
                executor.set_stats_queue(mp_stats_queue)
                while (req := request_queue.get()) is not None:
                    result = executor.submit(req)
                    request_id_queue.put(result.request_id)

                result_queue.put(None)
                mp_stats_queue.put(None)
            else:
                executor.block_subordinates()

    def dispatcher_thread(self):
        """ Collect centralized results from result queue and dispatch them in the
            correct GenerationResult queues. """

        while (res := self.result_queue.get()) is not None:
            req_id, *_ = res
            # Wait for this result ready in self._results
            while req_id not in self._results:
                self._request_id_dispatcher_queue.get()
            self._results[req_id].queue.put(res)
            while not self._request_id_dispatcher_queue.empty():
                self._request_id_dispatcher_queue.get()

    def stats_main(self):
        while (stats := self.mp_stats_queue.get()) is not None:
            time.sleep(0.1)
            while self.stats_queue.full():
                self.stats_queue.get()
            self.stats_queue.put(stats)

    def start(self):
        self.mpi_futures = self.mpi_session.submit(
            ExecutorBindingsProxy.workers_main, **self.workers_kwargs)
        self.workers_started = True
        self.workers_init_ok = self.result_queue.get()
        if not self.workers_init_ok:
            raise RuntimeError("worker initialization failed")
        self.dispatcher.start()
        self.create_stats_queue()
        self.stats_thread.start()

    def shutdown(self):
        if not self.workers_started:
            return
        if self.workers_init_ok:
            self.request_queue.put(None)
        for f in self.mpi_futures:
            f.result()
        if self.dispatcher.is_alive():
            self.result_queue.put(None)
            self.dispatcher.join()
        if self.stats_thread.is_alive():
            self.mp_stats_queue.put(None)
            self.stats_thread.join()
        self.workers_started = False

    def submit(self, request: GenerationRequest) -> GenerationResult:
        """
            Low-level API to the executor. Return a "future" GenerationResult which can be waited.
            Forwards the request to the workers through the request queue.
        """
        if not self.workers_started:
            self.start()

        self.request_queue.put(request)

        # Await req id.
        req_id = self.request_id_queue.get()
        request.set_id(req_id)

        result = GenerationResult(request)
        self._results[req_id] = result
        self._request_id_dispatcher_queue.put(req_id)

        return result

    def __del__(self):
        self.shutdown()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.shutdown()
        return False
