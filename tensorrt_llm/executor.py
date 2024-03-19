import asyncio
import secrets
from abc import ABC, abstractmethod
from multiprocessing.managers import BaseManager
from pathlib import Path
from queue import Empty, Queue
from threading import Lock, Semaphore, Thread
from typing import Any, Dict, Generator, List, Optional, Set, Tuple, Union

import numpy as np
import torch
from janus import Queue as AsyncQueue
from mpi4py import MPI

from tensorrt_llm._utils import mpi_rank, mpi_world_size
from tensorrt_llm.hlapi.mpi_session import MpiSession, find_free_port
from tensorrt_llm.hlapi.tokenizer import TokenizerBase, tokenizer_factory
from tensorrt_llm.hlapi.utils import GenerationOutput

from . import bindings as tllm


def has_event_loop() -> bool:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return False
    return True


class GenerationRequest:

    def __init__(self,
                 ids_or_prompt: Union[torch.Tensor, np.ndarray, list, str],
                 streaming: bool = True,
                 tokenizer: Optional[TokenizerBase] = None,
                 **kwargs):
        if isinstance(ids_or_prompt, str):
            assert tokenizer is not None, "GenerationRequest constructor with str prompt requires a tokenizer argument"
            self.input_ids = (tokenizer.encode(ids_or_prompt,
                                               return_tensors="pt",
                                               return_attention_mask=False).to(
                                                   torch.int32).numpy())
        else:
            if isinstance(ids_or_prompt, list):
                self.input_ids = np.array(ids_or_prompt, dtype="int32")
            elif isinstance(ids_or_prompt, torch.Tensor):
                self.input_ids = ids_or_prompt.to(torch.int32).numpy()
            elif isinstance(ids_or_prompt, np.ndarray):
                self.input_ids = ids_or_prompt
            else:
                raise ValueError(
                    f"ids_or_prompt (={ids_or_prompt}) should be an instance of str, torch.Tensor, np.ndarray or list"
                )

        self.tokenizer = tokenizer
        self.streaming = streaming
        self.options = kwargs
        if tokenizer is not None:
            end_id, pad_id = tokenizer.eos_token_id, tokenizer.pad_token_id
            self.options.setdefault("end_id", end_id)
            self.options.setdefault("pad_id",
                                    pad_id if pad_id is not None else end_id)

        self.id = -1

    def set_id(self, id):
        self.id = id
        return self

    def as_inference_request(self) -> tllm.InferenceRequest:
        ir = tllm.InferenceRequest(self.id)
        ir.input_ids = torch.from_numpy(self.input_ids)
        ir.is_streaming = self.streaming

        def set_property(name: str,
                         dtype: torch.dtype = torch.int32,
                         default: Any = None):
            if name in self.options or default is not None:
                value = self.options.get(name, default)
                setattr(ir, name, torch.tensor([value], dtype=dtype))

        if "max_new_tokens" in self.options:
            self.options["max_new_tokens"] = [self.options["max_new_tokens"]]
        set_property("beam_width")
        set_property("max_new_tokens", default=[32])
        set_property("end_id")
        set_property("pad_id")
        set_property("min_length")
        set_property("temperature", torch.float32)
        set_property("runtime_top_k", torch.float32)
        set_property("runtime_top_p", torch.float32)
        set_property("random_seed", torch.int64)

        return ir


class GenerationResult(GenerationOutput):

    def __init__(self,
                 generation_request: GenerationRequest,
                 tokenizer: Optional[TokenizerBase] = None) -> None:
        self._done = False
        self._cancelled = False
        self.generation_request = generation_request
        self.tokenizer = tokenizer
        self.streaming = generation_request.streaming

        if has_event_loop():
            aqueue = AsyncQueue()
            self.queue = aqueue.sync_q
            self.aqueue = aqueue.async_q
        else:
            self.queue = Queue()
            self.aqueue = None

        beam_width = generation_request.options.get("beam_width", 1)
        self.beam_search_enabled = beam_width > 1
        self._token_ids = [[] for _ in range(beam_width)]

        self.logprobs = []
        self.last_text = ""

    @property
    def token_ids(self):
        if not self.beam_search_enabled:
            return self._token_ids[0]
        return self._token_ids

    def handle_generation_msg(self, tensors: Dict[str, np.ndarray], error: str):
        if error:
            raise RuntimeError(error)
        new_ids = tensors["output_ids"].squeeze(0).tolist()
        for idx, beam_ids in enumerate(new_ids):
            self._token_ids[idx] += beam_ids

    def result_step(self, timeout: Optional[float] = None):
        _, tensors, self._done, error = self.queue.get(timeout=timeout)
        self.handle_generation_msg(tensors, error)

    async def aresult_step(self):
        assert self.aqueue is not None
        _, tensors, self._done, error = await self.aqueue.get()
        self.handle_generation_msg(tensors, error)

    @property
    def text_diff(self) -> str:
        assert self.streaming is not None
        assert not self.beam_search_enabled, "text_diff is not supported with beam_search"

        new_txt = self.text
        diff = new_txt[len(self.last_text):]
        self.last_text = new_txt
        return diff

    @property
    def text(self) -> Union[str, List[str]]:
        if self.tokenizer is None:
            return ''
        texts = self.tokenizer.batch_decode(self._token_ids)
        if not self.beam_search_enabled:
            return texts[0]
        return texts

    def result(self, timeout: Optional[float] = None) -> "GenerationResult":
        while not self._done:
            self.result_step(timeout)
        return self

    async def aresult(self) -> "GenerationResult":
        while not self._done:
            await self.aresult_step()
        return self

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


class GenerationExecutor(ABC):
    TERMINATE_REQUEST_ID = 0

    def __init__(self):
        self.id_counter = GenerationExecutor.TERMINATE_REQUEST_ID + 1
        self.tokenizer = None

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
            self, prompt: Union[str, List[int], List[str],
                                List[List[int]]], streaming: bool,
            **kwargs: Any) -> Union[GenerationResult, List[GenerationResult]]:
        unbatched = isinstance(prompt, str) or (isinstance(prompt, list)
                                                and isinstance(prompt[0], int))
        string_input = isinstance(
            prompt, str) or (not unbatched and isinstance(prompt[0], str))
        tokenizer = self.tokenizer if string_input else None

        if unbatched:
            results = self.submit(
                GenerationRequest(prompt, streaming, tokenizer, **kwargs))
        else:
            results = []
            for idx, p in enumerate(prompt):
                request_kwargs = {
                    k: v[idx] if isinstance(v, list) else v
                    for k, v in kwargs.items()
                }
                results.append(
                    self.submit(
                        GenerationRequest(p, streaming, tokenizer,
                                          **request_kwargs)))
        return results

    def generate(
            self,
            prompt: Union[str, List[int], List[str], List[List[int]]],
            streaming: bool = False,
            **kwargs: Any) -> Union[GenerationResult, List[GenerationResult]]:
        futures = self.generate_async(prompt, streaming=streaming, **kwargs)
        if isinstance(futures, GenerationRequest):
            futures.result()
        else:
            for future in futures:
                future.result()
        return futures

    @abstractmethod
    def shutdown(self):
        pass

    @abstractmethod
    def get_stats(self):
        pass

    @abstractmethod
    async def aget_stats(self):
        pass

    @staticmethod
    def create(
        engine_dir: Path,
        tokenizer: Union[str, Path, TokenizerBase],
        max_beam_width: int = 1,
        executor_type: tllm.TrtGptModelType = tllm.TrtGptModelType.
        InflightBatching,
        executor_policy: tllm.SchedulerPolicy = tllm.SchedulerPolicy.
        GUARANTEED_NO_EVICT,
        executor_config: tllm.TrtGptModelOptionalParams = tllm.
        TrtGptModelOptionalParams(),
        model_world_size: int = 1,
        world_size: int = 0,
        mpi_session: Optional[MpiSession] = None,
    ) -> Union["GenerationExecutorProxy", "GenerationExecutorWorker"]:

        if world_size == 0:
            world_size = mpi_world_size()

        if world_size > 1 and world_size < model_world_size:
            raise RuntimeError(
                "Cannot instantiate Generator for engine built "
                f"for {model_world_size} ranks, while currently running "
                f"on {world_size} ranks.")

        worker_kwargs = {
            "engine_dir": engine_dir,
            "tokenizer": tokenizer,
            "max_beam_width": max_beam_width,
            "executor_type": executor_type,
            "executor_policy": executor_policy,
            "executor_config": executor_config,
        }

        if world_size == 1 and model_world_size > 1:
            return GenerationExecutorProxy(worker_kwargs,
                                           model_world_size=model_world_size,
                                           mpi_session=mpi_session)

        return GenerationExecutorWorker(**worker_kwargs)


class GenerationExecutorWorker(GenerationExecutor):

    class WorkerExit(GeneratorExit):
        pass

    def __init__(
        self,
        engine_dir: Path,
        tokenizer: Union[str, Path, TokenizerBase, None],
        max_beam_width: int = 1,
        executor_type: tllm.TrtGptModelType = tllm.TrtGptModelType.
        InflightBatching,
        executor_policy: tllm.SchedulerPolicy = tllm.SchedulerPolicy.
        GUARANTEED_NO_EVICT,
        executor_config: tllm.TrtGptModelOptionalParams = tllm.
        TrtGptModelOptionalParams(),
    ) -> None:
        super().__init__()

        self.engine = None
        self.tokenizer = tokenizer_factory(tokenizer)

        # NOTE: underscore variables are used for communication with the C++ runtime
        self._requests: List[tllm.InferenceRequest] = []
        self._results: Dict[int, GenerationResult] = {}
        self._cancelled_ids: Set[int] = set()
        self._pending: set = set()
        if has_event_loop():
            self._stats = AsyncQueue()
            self.stats_queue = self._stats.sync_q
            self.stats_aqueue = self._stats.async_q
        else:
            self._stats = Queue()
            self.stats_queue = self._stats
            self.stats_aqueue = None
        """
            Note: in single-node only (when using .block_subordinates()) the termination
            process is as follow:
                0. Nodes > 0 (main threads) directly wait on termination_ack. Node 0 continues execution.
                1. Node 0 (main thread) is finishing and must close GptManager.
                2. Node 0 (main thread) sets _termination_requested and wait on termination_ack
                3. Node 0 (BatchManager thread) exchange _termination_requested via MPI.bcast with all other nodes.
                4. All nodes (BatchManager threads) signal the _termination_ack semaphore and set _termination_pending to avoid fetching new requests.
                5. All nodes (main threads) go through _termination_ack and ask BatchManager to join its threads.
        """
        self._block_subordinates = False
        self._termination_requested = False
        self._termination_pending = False
        self._termination_ack = Semaphore(0)
        self._termination_lock = Lock()
        self.result_queue = None

        self.comm = MPI.COMM_WORLD
        self.rank = mpi_rank()

        self.engine = tllm.GptManager(engine_dir, executor_type, max_beam_width,
                                      executor_policy, self.fetch_requests,
                                      self.handle_response,
                                      self.get_cancelled_ids, self.handle_stats,
                                      executor_config,
                                      GenerationExecutor.TERMINATE_REQUEST_ID)

    def shutdown(self):
        if self.engine is not None:
            self.engine.shutdown()
            self.engine = None

    def block_subordinates(self):
        self._block_subordinates = True
        if self.rank != 0:
            self._termination_ack.acquire()
            self.shutdown()
            raise self.WorkerExit(
                "block_subordinates() should be used in a `with GenerationExecutorWorker() as ...:` block"
            )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        del exc_value, traceback  # unused arguments

        if self._block_subordinates and self.rank == 0:
            if self.rank == 0:
                self._termination_lock.acquire()
                self._termination_requested = True
                self._termination_lock.release()

                self._termination_ack.acquire()

        self.shutdown()

        return exc_type is None or exc_type == GenerationExecutorWorker.WorkerExit

    def submit(self, request: GenerationRequest) -> GenerationResult:
        """
            Low-level API to the executor. Return a "future" GenerationResult which can be waited.
        """
        result = GenerationResult(request, request.tokenizer)
        req_id = self.generate_id()

        request.set_id(req_id)
        self._results[req_id] = result
        self._pending.add(req_id)
        self._requests.append(request.as_inference_request())

        return result

    def get_stats(self):
        return self.stats_queue.get()

    async def aget_stats(self):
        assert self.stats_aqueue is not None
        return await self.stats_aqueue.get()

    def wait_first_completed(
        self, futures: List[GenerationResult]
    ) -> Generator[GenerationResult, None, None]:
        wait_set = set(f.generation_request.id for f in futures)

        # clear already-finished requests
        for f in futures:
            if f._done:
                wait_set.remove(f.generation_request.id)
                yield f

        # wait remaining active requests
        while len(wait_set) > 0:
            req_id = wait_set.pop()

            if req_id not in self._pending:
                yield self._results[req_id]
            else:
                wait_set.add(req_id)

    def set_result_queue(self, queue):
        self.result_queue = queue

    def return_queue(self, req_id: int):
        """ If a centralized result queue is registered (used for communication with the proxy)
            send the message there.
            Otherwise, push the result directly in the GenerationResult queue.
        """

        if self.result_queue is not None:
            return self.result_queue
        return self._results[req_id].queue

    # Callbacks for BatchManager
    def fetch_requests(self, max_num_sequences) -> List[tllm.InferenceRequest]:
        if self._termination_pending:
            return []

        fetched = []
        if not self._block_subordinates or self.rank == 0:
            for _ in range(max_num_sequences):
                if len(self._requests) == 0:
                    break
                fetched.append(self._requests.pop())

        if self._block_subordinates:
            self._termination_lock.acquire()
            self._termination_requested = self.comm.bcast(
                self._termination_requested)

            if self._termination_requested:
                self._termination_ack.release()
                self._termination_pending = True
            else:
                fetched = self.comm.bcast(fetched)

            self._termination_lock.release()

        return fetched

    def handle_response(self, req_id: int, tensors: List[tllm.NamedTensor],
                        finished: bool, err: str) -> None:
        if self._block_subordinates and self.rank != 0:
            return

        self.return_queue(req_id).put((req_id, {
            t.name: t.tensor.numpy()
            for t in tensors if t.tensor is not None
        }, finished, err))
        if finished:
            self._pending.remove(req_id)

    def get_cancelled_ids(self) -> Set[int]:
        return self._cancelled_ids

    def handle_stats(self, stats: str):
        while self.stats_queue.full():
            self.stats_queue.get()

        self.stats_queue.put(stats)

    def __del__(self):
        self.shutdown()


class GenerationExecutorProxy(GenerationExecutor):

    class ExecutorManager(BaseManager):
        pass

    def __init__(
        self,
        workers_kwargs,
        model_world_size: int = 1,
        mpi_session: Optional[MpiSession] = None,
    ) -> None:
        super().__init__()

        self.workers_started = False
        self.tokenizer = tokenizer_factory(workers_kwargs["tokenizer"])

        manager_address = ("localhost", find_free_port())
        manager_secret = secrets.token_bytes(512)
        self.manager = GenerationExecutorProxy.ExecutorManager(
            manager_address, manager_secret)
        request_queue, result_queue = Queue(), Queue()
        GenerationExecutorProxy.ExecutorManager.register(
            "request_queue", lambda: request_queue)
        GenerationExecutorProxy.ExecutorManager.register(
            "result_queue", lambda: result_queue)
        self.manager.start()
        self._results: Dict[int, GenerationResult] = {}

        if mpi_session is None:
            self.mpi_session = MpiSession(n_workers=model_world_size)
        else:
            self.mpi_session = mpi_session
        self.model_world_size = model_world_size

        self.workers_kwargs = workers_kwargs
        self.workers_kwargs.update({
            "manager_address": manager_address,
            "manager_secret": manager_secret
        })
        self.dispatcher = Thread(target=self.dispatcher_thread)

    @staticmethod
    def workers_main(engine_dir: Path,
                     tokenizer: Union[str, Path, TokenizerBase],
                     max_beam_width: int = 1,
                     executor_type: tllm.TrtGptModelType = tllm.TrtGptModelType.
                     InflightBatching,
                     executor_policy: tllm.SchedulerPolicy = tllm.
                     SchedulerPolicy.GUARANTEED_NO_EVICT,
                     executor_config: tllm.TrtGptModelOptionalParams = tllm.
                     TrtGptModelOptionalParams(),
                     manager_address: Tuple[str, int] = ("", 0),
                     manager_secret: bytes = b"") -> None:

        with GenerationExecutorWorker(engine_dir, tokenizer, max_beam_width,
                                      executor_type, executor_policy,
                                      executor_config) as executor:
            executor.block_subordinates()

            manager = GenerationExecutorProxy.ExecutorManager(
                manager_address, manager_secret)
            GenerationExecutorProxy.ExecutorManager.register("request_queue")
            GenerationExecutorProxy.ExecutorManager.register("result_queue")
            manager.connect()
            request_queue = manager.request_queue()
            manager.result_queue().put(True)

            executor.set_result_queue(manager.result_queue())
            while (req := request_queue.get()) is not None:
                executor.submit(req)

    def dispatcher_thread(self):
        """ Collect centralized results from Manager's result queue and dispatch them in the
            correct GenerationResult queues. """

        while (res := self.manager.result_queue().get()) is not None:
            id, tensors, finished, err = res
            self._results[id].queue.put(
                (id,
                 {name: torch.tensor(value)
                  for name, value in tensors.items()}, finished, err))

    def start(self):
        self.mpi_futures = self.mpi_session.submit(
            GenerationExecutorProxy.workers_main, **self.workers_kwargs)
        self.workers_started = True

        while True:
            ack = False
            try:
                ack = self.manager.result_queue().get(timeout=0.5)
            except Empty:
                pass
            if not ack:
                if any(f.done() for f in self.mpi_futures):
                    self.shutdown()
                    raise RuntimeError(
                        "GenerationExecutorWorker has exited unexpectedly")
            else:
                break

        self.dispatcher.start()

    def shutdown(self):
        if not self.workers_started:
            return
        self.manager.request_queue().put(None)
        self.manager.result_queue().put(None)
        for f in self.mpi_futures:
            f.result()
        self.dispatcher.join()
        self.workers_started = False

    def submit(self, request: GenerationRequest) -> GenerationResult:
        """
            Low-level API to the executor. Return a "future" GenerationResult which can be waited.
            Forwards the request to the workers through the Manager's request queue.
        """
        if not self.workers_started:
            self.start()

        req_id = self.generate_id()
        request.set_id(req_id)

        result = GenerationResult(request, request.tokenizer)
        self._results[req_id] = result

        self.manager.request_queue().put(request)

        return result

    def get_stats(self):
        pass

    async def aget_stats(self):
        pass

    def __del__(self):
        self.shutdown()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.shutdown()
        return False
