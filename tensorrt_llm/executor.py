import asyncio
import secrets
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass
from multiprocessing.connection import Client, Listener
from pathlib import Path
from queue import Queue
from threading import Lock, Semaphore, Thread
from typing import Any, Dict, Generator, List, Optional, Set, Tuple, Union

import numpy as np
import torch
from janus import Queue as AsyncQueue
from mpi4py import MPI

from tensorrt_llm._utils import mpi_comm, mpi_rank, mpi_world_size
from tensorrt_llm.hlapi.mpi_session import MpiSession, find_free_port
from tensorrt_llm.hlapi.tokenizer import TokenizerBase, tokenizer_factory
from tensorrt_llm.hlapi.utils import (ContextManager, GenerationOutput,
                                      SamplingConfig, print_traceback_on_error)

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
                 sampling_config: Optional[SamplingConfig] = None):
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
        self.sampling_config = sampling_config or SamplingConfig()

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
                         default: Any = None,
                         value=None):
            if value is None:
                value = getattr(self.sampling_config, name, None)
                value = value if value is not None else default
            if value is not None:
                setattr(ir, name, torch.tensor([value], dtype=dtype))

        top_k = self.sampling_config.top_k[
            0] if self.sampling_config.top_k is not None else None

        top_p = self.sampling_config.top_p[
            0] if self.sampling_config.top_p is not None else None
        temperature = self.sampling_config.temperature[
            0] if self.sampling_config.temperature is not None else None
        max_new_tokens = [
            self.sampling_config.max_new_tokens
        ] if self.sampling_config.max_new_tokens is not None else None
        min_length = self.sampling_config.min_length[
            0] if self.sampling_config.min_length is not None else None
        end_id = self.tokenizer.eos_token_id if self.tokenizer is not None else None
        pad_id = self.tokenizer.pad_token_id if self.tokenizer is not None else None
        pad_id = end_id if pad_id is None else pad_id

        set_property("beam_width")
        set_property("max_new_tokens", default=[32], value=max_new_tokens)
        set_property("end_id", value=end_id)
        set_property("pad_id", value=pad_id)
        set_property("min_length", value=min_length)
        set_property("temperature", torch.float32, value=temperature)
        set_property("runtime_top_k", torch.float32, value=top_k)
        set_property("runtime_top_p", torch.float32, value=top_p)
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

        beam_width = generation_request.sampling_config.beam_width

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
        self, prompt: Union[str, List[int], List[str], List[List[int]]],
        streaming: bool, sampling_config: Union[SamplingConfig,
                                                List[SamplingConfig]]
    ) -> Union[GenerationResult, List[GenerationResult]]:
        unbatched = isinstance(prompt, str) or (isinstance(prompt, list)
                                                and isinstance(prompt[0], int))
        string_input = isinstance(
            prompt, str) or (not unbatched and isinstance(prompt[0], str))
        tokenizer = self.tokenizer if string_input else None

        if unbatched:
            results = self.submit(
                GenerationRequest(prompt,
                                  streaming,
                                  tokenizer,
                                  sampling_config=sampling_config))
        else:
            sampling_config = [sampling_config] * len(prompt) if not isinstance(
                sampling_config, list) else sampling_config
            results = []
            for idx, p in enumerate(prompt):
                results.append(
                    self.submit(
                        GenerationRequest(
                            p,
                            streaming,
                            tokenizer,
                            sampling_config=sampling_config[idx])))
        return results

    def generate(
        self,
        prompt: Union[str, List[int], List[str], List[List[int]]],
        streaming: bool = False,
        sampling_config: Optional[Union[SamplingConfig,
                                        List[SamplingConfig]]] = None
    ) -> Union[GenerationResult, List[GenerationResult]]:
        futures = self.generate_async(prompt,
                                      streaming=streaming,
                                      sampling_config=sampling_config)
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

    @dataclass
    class WorkerInitStatus:
        ok: bool
        info: Optional[str] = None
        rank: Optional[int] = None

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
            self._termination_lock.release()

            fetched = self.comm.bcast(fetched)

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


class GenerationExecutorProxy(GenerationExecutor):

    def __init__(
        self,
        workers_kwargs,
        model_world_size: int = 1,
        mpi_session: Optional[MpiSession] = None,
    ) -> None:
        super().__init__()

        self.workers_started = False
        self.tokenizer = tokenizer_factory(workers_kwargs["tokenizer"])

        request_queue_addr = ("127.0.0.1", find_free_port(),
                              secrets.token_bytes(512))
        self.request_queue = Fifo(request_queue_addr, is_server=True)
        result_queue_addr = ("127.0.0.1", find_free_port(),
                             secrets.token_bytes(512))
        self.result_queue = Fifo(result_queue_addr, is_server=True)

        self._results: Dict[int, GenerationResult] = {}

        if mpi_session is None:
            self.mpi_session = MpiSession(n_workers=model_world_size)
        else:
            self.mpi_session = mpi_session
        self.model_world_size = model_world_size

        self.workers_kwargs = workers_kwargs
        self.workers_kwargs.update({
            "request_queue_addr": request_queue_addr,
            "result_queue_addr": result_queue_addr,
        })
        self.dispatcher = Thread(target=self.dispatcher_thread)

    @print_traceback_on_error
    @staticmethod
    def workers_main(
        engine_dir: Path,
        tokenizer: Union[str, Path, TokenizerBase],
        request_queue_addr: Tuple[str, int, bytes],
        result_queue_addr: Tuple[str, int, bytes],
        max_beam_width: int = 1,
        executor_type: tllm.TrtGptModelType = tllm.TrtGptModelType.
        InflightBatching,
        executor_policy: tllm.SchedulerPolicy = tllm.SchedulerPolicy.
        GUARANTEED_NO_EVICT,
        executor_config: tllm.TrtGptModelOptionalParams = tllm.
        TrtGptModelOptionalParams()
    ) -> None:
        result_queue = None

        if mpi_rank() == 0:
            # Only rank0 need to communicate with the Python main process
            request_queue = Fifo(request_queue_addr, is_server=False)
            result_queue = Fifo(result_queue_addr, is_server=False)

        init_status = None
        try:
            executor = GenerationExecutorWorker(engine_dir, tokenizer,
                                                max_beam_width, executor_type,
                                                executor_policy,
                                                executor_config)
        except Exception as e:
            error_info = f"{str(e)}\nTraceback: {traceback.format_exc()}"
            init_status = GenerationExecutorWorker.WorkerInitStatus(
                ok=False, info=error_info, rank=mpi_rank())
            # Either one of the failed rank will occupy the result_queue comm and make the Python main process raise exception
            result_queue.put(init_status)
            raise e

        else:
            init_status = GenerationExecutorWorker.WorkerInitStatus(ok=True)

        finally:
            init_statuses = mpi_comm().gather(init_status, root=0)

            if mpi_rank() == 0 and all(status.ok for status in init_statuses):
                result_queue.put(init_status)

        with ContextManager(executor) as executor:
            executor.block_subordinates()
            if mpi_rank() == 0:
                executor.set_result_queue(result_queue)
            while (req := request_queue.get()) is not None:
                executor.submit(req)

        if mpi_rank() == 0:
            result_queue.put(None)

    def dispatcher_thread(self):
        """ Collect centralized results from result queue and dispatch them in the
            correct GenerationResult queues. """

        while (res := self.result_queue.get()) is not None:
            id, tensors, finished, err = res
            self._results[id].queue.put(
                (id,
                 {name: torch.tensor(value)
                  for name, value in tensors.items()}, finished, err))

    def start(self):
        self.mpi_futures = self.mpi_session.submit(
            GenerationExecutorProxy.workers_main, **self.workers_kwargs)
        self.workers_started = True

        # It will get the first failure status or get a success status if all ranks are successful
        ack: GenerationExecutorWorker.WorkerInitStatus = self.result_queue.get()
        if not ack.ok:
            raise RuntimeError(
                f"#node-{ack.rank}: worker initialization failed: {ack.info}")

        self.dispatcher.start()

    def shutdown(self):
        if not self.workers_started:
            return
        self.request_queue.put(None)
        for f in self.mpi_futures:
            f.result()
        self.dispatcher.join()
        self.workers_started = False

    def submit(self, request: GenerationRequest) -> GenerationResult:
        """
            Low-level API to the executor. Return a "future" GenerationResult which can be waited.
            Forwards the request to the workers through the request queue.
        """
        if not self.workers_started:
            self.start()

        req_id = self.generate_id()
        request.set_id(req_id)

        tokenizer = request.tokenizer
        result = GenerationResult(request, tokenizer)
        self._results[req_id] = result

        # no need to send the tokenizer to the executor,
        # saves communication time
        request.tokenizer = None
        self.request_queue.put(request)
        request.tokenizer = tokenizer

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
