import asyncio
from pathlib import Path
from queue import Queue
from typing import Any, Generator, Optional

import torch
from janus import Queue as AsyncQueue
from transformers import AutoTokenizer

import tensorrt_llm.bindings as tllm
from tensorrt_llm.hlapi.tokenizer import TokenizerBase
from tensorrt_llm.hlapi.utils import GenerationOutput


def has_event_loop() -> bool:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return False
    return True


class GenerationRequest:

    def __init__(self,
                 prompt_or_ids: str | torch.Tensor,
                 streaming: bool = True,
                 **kwargs):
        self.prompt = None
        self.ids = None
        self.streaming = streaming
        self.kwargs = kwargs

        if isinstance(prompt_or_ids, str):
            self.prompt = prompt_or_ids
        else:
            self.ids = prompt_or_ids

    def bind_executor(self, executor: "GenerationExecutor"):
        self.executor = executor
        self._id = self.executor.get_next_request_id()

    def get_inference_request(self) -> tllm.InferenceRequest:
        if self.ids is None:
            self.ids = self.executor.tokenizer.encode(
                self.prompt, return_tensors="pt", return_attention_mask=False)

        ir = tllm.InferenceRequest(self._id)
        ir.input_ids = self.ids.to(dtype=torch.int32)
        ir.is_streaming = self.streaming

        def set_property(name: str,
                         dtype: torch.dtype = torch.int32,
                         default: Any = None):
            if name in self.kwargs or default is not None:
                value = self.kwargs.get(name, default)
                setattr(ir, name, torch.tensor([value], dtype=dtype))

        set_property("max_new_tokens", default=[8])

        default_end_id = self.executor.tokenizer.eos_token_id
        default_pad_id = getattr(self.executor.tokenizer, "pad_token_id",
                                 default_end_id)
        set_property("end_id", default=default_end_id)
        set_property("pad_id", default=default_pad_id)

        set_property("min_length")
        set_property("temperature", torch.float32)
        set_property("runtime_top_k", torch.float32)
        set_property("runtime_top_p", torch.float32)
        set_property("random_seed", torch.int64)

        return ir


class GenerationResult(GenerationOutput):

    def __init__(self, generation_request: GenerationRequest) -> None:
        self.running = True
        self.done = False
        self.generation_request = generation_request

        if has_event_loop():
            self._base_queue = AsyncQueue()
            self.queue = self._base_queue.sync_q
            self.aqueue = self._base_queue.async_q
        else:
            self._base_queue = Queue()
            self.queue = self._base_queue
            self.aqueue = None

        self.generation: Optional[torch.Tensor]
        if generation_request.streaming:
            self.generation = generation_request.ids
        else:
            self.generation = None

        # TODO: fill the following fields from GenerationOutput
        self.token_ids = []
        self.logprobs = []

    def enqueue(self, msg: tuple[str | dict[str, torch.Tensor], bool]):
        self.queue.put(msg)

    def handle_generation_msg(self, msg: str | dict[str, torch.Tensor]):
        if isinstance(msg, str):
            raise RuntimeError(msg)

        if self.generation is None:
            self.generation = msg["output_ids"][0]
        else:
            self.generation = torch.cat([self.generation, msg["output_ids"][0]],
                                        dim=-1)

        self.token_ids = msg["output_ids"][0]

    def wait_step(self, timeout: float | None = None):
        msg, self.done = self.queue.get(timeout=timeout)
        self.handle_generation_msg(msg)

    async def await_step(self):
        assert self.aqueue is not None
        msg, self.done = await self.aqueue.get()
        self.handle_generation_msg(msg)

    @property
    def text(self) -> str:
        assert self.generation is not None
        return self.generation_request.executor.tokenizer.decode(
            self.generation[0])

    def wait_completion(self,
                        timeout: float | None = None) -> "GenerationResult":
        while not self.done:
            self.wait_step(timeout)
        return self

    async def await_completion(self) -> "GenerationResult":
        while not self.done:
            await self.await_step()
        return self

    def __iter__(self):
        return self

    def __next__(self):
        if self.done:
            raise StopIteration

        self.wait_step()
        return self

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.done:
            raise StopAsyncIteration

        await self.await_step()
        return self


class GenerationExecutor:
    TERMINATE_REQUEST_ID = 0

    def __init__(
        self,
        engine_dir: Path,
        tokenizer: str | Path | TokenizerBase,
        max_beam_width: int = 1,
        executor_type: tllm.TrtGptModelType = tllm.TrtGptModelType.
        InflightBatching,
        executor_policy: tllm.SchedulerPolicy = tllm.SchedulerPolicy.
        GUARANTEED_NO_EVICT,
        executor_config: tllm.TrtGptModelOptionalParams = tllm.
        TrtGptModelOptionalParams(),
    ) -> None:

        self.active_requests = 0

        self.tokenizer = tokenizer
        if not isinstance(tokenizer, TokenizerBase):
            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer,
                legacy=False,
                padding_side='left',
                truncation_side='left',
                trust_remote_code=True,
                use_fast=True)

        # NOTE: underscore variables are used for communication with the C++ runtime
        self._requests: list[tllm.InferenceRequest] = []
        self._results: dict[int, GenerationResult] = {}
        self._cancelled_ids: set[int] = set()
        self._completed: Queue = Queue()
        if has_event_loop():
            self._stats = AsyncQueue()
            self.stats_queue = self._stats.sync_q
            self.stats_aqueue = self._stats.async_q
        else:
            self._stats = Queue()
            self.stats_queue = self._stats
            self.stats_aqueue = None

        self.engine = tllm.GptManager(engine_dir, executor_type, max_beam_width,
                                      executor_policy, self.fetch_requests,
                                      self.handle_response,
                                      self.get_cancelled_ids, self.handle_stats,
                                      executor_config,
                                      GenerationExecutor.TERMINATE_REQUEST_ID)

        self._next_request_id = GenerationExecutor.TERMINATE_REQUEST_ID + 1

    def submit(self, request: GenerationRequest) -> GenerationResult:
        """
            Low-level API to the executor. Return a "future" GenerationResult which can be waited.
        """

        request.bind_executor(self)
        inference_request = request.get_inference_request()

        result = GenerationResult(request)
        self._results[inference_request.request_id] = result

        self.active_requests += 1
        self._requests.append(inference_request)

        return result

    def get_next_request_id(self) -> int:
        # underlying type is uint64
        uint64_max = 2**64 - 1
        request_id = self._next_request_id
        self._next_request_id = (request_id + 1) % uint64_max
        return request_id

    def generate_async(
        self, prompt: str | list[str], streaming: bool,
        max_new_tokens: int | list[int]
    ) -> GenerationResult | list[GenerationResult]:
        unbatched = isinstance(prompt, str)
        if unbatched:
            assert isinstance(max_new_tokens, int)
            prompt = [prompt]
            max_new_tokens = [max_new_tokens]

        results = [
            self.submit(
                GenerationRequest(p, streaming=streaming, max_new_tokens=[m]))
            for p, m in zip(prompt, max_new_tokens)
        ]
        if unbatched:
            results = results[0]
        return results

    def generate(
        self, prompt: str | list[str], max_new_tokens: int | list[int]
    ) -> GenerationResult | list[GenerationResult]:
        results = self.generate_async(prompt, False, max_new_tokens)
        result_list = [results] if isinstance(results,
                                              GenerationRequest) else results
        for result in result_list:
            result.wait_completion()
        return results

    def get_stats(self):
        return self.stats_queue.get()

    async def aget_stats(self):
        assert self.stats_aqueue is not None
        return await self.stats_aqueue.get()

    def wait_first_completed(
        self, futures: list[GenerationResult]
    ) -> Generator[GenerationResult, None, None]:
        wait_set = set(f.generation_request._id for f in futures)

        # clear already-finished requests
        for f in futures:
            if f.done:
                wait_set.remove(f.generation_request._id)
                yield f

        # wait remaining active requests
        while len(wait_set) > 0:
            req_id = self._completed.get()
            if req_id in wait_set:
                wait_set.remove(req_id)
                yield self._results[req_id]

    # Callbacks for BatchManager
    def fetch_requests(self, max_num_sequences) -> list[tllm.InferenceRequest]:
        fetched = []
        for _ in range(max_num_sequences):
            if len(self._requests) == 0:
                break
            fetched.append(self._requests.pop())
        return fetched

    def handle_response(self, req_id: int, tensors: list[tllm.NamedTensor],
                        finished: bool, err: str) -> None:
        self._results[req_id].enqueue(
            ({t.name: t.tensor
              for t in tensors
              if t.tensor is not None} if not err else err, finished))
        if finished:
            self._completed.put(req_id)

    def get_cancelled_ids(self) -> set[int]:
        return self._cancelled_ids

    def handle_stats(self, stats: str):
        while self.stats_queue.full():
            self.stats_queue.get()

        self.stats_queue.put(stats)
