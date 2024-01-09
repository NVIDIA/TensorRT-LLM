from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Optional

import janus
import torch

import tensorrt_llm.bindings as tllm

from .hlapi.utils import GenerationOutput
from .logger import logger
from .runtime import SamplingConfig

# TODO[chuweiy]: fix circle import later
# from .hlapi.tokenizer import TokenizerBase


class AsyncLLMEngine:
    TERMINATE_REQUEST_ID = 0

    def __init__(self,
                 engine_dir: Path,
                 tokenizer: str | Path | Any,
                 max_beam_width: int = 1) -> None:
        self.requests: list[tllm.InferenceRequest] = []
        self.results: dict[int, janus.Queue] = {}
        self.stop_set: set[int] = set()
        self.stats: Optional[janus.LifoQueue] = None

        from .hlapi.tokenizer import TokenizerBase

        self.tokenizer = tokenizer
        if not isinstance(tokenizer, TokenizerBase):
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer,
                legacy=False,
                padding_side='left',
                truncation_side='left',
                trust_remote_code=True,
                use_fast=True)
        opt_params = tllm.TrtGptModelOptionalParams()
        # TODO[chunweiy]: Expose the runtime configs
        self.engine = tllm.GptManager(
            engine_dir, tllm.TrtGptModelType.InflightFusedBatching,
            max_beam_width, tllm.SchedulerPolicy.GUARANTEED_NO_EVICT,
            self._fetch_requests_callback, self._handle_response_callback,
            self._get_stop_set_callback, self._handle_stats_callback,
            opt_params, AsyncLLMEngine.TERMINATE_REQUEST_ID)

        self._next_request_id = AsyncLLMEngine.TERMINATE_REQUEST_ID + 1

    # TODO[chunweiy]: support token-ids as prompt when Tokenizer is disabled in LLM()
    # TODO[chunweiy]: Align the keys between SamplingConfig and gptManager
    async def generate(
        self,
        prompt: str,
        streaming: bool = True,
        sampling_config: Optional[SamplingConfig] = None
    ) -> Iterable[GenerationOutput]:

        sampling_options: dict = asdict(
            sampling_config) if sampling_config is not None else dict()
        if sampling_options:
            sampling_options["max_new_tokens"] = [
                sampling_options['max_new_tokens']
            ]

        tllm_request = self.add_request({
            "prompt": prompt,
            "streaming": streaming,
            **sampling_options
        })
        request_id = tllm_request.request_id
        tllm_request.input_ids[0].numpy().tolist()

        finished = False
        while not finished:
            output, finished = await self.get_response(request_id)
            diff_ids = output.numpy().tolist()
            diff_str = self.tokenizer.decode(diff_ids)

            output = GenerationOutput(
                diff_str,
                diff_ids,
                # TODO[chunweiy]: return the probs as well
            )
            yield output

    @property
    def next_request_id(self) -> int:
        # underlying type is uint64
        uint64_max = 2**64 - 1
        request_id = self._next_request_id
        self._next_request_id = (request_id + 1) % uint64_max
        return request_id

    @staticmethod
    def create_inference_request(
            req_id: int, parameters: dict[str, Any]) -> tllm.InferenceRequest:

        def set_property(name: str, dtype: torch.dtype = torch.int32):
            if name in parameters and parameters[name] is not None:
                setattr(request, name,
                        torch.tensor([parameters[name]], dtype=dtype))

        request = tllm.InferenceRequest(req_id)
        request.input_ids = parameters["input_ids"]
        set_property("end_id")
        set_property("pad_id")
        set_property("max_new_tokens")
        set_property("min_length")
        set_property("temperature", torch.float32)
        set_property("runtime_top_k", torch.float32)
        set_property("runtime_top_p", torch.float32)
        set_property("random_seed", torch.int64)
        if "streaming" in parameters:
            request.is_streaming = parameters["streaming"]

        return request

    def add_request(self, request_dict: dict[str,
                                             Any]) -> tllm.InferenceRequest:
        ids = self.tokenizer(request_dict.pop("prompt"),
                             return_tensors="pt",
                             return_attention_mask=False)
        request_dict["input_ids"] = ids["input_ids"].to(torch.int32)
        request_dict["end_id"] = self.tokenizer.eos_token_id
        if getattr(self.tokenizer, "pad_token_id") is not None:
            request_dict["pad_id"] = self.tokenizer.pad_token_id
        else:
            request_dict["pad_id"] = request_dict["end_id"]

        request = AsyncLLMEngine.create_inference_request(
            self.next_request_id, request_dict)

        self.results[request.request_id] = janus.Queue()
        self.requests.append(request)

        return request

    async def get_response(self,
                           request_id: int) -> tuple[dict[str, Any], bool]:
        outputs, finished = None, False
        while outputs is None:
            outputs, finished = await self.results[request_id].async_q.get()

        last_idx = outputs["sequence_length"][0, 0].item()
        output = outputs["output_ids"][0, 0, :last_idx]

        if finished:
            self.results.pop(request_id)

        return output, finished

    # Callbacks for BatchManager

    def _fetch_requests_callback(
            self, max_num_sequences) -> list[tllm.InferenceRequest]:
        fetched = []
        for _ in range(max_num_sequences):
            if len(self.requests) == 0:
                break
            fetched.append(self.requests.pop())
        return fetched

    def _handle_response_callback(self, req_id: int,
                                  tensors: list[tllm.NamedTensor], is_ok: bool,
                                  err_msg: str) -> None:
        if err_msg:
            logger.error(f"AsyncLLMEngine process request failed: {err_msg}")

        self.results[req_id].sync_q.put(
            [{t.name: t.tensor
              for t in tensors}, is_ok] if not err_msg else err_msg)

    def _get_stop_set_callback(self) -> set[int]:
        return self.stop_set

    def _handle_stats_callback(self, stats: str):
        # TODO[chunweiy]: fix this
        return
        if self.stats is None:
            self.stats = janus.LifoQueue()

        while self.stats.sync_q.full():
            self.stats.sync_q.get()

        self.stats.sync_q.put(stats)
