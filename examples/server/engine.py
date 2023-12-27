import random
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import torch
from janus import LifoQueue, Queue
from transformers import AutoTokenizer

import tensorrt_llm.bindings as tllm
from tensorrt_llm.hlapi.llm import LLM, ModelConfig


class AsyncLLMEngine:
    TERMINATE_REQUEST_ID = 0

    def __init__(self,
                 engine_dir: Path,
                 tokenizer: str | Path,
                 max_beam_width: int = 1,
                 max_num_sequences: int = 10) -> None:
        self.requests: list[tllm.InferenceRequest] = []
        self.results: dict[int, Queue] = {}
        self.stop_set: set[int] = set()
        self.stats: LifoQueue = LifoQueue()
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer,
                                                       legacy=False,
                                                       padding_side='left',
                                                       truncation_side='left',
                                                       trust_remote_code=True,
                                                       use_fast=True)
        opt_params = tllm.TrtGptModelOptionalParams(
            max_num_sequences=max_num_sequences)
        self.engine = tllm.GptManager(
            engine_dir, tllm.TrtGptModelType.InflightBatching, max_beam_width,
            tllm.SchedulerPolicy.MAX_UTILIZATION, self.fetch_requests,
            self.handle_response, self.get_stop_set, self.handle_stats,
            opt_params, AsyncLLMEngine.TERMINATE_REQUEST_ID)

    @staticmethod
    def from_hf_dir(model_dir: str | Path):
        config = ModelConfig(model_dir=str(model_dir))
        config.build_config.plugin_config.set_gemm_plugin()
        config.build_config.plugin_config.set_context_fmha()
        config.build_config.plugin_config.set_gpt_attention_plugin()
        config.build_config.plugin_config.enable_paged_kv_cache()
        config.build_config.plugin_config.enable_remove_input_padding()

        engine_dir = TemporaryDirectory()
        LLM(config).save(engine_dir.name)
        engine = AsyncLLMEngine(Path(engine_dir.name), model_dir)
        # Reference the tmp dir in the object so it's cleaned once the engine disappears
        setattr(engine, "_tmp_dir", engine_dir)

        return engine

    @staticmethod
    def gen_id() -> int:
        # underlying type is uint64
        uint64_max = 2**64 - 1
        return random.randint(AsyncLLMEngine.TERMINATE_REQUEST_ID + 1,
                              uint64_max)

    @staticmethod
    def create_inference_request(
            req_id: int, parameters: dict[str, Any]) -> tllm.InferenceRequest:

        def set_property(name: str, dtype: torch.dtype = torch.int32):
            if name in parameters:
                setattr(request, name,
                        torch.tensor([parameters[name]], dtype=dtype))

        request = tllm.InferenceRequest(req_id)
        request.input_ids = parameters["input_ids"]
        set_property("max_new_tokens")
        set_property("end_id")
        set_property("pad_id")
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
            AsyncLLMEngine.gen_id(), request_dict)

        self.results[request.request_id] = Queue()
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

    async def generate(self,
                       prompt: str,
                       max_new_tokens: int,
                       streaming: bool = True):
        tllm_request = self.add_request({
            "prompt": prompt,
            "max_new_tokens": [max_new_tokens],
            "streaming": streaming
        })
        request_id = tllm_request.request_id
        current_tokens = tllm_request.input_ids[0].numpy().tolist()
        current_str = self.tokenizer.decode(current_tokens)

        finished = False
        while not finished:
            output, finished = await self.get_response(request_id)

            current_tokens += output.numpy().tolist()
            new_str = self.tokenizer.decode(current_tokens)
            diff_str = new_str[len(current_str):]
            current_str = new_str

            yield diff_str

    # Callbacks for BatchManager
    def fetch_requests(self, max_num_sequences) -> list[tllm.InferenceRequest]:
        fetched = []
        for _ in range(max_num_sequences):
            if len(self.requests) == 0:
                break
            fetched.append(self.requests.pop())
        return fetched

    def handle_response(self, req_id: int, tensors: list[tllm.NamedTensor],
                        is_ok: bool, err_msg: str) -> None:
        self.results[req_id].sync_q.put(
            [{t.name: t.tensor
              for t in tensors}, is_ok] if not err_msg else err_msg)

    def get_stop_set(self) -> set[int]:
        return self.stop_set

    def handle_stats(self, stats: str):
        while self.stats.sync_q.full():
            self.stats.sync_q.get()

        self.stats.sync_q.put(stats)
