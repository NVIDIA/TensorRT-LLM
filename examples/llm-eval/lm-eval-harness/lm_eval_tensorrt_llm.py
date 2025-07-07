# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import gc
import json
import logging
import os
import signal
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
import transformers
from lm_eval.__main__ import cli_evaluate
from lm_eval.api.model import TemplateLM
from lm_eval.api.registry import register_model
from packaging.version import parse
from tqdm import tqdm

import tensorrt_llm
from tensorrt_llm import LLM as TORCH_LLM
from tensorrt_llm._tensorrt_engine import LLM as TRT_LLM
from tensorrt_llm._torch.pyexecutor.config import PyTorchConfig
from tensorrt_llm.bindings.executor import DecodingConfig
from tensorrt_llm.llmapi import KvCacheConfig as TRT_KvCacheConfig
from tensorrt_llm.llmapi import RequestOutput, SamplingParams

logger = logging.getLogger(__name__)


@register_model("trt-llm")
class TRTLLMEvalBase(TemplateLM):

    def __init__(
        self,
        model: str,
        tokenizer: Optional[str] = None,
        tp: int = 0,  # tensor_parallel_size
        max_gen_toks: int = 256,
        chunk_size: int = 200,
        max_tokens_kv_cache: Optional[int] = None,
        free_gpu_memory_fraction: float = 0.9,
        trust_remote_code: bool = False,
        use_cuda_graph: bool = True,
        backend: str = 'trt',
        max_context_length: Optional[int] = None,
        moe_expert_parallel_size: Optional[int] = None,
        moe_backend: Optional[str] = "TRTLLM",
        enable_chunked_prefill: bool = False,
        max_num_tokens: Optional[int] = None,
        **kwargs,
    ):
        # initialize TemplateLM, copied from TemplateAPI
        super().__init__()
        assert isinstance(model, str)
        assert parse(tensorrt_llm.__version__) >= parse("0.15.0")

        self.max_gen_toks = max_gen_toks
        self.chunk_size = chunk_size
        self.backend = backend
        self.max_context_length = max_context_length
        self.moe_expert_parallel_size = moe_expert_parallel_size
        self.moe_backend = moe_backend
        trt_kv_cache_config = TRT_KvCacheConfig(enable_block_reuse=False)
        trt_kv_cache_config.free_gpu_memory_fraction = free_gpu_memory_fraction
        if max_tokens_kv_cache is not None:
            trt_kv_cache_config.max_tokens = max_tokens_kv_cache

        if tokenizer is None:
            # Assume the tokenizer is stored in the model_dir if not specified.
            tokenizer = model
        logger.info(f"Tokenizer: {tokenizer}")
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            tokenizer, trust_remote_code=trust_remote_code)

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        if self.backend == 'torch':
            kwargs.pop('batch_size')
            if tp < 1:
                tp = torch.cuda.device_count()

            pytorch_config_params = {
                'cuda_graph_config': {} if use_cuda_graph else None,
                "print_iter_log": False,
            }
            if hasattr(PyTorchConfig, "moe_backend"):
                pytorch_config_params["moe_backend"] = self.moe_backend
                print(f"Info: moe_backend is set to {self.moe_backend}")

            # stop words not currently supported by torch backend
            self.use_stop_words = False

            self.llm = TORCH_LLM(
                model=model,
                tensor_parallel_size=tp,
                trust_remote_code=trust_remote_code,
                enable_chunked_prefill=enable_chunked_prefill,
                max_num_tokens=max_num_tokens,
                **pytorch_config_params,
                tokenizer=self.tokenizer,
                kv_cache_config=trt_kv_cache_config,
                moe_expert_parallel_size=self.moe_expert_parallel_size,
                **kwargs)
            logger.info("Loaded TRT-LLM Torch engine")
        else:
            with open(Path(model) / "config.json", "r") as engine_config_file:
                engine_config = json.load(engine_config_file)
                build_config = engine_config["build_config"]
                world_size = (engine_config.get("pretrained_config", {}).get(
                    "mapping", {}).get("world_size", 1))
                if max_tokens_kv_cache is None:
                    max_tokens_kv_cache = build_config[
                        "max_seq_len"] * build_config["max_batch_size"]
                self.gather_context_logits = build_config.get(
                    "gather_context_logits", False)

            medusa_choices = kwargs[
                'medusa_choices'] if 'medusa_choices' in kwargs else None
            kwargs = {}
            if medusa_choices is not None:
                decoding_config = DecodingConfig()
                decoding_config.medusa_choices = medusa_choices
                kwargs["decoding_config"] = decoding_config
                assert world_size == 1, "decoding_config does not support multi TP in HLAPI."

            self.llm = TRT_LLM(model=model,
                               tokenizer=self.tokenizer,
                               kv_cache_config=trt_kv_cache_config,
                               **kwargs)
            self.max_length = build_config['max_seq_len'] - 1
            logger.info("Loaded TRT-LLM engine")

    @property
    def eot_token_id(self) -> int:
        return self.llm.tokenizer.eos_token_id

    def tok_encode(self, string, add_special_tokens=False, **kwargs):
        return self.llm.tokenizer.encode(string,
                                         add_special_tokens=add_special_tokens,
                                         **kwargs)

    def _loglikelihood_tokens(
            self,
            requests: List[Any],
            disable_tqdm: bool = False) -> List[Tuple[float, bool]]:
        """Compute the log likelihood of the continuation given the context."""
        if self.backend == 'torch':
            raise NotImplementedError(
                'Torch backend does not return context logits yet')

        num_r = len(requests)
        desc = "Processing loglikelihood requests"
        sampling_params = SamplingParams(max_tokens=1,
                                         return_context_logits=True)

        # process requests
        futures: Dict[int, RequestOutput] = {}
        results = []
        for i, request in tqdm(enumerate(requests),
                               desc=desc,
                               total=num_r,
                               disable=disable_tqdm):
            # asynchronously submit a chunk of requests ahead of time...
            if i % self.chunk_size == 0:
                for j in range(i, min(i + self.chunk_size, num_r)):
                    prompt_ids = requests[j][1] + requests[j][2]
                    futures[j] = self.llm.generate_async(
                        prompt_ids, sampling_params)

            # process the output of the request i
            r_out: RequestOutput = futures.pop(i).result()

            # check continuation portion of the prompt
            # NOTE: context_logits are offset by 1 since they predict future token
            ctxlen = len(request[1])
            token_ids_cont = request[2]
            logits_cont = r_out.context_logits[ctxlen - 1:-1]  # [sl, vocab]
            logprobs_cont = F.log_softmax(logits_cont, dim=-1)  # [sl, vocab]
            top_tokens_cont = logprobs_cont.argmax(dim=-1).tolist()  # [sl]

            # compute logprob and check for greedy
            logprob_sum = sum(logprobs_cont[list(range(len(logprobs_cont))),
                                            token_ids_cont]).item()
            is_greedy = top_tokens_cont == token_ids_cont

            results.append((logprob_sum, is_greedy))

            # clear response
            del r_out

        return results

    def loglikelihood_rolling(self, requests, disable_tqdm: bool = False):
        raise NotImplementedError

    def generate_until(self,
                       requests: List[Any],
                       disable_tqdm: bool = False) -> List[str]:
        # some book-keeping and parameters...
        num_r = len(requests)
        desc = "Processing generate requests"

        if self.max_context_length is not None:
            """
            Create updated_requests to contain qualified requests with the context length <= max_context_length.
            Unqualified requests cannot simply be dropped as lm-eval library requires the number of requests to be the same.

            Note: The final score will drop if disqualified requests exist.
            """
            request_idx_to_replace = []
            qualified_requests = []
            updated_requests = []
            for i, request in enumerate(requests):
                context, gen_kwargs = request.args
                if len(self.tok_encode(context)) > self.max_context_length:
                    request_idx_to_replace.append(i)
                else:
                    qualified_requests.append(request)

            assert len(
                qualified_requests
            ) > 1, "No requests with context length <= max_context_length. Cannot run the evaluation."
            if len(request_idx_to_replace) > 0:
                print(
                    f"Warning: {len(request_idx_to_replace)} requests with context length > max_context_length will be replaced. The final score will drop."
                )

            for i, request in enumerate(requests):
                if i in request_idx_to_replace:
                    # Replace the requests with context length > max_context_length with the qualified requests
                    updated_requests.append(
                        qualified_requests[i % len(qualified_requests)])
                else:
                    updated_requests.append(request)
            assert len(
                updated_requests
            ) == num_r, "Number of updated requests does not match the number of requests."
            requests = updated_requests

        def _get_sp(gen_kwargs):
            k_mapping = {
                "temperature": "temperature",
                "top_p": "top_p",
                "max_gen_toks": "max_tokens",
                "until": "stop",
            }
            kwargs_mapped = {
                k_sp: gen_kwargs[k_gen]
                for k_gen, k_sp in k_mapping.items() if k_gen in gen_kwargs
            }
            if "max_tokens" not in kwargs_mapped:
                kwargs_mapped["max_tokens"] = self.max_gen_toks
            return SamplingParams(**kwargs_mapped)

        # process requests
        futures: Dict[int, RequestOutput] = {}
        future_stop_words: Dict[int, RequestOutput] = {}
        results = []
        for i, _ in tqdm(enumerate(requests),
                         desc=desc,
                         total=num_r,
                         disable=disable_tqdm):
            # asynchronously submit a chunk of requests ahead of time...
            if i % self.chunk_size == 0:
                for j in range(i, min(i + self.chunk_size, num_r)):
                    context, gen_kwargs = requests[j].args
                    prompt_ids = self.tok_encode(context)
                    if self.max_context_length is not None:
                        assert len(
                            prompt_ids
                        ) <= self.max_context_length, f"Prompt length > {self.max_context_length}, {len(prompt_ids)}, should be filtered out."
                    kwargs_mapped = _get_sp(gen_kwargs)
                    futures[j] = self.llm.generate_async(
                        prompt_ids, kwargs_mapped)
                    del kwargs_mapped
                    future_stop_words[j] = gen_kwargs["until"]

            # process the output of the request i
            r_out: RequestOutput = futures.pop(i).result()
            stop_words = future_stop_words.pop(i)
            txt = r_out.outputs[0].text
            if stop_words:
                for word in stop_words:
                    word_index = txt.find(word)
                    if word_index >= 0:
                        txt = txt[:word_index]
            results.append(txt)

        return results


if __name__ == "__main__":
    cli_evaluate()
    # Force clean up the LLM instance and void hanging.
    gc.collect()

    # Force terminate in case gc.collect() is not enough.
    def _terminate():
        time.sleep(10)
        os.kill(os.getpid(), signal.SIGTERM)

    termination_thread = threading.Thread(target=_terminate, daemon=True)
    termination_thread.start()
