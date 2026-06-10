# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Short-token-boundary smoke test for the DeepSeek-V4-Pro aggregate path.

Drives the same deployment config as the Pro GSM8K gate (TP=8, EP=8, attention
DP, TRTLLM MoE, FP8 KV cache, MTP, padded CUDA graphs) over deliberately short /
empty-KV token-id prompts. This deterministically exercises the boundary
batches (e.g. bs=32 all-generation -> empty context lengths) that the
aggregated GSM8K run only hits non-deterministically, surfacing sparse-MLA /
paged-MQA-logits edge cases at engine warmup and short-sequence generation.
"""

import json
import os

import pytest
from defs.conftest import llm_models_root, skip_pre_blackwell

from tensorrt_llm import LLM
from tensorrt_llm.llmapi import (
    CudaGraphConfig,
    KvCacheConfig,
    MoeConfig,
    MTPDecodingConfig,
    SamplingParams,
)


def _deepseekv4_pro_agg_llm_kwargs(**overrides):
    kwargs = dict(
        tensor_parallel_size=8,
        moe_expert_parallel_size=8,
        moe_config=MoeConfig(backend="TRTLLM"),
        enable_attention_dp=True,
        max_seq_len=4096,
        max_batch_size=16,
        max_num_tokens=8192,
        custom_tokenizer="deepseek_v4",
        # Cap the KV cache so the padded CUDA-graph capture (batch sizes up to
        # 1024) and other post-KV resources have headroom; without this the
        # default fraction (~0.9) leaves too little and engine init OOMs on
        # large-memory GPUs. This test needs very little KV.
        kv_cache_config=KvCacheConfig(
            enable_block_reuse=False,
            tokens_per_block=128,
            dtype="fp8",
            host_cache_size=0,
            free_gpu_memory_fraction=0.6,
        ),
        cuda_graph_config=CudaGraphConfig(
            batch_sizes=[1, 2, 4, 8, 16, 32, 64, 128, 192, 256, 320, 384, 448, 512, 768, 1024],
            enable_padding=True,
        ),
        speculative_config=MTPDecodingConfig(max_draft_len=1),
    )
    kwargs.update(overrides)
    return kwargs


def _get_llm_vocab_size(llm):
    input_processor = getattr(llm, "input_processor", None)
    if input_processor is not None and hasattr(input_processor, "get_vocab_size"):
        vocab_size = input_processor.get_vocab_size()
        if vocab_size is not None:
            return int(vocab_size)

    tokenizer = getattr(llm, "tokenizer", None)
    for candidate in (tokenizer, getattr(tokenizer, "tokenizer", None)):
        if candidate is None:
            continue
        # The DeepSeek-V4 tokenizer raises NotImplementedError (not
        # AttributeError) from vocab_size/__len__/get_vocab, which a plain
        # getattr(..., None) would not swallow, so probe each defensively.
        for getter in (
            lambda c: getattr(c, "vocab_size", None),
            lambda c: len(c.get_vocab()),
            lambda c: len(c),
        ):
            try:
                vocab_size = getter(candidate)
            except (AttributeError, TypeError, NotImplementedError):
                vocab_size = None
            if vocab_size:
                return int(vocab_size)
        # Final fallback: read vocab_size straight from the model config.
        name_or_path = getattr(candidate, "name_or_path", None)
        if name_or_path and os.path.isdir(str(name_or_path)):
            try:
                with open(os.path.join(str(name_or_path), "config.json")) as f:
                    config_vocab_size = json.load(f).get("vocab_size")
            except (OSError, ValueError):
                config_vocab_size = None
            if config_vocab_size:
                return int(config_vocab_size)

    pytest.fail("Cannot determine tokenizer vocab size for token-id smoke test")


def _make_token_id_prompt(length, vocab_size, salt):
    first_regular_token = 2 if vocab_size > 4 else 0
    regular_vocab_size = vocab_size - first_regular_token
    return [first_regular_token + ((salt + i) % regular_vocab_size) for i in range(length)]


def _run_token_id_smoke_wave(llm, prompt_lengths, max_tokens, vocab_size):
    inputs = [
        {"prompt_token_ids": _make_token_id_prompt(length, vocab_size, salt=idx * 257)}
        for idx, length in enumerate(prompt_lengths)
    ]
    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        min_tokens=max_tokens,
        temperature=0,
        ignore_eos=True,
        detokenize=False,
        add_special_tokens=False,
    )
    outputs = llm.generate(inputs, sampling_params=sampling_params, use_tqdm=False)
    for output in outputs:
        generated_tokens = output.outputs[0].token_ids
        assert len(generated_tokens) == max_tokens, (
            f"Expected {max_tokens} generated tokens, got {len(generated_tokens)}"
        )


@pytest.mark.timeout(14400)
@pytest.mark.skip_less_device(8)
@pytest.mark.skip_less_device_memory(140000)
@pytest.mark.skip_less_mpi_world_size(8)
@skip_pre_blackwell
def test_short_token_boundary_smoke():
    model_path = f"{llm_models_root()}/DeepSeek-V4-Pro"
    with LLM(model_path, **_deepseekv4_pro_agg_llm_kwargs(max_batch_size=32)) as llm:
        vocab_size = _get_llm_vocab_size(llm)
        _run_token_id_smoke_wave(llm, [1] * 32, 1, vocab_size)
        _run_token_id_smoke_wave(llm, [1] * 31, 2, vocab_size)
        _run_token_id_smoke_wave(
            llm, [1, 2, 3, 4, 63, 64, 65, 127, 128, 129, 511, 512, 513, 2048], 8, vocab_size
        )
