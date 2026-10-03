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
"""End-to-end test for LlmArgs.lm_head_dtype.

With the default LM head, logits are rounded to the model dtype before the float32
upcast, so every returned logit is exactly representable in the model dtype. With
lm_head_dtype="float32" the LM head GEMM writes its float32 accumulator, so the
returned logits are not rounded, and greedy decoding still follows them.
"""

import json
import os

import pytest
import torch
from utils.llm_data import llm_models_root

from tensorrt_llm import LLM
from tensorrt_llm.llmapi import KvCacheConfig
from tensorrt_llm.sampling_params import SamplingParams

_MODEL = os.path.join("llama-models-v2", "TinyLlama-1.1B-Chat-v1.0")


def _model_dir() -> str:
    root = llm_models_root()
    path = os.path.join(root, _MODEL) if root else None
    if not path or not os.path.isdir(path):
        pytest.skip(f"{_MODEL} not available under llm_models_root()")
    return path


def _rounded_to(logits: torch.Tensor, dtype: torch.dtype) -> bool:
    return torch.equal(logits, logits.to(dtype).float())


@pytest.mark.parametrize("lm_head_dtype", ["auto", "float32"])
def test_lm_head_dtype_returned_logits(lm_head_dtype: str):
    model_dir = _model_dir()
    with open(os.path.join(model_dir, "config.json")) as f:
        model_dtype = getattr(torch, json.load(f)["torch_dtype"])

    sampling_params = SamplingParams(
        max_tokens=8,
        temperature=0.0,
        return_context_logits=True,
        return_generation_logits=True,
    )
    with LLM(
        model=model_dir,
        lm_head_dtype=lm_head_dtype,
        max_batch_size=8,
        # Context logits are not recomputed for reused KV cache blocks.
        kv_cache_config=KvCacheConfig(max_tokens=10000, enable_block_reuse=False),
    ) as llm:
        output = llm.generate(["The capital of France is"], sampling_params)[0]

    completion = output.outputs[0]
    context_logits = output.context_logits
    generation_logits = completion.generation_logits
    assert context_logits.dtype == torch.float32
    assert generation_logits.dtype == torch.float32
    assert context_logits.shape[0] == len(output.prompt_token_ids)
    assert generation_logits.shape[0] == len(completion.token_ids)

    rounded = lm_head_dtype == "auto"
    assert _rounded_to(context_logits, model_dtype) == rounded
    assert _rounded_to(generation_logits, model_dtype) == rounded
    assert torch.argmax(generation_logits, dim=-1).tolist() == completion.token_ids
