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
"""Engine options and GSM8K protocol shared by the HY4-Flash sm_103 accuracy tests."""

import math
import os

import pytest

from tensorrt_llm._torch.modeling_v2 import MODELING_V2_VALIDATE_ENV
from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.llmapi import CudaGraphConfig, DFlashDecodingConfig, KvCacheConfig, SamplingParams
from tensorrt_llm.quantization import QuantAlgo

from ....accuracy_core import GSM8K

skip_not_sm103 = pytest.mark.skipif(
    get_sm_version() != 103, reason="target is certified on sm_103 only"
)

# The layouts routing admits.
TP4 = dict(
    tensor_parallel_size=4,
    moe_tensor_parallel_size=4,
    moe_expert_parallel_size=1,
    enable_attention_dp=False,
)
TEP4 = dict(tensor_parallel_size=4, moe_expert_parallel_size=4, enable_attention_dp=False)
DEP4 = dict(tensor_parallel_size=4, moe_expert_parallel_size=4, enable_attention_dp=True)
TEP8 = dict(tensor_parallel_size=8, moe_expert_parallel_size=8, enable_attention_dp=False)

MAX_SEQ_LEN = 5120
MAX_BATCH_SIZE = 64
KV_CACHE_FRACTION = 0.7

VALIDATE = {MODELING_V2_VALIDATE_ENV: "1"}

GSM8K_EVALUATOR_KWARGS = dict(
    apply_chat_template=True,
    fewshot_as_multiturn=True,
    chat_template_kwargs=dict(reasoning_effort="no_think"),
)


def engine(**overrides) -> dict:
    """LLM arguments with fresh config objects: an engine resolves fields of the configs it is given."""
    options = dict(
        max_seq_len=MAX_SEQ_LEN,
        max_batch_size=MAX_BATCH_SIZE,
        enable_chunked_prefill=False,
        kv_cache_config=KvCacheConfig(
            free_gpu_memory_fraction=KV_CACHE_FRACTION, enable_block_reuse=False
        ),
        cuda_graph_config=CudaGraphConfig(enable_padding=True, max_batch_size=MAX_BATCH_SIZE),
    )
    options.update(overrides)
    return options


def dfly2() -> DFlashDecodingConfig:
    return DFlashDecodingConfig(max_draft_len=5, mask_token_id=120009)


def sampling() -> SamplingParams:
    return SamplingParams(max_tokens=1024, truncate_prompt_tokens=4096, temperature=0.0)


def gsm8k_samples() -> int:
    """Sample count ``AccuracyTask.evaluate`` uses (one under ``INTEGRATION_TEST``)."""
    return 1 if os.getenv("INTEGRATION_TEST", "0") == "1" else GSM8K.NUM_SAMPLES


def assert_target_engine(llm) -> None:
    """Assert the modeling_v2 route and set the quant key the reference rows are stored under.

    The checkpoint keeps its quantization in ``text_config.quantization_config``, which the LLM API
    front end does not read, so ``llm.args`` leaves ``quant_algo`` unset.
    """
    assert llm.args.modeling_v2 == "require"
    assert llm.args.quant_config.quant_algo in (None, QuantAlgo.FP8_BLOCK_SCALES)
    llm.args.quant_config.quant_algo = QuantAlgo.FP8_BLOCK_SCALES


def relative_band(score: float, samples: int, k: float = 2.0) -> float:
    """``k`` standard errors of the difference of two binomial measurements of ``score``."""
    p = min(max(score / 100.0, 0.5 / samples), 1.0 - 0.5 / samples)
    return k * math.sqrt(2.0) * 100.0 * math.sqrt(p * (1.0 - p) / samples)
