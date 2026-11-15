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
"""``CrossAttention`` hands the model's quantization config to its backend.

The attention op sizes every KV cache token from the backend's quant mode,
and the cross pool is allocated from the model config. A backend that never
sees the config addresses an FP8 pool at 16 bits per element and writes past
it (#19861). These tests pin the handoff for both construction orders.
"""

import pytest
import torch

from tensorrt_llm._torch.attention.cross_attention import CrossAttention
from tensorrt_llm._torch.flashinfer_utils import IS_FLASHINFER_AVAILABLE
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo, QuantMode

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="attention backends allocate CUDA state at construction"
)

_FP8_KV = QuantConfig(kv_cache_quant_algo=QuantAlgo.FP8)


def _cross_attention(
    quant_config: QuantConfig, *, backend: str = "TRTLLM", skip_create_weights_in_init: bool = False
) -> CrossAttention:
    return CrossAttention(
        hidden_size=128,
        num_attention_heads=2,
        num_key_value_heads=2,
        layer_idx=0,
        dtype=torch.bfloat16,
        config=ModelConfig(
            quant_config=quant_config,
            attn_backend=backend,
            skip_create_weights_in_init=skip_create_weights_in_init,
        ),
    )


@pytest.mark.parametrize(
    "backend, backend_cls_name",
    [
        ("TRTLLM", "TrtllmAttention"),
        pytest.param(
            "FLASHINFER",
            "FlashInferAttention",
            marks=pytest.mark.skipif(
                not IS_FLASHINFER_AVAILABLE, reason="FlashInfer is not installed"
            ),
        ),
    ],
)
def test_backend_receives_fp8_kv_cache_mode(backend, backend_cls_name):
    module = _cross_attention(_FP8_KV, backend=backend)
    # Backend selection falls back to TRTLLM when FlashInfer is missing; make
    # sure the case under test is the one that ran.
    assert type(module.attn).__name__ == backend_cls_name
    assert module.attn.quant_config is not None
    assert module.attn.quant_config.layer_quant_mode.has_fp8_kv_cache()
    assert module.attn.has_fp8_kv_cache


def test_trtllm_op_quant_mode_carries_fp8_kv_cache_bit():
    # ``quant_mode`` is the integer the C++ op gets; it decides the bytes per
    # cache element, so it must agree with the pool the model config produces.
    module = _cross_attention(_FP8_KV)
    assert QuantMode(module.attn.quant_mode).has_fp8_kv_cache()


def test_deferred_weight_creation_updates_backend():
    module = _cross_attention(_FP8_KV, skip_create_weights_in_init=True)
    module.create_weights()
    assert module.attn.has_fp8_kv_cache
    assert QuantMode(module.attn.quant_mode).has_fp8_kv_cache()


def test_unquantized_cache_leaves_backend_unquantized():
    module = _cross_attention(QuantConfig())
    assert not module.attn.has_fp8_kv_cache
    assert not QuantMode(module.attn.quant_mode).has_fp8_kv_cache()
