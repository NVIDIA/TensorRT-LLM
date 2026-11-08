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

from types import SimpleNamespace

import pytest
import torch
from fmha_test_utils import FakeAttention

from tensorrt_llm._torch.attention.backends.fmha.interface import FmhaPhase
from tensorrt_llm._torch.attention.backends.fmha.triton_custom_mask import TritonCustomMaskFmha
from tensorrt_llm._torch.attention.backends.interface import (
    AttentionForwardArgs,
    CustomAttentionMask,
)
from tensorrt_llm.quantization.mode import QuantMode


def test_triton_custom_mask_rejects_whole_request_probe() -> None:
    fmha = object.__new__(TritonCustomMaskFmha)

    assert not fmha.is_supported(
        torch.empty((1, 4)),
        None,
        None,
        SimpleNamespace(),
        AttentionForwardArgs(),
    )


@pytest.mark.parametrize("output_dtype", [torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize(
    "quant_mode,supported",
    [
        (QuantMode(0), True),
        (QuantMode.from_description(use_fp8_kv_cache=True), True),
        (QuantMode.from_description(use_int8_kv_cache=True), False),
        (QuantMode(0).set_fp4_kv_cache(), False),
    ],
)
def test_custom_mask_quantized_context_support(
    output_dtype: torch.dtype, quant_mode: QuantMode, supported: bool
) -> None:
    attn = FakeAttention()
    attn.quant_mode = int(quant_mode)
    fmha = TritonCustomMaskFmha(attn)
    metadata = SimpleNamespace(
        num_contexts=1,
        is_cross=False,
        kv_cache_block_offsets=object(),
        kv_layout="HND",
        kv_cache_manager=None,
    )
    forward_args = AttentionForwardArgs(
        output=torch.empty((1, attn.head_dim), dtype=output_dtype),
        attention_mask=CustomAttentionMask.CUSTOM,
        attention_mask_data=torch.ones((1, 1), dtype=torch.bool),
        is_fused_qkv=True,
    )
    assert (
        fmha.is_supported(
            torch.empty((1, 3 * attn.head_dim), dtype=torch.bfloat16),
            None,
            None,
            metadata,
            forward_args,
            phase=FmhaPhase.CONTEXT,
        )
        is supported
    )
