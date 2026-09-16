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
from fmha_test_utils import FakeAttention, make_fmha_forward_args

from tensorrt_llm._torch.attention.backends.fmha.flashinfer_trtllm_gen import (
    FlashInferTrtllmGenFmha,
)
from tensorrt_llm._torch.attention.backends.fmha.interface import FmhaPhase
from tensorrt_llm._torch.attention.backends.interface import AttentionInputType
from tensorrt_llm.bindings import DataType
from tensorrt_llm.quantization.mode import QuantMode


def test_flashinfer_fp8_mode_remains_implementation_local() -> None:
    attn = FakeAttention()
    attn.quant_mode = int(QuantMode.from_description(use_fp8_kv_cache=True))
    fmha = FlashInferTrtllmGenFmha(attn)
    output = torch.empty(1, dtype=torch.bfloat16)

    assert fmha._use_fp8_context_fmha(output, AttentionInputType.context_only)
    assert fmha._use_fp8_context_fmha(output, AttentionInputType.mixed)
    assert not fmha._use_fp8_context_fmha(output, AttentionInputType.generation_only)


@pytest.mark.parametrize(
    "num_contexts,num_generations,dtype,expect_fallback",
    [
        (1, 0, torch.bfloat16, True),
        (4, 0, torch.bfloat16, True),
        (5, 0, torch.bfloat16, False),
        (1, 1, torch.bfloat16, False),
        (2, 9, torch.bfloat16, False),
        (2, 19, torch.bfloat16, False),
        (2, 38, torch.bfloat16, False),
        (4, 1, torch.bfloat16, False),
        (0, 1, torch.bfloat16, False),
        (1, 0, torch.float16, False),
    ],
)
def test_small_context_fallback_preserves_mixed_batch_generation(
    num_contexts: int,
    num_generations: int,
    dtype: torch.dtype,
    expect_fallback: bool,
) -> None:
    attn = FakeAttention()
    attn.head_dim = 256
    attn.sparse_params = None
    attn.position_embedding_type = 0
    fmha = FlashInferTrtllmGenFmha(attn)
    metadata = SimpleNamespace(
        num_contexts=num_contexts,
        num_generations=num_generations,
        helix_position_offsets=None,
        num_sparse_topk=0,
        use_spec_decoding=False,
        kv_cache_block_offsets=object(),
        kv_cache_manager=None,
        tokens_per_block=32,
        is_cross=False,
        beam_width=1,
    )
    if num_contexts == 0:
        input_type = AttentionInputType.generation_only
        phases = (None, FmhaPhase.GENERATION)
    elif num_generations == 0:
        input_type = AttentionInputType.context_only
        phases = (None, FmhaPhase.CONTEXT)
    else:
        input_type = AttentionInputType.mixed
        phases = (None, FmhaPhase.CONTEXT, FmhaPhase.GENERATION)
    num_tokens = num_contexts + num_generations
    q = torch.empty((num_tokens, 3 * attn.head_dim), dtype=dtype)
    forward_args = make_fmha_forward_args(
        output=torch.empty((num_tokens, attn.head_dim), dtype=dtype),
        attention_input_type=input_type,
        is_fused_qkv=True,
    )

    for phase in phases:
        supported, reason = fmha._is_supported_with_reason(
            q, None, None, attn, metadata, forward_args, phase=phase
        )
        assert supported is not expect_fallback, reason
        if expect_fallback:
            assert "small-batch BF16 context attention" in reason


@pytest.mark.parametrize("kv_cache_dtype", [DataType.FP8, DataType.NVFP4])
@pytest.mark.parametrize(
    "dtype,sm_version",
    [(torch.bfloat16, 100), (torch.float16, 103)],
    ids=["small_bf16_batch", "sm103_fp16"],
)
def test_flashinfer_quantized_kv_context_avoids_fp16_bf16_fallback(
    monkeypatch: pytest.MonkeyPatch,
    kv_cache_dtype: DataType,
    dtype: torch.dtype,
    sm_version: int,
) -> None:
    monkeypatch.setattr(
        "tensorrt_llm._torch.attention.backends.fmha.flashinfer_trtllm_gen.get_sm_version",
        lambda: sm_version,
    )
    fmha = object.__new__(FlashInferTrtllmGenFmha)
    fmha.kv_factor = 2
    monkeypatch.setattr(fmha, "_get_total_num_blocks", lambda _: 0)
    attn = SimpleNamespace(
        is_mla_enable=False,
        sparse_params=None,
        position_embedding_type=0,
        head_dim=256,
        num_heads=32,
        num_kv_heads=2,
    )
    q_hidden_size = attn.num_heads * attn.head_dim
    qkv_hidden_size = q_hidden_size + 2 * attn.num_kv_heads * attn.head_dim
    q = torch.empty((1, qkv_hidden_size), dtype=dtype)
    metadata = SimpleNamespace(
        num_contexts=1,
        helix_position_offsets=None,
        num_sparse_topk=0,
        use_spec_decoding=False,
        is_spec_dec_tree=False,
        kv_cache_block_offsets=object(),
        kv_cache_manager=SimpleNamespace(dtype=kv_cache_dtype),
        is_cross=False,
        is_spec_decoding_enabled=False,
        tokens_per_block=64,
        beam_width=1,
    )
    forward_args = make_fmha_forward_args(
        output=torch.empty((1, q_hidden_size), dtype=dtype),
        attention_input_type=AttentionInputType.context_only,
        is_fused_qkv=True,
        update_kv_cache=True,
    )

    for phase in (None, FmhaPhase.CONTEXT):
        supported, reason = fmha._is_supported_with_reason(
            q,
            None,
            None,
            attn,
            metadata,
            forward_args,
            phase=phase,
        )

        assert supported, reason
        assert reason == ""
