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

from tensorrt_llm._torch.attention.backends.fmha.flashinfer_trtllm_gen import (
    FlashInferTrtllmGenFmha,
    _get_generation_workspace_size,
)
from tensorrt_llm._torch.attention.backends.fmha.interface import FmhaPhase
from tensorrt_llm._torch.attention.backends.interface import (
    AttentionForwardArgs,
    AttentionInputType,
)
from tensorrt_llm.bindings import DataType
from tensorrt_llm.quantization.mode import QuantMode


@pytest.mark.parametrize("max_num_sequences", [None, 64])
@pytest.mark.parametrize("warmup_tokens", [1, 64])
def test_flashinfer_workspace_covers_sequence_capacity(
    monkeypatch: pytest.MonkeyPatch, max_num_sequences: int | None, warmup_tokens: int
) -> None:
    attn = FakeAttention()
    attn.num_heads = attn.num_kv_heads = 16
    attn.head_dim = 64
    attn.rope_dim = 0
    attn.quant_mode = 0
    fmha = FlashInferTrtllmGenFmha(attn)
    monkeypatch.setattr(fmha, "_get_multi_processor_count", lambda _: 148)
    metadata = SimpleNamespace(
        max_num_requests=16,
        max_num_sequences=max_num_sequences,
        max_context_length=1,
        num_ctx_tokens=0,
        is_cuda_graph=False,
    )
    workspace = torch.empty(0, dtype=torch.uint8)
    q = torch.empty((warmup_tokens, 16 * 64), dtype=torch.bfloat16)
    forward_args = AttentionForwardArgs(
        output=torch.empty_like(q), attention_input_type=AttentionInputType.generation_only
    )
    fmha.prepare_workspace(q, None, None, metadata, forward_args, workspace)

    num_sequences = max_num_sequences or metadata.max_num_requests
    required = _get_generation_workspace_size(
        torch.bfloat16, num_sequences, num_sequences, 16, 64, 16, 0
    )
    assert workspace.nbytes >= required

    # Capture must reuse the allocation reserved by a smaller warmup batch.
    q = torch.empty((num_sequences, 16 * 64), dtype=torch.bfloat16)
    forward_args.output = torch.empty_like(q)
    workspace_ptr = workspace.data_ptr()
    metadata.is_cuda_graph = True
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    fmha.prepare_workspace(q, None, None, metadata, forward_args, workspace)
    assert workspace.data_ptr() == workspace_ptr


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
    forward_args = AttentionForwardArgs(
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
    forward_args = AttentionForwardArgs(
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
