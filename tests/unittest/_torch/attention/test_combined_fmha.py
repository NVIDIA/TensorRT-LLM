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
from fmha_test_utils import FakeAttention, FakePhasedFmha

from tensorrt_llm._torch.attention.backends.fmha.combined import CombinedFmha
from tensorrt_llm._torch.attention.backends.fmha.flashinfer_trtllm_gen import (
    FlashInferTrtllmGenFmha,
)
from tensorrt_llm._torch.attention.backends.fmha.interface import FmhaPhase
from tensorrt_llm._torch.attention.backends.fmha.triton_custom_mask import TritonCustomMaskFmha
from tensorrt_llm._torch.attention.backends.interface import (
    AttentionForwardArgs,
    AttentionInputType,
)
from tensorrt_llm.bindings import DataType
from tensorrt_llm.quantization.mode import QuantMode


def test_combined_fmha_delegates_phases_and_prepares_max_workspace() -> None:
    events: list[tuple] = []
    attn = FakeAttention()
    context_fmha = FakePhasedFmha(
        attn,
        {FmhaPhase.CONTEXT},
        "context",
        events,
        workspace_size=8,
    )
    generation_fmha = FakePhasedFmha(
        attn,
        {FmhaPhase.GENERATION},
        "generation",
        events,
        workspace_size=4,
    )
    combined_fmha = CombinedFmha(attn)
    combined_fmha.set_fmha_impls(context_fmha, generation_fmha)
    metadata = SimpleNamespace(
        kv_cache_block_offsets=object(),
        effective_workspace=torch.empty(0, dtype=torch.uint8),
        num_contexts=1,
        num_ctx_tokens=2,
        num_generations=1,
        kv_lens_cuda_runtime=torch.tensor([2, 5], dtype=torch.int32),
        kv_lens_runtime=torch.tensor([2, 5], dtype=torch.int32),
        prompt_lens_cuda_runtime=torch.tensor([2, 1], dtype=torch.int32),
        prompt_lens_cpu_runtime=torch.tensor([2, 1], dtype=torch.int32),
        beam_width=1,
        cache_indirection=None,
        tokens_per_block=32,
        kv_cache_manager=None,
        is_cross=False,
        is_spec_decoding_enabled=False,
    )
    forward_args = AttentionForwardArgs(
        output=torch.empty((3, 4)),
        attention_input_type=AttentionInputType.mixed,
        attention_window_size=8,
    )

    combined_fmha.forward(torch.empty((3, 4)), None, None, metadata, forward_args)

    assert events == [
        ("prepare", "context"),
        ("prepare", "generation"),
        ("run", "context", FmhaPhase.CONTEXT, 2),
        ("run", "generation", FmhaPhase.GENERATION, 1),
    ]
    assert metadata.effective_workspace.numel() == 8


def test_combined_fmha_uses_flattened_v2_page_bound() -> None:
    attn = FakeAttention()
    combined_fmha = CombinedFmha(attn)
    kv_cache_manager = SimpleNamespace(
        impl=SimpleNamespace(get_page_index_upper_bound=lambda: 23),
        blocks_in_primary_pool=23,
        num_local_layers=4,
    )

    assert (
        combined_fmha._get_total_num_blocks(SimpleNamespace(kv_cache_manager=kv_cache_manager))
        == 23
    )


def test_flashinfer_fp8_mode_remains_implementation_local() -> None:
    attn = FakeAttention()
    attn.quant_mode = int(QuantMode.from_description(use_fp8_kv_cache=True))
    fmha = FlashInferTrtllmGenFmha(attn)
    output = torch.empty(1, dtype=torch.bfloat16)

    assert fmha._use_fp8_context_fmha(output, AttentionInputType.context_only)
    assert fmha._use_fp8_context_fmha(output, AttentionInputType.mixed)
    assert not fmha._use_fp8_context_fmha(output, AttentionInputType.generation_only)


def test_triton_custom_mask_rejects_whole_request_probe() -> None:
    fmha = object.__new__(TritonCustomMaskFmha)

    assert not fmha.is_supported(
        torch.empty((1, 4)),
        None,
        None,
        SimpleNamespace(),
        AttentionForwardArgs(),
    )


@pytest.mark.parametrize("is_fused_qkv", [True, False], ids=["fused_qkv", "q_only"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("sm_version", [100, 103])
@pytest.mark.parametrize("tokens_per_block", [32, 64])
@pytest.mark.parametrize("num_contexts", [1, 4, 5])
@pytest.mark.parametrize("head_dim", [64, 256, 512])
def test_flashinfer_context_fallback_scope(
    monkeypatch: pytest.MonkeyPatch,
    dtype: torch.dtype,
    sm_version: int,
    tokens_per_block: int,
    num_contexts: int,
    head_dim: int,
    is_fused_qkv: bool,
) -> None:
    monkeypatch.setattr(
        "tensorrt_llm._torch.attention.backends.fmha.flashinfer_trtllm_gen.get_sm_version",
        lambda: sm_version,
    )
    fmha = object.__new__(FlashInferTrtllmGenFmha)
    fmha.kv_factor = 2
    attn = FakeAttention()
    attn.sparse_params = None
    attn.position_embedding_type = 0
    attn.head_dim = head_dim
    q_hidden_size = attn.num_heads * attn.head_dim
    # Both layouts the trtllm-gen self-attention path accepts, wired the way
    # TrtllmAttention.forward wires them: fused QKV carries K/V inline and writes
    # the cache, while a Q-only input only reads an already-populated cache.
    input_hidden_size = q_hidden_size
    if is_fused_qkv:
        input_hidden_size += 2 * attn.num_kv_heads * attn.head_dim
    q = torch.empty((num_contexts, input_hidden_size), dtype=dtype)
    kv_cache_dtype = DataType.BF16 if dtype == torch.bfloat16 else DataType.HALF
    metadata = SimpleNamespace(
        num_contexts=num_contexts,
        helix_position_offsets=None,
        num_sparse_topk=0,
        use_spec_decoding=False,
        is_spec_dec_tree=False,
        kv_cache_block_offsets=object(),
        kv_cache_manager=SimpleNamespace(dtype=kv_cache_dtype),
        is_cross=False,
        is_spec_decoding_enabled=False,
        tokens_per_block=tokens_per_block,
        beam_width=1,
    )
    forward_args = AttentionForwardArgs(
        output=torch.empty((num_contexts, q_hidden_size), dtype=dtype),
        attention_input_type=AttentionInputType.mixed,
        is_fused_qkv=is_fused_qkv,
        update_kv_cache=is_fused_qkv,
    )

    small_bf16_fallback = (
        dtype == torch.bfloat16 and num_contexts <= 4 and is_fused_qkv and head_dim != 512
    )
    expected_fallback = small_bf16_fallback or sm_version == 103
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
        if expected_fallback:
            assert not supported
            assert "fallback FMHA" in reason
        else:
            assert supported, reason
            assert reason == ""

    generation_supported, generation_reason = fmha._is_supported_with_reason(
        q,
        None,
        None,
        attn,
        metadata,
        forward_args,
        phase=FmhaPhase.GENERATION,
    )
    assert generation_supported, generation_reason
    assert generation_reason == ""


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
