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
from fmha_test_utils import FakeAttention, metadata_stub

from tensorrt_llm._torch.attention.backends.fmha.fallback import FallbackFmha
from tensorrt_llm._torch.attention.backends.fmha.flashinfer_trtllm_gen import (
    FlashInferTrtllmGenFmha,
)
from tensorrt_llm._torch.attention.backends.fmha.interface import FmhaPhase
from tensorrt_llm._torch.attention.backends.interface import (
    AttentionForwardArgs,
    AttentionInputType,
    CustomAttentionMask,
    PredefinedAttentionMask,
)
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm.bindings import DataType
from tensorrt_llm.quantization.mode import QuantMode

_FLASHINFER_TRTLLM_GEN_MODULE = "tensorrt_llm._torch.attention.backends.fmha.flashinfer_trtllm_gen"


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
@pytest.mark.cpu_only
@pytest.mark.parametrize("page_size,paged_context", [(32, False), (128, True)])
def test_small_context_fallback_preserves_mixed_batch_generation(
    monkeypatch: pytest.MonkeyPatch,
    num_contexts: int,
    num_generations: int,
    dtype: torch.dtype,
    expect_fallback: bool,
    page_size: int,
    paged_context: bool,
) -> None:
    monkeypatch.setattr(f"{_FLASHINFER_TRTLLM_GEN_MODULE}.get_sm_version", lambda: 100)
    attn = FakeAttention()
    attn.head_dim = 256
    attn.sparse_params = None
    attn.position_embedding_type = 0
    fmha = FlashInferTrtllmGenFmha(attn)
    manager = object.__new__(KVCacheManagerV2)
    manager.dtype = None
    manager.trtllm_gen_extra_tokens_per_block = frozenset({128})
    manager.impl = SimpleNamespace(get_page_index_upper_bound=lambda *args: 32)
    metadata = metadata_stub(
        num_contexts=num_contexts,
        num_generations=num_generations,
        helix_position_offsets=None,
        num_sparse_topk=0,
        use_spec_decoding=False,
        kv_cache_block_offsets=object(),
        kv_cache_manager=manager,
        tokens_per_block=page_size,
        use_paged_context_fmha=paged_context,
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

    expect_fallback = expect_fallback and not (paged_context and page_size == 128)
    for phase in phases:
        supported, reason = fmha._is_supported_with_reason(
            q, None, None, attn, metadata, forward_args, phase=phase
        )
        assert supported is not expect_fallback, reason
        if expect_fallback:
            assert "small-batch BF16 context attention" in reason

    # Reject P128 at the fallback boundary even if TRTLLM-Gen rejects for
    # another reason; generation-only and ordinary context remain usable.
    fallback = FallbackFmha(attn)
    for phase in phases:
        if paged_context and num_contexts and phase != FmhaPhase.GENERATION:
            with pytest.raises(RuntimeError, match="128-token KV pages"):
                fallback.is_supported(q, None, None, metadata, forward_args, phase=phase)
        else:
            assert fallback.is_supported(q, None, None, metadata, forward_args, phase=phase)


@pytest.mark.parametrize("kv_cache_dtype", [DataType.FP8, DataType.NVFP4])
@pytest.mark.parametrize(
    "attention_mask", [PredefinedAttentionMask.CAUSAL, CustomAttentionMask.CUSTOM]
)
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
    attention_mask: PredefinedAttentionMask | CustomAttentionMask,
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
    metadata = metadata_stub(
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
        attention_mask=attention_mask,
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

        if attention_mask == CustomAttentionMask.CUSTOM:
            assert not supported
            assert "unsupported attention mask" in reason
        else:
            assert supported, reason
            assert reason == ""


@pytest.mark.parametrize(
    "sm_version,expected",
    [(90, False), (100, True), (103, True), (107, True), (120, False)],
)
def test_flashinfer_availability_by_sm(
    monkeypatch: pytest.MonkeyPatch,
    sm_version: int,
    expected: bool,
) -> None:
    monkeypatch.setattr(f"{_FLASHINFER_TRTLLM_GEN_MODULE}.IS_FLASHINFER_AVAILABLE", True)
    monkeypatch.setattr(
        FlashInferTrtllmGenFmha, "_missing_fused_nanobind_ops", staticmethod(lambda: [])
    )
    monkeypatch.setattr(f"{_FLASHINFER_TRTLLM_GEN_MODULE}.get_sm_version", lambda: sm_version)
    attn = FakeAttention()
    attn.sparse_params = None

    assert FlashInferTrtllmGenFmha._is_available(attn) is expected


@pytest.mark.parametrize(
    "input_type",
    [AttentionInputType.context_only, AttentionInputType.generation_only],
    ids=["context", "generation"],
)
@pytest.mark.parametrize(
    "sm_version,tokens_per_block,extra_tokens_per_block,expected",
    [
        (107, 128, frozenset({128}), True),
        (107, 32, frozenset({128}), False),
        (107, 32, frozenset(), False),
        (103, 32, frozenset(), True),
        (103, 128, frozenset({128}), True),
    ],
    ids=[
        "sm107_extra_page_size",
        "sm107_default_page_size_with_extra",
        "sm107_default_page_size",
        "sm103_default_page_size",
        "sm103_extra_page_size",
    ],
)
def test_sm107_serves_only_extra_page_sizes(
    monkeypatch: pytest.MonkeyPatch,
    input_type: AttentionInputType,
    sm_version: int,
    tokens_per_block: int,
    extra_tokens_per_block: frozenset,
    expected: bool,
) -> None:
    """FP8-KV GQA 16:1 head-128 paged attention, the MiniMax-M3 Eagle3 draft layer shape."""
    monkeypatch.setattr(f"{_FLASHINFER_TRTLLM_GEN_MODULE}.get_sm_version", lambda: sm_version)
    fmha = object.__new__(FlashInferTrtllmGenFmha)
    fmha.kv_factor = 2
    monkeypatch.setattr(fmha, "_get_total_num_blocks", lambda _: 0)
    attn = SimpleNamespace(
        is_mla_enable=False,
        sparse_params=None,
        position_embedding_type=0,
        head_dim=128,
        num_heads=16,
        num_kv_heads=1,
    )
    is_context = input_type == AttentionInputType.context_only
    num_tokens = 4
    q_hidden_size = attn.num_heads * attn.head_dim
    qkv_hidden_size = q_hidden_size + 2 * attn.num_kv_heads * attn.head_dim
    q = torch.empty((num_tokens, qkv_hidden_size), dtype=torch.bfloat16)
    metadata = metadata_stub(
        num_contexts=1 if is_context else 0,
        num_generations=0 if is_context else 1,
        helix_position_offsets=None,
        num_sparse_topk=0,
        use_spec_decoding=False,
        is_spec_dec_tree=False,
        kv_cache_block_offsets=object(),
        kv_cache_manager=SimpleNamespace(
            dtype=DataType.FP8,
            trtllm_gen_extra_tokens_per_block=extra_tokens_per_block,
        ),
        is_cross=False,
        is_spec_decoding_enabled=False,
        tokens_per_block=tokens_per_block,
        beam_width=1,
    )
    forward_args = AttentionForwardArgs(
        output=torch.empty((num_tokens, q_hidden_size), dtype=torch.bfloat16),
        attention_input_type=input_type,
        is_fused_qkv=True,
        update_kv_cache=True,
    )

    phase = FmhaPhase.CONTEXT if is_context else FmhaPhase.GENERATION
    for requested_phase in (None, phase):
        supported, reason = fmha._is_supported_with_reason(
            q, None, None, attn, metadata, forward_args, phase=requested_phase
        )
        assert supported is expected, reason
        if not expected:
            assert "SM107" in reason
