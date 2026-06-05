# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import torch

from tensorrt_llm._torch.attention_backend.fmha import (
    BaseFmha,
    FallbackFmha,
    FmhaFeature,
    FmhaSupportContext,
)
from tensorrt_llm._torch.attention_backend.interface import AttentionInputType
from tensorrt_llm._torch.attention_backend.trtllm import iter_enabled_fmha_libs
from tensorrt_llm._torch.attention_backend.trtllm_gen import (
    FLASHINFER_TRTLLM_GEN_CAPABILITIES,
    FlashInferTrtllmGenFmha,
)
from tensorrt_llm.bindings import DataType
from tensorrt_llm.functional import AttentionMaskType, PositionEmbeddingType


class StaticFlashInferTrtllmGenFmha(BaseFmha):
    capabilities = FLASHINFER_TRTLLM_GEN_CAPABILITIES


def _context(**overrides) -> FmhaSupportContext:
    values = dict(
        sm=100,
        q_dtype=torch.float16,
        kv_cache_dtype=DataType.HALF,
        output_dtype=torch.float16,
        num_heads=16,
        num_kv_heads=8,
        head_size=64,
        attention_input_type=AttentionInputType.context_only,
        mask_type=AttentionMaskType.causal,
        position_embedding_type=PositionEmbeddingType.rope_gpt_neox,
        beam_width=1,
        tokens_per_block=16,
        has_kv_cache_manager=True,
        use_paged_kv_cache=True,
        is_mla_enable=False,
        kv_lora_rank=None,
        qk_rope_head_dim=None,
        has_context_phase=True,
        has_generation_phase=False,
        is_cross_attention=False,
        is_spec_decoding=False,
        is_padded=False,
        position_shift_enabled=False,
        active_helix=False,
        use_sage_attention=False,
        has_sparse_attention=False,
        has_skip_softmax_attention=False,
    )
    values.update(overrides)
    return FmhaSupportContext(**values)


def test_default_fmha_order(monkeypatch):
    monkeypatch.delenv("TLLM_FMHA_LIBS", raising=False)
    assert [fmha.capabilities.name for fmha in iter_enabled_fmha_libs()] == [
        "flashinfer_trtllm_gen",
        "fallback",
    ]


def test_fmha_env_delta_disables_flashinfer_but_keeps_fallback(monkeypatch):
    monkeypatch.setenv("TLLM_FMHA_LIBS", "-flashinfer_trtllm_gen,-fallback")
    assert [fmha.capabilities.name for fmha in iter_enabled_fmha_libs()] == ["fallback"]


def test_fmha_env_delta_raises_unknown_name(monkeypatch):
    monkeypatch.setenv("TLLM_FMHA_LIBS", "+does_not_exist")
    with pytest.raises(ValueError, match="Unknown FMHA library"):
        iter_enabled_fmha_libs()


def test_flashinfer_fmha_uses_declarative_capabilities():
    assert FlashInferTrtllmGenFmha.capabilities is FLASHINFER_TRTLLM_GEN_CAPABILITIES
    assert FLASHINFER_TRTLLM_GEN_CAPABILITIES.context.head_sizes.values
    assert FLASHINFER_TRTLLM_GEN_CAPABILITIES.mla.generation_cases
    assert (576, 512, 32) not in {
        (case.head_dim_qk, case.head_dim_v, case.tokens_per_block)
        for case in FLASHINFER_TRTLLM_GEN_CAPABILITIES.mla.generation_cases
    }


def test_fallback_fmha_has_declarative_capabilities():
    assert FallbackFmha.capabilities.runtime_features == frozenset(FmhaFeature)
    assert FallbackFmha.capabilities.generation_head_ratios.min_value == 1
    assert FallbackFmha.is_supported(
        _context(
            sm=90,
            kv_cache_dtype=None,
            output_dtype=None,
            tokens_per_block=None,
            has_kv_cache_manager=False,
            use_paged_kv_cache=False,
        )
    )


def test_flashinfer_capabilities_accept_supported_context():
    assert StaticFlashInferTrtllmGenFmha.is_supported(_context())


def test_flashinfer_capabilities_do_not_accept_context_head_size_without_kernel():
    context = _context(head_size=96)
    assert not StaticFlashInferTrtllmGenFmha.is_supported(context)


def test_flashinfer_capabilities_do_not_accept_alibi():
    context = _context(position_embedding_type=PositionEmbeddingType.alibi)
    assert not StaticFlashInferTrtllmGenFmha.is_supported(context)


def test_flashinfer_capabilities_do_not_accept_generation_beam_search():
    context = _context(
        attention_input_type=AttentionInputType.generation_only,
        has_context_phase=False,
        has_generation_phase=True,
        beam_width=2,
    )
    assert not StaticFlashInferTrtllmGenFmha.is_supported(context)


def test_flashinfer_capabilities_do_not_accept_mla_context_phase():
    context = _context(
        attention_input_type=AttentionInputType.mixed,
        has_context_phase=True,
        has_generation_phase=True,
        is_mla_enable=True,
        head_size=320,
        kv_lora_rank=256,
        qk_rope_head_dim=64,
    )
    assert not StaticFlashInferTrtllmGenFmha.is_supported(context)


def test_flashinfer_capabilities_do_not_accept_tokens_per_block_without_kernel():
    context = _context(
        attention_input_type=AttentionInputType.generation_only,
        has_context_phase=False,
        has_generation_phase=True,
        tokens_per_block=8,
    )
    assert not StaticFlashInferTrtllmGenFmha.is_supported(context)


def test_flashinfer_capabilities_accept_declared_mla_generation_case():
    context = _context(
        attention_input_type=AttentionInputType.generation_only,
        has_context_phase=False,
        has_generation_phase=True,
        is_mla_enable=True,
        head_size=576,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        tokens_per_block=64,
    )
    assert StaticFlashInferTrtllmGenFmha.is_supported(context)


def test_flashinfer_capabilities_do_not_accept_undeclared_mla_generation_case():
    context = _context(
        attention_input_type=AttentionInputType.generation_only,
        has_context_phase=False,
        has_generation_phase=True,
        is_mla_enable=True,
        head_size=576,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        tokens_per_block=32,
    )
    assert not StaticFlashInferTrtllmGenFmha.is_supported(context)


def test_flashinfer_dynamic_checks_extend_base_support(monkeypatch):
    monkeypatch.setenv("TRTLLM_ENABLE_TRTLLM_GEN_ATTENTION", "0")
    assert not FlashInferTrtllmGenFmha.is_supported(_context())
