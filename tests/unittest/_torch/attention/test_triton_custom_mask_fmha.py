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

from tensorrt_llm._torch.attention_backend import trtllm as trtllm_module
from tensorrt_llm._torch.attention_backend.fmha import flashinfer_trtllm_gen as trtllm_gen_module
from tensorrt_llm._torch.attention_backend.fmha.combined import CombinedFmha
from tensorrt_llm._torch.attention_backend.fmha.flashinfer_trtllm_gen import FlashInferTrtllmGenFmha
from tensorrt_llm._torch.attention_backend.fmha.interface import FmhaPhase
from tensorrt_llm._torch.attention_backend.fmha.phased import PhasedFmha
from tensorrt_llm._torch.attention_backend.fmha.registry import DEFAULT_FMHA_LIBS
from tensorrt_llm._torch.attention_backend.fmha.triton_custom_mask import TritonCustomMaskFmha
from tensorrt_llm._torch.attention_backend.interface import (
    AttentionForwardArgs,
    AttentionInputType,
    CustomAttentionMask,
    PredefinedAttentionMask,
)
from tensorrt_llm._torch.attention_backend.trtllm import TrtllmAttention, TrtllmAttentionMetadata
from tensorrt_llm.bindings.internal import thop
from tensorrt_llm.functional import PositionEmbeddingType
from tensorrt_llm.quantization.mode import QuantMode


def test_triton_custom_mask_precedes_general_fmha_libraries() -> None:
    assert DEFAULT_FMHA_LIBS == (
        "triton_custom_mask",
        "flashinfer_trtllm_gen",
        "fallback",
    )


def test_triton_custom_mask_implements_only_context_runner() -> None:
    assert TritonCustomMaskFmha.__bases__ == (PhasedFmha,)
    assert "run_context" in TritonCustomMaskFmha.__dict__
    assert "run_generation" not in TritonCustomMaskFmha.__dict__
    assert "_run_preprocessed_context" not in TritonCustomMaskFmha.__dict__


def test_trtllm_metadata_does_not_add_custom_mask_buffers() -> None:
    fields = TrtllmAttentionMetadata.__dataclass_fields__
    assert "custom_mask_qo_indptr" not in fields
    assert "custom_mask_cached_token_lens" not in fields


def test_custom_mask_context_skips_trtllm_gen_context_checks() -> None:
    attn = SimpleNamespace(is_mla_enable=False, quant_mode=0)
    fmha = object.__new__(TritonCustomMaskFmha)
    fmha._attn_ref = lambda: attn
    metadata = SimpleNamespace(
        num_contexts=1,
        num_generations=0,
        is_cross=False,
        kv_cache_block_offsets=object(),
    )
    q = torch.empty((2, 12), dtype=torch.float16)
    output = torch.empty((2, 4), dtype=torch.float16)
    forward_args = AttentionForwardArgs(
        output=output,
        attention_mask=CustomAttentionMask.CUSTOM,
        attention_mask_data=torch.ones((4,), dtype=torch.bool),
        is_fused_qkv=True,
    )

    assert fmha.is_supported(
        q,
        None,
        None,
        metadata,
        forward_args,
        phase=FmhaPhase.CONTEXT,
    )
    assert not fmha.is_supported(
        q,
        None,
        None,
        metadata,
        forward_args,
        phase=FmhaPhase.GENERATION,
    )


def test_flashinfer_custom_mask_support_is_phase_specific() -> None:
    attn = SimpleNamespace(
        is_mla_enable=False,
        sparse_params=None,
        head_dim=128,
        num_heads=2,
        num_kv_heads=1,
        position_embedding_type=int(PositionEmbeddingType.learned_absolute),
    )
    fmha = object.__new__(FlashInferTrtllmGenFmha)
    fmha._attn_ref = lambda: attn
    metadata = SimpleNamespace(
        helix_position_offsets=None,
        num_sparse_topk=0,
        use_spec_decoding=False,
        is_spec_dec_tree=False,
        kv_cache_block_offsets=object(),
        kv_cache_manager=None,
        tokens_per_block=32,
        is_cross=False,
        beam_width=1,
    )
    q = torch.empty((3, 512), dtype=torch.bfloat16)
    forward_args = AttentionForwardArgs(
        output=torch.empty((3, 256), dtype=torch.bfloat16),
        attention_mask=CustomAttentionMask.CUSTOM,
        attention_mask_data=torch.ones((4,), dtype=torch.bool),
        attention_input_type=AttentionInputType.mixed,
        is_fused_qkv=True,
    )

    assert not fmha.is_supported(
        q,
        None,
        None,
        metadata,
        forward_args,
        phase=FmhaPhase.CONTEXT,
    )
    assert fmha.is_supported(
        q,
        None,
        None,
        metadata,
        forward_args,
        phase=FmhaPhase.GENERATION,
    )


def test_custom_mask_mixed_batch_uses_generation_followup(monkeypatch) -> None:
    fmha = object.__new__(TritonCustomMaskFmha)
    generation_fmha = object.__new__(FlashInferTrtllmGenFmha)
    combined_fmha = object.__new__(CombinedFmha)
    backend = SimpleNamespace(
        phased_fmha_libs=[fmha, generation_fmha],
        combined_fmha=combined_fmha,
    )
    metadata = SimpleNamespace(
        num_contexts=1,
        num_generations=1,
        is_cross=False,
        kv_cache_block_offsets=object(),
    )
    q = torch.empty((3, 12), dtype=torch.float16)
    output = torch.empty((3, 4), dtype=torch.float16)
    attention_mask_data = torch.ones((4,), dtype=torch.bool)
    forward_args = AttentionForwardArgs(
        output=output,
        attention_mask=CustomAttentionMask.CUSTOM,
        attention_mask_data=attention_mask_data,
        attention_input_type=AttentionInputType.mixed,
        is_fused_qkv=True,
    )
    checked_calls = []

    def _check_context_request(
        self,
        checked_q,
        k,
        v,
        checked_metadata,
        checked_forward_args,
        *,
        phase,
    ):
        checked_calls.append(
            (
                self,
                phase,
                checked_q,
                checked_metadata,
                checked_forward_args,
            )
        )
        return phase == FmhaPhase.CONTEXT

    def _check_generation_request(
        self,
        checked_q,
        k,
        v,
        checked_metadata,
        checked_forward_args,
        *,
        phase,
    ):
        checked_calls.append(
            (
                self,
                phase,
                checked_q,
                checked_metadata,
                checked_forward_args,
            )
        )
        return phase == FmhaPhase.GENERATION

    monkeypatch.setattr(
        TritonCustomMaskFmha,
        "is_supported",
        _check_context_request,
    )
    monkeypatch.setattr(
        FlashInferTrtllmGenFmha,
        "is_supported",
        _check_generation_request,
    )
    selected_fmha = TrtllmAttention._select_non_mla_phased_fmha(
        backend,
        q,
        None,
        None,
        metadata,
        forward_args,
    )
    assert selected_fmha is combined_fmha
    assert [(call[0], call[1]) for call in checked_calls] == [
        (fmha, FmhaPhase.CONTEXT),
        (fmha, FmhaPhase.GENERATION),
        (generation_fmha, FmhaPhase.GENERATION),
    ]
    assert all(call[2] is q for call in checked_calls)
    assert all(call[3] is metadata for call in checked_calls)
    assert all(call[4] is forward_args for call in checked_calls)
    assert forward_args.attention_mask == CustomAttentionMask.CUSTOM
    assert forward_args.attention_mask_data is attention_mask_data
    assert forward_args.attention_input_type == AttentionInputType.mixed


def test_mixed_batch_reuses_one_phased_fmha_when_it_supports_both_phases(
    monkeypatch,
) -> None:
    fmha = object.__new__(FlashInferTrtllmGenFmha)
    checked_phases = []

    def _accept_phase(self, q, k, v, metadata, forward_args, *, phase):
        checked_phases.append(phase)
        return True

    monkeypatch.setattr(
        FlashInferTrtllmGenFmha,
        "is_supported",
        _accept_phase,
    )
    combined_fmha = object.__new__(CombinedFmha)
    backend = SimpleNamespace(
        phased_fmha_libs=[fmha],
        combined_fmha=combined_fmha,
    )
    metadata = SimpleNamespace(
        num_contexts=1,
        num_generations=1,
        is_cross=False,
    )
    forward_args = AttentionForwardArgs(attention_input_type=AttentionInputType.mixed)

    selected_fmha = TrtllmAttention._select_non_mla_phased_fmha(
        backend,
        torch.empty((2, 4)),
        None,
        None,
        metadata,
        forward_args,
    )

    assert selected_fmha is fmha
    assert checked_phases == [
        FmhaPhase.CONTEXT,
        FmhaPhase.GENERATION,
    ]


def test_mixed_batch_checks_followup_fmha_support(monkeypatch) -> None:
    context_fmha = object.__new__(TritonCustomMaskFmha)
    followup_fmha = object.__new__(FlashInferTrtllmGenFmha)
    checked_followup_phases = []

    monkeypatch.setattr(
        TritonCustomMaskFmha,
        "is_supported",
        lambda self, q, k, v, metadata, forward_args, *, phase: (phase == FmhaPhase.CONTEXT),
    )

    def _reject_followup(self, q, k, v, metadata, forward_args, *, phase):
        checked_followup_phases.append(phase)
        return False

    monkeypatch.setattr(
        FlashInferTrtllmGenFmha,
        "is_supported",
        _reject_followup,
    )
    backend = SimpleNamespace(
        phased_fmha_libs=[context_fmha, followup_fmha],
        combined_fmha=object.__new__(CombinedFmha),
    )
    metadata = SimpleNamespace(
        num_contexts=1,
        num_generations=1,
        is_cross=False,
    )
    forward_args = AttentionForwardArgs(attention_input_type=AttentionInputType.mixed)

    selected_fmha = TrtllmAttention._select_non_mla_phased_fmha(
        backend,
        torch.empty((2, 4)),
        None,
        None,
        metadata,
        forward_args,
    )

    assert selected_fmha is None
    assert checked_followup_phases == [FmhaPhase.GENERATION]


def test_non_custom_request_uses_flashinfer_anchor(
    monkeypatch,
) -> None:
    context_fmha = object.__new__(TritonCustomMaskFmha)
    generation_fmha = object.__new__(FlashInferTrtllmGenFmha)
    monkeypatch.setattr(
        TritonCustomMaskFmha,
        "is_supported",
        lambda self, q, k, v, metadata, forward_args, *, phase: False,
    )
    monkeypatch.setattr(
        FlashInferTrtllmGenFmha,
        "is_supported",
        lambda self, q, k, v, metadata, forward_args, *, phase: True,
    )
    backend = SimpleNamespace(
        phased_fmha_libs=[context_fmha, generation_fmha],
        combined_fmha=object.__new__(CombinedFmha),
    )
    metadata = SimpleNamespace(
        num_contexts=1,
        num_generations=1,
        is_cross=False,
    )
    forward_args = AttentionForwardArgs(attention_input_type=AttentionInputType.mixed)

    selected_fmha = TrtllmAttention._select_non_mla_phased_fmha(
        backend,
        torch.empty((2, 4)),
        None,
        None,
        metadata,
        forward_args,
    )

    assert selected_fmha is generation_fmha


def test_mla_selection_passes_generation_phase_to_request_support(
    monkeypatch,
) -> None:
    fmha = object.__new__(FlashInferTrtllmGenFmha)
    checked_args = None
    checked_phase = None

    def _accept_request(self, q, k, v, metadata, forward_args, *, phase):
        nonlocal checked_args, checked_phase
        checked_args = forward_args
        checked_phase = phase
        return True

    monkeypatch.setattr(
        FlashInferTrtllmGenFmha,
        "is_supported",
        _accept_request,
    )
    backend = SimpleNamespace(fmha_libs=[fmha])
    forward_args = AttentionForwardArgs(attention_input_type=AttentionInputType.generation_only)

    selected_fmha = TrtllmAttention._select_mla_fmha(
        backend,
        torch.empty((1, 4)),
        None,
        None,
        SimpleNamespace(),
        forward_args,
    )

    assert selected_fmha is fmha
    assert checked_args is forward_args
    assert checked_phase == FmhaPhase.GENERATION


@pytest.mark.parametrize("is_mla_enable", [False, True])
def test_combined_fmha_is_created_only_for_non_mla_layers(
    monkeypatch,
    is_mla_enable: bool,
) -> None:
    class TestPhasedFmha(PhasedFmha):
        @classmethod
        def is_available(cls, attn) -> bool:
            return True

        def __init__(self, attn):
            self._attn_ref = lambda: attn

    monkeypatch.setattr(
        trtllm_module,
        "get_enabled_fmha_lib_classes",
        lambda: [TestPhasedFmha],
    )

    class TestAttention:
        pass

    backend = TestAttention()
    backend.is_mla_enable = is_mla_enable
    backend.kv_lora_rank = None
    backend.head_dim = 128
    backend.v_head_dim = None

    TrtllmAttention.create_fmha_libs(backend)

    assert len(backend.phased_fmha_libs) == 1
    assert not backend.non_phased_fmha_libs
    assert isinstance(backend.combined_fmha, CombinedFmha) is not is_mla_enable


def test_mixed_batch_runs_context_and_generation_on_different_fmhas(monkeypatch) -> None:
    attn = SimpleNamespace(
        is_mla_enable=False,
        num_heads=1,
        num_kv_heads=1,
        head_dim=4,
        v_head_dim=None,
        kv_lora_rank=None,
        predicted_tokens_per_seq=1,
    )
    context_fmha = object.__new__(TritonCustomMaskFmha)
    context_fmha._attn_ref = lambda: attn
    context_fmha.kv_factor = 2
    context_fmha.context_out_head_size = 4
    context_fmha.generation_out_head_size = 4
    generation_fmha = object.__new__(FlashInferTrtllmGenFmha)
    generation_fmha._attn_ref = lambda: attn
    combined_fmha = object.__new__(CombinedFmha)
    combined_fmha._attn_ref = lambda: attn
    combined_fmha.kv_factor = 2
    combined_fmha.context_out_head_size = 4
    combined_fmha.generation_out_head_size = 4

    called_phases = []
    prepared_phases = []

    def _prepare_context(self, q, k, v, metadata, forward_args, workspace):
        prepared_phases.append(
            (
                "context",
                forward_args,
                forward_args.attention_input_type,
                forward_args.attention_mask,
                forward_args.attention_mask_data,
            )
        )
        workspace.resize_(4)

    def _prepare_generation(self, q, k, v, metadata, forward_args, workspace):
        prepared_phases.append(
            (
                "generation",
                forward_args,
                forward_args.attention_input_type,
                forward_args.attention_mask,
                forward_args.attention_mask_data,
            )
        )
        workspace.resize_(8)

    def _run_context(self, params):
        called_phases.append(
            (
                "context",
                self,
                params.fwd,
                params.fwd.attention_input_type,
                params.fwd.attention_mask,
                params.fwd.attention_mask_data,
            )
        )

    def _run_generation(self, params):
        called_phases.append(
            (
                "generation",
                self,
                params.fwd,
                params.fwd.attention_input_type,
                params.fwd.attention_mask,
                params.fwd.attention_mask_data,
            )
        )

    monkeypatch.setattr(TritonCustomMaskFmha, "run_context", _run_context)
    monkeypatch.setattr(FlashInferTrtllmGenFmha, "run_generation", _run_generation)
    monkeypatch.setattr(TritonCustomMaskFmha, "prepare_workspace", _prepare_context)
    monkeypatch.setattr(FlashInferTrtllmGenFmha, "prepare_workspace", _prepare_generation)

    metadata = SimpleNamespace(
        kv_cache_block_offsets=object(),
        effective_workspace=torch.empty(0, dtype=torch.int8),
        num_contexts=1,
        num_ctx_tokens=2,
        num_generations=1,
        kv_lens_cuda_runtime=torch.tensor([2, 5], dtype=torch.int32),
        kv_lens_runtime=torch.tensor([2, 5], dtype=torch.int32),
        prompt_lens_cuda_runtime=torch.tensor([2, 4], dtype=torch.int32),
        prompt_lens_cpu_runtime=torch.tensor([2, 4], dtype=torch.int32),
        beam_width=1,
        cache_indirection=None,
        tokens_per_block=32,
        kv_cache_manager=None,
        is_cross=False,
        is_spec_decoding_enabled=False,
    )
    q = torch.empty((3, 12), dtype=torch.float16)
    attention_mask_data = torch.ones(4, dtype=torch.bool)
    forward_args = AttentionForwardArgs(
        output=torch.empty((3, 4), dtype=torch.float16),
        attention_mask=CustomAttentionMask.CUSTOM,
        attention_mask_data=attention_mask_data,
        attention_input_type=AttentionInputType.mixed,
        attention_window_size=8,
        is_fused_qkv=True,
    )
    combined_fmha.set_fmha_impls(
        context_fmha,
        generation_fmha,
    )

    combined_fmha.forward(q, None, None, metadata, forward_args)

    assert [(phase, fmha) for phase, fmha, *_ in called_phases] == [
        ("context", context_fmha),
        ("generation", generation_fmha),
    ]
    assert all(call[2] is forward_args for call in called_phases)
    assert called_phases[0][3] == AttentionInputType.mixed
    assert called_phases[0][4] == CustomAttentionMask.CUSTOM
    assert called_phases[0][5] is attention_mask_data
    assert called_phases[1][3] == AttentionInputType.mixed
    assert called_phases[1][4] == PredefinedAttentionMask.CAUSAL
    assert called_phases[1][5] is None
    assert [phase for phase, *_ in prepared_phases] == ["context", "generation"]
    assert all(prepared[1] is forward_args for prepared in prepared_phases)
    assert all(prepared[2] == AttentionInputType.mixed for prepared in prepared_phases)
    assert all(prepared[3] == CustomAttentionMask.CUSTOM for prepared in prepared_phases)
    assert all(prepared[4] is attention_mask_data for prepared in prepared_phases)
    assert forward_args.attention_mask == PredefinedAttentionMask.CAUSAL
    assert forward_args.attention_mask_data is None
    assert metadata.effective_workspace.numel() == 8


@pytest.mark.parametrize(
    "quant_mode",
    [
        QuantMode(0),
        QuantMode.from_description(use_fp8_kv_cache=True),
    ],
)
def test_large_head_generation_support_is_owned_by_trtllm_gen(
    quant_mode: QuantMode,
) -> None:
    attn = SimpleNamespace(
        head_dim=512,
        is_mla_enable=False,
        quant_mode=int(quant_mode),
        position_embedding_type=int(PositionEmbeddingType.learned_absolute),
    )
    fmha = object.__new__(FlashInferTrtllmGenFmha)
    fmha._attn_ref = lambda: attn
    metadata = SimpleNamespace(
        num_contexts=0,
        num_generations=1,
        is_cross=False,
        kv_cache_block_offsets=object(),
    )
    q = torch.empty((1, 8), dtype=torch.bfloat16)
    k = torch.empty((1, 4), dtype=torch.bfloat16)
    v = torch.empty((1, 4), dtype=torch.bfloat16)
    output = torch.empty_like(q)
    forward_args = AttentionForwardArgs(
        output=output,
        attention_mask=PredefinedAttentionMask.CAUSAL,
        attention_input_type=AttentionInputType.generation_only,
        is_fused_qkv=False,
    )

    supported, reason = fmha._check_preprocessed_generation_with_reason(
        q,
        k,
        v,
        metadata,
        forward_args,
    )

    assert supported, reason


def test_fp8_kv_uses_fp8_context_fmha() -> None:
    attn = SimpleNamespace(
        quant_mode=int(QuantMode.from_description(use_fp8_kv_cache=True)),
    )
    fmha = object.__new__(FlashInferTrtllmGenFmha)
    fmha._attn_ref = lambda: attn

    assert fmha._use_fp8_context_fmha(torch.empty(1, dtype=torch.bfloat16))


def test_preprocessed_generation_launches_trtllm_gen_directly(monkeypatch) -> None:
    fp8_kv_quant_mode = QuantMode.from_description(use_fp8_kv_cache=True)
    attn = SimpleNamespace(
        num_heads=2,
        num_kv_heads=1,
        head_dim=512,
        quant_mode=int(fp8_kv_quant_mode),
        local_layer_idx=0,
        q_scaling=1.0,
        attention_chunk_size=None,
    )
    metadata = SimpleNamespace(
        kv_cache_manager=object(),
        num_generations=1,
        seq_lens=torch.tensor([1], dtype=torch.int32),
        kv_cache_params=SimpleNamespace(num_cached_tokens_per_seq=[4]),
        beam_width=1,
        max_num_requests=1,
        host_kv_cache_pool_pointers=torch.empty(0),
        host_kv_cache_pool_mapping=torch.empty(0),
        kv_cache_block_offsets=torch.empty(0),
    )
    params = SimpleNamespace(
        attn=attn,
        meta=metadata,
        fwd=AttentionForwardArgs(),
        qkv_input=torch.randn(1, 2 * 512, dtype=torch.bfloat16),
        key_input=torch.randn(1, 512, dtype=torch.bfloat16),
        value_input=torch.randn(1, 512, dtype=torch.bfloat16),
        context_buf=torch.empty(1, 2, 512, dtype=torch.bfloat16),
        sequence_lengths=torch.tensor([5], dtype=torch.int32),
        workspace=torch.empty(1024, dtype=torch.int8),
        num_tokens=1,
        seq_offset=0,
        num_requests=1,
        tokens_per_block=32,
        kv_factor=2,
        total_num_blocks=4,
        input_seq_length=1,
        max_past_kv_length=5,
        cyclic_attention_window_size=5,
    )
    fmha = object.__new__(FlashInferTrtllmGenFmha)
    fmha._attn_ref = lambda: attn
    fmha._enable_pdl = False
    fmha._multi_processor_count = 1

    appended_kv_dtypes = None

    def _capture_append(params, q, k, v, *args, **kwargs):
        nonlocal appended_kv_dtypes
        appended_kv_dtypes = (q.dtype, k.dtype, v.dtype)
        return object(), object(), object(), object(), [5]

    monkeypatch.setattr(
        FlashInferTrtllmGenFmha,
        "_append_preprocessed_kv",
        staticmethod(_capture_append),
    )
    kv_pool = torch.empty(1)
    block_tables = torch.zeros(1, 2, 4, dtype=torch.int32)
    metadata_dtype = None

    def _build_metadata(*args, **kwargs):
        nonlocal metadata_dtype
        metadata_dtype = args[-1]
        return kv_pool, block_tables

    monkeypatch.setattr(
        thop,
        "build_trtllm_gen_kv_cache_metadata",
        _build_metadata,
    )
    monkeypatch.setattr(
        trtllm_gen_module,
        "_clear_multi_ctas_kv_counter_workspace",
        lambda *args, **kwargs: None,
    )
    decode_args = None

    def _capture_decode(*args):
        nonlocal decode_args
        decode_args = args

    monkeypatch.setattr(
        trtllm_gen_module,
        "_trtllm_gen_batch_decode_with_kv_cache",
        _capture_decode,
    )

    fmha._run_preprocessed_generation(params)

    assert decode_args is not None
    assert decode_args[0].shape == (1, 2, 512)
    assert decode_args[0].dtype == torch.float8_e4m3fn
    assert appended_kv_dtypes == (torch.float8_e4m3fn,) * 3
    assert metadata_dtype == torch.float8_e4m3fn
    assert decode_args[1] is kv_pool
    assert decode_args[3] is block_tables
    assert decode_args[5] == 5


def test_large_head_context_requires_module_side_rope() -> None:
    attn = SimpleNamespace(
        head_dim=512,
        is_mla_enable=False,
        quant_mode=0,
        position_embedding_type=int(PositionEmbeddingType.rope_gpt_neox),
    )
    fmha = object.__new__(TritonCustomMaskFmha)
    fmha._attn_ref = lambda: attn
    metadata = SimpleNamespace(
        num_contexts=1,
        num_generations=0,
        is_cross=False,
        kv_cache_block_offsets=object(),
    )
    q = torch.empty((2, 8), dtype=torch.bfloat16)
    output = torch.empty_like(q)
    forward_args = AttentionForwardArgs(
        output=output,
        attention_mask=CustomAttentionMask.CUSTOM,
        attention_mask_data=torch.ones((4,), dtype=torch.bool),
        attention_input_type=AttentionInputType.context_only,
        is_fused_qkv=True,
    )

    assert not fmha.is_supported(
        q,
        None,
        None,
        metadata,
        forward_args,
        phase=FmhaPhase.CONTEXT,
    )


def test_triton_prefill_accepts_separate_kv_page_tables() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the Triton kernel test.")

    from tensorrt_llm._torch.attention_backend.triton_prefill import triton_prefill_with_custom_mask

    torch.manual_seed(0)
    device = torch.device("cuda")
    dtype = torch.float16
    q_len = 2
    prefix_len = 2
    head_dim = 64
    page_size = 2
    q = torch.randn(q_len, 1, head_dim, device=device, dtype=dtype)
    k = torch.randn(q_len, 1, head_dim, device=device, dtype=dtype)
    v = torch.randn(q_len, 1, head_dim, device=device, dtype=dtype)
    k_prefix = torch.randn(prefix_len, 1, head_dim, device=device, dtype=dtype)
    v_prefix = torch.randn(prefix_len, 1, head_dim, device=device, dtype=dtype)

    pool = torch.zeros(2, 1, page_size, head_dim, device=device, dtype=dtype)
    pool[0].copy_(k_prefix.transpose(0, 1))
    pool[1].copy_(v_prefix.transpose(0, 1))
    output = torch.empty_like(q)
    custom_mask = torch.tensor(
        [
            [True, False, True, False],
            [False, True, True, True],
        ],
        device=device,
        dtype=torch.bool,
    )

    triton_prefill_with_custom_mask(
        q=q,
        k=k,
        v=v,
        output=output,
        qo_indptr=torch.tensor([0, q_len], device=device, dtype=torch.int32),
        kv_cache=None,
        prefix_lens=torch.tensor([prefix_len], device=device, dtype=torch.int32),
        page_table_indptr=torch.tensor([0, 1], device=device, dtype=torch.int32),
        page_table_indices=torch.tensor([0], device=device, dtype=torch.int32),
        page_size=page_size,
        custom_mask=custom_mask.flatten(),
        sm_scale=head_dim**-0.5,
        k_cache=pool,
        v_cache=pool,
        v_page_table_indices=torch.tensor([1], device=device, dtype=torch.int32),
    )

    keys = torch.cat([k_prefix, k], dim=0).squeeze(1).float()
    values = torch.cat([v_prefix, v], dim=0).squeeze(1).float()
    scores = q.squeeze(1).float() @ keys.T * head_dim**-0.5
    scores.masked_fill_(~custom_mask, float("-inf"))
    reference = (scores.softmax(dim=-1) @ values).to(dtype).unsqueeze(1)
    torch.testing.assert_close(output, reference, atol=2e-2, rtol=2e-2)


def test_triton_prefill_causal_generation_with_large_head() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the Triton kernel test.")

    from tensorrt_llm._torch.attention_backend.triton_prefill import triton_prefill_with_custom_mask

    torch.manual_seed(1)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    head_dim = 512
    prefix_len = 3
    page_size = 4
    q = torch.randn(1, 1, head_dim, device=device, dtype=dtype)
    k = torch.randn(1, 1, head_dim, device=device, dtype=dtype)
    v = torch.randn(1, 1, head_dim, device=device, dtype=dtype)
    k_prefix = torch.randn(prefix_len, 1, head_dim, device=device, dtype=dtype)
    v_prefix = torch.randn(prefix_len, 1, head_dim, device=device, dtype=dtype)
    kv_cache = torch.zeros(
        1,
        2,
        1,
        page_size,
        head_dim,
        device=device,
        dtype=dtype,
    )
    kv_cache[0, 0, 0, :prefix_len].copy_(k_prefix.squeeze(1))
    kv_cache[0, 1, 0, :prefix_len].copy_(v_prefix.squeeze(1))
    output = torch.empty_like(q)

    triton_prefill_with_custom_mask(
        q=q,
        k=k,
        v=v,
        output=output,
        qo_indptr=torch.tensor([0, 1], device=device, dtype=torch.int32),
        kv_cache=kv_cache,
        prefix_lens=torch.tensor([prefix_len], device=device, dtype=torch.int32),
        page_table_indptr=torch.tensor([0, 1], device=device, dtype=torch.int32),
        page_table_indices=torch.tensor([0], device=device, dtype=torch.int32),
        page_size=page_size,
        custom_mask=None,
        sm_scale=head_dim**-0.5,
    )

    keys = torch.cat([k_prefix, k], dim=0).squeeze(1).float()
    values = torch.cat([v_prefix, v], dim=0).squeeze(1).float()
    reference = ((q.squeeze(1).float() @ keys.T * head_dim**-0.5).softmax(dim=-1) @ values).to(
        dtype
    )
    torch.testing.assert_close(output.squeeze(1), reference, atol=3e-2, rtol=3e-2)


def test_gemma4_large_head_context_and_generation_match_flashinfer() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the Gemma4 attention test.")

    transformers = pytest.importorskip("transformers")
    if not hasattr(transformers, "Gemma4TextConfig"):
        pytest.skip("The installed transformers version does not support Gemma4.")

    import tensorrt_llm
    from tensorrt_llm._torch.attention_backend.utils import get_attention_backend
    from tensorrt_llm._torch.metadata import KVCacheParams
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm._torch.models.modeling_gemma4 import Gemma4Attention
    from tensorrt_llm._torch.pyexecutor.kv_cache_manager_v2 import KVCacheManagerV2
    from tensorrt_llm.llmapi.llm_args import KvCacheConfig
    from tensorrt_llm.mapping import Mapping

    config = transformers.Gemma4TextConfig(
        model_type="gemma4_text",
        vocab_size=128,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=256,
        global_head_dim=512,
        num_global_key_value_heads=1,
        hidden_activation="gelu_pytorch_tanh",
        max_position_embeddings=128,
        rms_norm_eps=1e-6,
        sliding_window=64,
        attention_k_eq_v=True,
        use_bidirectional_attention="vision",
        rope_parameters={
            "sliding_attention": {
                "rope_type": "default",
                "rope_theta": 10000.0,
            },
            "full_attention": {
                "rope_type": "proportional",
                "partial_rotary_factor": 0.25,
                "rope_theta": 1000000.0,
            },
        },
        torch_dtype="bfloat16",
        tie_word_embeddings=True,
        attention_bias=False,
        attention_dropout=0.0,
    )
    mapping = Mapping(world_size=1, tp_size=1, rank=0)
    seq_len = 4
    hidden = torch.randn(seq_len, config.hidden_size, dtype=torch.bfloat16, device="cuda")
    generation_hidden = torch.randn(
        1,
        config.hidden_size,
        dtype=torch.bfloat16,
        device="cuda",
    )
    position_ids = torch.arange(seq_len, dtype=torch.int32, device="cuda").unsqueeze(0)
    generation_position_ids = torch.tensor([[seq_len]], dtype=torch.int32, device="cuda")
    custom_mask = torch.ones(seq_len * seq_len, dtype=torch.bool, device="cuda")

    mixed_hidden = torch.cat([hidden, generation_hidden], dim=0)
    mixed_position_ids = torch.cat([position_ids, generation_position_ids], dim=1)

    def run_backend(backend: str) -> tuple[torch.Tensor, torch.Tensor]:
        model_config = ModelConfig(
            pretrained_config=config,
            mapping=mapping,
            attn_backend=backend,
        )
        attention = (
            Gemma4Attention(
                model_config,
                layer_idx=0,
                is_sliding=False,
            )
            .cuda()
            .eval()
        )
        torch.manual_seed(7)
        with torch.no_grad():
            for parameter in attention.parameters():
                if parameter.is_floating_point():
                    parameter.normal_(mean=0.0, std=0.02)

        manager = KVCacheManagerV2(
            KvCacheConfig(max_tokens=64, enable_block_reuse=False),
            tensorrt_llm.bindings.internal.batch_manager.CacheType.SELF,
            num_layers=1,
            num_kv_heads=1,
            head_dim=512,
            tokens_per_block=32,
            max_seq_len=64,
            max_batch_size=2,
            mapping=mapping,
            dtype=tensorrt_llm.bindings.DataType.BF16,
        )
        assert manager.add_dummy_requests([1, 2], [seq_len + 1, seq_len]) is not None
        metadata_cls = get_attention_backend(backend).Metadata
        context_metadata = metadata_cls(
            seq_lens=torch.tensor([seq_len], dtype=torch.int32),
            num_contexts=1,
            kv_cache_params=KVCacheParams(
                use_cache=True,
                num_cached_tokens_per_seq=[0],
            ),
            max_num_requests=2,
            max_num_tokens=64,
            kv_cache_manager=manager,
            request_ids=[1],
            prompt_lens=[seq_len],
        )
        context_metadata.prepare()
        with torch.inference_mode():
            context_output = attention(
                position_ids,
                hidden,
                context_metadata,
                attention_mask=CustomAttentionMask.CUSTOM,
                attention_mask_data=custom_mask,
            ).clone()

        mixed_metadata = metadata_cls(
            seq_lens=torch.tensor([seq_len, 1], dtype=torch.int32),
            num_contexts=1,
            kv_cache_params=KVCacheParams(
                use_cache=True,
                num_cached_tokens_per_seq=[0, seq_len],
            ),
            max_num_requests=2,
            max_num_tokens=64,
            kv_cache_manager=manager,
            request_ids=[2, 1],
            prompt_lens=[seq_len, seq_len],
        )
        mixed_metadata.prepare()
        with torch.inference_mode():
            mixed_output = attention(
                mixed_position_ids,
                mixed_hidden,
                mixed_metadata,
                attention_mask=CustomAttentionMask.CUSTOM,
                attention_mask_data=custom_mask,
            ).clone()
        manager.shutdown()
        return context_output, mixed_output

    flashinfer_context, flashinfer_mixed = run_backend("FLASHINFER")
    trtllm_context, trtllm_mixed = run_backend("TRTLLM")

    torch.testing.assert_close(trtllm_context, flashinfer_context, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(trtllm_mixed, flashinfer_mixed, atol=2e-2, rtol=2e-2)
