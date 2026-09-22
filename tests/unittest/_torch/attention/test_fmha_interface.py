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

import inspect
from types import SimpleNamespace
from typing import cast
from unittest.mock import Mock, patch

import pytest
import torch
from fmha_test_utils import FakeAttention

from tensorrt_llm._torch.attention.backends.fmha.fallback import FallbackFmha
from tensorrt_llm._torch.attention.backends.fmha.interface import Fmha, FmhaPhase
from tensorrt_llm._torch.attention.backends.fmha.phased import PhasedFmha
from tensorrt_llm._torch.attention.backends.fmha.registry import FMHA_LIBS
from tensorrt_llm._torch.attention.backends.interface import (
    AttentionForwardArgs,
    AttentionInputType,
)
from tensorrt_llm._torch.attention.backends.sparse.params import (
    BlockSparseForwardInputs,
    SparseRuntimeParams,
)
from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttention, TrtllmAttentionMetadata
from tensorrt_llm.bindings.internal import thop
from tensorrt_llm.functional import AttentionMaskType


class _MinimalFmha(Fmha):
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor | None,
        v: torch.Tensor | None,
        metadata: TrtllmAttentionMetadata,
        forward_args: AttentionForwardArgs,
    ) -> None:
        pass


@pytest.mark.parametrize(
    "input_type,num_contexts",
    [
        (AttentionInputType.context_only, 1),
        (AttentionInputType.generation_only, 0),
        (AttentionInputType.generation_only, 1),
        (AttentionInputType.mixed, 0),
        (AttentionInputType.mixed, 1),
        (None, 0),
        (None, 1),
    ],
)
@pytest.mark.parametrize(
    "is_mla,is_cross,beam_width",
    [(False, False, 1), (True, False, 1), (False, False, 4), (False, True, 4)],
    ids=["mha", "mla", "self-beams", "cross-beams"],
)
@pytest.mark.parametrize("use_spec_decoding", [False, True])
@pytest.mark.parametrize("is_spec_decoding_enabled", [False, True])
def test_legacy_fallback_dispatch_uses_native_phase_params(
    monkeypatch: pytest.MonkeyPatch,
    input_type: AttentionInputType | None,
    num_contexts: int,
    is_mla: bool,
    is_cross: bool,
    beam_width: int,
    use_spec_decoding: bool,
    is_spec_decoding_enabled: bool,
) -> None:
    has_context = num_contexts > 0 and input_type != AttentionInputType.generation_only
    num_generations = 0 if input_type == AttentionInputType.context_only else 4
    num_ctx_tokens = 3 if has_context else 0
    num_tokens = num_ctx_tokens + num_generations
    prompts = torch.tensor([3] * num_contexts + [9] * num_generations, dtype=torch.int32)
    kv_lengths = torch.tensor([5] * num_contexts + [10] * num_generations, dtype=torch.int32)
    past_lengths = kv_lengths.clone()
    if is_mla:
        past_lengths[:num_contexts].zero_()
    total_kv_lengths = torch.tensor([5 * num_contexts, 10 * num_generations], dtype=torch.int32)
    head_dim = 6 if is_mla else 4
    q = torch.empty((num_tokens, head_dim if is_mla else 3 * head_dim), dtype=torch.bfloat16)
    output = torch.empty((num_tokens, 4), dtype=torch.bfloat16)
    counter = torch.zeros(64, dtype=torch.uint8)
    spec_lengths = torch.ones(4, dtype=torch.int32)
    spec_offsets = torch.zeros((4, 1), dtype=torch.int32)
    spec_mask = torch.zeros((4, 1), dtype=torch.int32)
    spec_tree_offsets = torch.zeros(4, dtype=torch.int64)
    spec_tree_mask = torch.zeros(4, dtype=torch.uint32)
    spec_sparse_offsets = torch.zeros(4, dtype=torch.int32)
    calls: list[tuple] = []
    phase_params: list[thop.FmhaParams] = []

    def record(phase: str, params: thop.FmhaParams) -> int:
        assert isinstance(params, thop.FmhaParams)
        assert params.max_num_sequences == 8
        assert params.host_context_lengths.data_ptr() == prompts.data_ptr()
        assert params.host_past_key_value_lengths.data_ptr() == past_lengths.data_ptr()
        assert params.host_total_kv_lens.data_ptr() == total_kv_lengths.data_ptr()
        # The flat compatibility entry point delegates activation to native prepare().
        for native_tensor, source_tensor in (
            (params.spec_decoding_generation_lengths, spec_lengths),
            (params.spec_decoding_position_offsets, spec_offsets),
            (params.spec_decoding_packed_mask, spec_mask),
            (params.spec_decoding_bl_tree_mask_offset, spec_tree_offsets),
            (params.spec_decoding_bl_tree_mask, spec_tree_mask),
            (params.spec_bl_tree_first_sparse_mask_offset_kv, spec_sparse_offsets),
        ):
            assert native_tensor.data_ptr() == source_tensor.data_ptr()
        if phase != "workspace":
            phase_params.append(params)
            offset = 0 if phase == "context" else num_contexts
            assert (
                params.context_lengths.tolist()
                == prompts[offset : offset + params.num_seqs].tolist()
            )
            assert params.qkv_or_q.shape[0] == params.num_tokens
            assert params.output.shape[0] == params.num_tokens
        calls.append(
            (
                phase,
                params.num_seqs,
                params.num_tokens,
                params.seq_offset,
                params.token_offset,
                params.beam_width,
                params.num_requests,
            )
        )
        return 0

    op = SimpleNamespace(
        get_attention_workspace_size=lambda params, *args: record("workspace", params),
        run_context=lambda params: record("context", params),
        run_generation=lambda params: record("generation", params),
        run_mla_generation=lambda params: record("mla_generation", params),
    )
    monkeypatch.setattr(
        FallbackFmha, "_compat_attention_ops", SimpleNamespace(get=lambda *args: op)
    )
    monkeypatch.setattr(
        "tensorrt_llm._torch.attention.backends.fmha.fallback.get_multi_ctas_kv_counter",
        lambda *args: counter,
    )
    arguments = {
        name: param.default if param.default is not inspect.Parameter.empty else None
        for name, param in inspect.signature(FallbackFmha.attention).parameters.items()
    }
    arguments.update(
        q=q,
        output=output,
        workspace=torch.empty(0, dtype=torch.uint8),
        sequence_length=kv_lengths,
        context_lengths=prompts,
        host_context_lengths=prompts,
        host_past_key_value_lengths=past_lengths,
        host_total_kv_lens=total_kv_lengths,
        num_heads=1,
        num_kv_heads=1,
        head_size=head_dim,
        tokens_per_block=64,
        max_num_requests=8,
        max_context_length=16,
        max_seq_len=16,
        attention_window_size=16,
        beam_width=beam_width,
        mask_type=AttentionMaskType.causal,
        quant_mode=0,
        q_scaling=1.0,
        position_embedding_type=0,
        local_layer_idx=0,
        rope_dim=0,
        rope_base=10000.0,
        rope_scale_type=0,
        rope_scale=1.0,
        rope_short_m_scale=1.0,
        rope_long_m_scale=1.0,
        rope_max_positions=16,
        rope_original_max_positions=16,
        predicted_tokens_per_seq=1,
        is_spec_decoding_enabled=is_spec_decoding_enabled,
        use_spec_decoding=use_spec_decoding,
        spec_decoding_generation_lengths=spec_lengths,
        spec_decoding_position_offsets=spec_offsets,
        spec_decoding_packed_mask=spec_mask,
        spec_decoding_bl_tree_mask_offset=spec_tree_offsets,
        spec_decoding_bl_tree_mask=spec_tree_mask,
        spec_bl_tree_first_sparse_mask_offset_kv=spec_sparse_offsets,
        is_fused_qkv=not is_mla,
        update_kv_cache=True,
        use_paged_context_fmha=False,
        is_mla_enable=is_mla,
        is_cross=is_cross,
        attention_input_type=input_type,
        kv_lora_rank=4 if is_mla else None,
        qk_rope_head_dim=2 if is_mla else None,
        qk_nope_head_dim=4 if is_mla else None,
        v_head_dim=4 if is_mla else None,
        rope_append=True if is_mla else None,
        num_contexts=num_contexts,
        num_ctx_tokens=3 * num_contexts,
        sparse_attn_indices_block_size=1,
    )
    FallbackFmha.attention(**arguments)

    effective_beams = 1 if is_cross else beam_width
    sizing_contexts = num_contexts if has_context else 0
    expected = [
        ("workspace", sizing_contexts, num_ctx_tokens, 0, 0, effective_beams, sizing_contexts)
    ]
    if has_context:
        expected.append(
            ("context", num_contexts, num_ctx_tokens, 0, 0, effective_beams, num_contexts)
        )
    if num_generations:
        phase = "mla_generation" if is_mla else "generation"
        expected.append(
            (phase, 4, 4, num_contexts, num_ctx_tokens, effective_beams, 4 // effective_beams)
        )
    assert calls == expected
    if len(phase_params) == 2:
        assert phase_params[0] is not phase_params[1]


@pytest.mark.parametrize("is_cross", [False, True])
@pytest.mark.parametrize("beam_width", [1, 4])
@pytest.mark.parametrize(
    "input_type", [AttentionInputType.mixed, AttentionInputType.generation_only]
)
@pytest.mark.parametrize("use_spec_decoding", [False, True])
@pytest.mark.parametrize("is_spec_decoding_enabled", [False, True])
@pytest.mark.parametrize("predicted_tokens_per_seq", [1, 4])
@pytest.mark.parametrize("output_dtype", [torch.bfloat16, torch.uint8])
def test_fallback_native_generation_lengths_and_beams(
    monkeypatch: pytest.MonkeyPatch,
    is_cross: bool,
    beam_width: int,
    input_type: AttentionInputType,
    use_spec_decoding: bool,
    is_spec_decoding_enabled: bool,
    predicted_tokens_per_seq: int,
    output_dtype: torch.dtype,
) -> None:
    monkeypatch.setattr(TrtllmAttentionMetadata, "_post_init_with_buffers", lambda *args: None)
    metadata = TrtllmAttentionMetadata(
        max_num_requests=8,
        max_num_sequences=8,
        max_num_tokens=7,
        num_contexts=1,
        beam_width=beam_width,
        workspace=torch.empty(0, dtype=torch.uint8),
    )
    metadata.max_seq_len = 16
    metadata._seq_lens = torch.tensor([3, 1, 1, 1, 1], dtype=torch.int32)
    if is_cross:
        metadata._seq_lens_kv = torch.tensor([5, 10, 10, 10, 10], dtype=torch.int32)
    metadata.num_generations = 4
    metadata._num_ctx_tokens = 3
    metadata.prompt_lens_cuda_runtime = torch.tensor([3, 9, 9, 9, 9], dtype=torch.int32)
    metadata.prompt_lens_cpu_runtime = metadata.prompt_lens_cuda_runtime
    metadata.kv_lens_runtime = torch.tensor([5, 10, 10, 10, 10], dtype=torch.int32)
    metadata.kv_lens_cuda_runtime = metadata.kv_lens_runtime
    metadata.is_spec_decoding_enabled = is_spec_decoding_enabled
    metadata.use_spec_decoding = use_spec_decoding
    metadata.spec_decoding_generation_lengths = torch.full((4,), 4, dtype=torch.int32)
    metadata.spec_decoding_position_offsets = torch.zeros((8, 4), dtype=torch.int32)
    metadata.spec_decoding_packed_mask = torch.zeros((4,), dtype=torch.int32)
    metadata.spec_decoding_bl_tree_mask_offset = torch.zeros((4,), dtype=torch.int64)
    metadata.spec_decoding_bl_tree_mask = torch.zeros((4,), dtype=torch.uint32)
    metadata.spec_bl_tree_first_sparse_mask_offset_kv = torch.zeros((4,), dtype=torch.int32)
    spec_tensors = {
        name: getattr(metadata, name)
        for name in (
            "spec_decoding_generation_lengths",
            "spec_decoding_position_offsets",
            "spec_decoding_packed_mask",
            "spec_decoding_bl_tree_mask_offset",
            "spec_decoding_bl_tree_mask",
            "spec_bl_tree_first_sparse_mask_offset_kv",
        )
    }
    attn = FakeAttention()
    attn.predicted_tokens_per_seq = predicted_tokens_per_seq
    attn.layer_idx = 0
    attn.get_local_layer_idx = lambda _: 0
    attn.attention_chunk_size = None
    attn.rotary_inv_freq = attn.rotary_cos_sin = attn.rope_params = None
    fmha = FallbackFmha(attn)
    native_calls = []
    context_calls = []
    op = SimpleNamespace(run_context=context_calls.append, run_generation=native_calls.append)
    monkeypatch.setattr(fmha, "prepare_workspace", lambda *args: None)
    monkeypatch.setattr(fmha, "attention_op", lambda params: op)
    spec_active = is_spec_decoding_enabled and use_spec_decoding
    num_gen_tokens = 16 if spec_active else 4
    num_tokens = num_gen_tokens + (3 if input_type == AttentionInputType.mixed else 0)
    q = torch.empty((num_tokens, 12), dtype=torch.bfloat16)
    stored_head_size = 2 if output_dtype == torch.uint8 else 4
    forward_args = AttentionForwardArgs(
        output=torch.empty((num_tokens, stored_head_size), dtype=output_dtype),
        output_sf=torch.empty(16, dtype=torch.uint8) if output_dtype == torch.uint8 else None,
        attention_input_type=input_type,
        is_fused_qkv=True,
    )

    fmha.forward(q, None, None, metadata, forward_args)

    assert len(context_calls) == int(input_type == AttentionInputType.mixed)
    for context_params in context_calls:
        for name in spec_tensors:
            assert getattr(context_params, name) is None
    assert len(native_calls) == 1
    params = native_calls[0]
    assert params.context_lengths.tolist() == [9, 9, 9, 9]
    assert params.context_lengths.data_ptr() == metadata.prompt_lens_cuda_runtime[1:].data_ptr()
    assert params.beam_width == (1 if is_cross else beam_width)
    assert params.num_requests == (4 if is_cross else 4 // beam_width)
    assert params.num_seqs == params.num_requests * params.beam_width == 4
    assert metadata.beam_width == beam_width
    assert params.output.shape == (num_gen_tokens, 1, stored_head_size)
    token_offset = 3 if input_type == AttentionInputType.mixed else 0
    assert params.token_offset == token_offset
    assert params.output.data_ptr() == forward_args.output[token_offset:].data_ptr()
    if output_dtype == torch.uint8:
        assert params.fwd.output_sf.data_ptr() == forward_args.output_sf.data_ptr()
    for name, source_tensor in spec_tensors.items():
        native_tensor = getattr(params, name)
        if spec_active:
            assert native_tensor.data_ptr() == source_tensor.data_ptr()
        else:
            assert native_tensor is None
        assert getattr(metadata, name).data_ptr() == source_tensor.data_ptr()


@pytest.mark.parametrize("sm", [90, 100, 103, 120])
def test_phased_position_offsets_view_tracks_query_width_without_copy(
    monkeypatch: pytest.MonkeyPatch, sm: int
) -> None:
    monkeypatch.setattr(TrtllmAttentionMetadata, "_post_init_with_buffers", lambda *args: None)
    monkeypatch.setattr(
        "tensorrt_llm._torch.attention.backends.fmha.phased.get_sm_version", lambda: sm
    )
    metadata = TrtllmAttentionMetadata(
        max_num_requests=4,
        max_num_tokens=32,
        num_contexts=0,
        workspace=torch.empty(0, dtype=torch.uint8),
    )
    metadata.max_seq_len = 32
    metadata.num_generations = 2
    metadata._num_ctx_tokens = 0
    metadata._seq_lens = torch.ones(2, dtype=torch.int32)
    metadata.prompt_lens_cuda_runtime = torch.ones(2, dtype=torch.int32)
    metadata.prompt_lens_cpu_runtime = metadata.prompt_lens_cuda_runtime
    metadata.kv_lens_runtime = torch.full((2,), 16, dtype=torch.int32)
    metadata.kv_lens_cuda_runtime = metadata.kv_lens_runtime
    metadata.is_spec_decoding_enabled = metadata.use_spec_decoding = True
    metadata.spec_decoding_generation_lengths = torch.ones(4, dtype=torch.int32)
    offsets_buffer = torch.arange(32, dtype=torch.int32)
    metadata.spec_decoding_position_offsets = offsets_buffer
    attn = FakeAttention()
    fmha = PhasedFmha(attn)
    monkeypatch.setattr(fmha, "REQUIRES_PAGED_KV", False)
    monkeypatch.setattr(fmha, "NEEDS_BLOCK_EXTENT", False)
    views = []
    monkeypatch.setattr(
        fmha, "run_generation", lambda params: views.append(params.spec_decoding_position_offsets)
    )

    for query_len in (3, 5, 3):
        metadata.spec_decoding_query_len = query_len
        metadata.spec_decoding_generation_lengths.fill_(query_len)
        metadata._seq_lens.fill_(query_len)
        q = torch.empty((2 * query_len, 12), dtype=torch.bfloat16)
        fmha.forward(
            q,
            None,
            None,
            metadata,
            AttentionForwardArgs(
                output=torch.empty((2 * query_len, 4), dtype=torch.bfloat16),
                attention_input_type=AttentionInputType.generation_only,
                is_fused_qkv=True,
            ),
        )

        view = views[-1]
        expected_width = 8 if sm in (100, 103) else query_len
        assert view.shape == (4, expected_width)
        assert view.data_ptr() == offsets_buffer.data_ptr()
        assert view.untyped_storage().data_ptr() == offsets_buffer.untyped_storage().data_ptr()
        offsets_buffer[expected_width] += 100
        assert view[1, 0] == offsets_buffer[expected_width]
        assert metadata.spec_decoding_position_offsets is offsets_buffer
        assert offsets_buffer.shape == (32,)


@pytest.mark.parametrize("initial_fused_qkv", [False, True])
@pytest.mark.parametrize(
    "input_type,sparse",
    [
        (AttentionInputType.context_only, False),
        (AttentionInputType.context_only, True),
        (AttentionInputType.generation_only, False),
    ],
)
def test_mla_forward_clears_fused_qkv_before_fmha_selection(
    monkeypatch: pytest.MonkeyPatch,
    input_type: AttentionInputType,
    sparse: bool,
    initial_fused_qkv: bool,
) -> None:
    sparse_params = SparseRuntimeParams(
        sparse_attn_indices=torch.zeros(1, dtype=torch.int32) if sparse else None
    )
    monkeypatch.setattr(
        "tensorrt_llm._torch.attention.backends.trtllm.prepare_sparse_runtime_params",
        lambda *args: sparse_params,
    )
    attn = Mock(spec=TrtllmAttention)
    attn.sparse_params = None
    attn.is_mla_enable = True
    attn.num_heads = 2
    attn.kv_lora_rank = 8
    attn.qk_nope_head_dim = 4
    attn.qk_rope_head_dim = 2
    attn.print_skip_softmax_stat = False
    attn.kv_scale_orig_quant = None
    attn.kv_scale_quant_orig = None
    attn.get_local_layer_idx.return_value = 0
    attn._ensure_rope_table_size = Mock()
    fmha = Mock()
    attn._fmha_manager = Mock()
    attn._fmha_manager.select.return_value = fmha

    metadata = Mock(spec=TrtllmAttentionMetadata)
    metadata.is_cross = False
    metadata.enable_flash_mla = False
    metadata.spec_bl_tree_first_sparse_mask_offset_kv = None
    metadata.spec_decoding_bl_tree_mask = None
    metadata.max_context_q_len_override = None
    metadata.kv_cache_manager = None
    lengths = torch.ones(1, dtype=torch.int32)
    metadata.kv_lens_cuda_runtime = lengths
    metadata.kv_lens_runtime = lengths
    metadata.prompt_lens_cuda_runtime = lengths
    metadata.prompt_lens_cpu_runtime = lengths
    metadata.host_request_types_runtime = lengths
    metadata.max_seq_len = 8

    has_kv = input_type == AttentionInputType.context_only and not sparse
    q_head_dim = (attn.qk_nope_head_dim if has_kv else attn.kv_lora_rank) + attn.qk_rope_head_dim
    q = torch.empty((1, attn.num_heads * q_head_dim))
    k = torch.empty_like(q) if has_kv else None
    v = torch.empty_like(q) if has_kv else None
    forward_args = AttentionForwardArgs(
        output=torch.empty_like(q),
        attention_input_type=input_type,
        is_fused_qkv=initial_fused_qkv,
    )

    output = TrtllmAttention.forward(attn, q, k, v, metadata, forward_args)

    attn._fmha_manager.select.assert_called_once_with(attn, q, k, v, metadata, forward_args)
    fmha.forward.assert_called_once_with(q, k, v, metadata, forward_args)
    assert not forward_args.is_fused_qkv
    assert forward_args.update_kv_cache
    assert output is forward_args.output


@pytest.mark.parametrize("fmha_cls", FMHA_LIBS.values(), ids=FMHA_LIBS.keys())
@pytest.mark.parametrize("threshold", [0.0, 0.1])
@pytest.mark.parametrize("implementation_available", [False, True])
def test_availability_checks_capabilities_before_implementation(
    fmha_cls: type[Fmha],
    threshold: float,
    implementation_available: bool,
) -> None:
    attn = cast(TrtllmAttention, FakeAttention())
    attn.skip_correction_threshold = threshold
    capability_supported = threshold == 0.0 or fmha_cls is FallbackFmha

    with patch.object(fmha_cls, "_is_available", return_value=implementation_available) as hook:
        assert fmha_cls.is_available(attn) is (capability_supported and implementation_available)
        if capability_supported:
            hook.assert_called_once_with(attn)
        else:
            hook.assert_not_called()


@pytest.mark.parametrize("phase", [None, FmhaPhase.CONTEXT, FmhaPhase.GENERATION])
@pytest.mark.parametrize("supported", [False, True])
def test_support_forwards_request_and_phase(phase: FmhaPhase | None, supported: bool) -> None:
    attn = cast(TrtllmAttention, FakeAttention())
    fmha = _MinimalFmha(attn)
    q, k, v = (torch.empty((2, 4)) for _ in range(3))
    metadata = Mock(spec=TrtllmAttentionMetadata)
    forward_args = AttentionForwardArgs()

    with patch.object(fmha, "_is_supported", return_value=supported) as hook:
        assert fmha.is_supported(q, k, v, metadata, forward_args, phase=phase) is supported
        hook.assert_called_once_with(q, k, v, metadata, forward_args, phase=phase)


def test_default_hooks_accept_requests() -> None:
    attn = cast(TrtllmAttention, FakeAttention())
    fmha = _MinimalFmha(attn)

    assert _MinimalFmha.is_available(attn)
    assert fmha.is_supported(
        torch.empty((2, 4)), None, None, Mock(spec=TrtllmAttentionMetadata), AttentionForwardArgs()
    )


def test_inherited_availability_hook_uses_subclass_capabilities() -> None:
    class _SkipCorrectionFmha(_MinimalFmha):
        supports_skip_correction = True

        @classmethod
        def _is_available(cls, attn: TrtllmAttention) -> bool:
            return super()._is_available(attn)

    attn = cast(TrtllmAttention, FakeAttention())
    attn.skip_correction_threshold = 0.1

    assert not _MinimalFmha.is_available(attn)
    assert _SkipCorrectionFmha.is_available(attn)


@pytest.mark.parametrize("has_block_sparse_inputs", [False, True])
@pytest.mark.parametrize("supported", [False, True])
def test_support_checks_block_sparse_capability_before_implementation(
    has_block_sparse_inputs: bool, supported: bool
) -> None:
    class _BlockSparseFmha(_MinimalFmha):
        supports_block_sparse_inputs = True

    attn = cast(TrtllmAttention, FakeAttention())
    q, k, v = (torch.empty((2, 4)) for _ in range(3))
    metadata = Mock(spec=TrtllmAttentionMetadata)
    forward_args = AttentionForwardArgs()
    if has_block_sparse_inputs:
        forward_args.sparse_runtime_params = SparseRuntimeParams(
            block_sparse_inputs=BlockSparseForwardInputs(
                q_block_size=64,
                kv_block_size=64,
                exact_block_bits=torch.zeros((1, 1, 1, 1), dtype=torch.int32),
            )
        )

    for fmha in (_MinimalFmha(attn), _BlockSparseFmha(attn)):
        capability_supported = not has_block_sparse_inputs or fmha.supports_block_sparse_inputs
        with patch.object(fmha, "_is_supported", return_value=supported) as hook:
            assert fmha.is_supported(q, k, v, metadata, forward_args) is (
                capability_supported and supported
            )
            if capability_supported:
                hook.assert_called_once_with(q, k, v, metadata, forward_args, phase=None)
            else:
                hook.assert_not_called()
