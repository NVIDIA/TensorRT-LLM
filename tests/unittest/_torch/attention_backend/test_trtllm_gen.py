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

from types import SimpleNamespace

import torch

import tensorrt_llm._torch.attention_backend.trtllm_gen as trtllm_gen
from tensorrt_llm._torch.attention_backend.interface import AttentionForwardArgs, AttentionInputType
from tensorrt_llm.bindings import DataType


def _patch_blackwell(monkeypatch):
    monkeypatch.setattr(trtllm_gen, "IS_FLASHINFER_AVAILABLE", True)
    monkeypatch.setattr(trtllm_gen, "get_sm_version", lambda: 100)
    monkeypatch.setattr(trtllm_gen, "is_sm_100f", lambda sm: sm in (100, 103))


def _make_attn(**overrides):
    fields = dict(
        is_mla_enable=False,
        skip_softmax_threshold_scale_factor_prefill=None,
        skip_softmax_threshold_scale_factor_decode=None,
        position_embedding_type=0,
        num_heads=8,
        num_kv_heads=8,
        head_dim=64,
        kv_lora_rank=None,
        qk_rope_head_dim=None,
        attention_chunk_size=None,
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


def _make_meta(**overrides):
    fields = dict(
        is_cross=True,
        is_spec_decoding_enabled=False,
        use_spec_decoding=False,
        is_spec_dec_tree=False,
        kv_cache_block_offsets=torch.empty((1, 1, 1), dtype=torch.int32),
        kv_cache_manager=SimpleNamespace(dtype=DataType.BF16),
        tokens_per_block=64,
        beam_width=1,
        effective_beam_width=1,
        helix_position_offsets=None,
        num_sparse_topk=0,
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


def _make_forward_args(**overrides):
    fields = dict(
        output=torch.empty((2, 8, 64), dtype=torch.bfloat16),
        attention_input_type=AttentionInputType.context_only,
    )
    fields.update(overrides)
    return AttentionForwardArgs(**fields)


def _check_support(monkeypatch, *, attn=None, meta=None, fwd=None, q_dtype=torch.bfloat16):
    _patch_blackwell(monkeypatch)
    backend = object.__new__(trtllm_gen.FlashInferTrtllmGenAttention)
    q = torch.empty((2, 8 * 64), dtype=q_dtype)
    return backend.is_supported(
        q,
        None,
        None,
        attn=attn or _make_attn(),
        meta=meta or _make_meta(),
        fwd=fwd or _make_forward_args(),
    )


def test_is_supported_allows_cross_attention_on_blackwell(monkeypatch):
    supported, reason = _check_support(monkeypatch)

    assert supported, reason


def test_is_supported_rejects_cross_attention_mla(monkeypatch):
    supported, reason = _check_support(
        monkeypatch,
        attn=_make_attn(is_mla_enable=True),
        fwd=_make_forward_args(attention_input_type=AttentionInputType.generation_only),
    )

    assert not supported
    assert "Cross attention with MLA" in reason


def test_is_supported_rejects_cross_attention_nvfp4(monkeypatch):
    supported, reason = _check_support(
        monkeypatch,
        meta=_make_meta(kv_cache_manager=SimpleNamespace(dtype=DataType.NVFP4)),
    )

    assert not supported
    assert "Cross attention with NVFP4" in reason


def test_is_supported_rejects_cross_attention_spec_decoding(monkeypatch):
    supported, reason = _check_support(
        monkeypatch,
        meta=_make_meta(is_spec_decoding_enabled=True, use_spec_decoding=True),
    )

    assert not supported
    assert "Cross attention with speculative decoding" in reason


def test_is_supported_rejects_relative_attention_bias(monkeypatch):
    supported, reason = _check_support(
        monkeypatch,
        fwd=_make_forward_args(relative_attention_bias=torch.empty((1, 1, 1, 1))),
    )

    assert not supported
    assert "Relative attention bias" in reason


def test_is_supported_uses_effective_beam_width_for_cross_attention(monkeypatch):
    supported, reason = _check_support(
        monkeypatch,
        meta=_make_meta(beam_width=4, effective_beam_width=1),
        fwd=_make_forward_args(attention_input_type=AttentionInputType.generation_only),
    )

    assert supported, reason


def test_cross_attention_context_uses_fused_preprocess(monkeypatch):
    preprocess_calls = []
    context_calls = []
    postprocess_called = False

    def fake_context_preprocess(*args):
        preprocess_calls.append(args)
        return (
            torch.empty((2, 2, 4), dtype=torch.bfloat16),
            torch.empty((1,), dtype=torch.bfloat16),
            torch.empty((1, 1), dtype=torch.int32),
            None,
            None,
            None,
            torch.empty((1,), dtype=torch.uint8),
            torch.tensor([0, 2], dtype=torch.int32),
            torch.tensor([0, 3], dtype=torch.int32),
            2,
            3,
            -1,
        )

    def fake_context_attention(*args):
        context_calls.append(args)

    def fake_context_postprocess(*args):
        nonlocal postprocess_called
        postprocess_called = True

    monkeypatch.setattr(trtllm_gen.thop, "trtllm_gen_context_preprocess", fake_context_preprocess)
    monkeypatch.setattr(trtllm_gen.thop, "trtllm_gen_context_postprocess", fake_context_postprocess)
    monkeypatch.setattr(
        trtllm_gen, "_trtllm_gen_batch_context_with_kv_cache", fake_context_attention
    )

    backend = object.__new__(trtllm_gen.FlashInferTrtllmGenAttention)
    backend._multi_processor_count = 1
    backend._enable_pdl = False

    cross_kv = torch.empty((3, 16), dtype=torch.bfloat16)
    attn = SimpleNamespace(
        rope_params=SimpleNamespace(dim=0, theta=10000.0, scale_type=0, scale=1.0, max_positions=0),
        rotary_inv_freq=None,
        rotary_cos_sin=None,
        local_layer_idx=0,
        num_heads=2,
        num_kv_heads=2,
        head_dim=4,
        quant_mode=0,
        q_scaling=1.0,
        position_embedding_type=0,
        is_mla_enable=False,
        attention_chunk_size=None,
    )
    meta = SimpleNamespace(
        kv_cache_block_offsets=torch.empty((1, 1, 1), dtype=torch.int32),
        host_kv_cache_pool_pointers=torch.empty((1,), dtype=torch.int64),
        host_kv_cache_pool_mapping=torch.empty((1,), dtype=torch.int32),
        use_paged_context_fmha=False,
    )
    fwd = AttentionForwardArgs(cross_kv=cross_kv, update_kv_cache=True)
    params = trtllm_gen.FmhaParams(
        attn=attn,
        meta=meta,
        fwd=fwd,
        workspace=torch.empty((1,), dtype=torch.uint8),
        qkv_input=torch.empty((2, 24), dtype=torch.bfloat16),
        context_buf=torch.empty((2, 2, 4), dtype=torch.bfloat16),
        sequence_lengths=torch.tensor([3], dtype=torch.int32),
        context_lengths=torch.tensor([2], dtype=torch.int32),
        input_seq_length=2,
        max_past_kv_length=3,
        max_attention_window_size=3,
        cyclic_attention_window_size=3,
        num_tokens=2,
        tokens_per_block=64,
        kv_factor=2,
        total_num_blocks=1,
        batch_size=1,
        cross_attention=True,
    )

    backend.run_context(params)

    assert preprocess_calls[0][-2] is cross_kv
    assert preprocess_calls[0][-1] is True
    assert context_calls[0][-1] is False
    assert not postprocess_called
