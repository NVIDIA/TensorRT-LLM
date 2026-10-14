# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU coverage of ragged preparation, update and backend dispatch contracts."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from tensorrt_llm._torch.attention.backends.interface import AttentionInputType
from tensorrt_llm._torch.attention.backends.sparse.deepseek_v4.module import (
    forward_generation_sparse_attn,
)
from tensorrt_llm._torch.attention.backends.sparse.dsa.indexer import Indexer
from tensorrt_llm._torch.attention.backends.sparse.dsa.metadata import DSAtrtllmAttentionMetadata
from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttention, TrtllmAttentionMetadata

pytestmark = pytest.mark.cpu_only


def _metadata(seq_lens: list[int], num_contexts: int = 0) -> DSAtrtllmAttentionMetadata:
    metadata = DSAtrtllmAttentionMetadata.__new__(DSAtrtllmAttentionMetadata)
    metadata._seq_lens = torch.tensor(seq_lens, dtype=torch.int32)
    metadata._seq_lens_cuda = metadata._seq_lens.clone()
    metadata._num_contexts = num_contexts
    metadata._num_generations = len(seq_lens) - num_contexts
    metadata._num_ctx_tokens = sum(seq_lens[:num_contexts])
    metadata._num_tokens = sum(seq_lens)
    metadata.max_draft_tokens = 5
    metadata.runtime_tokens_per_gen_step = 0
    metadata._indexer_compress_ratio = 1
    metadata.kv_cache_manager = None
    metadata.num_sms = 1
    return metadata


@pytest.mark.parametrize("runtime_width", [1, 2, 6])
def test_schedule_update_keeps_runtime_query_width(monkeypatch, runtime_width: int) -> None:
    metadata = _metadata([runtime_width, runtime_width])
    metadata.runtime_tokens_per_gen_step = runtime_width
    metadata.kv_lens_cuda = torch.tensor([102, 204], dtype=torch.int32)
    metadata.kv_lens_cuda_runtime = metadata.kv_lens_cuda
    metadata.kv_lens_cuda_2d = torch.full((2, 6), -1, dtype=torch.int32)
    metadata.scheduler_metadata_buffer = torch.empty(1, dtype=torch.int32)
    metadata.scheduler_metadata_buffer_full_next_n = torch.empty(1, dtype=torch.int32)
    metadata.req_idx_per_token = torch.empty(metadata.num_tokens, dtype=torch.int32)
    metadata.gen_kv_indptr = torch.zeros(3, dtype=torch.int64)
    metadata.gen_cached_token_indptr = torch.zeros(3, dtype=torch.int64)
    metadata.enable_ragged_verification = False
    metadata.enable_flash_mla = False
    metadata.use_fp8_ds_mla = False
    metadata.use_expanded_buffers_for_mtp = False
    metadata.expand_for_dsl = False
    metadata._invalidate_pool_view_cache = Mock()
    metadata._compute_kv_lens_row_reorder = Mock()
    metadata.prepare_dense_topk_indices = Mock()
    scheduler_inputs = []

    def schedule(context_lens, block_kv, num_sms):
        assert context_lens.is_contiguous()
        scheduler_inputs.append(context_lens.clone())
        return torch.zeros(1, dtype=torch.int32)

    monkeypatch.setattr(
        "tensorrt_llm._torch.attention.backends.sparse.dsa.indexer.get_paged_mqa_logits_metadata",
        schedule,
    )
    monkeypatch.setattr(
        "tensorrt_llm._torch.attention.backends.sparse.dsa.metadata.get_paged_mqa_logits_metadata",
        schedule,
    )
    Indexer.prepare_scheduler_metadata(metadata)
    metadata.kv_lens_cuda.add_(1)
    metadata.on_update_kv_lens()

    assert [tuple(value.shape) for value in scheduler_inputs] == [
        (2, 1),
        (2, runtime_width),
        (2, 1),
        (2, runtime_width),
    ]
    torch.testing.assert_close(
        scheduler_inputs[-1],
        metadata.kv_lens_cuda[:, None].expand(-1, runtime_width),
    )


@pytest.mark.parametrize("num_contexts", [0, 1])
def test_prepare_initializes_row_extents_and_refresh_reuses_storage(
    monkeypatch, num_contexts: int
) -> None:
    metadata = _metadata(([3] if num_contexts else []) + [2, 4], num_contexts)
    metadata.ragged_verify_lens = [2, 4]
    metadata.enable_ragged_verification = True
    metadata.is_spec_decoding_enabled = False
    metadata.sparse_metadata_params = SimpleNamespace(use_cute_dsl_paged_mqa_logits=False)
    metadata.gen_token_repeats_cuda = torch.empty(2, dtype=torch.int64)
    metadata.host_gen_token_repeats = torch.empty(2, dtype=torch.int64)
    metadata.kv_lens_expanded_host = torch.empty(6, dtype=torch.int32)
    metadata.kv_lens_expanded_cuda = torch.empty(6, dtype=torch.int32)
    for name in ("row_kv_lens", "row_kv_correction", "row_req_idx"):
        dtype = torch.long if name == "row_req_idx" else torch.int32
        for suffix in ("host", "cuda"):
            setattr(metadata, f"{name}_{suffix}", torch.full((6,), -777, dtype=dtype))
    for name in ("attn_row_kv_lens", "attn_row_kv_correction", "attn_row_req_idx"):
        dtype = torch.long if name == "attn_row_req_idx" else torch.int32
        for suffix in ("host", "cuda"):
            setattr(metadata, f"{name}_{suffix}", torch.full((7,), -777, dtype=dtype))
    metadata.prompt_lens_cpu = torch.tensor(([3] if num_contexts else []) + [100, 200])
    metadata.attn_row_prompt_lens_cpu = torch.empty(7, dtype=torch.int32)
    metadata.attn_row_prompt_lens_cuda = torch.empty(7, dtype=torch.int32)
    metadata.attn_row_request_types_host = torch.empty(7, dtype=torch.int32)
    metadata.attn_row_block_offsets = torch.empty(1, 7, 2, 1, dtype=torch.int32)
    metadata.kv_cache_block_offsets = None
    monkeypatch.setattr(
        "tensorrt_llm._torch.attention.backends.sparse.dsa.metadata.get_sm_version",
        lambda: 100,
    )
    kv_lens = torch.tensor(([3] if num_contexts else []) + [102, 204], dtype=torch.int32)
    metadata.prepare_for_spec_decode(kv_lens)

    expected = torch.tensor([101, 102, 201, 202, 203, 204], dtype=torch.int32)
    torch.testing.assert_close(metadata.ragged_row_kv_lens(6), expected)
    view = metadata.token_major_gen_view()
    torch.testing.assert_close(view.sequence_length[num_contexts:], expected)
    if num_contexts:
        assert view.sequence_length[0] == 3
    row_ptr = metadata.row_kv_lens_cuda.data_ptr()
    attention_ptr = metadata.attn_row_kv_lens_cuda.data_ptr()
    metadata.kv_lens_cuda = kv_lens.clone()
    metadata.kv_lens_cuda[num_contexts:] -= torch.tensor([3, 6])
    metadata.refresh_ragged_row_kv_lens()
    metadata.refresh_token_major_gen_rows()

    expected -= torch.tensor([3, 3, 6, 6, 6, 6])
    torch.testing.assert_close(metadata.ragged_row_kv_lens(6), expected)
    torch.testing.assert_close(view.sequence_length[num_contexts:], expected)
    assert metadata.row_kv_lens_cuda.data_ptr() == row_ptr
    assert metadata.attn_row_kv_lens_cuda.data_ptr() == attention_ptr
    assert metadata.num_seqs == num_contexts + 2


class _StopBeforeDispatch(Exception):
    pass


def _prefix_metadata(monkeypatch, num_contexts=0, ragged=True):
    seq_lens = ([3] if num_contexts else []) + [1, 2, 4]
    metadata = _metadata(seq_lens, num_contexts)
    metadata.enable_ragged_verification = ragged
    metadata.ragged_verify_lens = [1, 2, 4] if ragged else None
    metadata.max_num_sequences = len(seq_lens)
    metadata.cuda_graph_buffers = {}
    metadata.draft_kv_cache_manager = None
    metadata.kv_cache_manager = SimpleNamespace(
        num_pools=1, num_attention_op_pools=1, max_blocks_per_seq=1
    )
    allocation_names = []

    def allocate(buffers, shape, *, cache_name, dtype, capture_graph):
        allocation_names.append(cache_name)
        return torch.empty(shape, dtype=dtype)

    metadata.get_empty = allocate
    monkeypatch.setattr(
        "tensorrt_llm._torch.attention.backends.sparse.dsa.metadata.prefer_pinned",
        lambda: False,
    )
    monkeypatch.setattr(
        "tensorrt_llm._torch.attention.backends.sparse.dsa.metadata.is_dsa_cache_manager",
        lambda value: False,
    )
    metadata.create_expanded_buffers()
    metadata.kv_cache_manager = None
    metadata.mla_cu_q_rows = torch.zeros(len(seq_lens) + 1, dtype=torch.int32)
    metadata.mla_cu_kv_seqlens = torch.zeros(len(seq_lens) + 1, dtype=torch.int32)
    metadata.mla_ctx_cu_q_seqlens = torch.zeros(len(seq_lens) + 1, dtype=torch.int32)
    metadata._mla_ctx_cu_seqlens_valid = False
    metadata._mla_scheduler_buffers_valid = False
    metadata.prompt_lens_cpu = metadata.seq_lens.clone()
    metadata.kv_cache_block_offsets = None
    metadata.is_spec_decoding_enabled = False
    metadata.sparse_metadata_params = SimpleNamespace(use_cute_dsl_paged_mqa_logits=False)
    monkeypatch.setattr(
        "tensorrt_llm._torch.attention.backends.sparse.dsa.metadata.get_sm_version",
        lambda: 100,
    )
    kv_lens = torch.tensor(([17] if num_contexts else []) + [101, 202, 304], dtype=torch.int32)
    metadata.kv_lens_cuda = kv_lens.clone()
    # Reserved extra tokens must not leak into the kernel's causal KV scan.
    metadata.kv_lens = kv_lens + 9
    if ragged:
        metadata.prepare_for_spec_decode(kv_lens)
    return metadata, allocation_names


def test_feature_off_keeps_request_major_prefixes_and_allocations(monkeypatch):
    metadata, names = _prefix_metadata(monkeypatch, ragged=False)
    q_prefix, kv_prefix = metadata.mla_prepare_scheduler_buffers(2)
    assert "ragged_mla_cu_q_rows" not in names
    assert "ragged_mla_cu_kv_seqlens" not in names
    assert q_prefix.data_ptr() == metadata.mla_cu_q_rows.data_ptr()
    assert kv_prefix.data_ptr() == metadata.mla_cu_kv_seqlens.data_ptr()
    torch.testing.assert_close(q_prefix, torch.tensor([0, 2, 6, 14], dtype=torch.int32))
    torch.testing.assert_close(kv_prefix, torch.tensor([0, 101, 303, 607], dtype=torch.int32))


def test_uniform_step_with_opt_in_keeps_request_major_prefixes(monkeypatch):
    metadata, _ = _prefix_metadata(monkeypatch)
    metadata.ragged_verify_lens = None
    metadata._seq_lens.fill_(2)
    metadata._seq_lens_cuda.copy_(metadata._seq_lens)
    q_prefix, kv_prefix = metadata.mla_prepare_scheduler_buffers(2)
    assert q_prefix.data_ptr() == metadata.mla_cu_q_rows.data_ptr()
    torch.testing.assert_close(q_prefix, torch.tensor([0, 4, 8, 12], dtype=torch.int32))
    torch.testing.assert_close(kv_prefix, torch.tensor([0, 101, 303, 607], dtype=torch.int32))


@pytest.mark.parametrize("num_contexts", [0, 1])
def test_module_precomputes_causal_token_major_prefixes_and_refreshes(monkeypatch, num_contexts):
    metadata, names = _prefix_metadata(monkeypatch, num_contexts)
    if num_contexts:
        torch.testing.assert_close(
            metadata.mla_prepare_ctx_cu_seqlens(), torch.tensor([0, 3], dtype=torch.int32)
        )
    else:
        assert metadata.mla_prepare_ctx_cu_seqlens() is None
    observed = []

    def capture(*values, **controls):
        assert controls["precomputed_cu_seqlens"] is True
        observed.append(
            (values[4].clone(), values[5].clone(), values[4].data_ptr(), values[5].data_ptr())
        )
        raise _StopBeforeDispatch

    monkeypatch.setattr(
        "tensorrt_llm._torch.attention.backends.sparse.deepseek_v4.module._get_validated_sm_version",
        lambda: 100,
    )
    monkeypatch.setattr(
        "tensorrt_llm._torch.attention.backends.sparse.deepseek_v4.module._mla_gen_scheduler_scalars",
        lambda *unused: (None, None, None),
    )
    module = SimpleNamespace(
        num_heads_tp=2,
        qk_head_dim=4,
        qk_nope_head_dim=2,
        mqa=SimpleNamespace(has_fp8_kv_cache=False, mla_rope_generation=capture),
        kv_cache_dtype="bfloat16",
        kv_a_layernorm=SimpleNamespace(variance_epsilon=1e-6),
    )

    def launch():
        with pytest.raises(_StopBeforeDispatch):
            forward_generation_sparse_attn(
                module,
                torch.zeros(7, 8),
                torch.zeros(7, 2),
                torch.zeros(7, 2),
                metadata,
                torch.zeros(7, 8),
                latent_cache=torch.zeros(7, 4),
            )

    launch()
    expected_q = torch.arange(0, 15, 2, dtype=torch.int32)
    expected_kv = torch.tensor([101, 201, 202, 301, 302, 303, 304], dtype=torch.int32)
    torch.testing.assert_close(observed[-1][0], expected_q)
    torch.testing.assert_close(observed[-1][1][1:], expected_kv.cumsum(0, dtype=torch.int32))
    assert observed[-1][1][0] == 0
    assert names.count("ragged_mla_cu_q_rows") == 1
    assert names.count("ragged_mla_cu_kv_seqlens") == 1
    launch()
    assert observed[-1][2:] == observed[-2][2:]

    # Exercise the actual overlap hook: refresh device rows and invalidate the
    # cached prefix before the next module-level dispatch, without host refresh.
    metadata.kv_cache_manager = None
    metadata.enable_flash_mla = False
    metadata.use_fp8_ds_mla = False
    metadata.expand_for_dsl = False
    metadata.kv_lens_cuda_2d = torch.empty(metadata.num_generations, 6, dtype=torch.int32)
    metadata.scheduler_metadata_buffer = torch.empty(1, dtype=torch.int32)
    metadata.scheduler_metadata_buffer_expanded = torch.empty(1, dtype=torch.int32)
    metadata.req_idx_per_token = torch.empty(metadata.num_tokens, dtype=torch.int32)
    metadata.gen_kv_indptr = torch.zeros(4, dtype=torch.int64)
    metadata.gen_cached_token_indptr = torch.zeros(4, dtype=torch.int64)
    metadata._invalidate_pool_view_cache = Mock()
    metadata._compute_kv_lens_row_reorder = Mock()
    metadata.prepare_dense_topk_indices = Mock()
    monkeypatch.setattr(
        "tensorrt_llm._torch.attention.backends.sparse.dsa.metadata.get_paged_mqa_logits_metadata",
        lambda *unused: torch.zeros(1, dtype=torch.int32),
    )
    metadata.kv_lens_cuda[num_contexts:] -= torch.tensor([2, 3, 6])
    metadata.on_update_kv_lens()
    assert metadata._mla_scheduler_buffers_valid is False
    launch()
    expected_kv -= torch.tensor([2, 3, 3, 6, 6, 6, 6])
    torch.testing.assert_close(observed[-1][0], expected_q)
    torch.testing.assert_close(observed[-1][1][1:], expected_kv.cumsum(0, dtype=torch.int32))
    assert observed[-1][2:] == observed[-2][2:]
    assert metadata.num_seqs == num_contexts + 3


@pytest.mark.parametrize("metadata_valid", [False, True])
def test_flash_mla_rejects_token_major_generation_before_dispatch(metadata_valid: bool) -> None:
    metadata = TrtllmAttentionMetadata.__new__(TrtllmAttentionMetadata)
    metadata.enable_flash_mla = True
    metadata._flash_mla_metadata_valid = metadata_valid
    metadata.token_major_gen_view = Mock(return_value=object())
    attention = SimpleNamespace(
        sparse_params=None,
        is_mla_enable=False,
        create_output=Mock(side_effect=_StopBeforeDispatch),
    )
    metadata.use_paged_context_fmha = False

    with pytest.raises(ValueError, match="FlashMLA.*ragged"):
        TrtllmAttention.forward(
            attention,
            torch.empty(3, 8),
            None,
            None,
            metadata,
            attention_input_type=AttentionInputType.generation_only,
        )
    attention.create_output.assert_not_called()


@pytest.mark.parametrize("flash_mla,token_major", [(True, False), (False, True)])
def test_uniform_flash_mla_and_ragged_native_pass_the_guard(
    flash_mla: bool, token_major: bool
) -> None:
    metadata = TrtllmAttentionMetadata.__new__(TrtllmAttentionMetadata)
    metadata.enable_flash_mla = flash_mla
    metadata.token_major_gen_view = Mock(return_value=object() if token_major else None)
    metadata.use_paged_context_fmha = False
    attention = SimpleNamespace(
        sparse_params=None,
        is_mla_enable=False,
        create_output=Mock(side_effect=_StopBeforeDispatch),
    )
    with pytest.raises(_StopBeforeDispatch):
        TrtllmAttention.forward(
            attention,
            torch.empty(3, 8),
            None,
            None,
            metadata,
            attention_input_type=AttentionInputType.generation_only,
        )
    attention.create_output.assert_called_once()
