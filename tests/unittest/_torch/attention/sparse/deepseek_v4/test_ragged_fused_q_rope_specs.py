# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import MethodType, SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.attention.backends.sparse.deepseek_v4.metadata import (
    DeepseekV4TrtllmAttentionMetadata,
)
from tensorrt_llm._torch.attention.backends.sparse.deepseek_v4.module import _fused_q_rope_specs


class _Mqa:
    def __init__(self):
        self.rotary_cos_sin = object()

    def _ensure_rope_table_size(self, size):
        pass


def _ragged_metadata(verify_lens, *, context_lens=(), kv_lens=None):
    num_generations = len(verify_lens)
    if kv_lens is None:
        kv_lens = [1000 + index + length for index, length in enumerate(verify_lens)]
    metadata = SimpleNamespace(
        kv_lens_cuda_runtime=torch.tensor(kv_lens, dtype=torch.int32),
        num_ctx_tokens=sum(context_lens),
        num_tokens=sum(context_lens) + sum(verify_lens),
        max_seq_len=max(kv_lens),
        is_ragged_verify=True,
        num_contexts=len(context_lens),
        num_generations=num_generations,
        seq_lens_cuda=torch.tensor([*context_lens, *verify_lens], dtype=torch.int32),
        fused_q_gen_cu_seqlens=torch.zeros(num_generations + 1, dtype=torch.int32),
        _fused_q_gen_cu_seqlens_valid=False,
    )
    metadata.mla_prepare_fused_q_gen_cu_seqlens = MethodType(
        DeepseekV4TrtllmAttentionMetadata.mla_prepare_fused_q_gen_cu_seqlens, metadata
    )
    return metadata


@pytest.mark.parametrize(
    "verify_lens",
    ([3] * 64 + [5] * 64, [4] * 64 + [6] * 64, [5] * 64 + [6] * 64),
)
def test_heterogeneous_g128_generation_contract(verify_lens):
    mla = SimpleNamespace(mqa=_Mqa())
    metadata = _ragged_metadata(verify_lens)
    cos_sin, specs = _fused_q_rope_specs(mla, metadata, 0, len(verify_lens))

    assert cos_sin is mla.mqa.rotary_cos_sin
    assert len(specs) == 1
    rows, cache_lens, seq_len, cu_q_seqlens = specs[0]
    assert (rows.start, rows.stop) == (0, sum(verify_lens))
    torch.testing.assert_close(cache_lens, metadata.kv_lens_cuda_runtime)
    assert seq_len == 0
    assert cu_q_seqlens.tolist() == [0, *torch.tensor(verify_lens).cumsum(0).tolist()]


def test_uniform_k5_keeps_scalar_generation_spec():
    metadata = SimpleNamespace(
        kv_lens_cuda_runtime=torch.full((128,), 1006, dtype=torch.int32),
        num_ctx_tokens=0,
        num_tokens=768,
        max_seq_len=1006,
        is_ragged_verify=False,
    )
    _, specs = _fused_q_rope_specs(SimpleNamespace(mqa=_Mqa()), metadata, 0, 128)
    assert len(specs) == 1
    rows, cache_lens, seq_len, cu_q_seqlens = specs[0]
    assert (rows.start, rows.stop) == (0, 768)
    assert cache_lens.tolist() == [1006] * 128
    assert seq_len == 6
    assert cu_q_seqlens is None


def test_mixed_batch_uses_disjoint_context_and_generation_prefixes():
    ctx_prefix = torch.tensor([0, 2, 5], dtype=torch.int32)
    metadata = _ragged_metadata([2, 1, 3], context_lens=[2, 3], kv_lens=[102, 203, 302, 401, 503])
    metadata.mla_prepare_ctx_cu_seqlens = lambda: ctx_prefix
    _, specs = _fused_q_rope_specs(SimpleNamespace(mqa=_Mqa()), metadata, 2, 3)
    assert len(specs) == 2
    ctx_rows, ctx_cache, ctx_seq_len, ctx_cu_q = specs[0]
    gen_rows, gen_cache, gen_seq_len, gen_cu_q = specs[1]
    assert (ctx_rows.start, ctx_rows.stop) == (0, 5)
    assert (gen_rows.start, gen_rows.stop) == (5, 11)
    assert ctx_cache.tolist() == [102, 203]
    assert gen_cache.tolist() == [302, 401, 503]
    assert (ctx_seq_len, gen_seq_len) == (0, 0)
    assert ctx_cu_q is ctx_prefix
    assert gen_cu_q.tolist() == [0, 2, 3, 6]
    assert ctx_cu_q.data_ptr() != gen_cu_q.data_ptr()


def test_context_only_keeps_its_original_prefix_boundary():
    ctx_prefix = torch.tensor([0, 2, 5], dtype=torch.int32)
    metadata = SimpleNamespace(
        kv_lens_cuda_runtime=torch.tensor([102, 203], dtype=torch.int32),
        num_ctx_tokens=5,
        num_tokens=5,
        max_seq_len=203,
        mla_prepare_ctx_cu_seqlens=lambda: ctx_prefix,
    )
    _, specs = _fused_q_rope_specs(SimpleNamespace(mqa=_Mqa()), metadata, 2, 0)
    assert len(specs) == 1
    rows, cache_lens, seq_len, cu_q_seqlens = specs[0]
    assert (rows.start, rows.stop) == (0, 5)
    assert cache_lens.tolist() == [102, 203]
    assert seq_len == 0
    assert cu_q_seqlens is ctx_prefix


def test_ragged_generation_fails_closed_without_stable_prefix_hook():
    metadata = SimpleNamespace(
        kv_lens_cuda_runtime=torch.tensor([10, 20], dtype=torch.int32),
        num_ctx_tokens=0,
        num_tokens=3,
        max_seq_len=20,
        is_ragged_verify=True,
    )
    mla = SimpleNamespace(mqa=_Mqa())
    assert _fused_q_rope_specs(mla, metadata, 0, 2) == (None, [])
    metadata.mla_prepare_fused_q_gen_cu_seqlens = lambda: None
    assert _fused_q_rope_specs(mla, metadata, 0, 2) == (None, [])


def test_prefix_storage_address_is_fixed_across_rebuilds():
    metadata = _ragged_metadata([2, 1, 3])
    first = metadata.mla_prepare_fused_q_gen_cu_seqlens()
    first_address = first.data_ptr()
    assert first.tolist() == [0, 2, 3, 6]

    metadata.seq_lens_cuda.copy_(torch.tensor([1, 4, 1], dtype=torch.int32))
    metadata._fused_q_gen_cu_seqlens_valid = False
    second = metadata.mla_prepare_fused_q_gen_cu_seqlens()
    assert second.data_ptr() == first_address
    assert second.tolist() == [0, 1, 5, 6]


@pytest.mark.parametrize("capacity", [None, 2])
def test_prefix_builder_fails_closed_without_graph_storage(capacity):
    metadata = _ragged_metadata([1, 2])
    metadata.fused_q_gen_cu_seqlens = (
        None if capacity is None else torch.zeros(capacity, dtype=torch.int32)
    )
    assert metadata.mla_prepare_fused_q_gen_cu_seqlens() is None


def test_sequence_update_invalidates_cached_prefix():
    metadata = DeepseekV4TrtllmAttentionMetadata.__new__(DeepseekV4TrtllmAttentionMetadata)
    metadata._fused_q_gen_cu_seqlens_valid = True
    metadata._invalidate_mla_scheduler_buffers()
    assert not metadata._fused_q_gen_cu_seqlens_valid
