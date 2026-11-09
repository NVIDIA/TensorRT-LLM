# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""update_for_spec_dec() skips the indexer schedule rebuild only inside an index-sharing draft loop."""

import pytest
import torch
from utils.util import skip_pre_hopper

from tensorrt_llm._torch.attention.backends.sparse.dsa import (
    DSACacheManager,
    DSAtrtllmAttentionMetadata,
)
from tensorrt_llm._torch.metadata import KVCacheParams
from tensorrt_llm.bindings import DataType
from tensorrt_llm.bindings.executor import KvCacheConfig
from tensorrt_llm.bindings.internal.batch_manager import CacheType
from tensorrt_llm.llmapi.llm_args import DeepSeekSparseAttentionConfig
from tensorrt_llm.mapping import Mapping

HEAD_DIM = 128
TOKENS_PER_BLOCK = 64
MAX_SEQ_LEN = 4096
KV_LENS = [1000, 1500]
# One draft token so the full-next_n schedule is part of the rebuild.
MAX_DRAFT_TOKENS = 1


def _has_deep_gemm() -> bool:
    try:
        from tensorrt_llm import deep_gemm

        return deep_gemm is not None
    except Exception:
        return False


pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA"),
    pytest.mark.skipif(not _has_deep_gemm(), reason="DeepGEMM not available"),
    skip_pre_hopper,
]


@pytest.fixture
def decode_metadata():
    """A prepared one-token-per-request generation batch, as draft steps >= 1 see it.

    The cache manager is shut down on teardown: it owns CUDA reservations that a
    garbage-collector free would release on a thread without a current CUDA context.
    """
    batch_size = len(KV_LENS)
    sparse_config = DeepSeekSparseAttentionConfig(
        index_n_heads=32,
        index_head_dim=HEAD_DIM,
        index_topk=2048,
        indexer_k_dtype="fp8",
        skip_indexer_for_short_seqs=False,
        index_share_for_mtp_iteration=True,
    )
    mapping = Mapping(world_size=1, rank=0, tp_size=1, pp_size=1)
    kv_cache_manager = DSACacheManager(
        kv_cache_config=KvCacheConfig(
            enable_block_reuse=False, max_tokens=MAX_SEQ_LEN * batch_size
        ),
        kv_cache_type=CacheType.SELFKONLY,
        num_layers=1,
        num_kv_heads=1,
        head_dim=HEAD_DIM,
        tokens_per_block=TOKENS_PER_BLOCK,
        max_seq_len=MAX_SEQ_LEN,
        max_batch_size=batch_size,
        mapping=mapping,
        dtype=DataType.HALF,
        sparse_attention_config=sparse_config,
    )
    try:
        request_ids = list(range(batch_size))
        kv_cache_manager.add_dummy_requests(
            request_ids=request_ids, token_nums=KV_LENS, is_gen=False
        )
        cached = [kv_len - 1 for kv_len in KV_LENS]
        metadata = DSAtrtllmAttentionMetadata(
            seq_lens=torch.ones(batch_size, dtype=torch.int),
            request_ids=request_ids,
            max_num_requests=batch_size,
            num_contexts=0,
            prompt_lens=cached,
            max_num_tokens=batch_size * (1 + MAX_DRAFT_TOKENS),
            kv_cache_manager=kv_cache_manager,
            kv_cache_params=KVCacheParams(use_cache=True, num_cached_tokens_per_seq=cached),
            mapping=mapping,
            sparse_metadata_params=sparse_config.to_sparse_metadata_params(),
        )
        # The draft depth is set the way the engine sets it; this also sizes the
        # next_n-dependent buffers. Spec-dec attention stays off, as in draft steps >= 1.
        metadata.update_spec_dec_param(
            batch_size=batch_size,
            is_spec_decoding_enabled=False,
            is_spec_dec_tree=False,
            is_spec_dec_dynamic_tree=False,
            max_draft_len=MAX_DRAFT_TOKENS,
            max_total_draft_tokens=MAX_DRAFT_TOKENS,
        )
        metadata.prepare()
        torch.cuda.synchronize()
        assert metadata.sparse_metadata_params.mtp_index_share
        yield metadata
    finally:
        kv_cache_manager.shutdown()


def _schedule_buffers(metadata):
    """What on_update_kv_lens(skip_indexer_schedule=True) leaves untouched."""
    return [
        metadata.scheduler_metadata_buffer,
        metadata.scheduler_metadata_buffer_full_next_n,
        metadata.kv_lens_cuda_2d[: metadata.num_generations],
    ]


def _poison(metadata):
    for buf in _schedule_buffers(metadata):
        buf.fill_(-1)


def _is_poisoned(metadata):
    torch.cuda.synchronize()
    return [bool((buf == -1).all()) for buf in _schedule_buffers(metadata)]


def _kv_lens_buffers(metadata):
    """What on_update_kv_lens() refreshes from the KV lengths even when it skips the schedule."""
    num_tokens = metadata.num_tokens
    num_generations = metadata.num_generations
    return [
        metadata.slot_mapping_fp8[:num_tokens],
        metadata.slot_mapping_scale[:num_tokens],
        metadata.gen_kv_indptr[1 : num_generations + 1],
        metadata.gen_cached_token_indptr[1 : num_generations + 1],
    ]


def _snapshot(buffers):
    torch.cuda.synchronize()
    return [buf.clone() for buf in buffers]


def test_schedule_is_skipped_only_inside_the_draft_loop(decode_metadata):
    metadata = decode_metadata

    # Outside the draft loop: full rebuild.
    _poison(metadata)
    metadata.update_for_spec_dec()
    assert not metadata.indexer_schedule_stale
    assert _is_poisoned(metadata) == [False, False, False]
    before_advance = _snapshot(_kv_lens_buffers(metadata))

    # Inside the index-sharing draft loop, after a draft step appends a token: the schedule
    # is left as is and marked stale, but the KV-lens-dependent buffers are still rewritten.
    metadata.set_in_mtp_draft_loop(True)
    metadata.kv_lens_cuda[: metadata.num_generations] += 1
    _poison(metadata)
    for buf in _kv_lens_buffers(metadata):
        buf.fill_(-1)
    metadata.update_for_spec_dec()
    assert metadata.indexer_schedule_stale
    assert _is_poisoned(metadata) == [True, True, True]
    in_draft_loop = _snapshot(_kv_lens_buffers(metadata))

    # The next full rebuild (target forward) restores the schedule, clears the marker and
    # writes the same slot mappings and gen indptrs, which follow the advanced lengths.
    metadata.on_update_kv_lens()
    assert not metadata.indexer_schedule_stale
    assert _is_poisoned(metadata) == [False, False, False]
    full_rebuild = _snapshot(_kv_lens_buffers(metadata))
    for skipped, rebuilt, stale in zip(in_draft_loop, full_rebuild, before_advance):
        assert torch.equal(skipped, rebuilt)
        assert not torch.equal(skipped, stale)
