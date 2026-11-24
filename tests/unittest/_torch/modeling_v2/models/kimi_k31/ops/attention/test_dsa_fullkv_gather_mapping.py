# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the full-KV gather mapping of the K3.1 DSA indexer.

``K31Indexer.recompute_context_kv_gather_mappings`` maps every cached and new index-key token of each context request to
its flat byte offset in the INDEX_KEY pool. It builds the mapping on the device from the device block-offset table;
this test checks it bit for bit against the same arithmetic run on the CPU over the host table, at 1 to 8 requests and
64k to 200k cached tokens, and checks the cases that keep the regular mappings.
"""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.attention.dsa_indexer import (
    K31Indexer,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.attention.dsa_indexer_plan import (
    _compute_slot_mappings,
)

assert torch.cuda.is_available(), "the full-KV gather mapping is built on a CUDA device"

TOKENS_PER_BLOCK = 64
HEAD_DIM = 128
QUANT_BLOCK = 128
MAX_BLOCKS_PER_SEQ = 4096  # 256k tokens at 64 tokens per block
MAX_SEQS = 16


def _cpu_reference(params, host_block_offsets):
    """The mapping computed on the CPU from the host table, token by token."""
    kv_lens = params.kv_lens[: params.num_contexts]
    total = int(kv_lens.sum())
    req_indices = torch.repeat_interleave(
        torch.arange(params.num_contexts, dtype=torch.int64), kv_lens
    )
    cu_kv = torch.zeros(params.num_contexts + 1, dtype=torch.int64)
    cu_kv[1:] = kv_lens.to(torch.int64).cumsum(0)
    positions = torch.arange(total, dtype=torch.int64) - cu_kv[:-1].repeat_interleave(kv_lens)
    return _compute_slot_mappings(
        positions,
        host_block_offsets,
        req_indices,
        params.head_dim,
        params.tokens_per_block,
        params.quant_block_size,
    )


def _make(cached, new, seed=0):
    """Metadata / indexer params of one context batch: per request ``cached[i]`` + ``new[i]`` index-key tokens."""
    g = torch.Generator().manual_seed(seed)
    n = len(cached)
    kv_lens = torch.tensor([c + u for c, u in zip(cached, new)], dtype=torch.int32)
    # Distinct physical pages per request, -1 past the pages a request holds (as in the host table).
    host = torch.full((MAX_SEQS, MAX_BLOCKS_PER_SEQ), -1, dtype=torch.int32)
    pages = torch.randperm(MAX_SEQS * MAX_BLOCKS_PER_SEQ, generator=g, dtype=torch.int64).to(
        torch.int32
    )
    for i in range(n):
        num_blocks = (int(kv_lens[i]) + TOKENS_PER_BLOCK - 1) // TOKENS_PER_BLOCK
        host[i, :num_blocks] = pages[i * MAX_BLOCKS_PER_SEQ : i * MAX_BLOCKS_PER_SEQ + num_blocks]
    # The device copy is sanitized in place after the H2D copy, like _update_indexer_k_cache_block_offsets.
    device = host.cuda().clamp_(min=0)
    regular_fp8 = torch.zeros(8, dtype=torch.int64, device="cuda")
    regular_scale = torch.zeros(8, dtype=torch.int64, device="cuda")
    metadata = SimpleNamespace(
        enable_context_mla_with_cached_kv=True,
        skip_indexer_for_ctx_reqs=False,
        host_indexer_k_cache_block_offsets=host,
        indexer_k_cache_block_offsets=device,
        slot_mapping_fp8=regular_fp8,
        slot_mapping_scale=regular_scale,
        slot_mapping_fp8_fullkv=None,
        slot_mapping_scale_fullkv=None,
    )
    params = SimpleNamespace(
        num_contexts=n,
        cached_kv_tokens=torch.tensor(cached, dtype=torch.int32),
        kv_lens=kv_lens,
        head_dim=HEAD_DIM,
        tokens_per_block=TOKENS_PER_BLOCK,
        quant_block_size=QUANT_BLOCK,
    )
    return metadata, params


@pytest.mark.parametrize(
    "cached,new",
    [
        ([65536], [2048]),
        ([65536], [16384]),
        ([131072], [16384]),
        ([204800], [2048]),
        ([204800], [16384]),
        ([65536, 131072, 204800, 98304], [4096, 1, 4095, 63]),
        ([204800] * 4, [4096] * 4),
        (
            [204800, 0, 65537, 131072, 1, 64, 204800, 100000],
            [2048, 16384, 1, 2048, 2047, 64, 1, 3000],
        ),
    ],
    ids=[
        "1x64k+2k",
        "1x64k+16k",
        "1x128k+16k",
        "1x200k+2k",
        "1x200k+16k",
        "4x-mixed",
        "4x200k+4k",
        "8x-mixed",
    ],
)
def test_fullkv_gather_mapping_matches_cpu_reference(cached, new):
    metadata, params = _make(cached, new)
    ref_fp8, ref_scale = _cpu_reference(params, metadata.host_indexer_k_cache_block_offsets)

    K31Indexer.recompute_context_kv_gather_mappings(metadata, params)

    assert metadata.slot_mapping_fp8_fullkv.is_cuda and metadata.slot_mapping_scale_fullkv.is_cuda
    assert metadata.slot_mapping_fp8_fullkv.dtype == torch.int64
    torch.testing.assert_close(metadata.slot_mapping_fp8_fullkv.cpu(), ref_fp8, rtol=0, atol=0)
    torch.testing.assert_close(metadata.slot_mapping_scale_fullkv.cpu(), ref_scale, rtol=0, atol=0)


@pytest.mark.parametrize("case", ["no_cached_tokens", "skip_indexer", "feature_off"])
def test_fullkv_gather_mapping_aliases_regular_mappings(case):
    cached = [0, 0] if case == "no_cached_tokens" else [65536, 0]
    metadata, params = _make(cached, [2048, 4096])
    if case == "skip_indexer":
        metadata.skip_indexer_for_ctx_reqs = True
    if case == "feature_off":
        metadata.enable_context_mla_with_cached_kv = False

    K31Indexer.recompute_context_kv_gather_mappings(metadata, params)

    assert metadata.slot_mapping_fp8_fullkv is metadata.slot_mapping_fp8
    assert metadata.slot_mapping_scale_fullkv is metadata.slot_mapping_scale


def test_fullkv_gather_mapping_skips_generation_only_batches():
    metadata, params = _make([65536], [1])
    params.num_contexts = 0

    K31Indexer.recompute_context_kv_gather_mappings(metadata, params)

    assert metadata.slot_mapping_fp8_fullkv is None and metadata.slot_mapping_scale_fullkv is None
