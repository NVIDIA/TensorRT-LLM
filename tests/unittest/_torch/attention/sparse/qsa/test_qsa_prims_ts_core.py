# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.attention.backends.sparse.qsa.module import qsa_sparse_gqa_reference
from tensorrt_llm._torch.attention.backends.sparse.qsa.params import (
    QSA_SPARSE_CORE_ENV,
    QSASparseParams,
)
from tensorrt_llm._torch.attention.backends.sparse.qsa.prims_ts_core import (
    prims_ts_available,
    prims_ts_sparse_gqa,
)

NUM_QO_HEADS = 12
NUM_KV_HEADS = 1
HEAD_DIM = 256
PAGE_SIZE = 32
COMPRESS_RATIO = 4
BLOCK_TOPK = 64
CONTEXT = 2048
PAGES_PER_REQUEST = CONTEXT // PAGE_SIZE


def _params(**overrides) -> QSASparseParams:
    kwargs = dict(
        index_n_heads=4,
        index_kv_heads=1,
        index_head_dim=128,
        token_topk=BLOCK_TOPK * COMPRESS_RATIO,
        compress_ratio=COMPRESS_RATIO,
    )
    kwargs.update(overrides)
    return QSASparseParams(**kwargs)


def test_sparse_core_backend_defaults_to_triton(monkeypatch) -> None:
    monkeypatch.delenv(QSA_SPARSE_CORE_ENV, raising=False)
    assert _params().sparse_core_backend == "triton"
    assert _params(sparse_core_backend="prims_ts").sparse_core_backend == "prims_ts"


def test_sparse_core_backend_env_override(monkeypatch) -> None:
    monkeypatch.setenv(QSA_SPARSE_CORE_ENV, "Auto")
    assert _params().sparse_core_backend == "auto"
    monkeypatch.setenv(QSA_SPARSE_CORE_ENV, "bogus")
    with pytest.raises(ValueError, match="sparse_core_backend"):
        _params()


def _requires_prims_ts() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    if torch.cuda.get_device_capability() not in ((10, 0), (10, 3), (10, 7)):
        pytest.skip("PrimTS QToken sparse attention requires SM100/SM103/SM107")
    if not prims_ts_available():
        pytest.skip("flashinfer.attention.prims_ts is not installed")


def _make_case(query_lens: list[int], seed: int):
    torch.manual_seed(seed)
    device = torch.device("cuda")
    num_requests = len(query_lens)
    rows = sum(query_lens)
    num_pages = num_requests * PAGES_PER_REQUEST
    k_cache = torch.randn(
        num_pages, NUM_KV_HEADS, PAGE_SIZE, HEAD_DIM, dtype=torch.bfloat16, device=device
    )
    v_cache = torch.randn_like(k_cache)
    block_table = torch.randperm(num_pages, dtype=torch.int32, device=device).view(
        num_requests, PAGES_PER_REQUEST
    )
    q = torch.randn(rows, NUM_QO_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=device)
    request_indices = torch.repeat_interleave(
        torch.arange(num_requests, dtype=torch.int32),
        torch.tensor(query_lens),
    )
    positions = torch.cat(
        [torch.arange(CONTEXT - length, CONTEXT, dtype=torch.int32) for length in query_lens]
    )
    block_ids = torch.full((rows, BLOCK_TOPK), -1, dtype=torch.int32)
    selected_tokens = torch.full(
        (rows, BLOCK_TOPK * COMPRESS_RATIO + COMPRESS_RATIO - 1), -1, dtype=torch.int32
    )
    for row in range(rows):
        position = int(positions[row])
        visible = (position + 1) // COMPRESS_RATIO
        chosen = torch.randperm(visible)[: min(BLOCK_TOPK, visible)]
        block_ids[row, : chosen.numel()] = chosen.to(torch.int32)
        tokens = (chosen[:, None] * COMPRESS_RATIO + torch.arange(COMPRESS_RATIO)).flatten()
        tail = torch.arange(visible * COMPRESS_RATIO, position + 1)
        tokens = torch.cat((tokens, tail))
        selected_tokens[row, : tokens.numel()] = tokens.to(torch.int32)
    metadata = SimpleNamespace(
        qsa_block_table=block_table,
        kv_cache_manager=SimpleNamespace(tokens_per_block=PAGE_SIZE),
    )
    return (
        q,
        k_cache,
        v_cache,
        block_table,
        block_ids.to(device),
        request_indices.to(device),
        positions.to(device),
        selected_tokens.to(device),
        metadata,
    )


@pytest.mark.parametrize(
    ("query_lens", "is_decode"),
    [
        ([37, 91], False),
        ([1] * 8, True),
    ],
    ids=["packed_prefill", "fixed_decode"],
)
def test_prims_ts_sparse_gqa_matches_reference(query_lens, is_decode) -> None:
    _requires_prims_ts()
    (
        q,
        k_cache,
        v_cache,
        block_table,
        block_ids,
        request_indices,
        positions,
        selected_tokens,
        metadata,
    ) = _make_case(query_lens, seed=1234 + len(query_lens))
    softmax_scale = HEAD_DIM**-0.5

    actual = prims_ts_sparse_gqa(
        q,
        k_cache,
        v_cache,
        block_table,
        block_ids,
        request_indices,
        positions,
        page_size=PAGE_SIZE,
        max_seq_len_kv=CONTEXT,
        softmax_scale=softmax_scale,
        is_decode=is_decode,
    )
    expected = qsa_sparse_gqa_reference(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        selected_tokens=selected_tokens,
        request_indices=request_indices,
        metadata=metadata,
        softmax_scale=softmax_scale,
    )
    torch.cuda.synchronize()
    assert actual.shape == expected.shape
    torch.testing.assert_close(actual.float(), expected.float(), atol=2e-2, rtol=2e-2)
