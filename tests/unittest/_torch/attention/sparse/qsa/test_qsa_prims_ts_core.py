# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import itertools
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.attention.backends.sparse.qsa.module import qsa_sparse_gqa_reference
from tensorrt_llm._torch.attention.backends.sparse.qsa.params import QSASparseParams
from tensorrt_llm._torch.attention.backends.sparse.qsa.prims_ts_core import (
    align_prefill_chunk,
    prims_ts_available,
    prims_ts_sparse_gqa,
    slice_request_boundaries,
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


def test_sparse_core_backend_defaults_to_auto() -> None:
    assert _params().sparse_core_backend == "auto"
    assert _params().indexer_backend == "follow"
    assert _params(sparse_core_backend="prims_ts").sparse_core_backend == "prims_ts"
    with pytest.raises(ValueError, match="sparse_core_backend"):
        _params(sparse_core_backend="bogus")
    with pytest.raises(ValueError, match="indexer_backend"):
        _params(indexer_backend="bogus")


def _requires_prims_ts() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    if torch.cuda.get_device_capability() not in ((10, 0), (10, 3), (10, 7)):
        pytest.skip("PrimTS QToken sparse attention requires SM100/SM103/SM107")
    # The vendored package ships with TRT-LLM, so a missing kernel on a
    # supported GPU is a failure, not a skip.
    assert prims_ts_available()


def _make_case(query_lens: list[int], seed: int, kv_dtype: torch.dtype = torch.bfloat16):
    torch.manual_seed(seed)
    device = torch.device("cuda")
    num_requests = len(query_lens)
    rows = sum(query_lens)
    num_pages = num_requests * PAGES_PER_REQUEST
    k_cache = torch.randn(
        num_pages, NUM_KV_HEADS, PAGE_SIZE, HEAD_DIM, dtype=torch.bfloat16, device=device
    ).to(kv_dtype)
    v_cache = torch.randn(
        num_pages, NUM_KV_HEADS, PAGE_SIZE, HEAD_DIM, dtype=torch.bfloat16, device=device
    ).to(kv_dtype)
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


def test_slice_request_boundaries() -> None:
    cu_seq_lens = [37, 128, 131, 200]
    assert slice_request_boundaries(cu_seq_lens, 0, 200) == (0, 37, 128, 131, 200)
    assert slice_request_boundaries(cu_seq_lens, 0, 100) == (0, 37, 100)
    assert slice_request_boundaries(cu_seq_lens, 37, 128) == (0, 91)
    assert slice_request_boundaries(cu_seq_lens, 100, 140) == (0, 28, 31, 40)
    assert slice_request_boundaries(cu_seq_lens, 131, 200) == (0, 69)


def test_prims_ts_sparse_gqa_host_boundaries_match_readback() -> None:
    _requires_prims_ts()
    query_lens = [37, 91, 5]
    q, k_cache, v_cache, block_table, block_ids, request_indices, positions, _, _ = _make_case(
        query_lens, seed=99
    )
    kwargs = dict(
        page_size=PAGE_SIZE, max_seq_len_kv=CONTEXT, softmax_scale=HEAD_DIM**-0.5, is_decode=False
    )
    readback = prims_ts_sparse_gqa(
        q, k_cache, v_cache, block_table, block_ids, request_indices, positions, **kwargs
    )
    cu_seq_lens = list(itertools.accumulate(query_lens))
    host = prims_ts_sparse_gqa(
        q,
        k_cache,
        v_cache,
        block_table,
        block_ids,
        request_indices,
        positions,
        request_boundaries=slice_request_boundaries(cu_seq_lens, 0, sum(query_lens)),
        **kwargs,
    )
    torch.cuda.synchronize()
    assert torch.equal(readback, host)


def test_prefill_rows_share_plans_within_bucket() -> None:
    _requires_prims_ts()
    from tensorrt_llm._torch.attention.backends.sparse.qsa.prims_ts_core import _CORE

    kwargs = dict(
        page_size=PAGE_SIZE, max_seq_len_kv=CONTEXT, softmax_scale=HEAD_DIM**-0.5, is_decode=False
    )
    q, k_cache, v_cache, block_table, block_ids, request_indices, positions, _, _ = _make_case(
        [37, 91], seed=5
    )
    prims_ts_sparse_gqa(
        q, k_cache, v_cache, block_table, block_ids, request_indices, positions, **kwargs
    )
    plans = len(_CORE._plans)
    q2, _, _, block_table2, block_ids2, request_indices2, positions2, selected2, metadata = (
        _make_case([100, 30], seed=6)
    )
    out = prims_ts_sparse_gqa(
        q2, k_cache, v_cache, block_table2, block_ids2, request_indices2, positions2, **kwargs
    )
    assert len(_CORE._plans) == plans
    assert out.shape[0] == 130
    expected = qsa_sparse_gqa_reference(
        q=q2,
        k_cache=k_cache,
        v_cache=v_cache,
        selected_tokens=selected2,
        request_indices=request_indices2,
        metadata=metadata,
        softmax_scale=HEAD_DIM**-0.5,
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(out, expected, atol=2e-2, rtol=2e-2)


def test_route_offsets_follow_layout_changes_without_replanning() -> None:
    _requires_prims_ts()
    import flashinfer.attention.prims_ts.q_token_kv_block_sparse_metadata as fi_metadata

    kwargs = dict(
        page_size=PAGE_SIZE, max_seq_len_kv=CONTEXT, softmax_scale=HEAD_DIM**-0.5, is_decode=False
    )
    # Same row count and route count, different request boundaries.
    layouts = ([36, 92], [64, 64], [36, 92])
    cases = [_make_case(lens, seed=40 + i) for i, lens in enumerate(layouts)]
    k_cache, v_cache, block_table = cases[0][1], cases[0][2], cases[0][3]

    def run(case, lens):
        q, _, _, _, block_ids, request_indices, positions, _, _ = case
        cu_seq_lens = list(itertools.accumulate(lens))
        return prims_ts_sparse_gqa(
            q,
            k_cache,
            v_cache,
            block_table,
            block_ids,
            request_indices,
            positions,
            request_boundaries=slice_request_boundaries(cu_seq_lens, 0, sum(lens)),
            **kwargs,
        )

    expected = [
        prims_ts_sparse_gqa(
            case[0], k_cache, v_cache, block_table, case[4], case[5], case[6], **kwargs
        ).clone()
        for case in cases
    ]
    prepares = 0
    original = fi_metadata._prepare_q_token_kv_block_sparse_attention

    def counted(*args, **kw):
        nonlocal prepares
        prepares += 1
        return original(*args, **kw)

    fi_metadata._prepare_q_token_kv_block_sparse_attention = counted
    try:
        outputs = [run(case, lens).clone() for case, lens in zip(cases, layouts)]
    finally:
        fi_metadata._prepare_q_token_kv_block_sparse_attention = original
    torch.cuda.synchronize()
    assert prepares == 0
    for out, ref in zip(outputs, expected):
        assert torch.equal(out, ref)


def test_auto_gates_small_decode_batches() -> None:
    _requires_prims_ts()
    from tensorrt_llm._torch.attention.backends.sparse.qsa.prims_ts_core import (
        resolve_sparse_core_backend,
        use_prims_ts_indexer,
    )

    device = torch.device("cuda")
    k_cache = torch.empty(1, NUM_KV_HEADS, PAGE_SIZE, HEAD_DIM, dtype=torch.bfloat16, device=device)

    def core(backend, rows, is_decode):
        q = torch.empty(rows, NUM_QO_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=device)
        return resolve_sparse_core_backend(
            _params(sparse_core_backend=backend),
            q=q,
            k_cache=k_cache,
            page_size=PAGE_SIZE,
            is_decode=is_decode,
            num_rows=rows,
        )

    def indexer(backend, rows, is_decode):
        q = torch.empty(rows, 4, 128, dtype=torch.bfloat16, device=device)
        return use_prims_ts_indexer(_params(sparse_core_backend=backend), q, is_decode=is_decode)

    assert core("auto", 1, True) == "triton"
    assert core("auto", 2, True) == "prims_ts"
    assert core("auto", 1, False) == "prims_ts"
    assert core("prims_ts", 1, True) == "prims_ts"
    assert not indexer("auto", 31, True)
    assert indexer("auto", 32, True)
    assert indexer("auto", 1, False)
    assert indexer("prims_ts", 1, True)
    assert not indexer("triton", 4096, False)
    q64 = torch.empty(64, 4, 128, dtype=torch.bfloat16, device=device)
    q31 = torch.empty(31, 4, 128, dtype=torch.bfloat16, device=device)
    assert not use_prims_ts_indexer(
        _params(sparse_core_backend="prims_ts", indexer_backend="triton"), q64
    )
    assert use_prims_ts_indexer(
        _params(sparse_core_backend="triton", indexer_backend="prims_ts"), q31, is_decode=True
    )
    assert not use_prims_ts_indexer(
        _params(sparse_core_backend="prims_ts", indexer_backend="auto"), q31, is_decode=True
    )
    assert not use_prims_ts_indexer(_params(), q64, tokens_per_block=48)
    assert use_prims_ts_indexer(_params(), q64, tokens_per_block=64)


def test_align_prefill_chunk() -> None:
    assert align_prefill_chunk(4048) == 3840
    assert align_prefill_chunk(8192) == 8192
    assert align_prefill_chunk(256) == 256
    assert align_prefill_chunk(255) == 255
    assert align_prefill_chunk(96) == 96


def test_prefill_writes_into_caller_output() -> None:
    _requires_prims_ts()
    kwargs = dict(
        page_size=PAGE_SIZE, max_seq_len_kv=CONTEXT, softmax_scale=HEAD_DIM**-0.5, is_decode=False
    )
    for query_lens in ([128, 128], [37, 91]):
        q, k_cache, v_cache, block_table, block_ids, request_indices, positions, _, _ = _make_case(
            query_lens, seed=77
        )
        expected = prims_ts_sparse_gqa(
            q, k_cache, v_cache, block_table, block_ids, request_indices, positions, **kwargs
        ).clone()
        out = torch.full_like(q, float("nan"))
        result = prims_ts_sparse_gqa(
            q,
            k_cache,
            v_cache,
            block_table,
            block_ids,
            request_indices,
            positions,
            out=out,
            **kwargs,
        )
        torch.cuda.synchronize()
        assert result.data_ptr() == out.data_ptr()
        assert torch.equal(out, expected)


@pytest.mark.parametrize("is_decode", [False, True], ids=["packed_prefill", "fixed_decode"])
def test_prims_ts_sparse_gqa_fp8_kv_matches_reference(is_decode) -> None:
    _requires_prims_ts()
    query_lens = [1] * 8 if is_decode else [37, 91]
    q, k_cache, v_cache, block_table, block_ids, request_indices, positions, selected, metadata = (
        _make_case(query_lens, seed=4321, kv_dtype=torch.float8_e4m3fn)
    )
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
        k_cache=k_cache.to(torch.bfloat16),
        v_cache=v_cache.to(torch.bfloat16),
        selected_tokens=selected,
        request_indices=request_indices,
        metadata=metadata,
        softmax_scale=softmax_scale,
    )
    torch.cuda.synchronize()
    # FP8 Q on the PrimTS side vs BF16 Q upcast on the reference side.
    torch.testing.assert_close(actual.float(), expected.float(), atol=5e-2, rtol=5e-2)


@pytest.mark.parametrize("kv_dtype", [torch.bfloat16, torch.float8_e4m3fn], ids=["bf16", "fp8"])
def test_decode_graph_capture_replay(kv_dtype) -> None:
    _requires_prims_ts()
    q, k_cache, v_cache, block_table, block_ids, request_indices, positions, _, _ = _make_case(
        [1] * 4, seed=99, kv_dtype=kv_dtype
    )
    kwargs = dict(
        page_size=PAGE_SIZE, max_seq_len_kv=CONTEXT, softmax_scale=HEAD_DIM**-0.5, is_decode=True
    )
    out = torch.empty_like(q)
    # Eager run prepares the plan; capture must not need the host afterwards.
    prims_ts_sparse_gqa(
        q, k_cache, v_cache, block_table, block_ids, request_indices, positions, out=out, **kwargs
    )
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        prims_ts_sparse_gqa(
            q,
            k_cache,
            v_cache,
            block_table,
            block_ids,
            request_indices,
            positions,
            out=out,
            **kwargs,
        )
    q.copy_(torch.randn_like(q, dtype=torch.float32).to(q.dtype))
    out.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    eager = prims_ts_sparse_gqa(
        q, k_cache, v_cache, block_table, block_ids, request_indices, positions, **kwargs
    )
    torch.cuda.synchronize()
    assert torch.equal(out, eager)


def test_decode_rows_share_plans_within_bucket() -> None:
    _requires_prims_ts()
    from tensorrt_llm._torch.attention.backends.sparse.qsa.prims_ts_core import _CORE

    kwargs = dict(
        page_size=PAGE_SIZE, max_seq_len_kv=CONTEXT, softmax_scale=HEAD_DIM**-0.5, is_decode=True
    )
    q, k_cache, v_cache, block_table, block_ids, request_indices, positions, selected, metadata = (
        _make_case([1] * 5, seed=7)
    )
    prims_ts_sparse_gqa(
        q[:3],
        k_cache,
        v_cache,
        block_table,
        block_ids[:3],
        request_indices[:3],
        positions[:3],
        **kwargs,
    )
    plans = len(_CORE._plans)
    out = prims_ts_sparse_gqa(
        q, k_cache, v_cache, block_table, block_ids, request_indices, positions, **kwargs
    )
    assert len(_CORE._plans) == plans
    assert out.shape[0] == 5
    expected = qsa_sparse_gqa_reference(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        selected_tokens=selected,
        request_indices=request_indices,
        metadata=metadata,
        softmax_scale=HEAD_DIM**-0.5,
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(out, expected, atol=2e-2, rtol=2e-2)
