# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

HEAD_DIM = 128
NUM_HEADS = 4
RATIO = 4
# BF16 operands are exact; the fp32 head-sum order differs between the UMMA
# accumulator and the einsum reference, so the tolerance scales with |score|.
REL_TOL = 2e-5
ABS_TOL = 1e-4


def _reference(q, index_cache, block_table, positions, request_indices, tokens_per_block):
    max_blocks = block_table.shape[1] * tokens_per_block // RATIO
    blocks = torch.arange(max_blocks, device=q.device)
    anchors = blocks * RATIO + RATIO - 1
    pages = block_table[request_indices.long()][:, anchors // tokens_per_block]
    keys = index_cache[pages, anchors % tokens_per_block, 0]
    scores = torch.einsum("rhd,rbd->rhb", q.float(), keys.float())
    scores = scores.clamp_min(0).sum(dim=1) * HEAD_DIM**-0.5
    visible = blocks[None, :] < ((positions + 1) // RATIO)[:, None]
    return scores.masked_fill(~visible, float("-inf")), visible


def _case(request_lengths, ctx, tokens_per_block, seed, *, q_scale=1.0, positions=None):
    torch.manual_seed(seed)
    device = torch.device("cuda")
    pages_per_request = (ctx + tokens_per_block - 1) // tokens_per_block
    num_requests = len(request_lengths)
    num_pages = num_requests * pages_per_request
    block_table = torch.randperm(num_pages, device=device).to(torch.int32).view(num_requests, -1)
    index_cache = torch.randn(
        num_pages, tokens_per_block, 1, HEAD_DIM, dtype=torch.bfloat16, device=device
    )
    rows = sum(request_lengths)
    q = (torch.randn(rows, NUM_HEADS, HEAD_DIM, device=device) * q_scale).to(torch.bfloat16)
    request_indices = torch.repeat_interleave(
        torch.arange(num_requests, device=device),
        torch.tensor(request_lengths, device=device),
    ).to(torch.int32)
    if positions is None:
        positions = torch.cat([torch.arange(ctx - n, ctx, device=device) for n in request_lengths])
    positions = positions.to(torch.int64)
    return q, index_cache, block_table, positions, request_indices


def _requires_prims_ts():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if torch.cuda.get_device_capability() not in ((10, 0), (10, 3), (10, 7)):
        pytest.skip("PrimTS QSA scorer requires SM100/SM103/SM107")
    pytest.importorskip("cutlass.experimental.task_scheduling")


def _scorer():
    from tensorrt_llm._torch.attention.backends.sparse.qsa.prims_ts_indexer import (
        prims_ts_qsa_index_scores,
    )

    return prims_ts_qsa_index_scores


def _check(out, ref, visible, only_visible):
    assert out.shape == ref.shape
    assert out.dtype == torch.float32
    if not only_visible:
        assert torch.equal(torch.isfinite(out), visible)
        assert torch.all(out[~visible] == float("-inf"))
    got, want = out[visible], ref[visible]
    assert torch.isfinite(got).all()
    err = (got - want).abs().max().item() if got.numel() else 0.0
    bound = ABS_TOL + REL_TOL * (want.abs().max().item() if want.numel() else 0.0)
    assert err <= bound, f"max abs err {err:.3e} > {bound:.3e}"
    return err


def _run(q, cache, table, pos, req, tokens_per_block, only_visible):
    out = _scorer()(
        q,
        cache,
        table,
        pos,
        req,
        tokens_per_block=tokens_per_block,
        compress_ratio=RATIO,
        only_visible_blocks=only_visible,
    )
    ref, visible = _reference(q, cache, table, pos, req, tokens_per_block)
    return out, ref, visible, _check(out, ref, visible, only_visible)


@pytest.mark.parametrize("tokens_per_block", (16, 32, 64))
@pytest.mark.parametrize("only_visible", (False, True))
@pytest.mark.parametrize(
    "request_lengths, ctx",
    [
        ((1,), 2048),
        ((5, 37), 2048),
        ((40, 24, 3, 61), 8192),
        ((1024,), 32768 + 36),
        ((4096,), 32768),
    ],
)
def test_prefill_rows(request_lengths, ctx, tokens_per_block, only_visible):
    _requires_prims_ts()
    q, cache, table, pos, req = _case(request_lengths, ctx, tokens_per_block, 7, q_scale=8.0)
    _run(q, cache, table, pos, req, tokens_per_block, only_visible)


@pytest.mark.parametrize("tokens_per_block", (16, 32, 64))
@pytest.mark.parametrize("num_requests", (1, 8, 64))
def test_decode_rows(num_requests, tokens_per_block):
    _requires_prims_ts()
    ctx = 8192
    q, cache, table, _, req = _case([1] * num_requests, ctx, tokens_per_block, 11, q_scale=4.0)
    device = q.device
    # Rows at the first positions have no complete block, rows inside a partial
    # block or at a page boundary exercise the tail of the visible range.
    specials = torch.tensor(
        [
            0,
            1,
            2,
            3,
            4,
            7,
            tokens_per_block - 1,
            tokens_per_block,
            tokens_per_block * 3 - 1,
            ctx - 1,
        ],
        device=device,
    )
    pos = torch.randint(0, ctx, (num_requests,), device=device)
    pos[: specials.numel()] = specials[:num_requests]
    for only_visible in (False, True):
        _run(q, cache, table, pos.to(torch.int64), req, tokens_per_block, only_visible)


def test_prefill_mixed_positions_within_request():
    _requires_prims_ts()
    q, cache, table, pos, req = _case((100, 60), 8192, 32, 5)
    pos = pos.clone()
    pos[10:20] = 3
    pos[130] = 0
    _run(q, cache, table, pos, req, 32, False)


@pytest.mark.parametrize("seed", (1, 2, 3))
def test_prefill_seeds_and_large_scores(seed):
    _requires_prims_ts()
    q, cache, table, pos, req = _case((513, 7), 16384, 32, seed, q_scale=32.0)
    _run(q, cache, table, pos, req, 32, False)


def test_prefill_128k():
    _requires_prims_ts()
    q, cache, table, pos, req = _case((2048,), 131072, 32, 13)
    _run(q, cache, table, pos, req, 32, True)


def test_matches_triton_scorer():
    _requires_prims_ts()
    pytest.importorskip("tensorrt_llm")
    from tensorrt_llm._torch.attention.backends.sparse.qsa.kernels import (
        triton_qsa_paged_index_scores,
    )

    for lengths, ctx, tpb in (
        ((300, 4), 8192, 32),
        ((1,) * 16, 32768, 64),
        ((2000,), 16384, 16),
    ):
        q, cache, table, pos, req = _case(lengths, ctx, tpb, 21, q_scale=8.0)
        ours = _scorer()(
            q,
            cache,
            table,
            pos,
            req,
            tokens_per_block=tpb,
            compress_ratio=RATIO,
            only_visible_blocks=True,
        )
        theirs = triton_qsa_paged_index_scores(
            q=q,
            index_cache=cache,
            block_table=table,
            query_positions=pos,
            request_indices=req,
            tokens_per_block=tpb,
            compress_ratio=RATIO,
            only_visible_blocks=True,
        )
        visible = (
            torch.arange(ours.shape[1], device=q.device)[None, :] < ((pos + 1) // RATIO)[:, None]
        )
        err = (ours[visible] - theirs[visible]).abs().max().item()
        assert err <= ABS_TOL + REL_TOL * theirs[visible].abs().max().item(), err


def test_deterministic():
    _requires_prims_ts()
    q, cache, table, pos, req = _case((257, 31), 8192, 32, 17)
    a = _scorer()(q, cache, table, pos, req, tokens_per_block=32, compress_ratio=RATIO)
    b = _scorer()(q, cache, table, pos, req, tokens_per_block=32, compress_ratio=RATIO)
    assert torch.equal(a, b)


def test_prepared_launch_graph_replay():
    _requires_prims_ts()
    from tensorrt_llm._torch.attention.backends.sparse.qsa.prims_ts_indexer import (
        prepare_qsa_index_scores,
    )

    q, cache, table, pos, req = _case((64, 64), 8192, 32, 19)
    launch, scores = prepare_qsa_index_scores(
        q, cache, table, pos, req, tokens_per_block=32, compress_ratio=RATIO
    )
    launch()
    torch.cuda.synchronize()
    ref, visible = _reference(q, cache, table, pos, req, 32)
    _check(scores, ref, visible, False)
    q.copy_(torch.randn_like(q, dtype=torch.float32).to(torch.bfloat16))
    scores.fill_(float("nan"))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch()
    graph.replay()
    torch.cuda.synchronize()
    ref, visible = _reference(q, cache, table, pos, req, 32)
    _check(scores, ref, visible, False)


@pytest.mark.parametrize("tokens_per_block", (16, 32))
def test_strided_index_cache_view_is_not_copied(tokens_per_block):
    _requires_prims_ts()
    from tensorrt_llm._torch.attention.backends.sparse.qsa.prims_ts_indexer import (
        prepare_qsa_index_scores,
    )

    q, cache, table, pos, req = _case((200, 56), 4096, tokens_per_block, 23)
    # The KV pool coalesces the index cache with K/V, so production views have
    # a page stride larger than one page.
    pool = torch.zeros(
        (cache.shape[0], 5, *cache.shape[1:]), dtype=cache.dtype, device=cache.device
    )
    view = pool[:, 2]
    view.copy_(cache)
    assert not view.is_contiguous()
    launch, scores = prepare_qsa_index_scores(
        q, view, table, pos, req, tokens_per_block=tokens_per_block, compress_ratio=RATIO
    )
    assert any(t.data_ptr() == view.data_ptr() for t in launch.keepalive)
    launch()
    dense = _scorer()(
        q, cache, table, pos, req, tokens_per_block=tokens_per_block, compress_ratio=RATIO
    )
    torch.cuda.synchronize()
    assert torch.equal(scores, dense)
