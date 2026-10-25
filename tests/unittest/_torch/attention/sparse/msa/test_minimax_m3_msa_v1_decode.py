# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The opt-in MiniMax MSA-v1 Q8KV4 decode against the Triton MiniMax-M3 sparse decode.

Both read one NVFP4 pool written by the production cache writer (K scales token-major, V scales
in 4x4 token quads) through the same page table, lengths and top-k pages. Q8KV4 rounds the
dequantized K/V and the softmax probabilities to E4M3, so the comparison is by relative L2
error rather than element-wise. Skipped unless a MiniMax MSA build with the quad4 V-scale
layout is importable on SM100/SM103.
"""

import pytest
import torch

from tensorrt_llm._torch.attention.backends.sparse.minimax_m3.kernels import msa_v1_decode
from tensorrt_llm._torch.attention.backends.sparse.minimax_m3.kernels.msa_scatter import (
    fused_write_layer_caches_nvfp4,
)
from tensorrt_llm._torch.attention.backends.sparse.minimax_m3.kernels.triton_sparse_decode import (
    minimax_m3_sparse_attn_decode,
)
from tensorrt_llm._utils import get_sm_version

PAGE = 128
HEAD_DIM = 128
TOPK = 16

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or get_sm_version() not in (100, 103),
    reason="MSA-v1 Q8KV4 decode requires SM100/SM103.",
)


@pytest.fixture
def msa_enabled(monkeypatch):
    monkeypatch.setattr(msa_v1_decode, "_ENABLED", True)
    msa_v1_decode._wrapper_class.cache_clear()
    if not msa_v1_decode.is_enabled():
        msa_v1_decode._wrapper_class.cache_clear()
        pytest.skip("MiniMax MSA with the quad4 V-scale layout is not importable.")
    yield
    msa_v1_decode._wrapper_class.cache_clear()


def _make_case(seq_lens, *, num_kv_heads, group, q_len, seed):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    batch = len(seq_lens)
    max_blocks = max(1, -(-max(seq_lens) // PAGE))
    num_pages = batch * max_blocks + 3
    num_slots = num_pages * PAGE
    data_pool = torch.zeros(
        num_pages, 2, num_kv_heads, PAGE, HEAD_DIM // 2, dtype=torch.uint8, device="cuda"
    )
    scale_pool = torch.zeros(
        num_pages, 2, num_kv_heads, PAGE, HEAD_DIM // 16, dtype=torch.uint8, device="cuda"
    )
    k, v = (
        torch.randn(num_slots, num_kv_heads * HEAD_DIM, device="cuda", generator=gen).to(
            torch.bfloat16
        )
        for _ in range(2)
    )
    unit = torch.ones(3, dtype=torch.float32, device="cuda")
    slots = torch.arange(num_slots, dtype=torch.int32, device="cuda")
    assert fused_write_layer_caches_nvfp4(
        data_pool[:, 0], data_pool[:, 1], scale_pool[:, 0], scale_pool[:, 1],
        None, slots, k, v, None, unit,
    )
    block_table = (
        torch.randperm(num_pages, device="cuda", generator=gen)[: batch * max_blocks]
        .to(torch.int32)
        .view(batch, max_blocks)
    )
    total_q = batch * q_len
    topk = torch.full((num_kv_heads, total_q, TOPK), -1, dtype=torch.int32, device="cuda")
    for req, seq_len in enumerate(seq_lens):
        for intra in range(q_len):
            local = (seq_len - q_len + intra) // PAGE
            for head in range(num_kv_heads):
                hist = torch.randperm(max(local, 1), device="cuda", generator=gen)[: TOPK - 1]
                hist = hist[hist < local]
                chosen = torch.cat([hist.sort().values, hist.new_tensor([local])])
                topk[head, req * q_len + intra, : chosen.numel()] = chosen.to(torch.int32)
    q = (torch.randn(total_q, num_kv_heads * group, HEAD_DIM, device="cuda", generator=gen)).to(
        torch.float8_e4m3fn
    )
    one = torch.ones(1, dtype=torch.float32, device="cuda")
    return dict(
        args=(q, data_pool[:, 0], data_pool[:, 1], topk, block_table,
              torch.tensor(seq_lens, dtype=torch.int32, device="cuda")),
        kwargs=dict(
            sm_scale=HEAD_DIM**-0.5,
            decode_query_len=q_len,
            k_block_scale=scale_pool[:, 0],
            v_block_scale=scale_pool[:, 1],
            k_global_scale=one,
            v_global_scale=one.clone(),
        ),
    )


def _out(case):
    q = case["args"][0]
    return torch.empty(q.shape, dtype=torch.bfloat16, device=q.device)


def _rel_l2(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


@pytest.mark.parametrize("num_kv_heads,group", [(2, 16), (1, 16), (2, 8)])
@pytest.mark.parametrize("q_len", [1, 4, 8])
def test_msa_v1_decode_matches_triton(msa_enabled, num_kv_heads, group, q_len):
    case = _make_case([300, 2049, 9000, 20000], num_kv_heads=num_kv_heads, group=group,
                      q_len=q_len, seed=11 + q_len)
    expected = _out(case)
    minimax_m3_sparse_attn_decode(*case["args"], output=expected, **case["kwargs"])
    out = _out(case)
    assert msa_v1_decode.try_decode(*case["args"], output=out, plan_cache={}, **case["kwargs"])
    assert torch.isfinite(out.float()).all()
    assert _rel_l2(out, expected) < 0.05


def test_msa_v1_decode_cuda_graph_replay_tracks_inputs(msa_enabled):
    case = _make_case([5000, 7000], num_kv_heads=2, group=16, q_len=4, seed=5)
    plan_cache = {}
    out = _out(case)
    # The eager call plans; the capture reuses the plan and copies the live top-k table.
    assert msa_v1_decode.try_decode(*case["args"], output=out, plan_cache=plan_cache,
                                    **case["kwargs"])
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        assert msa_v1_decode.try_decode(*case["args"], output=out, plan_cache=plan_cache,
                                        **case["kwargs"])
    topk = case["args"][3]
    topk[:, :, 0] = torch.where(topk[:, :, 0] > 0, topk[:, :, 0] - 1, topk[:, :, 0])
    graph.replay()
    expected = _out(case)
    minimax_m3_sparse_attn_decode(*case["args"], output=expected, **case["kwargs"])
    assert _rel_l2(out, expected) < 0.05


def test_msa_v1_decode_declines_outside_contract(msa_enabled):
    case = _make_case([3000], num_kv_heads=2, group=16, q_len=4, seed=3)
    q = case["args"][0]
    bf16_args = (q.to(torch.bfloat16),) + case["args"][1:]
    assert not msa_v1_decode.try_decode(*bf16_args, output=_out(case), plan_cache={},
                                        **case["kwargs"])
    kwargs = dict(case["kwargs"], k_global_scale=torch.full_like(case["kwargs"]["k_global_scale"], 0.5))
    assert not msa_v1_decode.try_decode(*case["args"], output=_out(case), plan_cache={}, **kwargs)
    # No plan may be built under capture: a block table the eager call never planned for
    # leaves the graph on the Triton kernel.
    plan_cache = {}
    out = _out(case)
    assert msa_v1_decode.try_decode(*case["args"], output=out, plan_cache=plan_cache,
                                    **case["kwargs"])
    unplanned = case["args"][:4] + (case["args"][4].clone(),) + case["args"][5:]
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        ran = msa_v1_decode.try_decode(*unplanned, output=out, plan_cache=plan_cache,
                                       **case["kwargs"])
    assert not ran


def test_msa_v1_decode_off_without_env(monkeypatch):
    monkeypatch.setattr(msa_v1_decode, "_ENABLED", False)
    msa_v1_decode._wrapper_class.cache_clear()
    try:
        assert not msa_v1_decode.is_enabled()
    finally:
        msa_v1_decode._wrapper_class.cache_clear()
