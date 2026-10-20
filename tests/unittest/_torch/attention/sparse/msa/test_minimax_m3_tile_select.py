# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MiniMax-M3 SM107 prefill tile select: same top-16 tables as minimax_m3_select_blocks, and its gate."""

import os
import subprocess
import sys

import pytest
import torch

import tensorrt_llm  # noqa: F401  (registers torch.ops.trtllm)
from tensorrt_llm._torch.attention.backends.sparse.minimax_m3 import msa_indexer
from tensorrt_llm._utils import get_sm_version

PAGE = 128

sm107 = pytest.mark.skipif(
    not torch.cuda.is_available() or get_sm_version() != 107, reason="SM107 op"
)


def _causal_valid(cached: int, total_q: int, device) -> torch.Tensor:
    pos = cached + torch.arange(total_q, device=device) + 1
    return ((pos + PAGE - 1) // PAGE).to(torch.int32)


def _stock(scores, n_valid, init_blocks, local_blocks):
    return torch.ops.trtllm.minimax_m3_select_blocks(
        scores, n_valid, 16, init_blocks, local_blocks, True
    )


def _tile(scores, n_valid, init_blocks, local_blocks):
    heads, _, total_q = scores.shape
    out = torch.empty((heads, total_q, 16), dtype=torch.int32, device=scores.device).permute(
        1, 0, 2
    )
    torch.ops.trtllm.minimax_m3_select_prefill_tile(scores, n_valid, out, init_blocks, local_blocks)
    return out


def _assert_same(scores, n_valid, init_blocks=0, local_blocks=1):
    expected = _stock(scores, n_valid, init_blocks, local_blocks)
    actual = _tile(scores, n_valid, init_blocks, local_blocks)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@sm107
@pytest.mark.parametrize(
    "num_blocks,total_q,cached",
    [
        (1408, 2000, 165000),
        (640, 8192, 65000),
        (2688, 512, 330000),
        (1408, 1003, 160000),
        (129, 300, 0),
        (17, 24, 0),
        (2688, 4096, 327680),
        (4864, 1024, 606208),
        (4864, 1001, 606000),
    ],
)
def test_tile_select_matches_stock_random_causal(num_blocks, total_q, cached):
    g = torch.Generator(device="cuda").manual_seed(num_blocks * 7 + total_q)
    scores = torch.randn(4, num_blocks, total_q, device="cuda", generator=g)
    n_valid = _causal_valid(cached, total_q, "cuda").clamp(max=num_blocks + 3)
    _assert_same(scores, n_valid)


@sm107
@pytest.mark.parametrize("init_blocks,local_blocks", [(0, 1), (2, 3), (0, 0)])
def test_tile_select_matches_stock_mixed_valid_and_forcing(init_blocks, local_blocks):
    g = torch.Generator(device="cuda").manual_seed(11)
    scores = torch.randn(4, 1408, 512, device="cuda", generator=g)
    n_valid = torch.randint(0, 1500, (512,), device="cuda", generator=g, dtype=torch.int32)
    n_valid[:8] = torch.tensor(
        [0, 1, 15, 16, 17, 1408, 1409, 5000], device="cuda", dtype=torch.int32
    )
    _assert_same(scores, n_valid, init_blocks, local_blocks)


@sm107
@pytest.mark.parametrize("distinct", [3, 40])
def test_tile_select_matches_stock_ties(distinct):
    # Three distinct values put hundreds of candidates at the bound: the radix fallback.
    g = torch.Generator(device="cuda").manual_seed(distinct)
    scores = torch.randint(0, distinct, (4, 1408, 256), device="cuda", generator=g).float()
    _assert_same(scores, _causal_valid(165000, 256, "cuda"))


@sm107
def test_tile_select_matches_stock_nonfinite():
    # +inf, -inf, +NaN, +0 and -0. A negative NaN is left out: the contract ranks it below -inf, the in-tree
    # selector's fp16 first pass does not, and the proxy max-scores never produce one.
    g = torch.Generator(device="cuda").manual_seed(5)
    scores = torch.randn(4, 1408, 256, device="cuda", generator=g)
    m = torch.rand(scores.shape, device="cuda", generator=g)
    scores[m < 0.002] = float("inf")
    scores[(m >= 0.002) & (m < 0.004)] = float("nan")
    scores[(m >= 0.004) & (m < 0.5)] = float("-inf")
    scores[(m >= 0.5) & (m < 0.51)] = -0.0
    scores[(m >= 0.51) & (m < 0.52)] = 0.0
    _assert_same(scores, _causal_valid(165000, 256, "cuda"))
    _assert_same(
        torch.full((4, 200, 256), float("-inf"), device="cuda"),
        torch.arange(256, device="cuda", dtype=torch.int32) % 200,
    )


@sm107
def test_tile_select_strided_blocks_and_dispatch():
    g = torch.Generator(device="cuda").manual_seed(3)
    full = torch.randn(4, 1408, 1024, device="cuda", generator=g)
    scores = full[:, :, :700]  # unit query stride, block stride 1024
    n_valid = _causal_valid(165000, 700, "cuda")
    _assert_same(scores, n_valid)
    table = msa_indexer._tile_select(scores, n_valid, 0, 1)
    torch.testing.assert_close(table, _stock(scores, n_valid, 0, 1), rtol=0, atol=0)
    assert table.permute(1, 0, 2).is_contiguous()


@sm107
def test_tile_select_unaligned_base_and_strides():
    g = torch.Generator(device="cuda").manual_seed(4)
    full = torch.randn(4, 1408, 1024, device="cuda", generator=g)
    # A base one query in: 16-byte loads would be misaligned, so the rows are read one float at a time.
    _assert_same(full[:, :, 1:701], _causal_valid(165000, 700, "cuda"), 2, 3)
    # Odd block stride.
    odd = torch.randn(4, 1408, 1027, device="cuda", generator=g)[:, :, :1024]
    _assert_same(odd, _causal_valid(165000, 1024, "cuda"), 1, 2)


@sm107
@pytest.mark.parametrize("distinct", [3, 40])
@pytest.mark.parametrize("init_blocks,local_blocks", [(0, 1), (2, 1), (1, 2)])
def test_tile_select_wide_tiles_ties_and_forcing(distinct, init_blocks, local_blocks):
    # 4864 blocks: the tile is staged by the wide launch; heavy ties exercise the bound, the candidate cap and the
    # radix fallback, with forced init / local blocks.
    g = torch.Generator(device="cuda").manual_seed(distinct * 13 + init_blocks)
    scores = torch.randint(0, distinct, (1, 4864, 512), device="cuda", generator=g).float()
    _assert_same(scores, _causal_valid(606208, 512, "cuda"), init_blocks, local_blocks)


@sm107
@pytest.mark.parametrize("num_blocks,cached", [(1408, 165000), (4864, 606208)])
def test_tile_select_graph_replay_matches_eager(num_blocks, cached):
    g = torch.Generator(device="cuda").manual_seed(num_blocks)
    scores = torch.randn(1, num_blocks, 1024, device="cuda", generator=g)
    n_valid = _causal_valid(cached, 1024, "cuda")
    eager = _tile(scores, n_valid, 0, 1)
    out = torch.full_like(eager, -7)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        torch.ops.trtllm.minimax_m3_select_prefill_tile(scores, n_valid, out, 0, 1)
    out.fill_(-7)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, eager)


_KILL_SNIPPET = """
import sys, torch
import tensorrt_llm
g = torch.Generator(device="cuda").manual_seed(29)
scores = torch.randint(0, 40, (1, 4864, 512), device="cuda", generator=g).float()
pos = 606208 + torch.arange(512, device="cuda") + 1
n_valid = ((pos + 127) // 128).to(torch.int32)
out = torch.empty((1, 512, 16), dtype=torch.int32, device="cuda").permute(1, 0, 2)
torch.ops.trtllm.minimax_m3_select_prefill_tile(scores, n_valid, out, 1, 2)
torch.save(out.cpu(), sys.argv[1])
"""


@sm107
def test_tile_select_v2_kill_switch_same_tables(tmp_path):
    tables = {}
    # The v2 launch is the default; TRTLLM_M3_SELECT_TILE_V2=0 opts in to the v4 launch.
    for name, env in (
        ("v3", {"TRTLLM_M3_SELECT_TILE_V2": "0"}),
        ("v2", {"TRTLLM_M3_SELECT_TILE_V2": "1"}),
    ):
        path = tmp_path / f"{name}.pt"
        subprocess.run(
            [sys.executable, "-c", _KILL_SNIPPET, str(path)],
            env={**os.environ, **env},
            check=True,
        )
        tables[name] = torch.load(path)
    assert torch.equal(tables["v2"], tables["v3"])


@pytest.mark.parametrize(
    "sm_version,enabled,total_q,num_blocks",
    [
        (103, True, 2000, 1408),
        (107, False, 2000, 1408),
        (107, True, 64, 1408),
        (107, True, 2000, 6000),
    ],
)
def test_tile_select_gate(monkeypatch, sm_version, enabled, total_q, num_blocks):
    monkeypatch.setattr(msa_indexer, "get_sm_version", lambda: sm_version)
    monkeypatch.setattr(msa_indexer, "_SM107_TILE_SELECT", enabled)
    scores = torch.zeros(4, num_blocks, total_q)
    assert msa_indexer._tile_select(scores, torch.ones(total_q, dtype=torch.int32), 0, 1) is None
