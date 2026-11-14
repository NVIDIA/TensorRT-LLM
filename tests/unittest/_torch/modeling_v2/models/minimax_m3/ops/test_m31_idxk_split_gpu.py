# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""IDXK_SPLIT's four ops (ops/idxk_split.py) on one GPU with the eight TP8 ranks emulated, against torch references.

Each emulated rank stores into every rank's local workspace (the unicast form of the exchange). Per step:
``plan_step`` writes each rank's compact block table and scorer lengths (checked against the formula), the even ranks
publish their index-q head and every rank gathers the [rows, 4, 128] tile (checked against the four heads), and from
synthetic per-block scores (rank ``r`` holds the scores of global blocks ``j * 8 + r``, as its scorer would write them)
``local_topk`` + ``merge`` give every rank the FSEL table of its head, which must equal ``reference_table`` (the FSEL
contract in torch) on the full score rows, eagerly and on CUDA-graph replays across both workspace halves. Ties
(duplicated scores across ranks), equal scores everywhere and rows of at most 16 valid blocks are covered.
"""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.ops import idxk_split

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="the IDXK_SPLIT kernels run on SM10x",
)

QLEN = 8
ROWS_MAX = 32
INIT, LOCAL = 0, 1
WORLD = idxk_split.WORLD
HEADS = idxk_split.HEADS


def _ops():
    if not idxk_split.ops_available():
        pytest.skip("the family library lacks the minimax_m3_sm107tp8_idxk_* ops")


def _fixture(lens, mode, seed=1234):
    """Block table, sequence lengths, per-row valid blocks and the global scores [4 heads, rows, num_cols]."""
    g = torch.Generator(device="cuda")
    g.manual_seed(seed)
    pages = [(s + 127) // 128 for s in lens]
    num_cols = max(pages)
    rows = len(lens) * QLEN
    block_table = torch.randint(
        0, 1 << 20, (len(lens), num_cols), generator=g, device="cuda", dtype=torch.int32
    )
    seq_lens = torch.tensor(lens, dtype=torch.int32, device="cuda")
    pos = torch.cat([torch.arange(s - QLEN, s) for s in lens])
    n_valid = (pos // 128 + 1).to(torch.int32).cuda()
    scores = torch.randn(HEADS, rows, num_cols, generator=g, device="cuda")
    if mode == "ties":
        # The same scores repeat every 37 blocks: equal keys on blocks of every rank.
        scores = scores[:, :, :37].repeat(1, 1, (num_cols + 36) // 37)[:, :, :num_cols].contiguous()
    elif mode == "equal":
        scores.zero_()
    return block_table, seq_lens, n_valid, scores, num_cols


def _compact_reference(block_table, seq_lens, rank, width):
    """idxkSplit.h ``invokeCompact``: (cbt [B, width], score_lens [B])."""
    cbt = block_table[:, :1].repeat(1, width).cpu()
    lens = []
    for b, s in enumerate(seq_lens.tolist()):
        n_local = -(-((s - 1) // 128 - rank) // WORLD)
        n_local = max(0, min(n_local, width - 1))
        for j in range(n_local):
            cbt[b, j] = block_table[b, j * WORLD + rank]
        lens.append(n_local * 128 + QLEN)
    return cbt, torch.tensor(lens, dtype=torch.int32)


def _rank_scores(scores, rank, plan):
    """Rank ``rank``'s compact scores [4, W, rows]: global block j * 8 + rank, -inf past the row."""
    out = torch.full_like(plan.scores, -float("inf"))
    owned = scores[:, :, rank::WORLD]
    n = min(owned.shape[2], out.shape[1])
    out[:, :n, : scores.shape[1]] = owned[:, :, :n].transpose(1, 2)
    return out


@pytest.mark.parametrize(
    "lens,mode",
    [
        ([131072 - 1031 * i for i in range(2)], "random"),
        ([70000, 9000, 2300, 2050], "random"),
        ([262144, 262144 - 77], "ties"),
        ([70000, 9000, 2300, 2050], "equal"),
        ([129, 1000, 2048 + 8, 2176 + 3], "random"),
        ([128 * 600, 128 * 600 + 1, 128 * 600 + 7, 128 * 600 + 8], "random"),
    ],
    ids=["b2-128K", "b4-mix", "b2-ties", "b4-equal", "b4-short", "b4-bounds"],
)
@torch.inference_mode()
def test_emulated_ranks_match_the_references(lens, mode):
    _ops()
    block_table, seq_lens, n_valid, scores, num_cols = _fixture(lens, mode)
    requests, rows = len(lens), len(lens) * QLEN
    ws = idxk_split.create_workspaces_emulated(ROWS_MAX, 1, torch.device("cuda"))
    plans = [idxk_split.make_plan(requests, QLEN, num_cols, "cuda") for _ in range(WORLD)]
    outs = [torch.full((rows, 1, 16), -7, dtype=torch.int32, device="cuda") for _ in range(WORLD)]
    idx_q = (torch.randn(rows, HEADS, 128, device="cuda") * 0.5).to(torch.float8_e4m3fn)

    def step():
        for r in range(WORLD):
            idxk_split.plan_step(ws[r], plans[r], block_table, seq_lens)
        for r in range(WORLD):
            idxk_split.exchange_q(ws[r], plans[r], 0, idx_q[:, r // 2], publish=True, gather=False)
        for r in range(WORLD):
            idxk_split.exchange_q(ws[r], plans[r], 0, idx_q[:, r // 2], publish=False, gather=True)
        for r in range(WORLD):
            plans[r].scores.copy_(_rank_scores(scores, r, plans[r]))
            idxk_split.local_topk(ws[r], plans[r], 0, n_valid, INIT, LOCAL)
        for r in range(WORLD):
            idxk_split.merge(ws[r], plans[r], 0, n_valid, outs[r], INIT, LOCAL)

    step()
    torch.cuda.synchronize()
    want = [
        idxk_split.reference_table(
            scores[h : h + 1].transpose(1, 2).contiguous(), n_valid.cpu(), INIT, LOCAL
        )
        for h in range(HEADS)
    ]
    q_tile = idx_q.view(torch.uint8)
    for r in range(WORLD):
        cbt, score_lens = _compact_reference(block_table, seq_lens, r, plans[r].cbt.shape[1])
        assert torch.equal(plans[r].cbt.cpu(), cbt), r
        assert torch.equal(plans[r].score_lens.cpu(), score_lens), r
        assert torch.equal(plans[r].q_tile, q_tile), r
        assert torch.equal(outs[r].cpu(), want[r // 2]), r

    # Captured once and replayed three times: both workspace halves, and across the 18-bit tag wrap.
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        step()
    for o in outs:
        o.fill_(-7)
    for w in ws:
        w.epoch.fill_((1 << 18) - 2)
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    for r in range(WORLD):
        assert torch.equal(outs[r].cpu(), want[r // 2]), r


def test_workspace_layout_helpers():
    """The Python mirrors of idxkSplit.h's sizes and the workspace's start state."""
    assert (
        idxk_split.layer_words(ROWS_MAX)
        == 2 * HEADS * ROWS_MAX * 32 + 2 * WORLD * HEADS * ROWS_MAX * 16
    )
    assert idxk_split.compact_width(9216) == 9216 // WORLD + 1
    assert idxk_split.compact_width(20000) == idxk_split.compact_width(9216)
    local = torch.full((idxk_split.layer_words(ROWS_MAX),), -1, dtype=torch.int64)
    idxk_split.init_local(local, ROWS_MAX, 1)
    qw = 2 * HEADS * ROWS_MAX * 32
    half = WORLD * HEADS * ROWS_MAX * 16
    assert int(local[:qw].abs().sum()) == 0
    assert bool((local[qw : qw + half] == 1 << 14).all())
    assert int(local[qw + half :].abs().sum()) == 0
