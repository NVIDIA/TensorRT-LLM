# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the attention/m3_index_score_t2sc entry (T2SC, dispatch.INDEX_SCORE_T2SC): the TP2 / TP8 MSA index
scorer with its pre-wait schedule, as the TP4 target's copy at its 1 index head per rank.

Every cell is held to the fp32 reference (a few fp32 ulps: the kernel sums the 128 exact products in its k-split order)
and to the in-tree scorer (``cute_dsl_minimax_m3_index_decode_score`` at its own 256-CTA split) bit for bit, which the
pre-wait prefetch, the early tile issue and the cell's split must not change.
"""

import zlib

import pytest
import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch.modeling_v2._native import NoTarget

try:
    from tensorrt_llm._torch.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__tp4 import (  # noqa: E501
        modeling as m,
    )
    from tensorrt_llm._torch.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__tp4.ops.attention.m3_index_score_t2sc import (  # noqa: E501
        PAGE,
    )
    from tensorrt_llm._torch.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__tp4.ops.attention.m3_index_score_t2sc import (  # noqa: E501
        m3_index_score_t2sc as op,
    )
    from tensorrt_llm._torch.modeling_v2.models.minimax_m3.ops import m3_fsel_topk as fsel
except NoTarget as exc:
    pytest.skip(str(exc), allow_module_level=True)

__extra_import_path__ = ["../../../.."]
from _validating import certifying, validating  # noqa: E402, F401

HEADS = 1  # index heads per rank at TP4 (4 index heads over 4 KV groups)
HEAD_DIM = 128


def _inputs(seq_lens, qlen: int, seed: int, dtype=torch.float8_e4m3fn):
    g = torch.Generator(device="cuda").manual_seed(seed)
    batch = len(seq_lens)
    max_blocks = (max(seq_lens) + PAGE - 1) // PAGE
    stride = ((max_blocks + 15) // 16) * 16
    pages = batch * max_blocks
    block_table = (
        torch.randperm(pages, device="cuda", generator=g).to(torch.int32).view(batch, max_blocks)
    )
    idx_q = torch.randn(batch * qlen, HEADS, HEAD_DIM, device="cuda", generator=g).to(dtype)
    index_k = torch.randn(pages, PAGE, HEAD_DIM, device="cuda", generator=g).to(dtype)
    seq = torch.tensor(seq_lens, device="cuda", dtype=torch.int32)
    return idx_q, index_k, block_table, seq, stride


def _score(stride: int, total_q: int):
    backing = torch.full(
        (HEADS, stride, total_q), -float("inf"), device="cuda", dtype=torch.float32
    )
    return backing.transpose(1, 2)


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture by this name
    spec = cell.spec
    qlen = spec["qlen"]
    idx_q, index_k, block_table, seq, stride = _inputs(
        spec["seq_lens"], qlen, zlib.crc32(cell.why.encode())
    )
    score = _score(stride, idx_q.shape[0])
    assert spec["heads"] == HEADS
    args = (idx_q, index_k, block_table, seq, score, qlen, spec["split_k"], spec["early"])
    certifying(op)
    ref = op.reference(*args)
    op(*args)
    stock = _score(stride, idx_q.shape[0])
    torch.ops.trtllm.cute_dsl_minimax_m3_index_decode_score(
        idx_q, index_k, block_table, seq, stock, qlen
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(score, ref, rtol=1e-5, atol=1e-4)
    assert torch.equal(score, stock), "the scores differ from the in-tree scorer"


@pytest.mark.parametrize("early", [True, False], ids=["early", "prefetch"])
def test_bf16_index_is_the_in_tree_scorer(early) -> None:
    idx_q, index_k, block_table, seq, stride = _inputs((5000,), 4, 11, dtype=torch.bfloat16)
    score, stock = _score(stride, 4), _score(stride, 4)
    op(idx_q, index_k, block_table, seq, score, 4, 1536, early)
    torch.ops.trtllm.cute_dsl_minimax_m3_index_decode_score(
        idx_q, index_k, block_table, seq, stock, 4
    )
    assert torch.equal(score, stock)


@pytest.mark.parametrize("qlen", [1, 2, 3, 4])
def test_prestaged_blocks_at_the_stable_edge_are_the_in_tree_scorer(qlen) -> None:
    """Lengths around the edge between pre-staged (all-old) and post-wait first blocks, 1 and 2 requests."""
    for k in (1, 2, 1294):
        for length in (k * PAGE - qlen - 1, k * PAGE - qlen, k * PAGE - qlen + 1, k * PAGE + 1):
            for seq_lens, split in (
                ((length,), 1536),
                ((length,), 1272),
                ((length, max(qlen, length // 3)), 768),
            ):
                idx_q, index_k, block_table, seq, stride = _inputs(seq_lens, qlen, 31 + length)
                score, stock = _score(stride, idx_q.shape[0]), _score(stride, idx_q.shape[0])
                op(idx_q, index_k, block_table, seq, score, qlen, split, True)
                torch.ops.trtllm.cute_dsl_minimax_m3_index_decode_score(
                    idx_q, index_k, block_table, seq, stock, qlen
                )
                assert torch.equal(score, stock), (qlen, seq_lens)


def test_guard_refuses_a_request_past_its_block_table() -> None:
    idx_q, index_k, block_table, seq, stride = _inputs((1000,), 1, 5)
    seq = seq + 4 * PAGE
    with pytest.raises(AssertionError, match="past its block-table row"):
        with validating(op):
            op(idx_q, index_k, block_table, seq, _score(stride + 16, 1), 1, 1536, True)


# T2SC_QOWN (opt-in): Q in its own buffer, so the CTA's second block is issued at the grid wait. The blocks and the
# per-block code are the default's, so the scores are the default schedule's bit for bit at every length, request
# count and query length, through repeated launches and graph replays.
_QOWN_CASES = [  # one served case per query length
    ((600000,), 1, 1272),
    ((600000,), 3, 1272),
    ((400000, 70000), 4, 768),
    ((165521,), 8, 1272),
]


def _kern():
    from tensorrt_llm._torch.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__tp4.kernels import (  # noqa: E501
        m3_index_score_t2sc as kern,
    )

    return kern


@pytest.mark.parametrize("seq_lens,qlen,split", _QOWN_CASES)
def test_qown_schedule_is_bitwise_the_default(seq_lens, qlen, split) -> None:
    kern = _kern()
    idx_q, index_k, block_table, seq, stride = _inputs(seq_lens, qlen, 11 + sum(seq_lens) % 1009)
    key = (idx_q.dtype, HEADS, qlen, HEAD_DIM, split, True)
    base = kern.IndexScoreRunner.compile_schedule(key)
    alt = kern.IndexScoreRunner.compile_schedule(key, qown=True)
    s_base = _score(stride, idx_q.shape[0])
    base(idx_q, index_k, block_table, s_base, seq)
    for _ in range(3):
        s_alt = _score(stride, idx_q.shape[0])
        alt(idx_q, index_k, block_table, s_alt, seq)
        torch.cuda.synchronize()
        assert torch.equal(s_base.view(torch.int32), s_alt.view(torch.int32)), (
            "a score differs from the default"
        )


def test_qown_in_a_cuda_graph_replays_bitwise() -> None:
    kern = _kern()
    cases = []
    for length in (600000, 330000):
        idx_q, index_k, block_table, seq, stride = _inputs((length,), 8, 77 + length % 1000)
        key = (idx_q.dtype, HEADS, 8, HEAD_DIM, 1272, True)
        base = kern.IndexScoreRunner.compile_schedule(key)
        alt = kern.IndexScoreRunner.compile_schedule(key, qown=True)
        ref = _score(stride, 8)
        base(idx_q, index_k, block_table, ref, seq)
        cases.append((alt, idx_q, index_k, block_table, seq, _score(stride, 8), ref))
    for alt, idx_q, index_k, block_table, seq, out, _ in cases:
        alt(idx_q, index_k, block_table, out, seq)  # load the modules before capture
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for alt, idx_q, index_k, block_table, seq, out, _ in cases:
            alt(idx_q, index_k, block_table, out, seq)
    for _ in range(50):
        for c in cases:
            c[5].fill_(-float("inf"))
        graph.replay()
        torch.cuda.synchronize()
        for c in cases:
            assert torch.equal(c[5].view(torch.int32), c[6].view(torch.int32))


# T2SC_TOKEN_MAJOR (modeling.T2SC_TOKEN_MAJOR) on the target's own routes: the scorer and the select the target takes
# at that row count (FSEL at <= FSEL_MAX_ROWS rows plus the 384-thread DSL select, the DSL select alone wider) run once
# on a block-major view of a flat store and once on the token-major view of a second store. The logical scores must be
# the same bit for bit, the token-major store must hold them token-major, and the top-16 tables must be the same bit for
# bit (and the FSEL reference's).
_TM_QLEN = 8  # K7 verify rows per request
_TM_NB = 8193  # the 1M engine's selector width (worst-case k tiles)
_TM_CASES = {
    "b1_128k": (131072,),  # 8 rows: FSEL and DSL
    "b2_mixed": (524288, 61000),  # 16 rows: DSL
}


def _tm_inputs(seq_lens, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    batch = len(seq_lens)
    max_blocks = (max(seq_lens) + PAGE - 1) // PAGE
    pages = batch * max_blocks
    block_table = (
        torch.randperm(pages, device="cuda", generator=g).to(torch.int32).view(batch, max_blocks)
    )
    idx_q = torch.randn(batch * _TM_QLEN, HEADS, HEAD_DIM, device="cuda", generator=g).to(
        torch.float8_e4m3fn
    )
    index_k = torch.randn(pages, PAGE, HEAD_DIM, device="cuda", generator=g).to(torch.float8_e4m3fn)
    seq = torch.tensor(seq_lens, device="cuda", dtype=torch.int32)
    # Each verify row's causal valid-block count (the metadata's msa_n_valid_blocks).
    nvb = torch.tensor(
        [(s - _TM_QLEN + t + 1 + PAGE - 1) // PAGE for s in seq_lens for t in range(_TM_QLEN)],
        dtype=torch.int32,
        device="cuda",
    )
    return idx_q, index_k, block_table, seq, nvb


def _tm_view(token_major: bool, total_q: int):
    store = torch.full(
        (HEADS * _TM_NB * total_q + 4096,), -float("inf"), device="cuda", dtype=torch.float32
    )
    view = store[: HEADS * _TM_NB * total_q].view(HEADS, _TM_NB, total_q)
    return store, (m.max_score_token_major(view) if token_major else view)


def _tm_selects(scores, nvb, rows):
    """{name: table} for every select the target can run at ``rows`` (FSEL at <= FSEL_MAX_ROWS, DSL at any)."""
    from tensorrt_llm._torch.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__tp4.kernels.long_select import (  # noqa: E501
        long_select_blocks,
    )

    out = {}
    if rows <= m.FSEL_MAX_ROWS:
        out["fsel"] = fsel.fsel_select_blocks(
            scores,
            n_valid_blocks=nvb,
            init_blocks=0,
            local_blocks=1,
            use_pdl=False,
            trigger_phase=m.FSEL_TRIGGER_PHASE,
            variant=m.FSEL_VARIANT,
        )
    out["dsl"] = long_select_blocks(
        scores,
        n_valid_blocks=nvb,
        init_blocks=0,
        local_blocks=1,
        use_pdl=False,
        early_trigger=rows <= m.SELECT_EARLY_MAX_ROWS,
        trigger_phase=m.SELECT_LONG_TRIGGER_PHASE if rows <= m.SELECT_LONG_TRIGGER_MAX_ROWS else 0,
        dsl=True,
    )
    return out


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="T2SC_TOKEN_MAJOR kernels require an sm_10x CUDA device",
)
@pytest.mark.parametrize("case", list(_TM_CASES))
def test_token_major_scores_and_tables_are_bitwise(case) -> None:
    seq_lens = _TM_CASES[case]
    requests = len(seq_lens)
    assert m.INDEX_SCORE_T2SC_PATH.covers(requests)
    split_k, early = m.index_score_t2sc_schedule(requests)
    idx_q, index_k, block_table, seq, nvb = _tm_inputs(
        seq_lens, 1234 + requests * 7 + seq_lens[0] % 97
    )
    rows = requests * _TM_QLEN
    res = {}
    for tm in (False, True):
        store, view = _tm_view(tm, rows)
        op(idx_q, index_k, block_table, seq, view.transpose(1, 2), _TM_QLEN, split_k, early)
        res[tm] = (store, view, _tm_selects(view, nvb, rows))
    torch.cuda.synchronize()
    (bstore, bview, btab), (tstore, tview, ttab) = res[False], res[True]
    assert torch.equal(bview, tview), "the logical scores differ"
    n = HEADS * _TM_NB * rows
    assert torch.equal(tstore[:n].view(HEADS, rows, _TM_NB), bview.permute(0, 2, 1)), (
        "not backed token-major"
    )
    assert torch.equal(bstore[:n].view(HEADS, _TM_NB, rows), bview)
    ref = fsel.M3FselTopk().reference(bview, nvb, None, 0, 1)
    for name in btab:
        assert btab[name] is not None and ttab[name] is not None, name
        assert torch.equal(btab[name], ttab[name]), f"{name}: the top-16 tables differ"
        assert torch.equal(ttab[name], ref), f"{name}: the table is not the select contract's"
