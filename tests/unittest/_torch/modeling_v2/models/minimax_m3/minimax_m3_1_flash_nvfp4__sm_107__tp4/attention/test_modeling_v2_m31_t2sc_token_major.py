# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU oracle for T2SC_TOKEN_MAJOR (modeling.T2SC_TOKEN_MAJOR): the T2SC scorer writes, and the long-row select reads,
the [heads, blocks, tokens] selector buffer backed token-major instead of block-major.

Per case the scorer and the select the target takes at that row count (FSEL at <= FSEL_MAX_ROWS rows, the 384-thread
DSL select wider; both also run at 8 rows) run once on a block-major view of a flat store and once on the token-major
view of a second store. The logical scores must be the same bit for bit, the token-major store must hold them
token-major, and the top-16 tables must be the same bit for bit (and the FSEL reference's).
"""

import pytest
import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__tp4 import (
    modeling as m,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__tp4.kernels.long_select import (  # noqa: E501
    long_select_blocks,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__tp4.ops.attention.m3_index_score_t2sc import (  # noqa: E501
    PAGE,
    m3_index_score_t2sc,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.ops import m3_fsel_topk as fsel

requires_sm100_family = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="T2SC_TOKEN_MAJOR kernels require an sm_10x CUDA device",
)

HEADS = 1  # index heads (= KV heads) per rank at TP4
HEAD_DIM = 128
QLEN = 8  # K7 verify rows per request
NB = 8193  # the 1M engine's selector width (worst-case k tiles)

CASES = {
    "b1_128k": (131072,),
    "b1_512k": (524288,),
    "b1_short": (20000,),
    "b2_128k": (131072, 131072),
    "b2_mixed": (524288, 61000),
}


def _inputs(seq_lens, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    batch = len(seq_lens)
    max_blocks = (max(seq_lens) + PAGE - 1) // PAGE
    pages = batch * max_blocks
    block_table = (
        torch.randperm(pages, device="cuda", generator=g).to(torch.int32).view(batch, max_blocks)
    )
    idx_q = torch.randn(batch * QLEN, HEADS, HEAD_DIM, device="cuda", generator=g).to(
        torch.float8_e4m3fn
    )
    index_k = torch.randn(pages, PAGE, HEAD_DIM, device="cuda", generator=g).to(torch.float8_e4m3fn)
    seq = torch.tensor(seq_lens, device="cuda", dtype=torch.int32)
    # Each verify row's causal valid-block count (the metadata's msa_n_valid_blocks).
    nvb = torch.tensor(
        [(s - QLEN + t + 1 + PAGE - 1) // PAGE for s in seq_lens for t in range(QLEN)],
        dtype=torch.int32,
        device="cuda",
    )
    return idx_q, index_k, block_table, seq, nvb


def _view(token_major: bool, total_q: int):
    store = torch.full(
        (HEADS * NB * total_q + 4096,), -float("inf"), device="cuda", dtype=torch.float32
    )
    view = store[: HEADS * NB * total_q].view(HEADS, NB, total_q)
    return store, (m.max_score_token_major(view) if token_major else view)


def _selects(scores, nvb, rows):
    """{name: table} for every select the target can run at ``rows`` (FSEL at <= FSEL_MAX_ROWS, DSL at any)."""
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


def test_token_major_view_is_the_same_logical_tensor() -> None:
    store = torch.arange(2 * 300 * 8, dtype=torch.float32)
    bm = store.view(2, 300, 8)
    tm = m.max_score_token_major(bm)
    assert tm.shape == bm.shape and tm.data_ptr() == bm.data_ptr()
    assert tm.stride() == (300 * 8, 1, 300)
    assert torch.equal(tm.permute(0, 2, 1).contiguous().view(-1), store)


def test_default_on_and_gated_to_the_t2sc_selects() -> None:
    assert m.T2SC_TOKEN_MAJOR, "default on (kill switch TRTLLM_M31_DISABLE_T2SC_TOKEN_MAJOR)"
    for rows in sorted(m.INDEX_SCORE_T2SC_PATH.keys):
        assert m.t2sc_token_major(rows, 1, 1, NB)
        assert not m.t2sc_token_major(rows, 4, 1, NB), (
            "a head reduction reads the buffer block-major"
        )
        assert not m.t2sc_token_major(rows, 1, 1, 128), (
            "the built-in selector keeps the block-major view"
        )
        assert not m.t2sc_token_major(rows, 1, 1, 9217)
    assert not m.t2sc_token_major(max(m.INDEX_SCORE_T2SC_PATH.keys) + 1, 1, 1, NB)


@requires_sm100_family
@pytest.mark.parametrize("case", list(CASES))
def test_token_major_scores_and_tables_are_bitwise(case) -> None:
    seq_lens = CASES[case]
    requests = len(seq_lens)
    assert m.INDEX_SCORE_T2SC_PATH.covers(requests)
    split_k, early = m.index_score_t2sc_schedule(requests)
    idx_q, index_k, block_table, seq, nvb = _inputs(
        seq_lens, 1234 + requests * 7 + seq_lens[0] % 97
    )
    rows = requests * QLEN
    res = {}
    for tm in (False, True):
        store, view = _view(tm, rows)
        m3_index_score_t2sc(
            idx_q, index_k, block_table, seq, view.transpose(1, 2), QLEN, split_k, early
        )
        res[tm] = (store, view, _selects(view, nvb, rows))
    torch.cuda.synchronize()
    (bstore, bview, btab), (tstore, tview, ttab) = res[False], res[True]
    assert torch.equal(bview, tview), "the logical scores differ"
    n = HEADS * NB * rows
    assert torch.equal(tstore[:n].view(HEADS, rows, NB), bview.permute(0, 2, 1)), (
        "not backed token-major"
    )
    assert torch.equal(bstore[:n].view(HEADS, NB, rows), bview)
    ref = fsel.M3FselTopk().reference(bview, nvb, None, 0, 1)
    for name in btab:
        assert btab[name] is not None and ttab[name] is not None, name
        assert torch.equal(btab[name], ttab[name]), f"{name}: the top-16 tables differ"
        assert torch.equal(ttab[name], ref), f"{name}: the table is not the select contract's"
