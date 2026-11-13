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
from tensorrt_llm._torch._experimental.modeling_v2._native import NoTarget

try:
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__tp4.ops.attention.m3_index_score_t2sc import (  # noqa: E501
        PAGE,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__tp4.ops.attention.m3_index_score_t2sc import (  # noqa: E501
        m3_index_score_t2sc as op,
    )
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


# T2SC_TOKEN_MAJOR (opt-in): the scorer writes and the long-row select reads the selector buffer backed token-major.
# The logical [heads, blocks, tokens] values and the picks (ties included) are the block-major buffer's.
@pytest.mark.parametrize("ties", [False, True], ids=["random", "ties"])
@pytest.mark.parametrize("length", [20000, 165521, 330000, 600000, 1000000])
def test_token_major_selector_buffer_is_bitwise_the_block_major(length, ties) -> None:
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__tp4.kernels.long_select import (  # noqa: E501
        long_select_blocks,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__tp4.modeling import (  # noqa: E501
        max_score_token_major,
    )

    qlen = 8
    idx_q, index_k, block_table, seq, _ = _inputs((length,), qlen, 3 + length % 997)
    if ties:
        index_k = index_k.view(torch.uint8)[:, :, :1].expand(-1, -1, HEAD_DIM).clone()
        index_k = (index_k % 4).to(torch.float32).to(torch.float8_e4m3fn)
    width = 9216  # the worst-case block extent of a 1M-token engine
    store = torch.full(
        (HEADS * width * qlen + 64,), -float("inf"), device="cuda", dtype=torch.float32
    )
    bm = store[: HEADS * width * qlen].view(HEADS, width, qlen)
    tm = max_score_token_major(bm)
    assert (
        tm.shape == bm.shape
        and tm.data_ptr() == bm.data_ptr()
        and tm.stride() == (width * qlen, 1, width)
    )
    nb = (length + PAGE - 1) // PAGE
    nvb = torch.tensor([nb] * qlen, dtype=torch.int32, device="cuda")
    out = {}
    for name, buf in (("bm", bm), ("tm", tm)):
        buf.fill_(-float("inf"))
        op(idx_q, index_k, block_table, seq, buf.transpose(1, 2), qlen, 1272, True)
        table = long_select_blocks(buf, n_valid_blocks=nvb, init_blocks=1, local_blocks=2, use_pdl=False,
                                   early_trigger=False, trigger_phase=1, fast_tail=False, dsl=True)  # fmt: skip
        torch.cuda.synchronize()
        out[name] = (buf[:, :nb, :].contiguous().clone(), table.clone())
    assert torch.equal(out["bm"][0].view(torch.int32), out["tm"][0].view(torch.int32)), (
        "a score differs"
    )
    assert torch.equal(out["bm"][1], out["tm"][1]), "the selection differs"


# T2SC_QOWN (opt-in): Q in its own buffer, so the CTA's second block is issued at the grid wait. The blocks and the
# per-block code are the default's, so the scores are the default schedule's bit for bit at every length, request
# count and query length, through repeated launches and graph replays.
_QOWN_CASES = [
    ((129,), 4, 1272),
    ((5000,), 8, 1272),
    ((131072,), 8, 1272),
    ((165521,), 8, 1272),
    ((330000,), 8, 1272),
    ((450000,), 8, 1272),
    ((600000,), 8, 1272),
    ((1000000,), 8, 1272),
    ((600000,), 1, 1272),
    ((600000,), 3, 1272),
    ((400000, 70000), 4, 768),
    ((300, 262144), 4, 768),
]


def _kern():
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__tp4.kernels import (  # noqa: E501
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


@pytest.mark.parametrize("qlen", [1, 4, 8])
def test_qown_at_the_prestage_edge_is_bitwise_the_default(qlen) -> None:
    kern = _kern()
    for k in (1, 2, 1272, 2544, 2545, 3816):
        for length in (k * PAGE - qlen - 1, k * PAGE - qlen, k * PAGE - qlen + 1, k * PAGE + 1):
            idx_q, index_k, block_table, seq, stride = _inputs((length,), qlen, 31 + length)
            key = (idx_q.dtype, HEADS, qlen, HEAD_DIM, 1272, True)
            base = kern.IndexScoreRunner.compile_schedule(key)
            alt = kern.IndexScoreRunner.compile_schedule(key, qown=True)
            s_base, s_alt = _score(stride, idx_q.shape[0]), _score(stride, idx_q.shape[0])
            base(idx_q, index_k, block_table, s_base, seq)
            alt(idx_q, index_k, block_table, s_alt, seq)
            torch.cuda.synchronize()
            assert torch.equal(s_base.view(torch.int32), s_alt.view(torch.int32)), (qlen, length)


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


def test_qown_switch_compiles_the_qown_kernel() -> None:
    """With TRTLLM_M31_ENABLE_T2SC_QOWN=1 the runner's early keys are the qown kernel (run under that env too)."""
    kern = _kern()
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__tp4.m31_switches import (  # noqa: E501
        m31_enabled,
    )

    assert kern.T2SC_QOWN == m31_enabled("T2SC_QOWN")
