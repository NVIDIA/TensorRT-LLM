# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DEP_OWNT (OWN-T, the block-owner sparse decode) on an SM107 GPU.

OWN-T's rows stay within the float64-reference band (relative 1e-2 over the sampled rows, 2e-2 per row and head) at
7-token verify steps of 1..56 requests over 130..40000-token contexts; it is deterministic, and a captured graph
replays the eager output bitwise; the rows of zero-length requests are finite zeros, also as the CUDA-graph padding
requests of the decode hook; and the hook runs OWN-T on the calls LC2 takes, its MXFP8 o_proj hand-off bitwise
trtllm::mxfp8_quantize of its rows (byte-0 groups included), and LC2 bitwise when off. A maximal block that only some
of an item's rows selected leaves the other rows bitwise unchanged, and OWN-T's union keeps LC2's selection rule
(slot order, slots past a row's blocks and ids past the request's last block do not change a bit). At the served
widths (128 and 384 rows) the hook's rows are bitwise the OWN-T op's, and a step it cannot take raises without
running a kernel or Triton. At the plan cells (88 / 128 requests at top-16 overlap 100 / 75 / 50 %, and 11 requests
of 256 tokens), where the main kernel's plan stages its candidates in the K tile and derives the union's block
weights, the rows stay within the band and repeated calls and a graph replay are bitwise the first call.
"""

import sys
import types
from typing import List, Optional

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__dep import (  # noqa: E501
    lc2_decode,
    merge_mxfp8,
    native_ops,
    pad_mask,
)


def _sm107() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() == (10, 7)


pytestmark = pytest.mark.skipif(not _sm107(), reason="SM107 (Rubin) only")

KV_HEADS, HEADS_PER_KV, HEAD_DIM, PAGE, TOPK, ROWS = 4, 16, 128, 128, 16, 8
WIDTH = KV_HEADS * HEADS_PER_KV * HEAD_DIM
CONTEXTS = (1000, 5000, 20000, 40000, 3000, 130)
#: The float64-reference band: the relative error over the sampled rows, and of each sampled (row, head).
REL_GLOBAL = 1e-2
REL_ROW_HEAD = 2e-2
_E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)


def _overlap_rows(topk: torch.Tensor, req: int, seq: int, k_swap: int, gen: torch.Generator):
    """The overlap tables for one request: per KV head a base of 15 blocks before the last; each row keeps the base
    blocks it can see, swaps ``k_swap`` of them for other random blocks and adds its own last block."""
    pool = max((seq + PAGE - 1) // PAGE - 1, 0)
    for h in range(KV_HEADS):
        base = torch.randperm(pool, generator=gen)[: TOPK - 1].tolist() if pool else []
        for r in range(ROWS):
            blocks = (max(seq - ROWS + r + 1, 0) + PAGE - 1) // PAGE
            if blocks <= 0:
                continue
            keep = [b for b in base if b < blocks - 1]
            if k_swap and keep:
                drop = set(torch.randperm(len(keep), generator=gen)[:k_swap].tolist())
                keep = [b for i, b in enumerate(keep) if i not in drop]
                cand = torch.randperm(max(blocks - 1, 1), generator=gen).tolist()
                fresh = [b for b in cand if b not in keep and b < blocks - 1]
                keep += fresh[: TOPK - 1 - len(keep)]
            chosen = sorted(set(keep[: TOPK - 1] + [blocks - 1]))[:TOPK]
            topk[h, req * ROWS + r, : len(chosen)] = torch.tensor(chosen, dtype=torch.int32)


def _step(reqs: int, seed: int, k_swap: Optional[int] = None, seqs: Optional[List[int]] = None):
    """The decode hook's arguments for one K7 verify step of ``reqs`` requests: the DEP cache's interleaved K / V page
    views of random packed NVFP4 with E4M3 block scales in [0.25, 2], each request's pages in a random order, and each
    row's top-16 table an ascending random choice of its causal blocks with a -1 tail (with ``k_swap``, the plan cells'
    overlap tables instead)."""
    if seqs is None:
        seqs = [CONTEXTS[(r + seed) % len(CONTEXTS)] for r in range(reqs)]
    pages = [(s + PAGE - 1) // PAGE for s in seqs]
    gen = torch.Generator().manual_seed(seed)
    dgen = torch.Generator(device="cuda").manual_seed(seed)
    perm = torch.randperm(sum(pages), generator=gen).to(torch.int32)
    table = torch.zeros((reqs, max(pages)), dtype=torch.int32)
    start = 0
    for r, n in enumerate(pages):
        table[r, :n] = perm[start : start + n]
        start += n
    rows = reqs * ROWS
    topk = torch.full((KV_HEADS, rows, TOPK), -1, dtype=torch.int32)
    for req, seq in enumerate(seqs):
        if k_swap is not None:
            _overlap_rows(topk, req, seq, k_swap, gen)
            continue
        for r in range(ROWS):
            blocks = (max(seq - ROWS + r + 1, 0) + PAGE - 1) // PAGE
            for h in range(KV_HEADS):
                chosen = torch.randperm(blocks, generator=gen)[:TOPK].sort().values
                topk[h, req * ROWS + r, : chosen.numel()] = chosen
    cache = torch.randint(
        0,
        256,
        (start, 2, KV_HEADS, PAGE, HEAD_DIM // 2),
        dtype=torch.uint8,
        device="cuda",
        generator=dgen,
    )
    scales = (
        (
            torch.rand((start, 2, KV_HEADS, PAGE, HEAD_DIM // 16), device="cuda", generator=dgen)
            * 1.75
            + 0.25
        )
        .to(torch.float8_e4m3fn)
        .view(torch.uint8)
    )
    q = (
        torch.randn((rows, KV_HEADS * HEADS_PER_KV, HEAD_DIM), device="cuda", generator=dgen) * 2.0
    ).to(torch.float8_e4m3fn)
    args = (
        q,
        cache[:, 0],
        cache[:, 1],
        topk.cuda(),
        table.cuda(),
        torch.tensor(seqs, dtype=torch.int32, device="cuda"),
    )
    kwargs = dict(
        sm_scale=HEAD_DIM**-0.5,
        decode_query_len=ROWS,
        k_block_scale=scales[:, 0],
        v_block_scale=scales[:, 1],
        k_global_scale=torch.tensor([0.05], device="cuda"),
        v_global_scale=torch.tensor([0.03], device="cuda"),
    )
    return args, kwargs


def _dequant(packed: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """[n, 64] packed E2M1 (low nibble = even dim) x [n, 8] E4M3 scales (one per 16 dims) -> float64 [n, 128]."""
    lut = torch.tensor(_E2M1, dtype=torch.float64, device=packed.device)
    vals = torch.stack([lut[(packed & 0xF).long()], lut[(packed >> 4).long()]], dim=-1)
    sc = scales.view(torch.float8_e4m3fn).to(torch.float64).repeat_interleave(16, dim=-1)
    return vals.reshape(packed.shape[0], HEAD_DIM) * sc


def _v_scales(page_scales: torch.Tensor) -> torch.Tensor:
    """One page's V scales [128, 8] in the cache's 4x4 token-quad order -> [token, column]."""
    flat = page_scales.reshape(-1)
    n = torch.arange(PAGE, device=flat.device).unsqueeze(1)
    c = torch.arange(HEAD_DIM // 16, device=flat.device).unsqueeze(0)
    return flat[(n >> 2) * 32 + (n & 3) + 4 * c]


def _band(out: torch.Tensor, args, kwargs, toks):
    """The relative float64-reference error of ``out`` over the rows ``toks`` (each row's attention over its selected
    keys, per KV head), and the worst one of a (row, head)."""
    q, k, v, topk, table, seq_lens = args
    q64 = q.to(torch.float64)
    kgs = float(kwargs["k_global_scale"].item())
    vgs = float(kwargs["v_global_scale"].item())
    ksc, vsc = kwargs["k_block_scale"], kwargs["v_block_scale"]
    o3 = out.view(out.shape[0], KV_HEADS * HEADS_PER_KV, HEAD_DIM)
    num = den = worst = 0.0
    for tok in toks:
        req, r = divmod(tok, ROWS)
        kv_len = max(int(seq_lens[req]) - ROWS + r + 1, 0)
        real = min(TOPK, (kv_len + PAGE - 1) // PAGE)
        for h in range(KV_HEADS):
            ks, vs = [], []
            for blk in [int(b) for b in topk[h, tok, :real].tolist() if b >= 0]:
                page = int(table[req, blk])
                n = min(PAGE, kv_len - blk * PAGE)
                ks.append(_dequant(k[page, h, :n], ksc[page, h, :n]))
                vs.append(_dequant(v[page, h, :n], _v_scales(vsc[page, h])[:n]))
            qq = q64[tok, h * HEADS_PER_KV : (h + 1) * HEADS_PER_KV]
            p = torch.softmax(qq @ (torch.cat(ks) * kgs).T * kwargs["sm_scale"], dim=-1)
            ref = p @ (torch.cat(vs) * vgs)
            diff = o3[tok, h * HEADS_PER_KV : (h + 1) * HEADS_PER_KV].double() - ref
            num += float((diff * diff).sum())
            den += float((ref * ref).sum())
            worst = max(worst, float((diff.norm(dim=-1) / ref.norm(dim=-1).clamp_min(1e-30)).max()))
    return (num / den) ** 0.5, worst


def _own_t(args, kwargs, out=None, seq_lens=None, fast_plan=True):
    """OWN-T straight through its op, into ``out`` ([T, 8192] BF16) or a new buffer."""
    q, k, v, topk, table, seqs = args
    if out is None:
        out = torch.empty((q.shape[0], WIDTH), dtype=torch.bfloat16, device="cuda")
    lc2_decode.own_t_op()(
        q.view(torch.uint8),
        k,
        v,
        kwargs["k_block_scale"],
        kwargs["v_block_scale"],
        kwargs["k_global_scale"],
        kwargs["v_global_scale"],
        topk,
        table,
        seqs if seq_lens is None else seq_lens,
        out,
        None,
        None,
        WIDTH // 128,
        ROWS,
        kwargs["sm_scale"],
        True,
        fast_plan,
    )
    return out


def _scale_positions(rows: int) -> torch.Tensor:
    """The flat byte positions of the step's own rows in the 128x4 swizzled scales of [rows, 8192]."""
    r = torch.arange(rows, device="cuda").view(-1, 1)
    k = torch.arange(WIDTH // 32, device="cuda").view(1, -1)
    return (r // 128) * 64 * 512 + (k // 4) * 512 + (r % 32) * 16 + ((r % 128) // 32) * 4 + k % 4


def _reserve_workspaces():
    lc2_decode.reserve_workspace(torch.device("cuda"))
    native_ops.minimax_m3_sm107_sparse_decode_own_t_prepare(
        torch.empty(0, dtype=torch.uint8, device="cuda"), lc2_decode.OWNT_MAX_ITEMS
    )


@pytest.fixture(scope="module", autouse=True)
def _workspaces():
    _reserve_workspaces()


def test_the_reserve_holds_the_widest_step(monkeypatch):
    monkeypatch.setattr(lc2_decode, "DEP_OWNT", True)
    reserved = lc2_decode.reserve_own_t_workspace(torch.device("cuda"))
    # At least the fp32 partial rows of 256 items: 128 query rows x head_dim 128 each.
    assert reserved >= lc2_decode.OWNT_MAX_ITEMS * 128 * HEAD_DIM * 4


@pytest.mark.parametrize("reqs", [1, 4, 11, 16, 33, 56])
def test_own_t_is_within_the_float64_band_and_deterministic(reqs):
    args, kwargs = _step(reqs, seed=reqs)
    out = _own_t(args, kwargs)
    again = _own_t(args, kwargs)
    torch.cuda.synchronize()
    assert torch.isfinite(out.float()).all()
    assert torch.equal(out.view(torch.int16), again.view(torch.int16))
    rows = reqs * ROWS
    rel, worst = _band(out, args, kwargs, sorted({0, 7, rows // 2, rows - 8, rows - 1}))
    assert rel <= REL_GLOBAL and worst <= REL_ROW_HEAD, (rel, worst)


#: The plan cells: (requests, top-16 overlap swaps, contexts or None for the default mix, seed).
_PLAN_CELLS = {
    "C88_s0_o100": (11, 0, None, 0),
    "C88_s2_o100": (11, 0, None, 2),
    "C88_s4_o100": (11, 0, None, 4),
    "C128_s0_o100": (16, 0, None, 0),
    "C128_s2_o100": (16, 0, None, 2),
    "C128_s4_o100": (16, 0, None, 4),
    "C88_s2_o75": (11, 4, None, 2),
    "C88_s2_o50": (11, 8, None, 2),
    "C128_s2_o75": (16, 4, None, 2),
    "C128_s2_o50": (16, 8, None, 2),
    "C88_fix256_o100": (11, 0, [256] * 11, 0),
}


def _plan_step(cell: str):
    reqs, k_swap, seqs, seed = _PLAN_CELLS[cell]
    return _step(reqs, seed=1000 + seed, k_swap=k_swap, seqs=seqs)


@pytest.mark.parametrize("cell", list(_PLAN_CELLS))
def test_own_t_at_the_plan_cells_is_within_the_band_and_bitwise_stable(cell):
    """The plan stages its candidates in the K tile, which the K transform overwrites before the first QK, and derives
    the union's fixed block weights: at every plan cell the rows are within the float64 band, and two more eager calls
    and a graph replay, each into a NaN-filled buffer, are bitwise the first call."""
    args, kwargs = _plan_step(cell)
    reqs = _PLAN_CELLS[cell][0]
    first = _own_t(args, kwargs)
    torch.cuda.synchronize()
    assert torch.isfinite(first.float()).all()
    rows = reqs * ROWS
    rel, worst = _band(first, args, kwargs, sorted({0, 7, rows // 2, rows - 8, rows - 1}))
    assert rel <= REL_GLOBAL and worst <= REL_ROW_HEAD, (rel, worst)
    out = torch.empty_like(first)
    for _ in range(2):
        out.fill_(float("nan"))
        _own_t(args, kwargs, out=out)
        torch.cuda.synchronize()
        assert torch.equal(out.view(torch.int16), first.view(torch.int16))
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        _own_t(args, kwargs, out=out)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _own_t(args, kwargs, out=out)
    out.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out.view(torch.int16), first.view(torch.int16))


def test_zero_length_requests_write_finite_zeros():
    args, kwargs = _step(12, seed=3)
    seq_lens = args[5].clone()
    seq_lens[-3:] = 0
    out = torch.full((12 * ROWS, WIDTH), float("nan"), dtype=torch.bfloat16, device="cuda")
    _own_t(args, kwargs, out=out, seq_lens=seq_lens)
    torch.cuda.synchronize()
    assert torch.isfinite(out.float()).all()
    assert not out[9 * ROWS :].any()
    rel, worst = _band(out, args, kwargs, [0, 7, 40, 9 * ROWS - 1])
    assert rel <= REL_GLOBAL and worst <= REL_ROW_HEAD, (rel, worst)


def test_own_t_replays_in_a_cuda_graph():
    args, kwargs = _step(16, seed=5)
    eager = _own_t(args, kwargs)
    out = torch.empty_like(eager)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        _own_t(args, kwargs, out=out)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _own_t(args, kwargs, out=out)
    out.zero_()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out.view(torch.int16), eager.view(torch.int16))


def test_the_hook_hands_o_proj_the_mxfp8_quantize_of_own_ts_rows(monkeypatch):
    """From the widest step down: the rows are bitwise OWN-T's without the hand-off, and the hand-off it publishes is
    bitwise trtllm::mxfp8_quantize of them (the E4M3 rows and the step's own scale bytes)."""
    monkeypatch.setattr(lc2_decode, "DEP_OWNT", True)
    monkeypatch.setattr(merge_mxfp8, "DEP_MERGE_MXFP8", True)
    monkeypatch.setattr(merge_mxfp8, "_BUFFERS", {})
    merge_mxfp8.reserve(torch.device("cuda", torch.cuda.current_device()))
    for reqs in (56, 33, 16, 11, 4, 1):
        rows = reqs * ROWS
        args, kwargs = _step(reqs, seed=reqs)
        out = torch.empty(
            (rows, KV_HEADS * HEADS_PER_KV, HEAD_DIM), dtype=torch.bfloat16, device="cuda"
        )
        lc2_decode.dep_lc2_decode(*args, output=out, **kwargs)
        out = out.view(rows, WIDTH)
        q8, sf = merge_mxfp8.SLOT.take(out)
        direct = _own_t(args, kwargs)
        ref_q, ref_sf = torch.ops.trtllm.mxfp8_quantize(out, True)
        torch.cuda.synchronize()
        assert torch.equal(out.view(torch.int16), direct.view(torch.int16)), rows
        assert torch.equal(q8.view(torch.uint8), ref_q.view(torch.uint8)), rows
        at = _scale_positions(rows)
        assert torch.equal(
            sf.view(torch.uint8).flatten()[at], ref_sf.view(torch.uint8).flatten()[at]
        ), rows


def test_the_hand_off_quantizes_a_byte_0_group_like_mxfp8_quantize(monkeypatch):
    """A V scale small enough that 32-value groups take UE8M0 byte 0 (0 < amax <= 448 * 2^-127): the hand-off is still
    bitwise trtllm::mxfp8_quantize, so those groups are zero codes, and no code is a NaN."""
    monkeypatch.setattr(lc2_decode, "DEP_OWNT", True)
    monkeypatch.setattr(merge_mxfp8, "DEP_MERGE_MXFP8", True)
    monkeypatch.setattr(merge_mxfp8, "_BUFFERS", {})
    merge_mxfp8.reserve(torch.device("cuda", torch.cuda.current_device()))
    args, kwargs = _step(11, seed=11)
    kwargs["v_global_scale"] = torch.full((1,), 3e-37, device="cuda")
    rows = 11 * ROWS
    out = torch.empty(
        (rows, KV_HEADS * HEADS_PER_KV, HEAD_DIM), dtype=torch.bfloat16, device="cuda"
    )
    lc2_decode.dep_lc2_decode(*args, output=out, **kwargs)
    out = out.view(rows, WIDTH)
    q8, sf = merge_mxfp8.SLOT.take(out)
    ref_q, ref_sf = torch.ops.trtllm.mxfp8_quantize(out, True)
    codes = q8.view(torch.uint8)
    assert torch.equal(codes, ref_q.view(torch.uint8))
    at = _scale_positions(rows)
    assert torch.equal(sf.view(torch.uint8).flatten()[at], ref_sf.view(torch.uint8).flatten()[at])
    amax = out.float().abs().view(rows, WIDTH // 32, 32).amax(-1)
    byte_0 = (amax > 0) & (amax <= 448 * 2.0**-127)
    assert byte_0.any(), "no group reached UE8M0 byte 0"
    assert not (codes.view(rows, WIDTH // 32, 32)[byte_0] & 0x7F).any()
    assert not ((codes & 0x7F) == 0x7F).any()


@pytest.mark.parametrize("reqs", [11, 4], ids=["split", "cluster"])
def test_the_kill_switch_restores_lc2_bitwise(monkeypatch, reqs):
    """With DEP_OWNT off the hook's rows are bitwise the LC2 op's on the same operands, on the split grid (88 rows)
    and the cluster grid (32 rows)."""
    monkeypatch.setattr(lc2_decode, "DEP_OWNT", False)
    monkeypatch.setattr(
        lc2_decode, "own_t_op", lambda: pytest.fail("TRTLLM_M31_DISABLE_DEP_OWNT=1 runs LC2")
    )
    monkeypatch.setattr(merge_mxfp8, "_BUFFERS", {})
    rows = reqs * ROWS
    args, kwargs = _step(reqs, seed=reqs)
    out = torch.empty(
        (rows, KV_HEADS * HEADS_PER_KV, HEAD_DIM), dtype=torch.bfloat16, device="cuda"
    )
    lc2_decode.dep_lc2_decode(*args, output=out, **kwargs)
    q, k, v, topk, table, seqs = args
    ref = torch.empty((rows, WIDTH), dtype=torch.bfloat16, device="cuda")
    lc2_decode.lc2_op()(
        q.view(torch.uint8),
        k,
        v,
        kwargs["k_block_scale"],
        kwargs["v_block_scale"],
        kwargs["k_global_scale"],
        kwargs["v_global_scale"],
        topk,
        table,
        seqs,
        ref,
        None,
        None,
        WIDTH // 128,
        ROWS,
        kwargs["sm_scale"],
        False,
        lc2_decode.DEP_LC2_PDL and rows >= lc2_decode.SPLIT_MIN_ROWS,
        False,
        None,
    )
    torch.cuda.synchronize()
    assert torch.equal(out.view(rows, WIDTH).view(torch.int16), ref.view(torch.int16))


def test_a_block_some_rows_did_not_select_leaves_their_rows_bitwise():
    """A union block that some rows of a (request, KV head) selected and others did not gets maximal K / V: the rows
    that did not select it are bitwise unchanged (their scores for it are unused and their P is zero), and the rows
    that did change."""
    args, kwargs = _step(11, seed=21)
    q, k, v, topk, table, seqs = args
    base = _own_t(args, kwargs).clone()
    picks = topk.cpu()
    found = None
    for req in range(11):
        for h in range(KV_HEADS):
            sel = [
                {int(b) for b in picks[h, req * ROWS + r].tolist() if b >= 0} for r in range(ROWS)
            ]
            for blk in sorted(set().union(*sel)):
                without = [r for r in range(ROWS) if blk not in sel[r]]
                if 0 < len(without) < ROWS:
                    found = (req, h, blk, without)
                    break
            if found is not None:
                break
        if found is not None:
            break
    assert found is not None, "no block that only some rows selected"
    req, h, blk, without = found
    page = int(table[req, blk])
    for t in (k, v):
        t[page, h] = 0x77
    for t in (kwargs["k_block_scale"], kwargs["v_block_scale"]):
        t[page, h] = 0x7E
    out = _own_t(args, kwargs)
    torch.cuda.synchronize()
    assert torch.isfinite(out.float()).all()
    heads = slice(h * HEADS_PER_KV, (h + 1) * HEADS_PER_KV)
    o3 = out.view(-1, KV_HEADS * HEADS_PER_KV, HEAD_DIM)
    b3 = base.view(-1, KV_HEADS * HEADS_PER_KV, HEAD_DIM)
    for r in range(ROWS):
        same = torch.equal(
            o3[req * ROWS + r, heads].view(torch.int16), b3[req * ROWS + r, heads].view(torch.int16)
        )
        assert same == (r in without), r


def test_the_union_keeps_lc2s_selection_rule():
    """OWN-T attends what LC2 attends: a row's slots below min(16, its blocks), with ids >= 0 below the request's
    block count. Reordering those slots, filling the slots past them with valid ids, and writing an id past the
    request's last block into a -1 slot leave the rows bitwise unchanged."""
    args, kwargs = _step(11, seed=31)
    q, k, v, topk, table, seqs = args
    lens = seqs.cpu().tolist()

    def real_slots(seq, r):
        return min(TOPK, (max(seq - ROWS + r + 1, 0) + PAGE - 1) // PAGE)

    # The reference table: each row drops its last real slot to -1.
    ref = topk.cpu().clone()
    for req, seq in enumerate(lens):
        for r in range(ROWS):
            real = real_slots(seq, r)
            if real > 1:
                ref[:, req * ROWS + r, real - 1] = -1
    base = _own_t((q, k, v, ref.cuda(), table, seqs), kwargs).clone()
    gen = torch.Generator().manual_seed(31)
    edited = ref.clone()
    counts = {"reordered": 0, "past_real": 0, "past_last": 0}
    for req, seq in enumerate(lens):
        blocks = (seq + PAGE - 1) // PAGE
        for r in range(ROWS):
            real = real_slots(seq, r)
            for h in range(KV_HEADS):
                row = edited[h, req * ROWS + r]
                if real > 1:
                    row[:real] = row[:real][torch.randperm(real, generator=gen)]
                    row[int(row[:real].eq(-1).nonzero()[0, 0])] = blocks + 3
                    counts["reordered"] += 1
                    counts["past_last"] += 1
                if real < TOPK:
                    row[real:] = 0
                    counts["past_real"] += 1
    assert all(counts.values()), counts
    out = _own_t((q, k, v, edited.cuda(), table, seqs), kwargs)
    torch.cuda.synchronize()
    assert torch.equal(out.view(torch.int16), base.view(torch.int16))


@pytest.mark.parametrize("bucket, real", [(12, 11), (16, 13), (12, 0)])
def test_the_hook_writes_zeros_for_the_padding_requests(monkeypatch, bucket, real):
    monkeypatch.setattr(lc2_decode, "DEP_OWNT", True)
    args, kwargs = _step(bucket, seed=bucket + real)
    state = pad_mask.PadMask()
    state.lc2_layers, state.requests = 1, torch.arange(bucket, device="cuda") >= real
    out = torch.full(
        (bucket * ROWS, KV_HEADS * HEADS_PER_KV, HEAD_DIM),
        float("nan"),
        dtype=torch.bfloat16,
        device="cuda",
    )
    lc2_decode.set_pad_rows(state)
    try:
        lc2_decode.dep_lc2_decode(*args, output=out, **kwargs)
    finally:
        lc2_decode.set_pad_rows(None)
    torch.cuda.synchronize()
    assert torch.isfinite(out.float()).all()
    assert not out[real * ROWS :].any()
    if real:
        rel, worst = _band(out, args, kwargs, sorted({0, real * ROWS // 2, real * ROWS - 1}))
        assert rel <= REL_GLOBAL and worst <= REL_ROW_HEAD, (rel, worst)


@pytest.mark.parametrize("reqs", [16, 48], ids=["128_rows", "384_rows"])
def test_the_hook_runs_own_t_bitwise_at_the_served_widths(monkeypatch, reqs):
    """Generation steps of 88 and 128 requests per rank (up to max_num_tokens 384) take OWN-T, bitwise its op."""
    monkeypatch.setattr(lc2_decode, "DEP_OWNT", True)
    monkeypatch.setattr(lc2_decode, "DEP_LC2_PDL", True)
    monkeypatch.setattr(lc2_decode, "_PAD_ROWS", None)
    monkeypatch.setattr(merge_mxfp8, "_BUFFERS", {})
    rows = reqs * ROWS
    args, kwargs = _step(reqs, seed=100 + reqs)
    out = torch.empty(
        (rows, KV_HEADS * HEADS_PER_KV, HEAD_DIM), dtype=torch.bfloat16, device="cuda"
    )
    lc2_decode.dep_lc2_decode(*args, output=out, **kwargs)
    ref = _own_t(args, kwargs)
    torch.cuda.synchronize()
    assert torch.equal(out.view(rows, WIDTH).view(torch.int16), ref.view(torch.int16))


def test_a_step_wider_than_the_reserve_raises_and_writes_nothing(monkeypatch):
    """57 requests (456 rows) exceed LC2's 448-row reserve: the hook raises, runs no kernel and has no Triton route."""
    fake = types.ModuleType(
        "tensorrt_llm._torch.attention.backends.sparse.minimax_m3.kernels.triton_sparse_decode"
    )
    fake.minimax_m3_sparse_attn_decode = lambda *a, **kw: pytest.fail(
        "the DEP hook has no Triton fallback"
    )
    monkeypatch.setitem(sys.modules, fake.__name__, fake)
    monkeypatch.setattr(lc2_decode, "DEP_OWNT", True)
    monkeypatch.setattr(
        lc2_decode, "lc2_op", lambda: lambda *a: pytest.fail("a declined step must not run LC2")
    )
    monkeypatch.setattr(
        lc2_decode, "own_t_op", lambda: lambda *a: pytest.fail("a declined step must not run OWN-T")
    )
    reqs = lc2_decode.MAX_ROWS // ROWS + 1
    args, kwargs = _step(reqs, seed=7)
    out = torch.full(
        (reqs * ROWS, KV_HEADS * HEADS_PER_KV, HEAD_DIM), 3.0, dtype=torch.bfloat16, device="cuda"
    )
    with pytest.raises(RuntimeError, match=r"456 rows \(outside 1\.\.448\)"):
        lc2_decode.dep_lc2_decode(*args, output=out, **kwargs)
    torch.cuda.synchronize()
    assert bool((out == 3.0).all())


@pytest.mark.parametrize("reqs", [1, 16, 56])
def test_the_fastplan_kill_switch_keeps_the_bits(reqs):
    """fast_plan=False (DEP_OWNT_FASTPLAN off) loads the union's block weights in the plan instead of deriving them:
    the same split, so the same rows bit for bit."""
    args, kwargs = _step(reqs, seed=100 + reqs)
    fast = _own_t(args, kwargs).clone()
    loaded = _own_t(args, kwargs, fast_plan=False)
    torch.cuda.synchronize()
    assert torch.equal(fast.view(torch.int16), loaded.view(torch.int16))
