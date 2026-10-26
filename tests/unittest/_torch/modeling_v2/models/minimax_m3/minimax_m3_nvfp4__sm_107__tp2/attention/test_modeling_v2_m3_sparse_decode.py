# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the m3_sparse_decode entry (the target's Triton block-sparse decode and its routes) at the TP2 heads.

Besides the band of every route around a float64 reference, the Triton merge's MXFP8 o_proj hand-off (published to
``m3_prequant.SLOT``) is held to ``torch.ops.trtllm.mxfp8_quantize(output, True)`` bit for bit: on identity steps and on
selected steps with the long-context range killed (the routes the o_proj consumes when LC2 does not run).
"""

import zlib
from types import SimpleNamespace

import pytest
import torch

import tensorrt_llm  # noqa: F401  (registers torch.ops.trtllm)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_107__tp2 import (
    modeling,  # noqa: E501
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_107__tp2.ops import (  # noqa: E501
    m3_prequant,
    triton_sparse_decode,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_107__tp2.ops import (  # noqa: E501
    triton_sparse_decode as tsd,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_107__tp2.ops.attention.m3_sparse_decode import (  # noqa: E501
    m3_sparse_decode as op,
)
from tensorrt_llm._torch.attention.backends.sparse.minimax_m3.kernels.triton_sparse_decode import (
    minimax_m3_sparse_attn_decode as stock_decode,
)

__extra_import_path__ = ["../../../..", "."]
from _m3_nvfp4_cache import (  # noqa: E402 — needs the path declared above
    HEAD_DIM,
    NUM_HEADS,
    build_step,
)
from _validating import assert_within_ulp, certifying, validating  # noqa: E402, F401

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7),
    reason="the sm_107 TP2 target runs on SM107 only",
)

# Band, in BF16 ulps (2^-8) of each [32, 128] row's largest magnitude, and of the relative RMS. Every route
# computes from exact operands (E4M3 q and E2M1 x E4M3 K/V are exact in FP16), so the gap to the float64 reference
# is the BF16 rounding on both sides (one ulp of the row at most), the FP16 probabilities before PV, the split-K
# partials kept in FP32 and merged by their log2-sum-exp, and approximate exp2/log2: 2 ulps per element, the two
# roundings (each at most half an ulp) dominating the RMS, so half an ulp. (GB300 TP4: at most 1.80 ulps per element
# and 0.20 ulps RMS over the cells.)
_ELEMENT_ULP = 2.0
_RMS_ULP = 0.5


def _call(case, output, identity):
    return op(
        case.q,
        case.k_paged,
        case.v_paged,
        case.topk,
        case.block_table,
        case.seq_lens,
        case.sm_scale,
        output,
        case.decode_query_len,
        None,
        case.k_block_scale,
        case.v_block_scale,
        case.k_global_scale,
        case.v_global_scale,
        None,
        identity,
    )


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture by this name
    spec = cell.spec
    case = build_step(
        spec["seq_lens"],
        spec["decode_query_len"],
        identity=spec["identity"],
        seed=zlib.crc32(cell.why.encode()) % 2**31,
    )
    output = torch.empty(case.q.shape[0], NUM_HEADS, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    certifying(op)
    _call(case, output, spec["identity"])
    ref = op.reference(
        case.q,
        case.k_paged,
        case.v_paged,
        case.topk,
        case.block_table,
        case.seq_lens,
        case.sm_scale,
        output,
        case.decode_query_len,
        None,
        case.k_block_scale,
        case.v_block_scale,
        case.k_global_scale,
        case.v_global_scale,
        None,
        spec["identity"],
    )
    assert torch.isfinite(output.float()).all(), "padding rows must be zeros, not NaN"
    assert_within_ulp(output, ref, element_ulp=_ELEMENT_ULP, rms_ulp=_RMS_ULP)


def test_identity_flag_does_not_change_the_answer() -> None:
    """On an identity table the identity route (the per-row producer) and the selected-table route (the
    long-context kernel) compute the same attention."""
    case = build_step((2048, 130), 4, identity=True, seed=17)
    outs = []
    for identity in (True, False):
        output = torch.empty(
            case.q.shape[0], NUM_HEADS, HEAD_DIM, dtype=torch.bfloat16, device="cuda"
        )
        _call(case, output, identity)
        outs.append(output)
    assert_within_ulp(outs[0], outs[1], element_ulp=_ELEMENT_ULP, rms_ulp=_RMS_ULP)


def test_guard_refuses_a_padded_selection_slot() -> None:
    """The Triton producers index the block table with every one of a token's first min(16, blocks) ids,
    -1 included."""
    case = build_step((1500,), 1, identity=True, seed=23)
    case.topk[0, 0, 3] = -1
    output = torch.empty(1, NUM_HEADS, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(AssertionError, match="blocks of the request"):
        with validating(op):
            _call(case, output, True)


def test_guard_refuses_a_false_identity_promise() -> None:
    """identity_blocks lets the multi-query producer read one table for the whole request."""
    case = build_step((1500, 1600, 1700, 1800, 1500, 1600, 1700, 1800), 4, identity=True, seed=29)
    case.topk[0, 5, :2] = case.topk[0, 5, :2].flip(0)
    output = torch.empty(case.q.shape[0], NUM_HEADS, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(AssertionError, match="identity_blocks promises"):
        with validating(op):
            _call(case, output, True)


def _swizzled_scale_offsets(rows: int, width: int) -> torch.Tensor:
    """Offsets of rows [0, rows)'s UE8M0 scales in the 128x4-swizzled layout of mxfp8_quantize(x, True)."""
    k_sf = width // 32
    k_tiles = (k_sf + 3) // 4
    r = torch.arange(rows, device="cuda").view(-1, 1)
    k = torch.arange(k_sf, device="cuda").view(1, -1)
    offsets = (
        (r // 128) * k_tiles * 512 + (k // 4) * 512 + (r % 32) * 16 + ((r % 128) // 32) * 4 + k % 4
    )
    return offsets.reshape(-1)


def _mxfp8_handoff_matches_stock(output) -> None:
    """The real rows' E4M3 bytes and UE8M0 scales equal mxfp8_quantize's. The merge writes only the step's rows into
    its persistent 128-row scale tile; the padding rows' scales are never read by the o_proj and are not compared."""
    rows = output.shape[0]
    flat = output.view(rows, -1)
    taken = m3_prequant.SLOT.take(flat)
    assert taken is not None, "the route published no MXFP8 hand-off for its output"
    q8, sf = taken
    stock_q, stock_sf = torch.ops.trtllm.mxfp8_quantize(flat, True)
    assert torch.equal(
        q8.view(torch.uint8).reshape(rows, -1), stock_q.view(torch.uint8).reshape(rows, -1)
    )
    at = _swizzled_scale_offsets(rows, flat.shape[1])
    assert torch.equal(
        sf.view(torch.uint8).reshape(-1)[at], stock_sf.view(torch.uint8).reshape(-1)[at]
    )


@pytest.mark.parametrize("seq_lens", [(1900,), (2048, 0), (700, 1500)])
def test_identity_merge_mxfp8_handoff_is_stock_mxfp8_quantize(seq_lens) -> None:
    """The Triton merge writes the o_proj hand-off (E4M3 rows, 128x4-swizzled UE8M0 scales) as mxfp8_quantize."""
    case = build_step(seq_lens, 4, identity=True, seed=31 + len(seq_lens))
    output = torch.empty(case.q.shape[0], NUM_HEADS, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    _call(case, output, True)
    _mxfp8_handoff_matches_stock(output)


@pytest.mark.parametrize("seq_lens", [(40000,), (9000, 40000)])
def test_killed_long_context_route(seq_lens) -> None:
    """With SPARSE_DECODE_LC killed a selected table runs the Triton producer + merge: in the band of the reference,
    and its MXFP8 hand-off is mxfp8_quantize of the rows it wrote."""
    case = build_step(seq_lens, 4, identity=False, seed=41 + len(seq_lens))
    landed = triton_sparse_decode.LC_KEYS
    triton_sparse_decode.LC_KEYS = frozenset()
    try:
        output = torch.empty(
            case.q.shape[0], NUM_HEADS, HEAD_DIM, dtype=torch.bfloat16, device="cuda"
        )
        _call(case, output, False)
        _mxfp8_handoff_matches_stock(output)
    finally:
        triton_sparse_decode.LC_KEYS = landed
    ref = op.reference(
        case.q,
        case.k_paged,
        case.v_paged,
        case.topk,
        case.block_table,
        case.seq_lens,
        case.sm_scale,
        output,
        case.decode_query_len,
        None,
        case.k_block_scale,
        case.v_block_scale,
        case.k_global_scale,
        case.v_global_scale,
        None,
        False,
    )
    assert_within_ulp(output, ref, element_ulp=_ELEMENT_ULP, rms_ulp=_RMS_ULP)


def test_identity_route_is_the_certified_producer_range() -> None:
    """Every identity step at the TP2 heads takes the per-row producer, the key modeling.SPARSE_DECODE_PRODUCER
    covers (the multi-query producer needs one KV head)."""
    assert modeling.SPARSE_DECODE_PRODUCER.covers("per_row")
    assert not modeling.SPARSE_DECODE_PRODUCER.covers("multi_query")
    for cell in op.CELLS:
        keys = tuple(modeling._sparse_decode_producer(cell.spec))
        assert keys == (("per_row",) if cell.spec["identity"] else ()), (cell.why, keys)


_IDENTITY_LENS = (1900, 2048, 700, 1500, 1200, 2000, 300, 1024)
_SELECTED_LENS = (9000, 40000, 20000, 5000, 7000, 12000, 30000, 3000)


def _stock(case, output):
    stock_decode(
        case.q,
        case.k_paged,
        case.v_paged,
        case.topk,
        case.block_table,
        case.seq_lens,
        sm_scale=case.sm_scale,
        output=output,
        decode_query_len=case.decode_query_len,
        k_block_scale=case.k_block_scale,
        v_block_scale=case.v_block_scale,
        k_global_scale=case.k_global_scale,
        v_global_scale=case.v_global_scale,
    )


@pytest.mark.parametrize("identity", [True, False], ids=["identity", "selected_lc_killed"])
@pytest.mark.parametrize("requests", [1, 2, 4, 8])
def test_triton_route_is_bitwise_the_stock_decode(requests, identity) -> None:
    """The target's Triton producer + split-K merge writes the stock in-tree MSA decode's rows bit for bit: on SM107 it
    takes stock's split-K factor at every row count (4 / 8 / 16 / 32 verify rows here)."""
    lens = (_IDENTITY_LENS if identity else _SELECTED_LENS)[:requests]
    case = build_step(lens, 4, identity=identity, seed=50 + requests + 100 * identity)
    rows = case.q.shape[0]
    target = torch.empty(rows, NUM_HEADS, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    stock = torch.empty_like(target)
    landed = triton_sparse_decode.LC_KEYS
    triton_sparse_decode.LC_KEYS = frozenset()
    try:
        _call(case, target, identity)
    finally:
        triton_sparse_decode.LC_KEYS = landed
    _stock(case, stock)
    torch.cuda.synchronize()
    assert torch.equal(target.view(torch.int16), stock.view(torch.int16))


@pytest.mark.parametrize("requests", [33, 46, 48], ids=["132_rows", "184_rows", "192_rows"])
def test_plain_route_is_bitwise_the_stock_decode(requests) -> None:
    """Above MergeMxfp8.MAX_ROWS (128 rows) a selected table is outside SPARSE_DECODE_LC ("plain") and runs the
    Triton producer + merge: the stock decode's rows bit for bit, and no MXFP8 hand-off is published (the o_proj
    quantizes its BF16 input itself). 33..48 requests x 4 verify rows: the wide target's steps past one scale tile,
    up to WIDE_MAX_TOKENS."""
    case = build_step((3000,) * requests, 4, identity=False, seed=77)
    rows = case.q.shape[0]
    assert rows > tsd.MergeMxfp8.MAX_ROWS
    target = torch.empty(rows, NUM_HEADS, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    stock = torch.empty_like(target)
    _call(case, target, False)
    assert m3_prequant.SLOT.take(target.view(rows, -1)) is None
    _stock(case, stock)
    torch.cuda.synchronize()
    assert torch.equal(target.view(torch.int16), stock.view(torch.int16))


@pytest.mark.parametrize("identity", [True, False], ids=["triton_identity", "lc2_selected"])
def test_handoff_kill_publishes_nothing_and_keeps_the_rows(identity) -> None:
    """MERGE_MXFP8_HANDOFF False: no route publishes the MXFP8 hand-off (the o_proj then quantizes its BF16 input), and
    the attention rows are the same bits as with the hand-off."""
    case = build_step((1900,) if identity else (40000,), 4, identity=identity, seed=91 + identity)
    on = torch.empty(case.q.shape[0], NUM_HEADS, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    off = torch.empty_like(on)
    _call(case, on, identity)
    assert m3_prequant.SLOT.take(on.view(on.shape[0], -1)) is not None
    triton_sparse_decode.MERGE_MXFP8_HANDOFF = False
    try:
        _call(case, off, identity)
        assert m3_prequant.SLOT.take(off.view(off.shape[0], -1)) is None
    finally:
        triton_sparse_decode.MERGE_MXFP8_HANDOFF = True
    assert torch.equal(on.view(torch.int16), off.view(torch.int16))


def _short_lens(requests: int, g: torch.Generator) -> tuple:
    """Short contexts of 1..16 blocks of 128 tokens (the GSM / identity-route regime), a few exactly on a block edge."""
    lens = torch.randint(4, 16 * 128 + 1, (requests,), generator=g).tolist()
    if requests > 1:
        lens[0] = 128 * int(torch.randint(1, 17, (1,), generator=g))
    return tuple(lens)


@pytest.mark.parametrize("seed", range(12))
@pytest.mark.parametrize("requests", [1, 2, 3, 4])
def test_short_table_stress_is_bitwise_the_stock_decode(requests, seed) -> None:
    """Short contexts (1..16 blocks) at 1..4 requests x 4 verify rows, the identity route and a random selected route
    with -1 tails past each token's blocks: the target's Triton route writes the stock decode's rows bit for bit."""
    g = torch.Generator().manual_seed(1000 * requests + seed)
    for identity in (True, False):
        case = build_step(_short_lens(requests, g), 4, identity=identity, seed=7 * seed + requests)
        target = torch.empty(
            case.q.shape[0], NUM_HEADS, HEAD_DIM, dtype=torch.bfloat16, device="cuda"
        )
        stock = torch.empty_like(target)
        landed = triton_sparse_decode.LC_KEYS
        triton_sparse_decode.LC_KEYS = frozenset()
        try:
            _call(case, target, identity)
        finally:
            triton_sparse_decode.LC_KEYS = landed
        _stock(case, stock)
        torch.cuda.synchronize()
        assert torch.equal(target.view(torch.int16), stock.view(torch.int16)), (
            identity,
            case.seq_lens.tolist(),
        )


@pytest.mark.parametrize("fault", ["neg_in_window", "id_past_row", "page_out_of_pool"])
def test_invalid_slot_fails_closed(fault) -> None:
    """A corrupt decode input (a -1 inside a token's block window, an id past the block-table row, or a block-table
    page outside the pool) is read as an absent block instead of a wild KV read: the rows equal the reference over the
    remaining blocks."""
    case = build_step((1500, 2048, 700), 4, identity=False, seed=303)
    tok, slot = 5, 2
    if fault == "neg_in_window":
        case.topk[1, tok, slot] = -1
    elif fault == "id_past_row":
        case.topk[1, tok, slot] = case.block_table.shape[1] + 7
    else:
        req = tok // 4
        blk = int(case.topk[1, tok, slot])
        bad_page = case.k_paged.shape[0] + 1000
        case.block_table = case.block_table.clone()
        case.block_table[req, blk] = bad_page
        # Every token of the request that selected that block loses it.
        hits = (case.topk[:, req * 4 : req * 4 + 4] == blk).nonzero().tolist()
    output = torch.empty(case.q.shape[0], NUM_HEADS, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    landed = triton_sparse_decode.LC_KEYS
    triton_sparse_decode.LC_KEYS = frozenset()
    try:
        _call(case, output, False)
    finally:
        triton_sparse_decode.LC_KEYS = landed
    torch.cuda.synchronize()
    # The reference walks each token's first min(16, blocks) entries and drops -1: the corrupt slots as -1.
    ref_topk = case.topk.clone()
    if fault == "page_out_of_pool":
        for h, t, k in hits:
            ref_topk[h, req * 4 + t, k] = -1
    else:
        ref_topk[1, tok, slot] = -1
    ref = op.reference(
        case.q,
        case.k_paged,
        case.v_paged,
        ref_topk,
        case.block_table,
        case.seq_lens,
        case.sm_scale,
        output,
        case.decode_query_len,
        None,
        case.k_block_scale,
        case.v_block_scale,
        case.k_global_scale,
        case.v_global_scale,
        None,
        False,
    )
    assert torch.isfinite(output.float()).all()
    assert_within_ulp(output, ref, element_ulp=_ELEMENT_ULP, rms_ulp=_RMS_ULP)


def _identity_table(
    seq_lens: torch.Tensor, decode_query_len: int, heads: int, lag: int = 0
) -> torch.Tensor:
    """The identity selection of every verify token, staged from lengths `lag` tokens behind seq_lens."""
    seq = (seq_lens.cpu().to(torch.int64) - lag).clamp(min=0).repeat_interleave(decode_query_len)
    intra = torch.arange(seq.shape[0]) % decode_query_len
    n_valid = ((seq - decode_query_len + intra + 1).clamp(min=0) + 127) // 128
    ar = torch.arange(16)
    table = torch.where(ar[None, :] < n_valid[:, None], ar[None, :], -1).to(torch.int32)
    return table.unsqueeze(0).expand(heads, -1, -1).contiguous().cuda()


# Each crosses a 128-token edge within its four verify tokens or between the stale and the true lengths.
_EDGE_LENS = (1284, 1287, 1281, 129, 132, 2048, 1028, 5, 643, 1920, 256, 1)


@pytest.mark.parametrize("fault", ["stale_one_step", "all_minus_one", "random_ids"])
@pytest.mark.parametrize("requests", [1, 2, 3, 4])
def test_identity_route_reads_its_own_lengths_not_the_table(requests, fault) -> None:
    """The identity route derives block i for slot i from the kv lengths the kernel reads: a staged table that lags
    the lengths by a verify step (a 128-token edge crossed), or is garbage, gives the stock decode's rows over the
    true identity selection bit for bit."""
    lens = tuple(_EDGE_LENS[(requests * 5 + k) % len(_EDGE_LENS)] for k in range(requests))
    case = build_step(lens, 4, identity=True, seed=400 + requests)
    heads = case.topk.shape[0]
    true_table = _identity_table(case.seq_lens, 4, heads)
    assert torch.equal(case.topk, true_table), (
        "build_step's identity table is the identity selection"
    )
    if fault == "stale_one_step":
        staged = _identity_table(case.seq_lens, 4, heads, lag=4)
    elif fault == "all_minus_one":
        staged = torch.full_like(true_table, -1)
    else:
        g = torch.Generator().manual_seed(requests)
        staged = (
            torch.randint(-1, case.block_table.shape[1], true_table.shape, generator=g)
            .to(torch.int32)
            .cuda()
        )
    rows = case.q.shape[0]
    target = torch.empty(rows, NUM_HEADS, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    stock = torch.empty_like(target)
    _stock(case, stock)
    case_staged = SimpleNamespace(**{**vars(case), "topk": staged})
    _call(case_staged, target, True)
    torch.cuda.synchronize()
    assert torch.equal(target.view(torch.int16), stock.view(torch.int16)), (
        fault,
        case.seq_lens.tolist(),
    )
