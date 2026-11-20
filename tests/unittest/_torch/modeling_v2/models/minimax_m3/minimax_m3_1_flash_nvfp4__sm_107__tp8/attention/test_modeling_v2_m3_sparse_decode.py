# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the m3_sparse_decode entry (the target's Triton block-sparse decode and its routes) at the TP8 heads.

Besides the band of every route around a float64 reference, the Triton merge's MXFP8 o_proj hand-off (published to
``m3_prequant.SLOT``) is held to ``torch.ops.trtllm.mxfp8_quantize(output, True)`` bit for bit: on identity steps and on
selected steps with the long-context range killed (the routes the o_proj consumes when LC8 does not run).
"""

import zlib
from types import SimpleNamespace

import pytest
import torch

import tensorrt_llm  # noqa: F401  (registers torch.ops.trtllm)
from tensorrt_llm._torch.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__tp8 import (
    modeling,  # noqa: E501
)
from tensorrt_llm._torch.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__tp8.kernels import (  # noqa: E501
    m3_prequant,
    triton_sparse_decode,
)
from tensorrt_llm._torch.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__tp8.ops.attention.m3_sparse_decode import (  # noqa: E501
    m3_sparse_decode as op,
)

__extra_import_path__ = ["../../../..", "../../ops/attention"]
import _m31_flash_nvfp4_cache as cache  # noqa: E402 — needs the path declared above
from _validating import assert_within_ulp, certifying, validating  # noqa: E402, F401

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7),
    reason="the sm_107 TP8 target runs on SM107 only",
)

TARGET = "minimax_m3_1_flash_nvfp4__sm_107__tp8"


def _build_step(*args, **kwargs):
    return cache.build_step(*args, target=TARGET, **kwargs)


# Band, in BF16 ulps (2^-8) of each [8, 128] row's largest magnitude, and of the relative RMS. Every route
# computes from exact operands (E4M3 q and E2M1 x E4M3 K/V are exact in FP16), so the gap to the float64 reference
# is the BF16 rounding on both sides (one ulp of the row at most), the FP16 probabilities before PV, the split-K
# partials kept in FP32 and merged by their log2-sum-exp, and approximate exp2/log2: 2 ulps per element, the two
# roundings (each at most half an ulp) dominating the RMS, so half an ulp. (GB300 TP4: at most 1.80 ulps per element
# and 0.20 ulps RMS over the cells.)
_ELEMENT_ULP = 2.0
_RMS_ULP = 0.5


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture by this name
    spec = cell.spec
    case = _build_step(
        spec["seq_lens"],
        spec["decode_query_len"],
        identity=spec["identity"],
        seed=zlib.crc32(cell.why.encode()) % 2**31,
    )
    output = cache.empty_output(case)
    certifying(op)
    op(*cache.sparse_decode_args(case, output, spec["identity"]))
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
    case = _build_step(seq_lens, 4, identity=True, seed=31 + len(seq_lens))
    output = cache.empty_output(case)
    op(*cache.sparse_decode_args(case, output, True))
    _mxfp8_handoff_matches_stock(output)


_IDENTITY_LENS = (1900, 2048, 700, 1500, 1200, 2000, 300, 1024)
_SELECTED_LENS = (9000, 40000, 20000, 5000, 7000, 12000, 30000, 3000)


@pytest.mark.parametrize("identity", [True, False], ids=["identity", "selected_lc_killed"])
@pytest.mark.parametrize("requests", [1, 2, 4, 8])
def test_triton_route_is_bitwise_the_stock_decode(requests, identity) -> None:
    """The target's per-row Triton producer + split-K merge writes the stock in-tree MSA decode's rows bit for bit: on
    SM107 it takes stock's split-K factor at every row count (4 / 8 / 16 / 32 verify rows here). An identity step of
    at least SPARSE_DECODE_MQ_MIN_TOKENS rows takes the multi-query producer (one KV head), which stock does not run:
    exact-class instead, in the band of the float64 reference and the same bits on every run."""
    lens = (_IDENTITY_LENS if identity else _SELECTED_LENS)[:requests]
    case = _build_step(lens, 4, identity=identity, seed=50 + requests + 100 * identity)
    rows = case.q.shape[0]
    multi_query = identity and rows >= modeling.SPARSE_DECODE_MQ_MIN_TOKENS
    target = cache.empty_output(case)
    again = torch.empty_like(target)
    stock = torch.empty_like(target)
    landed = triton_sparse_decode.LC8_KEYS
    triton_sparse_decode.LC8_KEYS = frozenset()
    try:
        op(*cache.sparse_decode_args(case, target, identity))
        op(*cache.sparse_decode_args(case, again, identity))
    finally:
        triton_sparse_decode.LC8_KEYS = landed
    torch.cuda.synchronize()
    assert torch.equal(target.view(torch.int16), again.view(torch.int16)), "run to run"
    if not multi_query:
        cache.stock(case, stock)
        torch.cuda.synchronize()
        assert torch.equal(target.view(torch.int16), stock.view(torch.int16))
        return
    ref = op.reference(
        case.q,
        case.k_paged,
        case.v_paged,
        case.topk,
        case.block_table,
        case.seq_lens,
        case.sm_scale,
        target,
        case.decode_query_len,
        None,
        case.k_block_scale,
        case.v_block_scale,
        case.k_global_scale,
        case.v_global_scale,
        None,
        True,
    )
    assert_within_ulp(target, ref, element_ulp=_ELEMENT_ULP, rms_ulp=_RMS_ULP)


@pytest.mark.parametrize("fault", ["neg_in_window", "id_past_row", "page_out_of_pool"])
def test_invalid_slot_fails_closed(fault) -> None:
    """A corrupt decode input (a -1 inside a token's block window, an id past the block-table row, or a block-table
    page outside the pool) is read as an absent block instead of a wild KV read: the rows equal the reference over the
    remaining blocks."""
    case = _build_step((1500, 2048, 700), 4, identity=False, seed=303)
    tok, slot = 5, 2
    if fault == "neg_in_window":
        case.topk[0, tok, slot] = -1
    elif fault == "id_past_row":
        case.topk[0, tok, slot] = case.block_table.shape[1] + 7
    else:
        req = tok // 4
        blk = int(case.topk[0, tok, slot])
        bad_page = case.k_paged.shape[0] + 1000
        case.block_table = case.block_table.clone()
        case.block_table[req, blk] = bad_page
        # Every token of the request that selected that block loses it.
        hits = (case.topk[:, req * 4 : req * 4 + 4] == blk).nonzero().tolist()
    output = cache.empty_output(case)
    landed = triton_sparse_decode.LC8_KEYS
    triton_sparse_decode.LC8_KEYS = frozenset()
    try:
        op(*cache.sparse_decode_args(case, output, False))
    finally:
        triton_sparse_decode.LC8_KEYS = landed
    torch.cuda.synchronize()
    # The reference walks each token's first min(16, blocks) entries and drops -1: the corrupt slots as -1.
    ref_topk = case.topk.clone()
    if fault == "page_out_of_pool":
        for h, t, k in hits:
            ref_topk[h, req * 4 + t, k] = -1
    else:
        ref_topk[0, tok, slot] = -1
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


@pytest.mark.parametrize("fault", ["stale_one_step", "all_minus_one", "random_ids"])
@pytest.mark.parametrize("requests", [1, 4])
def test_identity_route_reads_its_own_lengths_not_the_table(requests, fault) -> None:
    """The identity route derives block i for slot i from the kv lengths the kernel reads: a staged table that lags
    the lengths by a verify step (a 128-token edge crossed), or is garbage, gives the stock decode's rows over the
    true identity selection bit for bit."""
    lens = tuple(
        cache.EDGE_LENS[(requests * 5 + k) % len(cache.EDGE_LENS)] for k in range(requests)
    )
    case = _build_step(lens, 4, identity=True, seed=400 + requests)
    true_table = cache.identity_table(case.seq_lens, 4, case.topk.shape[0])
    assert torch.equal(case.topk, true_table), (
        "build_step's identity table is the identity selection"
    )
    staged = cache.staged_table(case, fault, seed=requests)
    target = cache.empty_output(case)
    stock = torch.empty_like(target)
    cache.stock(case, stock)
    case_staged = SimpleNamespace(**{**vars(case), "topk": staged})
    # The Triton identity route (triton_sparse_decode.LC8_IDENTITY off); the LC8 identity route has its own test below.
    triton_sparse_decode.LC8_IDENTITY = False
    try:
        op(*cache.sparse_decode_args(case_staged, target, True))
    finally:
        triton_sparse_decode.LC8_IDENTITY = True
    torch.cuda.synchronize()
    assert torch.equal(target.view(torch.int16), stock.view(torch.int16)), (
        fault,
        case.seq_lens.tolist(),
    )
