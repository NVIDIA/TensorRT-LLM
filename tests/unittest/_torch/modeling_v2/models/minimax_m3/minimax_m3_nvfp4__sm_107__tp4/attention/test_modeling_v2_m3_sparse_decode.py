# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the m3_sparse_decode entry (the target's Triton block-sparse decode and its routes)."""

import zlib
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_107__tp4.ops import (  # noqa: E501
    m3_report,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_107__tp4.ops.attention.m3_sparse_decode import (  # noqa: E501
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
    reason="the sm_107 TP4 target runs on SM107 only",
)

assert torch.cuda.is_available(), "m3_sparse_decode requires a CUDA device"

# Band, in BF16 ulps (2^-8) of each [16, 128] head row's largest magnitude, and of the relative RMS. Every route
# computes from exact operands (E4M3 q and E2M1 x E4M3 K/V are exact in FP16), so the gap to the float64 reference
# is the BF16 rounding on both sides (one ulp of the row at most), the FP16 probabilities before PV, the split-K
# partials kept in FP32 and merged by their log2-sum-exp, and approximate exp2/log2: 2 ulps per element, the two
# roundings (each at most half an ulp) dominating the RMS, so half an ulp. Measured on GB300 over the cells: at most
# 1.80 ulps per element and 0.20 ulps RMS.
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
    """On an identity table the identity routes (the producer, or the multi-query producer from 17 rows) and the
    selected-table route (the long-context kernel) compute the same attention."""
    case = build_step((2048, 130, 5, 0, 1777, 900, 2000, 0), 4, identity=True, seed=17)
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
def test_identity_route_reads_its_own_lengths_not_the_table(requests, fault, monkeypatch) -> None:
    """The Triton identity route (single / pair producer, up to 16 rows) derives block i for slot i from the kv lengths
    the kernel reads: a staged table that lags the lengths by a verify step (a 128-token edge crossed), or is garbage,
    gives the stock decode's rows over the true identity selection bit for bit. (The LC identity route has its own test
    below.)"""
    _triton_route(monkeypatch)
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


@pytest.mark.parametrize("requests", [1, 2, 4])
def test_identity_route_is_bitwise_the_stock_decode(requests, monkeypatch) -> None:
    """The target's Triton producer + split-K merge takes the stock in-tree decode's split-K at every row count
    (4 / 8 / 16 identity rows here) and so writes its rows bit for bit."""
    _triton_route(monkeypatch)
    case = build_step((1500, 2048, 900, 130)[:requests], 4, identity=True, seed=60 + requests)
    rows = case.q.shape[0]
    target = torch.empty(rows, NUM_HEADS, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    stock = torch.empty_like(target)
    _call(case, target, True)
    _stock(case, stock)
    torch.cuda.synchronize()
    assert torch.equal(target.view(torch.int16), stock.view(torch.int16)), case.seq_lens.tolist()


def _triton_route(monkeypatch) -> None:
    """Route selected steps to the target's Triton producer + merge (this target takes the LC whenever it applies)."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_107__tp4.ops import (
        m3_attn_lc,
    )

    monkeypatch.setattr(m3_attn_lc, "lc_applies", lambda *args, **kwargs: False)


def _short_lens(requests: int, g: torch.Generator) -> tuple:
    """Short contexts of 1..16 blocks of 128 tokens (the identity-route regime), a few exactly on a block edge."""
    lens = torch.randint(4, 16 * 128 + 1, (requests,), generator=g).tolist()
    if requests > 1:
        lens[0] = 128 * int(torch.randint(1, 17, (1,), generator=g))
    return tuple(lens)


@pytest.mark.parametrize("seed", range(12))
@pytest.mark.parametrize("requests", [1, 2, 3, 4])
def test_short_table_stress_is_bitwise_the_stock_decode(requests, seed, monkeypatch) -> None:
    """The fail-closed id / page guard leaves valid inputs alone: short contexts (1..16 blocks) at 1..4 requests x 4
    verify rows, the identity route and a random selected route with -1 tails past each token's blocks, write the
    stock decode's rows bit for bit."""
    _triton_route(monkeypatch)
    g = torch.Generator().manual_seed(1000 * requests + seed)
    for identity in (True, False):
        case = build_step(_short_lens(requests, g), 4, identity=identity, seed=7 * seed + requests)
        target = torch.empty(
            case.q.shape[0], NUM_HEADS, HEAD_DIM, dtype=torch.bfloat16, device="cuda"
        )
        stock = torch.empty_like(target)
        _call(case, target, identity)
        _stock(case, stock)
        torch.cuda.synchronize()
        assert torch.equal(target.view(torch.int16), stock.view(torch.int16)), (
            identity,
            case.seq_lens.tolist(),
        )


@pytest.mark.parametrize("fault", ["neg_in_window", "id_past_row", "page_out_of_pool"])
def test_invalid_slot_fails_closed(fault, monkeypatch) -> None:
    """A corrupt decode input (a -1 inside a token's block window, an id past the block-table row, or a block-table
    page outside the pool) is read as an absent block instead of a wild KV read: the rows equal the reference over the
    remaining blocks."""
    _triton_route(monkeypatch)
    case = build_step((1500, 2048, 700), 4, identity=False, seed=303)
    tok, slot = 5, 2
    if fault == "neg_in_window":
        case.topk[0, tok, slot] = -1
    elif fault == "id_past_row":
        case.topk[0, tok, slot] = case.block_table.shape[1] + 7
    else:
        req = tok // 4
        blk = int(case.topk[0, tok, slot])
        case.block_table = case.block_table.clone()
        case.block_table[req, blk] = case.k_paged.shape[0] + 1000
        # Every token of the request that selected that block loses it.
        hits = (case.topk[:, req * 4 : req * 4 + 4] == blk).nonzero().tolist()
    output = torch.empty(case.q.shape[0], NUM_HEADS, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    _call(case, output, False)
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


@pytest.mark.parametrize("fault", ["none", "stale_one_step", "all_minus_one", "random_ids"])
@pytest.mark.parametrize("requests", [1, 2, 3, 4])
def test_lc_identity_route_reads_its_own_lengths_not_the_table(
    requests, fault, monkeypatch
) -> None:
    """An identity step runs the long-context kernel, which takes block i for CTA i from the kv lengths it reads.
    Whatever the staged table holds (true, lagging a verify step, all -1, garbage), the rows are the LC's on the true
    identity table (the selected-table route) bit for bit, and in the band of the float64 reference."""
    lens = tuple(_EDGE_LENS[(requests * 5 + k) % len(_EDGE_LENS)] for k in range(requests))
    case = build_step(lens, 4, identity=True, seed=700 + requests)
    heads = case.topk.shape[0]
    true_table = _identity_table(case.seq_lens, 4, heads)
    if fault == "none":
        staged = true_table.clone()
    elif fault == "stale_one_step":
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
    selected = torch.empty_like(target)
    case_staged = SimpleNamespace(**{**vars(case), "topk": staged})
    case_true = SimpleNamespace(**{**vars(case), "topk": true_table})
    taken = []
    monkeypatch.setattr(m3_report, "taken", lambda path, rows, what="": taken.append(path))
    _call(case_staged, target, True)
    monkeypatch.undo()
    # The LC engagement, plus its one-division merge's.
    assert sorted(taken) == sorted(["m3_sparse_decode_lc", "sparse_decode_lc_one_div_merge"]), taken
    _call(case_true, selected, False)
    torch.cuda.synchronize()
    assert torch.equal(target.view(torch.int16), selected.view(torch.int16)), (
        fault,
        case.seq_lens.tolist(),
    )
    ref = op.reference(
        case.q,
        case.k_paged,
        case.v_paged,
        true_table,
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
        False,
    )
    assert_within_ulp(target, ref, element_ulp=_ELEMENT_ULP, rms_ulp=_RMS_ULP)


@pytest.mark.parametrize("requests", [1, 2])
def test_lc_identity_step_reads_q_after_the_wait(requests, monkeypatch) -> None:
    """An identity step has no select, so the LC must read q after its grid wait: the op gets identity_step (and the
    device identity selection) on every identity step."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_107__tp4.ops import (  # noqa: E501
        m3_attn_lc,
        m3_prequant,
    )

    lens = tuple(_EDGE_LENS[(requests * 3 + k) % len(_EDGE_LENS)] for k in range(requests))
    case = build_step(lens, 4, identity=True, seed=900 + requests)
    true_table = _identity_table(case.seq_lens, 4, case.topk.shape[0])
    case_true = SimpleNamespace(**{**vars(case), "topk": true_table})
    rows = case.q.shape[0]
    seen = []
    real = m3_attn_lc.m3_sparse_decode_lc

    def spy(*args, **kwargs):
        seen.append((kwargs.get("identity", False), kwargs.get("identity_step", False)))
        return real(*args, **kwargs)

    monkeypatch.setattr(m3_attn_lc, "m3_sparse_decode_lc", spy)
    out = torch.empty(rows, NUM_HEADS, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    _call(case_true, out, True)
    m3_prequant.SLOT.clear()
    torch.cuda.synchronize()
    assert seen[0] == (True, True), seen
    assert bool(torch.isfinite(out.float()).all())


@pytest.mark.parametrize("requests", [2, 3])
def test_lc_fails_closed_on_ids_past_the_row(requests) -> None:
    """A selected id past the token's row (a stale table, e.g. a padded row keeping a 500K request's n_valid) is
    skipped, as the Triton producer skips it: the rows equal those with that slot at -1, bit for bit, and stay finite
    even though the block-table tail it would reach points at a page whose block scales are NaN (0xFF)."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_107__tp4.ops import (  # noqa: E501
        m3_prequant,
    )

    lens = (165521, 700, 3000)[:requests]
    case = build_step(lens, 4, identity=False, seed=1300 + requests)
    dql = case.decode_query_len
    good = case.topk.clone()
    bad = case.topk.clone()
    for tok in range(case.q.shape[0]):
        req, qoff = divmod(tok, dql)
        kv_len = max(int(case.seq_lens[req]) - dql + qoff + 1, 0)
        nblk = (kv_len + 127) // 128
        past = nblk + 1
        if 2 <= nblk < 16 and past < case.block_table.shape[1]:
            bad[:, tok, nblk - 1] = past
            good[:, tok, nblk - 1] = -1
            page = int(case.block_table[req, past])
            case.k_block_scale[page].fill_(0xFF)
            case.v_block_scale[page].fill_(0xFF)
    assert not torch.equal(bad, good)
    outs = []
    for table in (bad, good):
        out = torch.empty(case.q.shape[0], NUM_HEADS, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
        _call(SimpleNamespace(**{**vars(case), "topk": table}), out, False)
        outs.append(out)
    m3_prequant.SLOT.clear()
    torch.cuda.synchronize()
    assert torch.isfinite(outs[0].float()).all()
    assert torch.equal(outs[0].view(torch.int16), outs[1].view(torch.int16))
