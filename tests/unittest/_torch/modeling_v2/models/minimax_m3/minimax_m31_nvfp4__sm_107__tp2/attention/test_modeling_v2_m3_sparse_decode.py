# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the m3_sparse_decode entry of the MiniMax-M3.1 sm_107 tp2 target at its per-rank geometry: 32 query
heads over 2 KV heads, head_dim 128, FP8 q, a packed NVFP4 cache [pages, 2, 2, 128, 64] (``_m3_nvfp4_cache``, the TP2
helper).

LC_TP2 (off by default) + TP2WIDE_LC2: steps of at most 32 rows run the Triton block-sparse decode, never the C++
``sparse_decode_lc`` op (LC2 takes selected tables at 33..128 rows only; its tests are
``test_modeling_v2_m3_sparse_decode_lc.py``). The tests check that route at TP2 shapes of at most 32 rows: it never
takes the LC (even with LC_TP2 forced on, now that ``lc_applies`` accepts the TP2 q), reports its own engagement path,
and writes the stock in-tree decode's rows (bit for bit on the
identity and short-table routes, as the target's Triton copy takes the stock split-K), within the band of the float64
reference. The entry's own ``reference`` mirrors one KV head (its TP4 lineage), so the band is checked against a float64
reference built per KV head. LC2_ALLROWS and LC2_IDENTITY (default ON: LC2 from one row and on identity steps) are
killed in this file.
"""

import zlib
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_107__tp2.kernels import (  # noqa: E501
    m3_attn_lc,
    m3_prequant,
    m3_report,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_107__tp2.kernels import (  # noqa: E501
    triton_sparse_decode as target_kernel,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_107__tp2.ops.attention.m3_sparse_decode import (  # noqa: E501
    m3_sparse_decode as op,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_107__tp2.ops.attention.m3_sparse_decode_lc import (  # noqa: E501
    sparse_attention_reference,
)
from tensorrt_llm._torch.attention.backends.sparse.minimax_m3.kernels.triton_sparse_decode import (
    minimax_m3_sparse_attn_decode as stock_decode,
)

__extra_import_path__ = ["../../../..", "."]
from _m3_nvfp4_cache import (  # noqa: E402 — needs the path declared above
    HEAD_DIM,
    NUM_HEADS,
    NUM_KV_HEADS,
    build_step,
)
from _validating import assert_within_ulp, validating  # noqa: E402

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7),
    reason="the sm_107 TP2 target runs on SM107 only",
)


@pytest.fixture(autouse=True)
def _lc2_band_only(monkeypatch):
    """These tests cover the Triton route of at most 32 rows: LC2_ALLROWS (LC2 from one row) and LC2_IDENTITY (LC2 on
    identity steps) are killed here; their defaults are tested in test_modeling_v2_m3_sparse_decode_lc.py."""
    monkeypatch.setattr(target_kernel, "LC2_ALLROWS", False)
    monkeypatch.setattr(target_kernel, "LC2_IDENTITY", False)
    monkeypatch.setattr(target_kernel, "LC_KEYS", target_kernel.LC_KEYS - {"plain"})


# Band, in BF16 ulps (2^-8) of each [heads, 128] row's largest magnitude, and of the relative RMS (the TP4 entry's
# band: exact operands, BF16 rounding on both sides, FP16 probabilities, FP32 split-K partials merged by their
# log2-sum-exp, approximate exp2 / log2).
_ELEMENT_ULP = 2.0
_RMS_ULP = 0.5

# The entry's certified step shapes (op.CELLS' lengths and query lengths), rebuilt at the TP2 geometry, plus the K7
# verify rows (decode_query_len 8) of c1..c4 and a 165521-token AgentX request.
_STEPS = [
    *((c.spec["seq_lens"], c.spec["decode_query_len"], c.spec["identity"]) for c in op.CELLS),
    ((165521,), 8, False),
    ((165521, 20000), 8, False),
    ((900, 2048, 130), 8, True),
    ((40000, 3000, 165521, 129), 8, False),
]


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


def _reference(case, topk, output):
    """Float64 reference per KV head: KV head ``h`` serves query heads ``[h * group, (h + 1) * group)`` with its own
    top-k row (the TP2 geometry: two KV heads)."""
    kv_heads = case.k_paged.shape[1]
    group = case.q.shape[1] // kv_heads
    heads = [
        sparse_attention_reference(
            case.q[:, h * group : (h + 1) * group],
            case.k_paged.view(torch.uint8),
            case.v_paged.view(torch.uint8),
            case.k_block_scale.view(torch.uint8),
            case.v_block_scale.view(torch.uint8),
            case.k_global_scale,
            case.v_global_scale,
            topk[h],
            case.block_table,
            case.seq_lens,
            case.decode_query_len,
            case.sm_scale,
            kv_head=h,
        )
        for h in range(kv_heads)
    ]
    return torch.cat(heads, dim=1).to(output.dtype)


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


def _empty(case):
    return torch.empty(case.q.shape[0], NUM_HEADS, HEAD_DIM, dtype=torch.bfloat16, device="cuda")


def test_tp2_geometry() -> None:
    case = build_step((3000,), 4, identity=False, seed=1)
    assert tuple(case.q.shape[1:]) == (32, 128) and NUM_KV_HEADS == 2
    assert int(case.k_paged.shape[1]) == 2 and int(case.topk.shape[0]) == 2


@pytest.mark.parametrize("step", _STEPS, ids=[f"{len(s[0])}req_q{s[1]}_{s[2]}" for s in _STEPS])
def test_triton_route_at_tp2_shapes(step, monkeypatch) -> None:
    """Every step runs the Triton route (never the LC op), reports the LC_TP2 Triton path, stays finite on padding
    rows and lies in the band of the float64 reference."""
    seq_lens, qlen, identity = step
    # The Triton route itself: K3 steps from 9 rows can opt into LC2 (TP2_LC2_SMALL, its own tests), so pin it off.
    monkeypatch.setattr(target_kernel, "TP2_LC2_SMALL", False)
    assert len(seq_lens) * qlen < target_kernel.lc_min_rows(qlen), (
        "this test drives the Triton band (<= 32 rows)"
    )
    case = build_step(
        seq_lens, qlen, identity=identity, seed=zlib.crc32(repr(step).encode()) % 2**31
    )
    taken = []
    monkeypatch.setattr(m3_report, "taken", lambda path, rows, what="": taken.append(path))
    lc_calls = []
    real = m3_attn_lc.m3_sparse_decode_lc
    monkeypatch.setattr(
        m3_attn_lc,
        "m3_sparse_decode_lc",
        lambda *a, **k: (lc_calls.append(1), real(*a, **k))[1],
    )
    output = _empty(case)
    _call(case, output, identity)
    m3_prequant.SLOT.clear()
    torch.cuda.synchronize()
    assert not lc_calls, "the C++ long-context op ran at TP2"
    assert "m3_sparse_decode_lc" not in taken, taken
    assert target_kernel.LC_TRITON_PATH in taken, taken
    assert torch.isfinite(output.float()).all(), "padding rows must be zeros, not NaN"
    ref = _reference(case, case.topk, output)
    assert_within_ulp(output, ref, element_ulp=_ELEMENT_ULP, rms_ulp=_RMS_ULP)


def test_rows_up_to_32_keep_the_triton_route_even_when_forced_on(monkeypatch) -> None:
    """``lc_applies`` now accepts the TP2 q [T, 32, 128] on two KV heads (LC2), but at most 32 rows the dispatch never
    asks it: with LC_TP2 forced on the step still runs the Triton route, bit for bit the default (LC_TP2 off) rows.
    (8 rows at the K3 verify: below the opt-in lc_min_rows(4) = 9.)"""
    case = build_step((165521, 3000), 4, identity=False, seed=31)
    assert 2 * 4 < target_kernel.lc_min_rows(4)
    assert m3_attn_lc.lc_applies(
        case.q,
        case.k_paged,
        case.v_paged,
        case.topk,
        case.k_block_scale,
        case.v_block_scale,
        _empty(case),
    )
    lc_calls = []
    real = m3_attn_lc.m3_sparse_decode_lc
    monkeypatch.setattr(
        m3_attn_lc,
        "m3_sparse_decode_lc",
        lambda *a, **k: (lc_calls.append(1), real(*a, **k))[1],
    )
    default = _empty(case)
    _call(case, default, False)
    monkeypatch.setattr(target_kernel, "LC_TP2", True)
    forced = _empty(case)
    _call(case, forced, False)
    m3_prequant.SLOT.clear()
    torch.cuda.synchronize()
    assert not lc_calls, "the C++ long-context op ran at 8 rows"
    assert torch.equal(default.view(torch.int16), forced.view(torch.int16))


@pytest.mark.parametrize("requests", [1, 2, 4])
@pytest.mark.parametrize("qlen", [4, 8])
def test_identity_route_is_bitwise_the_stock_decode(requests, qlen) -> None:
    """The target's Triton producer + split-K merge takes the stock in-tree decode's split-K and writes its rows bit for
    bit at the TP2 geometry (per-row producer: the multi-query producer needs one KV head)."""
    case = build_step((1500, 2048, 900, 130)[:requests], qlen, identity=True, seed=60 + requests)
    target, stock = _empty(case), _empty(case)
    _call(case, target, True)
    _stock(case, stock)
    m3_prequant.SLOT.clear()
    torch.cuda.synchronize()
    assert torch.equal(target.view(torch.int16), stock.view(torch.int16)), case.seq_lens.tolist()


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
    """The Triton identity route derives block i for slot i from the kv lengths the kernel reads: a staged table that
    lags the lengths by a verify step, or is garbage, gives the stock decode's rows over the true identity selection,
    bit for bit, on both KV heads."""
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
    target, stock = _empty(case), _empty(case)
    _stock(case, stock)
    _call(SimpleNamespace(**{**vars(case), "topk": staged}), target, True)
    m3_prequant.SLOT.clear()
    torch.cuda.synchronize()
    assert torch.equal(target.view(torch.int16), stock.view(torch.int16)), (
        fault,
        case.seq_lens.tolist(),
    )


def _short_lens(requests: int, g: torch.Generator) -> tuple:
    """Short contexts of 1..16 blocks of 128 tokens (the identity-route regime), a few exactly on a block edge."""
    lens = torch.randint(4, 16 * 128 + 1, (requests,), generator=g).tolist()
    if requests > 1:
        lens[0] = 128 * int(torch.randint(1, 17, (1,), generator=g))
    return tuple(lens)


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("requests", [1, 2, 3, 4])
def test_short_table_stress_is_bitwise_the_stock_decode(requests, seed, monkeypatch) -> None:
    """Short contexts (1..16 blocks) at 1..4 requests x 4 verify rows, the identity route and a random selected route
    with -1 tails past each token's blocks (each KV head its own selection): the stock decode's rows bit for bit.
    The Triton route's contract: 3 / 4 requests (12 / 16 rows) can opt into LC2 (TP2_LC2_SMALL, in the reference
    band, not bitwise; test_modeling_v2_m3_sparse_decode_lc.py), so it is pinned off here."""
    monkeypatch.setattr(target_kernel, "TP2_LC2_SMALL", False)
    g = torch.Generator().manual_seed(1000 * requests + seed)
    for identity in (True, False):
        case = build_step(_short_lens(requests, g), 4, identity=identity, seed=7 * seed + requests)
        target, stock = _empty(case), _empty(case)
        _call(case, target, identity)
        _stock(case, stock)
        m3_prequant.SLOT.clear()
        torch.cuda.synchronize()
        assert torch.equal(target.view(torch.int16), stock.view(torch.int16)), (
            identity,
            case.seq_lens.tolist(),
        )


@pytest.mark.parametrize("head", [0, 1])
@pytest.mark.parametrize("fault", ["neg_in_window", "id_past_row"])
def test_invalid_slot_fails_closed(fault, head, monkeypatch) -> None:
    """A corrupt selection slot on either KV head (a -1 inside a token's block window, or an id past the block-table
    row) is read as an absent block instead of a wild KV read: the rows equal the reference over the remaining
    blocks. The Triton route's contract (12 rows can opt into LC2 under TP2_LC2_SMALL, which does not mask an id
    past the row, as at its 33..128 rows; the selector never writes one), so TP2_LC2_SMALL is pinned off here."""
    monkeypatch.setattr(target_kernel, "TP2_LC2_SMALL", False)
    case = build_step((1500, 2048, 700), 4, identity=False, seed=303 + head)
    tok, slot = 5, 2
    case.topk[head, tok, slot] = -1 if fault == "neg_in_window" else case.block_table.shape[1] + 7
    output = _empty(case)
    _call(case, output, False)
    m3_prequant.SLOT.clear()
    torch.cuda.synchronize()
    ref_topk = case.topk.clone()
    ref_topk[head, tok, slot] = -1
    assert torch.isfinite(output.float()).all()
    assert_within_ulp(
        output, _reference(case, ref_topk, output), element_ulp=_ELEMENT_ULP, rms_ulp=_RMS_ULP
    )


def test_guard_refuses_a_padded_selection_slot() -> None:
    """The Triton producers index the block table with every one of a token's first min(16, blocks) ids, -1 included,
    on every KV head."""
    case = build_step((1500,), 1, identity=True, seed=23)
    case.topk[1, 0, 3] = -1
    with pytest.raises(AssertionError, match="blocks of the request"):
        with validating(op):
            _call(case, _empty(case), True)


def test_guard_refuses_a_false_identity_promise() -> None:
    """identity_blocks promises every token's table holds every block of its request in order."""
    case = build_step((1500, 1600, 1700, 1800), 4, identity=True, seed=29)
    case.topk[1, 5, :2] = case.topk[1, 5, :2].flip(0)
    with pytest.raises(AssertionError, match="identity_blocks promises"):
        with validating(op):
            _call(case, _empty(case), True)
