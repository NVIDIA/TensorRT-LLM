# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the m3_sparse_decode entry of the MiniMax-M3.1 sm_107 tp2 target at its per-rank geometry: 32 query
heads over 2 KV heads, head_dim 128, FP8 q, a packed NVFP4 cache [pages, 2, 2, 128, 64] (``_m31_flash_nvfp4_cache``).

TP2WIDE_LC2: steps of at most 32 rows run the Triton block-sparse decode, never the C++ ``sparse_decode_lc`` op (LC2
takes selected tables at 33..128 rows only; its tests are ``test_modeling_v2_m3_sparse_decode_lc.py``). The tests check
that route at TP2 shapes of at most 32 rows: it never takes the LC (although ``lc_applies`` accepts the TP2 q), reports
its own engagement path, and writes the stock in-tree decode's rows (bit for bit on the
identity and short-table routes, as the target's Triton copy takes the stock split-K), within the band of the float64
reference. The entry's own ``reference`` mirrors one KV head (its TP4 lineage), so the band is checked against a float64
reference built per KV head. LC2_ALLROWS and LC2_IDENTITY (default ON: LC2 from one row and on identity steps) are
killed in this file.
"""

import zlib

import pytest
import torch

from tensorrt_llm._torch.modeling_v2._native import NoTarget

try:
    from tensorrt_llm._torch.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__tp2.kernels import (  # noqa: E501
        m3_attn_lc,
        m3_prequant,
        m3_report,
    )
    from tensorrt_llm._torch.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__tp2.kernels import (  # noqa: E501
        triton_sparse_decode as target_kernel,
    )
    from tensorrt_llm._torch.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__tp2.ops.attention.m3_sparse_decode import (  # noqa: E501
        m3_sparse_decode as op,
    )
    from tensorrt_llm._torch.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__tp2.ops.attention.m3_sparse_decode_lc import (  # noqa: E501
        sparse_attention_reference,
    )
except NoTarget as exc:
    pytest.skip(str(exc), allow_module_level=True)

__extra_import_path__ = ["../../../..", "../../ops/attention"]
import _m31_flash_nvfp4_cache as cache  # noqa: E402 — needs the path declared above
from _validating import assert_within_ulp  # noqa: E402

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7),
    reason="the sm_107 TP2 target runs on SM107 only",
)


TARGET = "minimax_m3_1_flash_nvfp4__sm_107__tp2"


def _build_step(*args, **kwargs):
    return cache.build_step(*args, target=TARGET, **kwargs)


# Band, in BF16 ulps (2^-8) of each [heads, 128] row's largest magnitude, and of the relative RMS (the TP4 entry's
# band: exact operands, BF16 rounding on both sides, FP16 probabilities, FP32 split-K partials merged by their
# log2-sum-exp, approximate exp2 / log2).
_ELEMENT_ULP = 2.0
_RMS_ULP = 0.5


# The entry's certified step shapes (op.CELLS' lengths and query lengths), rebuilt at the TP2 geometry, plus the K7
# verify rows (decode_query_len 8) of c1..c4 and a 165521-token request.
@pytest.fixture(autouse=True)
def _lc2_band_only(monkeypatch):
    """These tests cover the Triton route of at most 32 rows: LC2_ALLROWS (LC2 from one row) and LC2_IDENTITY (LC2 on
    identity steps) are turned off here; their defaults are tested with the LC2 kernel."""
    monkeypatch.setattr(target_kernel, "LC2_ALLROWS", False)
    monkeypatch.setattr(target_kernel, "LC2_IDENTITY", False)
    monkeypatch.setattr(target_kernel, "LC_KEYS", target_kernel.LC_KEYS - {"plain"})


_STEPS = [
    *((c.spec["seq_lens"], c.spec["decode_query_len"], c.spec["identity"]) for c in op.CELLS),
    ((165521,), 8, False),
    ((165521, 20000), 8, False),
    ((900, 2048, 130), 8, True),
    ((40000, 3000, 165521, 129), 8, False),
]


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


@pytest.mark.parametrize("step", _STEPS, ids=[f"{len(s[0])}req_q{s[1]}_{s[2]}" for s in _STEPS])
def test_triton_route_at_tp2_shapes(step, monkeypatch) -> None:
    """Every step runs the Triton route (never the LC op), reports the Triton route's engagement path, stays finite on
    padding rows and lies in the band of the float64 reference."""
    seq_lens, qlen, identity = step
    # The Triton route itself: K3 steps from 9 rows can opt into LC2 (TP2_LC2_SMALL, its own tests), so pin it off.
    monkeypatch.setattr(target_kernel, "TP2_LC2_SMALL", False)
    assert len(seq_lens) * qlen < target_kernel.lc_min_rows(qlen), (
        "this test drives the Triton band (<= 32 rows)"
    )
    case = _build_step(
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
    output = cache.empty_output(case)
    op(*cache.sparse_decode_args(case, output, identity))
    m3_prequant.SLOT.clear()
    torch.cuda.synchronize()
    assert not lc_calls, "the C++ long-context op ran at TP2"
    assert "m3_sparse_decode_lc" not in taken, taken
    assert target_kernel.LC_TRITON_PATH in taken, taken
    assert torch.isfinite(output.float()).all(), "padding rows must be zeros, not NaN"
    ref = _reference(case, case.topk, output)
    assert_within_ulp(output, ref, element_ulp=_ELEMENT_ULP, rms_ulp=_RMS_ULP)


@pytest.mark.parametrize("requests", [1, 2, 4])
@pytest.mark.parametrize("qlen", [4, 8])
def test_identity_route_is_bitwise_the_stock_decode(requests, qlen) -> None:
    """The target's Triton producer + split-K merge takes the stock in-tree decode's split-K and writes its rows bit for
    bit at the TP2 geometry (per-row producer: the multi-query producer needs one KV head)."""
    case = _build_step((1500, 2048, 900, 130)[:requests], qlen, identity=True, seed=60 + requests)
    target, stock = cache.empty_output(case), cache.empty_output(case)
    op(*cache.sparse_decode_args(case, target, True))
    cache.stock(case, stock)
    m3_prequant.SLOT.clear()
    torch.cuda.synchronize()
    assert torch.equal(target.view(torch.int16), stock.view(torch.int16)), case.seq_lens.tolist()


@pytest.mark.parametrize("fault", ["stale_one_step", "all_minus_one", "random_ids"])
@pytest.mark.parametrize("requests", [1, 4])
def test_identity_route_reads_its_own_lengths_not_the_table(requests, fault) -> None:
    """The Triton identity route derives block i for slot i from the kv lengths the kernel reads: a staged table that
    lags the lengths by a verify step, or is garbage, gives the stock decode's rows over the true identity selection,
    bit for bit, on both KV heads."""
    lens = tuple(
        cache.EDGE_LENS[(requests * 5 + k) % len(cache.EDGE_LENS)] for k in range(requests)
    )
    case = _build_step(lens, 4, identity=True, seed=400 + requests)
    true_table = cache.identity_table(case.seq_lens, 4, case.topk.shape[0])
    assert torch.equal(case.topk, true_table), (
        "build_step's identity table is the identity selection"
    )
    staged = cache.staged_table(case, fault, seed=requests)
    target, stock = cache.empty_output(case), cache.empty_output(case)
    cache.stock(case, stock)
    op(*cache.sparse_decode_args(case, target, True, topk=staged))
    m3_prequant.SLOT.clear()
    torch.cuda.synchronize()
    assert torch.equal(target.view(torch.int16), stock.view(torch.int16)), (
        fault,
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
    case = _build_step((1500, 2048, 700), 4, identity=False, seed=303 + head)
    tok, slot = 5, 2
    case.topk[head, tok, slot] = -1 if fault == "neg_in_window" else case.block_table.shape[1] + 7
    output = cache.empty_output(case)
    op(*cache.sparse_decode_args(case, output, False))
    m3_prequant.SLOT.clear()
    torch.cuda.synchronize()
    ref_topk = case.topk.clone()
    ref_topk[head, tok, slot] = -1
    assert torch.isfinite(output.float()).all()
    assert_within_ulp(
        output, _reference(case, ref_topk, output), element_ulp=_ELEMENT_ULP, rms_ulp=_RMS_ULP
    )
