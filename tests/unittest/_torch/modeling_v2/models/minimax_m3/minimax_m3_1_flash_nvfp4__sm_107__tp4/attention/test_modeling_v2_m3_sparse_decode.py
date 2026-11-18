# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the m3_sparse_decode entry (the target's Triton block-sparse decode and its routes)."""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__tp4.kernels import (  # noqa: E501
    m3_attn_lc,
    m3_report,
)
from tensorrt_llm._torch.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__tp4.ops.attention.m3_sparse_decode import (  # noqa: E501
    m3_sparse_decode as op,
)

__extra_import_path__ = ["../../../..", "../../ops/attention"]
import _m31_flash_nvfp4_cache as cache  # noqa: E402 — needs the path declared above
from _validating import assert_within_ulp, certifying, validating  # noqa: E402, F401

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7),
    reason="the sm_107 TP4 target runs on SM107 only",
)


TARGET = "minimax_m3_1_flash_nvfp4__sm_107__tp4"


def _build_step(*args, **kwargs):
    return cache.build_step(*args, target=TARGET, **kwargs)


# Band, in BF16 ulps (2^-8) of each [16, 128] head row's largest magnitude, and of the relative RMS. Every route
# computes from exact operands (E4M3 q and E2M1 x E4M3 K/V are exact in FP16), so the gap to the float64 reference
# is the BF16 rounding on both sides (one ulp of the row at most), the FP16 probabilities before PV, the split-K
# partials kept in FP32 and merged by their log2-sum-exp, and approximate exp2/log2: 2 ulps per element, the two
# roundings (each at most half an ulp) dominating the RMS, so half an ulp. Measured on GB300 over the cells: at most
# 1.80 ulps per element and 0.20 ulps RMS.
_ELEMENT_ULP = 2.0
_RMS_ULP = 0.5


def _triton_route(monkeypatch) -> None:
    """Route selected steps to the target's Triton producer + merge (this target takes the LC whenever it applies)."""
    from tensorrt_llm._torch.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__tp4.kernels import (  # noqa: E501
        m3_attn_lc,
    )

    monkeypatch.setattr(m3_attn_lc, "lc_applies", lambda *args, **kwargs: False)


def _short_lens(requests: int, g: torch.Generator) -> tuple:
    """Short contexts of 1..16 blocks of 128 tokens (the identity-route regime), a few exactly on a block edge."""
    lens = torch.randint(4, 16 * 128 + 1, (requests,), generator=g).tolist()
    if requests > 1:
        lens[0] = 128 * int(torch.randint(1, 17, (1,), generator=g))
    return tuple(lens)


@pytest.mark.parametrize("seed", range(2))
@pytest.mark.parametrize("requests", [1, 4])
def test_short_table_stress_is_bitwise_the_stock_decode(requests, seed, monkeypatch) -> None:
    """The fail-closed id / page guard leaves valid inputs alone: short contexts (1..16 blocks) at 1..4 requests x 4
    verify rows, the identity route and a random selected route with -1 tails past each token's blocks, write the
    stock decode's rows bit for bit."""
    _triton_route(monkeypatch)
    g = torch.Generator().manual_seed(1000 * requests + seed)
    for identity in (True, False):
        case = _build_step(_short_lens(requests, g), 4, identity=identity, seed=7 * seed + requests)
        target = cache.empty_output(case)
        stock = torch.empty_like(target)
        op(*cache.sparse_decode_args(case, target, identity))
        cache.stock(case, stock)
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
    case = _build_step((1500, 2048, 700), 4, identity=False, seed=303)
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
    output = cache.empty_output(case)
    op(*cache.sparse_decode_args(case, output, False))
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


@pytest.mark.parametrize("fault", ["stale_one_step", "random_ids"])
@pytest.mark.parametrize("requests", [1, 4])
def test_lc_identity_route_reads_its_own_lengths_not_the_table(
    requests, fault, monkeypatch
) -> None:
    """An identity step runs the long-context kernel, which takes block i for CTA i from the kv lengths it reads.
    Whatever the staged table holds (lagging a verify step, garbage), the rows are the LC's on the true
    identity table (the selected-table route) bit for bit, and in the band of the float64 reference."""
    lens = tuple(
        cache.EDGE_LENS[(requests * 5 + k) % len(cache.EDGE_LENS)] for k in range(requests)
    )
    case = _build_step(lens, 4, identity=True, seed=700 + requests)
    true_table = cache.identity_table(case.seq_lens, 4, case.topk.shape[0])
    staged = cache.staged_table(case, fault, seed=requests)
    rows = case.q.shape[0]
    target = cache.empty_output(case)
    selected = torch.empty_like(target)
    case_staged = SimpleNamespace(**{**vars(case), "topk": staged})
    case_true = SimpleNamespace(**{**vars(case), "topk": true_table})
    taken = []
    monkeypatch.setattr(m3_report, "taken", lambda path, rows, what="": taken.append(path))
    op(*cache.sparse_decode_args(case_staged, target, True))
    monkeypatch.undo()
    # The LC engagement, plus its one-division merge's at these rows (m3_attn_lc.one_div_applies).
    assert sorted(taken) == sorted(
        ["m3_sparse_decode_lc"]
        + (["sparse_decode_lc_one_div_merge"] if m3_attn_lc.one_div_applies(rows) else [])
    ), taken
    op(*cache.sparse_decode_args(case_true, selected, False))
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


@pytest.mark.parametrize("requests", [2, 3])
def test_lc_fails_closed_on_ids_past_the_row(requests) -> None:
    """A selected id past the token's row (a stale table, e.g. a padded row keeping a 500K request's n_valid) is
    skipped, as the Triton producer skips it: the rows equal those with that slot at -1, bit for bit, and stay finite
    even though the block-table tail it would reach points at a page whose block scales are NaN (0xFF)."""
    from tensorrt_llm._torch.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__tp4.kernels import (  # noqa: E501
        m3_prequant,
    )

    lens = (165521, 700, 3000)[:requests]
    case = _build_step(lens, 4, identity=False, seed=1300 + requests)
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
        out = cache.empty_output(case)
        op(*cache.sparse_decode_args(case, out, False, topk=table))
        outs.append(out)
    m3_prequant.SLOT.clear()
    torch.cuda.synchronize()
    assert torch.isfinite(outs[0].float()).all()
    assert torch.equal(outs[0].view(torch.int16), outs[1].view(torch.int16))
