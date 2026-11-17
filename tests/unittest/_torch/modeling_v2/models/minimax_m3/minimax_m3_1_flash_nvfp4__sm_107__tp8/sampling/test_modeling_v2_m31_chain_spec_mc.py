# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for REJ_MC: the spec sampler's chain speculative sampling as
``mv2_minimax_m3::minimax_m31_chain_speculative_sampling_mc`` through ``ops/sampling/m31_chain_spec_mc``.

It must equal the stock ``torch.ops.trtllm.chain_speculative_sampling`` bit for bit, tokens and counts (realistic /
same / far / ties / onehot drafts and targets, K7, the slot-gathered form, adversarial rows of the v3 kernel).
Outside its contract the override declines."""

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__tp8.ops.sampling import (  # noqa: E501
    m31_chain_spec_mc as entry,
)

CASES = ("realistic", "same", "far", "ties", "onehot")
op = entry.m31_chain_spec_mc


def _gpu():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7):
        pytest.skip("REJ_MC is built for sm_107 here")
    assert entry.chain_spec_mc_op() is not None, (
        "this build has no minimax_m31_chain_speculative_sampling_mc"
    )
    assert hasattr(torch.ops.trtllm, "chain_speculative_sampling")
    pytest.importorskip("flashinfer.sampling")


def _probs(logits, top_p=0.95):
    """softmax then FlashInfer's top-p renorm, as the sampler feeds the rejection step."""
    import flashinfer.sampling

    v = logits.shape[-1]
    p = torch.softmax(logits, -1)
    renormed = flashinfer.sampling.top_p_renorm_probs(p.view(-1, v).contiguous(), top_p)
    return renormed.view(logits.shape).contiguous()


def _make(case, batch, k, v, seed):
    """The microbench generators."""
    g = torch.Generator(device="cuda").manual_seed(seed)

    def rl(shape, scale=3.0):
        return torch.randn(*shape, v, generator=g, device="cuda") * scale

    dl = rl((batch, k))
    if case == "realistic":  # independent draft / target
        tl = torch.cat([dl, rl((batch, 1))], 1) + rl((batch, k + 1), 1.0)
    elif case == "same":  # target == draft: all-accept and zero-residual paths
        tl = torch.cat([dl, rl((batch, 1))], 1)
    elif case == "far":  # target peaked elsewhere: early reject
        tl = rl((batch, k + 1))
        tl[:, :, :8] += 20
    elif case == "ties":
        dl = torch.round(dl * 2) / 2
        tl = torch.round(torch.cat([dl, rl((batch, 1))], 1) * 2) / 2
    elif case == "onehot":
        tl = torch.full((batch, k + 1, v), -1e4, device="cuda")
        tl[:, :, 1234] = 0
    else:
        raise ValueError(case)
    dp, tp = _probs(dl), _probs(tl)
    ids = torch.multinomial(dp.view(-1, v), 1, generator=g).view(batch, k).to(torch.int32)
    seeds = torch.randint(0, 2**31, (batch,), generator=g, device="cuda", dtype=torch.int64)
    offsets = torch.randint(0, 2**20, (batch,), generator=g, device="cuda", dtype=torch.int64)
    return dp, ids.contiguous(), tp, seeds, offsets


def _stock(*args):
    return torch.ops.trtllm.chain_speculative_sampling(*args)


def _same(got, ref, what):
    assert torch.equal(got[0], ref[0]), f"{what}: accepted tokens differ"
    assert torch.equal(got[1], ref[1]), f"{what}: token counts differ"


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:40] for c in op.CELLS])
def test_bitwise_stock_op(cell, case) -> None:
    """At each certified (batch, K, vocab): the override equals the stock op (tokens and counts), and the explicit
    entry call at ctas_per_row 1 / 4 / 8 / 16 / 32 / 64 gives the same result again (REJ v2: the grid size never changes
    the result)."""
    _gpu()
    batch, k, v = cell.spec["batch"], cell.spec["k"], cell.spec["v"]
    for seed in range(4):
        args = _make(case, batch, k, v, 1000 * seed + batch)
        got = entry.chain_spec_mc(*args)
        assert got is not None, "an in-contract input must be served"
        _same(got, _stock(*args), f"{case} batch={batch} seed={seed}")
        for ctas in (1, 4, 8, 16, 32, 64):
            if ctas > 1 and batch * ctas > entry.sm_count(args[0].device):
                continue
            _same(op(*args, entry.prebuild(args[0].device), ctas), got, f"ctas_per_row={ctas}")


def test_outside_the_contract_the_override_declines() -> None:
    """Non-fp32 probs, non-int32 draft tokens, non-int64 or mis-sized seed / offset, a vocab not divisible by 4, a
    misaligned or non-contiguous probs, mismatched shapes, CPU inputs or more requests than MAX_BATCH return None (the
    sampler runs the stock op)."""
    _gpu()
    dp, ids, tp, seeds, offsets = _make("realistic", 2, 3, 32768, 1)
    assert entry.chain_spec_mc(dp, ids, tp, seeds, offsets) is not None
    assert entry.chain_spec_mc(dp.double(), ids, tp, seeds, offsets) is None
    assert entry.chain_spec_mc(dp, ids, tp.half(), seeds, offsets) is None
    assert entry.chain_spec_mc(dp, ids.long(), tp, seeds, offsets) is None
    assert entry.chain_spec_mc(dp, ids, tp, seeds.int(), offsets) is None
    assert entry.chain_spec_mc(dp, ids, tp, seeds[:1], offsets) is None
    assert entry.chain_spec_mc(dp, ids, tp, seeds, offsets[:1]) is None
    assert entry.chain_spec_mc(dp, ids[:, :2].contiguous(), tp, seeds, offsets) is None
    assert entry.chain_spec_mc(dp, ids, tp[:, :3].contiguous(), seeds, offsets) is None
    assert entry.chain_spec_mc(dp[..., :-4], ids, tp[..., :-4], seeds, offsets) is None
    assert (
        entry.chain_spec_mc(
            dp[..., :-2].contiguous(), ids, tp[..., :-2].contiguous(), seeds, offsets
        )
        is None
    )
    flat = torch.empty(dp.numel() + 1, device="cuda")
    assert entry.chain_spec_mc(flat[1:].view(dp.shape), ids, tp, seeds, offsets) is None
    assert entry.chain_spec_mc(dp.cpu(), ids.cpu(), tp.cpu(), seeds.cpu(), offsets.cpu()) is None
    b = entry.MAX_BATCH + 1
    wide = _make("realistic", b, 1, 32768, 2)
    assert entry.chain_spec_mc(*wide) is None
    # More draft tokens than the kernel's shared acceptance probabilities hold.
    deep = _make("realistic", 1, entry.MAX_DRAFT + 1, 32768, 3)
    assert entry.chain_spec_mc(*deep) is None
    edge = _make("realistic", 1, entry.MAX_DRAFT, 32768, 4)
    _same(entry.chain_spec_mc(*edge), _stock(*edge), "k = MAX_DRAFT")
    assert entry.ctas_per_row(1, 148) == 64 and entry.ctas_per_row(2, 148) == 64
    assert entry.ctas_per_row(4, 148) == 32 and entry.ctas_per_row(8, 148) == 16
    assert entry.ctas_per_row(16, 148) == 8 and entry.ctas_per_row(64, 148) == 2
    assert entry.ctas_per_row(200, 148) == 1
    # This device's own SM count keeps the largest grid co-resident.
    sms = entry.sm_count(torch.device("cuda"))
    assert entry.MAX_BATCH * entry.ctas_per_row(entry.MAX_BATCH, sms) <= sms


def _slot_case(case, batch, k, v, seed, slots=16, max_draft=8, v_pad=64):
    """A spec-metadata-like slot buffer [slots, max_draft, v + v_pad] with the request rows' draft probs at random
    distinct slots (the other cells hold junk), plus the gathered contiguous copy the sampler would pass."""
    dp, ids, tp, seeds, offs = _make(case, batch, k, v, seed)
    g = torch.Generator(device="cuda").manual_seed(seed + 1)
    buf = torch.rand(slots, max_draft, v + v_pad, generator=g, device="cuda")
    slot_ids = (
        torch.randperm(slots, generator=g, device="cuda")[:batch].to(torch.int64).contiguous()
    )
    buf[slot_ids, :k, :v] = dp
    gathered = buf[slot_ids, :k, :v].contiguous()
    assert torch.equal(gathered, dp)
    return buf, slot_ids, gathered, ids, tp, seeds, offs


def _slots_gpu():
    _gpu()
    assert entry.chain_spec_mc_slots_op() is not None, (
        "this build has no minimax_m31_chain_speculative_sampling_mc_slots"
    )


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("batch", [1, 2, 8])
def test_slots_bitwise_the_gathered_call(case, batch) -> None:
    """REJ_SLOTS: reading the draft rows in place by slot (padded draft length and vocab stride, junk in the other
    cells) gives exactly the gathered call's tokens and counts, and the stock op's."""
    _slots_gpu()
    k, v = 7, 200064
    for seed in range(3):
        buf, slot_ids, gathered, ids, tp, seeds, offs = _slot_case(
            case, batch, k, v, 100 * seed + batch
        )
        got = entry.chain_spec_mc_slots(buf, slot_ids, k, ids, tp, seeds, offs)
        assert got is not None, "an in-contract slot call must be served"
        _same(
            got,
            entry.chain_spec_mc(gathered, ids, tp, seeds, offs),
            f"{case} batch={batch} seed={seed} vs gathered",
        )
        _same(
            got,
            _stock(gathered, ids, tp, seeds, offs),
            f"{case} batch={batch} seed={seed} vs stock",
        )


def _v3_adversarial(case, batch, k, v, seed):
    """Rows aimed at v3's moved steps: the staged chain tails, both tile variants and the leader's tile walk."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    dp = torch.softmax(torch.randn(batch, k, v, generator=g, device="cuda") * 3.0, -1)
    bonus = torch.softmax(torch.randn(batch, 1, v, generator=g, device="cuda") * 3.0, -1)
    if (
        case == "allaccept"
    ):  # target == draft: every draft token accepted, the target-only (bonus) path
        tp = torch.cat([dp, bonus], 1)
    elif case == "lasttile":  # the residual has mass only in the last tile
        tp = torch.cat([dp, bonus], 1)
        tp[:, :, (v - 1) // 4096 * 4096 :] += 1e-3
        tp = tp / tp.sum(-1, keepdim=True)
    elif (
        case == "dyadic"
    ):  # power-of-two masses: tile sums and prefixes land exactly on the threshold grid
        tp = torch.zeros(batch, k + 1, v, device="cuda")
        tp[:, :, : 4 * 4096] = 1.0 / (4 * 4096)
        dp = torch.zeros(batch, k, v, device="cuda")
        dp[:, :, 4096 : 3 * 4096] = 1.0 / (2 * 4096)
    elif case == "zerotail":  # the target has no mass past the third tile
        tp = torch.softmax(torch.randn(batch, k + 1, v, generator=g, device="cuda") * 3.0, -1)
        tp[:, :, 3 * 4096 :] = 0.0
        tp = tp / tp.sum(-1, keepdim=True)
    else:
        raise ValueError(case)
    dp, tp = dp.contiguous(), tp.contiguous()
    ids = (
        torch.multinomial(dp.clamp_min(1e-12).view(-1, v), 1, generator=g)
        .view(batch, k)
        .to(torch.int32)
    )
    seeds = torch.randint(0, 2**31, (batch,), generator=g, device="cuda", dtype=torch.int64)
    offsets = torch.randint(0, 2**20, (batch,), generator=g, device="cuda", dtype=torch.int64)
    return dp, ids.contiguous(), tp, seeds, offsets


@pytest.mark.parametrize("case", ["allaccept", "lasttile", "dyadic", "zerotail"])
@pytest.mark.parametrize("v", [200064, 32768 + 4, 4096 * 3 + 8])
def test_v3_adversarial_rows_bitwise(case, v) -> None:
    """REJ v3 equals the stock op on rows aimed at its moved steps, at every grid (1..64 CTAs per request), batch 1 / 2
    / 8, K 7."""
    _gpu()
    if case == "dyadic" and v < 4 * 4096:
        pytest.skip("the dyadic rows need 4 tiles")
    for batch in (1, 2, 8):
        for seed in range(8):
            args = _v3_adversarial(case, batch, 7, v, 100 * seed + batch)
            ref = _stock(*args)
            for ctas in (1, 2, 8, 16, 32, 64):
                if ctas > 1 and batch * ctas > entry.sm_count(args[0].device):
                    continue
                _same(
                    op(*args, entry.prebuild(args[0].device), ctas),
                    ref,
                    f"{case} v={v} batch={batch} seed={seed} ctas={ctas}",
                )
