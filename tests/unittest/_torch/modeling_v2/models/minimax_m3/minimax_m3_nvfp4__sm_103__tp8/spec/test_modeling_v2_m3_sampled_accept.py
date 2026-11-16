# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the m3_sampled_accept entry (fused temperature + top-p verify) and its workspace size."""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_103__tp8.ops.spec.m3_sampled_accept import (  # noqa: E501
    m3_sampled_accept as op,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_103__tp8.ops.spec.m3_sampled_accept_ws_words import (  # noqa: E501
    m3_sampled_accept_ws_words as ws_words,
)

__extra_import_path__ = ["../../../..", "."]
from _m3_spec_inputs import gen, seed_of  # noqa: E402 — needs the path declared above
from _validating import certifying, validating  # noqa: E402, F401

assert torch.cuda.is_available(), "m3_sampled_accept requires a CUDA device"

DRAFT_LEN = 3
#: Hot tokens of the distribution cells; every other logit sits 22 below the top (mass ~ 1e-5 of the total).
HEAD = 12
#: Draws per distribution cell, and the total-variation band against the reference distribution. With 12
#: categories and >= 4000 draws the expected TV of an exact sampler is ~0.02 (sd ~0.006); 0.06 is ~6 sd above it
#: while a wrong nucleus cut (one category too many or too few, >= 1 / 12 of the mass moved) lands at >= 0.08.
DRAWS = 4000
TV_BAND = 0.06


def _logits(spec, g, rows):
    vs = spec["vs"]
    if spec["check"] == "flat":
        return (torch.randn((rows, vs), generator=g, device="cuda") * 0.3).to(torch.bfloat16)
    if spec["check"] == "argmax":
        return (torch.randn((rows, vs), generator=g, device="cuda") * 3).to(torch.bfloat16)
    row = torch.full((vs,), -20.0, device="cuda")
    hot = torch.randperm(vs, generator=g, device="cuda")[:HEAD]
    row[hot] = torch.linspace(2.0, 0.2, HEAD, device="cuda")
    return row.to(torch.bfloat16).expand(rows, vs).contiguous()


def _case(spec, seed):
    g = gen(seed)
    nreq = spec["nreq"]
    rows = nreq * (DRAFT_LEN + 1)
    logits = _logits(spec, g, rows)
    words = ws_words(1)
    uc = torch.zeros(words, dtype=torch.int64, device="cuda")
    counters = torch.zeros(2, dtype=torch.int32, device="cuda")
    tokens = torch.zeros(48, dtype=torch.int32, device="cuda")
    extra = torch.zeros(1, dtype=torch.int32, device="cuda")
    temperatures = torch.full((rows,), spec["temperature"], dtype=torch.float32, device="cuda")
    top_ps = torch.full((rows,), spec["top_p"], dtype=torch.float32, device="cuda")
    seeds = torch.randint(0, 2**62, (rows,), generator=g, device="cuda", dtype=torch.int64)
    offsets = torch.zeros(rows, dtype=torch.int64, device="cuda")
    # Drafts drawn from the likeliest tokens, so some prefixes match.
    top = logits[0].float().topk(3).indices.to(torch.int32) + spec["vocab_offset"]
    draft = top[torch.randint(0, 3, (nreq, DRAFT_LEN), generator=g, device="cuda")].contiguous()
    batch_idx = torch.arange(nreq, dtype=torch.int32, device="cuda")
    acc = torch.full((nreq, DRAFT_LEN + 1), -7, dtype=torch.int32, device="cuda")
    num = torch.full((nreq,), -7, dtype=torch.int32, device="cuda")
    gather = torch.full((nreq,), -7, dtype=torch.int64, device="cuda")
    next_new = torch.full((nreq, DRAFT_LEN + 1), -7, dtype=torch.int32, device="cuda")
    force = spec.get("force", "none")
    force_total = 3 if force == "total" else 0
    force_vec = (torch.randint(1, DRAFT_LEN + 2, (nreq,), generator=g, device="cuda", dtype=torch.int32)
                 if force == "vec" else None)  # fmt: skip
    return [logits, spec["vocab_offset"], temperatures, top_ps, seeds, offsets, uc, uc.data_ptr(), counters, tokens,
            0, 1, draft, batch_idx, acc, num, gather, force_total, next_new, force_vec, extra, True]  # fmt: skip


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture by this name
    """Statistical for the draw, exact for everything derived from it.

    Every drawn token has nonzero reference probability (lies in the nucleus); the acceptance outputs equal the
    reference's strict acceptance of the drawn tokens, every call; the call counter advances once per call; and,
    for the distribution cells, the draws over many calls (fresh Philox offsets) are within TV_BAND of the
    reference distribution (see DRAWS / TV_BAND).
    """
    spec = cell.spec
    args = _case(spec, seed_of(cell.why))
    logits, offset, rows = args[0], spec["vocab_offset"], args[0].shape[0]
    certifying(op)
    ref = op.reference(*args)
    probs = ref["probs"]
    calls = 1 if spec["check"] == "argmax" else -(-DRAWS // rows)
    counts = torch.zeros(logits.shape[1], dtype=torch.float64, device="cuda")
    for c in range(calls):
        args[5].copy_(torch.arange(rows, device="cuda", dtype=torch.int64) + c * 1000)
        op(*args)
        drawn = args[9][:rows].clone()
        local = drawn.long() - offset
        assert bool(((local >= 0) & (local < logits.shape[1])).all()), "token outside the shard"
        assert bool((probs.gather(1, local[:, None]) > 0).all()), (
            "a drawn token outside the nucleus"
        )
        acc = ref["accept"](drawn)
        assert torch.equal(args[14], acc["accepted"])
        assert torch.equal(args[15], acc["num_accepted"])
        assert torch.equal(args[16], acc["gather_ids"])
        assert torch.equal(args[18], acc["next_new"])
        assert args[8].tolist() == [0, c + 1]
        counts.index_add_(0, local, torch.ones(rows, dtype=torch.float64, device="cuda"))
    if spec["check"] == "distribution":
        emp = counts / counts.sum()
        tv = 0.5 * (emp - probs[0].double()).abs().sum().item()
        assert tv <= TV_BAND, f"total variation {tv:.3f} against the reference nucleus distribution"


@pytest.mark.parametrize("top_p", [0.001, 0.0002])
def test_long_rejection_chains_stay_in_the_nucleus(top_p) -> None:
    """A tiny nucleus over a flat tail: rows that need more exchange rounds than the workspace has slots.

    Each round's candidate is drawn over the tokens heavier than the last one, so a row takes ~ln(1 / top_p) + 1
    rounds on average, with a tail past the 16 exchange slots (about 1% of the rows at top_p 0.001). Rounds reuse the
    slots cyclically, told apart by a lap-parity bit in the wire tags, and no round accepts unconditionally: every
    draw, however many rounds it took, lies in the reference nucleus. World 1 (one rank's 8 CTAs per row exchange
    through the same slot ring); the cross-rank path needs the multi-device job.
    """
    spec = dict(nreq=1, vs=25008, vocab_offset=25008, check="flat", top_p=top_p, temperature=1.0)
    args = _case(spec, seed_of(f"flat tail top_p {top_p}"))
    logits, rows = args[0], args[0].shape[0]
    probs = op.reference(*args)["probs"]
    calls = -(-DRAWS // rows)
    outside = torch.zeros((), dtype=torch.int64, device="cuda")
    for c in range(calls):
        args[5].copy_(torch.arange(rows, device="cuda", dtype=torch.int64) + c * 1000)
        op(*args)
        local = args[9][:rows].long() - spec["vocab_offset"]
        assert bool(((local >= 0) & (local < logits.shape[1])).all()), "token outside the shard"
        outside += (probs.gather(1, local[:, None]) == 0).sum()
        assert args[8].tolist() == [0, c + 1]
    draws = calls * rows
    rounds = 2.0 + args[20].item() / draws
    kept = int((probs > 0).sum(1).min().item())
    assert outside.item() == 0, (
        f"{outside.item()} of {draws} draws outside the nucleus (kept >= {kept} of {logits.shape[1]}, "
        f"{rounds:.2f} rounds per row on average)"
    )
    assert rounds > 4.0, (
        f"only {rounds:.2f} rounds per row: the rows no longer stress the slot ring"
    )


@pytest.mark.parametrize("cell", ws_words.CELLS, ids=[c.why[:44] for c in ws_words.CELLS])
def test_workspace_words(cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture by this name
    certifying(ws_words)
    world = cell.spec["world"]
    assert ws_words(world) == ws_words.reference(world)


def test_guard_refuses_top_k_rows() -> None:
    args = _case(op.CELLS[0].spec, 1)
    top_ks = torch.full((args[0].shape[0],), 20, dtype=torch.int32, device="cuda")
    with pytest.raises(AssertionError, match="top-k rows"):
        with validating(op):
            op(*args, top_ks=top_ks)


def test_guard_refuses_fp64_temperatures() -> None:
    args = _case(op.CELLS[0].spec, 2)
    args[2] = args[2].double()
    with pytest.raises(AssertionError, match="fp32 sampling parameters"):
        with validating(op):
            op(*args)
