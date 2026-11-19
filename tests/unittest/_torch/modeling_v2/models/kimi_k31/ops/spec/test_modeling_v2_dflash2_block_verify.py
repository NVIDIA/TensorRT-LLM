# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the Kimi K3.1 DFlash2 block-verification entry (CuTe DSL).

A sample of the certified cells is gated bitwise against the entry's reference (the family's torch
``dflash2_uniforms`` and ``dflash2_block_verify``): the same accept counts and the same bonus
tokens for the same seeds and offsets, over several draws per cell.
"""

import zlib

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.spec.dflash2_block_verify import (  # noqa: E501
    dflash2_block_verify_seeded as seeded_op,
)

__extra_import_path__ = ["../../../..", "../.."]
from _k31_utils import on_sm, sample_cells
from _validating import certifying  # noqa: E402,F401

pytestmark = on_sm(100, 103, 107, reason="the Kimi K3.1 DFlash2 kernels are certified on SM10x")

#: Draws per certified cell (fresh target rows, proposals and uniforms each); single-request
#: cells take more so their accept counts cover more than tau = 0.
_DRAWS = 4
_DRAWS_SINGLE = 24


def _build(spec, seed):
    """Target rows, a selector-like proposal over each slot's top-k candidates and its draws."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    n, g, vocab, k = spec["batch"], spec["drafts"], spec["vocab"], spec["top_k"]
    temp = spec["temperature"]
    logits = 3.0 * torch.randn(n, g + 1, vocab, device="cuda", generator=gen)
    probs = torch.softmax(logits / temp, dim=-1)
    if spec["top_p_floor"] > 0:
        probs = torch.where(probs < spec["top_p_floor"], torch.zeros_like(probs), probs)
        probs = probs / probs.sum(-1, keepdim=True)
    if spec["padding_rows"]:
        probs[: spec["padding_rows"]] = 1.0 / vocab
    if spec["degenerate_rows"]:
        # Requests 0, 2, ... get all-zero target rows; 1, 3, ... a NaN in one row past the
        # first. Neither leaves mass to draw the bonus from.
        for b in range(spec["degenerate_rows"]):
            if b % 2 == 0:
                probs[b] = 0.0
            else:
                probs[b, 1 + b % g, 1000 + 37 * b] = float("nan")
    draft_logits = logits[:, :g] + spec["closeness"] * torch.randn(
        n, g, vocab, device="cuda", generator=gen
    )
    cand = draft_logits.topk(k, dim=-1).indices
    q = torch.softmax(draft_logits.gather(-1, cand) / temp, dim=-1)
    pick = torch.multinomial(q.reshape(-1, k), 1, generator=gen).view(n, g)
    drafts = cand.gather(-1, pick.unsqueeze(-1)).squeeze(-1)
    if spec["stale"] > 0:
        # DFlash2ProposalStore.load's point mass for a request the walk did not draft.
        stale = torch.rand(n, device="cuda", generator=gen) < spec["stale"]
        cand = torch.where(stale[:, None, None], drafts[..., None].expand_as(cand), cand)
        lane0 = (torch.arange(k, device="cuda") == 0).float().expand_as(q)
        q = torch.where(stale[:, None, None], lane0, q)
    if spec["outside"] > 0:
        outside = torch.rand(n, g, device="cuda", generator=gen) < spec["outside"]
        drafts = torch.where(outside, (drafts + 7) % vocab, drafts)
    # The worker slices one [n, K + 1] uniform block: strided coins and final coins.
    uniforms = torch.rand(n, g + 1, device="cuda", generator=gen)
    return (
        probs.contiguous(),
        drafts.to(spec["draft_dtype"]),
        cand.to(spec["cand_dtype"]),
        q,
        uniforms[:, :g],
        uniforms[:, g],
    )


def _single_request_reference(entry, args, coin_args):
    """A single request: ``tau`` from the reference on it (torch's one-row sums are
    deterministic), the bonus from the reference on the request duplicated to two rows, where
    torch.cumsum takes its deterministic scan (the one the kernel reproduces); the duplicated
    batch's own sums may round its ``h`` differently, so its bonus counts only where its ``tau``
    agrees (``None`` otherwise)."""
    probs, drafts, cand, q, *_ = args
    tau, _ = entry.reference(*args)
    dup = [a.expand(2, *a.shape[1:]) for a in coin_args]
    dup_tau, dup_bonus = entry.reference(
        probs.expand(2, -1, -1).contiguous(),
        drafts.expand(2, -1),
        cand.expand(2, -1, -1),
        q.expand(2, -1, -1),
        *dup,
    )
    return tau, (dup_bonus[:1] if int(dup_tau[0]) == int(tau[0]) else None)


def _assert_equal(got, want, what="") -> None:
    for name, a, b in zip(("tau", "bonus"), got, want):
        if b is None:
            continue
        assert a.shape == b.shape and a.dtype == b.dtype, (name, a.shape, b.shape, a.dtype, b.dtype)
        bad = (a != b).nonzero().flatten().tolist()
        assert not bad, (
            f"{what} {name}: requests {bad[:8]} differ: {a[bad[:8]].tolist()} vs {b[bad[:8]].tolist()}"
        )


def _seeded_args(args, seed):
    """The cell's tensors with full-range int64 seeds and offsets in place of the coins."""
    gen = torch.Generator().manual_seed(seed)
    n = args[0].shape[0]
    seeds = torch.randint(-(2**63), 2**63 - 1, (n,), generator=gen, dtype=torch.int64)
    offsets = torch.randint(0, 2**40, (n,), generator=gen, dtype=torch.int64)
    return (*args[:4], seeds.cuda(), offsets.cuda())


def _seeded_reference(args):
    if args[0].shape[0] != 1:
        return seeded_op.reference(*args)
    return _single_request_reference(seeded_op, args, args[4:6])


CELLS = sample_cells(seeded_op.CELLS, size_key="batch")[:2]


@pytest.mark.parametrize("cell", CELLS, ids=[c.why[:48] for c in CELLS])
def test_seeded_cells(cell, certifying) -> None:  # noqa: F811
    certifying(seeded_op)
    base = zlib.crc32(cell.why.encode()) % 2**31
    for draw in range(_DRAWS if cell.spec["batch"] > 1 else _DRAWS_SINGLE):
        args = _seeded_args(_build(cell.spec, seed=base + draw), seed=base + draw)
        _assert_equal(seeded_op(*args), _seeded_reference(args), f"draw {draw}")
