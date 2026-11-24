# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the ULL target's select_accept entry: every cell's selected tokens, accepted
counts, rewind and folded state (KDA replay records, kv_lens, the force counter, the forced-row
totals) bitwise against the reference; with real_rows, only the real rows forced and counted."""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Cell
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.spec.rubin_ull_select_accept import (  # noqa: E501
    select_accept as op,
)

__extra_import_path__ = ["../../../.."]
from _cells import certified_cells  # noqa: E402 — needs the path declared above
from _validating import validating  # noqa: E402 — needs the path declared above

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _helix8_layout(gens: int, k: int) -> dict:
    """The engine pair exchange's layout (``_PairExchange.layout``) of helix8's verify rows: TP8
    shards of 20,480, 2,048-wide local blocks, about 256 local programs."""
    rows, n_cols, block = gens * (k + 1), 20480, 2048
    splits = max(1, min(-(-n_cols // block), -(-256 // rows)))
    chunk = -(-(-(-n_cols // splits)) // block) * block
    slots = 8 * -(-n_cols // chunk)
    return dict(slots=slots, width=1 << (2 * slots - 1).bit_length())


def _helix8(why: str, gens: int, k: int = 5, **spec) -> Cell:
    base = dict(
        contexts=0,
        gens=gens,
        draft_len=k,
        force=2,
        kda=True,
        kv=True,
        matches=None,
        ties=False,
        count=True,
        real=None,
    )
    base.update(_helix8_layout(gens, k))
    base.update(spec)
    return Cell(why=f"helix8 {why}", spec=base)


TARGET_CELLS: dict[str, tuple[Cell, ...]] = {
    "kimi_k31_pro_nvfp4__sm_107__helix8": (
        *(
            _helix8(f"b{b} K 5, forced draw, KDA record, kv_lens prep, totals", b)
            for b in (2, 4, 9, 11, 13, 16)
        ),
        *(_helix8(f"b9 K {k}, forced draw", 9, k) for k in (1, 3, 7, 8)),
        _helix8("b9 unforced strict counts", 9, force=0, count=False),
        _helix8("b4 forced, placeholder rows", 4, real=(True, False, True, False)),
        _helix8("b9 with a dummy row", 9, dummy=3),
        # The ties planted at column 4000 need 64 slots or more of 64 columns: b4's 80.
        _helix8("b4 NaN and equal values across ranks", 4, ties=True),
    ),
}
CELLS = certified_cells(op, TARGET_CELLS)

CHUNK = 64
STATE_SLOTS = 40
POOL = 1 << 16


def _build(spec, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    nc, ng, k = spec["contexts"], spec["gens"], spec["draft_len"]
    slots, width = spec["slots"], spec["width"]
    bs, rows = nc + ng, nc + ng * (k + 1)
    logits = torch.randn(rows, slots * CHUNK, generator=g, device="cuda")
    if spec["ties"]:
        logits[0, 5] = logits[0, 4000] = logits[0].max() + 1
        logits[nc + 1, 300] = float("nan")
        logits[nc + 1, 70] = float("nan")
    split = logits.view(rows, slots, CHUNK)
    idx = split.argmax(-1)
    buf = torch.zeros(rows, width, device="cuda")
    buf[:, 0 : 2 * slots : 2] = split.gather(-1, idx[..., None])[..., 0]
    buf[:, 1 : 2 * slots : 2] = ((torch.arange(slots, device="cuda") * CHUNK)[None] + idx).float()
    target = logits.argmax(-1).to(torch.int32)
    gen_target = target[nc:].view(ng, k + 1)
    matches = spec["matches"] or tuple(
        int(v) for v in torch.randint(0, k + 1, (ng,), generator=g, device="cuda")
    )
    drafts = gen_target[:, :k].clone()
    for r, m in enumerate(matches):
        if m < k:
            drafts[r, m] = (drafts[r, m] + 1) % (slots * CHUNK)
    force = None
    if spec["force"] == 1:
        force = (3,)
    elif spec["force"] == 2:
        pool = torch.rand(POOL, generator=g, device="cuda")
        counter = torch.full((1,), 41, dtype=torch.int64, device="cuda")
        force = (3, 0.91, k + 1, pool, counter, 6007, 1009)
    kda = None
    if spec["kda"]:
        prev = (50 + torch.arange(STATE_SLOTS, device="cuda")).to(torch.int32)
        state_idx = torch.randperm(STATE_SLOTS, generator=g, device="cuda")[:bs].to(torch.int32)
        dummy = torch.zeros(bs + 3, dtype=torch.bool, device="cuda")
        if spec.get("dummy") is not None:
            dummy[nc + spec["dummy"]] = True
        kda = (prev, state_idx, dummy)
    kv = (100 + 7 * torch.arange(bs + 2, device="cuda")).to(torch.int32) if spec["kv"] else None
    count = torch.tensor([11, 29], dtype=torch.int64, device="cuda") if spec.get("count") else None
    real = None
    if spec.get("real") is not None:
        # Rows past the batch hold stale values the launch must not read.
        real = torch.zeros(ng + 3, dtype=torch.bool, device="cuda")
        real[:ng] = torch.tensor(spec["real"], dtype=torch.bool, device="cuda")
    return (buf, buf.stride(0), 2, slots), drafts.contiguous(), nc, bs, force, kda, kv, count, real


@pytest.mark.parametrize("cell", CELLS, ids=[c.why[:44] for c in CELLS])
def test_certified_cells(cell) -> None:
    pairs, drafts, nc, bs, force, kda, kv, count, real = _build(
        cell.spec, seed=zlib.crc32(cell.why.encode()) % 2**31
    )
    want = op.reference(pairs, drafts, nc, bs, force, kda, kv, count, real)
    with validating(op):
        accepted, num, rewind = op(
            pairs, drafts, nc, bs, force, kda_replay=kda, kv_lens=kv, count=count, real_rows=real
        )
    torch.cuda.synchronize()
    k = drafts.shape[-1]
    tokens = want["tokens"]
    assert torch.equal(accepted[:nc, 0], tokens[:nc])
    assert torch.equal(accepted[nc:bs], tokens[nc:].view(bs - nc, k + 1))
    assert torch.equal(num, want["num_accepted"])
    if kv is None:
        assert rewind is None
    else:
        assert torch.equal(rewind, want["rewind"]) and torch.equal(kv, want["kv_lens"])
    if kda is not None:
        assert torch.equal(kda[0], want["prev"])
    if want["counter"] is not None:
        assert torch.equal(force[4], want["counter"])
    if count is not None:
        assert torch.equal(count, want["count"])
    if cell.spec["matches"] and not cell.spec["force"]:
        expect = [min(m, k) + 1 for m in cell.spec["matches"]]
        assert num[nc:].tolist() == expect
    if cell.spec["matches"] and cell.spec["force"] and cell.spec.get("real") is not None:
        # A placeholder row keeps its strict count.
        for g, (m, r) in enumerate(zip(cell.spec["matches"], cell.spec["real"])):
            if not r:
                assert int(num[nc + g]) == min(m, k) + 1


def _forced_run(spec, real_rows, seed=7):
    """One forced launch on fresh operands; ``real_rows``: None, or a bool per generation row."""
    pairs, drafts, nc, bs, force, kda, kv, count, _ = _build(spec, seed=seed)
    real = None
    if real_rows is not None:
        real = torch.tensor(real_rows, dtype=torch.bool, device="cuda")
    accepted, num, rewind = op(
        pairs, drafts, nc, bs, force, kda_replay=kda, kv_lens=kv, count=count, real_rows=real
    )
    torch.cuda.synchronize()
    state = [accepted, num, rewind, kda[0] if kda else None, kv, count]
    if force is not None and len(force) > 1:
        state.append(force[4])
    return state


def _equal(a, b) -> bool:
    return all(
        (x is None and y is None) or (x is not None and y is not None and torch.equal(x, y))
        for x, y in zip(a, b)
    )


@pytest.mark.parametrize("force", [0, 1, 2])
def test_every_row_real_is_bitwise_the_launch_without_real_rows(force) -> None:
    """All rows real (and, unforced, any real_rows at all): every output and state as without."""
    spec = dict(CELLS[0].spec, gens=4, force=force, count=True, matches=(0, 7, 4, 1))
    base = _forced_run(spec, None)
    assert _equal(_forced_run(spec, (True,) * 4), base)
    if force == 0:
        # Natural acceptance never reads real_rows.
        assert _equal(_forced_run(spec, (False, True, False, False)), base)


@pytest.mark.parametrize("force", [1, 2])
def test_forced_totals_match_the_engine_rule(force) -> None:
    """The forced counts and the totals are the engine's for a mix of real and placeholder rows:
    ``torch.where(real, forced, strict)`` (``_apply_force_accepted_tokens``) and, for the totals,
    the real rows and their accepted tokens (``_count_forced``)."""
    spec = dict(CELLS[0].spec, gens=4, force=force, count=True, matches=(0, 7, 4, 1))
    real = (True, False, True, False)
    strict = _forced_run(dict(spec, force=0, count=False), None)[1]
    every = _forced_run(spec, None)
    mixed = _forced_run(spec, real)
    r = torch.tensor(real, device="cuda")
    assert torch.equal(mixed[1], torch.where(r, every[1], strict))
    totals = torch.tensor([11, 29], dtype=torch.int64, device="cuda")
    totals[0] += r.long().sum()
    totals[1] += torch.where(r, mixed[1].long(), 0).sum()
    assert torch.equal(mixed[5], totals)
    if force == 2:
        # The pool counter advances once per call whichever rows are forced.
        assert torch.equal(mixed[6], every[6])


def test_is_valid_refuses_a_state_index_past_the_records() -> None:
    pairs, drafts, nc, bs, force, kda, kv, _, _ = _build(CELLS[0].spec, seed=1)
    prev, state_idx, dummy = kda
    far = state_idx.clone()
    far[nc] = STATE_SLOTS
    with pytest.raises(ValueError, match="state index"):
        op.is_valid(pairs, drafts, nc, bs, force, (prev, far, dummy), kv)
