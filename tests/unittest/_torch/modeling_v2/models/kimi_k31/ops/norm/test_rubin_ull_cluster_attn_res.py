# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the ULL target's cluster AttnRes entries: cluster_attn_res (tensor delta) and
cluster_attn_res_ring (the Lamport mailbox delta).

Every cell builds the site's ``PreparedAttnRes`` as the target does, launches it twice in a row
on fresh inputs (the ring cells refill the mailbox between: the launch re-arms it), and checks:
the new prefix row (and the saved block it writes) bitwise ``bf16(prefix + delta)``, the output and
the aux mixture within a few BF16 ulp of the FP32 reference, the mailbox re-armed to the sentinel,
and nothing written past the call's rows.
"""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Cell
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.norm.rubin_ull_cluster_attn_res import (  # noqa: E501
    cluster_attn_res,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.norm.rubin_ull_cluster_attn_res_ring import (  # noqa: E501
    cluster_attn_res_ring,
)

__extra_import_path__ = ["../../../.."]
from _cells import certified_cells  # noqa: E402 — needs the path declared above
from _validating import validating  # noqa: E402 — needs the path declared above

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="certified on SM10x",
)

TARGET_CELLS: dict[str, tuple[Cell, ...]] = {}
CELLS = certified_cells(cluster_attn_res, TARGET_CELLS)
RING_CELLS = certified_cells(cluster_attn_res_ring, TARGET_CELLS)

HIDDEN = 7168
EPS = (1e-5, 1e-5)
BLOCKS = 9  # snapshot slots the target allocates: ceil(93 / 12) + 1
CAP = 64  # the MoE tail output box's rows
SENTINEL_WORD = -0x80000000  # 0x80000000 as int32
POISON = -7.0


def _prepared_cls():
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kernels.vendor.vllm_mr8 import (  # noqa: E501
        module,
    )

    return module("cutedsl.attn_res.prepared").PreparedAttnRes


def _weights(seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    sel = (1 + 0.1 * torch.randn(HIDDEN, generator=g, device="cuda")).to(torch.bfloat16)
    proj = (0.05 * torch.randn(1, HIDDEN, generator=g, device="cuda")).to(torch.bfloat16)
    gamma = (1 + 0.1 * torch.randn(HIDDEN, generator=g, device="cuda")).to(torch.bfloat16)
    return sel, proj, gamma


def _rows(shape, seed, scale):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return (torch.randn(*shape, generator=g, device="cuda") * scale).to(torch.bfloat16)


def _operands(rows, seed):
    """prefix [T, 7168], the blocks view [T, BLOCKS, 7168] of a [BLOCKS, T, 7168] bank (the
    target's ``blocks.permute(1, 0, 2)``), and poisoned out / aux."""
    prefix = _rows((rows, HIDDEN), seed, 1.0)
    bank = _rows((BLOCKS, rows, HIDDEN), seed + 1, 1.0)
    out = torch.full((rows, HIDDEN), POISON, dtype=torch.bfloat16, device="cuda")
    aux = torch.full_like(out, POISON)
    return prefix, bank, out, aux


def _check(spec, got, ref, bank, bank_before, where):
    nb, has_delta, write, final = spec["key"]
    prefix, out, aux = got
    assert torch.equal(prefix.view(torch.int16), ref["prefix"].view(torch.int16)), (
        f"{where}: new prefix row is not bf16(prefix + delta)"
    )
    for b in range(BLOCKS):
        want = ref["prefix"] if b == write else bank_before[b]
        assert torch.equal(bank[b].view(torch.int16), want.view(torch.int16)), (
            f"{where}: saved block {b} " + ("not written" if b == write else "changed")
        )
    _within(out, ref["out"], 4, f"{where}: out")
    if aux is not None:
        _within(aux, ref["aux"], 2, f"{where}: aux")


def _within(out, ref, ulps, where):
    o, r = out.float(), ref.float()
    scale = r.abs().amax(dim=-1, keepdim=True).clamp_min(2.0**-100)
    ulp = torch.exp2(torch.floor(torch.log2(scale)) - 7)
    err = (o - r).abs()
    assert bool((err <= ulps * ulp).all()), f"{where}: max err {(err / ulp).max().item():.2f} ulp"


@pytest.mark.parametrize("cell", CELLS, ids=[c.why[:44] for c in CELLS])
def test_certified_cells(cell) -> None:
    spec = cell.spec
    seed = zlib.crc32(cell.why.encode()) % 2**31
    weights = _weights(seed)
    prepared = _prepared_cls()(weights, spec["key"], EPS)
    if spec["aux"]:
        prepared.bind_aux(mix=True)
    for launch in range(2):
        prefix, bank, out, aux = _operands(spec["rows"], seed + 10 * launch)
        aux = aux if spec["aux"] else None
        # The before sites pass the prefix itself as the delta (no add); the post sites the
        # reduced o-projection rows.
        has_delta = spec["key"][1]
        delta = _rows((spec["rows"], HIDDEN), seed + 10 * launch + 5, 0.5) if has_delta else prefix
        prefix0, bank0 = prefix.clone(), bank.clone()
        ref = cluster_attn_res.reference(
            prepared, delta.clone(), prefix0, bank0.permute(1, 0, 2), out, aux,
            weights=weights, eps=EPS,
        )  # fmt: skip
        with validating(cluster_attn_res):
            result = cluster_attn_res(prepared, delta, prefix, bank.permute(1, 0, 2), out, aux)
        torch.cuda.synchronize()
        assert result is out
        _check(spec, (prefix, out, aux), ref, bank, bank0, f"{cell.why}, launch {launch}")


@pytest.mark.parametrize("cell", RING_CELLS, ids=[c.why[:44] for c in RING_CELLS])
def test_certified_cells_ring(cell) -> None:
    spec = cell.spec
    seed = zlib.crc32(cell.why.encode()) % 2**31
    weights = _weights(seed)
    prepared = _prepared_cls()(weights, spec["key"], EPS)
    mailbox = torch.empty(spec["slots"], CAP, HIDDEN, dtype=torch.bfloat16, device="cuda")
    mailbox.view(torch.int32).fill_(SENTINEL_WORD)
    prepared.bind_ring(mailbox)
    prepared.bind_aux(mix=True)
    rows = spec["rows"]
    for launch in range(2):
        prefix, bank, out, aux = _operands(rows, seed + 10 * launch)
        assert bool((mailbox.view(torch.int32) == SENTINEL_WORD).all()), "mailbox not armed"
        mailbox[:, :rows] = _rows((spec["slots"], rows, HIDDEN), seed + 10 * launch + 5, 0.5)
        pushed = mailbox.clone()
        prefix0, bank0 = prefix.clone(), bank.clone()
        ref = cluster_attn_res_ring.reference(
            prepared, prefix0, bank0.permute(1, 0, 2), out, aux,
            weights=weights, eps=EPS, mailbox=pushed,
        )  # fmt: skip
        with validating(cluster_attn_res_ring):
            result = cluster_attn_res_ring(prepared, prefix, bank.permute(1, 0, 2), out, aux)
        torch.cuda.synchronize()
        assert result is out
        _check(spec, (prefix, out, aux), ref, bank, bank0, f"{cell.why}, launch {launch}")
        # The launch re-armed the rows it read (the next push polls on them).
        assert bool((mailbox.view(torch.int32) == SENTINEL_WORD).all()), (
            f"{cell.why}, launch {launch}: mailbox not re-armed"
        )


def test_the_guard_refuses_out_aliasing_prefix() -> None:
    weights = _weights(1)
    prepared = _prepared_cls()(weights, (1, True, -1, False), EPS)
    prefix, bank, _, _ = _operands(8, 2)
    with pytest.raises(ValueError, match="aliases"):
        cluster_attn_res.is_valid(prepared, prefix, prefix, bank.permute(1, 0, 2), prefix, None)


def test_the_gate_discriminates_a_dropped_block() -> None:
    weights = _weights(3)
    key = (5, True, -1, False)
    prepared = _prepared_cls()(weights, key, EPS)
    prefix, bank, out, _ = _operands(8, 4)
    delta = _rows((8, HIDDEN), 6, 0.5)
    ref = cluster_attn_res.reference(
        prepared, delta, prefix.clone(), bank.permute(1, 0, 2), out, None, weights=weights, eps=EPS
    )
    short = bank.clone()
    short[4] = 0
    wrong = cluster_attn_res.reference(
        prepared, delta, prefix.clone(), short.permute(1, 0, 2), out, None, weights=weights, eps=EPS
    )
    cluster_attn_res(prepared, delta, prefix, bank.permute(1, 0, 2), out, None)
    torch.cuda.synchronize()
    _within(out, ref["out"], 4, "the right blocks")
    with pytest.raises(AssertionError):
        _within(out, wrong["out"], 4, "a dropped block")


@pytest.mark.parametrize("ring", [False, True], ids=["plain", "ring"])
def test_aux_into_a_capture_buffer_column_block(ring) -> None:
    """The flat chain stores a captured tap straight into DFlash's capture buffer
    (``FlatModel._capture_rows``): ``aux`` is then the column block [:T, i*7168:(i+1)*7168] of a
    [T, k*7168] buffer, row stride k*7168. It must hold the contiguous aux's bits and leave the
    buffer's other columns alone."""
    rows, layers, index = 8, 7, 3
    key = (4, True, -1, False) if ring else (4, False, 4, False)
    weights = _weights(11)
    prepared = _prepared_cls()(weights, key, EPS)
    if ring:
        mailbox = torch.empty(1, CAP, HIDDEN, dtype=torch.bfloat16, device="cuda")
        mailbox.view(torch.int32).fill_(SENTINEL_WORD)
        prepared.bind_ring(mailbox)
    prepared.bind_aux(mix=True)
    prefix, bank, out, aux = _operands(rows, 12)
    pushed = _rows((1, rows, HIDDEN), 15, 0.5)
    buffer = torch.full((rows, layers * HIDDEN), POISON, dtype=torch.bfloat16, device="cuda")
    block = buffer[:, index * HIDDEN : (index + 1) * HIDDEN]
    got = {}
    for name, dest in (("contiguous", aux), ("column block", block)):
        p, b, o = prefix.clone(), bank.clone(), out.clone()
        if ring:
            mailbox[:, :rows] = pushed
            cluster_attn_res_ring(prepared, p, b.permute(1, 0, 2), o, dest)
        else:
            cluster_attn_res(prepared, p, p, b.permute(1, 0, 2), o, dest)
        torch.cuda.synchronize()
        got[name] = (p, b, o, dest.clone())
    for a, b in zip(got["contiguous"], got["column block"]):
        assert torch.equal(a.view(torch.int16), b.view(torch.int16))
    others = torch.cat((buffer[:, : index * HIDDEN], buffer[:, (index + 1) * HIDDEN :]), dim=1)
    assert bool((others == POISON).all()), "the launch wrote outside its column block"
