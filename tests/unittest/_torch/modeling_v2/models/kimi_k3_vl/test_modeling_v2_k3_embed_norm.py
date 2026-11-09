# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the k3_embed_norm catalog entry.

Kimi K3's replicated embedding table [163840, 7168] at every token count of the K3 decode steps (N = 1..8, and
16..64 for DSpark verify steps of up to 8 requests x 8 tokens), against a native torch reference: the rows written into
`raw` bit for bit against a torch gather, the normed rows against an fp32 RMSNorm (within 8e-3 of each row's max
|ref|). Also: run-to-run bits, each row's bits independent of N, int64 ids equal to int32 ids, another eps and bank
slot, ids outside [0, V), a CUDA graph replayed with its ids rewritten, and the refused calls.
"""

import functools
import math
import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.ops.norm.k3_embed_norm import (
    k3_embed_norm,
)

__extra_import_path__ = ["../.."]
from _validating import validating

assert torch.cuda.is_available(), "k3_embed_norm requires a CUDA device"
# The receipts are sm_100 ones; CI's other architectures skip this file.
pytestmark = pytest.mark.skipif(
    torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="certified on sm_100 and sm_103",
)

VOCAB, HIDDEN = 163840, 7168
EPS = 1e-6
TOL = 8e-3
BANK_SLOTS = 4  # the attention-residual snapshot bank; the model passes slot 0 as raw
SENTINEL = 7.0


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.int16)


def _same(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.shape == b.shape and torch.equal(_bits(a), _bits(b))


@functools.lru_cache(maxsize=None)
def _table() -> tuple[torch.Tensor, int]:
    """The table, built once: N(0, 1) rows at log-uniform scales 1e-4..1e2, row 1 all zero; and its smallest-scale
    row's id (a row whose norm eps dominates)."""
    gen = torch.Generator(device="cuda").manual_seed(20260930)
    table = torch.randn(VOCAB, HIDDEN, generator=gen, device="cuda", dtype=torch.bfloat16)
    scale = torch.empty(VOCAB, 1, device="cuda")
    scale.uniform_(math.log(1e-4), math.log(1e2), generator=gen).exp_()
    table.mul_(scale.bfloat16())
    table[1].zero_()
    scale[1] = math.inf
    return table, int(scale.argmin())


@functools.lru_cache(maxsize=None)
def _weight() -> torch.Tensor:
    """Layer 0's input-norm weight: around 1, every 101st element negative."""
    gen = torch.Generator(device="cuda").manual_seed(7)
    weight = 1.0 + 0.2 * torch.randn(HIDDEN, generator=gen, device="cuda")
    weight[::101] *= -1.0
    return weight.bfloat16()


@functools.lru_cache(maxsize=None)
def _ids64(seed: int = 11) -> torch.Tensor:
    """64 int32 ids in [0, V): random, with 0, V - 1, the zero row, the smallest-scale row and repeats in front."""
    _, small = _table()
    gen = torch.Generator(device="cuda").manual_seed(seed)
    ids = torch.randint(0, VOCAB, (64,), generator=gen, device="cuda", dtype=torch.int32)
    ids[0], ids[1], ids[2], ids[4] = 0, VOCAB - 1, 1, small
    ids[3] = ids[1]
    ids[6] = ids[5]
    torch.cuda.synchronize()
    return ids


def _bank(n: int) -> torch.Tensor:
    """A snapshot bank [BANK_SLOTS, n, H] filled with SENTINEL."""
    return torch.full((BANK_SLOTS, n, HIDDEN), SENTINEL, dtype=torch.bfloat16, device="cuda")


def _call(ids: torch.Tensor, eps: float = EPS, slot: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """k3_embed_norm into ``slot`` of a fresh bank: (out, bank)."""
    table, _ = _table()
    bank = _bank(ids.numel())
    with validating(k3_embed_norm):
        out = k3_embed_norm(ids, table, _weight(), eps, bank[slot])
    return out, bank


def _check(ids: torch.Tensor, eps: float = EPS, slot: int = 0) -> torch.Tensor:
    """One call's checks against the reference and a rerun; returns the call's output."""
    n = ids.numel()
    what = f"N={n} {ids.dtype} eps={eps:g} slot={slot}"
    out, bank = _call(ids, eps, slot)
    again, bank_again = _call(ids, eps, slot)
    table, _ = _table()
    rows = torch.empty_like(bank[slot])
    ref = k3_embed_norm.reference(ids, table, _weight(), eps, rows)
    assert out.shape == (n, HIDDEN) and out.dtype == torch.bfloat16 and out.device == ids.device
    assert out.is_contiguous()
    assert _same(bank[slot], rows), f"{what}: raw is not the gathered rows"
    others = [s for s in range(BANK_SLOTS) if s != slot]
    assert bool((bank[others] == SENTINEL).all()), f"{what}: wrote outside raw"
    scale = ref.abs().amax(dim=1)
    live = scale > 0
    err = ((out.double() - ref).abs().amax(dim=1)[live] / scale[live]).max().item()
    assert err <= TOL, f"{what}: rel {err:.3e}"
    assert bool((out[~live] == 0).all()), f"{what}: a zero row normed to non-zero"
    assert _same(again, out) and _same(bank_again, bank), f"{what}: a rerun differs"
    return out


@pytest.mark.parametrize("cell", k3_embed_norm.CELLS, ids=[c.why[:40] for c in k3_embed_norm.CELLS])
def test_certified_cells(cell) -> None:
    spec = cell.spec
    assert (spec["vocab"], spec["hidden"]) == (VOCAB, HIDDEN)
    ids64 = _ids64(zlib.crc32(cell.why.encode()) % 2**31).to(spec["dtype"])
    ids = ids64[: spec["tokens"]].clone()
    if "outside" in spec:
        ids[0::2] = torch.tensor(spec["outside"], dtype=spec["dtype"], device="cuda")
    out = _check(ids, spec["eps"], spec["slot"])
    if "outside" in spec:
        assert bool((out[0::2] == 0).all()), f"{cell.why}: non-zero rows"
    else:
        full = _call(ids64, spec["eps"], spec["slot"])[0]
        assert _same(out, full[: ids.numel()]), f"{cell.why}: rows differ from the 64-id call"
        if spec["dtype"] == torch.int64:
            assert _same(out, _call(ids.int(), spec["eps"], spec["slot"])[0]), cell.why


@pytest.mark.parametrize("overlap", ("table", "weight", "ids"))
def test_guard_refuses_raw_aliases(overlap) -> None:
    table = torch.ones(8, HIDDEN, dtype=torch.bfloat16, device="cuda")
    weight = torch.ones(HIDDEN, dtype=torch.bfloat16, device="cuda")
    ids = torch.zeros(8, dtype=torch.int32, device="cuda")
    raw = torch.empty_like(table)
    if overlap == "table":
        raw = table
    elif overlap == "weight":
        weight = raw[0]
    else:
        ids = raw.view(torch.int32).reshape(-1)[:8]
    with pytest.raises(AssertionError, match="overlap"):
        with validating(k3_embed_norm):
            k3_embed_norm(ids, table, weight, EPS, raw)


@pytest.mark.parametrize("index", (1, 2, 4))
def test_guard_refuses_mixed_devices(index) -> None:
    args = [
        torch.zeros(8, dtype=torch.int32, device="cuda"),
        torch.ones(8, HIDDEN, dtype=torch.bfloat16, device="cuda"),
        torch.ones(HIDDEN, dtype=torch.bfloat16, device="cuda"),
        EPS,
        torch.empty(8, HIDDEN, dtype=torch.bfloat16, device="cuda"),
    ]
    args[index] = args[index].cpu()
    with pytest.raises(AssertionError, match="device"):
        with validating(k3_embed_norm):
            k3_embed_norm(*args)


def test_graph_replay() -> None:
    """An N = 8 call captured once after an eager call compiled it, replayed with the ids rewritten in place (random,
    then some outside [0, V)): every replay bit-identical to an eager call on the same ids, raw the gathered rows."""
    table, _ = _table()
    ids_buf = _ids64()[:8].clone()
    bank = _bank(8)
    k3_embed_norm(ids_buf, table, _weight(), EPS, bank[0])  # compiles outside the capture
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = k3_embed_norm(ids_buf, table, _weight(), EPS, bank[0])
    gen = torch.Generator(device="cuda").manual_seed(4008)
    for rep in range(3):
        ids = torch.randint(0, VOCAB, (8,), generator=gen, device="cuda", dtype=torch.int32)
        if rep == 2:
            ids[0::2] = -1
        ids_buf.copy_(ids)
        bank.fill_(SENTINEL)
        graph.replay()
        eager, eager_bank = _call(ids_buf)
        assert _same(out, eager) and _same(bank, eager_bank), f"replay {rep} differs from eager"
        rows = torch.empty_like(bank[0])
        k3_embed_norm.reference(ids_buf, table, _weight(), EPS, rows)
        assert _same(bank[0], rows), f"replay {rep}: raw is not the gathered rows"


def test_refused_calls() -> None:
    """Calls outside the op's support raise ValueError before launching anything: raw stays untouched."""
    table, _ = _table()
    weight = _weight()
    ids8 = _ids64()[:8]
    bank = _bank(8)

    def refused(ids, tab, w, raw):
        with pytest.raises(ValueError, match="unsupported call"):
            k3_embed_norm(ids, tab, w, EPS, raw)

    refused(ids8[:0], table, weight, bank[0, :0])  # N = 0
    ids65 = torch.zeros(65, dtype=torch.int32, device="cuda")
    refused(ids65, table, weight, torch.empty(65, HIDDEN, dtype=torch.bfloat16, device="cuda"))
    refused(ids8.to(torch.int16), table, weight, bank[0])  # id dtype
    refused(ids8, table, weight, _bank(64)[0])  # raw [64, H] for 8 ids
    flat = torch.empty(8 * HIDDEN + 8, dtype=torch.bfloat16, device="cuda")
    refused(ids8, table, weight, flat[1 : 1 + 8 * HIDDEN].view(8, HIDDEN))  # raw 2 bytes off
    refused(ids8, table, weight[: HIDDEN // 2], bank[0])  # weight [H / 2]
    for hidden in (6144, 7680):  # outside the norm's geometry
        small = torch.zeros(16, hidden, dtype=torch.bfloat16, device="cuda")
        w = torch.ones(hidden, dtype=torch.bfloat16, device="cuda")
        raw = torch.empty(8, hidden, dtype=torch.bfloat16, device="cuda")
        refused(ids8 % 16, small, w, raw)
    assert bool((bank == SENTINEL).all())
