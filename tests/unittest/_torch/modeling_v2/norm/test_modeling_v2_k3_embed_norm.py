# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the k3_embed_norm catalog entry.

Kimi K3's replicated embedding table [163840, 7168] at every token count of the K3 decode steps (N = 1..8, and
16..64 for DSpark verify steps of up to 8 requests x 8 tokens), against a native torch reference: the rows written into
`raw` bit for bit against a torch gather, the normed rows against an fp64 RMSNorm (within 8e-3 of each row's max
|ref|). Also: run-to-run bits, each row's bits independent of N, int64 ids equal to int32 ids, another eps and bank
slot, ids outside [0, V), a CUDA graph replayed with its ids rewritten, and the refused calls.
"""

import functools
import math

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog.norm.k3_embed_norm import k3_embed_norm

assert torch.cuda.is_available(), "k3_embed_norm requires a CUDA device"
# The receipts are sm_100 ones; CI's other architectures skip this file.
pytestmark = pytest.mark.skipif(
    torch.cuda.get_device_capability() != (10, 0), reason="certified on sm_100 only"
)

VOCAB, HIDDEN = 163840, 7168
EPS, EPS_ALT = 1e-6, 1e-5
TOL = 8e-3  # per row: max |out - ref| / max |ref|, ref an fp64 RMSNorm
TOKENS = list(range(1, 9)) + list(range(16, 65, 8))
INT64_TOKENS = (1, 8, 64)
BANK_SLOTS = 4  # the attention-residual snapshot bank; the model passes slot 0 as raw
SENTINEL = 7.0
OUTSIDE_IDS = (-1, VOCAB, VOCAB + 7, -VOCAB, 2**31 - 1, -(2**31))
OUTSIDE_IDS_INT64 = (2**32, 2**32 + 5, 3 - 2**32, 2**40)


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
def _ids64() -> torch.Tensor:
    """64 int32 ids in [0, V): random, with 0, V - 1, the zero row, the smallest-scale row and repeats in front."""
    _, small = _table()
    gen = torch.Generator(device="cuda").manual_seed(11)
    ids = torch.randint(0, VOCAB, (64,), generator=gen, device="cuda", dtype=torch.int32)
    ids[0], ids[1], ids[2], ids[4] = 0, VOCAB - 1, 1, small
    ids[3] = ids[1]
    ids[6] = ids[5]
    torch.cuda.synchronize()
    return ids


def _bank(n: int) -> torch.Tensor:
    """A snapshot bank [BANK_SLOTS, n, H] filled with SENTINEL."""
    return torch.full((BANK_SLOTS, n, HIDDEN), SENTINEL, dtype=torch.bfloat16, device="cuda")


def _rows_ref(ids: torch.Tensor) -> torch.Tensor:
    """The torch gather: table[ids], zero rows for ids outside [0, V)."""
    table, _ = _table()
    valid = (ids >= 0) & (ids < VOCAB)
    rows = table.index_select(0, torch.where(valid, ids, torch.zeros_like(ids)))
    return rows.masked_fill(~valid[:, None], 0.0)


def _norm_ref(rows: torch.Tensor, eps: float) -> torch.Tensor:
    """fp64 RMSNorm: rows * rsqrt(mean(rows^2) + eps) * weight."""
    x = rows.double()
    return x * torch.rsqrt(x.pow(2).mean(dim=1, keepdim=True) + eps) * _weight().double()


def _call(ids: torch.Tensor, eps: float = EPS, slot: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """k3_embed_norm into ``slot`` of a fresh bank: (out, bank)."""
    table, _ = _table()
    bank = _bank(ids.numel())
    return k3_embed_norm(ids, table, _weight(), eps, bank[slot]), bank


def _check(ids: torch.Tensor, eps: float = EPS, slot: int = 0) -> torch.Tensor:
    """One call's checks against the reference and a rerun; returns the call's output."""
    n = ids.numel()
    what = f"N={n} {ids.dtype} eps={eps:g} slot={slot}"
    out, bank = _call(ids, eps, slot)
    again, bank_again = _call(ids, eps, slot)
    rows = _rows_ref(ids)
    ref = _norm_ref(rows, eps)
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


def test_certified_cells() -> None:
    """Every N with int32 ids, eps 1e-6, raw = slot 0; every row bit-identical to the same id's row of the 64-id
    call."""
    ids64 = _ids64()
    full = _check(ids64)
    for n in TOKENS:
        out = _check(ids64[:n])
        assert _same(out, full[:n]), f"N={n}: rows differ from the 64-id call"


def test_int64_ids_other_eps_and_slot() -> None:
    """int64 ids at N 1, 8 and 64, bit-identical to int32 ids; eps 1e-5 into bank slot 1 at N 8."""
    ids64 = _ids64()
    for n in INT64_TOKENS:
        ids = ids64[:n]
        assert _same(_check(ids.long()), _call(ids)[0]), f"N={n}: int64 ids differ from int32"
    _check(ids64[:8], eps=EPS_ALT, slot=1)


def test_ids_outside_vocab() -> None:
    """N 8, every out-of-range value at the even positions (int32 and int64): zero raw rows, zero normed rows, the other
    rows as the reference."""
    gen = torch.Generator(device="cuda").manual_seed(3008)
    for dtype, bad in (
        (torch.int32, OUTSIDE_IDS),
        (torch.int64, OUTSIDE_IDS + OUTSIDE_IDS_INT64),
    ):
        for start in range(0, len(bad), 4):
            ids = torch.randint(0, VOCAB, (8,), generator=gen, device="cuda", dtype=dtype)
            values = [bad[(start + i) % len(bad)] for i in range(4)]
            ids[0::2] = torch.tensor(values, dtype=dtype, device="cuda")
            out = _check(ids)
            assert bool((out[0::2] == 0).all()), f"{dtype} {values}: non-zero rows"


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
        assert _same(bank[0], _rows_ref(ids_buf)), f"replay {rep}: raw is not the gathered rows"


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
