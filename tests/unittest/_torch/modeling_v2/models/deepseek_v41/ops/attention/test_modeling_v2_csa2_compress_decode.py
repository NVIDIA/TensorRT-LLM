# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test for the csa2_compress_decode entry: the index-owner compressor in one launch.

Each cell replays a schedule of steps (one chunk per request per step) through the kernel and through the
plain-torch reference on separate state pools. Latents and index keys are bf16 after an fp32 pooling / norm /
projection computed in another order, so they are gated in bf16 ulps (`assert_within_ulp`); slots, first positions
and the state pools (copies of the inputs) must agree exactly after every step.
"""

import pytest
import torch
from _v41_util import DEV, gen_for

__extra_import_path__ = ["../../../.."]
from _validating import assert_within_ulp, certifying, validating  # noqa: E402,F401

from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.attention.csa2_compress_decode import (  # noqa: E501
    csa2_compress_decode,
)

D, NI, EPS = 512, 128, 1e-6


def _schedule(reqs):
    steps = max(len(r) for r in reqs)
    out, starts = [], [0] * len(reqs)
    for s in range(steps):
        step = []
        for i, r in enumerate(reqs):
            if s < len(r):
                step.append((i, starts[i], r[s]))
                starts[i] += r[s]
        out.append(step)
    return out


def _step(step, rows):
    pos, rid, cs, ce, x = [], [], [], [], []
    for r, a, c in step:
        pos += list(range(a, a + c))
        rid += [r] * c
        cs += [a] * c
        ce += [a + c] * c
        x.append(rows[r][a : a + c])
    t = lambda v, dt: torch.tensor(v, dtype=dt, device=DEV)  # noqa: E731
    return (
        torch.cat(x),
        t(pos, torch.int32),
        t(rid, torch.int64),
        t(cs, torch.int32),
        t(ce, torch.int32),
    )


def _pools(nreq):
    pages = 3 * nreq + 1
    pools = [torch.zeros(pages, 128, D, device=DEV) for _ in range(2)]
    ids = torch.arange(pages - 1, 0, -1, dtype=torch.int32, device=DEV)  # shuffled, page 0 unused
    table = ids[: 3 * nreq].view(nreq, 3).contiguous()
    return pools, table


@pytest.mark.parametrize(
    "cell", csa2_compress_decode.CELLS, ids=[c.why[:44] for c in csa2_compress_decode.CELLS]
)
def test_certified_cells(cell, certifying):  # noqa: F811
    certifying(csa2_compress_decode)
    s, gen = cell.spec, gen_for(cell)
    ratio, reqs = s["ratio"], s["reqs"]
    nreq = len(reqs)
    width = 2 * D if ratio == 2 else D
    rows = [torch.randn(sum(r), width, generator=gen, device=DEV) for r in reqs]
    if ratio == 1 or s.get("bf16"):  # the projection's own output dtype
        rows = [r.bfloat16() for r in rows]
    norm_w = torch.randn(D, generator=gen, device=DEV).bfloat16()
    wk = (torch.randn(NI, D, generator=gen, device=DEV) * 0.05).bfloat16()
    (pk, ps), table = _pools(nreq)
    (rk, rs) = (pk.clone(), ps.clone())
    rpp = 128 // ratio
    gt = torch.randperm(nreq * 3 + 5, generator=gen, device=DEV)[: nreq * 3].view(nreq, 3).int()
    gt[:, 2] = -1  # an unmapped block
    for step in _schedule(reqs):
        x, pos, rid, cs, ce = _step(step, rows)
        gemm = s.get("index_gemm", False)
        args = (x, pos, rid, cs, ce, pk, ps, table, table, norm_w, EPS, wk, gt, ratio, gemm)
        got = csa2_compress_decode(*args)
        ref = csa2_compress_decode.reference(
            x, pos, rid, cs, ce, rk, rs, table, table, norm_w, EPS, wk, gt, ratio, gemm
        )
        assert_within_ulp(got[0], ref[0], element_ulp=2, rms_ulp=0.5)
        assert_within_ulp(got[1], ref[1], element_ulp=4, rms_ulp=1.0)
        assert torch.equal(got[2], ref[2]) and torch.equal(got[3], ref[3])
        if ratio == 2:
            assert torch.equal(pk, rk) and torch.equal(ps, rs)
        assert (got[2] >= 0).any() and rpp > 0


def test_rejections():
    (pk, ps), table = _pools(1)
    x = torch.zeros(2, 2 * D, device=DEV)
    pos = torch.tensor([0, 1], dtype=torch.int32, device=DEV)
    rid = torch.zeros(2, dtype=torch.int64, device=DEV)
    cs, ce = (
        torch.zeros(2, dtype=torch.int32, device=DEV),
        torch.full((2,), 2, dtype=torch.int32, device=DEV),
    )
    norm_w, wk = torch.ones(D, device=DEV).bfloat16(), torch.zeros(NI, D, device=DEV).bfloat16()
    gt = torch.zeros(1, 3, dtype=torch.int32, device=DEV)
    args = (x, pos, rid, cs, ce, pk, ps, table, table, norm_w, EPS, wk, gt, 2)
    csa2_compress_decode.is_valid(*args)
    with pytest.raises(
        AssertionError, match="promoted"
    ):  # fp16 rows are neither the fp32 nor the bf16 form
        csa2_compress_decode.is_valid(x.half(), *args[1:])
    with pytest.raises(AssertionError, match="adjacent"):
        csa2_compress_decode.is_valid(
            x, torch.tensor([0, 2], dtype=torch.int32, device=DEV), rid, cs, ce + 1, *args[5:]
        )
    with pytest.raises(AssertionError, match="cs, ce"):
        csa2_compress_decode.is_valid(x, pos, rid, cs, ce - 1, *args[5:])
    # a request id recurring after another request: the adjacency check alone passes it
    x3, z3 = torch.zeros(3, 2 * D, device=DEV), torch.zeros(3, dtype=torch.int32, device=DEV)
    with pytest.raises(AssertionError, match="one run"):
        csa2_compress_decode.is_valid(
            x3,
            torch.tensor([0, 0, 1], dtype=torch.int32, device=DEV),
            torch.tensor([0, 1, 0], device=DEV),
            z3,
            z3 + 2,
            *args[5:],
        )
    with pytest.raises(AssertionError, match="consecutive"):  # a strided norm weight view
        csa2_compress_decode.is_valid(
            *args[:9], torch.ones(2 * D, device=DEV).bfloat16()[::2], *args[10:]
        )


def test_padding_rows_read_no_row_past_the_launch():
    """A step padded to a bucket with copies of the last row whose `ce` is out of range finds, for each padding row,
    its partner "in flight" in the row after it. The last padding
    row's partner must stay inside the launch's rows (as `csa2_compress_pairs` clamps it): before, it read the row past
    `x`, an illegal access when `x` ended its allocation segment (the dep2 CI gate's engine start). Here that row is
    NaN, so a read of it shows in the latent; every padding row must give the step's last row's latent exactly."""
    n, rows = (
        5,
        256,
    )  # a 5-row chunk at positions 0..4 (the last row's partner is outside the step), padded to 256
    gen = torch.Generator(device=DEV).manual_seed(6)
    big = torch.randn(rows + 1, 2 * D, generator=gen, device=DEV).bfloat16()
    big[n:rows] = big[n - 1]  # padding rows copy the last row's inputs
    big[rows] = float("nan")  # the row past the launch
    x = big[:rows]
    pos = torch.tensor(list(range(n)) + [n - 1] * (rows - n), dtype=torch.int32, device=DEV)
    rid = torch.zeros(rows, dtype=torch.int64, device=DEV)
    cs = torch.zeros(rows, dtype=torch.int32, device=DEV)
    ce = torch.tensor([n] * n + [1 << 30] * (rows - n), dtype=torch.int32, device=DEV)
    norm_w = torch.randn(D, generator=gen, device=DEV).bfloat16()
    wk = (torch.randn(NI, D, generator=gen, device=DEV) * 0.05).bfloat16()
    (pk, ps), table = _pools(1)
    gt = torch.arange(3, dtype=torch.int32, device=DEV).view(1, 3)
    lat = csa2_compress_decode(
        x, pos, rid, cs, ce, pk, ps, table, table, norm_w, EPS, wk, gt, 2, True
    )[0]
    assert torch.isfinite(lat).all()
    assert torch.equal(lat[n:], lat[n - 1 : n].expand(rows - n, D))


def test_padding_rows_write_no_state():
    """The dep2 static body's padding rows copy the step's last row's (request, position) but not necessarily its
    activations (a padding row's Engram look-back reads the ring where the live row reads the step), and the state slot
    they would write is the live row's: the partner the next chunk pools that position with. A padded launch must leave
    the state pools exactly as the unpadded one does, whatever the padding rows carry."""
    n, rows = (
        5,
        256,
    )  # a 5-row chunk at positions 0..4: position 4's partner is the next chunk's first row
    gen = torch.Generator(device=DEV).manual_seed(7)
    live = torch.randn(n, 2 * D, generator=gen, device=DEV).bfloat16()
    pad = torch.randn(
        rows - n, 2 * D, generator=gen, device=DEV
    ).bfloat16()  # differs from the last live row
    norm_w = torch.randn(D, generator=gen, device=DEV).bfloat16()
    wk = (torch.randn(NI, D, generator=gen, device=DEV) * 0.05).bfloat16()
    gt = torch.arange(3, dtype=torch.int32, device=DEV).view(1, 3)
    i32 = lambda v: torch.tensor(v, dtype=torch.int32, device=DEV)  # noqa: E731

    def run(x, pos, ce):
        (pk, ps), table = _pools(1)
        rid, cs = torch.zeros(x.shape[0], dtype=torch.int64, device=DEV), torch.zeros_like(pos)
        csa2_compress_decode(
            x, pos, rid, cs, ce, pk, ps, table, table, norm_w, EPS, wk, gt, 2, True
        )
        torch.cuda.synchronize()
        return pk, ps

    want = run(live, i32(list(range(n))), i32([n] * n))
    padded_pos = i32(list(range(n)) + [n - 1] * (rows - n))
    got = run(torch.cat([live, pad]), padded_pos, i32([n] * n + [1 << 30] * (rows - n)))
    assert torch.equal(got[0], want[0]) and torch.equal(got[1], want[1])


def test_cuda_graph_replay_follows_device_data():
    """Positions and state pools are device data: a replay one step on must give what an eager call gives."""
    reqs = ((8, 6, 6), (5, 6, 6))
    gen = torch.Generator(device=DEV).manual_seed(4)
    rows = [torch.randn(sum(r), 2 * D, generator=gen, device=DEV).bfloat16() for r in reqs]
    norm_w = torch.randn(D, generator=gen, device=DEV).bfloat16()
    wk = (torch.randn(NI, D, generator=gen, device=DEV) * 0.05).bfloat16()
    (pk, ps), table = _pools(len(reqs))
    gt = (
        torch.randperm(len(reqs) * 3 + 5, generator=gen, device=DEV)[: len(reqs) * 3]
        .view(len(reqs), 3)
        .int()
    )
    steps = [_step(step, rows) for step in _schedule(reqs)]

    def run(s):
        return csa2_compress_decode(
            s[0], s[1], s[2], s[3], s[4], pk, ps, table, table, norm_w, EPS, wk, gt, 2
        )

    run(steps[0])  # eager warm-up (kernel JIT) and step 0
    snap = (pk.clone(), ps.clone())
    static = [t.clone() for t in steps[1]]
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = run(static)
    (
        pk.copy_(snap[0]),
        ps.copy_(snap[1]),
    )  # capture ran the kernel once: back to the state after step 0
    run(
        steps[1]
    )  # step 1 eagerly; the replay then runs step 2 from new inputs copied into the captured tensors
    snap = (pk.clone(), ps.clone())
    assert all(a.shape == b.shape for a, b in zip(steps[1], steps[2])) and not torch.equal(
        steps[1][1], steps[2][1]
    )
    for t, v in zip(static, steps[2]):
        t.copy_(v)
    g.replay()
    torch.cuda.synchronize()
    state_replay = (pk.clone(), ps.clone())
    pk.copy_(snap[0]), ps.copy_(snap[1])
    ref = run(steps[2])
    assert all(torch.equal(a, b) for a, b in zip(out, ref))
    assert torch.equal(pk, state_replay[0]) and torch.equal(ps, state_replay[1])


def test_speculative_rollback_reads_the_accepted_partner():
    """Ratio 2 under speculative verify steps (1 + 3 drafts, `accepted` kept, the next step starting right after them,
    often at an odd position inside the previous step): every completed pair of accepted tokens gives the latent of the
    true pair (pooled in fp64 from the true values, rounded and normalised as the op), never one pooled with the page's
    old content or a rejected draft."""
    gen = torch.Generator(device=DEV).manual_seed(5)
    plan, start = [(0, 5, 5)], 5
    for a in (1, 2, 1, 4, 3, 1, 2, 2, 1, 3):
        plan.append((start, 4, a))
        start += a
    total = start + 4
    x = torch.randn(total, 2 * D, generator=gen, device=DEV)
    norm_w = torch.randn(D, generator=gen, device=DEV).bfloat16()
    wk = (torch.randn(NI, D, generator=gen, device=DEV) * 0.05).bfloat16()
    (pk, ps), table = _pools(1)
    (
        pk.normal_(generator=gen),
        ps.normal_(generator=gen),
    )  # a reused page: whatever an earlier request left there
    gt = torch.arange(3, dtype=torch.int32, device=DEV).view(1, 3)
    kv, sc = x[:, :D].double(), x[:, D:].double()
    pairs = (
        kv[: total // 2 * 2].unflatten(0, (-1, 2))
        * sc[: total // 2 * 2].unflatten(0, (-1, 2)).softmax(dim=1)
    ).sum(1)
    v = pairs.to(torch.bfloat16).float()
    want = (v * torch.rsqrt(v.square().mean(-1, keepdim=True) + EPS) * norm_w.float()).to(
        torch.bfloat16
    )
    checked = 0
    for s0, rows, accepted in plan:
        xs = x[s0 : s0 + rows].clone()
        xs[accepted:] = torch.randn(
            xs[accepted:].shape, generator=gen, device=DEV
        )  # rejected drafts
        _, pos, rid, cs, ce = _step([(0, s0, rows)], [x])
        lat = csa2_compress_decode(
            xs, pos, rid, cs, ce, pk, ps, table, table, norm_w, EPS, wk, gt, 2
        )[0]
        for t in range(accepted):
            p = s0 + t
            if p % 2 == 1:
                assert_within_ulp(
                    lat[t : t + 1], want[p // 2 : p // 2 + 1], element_ulp=2, rms_ulp=0.5
                )
                checked += 1
    assert checked >= 8
