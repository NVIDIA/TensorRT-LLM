# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""DeepSeek-V4-Flash large-batch fused sparse decode kernels (CuTe DSL, SM100) against an fp64 reference of the chain.

A batch is built from a few distinct base rows (own queries, indexer K pages, compressed KV rows, length); row ``b``
uses base ``b % len(bases)`` and its own window pages and attention query, so a 1,024-row batch needs the memory of
a handful of rows. Every row's top-512 set and attention output is checked. Bases get an opened K-th gap (so the
set is well defined), planted exact ties or all-equal logits; an attention mode in which every score is bit-equal
makes one wrong KV row visible far above the bf16 noise.
"""

import math

import pytest
import torch

from tensorrt_llm._utils import get_sm_version

K_TOP, H_IDX, D_IDX, H_ATT, D_ATT = 512, 64, 128, 64, 512
TPB_C, TPB_SWA, WINDOW, RATIO = (
    32,
    128,
    128,
    4,
)  # compressed / SWA rows per page, window, compress ratio
PAGE_BYTES = TPB_C * (D_IDX + 4)  # 32 x 128 e4m3 values, then 32 fp32 scales
N_MAX = 1 << 18
SM_SCALE = 1.0 / math.sqrt(512.0)

# Lengths around the schedule's granule (256 tokens, 128 units = 32,768 tokens) and the limits.
LENGTHS = [
    262144,
    512,
    100003,
    33000,
    8197,
    262143,
    4097,
    70000,
    1000,
    131072,
    32768,
    32769,
    513,
    200000,
    24576,
    65535,
]


def _op():
    from tensorrt_llm._torch.attention.kernels.blackwell.dsv4_sparse_decode_large_batch import op

    return op


def _supported() -> bool:
    return torch.cuda.is_available() and get_sm_version() == 100 and _op().is_supported()[0]


pytestmark = pytest.mark.skipif(
    not _supported(), reason="persistent 148-CTA kernel, qualified on B200 only"
)


def _cdiv(a, b):
    return (a + b - 1) // b


def _e4m3(x):
    return x.clamp(-448.0, 448.0).to(torch.float8_e4m3fn)


def _ue8m0(amax):
    return torch.exp2(torch.ceil(torch.log2(amax.clamp_min(1e-30) / 448.0)))


def _ref_topk(lg):
    v, idx = torch.sort(lg, descending=True, stable=True)
    tied = lg.numel() > K_TOP and bool(v[K_TOP - 1] == v[K_TOP])
    return idx[:K_TOP], v[:K_TOP], tied


@torch.no_grad()
def _ref_attention(inp, b, sel, scale):
    """fp64 sparse MLA over the 128 window rows and the selected compressed rows, sink in the denominator."""
    pos = int(inp["kv_lens"][b]) - 1
    p = torch.arange(pos - WINDOW + 1, pos + 1, device=sel.device)
    swa_rows = inp["bt_swa"][b, p // TPB_SWA].long() * TPB_SWA + p % TPB_SWA
    c_rows = inp["bt_c"][b, sel // TPB_C].long() * TPB_C + sel % TPB_C
    swa = inp["swa_pool"].view(torch.uint8).reshape(-1, D_ATT)
    cpl = inp["c_pool"].view(torch.uint8).reshape(-1, D_ATT)
    kv = torch.cat([swa[swa_rows], cpl[c_rows]], 0).view(torch.float8_e4m3fn).double()
    q = inp["q_att"][b].view(H_ATT, D_ATT).double()
    s = (q @ kv.t()) * scale
    m = s.amax(-1, keepdim=True)
    e = torch.exp(s - m)
    den = e.sum(-1, keepdim=True) + torch.exp(inp["sinks"].double().view(-1, 1) - m)
    return (e @ kv) / den


def _calc_diff(x, y):
    x, y = x.double(), y.double()
    den = (x * x + y * y).sum()
    return 0.0 if float(den) == 0.0 else float(1.0 - 2.0 * (x * y).sum() / den)


def _open_gap(lg, sc, gap_rel, above=False):
    """Scale the scores just below rank 512 by an exact power of two until the K-th gap exceeds fp32 noise.

    above: also lift the winners within the margin of the K-th value. A tie group planted at that value is not told
    apart from such a winner in fp32, so the kernel may take either.
    """
    if lg.numel() <= K_TOP:  # every position is selected
        return
    for _ in range(8):
        v, order = torch.sort(lg, descending=True, stable=True)
        kth, nxt = float(v[K_TOP - 1]), float(v[K_TOP])
        margin = gap_rel * max(1.0, abs(kth))
        up = order[: K_TOP - 1][v[: K_TOP - 1] < kth + margin] if above else order[:0]
        if kth - nxt >= margin and not up.numel():
            return
        band = order[K_TOP:][v[K_TOP:] > kth - margin]
        if kth > 4 * margin:
            f = 0.25
        elif kth < -margin and not (above and kth > -4 * margin):
            f = 4.0
        else:
            raise RuntimeError("K-th logit too close to 0: pick another seed")
        sc[band] *= f
        lg[band] *= f
        sc[up] /= f  # exact; the winners move away from the K-th value
        lg[up] /= f
    raise RuntimeError("could not open the K-th gap")


@torch.no_grad()
def _make_base(n, seed, ties=0, flat=False):
    """One distinct row: indexer query and weights, K pages (whole 128-row tiles) with their fp64 logits.

    ties: copies the K-th position's K row onto ``ties`` losers (an exact tie group at the K-th value, with the gap
    above it opened as well).
    flat: every position shares one K row (every logit equal).
    """
    dev = torch.device("cuda")
    gh = torch.Generator().manual_seed(seed)
    gd = torch.Generator(device=dev).manual_seed(seed + 1)
    q = _e4m3(torch.randn(H_IDX, D_IDX, generator=gd, device=dev) * 2.0)
    w = (torch.randn(H_IDX, generator=gd, device=dev) * (D_IDX * H_IDX) ** -0.5).float()
    pg = _cdiv(n, 128) * (128 // TPB_C)
    npad = pg * TPB_C
    kf = torch.randn(npad, D_IDX, generator=gd, device=dev) * torch.exp(
        0.5 * torch.randn(npad, 1, generator=gd, device=dev)
    )
    if flat:
        kf = kf[:1].repeat(npad, 1)
    sc = _ue8m0(kf.abs().amax(-1))
    k8u = _e4m3(kf / sc[:, None]).view(torch.uint8)
    k8 = k8u.view(torch.float8_e4m3fn)
    lg = (k8[:n].double() @ q.double().t()).clamp_min(0.0) @ w.double() * sc[:n].double()
    if not flat:
        _open_gap(lg, sc, 1e-4, above=bool(ties))
    if ties:
        order = torch.sort(lg, descending=True, stable=True).indices
        src = int(order[K_TOP - 1])
        losers = order[K_TOP:]
        dst = losers[torch.randperm(losers.numel(), generator=gh)[:ties].to(dev)]
        k8u[dst] = k8u[src].clone()
        sc[dst] = sc[src].clone()
        lg[dst] = lg[src].clone()
    top1 = int(torch.argmax(lg))
    if npad > n:  # positions >= n get scores above every real one and must be ignored
        k8u[n:] = k8u[top1].clone()
        sc[n:] = sc[top1] * (256.0 if float(lg[top1]) > 0 else 1.0 / 256.0)
    pages = torch.empty((pg, PAGE_BYTES), dtype=torch.uint8, device=dev)
    pages[:, : TPB_C * D_IDX] = k8u.reshape(pg, TPB_C * D_IDX)
    pages[:, TPB_C * D_IDX :] = sc.float().contiguous().view(torch.uint8).reshape(pg, TPB_C * 4)
    ref_sel, ref_v, tied = _ref_topk(lg)
    return {
        "n": n,
        "q_idx": q,
        "w_idx": w,
        "pages": pages,
        "lg": lg,
        "ref_sel": torch.sort(ref_sel).values,
        "ref_v": ref_v,
        "tied": tied,
    }


@torch.no_grad()
def _make_case(batch, specs, seed, attn="realistic"):
    """Inputs for ``batch`` rows; row b uses base ``b % len(specs)``. specs: dicts of n, ties, flat."""
    dev = torch.device("cuda")
    gh = torch.Generator().manual_seed(seed)
    gd = torch.Generator(device=dev).manual_seed(seed + 7)
    nb = len(specs)
    bases = [
        _make_base(s["n"], seed + 100 * j, s.get("ties", 0), s.get("flat", False))
        for j, s in enumerate(specs)
    ]
    k_off, off = [], 0
    for bs in bases:
        k_off.append(off)
        off += bs["pages"].shape[0]
    spare_k = off
    k_cache = torch.cat([bs["pages"] for bs in bases] + [bases[0]["pages"][:1]])
    idx_bt = torch.full(
        (batch, _cdiv(max(b["pages"].shape[0] for b in bases), 4) * 4), spare_k, dtype=torch.int32
    )
    for b in range(batch):
        j = b % nb
        pg = bases[j]["pages"].shape[0]
        idx_bt[b, :pg] = torch.arange(pg, dtype=torch.int32) + k_off[j]
    ns = [bases[b % nb]["n"] for b in range(batch)]
    kv_lens = [RATIO * n + int(torch.randint(0, RATIO, (1,), generator=gh)) for n in ns]
    q_idx = torch.stack([bases[b % nb]["q_idx"] for b in range(batch)]).unsqueeze(1)
    w_idx = torch.stack([bases[b % nb]["w_idx"] for b in range(batch)])

    def rnd(*shape):
        return torch.randn(*shape, generator=gd, device=dev)

    if attn == "gather_exact":
        # scores depend only on dims [0, 64), shared by every KV row of a base
        q_att = torch.randint(-2, 3, (batch, H_ATT, D_ATT), generator=gd, device=dev).float() * 0.25
        q_att[:, :, 64:] = 0.0
        key = torch.randint(-2, 3, (nb, 64), generator=gd, device=dev).float() * 0.25
    else:
        q_att = rnd(batch, H_ATT, D_ATT)
    q_att = _e4m3(q_att).reshape(batch, H_ATT * D_ATT)

    def kv_rows(j, r):
        x = rnd(r, D_ATT)
        if attn == "gather_exact":
            x[:, :64] = key[j]
        return _e4m3(x).view(torch.uint8)

    poison = torch.full((D_ATT,), 448.0, device=dev)
    poison[:64] = -448.0
    poison = _e4m3(poison).view(torch.uint8)
    n_swa = 2 * batch + 1  # two window pages per row, then a poison spare
    swa_pool = torch.empty((n_swa, TPB_SWA, D_ATT), dtype=torch.uint8, device=dev)
    swa_pool[n_swa - 1] = poison
    perm_s = torch.randperm(2 * batch, generator=gh)
    bt_swa = torch.full((batch, max(kv_lens) // TPB_SWA + 1), n_swa - 1, dtype=torch.int32)
    for b in range(batch):
        hi = (kv_lens[b] - 1) // TPB_SWA
        for t, blk in enumerate((hi - 1, hi)):
            pgs = int(perm_s[2 * b + t])
            if blk >= 0:
                bt_swa[b, blk] = pgs
            swa_pool[pgs] = kv_rows(b % nb, TPB_SWA)
    pg_c = [_cdiv(bs["n"], TPB_C) for bs in bases]
    c_off = [sum(pg_c[:j]) for j in range(nb)]
    n_c = sum(pg_c) + 1
    c_pool = torch.empty((n_c, TPB_C, D_ATT), dtype=torch.uint8, device=dev)
    c_pool[n_c - 1] = poison
    for j in range(nb):
        c_pool[c_off[j] : c_off[j] + pg_c[j]] = kv_rows(j, pg_c[j] * TPB_C).view(
            pg_c[j], TPB_C, D_ATT
        )
    bt_c = torch.full((batch, _cdiv(max(pg_c), 4) * 4), n_c - 1, dtype=torch.int32)
    for b in range(batch):
        j = b % nb
        bt_c[b, : pg_c[j]] = torch.arange(pg_c[j], dtype=torch.int32) + c_off[j]
    sinks = (torch.linspace(-6.0, 9.0, H_ATT, device=dev) + 0.1 * rnd(H_ATT)).float()
    inp = {
        "q_idx": q_idx,
        "w_idx": w_idx.contiguous(),
        "k_cache": k_cache.view(-1, TPB_C, 1, D_IDX + 4),
        "idx_block_table": idx_bt.to(dev),
        "ctx_lens": torch.tensor(ns, dtype=torch.int32, device=dev),
        "kv_lens": torch.tensor(kv_lens, dtype=torch.int32, device=dev),
        "q_att": q_att,
        "swa_pool": swa_pool.view(torch.float8_e4m3fn),
        "bt_swa": bt_swa.to(dev),
        "c_pool": c_pool.view(torch.float8_e4m3fn),
        "bt_c": bt_c.to(dev),
        "sinks": sinks.contiguous(),
    }
    return inp, bases


def _launch(op, state, inp, topk, out, scale=SM_SCALE):
    op.run(
        state,
        inp["q_idx"],
        inp["w_idx"],
        inp["k_cache"],
        inp["idx_block_table"],
        inp["ctx_lens"],
        inp["kv_lens"],
        inp["q_att"],
        inp["swa_pool"],
        inp["bt_swa"],
        inp["c_pool"],
        inp["bt_c"],
        topk,
        out,
        softmax_scale=scale,
    )


def _check_rows(inp, bases, topk, out, attn, scale=SM_SCALE):
    nb = len(bases)
    for b in range(topk.shape[0]):
        base = bases[b % nb]
        sel = topk[b].long()
        n = base["n"]
        assert bool(((sel >= 0) & (sel < n)).all()), f"row {b}: index outside [0, n)"
        s = torch.sort(sel).values
        assert not bool((s[1:] == s[:-1]).any()), f"row {b}: duplicate index"
        if base["tied"]:
            got = torch.sort(base["lg"][sel], descending=True).values
            assert torch.equal(got, base["ref_v"]), f"row {b}: top-512 value multiset differs"
        else:
            assert torch.equal(s, base["ref_sel"]), f"row {b}: top-512 set differs"
        ref = _ref_attention(inp, b, sel, scale)
        o = out[b].view(H_ATT, D_ATT).double()
        assert bool(torch.isfinite(o).all()), f"row {b}: non-finite output"
        if attn == "gather_exact":
            cd = _calc_diff(o[:, 64:], ref[:, 64:])
            assert cd < 2e-5, f"row {b}: calc_diff {cd:.3e}"
            assert float((o - ref).abs().max()) <= 2.0**-6 * float(ref.abs().max())
        else:
            cd = _calc_diff(o, ref)
            assert cd < 1e-3, f"row {b}: calc_diff {cd:.3e}"


def _new_outputs(batch):
    dev = torch.device("cuda")
    topk = torch.full((batch, K_TOP), -5, dtype=torch.int32, device=dev)
    out = torch.zeros((batch, H_ATT * D_ATT), dtype=torch.bfloat16, device=dev)
    return topk, out


def _plain(ns):
    return [{"n": n} for n in ns]


@pytest.mark.parametrize("batch", [1, 7, 100, 148, 149, 256, 777, 1024])
def test_mixed_lengths(batch):
    op = _op()
    inp, bases = _make_case(batch, _plain(LENGTHS), seed=11)
    state = op.prepare(inp["ctx_lens"].cpu(), inp["sinks"])
    topk, out = _new_outputs(batch)
    _launch(op, state, inp, topk, out)
    torch.cuda.synchronize()
    assert int(op.error_bits(state)[0]) == 0
    _check_rows(inp, bases, topk, out, "realistic")


@pytest.mark.parametrize("batch", [256, 1024])
def test_uniform_long_rows(batch):
    op = _op()
    inp, bases = _make_case(batch, _plain([N_MAX - 17] * 4 + [N_MAX] * 4), seed=23)
    state = op.prepare(inp["ctx_lens"].cpu(), inp["sinks"])
    topk, out = _new_outputs(batch)
    _launch(op, state, inp, topk, out)
    torch.cuda.synchronize()
    assert int(op.error_bits(state)[0]) == 0
    _check_rows(inp, bases, topk, out, "realistic")


def test_gather_exact():
    op = _op()
    batch = 256
    inp, bases = _make_case(batch, _plain(LENGTHS), seed=31, attn="gather_exact")
    state = op.prepare(inp["ctx_lens"].cpu(), inp["sinks"])
    topk, out = _new_outputs(batch)
    _launch(op, state, inp, topk, out)
    torch.cuda.synchronize()
    _check_rows(inp, bases, topk, out, "gather_exact")


def test_ties_and_all_equal_rows():
    op = _op()
    batch = 256
    specs = [
        {"n": 100000, "ties": 3},
        {"n": 262144, "ties": 6000},
        {"n": 5000, "flat": True},
        {"n": 262143, "flat": True},
        {"n": 700, "ties": 100},
        {"n": 512},
        {"n": 90000},
        {"n": 40000, "ties": 600},
    ]
    inp, bases = _make_case(batch, specs, seed=41)
    state = op.prepare(inp["ctx_lens"].cpu(), inp["sinks"])
    topk, out = _new_outputs(batch)
    _launch(op, state, inp, topk, out)
    torch.cuda.synchronize()
    _check_rows(inp, bases, topk, out, "realistic")


def test_state_reuse_across_steps():
    """One state serves several steps with different lengths; prepare refreshes the schedule in place."""
    op = _op()
    batch = 256
    state = None
    topk, out = _new_outputs(batch)
    for step, ns in enumerate((LENGTHS, LENGTHS[5:] + LENGTHS[:5], [N_MAX] * 8)):
        inp, bases = _make_case(batch, _plain(ns), seed=50 + step)
        state = op.prepare(inp["ctx_lens"].cpu(), inp["sinks"], state)
        for _ in range(2):  # a repeated launch on the same step
            _launch(op, state, inp, topk, out)
            torch.cuda.synchronize()
            assert int(op.error_bits(state)[0]) == 0
            _check_rows(inp, bases, topk, out, "realistic")


def test_stale_schedule_is_refused():
    op = _op()
    batch = 256
    inp, bases = _make_case(batch, _plain(LENGTHS), seed=61)
    other = torch.tensor([LENGTHS[(b + 3) % len(LENGTHS)] for b in range(batch)], dtype=torch.int32)
    state = op.prepare(other, inp["sinks"])  # a schedule for other lengths
    topk, out = _new_outputs(batch)
    out.fill_(5.0)
    _launch(op, state, inp, topk, out)
    torch.cuda.synchronize()
    assert int(op.error_bits(state)[0]) & op.ERR_SCHEDULE
    assert bool((topk == -1).all()) and bool((out.float() == 0).all())
    state = op.prepare(inp["ctx_lens"].cpu(), inp["sinks"], state)  # refreshed: the launch recovers
    _launch(op, state, inp, topk, out)
    torch.cuda.synchronize()
    assert int(op.error_bits(state)[0]) == 0
    _check_rows(inp, bases, topk, out, "realistic")


def test_cuda_graph_replay():
    """One captured launch replays over several steps; lengths change in place and prepare refreshes the schedule."""
    op = _op()
    batch = 256
    steps = [
        LENGTHS[r:] + LENGTHS[:r] for r in (0, 5, 9)
    ]  # same pool sizes, other rows get other lengths
    inp, bases = _make_case(batch, _plain(steps[0]), seed=71)
    state = op.prepare(inp["ctx_lens"].cpu(), inp["sinks"])
    op.precompile(torch.device("cuda"))
    topk, out = _new_outputs(batch)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        _launch(op, state, inp, topk, out)  # warm-up
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _launch(op, state, inp, topk, out)
    for step, ns in enumerate(steps):
        if step:
            new_inp, bases = _make_case(batch, _plain(ns), seed=71 + step)
            for key, buf in inp.items():
                assert buf.shape == new_inp[key].shape, key
                buf.copy_(new_inp[key])
            op.prepare(inp["ctx_lens"].cpu(), inp["sinks"], state)
        topk.fill_(-5)
        out.zero_()
        graph.replay()
        torch.cuda.synchronize()
        assert int(op.error_bits(state)[0]) == 0
        _check_rows(inp, bases, topk, out, "realistic")


def _schedule_reference(ctx_lens, num_ctas, unit=256, cmin=128):
    """Row r costs max(ceil(n_r / unit), cmin) units; CTA s starts at unit s * (T // G) + min(s, T % G)."""
    units = [-(-n // unit) for n in ctx_lens]
    cost = [max(u, cmin) for u in units]
    prefix, total = [], 0
    for c in cost:
        total += c
        prefix.append(total)
    quota, rem = divmod(total, num_ctas)
    rows = []
    for s in range(num_ctas + 1):
        w = s * quota + min(s, rem)
        r = next((i for i, e in enumerate(prefix) if e > w), len(ctx_lens))
        if r >= len(ctx_lens):
            rows.append((len(ctx_lens), 0))
            continue
        x = w - (prefix[r - 1] if r else 0)
        rows.append((r, x if units[r] >= cmin else 0))
    return rows


@pytest.mark.parametrize("batch", [1, 7, 148, 256, 1024])
def test_schedule_matches_reference(batch):
    op = _op()
    gen = torch.Generator().manual_seed(5 + batch)
    for trial in range(4):
        if trial == 0:
            ns = [N_MAX] * batch
        elif trial == 1:
            ns = [op.MIN_CTX_LEN + int(x) for x in torch.randint(0, 3000, (batch,), generator=gen)]
        else:
            ns = [int(x) for x in torch.randint(op.MIN_CTX_LEN, N_MAX + 1, (batch,), generator=gen)]
        got = op._schedule(ns, op.NUM_SMS)
        assert got.tolist() == [list(r) for r in _schedule_reference(ns, op.NUM_SMS)]


def test_rejects_unsupported_arguments():
    op = _op()
    dev = torch.device("cuda")
    sinks = torch.zeros(H_ATT, device=dev)
    ok = torch.full((256,), 4096, dtype=torch.int32)
    with pytest.raises(ValueError):
        op.prepare(torch.full((256,), op.MAX_CTX_LEN + 1, dtype=torch.int32), sinks)
    with pytest.raises(ValueError):
        op.prepare(torch.full((256,), op.MIN_CTX_LEN - 1, dtype=torch.int32), sinks)
    with pytest.raises(ValueError):
        op.prepare(torch.empty(0, dtype=torch.int32), sinks)
    with pytest.raises(ValueError):
        op.prepare(torch.full((op.MAX_BATCH_SIZE + 1,), 4096, dtype=torch.int32), sinks)
    with pytest.raises(ValueError):
        op.prepare(ok, torch.zeros(H_ATT - 1, device=dev))
    state = op.prepare(ok, sinks)
    with pytest.raises(ValueError):
        op.prepare(torch.full((128,), 4096, dtype=torch.int32), sinks, state)
