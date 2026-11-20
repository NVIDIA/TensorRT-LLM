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
"""specDSA megakernel (CuTe DSL, SM100) against an fp64 reference of the unfused DeepSeek-V4-Flash decode chain
(FP8 indexer K cache; the MXFP4 cache is covered by ``test_fp4_matches_reference``, with the case builder and
references of the large-batch test).

Synthetic inputs: per-row indexer K pages and KV rows scattered over shared pools, an opened K-th gap so the top-512
set is well defined (rows with planted exact ties or all-equal rows are checked as value multisets), and an attention
mode in which every score is bit-equal so one wrong KV row is visible far above the bf16 noise.
"""

import zlib

import pytest
import torch

from tensorrt_llm._torch.attention.kernels.blackwell.spec_dsa.config import DS_V4_FLASH as SPEC
from tensorrt_llm._torch.attention.kernels.blackwell.spec_dsa.config import (
    DS_V4_FLASH_FP4 as SPEC_FP4,
)
from tensorrt_llm._utils import get_sm_version

# The shapes under test are those of the spec (``config.py``); test_cute_dsl_spec_dsa_config.py pins its values.
K_TOP, H_IDX, D_IDX = SPEC.topk, SPEC.index_heads, SPEC.index_head_dim
H_ATT, D_ATT = SPEC.attn_heads, SPEC.kv_dim
# compressed / SWA rows per page, window, compress ratio
TPB_C, TPB_SWA, WINDOW, RATIO = (
    SPEC.kv_page_rows,
    SPEC.swa_page_rows,
    SPEC.window,
    SPEC.compress_ratio,
)
PAGE_BYTES = SPEC.index_page_bytes  # 32 x 128 e4m3 values, then 32 fp32 scales
N_MAX = SPEC.max_index_len
SM_SCALE = SPEC.default_softmax_scale


def _op():
    from tensorrt_llm._torch.attention.kernels.blackwell.spec_dsa import megakernel_op

    return megakernel_op


def _mk():
    from tensorrt_llm._torch.attention.kernels.blackwell.spec_dsa import megakernel

    return megakernel


def _supported() -> bool:
    return torch.cuda.is_available() and get_sm_version() == 100 and _op().is_supported()[0]


pytestmark = pytest.mark.skipif(
    not _supported(), reason="148-CTA cooperative kernel, qualified on B200 only"
)


def _cdiv(a, b):
    return (a + b - 1) // b


def _e4m3(x):
    return x.clamp(-448.0, 448.0).to(torch.float8_e4m3fn)


def _ue8m0(amax):
    return torch.exp2(torch.ceil(torch.log2(amax.clamp_min(1e-30) / 448.0)))


def _gather_k(k_cache, bt_row, n):
    npg = _cdiv(n, TPB_C)
    pg = k_cache.reshape(k_cache.shape[0], PAGE_BYTES).index_select(0, bt_row[:npg].long())
    k8 = pg[:, : TPB_C * D_IDX].contiguous().view(torch.float8_e4m3fn).view(npg * TPB_C, D_IDX)[:n]
    sc = pg[:, TPB_C * D_IDX :].contiguous().view(torch.float32).reshape(npg * TPB_C)[:n]
    return k8, sc


@torch.no_grad()
def _ref_logits(inp, b, chunk=1 << 15):
    """fp64 indexer logits: scale[t] * sum_h relu(q[h] . k[t]) * w[h]."""
    n = int(inp["ctx_lens"][b])
    k8, sc = _gather_k(inp["k_cache"], inp["idx_block_table"][b], n)
    q, w = inp["q_idx"][b, 0].double(), inp["w_idx"][b].double()
    out = torch.empty(n, dtype=torch.float64, device=q.device)
    for t0 in range(0, n, chunk):
        t1 = min(n, t0 + chunk)
        out[t0:t1] = (k8[t0:t1].double() @ q.t()).clamp_min(0.0) @ w * sc[t0:t1].double()
    return out


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


def _open_gap(lg, sc, gap_rel):
    """Scale the scores just below rank 512 by an exact power of two until the K-th gap exceeds fp32 noise."""
    if lg.numel() <= K_TOP:  # every position is selected
        return
    for _ in range(8):
        v, order = torch.sort(lg, descending=True, stable=True)
        kth, nxt = float(v[K_TOP - 1]), float(v[K_TOP])
        margin = gap_rel * max(1.0, abs(kth))
        if kth - nxt >= margin:
            return
        band = order[K_TOP:][v[K_TOP:] > kth - margin]
        if kth > 4 * margin:
            f = 0.25
        elif kth < -margin:
            f = 4.0
        else:
            raise RuntimeError("K-th logit too close to 0: pick another seed")
        sc[band] *= f
        lg[band] *= f
    raise RuntimeError("could not open the K-th gap")


@torch.no_grad()
def _make_case(ns, seed, attn, ties=None, flat=()):
    """Inputs for len(ns) rows of compressed lengths ns; attn is "gather_exact" or "realistic".

    ties: {row: count} copies the K-th position's K row onto ``count`` losers (an exact tie group at the K-th value).
    flat: rows whose positions all share one K row (every logit equal).
    """
    ties = ties or {}
    dev = torch.device("cuda")
    batch = len(ns)
    gh = torch.Generator().manual_seed(seed)
    gd = torch.Generator(device=dev).manual_seed(seed + 1)

    def rnd(*shape):
        return torch.randn(*shape, generator=gd, device=dev)

    kv_lens = [RATIO * n + int(torch.randint(0, RATIO, (1,), generator=gh)) for n in ns]
    pg_idx = [_cdiv(n, 128) * (128 // TPB_C) for n in ns]  # whole 128-row tiles per row
    pages = sum(pg_idx) + 1  # last page: spare
    perm = torch.randperm(pages - 1, generator=gh)
    idx_bt = torch.full((batch, _cdiv(max(pg_idx), 4) * 4), pages - 1, dtype=torch.int32)
    k_cache = torch.zeros((pages, PAGE_BYTES), dtype=torch.uint8, device=dev)
    q_idx = _e4m3(rnd(batch, 1, H_IDX, D_IDX) * 2.0)
    w_idx = (rnd(batch, H_IDX) * (D_IDX * H_IDX) ** -0.5).float()
    off = 0
    for b, n in enumerate(ns):
        npad = pg_idx[b] * TPB_C
        rows = perm[off : off + pg_idx[b]].to(dev)
        off += pg_idx[b]
        idx_bt[b, : pg_idx[b]] = rows.to(torch.int32).cpu()
        kf = rnd(npad, D_IDX) * torch.exp(0.5 * rnd(npad, 1))
        if b in flat:
            kf = kf[:1].repeat(npad, 1)
        sc = _ue8m0(kf.abs().amax(-1))
        k8u = _e4m3(kf / sc[:, None]).view(torch.uint8)
        k8 = k8u.view(torch.float8_e4m3fn)
        lg = (
            (k8[:n].double() @ q_idx[b, 0].double().t()).clamp_min(0.0)
            @ w_idx[b].double()
            * sc[:n].double()
        )
        if b not in flat:
            _open_gap(lg, sc, 1e-4)
        if b in ties:  # exact tie group at the K-th value
            order = torch.sort(lg, descending=True, stable=True).indices
            src = int(order[K_TOP - 1])
            losers = order[K_TOP:]
            dst = losers[torch.randperm(losers.numel(), generator=gh)[: ties[b]].to(dev)]
            k8u[dst] = k8u[src].clone()
            sc[dst] = sc[src].clone()
            lg[dst] = lg[src].clone()
        top1 = int(torch.argmax(lg))
        if npad > n:  # positions >= n get scores above every real one and must be ignored
            k8u[n:] = k8u[top1].clone()
            sc[n:] = sc[top1] * (256.0 if float(lg[top1]) > 0 else 1.0 / 256.0)
        kp = torch.empty((pg_idx[b], PAGE_BYTES), dtype=torch.uint8, device=dev)
        kp[:, : TPB_C * D_IDX] = k8u.reshape(pg_idx[b], TPB_C * D_IDX)
        kp[:, TPB_C * D_IDX :] = (
            sc.float().contiguous().view(torch.uint8).reshape(pg_idx[b], TPB_C * 4)
        )
        k_cache[rows] = kp
    if attn == "gather_exact":
        # scores depend only on dims [0, 64), shared by every KV row of a batch row
        q_att = torch.randint(-2, 3, (batch, H_ATT, D_ATT), generator=gd, device=dev).float() * 0.25
        q_att[:, :, 64:] = 0.0
        key = torch.randint(-2, 3, (batch, 64), generator=gd, device=dev).float() * 0.25
    else:
        q_att = rnd(batch, H_ATT, D_ATT)
    q_att = _e4m3(q_att).reshape(batch, H_ATT * D_ATT)

    def kv_rows(b, r):
        x = rnd(r, D_ATT)
        if attn == "gather_exact":
            x[:, :64] = key[b]
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
        for j, blk in enumerate((hi - 1, hi)):
            pgs = int(perm_s[2 * b + j])
            if blk >= 0:
                bt_swa[b, blk] = pgs
            swa_pool[pgs] = kv_rows(b, TPB_SWA)
    pg_c = [_cdiv(n, TPB_C) for n in ns]
    n_c = sum(pg_c) + 1
    c_pool = torch.empty((n_c, TPB_C, D_ATT), dtype=torch.uint8, device=dev)
    c_pool[n_c - 1] = poison
    perm_c = torch.randperm(n_c - 1, generator=gh)
    # rows 16-byte aligned: width a multiple of 4 entries
    bt_c = torch.full((batch, _cdiv(max(pg_c), 4) * 4), n_c - 1, dtype=torch.int32)
    off = 0
    for b in range(batch):
        rows = perm_c[off : off + pg_c[b]]
        off += pg_c[b]
        bt_c[b, : pg_c[b]] = rows.to(torch.int32)
        c_pool[rows.to(dev)] = kv_rows(b, pg_c[b] * TPB_C).view(pg_c[b], TPB_C, D_ATT)
    sinks = (torch.linspace(-6.0, 9.0, H_ATT, device=dev) + 0.1 * rnd(H_ATT)).float()
    return {
        "q_idx": q_idx,
        "w_idx": w_idx.contiguous(),
        "k_cache": k_cache.view(pages, TPB_C, 1, D_IDX + 4),
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


def _check_row(inp, b, lg, topk_row, out_row, attn, scale):
    sel = topk_row.long()
    assert bool(((sel >= 0) & (sel < lg.numel())).all()), "index outside [0, n)"
    s = torch.sort(sel).values
    assert not bool((s[1:] == s[:-1]).any()), "duplicate index"
    ref_sel, ref_v, tied = _ref_topk(lg)
    if tied:
        assert torch.equal(torch.sort(lg[sel], descending=True).values, ref_v), (
            "top-512 value multiset differs"
        )
    else:
        assert torch.equal(s, torch.sort(ref_sel).values), "top-512 set differs"
    ref = _ref_attention(inp, b, sel, scale)
    o = out_row.view(H_ATT, D_ATT).double()
    assert bool(torch.isfinite(o).all()), "non-finite output"
    if attn == "gather_exact":
        cd = _calc_diff(o[:, 64:], ref[:, 64:])
        assert cd < 2e-5, f"row {b}: calc_diff {cd:.3e}"
        assert float((o - ref).abs().max()) <= 2.0**-6 * float(ref.abs().max())
    else:
        cd = _calc_diff(o, ref)
        assert cd < 1e-3, f"row {b}: calc_diff {cd:.3e}"


def _run(op, state, inp, topk, out, scale):
    op.run(
        state,
        inp["q_idx"],
        inp["w_idx"],
        inp["k_cache"],
        inp["idx_block_table"],
        inp["ctx_lens"],
        inp["q_att"],
        inp["swa_pool"],
        inp["bt_swa"],
        inp["c_pool"],
        inp["bt_c"],
        inp["sinks"],
        topk,
        out,
        softmax_scale=scale,
    )


def _prepare(op, inp, state=None):
    # host lengths: prepare() checks them without a device sync
    return op.prepare(
        inp["ctx_lens"].cpu(), inp["kv_lens"].cpu(), inp["idx_block_table"], state=state
    )


CASES = [  # (rows' compressed lengths, attention mode, extra case options)
    pytest.param([262144], "gather_exact", {}, id="b1-max"),
    pytest.param([262127], "realistic", {"scale": 0.03}, id="b1-scale"),
    pytest.param([8000], "gather_exact", {"ties": {0: 9}}, id="b1-ties"),
    pytest.param([32771], "gather_exact", {"ties": {0: 6000}}, id="b1-tie6000"),
    pytest.param([512], "realistic", {}, id="b1-select-all"),
    pytest.param([131071, 65543], "gather_exact", {"ties": {1: 7}}, id="b2-ties"),
    pytest.param([65536, 32771], "realistic", {}, id="b2-mixed"),
    pytest.param([32771, 32771], "gather_exact", {"flat": (1,)}, id="b2-flat"),
    pytest.param([600, 5000], "realistic", {}, id="b2-short"),
    pytest.param([262127] * 4, "realistic", {}, id="b4-max"),
    pytest.param([131071, 98309, 50021, 32771], "gather_exact", {}, id="b4-mixed"),
    pytest.param([512, 5632, 8192, 700], "gather_exact", {}, id="b4-short"),
    # 3 and 4 rows run as one team of 4 CTAs per row (a single pass)
    pytest.param([262127] * 3, "realistic", {}, id="b3-max"),
    pytest.param(
        [262127, 700, 98309, 32771],
        "gather_exact",
        {"ties": {1: 5, 3: 77}, "flat": (2,)},
        id="b4-short-ties-flat",
    ),
    pytest.param([262127] * 8, "gather_exact", {"flat": (6,)}, id="b8-flat"),
    pytest.param(
        [262127 - 4099 * r for r in range(8)], "realistic", {"ties": {3: 31}}, id="b8-mixed-ties"
    ),
    # batch sizes between the powers of two, and the ring of 9 to 47 rows
    pytest.param([131071, 98309, 50021], "gather_exact", {}, id="b3-mixed"),
    pytest.param([262127] * 5, "realistic", {}, id="b5-max"),
    pytest.param(
        [65536, 32771, 700, 262127, 8192, 5000],
        "gather_exact",
        {"ties": {2: 13}},
        id="b6-short-ties",
    ),
    pytest.param(
        [262127 - 5003 * r for r in range(7)], "gather_exact", {"flat": (4,)}, id="b7-flat"
    ),
    pytest.param([262127 - 7001 * r for r in range(9)], "realistic", {}, id="b9-mixed"),
    pytest.param([131071] * 12, "gather_exact", {"ties": {5: 17}}, id="b12-ties"),
    pytest.param(
        [262127 - 3001 * r for r in range(13)], "gather_exact", {"flat": (9,)}, id="b13-flat"
    ),
    pytest.param([262127] * 16, "realistic", {}, id="b16-max"),
    pytest.param(
        [262127 - 4093 * r for r in range(16)],
        "gather_exact",
        {"ties": {11: 29, 3: 6000}},
        id="b16-mixed-ties",
    ),
    pytest.param([262127 - 6007 * r for r in range(17)], "realistic", {}, id="b17-mixed"),
    pytest.param(
        [131071 - 3001 * r for r in range(20)], "gather_exact", {"ties": {7: 21}}, id="b20-ties"
    ),
    pytest.param([262127] * 24, "gather_exact", {"flat": (13, 20)}, id="b24-flat"),
    pytest.param(
        [262127 - 2999 * r for r in range(27)],
        "realistic",
        {"ties": {26: 9, 0: 6000}},
        id="b27-mixed-ties",
    ),
    pytest.param([262127] * 32, "realistic", {}, id="b32-max"),
    pytest.param(
        [262127 - 3989 * r if r % 3 else 700 + 17 * r for r in range(32)],
        "gather_exact",
        {"ties": {5: 3, 31: 77}, "flat": (18,)},
        id="b32-mixed-short-ties",
    ),
    # 33 to 47 rows: the block-table slices of rows 32 and up reuse the entries of rows 0 and up
    pytest.param([262127 - 4999 * r for r in range(33)], "realistic", {}, id="b33-mixed"),
    pytest.param(
        [131071 - 2003 * r for r in range(37)],
        "gather_exact",
        {"ties": {20: 21, 36: 6000}},
        id="b37-ties",
    ),
    pytest.param([262127] * 41, "gather_exact", {"flat": (13, 33, 40)}, id="b41-flat"),
    pytest.param(
        [262127 - 3517 * r if r % 4 else 900 + 31 * r for r in range(45)],
        "realistic",
        {"ties": {0: 5, 32: 11, 44: 77}},
        id="b45-mixed-ties",
    ),
    pytest.param([262127] * 47, "realistic", {}, id="b47-max"),
    pytest.param(
        [262127 - 3989 * r if r % 3 else 700 + 17 * r for r in range(47)],
        "gather_exact",
        {"ties": {5: 3, 31: 77, 46: 6000}, "flat": (18, 40)},
        id="b47-mixed-short-ties",
    ),
]


@pytest.mark.parametrize("ns,attn,extra", CASES)
def test_matches_reference(ns, attn, extra):
    op, mk = _op(), _mk()
    batch = len(ns)
    extra = dict(extra)
    scale = extra.pop("scale", SM_SCALE)
    seed = zlib.crc32(repr((ns, attn, extra)).encode()) % 100003
    inp = _make_case(ns, seed=seed, attn=attn, **extra)
    lgs = [_ref_logits(inp, b) for b in range(batch)]
    state = _prepare(op, inp)
    topk = torch.full((batch, K_TOP), -1, dtype=torch.int32, device="cuda")
    out = torch.zeros((batch, H_ATT * D_ATT), dtype=torch.bfloat16, device="cuda")
    launches = 0

    def launch(case):
        topk.fill_(-1)
        out.zero_()
        _run(op, state, case, topk, out, scale)

    def check(case, logits):
        torch.cuda.synchronize()
        for b in range(batch):
            _check_row(case, b, logits[b], topk[b], out[b], attn, scale)

    for _ in range(2):  # eager; the second launch runs on the next workspace generation
        launch(inp)
        launches += 1
        check(inp, lgs)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        launch(inp)
        launches += 1
    torch.cuda.current_stream().wait_stream(side)
    check(inp, lgs)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _run(op, state, inp, topk, out, scale)
    for _ in range(3):
        topk.fill_(-1)
        out.zero_()
        graph.replay()
        launches += 1
        check(inp, lgs)
    # Next step on the same state: new inputs of the same shapes, the state refreshed in place. Leftovers of the
    # previous generation would show up as a wrong top-512 or attention.
    inp2 = _make_case(ns, seed=seed + 1, attn=attn, **extra)
    lgs2 = [_ref_logits(inp2, b) for b in range(batch)]
    assert _prepare(op, inp2, state=state) is state
    launch(inp2)
    launches += 1
    check(inp2, lgs2)
    assert int(op.error_bits(state).max()) == 0, "kernel error bits set"
    ws = state.ws64.view(batch, mk.WS_WORDS)
    assert bool((ws[:, mk.W_DONE] == launches).all()), "one workspace generation per launch"


def test_rejects_unsupported_shapes():
    op = _op()
    dev = torch.device("cuda")
    i32 = torch.int32
    many = SPEC.megakernel_max_batch + 1
    bt = torch.zeros((many, 8), dtype=i32, device=dev)
    with pytest.raises(ValueError, match="batch size"):
        op.prepare(torch.full((many,), 4096, dtype=i32), torch.full((many,), 16384, dtype=i32), bt)
    with pytest.raises(ValueError, match="compressed lengths"):
        op.prepare(torch.tensor([300], dtype=i32), torch.tensor([1200], dtype=i32), bt[:1])
    with pytest.raises(ValueError, match="compressed lengths"):
        op.prepare(
            torch.tensor([N_MAX + 1], dtype=i32), torch.tensor([4 * N_MAX + 4], dtype=i32), bt[:1]
        )
    with pytest.raises(ValueError, match="uncompressed lengths"):
        op.prepare(torch.tensor([512], dtype=i32), torch.tensor([100], dtype=i32), bt[:1])
    with pytest.raises(ValueError, match="pages per row"):
        op.prepare(torch.tensor([4096], dtype=i32), torch.tensor([16384], dtype=i32), bt[:1])
    bt2 = torch.zeros((2, 128), dtype=i32, device=dev)
    state = op.prepare(torch.tensor([4096], dtype=i32), torch.tensor([16384], dtype=i32), bt2[:1])
    with pytest.raises(ValueError, match="state is for"):
        op.prepare(
            torch.full((2,), 4096, dtype=i32), torch.full((2,), 16384, dtype=i32), bt2, state=state
        )


def _fp4_rows(batch, step=4093, short=(), ties=None, flat=(), base=262127):
    """Specs of ``batch`` rows: lengths ``base - step * r``, the rows in ``short`` 700 + 17 r long, planted ties
    {row: count}, all-equal rows ``flat``."""
    ties = ties or {}
    return [
        {
            "n": 700 + 17 * r if r in short else base - step * r,
            "ties": ties.get(r, 0),
            "flat": r in flat,
        }
        for r in range(batch)
    ]


# The knob sets of the FP4 megakernel: 1 row, 2 rows, 3 to 4 rows (one team of 4 CTAs per row), 5 to 8 rows and 9 up to
# megakernel_max_batch rows (the ring of logits slots); from 5 rows four shared teams, two dedicated T-server CTAs and
# rotated split ranges. Odd and in-between sizes, the largest size, short rows (most stream CTAs have an empty split
# range) and rows of the longest length are all part of it.
FP4_N = SPEC_FP4.megakernel_max_batch
FP4_CASES = [
    pytest.param(specs, attn, id=f"b{len(specs)}-{name}")
    for name, specs, attn in [
        ("max", [{"n": 262127}], "gather_exact"),
        ("ties", [{"n": 8000, "ties": 9}], "gather_exact"),
        ("select-all", [{"n": 512}], "realistic"),
        ("ties", [{"n": 131071}, {"n": 65543, "ties": 7}], "gather_exact"),
        (
            "short-ties-flat",
            [{"n": 262127}, {"n": 700, "ties": 5}, {"n": 98309, "flat": True}],
            "gather_exact",
        ),
        (
            "mixed",
            [{"n": 131071}, {"n": 98309}, {"n": 50021}, {"n": 32771, "ties": 77}],
            "realistic",
        ),
        ("mixed", _fp4_rows(5, step=7001), "realistic"),
        ("max", [{"n": 262127}] * 5, "gather_exact"),
        ("short-ties", _fp4_rows(6, short=(1, 4), ties={2: 13}), "gather_exact"),
        ("flat", _fp4_rows(7, step=5003, flat=(4,)), "gather_exact"),
        ("mixed-ties", _fp4_rows(8, step=4099, ties={3: 31}), "realistic"),
        ("mixed", _fp4_rows(9, step=7001), "gather_exact"),
        ("mixed-ties", _fp4_rows(10, step=5501, ties={9: 41}), "gather_exact"),
        ("short-ties", _fp4_rows(11, short=(3, 9), ties={5: 17}), "realistic"),
        ("flat", _fp4_rows(13, step=3001, flat=(9,)), "gather_exact"),
        ("short-flat", _fp4_rows(14, short=(0, 7), flat=(11,)), "realistic"),
        ("mixed", _fp4_rows(15, step=6101), "gather_exact"),
        ("mixed-ties", _fp4_rows(16, ties={11: 29}), "realistic"),
        ("mixed", _fp4_rows(17, step=6007), "gather_exact"),
        ("short-ties", _fp4_rows(18, short=(4, 13), ties={8: 600}), "realistic"),
        (
            "short-ties-flat",
            _fp4_rows(19, short=(6, 12, 18), ties={0: 6000, 15: 77}, flat=(10,)),
            "gather_exact",
        ),
        ("mixed", _fp4_rows(20, step=5003), "realistic"),
        # rows of fewer than 8,192 positions: only the first stream CTAs have a split range
        ("all-short", [{"n": n} for n in (512, 700, 1500, 4097, 513, 8000, 2049)], "gather_exact"),
        ("all-short", [{"n": 512 + 331 * r} for r in range(12)], "realistic"),
        ("max", [{"n": 262127}] * FP4_N, "realistic"),
        (
            "mixed-short-ties-flat",
            _fp4_rows(
                FP4_N, step=3989, short=(2, 5, 8, 11, 14, 17), ties={5: 3, 7: 77}, flat=(18,)
            ),
            "gather_exact",
        ),
        ("all-short", [{"n": 512 + 97 * r} for r in range(FP4_N)], "gather_exact"),
    ]
    if len(specs) <= FP4_N
]


@pytest.mark.parametrize("specs,attn", FP4_CASES)
def test_fp4_matches_reference(specs, attn):
    """MXFP4 indexer K cache: exact top-512 sets / value multisets and the attention against the fp64 references."""
    from . import test_cute_dsl_spec_dsa_large_batch as H

    op = _op()
    batch = len(specs)
    inp, bases = H._make_case(batch, specs, seed=300 + batch, attn=attn, idx="fp4")
    state = op.prepare(
        inp["ctx_lens"].cpu(), inp["kv_lens"].cpu(), inp["idx_block_table"], config=H.SPEC_FP4
    )
    topk, out = H._new_outputs(batch)

    def launch():
        topk.fill_(-1)
        out.zero_()
        op.run(
            state,
            inp["q_idx"],
            inp["w_idx"],
            inp["k_cache"],
            inp["idx_block_table"],
            inp["ctx_lens"],
            inp["q_att"],
            inp["swa_pool"],
            inp["bt_swa"],
            inp["c_pool"],
            inp["bt_c"],
            inp["sinks"],
            topk,
            out,
            softmax_scale=SM_SCALE,
            q_scale=inp["q_scale"],
        )
        torch.cuda.synchronize()

    for _ in range(2):  # the second launch runs on the next workspace generation
        launch()
        H._check_rows(inp, bases, topk, out, attn)
    assert int(op.error_bits(state).max()) == 0, "kernel error bits set"


def test_fp4_requires_query_scales():
    from . import test_cute_dsl_spec_dsa_large_batch as H

    op = _op()
    inp, _ = H._make_case(1, H._plain([4096]), seed=5, idx="fp4")
    state = op.prepare(
        inp["ctx_lens"].cpu(), inp["kv_lens"].cpu(), inp["idx_block_table"], config=H.SPEC_FP4
    )
    topk, out = H._new_outputs(1)
    args = (
        state,
        inp["q_idx"],
        inp["w_idx"],
        inp["k_cache"],
        inp["idx_block_table"],
        inp["ctx_lens"],
        inp["q_att"],
        inp["swa_pool"],
        inp["bt_swa"],
        inp["c_pool"],
        inp["bt_c"],
        inp["sinks"],
        topk,
        out,
    )
    with pytest.raises(ValueError, match="q_scale"):
        op.run(*args)
    with pytest.raises(ValueError, match="q_scale"):
        op.run(*args, q_scale=inp["q_scale"].view(-1)[:3])
