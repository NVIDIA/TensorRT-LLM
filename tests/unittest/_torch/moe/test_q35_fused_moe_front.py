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
"""Parity and PDL-ordering tests for the Qwen3.5 MoE decode front end (q35_fused_front). Needs SM100/103."""

import contextlib
import importlib.util
import os

import pytest
import torch
import triton
import triton.language as tl

from tensorrt_llm._torch import q35_env
from tensorrt_llm._torch.moe.fused_moe.q35_fused_front import (
    Q35_MOE_ROUTER_MAX_TOKENS,
    Q35RouterGemm,
    Q35RouteTables,
    _q35_route_tables_kernel,
    _q35_router_gemm_kernel,
    q35_lowm_router,
    q35_nvfp4_quant,
    q35_prerouted_sizes,
    q35_prerouted_views,
    q35_routed_forward,
)

E, K, H = 512, 10, 4096


def _sm():
    if not torch.cuda.is_available():
        return 0
    p = torch.cuda.get_device_properties(0)
    return p.major * 10 + p.minor


pytestmark = pytest.mark.skipif(_sm() not in (100, 103), reason="needs SM100/SM103")


def _inputs(m, seed, ties=False):
    g = torch.Generator(device="cuda").manual_seed(seed)
    logits = torch.randn((m, E), generator=g, device="cuda") * 2.0
    if ties:
        # Coarse grid -> many exact bf16 ties around the top-k boundary.
        logits = torch.round(logits * 4.0) / 4.0
    logits = logits.to(torch.bfloat16)
    x = (torch.randn((m, H), generator=g, device="cuda") * 0.5).to(torch.bfloat16)
    x[0, :32] = 0  # all-zero scale blocks
    if m > 1:
        x[1, 100] = 40.0  # an outlier block
    gs = torch.tensor(
        448.0 * 6.0 / x.float().abs().max().item(), dtype=torch.float32, device="cuda"
    )
    return logits, x, gs


def _rand_fp4_moe(nl, inter, seed):
    """Random NVFP4 expert weights (valid e4m3 scales) for nl local experts."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    w1 = torch.randint(
        0, 256, (nl, 2 * inter, H // 2), generator=g, device="cuda", dtype=torch.int32
    ).to(torch.uint8)
    s1 = torch.randint(
        0x28, 0x38, (nl, 2 * inter, H // 16), generator=g, device="cuda", dtype=torch.int32
    ).to(torch.uint8)
    w2 = torch.randint(
        0, 256, (nl, H, inter // 2), generator=g, device="cuda", dtype=torch.int32
    ).to(torch.uint8)
    s2 = torch.randint(
        0x28, 0x38, (nl, H, inter // 16), generator=g, device="cuda", dtype=torch.int32
    ).to(torch.uint8)
    a1 = torch.rand((nl,), generator=g, device="cuda") * 1e-2 + 1e-3
    a2 = torch.rand((nl,), generator=g, device="cuda") * 1e-2 + 1e-3
    return w1, s1, a1, w2, s2, a2


@pytest.mark.parametrize("m", [1, 7, 17, 32])
def test_router_gemm(m):
    g = torch.Generator(device="cuda").manual_seed(11 + m)
    x = torch.randn((m, H), generator=g, device="cuda").to(torch.bfloat16)
    w = (torch.randn((E, H), generator=g, device="cuda") * 0.02).to(torch.bfloat16)
    ref = torch.ops.trtllm.cublas_mm(x, w.t(), bias=None, out_dtype=torch.bfloat16)
    rg = Q35RouterGemm()
    a = rg(x, w)
    b = rg(x, w)
    gs = torch.tensor(
        448.0 * 6.0 / x.float().abs().max().item(), dtype=torch.float32, device="cuda"
    )
    c, cq, csf = rg(x, w, gs)
    xq, xsf = torch.ops.trtllm.fp4_quantize(x, gs, 16, False, False)
    torch.cuda.synchronize()
    assert torch.equal(a, b), "router GEMM not deterministic"
    assert torch.equal(a, c), "router GEMM differs with fused quant"
    assert torch.equal(cq, xq.view(torch.uint8)), "fused quant payload"
    assert torch.equal(csf, xsf.view(m, -1)), "fused quant scales"
    sq, ssf = q35_nvfp4_quant(x, gs)
    torch.cuda.synchronize()
    assert torch.equal(sq, xq.view(torch.uint8)), "standalone quant payload"
    assert torch.equal(ssf, xsf.view(m, -1)), "standalone quant scales"
    exact = x.float() @ w.float().t()
    err_mine = (a.float() - exact).abs().max().item()
    err_ref = (ref.float() - exact).abs().max().item()
    # Both round an fp32 accumulation to bf16: errors of the same order.
    assert err_mine <= 2 * err_ref + 1e-3, (m, err_mine, err_ref)


@triton.jit
def _delayed_producer_kernel(src_ptr, dst_ptr, n, spin_ns, BLOCK: tl.constexpr):
    """Adversarial PDL producer: releases its dependents first, then spins
    ~spin_ns, then writes dst (the consumer's input)."""
    tl.extra.cuda.gdc_launch_dependents()
    t0 = tl.inline_asm_elementwise(
        "mov.u64 $0, %globaltimer;", "=l", [], dtype=tl.int64, is_pure=False, pack=1
    )
    t1 = t0
    while t1 - t0 < spin_ns:
        t1 = tl.inline_asm_elementwise(
            "mov.u64 $0, %globaltimer;", "=l", [], dtype=tl.int64, is_pure=False, pack=1
        )
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(dst_ptr + offs, tl.load(src_ptr + offs, mask=offs < n), mask=offs < n)


@triton.jit
def _nowait_copy_kernel(src_ptr, dst_ptr, n, BLOCK: tl.constexpr):
    """Positive control: a PDL dependent that does NOT griddepcontrol.wait."""
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(dst_ptr + offs, tl.load(src_ptr + offs, mask=offs < n), mask=offs < n)


def test_pdl_delayed_producer(m=7):
    """Router GEMM + quant, standalone quant and low-M router must not read x before
    griddepcontrol.wait: a delayed producer releases dependents before writing x; graph
    replays must match the eager reference (with a no-wait positive control)."""
    g = torch.Generator(device="cuda").manual_seed(5)
    x_src = torch.randn((m, H), generator=g, device="cuda").to(torch.bfloat16)
    w = (torch.randn((E, H), generator=g, device="cuda") * 0.02).to(torch.bfloat16)
    gs = torch.tensor(
        448.0 * 6.0 / x_src.float().abs().max().item(), dtype=torch.float32, device="cuda"
    )
    x = torch.full_like(x_src, 3.0e4)
    rg = Q35RouterGemm()
    ref_l, ref_q, ref_sf = rg(x_src, w, gs)
    ref_q2, ref_sf2 = q35_nvfp4_quant(x_src, gs)
    ref_lowm = q35_lowm_router(x_src, w, gs)
    n = x.numel()

    y = torch.empty_like(x)

    def producer():
        x.fill_(3.0e4)
        _delayed_producer_kernel[(triton.cdiv(n, 4096),)](
            x_src, x, n, 50000, BLOCK=4096, launch_pdl=True
        )

    def step():
        producer()
        out = rg(x, w, gs)
        out2 = q35_nvfp4_quant(x, gs)
        out3 = q35_lowm_router(x, w, gs)
        return out, out2, out3

    def control():
        producer()
        _nowait_copy_kernel[(triton.cdiv(n, 4096),)](x, y, n, BLOCK=4096, launch_pdl=True)

    step()
    control()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        (lg, q, sf), (q2, sf2), lowm_out = step()
    cgraph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(cgraph):
        control()
    bad = 0
    stale = 0
    for _ in range(20):
        graph.replay()
        torch.cuda.synchronize()
        ok = (
            torch.equal(lg, ref_l)
            and torch.equal(q, ref_q)
            and torch.equal(sf, ref_sf)
            and torch.equal(q2, ref_q2)
            and torch.equal(sf2, ref_sf2)
        )
        if ref_lowm is not None:
            ok = ok and all(torch.equal(a, b) for a, b in zip(lowm_out, ref_lowm))
        bad += int(not ok)
        cgraph.replay()
        torch.cuda.synchronize()
        stale += int(not torch.equal(y, x_src))
    assert stale > 0, "positive control never read early: the test cannot detect a missing wait"
    assert bad == 0, f"{bad}/20 replays read the producer's data early"


def test_pdl_prerouted_chain(m=7, nl=128, ls=128, inter=512, tile=8):
    """Front-end launch chains under PDL in a CUDA graph behind a delayed producer: router +
    quant -> stock MoE; -> route + tables -> pre-routed MoE; low-M router -> route + tables +
    x quant -> pre-routed MoE. Every replay must equal the eager result bitwise."""
    from tensorrt_llm._torch.custom_ops.trtllm_gen_custom_ops import FP4BlockScaleMoERunner

    runner, inputs, _ = _prerouted_runner_inputs(m, nl, ls, inter, seed=1400 + m)
    g = torch.Generator(device="cuda").manual_seed(9)
    x_src = torch.randn((m, H), generator=g, device="cuda").to(torch.bfloat16)
    w = (torch.randn((E, H), generator=g, device="cuda") * 0.02).to(torch.bfloat16)
    gs = torch.tensor(
        448.0 * 6.0 / x_src.float().abs().max().item(), dtype=torch.float32, device="cuda"
    )
    x = torch.full_like(x_src, 3.0e4)
    n = x.numel()
    cfg = sorted(
        int(c)
        for t, c in runner.get_runner().get_valid_configs(K, H, inter, nl, m)
        if int(t) == tile
    )[0]
    has_pre = FP4BlockScaleMoERunner.supports_prerouted()
    rg = Q35RouterGemm()
    rt = Q35RouteTables()
    rt2 = Q35RouteTables()
    flat = torch.empty(
        (sum(q35_prerouted_sizes(m, K, nl, tile)),), dtype=torch.int32, device="cuda"
    )
    flat2 = torch.empty_like(flat)

    def moe(lg, xq, xsf, pre=None):
        inp = list(inputs)
        inp[0], inp[2], inp[3] = lg, xq, xsf.view(-1).view(torch.float8_e4m3fn)
        if pre is None:
            return runner(inp, tactic=[tile, cfg])[0]
        return runner(inp, tactic=[tile, cfg], prerouted=pre, prerouted_tile=tile)[0]

    def chain():
        lg, xq, xsf = rg(x, w, gs)
        res, bufs = [moe(lg, xq, xsf)], [lg, xq, xsf, flat]
        if has_pre:
            ids, wts = rt(lg, K, ls, nl, tile, flat=flat)[:2]
            res.append(moe(lg, xq, xsf, (q35_prerouted_views(flat, m, K, nl, tile), ids, wts)))
            bufs += [ids, wts]
            lg2 = q35_lowm_router(x, w)
            if lg2 is not None:
                o = rt2(lg2, K, ls, nl, tile, flat=flat2, x=x, global_scale=gs)
                res.append(
                    moe(lg2, o[8], o[9], (q35_prerouted_views(flat2, m, K, nl, tile), o[0], o[1]))
                )
                bufs += [lg2, o[0], o[1], o[8], o[9], flat2]
        return res, bufs

    def producer():
        x.fill_(3.0e4)
        _delayed_producer_kernel[(triton.cdiv(n, 4096),)](
            x_src, x, n, 50000, BLOCK=4096, launch_pdl=True
        )

    x.copy_(x_src)
    ref = [t.clone() for t in chain()[0]]

    def step():
        producer()
        return chain()

    step()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outs, bufs = step()
    bad = 0
    for _ in range(20):
        for b in bufs:  # the graph's intermediates: a too-early reader sees zeros
            b.zero_()
        graph.replay()
        torch.cuda.synchronize()
        bad += int(
            not all(
                torch.equal(a.view(torch.int16), b.view(torch.int16)) for a, b in zip(outs, ref)
            )
        )
    assert bad == 0, f"{bad}/20 replays differ from the eager result"


def _pdl_ptx_check():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pdl_ptx_check.py")
    spec = importlib.util.spec_from_file_location("pdl_ptx_check", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _launch_route_tables(logits, ls, nl, tile, x=None, gs=None):
    """The launch Q35RouteTables makes, returning the CompiledKernel (for its PTX)."""
    m, e = logits.shape
    dev = logits.device
    nlp = triton.next_power_of_2(nl)
    hist = torch.zeros((nlp,), dtype=torch.int32, device=dev)
    counter = torch.zeros((1,), dtype=torch.int32, device=dev)
    ids = torch.empty((m, K), dtype=torch.int32, device=dev)
    wts = torch.empty((m, K), dtype=torch.bfloat16, device=dev)
    rank = torch.empty((m * K,), dtype=torch.int32, device=dev)
    flat = torch.empty((sum(q35_prerouted_sizes(m, K, nl, tile)),), dtype=torch.int32, device=dev)
    tables = q35_prerouted_views(flat, m, K, nl, tile)
    quant = x is not None
    if quant:
        xq = torch.empty((m, H // 2), dtype=torch.uint8, device=dev)
        xsf = torch.empty((m, H // 16), dtype=torch.uint8, device=dev)
        qargs = (x, x.stride(0), gs, xq, xsf)
    else:
        qargs = (logits, 0, counter, counter, counter)
    return _q35_route_tables_kernel[(2 * m if quant else m,)](
        logits,
        ids,
        wts,
        hist,
        rank,
        counter,
        *tables,
        m,
        ls,
        *qargs,
        E=e,
        K=K,
        KP=16,
        TILE=tile,
        NL=nl,
        NLP=nlp,
        EXPANDED_PAD=triton.next_power_of_2(Q35_MOE_ROUTER_MAX_TOKENS * K),
        MAX_CTAS_PER_EXPERT=triton.cdiv(Q35_MOE_ROUTER_MAX_TOKENS, tile),
        USE_PDL=True,
        QUANT_H=H if quant else 0,
        num_warps=4,
        launch_pdl=True,
    )


def test_route_kernel_waits_before_global_access(m=7):
    """The route + tables launch (with and without x quant) touches no global memory before
    griddepcontrol.wait and triggers dependents after its last access (PTX order)."""
    logits, x, gs = _inputs(m, seed=1700)
    pdl = _pdl_ptx_check()
    for tag, kw in (
        ("ep", dict(ls=448, nl=64)),
        ("ep+quant", dict(ls=128, nl=128, x=x, gs=gs)),
    ):
        ck = _launch_route_tables(logits, tile=8, **kw)
        torch.cuda.synchronize()
        pdl.assert_wait_first(ck, f"route {tag}")
        pdl.assert_trigger_last(ck, f"route {tag}")


def test_router_weight_load_before_pdl_wait(m=7):
    """The router GEMM issues its weight loads before griddepcontrol.wait (they overlap the
    producer) and loads x, stores and atomics only after it (PTX order)."""
    x = torch.randn((m, H), device="cuda").to(torch.bfloat16)
    w = (torch.randn((E, H), device="cuda") * 0.02).to(torch.bfloat16)
    rg = Q35RouterGemm()
    rg(x, w)
    out = torch.empty((m, E), dtype=torch.bfloat16, device="cuda")
    # QUANT=False: the global scale / xq / xsf pointers are unused.
    ck = _q35_router_gemm_kernel[(E // rg.BN * rg.SPLITK,)](
        x,
        x.stride(0),
        w,
        out,
        rg.part,
        rg.cnt,
        rg.cnt,
        out,
        out,
        m,
        N=E,
        K=H,
        BM=16,
        BN=rg.BN,
        SPLITK=rg.SPLITK,
        QUANT=False,
        USE_PDL=True,
        num_warps=4,
        launch_pdl=True,
    )
    torch.cuda.synchronize()
    ptx = ck.asm["ptx"]
    i_wait = ptx.find("griddepcontrol.wait")
    loads = [i for i in range(len(ptx)) if ptx.startswith("ld.global", i)]
    n_before = sum(1 for i in loads if i < i_wait)
    assert i_wait > 0 and n_before > 0, "no weight load issued before the PDL wait"
    assert n_before < len(loads), "x must be loaded after the PDL wait"
    pdl = _pdl_ptx_check()
    pdl.assert_wait_first(ck, "router", allow_loads_before=True)
    pdl.assert_trigger_last(ck, "router")


ROUTE_CASES = [
    (m, nl, ls, tile) for m in (1, 7, 32) for (nl, ls) in ((512, 0), (64, 448)) for tile in (8, 16)
]


@pytest.mark.parametrize("m,nl,ls,tile", ROUTE_CASES)
@pytest.mark.parametrize("ties", [False, True])
def test_route_tables(m, nl, ls, tile, ties):
    """Q35RouteTables vs the trtllm-gen routing runner (moe_topk_sort, Renormalize)."""
    logits, _, _ = _inputs(m, seed=500 + m + nl + tile, ties=ties)
    rt = Q35RouteTables()
    ids, wts, e2p, p2t, cte, ctl, nct, npd = rt(logits, K, ls, nl, tile)
    ref = torch.ops.trtllm.moe_topk_sort(logits, None, E, K, None, None, ls, nl, None, tile, 1)
    r_te, r_ml, r_e2p, r_p2e, r_tot, r_nt = ref[:6]
    r_w = ref[6] if len(ref) > 6 else None
    torch.cuda.synchronize()
    tag = (m, nl, ls, tile, ties)
    nt = int(r_nt.item())
    assert int(nct.item()) == nt, (tag, nt, int(nct.item()))
    assert int(npd.item()) == int(r_tot.item()), tag
    assert torch.equal(cte[:nt], r_te[:nt]), tag
    assert torch.equal(ctl[:nt], r_ml[:nt]), tag
    re2p = r_e2p.view(-1)
    assert torch.equal(e2p < 0, re2p < 0), tag
    valid = (e2p >= 0).nonzero().view(-1)
    # same expanded-index set per CTA (row order inside a block is arbitrary)
    m_p2e = torch.full_like(p2t, -1)
    m_p2e[e2p[valid].long()] = valid.to(torch.int32)
    for t in range(nt):
        lo, hi = t * tile, int(r_ml[t].item())
        assert torch.equal(torch.sort(m_p2e[lo:hi]).values, torch.sort(r_p2e[lo:hi]).values), (
            tag,
            t,
        )
    assert torch.equal(p2t[e2p[valid].long()], (valid // K).to(torch.int32)), tag
    # ids: the same per-token sets as the reference top-k; renormalized weights
    # within bf16 rounding of the runner's.
    ref_ids = torch.topk(logits.float(), K, dim=-1).indices
    assert torch.equal(torch.sort(ids.long(), -1).values, torch.sort(ref_ids, -1).values) or ties, (
        tag
    )
    if r_w is not None:
        # r_w is indexed by (token, k) in the runner's own k order: compare as sets
        assert torch.allclose(
            torch.sort(wts.float(), -1).values,
            torch.sort(r_w.float().view(m, K), -1).values,
            atol=1e-2,
            rtol=0,
        ), tag
        if r_w.dtype == wts.dtype:
            # same dtype as the runner's weights: bit-identical (calcSoftmax order)
            assert torch.equal(
                torch.sort(wts.float(), -1).values, torch.sort(r_w.float().view(m, K), -1).values
            ), tag
    # the histogram and the ticket must be back to zero for the next launch
    assert int(rt.hist.abs().sum().item()) == 0 and int(rt.counter.item()) == 0


def _check_tables_equivalent(got, ref, tile, tag):
    """Two routings' tables describe the same layout; row order inside an expert's block is
    free (the per-token kernel ranks rows by atomics)."""
    e2p, p2t, cte, ctl, nct, npd = got
    r_e2p, r_p2t, r_cte, r_ctl, r_nct, r_npd = ref
    nt = int(r_nct.item())
    assert int(nct.item()) == nt and int(npd.item()) == int(r_npd.item()), (tag, "counts")
    assert torch.equal(cte[:nt], r_cte[:nt]) and torch.equal(ctl[:nt], r_ctl[:nt]), (tag, "ctas")
    assert torch.equal(e2p < 0, r_e2p < 0), (tag, "local picks")
    valid = (e2p >= 0).nonzero().view(-1)
    got_p2e = torch.full((nt * tile,), -1, dtype=torch.int64, device=e2p.device)
    ref_p2e = torch.full((nt * tile,), -1, dtype=torch.int64, device=e2p.device)
    got_p2e[e2p[valid].long()] = valid
    ref_p2e[r_e2p[valid].long()] = valid
    for t in range(nt):
        lo, hi = t * tile, int(r_ctl[t].item())
        assert torch.equal(torch.sort(got_p2e[lo:hi]).values, torch.sort(ref_p2e[lo:hi]).values), (
            tag,
            "cta",
            t,
        )
    assert torch.equal(p2t[e2p[valid].long()], (valid // K).to(torch.int32)), (tag, "p2t")


ROUTE_QUANT_CASES = [(1, 64, 448, 8), (7, 128, 128, 8), (32, 64, 448, 16)]


@pytest.mark.parametrize("m,nl,ls,tile", ROUTE_QUANT_CASES)
def test_route_quant(m, nl, ls, tile):
    """Route + tables with x quant: xq / xsf byte-identical to q35_nvfp4_quant, ids / weights
    and tables equal to the launch without x, histogram and ticket back at zero."""
    logits, x, gs = _inputs(m, seed=1600 + m + nl + tile)
    sizes = sum(q35_prerouted_sizes(m, K, nl, tile))
    rt = Q35RouteTables()
    flat_r = torch.full((sizes,), -7, dtype=torch.int32, device="cuda")
    ids_r, wts_r = rt(logits, K, ls, nl, tile, flat=flat_r)[:2]
    flat = torch.full((sizes,), -7, dtype=torch.int32, device="cuda")
    out = rt(logits, K, ls, nl, tile, flat=flat, x=x, global_scale=gs)
    xq_r, xsf_r = q35_nvfp4_quant(x, gs)
    torch.cuda.synchronize()
    tag = (m, nl, ls, tile)
    assert len(out) == 10, tag
    assert torch.equal(out[0], ids_r) and torch.equal(out[1], wts_r), (tag, "ids")
    assert torch.equal(out[8], xq_r) and torch.equal(out[9], xsf_r), (tag, "quant")
    _check_tables_equivalent(
        q35_prerouted_views(flat, m, K, nl, tile),
        q35_prerouted_views(flat_r, m, K, nl, tile),
        tile,
        tag,
    )
    assert int(rt.hist.abs().sum().item()) == 0 and int(rt.counter.item()) == 0, (tag, "hist")


# (tokens, local experts, first local expert, MoE intermediate size per rank): TP4xEP2, TP2xEP4,
# TP2xEP8 (TEP16) shards of the 512-expert layer, plus one all-local case.
PREROUTED_CASES = [
    (1, 128, 128, 512),
    (7, 128, 128, 512),
    (28, 128, 128, 512),
    (7, 64, 448, 512),
    (14, 256, 256, 256),
    (7, 512, 0, 128),
]


def _prerouted_runner_inputs(m, nl, ls, inter, seed):
    from tensorrt_llm._torch.custom_ops.trtllm_gen_custom_ops import FP4BlockScaleMoERunner
    from tensorrt_llm._torch.moe.fused_moe.routing import RoutingMethodType
    from tensorrt_llm._torch.utils import ActType_TrtllmGen

    w1, s1, a1, w2, s2, a2 = _rand_fp4_moe(nl, inter, seed=seed)
    logits, x, gs = _inputs(m, seed=seed + 1)
    xq, xsf = q35_nvfp4_quant(x, gs)
    runner = FP4BlockScaleMoERunner(
        E,
        K,
        None,
        None,
        inter,
        ls,
        nl,
        None,
        int(RoutingMethodType.Renormalize),
        True,
        ActType_TrtllmGen.SwiGlu.value,
    )
    e4m3 = torch.float8_e4m3fn
    inputs = [
        logits,
        None,
        xq,
        xsf.view(-1).view(e4m3),
        w1,
        s1.view(e4m3),
        None,
        None,
        None,
        None,
        w2,
        s2.view(e4m3),
        None,
        a1,
        a1,
        a2,
        None,
        None,
    ]
    return runner, inputs, logits


@pytest.mark.parametrize("m,nl,ls,inter", PREROUTED_CASES)
def test_prerouted_moe_parity(m, nl, ls, inter):
    """run_moe_prerouted with Q35RouteTables' tables vs the stock op routing the same logits:
    bit-identical for several tactics of each tile; corrupting one CTA's expert must change
    the output (negative control)."""
    from tensorrt_llm._torch.custom_ops.trtllm_gen_custom_ops import FP4BlockScaleMoERunner

    if not FP4BlockScaleMoERunner.supports_prerouted():
        pytest.skip("libth_common without run_moe_prerouted")
    runner, inputs, logits = _prerouted_runner_inputs(m, nl, ls, inter, seed=700 + m + nl)
    cfgs = runner.get_runner().get_valid_configs(K, H, inter, nl, m)
    by_tile = {}
    for tile, cfg in cfgs:
        by_tile.setdefault(int(tile), []).append(int(cfg))
    assert by_tile, (m, nl, "no tactics")
    rt = Q35RouteTables()
    n_checked = 0
    for tile, tile_cfgs in sorted(by_tile.items()):
        flat = torch.full(
            (sum(q35_prerouted_sizes(m, K, nl, tile)),), -7, dtype=torch.int32, device="cuda"
        )
        ids, wts = rt(logits, K, ls, nl, tile, flat=flat)[:2]
        tables = q35_prerouted_views(flat, m, K, nl, tile)
        # first, last and one middle config of this tile
        picks = sorted({tile_cfgs[0], tile_cfgs[len(tile_cfgs) // 2], tile_cfgs[-1]})
        for cfg in picks:
            tag = (m, nl, ls, inter, tile, cfg)
            ref = runner(inputs, tactic=[tile, cfg])[0]
            pre = runner(
                inputs, tactic=[tile, cfg], prerouted=(tables, ids, wts), prerouted_tile=tile
            )[0]
            torch.cuda.synchronize()
            assert ref.shape == pre.shape, tag
            assert torch.equal(ref.view(torch.int16), pre.view(torch.int16)), (
                tag,
                (ref.float() - pre.float()).abs().max().item(),
            )
            n_checked += 1
        # negative control: route the first CTA to the next local expert
        nct = int(tables[4].item())
        if nct > 0:
            cte = tables[2]
            saved = cte[0].clone()
            cte[0] = (saved + 1) % nl
            bad = runner(
                inputs, tactic=[tile, picks[0]], prerouted=(tables, ids, wts), prerouted_tile=tile
            )[0]
            ref = runner(inputs, tactic=[tile, picks[0]])[0]
            torch.cuda.synchronize()
            cte[0] = saved
            assert not torch.equal(ref.view(torch.int16), bad.view(torch.int16)), (
                m,
                nl,
                tile,
                "control",
            )
    assert n_checked > 0


def _finalize_ref(fc2, weights, perm, hidden):
    """The trtllm-gen finalize kernel's result: FP32 sum over k of weight * FC2 row
    (expanded -> permuted index, -1 = another rank's pick) in k order, BF16 once."""
    acc = torch.zeros((perm.shape[0], hidden), dtype=torch.float32, device=fc2.device)
    for k in range(perm.shape[1]):
        p = perm[:, k]
        rows = fc2[p.clamp(min=0).long(), :hidden].float()
        acc = torch.where((p >= 0)[:, None], acc + weights[:, k : k + 1].float() * rows, acc)
    return acc.to(torch.bfloat16)


@contextlib.contextmanager
def _pinned_tactic(inter, nl, m, tile, cfg):
    """Make the FP4 MoE op run [tile, cfg] at this shape via the Qwen3.5 tactic table."""
    from tensorrt_llm._torch.custom_ops import trtllm_gen_custom_ops as ops

    saved = (ops._Q35_SM103_MOE_TACTICS, ops._q35_tactic_valid, q35_env.fast_paths_enabled())
    ops._Q35_SM103_MOE_TACTICS = {(inter, nl, m): [tile, cfg]}
    ops._q35_tactic_valid = {}
    q35_env.set_fast_paths(True)
    try:
        yield
    finally:
        ops._Q35_SM103_MOE_TACTICS, ops._q35_tactic_valid = saved[0], saved[1]
        q35_env.set_fast_paths(saved[2])


UNFINALIZED_CASES = [(7, 64, 448, 512), (28, 128, 128, 512)]


@pytest.mark.parametrize("m,nl,ls,inter", UNFINALIZED_CASES)
def test_prerouted_unfinalized_op(m, nl, ls, inter, tile=8):
    """do_finalize=False on the pre-routed path through the custom op (placeholders for the
    aliased outputs) and TRTLLMOpBackend (which puts the caller's tensors back): equal to the
    stock op's unfinalized output by expanded index, and finalizing to its finalized output."""
    from tensorrt_llm._torch.custom_ops.trtllm_gen_custom_ops import FP4BlockScaleMoERunner
    from tensorrt_llm._torch.moe.fused_moe.moe_op_backend import TRTLLMOpBackend
    from tensorrt_llm._torch.moe.fused_moe.routing import RoutingMethodType
    from tensorrt_llm._torch.utils import ActType_TrtllmGen

    if not FP4BlockScaleMoERunner.supports_prerouted():
        pytest.skip("libth_common without run_moe_prerouted")
    runner, inputs, logits = _prerouted_runner_inputs(m, nl, ls, inter, seed=1100 + m + nl)
    rmt = int(RoutingMethodType.Renormalize)
    act = ActType_TrtllmGen.SwiGlu.value
    runner_nf = FP4BlockScaleMoERunner(E, K, None, None, inter, ls, nl, None, rmt, False, act)
    cfgs = sorted(
        int(c)
        for t, c in runner.get_runner().get_valid_configs(K, H, inter, nl, m)
        if int(t) == tile
    )
    assert cfgs, (m, nl, "no tile-8 config")
    cfg = cfgs[len(cfgs) // 2]
    flat = torch.full(
        (sum(q35_prerouted_sizes(m, K, nl, tile)),), -7, dtype=torch.int32, device="cuda"
    )
    ids, wts = Q35RouteTables()(logits, K, ls, nl, tile, flat=flat)[:2]
    tag = (m, nl, ls, inter, cfg)
    # the op runs the pre-routed path only for the tile of its tactic
    with _pinned_tactic(inter, nl, m, tile, cfg):
        op_args = (*inputs[:16], E, K, None, None, inter, ls, nl, None, rmt)
        raw = torch.ops.trtllm.fp4_block_scale_moe_runner(
            *op_args,
            False,
            act,
            prerouted=flat,
            prerouted_tile=tile,
            prerouted_ids=ids,
            prerouted_weights=wts,
        )
        out = TRTLLMOpBackend().run_fp4_block_scale_moe(
            *op_args,
            do_finalize=False,
            topk_weights=None,
            topk_ids=None,
            gated_act_type=act,
            prerouted=(flat, tile, ids, wts),
        )
    ref = runner_nf(inputs, tactic=[tile, cfg])
    fin = runner(inputs, tactic=[tile, cfg])[0]
    torch.cuda.synchronize()
    assert raw[1].numel() == 0 and raw[2].numel() == 0, (tag, "pre-routed path not taken")
    fc2, w, perm = out
    assert fc2.dtype == torch.bfloat16 and fc2.dim() == 2 and fc2.shape[1] >= H, (tag, fc2.shape)
    assert w.dtype == torch.bfloat16 and tuple(w.shape) == (m, K), (tag, w.dtype, w.shape)
    assert perm.dtype == torch.int32 and tuple(perm.shape) == (m, K), (tag, perm.dtype, perm.shape)
    assert w.data_ptr() == wts.data_ptr() and perm.data_ptr() == flat.data_ptr(), tag
    assert torch.equal(w.view(torch.int16), ref[1].view(torch.int16)), (tag, "weights")
    p_pre, p_ref = perm.view(-1).cpu(), ref[2].view(-1).cpu()
    for i in range(m * K):
        a, b = int(p_pre[i]), int(p_ref[i])
        assert (a >= 0) == (b >= 0), (tag, i, a, b)
        if a >= 0:
            assert torch.equal(fc2[a, :H].view(torch.int16), ref[0][b, :H].view(torch.int16)), (
                tag,
                i,
                "row",
            )
    emu = _finalize_ref(fc2, w, perm, H)
    assert torch.equal(emu.view(torch.int16), fin[:, :H].view(torch.int16)), (
        tag,
        (emu.float() - fin[:, :H].float()).abs().max().item(),
    )


def test_routed_forward_keeps_token_count_dynamic_under_compile():
    """Tracing the front end leaves the token dimension symbolic (piecewise CUDA graphs look for it)."""
    gate = type("Gate", (), {"out_dtype": torch.bfloat16})()

    def routed(x):
        out = q35_routed_forward(gate, object(), x)
        return x * 2 if out is None else out

    x = torch.randn(70, 16, dtype=torch.bfloat16, device="cuda")  # 70 is in Q35_MOE_FRONT_BIG_MS
    torch._dynamo.reset()
    torch._dynamo.mark_dynamic(x, 0)
    out = torch.compile(routed, backend="eager", fullgraph=True)(x)
    torch.testing.assert_close(out, x * 2)


def test_router_gate_under_compile_matches_eager():
    """The Qwen3.5 router gate gives the eager logits inside a torch.compile'd (functionalized) graph."""
    from tensorrt_llm._torch.models.modeling_qwen3_next import Qwen3NextGate

    gate = Qwen3NextGate(hidden_size=H, num_experts=E, top_k=K, dtype=torch.bfloat16).cuda()
    gate.weight.data.normal_(0, 0.02)
    x = torch.randn(8, H, dtype=torch.bfloat16, device="cuda")
    ref = (x.float() @ gate.weight.float().t()).to(torch.bfloat16)
    q35_env.set_fast_paths(True)
    try:
        eager = gate(x)
        torch._dynamo.reset()
        compiled = torch.compile(gate.forward, backend="aot_eager", fullgraph=True)(x)
    finally:
        q35_env.set_fast_paths(False)
    torch.testing.assert_close(eager, ref, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(compiled, ref, rtol=2e-2, atol=2e-2)
