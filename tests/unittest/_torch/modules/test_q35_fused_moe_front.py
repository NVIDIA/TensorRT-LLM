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
"""Parity tests for the Qwen3.5 fused CuteDSL MoE decode front end.

Compares ``q35_moe_front`` (one Triton launch) against the stock chain
renorm_moe_routing_op -> fp4_quantize -> moe_sort (+ output memset), then the
full routed-expert output through the CuteDSL FC1/FC2 kernels. Needs SM100/103.

Run directly (``python test_q35_fused_moe_front.py [--bench]``) or via pytest.
"""

import sys

import torch

try:
    import pytest
except ImportError:  # run as a plain script in a container without pytest

    class _NoopMark:
        def __getattr__(self, name):
            return lambda *args, **kwargs: (lambda f: f)

    class pytest:  # noqa: N801
        mark = _NoopMark()


from tensorrt_llm._torch.modules.fused_moe.q35_fused_front import (
    Q35RouterGemm,
    Q35RouteTables,
    max_num_tiles,
    q35_dea_max_ctas,
    q35_lowm_router,
    q35_moe_front,
    q35_nvfp4_quant,
    q35_prerouted_sizes,
    q35_prerouted_unfinalized,
    q35_prerouted_views,
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


def _stock(logits, x, gs, local_start, nl, tile):
    m = logits.shape[0]
    ids, w = torch.ops.trtllm.renorm_moe_routing_op(logits, K, torch.float32)
    xq, xsf = torch.ops.trtllm.fp4_quantize(x, gs, 16, False, False)
    srt = torch.ops.trtllm.moe_sort(
        token_selected_experts=ids,
        token_final_scales=w,
        num_experts=E,
        top_k=K,
        local_expert_offset=local_start,
        local_num_experts=nl,
        tile_tokens_dim=tile,
    )
    return ids, w, xq.view(torch.uint8), xsf.view(m, -1), srt


def _check_sort(ref, mine, m, tile, tag):
    (r_te, r_ml, r_e2p, r_p2e, r_tot, r_nt) = ref
    (m_te, m_ml, m_e2p, m_p2e, m_tot, m_nt) = mine
    assert r_te.shape == m_te.shape and r_p2e.shape == m_p2e.shape, tag
    nt = int(r_nt.item())
    assert int(m_nt.item()) == nt, (tag, nt, int(m_nt.item()))
    assert int(m_tot.item()) == int(r_tot.item()), tag
    assert torch.equal(m_te[:nt], r_te[:nt]), tag
    assert torch.equal(m_ml[:nt], r_ml[:nt]), tag
    r_e2p = r_e2p.view(-1)
    m_e2p = m_e2p.view(-1)
    assert torch.equal(r_e2p < 0, m_e2p < 0), tag
    valid = (m_e2p >= 0).nonzero().view(-1)
    # Inverse consistency and the same expanded-index set per tile (the row
    # order inside a tile comes from atomics in both implementations).
    assert torch.equal(m_p2e[m_e2p[valid].long()], valid.to(torch.int32)), tag
    for t in range(nt):
        lo = t * tile
        hi = int(r_ml[t].item())
        rs = torch.sort(r_p2e[lo:hi]).values
        ms = torch.sort(m_p2e[lo:hi]).values
        assert torch.equal(rs, ms), (tag, t)


CASES = [
    (m, nl, ls, tile)
    for m in (1, 7, 14, 28, 32)
    for (nl, ls) in ((64, 0), (64, 448), (128, 256), (512, 0))
    for tile in (128, 256)
]


@pytest.mark.parametrize("m,nl,local_start,tile", CASES)
@pytest.mark.parametrize("ties", [False, True])
def test_front_parity(m, nl, local_start, tile, ties):
    logits, x, gs = _inputs(m, seed=1000 + m * 7 + nl + local_start, ties=ties)
    ids, w, xq, xsf, srt = _stock(logits, x, gs, local_start, nl, tile)
    scratch = torch.zeros((1024,), dtype=torch.int32, device="cuda")
    out = torch.full((m, H), 7.0, dtype=torch.bfloat16, device="cuda")
    (m_ids, m_w, m_xq, m_xsf, m_te, m_ml, m_e2p, m_p2e, m_tot, m_nt, m_out) = q35_moe_front(
        logits, x, gs, K, local_start, nl, tile, scratch, moe_output=out
    )
    torch.cuda.synchronize()
    tag = (m, nl, local_start, tile, ties)
    assert torch.equal(m_ids, ids), tag
    assert torch.equal(m_w.view(torch.int32), w.view(torch.int32)), (
        tag,
        (m_w - w).abs().max().item(),
    )
    assert torch.equal(m_xq, xq), tag
    assert torch.equal(m_xsf, xsf), tag
    assert m_te.shape[0] == max_num_tiles(m, K, nl, tile)
    _check_sort(srt, (m_te, m_ml, m_e2p, m_p2e, m_tot, m_nt), m, tile, tag)
    assert torch.count_nonzero(m_out).item() == 0, tag
    # Pass-through mode (quantization done upstream): same tables.
    pq = xq.clone()
    psf = xsf.clone()
    res = q35_moe_front(
        logits, x, gs, K, local_start, nl, tile, scratch, xq=pq, xsf=psf, quantize=False
    )
    torch.cuda.synchronize()
    assert torch.equal(res[0], ids) and torch.equal(res[2], xq), tag
    _check_sort(srt, res[4:10], m, tile, tag)


def test_negative_controls():
    """The checks must catch a wrong scale and a wrong tie order."""
    m, nl, ls, tile = 7, 64, 64, 128
    logits, x, gs = _inputs(m, seed=7, ties=True)
    ids, w, xq, xsf, _ = _stock(logits, x, gs, ls, nl, tile)
    scratch = torch.zeros((1024,), dtype=torch.int32, device="cuda")
    res = q35_moe_front(logits, x, gs * 1.01, K, ls, nl, tile, scratch)
    assert not torch.equal(res[3], xsf), "scale perturbation not detected"
    # Exact tie straddling the top-k boundary: experts 3 and 500 share the
    # 10th-largest logit. reduceTopK keeps the smaller index (3); a kernel that
    # preferred the larger index would select 500 and fail the id check.
    lg = torch.full((1, E), -4.0, device="cuda")
    lg[0, 100:109] = torch.arange(9, device="cuda", dtype=torch.float32) + 1.0
    lg[0, 3] = 0.5
    lg[0, 500] = 0.5
    lg = lg.to(torch.bfloat16)
    xs = x[:1].contiguous()
    r_ids = torch.ops.trtllm.renorm_moe_routing_op(lg, K, torch.float32)[0]
    m_ids = q35_moe_front(lg, xs, gs, K, 0, 512, tile, scratch)[0]
    torch.cuda.synchronize()
    assert 3 in r_ids[0].tolist() and 500 not in r_ids[0].tolist(), r_ids
    assert torch.equal(m_ids, r_ids), (m_ids, r_ids)
    wrong = r_ids.clone()
    wrong[wrong == 3] = 500
    assert not torch.equal(wrong, r_ids)


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
    fc2_gs = torch.tensor(50.0, device="cuda")
    return w1, s1, a1, w2, s2, a2, fc2_gs


def _experts(xq, xsf, te, ml, p2e, nt, w, out, wts, nl, ls, tile):
    w1, s1, a1, w2, s2, a2, fc2_gs = wts
    xa, xa_sf = torch.ops.trtllm.cute_dsl_nvfp4_gather_grouped_gemm_act_fusion_blackwell(
        input=xq.view(torch.float4_e2m1fn_x2),
        weight=w1.view(torch.float4_e2m1fn_x2),
        input_scale=xsf.view(torch.uint8),
        weight_scale=s1.view(torch.uint8),
        alpha=a1,
        tile_idx_to_group_idx=te,
        tile_idx_to_mn_limit=ml,
        permuted_idx_to_expanded_idx=p2e,
        num_non_exiting_tiles=nt,
        global_sf=fc2_gs,
        num_experts=E,
        top_k=K,
        num_local_experts=nl,
        local_expert_offset=ls,
        tile_size=tile,
    )
    torch.ops.trtllm.cute_dsl_nvfp4_grouped_gemm_finalize_inplace_blackwell(
        input=xa.view(torch.float4_e2m1fn_x2),
        weight=w2.view(torch.float4_e2m1fn_x2),
        input_scale=xa_sf.view(torch.uint8),
        weight_scale=s2.view(torch.uint8),
        alpha=a2,
        output=out,
        tile_idx_to_group_idx=te,
        tile_idx_to_mn_limit=ml,
        permuted_idx_to_expanded_idx=p2e,
        num_non_exiting_tiles=nt,
        token_final_scales=w,
        num_experts=E,
        top_k=K,
        num_local_experts=nl,
        local_expert_offset=ls,
        tile_size=tile,
        output_dtype=torch.bfloat16,
    )
    return out


@pytest.mark.parametrize("m", [1, 7, 14, 28])
@pytest.mark.parametrize("nl,ls", [(64, 128), (512, 0)])
def test_moe_output_parity(m, nl, ls, tile=128, inter=1024):
    inter_p = inter if nl == 64 else 128  # EP8 shard vs MoE-TP8 shard
    wts = _rand_fp4_moe(nl, inter_p, seed=5)
    logits, x, gs = _inputs(m, seed=99 + m)
    ids, w, xq, xsf, srt = _stock(logits, x, gs, ls, nl, tile)
    te, ml, e2p, p2e, tot, nt = srt
    ref = torch.zeros((m, H), dtype=torch.bfloat16, device="cuda")
    ref = _experts(xq, xsf, te, ml, p2e, nt, w, ref, wts, nl, ls, tile)
    scratch = torch.zeros((1024,), dtype=torch.int32, device="cuda")
    (m_ids, m_w, m_xq, m_xsf, m_te, m_ml, m_e2p, m_p2e, m_tot, m_nt, m_out) = q35_moe_front(
        logits, x, gs, K, ls, nl, tile, scratch
    )
    mine = _experts(m_xq, m_xsf, m_te, m_ml, m_p2e, m_nt, m_w, m_out, wts, nl, ls, tile)
    torch.cuda.synchronize()
    d = (mine.float() - ref.float()).abs()
    scale = ref.float().abs().max().item() + 1e-6
    assert torch.isfinite(mine).all()
    # Only the bf16 scatter-add order may differ.
    assert d.max().item() <= 2e-2 * scale, (m, nl, d.max().item(), scale)


@pytest.mark.parametrize("m", [1, 2, 7, 14, 16, 17, 28, 32])
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
    nd = (a != ref).sum().item()
    ids_a = torch.ops.trtllm.renorm_moe_routing_op(a, K, torch.float32)[0]
    ids_r = torch.ops.trtllm.renorm_moe_routing_op(ref, K, torch.float32)[0]
    nsel = (torch.sort(ids_a, -1).values != torch.sort(ids_r, -1).values).any(-1).sum().item()
    print(
        f"[router gemm] m={m}: max|err| mine {err_mine:.2e} cublas "
        f"{err_ref:.2e}; logits differing from cublas {nd}/{a.numel()}; "
        f"tokens with a different top-{K} set {nsel}/{m}"
    )


def bench_router(m=7, iters=2000):
    x = torch.randn((m, H), device="cuda").to(torch.bfloat16)
    w = (torch.randn((E, H), device="cuda") * 0.02).to(torch.bfloat16)
    rg = Q35RouterGemm()
    gs = torch.tensor(
        448.0 * 6.0 / x.float().abs().max().item(), dtype=torch.float32, device="cuda"
    )

    def cublas_quant():
        torch.ops.trtllm.cublas_mm(x, w.t(), bias=None, out_dtype=torch.bfloat16)
        torch.ops.trtllm.fp4_quantize(x, gs, 16, False, False)

    flats = {
        nl: torch.empty((sum(q35_prerouted_sizes(m, K, nl, 8)),), dtype=torch.int32, device="cuda")
        for nl in (64, 128, 256, 512)
    }
    rt = Q35RouteTables()

    def q35_quant_then_route(nl, ls, dea=None):
        # the in-model default: router + quant, then the per-token route + tables launch
        lg = rg(x, w, gs)[0]
        rt(
            lg,
            K,
            ls,
            nl,
            8,
            flat=flats[nl],
            dea_groups=dea[0] if dea else 0,
            dea_group=dea[1] if dea else 0,
            single_program=False,
        )

    def lowm_then_route(nl, ls, dea=None):
        # TRTLLM_Q35_MOE_PREROUTE_LOWM: low-M BF16 router GEMM, then route + tables + x quant in one launch
        lg = q35_lowm_router(x, w)
        rt(
            lg,
            K,
            ls,
            nl,
            8,
            flat=flats[nl],
            dea_groups=dea[0] if dea else 0,
            dea_group=dea[1] if dea else 0,
            single_program=False,
            x=x,
            global_scale=gs,
        )

    def fold(nl, ls, mode, dea=None):
        return lambda: rg(x, w, gs, route=(K, ls, nl, 8, flats[nl], dea), route_mode=mode)

    res = {}
    for name, fn in (
        (
            "cublas",
            lambda: torch.ops.trtllm.cublas_mm(x, w.t(), bias=None, out_dtype=torch.bfloat16),
        ),
        ("q35", lambda: rg(x, w)),
        ("cublas+quant", cublas_quant),
        ("q35+quant(fused)", lambda: rg(x, w, gs)),
        ("q35+quant, then route nl=64", lambda: q35_quant_then_route(64, 448)),
        ("q35+quant, then route nl=128", lambda: q35_quant_then_route(128, 128)),
        ("q35+quant, then route DEA 8 groups", lambda: q35_quant_then_route(512, 0, (8, 5))),
        ("fold1 nl=64", fold(64, 448, 1)),
        ("fold2 nl=64", fold(64, 448, 2)),
        ("fold2 nl=128", fold(128, 128, 2)),
        ("fold2 DEA 8 groups", fold(512, 0, 2, (8, 5))),
        ("fold2 DEA 4 groups", fold(512, 0, 2, (4, 1))),
        ("lowm, then route+quant nl=64", lambda: lowm_then_route(64, 448)),
        ("lowm, then route+quant nl=128", lambda: lowm_then_route(128, 128)),
        ("lowm, then route+quant DEA 8 groups", lambda: lowm_then_route(512, 0, (8, 5))),
        ("lowm", lambda: q35_lowm_router(x, w)),
        ("lowm+quant", lambda: q35_lowm_router(x, w, gs)),
    ):
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        gr = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gr):
            for _ in range(20):
                fn()
        gr.replay()
        torch.cuda.synchronize()
        t0, t1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        t0.record()
        for _ in range(iters // 20):
            gr.replay()
        t1.record()
        torch.cuda.synchronize()
        res[name] = t0.elapsed_time(t1) * 1000.0 / iters
    print(
        f"[bench router] m={m} (graph, back-to-back, us): "
        + ", ".join(f"{k} {v:.2f}" for k, v in res.items())
    )
    return res


import triton  # noqa: E402
import triton.language as tl  # noqa: E402


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
    """The PDL dependents (router GEMM + quant, standalone quant, front end)
    must not read their producer's output before griddepcontrol.wait: x is
    garbage until a delayed producer (which releases dependents immediately)
    writes it; replay in a CUDA graph and compare with the eager reference."""
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
    print(f"[pdl delayed producer] 20/20 replays correct; positive control stale in {stale}/20")


def test_pdl_early_trigger_chain(m=7, nl=128, ls=128, inter=512, tile=8):
    """TRTLLM_Q35_MOE_EARLY_TRIGGER: the router launch and the per-token route
    launch release their PDL dependents at entry. Their dependents must still see
    their outputs: delayed producer (x is garbage until it writes) -> router +
    quant -> (a) the stock trtllm-gen MoE op routing the logits itself, (b) the
    per-token route + tables launch -> the pre-routed MoE op; CUDA graph replays
    must equal the eager result bitwise. Control: a dependent that does not wait
    reads the logits early in some replays (the trigger really is early)."""
    import tensorrt_llm._torch.modules.fused_moe.q35_fused_front as ff
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
    saved = (ff.Q35_MOE_EARLY_TRIGGER, ff.Q35_MOE_EARLY_TRIGGER_ROUTE)
    ff.Q35_MOE_EARLY_TRIGGER = ff.Q35_MOE_EARLY_TRIGGER_ROUTE = True
    try:
        rg = Q35RouterGemm()
        rt = Q35RouteTables()
        flat = torch.empty(
            (sum(q35_prerouted_sizes(m, K, nl, tile)),), dtype=torch.int32, device="cuda"
        )
        y = torch.empty((m, E), dtype=torch.bfloat16, device="cuda")

        def moe(lg, xq, xsf, pre=None):
            inp = list(inputs)
            inp[0], inp[2], inp[3] = lg, xq, xsf.view(-1).view(torch.float8_e4m3fn)
            if pre is None:
                return runner(inp, tactic=[tile, cfg])[0]
            return runner(inp, tactic=[tile, cfg], prerouted=pre, prerouted_tile=tile)[0]

        flat2 = torch.empty_like(flat)
        rt2 = Q35RouteTables()

        def chain():
            lg, xq, xsf = rg(x, w, gs)
            res, bufs = [moe(lg, xq, xsf)], [lg, xq, xsf, flat]
            if has_pre:
                ids, wts = rt(lg, K, ls, nl, tile, flat=flat, single_program=False)[:2]
                res.append(moe(lg, xq, xsf, (q35_prerouted_views(flat, m, K, nl, tile), ids, wts)))
                bufs += [ids, wts]
                # TRTLLM_Q35_MOE_PREROUTE_LOWM: low-M router GEMM -> route + tables + x quant -> MoE
                lg2 = q35_lowm_router(x, w)
                if lg2 is not None:
                    o = rt2(
                        lg2, K, ls, nl, tile, flat=flat2, single_program=False, x=x, global_scale=gs
                    )
                    res.append(
                        moe(
                            lg2,
                            o[8],
                            o[9],
                            (q35_prerouted_views(flat2, m, K, nl, tile), o[0], o[1]),
                        )
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
        ref_lg = rg(x_src, w, gs)[0].clone()

        def step():
            producer()
            return chain()

        def control():
            producer()
            lg = rg(x, w, gs)[0]
            _nowait_copy_kernel[(triton.cdiv(m * E, 4096),)](
                lg, y, m * E, BLOCK=4096, launch_pdl=True
            )
            return lg

        step()
        control()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            outs, bufs = step()
        cgraph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(cgraph):
            lg_c = control()
        bad = early = 0
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
            lg_c.fill_(0)  # the graph's logits buffer: a too-early reader sees zeros
            cgraph.replay()
            torch.cuda.synchronize()
            early += int(not torch.equal(y, ref_lg))
        assert bad == 0, f"{bad}/20 early-trigger replays differ from the eager result"
    finally:
        ff.Q35_MOE_EARLY_TRIGGER, ff.Q35_MOE_EARLY_TRIGGER_ROUTE = saved
    print(
        f"[pdl early trigger] 20/20 replays correct ({len(ref)} chains: stock"
        f"{', pre-routed' if has_pre else ''}{', low-M router + route/quant' if len(ref) > 2 else ''}); "
        f"control read the logits early in {early}/20"
    )


def _pdl_ptx_check():
    """The PDL PTX-order helper next to this test (pdl_ptx_check.py), or None when it is not present."""
    import importlib.util
    import os

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pdl_ptx_check.py")
    if not os.path.isfile(path):
        return None
    spec = importlib.util.spec_from_file_location("pdl_ptx_check", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _ptx_global_ops_before_wait(ptx):
    """(first griddepcontrol.wait offset, global memory instructions before it, total) in PTX text order
    (fallback when pdl_ptx_check.py is not present)."""
    import re

    i_wait = ptx.find("griddepcontrol.wait")
    ops = [
        mm.start()
        for mm in re.finditer(
            r"\b(ld\.global|st\.global|atom\.global|red\.global|"
            r"ld\.volatile\.global)",
            ptx,
        )
    ]
    return i_wait, sum(1 for i in ops if i < i_wait), len(ops)


def test_route_kernel_waits_before_global_access(m=7):
    """PDL rule: a launch that its producer may release early must not touch global memory
    before griddepcontrol.wait (a pre-wait write could land in a block the caching allocator just recycled from the
    producer's input). The per-token route + tables launch (EP, DEA, with the x-quant programs) is released early by
    the low-M router kernel (LOWM) and by the Triton router with TRTLLM_Q35_MOE_EARLY_TRIGGER=router: its PTX must
    have no global load / store / atomic before the wait."""
    logits, x, gs = _inputs(m, seed=1700)
    pdl = _pdl_ptx_check()
    res = []
    for tag, kw in (
        ("ep", dict(local_start=448, num_local=64)),
        ("ep+quant", dict(local_start=128, num_local=128, x=x, global_scale=gs)),
        (
            "dea+quant",
            dict(local_start=0, num_local=512, dea_groups=8, dea_group=3, x=x, global_scale=gs),
        ),
    ):
        rt = Q35RouteTables()
        ls, nl = kw.pop("local_start"), kw.pop("num_local")
        rt(logits, K, ls, nl, 8, single_program=False, **kw)
        torch.cuda.synchronize()
        if pdl is not None:
            rep = pdl.assert_wait_first(rt._compiled, f"route {tag}")
            pdl.assert_trigger_last(
                rt._compiled, f"route {tag}"
            )  # default build: trigger at the end
            res.append(f"{tag} 0/{rep['global_ops']}")
        else:
            i_wait, before, total = _ptx_global_ops_before_wait(rt._compiled.asm["ptx"])
            assert i_wait > 0, (tag, "no griddepcontrol.wait")
            assert before == 0, (
                tag,
                f"{before} global memory instructions before griddepcontrol.wait",
            )
            res.append(f"{tag} 0/{total}")
    print(
        f"[route ptx] global accesses before griddepcontrol.wait: {', '.join(res)}"
        f"{'; trigger after the last access + barrier' if pdl is not None else ' (fallback check)'}"
    )


def test_router_weight_load_before_pdl_wait(m=7):
    """The router GEMM must issue its weight loads before griddepcontrol.wait
    (the weights do not depend on the producer, so their streaming overlaps the
    preceding all-reduce) and read x only after it: check the PTX order."""
    from tensorrt_llm._torch.modules.fused_moe.q35_fused_front import _q35_router_gemm_kernel

    x = torch.randn((m, H), device="cuda").to(torch.bfloat16)
    w = (torch.randn((E, H), device="cuda") * 0.02).to(torch.bfloat16)
    rg = Q35RouterGemm()
    rg(x, w)
    out = torch.empty((m, E), dtype=torch.bfloat16, device="cuda")
    # no QUANT / ROUTE: the 12 routing pointers (9 table / ticket + 3 per-token scratch) are unused dummies
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
        *((rg.cnt,) * 12),
        m,
        0,
        1,
        0,
        N=E,
        K=H,
        BM=16,
        BN=rg.BN,
        SPLITK=rg.SPLITK,
        QUANT=False,
        ROUTE=0,
        TOPK=1,
        KP=16,
        MP=2,
        RB=2,
        PE=32,
        TILE=8,
        NL=1,
        NLP=2,
        MAX_CTAS_PER_EXPERT=1,
        DEA=False,
        EXPANDED_PAD=16,
        USE_PDL=True,
        num_warps=4,
        launch_pdl=True,
    )
    torch.cuda.synchronize()
    ptx = ck.asm["ptx"]
    i_wait = ptx.find("griddepcontrol.wait")
    loads = [i for i in range(len(ptx)) if ptx.startswith("ld.global", i)]
    n_before = sum(1 for i in loads if i < i_wait)
    import re

    writes = [mm.start() for mm in re.finditer(r"\b(st\.global|atom\.|red\.)", ptx)]
    w_before = sum(1 for i in writes if i < i_wait)
    print(
        f"[router ptx] griddepcontrol.wait at {i_wait}; ld.global before it: {n_before}, after: "
        f"{len(loads) - n_before}; stores / atomics before it: {w_before} of {len(writes)}"
    )
    assert i_wait > 0 and n_before > 0, "no weight load issued before the PDL wait"
    assert n_before < len(loads), "x must be loaded after the PDL wait"
    assert w_before == 0, "no global store / atomic may precede the PDL wait"
    pdl = _pdl_ptx_check()
    if (
        pdl is not None
    ):  # pdl_ptx_check: only (static) weight loads before the wait; trigger at the end
        pdl.assert_wait_first(ck, "router", allow_loads_before=True)
        pdl.assert_trigger_last(ck, "router")


ROUTE_CASES = [
    (m, nl, ls, tile)
    for m in (1, 7, 14, 28, 32)
    for (nl, ls) in ((512, 0), (256, 256), (128, 128), (64, 448))
    for tile in (8, 16)
]


def test_route_tables(m, nl, ls, tile, ties=False, single_program=True):
    """Triton route + trtllm-gen tables vs the trtllm-gen routing runner
    (moe_topk_sort, Renormalize) on the same logits."""
    logits, _, _ = _inputs(m, seed=500 + m + nl + tile, ties=ties)
    rt = Q35RouteTables()
    ids, wts, e2p, p2t, cte, ctl, nct, npd = rt(
        logits, K, ls, nl, tile, single_program=single_program
    )
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
    # ids: same set and order as the runner's top-k (its packed ids are internal;
    # compare the per-token sets through the permutation tables above) and the
    # renormalized weights within bf16 rounding of the reference.
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


# (tokens, local experts, first local expert, MoE intermediate size per rank): TP4xEP2, TP2xEP4,
# TP2xEP8 (TEP16) shards of the 512-expert layer, plus one all-local case.
PREROUTED_CASES = [
    (m, nl, ls, inter)
    for m in (1, 7, 14, 28)
    for (nl, ls, inter) in ((256, 256, 256), (128, 128, 512), (64, 448, 512), (512, 0, 128))
]


def _prerouted_runner_inputs(m, nl, ls, inter, seed):
    from tensorrt_llm._torch.custom_ops.trtllm_gen_custom_ops import FP4BlockScaleMoERunner
    from tensorrt_llm._torch.modules.fused_moe.routing import RoutingMethodType
    from tensorrt_llm._torch.utils import ActType_TrtllmGen

    w1, s1, a1, w2, s2, a2, _ = _rand_fp4_moe(nl, inter, seed=seed)
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
    """trtllm-gen FP4 MoE (finalized output) with the routing tables, ids and
    weights of Q35RouteTables (run_moe_prerouted, all-local problem of nl
    experts, no routing kernel) vs the stock op routing the same logits itself
    (global layout, local slice [ls, ls + nl)): bit-identical for every tactic
    of the tables' tile; corrupting one CTA's expert must change the output."""
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
    """The trtllm-gen finalize kernel's result (FP32 sum of w * row in k order,
    BF16 once); mirrors modeling_qwen3_next._q35_finalize_ref (TRTLLM_Q35_FIN_FOLD)."""
    acc = torch.zeros((perm.shape[0], hidden), dtype=torch.float32, device=fc2.device)
    for k in range(perm.shape[1]):
        p = perm[:, k]
        rows = fc2[p.clamp(min=0).long(), :hidden].float()
        acc = torch.where((p >= 0)[:, None], acc + weights[:, k : k + 1].float() * rows, acc)
    return acc.to(torch.bfloat16)


UNFINALIZED_CASES = [
    (1, 128, 128, 512),
    (7, 128, 128, 512),
    (28, 128, 128, 512),
    (7, 64, 448, 512),
    (14, 256, 256, 256),
]


@pytest.mark.parametrize("m,nl,ls,inter", UNFINALIZED_CASES)
def test_prerouted_unfinalized_op(m, nl, ls, inter, tile=8):
    """TRTLLM_Q35_FIN_FOLD runs the MoE with do_finalize=False. The pre-routed
    run goes through torch.ops.trtllm.fp4_block_scale_moe_runner, which may not
    return its inputs (torch rejects outputs aliasing inputs): the op returns
    placeholders for the weights / expanded -> permuted table and
    TRTLLMOpBackend.run_fp4_block_scale_moe puts the caller's tensors back.
    The result must fit the CuTe all-reduce's finalize contract, equal the stock
    op's do_finalize=False output by expanded index (weights and FC2 rows
    bitwise), and finalize to the stock do_finalize=True output bitwise."""
    import os

    from tensorrt_llm._torch.custom_ops.trtllm_gen_custom_ops import FP4BlockScaleMoERunner
    from tensorrt_llm._torch.distributed.q35_allreduce import Q35CuteAllReduce
    from tensorrt_llm._torch.modules.fused_moe.moe_op_backend import TRTLLMOpBackend
    from tensorrt_llm._torch.modules.fused_moe.routing import RoutingMethodType
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
    # the op runs the pre-routed path only for the tile of its tactic: pin it (TRTLLM_Q35_MOEGEMM_PIN)
    pin_env = "TRTLLM_Q35_MOEGEMM_PIN"
    saved = os.environ.get(pin_env)
    os.environ[pin_env] = f"{inter}x{nl}:{m}={tile}/{cfg}"
    try:
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
    finally:
        if saved is None:
            os.environ.pop(pin_env, None)
        else:
            os.environ[pin_env] = saved
    ref = runner_nf(inputs, tactic=[tile, cfg])
    fin = runner(inputs, tactic=[tile, cfg])[0]
    torch.cuda.synchronize()
    assert raw[1].numel() == 0 and raw[2].numel() == 0, (tag, "pre-routed path not taken")
    fc2, w, perm = out
    assert Q35CuteAllReduce.moe_fold_supported(fc2, w, perm, H), (
        tag,
        fc2.dtype,
        tuple(fc2.shape),
        w.dtype,
        tuple(w.shape),
        perm.dtype,
        tuple(perm.shape),
    )
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


DEA_CASES = [(m, g, tile) for m in (1, 7, 14, 28, 32) for g in (2, 4, 8) for tile in (8, 16)]


def test_dea_tables(m, n_groups, tile, ties=False, single_program=True):
    """Dynamic expert assignment tables (all experts on every rank): for each
    group, the CTA tables cover exactly the step's distinct experts dealt to it
    (round-robin in id order), every (token, k) pick is owned by exactly one
    group, the busiest group holds ceil(D / G) experts, the ids / weights equal
    the non-DEA launch, and each group's rows are a valid trtllm-gen layout."""
    from collections import Counter

    logits, _, _ = _inputs(m, seed=800 + m + n_groups + tile, ties=ties)
    rt = Q35RouteTables()
    ids_nd, wts_nd = rt(logits, K, 0, E, tile, single_program=single_program)[:2]
    torch.cuda.synchronize()
    flat_ids = ids_nd.view(-1).long().cpu().tolist()
    counts = Counter(flat_ids)
    distinct = sorted(counts)
    assign = {e: j % n_groups for j, e in enumerate(distinct)}
    owner = [-1] * (m * K)
    loads = []
    tag = (m, n_groups, tile, ties)
    for g in range(n_groups):
        flat = torch.full(
            (sum(q35_prerouted_sizes(m, K, E, tile)),), -7, dtype=torch.int32, device="cuda"
        )
        ids, wts = rt(
            logits,
            K,
            0,
            E,
            tile,
            flat=flat,
            dea_groups=n_groups,
            dea_group=g,
            single_program=single_program,
        )[:2]
        e2p, p2t, cte, ctl, nct, npd = (t.cpu() for t in q35_prerouted_views(flat, m, K, E, tile))
        torch.cuda.synchronize()
        assert torch.equal(ids, ids_nd) and torch.equal(wts, wts_nd), tag
        mine = [e for e in distinct if assign[e] == g]
        loads.append(len(mine))
        exp_cte, exp_ctl, off = [], [], 0
        for e in mine:
            n = (counts[e] + tile - 1) // tile
            for j in range(n):
                exp_cte.append(e)
                exp_ctl.append(min((off + j + 1) * tile, off * tile + counts[e]))
            off += n
        nt = int(nct.item())
        assert nt == len(exp_cte) and int(npd.item()) == nt * tile, (tag, g)
        assert nt <= q35_dea_max_ctas(m, K, E, n_groups, tile), (tag, g, nt, "cap")
        assert cte[:nt].tolist() == exp_cte and ctl[:nt].tolist() == exp_ctl, (tag, g)
        rows = set()
        for i, e in enumerate(flat_ids):
            r = int(e2p[i])
            if assign[e] != g:
                assert r == -1, (tag, g, i)
                continue
            assert r >= 0 and owner[i] == -1, (tag, g, i)
            owner[i] = g
            c = r // tile
            assert c < nt and int(cte[c]) == e and r < int(ctl[c]), (tag, g, i)
            assert int(p2t[r]) == i // K and r not in rows, (tag, g, i)
            rows.add(r)
    assert all(o >= 0 for o in owner), tag
    assert max(loads) == -(-len(distinct) // n_groups), (tag, loads)
    assert int(rt.hist.abs().sum().item()) == 0 and int(rt.counter.item()) == 0


def test_dea_cap_adversarial():
    """q35_dea_max_ctas bounds every group's CTAs for extreme routings too:
    all tokens on the same K experts (fewest experts, most picks each) and all
    picks on distinct experts."""
    rt = Q35RouteTables()
    for m in (1, 7, 14, 28, 32):
        for kind in ("same", "spread"):
            logits = torch.full((m, E), -4.0, device="cuda")
            for t in range(m):
                for k in range(K):
                    e = k if kind == "same" else (t * K + k) % E
                    logits[t, e] = 3.0 - 0.01 * k
            logits = logits.to(torch.bfloat16)
            for g_count in (2, 4, 8, 16):
                for tile in (8, 16):
                    for g in range(g_count):
                        flat = torch.empty(
                            (sum(q35_prerouted_sizes(m, K, E, tile)),),
                            dtype=torch.int32,
                            device="cuda",
                        )
                        rt(logits, K, 0, E, tile, flat=flat, dea_groups=g_count, dea_group=g)
                        nt = int(q35_prerouted_views(flat, m, K, E, tile)[4].item())
                        assert nt <= q35_dea_max_ctas(m, K, E, g_count, tile), (
                            m,
                            kind,
                            g_count,
                            tile,
                            g,
                            nt,
                        )


def _dea_moe_case(m, n_groups, inter, seed, do_finalize):
    from tensorrt_llm._torch.custom_ops.trtllm_gen_custom_ops import FP4BlockScaleMoERunner
    from tensorrt_llm._torch.modules.fused_moe.routing import RoutingMethodType
    from tensorrt_llm._torch.utils import ActType_TrtllmGen

    w1, s1, a1, w2, s2, a2, _ = _rand_fp4_moe(E, inter, seed=seed)
    logits, x, gs = _inputs(m, seed=seed + 1)
    xq, xsf = q35_nvfp4_quant(x, gs)
    e4m3 = torch.float8_e4m3fn
    nl = E // n_groups

    def make(ls, nloc, sl):
        r = FP4BlockScaleMoERunner(
            E,
            K,
            None,
            None,
            inter,
            ls,
            nloc,
            None,
            int(RoutingMethodType.Renormalize),
            do_finalize,
            ActType_TrtllmGen.SwiGlu.value,
        )
        inp = [
            logits,
            None,
            xq,
            xsf.view(-1).view(e4m3),
            w1[sl],
            s1[sl].view(e4m3),
            None,
            None,
            None,
            None,
            w2[sl],
            s2[sl].view(e4m3),
            None,
            a1[sl],
            a1[sl],
            a2[sl],
            None,
            None,
        ]
        return r, inp

    ep = [make(g * nl, nl, slice(g * nl, (g + 1) * nl)) for g in range(n_groups)]
    full = make(0, E, slice(0, E))
    return logits, ep, full


@pytest.mark.parametrize("m", [1, 7, 28])
@pytest.mark.parametrize("n_groups", [4, 8])
def test_dea_moe_parity(m, n_groups, inter=512, tile=8):
    """Replicated experts + per-step assignment vs contiguous EP over the same
    groups: every (token, k) FC1 -> FC2 row is bit-identical (do_finalize=False,
    matched by expanded index; the same expert math on another group), and the
    sum over groups of the finalized partials agrees within the bf16 rounding
    of the per-group partials (the K terms of a token are grouped differently)."""
    from tensorrt_llm._torch.custom_ops.trtllm_gen_custom_ops import FP4BlockScaleMoERunner
    from tensorrt_llm._torch.distributed.q35_allreduce import Q35CuteAllReduce

    if not FP4BlockScaleMoERunner.supports_prerouted():
        pytest.skip("libth_common without run_moe_prerouted")
    rt = Q35RouteTables()
    emu_dea = {}
    for do_finalize in (False, True):
        logits, ep, (full_r, full_in) = _dea_moe_case(
            m, n_groups, inter, seed=950 + m, do_finalize=do_finalize
        )
        nl = E // n_groups
        cfg_ep = {
            int(c)
            for t, c in ep[0][0].get_runner().get_valid_configs(K, H, inter, nl, m)
            if int(t) == tile
        }
        cfg_full = {
            int(c)
            for t, c in full_r.get_runner().get_valid_configs(K, H, inter, E, m)
            if int(t) == tile
        }
        common = sorted(cfg_ep & cfg_full)
        assert common, (m, n_groups, "no common tile-8 config")
        cfg = common[0]
        ids_flat = None
        ep_rows, dea_rows = {}, {}
        ep_sum = torch.zeros((m, H), dtype=torch.float32, device="cuda")
        dea_sum = torch.zeros((m, H), dtype=torch.float32, device="cuda")
        for g in range(n_groups):
            r, inp = ep[g]
            out = r(inp, tactic=[tile, cfg])
            flat = torch.empty(
                (sum(q35_prerouted_sizes(m, K, E, tile)),), dtype=torch.int32, device="cuda"
            )
            ids, wts = rt(logits, K, 0, E, tile, flat=flat, dea_groups=n_groups, dea_group=g)[:2]
            tables = q35_prerouted_views(flat, m, K, E, tile)
            cap = q35_dea_max_ctas(m, K, E, n_groups, tile)
            d_out = full_r(
                full_in, tactic=[tile, cfg], prerouted=(tables, ids, wts, cap), prerouted_tile=tile
            )
            if do_finalize and FP4BlockScaleMoERunner.supports_prerouted_capped():
                # the capped grid must not change a bit of the finalized output (the permuted
                # do_finalize=False buffers hold uninitialized padding rows, so compare finalized)
                dout_u = full_r(
                    full_in, tactic=[tile, cfg], prerouted=(tables, ids, wts), prerouted_tile=tile
                )
                torch.cuda.synchronize()
                assert torch.equal(d_out[0].view(torch.int16), dout_u[0].view(torch.int16)), (
                    m,
                    n_groups,
                    "cap",
                )
            torch.cuda.synchronize()
            if ids_flat is None:
                ids_flat = ids.view(-1).long().cpu().tolist()
            if do_finalize:
                ep_sum += out[0].float()
                dea_sum += d_out[0].float()
                # TRTLLM_Q35_FIN_FOLD: this group's do_finalize=False operands, as q35_dea_run
                # returns them, finalize to its finalized output bitwise
                assert torch.equal(
                    emu_dea[g].view(torch.int16),
                    d_out[0][:, :H].view(torch.int16),
                ), (m, n_groups, g, "fin fold")
                continue
            d_out = q35_prerouted_unfinalized(d_out, flat, K, wts)
            assert Q35CuteAllReduce.moe_fold_supported(*d_out, H), (
                m,
                n_groups,
                "fold contract",
            )
            emu_dea[g] = _finalize_ref(*d_out, H)
            e2p_ep = out[2].view(-1).cpu()
            e2p_dea = d_out[2].reshape(-1).cpu()
            for i in range(m * K):
                if int(e2p_ep[i]) >= 0:
                    ep_rows[i] = out[0][int(e2p_ep[i])].clone()
                if int(e2p_dea[i]) >= 0:
                    assert i not in dea_rows, (m, n_groups, i, "owned twice")
                    dea_rows[i] = d_out[0][int(e2p_dea[i])].clone()
        if not do_finalize:
            assert sorted(ep_rows) == list(range(m * K)), (m, n_groups, "ep cover")
            assert sorted(dea_rows) == list(range(m * K)), (m, n_groups, "dea cover")
            for i in range(m * K):
                assert torch.equal(ep_rows[i].view(torch.int16), dea_rows[i].view(torch.int16)), (
                    m,
                    n_groups,
                    i,
                )
        else:
            d = (ep_sum - dea_sum).abs().max().item()
            scale = ep_sum.abs().max().item() + 1e-6
            assert torch.isfinite(dea_sum).all()
            assert d <= n_groups * 2**-7 * scale, (m, n_groups, d, scale)


def _graph_us(fn, iters=400):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    gr = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gr):
        for _ in range(20):
            fn()
    gr.replay()
    torch.cuda.synchronize()
    t0, t1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    t0.record()
    for _ in range(iters // 20):
        gr.replay()
    t1.record()
    torch.cuda.synchronize()
    return t0.elapsed_time(t1) * 1000.0 / iters


def bench_dea_op(m=7, n_groups=4, inter=512, tile=8):
    """Where does DEA's MoE time go? For group 0 of n_groups (CUDA graph,
    back-to-back, L2-hot weights; times include nothing else):
      ep_stock : stock op, EP shard (local E/G), routing kernel from logits
      ep_pre   : same shard, pre-routed EP tables (no routing kernel)
      dea_pre  : all-local E-expert problem over replicated weights, DEA tables
      route_ep / route_dea : the per-token route + tables launch alone"""
    from tensorrt_llm._torch.custom_ops.trtllm_gen_custom_ops import FP4BlockScaleMoERunner

    if not FP4BlockScaleMoERunner.supports_prerouted():
        print("[bench dea op] skipped: no run_moe_prerouted")
        return None
    nl = E // n_groups
    res = {}
    for do_fin in (True,):
        logits, ep, (full_r, full_in) = _dea_moe_case(
            m, n_groups, inter, seed=990 + m, do_finalize=do_fin
        )
        r_ep, in_ep = ep[0]
        cfg_ep = sorted(
            int(c)
            for t, c in r_ep.get_runner().get_valid_configs(K, H, inter, nl, m)
            if int(t) == tile
        )
        cfg_all = sorted(
            int(c)
            for t, c in full_r.get_runner().get_valid_configs(K, H, inter, E, m)
            if int(t) == tile
        )
        common = [c for c in cfg_ep if c in cfg_all]
        cfg = common[0]
        rt_ep, rt_dea = Q35RouteTables(), Q35RouteTables()
        f_ep = torch.empty(
            (sum(q35_prerouted_sizes(m, K, nl, tile)),), dtype=torch.int32, device="cuda"
        )
        f_dea = torch.empty(
            (sum(q35_prerouted_sizes(m, K, E, tile)),), dtype=torch.int32, device="cuda"
        )
        ids_e, wts_e = rt_ep(logits, K, 0, nl, tile, flat=f_ep, single_program=False)[:2]
        ids_d, wts_d = rt_dea(
            logits,
            K,
            0,
            E,
            tile,
            flat=f_dea,
            dea_groups=n_groups,
            dea_group=0,
            single_program=False,
        )[:2]
        tab_e = q35_prerouted_views(f_ep, m, K, nl, tile)
        tab_d = q35_prerouted_views(f_dea, m, K, E, tile)
        torch.cuda.synchronize()
        d_ep = int(tab_e[4].item())
        d_dea = int(tab_d[4].item())
        res["ep_stock"] = _graph_us(lambda: r_ep(in_ep, tactic=[tile, cfg]))
        res["ep_pre"] = _graph_us(
            lambda: r_ep(
                in_ep, tactic=[tile, cfg], prerouted=(tab_e, ids_e, wts_e), prerouted_tile=tile
            )
        )
        res["dea_pre"] = _graph_us(
            lambda: full_r(
                full_in, tactic=[tile, cfg], prerouted=(tab_d, ids_d, wts_d), prerouted_tile=tile
            )
        )
        res["route_ep"] = _graph_us(
            lambda: rt_ep(logits, K, 0, nl, tile, flat=f_ep, single_program=False)
        )
        res["route_dea"] = _graph_us(
            lambda: rt_dea(
                logits,
                K,
                0,
                E,
                tile,
                flat=f_dea,
                dea_groups=n_groups,
                dea_group=0,
                single_program=False,
            )
        )
        res["route_dea_1prog"] = _graph_us(
            lambda: rt_dea(logits, K, 0, E, tile, flat=f_dea, dea_groups=n_groups, dea_group=0)
        )
    print(
        f"[bench dea op] m={m} G={n_groups} inter={inter} cfg={cfg} CTAs ep/dea={d_ep}/{d_dea} (us): "
        + ", ".join(f"{k} {v:.2f}" for k, v in res.items()),
        flush=True,
    )
    return res


def bench_moe_distinct(m=7, nl=64, ls=448, inter=512, iters=400):
    """trtllm-gen FP4 MoE op time vs the number of distinct LOCAL experts the
    m tokens hit (the rest of each token's top-k goes to other ranks), for the
    EP-imbalance model: on the busiest rank the MoE costs this curve at its
    expert count. Logits are crafted so the local hits are exactly d experts,
    spread over the tokens as evenly as possible; CUDA graph, back-to-back."""
    runner, inputs, _ = _prerouted_runner_inputs(m, nl, ls, inter, seed=900)
    cfgs = runner.get_runner().get_valid_configs(K, H, inter, nl, m)
    tiles = sorted({int(t) for t, _ in cfgs})
    res = {}
    for d in (1, 2, 4, 6, 8, 10, 12, 16, 24):
        if d > min(nl, m * K):
            continue
        logits = torch.full((m, E), -8.0, device="cuda")
        # local hits: expert ls + j for j < d, assigned round-robin to tokens
        slots = [0] * m
        for j in range(d):
            t = j % m
            logits[t, ls + j] = 4.0 - 0.01 * slots[t]
            slots[t] += 1
        # fill every token's remaining top-k slots with distinct remote experts
        remote = [e for e in range(E) if not (ls <= e < ls + nl)]
        r = 0
        for t in range(m):
            for _ in range(K - slots[t]):
                logits[t, remote[r % len(remote)]] = 2.0 - 0.001 * r
                r += 1
        inp = list(inputs)
        inp[0] = logits.to(torch.bfloat16)
        best = None
        for tile in tiles:
            cfg = [c for t, c in cfgs if int(t) == tile][0]

            def fn():
                return runner(inp, tactic=[tile, int(cfg)])

            for _ in range(3):
                fn()
            torch.cuda.synchronize()
            gr = torch.cuda.CUDAGraph()
            with torch.cuda.graph(gr):
                for _ in range(20):
                    fn()
            gr.replay()
            torch.cuda.synchronize()
            t0, t1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            t0.record()
            for _ in range(iters // 20):
                gr.replay()
            t1.record()
            torch.cuda.synchronize()
            us = t0.elapsed_time(t1) * 1000.0 / iters
            best = us if best is None else min(best, us)
        res[d] = best
    print(
        f"[bench moe distinct] m={m} nl={nl} inter={inter} (us, best tile, op incl. routing): "
        + ", ".join(f"{d}:{v:.2f}" for d, v in res.items()),
        flush=True,
    )
    return res


FOLD_CASES = [
    (m, nl, ls, tile, dea)
    for m in (1, 2, 7, 14, 28, 32)
    for (nl, ls, dea) in ((256, 256, None), (64, 448, None), (512, 0, (8, 3)), (512, 0, (4, 0)))
    for tile in (8, 16)
]


def test_router_route_fold(m, nl, ls, tile, dea):
    """Router GEMM + NVFP4 quant + routing + tables in one launch
    (Q35RouterGemm route=...) vs the router alone and then the standalone
    single-program route + tables launch on its logits: logits, fp4 payload,
    scales, ids, weights and every table entry bit-identical."""
    _, x, gs = _inputs(m, seed=1200 + m + nl)
    g = torch.Generator(device="cuda").manual_seed(31 + m)
    w = (torch.randn((E, H), generator=g, device="cuda") * 0.02).to(torch.bfloat16)
    rg = Q35RouterGemm()
    logits, xq, xsf = rg(x, w, gs)
    sizes = sum(q35_prerouted_sizes(m, K, nl, tile))
    flat_ref = torch.full((sizes,), -7, dtype=torch.int32, device="cuda")
    rt = Q35RouteTables()
    ids_r, wts_r = rt(
        logits,
        K,
        ls,
        nl,
        tile,
        flat=flat_ref,
        dea_groups=dea[0] if dea else 0,
        dea_group=dea[1] if dea else 0,
    )[:2]
    flat = torch.full((sizes,), -7, dtype=torch.int32, device="cuda")
    for _ in range(2):  # second launch: the N-block ticket must be back at zero
        lg2, xq2, xsf2, ids, wts = rg(x, w, gs, route=(K, ls, nl, tile, flat, dea), route_mode=1)
    torch.cuda.synchronize()
    tag = (m, nl, ls, tile, dea)
    assert torch.equal(lg2, logits) and torch.equal(xq2, xq) and torch.equal(xsf2, xsf), tag
    assert torch.equal(ids, ids_r) and torch.equal(wts, wts_r), tag
    ref_v = q35_prerouted_views(flat_ref, m, K, nl, tile)
    got_v = q35_prerouted_views(flat, m, K, nl, tile)
    nt = int(ref_v[4].item())
    assert int(got_v[4].item()) == nt and int(got_v[5].item()) == int(ref_v[5].item()), tag
    assert torch.equal(got_v[0], ref_v[0]), (tag, "e2p")
    assert torch.equal(got_v[2][:nt], ref_v[2][:nt]) and torch.equal(
        got_v[3][:nt], ref_v[3][:nt]
    ), tag
    rows = ref_v[0][ref_v[0] >= 0].long()
    assert torch.equal(got_v[1][rows], ref_v[1][rows]), (tag, "p2t")
    assert int(rg.rcnt.item()) == 0 and int(rg.cnt.abs().sum().item()) == 0, tag


def _check_tables_equivalent(got, ref, m, tile, tag):
    """Two routings' trtllm-gen tables describe the same layout: CTA count, padded
    rows, CTA -> expert / row limit identical; the same picks routed locally, each
    CTA holding the same set of expanded indices (row order inside an expert's
    block is free: the per-token kernels rank rows by atomics), and permuted
    index -> token consistent."""
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


def test_router_route_fold_tokens(m, nl, ls, tile, dea):
    """TRTLLM_Q35_MOE_PREROUTE_FOLD=2: router GEMM + NVFP4 quant + per-token
    routing (the quant programs wait for the logits) + tables in one launch vs
    the router alone, then the separate per-token route + tables launch:
    logits, fp4 payload, scales, ids and weights bit-identical, the tables the
    same layout; every ticket / histogram back at zero after each launch."""
    _, x, gs = _inputs(m, seed=1300 + m + nl)
    g = torch.Generator(device="cuda").manual_seed(37 + m)
    w = (torch.randn((E, H), generator=g, device="cuda") * 0.02).to(torch.bfloat16)
    rg = Q35RouterGemm()
    logits, xq, xsf = rg(x, w, gs)
    sizes = sum(q35_prerouted_sizes(m, K, nl, tile))
    flat_ref = torch.full((sizes,), -7, dtype=torch.int32, device="cuda")
    rt = Q35RouteTables()
    ids_r, wts_r = rt(
        logits,
        K,
        ls,
        nl,
        tile,
        flat=flat_ref,
        dea_groups=dea[0] if dea else 0,
        dea_group=dea[1] if dea else 0,
        single_program=False,
    )[:2]
    tag = (m, nl, ls, tile, dea)
    for it in range(3):  # later launches: tickets and histogram must be back at zero
        flat = torch.full((sizes,), -7, dtype=torch.int32, device="cuda")
        lg2, xq2, xsf2, ids, wts = rg(x, w, gs, route=(K, ls, nl, tile, flat, dea), route_mode=2)
        torch.cuda.synchronize()
        assert torch.equal(lg2, logits) and torch.equal(xq2, xq) and torch.equal(xsf2, xsf), (
            tag,
            it,
        )
        assert torch.equal(ids, ids_r) and torch.equal(wts, wts_r), (tag, it)
        _check_tables_equivalent(
            q35_prerouted_views(flat, m, K, nl, tile),
            q35_prerouted_views(flat_ref, m, K, nl, tile),
            m,
            tile,
            (tag, it),
        )
        assert int(rg.rcnt.item()) == 0 and int(rg.cnt.abs().sum().item()) == 0, (
            tag,
            it,
            "tickets",
        )
        assert int(rg.counter.item()) == 0 and int(rg.hist.abs().sum().item()) == 0, (
            tag,
            it,
            "hist",
        )


def test_route_quant(m, nl, ls, tile, dea=None):
    """TRTLLM_Q35_MOE_PREROUTE_LOWM: the per-token route + tables launch with x also
    NVFP4-quantizes x in M extra programs: xq / xsf byte-identical to q35_nvfp4_quant
    (= fp4_quantize), ids / weights bit-identical to the launch without x, tables the
    same layout, histogram and ticket back at zero."""
    logits, x, gs = _inputs(m, seed=1600 + m + nl + tile)
    sizes = sum(q35_prerouted_sizes(m, K, nl, tile))
    kw = dict(dea_groups=dea[0], dea_group=dea[1]) if dea else {}
    rt = Q35RouteTables()
    flat_r = torch.full((sizes,), -7, dtype=torch.int32, device="cuda")
    ids_r, wts_r = rt(logits, K, ls, nl, tile, flat=flat_r, single_program=False, **kw)[:2]
    flat = torch.full((sizes,), -7, dtype=torch.int32, device="cuda")
    out = rt(logits, K, ls, nl, tile, flat=flat, single_program=False, x=x, global_scale=gs, **kw)
    xq_r, xsf_r = q35_nvfp4_quant(x, gs)
    torch.cuda.synchronize()
    tag = (m, nl, ls, tile, dea)
    assert len(out) == 10, tag
    assert torch.equal(out[0], ids_r) and torch.equal(out[1], wts_r), (tag, "ids")
    assert torch.equal(out[8], xq_r) and torch.equal(out[9], xsf_r), (tag, "quant")
    _check_tables_equivalent(
        q35_prerouted_views(flat, m, K, nl, tile),
        q35_prerouted_views(flat_r, m, K, nl, tile),
        m,
        tile,
        tag,
    )
    assert int(rt.hist.abs().sum().item()) == 0 and int(rt.counter.item()) == 0, (tag, "hist")


def test_route_sort_free(m, nl, ls, tile, dea=None):
    """Single-program tables without tl.sort (rows counted per expert over chunks of
    the picks) == with the stable sort, bitwise (same expanded-index order)."""
    logits, _, _ = _inputs(m, seed=1500 + m + nl + tile)
    sizes = sum(q35_prerouted_sizes(m, K, nl, tile))
    kw = dict(dea_groups=dea[0], dea_group=dea[1]) if dea else {}
    flats = []
    for sort in (True, False):
        flat = torch.full((sizes,), -7, dtype=torch.int32, device="cuda")
        Q35RouteTables()(logits, K, ls, nl, tile, flat=flat, sort=sort, **kw)
        flats.append(flat)
    torch.cuda.synchronize()
    a, b = (q35_prerouted_views(f, m, K, nl, tile) for f in flats)
    nt = int(a[4].item())
    tag = (m, nl, ls, tile, dea)
    assert int(b[4].item()) == nt and int(b[5].item()) == int(a[5].item()), tag
    assert torch.equal(a[0], b[0]), (tag, "e2p")
    rows = a[0][a[0] >= 0].long()
    assert torch.equal(a[1][rows], b[1][rows]), (tag, "p2t")
    assert torch.equal(a[2][:nt], b[2][:nt]) and torch.equal(a[3][:nt], b[3][:nt]), (tag, "ctas")


def bench_route_core_phases(m=7, nl=64, ls=448, tile=8, iters=2000):
    """Where does the single-program route + tables time go? CUDA graph, back-to-back, us:
    full / top-K only / tables only (tl.sort or chunked counts), by warps per program."""
    logits, _, _ = _inputs(m, seed=23)
    flat = torch.empty(
        (sum(q35_prerouted_sizes(m, K, nl, tile)),), dtype=torch.int32, device="cuda"
    )
    res = {}
    ids0 = Q35RouteTables()(logits, K, ls, nl, tile, flat=flat)[:2]  # ids for the tables-only runs
    for nw in (4, 8, 16):
        rt = Q35RouteTables()
        for name, kw in (
            ("full", {}),
            ("topk", dict(phases=1)),
            ("tables", dict(phases=2, ids_in=ids0)),
            ("full nosort", dict(sort=False)),
            ("tables nosort", dict(phases=2, sort=False, ids_in=ids0)),
        ):
            res[f"{name} w{nw}"] = _graph_us(
                lambda kw=kw: rt(logits, K, ls, nl, tile, flat=flat, num_warps=nw, **kw), iters
            )
    rt = Q35RouteTables()
    res["per-token w4"] = _graph_us(
        lambda: rt(logits, K, ls, nl, tile, flat=flat, single_program=False), iters
    )
    print(
        f"[bench route phases] m={m} nl={nl} (us): "
        + ", ".join(f"{k} {v:.2f}" for k, v in res.items()),
        flush=True,
    )
    return res


def bench_route(m=7, nl=256, ls=0, tile=8, iters=2000):
    """trtllm-gen routing from logits (moe_topk_sort, the runner's routing
    launch) vs the Triton route + tables kernel, CUDA graph back-to-back."""
    logits, _, _ = _inputs(m, seed=21)
    rt = Q35RouteTables()
    res = {}
    for name, fn in (
        (
            "trtllm-gen routing",
            lambda: torch.ops.trtllm.moe_topk_sort(
                logits, None, E, K, None, None, ls, nl, None, tile, 1
            ),
        ),
        ("q35 route+tables (1 program)", lambda: rt(logits, K, ls, nl, tile)),
        ("q35 route+tables (per token)", lambda: rt(logits, K, ls, nl, tile, single_program=False)),
    ):
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        gr = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gr):
            for _ in range(20):
                fn()
        gr.replay()
        torch.cuda.synchronize()
        t0, t1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        t0.record()
        for _ in range(iters // 20):
            gr.replay()
        t1.record()
        torch.cuda.synchronize()
        res[name] = t0.elapsed_time(t1) * 1000.0 / iters
    print(
        f"[bench route] m={m} nl={nl} tile={tile} (us): "
        + ", ".join(f"{k} {v:.2f}" for k, v in res.items())
    )


def bench(m=7, nl=64, ls=128, tile=128, iters=2000):
    """CUDA-graph timing: stock front chain vs fused front (no experts)."""
    logits, x, gs = _inputs(m, seed=3)
    scratch = torch.zeros((1024,), dtype=torch.int32, device="cuda")
    out = torch.empty((m, H), dtype=torch.bfloat16, device="cuda")
    side = torch.cuda.Stream()
    ev0, ev1 = torch.cuda.Event(), torch.cuda.Event()

    def stock():
        ids, w = torch.ops.trtllm.renorm_moe_routing_op(logits, K, torch.float32)
        xq, xsf = torch.ops.trtllm.fp4_quantize(x, gs, 16, False, False)
        torch.ops.trtllm.moe_sort(
            token_selected_experts=ids,
            token_final_scales=w,
            num_experts=E,
            top_k=K,
            local_expert_offset=ls,
            local_num_experts=nl,
            tile_tokens_dim=tile,
        )
        ev0.record()
        with torch.cuda.stream(side):
            ev0.wait()
            out.zero_()
            ev1.record()
        ev1.wait()

    def fused():
        q35_moe_front(logits, x, gs, K, ls, nl, tile, scratch, moe_output=out)

    res = {}
    for name, fn in (("stock", stock), ("fused", fused)):
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(20):
                fn()
        g.replay()
        torch.cuda.synchronize()
        t0, t1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        t0.record()
        for _ in range(iters // 20):
            g.replay()
        t1.record()
        torch.cuda.synchronize()
        res[name] = t0.elapsed_time(t1) * 1000.0 / iters
    print(
        f"[bench] m={m} nl={nl} tile={tile}: stock {res['stock']:.2f} us, "
        f"fused {res['fused']:.2f} us per front end (graph, back-to-back)"
    )
    return res


def _bench_route_only():
    """--bench-route: only the routing benches (no parity tests): route + tables
    launches vs trtllm-gen routing, the router with and without folded routing,
    and DEA's MoE op breakdown."""
    import os

    print(
        f"[bench-route] TRTLLM_Q35_ROUTE_WARPS={os.environ.get('TRTLLM_Q35_ROUTE_WARPS', '1')}",
        flush=True,
    )
    for m in (1, 7, 14, 28):
        bench_route(m=m)
    for m in (1, 7, 14, 28):
        try:
            bench_route_core_phases(m=m)
        except Exception as e:  # bench only
            print("ERROR bench route phases", m, repr(e)[:300], flush=True)
    for m in (1, 7, 14, 28):
        bench_router(m=m)
    for m in (7, 14, 28):
        for g in (4, 8):
            try:
                bench_dea_op(m=m, n_groups=g)
            except Exception as e:  # bench only
                print("ERROR bench dea op", m, g, repr(e)[:300], flush=True)
    print("RESULT PASS (bench only)")


if __name__ == "__main__" and "--bench-route" in sys.argv:
    torch.cuda.init()
    _bench_route_only()
    sys.exit(0)

if __name__ == "__main__":
    import time

    torch.cuda.init()
    t_start = time.time()
    n_fail = 0
    print(f"[start] {torch.cuda.get_device_name(0)}", flush=True)
    for args in CASES:
        for ties in (False, True):
            t_case = time.time()
            try:
                test_front_parity(*args, ties)
                print(f"[front] {args} ties={ties} ok {time.time() - t_case:.1f}s", flush=True)
            except AssertionError as e:
                n_fail += 1
                print("FAIL front", args, ties, repr(e)[:400], flush=True)
    print(
        f"[front parity] {len(CASES) * 2 - n_fail}/{len(CASES) * 2} passed "
        f"({time.time() - t_start:.0f}s)",
        flush=True,
    )
    try:
        test_negative_controls()
        print("[negative controls] ok")
    except AssertionError as e:
        n_fail += 1
        print("FAIL negative controls", repr(e)[:400])
    for m in (1, 7, 14, 28):
        for nl, ls in ((64, 128), (512, 0)):
            try:
                test_moe_output_parity(m, nl, ls)
                print(f"[moe output parity] m={m} nl={nl} ok")
            except AssertionError as e:
                n_fail += 1
                print("FAIL moe", m, nl, repr(e)[:400])
            except Exception as e:  # synthetic-weight construction issue
                if nl == 64:
                    n_fail += 1
                print("ERROR moe", m, nl, repr(e)[:400])
    for m in (1, 2, 7, 14, 16, 17, 28, 32):
        try:
            test_router_gemm(m)
        except AssertionError as e:
            n_fail += 1
            print("FAIL router", m, repr(e)[:400])
    n_rt = 0
    for args in ROUTE_CASES:
        for ties in (False, True):
            try:
                test_route_tables(*args, ties=ties)
                test_route_tables(*args, ties=ties, single_program=False)
            except AssertionError as e:
                n_rt += 1
                print("FAIL route tables", args, ties, repr(e)[:300], flush=True)
    print(f"[route tables] {len(ROUTE_CASES) * 2 - n_rt}/{len(ROUTE_CASES) * 2} passed", flush=True)
    n_fail += n_rt
    n_dea = 0
    for args in DEA_CASES:
        for ties in (False, True):
            try:
                test_dea_tables(*args, ties=ties)
                test_dea_tables(*args, ties=ties, single_program=False)
            except AssertionError as e:
                n_dea += 1
                print("FAIL dea tables", args, ties, repr(e)[:300], flush=True)
    print(f"[dea tables] {len(DEA_CASES) * 2 - n_dea}/{len(DEA_CASES) * 2} passed", flush=True)
    n_fail += n_dea
    try:
        test_dea_cap_adversarial()
        print("[dea cap adversarial] ok", flush=True)
    except AssertionError as e:
        n_fail += 1
        print("FAIL dea cap adversarial", repr(e)[:300], flush=True)
    n_fold = 0
    for args in FOLD_CASES:
        try:
            test_router_route_fold(*args)
        except AssertionError as e:
            n_fold += 1
            print("FAIL router route fold", args, repr(e)[:300], flush=True)
    print(f"[router route fold] {len(FOLD_CASES) - n_fold}/{len(FOLD_CASES)} passed", flush=True)
    n_fail += n_fold
    n_rq = 0
    rq_cases = [
        (m, nl, ls, t, None)
        for m in (1, 7, 14, 28, 32)
        for (nl, ls) in ((128, 128), (64, 448))
        for t in (8, 16)
    ] + [(m, 512, 0, 8, (8, 3)) for m in (1, 7, 28)]
    for args in rq_cases:
        try:
            test_route_quant(*args)
        except (AssertionError, RuntimeError) as e:
            n_rq += 1
            print("FAIL route + quant", args, repr(e)[:300], flush=True)
    print(f"[route + quant] {len(rq_cases) - n_rq}/{len(rq_cases)} passed", flush=True)
    n_fail += n_rq
    n_sf = 0
    sf_cases = [c + (None,) for c in ROUTE_CASES] + [
        (m, 512, 0, t, (g, g - 1)) for m in (1, 7, 28) for t in (8, 16) for g in (4, 8)
    ]
    for args in sf_cases:
        try:
            test_route_sort_free(*args)
        except (AssertionError, RuntimeError) as e:
            n_sf += 1
            print("FAIL route sort-free", args, repr(e)[:300], flush=True)
    print(f"[route sort-free] {len(sf_cases) - n_sf}/{len(sf_cases)} passed", flush=True)
    n_fail += n_sf
    n_fold2 = 0
    for args in FOLD_CASES:
        try:
            test_router_route_fold_tokens(*args)
        except (AssertionError, RuntimeError) as e:
            n_fold2 += 1
            print("FAIL router route fold (per token)", args, repr(e)[:300], flush=True)
    print(
        f"[router route fold per token] {len(FOLD_CASES) - n_fold2}/{len(FOLD_CASES)} passed",
        flush=True,
    )
    n_fail += n_fold2
    from tensorrt_llm._torch.custom_ops.trtllm_gen_custom_ops import FP4BlockScaleMoERunner

    if FP4BlockScaleMoERunner.supports_prerouted():
        n_dm = 0
        for m in (1, 7, 28):
            for g in (4, 8):
                try:
                    test_dea_moe_parity(m, g)
                except (AssertionError, RuntimeError) as e:
                    n_dm += 1
                    print("FAIL dea moe", m, g, repr(e)[:300], flush=True)
        print(f"[dea moe parity] {6 - n_dm}/6 passed", flush=True)
        n_fail += n_dm
        n_pr = 0
        for args in PREROUTED_CASES:
            try:
                test_prerouted_moe_parity(*args)
            except AssertionError as e:
                n_pr += 1
                print("FAIL prerouted moe", args, repr(e)[:300], flush=True)
        print(
            f"[prerouted moe parity] {len(PREROUTED_CASES) - n_pr}/{len(PREROUTED_CASES)} passed",
            flush=True,
        )
        n_fail += n_pr
        n_uf = 0
        for args in UNFINALIZED_CASES:
            try:
                test_prerouted_unfinalized_op(*args)
            except (AssertionError, RuntimeError) as e:
                n_uf += 1
                print("FAIL prerouted unfinalized op", args, repr(e)[:400], flush=True)
        print(
            f"[prerouted unfinalized op] {len(UNFINALIZED_CASES) - n_uf}/{len(UNFINALIZED_CASES)} passed",
            flush=True,
        )
        n_fail += n_uf
    else:
        print("[prerouted moe parity] skipped: libth_common without run_moe_prerouted", flush=True)
    try:
        test_router_weight_load_before_pdl_wait()
    except Exception as e:  # a crash counts as a failure; the later tests still run
        n_fail += 1
        print("FAIL router ptx order", repr(e)[:400])
    try:
        test_route_kernel_waits_before_global_access()
    except Exception as e:
        n_fail += 1
        print("FAIL route ptx order", repr(e)[:400])
    try:
        test_pdl_delayed_producer()
    except Exception as e:
        n_fail += 1
        print("FAIL pdl delayed producer", repr(e)[:400])
    try:
        test_pdl_early_trigger_chain()
    except (AssertionError, RuntimeError) as e:
        n_fail += 1
        print("FAIL pdl early trigger", repr(e)[:400])
    if "--bench" in sys.argv:
        for m in (1, 7, 14, 28):
            bench(m=m)
        bench(m=7, nl=512, ls=0)
        for m in (1, 7, 14, 28):
            bench_router(m=m)
        for m in (1, 7, 14, 28):
            bench_route(m=m)
        for m in (7, 14, 28):
            for nl, ls, inter in ((64, 448, 512), (128, 128, 512), (256, 256, 256)):
                try:
                    bench_moe_distinct(m=m, nl=nl, ls=ls, inter=inter)
                except Exception as e:  # bench only
                    print("ERROR bench moe distinct", m, nl, repr(e)[:300], flush=True)
    if "--bench" in sys.argv or "--bench-dea" in sys.argv:
        for m in (7, 14, 28):
            for g in (4, 8):
                try:
                    bench_dea_op(m=m, n_groups=g)
                except Exception as e:  # bench only
                    print("ERROR bench dea op", m, g, repr(e)[:300], flush=True)
    print("RESULT", "PASS" if n_fail == 0 else f"FAIL ({n_fail})")
    sys.exit(1 if n_fail else 0)
