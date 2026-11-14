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
"""TRTLLM-gen NVFP4 MoE decode front (trtllm_gen/decode_front.py): router GEMM, route + tables, the
one-launch fused front and the route front against reference routing, and the pre-routed MoE output."""

from types import SimpleNamespace

import pytest
import torch
import triton
import triton.language as tl

from tensorrt_llm._torch.modules.low_m_mma_gemm import default_tactic, low_m_mma_gemm
from tensorrt_llm._torch.moe.fused_moe.impl_contract import EngineRouting, PreroutedRouting
from tensorrt_llm._torch.moe.fused_moe.trtllm_gen import decode_front as df

E, K, H = 512, 10, 4096

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="needs SM100/SM103",
)

# The low-M BF16 router tactic for [E, H] (one tile per CTA, no split-K), shared by both small-M fronts.
_ROUTER_TACTIC = default_tactic(1, E, H, fp8=False)
_K_WARPS = _ROUTER_TACTIC[0]


def _inputs(
    m: int, seed: int, dup: bool = False, crowd: bool = False, h: int = H
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = (torch.randn((m, h), generator=g, device="cuda") * 0.5).to(torch.bfloat16)
    x[0, :32] = 0  # all-zero quant blocks
    if m > 1:
        x[1, 100] = 40.0  # an outlier block
    w = (torch.randn((E, h), generator=g, device="cuda") * 0.02).to(torch.bfloat16)
    if dup:
        w[1::2] = w[0::2]  # identical router row pairs: exact logit ties in every token's top-k
    if crowd:
        # Experts 16..27 (one routing lane) dominate the even tokens: the route front's full-sort fallback.
        base = x[0::2].float().mean(0)
        base = base / base.norm()
        for i in range(12):
            noise = 0.01 * torch.randn(h, generator=g, device="cuda")
            w[16 + i] = (base * (3.0 - 0.1 * i) + noise).to(torch.bfloat16)
    gs = torch.tensor(
        [448.0 * 6.0 / x.float().abs().max().item()], dtype=torch.float32, device="cuda"
    )
    return x, w, gs


def _cublas(x, w):
    return torch.ops.trtllm.cublas_mm(x, w.t(), bias=None, out_dtype=torch.bfloat16)


def _flat(m, nl, tile):
    size = sum(df.prerouted_table_sizes(m, K, nl, tile))
    return torch.full((size,), -7, dtype=torch.int32, device="cuda")


def _tables(flat, m, nl, tile):
    return df.prerouted_table_views(flat, m, K, nl, tile)


def _bits(t):
    return t.view(torch.int16) if t.element_size() == 2 else t


def _check_tables(got, ref, tile, tag, token_order=False):
    """Same layout up to the row order inside an expert's block (the finalize does not depend on it)."""
    e2p, p2t, cte, ctl, nct, npd = got
    r_e2p, _, r_cte, r_ctl, r_nct, r_npd = ref
    nt = int(r_nct.item())
    assert int(nct.item()) == nt and int(npd.item()) == int(r_npd.item()), (tag, "counts")
    assert torch.equal(cte[:nt], r_cte[:nt]) and torch.equal(ctl[:nt], r_ctl[:nt]), (tag, "ctas")
    assert torch.equal(e2p < 0, r_e2p < 0), (tag, "local picks")
    valid = (e2p >= 0).nonzero().view(-1)
    got_p2e = torch.full((nt * tile,), -1, dtype=torch.int64, device=e2p.device)
    ref_p2e = torch.full_like(got_p2e, -1)
    got_p2e[e2p[valid].long()] = valid
    ref_p2e[r_e2p[valid].long()] = valid
    t = 0
    while t < nt:  # one expert's CTAs together
        t1 = t
        while t1 + 1 < nt and int(r_cte[t1 + 1].item()) == int(r_cte[t].item()):
            t1 += 1
        lo, hi = t * tile, int(r_ctl[t1].item())
        assert torch.equal(got_p2e[lo:hi].sort().values, ref_p2e[lo:hi].sort().values), (tag, t)
        t = t1 + 1
    assert torch.equal(p2t[e2p[valid].long()], (valid // K).to(torch.int32)), (tag, "p2t")
    if token_order:
        p2t, ctl = p2t.cpu(), ctl.cpu()
        for c in range(nt):
            rows = p2t[c * tile : int(ctl[c].item())].tolist()
            assert rows == sorted(rows), (tag, "block rows not in token order", c)


def _check_dup_ties(ids, tag):
    # Router rows 2i / 2i + 1 tie exactly: a picked odd expert implies its even twin was picked.
    for row in ids.tolist():
        for e in row:
            assert e % 2 == 0 or (e - 1) in row, (tag, "tie went to the larger id", row)


def _two_launch(x, w, gs, ls, nl, tile):
    """Low-M router GEMM, then the per-token route + tables + quant launch."""
    flat = _flat(x.shape[0], nl, tile)
    logits = low_m_mma_gemm(x16=x, w_bf16=w, out_dtype=torch.bfloat16, tactic_bf16=_ROUTER_TACTIC)
    out = df.RouteTables()(logits, K, ls, nl, tile, flat=flat, x=x, global_scale=gs)
    return logits, out[8], out[9], out[0], out[1], flat


def _route_front(x, w, gs, flat, ls, nl, tile, lean=False):
    front = df.RouteFront(lean=lean)
    fn = front.compiled(x, E, tile, nl, K)
    assert fn is not None, "route front does not apply"
    logits = _cublas(x, w)
    return (logits,) + front(fn, logits, x, gs, flat, tile, ls, nl, K)


def _ref_route(
    logits: torch.Tensor,
    x: torch.Tensor,
    gs: torch.Tensor,
    ls: int,
    nl: int,
    tile: int,
    quant: bool = True,
) -> tuple[torch.Tensor, ...]:
    """The Triton per-token route + tables (+ quant, power-of-two H only) kernel at any token count
    (RouteTables caps it at 32)."""
    m, h = x.shape
    nlp = triton.next_power_of_2(nl)
    hist = torch.zeros((nlp,), dtype=torch.int32, device="cuda")
    counter = torch.zeros((1,), dtype=torch.int32, device="cuda")
    ids = torch.empty((m, K), dtype=torch.int32, device="cuda")
    wts = torch.empty((m, K), dtype=torch.bfloat16, device="cuda")
    rank = torch.empty((m * K,), dtype=torch.int32, device="cuda")
    flat = _flat(m, nl, tile)
    xq = torch.empty((m, h // 2), dtype=torch.uint8, device="cuda")
    xsf = torch.empty((m, h // 16), dtype=torch.uint8, device="cuda")
    df._route_tables_kernel[(2 * m if quant else m,)](
        logits, ids, wts, hist, rank, counter, *_tables(flat, m, nl, tile), m, ls,
        x, x.stride(0), gs, xq, xsf,
        E=E, K=K, KP=16, TILE=tile, NL=nl, NLP=nlp,
        EXPANDED_PAD=triton.next_power_of_2(m * K),
        MAX_CTAS_PER_EXPERT=triton.cdiv(m, tile),
        USE_PDL=False, QUANT_H=h if quant else 0, num_warps=4,
    )  # fmt: skip
    return ids, wts, flat, xq, xsf


def _moe_runner(nl, ls, inter, seed, finalize=True):
    from tensorrt_llm._torch.custom_ops.trtllm_gen_custom_ops import FP4BlockScaleMoERunner
    from tensorrt_llm._torch.moe.fused_moe.routing import RoutingMethodType
    from tensorrt_llm._torch.utils import ActType_TrtllmGen

    g = torch.Generator(device="cuda").manual_seed(seed)

    def u8(*s, lo=0, hi=256):
        return torch.randint(lo, hi, s, generator=g, device="cuda", dtype=torch.int32).to(
            torch.uint8
        )

    e4m3 = torch.float8_e4m3fn
    w1, s1 = u8(nl, 2 * inter, H // 2), u8(nl, 2 * inter, H // 16, lo=0x28, hi=0x38).view(e4m3)
    w2, s2 = u8(nl, H, inter // 2), u8(nl, H, inter // 16, lo=0x28, hi=0x38).view(e4m3)
    a1 = torch.rand((nl,), generator=g, device="cuda") * 1e-2 + 1e-3
    a2 = torch.rand((nl,), generator=g, device="cuda") * 1e-2 + 1e-3
    rmt, act = int(RoutingMethodType.Renormalize), ActType_TrtllmGen.SwiGlu.value
    runner = FP4BlockScaleMoERunner(E, K, None, None, inter, ls, nl, None, rmt, finalize, act)

    def inputs(logits, xq, xsf):
        sf = xsf.reshape(-1).view(e4m3)
        return [logits, None, xq, sf, w1, s1, None, None, None, None, w2, s2, None, a1, a1, a2]

    def run(logits, xq, xsf, tactic, pre=None):
        inp = inputs(logits, xq, xsf) + [None, None]
        if pre is None:
            return runner(inp, tactic=tactic)
        return runner(inp, tactic=tactic, prerouted=pre)

    return runner, run, inputs


def _configs(runner, m, nl, inter):
    by_tile = {}
    for tile, cfg in runner.get_runner().get_valid_configs(K, H, inter, nl, m):
        by_tile.setdefault(int(tile), []).append(int(cfg))
    return {t: sorted(c) for t, c in by_tile.items()}


@pytest.mark.parametrize("m", [33, 40, 77, 96, 97, 128])
def test_route_front_router_gemm(m):
    """The route front steps' CuTe router GEMM (moe_router_gemm): as accurate as cuBLAS, and a token's logits do
    not depend on the batch it runs in (fixed split-K order); declined above ROUTER_GEMM_MAX_TOKENS rows."""
    x, w, _ = _inputs(m, seed=21 + m)
    rg = df.RouteFrontRouterGemm()
    a = rg(x, w)
    assert a is not None and torch.equal(a, rg(x, w))
    exact = x.float() @ w.float().t()
    err = (a.float() - exact).abs().max().item()
    err_ref = (_cublas(x, w).float() - exact).abs().max().item()
    assert err <= 2 * err_ref + 1e-3, (err, err_ref)
    head = rg(x[:33].contiguous(), w)
    assert torch.equal(head, a[:33])
    big, _, _ = _inputs(df.ROUTER_GEMM_MAX_TOKENS + 1, seed=21)
    assert rg(big, w) is None


def _ref_candidates(logits: torch.Tensor, top_k: int = K) -> torch.Tensor:
    """trtllm-gen reduceTopK keys (twiddled BF16 << 16 | 65535 - expert) of each 128-expert tile, top ``top_k``
    descending: equal logits go to the smaller expert."""
    m, e = logits.shape
    bits = logits.view(torch.int16).to(torch.int64) & 0xFFFF
    tw = torch.where((bits & 0x8000) != 0, bits ^ 0xFFFF, bits | 0x8000)
    keys = (tw << 16) | (65535 - torch.arange(e, device=logits.device))
    top = keys.view(m, e // 128, 128).sort(dim=-1, descending=True).values[..., :top_k]
    return torch.where(top >= 2**31, top - 2**32, top).to(torch.int32)


def _engine_route_gemm(x, w, gs, poison: int, top_k: int = K):
    """The engine-route router GEMM into outputs filled with ``poison`` (a missing store keeps it)."""
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute

    from tensorrt_llm._torch.flashinfer_utils import get_env_enable_pdl
    from tensorrt_llm._torch.kernels.blackwell.utils import make_ptr

    m, h = x.shape
    fn = df._router_gemm_compile(m, h, E, x.stride(0), get_env_enable_pdl(), route_top_k=top_k)
    assert fn is not None, "the engine-route router GEMM declined"
    dev = x.device
    logits = torch.full((m, E), poison, dtype=torch.int16, device=dev).view(torch.bfloat16)
    cand = torch.full((m, E // 128, top_k), poison, dtype=torch.int32, device=dev)
    xq = torch.full((m, h // 2), poison & 0xFF, dtype=torch.uint8, device=dev)
    xsf = torch.full((m, h // 16), poison & 0xFF, dtype=torch.uint8, device=dev)

    def bf(t):
        return make_ptr(cutlass.BFloat16, t.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)

    p64 = lambda t: cutlass.Int64(t.data_ptr())  # noqa: E731
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    fn(bf(x), bf(w), p64(logits), stream, p64(cand), p64(gs), p64(xq), p64(xsf))
    return logits, cand, xq, xsf


# Steps the MoE engine routes (above the router GEMM's 32 rows, up to the engine's 64): MTP-6 decode at CUDA-graph
# batches 5..9 (35..63 rows) and the bounds.
@pytest.mark.parametrize(
    "m,dup", [(33, False), (35, False), (42, True), (49, False), (56, False), (63, False)]
)
def test_engine_route_router_gemm(m, dup):
    """The router GEMM with the engine-route epilogue: logits bitwise the plain kernel's, each token's top-10 keys per
    128-expert tile equal to reduceTopK's (tie order included), x bytes equal to fp4_quantize; every output written
    (two runs over different poison)."""
    x, w, gs = _inputs(m, seed=1900 + m, dup=dup)
    plain = df.RouteFrontRouterGemm()(x, w)
    assert plain is not None
    q_ref, sf_ref = torch.ops.trtllm.fp4_quantize(x, gs, 16, False, False)
    q_ref = q_ref.view(torch.uint8).view(m, H // 2)
    sf_ref = sf_ref.view(torch.uint8).flatten()[: m * H // 16].view(m, H // 16)
    cand_ref = _ref_candidates(plain)
    for poison in (-1, 0x2AAA):
        logits, cand, xq, xsf = _engine_route_gemm(x, w, gs, poison)
        tag = (m, dup, poison)
        assert torch.equal(_bits(logits), _bits(plain)), (tag, "logits")
        assert torch.equal(cand, cand_ref), (tag, "candidates")
        assert torch.equal(xq, q_ref) and torch.equal(xsf, sf_ref), (tag, "x quant")
    out = df.RouteFrontRouterGemm(engine_route=True)(x, w, gs, K)
    assert out is not None
    for name, a, b in zip(("logits", "cand", "xq", "xsf"), out, (plain, cand_ref, q_ref, sf_ref)):
        assert torch.equal(_bits(a), _bits(b)), (m, name)


@pytest.mark.parametrize("m,top_k", [(35, 8), (56, 4)])
def test_engine_route_router_gemm_top_k(m, top_k):
    """The routing epilogue for another top-k: each token's top-``top_k`` keys per expert tile equal to reduceTopK's,
    logits and x bytes as at top-10."""
    x, w, gs = _inputs(m, seed=1950 + m)
    plain = df.RouteFrontRouterGemm()(x, w)
    ref = df.RouteFrontRouterGemm(engine_route=True)(x, w, gs, K)
    logits, cand, xq, xsf = _engine_route_gemm(x, w, gs, -1, top_k)
    assert torch.equal(_bits(logits), _bits(plain))
    assert torch.equal(cand, _ref_candidates(plain, top_k))
    assert torch.equal(xq, ref[2]) and torch.equal(xsf, ref[3])


def test_engine_route_router_gemm_rejects():
    """The routing epilogue takes 1..32 keys per tile and 8 token rows per expert tile in a 32- or 64-token tile:
    256 experts (a 16-token tile) and 33 keys are declined (the caller keeps the route front)."""
    from tensorrt_llm._torch.flashinfer_utils import get_env_enable_pdl
    from tensorrt_llm._torch.kernels.blackwell.moe_router_gemm import MoeRouterGemmKernel

    pdl = get_env_enable_pdl()
    assert df._router_gemm_compile(35, H, 256, H, pdl, route_top_k=K) is None
    assert df._router_gemm_compile(35, H, E, H, pdl, route_top_k=33) is None
    shape = dict(num_rows=35, k_extent=H, num_experts=E, a_stride=H, token_tile=32)
    assert MoeRouterGemmKernel.supports(**shape, route_top_k=K)
    assert not MoeRouterGemmKernel.supports(**dict(shape, token_tile=64), route_top_k=K)
    assert MoeRouterGemmKernel.supports(
        **dict(shape, num_experts=1024, token_tile=64), route_top_k=K
    )


@pytest.mark.parametrize("m", [1, 7, 32])
def test_router_gemm(m):
    """Triton router GEMM: deterministic, as accurate as cuBLAS; its fused and the standalone NVFP4 quant are
    byte-identical to fp4_quantize."""
    x, w, gs = _inputs(m, seed=11 + m)
    rg = df.RouterGemm()
    a = rg(x, w)
    c, cq, csf = rg(x, w, gs)
    sq, ssf = df.nvfp4_quant_rows(x, gs)
    xq, xsf = torch.ops.trtllm.fp4_quantize(x, gs, 16, False, False)
    xq, xsf = xq.view(torch.uint8), xsf.view(m, -1)
    assert torch.equal(a, rg(x, w)) and torch.equal(a, c)
    assert torch.equal(cq, xq) and torch.equal(csf, xsf)
    assert torch.equal(sq, xq) and torch.equal(ssf, xsf)
    exact = x.float() @ w.float().t()
    err = (a.float() - exact).abs().max().item()
    err_ref = (_cublas(x, w).float() - exact).abs().max().item()
    assert err <= 2 * err_ref + 1e-3, (err, err_ref)


@pytest.mark.parametrize(
    "m,nl,ls,tile,dup",
    [
        (1, 512, 0, 8, False),
        (7, 64, 448, 16, False),
        (32, 128, 128, 8, False),
        (7, 512, 0, 16, True),
        (32, 64, 448, 8, True),
    ],
)
def test_route_tables(m, nl, ls, tile, dup):
    """RouteTables vs the trtllm-gen routing runner (moe_topk_sort, Renormalize); the launch with x quant gives
    the same routing plus fp4_quantize's bytes, and leaves the histogram and ticket at zero."""
    x, w, gs = _inputs(m, seed=500 + m + nl + tile, dup=dup)
    logits = _cublas(x, w)
    rt = df.RouteTables()
    flat = _flat(m, nl, tile)
    ids, wts = rt(logits, K, ls, nl, tile, flat=flat)[:2]
    ref = torch.ops.trtllm.moe_topk_sort(logits, None, E, K, None, None, ls, nl, None, tile, 1)
    tag = (m, nl, ls, tile, dup)
    r_te, r_ml, r_e2p, _, r_tot, r_nt = ref[:6]
    _check_tables(
        _tables(flat, m, nl, tile), (r_e2p.view(-1), None, r_te, r_ml, r_nt, r_tot), tile, tag
    )
    ref_ids = torch.topk(logits.float(), K, dim=-1).indices
    assert dup or torch.equal(ids.long().sort(-1).values, ref_ids.sort(-1).values), tag
    if dup:
        _check_dup_ties(ids, tag)
    if len(ref) > 6:  # the runner's weights in its own k order: compare as sets
        r_w = ref[6].float().view(m, K).sort(-1).values
        torch.testing.assert_close(wts.float().sort(-1).values, r_w, atol=1e-2, rtol=0)
        if ref[6].dtype == wts.dtype:
            assert torch.equal(wts.float().sort(-1).values, r_w), tag
    flat_q = _flat(m, nl, tile)
    out = rt(logits, K, ls, nl, tile, flat=flat_q, x=x, global_scale=gs)
    xq, xsf = torch.ops.trtllm.fp4_quantize(x, gs, 16, False, False)
    assert torch.equal(out[0], ids) and torch.equal(out[1], wts), tag
    assert torch.equal(out[8], xq.view(torch.uint8)) and torch.equal(out[9], xsf.view(m, -1)), tag
    _check_tables(_tables(flat_q, m, nl, tile), _tables(flat, m, nl, tile), tile, tag)
    assert int(rt.hist.abs().sum().item()) == 0 and int(rt.counter.item()) == 0, tag


@pytest.mark.parametrize(
    "m,nl,ls,tile,dup",
    [
        (1, 64, 448, 8, False),
        (9, 64, 0, 8, False),
        (16, 128, 256, 16, False),
        (7, 64, 448, 8, True),
        (16, 256, 0, 8, True),
    ],
)
def test_fused_front_parity(m, nl, ls, tile, dup):
    """The one-launch front equals the two-launch front bitwise, writes block rows in token order, and leaves
    its staging buffer clear for the next launch."""
    x, w, gs = _inputs(m, seed=100 + 7 * m + nl + tile, dup=dup)
    ref = _two_launch(x, w, gs, ls, nl, tile)
    front = df.FusedFront()
    for it in range(2):
        flat = torch.full_like(ref[5], -7)
        out = front(x, w, gs, flat, tile, ls, nl, K, _K_WARPS)
        assert out is not None, "fused front declined"
        tag = (m, nl, ls, tile, dup, it)
        for name, a, b in zip(("logits", "xq", "xsf", "ids", "wts"), out, ref):
            assert torch.equal(_bits(a), _bits(b)), (tag, name)
        _check_tables(
            _tables(flat, m, nl, tile), _tables(ref[5], m, nl, tile), tile, tag, token_order=True
        )
        assert int(front.stage.abs().sum().item()) == 0, (tag, "staging buffer not clear")
    if dup:
        _check_dup_ties(out[3], tag)


# M = 40 (odd split over the 8 routing CTAs), 112 / 224 (MTP-6 verify at batch 16 / 32), 256 (clusters of 8);
# 257..1024 run on clusters of 16, above 512 with two-word token masks.
_ROUTE_FRONT_CASES = [
    (21, 64, 448, 8, False, False),
    (40, 128, 0, 8, False, False),
    (56, 512, 0, 16, False, False),
    (224, 128, 384, 32, False, False),
    (256, 256, 256, 8, False, False),
    (112, 128, 128, 8, True, False),
    (224, 128, 0, 8, False, True),
    (280, 128, 128, 16, False, False),
    (448, 64, 448, 16, False, False),
    (448, 128, 0, 16, True, False),
    (448, 128, 0, 16, False, True),
    (513, 128, 0, 16, False, False),
    (896, 256, 256, 32, True, False),
    (1024, 128, 0, 16, False, True),
]
# The lean table build runs on clusters of 8 only; above 256 rows the route front takes the default kernel.
_LEAN_MAX_M = 256


@pytest.mark.parametrize(
    "m,nl,ls,tile,dup,crowd,lean",
    [
        pytest.param(*c, lean, id="-".join(map(str, ("lean" if lean else "default", *c))))
        for lean in (False, True)
        for c in _ROUTE_FRONT_CASES
        if not lean or c[0] <= _LEAN_MAX_M
    ],
)
def test_route_front_parity(m, nl, ls, tile, dup, crowd, lean):
    """The route front after the stock router GEMM equals the Triton route kernel and fp4_quantize bitwise,
    tables in token order; a repeated launch gives the same result. ``lean`` (the decode kernels' table build, up
    to 256 rows) gives the same bits."""
    x, w, gs = _inputs(m, seed=100 + m + nl + tile, dup=dup, crowd=crowd)
    flat = _flat(m, nl, tile)
    lg, xq, xsf, ids, wts = _route_front(x, w, gs, flat, ls, nl, tile, lean)
    ids_r, wts_r, flat_r, xq_r, xsf_r = _ref_route(_cublas(x, w), x, gs, ls, nl, tile)
    q_ref, sf_ref = torch.ops.trtllm.fp4_quantize(x, gs, 16, False, False)
    tag = (m, nl, ls, tile, dup, crowd)
    assert torch.equal(ids, ids_r) and torch.equal(_bits(wts), _bits(wts_r)), (tag, "routing")
    assert torch.equal(xq, q_ref.view(torch.uint8).view(m, H // 2)), (tag, "xq")
    assert torch.equal(xsf.flatten(), sf_ref.view(torch.uint8).flatten()[: m * H // 16]), tag
    assert torch.equal(xq, xq_r) and torch.equal(xsf, xsf_r), (tag, "quant vs the route kernel")
    _check_tables(
        _tables(flat, m, nl, tile), _tables(flat_r, m, nl, tile), tile, tag, token_order=True
    )
    if crowd:
        lane1 = ((ids_r >= 16) & (ids_r < 32)).sum(-1)
        assert int((lane1 >= 5).sum().item()) > 0, (tag, "the crowd case misses the fallback")
    if dup:
        _check_dup_ties(ids, tag)
    flat2 = torch.full_like(flat, -7)
    again = _route_front(x, w, gs, flat2, ls, nl, tile, lean)
    for a, b in zip(again, (lg, xq, xsf, ids, wts)):
        assert torch.equal(_bits(a), _bits(b)), (tag, "repeat")
    _check_tables(_tables(flat2, m, nl, tile), _tables(flat, m, nl, tile), tile, (tag, "repeat"))


# M = 64: the route-front step of MTP-3 at CUDA-graph batch 16 (perf-sanity qwen3_8_flash_next_fp4_tp1_mtp3_8k1k).
@pytest.mark.parametrize("m,nl,ls,tile", [(64, 512, 0, 8), (40, 128, 384, 16)])
def test_route_front_non_pow2_hidden(m: int, nl: int, ls: int, tile: int, h: int = 2560) -> None:
    """The route front at a hidden size that is not a power of two (Qwen3.8-Flash-Next), which the small-M
    fronts decline: routing equal to the Triton route kernel, x bytes equal to fp4_quantize."""
    x, w, gs = _inputs(m, seed=1700 + m + nl, h=h)
    flat = _flat(m, nl, tile)
    _, xq, xsf, ids, wts = _route_front(x, w, gs, flat, ls, nl, tile)
    ids_r, wts_r, flat_r, _, _ = _ref_route(_cublas(x, w), x, gs, ls, nl, tile, quant=False)
    q_ref, sf_ref = torch.ops.trtllm.fp4_quantize(x, gs, 16, False, False)
    tag = (m, nl, ls, tile, h)
    assert torch.equal(ids, ids_r) and torch.equal(_bits(wts), _bits(wts_r)), (tag, "routing")
    assert torch.equal(xq, q_ref.view(torch.uint8).view(m, h // 2)), (tag, "xq")
    assert torch.equal(xsf.flatten(), sf_ref.view(torch.uint8).flatten()[: m * h // 16]), tag
    _check_tables(
        _tables(flat, m, nl, tile), _tables(flat_r, m, nl, tile), tile, tag, token_order=True
    )


class _Gate:
    """A router gate as FrontRunner uses it: ``weight`` and the projection (cuBLAS here)."""

    def __init__(self, w):
        self.weight = w

    def __call__(self, x):
        return _cublas(x, self.weight)


def _front_runner(w, nl, ls, tile, decode_kernels=False):
    front = df.DecodeFront(frozenset(), decode_kernels=decode_kernels)
    return df.FrontRunner(front, _Gate(w), nl, ls, K, lambda m: tile)


@pytest.mark.parametrize("m,nl,ls", [(1, 64, 448), (7, 64, 448), (14, 128, 128)])
def test_front_runner_tactics(m, nl, ls, tile=8):
    """FrontRunner offers the two launches (tactic 0) and the one-launch front for every front-compatible
    warp split of the low-M candidates; each gives the routed experts' inputs of its own router GEMM bitwise."""
    x, w, gs = _inputs(m, seed=1500 + m + nl)
    runner = _front_runner(w, nl, ls, tile)
    tactics = runner.get_valid_tactics([x, gs], None)
    assert tactics[0] == 0 and len(tactics) > 1, tactics
    for t in tactics:
        out = runner([x, gs], tactic=t)
        if t == 0:
            logits = _cublas(x, w)
        else:
            logits = low_m_mma_gemm(
                x16=x, w_bf16=w, out_dtype=torch.bfloat16, tactic_bf16=(t, 1, 1, 0, 1)
            )
        ref = df.RouteTables()(
            logits, K, ls, nl, tile, flat=_flat(m, nl, tile), x=x, global_scale=gs
        )
        for name, a, b in zip(
            ("logits", "xq", "xsf", "ids", "wts"), out, (logits, ref[8], ref[9], ref[0], ref[1])
        ):
            assert torch.equal(_bits(a), _bits(b)), (m, nl, t, name)


def test_front_runner_race():
    """Under autotune() the AutoTuner times both fronts at every front token count in CUDA graphs and caches
    one of the offered tactics per count, which inference-time lookups return."""
    from tensorrt_llm._torch.autotuner import AutoTuner, autotune

    nl, ls, tile = 64, 448, 8
    x, w, gs = _inputs(df.FUSED_FRONT_MAX_TOKENS, seed=1600)
    runner = _front_runner(w, nl, ls, tile)
    tuner = AutoTuner.get()
    tuner.clear_cache()
    with autotune():
        tuner.choose_one(df.FrontRunner.OP, [runner], df.FrontRunner.TUNING_CONFIG, [x, gs])
    for m in range(1, df.FUSED_FRONT_MAX_TOKENS + 1):
        _, tactic = tuner.choose_one(
            df.FrontRunner.OP, [runner], df.FrontRunner.TUNING_CONFIG, [x[:m], gs]
        )
        assert tactic in runner.get_valid_tactics([x[:m], gs], None), (m, tactic)
    tuner.clear_cache()


@pytest.mark.parametrize("nl,ls", [(32, 224), (64, 192)])
def test_front_runner_fused_only(nl, ls, tile=8):
    """A front declared with the decode kernels races only the one-launch front's warp splits wherever one takes
    the step, so the AutoTuner caches the one-launch front at every row count; the two launches stay the only
    tactic where none does (rows not 16-byte aligned). Undeclared fronts keep both."""
    from tensorrt_llm._torch.autotuner import AutoTuner, autotune

    x, w, gs = _inputs(df.FUSED_FRONT_MAX_TOKENS, seed=1650 + nl)
    runner = _front_runner(w, nl, ls, tile, decode_kernels=True)
    stock = _front_runner(w, nl, ls, tile)
    for m in range(1, df.FUSED_FRONT_MAX_TOKENS + 1):
        tactics = runner.get_valid_tactics([x[:m], gs], None)
        assert tactics and 0 not in tactics, (m, tactics)
        assert stock.get_valid_tactics([x[:m], gs], None) == [0, *tactics], m
    padded = torch.zeros((7, H + 4), dtype=x.dtype, device=x.device)[:, :H]
    assert runner.get_valid_tactics([padded, gs], None) == [0]
    tuner = AutoTuner.get()
    tuner.clear_cache()
    with autotune():
        tuner.choose_one(df.FrontRunner.OP, [runner], df.FrontRunner.TUNING_CONFIG, [x[:7], gs])
    for m in range(1, df.FUSED_FRONT_MAX_TOKENS + 1):
        _, tactic = tuner.choose_one(
            df.FrontRunner.OP, [runner], df.FrontRunner.TUNING_CONFIG, [x[:m], gs]
        )
        assert tactic > 0, (m, tactic)
    tuner.clear_cache()


# front: tables = Triton router GEMM + RouteTables, fused = FusedFront, route = cuBLAS router + RouteFront.
# tile None: every tile the runner offers at m; "min": its smallest (above 512 tokens it drops the small tiles).
# cfg None: first and last config of the tile; an int forces it (m 448 tile 16 config 0 is what the autotuner
# picks there from its 256 bucket although get_valid_configs does not list it).
@pytest.mark.parametrize(
    "front,m,nl,ls,inter,tile,cfg",
    [
        ("tables", 1, 128, 128, 512, None, None),
        ("tables", 7, 64, 448, 512, None, None),
        ("tables", 28, 128, 128, 512, None, None),
        ("tables", 14, 256, 256, 256, None, None),
        ("tables", 7, 512, 0, 128, None, None),
        ("fused", 7, 64, 448, 512, 8, None),
        ("fused", 16, 128, 0, 512, 8, None),
        ("route", 112, 128, 128, 512, 8, None),
        ("route", 224, 128, 384, 512, 16, None),
        ("route", 448, 128, 128, 512, 32, None),
        ("route", 448, 128, 128, 512, 16, 0),
        ("route", 896, 128, 128, 512, "min", None),
    ],
)
def test_prerouted_moe(front, m, nl, ls, inter, tile, cfg):
    """The MoE op with a front's (xq, xsf, tables, ids, weights) equals the stock op routing the same logits
    itself, bitwise; corrupting one CTA's expert changes the output (negative control)."""
    x, w, gs = _inputs(m, seed=700 + m + nl)
    runner, run, _ = _moe_runner(nl, ls, inter, seed=701 + m)
    by_tile = _configs(runner, m, nl, inter)
    tiles = sorted(by_tile) if tile is None else [min(by_tile)] if tile == "min" else [tile]
    for t in tiles:
        cfgs = [cfg] if cfg is not None else sorted({by_tile[t][0], by_tile[t][-1]})
        flat = _flat(m, nl, t)
        if front == "tables":
            lg, xq, xsf = df.RouterGemm()(x, w, gs)
            ids, wts = df.RouteTables()(lg, K, ls, nl, t, flat=flat)[:2]
        elif front == "fused":
            lg, xq, xsf, ids, wts = df.FusedFront()(x, w, gs, flat, t, ls, nl, K, _K_WARPS)
        else:
            lg, xq, xsf, ids, wts = _route_front(x, w, gs, flat, ls, nl, t)
        tables = _tables(flat, m, nl, t)
        refs = [run(lg, xq, xsf, [t, c])[0] for c in cfgs]
        for c, ref in zip(cfgs, refs):
            pre = run(lg, xq, xsf, [t, c], (tables, ids, wts))[0]
            assert torch.equal(_bits(ref), _bits(pre)), (front, m, nl, t, c)
        if int(tables[4].item()) > 0:
            tables[2][0] = (tables[2][0] + 1) % nl
            bad = run(lg, xq, xsf, [t, cfgs[0]], (tables, ids, wts))[0]
            assert not torch.equal(_bits(refs[0]), _bits(bad)), (front, m, nl, t, "control")


def _finalize_ref(fc2, weights, perm, hidden):
    """The trtllm-gen finalize: FP32 sum over k (in k order) of weight * FC2 row, BF16 once."""
    acc = torch.zeros((perm.shape[0], hidden), dtype=torch.float32, device=fc2.device)
    for k in range(perm.shape[1]):
        p = perm[:, k]
        rows = fc2[p.clamp(min=0).long(), :hidden].float()
        acc = torch.where((p >= 0)[:, None], acc + weights[:, k : k + 1].float() * rows, acc)
    return acc.to(torch.bfloat16)


@pytest.mark.parametrize("m,nl,ls", [(7, 64, 448), (28, 128, 128)])
def test_prerouted_unfinalized(m, nl, ls, inter=512, tile=8):
    """do_finalize=False on the pre-routed path (custom op and TRTLLMOpBackend): FC2 rows equal the stock
    unfinalized output by expanded index, and finalizing them gives the stock finalized output."""
    from tensorrt_llm._torch.moe.fused_moe.moe_op_backend import TRTLLMOpBackend
    from tensorrt_llm._torch.moe.fused_moe.routing import RoutingMethodType
    from tensorrt_llm._torch.utils import ActType_TrtllmGen

    x, w, gs = _inputs(m, seed=1100 + m + nl)
    runner, run, inputs = _moe_runner(nl, ls, inter, seed=1101 + m)
    _, run_nf, _ = _moe_runner(nl, ls, inter, seed=1101 + m, finalize=False)
    cfgs = _configs(runner, m, nl, inter)[tile]
    tactic = [tile, cfgs[len(cfgs) // 2]]
    lg, xq, xsf = df.RouterGemm()(x, w, gs)
    flat = _flat(m, nl, tile)
    ids, wts = df.RouteTables()(lg, K, ls, nl, tile, flat=flat)[:2]
    rmt, act = int(RoutingMethodType.Renormalize), ActType_TrtllmGen.SwiGlu.value
    op_args = (*inputs(lg, xq, xsf), E, K, None, None, inter, ls, nl, None, rmt)
    raw = torch.ops.trtllm.fp4_block_scale_moe_runner(
        *op_args,
        False,
        act,
        prerouted_tables=flat,
        prerouted_ids=ids,
        prerouted_weights=wts,
        prerouted_tactic=tactic,
    )
    assert len(raw) == 1, "the pre-routed path returns the FC2 rows only"
    fc2, wt, perm = TRTLLMOpBackend().run_fp4_block_scale_moe(
        *op_args,
        do_finalize=False,
        topk_weights=None,
        topk_ids=None,
        gated_act_type=act,
        prerouted=PreroutedRouting(flat, tactic, ids, wts),
    )
    ref = run_nf(lg, xq, xsf, tactic)
    fin = run(lg, xq, xsf, tactic)[0]
    assert torch.equal(_bits(wt), _bits(ref[1])), "weights"
    p_pre, p_ref = perm.view(-1).cpu(), ref[2].view(-1).cpu()
    assert torch.equal(p_pre >= 0, p_ref >= 0)
    valid = p_pre >= 0
    assert torch.equal(
        _bits(fc2[p_pre[valid].long(), :H]), _bits(ref[0][p_ref[valid].long(), :H])
    ), "FC2 rows"
    assert torch.equal(_bits(_finalize_ref(fc2, wt, perm, H)), _bits(fin[:, :H])), "finalize"


@triton.jit
def _delayed_producer_kernel(src_ptr, dst_ptr, n, spin_ns, BLOCK: tl.constexpr):
    # Releases its PDL dependents first, then spins, then writes the consumer's input.
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
    # Positive control: a PDL dependent that does not griddepcontrol.wait.
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(dst_ptr + offs, tl.load(src_ptr + offs, mask=offs < n), mask=offs < n)


def test_pdl_delayed_producer(m=7, nl=128, ls=128, inter=512, tile=8):
    """Every front's launch chain into the MoE, captured in a CUDA graph behind a producer that releases its
    dependents before writing x: each replay (intermediates zeroed) equals the eager result bitwise."""
    runner, run, _ = _moe_runner(nl, ls, inter, seed=1400)
    tactic = [tile, _configs(runner, m, nl, inter)[tile][0]]
    x_src, w, gs = _inputs(m, seed=9)
    x = torch.full_like(x_src, 3.0e4)
    y = torch.empty_like(x)
    n = x.numel()
    rg, rt, rt2, fused = df.RouterGemm(), df.RouteTables(), df.RouteTables(), df.FusedFront()
    flats = [_flat(m, nl, tile) for _ in range(3)]

    def chain():
        lg, xq, xsf = rg(x, w, gs)
        outs, bufs = [run(lg, xq, xsf, tactic)[0]], [lg, xq, xsf, *flats]
        ids, wts = rt(lg, K, ls, nl, tile, flat=flats[0])[:2]
        outs.append(run(lg, xq, xsf, tactic, (_tables(flats[0], m, nl, tile), ids, wts))[0])
        lg2 = low_m_mma_gemm(x16=x, w_bf16=w, out_dtype=torch.bfloat16, tactic_bf16=_ROUTER_TACTIC)
        o = rt2(lg2, K, ls, nl, tile, flat=flats[1], x=x, global_scale=gs)
        outs.append(run(lg2, o[8], o[9], tactic, (_tables(flats[1], m, nl, tile), o[0], o[1]))[0])
        f = fused(x, w, gs, flats[2], tile, ls, nl, K, _K_WARPS)
        outs.append(run(f[0], f[1], f[2], tactic, (_tables(flats[2], m, nl, tile), f[3], f[4]))[0])
        return outs, bufs + [ids, wts, lg2, o[0], o[1], o[8], o[9], *f]

    def producer():
        x.fill_(3.0e4)
        grid = (triton.cdiv(n, 4096),)
        _delayed_producer_kernel[grid](x_src, x, n, 50000, BLOCK=4096, launch_pdl=True)

    def control():
        producer()
        _nowait_copy_kernel[(triton.cdiv(n, 4096),)](x, y, n, BLOCK=4096, launch_pdl=True)

    def step():
        producer()
        return chain()

    x.copy_(x_src)
    ref = [t.clone() for t in chain()[0]]
    step()
    control()
    torch.cuda.synchronize()
    graph, cgraph = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outs, bufs = step()
    with torch.cuda.graph(cgraph):
        control()
    bad = stale = 0
    for _ in range(20):
        for b in bufs:
            b.zero_()
        graph.replay()
        torch.cuda.synchronize()
        bad += int(not all(torch.equal(_bits(a), _bits(b)) for a, b in zip(outs, ref)))
        cgraph.replay()
        torch.cuda.synchronize()
        stale += int(not torch.equal(y, x_src))
    assert stale > 0, "positive control never read early: the test cannot detect a missing wait"
    assert bad == 0, f"{bad}/20 replays differ from the eager result"
    assert int(fused.stage.abs().sum().item()) == 0, "staging buffer not clear"


def test_decode_front_keeps_token_count_dynamic_under_compile():
    """Under torch.compile the front declines without specializing the token dimension (piecewise CUDA
    graphs need it symbolic)."""
    gate = type("Gate", (), {"out_dtype": torch.bfloat16})()

    def routed(x):
        out = df.DecodeFront(frozenset({70}))(gate, object(), x)
        return x * 2 if out is None else out

    x = torch.randn(70, 16, dtype=torch.bfloat16, device="cuda")
    torch._dynamo.reset()
    torch._dynamo.mark_dynamic(x, 0)
    out = torch.compile(routed, backend="eager", fullgraph=True)(x)
    torch.testing.assert_close(out, x * 2)


class _RecordingMoe:
    """The routed experts as ``DecodeFront`` calls them (CPU, no kernels), recording into ``calls``."""

    def __init__(self, calls: list[str]) -> None:
        self.calls = calls
        self.backend = type(
            "Backend",
            (),
            dict(
                decode_tactic=lambda _, m, do_finalize: [8, 0],
                slot_start=0,
                expert_size_per_partition=E,
                fc31_input_scale=torch.ones(1),
            ),
        )()
        self.routing_method = type("Routing", (), dict(experts_per_token=K))()

    def __call__(
        self,
        x: object,
        logits: torch.Tensor,
        output_dtype: torch.dtype,
        prerouted: PreroutedRouting | None = None,
        **kwargs: object,
    ) -> str:
        self.calls.append("moe")
        return "front"


# H 4096 / 2048: Qwen3.5-397B / Qwen3-Next-80B; H 2560: Qwen3.8-Flash-Next, whose MTP-3 step at CUDA-graph batch
# 8 (M 32) takes the stock path and at batch 16 (M 64) the route front.
# mode: "router" = no low-M dispatcher (RouterGemm), "dispatcher" = a low-M dispatcher attached (gate + RouteTables
# with x), "tune" = under autotune() (the stock router and MoE of the tuning step).
@pytest.mark.parametrize(
    "h,m,route_steps,mode,path",
    [
        (4096, 32, (), "router", ["router_gemm", "route", "moe"]),
        (2048, 8, (), "router", ["router_gemm", "route", "moe"]),
        (2560, 32, (), "router", None),
        (2560, 1, (), "router", None),
        (2560, 64, (64,), "router", ["gate", "route_front", "moe"]),
        (4096, 32, (), "dispatcher", ["gate", "route", "moe"]),
        (2560, 32, (), "dispatcher", None),
        (4096, 32, (), "tune", ["gate", "moe"]),
        (2560, 32, (), "tune", None),
    ],
)
def test_decode_front_hidden_size(
    monkeypatch: pytest.MonkeyPatch,
    h: int,
    m: int,
    route_steps: tuple[int, ...],
    mode: str,
    path: list[str] | None,
) -> None:
    """Steps of up to ROUTER_MAX_TOKENS tokens take the front only with a power-of-two hidden size (its Triton
    launches span whole rows) and otherwise decline before any launch, with or without a low-M dispatcher and
    under autotuning; the route front takes other sizes."""
    from tensorrt_llm._torch.autotuner import AutoTuner, autotune

    calls: list[str] = []
    xq = torch.zeros((m, h // 2), dtype=torch.uint8)
    xsf = torch.zeros((m, h // 16), dtype=torch.uint8)
    ids, wts = torch.zeros((m, K), dtype=torch.int32), torch.zeros((m, K), dtype=torch.bfloat16)

    class Gate:
        out_dtype = torch.bfloat16
        weight = torch.zeros((E, h), dtype=torch.bfloat16)

        def __call__(self, x: torch.Tensor) -> torch.Tensor:
            calls.append("gate")
            return torch.zeros((x.shape[0], E), dtype=torch.bfloat16)

    class RouteFront:
        def compiled(self, x: torch.Tensor, n: int, tile: int, num_local: int, top_k: int) -> str:
            return "compiled"

        def __call__(
            self,
            fn: str,
            logits: torch.Tensor,
            x: torch.Tensor,
            gs: torch.Tensor,
            flat: torch.Tensor,
            tile: int,
            local_start: int,
            num_local: int,
            top_k: int,
        ) -> tuple[torch.Tensor, ...]:
            calls.append("route_front")
            return xq, xsf, ids, wts

    def router_gemm(
        x: torch.Tensor, weight: torch.Tensor, global_scale: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        calls.append("router_gemm")
        return torch.zeros((x.shape[0], E), dtype=torch.bfloat16), xq, xsf

    def route(
        logits: torch.Tensor,
        top_k: int,
        local_start: int,
        num_local: int,
        tile: int,
        flat: torch.Tensor | None = None,
        x: torch.Tensor | None = None,
        global_scale: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, ...]:
        calls.append("route")
        return (ids, wts) if x is None else (ids, wts, xq, xsf)

    dispatcher = object() if mode == "dispatcher" else None
    monkeypatch.setattr(df, "front_eligible", lambda moe, x, max_tokens: True)
    monkeypatch.setattr(df, "low_m_gemm_dispatcher", lambda gate: dispatcher)
    front = df.DecodeFront(frozenset(route_steps))
    front.router_gemm, front.route, front.route_front = router_gemm, route, RouteFront()
    x = torch.zeros((m, h), dtype=torch.bfloat16)
    if mode == "tune":
        AutoTuner.get().clear_cache()
        with autotune():
            out = front(Gate(), _RecordingMoe(calls), x)
        AutoTuner.get().clear_cache()
    else:
        out = front(Gate(), _RecordingMoe(calls), x)
    if path is None:
        assert out is None and calls == [], calls
    else:
        assert out == "front" and calls == path, calls


def _engine_backend(
    decode_tuning: bool = True, engine_state: bool = True, num_local: int = 128, inter: int = 512
) -> SimpleNamespace:
    """An NVFP4 TRTLLM-gen backend as ``decode_moe.engine_routes`` reads it (meta weights)."""
    from tensorrt_llm._torch.utils import ActivationType

    def meta(*shape, dtype=torch.uint8):
        return torch.empty(shape, dtype=dtype, device="meta")

    return SimpleNamespace(
        decode_tactic=lambda m, do_finalize: [8, 0],
        slot_start=0,
        expert_size_per_partition=num_local,
        fc31_input_scale=torch.ones(1),
        decode_tuning=decode_tuning,
        decode_engine_state=torch.zeros(1, dtype=torch.int32) if engine_state else None,
        activation_type=ActivationType.Swiglu,
        bias=False,
        w3_w1_bias=None,
        w2_bias=None,
        act_alpha=None,
        act_beta=None,
        act_clamp=None,
        w3_w1_weight=meta(num_local, 2 * inter, H // 2),
        w2_weight=meta(num_local, H, inter // 2),
        fc31_scale_c=meta(num_local, dtype=torch.float32),
        fc31_alpha=meta(num_local, dtype=torch.float32),
        fc2_alpha=meta(num_local, dtype=torch.float32),
    )


_TODAY = ["router_gemm", "route_front", "moe"]


@pytest.mark.parametrize(
    "top_k,num_experts,routes",
    [
        (10, 512, True),
        (8, 512, True),
        (9, 512, True),
        (6, 1024, True),
        (13, 512, False),  # 64 tokens x 52 keys: more than two 16-byte loads per engine thread
        (7, 128, False),  # 7 keys of one list: not whole 16-byte vectors
        (10, 1024, False),  # 80 keys per token: more than two loads per thread
        (10, 256, False),  # two lists: the router epilogue's 16-token tile does not exist
        (33, 512, False),  # the router epilogue stores at most 32 keys per tile
    ],
)
def test_engine_routes_top_k_and_lists(top_k: int, num_experts: int, routes: bool) -> None:
    """The routing engine and its router epilogue take top-k and the number of 128-expert lists as compile-time
    parameters; ``engine_routes`` declines what either kernel cannot run."""
    from tensorrt_llm._torch.moe.fused_moe.trtllm_gen import decode_moe as dm

    assert dm.engine_routes(_engine_backend(), 35, top_k, num_experts) == routes
    assert not dm.engine_routes(_engine_backend(), 65, top_k, num_experts)


@pytest.mark.parametrize(
    "engine_route,finalize,backend_kw,path",
    [
        (True, False, {}, ["engine_router", "moe"]),
        (True, True, {}, _TODAY),
        (False, False, {}, _TODAY),
        (True, False, dict(engine_state=False), _TODAY),
        (True, False, dict(decode_tuning=False), _TODAY),
        (True, False, dict(num_local=256), _TODAY),
    ],
    ids=["engine", "finalized", "not_declared", "no_engine_state", "no_decode_tuning", "256_local"],
)
def test_decode_front_engine_route_gate(
    monkeypatch: pytest.MonkeyPatch,
    engine_route: bool,
    finalize: bool,
    backend_kw: dict,
    path: list[str],
) -> None:
    """A step of 35 rows routes in the MoE engine (the epilogue router GEMM, then the MoE with the candidates and no
    route launch) only where the front declares ``engine_route`` and the call is un-finalized and runs the engine;
    otherwise the route front steps' CuTe router GEMM and the route front, without candidates."""
    m = 35
    calls: list[str] = []
    seen: list[PreroutedRouting] = []
    xq = torch.zeros((m, H // 2), dtype=torch.uint8)
    xsf = torch.zeros((m, H // 16), dtype=torch.uint8)
    ids, wts = torch.zeros((m, K), dtype=torch.int32), torch.zeros((m, K), dtype=torch.bfloat16)
    logits = torch.zeros((m, E), dtype=torch.bfloat16)
    cand = torch.zeros((m, E // 128, K), dtype=torch.int32)

    class Gate:
        out_dtype = torch.bfloat16
        weight = torch.zeros((E, H), dtype=torch.bfloat16)

        def __call__(self, x: torch.Tensor) -> torch.Tensor:
            calls.append("gate")
            return logits

    class RouteFront:
        def compiled(self, x, n, tile, num_local, top_k):
            return "compiled"

        def __call__(self, fn, lg, x, gs, flat, tile, local_start, num_local, top_k):
            calls.append("route_front")
            return xq, xsf, ids, wts

    def launch(name: str, out):
        def run(*args, **kwargs):
            calls.append(name)
            return out

        return run

    class Moe:
        backend = _engine_backend(**backend_kw)
        routing_method = SimpleNamespace(experts_per_token=K)

        def __call__(self, x, lg, output_dtype, prerouted=None, **kwargs) -> str:
            calls.append("moe")
            seen.append(prerouted)
            return "front"

    monkeypatch.setattr(df, "front_eligible", lambda moe, x, max_tokens: True)
    front = df.DecodeFront(frozenset({35, 42}), decode_kernels=True, engine_route=engine_route)
    front.route_front = RouteFront()
    front.route_front_router = launch("router_gemm", logits)
    if engine_route:
        front.engine_router = launch("engine_router", (logits, cand, xq, xsf))
    x = torch.zeros((m, H), dtype=torch.bfloat16)
    out = front(Gate(), Moe(), x, do_finalize=finalize)
    assert out == "front" and calls == path, calls
    engine = path[0] == "engine_router"
    # The engine's hand-off is an EngineRouting carrying the candidates; otherwise a plain PreroutedRouting.
    assert isinstance(seen[0], EngineRouting) == engine
    assert not engine or seen[0].candidates is cand


def test_engine_route_decision_is_kept_per_token_count(monkeypatch: pytest.MonkeyPatch) -> None:
    """``DecodeFront._engine_routes`` evaluates ``decode_moe.engine_routes`` once per token count after the backend's
    engine state exists (everything else it reads is fixed per layer), and every call before that."""
    from tensorrt_llm._torch.moe.fused_moe.trtllm_gen import decode_moe as dm

    evaluated: list[int] = []
    real = dm.engine_routes

    def counted(backend, m, top_k, num_experts):
        evaluated.append(m)
        return real(backend, m, top_k, num_experts)

    monkeypatch.setattr(dm, "engine_routes", counted)
    front = df.DecodeFront(frozenset({35, 42}), decode_kernels=True, engine_route=True)
    front.engine_router = object()
    unfinalized = {"do_finalize": False}
    backend = _engine_backend(engine_state=False)
    assert not front._engine_routes(backend, 35, K, E, unfinalized)
    assert not front._engine_routes(backend, 35, K, E, unfinalized)
    assert evaluated == [35, 35]
    backend.decode_engine_state = torch.zeros(1, dtype=torch.int32)
    for _ in range(3):
        assert front._engine_routes(backend, 35, K, E, unfinalized)
        assert not front._engine_routes(backend, 65, K, E, unfinalized)
    assert not front._engine_routes(backend, 35, K, E, {"do_finalize": True})
    assert evaluated == [35, 35, 35, 65]
