# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The Qwen3.5 MoE front above 32 tokens (kernels/blackwell/q35_moe_front_big.py) after the stock router GEMM:
ids / weights / xq / xsf bitwise vs the Triton route kernel and ``fp4_quantize``, tables equivalent up to the row
order inside an expert's block, and the pre-routed trtllm-gen MoE output vs stock routing; 257..1024 tokens on
clusters of 16 (TRTLLM_Q35_MOE_FRONT_BIG_C16 / _C16W). Needs SM100/SM103.
"""

import pytest
import torch
import triton

from tensorrt_llm._torch.moe.fused_moe import q35_fused_front as qf

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="needs SM100 / SM103",
)

E, K, H = 512, 10, 4096


def _inputs(m, seed, dup=False, crowd=False):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = (torch.randn((m, H), generator=g, device="cuda") * 1.5).to(torch.bfloat16)
    x[0, :32] = 0  # all-zero quant blocks
    x[1, 100] = 400.0  # an outlier block
    w = (torch.randn((E, H), generator=g, device="cuda") * 0.02).to(torch.bfloat16)
    if dup:
        w[1::2] = w[0::2]  # pairs of identical router rows: exact logit ties in every token's top-k
    if crowd:
        # Experts 16..27 (one routing lane) dominate the even tokens: >= 5 of their top 10 in one lane takes the
        # full-sort fallback.
        base = x[0::2].float().mean(0)
        base = base / base.norm()
        for i in range(12):
            w[16 + i] = (
                base * (3.0 - 0.1 * i) + 0.01 * torch.randn(H, generator=g, device="cuda")
            ).to(torch.bfloat16)
    gs = torch.tensor(
        [448.0 * 6.0 / x.float().abs().max().item()], dtype=torch.float32, device="cuda"
    )
    return x, w, gs


def _sizes(m, nl, tile):
    return sum(qf.q35_prerouted_sizes(m, K, nl, tile))


def _cublas(x, w):
    return torch.ops.trtllm.cublas_mm(x, w.t(), bias=None, out_dtype=torch.bfloat16)


def _ref_route(logits, x, gs, ls, nl, tile):
    """The Triton per-token route + tables + quant launch."""
    m = logits.shape[0]
    nlp = triton.next_power_of_2(nl)
    hist = torch.zeros((nlp,), dtype=torch.int32, device="cuda")
    counter = torch.zeros((1,), dtype=torch.int32, device="cuda")
    ids = torch.empty((m, K), dtype=torch.int32, device="cuda")
    wts = torch.empty((m, K), dtype=torch.bfloat16, device="cuda")
    rank = torch.empty((m * K,), dtype=torch.int32, device="cuda")
    flat = torch.full((_sizes(m, nl, tile),), -7, dtype=torch.int32, device="cuda")
    e2p, p2t, cte, ctl, nct, npd = qf.q35_prerouted_views(flat, m, K, nl, tile)
    xq = torch.empty((m, H // 2), dtype=torch.uint8, device="cuda")
    xsf = torch.empty((m, H // 16), dtype=torch.uint8, device="cuda")
    qf._q35_route_tables_kernel[(2 * m,)](
        logits,
        ids,
        wts,
        hist,
        rank,
        counter,
        e2p,
        p2t,
        cte,
        ctl,
        nct,
        npd,
        m,
        ls,
        x,
        x.stride(0),
        gs,
        xq,
        xsf,
        E=E,
        K=K,
        KP=16,
        TILE=tile,
        NL=nl,
        NLP=nlp,
        EXPANDED_PAD=triton.next_power_of_2(m * K),
        MAX_CTAS_PER_EXPERT=triton.cdiv(m, tile),
        USE_PDL=False,
        QUANT_H=H,
        num_warps=4,
    )
    return ids, wts, flat, xq, xsf


def _check_tables(flat, flat_r, m, nl, tile, tag):
    e2p, p2t, cte, ctl, nct, npd = [t.cpu() for t in qf.q35_prerouted_views(flat, m, K, nl, tile)]
    r_e2p, r_p2t, r_cte, r_ctl, r_nct, r_npd = [
        t.cpu() for t in qf.q35_prerouted_views(flat_r, m, K, nl, tile)
    ]
    n = int(r_nct.item())
    assert int(nct.item()) == n and int(npd.item()) == int(r_npd.item()), (tag, "totals")
    assert torch.equal(cte[:n], r_cte[:n]) and torch.equal(ctl[:n], r_ctl[:n]), (tag, "CTA tables")
    assert torch.equal(e2p < 0, r_e2p < 0) and bool(torch.all(e2p[e2p < 0] == -1)), (
        tag,
        "remote picks",
    )
    loc = r_e2p >= 0
    ra, rb = e2p[loc].long(), r_e2p[loc].long()
    assert torch.equal(torch.sort(ra).values, torch.sort(rb).values), (tag, "row sets")
    assert torch.equal(cte.long()[ra // tile], cte.long()[rb // tile]), (
        tag,
        "expert block of a pick",
    )
    assert torch.equal(p2t[ra].long(), torch.arange(m * K)[loc] // K), (
        tag,
        "p2t[e2p[i]] == token(i)",
    )
    for c in range(n):  # token order inside a block
        rows = p2t[c * tile : int(ctl[c].item())].tolist()
        assert rows == sorted(rows), (tag, "block rows not in token order", c)


def _front(m, nl, tile):
    """fn(x, w, gs, flat, ls) -> (logits, xq, xsf, ids, wts) of the big front (stock router GEMM + route launch)."""
    front = qf.Q35RouteBig()

    def run(x, w, gs, flat, ls):
        fn = front.compiled(x, E, gs, tile, nl, K)
        assert fn is not None, "route front does not apply"
        logits = _cublas(x, w)
        return (logits,) + front(fn, logits, x, gs, flat, tile, ls, nl, K)

    return run


# (m, num_local, local_start, tile, dup, crowd): M = 40 (odd split over the 8 routing CTAs) / 112, 224 (MTP-6
# verify at batch 16 / 32) / 256 (max); num_local 64 / 128 = Qwen3.5-397B at TP8 / TP4.
@pytest.mark.parametrize(
    "m,nl,ls,tile,dup,crowd",
    [
        (40, 128, 0, 8, False, False),
        (112, 64, 448, 16, False, False),
        (224, 128, 384, 32, False, False),
        (256, 256, 256, 8, False, False),
        (112, 128, 128, 8, True, False),
        (224, 128, 0, 8, False, True),
    ],
)
def test_parity(m, nl, ls, tile, dup, crowd):
    x, w, gs = _inputs(m, seed=100 + m + nl + tile + (7 if dup else 0), dup=dup, crowd=crowd)
    run = _front(m, nl, tile)
    flat = torch.full((_sizes(m, nl, tile),), -7, dtype=torch.int32, device="cuda")
    lg, xq, xsf, ids, wts = run(x, w, gs, flat, ls)
    lg_ref = _cublas(x, w)
    ids_r, wts_r, flat_r, xq_r, xsf_r = _ref_route(lg_ref, x, gs, ls, nl, tile)
    q_ref, sf_ref = torch.ops.trtllm.fp4_quantize(x, gs, 16, False, False)
    torch.cuda.synchronize()
    tag = (m, nl, ls, tile, dup)
    assert torch.equal(lg.view(torch.int16), lg_ref.view(torch.int16)), (tag, "logits")
    assert torch.equal(ids, ids_r), (tag, "ids")
    assert torch.equal(wts.view(torch.int16), wts_r.view(torch.int16)), (tag, "weights")
    assert torch.equal(xq, q_ref.view(torch.uint8).view(m, H // 2)), (tag, "xq vs fp4_quantize")
    assert torch.equal(xsf.flatten(), sf_ref.view(torch.uint8).flatten()[: m * H // 16]), (
        tag,
        "xsf",
    )
    assert torch.equal(xq, xq_r) and torch.equal(xsf, xsf_r), (tag, "quant vs the route kernel's")
    _check_tables(flat, flat_r, m, nl, tile, tag)
    if crowd:
        lane1 = ((ids_r.cpu() >= 16) & (ids_r.cpu() < 32)).sum(-1)
        n_crowded = int((lane1 >= 5).sum().item())
        assert n_crowded > 0, (tag, "the crowd case does not reach the fallback")
    if dup:
        # Rows 2i / 2i + 1 tie exactly: a picked odd expert implies its even twin was picked.
        sel = ids.cpu()
        for row in sel.tolist():
            for e in row:
                assert e % 2 == 0 or (e - 1) in row, (tag, "tie went to the larger id", row)
    flat2 = torch.full_like(flat, -7)
    lg2, xq2, xsf2, ids2, wts2 = run(x, w, gs, flat2, ls)
    torch.cuda.synchronize()
    assert (
        torch.equal(lg2.view(torch.int16), lg.view(torch.int16))
        and torch.equal(ids2, ids)
        and torch.equal(wts2.view(torch.int16), wts.view(torch.int16))
        and torch.equal(xq2, xq)
        and torch.equal(xsf2, xsf)
    ), (tag, "repeat")
    _check_tables(flat2, flat, m, nl, tile, (tag, "repeat"))


def _rand_fp4_moe(nl, inter, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)

    def u8(*s, lo=0, hi=256):
        return torch.randint(lo, hi, s, generator=g, device="cuda", dtype=torch.int32).to(  # noqa
            torch.uint8
        )

    w1, s1 = u8(nl, 2 * inter, H // 2), u8(nl, 2 * inter, H // 16, lo=0x28, hi=0x38)
    w2, s2 = u8(nl, H, inter // 2), u8(nl, H, inter // 16, lo=0x28, hi=0x38)
    a1 = torch.rand((nl,), generator=g, device="cuda") * 1e-2 + 1e-3
    a2 = torch.rand((nl,), generator=g, device="cuda") * 1e-2 + 1e-3
    return w1, s1, a1, w2, s2, a2


@pytest.mark.parametrize(
    "m,nl,ls,inter,tile",
    [(112, 128, 128, 512, 8), (224, 128, 384, 512, 16)],
)
def test_moe_parity(m, nl, ls, inter, tile, force_cfgs=None):
    """Pre-routed trtllm-gen MoE output with the front's (xq, xsf, tables, ids, weights) == the stock op routing the
    same logits itself (bitwise). force_cfgs: run these configs of ``tile`` even when get_valid_configs does not list
    them at m (the autotuner tunes 257..511 tokens in its 256 bucket, e.g. tactic (16, 0) at m = 448)."""
    from tensorrt_llm._torch.custom_ops.trtllm_gen_custom_ops import FP4BlockScaleMoERunner
    from tensorrt_llm._torch.moe.fused_moe.routing import RoutingMethodType
    from tensorrt_llm._torch.utils import ActType_TrtllmGen

    if not FP4BlockScaleMoERunner.supports_prerouted():
        pytest.skip("libth_common without run_moe_prerouted")
    x, w, gs = _inputs(m, seed=900 + m + nl)
    w1, s1, a1, w2, s2, a2 = _rand_fp4_moe(nl, inter, seed=901 + m)
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

    def run_moe(logits, xq, xsf, cfg, pre=None):
        inp = [
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
        if pre is None:
            return runner(inp, tactic=[tile, cfg])[0]
        return runner(inp, tactic=[tile, cfg], prerouted=pre, prerouted_tile=tile)[0]

    cfgs = sorted(
        int(c)
        for t, c in runner.get_runner().get_valid_configs(K, H, inter, nl, m)
        if int(t) == tile
    )
    if force_cfgs is not None:
        cfgs = list(force_cfgs)
    assert cfgs, f"no trtllm-gen config of tile {tile} at m={m}"
    run = _front(m, nl, tile)
    flat = torch.full((_sizes(m, nl, tile),), -7, dtype=torch.int32, device="cuda")
    lg, xq, xsf, ids, wts = run(x, w, gs, flat, ls)
    q_ref, sf_ref = torch.ops.trtllm.fp4_quantize(x, gs, 16, False, False)
    cfg = cfgs[0]
    out_f = run_moe(lg, xq, xsf, cfg, (qf.q35_prerouted_views(flat, m, K, nl, tile), ids, wts))
    out_s = run_moe(
        lg,
        q_ref.view(torch.uint8).view(m, H // 2),
        sf_ref.view(torch.uint8).view(-1)[: m * H // 16].view(m, H // 16),
        cfg,
    )
    torch.cuda.synchronize()
    assert torch.equal(out_f.view(torch.int16), out_s.view(torch.int16)), (
        (m, nl, ls, inter, tile, cfg),
        "pre-routed vs stock routing",
    )


@pytest.fixture
def route_max_1024(monkeypatch):
    """Route launches up to 1024 tokens on clusters of 16 (what TRTLLM_Q35_MOE_FRONT_BIG_C16 / _C16W enable)."""
    monkeypatch.setattr(qf, "Q35_MOE_FRONT_BIG_ROUTE_MAX", 1024)


# 257..512 tokens (MTP-6 verify of b37-b73, C16) and 513..1024 tokens (b74-b146, C16W: two-word token masks) on
# clusters of 16.
@pytest.mark.parametrize(
    "m,nl,ls,tile,dup,crowd",
    [
        (m, 128, 128, tile, False, False)
        for m in (257, 280, 336, 392, 448, 512)
        for tile in (8, 16, 32)
    ]
    + [(448, nl, ls, 16, False, False) for nl, ls in ((128, 0), (256, 256), (64, 448))]
    + [(448, 128, 0, 16, True, False), (448, 128, 0, 16, False, True)]
    + [(m, 128, 0, 16, False, False) for m in (513, 560, 672, 784, 896, 1024)]
    + [(896, nl, ls, 32, False, False) for nl, ls in ((128, 384), (256, 256), (64, 448))]
    + [(896, 128, 0, 16, True, False), (896, 128, 0, 16, False, True)],
)
def test_parity_c16(m, nl, ls, tile, dup, crowd, route_max_1024):
    test_parity(m, nl, ls, tile, dup, crowd)


# m 448 tile 16 config 0: the tactic the autotuner picks there from its 256 bucket
@pytest.mark.parametrize("m,tile,force_cfgs", [(336, 16, None), (448, 32, None), (448, 16, [0])])
def test_moe_parity_c16(m, tile, force_cfgs, route_max_1024):
    test_moe_parity(m, 128, 128, 512, tile, force_cfgs=force_cfgs)


@pytest.mark.parametrize("m", [560, 672, 784, 896])
def test_moe_parity_c16w(m, route_max_1024):
    """test_moe_parity at the two smallest tiles the trtllm-gen runner offers at m (above 512 tokens it drops the small
    tiles, so a fixed tile can have no config there)."""
    from tensorrt_llm._torch.custom_ops.trtllm_gen_custom_ops import FP4BlockScaleMoERunner
    from tensorrt_llm._torch.moe.fused_moe.routing import RoutingMethodType
    from tensorrt_llm._torch.utils import ActType_TrtllmGen

    if not FP4BlockScaleMoERunner.supports_prerouted():
        pytest.skip("libth_common without run_moe_prerouted")
    runner = FP4BlockScaleMoERunner(
        E,
        K,
        None,
        None,
        512,
        128,
        128,
        None,
        int(RoutingMethodType.Renormalize),
        True,
        ActType_TrtllmGen.SwiGlu.value,
    )
    tiles = sorted({int(t) for t, _ in runner.get_runner().get_valid_configs(K, H, 512, 128, m)})
    assert tiles, f"no trtllm-gen config at m={m}"
    for tile in tiles[:2]:
        test_moe_parity(m, 128, 128, 512, tile)
