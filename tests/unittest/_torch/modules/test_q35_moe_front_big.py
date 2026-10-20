# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TRTLLM_Q35_MOE_FRONT_BIG: the MoE front above 32 tokens (cute_dsl_kernels/blackwell/q35_moe_front_big.py) against
the unfused path at the Qwen3.5 TEP8 MTP-6 verify shapes (112 / 224 tokens, 128 local experts).

* route mode (=1): routing + pre-routed tables + NVFP4 x in one launch after the stock router GEMM. Top-k ids and
  BF16 weights bitwise those of the per-token route + tables kernel (the two-launch front's Triton kernel) on the
  same cuBLAS logits; tables: CTA -> expert / row limit / totals bitwise, expanded -> permuted and permuted -> token
  up to the row order inside an expert's block (token order here); xq / xsf bitwise trtllm::fp4_quantize.
* gemm mode (=gemm): the router GEMM in the same launch. Logits bitwise those of the low-M BF16 router GEMM
  (LowMMmaGemmKernel, tactic (16, 1)) run on 56-row chunks (rows are independent); the rest as above. The
  per-launch state (ticket, token masks) returns to zero, the epoch advances by one.
* both: exact logit ties (duplicated router rows) go to the smaller expert id; 100 back-to-back launches are
  bitwise identical; the pre-routed trtllm-gen MoE output with the front's (xq, xsf, tables, ids, weights) equals
  the stock op routing the same logits itself (bitwise).

Run directly: python3 test_q35_moe_front_big.py. Needs SM100/SM103 (and a libth_common with run_moe_prerouted for
the MoE parity).
"""

import sys
import time
import traceback

import torch
import triton

from tensorrt_llm._torch.modules.fused_moe import q35_fused_front as qf
from tensorrt_llm._torch.modules.lowm_linear import lowm_bf16_gemm

E, K, H = 512, 10, 4096
RESULTS = []


def _inputs(m, seed, dup=False, crowd=False):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = (torch.randn((m, H), generator=g, device="cuda") * 1.5).to(torch.bfloat16)
    x[0, :32] = 0  # all-zero quant blocks
    x[1, 100] = 400.0  # an outlier block
    w = (torch.randn((E, H), generator=g, device="cuda") * 0.02).to(torch.bfloat16)
    if dup:
        w[1::2] = w[0::2]  # pairs of identical router rows: exact logit ties in every token's top-k
    if crowd:
        # experts 16..27 (one lane's 16 logits in the routing warp) dominate the even tokens: >= 5 of their top 10
        # in one lane (the route kernel's full-sort fallback)
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


def _lowm(x, w):
    return torch.cat(
        [
            lowm_bf16_gemm(x[i : i + 56], w, out_dtype=torch.bfloat16, tactic=(16, 1))
            for i in range(0, x.shape[0], 56)
        ],
        0,
    )


def _ref_route(logits, x, gs, ls, nl, tile):
    """The two-launch front's per-token route + tables + quant launch (Triton), sized for m tokens."""
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
        1,
        0,
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
        DEA=False,
        USE_PDL=False,
        EARLY_TRIGGER=False,
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


def _front(mode, m, nl, tile):
    """fn(x, w, gs, flat, ls) -> (logits, xq, xsf, ids, wts) of the big front in ``mode``."""
    if mode == "gemm":
        front = qf.Q35FusedFrontBig()
        return front, lambda x, w, gs, flat, ls: front(x, w, gs, flat, tile, ls, nl, K)
    front = qf.Q35RouteBig()

    def run(x, w, gs, flat, ls):
        fn = front.compiled(x, E, gs, tile, nl, K)
        assert fn is not None, "route front does not apply"
        logits = _cublas(x, w)
        return (logits,) + front(fn, logits, x, gs, flat, tile, ls, nl, K)

    return front, run


def test_parity(mode, m, nl, ls, tile, dup=False, repeats=4, crowd=False):
    x, w, gs = _inputs(m, seed=100 + m + nl + tile + (7 if dup else 0), dup=dup, crowd=crowd)
    front, run = _front(mode, m, nl, tile)
    flat = torch.full((_sizes(m, nl, tile),), -7, dtype=torch.int32, device="cuda")
    lg, xq, xsf, ids, wts = run(x, w, gs, flat, ls)
    lg_ref = _lowm(x, w) if mode == "gemm" else _cublas(x, w)
    ids_r, wts_r, flat_r, xq_r, xsf_r = _ref_route(lg_ref, x, gs, ls, nl, tile)
    q_ref, sf_ref = torch.ops.trtllm.fp4_quantize(x, gs, 16, False, False)
    torch.cuda.synchronize()
    tag = (mode, m, nl, ls, tile, dup)
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
        print(
            f"  [crowd] {n_crowded} / {m} tokens with >= 5 of their top 10 in one routing lane",
            flush=True,
        )
        assert n_crowded > 0, (tag, "the crowd case does not reach the fallback")
    if dup:
        # duplicated rows 2i / 2i + 1 tie exactly: a picked odd expert implies its even twin was picked first
        sel = ids.cpu()
        for row in sel.tolist():
            for e in row:
                assert e % 2 == 0 or (e - 1) in row, (tag, "tie went to the larger id", row)
    if mode == "gemm":
        st = front.state
        from tensorrt_llm._torch.cute_dsl_kernels.blackwell.q35_moe_front_big import (
            STATE_EPOCH,
            STATE_MASKS,
            STATE_TICKET,
        )

        assert int(st[STATE_TICKET].item()) == 0 and not bool(st[STATE_MASKS:].any()), (
            tag,
            "state not reset",
        )
        assert int(st[STATE_EPOCH].item()) == 1, (tag, "epoch")
    for it in range(repeats):
        flat2 = torch.full_like(flat, -7)
        lg2, xq2, xsf2, ids2, wts2 = run(x, w, gs, flat2, ls)
        torch.cuda.synchronize()
        assert (
            torch.equal(lg2.view(torch.int16), lg.view(torch.int16))
            and torch.equal(ids2, ids)
            and torch.equal(wts2.view(torch.int16), wts.view(torch.int16))
            and torch.equal(xq2, xq)
            and torch.equal(xsf2, xsf)
        ), (tag, "repeat", it)
        _check_tables(flat2, flat, m, nl, tile, (tag, "repeat", it))


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


def test_moe_parity(mode, m, nl, ls, inter, tile):
    """Pre-routed trtllm-gen MoE output with the front's (xq, xsf, tables, ids, weights) == the stock op routing the
    same logits itself (bitwise)."""
    from tensorrt_llm._torch.custom_ops.trtllm_gen_custom_ops import FP4BlockScaleMoERunner
    from tensorrt_llm._torch.modules.fused_moe.routing import RoutingMethodType
    from tensorrt_llm._torch.utils import ActType_TrtllmGen

    if not FP4BlockScaleMoERunner.supports_prerouted():
        print("[moe parity] SKIP: libth_common without run_moe_prerouted", flush=True)
        return
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
    assert cfgs, f"no trtllm-gen config of tile {tile} at m={m}"
    _, run = _front(mode, m, nl, tile)
    flat = torch.full((_sizes(m, nl, tile),), -7, dtype=torch.int32, device="cuda")
    lg, xq, xsf, ids, wts = run(x, w, gs, flat, ls)
    q_ref, sf_ref = torch.ops.trtllm.fp4_quantize(x, gs, 16, False, False)
    for cfg in cfgs[:2]:
        out_f = run_moe(lg, xq, xsf, cfg, (qf.q35_prerouted_views(flat, m, K, nl, tile), ids, wts))
        out_s = run_moe(
            lg,
            q_ref.view(torch.uint8).view(m, H // 2),
            sf_ref.view(torch.uint8).view(-1)[: m * H // 16].view(m, H // 16),
            cfg,
        )
        torch.cuda.synchronize()
        assert torch.equal(out_f.view(torch.int16), out_s.view(torch.int16)), (
            (mode, m, nl, ls, inter, tile, cfg),
            "pre-routed vs stock routing",
        )


def _run(name, fn, *args, **kw):
    t0 = time.time()
    try:
        fn(*args, **kw)
        RESULTS.append((name, True, ""))
        return True
    except Exception as e:  # noqa: BLE001
        msg = f"{e!r}"[:1500]
        RESULTS.append((name, False, msg))
        print(f"FAIL {name}: {msg}\n{traceback.format_exc()[-3000:]}", flush=True)
        return False
    finally:
        print(f"[{name}] {time.time() - t0:.1f}s", flush=True)


def main():
    torch.cuda.init()
    props = torch.cuda.get_device_properties(0)
    print(
        f"device {props.name} sm{props.major}{props.minor} sms={props.multi_processor_count}",
        flush=True,
    )
    modes = ("route", "gemm") if "--route-only" not in sys.argv else ("route",)
    for mode in modes:
        for m in (112, 224):
            for tile in (8, 16, 32):
                for nl, ls in ((128, 0), (128, 384), (256, 256), (64, 448)):
                    _run(
                        f"parity {mode} m{m} tile{tile} nl{nl} ls{ls}",
                        test_parity,
                        mode,
                        m,
                        nl,
                        ls,
                        tile,
                    )
            for nl, ls in ((128, 128), (64, 0)):
                _run(
                    f"parity dup {mode} m{m} nl{nl} ls{ls}",
                    test_parity,
                    mode,
                    m,
                    nl,
                    ls,
                    8,
                    dup=True,
                )
        for m in (112, 224):
            _run(f"parity crowd {mode} m{m}", test_parity, mode, m, 128, 0, 8, crowd=True)
        if mode == "route":
            for m in (40, 77, 128, 256):  # other token counts (the route kernel takes any M <= 256)
                _run(f"parity route m{m}", test_parity, mode, m, 128, 128, 8)
        for m, nl, ls, inter, tile in (
            (112, 128, 128, 512, 8),
            (224, 128, 384, 512, 16),
            (112, 64, 448, 512, 16),
            (224, 256, 0, 256, 16),
        ):
            _run(
                f"moe parity {mode} m{m} nl{nl} tile{tile}",
                test_moe_parity,
                mode,
                m,
                nl,
                ls,
                inter,
                tile,
            )
    failed = [r for r in RESULTS if not r[1]]
    print(f"[summary] {len(RESULTS) - len(failed)}/{len(RESULTS)} passed", flush=True)
    for name, _, msg in failed:
        print(f"  FAILED {name}: {msg[:300]}", flush=True)
    print("RESULT PASS" if not failed else "RESULT FAIL", flush=True)


if __name__ == "__main__":
    main()
