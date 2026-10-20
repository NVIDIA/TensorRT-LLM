# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TRTLLM_Q35_MOE_FRONT_FUSED: the one-launch MoE front (cute_dsl_kernels/blackwell/q35_moe_front.py) against the
two-launch front it replaces (low-M BF16 router GEMM, then the per-token route + tables + NVFP4 quant launch).

* parity: logits, top-k ids, BF16 weights, xq / xsf bitwise; the tables exactly those of the stable single-program
  route kernel (rows in token order inside an expert's block) and equivalent to the per-token kernel's; the ticket
  back at zero after every launch; exact logit ties (duplicated router rows) go to the smaller expert id;
* PDL: a delayed producer (x is garbage until it writes) under CUDA graph replay; the kernel's PTX: nothing but
  static loads before griddepcontrol.wait, the trigger after the last global access + a CTA barrier;
* the pre-routed trtllm-gen MoE output with the fused front's tables equals the one with the two-launch front's
  tables (bitwise);
* benches (CUDA graph): back-to-back fronts (L2-hot weights), fronts behind an early-releasing 5 us predecessor
  with cold router weights (64 weight sets), and front + pre-routed MoE.

Run directly: python3 test_q35_moe_front_fused.py [--nobench | --bench-only]. Needs SM100/SM103.
"""

import os
import sys
import time
import traceback

# PTX of the CuTe DSL kernels for the PDL order check (before cutlass is imported)
_DUMP = f"/tmp/q35_front_dump_{os.getpid()}"
os.makedirs(_DUMP, exist_ok=True)
os.environ.setdefault("CUTE_DSL_KEEP_PTX", "1")
os.environ.setdefault("CUTE_DSL_DUMP_DIR", _DUMP)
os.environ.setdefault("CUTE_DSL_DISABLE_FILE_CACHING", "1")

import torch  # noqa: E402
import triton  # noqa: E402
import triton.language as tl  # noqa: E402

from tensorrt_llm._torch.modules.fused_moe import q35_fused_front as ff  # noqa: E402
from tensorrt_llm._torch.modules.fused_moe.q35_fused_front import (  # noqa: E402
    Q35FusedFront, Q35RouteTables, q35_lowm_router, q35_nvfp4_quant, q35_prerouted_sizes, q35_prerouted_views)

E, K, H = 512, 10, 4096
RESULTS = []


def _sizes(m, nl, tile):
    return sum(q35_prerouted_sizes(m, K, nl, tile))


def _inputs(m, seed, dup=False):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = (torch.randn((m, H), generator=g, device="cuda") * 0.5).to(torch.bfloat16)
    x[0, :32] = 0  # all-zero quant blocks
    if m > 1:
        x[1, 100] = 40.0  # an outlier block
    w = (torch.randn((E, H), generator=g, device="cuda") * 0.02).to(torch.bfloat16)
    if dup:
        # exact logit ties inside every token's top-k: duplicated router rows aligned with the tokens' mean
        base = x.float().mean(0)
        base = base / base.norm()
        for i, (a, b) in enumerate(((300, 7), (65, 64), (13, 200), (0, 511), (130, 129))):
            row = (base * (4.0 - 0.3 * i)).to(torch.bfloat16)
            w[a] = row
            w[b] = row
    gs = torch.tensor(448.0 * 6.0 / x.float().abs().max().item(), dtype=torch.float32, device="cuda")
    return x, w, gs


def _pl0(x, w, gs, ls, nl, tile, rt=None):
    """The two-launch front: low-M router, then the per-token route + tables + quant launch."""
    rt = rt or Q35RouteTables()
    flat = torch.full((_sizes(x.shape[0], nl, tile), ), -7, dtype=torch.int32, device="cuda")
    logits = q35_lowm_router(x, w)
    out = rt(logits, K, ls, nl, tile, flat=flat, single_program=False, x=x, global_scale=gs)
    return logits, out[8], out[9], out[0], out[1], flat


def _tables(flat, m, nl, tile):
    return q35_prerouted_views(flat, m, K, nl, tile)


def _check_tables_exact(got, ref, tag):
    e2p, p2t, cte, ctl, nct, npd = got
    r_e2p, r_p2t, r_cte, r_ctl, r_nct, r_npd = ref
    nt = int(r_nct.item())
    assert int(nct.item()) == nt and int(npd.item()) == int(r_npd.item()), (tag, "counts", int(nct.item()), nt)
    assert torch.equal(cte[:nt], r_cte[:nt]), (tag, "cta -> expert")
    assert torch.equal(ctl[:nt], r_ctl[:nt]), (tag, "cta -> limit")
    assert torch.equal(e2p, r_e2p), (tag, "expanded -> permuted")
    rows = e2p[e2p >= 0].long()
    assert torch.equal(p2t[rows], r_p2t[rows]), (tag, "permuted -> token")


def _check_tables_equivalent(got, ref, m, tile, tag):
    """Same layout up to the row order inside an expert's block (the per-token kernel ranks rows by atomics)."""
    e2p, p2t, cte, ctl, nct, npd = got
    r_e2p, r_p2t, r_cte, r_ctl, r_nct, r_npd = ref
    nt = int(r_nct.item())
    assert int(nct.item()) == nt and int(npd.item()) == int(r_npd.item()), (tag, "counts")
    assert torch.equal(cte[:nt], r_cte[:nt]) and torch.equal(ctl[:nt], r_ctl[:nt]), (tag, "ctas")
    assert torch.equal(e2p < 0, r_e2p < 0), (tag, "local picks")
    valid = (e2p >= 0).nonzero().view(-1)
    got_p2e = torch.full((nt * tile, ), -1, dtype=torch.int64, device=e2p.device)
    ref_p2e = torch.full((nt * tile, ), -1, dtype=torch.int64, device=e2p.device)
    got_p2e[e2p[valid].long()] = valid
    ref_p2e[r_e2p[valid].long()] = valid
    # per expert block (its CTAs together): the same expanded indices; the row order inside the block is free
    t = 0
    while t < nt:
        e = int(r_cte[t].item())
        t1 = t
        while t1 + 1 < nt and int(r_cte[t1 + 1].item()) == e:
            t1 += 1
        lo, hi = t * tile, int(r_ctl[t1].item())
        assert torch.equal(torch.sort(got_p2e[lo:hi]).values, torch.sort(ref_p2e[lo:hi]).values), (tag, "expert", e)
        t = t1 + 1
    assert torch.equal(p2t[e2p[valid].long()], (valid // K).to(torch.int32)), (tag, "p2t")


PARITY_CASES = [(m, nl, ls, tile) for m in (1, 7, 8, 9, 14, 16, 21, 28, 32)
                for (nl, ls) in ((64, 448), (64, 0), (128, 256), (256, 256), (256, 0))
                for tile in (8, 16)]


MODES = {"v3": (1, 1), "v2": (0, 1), "v1": (0, 0)}  # (sync_mode, topk_mode): v3 = staging + sorted top-k


def _sync_clear(front, sync_mode):
    if sync_mode == 1:
        return int(front.stage.abs().sum().item()) == 0
    return int(front.ticket.item()) == 0


def test_parity(m, nl, ls, tile, dup=False, mode="v3"):
    sync_mode, topk_mode = MODES[mode]
    x, w, gs = _inputs(m, seed=100 + 7 * m + nl + tile + (1000 if dup else 0), dup=dup)
    tag = (m, nl, ls, tile, "dup" if dup else "", mode)
    lg_r, xq_r, xsf_r, ids_r, wts_r, flat_r = _pl0(x, w, gs, ls, nl, tile)
    flat_s = torch.full_like(flat_r, -7)
    ids_s, wts_s = Q35RouteTables()(lg_r, K, ls, nl, tile, flat=flat_s)[:2]  # stable (sorted) tables
    front = Q35FusedFront()
    for it in range(3):  # later launches: the ticket must be back at zero
        flat = torch.full_like(flat_r, -7)
        out = front(x, w, gs, flat, tile, ls, nl, K, sync_mode=sync_mode, topk_mode=topk_mode)
        assert out is not None, (tag, "fused front declined")
        lg, xq, xsf, ids, wts = out
        torch.cuda.synchronize()
        assert torch.equal(lg.view(torch.int16), lg_r.view(torch.int16)), (
            tag, it, "logits", (lg.float() - lg_r.float()).abs().max().item())
        assert torch.equal(xq, xq_r), (tag, it, "xq", int((xq != xq_r).sum().item()))
        assert torch.equal(xsf, xsf_r), (tag, it, "xsf", int((xsf != xsf_r).sum().item()))
        assert torch.equal(ids, ids_r), (tag, it, "ids", ids.tolist()[:2], ids_r.tolist()[:2])
        assert torch.equal(wts.view(torch.int16), wts_r.view(torch.int16)), (tag, it, "weights")
        assert torch.equal(ids, ids_s) and torch.equal(wts.view(torch.int16), wts_s.view(torch.int16)), (tag, "sp")
        _check_tables_exact(_tables(flat, m, nl, tile), _tables(flat_s, m, nl, tile), (tag, it))
        _check_tables_equivalent(_tables(flat, m, nl, tile), _tables(flat_r, m, nl, tile), m, tile, (tag, it))
        assert _sync_clear(front, sync_mode), (tag, it, "ticket / staging not clear")
    if dup:
        # the duplicated pairs: both members tie exactly; the smaller expert id must come first
        lg_f = lg_r.float()
        for a, b in ((300, 7), (65, 64), (13, 200), (0, 511), (130, 129)):
            assert torch.equal(lg_f[:, a], lg_f[:, b]), (tag, "no tie", a, b)
        lst = ids.tolist()
        for row in lst:
            for a, b in ((7, 300), (64, 65), (13, 200), (0, 511), (129, 130)):
                if a in row and b in row:
                    assert row.index(a) < row.index(b), (tag, "tie order", row)


@triton.jit
def _delayed_producer_kernel(src_ptr, dst_ptr, n, spin_ns, BLOCK: tl.constexpr):
    """Adversarial PDL producer: releases its dependents first, then spins, then writes the consumer's input."""
    tl.extra.cuda.gdc_launch_dependents()
    t0 = tl.inline_asm_elementwise("mov.u64 $0, %globaltimer;", "=l", [], dtype=tl.int64, is_pure=False, pack=1)
    t1 = t0
    while t1 - t0 < spin_ns:
        t1 = tl.inline_asm_elementwise("mov.u64 $0, %globaltimer;", "=l", [], dtype=tl.int64, is_pure=False,
                                       pack=1)
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(dst_ptr + offs, tl.load(src_ptr + offs, mask=offs < n), mask=offs < n)


@triton.jit
def _nowait_copy_kernel(src_ptr, dst_ptr, n, BLOCK: tl.constexpr):
    """Positive control: a PDL dependent that does NOT griddepcontrol.wait."""
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(dst_ptr + offs, tl.load(src_ptr + offs, mask=offs < n), mask=offs < n)


def test_pdl_delayed_producer(m=7, nl=64, ls=448, tile=8, mode="v3"):
    """x is garbage until a delayed producer (which releases its dependents at entry) writes it; the fused front
    (a PDL dependent) must still see the final x. 20 CUDA graph replays, each with every output buffer zeroed,
    must equal the eager result; a dependent without a wait must read early in some replays (control)."""
    sync_mode, topk_mode = MODES[mode]
    x_src, w, gs = _inputs(m, seed=55)
    x = torch.full_like(x_src, 3.0e4)
    n = x.numel()
    ref = Q35FusedFront()(x_src, w, gs, torch.full((_sizes(m, nl, tile), ), -7, dtype=torch.int32, device="cuda"),
                          tile, ls, nl, K, sync_mode=sync_mode, topk_mode=topk_mode)
    ref = [t.clone() for t in ref]
    front = Q35FusedFront()
    flat = torch.full((_sizes(m, nl, tile), ), -7, dtype=torch.int32, device="cuda")
    y = torch.empty_like(x)

    def producer():
        x.fill_(3.0e4)
        _delayed_producer_kernel[(triton.cdiv(n, 4096), )](x_src, x, n, 50000, BLOCK=4096, launch_pdl=True)

    def step():
        producer()
        return front(x, w, gs, flat, tile, ls, nl, K, sync_mode=sync_mode, topk_mode=topk_mode)

    def control():
        producer()
        _nowait_copy_kernel[(triton.cdiv(n, 4096), )](x, y, n, BLOCK=4096, launch_pdl=True)

    step()
    control()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outs = step()
    cgraph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(cgraph):
        control()
    bad = stale = 0
    for _ in range(20):
        for b in outs:
            b.zero_()
        graph.replay()
        torch.cuda.synchronize()
        bad += int(not all(torch.equal(a.view(torch.uint8), b.view(torch.uint8)) for a, b in zip(outs, ref)))
        cgraph.replay()
        torch.cuda.synchronize()
        stale += int(not torch.equal(y, x_src))
    assert _sync_clear(front, sync_mode), "ticket / staging not clear"
    assert stale > 0, "positive control never read early: the test cannot detect a missing wait"
    assert bad == 0, f"{bad}/20 replays differ (read the producer's data early?)"
    print(f"[pdl delayed producer {mode} m{m}] 20/20 replays correct; positive control stale in {stale}/20",
          flush=True)


def _pdl_ptx_checker():
    """The PDL PTX-order helper next to this test (pdl_ptx_check.py), or None when it is not present."""
    import importlib.util
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pdl_ptx_check.py")
    if not os.path.isfile(path):
        return None
    spec = importlib.util.spec_from_file_location("pdl_ptx_check", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_ptx_order(mode="v3"):
    """The fused kernel's PTX: before griddepcontrol.wait only loads (router weights, the global scale) and no
    global store / atomic; every launch_dependents after a CTA barrier that follows the last global access."""
    import glob
    sync_mode, topk_mode = MODES[mode]
    for m, nl in ((7, 64), (14, 128), (28, 256)):
        before = set(glob.glob(os.path.join(_DUMP, "**", "*.ptx"), recursive=True))
        x, w, gs = _inputs(m, seed=77)
        ff._Q35_FRONT_COMPILED.clear()
        Q35FusedFront()(x, w, gs, torch.empty((_sizes(m, nl, 8), ), dtype=torch.int32, device="cuda"), 8, 0, nl, K,
                        sync_mode=sync_mode, topk_mode=topk_mode)
        torch.cuda.synchronize()
        new = sorted(set(glob.glob(os.path.join(_DUMP, "**", "*.ptx"), recursive=True)) - before)
        fn = next(iter(ff._Q35_FRONT_COMPILED.values()))[0]
        ptx = getattr(fn, "__ptx__", None)
        if isinstance(ptx, bytes):
            ptx = ptx.decode(errors="replace")
        if not ptx and new:
            ptx = open(new[-1], errors="replace").read()
        if not ptx:
            files = glob.glob(os.path.join(_DUMP, "**", "*"), recursive=True)
            raise AssertionError(f"no PTX for m={m} (dump dir: {files[:10]})")
        out = os.path.join(os.environ.get("OUT", "/tmp"), f"q35_moe_front_{mode}_m{m}_nl{nl}.ptx")
        with open(out, "w") as f:
            f.write(ptx)
        txt = _pdl_ptx_checker()
        assert "griddepcontrol.wait" in ptx and "griddepcontrol.launch_dependents" in ptx, (m, "no PDL ops")
        msg = f"[ptx {mode}] m={m} nl={nl}: {out}"
        if txt is not None:
            rep = txt.assert_wait_first(ptx, f"front m{m}", allow_loads_before=True)
            txt.assert_trigger_last(ptx, f"front m{m}")
            msg += f"; text order: {len(rep['before_wait'])} loads before the first wait of {rep['global_ops']}"
        print(msg, flush=True)


def _rand_fp4_moe(nl, inter, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    u8 = lambda *s, lo=0, hi=256: torch.randint(lo, hi, s, generator=g, device="cuda", dtype=torch.int32).to(  # noqa
        torch.uint8)
    w1, s1 = u8(nl, 2 * inter, H // 2), u8(nl, 2 * inter, H // 16, lo=0x28, hi=0x38)
    w2, s2 = u8(nl, H, inter // 2), u8(nl, H, inter // 16, lo=0x28, hi=0x38)
    a1 = torch.rand((nl, ), generator=g, device="cuda") * 1e-2 + 1e-3
    a2 = torch.rand((nl, ), generator=g, device="cuda") * 1e-2 + 1e-3
    return w1, s1, a1, w2, s2, a2


def _moe_runner(m, nl, ls, inter, seed):
    from tensorrt_llm._torch.custom_ops.trtllm_gen_custom_ops import FP4BlockScaleMoERunner
    from tensorrt_llm._torch.modules.fused_moe.routing import RoutingMethodType
    from tensorrt_llm._torch.utils import ActType_TrtllmGen
    w1, s1, a1, w2, s2, a2 = _rand_fp4_moe(nl, inter, seed)
    runner = FP4BlockScaleMoERunner(E, K, None, None, inter, ls, nl, None, int(RoutingMethodType.Renormalize), True,
                                    ActType_TrtllmGen.SwiGlu.value)
    e4m3 = torch.float8_e4m3fn

    def run(logits, xq, xsf, tile, cfg, pre=None):
        inp = [logits, None, xq, xsf.view(-1).view(e4m3), w1, s1.view(e4m3), None, None, None, None, w2,
               s2.view(e4m3), None, a1, a1, a2, None, None]
        if pre is None:
            return runner(inp, tactic=[tile, cfg])[0]
        return runner(inp, tactic=[tile, cfg], prerouted=pre, prerouted_tile=tile)[0]

    return runner, run


def test_moe_parity(m, nl, ls, inter, tile=8):
    """Pre-routed trtllm-gen MoE output with the fused front's (xq, xsf, tables, ids, weights) == with the
    two-launch front's, and == the stock op routing the same logits itself (bitwise)."""
    from tensorrt_llm._torch.custom_ops.trtllm_gen_custom_ops import FP4BlockScaleMoERunner
    if not FP4BlockScaleMoERunner.supports_prerouted():
        print("[moe parity] SKIP: libth_common without run_moe_prerouted", flush=True)
        return
    x, w, gs = _inputs(m, seed=900 + m + nl)
    runner, run = _moe_runner(m, nl, ls, inter, seed=901 + m)
    cfg = sorted(int(c) for t, c in runner.get_runner().get_valid_configs(K, H, inter, nl, m) if int(t) == tile)[0]
    lg_r, xq_r, xsf_r, ids_r, wts_r, flat_r = _pl0(x, w, gs, ls, nl, tile)
    flat = torch.full_like(flat_r, -7)
    lg, xq, xsf, ids, wts = Q35FusedFront()(x, w, gs, flat, tile, ls, nl, K)
    out_f = run(lg, xq, xsf, tile, cfg, (_tables(flat, m, nl, tile), ids, wts))
    out_r = run(lg_r, xq_r, xsf_r, tile, cfg, (_tables(flat_r, m, nl, tile), ids_r, wts_r))
    out_s = run(lg_r, xq_r, xsf_r, tile, cfg)
    torch.cuda.synchronize()
    tag = (m, nl, ls, inter, tile, cfg)
    assert torch.equal(out_f.view(torch.int16), out_r.view(torch.int16)), (tag, "fused vs two-launch")
    assert torch.equal(out_f.view(torch.int16), out_s.view(torch.int16)), (tag, "fused vs stock routing")


# ------------------------------------------------------------------------------------------------- benches
def _graph_us(fn, iters=2000, per_graph=20):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    gr = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gr):
        for _ in range(per_graph):
            fn()
    gr.replay()
    torch.cuda.synchronize()
    t0, t1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    t0.record()
    for _ in range(iters // per_graph):
        gr.replay()
    t1.record()
    torch.cuda.synchronize()
    return t0.elapsed_time(t1) * 1000.0 / (iters // per_graph * per_graph)


@triton.jit
def _spin_kernel(spin_ns):
    """A predecessor like the post-attention all-reduce: waits for its own predecessor, releases its dependents
    (which then prefetch their static weights), then runs for spin_ns."""
    tl.extra.cuda.gdc_wait()
    tl.extra.cuda.gdc_launch_dependents()
    t0 = tl.inline_asm_elementwise("mov.u64 $0, %globaltimer;", "=l", [], dtype=tl.int64, is_pure=False, pack=1)
    t1 = t0
    while t1 - t0 < spin_ns:
        t1 = tl.inline_asm_elementwise("mov.u64 $0, %globaltimer;", "=l", [], dtype=tl.int64, is_pure=False,
                                       pack=1)


def _spin(ns):
    _spin_kernel[(1, )](ns, num_warps=1, launch_pdl=True)


def bench(m, nl, ls, tile=8, iters=2000, modes=("v1", "v2", "v3")):
    x, w, gs = _inputs(m, seed=31)
    rt = Q35RouteTables()
    flat_a = torch.empty((_sizes(m, nl, tile), ), dtype=torch.int32, device="cuda")
    flat_b = torch.empty_like(flat_a)
    nw = 64  # 64 x 4 MiB router weights: more than L2, so every front reads its weights from HBM
    ws = [(torch.randn((E, H), device="cuda") * 0.02).to(torch.bfloat16) for _ in range(nw)]
    state = {"i": 0}

    def pl0(wt=w):
        lg = q35_lowm_router(x, wt)
        rt(lg, K, ls, nl, tile, flat=flat_a, single_program=False, x=x, global_scale=gs)

    def fused_fn(mode):
        front = Q35FusedFront()
        sync_mode, topk_mode = MODES[mode]

        def f(wt=w):
            front(x, wt, gs, flat_b, tile, ls, nl, K, sync_mode=sync_mode, topk_mode=topk_mode)
        return f

    def cold(fn):
        def f():
            fn(ws[state["i"] % nw])
            state["i"] += 1
        return f

    def spun(fn, ns=5000):
        def f():
            _spin(ns)
            fn()
        return f

    res = {"lowm router": _graph_us(lambda: q35_lowm_router(x, w), iters)}
    spin_only = _graph_us(lambda: _spin(5000), iters // 4)
    arms = [("two-launch", pl0)] + [(md, fused_fn(md)) for md in modes]
    for name, fn in arms:
        res[f"{name} b2b"] = _graph_us(fn, iters)
        res[f"{name} spin"] = _graph_us(spun(cold(fn)), iters // 4, per_graph=nw) - spin_only
    line = ", ".join(f"{k} {v:.2f}" for k, v in res.items())
    sav = ", ".join(f"{md} {res['two-launch b2b'] - res[md + ' b2b']:.2f}/"
                    f"{res['two-launch spin'] - res[md + ' spin']:.2f}" for md in modes)
    print(f"[bench front] m={m} nl={nl} tile={tile} (us/front): {line} | saving vs two-launch (b2b/spin): {sav}",
          flush=True)
    return res


def bench_moe(m, nl, ls, inter, tile=8, iters=1000, modes=("v2", "v3")):
    """front + pre-routed trtllm-gen MoE (FC1 + FC2), back-to-back and behind the 5 us spin."""
    from tensorrt_llm._torch.custom_ops.trtllm_gen_custom_ops import FP4BlockScaleMoERunner
    if not FP4BlockScaleMoERunner.supports_prerouted():
        print("[bench moe] SKIP: no run_moe_prerouted", flush=True)
        return
    x, w, gs = _inputs(m, seed=41)
    runner, run = _moe_runner(m, nl, ls, inter, seed=42)
    cfgs = sorted(int(c) for t, c in runner.get_runner().get_valid_configs(K, H, inter, nl, m) if int(t) == tile)
    cfg = cfgs[len(cfgs) // 2]
    rt = Q35RouteTables()
    flat_a = torch.empty((_sizes(m, nl, tile), ), dtype=torch.int32, device="cuda")
    flat_b = torch.empty_like(flat_a)

    def pl0():
        lg = q35_lowm_router(x, w)
        o = rt(lg, K, ls, nl, tile, flat=flat_a, single_program=False, x=x, global_scale=gs)
        run(lg, o[8], o[9], tile, cfg, (_tables(flat_a, m, nl, tile), o[0], o[1]))

    def fused_fn(mode):
        front = Q35FusedFront()
        sync_mode, topk_mode = MODES[mode]

        def f():
            lg, xq, xsf, ids, wts = front(x, w, gs, flat_b, tile, ls, nl, K, sync_mode=sync_mode,
                                          topk_mode=topk_mode)
            run(lg, xq, xsf, tile, cfg, (_tables(flat_b, m, nl, tile), ids, wts))
        return f

    res = {}
    spin_only = _graph_us(lambda: _spin(5000), iters // 2)
    for name, fn in [("two-launch", pl0)] + [(md, fused_fn(md)) for md in modes]:
        res[f"{name}+moe b2b"] = _graph_us(fn, iters)
        res[f"{name}+moe spin"] = _graph_us(lambda fn=fn: (_spin(5000), fn()), iters // 2) - spin_only
    sav = ", ".join(f"{md} {res['two-launch+moe b2b'] - res[md + '+moe b2b']:.2f}/"
                    f"{res['two-launch+moe spin'] - res[md + '+moe spin']:.2f}" for md in modes)
    print(f"[bench moe] m={m} nl={nl} inter={inter} cfg={cfg} (us): "
          + ", ".join(f"{k} {v:.2f}" for k, v in res.items()) + f" | saving vs two-launch (b2b/spin): {sav}",
          flush=True)


def trace_phases(m, nl, ls, mode, tile=8, reps=40):
    """%globaltimer phases of the fused front (trace build) behind the 5 us spin, cold-ish (eager launches),
    medians over reps, ns relative to the GEMM CTAs' earliest return from griddepcontrol.wait (= the predecessor's
    end): GEMM CTAs' logits out (median / max over CTAs), routing CTA: start (ticket / wait), logits in registers,
    top-k done, masks barrier, scan barrier, end; quant CTAs' end (max)."""
    import statistics
    sync_mode, topk_mode = MODES[mode]
    x, w, gs = _inputs(m, seed=61)
    front = Q35FusedFront()
    flat = torch.empty((_sizes(m, nl, tile), ), dtype=torch.int32, device="cuda")
    cols = {k: [] for k in ("gemm_out_med", "gemm_out_max", "route_start", "data", "topk", "masks", "scan", "end",
                            "quant_end")}
    for rep in range(reps + 3):
        _spin(5000)
        front(x, w, gs, flat, tile, ls, nl, K, sync_mode=sync_mode, topk_mode=topk_mode, trace=True)
        torch.cuda.synchronize()
        if rep < 3:
            continue
        tb = front.trace_buf.view(-1, 8).cpu().tolist()
        g0 = 1 if sync_mode == 1 else 0
        gemm = [tb[c] for c in range(0, 64)]
        router = tb[-1]
        n_ctas = len(tb) - 1
        q0 = g0 + 64
        quant = [tb[c] for c in range(q0, n_ctas)]
        t0 = min(r[1] for r in gemm)
        outs = sorted(r[2] - t0 for r in gemm)
        cols["gemm_out_med"].append(outs[len(outs) // 2])
        cols["gemm_out_max"].append(outs[-1])
        for i, k in ((1, "route_start"), (2, "data"), (3, "topk"), (4, "masks"), (5, "scan"), (6, "end")):
            cols[k].append(router[i] - t0)
        cols["quant_end"].append(max(r[2] for r in quant) - t0 if quant else 0)
    med = {k: statistics.median(v) for k, v in cols.items()}
    print(f"[trace {mode}] m={m} nl={nl} (ns after the predecessor's end, median of {reps}): "
          + ", ".join(f"{k} {v:.0f}" for k, v in med.items()), flush=True)
    return med


# ------------------------------------------------------------------------------------------------- runner
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
    print(f"device {props.name} sm{props.major}{props.minor} sms={props.multi_processor_count}", flush=True)
    bench_only = "--bench-only" in sys.argv
    if not bench_only:
        ok = _run("parity m7 nl64 v3", test_parity, 7, 64, 448, 8)  # compile + first parity, fail fast
        ok = _run("parity m28 nl128 v3", test_parity, 28, 128, 128, 8) and ok
        if ok:
            n_pass = n_all = 0
            for case in PARITY_CASES:
                n_all += 1
                n_pass += int(_run(f"parity {case} v3", test_parity, *case))
            for case in ((7, 64, 448, 8), (14, 128, 256, 8), (28, 256, 0, 8), (32, 64, 0, 16)):
                for mode in ("v3", "v2"):
                    n_all += 1
                    n_pass += int(_run(f"parity dup {case} {mode}", test_parity, *case, dup=True, mode=mode))
            for m in (1, 7, 14, 28, 32):
                for nl, ls in ((64, 448), (256, 0)):
                    for mode in ("v2", "v1"):
                        n_all += 1
                        n_pass += int(_run(f"parity {(m, nl, ls, 8)} {mode}", test_parity, m, nl, ls, 8,
                                           mode=mode))
            print(f"[parity] {n_pass}/{n_all} cases pass", flush=True)
            for mode in ("v3", "v2"):
                _run(f"pdl delayed producer {mode}", test_pdl_delayed_producer, mode=mode)
                _run(f"pdl delayed producer m28 {mode}", test_pdl_delayed_producer, 28, 128, 128, 8, mode=mode)
                _run(f"ptx order {mode}", test_ptx_order, mode)
            for case in ((7, 64, 448, 512), (14, 128, 256, 512), (28, 128, 0, 512), (7, 256, 256, 256)):
                _run(f"moe parity {case}", test_moe_parity, *case)
    if "--nobench" not in sys.argv:
        for m in (1, 7, 14, 28):
            for nl, ls in ((64, 448), (128, 256), (256, 0)):
                _run(f"bench m{m} nl{nl}", bench, m, nl, ls)
        for m, nl, ls, inter in ((7, 64, 448, 512), (7, 256, 256, 256), (14, 128, 128, 512), (28, 64, 0, 512),
                                 (28, 128, 0, 512)):
            _run(f"bench moe m{m} nl{nl}", bench_moe, m, nl, ls, inter)
        for m in (1, 7, 14, 28):
            for mode in ("v2", "v3"):
                _run(f"trace {mode} m{m}", trace_phases, m, 64, 448, mode)
    failed = [r for r in RESULTS if not r[1]]
    print(f"[summary] {len(RESULTS) - len(failed)}/{len(RESULTS)} passed", flush=True)
    for name, _, msg in failed:
        print(f"  FAILED {name}: {msg[:300]}", flush=True)
    print("RESULT PASS" if not failed else "RESULT FAIL", flush=True)


if __name__ == "__main__":
    main()
