# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The one-launch Qwen3.5 MoE front (kernels/blackwell/q35_moe_front.py) against the two-launch front
(low-M BF16 router GEMM, then the Triton route + tables + NVFP4 quant launch): bitwise parity, PDL ordering, and
the pre-routed trtllm-gen MoE output. Needs SM100/SM103.
"""

import os

# Dump the CuTe DSL PTX for the PDL order check; must be set before cutlass is imported.
_DUMP = f"/tmp/q35_front_dump_{os.getpid()}"
os.makedirs(_DUMP, exist_ok=True)
os.environ.setdefault("CUTE_DSL_KEEP", "ptx")
os.environ.setdefault("CUTE_DSL_DUMP_DIR", _DUMP)
os.environ.setdefault("CUTE_DSL_DISABLE_FILE_CACHING", "1")

import pytest  # noqa: E402
import torch  # noqa: E402
import triton  # noqa: E402
import triton.language as tl  # noqa: E402

from tensorrt_llm._torch.moe.fused_moe import q35_fused_front as ff  # noqa: E402
from tensorrt_llm._torch.moe.fused_moe.q35_fused_front import (  # noqa: E402
    Q35FusedFront,
    Q35RouteTables,
    q35_lowm_router,
    q35_prerouted_sizes,
    q35_prerouted_views,
)

E, K, H = 512, 10, 4096

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="needs SM100 / SM103",
)


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
    gs = torch.tensor(
        448.0 * 6.0 / x.float().abs().max().item(), dtype=torch.float32, device="cuda"
    )
    return x, w, gs


def _pl0(x, w, gs, ls, nl, tile, rt=None):
    """The two-launch front: low-M router, then the per-token route + tables + quant launch."""
    rt = rt or Q35RouteTables()
    flat = torch.full((_sizes(x.shape[0], nl, tile),), -7, dtype=torch.int32, device="cuda")
    logits = q35_lowm_router(x, w)
    out = rt(logits, K, ls, nl, tile, flat=flat, x=x, global_scale=gs)
    return logits, out[8], out[9], out[0], out[1], flat


def _tables(flat, m, nl, tile):
    return q35_prerouted_views(flat, m, K, nl, tile)


def _check_token_order(got, tile, tag):
    """Inside an expert's block the one-launch front writes the rows in token order."""
    e2p, p2t, cte, ctl, nct, npd = got
    p2t, ctl = p2t.cpu(), ctl.cpu()
    for c in range(int(nct.item())):
        rows = p2t[c * tile : int(ctl[c].item())].tolist()
        assert rows == sorted(rows), (tag, "block rows not in token order", c)


def _check_tables_equivalent(got, ref, m, tile, tag):
    """Same layout up to the row order inside an expert's block (the per-token kernel ranks rows by atomics)."""
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
    # per expert block (its CTAs together): the same expanded indices; the row order inside the block is free
    t = 0
    while t < nt:
        e = int(r_cte[t].item())
        t1 = t
        while t1 + 1 < nt and int(r_cte[t1 + 1].item()) == e:
            t1 += 1
        lo, hi = t * tile, int(r_ctl[t1].item())
        assert torch.equal(torch.sort(got_p2e[lo:hi]).values, torch.sort(ref_p2e[lo:hi]).values), (
            tag,
            "expert",
            e,
        )
        t = t1 + 1
    assert torch.equal(p2t[e2p[valid].long()], (valid // K).to(torch.int32)), (tag, "p2t")


# (m, num_local, local_start, tile, dup): M = 1 / 8 (32 warps) / 9 (16 warps) / 21, 32 (two tokens per warp,
# several CTAs per expert); num_local 64 / 128 = Qwen3.5-397B at TP8 / TP4. dup: exact logit ties.
PARITY_CASES = [
    (1, 64, 448, 8, False),
    (8, 128, 256, 16, False),
    (9, 64, 0, 8, False),
    (21, 256, 256, 16, False),
    (32, 64, 448, 8, False),
    (32, 128, 256, 16, False),
    (7, 64, 448, 8, True),
    (28, 256, 0, 8, True),
]


def _stage_clear(front):
    return int(front.stage.abs().sum().item()) == 0


@pytest.mark.parametrize("m,nl,ls,tile,dup", PARITY_CASES)
def test_parity(m, nl, ls, tile, dup):
    x, w, gs = _inputs(m, seed=100 + 7 * m + nl + tile + (1000 if dup else 0), dup=dup)
    tag = (m, nl, ls, tile, "dup" if dup else "")
    lg_r, xq_r, xsf_r, ids_r, wts_r, flat_r = _pl0(x, w, gs, ls, nl, tile)
    front = Q35FusedFront()
    for it in range(2):  # the second launch needs the staging buffer back at zero
        flat = torch.full_like(flat_r, -7)
        out = front(x, w, gs, flat, tile, ls, nl, K)
        assert out is not None, (tag, "fused front declined")
        lg, xq, xsf, ids, wts = out
        torch.cuda.synchronize()
        assert torch.equal(lg.view(torch.int16), lg_r.view(torch.int16)), (
            tag,
            it,
            "logits",
            (lg.float() - lg_r.float()).abs().max().item(),
        )
        assert torch.equal(xq, xq_r), (tag, it, "xq", int((xq != xq_r).sum().item()))
        assert torch.equal(xsf, xsf_r), (tag, it, "xsf", int((xsf != xsf_r).sum().item()))
        assert torch.equal(ids, ids_r), (tag, it, "ids", ids.tolist()[:2], ids_r.tolist()[:2])
        assert torch.equal(wts.view(torch.int16), wts_r.view(torch.int16)), (tag, it, "weights")
        _check_token_order(_tables(flat, m, nl, tile), tile, (tag, it))
        _check_tables_equivalent(
            _tables(flat, m, nl, tile), _tables(flat_r, m, nl, tile), m, tile, (tag, it)
        )
        assert _stage_clear(front), (tag, it, "staging buffer not clear")
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


def test_pdl_delayed_producer(m=28, nl=128, ls=128, tile=8):
    """x is garbage until a delayed producer (which releases its dependents at entry) writes it; the fused front
    (a PDL dependent) must still see the final x. 20 CUDA graph replays, each with every output buffer zeroed,
    must equal the eager result; a dependent without a wait must read early in some replays (control)."""
    x_src, w, gs = _inputs(m, seed=55)
    x = torch.full_like(x_src, 3.0e4)
    n = x.numel()
    ref = Q35FusedFront()(
        x_src,
        w,
        gs,
        torch.full((_sizes(m, nl, tile),), -7, dtype=torch.int32, device="cuda"),
        tile,
        ls,
        nl,
        K,
    )
    ref = [t.clone() for t in ref]
    front = Q35FusedFront()
    flat = torch.full((_sizes(m, nl, tile),), -7, dtype=torch.int32, device="cuda")
    y = torch.empty_like(x)

    def producer():
        x.fill_(3.0e4)
        _delayed_producer_kernel[(triton.cdiv(n, 4096),)](
            x_src, x, n, 50000, BLOCK=4096, launch_pdl=True
        )

    def step():
        producer()
        return front(x, w, gs, flat, tile, ls, nl, K)

    def control():
        producer()
        _nowait_copy_kernel[(triton.cdiv(n, 4096),)](x, y, n, BLOCK=4096, launch_pdl=True)

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
        bad += int(
            not all(
                torch.equal(a.view(torch.uint8), b.view(torch.uint8)) for a, b in zip(outs, ref)
            )
        )
        cgraph.replay()
        torch.cuda.synchronize()
        stale += int(not torch.equal(y, x_src))
    assert _stage_clear(front), "staging buffer not clear"
    assert stale > 0, "positive control never read early: the test cannot detect a missing wait"
    assert bad == 0, f"{bad}/20 replays differ (read the producer's data early?)"


def _pdl_ptx_checker():
    import importlib.util

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pdl_ptx_check.py")
    spec = importlib.util.spec_from_file_location("pdl_ptx_check", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_ptx_order(monkeypatch):
    """The fused kernel's PTX: before griddepcontrol.wait only loads (router weights, the global scale) and no
    global store / atomic; every launch_dependents after a CTA barrier that follows the last global access."""
    import glob

    import cutlass.cute as cute

    # The DSL reads its keep / dump environment once, possibly before this module set it (pytest imports
    # tensorrt_llm first), so request the PTX through the compile options instead.
    compile_fn = cute.compile

    def compile_keep_ptx(*args, **kwargs):
        kwargs.setdefault("options", f"--keep-ptx --dump-dir {_DUMP}")
        return compile_fn(*args, **kwargs)

    monkeypatch.setattr(cute, "compile", compile_keep_ptx)
    txt = _pdl_ptx_checker()
    for m, nl in ((7, 64), (28, 128)):  # 32 warps / 16 warps with two tokens per warp
        before = set(glob.glob(os.path.join(_DUMP, "**", "*.ptx"), recursive=True))
        x, w, gs = _inputs(m, seed=77)
        ff._Q35_FRONT_COMPILED.clear()
        Q35FusedFront()(
            x,
            w,
            gs,
            torch.empty((_sizes(m, nl, 8),), dtype=torch.int32, device="cuda"),
            8,
            0,
            nl,
            K,
        )
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
        assert "griddepcontrol.wait" in ptx and "griddepcontrol.launch_dependents" in ptx, (
            m,
            "no PDL ops",
        )
        txt.assert_wait_first(ptx, f"front m{m}", allow_loads_before=True)
        txt.assert_trigger_last(ptx, f"front m{m}")


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


def _moe_runner(m, nl, ls, inter, seed):
    from tensorrt_llm._torch.custom_ops.trtllm_gen_custom_ops import FP4BlockScaleMoERunner
    from tensorrt_llm._torch.moe.fused_moe.routing import RoutingMethodType
    from tensorrt_llm._torch.utils import ActType_TrtllmGen

    w1, s1, a1, w2, s2, a2 = _rand_fp4_moe(nl, inter, seed)
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

    def run(logits, xq, xsf, tile, cfg, pre=None):
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

    return runner, run


@pytest.mark.parametrize("m,nl,ls,inter", [(7, 64, 448, 512), (28, 128, 0, 512)])
def test_moe_parity(m, nl, ls, inter, tile=8):
    """Pre-routed trtllm-gen MoE output with the fused front's (xq, xsf, tables, ids, weights) == with the
    two-launch front's, and == the stock op routing the same logits itself (bitwise)."""
    from tensorrt_llm._torch.custom_ops.trtllm_gen_custom_ops import FP4BlockScaleMoERunner

    if not FP4BlockScaleMoERunner.supports_prerouted():
        pytest.skip("libth_common without run_moe_prerouted")
    x, w, gs = _inputs(m, seed=900 + m + nl)
    runner, run = _moe_runner(m, nl, ls, inter, seed=901 + m)
    cfg = sorted(
        int(c)
        for t, c in runner.get_runner().get_valid_configs(K, H, inter, nl, m)
        if int(t) == tile
    )[0]
    lg_r, xq_r, xsf_r, ids_r, wts_r, flat_r = _pl0(x, w, gs, ls, nl, tile)
    flat = torch.full_like(flat_r, -7)
    lg, xq, xsf, ids, wts = Q35FusedFront()(x, w, gs, flat, tile, ls, nl, K)
    out_f = run(lg, xq, xsf, tile, cfg, (_tables(flat, m, nl, tile), ids, wts))
    out_r = run(lg_r, xq_r, xsf_r, tile, cfg, (_tables(flat_r, m, nl, tile), ids_r, wts_r))
    out_s = run(lg_r, xq_r, xsf_r, tile, cfg)
    torch.cuda.synchronize()
    tag = (m, nl, ls, inter, tile, cfg)
    assert torch.equal(out_f.view(torch.int16), out_r.view(torch.int16)), (
        tag,
        "fused vs two-launch",
    )
    assert torch.equal(out_f.view(torch.int16), out_s.view(torch.int16)), (
        tag,
        "fused vs stock routing",
    )
