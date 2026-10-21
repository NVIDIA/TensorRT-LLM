# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Gate for the SM107 CLC MXFP8 GEMM used for MiniMax-M3 CTX prefill GEMMs.

The GEMM-only handle of the fused QKV kernel (``m3_qkv_fusion.plain_gemm``)
replaces CUTLASS ``mxfp8_mxfp8_gemm`` in ``MXFP8LinearMethod.apply`` for the
o_proj / shared gate_up / dense gate_up / dense down shapes at >= 1024 rows.
Per shape and M the gate checks: 3-run determinism, CUDA-graph replay ==
eager, bitwise equality with the fused kernel's own ``debug_gemm_out`` output
(same mainloop, real producer tensors), and the fp64 error band against
CUTLASS. The dispatch test runs a real MXFP8 ``Linear`` and checks the
engagement log, the pre-quantized activation slot, and that the kill switch
gives CUTLASS's bytes.

Also runnable as a script::

    python3 test_m3_ctx_clc_gemm.py [--shapes o_proj,...] [--m 16384,1500] [--timing 16384,8192] [--no-dispatch]

``M3QKV_PKG`` may name a directory holding an ``m3_qkv_fusion`` package to
gate instead of the in-tree one (kernel-level checks; the dispatch test and
the custom-op timings always use the importable ``tensorrt_llm``).
"""

import argparse
import importlib
import os
import sys
import time

import pytest
import torch

# tensorrt_llm is imported inside the functions that need it, so that _pkg() first sets
# CUTE_DSL_ARCH (read by the CuTe DSL when it is imported) and puts any M3QKV_PKG on sys.path.

HEAD_DIM = 128
HEADS_KV, HEADS_INDEX = 2, 2
# name -> (N, K, heads per tile below WIDE_ROWS rows, at or above it)
SHAPES = {
    "o_proj": (6144, 4096, 2, 2),
    "shared_gate_up": (3072, 6144, 2, 4),
    "dense_gate_up": (12288, 6144, 2, 4),
    "dense_down": (6144, 6144, 2, 4),
}
WIDE_ROWS = 12288
MIN_ROWS = 1024
M_LIST = (16384, 8192, 4096, 1024, 1500)


def _is_sm107():
    return torch.cuda.is_available() and torch.cuda.get_device_capability() == (10, 7)


pytestmark = pytest.mark.skipif(not _is_sm107(), reason="SM107 (Rubin) required")


def _pkg():
    """(kernel module, plain_gemm module), from M3QKV_PKG when set."""
    os.environ.setdefault("CUTE_DSL_ARCH", "sm_107a")
    pkg = os.environ.get("M3QKV_PKG")
    if pkg:
        if pkg not in sys.path:
            sys.path.insert(0, pkg)
        base = "m3_qkv_fusion"
    else:
        base = "tensorrt_llm._torch.kernels.rubin.m3_qkv_fusion"
    return importlib.import_module(base + ".kernel"), importlib.import_module(base + ".plain_gemm")


def _dispatch_module():
    """The in-model dispatcher (registers trtllm::m3_ctx_clc_mxfp8_gemm), or None if this tensorrt_llm lacks it."""
    try:
        return importlib.import_module("tensorrt_llm._torch.modules.m3_ctx_clc_gemm")
    except ImportError:
        return None


def hpt_for(name, m):
    _, _, narrow, wide = SHAPES[name]
    return wide if m >= WIDE_ROWS else narrow


def _bits(t):
    return t.view(torch.int16)


class Problem:
    def __init__(self, n, k, m, seed=0):
        import tensorrt_llm  # noqa: F401  (registers torch.ops.trtllm)

        g = torch.Generator(device="cuda").manual_seed(seed)
        self.n, self.k, self.m = n, k, m
        x = torch.randn(m, k, device="cuda", generator=g, dtype=torch.bfloat16)
        w = (torch.randn(n, k, device="cuda", generator=g) * 0.02).to(torch.bfloat16)
        self.act, self.act_sf = torch.ops.trtllm.mxfp8_quantize(x, True)
        self.act_lin_sf = torch.ops.trtllm.mxfp8_quantize(x, False)[1]
        # Weight scales in the layout MXFP8LinearMethod stores: block_scale_interleave of the [N, K/32] scales.
        self.weight, self.weight_lin_sf = torch.ops.trtllm.mxfp8_quantize(w, False)
        self.weight_sf = torch.ops.trtllm.block_scale_interleave(
            self.weight_lin_sf.view(torch.uint8).reshape(n, k // 32)
        ).reshape(-1)
        self.weight_quant_sf = torch.ops.trtllm.mxfp8_quantize(w, True)[1]
        self.alpha = torch.ones(1, dtype=torch.float32, device="cuda")

    def cutlass(self):
        return torch.ops.trtllm.mxfp8_mxfp8_gemm(
            self.act, self.act_sf, self.weight, self.weight_sf, self.alpha, torch.bfloat16
        )

    def fp64_err(self, out, rows_per_chunk=4096):
        """(max, mean) |out - fp64 dequant reference|, in row chunks."""

        def deq(v, sf_lin, r0, r1):
            sf = sf_lin.view(torch.uint8).reshape(-1)[: v.shape[0] * (self.k // 32)]
            sf = sf.reshape(v.shape[0], self.k // 32)[r0:r1]
            scale = torch.exp2(sf.to(torch.float64) - 127.0)
            vals = v[r0:r1].to(torch.float64).reshape(r1 - r0, self.k // 32, 32)
            return (vals * scale[..., None]).reshape(r1 - r0, self.k)

        w64 = deq(self.weight, self.weight_lin_sf, 0, self.n)
        e_max, e_sum = 0.0, 0.0
        for r0 in range(0, self.m, rows_per_chunk):
            r1 = min(self.m, r0 + rows_per_chunk)
            ref = deq(self.act, self.act_lin_sf, r0, r1) @ w64.T
            e = (out[r0:r1].to(torch.float64) - ref).abs()
            e_max = max(e_max, e.max().item())
            e_sum += e.sum().item()
        return e_max, e_sum / (self.m * self.n)


def _debug_gemm_reference(kmod, prob, hpt):
    """The fused kernel's debug_gemm_out output with real (small) producer tensors."""
    m, n = prob.m, prob.n
    hq = n // HEAD_DIM - (2 * HEADS_KV + HEADS_INDEX + 1)
    runner = kmod.M3QkvFusedRunner(
        num_heads_q=hq,
        num_heads_kv=HEADS_KV,
        num_heads_index=HEADS_INDEX,
        heads_per_tile=hpt,
        debug_gemm_out=True,
    )
    pages = (m + 127) // 128 + 2
    data = torch.zeros((pages, 2, HEADS_KV, 128, 64), dtype=torch.uint8, device="cuda")
    scale = torch.zeros((pages, 2, HEADS_KV, 128, 8), dtype=torch.uint8, device="cuda")
    index = torch.zeros((pages, 1, 128, 128), dtype=torch.uint8, device="cuda")
    q = torch.zeros((m, hq, HEAD_DIM), dtype=torch.uint8, device="cuda")
    iq = torch.zeros((m, HEADS_INDEX, HEAD_DIM), dtype=torch.uint8, device="cuda")
    slots = torch.arange(m, dtype=torch.int32, device="cuda")
    rope = torch.zeros((m + 8, 2, 32), dtype=torch.float32, device="cuda")
    kvs = torch.tensor([1.0, 0.05, 0.03], dtype=torch.float32, device="cuda")
    normw = torch.ones(4 * HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    g = torch.empty(m, n, dtype=torch.bfloat16, device="cuda")
    args = runner.make_args(
        prob.act,
        prob.act_sf,
        prob.weight,
        prob.weight_sf,
        prob.alpha,
        q.view(torch.float8_e4m3fn),
        iq.view(torch.float8_e4m3fn),
        data,
        scale,
        index.view(torch.float8_e4m3fn),
        slots,
        kvs,
        rope,
        slots,
        normw,
        1e-6,
        gemm_out=g,
    )
    runner.run_args(args)
    return g


def gate(name, m, log=print, hpt=None):
    kmod, pg = _pkg()
    n, k = SHAPES[name][:2]
    hpt = hpt_for(name, m) if hpt is None else hpt
    prob = Problem(n, k, m)
    tag = f"GATE {name} N={n} K={k} M={m} hpt={hpt}"
    layout_ok = torch.equal(
        prob.weight_sf.view(torch.uint8).reshape(-1),
        prob.weight_quant_sf.view(torch.uint8).reshape(-1),
    )
    gemm = pg.get_gemm(n, hpt)
    reason = gemm.check(prob.act, prob.act_sf, prob.weight, prob.weight_sf, prob.alpha)
    if reason:
        log(f"{tag} INELIGIBLE: {reason}")
        return False

    outs = [gemm(prob.act, prob.act_sf, prob.weight, prob.weight_sf, prob.alpha) for _ in range(3)]
    torch.cuda.synchronize()
    det = all(torch.equal(_bits(outs[0]), _bits(o)) for o in outs[1:])

    out = torch.empty(m, n, dtype=torch.bfloat16, device="cuda")
    args = gemm.make_args(prob.act, prob.act_sf, prob.weight, prob.weight_sf, prob.alpha, out)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        gemm.run_args(args)  # warm-up outside capture
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        gemm.run_args(args)
    out.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    graph_ok = torch.equal(_bits(out), _bits(outs[0]))

    dbg = _debug_gemm_reference(kmod, prob, hpt)
    torch.cuda.synchronize()
    dbg_ok = torch.equal(_bits(dbg), _bits(outs[0]))

    cut = prob.cutlass()
    torch.cuda.synchronize()
    same = (_bits(outs[0]) == _bits(cut)).float().mean().item()
    c_max, c_mean = prob.fp64_err(outs[0])
    u_max, u_mean = prob.fp64_err(cut)
    band_ok = c_max <= 1.5 * u_max and c_mean <= 1.1 * u_mean
    finite = bool(torch.isfinite(outs[0].float()).all().item())
    log(
        f"{tag} module_sf_layout==mxfp8_quantize(w,True)={'OK' if layout_ok else 'FAIL'} "
        f"determinism(3)={'OK' if det else 'FAIL'} graph==eager={'OK' if graph_ok else 'FAIL'} "
        f"==debug_gemm_out={'OK' if dbg_ok else 'FAIL'} finite={'OK' if finite else 'FAIL'}"
    )
    log(
        f"{tag} fp64 err clc max={c_max:.4e} mean={c_mean:.4e} | cutlass max={u_max:.4e} "
        f"mean={u_mean:.4e} | bitwise==cutlass {same * 100:.4f}% | band={'OK' if band_ok else 'FAIL'}"
    )
    return layout_ok and det and graph_ok and dbg_ok and band_ok and finite


def dispatch(log=print, name="o_proj"):
    """MXFP8 Linear through the real loader: engagement, slot, threshold and kill switch."""
    mod = _dispatch_module()
    if mod is None:
        log("DISPATCH SKIP: tensorrt_llm._torch.modules.m3_ctx_clc_gemm not importable")
        return True
    from tensorrt_llm._torch.modules import linear as linear_module
    from tensorrt_llm._torch.modules.linear import Linear, MXFP8LinearMethod
    from tensorrt_llm.logger import logger as trt_logger
    from tensorrt_llm.models.modeling_utils import QuantConfig
    from tensorrt_llm.quantization.mode import QuantAlgo

    n, k = SHAPES[name][:2]
    ok = all(
        mod._CTX_CLC_GEMM_SHAPES.get((sn, sk)) == (a, b) for sn, sk, a, b in SHAPES.values()
    ) and len(mod._CTX_CLC_GEMM_SHAPES) == len(SHAPES)
    ok &= mod._CTX_CLC_GEMM_MIN_ROWS == MIN_ROWS and mod._CTX_CLC_GEMM_WIDE_ROWS == WIDE_ROWS
    log(f"DISPATCH table matches gate shapes: {'OK' if ok else 'FAIL'}")
    if not mod.ctx_clc_gemm_enabled():
        log(f"DISPATCH FAIL: not enabled (knob={mod._CTX_CLC_GEMM})")
        return False

    g = torch.Generator(device="cuda").manual_seed(1)
    w = (torch.randn(n, k, device="cuda", generator=g) * 0.02).to(torch.bfloat16)
    w_e4m3, w_lin_sf = torch.ops.trtllm.mxfp8_quantize(w, False)
    lin = Linear(
        in_features=k,
        out_features=n,
        bias=False,
        dtype=torch.bfloat16,
        quant_config=QuantConfig(quant_algo=QuantAlgo.MXFP8, group_size=32),
    ).cuda()
    lin.load_weights(
        [
            {
                "weight": w_e4m3.cpu(),
                "weight_scale_inv": w_lin_sf.view(torch.uint8).reshape(n, k // 32).cpu(),
            }
        ]
    )
    assert isinstance(lin.quant_method, MXFP8LinearMethod) and lin.quant_method.use_cutlass

    def cutlass(act, act_sf):
        return torch.ops.trtllm.mxfp8_mxfp8_gemm(
            act, act_sf, lin.weight, lin.weight_scale, lin._mxfp8_alpha, torch.bfloat16
        )

    # The in-model path always uses the importable package (not M3QKV_PKG).
    pg = importlib.import_module("tensorrt_llm._torch.kernels.rubin.m3_qkv_fusion.plain_gemm")
    for m in (2048, 1000):
        hpt = hpt_for(name, m)
        x = torch.randn(m, k, device="cuda", generator=g, dtype=torch.bfloat16)
        act, act_sf = torch.ops.trtllm.mxfp8_quantize(x, True)
        clc = pg.get_gemm(n, hpt)(act, act_sf, lin.weight, lin.weight_scale, lin._mxfp8_alpha)
        ref_cut = cutlass(act, act_sf)
        key = f"m3_ctx_clc_gemm_taken_{n}_{k}_{hpt}"
        trt_logger._appeared_keys.discard(key)
        got = lin(x)
        engaged = key in trt_logger._appeared_keys
        want = clc if m >= MIN_ROWS else ref_cut
        res = torch.equal(_bits(got), _bits(want)) and engaged == (m >= MIN_ROWS)
        log(
            f"DISPATCH {name} M={m}: engaged={engaged} out=={'clc' if m >= MIN_ROWS else 'cutlass'} "
            f"{'OK' if res else 'FAIL'}"
        )
        ok &= res

        # The pre-quantized slot is consumed by the CLC path (same act / act_sf as CUTLASS would take).
        linear_module.set_prequantized_mxfp8(x, act, act_sf)
        got_slot = lin(x)
        res = (
            torch.equal(_bits(got_slot), _bits(want))
            and linear_module._PREQUANTIZED_MXFP8[0] is None
        )
        log(f"DISPATCH {name} M={m}: prequantized slot {'OK' if res else 'FAIL'}")
        ok &= res

        # Kill switch: the unchanged CUTLASS call, bitwise.
        saved = mod._CTX_CLC_GEMM, mod._platform_ok
        mod._CTX_CLC_GEMM = False
        try:
            killed = lin(x)
        finally:
            mod._CTX_CLC_GEMM, mod._platform_ok = saved
        res = torch.equal(_bits(killed), _bits(ref_cut))
        log(f"DISPATCH {name} M={m}: TRTLLM_CTX_CLC_GEMM=0 == cutlass {'OK' if res else 'FAIL'}")
        ok &= res

        # The registered custom op (eager and compiled with a dynamic M) matches the direct call.
        op_out = torch.ops.trtllm.m3_ctx_clc_mxfp8_gemm(
            act, act_sf, lin.weight, lin.weight_scale, lin._mxfp8_alpha, hpt
        )
        res = torch.equal(_bits(op_out), _bits(clc))
        if m == 2048:
            compiled = torch.compile(
                lambda a, s: torch.ops.trtllm.m3_ctx_clc_mxfp8_gemm(
                    a, s, lin.weight, lin.weight_scale, lin._mxfp8_alpha, hpt
                ),
                fullgraph=True,
                dynamic=True,
            )
            res &= torch.equal(_bits(compiled(act, act_sf)), _bits(clc))
        log(
            f"DISPATCH {name} M={m}: custom op (eager{', compiled' if m == 2048 else ''}) {'OK' if res else 'FAIL'}"
        )
        ok &= res
    torch.cuda.synchronize()
    return bool(ok)


def timing(name, m, log=print, reps=21, host_calls=100):
    _, pg = _pkg()
    from tensorrt_llm._torch.custom_ops.torch_custom_ops import MXFP8GemmRunner

    n, k = SHAPES[name][:2]
    hpt = hpt_for(name, m)
    prob = Problem(n, k, m)
    flush = torch.empty(512 << 20, dtype=torch.uint8, device="cuda")

    def med(fn, cold=True):
        fn()
        torch.cuda.synchronize()
        ts = []
        for _ in range(reps):
            if cold:
                flush.zero_()
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record()
            fn()
            e1.record()
            torch.cuda.synchronize()
            ts.append(e0.elapsed_time(e1) * 1e3)
        return sorted(ts)[reps // 2]

    def host(fn):
        fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(host_calls):
            fn()
        t1 = time.perf_counter()
        torch.cuda.synchronize()
        return (t1 - t0) * 1e6 / host_calls

    runner = MXFP8GemmRunner(torch.bfloat16)
    inputs = [prob.act, prob.act_sf, prob.weight, prob.weight_sf, prob.alpha]
    best = None
    for t in range(runner.mxfp8_gemm_runner.get_num_configs()):
        try:
            us = med(lambda t=t: runner(inputs=inputs, tactic=t))
        except Exception as exc:  # noqa: BLE001
            log(f"TIME {name} M={m} cutlass tactic {t} failed: {type(exc).__name__}")
            continue
        if best is None or us < best[0]:
            best = (us, t)
    gemm = pg.get_gemm(n, hpt)
    out = torch.empty(m, n, dtype=torch.bfloat16, device="cuda")
    args = gemm.make_args(prob.act, prob.act_sf, prob.weight, prob.weight_sf, prob.alpha, out)
    t_clc = med(lambda: gemm.run_args(args))
    t_clc_warm = med(lambda: gemm.run_args(args), cold=False)
    log(
        f"TIME {name} N={n} K={k} M={m} hpt={hpt}: cutlass_best t{best[1]}={best[0]:.1f} us | "
        f"clc cold={t_clc:.1f} us warm={t_clc_warm:.1f} us | delta={best[0] - t_clc:+.1f} us "
        f"({best[0] / t_clc:.3f}x)"
    )

    calls = {
        "clc prebuilt args (run_args)": lambda: gemm.run_args(args),
        "clc handle __call__ (check + make_args + run)": lambda: gemm(
            prob.act, prob.act_sf, prob.weight, prob.weight_sf, prob.alpha
        ),
        "cutlass mxfp8_mxfp8_gemm op": prob.cutlass,
    }
    mod = _dispatch_module()
    if mod is not None:
        calls["dispatcher try_ctx_clc_mxfp8_gemm (in-model path)"] = (
            lambda: mod.try_ctx_clc_mxfp8_gemm(
                prob.act, prob.act_sf, prob.weight, prob.weight_sf, prob.alpha, torch.bfloat16
            )
        )
        calls["custom op trtllm::m3_ctx_clc_mxfp8_gemm"] = (
            lambda: torch.ops.trtllm.m3_ctx_clc_mxfp8_gemm(
                prob.act, prob.act_sf, prob.weight, prob.weight_sf, prob.alpha, hpt
            )
        )
    for label, fn in calls.items():
        log(
            f"TIME {name} M={m} host/call {label}: {host(fn):.1f} us (wall of {host_calls} calls, no sync)"
        )


@pytest.mark.parametrize("m", M_LIST)
@pytest.mark.parametrize("name", list(SHAPES))
def test_m3_ctx_clc_gemm_gate(name, m):
    assert gate(name, m)


def test_m3_ctx_clc_gemm_dispatch():
    assert dispatch()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", default=",".join(SHAPES))
    ap.add_argument("--m", default=",".join(str(v) for v in M_LIST))
    ap.add_argument("--hpt", type=int, default=None, help="override heads per tile in the gate")
    ap.add_argument("--timing", default="", help="comma list of M for timing")
    ap.add_argument("--no-gate", action="store_true")
    ap.add_argument("--no-dispatch", action="store_true")
    a = ap.parse_args()
    names = [s for s in a.shapes.split(",") if s]
    all_ok = True
    plog = lambda s: print(s, flush=True)  # noqa: E731
    if not a.no_gate:
        for nm in names:
            for mm in [int(v) for v in a.m.split(",") if v]:
                all_ok &= gate(nm, mm, log=plog, hpt=a.hpt)
    if not a.no_dispatch:
        all_ok &= dispatch(log=plog)
    for nm in names:
        for mm in [int(v) for v in a.timing.split(",") if v]:
            timing(nm, mm, log=plog)
    print(f"GATE ALL {'PASS' if all_ok else 'FAIL'}", flush=True)
