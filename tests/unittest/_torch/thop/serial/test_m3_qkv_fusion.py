# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Gate for the SM107 fused MiniMax-M3 QKV MXFP8 GEMM + producer epilogue.

The fused kernel's debug handle writes its own bf16 GEMM output G; the stock
``minimax_m3_nvfp4_qkv_indexer_norm_rope_kv_insert`` producer run on G is the
reference, and the fused outputs (q, index-q, K/V NVFP4 data, K/V scales,
index-K) must match it bitwise, including the cache bytes it must not touch.

Also runnable as a script (``python3 test_m3_qkv_fusion.py --m 16384,300
--timing``); ``M3QKV_PKG`` may name a directory holding an ``m3_qkv_fusion``
package to test instead of the in-tree one.
"""

import argparse
import importlib
import os
import sys

import pytest
import torch

# tensorrt_llm is imported inside the functions that need it, so that _kernel_module() first sets
# CUTE_DSL_ARCH (read by the CuTe DSL when it is imported) and puts any M3QKV_PKG on sys.path.

HEADS_Q, HEADS_KV, HEADS_INDEX, HEAD_DIM, K_DIM = 32, 2, 2, 128, 6144
N_DIM = (HEADS_Q + 2 * HEADS_KV + HEADS_INDEX + 1) * HEAD_DIM
EPS = 1e-6
SENTINEL_DATA, SENTINEL_SCALE, SENTINEL_INDEX, SENTINEL_Q = 0xA5, 0x5A, 0x3C, 0x77


def _is_sm107():
    return torch.cuda.is_available() and torch.cuda.get_device_capability() == (10, 7)


pytestmark = pytest.mark.skipif(not _is_sm107(), reason="SM107 (Rubin) required")


def _kernel_module():
    os.environ.setdefault("CUTE_DSL_ARCH", "sm_107a")
    pkg = os.environ.get("M3QKV_PKG")
    if pkg:
        sys.path.insert(0, pkg)
        return importlib.import_module("m3_qkv_fusion.kernel")
    return importlib.import_module("tensorrt_llm._torch.kernels.rubin.m3_qkv_fusion.kernel")


def _rope_cache(max_positions, rotary_dim=64, base=5_000_000.0):
    positions = torch.arange(max_positions, dtype=torch.float32, device="cuda")
    inv_freq = 1.0 / (
        base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32, device="cuda") / rotary_dim)
    )
    freq = torch.outer(positions, inv_freq)
    return torch.stack((freq.cos(), freq.sin()), dim=1).contiguous()


class Problem:
    def __init__(self, m, seed=0):
        import tensorrt_llm  # noqa: F401  (registers torch.ops.trtllm)

        g = torch.Generator(device="cuda").manual_seed(seed)
        self.m = m
        x = torch.randn(m, K_DIM, device="cuda", generator=g, dtype=torch.bfloat16)
        w = (torch.randn(N_DIM, K_DIM, device="cuda", generator=g) * 0.02).to(torch.bfloat16)
        self.act, self.act_sf = torch.ops.trtllm.mxfp8_quantize(x, True)
        self.weight, self.weight_sf = torch.ops.trtllm.mxfp8_quantize(w, True)
        self.act_lin_sf = torch.ops.trtllm.mxfp8_quantize(x, False)[1]
        self.weight_lin_sf = torch.ops.trtllm.mxfp8_quantize(w, False)[1]
        self.alpha = torch.ones(1, dtype=torch.float32, device="cuda")
        self.norm_w = [
            (torch.randn(HEAD_DIM, device="cuda", generator=g) * 0.2).to(torch.bfloat16)
            for _ in range(4)
        ]
        self.norm_cat = torch.cat(self.norm_w)
        max_pos = m + 4096
        self.rope = _rope_cache(max_pos)
        self.positions = torch.randint(
            0, max_pos, (m,), device="cuda", generator=g, dtype=torch.int32
        )
        self.num_pages = (m + 127) // 128 + 2
        slots = torch.randperm(self.num_pages * 128, device="cuda", generator=g)[:m].to(torch.int32)
        drop = torch.rand(m, device="cuda", generator=g) < 0.03
        slots[drop] = -1
        beyond = torch.rand(m, device="cuda", generator=g) < 0.01
        slots[beyond] = self.num_pages * 128 + 5  # page >= num_pages: skipped
        self.slots = slots
        self.kv_quant_scale = torch.tensor([1.0, 0.05, 0.03], dtype=torch.float32, device="cuda")

    def caches(self):
        # Strided pools (every other page) like the producer's own test.
        p = self.num_pages
        data = torch.full(
            (2 * p, 2, HEADS_KV, 128, 64), SENTINEL_DATA, dtype=torch.uint8, device="cuda"
        )[::2]
        scale = torch.full(
            (2 * p, 2, HEADS_KV, 128, 8), SENTINEL_SCALE, dtype=torch.uint8, device="cuda"
        )[::2]
        index = torch.full((2 * p, 1, 128, 128), SENTINEL_INDEX, dtype=torch.uint8, device="cuda")[
            ::2
        ]
        return data, scale, index.view(torch.float8_e4m3fn)

    def outputs(self):
        q = torch.full((self.m, HEADS_Q, HEAD_DIM), SENTINEL_Q, dtype=torch.uint8, device="cuda")
        iq = torch.full(
            (self.m, HEADS_INDEX, HEAD_DIM), SENTINEL_Q, dtype=torch.uint8, device="cuda"
        )
        return q.view(torch.float8_e4m3fn), iq.view(torch.float8_e4m3fn)

    def reference(self, gemm_bf16):
        data, scale, index = self.caches()
        q, iq = torch.ops.trtllm.minimax_m3_nvfp4_qkv_indexer_norm_rope_kv_insert(
            gemm_bf16,
            data,
            scale,
            index,
            self.slots,
            self.kv_quant_scale,
            HEADS_Q,
            HEADS_KV,
            HEADS_INDEX,
            HEAD_DIM,
            64,
            EPS,
            *self.norm_w,
            self.rope,
            self.positions,
        )
        return q, iq, data, scale, index

    def fused_args(self, runner, outs, caches, gemm_out=None):
        q, iq = outs
        data, scale, index = caches
        return runner.make_args(
            self.act,
            self.act_sf,
            self.weight,
            self.weight_sf,
            self.alpha,
            q,
            iq,
            data,
            scale,
            index,
            self.slots,
            self.kv_quant_scale,
            self.rope,
            self.positions,
            self.norm_cat,
            EPS,
            gemm_out=gemm_out,
        )

    def run_fused(self, runner):
        outs, caches = self.outputs(), self.caches()
        runner.run_args(self.fused_args(runner, outs, caches))
        return (*outs, *caches)

    def run_debug(self, runner):
        g = torch.empty(self.m, N_DIM, dtype=torch.bfloat16, device="cuda")
        runner.run_args(self.fused_args(runner, self.outputs(), self.caches(), gemm_out=g))
        return g

    def fp64_reference(self):
        def deq(v, sf_lin):
            rows = v.shape[0]
            sf = (
                sf_lin.view(torch.uint8)
                .reshape(-1)[: rows * (K_DIM // 32)]
                .reshape(rows, K_DIM // 32)
            )
            scale = torch.exp2(sf.to(torch.float64) - 127.0)
            return (v.to(torch.float64).reshape(rows, K_DIM // 32, 32) * scale[..., None]).reshape(
                rows, K_DIM
            )

        return deq(self.act, self.act_lin_sf) @ deq(self.weight, self.weight_lin_sf).T


NAMES = ("q", "index_q", "kv_data", "kv_scale", "index_k")


def _bytes(t):
    return t.view(torch.uint8)


def compare(got, ref):
    res = {}
    for name, a, b in zip(NAMES, got, ref):
        a8, b8 = _bytes(a), _bytes(b)
        res[name] = (torch.equal(a8, b8), int((a8 != b8).sum().item()))
    return res


def gate(m, cfg, log=print):
    k = _kernel_module()
    prob = Problem(m)
    fused = k.M3QkvFusedRunner(**cfg)
    debug = k.M3QkvFusedRunner(**cfg, debug_gemm_out=True)
    g = prob.run_debug(debug)
    torch.cuda.synchronize()
    ref = prob.reference(g)
    got = prob.run_fused(fused)
    torch.cuda.synchronize()
    res = compare(got, ref)
    ok = all(v[0] for v in res.values())
    log(
        f"GATE m={m} bitwise "
        + " ".join(f"{n}={'OK' if v[0] else f'DIFF({v[1]})'}" for n, v in res.items())
    )

    runs = [prob.run_fused(fused) for _ in range(3)]
    det = all(compare(r, got)[n][0] for r in runs for n in NAMES)
    g2 = prob.run_debug(debug)
    det = det and torch.equal(g.view(torch.int16), g2.view(torch.int16))
    log(f"GATE m={m} determinism(3 runs + debug)={'OK' if det else 'FAIL'}")

    outs, caches = prob.outputs(), prob.caches()
    args = prob.fused_args(fused, outs, caches)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        fused.run_args(args)  # warm-up outside capture
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fused.run_args(args)
    for t, v in zip((*outs, *caches), (0, 0, SENTINEL_DATA, SENTINEL_SCALE, SENTINEL_INDEX)):
        _bytes(t).fill_(v)
    graph.replay()
    torch.cuda.synchronize()
    gres = compare((*outs, *caches), got)
    graph_ok = all(v[0] for v in gres.values())
    log(f"GATE m={m} cuda_graph_replay==eager={'OK' if graph_ok else 'FAIL ' + str(gres)}")

    ref64 = prob.fp64_reference()
    cut = torch.ops.trtllm.mxfp8_mxfp8_gemm(
        prob.act, prob.act_sf, prob.weight, prob.weight_sf, prob.alpha, torch.bfloat16
    )
    e_f = (g.to(torch.float64) - ref64).abs()
    e_c = (cut.to(torch.float64) - ref64).abs()
    same = (g.view(torch.int16) == cut.view(torch.int16)).float().mean().item()
    band_ok = (
        e_f.max().item() <= 1.5 * e_c.max().item() and e_f.mean().item() <= 1.1 * e_c.mean().item()
    )
    log(
        f"GATE m={m} fp64 err fused max={e_f.max().item():.4e} mean={e_f.mean().item():.4e} | "
        f"cutlass max={e_c.max().item():.4e} mean={e_c.mean().item():.4e} | "
        f"bitwise==cutlass {same * 100:.3f}% | band={'OK' if band_ok else 'FAIL'}"
    )
    del ref64
    return ok and det and graph_ok and band_ok


def timing(m, cfgs, log=print, reps=21):
    k = _kernel_module()
    from tensorrt_llm._torch.custom_ops.torch_custom_ops import MXFP8GemmRunner

    prob = Problem(m)
    log(f"TIME SMs={torch.cuda.get_device_properties(0).multi_processor_count}")
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

    runner = MXFP8GemmRunner(torch.bfloat16)
    inputs = [prob.act, prob.act_sf, prob.weight, prob.weight_sf, prob.alpha]
    best = None
    for t in range(runner.mxfp8_gemm_runner.get_num_configs()):
        try:
            us = med(lambda t=t: runner(inputs=inputs, tactic=t))
        except Exception as exc:  # noqa: BLE001
            log(f"TIME m={m} cutlass tactic {t} failed: {type(exc).__name__}")
            continue
        if best is None or us < best[0]:
            best = (us, t)
    log(f"TIME m={m} cutlass_best tactic={best[1]} cold={best[0]:.1f} us")
    g = runner(inputs=inputs, tactic=best[1])
    data, scale, index = prob.caches()
    prod = lambda: torch.ops.trtllm.minimax_m3_nvfp4_qkv_indexer_norm_rope_kv_insert(  # noqa: E731
        g, data, scale, index, prob.slots, prob.kv_quant_scale, HEADS_Q, HEADS_KV, HEADS_INDEX,
        HEAD_DIM, 64, EPS, *prob.norm_w, prob.rope, prob.positions,
    )  # fmt: skip
    t_prod = med(prod)
    both = lambda: (runner(inputs=inputs, tactic=best[1]), prod())  # noqa: E731
    t_both = med(both)
    log(
        f"TIME m={m} producer cold={t_prod:.1f} us; cutlass_best+producer back-to-back cold={t_both:.1f} us"
    )
    for cfg in cfgs:
        fused = k.M3QkvFusedRunner(**cfg)
        args = prob.fused_args(fused, prob.outputs(), prob.caches())
        t_f = med(lambda: fused.run_args(args))
        t_w = med(lambda: fused.run_args(args), cold=False)
        log(
            f"TIME m={m} fused {cfg}: cold={t_f:.1f} us warm={t_w:.1f} us  (vs {t_both:.1f}: {t_both / t_f:.3f}x)"
        )
        if os.environ.get("M3QKV_TIME_DEBUG"):
            dbg = k.M3QkvFusedRunner(**cfg, debug_gemm_out=True)
            gout = torch.empty(m, N_DIM, dtype=torch.bfloat16, device="cuda")
            dargs = prob.fused_args(dbg, prob.outputs(), prob.caches(), gemm_out=gout)
            log(f"TIME m={m} debug-gemm-out {cfg}: cold={med(lambda: dbg.run_args(dargs)):.1f} us")


def _parse_cfg(text):
    keys = {
        "hpt": "heads_per_tile",
        "acc": "num_acc_stage",
        "cl": "cluster_shape_mn",
        "pdl": "enable_pdl",
        "rm": "raster_along_m",
        "sw": "swizzle_size",
        "mm": "mma_m",
        "mn": "mma_n",
        "ew": "epi_wgs",
    }
    cfg = {}
    for item in filter(None, text.split(",")):
        key, val = item.split("=")
        if key == "cl":
            cfg[keys[key]] = tuple(int(v) for v in val.split("x"))
        elif key in ("pdl", "rm"):
            cfg[keys[key]] = bool(int(val))
        else:
            cfg[keys[key]] = int(val)
    return cfg


@pytest.mark.parametrize("m", [16384, 8200, 1000, 300])
def test_m3_qkv_fusion_bitwise(m):
    assert gate(m, {})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", default="16384,8200,1000,300")
    ap.add_argument(
        "--cfg", action="append", default=[], help="compile kwargs, e.g. hpt=2,cl=1x2,acc=2,pdl=0"
    )
    ap.add_argument("--timing", default="", help="comma list of M for timing")
    ap.add_argument("--no-gate", action="store_true")
    a = ap.parse_args()
    cfgs = [_parse_cfg(c) for c in a.cfg] or [{}]
    all_ok = True
    if not a.no_gate:
        for mm in [int(v) for v in a.m.split(",") if v]:
            for cfg in cfgs:
                print(f"GATE cfg={cfg}", flush=True)
                all_ok &= gate(mm, cfg, log=lambda s: print(s, flush=True))
    for mm in [int(v) for v in a.timing.split(",") if v]:
        timing(mm, cfgs, log=lambda s: print(s, flush=True))
    print(f"GATE ALL {'PASS' if all_ok else 'FAIL'}", flush=True)
