# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A/C benchmark for Kimi K3 KDA single-sequence CTX prefill input padding.

A = baseline: post-conv, beta.float(), five F.pad calls in the dispatch.
C = producer: post-conv writes chunk-padded Q/K/V and fp32 beta; g unpadded.

Every variant runs the same real trtllm::kda_prefill op. Reports per-variant
GPU time for the input-prep segment and for prep + op, peak allocated memory
over prep + op, and checks that output and state match A bit for bit.
"""

import argparse
import json

import torch
import torch.nn.functional as F

from tensorrt_llm._torch.modules.kimi_kda._kda_kernels import KDAKernelDispatch, fused_kda_post_conv

D = 128
CHUNK = 64


def make_dispatch() -> KDAKernelDispatch:
    dispatch = KDAKernelDispatch(use_optimized_decode=False, use_optimized_verify=False)
    assert dispatch.prefill_kernel_path == "optimized"
    return dispatch


def make_inputs(T, H, seed=0):
    g_ = torch.Generator(device="cuda").manual_seed(seed)
    packed = torch.randn(3 * H * D, T, device="cuda", dtype=torch.bfloat16, generator=g_)
    g = torch.randn(1, T, H, D, device="cuda", dtype=torch.bfloat16, generator=g_)
    # beta is a narrow slice of a wider fused projection, as in the mixer.
    bfa = torch.randn(1, T, D + H + 8, device="cuda", dtype=torch.bfloat16, generator=g_)
    beta_src = bfa[..., D : D + H]
    a_log = torch.randn(H, device="cuda", dtype=torch.float32, generator=g_) * 0.1
    dt_bias = torch.randn(H * D, device="cuda", dtype=torch.float32, generator=g_) * 0.1
    return packed, g, beta_src, a_log, dt_bias


def prep(variant, packed, g, beta_src, H, T):
    """Input preparation exactly as each variant would run it."""
    if variant == "C":
        Tp = T + CHUNK - T % CHUNK
        q, k, v, beta = fused_kda_post_conv(packed, H, D, padded_tokens=Tp, beta=beta_src)
        return q, k, v, g, beta
    q, k, v = fused_kda_post_conv(packed, H, D)
    beta = beta_src.float()
    pad = CHUNK - T % CHUNK
    assert variant == "A"
    return (
        F.pad(q, (0, 0, 0, 0, 0, pad)),
        F.pad(k, (0, 0, 0, 0, 0, pad)),
        F.pad(v, (0, 0, 0, 0, 0, pad)),
        F.pad(g, (0, 0, 0, 0, 0, pad)),
        F.pad(beta, (0, 0, 0, pad)),
    )


def run_full(variant, dispatch, packed, g, beta_src, a_log, dt_bias, pool, slots, cu, H, T):
    """Prep + op. A hands unpadded inputs to the dispatch, which pads them."""
    if variant == "C":
        q, k, v, g_in, beta = prep("C", packed, g, beta_src, H, T)
    else:
        q, k, v = fused_kda_post_conv(packed, H, D)
        g_in, beta = g, beta_src.float()
    pool.zero_()
    out, _ = dispatch.prefill_chunk_kda(
        q=q,
        k=k,
        v=v,
        g=g_in,
        beta=beta,
        A_log=a_log,
        dt_bias=dt_bias,
        scale=D**-0.5,
        initial_state=None,
        safe_gate=True,
        lower_bound=-5.0,
        cu_seqlens=cu,
        state_pool=pool,
        state_indices=slots,
        single_sequence_length=T,
    )
    return out


def bits(t):
    """Bitwise view so +0 and -0 (and NaN payloads) compare as distinct."""
    return t.view(torch.int16 if t.element_size() == 2 else torch.int32)


def time_fn(fn, iters, warmup=3):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for s, e in zip(starts, ends):
        s.record()
        fn()
        e.record()
    torch.cuda.synchronize()
    ts = sorted(s.elapsed_time(e) * 1e3 for s, e in zip(starts, ends))
    return ts[len(ts) // 2]


def peak_mem(fn):
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    fn()
    torch.cuda.synchronize()
    return (torch.cuda.max_memory_allocated() - base) / 2**20


def bench_one(dispatch, H, T):
    """Check parity and time A/C for one (heads, tokens) point."""
    assert T % CHUNK
    packed, g, beta_src, a_log, dt_bias = make_inputs(T, H)
    pools = {v: torch.zeros(4, H, D, D, device="cuda", dtype=torch.float32) for v in "AC"}
    slots = torch.tensor([2], device="cuda", dtype=torch.int32)
    cu = torch.tensor([0, T], device="cuda", dtype=torch.int64)

    outs = {}
    for v in "AC":
        o = run_full(v, dispatch, packed, g, beta_src, a_log, dt_bias, pools[v], slots, cu, H, T)
        outs[v] = (o.clone(), pools[v].clone())
    pa, pc = prep("A", packed, g, beta_src, H, T), prep("C", packed, g, beta_src, H, T)
    prep_equal = {
        name: torch.equal(bits(a), bits(c))
        for name, a, c in zip(
            ("q", "k", "v", "beta"),
            (pa[0], pa[1], pa[2], pa[4]),
            (pc[0], pc[1], pc[2], pc[4]),
        )
    }
    assert all(prep_equal.values()), f"H={H} T={T} producer vs F.pad mismatch: {prep_equal}"
    del pa, pc
    output_exact = torch.equal(bits(outs["C"][0]), bits(outs["A"][0]))
    state_exact = torch.equal(bits(outs["C"][1]), bits(outs["A"][1]))
    exact = output_exact and state_exact
    assert exact, f"H={H} T={T} full-op bit mismatch: output={output_exact} state={state_exact}"
    assert outs["A"][0].shape[1] == T and outs["C"][0].shape[1] == T

    iters = 50 if T <= 16384 else 20
    r = {"H": H, "T": T, "exact": exact}
    for v in "AC":
        r[f"prep_{v}"] = time_fn(lambda v=v: prep(v, packed, g, beta_src, H, T), iters)
        r[f"full_{v}"] = time_fn(
            lambda v=v: run_full(
                v, dispatch, packed, g, beta_src, a_log, dt_bias, pools[v], slots, cu, H, T
            ),
            iters,
        )
        r[f"mem_{v}"] = peak_mem(
            lambda v=v: run_full(
                v, dispatch, packed, g, beta_src, a_log, dt_bias, pools[v], slots, cu, H, T
            )
        )
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--heads", type=int, nargs="+", default=[96, 12, 6])
    ap.add_argument(
        "--tokens", type=int, nargs="+", default=[1000, 4000, 8100, 16050, 32700, 65500]
    )
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = False
    dispatch = make_dispatch()
    rows = []
    print(f"GPU: {torch.cuda.get_device_name()}  (times = median us, mem = MiB peak over call)")
    hdr = (
        f"{'H':>3} {'T':>6} | {'prep A':>8} {'prep C':>8} | "
        f"{'full A':>9} {'full C':>9} | {'C vs A':>7} | "
        f"{'mem A':>7} {'mem C':>7} | exact"
    )
    print(hdr)
    for H in args.heads:
        for T in args.tokens:
            r = bench_one(dispatch, H, T)
            rows.append(r)
            print(
                f"{H:>3} {T:>6} | {r['prep_A']:>8.1f} {r['prep_C']:>8.1f} | "
                f"{r['full_A']:>9.1f} {r['full_C']:>9.1f} | "
                f"{r['full_A'] / r['full_C']:>6.3f}x | "
                f"{r['mem_A']:>7.0f} {r['mem_C']:>7.0f} | {r['exact']}",
                flush=True,
            )
            torch.cuda.empty_cache()
    if args.json:
        with open(args.json, "w") as f:
            json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()
