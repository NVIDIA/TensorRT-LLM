# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Drive `torch.ops.trtllm.mxfp8_mxfp8_gemm` at DeepSeek-V4.1-Flash geometry.

Goal 1.2 owns the dense/norm/embedding/head module, and every FP8 projection in
that module -- plus the shared expert, which this checkpoint stores in FP8 and
not FP4 -- would ride on this one op. Before an entry is written for it, the
questions a contract has to answer are measured here rather than read out of the
source:

  * does it accept every (K, N) this checkpoint actually has, and every runtime
    row count, or only the shapes someone happened to test;
  * what the two swizzled scale buffers must contain and how large they must be,
    since the op checks neither;
  * whether the result depends on the serving tactic cache, which
    `mxfp8_mxfp8_gemm` reads (`useTacticCache=true`) and which is empty in a
    test and warm in a served run -- the difference a contract is most likely to
    get wrong;
  * what a zero-row call does, because dep4 gives ranks zero logical rows.

Reads nothing from the catalog and asserts nothing: it prints measurements. A
negative result here is a complete result.
"""

from __future__ import annotations

import math
import os
import subprocess
import sys

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*

# Every fp32 reference here is the thing a tolerance is chosen against, so it
# must be true fp32 and never tf32.
torch.backends.cuda.matmul.allow_tf32 = False

#: Every distinct FP8-E4M3 (K, N) the TARGET calls, read from the RAW
#: checkpoint's safetensors headers (weights are stored `[out, in]`, so
#: `(K, N) = (in, out)`). Counts are per model.
#:
#: An earlier revision of this table read `staircase-v41/ckpt-mp4/
#: model0-mp4.safetensors` -- the checkpoint already CONVERTED to the reference
#: implementation's four tensor-parallel ranks -- and so recorded three of these
#: as the reference's shard rather than the target's surface. The staircase
#: target replicates every dense projection under attention DP (plan.md lines
#: 30, 78-79, 82), so the raw header is the target's width:
#:     attn.wo_b          (8192, 5120)   shard was (2048, 5120)  RowParallel
#:     attn.wq_b          (1280, 32768)  shard was (1280,  8192)  ColumnParallel
#:     attn.indexer.wq_b  (1280,  4096)  shard was (1280,  1024)  ColumnParallel
V41_FP8_SURFACES = [
    # (K, N, count, what)
    (6144, 25600, 2, "engram.wkv"),
    (5120, 2304, 80, "ffn.shared_experts.w1/w3"),
    (2304, 5120, 40, "ffn.shared_experts.w2"),
    (8192, 5120, 40, "attn.wo_b"),
    (1280, 32768, 40, "attn.wq_b"),
    (5120, 1280, 40, "attn.wq_a"),
    (5120, 512, 40, "attn.wkv"),
    (1280, 4096, 8, "attn.indexer.wq_b"),
]

#: Row counts a served target reaches: single decode, captured decode batches,
#: prefill chunks, and deliberately awkward ones between the 128-row padding
#: boundaries of the swizzled scale layout.
ROW_BUCKETS = [0, 1, 2, 3, 7, 8, 16, 31, 32, 64, 127, 128, 129, 255, 256, 512, 1024, 4096]


def _dequant_mxfp8(q: torch.Tensor, scale_ue8m0: torch.Tensor) -> torch.Tensor:
    """Lift e4m3 values plus one UE8M0 exponent per 32 channels back to fp32.

    Built from the *unswizzled* scales, so this stays independent of whatever
    layout the kernel wants.
    """
    exp = scale_ue8m0.to(torch.int32) - 127
    s = torch.ldexp(torch.ones_like(exp, dtype=torch.float32), exp)
    return q.float().unflatten(-1, (-1, 32)).mul(s.unsqueeze(-1)).flatten(-2)


def _quantize(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (e4m3 values, swizzled scales, plain UE8M0 scales) for a bf16 tensor."""
    q, s_swizzled = torch.ops.trtllm.mxfp8_quantize(x, True, 32)
    _, s_plain = torch.ops.trtllm.mxfp8_quantize(x, False, 32)
    return q, s_swizzled, s_plain


def probe_surfaces() -> None:
    print("=" * 100)
    print("1. every FP8 (K,N) this checkpoint has, at a representative row count")
    print("=" * 100)
    torch.manual_seed(0)
    m = 64
    for k, n, count, what in V41_FP8_SURFACES:
        a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16) * 0.1
        w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.1
        aq, a_sw, a_pl = _quantize(a)
        wq, _, w_pl = _quantize(w)
        w_sw = torch.ops.trtllm.block_scale_interleave(w_pl.view(n, k // 32))
        gs = torch.ones(1, device="cuda", dtype=torch.float32)
        try:
            out = torch.ops.trtllm.mxfp8_mxfp8_gemm(
                aq, a_sw, wq, w_sw.flatten(), gs, torch.bfloat16
            )
        except Exception as exc:  # noqa: BLE001 — the rejection is the measurement
            print(f"  K={k:5d} N={n:5d}  {what:26s} REJECTED: {type(exc).__name__}: {exc}")
            continue
        ref = (
            _dequant_mxfp8(aq, a_pl.view(m, k // 32))
            @ _dequant_mxfp8(wq, w_pl.view(n, k // 32)).t()
        )
        err = (out.float() - ref).abs().max().item()
        scale = ref.abs().max().item()
        print(
            f"  K={k:5d} N={n:5d}  {what:26s} ok  out={tuple(out.shape)} {out.dtype} "
            f"max_abs_err={err:.3e} rel={err / scale:.3e}  ({count} such weights)"
        )


def probe_rows() -> None:
    print()
    print("=" * 100)
    print("2. runtime row buckets, on the widest and narrowest surface")
    print("=" * 100)
    torch.manual_seed(1)
    for k, n in ((6144, 25600), (5120, 512)):
        for m in ROW_BUCKETS:
            a = torch.randn(max(m, 1), k, device="cuda", dtype=torch.bfloat16) * 0.1
            if m == 0:
                a = a[:0]
            w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.1
            try:
                aq, a_sw, a_pl = _quantize(a)
            except Exception as exc:  # noqa: BLE001
                print(f"  K={k:5d} N={n:5d} M={m:5d}  quantize REJECTED: {exc}")
                continue
            wq, _, w_pl = _quantize(w)
            w_sw = torch.ops.trtllm.block_scale_interleave(w_pl.view(n, k // 32))
            gs = torch.ones(1, device="cuda", dtype=torch.float32)
            try:
                out = torch.ops.trtllm.mxfp8_mxfp8_gemm(
                    aq, a_sw, wq, w_sw.flatten(), gs, torch.bfloat16
                )
            except Exception as exc:  # noqa: BLE001
                print(f"  K={k:5d} N={n:5d} M={m:5d}  GEMM REJECTED: {type(exc).__name__}: {exc}")
                continue
            if m == 0:
                print(f"  K={k:5d} N={n:5d} M={m:5d}  ok out={tuple(out.shape)} (zero rows)")
                continue
            ref = (
                _dequant_mxfp8(aq, a_pl.view(m, k // 32))
                @ _dequant_mxfp8(wq, w_pl.view(n, k // 32)).t()
            )
            err = (out.float() - ref).abs().max().item()
            print(
                f"  K={k:5d} N={n:5d} M={m:5d}  ok  act_scale={a_sw.numel():8d} B "
                f"weight_scale={w_sw.numel():9d} B  max_abs_err={err:.3e}"
            )


def probe_scale_layout() -> None:
    print()
    print("=" * 100)
    print("3. what the two scale buffers actually are")
    print("=" * 100)
    for m, k in ((64, 5120), (1, 5120), (129, 1280), (4096, 6144)):
        a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        _, sw, pl = _quantize(a)
        blocks = k // 32
        padded_rows = -(-m // 128) * 128
        padded_blocks = -(-blocks // 4) * 4
        print(
            f"  act   M={m:5d} K={k:5d}: plain={pl.numel():8d} ({m}x{blocks}) "
            f"swizzled={sw.numel():8d} dtype={sw.dtype}  "
            f"128x4-padded={padded_rows * padded_blocks:8d} "
            f"{'MATCH' if sw.numel() == padded_rows * padded_blocks else 'DIFFERS'}"
        )
    for n, k in ((25600, 6144), (512, 5120), (1024, 1280)):
        w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
        _, _, pl = _quantize(w)
        sw = torch.ops.trtllm.block_scale_interleave(pl.view(n, k // 32))
        blocks = k // 32
        padded = -(-n // 128) * 128 * (-(-blocks // 4) * 4)
        print(
            f"  weight N={n:5d} K={k:5d}: plain={pl.numel():8d} "
            f"interleaved={sw.numel():8d} dtype={sw.dtype}  128x4-padded={padded:8d} "
            f"{'MATCH' if sw.numel() == padded else 'DIFFERS'}"
        )


def probe_tactic_cache() -> None:
    print()
    print("=" * 100)
    print("4. THE SERVING TACTIC CACHE -- does the answer depend on it?")
    print("=" * 100)
    print("   mxfp8_mxfp8_gemm passes useTacticCache=true, so a served process")
    print("   whose autotuner has registered a tactic takes a different kernel")
    print("   from a test process whose cache is empty. Both are driven here.")
    torch.manual_seed(2)
    k, n, m = 5120, 1280, 128
    a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16) * 0.1
    w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.1
    aq, a_sw, a_pl = _quantize(a)
    wq, _, w_pl = _quantize(w)
    w_sw = torch.ops.trtllm.block_scale_interleave(w_pl.view(n, k // 32)).flatten()
    gs = torch.ones(1, device="cuda", dtype=torch.float32)

    # Bound method names are snake_case; the C++ declares camelCase. Following
    # the source names gives AttributeError at runtime, not a wrong answer.
    runner = torch.classes.trtllm.MXFP8GemmRunner(torch.bfloat16)
    runner.clear_tactic_cache()
    cold = torch.ops.trtllm.mxfp8_mxfp8_gemm(aq, a_sw, wq, w_sw, gs, torch.bfloat16)
    n_cfg = runner.get_num_configs()
    print(f"   compiled tactics available: {n_cfg}; cache state for this shape now:", end=" ")
    print(runner.get_cached_tactic(m, n, k))

    ref = _dequant_mxfp8(aq, a_pl.view(m, k // 32)) @ _dequant_mxfp8(wq, w_pl.view(n, k // 32)).t()
    print(
        f"   cold (empty cache) vs fp32 reference: max_abs_err={(cold.float() - ref).abs().max():.3e}"
    )

    differing = []
    for idx in range(-1, n_cfg):
        try:
            got = runner.run_gemm(aq, a_sw, wq, w_sw, gs, idx)
        except Exception as exc:  # noqa: BLE001 — an unavailable tactic is a measurement
            print(f"   tactic {idx:3d}: unavailable ({type(exc).__name__})")
            continue
        d = (got.float() - cold.float()).abs().max().item()
        e = (got.float() - ref).abs().max().item()
        if d != 0.0:
            differing.append((idx, d, e))
    print(
        f"   tactics driven: {n_cfg + 1}; tactics whose result differs from cold: {len(differing)}"
    )
    for idx, d, e in differing[:8]:
        print(f"     tactic {idx:3d}: max_abs_diff_vs_cold={d:.3e}  max_abs_err_vs_ref={e:.3e}")

    # And the path a served process actually takes: register a tactic, then call
    # the plain op again and see whether the cache changed the answer.
    if n_cfg:
        for idx in (0, n_cfg - 1):
            runner.clear_tactic_cache()
            runner.register_tactic(m, n, k, idx)
            warm = torch.ops.trtllm.mxfp8_mxfp8_gemm(aq, a_sw, wq, w_sw, gs, torch.bfloat16)
            print(
                f"   registered tactic {idx}: cached={runner.get_cached_tactic(m, n, k)} "
                f"max_abs_diff(warm vs cold)={(warm.float() - cold.float()).abs().max():.3e}"
            )
        runner.clear_tactic_cache()


def probe_unchecked_domains() -> None:
    print()
    print("=" * 100)
    print("5. domains the op does NOT check (a wrapper guard is only earned here)")
    print("=" * 100)
    torch.manual_seed(3)
    k, n, m = 5120, 1280, 64
    a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16) * 0.1
    w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.1
    aq, a_sw, a_pl = _quantize(a)
    wq, _, w_pl = _quantize(w)
    w_sw = torch.ops.trtllm.block_scale_interleave(w_pl.view(n, k // 32)).flatten()
    good = torch.ops.trtllm.mxfp8_mxfp8_gemm(
        aq, a_sw, wq, w_sw, torch.ones(1, device="cuda", dtype=torch.float32), torch.bfloat16
    )

    # The claim is that globalScale is alpha in the epilogue, so the test is
    # linearity against the alpha=1 result. A ratio of medians says nothing:
    # the baseline has near-zero elements and the quotient explodes on them.
    for label, gs, expect in (
        ("global_scale [1.0, 7.0]", torch.tensor([1.0, 7.0], device="cuda"), 1.0),
        ("global_scale [2.0]", torch.tensor([2.0], device="cuda"), 2.0),
        ("global_scale [0.5]", torch.tensor([0.5], device="cuda"), 0.5),
    ):
        try:
            got = torch.ops.trtllm.mxfp8_mxfp8_gemm(aq, a_sw, wq, w_sw, gs, torch.bfloat16)
            want = (good.float() * expect).to(torch.bfloat16).float()
            note = " (element 0 used, rest ignored)" if expect == 1.0 else ""
            d = (got.float() - want).abs().max().item()
            print(f"  {label:34s} ACCEPTED; vs {expect} x alpha1: max_abs_diff={d:.3e}{note}")
        except Exception as exc:  # noqa: BLE001
            print(f"  {label:34s} raised {type(exc).__name__}: {exc}")

    # .clone() is load-bearing: a slice shares storage with the full buffer, so
    # an out-of-bounds read lands on the same valid bytes and returns a
    # bit-identical result for a reason that has nothing to do with the op. A
    # fresh allocation is the only way to see what a genuinely short buffer does.
    for label, buf in (
        ("weight_scale one byte short", w_sw[:-1].clone()),
        ("weight_scale a tenth of the size", w_sw[: w_sw.numel() // 10].clone()),
        ("weight_scale oversized by 4096", torch.cat([w_sw, w_sw[:4096]]).clone()),
    ):
        try:
            got = torch.ops.trtllm.mxfp8_mxfp8_gemm(
                aq, a_sw, wq, buf, torch.ones(1, device="cuda", dtype=torch.float32), torch.bfloat16
            )
            torch.cuda.synchronize()
            d = (got.float() - good.float()).abs().max().item()
            print(f"  {label:34s} ACCEPTED, max_abs_diff_vs_correct={d:.3e}")
        except Exception as exc:  # noqa: BLE001
            print(f"  {label:34s} raised {type(exc).__name__}: {str(exc)[:70]}")

    for label, dt in (("out_dtype fp32", torch.float32), ("out_dtype fp16", torch.float16)):
        try:
            got = torch.ops.trtllm.mxfp8_mxfp8_gemm(
                aq, a_sw, wq, w_sw, torch.ones(1, device="cuda", dtype=torch.float32), dt
            )
            print(f"  {label:34s} ACCEPTED, dtype={got.dtype} shape={tuple(got.shape)}")
        except Exception as exc:  # noqa: BLE001
            print(f"  {label:34s} raised {type(exc).__name__}: {str(exc)[:70]}")


def probe_claimed_rejections() -> None:
    """Drive every rejection the contract claims, and quote what actually came back.

    A "the op rejects this" sentence promises a guard the caller then does not
    write, so it is the expensive kind to get wrong. The previous contract
    listed six of these read out of `CHECK_INPUT` and `TORCH_CHECK` in the
    source without ever driving one. Each is driven here and the contract quotes
    the observed text, or the claim comes out.
    """
    print()
    print("=" * 100)
    print("6. every rejection the contract claims -- observed, not read from source")
    print("=" * 100)
    torch.manual_seed(4)
    k, n, m = 5120, 1280, 64
    a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16) * 0.1
    w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.1
    aq, a_sw, _ = _quantize(a)
    wq, _, w_pl = _quantize(w)
    w_sw = torch.ops.trtllm.block_scale_interleave(w_pl.view(n, k // 32)).flatten()
    gs = torch.ones(1, device="cuda", dtype=torch.float32)

    # A short K that is still a multiple of 32, for the mismatched-K case.
    a_short, a_short_sw, _ = _quantize(torch.randn(m, 2048, device="cuda", dtype=torch.bfloat16))
    # K not divisible by 32, and N not divisible by the N alignment.
    a_k33 = torch.randn(m, 33, device="cuda", dtype=torch.bfloat16).to(torch.float8_e4m3fn)
    w_n33 = torch.randn(33, k, device="cuda", dtype=torch.bfloat16).to(torch.float8_e4m3fn)

    cases = [
        ("act dtype bf16 not e4m3", lambda: (a, a_sw, wq, w_sw, gs, torch.bfloat16)),
        ("weight dtype bf16 not e4m3", lambda: (aq, a_sw, w, w_sw, gs, torch.bfloat16)),
        (
            "act_scale dtype int8 not uint8",
            lambda: (aq, a_sw.view(torch.int8), wq, w_sw, gs, torch.bfloat16),
        ),
        (
            "global_scale dtype bf16 not fp32",
            lambda: (aq, a_sw, wq, w_sw, gs.bfloat16(), torch.bfloat16),
        ),
        (
            "act non-contiguous (transposed view)",
            lambda: (aq.t().contiguous().t(), a_sw, wq, w_sw, gs, torch.bfloat16),
        ),
        (
            "weight non-contiguous (transposed view)",
            lambda: (aq, a_sw, wq.t().contiguous().t(), w_sw, gs, torch.bfloat16),
        ),
        ("act on CPU", lambda: (aq.cpu(), a_sw, wq, w_sw, gs, torch.bfloat16)),
        ("act rank 3 not 2", lambda: (aq.view(1, m, k), a_sw, wq, w_sw, gs, torch.bfloat16)),
        ("weight rank 1 not 2", lambda: (aq, a_sw, wq.flatten(), w_sw, gs, torch.bfloat16)),
        (
            "K mismatch act 2048 vs weight 5120",
            lambda: (a_short, a_short_sw, wq, w_sw, gs, torch.bfloat16),
        ),
        (
            "K = 33, not a multiple of 32",
            lambda: (a_k33, a_sw, w_n33[:, :33].contiguous(), w_sw, gs, torch.bfloat16),
        ),
        (
            "N = 33, not a multiple of the N alignment",
            lambda: (aq, a_sw, w_n33, w_sw, gs, torch.bfloat16),
        ),
        ("out_dtype int32", lambda: (aq, a_sw, wq, w_sw, gs, torch.int32)),
    ]
    for label, build in cases:
        try:
            args = build()
        except Exception as exc:  # noqa: BLE001 — building the bad input can itself fail
            print(f"  {label:44s} could not be built: {type(exc).__name__}: {str(exc)[:60]}")
            continue
        try:
            torch.ops.trtllm.mxfp8_mxfp8_gemm(*args)
            torch.cuda.synchronize()
            print(f"  {label:44s} ** ACCEPTED, NO RAISE **")
        except Exception as exc:  # noqa: BLE001 — the raise is the measurement
            text = " ".join(str(exc).split())
            print(f"  {label:44s} {type(exc).__name__}: {text[:120]}")


def _native_operand(rows: int, k: int, seed: int):
    """The test's own operand construction, so probe and test measure the same data.

    Scale per 32-wide block derived from that block's amax (the OCP rule), which
    is what a quantizer emits. Written in native torch, independent of
    `mxfp8_quantize`.
    """
    gen = torch.Generator(device="cuda").manual_seed(seed)
    blocks = k // 32
    x = torch.randn(rows, k, generator=gen, device="cuda", dtype=torch.float32)
    xb = x.unflatten(-1, (blocks, 32))
    amax = xb.abs().amax(dim=-1).clamp_min(torch.finfo(torch.float32).tiny)
    exp = torch.ceil(torch.log2(amax / 448.0)).clamp(-127.0, 127.0)
    scale = torch.ldexp(torch.ones_like(exp), exp.to(torch.int32))
    values = (xb / scale.unsqueeze(-1)).to(torch.float8_e4m3fn)
    truth = values.float().mul(scale.unsqueeze(-1)).flatten(-2)
    exps = (exp.to(torch.int32) + 127).to(torch.uint8)
    return values.flatten(-2).contiguous(), exps, truth


def _swz(sf_2d: torch.Tensor) -> torch.Tensor:
    rows, cols = sf_2d.shape
    pc = -(-cols // 4) * 4
    r = torch.arange(rows, device="cuda").view(-1, 1)
    c = torch.arange(cols, device="cuda").view(1, -1)
    off = (c % 4) + (c // 4) * 512 + (r % 32) * 16 + ((r % 128) // 32) * 4 + (r // 128) * 128 * pc
    out = torch.zeros((-(-rows // 128) * 128) * pc, dtype=torch.uint8, device="cuda")
    out[off.flatten()] = sf_2d.flatten()
    return out


def _short_buffer_case(m: int, which: str, keep_of, poison: bool):
    """One short/oversized scale-buffer call. Returns (buf, full, good, got) or raises."""
    k, n = 5120, 1280
    act, a_exp, _ = _native_operand(m, k, 70)
    wgt, w_exp, _ = _native_operand(n, k, 71)
    a_sf, w_sf = _swz(a_exp), _swz(w_exp)
    gs = torch.ones(1, device="cuda", dtype=torch.float32)
    good = torch.ops.trtllm.mxfp8_mxfp8_gemm(act, a_sf, wgt, w_sf, gs, torch.bfloat16)
    full = a_sf if which == "act_scale" else w_sf
    keep = keep_of(full.numel())
    if keep > full.numel():
        buf = torch.cat([full, full[: keep - full.numel()]]).clone()
    elif poison:
        # An arena LARGER than the buffer the op will read, with the tail set to
        # 0x00 (UE8M0 exponent 2**-127). The out-of-bounds read then lands on
        # mapped memory holding a known-wrong byte: the wrong answer is the same
        # on every run and nothing else in the process is disturbed.
        arena = torch.zeros(full.numel() + 4096, dtype=torch.uint8, device="cuda")
        arena[:keep] = full[:keep]
        buf = arena[:keep]
    else:
        buf = full[:keep].clone()
    args = (
        (act, buf, wgt, w_sf, gs, torch.bfloat16)
        if which == "act_scale"
        else (act, a_sf, wgt, buf, gs, torch.bfloat16)
    )
    got = torch.ops.trtllm.mxfp8_mxfp8_gemm(*args)
    torch.cuda.synchronize()
    return buf, full, good, got


def _oob_child(which: str) -> int:
    """Child process: drive a genuinely out-of-bounds tenth-size scale buffer.

    Run in its own process on purpose. A `.clone()`d tenth-size buffer is not
    merely read past its end -- the read leaves the allocation entirely, and
    what happens next is not deterministic: observed here both as `inf` output
    with thousands of NaNs and as `CUDA error: an illegal memory access was
    encountered`, which poisons the CUDA context for everything after it. In the
    parent that second outcome cost two whole probe sections. The measurement is
    worth keeping; the blast radius is not. Both outcomes are reported.
    """
    try:
        buf, full, good, got = _short_buffer_case(128, which, lambda x: x // 10, poison=False)
    except Exception as exc:  # noqa: BLE001 — the failure IS the measurement
        print(f"RAISED {type(exc).__name__}: {' '.join(str(exc).split())[:160]}")
        return 3
    d = (got.float() - good.float()).abs()
    print(
        f"SURVIVED {buf.numel()}/{full.numel()} B  "
        f"max_abs_diff={torch.nan_to_num(d, nan=float('inf')).max().item():.3e} "
        f"nan={int(got.isnan().sum().item())}"
    )
    return 0


def probe_scale_buffer_lengths() -> None:
    """What a short scale buffer does, for BOTH operands and by two constructions.

    Three things this has to separate, because the entry's guard depends on all
    three:

      * ONE BYTE SHORT is read past the end, and at these sizes that address has
        so far always fallen inside the caching allocator's 512-byte rounding,
        so every run of it here has returned a silently wrong result rather than
        faulting. That is an observation about where the allocator happened to
        put the block, not a guarantee. Driven fresh (allocator garbage) and
        poisoned (known 0x00).
      * A TENTH OF THE SIZE leaves the allocation entirely. Driven only in a
        poisoned arena here, because the fresh variant has been observed both
        returning inf/NaN and raising an illegal memory access that kills the
        CUDA context -- measured in a child process below rather than in this
        one, since the second outcome ends the run.
      * OVERSIZED must stay accepted and inert.

    And the trap in between: at `M=64` the act buffer's last bytes belong to
    padding rows 64..127, so dropping them cannot change a real row's result.
    Measured at both M=64 and M=128 precisely so "a short act buffer is
    harmless" never gets recorded from the one geometry where it happens to be.
    """
    print()
    print("=" * 100)
    print("8. scale-buffer length: both operands, fresh-clone vs poisoned-arena truncation")
    print("=" * 100)
    cases = [
        ("one byte short", lambda x: x - 1, False, "fresh"),
        ("one byte short", lambda x: x - 1, True, "poisoned"),
        ("a tenth of the size", lambda x: x // 10, True, "poisoned"),
        ("oversized by 4096", lambda x: x + 4096, False, "fresh"),
    ]
    for m in (128, 64):
        for which in ("act_scale", "weight_scale"):
            for mode, keep_of, poison, how in cases:
                try:
                    buf, full, good, got = _short_buffer_case(m, which, keep_of, poison)
                except Exception as exc:  # noqa: BLE001 — a raise is the measurement
                    print(
                        f"  M={m:4d} {which:12s} {mode:19s} {how:8s} raised "
                        f"{type(exc).__name__}: {str(exc)[:60]}"
                    )
                    continue
                d = (got.float() - good.float()).abs()
                print(
                    f"  M={m:4d} {which:12s} {mode:19s} {how:8s} ACCEPTED "
                    f"{buf.numel():8d}/{full.numel():8d} B  "
                    f"max_abs_diff={torch.nan_to_num(d, nan=float('inf')).max().item():.3e} "
                    f"nan={int(got.isnan().sum().item()):7d}  "
                    f"{'IDENTICAL' if torch.equal(good, got) else 'DIFFERS'}"
                )

    print("  -- genuinely out-of-bounds (tenth size, freshly allocated), one child process each:")
    for which in ("act_scale", "weight_scale"):
        proc = subprocess.run(  # noqa: S603 — this file, this interpreter
            [sys.executable, os.path.abspath(__file__), "--oob-child", which],
            capture_output=True,
            text=True,
            timeout=900,
            check=False,
        )
        verdict = next(
            (ln for ln in proc.stdout.splitlines() if ln.startswith(("SURVIVED", "RAISED"))),
            "",
        )
        if not verdict:
            tail = (proc.stderr.strip().splitlines() or [""])[-1]
            verdict = f"CHILD DIED rc={proc.returncode} signal={-proc.returncode} {tail[:100]}"
        print(
            f"     M= 128 {which:12s} a tenth of the size fresh    rc={proc.returncode} {verdict}"
        )


def probe_tolerance() -> None:
    """Which elementwise gate actually holds, measured over the whole surface set.

    Three candidates, all dtype-aware, differing only in the absolute floor:

      default    rtol_default, atol = 1e-5      (a floor chosen for unit-scale
                                                 tensors; these outputs are
                                                 O(100-1000))
      step       rtol_default, atol = one output-dtype step at the tensor's
                                     own magnitude
      final      rtol_default, atol = max(one output-dtype step,
                                          sqrt(K) * 2**-24) at the tensor's own
                                     magnitude -- the bound the entry actually
                                     uses, so it is the one whose
                                     discrimination has to be on the record

    `step` is kept as its own column because it is what separates the two terms:
    where `>step` is nonzero and `>final` is zero, the `sqrt(K)` accumulation
    term is the thing doing the work. The deliberately wrong variant is counted
    against `final`, not against the looser intermediate, so the separation
    printed here is the separation the gate actually has.
    """
    print()
    print("=" * 100)
    print("7. tolerance: default vs output-step vs final floor, over every surface")
    print("=" * 100)
    RTOL = {torch.bfloat16: 1.6e-2, torch.float16: 1e-3, torch.float32: 1.3e-6}
    STEP = {torch.bfloat16: 2.0**-8, torch.float16: 2.0**-11, torch.float32: 2.0**-24}
    print(
        f"  {'surface':>18s} {'M':>5s} {'out':>9s} {'scale':>10s} {'max_abs':>10s} "
        f"{'>default':>9s} {'>step':>7s} {'>final':>7s} {'wrong>final':>12s} {'wrong/corr':>11s}"
    )
    for k, n, _cnt, _what in V41_FP8_SURFACES:
        for m in (64, 4096):
            for odt in (torch.bfloat16, torch.float16, torch.float32):
                act, a_exp, a_truth = _native_operand(m, k, (k + n + m) % 4096)
                wgt, w_exp, w_truth = _native_operand(n, k, (k + n + m) % 4096 + 1)
                gs = torch.ones(1, device="cuda", dtype=torch.float32)
                got = torch.ops.trtllm.mxfp8_mxfp8_gemm(act, _swz(a_exp), wgt, _swz(w_exp), gs, odt)
                want = a_truth @ w_truth.t()
                scale = want.abs().max().item()
                diff = (got.float() - want).abs()
                rtol, step = RTOL[odt], STEP[odt]
                rel = rtol * want.abs()
                final = max(step, math.sqrt(k) * 2.0**-24) * scale
                over_def = int((diff > 1e-5 + rel).sum().item())
                over_step = int((diff > step * scale + rel).sum().item())
                over_final = int((diff > final + rel).sum().item())
                rolled = torch.ops.trtllm.mxfp8_mxfp8_gemm(
                    act, _swz(a_exp), wgt, _swz(w_exp.roll(1, -1)), gs, odt
                )
                wd = (rolled.float() - want).abs()
                wrong_final = int((wd > final + rel).sum().item())
                ratio = wd.max().item() / max(diff.max().item(), 1e-30)
                print(
                    f"  {k:8d}x{n:<9d} {m:5d} {str(odt).replace('torch.', ''):>9s} "
                    f"{scale:10.3e} {diff.max().item():10.3e} {over_def:9d} {over_step:7d} "
                    f"{over_final:7d} {wrong_final:12d} {ratio:10.1f}x"
                )


def main() -> int:
    assert torch.cuda.is_available(), "this is a GPU measurement"
    cap = torch.cuda.get_device_capability()
    import tensorrt_llm

    print(
        f"device={torch.cuda.get_device_name()} sm_{cap[0]}{cap[1]} trtllm={tensorrt_llm.__version__}"
    )
    print(f"tensorrt_llm from {tensorrt_llm.__file__}")
    failed = []
    # `probe_scale_buffer_lengths` runs LAST on purpose: it is the only section
    # that deliberately makes the kernel read outside a buffer, and a CUDA
    # context poisoned there would take every later measurement with it. The
    # genuinely out-of-bounds cases are in child processes, this ordering is the
    # second line of defence.
    for fn in (
        probe_surfaces,
        probe_rows,
        probe_scale_layout,
        probe_tactic_cache,
        probe_unchecked_domains,
        probe_claimed_rejections,
        probe_tolerance,
        probe_scale_buffer_lengths,
    ):
        try:
            fn()
        except Exception as exc:  # noqa: BLE001 — one probe dying must not hide the rest
            print(f"\n!! {fn.__name__} raised {type(exc).__name__}: {exc}")
            import traceback

            traceback.print_exc()
            failed.append(fn.__name__)
    # Every section still runs after one dies, so a single failure reports the
    # rest of the measurements -- but the exit code says so. Returning 0 here
    # regardless made a dead section indistinguishable from a measured one, and
    # a probe that cannot run measures nothing.
    if failed:
        print(f"\nprobe INCOMPLETE: {len(failed)} section(s) raised: {', '.join(failed)}")
        return 1
    print("\nprobe complete")
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--oob-child":
        raise SystemExit(_oob_child(sys.argv[2]))
    raise SystemExit(main())
