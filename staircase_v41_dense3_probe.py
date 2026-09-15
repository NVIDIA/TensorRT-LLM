# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Drive the Goal-1.2 dense entries at the staircase TARGET's geometry, not the reference's.

The previous revision of this probe derived every shape from the reference
implementation's `ColumnParallelLinear` / `RowParallelLinear` / `ParallelHead`,
which shard over `world_size=4`. The staircase target does not: `plan.md` fixes
attention DP at dep4 and **replicates** attention, dense projections, norms,
embedding and the untied head, so the target's own `(K, N)` are the *unsharded*
ones. Certifying the reference's shard widths certifies shapes the target never
calls.

The derivation source is therefore the **raw checkpoint's safetensors headers**,
which store `[out, in]` and are what `weights.py` will read -- section 0 prints
them beside the shard each would become, so the TP-versus-ADP distinction is on
the record rather than in a summary.

Sections:

  0. the projection audit: raw `[N, K]` header vs reference TP4 shard vs target
  1. `gemm/mxfp8_mxfp8_gemm` at the three target surfaces the audit corrects
  2. `quantization/mxfp8_quantize` at the target activation widths, incl. K=8192
  3. `gemm/cublas_mm` at the replicated head, 5120 -> 129280, every row bucket
  4. `gemm/bmm_out` at the grouped output LoRA, batch 8, independent reference
  5. `cublas_mm` preconditions re-driven on THIS arch and version
  6. `bmm_out` preconditions re-driven on THIS arch and version
  7. `mxfp8_quantize` rejection texts, negative alignment, and the SIGFPE domain

Sections 5-7 exist because the three contracts attribute their guard and
silent-failure behaviour to trtllm 1.3.0rc21 on sm_100 -- a path no receipt here
covers. Each claim is driven and its exact outcome printed, so the contracts can
be rewritten from what this machine does.

Reads nothing from the catalog and asserts nothing: it prints measurements.
"""

from __future__ import annotations

import json
import os
import struct
import subprocess
import sys
from collections.abc import Callable

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*

# Every fp32 reference here is what a tolerance is judged against.
torch.backends.cuda.matmul.allow_tf32 = False

CKPT = "/scratch/fsw/portfolios/coreai/projects/coreai_comparch_trtllm/users/fredw/models/DeepSeek-V4.1-Flash"

WORLD = 4  # dep4

#: Row counts a served target reaches.
ROW_BUCKETS = [1, 2, 3, 7, 8, 16, 31, 32, 64, 127, 128, 129, 255, 256, 512, 1024, 4096]

#: A readable subset for per-row printing; every bucket still runs.
PRINT_ROWS = (1, 129, 4096)

#: The dense surfaces of one backbone layer, as raw checkpoint keys. Layer 2 is
#: the first layer that owns every optional piece (KV source + index source), so
#: one layer's key set covers the whole dense vocabulary.
DENSE_KEYS = [
    ("layers.2.attn.wq_a.weight", "column-replicated", "attn wq_a"),
    ("layers.2.attn.wq_b.weight", "ColumnParallelLinear", "attn wq_b"),
    ("layers.2.attn.wkv.weight", "column-replicated", "attn wkv"),
    ("layers.2.attn.wo_a.weight", "ColumnParallelLinear", "attn wo_a (grouped)"),
    ("layers.2.attn.wo_b.weight", "RowParallelLinear", "attn wo_b"),
    ("layers.2.attn.indexer.wq_b.weight", "ColumnParallelLinear", "indexer wq_b"),
    ("layers.2.ffn.shared_experts.w1.weight", "column-replicated", "shared w1"),
    ("layers.2.ffn.shared_experts.w3.weight", "column-replicated", "shared w3"),
    ("layers.2.ffn.shared_experts.w2.weight", "column-replicated", "shared w2"),
    ("layers.1.engram.wkv.weight", "column-replicated", "engram wkv"),
    ("head.weight", "ParallelHead", "language head"),
    ("embed.weight", "ParallelEmbedding", "token embedding"),
]

#: The three FP8 GEMM surfaces the audit corrects: (K, N, tag). N is the raw
#: header's out dim; the reference's shard is N // 4 (column) or K // 4 (row).
TARGET_GEMM_FIX = [
    (1280, 32768, "attn_wq_b", 8192, "N"),
    (8192, 5120, "attn_wo_b", 2048, "K"),
    (1280, 4096, "indexer_wq_b", 1024, "N"),
]

HEAD_DIM, HEAD_VOCAB_FULL, HEAD_VOCAB_SHARD = 5120, 129280, 32320

#: Grouped output LoRA under ADP: all 8 groups are local, each 4096 -> 1024.
LORA_GROUPS, LORA_IN, LORA_RANK = 8, 4096, 1024

E4M3_MAX = 448.0
BLOCK = 32


def _pad_up(x: int, m: int) -> int:
    return (x + m - 1) // m * m


def _c_pad_up(k: int, alignment: int) -> int:
    """`((k + a - 1) / a) * a` under C's integer division, which TRUNCATES toward zero.

    Python's `//` FLOORS, and the two disagree exactly where this probe is
    looking: for a negative `alignment` the quotient is negative, so flooring
    rounds it away from zero and truncation rounds it toward zero. An earlier
    revision of this probe printed the Python value as "the C formula" and so
    claimed 2848 / 2816 / 2560 beside the op's observed 2816 / 2752 / 2048 --
    labelling the op's behaviour as a mismatch when the formula was simply the
    wrong one. With truncation the claim reproduces the observation exactly.
    """
    if alignment == 0:
        return 0
    num = k + alignment - 1
    quotient = abs(num) // abs(alignment)
    if (num < 0) != (alignment < 0):
        quotient = -quotient
    return quotient * alignment


def _outcome(fn: Callable[[], object], width: int = 150) -> str:
    """Run `fn` and describe what happened: a raise with its text, or a result."""
    try:
        out = fn()
    except Exception as exc:  # noqa: BLE001 — describing the raise IS the measurement
        text = str(exc).strip().splitlines()
        first = text[0] if text else ""
        return f"RAISED {type(exc).__name__}: {first[:width]}"
    if isinstance(out, torch.Tensor):
        return f"ACCEPTED -> {tuple(out.shape)} {str(out.dtype).replace('torch.', '')}"
    return f"ACCEPTED -> {out!r}"


def _bf16_matches(got: torch.Tensor, ref: torch.Tensor) -> str:
    """max_abs plus a verdict at THIS entry's bf16 gate, never at a unit-scale atol.

    An earlier revision of this probe asked the same question with
    `torch.allclose(..., atol=1e-2)` and got `False` for three results that are
    in fact correct: bf16 values of scale ~60 have a 0.25 ulp, so a unit-scale
    absolute tolerance rejects a correct bf16 answer. The verdict below uses the
    gate the entry certifies (`rtol=1.6e-2, atol=1e-3`).
    """
    diff = (got.float() - ref).abs()
    over = int((diff > 1e-3 + 1.6e-2 * ref.abs()).sum().item())
    return f"max_abs={diff.max().item():.3e} over_gate={over}/{ref.numel()}"


# ── 0. the projection audit ───────────────────────────────────────────────────


def probe_projection_audit() -> None:
    print("=" * 108)
    print(
        "0. dense projection audit: raw checkpoint [out, in] vs reference TP4 shard vs target ADP"
    )
    print("=" * 108)
    index = json.load(open(os.path.join(CKPT, "model.safetensors.index.json")))
    weight_map = index["weight_map"]
    headers: dict[str, dict] = {}
    for key, _, _ in DENSE_KEYS:
        shard = weight_map[key]
        if shard not in headers:
            with open(os.path.join(CKPT, shard), "rb") as fh:
                size = struct.unpack("<Q", fh.read(8))[0]
                headers[shard] = json.loads(fh.read(size))
    print(
        f"  {'checkpoint key':40s} {'raw [out,in]':>18s} {'dtype':>9s} "
        f"{'reference shard':>18s} {'target (K,N)':>18s}"
    )
    for key, kind, _tag in DENSE_KEYS:
        meta = headers[weight_map[key]][key]
        out_dim, in_dim = meta["shape"]
        if kind == "ColumnParallelLinear":
            shard = f"[{out_dim // WORLD}, {in_dim}]"
        elif kind == "RowParallelLinear":
            shard = f"[{out_dim}, {in_dim // WORLD}]"
        elif kind in ("ParallelHead", "ParallelEmbedding"):
            shard = f"[{out_dim // WORLD}, {in_dim}]"
        else:
            shard = "replicated"
        print(
            f"  {key:40s} {str([out_dim, in_dim]):>18s} {meta['dtype']:>9s} "
            f"{shard:>18s} {f'({in_dim}, {out_dim})':>18s}"
        )
    print(
        "  NOTE: the target replicates every row above (plan.md lines 30, 78-89, 97, 110-112),\n"
        "        so the target column is the raw header, never the reference shard."
    )


# ── 1. MXFP8 GEMM at the corrected surfaces ───────────────────────────────────


def _mxfp8_operand(rows: int, k: int, seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """MXFP8 operand built the way a quantizer does: values, UE8M0 bytes, fp32 truth."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    blocks = k // BLOCK
    x = torch.randn(rows, k, generator=gen, device="cuda", dtype=torch.float32)
    xb = x.unflatten(-1, (blocks, BLOCK))
    amax = xb.abs().amax(dim=-1).clamp_min(torch.finfo(torch.float32).tiny)
    exp = torch.ceil(torch.log2(amax / E4M3_MAX)).clamp(-127.0, 127.0)
    scale = torch.ldexp(torch.ones_like(exp), exp.to(torch.int32))
    values = (xb / scale.unsqueeze(-1)).to(torch.float8_e4m3fn)
    truth = values.float().mul(scale.unsqueeze(-1)).flatten(-2)
    return values.flatten(-2).contiguous(), (exp.to(torch.int32) + 127).to(torch.uint8), truth


def _swizzle(sf_2d: torch.Tensor) -> torch.Tensor:
    """`[rows, blocks]` UE8M0 bytes -> the flat 128x4-swizzled buffer, in native torch."""
    rows, cols = sf_2d.shape
    padded_cols = _pad_up(cols, 4)
    r = torch.arange(rows, device="cuda").view(-1, 1)
    c = torch.arange(cols, device="cuda").view(1, -1)
    offset = (
        (c % 4)
        + (c // 4) * 512
        + (r % 32) * 16
        + ((r % 128) // 32) * 4
        + (r // 128) * 128 * padded_cols
    )
    buf = torch.zeros(_pad_up(rows, 128) * padded_cols, dtype=torch.uint8, device="cuda")
    buf[offset.flatten()] = sf_2d.flatten()
    return buf


def probe_gemm_target_surfaces() -> None:
    print()
    print("=" * 108)
    print("1. mxfp8_mxfp8_gemm at the three TARGET surfaces the audit corrects (and the shard each")
    print("   replaces), fp32-accumulated reference from the dequantized operands, bf16 out")
    print("=" * 108)
    print(
        f"  {'surface':>14s} {'K':>6s} {'N':>6s} {'M':>5s} {'scale':>10s} {'max_abs':>10s} "
        f"{'>gate':>7s} {'rolled-scale >gate':>19s} {'ratio':>9s}"
    )
    for k, n, tag, shard, axis in TARGET_GEMM_FIX:
        for label, kk, nn in (
            (tag, k, n),
            (f"{tag}/shard", shard if axis == "K" else k, shard if axis == "N" else n),
        ):
            over_all = 0
            for m in ROW_BUCKETS:
                act, act_exp, act_truth = _mxfp8_operand(m, kk, seed=(kk + nn + m) % 4096)
                wgt, wgt_exp, wgt_truth = _mxfp8_operand(nn, kk, seed=(kk + nn + m) % 4096 + 1)
                gs = torch.ones(1, device="cuda", dtype=torch.float32)
                got = torch.ops.trtllm.mxfp8_mxfp8_gemm(
                    act, _swizzle(act_exp), wgt, _swizzle(wgt_exp), gs, torch.bfloat16
                )
                want = act_truth @ wgt_truth.t()
                scale = want.abs().max().item()
                # The entry's own gate: default bf16 rtol, floor
                # max(2**-8, sqrt(K) * 2**-24) * |want|.max().
                atol = max(2.0**-8, (kk**0.5) * 2.0**-24) * scale
                diff = (got.float() - want).abs()
                bound = atol + 1.6e-2 * want.abs()
                over = int((diff > bound).sum().item())
                over_all += over
                if m in PRINT_ROWS:
                    rolled = torch.ops.trtllm.mxfp8_mxfp8_gemm(
                        act,
                        _swizzle(act_exp),
                        wgt,
                        _swizzle(torch.roll(wgt_exp, 1, dims=-1)),
                        gs,
                        torch.bfloat16,
                    )
                    wrong = (rolled.float() - want).abs()
                    over_wrong = int((wrong > bound).sum().item())
                    print(
                        f"  {label:>14s} {kk:6d} {nn:6d} {m:5d} {scale:10.3e} "
                        f"{diff.max().item():10.3e} {over:7d} {over_wrong:19d} "
                        f"{wrong.max().item() / max(diff.max().item(), 1e-30):9.1f}x"
                    )
            print(
                f"  {label:>14s} all {len(ROW_BUCKETS)} row buckets: {over_all} elements over gate"
            )


# ── 2. quantize at the target activation widths ───────────────────────────────


def _ref_quant(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Native-torch MXFP8 quantization: e4m3 data plus one UE8M0 byte per 32 elements."""
    m, k = x.shape
    blk = x.float().view(m, k // BLOCK, BLOCK)
    amax = blk.abs().amax(dim=-1)
    u = (amax / E4M3_MAX).view(torch.int32)
    byte = (((u >> 23) & 0xFF) + ((u & 0x7FFFFF) > 0).to(torch.int32)).clamp(0, 254)
    scale = torch.exp2(byte.to(torch.float32) - 127.0).unsqueeze(-1)
    return (blk / scale).view(m, k).to(torch.float8_e4m3fn), byte.to(torch.uint8)


def probe_quantize_target_widths() -> None:
    print()
    print("=" * 108)
    print("2. mxfp8_quantize at the TARGET activation widths (K=8192 is the wo_b input the")
    print("   reference never sees: it row-shards wo_b to K=2048)")
    print("=" * 108)
    widths = [
        (8192, "attn wo_b input        TARGET"),
        (6144, "engram.wkv input       TARGET"),
        (5120, "wq_a/wkv/shared w1,w3  TARGET"),
        (2304, "shared w2 input        TARGET"),
        (1280, "wq_b/indexer wq_b in   TARGET"),
        (2048, "wo_b input, TP4 shard  reference-only"),
    ]
    print(
        f"  {'K':>6s} {'what':30s} {'rows':>5s} {'data bit-exact':>15s} "
        f"{'scale bit-exact':>16s} {'swizzled@M=128':>15s} {'GEMM expects':>13s}"
    )
    for k, what in widths:
        data_ok, sf_ok, note = True, True, ""
        for m in ROW_BUCKETS:
            gen = torch.Generator(device="cuda").manual_seed(k * 131 + m)
            x = torch.randn(m, k, generator=gen, device="cuda", dtype=torch.bfloat16)
            data, sf_sw = torch.ops.trtllm.mxfp8_quantize(x, True, 32)
            _, sf_lin = torch.ops.trtllm.mxfp8_quantize(x, False, 32)
            ref_data, ref_byte = _ref_quant(x)
            if not torch.equal(data.view(torch.uint8), ref_data.view(torch.uint8)):
                data_ok = False
            if not torch.equal(sf_lin.view(m, k // BLOCK), ref_byte):
                sf_ok = False
            if sf_sw.numel() != _pad_up(m, 128) * _pad_up(k // BLOCK, 4):
                note = f"  MISMATCH at M={m}"
        gen = torch.Generator(device="cuda").manual_seed(k * 131 + 128)
        x = torch.randn(128, k, generator=gen, device="cuda", dtype=torch.bfloat16)
        _, sf_sw = torch.ops.trtllm.mxfp8_quantize(x, True, 32)
        want = _pad_up(128, 128) * _pad_up(k // BLOCK, 4)
        print(
            f"  {k:6d} {what:30s} {len(ROW_BUCKETS):5d} {str(data_ok):>15s} "
            f"{str(sf_ok):>16s} {sf_sw.numel():15d} {want:13d}{note}"
        )


# ── 3. the replicated language head ───────────────────────────────────────────


def probe_head_replicated() -> None:
    print()
    print("=" * 108)
    print("3. cublas_mm at the TARGET head: replicated 5120 -> 129280, complete local logits")
    print("=" * 108)
    rtol = {torch.bfloat16: 1.6e-2, torch.float32: 1.3e-6}
    print(
        f"  {'N':>7s} {'dtype':>9s} {'M':>5s} {'scale':>10s} {'max_abs':>10s} "
        f"{'>1e-5':>9s} {'>1e-3':>8s} {'reversed-vocab >1e-3':>21s}"
    )
    for n, label in ((HEAD_VOCAB_FULL, "TARGET"), (HEAD_VOCAB_SHARD, "reference-only")):
        for dt in (torch.bfloat16, torch.float32):
            over3_all, worst, worst_m = 0, 0.0, -1
            for m in ROW_BUCKETS:
                gen = torch.Generator(device="cuda").manual_seed(n + m)
                a = torch.randn(m, HEAD_DIM, generator=gen, device="cuda", dtype=dt)
                w = torch.randn(n, HEAD_DIM, generator=gen, device="cuda", dtype=dt)
                out = torch.ops.trtllm.cublas_mm(a, w.t(), None, None, 0, None)
                ref = a.float() @ w.float().t()
                diff = (out.float() - ref).abs()
                rel = rtol[dt] * ref.abs()
                over3 = int((diff > 1e-3 + rel).sum().item())
                over3_all += over3
                if diff.max().item() > worst:
                    worst, worst_m = diff.max().item(), m
                if m in PRINT_ROWS:
                    wrong = (a.float() @ torch.flip(w, dims=(0,)).float().t() - ref).abs()
                    print(
                        f"  {n:7d} {str(dt).replace('torch.', ''):>9s} {m:5d} "
                        f"{ref.abs().max().item():10.3e} {diff.max().item():10.3e} "
                        f"{int((diff > 1e-5 + rel).sum().item()):9d} {over3:8d} "
                        f"{int((wrong > 1e-3 + rel).sum().item()):21d}"
                    )
                del out, ref, diff, rel, a, w
                torch.cuda.empty_cache()
            print(
                f"  N={n} ({label}) {str(dt).replace('torch.', '')}: all {len(ROW_BUCKETS)} row "
                f"buckets, worst max_abs={worst:.3e} (M={worst_m}), over the entry's gate={over3_all}"
            )


# ── 4. the grouped output LoRA at batch 8 ─────────────────────────────────────


def _ref_bmm_independent(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Per-batch fp32 `torch.mm` -- NOT `torch.bmm`, which is the op under test."""
    return torch.stack([torch.mm(a[i].float(), b[i].float()) for i in range(a.shape[0])])


def probe_output_lora_bmm() -> None:
    print()
    print("=" * 108)
    print("4. bmm_out at the TARGET output LoRA: batch 8 (all groups local under ADP),")
    print("   [M,4096] @ [4096,1024], b a transpose view; reference is per-batch fp32 mm")
    print("=" * 108)
    # The per-batch `mm` reference is itself checked once against a construction
    # that launches no GEMM at all, so "independent" is measured, not asserted.
    gen = torch.Generator(device="cuda").manual_seed(11)
    sa = torch.randn(3, 5, 64, generator=gen, device="cuda", dtype=torch.bfloat16)
    sb = torch.randn(3, 64, 7, generator=gen, device="cuda", dtype=torch.bfloat16)
    mulsum = (sa.float().unsqueeze(-1) * sb.float().unsqueeze(1)).sum(dim=2)
    mmref = _ref_bmm_independent(sa, sb)
    print(f"  per-batch mm vs elementwise mul+sum: max_abs={(mmref - mulsum).abs().max():.3e}")
    print(
        f"  {'batch':>6s} {'M':>6s} {'scale':>10s} {'max_abs':>10s} {'>1e-5':>9s} "
        f"{'>1e-3':>8s} {'rolled-group >1e-3':>19s}"
    )
    for batch, label in ((LORA_GROUPS, "TARGET"), (LORA_GROUPS // WORLD, "reference-only")):
        over3_all = 0
        for m in ROW_BUCKETS:
            gen = torch.Generator(device="cuda").manual_seed(4096 + m + batch)
            o = torch.randn(batch, m, LORA_IN, generator=gen, device="cuda", dtype=torch.bfloat16)
            wo_a = torch.randn(
                batch, LORA_RANK, LORA_IN, generator=gen, device="cuda", dtype=torch.bfloat16
            )
            b = wo_a.transpose(1, 2)
            out = torch.empty(batch, m, LORA_RANK, device="cuda", dtype=torch.bfloat16)
            torch.ops.trtllm.bmm_out(o, b, out)
            ref = _ref_bmm_independent(o, b)
            diff = (out.float() - ref).abs()
            bound3 = 1e-3 + 1.6e-2 * ref.abs()
            over3 = int((diff > bound3).sum().item())
            over3_all += over3
            if m in PRINT_ROWS:
                wrong_out = torch.empty_like(out)
                torch.ops.trtllm.bmm_out(o, b.roll(1, dims=0), wrong_out)
                wrong = (wrong_out.float() - ref).abs()
                print(
                    f"  {batch:6d} {m:6d} {ref.abs().max().item():10.3e} "
                    f"{diff.max().item():10.3e} "
                    f"{int((diff > 1e-5 + 1.6e-2 * ref.abs()).sum().item()):9d} {over3:8d} "
                    f"{int((wrong > bound3).sum().item()):19d}"
                )
        print(
            f"  batch={batch} ({label}): all {len(ROW_BUCKETS)} row buckets, "
            f"over the entry's gate={over3_all}"
        )


# ── 5. cublas_mm preconditions, re-driven here ────────────────────────────────


def probe_cublas_preconditions() -> None:
    print()
    print("=" * 108)
    print("5. cublas_mm preconditions re-driven on THIS arch/version (the contract attributes")
    print("   all of them to trtllm 1.3.0rc21 on sm_100)")
    print("=" * 108)
    gen = torch.Generator(device="cuda").manual_seed(5)
    m, k, n = 64, 256, 128
    a = torch.randn(m, k, generator=gen, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(n, k, generator=gen, device="cuda", dtype=torch.bfloat16)
    bias = torch.randn(n, generator=gen, device="cuda", dtype=torch.bfloat16)
    ref = a.float() @ w.float().t()

    def mm(*args) -> torch.Tensor:
        return torch.ops.trtllm.cublas_mm(*args)

    # 5a. loud domains
    wide = torch.randn(m, 2 * k, generator=gen, device="cuda", dtype=torch.bfloat16)
    cases: list[tuple[str, Callable[[], object]]] = [
        (
            "mat_b stride(0) != 1 (row-major [K,N])",
            lambda: mm(a, w.t().contiguous(), None, None, 0, None),
        ),
        ("3-D mat_a", lambda: mm(a.unsqueeze(0), w.t(), None, None, 0, None)),
        ("mat_a bf16 / mat_b fp16", lambda: mm(a, w.to(torch.float16).t(), None, None, 0, None)),
        (
            "fp8 inputs, out_dtype=None",
            lambda: mm(
                a.to(torch.float8_e4m3fn), w.to(torch.float8_e4m3fn).t(), None, None, 0, None
            ),
        ),
        ("bf16 -> fp16 out", lambda: mm(a, w.t(), None, torch.float16, 0, None)),
        ("mat_a on cpu", lambda: mm(a.cpu(), w.t().cpu(), None, None, 0, None)),
        (
            "K mismatch",
            lambda: mm(
                a, torch.randn(2 * k, n, device="cuda", dtype=torch.bfloat16), None, None, 0, None
            ),
        ),
    ]
    for name, fn in cases:
        print(f"  {name:45s} {_outcome(fn, width=400)}")

    # 5b. the silent ones: accepted, and wrong
    print("  -- accepted-but-wrong domains (the four the wrapper guards) --")
    strided = wide[:, :k]
    got = mm(strided, w.t(), None, None, 0, None)
    ref_strided = strided.float() @ w.float().t()
    print(
        f"  {'mat_a row-strided view':45s} ACCEPTED, max_abs from its own truth = "
        f"{(got.float() - ref_strided).abs().max().item():.3e} "
        f"(scale {ref_strided.abs().max().item():.3e})"
    )
    wide_w = torch.randn(n, 2 * k, generator=gen, device="cuda", dtype=torch.bfloat16)
    w_strided = wide_w[:, :k]
    got = mm(a, w_strided.t(), None, None, 0, None)
    ref_ws = a.float() @ w_strided.float().t()
    print(
        f"  {'mat_b = t() of a row-strided weight':45s} "
        f"stride={tuple(w_strided.t().stride())} shape={tuple(w_strided.t().shape)} "
        f"ACCEPTED, max_abs = {(got.float() - ref_ws).abs().max().item():.3e} "
        f"(scale {ref_ws.abs().max().item():.3e})"
    )
    short = torch.randn(n // 2, generator=gen, device="cuda", dtype=torch.bfloat16)
    print(f"  {'bias of length N//2':45s} {_outcome(lambda: mm(a, w.t(), short, None, 0, None))}")
    bias_fp32 = bias.float()
    got = mm(a, w.t(), bias_fp32, None, 0, None)
    want_biased = (ref + bias.float()).to(torch.bfloat16)
    print(
        f"  {'fp32 bias with bf16 output':45s} ACCEPTED, max_abs from the biased truth = "
        f"{(got.float() - want_biased.float()).abs().max().item():.3e}"
    )
    a32 = a.float()
    w32 = w.float()
    got = mm(a32, w32.t(), bias.float(), None, 0, None)
    unbiased = a32 @ w32.t()
    print(
        f"  {'bias with fp32 inputs':45s} ACCEPTED, distance to the UNBIASED product = "
        f"{(got - unbiased).abs().max().item():.3e}, to the biased one = "
        f"{(got - (unbiased + bias.float())).abs().max().item():.3e}"
    )
    # 5c. the meta/fake registration claim
    try:
        meta_a = torch.empty(2, m, k, device="meta", dtype=torch.bfloat16)
        meta_b = torch.empty(k, n, device="meta", dtype=torch.bfloat16)
        out = torch.ops.trtllm.cublas_mm(meta_a, meta_b, None, None, 0, None)
        print(f"  {'meta registration with 3-D mat_a':45s} ACCEPTED -> {tuple(out.shape)}")
    except Exception as exc:  # noqa: BLE001
        print(f"  {'meta registration with 3-D mat_a':45s} RAISED {type(exc).__name__}: {exc}")


# ── 6. bmm_out preconditions, re-driven here ──────────────────────────────────


def probe_bmm_preconditions() -> None:
    print()
    print("=" * 108)
    print("6. bmm_out preconditions re-driven on THIS arch/version (the contract attributes")
    print("   all of them to trtllm 1.3.0rc21 / torch 2.11.0 on sm_100)")
    print("=" * 108)
    gen = torch.Generator(device="cuda").manual_seed(6)
    bsz, m, k, n = 4, 8, 64, 32
    a = torch.randn(bsz, m, k, generator=gen, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(bsz, k, n, generator=gen, device="cuda", dtype=torch.bfloat16)

    def run(x: torch.Tensor, y: torch.Tensor, out: torch.Tensor) -> object:
        torch.ops.trtllm.bmm_out(x, y, out)
        return tuple(out.shape)

    out = torch.empty(bsz, m, n, device="cuda", dtype=torch.bfloat16)
    cases: list[tuple[str, Callable[[], object]]] = [
        ("2-D a", lambda: run(a[0], b[0], out[0])),
        ("batch mismatch", lambda: run(a, b[:2], out)),
        (
            "K mismatch",
            lambda: run(a, torch.randn(bsz, 2 * k, n, device="cuda", dtype=torch.bfloat16), out),
        ),
        ("out dtype != input dtype", lambda: run(a, b, out.float())),
        (
            "fp8 inputs",
            lambda: run(
                a.to(torch.float8_e4m3fn), b.to(torch.float8_e4m3fn), out.to(torch.float8_e4m3fn)
            ),
        ),
        ("all three on cpu", lambda: run(a.cpu(), b.cpu(), out.cpu())),
        ("a on cpu, b/out on cuda", lambda: run(a.cpu(), b, out)),
        ("3-D a/b with a 2-D out", lambda: run(a, b, out.reshape(bsz * m, n))),
        ("3-D a/b with a 4-D out", lambda: run(a, b, out.reshape(1, bsz, m, n))),
    ]
    for name, fn in cases:
        print(f"  {name:45s} {_outcome(fn)}")
    # CPU is accepted, so say what it computed rather than only that it returned.
    cpu_out = torch.empty(bsz, m, n, dtype=torch.bfloat16)
    torch.ops.trtllm.bmm_out(a.cpu(), b.cpu(), cpu_out)
    print(
        f"  {'all three on cpu: result':45s} "
        f"{_bf16_matches(cpu_out.cuda(), _ref_bmm_independent(a, b))}"
    )

    print("  -- all six mixed-dtype combinations over {bf16, fp16, fp32} --")
    dts = [torch.bfloat16, torch.float16, torch.float32]
    for da in dts:
        for db in dts:
            if da is db:
                continue
            for dout in (da, db):
                label = (
                    f"a={str(da).replace('torch.', '')} b={str(db).replace('torch.', '')} "
                    f"out={str(dout).replace('torch.', '')}"
                )
                print(
                    f"  {label:45s} "
                    f"{_outcome(lambda da=da, db=db, dout=dout: run(a.to(da), b.to(db), out.to(dout)))}"
                )

    print("  -- the wrong-shaped `out`: is it a lost write, a detach, or a shape rewrite? --")
    ref = _ref_bmm_independent(a, b)
    arena = torch.zeros(bsz, m, 4 * n, device="cuda", dtype=torch.bfloat16)
    view = arena[:, :, :n]  # right shape, aliased, non-contiguous
    ptr_before = view.data_ptr()
    torch.ops.trtllm.bmm_out(a, b, view)
    print(
        f"  correctly-shaped aliased view: data_ptr moved={view.data_ptr() != ptr_before}, "
        f"in the arena: {_bf16_matches(arena[:, :, :n], ref)}"
    )
    small = torch.zeros(bsz, m, n // 2, device="cuda", dtype=torch.bfloat16)
    ptr_before = small.data_ptr()
    import warnings

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        torch.ops.trtllm.bmm_out(a, b, small)
    print(
        f"  half-width out: shape now {tuple(small.shape)}, data_ptr moved="
        f"{small.data_ptr() != ptr_before}, {_bf16_matches(small, ref)}, warnings="
        f"{[str(w.message)[:70] for w in caught]}"
    )
    big_arena = torch.zeros(bsz, m, 4 * n, device="cuda", dtype=torch.bfloat16)
    sub = big_arena[:, :, : n // 2]
    ptr_before = sub.data_ptr()
    try:
        torch.ops.trtllm.bmm_out(a, b, sub)
        print(
            f"  half-width view INTO an arena: shape now {tuple(sub.shape)}, data_ptr moved="
            f"{sub.data_ptr() != ptr_before}, the view itself: {_bf16_matches(sub, ref)}, "
            f"the arena's first {n} cols: {_bf16_matches(big_arena[:, :, :n], ref)}"
        )
    except Exception as exc:  # noqa: BLE001
        print(f"  half-width view INTO an arena: RAISED {type(exc).__name__}: {str(exc)[:120]}")
    # A sibling view of the same storage, to answer "detached or followed?".
    sib_arena = torch.zeros(bsz, m, 4 * n, device="cuda", dtype=torch.bfloat16)
    target = sib_arena[:, :, : n // 2]
    sibling = sib_arena[:, :, : n // 2]
    torch.ops.trtllm.bmm_out(a, b, target)
    print(
        f"  sibling view after the resize: shares storage="
        f"{sibling.data_ptr() == target.data_ptr()}, sibling still shape {tuple(sibling.shape)}, "
        f"first half of the product visible through it: "
        f"{_bf16_matches(sibling, ref[:, :, : n // 2])}"
    )


# ── 7. quantize rejection texts and the two alignment domains ─────────────────


def probe_quantize_domains() -> None:
    print()
    print("=" * 108)
    print("7. mxfp8_quantize: exact rejection texts, negative alignment, and the SIGFPE domain")
    print("=" * 108)
    x = torch.randn(4, 128, device="cuda", dtype=torch.bfloat16)

    def q(t: torch.Tensor, sw: bool, al: int) -> object:
        data, sf = torch.ops.trtllm.mxfp8_quantize(t, sw, al)
        return (tuple(data.shape), int(sf.numel()))

    cases: list[tuple[str, Callable[[], object]]] = [
        (
            "K not a multiple of 32",
            lambda: q(torch.randn(4, 112, device="cuda", dtype=torch.bfloat16), False, 32),
        ),
        ("alignment not a multiple of 32", lambda: q(x, False, 48)),
        ("alignment below the block size", lambda: q(x, False, 16)),
        ("fp32 input", lambda: q(x.float(), False, 32)),
        ("1-D input", lambda: q(x.reshape(-1), False, 32)),
        (
            "non-contiguous column slice",
            lambda: q(torch.randn(8, 256, device="cuda", dtype=torch.bfloat16)[:, :128], False, 32),
        ),
        (
            "non-contiguous transpose",
            lambda: q(torch.randn(128, 8, device="cuda", dtype=torch.bfloat16).t(), False, 32),
        ),
        ("cpu input", lambda: q(torch.randn(4, 128, dtype=torch.bfloat16), False, 32)),
    ]
    for name, fn in cases:
        print(f"  {name:38s} {_outcome(fn)}")

    print("  -- negative alignment: quiet truncation, or a raise? --")
    k = 2880
    y = torch.randn(8, k, device="cuda", dtype=torch.bfloat16)
    tight, _ = torch.ops.trtllm.mxfp8_quantize(y, False, 32)
    for al in (-32, -64, -512):
        try:
            data, sf = torch.ops.trtllm.mxfp8_quantize(y, False, al)
            padded_k = data.shape[-1]
            same = torch.equal(
                data.view(torch.uint8), tight[:, :padded_k].reshape(data.shape).view(torch.uint8)
            )
            print(
                f"  alignment={al:5d}: ACCEPTED padded_k={padded_k} "
                f"(c-truncating ((K+a-1)/a)*a = {_c_pad_up(k, al)}), "
                f"sf={sf.numel()}, surviving columns bit-exact vs alignment=32: {same}"
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  alignment={al:5d}: RAISED {type(exc).__name__}: {str(exc)[:120]}")

    print("  -- alignment=0 in a CHILD process (the contract claims SIGFPE kills the process) --")
    # A child, because if the claim held this probe would die with it. It does
    # NOT hold here: `mxFp8Quantize.cpp:60` computes
    # `padded_k = ((k + alignment - 1) / alignment) * alignment` with no zero
    # guard, and aarch64's SDIV returns 0 for a zero divisor instead of
    # trapping the way x86's DIV does. So the child prints what the call
    # RETURNS, which is what the contract has to say instead.
    child = (
        "import torch, tensorrt_llm._torch.custom_ops;"
        "x = torch.randn(4, 128, device='cuda', dtype=torch.bfloat16);"
        "d, s = torch.ops.trtllm.mxfp8_quantize(x, False, 0);"
        "print('RETURNED data', tuple(d.shape), str(d.dtype), 'sf', tuple(s.shape),"
        " 'data_bytes', int(d.numel()), 'sf_bytes', int(s.numel()))"
    )
    proc = subprocess.run(
        [sys.executable, "-c", child], capture_output=True, text=True, timeout=600
    )
    tail = proc.stderr.strip().splitlines()
    print(
        f"  alignment=0 child: returncode={proc.returncode} (-8 would be SIGFPE), "
        f"stdout={proc.stdout.strip().splitlines()[-1][:150] if proc.stdout.strip() else ''!r}"
    )
    if proc.returncode != 0:
        print(f"  alignment=0 child stderr tail: {tail[-1][:150] if tail else ''!r}")
    print(f"  host machine: {os.uname().machine} (x86_64 DIV traps on /0; aarch64 SDIV returns 0)")


def main() -> int:
    assert torch.cuda.is_available(), "this is a GPU measurement"
    cap = torch.cuda.get_device_capability()
    import tensorrt_llm

    print(
        f"device={torch.cuda.get_device_name()} sm_{cap[0]}{cap[1]} "
        f"trtllm={tensorrt_llm.__version__} torch={torch.__version__}"
    )
    print(f"tensorrt_llm from {tensorrt_llm.__file__}")
    failed = []
    for fn in (
        probe_projection_audit,
        probe_gemm_target_surfaces,
        probe_quantize_target_widths,
        probe_head_replicated,
        probe_output_lora_bmm,
        probe_cublas_preconditions,
        probe_bmm_preconditions,
        probe_quantize_domains,
    ):
        try:
            fn()
        except Exception as exc:  # noqa: BLE001 — one probe dying must not hide the rest
            print(f"\n!! {fn.__name__} raised {type(exc).__name__}: {exc}")
            import traceback

            traceback.print_exc()
            failed.append(fn.__name__)
    if failed:
        print(f"\nprobe INCOMPLETE: {len(failed)} section(s) raised: {', '.join(failed)}")
        return 1
    print("\nprobe complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
