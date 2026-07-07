# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Comprehensive benchmark: DSL vs C++ top-k for both prefill and decode.

Covers:
  - Prefill fixed-length:  (num_cols, num_rows, top_k)
  - Prefill varlen:        (isl, bs, top_k) — lower-triangular causal sequences
  - Decode fixed-length:   (num_tokens, batch_size, top_k)
  - Decode varlen:         (max_tokens, batch_size, top_k) — random seq_lens

Timing: CuptiProfiler (GPU kernel time only).

Usage:
    python bench_topk_all.py --mode prefill --style fixlen --num_cols 8192
    python bench_topk_all.py --mode prefill --style varlen
    python bench_topk_all.py --mode decode  --style fixlen --num_tokens 32768
    python bench_topk_all.py --mode decode  --style varlen --max_tokens 32768
    python bench_topk_all.py --sweep_all          # run every combination
    python bench_topk_all.py --sweep_prefill       # prefill only (fixlen + varlen)
    python bench_topk_all.py --sweep_decode        # decode only  (fixlen + varlen)
"""

import argparse
import sys
from pathlib import Path

import torch

import tensorrt_llm  # noqa: F401
from tensorrt_llm._torch.custom_ops import cute_dsl_custom_ops  # noqa: F401

try:
    from tests.scripts.cute_dsl_kernels.testing import CuptiProfiler
except ImportError:
    sys.path.insert(0, str(Path(__file__).parents[1]))
    from testing import CuptiProfiler  # type: ignore[no-redef]

IS_B200 = torch.cuda.get_device_capability() >= (10, 0)

_K_MAX_BLOCKS_PER_ROW_DECODE = 10  # mirrors kMaxBlocksPerRowDecode

# Result-dict key suffix per overflow policy (must be unique).
POLICY_SUFFIX = {
    "GMEM_SPILL": "",
    "TRUNCATE": "_t",
    "REREAD_ALWAYS": "_ra",
    "REREAD": "_r",
}

# ── sweep grids ──────────────────────────────────────────────────────────────

TOP_K_LIST = [512, 1024, 2048]

# Prefill fixed-len
PREFILL_NC_LIST = [4096, 8192, 16384, 32768, 65536, 131072, 262144]
PREFILL_NR_LIST = [256, 512, 1024, 2048, 4096, 16384, 32768]

# Prefill varlen (lower-triangular causal; bs*isl <= MAX_CHUNK)
PREFILL_ISL_LIST = [4096, 8192, 16384, 32768]
PREFILL_BS_LIST = [1, 4]
MAX_CHUNK_SIZE = 32768

# Decode fixed-len
DECODE_NT_LIST = [4096, 8192, 16384, 32768, 65536, 131072, 262144]
DECODE_BS_LIST = [1, 16, 32, 64, 128, 256, 384, 512]

# Decode varlen (random seq_lens in [1, max_tokens])
DECODE_VNT_LIST = [8192, 32768, 131072]
DECODE_VBS_LIST = [16, 32, 64, 128, 256, 384, 512]


# ── timing helper ─────────────────────────────────────────────────────────────


def cupti_timer(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    profiler = CuptiProfiler()
    profiler.start()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    profiler.stop()
    return profiler.get_duration() / iters * 1e3  # ms → us


def _has_op(name: str) -> bool:
    try:
        torch.ops.trtllm.__getattr__(name)
        return True
    except (AttributeError, RuntimeError):
        return False


# ── prefill fixed-len ─────────────────────────────────────────────────────────


def bench_prefill_fixlen(
    num_rows: int,
    num_cols: int,
    top_k: int,
    warmup: int,
    iters: int,
    overflow_policies=("GMEM_SPILL",),
) -> dict:
    row_starts = torch.zeros(num_rows, dtype=torch.int32, device="cuda")
    row_ends = torch.full((num_rows,), num_cols, dtype=torch.int32, device="cuda")
    logits_fp32 = torch.randn(num_rows, num_cols, dtype=torch.float32, device="cuda")
    indices_cpp = torch.empty(num_rows, top_k, dtype=torch.int32, device="cuda")
    results = {}

    def run_cpp():
        torch.ops.trtllm.indexer_topk_prefill(logits_fp32, row_starts, row_ends, indices_cpp, top_k)

    results["cpp_fp32"] = cupti_timer(run_cpp, warmup, iters)

    if IS_B200 and _has_op("cute_dsl_indexer_topk_prefill_blackwell"):
        for pol in overflow_policies:
            sfx = POLICY_SUFFIX[pol]

            def run_dsl_fp32(p=pol):
                torch.ops.trtllm.cute_dsl_indexer_topk_prefill_blackwell(
                    logits_fp32, row_starts, row_ends, top_k, overflow_policy=p
                )

            results[f"dsl_fp32{sfx}"] = cupti_timer(run_dsl_fp32, warmup, iters)

            logits_bf16 = logits_fp32.bfloat16()

            def run_dsl_bf16(p=pol):
                torch.ops.trtllm.cute_dsl_indexer_topk_prefill_blackwell(
                    logits_bf16, row_starts, row_ends, top_k, overflow_policy=p
                )

            results[f"dsl_bf16{sfx}"] = cupti_timer(run_dsl_bf16, warmup, iters)

    return results


# ── prefill varlen (lower-triangular causal) ──────────────────────────────────


def _make_causal_varlen(isl: int, bs: int):
    seq_offsets = torch.arange(bs, dtype=torch.int32) * isl
    row_starts = seq_offsets.repeat_interleave(isl).cuda()
    within = torch.tile(torch.arange(1, isl + 1, dtype=torch.int32), (bs,)).cuda()
    row_ends = row_starts + within
    return row_starts, row_ends, bs * isl


def bench_prefill_varlen(
    isl: int, bs: int, top_k: int, warmup: int, iters: int, overflow_policies=("GMEM_SPILL",)
) -> dict:
    num_rows = bs * isl
    row_starts, row_ends, total_kv = _make_causal_varlen(isl, bs)
    logits_fp32 = torch.randn(num_rows, total_kv, dtype=torch.float32, device="cuda")
    indices_cpp = torch.empty(num_rows, top_k, dtype=torch.int32, device="cuda")
    results = {}

    def run_cpp():
        torch.ops.trtllm.indexer_topk_prefill(logits_fp32, row_starts, row_ends, indices_cpp, top_k)

    results["cpp_fp32"] = cupti_timer(run_cpp, warmup, iters)

    if IS_B200 and _has_op("cute_dsl_indexer_topk_prefill_blackwell"):
        for pol in overflow_policies:
            sfx = POLICY_SUFFIX[pol]

            def run_dsl_fp32(p=pol):
                torch.ops.trtllm.cute_dsl_indexer_topk_prefill_blackwell(
                    logits_fp32, row_starts, row_ends, top_k, overflow_policy=p
                )

            results[f"dsl_fp32{sfx}"] = cupti_timer(run_dsl_fp32, warmup, iters)

            logits_bf16 = logits_fp32.bfloat16()

            def run_dsl_bf16(p=pol):
                torch.ops.trtllm.cute_dsl_indexer_topk_prefill_blackwell(
                    logits_bf16, row_starts, row_ends, top_k, overflow_policy=p
                )

            results[f"dsl_bf16{sfx}"] = cupti_timer(run_dsl_bf16, warmup, iters)

    return results


# ── decode fixed-len ──────────────────────────────────────────────────────────


def bench_decode_fixlen(
    batch_size: int,
    num_tokens: int,
    top_k: int,
    next_n: int,
    warmup: int,
    iters: int,
    overflow_policies=("GMEM_SPILL",),
) -> dict:
    num_gen = batch_size * next_n
    seq_lens = torch.full((batch_size,), num_tokens, dtype=torch.int32, device="cuda")
    logits_fp32 = torch.randn(num_gen, num_tokens, dtype=torch.float32, device="cuda")
    aux_idx = torch.empty(
        (num_gen, _K_MAX_BLOCKS_PER_ROW_DECODE, top_k), dtype=torch.int32, device="cuda"
    )
    aux_log = torch.empty(
        (num_gen, _K_MAX_BLOCKS_PER_ROW_DECODE, top_k), dtype=torch.float32, device="cuda"
    )
    indices_cpp = torch.empty((num_gen, top_k), dtype=torch.int32, device="cuda")
    results = {}

    def run_cpp():
        torch.ops.trtllm.indexer_topk_decode(
            logits_fp32,
            seq_lens,
            indices_cpp,
            next_n,
            top_k,
            radix_aux_indices=aux_idx,
            radix_aux_logits=aux_log,
        )

    results["cpp_fp32"] = cupti_timer(run_cpp, warmup, iters)

    if IS_B200 and _has_op("cute_dsl_indexer_topk_decode"):
        for pol in overflow_policies:
            sfx = POLICY_SUFFIX[pol]
            out_fp32 = torch.empty((num_gen, top_k), dtype=torch.int32, device="cuda")

            def run_dsl_fp32(p=pol):
                torch.ops.trtllm.cute_dsl_indexer_topk_decode(
                    input_values=logits_fp32,
                    seq_lens=seq_lens,
                    output_indices=out_fp32,
                    top_k=top_k,
                    next_n=next_n,
                    overflow_policy=p,
                )

            results[f"dsl_fp32{sfx}"] = cupti_timer(run_dsl_fp32, warmup, iters)

            logits_bf16 = logits_fp32.bfloat16()
            out_bf16 = torch.empty((num_gen, top_k), dtype=torch.int32, device="cuda")

            def run_dsl_bf16(p=pol):
                torch.ops.trtllm.cute_dsl_indexer_topk_decode(
                    input_values=logits_bf16,
                    seq_lens=seq_lens,
                    output_indices=out_bf16,
                    top_k=top_k,
                    next_n=next_n,
                    overflow_policy=p,
                )

            results[f"dsl_bf16{sfx}"] = cupti_timer(run_dsl_bf16, warmup, iters)

    return results


# ── decode varlen (random seq_lens) ───────────────────────────────────────────


def bench_decode_varlen(
    batch_size: int,
    max_tokens: int,
    top_k: int,
    next_n: int,
    warmup: int,
    iters: int,
    overflow_policies=("GMEM_SPILL",),
) -> dict:
    num_gen = batch_size * next_n
    torch.manual_seed(42)
    seq_lens = torch.randint(1, max_tokens + 1, (batch_size,), dtype=torch.int32).cuda()
    logits_fp32 = torch.randn(num_gen, max_tokens, dtype=torch.float32, device="cuda")
    aux_idx = torch.empty(
        (num_gen, _K_MAX_BLOCKS_PER_ROW_DECODE, top_k), dtype=torch.int32, device="cuda"
    )
    aux_log = torch.empty(
        (num_gen, _K_MAX_BLOCKS_PER_ROW_DECODE, top_k), dtype=torch.float32, device="cuda"
    )
    indices_cpp = torch.empty((num_gen, top_k), dtype=torch.int32, device="cuda")
    results = {}

    def run_cpp():
        torch.ops.trtllm.indexer_topk_decode(
            logits_fp32,
            seq_lens,
            indices_cpp,
            next_n,
            top_k,
            radix_aux_indices=aux_idx,
            radix_aux_logits=aux_log,
        )

    results["cpp_fp32"] = cupti_timer(run_cpp, warmup, iters)

    if IS_B200 and _has_op("cute_dsl_indexer_topk_decode"):
        for pol in overflow_policies:
            sfx = POLICY_SUFFIX[pol]
            out_fp32 = torch.empty((num_gen, top_k), dtype=torch.int32, device="cuda")

            def run_dsl_fp32(p=pol):
                torch.ops.trtllm.cute_dsl_indexer_topk_decode(
                    input_values=logits_fp32,
                    seq_lens=seq_lens,
                    output_indices=out_fp32,
                    top_k=top_k,
                    next_n=next_n,
                    overflow_policy=p,
                )

            results[f"dsl_fp32{sfx}"] = cupti_timer(run_dsl_fp32, warmup, iters)

            logits_bf16 = logits_fp32.bfloat16()
            out_bf16 = torch.empty((num_gen, top_k), dtype=torch.int32, device="cuda")

            def run_dsl_bf16(p=pol):
                torch.ops.trtllm.cute_dsl_indexer_topk_decode(
                    input_values=logits_bf16,
                    seq_lens=seq_lens,
                    output_indices=out_bf16,
                    top_k=top_k,
                    next_n=next_n,
                    overflow_policy=p,
                )

            results[f"dsl_bf16{sfx}"] = cupti_timer(run_dsl_bf16, warmup, iters)

    return results


# ── print helpers ─────────────────────────────────────────────────────────────


def _header(variants, col_w=13):
    cols = "  ".join(f"{v:>{col_w}}" for v in variants)
    sp = "  ".join(f"{'cpp/dsl_fp32':>{col_w}}" for _ in ["fp32"])
    sp += "  " + f"{'cpp/dsl_bf16':>{col_w}}"
    return cols, sp


def _print_row(key1, key2, res, variants, col_w=13, extra_policies=()):
    line = f"{key1:>6}  {key2:>7}"
    for v in variants:
        us = res.get(v, float("nan"))
        line += f"  {us:>{col_w}.2f}us"
    if IS_B200:
        cpp = res.get("cpp_fp32", float("nan"))
        for v in ["dsl_fp32", "dsl_bf16"]:
            if res.get(v):
                sp = cpp / res[v]
                line += f"  {sp:>{col_w}.3f}x"
        for pol in extra_policies:
            sfx = POLICY_SUFFIX[pol]
            for dtype in ["fp32", "bf16"]:
                base, alt = f"dsl_{dtype}", f"dsl_{dtype}{sfx}"
                if res.get(base) and res.get(alt):
                    ratio = res[base] / res[alt]  # >1 means alt is faster
                    line += f"  {ratio:>{col_w}.3f}x"
        if "REREAD" in extra_policies:
            sfx = POLICY_SUFFIX["REREAD"]
            for dtype in ["fp32", "bf16"]:
                r_col = f"dsl_{dtype}{sfx}"
                if res.get(r_col):
                    ratio = cpp / res[r_col]  # >1 means REREAD faster than C++
                    line += f"  {ratio:>{col_w}.3f}x"
    print(line)


# ── sweep runners ─────────────────────────────────────────────────────────────


POLICY_HEADER_TAG = {
    "GMEM_SPILL": "",
    "TRUNCATE": "T",
    "REREAD_ALWAYS": "RA",
    "REREAD": "R",
}


def _make_variants_and_header_suffix(overflow_policies, col_w):
    dsl_variants = []
    hdr_suffix = ""
    for pol in overflow_policies:
        sfx = POLICY_SUFFIX[pol]
        dsl_variants += [f"dsl_fp32{sfx}", f"dsl_bf16{sfx}"]
    variants = ["cpp_fp32"] + (dsl_variants if IS_B200 else [])
    if IS_B200:
        hdr_suffix += f"  {'cpp/dsl_fp32':>{col_w}}  {'cpp/dsl_bf16':>{col_w}}"
        for pol in overflow_policies:
            if pol == "GMEM_SPILL":
                continue
            tag = POLICY_HEADER_TAG[pol]
            hdr_suffix += f"  {f'G/{tag}_fp32':>{col_w}}  {f'G/{tag}_bf16':>{col_w}}"
        if "REREAD" in overflow_policies:
            hdr_suffix += f"  {'cpp/R_fp32':>{col_w}}  {'cpp/R_bf16':>{col_w}}"
    return variants, hdr_suffix


def sweep_prefill_fixlen(warmup, iters, overflow_policies=("GMEM_SPILL",)):
    col_w = 13
    extra = [p for p in overflow_policies if p != "GMEM_SPILL"]
    for nc in PREFILL_NC_LIST:
        variants, hdr_sfx = _make_variants_and_header_suffix(overflow_policies, col_w)
        print(f"\n=== mode=prefill style=fixlen num_cols={nc} ===")
        hdr = f"{'top_k':>6}  {'num_rows':>7}"
        for v in variants:
            hdr += f"  {v:>{col_w}}"
        hdr += hdr_sfx
        print(hdr)
        print("-" * len(hdr))
        for tk in TOP_K_LIST:
            for nr in PREFILL_NR_LIST:
                try:
                    res = bench_prefill_fixlen(nr, nc, tk, warmup, iters, overflow_policies)
                    _print_row(tk, nr, res, variants, col_w, extra)
                except Exception as e:
                    print(f"{tk:>6}  {nr:>7}  ERROR: {e}")
            print()


def sweep_prefill_varlen(warmup, iters, overflow_policies=("GMEM_SPILL",)):
    col_w = 13
    extra = [p for p in overflow_policies if p != "GMEM_SPILL"]
    for isl in PREFILL_ISL_LIST:
        variants, hdr_sfx = _make_variants_and_header_suffix(overflow_policies, col_w)
        print(f"\n=== mode=prefill style=varlen isl={isl} ===")
        hdr = f"{'top_k':>6}  {'bs':>7}"
        for v in variants:
            hdr += f"  {v:>{col_w}}"
        hdr += hdr_sfx
        print(hdr)
        print("-" * len(hdr))
        for tk in TOP_K_LIST:
            for bs in PREFILL_BS_LIST:
                if bs * isl > MAX_CHUNK_SIZE:
                    continue
                try:
                    res = bench_prefill_varlen(isl, bs, tk, warmup, iters, overflow_policies)
                    _print_row(tk, bs, res, variants, col_w, extra)
                except Exception as e:
                    print(f"{tk:>6}  {bs:>7}  ERROR: {e}")
            print()


def sweep_decode_fixlen(warmup, iters, next_n=1, overflow_policies=("GMEM_SPILL",)):
    col_w = 13
    extra = [p for p in overflow_policies if p != "GMEM_SPILL"]
    for nt in DECODE_NT_LIST:
        variants, hdr_sfx = _make_variants_and_header_suffix(overflow_policies, col_w)
        print(f"\n=== mode=decode style=fixlen num_tokens={nt} next_n={next_n} ===")
        hdr = f"{'top_k':>6}  {'batch':>7}"
        for v in variants:
            hdr += f"  {v:>{col_w}}"
        hdr += hdr_sfx
        print(hdr)
        print("-" * len(hdr))
        for tk in TOP_K_LIST:
            for bs in DECODE_BS_LIST:
                try:
                    res = bench_decode_fixlen(bs, nt, tk, next_n, warmup, iters, overflow_policies)
                    _print_row(tk, bs, res, variants, col_w, extra)
                except Exception as e:
                    print(f"{tk:>6}  {bs:>7}  ERROR: {e}")
            print()


def sweep_decode_varlen(warmup, iters, next_n=1, overflow_policies=("GMEM_SPILL",)):
    col_w = 13
    extra = [p for p in overflow_policies if p != "GMEM_SPILL"]
    for nt in DECODE_VNT_LIST:
        variants, hdr_sfx = _make_variants_and_header_suffix(overflow_policies, col_w)
        print(f"\n=== mode=decode style=varlen max_tokens={nt} next_n={next_n} ===")
        hdr = f"{'top_k':>6}  {'batch':>7}"
        for v in variants:
            hdr += f"  {v:>{col_w}}"
        hdr += hdr_sfx
        print(hdr)
        print("-" * len(hdr))
        for tk in TOP_K_LIST:
            for bs in DECODE_VBS_LIST:
                try:
                    res = bench_decode_varlen(bs, nt, tk, next_n, warmup, iters, overflow_policies)
                    _print_row(tk, bs, res, variants, col_w, extra)
                except Exception as e:
                    print(f"{tk:>6}  {bs:>7}  ERROR: {e}")
            print()


# ── main ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--mode", choices=["prefill", "decode"], default="prefill")
    parser.add_argument("--style", choices=["fixlen", "varlen"], default="fixlen")
    parser.add_argument("--num_cols", type=int, default=8192)
    parser.add_argument("--num_rows", type=int, default=4096)
    parser.add_argument("--num_tokens", type=int, default=32768)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--isl", type=int, default=4096)
    parser.add_argument("--bs", type=int, default=1)
    parser.add_argument("--max_tokens", type=int, default=32768)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--next_n", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument(
        "--compare_policies",
        action="store_true",
        help="bench GMEM_SPILL, TRUNCATE, REREAD_ALWAYS, and REREAD side by side",
    )
    parser.add_argument("--sweep_all", action="store_true")
    parser.add_argument("--sweep_prefill", action="store_true")
    parser.add_argument("--sweep_decode", action="store_true")
    parser.add_argument("--sweep_prefill_fixlen", action="store_true")
    parser.add_argument("--sweep_prefill_varlen", action="store_true")
    parser.add_argument("--sweep_decode_fixlen", action="store_true")
    parser.add_argument("--sweep_decode_varlen", action="store_true")
    args = parser.parse_args()

    overflow_policies = (
        ("GMEM_SPILL", "TRUNCATE", "REREAD_ALWAYS", "REREAD")
        if args.compare_policies
        else ("GMEM_SPILL",)
    )

    cap = torch.cuda.get_device_capability()
    print(f"[TensorRT-LLM] Device: {torch.cuda.get_device_name(0)}  (sm_{cap[0]}{cap[1]})")
    print("Timing: CuptiProfiler (GPU kernel time only)")
    print(f"overflow_policies: {overflow_policies}")
    if not IS_B200:
        print("DSL kernel requires Blackwell (sm100+); DSL columns will be absent.")

    do_pf = args.sweep_all or args.sweep_prefill or args.sweep_prefill_fixlen
    do_pv = args.sweep_all or args.sweep_prefill or args.sweep_prefill_varlen
    do_df = args.sweep_all or args.sweep_decode or args.sweep_decode_fixlen
    do_dv = args.sweep_all or args.sweep_decode or args.sweep_decode_varlen
    if do_pf:
        sweep_prefill_fixlen(args.warmup, args.iters, overflow_policies)
    if do_pv:
        sweep_prefill_varlen(args.warmup, args.iters, overflow_policies)
    if do_df:
        sweep_decode_fixlen(args.warmup, args.iters, args.next_n, overflow_policies)
    if do_dv:
        sweep_decode_varlen(args.warmup, args.iters, args.next_n, overflow_policies)

    if not (args.sweep_all or args.sweep_prefill or args.sweep_decode):
        tk_list = [args.top_k] if args.top_k else TOP_K_LIST
        if args.mode == "prefill" and args.style == "fixlen":
            res = bench_prefill_fixlen(
                args.num_rows, args.num_cols, tk_list[0], args.warmup, args.iters, overflow_policies
            )
            print(res)
        elif args.mode == "prefill" and args.style == "varlen":
            res = bench_prefill_varlen(
                args.isl, args.bs, tk_list[0], args.warmup, args.iters, overflow_policies
            )
            print(res)
        elif args.mode == "decode" and args.style == "fixlen":
            res = bench_decode_fixlen(
                args.batch_size,
                args.num_tokens,
                tk_list[0],
                args.next_n,
                args.warmup,
                args.iters,
                overflow_policies,
            )
            print(res)
        elif args.mode == "decode" and args.style == "varlen":
            res = bench_decode_varlen(
                args.batch_size,
                args.max_tokens,
                tk_list[0],
                args.next_n,
                args.warmup,
                args.iters,
                overflow_policies,
            )
            print(res)


if __name__ == "__main__":
    main()
