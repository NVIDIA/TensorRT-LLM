# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Latency of the MiniMax-M3 indexer decode scorer, CuTe DSL vs the MSA proxy.

Both produce the per-block index scores the selector ranks: the CuTe DSL
kernel on a resolved decode span, the fmha_sm100 output_maxscore pass on
everything else. Run this on one SM100 GPU to see the per-call cost of each at
a given batch size and context length; no model weights are needed.

    python tests/microbenchmarks/minimax_m3_index_decode_score.py \
        --batch 1 --seq-len 8192 --num-heads 1
"""

import argparse

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401
from tensorrt_llm._torch.attention_backend.sparse.minimax_m3.msa_indexer import _proxy_max_score
from tensorrt_llm._torch.attention_backend.sparse.minimax_m3.msa_utils import (
    build_kv_page_indices,
    msa_package_available,
)

PAGE_SIZE = 128
HEAD_DIM = 128


def _time_us(fn, warmup: int = 20, iters: int = 100) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1000.0 / iters


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=8192)
    parser.add_argument("--num-heads", type=int, default=1, help="Sharded index heads.")
    parser.add_argument("--decode-query-len", type=int, default=1)
    parser.add_argument("--dtype", choices=("bfloat16", "fp8_e4m3"), default="fp8_e4m3")
    args = parser.parse_args()

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float8_e4m3fn
    batch, seq_len, num_heads = args.batch, args.seq_len, args.num_heads
    dql = args.decode_query_len
    total_q = batch * dql
    num_blocks = (seq_len + PAGE_SIZE - 1) // PAGE_SIZE
    score_width = ((num_blocks + 15) // 16) * 16
    num_pages = batch * num_blocks

    block_table = (
        torch.randperm(num_pages, device="cuda").to(torch.int32).reshape(batch, num_blocks)
    )
    seq_lens = torch.full((batch,), seq_len, device="cuda", dtype=torch.int32)
    idx_q = torch.randn(total_q, num_heads, HEAD_DIM, device="cuda").to(dtype)
    k_cache = torch.randn(num_pages, PAGE_SIZE, HEAD_DIM, device="cuda").to(dtype)
    backing = torch.full((num_heads, score_width, total_q), -float("inf"), device="cuda")
    score = backing.transpose(1, 2)

    def run_cutedsl():
        torch.ops.trtllm.cute_dsl_minimax_m3_index_decode_score(
            idx_q, k_cache, block_table, seq_lens, score, dql
        )

    print(f"batch={batch} seq_len={seq_len} heads={num_heads} dql={dql} dtype={args.dtype}")
    print(f"  cutedsl : {_time_us(run_cutedsl):8.2f} us/call")

    if not msa_package_available():
        print("  msa     :   skipped (fmha_sm100 submodule not available)")
        return

    qo_lens_cpu = torch.full((batch,), dql, dtype=torch.int32)
    kv_lens_cpu = torch.full((batch,), seq_len, dtype=torch.int32)
    kv_indices = build_kv_page_indices(block_table.cpu(), kv_lens_cpu, PAGE_SIZE).cuda()
    k_paged = k_cache.unsqueeze(1)

    def run_msa():
        _proxy_max_score(
            idx_q,
            k_paged,
            qo_lens_cpu=qo_lens_cpu,
            kv_lens_cpu=kv_lens_cpu,
            qo_offset_cpu=kv_lens_cpu - qo_lens_cpu,
            kv_indices=kv_indices,
            sm_scale=HEAD_DIM**-0.5,
            causal=True,
        )

    print(f"  msa     : {_time_us(run_msa):8.2f} us/call")


if __name__ == "__main__":
    main()
