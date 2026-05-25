# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Standalone benchmark for the FP4 MLA decode kernel.

Compares the Triton, CuTile, and CuTe DSL FP4 backends against the
trtllm-gen bf16 baseline.
Run with:
    python bench_fp4_mla_decode.py [--batch B] [--seq S] [--heads H]
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "tests/unittest/_torch/attention"))

os.environ.setdefault("TRTLLM_FLASHINFER_FP4_MLA_ATTENTION", "1")

import flashinfer  # noqa: E402
from test_fp4_mla_kv import _build_fp4_mla_attention_decode_case  # noqa: E402

from tensorrt_llm._torch.attention_backend.fp4_mla_kv import (  # noqa: E402
    FLASHINFER_FP4_MLA_ATTENTION_BACKEND_ENV,
    FLASHINFER_FP4_MLA_ATTENTION_ENV,
    FP4_MLA_Q_RESIDUAL_DIM,
    run_fp4_mla_attention_decode,
)

BACKEND_CHOICES = ("trtllm", "triton", "cutile", "cute_dsl")


def _bench(fn, warmup=10, iters=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


# B200 HBM3e peak bandwidth (~8 TB/s).  Override via TRTLLM_HBM_GB_S=<value> if needed.
HBM_PEAK_GB_S = float(os.environ.get("TRTLLM_HBM_GB_S", "8000"))
FP4_BLOCK_SIZE = 16  # E2M1 scale-group size used by the kernel
PAGE_SIZE = 128


def _kernel_io_bytes(batch, seq, heads, kv_lora_rank, qk_rope_head_dim):
    """Per-kernel-call HBM input + output bytes (FP4 = 0.5 B, scale = 1 B, output = 2 B)."""
    q_head_dim = kv_lora_rank + qk_rope_head_dim + FP4_MLA_Q_RESIDUAL_DIM
    k_head_dim = kv_lora_rank + qk_rope_head_dim
    q_sf_per_token = q_head_dim // FP4_BLOCK_SIZE
    k_sf_per_token = k_head_dim // FP4_BLOCK_SIZE
    pages = (seq + PAGE_SIZE - 1) // PAGE_SIZE
    sf_per_page = PAGE_SIZE // FP4_BLOCK_SIZE

    q_fp4 = batch * heads * q_head_dim // 2
    q_sf = batch * heads * q_sf_per_token
    kv_cache = batch * seq * k_head_dim // 2
    k_sf_cache = batch * seq * k_sf_per_token
    v_sf_cache = batch * pages * kv_lora_rank * sf_per_page
    out = batch * heads * kv_lora_rank * 2  # bf16/half
    return q_fp4 + q_sf + kv_cache + k_sf_cache + v_sf_cache + out


def _bf16_mla_io_bytes(batch, seq, heads, kv_lora_rank, qk_rope_head_dim):
    """Per-call HBM bytes for the bf16 MLA decode baseline (2 B/elem)."""
    head_dim = kv_lora_rank + qk_rope_head_dim
    q = batch * heads * head_dim * 2
    kv = batch * seq * head_dim * 2  # ckv + kpe paged caches
    out = batch * heads * kv_lora_rank * 2
    return q + kv + out


def run_one_trtllm(batch, seq, heads):
    """Bf16 baseline using the FlashInfer trtllm-gen MLA decode kernel."""
    device = torch.device("cuda")
    kv_lora_rank = 512
    qk_rope_head_dim = 64
    qk_nope_head_dim = 128  # DeepSeek-V3 default; only used for fused scale convention.
    head_dim_qk = kv_lora_rank + qk_rope_head_dim
    page_size = 64  # trtllm-gen MLA decode only supports page_size of 32 or 64.
    blocks_per_seq = (seq + page_size - 1) // page_size
    total_pages = batch * blocks_per_seq

    torch.manual_seed(8)
    # query layout: [batch, q_len=1, heads, kv_lora_rank + qk_rope_head_dim]
    query = torch.randn(batch, 1, heads, head_dim_qk, dtype=torch.bfloat16, device=device)
    # kv_cache layout: [num_pages, page_size, head_dim_ckv + head_dim_kpe]
    kv_cache = torch.randn(total_pages, page_size, head_dim_qk, dtype=torch.bfloat16, device=device)

    block_tables = torch.arange(total_pages, dtype=torch.int32, device=device).view(
        batch, blocks_per_seq
    )
    seq_lens = torch.full((batch,), seq, dtype=torch.int32, device=device)
    workspace = torch.zeros(128 * 1024 * 1024, dtype=torch.int8, device=device).view(-1, 4)

    output = torch.empty(batch, 1, heads, kv_lora_rank, dtype=torch.bfloat16, device=device)

    def run():
        flashinfer.mla.trtllm_batch_decode_with_kv_cache_mla(
            query=query,
            kv_cache=kv_cache,
            workspace_buffer=workspace,
            qk_nope_head_dim=qk_nope_head_dim,
            kv_lora_rank=kv_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
            block_tables=block_tables,
            seq_lens=seq_lens,
            max_seq_len=seq,
            out=output,
            bmm1_scale=0.1,
            bmm2_scale=1.0,
        )

    run()
    torch.cuda.synchronize()
    avg_ms = _bench(run)
    qk_dim = kv_lora_rank + qk_rope_head_dim
    pv_dim = kv_lora_rank
    flops = 2 * batch * heads * seq * (qk_dim + pv_dim)
    tflops = flops / avg_ms / 1e9
    bytes_per_call = _bf16_mla_io_bytes(batch, seq, heads, kv_lora_rank, qk_rope_head_dim)
    gb_s = bytes_per_call / (avg_ms * 1e-3) / 1e9
    hbm_pct = 100.0 * gb_s / HBM_PEAK_GB_S
    print(
        f"backend={'trtllm':>12s} bs={batch:>3d} seq={seq:>5d} heads={heads:>3d}: "
        f"{avg_ms:>7.3f} ms  {tflops:>6.2f} TFLOP/s  "
        f"HBM {gb_s:>6.1f} GB/s ({hbm_pct:>4.1f}% of {HBM_PEAK_GB_S:.0f})",
        flush=True,
    )
    return avg_ms


def run_one(batch, seq, heads, backend):
    os.environ[FLASHINFER_FP4_MLA_ATTENTION_ENV] = "1"
    os.environ[FLASHINFER_FP4_MLA_ATTENTION_BACKEND_ENV] = backend
    (
        kv_cache_manager,
        metadata,
        q_nope,
        q_pe,
        kv_lora_rank,
        qk_rope_head_dim,
    ) = _build_fp4_mla_attention_decode_case(seq_lens=[seq] * batch, num_heads=heads, seed=8)
    try:
        output = torch.empty_like(q_nope)

        def run():
            run_fp4_mla_attention_decode(
                metadata,
                layer_idx=0,
                local_layer=0,
                q_nope=q_nope,
                q_pe=q_pe,
                output=output,
                sm_scale=0.1,
                kv_lora_rank=kv_lora_rank,
                qk_rope_head_dim=qk_rope_head_dim,
            )

        run()
        torch.cuda.synchronize()
        avg_ms = _bench(run)
        qk_dim = kv_lora_rank + qk_rope_head_dim + FP4_MLA_Q_RESIDUAL_DIM
        pv_dim = kv_lora_rank
        flops = 2 * batch * heads * seq * (qk_dim + pv_dim)
        tflops = flops / avg_ms / 1e9
        # HBM bandwidth = sum of global-mem reads (Q, Q-SF, KV, K-SF, V-SF) + output writes per
        # call.  Note: typically <1% of peak because L2 absorbs the KV reuse and the actual
        # hot lane is SMEM (~85% of peak L1/SMEM throughput per ncu).  HBM low ~= good caching.
        bytes_per_call = _kernel_io_bytes(batch, seq, heads, kv_lora_rank, qk_rope_head_dim)
        gb_s = bytes_per_call / (avg_ms * 1e-3) / 1e9
        hbm_pct = 100.0 * gb_s / HBM_PEAK_GB_S
        print(
            f"backend={backend:>12s} bs={batch:>3d} seq={seq:>5d} heads={heads:>3d}: "
            f"{avg_ms:>7.3f} ms  {tflops:>6.2f} TFLOP/s  "
            f"HBM {gb_s:>6.1f} GB/s ({hbm_pct:>4.1f}% of {HBM_PEAK_GB_S:.0f})",
            flush=True,
        )
        return avg_ms
    finally:
        kv_cache_manager.shutdown()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batch", type=int, default=None)
    p.add_argument("--seq", type=int, default=32768)
    p.add_argument("--heads", type=int, default=128)
    p.add_argument(
        "--backend",
        default=None,
        choices=BACKEND_CHOICES,
        help=("Backend to benchmark; default runs all fast backends."),
    )
    args = p.parse_args()

    batches = [args.batch] if args.batch else [16, 32, 64, 128, 256]
    if args.backend:
        backends = [args.backend]
    else:
        backends = list(BACKEND_CHOICES)

    for b in batches:
        for be in backends:
            if be == "trtllm":
                run_one_trtllm(b, args.seq, args.heads)
            else:
                run_one(b, args.seq, args.heads, be)


if __name__ == "__main__":
    main()
