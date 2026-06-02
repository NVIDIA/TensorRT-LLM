# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Standalone benchmark for the FP4 MLA decode kernel.

Compares the Triton and CuTile FP4 backends against the
trtllm-gen fp8 ("trtllm_fp8") baseline.
Run with:
    python bench_fp4_mla_decode.py [--batch B] [--seq S] [--heads H] [--q-len Q]

--q-len (alias --mtp-len) sets the number of query tokens per sequence (>1 for
MTP / speculative decoding); it defaults to 1 (plain decode).
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "tests/unittest/_torch/attention"))

os.environ.setdefault("TRTLLM_FLASHINFER_FP4_MLA_ATTENTION", "1")

import flashinfer  # noqa: E402
from test_fp4_mla import _build_fp4_mla_attention_decode_case  # noqa: E402

from tensorrt_llm._torch.attention_backend.fp4_mla import (  # noqa: E402
    FLASHINFER_FP4_MLA_ATTENTION_BACKEND_ENV,
    FLASHINFER_FP4_MLA_ATTENTION_ENV,
    FP4_MLA_Q_RESIDUAL_DIM,
    run_fp4_mla_attention_decode,
)

BACKEND_CHOICES = (
    "trtllm_fp8",
    "triton",
    "cutile",
)


def _bench(fn, warmup=0, iters=1):
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


def _seq_lens_for_batch(batch, seq):
    return [seq] * batch


def _seq_label(seq_lens):
    return f"{seq_lens[0]}-{seq_lens[-1]}" if len(seq_lens) > 1 else str(seq_lens[0])


def _kernel_io_bytes(seq_lens, heads, kv_lora_rank, qk_rope_head_dim, q_len=1):
    """Per-kernel-call HBM input + output bytes (FP4 = 0.5 B, scale = 1 B, output = 2 B)."""
    batch = len(seq_lens)
    total_seq = sum(seq_lens)
    q_head_dim = kv_lora_rank + qk_rope_head_dim + FP4_MLA_Q_RESIDUAL_DIM
    k_head_dim = kv_lora_rank + qk_rope_head_dim
    q_sf_per_token = q_head_dim // FP4_BLOCK_SIZE
    k_sf_per_token = k_head_dim // FP4_BLOCK_SIZE
    pages = sum((seq_len + PAGE_SIZE - 1) // PAGE_SIZE for seq_len in seq_lens)
    sf_per_page = PAGE_SIZE // FP4_BLOCK_SIZE

    q_fp4 = batch * q_len * heads * q_head_dim // 2
    q_sf = batch * q_len * heads * q_sf_per_token
    kv_cache = total_seq * k_head_dim // 2
    k_sf_cache = total_seq * k_sf_per_token
    v_sf_cache = pages * kv_lora_rank * sf_per_page
    out = batch * q_len * heads * kv_lora_rank * 2  # bf16/half
    return q_fp4 + q_sf + kv_cache + k_sf_cache + v_sf_cache + out


def _trtllm_mla_io_bytes(seq_lens, heads, kv_lora_rank, qk_rope_head_dim, q_len=1, elem_bytes=2):
    """Per-call HBM bytes for the trtllm-gen MLA decode baseline.

    ``elem_bytes`` is the byte width of the Q and KV-cache elements (2 for bf16,
    1 for fp8).  The output is always written as bf16 (2 B/elem).
    """
    batch = len(seq_lens)
    total_seq = sum(seq_lens)
    head_dim = kv_lora_rank + qk_rope_head_dim
    q = batch * q_len * heads * head_dim * elem_bytes
    kv = total_seq * head_dim * elem_bytes  # ckv + kpe paged caches
    out = batch * q_len * heads * kv_lora_rank * 2
    return q + kv + out


def run_one_trtllm(batch, seq, heads, q_len=1, warmup=0, iters=1):
    """Fp8 baseline using the FlashInfer trtllm-gen MLA decode kernel.

    Feeds fp8 (e4m3) Q and KV cache so the kernel uses fp8 tensor cores
    (output stays bf16).
    """
    device = torch.device("cuda")
    label = "trtllm_fp8"
    io_dtype = torch.float8_e4m3fn
    elem_bytes = 1
    kv_lora_rank = 512
    qk_rope_head_dim = 64
    qk_nope_head_dim = 128  # DeepSeek-V3 default; only used for fused scale convention.
    head_dim_qk = kv_lora_rank + qk_rope_head_dim
    page_size = 64  # trtllm-gen MLA decode only supports page_size of 32 or 64.
    seq_lens_list = _seq_lens_for_batch(batch, seq)
    max_seq = max(seq_lens_list)
    total_seq = sum(seq_lens_list)
    blocks_per_seq = [(seq_len + page_size - 1) // page_size for seq_len in seq_lens_list]
    max_blocks_per_seq = max(blocks_per_seq)
    total_pages = sum(blocks_per_seq)

    torch.manual_seed(8)
    # query layout: [batch, q_len, heads, kv_lora_rank + qk_rope_head_dim]
    query = torch.randn(batch, q_len, heads, head_dim_qk, dtype=torch.bfloat16, device=device)
    # kv_cache layout: [num_pages, page_size, head_dim_ckv + head_dim_kpe]
    kv_cache = torch.randn(total_pages, page_size, head_dim_qk, dtype=torch.bfloat16, device=device)
    # Quantize to fp8 e4m3 (randn ~ N(0,1) is well within e4m3 range).
    query = query.to(io_dtype)
    kv_cache = kv_cache.to(io_dtype)

    block_tables = torch.zeros((batch, max_blocks_per_seq), dtype=torch.int32, device=device)
    page_start = 0
    for batch_idx, num_blocks in enumerate(blocks_per_seq):
        block_tables[batch_idx, :num_blocks] = torch.arange(
            page_start, page_start + num_blocks, dtype=torch.int32, device=device
        )
        page_start += num_blocks
    seq_lens = torch.tensor(seq_lens_list, dtype=torch.int32, device=device)
    workspace = torch.zeros(128 * 1024 * 1024, dtype=torch.int8, device=device).view(-1, 4)

    output = torch.empty(batch, q_len, heads, kv_lora_rank, dtype=torch.bfloat16, device=device)

    # bmm1_scale folds q_scale * k_scale * sm_scale / sqrt(head_dim_qk); a
    # representative value (q_scale = k_scale = 1.0 for the fp8 unit-scale tensors).
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
            max_seq_len=max_seq,
            out=output,
            bmm1_scale=0.1,
            bmm2_scale=1.0,
            backend="trtllm-gen",
        )

    run()
    torch.cuda.synchronize()
    avg_ms = _bench(run, warmup=warmup, iters=iters)
    qk_dim = kv_lora_rank + qk_rope_head_dim
    pv_dim = kv_lora_rank
    flops = 2 * q_len * heads * total_seq * (qk_dim + pv_dim)
    tflops = flops / avg_ms / 1e9
    bytes_per_call = _trtllm_mla_io_bytes(
        seq_lens_list, heads, kv_lora_rank, qk_rope_head_dim, q_len, elem_bytes
    )
    gb_s = bytes_per_call / (avg_ms * 1e-3) / 1e9
    hbm_pct = 100.0 * gb_s / HBM_PEAK_GB_S
    print(
        f"backend={label:>12s} bs={batch:>3d} seq={_seq_label(seq_lens_list):>11s} "
        f"heads={heads:>3d} qlen={q_len:>2d}: "
        f"{avg_ms:>7.3f} ms  {tflops:>6.2f} TFLOP/s  "
        f"HBM {gb_s:>6.1f} GB/s ({hbm_pct:>4.1f}% of {HBM_PEAK_GB_S:.0f})",
        flush=True,
    )
    return avg_ms


def run_one(batch, seq, heads, backend, q_len=1, warmup=0, iters=1):
    os.environ[FLASHINFER_FP4_MLA_ATTENTION_ENV] = "1"
    os.environ[FLASHINFER_FP4_MLA_ATTENTION_BACKEND_ENV] = backend
    seq_lens = _seq_lens_for_batch(batch, seq)
    total_seq = sum(seq_lens)
    (
        kv_cache_manager,
        metadata,
        q_nope,
        q_pe,
        kv_lora_rank,
        qk_rope_head_dim,
    ) = _build_fp4_mla_attention_decode_case(
        seq_lens=seq_lens, num_heads=heads, seed=8, query_len_per_seq=q_len
    )
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
        avg_ms = _bench(run, warmup=warmup, iters=iters)
        qk_dim = kv_lora_rank + qk_rope_head_dim + FP4_MLA_Q_RESIDUAL_DIM
        pv_dim = kv_lora_rank
        flops = 2 * q_len * heads * total_seq * (qk_dim + pv_dim)
        tflops = flops / avg_ms / 1e9
        # HBM bandwidth = sum of global-mem reads (Q, Q-SF, KV, K-SF, V-SF) + output writes per
        # call.  Note: typically <1% of peak because L2 absorbs the KV reuse and the actual
        # hot lane is SMEM (~85% of peak L1/SMEM throughput per ncu).  HBM low ~= good caching.
        bytes_per_call = _kernel_io_bytes(seq_lens, heads, kv_lora_rank, qk_rope_head_dim, q_len)
        gb_s = bytes_per_call / (avg_ms * 1e-3) / 1e9
        hbm_pct = 100.0 * gb_s / HBM_PEAK_GB_S
        print(
            f"backend={backend:>12s} bs={batch:>3d} seq={_seq_label(seq_lens):>11s} "
            f"heads={heads:>3d} qlen={q_len:>2d}: "
            f"{avg_ms:>7.3f} ms  {tflops:>6.2f} TFLOP/s  "
            f"HBM {gb_s:>6.1f} GB/s ({hbm_pct:>4.1f}% of {HBM_PEAK_GB_S:.0f})",
            flush=True,
        )
        return avg_ms
    finally:
        kv_cache_manager.shutdown()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batch", type=int, nargs="+", default=None)
    p.add_argument("--seq", type=int, default=30080)
    p.add_argument("--heads", type=int, default=128)
    p.add_argument(
        "--q-len",
        "--mtp-len",
        dest="q_len",
        type=int,
        default=1,
        help="Query tokens per sequence (>1 for MTP / speculative decoding).",
    )
    p.add_argument(
        "--backend",
        default=None,
        choices=BACKEND_CHOICES,
        help=("Backend to benchmark; default runs all fast backends."),
    )
    p.add_argument("--warmup", type=int, default=0)
    p.add_argument("--iters", type=int, default=1)
    args = p.parse_args()

    batches = args.batch if args.batch else [16, 30, 60, 120, 200, 300]
    if args.backend:
        backends = [args.backend]
    else:
        backends = list(BACKEND_CHOICES)

    for b in batches:
        for be in backends:
            if be == "trtllm_fp8":
                run_one_trtllm(b, args.seq, args.heads, args.q_len, args.warmup, args.iters)
            else:
                run_one(b, args.seq, args.heads, be, args.q_len, args.warmup, args.iters)


if __name__ == "__main__":
    main()
