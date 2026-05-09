# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Standalone launch script for the CuTe DSL FP4 paged MQA logits kernel.

Generates random inputs, runs the kernel, and compares against a reference
implementation. Intended for local development / sanity checks; not wired into
CI (matching the convention of other scripts under ``tests/scripts/cute_dsl_kernels/``).

Example:
    python run_fp4.py --batch_size 4 --next_n 2 --avg_ctx 4096
"""

import argparse
import sys
from pathlib import Path

import cutlass
import torch

try:
    from tensorrt_llm._torch.cute_dsl_kernels.blackwell.paged_mqa_logits import (
        fp4_paged_mqa_logits as kernel_module,
    )
except (ModuleNotFoundError, ImportError):
    sys.path.insert(0, str(Path(__file__).parents[4] / "tensorrt_llm/_torch/cute_dsl_kernels"))
    from blackwell.paged_mqa_logits import fp4_paged_mqa_logits as kernel_module

_per_token_cast_to_fp4 = kernel_module._per_token_cast_to_fp4
_cast_back_from_fp4 = kernel_module._cast_back_from_fp4
_kv_cache_cast_to_fp4 = kernel_module._kv_cache_cast_to_fp4
_compute_schedule_metadata = kernel_module._compute_schedule_metadata
_ref_paged_mqa_logits = kernel_module._ref_paged_mqa_logits
fp4_paged_mqa_logits = kernel_module.fp4_paged_mqa_logits


def _calc_diff(x: torch.Tensor, y: torch.Tensor) -> float:
    x, y = x.double(), y.double()
    denom = (x * x + y * y).sum()
    if denom == 0:
        return 0.0
    return float(1 - 2 * (x * y).sum() / denom)


def run(
    batch_size: int,
    next_n: int,
    avg_ctx: int,
    num_heads: int = 64,
    head_dim: int = 128,
    phys_block_kv: int = 128,
    num_epi_subtiles: int = 1,
    epi_dtype=cutlass.Float32,
    output_dtype=cutlass.Float32,
    fix_length: bool = False,
    tol: float = 0.02,
    seed: int = 42,
    num_sms: int = 148,
) -> float:
    """Generate random inputs, run kernel, compare to reference, print result.

    Returns the cosine diff for caller use.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    device = "cuda"
    max_model_len = max(avg_ctx * 2, 2048)

    if fix_length:
        context_lens = torch.full((batch_size,), avg_ctx, dtype=torch.int32, device=device)
    else:
        lo = max(phys_block_kv, int(0.7 * avg_ctx))
        hi = int(1.3 * avg_ctx) + 1
        context_lens = torch.randint(lo, hi, (batch_size,), dtype=torch.int32, device=device).clamp(
            max=max_model_len
        )

    n_blk_per_seq = (context_lens + phys_block_kv - 1) // phys_block_kv
    total = int(n_blk_per_seq.sum().item())
    num_total_blocks = total + batch_size * 2
    max_blk = int(n_blk_per_seq.max().item())
    block_table = torch.zeros((batch_size, max_blk), dtype=torch.int32, device=device)
    pool = torch.randperm(num_total_blocks, device=device, dtype=torch.int32)
    off = 0
    for i, nb in enumerate(n_blk_per_seq.tolist()):
        block_table[i, :nb] = pool[off : off + nb]
        off += nb

    q = torch.randn(
        (batch_size, next_n, num_heads, head_dim),
        device=device,
        dtype=torch.bfloat16,
    )
    kv_cache = torch.randn(
        (num_total_blocks, phys_block_kv, 1, head_dim),
        device=device,
        dtype=torch.bfloat16,
    )
    weights = torch.randn((batch_size * next_n, num_heads), device=device, dtype=torch.float32)

    q_packed, sf_q_packed = _per_token_cast_to_fp4(q.view(-1, head_dim), gran_k=32)
    q_fp4 = q_packed.view(torch.uint8).view(batch_size, next_n, num_heads, head_dim // 2)
    sf_q = sf_q_packed.view(torch.int32).view(batch_size, next_n, num_heads)
    q_sim = (
        _cast_back_from_fp4(q_packed, sf_q_packed, gran_k=32)
        .view(batch_size, next_n, num_heads, head_dim)
        .to(torch.bfloat16)
    )

    kv_fused, kv_sim = _kv_cache_cast_to_fp4(kv_cache)

    # Reference uses original (B, next_n) layout regardless of how the kernel
    # is invoked below.
    ref = _ref_paged_mqa_logits(
        q_sim.float(),
        kv_sim.float(),
        weights,
        context_lens,
        block_table,
        max_model_len,
    )

    # next_n=4 isn't natively supported by the FP4 kernel (TMEM cap). Run it
    # as caller-side atom-split (kNextNAtom=2, kNumNextNAtoms=2; 2× HBM, same
    # as DeepGEMM's internal split): [B,4,H,D//2] → [2B,2,H,D//2], duplicate
    # block_table, split ctx_lens into (ctx-2, ctx) to preserve causal mask.
    # Output rows [4b..4b+3] map back to (batch=b, t∈0..3) by construction.
    if next_n == 4:
        exp_B = batch_size * 2
        kernel_q_fp4 = q_fp4.reshape(exp_B, 2, num_heads, head_dim // 2)
        kernel_sf_q = sf_q.reshape(exp_B, 2, num_heads)
        # weights [B*4, H] is layout-equivalent to [exp_B*2, H], unchanged.
        ctx_pair = torch.stack([context_lens - 2, context_lens], dim=1)  # [B,2]
        kernel_ctx_lens = ctx_pair.reshape(exp_B).contiguous()
        kernel_block_table = block_table.repeat_interleave(2, dim=0)
    else:
        kernel_q_fp4 = q_fp4
        kernel_sf_q = sf_q
        kernel_ctx_lens = context_lens
        kernel_block_table = block_table

    schedule_meta = _compute_schedule_metadata(kernel_ctx_lens.cpu(), 128, num_sms).to(device)

    logits = fp4_paged_mqa_logits(
        kernel_q_fp4,
        kernel_sf_q,
        kv_fused,
        weights,
        kernel_ctx_lens,
        kernel_block_table,
        schedule_meta,
        max_model_len,
        num_epi_subtiles=num_epi_subtiles,
        epi_dtype=epi_dtype,
        output_dtype=output_dtype,
        num_sms=num_sms,
    )

    positions = (
        torch.arange(max_model_len, device=device).unsqueeze(0).expand(batch_size * next_n, -1)
    )
    offsets = torch.arange(batch_size * next_n, device=device)
    limits = (context_lens[offsets // next_n] - next_n + offsets % next_n).unsqueeze(1)
    neginf_mask = ~(positions <= limits)

    logits_m = logits.float().masked_fill(neginf_mask, 0)
    ref_m = ref.float().masked_fill(neginf_mask, 0)
    finite = torch.isfinite(logits_m) & torch.isfinite(ref_m)
    diff = _calc_diff(logits_m.masked_fill(~finite, 0), ref_m.masked_fill(~finite, 0))
    ok = diff < tol
    epi_name = {cutlass.Float32: "fp32", cutlass.Float16: "fp16", cutlass.BFloat16: "bf16"}.get(
        epi_dtype, str(epi_dtype)
    )
    out_name = {cutlass.Float32: "fp32", cutlass.Float16: "fp16", cutlass.BFloat16: "bf16"}.get(
        output_dtype, str(output_dtype)
    )
    print(
        f"  B={batch_size} next_n={next_n} avg_ctx={avg_ctx} "
        f"pbk={phys_block_kv} epi={epi_name} out={out_name} "
        f"subtile={num_epi_subtiles} fix_len={fix_length}: "
        f"diff={diff:.3e} {'PASS' if ok else 'FAIL'}"
    )
    return diff


if __name__ == "__main__":
    DT = {
        "fp32": cutlass.Float32,
        "fp16": cutlass.Float16,
        "bf16": cutlass.BFloat16,
    }
    parser = argparse.ArgumentParser(
        description="Standalone FP4 paged MQA logits test (no trtllm)."
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--next_n", type=int, default=1)
    parser.add_argument("--avg_ctx", type=int, default=4096)
    parser.add_argument("--num_heads", type=int, default=64)
    parser.add_argument("--head_dim", type=int, default=128)
    parser.add_argument("--phys_block_kv", type=int, default=64)
    parser.add_argument("--num_epi_subtiles", type=int, default=1)
    parser.add_argument("--epi_dtype", choices=DT.keys(), default="fp32")
    parser.add_argument("--output_dtype", choices=DT.keys(), default="bf16")
    parser.add_argument(
        "--varlen", action="store_true", help="use variable context lengths (default: fixed)"
    )
    parser.add_argument(
        "--tol", type=float, default=0.02, help="cosine-diff PASS threshold (default 0.02)"
    )
    parser.add_argument("--num_sms", type=int, default=148)
    args = parser.parse_args()

    print("=== FP4 paged MQA logits standalone test ===")
    run(
        batch_size=args.batch_size,
        next_n=args.next_n,
        avg_ctx=args.avg_ctx,
        num_heads=args.num_heads,
        head_dim=args.head_dim,
        phys_block_kv=args.phys_block_kv,
        num_epi_subtiles=args.num_epi_subtiles,
        epi_dtype=DT[args.epi_dtype],
        output_dtype=DT[args.output_dtype],
        fix_length=not args.varlen,
        tol=args.tol,
        num_sms=args.num_sms,
    )
