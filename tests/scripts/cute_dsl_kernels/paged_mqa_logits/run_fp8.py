# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Standalone launch script for the CuTe DSL FP8 paged MQA logits kernel.

Generates random inputs, runs the kernel, and compares against a pure-torch
reference. Intended for local development / sanity checks; not wired into CI
(matching the convention of other scripts under ``tests/scripts/cute_dsl_kernels/``).

- Reference and data prep are inlined from
  tests/unittest/_torch/attention/sparse/test_cute_dsl_fp8_paged_mqa_logits.py.
- Schedule metadata is computed in pure Python (mirrors DeepGEMM's
  PagedMQALogitsScheduler), avoiding the deep_gemm C++ binding. Same algorithm
  as run_fp4.py — both kernels use compute_block_kv=128 + NUM_MATH_WG=2 →
  SPLIT_KV=256.
- Compile + dispatch follows tensorrt_llm/_torch/custom_ops/cute_dsl_custom_ops.py
  (CuteDSLPagedMQALogitsRunner — fake tensors + TVM FFI; runtime call passes
  raw torch tensors).

Example:
    python run_fp8.py --batch_size 4 --next_n 2 --avg_ctx 4096
"""

import argparse
import sys
from pathlib import Path

import cutlass
import cutlass.cute as cute
import torch

try:
    from tensorrt_llm._torch.cute_dsl_kernels.blackwell.paged_mqa_logits import FP8MQALogitsKernel
except (ModuleNotFoundError, ImportError):
    sys.path.insert(0, str(Path(__file__).parents[4] / "tensorrt_llm/_torch/cute_dsl_kernels"))
    from blackwell.paged_mqa_logits import FP8MQALogitsKernel


# ---- Constants --------------------------------------------------------------


_CUTLASS_TO_TORCH = {
    cutlass.Float32: torch.float32,
    cutlass.Float16: torch.float16,
    cutlass.BFloat16: torch.bfloat16,
}

_TORCH_TO_CUTLASS = {
    torch.float32: cutlass.Float32,
    torch.float16: cutlass.Float16,
    torch.bfloat16: cutlass.BFloat16,
}

# Element-wise tolerance keyed by output_dtype — mirrors the unit test
# tests/unittest/_torch/attention/sparse/test_cute_dsl_fp8_paged_mqa_logits.py
# which sets atol/rtol purely from output_dtype (epi_dtype = acc_dtype =
# output_dtype in that test).
_ELEM_TOL = {
    torch.float32: (5e-5, 1e-5),
    torch.float16: (1e-3, 1e-3),
}


# ---- Plain helpers ----------------------------------------------------------


def _ceil_div(x, y):
    return (x + y - 1) // y


def _ceil_to_ue8m0(x: torch.Tensor) -> torch.Tensor:
    return torch.pow(2.0, torch.ceil(torch.log2(x.abs())))


def _calc_diff(x: torch.Tensor, y: torch.Tensor) -> float:
    x, y = x.double(), y.double()
    denom = (x * x + y * y).sum()
    if denom == 0:
        return 0.0
    return float(1 - 2 * (x * y).sum() / denom)


def _compute_schedule_metadata(
    context_lens_cpu: torch.Tensor, block_kv: int, num_ctas: int
) -> torch.Tensor:
    """Return [num_ctas+1, 2] int32. Each row = (q_idx, kv_idx_half) CTA boundary.

    Mirrors DeepGEMM PagedMQALogitsScheduler. Pass block_kv=128 for compatibility
    with the kernel's compute_block_kv=128 + NUM_MATH_WG=2 (final SPLIT_KV=256).
    """
    batch_size = context_lens_cpu.shape[0]
    splits_per_seq = []
    total_splits = 0
    for b in range(batch_size):
        ctx = int(context_lens_cpu[b].item())
        num_kv = _ceil_div(ctx, block_kv)
        ns = _ceil_div(num_kv, 2)
        splits_per_seq.append(ns)
        total_splits += ns

    q_div = total_splits // num_ctas
    r_mod = total_splits % num_ctas
    schedule = torch.zeros((num_ctas + 1, 2), dtype=torch.int32)

    cum = 0
    seq_idx = 0
    seq_offset = 0
    for i in range(num_ctas + 1):
        target = i * q_div + min(i, r_mod)
        while seq_idx < batch_size and cum + (splits_per_seq[seq_idx] - seq_offset) <= target:
            cum += splits_per_seq[seq_idx] - seq_offset
            seq_idx += 1
            seq_offset = 0
            if seq_idx >= batch_size:
                break
        if seq_idx >= batch_size:
            schedule[i] = torch.tensor([batch_size, 0], dtype=torch.int32)
        else:
            local = target - cum + seq_offset
            schedule[i] = torch.tensor([seq_idx, local], dtype=torch.int32)
    return schedule


def _make_fused_kv(
    kv_fp8: torch.Tensor, kv_scales: torch.Tensor, block_kv: int, head_dim: int
) -> torch.Tensor:
    """Pack KV into per-block packed-by-type layout matching DeepGEMM/DSL kernel.

    Per block: [all FP8 bytes (block_kv * head_dim)] [all scale bytes (block_kv * 4)].
    Returns shape [num_blocks, block_kv, 1, head_dim + 4] uint8.
    """
    num_phys_blocks = kv_fp8.shape[0]
    per_token_size = head_dim + 4
    block_bytes = block_kv * per_token_size
    scale_offset = block_kv * head_dim

    fused = torch.zeros(num_phys_blocks, block_bytes, dtype=torch.uint8, device=kv_fp8.device)
    for blk in range(num_phys_blocks):
        fused[blk, :scale_offset] = kv_fp8[blk].view(torch.uint8).reshape(-1)
        fused[blk, scale_offset:] = (
            kv_scales[blk].float().contiguous().view(torch.uint8).reshape(-1)
        )
    return fused.view(num_phys_blocks, block_kv, 1, per_token_size)


# ---- Compile + dispatch (mirrors cute_dsl_custom_ops.py compile pattern) ----


_compiled_cache = {}


def _compile_fp8_kernel(
    compute_block_kv: int,
    phys_block_kv: int,
    num_heads: int,
    head_dim: int,
    next_n: int,
    num_sms: int,
    num_epi_subtiles: int,
    epi_dtype: torch.dtype,
    acc_dtype: torch.dtype,
    output_dtype: torch.dtype,
):
    """Compile FP8 kernel with fake tensors + TVM FFI; cached by static config.

    Mirrors `CuteDSLPagedMQALogitsRunner._compile` in cute_dsl_custom_ops.py.
    """
    key = (
        compute_block_kv,
        phys_block_kv,
        num_heads,
        head_dim,
        next_n,
        num_sms,
        num_epi_subtiles,
        epi_dtype,
        acc_dtype,
        output_dtype,
    )
    if key in _compiled_cache:
        return _compiled_cache[key]

    N = next_n * num_heads
    block_bytes = phys_block_kv * (head_dim + 4)

    sym_npb = cute.sym_int()
    sym_B = cute.sym_int()
    max_ctx = cute.sym_int()
    max_blocks = cute.sym_int()
    num_ctas = cute.sym_int()

    kv_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8, (sym_npb, block_bytes), stride_order=(1, 0)
    )
    q_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8, (N, head_dim, sym_B), stride_order=(1, 0, 2)
    )
    # The kernel always materialises weights as fp16 internally when epi=fp16
    # (it does .half() on the host wrapper); align the fake-tensor dtype
    # accordingly so the cache key matches the runtime call.
    w_dtype = cutlass.Float16 if epi_dtype == torch.float16 else _TORCH_TO_CUTLASS[epi_dtype]
    w_fake = cute.runtime.make_fake_compact_tensor(w_dtype, (N, sym_B), stride_order=(0, 1))
    logits_fake = cute.runtime.make_fake_tensor(
        _TORCH_TO_CUTLASS[output_dtype],
        (cute.sym_int(), max_ctx),
        stride=(cute.sym_int64(), 1),
    )
    bt_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (sym_B, max_blocks), stride_order=(1, 0)
    )
    cl_fake = cute.runtime.make_fake_compact_tensor(cutlass.Int32, (sym_B,), stride_order=(0,))
    sm_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (num_ctas, 2), stride_order=(1, 0)
    )
    fake_stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)

    kernel = FP8MQALogitsKernel(
        block_kv=compute_block_kv,
        phys_block_kv=phys_block_kv,
        num_heads=num_heads,
        head_dim=head_dim,
        next_n=next_n,
        num_sms=num_sms,
        num_epi_subtiles=num_epi_subtiles,
        epi_dtype=_TORCH_TO_CUTLASS[epi_dtype],
        acc_dtype=_TORCH_TO_CUTLASS[acc_dtype],
        output_dtype=_TORCH_TO_CUTLASS[output_dtype],
    )
    compiled = cute.compile(
        kernel,
        kv_fake,
        q_fake,
        w_fake,
        logits_fake,
        bt_fake,
        cl_fake,
        sm_fake,
        cutlass.Int32(1),
        cutlass.Int32(1),
        fake_stream,
        options="--enable-tvm-ffi",
    )
    _compiled_cache[key] = compiled
    print(
        f"[compile] block_kv={compute_block_kv} phys_block_kv={phys_block_kv} "
        f"H={num_heads} D={head_dim} next_n={next_n} "
        f"subtile={num_epi_subtiles} epi={epi_dtype} acc={acc_dtype} out={output_dtype}"
    )
    return compiled


def fp8_paged_mqa_logits(
    q: torch.Tensor,  # [B, next_n, H, D] float8_e4m3fn
    kv_fused: torch.Tensor,  # [num_blocks, phys_block_kv, 1, D + 4] uint8
    weights: torch.Tensor,  # [B*next_n, H] float32
    context_lens: torch.Tensor,  # [B] int32 (cuda)
    block_table: torch.Tensor,  # [B, max_blocks] int32 (cuda)
    schedule_meta: torch.Tensor,  # [num_sms+1, 2] int32 (cuda)
    max_context_len: int,
    num_epi_subtiles: int = 1,
    epi_dtype: torch.dtype = torch.float32,
    acc_dtype: torch.dtype = torch.float32,
    output_dtype: torch.dtype = torch.float32,
    num_sms: int = 148,
) -> torch.Tensor:
    """Standalone wrapper around FP8MQALogitsKernel; no trtllm dependency.

    Mirrors the production `CuteDSLPagedMQALogitsRunner.forward` in
    cute_dsl_custom_ops.py.
    """
    B, next_n, H, D = q.shape
    N = next_n * H
    phys_block_kv = kv_fused.shape[1]
    compute_block_kv = 128
    num_phys_blocks = kv_fused.shape[0]

    # Reshape (no .contiguous() — strides must stay B-independent to keep
    # the fake-tensor compile cache hot).
    q_3d = q.reshape(B, N, D).permute(1, 2, 0)
    if epi_dtype == torch.float16:
        # TODO: move type conversion to weight loading (matches production
        # wrapper note in cute_dsl_custom_ops.py).
        w_2d = weights.reshape(B, N).half().t()
    else:
        w_2d = weights.reshape(B, N).t()
    kv_flat = kv_fused.reshape(num_phys_blocks, -1)

    # Output alignment to SPLIT_KV = compute_block_kv * NUM_MATH_WG (the kernel
    # may store unconditionally into the trailing pad).
    SPLIT_KV = compute_block_kv * 2
    aligned_max_ctx = ((max_context_len + SPLIT_KV - 1) // SPLIT_KV) * SPLIT_KV
    logits = torch.empty((B * next_n, aligned_max_ctx), device=q.device, dtype=output_dtype)
    logits = logits[:, :max_context_len]

    compiled = _compile_fp8_kernel(
        compute_block_kv,
        phys_block_kv,
        H,
        D,
        next_n,
        num_sms,
        num_epi_subtiles,
        epi_dtype,
        acc_dtype,
        output_dtype,
    )

    # FP8 q needs uint8 view to match compile-time dtype.
    q_for_ffi = (
        q_3d.view(torch.uint8) if q_3d.dtype in (torch.float8_e4m3fn, torch.float8_e5m2) else q_3d
    )
    compiled(
        kv_flat,
        q_for_ffi,
        w_2d,
        logits,
        block_table,
        context_lens,
        schedule_meta,
        num_phys_blocks,
        B,
    )
    return logits


# ---- Reference + driver -----------------------------------------------------


def _ref_fp8_paged_mqa_logits(
    q_fp8: torch.Tensor,
    kv_fp8: torch.Tensor,
    kv_scales: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_table: torch.Tensor,
    max_model_len: int,
    block_kv: int,
    epi_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Pure-torch reference (verbatim from the FP8 unit test).

    GEMM stays in fp32; weighted sum + scale apply use ``epi_dtype``.
    """
    B, next_n, H, D = q_fp8.shape
    device = q_fp8.device
    logits = torch.full((B * next_n, max_model_len), float("-inf"), device=device, dtype=epi_dtype)
    q_f32 = q_fp8.float()

    for b in range(B):
        ctx_len = context_lens[b].item()
        q_positions = torch.arange(ctx_len - next_n, ctx_len, device=device)
        w = weights[b * next_n : (b + 1) * next_n, :].to(epi_dtype)

        for blk_idx in range((ctx_len + block_kv - 1) // block_kv):
            phys_blk = block_table[b, blk_idx].item()
            k_f32 = kv_fp8[phys_blk].float()
            scales = kv_scales[phys_blk].to(epi_dtype)

            k_positions = torch.arange(blk_idx * block_kv, (blk_idx + 1) * block_kv, device=device)
            mask = (k_positions[None, :] < ctx_len) & (k_positions[None, :] <= q_positions[:, None])

            qk = torch.matmul(q_f32[b].permute(1, 0, 2), k_f32.T)  # [H, next_n, block_kv]
            qk = torch.where(mask[None, :, :], qk, torch.zeros(1, device=device))
            qk = torch.relu(qk)

            qk = qk.to(epi_dtype)
            weighted = (w.T[:, :, None] * qk).sum(dim=0)  # [next_n, block_kv]
            weighted = weighted * scales[None, :]

            start_pos = blk_idx * block_kv
            end_pos = start_pos + block_kv
            logits[b * next_n : (b + 1) * next_n, start_pos:end_pos] = torch.where(
                mask, weighted, torch.tensor(float("-inf"), device=device, dtype=epi_dtype)
            )

    return logits


def run(
    batch_size: int,
    next_n: int,
    avg_ctx: int,
    num_heads: int = 64,
    head_dim: int = 128,
    phys_block_kv: int = 128,
    num_epi_subtiles: int = 1,
    epi_dtype: torch.dtype = torch.float32,
    acc_dtype: torch.dtype = torch.float32,
    output_dtype: torch.dtype = torch.float32,
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

    # FP8 inputs (matches unit test data prep).
    q_bf16 = torch.randn(batch_size, next_n, num_heads, head_dim, device=device)
    q_fp8 = q_bf16.to(torch.float8_e4m3fn)

    kv_bf16 = torch.randn(num_total_blocks, phys_block_kv, head_dim, device=device)
    kv_amax = kv_bf16.abs().float().amax(dim=-1, keepdim=True).clamp(1e-4)
    kv_scale = _ceil_to_ue8m0(kv_amax / 448.0).squeeze(-1)
    kv_fp8 = (kv_bf16 / kv_scale.unsqueeze(-1)).to(torch.float8_e4m3fn)

    weights = torch.randn(batch_size * next_n, num_heads, device=device, dtype=torch.float32)

    kv_fused = _make_fused_kv(kv_fp8, kv_scale, phys_block_kv, head_dim)

    # Reference. epi_dtype carries the weighted-sum + scale-apply dtype; the
    # GEMM itself stays in fp32 inside the helper.
    ref = _ref_fp8_paged_mqa_logits(
        q_fp8,
        kv_fp8,
        kv_scale,
        weights,
        context_lens,
        block_table,
        max_model_len,
        phys_block_kv,
        epi_dtype=output_dtype,
    )

    # Schedule: pure-Python equivalent of deep_gemm.get_paged_mqa_logits_metadata
    # at SPLIT_KV=256. Same call as run_fp4.py.
    schedule_meta = _compute_schedule_metadata(context_lens.cpu(), 128, num_sms).to(device)

    logits = fp8_paged_mqa_logits(
        q_fp8,
        kv_fused,
        weights,
        context_lens,
        block_table,
        schedule_meta,
        max_model_len,
        num_epi_subtiles=num_epi_subtiles,
        epi_dtype=epi_dtype,
        acc_dtype=acc_dtype,
        output_dtype=output_dtype,
        num_sms=num_sms,
    )

    # Mask out-of-context positions before comparison.
    positions = (
        torch.arange(max_model_len, device=device).unsqueeze(0).expand(batch_size * next_n, -1)
    )
    offsets = torch.arange(batch_size * next_n, device=device)
    limits = (context_lens[offsets // next_n] - next_n + offsets % next_n).unsqueeze(1)
    neginf_mask = ~(positions <= limits)

    logits_m = logits.float().masked_fill(neginf_mask, 0)
    ref_m = ref.float().masked_fill(neginf_mask, 0)
    finite = torch.isfinite(logits_m) & torch.isfinite(ref_m)
    logits_clean = logits_m.masked_fill(~finite, 0)
    ref_clean = ref_m.masked_fill(~finite, 0)

    # Element-wise check (mirrors torch.testing.assert_close semantics from
    # the unit test): |a-b| <= atol + rtol*|b| pointwise on in-context, finite
    # entries.
    atol, rtol = _ELEM_TOL[output_dtype]
    valid = (~neginf_mask) & finite
    elem_abs = (logits_clean - ref_clean).abs()[valid]
    ref_abs = ref_clean.abs()[valid]
    if elem_abs.numel() > 0:
        max_abs = elem_abs.max().item()
        mean_abs = elem_abs.mean().item()
        elem_ok = bool((elem_abs <= atol + rtol * ref_abs).all().item())
    else:
        max_abs = 0.0
        mean_abs = 0.0
        elem_ok = True

    # Cosine diff: supplementary global similarity metric.
    diff = _calc_diff(logits_clean, ref_clean)
    diff_ok = diff < tol

    ok = elem_ok and diff_ok
    dt_name = {torch.float32: "fp32", torch.float16: "fp16", torch.bfloat16: "bf16"}.get(
        output_dtype, str(output_dtype)
    )
    print(
        f"  B={batch_size} next_n={next_n} avg_ctx={avg_ctx} "
        f"pbk={phys_block_kv} dtype={dt_name} "
        f"subtile={num_epi_subtiles} fix_len={fix_length}: "
        f"max_abs={max_abs:.3e} mean_abs={mean_abs:.3e} "
        f"(atol={atol:.0e} rtol={rtol:.0e}) "
        f"diff={diff:.3e} {'PASS' if ok else 'FAIL'}"
    )
    return diff


if __name__ == "__main__":
    DT = {
        "fp32": torch.float32,
        "fp16": torch.float16,
    }
    parser = argparse.ArgumentParser(
        description="Standalone FP8 paged MQA logits test (no trtllm)."
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--next_n", type=int, default=1)
    parser.add_argument("--avg_ctx", type=int, default=4096)
    parser.add_argument("--num_heads", type=int, default=64)
    parser.add_argument("--head_dim", type=int, default=128)
    parser.add_argument("--phys_block_kv", type=int, default=64)
    parser.add_argument("--num_epi_subtiles", type=int, default=1)
    parser.add_argument(
        "--dtype",
        choices=DT.keys(),
        default="fp32",
        help="epi/acc/output dtype (the FP8 unit test sets these all the same)",
    )
    parser.add_argument(
        "--varlen", action="store_true", help="use variable context lengths (default: fixed)"
    )
    parser.add_argument(
        "--tol", type=float, default=0.02, help="cosine-diff PASS threshold (default 0.02)"
    )
    parser.add_argument("--num_sms", type=int, default=148)
    args = parser.parse_args()

    print("=== FP8 paged MQA logits standalone test ===")
    run(
        batch_size=args.batch_size,
        next_n=args.next_n,
        avg_ctx=args.avg_ctx,
        num_heads=args.num_heads,
        head_dim=args.head_dim,
        phys_block_kv=args.phys_block_kv,
        num_epi_subtiles=args.num_epi_subtiles,
        epi_dtype=DT[args.dtype],
        acc_dtype=DT[args.dtype],
        output_dtype=DT[args.dtype],
        fix_length=not args.varlen,
        tol=args.tol,
        num_sms=args.num_sms,
    )
