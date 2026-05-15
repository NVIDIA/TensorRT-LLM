# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Standalone launch script for the CuTe DSL FP4 paged MQA logits kernel.

Generates random inputs, runs the kernel, and compares against a reference
implementation. Intended for local development / sanity checks; not wired into
CI (matching the convention of other scripts under ``tests/scripts/cute_dsl_kernels/``).

- Helpers (FP4 quant, KV cast, ref) are inlined from
  tests/unittest/_torch/attention/sparse/test_cute_dsl_fp4_paged_mqa_logits.py.
- Schedule metadata is computed in pure Python (mirrors DeepGEMM's
  PagedMQALogitsScheduler), avoiding the deep_gemm C++ binding.
- Compile + dispatch follows tensorrt_llm/_torch/custom_ops/cute_dsl_custom_ops.py
  (fake tensors + TVM FFI; runtime call passes raw torch tensors).

Example:
    python run_fp4.py --batch_size 4 --next_n 2 --avg_ctx 4096
"""

import argparse
import functools
import sys
from pathlib import Path

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass.utils.smem_allocator import SmemAllocator

try:
    from tensorrt_llm._torch.cute_dsl_kernels.blackwell.paged_mqa_logits import FP4MQALogitsKernel
except ImportError:
    sys.path.insert(0, str(Path(__file__).parents[4] / "tensorrt_llm/_torch/cute_dsl_kernels"))
    from blackwell.paged_mqa_logits import FP4MQALogitsKernel


# ---- FP4 quant helpers (verbatim from DeepGEMM/deep_gemm/utils/math.py) -----


def _ceil_div(x, y):
    return (x + y - 1) // y


def _align(x, y):
    return _ceil_div(x, y) * y


def _ceil_to_ue8m0(x: torch.Tensor) -> torch.Tensor:
    bits = x.abs().float().view(torch.int)
    exp = ((bits >> 23) & 0xFF) + (bits & 0x7FFFFF).bool().int()
    return (exp.clamp(1, 254) << 23).view(torch.float)


def _pack_ue8m0_to_int(x: torch.Tensor) -> torch.Tensor:
    assert x.dtype == torch.float and x.size(-1) % 4 == 0
    return (x.view(torch.int) >> 23).to(torch.uint8).view(torch.int)


def _unpack_ue8m0_from_int(packed: torch.Tensor) -> torch.Tensor:
    return (packed.view(torch.uint8).to(torch.int) << 23).view(torch.float)


def _quantize_to_fp4_e2m1(x: torch.Tensor) -> torch.Tensor:
    ax = x.abs().clamp_max(6.0)
    boundaries = torch.tensor(
        [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0],
        device=x.device,
        dtype=ax.dtype,
    )
    idx = torch.bucketize(ax, boundaries)
    code = idx.to(torch.uint8)
    sign = (x < 0) & (idx != 0)
    code = code | (sign.to(torch.uint8) << 3)
    return code.view(torch.int8)


def _dequantize_from_fp4_e2m1(x: torch.Tensor) -> torch.Tensor:
    fp4_values = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
        device=x.device,
        dtype=torch.float,
    )
    sign, value_idx = (x & 0x08) != 0, (x & 0x07).to(torch.int)
    value = fp4_values[value_idx]
    return torch.where(sign & (value_idx != 0), -value, value)


def _per_token_cast_to_fp4(x: torch.Tensor, gran_k: int = 32):
    m, n = x.shape
    padded_n = _align(n, gran_k)
    x_padded = torch.zeros((m, padded_n), dtype=x.dtype, device=x.device)
    x_padded[:, :n] = x
    x_view = x_padded.view(m, -1, gran_k)
    x_amax = x_view.abs().float().amax(dim=2).clamp_min(1e-4)
    sf = _ceil_to_ue8m0(x_amax / 6.0)
    x_scaled = x_view * (1.0 / sf.unsqueeze(2))
    codes = _quantize_to_fp4_e2m1(x_scaled).view(m, padded_n)
    codes2 = codes.view(m, padded_n // 2, 2)
    packed = (codes2[:, :, 0] & 0x0F) | ((codes2[:, :, 1] & 0x0F) << 4)
    return packed[:, : n // 2].contiguous(), _pack_ue8m0_to_int(sf)


def _cast_back_from_fp4(packed: torch.Tensor, sf: torch.Tensor, gran_k: int = 32):
    m, n2 = packed.shape
    n = n2 * 2
    sf = _unpack_ue8m0_from_int(sf)
    unpacked = torch.zeros((m, n), dtype=torch.int8, device=packed.device)
    unpacked[:, ::2] = packed & 0x0F
    unpacked[:, 1::2] = (packed >> 4) & 0x0F
    x = _dequantize_from_fp4_e2m1(unpacked)
    return x * sf[:, torch.arange(n, device=packed.device) // gran_k]


def _kv_cache_cast_to_fp4(x: torch.Tensor):
    num_blocks, block_size, num_heads, head_dim = x.shape
    assert num_heads == 1 and head_dim == 128
    x_scaled, sf = _per_token_cast_to_fp4(x.view(-1, head_dim), gran_k=32)
    x_back = _cast_back_from_fp4(x_scaled, sf, gran_k=32).view(num_blocks, block_size, 1, head_dim)
    x_fp4 = torch.empty(
        (num_blocks, block_size * (head_dim // 2 + 4)),
        device=x.device,
        dtype=torch.uint8,
    )
    x_fp4[:, : block_size * head_dim // 2] = x_scaled.view(
        num_blocks, block_size * head_dim // 2
    ).view(torch.uint8)
    x_fp4[:, block_size * head_dim // 2 :] = sf.view(num_blocks, block_size).view(torch.uint8)
    return (
        x_fp4.view(num_blocks, block_size, num_heads, head_dim // 2 + 4),
        x_back.to(x.dtype),
    )


# ---- Pure-Python schedule metadata on CPU
def _compute_schedule_metadata(
    context_lens_cpu: torch.Tensor, block_kv: int, num_ctas: int
) -> torch.Tensor:
    """Return [num_ctas+1, 2] int32. Each row = (q_idx, kv_idx_half) CTA boundary.

    Mirrors DeepGEMM PagedMQALogitsScheduler — kernel multiplies col 1 by
    NUM_MATH_WG=2 to get block-granularity kv_idx.
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
    for i in range(num_ctas + 1):
        target = i * q_div + min(i, r_mod)
        while seq_idx < batch_size and cum + splits_per_seq[seq_idx] <= target:
            cum += splits_per_seq[seq_idx]
            seq_idx += 1
            if seq_idx >= batch_size:
                break
        if seq_idx >= batch_size:
            schedule[i] = torch.tensor([batch_size, 0], dtype=torch.int32)
        else:
            local = target - cum
            schedule[i] = torch.tensor([seq_idx, local], dtype=torch.int32)
    return schedule


# Single-CTA single-warp kernel: register-array num_segs → warp inclusive scan
# with carry → SMEM prefix_sum → per-SM linear scan emitting (q_idx, kv_split).
class PagedMQALogitsMetadataKernel:
    """Compile-time params: aligned_batch_size (multiple of 32), split_kv, num_sms.

    Runtime: context_lens [B] int32 (cuda), schedule_meta [num_sms+1, 2] int32 (cuda),
    batch_size (Int32).
    """

    def __init__(self, aligned_batch_size: int, split_kv: int, num_sms: int):
        assert aligned_batch_size > 0 and aligned_batch_size % 32 == 0
        self.aligned_batch_size = aligned_batch_size
        self.split_kv = split_kv
        self.num_sms = num_sms

    @cute.jit
    def __call__(
        self,
        context_lens: cute.Tensor,
        schedule_meta: cute.Tensor,
        batch_size: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(context_lens, schedule_meta, batch_size).launch(
            grid=(1, 1, 1), block=(32, 1, 1), stream=stream
        )

    @cute.kernel
    def kernel(
        self,
        context_lens: cute.Tensor,
        schedule_meta: cute.Tensor,
        batch_size: cutlass.Int32,
    ):
        kAligned = cutlass.const_expr(self.aligned_batch_size)
        SPLIT_KV = cutlass.const_expr(self.split_kv)
        kNumSMs = cutlass.const_expr(self.num_sms)
        kNumChunks = cutlass.const_expr(kAligned // 32)
        # Cover sm_idx ∈ [0, kNumSMs] (inclusive) in 32-lane strips.
        kMaxSmChunks = cutlass.const_expr((kNumSMs + 32) // 32)

        lane_idx = cute.arch.lane_idx()

        smem = SmemAllocator()
        prefix_sum = smem.allocate_tensor(
            element_type=cutlass.Int32,
            layout=cute.make_ordered_layout((kAligned,), order=(0,)),
            byte_alignment=128,
        )

        # Phase 1: per-lane register array of ceil_div(ctx, SPLIT_KV).
        # Out-of-range lanes contribute 0 (matches CUDA's q_idx<batch_size guard).
        num_segs = [cutlass.Int32(0)] * kNumChunks
        for k in cutlass.range_constexpr(kNumChunks):
            q_idx = cutlass.Int32(k * 32) + lane_idx
            ctx_len = cutlass.Int32(0)
            if q_idx < batch_size:
                ctx_len = context_lens[q_idx]
            num_segs[k] = (ctx_len + (SPLIT_KV - 1)) // SPLIT_KV

        # Phase 2: warp-level inclusive scan, with carry across chunks, → SMEM.
        sum_carry = cutlass.Int32(0)
        for k in cutlass.range_constexpr(kNumChunks):
            x = num_segs[k]
            for i in cutlass.range_constexpr(5):  # log2(32) = 5
                offset = 1 << i
                y = cute.arch.shuffle_sync_up(x, offset, mask=0xFFFFFFFF, mask_and_clamp=0)
                if lane_idx >= offset:
                    x = x + y
            x = x + sum_carry
            prefix_sum[k * 32 + lane_idx] = x
            # Broadcast lane-31's value to all lanes for the next chunk.
            sum_carry = cute.arch.shuffle_sync(x, 31)

        # Phase 3: distribute `total` segments across kNumSMs CTAs.
        # Each lane handles sm_idx = lane, lane+32, ... up to kNumSMs inclusive.
        total = sum_carry
        q_div = total // kNumSMs
        r_mod = total % kNumSMs

        for s in cutlass.range_constexpr(kMaxSmChunks):
            sm_idx_local = cutlass.Int32(s * 32) + lane_idx
            if sm_idx_local <= kNumSMs:
                # seg_starts = sm * q + min(sm, r). Pre-declare so it survives
                # the dynamic if/else (DSL rule: no first-assignment inside
                # dynamic control flow).
                seg_starts = sm_idx_local * q_div
                if sm_idx_local <= r_mod:
                    seg_starts = seg_starts + sm_idx_local
                else:
                    seg_starts = seg_starts + r_mod

                # Linear scan: q_idx_out = number of j ∈ [0, batch_size) with
                # prefix_sum[j] <= seg_starts. prefix_sum is non-decreasing, so
                # the predicate is monotone — we can replace the CUDA `while`
                # with a constexpr-bounded for-scan.
                q_idx_out = cutlass.Int32(0)
                for j in cutlass.range_constexpr(kAligned):
                    in_range = cutlass.Int32(j) < batch_size
                    advance = cutlass.Boolean(False)
                    if in_range:
                        if prefix_sum[j] <= seg_starts:
                            advance = cutlass.Boolean(True)
                    if advance:
                        q_idx_out = cutlass.Int32(j + 1)

                kv_split_idx = seg_starts
                if q_idx_out > 0:
                    kv_split_idx = seg_starts - prefix_sum[q_idx_out - 1]

                schedule_meta[sm_idx_local, 0] = q_idx_out
                schedule_meta[sm_idx_local, 1] = kv_split_idx


@functools.cache
def compile_paged_mqa_logits_metadata_kernel(aligned_b: int, split_kv: int, num_ctas: int):
    """Compile (and cache) the metadata kernel using fake tensors + TVM FFI.

    Mirrors `_compile_fp4_kernel` in this file: explicit fake-tensor signature
    so runtime calls can pass raw torch tensors directly.
    """
    sym_B = cute.sym_int()
    cl_fake = cute.runtime.make_fake_compact_tensor(cutlass.Int32, (sym_B,), stride_order=(0,))
    sm_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (num_ctas + 1, 2), stride_order=(1, 0)
    )
    fake_stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)

    kern = PagedMQALogitsMetadataKernel(aligned_b, split_kv, num_ctas)
    compiled = cute.compile(
        kern,
        cl_fake,
        sm_fake,
        cutlass.Int32(1),
        fake_stream,
        options="--enable-tvm-ffi",
    )
    print(f"[compile] schedule_meta aligned_b={aligned_b} split_kv={split_kv} num_sms={num_ctas}")
    return compiled


def get_paged_mqa_logits_metadata_cute_dsl(
    context_lens: torch.Tensor,
    compute_block_kv_per_wg: int,
    num_ctas: int,
    num_math_wg: int = 2,
) -> torch.Tensor:
    """CuTe DSL replacement for `_compute_schedule_metadata` (kernel runs on GPU).

    Equivalent contract to the pure-Python version, but expects `context_lens`
    on CUDA (int32, 1D) and returns the output on the same device. SPLIT_KV is
    compute_block_kv_per_wg * num_math_wg.
    """
    assert context_lens.is_cuda and context_lens.dtype == torch.int32 and context_lens.dim() == 1
    batch_size = int(context_lens.shape[0])
    aligned_b = max(((batch_size + 31) // 32) * 32, 32)
    # one cta handles tile_m is block_kv
    split_kv = compute_block_kv_per_wg * num_math_wg

    schedule = torch.empty((num_ctas + 1, 2), dtype=torch.int32, device=context_lens.device)
    compiled = compile_paged_mqa_logits_metadata_kernel(aligned_b, split_kv, num_ctas)
    compiled(context_lens, schedule, batch_size)
    return schedule


# ---- Compile + dispatch (mirrors cute_dsl_custom_ops.py compile pattern) ----


_compiled_cache = {}

_CUTLASS_TO_TORCH = {
    cutlass.Float32: torch.float32,
    cutlass.Float16: torch.float16,
    cutlass.BFloat16: torch.bfloat16,
}

# Element-wise tolerance keyed by (epi_dtype, output_dtype) — mirrors the
# unit test's ELEM_TOL table at
# tests/unittest/_torch/attention/sparse/test_cute_dsl_fp4_paged_mqa_logits.py.
_ELEM_TOL = {
    (torch.float32, torch.float32): (5e-5, 1e-5),
    (torch.bfloat16, torch.bfloat16): (1e-2, 1e-2),
    (torch.float16, torch.float16): (1e-3, 1e-3),
    (torch.float32, torch.bfloat16): (1e-2, 1e-2),
    (torch.float32, torch.float16): (1e-3, 1e-3),
}


def _compile_fp4_kernel(
    compute_block_kv: int,
    phys_block_kv: int,
    num_heads: int,
    head_dim: int,
    next_n: int,
    num_sms: int,
    num_epi_subtiles: int,
    epi_dtype,
    output_dtype,
):
    """Compile FP4 kernel with fake tensors + TVM FFI; cached by static config."""
    key = (
        compute_block_kv,
        phys_block_kv,
        num_heads,
        head_dim,
        next_n,
        num_sms,
        num_epi_subtiles,
        epi_dtype,
        output_dtype,
    )
    if key in _compiled_cache:
        return _compiled_cache[key]

    N = next_n * num_heads
    half_D = head_dim // 2
    block_bytes = phys_block_kv * (half_D + 4)

    sym_npb = cute.sym_int()
    sym_B = cute.sym_int()
    max_ctx = cute.sym_int()
    max_blocks = cute.sym_int()
    num_ctas = cute.sym_int()

    kv_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8, (sym_npb, block_bytes), stride_order=(1, 0)
    )
    q_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8, (N, half_D, sym_B), stride_order=(1, 0, 2)
    )
    sf_q_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (N, sym_B), stride_order=(0, 1)
    )
    w_fake = cute.runtime.make_fake_compact_tensor(epi_dtype, (N, sym_B), stride_order=(0, 1))
    logits_fake = cute.runtime.make_fake_tensor(
        output_dtype, (cute.sym_int(), max_ctx), stride=(cute.sym_int64(), 1)
    )
    bt_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (sym_B, max_blocks), stride_order=(1, 0)
    )
    cl_fake = cute.runtime.make_fake_compact_tensor(cutlass.Int32, (sym_B,), stride_order=(0,))
    sm_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (num_ctas, 2), stride_order=(1, 0)
    )
    fake_stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)

    kernel = FP4MQALogitsKernel(
        block_kv=compute_block_kv,
        phys_block_kv=phys_block_kv,
        num_heads=num_heads,
        head_dim=head_dim,
        next_n=next_n,
        num_sms=num_sms,
        num_epi_subtiles=num_epi_subtiles,
        epi_dtype=epi_dtype,
        output_dtype=output_dtype,
    )
    compiled = cute.compile(
        kernel,
        kv_fake,
        q_fake,
        sf_q_fake,
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
        f"subtile={num_epi_subtiles} epi={epi_dtype} out={output_dtype}"
    )
    return compiled


def fp4_paged_mqa_logits(
    q: torch.Tensor,  # [B, next_n, H, D//2] uint8 (FP4 packed)
    sf_q: torch.Tensor,  # [B, next_n, H] int32 (4 UE8M0 packed)
    kv_fused: torch.Tensor,  # [num_blocks, phys_block_kv, 1, D//2 + 4] uint8
    weights: torch.Tensor,  # [B*next_n, H] float32
    context_lens: torch.Tensor,  # [B] int32 (cuda)
    block_table: torch.Tensor,  # [B, max_blocks] int32 (cuda)
    schedule_meta: torch.Tensor,  # [num_sms+1, 2] int32 (cuda)
    max_context_len: int,
    num_epi_subtiles: int = 1,
    epi_dtype=cutlass.Float32,
    output_dtype=cutlass.Float32,
    num_sms: int = 148,
) -> torch.Tensor:
    """Standalone wrapper around FP4MQALogitsKernel; no trtllm dependency.

    Mirrors the production wrapper in cute_dsl_custom_ops.py: native
    next_n ∈ {1, 2, 3} only. next_n=4 is not handled here — callers needing
    it should apply caller-side atom-split reshape (e.g. ``run()`` below).
    """
    B, next_n, H, half_D = q.shape
    D = half_D * 2
    N = next_n * H
    phys_block_kv = kv_fused.shape[1]
    compute_block_kv = 128
    num_phys_blocks = kv_fused.shape[0]

    # Reshape (no .contiguous() — strides must stay B-independent to keep
    # the fake-tensor compile cache hot; see cute_dsl_custom_ops.py:5578-5584).
    q_3d = q.reshape(B, N, half_D).permute(1, 2, 0)
    sf_q_2d = sf_q.reshape(B, N).t()
    if epi_dtype == cutlass.Float16:
        w_2d = weights.reshape(B, N).half().t()
    elif epi_dtype == cutlass.BFloat16:
        w_2d = weights.reshape(B, N).bfloat16().t()
    else:
        w_2d = weights.reshape(B, N).t()
    kv_flat = kv_fused.reshape(num_phys_blocks, -1)

    # Output alignment to SPLIT_KV = compute_block_kv * NUM_MATH_WG (the kernel
    # may store unconditionally into the trailing pad).
    SPLIT_KV = compute_block_kv * 2
    aligned_max_ctx = ((max_context_len + SPLIT_KV - 1) // SPLIT_KV) * SPLIT_KV
    logits = torch.empty(
        (B * next_n, aligned_max_ctx),
        device=q.device,
        dtype=_CUTLASS_TO_TORCH[output_dtype],
    )
    logits = logits[:, :max_context_len]

    compiled = _compile_fp4_kernel(
        compute_block_kv,
        phys_block_kv,
        H,
        D,
        next_n,
        num_sms,
        num_epi_subtiles,
        epi_dtype,
        output_dtype,
    )
    compiled(
        kv_flat,
        q_3d,
        sf_q_2d,
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


def _ref_paged_mqa_logits(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
    max_model_len: int,
) -> torch.Tensor:
    """Pure-torch reference (mirrors DeepGEMM ref_paged_mqa_logits)."""
    batch_size, next_n, num_heads, dim = q.size()
    _, block_size, _, _ = kv_cache.size()
    logits = torch.full(
        [batch_size * next_n, max_model_len],
        float("-inf"),
        device=q.device,
        dtype=torch.float32,
    )
    cl_list = context_lens.tolist()
    for i in range(batch_size):
        ctx = cl_list[i]
        q_offsets = torch.arange(ctx - next_n, ctx, device=q.device)
        weight_slice = weights[i * next_n : (i + 1) * next_n, :].transpose(0, 1).contiguous()
        n_blk = (ctx + block_size - 1) // block_size
        block_idxs = block_tables[i][:n_blk]
        kv_slice = kv_cache[block_idxs]
        kx = kv_slice.permute(2, 3, 0, 1).reshape(kv_slice.size(2), dim, -1)
        qx = q[i].transpose(0, 1)
        s = torch.matmul(qx, kx).to(logits.dtype)
        total_len = n_blk * block_size
        k_offsets = torch.arange(0, total_len, device=q.device)
        mask = (k_offsets[None, :] < ctx) & (k_offsets[None, :] <= q_offsets[:, None])
        s = torch.where(mask[None, :, :], s, float("-inf"))
        s = torch.relu(s) * weight_slice[..., None]
        s = s.sum(dim=0)
        logits[i * next_n : (i + 1) * next_n, :total_len] = torch.where(
            k_offsets[None, :] <= q_offsets[:, None], s, float("-inf")
        )
    return logits


def _calc_diff(x: torch.Tensor, y: torch.Tensor) -> float:
    x, y = x.double(), y.double()
    denom = (x * x + y * y).sum()
    if denom == 0:
        return 0.0
    return float(1 - 2 * (x * y).sum() / denom)


def verify_schedule_meta(kernel_ctx_lens: torch.Tensor, num_sms: int) -> None:
    """Bit-exact equivalence check: CuTe DSL kernel vs. pure-Python reference."""
    ref = _compute_schedule_metadata(kernel_ctx_lens.cpu(), 128, num_sms).to(kernel_ctx_lens.device)
    dsl = get_paged_mqa_logits_metadata_cute_dsl(kernel_ctx_lens, 128, num_sms)
    if not torch.equal(ref, dsl):
        diff = (ref != dsl).nonzero(as_tuple=False)
        raise AssertionError(
            f"[verify_meta] FAIL: {diff.shape[0]} mismatched entries "
            f"(B={kernel_ctx_lens.shape[0]} num_sms={num_sms}); "
            f"first diff at {diff[0].tolist()}: ref={ref[tuple(diff[0])]}, dsl={dsl[tuple(diff[0])]}"
        )
    print(f"[verify_meta] PASS (B={kernel_ctx_lens.shape[0]} num_sms={num_sms})")


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
    verify_meta: bool = False,
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

    if verify_meta:
        verify_schedule_meta(kernel_ctx_lens, num_sms)

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
    logits_clean = logits_m.masked_fill(~finite, 0)
    ref_clean = ref_m.masked_fill(~finite, 0)

    # Element-wise check (matches `torch.testing.assert_close` semantics from
    # the unit test): |a-b| <= atol + rtol*|b| pointwise on in-context, finite
    # entries.
    atol, rtol = _ELEM_TOL[(_CUTLASS_TO_TORCH[epi_dtype], _CUTLASS_TO_TORCH[output_dtype])]
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

    # Cosine diff: supplementary global similarity metric (kept for continuity
    # with the previous PASS/FAIL output).
    diff = _calc_diff(logits_clean, ref_clean)
    diff_ok = diff < tol

    ok = elem_ok and diff_ok
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
        f"max_abs={max_abs:.3e} mean_abs={mean_abs:.3e} "
        f"(atol={atol:.0e} rtol={rtol:.0e}) "
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
    parser.add_argument(
        "--verify_meta",
        action="store_true",
        help="verify CuTe DSL schedule_meta kernel against the pure-Python reference",
    )
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
        verify_meta=args.verify_meta,
    )
