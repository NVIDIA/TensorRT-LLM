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
import sys
from pathlib import Path

import cutlass
import cutlass.cute as cute
import torch

try:
    from tensorrt_llm._torch.cute_dsl_kernels.blackwell.paged_mqa_logits import FP4MQALogitsKernel
except (ModuleNotFoundError, ImportError):
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


# ---- Pure-Python schedule metadata (replaces deep_gemm.get_paged_mqa_logits_metadata)


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
