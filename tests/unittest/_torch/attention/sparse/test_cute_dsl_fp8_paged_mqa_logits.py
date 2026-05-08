# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Test CuTe DSL fp8_paged_mqa_logits kernel against C++ DeepGEMM reference.
"""

import pytest
import torch

from tensorrt_llm._utils import get_sm_version

skip_not_sm100 = pytest.mark.skipif(
    get_sm_version() not in (100, 103),
    reason=f"CuTe DSL FP8 Paged MQA Logits only supports SM 100/103, got SM {get_sm_version()}",
)


def _ceil_to_ue8m0(x: torch.Tensor):
    return torch.pow(2.0, torch.ceil(torch.log2(x.abs())))


def _ref_fp8_paged_mqa_logits(
    q_fp8,
    kv_fp8,
    kv_scales,
    weights,
    context_lens,
    block_table,
    max_model_len,
    block_kv,
    epi_dtype=torch.float32,
):
    """Pure PyTorch reference for fp8_paged_mqa_logits.

    Args:
        q_fp8: [B, next_n, H, D] float8_e4m3fn
        kv_fp8: [num_blocks, block_kv, D] float8_e4m3fn
        kv_scales: [num_blocks, block_kv] float32
        weights: [B*next_n, H] float32
        context_lens: [B] int32
        block_table: [B, max_blocks] int32
        max_model_len: int
        block_kv: int
        epi_dtype: epilogue dtype — GEMM stays fp32, weighted sum + scale
            use this dtype (torch.float32 or torch.float16)

    Returns:
        logits: [B*next_n, max_model_len] epi_dtype
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

            # GEMM in fp32
            qk = torch.matmul(q_f32[b].permute(1, 0, 2), k_f32.T)  # [H, next_n, block_kv]
            qk = torch.where(mask[None, :, :], qk, torch.zeros(1, device=device))
            qk = torch.relu(qk)

            # Epilogue in epi_dtype
            qk = qk.to(epi_dtype)
            weighted = (w.T[:, :, None] * qk).sum(dim=0)  # [next_n, block_kv]
            weighted = weighted * scales[None, :]

            start_pos = blk_idx * block_kv
            end_pos = start_pos + block_kv
            logits[b * next_n : (b + 1) * next_n, start_pos:end_pos] = torch.where(
                mask, weighted, torch.tensor(float("-inf"), device=device, dtype=epi_dtype)
            )

    return logits


def _make_fused_kv(kv_fp8, kv_scales, block_kv, head_dim):
    """Create fused KV in packed-by-type layout matching DeepGEMM/DSL kernel.

    Per block: [all FP8 bytes (block_kv * head_dim)] [all scale bytes (block_kv * 4)]
    Viewed as [num_blocks, block_kv, 1, head_dim + 4].
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


def _generate_test_data(
    batch_size,
    next_n,
    num_heads,
    head_dim,
    block_kv,
    avg_context_len,
    max_model_len,
    device="cuda",
    use_int_data=False,
    fix_length=True,
):
    """Generate test data for fp8 paged MQA logits.

    Args:
        use_int_data: When True, use small random integers ([-3, 3]) for Q/KV
            and integer weights so that GEMM accumulation is exact across
            FP8/FP16/FP32. Useful for isolating kernel bugs from precision.
    """
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    if fix_length:
        context_lens = torch.full((batch_size,), max_model_len, dtype=torch.int32, device="cpu")
    else:
        context_lens = torch.randint(
            max(block_kv, int(0.7 * avg_context_len)),
            int(1.3 * avg_context_len) + 1,
            (batch_size,),
            dtype=torch.int32,
            device="cpu",
        )
        context_lens = context_lens.clamp(max=max_model_len)

    max_blocks_per_seq = (max_model_len + block_kv - 1) // block_kv
    total_blocks = ((context_lens + block_kv - 1) // block_kv).sum().item()
    num_phys_blocks = total_blocks + batch_size * 2

    block_table = torch.full((batch_size, max_blocks_per_seq), 0, dtype=torch.int32, device=device)
    blk_offset = 0
    for i in range(batch_size):
        n_blks = (context_lens[i].item() + block_kv - 1) // block_kv
        block_table[i, :n_blks] = torch.arange(
            blk_offset, blk_offset + n_blks, dtype=torch.int32, device=device
        )
        blk_offset += n_blks

    if use_int_data:
        q_fp8 = torch.randint(
            -3,
            4,
            (batch_size, next_n, num_heads, head_dim),
            device=device,
            dtype=torch.float32,
        ).to(torch.float8_e4m3fn)

        kv_fp8 = torch.randint(
            -3,
            4,
            (num_phys_blocks, block_kv, head_dim),
            device=device,
            dtype=torch.float32,
        ).to(torch.float8_e4m3fn)
        kv_scale = torch.ones(num_phys_blocks, block_kv, device=device, dtype=torch.float32)

        weights = torch.randint(
            -3,
            4,
            (batch_size * next_n, num_heads),
            device=device,
            dtype=torch.float32,
        )
    else:
        q_bf16 = torch.randn(batch_size, next_n, num_heads, head_dim, device=device)
        q_fp8 = q_bf16.to(torch.float8_e4m3fn)

        kv_bf16 = torch.randn(num_phys_blocks, block_kv, head_dim, device=device)
        kv_amax = kv_bf16.abs().float().amax(dim=-1, keepdim=True).clamp(1e-4)
        kv_scale = _ceil_to_ue8m0(kv_amax / 448.0).squeeze(-1)
        kv_fp8 = (kv_bf16 / kv_scale.unsqueeze(-1)).to(torch.float8_e4m3fn)

        weights = torch.randn(batch_size * next_n, num_heads, device=device, dtype=torch.float32)

    kv_fused = _make_fused_kv(kv_fp8, kv_scale, block_kv, head_dim)

    return {
        "q_fp8": q_fp8,
        "kv_fp8": kv_fp8,
        "kv_scales": kv_scale,
        "kv_fused": kv_fused,
        "weights": weights,
        "context_lens": context_lens.to(device),
        "block_table": block_table,
        "max_model_len": max_model_len,
        "block_kv": block_kv,
        "num_phys_blocks": num_phys_blocks,
    }


@skip_not_sm100
@pytest.mark.parametrize("batch_size", [1, 4, 32])
@pytest.mark.parametrize("next_n", [1, 2, 3, 4])
@pytest.mark.parametrize("num_heads", [64])
@pytest.mark.parametrize("avg_ctx", [256, 4096, 32768])
@pytest.mark.parametrize("output_dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("fix_length", [True, False])
def test_cute_dsl_fp8_paged_mqa_logits(
    batch_size, next_n, num_heads, avg_ctx, output_dtype, fix_length
):
    """Compare CuTe DSL kernel output against a pure PyTorch reference.

    Tests both fp32 and fp16 epi/acc/output paths.
    """
    head_dim = 128
    block_kv = 128
    max_model_len = max(avg_ctx * 2, 2048)

    data = _generate_test_data(
        batch_size,
        next_n,
        num_heads,
        head_dim,
        block_kv,
        avg_ctx,
        max_model_len,
        use_int_data=(output_dtype == torch.float16),
        fix_length=fix_length,
    )

    from tensorrt_llm.deep_gemm import get_paged_mqa_logits_metadata

    num_sms = torch.cuda.get_device_properties(0).multi_processor_count

    # New DeepGEMM `get_paged_mqa_logits_metadata` arg conventions on SM100:
    #
    # 1) `context_lens` must be 2D. Passing (B, 1) makes the wrapper see
    #    `next_n = size(1) = 1` and compute `num_next_n_atoms = 1`, which
    #    matches DSL's 1-atom-per-q design (DSL always processes all real
    #    next_n positions in one atom regardless of value).
    #
    # 2) `block_kv` arg must be 64 — independent of the physical cache page
    #    size. The metadata kernel computes `SPLIT_KV = block_kv * 4` (the
    #    multiplier 4 is hardcoded in DeepGEMM's JIT impl, not arch-aware
    #    on SM100 since #304). Both DSL and DG compute kernels assume
    #    `SPLIT_KV = 256` (DG hardcodes it at apis/attention.hpp:353; DSL
    #    expects compute_tile=128 × kNumMathWarpGroups=2 = 256). So
    #    metadata must give SPLIT_KV=256 → `block_kv = 256 / 4 = 64`.
    #    Production passes `tokens_per_block` here, which equals 64 by
    #    DSV3 indexer-cache convention.
    DG_METADATA_BLOCK_KV = 64
    dsl_schedule_meta = get_paged_mqa_logits_metadata(
        data["context_lens"].unsqueeze(-1), DG_METADATA_BLOCK_KV, num_sms
    )

    ref_logits = _ref_fp8_paged_mqa_logits(
        data["q_fp8"],
        data["kv_fp8"],
        data["kv_scales"],
        data["weights"],
        data["context_lens"],
        data["block_table"],
        max_model_len,
        block_kv,
        epi_dtype=output_dtype,
    )

    # CuTe DSL kernel
    dsl_logits = torch.ops.trtllm.cute_dsl_fp8_paged_mqa_logits(
        data["q_fp8"],
        data["kv_fused"],
        data["weights"],
        data["context_lens"],
        data["block_table"],
        dsl_schedule_meta,
        max_model_len,
        epi_dtype=output_dtype,
        acc_dtype=output_dtype,
        output_dtype=output_dtype,
    )

    assert dsl_logits.dtype == output_dtype

    # Mask invalid positions
    B = batch_size
    positions = torch.arange(max_model_len, device="cuda").unsqueeze(0)
    row_indices = torch.arange(B * next_n, device="cuda") // next_n
    next_n_offset = torch.arange(B * next_n, device="cuda") % next_n
    end_pos = data["context_lens"][row_indices] - next_n + next_n_offset
    mask = positions <= end_pos.unsqueeze(1)

    dsl_masked = dsl_logits.float().masked_fill(~mask, 0)
    ref_masked = ref_logits.float().masked_fill(~mask, 0)
    finite = torch.isfinite(dsl_masked) & torch.isfinite(ref_masked)
    dsl_clean = dsl_masked.masked_fill(~finite, 0)
    ref_clean = ref_masked.masked_fill(~finite, 0)

    # Element-wise check on the valid (finite + in-context) region.
    # Kernel is deterministic (disjoint CTA writes, no atomics), so every
    # element must be within elem_atol.
    elem_atol = 1e-3 if output_dtype == torch.float16 else 5e-5
    elem_rtol = 1e-3 if output_dtype == torch.float16 else 1e-5

    # Debug probe: print max/mean abs error for CI failure diagnosis.
    valid = mask & finite
    elem_abs = (dsl_clean - ref_clean).abs()[valid]
    if elem_abs.numel() > 0:
        print(
            f"[acc-probe] B={batch_size} next_n={next_n} avg_ctx={avg_ctx} "
            f"dtype={output_dtype} -> "
            f"max_abs={elem_abs.max().item():.3e} "
            f"mean_abs={elem_abs.mean().item():.3e}"
        )

    torch.testing.assert_close(
        dsl_clean,
        ref_clean,
        atol=elem_atol,
        rtol=elem_rtol,
        msg=lambda m: (
            f"{m}\nB={batch_size}, next_n={next_n}, avg_ctx={avg_ctx}, dtype={output_dtype}"
        ),
    )


@skip_not_sm100
@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("next_n", [1, 2, 3, 4])
@pytest.mark.parametrize("num_heads", [64])
@pytest.mark.parametrize("avg_ctx", [256, 4096])
@pytest.mark.parametrize("phys_block_kv", [32, 64])
def test_cute_dsl_fp8_paged_mqa_logits_multi_block(
    batch_size, next_n, num_heads, avg_ctx, phys_block_kv
):
    """Test multi-block TMA: physical block < compute tile (128).

    When phys_block_kv < 128, the kernel issues num_blocks_per_mma
    separate TMA copies per compute tile to fill the 128-token SMEM.
    """
    head_dim = 128
    max_model_len = max(avg_ctx * 2, 2048)
    output_dtype = torch.float32

    data = _generate_test_data(
        batch_size,
        next_n,
        num_heads,
        head_dim,
        phys_block_kv,
        avg_ctx,
        max_model_len,
        fix_length=True,
    )

    from tensorrt_llm.deep_gemm import get_paged_mqa_logits_metadata

    num_sms = torch.cuda.get_device_properties(0).multi_processor_count

    # See `test_cute_dsl_fp8_paged_mqa_logits` above for the full reasoning.
    # Short version: DG metadata wrapper requires 2D context_lens, and
    # `block_kv` arg must be 64 (yields SPLIT_KV = 64 * 4 = 256, matching
    # DSL's compute-tile expectation). Independent of `phys_block_kv` /
    # `compute_block_kv` of the test cache.
    DG_METADATA_BLOCK_KV = 64
    dsl_schedule_meta = get_paged_mqa_logits_metadata(
        data["context_lens"].unsqueeze(-1), DG_METADATA_BLOCK_KV, num_sms
    )

    ref_logits = _ref_fp8_paged_mqa_logits(
        data["q_fp8"],
        data["kv_fp8"],
        data["kv_scales"],
        data["weights"],
        data["context_lens"],
        data["block_table"],
        max_model_len,
        phys_block_kv,
        epi_dtype=output_dtype,
    )

    dsl_logits = torch.ops.trtllm.cute_dsl_fp8_paged_mqa_logits(
        data["q_fp8"],
        data["kv_fused"],
        data["weights"],
        data["context_lens"],
        data["block_table"],
        dsl_schedule_meta,
        max_model_len,
        epi_dtype=output_dtype,
        acc_dtype=output_dtype,
        output_dtype=output_dtype,
    )

    assert dsl_logits.dtype == output_dtype

    B = batch_size
    positions = torch.arange(max_model_len, device="cuda").unsqueeze(0)
    row_indices = torch.arange(B * next_n, device="cuda") // next_n
    next_n_offset = torch.arange(B * next_n, device="cuda") % next_n
    end_pos = data["context_lens"][row_indices] - next_n + next_n_offset
    mask = positions <= end_pos.unsqueeze(1)

    dsl_masked = dsl_logits.float().masked_fill(~mask, 0)
    ref_masked = ref_logits.float().masked_fill(~mask, 0)
    finite = torch.isfinite(dsl_masked) & torch.isfinite(ref_masked)
    dsl_clean = dsl_masked.masked_fill(~finite, 0)
    ref_clean = ref_masked.masked_fill(~finite, 0)

    elem_atol = 5e-5
    elem_rtol = 1e-5

    valid = mask & finite
    elem_abs = (dsl_clean - ref_clean).abs()[valid]
    if elem_abs.numel() > 0:
        print(
            f"[multi-block] B={batch_size} next_n={next_n} avg_ctx={avg_ctx} "
            f"phys_block_kv={phys_block_kv} -> "
            f"max_abs={elem_abs.max().item():.3e} "
            f"mean_abs={elem_abs.mean().item():.3e}"
        )

    torch.testing.assert_close(
        dsl_clean,
        ref_clean,
        atol=elem_atol,
        rtol=elem_rtol,
        msg=lambda m: (
            f"{m}\nB={batch_size}, next_n={next_n}, avg_ctx={avg_ctx}, "
            f"phys_block_kv={phys_block_kv}"
        ),
    )


def _profile_kernel_us(fn, num_warmup=10, num_iterations=30):
    """Profile CUDA kernel time in microseconds using torch.profiler."""
    from torch.profiler import ProfilerActivity, profile

    for _ in range(num_warmup):
        fn()
    torch.cuda.synchronize()

    with profile(activities=[ProfilerActivity.CUDA], record_shapes=False) as prof:
        for _ in range(num_iterations):
            fn()
        torch.cuda.synchronize()

    total_cuda_us = 0
    for evt in prof.events():
        if evt.device_type == torch.autograd.DeviceType.CUDA:
            # for fp16 dtype, we use .half() to convert weights to fp16 dtype currently.
            # so we need to skip the vectorized_elementwise_kernel event.
            if "vectorized_elementwise_kernel" in evt.name:
                continue
            total_cuda_us += evt.device_time_total
    return total_cuda_us / num_iterations


def _generate_bench_data(
    batch_size,
    context_len,
    next_n,
    num_heads=64,
    head_dim=128,
    block_kv=128,
    varlen=False,
    device="cuda",
):
    """Generate benchmark data.

    ``context_len`` is treated as the max length. When varlen=False, all
    sequences use this exact length. When varlen=True, per-sequence lengths
    are drawn uniformly from [min(2048, max), max] to mimic real mixed-batch
    serving workloads.
    """
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    num_blocks_per_seq = (context_len + block_kv - 1) // block_kv

    if varlen:
        lo = min(2048, context_len)
        context_lens = torch.randint(
            lo, context_len + 1, (batch_size,), dtype=torch.int32, device=device
        )
        total_blocks = ((context_lens + block_kv - 1) // block_kv).sum().item()
        block_table = torch.zeros(
            (batch_size, num_blocks_per_seq), dtype=torch.int32, device=device
        )
        cursor = 0
        for i in range(batch_size):
            n_blks = (context_lens[i].item() + block_kv - 1) // block_kv
            block_table[i, :n_blks] = torch.arange(
                cursor, cursor + n_blks, dtype=torch.int32, device=device
            )
            cursor += n_blks
    else:
        total_blocks = batch_size * num_blocks_per_seq
        # fix-length workload: all sequences have the same context length.
        context_lens = torch.full((batch_size,), context_len, dtype=torch.int32, device=device)
        block_table = torch.arange(total_blocks, dtype=torch.int32, device=device).reshape(
            batch_size, num_blocks_per_seq
        )

    q_fp8 = torch.randn(
        batch_size, next_n, num_heads, head_dim, device=device, dtype=torch.bfloat16
    ).to(torch.float8_e4m3fn)
    weights = torch.randn(batch_size * next_n, num_heads, device=device, dtype=torch.float32)

    kv_fp8 = torch.randn(total_blocks, block_kv, head_dim, device=device, dtype=torch.bfloat16).to(
        torch.float8_e4m3fn
    )
    kv_scales = (
        torch.rand(total_blocks, block_kv, device=device, dtype=torch.float32) * 0.01 + 0.001
    )

    kv_fused = _make_fused_kv(kv_fp8, kv_scales, block_kv, head_dim)

    return {
        "q_fp8": q_fp8,
        "kv_fused": kv_fused,
        "weights": weights,
        "context_lens": context_lens,
        "block_table": block_table,
        "max_model_len": context_len,
        "total_blocks": total_blocks,
    }


def benchmark_fp8_paged_mqa_logits(
    batch_sizes,
    next_ns,
    context_lens,
    num_warmup=10,
    num_iterations=30,
    output_dtype=torch.float32,
    num_epi_subtiles=1,
    varlen=False,
    block_kv=128,
):
    """Benchmark CuTe DSL vs C++ DeepGEMM kernel time.

    Args:
        block_kv: physical block size (tokens per page). DSL scheduler always
            uses compute_block_kv=128; when block_kv < 128, the DSL kernel
            issues num_blocks_per_mma TMA copies per compute tile.
    """
    from tensorrt_llm.deep_gemm import get_paged_mqa_logits_metadata

    num_heads = 64
    head_dim = 128
    compute_block_kv = 128  # DSL scheduler / compute tile (always 128 on SM100)
    assert compute_block_kv % block_kv == 0, (
        f"compute_block_kv={compute_block_kv} must be divisible by block_kv={block_kv}"
    )
    num_sms = torch.cuda.get_device_properties(0).multi_processor_count

    dtype_str = str(output_dtype).split(".")[-1]
    mode_str = "varlen" if varlen else "fix-len"
    print(
        f"output_dtype={dtype_str}  num_epi_subtiles={num_epi_subtiles}  "
        f"mode={mode_str}  block_kv={block_kv}"
    )
    is_non_default = output_dtype != torch.float32 or num_epi_subtiles != 1
    hdr = (
        f"{'batch':>5s} {'ctx':>7s} {'next_n':>6s} {'nblk':>7s} | "
        f"{'DSL(us)':>8s} {'DG(fp32,us)':>12s} {'DG/DSL':>7s}"
    )
    if is_non_default:
        hdr += f" {'DSL(fp32,us)':>13s} {'DSL(fp32)/DSL':>13s}"
    print(hdr)
    print("-" * len(hdr))

    for next_n in next_ns:
        for context_len in context_lens:
            for batch_size in batch_sizes:
                nblk = batch_size * ((context_len + block_kv - 1) // block_kv)

                data = _generate_bench_data(
                    batch_size,
                    context_len,
                    next_n,
                    num_heads,
                    head_dim,
                    block_kv,
                    varlen=varlen,
                )

                # See `test_cute_dsl_fp8_paged_mqa_logits` for full reasoning
                # on the `block_kv = 64` choice. Short version: DG metadata
                # SPLIT_KV = block_kv * 4; we need SPLIT_KV = 256 (DSL
                # compute tile = 128 × kNumMathWarpGroups = 2), so pass 64.
                # 2D `(B, 1)` context_lens forces num_next_n_atoms = 1.
                DG_METADATA_BLOCK_KV = 64
                dsl_schedule_meta = get_paged_mqa_logits_metadata(
                    data["context_lens"].unsqueeze(-1),
                    DG_METADATA_BLOCK_KV,
                    num_sms,
                )

                def dsl_fn(data=data):
                    torch.ops.trtllm.cute_dsl_fp8_paged_mqa_logits(
                        data["q_fp8"],
                        data["kv_fused"],
                        data["weights"],
                        data["context_lens"],
                        data["block_table"],
                        dsl_schedule_meta,
                        data["max_model_len"],
                        num_epi_subtiles=num_epi_subtiles,
                        epi_dtype=output_dtype,
                        acc_dtype=output_dtype,
                        output_dtype=output_dtype,
                    )

                dsl_us = _profile_kernel_us(dsl_fn, num_warmup, num_iterations)

                dg_us = None
                try:
                    from tensorrt_llm.deep_gemm import fp8_paged_mqa_logits

                    # SM100 always uses num_kv_multicast=1 in upgraded DeepGEMM
                    # (cluster(2,1,1) for next_n=4 was removed). Atom-split is
                    # encoded in metadata via num_next_n_atoms which the wrapper
                    # derives from context_lens.size(1). DG natively supports
                    # next_n in {1,2,3,4}.
                    num_clusters = num_sms
                    # 2D context_lens shape (B, next_n): for next_n>1 the wrapper
                    # computes `num_next_n_atoms = next_n / next_n_atom_size`
                    # which DG's compute kernel expects. All next_n positions
                    # of a batch share the same KV length here (broadcast via
                    # expand) — TRT-LLM does the same in production.
                    dg_ctx_2d = data["context_lens"].unsqueeze(-1).expand(-1, next_n).contiguous()
                    # `block_kv = 64` for the same reason as the DSL path:
                    # metadata SPLIT_KV = block_kv * 4 must equal DG compute
                    # kernel's hardcoded SPLIT_KV = 256 (apis/attention.hpp:353).
                    # Independent of `compute_block_kv` of the test cache.
                    DG_METADATA_BLOCK_KV = 64
                    dg_schedule_meta = get_paged_mqa_logits_metadata(
                        dg_ctx_2d, DG_METADATA_BLOCK_KV, num_clusters
                    )

                    def dg_fn(data=data, dg_ctx_2d=dg_ctx_2d):
                        fp8_paged_mqa_logits(
                            data["q_fp8"],
                            data["kv_fused"],
                            data["weights"],
                            dg_ctx_2d,
                            data["block_table"],
                            dg_schedule_meta,
                            data["max_model_len"],
                        )

                    dg_us = _profile_kernel_us(dg_fn, num_warmup, num_iterations)
                except RuntimeError:
                    pass

                dsl_f32_us = None
                if is_non_default:

                    def dsl_f32_fn(data=data):
                        torch.ops.trtllm.cute_dsl_fp8_paged_mqa_logits(
                            data["q_fp8"],
                            data["kv_fused"],
                            data["weights"],
                            data["context_lens"],
                            data["block_table"],
                            dsl_schedule_meta,
                            data["max_model_len"],
                        )

                    dsl_f32_us = _profile_kernel_us(dsl_f32_fn, num_warmup, num_iterations)

                ratio_str = f"{dg_us / dsl_us:6.3f}x" if dg_us else "   N/A "
                dg_str = f"{dg_us:11.1f}" if dg_us else "        N/A"
                line = (
                    f"{batch_size:5d} {context_len:7d} {next_n:6d} "
                    f"{nblk:7d} | {dsl_us:7.1f} {dg_str} {ratio_str}"
                )
                if is_non_default:
                    f32_str = f"{dsl_f32_us:12.1f}" if dsl_f32_us else "         N/A"
                    f32_ratio = f"{dsl_f32_us / dsl_us:12.3f}x" if dsl_f32_us else "         N/A "
                    line += f" {f32_str} {f32_ratio}"
                print(line)

                del data
                torch.cuda.empty_cache()
            print()


if __name__ == "__main__":
    import argparse
    import os
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir))

    parser = argparse.ArgumentParser(description="Benchmark CuTe DSL fp8_paged_mqa_logits kernel")
    parser.add_argument(
        "--batch_size",
        type=int,
        nargs="+",
        default=[1, 32, 128],
        help="batch sizes (default: 1 32 128)",
    )
    parser.add_argument(
        "--next_n", type=int, nargs="+", default=[1, 2, 4], help="next_n values (default: 1 2 4)"
    )
    parser.add_argument(
        "--context_len",
        type=int,
        nargs="+",
        default=[4096, 32768, 131072],
        help="context lengths (default: 4096 32768 131072)",
    )
    parser.add_argument("--warmup", type=int, default=10, help="warmup iterations (default: 10)")
    parser.add_argument("--repeat", type=int, default=30, help="profiling iterations (default: 30)")
    parser.add_argument(
        "--output_dtype",
        type=str,
        default="float32",
        choices=["float32", "float16"],
        help="output dtype (default: float32)",
    )
    parser.add_argument(
        "--num_epi_subtiles",
        type=int,
        default=1,
        choices=[1, 2, 4],
        help="epilogue sub-tile count (default: 1)",
    )
    parser.add_argument(
        "--varlen",
        action="store_true",
        help="use varlen workload (per-seq lengths in [min(2048,max), max]); "
        "default is fix-length where all sequences use --context_len",
    )
    parser.add_argument(
        "--block_kv",
        type=int,
        default=64,
        choices=[32, 64, 128],
        help="physical block size / tokens per page (default: 64). "
        "DSL compute tile is always 128; when block_kv<128, DSL issues "
        "num_blocks_per_mma=128/block_kv TMA copies per compute tile.",
    )
    args = parser.parse_args()

    dtype_map = {"float32": torch.float32, "float16": torch.float16}
    benchmark_fp8_paged_mqa_logits(
        batch_sizes=args.batch_size,
        next_ns=args.next_n,
        context_lens=args.context_len,
        num_warmup=args.warmup,
        num_iterations=args.repeat,
        output_dtype=dtype_map[args.output_dtype],
        num_epi_subtiles=args.num_epi_subtiles,
        varlen=args.varlen,
        block_kv=args.block_kv,
    )
