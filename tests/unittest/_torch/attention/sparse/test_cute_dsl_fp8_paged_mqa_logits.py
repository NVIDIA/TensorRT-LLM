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

import random

import pytest
import torch

from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE
from tensorrt_llm._utils import get_sm_version

skip_not_sm100 = pytest.mark.skipif(
    get_sm_version() not in (100, 103),
    reason=f"CuTe DSL FP8 Paged MQA Logits only supports SM 100/103, got SM {get_sm_version()}",
)


def has_deep_gemm():
    try:
        from tensorrt_llm import deep_gemm

        return deep_gemm is not None
    except Exception:
        return False


def _ceil_to_ue8m0(x: torch.Tensor):
    return torch.pow(2.0, torch.ceil(torch.log2(x.abs())))


def _calc_diff(x: torch.Tensor, y: torch.Tensor):
    x, y = x.double(), y.double()
    denominator = (x * x + y * y).sum()
    if denominator == 0:
        return 0.0
    sim = 2 * (x * y).sum() / denominator
    return (1 - sim).item()


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
):
    """Generate test data for fp8 paged MQA logits.

    Args:
        use_int_data: When True, use small random integers ([-3, 3]) for Q/KV
            and integer weights so that GEMM accumulation is exact across
            FP8/FP16/FP32. Useful for isolating kernel bugs from precision.
    """
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


skip_if_unsupported = pytest.mark.skipif(
    not (has_deep_gemm() and IS_CUTLASS_DSL_AVAILABLE), reason="Requires DeepGEMM and CuTe DSL"
)


@skip_if_unsupported
@skip_not_sm100
@pytest.mark.parametrize("batch_size", [1, 4, 32])
@pytest.mark.parametrize("next_n", [1, 2, 3, 4])
@pytest.mark.parametrize("avg_ctx", [256, 4096, 32768])
@pytest.mark.parametrize("output_dtype", [torch.float32, torch.float16])
def test_cute_dsl_fp8_paged_mqa_logits(batch_size, next_n, avg_ctx, output_dtype):
    """Compare CuTe DSL kernel output against reference.

    Uses C++ DeepGEMM as reference when available (next_n in {1,2,4}),
    falls back to pure PyTorch reference otherwise (e.g. next_n=3).
    Tests both fp32 and fp16 epi/acc/output paths.
    """
    torch.manual_seed(42)
    random.seed(42)

    num_heads = 64
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
    )

    from tensorrt_llm.deep_gemm import get_paged_mqa_logits_metadata

    num_sms = torch.cuda.get_device_properties(0).multi_processor_count

    # DSL kernel always uses full num_sms as grid size.
    dsl_schedule_meta = get_paged_mqa_logits_metadata(data["context_lens"], block_kv, num_sms)

    # Reference: C++ DeepGEMM is fp32-only and doesn't support next_n=3,
    # so only used for fp32 + next_n ∈ {1,2,4}. All other cases use PyTorch ref.
    ref_logits = None
    if output_dtype == torch.float32:
        try:
            from tensorrt_llm.deep_gemm import fp8_paged_mqa_logits

            num_kv_multicast = 2 if next_n == 4 else 1
            num_clusters = num_sms // num_kv_multicast
            dg_schedule_meta = get_paged_mqa_logits_metadata(
                data["context_lens"], block_kv, num_clusters
            )
            ref_logits = fp8_paged_mqa_logits(
                data["q_fp8"],
                data["kv_fused"],
                data["weights"],
                data["context_lens"],
                data["block_table"],
                dg_schedule_meta,
                max_model_len,
            )
        except RuntimeError:
            pass

    if ref_logits is None:
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
):
    """Benchmark CuTe DSL vs C++ DeepGEMM kernel time."""
    from tensorrt_llm.deep_gemm import get_paged_mqa_logits_metadata

    num_heads = 64
    head_dim = 128
    block_kv = 128
    num_sms = torch.cuda.get_device_properties(0).multi_processor_count

    dtype_str = str(output_dtype).split(".")[-1]
    mode_str = "varlen" if varlen else "fix-len"
    print(f"output_dtype={dtype_str}  num_epi_subtiles={num_epi_subtiles}  mode={mode_str}")
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

                dsl_schedule_meta = get_paged_mqa_logits_metadata(
                    data["context_lens"], block_kv, num_sms
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

                    num_kv_multicast = 2 if next_n == 4 else 1
                    num_clusters = num_sms // num_kv_multicast
                    dg_schedule_meta = get_paged_mqa_logits_metadata(
                        data["context_lens"], block_kv, num_clusters
                    )

                    def dg_fn(data=data):
                        fp8_paged_mqa_logits(
                            data["q_fp8"],
                            data["kv_fused"],
                            data["weights"],
                            data["context_lens"],
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
    )
