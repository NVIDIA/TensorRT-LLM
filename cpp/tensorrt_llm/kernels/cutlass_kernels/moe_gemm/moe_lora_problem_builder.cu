/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "tensorrt_llm/kernels/cutlass_kernels/include/moe_lora_problem_builder.h"

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaUtils.h"

#include <cassert>
#include <cstdint>

TRTLLM_NAMESPACE_BEGIN

namespace kernels::cutlass_kernels
{

namespace
{

// Threads-per-block. The kernel is bandwidth-bound, so block size mainly
// controls occupancy. 256 is a good default for Hopper/Blackwell.
constexpr int kBlockSize = 256;

// One thread per permuted row writes all output arrays. Each store stream is
// contiguous, so accesses coalesce; there is no inter-thread communication.
__global__ void moeLoraProblemBuilderKernel(int32_t const* __restrict__ ranks, int64_t const* __restrict__ ptrs,
    int64_t input_base, int64_t lowrank_workspace, int64_t output_base, int64_t num_permuted_tokens,
    int64_t in_hidden_size, int64_t out_hidden_size, int64_t max_lora_rank, int64_t dtype_bytes, int64_t splitk_slices,
    cutlass::gemm::GemmCoord* __restrict__ problem_sizes_in, cutlass::gemm::GemmCoord* __restrict__ problem_sizes_out,
    void** __restrict__ a_ptrs_in, void** __restrict__ b_ptrs_in, void** __restrict__ d_ptrs_in,
    void** __restrict__ b_ptrs_out, void** __restrict__ d_ptrs_out, int64_t* __restrict__ lda_in,
    int64_t* __restrict__ ldb_in, int64_t* __restrict__ ldd_in, int64_t* __restrict__ ldb_out,
    int64_t* __restrict__ ldd_out, int64_t* __restrict__ splitk_offsets)
{
    int64_t const i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= num_permuted_tokens)
    {
        // The +1 splitk_offsets sentinel (one past num_permuted_tokens) is
        // written by thread 0 of the last block; everyone else returns.
        if (i == num_permuted_tokens && splitk_offsets != nullptr)
        {
            splitk_offsets[num_permuted_tokens] = num_permuted_tokens * max_lora_rank * splitk_slices;
        }
        return;
    }

    int32_t const rank = ranks[i];
    // The workspace row and ldd_in[i] use max_lora_rank, so a larger rank makes
    // the in-GEMM write past its slice. Callers validate ranks host-side (see
    // moeOp.cpp); this assert is a debug-build backstop.
    assert(rank <= max_lora_rank);
    int64_t const a_ptr_bits = ptrs[2 * i + 0];
    int64_t const b_ptr_bits = ptrs[2 * i + 1];

    // Problem sizes: each permuted token gets its own (M=1) GEMM. This matches
    // worst-case scheduling with no run-length aggregation; a future
    // optimization can aggregate consecutive identical-adapter tokens.
    problem_sizes_in[i] = cutlass::gemm::GemmCoord(1, rank, static_cast<int>(in_hidden_size));
    problem_sizes_out[i] = cutlass::gemm::GemmCoord(1, static_cast<int>(out_hidden_size), rank);

    // Pointer rows. dtype_bytes scales the per-row stride so the same
    // builder serves bf16/fp16/fp32 adapters without templating.
    int64_t const in_row_stride = in_hidden_size * dtype_bytes;
    int64_t const work_row_stride = max_lora_rank * dtype_bytes;
    int64_t const out_row_stride = out_hidden_size * dtype_bytes;

    a_ptrs_in[i] = reinterpret_cast<void*>(input_base + i * in_row_stride);
    b_ptrs_in[i] = reinterpret_cast<void*>(a_ptr_bits);
    d_ptrs_in[i] = reinterpret_cast<void*>(lowrank_workspace + i * work_row_stride);
    b_ptrs_out[i] = reinterpret_cast<void*>(b_ptr_bits);
    d_ptrs_out[i] = reinterpret_cast<void*>(output_base + i * out_row_stride);

    // Leading dimensions. For the in-/out- GEMMs, lda/ldd correspond to the
    // input row-stride / workspace row-stride / output row-stride; ldb is
    // the per-problem stride in the LoRA adapter's storage and matches
    // the cuda_graph_grouped_gemm convention used by attention LoRA
    // (loraOp.cpp):
    //   in-GEMM:  adapter A stored as [rank, in_hidden_size]
    //             -> ldb_in = in_hidden_size
    //   out-GEMM: adapter B stored as [out_hidden_size, rank]
    //             -> ldb_out = rank (per-token, since per-token rank
    //                          can differ in slot-indexed multi-LoRA mode)
    lda_in[i] = in_hidden_size;
    ldb_in[i] = in_hidden_size;
    ldd_in[i] = max_lora_rank;
    ldb_out[i] = rank;
    ldd_out[i] = out_hidden_size;

    // Split-K scratch offsets. Worst-case fixed stride (independent of
    // per-token rank) so each thread computes its own offset locally; no
    // cross-thread scan needed.
    if (splitk_offsets != nullptr)
    {
        splitk_offsets[i] = i * max_lora_rank * splitk_slices;
    }
}

} // namespace

void launchMoeLoraProblemBuilder(int32_t const* ranks_dev, int64_t const* ptrs_dev, void const* input_base,
    void* lowrank_workspace, void* output_base, int64_t num_permuted_tokens, int64_t in_hidden_size,
    int64_t out_hidden_size, int64_t max_lora_rank, int64_t dtype_bytes, int64_t splitk_slices,
    MoeLoraGemmGroupArrays const& out, cudaStream_t stream)
{
    if (num_permuted_tokens <= 0)
    {
        return;
    }
    TLLM_CHECK_WITH_INFO(ranks_dev != nullptr, "ranks_dev must be non-null");
    TLLM_CHECK_WITH_INFO(ptrs_dev != nullptr, "ptrs_dev must be non-null");
    TLLM_CHECK_WITH_INFO(out.problem_sizes_in != nullptr, "problem_sizes_in must be non-null");
    TLLM_CHECK_WITH_INFO(out.problem_sizes_out != nullptr, "problem_sizes_out must be non-null");
    TLLM_CHECK_WITH_INFO(out.a_ptrs_in && out.b_ptrs_in && out.d_ptrs_in && out.b_ptrs_out && out.d_ptrs_out,
        "All ptr_*_in/out arrays must be non-null");
    TLLM_CHECK_WITH_INFO(
        out.lda_in && out.ldb_in && out.ldd_in && out.ldb_out && out.ldd_out, "All ld* arrays must be non-null");
    TLLM_CHECK_WITH_INFO(dtype_bytes > 0, "dtype_bytes must be positive");
    TLLM_CHECK_WITH_INFO(max_lora_rank > 0, "max_lora_rank must be positive");
    TLLM_CHECK_WITH_INFO(in_hidden_size > 0 && out_hidden_size > 0, "hidden sizes must be positive");
    TLLM_CHECK_WITH_INFO(splitk_slices > 0, "splitk_slices must be positive");

    // Launch one extra thread so the splitk_offsets[num_permuted_tokens]
    // sentinel can be filled by exactly one thread (cleaner than a
    // dedicated tail launch).
    int64_t const launch_count = num_permuted_tokens + (out.splitk_offsets != nullptr ? 1 : 0);
    int64_t const grid = (launch_count + kBlockSize - 1) / kBlockSize;

    moeLoraProblemBuilderKernel<<<static_cast<unsigned int>(grid), kBlockSize, 0, stream>>>(ranks_dev, ptrs_dev,
        reinterpret_cast<int64_t>(input_base), reinterpret_cast<int64_t>(lowrank_workspace),
        reinterpret_cast<int64_t>(output_base), num_permuted_tokens, in_hidden_size, out_hidden_size, max_lora_rank,
        dtype_bytes, splitk_slices, out.problem_sizes_in, out.problem_sizes_out, out.a_ptrs_in, out.b_ptrs_in,
        out.d_ptrs_in, out.b_ptrs_out, out.d_ptrs_out, out.lda_in, out.ldb_in, out.ldd_in, out.ldb_out, out.ldd_out,
        out.splitk_offsets);
    sync_check_cuda_error(stream);
}

} // namespace kernels::cutlass_kernels

TRTLLM_NAMESPACE_END
