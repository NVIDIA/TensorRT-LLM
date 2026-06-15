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

// Threads-per-block. The kernel is launched as a single block: the phase-1
// no-op initialization is parallelized across these threads, while the phase-2
// run-merge is a sequential scan owned by thread 0 (see below).
constexpr int kBlockSize = 256;

// Run-length-aggregating problem builder.
//
// Instead of emitting one M=1 grouped-GEMM problem per permuted row, this
// merges each maximal run of consecutive rows that share the same
// (rank, A_ptr, B_ptr) into a single M=run_length problem. Because the builder
// addresses input/workspace/output by `row * row_stride`, a run of consecutive
// rows [s, e) is already one contiguous [M=e-s, K] block in every operand, so
// the merge needs no gather - it collapses to one problem with M = e - s. The
// per-expert weight offset is baked into A_ptr/B_ptr by the pointer-expand
// kernel, so crossing an expert (or adapter) boundary changes the pointers and
// breaks the run; expert boundaries are therefore respected automatically.
//
// Capture safety: the emitted array length stays fixed at P_max =
// num_permuted_tokens. The first `num_groups` slots hold real problems; the
// remaining slots [num_groups, P_max) are padded as N=0 no-ops (the grouped
// GEMM schedules zero tiles for them and never dereferences a pointer), so the
// host `problemCount` baked into a captured graph stays constant across
// replays - only the amount of real work shrinks.
//
// Implementation: phase 1 (all threads) initializes every slot as a no-op;
// phase 2 (thread 0 only) performs the sequential run-merge over the P rows,
// overwriting slots [0, num_groups) and computing the split-K prefix sum. The
// scan is sequential because run detection is inherently a neighbor-dependent
// prefix operation; the work is trivial metadata (a few int loads per row) and
// dwarfed by the GEMMs. A parallel flag-and-scan implementation is a possible
// future optimization (see lora_gemm_by_adapter.md).
__global__ void moeLoraProblemBuilderKernel(int32_t const* __restrict__ ranks, int64_t const* __restrict__ ptrs,
    int64_t input_base, int64_t lowrank_workspace, int64_t output_base, int64_t num_permuted_tokens,
    int64_t in_hidden_size, int64_t out_hidden_size, int64_t max_lora_rank, int64_t dtype_bytes, int64_t splitk_slices,
    cutlass::gemm::GemmCoord* __restrict__ problem_sizes_in, cutlass::gemm::GemmCoord* __restrict__ problem_sizes_out,
    void** __restrict__ a_ptrs_in, void** __restrict__ b_ptrs_in, void** __restrict__ d_ptrs_in,
    void** __restrict__ b_ptrs_out, void** __restrict__ d_ptrs_out, int64_t* __restrict__ lda_in,
    int64_t* __restrict__ ldb_in, int64_t* __restrict__ ldd_in, int64_t* __restrict__ ldb_out,
    int64_t* __restrict__ ldd_out, int64_t* __restrict__ splitk_offsets)
{
    int64_t const P = num_permuted_tokens;
    int64_t const in_row_stride = in_hidden_size * dtype_bytes;
    int64_t const work_row_stride = max_lora_rank * dtype_bytes;
    int64_t const out_row_stride = out_hidden_size * dtype_bytes;

    // Phase 1: initialize every slot as an N=0 no-op. The grouped GEMM
    // schedules zero tiles for these and never dereferences the pointers, so
    // the padded tail [num_groups, P_max) is a safe constant-count filler.
    // splitk_offsets is intentionally left untouched here - thread 0 owns the
    // entire split-K array in phase 2 to avoid a write race.
    for (int64_t p = static_cast<int64_t>(threadIdx.x); p < P; p += blockDim.x)
    {
        problem_sizes_in[p] = cutlass::gemm::GemmCoord(1, 0, static_cast<int>(in_hidden_size));
        problem_sizes_out[p] = cutlass::gemm::GemmCoord(1, 0, 0);
        a_ptrs_in[p] = reinterpret_cast<void*>(input_base);
        b_ptrs_in[p] = nullptr;
        d_ptrs_in[p] = reinterpret_cast<void*>(lowrank_workspace);
        b_ptrs_out[p] = nullptr;
        d_ptrs_out[p] = reinterpret_cast<void*>(output_base);
        lda_in[p] = in_hidden_size;
        ldb_in[p] = in_hidden_size;
        ldd_in[p] = max_lora_rank;
        ldb_out[p] = 0;
        ldd_out[p] = out_hidden_size;
    }

    __syncthreads();

    if (threadIdx.x != 0)
    {
        return;
    }

    // Phase 2: sequential run-merge over [0, P). Each maximal run of rows
    // sharing (rank, A_ptr, B_ptr) becomes one M=run_length problem.
    int64_t group = 0;
    int64_t splitk_acc = 0;
    int64_t i = 0;
    while (i < P)
    {
        int32_t const rank = ranks[i];
        // The workspace row and ldd_in use max_lora_rank, so a larger rank
        // makes the in-GEMM write past its slice. Callers validate ranks
        // host-side (see moeOp.cpp); this assert is a debug-build backstop.
        assert(rank <= max_lora_rank);
        int64_t const a_ptr_bits = ptrs[2 * i + 0];
        int64_t const b_ptr_bits = ptrs[2 * i + 1];

        // Extend the run while the next row shares the same adapter key.
        int64_t j = i + 1;
        while (j < P && ranks[j] == rank && ptrs[2 * j + 0] == a_ptr_bits && ptrs[2 * j + 1] == b_ptr_bits)
        {
            ++j;
        }
        int const M = static_cast<int>(j - i);

        // Rank-0 runs carry no active adapter (base/no-LoRA request, padding,
        // or warmup) and have null A/B pointers, so their delta is zero. The
        // in-GEMM already collapses to N=0 (rank is its N) and is skipped; the
        // out-GEMM's N is out_hidden_size, so force it to zero here too to let
        // the grouped GEMM skip these rows instead of dereferencing a null B.
        int const out_n = (rank > 0) ? static_cast<int>(out_hidden_size) : 0;
        problem_sizes_in[group] = cutlass::gemm::GemmCoord(M, rank, static_cast<int>(in_hidden_size));
        problem_sizes_out[group] = cutlass::gemm::GemmCoord(M, out_n, rank);

        // The run is contiguous in memory, so the merged problem starts at the
        // run's first row `i` in every operand; dtype_bytes scales the per-row
        // stride so the same builder serves bf16/fp16/fp32 without templating.
        a_ptrs_in[group] = reinterpret_cast<void*>(input_base + i * in_row_stride);
        b_ptrs_in[group] = reinterpret_cast<void*>(a_ptr_bits);
        d_ptrs_in[group] = reinterpret_cast<void*>(lowrank_workspace + i * work_row_stride);
        b_ptrs_out[group] = reinterpret_cast<void*>(b_ptr_bits);
        d_ptrs_out[group] = reinterpret_cast<void*>(output_base + i * out_row_stride);

        // Leading dimensions are uniform within a run. ldb is the per-problem
        // stride in the LoRA adapter's storage (matches the attention LoRA
        // cuda_graph_grouped_gemm convention in loraOp.cpp):
        //   in-GEMM:  adapter A stored as [rank, in_hidden_size] -> ldb_in = in_hidden_size
        //   out-GEMM: adapter B stored as [out_hidden_size, rank] -> ldb_out = rank
        // ldb_out = rank is uniform across the run precisely because the run
        // shares one adapter (and therefore one rank).
        lda_in[group] = in_hidden_size;
        ldb_in[group] = in_hidden_size;
        ldd_in[group] = max_lora_rank;
        ldb_out[group] = rank;
        ldd_out[group] = out_hidden_size;

        // Split-K scratch offsets become an exclusive prefix sum over per-run
        // M, since each run needs M * max_lora_rank * splitk_slices scratch.
        // The total (sum of M = P) is unchanged from the per-row layout.
        if (splitk_offsets != nullptr)
        {
            splitk_offsets[group] = splitk_acc;
        }
        splitk_acc += static_cast<int64_t>(M) * max_lora_rank * splitk_slices;

        ++group;
        i = j;
    }

    // Pad the split-K offsets for the no-op tail [num_groups, P) and the
    // element [P] sentinel with the (unchanged) total fp32 scratch size, which
    // matches what cuda_graph_split_k_grouped_gemm consumes.
    if (splitk_offsets != nullptr)
    {
        for (int64_t p = group; p < P; ++p)
        {
            splitk_offsets[p] = splitk_acc;
        }
        splitk_offsets[P] = splitk_acc;
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

    // Single block: phase 1 parallelizes the no-op initialization across the
    // block's threads, phase 2 runs the sequential run-merge on thread 0. The
    // emitted array length stays P_max = num_permuted_tokens regardless of how
    // many runs are produced, keeping the grouped-GEMM problem count constant
    // (and therefore CUDA-graph-capture-safe).
    moeLoraProblemBuilderKernel<<<1, kBlockSize, 0, stream>>>(ranks_dev, ptrs_dev,
        reinterpret_cast<int64_t>(input_base), reinterpret_cast<int64_t>(lowrank_workspace),
        reinterpret_cast<int64_t>(output_base), num_permuted_tokens, in_hidden_size, out_hidden_size, max_lora_rank,
        dtype_bytes, splitk_slices, out.problem_sizes_in, out.problem_sizes_out, out.a_ptrs_in, out.b_ptrs_in,
        out.d_ptrs_in, out.b_ptrs_out, out.d_ptrs_out, out.lda_in, out.ldb_in, out.ldd_in, out.ldb_out, out.ldd_out,
        out.splitk_offsets);
    sync_check_cuda_error(stream);
}

} // namespace kernels::cutlass_kernels

TRTLLM_NAMESPACE_END
