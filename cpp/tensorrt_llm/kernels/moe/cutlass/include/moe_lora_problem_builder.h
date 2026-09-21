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

#pragma once

#include "tensorrt_llm/common/config.h"

#include "cutlass/gemm_coord.h"

#include <cstdint>
#include <cuda_runtime.h>

TRTLLM_NAMESPACE_BEGIN

namespace kernels::cutlass_kernels
{

// Caller-owned device-output bundle for one LoRA module. Each array is sized
// for the maximum permuted-token count the FusedMoeRunner expects to see;
// the builder fills the first num_permuted_tokens entries each call.
//
// Layout convention (mirrors attention LoRA in cuda_graph_grouped_gemm.h):
//   In-GEMM:  D = A @ B with C aliased to D when there's no bias.
//             A = input slice  [M=1, K=in_hidden_size]
//             B = adapter A   [K=in_hidden_size, N=rank]
//             D = lowrank slice [M=1, N=rank]
//   Out-GEMM: D = A @ B with C aliased to D.
//             A = lowrank slice [M=1, K=rank]    (= in-GEMM's D)
//             B = adapter B    [K=rank, N=out_hidden_size]
//             D = output slice [M=1, N=out_hidden_size]
//
// Because ptrC aliases ptrD in both GEMMs (no bias), only ptrD is exposed
// per GEMM; the cuda_graph_grouped_gemm wrapper accepts the same address
// for both. d_ptrs_in also serves as a_ptrs_out (the LoRA intermediate is
// the input to the second GEMM); only one set of low-rank pointers is
// produced for that reason.
struct MoeLoraGemmGroupArrays
{
    // Per-problem (M, N, K) for the in-GEMM and out-GEMM respectively.
    cutlass::gemm::GemmCoord* problem_sizes_in = nullptr;  // [P]
    cutlass::gemm::GemmCoord* problem_sizes_out = nullptr; // [P]

    // In-GEMM pointer arrays. ptr_c_in is implicit (== d_ptrs_in).
    void** a_ptrs_in = nullptr; // [P]: input row pointer
    void** b_ptrs_in = nullptr; // [P]: adapter A pointer (with per-expert offset)
    void** d_ptrs_in = nullptr; // [P]: lowrank workspace row (also a_ptrs_out)

    // Out-GEMM pointer arrays. ptr_c_out is implicit (== d_ptrs_out).
    void** b_ptrs_out = nullptr; // [P]: adapter B pointer (with per-expert offset)
    void** d_ptrs_out = nullptr; // [P]: output row pointer

    // Leading dimensions. All row-major, fixed per problem given uniform
    // input / lowrank-workspace / output strides.
    int64_t* lda_in = nullptr;  // [P]: in_hidden_size
    int64_t* ldb_in = nullptr;  // [P]: in_hidden_size (stride in adapter-A storage)
    int64_t* ldd_in = nullptr;  // [P]: max_lora_rank (workspace stride)
    int64_t* ldb_out = nullptr; // [P]: per-token rank (stride in adapter-B storage)
    int64_t* ldd_out = nullptr; // [P]: out_hidden_size

    // Per-problem exclusive prefix offset into the split-K scratch buffer
    // used by the in-GEMM. Element [P] (one past the end) holds the total
    // scratch size in fp32 elements, matching the layout that
    // cuda_graph_split_k_grouped_gemm consumes.
    int64_t* splitk_offsets = nullptr; // [P + 1]
};

// Device-side problem-and-pointer builder for one MoE LoRA module. It consumes
// the per-permuted-row outputs of launchMoeLoraPointerExpand plus uniform
// input, workspace, and output base addresses, and writes every device-resident
// input the cuda_graph_(split_k_)grouped_gemm wrappers need.
//
// Inputs:
//   ranks_dev: int32 [P], per-permuted-row LoRA rank.
//   ptrs_dev:  int64 [P*2], per-permuted-row (A_ptr + offset, B_ptr + offset),
//              already adjusted for the per-expert weight stride by the
//              pointer-expand kernel.
//
// Base pointers (the per-token row offset is computed inside the kernel from
// i * stride * dtype_bytes):
//   input_base:        [P, in_hidden_size]
//   lowrank_workspace: [P, max_lora_rank], reused as the in-GEMM output and
//                      the out-GEMM input.
//   output_base:       [P, out_hidden_size]
//
// Scalars:
//   in_hidden_size:  K for the in-GEMM, also lda_in[i] and ldb_in[i].
//   out_hidden_size: N for the out-GEMM, also ldd_out[i].
//   max_lora_rank:   ldd_in[i], the workspace stride, fixed regardless of the
//                    per-token rank so the GEMM lands at a known offset.
//                    The out-GEMM's ldb_out[i] is the per-token rank (adapter B
//                    is stored [out_hidden_size, rank]), not out_hidden_size.
//   dtype_bytes:     scalar size in bytes (2 for bf16/fp16, 4 for fp32).
//   splitk_slices:   split-K factor for the in-GEMM; drives the per-problem
//                    split-K scratch stride.
//
// The split-K stride is a worst-case fixed value (max_lora_rank * splitk_slices
// per problem) so the offsets can be computed from i alone without a prefix-sum.
void launchMoeLoraProblemBuilder(int32_t const* ranks_dev, int64_t const* ptrs_dev, void const* input_base,
    void* lowrank_workspace, void* output_base, int64_t num_permuted_tokens, int64_t in_hidden_size,
    int64_t out_hidden_size, int64_t max_lora_rank, int64_t dtype_bytes, int64_t splitk_slices,
    MoeLoraGemmGroupArrays const& out, cudaStream_t stream);

} // namespace kernels::cutlass_kernels

TRTLLM_NAMESPACE_END
