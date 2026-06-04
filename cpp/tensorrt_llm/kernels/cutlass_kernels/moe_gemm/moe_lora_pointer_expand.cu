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

#include "tensorrt_llm/kernels/cutlass_kernels/include/moe_lora_pointer_expand.h"

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaUtils.h"

#include <cstdint>

TRTLLM_NAMESPACE_BEGIN

namespace kernels::cutlass_kernels
{

namespace
{

// Threads-per-block. The kernel is bandwidth-bound, so block size mainly
// affects occupancy. 256 is a good default for Hopper/Blackwell.
constexpr int kBlockSize = 256;

// Cap on num_experts_per_node staged in shared memory for the expert lookup.
// Above this the kernel falls back to a global-memory scan (still correct).
// Set well above realistic values (typical MoE uses 8-64 experts per node).
constexpr int kMaxExpertsInSmem = 1024;

// Per-module expansion. Inlined into the main kernel so we only pay one
// permuted_rows[i] and expert lookup per output row.
__device__ inline void expandOneModule(
    MoeLoraExpandModule const& mod, int64_t i, int32_t source_index, int64_t weight_index, int64_t lora_dtype_bytes)
{
    int32_t const rank = mod.ranks_src[source_index];

    // Per-expert byte offsets: weight_index * dim * rank * sizeof(scalar).
    int64_t const a_stride = weight_index * mod.dim_a * rank * lora_dtype_bytes;
    int64_t const b_stride = weight_index * mod.dim_b * rank * lora_dtype_bytes;

    int64_t const a_src = mod.ptrs_src[2 * source_index + 0];
    int64_t const b_src = mod.ptrs_src[2 * source_index + 1];

    // Pointer arithmetic in raw bytes (uintptr_t-equivalent). Consumers
    // reinterpret to the LoRA scalar type with no further offset, matching
    // the existing host-loop semantics in setupLoraWorkspace.
    mod.ptrs_out[2 * i + 0] = a_src + a_stride;
    mod.ptrs_out[2 * i + 1] = b_src + b_stride;
    mod.ranks_out[i] = rank;
}

// One thread per permuted row: find its expert via search over
// expert_first_token_offset (staged in shared memory), compute
// source_index = permuted_rows[i] % num_rows, and expand fc1, fc2, and
// (optionally) gated. Rank 0 is a per-token no-op in the grouped GEMM, so no
// global "any-token-has-lora" reduction is needed.
__global__ void moeLoraPointerExpandKernel(int32_t const* __restrict__ permuted_rows,
    int64_t const* __restrict__ expert_first_token_offset, int32_t num_experts_per_node, int32_t start_expert,
    int64_t num_rows, int64_t expanded_num_rows, int64_t lora_dtype_bytes, MoeLoraExpandModule fc1,
    MoeLoraExpandModule fc2, MoeLoraExpandModule gated, bool has_gated)
{
    // Stage expert_first_token_offset in shared memory once per block; every
    // thread reads it during the expert lookup below.
    extern __shared__ int64_t smem_first_token_offset[];
    bool const use_smem = num_experts_per_node + 1 <= kMaxExpertsInSmem;
    if (use_smem)
    {
        for (int e = threadIdx.x; e < num_experts_per_node + 1; e += blockDim.x)
        {
            smem_first_token_offset[e] = expert_first_token_offset[e];
        }
        __syncthreads();
    }

    int64_t const i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= expanded_num_rows)
    {
        return;
    }

    // Find expert_idx s.t. first_offset[expert_idx] <= i < first_offset[expert_idx + 1].
    // Linear scan; num_experts_per_node is small enough (~8-64 typical) that
    // a binary search adds branch divergence with no meaningful speedup.
    int64_t const* offsets = use_smem ? smem_first_token_offset : expert_first_token_offset;
    int32_t expert_idx = 0;
    for (int32_t e = 0; e < num_experts_per_node; ++e)
    {
        if (offsets[e + 1] > i)
        {
            expert_idx = e;
            break;
        }
        expert_idx = e + 1;
    }
    // Tokens past the last valid offset (padding "ghost" rows) get
    // expert_idx == num_experts_per_node; drop them so weight_index cannot run
    // off the expert table.
    if (expert_idx >= num_experts_per_node)
    {
        return;
    }

    int64_t const weight_index = static_cast<int64_t>(expert_idx) + start_expert;
    int32_t const source_index = static_cast<int32_t>(permuted_rows[i] % num_rows);

    expandOneModule(fc1, i, source_index, weight_index, lora_dtype_bytes);
    expandOneModule(fc2, i, source_index, weight_index, lora_dtype_bytes);
    if (has_gated)
    {
        expandOneModule(gated, i, source_index, weight_index, lora_dtype_bytes);
    }
}

} // namespace

void launchMoeLoraPointerExpand(int32_t const* permuted_rows, int64_t const* expert_first_token_offset,
    int32_t num_experts_per_node, int32_t start_expert, int64_t num_rows, int64_t expanded_num_rows,
    int64_t lora_dtype_bytes, MoeLoraExpandModule const& fc1, MoeLoraExpandModule const& fc2,
    MoeLoraExpandModule const* gated, cudaStream_t stream)
{
    if (expanded_num_rows <= 0)
    {
        return;
    }
    TLLM_CHECK_WITH_INFO(permuted_rows != nullptr, "permuted_rows must be non-null");
    TLLM_CHECK_WITH_INFO(expert_first_token_offset != nullptr, "expert_first_token_offset must be non-null");
    TLLM_CHECK_WITH_INFO(num_experts_per_node > 0, "num_experts_per_node must be positive");
    TLLM_CHECK_WITH_INFO(num_rows > 0, "num_rows must be positive");
    TLLM_CHECK_WITH_INFO(lora_dtype_bytes > 0, "lora_dtype_bytes must be positive");

    bool const has_gated = gated != nullptr;
    MoeLoraExpandModule const gated_arg = has_gated ? *gated : MoeLoraExpandModule{};

    int64_t const grid = (expanded_num_rows + kBlockSize - 1) / kBlockSize;
    // Reserve shared memory only when the expert table fits. Above the cap the
    // kernel falls back to global-memory reads (still correct, no staging), so
    // we pass 0 bytes to avoid allocating shared memory we will not touch.
    int const smem_entries = num_experts_per_node + 1;
    size_t const smem_bytes
        = (smem_entries <= kMaxExpertsInSmem) ? static_cast<size_t>(smem_entries) * sizeof(int64_t) : 0;

    moeLoraPointerExpandKernel<<<static_cast<unsigned int>(grid), kBlockSize, smem_bytes, stream>>>(permuted_rows,
        expert_first_token_offset, num_experts_per_node, start_expert, num_rows, expanded_num_rows, lora_dtype_bytes,
        fc1, fc2, gated_arg, has_gated);
    sync_check_cuda_error(stream);
}

} // namespace kernels::cutlass_kernels

TRTLLM_NAMESPACE_END
