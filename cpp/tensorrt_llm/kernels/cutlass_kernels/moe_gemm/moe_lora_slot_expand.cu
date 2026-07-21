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

#include "tensorrt_llm/kernels/cutlass_kernels/include/moe_lora_slot_expand.h"

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaUtils.h"

#include <cstdint>

TRTLLM_NAMESPACE_BEGIN

namespace kernels::cutlass_kernels
{

namespace
{

// Threads-per-block. The kernel is a small per-token gather (one slot lookup,
// a handful of stores per module), so block size only affects occupancy. 256
// is a safe default for the Hopper / Blackwell targets shared with the rest of
// the MoE path.
constexpr int kBlockSize = 256;

// Per-module gather. Inlined into the main kernel so a single thread handles
// fc1, fc2, and (optionally) gated for its token after one slot lookup.
__device__ inline void slotExpandOneModule(MoeLoraSlotExpandModule const& mod, int64_t t, int32_t slot, bool valid_slot)
{
    if (valid_slot)
    {
        mod.ranks_out[t] = mod.slot_ranks[slot];
        mod.ptrs_out[2 * t + 0] = mod.slot_ptrs[3 * slot + 0];
        mod.ptrs_out[2 * t + 1] = mod.slot_ptrs[3 * slot + 1];
    }
    else
    {
        // Out-of-range slot: emit an inactive (rank 0, null pointer) entry so the
        // downstream pointer-expand / grouped-GEMM treats this token as a no-op
        // instead of indexing out of bounds.
        mod.ranks_out[t] = 0;
        mod.ptrs_out[2 * t + 0] = 0;
        mod.ptrs_out[2 * t + 1] = 0;
    }
}

// One thread per source token. Reads its adapter slot from token_to_slot and
// scatters the slot's (rank, A_ptr, B_ptr) into the per-token output tables for
// each module.
__global__ void moeLoraSlotExpandKernel(int32_t const* __restrict__ token_to_slot, int64_t num_tokens,
    int64_t num_slots, MoeLoraSlotExpandModule fc1, MoeLoraSlotExpandModule fc2, MoeLoraSlotExpandModule gated,
    bool has_gated)
{
    int64_t const t = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (t >= num_tokens)
    {
        return;
    }

    int32_t const slot = token_to_slot[t];
    bool const valid_slot = (slot >= 0) && (static_cast<int64_t>(slot) < num_slots);

    slotExpandOneModule(fc1, t, slot, valid_slot);
    slotExpandOneModule(fc2, t, slot, valid_slot);
    if (has_gated)
    {
        slotExpandOneModule(gated, t, slot, valid_slot);
    }
}

} // namespace

void launchMoeLoraSlotExpand(int32_t const* token_to_slot, int64_t num_tokens, int64_t num_slots,
    MoeLoraSlotExpandModule const& fc1, MoeLoraSlotExpandModule const& fc2, MoeLoraSlotExpandModule const* gated,
    cudaStream_t stream)
{
    if (num_tokens <= 0)
    {
        return;
    }
    TLLM_CHECK_WITH_INFO(token_to_slot != nullptr, "token_to_slot must be non-null");
    TLLM_CHECK_WITH_INFO(num_slots > 0, "num_slots must be positive");
    TLLM_CHECK_WITH_INFO(
        fc1.slot_ranks != nullptr && fc1.slot_ptrs != nullptr && fc1.ranks_out != nullptr && fc1.ptrs_out != nullptr,
        "fc1 slot-expand module is missing a buffer");
    TLLM_CHECK_WITH_INFO(
        fc2.slot_ranks != nullptr && fc2.slot_ptrs != nullptr && fc2.ranks_out != nullptr && fc2.ptrs_out != nullptr,
        "fc2 slot-expand module is missing a buffer");

    bool const has_gated = gated != nullptr;
    MoeLoraSlotExpandModule const gated_arg = has_gated ? *gated : MoeLoraSlotExpandModule{};

    int64_t const grid = (num_tokens + kBlockSize - 1) / kBlockSize;
    moeLoraSlotExpandKernel<<<static_cast<unsigned int>(grid), kBlockSize, 0, stream>>>(
        token_to_slot, num_tokens, num_slots, fc1, fc2, gated_arg, has_gated);
    sync_check_cuda_error(stream);
}

} // namespace kernels::cutlass_kernels

TRTLLM_NAMESPACE_END
