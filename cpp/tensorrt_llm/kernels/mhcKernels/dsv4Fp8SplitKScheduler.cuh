/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/types.cuh>

TRTLLM_NAMESPACE_BEGIN

namespace kernels::mhc::dsv4_splitk
{

enum class IndexType
{
    MN,
    K,
    SF_K,
};

template <uint32_t BlockM, uint32_t BlockN, uint32_t NumSms, bool IsMulticastOnA>
static constexpr uint32_t getNum1DBlocksPerGroup()
{
    uint32_t best = 0;
    uint32_t minUsage = cute::numeric_limits<uint32_t>::max();
    for (uint32_t candidate : {8u, 16u})
    {
        uint32_t const usage = IsMulticastOnA
            ? candidate * BlockN + deep_gemm::math::constexpr_ceil_div(NumSms, candidate) * BlockM
            : candidate * BlockM + deep_gemm::math::constexpr_ceil_div(NumSms, candidate) * BlockN;
        if (usage < minUsage)
        {
            best = candidate;
            minUsage = usage;
        }
    }
    return best;
}

// Normal-GEMM-only persistent scheduler owned by the DSV4 O_b path. Keeping
// the split index outside the M/N tile index preserves adjacent CTA pairing
// for the 2-CTA cluster-N multicast used by the kernel.
template <uint32_t BlockM, uint32_t BlockN, uint32_t NumMulticast, bool IsMulticastOnA, uint32_t NumSms,
    uint32_t SplitKFactor,
    uint32_t Num1DBlocksPerGroup = getNum1DBlocksPerGroup<BlockM, BlockN, NumSms, IsMulticastOnA>()>
struct SplitKScheduler
{
    int current_iter = -1;
    uint32_t num_blocks;
    uint32_t num_mn_blocks;
    uint32_t num_m_blocks;
    uint32_t num_n_blocks;
    uint32_t num_blocks_in_group;
    uint32_t split_k_idx = 0;
    uint32_t current_group_idx = 0;
    uint32_t current_shape_k;

    CUTLASS_DEVICE explicit SplitKScheduler(
        uint32_t shapeM, uint32_t shapeN, uint32_t shapeK, int* groupedLayout = nullptr)
        : num_m_blocks(deep_gemm::math::ceil_div(shapeM, BlockM))
        , num_n_blocks(deep_gemm::math::ceil_div(shapeN, BlockN))
        , current_shape_k(shapeK)
    {
        static_cast<void>(groupedLayout);
        num_mn_blocks = num_m_blocks * num_n_blocks;
        num_blocks = num_mn_blocks * SplitKFactor;
    }

    CUTLASS_DEVICE void getSwizzledBlockIdx(uint32_t blockIdx, uint32_t& mBlockIdx, uint32_t& nBlockIdx)
    {
        static_assert(Num1DBlocksPerGroup % NumMulticast == 0, "Invalid split-K scheduler group size");
        uint32_t const primaryBlocks = IsMulticastOnA ? num_n_blocks : num_m_blocks;
        uint32_t const secondaryBlocks = IsMulticastOnA ? num_m_blocks : num_n_blocks;
        uint32_t const blocksPerGroup = secondaryBlocks * Num1DBlocksPerGroup;
        uint32_t const groupIdx = blockIdx / blocksPerGroup;
        uint32_t const firstBlockIdx = groupIdx * Num1DBlocksPerGroup;
        uint32_t const inGroupIdx = blockIdx % blocksPerGroup;
        num_blocks_in_group = cute::min(Num1DBlocksPerGroup, primaryBlocks - firstBlockIdx);

        if constexpr (IsMulticastOnA)
        {
            mBlockIdx = inGroupIdx / num_blocks_in_group;
            nBlockIdx = firstBlockIdx + inGroupIdx % num_blocks_in_group;
        }
        else
        {
            mBlockIdx = firstBlockIdx + inGroupIdx % num_blocks_in_group;
            nBlockIdx = inGroupIdx / num_blocks_in_group;
        }
    }

    template <bool WithGroupOffset, IndexType Type = IndexType::MN>
    CUTLASS_DEVICE uint32_t get_global_idx(
        uint32_t shapeDim, uint32_t blockSize, uint32_t blockIdx, uint32_t mBlockIdx = 0) const
    {
        static_cast<void>(WithGroupOffset);
        static_cast<void>(Type);
        static_cast<void>(shapeDim);
        static_cast<void>(mBlockIdx);
        return blockIdx * blockSize;
    }

    CUTLASS_DEVICE uint32_t get_aligned_effective_m_in_block(uint32_t mBlockIdx) const
    {
        static_cast<void>(mBlockIdx);
        static_assert(BlockM % 16 == 0, "DSV4 split-K block M must be 16-aligned");
        return BlockM;
    }

    CUTLASS_DEVICE bool get_next_block(uint32_t& mBlockIdx, uint32_t& nBlockIdx)
    {
        uint32_t const nextBlockIdx = static_cast<uint32_t>(++current_iter) * gridDim.x + blockIdx.x;
        if (nextBlockIdx >= num_blocks)
        {
            return false;
        }

        uint32_t const mnBlockIdx = nextBlockIdx % num_mn_blocks;
        split_k_idx = nextBlockIdx / num_mn_blocks;
        getSwizzledBlockIdx(mnBlockIdx, mBlockIdx, nBlockIdx);
        return true;
    }
};

} // namespace kernels::mhc::dsv4_splitk

TRTLLM_NAMESPACE_END
