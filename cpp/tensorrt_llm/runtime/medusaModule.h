/*
 * Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/runtime/iTensor.h"
#include <vector>

namespace tensorrt_llm::runtime
{

class MedusaModule
{
public:
    using TensorPtr = ITensor::SharedPtr;
    using MedusaChoices = std::vector<std::vector<SizeType>>;

    explicit MedusaModule(SizeType medusaHeads, SizeType maxMedusaTokens) noexcept
        : mMedusaHeads(medusaHeads)
        , mMaxMedusaTokens(maxMedusaTokens)
        , mTokensPerStep(mMaxMedusaTokens + 1)
    {
        mNumPackedMasks = tensorrt_llm::common::divUp(mTokensPerStep, 32);
    }

    explicit MedusaModule() noexcept
        : MedusaModule(0, 0)
    {
    }

    MedusaModule(MedusaModule const& o) = default;
    MedusaModule& operator=(MedusaModule const& o) = default;

    [[nodiscard]] SizeType medusaHeads() const noexcept
    {
        return mMedusaHeads;
    }

    [[nodiscard]] SizeType maxMedusaTokens() const noexcept
    {
        return mMaxMedusaTokens;
    }

    [[nodiscard]] SizeType tokensPerStep() const noexcept
    {
        return mTokensPerStep;
    }

    [[nodiscard]] SizeType numPackedMasks() const noexcept
    {
        return mNumPackedMasks;
    }

    void initMedusaTensorsFromChoices(MedusaChoices& choices, TensorPtr& topKs, TensorPtr& positionOffsets,
        TensorPtr& treeIds, TensorPtr& paths, TensorPtr& packedMask, SizeType& totalPaths) const noexcept;

private:
    using Prefix = uint64_t;
    static SizeType constexpr PREFIX_CHUNK_SIZE_BITS = 4;
    static SizeType constexpr PREFIX_MAX_VALUE = 16;

    struct MedusaTreeNode
    {
        SizeType nodeId;
        SizeType depth;
        SizeType parentLinearIdx;
        SizeType linearIdx;
        std::vector<SizeType> childLinearIndices;
    };

    SizeType computePathsAndMask(
        std::vector<MedusaTreeNode> const& tree, TensorPtr& packedMask, TensorPtr& paths) const;

    void copyPackedMask(TensorPtr& mask, SizeType srcIdx, SizeType dstIdx) const;
    void setOnePackedMask(TensorPtr& mask, SizeType row, SizeType col) const;
    Prefix computePrefix(std::vector<SizeType> const& vec, SizeType len) const;

    void dumpChoices(MedusaChoices const& choices, std::vector<SizeType> const& indices) const;

private:
    SizeType mMedusaHeads;
    SizeType mMaxMedusaTokens;
    SizeType mTokensPerStep;
    SizeType mNumPackedMasks;
    MedusaChoices mMedusaChoices;
};
} // namespace tensorrt_llm::runtime
