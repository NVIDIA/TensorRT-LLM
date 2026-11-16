/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "tensorrt_llm/batch_manager/kvCacheManagerV2Utils.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/common/memoryUtils.h"
#include <cassert>
#include <cstdio>
#include <cstring>
#include <cuda.h>
#include <fcntl.h>
#include <memory>
#include <unistd.h>
#include <vector>

namespace tc = tensorrt_llm::common;
using namespace tensorrt_llm::runtime;

namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
{

SizeType32 IndexMapper::addNewSequence(LlmRequest::RequestIdType requestId)
{
    TLLM_CHECK(indexMap_.find(requestId) == indexMap_.end());
    auto iter = freeIndices_.begin();
    TLLM_CHECK_WITH_INFO(iter != freeIndices_.end(), "No free index found");
    auto index = *iter;
    freeIndices_.erase(iter);
    indexMap_[requestId] = index;
    return index;
}

SizeType32 IndexMapper::getIndex(LlmRequest::RequestIdType requestId)
{
    auto iter = indexMap_.find(requestId);
    TLLM_CHECK_WITH_INFO(iter != indexMap_.end(), "Request ID not found in IndexMapper");
    return iter->second;
}

void IndexMapper::removeSequence(LlmRequest::RequestIdType requestId)
{
    auto iter = indexMap_.find(requestId);
    TLLM_CHECK(iter != indexMap_.end());
    auto index = iter->second;
    freeIndices_.insert(index);
    indexMap_.erase(iter);
}

at::Tensor IndexMapper::getCopyIndex(std::vector<LlmRequest::RequestIdType> const& requestIds, SizeType32 numContext,
    SizeType32 beamWidth, bool replicateBeamZero)
{
    TLLM_CHECK_WITH_INFO(beamWidth <= maxCopyBeamWidth_, "Requested beam width %d exceeds the copy-index capacity %d",
        beamWidth, maxCopyBeamWidth_);
    TLLM_CHECK_WITH_INFO(replicateBeamZero || beamWidth <= maxBeamWidth_,
        "Requested beam width %d exceeds the source page-table width %d", beamWidth, maxBeamWidth_);
    int numSeqs = numContext + beamWidth * (requestIds.size() - numContext);
    SizeType32 batchSize = static_cast<SizeType32>(requestIds.size());
    SizeType32 idx = 0;
    for (SizeType32 i = 0; i < batchSize; i++)
    {
        if (i < numContext)
        {
            copyIndex_[idx++] = this->getIndex(requestIds[i]) * maxBeamWidth_;
        }
        else
        {
            for (SizeType32 j = 0; j < beamWidth; j++)
            {
                auto const sourceBeam = replicateBeamZero ? 0 : j;
                copyIndex_[idx++] = this->getIndex(requestIds[i]) * maxBeamWidth_ + sourceBeam;
            }
        }
    }

    TLLM_CHECK_WITH_INFO(idx == numSeqs, "Index mapper failed to generate copy index");

    return copyIndex_.slice(0, 0, numSeqs);
}

namespace
{
void gatherBasePageRowsImpl(at::Tensor const& source, at::Tensor destination, SizeType32 const* copyIndexData,
    int64_t copyIndexSize, SizeType32 numBlocks)
{
    constexpr int64_t kKvFactor = 2;
    TLLM_CHECK_WITH_INFO(source.device().is_cpu(), "source must be a CPU tensor");
    TLLM_CHECK_WITH_INFO(destination.device().is_cpu(), "destination must be a CPU tensor");
    TLLM_CHECK_WITH_INFO(source.scalar_type() == at::kInt, "source must contain int32 values");
    TLLM_CHECK_WITH_INFO(destination.scalar_type() == at::kInt, "destination must contain int32 values");
    TLLM_CHECK_WITH_INFO(source.is_contiguous(), "source must be contiguous");
    TLLM_CHECK_WITH_INFO(destination.is_contiguous(), "destination must be contiguous");
    TLLM_CHECK_WITH_INFO(
        source.dim() == 4 && source.size(2) == kKvFactor, "source must be [numPools, rowCapacity, 2, maxBlocksPerSeq]");
    TLLM_CHECK_WITH_INFO(destination.dim() == 4 && destination.size(2) == kKvFactor,
        "destination must be [numPools, numSequences, 2, numBlocksPerSeq]");
    TLLM_CHECK_WITH_INFO(destination.size(0) == source.size(0), "source and destination pool counts must match");
    TLLM_CHECK_WITH_INFO(destination.size(1) >= copyIndexSize, "destination must have one row per copyIndex entry");
    TLLM_CHECK_WITH_INFO(numBlocks > 0 && numBlocks <= source.size(3) && numBlocks <= destination.size(3),
        "numBlocks must fit both source and destination");

    auto const* sourceData = source.data_ptr<int32_t>();
    auto* destinationData = destination.data_ptr<int32_t>();
    auto const numPools = source.size(0);
    auto const sourceRows = source.size(1);
    auto const sourceBlocks = source.size(3);
    auto const destinationRows = destination.size(1);
    auto const destinationBlocks = destination.size(3);
    auto const copyBytes = static_cast<size_t>(numBlocks) * sizeof(int32_t);

    for (int64_t pool = 0; pool < numPools; ++pool)
    {
        for (int64_t destinationRow = 0; destinationRow < copyIndexSize; ++destinationRow)
        {
            auto const sourceRow = static_cast<int64_t>(copyIndexData[destinationRow]);
            TLLM_CHECK_WITH_INFO(sourceRow >= 0 && sourceRow < sourceRows, "copyIndex row is out of bounds");
            auto const sourceOffset = ((pool * sourceRows + sourceRow) * kKvFactor) * sourceBlocks;
            auto const destinationOffset = ((pool * destinationRows + destinationRow) * kKvFactor) * destinationBlocks;
            std::memcpy(destinationData + destinationOffset, sourceData + sourceOffset, copyBytes);
        }
    }
}
} // namespace

void gatherBasePageRows(
    at::Tensor const& source, at::Tensor destination, at::Tensor const& copyIndex, SizeType32 numBlocks)
{
    TLLM_CHECK_WITH_INFO(copyIndex.device().is_cpu(), "copyIndex must be a CPU tensor");
    TLLM_CHECK_WITH_INFO(copyIndex.scalar_type() == at::kInt, "copyIndex must contain int32 values");
    TLLM_CHECK_WITH_INFO(copyIndex.is_contiguous(), "copyIndex must be contiguous");
    TLLM_CHECK_WITH_INFO(copyIndex.dim() == 1, "copyIndex must be one-dimensional");
    gatherBasePageRowsImpl(source, destination, copyIndex.data_ptr<SizeType32>(), copyIndex.size(0), numBlocks);
}

void IndexMapper::gatherKBlockOffsets(at::Tensor const& source, at::Tensor destination,
    std::vector<LlmRequest::RequestIdType> const& requestIds, SizeType32 numBlocks)
{
    std::vector<SizeType32> sourceRows;
    sourceRows.reserve(requestIds.size());
    for (auto const requestId : requestIds)
    {
        sourceRows.push_back(getIndex(requestId) * maxBeamWidth_);
    }
    gatherBasePageRowsImpl(source, destination, sourceRows.data(), static_cast<int64_t>(sourceRows.size()), numBlocks);
}

IndexMapper::IndexMapper(SizeType32 maxBatchSize, SizeType32 maxBeamWidth, SizeType32 maxCopyBeamWidth)
    : maxBeamWidth_(maxBeamWidth)
    , maxCopyBeamWidth_(maxCopyBeamWidth == 0 ? maxBeamWidth : maxCopyBeamWidth)
{
    TLLM_CHECK_WITH_INFO(maxCopyBeamWidth_ >= maxBeamWidth_,
        "Copy-index beam width %d must be at least the source page-table beam width %d", maxCopyBeamWidth_,
        maxBeamWidth_);
    indexMap_.reserve(maxBatchSize);
    for (SizeType32 i = 0; i < maxBatchSize; i++)
    {
        freeIndices_.insert(i);
    }
    // Allocate copyIndex_ memory as pinned (page-locked) host memory
    copyIndex_ = at::empty(
        {maxBatchSize * maxCopyBeamWidth_}, at::TensorOptions().dtype(at::ScalarType::Int).pinned_memory(true));
}

IndexMapper::~IndexMapper()
{
    indexMap_.clear();
    freeIndices_.clear();
}

} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
