/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#pragma once

#include "cacheTransBuffer.h"
#include "tensorrt_llm/batch_manager/kvCacheManager.h"
#include "tensorrt_llm/batch_manager/kvCacheUtils.h"
#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/envUtils.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/executor/cacheCommunicator.h"
#include "tensorrt_llm/executor/cache_transmission/cacheSplitConcat.h"
#include "tensorrt_llm/executor/dataTransceiverState.h"
#include "tensorrt_llm/runtime/bufferManager.h"
#include "tensorrt_llm/runtime/utils/mpiUtils.h"
#include <NvInferRuntimeBase.h>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <vector>

// Forward declare TransferSession in the correct global namespace scope
namespace tensorrt_llm::batch_manager
{
class TransferSession;
size_t computeBufferIdx(size_t processIdx, executor::kv_cache::TargetRanksInfo const& targetInfo);

void sendBuffer(TransferSession& session, int deviceId, size_t processIdx,
    std::vector<runtime::ITensor::SharedPtr> const& outputBuffers, size_t bufferCoverTargetNum,
    runtime::ITensor::SharedPtr const& preAllocSendBuffer, runtime::BufferManager const& bufferManager,
    executor::kv_cache::TargetRanksInfo const& targetInfo);

void sendAllBuffers(TransferSession& session, int deviceId,
    std::vector<runtime::ITensor::SharedPtr> const& outputBuffers, size_t bufferCoverTargetNum,
    runtime::ITensor::SharedPtr const& preAllocSendBuffer, runtime::BufferManager const& bufferManager,
    executor::kv_cache::TargetRanksInfo const& targetInfo, std::vector<size_t> const& pickUpConnections);

namespace cache_formatter_utils
{
/**
 * @brief Check if this rank should send cache data
 * @tparam CacheStateT Either kv_cache::CacheState or rnn_cache::RnnCacheState
 */
template <typename CacheStateT>
bool needSendCache(CacheStateT const& selfConfig, CacheStateT const& destConfig, runtime::SizeType32 selfIdx)
{
    auto targetInfo = executor::kv_cache::targetIRanks(destConfig, selfConfig, selfIdx);
    if (targetInfo.mDupHeadFactor <= 1)
    {
        return true;
    }

    int selfCpSize = selfConfig.getParallelConfig().mContextParallelism;
    int selfTpRank = (selfIdx % (selfConfig.getParallelConfig().mTensorParallelism * selfCpSize)) / selfCpSize;
    int selfTpRankInDpGroup = selfTpRank;

    if (selfConfig.getParallelConfig().mEnableAttentionDP)
    {
        int selfTPNumInDPGroup
            = selfConfig.getParallelConfig().mTensorParallelism / selfConfig.getParallelConfig().mDPsize;
        selfTpRankInDpGroup = selfTpRank % selfTPNumInDPGroup;
    }

    int destDPRank = destConfig.getParallelConfig().mEnableAttentionDP ? destConfig.getParallelConfig().mDPrank : 0;

    return (destDPRank % targetInfo.mDupHeadFactor) == (selfTpRankInDpGroup % targetInfo.mDupHeadFactor);
}

/**
 * @brief Pick send connections
 * @tparam CacheStateT Either kv_cache::CacheState or rnn_cache::RnnCacheState
 */
template <typename CacheStateT>
std::vector<size_t> pickSendConnections(size_t numConnections, CacheStateT const& selfConfig, SizeType32 selfIdx,
    CacheStateT const& destConfig, std::vector<SizeType32> const& counterPartRanks)
{
    TLLM_CHECK(numConnections == counterPartRanks.size());
    auto targetInfo = executor::kv_cache::targetIRanks(destConfig, selfConfig, selfIdx);

    // NO duplicate head filtering - format/sendBuffer handles that
    std::vector<size_t> indices;
    for (auto rank : targetInfo.mIRanks)
    {
        auto it = std::find(counterPartRanks.begin(), counterPartRanks.end(), rank);
        TLLM_CHECK_WITH_INFO(it != counterPartRanks.end(), "Required rank %d not found in counterPartRanks", rank);
        indices.push_back(std::distance(counterPartRanks.begin(), it));
    }
    return indices;
}

/**
 * @brief Pick receive connections
 * @tparam CacheStateT Either kv_cache::CacheState or rnn_cache::RnnCacheState
 */
template <typename CacheStateT>
std::vector<size_t> pickRecvConnections(size_t numConnections, CacheStateT const& selfConfig, SizeType32 selfIdx,
    CacheStateT const& destConfig, std::vector<SizeType32> const& counterPartRanks)
{
    auto targetInfo = executor::kv_cache::targetIRanks(destConfig, selfConfig, selfIdx);
    if (targetInfo.mIRanks.empty())
    {
        return {};
    }

    auto baseIndices = pickSendConnections(numConnections, selfConfig, selfIdx, destConfig, counterPartRanks);

    if (targetInfo.mPeerDupHeadFactor <= 1)
    {
        return baseIndices;
    }

    int selfDPRank = selfConfig.getParallelConfig().mEnableAttentionDP ? selfConfig.getParallelConfig().mDPrank : 0;

    std::vector<size_t> ret;
    for (int i = 0; i < targetInfo.mDomainTPSize; i++)
    {
        if ((i % targetInfo.mPeerDupHeadFactor) == (selfDPRank % targetInfo.mPeerDupHeadFactor))
        {
            for (int j = 0; j < targetInfo.mDomainPPSize; j++)
            {
                size_t localIdx = (i * targetInfo.mDomainPPSize) + j;
                ret.push_back(baseIndices.at(localIdx));
            }
        }
    }
    return ret;
}
} // namespace cache_formatter_utils
} // namespace tensorrt_llm::batch_manager

namespace tensorrt_llm::batch_manager::kv_cache_manager
{
BlockRange getBlockRangeForSending(BaseKVCacheManager* cacheManager, LlmRequest const& llmRequest,
    BlockKey const& lastBlockKey, SizeType32 indexFromEnd, bool recvSideHasCP = false);

using DataContext = tensorrt_llm::executor::kv_cache::DataContext;
using Connection = tensorrt_llm::executor::kv_cache::Connection;
using SizeType32 = tensorrt_llm::runtime::SizeType32;
using BaseKVCacheManager = kv_cache_manager::BaseKVCacheManager;
using CacheTransBufferManager = kv_cache_manager::CacheTransBufferManager;
using BlockRange = kv_cache_manager::BlockRange;

BlockRange getBlockRangeForReceiving(BaseKVCacheManager* cacheManager, LlmRequest const& llmRequest,
    bool srcEnableBlockReuse, bool recvSideHasCP = false);

// Used to support the cache transmission with different layouts and different protocols.
class BaseCacheFormatter
{
public:
    using CacheState = executor::kv_cache::CacheState;

    /// @brief Format the cache data into bytes for sending.
    /// @param session The transfer session.
    virtual void format(tensorrt_llm::batch_manager::TransferSession& session) = 0;

    /// @brief Unformat the cache data from received bytes.
    /// @param session The transfer session.
    virtual void unformat(tensorrt_llm::batch_manager::TransferSession& session) = 0;

    /// @brief Determine whether the sender is applicable to the source and target.
    /// @param selfConfig Source data arrangement.
    /// @param destConfig Target data arrangement.
    /// @return Whether the sender is applicable to the source and target.
    [[nodiscard]] virtual bool inquireSupport(CacheState const& selfConfig, CacheState const& destConfig) const = 0;

    /// @brief Obtain the indies of the counterparts that need to be actually communicated with.
    /// @param selfConfig Source data arrangement.
    /// @param selfIdx The sequential index of the current executor process within the entire parallel group.
    /// @param destConfig Target data arrangement.
    /// @return The indies of the counterparts.
    [[nodiscard]] virtual std::vector<SizeType32> getCounterparts(
        CacheState const& selfConfig, SizeType32 selfIdx, CacheState const& destConfig) const
        = 0;

    [[nodiscard]] virtual BaseKVCacheManager* getCacheManager() const noexcept = 0;

    [[nodiscard]] virtual std::vector<size_t> pickRecvConnections(size_t numConnections, CacheState const& selfConfig,
        SizeType32 selfIdx, CacheState const& destConfig, std::vector<SizeType32> const& counterPartRanks) const
        = 0;

    /// @brief Destructor.
    virtual ~BaseCacheFormatter() = default;
};

// Simple cache block copy. Because it does not involve data splitting or merging, it performs best when the
// parallel topology is completely identical, making it the preferred method.
class CacheFormatter final : public BaseCacheFormatter
{
public:
    CacheFormatter(BaseKVCacheManager* cacheManager, CacheTransBufferManager* cacheTransBufferManager)
        : mCacheManager{cacheManager}
        , mCacheTransBufferManager{cacheTransBufferManager}
    {
        TLLM_CHECK(mCacheManager);
        TLLM_CHECK(mCacheTransBufferManager);
    }

    void format(tensorrt_llm::batch_manager::TransferSession& session) override;

    void unformat(tensorrt_llm::batch_manager::TransferSession& session) override;

    [[nodiscard]] bool inquireSupport(CacheState const& selfConfig, CacheState const& destConfig) const override;

    [[nodiscard]] std::vector<SizeType32> getCounterparts(
        CacheState const& selfConfig, SizeType32 selfIdx, CacheState const& destConfig) const override
    {
        return executor::kv_cache::targetIRanks(destConfig, selfConfig, selfIdx).mIRanks;
    }

    [[nodiscard]] BaseKVCacheManager* getCacheManager() const noexcept override
    {
        return mCacheManager;
    }

    [[nodiscard]] std::vector<size_t> pickRecvConnections(size_t numConnections, CacheState const& selfConfig,
        SizeType32 selfIdx, CacheState const& destConfig,
        std::vector<SizeType32> const& counterPartRanks) const override;

private:
    BaseKVCacheManager* mCacheManager;
    CacheTransBufferManager* mCacheTransBufferManager;
};

std::unique_ptr<BaseCacheFormatter> createCacheFormatter(BaseKVCacheManager* cacheManager,
    std::vector<CacheTransBufferManager*> const& cacheTransBufferManagers, bool isMLA = false);

} // namespace tensorrt_llm::batch_manager::kv_cache_manager
