/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "cacheTransferLayer.h"

#include "tensorrt_llm/batch_manager/cacheFormatter.h"
#include "tensorrt_llm/batch_manager/rnnCacheFormatter.h"
#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/executor/cache_transmission/agent_utils/connection.h"
#include "tensorrt_llm/executor/cache_transmission/cacheSplitConcat.h"

#include <algorithm>

namespace tensorrt_llm::batch_manager
{

CacheTransferLayer::CacheTransferLayer(executor::kv_cache::CacheState cacheState,
    std::unique_ptr<BaseCacheFormatter> kvFormatter, std::unique_ptr<RnnCacheFormatter> rnnFormatter,
    bool const enableInflightCancel)
    : mCacheState{std::move(cacheState)}
    , mKvFormatter{std::move(kvFormatter)}
    , mRnnFormatter{std::move(rnnFormatter)}
    , mEnableInflightCancel{enableInflightCancel}
{
    TLLM_CHECK(mKvFormatter);
}

CacheTransferLayer::~CacheTransferLayer() = default;

CacheTransferLayer::CacheTransferLayer(CacheTransferLayer&&) noexcept = default;
CacheTransferLayer& CacheTransferLayer::operator=(CacheTransferLayer&&) noexcept = default;

void CacheTransferLayer::validateSupport(executor::DataTransceiverState const& peerState) const
{
    validateCacheSupport(peerState);
    auto const compatibility = getPeerProtocolCompatibility(peerState);
    TLLM_CHECK_WITH_INFO(
        compatibility.compatible, "Disaggregated cache-transfer protocol mismatch: %s", compatibility.reason.c_str());
}

void CacheTransferLayer::validateCacheSupport(executor::DataTransceiverState const& peerState) const
{
    TLLM_CHECK_WITH_INFO(mKvFormatter->inquireSupport(mCacheState, peerState.getCacheState().value()),
        "Disagg server does not currently support these cacheState, please check the cacheState of the context and "
        "gen executors");

    bool const selfHasRnn = mCacheState.hasRnnConfig();
    bool const peerHasRnn = peerState.getCacheState().value().hasRnnConfig();

    if (mRnnFormatter && selfHasRnn)
    {
        // Both slot-based (CppMambaCacheManager) and unified pool (CppMambaHybridCacheManager)
        // paths now use RnnCacheFormatter.
        if (peerHasRnn)
        {
            TLLM_CHECK_WITH_INFO(mRnnFormatter->inquireSupport(mCacheState, peerState.getCacheState().value()),
                "Disagg server does not currently support these RNN state configurations, please check the RNN "
                "state of the context and gen executors");
        }
        else
        {
            TLLM_LOG_WARNING("Self has RNN state but peer does not. RNN transfer will be skipped.");
        }
    }
    else if (!selfHasRnn && peerHasRnn)
    {
        TLLM_LOG_WARNING("Peer has RNN state but self does not. RNN transfer will be skipped.");
    }
}

executor::kv_cache::PeerProtocolCompatibility CacheTransferLayer::getPeerProtocolCompatibility(
    executor::DataTransceiverState const& peerState) const
{
    if (!peerState.getCommState().has_value() || !peerState.getCommState()->isAgentState())
    {
        return {true, false, false, "peer protocol negotiation is not applicable to this transport"};
    }

    std::vector<std::string> peerAgentNames;
    auto const& peerAgentStates = peerState.getCommState()->getAgentState();
    peerAgentNames.reserve(peerAgentStates.size());
    for (auto const& peerAgentState : peerAgentStates)
    {
        peerAgentNames.push_back(peerAgentState.mAgentName);
    }
    auto const localMode = mEnableInflightCancel ? executor::kv_cache::PeerCancellationMode::kEnabled
                                                 : executor::kv_cache::PeerCancellationMode::kBaseline;
    return executor::kv_cache::validatePeerProtocol(localMode, peerAgentNames);
}

std::vector<SizeType32> CacheTransferLayer::computeCounterparts(
    SizeType32 selfIdx, executor::DataTransceiverState const& peerState) const
{
    auto counterparts
        = executor::kv_cache::targetIRanks(peerState.getCacheState().value(), mCacheState, selfIdx).mIRanks;

    // Add RNN counterparts that are not already in the KV set
    if (mRnnFormatter && mCacheState.hasRnnConfig() && peerState.getCacheState().value().hasRnnConfig())
    {
        auto rnnCounterparts
            = executor::kv_cache::targetIRanksForRnn(peerState.getCacheState().value(), mCacheState, selfIdx).mIRanks;
        for (auto rank : rnnCounterparts)
        {
            if (std::find(counterparts.begin(), counterparts.end(), rank) == counterparts.end())
            {
                counterparts.push_back(rank);
            }
        }
    }

    return counterparts;
}

void CacheTransferLayer::format(TransferSession& session) const
{
    mKvFormatter->format(session);
    if (mRnnFormatter)
    {
        for (auto const* conn : session.getConnections())
        {
            if (conn != nullptr)
            {
                conn->activateBuffer(static_cast<uint8_t>(BufferKind::kRNN));
            }
        }
        mRnnFormatter->format(session);
    }
}

void CacheTransferLayer::unformat(TransferSession& session) const
{
    mKvFormatter->unformat(session);
    if (mRnnFormatter)
    {
        mRnnFormatter->unformat(session);
    }
}

void CacheTransferLayer::setRnnConfig(executor::kv_cache::CacheState::RnnModelConfig rnnModelConfig,
    std::vector<SizeType32> rnnLayerNumPerPP, nvinfer1::DataType convStateDataType, nvinfer1::DataType ssmStateDataType)
{
    mCacheState.setRnnConfig(
        std::move(rnnModelConfig), std::move(rnnLayerNumPerPP), convStateDataType, ssmStateDataType);
}

executor::kv_cache::CacheState const& CacheTransferLayer::getCacheState() const noexcept
{
    return mCacheState;
}

kv_cache_manager::BaseKVCacheManager* CacheTransferLayer::getCacheManager() const noexcept
{
    return mKvFormatter->getCacheManager();
}

BaseCacheFormatter* CacheTransferLayer::getKvFormatter() const noexcept
{
    return mKvFormatter.get();
}

bool CacheTransferLayer::isInflightCancelEnabled() const noexcept
{
    return mEnableInflightCancel;
}

} // namespace tensorrt_llm::batch_manager
