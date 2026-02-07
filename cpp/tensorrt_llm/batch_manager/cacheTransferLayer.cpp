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
    std::unique_ptr<BaseCacheFormatter> kvFormatter, std::unique_ptr<RnnCacheFormatter> rnnFormatter)
    : mCacheState{std::move(cacheState)}
    , mKvFormatter{std::move(kvFormatter)}
    , mRnnFormatter{std::move(rnnFormatter)}
{
    TLLM_CHECK(mKvFormatter);
}

CacheTransferLayer::~CacheTransferLayer() = default;

CacheTransferLayer::CacheTransferLayer(CacheTransferLayer&&) noexcept = default;
CacheTransferLayer& CacheTransferLayer::operator=(CacheTransferLayer&&) noexcept = default;

void CacheTransferLayer::validateSupport(executor::DataTransceiverState const& peerState) const
{
    TLLM_CHECK_WITH_INFO(mKvFormatter->inquireSupport(mCacheState, peerState.getCacheState().value()),
        "Disagg server does not currently support these cacheState, please check the cacheState of the context and "
        "gen executors");

    if (mRnnFormatter && mCacheState.hasRnnConfig())
    {
        if (peerState.getCacheState().value().hasRnnConfig())
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
    else if (peerState.getCacheState().value().hasRnnConfig() && (!mRnnFormatter || !mCacheState.hasRnnConfig()))
    {
        TLLM_LOG_WARNING("Peer has RNN state but self does not. RNN transfer will be skipped.");
    }
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
        // For NIXL agent connections, switch the active buffer index to the RNN buffer
        // before running the RNN formatter. The buffer descs are ordered [KV..., RNN],
        // so the RNN buffer is always the last one.
        auto const& sessionConnections = session.getConnections();
        for (auto const* conn : sessionConnections)
        {
            if (conn != nullptr)
            {
                auto const* agentConn = dynamic_cast<executor::kv_cache::AgentConnection const*>(conn);
                if (agentConn != nullptr && agentConn->getSenderBufferCount() > 1)
                {
                    size_t rnnBufferIdx = agentConn->getSenderBufferCount() - 1;
                    const_cast<executor::kv_cache::AgentConnection*>(agentConn)->setActiveSenderBufferIdx(rnnBufferIdx);
                }
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

} // namespace tensorrt_llm::batch_manager
