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
#include "tensorrt_llm/executor/cache_transmission/cacheSplitConcat.h"

#include <algorithm>

namespace tensorrt_llm::batch_manager
{

CacheTransferLayer::CacheTransferLayer(executor::kv_cache::CacheState kvState,
    std::unique_ptr<BaseCacheFormatter> kvFormatter, std::optional<executor::rnn_cache::RnnCacheState> rnnState,
    std::unique_ptr<RnnCacheFormatter> rnnFormatter)
    : mKvState{std::move(kvState)}
    , mKvFormatter{std::move(kvFormatter)}
    , mRnnState{std::move(rnnState)}
    , mRnnFormatter{std::move(rnnFormatter)}
{
    TLLM_CHECK(mKvFormatter);
}

CacheTransferLayer::~CacheTransferLayer() = default;

CacheTransferLayer::CacheTransferLayer(CacheTransferLayer&&) noexcept = default;
CacheTransferLayer& CacheTransferLayer::operator=(CacheTransferLayer&&) noexcept = default;

void CacheTransferLayer::validateSupport(executor::DataTransceiverState const& peerState) const
{
    TLLM_CHECK_WITH_INFO(mKvFormatter->inquireSupport(mKvState, peerState.getCacheState().value()),
        "Disagg server does not currently support these cacheState, please check the cacheState of the context and "
        "gen executors");

    if (mRnnFormatter && mRnnState.has_value())
    {
        if (peerState.hasRnnCacheState())
        {
            TLLM_CHECK_WITH_INFO(mRnnFormatter->inquireSupport(mRnnState.value(), peerState.getRnnCacheState().value()),
                "Disagg server does not currently support these RNN state configurations, please check the RNN "
                "state of the context and gen executors");
        }
        else
        {
            TLLM_LOG_WARNING("Self has RNN state but peer does not. RNN transfer will be skipped.");
        }
    }
    else if (peerState.hasRnnCacheState() && (!mRnnFormatter || !mRnnState.has_value()))
    {
        TLLM_LOG_WARNING("Peer has RNN state but self does not. RNN transfer will be skipped.");
    }
}

std::vector<SizeType32> CacheTransferLayer::computeCounterparts(
    SizeType32 selfIdx, executor::DataTransceiverState const& peerState) const
{
    auto counterparts = executor::kv_cache::targetIRanks(peerState.getCacheState().value(), mKvState, selfIdx).mIRanks;

    // Add RNN counterparts that are not already in the KV set
    if (mRnnFormatter && mRnnState.has_value() && peerState.hasRnnCacheState())
    {
        auto rnnCounterparts
            = executor::kv_cache::targetIRanks(peerState.getRnnCacheState().value(), mRnnState.value(), selfIdx)
                  .mIRanks;
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

void CacheTransferLayer::populateSelfState(executor::DataTransceiverState& state) const
{
    if (mRnnState.has_value())
    {
        state.setRnnCacheState(mRnnState.value());
    }
}

executor::kv_cache::CacheState const& CacheTransferLayer::getKvState() const noexcept
{
    return mKvState;
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
