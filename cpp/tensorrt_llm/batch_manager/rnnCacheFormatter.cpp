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

#include "tensorrt_llm/batch_manager/rnnCacheFormatter.h"
#include "tensorrt_llm/batch_manager/dataTransceiver.h"
#include "tensorrt_llm/batch_manager/rnnStateManager.h"
#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/executor/cache_transmission/cacheSplitConcat.h"

#include <algorithm>

namespace tensorrt_llm::batch_manager
{

RnnCacheFormatter::RnnCacheFormatter(rnn_state_manager::RnnStateManager* rnnStateManager,
    rnn_state_manager::RnnCacheTransBufferManager* rnnCacheTransBufferManager)
    : mRnnStateManager{rnnStateManager}
    , mRnnCacheTransBufferManager{rnnCacheTransBufferManager}
{
    TLLM_CHECK(mRnnStateManager != nullptr);
    TLLM_CHECK(mRnnCacheTransBufferManager != nullptr);
}

void RnnCacheFormatter::format(TransferSession& session)
{
    // TODO: Implement RNN state formatting for sending
    TLLM_THROW("RnnCacheFormatter::format not yet implemented");
}

void RnnCacheFormatter::unformat(TransferSession& session)
{
    // TODO: Implement RNN state unformatting for receiving
    TLLM_THROW("RnnCacheFormatter::unformat not yet implemented");
}

bool RnnCacheFormatter::inquireSupport(RnnCacheState const& selfConfig, RnnCacheState const& destConfig) const
{
    if (selfConfig.getConvStateDataType() != destConfig.getConvStateDataType())
    {
        TLLM_LOG_WARNING("RnnCacheFormatter::inquireSupport: conv state data type mismatch (self=%d, dest=%d)",
            static_cast<int>(selfConfig.getConvStateDataType()), static_cast<int>(destConfig.getConvStateDataType()));
        return false;
    }

    if (selfConfig.getSsmStateDataType() != destConfig.getSsmStateDataType())
    {
        TLLM_LOG_WARNING("RnnCacheFormatter::inquireSupport: SSM state data type mismatch (self=%d, dest=%d)",
            static_cast<int>(selfConfig.getSsmStateDataType()), static_cast<int>(destConfig.getSsmStateDataType()));
        return false;
    }

    auto const& selfModel = selfConfig.getModelConfig();
    auto const& destModel = destConfig.getModelConfig();

    if (selfModel.mDState != destModel.mDState || selfModel.mHeadDim != destModel.mHeadDim
        || selfModel.mDConv != destModel.mDConv || selfModel.mNGroups != destModel.mNGroups
        || selfModel.mNumLayers != destModel.mNumLayers)
    {
        TLLM_LOG_WARNING("RnnCacheFormatter::inquireSupport: model config mismatch");
        return false;
    }

    auto const& selfParallel = selfConfig.getParallelConfig();
    auto const& destParallel = destConfig.getParallelConfig();

    // Require same TP for now (different TP would require state resharding)
    // TODO
    if (selfParallel.mTensorParallelism != destParallel.mTensorParallelism)
    {
        TLLM_LOG_WARNING("RnnCacheFormatter::inquireSupport: different TP not yet supported (self=%d, dest=%d)",
            selfParallel.mTensorParallelism, destParallel.mTensorParallelism);
        return false;
    }

    return true;
}

std::vector<RnnCacheFormatter::SizeType32> RnnCacheFormatter::getCounterparts(
    RnnCacheState const& selfConfig, SizeType32 selfIdx, RnnCacheState const& destConfig) const
{
    auto targetInfo = rnnTargetIRanks(destConfig, selfConfig, selfIdx);
    return targetInfo.mIRanks;
}

std::vector<size_t> RnnCacheFormatter::pickRecvConnections(
    size_t numConnections, RnnCacheState const& selfConfig, SizeType32 selfIdx, RnnCacheState const& destConfig) const
{
    // TODO: Implement connection selection for RNN cache transfer
    TLLM_THROW("RnnCacheFormatter::pickRecvConnections not yet implemented");
    return {};
}

executor::kv_cache::TargetRanksInfo rnnTargetIRanks(executor::rnn_cache::RnnCacheState const& peerState,
    executor::rnn_cache::RnnCacheState const& selfState, int selfRank)
{

    auto const& peerParConfig = peerState.getParallelConfig();
    auto const& selfParConfig = selfState.getParallelConfig();

    auto const peerPPNum = peerParConfig.mPipelineParallelism;
    auto const selfPPNum = selfParConfig.mPipelineParallelism;
    auto const peerTPNum = peerParConfig.mTensorParallelism;
    auto const selfTPNum = selfParConfig.mTensorParallelism;

    // We require same TP for RNN (checked in inquireSupport)
    // TODO
    TLLM_CHECK(selfTPNum == peerTPNum);

    // Compute self ranks (no CP for RNN)
    auto const selfTPRank = selfRank % selfTPNum;
    auto const selfPPRank = selfRank / selfTPNum;

    // Get layer distribution per PP rank
    auto const& peerNumLayerPerPP = peerParConfig.mRnnLayerNumPerPP;
    auto const& selfNumLayerPerPP = selfParConfig.mRnnLayerNumPerPP;
    TLLM_CHECK(peerNumLayerPerPP.size() == static_cast<size_t>(peerPPNum));
    TLLM_CHECK(selfNumLayerPerPP.size() == static_cast<size_t>(selfPPNum));

    // Find peer PP ranks that have overlapping layers
    int selfStartLayerId = 0;
    for (int ppRank = 0; ppRank < selfPPRank; ppRank++)
    {
        selfStartLayerId += selfNumLayerPerPP[ppRank];
    }
    int selfEndLayerId = selfStartLayerId + selfNumLayerPerPP[selfPPRank];

    std::vector<int> targetPeerPPRanks;
    std::vector<int> targetPeerPPLayerNum;
    int prePeerPPLayerId = 0;

    for (int ppRank = 0; ppRank < peerPPNum; ppRank++)
    {
        int peerPPStartLayerId = prePeerPPLayerId;
        int peerPPEndLayerId = peerPPStartLayerId + peerNumLayerPerPP[ppRank];
        prePeerPPLayerId += peerNumLayerPerPP[ppRank];

        // Check if layer ranges overlap
        if (selfStartLayerId < peerPPEndLayerId && selfEndLayerId > peerPPStartLayerId)
        {
            targetPeerPPRanks.push_back(ppRank);
            int layerNumInDomainPP
                = std::min(peerPPEndLayerId, selfEndLayerId) - std::max(peerPPStartLayerId, selfStartLayerId);
            targetPeerPPLayerNum.push_back(layerNumInDomainPP);
        }
    }

    int mDomainPPSize = static_cast<int>(targetPeerPPRanks.size());
    TLLM_CHECK(mDomainPPSize > 0);

    // Rank formula: ppRank * tpNum + tpRank
    std::vector<int> retRanks;
    for (int ppRank : targetPeerPPRanks)
    {
        int irank = ppRank * peerTPNum + selfTPRank;
        retRanks.push_back(irank);
    }

    return executor::kv_cache::TargetRanksInfo{
        mDomainPPSize,                  // mDomainPPSize
        1,                              // mDomainTPSize (1:1 since same TP)
        1,                              // mDomainCPSize (no CP for RNN)
        std::move(retRanks),            // mIRanks
        1,                              // mDupHeadFactor (no head duplication)
        1,                              // mPeerDupHeadFactor
        std::move(targetPeerPPLayerNum) // mPeerAttentionLayerNumInDomainPP
    };
}

} // namespace tensorrt_llm::batch_manager
