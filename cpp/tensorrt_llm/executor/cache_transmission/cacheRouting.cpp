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

#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/executor/cache_transmission/cacheSplitConcat.h"
#include "tensorrt_llm/runtime/utils/mpiUtils.h"

#include <algorithm>
#include <numeric>
#include <sstream>

namespace tensorrt_llm::executor::kv_cache
{

namespace
{
inline bool isPowerOfTwo(int n)
{
    return n > 0 && (n & (n - 1)) == 0;
}
} // namespace

int getBlockNumAccountingForCP(int cpRank, int cpSize, int numTotalBlocks)
{
    TLLM_CHECK(cpRank >= 0 && cpRank < cpSize);
    if (cpSize == 1)
    {
        return numTotalBlocks;
    }
    // Blocks are distributed among CP ranks in a round-robin fashion. When the number of blocks is not
    // divisible by cpSize, the lowest-indexed CP ranks each receive one extra block.
    int numBlocksCurrRank = numTotalBlocks / cpSize;
    if (numTotalBlocks % cpSize > cpRank)
    {
        numBlocksCurrRank++;
    }
    return numBlocksCurrRank;
}

// inputBlockNums: [outputBlockNum, inputRanks.size]
// [PP, TP]
// Core implementation for computing target ranks info for data parallelism.
// Takes explicit layer vectors and head counts so it can be reused for both KV and RNN cache.
TargetRanksInfo TargetRanksInfoForDPImpl(kv_cache::CacheState const& peerCacheState,
    kv_cache::CacheState const& selfCacheState, int selfRank, std::vector<SizeType32> const& peerNumLayerPerPP,
    std::vector<SizeType32> const& selfNumLayerPerPP, SizeType32 peerNbHeadsPerLayer, SizeType32 selfNbHeadsPerLayer)
{
    auto const& peerParConfig = peerCacheState.getParallelConfig();
    auto const& selfParConfig = selfCacheState.getParallelConfig();

    auto const peerPPNum = peerParConfig.mPipelineParallelism;
    auto const selfPPNum = selfParConfig.mPipelineParallelism;
    auto const peerTPNum = peerParConfig.mTensorParallelism;
    auto const selfTPNum = selfParConfig.mTensorParallelism;
    auto const peerCPNum = peerParConfig.mContextParallelism;
    auto const selfCPNum = selfParConfig.mContextParallelism;

    auto const selfCPRank = selfRank % selfCPNum;
    auto const selfTPRank = (selfRank % (selfTPNum * selfCPNum)) / selfCPNum;
    auto const selfPPRank = selfRank / (selfTPNum * selfCPNum);

    int peerPPRankStart = 0;
    int mDomainPPSize = 1;
    int peerPPRankEnd = 0;

    if (selfNumLayerPerPP[selfPPRank] == 0)
    {
        return TargetRanksInfo{.mDomainPPSize = 0,
            .mDomainTPSize = 0,
            .mDomainCPSize = 0,
            .mIRanks = {}, // caller should handle this case. No transfer needed.
            .mDupHeadFactor = 1,
            .mPeerDupHeadFactor = 1,
            .mPeerLayerNumInDomainPP = {}};
    }

    TLLM_CHECK(peerNumLayerPerPP.size() == static_cast<size_t>(peerPPNum));
    TLLM_CHECK(selfNumLayerPerPP.size() == static_cast<size_t>(selfPPNum));
    int selfStartLayerId = 0;
    // global start layer id for selfPPrank, which is the sum of the layer num of the previous PP ranks.
    // compute the target PP ranks and layer num need to be fetched from each target PP rank, according to [global start
    // layer id, global end layer id)

    for (int ppRank = 0; ppRank < selfPPRank; ppRank++)
    {
        selfStartLayerId += selfNumLayerPerPP[ppRank];
    }
    int selfEndLayerId = selfStartLayerId + selfNumLayerPerPP[selfPPRank];

    int prePeerPPLayerId = 0;
    std::vector<int> targetPeerPPRanks;
    std::vector<int> targetPeerPPLayerNum;
    for (int ppRank = 0; ppRank < peerPPNum; ppRank++)
    {
        int peerPPStartLayerId = prePeerPPLayerId;
        int peerPPEndLayerId = peerPPStartLayerId + peerNumLayerPerPP[ppRank];

        prePeerPPLayerId += peerNumLayerPerPP[ppRank];

        if (selfStartLayerId < peerPPEndLayerId && selfEndLayerId > peerPPStartLayerId)
        {
            targetPeerPPRanks.push_back(ppRank);
            int layerNumInDomainPP
                = std::min(peerPPEndLayerId, selfEndLayerId) - std::max(peerPPStartLayerId, selfStartLayerId);
            targetPeerPPLayerNum.push_back(layerNumInDomainPP);
        }
    }
    mDomainPPSize = static_cast<int>(targetPeerPPRanks.size());
    peerPPRankStart = targetPeerPPRanks.front();
    peerPPRankEnd = peerPPRankStart + mDomainPPSize;
    TLLM_CHECK(targetPeerPPLayerNum.size() == static_cast<size_t>(mDomainPPSize));

    int targetPeerPpLayerNumSum = std::accumulate(targetPeerPPLayerNum.begin(), targetPeerPPLayerNum.end(), 0);
    TLLM_CHECK(targetPeerPpLayerNumSum == selfNumLayerPerPP[selfPPRank]);

    TLLM_LOG_DEBUG(mpi::MpiComm::world().getRank(),
        "selfPPRank:%d,selfPPNum:%d,peerPPNum:%d,selfTPNum:%d,peerTPNum:%d,peerPPRankStart:%d,peerPPRankEnd:%d",
        selfPPRank, selfPPNum, peerPPNum, selfTPNum, peerTPNum, peerPPRankStart, peerPPRankEnd);
    int peerTPRankStart = 0;
    int mDomainTPSize = 1;
    int peerTPRankEnd = 0;

    int const peerDpRank = peerParConfig.mEnableAttentionDP ? peerParConfig.mDPrank : 0;
    int const selfTPSizePerDPGroup = selfParConfig.mEnableAttentionDP ? selfTPNum / selfParConfig.mDPsize : selfTPNum;
    int const peerTPSizePerDPGroup = peerParConfig.mEnableAttentionDP ? peerTPNum / peerParConfig.mDPsize : peerTPNum;

    int const selfTPrankInDPGroup = selfTPRank % selfTPSizePerDPGroup;
    for (auto val : {peerTPSizePerDPGroup, selfTPSizePerDPGroup})
    {
        TLLM_CHECK(isPowerOfTwo(val));
    }
    if (selfTPSizePerDPGroup <= peerTPSizePerDPGroup)
    {
        mDomainTPSize = peerTPSizePerDPGroup / selfTPSizePerDPGroup;
        peerTPRankStart = selfTPrankInDPGroup * mDomainTPSize + peerDpRank * peerTPSizePerDPGroup;
        peerTPRankEnd = peerTPRankStart + mDomainTPSize;
    }
    else
    {
        peerTPRankStart
            = selfTPrankInDPGroup / (selfTPSizePerDPGroup / peerTPSizePerDPGroup) + peerDpRank * peerTPSizePerDPGroup;
        peerTPRankEnd = peerTPRankStart + mDomainTPSize;
    }

    int mDomainCPSize = 1;
    int peerCPRankStart = 0;
    int peerCPRankEnd = 0;
    for (auto val : {peerCPNum, selfCPNum})
    {
        TLLM_CHECK(isPowerOfTwo(val));
    }
    if (selfCPNum <= peerCPNum)
    {
        mDomainCPSize = peerCPNum / selfCPNum;
        peerCPRankStart = selfCPRank * mDomainCPSize;
        peerCPRankEnd = (selfCPRank + 1) * mDomainCPSize;
    }
    else
    {
        peerCPRankStart = selfCPRank / (selfCPNum / peerCPNum);
        peerCPRankEnd = peerCPRankStart + mDomainCPSize;
    }

    std::vector<int> retRanks;
    for (int i = peerCPRankStart; i < peerCPRankEnd; i++)
    {
        for (int j = peerTPRankStart; j < peerTPRankEnd; j++)
        {
            for (int k = peerPPRankStart; k < peerPPRankEnd; k++)
            {
                // Rank formula: ppRank * (tpNum * cpNum) + tpRank * cpNum + cpRank.
                int irank = (k * peerTPNum * peerCPNum) + (j * peerCPNum) + i;
                retRanks.push_back(irank);
            }
        }
    }

    int mDupHeadFactor = 1;
    int mPeerDupHeadFactor = 1;

    if (selfNbHeadsPerLayer * selfTPSizePerDPGroup > peerNbHeadsPerLayer * peerTPSizePerDPGroup)
    {
        mDupHeadFactor = (selfNbHeadsPerLayer * selfTPSizePerDPGroup) / (peerNbHeadsPerLayer * peerTPSizePerDPGroup);
    }
    if (peerNbHeadsPerLayer * peerTPSizePerDPGroup > selfNbHeadsPerLayer * selfTPSizePerDPGroup)
    {
        mPeerDupHeadFactor
            = (peerNbHeadsPerLayer * peerTPSizePerDPGroup) / (selfNbHeadsPerLayer * selfTPSizePerDPGroup);
    }

    TLLM_LOG_DEBUG(mpi::MpiComm::world().getRank(),
        "mDomainPPSize:%d, mDomainTPSize:%d, mDupHeadFactor:%d, mPeerDupHeadFactor:%d, selfPPRank:%d, selfPPNum:%d, "
        "peerPPNum:%d, selfTPNum:%d, peerTPNum:%d, selfTPSizePerDPGroup:%d, peerTPSizePerDPGroup:%d, "
        "selfNbHeadsPerLayer:%d, peerNbHeadsPerLayer:%d, selfTPrankInDPGroup:%d, peerDpRank:%d, selfRank:%d",
        mDomainPPSize, mDomainTPSize, mDupHeadFactor, mPeerDupHeadFactor, selfPPRank, selfPPNum, peerPPNum, selfTPNum,
        peerTPNum, selfTPSizePerDPGroup, peerTPSizePerDPGroup, selfNbHeadsPerLayer, peerNbHeadsPerLayer,
        selfTPrankInDPGroup, peerDpRank, selfRank);

    auto vector_to_string = [](std::vector<int> const& vec)
    {
        std::stringstream ss;
        for (auto val : vec)
        {
            ss << val << ",";
        }
        return ss.str();
    };
    TLLM_LOG_DEBUG(mpi::MpiComm::world().getRank(), "retRanks:%s , targetPeerPPLayerNum:%s",
        vector_to_string(retRanks).c_str(), vector_to_string(targetPeerPPLayerNum).c_str());
    return {mDomainPPSize, mDomainTPSize, mDomainCPSize, std::move(retRanks), mDupHeadFactor, mPeerDupHeadFactor,
        std::move(targetPeerPPLayerNum)};
}

TargetRanksInfo targetIRanks(
    kv_cache::CacheState const& peerCacheState, kv_cache::CacheState const& selfCacheState, int selfRank)
{
    auto const& peerHeads = peerCacheState.getModelConfig().mNbKvHeadsPerLayer;
    auto const& selfHeads = selfCacheState.getModelConfig().mNbKvHeadsPerLayer;
    return TargetRanksInfoForDPImpl(peerCacheState, selfCacheState, selfRank,
        peerCacheState.getParallelConfig().mAttentionLayerNumPerPP,
        selfCacheState.getParallelConfig().mAttentionLayerNumPerPP, peerHeads.empty() ? 0 : peerHeads[0],
        selfHeads.empty() ? 0 : selfHeads[0]);
}

TargetRanksInfo targetIRanksForRnn(
    kv_cache::CacheState const& peerCacheState, kv_cache::CacheState const& selfCacheState, int selfRank)
{
    auto targetInfo = TargetRanksInfoForDPImpl(peerCacheState, selfCacheState, selfRank,
        peerCacheState.getRnnCacheState().mLayerNumPerPP, selfCacheState.getRnnCacheState().mLayerNumPerPP,
        peerCacheState.getRnnModelConfig().mNumHeads, selfCacheState.getRnnModelConfig().mNumHeads);
    targetInfo.mDupHeadFactor = 1;
    targetInfo.mPeerDupHeadFactor = 1; // RNN cache does not have head duplication.
    return targetInfo;
}

TargetRanksInfo targetIRanksForIndexerKCache(
    kv_cache::CacheState const& peerCacheState, kv_cache::CacheState const& selfCacheState, int selfRank)
{
    auto targetInfo = targetIRanks(peerCacheState, selfCacheState, selfRank);
    if (targetInfo.mIRanks.empty())
    {
        return targetInfo;
    }
    auto const& peerIndexerPerPP = peerCacheState.getIndexerLayerNumPerPP();
    auto const& selfIndexerPerPP = selfCacheState.getIndexerLayerNumPerPP();
    auto const& peerParConfig = peerCacheState.getParallelConfig();
    auto const& selfParConfig = selfCacheState.getParallelConfig();
    TLLM_CHECK(static_cast<SizeType32>(peerIndexerPerPP.size()) == peerParConfig.mPipelineParallelism);
    TLLM_CHECK(static_cast<SizeType32>(selfIndexerPerPP.size()) == selfParConfig.mPipelineParallelism);

    auto const selfPPRank = selfRank / (selfParConfig.mTensorParallelism * selfParConfig.mContextParallelism);
    int64_t selfStart = 0;
    for (int ppRank = 0; ppRank < selfPPRank; ppRank++)
    {
        selfStart += selfIndexerPerPP[ppRank];
    }
    int64_t const selfEnd = selfStart + selfIndexerPerPP[selfPPRank];

    // First peer PP rank of the attention domain (mIRanks is ordered CP-major, then TP, then
    // PP ascending from the domain start).
    auto const peerPPRankStart
        = targetInfo.mIRanks.front() / (peerParConfig.mTensorParallelism * peerParConfig.mContextParallelism);
    int64_t peerStart = 0;
    for (int ppRank = 0; ppRank < peerPPRankStart; ppRank++)
    {
        peerStart += peerIndexerPerPP[ppRank];
    }

    int64_t peerLayerNumSum = 0;
    for (int i = 0; i < targetInfo.mDomainPPSize; i++)
    {
        int64_t const peerEnd = peerStart + peerIndexerPerPP.at(peerPPRankStart + i);
        auto const overlap = std::max<int64_t>(0, std::min(peerEnd, selfEnd) - std::max(peerStart, selfStart));
        targetInfo.mPeerLayerNumInDomainPP.at(i) = static_cast<int>(overlap);
        peerLayerNumSum += overlap;
        peerStart = peerEnd;
    }
    TLLM_CHECK_WITH_INFO(peerLayerNumSum == selfIndexerPerPP[selfPPRank],
        "Indexer K cache layer counts are inconsistent between the two sides: the attention-domain peers cover "
        "%ld indexer layers but this rank owns %d. Both sides must derive the same per-layer indexer schedule.",
        static_cast<long>(peerLayerNumSum), selfIndexerPerPP[selfPPRank]);
    return targetInfo;
}

} // namespace tensorrt_llm::executor::kv_cache
