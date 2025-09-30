
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

// we have blockNums Block, which is 3D  [PPs,TPs,(BlockIDs in one rank) tokens/tokens_per_block]

// input [PPs,TPs, BlockS] Block
// output [Blocks]Block. but each block has same tokens_per_block. so we can ignore tokens_per_block

#pragma once

#include "tensorrt_llm/executor/dataTransceiverState.h"
#include "tensorrt_llm/runtime/bufferManager.h"
#include "tensorrt_llm/runtime/iTensor.h"

#include <NvInferRuntimeBase.h>

namespace tensorrt_llm::executor::kv_cache
{

struct TargetRanksInfo
{
    int mDomainPPSize;
    int mDomainTPSize;
    int mDomainCPSize;
    std::vector<int> mIRanks;
    int mDupHeadFactor;
    int mPeerDupHeadFactor;

    // the size of the vector is equal to the mDomainPPSize. the value of the vector is the layer num should be fetched
    // from each target PP rank in domain PP.
    std::vector<int> mPeerAttentionLayerNumInDomainPP;

    int getPeerPPDomainLayerNum(int targetRankIdx) const
    {
        int ppDomainRankIdx = targetRankIdx % mDomainPPSize;
        return mPeerAttentionLayerNumInDomainPP.at(ppDomainRankIdx);
    }
};

TargetRanksInfo targetIRanks(
    kv_cache::CacheState const& peerCacheState, kv_cache::CacheState const& selfCacheState, int selfRank);

TargetRanksInfo TargetRanksInfoForDP(
    kv_cache::CacheState const& peerCacheState, kv_cache::CacheState const& selfCacheState, int selfRank);

void concatKVCacheDispatch(runtime::ITensor::SharedPtr* inputBlocks, int inputBlockNum,
    std::vector<int> const& inputRanks, kv_cache::CacheState const& peerCacheState,
    runtime::ITensor::SharedPtr* outputBlocks, int outputBlockNum, int selfRank,
    kv_cache::CacheState const& selfCacheState, runtime::BufferManager const& bufferManager);

nvinfer1::Dims makeShapeFromCacheState(kv_cache::CacheState const& cacheState);

void splitKVCacheDispatch(std::map<SizeType32, std::vector<runtime::ITensor::SharedPtr>> const& kVCacheBlocksPerWindow,
    std::vector<runtime::ITensor::SharedPtr>& ouputSplitBlocks, kv_cache::CacheState const& peerCacheState,
    kv_cache::CacheState const& selfCacheState, int selfIdx, runtime::BufferManager const& bufferManager);

void concatKvCacheV2Dispatch(std::vector<runtime::ITensor::SharedPtr> const& inputSplitBlocksPerWindow,
    std::map<SizeType32, std::vector<runtime::ITensor::SharedPtr>>& outputKvCacheBlocksPerWindow,
    kv_cache::CacheState const& peerCacheState, kv_cache::CacheState const& selfCacheState, int selfIdx,
    runtime::BufferManager const& bufferManager);

} // namespace tensorrt_llm::executor::kv_cache
