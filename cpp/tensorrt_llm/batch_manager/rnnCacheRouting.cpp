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

#include "tensorrt_llm/batch_manager/rnnCacheRouting.h"
#include "tensorrt_llm/batch_manager/cacheFormatterRouting.h"
#include "tensorrt_llm/common/logger.h"

namespace tensorrt_llm::batch_manager::rnn_cache_routing
{

bool inquireSupport(CacheState const& selfConfig, CacheState const& destConfig)
{
    if (!selfConfig.hasRnnConfig() || !destConfig.hasRnnConfig())
    {
        return false;
    }

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

    auto const& selfModel = selfConfig.getRnnModelConfig();
    auto const& destModel = destConfig.getRnnModelConfig();

    if (selfModel.mDState != destModel.mDState || selfModel.mHeadDim != destModel.mHeadDim
        || selfModel.mDConv != destModel.mDConv || selfModel.mNGroups != destModel.mNGroups
        || selfModel.mNumLayers != destModel.mNumLayers)
    {
        TLLM_LOG_WARNING("RnnCacheFormatter::inquireSupport: model config mismatch");
        return false;
    }

    auto const& selfParallel = selfConfig.getParallelConfig();
    auto const& destParallel = destConfig.getParallelConfig();

    if (selfParallel.mContextParallelism != 1 || destParallel.mContextParallelism != 1)
    {
        TLLM_LOG_WARNING("RnnCacheFormatter::inquireSupport: RNN only supports CP=1 (selfCP=%d, destCP=%d)",
            selfParallel.mContextParallelism, destParallel.mContextParallelism);
        return false;
    }

    return true;
}

std::vector<SizeType32> getCounterparts(CacheState const& selfConfig, SizeType32 selfIdx, CacheState const& destConfig)
{
    auto targetInfo = executor::kv_cache::targetIRanksForRnn(destConfig, selfConfig, selfIdx);
    return targetInfo.mIRanks;
}

std::pair<std::vector<size_t>, std::vector<size_t>> pickRecvConnections(size_t numConnections,
    CacheState const& selfConfig, SizeType32 selfIdx, CacheState const& destConfig,
    std::vector<SizeType32> const& counterPartRanks)
{
    auto targetInfo = executor::kv_cache::targetIRanksForRnn(destConfig, selfConfig, selfIdx);
    return cache_formatter_utils::pickRecvConnections(
        numConnections, selfConfig, selfIdx, destConfig, counterPartRanks, targetInfo);
}

} // namespace tensorrt_llm::batch_manager::rnn_cache_routing
