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

#include "tensorrt_llm/executor/dynamicBatchTuner.h"
#include "tensorrt_llm/common/logger.h"
#include <cmath>

namespace
{
using namespace tensorrt_llm::executor;

void updateStats(SizeType32 value, std::deque<SizeType32>& stats, int64_t& sum, SizeType32 windowSize)
{
    while (static_cast<SizeType32>(stats.size()) >= windowSize)
    {
        sum -= stats.front();
        stats.pop_front();
    }
    stats.push_back(value);
    sum += value;
}
} // namespace

namespace tensorrt_llm::executor
{

DynamicBatchTuner::DynamicBatchTuner(DynamicBatchConfig const& config)
    : mEnableBatchSizeTuning(config.getEnableBatchSizeTuning())
    , mEnableMaxNumTokensTuning(config.getEnableMaxNumTokensTuning())
    , mDynamicBatchMovingAverageWindow(config.getDynamicBatchMovingAverageWindow())
    , mBatchSizeTable(config.getBatchSizeTable())
{
    TLLM_CHECK_WITH_INFO(!mBatchSizeTable.empty(), "Batch size table is empty.");
    for (size_t i = 1; i < mBatchSizeTable.size(); ++i)
    {
        TLLM_CHECK_WITH_INFO(mBatchSizeTable[i - 1].first < mBatchSizeTable[i].first,
            "Batch size table is not sorted in ascending order.");
    }
}

void DynamicBatchTuner::updateStats(SizeType32 inputLength, SizeType32 outputLength)
{
    ::updateStats(inputLength, mInputLengthStats, mInputLengthSum, mDynamicBatchMovingAverageWindow);
    ::updateStats(outputLength, mOutputLengthStats, mOutputLengthSum, mDynamicBatchMovingAverageWindow);
}

double DynamicBatchTuner::getAverageInputLength() const
{
    return mInputLengthStats.empty() ? 0 : static_cast<double>(mInputLengthSum) / mInputLengthStats.size();
}

double DynamicBatchTuner::getAverageOutputLength() const
{
    return mOutputLengthStats.empty() ? 0 : static_cast<double>(mOutputLengthSum) / mOutputLengthStats.size();
}

SizeType32 DynamicBatchTuner::getRuntimeBatchSize(SizeType32 maxCapacityBatchSize) const
{
    for (auto const& [batchSizeLimit, batchSize] : mBatchSizeTable)
    {
        if (maxCapacityBatchSize < batchSizeLimit)
        {
            return batchSize;
        }
    }
    SizeType32 threshold = maxCapacityBatchSize / kBatchSizeFallbackGranularity * kBatchSizeFallbackGranularity;
    if (maxCapacityBatchSize < (threshold + kBatchSizeFallbackThreshold))
    {
        return threshold;
    }
    return maxCapacityBatchSize;
}

SizeType32 DynamicBatchTuner::getRuntimeMaxNumTokens(SizeType32 maxRuntimeBatchSize) const
{
    // calculate max num token in fully overlapped case
    SizeType32 adjustedNumTokens
        = 1.0 * (maxRuntimeBatchSize * getAverageInputLength() / getAverageOutputLength() + maxRuntimeBatchSize);
    SizeType32 tokenThreshold;
    // context heavy (avg ISL/OSL > kMaxNumTokensRatioContextHeavy)
    if (getAverageInputLength() / getAverageOutputLength() > kMaxNumTokensRatioContextHeavy)
    {
        tokenThreshold = kMaxNumTokensThresholdContextHeavy;
    }
    // balanced case (kMaxNumTokensRatioBalanced < avg ISL/OSL < kMaxNumTokensRatioContextHeavy)
    else if (getAverageInputLength() / getAverageOutputLength() > kMaxNumTokensRatioBalanced)
    {
        tokenThreshold = kMaxNumTokensThresholdBalanced;
    }
    // gen heavy (avg ISL/OSL < kMaxNumTokensRatioBalanced)
    else
    {
        tokenThreshold = kMaxNumTokensThresholdGenHeavy;
    }
    // pad it to pow of 2 and max of this value and threshold.
    return (std::max(1 << int(ceil(log2(adjustedNumTokens))), tokenThreshold));
}

} // namespace tensorrt_llm::executor
