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

#include "tensorrt_llm/executor/executor.h"
#include "tensorrt_llm/executor/types.h"

#include <deque>

namespace tensorrt_llm::executor
{

/// @brief A class that maintains runtime input and output length statistics and computes runtime dynamic batch size.
class DynamicBatchTuner
{
public:
    explicit DynamicBatchTuner(DynamicBatchConfig const& config);

    /// @brief Check if dynamic batch size tuning is enabled.
    [[nodiscard]] bool isBatchSizeTuningEnabled() const
    {
        return mEnableBatchSizeTuning;
    }

    /// @brief Check if max num tokens tuning is enabled.
    [[nodiscard]] bool isMaxNumTokensTuningEnabled() const
    {
        return mEnableMaxNumTokensTuning;
    }

    /// @brief Update current stats given the input and output length from a single request.
    void updateStats(SizeType32 inputLen, SizeType32 outputLen);

    /// @brief Get average input length.
    [[nodiscard]] double getAverageInputLength() const;

    /// @brief Get average output length.
    [[nodiscard]] double getAverageOutputLength() const;

    /// @brief Get the dynamic batch size based on the current statistics.
    [[nodiscard]] SizeType32 getRuntimeBatchSize(SizeType32 maxCapacityBatchSize) const;

    /// @brief Get the dynamic max num tokens based on the current statistics.
    [[nodiscard]] SizeType32 getRuntimeMaxNumTokens(SizeType32 runtimeBatchSize) const;

private:
    bool mEnableBatchSizeTuning = false;

    bool mEnableMaxNumTokensTuning = false;

    SizeType32 mDynamicBatchMovingAverageWindow = 0;

    std::vector<std::pair<SizeType32, SizeType32>> mBatchSizeTable;

    int64_t mInputLengthSum = 0;
    std::deque<SizeType32> mInputLengthStats;

    int64_t mOutputLengthSum = 0;
    std::deque<SizeType32> mOutputLengthStats;

    static SizeType32 const kBatchSizeFallbackGranularity = 512;
    static SizeType32 const kBatchSizeFallbackThreshold = 128;

    static double constexpr kMaxNumTokensRatioContextHeavy = 2.0;
    static double constexpr kMaxNumTokensRatioBalanced = 0.5;

    static SizeType32 const kMaxNumTokensThresholdContextHeavy = 8192;
    static SizeType32 const kMaxNumTokensThresholdBalanced = 4096;
    static SizeType32 const kMaxNumTokensThresholdGenHeavy = 2048;
};

} // namespace tensorrt_llm::executor
