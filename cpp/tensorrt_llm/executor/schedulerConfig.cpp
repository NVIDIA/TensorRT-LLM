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

#include "tensorrt_llm/executor/executor.h"

namespace tensorrt_llm::executor
{

namespace
{

bool dynamicBatchConfigsEqual(
    std::optional<DynamicBatchConfig> const& lhs, std::optional<DynamicBatchConfig> const& rhs)
{
    if (lhs.has_value() != rhs.has_value())
    {
        return false;
    }
    if (!lhs.has_value())
    {
        return true;
    }

    return lhs->getEnableBatchSizeTuning() == rhs->getEnableBatchSizeTuning()
        && lhs->getEnableMaxNumTokensTuning() == rhs->getEnableMaxNumTokensTuning()
        && lhs->getDynamicBatchMovingAverageWindow() == rhs->getDynamicBatchMovingAverageWindow()
        && lhs->getBatchSizeTable() == rhs->getBatchSizeTable();
}

} // namespace

SchedulerConfig::SchedulerConfig(CapacitySchedulerPolicy capacitySchedulerPolicy,
    std::optional<ContextChunkingPolicy> contextChunkingPolicy, std::optional<DynamicBatchConfig> dynamicBatchConfig,
    bool enablePrefixAwareScheduling)
    : mCapacitySchedulerPolicy(capacitySchedulerPolicy)
    , mContextChunkingPolicy(std::move(contextChunkingPolicy))
    , mDynamicBatchConfig(std::move(dynamicBatchConfig))
    , mEnablePrefixAwareScheduling(enablePrefixAwareScheduling)
{
}

bool SchedulerConfig::operator==(SchedulerConfig const& other) const
{
    return mCapacitySchedulerPolicy == other.mCapacitySchedulerPolicy
        && mContextChunkingPolicy == other.mContextChunkingPolicy
        && dynamicBatchConfigsEqual(mDynamicBatchConfig, other.mDynamicBatchConfig)
        && mEnablePrefixAwareScheduling == other.mEnablePrefixAwareScheduling;
}

[[nodiscard]] CapacitySchedulerPolicy SchedulerConfig::getCapacitySchedulerPolicy() const
{
    return mCapacitySchedulerPolicy;
}

[[nodiscard]] std::optional<ContextChunkingPolicy> SchedulerConfig::getContextChunkingPolicy() const
{
    return mContextChunkingPolicy;
}

[[nodiscard]] std::optional<DynamicBatchConfig> SchedulerConfig::getDynamicBatchConfig() const
{
    return mDynamicBatchConfig;
}

[[nodiscard]] bool SchedulerConfig::getEnablePrefixAwareScheduling() const
{
    return mEnablePrefixAwareScheduling;
}

} // namespace tensorrt_llm::executor
