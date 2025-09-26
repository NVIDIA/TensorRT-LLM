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

#include "tensorrt_llm/executor/executor.h"
#include "tensorrt_llm/executor/types.h"

#include <optional>

namespace tensorrt_llm::executor
{
LoraConfig::LoraConfig(IdType taskId, std::optional<Tensor> weights, std::optional<Tensor> config)
    : mTaskId(taskId)
    , mWeights(std::move(weights))
    , mConfig(std::move(config))
{
    if (mConfig.has_value())
    {
        SizeType32 constexpr expectedConfigDims = 2;
        TLLM_CHECK_WITH_INFO(
            mConfig.value().getShape().size() == expectedConfigDims, "Expected config tensor to have 2 dimensions");
        TLLM_CHECK_WITH_INFO(mConfig.value().getMemoryType() != MemoryType::kGPU
                && mConfig.value().getMemoryType() != MemoryType::kUNKNOWN,
            "Expected lora config to be in CPU memory");
        TLLM_CHECK_WITH_INFO(
            mConfig.value().getDataType() == DataType::kINT32, "Expected lora config tensor to have type kINT32");
    }
    if (mWeights.has_value())
    {
        SizeType32 constexpr expectedWeightsDims = 2;
        TLLM_CHECK_WITH_INFO(
            mConfig.has_value(), "Request for LoRA inference with lora weights must also have lora config");

        TLLM_CHECK_WITH_INFO(
            mWeights.value().getShape().size() == expectedWeightsDims, "Expected weights tensor to have 2 dimensions");

        TLLM_CHECK_WITH_INFO(mWeights.value().getMemoryType() != MemoryType::kGPU
                && mWeights.value().getMemoryType() != MemoryType::kUNKNOWN,
            "Expected lora weights to be in CPU memory");

        TLLM_CHECK_WITH_INFO(mConfig.value().getShape()[0] == mWeights.value().getShape()[0],
            "Expected dim 0 of lora weights and lora config to have the same size");
    }
}

IdType LoraConfig::getTaskId() const
{
    return mTaskId;
}

std::optional<Tensor> LoraConfig::getWeights() const
{
    return mWeights;
}

std::optional<Tensor> LoraConfig::getConfig() const
{
    return mConfig;
}

} // namespace tensorrt_llm::executor
