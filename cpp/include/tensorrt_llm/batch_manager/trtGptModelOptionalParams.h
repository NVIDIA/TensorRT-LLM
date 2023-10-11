/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "tensorrt_llm/runtime/common.h"

#include <optional>

namespace tensorrt_llm::batch_manager
{

class TrtGptModelOptionalParams
{
public:
    using SizeType = tensorrt_llm::runtime::SizeType;

    TrtGptModelOptionalParams()
        : mMaxNumSequences(std::nullopt)
        , mMaxTokensInPagedKvCache(std::nullopt)
        , mKvCacheFreeGpuMemFraction(std::nullopt)
        , mEnableTrtOverlap(std::nullopt)
    {
    }

    TrtGptModelOptionalParams(std::optional<SizeType> maxNumSequences, std::optional<SizeType> maxTokensInPagedKvCache,
        std::optional<float> kvCacheFreeGpuMemFraction, std::optional<bool> enableTrtOverlap)
        : mMaxNumSequences(maxNumSequences)
        , mMaxTokensInPagedKvCache(maxTokensInPagedKvCache)
        , mKvCacheFreeGpuMemFraction(kvCacheFreeGpuMemFraction)
        , mEnableTrtOverlap(enableTrtOverlap)
    {
    }

    [[nodiscard]] std::optional<SizeType> getMaxTokensInPagedKvCache() const
    {
        return mMaxTokensInPagedKvCache;
    }

    [[nodiscard]] std::optional<float> getKvCacheFreeGpuMemFraction() const
    {
        return mKvCacheFreeGpuMemFraction;
    }

    [[nodiscard]] std::optional<float> getMaxNumSequences() const
    {
        return mMaxNumSequences;
    }

    [[nodiscard]] std::optional<bool> getEnableTrtOverlap() const
    {
        return mEnableTrtOverlap;
    }

private:
    std::optional<SizeType> mMaxNumSequences;
    std::optional<SizeType> mMaxTokensInPagedKvCache;
    std::optional<float> mKvCacheFreeGpuMemFraction;
    std::optional<bool> mEnableTrtOverlap;
};

} // namespace tensorrt_llm::batch_manager
