/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "tensorrt_llm/batch_manager/kvCacheConfig.h"
#include "tensorrt_llm/batch_manager/peftCacheManagerConfig.h"
#include "tensorrt_llm/executor/executor.h"
#include "tensorrt_llm/runtime/common.h"
#include "tensorrt_llm/runtime/decodingMode.h"
#include "tensorrt_llm/runtime/medusaModule.h"

#include <optional>
#include <vector>

namespace tensorrt_llm::batch_manager
{

class TrtGptModelOptionalParams
{
    using KvCacheConfig = kv_cache_manager::KvCacheConfig;

public:
    using SizeType = tensorrt_llm::runtime::SizeType;

    explicit TrtGptModelOptionalParams(KvCacheConfig const& kvCacheConfig = KvCacheConfig{},
        bool enableTrtOverlap = false, std::optional<std::vector<SizeType>> const& deviceIds = std::nullopt,
        bool normalizeLogProbs = true, bool enableChunkedContext = false,
        std::optional<runtime::DecodingMode> const& decodingMode = std::nullopt,
        PeftCacheManagerConfig const& peftCacheManagerConfig = PeftCacheManagerConfig{},
        std::optional<runtime::MedusaModule::MedusaChoices> const& medusaChoices = std::nullopt)
        : kvCacheConfig{kvCacheConfig}
        , enableTrtOverlap{enableTrtOverlap}
        , deviceIds(deviceIds)
        , normalizeLogProbs{normalizeLogProbs}
        , enableChunkedContext{enableChunkedContext}
        , decodingMode{decodingMode}
        , peftCacheManagerConfig(peftCacheManagerConfig)
        , medusaChoices(medusaChoices)
    {
    }

    explicit TrtGptModelOptionalParams(executor::ExecutorConfig const& executorConfig)
        : TrtGptModelOptionalParams(KvCacheConfig(executorConfig.getKvCacheConfig()), false,
            executorConfig.getParallelConfig().value_or(executor::ParallelConfig()).getDeviceIds(),
            executorConfig.getNormalizeLogProbs(), executorConfig.getEnableChunkedContext(), std::nullopt,
            PeftCacheManagerConfig(executorConfig.getPeftCacheConfig().value_or(executor::PeftCacheConfig())),
            executorConfig.getMedusaChoices())
    {
    }

    bool operator==(TrtGptModelOptionalParams const& other) const
    {
        return kvCacheConfig == other.kvCacheConfig && enableTrtOverlap == other.enableTrtOverlap
            && deviceIds == other.deviceIds && normalizeLogProbs == other.normalizeLogProbs
            && enableChunkedContext == other.enableChunkedContext && decodingMode == other.decodingMode;
    }

    KvCacheConfig kvCacheConfig;

    bool enableTrtOverlap;
    std::optional<std::vector<SizeType>> deviceIds;
    bool normalizeLogProbs;
    bool enableChunkedContext;
    std::optional<runtime::DecodingMode> decodingMode;
    PeftCacheManagerConfig peftCacheManagerConfig;
    std::optional<runtime::MedusaModule::MedusaChoices> medusaChoices;
};

} // namespace tensorrt_llm::batch_manager
