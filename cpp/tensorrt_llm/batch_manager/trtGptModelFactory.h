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

#include "tensorrt_llm/runtime/gptJsonConfig.h"
#include "tensorrt_llm/runtime/modelConfig.h"
#include "tensorrt_llm/runtime/rawEngine.h"
#include "tensorrt_llm/runtime/tllmLogger.h"
#include "tensorrt_llm/runtime/worldConfig.h"
#include "trtGptModelInflightBatching.h"
#include "trtGptModelV1.h"

#include <NvInferPlugin.h>

#include <memory>
#include <optional>

namespace tensorrt_llm::batch_manager
{

class TrtGptModelFactory
{
public:
    using SizeType32 = tensorrt_llm::runtime::SizeType32;

    static std::shared_ptr<TrtGptModel> create(std::filesystem::path const& trtEnginePath, TrtGptModelType modelType,
        TrtGptModelOptionalParams const& optionalParams = TrtGptModelOptionalParams())
    {
        auto const jsonConfig = runtime::GptJsonConfig::parse(trtEnginePath / "config.json");
        auto worldConfig = getWorldConfig(jsonConfig, optionalParams.deviceIds);
        auto const enginePath = trtEnginePath / jsonConfig.engineFilename(worldConfig);

        auto const& modelConfig = jsonConfig.getModelConfig();
        return create(runtime::RawEngine(enginePath), modelConfig, worldConfig, modelType, optionalParams);
    }

    static std::shared_ptr<TrtGptModel> create(std::filesystem::path const& trtEnginePath, TrtGptModelType modelType,
        runtime::GptJsonConfig const& jsonConfig, runtime::WorldConfig const& worldConfig,
        TrtGptModelOptionalParams const& optionalParams = TrtGptModelOptionalParams())
    {
        auto const enginePath = trtEnginePath / jsonConfig.engineFilename(worldConfig);
        auto const& modelConfig = jsonConfig.getModelConfig();
        return create(runtime::RawEngine(enginePath), modelConfig, worldConfig, modelType, optionalParams);
    }

    static std::shared_ptr<TrtGptModel> create(runtime::RawEngine const& rawEngine,
        runtime::ModelConfig const& modelConfig, runtime::WorldConfig const& worldConfig, TrtGptModelType modelType,
        TrtGptModelOptionalParams const& optionalParams = TrtGptModelOptionalParams())
    {
        auto logger = std::make_shared<runtime::TllmLogger>();
        auto const device = worldConfig.getDevice();
        auto const rank = worldConfig.getRank();
        TLLM_LOG_INFO("Rank %d is using GPU %d", rank, device);
        TLLM_CUDA_CHECK(cudaSetDevice(device));

        if (modelType == TrtGptModelType::V1)
        {
            TLLM_LOG_WARNING(
                "TrtGptModelType::V1 is deprecated and will be removed in a future release."
                " Please use TrtGptModelType::InflightBatching or TrtGptModelType::InflightFusedBatching instead.");

            TrtGptModelOptionalParams const& fixedOptionalParams
                = TrtGptModelV1::optionalParamsAreValid(modelConfig, optionalParams)
                ? optionalParams
                : TrtGptModelV1::fixOptionalParams(modelConfig, optionalParams);
            return std::make_shared<TrtGptModelV1>(logger, modelConfig, worldConfig, rawEngine, fixedOptionalParams);
        }
        else if ((modelType == TrtGptModelType::InflightBatching)
            || (modelType == TrtGptModelType::InflightFusedBatching))
        {
            TrtGptModelOptionalParams const& fixedOptionalParams
                = TrtGptModelInflightBatching::optionalParamsAreValid(modelConfig, optionalParams)
                ? optionalParams
                : TrtGptModelInflightBatching::fixOptionalParams(modelConfig, optionalParams);
            return std::make_shared<TrtGptModelInflightBatching>(logger, modelConfig, worldConfig, rawEngine,
                (modelType == TrtGptModelType::InflightFusedBatching), fixedOptionalParams);
        }
        else
        {
            throw std::runtime_error("Invalid modelType in trtGptModelFactory");
        }
    }

private:
    static runtime::WorldConfig getWorldConfig(
        runtime::GptJsonConfig const& json, std::optional<std::vector<SizeType32>> const& deviceIds)
    {
        return runtime::WorldConfig::mpi(json.getGpusPerNode(), json.getTensorParallelism(),
            json.getPipelineParallelism(), json.getContextParallelism(), deviceIds);
    }
};

} // namespace tensorrt_llm::batch_manager
