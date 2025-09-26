/*
 * Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "tensorrt_llm/runtime/loraUtils.h"
#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/runtime/common.h"
#include "tensorrt_llm/runtime/iTensor.h"
#include "tensorrt_llm/runtime/modelConfig.h"
#include "tensorrt_llm/runtime/worldConfig.h"
#include <string>

namespace tensorrt_llm::runtime::lora
{

LoraModuleConfig::LoraModuleConfig(LoraModuleConfig::TensorPtr loraConfigTensor)
{
    auto const dataPtr = bufferCast<int32_t>(*loraConfigTensor);
    // backwards compatibility with pre-dora configs
    auto const configSupportsDora = loraConfigTensor->getShape().d[1] >= lora::kLORA_CONFIG_ROW_SIZE;
    moduleId = dataPtr[lora::kLORA_CONFIG_MODULE_OFF];
    layerId = dataPtr[lora::kLORA_CONFIG_LAYER_OFF];
    adapterSize = dataPtr[lora::kLORA_CONFIG_ADAPTER_SIZE_OFF];
    isDora = configSupportsDora ? dataPtr[lora::kLORA_CONFIG_IS_DORA_OFF] : false;
}

void loraValidateRequestTensorDims(std::optional<ITensor::SharedPtr> const& optReqLoraWeights,
    std::optional<ITensor::SharedPtr> const& optReqLoraConfig)
{
    TLLM_CHECK_WITH_INFO(optReqLoraWeights.has_value() && optReqLoraConfig.has_value(),
        "Request for LoRA inference must have both lora_weights and lora_keys");

    SizeType32 constexpr expectedBatchSize = 1;
    SizeType32 constexpr expectedWeightsDims = 3;
    SizeType32 constexpr expectedKeysDims = 3;

    auto weights = optReqLoraWeights.value();
    auto keys = optReqLoraConfig.value();
    TLLM_CHECK_WITH_INFO(weights->getShape().nbDims == expectedWeightsDims, "Invalid shape for lora_weights tensor");
    TLLM_CHECK_WITH_INFO(keys->getShape().nbDims == expectedKeysDims, "Invalid shape for lora_keys tensor");
    TLLM_CHECK_WITH_INFO(
        weights->getShape().d[0] == expectedBatchSize, "Expected batch dimension to be 1 for each lora request");
    TLLM_CHECK_WITH_INFO(
        keys->getShape().d[0] == expectedBatchSize, "Expected batch dimension to be 1 for each lora request");
    TLLM_CHECK_WITH_INFO(weights->getMemoryType() != MemoryType::kGPU, "Expected lora weights to be in CPU memory");
    TLLM_CHECK_WITH_INFO(keys->getMemoryType() != MemoryType::kGPU, "Expected lora weights to be in CPU memory");
    TLLM_CHECK_WITH_INFO(keys->getDataType() == nvinfer1::DataType::kINT32,
        "Expected  lora keys to have TYPE_INT32 but was " + std::string(keys->getDataTypeName()));

    TLLM_CHECK_WITH_INFO(keys->getShape().d[1] == weights->getShape().d[1],
        "Expected dim1 lora_weights and lora_keys to have the same size");
    TLLM_CHECK_WITH_INFO(
        keys->getShape().d[2] == kLORA_CONFIG_ROW_SIZE or keys->getShape().d[2] == kLORA_CONFIG_PRE_DORA_ROW_SIZE,
        "Expected dim2 of lora_keys to have a size in [" + std::to_string(kLORA_CONFIG_PRE_DORA_ROW_SIZE) + ", "
            + std::to_string(kLORA_CONFIG_ROW_SIZE) + "]");
}

void loraValidateRequestTensors(std::optional<std::uint64_t> const& optTaskId,
    std::optional<ITensor::SharedPtr> const& optReqLoraWeights,
    std::optional<ITensor::SharedPtr> const& optReqLoraConfig, runtime::ModelConfig const& modelConfig,
    runtime::WorldConfig const& worldConfig)
{
    TLLM_CHECK_WITH_INFO(optTaskId.has_value(), "lora_task_id must be set for LoRA inference");
    if (optReqLoraWeights.has_value() || optReqLoraConfig.has_value())
    {
        loraValidateRequestTensorDims(optReqLoraWeights, optReqLoraConfig);

        auto weights = optReqLoraWeights.value();
        auto config = optReqLoraConfig.value();
        config = config->getShape().nbDims == 2
            ? config
            : ITensor::view(config, ITensor::makeShape({config->getShape().d[1], config->getShape().d[2]}));

        SizeType32 nbModelLayers = modelConfig.getNbAttentionLayers();
        TLLM_CHECK_WITH_INFO(weights->getDataType() == modelConfig.getDataType(),
            "Expected lora weights to be the same data type as base model");

        auto loraModules = modelConfig.getLoraModules();
        auto maxAdapterSize = modelConfig.getMaxLoraRank();
        for (SizeType32 row = 0; row < config->getShape().d[0]; ++row)
        {
            LoraModuleConfig const loraConfig(ITensor::slice(config, row, 1));
            auto modId = loraConfig.moduleId;
            auto layerId = loraConfig.layerId;
            auto adapterSize = loraConfig.adapterSize;
            bool const isDora = loraConfig.isDora;

            TLLM_CHECK_WITH_INFO(
                layerId >= 0 && layerId < nbModelLayers, "Expected layerId to be in the range [0, numModelLayers)");
            TLLM_CHECK_WITH_INFO(adapterSize > 0, "Expected adapterSize to be > 0");
            auto it = std::find_if(
                loraModules.begin(), loraModules.end(), [modId](LoraModule const& m) { return m.value() == modId; });
            std::string moduleName(LoraModule::toModuleName(modId));
            TLLM_CHECK_WITH_INFO(it != loraModules.end(), "lora module " + moduleName + " not enabled for this model");
            TLLM_CHECK_WITH_INFO(it->flattenedInOutSize(adapterSize, isDora) <= weights->getShape().d[2],
                "lora_weights has to few values for " + moduleName);
            TLLM_CHECK_WITH_INFO(adapterSize <= maxAdapterSize,
                "Invalid low_rank (" + std::to_string(adapterSize) + "). low_rank must be smaller than mMaxLowRank ("
                    + std::to_string(maxAdapterSize) + ")");
        }
    }
}
} // namespace tensorrt_llm::runtime::lora
