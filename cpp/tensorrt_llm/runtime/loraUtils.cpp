/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
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
    TLLM_CHECK_WITH_INFO(keys->getShape().d[2] == kLORA_CONFIG_ROW_SIZE,
        "Expected dim2 of lora_keys to have a size of " + std::to_string(kLORA_CONFIG_ROW_SIZE));
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
        SizeType32 nbModelLayers = modelConfig.getNbAttentionLayers();
        TLLM_CHECK_WITH_INFO(weights->getDataType() == modelConfig.getDataType(),
            "Expected lora weights to be the same data type as base model");

        auto loraModules = modelConfig.getLoraModules();
        auto configPtr = bufferCast<SizeType32>(*config);
        for (SizeType32 row = 0; row < config->getShape().d[1]; ++row)
        {
            auto modId = configPtr[row * kLORA_CONFIG_ROW_SIZE + kLORA_CONFIG_MODULE_OFF];
            auto layerId = configPtr[row * kLORA_CONFIG_ROW_SIZE + kLORA_CONFIG_LAYER_OFF];
            auto adapterSize = configPtr[row * kLORA_CONFIG_ROW_SIZE + kLORA_CONFIG_ADAPTER_SIZE_OFF];

            TLLM_CHECK_WITH_INFO(
                layerId >= 0 && layerId < nbModelLayers, "Expected layerId to be in the range [0, numModelLayers)");
            TLLM_CHECK_WITH_INFO(adapterSize > 0, "Expected adapterSize to be > 0");
            auto it = std::find_if(
                loraModules.begin(), loraModules.end(), [modId](LoraModule const& m) { return m.value() == modId; });
            std::string moduleName(LoraModule::toModuleName(modId));
            TLLM_CHECK_WITH_INFO(it != loraModules.end(), "lora module " + moduleName + " not enabled for this model");
            TLLM_CHECK_WITH_INFO(it->flattenedInOutSize(adapterSize) <= weights->getShape().d[2],
                "lora_weights has to few values for " + moduleName);
        }
    }
}
} // namespace tensorrt_llm::runtime::lora
