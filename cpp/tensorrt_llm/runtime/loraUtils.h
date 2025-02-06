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

#pragma once

#include "tensorrt_llm/runtime/common.h"
#include "tensorrt_llm/runtime/iTensor.h"
#include "tensorrt_llm/runtime/modelConfig.h"
#include "tensorrt_llm/runtime/worldConfig.h"

namespace tensorrt_llm::runtime::lora
{

// old lora configs without isDora field have only 3 fields
SizeType32 constexpr kLORA_CONFIG_PRE_DORA_ROW_SIZE = 3;
SizeType32 constexpr kLORA_CONFIG_ROW_SIZE = 4;
SizeType32 constexpr kLORA_CONFIG_MODULE_OFF = 0;
SizeType32 constexpr kLORA_CONFIG_LAYER_OFF = 1;
SizeType32 constexpr kLORA_CONFIG_ADAPTER_SIZE_OFF = 2;
// new lora configs have an additional isDora field in addition to the previous 3
SizeType32 constexpr kLORA_CONFIG_IS_DORA_OFF = 3;

// old lora weights without dora scaling ptr
SizeType32 constexpr kLORA_NUM_WEIGHTS_POINTERS_PRE_DORA = 2;
SizeType32 constexpr kLORA_NUM_WEIGHTS_POINTERS = 3;

class LoraModuleConfig
{
    using TensorPtr = ITensor::SharedPtr;

public:
    explicit LoraModuleConfig(TensorPtr loraConfigTensor);

    int32_t moduleId;
    int32_t layerId;
    int32_t adapterSize;
    bool isDora;
};

void loraValidateRequestTensorDims(std::optional<ITensor::SharedPtr> const& optReqLoraWeights,
    std::optional<ITensor::SharedPtr> const& optReqLoraConfig);

void loraValidateRequestTensors(std::optional<std::uint64_t> const& optTaskId,
    std::optional<ITensor::SharedPtr> const& optReqLoraWeights,
    std::optional<ITensor::SharedPtr> const& optReqLoraConfig, runtime::ModelConfig const& modelConfig,
    runtime::WorldConfig const& worldConfig);
} // namespace tensorrt_llm::runtime::lora
