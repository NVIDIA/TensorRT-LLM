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
