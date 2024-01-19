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
#include "tensorrt_llm/runtime/gptModelConfig.h"
#include "tensorrt_llm/runtime/iTensor.h"
#include "tensorrt_llm/runtime/worldConfig.h"

namespace tensorrt_llm::runtime::lora
{

SizeType constexpr kLORA_CONFIG_ROW_SIZE = 3;
SizeType constexpr kLORA_CONFIG_MODULE_OFF = 0;
SizeType constexpr kLORA_CONFIG_LAYER_OFF = 1;
SizeType constexpr kLORA_CONFIG_ADAPTER_SIZE_OFF = 2;

SizeType constexpr kLORA_NUM_WEIGHTS_POINTERS = 2;

void loraValidateRequestTensorDims(const std::optional<ITensor::SharedPtr>& optReqLoraWeights,
    const std::optional<ITensor::SharedPtr>& optReqLoraConfig);

void loraValidateRequestTensors(const std::optional<ITensor::SharedPtr>& optReqLoraWeights,
    const std::optional<ITensor::SharedPtr>& optReqLoraConfig, runtime::GptModelConfig const& modelConfig,
    runtime::WorldConfig const& worldConfig);
} // namespace tensorrt_llm::runtime::lora
