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

#include "tensorrt_llm/batch_manager/common.h"
#include "tensorrt_llm/runtime/bufferManager.h"
#include "tensorrt_llm/runtime/common.h"
#include "tensorrt_llm/runtime/iTensor.h"
#include "tensorrt_llm/runtime/loraManager.h"
#include "tensorrt_llm/runtime/modelConfig.h"
#include "tensorrt_llm/runtime/tllmRuntime.h"
#include "tensorrt_llm/runtime/worldConfig.h"

namespace tensorrt_llm::batch_manager
{

class LoraBuffers
{
public:
    using SizeType32 = tensorrt_llm::runtime::SizeType32;
    using PeftTable = runtime::LoraManager::PeftTable;
    using TensorPtr = runtime::ITensor::SharedPtr;
    using TensorMap = runtime::StringPtrMap<runtime::ITensor>;

    TensorPtr mLoraWeightsPointersHost;
    TensorPtr mLoraAdapterSizesHost;

    runtime::LoraManager mLoraManager;

    LoraBuffers(SizeType32 maxBatchSize, SizeType32 maxBeamWidth, runtime::TllmRuntime const& tllmRuntime,
        runtime::ModelConfig const& modelConfig, runtime::WorldConfig const& worldConfig);

    static void validate(std::optional<std::uint64_t> const& optTaskId,
        std::optional<TensorPtr> const& optReqLoraWeights, std::optional<TensorPtr> const& optReqLoraConfig,
        runtime::ModelConfig const& modelConfig, runtime::WorldConfig const& worldConfig);

    void fill(RequestVector const& contextRequests, RequestVector const& genRequests, PeftTable const& peftTable,
        runtime::BufferManager const& manager, runtime::ModelConfig const& modelConfig,
        runtime::WorldConfig const& worldConfig);

    void insertInputTensors(TensorMap& inputTensors, TensorPtr weightsPtrs, TensorPtr adapterSizes,
        runtime::ModelConfig const& modelConfig, runtime::WorldConfig const& worldConfig) const;

    void reshape(SizeType32 numSequences);
};
} // namespace tensorrt_llm::batch_manager
