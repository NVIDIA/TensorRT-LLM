/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.
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

#include "tensorrt_llm/runtime/bufferManager.h"
#include "tensorrt_llm/runtime/common.h"
#include "tensorrt_llm/runtime/loraCache.h"
#include "tensorrt_llm/runtime/loraModule.h"
#include "tensorrt_llm/runtime/modelConfig.h"
#include "tensorrt_llm/runtime/worldConfig.h"
#include <unordered_map>

namespace tensorrt_llm::runtime
{

/**
 * \brief Manages LoRA tensors.
 * \details Handles formatting input tensors and populating trt engine params related to LoRA.
 */
class LoraManager
{
public:
    using TensorPtr = ITensor::SharedPtr;
    using ReqIdsVec = std::vector<uint64_t>;
    using TensorMap = runtime::StringPtrMap<runtime::ITensor>;
    using LoraWeightsTensorPtr = TensorPtr;
    using LoraConfigTensorPtr = TensorPtr;
    using LoraReqTensors = std::tuple<LoraWeightsTensorPtr, LoraConfigTensorPtr>;
    using TaskIdType = std::int64_t;
    using PeftValues = std::vector<runtime::LoraCache::TaskLayerModuleConfig>;
    using PeftTable = std::map<uint64_t, std::vector<runtime::LoraCache::TaskLayerModuleConfig>>;

    explicit LoraManager() {}

    /**
     * \brief Sets up and configures LoraManager. Allocates and needed device / host memory
     * \param[in] modelConfig: a ModelConfig.
     * \param[in] worldConfig: a WorldConfig
     * \param[in] manager: and BufferManager used to allocate memory
     */
    void create(ModelConfig const& modelConfig);

    /**
     * \brief same as fillInputTensors but for an entire batch
     */
    void fillInputTensors(TensorPtr weightsPtrs, TensorPtr adapterSizes, PeftTable const& peftTable,
        ReqIdsVec const& reqIds, std::vector<SizeType32> const& reqBeamWidth, ModelConfig const& modelConfig,
        WorldConfig const& worldConfig);

    /**
     * \brief fill batch input tensors for LoRA.  This method fills on batch slot.
     * \param[out] weightsPtrs: the tensor of pointers to lora weights to fill.
     *                          (ie for `*_lora_weights_pointers_*` fields)
     * \param[out] adapterSizes: the adapter sizes tensor to fill
     *                           (ie for `*lora_low_rank_*` fields)
     * \param[in] peftTable: reqId to LoraCache::Values
     * \param[in] batchIdx: the request batch index
     * \param[in] beamWidth: the request beam width
     * \param[in] modelConfig: a ModelConfig
     * \param[in] worldConfig: a WorldConfig
     */
    void fillInputTensors(TensorPtr weightsPtrs, TensorPtr adapterSizes, PeftValues const& peftValues,
        SizeType32 batchIdx, SizeType32 beamWidth, ModelConfig const& modelConfig, WorldConfig const& worldConfig);

    /**
     * \brief fill tensor map for trt engine context
     * \param[out] inputTensors: the tensor map to fill
     * \param[in] weightsPtrs: tensor of weights pointers as filled in fillInputTensors
     * \param[in] adapterSizes: tensor of adapter sizes as filled in fillInputTensors
     * \param[in] modelConfig: a ModelConfig
     * \param[in] worldConfig: a WorldConfig
     */
    void insertInputTensors(TensorMap& inputTensors, TensorPtr weightsPtrs, TensorPtr adapterSizes,
        ModelConfig const& modelConfig, WorldConfig const& worldConfig) const;

private:
    std::unordered_map<SizeType32, LoraModule> mModuleIdToModule;
    std::unordered_map<SizeType32, SizeType32> mModuleOffset;
};
} // namespace tensorrt_llm::runtime
