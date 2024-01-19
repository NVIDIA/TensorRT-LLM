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
#include "tensorrt_llm/runtime/gptModelConfig.h"
#include "tensorrt_llm/runtime/loraModule.h"
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

    explicit LoraManager() {}

    /**
     * \brief Sets up and configures LoraManager. Allocates and needed device / host memory
     * \param[in] modelConfig: a GptModelConfig.
     * \param[in] worldConfig: a WorldConfig
     * \param[in] manager: and BufferManager used to allocate memory
     */
    void create(GptModelConfig const& modelConfig, WorldConfig const& worldConfig, BufferManager const& manager);

    /**
     * \brief Add Task (LoRA tensor to manager)
     * \details weights and config are assumed to be in the proper format
     *          and to have been formatted with formatTaskTensors
     * \param[in] taskId: id associated with these lora weights
     * \param[in] weights: LoRA weights tensor [num_modules_layers, D x Hi + Ho x D].
     *                     Each row contains the flattened in / out LoRA weights for a single module / layer.
     *                     D=adapter size (R value); Hi=hidden dim of in weights; Ho=hidden dim of out weights
     * \param[in] config: LoRA config tensor [num_modules_layers, 3]
     *                    each row contains 3 values (module_id, layer_idx, D)
     *                    See LoraModule::ModelType for module_id details
     */
    void addTask(TaskIdType taskId, LoraWeightsTensorPtr weights, LoraConfigTensorPtr config);

    /**
     * \brief getTask by taskId
     * \param[in] taskId: task id
     */
    LoraReqTensors& getTask(TaskIdType taskId);

    /**
     * \brief format tensors for addTask.  See addTask for details on expected format
     * \param[out] weights: LoRA weights tensor. See addTask for details
     * \param[out] config: LoRA config tensor. See addTask for details
     * \param[in] modelConfig: A GptModelConfig
     * \param[in] worldConfig: A WorldConfig
     * \param[in]: manager: A BufferManager
     */
    void formatTaskTensors(LoraWeightsTensorPtr weights, LoraConfigTensorPtr config, GptModelConfig const& modelConfig,
        WorldConfig const& worldConfig, BufferManager const& manager);

    /**
     * \brief same as fillInputTensors but for an entire batch
     */
    void fillInputTensors(TensorPtr weightsPtrs, TensorPtr adapterSizes, ReqIdsVec const& reqIds,
        std::vector<SizeType> const& reqBeamWidth, std::vector<bool> const& loraEnabled, SizeType numContextRequests,
        GptModelConfig const& modelConfig, WorldConfig const& worldConfig);

    /**
     * \brief fill batch input tensors for LoRA.  This method fills on batch slot.
     * \param[out] weightsPtrs: the tensor of pointers to lora weights to fill.
     *                          (ie for `*_lora_weights_pointers_*` fields)
     * \param[out] adapterSizes: the adapter sizes tensor to fill
     *                           (ie for `*lora_low_rank_*` fields)
     * \param[in] batchIdx: the request batch index
     * \param[in] taskId: the LoRA task id to use
     * \param[in] beamWidth: the request beam width
     * \param[in] firstLayerId: firstLaterId in this rank for pipeline parallel models
     * \param[in] lastLayerId: firstLayerId in this rank for pipeline parallel models
     * \param[in] tpSize: tensor parallel size
     * \param[in] tpRank: tensor parallel rank
     */
    void fillInputTensors(TensorPtr weightsPtrs, TensorPtr adapterSizes, SizeType batchIdx, TaskIdType taskId,
        SizeType beamWidth, SizeType firstLayerId, SizeType lastLayerId, SizeType tpSize, SizeType tpRank);

    /**
     * \brief fill tensor map for trt engine context
     * \param[out] inputTensors: the tensor map to fill
     * \param[in] weightsPtrs: tensor of weights pointers as filled in fillInputTensors
     * \param[in] adapterSizes: tensor of adapter sizes as filled in fillInputTensors
     * \param[in] modelConfig: a GptModelConfig
     * \param[in] worldConfig: a WorldConfig
     */
    void insertInputTensors(TensorMap& inputTensors, TensorPtr weightsPtrs, TensorPtr adapterSizes,
        GptModelConfig const& modelConfig, WorldConfig const& worldConfig) const;

    void reset();

private:
    TensorPtr mWorkspace;
    std::unordered_map<TaskIdType, LoraReqTensors> mLoras;
    std::unordered_map<SizeType, LoraModule> mModuleIdToModule;
    std::unordered_map<SizeType, SizeType> mModuleOffest;
};
} // namespace tensorrt_llm::runtime
