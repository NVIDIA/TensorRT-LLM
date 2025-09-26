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

#include "loraBuffers.h"

#include "tensorrt_llm/batch_manager/llmRequest.h"
#include "tensorrt_llm/runtime/loraUtils.h"

namespace tensorrt_llm::batch_manager
{

LoraBuffers::LoraBuffers(SizeType32 maxBatchSize, SizeType32 maxBeamWidth, runtime::TllmRuntime const& tllmRuntime,
    runtime::ModelConfig const& modelConfig, runtime::WorldConfig const& worldConfig)
{
    auto const localNbLayers
        = modelConfig.getNbAttentionLayers(worldConfig.getPipelineParallelism(), worldConfig.getPipelineParallelRank());
    auto const firstLayerId = worldConfig.getPipelineParallelRank() * localNbLayers;

    auto nbModelConfigs = static_cast<SizeType32>(modelConfig.getLoraModules().size());

    // there are 3 pointers: LoRA A, LoRA B, and a DoRA magnitude (null if not DoRA)
    auto loraWeightsPtrsShape
        = runtime::ITensor::makeShape({nbModelConfigs, localNbLayers, maxBatchSize * maxBeamWidth, 3});
    auto loraAdapterSizesShape
        = runtime::ITensor::makeShape({nbModelConfigs, localNbLayers, maxBatchSize * maxBeamWidth});

    auto firstModuleName = std::string(modelConfig.getLoraModules().front().name());
    auto ptrsFieldName = firstModuleName + "_lora_weights_pointers_" + std::to_string(firstLayerId);
    auto rankFieldName = firstModuleName + "_lora_ranks_" + std::to_string(firstLayerId);
    auto weightsPtrDtype = tllmRuntime.getEngine().getTensorDataType(ptrsFieldName.c_str());
    auto ranksDtype = tllmRuntime.getEngine().getTensorDataType(rankFieldName.c_str());

    mLoraManager.create(modelConfig);

    mLoraWeightsPointersHost = runtime::BufferManager::pinned(loraWeightsPtrsShape, weightsPtrDtype);
    mLoraAdapterSizesHost = runtime::BufferManager::pinned(loraAdapterSizesShape, ranksDtype);
}

void LoraBuffers::fill(RequestVector const& contextRequests, RequestVector const& genRequests,
    PeftTable const& peftTable, runtime::BufferManager const& manager, runtime::ModelConfig const& modelConfig,
    runtime::WorldConfig const& worldConfig)
{
    manager.setZero(*mLoraWeightsPointersHost);
    manager.setZero(*mLoraAdapterSizesHost);

    SizeType32 batchIdx{0};
    for (auto const& requests : {contextRequests, genRequests})
    {
        for (auto const& llmReq : requests)
        {
            auto const optReqLoraWeights = llmReq->getLoraWeights();
            auto const optReqLoraConfig = llmReq->getLoraConfig();

            auto const isContextRequest = llmReq->isContextInitState();
            auto const beamWidth = isContextRequest ? 1 : llmReq->mSamplingConfig.beamWidth;
            auto const peftIt = peftTable.find(llmReq->mRequestId);
            if (peftIt != peftTable.end())
            {
                auto const& peftValues = peftIt->second;
                if (!peftValues.empty())
                {
                    mLoraManager.fillInputTensors(mLoraWeightsPointersHost, mLoraAdapterSizesHost, peftIt->second,
                        batchIdx, beamWidth, modelConfig, worldConfig);
                }
            }
            ++batchIdx;
        }
    }
}

void LoraBuffers::validate(std::optional<std::uint64_t> const& optTaskId,
    std::optional<TensorPtr> const& optReqLoraWeights, std::optional<TensorPtr> const& optReqLoraConfig,
    runtime::ModelConfig const& modelConfig, runtime::WorldConfig const& worldConfig)
{
    runtime::lora::loraValidateRequestTensors(optTaskId, optReqLoraWeights, optReqLoraConfig, modelConfig, worldConfig);
}

void LoraBuffers::insertInputTensors(TensorMap& inputTensors, TensorPtr weightsPtrs, TensorPtr adapterSizes,
    runtime::ModelConfig const& modelConfig, runtime::WorldConfig const& worldConfig) const
{
    mLoraManager.insertInputTensors(inputTensors, weightsPtrs, adapterSizes, modelConfig, worldConfig);
}

void LoraBuffers::reshape(SizeType32 numSequences)
{
    auto weightsPtrsShape = mLoraWeightsPointersHost->getShape();
    weightsPtrsShape.d[2] = numSequences;
    mLoraWeightsPointersHost->reshape(weightsPtrsShape);

    auto adapterSizesShape = mLoraAdapterSizesHost->getShape();
    adapterSizesShape.d[2] = numSequences;
    mLoraAdapterSizesHost->reshape(adapterSizesShape);
}

} // namespace tensorrt_llm::batch_manager
