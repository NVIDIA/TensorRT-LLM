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

#include "tensorrt_llm/runtime/loraManager.h"
#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/common/memoryUtils.h"
#include "tensorrt_llm/runtime/common.h"
#include "tensorrt_llm/runtime/iBuffer.h"
#include "tensorrt_llm/runtime/iTensor.h"
#include "tensorrt_llm/runtime/loraUtils.h"
#include "tensorrt_llm/runtime/modelConfig.h"
#include "tensorrt_llm/runtime/utils/runtimeUtils.h"
#include "tensorrt_llm/runtime/worldConfig.h"

#include <NvInferRuntime.h>

namespace tensorrt_llm::runtime
{

void LoraManager::create(ModelConfig const& modelConfig)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    auto modules = modelConfig.getLoraModules();
    SizeType32 modOff = 0;
    for (auto const& m : modules)
    {
        mModuleIdToModule[m.value()] = m;
        mModuleOffset[m.value()] = modOff++;
    }

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void LoraManager::fillInputTensors(TensorPtr weightsPtrs, TensorPtr adapterSizes, PeftTable const& peftTable,
    ReqIdsVec const& reqIds, std::vector<SizeType32> const& reqBeamWidth, ModelConfig const& modelConfig,
    WorldConfig const& worldConfig)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    auto batchSize = static_cast<SizeType32>(reqIds.size());
    for (SizeType32 bid = 0; bid < batchSize; ++bid)
    {
        auto it = peftTable.find(reqIds[bid]);
        if (it == peftTable.end())
        {
            continue;
        }
        auto peftValues = it->second;
        fillInputTensors(weightsPtrs, adapterSizes, peftValues, bid, reqBeamWidth[bid], modelConfig, worldConfig);
    }
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void LoraManager::fillInputTensors(TensorPtr weightsPtrs, TensorPtr adapterSizes, PeftValues const& peftValues,
    SizeType32 batchIdx, SizeType32 beamWidth, ModelConfig const& modelConfig, WorldConfig const& worldConfig)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    auto const ppSize = worldConfig.getPipelineParallelism();
    auto const ppRank = worldConfig.getPipelineParallelRank();
    auto const localNumLayers = modelConfig.getNbAttentionLayers(ppSize, ppRank);
    auto const firstLayerId = ppRank * localNumLayers;

    auto weightsPointersPtr = bufferCast<int64_t>(*weightsPtrs);
    auto adapterSizesPtr = bufferCast<int32_t>(*adapterSizes);

    TLLM_CHECK(!peftValues.empty());

    auto const numRows = static_cast<SizeType32>(peftValues.size());
    for (SizeType32 row = 0; row < numRows; ++row)
    {
        auto const& peftValue = peftValues.at(row);
        auto const moduleId = peftValue.moduleId;
        auto const adapterSize = peftValue.adapterSize;
        auto const modOff = mModuleOffset.at(moduleId);
        auto const layerIdx = peftValue.layerId;

        auto const inWeightsPtr = peftValue.weightsInPointer;
        auto const outWeightsPtr = peftValue.weightsOutPointer;
        auto const scalingVecPtr = peftValue.scalingVecPointer.value_or(0);

        auto weightsPointersPtrOffset = common::flat_index4(modOff, layerIdx - firstLayerId, batchIdx, SizeType32{0},
            weightsPtrs->getShape().d[1], weightsPtrs->getShape().d[2], weightsPtrs->getShape().d[3]);
        auto adapterSizesPtrOffset = common::flat_index3(
            modOff, layerIdx - firstLayerId, batchIdx, adapterSizes->getShape().d[1], adapterSizes->getShape().d[2]);

        TLLM_CHECK_WITH_INFO(static_cast<SizeType32>(weightsPtrs->getSize())
                >= weightsPointersPtrOffset + lora::kLORA_NUM_WEIGHTS_POINTERS * beamWidth,
            "Coding error attempting to write lora ptrs outside range of buffer");
        TLLM_CHECK_WITH_INFO(static_cast<SizeType32>(adapterSizes->getSize()) >= adapterSizesPtrOffset + beamWidth,
            "Coding error attempting to write lora low ranks outside range of buffer");

        auto const writeWeightsPtr = weightsPointersPtr + weightsPointersPtrOffset;
        auto const writeAdapterSizesPtr = adapterSizesPtr + adapterSizesPtrOffset;

        SizeType32 weightsPtrsOff = 0;
        for (SizeType32 beamIdx = 0; beamIdx < beamWidth; ++beamIdx)
        {
            writeWeightsPtr[weightsPtrsOff++] = inWeightsPtr;
            writeWeightsPtr[weightsPtrsOff++] = outWeightsPtr;
            writeWeightsPtr[weightsPtrsOff++] = scalingVecPtr;
        }
        std::fill_n(writeAdapterSizesPtr, beamWidth, adapterSize);
    }
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void LoraManager::insertInputTensors(TensorMap& inputTensors, TensorPtr weightsPtrs, TensorPtr adapterSizes,
    ModelConfig const& modelConfig, WorldConfig const& worldConfig) const
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    auto localNbLayers
        = modelConfig.getNbAttentionLayers(worldConfig.getPipelineParallelism(), worldConfig.getPipelineParallelRank());
    auto firstLayerId = worldConfig.getPipelineParallelRank() * localNbLayers;

    for (auto const& [modId, mod] : mModuleIdToModule)
    {
        auto modOff = mModuleOffset.at(modId);

        TensorPtr weightPtrsModSlice = ITensor::slice(weightsPtrs, modOff, 1);
        weightPtrsModSlice->squeeze(0);
        TensorPtr adapterSizesModSlice = ITensor::slice(adapterSizes, modOff, 1);
        adapterSizesModSlice->squeeze(0);

        auto weightsPtrsFieldName = std::string(mod.name()) + "_lora_weights_pointers_";
        auto lowRankFieldName = std::string(mod.name()) + "_lora_ranks_";

        utils::insertTensorSlices(inputTensors, weightsPtrsFieldName, weightPtrsModSlice, firstLayerId);
        utils::insertTensorSlices(inputTensors, lowRankFieldName, adapterSizesModSlice, firstLayerId);

        TLLM_LOG_DEBUG("weightPtrsModSlice shape %s", ITensor::toString(weightPtrsModSlice->getShape()).c_str());
        TLLM_LOG_DEBUG("adapterSizesModSlice shape %s", ITensor::toString(adapterSizesModSlice->getShape()).c_str());
        TLLM_LOG_DEBUG("lora fields");
        for (auto i : inputTensors)
        {
            auto name = i.first;
            if (name.find("lora") != std::string::npos)
            {
                TLLM_LOG_DEBUG("%s %s", name.c_str(), ITensor::toString(i.second->getShape()).c_str());
            }
        }
    }
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}
} // namespace tensorrt_llm::runtime
