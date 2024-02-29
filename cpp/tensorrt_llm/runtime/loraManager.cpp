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
#include "tensorrt_llm/runtime/bufferManager.h"
#include "tensorrt_llm/runtime/common.h"
#include "tensorrt_llm/runtime/gptModelConfig.h"
#include "tensorrt_llm/runtime/iBuffer.h"
#include "tensorrt_llm/runtime/iTensor.h"
#include "tensorrt_llm/runtime/loraUtils.h"
#include "tensorrt_llm/runtime/runtimeKernels.h"
#include "tensorrt_llm/runtime/utils/sessionUtils.h"
#include "tensorrt_llm/runtime/worldConfig.h"
#include <NvInferRuntimeBase.h>

namespace tensorrt_llm::runtime
{
void LoraManager::addTask(TaskIdType reqId, TensorPtr weights, TensorPtr config)
{
    if (mLoras.find(reqId) != mLoras.end())
    {
        return;
    }

    mLoras[reqId] = std::make_tuple(weights, config);
}

LoraManager::LoraReqTensors& LoraManager::getTask(TaskIdType reqId)
{
    return mLoras.at(reqId);
}

void LoraManager::create(
    GptModelConfig const& modelConfig, WorldConfig const& worldConfig, BufferManager const& manager)
{
    TLLM_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);

    auto modules = modelConfig.getLoraModules();
    SizeType modOff = 0;
    for (auto const& m : modules)
    {
        mModuleIdToModule[m.value()] = m;
        mModuleOffest[m.value()] = modOff++;
    }

    // TODO set this size from max adapter size
    mWorkspace = manager.emptyTensor(MemoryType::kGPU, modelConfig.getDataType());

    TLLM_LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
}

void LoraManager::fillInputTensors(TensorPtr weightsPtrs, TensorPtr adapterSizes, ReqIdsVec const& reqIds,
    std::vector<SizeType> const& reqBeamWidth, std::vector<bool> const& loraEnabled, SizeType numContextRequests,
    GptModelConfig const& modelConfig, WorldConfig const& worldConfig)
{
    TLLM_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
    auto localNbLayers = modelConfig.getNbLayers(worldConfig.getPipelineParallelism());
    auto firstLayerId = worldConfig.getPipelineParallelRank() * localNbLayers;
    auto lastLayerId = firstLayerId + localNbLayers;
    auto tpSize = worldConfig.getTensorParallelism();
    auto tpRank = worldConfig.getTensorParallelRank();

    auto batchSize = static_cast<SizeType>(reqIds.size());
    for (SizeType bid = 0; bid < batchSize; ++bid)
    {
        if (!loraEnabled[bid])
            continue;

        fillInputTensors(
            weightsPtrs, adapterSizes, bid, reqIds[bid], reqBeamWidth[bid], firstLayerId, lastLayerId, tpSize, tpRank);
    }
    TLLM_LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
}

void LoraManager::fillInputTensors(TensorPtr weightsPtrs, TensorPtr adapterSizes, SizeType batchIdx, TaskIdType taskId,
    SizeType beamWidth, SizeType firstLayerId, SizeType lastLayerId, SizeType tpSize, SizeType tpRank)
{
    TLLM_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);

    auto weightsPointersPtr = bufferCast<int64_t>(*weightsPtrs);
    auto adapterSizesPtr = bufferCast<int32_t>(*adapterSizes);

    auto [reqWeights, reqKeys] = getTask(taskId);
    auto reqKeysPtr = bufferCast<SizeType>(*reqKeys);
    auto numRows = reqKeys->getShape().d[0];
    if (reqKeys->getShape().d[1] != lora::kLORA_CONFIG_ROW_SIZE)
    {
        throw std::runtime_error(
            "Expected request lora_keys tor have row size of " + std::to_string(lora::kLORA_CONFIG_ROW_SIZE));
    }
    for (SizeType row = 0; row < numRows; ++row)
    {
        auto layerIdx = reqKeysPtr[row * lora::kLORA_CONFIG_ROW_SIZE + lora::kLORA_CONFIG_LAYER_OFF];
        if (layerIdx < firstLayerId || layerIdx >= lastLayerId)
            continue;

        auto moduleId = reqKeysPtr[row * lora::kLORA_CONFIG_ROW_SIZE + lora::kLORA_CONFIG_MODULE_OFF];
        auto adapterSize = reqKeysPtr[row * lora::kLORA_CONFIG_ROW_SIZE + lora::kLORA_CONFIG_ADAPTER_SIZE_OFF];

        auto modOff = mModuleOffest.at(moduleId);
        auto& module = mModuleIdToModule.at(moduleId);

        auto inDim = (module.inDimFirst() && module.inTpSplitDim() == 0)
                || (!module.inDimFirst() && module.inTpSplitDim() == 1)
            ? module.inDim() / tpSize
            : module.inDim();
        auto inTpSize = module.inTpSplitDim() == -1 ? 1 : tpSize;
        auto inTpRank = module.inTpSplitDim() == -1 ? 0 : tpRank;

        auto outDim = (module.outDimFirst() && module.outTpSplitDim() == 0)
                || (!module.outDimFirst() && module.outTpSplitDim() == 1)
            ? module.outDim() / tpSize
            : module.outDim();
        auto outTpSize = module.outTpSplitDim() == -1 ? 1 : tpSize;
        auto outTpRank = module.outTpSplitDim() == -1 ? 0 : tpRank;

        auto inWeightsShape = module.inDimFirst() ? ITensor::makeShape({inTpSize, inDim, adapterSize})
                                                  : ITensor::makeShape({inTpSize, adapterSize, inDim});
        auto outWeightsShape = module.outDimFirst() ? ITensor::makeShape({outTpSize, outDim, adapterSize})
                                                    : ITensor::makeShape({outTpSize, adapterSize, outDim});

        TensorPtr reqRowWeights = ITensor::slice(reqWeights, row, 1);
        reqRowWeights->squeeze(0);
        TensorPtr allInWeights = ITensor::view(reqRowWeights, inWeightsShape);

        TensorPtr allOutWeights
            = ITensor::slice(reqRowWeights, allInWeights->getSize(), ITensor::volume(outWeightsShape));
        allOutWeights->reshape(outWeightsShape);

        auto inWeightsPtr = reinterpret_cast<int64_t>(ITensor::slice(allInWeights, inTpRank, 1)->data());
        auto outWeightsPtr = reinterpret_cast<int64_t>(ITensor::slice(allOutWeights, outTpRank, 1)->data());

        auto weightsPointersPtrOffset = common::flat_index4(modOff, layerIdx - firstLayerId, batchIdx, 0,
            weightsPtrs->getShape().d[1], weightsPtrs->getShape().d[2], weightsPtrs->getShape().d[3]);
        auto adapterSizesPtrOffset = common::flat_index3(
            modOff, layerIdx - firstLayerId, batchIdx, adapterSizes->getShape().d[1], adapterSizes->getShape().d[2]);

        if (static_cast<SizeType>(weightsPtrs->getSize())
            < weightsPointersPtrOffset + lora::kLORA_NUM_WEIGHTS_POINTERS * beamWidth)
        {
            throw std::runtime_error("Coding error attempting to write lora ptrs outside range of buffer");
        }
        if (static_cast<SizeType>(adapterSizes->getSize()) < adapterSizesPtrOffset + beamWidth)
        {
            throw std::runtime_error("Coding error attempting to write lora low ranks outside range of buffer");
        }

        auto const writeWeightsPtr = weightsPointersPtr + weightsPointersPtrOffset;
        auto const writeAdapterSizesPtr = adapterSizesPtr + adapterSizesPtrOffset;

        SizeType weightsPtrsOff = 0;
        for (SizeType beamIdx = 0; beamIdx < beamWidth; ++beamIdx)
        {
            writeWeightsPtr[weightsPtrsOff++] = inWeightsPtr;
            writeWeightsPtr[weightsPtrsOff++] = outWeightsPtr;
        }
        std::fill_n(writeAdapterSizesPtr, beamWidth, adapterSize);
    }
    TLLM_LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
}

void LoraManager::insertInputTensors(TensorMap& inputTensors, TensorPtr weightsPtrs, TensorPtr adapterSizes,
    GptModelConfig const& modelConfig, WorldConfig const& worldConfig) const
{
    TLLM_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
    auto localNbLayers = modelConfig.getNbLayers(worldConfig.getPipelineParallelism());
    auto firstLayerId = worldConfig.getPipelineParallelRank() * localNbLayers;

    for (auto const& [modId, mod] : mModuleIdToModule)
    {
        auto modOff = mModuleOffest.at(modId);

        TensorPtr weightPtrsModSlice = ITensor::slice(weightsPtrs, modOff, 1);
        weightPtrsModSlice->squeeze(0);
        TensorPtr adapterSizesModSlice = ITensor::slice(adapterSizes, modOff, 1);
        adapterSizesModSlice->squeeze(0);

        auto weightsPtrsFieldName = std::string(mod.name()) + "_lora_weights_pointers_";
        auto lowRankFieldName = std::string(mod.name()) + "_lora_ranks_";

        utils::insertTensorSlices(inputTensors, weightsPtrsFieldName, weightPtrsModSlice, firstLayerId);
        utils::insertTensorSlices(inputTensors, lowRankFieldName, adapterSizesModSlice, firstLayerId);

        TLLM_LOG_DEBUG("weightPtrsModSlice shape %s", weightPtrsModSlice->getShape());
        TLLM_LOG_DEBUG("adapterSizesModSlice shape %s", adapterSizesModSlice->getShape());
        TLLM_LOG_DEBUG("lora fields");
        for (auto i : inputTensors)
        {
            auto name = i.first;
            if (name.find("lora") != std::string::npos)
            {
                TLLM_LOG_DEBUG("%s %s", name, i.second->getShape());
                TLLM_LOG_DEBUG("%s", i.second->toString(i.second->getShape()));
            }
        }
    }
    TLLM_LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
}

void LoraManager::formatTaskTensors(LoraWeightsTensorPtr weights, LoraConfigTensorPtr config,
    GptModelConfig const& modelConfig, WorldConfig const& worldConfig, BufferManager const& manager)
{
    TLLM_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
    weights->squeeze(0);
    config->squeeze(0);

    auto tpSize = worldConfig.getTensorParallelism();

    SizeType nbRows = config->getShape().d[0];
    for (SizeType row = 0; row < nbRows; ++row)
    {
        auto rowPtr = bufferCast<SizeType>(*ITensor::slice(config, row, 1));
        auto modId = rowPtr[lora::kLORA_CONFIG_MODULE_OFF];
        auto adapterSize = rowPtr[lora::kLORA_CONFIG_ADAPTER_SIZE_OFF];

        auto& module = mModuleIdToModule.at(modId);
        TLLM_CHECK_WITH_INFO(!module.inDimFirst() && module.outDimFirst(), "unsupported module");
        if (module.inTpSplitDim() == 1)
        {
            TensorPtr inWeights = ITensor::slice(weights, row, 1);
            inWeights->squeeze(0);
            inWeights->reshape(ITensor::makeShape({adapterSize, module.inDim()}));
            if (mWorkspace->getSize() < inWeights->getSize())
            {
                mWorkspace = manager.gpu(inWeights->getShape(), inWeights->getDataType());
            }
            mWorkspace->reshape(ITensor::makeShape({tpSize, adapterSize, module.inDim() / tpSize}));
            kernels::splitTransposed(*mWorkspace, *inWeights, tpSize, manager.getStream());
            manager.copy(*mWorkspace, *inWeights);
        }
        if (module.outTpSplitDim() == 1)
        {
            TensorPtr rowWeights = ITensor::slice(weights, row, 1);
            rowWeights->squeeze(0);
            TensorPtr weightsOut
                = ITensor::slice(rowWeights, adapterSize * module.inDim(), adapterSize * module.outDim());
            weightsOut->squeeze(0);
            weightsOut->reshape(ITensor::makeShape({module.outDim(), adapterSize}));
            if (mWorkspace->getSize() < weightsOut->getSize())
            {
                mWorkspace = manager.gpu(weightsOut->getShape(), weightsOut->getDataType());
            }
            mWorkspace->reshape(weightsOut->getShape());
            kernels::splitTransposed(*mWorkspace, *weightsOut, tpSize, manager.getStream());
            manager.copy(*mWorkspace, *weightsOut);
        }
    }
    TLLM_LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
}

void LoraManager::reset()
{
    mLoras.clear();
}

} // namespace tensorrt_llm::runtime
