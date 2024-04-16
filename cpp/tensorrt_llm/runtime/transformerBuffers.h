/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.  All rights reserved.
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
#include "tensorrt_llm/runtime/generationConfig.h"
#include "tensorrt_llm/runtime/gptModelConfig.h"
#include "tensorrt_llm/runtime/iTensor.h"
#include "tensorrt_llm/runtime/tllmRuntime.h"
#include "tensorrt_llm/runtime/worldConfig.h"

namespace tensorrt_llm::batch_manager::kv_cache_manager
{
class KVCacheManager;
}

namespace tensorrt_llm::runtime
{

class RuntimeBuffers;

class TransformerBuffers
{
public:
    using TensorPtr = ITensor::SharedPtr;
    using KvCacheManager = batch_manager::kv_cache_manager::KVCacheManager;
    using TensorMap = StringPtrMap<ITensor>;

    TransformerBuffers();

    TransformerBuffers(TllmRuntime const& runtime, runtime::GptModelConfig const& modelConfig,
        runtime::WorldConfig const& worldConfig);

    void reshape(GenerationConfig const& generationConfig, KvCacheManager const* kvCacheManager,
        GptModelConfig const& modelConfig, WorldConfig const& worldConfig);

    void reset(BufferManager& manager);

    TransformerBuffers sliceTo(GenerationConfig const& generationConfig, GptModelConfig const& modelConfig,
        SizeType offset, SizeType batchSize);

    void prepareContextStep(RuntimeBuffers* runtimeBuffers, TensorPtr const& inputIds, TokenIdType const padId,
        BufferManager& manager, KvCacheManager const* kvCacheManager, SizeType firstBatchSlotIdx,
        GptModelConfig const& modelConfig, WorldConfig const& worldConfig);

    void postContextStep(RuntimeBuffers* runtimeBuffers, std::vector<RuntimeBuffers> const& contextBuffers,
        BufferManager& manager, GptModelConfig const& modelConfig, WorldConfig const& worldConfig);

    void prepareNextStep(RuntimeBuffers* runtimeBuffers, SizeType const step, BufferManager& manager,
        KvCacheManager* kvCacheManager, SizeType firstBatchSlotIdx, GptModelConfig const& modelConfig,
        WorldConfig const& worldConfig);

    void getRuntimeBuffers(RuntimeBuffers const* runtimeBuffers, TensorMap& inputBuffers, TensorMap& outputBuffers,
        SizeType const step, TensorPtr const& inputIds, TensorPtr const& commPtrs, GptModelConfig const& modelConfig,
        WorldConfig const& worldConfig) const;

protected:
    void copyAttentionMasks(
        RuntimeBuffers* runtimeBuffers, std::vector<RuntimeBuffers> const& contextBatches, BufferManager& manager);

    void tile(RuntimeBuffers* runtimeBuffers, BufferManager& manager, GptModelConfig const& modelConfig,
        WorldConfig const& worldConfig);

public:
    // engine
    TensorPtr pastKeyValueLengths; // with attention plugin, host tensor
    TensorPtr attentionMask;       // without attention plugin
    TensorPtr positionIds;

    std::vector<TensorPtr> presentKeysVals;
    std::vector<TensorPtr> presentKeysValsAlt; // without attention plugin
    TensorPtr maxAttentionWindows;             // with attention plugin, host tensor
    TensorPtr sinkTokenLengths;                // with attention plugin, host tensor
    TensorPtr kvCacheBlockPoolPointers;
    TensorPtr kvCacheBlockOffsetsHost;         // [batchSize * beamWidth, 2, maxBlocksPerSeq * 2]
    TensorPtr kvCacheBlockOffsetsDevice;       // [batchSize * beamWidth, 2, maxBlocksPerSeq * 2]
};

} // namespace tensorrt_llm::runtime
