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
#include "tensorrt_llm/runtime/gptModelConfig.h"
#include "tensorrt_llm/runtime/iTensor.h"

namespace tensorrt_llm::batch_manager::kv_cache_manager
{
class KVCacheManager;
}

namespace tensorrt_llm::runtime
{
class TllmRuntime;

class RuntimeBuffers
{
    using TensorPtr = ITensor::SharedPtr;
    using KvCacheManager = batch_manager::kv_cache_manager::KVCacheManager;

public:
    using TensorMap = StringPtrMap<ITensor>;

    // general
    TensorPtr contextLengthsHost;
    TensorPtr contextLengthsDevice;
    TensorPtr inputOffsets;

    // engine
    TensorPtr logits;
    TensorPtr sequenceLengths;     // with attention plugin
    TensorPtr pastKeyValueLengths; // with attention plugin, host tensor
    TensorPtr attentionMask;       // without attention plugin
    TensorPtr positionIds;
    TensorPtr lastTokenIds;
    TensorPtr requestTypes; // with attention plugin and inflight batching. Host tensor

    std::vector<TensorPtr> presentKeysVals;
    std::vector<TensorPtr> presentKeysValsAlt; // without attention plugin
    std::vector<TensorPtr> kvCacheBlockPointers;

    // beam search (shared between engine and decoder)
    TensorPtr cacheIndirectionDecoderInput;
    TensorPtr cacheIndirectionDecoderOutput;

    bool allocated{false};

public:
    class GenerationConfig
    {
    public:
        GenerationConfig() = default;

        GenerationConfig(SizeType batchSize, SizeType beamWidth, SizeType maxInputLength, SizeType maxNewTokens,
            SizeType maxSeqLength)
            : batchSize{batchSize}
            , beamWidth{beamWidth}
            , maxInputLength{maxInputLength}
            , maxNewTokens{maxNewTokens}
            , maxSeqLength{maxSeqLength}
        {
        }

        SizeType batchSize{};
        SizeType beamWidth{};
        SizeType maxInputLength{};
        SizeType maxNewTokens{};
        SizeType maxSeqLength{};

        static GenerationConfig fromInput(ITensor::SharedPtr const& inputIds, ITensor::SharedPtr const& inputLengths,
            bool inputPacked, SizeType beamWidth, SizeType maxSequenceLength,
            std::optional<SizeType> const& maxNewTokensOpt, BufferManager& manager);
    };

public:
    void clear();

    void create(TllmRuntime& runtime, GptModelConfig const& modelConfig);

    void reshape(GenerationConfig const& generationConfig, GptModelConfig const& modelConfig, SizeType worldSize);

    void postContextStep(
        BufferManager& manager, GenerationConfig const& generationConfig, GptModelConfig const& modelConfig);

    void prepareContextStep(TensorPtr const& inputIds, TokenIdType padId, BufferManager& manager,
        KvCacheManager& kvCacheManager, GenerationConfig const& generationConfig, GptModelConfig const& modelConfig);
    TensorPtr prepareNextStep(SizeType step, TensorPtr const& outputIds, BufferManager& manager,
        KvCacheManager& kvCacheManager, GenerationConfig const& generationConfig, GptModelConfig const& modelConfig);

    void getRuntimeBuffers(TensorMap& inputBuffers, TensorMap& outputBuffers, SizeType step, TensorPtr const& inputIds,
        KvCacheManager& kvCacheManager, GptModelConfig const& modelConfig) const;

private:
    // Some tensors are properly tiled, some are just reshaped.
    void tile(BufferManager& manager, GenerationConfig const& generationConfig, GptModelConfig const& modelConfig);
};

} // namespace tensorrt_llm::runtime
