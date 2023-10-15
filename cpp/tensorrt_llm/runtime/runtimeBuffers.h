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

#include "ipcUtils.h"
#include "tensorrt_llm/runtime/bufferManager.h"
#include "tensorrt_llm/runtime/gptModelConfig.h"
#include "tensorrt_llm/runtime/iTensor.h"
#include "tensorrt_llm/runtime/worldConfig.h"

namespace tensorrt_llm::batch_manager::kv_cache_manager
{
class KVCacheManager;
}

namespace tensorrt_llm::runtime
{
class TllmRuntime;

class RuntimeBuffers
{
protected:
    using TensorPtr = ITensor::SharedPtr;
    using KvCacheManager = batch_manager::kv_cache_manager::KVCacheManager;

public:
    using TensorMap = StringPtrMap<ITensor>;

    // general
    TensorPtr contextLengthsHost;
    TensorPtr contextLengthsDevice;
    TensorPtr inputOffsets;             // helper for packed input
    TensorPtr kvCacheBlockPointersHost; // [numLayers, batchSize * beamWidth, 2, maxBlocksPerSeq * 2]

    // engine
    TensorPtr logits;
    TensorPtr sequenceLengths;     // with attention plugin
    TensorPtr pastKeyValueLengths; // with attention plugin, host tensor
    TensorPtr attentionMask;       // without attention plugin
    TensorPtr positionIds;
    TensorPtr lastTokenIds;
    TensorPtr requestTypes; // with attention plugin. Host tensor

    std::vector<TensorPtr> presentKeysVals;
    std::vector<TensorPtr> presentKeysValsAlt; // without attention plugin
    TensorPtr kvCacheBlockPointersDevice;      // [numLayers, batchSize * beamWidth, 2, maxBlocksPerSeq * 2]

    // beam search (shared between engine and decoder)
    TensorPtr cacheIndirectionDecoderInput;
    TensorPtr cacheIndirectionDecoderOutput;

    // decoder
    TensorPtr nbFinished;

    // pipeline parallelism
    TensorPtr hiddenStates;

    // tensor parallelism
    TensorPtr commPtrs;

    bool allocated{false};

    std::vector<std::shared_ptr<IpcMemory>> mIpcMemoryHandles;

public:
    class GenerationConfig
    {
    public:
        GenerationConfig() = default;

        GenerationConfig(SizeType batchSize, SizeType beamWidth, SizeType maxInputLength, SizeType maxNewTokens,
            SizeType maxSeqLength, SizeType inputLengthSum = SizeType(0))
            : batchSize{batchSize}
            , beamWidth{beamWidth}
            , maxInputLength{maxInputLength}
            , maxNewTokens{maxNewTokens}
            , maxSeqLength{maxSeqLength}
            , inputLengthSum{inputLengthSum}
        {
        }

        SizeType batchSize{};
        SizeType beamWidth{};
        SizeType maxInputLength{};
        SizeType maxNewTokens{};
        SizeType maxSeqLength{};
        SizeType inputLengthSum{}; // Initialized only if inputPacked is set to true in fromInput.

        static RuntimeBuffers::GenerationConfig fromInput(ITensor const& inputIds, ITensor const& inputLengths,
            bool const inputPacked, SizeType const beamWidth, SizeType const maxSequenceLength,
            std::optional<SizeType> const& maxNewTokensOpt);
    };

public:
    void clear();

    void create(TllmRuntime& runtime, GptModelConfig const& modelConfig, WorldConfig const& worldConfig);

    void initContextLengths(TensorPtr const& inputLengths, BufferManager& manager);

    void reshape(
        GenerationConfig const& generationConfig, GptModelConfig const& modelConfig, WorldConfig const& worldConfig);

    void postContextStep(BufferManager& manager, GenerationConfig const& generationConfig,
        GptModelConfig const& modelConfig, WorldConfig const& worldConfig);

    void prepareContextStep(TensorPtr const& inputIds, TokenIdType padId, BufferManager& manager,
        KvCacheManager const* kvCacheManager, GenerationConfig const& generationConfig,
        GptModelConfig const& modelConfig, WorldConfig const& worldConfig);
    TensorPtr prepareNextStep(SizeType step, TensorPtr const& outputIds, BufferManager& manager,
        KvCacheManager* kvCacheManager, GenerationConfig const& generationConfig, GptModelConfig const& modelConfig,
        WorldConfig const& worldConfig);

    void getRuntimeBuffers(TensorMap& inputBuffers, TensorMap& outputBuffers, SizeType step, TensorPtr const& inputIds,
        GptModelConfig const& modelConfig, WorldConfig const& worldConfig) const;

    void createCustomAllReduceWorkspace(SizeType maxBatchSize, SizeType maxBeamWidth, SizeType maxSequenceLength,
        SizeType hiddenSize, WorldConfig const& worldConfig, BufferManager& manager);

private:
    void gatherLastTokenLogits(BufferManager& manager, GenerationConfig const& generationConfig,
        GptModelConfig const& modelConfig, WorldConfig const& worldConfig);

    // Some tensors are properly tiled, some are just reshaped.
    void tile(BufferManager& manager, GenerationConfig const& generationConfig, GptModelConfig const& modelConfig,
        WorldConfig const& worldConfig);

    static std::vector<SizeType> getPositionIdsContextPhaseGlm(
        SizeType batchSize, SizeType maxInputLength, SizeType const* pInputLengths, bool useGptAttentionPlugin);

    static std::vector<SizeType> getPositionIdsGenerationPhaseGlm(SizeType batchSize, SizeType beamSize, SizeType step,
        SizeType const* pInputLengths, bool useGptAttentionPlugin);
};

} // namespace tensorrt_llm::runtime
