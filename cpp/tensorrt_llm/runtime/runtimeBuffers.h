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
#include "tensorrt_llm/runtime/gptModelConfig.h"
#include "tensorrt_llm/runtime/iTensor.h"
#include "tensorrt_llm/runtime/promptTuningParams.h"
#include "tensorrt_llm/runtime/worldConfig.h"

#include <array>
#include <vector>

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

    class GenerationConfig
    {
    public:
        GenerationConfig() = default;

        explicit GenerationConfig(SizeType batchSize, SizeType beamWidth, SizeType maxInputLength,
            SizeType maxAttentionWindow, SizeType sinkTokenLength, SizeType maxSeqLength,
            SizeType inputLengthSum = SizeType(0))
            : batchSize{batchSize}
            , beamWidth{beamWidth}
            , maxInputLength{maxInputLength}
            , maxAttentionWindow{maxAttentionWindow}
            , sinkTokenLength{sinkTokenLength}
            , maxSeqLength{maxSeqLength}
            , inputLengthSum{inputLengthSum}
        {
        }

        SizeType batchSize{};
        SizeType beamWidth{};
        SizeType maxInputLength{};
        SizeType maxAttentionWindow{};
        SizeType sinkTokenLength{};
        SizeType maxSeqLength{};
        SizeType inputLengthSum{}; // Initialized only if inputPacked is set to true in fromInput.

        static GenerationConfig fromInput(ITensor const& inputIds, ITensor const& inputLengths, bool inputPacked,
            SizeType beamWidth, SizeType maxAttentionWindow, SizeType sinkTokenLength, SizeType maxSequenceLength);
    };

public:
    GenerationConfig generationConfig{};
    std::array<TensorMap, 2> inputBuffers{};
    std::array<TensorMap, 2> outputBuffers{};

    // general
    TensorPtr contextLengthsHost;
    TensorPtr contextLengthsDevice;

    // engine
    TensorPtr logits;
    TensorPtr sequenceLengths;     // with attention plugin
    TensorPtr pastKeyValueLengths; // with attention plugin, host tensor
    TensorPtr attentionMask;       // without attention plugin
    TensorPtr positionIds;
    TensorPtr lastTokenIds;
    TensorPtr requestTypes;        // with attention plugin. Host tensor
    TensorPtr allGenerationLogits; // pre-allocate a buffer to save all generation logits, device tensor
    TensorPtr originalLogitsPtr;   // Record the initially created buffer address.
                                 // `logits` will point to new buffer (i.e. `allGenerationLogits`) for each iteration to
                                 // avoid overwrite during gather context/generation logits.
                                 // `originalLogitsPtr` could reset the `logits` point to the initially buffer when
                                 // microBatch call `buffer.reshape()`. This could avoid next microBatch's `logits`
                                 // still point to `allGenerationLogits` and bring overwrite conflict.

    std::vector<TensorPtr> presentKeysVals;
    std::vector<TensorPtr> presentKeysValsAlt;  // without attention plugin
    std::vector<TensorPtr> maxAttentionWindows; // with attention plugin, host tensor
    TensorPtr sinkTokenLengths;                 // with attention plugin, host tensor
    TensorPtr kvCacheBlockPointersHost;         // [numLayers, batchSize * beamWidth, 2, maxBlocksPerSeq * 2]
    TensorPtr kvCacheBlockPointersDevice;       // [numLayers, batchSize * beamWidth, 2, maxBlocksPerSeq * 2]

    // References to tmp buffers
    TensorPtr newTokens;
    TensorPtr outputIds;
    TensorPtr outputLengths;

    // beam search (shared between engine and decoder)
    TensorPtr cacheIndirectionDecoderInput;
    TensorPtr cacheIndirectionDecoderOutput;

    // decoder
    TensorPtr nbFinished;

    // Log probs
    TensorPtr cumLogProbs;
    TensorPtr logProbs;

    // pipeline parallelism
    TensorPtr hiddenStates;

    // Prompt tuning
    PromptTuningParams promptTuningParams;
    TensorPtr promptTuningTasksHost; // Tensor to hold tasks on host

    // generation logit pointer list
    std::shared_ptr<std::vector<TensorPtr>> generationLogitsFragments;
    TensorPtr
        cacheGenerationFragmentPointerDevice; // device pointer array, used in merge generation logits fragments kernel
    TensorPtr
        cacheGenerationFragmentPointerHost;   // host pointer array, used in merge generation logits fragments kernel

    bool allocated{false};

public:
    void clear();
    void clearTensorMaps();

    void create(TllmRuntime& runtime, GptModelConfig const& modelConfig, WorldConfig const& worldConfig);

    void initFromInput(ITensor const& inputIds, TensorPtr const& inputLengths, bool inputPacked, SizeType beamWidth,
        SizeType maxAttentionWindow, SizeType sinkTokenLength, SizeType maxSequenceLength, BufferManager& manager);

    //! \brief Reshape buffers based on current GenerationConfig
    void reshape(GptModelConfig const& modelConfig, WorldConfig const& worldConfig);

    void reset(BufferManager& manager);

    std::vector<RuntimeBuffers> split(
        SizeType contextBatchSize, GptModelConfig const& modelConfig, WorldConfig const& worldConfig);

    void postContextStep(std::vector<RuntimeBuffers> const& contextBuffers, BufferManager& manager,
        GptModelConfig const& modelConfig, WorldConfig const& worldConfig);

    void prepareContextStep(TensorPtr const& inputIds, TokenIdType padId, BufferManager& manager,
        KvCacheManager const* kvCacheManager, SizeType firstBatchSlotIdx, GptModelConfig const& modelConfig,
        WorldConfig const& worldConfig);
    TensorPtr prepareNextStep(SizeType step, BufferManager& manager, KvCacheManager* kvCacheManager,
        SizeType firstBatchSlotIdx, GptModelConfig const& modelConfig, WorldConfig const& worldConfig);

    void getRuntimeBuffers(TensorMap& inputBuffers, TensorMap& outputBuffers, SizeType const step,
        TensorPtr const& inputIds, TensorPtr const& commPtrs, GptModelConfig const& modelConfig,
        WorldConfig const& worldConfig) const;

private:
    void gatherLastTokenLogits(
        BufferManager& manager, GptModelConfig const& modelConfig, WorldConfig const& worldConfig);

    void copyAttentionMasks(std::vector<RuntimeBuffers> const& contextBatches, BufferManager& manager);

    // Some tensors are properly tiled, some are just reshaped.
    void tile(BufferManager& manager, GptModelConfig const& modelConfig, WorldConfig const& worldConfig);

    static std::vector<SizeType> getPositionIdsContextPhaseGlm(const SizeType& batchSize,
        const SizeType& maxInputLength, const SizeType* pInputLengths, const bool useGptAttentionPlugin,
        const bool usePackedInput);

    static std::vector<SizeType> getPositionIdsGenerationPhaseGlm(const SizeType& batchSize, const SizeType& beamSize,
        const SizeType& step, const SizeType* pInputLengths, const bool useGptAttentionPlugin,
        const bool usePackedInput);
};

} // namespace tensorrt_llm::runtime
