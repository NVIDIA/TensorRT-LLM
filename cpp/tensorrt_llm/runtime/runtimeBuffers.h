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
#include "tensorrt_llm/runtime/generationConfig.h"
#include "tensorrt_llm/runtime/iTensor.h"
#include "tensorrt_llm/runtime/modelConfig.h"
#include "tensorrt_llm/runtime/promptTuningParams.h"
#include "tensorrt_llm/runtime/rnnStateBuffers.h"
#include "tensorrt_llm/runtime/transformerBuffers.h"
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

public:
    GenerationConfig generationConfig{};
    std::array<TensorMap, 2> inputBuffers{};
    std::array<TensorMap, 2> outputBuffers{};

    // general
    TensorPtr contextLengthsHost;
    TensorPtr contextLengthsDevice;

    // engine
    TensorPtr logits;
    TensorPtr sequenceLengths; // with attention plugin
    TensorPtr lastTokenIds;
    TensorPtr requestTypes;    // Host tensor, with attention plugin for transformer-based model or for RNN based-model
    TensorPtr allGenerationLogits; // pre-allocate a buffer to save all generation logits, device tensor
    TensorPtr originalLogitsPtr;   // Record the initially created buffer address.
    // `logits` will point to new buffer (i.e. `allGenerationLogits`) for each iteration to
    // avoid overwrite during gather context/generation logits.
    // `originalLogitsPtr` could reset the `logits` point to the initially buffer when
    // microBatch call `buffer.reshape()`. This could avoid next microBatch's `logits`
    // still point to `allGenerationLogits` and bring overwrite conflict.

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

    // Transformer model buffer
    std::optional<TransformerBuffers> transformerBuffers;

    // Prompt tuning
    PromptTuningParams promptTuningParams;
    TensorPtr promptTuningTasksHost; // Tensor to hold tasks on host

    // RNN model buffer
    std::optional<RnnStateBuffers> rnnStateBuffers;

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

    void create(TllmRuntime const& runtime, ModelConfig const& modelConfig, WorldConfig const& worldConfig);

    void initFromInput(ITensor const& inputIds, TensorPtr const& inputLengths, bool inputPacked, SizeType32 beamWidth,
        SizeType32 maxAttentionWindow, SizeType32 sinkTokenLength, SizeType32 maxSequenceLength,
        BufferManager& manager);

    //! \brief Reshape buffers based on current GenerationConfig
    void reshape(ModelConfig const& modelConfig, WorldConfig const& worldConfig);

    void reset(BufferManager& manager);

    std::vector<RuntimeBuffers> split(
        SizeType32 contextBatchSize, ModelConfig const& modelConfig, WorldConfig const& worldConfig);

    void postContextStep(std::vector<RuntimeBuffers> const& contextBuffers, BufferManager& manager,
        ModelConfig const& modelConfig, WorldConfig const& worldConfig);

    void prepareContextStep(TensorPtr const& inputIds, TokenIdType padId, BufferManager& manager,
        KvCacheManager const* kvCacheManager, SizeType32 firstBatchSlotIdx, ModelConfig const& modelConfig,
        WorldConfig const& worldConfig);
    TensorPtr prepareNextStep(SizeType32 step, BufferManager& manager, KvCacheManager* kvCacheManager,
        SizeType32 firstBatchSlotIdx, ModelConfig const& modelConfig, WorldConfig const& worldConfig);

    void getRuntimeBuffers(TensorMap& inputBuffers, TensorMap& outputBuffers, SizeType32 const step,
        TensorPtr const& inputIds, TensorPtr const& commPtrs, ModelConfig const& modelConfig,
        WorldConfig const& worldConfig) const;

    void gatherLastTokenLogits(BufferManager& manager, ModelConfig const& modelConfig, WorldConfig const& worldConfig);
};

} // namespace tensorrt_llm::runtime
