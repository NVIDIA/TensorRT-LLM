/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.  All rights reserved.
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

#include "tensorrt_llm/batch_manager/common.h"
#include "tensorrt_llm/batch_manager/rnnStateManager.h"
#include "tensorrt_llm/common/optionalRef.h"
#include "tensorrt_llm/runtime/eagleBuffers.h"
#include "tensorrt_llm/runtime/explicitDraftTokensBuffers.h"
#include "tensorrt_llm/runtime/iTensor.h"
#include "tensorrt_llm/runtime/lookaheadBuffers.h"
#include "tensorrt_llm/runtime/loraManager.h"
#include "tensorrt_llm/runtime/modelConfig.h"
#include "tensorrt_llm/runtime/worldConfig.h"

#include <cstddef>
#include <memory>
#include <optional>
#include <vector>

namespace tensorrt_llm::runtime
{
class TllmRuntime;

namespace decoder
{
class DecoderState;
} // namespace decoder
} // namespace tensorrt_llm::runtime

namespace tensorrt_llm::batch_manager
{

namespace kv_cache_manager
{
class BaseKVCacheManager;
} // namespace kv_cache_manager

class LlmRequest;

class EncoderBuffers;
class LoraBuffers;
class MedusaBuffers;
class PromptTuningBuffers;
class RnnStateBuffers;
class TransformerBuffers;

class RuntimeBuffers
{
public:
    static constexpr auto kLogitsTensorName = "logits";
    static constexpr auto kHiddenStatesOutputTensorName = "hidden_states_output";
    static constexpr auto kHiddenStatesInputTensorName = "hidden_states_input";
    static constexpr auto kInputIdsTensorName = "input_ids";
    static constexpr auto kLastTokenIdsTensorName = "last_token_ids";
    static constexpr auto kHostRequestTypesTensorName = "host_request_types";
    static constexpr auto kContextLengthsTensorName = "context_lengths";
    static constexpr auto kHostContextLengthsTensorName = "host_context_lengths";
    static constexpr auto kSequenceLengthsTensorName = "sequence_length";
    static constexpr auto kPromptEmbeddingTableTensorName = "prompt_embedding_table";
    static constexpr auto kTasksTensorName = "tasks";
    static constexpr auto kPromptVocabSizeTensorName = "prompt_vocab_size";
    static constexpr auto kMRopeRotaryCosSinTensorName = "mrope_rotary_cos_sin";
    static constexpr auto kMRopePositionDeltasTensorName = "mrope_position_deltas";

    using SizeType32 = runtime::SizeType32;
    using TensorPtr = runtime::ITensor::SharedPtr;
    using TensorMap = runtime::ITensor::TensorMap;
    using PeftTable = runtime::LoraManager::PeftTable;
    template <typename T>
    using OptionalRef = tensorrt_llm::common::OptionalRef<T>;

    [[nodiscard]] SizeType32 constexpr getContextIndex() const noexcept
    {
        return contextIndex;
    };

    void constexpr setContextIndex(SizeType32 index) noexcept
    {
        contextIndex = index;
    };

    [[nodiscard]] SizeType32 constexpr getNumContextTokens() const noexcept
    {
        return numContextTokens;
    };

    [[nodiscard]] BatchState getBatchState() const noexcept
    {
        return {numContextRequests, numGenRequests, getNumTokens(), maxKvCacheLengthRounded};
    };

private:
    [[nodiscard]] SizeType32 constexpr getNumRequests() const noexcept
    {
        return numContextRequests + numGenRequests;
    };

    [[nodiscard]] SizeType32 constexpr getNumSequences() const noexcept
    {
        return numContextRequests + numGenSequences;
    };

    [[nodiscard]] SizeType32 constexpr getNumTokens() const noexcept
    {
        return numContextTokens + numGenTokens;
    };

    //! Sizes
    SizeType32 numContextRequests{};
    SizeType32 numGenRequests{};
    SizeType32 numGenSequences{};
    SizeType32 numContextTokens{};
    SizeType32 numGenTokens{};
    SizeType32 numLogits{};
    SizeType32 maxKvCacheLengthRounded{};

    //! General
    TensorPtr inputsIds;

    TensorPtr contextLengthsHost;
    TensorPtr contextLengthsDevice;
    TensorPtr sequenceLengthsHost;

    //! Index of selected runtime context.
    SizeType32 contextIndex{};
    SizeType32 maxContextLength{};

public:
    TensorPtr sequenceLengthsDevice;
    bool promptTableOffloading;

    //! Prompt-Tuning
    std::unique_ptr<PromptTuningBuffers> promptTuningBuffers;

private:
    //! Runtime
    //! Type of host tensor: 0 for context, 1 for generation
    TensorPtr requestTypes;

    TensorPtr lastTokenIdsHost;
    TensorPtr lastTokenIdsDevice;
    TensorPtr logitsIdsHost;

    //! Pipeline-Parallelism
    TensorPtr hiddenStates;

    //! Mrope
    TensorPtr mropeRotaryCosSin;
    TensorPtr mropePositionDeltas;

    //! LoRA
    std::unique_ptr<LoraBuffers> loraBuffers;

public:
    //! Additional buffers depending on model type
    std::unique_ptr<TransformerBuffers> transformerBuffers;
    std::unique_ptr<RnnStateBuffers> rnnStateBuffers;

    //! Encoder-Decoder
    std::unique_ptr<EncoderBuffers> encoderBuffers;

    //! Medusa
    std::unique_ptr<MedusaBuffers> mMedusaBuffers;
    //! Lookahead decoding
    std::unique_ptr<runtime::LookaheadRuntimeBuffers> mLookaheadBuffers;
    //! Explicit draft tokens decoding
    std::unique_ptr<runtime::ExplicitDraftTokensBuffers> mExplicitDraftTokensBuffers;
    //! Eagle decoding
    std::unique_ptr<runtime::EagleBuffers> mEagleBuffers;

    //! Language adapter routing information if language adapter is presented, [numTokens, numLanguages]
    TensorPtr languageAdapterRoutings;

    TensorPtr cacheIndirDecoderIOBatchedCopySrcOffsets;
    TensorPtr cacheIndirDecoderIOBatchedCopyDstOffsets;
    TensorPtr cacheIndirDecoderIOBatchedCopySizes;

    //! Logits
    std::vector<SizeType32> numContextLogits;
    TensorPtr logits;

    //! Helper cache for store generation logits
    struct GenerationLogitsCache
    {
        static constexpr auto kCACHE_LENGTH = 8;

        //! Buffer for logits between steps to prevent from being overwritten
        //! [kCACHE_LENGTH, maxBatchSize * maxBeamWidth, vocabSizePadded]
        TensorPtr logits;
        //! Record the usage offset of the cacheGenerationLogits buffer
        SizeType32 offset{0};

        //! Temporarily store the transposed results of multiple fragment logits, [maxBeamWidth, kCACHE_LENGTH]
        TensorPtr transposedLogits;

        //! Temporarily store logits buffer address during the transposing, [kCACHE_LENGTH]
        TensorPtr fragmentPointerDevice;

        //! Temporarily store logits buffer address during the transposing, [maxBatchSize, kCACHE_LENGTH]
        TensorPtr fragmentPointerHost;

        //! Cycling index for workspace
        size_t workIdx{0};

        void cycleWorkIdx()
        {
            workIdx = (workIdx + 1) % (fragmentPointerHost->getShape().d[0]);
        }

        [[nodiscard]] TensorPtr getFragmentPointerHost()
        {
            TensorPtr slice = runtime::ITensor::slice(fragmentPointerHost, workIdx, 1);
            cycleWorkIdx();
            return slice;
        };
    };

    GenerationLogitsCache generationLogitsCache;

    //! Mapping from batch idx to slot id
    TensorPtr seqSlots;
    TensorPtr seqSlotsDevice;

    //! Explicitly device-copy src offsets to reduce warp stalls in copy batch kernel invocation
    //! [mMaxNumRequests], on gpu
    TensorPtr mCacheIndirDecoderIOBatchedCopySrcOffsetsSliceDevice;
    //! Explicitly device-copy dst offsets to reduce warp stalls in copy batch kernel invocation
    //! [mMaxNumRequests], on gpu
    TensorPtr mCacheIndirDecoderIOBatchedCopyDstOffsetsSliceDevice;
    //! Explicitly device-copy size to reduce warp stalls in copy batch kernel invocation
    //! [mMaxNumRequests], on gpu
    TensorPtr mCacheIndirDecoderIOBatchedCopyCopySizesDevice;

private:
    //! Re-capture cuda graph when max kv cache len of the batch has changed on kKV_CACHE_LEN_CUDA_GRAPH_ROUND_SIZE.
    static SizeType32 constexpr kKV_CACHE_LEN_CUDA_GRAPH_ROUND_SIZE{256};

    TensorMap mAdditionalOutputTensors; // Tensors storing additional output tensors.

    //! Engine I/O
    TensorMap inputMap;
    TensorMap outputMap;

public:
    RuntimeBuffers(SizeType32 maxBatchSize, SizeType32 maxBeamWidth,
        std::vector<SizeType32> const& maxAttentionWindowVec, SizeType32 maxAttentionWindow, SizeType32 sinkTokenLen,
        runtime::TllmRuntime const& runtime, runtime::ModelConfig const& modelConfig,
        runtime::WorldConfig const& worldConfig, executor::DecodingConfig const& decodingConfig,
        bool gatherGenerationLogits, std::optional<SizeType32> maxNumTokens = std::nullopt,
        std::optional<std::vector<executor::AdditionalModelOutput>> const& additionalModelOutputs = std::nullopt,
        bool promptTableOffloading = false);

    RuntimeBuffers(RuntimeBuffers const& other) = delete;
    RuntimeBuffers& operator=(RuntimeBuffers const& other) = delete;
    RuntimeBuffers(RuntimeBuffers&& other) = delete;
    RuntimeBuffers& operator=(RuntimeBuffers&& other) = delete;

    ~RuntimeBuffers();

    std::tuple<SizeType32, TensorMap const&, TensorMap&> prepareStep(RequestVector const& contextRequests,
        RequestVector const& genRequests, SizeType32 maxBeamWidth, SizeType32 maxAttentionWindow,
        runtime::decoder::DecoderState const& decoderState, kv_cache_manager::BaseKVCacheManager* kvCacheManager,
        kv_cache_manager::BaseKVCacheManager* crossKvCacheManager, rnn_state_manager::RnnStateManager* rnnStateManager,
        PeftTable const& peftTable, runtime::TllmRuntime const& runtime, runtime::ModelConfig const& modelConfig,
        runtime::WorldConfig const& worldConfig, bool gatherGenerationLogits, bool trtOverlap,
        OptionalRef<runtime::ITensor const> newOutputTokens = std::nullopt);

    void prepareBuffersForCudaGraph(SizeType32 maxSequenceLength);

    void prepareExplicitDraftTokenBuffers(runtime::ExplicitDraftTokensBuffers::Inputs const& explicitDraftTokensBuffers,
        runtime::TllmRuntime const& runtime, runtime::ModelConfig const& modelConfig,
        runtime::WorldConfig const& worldConfig);

    void prepareEagleBuffers(RequestVector const& contextRequests, RequestVector const& genRequests,
        runtime::EagleBuffers::Inputs const& eagleBuffers, runtime::TllmRuntime const& runtime,
        runtime::ModelConfig const& modelConfig, runtime::WorldConfig const& worldConfig);

private:
    void create(SizeType32 maxBatchSize, SizeType32 maxBeamWidth, std::vector<SizeType32> const& maxAttentionWindowVec,
        SizeType32 maxAttentionWindow, SizeType32 sinkTokenLen, runtime::TllmRuntime const& runtime,
        runtime::ModelConfig const& modelConfig, runtime::WorldConfig const& worldConfig,
        executor::DecodingConfig const& decodingConfig, bool gatherGenerationLogits,
        std::optional<std::vector<executor::AdditionalModelOutput>> const& additionalModelOutputs = std::nullopt);

    //! @brief set max sizes for pre-allocation
    void setMaxBufferSizes(SizeType32 maxBatchSize, SizeType32 maxBeamWidth, runtime::ModelConfig const& modelConfig,
        std::optional<SizeType32> maxNumRuntimeTokens);

    //! @brief set sizes depending on scheduled requests
    void setBufferSizes(RequestVector const& contextRequests, RequestVector const& genRequests);

    void reshape(runtime::TllmRuntime const& runtime, runtime::ModelConfig const& modelConfig,
        runtime::WorldConfig const& worldConfig, bool gatherGenerationLogits);

    void setFromInputs(RequestVector const& contextRequests, RequestVector const& genRequests, SizeType32 maxBeamWidth,
        SizeType32 maxAttentionWindow, runtime::decoder::DecoderState const& decoderState,
        kv_cache_manager::BaseKVCacheManager* kvCacheManagerPtr,
        kv_cache_manager::BaseKVCacheManager* crossKvCacheManagerPtr,
        rnn_state_manager::RnnStateManager* rnnStateManagerPtr, PeftTable const& peftTable,
        runtime::TllmRuntime const& runtime, runtime::ModelConfig const& modelConfig,
        runtime::WorldConfig const& worldConfig, bool trtOverlap, OptionalRef<runtime::ITensor const> newOutputTokens);

    void fillIOMaps(runtime::ModelConfig const& modelConfig, runtime::WorldConfig const& worldConfig);
};

} // namespace tensorrt_llm::batch_manager
