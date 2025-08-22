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

#include "decodingInput.h"
#include "decodingOutput.h"
#include "tensorrt_llm/runtime/bufferManager.h"
#include "tensorrt_llm/runtime/iTensor.h"
#include "tensorrt_llm/runtime/speculativeDecodingMode.h"

namespace tensorrt_llm::runtime::decoder
{

class BeamSearchBuffers
{
public:
    explicit BeamSearchBuffers(BufferManager const& bufferManager);

    void reshape(SizeType32 maxBeamWidth, SizeType32 maxSequenceLength);

    // temporary buffers for the beam search + streaming case
    DecodingOutput::BeamHypotheses mOutputBeamHypotheses;
    // will store a slice of DecodingOutput::cumLogProbs
    DecodingOutput::TensorPtr mCumLogProbsTmp;
    SizeType32 mNumSMs;
};

class DecoderState
{
public:
    using TensorPtr = ITensor::SharedPtr;
    using LlmRequestPtr = std::shared_ptr<tensorrt_llm::batch_manager::LlmRequest>;
    using RequestVector = std::vector<LlmRequestPtr>;
    using DecodingInputPtr = std::unique_ptr<DecodingInput>;
    using DecodingOutputPtr = std::unique_ptr<DecodingOutput>;

    DecoderState();

    //! @brief Setup buffers for the decoder excluding speculative decoding.
    void setup(SizeType32 maxNumSequences, SizeType32 maxBeamWidth, SizeType32 maxAttentionWindow,
        SizeType32 sinkTokenLength, SizeType32 maxSequenceLength, nvinfer1::DataType dtype,
        ModelConfig const& modelConfig, WorldConfig const& worldConfig, BufferManager const& bufferManager);

    //! @brief Setup buffers for the cache indirection.
    //! @details This is used for beam search on pipeline parallel ranks without a decoder.
    void setupCacheIndirection(SizeType32 maxNumSequences, SizeType32 maxBeamWidth, SizeType32 maxAttentionWindow,
        BufferManager const& bufferManager);

    //! @brief Setup buffers for speculative decoding.
    void setupSpeculativeDecoding(SpeculativeDecodingMode const& speculativeDecodingMode,
        SizeType32 maxTokensPerEngineStep, nvinfer1::DataType dtype, ModelConfig const& modelConfig,
        WorldConfig const& worldConfig, BufferManager const& bufferManager);

    //! @brief Disable lookahead decoding.
    void disableLookahead(RequestVector const& genRequests);

    //! @returns [batchSize], number of finished sequences per request, on gpu
    [[nodiscard]] TensorPtr getFinishedSum() const;

    //! @returns [batchSize, beamWidth], finished states of type FinishedState, on gpu
    [[nodiscard]] TensorPtr getFinishReasons() const;

    //! @returns [batchSize, maxBeamWidth, maxInputLength + maxNewTokens], contains input token ids and generated token
    //! ids without padding, on gpu. In case of beam search, contains the ungathered data.
    [[nodiscard]] TensorPtr getIds() const;

    //! @param batchIdx index of the batch
    //! @returns [maxBeamWidth, maxInputLength + maxNewTokens], contains input token ids and generated token ids without
    //! padding for request `batchIdx`, on gpu. In case of beam search, contains the ungathered data.
    [[nodiscard]] TensorPtr getIds(SizeType32 batchIdx) const;

    //! @returns [batchSize, maxBeamWidth, maxInputLength + maxNewTokens], only used for beam search. It contains
    //! gathered token ids without padding, on gpu.
    [[nodiscard]] TensorPtr getGatheredIds() const;

    //! @param batchIdx index of the batch
    //! @returns [batchSize, maxBeamWidth, maxInputLength + maxNewTokens], only used for beam search. It contains
    //! gathered token ids without padding for request `batchIdx`, on gpu.
    [[nodiscard]] TensorPtr getGatheredIds(SizeType32 batchIdx) const;

    //! @returns [batchSize, maxBeamWidth, maxInputLength + maxNewTokens], contains parent ids collected during beam
    //! search without padding, on gpu
    [[nodiscard]] TensorPtr getParentIds() const;

    //! @returns [batchSize, maxBeamWidth], cumulative log probabilities (per beam), on gpu
    [[nodiscard]] TensorPtr getCumLogProbs() const;

    //! @returns [maxBeamWidth], cumulative log probabilities (per beam), on gpu
    [[nodiscard]] TensorPtr getCumLogProbs(SizeType32 batchIdx) const;

    //! @returns [batchSize, maxBeamWidth, maxSequenceLength], log probabilities (per beam), on gpu
    [[nodiscard]] TensorPtr getLogProbs() const;

    //! @returns [maxBeamWidth, maxSequenceLength], log probabilities (per beam), on gpu
    [[nodiscard]] TensorPtr getLogProbs(SizeType32 batchIdx) const;

    //! @returns [batchSize, maxBeamWidth], sequence lengths, on gpu
    [[nodiscard]] TensorPtr getSequenceLengths() const;

    //! @param batchIdx index of the batch
    //! @returns [maxBeamWidth], sequence lengths for request `batchIdx`, on gpu
    [[nodiscard]] TensorPtr getSequenceLengths(SizeType32 batchIdx) const;

    //! @brief Get maxTokensPerStep tokens generated in the last forward pass
    //! @returns [maxTokensPerStep, batchSize, maxBeamWidth], tokens generated in last forward pass, on gpu
    [[nodiscard]] TensorPtr getAllNewTokens() const;

    //! @returns [batchSize, maxDraftTokens], predicted draft tokens for next step, on gpu
    [[nodiscard]] TensorPtr getNextDraftTokens() const;

    //! @returns [batchSize], predicted draft tokens lengths for previous step, on gpu
    [[nodiscard]] TensorPtr getPrevDraftTokensLengths() const;

    //! @returns [batchSize], predicted draft tokens lengths for next step, on gpu
    [[nodiscard]] TensorPtr getNextDraftTokensLengths() const;

    //! @returns [batchSize + 1], exclusive sum of accepted draft token lengths, on gpu
    [[nodiscard]] TensorPtr getAcceptedLengthsCumSum() const;

    //! @returns [batchSize, maxAcceptedDraftTokensPerStep], accepted paths packed into continuous tensor, on gpu
    [[nodiscard]] TensorPtr getAcceptedPackedPaths() const;

    [[nodiscard]] SizeType32 getMaxNumSequences() const;

    [[nodiscard]] SizeType32 getMaxBeamWidth() const;

    [[nodiscard]] SizeType32 getMaxSequenceLength() const;

    [[nodiscard]] SizeType32 getMaxDecodingDecoderTokens() const;

    [[nodiscard]] SizeType32 getMaxDecodingEngineTokens() const;

    //! @brief Get the number of tokens for all requests in the batch.
    //! @returns The number of tokens for all requests in the batch.
    [[nodiscard]] std::vector<SizeType32> const& getNumDecodingEngineTokens() const;

    //! @brief Get the number of tokens for a specific request in the batch.
    //! @param batchIdx The index of the request in the batch.
    //! @returns The number of tokens for the specified request.
    [[nodiscard]] SizeType32 getNumDecodingEngineTokens(SizeType32 batchIdx) const;

    //! @brief Set the number of tokens for a specific request in the batch.
    //! @param batchIdx The index of the request in the batch.
    //! @param numTokens The number of tokens for the specified request.
    void setNumDecodingEngineTokens(SizeType32 batchIdx, SizeType32 numTokens);

    //! @brief Get the speculative decoding mode.
    [[nodiscard]] SpeculativeDecodingMode getSpeculativeDecodingMode() const;

    //! @brief Get the explicit draft tokens buffers.
    [[nodiscard]] ExplicitDraftTokensBuffers::Inputs const& getExplicitDraftTokensBuffers() const;

    //! @brief Get the eagle buffers.
    [[nodiscard]] EagleBuffers::Inputs const& getEagleBuffers() const;

    //! @brief Get the lookahead buffers.
    [[nodiscard]] LookaheadDecodingBuffers const& getLookaheadBuffers() const;

    //! @brief Workspace for beam search in streaming mode.
    [[nodiscard]] BeamSearchBuffers const& getBeamSearchBuffers() const;

    //! @brief Set the beam width for a specific request in the batch.
    //! @param batchIdx The index of the request in the batch.
    //! @param beamWidth The beam width for the specified request.
    void setBeamWidth(SizeType32 batchIdx, SizeType32 beamWidth);

    //! @brief Cache indirection input for beam search.
    [[nodiscard]] TensorPtr getCacheIndirectionInput() const;

    //! @brief Cache indirection output for beam search.
    [[nodiscard]] TensorPtr getCacheIndirectionOutput() const;

    //! @brief Get the generation steps for all requests in the batch.
    //! @returns The generation steps for all requests in the batch.
    [[nodiscard]] std::optional<std::vector<SizeType32>> const& getGenerationSteps() const;

    //! @brief Set the generation steps for all requests in the batch.
    //! @param generationSteps The generation steps for all requests in the batch.
    void setGenerationSteps(std::vector<SizeType32> const& generationSteps);

    //! @brief Stateful inputs for the decoder. Allocated for maxNumSequences slots.
    [[nodiscard]] DecodingInput& getJointDecodingInput() const;

    //! @brief Stateful outputs for the decoder. Allocated for maxNumSequences slots.
    [[nodiscard]] DecodingOutput& getJointDecodingOutput() const;

private:
    void setupBuffers(nvinfer1::DataType dtype, BufferManager const& bufferManager);
    void reshapeBuffers(SizeType32 maxBatchSize, SizeType32 maxBeamWidth, SizeType32 maxAttentionWindow,
        SizeType32 sinkTokenLength, SizeType32 maxSequenceLength, ModelConfig const& modelConfig,
        WorldConfig const& worldConfig, BufferManager const& bufferManager);

    void setupCacheIndirectionBuffers(BufferManager const& bufferManager);
    void reshapeCacheIndirectionBuffers(
        SizeType32 maxBatchSize, SizeType32 maxBeamWidth, SizeType32 maxAttentionWindow);

    void setupSpeculativeDecodingBuffers(
        SpeculativeDecodingMode speculativeDecodingMode, nvinfer1::DataType dtype, BufferManager const& bufferManager);
    void reshapeSpeculativeDecodingBuffers(SpeculativeDecodingMode const& speculativeDecodingMode,
        SizeType32 maxTokensPerEngineStep, ModelConfig const& modelConfig, WorldConfig const& worldConfig,
        BufferManager const& bufferManager);

    SizeType32 mMaxNumSequences{};
    SizeType32 mMaxBeamWidth{};
    SizeType32 mMaxSequenceLength{};

    //! @brief Stateful inputs for the decoder. Allocated for maxNumSequences slots.
    DecodingInputPtr mJointDecodingInput;
    //! @brief Stateful outputs for the decoder. Allocated for maxNumSequences slots.
    DecodingOutputPtr mJointDecodingOutput;

    //! @brief Workspace for beam search in streaming mode.
    std::unique_ptr<BeamSearchBuffers> mBeamSearchBuffers;

    // How many tokens for one request can be processed per mDecoders call.
    // It is maxDecodingTokens for non speculative decoding and Draft model approach.
    // Otherwise it is 1.
    SizeType32 mMaxDecodingDecoderTokens{1};
    // How many tokens predicted by the engine for one request.
    // It is maxDecodingTokens. >= 1 for speculative decoding and == 1 for non speculative decoding.
    SizeType32 mMaxDecodingEngineTokens{1};

    //! @brief [batchSize], the num tokens of each request.
    std::vector<SizeType32> mNumDecodingEngineTokens;

    SpeculativeDecodingMode mSpeculativeDecodingMode{SpeculativeDecodingMode::None()};
};

} // namespace tensorrt_llm::runtime::decoder
