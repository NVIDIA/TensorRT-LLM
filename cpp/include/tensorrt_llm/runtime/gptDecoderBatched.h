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
#include "tensorrt_llm/runtime/cudaEvent.h"
#include "tensorrt_llm/runtime/cudaStream.h"
#include "tensorrt_llm/runtime/generationOutput.h"
#include "tensorrt_llm/runtime/gptDecoder.h"
#include "tensorrt_llm/runtime/iGptDecoderBatched.h"
#include "tensorrt_llm/runtime/iTensor.h"

#include <memory>
#include <vector>

namespace tensorrt_llm::runtime
{

//! GPT decoder class with support for in-flight batching
class GptDecoderBatched : public IGptDecoderBatched
{
public:
    using CudaStreamPtr = std::shared_ptr<CudaStream>;
    using TensorPtr = ITensor::SharedPtr;
    using SharedConstPtr = ITensor::SharedConstPtr;

    enum class ForwardType
    {
        kASYNC,
        kSYNC
    };

    GptDecoderBatched(std::size_t vocabSize, std::size_t vocabSizePadded, CudaStreamPtr stream,
        SpeculativeDecodingMode const& speculativeDecodingMode, nvinfer1::DataType dtype);

    //! Setup the decoder before calling `forward()`
    void setup(executor::DecodingMode const& mode, SizeType32 maxBatchSize, SizeType32 maxBeamWidth,
        SizeType32 maxAttentionWindow, SizeType32 sinkTokenLength, SizeType32 maxSequenceLength,
        SizeType32 maxTokensPerStep, nvinfer1::DataType dtype, ModelConfig const& modelConfig) override;

    void setupExplicitDraftTokens(ExplicitDraftTokensBuffers::Inputs explicitDraftTokensBuffers) override;

    void setupLookahead(LookaheadDecodingBuffers lookaheadDecodingBuffers) override;

    void newBatch(
        GenerationInput const& inputs, GenerationOutput const& outputs, SamplingConfig const& samplingConfig) override;

    void newRequests(std::vector<SizeType32> const& seqSlots, std::vector<decoder_batch::Request> const& requests,
        std::vector<SamplingConfig> const& samplingConfigs) override;

    TokenPtr forwardAsync(decoder_batch::Output& output, decoder_batch::Input const& input) override;

    void forwardSync(decoder_batch::Token const& token) override;

    void forwardSync(
        decoder_batch::Token const& token, decoder_batch::Output& output, decoder_batch::Input const& input) override;

    void forwardAsync(decoder::Output& output, decoder::Input const& input) override;

    void forwardSync() override;

    //! @return [batchSize], indicators of finished requests
    [[nodiscard]] std::vector<bool> getFinished() const override
    {
        return {mFinished.begin(), mFinished.begin() + mActualBatchSize};
    }

    //! @returns [batchSize, beamWidth], FinishedState value, on gpu
    [[nodiscard]] TensorPtr getFinishReasons() const override
    {
        return ITensor::slice(mJointDecodingOutput->finishReasons, 0, mActualBatchSize);
    }

    //! @param batchIdx index of the batch
    //! @returns [maxBeamWidth, maxInputLength + maxNewTokens], contains input token ids and generated token ids without
    //! padding for request `batchIdx`, on gpu. In case of beam search, contains the ungathered data.
    [[nodiscard]] TensorPtr getIds(SizeType32 batchIdx) const override
    {
        TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
        auto tensor = ITensor::slice(mJointDecodingOutput->ids, batchIdx, 1);
        tensor->squeeze(0);
        TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
        return tensor;
    }

    //! @returns [batchSize, maxBeamWidth, maxInputLength + maxNewTokens], contains input token ids and generated token
    //! ids without padding, on gpu. In case of beam search, contains the ungathered data.
    [[nodiscard]] TensorPtr getIds() const override
    {
        TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
        auto tensor = ITensor::slice(mJointDecodingOutput->ids, 0, mActualBatchSize);
        TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
        return tensor;
    }

    //! @param batchIdx index of the batch
    //! @returns [batchSize, maxBeamWidth, maxInputLength + maxNewTokens], only used for beam search. It contains
    //! gathered token ids without padding for request `batchIdx`, on gpu.
    [[nodiscard]] TensorPtr getGatheredIds(SizeType32 batchIdx) const override
    {
        TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
        auto tensor = ITensor::slice(mJointDecodingOutput->gatheredIds, batchIdx, 1);
        tensor->squeeze(0);
        TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
        return tensor;
    }

    //! @returns [batchSize, maxBeamWidth, maxInputLength + maxNewTokens], only used for beam search. It contains
    //! gathered token ids without padding, on gpu
    [[nodiscard]] TensorPtr getGatheredIds() const override
    {
        TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
        auto tensor = ITensor::slice(mJointDecodingOutput->gatheredIds, 0, mActualBatchSize);
        TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
        return tensor;
    }

    //! @brief Gather final beam search results for request `batchSlot`.
    //! Result will only be available after event returned.
    [[nodiscard]] CudaEvent finalize(
        SizeType32 batchSlot, SamplingConfig const& samplingConfig, bool streaming) const override;

    //! @brief Gather final beam search results for all requests.
    void finalize(SamplingConfig const& samplingConfig) const override;

    //! @returns [batchSize, maxBeamWidth, maxInputLength + maxNewTokens], contains parent ids collected during beam
    //! search without padding, on gpu
    [[nodiscard]] TensorPtr getParentIds() const override
    {
        return ITensor::slice(mJointDecodingOutput->parentIds, 0, mActualBatchSize);
    }

    //! @returns [batchSize, maxBeamWidth], cumulative log probabilities (per beam), on gpu
    [[nodiscard]] TensorPtr getCumLogProbs() const override
    {
        return ITensor::slice(mJointDecodingOutput->cumLogProbs, 0, mActualBatchSize);
    }

    //! @returns [maxBeamWidth], cumulative log probabilities (per beam), on gpu
    [[nodiscard]] TensorPtr getCumLogProbs(SizeType32 batchIdx) const override
    {
        auto tensor = ITensor::slice(mJointDecodingOutput->cumLogProbs, batchIdx, 1);
        tensor->squeeze(0);
        return tensor;
    }

    //! @returns [batchSize, maxBeamWidth, maxSequenceLength], log probabilities (per beam), on gpu
    [[nodiscard]] TensorPtr getLogProbs() const override
    {
        return ITensor::slice(mJointDecodingOutput->logProbs, 0, mActualBatchSize);
    }

    //! @returns [maxBeamWidth, maxSequenceLength], log probabilities (per beam), on gpu
    [[nodiscard]] TensorPtr getLogProbs(SizeType32 batchIdx) const override
    {
        auto tensor = ITensor::slice(mJointDecodingOutput->logProbs, batchIdx, 1);
        tensor->squeeze(0);
        return tensor;
    }

    //! @brief Get maxTokensPerStep tokens generated in the last forward pass
    //! @returns [maxTokensPerStep, batchSize, maxBeamWidth], tokens generated in last forward pass, on gpu
    [[nodiscard]] TensorPtr getAllNewTokens() const override
    {
        return mJointDecodingOutput->newTokensSteps;
    }

    //! @brief Get tokens generated in one step of last forward pass
    //! @param iter The iteration within [0; maxTokensPerStep) for which to get the tokens
    //! @returns [batchSize, beamWidth], tokens generated in `iter` (per beam), on gpu
    [[nodiscard]] TensorPtr getNewTokens(SizeType32 iter = 0) const override
    {
        TensorPtr newTokensView = ITensor::slice(mJointDecodingOutput->newTokensSteps, iter, 1);
        newTokensView->squeeze(0);
        return ITensor::slice(newTokensView, 0, mActualBatchSize);
    }

    //! @returns [batchSize], the number of generation steps executed on each request
    [[nodiscard]] std::vector<SizeType32> getNbSteps() const override
    {
        return {mNbSteps.begin(), mNbSteps.begin() + mActualBatchSize};
    }

    //! @returns [1], number of finished sequences, in pinned host memory
    [[nodiscard]] TensorPtr getNbFinished() const override
    {
        return mFinishedSum;
    }

    //! @returns [batchSize, maxDraftTokens], predicted draft tokens for next step, on gpu
    [[nodiscard]] TensorPtr getNextDraftTokens() const override
    {
        return mJointDecodingOutput->speculativeDecodingOutputs->nextDraftTokens;
    }

    //! @returns [batchSize], predicted draft tokens lengths for previous step, on gpu
    [[nodiscard]] TensorPtr getPrevDraftTokensLengths() const override
    {
        return mJointDecodingOutput->speculativeDecodingOutputs->prevDraftTokensLen;
    }

    //! @returns [batchSize], predicted draft tokens lengths for next step, on gpu
    [[nodiscard]] TensorPtr getNextDraftTokensLengths() const override
    {
        return mJointDecodingOutput->speculativeDecodingOutputs->nextDraftTokensLen;
    }

    //! @returns [batchSize + 1], exclusive sum of accepted draft token lengths, on gpu
    [[nodiscard]] TensorPtr getAcceptedLengthsCumSum() const override
    {
        return mJointDecodingOutput->speculativeDecodingOutputs->acceptedLengthsCumSum;
    }

    //! @returns [batchSize, maxAcceptedDraftTokensPerStep], accepted paths packed into continuous tensor, on gpu
    [[nodiscard]] TensorPtr getAcceptedPackedPaths() const override
    {
        return mJointDecodingOutput->speculativeDecodingOutputs->pathsOffsets;
    }

    executor::DecodingMode getDecodingMode() const override
    {
        return mDecodingMode;
    }

private:
    //! @brief Gather final beam search results for request `batchIdx`.
    [[nodiscard]] CudaEvent postProcessRequest(
        SizeType32 batchIdx, SamplingConfig const& samplingConfig, bool streaming) const;

    //! @brief Initialize the decoder at `batchSlot` with a new `request`.
    void newRequest(SizeType32 batchSlot, decoder_batch::Request const& request, SamplingConfig const& samplingConfig);

    //! @brief Allocate buffers for speculative decoding.
    void allocateSpeculativeDecodingBuffers();

    //! @brief Setup buffers for speculative decoding.
    void setupSpeculativeDecoding(ModelConfig const& modelConfig);

    //! @brief Setup buffers for lookahead decoding.
    void setupLookahead(ModelConfig const& modelConfig);

    //! @brief Setups decoder internal tensors for new speculative decoding request
    void newRequestSpeculativeDecoding(
        SizeType32 batchIdx, decoder_batch::Request const& request, SamplingConfig const& samplingConfig);

    //! @brief Setups decoder internal tensors for new request in Draft model Sps mode
    void newRequestDraftTokensExternal(
        SizeType32 batchIdx, decoder_batch::Request const& request, SamplingConfig const& samplingConfig);

    //! @brief Setups decoder internal tensors for new Medusa request
    void newRequestMedusa(SizeType32 batchIdx, decoder_batch::Request const& request);

    //! @brief Setups decoder internal tensors for new Lookahead request
    void newRequestLookahead(SizeType32 batchIdx, decoder_batch::Request const& request);

    //! @brief Setups decoder internal tensors for new Explicit draft tokens request
    void newRequestExplicitDraftTokens(SizeType32 batchIdx, decoder_batch::Request const& request);

    //! @brief Updates finished state on host for all active requests
    void updateFinished(decoder_batch::Token const& token);

    //! @brief Sets inputs for explicit draft tokens.
    void setExplicitDraftTokensInputs(decoder_batch::Input const& input);

    //! @brief Calls decoders for tokens per engine step
    void forwardDispatch(decoder_batch::Output& output, decoder_batch::Input const& input, ForwardType forwardType);

    //! @brief Calls decoder for whole batch
    void forwardDecoder(
        SizeType32 step, decoder_batch::Output& output, decoder_batch::Input const& input, ForwardType forwardType);

private:
    std::size_t const mVocabSize;
    std::size_t const mVocabSizePadded;
    CudaStreamPtr mRuntimeStream;
    CudaStreamPtr mDecoderStream;
    BufferManager mBufferManager;
    TokenPtr mForwardToken;
    CudaEvent mForwardEvent;

    using GptDecoderPtr = std::unique_ptr<IGptDecoder>;
    GptDecoderPtr mDecoder;

    using DecodingInputPtr = std::unique_ptr<DecodingInput>;
    using DecodingOutputPtr = std::unique_ptr<DecodingOutput>;
    DecodingInputPtr mJointDecodingInput;
    DecodingOutputPtr mJointDecodingOutput;

    std::vector<bool> mAcceptByLogits;
    TensorPtr mNumDraftTokens;
    TensorPtr mCurandStates;

    std::vector<SizeType32> mNbSteps;
    std::vector<bool> mFinished;
    TensorPtr mFinishedSum;
    std::vector<SizeType32> mMaxNewTokens;
    std::vector<SizeType32> mBeamWidths;
    std::vector<SizeType32> mNumDecodingEngineTokens;

    TensorPtr mFinishedSteps;     // [maxTokensPerStep, batchSize, beamWidth] finished states of type FinishedState
                                  // for each generated token of maxTokensPerStep, on gpu
    TensorPtr mDraftProbs;        // [batchSize, maxTokensPerEngineStep, beamWidth, vocabPadded], temporary data for
                                  // speculative decoding accept by logits kernel, on gpu
    TensorPtr mTargetProbs;       // [batchSize, maxTokensPerEngineStep, beamWidth, vocabPadded], temporary data for
                                  // speculative decoding accept by logits kernel, on gpu
    TensorPtr mDraftTokenIds;     // [batchSize, maxTokensPerEngineStep], draft token indices, on gpu
    TensorPtr mDraftLogits;       // [batchSize, maxTokensPerEngineStep, vocabSizePadded], draft token logits, on gpu

    TensorPtr mBatchSlotsSetup;   // [maxBatchSize], int32_t, address map, pinned
    TensorPtr mBatchSlotsDecoder; // [maxTokensPerEngineStep, maxBatchSize], int32_t, address map, pinned
    TensorPtr mBatchSlotsAcceptTokens; // [maxTokensPerEngineStep, maxBatchSize], int32_t, address map, pinned
    TensorPtr mBatchSlotsAcceptLogits; // [maxTokensPerEngineStep, maxBatchSize], int32_t, address map, pinned
    TensorPtr mTargetLogitsPtrs;       // [maxBatchSize], float*, pointers to target logits, pinned
    SizeType32 mMaxSequenceLength{};
    SizeType32 mMaxAttentionWindow{};
    SizeType32 mSinkTokenLength{};
    SizeType32 mActualBatchSize{};
    // How many tokens for one request can be processed per mDecoders call.
    // It is maxDecodingTokens for non speculative decoding and Draft model approach.
    // Otherwise it is 1.
    SizeType32 mMaxDecodingDecoderTokens{};
    // How many tokens predicted by the engine for one request.
    // It is maxDecodingTokens. >= 1 for speculative decoding and == 1 for non speculative decoding.
    SizeType32 mMaxDecodingEngineTokens{};

    SpeculativeDecodingMode mSpeculativeDecodingMode;
    executor::DecodingMode mDecodingMode{executor::DecodingMode::Auto()};

    // temporary buffers for the beam search + streaming case
    std::shared_ptr<DecodingOutput::BeamHypotheses> mOutputBeamHypotheses{nullptr};
    // will store a slice of DecodingOutput::cumLogProbs
    DecodingOutput::TensorPtr mCumLogProbsTmp;
    SizeType32 mNumSMs;
};
} // namespace tensorrt_llm::runtime
