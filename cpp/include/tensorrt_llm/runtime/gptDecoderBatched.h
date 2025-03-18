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
#include "tensorrt_llm/runtime/gptDecoder.h"
#include "tensorrt_llm/runtime/iGptDecoderBatched.h"
#include "tensorrt_llm/runtime/iTensor.h"
#include "tensorrt_llm/runtime/worldConfig.h"

#include <memory>
#include <vector>

namespace tensorrt_llm::batch_manager
{
class LlmRequest;
} // namespace tensorrt_llm::batch_manager

namespace tensorrt_llm::runtime
{

//! GPT decoder class with support for in-flight batching
class GptDecoderBatched : public IGptDecoderBatched
{
public:
    using CudaStreamPtr = std::shared_ptr<CudaStream>;
    using LlmRequestPtr = std::shared_ptr<tensorrt_llm::batch_manager::LlmRequest>;
    using RequestVector = std::vector<LlmRequestPtr>;
    using TensorPtr = ITensor::SharedPtr;
    using SharedConstPtr = ITensor::SharedConstPtr;

    enum class ForwardType
    {
        kASYNC,
        kSYNC
    };

    GptDecoderBatched(
        CudaStreamPtr stream, SpeculativeDecodingMode const& speculativeDecodingMode, nvinfer1::DataType dtype);

    void setup(executor::DecodingMode const& mode, SizeType32 maxBatchSize, SizeType32 maxBeamWidth,
        SizeType32 maxAttentionWindow, SizeType32 sinkTokenLength, SizeType32 maxSequenceLength,
        SizeType32 maxTokensPerStep, nvinfer1::DataType dtype, ModelConfig const& modelConfig,
        WorldConfig const& worldConfig) override;

    void setupExplicitDraftTokens(ExplicitDraftTokensBuffers::Inputs explicitDraftTokensBuffers) override;
    void setupEagle(EagleBuffers::Inputs eagleBuffers) override;
    void setupLookahead(LookaheadDecodingBuffers lookaheadDecodingBuffers) override;
    void disableLookahead(
        SizeType32 maxBatchSize, RequestVector const& genRequests, TensorPtr const& batchSlots) override;

    DecoderFinishedEventPtr forwardAsync(decoder_batch::Output& output, decoder_batch::Input const& input) override;
    void forward(decoder_batch::Output& output, decoder_batch::Input const& input) override;

    //! @returns [batchSize], number of finished sequences per request, on gpu
    [[nodiscard]] TensorPtr getFinishedSum() const override
    {
        return ITensor::slice(mJointDecodingOutput->finishedSum, 0, mActualBatchSize);
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

    //! @brief Gather final beam search results for request `batchSlot`.
    //! Result will only be available after event returned.
    [[nodiscard]] CudaEvent finalize(
        SizeType32 batchSlot, SamplingConfig const& samplingConfig, bool streaming) const override;

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

    DecodingInput& getJointDecodingInput() const
    {
        return *mJointDecodingInput.get();
    }

    DecodingOutput& getJointDecodingOutput() const
    {
        return *mJointDecodingOutput.get();
    }

    CudaStreamPtr getDecoderStream() const
    {
        return mDecoderStream;
    }

    SpeculativeDecodingMode getSpeculativeDecodingMode() const
    {
        return mSpeculativeDecodingMode;
    }

    SizeType32 getMaxDecodingEngineTokens() const
    {
        return mMaxDecodingEngineTokens;
    }

    TensorPtr getFinishedSteps() const
    {
        return mFinishedSteps;
    }

    IGptDecoder& getUnderlyingDecoder() const
    {
        return *mDecoder.get();
    }

    [[nodiscard]] BufferManager const& getBufferManager() const
    {
        return mBufferManager;
    }

    [[nodiscard]] SizeType32 getActualBatchSize() const
    {
        return mActualBatchSize;
    }

    void setActualBatchSize(SizeType32 actualBatchSize)
    {
        mActualBatchSize = actualBatchSize;
    }

    [[nodiscard]] SizeType32 getMaxSequenceLength() const
    {
        return mMaxSequenceLength;
    }

private:
    //! @brief Allocate buffers for speculative decoding.
    void allocateSpeculativeDecodingBuffers(nvinfer1::DataType dtype);

    //! @brief Setup buffers for speculative decoding.
    void setupSpeculativeDecoding(ModelConfig const& modelConfig);

    //! @brief Setup buffers for lookahead decoding.
    void setupLookahead(ModelConfig const& modelConfig);

    //! @brief Sets inputs for explicit draft tokens.
    void setExplicitDraftTokensInputs(decoder_batch::Input const& input);

    //! @brief Sets inputs for eagle decoding.
    void setEagleInputs(decoder_batch::Input const& input);

    //! @brief Calls decoders for tokens per engine step
    void forwardDispatch(decoder_batch::Output& output, decoder_batch::Input const& input, ForwardType forwardType);

    //! @brief Prepare Input and Output for decoder step
    void prepareForward(SizeType32 step, decoder_batch::Output& output, decoder_batch::Input const& input);

    //! @brief Calls decoder for whole batch
    void forwardDecoder(DecodingOutput& output, DecodingInput const& input, ForwardType forwardType);

private:
    CudaStreamPtr mRuntimeStream;
    CudaStreamPtr mDecoderStream;
    BufferManager mBufferManager;

    using GptDecoderPtr = std::unique_ptr<IGptDecoder>;
    GptDecoderPtr mDecoder;

    using DecodingInputPtr = std::unique_ptr<DecodingInput>;
    using DecodingOutputPtr = std::unique_ptr<DecodingOutput>;
    DecodingInputPtr mJointDecodingInput;
    DecodingOutputPtr mJointDecodingOutput;

    TensorPtr mFinishedSteps; // [maxTokensPerStep, batchSize, beamWidth] finished states of type FinishedState
                              // for each generated token of maxTokensPerStep, on gpu

    SizeType32 mMaxSequenceLength{};
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
