/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "tensorrt_llm/batch_manager/createNewDecoderRequests.h"
#include "tensorrt_llm/runtime/decodingInput.h"
#include "tensorrt_llm/runtime/decodingOutput.h"
#include "tensorrt_llm/runtime/gptDecoderBatched.h"
#include "tensorrt_llm/runtime/runtimeKernels.h"
#include "tensorrt_llm/runtime/speculativeDecodingMode.h"
#include "tensorrt_llm/runtime/utils/speculativeChoicesUtils.h"

using namespace tensorrt_llm::runtime;

namespace tc = tensorrt_llm::common;
namespace tk = tensorrt_llm::kernels;

namespace tensorrt_llm::batch_manager
{

void CreateNewDecoderRequests::operator()(TensorPtr const& batchSlots,
    std::vector<runtime::decoder_batch::Request> const& requests, std::vector<SamplingConfig> const& samplingConfigs,
    runtime::ModelConfig const& modelConfig, GptDecoderBatched& decoder, CudaStream const& runtimeStream,
    SizeType32 maxSequenceLength) const
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    auto batchSlotsRange = BufferRange<SizeType32>(*batchSlots);
    auto const localBatchSize = batchSlots->getSize();
    for (size_t bi = 0; bi < localBatchSize; ++bi)
    {
        newRequest(batchSlotsRange[bi], requests[bi], samplingConfigs[bi], modelConfig, decoder, runtimeStream,
            maxSequenceLength);
    }

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void CreateNewDecoderRequests::newRequest(SizeType32 batchSlot, runtime::decoder_batch::Request const& request,
    SamplingConfig const& samplingConfig, runtime::ModelConfig const& modelConfig, GptDecoderBatched& decoder,
    CudaStream const& runtimeStream, SizeType32 maxSequenceLength) const
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    TLLM_CHECK(batchSlot >= 0);

    auto const& decoderStream = decoder.getDecoderStream();
    BufferManager manager{decoderStream};

    auto& decoderState = decoder.getDecoderState();

    auto const& jointOutputIdsShape = decoderState.getJointDecodingOutput().ids->getShape();
    auto const batchSize = jointOutputIdsShape.d[0];
    TLLM_CHECK(0 <= batchSize && batchSlot < batchSize);
    auto const maxBeamWidth = jointOutputIdsShape.d[1];
    auto const beamWidth = samplingConfig.beamWidth;
    TLLM_CHECK_WITH_INFO(beamWidth <= maxBeamWidth,
        tc::fmtstr("Beam width (%d) must be smaller than maxBeamWidth (" FMT_DIM ") passed to decoder setup function.",
            beamWidth, maxBeamWidth));
    auto const& requestIds = request.ids;
    auto const inputLength = request.inputLen;
    auto const numDecodingEngineTokens = request.generatedTokensPerEngineStep;
    auto const numDecodingDraftEngineTokens = numDecodingEngineTokens - 1;
    auto const maxNewTokens
        = request.maxNewTokens.value_or(maxSequenceLength - inputLength - numDecodingDraftEngineTokens);

    TLLM_CHECK_WITH_INFO(inputLength + maxNewTokens + numDecodingDraftEngineTokens <= maxSequenceLength,
        tc::fmtstr(
            "Input length (%d) + max new tokens (%d) + draft tokens (%d) must be less than max sequence length (%d).",
            inputLength, maxNewTokens, numDecodingDraftEngineTokens, maxSequenceLength));
    TLLM_CHECK(requestIds->getDataType() == TRTDataType<TokenIdType>::value);
    auto const endId = request.endId.value_or(-1);

    // input
    auto& dJointInput = decoderState.getJointDecodingInput();

    dJointInput.beamWidths.at(batchSlot) = beamWidth;
    decoderState.setNumDecodingEngineTokens(batchSlot, numDecodingEngineTokens);

    TensorPtr endIdTensorPtr{ITensor::slice(constPointerCast(dJointInput.endIds), batchSlot, 1)};
    runtime::kernels::invokeFill(*endIdTensorPtr, endId, *decoderStream);

    TensorPtr embeddingBiasSlice = ITensor::slice(constPointerCast(dJointInput.embeddingBias), batchSlot, 1);
    if (request.embeddingBias)
    {
        TLLM_CHECK(request.embeddingBias->getShape().nbDims == 2);
        TLLM_CHECK(request.embeddingBias->getShape().d[0] == 1);
        TLLM_CHECK_WITH_INFO(request.embeddingBias->getShape().d[1] == modelConfig.getVocabSize(),
            "The embedding bias shape is not as expected. Expected last dimension to be same as vocab size: %d.",
            modelConfig.getVocabSize());
        manager.copy(*request.embeddingBias, *embeddingBiasSlice);
    }
    else
    {
        manager.setZero(*embeddingBiasSlice);
    }

    auto setupWords = [](std::vector<runtime::ITensor::SharedPtr>& jointWordsLists, TensorPtr const& requestWordsList,
                          SharedConstPtr& jointWordsPtrs, SharedConstPtr& jointWordsLens, SizeType32& jointMaxWordsLen,
                          SizeType32 batchSlot)
    {
        if (requestWordsList)
        {
            auto const wordsLen = requestWordsList->getShape().d[1];
            BufferRange<int32_t*>(*constPointerCast(jointWordsPtrs))[batchSlot]
                = runtime::bufferCast<TokenIdType>(*requestWordsList);
            runtime::bufferCast<SizeType32>(*constPointerCast(jointWordsLens))[batchSlot] = wordsLen;
            // FIXME: this is monotonically growing size
            jointMaxWordsLen = std::max(static_cast<SizeType32>(wordsLen), jointMaxWordsLen);

            // NOTE: jointWordsList is not used in gptDecoder, but required to keep <name>WordsList's
            // memory allocated
            jointWordsLists[batchSlot] = requestWordsList;
        }
        else
        {
            runtime::bufferCast<SizeType32>(*constPointerCast(jointWordsLens))[batchSlot] = 0;
        }
    };

    setupWords(dJointInput.stopWordsLists, request.stopWordsList, dJointInput.stopWordsPtrs, dJointInput.stopWordsLens,
        dJointInput.maxStopWordsLen, batchSlot);

    setupWords(dJointInput.badWordsLists, request.badWordsList, dJointInput.badWordsPtrs, dJointInput.badWordsLens,
        dJointInput.maxBadWordsLen, batchSlot);

    TensorPtr sequenceLimitLength{ITensor::slice(constPointerCast(dJointInput.sequenceLimitLength), batchSlot, 1)};
    runtime::kernels::invokeFill(*sequenceLimitLength, inputLength + maxNewTokens, *decoderStream);

    TensorPtr inputLengths{ITensor::slice(constPointerCast(dJointInput.lengths), batchSlot, 1)};
    runtime::kernels::invokeFill(*inputLengths, inputLength, *decoderStream);

    // output
    auto& dJointOutput = decoderState.getJointDecodingOutput();
    auto const outputIdsShape = ITensor::makeShape({1, beamWidth, maxSequenceLength});

    auto finishedSum = ITensor::slice(dJointOutput.finishedSum, batchSlot, 1);
    manager.setZero(*finishedSum);

    for (SizeType32 ti = 0; ti < decoderState.getMaxDecodingEngineTokens(); ++ti)
    {
        TensorPtr newTokensStepView = ITensor::slice(dJointOutput.newTokensSteps, ti, 1);
        newTokensStepView->squeeze(0);
        auto newTokensVec = ITensor::slice(newTokensStepView, batchSlot, 1);
        manager.setZero(*newTokensVec);
    }

    // FIXME: we call setZero mMaxDecodingEngineTokens times for only 1 element
    for (SizeType32 ti = 0; ti < decoderState.getMaxDecodingEngineTokens(); ++ti)
    {
        TensorPtr finishedStepsView = ITensor::slice(decoderState.getFinishedSteps(), ti, 1);
        finishedStepsView->squeeze(0);
        TensorPtr finishedSteps = ITensor::slice(finishedStepsView, batchSlot, 1);
        if (ti < numDecodingEngineTokens)
        {
            manager.setZero(*finishedSteps);
        }
        else
        {
            runtime::kernels::invokeFill(
                *finishedSteps, tk::FinishedState::skipDecoding().toUnderlying(), *decoderStream);
        }
    }

    // cumLogProb is mandatory for beamWidth > 1
    if ((samplingConfig.cumLogProbs.has_value() && samplingConfig.cumLogProbs->at(0)) || beamWidth > 1)
    {
        auto cumLogProbs = ITensor::slice(dJointOutput.cumLogProbs, batchSlot, 1);
        manager.setZero(*cumLogProbs);
    }

    if (samplingConfig.outputLogProbs.has_value() && samplingConfig.outputLogProbs->at(0))
    {
        auto logProbs = ITensor::slice(dJointOutput.logProbs, batchSlot, 1);
        manager.setZero(*logProbs);
    }

    if (beamWidth > 1)
    {
        TensorPtr cumLogProbs = ITensor::slice(dJointOutput.cumLogProbs, batchSlot, 1);
        runtime::kernels::invokeFill(
            *IBuffer::slice(cumLogProbs, 1, beamWidth - 1), DecodingOutput::kNegativeInfinity, *decoderStream);

        auto parentIds = ITensor::slice(dJointOutput.parentIds, batchSlot, 1);
        parentIds->reshape(outputIdsShape);
        manager.setZero(*parentIds);

        auto beamHypotheses = dJointOutput.beamHypotheses.slice(batchSlot, 1);
        beamHypotheses.init(manager, endId);
    }

    // Speculative execution
    if (numDecodingEngineTokens > 1 || decoderState.getSpeculativeDecodingMode().isDraftTokensExternal())
    {
        TLLM_CHECK(beamWidth == 1);
        newRequestSpeculativeDecoding(batchSlot, request, samplingConfig, modelConfig,
            decoderState.getJointDecodingInput(), decoderState.getJointDecodingOutput(), runtimeStream, *decoderStream,
            decoderState.getSpeculativeDecodingMode(), decoderState.getMaxDecodingEngineTokens());
    }

    // fill outputIds with endIds
    TensorPtr outputIds = ITensor::slice(dJointOutput.ids, batchSlot, 1);
    auto outputIdsTileView = ITensor::view(outputIds, ITensor::makeShape({beamWidth, maxSequenceLength}));
    runtime::kernels::invokeFill(*outputIdsTileView, endId, *decoderStream);

    // copy the request ids into outputIds
    auto const requestIdsShape = requestIds->getShape();
    auto outputIdsView = ITensor::view(outputIds, requestIdsShape);
    manager.copy(*requestIds, *outputIdsView);
    if (beamWidth > 1)
    {
        runtime::kernels::tileTensorInplace(*outputIdsTileView, beamWidth, *decoderStream);
    }

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void CreateNewDecoderRequests::newRequestSpeculativeDecoding(SizeType32 batchIdx,
    runtime::decoder_batch::Request const& request, SamplingConfig const& samplingConfig,
    runtime::ModelConfig const& modelConfig, DecodingInput& jointDecodingInput, DecodingOutput& jointDecodingOutput,
    CudaStream const& runtimeStream, CudaStream const& decoderStream,
    SpeculativeDecodingMode const& speculativeDecodingMode, SizeType32 maxDecodingEngineTokens) const
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    if (speculativeDecodingMode.predictsDraftTokens())
    {
        auto const& stream = decoderStream;
        BufferManager manager{std::make_shared<CudaStream>(stream.get())};

        auto& dJointOutput = jointDecodingOutput;

        TensorPtr nextDraftTokens
            = ITensor::slice(dJointOutput.speculativeDecodingOutputs->nextDraftTokens, batchIdx, 1);
        // FIXME: can we skip this?
        manager.setZero(*nextDraftTokens);
        if (speculativeDecodingMode.variableDraftLength())
        {
            TensorPtr nextDraftTokensLen
                = ITensor::slice(dJointOutput.speculativeDecodingOutputs->nextDraftTokensLen, batchIdx, 1);
            manager.setZero(*nextDraftTokensLen);
        }
    }

    if (speculativeDecodingMode.isDraftTokensExternal())
    {
        newRequestDraftTokensExternal(batchIdx, request, samplingConfig, jointDecodingInput, decoderStream);
    }
    else if (speculativeDecodingMode.isMedusa())
    {
        newRequestMedusa(batchIdx, request, jointDecodingInput, decoderStream, maxDecodingEngineTokens);
    }
    else if (speculativeDecodingMode.isLookaheadDecoding())
    {
        newRequestLookahead(batchIdx, request, jointDecodingInput, jointDecodingOutput, runtimeStream);
    }
    else if (speculativeDecodingMode.isExplicitDraftTokens())
    {
        newRequestExplicitDraftTokens(batchIdx, request, jointDecodingOutput, runtimeStream);
    }
    else if (speculativeDecodingMode.isEagle())
    {
        newRequestEagle(batchIdx, request, modelConfig, jointDecodingOutput, runtimeStream);
    }
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void CreateNewDecoderRequests::newRequestDraftTokensExternal(SizeType32 batchIdx,
    runtime::decoder_batch::Request const& request, SamplingConfig const& samplingConfig,
    DecodingInput& jointDecodingInput, CudaStream const& decoderStream) const
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    BufferManager manager{std::make_shared<CudaStream>(decoderStream.get())};

    auto& dJointInput = jointDecodingInput;

    auto const numDraftTokens = request.generatedTokensPerEngineStep - 1;

    auto const useDraftLogits = request.draftLogits.has_value();
    if (useDraftLogits)
    {
        TensorPtr draftLogitsView = ITensor::view(request.draftLogits.value());

        TensorPtr draftLogitsReqBatchSlice
            = ITensor::slice(dJointInput.externalDraftTokensInputs->draftLogits, batchIdx, 1);
        draftLogitsReqBatchSlice->squeeze(0);
        TensorPtr draftLogitsReqTokensSlice = ITensor::slice(draftLogitsReqBatchSlice, 0, numDraftTokens);
        manager.copy(*draftLogitsView, *draftLogitsReqTokensSlice);
    }
    auto* useDraftLogitsHostPtr = runtime::bufferCast<bool>(*dJointInput.externalDraftTokensInputs->useDraftLogitsHost);
    useDraftLogitsHostPtr[batchIdx] = useDraftLogits;
    auto useDraftLogitsView = ITensor::slice(dJointInput.externalDraftTokensInputs->useDraftLogits, batchIdx, 1);
    runtime::kernels::invokeFill(*useDraftLogitsView, useDraftLogits, decoderStream);

    if (numDraftTokens > 0)
    {
        TensorPtr draftTokensReqBatchSlice
            = ITensor::slice(dJointInput.externalDraftTokensInputs->draftTokenIds, batchIdx, 1);
        draftTokensReqBatchSlice->squeeze(0);
        TensorPtr draftTokensReqTokensSlice = ITensor::slice(draftTokensReqBatchSlice, 0, numDraftTokens);
        TensorPtr draftTokensView = ITensor::view(request.draftTokens, ITensor::makeShape({numDraftTokens}));
        manager.copy(*draftTokensView, *draftTokensReqTokensSlice);
    }

    auto* numDraftTokensHostPtr
        = runtime::bufferCast<SizeType32>(*dJointInput.externalDraftTokensInputs->numDraftTokensHost);
    numDraftTokensHostPtr[batchIdx] = numDraftTokens;
    auto numDraftTokensView = ITensor::slice(dJointInput.externalDraftTokensInputs->numDraftTokens, batchIdx, 1);
    runtime::kernels::invokeFill(*numDraftTokensView, numDraftTokens, decoderStream);

    bool const useRandomAcceptanceThreshold = !samplingConfig.draftAcceptanceThreshold.has_value();
    float const constantThreshold
        = useRandomAcceptanceThreshold ? 0 : samplingConfig.draftAcceptanceThreshold.value()[0];

    dJointInput.externalDraftTokensInputs->useRandomAcceptanceThreshold = useRandomAcceptanceThreshold;
    dJointInput.externalDraftTokensInputs->constantThreshold = constantThreshold;

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void CreateNewDecoderRequests::newRequestMedusa(SizeType32 batchIdx, runtime::decoder_batch::Request const& request,
    DecodingInput& jointDecodingInput, CudaStream const& decoderStream, SizeType32 maxDecodingEngineTokens) const
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    BufferManager manager{std::make_shared<CudaStream>(decoderStream.get())};

    auto& dJointInput = jointDecodingInput;

    TensorPtr curTokensPerStepSlice
        = ITensor::slice(constPointerCast(dJointInput.medusaInputs->medusaCurTokensPerStep), batchIdx, 1);
    // Context phase Medusa processes 1 token only, new value from targetTokensPerStep will be filled at the end
    // of first decoder
    runtime::kernels::invokeFill(*curTokensPerStepSlice, 1, decoderStream);
    TensorPtr targetTokensPerStepSlice
        = ITensor::slice(constPointerCast(dJointInput.medusaInputs->medusaTargetTokensPerStep), batchIdx, 1);
    auto const generatedTokensPerEngineStep = request.generatedTokensPerEngineStep;
    TLLM_CHECK_WITH_INFO(generatedTokensPerEngineStep <= maxDecodingEngineTokens,
        "Tokens per step for (%d) is larger than maximum tokens per step (%d)", generatedTokensPerEngineStep,
        maxDecodingEngineTokens);
    runtime::kernels::invokeFill(*targetTokensPerStepSlice, generatedTokensPerEngineStep, decoderStream);

    TensorPtr pathsSlice = ITensor::slice(constPointerCast(dJointInput.medusaInputs->medusaPaths), batchIdx, 1);
    manager.copy(*request.medusaPaths, *pathsSlice);

    TensorPtr treeIdsSlice = ITensor::slice(constPointerCast(dJointInput.medusaInputs->medusaTreeIds), batchIdx, 1);
    manager.copy(*request.medusaTreeIds, *treeIdsSlice);

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void CreateNewDecoderRequests::newRequestLookahead(SizeType32 batchIdx, runtime::decoder_batch::Request const& request,
    DecodingInput& jointDecodingInput, DecodingOutput& jointDecodingOutput, CudaStream const& runtimeStream) const
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    TLLM_CHECK(jointDecodingOutput.lookaheadOutputs);

    // The first generation step only generate 1 token.
    TensorPtr curTokensPerStepSlice
        = ITensor::slice(constPointerCast(jointDecodingInput.lookaheadInputs->tokensPerStep), batchIdx, 1);
    runtime::kernels::invokeFill(*curTokensPerStepSlice, 1, runtimeStream);

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void CreateNewDecoderRequests::newRequestExplicitDraftTokens(SizeType32 batchIdx,
    runtime::decoder_batch::Request const& request, DecodingOutput& jointDecodingOutput,
    CudaStream const& runtimeStream) const
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    TLLM_CHECK(jointDecodingOutput.explicitDraftTokensBuffers);

    TensorPtr positionIdsBaseSlice
        = ITensor::slice(jointDecodingOutput.explicitDraftTokensBuffers->positionIdsBase, batchIdx, 1);
    runtime::kernels::invokeFill(*positionIdsBaseSlice, request.inputLen, runtimeStream);

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void CreateNewDecoderRequests::newRequestEagle(SizeType32 batchIdx, runtime::decoder_batch::Request const& request,
    runtime::ModelConfig const& modelConfig, DecodingOutput& jointDecodingOutput, CudaStream const& runtimeStream) const
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    TLLM_CHECK(jointDecodingOutput.eagleBuffers);

    BufferManager manager{std::make_shared<CudaStream>(runtimeStream.get())};

    TensorPtr eagleNetCtxRequestTypesHostSlice
        = ITensor::slice(jointDecodingOutput.eagleBuffers->eagleNetCtxRequestTypesHost, batchIdx, 1);
    TensorPtr eagleNetCtxContextLengthsHostSlice
        = ITensor::slice(jointDecodingOutput.eagleBuffers->eagleNetCtxContextLengthsHost, batchIdx, 1);
    TensorPtr eagleNetCtxPastKeyValueLengthsHostSlice
        = ITensor::slice(jointDecodingOutput.eagleBuffers->eagleNetCtxPastKeyValueLengthsHost, batchIdx, 1);

    runtime::bufferCast<SizeType32>(*eagleNetCtxRequestTypesHostSlice)[0] = 0;
    runtime::bufferCast<SizeType32>(*eagleNetCtxContextLengthsHostSlice)[0] = request.inputLen;
    runtime::bufferCast<SizeType32>(*eagleNetCtxPastKeyValueLengthsHostSlice)[0] = request.inputLen;

    TensorPtr eagleNetGenRequestTypesHostSlice
        = ITensor::slice(jointDecodingOutput.eagleBuffers->eagleNetGenRequestTypesHost, batchIdx, 1);
    TensorPtr eagleNetGenContextLengthsHostSlice
        = ITensor::slice(jointDecodingOutput.eagleBuffers->eagleNetGenContextLengthsHost, batchIdx, 1);
    TensorPtr eagleNetGenPastKeyValueLengthsHostSlice
        = ITensor::slice(jointDecodingOutput.eagleBuffers->eagleNetGenPastKeyValueLengthsHost, batchIdx, 1);

    runtime::bufferCast<SizeType32>(*eagleNetGenRequestTypesHostSlice)[0] = 1;
    runtime::bufferCast<SizeType32>(*eagleNetGenContextLengthsHostSlice)[0] = request.inputLen;
    runtime::bufferCast<SizeType32>(*eagleNetGenPastKeyValueLengthsHostSlice)[0] = request.inputLen;

    auto const eagleModule = std::dynamic_pointer_cast<tensorrt_llm::runtime::EagleModule const>(
        modelConfig.getSpeculativeDecodingModulePtr());
    std::optional<executor::EagleChoices> eagleChoicesOpt;

    TensorPtr draftPathsSlice = ITensor::slice(jointDecodingOutput.eagleBuffers->draftPaths, batchIdx, 1);

    if (request.eagleConfig)
    {
        eagleChoicesOpt = request.eagleConfig->getEagleChoices();
    }

    if (!request.eagleConfig || !request.eagleConfig->useDynamicTree())
    {
        // eagleConfig is nullptr or Eagle-1
        std::vector<SizeType32> topKs;
        TensorPtr draftPathsHost = manager.pinnedPool(draftPathsSlice->getShape(), nvinfer1::DataType::kINT32);
        auto const depth = utils::initTensorsFromChoices(modelConfig.getSpeculativeDecodingModule(),
            eagleChoicesOpt.value_or(eagleModule->getDefaultEagleChoices()), topKs, nullptr, nullptr, nullptr,
            draftPathsHost, nullptr, {eagleModule->getMaxNonLeafNodesPerLayer()});
        TLLM_CHECK_WITH_INFO(depth == modelConfig.getSpeculativeDecodingModule().getMaxDraftPathLen(),
            "EAGLE-1 requires Eagle-tree depth being equal to the the number of build-time EAGLE layers.");

        manager.copy(*draftPathsHost, *draftPathsSlice);
    }

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

} // namespace tensorrt_llm::batch_manager
