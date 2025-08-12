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
#include "tensorrt_llm/batch_manager/decoderBuffers.h"
#include "tensorrt_llm/batch_manager/llmRequest.h"
#include "tensorrt_llm/batch_manager/medusaBuffers.h"
#include "tensorrt_llm/batch_manager/utils/logitsThread.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/common/nvtxUtils.h"
#include "tensorrt_llm/runtime/decoderState.h"
#include "tensorrt_llm/runtime/decodingInput.h"
#include "tensorrt_llm/runtime/decodingOutput.h"
#include "tensorrt_llm/runtime/runtimeKernels.h"
#include "tensorrt_llm/runtime/speculativeDecodingMode.h"
#include "tensorrt_llm/runtime/utils/mpiUtils.h"
#include "tensorrt_llm/runtime/utils/speculativeChoicesUtils.h"

#include <NvInferRuntimeBase.h>

using namespace tensorrt_llm::runtime;

namespace tc = tensorrt_llm::common;
namespace te = tensorrt_llm::executor;
namespace tk = tensorrt_llm::kernels;
namespace tr = tensorrt_llm::runtime;

namespace tensorrt_llm::batch_manager
{

using SizeType32 = CreateNewDecoderRequests::SizeType32;
using TensorPtr = CreateNewDecoderRequests::TensorPtr;
using SharedConstPtr = CreateNewDecoderRequests::SharedConstPtr;

namespace
{

void copySequenceLengths(RequestVector const& contextRequests, DecoderInputBuffers& inputBuffers,
    ITensor& sequenceLengths, SizeType32 beamWidth, runtime::CudaStream const& stream)
{
    auto const bufferManager = BufferManager{std::make_shared<CudaStream>(stream.get())};

    auto const batchSize = contextRequests.size();
    auto batchSlotsView = tr::ITensor::slice(inputBuffers.setupBatchSlots, 0, batchSize);
    auto fillValuesView = tr::ITensor::slice(inputBuffers.fillValues, 0, batchSize);

    auto batchSlotsRange = tr::BufferRange<SizeType32>(*batchSlotsView);
    auto fillValuesRange = tr::BufferRange<SizeType32>(*fillValuesView);

    // fill buffers on host
    SizeType32 batchIdx{0};
    for (auto const& llmReq : contextRequests)
    {
        auto const disaggFirstGenTokenSize
            = llmReq->getContextPhaseParams() ? llmReq->getContextPhaseParams().value().getFirstGenTokens().size() : 0;
        auto const currentSequenceLen
            = llmReq->mPromptLen + llmReq->getMaxNumGeneratedTokens() + disaggFirstGenTokenSize;
        // Get position of the current sequence in the decoder
        auto const seqSlot = llmReq->mSeqSlot.value();
        batchSlotsRange[batchIdx] = seqSlot;
        fillValuesRange[batchIdx] = currentSequenceLen;
        ++batchIdx;
    }

    // copy sequence lengths
    {
        auto batchSlotsDeviceView = tr::ITensor::slice(inputBuffers.setupBatchSlotsDevice, 0, batchSize);
        auto fillValuesViewDevice = tr::ITensor::slice(inputBuffers.fillValuesDevice, 0, batchSize);

        bufferManager.copy(*batchSlotsView, *batchSlotsDeviceView);
        bufferManager.copy(*fillValuesView, *fillValuesViewDevice);
        tr::kernels::invokeFillBatch(sequenceLengths, *batchSlotsDeviceView, beamWidth, *fillValuesViewDevice, stream);
    }
}

/// @brief Retrieve the embedding bias from the request. This potentially makes a copy of the tensor
/// to the appropriate type if the input tensor does not match it.
[[nodiscard]] TensorPtr getEmbeddingBias(nvinfer1::DataType logitsType, TensorPtr const& tensor)
{
    // Check that embedding bias type is same as logits type. If so, we can return the tensor right away
    if (tensor->getDataType() == logitsType)
    {
        return tensor;
    }

    // Support FP32 input for FP16 embedding bias (in the case of FP8 models)
    if (tensor->getDataType() == nvinfer1::DataType::kFLOAT && logitsType == nvinfer1::DataType::kHALF)
    {
        // Do a deep copy of the tensor to the expected type
        TLLM_LOG_WARNING(
            "Embedding bias data type must be same as model logits type, will copy the tensor from float to half");

        TLLM_CHECK_WITH_INFO(
            tensor->getMemoryType() != MemoryType::kGPU, "Embedding bias tensor needs to be in CPU memory for casting");

        auto const shape = tensor->getShape();
        TLLM_CHECK(shape.nbDims == 2); // [1, vocabSizePadded]
        TLLM_CHECK(shape.d[0] == 1);
        auto newTensor = tensorrt_llm::runtime::BufferManager::pinnedPool(shape, logitsType);

        auto const tensorRange = BufferRange<float>(*tensor);
        auto newTensorRange = BufferRange<half>(*newTensor);

        std::transform(tensorRange.begin(), tensorRange.end(), newTensorRange.begin(),
            [](float value) -> half { return static_cast<half>(value); });

        return newTensor;
    }

    TLLM_THROW("Embedding bias data type must be same as model logits type.");
}

} // namespace

std::tuple<TensorPtr, std::vector<runtime::SamplingConfig>, std::vector<runtime::ITensor::SharedConstPtr>,
    std::vector<executor::LookaheadDecodingConfig>>
CreateNewDecoderRequests::operator()(runtime::ModelConfig const& modelConfig, runtime::WorldConfig const& worldConfig,
    executor::DecodingConfig const& decodingConfig, RequestVector const& contextRequests, nvinfer1::DataType logitsType,
    DecoderInputBuffers& inputBuffers, runtime::decoder::DecoderState& decoderState, CudaStream const& runtimeStream,
    CudaStream const& decoderStream, SizeType32 maxSequenceLength, SizeType32 beamWidth,
    OptionalRef<MedusaBuffers const> medusaBuffers) const
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    NVTX3_SCOPED_RANGE(CreateNewDecoderRequests);

    RequestVector finishedContextRequests;
    std::copy_if(contextRequests.begin(), contextRequests.end(), std::back_inserter(finishedContextRequests),
        [](auto const& llmReq) { return llmReq->isLastContextChunk(); });

    if (!finishedContextRequests.empty())
    {
        copySequenceLengths(
            finishedContextRequests, inputBuffers, *decoderState.getSequenceLengths(), beamWidth, runtimeStream);
    }

    auto [lookaheadPrompt, lookaheadAlgoConfigs]
        = createDecoderRequests(finishedContextRequests, inputBuffers.inputsIds, decodingConfig, decoderState,
            logitsType, modelConfig, worldConfig, runtimeStream, decoderStream, maxSequenceLength, medusaBuffers);

    auto const batchSize = finishedContextRequests.size();

    std::vector<SamplingConfig> samplingConfigs;
    samplingConfigs.reserve(batchSize);
    for (auto const& llmReq : finishedContextRequests)
    {
        samplingConfigs.push_back(llmReq->mSamplingConfig);
    }

    TensorPtr batchSlotsView = runtime::ITensor::slice(inputBuffers.setupBatchSlots, 0, batchSize);

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
    return {std::move(batchSlotsView), std::move(samplingConfigs), std::move(lookaheadPrompt),
        std::move(lookaheadAlgoConfigs)};
}

namespace
{

void initializeInputLengths(DecodingInput& dJointInput, SizeType32 batchSlot, SizeType32 inputLength,
    std::optional<SizeType32> maxNewTokensOpt, SizeType32 numDecodingEngineTokens, SizeType32 maxSequenceLength,
    BufferManager const& manager)
{
    auto const numDecodingDraftEngineTokens = numDecodingEngineTokens - 1;
    auto const maxNewTokens = maxNewTokensOpt.value_or(maxSequenceLength - inputLength - numDecodingDraftEngineTokens);

    TLLM_CHECK_WITH_INFO(inputLength + maxNewTokens + numDecodingDraftEngineTokens <= maxSequenceLength,
        tc::fmtstr(
            "Input length (%d) + max new tokens (%d) + draft tokens (%d) must be less than max sequence length (%d).",
            inputLength, maxNewTokens, numDecodingDraftEngineTokens, maxSequenceLength));

    TensorPtr const sequenceLimitLength{
        ITensor::slice(constPointerCast(dJointInput.sequenceLimitLength), batchSlot, 1)};
    runtime::kernels::invokeFill(*sequenceLimitLength, inputLength + maxNewTokens, manager.getStream());

    TensorPtr const inputLengths{ITensor::slice(constPointerCast(dJointInput.lengths), batchSlot, 1)};
    runtime::kernels::invokeFill(*inputLengths, inputLength, manager.getStream());
}

void initializeRequestIds(DecodingInput& dJointInput, DecodingOutput& dJointOutput, SizeType32 batchSlot,
    SharedConstPtr const& requestIds, SizeType32 endId, SizeType32 beamWidth, SizeType32 maxSequenceLength,
    BufferManager const& manager)
{
    TensorPtr const endIdTensorPtr{ITensor::slice(constPointerCast(dJointInput.endIds), batchSlot, 1)};
    runtime::kernels::invokeFill(*endIdTensorPtr, endId, manager.getStream());

    // fill outputIds with endIds
    TensorPtr const outputIds = ITensor::slice(dJointOutput.ids, batchSlot, 1);
    auto outputIdsTileView = ITensor::view(outputIds, ITensor::makeShape({beamWidth, maxSequenceLength}));
    runtime::kernels::invokeFill(*outputIdsTileView, endId, manager.getStream());

    // copy the request ids into outputIds
    auto const requestIdsShape = requestIds->getShape();
    auto outputIdsView = ITensor::view(outputIds, requestIdsShape);
    manager.copy(*requestIds, *outputIdsView);
}

void initializeBeamSearch(DecodingInput& dJointInput, DecodingOutput& dJointOutput, SizeType32 batchSlot,
    SizeType32 endId, SizeType32 beamWidth, SizeType32 maxSequenceLength, BufferManager const& manager)
{
    TensorPtr const cumLogProbs = ITensor::slice(dJointOutput.cumLogProbs, batchSlot, 1);
    runtime::kernels::invokeFill(
        *IBuffer::slice(cumLogProbs, 1, beamWidth - 1), DecodingOutput::kNegativeInfinity, manager.getStream());

    auto parentIds = ITensor::slice(dJointOutput.parentIds, batchSlot, 1);
    auto const outputIdsShape = ITensor::makeShape({1, beamWidth, maxSequenceLength});
    parentIds->reshape(outputIdsShape);
    manager.setZero(*parentIds);

    auto cacheIndirectionInput = ITensor::slice(dJointInput.cacheIndirection, batchSlot, 1);
    manager.setZero(*cacheIndirectionInput);

    auto cacheIndirectionOutput = ITensor::slice(dJointOutput.cacheIndirection, batchSlot, 1);
    manager.setZero(*cacheIndirectionOutput);

    auto beamHypotheses = dJointOutput.beamHypotheses.slice(batchSlot, 1);
    beamHypotheses.init(manager, endId);
}

void initializeEmbeddingBias(DecodingInput& dJointInput, SizeType32 batchSlot,
    std::optional<TensorPtr> const& embeddingBias, nvinfer1::DataType logitsType,
    runtime::ModelConfig const& modelConfig, BufferManager const& manager)
{
    TensorPtr const embeddingBiasSlice = ITensor::slice(constPointerCast(dJointInput.embeddingBias), batchSlot, 1);
    if (embeddingBias.has_value())
    {
        auto embeddingBiasTensor = getEmbeddingBias(logitsType, embeddingBias.value());

        TLLM_CHECK(embeddingBiasTensor->getShape().nbDims == 2);
        TLLM_CHECK(embeddingBiasTensor->getShape().d[0] == 1);
        TLLM_CHECK_WITH_INFO(embeddingBiasTensor->getShape().d[1] == modelConfig.getVocabSize(),
            "The embedding bias shape is not as expected. Expected last dimension to be same as vocab size: %d.",
            modelConfig.getVocabSize());
        manager.copy(*embeddingBiasTensor, *embeddingBiasSlice);
    }
    else
    {
        manager.setZero(*embeddingBiasSlice);
    }
}

void setupWords(std::vector<runtime::ITensor::SharedPtr>& jointWordsLists,
    std::optional<TensorPtr> const& requestWordsList, SharedConstPtr& jointWordsPtrs, SharedConstPtr& jointWordsLens,
    SizeType32& jointMaxWordsLen, SizeType32 batchSlot, BufferManager const& manager)
{
    if (requestWordsList.has_value())
    {
        // Move to GPU and remove leading bs1 dimension since this is what decoderRequest expects
        TensorPtr wordsList = manager.copyFrom(*requestWordsList.value(), MemoryType::kGPU);
        wordsList->squeeze(0);

        auto const wordsLen = wordsList->getShape().d[1];
        BufferRange<int32_t*>(*constPointerCast(jointWordsPtrs))[batchSlot]
            = runtime::bufferCast<TokenIdType>(*wordsList);
        runtime::bufferCast<SizeType32>(*constPointerCast(jointWordsLens))[batchSlot] = wordsLen;
        // FIXME: this is monotonically growing size
        jointMaxWordsLen = std::max(static_cast<SizeType32>(wordsLen), jointMaxWordsLen);

        // NOTE: jointWordsList is not used in gptDecoder, but required to keep <name>WordsList's
        // memory allocated
        jointWordsLists[batchSlot] = wordsList;
    }
    else
    {
        runtime::bufferCast<SizeType32>(*constPointerCast(jointWordsLens))[batchSlot] = 0;
    }
};

void initializeLogProbs(DecodingOutput& dJointOutput, SizeType32 batchSlot, SamplingConfig const& samplingConfig,
    BufferManager const& manager)
{
    auto const beamWidth = samplingConfig.beamWidth;

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
}

void initializeOutputs(DecodingOutput& dJointOutput, SizeType32 batchSlot, SizeType32 maxDecodingEngineTokens,
    BufferManager const& manager)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    auto finishedSum = ITensor::slice(dJointOutput.finishedSum, batchSlot, 1);
    manager.setZero(*finishedSum);

    for (SizeType32 ti = 0; ti < maxDecodingEngineTokens; ++ti)
    {
        TensorPtr const newTokensStepView = ITensor::slice(dJointOutput.newTokensSteps, ti, 1);
        newTokensStepView->squeeze(0);
        auto newTokensVec = ITensor::slice(newTokensStepView, batchSlot, 1);
        manager.setZero(*newTokensVec);
    }

    TensorPtr const finishedStepsSlice = ITensor::slice(dJointOutput.finishReasons, batchSlot, 1);
    manager.setZero(*finishedStepsSlice);

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

} // namespace

void CreateNewDecoderRequests::newRequestSpeculativeDecoding(SizeType32 batchIdx,
    runtime::decoder_batch::Request const& request, SamplingConfig const& samplingConfig,
    runtime::ModelConfig const& modelConfig, DecodingInput& jointDecodingInput, DecodingOutput& jointDecodingOutput,
    CudaStream const& runtimeStream, CudaStream const& decoderStream,
    SpeculativeDecodingMode const& speculativeDecodingMode, SizeType32 maxDecodingEngineTokens)
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
    DecodingInput& jointDecodingInput, CudaStream const& decoderStream)
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
    DecodingInput& jointDecodingInput, CudaStream const& decoderStream, SizeType32 maxDecodingEngineTokens)
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
    DecodingInput& jointDecodingInput, DecodingOutput& jointDecodingOutput, CudaStream const& runtimeStream)
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
    CudaStream const& runtimeStream)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    TLLM_CHECK(jointDecodingOutput.explicitDraftTokensBuffers);

    TensorPtr positionIdsBaseSlice
        = ITensor::slice(jointDecodingOutput.explicitDraftTokensBuffers->positionIdsBase, batchIdx, 1);
    runtime::kernels::invokeFill(*positionIdsBaseSlice, request.inputLen, runtimeStream);

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void CreateNewDecoderRequests::newRequestEagle(SizeType32 batchIdx, runtime::decoder_batch::Request const& request,
    runtime::ModelConfig const& modelConfig, DecodingOutput& jointDecodingOutput, CudaStream const& runtimeStream)
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

    if (request.eagleConfig)
    {
        eagleChoicesOpt = request.eagleConfig->getEagleChoices();
    }

    if (!request.eagleConfig || !request.eagleConfig->useDynamicTree())
    {
        TensorPtr draftPathsHostSlice = ITensor::slice(jointDecodingOutput.eagleBuffers->draftPathsHost, batchIdx, 1);
        TensorPtr draftPathsSlice = ITensor::slice(jointDecodingOutput.eagleBuffers->draftPaths, batchIdx, 1);

        // eagleConfig is nullptr or Eagle-1
        std::vector<SizeType32> topKs;
        auto const depth = runtime::utils::initTensorsFromChoices(modelConfig.getSpeculativeDecodingModule(),
            eagleChoicesOpt.value_or(eagleModule->getDefaultEagleChoices()), topKs, nullptr, nullptr, nullptr,
            draftPathsHostSlice, nullptr, {eagleModule->getMaxNonLeafNodesPerLayer()});
        TLLM_CHECK_WITH_INFO(depth == modelConfig.getSpeculativeDecodingModule().getMaxDraftPathLen(),
            "EAGLE-1 requires Eagle-tree depth being equal to the the number of build-time EAGLE layers.");

        manager.copy(*draftPathsHostSlice, *draftPathsSlice);
    }

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

std::tuple<std::vector<runtime::ITensor::SharedConstPtr>, std::vector<executor::LookaheadDecodingConfig>>
CreateNewDecoderRequests::createDecoderRequests(RequestVector const& finishedContextRequests, TensorPtr const& inputIds,
    executor::DecodingConfig const& decodingConfig, runtime::decoder::DecoderState& decoderState,
    nvinfer1::DataType logitsType, runtime::ModelConfig const& modelConfig, runtime::WorldConfig const& worldConfig,
    runtime::CudaStream const& runtimeStream, runtime::CudaStream const& decoderStream, SizeType32 maxSequenceLength,
    OptionalRef<MedusaBuffers const> medusaBuffers) const
{
    auto const decoderBufferManager = BufferManager{std::make_shared<CudaStream>(decoderStream.get())};

    unsigned decoderInputSize{0};
    for (auto const& llmReq : finishedContextRequests)
    {
        auto const& reqTokens = llmReq->getTokens(0);
        decoderInputSize += reqTokens.size();
    }
    inputIds->resize(decoderInputSize);

    std::vector<decoder_batch::Request> decoderRequests;
    decoderRequests.reserve(finishedContextRequests.size());

    std::vector<runtime::ITensor::SharedConstPtr> lookaheadPrompt;
    std::vector<executor::LookaheadDecodingConfig> lookaheadAlgoConfigs;
    if (modelConfig.getSpeculativeDecodingMode().isLookaheadDecoding())
    {
        TLLM_CHECK_WITH_INFO(
            decodingConfig.getLookaheadDecodingConfig().has_value(), "Lookahead decoding config must be provided");
        lookaheadPrompt.reserve(finishedContextRequests.size());
        lookaheadAlgoConfigs.reserve(finishedContextRequests.size());
    }

    SizeType32 inputOffset{0};
    for (auto const& llmReq : finishedContextRequests)
    {
        llmReq->mSamplingConfig.normalizeLogProbs = mIsNormalizeLogProbs;

        TLLM_CHECK(llmReq->mSeqSlot.has_value());
        auto const batchSlot = llmReq->mSeqSlot.value();
        auto const batchSize = decoderState.getMaxNumSequences();
        TLLM_CHECK(0 <= batchSlot && batchSlot < batchSize);

        auto const& samplingConfig = llmReq->mSamplingConfig;

        auto const beamWidth = samplingConfig.beamWidth;
        auto const maxBeamWidth = decoderState.getMaxBeamWidth();
        TLLM_CHECK_WITH_INFO(beamWidth <= maxBeamWidth,
            tc::fmtstr("Beam width (%d) must be smaller than maxBeamWidth (%d) passed to decoder setup function.",
                beamWidth, maxBeamWidth));
        decoderState.setBeamWidth(batchSlot, beamWidth);

        auto const promptLen = llmReq->getPromptLen();

        auto decoderRequest = decoder_batch::Request{promptLen};

        if (modelConfig.getSpeculativeDecodingMode().isDraftTokensExternal())
        {
            if (llmReq->hasDraftTokens())
            {
                auto const& draftTokens = llmReq->getDraftTokens();
                // Copy to pinned host memory (don't care about stream of bufferManager)
                decoderRequest.draftTokens = decoderBufferManager.copyFrom(*draftTokens, MemoryType::kPINNEDPOOL);
                auto const& draftLogits = llmReq->getDraftLogits();
                if (draftLogits.has_value())
                {
                    decoderRequest.draftLogits
                        = retrieveDraftLogits(modelConfig, worldConfig, draftLogits.value(), decoderBufferManager);
                }
                decoderRequest.generatedTokensPerEngineStep = draftTokens->size() + 1;
            }
            else
            {
                decoderRequest.generatedTokensPerEngineStep = 1;
            }
        }
        else if (!modelConfig.getSpeculativeDecodingMode().isNone())
        {
            decoderRequest.generatedTokensPerEngineStep = modelConfig.getMaxDecodingTokens();
        }

        auto& dJointInput = decoderState.getJointDecodingInput();

        auto const numDecodingEngineTokens = decoderRequest.generatedTokensPerEngineStep;
        initializeInputLengths(dJointInput, batchSlot, promptLen, llmReq->mMaxNewTokens, numDecodingEngineTokens,
            maxSequenceLength, decoderBufferManager);
        decoderState.setNumDecodingEngineTokens(batchSlot, numDecodingEngineTokens);

        initializeEmbeddingBias(
            dJointInput, batchSlot, llmReq->getEmbeddingBias(), logitsType, modelConfig, decoderBufferManager);

        setupWords(dJointInput.badWordsLists, llmReq->getBadWordsList(), dJointInput.badWordsPtrs,
            dJointInput.badWordsLens, dJointInput.maxBadWordsLen, batchSlot, decoderBufferManager);

        setupWords(dJointInput.stopWordsLists, llmReq->getStopWordsList(), dJointInput.stopWordsPtrs,
            dJointInput.stopWordsLens, dJointInput.maxStopWordsLen, batchSlot, decoderBufferManager);

        auto& dJointOutput = decoderState.getJointDecodingOutput();

        initializeOutputs(dJointOutput, batchSlot, decoderState.getMaxDecodingEngineTokens(), decoderBufferManager);

        initializeLogProbs(dJointOutput, batchSlot, samplingConfig, decoderBufferManager);

        auto const& reqTokens = llmReq->getTokens(0);
        TLLM_CHECK(reqTokens.size() == static_cast<decltype(reqTokens.size())>(promptLen));
        TensorPtr requestIds = ITensor::slice(inputIds, inputOffset, promptLen);
        // Copy to pinned host memory (don't care about stream of bufferManager)
        decoderBufferManager.copy(reqTokens.data(), *requestIds);
        auto const endId = llmReq->mEndId.value_or(-1);

        initializeRequestIds(dJointInput, dJointOutput, batchSlot, requestIds, endId, beamWidth, maxSequenceLength,
            decoderBufferManager);

        if (beamWidth > 1)
        {
            initializeBeamSearch(
                dJointInput, dJointOutput, batchSlot, endId, beamWidth, maxSequenceLength, decoderBufferManager);
        }

        // Speculative execution
        if (!decoderState.getSpeculativeDecodingMode().isNone())
        {
            TLLM_CHECK(beamWidth == 1);

            if (modelConfig.getSpeculativeDecodingMode().isMedusa())
            {
                TLLM_CHECK(medusaBuffers);
                llmReq->mSamplingConfig.topKMedusaHeads = {medusaBuffers->mTopKs};
                // FIXME: we must set medusa paths and tree ids not from seq slot, but from llmRequest?
                // When multiple microbatches buffers are used, runtime buffers can not be addressed with seqSlot.
                decoderRequest.medusaPaths = ITensor::slice(medusaBuffers->medusaPathsDevice, 0, 1);
                decoderRequest.medusaTreeIds = ITensor::slice(medusaBuffers->medusaTreeIdsDevice, 0, 1);
            }
            else if (modelConfig.getSpeculativeDecodingMode().isLookaheadDecoding())
            {
                lookaheadPrompt.emplace_back(requestIds);

                auto const& lookaheadRuntimeConfig
                    = llmReq->getLookaheadConfig().value_or(decodingConfig.getLookaheadDecodingConfig().value());
                lookaheadAlgoConfigs.emplace_back(lookaheadRuntimeConfig);
            }
            else if (modelConfig.getSpeculativeDecodingMode().isEagle())
            {
                decoderRequest.eagleConfig
                    = llmReq->getEagleConfig() ? llmReq->getEagleConfig() : decodingConfig.getEagleConfig();
            }

            newRequestSpeculativeDecoding(batchSlot, decoderRequest, samplingConfig, modelConfig,
                decoderState.getJointDecodingInput(), decoderState.getJointDecodingOutput(), runtimeStream,
                decoderStream, decoderState.getSpeculativeDecodingMode(), decoderState.getMaxDecodingEngineTokens());
        }

        decoderRequests.push_back(decoderRequest);

        inputOffset += promptLen;
    }

    return {std::move(lookaheadPrompt), std::move(lookaheadAlgoConfigs)};
}

std::shared_ptr<runtime::ITensor> CreateNewDecoderRequests::retrieveDraftLogits(ModelConfig const& modelConfig,
    WorldConfig const& worldConfig, std::shared_ptr<runtime::ITensor> const& tensor,
    BufferManager const& bufferManager) const
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    if (!mSpeculativeDecodingFastLogits)
    {
        TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
        return bufferManager.copyFrom(*tensor, MemoryType::kPINNEDPOOL);
    }

    if (mIsLeaderInOrchMode)
    {
        te::SpeculativeDecodingFastLogitsInfo fastLogitsInfo;
        std::memcpy(&fastLogitsInfo, tensor->data(), sizeof(fastLogitsInfo));
        auto logits = utils::targetModelReceiveLogits(fastLogitsInfo, modelConfig).value();

        // Broadcast to other ranks if needed
        if (worldConfig.isTensorParallel())
        {
            auto const& commSession = COMM_SESSION;
            auto shape = logits->getShape();
            commSession.bcastValue(shape.d[0], 0);
            commSession.bcastValue(shape.d[1], 0);
            commSession.bcast(logits->data(), logits->getSizeInBytes(), mpi::MpiType::kUINT8, 0);
        }
        TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
        return logits;
    }

    // Get logits from leader rank
    auto const& commSession = COMM_SESSION;
    int64_t dims[2];
    commSession.bcastValue(dims[0], 0);
    commSession.bcastValue(dims[1], 0);
    auto const logitsDtype = modelConfig.getLogitsDtype();
    auto logits = tensorrt_llm::runtime::BufferManager::pinnedPool(ITensor::makeShape({dims[0], dims[1]}), logitsDtype);
    commSession.bcast(logits->data(), logits->getSizeInBytes(), mpi::MpiType::kUINT8, 0);

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
    return logits;
};

} // namespace tensorrt_llm::batch_manager
