/*
 * Copyright (c) 2019-2024, NVIDIA CORPORATION.  All rights reserved.
 * Copyright (c) 2021, NAVER Corp.  Authored by CLOVA.
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

#include "penaltyLayer.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/nvtxUtils.h"
#include "tensorrt_llm/kernels/penaltyKernels.h"
#include "tensorrt_llm/kernels/penaltyTypes.h"
#include "tensorrt_llm/layers/defaultDecodingParams.h"
#include "tensorrt_llm/layers/layerUtils.h"
#include "tensorrt_llm/runtime/bufferManager.h"
#include "tensorrt_llm/runtime/common.h"

#include <algorithm>

using namespace tensorrt_llm::common;
using namespace tensorrt_llm::kernels;
using namespace tensorrt_llm::runtime;

namespace tensorrt_llm::layers
{

template <typename T>
size_t PenaltyLayer<T>::getWorkspaceSize() const noexcept
{
    return mWorkspaceSize;
}

template <typename T>
PenaltyLayer<T>::PenaltyLayer(executor::DecodingMode const& mode, DecoderDomain const& decoderDomain,
    std::shared_ptr<BufferManager> bufferManager)
    : BaseLayer(decoderDomain, bufferManager)
    , mDecodingMode(mode)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    initialize();

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void PenaltyLayer<T>::initialize()
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    allocateBuffer();

    mCyclicStep = 0;
    mRuntimeMaxSeqLen = 0;
    mConfiguredBeamWidth = -1;

    if (!mDecodingMode.isAuto())
    {
        mConfiguredBeamWidth = mDecoderDomain.getBeamWidth();

        allocateWorkspace();
    }

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void PenaltyLayer<T>::allocateWorkspace()
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    if (mDecodingMode.isUseOccurrencePenalty())
    {

        auto const workspaceSize = mDecoderDomain.getBatchSize() * mDecoderDomain.getMaxDecodingTokens()
            * mConfiguredBeamWidth * mDecoderDomain.getVocabSize();
        mPenaltyWorkspaceDevice = mBufferManager->gpu(workspaceSize, nvinfer1::DataType::kINT32);

        if (mDecodingMode.isBeamSearch())
        {
            mPenaltyWorkspacePrevDevice = mBufferManager->gpu(workspaceSize, nvinfer1::DataType::kINT32);
        }
    }

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void PenaltyLayer<T>::allocateBuffer()
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    mLogitsPtrsHost = mBufferManager->pinnedPool(ITensor::makeShape({}), TRTDataType<T*>::value);
    auto const batchSizeShape = ITensor::makeShape({mDecoderDomain.getBatchSize()});
    mTemperature = mBufferManager->pinnedPool(batchSizeShape, TRTDataType<float>::value);
    mRepetitionPenalty = mBufferManager->pinnedPool(batchSizeShape, TRTDataType<float>::value);
    mPresencePenalty = mBufferManager->pinnedPool(batchSizeShape, TRTDataType<float>::value);
    mFrequencyPenalty = mBufferManager->pinnedPool(batchSizeShape, TRTDataType<float>::value);
    mMinLength = mBufferManager->pinnedPool(batchSizeShape, TRTDataType<SizeType32>::value);

    if (mDecodingMode.isUseTemperature())
    {
        mTemperatureDevice = mBufferManager->gpu(batchSizeShape, nvinfer1::DataType::kFLOAT);
    }
    if (mDecodingMode.isUseRepetitionPenalty())
    {
        mRepetitionPenaltyDevice = mBufferManager->gpu(batchSizeShape, nvinfer1::DataType::kFLOAT);
    }
    if (mDecodingMode.isUsePresencePenalty())
    {
        mPresencePenaltyDevice = mBufferManager->gpu(batchSizeShape, nvinfer1::DataType::kFLOAT);
    }
    if (mDecodingMode.isUseFrequencyPenalty())
    {
        mFrequencyPenaltyDevice = mBufferManager->gpu(batchSizeShape, nvinfer1::DataType::kFLOAT);
    }
    if (mDecodingMode.isUseMinLength())
    {
        mMinLengthDevice = mBufferManager->gpu(batchSizeShape, nvinfer1::DataType::kINT32);
    }

    auto const logitsPtrDeviceDesc = std::make_pair(batchSizeShape, TRTDataType<T*>::value);
    mWorkspaceSize = DecodingLayerWorkspace::calculateRequiredWorkspaceSize(logitsPtrDeviceDesc);

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void PenaltyLayer<T>::setup(SizeType32 batchSize, SizeType32 beamWidth, TensorConstPtr batchSlots,
    std::shared_ptr<BaseSetupParams> const& baseSetupParams,
    std::shared_ptr<runtime::DecodingLayerWorkspace> const& workspace)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    NVTX3_SCOPED_RANGE(PenaltyLayer_setup);

    auto setupParams = std::dynamic_pointer_cast<DynamicDecodeSetupParams>(baseSetupParams);

    if (mConfiguredBeamWidth == -1)
    {
        // This code is left only for Python runtime
        // In C++ runtime given maxBeamWidth should always be equal to the runtime beamWidth
        TLLM_CHECK(mDecodingMode.isAuto());
        mConfiguredBeamWidth = beamWidth;
        mDecodingMode
            = mConfiguredBeamWidth == 1 ? executor::DecodingMode::TopKTopP() : executor::DecodingMode::BeamSearch();
        allocateWorkspace();
    }

    // Setup penalties.
    FillBuffers const fillBuffers{batchSize, mDecoderDomain.getBatchSize(), mBufferManager};

    auto const& penaltyParams = setupParams->penaltyParams;
    TLLM_CHECK_WITH_INFO(penaltyParams, "penaltyParams for setup is not set");

    bool const useTemperature = mDecodingMode.isUseTemperature() && penaltyParams->temperature.has_value();
    bool const useRepetitionPenalty
        = mDecodingMode.isUseRepetitionPenalty() && penaltyParams->repetitionPenalty.has_value();
    bool const usePresencePenalty = mDecodingMode.isUsePresencePenalty() && penaltyParams->presencePenalty.has_value();
    bool const useFrequencyPenalty
        = mDecodingMode.isUseFrequencyPenalty() && penaltyParams->frequencyPenalty.has_value();
    bool const useMinLength = mDecodingMode.isUseMinLength() && penaltyParams->minLength.has_value();
    // FIXME: once one of the requests has some penalty, we will always have to compute it.
    // To avoid that we need to scan through all active requests at each iteration.
    mUseTemperature |= useTemperature;
    mUseRepetitionPenalty |= useRepetitionPenalty;
    mUsePresencePenalty |= usePresencePenalty;
    mUseFrequencyPenalty |= useFrequencyPenalty;
    mUseMinLength |= useMinLength;

    if (mUseTemperature)
    {
        fillBuffers(penaltyParams->temperature, DefaultDecodingParams::getTemperature(), mTemperature,
            mTemperatureDevice, batchSlots, getLimitsPenalty(DecodingPenaltyType::Temperature), "temperature penalty");
    }
    if (mUseRepetitionPenalty)
    {
        fillBuffers(penaltyParams->repetitionPenalty, DefaultDecodingParams::getRepetitionPenalty(), mRepetitionPenalty,
            mRepetitionPenaltyDevice, batchSlots, getLimitsPenalty(DecodingPenaltyType::Repetition),
            "repetition penalty");
    }
    if (mUsePresencePenalty)
    {
        fillBuffers(penaltyParams->presencePenalty, DefaultDecodingParams::getPresencePenalty(), mPresencePenalty,
            mPresencePenaltyDevice, batchSlots, getLimitsPenalty(DecodingPenaltyType::Presence), "presence penalty");
    }
    if (mUseFrequencyPenalty)
    {
        fillBuffers(penaltyParams->frequencyPenalty, DefaultDecodingParams::getFrequencyPenalty(), mFrequencyPenalty,
            mFrequencyPenaltyDevice, batchSlots, getLimitsPenalty(DecodingPenaltyType::Frequency), "frequency penalty");
    }
    if (mUseMinLength)
    {
        fillBuffers(penaltyParams->minLength, DefaultDecodingParams::getMinLength(), mMinLength, mMinLengthDevice,
            batchSlots, getLimitsPenalty(DecodingPenaltyType::MinLength), "min length");
    }

    // Reset penalty workspace
    auto const workspaceSizePerBatch
        = mDecoderDomain.getMaxDecodingTokens() * mConfiguredBeamWidth * mDecoderDomain.getVocabSize();
    for (SizeType32 bi = 0; bi < batchSize; ++bi)
    {
        auto batchSlot = runtime::bufferCast<runtime::SizeType32>(*batchSlots)[bi];

        if (mPenaltyWorkspaceDevice)
        {
            auto deviceSlice = runtime::IBuffer::slice(
                mPenaltyWorkspaceDevice, batchSlot * workspaceSizePerBatch, workspaceSizePerBatch);
            mBufferManager->setZero(*deviceSlice);
        }

        if (mPenaltyWorkspacePrevDevice)
        {
            auto deviceSlice = runtime::IBuffer::slice(
                mPenaltyWorkspacePrevDevice, batchSlot * workspaceSizePerBatch, workspaceSizePerBatch);
            mBufferManager->setZero(*deviceSlice);
        }
    }

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void PenaltyLayer<T>::forwardAsync(std::shared_ptr<BaseDecodingOutputs> const& baseOutputs,
    std::shared_ptr<BaseDecodingInputs> const& baseInputs,
    std::shared_ptr<runtime::DecodingLayerWorkspace> const& workspace)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    NVTX3_SCOPED_RANGE(PenaltyLayer_forwardAsync);

    auto outputs = std::dynamic_pointer_cast<BaseDecodingOutputs>(baseOutputs);
    auto params = std::dynamic_pointer_cast<DecodingInputs>(baseInputs);

    auto const localDecoderDomain = getLocalDecoderDomain(params, mDecoderDomain);
    auto const maxSeqLen = outputs->outputIds->getDimension<-1>();

    if (mLogitsPtrsHost->data() == nullptr)
    {
        mLogitsPtrsHost->reshape(
            ITensor::makeShape({static_cast<int32_t>(maxSeqLen), static_cast<int32_t>(mDecoderDomain.getBatchSize())}));
        mRuntimeMaxSeqLen = maxSeqLen;
    }

    mCyclicStep = mCyclicStep % mRuntimeMaxSeqLen;

    TensorPtr logitsPtrsHost = ITensor::slice(mLogitsPtrsHost, mCyclicStep, 1);
    logitsPtrsHost->squeeze(0);
    auto logitsPtrsHostData = bufferCast<T const*>(*logitsPtrsHost);
    for (SizeType32 bi = 0; bi < localDecoderDomain.getBatchSize(); bi++)
    {
        if (params->logitsVec)
        {
            TLLM_CHECK_WITH_INFO(params->logitsVec->size() == static_cast<size_t>(localDecoderDomain.getBatchSize()),
                "Logits vector size (%lu) is not equal to the batchSize (%d)", params->logitsVec->size(),
                localDecoderDomain.getBatchSize());
            logitsPtrsHostData[bi] = bufferCastOrNull<T>(params->logitsVec.value()[bi]);
        }
        else
        {
            TensorConstPtr logitsForBatchIndex = ITensor::slice(params->logits.value(), ITensor::makeShape({bi}));
            auto const ptrToLogitsForBatchIndex = bufferCastOrNull<T>(logitsForBatchIndex);
            logitsPtrsHostData[bi] = ptrToLogitsForBatchIndex;
        }
    }

    auto const* inputLengths = bufferCastOrNull<SizeType32>(params->inputLengths);
    auto embeddingBias = bufferCastOrNull<T>(params->embeddingBias);
    auto const* batchSlotsHostPtr = bufferCast<SizeType32>(*params->batchSlots);
#define GET_PENALTIES(capital_name, type)                                                                              \
    (mUse##capital_name                                                                                                \
        && !allOfBatchSlots(batchSlotsHostPtr, bufferCast<type>(*m##capital_name), localDecoderDomain.getBatchSize(),  \
            DefaultDecodingParams::get##capital_name()))                                                               \
        ? m##capital_name##Device                                                                                      \
        : nullptr;

    auto temperatures = GET_PENALTIES(Temperature, float);
    auto repetitionPenalties = GET_PENALTIES(RepetitionPenalty, float);
    auto presencePenalties = GET_PENALTIES(PresencePenalty, float);
    auto frequencyPenalties = GET_PENALTIES(FrequencyPenalty, float);
    auto minLengths = GET_PENALTIES(MinLength, SizeType32);

#undef GET_PENALTIES

    auto* const tokensPerStep = bufferCastOrNull<SizeType32>(params->curTokensPerStep);

    InvokeBatchApplyPenaltyParams<T> penaltyParams{};

    TensorPtr logitsPtrsHostSlice = ITensor::slice(logitsPtrsHost, 0, localDecoderDomain.getBatchSize());
    auto [logitsPtrsDeviceSlice] = workspace->mirrorInWorkspace(logitsPtrsHostSlice);
    auto runtimeLogits = workspace->getDeviceRuntimeLogits();
    penaltyParams.inputLogits = reinterpret_cast<T const* const*>(bufferCast<T const*>(*logitsPtrsDeviceSlice));
    penaltyParams.outputLogits = bufferCast<T>(*runtimeLogits);
    penaltyParams.biases = embeddingBias;
    penaltyParams.penaltyWorkspace = bufferCastOrNull<TokenIdType>(mPenaltyWorkspaceDevice);
    penaltyParams.penaltyWorkspacePrev = bufferCastOrNull<TokenIdType>(mPenaltyWorkspacePrevDevice);
    penaltyParams.temperatures = bufferCastOrNull<float>(temperatures);
    penaltyParams.repetitionPenalties = bufferCastOrNull<float>(repetitionPenalties);
    penaltyParams.presencePenalties = bufferCastOrNull<float>(presencePenalties);
    penaltyParams.frequencyPenalties = bufferCastOrNull<float>(frequencyPenalties);
    penaltyParams.batchSize = localDecoderDomain.getBatchSize();
    penaltyParams.beamWidth = localDecoderDomain.getBeamWidth();
    penaltyParams.maxSeqLen = maxSeqLen;
    penaltyParams.vocabSize = mDecoderDomain.getVocabSize();
    penaltyParams.vocabSizePadded = mDecoderDomain.getVocabSizePadded();
    penaltyParams.outputIdsPtr = bufferCast<TokenIdType const*>(*outputs->outputIdsPtr);
    penaltyParams.parentIdsPtr = bufferCast<SizeType32 const*>(*outputs->parentIdsPtr);
    penaltyParams.inputLengths = inputLengths;
    penaltyParams.sequenceLengths = bufferCast<SizeType32>(*outputs->sequenceLength.value());
    penaltyParams.minLengths = bufferCastOrNull<SizeType32>(minLengths);
    penaltyParams.endIds = bufferCast<TokenIdType>(*params->endIds);
    penaltyParams.batchSlots = workspace->getDeviceBatchSlotsPtr();
    penaltyParams.maxTokensPerStep = mDecoderDomain.getMaxDecodingTokens();
    penaltyParams.tokensPerStep = tokensPerStep;
    penaltyParams.finished = (params->finished)
        ? reinterpret_cast<FinishedState const*>(bufferCast<FinishedState::UnderlyingType>(*params->finished.value()))
        : nullptr;
    penaltyParams.stream = getStream();

    if (penaltyParams.beamWidth > 1)
    {
        // Convert logits into logProbs before penalties, only necessary in Beam-Search.
        BiasSoftmaxParams<T> biasSoftmaxParams;
        biasSoftmaxParams.logitsPtrs = const_cast<T**>(penaltyParams.inputLogits);
        biasSoftmaxParams.bias = penaltyParams.biases;
        biasSoftmaxParams.endIds = penaltyParams.endIds;
        biasSoftmaxParams.batchSlots = penaltyParams.batchSlots;
        biasSoftmaxParams.batchSize = penaltyParams.batchSize;
        biasSoftmaxParams.maxBatchSize = mDecoderDomain.getBatchSize();
        biasSoftmaxParams.maxBeamWidth = penaltyParams.beamWidth;
        biasSoftmaxParams.vocabSize = penaltyParams.vocabSize;
        biasSoftmaxParams.vocabSizePadded = penaltyParams.vocabSizePadded;
        biasSoftmaxParams.skipSoftMax = false;
        biasSoftmaxParams.batchSlotsLogits = penaltyParams.batchSlots != nullptr;
        biasSoftmaxParams.checkParams();
        invokeAddBiasSoftMax(biasSoftmaxParams, penaltyParams.stream);
    }

    invokeBatchApplyPenalty(penaltyParams);
    sync_check_cuda_error(penaltyParams.stream);

    mCyclicStep += 1;

    auto const logitsShape = ITensor::makeShape({localDecoderDomain.getBatchSize(),
        mDecoderDomain.getMaxDecodingTokens(), localDecoderDomain.getBeamWidth(), mDecoderDomain.getVocabSizePadded()});
    params->logits = ITensor::view(runtimeLogits, logitsShape);

    if (mDecodingMode.isBeamSearch())
    {
        std::swap(mPenaltyWorkspaceDevice, mPenaltyWorkspacePrevDevice);
    }

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template class PenaltyLayer<float>;
template class PenaltyLayer<half>;

} // namespace tensorrt_llm::layers
