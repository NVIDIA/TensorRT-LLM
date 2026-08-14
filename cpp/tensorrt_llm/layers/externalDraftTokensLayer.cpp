/*
 * Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

#include "externalDraftTokensLayer.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/nvtxUtils.h"
#include "tensorrt_llm/kernels/decodingCommon.h"
#include "tensorrt_llm/kernels/samplingTopKKernels.h"
#include "tensorrt_llm/kernels/samplingTopPKernels.h"
#include "tensorrt_llm/kernels/speculativeDecoding/externalDraftTokensKernels.h"
#include "tensorrt_llm/layers/defaultDecodingParams.h"
#include "tensorrt_llm/layers/layerUtils.h"
#include "tensorrt_llm/runtime/runtimeKernels.h"

#include <algorithm>

namespace tksd = tensorrt_llm::kernels::speculative_decoding;

using namespace tensorrt_llm::common;
using namespace tensorrt_llm::kernels;
using namespace tensorrt_llm::runtime;

namespace tensorrt_llm::layers
{

template <typename T>
ExternalDraftTokensLayer<T>::ExternalDraftTokensLayer(executor::DecodingMode const& mode,
    DecoderDomain const& decoderDomain, std::shared_ptr<BufferManager> bufferManager, bool isDeterministic,
    bool isAirTopP)
    : BaseLayer(decoderDomain, bufferManager)
    , mDecodingMode(mode)
    , mIsDeterministic(isDeterministic)
    , mIsAirTopP(isAirTopP)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    TLLM_CHECK_WITH_INFO(!mDecodingMode.isBeamSearch(), "ExternalDraftTokensLayer does not support Beam search mode");

    auto const deviceId = getDevice();
    TLLM_CUDA_CHECK(cudaGetDeviceProperties(&mDeviceProp, deviceId));

    allocateBuffer(decoderDomain.getBatchSize());

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void ExternalDraftTokensLayer<T>::allocateBuffer(SizeType32 batchSize)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    // top k workspace size
    auto workspaceSize = getTopKWorkspaceSize<T>(batchSize, 1, TOP_K_MAX, mDecoderDomain.getVocabSizePadded());
    mWorkspaceSize = std::max(workspaceSize, mWorkspaceSize);
    // top p workspace size
    workspaceSize = getTopPWorkspaceSize<T>(batchSize, mDecoderDomain.getVocabSizePadded());
    mWorkspaceSize = std::max(workspaceSize, mWorkspaceSize);

    // multinomial (top p == 1) workspace size
    workspaceSize = getAirTopPWorkspaceSize<T>(batchSize, mDecoderDomain.getVocabSizePadded(), mIsDeterministic);
    mWorkspaceSize = std::max(workspaceSize, mWorkspaceSize);

    // batchsize here is maxBatchSize
    auto const batchSizeShape = ITensor::makeShape({batchSize});

    mCurandStatesDevice
        = mBufferManager->gpu(ITensor::makeShape({batchSize, sizeof(curandState_t)}), TRTDataType<int8_t>::value);
    mBatchIsAccepted = mBufferManager->gpu(batchSizeShape, TRTDataType<bool>::value);
    mRuntimeMultinomialDevice = mBufferManager->gpu(batchSizeShape, TRTDataType<float>::value);

    // host buffers.
    mSkipTopKDecodeDevice = mBufferManager->gpu(batchSizeShape, TRTDataType<bool>::value);
    mSkipTopKDecodeHost = BufferManager::pinnedPool(batchSizeShape, TRTDataType<bool>::value);
    mSkipTopPDecodeDevice = mBufferManager->gpu(batchSizeShape, TRTDataType<bool>::value);
    mSkipTopPDecodeHost = BufferManager::pinnedPool(batchSizeShape, TRTDataType<bool>::value);
    auto skipTopPDecodeHostRange = BufferRange<bool>(*mSkipTopPDecodeHost);
    std::fill(skipTopPDecodeHostRange.begin(), skipTopPDecodeHostRange.end(), true);

    mOutputIdsAfterSampling = mBufferManager->gpu(
        ITensor::makeShape({batchSize, mDecoderDomain.getVocabSizePadded()}), TRTDataType<TokenIdType>::value);
    mOutputIdsAfterSamplingPtrsHost = BufferManager::pinned(batchSizeShape, TRTDataType<TokenIdType*>::value);
    mOutputIdsAfterSamplingPtrsDevice = mBufferManager->gpu(batchSizeShape, TRTDataType<TokenIdType*>::value);
    mTargetOutputIds = mBufferManager->gpu(batchSizeShape, TRTDataType<TokenIdType>::value);

    mRuntimeTopKDevice = mBufferManager->gpu(batchSizeShape, TRTDataType<SizeType32>::value);
    mRuntimeTopKHost = BufferManager::cpu(batchSizeShape, TRTDataType<SizeType32>::value);

    mRuntimeTopPDevice = mBufferManager->gpu(batchSizeShape, TRTDataType<float>::value);

    mReturnAllSelectedTokensPerSlotHost = BufferManager::pinned(batchSizeShape, TRTDataType<bool>::value);
    mReturnAllSelectedTokensPerSlotDevice = mBufferManager->gpu(batchSizeShape, TRTDataType<bool>::value);

    mMaskBuffer = mBufferManager->gpu(
        ITensor::makeShape({batchSize, mDecoderDomain.getVocabSizePadded()}), TRTDataType<bool>::value);

    mSetupWorkspaceSize = std::max({mBatchIsAccepted->getSizeInBytes(), mRuntimeMultinomialDevice->getSizeInBytes(),
        mOutputIdsAfterSampling->getSizeInBytes(), mTargetOutputIds->getSizeInBytes(), mMaskBuffer->getSizeInBytes()});

    mTargetLogits = mBufferManager->gpu(
        ITensor::makeShape({batchSize, mDecoderDomain.getVocabSizePadded()}), TRTDataType<T>::value);

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void ExternalDraftTokensLayer<T>::setup(SizeType32 batchSize, SizeType32 beamWidth, TensorConstPtr batchSlots,
    std::shared_ptr<BaseSetupParams> const& baseSetupParams,
    std::shared_ptr<runtime::DecodingLayerWorkspace> const& workspace)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    NVTX3_SCOPED_RANGE(ExternalDraftTokensLayer_setup);

    auto setupParams = std::dynamic_pointer_cast<ExternalDraftTokensSetupParams>(baseSetupParams);

    workspace->initializeDeviceCurandStates(
        setupParams->randomSeed, batchSize, workspace->getDeviceBatchSlots(), mCurandStatesDevice);

    auto& runtimeMultinomialDeviceTensor = const_cast<ITensor&>(*mRuntimeMultinomialDevice);
    tensorrt_llm::runtime::kernels::invokeFill(runtimeMultinomialDeviceTensor, 1.0f, mBufferManager->getStream());

    // Prepare runtime top K
    auto runtimeTopK = setupParams->runtimeTopK.value_or(std::vector{DefaultDecodingParams::getTopK()});
    auto runtimeTopP = setupParams->runtimeTopP.value_or(std::vector{DefaultDecodingParams::getTopP()});

    auto const paramsSize = expandMatchElements(batchSize, runtimeTopK, runtimeTopP);

    TLLM_CHECK_WITH_INFO(paramsSize != 0,
        fmtstr("ExternalDraftTokensLayer got parameter with unexpected size, want 1 or batchSize(%d), got"
               "runtimeTopK.size() = %zu, runtimeTopP.size() = %zu",
            batchSize, runtimeTopK.size(), runtimeTopP.size()));

    for (size_t i = 0; i < paramsSize; ++i)
    {
        auto& topK = runtimeTopK[i];
        auto& topP = runtimeTopP[i];
        clampTopK(topK);
        clampTopP(topP);
        regularizeTopKTopP(topK, topP);
    }

    // Update parameters on both device and host, so we can
    // - determine whether we can skip launch TopK / TopP kernel by examine mSkipTopKDecodeHost / mSkipTopPDecodeHost
    // - select best kernel by examine mRuntimeTopKHost
    // without consulting device memory, or we'll have to do an expensive synchronization.
    SizeType32* topKsPtr = nullptr;
    float* topPsPtr = nullptr;

    if (paramsSize > 1)
    {
        auto initWorkspaceSizes = getTopKInitWorkspaceSizes(batchSize);
        calcAlignedPointers(workspace->getRawWorkspaceDevicePtr(), initWorkspaceSizes)(topKsPtr, topPsPtr);
        DecodingLayerWorkspace::copyToWorkspace(
            *mBufferManager, runtimeTopK, IBuffer::wrap(topKsPtr, initWorkspaceSizes[0] / sizeof(*topKsPtr)));
        DecodingLayerWorkspace::copyToWorkspace(
            *mBufferManager, runtimeTopP, IBuffer::wrap(topPsPtr, initWorkspaceSizes[1] / sizeof(*topPsPtr)));
    }
    auto const* batchSlotsDevicePtr = workspace->getDeviceBatchSlotsPtr();
    auto* skipTopKDecodeDevicePtr = bufferCastOrNull<bool>(mSkipTopKDecodeDevice);
    auto* skipTopPDecodeDevicePtr = bufferCastOrNull<bool>(mSkipTopPDecodeDevice);
    invokeSetupTopKTopPRuntimeArgs(batchSize,                                         //
        {topKsPtr, runtimeTopK.front(), bufferCast<SizeType32>(*mRuntimeTopKDevice)}, //
        {topPsPtr, runtimeTopP.front(), bufferCast<float>(*mRuntimeTopPDevice)},      //
        skipTopKDecodeDevicePtr, skipTopPDecodeDevicePtr, batchSlotsDevicePtr, true, getStream());

    auto const* batchSlotsHostPtr = bufferCast<SizeType32>(*batchSlots);
    auto* skipDecodeTopKHostPtr = bufferCastOrNull<bool>(mSkipTopKDecodeHost);
    auto* skipDecodeTopPHostPtr = bufferCastOrNull<bool>(mSkipTopPDecodeHost);
    topKsPtr = paramsSize > 1 ? runtimeTopK.data() : nullptr;
    invokeSetupTopKTopPRuntimeArgs(batchSize,                                           //
        {topKsPtr, runtimeTopK.front(), bufferCast<SizeType32>(*mRuntimeTopKHost)}, {}, //
        skipDecodeTopKHostPtr, skipDecodeTopPHostPtr, batchSlotsHostPtr, false);

    if (mIsAirTopP)
    {
        auto smCnt = mDeviceProp.multiProcessorCount;
        if (smCnt <= 0)
        {
            auto const deviceId = getDevice();
            cudaDeviceProp prop{};
            TLLM_CUDA_CHECK(cudaGetDeviceProperties(&prop, deviceId));
            smCnt = prop.multiProcessorCount;
        }
        mAirTopPBlockNum
            = calcAirTopPBlockNum<T>(batchSize, mDecoderDomain.getVocabSizePadded(), smCnt, mIsDeterministic);
    }

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void ExternalDraftTokensLayer<T>::prepareInputs(
    std::shared_ptr<BaseDecodingOutputs> const& outputs, std::shared_ptr<BaseDecodingInputs> const& baseInputs)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    // Fill the buffer for selected ids from sampling with zero. -1 will be set as a boundary if topP kernel is required
    auto& outputIdsAfterSamplingTensor = const_cast<ITensor&>(*mOutputIdsAfterSampling);
    mBufferManager->setZero(outputIdsAfterSamplingTensor);

    auto inputs = std::dynamic_pointer_cast<ExternalDraftTokensInputs>(baseInputs);

    if (inputs->step == 0)
    {
        // Prepare mReturnAllSelectedTokensPerSlot
        auto numDraftTokensHost = BufferRange<SizeType32>(*inputs->numDraftTokensHost);
        auto returnAllSelectedTokensPerSlot = BufferRange<bool>(*mReturnAllSelectedTokensPerSlotHost);
        std::transform(numDraftTokensHost.begin(), numDraftTokensHost.end(), returnAllSelectedTokensPerSlot.begin(),
            [](auto numDraftTokens) { return numDraftTokens > 0; });
        mBufferManager->copy(*mReturnAllSelectedTokensPerSlotHost, *mReturnAllSelectedTokensPerSlotDevice);

        // Prepare mOutputIdsAfterSamplingPtrs
        auto outputIdsAfterSamplingPtrsHost = BufferRange<TokenIdType*>(*mOutputIdsAfterSamplingPtrsHost);
        auto outputIdsPtrs = BufferRange<TokenIdType*>(*outputs->outputIdsPtrHost);

        auto const maxBatchSize = mDecoderDomain.getBatchSize();
        for (auto batchSlot = 0; batchSlot < maxBatchSize; ++batchSlot)
        {
            auto outputIdsAfterSamplingSlice = ITensor::slice(mOutputIdsAfterSampling, batchSlot);
            auto* outputIdsAfterSamplingPtr = bufferCast<TokenIdType>(*outputIdsAfterSamplingSlice);

            outputIdsAfterSamplingPtrsHost[batchSlot]
                = returnAllSelectedTokensPerSlot[batchSlot] ? outputIdsAfterSamplingPtr : outputIdsPtrs[batchSlot];
        }
        mBufferManager->copy(*mOutputIdsAfterSamplingPtrsHost, *mOutputIdsAfterSamplingPtrsDevice);
    }
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
};

template <typename T>
void ExternalDraftTokensLayer<T>::forwardAsync(std::shared_ptr<BaseDecodingOutputs> const& outputs,
    std::shared_ptr<BaseDecodingInputs> const& baseInputs,
    std::shared_ptr<runtime::DecodingLayerWorkspace> const& workspace)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    NVTX3_SCOPED_RANGE(ExternalDraftTokensLayer_forwardAsync);

    targetSoftmax(baseInputs, workspace);

    prepareInputs(outputs, baseInputs);

    // The logits from target engine should go through samplings first.
    // gptDecoderBatched.cpp is calling dynamic decoder step by step, in this step, dynamic Decoder already forwarded
    // PenaltyLayer, BanWordsLayer. For (TopK > 0) && (TopK == 0 && TopP == 0), we invoke TopK sampling kernel. The same
    // logic is implemented in SamplingLayer.cpp
    getAllTopKs(outputs, baseInputs, workspace);

    // Only for (TopK == 0 && TopP > 0), we invoke TopP sampling
    getAllTopPs(outputs, baseInputs, workspace);

    // After all selected tokens are filled in mOutputIdsAfterSampling by topK, topP kernels, token acceptance logics
    // starts. First we mask the logits of unselected token id to -inf as HF's TopK, TopP implementation. We compute the
    // logit probs of draft and target and go through acceptance logics.
    acceptDraftTokens(outputs, baseInputs, workspace);

    // If the token of the sequence is not accepted, a multinomial sampling is required for the bonus token.
    // Multinomial sampling is achieved through TopP kernel with TopP = 1 and already weighted-sum target logits.
    // The acceptance result of each batch is used as skipDecode in topP kernel. If is accepted, no sampling is needed
    // (early exit). Forwarding for the next step is also set in this kernel.
    multinomialSampling(outputs, baseInputs, workspace);

    // For the sequence with accepted tokens, we simply forward a step.
    forwardAcceptedTokens(outputs, baseInputs, workspace);

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
size_t ExternalDraftTokensLayer<T>::getWorkspaceSize() const noexcept
{
    return std::max(mWorkspaceSize, mSetupWorkspaceSize);
}

template <typename T>
void ExternalDraftTokensLayer<T>::targetSoftmax(std::shared_ptr<BaseDecodingInputs> const& baseInputs,
    std::shared_ptr<runtime::DecodingLayerWorkspace> const& workspace)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    auto inputs = std::dynamic_pointer_cast<ExternalDraftTokensInputs>(baseInputs);

    auto const batchSize = inputs->logits.value()->getDimension<0>();

    auto const* endIds = bufferCast<TokenIdType>(*inputs->endIds);

    FinishedState const* finishedInput = (inputs->finished)
        ? reinterpret_cast<FinishedState const*>(bufferCast<FinishedState::UnderlyingType>(*inputs->finished.value()))
        : nullptr;

    inputs->curandStates = reinterpret_cast<curandState_t*>(bufferCast<int8_t>(*mCurandStatesDevice));
    inputs->probsComputed = true;

    auto runtimeLogitsPtr = bufferCast<T>(*workspace->getDeviceRuntimeLogits());
    auto logitsPtrsPtr = static_cast<T**>(nullptr);
    auto biasPtr = static_cast<T*>(nullptr);
    auto const* batchSlotsPtr = workspace->getDeviceBatchSlotsPtr();
    mBufferManager->copy(runtimeLogitsPtr, *mTargetLogits);

    BiasSoftmaxParams<T> biasSoftmaxParams;
    biasSoftmaxParams.logits = runtimeLogitsPtr;
    biasSoftmaxParams.logitsPtrs = logitsPtrsPtr;
    biasSoftmaxParams.probs = runtimeLogitsPtr;
    biasSoftmaxParams.bias = biasPtr;
    biasSoftmaxParams.endIds = endIds;
    biasSoftmaxParams.finished = finishedInput;
    biasSoftmaxParams.batchSlots = batchSlotsPtr;
    biasSoftmaxParams.batchSize = batchSize;
    biasSoftmaxParams.maxBatchSize = mDecoderDomain.getBatchSize();
    biasSoftmaxParams.maxBeamWidth = 1;
    biasSoftmaxParams.vocabSize = mDecoderDomain.getVocabSize();
    biasSoftmaxParams.vocabSizePadded = mDecoderDomain.getVocabSizePadded();
    biasSoftmaxParams.skipSoftMax = false;
    biasSoftmaxParams.batchSlotsLogits = false;
    biasSoftmaxParams.checkParams();

    invokeAddBiasSoftMax(biasSoftmaxParams, getStream());
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void ExternalDraftTokensLayer<T>::acceptDraftTokens(std::shared_ptr<BaseDecodingOutputs> const& outputs,
    std::shared_ptr<BaseDecodingInputs> const& baseInputs,
    std::shared_ptr<runtime::DecodingLayerWorkspace> const& workspace)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    NVTX3_SCOPED_RANGE(ExternalDraftTokensLayer_acceptDraftTokens);

    auto inputs = std::dynamic_pointer_cast<ExternalDraftTokensInputs>(baseInputs);

    auto const draftLogitsShape = (*inputs->draftLogits).getShape();
    auto const maxBatchSize = mDecoderDomain.getBatchSize();
    auto const maxTokensPerStep = draftLogitsShape.d[1]; // 1
    auto const batchSize = static_cast<SizeType32>(inputs->logits.value()->getDimension<0>());
    auto constexpr beamWidth = 1;

    FinishedState const* finishedInput = (inputs->finished)
        ? reinterpret_cast<FinishedState const*>(bufferCastOrNull<FinishedState::UnderlyingType>(inputs->finished))
        : nullptr;

    FinishedState* finishedOutput = (outputs->finished)
        ? reinterpret_cast<FinishedState*>(bufferCastOrNull<FinishedState::UnderlyingType>(outputs->finished))
        : nullptr;

    tksd::invokeMaskTargetLogits(batchSize, bufferCast<T>(*mTargetLogits), workspace->getDeviceBatchSlotsPtr(),
        beamWidth, mDecoderDomain.getVocabSizePadded(), finishedInput, maxBatchSize,
        bufferCast<SizeType32>(*mOutputIdsAfterSampling), bufferCastOrNull<SizeType32>(mRuntimeTopKDevice),
        bufferCast<bool>(*mMaskBuffer), getStream());

    auto const* batchSlotsHost = bufferCast<SizeType32>(*inputs->batchSlots);
    auto const* useDraftLogitsHostPtr = bufferCastOrNull<bool>(inputs->useDraftLogitsHost);
    auto const skipDraftLogits = allOfBatchSlots(batchSlotsHost, useDraftLogitsHostPtr, batchSize, false);

    if (!skipDraftLogits && inputs->step == 0)
    {
        BiasSoftmaxParams<T> biasSoftmaxParams;
        biasSoftmaxParams.logits = bufferCast<T>(*inputs->draftLogits);
        biasSoftmaxParams.probs = bufferCast<T>(*inputs->draftProbs);
        biasSoftmaxParams.finished = finishedInput;
        biasSoftmaxParams.batchSlots = workspace->getDeviceBatchSlotsPtr();
        biasSoftmaxParams.batchSize = batchSize;
        biasSoftmaxParams.maxBatchSize = maxBatchSize;
        biasSoftmaxParams.maxBeamWidth = beamWidth * maxTokensPerStep;
        biasSoftmaxParams.vocabSize = mDecoderDomain.getVocabSize();
        biasSoftmaxParams.vocabSizePadded = mDecoderDomain.getVocabSizePadded();
        biasSoftmaxParams.skipSoftMax = false;
        biasSoftmaxParams.batchSlotsLogits = true;
        biasSoftmaxParams.checkParams();
        invokeAddBiasSoftMax(biasSoftmaxParams, getStream());
    }

    {
        BiasSoftmaxParams<T> biasSoftmaxParams;
        biasSoftmaxParams.logits = bufferCast<T>(*mTargetLogits);
        biasSoftmaxParams.probs = bufferCast<T>(*inputs->targetProbs);
        biasSoftmaxParams.finished = finishedInput;
        biasSoftmaxParams.batchSlots = workspace->getDeviceBatchSlotsPtr();
        biasSoftmaxParams.batchSize = batchSize;
        biasSoftmaxParams.maxBatchSize = maxBatchSize;
        biasSoftmaxParams.maxBeamWidth = beamWidth;
        biasSoftmaxParams.vocabSize = mDecoderDomain.getVocabSize();
        biasSoftmaxParams.vocabSizePadded = mDecoderDomain.getVocabSizePadded();
        biasSoftmaxParams.skipSoftMax = false;
        biasSoftmaxParams.batchSlotsLogits = false;
        biasSoftmaxParams.checkParams();
        invokeAddBiasSoftMax(biasSoftmaxParams, getStream());
    }

    sync_check_cuda_error(getStream());

    tksd::invokeAcceptDraftTokens(batchSize, bufferCast<T>(*inputs->draftProbs), bufferCast<T>(*inputs->targetProbs),
        bufferCast<SizeType32>(*inputs->numDraftTokens), bufferCast<bool>(*inputs->useDraftLogits),
        bufferCast<TokenIdType>(*inputs->draftTokenIds), finishedInput, finishedOutput, inputs->curandStates,
        workspace->getDeviceBatchSlotsPtr(), maxTokensPerStep, beamWidth, mDecoderDomain.getVocabSizePadded(),
        inputs->useRandomAcceptanceThreshold, inputs->constantThreshold, inputs->step,
        bufferCast<bool>(*mBatchIsAccepted), bufferCast<SizeType32>(*mTargetOutputIds), getStream());

    sync_check_cuda_error(getStream());

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void ExternalDraftTokensLayer<T>::multinomialSampling(std::shared_ptr<BaseDecodingOutputs> const& outputs,
    std::shared_ptr<BaseDecodingInputs> const& baseInputs,
    std::shared_ptr<runtime::DecodingLayerWorkspace> const& workspace)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    NVTX3_SCOPED_RANGE(ExternalDraftTokensLayer_multinomialSampling);

    auto inputs = std::dynamic_pointer_cast<ExternalDraftTokensInputs>(baseInputs);

    auto const batchSize = inputs->logits.value()->getDimension<0>();
    auto probs = bufferCastOrNull<T>(inputs->targetProbs);
    auto* sequenceLength = bufferCastOrNull<SizeType32>(outputs->sequenceLength);
    auto const* endIds = bufferCastOrNull<TokenIdType>(inputs->endIds);

    FinishedState* finishedOutput = (outputs->finished)
        ? reinterpret_cast<FinishedState*>(bufferCastOrNull<FinishedState::UnderlyingType>(outputs->finished))
        : nullptr;

    TopPSamplingKernelParams<T> params{};
    params.probs = probs;
    params.outputIdsPtrs = bufferCastOrNull<TokenIdType*>(outputs->outputIdsPtr);
    params.workspace = workspace->getRawWorkspaceDevicePtr();
    params.topPs = bufferCastOrNull<float>(mRuntimeMultinomialDevice);
    params.sequenceLength = sequenceLength;
    params.endIds = endIds;
    params.batchSlots = workspace->getDeviceBatchSlotsPtr();
    params.finishedInput = nullptr;
    params.finishedOutput = finishedOutput;
    params.skipDecode = bufferCastOrNull<bool>(mBatchIsAccepted);
    params.cumLogProbs = nullptr;
    params.outputLogProbs = nullptr;
    params.curandState = inputs->curandStates;
    params.batchSize = batchSize;
    params.maxBatchSize = mDecoderDomain.getBatchSize();
    params.vocabSizePadded = mDecoderDomain.getVocabSizePadded();

    if (!mIsAirTopP)
    {
        invokeBatchTopPSampling<T>(params, getStream());
    }
    else
    {
        params.blockNum = mAirTopPBlockNum;
        params.isDeterministic = mIsDeterministic;
        invokeBatchAirTopPSampling<T>(params, getStream());
    }

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void ExternalDraftTokensLayer<T>::getAllTopKs(std::shared_ptr<BaseDecodingOutputs> const& outputs,
    std::shared_ptr<BaseDecodingInputs> const& baseInputs,
    std::shared_ptr<runtime::DecodingLayerWorkspace> const& workspace)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    NVTX3_SCOPED_RANGE(ExternalDraftTokensLayer_getAllTopKs);

    auto inputs = std::dynamic_pointer_cast<ExternalDraftTokensInputs>(baseInputs);

    auto logits = bufferCastOrNull<T>(inputs->logits);

    auto const batchSize = static_cast<SizeType32>(inputs->logits.value()->getDimension<0>());

    auto const* batchSlotsHost = bufferCast<SizeType32>(*inputs->batchSlots);
    auto const* skipDecodeHostPtr = bufferCastOrNull<bool>(mSkipTopKDecodeHost);
    auto const skip = allOfBatchSlots(batchSlotsHost, skipDecodeHostPtr, batchSize, true);
    if (skip)
    {
        return;
    }

    auto* sequenceLength = bufferCastOrNull<SizeType32>(outputs->sequenceLength);
    auto const* endIds = bufferCastOrNull<TokenIdType>(inputs->endIds);

    FinishedState const* finishedInput = (inputs->finished)
        ? reinterpret_cast<FinishedState const*>(bufferCastOrNull<FinishedState::UnderlyingType>(inputs->finished))
        : nullptr;
    FinishedState* finishedOutput = (outputs->finished)
        ? reinterpret_cast<FinishedState*>(bufferCastOrNull<FinishedState::UnderlyingType>(outputs->finished))
        : nullptr;

    auto const* runtimeTopKHostPtr = bufferCast<SizeType32>(*mRuntimeTopKHost);

    TopKSamplingKernelParams<T> params{};
    params.logProbs = logits;
    params.outputIdsPtrs = bufferCastOrNull<TokenIdType*>(mOutputIdsAfterSamplingPtrsDevice);
    params.workspace = workspace->getRawWorkspaceDevicePtr();
    params.endIds = endIds;
    params.sequenceLengths = sequenceLength;
    params.maxTopP = 1.0F;
    params.topPs = bufferCastOrNull<float>(mRuntimeTopPDevice);
    params.maxTopK = maxOfBatchSlots(batchSlotsHost, runtimeTopKHostPtr, batchSize);
    params.topKs = bufferCastOrNull<SizeType32>(mRuntimeTopKDevice);
    params.batchSlots = workspace->getDeviceBatchSlotsPtr();
    params.finishedInput = finishedInput;
    params.finishedOutput = finishedOutput;
    params.skipDecode = bufferCastOrNull<bool>(mSkipTopKDecodeDevice);
    params.curandState = inputs->curandStates;
    params.batchSize = batchSize;
    params.maxBatchSize = mDecoderDomain.getBatchSize();
    params.maxTokensPerStep = 1;
    params.vocabSizePadded = mDecoderDomain.getVocabSizePadded();
    params.returnAllSelectedTokens = true;
    params.returnAllSelectedTokensPerSlot = bufferCastOrNull<bool>(mReturnAllSelectedTokensPerSlotDevice);
    params.logitsHasProbs = inputs->probsComputed;
    params.outputIdCurrentStep = bufferCastOrNull<TokenIdType>(mTargetOutputIds);
    params.skipOutputIdCurrentStep = bufferCast<bool>(*inputs->useDraftLogits);

    invokeBatchTopKSampling(params, getStream());
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void ExternalDraftTokensLayer<T>::getAllTopPs(std::shared_ptr<BaseDecodingOutputs> const& outputs,
    std::shared_ptr<BaseDecodingInputs> const& baseInputs,
    std::shared_ptr<runtime::DecodingLayerWorkspace> const& workspace)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    NVTX3_SCOPED_RANGE(ExternalDraftTokensLayer_getAllTopPs);

    auto inputs = std::dynamic_pointer_cast<ExternalDraftTokensInputs>(baseInputs);

    auto logits = bufferCastOrNull<T>(inputs->logits);

    auto const batchSize = static_cast<SizeType32>(inputs->logits.value()->getDimension<0>());

    auto const* batchSlotsHost = bufferCast<SizeType32>(*inputs->batchSlots);
    auto const* skipDecodeHostPtr = bufferCastOrNull<bool>(mSkipTopPDecodeHost);
    auto const skip = allOfBatchSlots(batchSlotsHost, skipDecodeHostPtr, batchSize, true);
    if (skip)
    {
        return;
    }

    auto* sequenceLength = bufferCastOrNull<SizeType32>(outputs->sequenceLength);
    auto const* endIds = bufferCastOrNull<TokenIdType>(inputs->endIds);

    FinishedState const* finishedInput = (inputs->finished)
        ? reinterpret_cast<FinishedState const*>(bufferCastOrNull<FinishedState::UnderlyingType>(inputs->finished))
        : nullptr;
    FinishedState* finishedOutput = (outputs->finished)
        ? reinterpret_cast<FinishedState*>(bufferCastOrNull<FinishedState::UnderlyingType>(outputs->finished))
        : nullptr;

    TopPSamplingKernelParams<T> params{};
    params.probs = logits;
    params.outputIdsPtrs = bufferCastOrNull<TokenIdType*>(mOutputIdsAfterSamplingPtrsDevice);
    params.workspace = workspace->getRawWorkspaceDevicePtr();
    params.endIds = endIds;
    params.sequenceLength = sequenceLength;
    params.topPs = bufferCastOrNull<float>(mRuntimeTopPDevice);
    params.batchSlots = workspace->getDeviceBatchSlotsPtr();
    params.finishedInput = finishedInput;
    params.finishedOutput = finishedOutput;
    params.skipDecode = bufferCastOrNull<bool>(mSkipTopPDecodeDevice);
    params.curandState = inputs->curandStates;
    params.batchSize = batchSize;
    params.maxBatchSize = mDecoderDomain.getBatchSize();
    params.vocabSizePadded = mDecoderDomain.getVocabSizePadded();
    params.returnAllSelectedTokens = true;
    params.returnAllSelectedTokensPerSlot = bufferCastOrNull<bool>(mReturnAllSelectedTokensPerSlotDevice);
    params.outputIdCurrentStep = bufferCastOrNull<TokenIdType>(mTargetOutputIds);
    params.skipOutputIdCurrentStep = bufferCast<bool>(*inputs->useDraftLogits);

    invokeBatchTopPSampling<T>(params, getStream());

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void ExternalDraftTokensLayer<T>::forwardAcceptedTokens(std::shared_ptr<BaseDecodingOutputs> const& outputs,
    std::shared_ptr<BaseDecodingInputs> const& baseInputs,
    std::shared_ptr<runtime::DecodingLayerWorkspace> const& workspace)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    NVTX3_SCOPED_RANGE(ExternalDraftTokensLayer_forwardAcceptedTokens);

    auto inputs = std::dynamic_pointer_cast<ExternalDraftTokensInputs>(baseInputs);
    auto const batchSize = inputs->logits.value()->getDimension<0>();

    auto const draftLogitsShape = (*inputs->draftLogits).getShape();
    auto const maxTokensPerStep = draftLogitsShape.d[1]; // 1

    FinishedState* finishedOutput = (outputs->finished)
        ? reinterpret_cast<FinishedState*>(bufferCastOrNull<FinishedState::UnderlyingType>(outputs->finished))
        : nullptr;

    tksd::invokeForwardAcceptedTokens(batchSize, workspace->getDeviceBatchSlotsPtr(),
        bufferCast<bool>(*mBatchIsAccepted), bufferCastOrNull<SizeType32>(outputs->sequenceLength),
        bufferCast<TokenIdType>(*inputs->draftTokenIds), bufferCastOrNull<TokenIdType*>(outputs->outputIdsPtr),
        inputs->step, maxTokensPerStep, bufferCastOrNull<TokenIdType>(inputs->endIds), finishedOutput, getStream());

    sync_check_cuda_error(getStream());

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template class ExternalDraftTokensLayer<float>;
template class ExternalDraftTokensLayer<half>;

} // namespace tensorrt_llm::layers
