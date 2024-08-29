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

#include "explicitDraftTokensLayer.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/kernels/decodingCommon.h"
#include "tensorrt_llm/kernels/penaltyTypes.h"
#include "tensorrt_llm/kernels/speculativeDecoding/common.h"
#include "tensorrt_llm/kernels/speculativeDecoding/explicitDraftTokensKernels.h"
#include "tensorrt_llm/layers/defaultDecodingParams.h"
#include "tensorrt_llm/layers/layerUtils.h"

#include <algorithm>

using namespace tensorrt_llm::common;
using namespace tensorrt_llm::kernels;
using namespace tensorrt_llm::kernels::speculative_decoding;
using namespace tensorrt_llm::runtime;

namespace tensorrt_llm::layers
{

template <typename T>
ExplicitDraftTokensLayer<T>::ExplicitDraftTokensLayer(
    DecoderDomain const& decoderDomain, std::shared_ptr<BufferManager> bufferManager)
    : BaseLayer(decoderDomain, bufferManager)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    allocateBuffer();

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void ExplicitDraftTokensLayer<T>::allocateBuffer()
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    mTemperature
        = mBufferManager->pinnedPool(ITensor::makeShape({mDecoderDomain.getBatchSize()}), TRTDataType<float>::value);

    mScanWorkspaceSizeInBytes = invokeScanGenerationLengths(
        nullptr, mScanWorkspaceSizeInBytes, nullptr, nullptr, mDecoderDomain.getBatchSize(), getStream());
    mReduceWorkspaceSizeInBytes = invokeReduceMaxGenerationLengths(
        nullptr, mReduceWorkspaceSizeInBytes, nullptr, nullptr, mDecoderDomain.getBatchSize(), getStream());

    auto workspaceSizeInBytes = std::max(mScanWorkspaceSizeInBytes, mReduceWorkspaceSizeInBytes);
    mWorkspaceDevice = mBufferManager->gpu(workspaceSizeInBytes, nvinfer1::DataType::kINT8);

    mCurandStatesDevice = mBufferManager->gpu(
        ITensor::makeShape({mDecoderDomain.getBatchSize(), sizeof(curandState_t)}), TRTDataType<int8_t>::value);
    auto const batchSizeShape = ITensor::makeShape({mDecoderDomain.getBatchSize()});
    mRandomSeedsDevice = mBufferManager->gpu(batchSizeShape, TRTDataType<int64_t>::value);
    mGenerationLengthInclusiveSum = mBufferManager->gpu(batchSizeShape, TRTDataType<SizeType32>::value);
    mMaxGenerationLength = mBufferManager->gpu(ITensor::makeShape({1}), TRTDataType<SizeType32>::value);
    mTemperatureDevice = mBufferManager->gpu(batchSizeShape, TRTDataType<float>::value);
    mBestPathIndicesSlots = mBufferManager->gpu(batchSizeShape, TRTDataType<SizeType32>::value);
    mLastDraftIndicesSlots = mBufferManager->gpu(ITensor::makeShape({mDecoderDomain.getBatchSize()
                                                     * mDecoderDomain.getSpeculativeDecodingModule()->getMaxNumPaths()
                                                     * mDecoderDomain.getSpeculativeDecodingModule()->getMaxPathLen()}),
        TRTDataType<SizeType32>::value);

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void ExplicitDraftTokensLayer<T>::setup(SizeType32 batchSize, SizeType32 beamWidth, BufferConstPtr batchSlots,
    std::shared_ptr<BaseSetupParams> const& baseSetupParams)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    auto setupParams = std::dynamic_pointer_cast<ExplicitDraftTokensSetupParams>(baseSetupParams);
    auto batchSlotsPtr = bufferCast<SizeType32>(*batchSlots);
    auto randomSeedDevicePtr = bufferCast<uint64_t>(*mRandomSeedsDevice);
    auto curandStatesDevicePtr = reinterpret_cast<curandState_t*>(bufferCast<int8_t>(*mCurandStatesDevice));

    if (setupParams->randomSeed)
    {
        if (setupParams->randomSeed->size() == 1)
        {
            invokeCurandInitialize(
                curandStatesDevicePtr, batchSlotsPtr, batchSize, setupParams->randomSeed->front(), getStream());
            sync_check_cuda_error();
        }
        else
        {
            TLLM_CHECK_WITH_INFO(setupParams->randomSeed->size() == batchSize, "Random seed vector size mismatch.");
            TensorPtr randomSeedsDeviceSlice = ITensor::slice(mRandomSeedsDevice, 0, batchSize);
            mBufferManager->copy(setupParams->randomSeed.value().data(), *randomSeedsDeviceSlice, MemoryType::kCPU);
            invokeCurandBatchInitialize(
                curandStatesDevicePtr, batchSlotsPtr, batchSize, randomSeedDevicePtr, getStream());
            sync_check_cuda_error();
        }
    }
    else
    {
        // Initialize curand states using the default seed 0.
        invokeCurandInitialize(
            curandStatesDevicePtr, batchSlotsPtr, batchSize, DefaultDecodingParams::getSeed(), getStream());
    }

    // Setup penalties.
    FillBuffers const fillBuffers{batchSize, mDecoderDomain.getBatchSize(), mBufferManager};

    fillBuffers(setupParams->temperature, DefaultDecodingParams::getTemperature(), mTemperature, mTemperatureDevice,
        batchSlots, getLimitsPenalty(DecodingPenaltyType::Temperature), "temperature penalty");

    fillContextBuffers(batchSize, batchSlots, *setupParams);

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void ExplicitDraftTokensLayer<T>::fillContextBuffers(
    SizeType32 batchSize, BufferConstPtr batchSlots, ExplicitDraftTokensSetupParams const& setupParams)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    FillContextExplicitDraftTokensParams<T> params;
    params.randDataSample = bufferCast<T>(*setupParams.randomDataSample);
    params.outputTemperatures = bufferCast<T>(*setupParams.temperatures);
    params.inputTemperatures = bufferCastOrNull<float>(mTemperatureDevice);
    params.curandState = reinterpret_cast<curandState_t*>(bufferCastOrNull<int8_t>(mCurandStatesDevice));
    params.batchSlots = bufferCast<SizeType32>(*batchSlots);
    params.batchSize = batchSize;

    params.checkParams();

    invokeFillContextBuffers(params, getStream());

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void ExplicitDraftTokensLayer<T>::forwardAsync(
    std::shared_ptr<BaseDecodingOutputs> const& baseOutputs, std::shared_ptr<BaseDecodingInputs> const& baseInputs)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    auto inputs = std::dynamic_pointer_cast<ExplicitDraftTokensInputs>(baseInputs);
    auto outputs = std::dynamic_pointer_cast<ExplicitDraftTokensOutputs>(baseOutputs);

    // DO NOT CHANGE THE ORDER.

    // Convert masks to packed masks per request.
    convertPackedMask(*outputs, *inputs);

    // Slice output ids, pos ids, next draft tokens.
    splitInputDataToBatchSlots(*outputs, *inputs);

    // Pack accepted paths for KV cache rewind.
    packAcceptedPaths(*outputs, *inputs);

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
size_t ExplicitDraftTokensLayer<T>::getWorkspaceSize() const noexcept
{
    return mWorkspaceDevice->getSizeInBytes();
}

template <typename T>
void ExplicitDraftTokensLayer<T>::convertPackedMask(
    ExplicitDraftTokensOutputs const& outputs, ExplicitDraftTokensInputs const& inputs)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    auto batchSlots = bufferCast<SizeType32>(*inputs.seqSlots);
    auto masksDevice = bufferCast<bool>(*inputs.masks);
    auto generationLengths = bufferCast<SizeType32>(*inputs.generationLengths);
    auto packedMasksDevice = bufferCast<SizeType32>(*outputs.packedMasks);

    auto const batchSize = inputs.localBatchSize;

    auto generationLengthInclusiveSumPtr = bufferCastOrNull<SizeType32>(mGenerationLengthInclusiveSum);
    auto workSpaceDevicePtr = mWorkspaceDevice->data();
    auto maxGenerationLengthPtr = bufferCastOrNull<SizeType32>(mMaxGenerationLength);
    invokeScanReduceGenerationLengths(batchSize, generationLengths, workSpaceDevicePtr, mScanWorkspaceSizeInBytes,
        generationLengthInclusiveSumPtr, workSpaceDevicePtr, mReduceWorkspaceSizeInBytes, maxGenerationLengthPtr,
        getStream());

    invokeConvertMaskToPackedMask(batchSize, generationLengthInclusiveSumPtr, maxGenerationLengthPtr, masksDevice,
        batchSlots, mDecoderDomain.getSpeculativeDecodingModule()->getMaxDecodingDraftTokens(),
        mDecoderDomain.getSpeculativeDecodingModule()->getMaxDecodingTokens(), packedMasksDevice, getStream());

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void ExplicitDraftTokensLayer<T>::splitInputDataToBatchSlots(
    ExplicitDraftTokensOutputs const& outputs, ExplicitDraftTokensInputs const& inputs)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    auto const batchSize = inputs.localBatchSize;
    auto const maxSeqLen = outputs.outputIds->getDimension<-1>();

    ExtractExplicitDraftTokensParams<T> params;

    params.outputIds = bufferCast<TokenIdType>(*outputs.outputIds);
    params.outputPositionIdsBase = bufferCast<SizeType32>(*outputs.positionIdsBase);
    params.outputPositionIds = bufferCast<SizeType32>(*outputs.nextDraftPosIds);
    params.outputNextDraftTokens = bufferCast<TokenIdType>(*outputs.nextDraftTokens);
    params.unpackedNextDraftTokens = bufferCast<TokenIdType>(*outputs.unpackedNextDraftTokens);
    params.unpackedNextDraftIndices = bufferCast<SizeType32>(*outputs.unpackedNextDraftIndices);
    params.acceptedLengths = bufferCast<SizeType32>(*outputs.numNewTokens.value());
    params.nextDraftLengths = bufferCast<SizeType32>(*outputs.nextDraftLengths);
    params.prevDraftLengths = bufferCast<SizeType32>(*outputs.prevDraftLengths);
    params.sequenceLengths = bufferCast<SizeType32>(*outputs.sequenceLength.value());
    params.randDataSample = bufferCast<T>(*outputs.randomDataSample);
    params.randDataVerification = bufferCast<T>(*outputs.randomDataValidation);
    params.outputDraftProbs = bufferCast<T>(*outputs.nextDraftProbs);
    params.outputTemperatures = bufferCast<T>(*outputs.temperatures);
    params.outputGenerationLengths = bufferCast<SizeType32>(*outputs.generationLengths);
    params.outputBestPathIndices = bufferCast<SizeType32>(*mBestPathIndicesSlots);
    params.outputLastDraftIndices = bufferCast<SizeType32>(*mLastDraftIndicesSlots);

    params.batchSlots = bufferCast<SizeType32>(*inputs.seqSlots);
    params.nextDraftTokens = bufferCast<TokenIdType>(*inputs.nextDraftTokens);
    params.lastDraftTokens = bufferCast<TokenIdType>(*inputs.lastDraftTokens);
    params.inputUnpackedNextDraftIndices = bufferCast<SizeType32>(*inputs.nextDraftIndices);
    params.bestPathLengths = bufferCast<SizeType32>(*inputs.bestPathLengths);
    params.bestPathIndices = bufferCast<SizeType32>(*inputs.bestPathIndices);
    params.inputPositionIdsBase = bufferCast<SizeType32>(*inputs.positionIdsBase);
    params.packedPositionIds = bufferCast<SizeType32>(*inputs.packedPosIds);
    params.nextFlatTokens = bufferCast<TokenIdType>(*inputs.nextFlatTokens);
    params.nextDraftProbs = bufferCast<T>(*inputs.nextDraftProbs);
    params.lastGenerationLengths = bufferCastOrNull<SizeType32>(inputs.lastGenerationLengths);
    params.generationLengthInclusiveSum = bufferCast<SizeType32>(*mGenerationLengthInclusiveSum);
    params.lastDraftIndices = bufferCast<SizeType32>(*inputs.lastDraftIndices);
    params.inputTemperatures = bufferCast<float>(*mTemperatureDevice);
    params.curandState = reinterpret_cast<curandState_t*>(bufferCastOrNull<int8_t>(mCurandStatesDevice));
    params.batchSize = batchSize;
    params.numPaths = mDecoderDomain.getSpeculativeDecodingModule()->getMaxNumPaths();
    params.maxPathLength = mDecoderDomain.getSpeculativeDecodingModule()->getMaxPathLen();
    params.maxSeqLen = maxSeqLen;
    params.vocabSize = mDecoderDomain.getVocabSizePadded();
    params.numContextRequests = batchSize - inputs.lastDraftTokens->getDimension<0>();
    params.numGenerationRequests = inputs.lastDraftTokens->getDimension<0>();

    params.checkParams();

    // Copy max generation length
    mBufferManager->copy(*inputs.maxGenLengthDevice, *outputs.maxGenLengthHost);

    invokeExtractExplicitDraftTokens(params, getStream());

    invokeCopyProbs(params, getStream());

    // Copy generation lengths
    mBufferManager->copy(*outputs.generationLengths, *outputs.generationLengthsHost);

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void ExplicitDraftTokensLayer<T>::packAcceptedPaths(
    ExplicitDraftTokensOutputs const& outputs, ExplicitDraftTokensInputs const& inputs)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    auto const batchSize = inputs.localBatchSize;

    auto numNewTokens = bufferCast<SizeType32>(*outputs.numNewTokens.value());
    auto numNewTokensCumSum = bufferCast<SizeType32>(*outputs.numNewTokensCumSum);
    auto pathsOffsets = bufferCast<SizeType32>(*outputs.pathsOffsets);
    auto batchSlots = bufferCast<SizeType32>(*inputs.batchSlots);
    auto bestPathIndicesSlotsPtr = bufferCastOrNull<SizeType32>(mBestPathIndicesSlots);
    auto lastDraftIndicesSlotsPtr = bufferCastOrNull<SizeType32>(mLastDraftIndicesSlots);

    TLLM_CHECK_WITH_INFO(batchSlots != nullptr, "Batch slots must be provided for ExplicitDraftTokensLayer");
    TLLM_CHECK_WITH_INFO(numNewTokens != nullptr, "Accepted lengths must be provided for ExplicitDraftTokensLayer");
    TLLM_CHECK_WITH_INFO(
        numNewTokensCumSum != nullptr, "numNewTokensCumSum must be provided for ExplicitDraftTokensLayer");
    TLLM_CHECK_WITH_INFO(pathsOffsets != nullptr, "pathsOffsets must be provided for ExplicitDraftTokensLayer");
    invokePackAcceptedPaths(numNewTokensCumSum, pathsOffsets, numNewTokens, bestPathIndicesSlotsPtr,
        lastDraftIndicesSlotsPtr, batchSlots, batchSize,
        mDecoderDomain.getSpeculativeDecodingModule()->getMaxNumPaths(),
        mDecoderDomain.getSpeculativeDecodingModule()->getMaxPathLen(), false, getStream());

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template class ExplicitDraftTokensLayer<float>;
template class ExplicitDraftTokensLayer<half>;

} // namespace tensorrt_llm::layers
