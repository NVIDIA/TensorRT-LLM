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

#include "tensorrt_llm/layers/explicitDraftTokensLayer.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/memoryUtils.h"
#include "tensorrt_llm/kernels/decodingCommon.h"
#include "tensorrt_llm/kernels/penaltyTypes.h"
#include "tensorrt_llm/kernels/speculativeDecoding/explicitDraftTokensKernels.h"
#include "tensorrt_llm/layers/defaultDecodingParams.h"
#include "tensorrt_llm/layers/layerUtils.h"

#include <algorithm>

using namespace tensorrt_llm::common;
using namespace tensorrt_llm::kernels;
using namespace tensorrt_llm::kernels::speculative_decoding;
using namespace tensorrt_llm::runtime;

namespace tensorrt_llm
{
namespace layers
{

template <typename T>
ExplicitDraftTokensLayer<T>::ExplicitDraftTokensLayer(
    DecoderDomain const& decoderDomain, cudaStream_t stream, std::shared_ptr<IAllocator> allocator)
    : BaseLayer(decoderDomain, stream, std::move(allocator))
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    allocateBuffer();

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
ExplicitDraftTokensLayer<T>::~ExplicitDraftTokensLayer()
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    freeBuffer();

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void ExplicitDraftTokensLayer<T>::allocateBuffer()
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    mTemperature.resize(mDecoderDomain.getBatchSize());

    mScanWorkspaceSizeInBytes = invokeScanGenerationLengths(
        nullptr, mScanWorkspaceSizeInBytes, nullptr, nullptr, mDecoderDomain.getBatchSize(), mStream);
    mReduceWorkspaceSizeInBytes = invokeReduceMaxGenerationLengths(
        nullptr, mReduceWorkspaceSizeInBytes, nullptr, nullptr, mDecoderDomain.getBatchSize(), mStream);

    mWorkspaceSizeInBytes = std::max(mScanWorkspaceSizeInBytes, mReduceWorkspaceSizeInBytes);

    std::array<size_t, 8> deviceBufferSizes
        = {sizeof(curandState_t) * mDecoderDomain.getBatchSize(), sizeof(uint64_t) * mDecoderDomain.getBatchSize(),
            mWorkspaceSizeInBytes, sizeof(SizeType32) * mDecoderDomain.getBatchSize(), sizeof(SizeType32),
            sizeof(float) * mDecoderDomain.getBatchSize(), sizeof(SizeType32) * mDecoderDomain.getBatchSize(),
            sizeof(SizeType32) * mDecoderDomain.getBatchSize()
                * mDecoderDomain.getSpeculativeDecodingModule()->getMaxNumPaths()
                * mDecoderDomain.getSpeculativeDecodingModule()->getMaxPathLen()};
    mCurandStatesDevice = mAllocator->reMalloc(mCurandStatesDevice, deviceBufferSizes[0], false);
    mRandomSeedsDevice = mAllocator->reMalloc(mRandomSeedsDevice, deviceBufferSizes[1], false);
    mWorkspaceDevice = mAllocator->reMalloc(mWorkspaceDevice, deviceBufferSizes[2], false);
    mGenerationLengthInclusiveSum = mAllocator->reMalloc(mGenerationLengthInclusiveSum, deviceBufferSizes[3], false);
    mMaxGenerationLength = mAllocator->reMalloc(mMaxGenerationLength, deviceBufferSizes[4], false);
    mTemperatureDevice = mAllocator->reMalloc(mTemperatureDevice, deviceBufferSizes[5], false);
    mBestPathIndicesSlots = mAllocator->reMalloc(mBestPathIndicesSlots, deviceBufferSizes[6], false);
    mLastDraftIndicesSlots = mAllocator->reMalloc(mLastDraftIndicesSlots, deviceBufferSizes[7], false);

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void ExplicitDraftTokensLayer<T>::freeBuffer()
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    mAllocator->free((void**) (&mCurandStatesDevice));
    mAllocator->free((void**) (&mRandomSeedsDevice));
    mAllocator->free((void**) (&mWorkspaceDevice));
    mAllocator->free((void**) (&mGenerationLengthInclusiveSum));
    mAllocator->free((void**) (&mMaxGenerationLength));
    mAllocator->free((void**) (&mTemperatureDevice));
    mAllocator->free((void**) (&mBestPathIndicesSlots));
    mAllocator->free((void**) (&mLastDraftIndicesSlots));

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void ExplicitDraftTokensLayer<T>::setup(SizeType32 batchSize, SizeType32 beamWidth, SizeType32 const* batchSlots,
    std::shared_ptr<BaseSetupParams> const& baseSetupParams)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    auto setupParams = std::dynamic_pointer_cast<ExplicitDraftTokensSetupParams>(baseSetupParams);

    if (setupParams->randomSeed)
    {
        if (setupParams->randomSeed->size() == 1)
        {
            invokeCurandInitialize(
                mCurandStatesDevice, batchSlots, batchSize, setupParams->randomSeed->front(), mStream);
            sync_check_cuda_error();
        }
        else
        {
            TLLM_CHECK_WITH_INFO(setupParams->randomSeed->size() == batchSize, "Random seed vector size mismatch.");
            cudaAutoCpy(mRandomSeedsDevice, setupParams->randomSeed->data(), batchSize, mStream);
            invokeCurandBatchInitialize(mCurandStatesDevice, batchSlots, batchSize, mRandomSeedsDevice, mStream);
            sync_check_cuda_error();
        }
    }
    else
    {
        // Initialize curand states using the default seed 0.
        invokeCurandInitialize(mCurandStatesDevice, batchSlots, batchSize, DefaultDecodingParams::getSeed(), mStream);
    }

    // Setup penalties.
    FillBuffers const fillBuffers{batchSize, mDecoderDomain.getBatchSize(), mStream};

    fillBuffers(setupParams->temperature, DefaultDecodingParams::getTemperature(), mTemperature, mTemperatureDevice,
        batchSlots, getLimitsPenalty(DecodingPenaltyType::Temperature), "temperature penalty");

    fillContextBuffers(batchSize, batchSlots, *setupParams);

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void ExplicitDraftTokensLayer<T>::fillContextBuffers(
    SizeType32 batchSize, SizeType32 const* batchSlots, ExplicitDraftTokensSetupParams const& setupParams)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    FillContextExplicitDraftTokensParams<T> params;
    params.randDataSample = setupParams.randomDataSample.template getPtr<T>();
    params.outputTemperatures = setupParams.temperatures.template getPtr<T>();
    params.inputTemperatures = mTemperatureDevice;
    params.curandState = mCurandStatesDevice;
    params.batchSlots = batchSlots;
    params.batchSize = batchSize;

    params.checkParams();

    invokeFillContextBuffers(params, mStream);

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
void ExplicitDraftTokensLayer<T>::convertPackedMask(
    ExplicitDraftTokensOutputs const& outputs, ExplicitDraftTokensInputs const& inputs)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    auto batchSlots = inputs.seqSlots.template getPtr<SizeType32 const>();
    auto masksDevice = inputs.masks.template getPtr<bool const>();
    auto generationLengths = inputs.generationLengths.template getPtr<SizeType32 const>();
    auto packedMasksDevice = outputs.packedMasks.template getPtr<SizeType32>();

    auto const batchSize = inputs.localBatchSize;

    invokeScanReduceGenerationLengths(batchSize, generationLengths, mWorkspaceDevice, mScanWorkspaceSizeInBytes,
        mGenerationLengthInclusiveSum, mWorkspaceDevice, mReduceWorkspaceSizeInBytes, mMaxGenerationLength, mStream);

    invokeConvertMaskToPackedMask(batchSize, mGenerationLengthInclusiveSum, mMaxGenerationLength, masksDevice,
        batchSlots, mDecoderDomain.getSpeculativeDecodingModule()->getMaxDecodingDraftTokens(),
        mDecoderDomain.getSpeculativeDecodingModule()->getMaxDecodingTokens(), packedMasksDevice, mStream);

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void ExplicitDraftTokensLayer<T>::splitInputDataToBatchSlots(
    ExplicitDraftTokensOutputs const& outputs, ExplicitDraftTokensInputs const& inputs)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    auto const batchSize = inputs.localBatchSize;
    auto const maxSeqLen = outputs.outputIds.shape[outputs.outputIds.shape.size() - 1];

    ExtractExplicitDraftTokensParams<T> params;

    params.outputIds = outputs.outputIds.template getPtr<TokenIdType>();
    params.outputPositionIdsBase = outputs.positionIdsBase.template getPtr<SizeType32>();
    params.outputPositionIds = outputs.nextDraftPosIds.template getPtr<SizeType32>();
    params.outputNextDraftTokens = outputs.nextDraftTokens.template getPtr<TokenIdType>();
    params.unpackedNextDraftTokens = outputs.unpackedNextDraftTokens.template getPtr<TokenIdType>();
    params.unpackedNextDraftIndices = outputs.unpackedNextDraftIndices.template getPtr<SizeType32>();
    params.acceptedLengths = outputs.numNewTokens->template getPtr<SizeType32>();
    params.nextDraftLengths = outputs.nextDraftLengths.template getPtr<SizeType32>();
    params.prevDraftLengths = outputs.prevDraftLengths.template getPtr<SizeType32>();
    params.sequenceLengths = outputs.sequenceLength->template getPtr<SizeType32>();
    params.randDataSample = outputs.randomDataSample.template getPtr<T>();
    params.randDataVerification = outputs.randomDataValidation.template getPtr<T>();
    params.outputDraftProbs = outputs.nextDraftProbs.template getPtr<T>();
    params.outputTemperatures = outputs.temperatures.template getPtr<T>();
    params.outputGenerationLengths = outputs.generationLengths.template getPtr<SizeType32>();
    params.outputBestPathIndices = mBestPathIndicesSlots;
    params.outputLastDraftIndices = mLastDraftIndicesSlots;

    params.batchSlots = inputs.seqSlots.template getPtr<SizeType32 const>();
    params.nextDraftTokens = inputs.nextDraftTokens.template getPtr<TokenIdType const>();
    params.lastDraftTokens = inputs.lastDraftTokens.template getPtr<TokenIdType const>();
    params.inputUnpackedNextDraftIndices = inputs.nextDraftIndices.template getPtr<SizeType32 const>();
    params.bestPathLengths = inputs.bestPathLengths.template getPtr<SizeType32 const>();
    params.bestPathIndices = inputs.bestPathIndices.template getPtr<SizeType32 const>();
    params.inputPositionIdsBase = inputs.positionIdsBase.template getPtr<SizeType32 const>();
    params.packedPositionIds = inputs.packedPosIds.template getPtr<SizeType32 const>();
    params.nextFlatTokens = inputs.nextFlatTokens.template getPtr<TokenIdType const>();
    params.nextDraftProbs = inputs.nextDraftProbs.template getPtr<T const>();
    params.lastGenerationLengths = inputs.lastGenerationLengths.template getPtr<SizeType32 const>();
    params.generationLengthInclusiveSum = mGenerationLengthInclusiveSum;
    params.lastDraftIndices = inputs.lastDraftIndices.template getPtr<SizeType32 const>();
    params.inputTemperatures = mTemperatureDevice;
    params.curandState = mCurandStatesDevice;
    params.batchSize = batchSize;
    params.numPaths = mDecoderDomain.getSpeculativeDecodingModule()->getMaxNumPaths();
    params.maxPathLength = mDecoderDomain.getSpeculativeDecodingModule()->getMaxPathLen();
    params.maxSeqLen = maxSeqLen;
    params.vocabSize = mDecoderDomain.getVocabSizePadded();
    params.numContextRequests = batchSize - inputs.lastDraftTokens.shape[0];
    params.numGenerationRequests = inputs.lastDraftTokens.shape[0];

    params.checkParams();

    // Copy max generation length
    cudaMemcpyAsync(outputs.maxGenLengthHost.template getPtr<SizeType32>(),
        inputs.maxGenLengthDevice.template getPtr<SizeType32 const>(), sizeof(SizeType32), cudaMemcpyDeviceToHost,
        mStream);

    params.checkParams();

    // Copy max generation length
    cudaMemcpyAsync(outputs.maxGenLengthHost.template getPtr<SizeType32>(),
        inputs.maxGenLengthDevice.template getPtr<SizeType32 const>(), sizeof(SizeType32), cudaMemcpyDeviceToHost,
        mStream);

    invokeExtractExplicitDraftTokens(params, mStream);

    invokeCopyProbs(params, mStream);

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void ExplicitDraftTokensLayer<T>::packAcceptedPaths(
    ExplicitDraftTokensOutputs const& outputs, ExplicitDraftTokensInputs const& inputs)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    auto const batchSize = inputs.localBatchSize;

    auto numNewTokens = outputs.numNewTokens->template getPtr<SizeType32 const>();
    auto numNewTokensCumSum = outputs.numNewTokensCumSum.template getPtr<SizeType32>();
    auto pathsOffsets = outputs.pathsOffsets.template getPtr<SizeType32>();
    auto batchSlots = inputs.batchSlots->template getPtr<SizeType32 const>();

    TLLM_CHECK_WITH_INFO(batchSlots != nullptr, "Batch slots must be provided for ExplicitDraftTokensLayer");
    TLLM_CHECK_WITH_INFO(numNewTokens != nullptr, "Accepted lengths must be provided for ExplicitDraftTokensLayer");
    TLLM_CHECK_WITH_INFO(
        numNewTokensCumSum != nullptr, "numNewTokensCumSum must be provided for ExplicitDraftTokensLayer");
    TLLM_CHECK_WITH_INFO(pathsOffsets != nullptr, "pathsOffsets must be provided for ExplicitDraftTokensLayer");
    invokePackAcceptedPaths(numNewTokensCumSum, pathsOffsets, numNewTokens, mBestPathIndicesSlots,
        mLastDraftIndicesSlots, batchSlots, batchSize, mDecoderDomain.getSpeculativeDecodingModule()->getMaxNumPaths(),
        mDecoderDomain.getSpeculativeDecodingModule()->getMaxPathLen(), false, mStream);

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template class ExplicitDraftTokensLayer<float>;
template class ExplicitDraftTokensLayer<half>;

} // namespace layers
} // namespace tensorrt_llm
