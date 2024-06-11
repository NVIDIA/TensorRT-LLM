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
#include "tensorrt_llm/kernels/decodingKernels.h"
#include "tensorrt_llm/kernels/penaltyKernels.h"
#include "tensorrt_llm/kernels/speculativeDecoding/explicitDraftTokensKernels.h"
#include "tensorrt_llm/layers/defaultDecodingParams.h"
#include "tensorrt_llm/layers/layerUtils.h"
#include "tensorrt_llm/runtime/bufferManager.h"
#include "tensorrt_llm/runtime/iBuffer.h"

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

    mScanWorkspaceSizeInBytes = invokeScanSpecDecodingGenerationLengths(
        nullptr, mScanWorkspaceSizeInBytes, nullptr, nullptr, mDecoderDomain.getBatchSize(), mStream);
    mReduceWorkspaceSizeInBytes = invokeReduceMaxSpecDecodingGenerationLengths(
        nullptr, mReduceWorkspaceSizeInBytes, nullptr, nullptr, mDecoderDomain.getBatchSize(), mStream);

    mWorkspaceSizeInBytes = std::max(mScanWorkspaceSizeInBytes, mReduceWorkspaceSizeInBytes);

    std::array<size_t, 6> deviceBufferSizes
        = {sizeof(curandState_t) * mDecoderDomain.getBatchSize(), sizeof(uint64_t) * mDecoderDomain.getBatchSize(),
            mWorkspaceSizeInBytes, sizeof(SizeType32) * mDecoderDomain.getBatchSize(), sizeof(SizeType32),
            sizeof(float) * mDecoderDomain.getBatchSize()};
    mCurandStatesDevice = mAllocator->reMalloc(mCurandStatesDevice, deviceBufferSizes[0], false);
    mRandomSeedsDevice = mAllocator->reMalloc(mRandomSeedsDevice, deviceBufferSizes[1], false);
    mWorkspaceDevice = mAllocator->reMalloc(mWorkspaceDevice, deviceBufferSizes[2], false);
    mGenerationLengthInclusiveSum = mAllocator->reMalloc(mGenerationLengthInclusiveSum, deviceBufferSizes[3], false);
    mMaxGenerationLength = mAllocator->reMalloc(mMaxGenerationLength, deviceBufferSizes[4], false);
    mTemperatureDevice = mAllocator->reMalloc(mTemperatureDevice, deviceBufferSizes[5], false);

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

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void ExplicitDraftTokensLayer<T>::setup(SizeType32 batchSize, SizeType32 beamWidth, SizeType32 const* batchSlots,
    std::shared_ptr<BaseSetupParams> baseSetupParams)
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

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void ExplicitDraftTokensLayer<T>::forwardAsync(
    std::shared_ptr<BaseOutputParams> baseOutputs, std::shared_ptr<BaseInputParams> baseInputs)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    auto inputs = std::dynamic_pointer_cast<ExplicitDraftTokensInputParams>(baseInputs);
    auto outputs = std::dynamic_pointer_cast<DynamicDecodeOutputParams>(baseOutputs);

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
    DynamicDecodeOutputParams const& outputs, ExplicitDraftTokensInputParams const& inputs)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    auto batchSlots = inputs.batch_slots->template getPtr<SizeType32 const>();
    auto masksDevice = inputs.masks.template getPtr<bool const>();
    auto specDecodingGenerationLengths = inputs.specDecodingGenerationLengths.template getPtr<SizeType32 const>();
    auto packedMasksDevice = outputs.explicitDraftTokensOutputs->packedMasks.template getPtr<SizeType32>();

    auto const batchSize = inputs.batch_slots->shape[0];

    invokeScanReduceSpecDecodingGenerationLengths(batchSize, specDecodingGenerationLengths, mWorkspaceDevice,
        mScanWorkspaceSizeInBytes, mGenerationLengthInclusiveSum, mWorkspaceDevice, mReduceWorkspaceSizeInBytes,
        mMaxGenerationLength, mStream);

    invokeConvertSpecDecodingMaskToPackedMask(batchSize, mGenerationLengthInclusiveSum, mMaxGenerationLength,
        masksDevice, batchSlots, mDecoderDomain.getSpeculativeDecodingModule()->getMaxDecodingDraftTokens(),
        mDecoderDomain.getSpeculativeDecodingModule()->getMaxDecodingTokens(), packedMasksDevice, mStream);

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void ExplicitDraftTokensLayer<T>::splitInputDataToBatchSlots(
    DynamicDecodeOutputParams const& outputs, ExplicitDraftTokensInputParams const& inputs)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    auto const batchSize = inputs.batch_slots->shape[0];
    auto const maxSeqLen = outputs.output_ids.shape[outputs.output_ids.shape.size() - 1];

    ExtractExplicitDraftTokensParams<T> params;

    params.outputIds = outputs.output_ids.template getPtr<TokenIdType>();
    params.outputPositionIdsBase = outputs.explicitDraftTokensOutputs->positionIdsBase.template getPtr<SizeType32>();
    params.outputPositionIds = outputs.explicitDraftTokensOutputs->nextDraftPosIds.template getPtr<SizeType32>();
    params.outputNextDraftTokens = outputs.explicitDraftTokensOutputs->nextDraftTokens.template getPtr<TokenIdType>();
    params.unpackedNextDraftTokens
        = outputs.explicitDraftTokensOutputs->unpackedNextDraftTokens.template getPtr<TokenIdType>();
    params.unpackedNextDraftIndices
        = outputs.explicitDraftTokensOutputs->unpackedNextDraftIndices.template getPtr<SizeType32>();
    params.acceptedLengths = outputs.explicitDraftTokensOutputs->acceptedLengths.template getPtr<SizeType32>();
    params.nextDraftLengths = outputs.explicitDraftTokensOutputs->nextDraftLengths.template getPtr<SizeType32>();
    params.sequenceLengths = outputs.sequence_length->template getPtr<SizeType32>();
    params.randDataSample = outputs.explicitDraftTokensOutputs->randomDataSample.template getPtr<T>();
    params.randDataVerification = outputs.explicitDraftTokensOutputs->randomDataValidation.template getPtr<T>();
    params.outputDraftProbs = outputs.explicitDraftTokensOutputs->nextDraftProbs.template getPtr<T>();
    params.outputTemperatures = outputs.explicitDraftTokensOutputs->temperatures.template getPtr<T>();

    params.batchSlots = inputs.batch_slots->template getPtr<SizeType32 const>();
    params.nextDraftTokens = inputs.nextDraftTokens.template getPtr<TokenIdType const>();
    params.lastDraftTokens = inputs.lastDraftTokens.template getPtr<TokenIdType const>();
    params.inputUnpackedNextDraftIndices = inputs.nextDraftIndices.template getPtr<SizeType32 const>();
    params.bestPathLengths = inputs.bestPathLengths.template getPtr<SizeType32 const>();
    params.bestPathIndices = inputs.bestPathIndices.template getPtr<SizeType32 const>();
    params.inputPositionIdsBase = inputs.positionIdsBase.template getPtr<SizeType32 const>();
    params.packedPositionIds = inputs.packedPosIds.template getPtr<SizeType32 const>();
    params.nextFlatTokens = inputs.nextFlatTokens.template getPtr<TokenIdType const>();
    params.nextDraftProbs = inputs.nextDraftProbs.template getPtr<T const>();
    params.generationLengthInclusiveSum = mGenerationLengthInclusiveSum;
    params.inputTemperatures = mTemperatureDevice;
    params.curandState = mCurandStatesDevice;
    params.curandState = mCurandStatesDevice;
    params.batchSize = batchSize;
    params.numPaths = mDecoderDomain.getSpeculativeDecodingModule()->getMaxNumPaths();
    params.maxPathLength = mDecoderDomain.getSpeculativeDecodingModule()->getMaxPathLen();
    params.maxSeqLen = maxSeqLen;
    params.vocabSize = mDecoderDomain.getVocabSizePadded();

    invokeExtractExplicitDraftTokens(params, mStream);

    invokeCopyProbs(params, mStream);

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void ExplicitDraftTokensLayer<T>::packAcceptedPaths(
    DynamicDecodeOutputParams const& outputs, ExplicitDraftTokensInputParams const& inputs)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    auto const batchSize = inputs.batch_slots->shape[0];

    auto paths = inputs.lastDraftIndices.template getPtr<SizeType32 const>();
    auto batchSlots = inputs.batch_slots ? inputs.batch_slots->template getPtr<SizeType32 const>() : nullptr;
    auto acceptedLengths = outputs.explicitDraftTokensOutputs->acceptedLengths.template getPtr<SizeType32 const>();
    auto acceptedLengthsCumSum
        = outputs.explicitDraftTokensOutputs->acceptedLengthsCumSum.template getPtr<SizeType32>();
    auto pathsOffsets = outputs.explicitDraftTokensOutputs->pathsOffsets.template getPtr<SizeType32>();
    auto bestPathIndices = inputs.bestPathIndices.template getPtr<SizeType32 const>();

    TLLM_CHECK_WITH_INFO(batchSlots != nullptr, "Batch slots must be provided for ExplicitDraftTokensLayer");
    TLLM_CHECK_WITH_INFO(acceptedLengths != nullptr, "Accepted lengths must be provided for ExplicitDraftTokensLayer");
    TLLM_CHECK_WITH_INFO(
        acceptedLengthsCumSum != nullptr, "acceptedLengthsCumSum must be provided for ExplicitDraftTokensLayer");
    TLLM_CHECK_WITH_INFO(pathsOffsets != nullptr, "pathsOffsets must be provided for ExplicitDraftTokensLayer");
    invokePackAcceptedPaths(acceptedLengthsCumSum, pathsOffsets, acceptedLengths, bestPathIndices, paths, batchSlots,
        batchSize, mDecoderDomain.getSpeculativeDecodingModule()->getMaxNumPaths(),
        mDecoderDomain.getSpeculativeDecodingModule()->getMaxPathLen(), true, mStream);

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template class ExplicitDraftTokensLayer<float>;
template class ExplicitDraftTokensLayer<half>;

} // namespace layers
} // namespace tensorrt_llm
