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

#include "tensorrt_llm/layers/dynamicDecodeLayer.h"
#include "tensorrt_llm/common/memoryUtils.h"
#include "tensorrt_llm/kernels/decodingKernels.h"
#include "tensorrt_llm/layers/beamSearchLayer.h"
#include "tensorrt_llm/layers/defaultDecodingParams.h"
#include "tensorrt_llm/layers/layerUtils.h"
#include "tensorrt_llm/layers/layersFactory.h"
#include "tensorrt_llm/runtime/bufferManager.h"
#include "tensorrt_llm/runtime/cudaStream.h"

using namespace tensorrt_llm::common;
using namespace tensorrt_llm::kernels;
using namespace tensorrt_llm::runtime;

namespace tensorrt_llm
{
namespace layers
{

template <typename T>
DynamicDecodeLayer<T>::DynamicDecodeLayer(executor::DecodingMode const& mode, DecoderDomain const& decoderDomain,
    cudaStream_t stream, std::shared_ptr<IAllocator> allocator)
    : BaseLayer(decoderDomain, stream, std::move(allocator))
    , mDecodingMode(mode)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    initialize();

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
DynamicDecodeLayer<T>::~DynamicDecodeLayer()
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    freeBuffer();

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void DynamicDecodeLayer<T>::initialize()
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    mIdsPtrHost = runtime::BufferManager::pinned(ITensor::makeShape({}), runtime::TRTDataType<TokenIdType*>::value);

    allocateBuffer();

    mCyclicStep = 0;
    mRuntimeMaxSeqLen = 0;
    mConfiguredBeamWidth = -1;

    if (!mDecodingMode.isAuto())
    {
        mConfiguredBeamWidth = mDecoderDomain.getBeamWidth();
        initializeLayers();
    }

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void DynamicDecodeLayer<T>::allocateBuffer()
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    mZeroParentIdsDevice
        = mAllocator->reMalloc(mZeroParentIdsDevice, sizeof(TokenIdType*) * 2 * mDecoderDomain.getBatchSize(), false);

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void DynamicDecodeLayer<T>::freeBuffer()
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    mAllocator->free((void**) &mZeroParentIdsDevice);

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void DynamicDecodeLayer<T>::initializeLayers()
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    mLayers = createLayers<T>(mDecodingMode, mDecoderDomain, mStream, mAllocator);

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void DynamicDecodeLayer<T>::setup(SizeType32 batchSize, SizeType32 beamWidth, SizeType32 const* batchSlots,
    std::shared_ptr<BaseSetupParams> baseSetupParams)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    auto setupParams = std::dynamic_pointer_cast<DynamicDecodeSetupParams>(baseSetupParams);

    if (setupParams->samplingParams.outputLogProbs)
    {
        // FIXME(nkorobov): monotonically growing
        mOutputLogProbs = std::any_of(setupParams->samplingParams.outputLogProbs->begin(),
            setupParams->samplingParams.outputLogProbs->end(),
            [this](bool outputLogProbs) { return this->mOutputLogProbs | outputLogProbs; });
    }

    if (mConfiguredBeamWidth == -1)
    {
        // This code is left only for Python runtime
        // In C++ runtime given maxBeamWidth should always be equal to the runtime beamWidth
        TLLM_CHECK(mDecodingMode.isAuto());
        mConfiguredBeamWidth = beamWidth;
        mDecodingMode
            = mConfiguredBeamWidth == 1 ? executor::DecodingMode::TopKTopP() : executor::DecodingMode::BeamSearch();
        initializeLayers();
    }

    TLLM_CHECK_WITH_INFO((mConfiguredBeamWidth == 1 && beamWidth == 1)
            || (mConfiguredBeamWidth > 1 && beamWidth > 1 && beamWidth <= mConfiguredBeamWidth),
        "Decoder is configured with beam width %d, but %d was given", mConfiguredBeamWidth, beamWidth);
    TLLM_CHECK_WITH_INFO(mConfiguredBeamWidth <= mDecoderDomain.getBeamWidth(),
        "Decoder is created with max beam width %d, but %d was given", mDecoderDomain.getBeamWidth(),
        mConfiguredBeamWidth);

    for (auto& layer : mLayers)
    {
        layer->setup(batchSize, beamWidth, batchSlots, baseSetupParams);
    }

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void DynamicDecodeLayer<T>::forwardAsync(
    std::shared_ptr<BaseOutputParams> baseOutputs, std::shared_ptr<BaseInputParams> baseInputs)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    auto params = std::dynamic_pointer_cast<DynamicDecodeInputParams>(baseInputs);
    auto outputs = std::dynamic_pointer_cast<DynamicDecodeOutputParams>(baseOutputs);

    TLLM_CHECK_WITH_INFO(params->logits || params->logits_vec, "Either logits or logits_vec have to be specified.");
    TLLM_CHECK_WITH_INFO(
        outputs->sequence_length.has_value(), "sequence_length tensor is mandatory in DynamicDecoderLayer.");

    auto const localDecoderDomain = getLocalDecoderDomain(params);
    auto const maxSeqLen = outputs->output_ids.shape[outputs->output_ids.shape.size() - 1];

    TLLM_CHECK_WITH_INFO((mConfiguredBeamWidth == 1 && localDecoderDomain.getBeamWidth() == 1)
            || (mConfiguredBeamWidth > 1 && localDecoderDomain.getBeamWidth() > 1
                && localDecoderDomain.getBeamWidth() <= mConfiguredBeamWidth),
        "Decoder is configured with beam width %d, but %d was given", mConfiguredBeamWidth,
        localDecoderDomain.getBeamWidth());

    if (!mIdsPtrHost->data())
    {
        mIdsPtrHost = runtime::BufferManager::pinnedPool(ITensor::makeShape({static_cast<int32_t>(maxSeqLen),
                                                             static_cast<int32_t>(2 * mDecoderDomain.getBatchSize())}),
            runtime::TRTDataType<int32_t*>::value);
        mRuntimeMaxSeqLen = maxSeqLen;
    }

    std::vector<SizeType32> batchSlotsVec(localDecoderDomain.getBatchSize());
    std::iota(batchSlotsVec.begin(), batchSlotsVec.end(), 0);
    auto batchSlotsHost
        = params->batch_slots ? params->batch_slots->template getPtr<SizeType32 const>() : batchSlotsVec.data();
    auto batchSlots = params->batch_slots ? params->batch_slots->template getPtr<SizeType32 const>() : nullptr;

    mCyclicStep = mCyclicStep % mRuntimeMaxSeqLen;
    prepareIdsPtrs(
        outputs, batchSlotsHost, localDecoderDomain.getBatchSize(), localDecoderDomain.getBeamWidth(), maxSeqLen);

    for (auto& layer : mLayers)
    {
        layer->forwardAsync(baseOutputs, baseInputs);
    }

    // Copy nextIds and transpose logits when needed
    prepareOutputData(outputs, params, mIdsPtrHost, batchSlots, localDecoderDomain.getBatchSize(),
        mDecoderDomain.getBatchSize(), localDecoderDomain.getBeamWidth(), maxSeqLen,
        mDecoderDomain.getMaxTokensPerStep(), mCyclicStep, mOutputLogProbs, mStream);

    mCyclicStep += 1;

    sync_check_cuda_error();
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void DynamicDecodeLayer<T>::forwardSync(
    std::shared_ptr<BaseOutputParams> baseOutputs, std::shared_ptr<BaseInputParams> baseInputs)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    for (auto& layer : mLayers)
    {
        layer->forwardSync(baseOutputs, baseInputs);
    }
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void DynamicDecodeLayer<T>::prepareIdsPtrs(std::shared_ptr<DynamicDecodeOutputParams> const& outputs,
    SizeType32 const* batchSlots, SizeType32 batchSize, SizeType32 beamWidth, SizeType32 maxSeqLen)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    auto idsPtrHostSlice = ITensor::slice(mIdsPtrHost, mCyclicStep, 1);
    auto idsPtrHost = reinterpret_cast<TokenIdType**>(runtime::bufferCast<int64_t>(*idsPtrHostSlice));
    for (SizeType32 bi = 0; bi < batchSize; bi++)
    {
        auto const batchSlot = batchSlots[bi];
        idsPtrHost[batchSlot]
            = outputs->output_ids.template getPtrWithOffset<TokenIdType>(batchSlot * beamWidth * maxSeqLen);
    }
    for (SizeType32 bi = 0; bi < batchSize; bi++)
    {
        auto const batchSlot = batchSlots[bi];
        if (beamWidth > 1)
        {
            idsPtrHost[mDecoderDomain.getBatchSize() + batchSlot]
                = outputs->parent_ids.value().template getPtrWithOffset<SizeType32>(bi * beamWidth * maxSeqLen);
        }
        else
        {
            idsPtrHost[mDecoderDomain.getBatchSize() + batchSlot] = mZeroParentIdsDevice + bi * beamWidth * maxSeqLen;
        }
    }

    outputs->output_ids_ptr = Tensor(MEMORY_GPU, DataType::TYPE_INT32_PTR,
        {static_cast<size_t>(mDecoderDomain.getBatchSize()), static_cast<size_t>(beamWidth),
            static_cast<size_t>(maxSeqLen)},
        idsPtrHost);
    outputs->parent_ids_ptr = Tensor(MEMORY_GPU, DataType::TYPE_INT32_PTR,
        {static_cast<size_t>(mDecoderDomain.getBatchSize()), static_cast<size_t>(beamWidth),
            static_cast<size_t>(maxSeqLen)},
        idsPtrHost + mDecoderDomain.getBatchSize());
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void DynamicDecodeLayer<T>::prepareOutputData(std::shared_ptr<DynamicDecodeOutputParams> const& outputs,
    std::shared_ptr<DynamicDecodeInputParams> const& params, runtime::ITensor::SharedPtr const& idsPtrsHost,
    SizeType32 const* batchSlots, SizeType32 batchSize, SizeType32 maxBatchSize, SizeType32 beamWidth,
    SizeType32 maxSeqLen, SizeType32 maxTokensPerStep, SizeType32 cyclicStep, bool outputLogProbs, cudaStream_t stream)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    auto idsPtrHostSlice = ITensor::slice(idsPtrsHost, cyclicStep, 1);
    auto idsPtrHost = reinterpret_cast<TokenIdType**>(runtime::bufferCast<int64_t>(*idsPtrHostSlice));
    auto const numNewTokens = outputs->speculativeDecodingOutputs
        ? outputs->speculativeDecodingOutputs->acceptedLengths.template getPtr<SizeType32 const>()
        : nullptr;
    invokeCopyNextStepIds(outputs->newTokens.template getPtr<TokenIdType>(), idsPtrHost,
        outputs->sequence_length->template getPtr<SizeType32>(), numNewTokens, batchSlots, batchSize, maxBatchSize,
        beamWidth, maxSeqLen, maxTokensPerStep, stream);

    // Transpose output log probs from [maxSeqLen, batchSize, beamWidth] to [batchSize, beamWidth, maxSeqLen]
    if (outputLogProbs && outputs->output_log_probs_tiled)
    {
        auto logProbsMaxSeqLen = outputs->output_log_probs_tiled.value().shape[0];

        invokeTransposeLogProbs(outputs->output_log_probs.value().template getPtr<float>(),
            outputs->output_log_probs_tiled.value().template getPtr<float>(),
            outputs->sequence_length->template getPtr<SizeType32>(), batchSlots, batchSize, maxBatchSize, beamWidth,
            logProbsMaxSeqLen, stream);
    }
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template class DynamicDecodeLayer<float>;
template class DynamicDecodeLayer<half>;

} // namespace layers
} // namespace tensorrt_llm
