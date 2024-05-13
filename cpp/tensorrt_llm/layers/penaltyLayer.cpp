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

#include "tensorrt_llm/layers/penaltyLayer.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/memoryUtils.h"
#include "tensorrt_llm/kernels/penaltyKernels.h"
#include "tensorrt_llm/layers/defaultDecodingParams.h"

#include <algorithm>

using namespace tensorrt_llm::common;
using namespace tensorrt_llm::kernels;
using namespace tensorrt_llm::runtime;

namespace tensorrt_llm
{
namespace layers
{

template <typename T>
PenaltyLayer<T>::PenaltyLayer(DecodingMode const& mode, DecoderDomain const& decoderDomain, cudaStream_t stream,
    std::shared_ptr<IAllocator> allocator)
    : BaseLayer(decoderDomain, stream, std::move(allocator))
    , mDecodingMode(mode)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    initialize();

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
PenaltyLayer<T>::~PenaltyLayer()
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    freeBuffer();

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void PenaltyLayer<T>::initialize()
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    mLogitsPtrsHost = runtime::BufferManager::pinned(ITensor::makeShape({}), runtime::TRTDataType<T*>::value);

    allocateBuffer();

    mCyclicStep = 0;
    mRuntimeMaxSeqLen = 0;
    mConfiguredBeamWidth = -1;

    mTemperature.resize(mDecoderDomain.getMaxBatchSize());
    mRepetitionPenalty.resize(mDecoderDomain.getMaxBatchSize());
    mPresencePenalty.resize(mDecoderDomain.getMaxBatchSize());
    mFrequencyPenalty.resize(mDecoderDomain.getMaxBatchSize());
    mMinLength.resize(mDecoderDomain.getMaxBatchSize());

    if (!mDecodingMode.isNone())
    {
        mConfiguredBeamWidth = mDecoderDomain.getMaxBeamWidth();

        allocateWorkspace();
    }

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void PenaltyLayer<T>::allocateWorkspace()
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    auto const workspaceSize = sizeof(SizeType) * mDecoderDomain.getMaxBatchSize()
        * mDecoderDomain.getMaxTokensPerStep() * mConfiguredBeamWidth * mDecoderDomain.getVocabSize();
    mPenaltyWorkspaceDevice = mAllocator->reMalloc(mPenaltyWorkspaceDevice, workspaceSize, false);

    if (mDecodingMode.isBeamSearch())
    {
        mPenaltyWorkspacePrevDevice = mAllocator->reMalloc(mPenaltyWorkspacePrevDevice, workspaceSize, false);
    }

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void PenaltyLayer<T>::allocateBuffer()
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    mTemperatureDevice
        = mAllocator->reMalloc(mTemperatureDevice, sizeof(float) * mDecoderDomain.getMaxBatchSize(), false);
    mRepetitionPenaltyDevice
        = mAllocator->reMalloc(mRepetitionPenaltyDevice, sizeof(float) * mDecoderDomain.getMaxBatchSize(), false);
    mPresencePenaltyDevice
        = mAllocator->reMalloc(mPresencePenaltyDevice, sizeof(float) * mDecoderDomain.getMaxBatchSize(), false);
    mFrequencyPenaltyDevice
        = mAllocator->reMalloc(mFrequencyPenaltyDevice, sizeof(float) * mDecoderDomain.getMaxBatchSize(), false);
    mMinLengthDevice
        = mAllocator->reMalloc(mMinLengthDevice, sizeof(SizeType32) * mDecoderDomain.getMaxBatchSize(), false);

    mRuntimeLogitsDevice = mAllocator->reMalloc(mRuntimeLogitsDevice,
        sizeof(T) * mDecoderDomain.getMaxBatchSize() * mDecoderDomain.getMaxTokensPerStep()
            * mDecoderDomain.getMaxBeamWidth() * mDecoderDomain.getVocabSizePadded(),
        false);

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void PenaltyLayer<T>::freeBuffer()
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    if (mPenaltyWorkspaceDevice != nullptr)
    {
        mAllocator->free((void**) &mPenaltyWorkspaceDevice);
    }
    if (mPenaltyWorkspacePrevDevice != nullptr)
    {
        mAllocator->free((void**) &mPenaltyWorkspacePrevDevice);
    }
    mAllocator->free((void**) (&mTemperatureDevice));
    mAllocator->free((void**) (&mRepetitionPenaltyDevice));
    mAllocator->free((void**) (&mPresencePenaltyDevice));
    mAllocator->free((void**) (&mFrequencyPenaltyDevice));
    mAllocator->free((void**) (&mMinLengthDevice));

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void PenaltyLayer<T>::setup(SizeType batchSize, SizeType beamWidth, SizeType const* batchSlots,
    std::shared_ptr<BaseSetupParams> baseSetupParams)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    auto setupParams = std::dynamic_pointer_cast<DynamicDecodeSetupParams>(baseSetupParams);

    if (mConfiguredBeamWidth == -1)
    {
        // This code is left only for Python runtime
        // In C++ runtime given maxBeamWidth should always be equal to the runtime beamWidth
        TLLM_CHECK(mDecodingMode.isNone());
        mConfiguredBeamWidth = beamWidth;
        mDecodingMode = mConfiguredBeamWidth == 1 ? DecodingMode::TopKTopP() : DecodingMode::BeamSearch();
        allocateWorkspace();
    }

    std::vector<SizeType> batchSlotsVec(batchSize);
    std::iota(batchSlotsVec.begin(), batchSlotsVec.end(), 0);
    auto batchSlotsHost = batchSlots ? batchSlots : batchSlotsVec.data();

    // Setup penalties.
    FillBuffers const fillBuffers{batchSize, mDecoderDomain.getMaxBatchSize(), mStream};

    auto const& penaltyParams = setupParams->penaltyParams;

    bool const useTemperature = penaltyParams.temperature.has_value();
    bool const useRepetitionPenalty = penaltyParams.repetitionPenalty.has_value();
    bool const usePresencePenalty = penaltyParams.presencePenalty.has_value();
    bool const useFrequencyPenalty = penaltyParams.frequencyPenalty.has_value();
    bool const useMinLength = penaltyParams.minLength.has_value();
    if (useTemperature)
    {
        fillBuffers(penaltyParams.temperature, DefaultDecodingParams::getTemperature(), mTemperature,
            mTemperatureDevice, batchSlotsHost, getLimitsPenalty(DecodingPenaltyType::Temperature),
            "temperature penalty");
    }
    if (useRepetitionPenalty)
    {
        fillBuffers(penaltyParams.repetitionPenalty, DefaultDecodingParams::getRepetitionPenalty(), mRepetitionPenalty,
            mRepetitionPenaltyDevice, batchSlotsHost, getLimitsPenalty(DecodingPenaltyType::Repetition),
            "repetition penalty");
    }
    if (usePresencePenalty)
    {
        fillBuffers(penaltyParams.presencePenalty, DefaultDecodingParams::getPresencePenalty(), mPresencePenalty,
            mPresencePenaltyDevice, batchSlotsHost, getLimitsPenalty(DecodingPenaltyType::Presence),
            "presence penalty");
    }
    if (useFrequencyPenalty)
    {
        fillBuffers(penaltyParams.frequencyPenalty, DefaultDecodingParams::getFrequencyPenalty(), mFrequencyPenalty,
            mFrequencyPenaltyDevice, batchSlotsHost, getLimitsPenalty(DecodingPenaltyType::Frequency),
            "frequency penalty");
    }
    if (useMinLength)
    {
        fillBuffers(penaltyParams.minLength, DefaultDecodingParams::getMinLength(), mMinLength, mMinLengthDevice,
            batchSlotsHost, getLimitsPenalty(DecodingPenaltyType::MinLength), "min length");
    }

    // FIXME(nkorobov): once of the requests has some penalty, we will always have to compute it.
    // To avoid that need scan through all active requests for each iteration.
    mUseTemperature |= useTemperature;
    mUseRepetitionPenalty |= useRepetitionPenalty;
    mUsePresencePenalty |= usePresencePenalty;
    mUseFrequencyPenalty |= useFrequencyPenalty;
    mUseMinLength |= useMinLength;

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void PenaltyLayer<T>::forward(
    std::shared_ptr<BaseOutputParams> baseOutputs, std::shared_ptr<BaseInputParams> baseInputs)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    auto outputs = std::dynamic_pointer_cast<DynamicDecodeOutputParams>(baseOutputs);
    auto params = std::dynamic_pointer_cast<DynamicDecodeInputParams>(baseInputs);

    SizeType batchSize{0};
    SizeType beamWidth{0};
    SizeType vocabSize{0};
    if (params->logits)
    {
        auto const& logitsShape = params->logits->shape;
        TLLM_CHECK(logitsShape.size() == 3 || logitsShape.size() == 4);
        batchSize = logitsShape[0];
        auto const idxOffset = logitsShape.size() - 3;
        beamWidth = logitsShape[idxOffset + 1];
        vocabSize = logitsShape[idxOffset + 2];
    }
    else
    {
        TLLM_CHECK(params->logits_vec->size());
        auto const& logitsShape = params->logits_vec.value()[0].shape;
        TLLM_CHECK(logitsShape.size() == 3 || logitsShape.size() == 4);
        auto const idxOffset = logitsShape.size() - 3;
        batchSize = params->logits_vec->size();
        beamWidth = logitsShape[idxOffset + 1];
        vocabSize = logitsShape[idxOffset + 2];
    }
    auto const maxSeqLen = outputs->output_ids.shape[outputs->output_ids.shape.size() - 1];
    auto batchSlots = params->batch_slots ? params->batch_slots->template getPtr<SizeType32 const>() : nullptr;
    std::vector<SizeType> batchSlotsVec(batchSize);
    std::iota(batchSlotsVec.begin(), batchSlotsVec.end(), 0);
    auto batchSlotsHost
        = params->batch_slots ? params->batch_slots->template getPtr<SizeType32 const>() : batchSlotsVec.data();

    if (!mLogitsPtrsHost->data())
    {
        mLogitsPtrsHost = runtime::BufferManager::pinnedPool(
            ITensor::makeShape(
                {static_cast<int32_t>(maxSeqLen), static_cast<int32_t>(mDecoderDomain.getMaxBatchSize())}),
            runtime::TRTDataType<T*>::value);
        mRuntimeMaxSeqLen = maxSeqLen;
    }

    mCyclicStep = mCyclicStep % mRuntimeMaxSeqLen;

    auto logitsPtrsHost = ITensor::slice(mLogitsPtrsHost, mCyclicStep, 1);
    auto logitsPtrsHostData = reinterpret_cast<T const**>(runtime::bufferCast<int64_t>(*logitsPtrsHost));
    for (SizeType bi = 0; bi < batchSize; bi++)
    {
        if (params->logits_vec)
        {
            TLLM_CHECK_WITH_INFO(params->logits_vec->size() == batchSize,
                "Logits vector size (%lu) is not equal to the batchSize (%d)", params->logits_vec->size(), batchSize);
            logitsPtrsHostData[bi] = params->logits_vec.value()[bi].template getPtr<T>();
        }
        else
        {
            logitsPtrsHostData[bi]
                = params->logits->template getPtrWithOffset<T>(bi * beamWidth * mDecoderDomain.getVocabSizePadded());
        }
    }

    SizeType32 const* inputLengths = nullptr;
    if (params->input_lengths)
    {
        auto& input_lengths = params->input_lengths.value();
        inputLengths = input_lengths.template getPtr<SizeType32 const>();
    }
    auto* embeddingBias = params->embedding_bias ? params->embedding_bias->template getPtr<T const>() : nullptr;
#define GET_PENALTIES(capital_name, type)                                                                              \
    (mUse##capital_name                                                                                                \
        && !allOfBatchSlots(                                                                                           \
            batchSlotsHost, m##capital_name.data(), batchSize, DefaultDecodingParams::get##capital_name()))            \
        ? m##capital_name##Device                                                                                      \
        : nullptr;

    auto* temperatures = GET_PENALTIES(Temperature, float);
    auto* repetitionPenalties = GET_PENALTIES(RepetitionPenalty, float);
    auto* presencePenalties = GET_PENALTIES(PresencePenalty, float);
    auto* frequencyPenalties = GET_PENALTIES(FrequencyPenalty, float);
    auto* minLengths = GET_PENALTIES(MinLength, SizeType32);

#undef GET_PENALTIES

    auto const tokensPerStep = params->medusaInputs
        ? params->medusaInputs->medusaCurTokensPerStep.template getPtr<SizeType32 const>()
        : nullptr;
    InvokeBatchApplyPenaltyParams<T> penaltyParams{reinterpret_cast<T const* const*>(logitsPtrsHostData),
        mRuntimeLogitsDevice, embeddingBias, mPenaltyWorkspaceDevice, mPenaltyWorkspacePrevDevice, temperatures,
        repetitionPenalties, presencePenalties, frequencyPenalties,
        (mUseRepetitionPenalty || mUsePresencePenalty || mUseFrequencyPenalty), batchSize,
        static_cast<SizeType>(beamWidth), static_cast<SizeType>(maxSeqLen), mDecoderDomain.getVocabSize(),
        mDecoderDomain.getVocabSizePadded(), outputs->output_ids_ptr.template getPtr<TokenIdType const*>(),
        outputs->parent_ids_ptr.template getPtr<SizeType32 const*>(), inputLengths,
        outputs->sequence_length->template getPtr<SizeType32 const>(), minLengths,
        params->end_ids.template getPtr<TokenIdType const>(), batchSlots, mDecoderDomain.getMaxTokensPerStep(),
        tokensPerStep, mStream};
    invokeBatchApplyPenalty(penaltyParams);
    sync_check_cuda_error();

    mCyclicStep += 1;

    params->logits = Tensor(MEMORY_GPU, std::is_same_v<T, float> ? DataType::TYPE_FP32 : DataType::TYPE_FP16,
        {static_cast<size_t>(batchSize), static_cast<size_t>(mDecoderDomain.getMaxTokensPerStep()),
            static_cast<size_t>(beamWidth), static_cast<size_t>(mDecoderDomain.getVocabSizePadded())},
        mRuntimeLogitsDevice);

    if (mDecodingMode.isBeamSearch())
    {
        std::swap(mPenaltyWorkspaceDevice, mPenaltyWorkspacePrevDevice);
    }

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template class PenaltyLayer<float>;
template class PenaltyLayer<half>;

} // namespace layers
} // namespace tensorrt_llm
