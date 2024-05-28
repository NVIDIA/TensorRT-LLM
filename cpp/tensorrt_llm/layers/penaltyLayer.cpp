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
PenaltyLayer<T>::PenaltyLayer(executor::DecodingMode const& mode, DecoderDomain const& decoderDomain,
    cudaStream_t stream, std::shared_ptr<IAllocator> allocator)
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

    mTemperature.resize(mDecoderDomain.getBatchSize());
    mRepetitionPenalty.resize(mDecoderDomain.getBatchSize());
    mPresencePenalty.resize(mDecoderDomain.getBatchSize());
    mFrequencyPenalty.resize(mDecoderDomain.getBatchSize());
    mMinLength.resize(mDecoderDomain.getBatchSize());

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
        auto const workspaceSize = sizeof(SizeType32) * mDecoderDomain.getBatchSize()
            * mDecoderDomain.getMaxTokensPerStep() * mConfiguredBeamWidth * mDecoderDomain.getVocabSize();
        mPenaltyWorkspaceDevice = mAllocator->reMalloc(mPenaltyWorkspaceDevice, workspaceSize, false);

        if (mDecodingMode.isBeamSearch())
        {
            mPenaltyWorkspacePrevDevice = mAllocator->reMalloc(mPenaltyWorkspacePrevDevice, workspaceSize, false);
        }
    }

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void PenaltyLayer<T>::allocateBuffer()
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    if (mDecodingMode.isUseTemperature())
    {
        mTemperatureDevice
            = mAllocator->reMalloc(mTemperatureDevice, sizeof(float) * mDecoderDomain.getBatchSize(), false);
    }
    if (mDecodingMode.isUseRepetitionPenalty())
    {
        mRepetitionPenaltyDevice
            = mAllocator->reMalloc(mRepetitionPenaltyDevice, sizeof(float) * mDecoderDomain.getBatchSize(), false);
    }
    if (mDecodingMode.isUsePresencePenalty())
    {
        mPresencePenaltyDevice
            = mAllocator->reMalloc(mPresencePenaltyDevice, sizeof(float) * mDecoderDomain.getBatchSize(), false);
    }
    if (mDecodingMode.isUseFrequencyPenalty())
    {
        mFrequencyPenaltyDevice
            = mAllocator->reMalloc(mFrequencyPenaltyDevice, sizeof(float) * mDecoderDomain.getBatchSize(), false);
    }
    if (mDecodingMode.isUseMinLength())
    {
        mMinLengthDevice
            = mAllocator->reMalloc(mMinLengthDevice, sizeof(SizeType32) * mDecoderDomain.getBatchSize(), false);
    }

    mRuntimeLogitsDevice = mAllocator->reMalloc(mRuntimeLogitsDevice,
        sizeof(T) * mDecoderDomain.getBatchSize() * mDecoderDomain.getMaxTokensPerStep() * mDecoderDomain.getBeamWidth()
            * mDecoderDomain.getVocabSizePadded(),
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
    if (mDecodingMode.isUseTemperature())
    {
        mAllocator->free((void**) (&mTemperatureDevice));
    }
    if (mDecodingMode.isUseRepetitionPenalty())
    {
        mAllocator->free((void**) (&mRepetitionPenaltyDevice));
    }
    if (mDecodingMode.isUsePresencePenalty())
    {
        mAllocator->free((void**) (&mPresencePenaltyDevice));
    }
    if (mDecodingMode.isUseFrequencyPenalty())
    {
        mAllocator->free((void**) (&mFrequencyPenaltyDevice));
    }
    if (mDecodingMode.isUseMinLength())
    {
        mAllocator->free((void**) (&mMinLengthDevice));
    }

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void PenaltyLayer<T>::setup(SizeType32 batchSize, SizeType32 beamWidth, SizeType32 const* batchSlots,
    std::shared_ptr<BaseSetupParams> baseSetupParams)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

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

    std::vector<SizeType32> batchSlotsVec(batchSize);
    std::iota(batchSlotsVec.begin(), batchSlotsVec.end(), 0);
    auto batchSlotsHost = batchSlots ? batchSlots : batchSlotsVec.data();

    // Setup penalties.
    FillBuffers const fillBuffers{batchSize, mDecoderDomain.getBatchSize(), mStream};

    auto const& penaltyParams = setupParams->penaltyParams;

    bool const useTemperature = mDecodingMode.isUseTemperature() && penaltyParams.temperature.has_value();
    bool const useRepetitionPenalty
        = mDecodingMode.isUseRepetitionPenalty() && penaltyParams.repetitionPenalty.has_value();
    bool const usePresencePenalty = mDecodingMode.isUsePresencePenalty() && penaltyParams.presencePenalty.has_value();
    bool const useFrequencyPenalty
        = mDecodingMode.isUseFrequencyPenalty() && penaltyParams.frequencyPenalty.has_value();
    bool const useMinLength = mDecodingMode.isUseMinLength() && penaltyParams.minLength.has_value();
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
void PenaltyLayer<T>::forwardAsync(
    std::shared_ptr<BaseOutputParams> baseOutputs, std::shared_ptr<BaseInputParams> baseInputs)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    auto outputs = std::dynamic_pointer_cast<DynamicDecodeOutputParams>(baseOutputs);
    auto params = std::dynamic_pointer_cast<DynamicDecodeInputParams>(baseInputs);

    auto const localDecoderDomain = getLocalDecoderDomain(params);
    auto const maxSeqLen = outputs->output_ids.shape[outputs->output_ids.shape.size() - 1];
    auto batchSlots = params->batch_slots ? params->batch_slots->template getPtr<SizeType32 const>() : nullptr;

    std::vector<SizeType32> batchSlotsVec(localDecoderDomain.getBatchSize());
    std::iota(batchSlotsVec.begin(), batchSlotsVec.end(), 0);
    auto batchSlotsHost
        = params->batch_slots ? params->batch_slots->template getPtr<SizeType32 const>() : batchSlotsVec.data();

    if (!mLogitsPtrsHost->data())
    {
        mLogitsPtrsHost = runtime::BufferManager::pinnedPool(
            ITensor::makeShape({static_cast<int32_t>(maxSeqLen), static_cast<int32_t>(mDecoderDomain.getBatchSize())}),
            runtime::TRTDataType<T*>::value);
        mRuntimeMaxSeqLen = maxSeqLen;
    }

    mCyclicStep = mCyclicStep % mRuntimeMaxSeqLen;

    auto logitsPtrsHost = ITensor::slice(mLogitsPtrsHost, mCyclicStep, 1);
    auto logitsPtrsHostData = reinterpret_cast<T const**>(runtime::bufferCast<int64_t>(*logitsPtrsHost));
    for (SizeType32 bi = 0; bi < localDecoderDomain.getBatchSize(); bi++)
    {
        if (params->logits_vec)
        {
            TLLM_CHECK_WITH_INFO(params->logits_vec->size() == localDecoderDomain.getBatchSize(),
                "Logits vector size (%lu) is not equal to the batchSize (%d)", params->logits_vec->size(),
                localDecoderDomain.getBatchSize());
            logitsPtrsHostData[bi] = params->logits_vec.value()[bi].template getPtr<T>();
        }
        else
        {
            logitsPtrsHostData[bi] = params->logits->template getPtrWithOffset<T>(
                bi * localDecoderDomain.getBeamWidth() * mDecoderDomain.getVocabSizePadded());
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
        && !allOfBatchSlots(batchSlotsHost, m##capital_name.data(), localDecoderDomain.getBatchSize(),                 \
            DefaultDecodingParams::get##capital_name()))                                                               \
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

    InvokeBatchApplyPenaltyParams<T> penaltyParams;
    penaltyParams.inputLogits = reinterpret_cast<T const* const*>(logitsPtrsHostData);
    penaltyParams.outputLogits = mRuntimeLogitsDevice;
    penaltyParams.biases = embeddingBias;
    penaltyParams.penaltyWorkspace = mPenaltyWorkspaceDevice;
    penaltyParams.penaltyWorkspacePrev = mPenaltyWorkspacePrevDevice;
    penaltyParams.temperatures = temperatures;
    penaltyParams.repetitionPenalties = repetitionPenalties;
    penaltyParams.presencePenalties = presencePenalties;
    penaltyParams.frequencyPenalties = frequencyPenalties;
    penaltyParams.batchSize = localDecoderDomain.getBatchSize();
    penaltyParams.beamWidth = localDecoderDomain.getBeamWidth();
    penaltyParams.maxSeqLen = maxSeqLen;
    penaltyParams.vocabSize = mDecoderDomain.getVocabSize();
    penaltyParams.vocabSizePadded = mDecoderDomain.getVocabSizePadded();
    penaltyParams.outputIdsPtr = outputs->output_ids_ptr.template getPtr<TokenIdType const*>();
    penaltyParams.parentIdsPtr = outputs->parent_ids_ptr.template getPtr<SizeType32 const*>();
    penaltyParams.inputLengths = inputLengths;
    penaltyParams.sequenceLengths = outputs->sequence_length->template getPtr<SizeType32 const>();
    penaltyParams.minLengths = minLengths;
    penaltyParams.endIds = params->end_ids.template getPtr<TokenIdType const>();
    penaltyParams.batchSlots = batchSlots;
    penaltyParams.maxTokensPerStep = mDecoderDomain.getMaxTokensPerStep();
    penaltyParams.tokensPerStep = tokensPerStep;
    penaltyParams.stream = mStream;
    invokeBatchApplyPenalty(penaltyParams);
    sync_check_cuda_error();

    mCyclicStep += 1;

    params->logits = Tensor(MEMORY_GPU, std::is_same_v<T, float> ? DataType::TYPE_FP32 : DataType::TYPE_FP16,
        {static_cast<size_t>(localDecoderDomain.getBatchSize()),
            static_cast<size_t>(mDecoderDomain.getMaxTokensPerStep()),
            static_cast<size_t>(localDecoderDomain.getBeamWidth()),
            static_cast<size_t>(mDecoderDomain.getVocabSizePadded())},
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
