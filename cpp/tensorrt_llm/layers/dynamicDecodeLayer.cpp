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
#include "tensorrt_llm/kernels/banBadWords.h"
#include "tensorrt_llm/kernels/banRepeatNgram.h"
#include "tensorrt_llm/kernels/decodingKernels.h"
#include "tensorrt_llm/kernels/penaltyKernels.h"
#include "tensorrt_llm/kernels/stopCriteriaKernels.h"
#include "tensorrt_llm/layers/baseBeamSearchLayer.h"
#include "tensorrt_llm/layers/fillBuffers.h"
#include "tensorrt_llm/layers/onlineBeamSearchLayer.h"
#include "tensorrt_llm/runtime/bufferManager.h"
#include "tensorrt_llm/runtime/cudaStream.h"

using namespace tensorrt_llm::common;
using namespace tensorrt_llm::kernels;
using namespace tensorrt_llm::runtime;

namespace tensorrt_llm
{
namespace layers
{

namespace
{
template <typename T>
bool allSame(std::optional<std::vector<T>> const& vOpt)
{
    if (!vOpt)
    {
        return true;
    }

    auto const& v = *vOpt;

    if (v.size() <= 1)
    {
        return true;
    }
    auto first = v[0];
    for (std::size_t i = 1; i < v.size(); ++i)
    {
        if (v[i] != first)
        {
            return false;
        }
    }
    return true;
}

bool hasDiffRuntimeArgs(DecodingSetupParams const& params)
{
    return !allSame(params.frequency_penalty) || !allSame(params.presence_penalty)
        || !allSame(params.repetition_penalty) || !allSame(params.temperature) || !allSame(params.min_length);
}
} // namespace

template <typename T>
DynamicDecodeLayer<T>::DynamicDecodeLayer(DecodingMode const& mode, size_t maxBatchSize, size_t maxBeamWidth,
    size_t vocabSize, size_t vocabSizePadded, cudaStream_t stream, std::shared_ptr<IAllocator> allocator,
    cudaDeviceProp* cudaDeviceProp)
    : BaseLayer(stream, std::move(allocator))
    , mDecodingMode(mode)
    , mMaxBatchSize(maxBatchSize)
    , mMaxBeamWidth(maxBeamWidth)
    , mVocabSize(vocabSize)
    , mVocabSizePadded(vocabSizePadded)
    , mCudaDeviceProp(cudaDeviceProp)
{
    TLLM_LOG_TRACE(__PRETTY_FUNCTION__);
    initialize();
}

template <typename T>
DynamicDecodeLayer<T>::DynamicDecodeLayer(DynamicDecodeLayer const& dynamicDecodeLayer)
    : BaseLayer(dynamicDecodeLayer)
    , mDecodingMode(dynamicDecodeLayer.mDecodingMode)
    , mMaxBatchSize(dynamicDecodeLayer.mMaxBatchSize)
    , mMaxBeamWidth(dynamicDecodeLayer.mMaxBeamWidth)
    , mVocabSize(dynamicDecodeLayer.mVocabSize)
    , mVocabSizePadded(dynamicDecodeLayer.mVocabSizePadded)
    , mCudaDeviceProp(dynamicDecodeLayer.mCudaDeviceProp)
{
    TLLM_LOG_TRACE(__PRETTY_FUNCTION__);
    initialize();
}

template <typename T>
DynamicDecodeLayer<T>::~DynamicDecodeLayer()
{
    TLLM_LOG_TRACE(__PRETTY_FUNCTION__);
    freeBuffer();
}

template <typename T>
void DynamicDecodeLayer<T>::initialize()
{
    TLLM_LOG_TRACE(__PRETTY_FUNCTION__);

    mIdsPtrHost = runtime::BufferManager::pinned(ITensor::makeShape({}), runtime::TRTDataType<int*>::value);
    mLogitsPtrsHost = runtime::BufferManager::pinned(ITensor::makeShape({}), runtime::TRTDataType<T*>::value);

    allocateBuffer();

    mCyclicStep = 0;
    mRuntimeMaxSeqLen = 0;
    mConfiguredBeamWidth = -1;

    mTemperature.resize(mMaxBatchSize);
    mRepetitionPenalty.resize(mMaxBatchSize);
    mPresencePenalty.resize(mMaxBatchSize);
    mFrequencyPenalty.resize(mMaxBatchSize);
    mMinLength.resize(mMaxBatchSize);

    if (!mDecodingMode.isNone())
    {
        mConfiguredBeamWidth = mMaxBeamWidth;
        initializeLayers();
    }
}

template <typename T>
void DynamicDecodeLayer<T>::allocateBuffer()
{
    TLLM_LOG_TRACE(__PRETTY_FUNCTION__);
    mZeroParentIdsDevice = mAllocator->reMalloc(mZeroParentIdsDevice, sizeof(int*) * 2 * mMaxBatchSize, false);
    mTemperatureDevice = mAllocator->reMalloc(mTemperatureDevice, sizeof(float) * mMaxBatchSize, false);
    mRepetitionPenaltyDevice = mAllocator->reMalloc(mRepetitionPenaltyDevice, sizeof(float) * mMaxBatchSize, false);
    mPresencePenaltyDevice = mAllocator->reMalloc(mPresencePenaltyDevice, sizeof(float) * mMaxBatchSize, false);
    mFrequencyPenaltyDevice = mAllocator->reMalloc(mFrequencyPenaltyDevice, sizeof(float) * mMaxBatchSize, false);
    mMinLengthDevice = mAllocator->reMalloc(mMinLengthDevice, sizeof(int32_t) * mMaxBatchSize, false);
    mRuntimeLogitsDevice = mAllocator->reMalloc(
        mRuntimeLogitsDevice, sizeof(T) * mMaxBatchSize * mMaxBeamWidth * mVocabSizePadded, false);
}

template <typename T>
void DynamicDecodeLayer<T>::freeBuffer()
{
    TLLM_LOG_TRACE(__PRETTY_FUNCTION__);
    mAllocator->free((void**) &mZeroParentIdsDevice);
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
    mAllocator->free((void**) (&mRuntimeLogitsDevice));
}

template <typename T>
void DynamicDecodeLayer<T>::initializeLayers()
{
    const size_t workspaceSize = sizeof(int) * mMaxBatchSize * mConfiguredBeamWidth * mVocabSize;
    mPenaltyWorkspaceDevice = mAllocator->reMalloc(mPenaltyWorkspaceDevice, workspaceSize, false);

    if (mDecodingMode.isTopKorTopP())
    {
        mSamplingLayer = std::make_unique<SamplingLayer<T>>(
            mDecodingMode, mMaxBatchSize, mVocabSize, mVocabSizePadded, mStream, mAllocator, mCudaDeviceProp);
    }
    else if (mDecodingMode.isBeamSearch())
    {
        mOnlineBeamSearchDecode
            = std::make_unique<OnlineBeamSearchLayer<T>>(mVocabSize, mVocabSizePadded, mStream, mAllocator);
        mPenaltyWorkspacePrevDevice = mAllocator->reMalloc(mPenaltyWorkspacePrevDevice, workspaceSize, false);
    }
    else
    {
        TLLM_CHECK_WITH_INFO(false, "Decoding mode is none of the supported {TopK, TopP, TopKTopP, BeamSearch}");
    }
}

template <typename T>
void DynamicDecodeLayer<T>::setup(
    size_t batchSize, size_t beamWidth, int32_t const* batchSlots, SetupParams const& setupParams)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    if (mConfiguredBeamWidth == -1)
    {
        // This code is left only for Python runtime
        // In C++ runtime given maxBeamWidth should always be equal to the runtime beamWidth
        TLLM_CHECK(mDecodingMode.isNone());
        mConfiguredBeamWidth = beamWidth;
        mDecodingMode = mConfiguredBeamWidth == 1 ? DecodingMode::TopKTopP() : DecodingMode::BeamSearch();
        initializeLayers();
    }

    TLLM_CHECK_WITH_INFO((mConfiguredBeamWidth == 1 && beamWidth == 1)
            || (mConfiguredBeamWidth > 1 && beamWidth > 1 && beamWidth <= mConfiguredBeamWidth),
        "Decoder is configured with beam width %d, but %lu was given", mConfiguredBeamWidth, beamWidth);
    TLLM_CHECK_WITH_INFO(mConfiguredBeamWidth <= mMaxBeamWidth,
        "Decoder is created with max beam width %lu, but %d was given", mMaxBeamWidth, mConfiguredBeamWidth);

    setupLayers(batchSize, beamWidth, batchSlots, setupParams);

    setupPenalties(batchSize, batchSlots, setupParams);

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void DynamicDecodeLayer<T>::setupLayers(
    size_t batchSize, size_t beamWidth, int32_t const* batchSlots, SetupParams const& setupParams)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    if (beamWidth == 1)
    { // sampling layers
        TLLM_CHECK_WITH_INFO(
            mDecodingMode.isTopKorTopP(), "beamWidth == 1 is given, but decoder is not configured as TopK or TopP");
        typename TopPSamplingLayer<T>::SetupParams samplingParams;

        samplingParams.runtime_top_k = setupParams.runtime_top_k;
        samplingParams.runtime_top_p = setupParams.runtime_top_p;
        samplingParams.randomSeed = setupParams.randomSeed;

        samplingParams.top_p_decay = setupParams.top_p_decay;
        samplingParams.top_p_min = setupParams.top_p_min;
        samplingParams.top_p_reset_ids = setupParams.top_p_reset_ids;
        samplingParams.normalize_log_probs = setupParams.normalize_log_probs;

        mSamplingLayer->setup(batchSize, batchSlots, samplingParams);
    }
    else
    { // beam search layer
        TLLM_CHECK_WITH_INFO(
            mDecodingMode.isBeamSearch(), "beamWidth > 1 is given, but decoder is not configured as BeamSearch");
        typename OnlineBeamSearchLayer<T>::SetupParams beamSearchParams;

        beamSearchParams.beam_search_diversity_rate = setupParams.beam_search_diversity_rate;
        beamSearchParams.length_penalty = setupParams.length_penalty;
        beamSearchParams.early_stopping = setupParams.early_stopping;

        mHasDiffRuntimeArgs = hasDiffRuntimeArgs(beamSearchParams);
        mOnlineBeamSearchDecode->setup(batchSize, beamSearchParams);
    }
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void DynamicDecodeLayer<T>::setupPenalties(size_t batchSize, int32_t const* batchSlots, SetupParams const& setupParams)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    std::vector<int32_t> batchSlotsVec(batchSize);
    std::iota(batchSlotsVec.begin(), batchSlotsVec.end(), 0);
    auto batchSlotsHost = batchSlots ? batchSlots : batchSlotsVec.data();

    // Setup penalties.
    FillBuffers const fillBuffers{batchSize, mMaxBatchSize, mStream};

    mUseTemperature = static_cast<bool>(setupParams.temperature);
    mUseRepetitionPenalty = static_cast<bool>(setupParams.repetition_penalty);
    mUsePresencePenalty = static_cast<bool>(setupParams.presence_penalty);
    mUseFrequencyPenalty = static_cast<bool>(setupParams.frequency_penalty);
    mUseMinLength = static_cast<bool>(setupParams.min_length);
    if (mUseTemperature)
    {
        fillBuffers(setupParams.temperature, getDefaultPenaltyValue(DecodingPenaltyType::Temperature), mTemperature,
            mTemperatureDevice, batchSlotsHost);
    }
    if (mUseRepetitionPenalty)
    {
        fillBuffers(setupParams.repetition_penalty, getDefaultPenaltyValue(DecodingPenaltyType::Repetition),
            mRepetitionPenalty, mRepetitionPenaltyDevice, batchSlotsHost);
    }
    if (mUsePresencePenalty)
    {
        fillBuffers(setupParams.presence_penalty, getDefaultPenaltyValue(DecodingPenaltyType::Presence),
            mPresencePenalty, mPresencePenaltyDevice, batchSlotsHost);
    }
    if (mUseFrequencyPenalty)
    {
        fillBuffers(setupParams.frequency_penalty, getDefaultPenaltyValue(DecodingPenaltyType::Frequency),
            mFrequencyPenalty, mFrequencyPenaltyDevice, batchSlotsHost);
    }
    if (mUseMinLength)
    {
        fillBuffers(setupParams.min_length, (int) getDefaultPenaltyValue(DecodingPenaltyType::MinLength), mMinLength,
            mMinLengthDevice, batchSlotsHost);
    }
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void DynamicDecodeLayer<T>::forward(OutputParams& outputs, ForwardParams const& params)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    TLLM_CHECK_WITH_INFO(params.logits || params.logits_vec, "Either logits or logits_vec have to be specified.");
    TLLM_CHECK_WITH_INFO(
        outputs.sequence_length.has_value(), "sequence_length tensor is mandatory in DynamicDecoderLayer.");

    size_t batchSize = 0;
    size_t beamWidth = 0;
    size_t vocabSize = 0;
    auto const maxSeqLen = outputs.output_ids.shape[outputs.output_ids.shape.size() - 1];
    if (params.logits)
    {
        auto const& logitsShape = params.logits->shape;
        TLLM_CHECK(logitsShape.size() == 3);
        batchSize = logitsShape[0];
        beamWidth = logitsShape[1];
        vocabSize = logitsShape[2];
    }
    else
    {
        TLLM_CHECK(params.logits_vec->size());
        auto const& logitsShape = params.logits_vec.value()[0].shape;
        TLLM_CHECK(logitsShape.size() == 3);
        batchSize = params.logits_vec->size();
        beamWidth = logitsShape[1];
        vocabSize = logitsShape[2];
    }

    TLLM_CHECK_WITH_INFO((mConfiguredBeamWidth == 1 && beamWidth == 1)
            || (mConfiguredBeamWidth > 1 && beamWidth > 1 && beamWidth <= mConfiguredBeamWidth),
        "Decoder is configured with beam width %d, but %lu was given", mConfiguredBeamWidth, beamWidth);

    if (!mLogitsPtrsHost->data())
    {
        mLogitsPtrsHost = runtime::BufferManager::pinnedPool(
            ITensor::makeShape({static_cast<int32_t>(maxSeqLen), static_cast<int32_t>(mMaxBatchSize)}),
            runtime::TRTDataType<T*>::value);
        mIdsPtrHost = runtime::BufferManager::pinnedPool(
            ITensor::makeShape({static_cast<int32_t>(maxSeqLen), static_cast<int32_t>(2 * mMaxBatchSize)}),
            runtime::TRTDataType<int32_t*>::value);
        mRuntimeMaxSeqLen = maxSeqLen;
    }

    std::vector<int32_t> batchSlotsVec(batchSize);
    std::iota(batchSlotsVec.begin(), batchSlotsVec.end(), 0);
    auto batchSlotsHost = params.batch_slots ? params.batch_slots->template getPtr<const int>() : batchSlotsVec.data();
    auto batchSlots = params.batch_slots ? params.batch_slots->template getPtr<const int>() : nullptr;

    mCyclicStep = mCyclicStep % mRuntimeMaxSeqLen;
    prepareIdsPtrs(outputs, batchSlotsHost, batchSize, beamWidth, maxSeqLen);

    auto logits = Tensor(MEMORY_GPU, std::is_same_v<T, float> ? DataType::TYPE_FP32 : DataType::TYPE_FP16,
        {batchSize, beamWidth, mVocabSizePadded}, mRuntimeLogitsDevice);

    // Apply penalties
    applyPenalties(outputs, params, batchSlotsHost, batchSlots, batchSize, beamWidth, maxSeqLen);

    // Ban bad words and NGrams
    banWords(logits, outputs, params, batchSlots, batchSize, beamWidth, maxSeqLen, mVocabSizePadded, mStream);

    // Main function that calls forward of the respective layers
    layersForward(logits, outputs, params, batchSlots, batchSize, beamWidth, maxSeqLen);

    // Check if stop conditions are met
    checkStopCriteria(outputs, params, batchSlots, batchSize, beamWidth, maxSeqLen, mStream);

    // Copy nextIds and transpose logits when needed
    prepareOutputData(
        outputs, params, mIdsPtrHost, batchSlots, batchSize, mMaxBatchSize, beamWidth, maxSeqLen, mCyclicStep, mStream);

    mCyclicStep += 1;

    sync_check_cuda_error();
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void DynamicDecodeLayer<T>::layersForward(Tensor& logits, OutputParams& outputs, ForwardParams const& params,
    int32_t const* batchSlots, size_t batchSize, size_t beamWidth, size_t maxSeqLen)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    auto const ite = params.ite;
    auto const step = params.step;

    // common inputs
    auto const& endIds = params.end_ids;
    auto const localBatchSize = static_cast<std::size_t>(params.local_batch_size);

    // dynamic decode GPT
    if (beamWidth > 1)
    {
        TLLM_CHECK_WITH_INFO(
            mDecodingMode.isBeamSearch(), "beamWidth > 1 is given, but decoder is not configured as BeamSearch");
        TLLM_CHECK_WITH_INFO(
            params.src_cache_indirection.has_value(), "src_cache_indirection is mandatory in beam search.");
        TLLM_CHECK_WITH_INFO(
            outputs.tgt_cache_indirection.has_value(), "tgt_cache_indirection is mandatory in beam search.");
        TLLM_CHECK_WITH_INFO(outputs.parent_ids.has_value(), "parent_ids tensor is mandatory in beam search.");
        TLLM_CHECK_WITH_INFO(outputs.finished.has_value(), "finished tensor is mandatory in beam search.");
        TLLM_CHECK_WITH_INFO(outputs.cum_log_probs.has_value(), "cum_log_probs tensor is mandatory in beam search.");

        // Because we still not support batch beam search now, so we need to compute
        // one by one if there are different runtime arguments.
        const size_t dynamic_decode_batch_size = mHasDiffRuntimeArgs ? 1 : localBatchSize;
        const int dynamic_decode_total_iteration = localBatchSize / dynamic_decode_batch_size;

        for (uint32_t dynamic_ite = 0; dynamic_ite < dynamic_decode_total_iteration; ++dynamic_ite)
        {
            const int dynamic_id_offset = dynamic_ite * dynamic_decode_batch_size * beamWidth;
            const int dynamic_decode_vocab_size_units_offset = dynamic_id_offset * mVocabSizePadded;

            auto const logits_offset = logits.slice(
                {dynamic_decode_batch_size, logits.shape[1], logits.shape[2]}, dynamic_decode_vocab_size_units_offset);
            auto const end_id_offset
                = endIds.slice({dynamic_decode_batch_size}, dynamic_ite * dynamic_decode_batch_size);
            typename BaseBeamSearchLayer<T>::ForwardParams dynamic_decode_input_tensors{step, ite, logits_offset,
                end_id_offset, *params.src_cache_indirection, static_cast<std::int32_t>(params.max_attention_window),
                static_cast<std::int32_t>(params.sink_token_length), static_cast<std::int32_t>(maxSeqLen)};

            if (params.input_lengths)
            {
                dynamic_decode_input_tensors.input_lengths
                    = params.input_lengths->slice({dynamic_decode_batch_size * beamWidth}, dynamic_id_offset);
            }

            // common outputs
            typename BaseBeamSearchLayer<T>::BeamSearchOutputParams dynamic_decode_outputs(
                outputs.output_ids, outputs.parent_ids.value(), outputs.tgt_cache_indirection.value());

            dynamic_decode_outputs.output_ids_ptr = std::move(outputs.output_ids_ptr);
            dynamic_decode_outputs.parent_ids_ptr = std::move(outputs.parent_ids_ptr);

            dynamic_decode_outputs.sequence_length
                = outputs.sequence_length->slice({dynamic_decode_batch_size * beamWidth}, dynamic_id_offset);
            dynamic_decode_outputs.finished
                = outputs.finished->slice({dynamic_decode_batch_size * beamWidth}, dynamic_id_offset);
            dynamic_decode_outputs.cum_log_probs
                = outputs.cum_log_probs->slice({dynamic_decode_batch_size * beamWidth}, dynamic_id_offset);

            dynamic_decode_outputs.beamHypotheses = outputs.beamHypotheses;
            dynamic_decode_outputs.output_log_probs = outputs.output_log_probs_tiled;

            // only OnlineBeamSearchLayer support beam_search_diversity_rate
            // when beamHypotheses is used
            mOnlineBeamSearchDecode->forward(dynamic_decode_outputs, dynamic_decode_input_tensors);
        } // end of dynamic_ite
        std::swap(mPenaltyWorkspaceDevice, mPenaltyWorkspacePrevDevice);
    }
    else
    { // beamWidth == 1
        TLLM_CHECK_WITH_INFO(
            mDecodingMode.isTopKorTopP(), "beamWidth == 1 is given, but decoder is not configured as TopK or TopP");

        // In sampling, we have supported batch sampling. So, we always compute all
        // sentences once.
        Tensor const logits_slice{logits.slice({localBatchSize, beamWidth, logits.shape[2]}, 0)};
        Tensor const end_id_slice{endIds.slice({localBatchSize}, 0)};
        typename BaseSamplingLayer<T>::ForwardParams decode_input_tensors{
            step, ite, logits_slice, end_id_slice, static_cast<std::int32_t>(maxSeqLen)};

        decode_input_tensors.finished = params.finished;

        if (params.input_lengths)
        {
            auto& input_lengths = params.input_lengths.value();
            decode_input_tensors.input_lengths = input_lengths.slice({localBatchSize, beamWidth}, 0);
        }
        decode_input_tensors.batch_slots = params.batch_slots;

        DecodingOutputParams decode_outputs(outputs.output_ids);
        decode_outputs.output_ids_ptr = std::move(outputs.output_ids_ptr);
        if (outputs.sequence_length)
        {
            decode_outputs.sequence_length = outputs.sequence_length->slice({localBatchSize * beamWidth}, 0);
        }
        if (outputs.finished)
        {
            decode_outputs.finished = outputs.finished->slice({localBatchSize * beamWidth}, 0);
        }
        if (outputs.cum_log_probs)
        {
            decode_outputs.cum_log_probs = outputs.cum_log_probs->slice({localBatchSize * beamWidth}, 0);
        }
        if (outputs.output_log_probs_tiled)
        {
            Tensor& output_log_probs = outputs.output_log_probs_tiled.value();
            decode_outputs.output_log_probs = output_log_probs.slice({1, localBatchSize * beamWidth}, 0);
        }

        // Run TopK + TopP decode layers.
        mSamplingLayer->forward(decode_outputs, decode_input_tensors);
    }
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void DynamicDecodeLayer<T>::applyPenalties(OutputParams& outputs, ForwardParams const& params,
    int32_t const* batchSlotsHost, int32_t const* batchSlots, size_t batchSize, size_t beamWidth, size_t maxSeqLen)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    auto logitsPtrsHost = ITensor::slice(mLogitsPtrsHost, mCyclicStep, 1);
    auto logitsPtrsHostData = reinterpret_cast<T const**>(runtime::bufferCast<int64_t>(*logitsPtrsHost));
    for (int32_t bi = 0; bi < batchSize; bi++)
    {
        if (params.logits_vec)
        {
            TLLM_CHECK_WITH_INFO(params.logits_vec->size() == batchSize,
                "Logits vector size (%lu) is not equal to the batchSize (%lu)", params.logits_vec->size(), batchSize);
            logitsPtrsHostData[bi] = params.logits_vec.value()[bi].template getPtr<T>();
        }
        else
        {
            logitsPtrsHostData[bi] = params.logits->template getPtrWithOffset<T>(bi * beamWidth * mVocabSizePadded);
        }
    }

    int32_t const* inputLengths = nullptr;
    if (params.input_lengths)
    {
        auto& input_lengths = params.input_lengths.value();
        inputLengths = input_lengths.template getPtr<const int>();
    }
    auto* embeddingBias = params.embedding_bias ? params.embedding_bias->template getPtr<T const>() : nullptr;
#define GET_PENALTIES(capital_name, penalty_name, type)                                                                \
    (mUse##capital_name                                                                                                \
        && !allOfBatchSlots(batchSlotsHost, m##capital_name.data(), batchSize,                                         \
            static_cast<type>(getDefaultPenaltyValue(DecodingPenaltyType::penalty_name))))                             \
        ? m##capital_name##Device                                                                                      \
        : nullptr;

    auto* temperatures = GET_PENALTIES(Temperature, Temperature, float);
    auto* repetitionPenalties = GET_PENALTIES(RepetitionPenalty, Repetition, float);
    auto* presencePenalties = GET_PENALTIES(PresencePenalty, Presence, float);
    auto* frequencyPenalties = GET_PENALTIES(FrequencyPenalty, Frequency, float);
    auto* minLengths = GET_PENALTIES(MinLength, MinLength, int32_t);

#undef GET_PENALTIES

    InvokeBatchApplyPenaltyParams<T> penaltyParams{reinterpret_cast<T const* const*>(logitsPtrsHostData),
        mRuntimeLogitsDevice, embeddingBias, mPenaltyWorkspaceDevice, mPenaltyWorkspacePrevDevice, temperatures,
        repetitionPenalties, presencePenalties, frequencyPenalties,
        (mUseRepetitionPenalty || mUsePresencePenalty || mUseFrequencyPenalty), batchSize,
        static_cast<int32_t>(beamWidth), static_cast<int32_t>(maxSeqLen), mVocabSize, mVocabSizePadded,
        outputs.output_ids_ptr.template getPtr<const int*>(), outputs.parent_ids_ptr.template getPtr<const int*>(),
        inputLengths, outputs.sequence_length->template getPtr<const int>(), minLengths,
        params.end_ids.template getPtr<const int>(), batchSlots, mStream};
    invokeBatchApplyPenalty(penaltyParams);
    sync_check_cuda_error();

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void DynamicDecodeLayer<T>::banWords(Tensor& logits, OutputParams& outputs, ForwardParams const& params,
    int32_t const* batchSlots, size_t batchSize, size_t beamWidth, size_t maxSeqLen, size_t vocabSizePadded,
    cudaStream_t stream)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    banRepeatNGrams(logits, outputs, params, batchSlots, batchSize, beamWidth, maxSeqLen, vocabSizePadded, stream);
    banBadWords(logits, outputs, params, batchSlots, batchSize, beamWidth, maxSeqLen, vocabSizePadded, stream);

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void DynamicDecodeLayer<T>::banRepeatNGrams(Tensor& logits, OutputParams& outputs, ForwardParams const& params,
    int32_t const* batchSlots, size_t batchSize, size_t beamWidth, size_t maxSeqLen, size_t vocabSizePadded,
    cudaStream_t stream)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    auto const max_step = params.step;
    if (params.no_repeat_ngram_size)
    {
        const int* noRepeatNgramSizeBuf = params.no_repeat_ngram_size.value().template getPtr<const int>();

        invokeBanRepeatNgram(logits.template getPtr<T>(), outputs.output_ids_ptr.template getPtr<const int*>(),
            reinterpret_cast<FinishedState*>(
                params.finished.value_or(Tensor{}).template getPtr<FinishedState::UnderlyingType>()),
            outputs.parent_ids_ptr.template getPtr<const int*>(), batchSlots,
            outputs.sequence_length->template getPtr<int>(), batchSize, beamWidth, maxSeqLen,
            params.no_repeat_ngram_size.value().template getPtr<const int>(), vocabSizePadded, max_step, stream);
    }
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void DynamicDecodeLayer<T>::banBadWords(Tensor& logits, OutputParams& outputs, ForwardParams const& params,
    int32_t const* batchSlots, size_t batchSize, size_t beamWidth, size_t maxSeqLen, size_t vocabSizePadded,
    cudaStream_t stream)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    auto const maxBadWordsLength = params.max_bad_words_len;
    if (maxBadWordsLength)
    {
        int32_t const** badWordsPtr = params.bad_words_ptr->template getPtr<int32_t const*>();
        int32_t const* badWordsLens = params.bad_words_lengths->template getPtr<int32_t>();

        invokeBanBadWords((T*) logits.template getPtr<T>(), outputs.output_ids_ptr.template getPtr<int32_t const*>(),
            beamWidth > 1 ? outputs.parent_ids_ptr.template getPtr<int32_t const*>() : nullptr, batchSlots, batchSize,
            beamWidth, badWordsPtr, badWordsLens, maxBadWordsLength, vocabSizePadded,
            outputs.sequence_length->template getPtr<int>(), maxSeqLen, stream);
    }
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void DynamicDecodeLayer<T>::checkStopCriteria(OutputParams& outputs, ForwardParams const& params,
    int32_t const* batchSlots, size_t batchSize, size_t beamWidth, size_t maxSeqLen, cudaStream_t stream)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    checkStopWordsStopCriteria(outputs, params, batchSlots, batchSize, beamWidth, maxSeqLen, stream);
    checkMaxLengthStopCriteria(outputs, params, batchSlots, batchSize, beamWidth, maxSeqLen, stream);

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void DynamicDecodeLayer<T>::checkStopWordsStopCriteria(OutputParams& outputs, ForwardParams const& params,
    int32_t const* batchSlots, size_t batchSize, size_t beamWidth, size_t maxSeqLen, cudaStream_t stream)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    auto const maxStopWordsLength = params.max_stop_words_len;
    if (maxStopWordsLength)
    {
        invokeStopWordsCriterion(outputs.output_ids_ptr.template getPtr<int32_t const*>(),
            outputs.parent_ids_ptr.template getPtr<int32_t const*>(),
            params.stop_words_ptr->template getPtr<int32_t const*>(),
            reinterpret_cast<FinishedState*>(outputs.finished->template getPtr<FinishedState::UnderlyingType>()),
            outputs.sequence_length->template getPtr<int32_t>(), batchSlots,
            params.stop_words_lengths->template getPtr<int32_t const>(), maxStopWordsLength, batchSize, beamWidth,
            maxSeqLen, stream);
    }
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void DynamicDecodeLayer<T>::checkMaxLengthStopCriteria(OutputParams& outputs, ForwardParams const& params,
    int32_t const* batchSlots, size_t batchSize, size_t beamWidth, size_t maxSeqLen, cudaStream_t stream)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    if (params.sequence_limit_length)
    {
        invokeLengthCriterion(
            reinterpret_cast<FinishedState*>(outputs.finished->template getPtr<FinishedState::UnderlyingType>()),
            outputs.finished_sum ? outputs.finished_sum->template getPtr<int>() : nullptr,
            params.sequence_limit_length->template getPtr<const uint32_t>(),
            outputs.sequence_length->template getPtr<int>(), batchSlots, batchSize, beamWidth, stream);
        sync_check_cuda_error();
    }
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void DynamicDecodeLayer<T>::prepareIdsPtrs(
    OutputParams& outputs, int32_t const* batchSlots, size_t batchSize, size_t beamWidth, size_t maxSeqLen)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    auto idsPtrHostSlice = ITensor::slice(mIdsPtrHost, mCyclicStep, 1);
    auto idsPtrHost = reinterpret_cast<int32_t**>(runtime::bufferCast<int64_t>(*idsPtrHostSlice));
    for (int bi = 0; bi < batchSize; bi++)
    {
        auto const batchSlot = batchSlots[bi];
        idsPtrHost[batchSlot]
            = outputs.output_ids.template getPtrWithOffset<int32_t>(batchSlot * beamWidth * maxSeqLen);
    }
    for (int bi = 0; bi < batchSize; bi++)
    {
        auto const batchSlot = batchSlots[bi];
        if (beamWidth > 1)
        {
            idsPtrHost[mMaxBatchSize + batchSlot]
                = outputs.parent_ids.value().template getPtrWithOffset<int32_t>(bi * beamWidth * maxSeqLen);
        }
        else
        {
            idsPtrHost[mMaxBatchSize + batchSlot] = mZeroParentIdsDevice + bi * beamWidth * maxSeqLen;
        }
    }

    outputs.output_ids_ptr
        = Tensor(MEMORY_GPU, DataType::TYPE_INT32_PTR, {mMaxBatchSize, beamWidth, maxSeqLen}, idsPtrHost);
    outputs.parent_ids_ptr = Tensor(
        MEMORY_GPU, DataType::TYPE_INT32_PTR, {mMaxBatchSize, beamWidth, maxSeqLen}, idsPtrHost + mMaxBatchSize);
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void DynamicDecodeLayer<T>::prepareOutputData(OutputParams& outputs, ForwardParams const& params,
    runtime::ITensor::SharedPtr const& idsPtrsHost, int32_t const* batchSlots, size_t batchSize, size_t maxBatchSize,
    size_t beamWidth, size_t maxSeqLen, int32_t cyclicStep, cudaStream_t stream)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    auto idsPtrHostSlice = ITensor::slice(idsPtrsHost, cyclicStep, 1);
    auto idsPtrHost = reinterpret_cast<int32_t**>(runtime::bufferCast<int64_t>(*idsPtrHostSlice));
    invokeCopyNextStepIds(outputs.newTokens.template getPtr<int>(), idsPtrHost,
        outputs.sequence_length->template getPtr<int>(), batchSlots, batchSize, beamWidth, maxSeqLen, stream);

    // Transpose the output log probs from [maxSeqLen, bs, beamWidth] to [batchSize, beamWidth, maxSeqLen]
    if (outputs.output_log_probs_tiled)
    {
        auto logProbsMaxSeqLen = outputs.output_log_probs_tiled.value().shape[0];

        invokeTransposeLogProbs(outputs.output_log_probs.value().template getPtr<float>(),
            outputs.output_log_probs_tiled.value().template getPtr<float>(),
            outputs.sequence_length->template getPtr<int>(), batchSlots, batchSize, maxBatchSize, beamWidth,
            logProbsMaxSeqLen, stream);
    }
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template class DynamicDecodeLayer<float>;
template class DynamicDecodeLayer<half>;

} // namespace layers
} // namespace tensorrt_llm
