/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.  All rights reserved.
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

#include "tensorrt_llm/layers/baseSamplingLayer.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/memoryUtils.h"
#include "tensorrt_llm/kernels/penaltyKernels.h"
#include "tensorrt_llm/kernels/samplingTopKKernels.h"
#include "tensorrt_llm/layers/fillBuffers.h"

#include <algorithm>

using namespace tensorrt_llm::common;
using namespace tensorrt_llm::kernels;

namespace tensorrt_llm
{
namespace layers
{
template <typename T>
void BaseSamplingLayer<T>::allocateBuffer(size_t batch_size)
{
    TLLM_LOG_TRACE(__PRETTY_FUNCTION__);
    curandstate_buf_ = allocator_->reMalloc(curandstate_buf_, sizeof(curandState_t) * batch_size, false);
    random_seeds_buf_ = allocator_->reMalloc(random_seeds_buf_, sizeof(uint64_t) * batch_size, false);
    temperature_buf_ = allocator_->reMalloc(temperature_buf_, sizeof(float) * batch_size, false);
    repetition_penalty_buf_ = allocator_->reMalloc(repetition_penalty_buf_, sizeof(float) * batch_size, false);
    presence_penalty_buf_ = allocator_->reMalloc(presence_penalty_buf_, sizeof(float) * batch_size, false);
    frequency_penalty_buf_ = allocator_->reMalloc(frequency_penalty_buf_, sizeof(float) * batch_size, false);
    min_lengths_buf_ = allocator_->reMalloc(min_lengths_buf_, sizeof(int) * batch_size, false);
    runtime_logits_buf_ = allocator_->reMalloc(runtime_logits_buf_, sizeof(T) * batch_size * vocab_size_padded_, false);
    skip_decode_buf_ = allocator_->reMalloc(skip_decode_buf_, sizeof(bool) * batch_size, false);

    // host buffers.
    skip_decode_ = (bool*) std::realloc(skip_decode_, sizeof(bool) * batch_size);
    TLLM_CHECK(skip_decode_ != nullptr);

    is_allocate_buffer_ = true;
}

template <typename T>
void BaseSamplingLayer<T>::freeBuffer()
{
    TLLM_LOG_TRACE(__PRETTY_FUNCTION__);
    if (is_allocate_buffer_)
    {
        allocator_->free((void**) (&curandstate_buf_));
        allocator_->free((void**) (&random_seeds_buf_));
        allocator_->free((void**) (&temperature_buf_));
        allocator_->free((void**) (&repetition_penalty_buf_));
        allocator_->free((void**) (&presence_penalty_buf_));
        allocator_->free((void**) (&frequency_penalty_buf_));
        allocator_->free((void**) (&min_lengths_buf_));
        allocator_->free((void**) (&runtime_logits_buf_));
        allocator_->free((void**) (&skip_decode_buf_));
        std::free(skip_decode_);
        is_allocate_buffer_ = false;
    }
}

template <typename T>
BaseSamplingLayer<T>::BaseSamplingLayer(size_t vocab_size, size_t vocab_size_padded, cudaStream_t stream,
    std::shared_ptr<IAllocator> allocator, bool is_free_buffer_after_forward, cudaDeviceProp* cuda_device_prop)
    : BaseLayer(stream, std::move(allocator), is_free_buffer_after_forward, cuda_device_prop)
    , vocab_size_(vocab_size)
    , vocab_size_padded_(vocab_size_padded)
{
}

template <typename T>
BaseSamplingLayer<T>::BaseSamplingLayer(BaseSamplingLayer const& sampling_layer)
    : BaseLayer(sampling_layer)
    , vocab_size_(sampling_layer.vocab_size_)
    , vocab_size_padded_(sampling_layer.vocab_size_padded_)
    , sampling_workspace_size_(sampling_layer.sampling_workspace_size_)
{
}

template <typename T>
BaseSamplingLayer<T>::~BaseSamplingLayer()
{
}

template <typename T>
void BaseSamplingLayer<T>::setupBase(const size_t batch_size, SetupParams const& setupParams)
{
    TLLM_LOG_TRACE(__PRETTY_FUNCTION__);
    allocateBuffer(batch_size);

    // If runtime argument has single random seed, using this random seed to
    // initialize the random table of all sentences. If the argument has
    // [batch_size] random seeds, initializing the random table by different
    // random seeds respectively. If no random seed, initialize the random table
    // of all sentences by 0 directly.
    if (setupParams.randomSeed)
    {
        if (setupParams.randomSeed->size() == 1)
        {
            invokeCurandInitialize(curandstate_buf_, batch_size, setupParams.randomSeed->front(), stream_);
            sync_check_cuda_error();
        }
        else
        {
            TLLM_CHECK_WITH_INFO(setupParams.randomSeed->size() == batch_size, "Random seed vector size mismatch.");
            cudaAutoCpy(random_seeds_buf_, setupParams.randomSeed->data(), batch_size, stream_);
            invokeCurandBatchInitialize(curandstate_buf_, batch_size, random_seeds_buf_, stream_);
            sync_check_cuda_error();
        }
    }
    else
    {
        // Initialize curand states using the default seed 0.
        invokeCurandInitialize(curandstate_buf_, batch_size, 0, stream_);
    }

    // Setup penalties.
    FillBuffers const fillBuffers{batch_size, stream_};

    use_temperature_ = static_cast<bool>(setupParams.temperature);
    use_repetition_penalty_ = static_cast<bool>(setupParams.repetition_penalty);
    use_presence_penalty_ = static_cast<bool>(setupParams.presence_penalty);
    use_frequency_penalty_ = static_cast<bool>(setupParams.frequency_penalty);
    use_min_lengths_ = static_cast<bool>(setupParams.min_length);
    if (use_temperature_)
    {
        fillBuffers(setupParams.temperature, getDefaultPenaltyValue(RepetitionPenaltyType::Temperature), mTemperature,
            temperature_buf_);
    }
    if (use_repetition_penalty_)
    {
        fillBuffers(setupParams.repetition_penalty, getDefaultPenaltyValue(RepetitionPenaltyType::Repetition),
            mRepetitionPenalty, repetition_penalty_buf_);
    }
    if (use_presence_penalty_)
    {
        fillBuffers(setupParams.presence_penalty, getDefaultPenaltyValue(RepetitionPenaltyType::Presence),
            mPresencePenalty, presence_penalty_buf_);
    }
    if (use_frequency_penalty_)
    {
        fillBuffers(setupParams.frequency_penalty, getDefaultPenaltyValue(RepetitionPenaltyType::Frequency),
            mFrequencyPenalty, frequency_penalty_buf_);
    }
    if (use_min_lengths_)
    {
        fillBuffers(setupParams.min_length, (int) getDefaultPenaltyValue(RepetitionPenaltyType::MinLength), mMinLengths,
            min_lengths_buf_);
    }
}

template <typename T>
void BaseSamplingLayer<T>::forward(DecodingOutputParams& outputs, ForwardParams const& params, int* penalty_workspace)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    auto const batch_size = outputs.output_ids_ptr.shape[0];
    auto const local_batch_size = params.logits.shape[0];
    auto const ite = params.ite;
    auto const step = params.step;
    auto* const input_lengths = params.input_lengths ? params.input_lengths->template getPtr<const int>() : nullptr;

    auto* logits = params.logits.template getPtr<T>();

#define ALL_OF(p_, sz_, dt_, v_) (std::all_of(p_, p_ + sz_, [&](dt_ b) { return b == v_; }))

    bool* skip_decode = skip_decode_ + ite * local_batch_size;
    if (ALL_OF(skip_decode, local_batch_size, bool, true))
    {
        // No sample in the current batch to do TopX sampling.
        return;
    }
    skip_any_ = std::any_of(skip_decode, skip_decode + local_batch_size, [](bool b) { return b; });
    if (skip_any_)
    {
        // A TopX Sampling layer directly changes the logit values. In case of
        // skip_any==true, meaning topk and topp layers will run simultaneously for
        // a batch in the same step. We copy the logits to an internal buffer, not
        // affecting the other sampling layers.
        TLLM_CHECK(params.logits.size() == local_batch_size * vocab_size_padded_);
        cudaD2Dcpy(runtime_logits_buf_, logits, params.logits.size(), stream_);
        logits = runtime_logits_buf_;
    }

    auto* embedding_bias = params.embedding_bias ? params.embedding_bias->template getPtr<T const>() : nullptr;
    auto* temperatures = (use_temperature_
                             && !ALL_OF(std::begin(mTemperature) + ite * local_batch_size, local_batch_size, float,
                                 getDefaultPenaltyValue(RepetitionPenaltyType::Temperature)))
        ? temperature_buf_ + ite * local_batch_size
        : nullptr;
    auto* repetition_penalties
        = (use_repetition_penalty_
              && !ALL_OF(std::begin(mRepetitionPenalty) + ite * local_batch_size, local_batch_size, float,
                  getDefaultPenaltyValue(RepetitionPenaltyType::Repetition)))
        ? repetition_penalty_buf_ + ite * local_batch_size
        : nullptr;
    auto* presence_penalties = (use_presence_penalty_
                                   && !ALL_OF(std::begin(mPresencePenalty) + ite * local_batch_size, local_batch_size,
                                       float, getDefaultPenaltyValue(RepetitionPenaltyType::Presence)))
        ? presence_penalty_buf_ + ite * local_batch_size
        : nullptr;
    auto* frequency_penalties = (use_frequency_penalty_
                                    && !ALL_OF(std::begin(mFrequencyPenalty) + ite * local_batch_size, local_batch_size,
                                        float, getDefaultPenaltyValue(RepetitionPenaltyType::Frequency)))
        ? frequency_penalty_buf_ + ite * local_batch_size
        : nullptr;
    auto* min_lengths = (use_min_lengths_
                            && !ALL_OF(std::begin(mMinLengths) + ite * local_batch_size, local_batch_size, int,
                                (int) getDefaultPenaltyValue(RepetitionPenaltyType::MinLength)))
        ? min_lengths_buf_ + ite * local_batch_size
        : nullptr;

    InvokeBatchApplyPenaltyParams<T> penalty_params{logits, embedding_bias,
        penalty_workspace + ite * local_batch_size * vocab_size_, nullptr, temperatures, repetition_penalties,
        presence_penalties, frequency_penalties,
        (use_repetition_penalty_ || use_presence_penalty_ || use_frequency_penalty_), local_batch_size, 1,
        params.max_seq_len, vocab_size_, vocab_size_padded_, outputs.output_ids_ptr.template getPtr<const int*>(),
        nullptr, input_lengths, outputs.sequence_length->getPtr<const int>(), min_lengths,
        params.end_ids.template getPtr<const int>(), stream_};
    invokeBatchApplyPenalty(penalty_params);
    sync_check_cuda_error();
#undef ALL_OF

    runSampling(outputs, params);

    if (is_free_buffer_after_forward_)
    {
        freeBuffer();
    }
    sync_check_cuda_error();
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template class BaseSamplingLayer<float>;
template class BaseSamplingLayer<half>;

} // namespace layers
} // namespace tensorrt_llm
