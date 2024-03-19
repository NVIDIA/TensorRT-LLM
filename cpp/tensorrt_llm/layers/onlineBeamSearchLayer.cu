/*
 * Copyright (c) 2019-2024, NVIDIA CORPORATION.  All rights reserved.
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

#include "tensorrt_llm/common/memoryUtils.h"
#include "tensorrt_llm/kernels/beamSearchTopkKernels.h"
#include "tensorrt_llm/layers/fillBuffers.h"
#include "tensorrt_llm/layers/onlineBeamSearchLayer.h"

using namespace tensorrt_llm::common;
using namespace tensorrt_llm::kernels;

namespace tensorrt_llm
{
namespace layers
{

static int const SMALL_TOP_K_SOFTMAX_MAX_VOC_PARTS = 128;
static int const MAX_K = 4;

template <typename T>
void OnlineBeamSearchLayer<T>::setup(runtime::SizeType batch_size, SetupParams const& setupParams)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    BaseBeamSearchLayer<T>::setupBase(batch_size, setupParams);
    allocateBuffer(batch_size);

    mDiversityRate.resize(batch_size);
    mLengthPenalty.resize(batch_size);
    mEarlyStopping.resize(batch_size);
    FillBuffers const fillBuffers{batch_size, batch_size, mStream};

    fillBuffers(setupParams.beam_search_diversity_rate, 0.0f, mDiversityRate, diversity_rates_buf_, (int*) nullptr);
    fillBuffers(setupParams.length_penalty, 0.0f, mLengthPenalty, length_penalties_buf_, (int*) nullptr);
    fillBuffers(setupParams.early_stopping, 1, mEarlyStopping, early_stoppings_buf_, (int*) nullptr);
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void OnlineBeamSearchLayer<T>::invokeSoftMax(BeamSearchOutputParams& outputs, SoftmaxParams const& params)
{
    TLLM_LOG_TRACE("%s", __PRETTY_FUNCTION__);
    TLLM_CHECK_WITH_INFO(outputs.beamHypotheses, std::string("Output BeamHypotheses is not set"));

    BeamHypotheses bh{*outputs.beamHypotheses};
    bh.end_ids = params.end_ids.template getPtr<int const>();
    bh.finished = reinterpret_cast<FinishedState*>(outputs.finished->template getPtr<FinishedState::UnderlyingType>());
    bh.cum_log_probs_src = outputs.cum_log_probs->template getPtr<float>();
    bh.log_probs_src = (outputs.output_log_probs) ? outputs.output_log_probs->template getPtr<float>() : nullptr;
    bh.sequence_lengths_src = outputs.sequence_length->template getPtr<int>();
    bh.output_ids_tgt_ptr = outputs.output_ids_ptr.template getPtr<int*>();
    bh.parent_ids_tgt_ptr = outputs.parent_ids_ptr.template getPtr<int*>();
    bh.diversity_rates = diversity_rates_buf_;
    bh.length_penalties = length_penalties_buf_;
    bh.early_stoppings = early_stoppings_buf_;

    bh.batch_size = static_cast<std::int32_t>(outputs.output_ids_ptr.shape[0]);
    bh.beam_width = static_cast<std::int32_t>(outputs.output_ids_ptr.shape[1]);
    bh.ite = params.ite;
    bh.local_batch_size = params.logits.shape[0];
    bh.max_seq_len = static_cast<std::int32_t>(outputs.output_ids_ptr.shape[2]);
    bh.vocab_size = vocab_size_padded_;

    T const* logits = params.logits.template getPtr<T>();
    T const* bias = static_cast<T const*>(nullptr);

    invokeTopkSoftMax(logits, bias, topk_softmax_workspace_, topk_softmax_workspace_size_, bh, mStream);
    sync_check_cuda_error();
}

template <typename T>
void OnlineBeamSearchLayer<T>::allocateBuffer(runtime::SizeType batch_size)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    // we need to check 2 * beam_width candidates each time
    // 64 is the max beam width we support now.
    topk_softmax_workspace_size_ = (size_t) (ceil(batch_size * 64 * (64 * 2) / 4.) * 4 * 2
        + ceil(batch_size * (64 * 2) * SMALL_TOP_K_SOFTMAX_MAX_VOC_PARTS * (2 * (MAX_K * 2) + 2) / 4.) * 4);

    topk_softmax_workspace_ = reinterpret_cast<float*>(
        mAllocator->reMalloc(topk_softmax_workspace_, sizeof(float) * topk_softmax_workspace_size_, true));
    diversity_rates_buf_ = mAllocator->reMalloc(diversity_rates_buf_, sizeof(float) * batch_size, false);
    length_penalties_buf_ = mAllocator->reMalloc(length_penalties_buf_, sizeof(float) * batch_size, false);
    early_stoppings_buf_ = mAllocator->reMalloc(early_stoppings_buf_, sizeof(int) * batch_size, false);

    mIsAllocateBuffer = true;
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void OnlineBeamSearchLayer<T>::freeBuffer()
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    if (mIsAllocateBuffer)
    {
        mAllocator->free((void**) (&topk_softmax_workspace_));
        mAllocator->free((void**) (&diversity_rates_buf_));
        mAllocator->free((void**) (&length_penalties_buf_));
        mAllocator->free((void**) (&early_stoppings_buf_));
        mIsAllocateBuffer = false;
    }
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
OnlineBeamSearchLayer<T>::OnlineBeamSearchLayer(runtime::SizeType vocab_size, runtime::SizeType vocab_size_padded,
    cudaStream_t stream, std::shared_ptr<IAllocator> allocator)
    : BaseBeamSearchLayer<T>(vocab_size, vocab_size_padded, stream, std::move(allocator))
{
}

template <typename T>
OnlineBeamSearchLayer<T>::OnlineBeamSearchLayer(OnlineBeamSearchLayer<T> const& beam_search_layer)
    : BaseBeamSearchLayer<T>(beam_search_layer)
{
    TLLM_LOG_TRACE(__PRETTY_FUNCTION__);
}

template <typename T>
OnlineBeamSearchLayer<T>::~OnlineBeamSearchLayer()
{
    TLLM_LOG_TRACE(__PRETTY_FUNCTION__);
}

template class OnlineBeamSearchLayer<float>;
template class OnlineBeamSearchLayer<half>;

} // namespace layers
} // namespace tensorrt_llm
