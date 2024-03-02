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

static const int SMALL_TOP_K_SOFTMAX_MAX_VOC_PARTS = 128;
static const int MAX_K = 4;

template <typename T>
__global__ void update_kernel(FinishedState* finished, BeamHypotheses beam_hyps)
{
    const int beam_width{beam_hyps.beam_width};
    const int ite{beam_hyps.ite};
    const int local_batch_size{beam_hyps.local_batch_size};
    const int max_seq_len{beam_hyps.max_seq_len};
    const int vocab_size{beam_hyps.vocab_size};
    const int end_id{beam_hyps.end_ids[blockIdx.x]};
    int* num_beams{beam_hyps.num_beams};
    int* sequence_lengths{beam_hyps.sequence_lengths_src};
    int** output_ids_ptr{beam_hyps.output_ids_tgt_ptr};
    int** parent_ids_ptr{beam_hyps.parent_ids_tgt_ptr};

    extern __shared__ char s_buf[]; // intermediate result
    int* s_sequence_lengths = reinterpret_cast<int*>(s_buf);

    for (int beam_idx = threadIdx.x; beam_idx < beam_width; beam_idx += blockDim.x)
    {
        const auto batch_beam_idx = blockIdx.x * beam_width + beam_idx;
        s_sequence_lengths[beam_idx] = sequence_lengths[batch_beam_idx];
    }
    __syncthreads();

    for (int beam_idx = threadIdx.x; beam_idx < beam_width; beam_idx += blockDim.x)
    {
        const auto batch_beam_idx = blockIdx.x * beam_width + beam_idx;
        const int current_step{s_sequence_lengths[beam_idx]};

        // Increase the seq_len even if the request has finished.
        // On the following iteration we check if the sequence has finished before
        const auto finish_state = finished[batch_beam_idx];
        if (!finish_state.isFinished())
        {
            s_sequence_lengths[beam_idx]++;
        }

        int new_word_id{output_ids_ptr[blockIdx.x][beam_idx * max_seq_len + current_step]};
        int new_beam_id{(new_word_id / vocab_size) % beam_width};
        new_word_id = new_word_id % vocab_size;

        sequence_lengths[batch_beam_idx] = s_sequence_lengths[new_beam_id];
        if (new_word_id == end_id)
        {
            finished[batch_beam_idx].setFinishedEOS();
        }
        parent_ids_ptr[blockIdx.x][beam_idx * max_seq_len + current_step] = new_beam_id;
        output_ids_ptr[blockIdx.x][beam_idx * max_seq_len + current_step] = new_word_id;
    }
    if (num_beams != nullptr && num_beams[ite * local_batch_size + blockIdx.x] == beam_width)
    {
        for (int beam_idx = threadIdx.x; beam_idx < beam_width; beam_idx += blockDim.x)
        {
            finished[blockIdx.x * beam_width + beam_idx].setFinished();
        }
    }
}

void invokeUpdate(FinishedState* finished, BeamHypotheses& beam_hyps, cudaStream_t stream)
{
    dim3 grid(beam_hyps.local_batch_size);
    dim3 block(min(beam_hyps.beam_width, 1024));
    update_kernel<float><<<grid, block, sizeof(int) * beam_hyps.beam_width, stream>>>(finished, beam_hyps);
}

template <typename T>
void OnlineBeamSearchLayer<T>::setup(size_t batch_size, SetupParams const& setupParams)
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
    auto* finished
        = reinterpret_cast<FinishedState*>(outputs.finished->template getPtr<FinishedState::UnderlyingType>());

    BeamHypotheses beam_hyps;
    if (outputs.beamHypotheses)
    {
        beam_hyps = *outputs.beamHypotheses;
        // Some of beam_hyps members have been initialized before function invokeSoftMax
        beam_hyps.end_ids = params.end_ids.template getPtr<const int>();
        beam_hyps.log_probs = (outputs.output_log_probs) ? outputs.output_log_probs->template getPtr<float>() : nullptr;
        beam_hyps.output_ids_src_ptr = outputs.output_ids_ptr.template getPtr<const int*>();
        beam_hyps.output_ids_tgt_ptr = outputs.output_ids_ptr.template getPtr<int*>();
        beam_hyps.parent_ids_src_ptr = outputs.parent_ids_ptr.template getPtr<const int*>();
        beam_hyps.parent_ids_tgt_ptr = outputs.parent_ids_ptr.template getPtr<int*>();
        beam_hyps.sequence_lengths_src = outputs.sequence_length->template getPtr<int>();

        beam_hyps.batch_size = static_cast<std::int32_t>(outputs.output_ids_ptr.shape[0]);
        beam_hyps.beam_width = static_cast<std::int32_t>(outputs.output_ids_ptr.shape[1]);
        beam_hyps.ite = params.ite;
        beam_hyps.local_batch_size = params.logits.shape[0];
        beam_hyps.max_seq_len = static_cast<std::int32_t>(outputs.output_ids_ptr.shape[2]);
        beam_hyps.vocab_size = vocab_size_padded_;
        beam_hyps.diversity_rates = diversity_rates_buf_;
        beam_hyps.length_penalties = length_penalties_buf_;
        beam_hyps.early_stoppings = early_stoppings_buf_;
    }

    invokeTopkSoftMax(params.logits.template getPtr<T>(), (const T*) (nullptr), finished,
        outputs.cum_log_probs->template getPtr<float>(), topk_softmax_workspace_, topk_softmax_workspace_size_,
        beam_hyps, mStream);
    sync_check_cuda_error();

    invokeUpdate(finished, beam_hyps, mStream);
    sync_check_cuda_error();
}

template <typename T>
void OnlineBeamSearchLayer<T>::allocateBuffer(size_t batch_size)
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
OnlineBeamSearchLayer<T>::OnlineBeamSearchLayer(
    size_t vocab_size, size_t vocab_size_padded, cudaStream_t stream, std::shared_ptr<IAllocator> allocator)
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
