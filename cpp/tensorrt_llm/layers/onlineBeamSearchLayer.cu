/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.  All rights reserved.
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
__global__ void update_kernel(FinishedState* finished, int** parent_ids_ptr, int* sequence_lengths,
    int** output_ids_ptr, BeamHypotheses beam_hyps, const int vocab_size, const int* end_ids,
    const int local_batch_size, const int beam_width, const int max_seq_len)
{
    extern __shared__ char s_buf[]; // intermediate result
    int* s_sequence_lengths = (int*) (s_buf);

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
        if (new_word_id == end_ids[blockIdx.x])
        {
            finished[batch_beam_idx].setFinishedEOS();
        }
        parent_ids_ptr[blockIdx.x][beam_idx * max_seq_len + current_step] = new_beam_id;
        output_ids_ptr[blockIdx.x][beam_idx * max_seq_len + current_step] = new_word_id;
    }
    if (beam_hyps.num_beams != nullptr)
    {
        if (beam_hyps.num_beams[beam_hyps.ite * beam_hyps.local_batch_size + blockIdx.x] == beam_width)
        {
            for (int beam_idx = threadIdx.x; beam_idx < beam_width; beam_idx += blockDim.x)
            {
                const auto batch_beam_idx = blockIdx.x * beam_width + beam_idx;
                finished[batch_beam_idx].setFinished();
            }
        }
    }
}

void invokeUpdate(FinishedState* finished, int** parent_ids_ptr, int* sequence_lengths, int** output_ids_ptr,
    BeamHypotheses* beam_hyps, const int local_batch_size, const int beam_width, const int vocab_size_padded,
    const int* end_ids, const int max_seq_len, cudaStream_t stream)
{
    dim3 grid(local_batch_size);
    dim3 block(min(beam_width, 1024));

    update_kernel<float><<<grid, block, sizeof(int) * beam_width, stream>>>(finished, parent_ids_ptr, sequence_lengths,
        output_ids_ptr, *beam_hyps, vocab_size_padded, end_ids, local_batch_size, beam_width, max_seq_len);
}

template <typename T>
void OnlineBeamSearchLayer<T>::setup(size_t batch_size, SetupParams const& setupParams)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    BaseBeamSearchLayer<T>::setupBase(batch_size, setupParams);
    allocateBuffer(batch_size);

    mDiversityRate = setupParams.beam_search_diversity_rate.value_or(std::vector<float>(0.0f));
    mLengthPenalty = setupParams.length_penalty.value_or(std::vector<float>(0.0f));

    FillBuffers const fillBuffers{batch_size, stream_};

    fillBuffers(setupParams.beam_search_diversity_rate, 0.0f, mDiversityRate, diversity_rates_buf_);
    fillBuffers(setupParams.length_penalty, 0.0f, mLengthPenalty, length_penalties_buf_);
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void OnlineBeamSearchLayer<T>::invokeSoftMax(BeamSearchOutputParams& outputs, SoftmaxParams const& params)
{
    TLLM_LOG_TRACE("%s", __PRETTY_FUNCTION__);
    Tensor const& output_ids_ptr = outputs.output_ids_ptr;
    const auto batch_size = static_cast<std::int32_t>(output_ids_ptr.shape[0]);
    const auto beam_width = static_cast<std::int32_t>(output_ids_ptr.shape[1]);
    const auto max_seq_len = static_cast<std::int32_t>(output_ids_ptr.shape[2]);
    const int ite{params.ite};
    Tensor const& logits{params.logits};
    const auto local_batch_size = logits.shape[0];

    BeamHypotheses beamHypotheses;
    auto* const end_ids = params.end_ids.template getPtr<const int>();
    float* output_log_probs = (outputs.output_log_probs) ? outputs.output_log_probs->template getPtr<float>() : nullptr;
    auto* finished
        = reinterpret_cast<FinishedState*>(outputs.finished->template getPtr<FinishedState::UnderlyingType>());
    auto* sequence_lengths = outputs.sequence_length->template getPtr<int>();
    if (outputs.beamHypotheses)
    {
        beamHypotheses = *outputs.beamHypotheses;
        beamHypotheses.ite = ite;
        beamHypotheses.local_batch_size = local_batch_size;
        beamHypotheses.batch_size = batch_size;
        beamHypotheses.max_seq_len = max_seq_len;
        beamHypotheses.output_ids_src_ptr = output_ids_ptr.template getPtr<const int*>();
        beamHypotheses.parent_ids_src_ptr = outputs.parent_ids_ptr.template getPtr<const int*>();
        beamHypotheses.sequence_lengths_src = sequence_lengths;
        beamHypotheses.log_probs_src = output_log_probs;
        beamHypotheses.length_penalties = length_penalties_buf_;
        beamHypotheses.end_ids = end_ids;
    }

    invokeTopkSoftMax(logits.template getPtr<T>(), (const T*) (nullptr), finished, sequence_lengths,
        outputs.cum_log_probs->template getPtr<float>(), output_log_probs, output_ids_ptr.getPtr<int*>(),
        topk_softmax_workspace_, topk_softmax_workspace_size_, &beamHypotheses, local_batch_size, beam_width,
        vocab_size_padded_, end_ids, diversity_rates_buf_, length_penalties_buf_, stream_);
    sync_check_cuda_error();

    invokeUpdate(finished, outputs.parent_ids_ptr.template getPtr<int*>(), sequence_lengths,
        output_ids_ptr.getPtr<int*>(), &beamHypotheses, local_batch_size, beam_width, vocab_size_padded_, end_ids,
        max_seq_len, stream_);
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
        allocator_->reMalloc(topk_softmax_workspace_, sizeof(float) * topk_softmax_workspace_size_, true));
    diversity_rates_buf_ = allocator_->reMalloc(diversity_rates_buf_, sizeof(float) * batch_size, false);
    length_penalties_buf_ = allocator_->reMalloc(length_penalties_buf_, sizeof(float) * batch_size, false);

    is_allocate_buffer_ = true;
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void OnlineBeamSearchLayer<T>::freeBuffer()
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    if (is_allocate_buffer_)
    {
        allocator_->free((void**) (&topk_softmax_workspace_));
        allocator_->free((void**) (&diversity_rates_buf_));
        allocator_->free((void**) (&length_penalties_buf_));
        is_allocate_buffer_ = false;
    }
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
OnlineBeamSearchLayer<T>::OnlineBeamSearchLayer(size_t vocab_size, size_t vocab_size_padded, cudaStream_t stream,
    std::shared_ptr<IAllocator> allocator, bool is_free_buffer_after_forward)
    : BaseBeamSearchLayer<T>(vocab_size, vocab_size_padded, stream, std::move(allocator), is_free_buffer_after_forward)
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
