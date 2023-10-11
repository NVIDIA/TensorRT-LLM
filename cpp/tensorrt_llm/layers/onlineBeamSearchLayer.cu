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

#include "tensorrt_llm/kernels/beamSearchTopkKernels.h"
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
__global__ void update_kernel(bool* finished, int** parent_ids_ptr, int* sequence_lengths, int** output_ids_ptr,
    BeamHypotheses beam_hyps, const int vocab_size, const int* end_ids, const int local_batch_size,
    const int beam_width, const int max_seq_len)
{
    extern __shared__ char s_buf[]; // intermediate result
    int* s_sequence_lengths = (int*) (s_buf);

    for (int beam_idx = threadIdx.x; beam_idx < beam_width; beam_idx += blockDim.x)
    {
        s_sequence_lengths[beam_idx] = sequence_lengths[blockIdx.x * beam_width + beam_idx];
    }
    __syncthreads();

    for (int beam_idx = threadIdx.x; beam_idx < beam_width; beam_idx += blockDim.x)
    {
        const int current_step{s_sequence_lengths[beam_idx]};

        if (!finished[blockIdx.x * beam_width + beam_idx])
        {
            s_sequence_lengths[beam_idx]++;
        }

        int new_word_id{output_ids_ptr[blockIdx.x][beam_idx * max_seq_len + current_step]};
        int new_beam_id{(new_word_id / vocab_size) % beam_width};
        new_word_id = new_word_id % vocab_size;

        sequence_lengths[blockIdx.x * beam_width + beam_idx] = s_sequence_lengths[new_beam_id];
        finished[beam_idx] = new_word_id == end_ids[blockIdx.x];
        parent_ids_ptr[blockIdx.x][beam_idx * max_seq_len + current_step] = new_beam_id;
        output_ids_ptr[blockIdx.x][beam_idx * max_seq_len + current_step] = new_word_id;
    }
    if (beam_hyps.num_beams != nullptr)
    {
        if (beam_hyps.num_beams[beam_hyps.ite * beam_hyps.local_batch_size + blockIdx.x] == beam_width)
        {
            for (int beam_idx = threadIdx.x; beam_idx < beam_width; beam_idx += blockDim.x)
            {
                finished[blockIdx.x * beam_width + beam_idx] = true;
            }
        }
    }
}

void invokeUpdate(bool* finished, int** parent_ids_ptr, int* sequence_lengths, int** output_ids_ptr,
    BeamHypotheses* beam_hyps, const int local_batch_size, const int beam_width, const int vocab_size_padded,
    const int* end_ids, const int max_seq_len, cudaStream_t stream)
{
    dim3 grid(local_batch_size);
    dim3 block(min(beam_width, 1024));

    update_kernel<float><<<grid, block, sizeof(int) * beam_width, stream>>>(finished, parent_ids_ptr, sequence_lengths,
        output_ids_ptr, *beam_hyps, vocab_size_padded, end_ids, local_batch_size, beam_width, max_seq_len);
}

template <typename T>
void OnlineBeamSearchLayer<T>::setup(SetupParams const& setupParams)
{
    TLLM_LOG_DEBUG(__PRETTY_FUNCTION__);
    BaseBeamSearchLayer<T>::setupBase(setupParams);

    mDiversityRate = setupParams.beam_search_diversity_rate.value_or(0.0f);
    mLengthPenalty = setupParams.length_penalty.value_or(0.0f);
}

template <typename T>
void OnlineBeamSearchLayer<T>::invokeSoftMax(BeamSearchOutputParams& outputs, SoftmaxParams const& params)
{
    TLLM_LOG_DEBUG("%s", __PRETTY_FUNCTION__);
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
    auto* finished = outputs.finished->template getPtr<bool>();
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
        beamHypotheses.length_penalty = mLengthPenalty;
        beamHypotheses.end_ids = end_ids;
    }

    output_log_probs = (outputs.output_log_probs) ? outputs.output_log_probs->template getPtr<float>() : nullptr;
    invokeTopkSoftMax(logits.template getPtr<T>(), (const T*) (nullptr), finished, sequence_lengths,
        outputs.cum_log_probs->template getPtr<float>(), output_log_probs, output_ids_ptr.getPtr<int*>(),
        topk_softmax_workspace_, topk_softmax_workspace_size_, &beamHypotheses, local_batch_size, beam_width,
        vocab_size_padded_, end_ids, mDiversityRate, mLengthPenalty, stream_);
    sync_check_cuda_error();

    invokeUpdate(finished, outputs.parent_ids_ptr.template getPtr<int*>(), sequence_lengths,
        output_ids_ptr.getPtr<int*>(), &beamHypotheses, local_batch_size, beam_width, vocab_size_padded_, end_ids,
        max_seq_len, stream_);
    sync_check_cuda_error();
}

template <typename T>
void OnlineBeamSearchLayer<T>::allocateBuffer(size_t batch_size, size_t beam_width)
{
    TLLM_LOG_DEBUG(__PRETTY_FUNCTION__);
    // we need to check 2 * beam_width candidates each time
    // 64 is the max beam width we support now.
    topk_softmax_workspace_size_ = (size_t) (ceil(batch_size * 64 * (64 * 2) / 4.) * 4 * 2
        + ceil(batch_size * (64 * 2) * SMALL_TOP_K_SOFTMAX_MAX_VOC_PARTS * (2 * (MAX_K * 2) + 2) / 4.) * 4);

    topk_softmax_workspace_ = reinterpret_cast<float*>(
        allocator_->reMalloc(topk_softmax_workspace_, sizeof(float) * topk_softmax_workspace_size_, true));
    is_allocate_buffer_ = true;
}

template <typename T>
OnlineBeamSearchLayer<T>::OnlineBeamSearchLayer(size_t vocab_size, size_t vocab_size_padded, cudaStream_t stream,
    IAllocator* allocator, bool is_free_buffer_after_forward)
    : BaseBeamSearchLayer<T>(vocab_size, vocab_size_padded, stream, allocator, is_free_buffer_after_forward)
{
}

template <typename T>
OnlineBeamSearchLayer<T>::OnlineBeamSearchLayer(OnlineBeamSearchLayer<T> const& beam_search_layer)
    : BaseBeamSearchLayer<T>(beam_search_layer)
{
    TLLM_LOG_DEBUG(__PRETTY_FUNCTION__);
}

template <typename T>
OnlineBeamSearchLayer<T>::~OnlineBeamSearchLayer()
{
    TLLM_LOG_DEBUG(__PRETTY_FUNCTION__);
}

template class OnlineBeamSearchLayer<float>;
template class OnlineBeamSearchLayer<half>;

} // namespace layers
} // namespace tensorrt_llm
