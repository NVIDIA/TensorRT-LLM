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

#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/memoryUtils.h"
#include "tensorrt_llm/kernels/penaltyKernels.h"
#include "tensorrt_llm/layers/baseBeamSearchLayer.h"
#include "tensorrt_llm/layers/fillBuffers.h"

#include <algorithm>

using namespace tensorrt_llm::common;
using namespace tensorrt_llm::kernels;

namespace tensorrt_llm
{
namespace layers
{

__global__ void update_indir_cache_kernel(int* tgt_indir_cache, const int* src_indir_cache, const int** parent_ids,
    const FinishedState* finished, const int* sequence_lengths, const int* input_lengths, int batch_dim,
    int local_batch_size, int beam_width, int max_attention_window, int sink_token_length, int max_seq_len)
{
    int time_step = threadIdx.x + blockIdx.x * blockDim.x;
    int bb_id = threadIdx.y + blockIdx.y * blockDim.y;   // should be just blockIdx.y?
    const int current_step{sequence_lengths[bb_id] - 1}; // the sequence_lengths is updated, need to minus 1
    const int input_length{input_lengths == nullptr ? 0 : input_lengths[bb_id]};
    const int batch_id = bb_id / beam_width;
    const int beam_id = bb_id % beam_width;
    // Exit when the batch_beam or timestep is out of the bound.
    // Assume that KV Cache is shared and fixed for context part,
    //  so we don't need to update the indices for context part.
    if (bb_id >= beam_width * local_batch_size || time_step >= max_seq_len || time_step < input_length
        || time_step < (max_seq_len - max_attention_window) || finished[bb_id].isFinished())
    {
        return;
    }
    int time_step_circ = time_step;
    if (time_step_circ >= sink_token_length)
    {
        time_step_circ
            = sink_token_length + (time_step - sink_token_length) % (max_attention_window - sink_token_length);
    }

    // for the parent_ids, we will still keep it for all past tokens (i.e. max_seq_len)
    const int src_beam = parent_ids[batch_id][beam_id * max_seq_len + current_step];

    // for the indir tables, we have the cyclic kv cache.
    const uint32_t tgt_offset
        = batch_id * beam_width * max_attention_window + beam_id * max_attention_window + time_step_circ;
    const uint32_t src_offset
        = batch_id * beam_width * max_attention_window + src_beam * max_attention_window + time_step_circ;

    tgt_indir_cache[tgt_offset] = (time_step == current_step) ? beam_id : src_indir_cache[src_offset];
}

void update_indir_cache_kernelLauncher(int* tgt_indir_cache, const int* src_indir_cache, const int** parent_ids,
    const FinishedState* finished, const int* sequence_lengths, const int* input_lengths, int batch_dim,
    int local_batch_size, int beam_width, int max_seq_len, int max_attention_window, int sink_token_length,
    cudaStream_t stream)
{
    const dim3 block(32);
    // Update indirections steps [input_length[bb_id], sequence_lengths[bb_id]], included
    const dim3 grid((max_seq_len + block.x - 1) / block.x, local_batch_size * beam_width);
    update_indir_cache_kernel<<<grid, block, 0, stream>>>(tgt_indir_cache, src_indir_cache, parent_ids, finished,
        sequence_lengths, input_lengths, batch_dim, local_batch_size, beam_width, max_attention_window,
        sink_token_length, max_seq_len);
}

template <typename T>
BaseBeamSearchLayer<T>::BaseBeamSearchLayer(size_t vocab_size, size_t vocab_size_padded, cudaStream_t stream,
    std::shared_ptr<IAllocator> allocator, bool is_free_buffer_after_forward)
    : BaseLayer(stream, std::move(allocator), is_free_buffer_after_forward, nullptr)
    , vocab_size_(vocab_size)
    , vocab_size_padded_(vocab_size_padded)
{
}

template <typename T>
BaseBeamSearchLayer<T>::BaseBeamSearchLayer(BaseBeamSearchLayer<T> const& beam_search_layer)
    : BaseLayer(beam_search_layer)
    , vocab_size_(beam_search_layer.vocab_size_)
    , vocab_size_padded_(beam_search_layer.vocab_size_padded_)
    , topk_softmax_workspace_size_(beam_search_layer.topk_softmax_workspace_size_)
{
}

template <typename T>
BaseBeamSearchLayer<T>::~BaseBeamSearchLayer()
{
    TLLM_LOG_TRACE(__PRETTY_FUNCTION__);
    freeBuffer();
}

template <typename T>
void BaseBeamSearchLayer<T>::freeBuffer()
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    if (is_allocate_buffer_)
    {
        allocator_->free((void**) (&temperature_buf_));
        allocator_->free((void**) (&min_lengths_buf_));
        allocator_->free((void**) (&repetition_penalty_buf_));
        allocator_->free((void**) (&presence_penalty_buf_));
        allocator_->free((void**) (&frequency_penalty_buf_));
        is_allocate_buffer_ = false;
    }
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void BaseBeamSearchLayer<T>::allocateBuffer(size_t batch_size)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    temperature_buf_ = allocator_->reMalloc(temperature_buf_, sizeof(float) * batch_size, false);
    min_lengths_buf_ = allocator_->reMalloc(min_lengths_buf_, sizeof(int) * batch_size, false);
    repetition_penalty_buf_ = allocator_->reMalloc(repetition_penalty_buf_, sizeof(float) * batch_size, false);
    presence_penalty_buf_ = allocator_->reMalloc(presence_penalty_buf_, sizeof(float) * batch_size, false);
    frequency_penalty_buf_ = allocator_->reMalloc(frequency_penalty_buf_, sizeof(float) * batch_size, false);

    is_allocate_buffer_ = true;
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void BaseBeamSearchLayer<T>::setupBase(size_t batch_size, SetupParams const& setupParams)
{
    allocateBuffer(batch_size);
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
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
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void BaseBeamSearchLayer<T>::forward(BeamSearchOutputParams& outputs, ForwardParams const& params,
    int* penalty_workspace, const int* penalty_workspace_prev)
{
    TLLM_LOG_TRACE("%s", __PRETTY_FUNCTION__);
    Tensor& output_ids_ptr = outputs.output_ids_ptr;

    const auto batch_size = static_cast<std::int32_t>(output_ids_ptr.shape[0]);
    const auto beam_width = static_cast<std::int32_t>(output_ids_ptr.shape[1]);
    const auto max_seq_len = static_cast<std::int32_t>(output_ids_ptr.shape[2]);

    TLLM_CHECK_WITH_INFO(params.ite == 0, "Pipeline Parallelism is not supported yet !");

    const int ite = params.ite;
    auto* const input_lengths = params.input_lengths ? params.input_lengths->template getPtr<const int>() : nullptr;
    int* sequence_length = (outputs.sequence_length) ? outputs.sequence_length->template getPtr<int>() : nullptr;
    Tensor const& logits = params.logits;
    const auto local_batch_size = logits.shape[0];

#define ALL_OF(p_, sz_, dt_, v_) (std::all_of(p_, p_ + sz_, [&](dt_ b) { return b == v_; }))

    const T* embedding_bias = params.embedding_bias ? params.embedding_bias->template getPtr<const T>() : nullptr;
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

    InvokeBatchApplyPenaltyParams<T> penalty_params{logits.getPtr<T>(), embedding_bias,
        penalty_workspace + ite * local_batch_size * beam_width * vocab_size_,
        penalty_workspace_prev + ite * local_batch_size * beam_width * vocab_size_, temperatures, repetition_penalties,
        presence_penalties, frequency_penalties,
        (use_repetition_penalty_ || use_presence_penalty_ || use_frequency_penalty_), local_batch_size, beam_width,
        max_seq_len, vocab_size_, vocab_size_padded_, output_ids_ptr.template getPtr<const int*>(),
        outputs.parent_ids_ptr.template getPtr<const int*>(), input_lengths, sequence_length, min_lengths,
        params.end_ids.template getPtr<const int>(), stream_};
    invokeBatchApplyPenalty(penalty_params);
    sync_check_cuda_error();

    invokeSoftMax(outputs, params);

    if (beam_width > 1)
    {
        update_indir_cache_kernelLauncher(outputs.tgt_cache_indirection.template getPtr<int>(),
            params.src_cache_indirection.template getPtr<const int>(),
            outputs.parent_ids_ptr.template getPtr<const int*>(),
            reinterpret_cast<const FinishedState*>(
                outputs.finished->template getPtr<const FinishedState::UnderlyingType>()),
            sequence_length, input_lengths, batch_size, local_batch_size, beam_width, max_seq_len,
            params.max_attention_window, params.sink_token_length, stream_);
        sync_check_cuda_error();
    }
    sync_check_cuda_error();
    if (is_free_buffer_after_forward_)
    {
        freeBuffer();
    }
    sync_check_cuda_error();
}

template class BaseBeamSearchLayer<float>;
template class BaseBeamSearchLayer<half>;

} // namespace layers
} // namespace tensorrt_llm
