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

#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/common/memoryUtils.h"
#include "tensorrt_llm/common/reduceKernelUtils.cuh"
#include "tensorrt_llm/kernels/decodingCommon.h"
#include "tensorrt_llm/kernels/samplingTopKKernels.h"
#include "tensorrt_llm/kernels/samplingTopPKernels.h"
#include "tensorrt_llm/layers/topPSamplingLayer.h"

#include <algorithm>
#include <float.h>

using namespace tensorrt_llm::common;
using namespace tensorrt_llm::kernels;

namespace tensorrt_llm
{
namespace layers
{

static __global__ void set_topp_runtime_args(int batch_size, std::uint32_t top_k, std::uint32_t* top_ks,
    int top_ks_size, float top_p, float* top_ps, int top_ps_size, bool* skip_decode, float* initial_top_p_buf,
    float* top_p_decay_buf, float* top_p_min_buf)
{
    /**
     * @brief Setup the runtime arguments for topp, broadcasting top_p to top_ps
              and top_k to top_ks, verifying value ranges of top_p_decay/top_p_min.
     *
     * \param batch_size
     * \param top_k
     * \param top_ks                [batch_size]
     * \param top_ks_size
     * \param top_p
     * \param top_ps                [batch_size]
     * \param top_ps_size
     * \param skip_decode           [batch_size]
     * \param initial_top_p_buf     [batch_size]
     * \param top_p_decay_buf       [batch_size]
     * \param top_p_min_buf         [batch_size]
     *
     */

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = index; i < batch_size; i += gridDim.x * blockDim.x)
    {
        std::uint32_t k = top_ks_size > 1 ? top_ks[i] : top_k;
        float p = top_ps_size > 1 ? top_ps[i] : top_p;
        if (k == 0 && p == 0.0f)
        {
            // TensorRT-LLM's topp implementation does not support topp = 0.0f, but it
            // equivalent to greedy search. So, we set the topk = 1 as an alternative
            // solution.
            k = 1;
        }
        top_ks[i] = k;
        // Clip p value if it is out of range. range = [0.0, 1.0].
        top_ps[i] = p < 0.0f ? 0.0f : (p > 1.0f ? 1.0f : p);
        if (p < 0.0f || p > 1.0f)
        {
            printf(
                "[WARNING] topp (%f) is out of range ([0.0, 1.0f]) for token %d"
                " clip to closest number %f.\n",
                p, i, top_ps[i]);
        }
        skip_decode[i] = k > 0;

        initial_top_p_buf[i] = top_ps[i];
        if (top_p_decay_buf[i] > 1.0f || top_p_decay_buf[i] <= 0.0f)
        {
            printf(
                "[WARNING] top_p_decay_buf (%f) is out of range ([0.0, 1.0f]) for "
                "token %d,"
                " change to 1.0f.\n",
                top_p_decay_buf[i], i);
            top_p_decay_buf[i] = 1.0f;
        }
        if (top_p_min_buf[i] > 1.0f || top_p_min_buf[i] <= 0.0f)
        {
            printf(
                "[WARNING] top_p_min_buf (%f) is out of range ([0.0, 1.0f]) for "
                "token %d,"
                " change to 0.5f.\n",
                top_p_min_buf[i], i);
            top_p_min_buf[i] = 0.5f;
        }
    }
}

template <typename T>
void TopPSamplingLayer<T>::allocateBuffer(std::size_t batch_size, std::vector<float> const& top_p)
{
    TLLM_LOG_TRACE(__PRETTY_FUNCTION__);
    float const max_top_p = (top_p.size() > 0) ? *std::max_element(std::begin(top_p), std::end(top_p)) : 0.0f;
    if (is_deterministic_)
    {
        invokeTopPSampling<T>(nullptr, // workspace
            sampling_workspace_size_, cub_temp_storage_size_,
            nullptr,                   // output_ids
            nullptr,                   // sequence_length
            nullptr,                   // finished_input_buffer
            nullptr,                   // finished_output_buffer
            nullptr,                   // cum_log_probs
            nullptr,                   // output_log_probs
            nullptr,                   // log_probs
            topp_id_vals_buf_, topp_offset_buf_, begin_topp_offset_buf_, curandstate_buf_, batch_size,
            vocab_size_padded_, nullptr, max_top_p, stream_, skip_decode_buf_);
    }
    else
    {
        invokeAirTopPSampling<T>(nullptr, sampling_workspace_size_,
            nullptr, // output_ids
            nullptr, // sequence_length
            nullptr, // finished_input_buffer
            nullptr, // finished_output_buffer
            nullptr, // cum_log_probs
            nullptr, // output_log_probs
            nullptr, // log_probs)
            curandstate_buf_, batch_size, vocab_size_padded_, nullptr, max_top_p, stream_, air_topp_block_num_,
            skip_decode_buf_);
    }

    sampling_workspace_ = allocator_->reMalloc(sampling_workspace_, sampling_workspace_size_, true);
    runtime_top_k_buf_ = allocator_->reMalloc(runtime_top_k_buf_, sizeof(std::uint32_t) * batch_size, false);
    runtime_top_p_buf_ = allocator_->reMalloc(runtime_top_p_buf_, sizeof(float) * batch_size, false);
    initial_top_p_buf_ = allocator_->reMalloc(initial_top_p_buf_, sizeof(float) * batch_size, false);
    top_p_decay_buf_ = allocator_->reMalloc(top_p_decay_buf_, sizeof(float) * batch_size, false);
    top_p_min_buf_ = allocator_->reMalloc(top_p_min_buf_, sizeof(float) * batch_size, false);
    top_p_reset_ids_buf_ = allocator_->reMalloc(top_p_reset_ids_buf_, sizeof(std::int32_t) * batch_size, false);
    topp_id_vals_buf_
        = allocator_->reMalloc(topp_id_vals_buf_, sizeof(std::int32_t) * batch_size * vocab_size_padded_, false);
    topp_offset_buf_ = allocator_->reMalloc(topp_offset_buf_, sizeof(std::int32_t) * (batch_size + 1), false);
    begin_topp_offset_buf_
        = allocator_->reMalloc(begin_topp_offset_buf_, sizeof(std::int32_t) * (batch_size + 1), false);
    is_allocate_buffer_ = true;
}

template <typename T>
void TopPSamplingLayer<T>::freeBuffer()
{
    TLLM_LOG_TRACE(__PRETTY_FUNCTION__);
    if (is_allocate_buffer_)
    {
        allocator_->free((void**) (&sampling_workspace_));
        allocator_->free((void**) (&topp_id_vals_buf_));
        allocator_->free((void**) (&topp_offset_buf_));
        allocator_->free((void**) (&begin_topp_offset_buf_));
        allocator_->free((void**) (&runtime_top_k_buf_));
        allocator_->free((void**) (&runtime_top_p_buf_));
        allocator_->free((void**) (&initial_top_p_buf_));
        allocator_->free((void**) (&top_p_decay_buf_));
        allocator_->free((void**) (&top_p_min_buf_));
        allocator_->free((void**) (&top_p_reset_ids_buf_));
    }
    BaseSamplingLayer<T>::freeBuffer();
    is_allocate_buffer_ = false;
}

template <typename T>
void TopPSamplingLayer<T>::setup(std::size_t const batch_size, SetupParams const& setupParams)
{
    TLLM_LOG_TRACE(__PRETTY_FUNCTION__);
    BaseSamplingLayer<T>::setupBase(batch_size, setupParams);

    std::uint32_t const default_top_k = 0;
    auto const runtime_top_k = setupParams.runtime_top_k.value_or(std::vector<uint32_t>{default_top_k});
    auto const runtime_top_p = setupParams.runtime_top_p.value_or(std::vector<float>{});

    allocateBuffer(batch_size, runtime_top_p);

    std::size_t const runtime_top_k_size = runtime_top_k.size();
    std::size_t const runtime_top_p_size = runtime_top_p.size();

    if (runtime_top_p_size == 0)
    {
        std::fill_n(skip_decode_, batch_size, true);
        return;
    }

    std::uint32_t const top_k = runtime_top_k.at(0);
    float const top_p = runtime_top_p.at(0);

    if (runtime_top_k_size > 1)
    {
        TLLM_CHECK_WITH_INFO(runtime_top_k.size() == batch_size,
            fmtstr(
                "runtime_top_k.size() (%lu) == batch_size (%lu) is not satisfied!", runtime_top_k.size(), batch_size));
        cudaAutoCpy(runtime_top_k_buf_, runtime_top_k.data(), batch_size, stream_);
    }
    if (runtime_top_p_size > 1)
    {
        TLLM_CHECK_WITH_INFO(runtime_top_p.size() == batch_size,
            fmtstr(
                "runtime_top_p.size() (%lu) == batch_size (%lu) is not satisfied!", runtime_top_p.size(), batch_size));
        cudaAutoCpy(runtime_top_p_buf_, runtime_top_p.data(), batch_size, stream_);
    }

    auto fillBuffers = [this, &batch_size](std::string name, auto const& vector, auto& deviceBuffer)
    {
        TLLM_CHECK_WITH_INFO(vector.size() == batch_size,
            fmtstr("%s.size() (%lu) == batch_size (%lu) is not satisfied!", name.c_str(), vector.size(), batch_size));
        cudaAutoCpy(deviceBuffer, vector.data(), batch_size, stream_);
    };

    float const defaultTopPDecay{1.0f};
    fillBuffers("top_p_decay", setupParams.top_p_decay.value_or(std::vector<float>(batch_size, defaultTopPDecay)),
        top_p_decay_buf_);

    float const defaultTopPMin{1e-6f}; // prevent topp becoming 0.0
    fillBuffers(
        "top_p_min", setupParams.top_p_min.value_or(std::vector<float>(batch_size, defaultTopPMin)), top_p_min_buf_);

    std::int32_t const defaultTopPResetId{-1};
    fillBuffers("top_p_reset_ids",
        setupParams.top_p_reset_ids.value_or(std::vector<std::int32_t>(batch_size, defaultTopPResetId)),
        top_p_reset_ids_buf_);

    dim3 block(std::min((int) batch_size, 256));
    dim3 grid(divUp((int) batch_size, (int) block.x));
    set_topp_runtime_args<<<grid, block, 0, stream_>>>(batch_size, top_k, runtime_top_k_buf_, runtime_top_k_size, top_p,
        runtime_top_p_buf_, runtime_top_p_size, skip_decode_buf_, initial_top_p_buf_, top_p_decay_buf_, top_p_min_buf_);
    sync_check_cuda_error();

    cudaAutoCpy(skip_decode_, skip_decode_buf_, batch_size, stream_);

    std::vector<float> runtime_top_ps(batch_size);
    cudaAutoCpy(runtime_top_ps.data(), runtime_top_p_buf_, batch_size, stream_);
    runtime_max_top_p_ = *std::max_element(std::begin(runtime_top_ps), std::end(runtime_top_ps));

    if (!is_deterministic_)
    {
        int sm_cnt = cuda_device_prop_->multiProcessorCount;
        air_topp_block_num_ = calcAirTopPBlockNum<T, int, float>(batch_size, (int) vocab_size_padded_, sm_cnt);
    }
}

template <typename T>
void TopPSamplingLayer<T>::runSampling(DecodingOutputParams& outputs, DecodingParams const& params)
{
    TLLM_LOG_TRACE(__PRETTY_FUNCTION__);

    auto const batch_size = outputs.output_ids_ptr.shape[0];
    auto const local_batch_size = params.logits.shape[0];
    auto const ite = params.ite;

    // in case of skip any, the logit value is already copied and processed.
    auto* logits = !skip_any_ ? params.logits.template getPtr<T>() : runtime_logits_buf_;
    auto* end_ids = params.end_ids.template getPtr<const int>();

    if (is_deterministic_)
    {
        invokeTopPInitialize(
            topp_id_vals_buf_, topp_offset_buf_, begin_topp_offset_buf_, local_batch_size, vocab_size_padded_, stream_);
        sync_check_cuda_error();
    }

    FinishedState* finished_input = (params.finished)
        ? reinterpret_cast<FinishedState*>(params.finished->template getPtr<FinishedState::UnderlyingType>())
        : nullptr;
    FinishedState* finished_output = (outputs.finished)
        ? reinterpret_cast<FinishedState*>(outputs.finished->template getPtr<FinishedState::UnderlyingType>())
        : nullptr;
    invokeAddBiasSoftMax(logits, logits, (T*) (nullptr), end_ids, finished_input, local_batch_size, vocab_size_,
        vocab_size_padded_, stream_);
    sync_check_cuda_error();

    float* cum_log_probs = (outputs.cum_log_probs) ? outputs.cum_log_probs->template getPtr<float>() : nullptr;
    float* output_log_probs = (outputs.output_log_probs) ? outputs.output_log_probs->template getPtr<float>() : nullptr;
    int* sequence_length = (outputs.sequence_length) ? outputs.sequence_length->template getPtr<int>() : nullptr;

    if (is_deterministic_)
    {
        invokeBatchTopPSampling<T>(sampling_workspace_, sampling_workspace_size_, cub_temp_storage_size_,
            outputs.output_ids_ptr.template getPtr<int*>(), sequence_length, finished_input, finished_output,
            cum_log_probs, output_log_probs, logits, topp_id_vals_buf_, topp_offset_buf_, begin_topp_offset_buf_,
            curandstate_buf_ + ite * local_batch_size, local_batch_size, vocab_size_padded_, end_ids,
            runtime_max_top_p_, runtime_top_p_buf_ + ite * local_batch_size, stream_,
            skip_decode_buf_ + ite * local_batch_size);
        sync_check_cuda_error();
        invokeComputeToppDecay(runtime_top_p_buf_ + ite * local_batch_size, initial_top_p_buf_ + ite * local_batch_size,
            outputs.output_ids_ptr.template getPtr<const int*>(), top_p_decay_buf_ + ite * local_batch_size,
            top_p_min_buf_ + ite * local_batch_size, top_p_reset_ids_buf_ + ite * local_batch_size, sequence_length,
            local_batch_size, stream_);
        sync_check_cuda_error();
    }
    else
    {
        invokeBatchAirTopPSampling<T>(sampling_workspace_, sampling_workspace_size_,
            outputs.output_ids_ptr.template getPtr<int*>(), sequence_length, finished_input, finished_output,
            cum_log_probs, output_log_probs, logits, curandstate_buf_ + ite * local_batch_size, local_batch_size,
            vocab_size_padded_, end_ids, runtime_max_top_p_, runtime_top_p_buf_ + ite * local_batch_size, stream_,
            air_topp_block_num_, skip_decode_buf_ + ite * local_batch_size);
        sync_check_cuda_error();
    }
}

template <typename T>
TopPSamplingLayer<T>::TopPSamplingLayer(std::size_t vocab_size, std::size_t vocab_size_padded, cudaStream_t stream,
    std::shared_ptr<IAllocator> allocator, bool is_free_buffer_after_forward, cudaDeviceProp* cuda_device_prop,
    bool is_deterministic)
    : BaseSamplingLayer<T>(
        vocab_size, vocab_size_padded, stream, std::move(allocator), is_free_buffer_after_forward, cuda_device_prop)
    , is_deterministic_(is_deterministic)
{
}

template <typename T>
TopPSamplingLayer<T>::TopPSamplingLayer(TopPSamplingLayer<T> const& top_p_sampling_layer)
    : BaseSamplingLayer<T>(top_p_sampling_layer)
{
}

template <typename T>
TopPSamplingLayer<T>::~TopPSamplingLayer()
{
    TLLM_LOG_TRACE(__PRETTY_FUNCTION__);
    freeBuffer();
}

template class TopPSamplingLayer<float>;
template class TopPSamplingLayer<half>;

} // namespace layers
} // namespace tensorrt_llm
