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
#include "tensorrt_llm/kernels/decodingCommon.h"
#include "tensorrt_llm/kernels/samplingTopKKernels.h"
#include "tensorrt_llm/kernels/samplingTopPKernels.h"
#include "tensorrt_llm/layers/topKSamplingLayer.h"
#include "tensorrt_llm/runtime/iTensor.h"

#include <algorithm>
#include <float.h>

using namespace tensorrt_llm::common;
using namespace tensorrt_llm::kernels;
using namespace tensorrt_llm::runtime;

namespace tensorrt_llm
{
namespace layers
{

template <uint32_t TOP_K_MAX>
__global__ void setup_topk_runtime_args(int batch_size, uint32_t top_k, uint32_t* top_ks, int top_ks_size, float top_p,
    float* top_ps, int top_ps_size, bool* skip_decode)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = index; i < batch_size; i += gridDim.x * blockDim.x)
    {
        uint32_t k = top_ks_size > 1 ? top_ks[i] : top_k;
        float p = top_ps_size > 1 ? top_ps[i] : top_p;
        if (k == 0 && p == 0.0f)
        {
            // TensorRT-LLM's topp implementation does not support topp = 0.0f, but it
            // equivalent to greedy search. So, we set the topk = 1 as an alternative
            // solution.
            k = 1;
        }
        if (k > 0 && p == 0.0f)
        {
            // for compatibility <= TensorRT-LLM5.0.
            // This case corresponds to the old topk sampling, which is equivalent to
            // the old topk_topp sampling with topp=1.0f. TopKSamplingLayer and
            // TopKTopPSamplingLayer are now merged by TopKSamplingLayer. Thus, we
            // replace the case topk>0 and topp=0.0f by topk>0 and topp=1.0f for the
            // compatibility.
            p = 1.0f;
        }
        // Clip k value. A topk sampling kernel supports up to TOP_K_MAX.
        top_ks[i] = k > TOP_K_MAX ? TOP_K_MAX : k;
        if (k > TOP_K_MAX)
        {
            printf(
                "[WARNING] topk (%d) is larger than max supported number (%d) for "
                "token %d"
                " clip to max supported number %d. \n",
                k, TOP_K_MAX, i, top_ks[i]);
        }
        // Clip p value if it is out of range. range = [0.0, 1.0].
        top_ps[i] = p < 0.0f ? 0.0f : (p > 1.0f ? 1.0f : p);
        if (p < 0.0f || p > 1.0f)
        {
            printf(
                "[WARNING] topp (%f) is out of range ([0.0, 1.0f]) for token %d"
                " clip to closest number %f.\n",
                p, i, top_ps[i]);
        }
        skip_decode[i] = k == 0;
    }
}

template <typename T>
void TopKSamplingLayer<T>::allocateBuffer(size_t const batch_size, std::vector<uint32_t> const& top_k)
{
    TLLM_LOG_TRACE(__PRETTY_FUNCTION__);
    uint32_t max_top_k = (top_k.size() > 0) ? *std::max_element(std::begin(top_k), std::end(top_k)) : 1;
    if (max_top_k == 0)
    {
        // for safety. TopKSamplingLayer handles a case of top_k=0 and top_p=0 as
        // a greedy decode, i.e. top_k=1, although such case has max_top_k=0.
        max_top_k = 1;
    }
    invokeTopKSampling<T>(nullptr, sampling_workspace_size_, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
        nullptr, nullptr, max_top_k, 1.0f, vocab_size_padded_, nullptr, stream_, batch_size, skip_decode_buf_,
        normalize_log_probs);
    sampling_workspace_ = allocator_->reMalloc(sampling_workspace_, sampling_workspace_size_, false);
    runtime_top_k_buf_ = allocator_->reMalloc(runtime_top_k_buf_, sizeof(uint32_t) * batch_size, false);
    runtime_top_p_buf_ = allocator_->reMalloc(runtime_top_p_buf_, sizeof(float) * batch_size, false);
    is_allocate_buffer_ = true;
}

template <typename T>
void TopKSamplingLayer<T>::freeBuffer()
{
    TLLM_LOG_TRACE(__PRETTY_FUNCTION__);
    if (is_allocate_buffer_)
    {
        allocator_->free((void**) (&sampling_workspace_));
        allocator_->free((void**) (&runtime_top_k_buf_));
        allocator_->free((void**) (&runtime_top_p_buf_));
    }
    BaseSamplingLayer<T>::freeBuffer();
    is_allocate_buffer_ = false;
}

template <typename T>
void TopKSamplingLayer<T>::setup(size_t const batch_size, SetupParams const& setupParams)
{
    TLLM_LOG_TRACE(__PRETTY_FUNCTION__);
    BaseSamplingLayer<T>::setupBase(batch_size, setupParams);

    uint32_t const default_top_k = 0;
    auto const runtime_top_k = setupParams.runtime_top_k.value_or(std::vector<uint32_t>{default_top_k});
    auto const runtime_top_p = setupParams.runtime_top_p.value_or(std::vector<float>{});

    allocateBuffer(batch_size, runtime_top_k);

    size_t const runtime_top_k_size = runtime_top_k.size();
    size_t const runtime_top_p_size = runtime_top_p.size();
    normalize_log_probs = setupParams.normalize_log_probs.has_value() && setupParams.normalize_log_probs.value();

    uint32_t const top_k = *std::max_element(std::begin(runtime_top_k), std::end(runtime_top_k));
    float const top_p = (runtime_top_p_size == 0) ? 0.0f : runtime_top_p.front();

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

    dim3 block(std::min((int) batch_size, 256));
    dim3 grid(divUp((int) batch_size, (int) block.x));
    // support top_k up to TOP_K_MAX.
    setup_topk_runtime_args<TOP_K_MAX><<<grid, block, 0, stream_>>>(batch_size, top_k, runtime_top_k_buf_,
        runtime_top_k_size, top_p, runtime_top_p_buf_, runtime_top_p_size, skip_decode_buf_);
    cudaAutoCpy(skip_decode_, skip_decode_buf_, batch_size, stream_);
    std::vector<uint32_t> runtime_top_ks(batch_size);
    cudaAutoCpy(runtime_top_ks.data(), runtime_top_k_buf_, batch_size, stream_);
    runtime_max_top_k_ = *std::max_element(std::begin(runtime_top_ks), std::end(runtime_top_ks));
}

template <typename T>
void TopKSamplingLayer<T>::runSampling(DecodingOutputParams& outputs, DecodingParams const& params)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    auto const batch_size = outputs.output_ids_ptr.shape[0];
    auto const local_batch_size = params.logits.shape[0];
    auto const ite = params.ite;

    // in case of skip any, the logit value is already copied and processed.
    auto* logits = !skip_any_ ? params.logits.template getPtr<T>() : runtime_logits_buf_;
    auto* end_ids = params.end_ids.template getPtr<const int>();

    FinishedState* finished_input = (params.finished)
        ? reinterpret_cast<FinishedState*>(params.finished->template getPtr<FinishedState::UnderlyingType>())
        : nullptr;
    FinishedState* finished_output = (outputs.finished)
        ? reinterpret_cast<FinishedState*>(outputs.finished->template getPtr<FinishedState::UnderlyingType>())
        : nullptr;
    invokeAddBiasEndMask(
        logits, (T*) (nullptr), end_ids, finished_input, local_batch_size, vocab_size_, vocab_size_padded_, stream_);
    sync_check_cuda_error();

    float* cum_log_probs = (outputs.cum_log_probs) ? outputs.cum_log_probs->template getPtr<float>() : nullptr;
    float* output_log_probs = (outputs.output_log_probs) ? outputs.output_log_probs->template getPtr<float>() : nullptr;

    if (cum_log_probs != nullptr || output_log_probs != nullptr)
    {
        invokeAddBiasSoftMax(logits, logits, (T*) (nullptr), end_ids, finished_input, local_batch_size, vocab_size_,
            vocab_size_padded_, stream_);
        sync_check_cuda_error();
    }

    int* sequence_length = (outputs.sequence_length) ? outputs.sequence_length->template getPtr<int>() : nullptr;

    invokeBatchTopKSampling(sampling_workspace_, sampling_workspace_size_, logits,
        outputs.output_ids_ptr.template getPtr<int*>(), sequence_length, finished_input, finished_output, cum_log_probs,
        output_log_probs, curandstate_buf_ + ite * local_batch_size,
        (int) runtime_max_top_k_, // useless because runtime_top_k_buf_ is never
                                  // nullptr. Keep for legacy.
        (int*) (runtime_top_k_buf_ + ite * local_batch_size),
        1.0f,                     // useless because runtime_top_p_buf_ is never nullptr. Keep for
                                  // legacy.
        runtime_top_p_buf_ + ite * local_batch_size, vocab_size_padded_, end_ids, stream_, local_batch_size,
        skip_decode_buf_ + ite * local_batch_size, normalize_log_probs);
    sync_check_cuda_error();
}

template <typename T>
TopKSamplingLayer<T>::TopKSamplingLayer(size_t vocab_size, size_t vocab_size_padded, cudaStream_t stream,
    std::shared_ptr<IAllocator> allocator, bool is_free_buffer_after_forward)
    : BaseSamplingLayer<T>(
        vocab_size, vocab_size_padded, stream, std::move(allocator), is_free_buffer_after_forward, nullptr)
{
}

template <typename T>
TopKSamplingLayer<T>::TopKSamplingLayer(TopKSamplingLayer<T> const& top_k_sampling_layer)
    : BaseSamplingLayer<T>(top_k_sampling_layer)
{
}

template <typename T>
TopKSamplingLayer<T>::~TopKSamplingLayer()
{
    TLLM_LOG_TRACE(__PRETTY_FUNCTION__);
    freeBuffer();
}

template class TopKSamplingLayer<float>;
template class TopKSamplingLayer<half>;

} // namespace layers
} // namespace tensorrt_llm
