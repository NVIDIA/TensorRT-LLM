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

static __global__ void set_topp_runtime_args(int batchSize, uint32_t top_k, uint32_t* top_ks, int top_ks_size,
    float top_p, float* top_ps, int top_ps_size, bool* skip_decode, float* initial_top_p_buf, float* top_p_decay_buf,
    float* top_p_min_buf, const int* batch_slots)
{
    /**
     * @brief Setup the runtime arguments for topp, broadcasting top_p to top_ps
              and top_k to top_ks, verifying value ranges of top_p_decay/top_p_min.
     *
     * \param batchSize
     * \param top_k
     * \param top_ks                [batchSize]
     * \param top_ks_size
     * \param top_p
     * \param top_ps                [batchSize]
     * \param top_ps_size
     * \param skip_decode           [batchSize]
     * \param initial_top_p_buf     [batchSize]
     * \param top_p_decay_buf       [batchSize]
     * \param top_p_min_buf         [batchSize]
     *
     */

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    for (int bi = index; bi < batchSize; bi += gridDim.x * blockDim.x)
    {
        auto const batch_slot = batch_slots != nullptr ? batch_slots[bi] : bi;
        std::uint32_t k = top_ks_size > 1 ? top_ks[batch_slot] : top_k;
        float p = top_ps_size > 1 ? top_ps[batch_slot] : top_p;
        if (k == 0 && p == 0.0f)
        {
            // TensorRT-LLM's topp implementation does not support topp = 0.0f, but it
            // equivalent to greedy search. So, we set the topk = 1 as an alternative
            // solution.
            k = 1;
        }
        top_ks[batch_slot] = k;
        top_ps[batch_slot] = p;
        skip_decode[batch_slot] = k > 0;

        initial_top_p_buf[batch_slot] = top_ps[batch_slot];
    }
}

template <typename T>
void TopPSamplingLayer<T>::allocateBuffer(size_t batchSize)
{
    TLLM_LOG_TRACE(__PRETTY_FUNCTION__);
    if (is_deterministic_)
    {
        invokeTopPSampling<T>(nullptr, // workspace
            mSamplingWorkspaceSize, cub_temp_storage_size_,
            nullptr,                   // output_ids
            nullptr,                   // sequence_length
            nullptr,                   // finished_input_buffer
            nullptr,                   // finished_output_buffer
            nullptr,                   // cum_log_probs
            nullptr,                   // output_log_probs
            nullptr,                   // log_probs
            topp_id_vals_buf_, topp_offset_buf_, begin_topp_offset_buf_, mCurandStatesDevice, batchSize,
            mVocabSizePadded, nullptr, 0.f, mStream, mSkipDecodeDevice, nullptr);
    }
    else
    {
        invokeAirTopPSampling<T>(nullptr, mSamplingWorkspaceSize,
            nullptr, // output_ids
            nullptr, // sequence_length
            nullptr, // finished_input_buffer
            nullptr, // finished_output_buffer
            nullptr, // cum_log_probs
            nullptr, // output_log_probs
            nullptr, // log_probs)
            mCurandStatesDevice, batchSize, mVocabSizePadded, nullptr, 0.f, mStream, air_topp_block_num_,
            mSkipDecodeDevice, nullptr);
    }

    std::array<size_t, 11> deviceBufferSizes;
    deviceBufferSizes[0] = mSamplingWorkspaceSize;
    deviceBufferSizes[1] = sizeof(int32_t) * batchSize * mVocabSizePadded;
    deviceBufferSizes[2] = sizeof(int32_t) * (batchSize + 1);
    deviceBufferSizes[3] = sizeof(int32_t) * (batchSize + 1);
    deviceBufferSizes[4] = sizeof(uint32_t) * batchSize;
    deviceBufferSizes[5] = sizeof(float) * batchSize;
    deviceBufferSizes[6] = sizeof(float) * batchSize;
    deviceBufferSizes[7] = sizeof(float) * batchSize;
    deviceBufferSizes[8] = sizeof(float) * batchSize;
    deviceBufferSizes[9] = sizeof(int32_t) * batchSize;
    deviceBufferSizes[10] = *std::max_element(&deviceBufferSizes[4], &deviceBufferSizes[10]);

    mSamplingWorkspaceDevice = mAllocator->reMalloc(mSamplingWorkspaceDevice, deviceBufferSizes[0], true);
    topp_id_vals_buf_ = mAllocator->reMalloc(topp_id_vals_buf_, deviceBufferSizes[1], false);
    topp_offset_buf_ = mAllocator->reMalloc(topp_offset_buf_, deviceBufferSizes[2], false);
    begin_topp_offset_buf_ = mAllocator->reMalloc(begin_topp_offset_buf_, deviceBufferSizes[3], false);
    runtime_top_k_buf_ = mAllocator->reMalloc(runtime_top_k_buf_, deviceBufferSizes[4], false);
    runtime_top_p_buf_ = mAllocator->reMalloc(runtime_top_p_buf_, deviceBufferSizes[5], false);
    initial_top_p_buf_ = mAllocator->reMalloc(initial_top_p_buf_, deviceBufferSizes[6], false);
    top_p_decay_buf_ = mAllocator->reMalloc(top_p_decay_buf_, deviceBufferSizes[7], false);
    top_p_min_buf_ = mAllocator->reMalloc(top_p_min_buf_, deviceBufferSizes[8], false);
    top_p_reset_ids_buf_ = mAllocator->reMalloc(top_p_reset_ids_buf_, deviceBufferSizes[9], false);
    setup_workspace_buf_ = mAllocator->reMalloc(setup_workspace_buf_, deviceBufferSizes[10], false);

    auto const bytesAllocated = std::accumulate(deviceBufferSizes.begin(), deviceBufferSizes.end(), (size_t) 0);
    TLLM_LOG_DEBUG("topPSamplingLayer allocated %lu bytes on GPU", (size_t) bytesAllocated);

    mIsAllocateBuffer = true;
}

template <typename T>
void TopPSamplingLayer<T>::freeBuffer()
{
    TLLM_LOG_TRACE(__PRETTY_FUNCTION__);
    if (mIsAllocateBuffer)
    {
        mAllocator->free((void**) (&mSamplingWorkspaceDevice));
        mAllocator->free((void**) (&topp_id_vals_buf_));
        mAllocator->free((void**) (&topp_offset_buf_));
        mAllocator->free((void**) (&begin_topp_offset_buf_));
        mAllocator->free((void**) (&runtime_top_k_buf_));
        mAllocator->free((void**) (&runtime_top_p_buf_));
        mAllocator->free((void**) (&initial_top_p_buf_));
        mAllocator->free((void**) (&top_p_decay_buf_));
        mAllocator->free((void**) (&top_p_min_buf_));
        mAllocator->free((void**) (&top_p_reset_ids_buf_));
        mAllocator->free((void**) (&setup_workspace_buf_));
    }
    BaseSamplingLayer<T>::freeBuffer();
    mIsAllocateBuffer = false;
}

template <typename T>
void TopPSamplingLayer<T>::setup(size_t const batchSize, int const* batchSlots, SetupParams const& setupParams)
{
    TLLM_LOG_TRACE(__PRETTY_FUNCTION__);
    BaseSamplingLayer<T>::setupBase(batchSize, batchSlots, setupParams);

    uint32_t const defaultTopK = 0;
    auto runtimeTopK = setupParams.runtime_top_k.value_or(std::vector<uint32_t>{defaultTopK});
    auto runtimeTopP = setupParams.runtime_top_p.value_or(std::vector<float>{});

    size_t const runtimeTopKSize = runtimeTopK.size();
    size_t const runtimeTopPSize = runtimeTopP.size();

    float const defaultTopPDecay{1.0f};
    auto decayVec = setupParams.top_p_decay.value_or(std::vector<float>(batchSize, defaultTopPDecay));

    float const defaultTopPMin{1e-6f}; // prevent topp becoming 0.0
    auto topPMinVec = setupParams.top_p_min.value_or(std::vector<float>(batchSize, defaultTopPMin));

    int32_t const defaultTopPResetId{-1};
    auto topPResetIdsVec = setupParams.top_p_reset_ids.value_or(std::vector<int32_t>(batchSize, defaultTopPResetId));

    if (runtimeTopPSize == 0)
    {
        std::fill_n(mSkipDecodeHost, batchSize, true);
        return;
    }

    for (auto& topP : runtimeTopP)
    {
        if (topP < 0.f || topP > 1.0f)
        {
            TLLM_LOG_WARNING("TopP (%f) is out of range ([0.0, 1.0f]). Clip to closest number.", topP);
            topP = std::clamp(topP, 0.f, 1.f);
        }
    }

    for (auto& decay : decayVec)
    {
        if (decay <= 0.f || decay > 1.0f)
        {
            TLLM_LOG_WARNING("Decay (%f) is out of range ([0.0, 1.0f]). Change to 1.0.", decay);
            decay = 1.0f;
        }
    }

    for (auto& topPMin : topPMinVec)
    {
        if (topPMin <= 0.f || topPMin > 1.0f)
        {
            TLLM_LOG_WARNING("TopP min (%f) is out of range ([0.0, 1.0f]). Change to 0.5.", topPMin);
            topPMin = 0.5f;
        }
    }

    uint32_t const topK = runtimeTopK.at(0);
    float const topP = runtimeTopP.at(0);

    if (runtimeTopKSize > 1)
    {
        TLLM_CHECK_WITH_INFO(runtimeTopK.size() == batchSize,
            fmtstr("runtimeTopK.size() (%lu) == batchSize (%lu) is not satisfied!", runtimeTopK.size(), batchSize));
        cudaAutoCpy(reinterpret_cast<uint32_t*>(setup_workspace_buf_), runtimeTopK.data(), batchSize, mStream);
        invokeScatterDecodingParams(
            reinterpret_cast<uint32_t*>(setup_workspace_buf_), runtime_top_k_buf_, batchSlots, batchSize, mStream);
    }
    if (runtimeTopPSize > 1)
    {
        TLLM_CHECK_WITH_INFO(runtimeTopP.size() == batchSize,
            fmtstr("runtime_top_p.size() (%lu) == batchSize (%lu) is not satisfied!", runtimeTopP.size(), batchSize));
        cudaAutoCpy(reinterpret_cast<float*>(setup_workspace_buf_), runtimeTopP.data(), batchSize, mStream);
        invokeScatterDecodingParams(
            reinterpret_cast<float*>(setup_workspace_buf_), runtime_top_p_buf_, batchSlots, batchSize, mStream);
    }

    auto fillBuffers
        = [this, &batchSize, &batchSlots](std::string name, auto const& vector, auto deviceTmpBuffer, auto deviceBuffer)
    {
        TLLM_CHECK_WITH_INFO(vector.size() == batchSize,
            fmtstr("%s.size() (%lu) == batchSize (%lu) is not satisfied!", name.c_str(), vector.size(), batchSize));
        cudaAutoCpy(deviceTmpBuffer, vector.data(), batchSize, mStream);
        invokeScatterDecodingParams(deviceTmpBuffer, deviceBuffer, batchSlots, batchSize, mStream);
    };

    fillBuffers("top_p_decay", decayVec, reinterpret_cast<float*>(setup_workspace_buf_), top_p_decay_buf_);

    fillBuffers("top_p_min", topPMinVec, reinterpret_cast<float*>(setup_workspace_buf_), top_p_min_buf_);

    fillBuffers(
        "top_p_reset_ids", topPResetIdsVec, reinterpret_cast<int32_t*>(setup_workspace_buf_), top_p_reset_ids_buf_);

    dim3 block(std::min((int) batchSize, 256));
    dim3 grid(divUp((int) batchSize, (int) block.x));
    set_topp_runtime_args<<<grid, block, 0, mStream>>>(batchSize, topK, runtime_top_k_buf_, runtimeTopKSize, topP,
        runtime_top_p_buf_, runtimeTopPSize, mSkipDecodeDevice, initial_top_p_buf_, top_p_decay_buf_, top_p_min_buf_,
        batchSlots);
    sync_check_cuda_error();

    cudaAutoCpy(mSkipDecodeHost, mSkipDecodeDevice, mMaxBatchSize, mStream);

    std::vector<float> runtime_top_ps(mMaxBatchSize);
    cudaAutoCpy(runtime_top_ps.data(), runtime_top_p_buf_, mMaxBatchSize, mStream);
    // TODO(nkorobov): find maxTopP using batch slots
    mRuntimeMaxTopP = *std::max_element(std::begin(runtime_top_ps), std::end(runtime_top_ps));

    if (!is_deterministic_)
    {
        int smCnt = mCudaDeviceProp->multiProcessorCount;
        air_topp_block_num_ = calcAirTopPBlockNum<T, int, float>(batchSize, (int) mVocabSizePadded, smCnt);
    }
}

template <typename T>
void TopPSamplingLayer<T>::runSampling(DecodingOutputParams& outputs, DecodingParams const& inputs)
{
    TLLM_LOG_TRACE(__PRETTY_FUNCTION__);

    auto const batchSize = inputs.logits.shape[0];

    // in case of skip any, the logit value is already copied and processed.
    auto* logits = !mSkipAny ? inputs.logits.template getPtr<T>() : mRuntimeLogitsDevice;
    auto* endIds = inputs.end_ids.template getPtr<const int>();
    auto* batchSlots = inputs.batch_slots ? inputs.batch_slots->template getPtr<const int>() : nullptr;

    if (is_deterministic_)
    {
        invokeTopPInitialize(
            topp_id_vals_buf_, topp_offset_buf_, begin_topp_offset_buf_, batchSize, mVocabSizePadded, mStream);
        sync_check_cuda_error();
    }

    FinishedState* finishedInput = (inputs.finished)
        ? reinterpret_cast<FinishedState*>(inputs.finished->template getPtr<FinishedState::UnderlyingType>())
        : nullptr;
    FinishedState* finishedOutput = (outputs.finished)
        ? reinterpret_cast<FinishedState*>(outputs.finished->template getPtr<FinishedState::UnderlyingType>())
        : nullptr;
    invokeAddBiasSoftMax(logits, logits, (T*) (nullptr), endIds, finishedInput, batchSlots, batchSize, mVocabSize,
        mVocabSizePadded, mStream);
    sync_check_cuda_error();

    float* cumLogProbs = (outputs.cum_log_probs) ? outputs.cum_log_probs->template getPtr<float>() : nullptr;
    float* outputLogProbs = (outputs.output_log_probs) ? outputs.output_log_probs->template getPtr<float>() : nullptr;
    int* sequenceLength = (outputs.sequence_length) ? outputs.sequence_length->template getPtr<int>() : nullptr;

    if (is_deterministic_)
    {
        invokeBatchTopPSampling<T>(mSamplingWorkspaceDevice, mSamplingWorkspaceSize, cub_temp_storage_size_,
            outputs.output_ids_ptr.template getPtr<int*>(), sequenceLength, finishedInput, finishedOutput, cumLogProbs,
            outputLogProbs, logits, topp_id_vals_buf_, topp_offset_buf_, begin_topp_offset_buf_, mCurandStatesDevice,
            batchSize, mVocabSizePadded, endIds, mRuntimeMaxTopP, runtime_top_p_buf_, mStream, mSkipDecodeDevice,
            batchSlots);
        sync_check_cuda_error();
        invokeComputeToppDecay(runtime_top_p_buf_, initial_top_p_buf_,
            outputs.output_ids_ptr.template getPtr<const int*>(), top_p_decay_buf_, top_p_min_buf_,
            top_p_reset_ids_buf_, sequenceLength, batchSlots, batchSize, mStream);
        sync_check_cuda_error();
    }
    else
    {
        invokeBatchAirTopPSampling<T>(mSamplingWorkspaceDevice, mSamplingWorkspaceSize,
            outputs.output_ids_ptr.template getPtr<int*>(), sequenceLength, finishedInput, finishedOutput, cumLogProbs,
            outputLogProbs, logits, mCurandStatesDevice, batchSize, mVocabSizePadded, endIds, mRuntimeMaxTopP,
            runtime_top_p_buf_, mStream, air_topp_block_num_, mSkipDecodeDevice, batchSlots);
        sync_check_cuda_error();
    }
}

template <typename T>
TopPSamplingLayer<T>::TopPSamplingLayer(std::size_t maxBatchSize, std::size_t vocabSize, std::size_t vocabSizePadded,
    cudaStream_t stream, std::shared_ptr<IAllocator> allocator, cudaDeviceProp* prop, bool isDeterministic)
    : BaseSamplingLayer<T>(maxBatchSize, vocabSize, vocabSizePadded, stream, std::move(allocator), prop)
    , is_deterministic_(isDeterministic)
{
    allocateBuffer(mMaxBatchSize);
}

template <typename T>
TopPSamplingLayer<T>::TopPSamplingLayer(TopPSamplingLayer<T> const& top_p_sampling_layer)
    : BaseSamplingLayer<T>(top_p_sampling_layer)
{
    allocateBuffer(mMaxBatchSize);
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
