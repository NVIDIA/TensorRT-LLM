/*
 * Copyright (c) 2019-2024, NVIDIA CORPORATION.  All rights reserved.
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
#include "tensorrt_llm/layers/defaultDecodingParams.h"
#include "tensorrt_llm/layers/layerUtils.h"
#include "tensorrt_llm/layers/topKSamplingLayer.h"

#include <algorithm>
#include <float.h>

using namespace tensorrt_llm::common;
using namespace tensorrt_llm::kernels;
using namespace tensorrt_llm::runtime;

namespace tensorrt_llm::layers
{

template <int32_t TOP_K_MAX>
__global__ void setupTopKRuntimeArgs(SizeType32 batchSize, SizeType32 topK, SizeType32* topKs, SizeType32 topKsSize,
    float topP, float* topPs, SizeType32 topPsSize, bool* skipDecode, SizeType32 const* batchSlots)
{
    auto const index = static_cast<SizeType32>(blockIdx.x * blockDim.x + threadIdx.x);
    for (auto bi = index; bi < batchSize; bi += static_cast<SizeType32>(gridDim.x * blockDim.x))
    {
        auto const batchSlot = batchSlots != nullptr ? batchSlots[bi] : bi;
        auto k = topKsSize > 1 ? topKs[batchSlot] : topK;
        auto p = topPsSize > 1 ? topPs[batchSlot] : topP;
        if (k == 0 && p == 0.0f)
        {
            // TensorRT-LLM's topp implementation does not support topp = 0.0f, but it
            // equivalent to greedy search. So, we set the topk = 1 as an alternative
            // solution.
            k = 1;
        }
        if (k > 0 && p == 0.0f)
        {
            // This case corresponds to the old topk sampling, which is equivalent to
            // the old topk_topp sampling with topp=1.0f. TopKSamplingLayer and
            // TopKTopPSamplingLayer are now merged by TopKSamplingLayer. Thus, we
            // replace the case topk>0 and topp=0.0f by topk>0 and topp=1.0f for the
            // compatibility.
            p = 1.0f;
        }
        // Clip k value. A topk sampling kernel supports up to TOP_K_MAX.
        topKs[batchSlot] = k;
        // Clip p value if it is out of range. range = [0.0, 1.0].
        topPs[batchSlot] = p;
        skipDecode[batchSlot] = k == 0;
    }
}

template <typename T>
TopKSamplingLayer<T>::TopKSamplingLayer(
    DecoderDomain const& decoderDomain, cudaStream_t stream, std::shared_ptr<IAllocator> allocator)
    : BaseLayer(decoderDomain, stream, std::move(allocator))
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    allocateBuffer(mDecoderDomain.getBatchSize());

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
TopKSamplingLayer<T>::~TopKSamplingLayer()
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    freeBuffer();

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void TopKSamplingLayer<T>::allocateBuffer(SizeType32 const batchSize)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    mWorkspaceSize = getTopKWorkspaceSize<T>(batchSize, 1, TOP_K_MAX, mDecoderDomain.getVocabSizePadded());

    std::array<size_t, 4> deviceBufferSizes;
    deviceBufferSizes[0] = sizeof(SizeType32) * batchSize;
    deviceBufferSizes[1] = sizeof(float) * batchSize;
    deviceBufferSizes[2] = sizeof(bool) * batchSize;
    deviceBufferSizes[3] = std::max(deviceBufferSizes[0], deviceBufferSizes[1]);

    mRuntimeTopKDevice = mAllocator->reMalloc(mRuntimeTopKDevice, deviceBufferSizes[0], false);
    mRuntimeTopPDevice = mAllocator->reMalloc(mRuntimeTopPDevice, deviceBufferSizes[1], false);
    mSkipDecodeDevice = mAllocator->reMalloc(mSkipDecodeDevice, deviceBufferSizes[2], false);
    mSetupWorkspaceDevice = mAllocator->reMalloc(mSetupWorkspaceDevice, deviceBufferSizes[3], false);

    mSkipDecodeHost = static_cast<bool*>(std::realloc(mSkipDecodeHost, sizeof(bool) * batchSize));

    mAllocatedSize = std::accumulate(deviceBufferSizes.begin(), deviceBufferSizes.end(), size_t{0});
    TLLM_LOG_DEBUG("topKSamplingLayer allocated %lu bytes on GPU", mAllocatedSize);

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void TopKSamplingLayer<T>::freeBuffer()
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    mAllocator->free((void**) (&mRuntimeTopKDevice));
    mAllocator->free((void**) (&mRuntimeTopPDevice));
    mAllocator->free((void**) (&mSkipDecodeDevice));
    mAllocator->free((void**) (&mSetupWorkspaceDevice));
    std::free(mSkipDecodeHost);

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void TopKSamplingLayer<T>::setup(SizeType32 batchSize, SizeType32 beamWidth, SizeType32 const* batchSlots,
    std::shared_ptr<BaseSetupParams> const& baseSetupParams)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    auto setupParams = std::dynamic_pointer_cast<SamplingSetupParams>(baseSetupParams);

    auto const defaultTopK = DefaultDecodingParams::getTopK();
    auto runtimeTopK = setupParams->runtimeTopK.value_or(std::vector<SizeType32>(batchSize, defaultTopK));
    auto runtimeTopP = setupParams->runtimeTopP.value_or(std::vector<float>{});

    auto const runtimeTopKSize = runtimeTopK.size();
    auto const runtimeTopPSize = runtimeTopP.size();
    mNormalizeLogProbs = setupParams->normalizeLogProbs.has_value() && setupParams->normalizeLogProbs.value();

    for (auto& topP : runtimeTopP)
    {
        if (topP < 0.f || topP > 1.0f)
        {
            TLLM_LOG_WARNING("TopP (%f) is out of range ([0.0, 1.0f]). Clip to closest number.", topP);
            topP = std::clamp(topP, 0.f, 1.f);
        }
    }
    for (auto& topK : runtimeTopK)
    {
        if (topK < 0 || topK > TOP_K_MAX)
        {
            TLLM_LOG_WARNING(
                "TopK (%d) is larger than max supported number (%d). Clip to max supported number.", topK, TOP_K_MAX);
            topK = std::clamp(topK, 0, static_cast<SizeType32>(TOP_K_MAX));
        }
    }

    auto const topK = *std::max_element(std::begin(runtimeTopK), std::end(runtimeTopK));
    auto const topP = (runtimeTopPSize == 0) ? DefaultDecodingParams::getTopP() : runtimeTopP.front();

    if (runtimeTopKSize > 1)
    {
        TLLM_CHECK_WITH_INFO(runtimeTopK.size() == batchSize,
            fmtstr("runtimeTopK.size() (%lu) == batchSize (%d) is not satisfied!", runtimeTopK.size(), batchSize));
        cudaAutoCpy(
            reinterpret_cast<runtime::SizeType32*>(mSetupWorkspaceDevice), runtimeTopK.data(), batchSize, mStream);
        invokeScatterDecodingParams(reinterpret_cast<runtime::SizeType32*>(mSetupWorkspaceDevice), mRuntimeTopKDevice,
            batchSlots, batchSize, mStream);
    }
    if (runtimeTopPSize > 1)
    {
        TLLM_CHECK_WITH_INFO(runtimeTopP.size() == batchSize,
            fmtstr("runtimeTopP.size() (%lu) == batchSize (%d) is not satisfied!", runtimeTopP.size(), batchSize));
        cudaAutoCpy(reinterpret_cast<float*>(mSetupWorkspaceDevice), runtimeTopP.data(), batchSize, mStream);
        invokeScatterDecodingParams(
            reinterpret_cast<float*>(mSetupWorkspaceDevice), mRuntimeTopPDevice, batchSlots, batchSize, mStream);
    }

    {
        dim3 block(std::min(static_cast<uint32_t>(batchSize), 256u));
        dim3 grid(divUp(static_cast<uint32_t>(batchSize), block.x));
        // support topK up to TOP_K_MAX.
        setupTopKRuntimeArgs<TOP_K_MAX><<<grid, block, 0, mStream>>>(batchSize, topK, mRuntimeTopKDevice,
            runtimeTopKSize, topP, mRuntimeTopPDevice, runtimeTopPSize, mSkipDecodeDevice, batchSlots);
    }

    cudaAutoCpy(mSkipDecodeHost, mSkipDecodeDevice, mDecoderDomain.getBatchSize(), mStream);
    std::vector<SizeType32> runtimeTopKs(mDecoderDomain.getBatchSize());
    cudaAutoCpy(runtimeTopKs.data(), mRuntimeTopKDevice, mDecoderDomain.getBatchSize(), mStream);
    {
        runtime::SizeType32 maxTopK = 0;
        for (SizeType32 bi = 0; bi < batchSize; ++bi)
        {
            auto bid = bi;
            if (batchSlots)
            {
                bid = batchSlots[bi];
            }
            maxTopK = std::max(maxTopK, runtimeTopKs[bid]);
        }
        mRuntimeMaxTopK = std::max(mRuntimeMaxTopK, maxTopK);
    }

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template <typename T>
void TopKSamplingLayer<T>::forwardAsync(
    std::shared_ptr<BaseDecodingOutputs> const& outputs, std::shared_ptr<BaseDecodingInputs> const& baseInputs)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    auto inputs = std::dynamic_pointer_cast<SamplingInputs>(baseInputs);

    auto const batchSize = inputs->logits->shape[0];

    auto logits = inputs->logits->template getPtr<T>();
    auto endIds = inputs->endIds.template getPtr<TokenIdType const>();
    auto batchSlots = inputs->batchSlots ? inputs->batchSlots->template getPtr<SizeType32 const>() : nullptr;
    auto curandStatesDevice = inputs->curandStates;
    auto samplingWorkspaceDevice = inputs->samplingWorkspace;
    auto const probsComputed = inputs->probsComputed;

    std::vector<int32_t> batchSlotsVec(batchSize);
    std::iota(batchSlotsVec.begin(), batchSlotsVec.end(), 0);
    auto batchSlotsHost = inputs->batchSlots ? inputs->batchSlots->template getPtr<int const>() : batchSlotsVec.data();
    auto const skip = allOfBatchSlots(batchSlotsHost, mSkipDecodeHost, batchSize, true);
    if (skip)
    {
        return;
    }

    TLLM_CHECK_WITH_INFO(curandStatesDevice, "No curand states provided");
    TLLM_CHECK_WITH_INFO(samplingWorkspaceDevice, "No sampling workspace provided");

    FinishedState* finishedInput = (inputs->finished)
        ? reinterpret_cast<FinishedState*>(inputs->finished->template getPtr<FinishedState::UnderlyingType>())
        : nullptr;
    FinishedState* finishedOutput = (outputs->finished)
        ? reinterpret_cast<FinishedState*>(outputs->finished->template getPtr<FinishedState::UnderlyingType>())
        : nullptr;

    auto cumLogProbs = (outputs->cumLogProbs) ? outputs->cumLogProbs->template getPtr<float>() : nullptr;
    auto outputLogProbs
        = (outputs->outputLogProbsTiled) ? outputs->outputLogProbsTiled->template getPtr<float>() : nullptr;
    auto sequenceLengths = (outputs->sequenceLength) ? outputs->sequenceLength->template getPtr<SizeType32>() : nullptr;

    TopKSamplingKernelParams<T> params;
    params.logProbs = logits;
    params.outputIdsPtrs = outputs->outputIdsPtr.template getPtr<TokenIdType*>();
    params.workspace = samplingWorkspaceDevice;
    params.maxTopP = 1.0f;
    params.topPs = mRuntimeTopPDevice;
    params.maxTopK = mRuntimeMaxTopK;
    params.topKs = mRuntimeTopKDevice;
    params.sequenceLengths = sequenceLengths;
    params.endIds = endIds;
    params.batchSlots = batchSlots;
    params.finishedInput = finishedInput;
    params.finishedOutput = finishedOutput;
    params.skipDecode = mSkipDecodeDevice;
    params.cumLogProbs = cumLogProbs;
    params.outputLogProbs = outputLogProbs;
    params.curandState = curandStatesDevice;
    params.batchSize = batchSize;
    params.maxBatchSize = mDecoderDomain.getBatchSize();
    params.maxTokensPerStep = 1;
    params.vocabSizePadded = mDecoderDomain.getVocabSizePadded();
    params.normalizeLogProbs = mNormalizeLogProbs;
    params.logitsHasProbs = probsComputed;

    invokeBatchTopKSampling(params, mStream);
    sync_check_cuda_error();

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template class TopKSamplingLayer<float>;
template class TopKSamplingLayer<half>;

} // namespace tensorrt_llm::layers
