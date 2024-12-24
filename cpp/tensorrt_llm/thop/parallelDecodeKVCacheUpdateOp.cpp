/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
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

#include "tensorrt_llm/kernels/kvCacheIndex.h"
#include "tensorrt_llm/kernels/speculativeDecoding/kvCacheUpdateKernels.h"
#include "tensorrt_llm/thop/thUtils.h"

#include <cstdint>

namespace th = torch;
namespace tksd = tensorrt_llm::kernels::speculative_decoding;

namespace torch_ext
{

void updateKVCacheDraftTokenLocation(torch::Tensor seqAcceptedDraftTokenOffsetsTensor,
    torch::Tensor packedAcceptedDraftTokensIndicesTensor, torch::Tensor pastKeyValueLengthsTensor, bool usePagedKVCache,
    int64_t layerCount, int64_t numKVHeads, int64_t headSizeInBytes, int64_t rewindDraftTokenCount,
    int64_t maxKVCacheLen, th::optional<torch::Tensor> rewindDraftTokenTensor,
    th::optional<std::vector<torch::Tensor>> pastKeyValueListOpt = th::nullopt,
    th::optional<torch::Tensor> pointerArrayOpt = th::nullopt, th::optional<torch::Tensor> offsetArrayOpt = th::nullopt,
    th::optional<int64_t> maxBlocksPerSeqOpt = th::nullopt, th::optional<int64_t> tokensPerBlockOpt = th::nullopt,
    th::optional<int64_t> stream_ptr = th::nullopt)
{
    TLLM_CHECK_WITH_INFO(
        at::cuda::is_available(), "update_kv_cache_draft_token_location should be called with cuda enabled.");
    cudaStream_t stream;
    if (stream_ptr.has_value())
    {
        stream = reinterpret_cast<cudaStream_t>(stream_ptr.value());
    }
    else
    {
        stream = at::cuda::getCurrentCUDAStream();
    }
    TLLM_CHECK_WITH_INFO(seqAcceptedDraftTokenOffsetsTensor.dim() == 1
            && seqAcceptedDraftTokenOffsetsTensor.scalar_type() == torch::kInt,
        "accepted_draft_token_offsets tensor should be 1D int tensor.");
    int seqCount = seqAcceptedDraftTokenOffsetsTensor.size(0) - 1;
    TLLM_CHECK_WITH_INFO(seqCount > 0, "seqCount should be larger than 0");

    TLLM_CHECK_WITH_INFO(packedAcceptedDraftTokensIndicesTensor.dim() == 1
            && packedAcceptedDraftTokensIndicesTensor.scalar_type() == torch::kInt,
        "packed_accepted_draft_tokens_indices tensor should be 1D int tensor.");

    TLLM_CHECK_WITH_INFO(pastKeyValueLengthsTensor.dim() == 1 && pastKeyValueLengthsTensor.size(0) == seqCount
            && pastKeyValueLengthsTensor.scalar_type() == torch::kInt,
        "past_key_value_lengths tensor should be 1D int tensor with same length as seqCount");
    int* rewindDraftTokenTensorPtr = nullptr;
    if (rewindDraftTokenTensor.has_value())
    {
        TLLM_CHECK_WITH_INFO(rewindDraftTokenTensor.value().dim() == 1
                && rewindDraftTokenTensor.value().size(0) == seqCount
                && rewindDraftTokenTensor.value().scalar_type() == torch::kInt,
            "rewindDraftTokenTensor should be 1D int tensor same length as seqCount");
        rewindDraftTokenTensorPtr = rewindDraftTokenTensor.value().data_ptr<int>();
    }

    if (usePagedKVCache)
    {
        TLLM_CHECK_WITH_INFO(
            pointerArrayOpt.has_value(), "pool_pointer_array should be set when using paged KV cache.");
        TLLM_CHECK_WITH_INFO(offsetArrayOpt.has_value(), "block_offset_array should be set when using paged KV cache.");
        TLLM_CHECK_WITH_INFO(
            maxBlocksPerSeqOpt.has_value(), "max_blocks_per_seq should be set when using paged KV cache.");
        TLLM_CHECK_WITH_INFO(
            tokensPerBlockOpt.has_value(), "tokens_per_block should be set when using paged KV cache.");

        auto const& pointerArray = pointerArrayOpt.value();
        auto const& offsetArray = offsetArrayOpt.value();
        // chunked_context + sliding window attention is not supported in python runtime, so useOneMoreBlock can always
        // be supported.
        bool constexpr canUseOneMoreBlock{true};

        tksd::updateKVBlockArrayDraftTokenLocation(seqAcceptedDraftTokenOffsetsTensor.data_ptr<int>(),
            packedAcceptedDraftTokensIndicesTensor.data_ptr<int>(), pastKeyValueLengthsTensor.data_ptr<int>(),
            reinterpret_cast<void* const*>(pointerArray.data_ptr<int64_t>()),
            reinterpret_cast<tensorrt_llm::kernels::KVCacheIndex*>(
                offsetArray.data_ptr<tensorrt_llm::kernels::KVCacheIndex::UnderlyingType>()),
            layerCount, seqCount, numKVHeads, headSizeInBytes, rewindDraftTokenCount, rewindDraftTokenTensorPtr,
            nullptr, nullptr, maxKVCacheLen, maxBlocksPerSeqOpt.value(), tokensPerBlockOpt.value(), canUseOneMoreBlock,
            stream);
    }
    else
    {
        TLLM_CHECK_WITH_INFO(
            pastKeyValueListOpt.has_value(), "block_pointer_array should be set when using linear KV cache.");
        std::vector<int8_t*> pastKeyValueList;
        pastKeyValueList.reserve(layerCount);
        for (auto& pastKeyValueTensor : pastKeyValueListOpt.value())
        {
            pastKeyValueList.push_back(static_cast<int8_t*>(pastKeyValueTensor.data_ptr()));
        }
        tksd::updateLinearKVCacheDraftTokenLocation(seqAcceptedDraftTokenOffsetsTensor.data_ptr<int>(),
            packedAcceptedDraftTokensIndicesTensor.data_ptr<int>(), pastKeyValueLengthsTensor.data_ptr<int>(),
            pastKeyValueList.data(), layerCount, seqCount, numKVHeads, headSizeInBytes, rewindDraftTokenCount,
            rewindDraftTokenTensorPtr, nullptr, maxKVCacheLen, stream);
    }
}

} // namespace torch_ext

static auto update_kv_cache_draft_token_location = torch::RegisterOperators(
    "tensorrt_llm::update_kv_cache_draft_token_location", &torch_ext::updateKVCacheDraftTokenLocation);
