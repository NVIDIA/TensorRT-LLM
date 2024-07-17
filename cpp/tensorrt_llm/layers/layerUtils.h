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

#pragma once

#include <algorithm>
#include <optional>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/memoryUtils.h"
#include "tensorrt_llm/kernels/decodingCommon.h"
#include "tensorrt_llm/layers/decodingParams.h"
#include "tensorrt_llm/runtime/common.h"

namespace tensorrt_llm
{
namespace layers
{

// Using a local lambda in beam search layers to fill buffers causes an internal compiler error on nvcc windows.
// As a workaround and to promote DRY, the fill logic is refactored into FillBuffers below.
struct FillBuffers
{

    template <typename T>
    void operator()(std::optional<std::vector<T>> const& optParam, T const defaultValue, std::vector<T>& hostBuffer,
        T* deviceBuffer, runtime::SizeType32 const* batchSlots, std::pair<float, float> const& limits,
        std::string const& name) const
    {
        using tensorrt_llm::common::cudaAutoCpy;

        for (size_t bi = 0; bi < batchSize; ++bi)
        {
            auto value = defaultValue;
            auto const batchSlot = batchSlots ? batchSlots[bi] : bi;
            if (optParam)
            {
                if (optParam->size() == 1)
                {
                    value = optParam->front();
                }
                else
                {
                    TLLM_CHECK_WITH_INFO(optParam->size() == batchSize, "Argument vector size mismatch.");
                    value = optParam.value()[bi];
                }
            }
            TLLM_CHECK_WITH_INFO(limits.first < static_cast<float>(value) && static_cast<float>(value) <= limits.second,
                "%s param (%f) is out of limits (%f, %f]", name.c_str(), static_cast<float>(value), limits.first,
                limits.second);
            hostBuffer[batchSlot] = value;
        }

        if (batchSlots)
        {
            cudaAutoCpy(deviceBuffer, hostBuffer.data(), maxBatchSize, stream);
        }
        else
        {
            cudaAutoCpy(deviceBuffer, hostBuffer.data(), batchSize, stream);
        }
    }

    runtime::SizeType32 batchSize;
    runtime::SizeType32 maxBatchSize;
    cudaStream_t stream;
};

template <typename T>
inline bool allOfBatchSlots(
    runtime::SizeType32 const* batchSlotsHost, T const* data, runtime::SizeType32 batchSize, T value)
{
    return std::all_of(
        batchSlotsHost, batchSlotsHost + batchSize, [&](runtime::SizeType32 b) { return data[b] == value; });
}

inline DecoderDomain getLocalDecoderDomain(
    std::shared_ptr<BaseDecodingInputs> baseInputs, DecoderDomain const& globalDecoderDomain)
{
    auto inputs = std::dynamic_pointer_cast<DecodingInputs>(baseInputs);
    runtime::SizeType32 batchSize{baseInputs->localBatchSize};
    runtime::SizeType32 beamWidth{0};
    runtime::SizeType32 vocabSize{0};
    if (inputs->logits)
    {
        auto const& logitsShape = inputs->logits->shape;
        TLLM_CHECK(logitsShape.size() == 3 || logitsShape.size() == 4);
        auto const idxOffset = logitsShape.size() - 3;
        beamWidth = logitsShape[idxOffset + 1];
        vocabSize = logitsShape[idxOffset + 2];
    }
    else if (inputs->logitsVec)
    {
        TLLM_CHECK(inputs->logitsVec->size());
        auto const& logitsShape = inputs->logitsVec.value()[0].shape;
        TLLM_CHECK(logitsShape.size() == 3 || logitsShape.size() == 4);
        auto const idxOffset = logitsShape.size() - 3;
        beamWidth = logitsShape[idxOffset + 1];
        vocabSize = logitsShape[idxOffset + 2];
    }
    else if (inputs->batchSlots)
    {
        auto const& batchSlotsShape = inputs->batchSlots->shape;
        beamWidth = globalDecoderDomain.getBeamWidth();
        vocabSize = globalDecoderDomain.getVocabSize();
    }
    else
    {
        TLLM_THROW("Can't get local Decoder domain");
    }
    return DecoderDomain(batchSize, beamWidth, vocabSize);
}

} // namespace layers
} // namespace tensorrt_llm
