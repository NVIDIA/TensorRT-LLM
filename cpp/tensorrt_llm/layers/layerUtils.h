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
#include "tensorrt_llm/layers/decodingParams.h"
#include "tensorrt_llm/runtime/common.h"
#include "tensorrt_llm/runtime/iBuffer.h"

namespace tensorrt_llm::layers
{

// Using a local lambda in beam search layers to fill buffers causes an internal compiler error on nvcc windows.
// As a workaround and to promote DRY, the fill logic is refactored into FillBuffers below.
struct FillBuffers
{
    using BufferPtr = runtime::IBuffer::SharedPtr;
    using TensorConstPtr = runtime::ITensor::UniqueConstPtr;
    using BufferConstPtr = runtime::IBuffer::SharedConstPtr;

    template <typename T>
    void operator()(std::optional<std::vector<T>> const& optParam, T const defaultValue, BufferPtr hostBuffer,
        BufferPtr deviceBuffer, BufferConstPtr batchSlots, std::pair<float, float> const& limits,
        std::string const& name) const
    {
        auto hostBufferRange = runtime::BufferRange<T>(*hostBuffer);
        for (size_t bi = 0; bi < batchSize; ++bi)
        {
            auto value = defaultValue;
            auto batchSlot = runtime::bufferCast<runtime::SizeType32>(*batchSlots)[bi];
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
            hostBufferRange[batchSlot] = value;
        }

        if (batchSlots)
        {
            auto const hostSlice = runtime::IBuffer::slice(hostBuffer, 0, maxBatchSize);
            auto deviceSlice = runtime::IBuffer::slice(deviceBuffer, 0, maxBatchSize);
            mBufferManager->copy(*hostSlice, *deviceSlice);
        }
        else
        {
            auto const hostSlice = runtime::IBuffer::slice(hostBuffer, 0, batchSize);
            auto deviceSlice = runtime::IBuffer::slice(deviceBuffer, 0, batchSize);
            mBufferManager->copy(*hostSlice, *deviceSlice);
        }
    }

    runtime::SizeType32 batchSize;
    runtime::SizeType32 maxBatchSize;
    std::shared_ptr<runtime::BufferManager> mBufferManager;
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
        auto const& logitsShape = inputs->logits.value()->getShape();
        TLLM_CHECK(logitsShape.nbDims == 3 || logitsShape.nbDims == 4);
        beamWidth = inputs->logits.value()->getDimension<-2>();
        vocabSize = inputs->logits.value()->getDimension<-1>();
    }
    else if (inputs->logitsVec)
    {
        TLLM_CHECK(inputs->logitsVec->size());
        auto const& logitsShape = inputs->logitsVec.value()[0]->getShape();
        TLLM_CHECK(logitsShape.nbDims == 3 || logitsShape.nbDims == 4);
        beamWidth = inputs->logitsVec.value()[0]->getDimension<-2>();
        vocabSize = inputs->logitsVec.value()[0]->getDimension<-1>();
    }
    else if (inputs->batchSlots)
    {
        beamWidth = globalDecoderDomain.getBeamWidth();
        vocabSize = globalDecoderDomain.getVocabSize();
    }
    else
    {
        TLLM_THROW("Can't get local Decoder domain");
    }
    return {batchSize, beamWidth, vocabSize};
}

} // namespace tensorrt_llm::layers
