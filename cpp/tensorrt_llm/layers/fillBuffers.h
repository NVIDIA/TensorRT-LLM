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
        T* deviceBuffer, runtime::SizeType const* batchSlots, std::pair<float, float> const& limits,
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

    runtime::SizeType batchSize;
    runtime::SizeType maxBatchSize;
    cudaStream_t stream;
};

} // namespace layers

} // namespace tensorrt_llm
