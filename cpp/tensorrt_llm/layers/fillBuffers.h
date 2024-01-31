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

#pragma once

#include <algorithm>
#include <optional>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/memoryUtils.h"
#include "tensorrt_llm/kernels/decodingCommon.h"

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
        T* deviceBuffer, void* deviceTmpBuffer, const int* batchSlots) const
    {
        using tensorrt_llm::common::cudaAutoCpy;

        hostBuffer.resize(batchSize);
        if (!optParam)
        {
            std::fill(std::begin(hostBuffer), std::end(hostBuffer), defaultValue);
        }
        else if (optParam->size() == 1)
        {
            std::fill(std::begin(hostBuffer), std::end(hostBuffer), optParam->front());
        }
        else
        {
            TLLM_CHECK_WITH_INFO(optParam->size() == batchSize, "Argument vector size mismatch.");
            std::copy(optParam->begin(), optParam->end(), std::begin(hostBuffer));
        }
        if (deviceTmpBuffer && batchSlots)
        {
            cudaAutoCpy(reinterpret_cast<T*>(deviceTmpBuffer), hostBuffer.data(), batchSize, stream);
            tensorrt_llm::kernels::invokeScatterDecodingParams(
                reinterpret_cast<T*>(deviceTmpBuffer), deviceBuffer, batchSlots, batchSize, stream);
        }
        else
        {
            cudaAutoCpy(deviceBuffer, hostBuffer.data(), batchSize, stream);
        }
    }

    size_t batchSize;
    cudaStream_t stream;
};

} // namespace layers

} // namespace tensorrt_llm
