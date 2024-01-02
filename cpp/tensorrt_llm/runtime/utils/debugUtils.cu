/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.  All rights reserved.
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

#include "debugUtils.h"

#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/memoryUtils.h"

namespace
{

__global__ void checkTensorNanKernel(const float* data, std::size_t size, int* foundNan)
{
    auto tidx = blockIdx.x * blockDim.x + threadIdx.x;

    int32_t found = 0;

    for (auto idx = tidx; idx < size; idx += blockDim.x * gridDim.x)
    {
        auto value = data[idx];
        if (isnan(value))
        {
            found = 1;
            break;
        }
    }
    atomicCAS(foundNan, 0, found);
}
} // namespace

using namespace tensorrt_llm::runtime;
namespace tc = tensorrt_llm::common;

namespace tensorrt_llm::runtime::utils
{

void invokeCheckTensorNanKernel(const float* data, std::size_t size, int* foundNan, cudaStream_t stream)
{
    constexpr uint32_t kThreadsPerCta = 256;
    checkTensorNanKernel<<<tc::ceilDiv(size, kThreadsPerCta), kThreadsPerCta, 0, stream>>>(data, size, foundNan);
}

bool tensorHasNan(const IBuffer& tensor, BufferManager& manager)
{
    auto foundNan = manager.pinned(ITensor::makeShape({1}), nvinfer1::DataType::kINT32);
    auto foundNanPtr = bufferCast<int32_t>(*foundNan);
    foundNanPtr[0] = 0;
    const auto size = tensor.getSize();
    invokeCheckTensorNanKernel(bufferCast<float>(tensor), size, foundNanPtr, manager.getStream().get());
    manager.getStream().synchronize();
    return static_cast<bool>(foundNanPtr[0]);
}
} // namespace tensorrt_llm::runtime::utils
