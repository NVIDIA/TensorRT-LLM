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

#include "tensorrt_llm/runtime/utils/debugUtils.h"

#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/memoryUtils.h"
#include <cfloat>

namespace
{
template <typename T>
__global__ void checkTensorNanKernel(T const* data, std::size_t size, int* foundNan)
{
    auto tidx = blockIdx.x * blockDim.x + threadIdx.x;

    int32_t found = 0;

    for (auto idx = tidx; idx < size; idx += blockDim.x * gridDim.x)
    {
        auto value = static_cast<float>(data[idx]);
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

template <typename T>
void invokeCheckTensorNanKernel(T const* data, std::size_t size, int* foundNan, cudaStream_t stream)
{
    constexpr uint32_t kThreadsPerCta = 256;
    checkTensorNanKernel<<<tc::ceilDiv(size, kThreadsPerCta), kThreadsPerCta, 0, stream>>>(data, size, foundNan);
}

template void invokeCheckTensorNanKernel(float const* data, std::size_t size, int* foundNan, cudaStream_t stream);
template void invokeCheckTensorNanKernel(half const* data, std::size_t size, int* foundNan, cudaStream_t stream);
template void invokeCheckTensorNanKernel(
    __nv_bfloat16 const* data, std::size_t size, int* foundNan, cudaStream_t stream);
template void invokeCheckTensorNanKernel(
    __nv_fp8_e4m3 const* data, std::size_t size, int* foundNan, cudaStream_t stream);

template <typename T>
void printLogitsKeyInfo(ITensor const& tensor, std::string const& infoStr)
{
    auto const& shape = tensor.getShape();
    auto const volume = ITensor::volume(shape);

    BufferManager::ITensorPtr host{};
    T const* hostData;
    if (tensor.getMemoryType() == MemoryType::kGPU)
    {
        auto streamPtr = std::make_shared<CudaStream>();
        BufferManager manager{streamPtr};
        host = manager.copyFrom(tensor, MemoryType::kCPU);
        streamPtr->synchronize();
        hostData = bufferCast<T>(*host);
    }
    else
    {
        hostData = bufferCast<T>(tensor);
    }

    std::stringstream ss;
    ss << infoStr;
    ss << " Shape: " << shape;
    ss << "; Top 5: ";
    for (size_t ki = 0; ki < 5; ++ki)
    {
        ss << static_cast<float>(hostData[ki]) << ", ";
    }

    ss << " Last 5: ";
    for (size_t ki = volume - 6; ki < volume; ++ki)
    {
        ss << static_cast<float>(hostData[ki]) << ", ";
    }

    // find max, min, avg
    double mSum = 0.f;
    float mMax = -FLT_MAX;
    float mMin = FLT_MAX;

    for (size_t ki = 0; ki < volume; ++ki)
    {
        float value = static_cast<float>(hostData[ki]);
        mSum += value;
        if (value > mMax)
        {
            mMax = value;
        }
        if (value < mMin)
        {
            mMin = value;
        }
    }
    float mAvg = mSum / volume;

    ss << " avg: " << mAvg << ", min: " << mMin << ", max: " << mMax << std::endl;

    TLLM_LOG_TRACE(ss.str());
}

template void printLogitsKeyInfo<float>(ITensor const& tensor, std::string const& infoStr);
template void printLogitsKeyInfo<half>(ITensor const& tensor, std::string const& infoStr);
template void printLogitsKeyInfo<__nv_bfloat16>(ITensor const& tensor, std::string const& infoStr);
template void printLogitsKeyInfo<__nv_fp8_e4m3>(ITensor const& tensor, std::string const& infoStr);

template <typename T>
bool tensorHasNan(ITensor const& tensor, BufferManager const& manager, std::string const& infoStr)
{
    printLogitsKeyInfo<T>(tensor, infoStr);
    auto foundNan = BufferManager::pinnedPool(ITensor::makeShape({1}), nvinfer1::DataType::kINT32);
    auto foundNanPtr = bufferCast<int32_t>(*foundNan);
    foundNanPtr[0] = 0;
    auto const size = tensor.getSize();
    invokeCheckTensorNanKernel(bufferCast<T>(tensor), size, foundNanPtr, manager.getStream().get());
    manager.getStream().synchronize();
    return static_cast<bool>(foundNanPtr[0]);
}

template bool tensorHasNan<float>(ITensor const& tensor, BufferManager const& manager, std::string const& infoStr);
template bool tensorHasNan<half>(ITensor const& tensor, BufferManager const& manager, std::string const& infoStr);
template bool tensorHasNan<__nv_bfloat16>(
    ITensor const& tensor, BufferManager const& manager, std::string const& infoStr);
template bool tensorHasNan<__nv_fp8_e4m3>(
    ITensor const& tensor, BufferManager const& manager, std::string const& infoStr);

bool tensorHasNan(
    size_t M, size_t K, nvinfer1::DataType type, void const* data, cudaStream_t stream, std::string const& infoStr)
{
    auto tensorView = ITensor::wrap(
        const_cast<void*>(data), type, ITensor::makeShape({static_cast<int32_t>(M), static_cast<int32_t>(K)}));
    auto manager = BufferManager(std::make_shared<CudaStream>(stream));
    if (type == nvinfer1::DataType::kFLOAT)
    {
        return tensorHasNan<float>(*tensorView, manager, infoStr);
    }
    else if (type == nvinfer1::DataType::kHALF)
    {
        return tensorHasNan<half>(*tensorView, manager, infoStr);
    }
    else if (type == nvinfer1::DataType::kBF16)
    {
        return tensorHasNan<__nv_bfloat16>(*tensorView, manager, infoStr);
    }
    else if (type == nvinfer1::DataType::kFP8)
    {
        return tensorHasNan<__nv_fp8_e4m3>(*tensorView, manager, infoStr);
    }
    else
    {
        TLLM_THROW("Not supported type for Nan check");
    }
}

} // namespace tensorrt_llm::runtime::utils
