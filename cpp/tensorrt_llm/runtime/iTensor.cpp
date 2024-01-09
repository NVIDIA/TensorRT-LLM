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

#include "tensorrt_llm/runtime/iTensor.h"

#include "tensorrt_llm/common/memoryUtils.h"
#include "tensorrt_llm/common/stringUtils.h"
#include "tensorrt_llm/runtime/bufferManager.h"
#include "tensorrt_llm/runtime/tensorView.h"
#include "tensorrt_llm/runtime/tllmBuffers.h"

#include <initializer_list>
#include <memory>

using namespace tensorrt_llm::runtime;

namespace tc = tensorrt_llm::common;

ITensor::UniquePtr ITensor::slice(SharedPtr tensor, std::size_t offset, std::size_t size)
{
    return std::make_unique<TensorView>(std::move(tensor), offset, size);
}

ITensor::UniquePtr ITensor::view(IBuffer::SharedPtr buffer, nvinfer1::Dims const& dims)
{
    auto const size = buffer->getSize();
    return std::make_unique<TensorView>(std::move(buffer), 0, size, dims);
}

nvinfer1::Dims ITensor::makeShape(std::initializer_list<SizeType> const& dims)
{
    TLLM_CHECK_WITH_INFO(dims.size() <= nvinfer1::Dims::MAX_DIMS, "Number of dimensions is too large");
    nvinfer1::Dims shape{};
    shape.nbDims = static_cast<decltype(Shape::nbDims)>(dims.size());
    std::copy(dims.begin(), dims.end(), shape.d);
    return shape;
}

std::string ITensor::toString(nvinfer1::Dims const& dims)
{
    if (dims.nbDims < 0)
    {
        return "invalid";
    }
    else if (dims.nbDims == 0)
    {
        return "()";
    }
    else
    {
        return tc::arr2str(dims.d, dims.nbDims);
    }
}

ITensor::UniquePtr ITensor::wrap(void* data, nvinfer1::DataType type, nvinfer1::Dims const& shape, std::size_t capacity)
{
    auto const size = volumeNonNegative(shape);
    TLLM_CHECK_WITH_INFO(size <= capacity, "Requested size is larger than capacity");
    auto memoryType = IBuffer::memoryType(data);

    ITensor::UniquePtr result;
    auto const capacityInBytes = capacity * BufferDataType(type).getSize();
    switch (memoryType)
    {
    case MemoryType::kPINNED:
        result.reset(new GenericTensor<PinnedBorrowingAllocator>( // NOLINT(modernize-make-unique)
            shape, capacity, type, PinnedBorrowingAllocator(data, capacityInBytes)));
        break;
    case MemoryType::kCPU:
        result.reset( // NOLINT(modernize-make-unique)
            new GenericTensor<CpuBorrowingAllocator>(
                shape, capacity, type, CpuBorrowingAllocator(data, capacityInBytes)));
        break;
    case MemoryType::kGPU:
        result.reset( // NOLINT(modernize-make-unique)
            new GenericTensor<GpuBorrowingAllocator>(
                shape, capacity, type, GpuBorrowingAllocator(data, capacityInBytes)));
        break;
    default: TLLM_THROW("Unknown memory type");
    }
    return result;
}

ITensor::Shape ITensor::squeeze(Shape const& shape, SizeType dim)
{
    TLLM_CHECK_WITH_INFO(0 < shape.nbDims, "Cannot squeeze 1-dimensional tensor");
    TLLM_CHECK_WITH_INFO(
        dim < shape.nbDims, tc::fmtstr("Invalid index %d, tensor has %d dimensions", dim, shape.nbDims));
    TLLM_CHECK_WITH_INFO(shape.d[dim] == 1, "Can only squeeze dimension of size 1");

    Shape newDims{shape.nbDims - 1};
    std::copy(shape.d, shape.d + dim, newDims.d);
    std::copy(shape.d + dim + 1, shape.d + shape.nbDims, newDims.d + dim);
    return newDims;
}

ITensor::Shape ITensor::unsqueeze(Shape const& shape, SizeType dim)
{
    TLLM_CHECK_WITH_INFO(shape.nbDims < Shape::MAX_DIMS, "Too many dimensions to unsqueeze");
    TLLM_CHECK_WITH_INFO(
        0 <= dim && dim <= shape.nbDims, common::fmtstr("Invalid dim %d, tensor has %d dimensions", dim, shape.nbDims));

    Shape newDims{shape.nbDims + 1};
    std::copy(shape.d, shape.d + dim, newDims.d);
    newDims.d[dim] = 1;
    std::copy(shape.d + dim, shape.d + shape.nbDims, newDims.d + dim + 1);
    return newDims;
}

namespace
{
template <typename T>
void printTensor(ITensor const& tensor, std::ostream& out)
{
    TLLM_CHECK_WITH_INFO(tensor.getDataType() == TRTDataType<typename std::remove_cv<T>::type>::value,
        tc::fmtstr("Data type mismatch: %d vs %d", static_cast<std::int32_t>(tensor.getDataType()),
            static_cast<std::int32_t>(TRTDataType<typename std::remove_cv<T>::type>::value)));
    auto const& shape = tensor.getShape();
    out << "shape: " << shape << std::endl;
    out << "vals: " << std::endl;

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

    using TOutput
        = std::conditional_t<std::is_same_v<T, std::int8_t> || std::is_same_v<T, std::uint8_t>, std::int32_t, T>;
    if (shape.nbDims > 3)
    {
        out << "Not printing elements for more than 3 dims\n";
    }
    else if (shape.nbDims == 3 && shape.d[2] > 1)
    {
        for (int i = 0; i < shape.d[0]; ++i)
        {
            for (int j = 0; j < shape.d[1]; ++j)
            {
                out << "i=" << i << " j=" << j << ": ";
                tc::arr2outCasted<TOutput>(out, hostData + tc::flat_index(shape.d, i, j, 0), shape.d[2]) << "\n";
            }
        }
    }
    else if (shape.nbDims >= 2 && shape.d[1] > 1)
    {
        for (int i = 0; i < shape.d[0]; ++i)
        {
            out << "i=" << i << ": ";
            tc::arr2outCasted<TOutput>(out, hostData + tc::flat_index(shape.d, i, 0), shape.d[1]) << "\n";
        }
    }
    else
    {
        tc::arr2outCasted<TOutput>(out, hostData, shape.d[0]) << "\n";
    }
    out << std::flush;
}

} // namespace

std::ostream& tensorrt_llm::runtime::operator<<(std::ostream& out, ITensor const& tensor)
{
    switch (tensor.getDataType())
    {
    case nvinfer1::DataType::kFLOAT: printTensor<float>(tensor, out); break;
    case nvinfer1::DataType::kHALF: printTensor<half>(tensor, out); break;
    case nvinfer1::DataType::kBOOL: printTensor<bool>(tensor, out); break;
    case nvinfer1::DataType::kINT8: printTensor<std::int8_t>(tensor, out); break;
    case nvinfer1::DataType::kINT32: printTensor<std::int32_t>(tensor, out); break;
    case nvinfer1::DataType::kINT64: printTensor<std::int64_t>(tensor, out); break;
    case nvinfer1::DataType::kUINT8: printTensor<std::uint8_t>(tensor, out); break;
#ifdef ENABLE_BF16
    case nvinfer1::DataType::kBF16: printTensor<__nv_bfloat16>(tensor, out); break;
#endif
    default: TLLM_THROW("Unsupported data type");
    }

    return out;
}
