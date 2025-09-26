/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "tensorrt_llm/executor/tensor.h"

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/runtime/bufferManager.h"
#include "tensorrt_llm/runtime/iTensor.h"

#include <algorithm>
#include <cstring>
#include <memory>

namespace tr = tensorrt_llm::runtime;

namespace tensorrt_llm::executor
{

Tensor::Tensor(std::shared_ptr<runtime::ITensor> tensor)
    : mTensor(std::move(tensor))
{
}

void* Tensor::getData()
{
    return mTensor ? mTensor->data() : nullptr;
}

void const* Tensor::getData() const
{
    return mTensor ? mTensor->data() : nullptr;
}

DataType Tensor::getDataType() const
{
    if (!mTensor)
    {
        return DataType::kUNKNOWN;
    }
    switch (mTensor->getDataType())
    {
    case nvinfer1::DataType::kBOOL: return DataType::kBOOL;
    case nvinfer1::DataType::kINT8: return DataType::kINT8;
    case nvinfer1::DataType::kINT32: return DataType::kINT32;
    case nvinfer1::DataType::kUINT8: return DataType::kUINT8;
    case nvinfer1::DataType::kFP8: return DataType::kFP8;
    case nvinfer1::DataType::kHALF: return DataType::kFP16;
    case nvinfer1::DataType::kFLOAT: return DataType::kFP32;
    case nvinfer1::DataType::kBF16: return DataType::kBF16;
    case nvinfer1::DataType::kINT64: return DataType::kINT64;
    case nvinfer1::DataType::kINT4: [[fallthrough]] /* do nothing */;
    case nvinfer1::DataType::kFP4: [[fallthrough]] /* do nothing */;
    default: TLLM_THROW("Unsupported data type");
    }
}

MemoryType Tensor::getMemoryType() const
{
    if (!mTensor)
    {
        return MemoryType::kUNKNOWN;
    }

    switch (mTensor->getMemoryType())
    {

    case runtime::MemoryType::kGPU: return MemoryType::kGPU;
    case runtime::MemoryType::kCPU: return MemoryType::kCPU;
    case runtime::MemoryType::kPINNED: return MemoryType::kCPU_PINNED;
    case runtime::MemoryType::kUVM: return MemoryType::kUVM;
    case runtime::MemoryType::kPINNEDPOOL: return MemoryType::kCPU_PINNEDPOOL;
    }

    TLLM_THROW("Unsupported memory type");
}

Shape Tensor::getShape() const
{
    if (!mTensor)
    {
        return {};
    }

    auto const& shape = mTensor->getShape();
    if (shape.nbDims < 0)
    {
        return {};
    }

    return {shape.d, static_cast<Shape::size_type>(shape.nbDims)};
}

std::shared_ptr<runtime::ITensor> const& detail::toITensor(Tensor const& tensor)
{
    return tensor.mTensor;
}

Tensor detail::ofITensor(std::shared_ptr<runtime::ITensor> tensor)
{
    return Tensor(std::move(tensor));
}

std::size_t Tensor::getSize() const
{
    return mTensor ? mTensor->getSize() : 0;
}

std::size_t Tensor::getSizeInBytes() const
{
    return mTensor ? mTensor->getSizeInBytes() : 0;
}

namespace
{
tr::ITensor::Shape toDims(Shape const& shape)
{
    TLLM_CHECK(shape.size() <= tr::ITensor::Shape::MAX_DIMS);
    tr::ITensor::Shape dims;
    dims.nbDims = static_cast<decltype(dims.nbDims)>(shape.size());
    std::copy(shape.begin(), shape.end(), dims.d);
    return dims;
}

nvinfer1::DataType toDataType(DataType dataType)
{
    switch (dataType)
    {
    case DataType::kBOOL: return nvinfer1::DataType::kBOOL;
    case DataType::kUINT8: return nvinfer1::DataType::kUINT8;
    case DataType::kINT8: return nvinfer1::DataType::kINT8;
    case DataType::kINT32: return nvinfer1::DataType::kINT32;
    case DataType::kINT64: return nvinfer1::DataType::kINT64;
    case DataType::kBF16: return nvinfer1::DataType::kBF16;
    case DataType::kFP8: return nvinfer1::DataType::kFP8;
    case DataType::kFP16: return nvinfer1::DataType::kHALF;
    case DataType::kFP32: return nvinfer1::DataType::kFLOAT;
    case DataType::kUNKNOWN: TLLM_THROW("Unsupported data type");
    }

    TLLM_THROW("Unsupported data type");
}

} // namespace

Tensor Tensor::cpu(DataType dataType, Shape shape)
{
    auto const dims = toDims(shape);
    auto const dtype = toDataType(dataType);
    return Tensor{tr::BufferManager::cpu(dims, dtype)};
}

Tensor Tensor::pinned(DataType dataType, Shape shape)
{
    auto const dims = toDims(shape);
    auto const dtype = toDataType(dataType);
    return Tensor{tr::BufferManager::pinned(dims, dtype)};
}

Tensor Tensor::pooledPinned(DataType dataType, Shape shape)
{
    auto const dims = toDims(shape);
    auto const dtype = toDataType(dataType);
    return Tensor{tr::BufferManager::pinnedPool(dims, dtype)};
}

Tensor Tensor::managed(DataType dataType, Shape shape)
{
    auto const dims = toDims(shape);
    auto const dtype = toDataType(dataType);
    return Tensor{tr::BufferManager::managed(dims, dtype)};
}

Tensor Tensor::gpu(DataType dataType, Tensor::CudaStreamPtr stream, Shape shape)
{
    auto const dims = toDims(shape);
    auto const dtype = toDataType(dataType);
    auto manager = tr::BufferManager{std::move(stream)};
    return Tensor{manager.gpu(dims, dtype)};
}

Tensor Tensor::of(DataType dataType, void* data, Shape shape)
{
    return Tensor{tr::ITensor::wrap(data, toDataType(dataType), toDims(shape))};
}

Tensor Tensor::copyTo(std::shared_ptr<Impl> tensor, CudaStreamPtr stream) const
{
    if (mTensor->getMemoryType() == runtime::MemoryType::kGPU)
    {
        tr::BufferManager manager{std::move(stream)};
        manager.copy(*mTensor, *tensor);
    }
    else
    {
        std::memcpy(tensor->data(), getData(), getSizeInBytes());
    }
    return Tensor{std::move(tensor)};
}

Tensor Tensor::copyToCpu(Tensor::CudaStreamPtr stream) const
{
    TLLM_CHECK(*this);
    return copyTo(tr::BufferManager::cpu(mTensor->getShape(), mTensor->getDataType()), std::move(stream));
}

Tensor Tensor::copyToPinned(Tensor::CudaStreamPtr stream) const
{
    TLLM_CHECK(*this);
    return copyTo(tr::BufferManager::pinned(mTensor->getShape(), mTensor->getDataType()), std::move(stream));
}

Tensor Tensor::copyToPooledPinned(Tensor::CudaStreamPtr stream) const
{
    TLLM_CHECK(*this);
    return copyTo(tr::BufferManager::pinnedPool(mTensor->getShape(), mTensor->getDataType()), std::move(stream));
}

Tensor Tensor::copyToManaged(Tensor::CudaStreamPtr stream) const
{
    TLLM_CHECK(*this);
    return copyTo(tr::BufferManager::managed(mTensor->getShape(), mTensor->getDataType()), std::move(stream));
}

Tensor Tensor::copyToGpu(Tensor::CudaStreamPtr stream) const
{
    TLLM_CHECK(*this);
    tr::BufferManager manager{std::move(stream)};
    return Tensor{manager.copyFrom(*mTensor, runtime::MemoryType::kGPU)};
}

void Tensor::setZero(CudaStreamPtr stream)
{
    if (!mTensor)
    {
        return;
    }

    if (mTensor->getMemoryType() == runtime::MemoryType::kGPU)
    {
        auto manager = tr::BufferManager{std::move(stream)};
        manager.setZero(*mTensor);
    }
    else
    {
        std::memset(mTensor->data(), 0, getSizeInBytes());
    }
}

void Tensor::setFrom(Tensor const& other, Tensor::CudaStreamPtr stream)
{
    TLLM_CHECK(*this);
    TLLM_CHECK(other);
    mTensor->reshape(other.mTensor->getShape());
    if (mTensor->getMemoryType() == runtime::MemoryType::kGPU
        || other.mTensor->getMemoryType() == runtime::MemoryType::kGPU)
    {
        auto manager = tr::BufferManager{std::move(stream)};
        manager.copy(*other.mTensor, *mTensor);
    }
    else
    {
        std::memcpy(mTensor->data(), other.mTensor->data(), other.getSizeInBytes());
    }
}

} // namespace tensorrt_llm::executor
