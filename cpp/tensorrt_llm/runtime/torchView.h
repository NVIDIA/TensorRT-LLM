/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.
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

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/runtime/iTensor.h"
#include "tensorrt_llm/runtime/torchUtils.h"

#include <ATen/ATen.h>
#include <torch/types.h>

#include <memory>

namespace tensorrt_llm::runtime
{
class TorchView : virtual public ITensor
{
public:
    static ITensor::UniquePtr of(at::Tensor&& tensor)
    {
        return ITensor::UniquePtr{new TorchView{std::move(tensor)}};
    }

    static ITensor::UniquePtr of(at::Tensor tensor)
    {
        return ITensor::UniquePtr{new TorchView{std::move(tensor)}};
    }

    void* data() override
    {
        if (getSize() == 0)
            return nullptr;
        return mTensor.data_ptr();
    }

    [[nodiscard]] void const* data() const override
    {
        if (getSize() == 0)
            return nullptr;
        return mTensor.data_ptr();
    }

    [[nodiscard]] size_t getSize() const override
    {
        return mTensor.numel();
    }

    [[nodiscard]] std::size_t getCapacity() const override
    {
        return mCapacity;
    }

    [[nodiscard]] DataType getDataType() const override
    {
        return TorchUtils::dataType(mTensor.scalar_type());
    }

    [[nodiscard]] MemoryType getMemoryType() const override
    {
        return mTensor.is_cuda() ? MemoryType::kGPU : mTensor.is_pinned() ? MemoryType::kPINNED : MemoryType::kCPU;
    }

    void resize(std::size_t newSize) override
    {
        TLLM_CHECK(newSize <= getCapacity());

        if (newSize != getSize())
        {
            using dimType = std::remove_reference_t<decltype(mDims.d[0])>;
            auto constexpr max_size = std::numeric_limits<dimType>::max();
            TLLM_CHECK_WITH_INFO(newSize <= max_size, "New size is too large. Use reshape() instead.");
            mTensor.resize_({static_cast<at::IntArrayRef::value_type>(newSize)});
            mDims.nbDims = 1;
            mDims.d[0] = static_cast<dimType>(newSize);
        }
    }

    void release() override
    {
        resize(0);
    }

    [[nodiscard]] Shape const& getShape() const override
    {
        return mDims;
    }

    void reshape(Shape const& dims) override
    {
        TLLM_CHECK(volumeNonNegative(dims) <= getCapacity());
        mTensor.resize_(TorchUtils::shape(dims));
        mDims = dims;
    }

private:
    explicit TorchView(at::Tensor&& tensor)
        : mTensor(tensor)
        , mDims{TorchUtils::shape(mTensor.sizes())}
        , mCapacity{static_cast<std::size_t>(mTensor.numel())}
    {
        TLLM_CHECK(mTensor.is_contiguous());
    };

    at::Tensor mTensor;
    Shape mDims;
    std::size_t mCapacity;
};
} // namespace tensorrt_llm::runtime
