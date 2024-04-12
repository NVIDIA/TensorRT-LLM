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

#include "tensorrt_llm/common/allocator.h"
#include "tensorrt_llm/common/tensor.h"

namespace tensorrt_llm
{
namespace layers
{

class BaseLayer
{
public:
    BaseLayer(cudaStream_t stream, std::shared_ptr<tensorrt_llm::common::IAllocator> allocator,
        cudaDeviceProp* prop = nullptr)
        : mStream(stream)
        , mAllocator(std::move(allocator))
        , mCudaDeviceProp(prop)
    {
    }

    virtual ~BaseLayer() = default;

    virtual cudaStream_t getStream()
    {
        return mStream;
    }

    virtual void setStream(cudaStream_t stream)
    {
        mStream = stream;
    }

protected:
    // device environments
    cudaStream_t mStream;
    std::shared_ptr<tensorrt_llm::common::IAllocator> mAllocator;
    cudaDeviceProp* mCudaDeviceProp = nullptr;

    bool mIsAllocateBuffer = false; // TODO to be deprecated
};

} // namespace layers
} // namespace tensorrt_llm
