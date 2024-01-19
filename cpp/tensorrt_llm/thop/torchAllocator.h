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

#pragma once

#include "tensorrt_llm/common/allocator.h"

#ifdef TORCH_CUDA
#include "torch/extension.h"
#endif

#include <cuda_runtime.h>
#include <string>
#include <unordered_map>

namespace tc = tensorrt_llm::common;

namespace tensorrt_llm
{
namespace thop
{

class TorchAllocator : public tc::IAllocator
{
public:
    explicit TorchAllocator(cudaStream_t stream)
        : mStream(stream)
    {
    }

    ~TorchAllocator() override = default;

    void free(void** ptr) override;

protected:
    bool contains(void const* ptr) const override
    {
        return mPointerMapping.find(ptr) != mPointerMapping.end();
    }

    tc::ReallocType reallocType(void const* ptr, size_t size) const override;

    void* malloc(size_t size, bool setZero) override;

    void memSet(void* ptr, int val, size_t size) override;

private:
    std::unordered_map<void const*, torch::Tensor> mPointerMapping{};

    cudaStream_t mStream{};
};

} // namespace thop
} // namespace tensorrt_llm
