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

#include "tensorrt_llm/common/allocator.h"
#include "tensorrt_llm/runtime/bufferManager.h"

#include <cuda_runtime.h>
#include <unordered_map>

namespace tensorrt_llm
{

namespace common
{

class CudaAllocator : public IAllocator
{
public:
    explicit CudaAllocator(runtime::BufferManager bufferManager);

    ~CudaAllocator() override = default;

    void free(void** ptr) override;

protected:
    bool contains(void const* ptr) const override
    {
        return mPointerMapping.find(ptr) != mPointerMapping.end();
    }

    ReallocType reallocType(void const* ptr, size_t size) const override;

    void* malloc(size_t size, bool setZero) override;

    void memSet(void* ptr, int val, size_t size) override;

private:
    runtime::BufferManager mBufferManager;
    std::unordered_map<void const*, runtime::BufferManager::IBufferPtr> mPointerMapping{};
};

} // namespace common
} // namespace tensorrt_llm
