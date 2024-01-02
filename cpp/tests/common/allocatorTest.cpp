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

#include <gtest/gtest.h>

#include "tensorrt_llm/common/cudaAllocator.h"
#include "tensorrt_llm/runtime/bufferManager.h"
#include "tensorrt_llm/runtime/cudaStream.h"

#include <memory>

namespace tc = tensorrt_llm::common;
namespace tr = tensorrt_llm::runtime;

TEST(Allocator, DeviceDestruction)
{
    auto streamPtr = std::make_shared<tr::CudaStream>();
    {
        auto allocator = std::make_unique<tc::CudaAllocator>(tr::BufferManager(streamPtr));

        auto constexpr sizeBytes = 1024 * 1024;
        void* ptr{};
        // device alloc
        ptr = allocator->reMalloc(ptr, sizeBytes, false);
        EXPECT_NE(ptr, nullptr);
        allocator->free(&ptr);
        EXPECT_EQ(ptr, nullptr);
        ptr = allocator->reMalloc(ptr, sizeBytes, true);
        EXPECT_NE(ptr, nullptr);
        ptr = allocator->reMalloc(ptr, sizeBytes / 2, true);
        EXPECT_NE(ptr, nullptr);
        ptr = allocator->reMalloc(ptr, sizeBytes * 2, true);
        EXPECT_NE(ptr, nullptr);
        ptr = allocator->reMalloc(ptr, sizeBytes, false);
        EXPECT_NE(ptr, nullptr);
    }
    streamPtr->synchronize();
}
