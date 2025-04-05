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
#include <memory>

#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/runtime/tllmBuffers.h"

namespace tc = tensorrt_llm::common;
namespace tr = tensorrt_llm::runtime;

TEST(TestGetPtrCudaMemoryType, TestMemoryTypesAreAsExpected)
{
    auto const cpuBuffer = std::make_unique<tr::HostBuffer>(1024, tr::TRTDataType<float>::value);

    // Note: this I think will change with hardware HMM support. To be confirmed. If it does, check for this
    // support and change expected memory type value accordingly.
    ASSERT_EQ(tc::getPtrCudaMemoryType(cpuBuffer->data()), cudaMemoryType::cudaMemoryTypeUnregistered)
        << "Host paged memory should appear as unregistered to CUDA.";

    auto const pinnedCpuBuffer = std::make_unique<tr::PinnedBuffer>(1024, tr::TRTDataType<float>::value);
    ASSERT_EQ(tc::getPtrCudaMemoryType(pinnedCpuBuffer->data()), cudaMemoryType::cudaMemoryTypeHost)
        << "The memory type of a pinned CPU buffer was not 'host'. Is this system using Confidential Computing?";

    auto const pinnedPoolCpuBuffer = std::make_unique<tr::PinnedPoolBuffer>(1024, tr::TRTDataType<float>::value);
    ASSERT_EQ(tc::getPtrCudaMemoryType(pinnedPoolCpuBuffer->data()), cudaMemoryType::cudaMemoryTypeHost)
        << "The memory type of a pinned CPU buffer was not 'host'. Is this system using Confidential Computing?";

    if (tc::getDeviceCount() <= 0)
    {
        GTEST_SKIP() << "This test cannot run further when no devices are present on the system.";
    }
    auto const stream = std::make_shared<tr::CudaStream>();
    auto const pool = tensorrt_llm::runtime::CudaMemPool::getPrimaryPoolForDevice(stream->getDevice());
    auto const deviceBuffer
        = std::make_unique<tr::DeviceBuffer>(1024, tr::TRTDataType<float>::value, tr::CudaAllocatorAsync{stream, pool});
    ASSERT_EQ(tc::getPtrCudaMemoryType(deviceBuffer->data()), cudaMemoryType::cudaMemoryTypeDevice);
    auto const deviceSyncBuffer = std::make_unique<tr::StaticDeviceBuffer>(1024, tr::TRTDataType<float>::value);
    ASSERT_EQ(tc::getPtrCudaMemoryType(deviceSyncBuffer->data()), cudaMemoryType::cudaMemoryTypeDevice);
}
