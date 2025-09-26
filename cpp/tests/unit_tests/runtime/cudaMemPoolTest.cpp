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

#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/runtime/cudaMemPool.h"

TEST(CudaMemPool, TestPrimaryPoolState)
{
    auto const deviceCount = tensorrt_llm::common::getDeviceCount();
    for (int32_t deviceId = 0; deviceId < deviceCount; deviceId++)
    {
        auto const deviceSupportsMemoryPools = tensorrt_llm::runtime::CudaMemPool::supportsMemoryPool(deviceId);
        if (!deviceSupportsMemoryPools)
        {
            TLLM_LOG_INFO(
                "Testing primary memory pool state for device %i skipped as it does not support memory pools.");
            continue;
        }
        auto const primaryPool = tensorrt_llm::runtime::CudaMemPool::getPrimaryPoolForDevice(deviceId);
        if (primaryPool == nullptr)
        {
            FAIL() << "The device supports memory pools but the primary pool for it was not initialized!";
        }
        cudaMemPool_t cudaDefaultMemPool = nullptr;
        TLLM_CUDA_CHECK(::cudaDeviceGetDefaultMemPool(&cudaDefaultMemPool, deviceId));
        ASSERT_NE(primaryPool->getPool(), cudaDefaultMemPool)
            << "The primary TRTLLM device mem pool should NOT be the device's default memory pool.";
        cudaMemPool_t cudaCurrentMemPool = nullptr;
        TLLM_CUDA_CHECK(::cudaDeviceGetMemPool(&cudaCurrentMemPool, deviceId));
        ASSERT_NE(primaryPool->getPool(), cudaCurrentMemPool)
            << "The primary TRTLLM device mem pool should NOT be the device's current memory pool.";

        auto const primaryPoolASecondTime = tensorrt_llm::runtime::CudaMemPool::getPrimaryPoolForDevice(deviceId);
        ASSERT_EQ(primaryPoolASecondTime->getPool(), primaryPool->getPool())
            << "Getting the primary pool for the same device twice should return the same pool.";
    }
}
