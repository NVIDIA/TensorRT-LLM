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

#include "tensorrt_llm/common/cudaDriverWrapper.h"
#include "tensorrt_llm/common/cudaUtils.h"

TEST(TestCudaDriverWrapper, TllmCuCheckFailingWithValidParametersDoesNotThrow)
{
    auto const deviceCount = tensorrt_llm::common::getDeviceCount();
    if (deviceCount == 0)
    {
        GTEST_SKIP() << "No CUDA devices found";
    }
    CUmemGenericAllocationHandle handle{};
    CUmemAllocationProp const prop{CUmemAllocationType::CU_MEM_ALLOCATION_TYPE_PINNED,
        CUmemAllocationHandleType::CU_MEM_HANDLE_TYPE_NONE,
        CUmemLocation{
            CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE,
            0,
        },
        nullptr};
    auto const granularity = tensorrt_llm::common::getAllocationGranularity();
    ASSERT_NO_THROW(TLLM_CU_CHECK(cuMemCreate(&handle, granularity * 16, &prop, 0)));
    ASSERT_NO_THROW(TLLM_CU_CHECK(cuMemRelease(handle)));
}

TEST(TestCudaDriverWrapper, TllmCuCheckFailingWithInvalidParametersThrows)
{
    auto const deviceCount = tensorrt_llm::common::getDeviceCount();
    if (deviceCount == 0)
    {
        GTEST_SKIP() << "No CUDA devices found";
    }
    CUmemGenericAllocationHandle handle{};
    CUmemAllocationProp const prop{CUmemAllocationType::CU_MEM_ALLOCATION_TYPE_PINNED,
        CUmemAllocationHandleType::CU_MEM_HANDLE_TYPE_NONE,
        CUmemLocation{
            CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE,
            0,
        },
        nullptr};
    ASSERT_THROW(TLLM_CU_CHECK(cuMemCreate(&handle, -1, &prop, 0ULL)), tensorrt_llm::common::TllmException);
}
