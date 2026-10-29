/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/kernels/doraScaling.h"
#include <cstdint>
#include <cuda_fp16.h>
#include <gtest/gtest.h>
#include <vector>

namespace
{

class DoraScalingTest : public ::testing::Test
{
protected:
    void TearDown() override
    {
        for (void* deviceBuffer : mDeviceBuffers)
        {
            cudaFree(deviceBuffer);
        }
    }

    template <typename T>
    T* allocateDeviceBuffer(std::vector<T> const& hostValues)
    {
        T* deviceBuffer = static_cast<T*>(allocateDeviceMemory(hostValues.size() * sizeof(T)));
        for (size_t i = 0; i < hostValues.size(); i++)
        {
            deviceBuffer[i] = hostValues[i];
        }
        return deviceBuffer;
    }

private:
    void* allocateDeviceMemory(size_t numberOfBytes)
    {
        void* deviceBuffer = nullptr;
        TLLM_CUDA_CHECK(cudaMallocManaged(&deviceBuffer, numberOfBytes));
        mDeviceBuffers.push_back(deviceBuffer);
        return deviceBuffer;
    }

    std::vector<void*> mDeviceBuffers;
};

TEST_F(DoraScalingTest, ScalesEachChannelByTheMagnitudeForItsModuleAndToken)
{
    std::vector<size_t> const moduleEnds = {2, 3, 6};
    size_t const numModules = moduleEnds.size();
    size_t const numTokens = 2;
    size_t const numChannels = moduleEnds.back();
    size_t const numel = numChannels * numTokens;
    std::vector<int64_t> const moduleEndsInt64(moduleEnds.begin(), moduleEnds.end());
    int64_t* deviceModuleEnds = allocateDeviceBuffer(moduleEndsInt64);

    std::vector<std::vector<float>> const magnitudes = {
        {2.0f, 4.0f},          // module 0, token 0
        {0.5f, 0.25f},         // module 0, token 1
        {8.0f},                // module 1, token 0
        {},                    // module 1, token 1
        {16.0f, 32.0f, 64.0f}, // module 2, token 0
        {3.0f, 5.0f, 7.0f},    // module 2, token 1
    };
    ASSERT_EQ(magnitudes.size(), numModules * numTokens);
    for (size_t moduleId = 0; moduleId < numModules; moduleId++)
    {
        size_t const moduleBegin = moduleId == 0 ? 0 : moduleEnds[moduleId - 1];
        size_t const moduleChannels = moduleEnds[moduleId] - moduleBegin;
        for (size_t tokenId = 0; tokenId < numTokens; tokenId++)
        {
            size_t const rowSize = magnitudes[(numTokens * moduleId) + tokenId].size();
            ASSERT_TRUE(rowSize == 0 || rowSize == moduleChannels);
        }
    }

    std::vector<half*> hostMagnitudeTable;
    for (std::vector<float> const& moduleTokenMagnitudes : magnitudes)
    {
        if (moduleTokenMagnitudes.empty())
        {
            hostMagnitudeTable.push_back(nullptr);
            continue;
        }
        std::vector<half> hostMagnitudeValues;
        for (float magnitude : moduleTokenMagnitudes)
        {
            hostMagnitudeValues.push_back(__float2half(magnitude));
        }
        hostMagnitudeTable.push_back(allocateDeviceBuffer(hostMagnitudeValues));
    }
    half** deviceMagnitudeTable = allocateDeviceBuffer(hostMagnitudeTable);

    std::vector<half> const hostInput(numel, __float2half(1.0f));
    half* deviceInput = allocateDeviceBuffer(hostInput);

    constexpr size_t kOverrunElements = 256;
    constexpr float kOverrunValue = -1.0f;
    std::vector<half> const hostOutput(numel + kOverrunElements, __float2half(kOverrunValue));
    half* deviceOutput = allocateDeviceBuffer(hostOutput);

    tensorrt_llm::kernels::tokenPerChannelScale<half>(
        numel, numModules, numTokens, deviceModuleEnds, deviceInput, deviceMagnitudeTable, deviceOutput, nullptr);
    TLLM_CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> const expectedOutput = {
        2.0f,
        4.0f,
        8.0f,
        16.0f,
        32.0f,
        64.0f,
        0.5f,
        0.25f,
        1.0f,
        3.0f,
        5.0f,
        7.0f,
    };
    ASSERT_EQ(expectedOutput.size(), numel);

    for (size_t i = 0; i < expectedOutput.size(); i++)
    {
        EXPECT_EQ(__half2float(deviceOutput[i]), expectedOutput[i]);
    }
    for (size_t i = 0; i < kOverrunElements; i++)
    {
        EXPECT_EQ(__half2float(deviceOutput[numel + i]), kOverrunValue);
    }
}
} // namespace