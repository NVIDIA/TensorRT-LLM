/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "tensorrt_llm/kernels/preQuantScaleKernel.h"

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace
{

class DeviceBuffer
{
public:
    explicit DeviceBuffer(size_t bytes)
        : mData(nullptr)
        , mStatus(cudaMalloc(&mData, bytes))
    {
    }

    ~DeviceBuffer()
    {
        if (mData != nullptr)
        {
            cudaFree(mData);
        }
    }

    DeviceBuffer(DeviceBuffer const&) = delete;
    DeviceBuffer& operator=(DeviceBuffer const&) = delete;

    cudaError_t status() const
    {
        return mStatus;
    }

    template <typename T>
    T* get() const
    {
        return static_cast<T*>(mData);
    }

private:
    void* mData;
    cudaError_t mStatus;
};

void runScaleBoundaryCase(bool perExpert, int rows, int cols, int rowsPerBlock, int64_t validRows)
{
    constexpr int kNumExperts = 4;
    constexpr float kInputValue = 2.0F;
    constexpr float kSentinel = -1.0F;
    constexpr std::array<float, kNumExperts> kExpertScales{0.5F, 1.0F, 1.5F, 2.0F};

    int const storageRows = ((rows + rowsPerBlock - 1) / rowsPerBlock) * rowsPerBlock;
    size_t const numElements = static_cast<size_t>(storageRows) * cols;
    size_t const numScales = static_cast<size_t>(perExpert ? kNumExperts : 1) * cols;
    // Padding to a complete CTA makes any tail writes observable in the sentinel rows.
    std::vector<half> hostInput(numElements, __float2half(kInputValue));
    std::vector<half> hostOutput(numElements, __float2half(kSentinel));
    std::vector<half> hostScales(numScales);
    for (int expert = 0; expert < (perExpert ? kNumExperts : 1); ++expert)
    {
        float const scale = perExpert ? kExpertScales[expert] : kExpertScales[0];
        for (int col = 0; col < cols; ++col)
        {
            hostScales[static_cast<size_t>(expert) * cols + col] = __float2half(scale);
        }
    }

    DeviceBuffer deviceInput(numElements * sizeof(half));
    DeviceBuffer deviceOutput(numElements * sizeof(half));
    DeviceBuffer deviceScales(numScales * sizeof(half));
    DeviceBuffer deviceOffsets(kNumExperts * sizeof(int64_t));
    DeviceBuffer deviceValidRows(sizeof(int64_t));
    ASSERT_EQ(deviceInput.status(), cudaSuccess);
    ASSERT_EQ(deviceOutput.status(), cudaSuccess);
    ASSERT_EQ(deviceScales.status(), cudaSuccess);
    if (perExpert)
    {
        ASSERT_EQ(deviceOffsets.status(), cudaSuccess);
    }
    if (validRows >= 0)
    {
        ASSERT_EQ(deviceValidRows.status(), cudaSuccess);
    }

    ASSERT_EQ(cudaMemcpy(deviceInput.get<half>(), hostInput.data(), numElements * sizeof(half), cudaMemcpyHostToDevice),
        cudaSuccess);
    ASSERT_EQ(
        cudaMemcpy(deviceOutput.get<half>(), hostOutput.data(), numElements * sizeof(half), cudaMemcpyHostToDevice),
        cudaSuccess);
    ASSERT_EQ(cudaMemcpy(deviceScales.get<half>(), hostScales.data(), numScales * sizeof(half), cudaMemcpyHostToDevice),
        cudaSuccess);

    // These boundaries cut through both four-row and eight-row CTA groups.
    constexpr std::array<int64_t, kNumExperts> hostOffsets{0, 1001, 2003, 3001};
    if (perExpert)
    {
        ASSERT_EQ(
            cudaMemcpy(deviceOffsets.get<int64_t>(), hostOffsets.data(), sizeof(hostOffsets), cudaMemcpyHostToDevice),
            cudaSuccess);
    }
    int64_t const* validRowsPtr = nullptr;
    if (validRows >= 0)
    {
        ASSERT_EQ(cudaMemcpy(deviceValidRows.get<int64_t>(), &validRows, sizeof(validRows), cudaMemcpyHostToDevice),
            cudaSuccess);
        validRowsPtr = deviceValidRows.get<int64_t>();
    }

    if (perExpert)
    {
        tensorrt_llm::kernels::apply_per_channel_scale_per_expert_kernel_launcher<half, half>(deviceOutput.get<half>(),
            deviceInput.get<half>(), deviceScales.get<half>(), rows, cols, deviceOffsets.get<int64_t>(), kNumExperts,
            validRowsPtr, 0);
    }
    else
    {
        tensorrt_llm::kernels::apply_per_channel_scale_kernel_launcher<half, half>(
            deviceOutput.get<half>(), deviceInput.get<half>(), deviceScales.get<half>(), rows, cols, validRowsPtr, 0);
    }

    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    ASSERT_EQ(
        cudaMemcpy(hostOutput.data(), deviceOutput.get<half>(), numElements * sizeof(half), cudaMemcpyDeviceToHost),
        cudaSuccess);

    int64_t const expectedValidRows = validRows < 0 ? rows : validRows;
    size_t mismatchCount = 0;
    int firstMismatchRow = -1;
    int firstMismatchCol = -1;
    float firstActual = 0.0F;
    float firstExpected = 0.0F;
    for (int row = 0; row < storageRows; ++row)
    {
        int expert = 0;
        while (perExpert && expert + 1 < kNumExperts && row >= hostOffsets[expert + 1])
        {
            ++expert;
        }
        float const expectedScale
            = row < expectedValidRows ? (perExpert ? kExpertScales[expert] : kExpertScales[0]) : kSentinel;
        float const expectedValue = row < expectedValidRows ? kInputValue * expectedScale : kSentinel;
        for (int col = 0; col < cols; ++col)
        {
            float const actual = __half2float(hostOutput[static_cast<size_t>(row) * cols + col]);
            if (actual != expectedValue)
            {
                if (mismatchCount == 0)
                {
                    firstMismatchRow = row;
                    firstMismatchCol = col;
                    firstActual = actual;
                    firstExpected = expectedValue;
                }
                ++mismatchCount;
            }
        }
    }
    EXPECT_EQ(mismatchCount, 0U) << "first mismatch at row " << firstMismatchRow << ", col " << firstMismatchCol
                                 << ": got " << firstActual << ", expected " << firstExpected;
}

} // namespace

TEST(PreQuantScaleKernelTest, OrdinaryPathBoundsFourRowTail)
{
    // This shape selects four rows per CTA and leaves one row in the final CTA.
    constexpr int kRows = 4097;
    constexpr int kCols = 1024;
    constexpr int kRowsPerBlock = 4;
    runScaleBoundaryCase(false, kRows, kCols, kRowsPerBlock, -1);
}

TEST(PreQuantScaleKernelTest, OrdinaryPathStopsAtValidTokenBoundary)
{
    // The last four-row CTA starts before this valid-token count but straddles it.
    constexpr int kRows = 4097;
    constexpr int kCols = 1024;
    constexpr int kRowsPerBlock = 4;
    constexpr int64_t kValidRows = 4094;
    runScaleBoundaryCase(false, kRows, kCols, kRowsPerBlock, kValidRows);
}

TEST(PreQuantScaleKernelTest, PerExpertPathBoundsFourRowTail)
{
    constexpr int kRows = 4097;
    constexpr int kCols = 1024;
    constexpr int kRowsPerBlock = 4;
    runScaleBoundaryCase(true, kRows, kCols, kRowsPerBlock, -1);
}

TEST(PreQuantScaleKernelTest, PerExpertPathStopsAtValidTokenBoundary)
{
    constexpr int kRows = 4097;
    constexpr int kCols = 1024;
    constexpr int kRowsPerBlock = 4;
    constexpr int64_t kValidRows = 4094;
    runScaleBoundaryCase(true, kRows, kCols, kRowsPerBlock, kValidRows);
}

TEST(PreQuantScaleKernelTest, PerExpertPathBoundsEightRowTail)
{
    // This shape selects eight rows per CTA and leaves one row in the final CTA.
    constexpr int kRows = 4097;
    constexpr int kCols = 4096;
    constexpr int kRowsPerBlock = 8;
    runScaleBoundaryCase(true, kRows, kCols, kRowsPerBlock, -1);
}
