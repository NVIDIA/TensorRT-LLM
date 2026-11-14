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

#include <gtest/gtest.h>

#include "tensorrt_llm/common/nativeMemory.h"

#include <cuda_runtime_api.h>

#include <cstdint>
#include <string>

using namespace tensorrt_llm::common;

namespace
{
NativeMemoryUsage usageOf(std::string const& label, int device)
{
    for (auto const& [name, usage] : getNativeMemoryUsage(device))
    {
        if (name == label)
        {
            return usage;
        }
    }
    return {};
}

int currentDevice()
{
    int device = -1;
    EXPECT_EQ(cudaGetDevice(&device), cudaSuccess);
    return device;
}
} // namespace

TEST(NativeMemoryTest, ExactRecordsBalance)
{
    char const* label = "test_exact";
    recordNativeAllocation(label, 3, 100);
    recordNativeAllocation(label, 3, 50);
    recordNativeAllocation(label, 4, 70);
    EXPECT_EQ(usageOf(label, 3).count, 2);
    EXPECT_EQ(usageOf(label, 3).bytes, 150);
    EXPECT_EQ(usageOf(label, 4).count, 1) << "records are per device";
    EXPECT_EQ(usageOf(label, 4).bytes, 70) << "records are per device";
    recordNativeRelease(label, 3, 100);
    recordNativeRelease(label, 3, 50);
    auto const usage = usageOf(label, 3);
    EXPECT_EQ(usage.count, 0);
    EXPECT_EQ(usage.bytes, 0);
    EXPECT_EQ(usageOf(label, 4).count, 1) << "releases on device 3 leave device 4 alone";
    recordNativeRelease(label, 4, 70);
    EXPECT_EQ(usageOf(label, 4).count, 0);
}

TEST(NativeMemoryTest, AdjustChangesBytesButNotCount)
{
    char const* label = "test_adjust";
    recordNativeAllocation(label, 3, 200);
    adjustNativeAllocation(label, 3, -150);
    auto usage = usageOf(label, 3);
    EXPECT_EQ(usage.count, 1);
    EXPECT_EQ(usage.bytes, 50);
    recordNativeRelease(label, 3, 50);
    usage = usageOf(label, 3);
    EXPECT_EQ(usage.count, 0);
    EXPECT_EQ(usage.bytes, 0);
}

TEST(NativeMemoryTest, UnreadableOrNegativeReadingMakesTheLabelUnmeasured)
{
    char const* label = "test_measured";
    int const device = currentDevice();

    // Something else freed more than this allocation took: free memory now exceeds the earlier reading, so the
    // drop is negative. The earlier reading must itself be valid (>= 0) to reach that rule.
    std::int64_t const lowerReading = currentDeviceFreeBytes() - (std::int64_t{1} << 30);
    ASSERT_GE(lowerReading, 0);
    auto const negative = recordNativeAllocationMeasured(label, lowerReading);
    EXPECT_EQ(negative.bytes, NativeMemoryRecord::kUnmeasured);
    // A later positive reading does not hide the unknown one.
    std::int64_t const freeBefore = currentDeviceFreeBytes();
    void* buffer = nullptr;
    ASSERT_EQ(cudaMalloc(&buffer, 64 << 20), cudaSuccess);
    auto const positive = recordNativeAllocationMeasured(label, freeBefore);
    EXPECT_GE(positive.bytes, 64 << 20);
    auto const unreadable = recordNativeAllocationMeasured(label, -1);
    EXPECT_EQ(unreadable.bytes, NativeMemoryRecord::kUnmeasured);

    auto usage = usageOf(label, device);
    EXPECT_EQ(usage.count, 3);
    EXPECT_EQ(usage.unmeasured, 2);
    EXPECT_EQ(usage.bytes, positive.bytes);

    // Each owner releases exactly what its own record added.
    recordNativeRelease(negative);
    recordNativeRelease(unreadable);
    usage = usageOf(label, device);
    EXPECT_EQ(usage.count, 1);
    EXPECT_EQ(usage.unmeasured, 0);
    EXPECT_EQ(usage.bytes, positive.bytes);

    ASSERT_EQ(cudaFree(buffer), cudaSuccess);
    recordNativeRelease(positive);
    usage = usageOf(label, device);
    EXPECT_EQ(usage.count, 0);
    EXPECT_EQ(usage.bytes, 0);
}

TEST(NativeMemoryTest, EmptyRecordReleasesNothing)
{
    recordNativeRelease(NativeMemoryRecord{});
    for (auto const& [name, usage] : getNativeMemoryUsage(-1))
    {
        ADD_FAILURE() << "unexpected entry " << name;
    }
}

TEST(NativeMemoryTest, LabelScopesNestAndRestore)
{
    EXPECT_EQ(currentNativeMemoryLabel(), nullptr);
    {
        NativeMemoryLabelScope const outer("outer");
        EXPECT_STREQ(currentNativeMemoryLabel(), "outer");
        {
            NativeMemoryLabelScope const inner("inner");
            EXPECT_STREQ(currentNativeMemoryLabel(), "inner");
        }
        EXPECT_STREQ(currentNativeMemoryLabel(), "outer");
    }
    EXPECT_EQ(currentNativeMemoryLabel(), nullptr);
}

TEST(NativeMemoryTest, LabelsAreUnique)
{
    auto const& all = nativeMemoryLabels::kAll;
    for (std::size_t i = 0; i < all.size(); ++i)
    {
        for (std::size_t j = i + 1; j < all.size(); ++j)
        {
            EXPECT_STRNE(all[i], all[j]);
        }
    }
}
