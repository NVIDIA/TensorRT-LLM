/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: NVIDIA TensorRT Source Code License Agreement
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include "tensorrt_llm/batch_manager/utils/staticThreadPool.h"
#include <gmock/gmock.h>
#include <gtest/gtest.h>

using namespace tensorrt_llm::batch_manager::utils;

// ---------------------------------------
//          staticThreadPoolTest
// ---------------------------------------

class staticThreadPoolTest : public ::testing::Test // NOLINT(cppcoreguidelines-pro-type-member-init)
{
public:
    void SetUp() override {}

    void TearDown() override {}

    int add(int a, int b)
    {
        return a + b;
    }
};

TEST_F(staticThreadPoolTest, EmptyFunction)
{
    for (const std::size_t numThreads : {1, 5, 10})
    {
        StaticThreadPool pool{numThreads};
        for (int i = 0; i < 10; ++i)
        {
            auto res = pool.execute([] {});
            res.get();
        }
    }
}

TEST_F(staticThreadPoolTest, ReturnValueCheck)
{
    for (const std::size_t numThreads : {1, 5, 10})
    {
        StaticThreadPool pool{numThreads};
        std::vector<std::future<int>> results;
        for (int i = 0; i < 10; ++i)
        {
            results.emplace_back(pool.execute([i]() -> int { return i + 1; }));
        }
        for (int i = 0; i < 10; ++i)
        {
            TLLM_CHECK(results.at(i).get() == i + 1);
        }
    }
}

TEST_F(staticThreadPoolTest, MemberFunction)
{
    for (const std::size_t numThreads : {1, 5, 10})
    {
        StaticThreadPool pool{numThreads};
        std::vector<std::future<int>> results;
        for (int i = 0; i < 10; ++i)
        {
            results.emplace_back(pool.execute(&staticThreadPoolTest::add, this, i, 1));
        }
        for (int i = 0; i < 10; ++i)
        {
            TLLM_CHECK(results.at(i).get() == i + 1);
        }
    }
}
