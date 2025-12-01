/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
