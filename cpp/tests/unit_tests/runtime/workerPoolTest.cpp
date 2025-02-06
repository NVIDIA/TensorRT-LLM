/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.
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

#include "tensorrt_llm/runtime/workerPool.h"

#include <gtest/gtest.h>
#include <random>

namespace tensorrt_llm::runtime
{

TEST(WorkerPool, basic)
{
    WorkerPool pool(2);

    auto fn = []() { return 12345; };
    auto resultFuture = pool.enqueue(fn);

    auto fn2 = []() { return 12.345f; };
    auto f2 = pool.enqueue(fn2);

    auto fn3 = []() { return 40.78f; };
    auto f3 = pool.enqueue(fn3);

    auto r1 = resultFuture.get();
    auto r2 = f2.get();
    auto r3 = f3.get();

    EXPECT_EQ(12345, r1);
    EXPECT_FLOAT_EQ(12.345f, r2);
    EXPECT_FLOAT_EQ(40.78f, r3);
}

TEST(WorkerPool, voidReturn)
{
    WorkerPool pool(2);

    int32_t returnVal1 = 0;
    int32_t returnVal2 = 0;
    int32_t returnVal3 = 0;

    auto fn1 = [&returnVal1]() { returnVal1 = 10001; };
    auto f1 = pool.enqueue(fn1);

    auto fn2 = [&returnVal2]() { returnVal2 = 10002; };
    auto f2 = pool.enqueue(fn2);

    auto fn3 = [&returnVal3]() { returnVal3 = 10003; };
    auto f3 = pool.enqueue(fn3);

    f1.get();
    f2.get();
    f3.get();

    EXPECT_EQ(returnVal1, 10001);
    EXPECT_EQ(returnVal2, 10002);
    EXPECT_EQ(returnVal3, 10003);
}

class WorkerPoolTest : public ::testing::TestWithParam<std::tuple<int, int>>
{
protected:
    void SetUp() override
    {
        mNumTasks = std::get<0>(GetParam());
        mNumWorkers = std::get<1>(GetParam());
        pool = std::make_unique<WorkerPool>(mNumWorkers);
    }

    int mNumTasks;
    int mNumWorkers;
    std::unique_ptr<WorkerPool> pool;
};

TEST_P(WorkerPoolTest, ScheduleTasks)
{
    std::vector<std::future<void>> futures;
    std::random_device randomDevice;
    std::mt19937 generator(randomDevice());
    std::uniform_int_distribution<> distribution(1, 5);

    for (int i = 0; i < mNumTasks; ++i)
    {
        futures.push_back(
            pool->enqueue([&]() { std::this_thread::sleep_for(std::chrono::milliseconds(distribution(generator))); }));
    }

    for (auto& f : futures)
    {
        f.get();
    }

    // This is a smoke test to try and catch threading and synchronization issues by stress testing. No assertion.
}

INSTANTIATE_TEST_SUITE_P(WorkerPoolTests, WorkerPoolTest,
    ::testing::Combine(::testing::Values(1, 2, 4, 8, 16, 32, 64, 128), // Range for number of tasks
        ::testing::Values(1, 2, 4, 8, 16, 32, 64, 128)                 // Range for number of workers
        ));

} // namespace tensorrt_llm::runtime
