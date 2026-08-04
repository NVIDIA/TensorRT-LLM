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

#include "tensorrt_llm/common/opUtils.h"
#include "tensorrt_llm/thop/thUtils.h"

#include <atomic>
#include <memory>
#include <shared_mutex>
#include <thread>
#include <tuple>
#include <unordered_map>

using namespace tensorrt_llm::torch_ext;

TEST(ThUtils, ConvertShape2D)
{
    at::Tensor a = at::ones({2, 5}, at::kInt);
    auto const shape = convert_shape(a);
    ASSERT_EQ(shape.d[0], 2);
    ASSERT_EQ(shape.d[1], 5);
    ASSERT_EQ(shape.nbDims, 2);
}

TEST(ThUtils, ConvertShape1D)
{
    at::Tensor a = at::ones({20}, at::kInt);
    auto const shape = convert_shape(a);
    ASSERT_EQ(shape.d[0], 20);
    ASSERT_EQ(shape.nbDims, 1);
}

TEST(AttentionOpCache, ConcurrentAccessNoCorruption)
{
    using namespace tensorrt_llm::common::op;

    using CacheKey = std::tuple<int, int>;
    using CacheValue = std::shared_ptr<int>;
    using CacheMap = std::unordered_map<CacheKey, CacheValue, OpCustomHash<CacheKey>>;

    CacheMap cache;
    std::shared_mutex cacheMutex;

    int constexpr kNumThreads = 16;
    int constexpr kNumDistinctKeys = 8;
    int constexpr kItersPerThread = 200;

    for (int i = 0; i < kNumDistinctKeys / 2; ++i)
    {
        auto key = std::make_tuple(i, i * 10);
        cache.try_emplace(key, std::make_shared<int>(i));
    }

    std::atomic<int> readHits{0};
    std::atomic<int> writeInserts{0};

    auto worker = [&](int threadId)
    {
        for (int iter = 0; iter < kItersPerThread; ++iter)
        {
            int keyIdx = (threadId + iter) % kNumDistinctKeys;
            auto key = std::make_tuple(keyIdx, keyIdx * 10);

            {
                std::shared_lock<std::shared_mutex> readLock{cacheMutex};
                auto it = cache.find(key);
                if (it != cache.end())
                {
                    ASSERT_NE(it->second, nullptr);
                    ASSERT_EQ(*it->second, keyIdx);
                    readHits.fetch_add(1, std::memory_order_relaxed);
                    continue;
                }
            }

            {
                std::unique_lock<std::shared_mutex> writeLock{cacheMutex};
                auto [it, inserted] = cache.try_emplace(key, std::make_shared<int>(keyIdx));
                ASSERT_NE(it->second, nullptr);
                ASSERT_EQ(*it->second, keyIdx);
                if (inserted)
                {
                    writeInserts.fetch_add(1, std::memory_order_relaxed);
                }
            }
        }
    };

    std::vector<std::thread> threads;
    threads.reserve(kNumThreads);
    for (int t = 0; t < kNumThreads; ++t)
    {
        threads.emplace_back(worker, t);
    }
    for (auto& t : threads)
    {
        t.join();
    }

    ASSERT_EQ(static_cast<int>(cache.size()), kNumDistinctKeys);

    for (int i = 0; i < kNumDistinctKeys; ++i)
    {
        auto key = std::make_tuple(i, i * 10);
        auto it = cache.find(key);
        ASSERT_NE(it, cache.end());
        ASSERT_NE(it->second, nullptr);
        ASSERT_EQ(*it->second, i);
    }
}
