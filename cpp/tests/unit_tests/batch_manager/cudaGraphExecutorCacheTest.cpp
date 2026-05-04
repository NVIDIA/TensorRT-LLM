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

#include "tensorrt_llm/batch_manager/common.h"
#include "tensorrt_llm/batch_manager/utils/inflightBatchingUtils.h"

#include <gtest/gtest.h>

#include <memory>

namespace tb = tensorrt_llm::batch_manager;
namespace tbu = tensorrt_llm::batch_manager::utils;
using SizeType32 = tensorrt_llm::runtime::SizeType32;

namespace
{
// A default-constructed CudaGraphExecutor holds mInstance == nullptr, so its destructor
// is a no-op and these tests do not require an active CUDA context.
std::shared_ptr<tbu::CudaGraphExecutor> makeDummyExecutor()
{
    return std::make_shared<tbu::CudaGraphExecutor>();
}

tb::BatchState makeBatchState(SizeType32 numTokens)
{
    return tb::BatchState{/*numCtxRequests=*/0, /*numGenRequests=*/1, numTokens, /*maxKvCacheLength=*/256};
}
} // namespace

class CudaGraphExecutorCacheTest : public ::testing::Test // NOLINT(cppcoreguidelines-pro-type-member-init)
{
};

TEST_F(CudaGraphExecutorCacheTest, EmptyByDefault)
{
    tbu::CudaGraphExecutorCache cache(/*capacity=*/4);
    EXPECT_EQ(cache.size(), 0);
    EXPECT_FALSE(cache.get(makeBatchState(1)).has_value());
}

TEST_F(CudaGraphExecutorCacheTest, PutAndGetReturnsSameInstance)
{
    tbu::CudaGraphExecutorCache cache(/*capacity=*/4);

    auto bs = makeBatchState(1);
    auto exec = makeDummyExecutor();
    cache.put(bs, exec);

    ASSERT_EQ(cache.size(), 1);
    auto got = cache.get(bs);
    ASSERT_TRUE(got.has_value());
    EXPECT_EQ(got->get(), exec.get());
}

TEST_F(CudaGraphExecutorCacheTest, PutWithExistingKeyReplaces)
{
    tbu::CudaGraphExecutorCache cache(/*capacity=*/4);

    auto bs = makeBatchState(1);
    auto first = makeDummyExecutor();
    auto second = makeDummyExecutor();

    cache.put(bs, first);
    cache.put(bs, second);

    EXPECT_EQ(cache.size(), 1);
    auto got = cache.get(bs);
    ASSERT_TRUE(got.has_value());
    EXPECT_EQ(got->get(), second.get());
}

TEST_F(CudaGraphExecutorCacheTest, EvictsLeastRecentlyUsedAtCapacity)
{
    tbu::CudaGraphExecutorCache cache(/*capacity=*/2);

    auto bsA = makeBatchState(1);
    auto bsB = makeBatchState(2);
    auto bsC = makeBatchState(3);

    auto execA = makeDummyExecutor();
    auto execB = makeDummyExecutor();
    auto execC = makeDummyExecutor();

    cache.put(bsA, execA);
    cache.put(bsB, execB);
    ASSERT_EQ(cache.size(), 2);

    // Access A so that B becomes the LRU entry.
    EXPECT_TRUE(cache.get(bsA).has_value());

    // Inserting C must evict B (the LRU), not A (just touched).
    cache.put(bsC, execC);
    EXPECT_EQ(cache.size(), 2);
    EXPECT_TRUE(cache.get(bsA).has_value());
    EXPECT_FALSE(cache.get(bsB).has_value());
    EXPECT_TRUE(cache.get(bsC).has_value());
}

TEST_F(CudaGraphExecutorCacheTest, ClearDropsAllEntries)
{
    tbu::CudaGraphExecutorCache cache(/*capacity=*/4);

    auto bsA = makeBatchState(1);
    auto bsB = makeBatchState(2);
    auto bsC = makeBatchState(3);

    cache.put(bsA, makeDummyExecutor());
    cache.put(bsB, makeDummyExecutor());
    cache.put(bsC, makeDummyExecutor());
    ASSERT_EQ(cache.size(), 3);

    cache.clear();

    EXPECT_EQ(cache.size(), 0);
    EXPECT_FALSE(cache.get(bsA).has_value());
    EXPECT_FALSE(cache.get(bsB).has_value());
    EXPECT_FALSE(cache.get(bsC).has_value());

    // After clearing, the cache must remain functional (i.e. clear() must not
    // leave it in a broken state).
    auto execA2 = makeDummyExecutor();
    cache.put(bsA, execA2);
    EXPECT_EQ(cache.size(), 1);
    auto got = cache.get(bsA);
    ASSERT_TRUE(got.has_value());
    EXPECT_EQ(got->get(), execA2.get());
}

TEST_F(CudaGraphExecutorCacheTest, ClearReleasesExecutorOwnership)
{
    tbu::CudaGraphExecutorCache cache(/*capacity=*/4);

    auto exec = makeDummyExecutor();
    std::weak_ptr<tbu::CudaGraphExecutor> weak = exec;

    cache.put(makeBatchState(1), exec);
    exec.reset();
    // The cache still owns one strong reference at this point.
    ASSERT_FALSE(weak.expired());

    cache.clear();

    // After clear(), no strong references should remain. This guarantees that
    // ~CudaGraphExecutor (which calls cudaGraphExecDestroy) actually runs for
    // every cached entry - exactly what changeBeamWidth() relies on.
    EXPECT_TRUE(weak.expired());
}
