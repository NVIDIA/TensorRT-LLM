// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "tensorrt_llm/thop/allreduceTacticCache.h"
#include <gtest/gtest.h>
#include <limits>
#include <set>

using namespace tensorrt_llm::torch_ext;

namespace
{
AllReduceTacticCacheKey makeKey()
{
    return {{0, 1}, AllReduceFusionOp::NONE, false, c10::ScalarType::BFloat16, {}};
}
} // namespace

TEST(AllReduceTacticCache, LazyInitialization)
{
    AllReduceTacticCache cache;
    EXPECT_EQ(cache.mapToBucket(1), -1);
    EXPECT_FALSE(cache.find(makeKey(), 1).has_value());
    EXPECT_EQ(cache.configure(40000).back(), 40000);
    // Configuring buckets does not manufacture measured winners.
    EXPECT_FALSE(cache.find(makeKey(), 40000).has_value());

    AllReduceTacticCache standalone;
    EXPECT_EQ(standalone.getBuckets().back(), 16384);
    standalone.insert(makeKey(), 16384, 8);
    EXPECT_EQ(standalone.find(makeKey(), 16384), 8);

    AllReduceTacticCache registration;
    registration.insert(makeKey(), 8192, 8);
    EXPECT_EQ(registration.find(makeKey(), 8192), 8);
}

TEST(AllReduceTacticCache, BucketGeneration)
{
    EXPECT_EQ(makeAllReduceTuningBuckets(1), (std::vector<int64_t>{1}));
    EXPECT_EQ(makeAllReduceTuningBuckets(3), (std::vector<int64_t>{1, 2, 3}));
    EXPECT_THROW(makeAllReduceTuningBuckets(0), c10::Error);
    EXPECT_THROW(makeAllReduceTuningBuckets(-1), c10::Error);
    for (int64_t const ceiling : {4096, 6000, 8192, 16384, 16385, 20000, 40000, 65536})
    {
        auto const buckets = makeAllReduceTuningBuckets(ceiling);
        EXPECT_EQ(buckets.back(), ceiling);
        for (size_t i = 0; i + 1 < buckets.size(); ++i)
        {
            EXPECT_EQ(buckets[i], int64_t{1} << i);
        }
        EXPECT_EQ(buckets.size(), std::set<int64_t>(buckets.begin(), buckets.end()).size());
    }
    auto const maximum = std::numeric_limits<int64_t>::max();
    auto const buckets = makeAllReduceTuningBuckets(maximum);
    ASSERT_EQ(buckets.size(), 64);
    EXPECT_EQ(buckets[62], int64_t{1} << 62);
    EXPECT_EQ(buckets.back(), maximum);
}

TEST(AllReduceTacticCache, LookupMatchesRegistrationBoundaries)
{
    for (int64_t const ceiling : {1, 6000, 8192, 16384, 20000, 40000, 65536})
    {
        AllReduceTacticCache cache;
        auto const buckets = cache.configure(ceiling);
        for (auto const bucket : buckets)
        {
            cache.insert(makeKey(), bucket, bucket);
        }
        for (int64_t tokens = 0; tokens <= ceiling + 1; ++tokens)
        {
            int64_t expected = -1;
            if (tokens >= 1 && tokens <= ceiling)
            {
                for (auto const bucket : buckets)
                {
                    if (bucket <= tokens)
                    {
                        expected = bucket;
                    }
                }
            }
            EXPECT_EQ(cache.mapToBucket(tokens), expected);
            EXPECT_EQ(cache.find(makeKey(), tokens).value_or(-1), expected);
        }
    }
}

TEST(AllReduceTacticCache, ReconfigurationAndInvalidRegistration)
{
    AllReduceTacticCache cache;
    cache.configure(40000);
    cache.insert(makeKey(), 40000, 8);
    cache.configure(40000);
    EXPECT_EQ(cache.find(makeKey(), 40000), 8);
    EXPECT_THROW(cache.configure(0), c10::Error);
    EXPECT_EQ(cache.find(makeKey(), 40000), 8);
    EXPECT_THROW(cache.insert(makeKey(), 30000, 8), c10::Error);
    EXPECT_THROW(cache.insert(makeKey(), 0, 8), c10::Error);
    EXPECT_THROW(cache.insert(makeKey(), 40001, 8), c10::Error);
    cache.configure(65536);
    EXPECT_FALSE(cache.find(makeKey(), 40000).has_value());
    cache.insert(makeKey(), 65536, 9);
    cache.configure(4096);
    EXPECT_FALSE(cache.find(makeKey(), 4096).has_value());
    cache.insert(makeKey(), 4096, 8);
    cache.clear();
    EXPECT_FALSE(cache.find(makeKey(), 4096).has_value());
    EXPECT_EQ(cache.mapToBucket(4096), 4096);
}

TEST(AllReduceTacticCache, CompatibleOperationsShareOnlyMeasuredRange)
{
    AllReduceTacticCache cache;
    cache.configure(8192);
    auto const decoder = makeKey();
    auto const merger = decoder;
    cache.insert(decoder, 8192, 8);
    EXPECT_EQ(cache.find(merger, 8192), 8);
    EXPECT_FALSE(cache.find(merger, 8193).has_value());
    EXPECT_FALSE(cache.find(merger, 16383).has_value());
    EXPECT_FALSE(cache.find(merger, 4096).has_value());

    auto different = decoder;
    different.group = {2, 3};
    EXPECT_FALSE(cache.find(different, 8192).has_value());
    different = decoder;
    different.inputUsesNcclWindow = true;
    EXPECT_FALSE(cache.find(different, 8192).has_value());
    different = decoder;
    different.dtype = c10::ScalarType::Float;
    EXPECT_FALSE(cache.find(different, 8192).has_value());
    different = decoder;
    different.fusionOp = AllReduceFusionOp::RESIDUAL_RMS_NORM;
    EXPECT_FALSE(cache.find(different, 8192).has_value());
    different = decoder;
    different.staticInputShapes[0].push_back(7168);
    EXPECT_FALSE(cache.find(different, 8192).has_value());
}
