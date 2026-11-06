/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

#pragma once

#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/kernels/customAllReduceKernels.h"
#include <c10/core/ScalarType.h>
#include <c10/util/Exception.h>
#include <c10/util/SmallVector.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <map>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <tuple>
#include <utility>
#include <vector>

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{
using kernels::AllReduceFusionOp;

constexpr size_t kAllReduceNumInputs = 6;

using AllReduceInputShape = c10::SmallVector<int64_t, 8>;
using AllReduceInputShapes = std::array<AllReduceInputShape, kAllReduceNumInputs>;

struct AllReduceTacticCacheKey
{
    std::vector<int64_t> group;
    AllReduceFusionOp fusionOp;
    bool inputUsesNcclWindow;
    c10::ScalarType dtype;
    AllReduceInputShapes staticInputShapes;

    bool operator<(AllReduceTacticCacheKey const& other) const
    {
        return std::tie(group, fusionOp, inputUsesNcclWindow, dtype, staticInputShapes)
            < std::tie(other.group, other.fusionOp, other.inputUsesNcclWindow, other.dtype, other.staticInputShapes);
    }
};

constexpr int64_t kDefaultAllReduceTuningCeiling = 16384;

inline std::vector<int64_t> makeAllReduceTuningBuckets(int64_t maxNumTokens)
{
    TORCH_CHECK(maxNumTokens > 0, "AllReduce tuning ceiling must be positive.");
    auto const ceiling = maxNumTokens;
    std::vector<int64_t> buckets;
    for (int64_t bucket = 1; bucket < ceiling;)
    {
        buckets.push_back(bucket);
        if (bucket > ceiling / 2)
        {
            break;
        }
        bucket *= 2;
    }
    buckets.push_back(ceiling);
    return buckets;
}

inline std::optional<size_t> getAllReduceBucketIndex(std::vector<int64_t> const& buckets, int64_t numTokens)
{
    if (buckets.empty() || numTokens < buckets.front() || numTokens > buckets.back())
    {
        return std::nullopt;
    }
    return std::upper_bound(buckets.begin(), buckets.end(), numTokens) - buckets.begin() - 1;
}

class AllReduceTacticCache
{
public:
    std::vector<int64_t> configure(int64_t maxNumTokens)
    {
        std::unique_lock lock(mMutex);
        if (mBuckets.empty() || mBuckets.back() != maxNumTokens)
        {
            auto buckets = makeAllReduceTuningBuckets(maxNumTokens);
            if (!mTactics.empty())
            {
                TLLM_LOG_WARNING(
                    "Changing the process-wide AllReduce tuning ceiling from %lld to %lld tokens clears "
                    "previously measured tactics. Operations without new measurements will use fallback tactics.",
                    static_cast<long long>(mBuckets.back()), static_cast<long long>(maxNumTokens));
            }
            mBuckets = std::move(buckets);
            mTactics.clear();
        }
        return mBuckets;
    }

    std::vector<int64_t> getBuckets()
    {
        std::unique_lock lock(mMutex);
        ensureDefaultBuckets();
        return mBuckets;
    }

    int64_t mapToBucket(int64_t numTokens) const
    {
        std::shared_lock lock(mMutex);
        auto const index = getAllReduceBucketIndex(mBuckets, numTokens);
        return index.has_value() ? mBuckets[*index] : -1;
    }

    void insert(AllReduceTacticCacheKey key, int64_t bucket, int64_t tactic)
    {
        std::unique_lock lock(mMutex);
        ensureDefaultBuckets();
        auto const index = getAllReduceBucketIndex(mBuckets, bucket);
        TORCH_CHECK(index.has_value() && mBuckets[*index] == bucket, "Invalid AllReduce autotuner bucket: ", bucket);
        auto const entry = mTactics.try_emplace(std::move(key), mBuckets.size()).first;
        entry->second[*index] = tactic;
    }

    std::optional<int64_t> find(AllReduceTacticCacheKey const& key, int64_t numTokens) const
    {
        std::shared_lock lock(mMutex);
        auto const index = getAllReduceBucketIndex(mBuckets, numTokens);
        auto const entry = mTactics.find(key);
        return !index.has_value() || entry == mTactics.end() ? std::nullopt : entry->second[*index];
    }

    void clear()
    {
        std::unique_lock lock(mMutex);
        mTactics.clear();
    }

private:
    // Called with mMutex held exclusively, only by standalone consumers that
    // need a table before an engine has configured its resolved token budget.
    void ensureDefaultBuckets()
    {
        if (mBuckets.empty())
        {
            mBuckets = makeAllReduceTuningBuckets(kDefaultAllReduceTuningCeiling);
        }
    }

    mutable std::shared_mutex mMutex;
    std::vector<int64_t> mBuckets;
    std::map<AllReduceTacticCacheKey, std::vector<std::optional<int64_t>>> mTactics;
};

} // namespace torch_ext

TRTLLM_NAMESPACE_END
