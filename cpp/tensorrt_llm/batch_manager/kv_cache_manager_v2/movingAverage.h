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

#pragma once

#include "tensorrt_llm/common/assert.h"

namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
{

// ---------------------------------------------------------------------------
// Exponential moving average (mirrors _core/_moving_average.py::MovingAverage).
// weight-corrected bias: avg += (value - avg) / weight, weight = 1 + decay*weight
// ---------------------------------------------------------------------------
class MovingAverage
{
public:
    explicit MovingAverage(double decay = 0.9999) noexcept
        : mDecay(decay)
        , mAvg(0.0)
        , mWeight(0.0)
        , mNumUpdates(0)
    {
    }

    double update(double value) noexcept
    {
        mWeight = 1.0 + mDecay * mWeight;
        mAvg += (value - mAvg) / mWeight;
        ++mNumUpdates;
        return mAvg;
    }

    double value() const noexcept
    {
        return mAvg;
    }

    int numUpdates() const noexcept
    {
        return mNumUpdates;
    }

private:
    double mDecay;
    double mAvg;
    double mWeight;
    int mNumUpdates;
};

// ---------------------------------------------------------------------------
// Simple arithmetic mean (mirrors _core/_moving_average.py::Average).
// ---------------------------------------------------------------------------
class Average
{
public:
    Average() noexcept
        : mSum(0.0)
        , mCount(0)
    {
    }

    void update(double value) noexcept
    {
        mSum += value;
        ++mCount;
    }

    double value() const noexcept
    {
        TLLM_CHECK_DEBUG(mCount > 0);
        return mSum / mCount;
    }

    int count() const noexcept
    {
        return mCount;
    }

private:
    double mSum;
    int mCount;
};

} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
