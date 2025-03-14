/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "tensorrt_llm/executor/types.h"

namespace tensorrt_llm::executor
{

/// @brief An interval inclusive on both ends.
/// A single number interval is represented as [num, num].
template <typename T>
struct Interval
{
    T lowerEnd;
    T upperEnd;
};

template <typename T>
bool operator<(Interval<T> const& a, Interval<T> const& b)
{
    return a.lowerEnd < b.lowerEnd;
}

/// @brief A container to store unique numbers, represented as a vector of ordered and disjoint intervals.
template <typename NumType>
class IntervalSet
{
public:
    /// @brief Check if the given number is in set.
    bool contains(NumType num) const
    {
        // Binary search
        SizeType32 left = 0;
        SizeType32 right = static_cast<SizeType32>(mIntervals.size()) - 1;
        while (left <= right)
        {
            SizeType32 mid = left + (right - left) / 2;
            if (mIntervals[mid].lowerEnd <= num && num <= mIntervals[mid].upperEnd)
            {
                return true;
            }
            else if (num < mIntervals[mid].lowerEnd)
            {
                right = mid - 1;
            }
            else
            {
                left = mid + 1;
            }
        }
        return false;
    }

    /// @brief Insert a number into set. Do nothing if the number is already in the set.
    void insert(NumType num)
    {
        auto intervalToAdd = Interval<NumType>{num, num};

        if (mIntervals.size() == 0)
        {
            mIntervals.insert(mIntervals.begin(), intervalToAdd);
            mNumElements++;
            return;
        }

        // Iter is the first place in mIntervals such that num <= it.lowerEnd
        auto iter = std::lower_bound(mIntervals.begin(), mIntervals.end(), intervalToAdd);

        bool iterAtBegin = iter == mIntervals.begin();
        bool iterAtEnd = iter == mIntervals.end();

        if ((!iterAtEnd && iter->lowerEnd == num) || (!iterAtBegin && num <= (iter - 1)->upperEnd))
        {
            // Number falls within the current interval or previous interval. No need to add again.
            return;
        }

        if (!iterAtBegin && !iterAtEnd && (iter - 1)->upperEnd + 1 == num && iter->lowerEnd - 1 == num)
        {
            // Merge two adjacent intervals
            (iter - 1)->upperEnd = iter->upperEnd;
            mIntervals.erase(iter);
        }
        else if (!iterAtBegin && (iter - 1)->upperEnd + 1 == num)
        {
            // Number is adjacent to the upper end of the previous interval. Merge left.
            (iter - 1)->upperEnd = num;
        }
        else if (!iterAtEnd && iter->lowerEnd - 1 == num)
        {
            // Number is adjacent to the lower end of the current interval. Merge right.
            iter->lowerEnd = num;
        }
        else
        {
            mIntervals.insert(iter, intervalToAdd);
        }
        mNumElements++;
    }

    /// @brief Clear interval set and reset numElements to 0.
    void clear()
    {
        mIntervals.clear();
        mNumElements = 0;
    }

    /// @brief Return the size of the set.
    SizeType32 getNumElements() const
    {
        return mNumElements;
    }

    /// @brief Return the underlying mIntervals.
    std::vector<Interval<NumType>> const& getIntervals() const
    {
        return mIntervals;
    }

private:
    std::vector<Interval<NumType>> mIntervals;
    SizeType32 mNumElements{0};
};

} // namespace tensorrt_llm::executor
