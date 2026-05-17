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

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <map>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
{

// ---------------------------------------------------------------------------
// Integer math helpers (mirrors _utils.py)
// ---------------------------------------------------------------------------

template <typename T>
[[nodiscard]] inline T divUp(T x, T y) noexcept
{
    return (x + y - 1) / y;
}

template <typename T>
[[nodiscard]] inline T roundUp(T x, T y) noexcept
{
    return divUp(x, y) * y;
}

template <typename T>
[[nodiscard]] inline T roundDown(T x, T y) noexcept
{
    return (x / y) * y;
}

template <typename T>
[[nodiscard]] inline T exactDiv(T x, T y)
{
    assert(x % y == 0);
    return x / y;
}

template <typename T>
[[nodiscard]] inline bool inRange(T x, T lower, T upper) noexcept
{
    return lower <= x && x < upper;
}

// Returns the intersection of [a.first, a.second) and [b.first, b.second),
// or {0,0} if disjoint (caller must check first < second).
template <typename T>
[[nodiscard]] inline std::pair<T, T> overlap(std::pair<T, T> a, std::pair<T, T> b) noexcept
{
    T lo = a.first > b.first ? a.first : b.first;
    T hi = a.second < b.second ? a.second : b.second;
    return {lo, hi};
}

// Extract an attribute from the first element and assert (debug) all elements agree.
// Mirrors Python's get_uniform_attribute(_utils.py:202).
template <typename Range, typename Func>
[[nodiscard]] auto getUniformAttribute(Range const& range, Func&& func) -> decltype(func(*range.begin()))
{
    auto it = range.begin();
    assert(it != range.end());
    auto result = func(*it);
    assert(std::all_of(range.begin(), range.end(), [&](auto const& item) { return func(item) == result; }));
    return result;
}

// Find the first index in [begin, end) where predicate is true.
// Returns distance(begin, end) if not found.
// Mirrors Python's find_index(_utils.py:366).
template <typename Iter, typename Pred>
[[nodiscard]] int findIndex(Iter begin, Iter end, Pred pred)
{
    return static_cast<int>(std::distance(begin, std::find_if(begin, end, pred)));
}

// Steal items matching predicate from the container and return them (stable).
// Single-pass O(n), mirrors Python's remove_if(_utils.py:174).
template <typename T, typename Pred>
std::vector<T> stealIf(std::vector<T>& original, Pred pred)
{
    std::vector<T> removed;
    size_t writeIdx = 0;
    for (size_t i = 0; i < original.size(); ++i)
    {
        if (pred(original[i]))
            removed.push_back(std::move(original[i]));
        else
            original[writeIdx++] = std::move(original[i]);
    }
    original.erase(original.begin() + static_cast<ptrdiff_t>(writeIdx), original.end());
    return removed;
}

// Group items by a classifier function, returning a map of key → vector of items.
// Mirrors Python's partition(_utils.py:195).
template <typename T, typename Classifier>
auto partition(std::vector<T> const& items, Classifier classifier)
    -> std::map<decltype(classifier(std::declval<T const&>())), std::vector<T>>
{
    using Key = decltype(classifier(std::declval<T const&>()));
    std::map<Key, std::vector<T>> result;
    for (auto const& item : items)
        result[classifier(item)].push_back(item);
    return result;
}

// Normalize a vector of values to a ratio vector summing to 1.0.
// Mirrors Python's typed_map(values, lambda x: x / total).
template <typename T>
[[nodiscard]] std::vector<float> normalizeToRatio(std::vector<T> const& values)
{
    auto total = std::accumulate(values.begin(), values.end(), static_cast<T>(0));
    assert(total > 0);
    std::vector<float> ratio(values.size());
    for (size_t i = 0; i < values.size(); ++i)
        ratio[i] = static_cast<float>(values[i]) / static_cast<float>(total);
    return ratio;
}

// ---------------------------------------------------------------------------
// HalfOpenRange — a half-open range [beg, end). Empty when beg >= end.
// Mirrors _utils.py::HalfOpenRange.
// ---------------------------------------------------------------------------
struct HalfOpenRange
{
    int beg = 0;
    int end = 0;

    [[nodiscard]] int length() const noexcept
    {
        return beg < end ? end - beg : 0;
    }

    [[nodiscard]] bool empty() const noexcept
    {
        return beg >= end;
    }

    explicit operator bool() const noexcept
    {
        return beg < end;
    }

    bool operator==(HalfOpenRange const& o) const noexcept
    {
        if (beg >= end && o.beg >= o.end)
            return true;
        return beg == o.beg && end == o.end;
    }

    bool operator!=(HalfOpenRange const& o) const noexcept
    {
        return !(*this == o);
    }

    // Membership test: is value in [beg, end)?
    // Mirrors Python's HalfOpenRange.__contains__.
    [[nodiscard]] bool contains(int value) const noexcept
    {
        return beg <= value && value < end;
    }
};

// Returns the intersection of two half-open ranges.
// The result may be empty (beg >= end), which is safe to chain.
[[nodiscard]] inline HalfOpenRange intersect(HalfOpenRange a, HalfOpenRange b) noexcept
{
    return {std::max(a.beg, b.beg), std::min(a.end, b.end)};
}

// ---------------------------------------------------------------------------
// DynamicBitset — resizable bitset using 64-bit words.
// Mirrors _utils.py::DynamicBitset.
// ---------------------------------------------------------------------------
class DynamicBitset
{
public:
    explicit DynamicBitset(int capacity)
        : mWords(static_cast<size_t>(divUp(capacity, 64)), uint64_t{0})
        , mNumSetBits(0)
    {
    }

    void set(int index)
    {
        if (!get(index))
        {
            mWords[static_cast<size_t>(index / 64)] |= (uint64_t{1} << (index % 64));
            ++mNumSetBits;
        }
    }

    [[nodiscard]] bool get(int index) const noexcept
    {
        return (mWords[static_cast<size_t>(index / 64)] & (uint64_t{1} << (index % 64))) != 0;
    }

    void clear(int index)
    {
        if (get(index))
        {
            mWords[static_cast<size_t>(index / 64)] &= ~(uint64_t{1} << (index % 64));
            --mNumSetBits;
        }
    }

    [[nodiscard]] int numSetBits() const noexcept
    {
        return mNumSetBits;
    }

    void resize(int newCapacity)
    {
        int oldWords = static_cast<int>(mWords.size());
        int newWords = divUp(newCapacity, 64);
        if (newWords > oldWords)
        {
            mWords.resize(static_cast<size_t>(newWords), uint64_t{0});
        }
        else if (newWords < oldWords)
        {
            mWords.resize(static_cast<size_t>(newWords));
            // mask the last partial word if needed
            if (newCapacity % 64 != 0)
            {
                mWords.back() &= (uint64_t{1} << (newCapacity % 64)) - 1;
            }
        }
    }

    // Returns true if any bit in [start, end) is set.
    [[nodiscard]] bool anySet(int start, int end) const noexcept
    {
        if (start >= end)
        {
            return false;
        }
        int startWord = start / 64;
        int endWord = (end - 1) / 64;
        uint64_t startMask = ~uint64_t{0} << (start % 64);
        if (startWord == endWord)
        {
            int bitsInWord = end % 64;
            uint64_t endMask = bitsInWord ? ((uint64_t{1} << bitsInWord) - 1) : ~uint64_t{0};
            return (mWords[static_cast<size_t>(startWord)] & startMask & endMask) != 0;
        }
        if (mWords[static_cast<size_t>(startWord)] & startMask)
        {
            return true;
        }
        for (int w = startWord + 1; w < endWord; ++w)
        {
            if (mWords[static_cast<size_t>(w)])
            {
                return true;
            }
        }
        int bitsInLastWord = end % 64;
        if (bitsInLastWord == 0)
        {
            return mWords[static_cast<size_t>(endWord)] != 0;
        }
        return (mWords[static_cast<size_t>(endWord)] & ((uint64_t{1} << bitsInLastWord) - 1)) != 0;
    }

private:
    std::vector<uint64_t> mWords;
    int mNumSetBits;
};

// ---------------------------------------------------------------------------
// Array2D — row-major 2D array with typed row/column indices.
// Mirrors _utils.py::Array2D.
// ---------------------------------------------------------------------------
template <typename T>
class Array2D
{
public:
    Array2D(int rows, int cols, T initVal = T{})
        : mData(static_cast<size_t>(rows * cols), initVal)
        , mCols(cols)
    {
    }

    T& operator()(int row, int col) noexcept
    {
        return mData[static_cast<size_t>(row * mCols + col)];
    }

    T const& operator()(int row, int col) const noexcept
    {
        return mData[static_cast<size_t>(row * mCols + col)];
    }

    [[nodiscard]] int rows() const noexcept
    {
        return static_cast<int>(mData.size()) / mCols;
    }

    [[nodiscard]] int cols() const noexcept
    {
        return mCols;
    }

    // Pointer to start of row (for slicing / iteration).
    T* rowData(int row) noexcept
    {
        return mData.data() + row * mCols;
    }

    T const* rowData(int row) const noexcept
    {
        return mData.data() + row * mCols;
    }

private:
    std::vector<T> mData;
    int mCols;
};

} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
