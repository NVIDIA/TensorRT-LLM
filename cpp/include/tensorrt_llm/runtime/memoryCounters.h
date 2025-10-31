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

#pragma once

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/runtime/iBuffer.h"

#include <atomic>
#include <cstddef>
#include <limits>
#include <string>

namespace tensorrt_llm::runtime
{

class MemoryCounters
{
public:
    using SizeType32 = std::size_t;
    using DiffType = std::ptrdiff_t;

    MemoryCounters() = default;

    [[nodiscard]] SizeType32 getGpu() const
    {
        return mGpu;
    }

    [[nodiscard]] SizeType32 getCpu() const
    {
        return mCpu;
    }

    [[nodiscard]] SizeType32 getPinned() const
    {
        return mPinned;
    }

    [[nodiscard]] SizeType32 getUVM() const
    {
        return mUVM;
    }

    [[nodiscard]] SizeType32 getPinnedPool() const
    {
        return mPinnedPool;
    }

    [[nodiscard]] DiffType getGpuDiff() const
    {
        return mGpuDiff;
    }

    [[nodiscard]] DiffType getCpuDiff() const
    {
        return mCpuDiff;
    }

    [[nodiscard]] DiffType getPinnedDiff() const
    {
        return mPinnedDiff;
    }

    [[nodiscard]] DiffType getUVMDiff() const
    {
        return mUVMDiff;
    }

    [[nodiscard]] DiffType getPinnedPoolDiff() const
    {
        return mPinnedPoolDiff;
    }

    template <MemoryType T>
    struct always_false : std::false_type
    {
    };

    template <MemoryType T>
    void allocate(SizeType32 size)
    {
        if (size > static_cast<SizeType32>(std::numeric_limits<DiffType>::max()))
        {
            TLLM_THROW("Memory size too large for diff type: %zu", size);
        }
        auto const sizeDiff = static_cast<DiffType>(size);
        if constexpr (T == MemoryType::kGPU)
        {
            mGpu += size;
            mGpuDiff = sizeDiff;
        }
        else if constexpr (T == MemoryType::kCPU)
        {
            mCpu += size;
            mCpuDiff = sizeDiff;
        }
        else if constexpr (T == MemoryType::kPINNED)
        {
            mPinned += size;
            mPinnedDiff = sizeDiff;
        }
        else if constexpr (T == MemoryType::kUVM)
        {
            mUVM += size;
            mUVMDiff = sizeDiff;
        }
        else if constexpr (T == MemoryType::kPINNEDPOOL)
        {
            mPinnedPool += size;
            mPinnedPoolDiff = sizeDiff;
        }
        else
        {
            static_assert(always_false<T>::value, "Unknown memory type!");
        }
    }

    void allocate(MemoryType memoryType, SizeType32 size);

    template <MemoryType T>
    void deallocate(SizeType32 size)
    {
        if (size > static_cast<SizeType32>(std::numeric_limits<DiffType>::max()))
        {
            TLLM_THROW("Memory size too large for diff type: %zu", size);
        }
        auto const sizeDiff = -static_cast<DiffType>(size);
        if constexpr (T == MemoryType::kGPU)
        {
            mGpu -= size;
            mGpuDiff = sizeDiff;
        }
        else if constexpr (T == MemoryType::kCPU)
        {
            mCpu -= size;
            mCpuDiff = sizeDiff;
        }
        else if constexpr (T == MemoryType::kPINNED)
        {
            mPinned -= size;
            mPinnedDiff = sizeDiff;
        }
        else if constexpr (T == MemoryType::kUVM)
        {
            mUVM -= size;
            mUVMDiff = sizeDiff;
        }
        else if constexpr (T == MemoryType::kPINNEDPOOL)
        {
            mPinnedPool -= size;
            mPinnedPoolDiff = sizeDiff;
        }
        else
        {
            static_assert(always_false<T>::value, "Unknown memory type!");
        }
    }

    void deallocate(MemoryType memoryType, SizeType32 size);

    static MemoryCounters& getInstance();

    MemoryCounters(MemoryCounters const&) = delete;
    MemoryCounters& operator=(MemoryCounters const&) = delete;
    MemoryCounters(MemoryCounters&&) = delete;
    MemoryCounters& operator=(MemoryCounters&&) = delete;

    static std::string bytesToString(SizeType32 bytes, int precision = 2);

    static std::string bytesToString(DiffType bytes, int precision = 2);

    [[nodiscard]] std::string toString() const;

private:
    std::atomic<SizeType32> mGpu{}, mCpu{}, mPinned{}, mUVM{}, mPinnedPool{};
    std::atomic<DiffType> mGpuDiff{}, mCpuDiff{}, mPinnedDiff{}, mUVMDiff{}, mPinnedPoolDiff{};
};

} // namespace tensorrt_llm::runtime
