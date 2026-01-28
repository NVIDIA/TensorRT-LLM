/*
 * Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

#include <cstdint>
#include <cuda_runtime.h>

namespace tensorrt_llm::kernels
{

class KVCacheIndex
{
public:
    using UnderlyingType = std::int32_t;

    // Flag indicating KVCacheIndex refers to secondary pool
    static constexpr UnderlyingType kSecondaryPoolFlag = static_cast<UnderlyingType>(1)
        << (8 * sizeof(UnderlyingType) - 1);

    static constexpr UnderlyingType kNullFlag = static_cast<UnderlyingType>(~0UL);

    static const KVCacheIndex nullIndex;

    explicit KVCacheIndex(UnderlyingType value, bool isSecondary = false)
        : value{isSecondary ? value | kSecondaryPoolFlag : value}
    {
        TLLM_CHECK_DEBUG(value >= 0 && this->value != kNullFlag);
    }

    __host__ __device__ [[nodiscard]] UnderlyingType get() const
    {
        return value & (~kSecondaryPoolFlag);
    }

    __host__ __device__ [[nodiscard]] bool isPrimary() const
    {
        return (value & kSecondaryPoolFlag) == 0;
    }

    [[nodiscard]] constexpr bool isNull() const
    {
        return value == kNullFlag;
    }

private:
    UnderlyingType value;

    constexpr KVCacheIndex()
        : value{kNullFlag}
    {
    }
};

constexpr KVCacheIndex KVCacheIndex::nullIndex{};

} // namespace tensorrt_llm::kernels
