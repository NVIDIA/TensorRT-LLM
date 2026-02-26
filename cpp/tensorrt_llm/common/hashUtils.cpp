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

#include "hashUtils.h"
#include <cstring>

TRTLLM_NAMESPACE_BEGIN

namespace common
{

void Hasher::combineHash(uint64_t newHash)
{
    // Boost-style hash combining algorithm
    // Reference: cpp/tensorrt_llm/batch_manager/kvCacheManager.cpp line 201
    // Formula provides excellent bit distribution through:
    // - Golden ratio constant (0x9e3779b9) for uniform distribution
    // - Bit shifts (<<6, >>2) for avalanche effect
    mHash ^= newHash + 0x9e3779b9 + (mHash << 6) + (mHash >> 2);
}

Hasher& Hasher::update(uint64_t value)
{
    // Hash the value using MurmurHash3 mixing, then combine
    combineHash(hash64(value));
    return *this;
}

Hasher& Hasher::update(void const* data, size_t size)
{
    if (size == 0)
    {
        return *this;
    }

    auto const* bytes = static_cast<uint8_t const*>(data);
    size_t numFullChunks = size / 8;
    size_t remainingBytes = size % 8;

    // Process 8-byte chunks
    for (size_t i = 0; i < numFullChunks; ++i)
    {
        uint64_t chunk;
        std::memcpy(&chunk, bytes + i * 8, 8);
        update(chunk);
    }

    // Process remaining bytes (if any)
    if (remainingBytes > 0)
    {
        uint64_t lastChunk = 0;
        std::memcpy(&lastChunk, bytes + numFullChunks * 8, remainingBytes);
        update(lastChunk);
    }

    return *this;
}

std::array<uint8_t, 8> Hasher::digestBytes() const
{
    // Convert 64-bit hash to little-endian byte array
    std::array<uint8_t, 8> result;
    uint64_t hash = mHash;
    for (size_t i = 0; i < 8; ++i)
    {
        result[i] = static_cast<uint8_t>(hash & 0xFF);
        hash >>= 8;
    }
    return result;
}

} // namespace common

TRTLLM_NAMESPACE_END
