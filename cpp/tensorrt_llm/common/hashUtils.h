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

#include "tensorrt_llm/common/config.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>

TRTLLM_NAMESPACE_BEGIN

namespace common
{

inline size_t hash64(uint64_t value)
{
    value = (value ^ (value >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    value = (value ^ (value >> 27)) * UINT64_C(0x94d049bb133111eb);
    value = value ^ (value >> 31);
    return value;
}

inline size_t hash32(uint32_t value)
{
    value = ((value >> 16) ^ value) * 0x45d9f3b;
    value = ((value >> 16) ^ value) * 0x45d9f3b;
    value = (value >> 16) ^ value;
    return static_cast<size_t>(value);
}

//! \brief Fast 64-bit hasher using MurmurHash3-style mixing and Boost-style hash combining.
//!
//! Provides a drop-in replacement for SHA256 in KV cache block hashing with:
//! - 10-100x faster performance (non-cryptographic hash)
//! - 4x smaller output (8 bytes vs 32 bytes)
//! - Sufficient collision resistance for cache operations
//!
//! Example usage:
//! \code
//!   Hasher hasher(42);  // seed = 42
//!   hasher.update(100).update(200);
//!   uint64_t hash = hasher.digest();
//! \endcode
class Hasher
{
public:
    //! Default constructor with seed = 0
    Hasher();

    //! Constructor with explicit seed
    explicit Hasher(uint64_t seed);

    //! Constructor with optional seed (for Python None support)
    explicit Hasher(std::optional<uint64_t> seed);

    //! Update hash with a 64-bit integer value (chainable)
    Hasher& update(uint64_t value);

    //! Update hash with raw bytes (chainable)
    Hasher& update(void const* data, size_t size);

    //! Get the final 64-bit hash value
    [[nodiscard]] uint64_t digest() const;

    //! Get the final hash as an 8-byte array (little-endian)
    [[nodiscard]] std::array<uint8_t, 8> digestBytes() const;

private:
    uint64_t mHash;

    //! Combine a new hash value using Boost-style hash combining
    //! Formula: seed ^= hash + 0x9e3779b9 + (seed << 6) + (seed >> 2)
    void combineHash(uint64_t newHash);
};

// Inline simple methods
inline Hasher::Hasher()
    : mHash(0)
{
}

inline Hasher::Hasher(uint64_t seed)
    : mHash(seed)
{
}

inline Hasher::Hasher(std::optional<uint64_t> seed)
    : mHash(seed.value_or(0))
{
}

inline uint64_t Hasher::digest() const
{
    return mHash;
}

} // namespace common

TRTLLM_NAMESPACE_END
