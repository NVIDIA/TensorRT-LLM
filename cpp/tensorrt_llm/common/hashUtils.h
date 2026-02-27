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

#include <openssl/evp.h>

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

//! \brief SHA-256 based hasher for KV cache block hashing.
//!
//! Uses OpenSSL's EVP interface for SHA-256 hashing.
//! Produces a 32-byte (256-bit) digest.
//!
//! Example usage:
//! \code
//!   Hasher hasher(42);  // seed = 42
//!   hasher.update(100).update(200);
//!   auto hash = hasher.digestBytes();  // 32-byte SHA-256 digest
//! \endcode
class Hasher
{
public:
    static constexpr size_t kDigestSize = 32;

    //! Default constructor with seed = 0
    Hasher();

    //! Constructor with explicit seed
    explicit Hasher(uint64_t seed);

    //! Constructor with optional seed (for Python None support)
    explicit Hasher(std::optional<uint64_t> seed);

    ~Hasher();

    //! Copy constructor
    Hasher(Hasher const& other);

    //! Copy assignment
    Hasher& operator=(Hasher const& other);

    //! Move constructor
    Hasher(Hasher&& other) noexcept;

    //! Move assignment
    Hasher& operator=(Hasher&& other) noexcept;

    //! Update hash with a 64-bit integer value (chainable)
    Hasher& update(uint64_t value);

    //! Update hash with raw bytes (chainable)
    Hasher& update(void const* data, size_t size);

    //! Get the final 64-bit hash value (first 8 bytes of SHA-256 digest, little-endian)
    [[nodiscard]] uint64_t digest() const;

    //! Get the final hash as a 32-byte SHA-256 digest
    [[nodiscard]] std::array<uint8_t, kDigestSize> digestBytes() const;

private:
    EVP_MD_CTX* mCtx;

    void initCtx();
};

} // namespace common

TRTLLM_NAMESPACE_END
