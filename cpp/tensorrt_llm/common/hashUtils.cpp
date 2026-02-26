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
#include <stdexcept>
#include <utility>

TRTLLM_NAMESPACE_BEGIN

namespace common
{

void Hasher::initCtx()
{
    mCtx = EVP_MD_CTX_new();
    if (!mCtx)
    {
        throw std::runtime_error("Failed to create EVP_MD_CTX");
    }
    if (EVP_DigestInit_ex(mCtx, EVP_sha256(), nullptr) != 1)
    {
        EVP_MD_CTX_free(mCtx);
        mCtx = nullptr;
        throw std::runtime_error("Failed to initialize SHA-256 digest");
    }
}

Hasher::Hasher()
    : mCtx(nullptr)
{
    initCtx();
}

Hasher::Hasher(uint64_t seed)
    : mCtx(nullptr)
{
    initCtx();
    // Feed the seed as initial data (little-endian)
    uint8_t seedBytes[8];
    for (int i = 0; i < 8; ++i)
    {
        seedBytes[i] = static_cast<uint8_t>(seed & 0xFF);
        seed >>= 8;
    }
    update(seedBytes, 8);
}

Hasher::Hasher(std::optional<uint64_t> seed)
    : mCtx(nullptr)
{
    initCtx();
    if (seed.has_value())
    {
        uint64_t s = seed.value();
        uint8_t seedBytes[8];
        for (int i = 0; i < 8; ++i)
        {
            seedBytes[i] = static_cast<uint8_t>(s & 0xFF);
            s >>= 8;
        }
        update(seedBytes, 8);
    }
}

Hasher::~Hasher()
{
    if (mCtx)
    {
        EVP_MD_CTX_free(mCtx);
    }
}

Hasher::Hasher(Hasher const& other)
    : mCtx(nullptr)
{
    if (other.mCtx)
    {
        mCtx = EVP_MD_CTX_new();
        if (!mCtx || EVP_MD_CTX_copy_ex(mCtx, other.mCtx) != 1)
        {
            if (mCtx)
            {
                EVP_MD_CTX_free(mCtx);
                mCtx = nullptr;
            }
            throw std::runtime_error("Failed to copy EVP_MD_CTX");
        }
    }
}

Hasher& Hasher::operator=(Hasher const& other)
{
    if (this != &other)
    {
        Hasher tmp(other);
        std::swap(mCtx, tmp.mCtx);
    }
    return *this;
}

Hasher::Hasher(Hasher&& other) noexcept
    : mCtx(other.mCtx)
{
    other.mCtx = nullptr;
}

Hasher& Hasher::operator=(Hasher&& other) noexcept
{
    if (this != &other)
    {
        if (mCtx)
        {
            EVP_MD_CTX_free(mCtx);
        }
        mCtx = other.mCtx;
        other.mCtx = nullptr;
    }
    return *this;
}

Hasher& Hasher::update(uint64_t value)
{
    // Convert to little-endian bytes and feed into SHA-256
    uint8_t bytes[8];
    for (int i = 0; i < 8; ++i)
    {
        bytes[i] = static_cast<uint8_t>(value & 0xFF);
        value >>= 8;
    }
    return update(bytes, 8);
}

Hasher& Hasher::update(void const* data, size_t size)
{
    if (size == 0)
    {
        return *this;
    }
    if (EVP_DigestUpdate(mCtx, data, size) != 1)
    {
        throw std::runtime_error("Failed to update SHA-256 digest");
    }
    return *this;
}

std::array<uint8_t, Hasher::kDigestSize> Hasher::digestBytes() const
{
    // Finalize on a copy so the hasher remains usable
    EVP_MD_CTX* copyCtx = EVP_MD_CTX_new();
    if (!copyCtx || EVP_MD_CTX_copy_ex(copyCtx, mCtx) != 1)
    {
        if (copyCtx)
        {
            EVP_MD_CTX_free(copyCtx);
        }
        throw std::runtime_error("Failed to copy EVP_MD_CTX for digest");
    }

    std::array<uint8_t, kDigestSize> result;
    unsigned int len = 0;
    if (EVP_DigestFinal_ex(copyCtx, result.data(), &len) != 1)
    {
        EVP_MD_CTX_free(copyCtx);
        throw std::runtime_error("Failed to finalize SHA-256 digest");
    }
    EVP_MD_CTX_free(copyCtx);
    return result;
}

} // namespace common

TRTLLM_NAMESPACE_END
