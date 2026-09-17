// SPDX-FileCopyrightText: Copyright (c) 2014-present The Bitcoin Core developers
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
// THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS IN THE SOFTWARE.
//
// Minimal big-endian read/write helpers for the vendored Bitcoin Core SHA-256
// (sha256.cpp). This NVIDIA-authored header replaces the upstream
// <crypto/common.h> / <compat/endian.h> / <compat/byteswap.h> dependency chain
// (which required C++20) so the vendored sources build as plain C++17. The
// helper names and byte order match the upstream common.h functions they
// replace, so the SHA-256 output is unchanged.

#ifndef TENSORRT_LLM_3RDPARTY_SHA256_ENDIAN_H
#define TENSORRT_LLM_3RDPARTY_SHA256_ENDIAN_H

#include <cstdint>

inline uint32_t ReadBE32(unsigned char const* ptr)
{
    return (static_cast<uint32_t>(ptr[0]) << 24) | (static_cast<uint32_t>(ptr[1]) << 16)
        | (static_cast<uint32_t>(ptr[2]) << 8) | static_cast<uint32_t>(ptr[3]);
}

inline void WriteBE32(unsigned char* ptr, uint32_t x)
{
    ptr[0] = static_cast<unsigned char>(x >> 24);
    ptr[1] = static_cast<unsigned char>(x >> 16);
    ptr[2] = static_cast<unsigned char>(x >> 8);
    ptr[3] = static_cast<unsigned char>(x);
}

inline void WriteBE64(unsigned char* ptr, uint64_t x)
{
    ptr[0] = static_cast<unsigned char>(x >> 56);
    ptr[1] = static_cast<unsigned char>(x >> 48);
    ptr[2] = static_cast<unsigned char>(x >> 40);
    ptr[3] = static_cast<unsigned char>(x >> 32);
    ptr[4] = static_cast<unsigned char>(x >> 24);
    ptr[5] = static_cast<unsigned char>(x >> 16);
    ptr[6] = static_cast<unsigned char>(x >> 8);
    ptr[7] = static_cast<unsigned char>(x);
}

#endif // TENSORRT_LLM_3RDPARTY_SHA256_ENDIAN_H
