// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
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
