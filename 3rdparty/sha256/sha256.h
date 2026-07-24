// Copyright (c) 2014-present The Bitcoin Core developers
// Distributed under the MIT software license, see the accompanying
// file LICENSE or http://www.opensource.org/licenses/mit-license.php.
//
// Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// Modified by NVIDIA and redistributed under the MIT license (see LICENSE):
// removed the SHA256D64 double-hash declaration (unused by TensorRT-LLM).
// SPDX-License-Identifier: MIT

#ifndef BITCOIN_CRYPTO_SHA256_H
#define BITCOIN_CRYPTO_SHA256_H

#include <cstdint>
#include <cstdlib>
#include <string>

/** A hasher class for SHA-256. */
class CSHA256
{
private:
    uint32_t s[8];
    unsigned char buf[64];
    uint64_t bytes{0};

public:
    static const size_t OUTPUT_SIZE = 32;

    CSHA256();
    CSHA256& Write(const unsigned char* data, size_t len);
    void Finalize(unsigned char hash[OUTPUT_SIZE]);
    CSHA256& Reset();
};

namespace sha256_implementation {
enum UseImplementation : uint8_t {
    STANDARD = 0,
    USE_SSE4 = 1 << 0,
    USE_AVX2 = 1 << 1,
    USE_SHANI = 1 << 2,
    USE_SSE4_AND_AVX2 = USE_SSE4 | USE_AVX2,
    USE_SSE4_AND_SHANI = USE_SSE4 | USE_SHANI,
    USE_ALL = USE_SSE4 | USE_AVX2 | USE_SHANI,
};
}

/** Autodetect the best available SHA256 implementation.
 *  Returns the name of the implementation.
 */
std::string SHA256AutoDetect(sha256_implementation::UseImplementation use_implementation = sha256_implementation::USE_ALL);

#endif // BITCOIN_CRYPTO_SHA256_H
