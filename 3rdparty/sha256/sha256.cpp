// Copyright (c) 2014-present The Bitcoin Core developers
// Distributed under the MIT software license, see the accompanying
// file LICENSE or http://www.opensource.org/licenses/mit-license.php.
//
// Portions Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// Modified by NVIDIA and redistributed under the MIT license (see LICENSE):
//   - Reduced to the single-block CSHA256 hasher used by TensorRT-LLM.
//   - Removed the SHA256D64 double-hash API and the SSE4/SSE4.1/AVX2 multi-block
//     transforms; runtime dispatch selects scalar / x86 SHA-NI / ARMv8 crypto.
//   - Replaced <crypto/common.h> / <compat/*> endian helpers with sha256_endian.h.
//   - CPU detection via CPUID leaf 7 (x86) / getauxval (aarch64); plain C++17 (no <bit>).
// SPDX-License-Identifier: MIT

#include "sha256.h"
#include "sha256_endian.h"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <string>

#if defined(__x86_64__) || defined(__amd64__) || defined(__i386__)
#include <cpuid.h>
namespace sha256_x86_shani {
void Transform(uint32_t* s, const unsigned char* chunk, size_t blocks);
}
#endif

#if defined(__aarch64__)
#include <sys/auxv.h>
#include <asm/hwcap.h>
namespace sha256_arm_shani {
void Transform(uint32_t* s, const unsigned char* chunk, size_t blocks);
}
#endif

// Internal implementation code.
namespace
{
/// Internal SHA-256 implementation.
namespace sha256
{
uint32_t inline Ch(uint32_t x, uint32_t y, uint32_t z) { return z ^ (x & (y ^ z)); }
uint32_t inline Maj(uint32_t x, uint32_t y, uint32_t z) { return (x & y) | (z & (x | y)); }
uint32_t inline Sigma0(uint32_t x) { return (x >> 2 | x << 30) ^ (x >> 13 | x << 19) ^ (x >> 22 | x << 10); }
uint32_t inline Sigma1(uint32_t x) { return (x >> 6 | x << 26) ^ (x >> 11 | x << 21) ^ (x >> 25 | x << 7); }
uint32_t inline sigma0(uint32_t x) { return (x >> 7 | x << 25) ^ (x >> 18 | x << 14) ^ (x >> 3); }
uint32_t inline sigma1(uint32_t x) { return (x >> 17 | x << 15) ^ (x >> 19 | x << 13) ^ (x >> 10); }

/** One round of SHA-256. */
void inline Round(uint32_t a, uint32_t b, uint32_t c, uint32_t& d, uint32_t e, uint32_t f, uint32_t g, uint32_t& h, uint32_t k)
{
    uint32_t t1 = h + Sigma1(e) + Ch(e, f, g) + k;
    uint32_t t2 = Sigma0(a) + Maj(a, b, c);
    d += t1;
    h = t1 + t2;
}

/** Initialize SHA-256 state. */
void inline Initialize(uint32_t* s)
{
    s[0] = 0x6a09e667ul;
    s[1] = 0xbb67ae85ul;
    s[2] = 0x3c6ef372ul;
    s[3] = 0xa54ff53aul;
    s[4] = 0x510e527ful;
    s[5] = 0x9b05688cul;
    s[6] = 0x1f83d9abul;
    s[7] = 0x5be0cd19ul;
}

/** Perform a number of SHA-256 transformations, processing 64-byte chunks. */
void Transform(uint32_t* s, const unsigned char* chunk, size_t blocks)
{
    while (blocks--) {
        uint32_t a = s[0], b = s[1], c = s[2], d = s[3], e = s[4], f = s[5], g = s[6], h = s[7];
        uint32_t w0, w1, w2, w3, w4, w5, w6, w7, w8, w9, w10, w11, w12, w13, w14, w15;

        Round(a, b, c, d, e, f, g, h, 0x428a2f98 + (w0 = ReadBE32(chunk + 0)));
        Round(h, a, b, c, d, e, f, g, 0x71374491 + (w1 = ReadBE32(chunk + 4)));
        Round(g, h, a, b, c, d, e, f, 0xb5c0fbcf + (w2 = ReadBE32(chunk + 8)));
        Round(f, g, h, a, b, c, d, e, 0xe9b5dba5 + (w3 = ReadBE32(chunk + 12)));
        Round(e, f, g, h, a, b, c, d, 0x3956c25b + (w4 = ReadBE32(chunk + 16)));
        Round(d, e, f, g, h, a, b, c, 0x59f111f1 + (w5 = ReadBE32(chunk + 20)));
        Round(c, d, e, f, g, h, a, b, 0x923f82a4 + (w6 = ReadBE32(chunk + 24)));
        Round(b, c, d, e, f, g, h, a, 0xab1c5ed5 + (w7 = ReadBE32(chunk + 28)));
        Round(a, b, c, d, e, f, g, h, 0xd807aa98 + (w8 = ReadBE32(chunk + 32)));
        Round(h, a, b, c, d, e, f, g, 0x12835b01 + (w9 = ReadBE32(chunk + 36)));
        Round(g, h, a, b, c, d, e, f, 0x243185be + (w10 = ReadBE32(chunk + 40)));
        Round(f, g, h, a, b, c, d, e, 0x550c7dc3 + (w11 = ReadBE32(chunk + 44)));
        Round(e, f, g, h, a, b, c, d, 0x72be5d74 + (w12 = ReadBE32(chunk + 48)));
        Round(d, e, f, g, h, a, b, c, 0x80deb1fe + (w13 = ReadBE32(chunk + 52)));
        Round(c, d, e, f, g, h, a, b, 0x9bdc06a7 + (w14 = ReadBE32(chunk + 56)));
        Round(b, c, d, e, f, g, h, a, 0xc19bf174 + (w15 = ReadBE32(chunk + 60)));

        Round(a, b, c, d, e, f, g, h, 0xe49b69c1 + (w0 += sigma1(w14) + w9 + sigma0(w1)));
        Round(h, a, b, c, d, e, f, g, 0xefbe4786 + (w1 += sigma1(w15) + w10 + sigma0(w2)));
        Round(g, h, a, b, c, d, e, f, 0x0fc19dc6 + (w2 += sigma1(w0) + w11 + sigma0(w3)));
        Round(f, g, h, a, b, c, d, e, 0x240ca1cc + (w3 += sigma1(w1) + w12 + sigma0(w4)));
        Round(e, f, g, h, a, b, c, d, 0x2de92c6f + (w4 += sigma1(w2) + w13 + sigma0(w5)));
        Round(d, e, f, g, h, a, b, c, 0x4a7484aa + (w5 += sigma1(w3) + w14 + sigma0(w6)));
        Round(c, d, e, f, g, h, a, b, 0x5cb0a9dc + (w6 += sigma1(w4) + w15 + sigma0(w7)));
        Round(b, c, d, e, f, g, h, a, 0x76f988da + (w7 += sigma1(w5) + w0 + sigma0(w8)));
        Round(a, b, c, d, e, f, g, h, 0x983e5152 + (w8 += sigma1(w6) + w1 + sigma0(w9)));
        Round(h, a, b, c, d, e, f, g, 0xa831c66d + (w9 += sigma1(w7) + w2 + sigma0(w10)));
        Round(g, h, a, b, c, d, e, f, 0xb00327c8 + (w10 += sigma1(w8) + w3 + sigma0(w11)));
        Round(f, g, h, a, b, c, d, e, 0xbf597fc7 + (w11 += sigma1(w9) + w4 + sigma0(w12)));
        Round(e, f, g, h, a, b, c, d, 0xc6e00bf3 + (w12 += sigma1(w10) + w5 + sigma0(w13)));
        Round(d, e, f, g, h, a, b, c, 0xd5a79147 + (w13 += sigma1(w11) + w6 + sigma0(w14)));
        Round(c, d, e, f, g, h, a, b, 0x06ca6351 + (w14 += sigma1(w12) + w7 + sigma0(w15)));
        Round(b, c, d, e, f, g, h, a, 0x14292967 + (w15 += sigma1(w13) + w8 + sigma0(w0)));

        Round(a, b, c, d, e, f, g, h, 0x27b70a85 + (w0 += sigma1(w14) + w9 + sigma0(w1)));
        Round(h, a, b, c, d, e, f, g, 0x2e1b2138 + (w1 += sigma1(w15) + w10 + sigma0(w2)));
        Round(g, h, a, b, c, d, e, f, 0x4d2c6dfc + (w2 += sigma1(w0) + w11 + sigma0(w3)));
        Round(f, g, h, a, b, c, d, e, 0x53380d13 + (w3 += sigma1(w1) + w12 + sigma0(w4)));
        Round(e, f, g, h, a, b, c, d, 0x650a7354 + (w4 += sigma1(w2) + w13 + sigma0(w5)));
        Round(d, e, f, g, h, a, b, c, 0x766a0abb + (w5 += sigma1(w3) + w14 + sigma0(w6)));
        Round(c, d, e, f, g, h, a, b, 0x81c2c92e + (w6 += sigma1(w4) + w15 + sigma0(w7)));
        Round(b, c, d, e, f, g, h, a, 0x92722c85 + (w7 += sigma1(w5) + w0 + sigma0(w8)));
        Round(a, b, c, d, e, f, g, h, 0xa2bfe8a1 + (w8 += sigma1(w6) + w1 + sigma0(w9)));
        Round(h, a, b, c, d, e, f, g, 0xa81a664b + (w9 += sigma1(w7) + w2 + sigma0(w10)));
        Round(g, h, a, b, c, d, e, f, 0xc24b8b70 + (w10 += sigma1(w8) + w3 + sigma0(w11)));
        Round(f, g, h, a, b, c, d, e, 0xc76c51a3 + (w11 += sigma1(w9) + w4 + sigma0(w12)));
        Round(e, f, g, h, a, b, c, d, 0xd192e819 + (w12 += sigma1(w10) + w5 + sigma0(w13)));
        Round(d, e, f, g, h, a, b, c, 0xd6990624 + (w13 += sigma1(w11) + w6 + sigma0(w14)));
        Round(c, d, e, f, g, h, a, b, 0xf40e3585 + (w14 += sigma1(w12) + w7 + sigma0(w15)));
        Round(b, c, d, e, f, g, h, a, 0x106aa070 + (w15 += sigma1(w13) + w8 + sigma0(w0)));

        Round(a, b, c, d, e, f, g, h, 0x19a4c116 + (w0 += sigma1(w14) + w9 + sigma0(w1)));
        Round(h, a, b, c, d, e, f, g, 0x1e376c08 + (w1 += sigma1(w15) + w10 + sigma0(w2)));
        Round(g, h, a, b, c, d, e, f, 0x2748774c + (w2 += sigma1(w0) + w11 + sigma0(w3)));
        Round(f, g, h, a, b, c, d, e, 0x34b0bcb5 + (w3 += sigma1(w1) + w12 + sigma0(w4)));
        Round(e, f, g, h, a, b, c, d, 0x391c0cb3 + (w4 += sigma1(w2) + w13 + sigma0(w5)));
        Round(d, e, f, g, h, a, b, c, 0x4ed8aa4a + (w5 += sigma1(w3) + w14 + sigma0(w6)));
        Round(c, d, e, f, g, h, a, b, 0x5b9cca4f + (w6 += sigma1(w4) + w15 + sigma0(w7)));
        Round(b, c, d, e, f, g, h, a, 0x682e6ff3 + (w7 += sigma1(w5) + w0 + sigma0(w8)));
        Round(a, b, c, d, e, f, g, h, 0x748f82ee + (w8 += sigma1(w6) + w1 + sigma0(w9)));
        Round(h, a, b, c, d, e, f, g, 0x78a5636f + (w9 += sigma1(w7) + w2 + sigma0(w10)));
        Round(g, h, a, b, c, d, e, f, 0x84c87814 + (w10 += sigma1(w8) + w3 + sigma0(w11)));
        Round(f, g, h, a, b, c, d, e, 0x8cc70208 + (w11 += sigma1(w9) + w4 + sigma0(w12)));
        Round(e, f, g, h, a, b, c, d, 0x90befffa + (w12 += sigma1(w10) + w5 + sigma0(w13)));
        Round(d, e, f, g, h, a, b, c, 0xa4506ceb + (w13 += sigma1(w11) + w6 + sigma0(w14)));
        Round(c, d, e, f, g, h, a, b, 0xbef9a3f7 + (w14 + sigma1(w12) + w7 + sigma0(w15)));
        Round(b, c, d, e, f, g, h, a, 0xc67178f2 + (w15 + sigma1(w13) + w8 + sigma0(w0)));

        s[0] += a;
        s[1] += b;
        s[2] += c;
        s[3] += d;
        s[4] += e;
        s[5] += f;
        s[6] += g;
        s[7] += h;
        chunk += 64;
    }
}

} // namespace sha256

typedef void (*TransformType)(uint32_t*, const unsigned char*, size_t);
TransformType Transform = sha256::Transform;

[[maybe_unused]] bool SelfTest() {
    // Input state (equal to the initial SHA256 state)
    static const uint32_t init[8] = {
        0x6a09e667ul, 0xbb67ae85ul, 0x3c6ef372ul, 0xa54ff53aul, 0x510e527ful, 0x9b05688cul, 0x1f83d9abul, 0x5be0cd19ul
    };
    // Some random input data to test with
    static const unsigned char data[641] = "-" // Intentionally not aligned
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do "
        "eiusmod tempor incididunt ut labore et dolore magna aliqua. Et m"
        "olestie ac feugiat sed lectus vestibulum mattis ullamcorper. Mor"
        "bi blandit cursus risus at ultrices mi tempus imperdiet nulla. N"
        "unc congue nisi vita suscipit tellus mauris. Imperdiet proin fer"
        "mentum leo vel orci. Massa tempor nec feugiat nisl pretium fusce"
        " id velit. Telus in metus vulputate eu scelerisque felis. Mi tem"
        "pus imperdiet nulla malesuada pellentesque. Tristique magna sit.";
    // Expected output state for hashing the i*64 first input bytes above (excluding SHA256 padding).
    static const uint32_t result[9][8] = {
        {0x6a09e667ul, 0xbb67ae85ul, 0x3c6ef372ul, 0xa54ff53aul, 0x510e527ful, 0x9b05688cul, 0x1f83d9abul, 0x5be0cd19ul},
        {0x91f8ec6bul, 0x4da10fe3ul, 0x1c9c292cul, 0x45e18185ul, 0x435cc111ul, 0x3ca26f09ul, 0xeb954caeul, 0x402a7069ul},
        {0xcabea5acul, 0x374fb97cul, 0x182ad996ul, 0x7bd69cbful, 0x450ff900ul, 0xc1d2be8aul, 0x6a41d505ul, 0xe6212dc3ul},
        {0xbcff09d6ul, 0x3e76f36eul, 0x3ecb2501ul, 0x78866e97ul, 0xe1c1e2fdul, 0x32f4eafful, 0x8aa6c4e5ul, 0xdfc024bcul},
        {0xa08c5d94ul, 0x0a862f93ul, 0x6b7f2f40ul, 0x8f9fae76ul, 0x6d40439ful, 0x79dcee0cul, 0x3e39ff3aul, 0xdc3bdbb1ul},
        {0x216a0895ul, 0x9f1a3662ul, 0xe99946f9ul, 0x87ba4364ul, 0x0fb5db2cul, 0x12bed3d3ul, 0x6689c0c7ul, 0x292f1b04ul},
        {0xca3067f8ul, 0xbc8c2656ul, 0x37cb7e0dul, 0x9b6b8b0ful, 0x46dc380bul, 0xf1287f57ul, 0xc42e4b23ul, 0x3fefe94dul},
        {0x3e4c4039ul, 0xbb6fca8cul, 0x6f27d2f7ul, 0x301e44a4ul, 0x8352ba14ul, 0x5769ce37ul, 0x48a1155ful, 0xc0e1c4c6ul},
        {0xfe2fa9ddul, 0x69d0862bul, 0x1ae0db23ul, 0x471f9244ul, 0xf55c0145ul, 0xc30f9c3bul, 0x40a84ea0ul, 0x5b8a266cul},
    };
    // Test Transform() for 0 through 8 transformations.
    for (size_t i = 0; i <= 8; ++i) {
        uint32_t state[8];
        std::copy(init, init + 8, state);
        Transform(state, data + 1, i);
        if (!std::equal(state, state + 8, result[i])) return false;
    }

    return true;
}
} // namespace

std::string SHA256AutoDetect(sha256_implementation::UseImplementation use_implementation)
{
    std::string ret = "standard";
    Transform = sha256::Transform;

#if defined(__x86_64__) || defined(__amd64__) || defined(__i386__)
    if (use_implementation & sha256_implementation::USE_SHANI) {
        // Intel SHA extensions: CPUID.(EAX=7,ECX=0):EBX[bit 29]. Portable across
        // GCC and Clang (unlike __builtin_cpu_supports("sha")).
        unsigned int eax = 0, ebx = 0, ecx = 0, edx = 0;
        if (__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx) && (ebx & (1u << 29))) {
            Transform = sha256_x86_shani::Transform;
            ret = "x86_shani";
        }
    }
#endif

#if defined(__aarch64__)
    if ((use_implementation & sha256_implementation::USE_SHANI) && (getauxval(AT_HWCAP) & HWCAP_SHA2)) {
        Transform = sha256_arm_shani::Transform;
        ret = "arm_shani";
    }
#endif

    assert(SelfTest());
    return ret;
}

////// SHA-256

CSHA256::CSHA256()
{
    sha256::Initialize(s);
}

CSHA256& CSHA256::Write(const unsigned char* data, size_t len)
{
    const unsigned char* end = data + len;
    size_t bufsize = bytes % 64;
    if (bufsize && bufsize + len >= 64) {
        // Fill the buffer, and process it.
        memcpy(buf + bufsize, data, 64 - bufsize);
        bytes += 64 - bufsize;
        data += 64 - bufsize;
        Transform(s, buf, 1);
        bufsize = 0;
    }
    if (end - data >= 64) {
        size_t blocks = (end - data) / 64;
        Transform(s, data, blocks);
        data += 64 * blocks;
        bytes += 64 * blocks;
    }
    if (end > data) {
        // Fill the buffer with what remains.
        memcpy(buf + bufsize, data, end - data);
        bytes += end - data;
    }
    return *this;
}

void CSHA256::Finalize(unsigned char hash[OUTPUT_SIZE])
{
    static const unsigned char pad[64] = {0x80};
    unsigned char sizedesc[8];
    WriteBE64(sizedesc, bytes << 3);
    Write(pad, 1 + ((119 - (bytes % 64)) % 64));
    Write(sizedesc, 8);
    WriteBE32(hash, s[0]);
    WriteBE32(hash + 4, s[1]);
    WriteBE32(hash + 8, s[2]);
    WriteBE32(hash + 12, s[3]);
    WriteBE32(hash + 16, s[4]);
    WriteBE32(hash + 20, s[5]);
    WriteBE32(hash + 24, s[6]);
    WriteBE32(hash + 28, s[7]);
}

CSHA256& CSHA256::Reset()
{
    bytes = 0;
    sha256::Initialize(s);
    return *this;
}
