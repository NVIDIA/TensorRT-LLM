/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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

// The image's (vLLM fe1d68261) sampling arithmetic as host+device functions, op for op.
// Device: the PTX ops Triton 3.8 emits for gumbel.py / rejection_sampler_utils.py (tools/stock_ptx.py); host: the
// same IEEE ops emulated (fma/add with FTZ and RZ), so the CPU emulator and the CUDA kernels run one source.
// Only div.full.f32 and ex2.approx.f32 (hardware approximations) are not bit-emulated on the host.
#pragma once
#include "tensorrt_llm/modeling_v2/common/namespace.h"
#include <math.h>
#include <stdint.h>
#include <string.h>

#if defined(__CUDACC__)
#define DKR_HD __host__ __device__ __forceinline__
#else
#define DKR_HD inline
#endif

#if !defined(__CUDA_ARCH__)
#include <cfenv>
#endif

TRTLLM_MV2_NAMESPACE_BEGIN

namespace dkr
{
constexpr long long kDraftSalt = 1ll << 30;    // gumbel.py:20 _DRAFT_NOISE_SALT
constexpr uint32_t kRandMinBits = 0x2FFFFFFFu; // gumbel.py:14 _TL_RAND_MIN (4.6566127342e-10 as fp32)

DKR_HD float u2f(uint32_t u)
{
    float f;
    memcpy(&f, &u, 4);
    return f;
}

DKR_HD uint32_t f2u(float f)
{
    uint32_t u;
    memcpy(&u, &f, 4);
    return u;
}

DKR_HD float bf16f(uint16_t h)
{
    return u2f(static_cast<uint32_t>(h) << 16);
}

// ---------------------------------------------------------------------------------------------- IEEE op flavours
#if defined(__CUDA_ARCH__)
DKR_HD float fma_ftz(float a, float b, float c)
{
    float r;
    asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(r) : "f"(a), "f"(b), "f"(c));
    return r;
}

DKR_HD float add_rz_ftz(float a, float b)
{
    float r;
    asm("add.rz.ftz.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}

DKR_HD float mul_rn(float a, float b)
{
    float r;
    asm("mul.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}

DKR_HD float add_rn(float a, float b)
{
    float r;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}

DKR_HD float sub_rn(float a, float b)
{
    float r;
    asm("sub.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}

DKR_HD float i2f_rn(int32_t x)
{
    float r;
    asm("cvt.rn.f32.s32 %0, %1;" : "=f"(r) : "r"(x));
    return r;
}

// Triton lowers fp32 `/` to div.full.f32 (gumbel.py:111, rejection_sampler_utils.py:174/291/822, _temperature_kernel)
DKR_HD float div_full(float a, float b)
{
    float r;
    asm("div.full.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}

// tl.exp(x) = ex2.approx.f32(x * log2e)
DKR_HD float tl_exp(float x)
{
    float t = mul_rn(x, u2f(0x3FB8AA3Bu)), r;
    asm("ex2.approx.f32 %0, %1;" : "=f"(r) : "f"(t));
    return r;
}
#else
inline float ftz(float x)
{
    return fpclassify(x) == FP_SUBNORMAL ? copysignf(0.f, x) : x;
}

inline float fma_ftz(float a, float b, float c)
{
    return ftz(fmaf(ftz(a), ftz(b), ftz(c)));
}

inline float add_rz_ftz(float a, float b)
{
    float volatile va = ftz(a), vb = ftz(b);
    int const m = std::fegetround();
    std::fesetround(FE_TOWARDZERO);
    float volatile r = va + vb;
    std::fesetround(m);
    return ftz(r);
}

inline float mul_rn(float a, float b)
{
    float volatile r = a * b;
    return r;
}

inline float add_rn(float a, float b)
{
    float volatile r = a + b;
    return r;
}

inline float sub_rn(float a, float b)
{
    float volatile r = a - b;
    return r;
}

inline float i2f_rn(int32_t x)
{
    return static_cast<float>(x);
}

// host stand-in for div.full.f32 (not bit-emulated; CPU gates run at T in {0, 1} or compare against IEEE-div oracles)
inline float div_full(float a, float b)
{
    float volatile r = a / b;
    return r;
}

inline float tl_exp(float x)
{
    return exp2f(mul_rn(x, u2f(0x3FB8AA3Bu)));
}
#endif

// ---------------------------------------------------------------------------------------------- Triton Philox
DKR_HD uint32_t umulhi32(uint32_t a, uint32_t b)
{
    return static_cast<uint32_t>((static_cast<unsigned long long>(a) * b) >> 32);
}

// randint4x(seed, offset)[0] (triton/language/random.py:12-104): key = seed lo/hi, counter = (offset lo, hi, 0, 0)
DKR_HD uint32_t philox_c0(unsigned long long seed, unsigned long long offset)
{
    uint32_t c0 = static_cast<uint32_t>(offset), c1 = static_cast<uint32_t>(offset >> 32), c2 = 0u, c3 = 0u;
    uint32_t k0 = static_cast<uint32_t>(seed), k1 = static_cast<uint32_t>(seed >> 32);
#pragma unroll
    for (int i = 0; i < 10; ++i)
    {
        const uint32_t a0 = c0, a2 = c2;
        c0 = umulhi32(0xCD9E8D57u, a2) ^ c1 ^ k0;
        c2 = umulhi32(0xD2511F53u, a0) ^ c3 ^ k1;
        c1 = 0xCD9E8D57u * a2;
        c3 = 0xD2511F53u * a0;
        k0 += 0x9E3779B9u;
        k1 += 0xBB67AE85u;
    }
    return c0;
}

// uint_to_uniform_float (random.py:126-143) + gumbel.py:83-87 tl_rand32(includes_zero=False)
DKR_HD float uniform32(uint32_t c)
{
    int32_t x = static_cast<int32_t>(c);
    x = x < 0 ? ~x : x;
    float const u = mul_rn(i2f_rn(x), u2f(kRandMinBits));
    return fmaxf(u, u2f(kRandMinBits));
}

// gumbel.py:68-87 tl_rand32(seed, offset): rejection uniforms (rejection_sampler_utils.py:570)
DKR_HD float tl_rand32(long long seed, long long offset)
{
    return uniform32(philox_c0(static_cast<unsigned long long>(seed), static_cast<unsigned long long>(offset)));
}

// gumbel.py:119-120: per-row noise stream seed; drafting adds the salt to the position
DKR_HD uint32_t gumbel_seed(long long seed, long long pos, bool drafting)
{
    return philox_c0(
        static_cast<unsigned long long>(seed), static_cast<unsigned long long>(drafting ? pos + kDraftSalt : pos));
}

// ---------------------------------------------------------------------------------------------- libdevice (FTZ)
// __nv_log1pf as linked by Triton (PTX of _gumbel_sample_kernel, gumbel.py:127:37), valid for v > -1
DKR_HD float log1pf_tr(float v)
{
    float const t = add_rz_ftz(v, 1.0f);
    const int32_t e = static_cast<int32_t>(f2u(t) - 0x3F400000u) & static_cast<int32_t>(0xFF800000u);
    float const ef = mul_rn(i2f_rn(e), u2f(0x34000000u));
    float const s = fma_ftz(u2f(0x3E800000u), u2f(0x40800000u - static_cast<uint32_t>(e)), -1.0f);
    float const m = add_rn(s, u2f(f2u(v) - static_cast<uint32_t>(e)));
    float p = fma_ftz(u2f(0xBD39BF78u), m, u2f(0x3DD80012u));
    p = fma_ftz(p, m, u2f(0xBE0778E0u));
    p = fma_ftz(p, m, u2f(0x3E146475u));
    p = fma_ftz(p, m, u2f(0xBE2A68DDu));
    p = fma_ftz(p, m, u2f(0x3E4CAF9Eu));
    p = fma_ftz(p, m, u2f(0xBE800042u));
    p = fma_ftz(p, m, u2f(0x3EAAAAE6u));
    p = fma_ftz(p, m, -0.5f);
    float const q = fma_ftz(mul_rn(m, p), m, m);
    float r = fma_ftz(ef, u2f(0x3F317218u), q);
    if (!(f2u(v) < 0xBF800001u))
        r = fma_ftz(v, u2f(0x7F800000u), u2f(0x7F800000u));
    return r;
}

// __nv_logf(-l) as linked by Triton (gumbel.py:127:29): the operand arrives as the log1p result l <= 0 and the
// tiny-input rescale is applied to l, op for op as the PTX (setp.gt l, -FLT_MIN; mul l, -2^23)
DKR_HD float logf_neg_tr(float l)
{
    bool const tiny = l > u2f(0x80800000u);
    float const x = tiny ? mul_rn(l, u2f(0xCB000000u)) : -l;
    float const e0 = tiny ? -23.0f : 0.0f;
    const int32_t e = static_cast<int32_t>(f2u(x) - 0x3F2AAAABu) & static_cast<int32_t>(0xFF800000u);
    float const ef = fma_ftz(i2f_rn(e), u2f(0x34000000u), e0);
    float const f = add_rn(u2f(f2u(x) - static_cast<uint32_t>(e)), -1.0f);
    float q = fma_ftz(u2f(0xBE055027u), f, u2f(0x3E1039F6u));
    q = fma_ftz(q, f, u2f(0xBDF8CDCCu));
    q = fma_ftz(q, f, u2f(0x3E0F2955u));
    q = fma_ftz(q, f, u2f(0xBE2AD8B9u));
    q = fma_ftz(q, f, u2f(0x3E4CED0Bu));
    q = fma_ftz(q, f, u2f(0xBE7FFF22u));
    q = fma_ftz(q, f, u2f(0x3EAAAA78u));
    q = fma_ftz(q, f, -0.5f);
    float const g = fma_ftz(mul_rn(f, q), f, f);
    float r = fma_ftz(ef, u2f(0x3F317218u), g);
    if (!(f2u(x) < 0x7F800000u))
        r = fma_ftz(x, u2f(0x7F800000u), u2f(0x7F800000u));
    return x == 0.0f ? u2f(0xFF800000u) : r;
}

// L = log(-log1p(-u)); the Gumbel noise is -L and Triton folds `x + (-L)` into `x - L` (gumbel.py:127-128)
DKR_HD float gumbel_L(uint32_t gseed, uint32_t vocab_id)
{
    float const u
        = uniform32(philox_c0(static_cast<unsigned long long>(gseed), static_cast<unsigned long long>(vocab_id)));
    return logf_neg_tr(log1pf_tr(-u));
}

// gumbel_noised_argmax element value (gumbel.py:108-128): fp32 logit x, request temperature t
DKR_HD float gumbel_value(float x, float t, uint32_t gseed, uint32_t vocab_id, bool apply_temperature)
{
    if (t == 0.0f)
        return x;
    float const xt = apply_temperature ? div_full(x, t) : x;
    return sub_rn(xt, gumbel_L(gseed, vocab_id));
}

// ---------------------------------------------------------------------------------------------- argmax keys
// (value, lowest id wins) key; -0 == +0 as the Triton/torch compares; 0 is "empty"
DKR_HD unsigned long long make_key(float v, uint32_t id)
{
    uint32_t u = f2u(v);
    u = (u == 0x80000000u) ? 0u : u;
    u = (u & 0x80000000u) ? ~u : (u | 0x80000000u);
    return (static_cast<unsigned long long>(u) << 32) | static_cast<unsigned long long>(0xFFFFFFFFu - id);
}

DKR_HD uint32_t key_id(unsigned long long k)
{
    return 0xFFFFFFFFu - static_cast<uint32_t>(k);
}

DKR_HD float key_val(unsigned long long k)
{
    uint32_t u = static_cast<uint32_t>(k >> 32);
    u = (u & 0x80000000u) ? (u & 0x7FFFFFFFu) : ~u;
    return u2f(u);
}

// one exchange word: ordered value 32 | inverted id 18 | epoch 14 (single-copy atomic, dk_mk.cuh:243-247)
DKR_HD unsigned long long pack_word(unsigned long long key, unsigned long long epoch)
{
    const uint32_t id = key_id(key);
    return (key & 0xFFFFFFFF00000000ull) | (static_cast<unsigned long long>(0x3FFFFu - id) << 14) | (epoch & 0x3FFFull);
}

DKR_HD unsigned long long unpack_word(unsigned long long w)
{
    const uint32_t id = 0x3FFFFu - static_cast<uint32_t>((w >> 14) & 0x3FFFFull);
    return (w & 0xFFFFFFFF00000000ull) | static_cast<unsigned long long>(0xFFFFFFFFu - id);
}

DKR_HD bool word_ready(unsigned long long w, unsigned long long epoch)
{
    return (w & 0x3FFFull) == (epoch & 0x3FFFull);
}

// ---------------------------------------------------------------------------------------------- the reference
// sampler's murmur3 stream (gumbel.py); integer-exact here, kept for the port
DKR_HD uint32_t rotl32(uint32_t v, int s)
{
    return (v << s) | (v >> (32 - s));
}

DKR_HD uint32_t mm3_mix(uint32_t h, uint32_t k)
{
    k *= 0xCC9E2D51u;
    k = rotl32(k, 15);
    k *= 0x1B873593u;
    h ^= k;
    h = rotl32(h, 13);
    return h * 5u + 0xE6546B64u;
}

DKR_HD uint32_t murmur3_hash32(long long seed, long long pos, uint32_t offset, uint32_t domain)
{
    uint32_t h = domain;
    h = mm3_mix(h, static_cast<uint32_t>(seed & 0xFFFFFFFFll));
    h = mm3_mix(h, static_cast<uint32_t>((seed >> 32) & 0xFFFFFFFFll));
    h = mm3_mix(h, static_cast<uint32_t>(pos & 0xFFFFFFFFll));
    h = mm3_mix(h, offset);
    h ^= 16u;
    h ^= h >> 16;
    h *= 0x85EBCA6Bu;
    h ^= h >> 13;
    h *= 0xC2B2AE35u;
    return h ^ (h >> 16);
}
} // namespace dkr

TRTLLM_MV2_NAMESPACE_END
