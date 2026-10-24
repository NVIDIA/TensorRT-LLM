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

// One 16-element block of the BF16 -> NVFP4 activation quantization of nvfp4QuantKf.cu (E2M1 codes and one E4M3 scale
// per block, the bytes of invokeFP4Quantization<__nv_bfloat16, 16>(..., useUE8M0 = false, LINEAR, ...) on finite
// inputs), shared with every kernel that quantizes while it moves the data (the MoE A2A dispatch).
//
// Per block, as cvt_warp_fp16_to_fp4 computes it: amax = max |x|; SF = e4m3(gs * (amax * rcp(6))) (round to nearest
// even, saturating at 448); os = amax != 0 ? rcp(float(SF) * rcp(gs)) : 0 with rcp = rcp.approx.ftz; codes =
// e2m1(x * os) (round to nearest even, saturating at 6), the even element in the low nibble.
//
// The helpers exist in the SM10x device pass only (the conversions need the Blackwell instructions); a caller guards
// its use with TLLM_NVFP4_QUANT_KF_SM10X.

#pragma once

#include "tensorrt_llm/common/config.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000) && (__CUDA_ARCH__ < 1100) // SM10x
#define TLLM_NVFP4_QUANT_KF_SM10X 1
#else
#define TLLM_NVFP4_QUANT_KF_SM10X 0
#endif

TRTLLM_NAMESPACE_BEGIN

namespace kernels::nvfp4_quant_kf
{

constexpr int kBlockElems = 16;      // elements per scale
constexpr int kBlockInputBytes = 32; // 16 BF16 values
constexpr int kBlockCodeBytes = 8;   // 16 E2M1 codes

#if TLLM_NVFP4_QUANT_KF_SM10X

// 256-bit streaming load: the BF16 input is dead after this read, so L2 should evict it first.
__device__ __forceinline__ void loadBlock(uint8_t const* ptr, uint32_t (&r)[8])
{
    asm volatile("ld.global.nc.L1::no_allocate.L2::evict_first.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7])
                 : "l"(ptr));
}

// Per BF16 lane max(|a|, |b|); the sign bit of each lane is the xor of the input signs.
__device__ __forceinline__ uint32_t absMax2(uint32_t a, uint32_t b)
{
    uint32_t d;
    asm("max.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b));
    return d;
}

// The two FP32 products of a BF16x2 word and s, rounded to nearest even: {even element, odd element}.
__device__ __forceinline__ void scale2(uint32_t w, float s, float& even, float& odd)
{
    float const evenIn = __uint_as_float(w << 16);
    float const oddIn = __uint_as_float(w & 0xffff0000u);
    asm("{\n\t.reg .b64 a, b, d;\n\t"
        "mov.b64 a, {%2, %3};\n\t"
        "mov.b64 b, {%4, %4};\n\t"
        "mul.rn.f32x2 d, a, b;\n\t"
        "mov.b64 {%0, %1}, d;\n\t}"
        : "=f"(even), "=f"(odd)
        : "f"(evenIn), "f"(oddIn), "f"(s));
}

// Eight FP32 values (even element first) -> eight E2M1 codes in one word, element 0 in the lowest nibble.
__device__ __forceinline__ uint32_t toE2m1x8(float const (&v)[8])
{
    uint32_t r;
    asm("{\n\t.reg .b8 q0, q1, q2, q3;\n\t"
        "cvt.rn.satfinite.e2m1x2.f32 q0, %2, %1;\n\t"
        "cvt.rn.satfinite.e2m1x2.f32 q1, %4, %3;\n\t"
        "cvt.rn.satfinite.e2m1x2.f32 q2, %6, %5;\n\t"
        "cvt.rn.satfinite.e2m1x2.f32 q3, %8, %7;\n\t"
        "mov.b32 %0, {q0, q1, q2, q3};\n\t}"
        : "=r"(r)
        : "f"(v[0]), "f"(v[1]), "f"(v[2]), "f"(v[3]), "f"(v[4]), "f"(v[5]), "f"(v[6]), "f"(v[7]));
    return r;
}

// The production kernel's reciprocal (reciprocal_approximate_ftz in quantization.cuh).
__device__ __forceinline__ float rcpApproxFtz(float a)
{
    float r;
    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(a));
    return r;
}

//! The per-thread constants of a quantization: the global scale gs, rcp(gs) and rcp(6).
struct BlockConsts
{
    float gs;
    float rcpGs;
    float rcp6;
};

__device__ __forceinline__ BlockConsts makeBlockConsts(float gs)
{
    return BlockConsts{gs, rcpApproxFtz(gs), rcpApproxFtz(6.0f)};
}

//! The quantized block: sixteen E2M1 codes (the first eight in codesLo, element 0 in its lowest nibble) and the E4M3
//! scale byte.
struct QuantizedBlock
{
    uint32_t codesLo;
    uint32_t codesHi;
    uint8_t scale;
};

//! One 16-element block: r holds its 16 BF16 values.
__device__ __forceinline__ QuantizedBlock quantizeBlock(uint32_t const (&r)[8], BlockConsts const& c)
{
    uint32_t const m
        = absMax2(absMax2(absMax2(r[0], r[1]), absMax2(r[2], r[3])), absMax2(absMax2(r[4], r[5]), absMax2(r[6], r[7])));
    // The magnitude bits of finite BF16 values are ordered like the values.
    uint32_t const amaxBits = max(m & 0x7fffu, (m >> 16) & 0x7fffu);
    float const amax = __uint_as_float(amaxBits << 16);

    // SF = e4m3(gs * (amax * rcp(6))): two FP32 multiplications in this order, then rn with saturation at 448.
    float const sfValue = __fmul_rn(c.gs, __fmul_rn(amax, c.rcp6));
    uint16_t sfPair;
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(sfPair) : "f"(0.0f), "f"(sfValue));
    // float(SF), through FP16 (exact for every E4M3 value).
    uint32_t halfPair;
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(halfPair) : "h"(sfPair));
    float const sfFloat = __half2float(__ushort_as_half(static_cast<unsigned short>(halfPair & 0xffffu)));
    // A nonzero block whose SF rounds to 0 gets rcp(0) = inf, as in the production kernel. The zero test is the
    // production kernel's float comparison (vecMax != 0), so that the two also agree when built with -ftz=true.
    float const outputScale = amax != 0.0f ? rcpApproxFtz(__fmul_rn(sfFloat, c.rcpGs)) : 0.0f;

    float lo[8];
    float hi[8];
    scale2(r[0], outputScale, lo[0], lo[1]);
    scale2(r[1], outputScale, lo[2], lo[3]);
    scale2(r[2], outputScale, lo[4], lo[5]);
    scale2(r[3], outputScale, lo[6], lo[7]);
    scale2(r[4], outputScale, hi[0], hi[1]);
    scale2(r[5], outputScale, hi[2], hi[3]);
    scale2(r[6], outputScale, hi[4], hi[5]);
    scale2(r[7], outputScale, hi[6], hi[7]);
    return QuantizedBlock{toE2m1x8(lo), toE2m1x8(hi), static_cast<uint8_t>(sfPair & 0xffu)};
}

#endif // TLLM_NVFP4_QUANT_KF_SM10X

} // namespace kernels::nvfp4_quant_kf

TRTLLM_NAMESPACE_END
