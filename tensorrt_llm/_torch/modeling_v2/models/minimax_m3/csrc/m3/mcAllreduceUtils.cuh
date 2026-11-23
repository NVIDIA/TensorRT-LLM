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

#include "tensorrt_llm/modeling_v2/common/namespace.h"
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

TRTLLM_MV2_NAMESPACE_BEGIN

namespace kernels
{
namespace minimaxM3Sm107
{
namespace mcDetail
{
constexpr int kEmpty = static_cast<int>(0x80000000u);
constexpr int kFinMaxTopk = 8; // MoE finalize: top-k rows loaded together up to this k

// The Eagle3 capture written by a capture layer's boundary (cap_out, optional): rmsnorm(updated residual) * fc_norm[i],
// bit for bit the m3_spec_tail capture_norm kernel (kernels/minimaxM3/specTail.cu at H 6144: 256 threads, thread c
// sums the squares of the 8-column chunks c, c + 256 and c + 512 in that order, the warp butterfly 16..1, the 8 warp
// sums added in warp order, rsqrtf(sum / H + eps), bf16(v * rs * w)). Only its layout is reproduced here; its
// arithmetic is the same source.
constexpr int kCapThreads = 256;
constexpr int kCapIters = 3;
constexpr int kCapWarps = kCapThreads / 32;
constexpr int kCapHidden = kCapThreads * kCapIters * 8;

// capture_norm's per-thread chain over one 8-column chunk (continues `ss`).
__device__ __forceinline__ float cap_chunk_sq(uint4 const raw, float ss)
{
    __nv_bfloat16 const* e = reinterpret_cast<__nv_bfloat16 const*>(&raw);
#pragma unroll
    for (int q = 0; q < 8; ++q)
    {
        float const v = __bfloat162float(e[q]);
        ss += v * v;
    }
    return ss;
}

__device__ __forceinline__ float cap_warp_sum(float v)
{
#pragma unroll
    for (int o = 16; o > 0; o >>= 1)
        v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

// bf16(v * rs * w) for one 8-column chunk of the updated residual (v as floats) into cap.
__device__ __forceinline__ void cap_store(
    float const (&v)[8], float rs, __nv_bfloat16 const* __restrict__ w, __nv_bfloat16* __restrict__ cap)
{
    uint4 const wr = *reinterpret_cast<uint4 const*>(w);
    __nv_bfloat16 const* we = reinterpret_cast<__nv_bfloat16 const*>(&wr);
    uint4 o;
    __nv_bfloat16* oe = reinterpret_cast<__nv_bfloat16*>(&o);
#pragma unroll
    for (int q = 0; q < 8; ++q)
        oe[q] = __float2bfloat16(v[q] * rs * __bfloat162float(we[q]));
    *reinterpret_cast<uint4*>(cap) = o;
}

__device__ __forceinline__ uint32_t canon_bf16x2(uint32_t w)
{
    if ((w & 0xFFFFu) == 0x8000u)
        w &= 0xFFFF0000u;
    if ((w >> 16) == 0x8000u)
        w &= 0x0000FFFFu;
    return w;
}

__device__ __forceinline__ float bf16_lo(uint32_t w)
{
    return __uint_as_float(w << 16);
}

// The MXFP8 publish quantizes exactly as FlashInfer's CuTe DSL mxfp8 quantize (MXFP8QuantizeRunner.CUTE_DSL, MX-TAC's
// pinned decode quantize; quantization_cute_dsl_utils.py), the three helpers below being its PTX verbatim. It equals
// trtllm::mxfp8_quantize bit for bit except on blocks with 0 < amax < 448 * 2^-127 (UE8M0 byte 0), where the
// TRT-LLM kernel's rcp.approx.ftz of the denormal scale gives inf (NaN / garbage E4M3) and this flushes the block to 0.
__device__ __forceinline__ uint32_t mxfp8_ue8m0(float amax)
{
    float value;
    asm("mul.rn.f32 %0, %1, %2;" : "=f"(value) : "f"(amax), "f"(1.0f / 448.0f));
    uint32_t result;
    asm("{\n"
        " .reg .pred p_zero, p_has_mant, p_exp_zero, p_tiny_sub, p_ovf;\n"
        " .reg .u32 bits, exp_biased, mantissa, bump, r;\n"
        " setp.le.f32 p_zero, %1, 0f00000000;\n"
        " mov.b32 bits, %1;\n"
        " shr.b32 exp_biased, bits, 23;\n"
        " and.b32 exp_biased, exp_biased, 255;\n"
        " and.b32 mantissa, bits, 0x7FFFFF;\n"
        " setp.ne.u32 p_has_mant, mantissa, 0;\n"
        " selp.u32 bump, 1, 0, p_has_mant;\n"
        " setp.eq.u32 p_exp_zero, exp_biased, 0;\n"
        " setp.le.u32 p_tiny_sub, mantissa, 0x400000;\n"
        " and.pred p_tiny_sub, p_exp_zero, p_tiny_sub;\n"
        " @p_tiny_sub mov.u32 bump, 0;\n"
        " add.u32 r, exp_biased, bump;\n"
        " setp.gt.u32 p_ovf, r, 254;\n"
        " selp.u32 r, 254, r, p_ovf;\n"
        " selp.u32 %0, 0, r, p_zero;\n"
        "}\n"
        : "=r"(result)
        : "f"(value));
    return result;
}

__device__ __forceinline__ float mxfp8_inv_scale(uint32_t ue8m0)
{
    float inv;
    asm("{\n"
        " .reg .s32 new_exp;\n"
        " .reg .b32 float_bits;\n"
        " .reg .pred p_zero;\n"
        " setp.eq.u32 p_zero, %1, 0;\n"
        " sub.s32 new_exp, 254, %1;\n"
        " max.s32 new_exp, new_exp, 0;\n"
        " shl.b32 float_bits, new_exp, 23;\n"
        " mov.b32 %0, float_bits;\n"
        " @p_zero mov.b32 %0, 0;\n"
        "}\n"
        : "=f"(inv)
        : "r"(ue8m0));
    return inv;
}

// Two BF16-valued floats (lo, hi) times inv_scale -> two E4M3 bytes (lo in the low byte), saturating to +/-448.
__device__ __forceinline__ uint32_t mxfp8_e4m3x2(float lo, float hi, float inv)
{
    uint16_t pair;
    asm("{\n"
        " .reg .f32 f0, f1;\n"
        " mul.f32 f0, %1, %3;\n"
        " mul.f32 f1, %2, %3;\n"
        " min.f32 f0, f0, 0f43E00000;\n"
        " max.f32 f0, f0, 0fC3E00000;\n"
        " min.f32 f1, f1, 0f43E00000;\n"
        " max.f32 f1, f1, 0fC3E00000;\n"
        " cvt.rn.satfinite.e4m3x2.f32 %0, f1, f0;\n"
        "}\n"
        : "=h"(pair)
        : "f"(lo), "f"(hi), "f"(inv));
    return pair;
}

__device__ __forceinline__ float rcp_approx_ftz(float a)
{
    float b;
    asm volatile("rcp.approx.ftz.f32 %0, %1;\n" : "=f"(b) : "f"(a));
    return b;
}

__device__ __forceinline__ float bf16_hi(uint32_t w)
{
    return __uint_as_float(w & 0xFFFF0000u);
}

__device__ __forceinline__ int4 ld_volatile_v4(int const* p)
{
    int4 v;
    asm volatile("ld.volatile.global.v4.s32 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                 : "l"(p)
                 : "memory");
    return v;
}

} // namespace mcDetail
} // namespace minimaxM3Sm107
} // namespace kernels

TRTLLM_MV2_NAMESPACE_END
