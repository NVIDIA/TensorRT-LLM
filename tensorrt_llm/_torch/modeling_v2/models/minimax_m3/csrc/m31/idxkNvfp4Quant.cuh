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

// MiniMax-M3.1 NVFP4 index-K quantize-and-store of one token, shared by the index-K writer
// (csrc/m31/idxkNvfp4.cu, writeKernel) and the horizontal producers' NVFP4 index-K mode
// (kernels/fusedQKNormRopeKernel.cu), so both write the same bytes from the same values by construction.
//
// Page layout (the vLLM / MSA Q8KV4 contract, one 128-token page, one index head, head_dim 128):
//   bytes [0, 8192): packed E2M1 codes, token t at t * 64, low nibble = even element;
//   bytes [8192, 9216): E4M3 scales of 16-element groups, token t group g at 8192 + t * 8 + g (linear).
// Recipe (unit scale, for inputs exactly representable in bf16):
//   sf = E4M3_RN_satfinite(max(RN(amax * RN(1/6)), 2^-9)); q = RZ(x * rcp.approx.ftz(sf));
//   code = cvt.rn.satfinite.e2m1x2.f32(q); a zero magnitude is stored as +0.

#pragma once

#include "tensorrt_llm/modeling_v2/common/namespace.h"

#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

TRTLLM_MV2_NAMESPACE_BEGIN

namespace kernels
{
namespace minimaxM31
{

constexpr int kIdxkNvfp4PageTokens = 128;
constexpr int kIdxkNvfp4Dim = 128;
constexpr int kIdxkNvfp4DataBytes = kIdxkNvfp4PageTokens * kIdxkNvfp4Dim / 2;                                  // 8192
constexpr int kIdxkNvfp4ScaleBytesPerToken = kIdxkNvfp4Dim / 16;                                               // 8
constexpr int kIdxkNvfp4PageBytes = kIdxkNvfp4DataBytes + kIdxkNvfp4PageTokens * kIdxkNvfp4ScaleBytesPerToken; // 9216

__device__ __forceinline__ float idxkNvfp4RcpApproxFtz(float a)
{
    float b;
    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(b) : "f"(a));
    return b;
}

// Zero is stored as +0: clear the sign of every nibble whose magnitude is 0.
__device__ __forceinline__ uint16_t idxkNvfp4ClearNegZeroNibbles(uint16_t packed)
{
    return packed & (((packed & 0x7777u) + 0x7777u) | 0x7777u);
}

// Each value rounded to bf16 (RN) and back: the producer's NVFP4 mode quantizes the index-K it would store as bf16,
// so its bytes equal the writer's on those bf16 rows. The identity on bf16 and E4M3 inputs.
__device__ __forceinline__ void idxkNvfp4RoundToBf16(float const (&in)[4], float (&out)[4])
{
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        out[i] = __bfloat162float(__float2bfloat16_rn(in[i]));
    }
}

// Called by a whole warp for one token: lane l holds elements [4l, 4l + 4) of group l / 4 in x (values exactly
// representable in bf16). Writes the token's 32 E2M1 bytes (one u16 per lane, the warp's 64 contiguous bytes) and its
// 8 scale bytes (lanes l % 4 == 0) into the page at pageBase (16-byte aligned), token t of the page.
__device__ __forceinline__ void idxkNvfp4QuantStore(float const (&x)[4], int lane, uint8_t* pageBase, int t)
{
    float amax = 0.f;
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        amax = fmaxf(amax, fabsf(x[i]));
    }
    // Group max over the 4 lanes of the 16-element group.
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 2));
    constexpr float kRcp6 = 0x1.555556p-3f; // RN(1 / 6)
    __nv_fp8_e4m3 const sf(fmaxf(__fmul_rn(amax, kRcp6), 0x1p-9f));
    float const sfRcp = idxkNvfp4RcpApproxFtz(float(sf));
    float q[4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        q[i] = __fmul_rz(x[i], sfRcp);
    }
    uint16_t packed;
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    // Low nibble = even element; cvt packs its first source in the high nibble.
    asm("{\n"
        "  .reg .b8 lo, hi;\n"
        "  cvt.rn.satfinite.e2m1x2.f32 lo, %2, %1;\n"
        "  cvt.rn.satfinite.e2m1x2.f32 hi, %4, %3;\n"
        "  mov.b16 %0, {lo, hi};\n"
        "}"
        : "=h"(packed)
        : "f"(q[0]), "f"(q[1]), "f"(q[2]), "f"(q[3]));
#else
    // E2M1 conversion needs SM100+: an NVFP4 index-K store compiled for an older target traps rather than write
    // wrong bytes.
    packed = 0;
    __trap();
#endif
    packed = idxkNvfp4ClearNegZeroNibbles(packed);
    reinterpret_cast<uint16_t*>(pageBase + t * (kIdxkNvfp4Dim / 2))[lane] = packed;
    if ((lane & 3) == 0)
    {
        pageBase[kIdxkNvfp4DataBytes + t * kIdxkNvfp4ScaleBytesPerToken + (lane >> 2)] = sf.__x;
    }
}

} // namespace minimaxM31
} // namespace kernels

TRTLLM_MV2_NAMESPACE_END
