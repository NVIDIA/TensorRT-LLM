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

#include "minimax_m3/csrc/m31/swigluOaiBf16x2.cuh"
#include "minimax_m3/csrc/m31/swigluOaiMxfp8DecodeSm107.h"
#include "tensorrt_llm/modeling_v2/common/namespace.h"

#include <cuda_runtime.h>

TRTLLM_MV2_NAMESPACE_BEGIN

namespace kernels
{
namespace minimaxM31
{

namespace
{

__device__ __forceinline__ float rcpApproxFtz(float v)
{
    float r;
    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(v));
    return r;
}

// Two fp32 -> two E4M3 (RN, satfinite); `lo` lands in the lower byte (fp32_vec_to_e4m3's conversion).
__device__ __forceinline__ uint32_t e4m3x2(float lo, float hi)
{
    uint16_t r;
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(r) : "f"(hi), "f"(lo));
    return r;
}

// mxfp8_quantize(x, True)'s 128x4-swizzled offset of (row, K32 block) at nbPad (a multiple of 4) blocks per row.
__device__ __forceinline__ int64_t swizzledSfOffset(int row, int blk, int nbPad)
{
    return (blk % 4) + (blk / 4) * 512 + (row % 32) * 16 + ((row % 128) / 32) * 4
        + static_cast<int64_t>(row / 128) * (128 * nbPad);
}

//! CTA b serves row b (b < rows; rows strided over the grid) and zeroes the scale row b of the 128-row padding.
//! Thread t < d / 8 owns columns [8t, 8t + 8); the 4 lanes of one K32 block share its amax by two xor shuffles. The
//! block is a whole number of warps, so every shuffle runs on 32 converged lanes: a lane past d / 8 holds amax 0 and
//! its 4-lane group lies entirely past d / 8 (d / 8 is a multiple of 4), so it never mixes into a live block.
template <bool kFiniteCode0>
__global__ void swigluOaiMxfp8DecodeKernel(uint4 const* __restrict__ x, int64_t xRowStride16, uint2* __restrict__ qOut,
    uint8_t* __restrict__ sfOut, int rows, int d)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    cudaGridDependencySynchronize();
    int const t = threadIdx.x;
    int const ct = d / 8;
    int const nb = d / 32;
    int const nbPad = (nb + 3) / 4 * 4;
    int const padded = (rows + 127) / 128 * 128;
    bool const live = t < ct;
    for (int row = blockIdx.x; row < rows; row += gridDim.x)
    {
        uint4 g = make_uint4(0u, 0u, 0u, 0u);
        uint4 u = g;
        if (live)
        {
            int64_t const xr = static_cast<int64_t>(row) * xRowStride16;
            g = x[xr + t];
            u = x[xr + ct + t];
        }
        uint32_t const h[4] = {
            swigluOaiBf16x2(g.x, u.x), swigluOaiBf16x2(g.y, u.y), swigluOaiBf16x2(g.z, u.z), swigluOaiBf16x2(g.w, u.w)};
        float v[8];
#pragma unroll
        for (int i = 0; i < 4; ++i)
        {
            v[2 * i] = __uint_as_float(h[i] << 16);
            v[2 * i + 1] = __uint_as_float(h[i] & 0xFFFF0000u);
        }
        float m = 0.f;
#pragma unroll
        for (int i = 0; i < 8; ++i)
        {
            m = fmaxf(m, fabsf(v[i]));
        }
        m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 1));
        m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 2));
        if (live)
        {
            // __nv_cvt_float_to_e8m0(m * rcp.approx(448), satfinite, toward +inf), as cvt_warp_fp16_to_mxfp8.
            uint32_t const bits = __float_as_uint(m * rcpApproxFtz(448.f));
            int const ex = (bits >> 23) & 0xFF;
            int const e = min(ex > 0 ? ex + ((bits & 0x7FFFFFu) != 0) : (bits > 0x400000u ? 1 : 0), 254);
            float inv;
            if (e == 0)
            {
                // 2^-127 is an fp32 subnormal (rcp.approx.ftz of it is inf): m3_swq scales by the exact 2^127, the
                // in-tree quantize publishes zero data.
                inv = (kFiniteCode0 && m != 0.f) ? 1.7014118346046923e38f : 0.f;
            }
            else
            {
                inv = m != 0.f ? rcpApproxFtz(__uint_as_float(static_cast<uint32_t>(e) << 23)) : 0.f;
            }
            uint2 o;
            o.x = e4m3x2(v[0] * inv, v[1] * inv) | (e4m3x2(v[2] * inv, v[3] * inv) << 16);
            o.y = e4m3x2(v[4] * inv, v[5] * inv) | (e4m3x2(v[6] * inv, v[7] * inv) << 16);
            qOut[static_cast<int64_t>(row) * ct + t] = o;
            if ((t & 3) == 0)
            {
                sfOut[swizzledSfOffset(row, t >> 2, nbPad)] = static_cast<uint8_t>(e);
            }
        }
        else if ((t & 3) == 0 && (t >> 2) < nbPad)
        {
            // K32-block padding of a live row (d / 32 not a multiple of 4): zero, as mxfp8_quantize.
            sfOut[swizzledSfOffset(row, t >> 2, nbPad)] = 0;
        }
    }
    for (int pr = rows + blockIdx.x; pr < padded; pr += gridDim.x)
    {
        for (int b = t; b < nbPad; b += blockDim.x)
        {
            sfOut[swizzledSfOffset(pr, b, nbPad)] = 0;
        }
    }
#endif
}

} // namespace

cudaError_t invokeSwigluOaiMxfp8Decode(void const* x, int64_t xRowStride, void* qOut, uint8_t* sfOut, int64_t rows,
    int64_t d, bool finiteCode0, bool pdl, cudaStream_t stream)
{
    if (d <= 0 || d % 32 != 0 || d > 8192 || rows < 0 || rows > (int64_t{1} << 24) || xRowStride < 2 * d
        || xRowStride % 8 != 0 || reinterpret_cast<uintptr_t>(x) % 16 != 0
        || reinterpret_cast<uintptr_t>(qOut) % 8 != 0)
    {
        return cudaErrorInvalidValue;
    }
    if (rows == 0)
    {
        return cudaSuccess;
    }
    cudaLaunchConfig_t cfg{};
    // CTAs past `rows` zero the scale padding in parallel with the live rows.
    cfg.gridDim = dim3(static_cast<unsigned>((rows + 127) / 128 * 128));
    cfg.blockDim = dim3(static_cast<unsigned>((d / 8 + 31) / 32 * 32));
    cfg.stream = stream;
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = pdl ? 1 : 0;
    cfg.attrs = attr;
    cfg.numAttrs = 1;
    auto const* xv = static_cast<uint4 const*>(x);
    auto* qv = static_cast<uint2*>(qOut);
    int const r = static_cast<int>(rows);
    int const dd = static_cast<int>(d);
    int64_t const stride16 = xRowStride / 8;
    return finiteCode0 ? cudaLaunchKernelEx(&cfg, swigluOaiMxfp8DecodeKernel<true>, xv, stride16, qv, sfOut, r, dd)
                       : cudaLaunchKernelEx(&cfg, swigluOaiMxfp8DecodeKernel<false>, xv, stride16, qv, sfOut, r, dd);
}

} // namespace minimaxM31
} // namespace kernels

TRTLLM_MV2_NAMESPACE_END
