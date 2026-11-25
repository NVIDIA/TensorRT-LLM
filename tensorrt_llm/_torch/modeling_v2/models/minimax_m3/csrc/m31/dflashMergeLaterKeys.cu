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

#include "minimax_m3/csrc/m31/dflashMergeLaterKeys.h"
#include "tensorrt_llm/modeling_v2/common/namespace.h"

#include <cmath>

TRTLLM_MV2_NAMESPACE_BEGIN

namespace kernels
{
namespace minimaxM31
{
namespace
{

constexpr int kWarpsPerCta = 4;

__device__ __forceinline__ float warpSum(float v)
{
#pragma unroll
    for (int o = 16; o > 0; o >>= 1)
    {
        v += __shfl_xor_sync(0xffffffffu, v, o);
    }
    return v;
}

__device__ __forceinline__ void load4(__nv_bfloat16 const* p, float f[4])
{
    uint2 const raw = *reinterpret_cast<uint2 const*>(p);
    __nv_bfloat162 const a = *reinterpret_cast<__nv_bfloat162 const*>(&raw.x);
    __nv_bfloat162 const b = *reinterpret_cast<__nv_bfloat162 const*>(&raw.y);
    float2 const fa = __bfloat1622float2(a), fb = __bfloat1622float2(b);
    f[0] = fa.x;
    f[1] = fa.y;
    f[2] = fb.x;
    f[3] = fb.y;
}

__global__ void __launch_bounds__(kWarpsPerCta * 32) mergeLaterKeysKernel(DflashMergeParams p, float ln2, bool pdl)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    if (pdl)
    {
        asm volatile("griddepcontrol.wait;" ::: "memory");
    }
#endif
    int const warp = blockIdx.x * kWarpsPerCta + (threadIdx.x >> 5);
    int const lane = threadIdx.x & 31;
    int const rows = p.B * p.blk;
    if (warp < rows * p.numHeads)
    {
        int const r = warp / p.numHeads;
        int const h = warp - r * p.numHeads;
        int const b = r / p.blk;
        int const j = r - b * p.blk;
        int const hk = h / p.group;
        int const kEnd = p.windowRight >= 0 ? min(p.blk - 1, j + p.windowRight) : p.blk - 1;
        if (j + 1 <= kEnd)
        {
            int const d0 = lane * 4;
            float q[4];
            load4(p.query + b * p.qB + j * p.qT + h * p.qH + d0, q);
            float logit[kDflashMergeMaxBlock];
            float m = -INFINITY;
            for (int k = j + 1; k <= kEnd; ++k)
            {
                float kk[4];
                load4(p.key + b * p.kB + k * p.kT + hk * p.kH + d0, kk);
                float part = q[0] * kk[0];
                part += q[1] * kk[1];
                part += q[2] * kk[2];
                part += q[3] * kk[3];
                logit[k] = warpSum(part) * p.scale;
                m = fmaxf(m, logit[k]);
            }
            float s = 0.f;
            float e[kDflashMergeMaxBlock];
            for (int k = j + 1; k <= kEnd; ++k)
            {
                e[k] = expf(logit[k] - m);
                s += e[k];
            }
            float const lseCausal = p.lseLog2[r * p.lseRow + h * p.lseHead] * ln2;
            float const w = 1.f / (1.f + expf(-(m + logf(s) - lseCausal)));
            float const probScale = w / s;
            __nv_bfloat16* o = p.out + r * p.outRow + h * p.outHead + d0;
            float acc[4];
            load4(o, acc);
#pragma unroll
            for (int i = 0; i < 4; ++i)
            {
                acc[i] = acc[i] * (1.f - w);
            }
            for (int k = j + 1; k <= kEnd; ++k)
            {
                float vv[4];
                load4(p.value + b * p.vB + k * p.vT + hk * p.vH + d0, vv);
                float const a = probScale * e[k];
#pragma unroll
                for (int i = 0; i < 4; ++i)
                {
                    acc[i] = acc[i] + a * vv[i];
                }
            }
            __nv_bfloat162 const lo = __floats2bfloat162_rn(acc[0], acc[1]);
            __nv_bfloat162 const hi = __floats2bfloat162_rn(acc[2], acc[3]);
            uint2 raw;
            raw.x = *reinterpret_cast<uint32_t const*>(&lo);
            raw.y = *reinterpret_cast<uint32_t const*>(&hi);
            *reinterpret_cast<uint2*>(o) = raw;
        }
    }
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    if (pdl)
    {
        asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    }
#endif
}

} // namespace

cudaError_t invokeDflashMergeLaterKeys(DflashMergeParams const& params, bool pdl, cudaStream_t stream)
{
    if (params.blk > kDflashMergeMaxBlock || params.group <= 0)
    {
        return cudaErrorInvalidValue;
    }
    int64_t const warps = static_cast<int64_t>(params.B) * params.blk * params.numHeads;
    if (warps <= 0)
    {
        return cudaSuccess;
    }
    cudaLaunchConfig_t cfg{};
    cfg.gridDim = dim3(static_cast<unsigned>((warps + kWarpsPerCta - 1) / kWarpsPerCta));
    cfg.blockDim = dim3(kWarpsPerCta * 32);
    cfg.stream = stream;
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = pdl ? 1 : 0;
    cfg.attrs = attr;
    cfg.numAttrs = 1;
    float const ln2 = static_cast<float>(std::log(2.0));
    return cudaLaunchKernelEx(&cfg, mergeLaterKeysKernel, params, ln2, pdl);
}

} // namespace minimaxM31
} // namespace kernels

TRTLLM_MV2_NAMESPACE_END
