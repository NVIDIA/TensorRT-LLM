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

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/kernels/fusedResidualNormQuantKernel.h"

#include <cuda_bf16.h>
#include <cuda_fp8.h>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

namespace
{

constexpr int kBlock = 128;
// bf16 elements per 16-byte vector.
constexpr int kVec = 8;

// fp32 -> bf16 (round to nearest even) -> fp32.
__device__ __forceinline__ float roundBf16(float v)
{
    __nv_bfloat162 const p = __floats2bfloat162_rn(0.f, v);
    return __uint_as_float(*reinterpret_cast<uint32_t const*>(&p));
}

// One block per row; each thread owns kChunks 16-byte vectors of the row.
template <int HIDDEN, int MODE, bool HAS_Y, bool HAS_GATE, bool HAS_YBIAS>
__global__ void __launch_bounds__(kBlock, 8) residualNormQuantKernel(__nv_bfloat16 const* x, __nv_bfloat16 const* y,
    __nv_bfloat16 const* yBias, float const* gate, float const* weight, float const* bias, float const* invOutScale,
    int rowsPerBatch, float eps, __nv_bfloat16* xOut, __nv_fp8_e4m3* normed)
{
    constexpr int kChunks = HIDDEN / (kBlock * kVec);
    static_assert(HIDDEN % (kBlock * kVec) == 0, "hidden must be a multiple of 1024");

    int const tid = threadIdx.x;
    int const warp = tid >> 5;
    int const lane = tid & 31;
    int const row = blockIdx.x;
    int const batch = row / rowsPerBatch;
    int64_t const base = static_cast<int64_t>(row) * HIDDEN;
    int64_t const batchBase = static_cast<int64_t>(batch) * HIDDEN;

    // Issue the streaming row loads up front.
    uint4 xr[kChunks];
    uint4 yr[kChunks];
#pragma unroll
    for (int c = 0; c < kChunks; c++)
    {
        int const off = (c * kBlock + tid) * kVec;
        xr[c] = *reinterpret_cast<uint4 const*>(x + base + off);
        if constexpr (HAS_Y)
        {
            yr[c] = *reinterpret_cast<uint4 const*>(y + base + off);
        }
    }

    uint4 xb[kChunks];
    float sum = 0.f;
    float sumSq = 0.f;
#pragma unroll
    for (int c = 0; c < kChunks; c++)
    {
        int const off = (c * kBlock + tid) * kVec;
        if constexpr (HAS_Y)
        {
            uint4 br = make_uint4(0, 0, 0, 0);
            float4 gr[2] = {};
            if constexpr (HAS_YBIAS)
            {
                br = *reinterpret_cast<uint4 const*>(yBias + off);
            }
            if constexpr (HAS_GATE)
            {
                gr[0] = *reinterpret_cast<float4 const*>(gate + batchBase + off);
                gr[1] = *reinterpret_cast<float4 const*>(gate + batchBase + off + 4);
            }
            auto const* xh = reinterpret_cast<__nv_bfloat162 const*>(&xr[c]);
            auto const* yh = reinterpret_cast<__nv_bfloat162 const*>(&yr[c]);
            auto const* bh = reinterpret_cast<__nv_bfloat162 const*>(&br);
            auto const* gv = reinterpret_cast<float const*>(gr);
            __nv_bfloat162 o[4];
#pragma unroll
            for (int i = 0; i < 4; i++)
            {
                float2 const xx = __bfloat1622float2(xh[i]);
                float2 yy = __bfloat1622float2(yh[i]);
                if constexpr (HAS_YBIAS)
                {
                    float2 const bb = __bfloat1622float2(bh[i]);
                    yy = make_float2(roundBf16(yy.x + bb.x), roundBf16(yy.y + bb.y));
                }
                float2 r;
                // Round the product before the add; no FMA.
                r.x = HAS_GATE ? __fadd_rn(xx.x, __fmul_rn(yy.x, gv[2 * i])) : __fadd_rn(xx.x, yy.x);
                r.y = HAS_GATE ? __fadd_rn(xx.y, __fmul_rn(yy.y, gv[2 * i + 1])) : __fadd_rn(xx.y, yy.y);
                o[i] = __floats2bfloat162_rn(r.x, r.y);
            }
            xb[c] = *reinterpret_cast<uint4 const*>(o);
            *reinterpret_cast<uint4*>(xOut + base + off) = xb[c];
        }
        else
        {
            xb[c] = xr[c];
        }
        if constexpr (MODE != static_cast<int>(ResidualNormMode::kResidualOnly))
        {
            auto const* h = reinterpret_cast<__nv_bfloat162 const*>(&xb[c]);
#pragma unroll
            for (int i = 0; i < 4; i++)
            {
                float2 const f = __bfloat1622float2(h[i]);
                sum += f.x;
                sumSq += f.x * f.x;
                sum += f.y;
                sumSq += f.y * f.y;
            }
        }
    }

    if constexpr (MODE != static_cast<int>(ResidualNormMode::kResidualOnly))
    {
        __shared__ float warpSum[kBlock / 32];
        __shared__ float warpSumSq[kBlock / 32];
        __shared__ float stats[2];
#pragma unroll
        for (int o = 16; o > 0; o /= 2)
        {
            sum += __shfl_xor_sync(0xffffffffu, sum, o);
            sumSq += __shfl_xor_sync(0xffffffffu, sumSq, o);
        }
        if (lane == 0)
        {
            warpSum[warp] = sum;
            warpSumSq[warp] = sumSq;
        }
        __syncthreads();
        if (warp == 0)
        {
            float a = lane < kBlock / 32 ? warpSum[lane] : 0.f;
            float a2 = lane < kBlock / 32 ? warpSumSq[lane] : 0.f;
#pragma unroll
            for (int o = 16; o > 0; o /= 2)
            {
                a += __shfl_xor_sync(0xffffffffu, a, o);
                a2 += __shfl_xor_sync(0xffffffffu, a2, o);
            }
            if (lane == 0)
            {
                float const invHidden = 1.0f / static_cast<float>(HIDDEN);
                float const mean = a * invHidden;
                // Single-pass variance can cancel slightly below zero.
                float const var = fmaxf(a2 * invHidden - mean * mean, 0.f);
                stats[0] = mean;
                stats[1] = rsqrtf(var + eps);
            }
        }
        __syncthreads();
        float const mean = stats[0];
        float const rstd = stats[1];
        float const qmul = invOutScale[0];
        constexpr bool kAda = MODE == static_cast<int>(ResidualNormMode::kAdaLayerNorm);
        int64_t const modBase = kAda ? batchBase : 0;
#pragma unroll
        for (int c = 0; c < kChunks; c++)
        {
            int const off = (c * kBlock + tid) * kVec;
            float4 wr[2];
            float4 br[2];
            wr[0] = *reinterpret_cast<float4 const*>(weight + modBase + off);
            wr[1] = *reinterpret_cast<float4 const*>(weight + modBase + off + 4);
            br[0] = *reinterpret_cast<float4 const*>(bias + modBase + off);
            br[1] = *reinterpret_cast<float4 const*>(bias + modBase + off + 4);
            auto const* wv = reinterpret_cast<float const*>(wr);
            auto const* bv = reinterpret_cast<float const*>(br);
            auto const* h = reinterpret_cast<__nv_bfloat162 const*>(&xb[c]);
            __nv_fp8x2_storage_t packed[4];
#pragma unroll
            for (int i = 0; i < kVec; i += 2)
            {
                float2 const xf = __bfloat1622float2(h[i / 2]);
                float const xv[2] = {xf.x, xf.y};
                float yv[2];
#pragma unroll
                for (int j = 0; j < 2; j++)
                {
                    float const xn = (xv[j] - mean) * rstd;
                    float const ww = kAda ? 1.0f + wv[i + j] : wv[i + j];
                    // Round to bf16, then the saturating static FP8 quantization.
                    yv[j] = roundBf16(xn * ww + bv[i + j]) * qmul;
                }
                packed[i / 2] = __nv_cvt_float2_to_fp8x2(make_float2(yv[0], yv[1]), __NV_SATFINITE, __NV_E4M3);
            }
            *reinterpret_cast<uint2*>(normed + base + off) = *reinterpret_cast<uint2 const*>(packed);
        }
    }
}

template <int HIDDEN, int MODE, bool HAS_Y, bool HAS_GATE, bool HAS_YBIAS>
void launchKernel(ResidualNormQuantParams const& p, cudaStream_t stream)
{
    residualNormQuantKernel<HIDDEN, MODE, HAS_Y, HAS_GATE, HAS_YBIAS>
        <<<p.rows, kBlock, 0, stream>>>(static_cast<__nv_bfloat16 const*>(p.x), static_cast<__nv_bfloat16 const*>(p.y),
            static_cast<__nv_bfloat16 const*>(p.y_bias), p.gate, p.weight, p.bias, p.inv_out_scale, p.rows_per_batch,
            p.eps, static_cast<__nv_bfloat16*>(p.x_out), static_cast<__nv_fp8_e4m3*>(p.normed));
}

template <int HIDDEN, int MODE>
void dispatchResidual(ResidualNormQuantParams const& p, cudaStream_t stream)
{
    bool const hasGate = p.gate != nullptr;
    bool const hasYBias = p.y_bias != nullptr;
    if (p.y == nullptr)
    {
        if constexpr (MODE != static_cast<int>(ResidualNormMode::kResidualOnly))
        {
            launchKernel<HIDDEN, MODE, false, false, false>(p, stream);
        }
    }
    else if (hasGate && hasYBias)
    {
        launchKernel<HIDDEN, MODE, true, true, true>(p, stream);
    }
    else if (hasGate)
    {
        launchKernel<HIDDEN, MODE, true, true, false>(p, stream);
    }
    else if (hasYBias)
    {
        launchKernel<HIDDEN, MODE, true, false, true>(p, stream);
    }
    else
    {
        launchKernel<HIDDEN, MODE, true, false, false>(p, stream);
    }
}

template <int HIDDEN>
void dispatchMode(ResidualNormQuantParams const& p, cudaStream_t stream)
{
    switch (p.mode)
    {
    case ResidualNormMode::kResidualOnly:
        dispatchResidual<HIDDEN, static_cast<int>(ResidualNormMode::kResidualOnly)>(p, stream);
        break;
    case ResidualNormMode::kLayerNorm:
        dispatchResidual<HIDDEN, static_cast<int>(ResidualNormMode::kLayerNorm)>(p, stream);
        break;
    case ResidualNormMode::kAdaLayerNorm:
        dispatchResidual<HIDDEN, static_cast<int>(ResidualNormMode::kAdaLayerNorm)>(p, stream);
        break;
    }
}

} // namespace

bool residualNormQuantSupportsHidden(int hidden)
{
    return hidden == 3072 || hidden == 5120;
}

void launchResidualNormQuant(ResidualNormQuantParams const& p, cudaStream_t stream)
{
    TLLM_CHECK_WITH_INFO(
        residualNormQuantSupportsHidden(p.hidden), "residual norm quant: unsupported hidden %d", p.hidden);
    TLLM_CHECK_WITH_INFO(p.rows_per_batch > 0, "residual norm quant: rows_per_batch must be positive");
    bool const normalizes = p.mode != ResidualNormMode::kResidualOnly;
    TLLM_CHECK_WITH_INFO(p.y != nullptr || normalizes, "residual norm quant: residual-only mode needs y");
    TLLM_CHECK_WITH_INFO(p.y == nullptr || p.x_out != nullptr, "residual norm quant: y needs x_out");
    TLLM_CHECK_WITH_INFO(
        !normalizes || (p.weight != nullptr && p.bias != nullptr && p.inv_out_scale != nullptr && p.normed != nullptr),
        "residual norm quant: normalization needs weight, bias, inv_out_scale and normed");
    if (p.rows == 0)
    {
        return;
    }
    if (p.hidden == 3072)
    {
        dispatchMode<3072>(p, stream);
    }
    else
    {
        dispatchMode<5120>(p, stream);
    }
    sync_check_cuda_error(stream);
}

} // namespace kernels

TRTLLM_NAMESPACE_END
