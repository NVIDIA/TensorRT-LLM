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

// Exact MoE finalize: the native replacement of kernels/m3_moe_finalize
// (Triton). out[t, h] = bf16(sum_k fma(w[t, k], fc2[idx[t, k], h], acc)) over k in order (idx -1 skipped), float32;
// with shared: bf16(float(shared[t, h]) + float(that)). Bit for bit the Triton kernel and the TRTLLM-Gen runner's
// finalizeKernelVecLoad (+ shared.add_): the same per-element fp32 fma chain and RNE roundings. One CTA per
// (token, THREADS x NV x 8 columns), NV 16-byte vectors per thread (the tile / threads table never changes the order);
// every k gather is issued before the fma chain.
#include "tensorrt_llm/kernels/minimaxM31/moeFinalizeSm107.h"

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{
namespace minimaxM31
{
namespace
{
constexpr int kVec = 8;

// Explicit, exact widening of the weights (no reliance on the implicit bf16 -> float conversion, which builds that
// define __CUDA_NO_BFLOAT16_CONVERSIONS__ disable).
__device__ __forceinline__ float toFloat(float v)
{
    return v;
}

__device__ __forceinline__ float toFloat(__nv_bfloat16 v)
{
    return __bfloat162float(v);
}

template <typename W, int TOPK, bool SHARED, int THREADS, int NV>
__global__ void __launch_bounds__(THREADS) moeFinalizeKernel(__nv_bfloat16 const* __restrict__ fc2,
    int32_t const* __restrict__ idx, W const* __restrict__ weights, __nv_bfloat16 const* __restrict__ shared,
    __nv_bfloat16* __restrict__ out, int64_t ld_fc2, int hidden)
{
    int64_t const t = blockIdx.x;
    int const col0 = blockIdx.y * (THREADS * NV * kVec) + threadIdx.x * kVec;
    int32_t p[TOPK];
    float w[TOPK];
#pragma unroll
    for (int k = 0; k < TOPK; ++k)
    {
        p[k] = idx[t * TOPK + k];
        w[k] = toFloat(weights[t * TOPK + k]);
    }
    uint4 x[NV][TOPK];
    uint4 shv[NV];
#pragma unroll
    for (int v = 0; v < NV; ++v)
    {
        int const col = col0 + v * THREADS * kVec;
        // The shared row is loaded with the gathers, not after the fma chain (load order only).
        if constexpr (SHARED)
            shv[v] = col < hidden ? __ldg(reinterpret_cast<uint4 const*>(shared + t * hidden + col))
                                  : make_uint4(0, 0, 0, 0);
#pragma unroll
        for (int k = 0; k < TOPK; ++k)
            x[v][k] = (p[k] >= 0 && col < hidden)
                ? __ldg(reinterpret_cast<uint4 const*>(fc2 + static_cast<int64_t>(p[k]) * ld_fc2 + col))
                : make_uint4(0, 0, 0, 0);
    }
#pragma unroll
    for (int v = 0; v < NV; ++v)
    {
        int const col = col0 + v * THREADS * kVec;
        if (col >= hidden)
            continue;
        float acc[kVec];
#pragma unroll
        for (int j = 0; j < kVec; ++j)
            acc[j] = 0.f;
#pragma unroll
        for (int k = 0; k < TOPK; ++k)
        {
            if (p[k] < 0)
                continue;
            __nv_bfloat16 const* xv = reinterpret_cast<__nv_bfloat16 const*>(&x[v][k]);
#pragma unroll
            for (int j = 0; j < kVec; ++j)
                acc[j] = __fmaf_rn(w[k], __bfloat162float(xv[j]), acc[j]);
        }
        uint4 r;
        __nv_bfloat16* rv = reinterpret_cast<__nv_bfloat16*>(&r);
#pragma unroll
        for (int j = 0; j < kVec; ++j)
            rv[j] = __float2bfloat16_rn(acc[j]);
        if constexpr (SHARED)
        {
            __nv_bfloat16 const* sv = reinterpret_cast<__nv_bfloat16 const*>(&shv[v]);
#pragma unroll
            for (int j = 0; j < kVec; ++j)
                rv[j] = __float2bfloat16_rn(__bfloat162float(sv[j]) + __bfloat162float(rv[j]));
        }
        *reinterpret_cast<uint4*>(out + t * hidden + col) = r;
    }
}

template <typename W, bool SHARED, int THREADS, int NV>
cudaError_t launchTopk(int topk, int64_t tokens, int hidden, cudaStream_t st, __nv_bfloat16 const* fc2,
    int32_t const* idx, W const* w, __nv_bfloat16 const* sh, __nv_bfloat16* out, int64_t ld)
{
    dim3 grid(tokens, (hidden + THREADS * NV * kVec - 1) / (THREADS * NV * kVec));
    // clang-format off
    switch (topk)
    {
#define M31_FIN_CASE(K)                                                                                                \
    case K:                                                                                                            \
        moeFinalizeKernel<W, K, SHARED, THREADS, NV><<<grid, THREADS, 0, st>>>(fc2, idx, w, sh, out, ld, hidden);      \
        break;
        M31_FIN_CASE(1)
        M31_FIN_CASE(2)
        M31_FIN_CASE(3)
        M31_FIN_CASE(4)
        M31_FIN_CASE(5)
        M31_FIN_CASE(6)
        M31_FIN_CASE(7)
        M31_FIN_CASE(8)
        M31_FIN_CASE(9)
        M31_FIN_CASE(10)
#undef M31_FIN_CASE
    default: return cudaErrorInvalidValue;
    }
    // clang-format on
    return cudaGetLastError();
}

// (THREADS, NV) table; cfg -1 = by rows (the serving default).
template <typename W, bool SHARED>
cudaError_t launchCfg(int cfg, int topk, int64_t tokens, int hidden, cudaStream_t st, __nv_bfloat16 const* f,
    int32_t const* ix, W const* w, __nv_bfloat16 const* sh, __nv_bfloat16* o, int64_t ld)
{
    if (cfg < 0)
        cfg = 0;
    switch (cfg)
    {
    case 0: return launchTopk<W, SHARED, 256, 1>(topk, tokens, hidden, st, f, ix, w, sh, o, ld);
    case 1: return launchTopk<W, SHARED, 128, 2>(topk, tokens, hidden, st, f, ix, w, sh, o, ld);
    case 2: return launchTopk<W, SHARED, 64, 4>(topk, tokens, hidden, st, f, ix, w, sh, o, ld);
    case 3: return launchTopk<W, SHARED, 128, 1>(topk, tokens, hidden, st, f, ix, w, sh, o, ld);
    case 4: return launchTopk<W, SHARED, 64, 2>(topk, tokens, hidden, st, f, ix, w, sh, o, ld);
    case 5: return launchTopk<W, SHARED, 192, 2>(topk, tokens, hidden, st, f, ix, w, sh, o, ld);
    case 6: return launchTopk<W, SHARED, 96, 2>(topk, tokens, hidden, st, f, ix, w, sh, o, ld);
    case 7: return launchTopk<W, SHARED, 32, 4>(topk, tokens, hidden, st, f, ix, w, sh, o, ld);
    default: return cudaErrorInvalidValue;
    }
}
} // namespace

cudaError_t invokeMoeFinalize(__nv_bfloat16 const* fc2, int64_t ldFc2, void const* weights, bool weightsFp32,
    int32_t const* idx, __nv_bfloat16 const* shared, __nv_bfloat16* out, int64_t tokens, int topk, int hidden, int cfg,
    cudaStream_t stream)
{
    if (tokens == 0)
    {
        return cudaSuccess;
    }
    if (weightsFp32)
    {
        auto const* w = static_cast<float const*>(weights);
        return shared != nullptr
            ? launchCfg<float, true>(cfg, topk, tokens, hidden, stream, fc2, idx, w, shared, out, ldFc2)
            : launchCfg<float, false>(cfg, topk, tokens, hidden, stream, fc2, idx, w, shared, out, ldFc2);
    }
    auto const* w = static_cast<__nv_bfloat16 const*>(weights);
    return shared != nullptr
        ? launchCfg<__nv_bfloat16, true>(cfg, topk, tokens, hidden, stream, fc2, idx, w, shared, out, ldFc2)
        : launchCfg<__nv_bfloat16, false>(cfg, topk, tokens, hidden, stream, fc2, idx, w, shared, out, ldFc2);
}

} // namespace minimaxM31
} // namespace kernels

TRTLLM_NAMESPACE_END
