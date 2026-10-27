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

// The split-KV merge for the MiniMax-M3 NVFP4 TP2 (Rubin) full-context decode. The producer is the PrimTS split-KV
// decode with its separate-reduction scratch: per (request, KV head, split, row) one normalized 16-bit partial O and
// one fp32 log2-LSE. Active splits per row follow the producer's split ranges (the causal union of the row's grouped
// Q CTA, instruction-group aligned): splits past them hold nothing and are not read.
//
// One CTA per (request, KV head, kRowsPerCta rows); per row kGroups split groups x kThreadsPerRow threads x 8 dims.
// A thread issues the loads of every split it owns (strided over the groups) before using any, folds them under its
// group's max, and the groups merge through SMEM with one max rescale; the row is normalized once and rounded to
// BF16 once.

#include "tensorrt_llm/kernels/minimaxM3Sm107Tp8/fullCtxReduce.h"

#include <cstdint>
#include <cuda_fp16.h>
#include <math_constants.h>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{
namespace minimaxM3Sm107Tp8
{
namespace fullCtxReduce
{
namespace
{

constexpr int kGroups = 8;
constexpr int kThreadsPerRow = kHeadDim / 8;
constexpr int kRowsPerCta = 1;
constexpr int kSplitsPerGroup = kMaxSplits / kGroups;
constexpr int kThreads = kGroups * kThreadsPerRow * kRowsPerCta;

__device__ int activeSplits(Params const& p, int b, int row)
{
    int const seqKv = p.seqLensKv[b];
    int visible = seqKv;
    if (p.causal)
    {
        int const token = row / p.headsPerKv;
        int const base = (token / p.qTokensPerCta) * p.qTokensPerCta;
        int const end = min(base + p.qTokensPerCta, p.seqLenQ);
        visible = min(max(seqKv - p.seqLenQ + end, 0), seqKv);
    }
    int const tiles = (visible + p.tileKv - 1) / p.tileKv;
    int const tilesPerGroup = p.splits * p.instsKv;
    int const groups = (tiles + tilesPerGroup - 1) / tilesPerGroup;
    int const local = max(groups * p.instsKv, p.instsKv);
    return (tiles + local - 1) / local;
}

template <bool kBf16Partial>
__device__ float2 unpack(uint32_t word)
{
    if constexpr (kBf16Partial)
    {
        return __bfloat1622float2(*reinterpret_cast<__nv_bfloat162 const*>(&word));
    }
    else
    {
        return __half22float2(*reinterpret_cast<__half2 const*>(&word));
    }
}

template <bool kBf16Partial>
__global__ void __launch_bounds__(kThreads) fullCtxReduceKernel(Params const p)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("griddepcontrol.wait;" ::: "memory");
    // The dependent (the o_proj activation quantization / GEMV) reads `out` only after its own griddepcontrol.wait,
    // which waits for this whole grid, so it is released here and runs its weight-only prologue under the merge.
    asm volatile("griddepcontrol.launch_dependents;");
#endif
    int const b = blockIdx.z;
    int const h = blockIdx.y;
    int const localRow = threadIdx.x / (kGroups * kThreadsPerRow);
    int const row = blockIdx.x * kRowsPerCta + localRow;
    int const g = (threadIdx.x / kThreadsPerRow) % kGroups;
    int const t = threadIdx.x % kThreadsPerRow;
    bool const validRow = row < p.rows;
    int const active = validRow ? activeSplits(p, b, row) : 0;

    __shared__ float sMax[kRowsPerCta][kGroups];
    __shared__ float sSum[kRowsPerCta][kGroups];
    __shared__ float sAcc[kRowsPerCta][kGroups][kHeadDim];

    uint4 const* partialO = static_cast<uint4 const*>(p.partialO);
    size_t const kvIdx = static_cast<size_t>(b) * p.numKvHeads + h;
    float lse[kSplitsPerGroup];
    uint4 vec[kSplitsPerGroup];
#pragma unroll
    for (int i = 0; i < kSplitsPerGroup; ++i)
    {
        int const s = g + i * kGroups;
        if (s < active)
        {
            size_t const r = (kvIdx * p.splits + s) * p.rows + row;
            lse[i] = __ldcg(p.partialStats + r);
            vec[i] = __ldcg(partialO + r * kThreadsPerRow + t);
        }
    }
    float groupMax = -CUDART_INF_F;
#pragma unroll
    for (int i = 0; i < kSplitsPerGroup; ++i)
    {
        if (g + i * kGroups < active)
        {
            groupMax = fmaxf(groupMax, lse[i]);
        }
    }
    // A split of the CTA's causal union can hold no key of this row (log2-LSE -inf). Such a split, and a group of
    // only such splits, is skipped rather than weighted 0: its partial O carries no defined value.
    float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    float sum = 0.f;
#pragma unroll
    for (int i = 0; i < kSplitsPerGroup; ++i)
    {
        if (g + i * kGroups < active && lse[i] != -CUDART_INF_F)
        {
            float const w = exp2f(lse[i] - groupMax);
            sum += w;
            uint32_t const* words = reinterpret_cast<uint32_t const*>(&vec[i]);
#pragma unroll
            for (int j = 0; j < 4; ++j)
            {
                float2 const f = unpack<kBf16Partial>(words[j]);
                acc[2 * j] += w * f.x;
                acc[2 * j + 1] += w * f.y;
            }
        }
    }
    if (t == 0)
    {
        sMax[localRow][g] = groupMax;
        sSum[localRow][g] = sum;
    }
#pragma unroll
    for (int j = 0; j < 8; ++j)
    {
        sAcc[localRow][g][t * 8 + j] = acc[j];
    }
    __syncthreads();
    if (g != 0 || !validRow)
    {
        return;
    }
    float rowMax = -CUDART_INF_F;
#pragma unroll
    for (int k = 0; k < kGroups; ++k)
    {
        rowMax = fmaxf(rowMax, sMax[localRow][k]);
    }
    float total = 0.f;
    float o[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int k = 0; k < kGroups; ++k)
    {
        if (sMax[localRow][k] == -CUDART_INF_F)
        {
            continue;
        }
        float const scale = exp2f(sMax[localRow][k] - rowMax);
        total += scale * sSum[localRow][k];
#pragma unroll
        for (int j = 0; j < 8; ++j)
        {
            o[j] += scale * sAcc[localRow][k][t * 8 + j];
        }
    }
    float const inv = total > 0.f ? 1.f / total : 0.f;
    int const token = row / p.headsPerKv;
    int const head = row % p.headsPerKv;
    __nv_bfloat162 packed[4];
#pragma unroll
    for (int j = 0; j < 4; ++j)
    {
        packed[j] = __floats2bfloat162_rn(o[2 * j] * inv, o[2 * j + 1] * inv);
    }
    size_t const outRow
        = (static_cast<size_t>(b) * p.seqLenQ + token) * (p.numKvHeads * p.headsPerKv) + h * p.headsPerKv + head;
    reinterpret_cast<uint4*>(p.out + outRow * kHeadDim)[t] = *reinterpret_cast<uint4 const*>(packed);
    if (p.q8 == nullptr)
    {
        return;
    }
    // MXFP8 copy of the BF16 row: a 32-column block is 4 neighbouring threads x 8 columns (lanes 0..15 of warp 0
    // hold the row). Block scale = the UE8M0 exponent of ceil(amax / 448), values = satfinite E4M3 of x * 2^-scale.
    float x[8];
    float amax = 0.f;
#pragma unroll
    for (int j = 0; j < 4; ++j)
    {
        float2 const f = __bfloat1622float2(packed[j]);
        x[2 * j] = f.x;
        x[2 * j + 1] = f.y;
        amax = fmaxf(amax, fmaxf(fabsf(f.x), fabsf(f.y)));
    }
    amax = fmaxf(amax, __shfl_xor_sync(0x0000ffffu, amax, 1));
    amax = fmaxf(amax, __shfl_xor_sync(0x0000ffffu, amax, 2));
    float rcp;
    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(rcp) : "f"(448.0F));
    float const sfv = amax * rcp;
    int const bits = __float_as_int(sfv);
    int e = ((bits >> 23) & 0xFF) + ((bits & 0x7FFFFF) != 0);
    e = min(e, 254);
    float const qs = amax != 0.F ? __int_as_float((254 - e) << 23) : 0.F;
    uint16_t q2[4];
#pragma unroll
    for (int j = 0; j < 4; ++j)
    {
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(q2[j]) : "f"(x[2 * j + 1] * qs), "f"(x[2 * j] * qs));
    }
    int const tok = b * p.seqLenQ + token;
    int const col = (h * p.headsPerKv + head) * kHeadDim + t * 8;
    *reinterpret_cast<uint2*>(p.q8 + static_cast<int64_t>(tok) * (p.numSfKTiles * 128) + col)
        = *reinterpret_cast<uint2 const*>(q2);
    if (t % 4 == 0)
    {
        int const kSf = col / 32;
        int64_t const off = static_cast<int64_t>(tok / 128) * p.numSfKTiles * 512 + (kSf / 4) * 512 + (tok % 32) * 16
            + ((tok % 128) / 32) * 4 + kSf % 4;
        p.sf[off] = static_cast<uint8_t>(e);
    }
}

} // namespace

cudaError_t invokeFullCtxReduce(Params const& params, bool pdl, cudaStream_t stream)
{
    if (params.splits < 1 || params.splits > kMaxSplits || params.rows < 1)
    {
        return cudaErrorInvalidValue;
    }
    cudaLaunchConfig_t config = {};
    config.gridDim = dim3((params.rows + kRowsPerCta - 1) / kRowsPerCta, params.numKvHeads, params.batch);
    config.blockDim = dim3(kThreads);
    config.stream = stream;
    cudaLaunchAttribute attribute[1];
    attribute[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attribute[0].val.programmaticStreamSerializationAllowed = pdl ? 1 : 0;
    config.attrs = attribute;
    config.numAttrs = 1;
    return params.bf16Partial ? cudaLaunchKernelEx(&config, fullCtxReduceKernel<true>, params)
                              : cudaLaunchKernelEx(&config, fullCtxReduceKernel<false>, params);
}

} // namespace fullCtxReduce
} // namespace minimaxM3Sm107Tp8
} // namespace kernels

TRTLLM_NAMESPACE_END
