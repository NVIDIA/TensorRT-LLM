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

// The TP4 drafter layer's generation preprocess in one launch. The rotation is applyBiasRopeUpdateKVCacheV2's GPT-NeoX
// path at BF16 (apply_rotary_embedding_gptneox: cos * x + sin * pair in fp32, the first rotary half's sin negated,
// dims past the rotary width with (cos, sin) = (1, 0), rounded to BF16); the K / V store is its unit-scale E4M3
// store (__nv_fp8x2_e4m3 from BF16 pairs), which is also the query's E4M3: torch's CUDA cast of the BF16 query
// saturates the same way (448, not NaN).

#include "tensorrt_llm/kernels/minimaxM3Sm107Tp4/drafterRopeFp8Kv.h"

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{
namespace minimaxM3Sm107Tp4
{
namespace drafterRopeFp8Kv
{
namespace
{

constexpr int kElemsPerThread = kHeadDim / 32;
constexpr int kBlockSize = 256;

template <int kRopeMode>
__device__ __forceinline__ float rotate(float c, float x, float s, float pair)
{
    if constexpr (kRopeMode == 0)
    {
        return __fadd_rn(__fmul_rn(c, x), __fmul_rn(s, pair));
    }
    else if constexpr (kRopeMode == 1)
    {
        return __fmaf_rn(c, x, __fmul_rn(s, pair));
    }
    else
    {
        return __fmaf_rn(s, pair, __fmul_rn(c, x));
    }
}

__device__ __forceinline__ void storeE4m3Pairs(__nv_fp8_e4m3* dst, __nv_bfloat16 const (&v)[kElemsPerThread])
{
#pragma unroll
    for (int i = 0; i < kElemsPerThread; i += 2)
    {
        __nv_bfloat162 pair;
        pair.x = v[i];
        pair.y = v[i + 1];
        __nv_fp8x2_e4m3 const packed(pair);
        reinterpret_cast<__nv_fp8x2_storage_t*>(dst)[i / 2] = packed.__x;
    }
}

template <int kRopeMode>
__global__ void __launch_bounds__(kBlockSize) drafterRopeFp8KvKernel(Params const p)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("griddepcontrol.wait;" ::: "memory");
    asm volatile("griddepcontrol.launch_dependents;");
#endif
    int const lane = threadIdx.x % 32;
    int const warp = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int const slots = p.numHeadsQ + p.numHeadsKv;
    int const token = warp / slots;
    if (token >= p.numTokens)
    {
        return;
    }
    int const slot = warp % slots;
    bool const isQ = slot < p.numHeadsQ;
    int const head = isQ ? slot : slot - p.numHeadsQ;
    int const request = token / p.qLen;
    int const row = token % p.qLen;
    int const cacheLen = p.seqLens[request];
    int const kvPos = cacheLen - p.qLen + row;
    int const ropePos
        = p.specPositionOffsets != nullptr ? p.specPositionOffsets[request * p.qLen + row] + cacheLen - p.qLen : kvPos;
    int const rowWidth = (p.numHeadsQ + 2 * p.numHeadsKv) * kHeadDim;
    __nv_bfloat16 const* src
        = p.qkv + static_cast<long long>(token) * rowWidth + (isQ ? head * kHeadDim : (p.numHeadsQ + head) * kHeadDim);
    int const half = p.rotaryDim / 2;
    float2 const* coef = p.cosSin + static_cast<long long>(ropePos) * half;

    __nv_bfloat16 rotated[kElemsPerThread];
#pragma unroll
    for (int i = 0; i < kElemsPerThread; ++i)
    {
        int const d = lane * kElemsPerThread + i;
        bool const firstHalf = d < half;
        float const x = __bfloat162float(src[d]);
        float const pair = __bfloat162float(src[firstHalf ? d + half : d - half]);
        float2 const cs = d < p.rotaryDim ? coef[firstHalf ? d : d - half] : make_float2(1.0f, 0.0f);
        float const s = firstHalf ? -cs.y : cs.y;
        rotated[i] = __float2bfloat16(rotate<kRopeMode>(cs.x, x, s, pair));
    }

    if (isQ)
    {
        storeE4m3Pairs(
            p.qOut + (static_cast<long long>(token) * p.numHeadsQ + head) * kHeadDim + lane * kElemsPerThread, rotated);
        return;
    }

    int const block = kvPos / p.tokensPerBlock;
    long long const inPage = (static_cast<long long>(head) * p.tokensPerBlock + kvPos % p.tokensPerBlock) * kHeadDim
        + lane * kElemsPerThread;
    long long const pageElems = static_cast<long long>(p.numHeadsKv) * p.tokensPerBlock * kHeadDim;
    int const* table = p.blockTable + request * p.blockTableRequestStride;
    long long const kPage = table[block];
    long long const vPage = table[p.blockTableKvStride + block];
    storeE4m3Pairs(p.pool + kPage * pageElems + inPage, rotated);

    __nv_bfloat16 const* vSrc = p.qkv + static_cast<long long>(token) * rowWidth
        + (p.numHeadsQ + p.numHeadsKv + head) * kHeadDim + lane * kElemsPerThread;
    __nv_bfloat16 v[kElemsPerThread];
#pragma unroll
    for (int i = 0; i < kElemsPerThread; ++i)
    {
        v[i] = vSrc[i];
    }
    storeE4m3Pairs(p.pool + vPage * pageElems + inPage, v);
}

//! With PDL and maxSharedCarveout, the maximum shared-memory carveout lets the full-context decode's large-smem CTAs
//! (the dependent this kernel releases right after its wait) co-reside on the SMs this small kernel occupies. Set once
//! per instantiation.
template <int kRopeMode>
cudaError_t launchWithMaxSharedCarveout(
    cudaLaunchConfig_t const& config, Params const& params, bool pdl, bool maxSharedCarveout)
{
    auto const kernel = drafterRopeFp8KvKernel<kRopeMode>;
    if (pdl && maxSharedCarveout)
    {
        static cudaError_t const carveout = cudaFuncSetAttribute(
            kernel, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared);
        if (carveout != cudaSuccess)
        {
            return carveout;
        }
    }
    return cudaLaunchKernelEx(&config, kernel, params);
}

} // namespace

cudaError_t launch(Params const& params, bool pdl, bool maxSharedCarveout, cudaStream_t stream)
{
    int const warps = params.numTokens * (params.numHeadsQ + params.numHeadsKv);
    int const warpsPerBlock = kBlockSize / 32;
    cudaLaunchConfig_t config = {};
    config.gridDim = dim3((warps + warpsPerBlock - 1) / warpsPerBlock);
    config.blockDim = dim3(kBlockSize);
    config.dynamicSmemBytes = 0;
    config.stream = stream;
    cudaLaunchAttribute attribute[1];
    attribute[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attribute[0].val.programmaticStreamSerializationAllowed = pdl ? 1 : 0;
    config.attrs = attribute;
    config.numAttrs = 1;
    switch (params.ropeMode)
    {
    case 0: return launchWithMaxSharedCarveout<0>(config, params, pdl, maxSharedCarveout);
    case 1: return launchWithMaxSharedCarveout<1>(config, params, pdl, maxSharedCarveout);
    case 2: return launchWithMaxSharedCarveout<2>(config, params, pdl, maxSharedCarveout);
    default: return cudaErrorInvalidValue;
    }
}

} // namespace drafterRopeFp8Kv
} // namespace minimaxM3Sm107Tp4
} // namespace kernels

TRTLLM_NAMESPACE_END
