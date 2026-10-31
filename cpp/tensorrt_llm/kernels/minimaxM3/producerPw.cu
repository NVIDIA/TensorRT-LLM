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

#include "tensorrt_llm/kernels/minimaxM3/producerPw.h"

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

// MiniMax-M3 NVFP4 QKV + indexer norm/RoPE/KV-insert producer with an early PDL trigger.
//
// Same math and outputs as minimaxM3Nvfp4QKVIndexerNormRopeKVInsertKernel (fusedQKNormRopeKernel.cu, built with
// --use_fast_math like the rest of the library), specialized to 1 KV head, 1 index head, head_dim 128, rotary_dim 64,
// page 128, NVFP4 main KV and FP8 index-K, with 8 (TP8) or 16 (TP4) query heads per rank. One warp per (token,
// packed head).
//
// The decode producer is latency bound: after griddepcontrol.wait the native kernel runs a chain of dependent global
// loads (packed row -> norm weight -> position id -> RoPE row -> cache slot -> quant scale). This kernel reads the
// constant norm weights and KV quant scale before griddepcontrol.wait and issues the packed row, position id and cache
// slot loads together after it, followed by the RoPE row. With `early` it also calls griddepcontrol.launch_dependents
// on entry: consumers read its outputs only after their own griddepcontrol.wait, which waits for this grid to complete,
// so the early trigger only lets the next kernel launch and run its pre-wait prologue while this one runs.

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{
namespace minimaxM3
{
namespace producerPw
{
namespace
{

constexpr int kHeadDim = 128;
constexpr int kRotaryDim = 64;
constexpr int kPageSize = 128;
constexpr int kElems = kHeadDim / 32;
constexpr int kHeadsKV = 1;
constexpr int kHeadsIndex = 1;
constexpr int kBlock = 256;

__device__ __forceinline__ float rcp_approx_ftz(float a)
{
    float b;
    asm volatile("rcp.approx.ftz.f32 %0, %1;\n" : "=f"(b) : "f"(a));
    return b;
}

__device__ __forceinline__ uint32_t sf_group_mask4()
{
    uint32_t laneId;
    asm("mov.u32 %0, %%laneid;" : "=r"(laneId));
    return 0xFu << (laneId & ~3u);
}

__device__ __forceinline__ uint32_t fp32x8_to_e2m1(float (&a)[8])
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    uint32_t val;
    asm volatile(
        "{\n"
        ".reg .b8 byte0;\n"
        ".reg .b8 byte1;\n"
        ".reg .b8 byte2;\n"
        ".reg .b8 byte3;\n"
        "cvt.rn.satfinite.e2m1x2.f32   byte0, %2, %1;\n"
        "cvt.rn.satfinite.e2m1x2.f32   byte1, %4, %3;\n"
        "cvt.rn.satfinite.e2m1x2.f32   byte2, %6, %5;\n"
        "cvt.rn.satfinite.e2m1x2.f32   byte3, %8, %7;\n"
        "mov.b32 %0, {byte0, byte1, byte2, byte3};\n"
        "}"
        : "=r"(val)
        : "f"(a[0]), "f"(a[1]), "f"(a[2]), "f"(a[3]), "f"(a[4]), "f"(a[5]), "f"(a[6]), "f"(a[7]));
    return val;
#else
    return 0;
#endif
}

__device__ __forceinline__ void store_fp8x4(__nv_fp8_e4m3* out, int64_t off, float const (&e)[kElems])
{
    __nv_fp8x2_e4m3 const low(make_float2(e[0], e[1]));
    __nv_fp8x2_e4m3 const high(make_float2(e[2], e[3]));
    *reinterpret_cast<uint32_t*>(out + off) = static_cast<uint32_t>(low.__x) | (static_cast<uint32_t>(high.__x) << 16);
}

__device__ __forceinline__ uint16_t quantize_fp4x4(float const (&e)[4], float secondLevelScale, uint8_t* scaleOut)
{
    __nv_bfloat16 rounded[4];
    float localMax = 0.0F;
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        rounded[i] = __float2bfloat16_rn(e[i]);
        localMax = fmaxf(localMax, fabsf(__bfloat162float(rounded[i])));
    }
    uint32_t const groupMask = sf_group_mask4();
    localMax = fmaxf(localMax, __shfl_xor_sync(groupMask, localMax, 1));
    localMax = fmaxf(localMax, __shfl_xor_sync(groupMask, localMax, 2));
    float const scaleValue = secondLevelScale * (localMax * rcp_approx_ftz(6.0F));
    __nv_fp8_e4m3 const encodedScale(scaleValue);
    if (scaleOut != nullptr)
        *scaleOut = encodedScale.__x;
    float const narrowedScale = static_cast<float>(encodedScale);
    float const outputScale
        = localMax != 0.0F ? rcp_approx_ftz(narrowedScale * rcp_approx_ftz(secondLevelScale)) : 0.0F;
    float q[8] = {};
#pragma unroll
    for (int i = 0; i < 4; ++i)
        q[i] = __bfloat162float(rounded[i]) * outputScale;
    return static_cast<uint16_t>(fp32x8_to_e2m1(q));
}

template <int kHeadsQ, bool kEarly>
__global__ void __launch_bounds__(kBlock) m3_producer_kernel(Params p)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    constexpr int kTotalHeads = kHeadsQ + 2 * kHeadsKV + kHeadsIndex + 1;
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    if constexpr (kEarly)
    {
        asm volatile("griddepcontrol.launch_dependents;");
    }
#endif
    int const warpId = threadIdx.x / 32;
    int const laneId = threadIdx.x % 32;
    int const globalWarp = blockIdx.x * (kBlock / 32) + warpId;
    int const tokenIdx = globalWarp / kTotalHeads;
    int const localHead = globalWarp % kTotalHeads;
    if (tokenIdx >= p.numTokens)
        return;

    constexpr int kBegin = kHeadsQ;
    constexpr int vBegin = kBegin + kHeadsKV;
    constexpr int indexQBegin = vBegin + kHeadsKV;
    constexpr int indexKHead = indexQBegin + kHeadsIndex;
    bool const isQ = localHead < kBegin;
    bool const isK = localHead >= kBegin && localHead < vBegin;
    bool const isV = localHead >= vBegin && localHead < indexQBegin;
    bool const isIndexQ = localHead >= indexQBegin && localHead < indexKHead;
    bool const isIndexK = localHead == indexKHead;

    // Only constant data is read before the wait: norm weights and the KV quant scale.
    float w[kElems] = {};
    float quantScale = 0.0F;
    __nv_bfloat16 const* normWeight
        = isQ ? p.qWeight : (isK ? p.kWeight : (isIndexQ ? p.indexQWeight : p.indexKWeight));
    if (!isV)
    {
        uint2 const wv = *reinterpret_cast<uint2 const*>(normWeight + laneId * kElems);
        float2 const w01 = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162 const*>(&wv.x));
        float2 const w23 = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162 const*>(&wv.y));
        w[0] = w01.x;
        w[1] = w01.y;
        w[2] = w23.x;
        w[3] = w23.y;
    }
    if (isK || isV)
    {
        quantScale = p.kvQuantScale[isV ? 2 : 1];
    }

    int64_t const inputOffset = (static_cast<int64_t>(tokenIdx) * kTotalHeads + localHead) * kHeadDim + laneId * kElems;
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
    // Position ids and cache slots are per-step metadata, read after the wait. The packed row, the position id and
    // the slot are independent loads, and the RoPE row load is issued before the norm reduction, so the
    // native kernel's serial chain of dependent loads becomes two round trips.
    uint2 const packed = *reinterpret_cast<uint2 const*>(p.packed + inputOffset);
    bool const rotates = !isV && laneId * kElems < kRotaryDim;
    int const positionId = rotates ? p.positionIds[tokenIdx] : 0;
    int const slot = (isK || isV || isIndexK) ? p.outCacheLoc[tokenIdx] : 0;
    float cosine[kElems] = {};
    float sine[kElems] = {};
    if (rotates)
    {
        int64_t const ropeRow = static_cast<int64_t>(positionId) * kRotaryDim;
        int const coefficient = (laneId * kElems) % (kRotaryDim / 2);
        float4 const c = *reinterpret_cast<float4 const*>(p.rotaryCosSin + ropeRow + coefficient);
        float4 const s = *reinterpret_cast<float4 const*>(p.rotaryCosSin + ropeRow + kRotaryDim / 2 + coefficient);
        cosine[0] = c.x;
        cosine[1] = c.y;
        cosine[2] = c.z;
        cosine[3] = c.w;
        sine[0] = s.x;
        sine[1] = s.y;
        sine[2] = s.z;
        sine[3] = s.w;
    }

    float e[kElems];
    float sumSquares = 0.0F;
#pragma unroll
    for (int pair = 0; pair < 2; ++pair)
    {
        float2 const v = __bfloat1622float2(
            *reinterpret_cast<__nv_bfloat162 const*>(reinterpret_cast<uint const*>(&packed) + pair));
        e[2 * pair] = v.x;
        e[2 * pair + 1] = v.y;
        if (!isV)
            sumSquares += v.x * v.x + v.y * v.y;
    }

    if (!isV)
    {
#pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1)
            sumSquares += __shfl_xor_sync(0xffffffff, sumSquares, mask, 32);
        float const rmsReciprocal = rsqrtf(sumSquares / static_cast<float>(kHeadDim) + p.eps);
#pragma unroll
        for (int i = 0; i < kElems; ++i)
        {
            e[i] *= rmsReciprocal * (1.0F + w[i]);
        }
        constexpr int kPairOffset = (kRotaryDim / 2) / kElems;
#pragma unroll
        for (int i = 0; i < kElems; ++i)
        {
            int const dim = laneId * kElems + i;
            float paired = __shfl_xor_sync(0xffffffff, e[i], kPairOffset);
            if (dim < kRotaryDim)
            {
                if (dim < kRotaryDim / 2)
                    paired = -paired;
                // The native kernel's contraction of e * cos + paired * sin; spelled out so this build matches it.
                e[i] = __fmaf_rn(paired, sine[i], __fmul_rn(e[i], cosine[i]));
            }
        }
    }

    if (isQ)
    {
        store_fp8x4(p.qOut, (static_cast<int64_t>(tokenIdx) * kHeadsQ + localHead) * kHeadDim + laneId * kElems, e);
        return;
    }
    if (isIndexQ)
    {
        store_fp8x4(p.indexQOut,
            (static_cast<int64_t>(tokenIdx) * kHeadsIndex + localHead - indexQBegin) * kHeadDim + laneId * kElems, e);
        return;
    }

    if (slot < 0)
        return;
    int const page = slot / kPageSize;
    if (page >= p.numPages)
        return;
    int const withinPage = slot & (kPageSize - 1);
    if (isIndexK)
    {
        store_fp8x4(p.indexK,
            static_cast<int64_t>(page) * p.indexPageStride + static_cast<int64_t>(withinPage) * p.indexTokenStride
                + laneId * kElems,
            e);
        return;
    }

    int const plane = isV ? 1 : 0;
    int const scaleColumn = laneId / 4;
    int64_t const scaleWithinPage = isV
        ? static_cast<int64_t>(withinPage / 4) * 32 + static_cast<int64_t>(scaleColumn) * 4 + withinPage % 4
        : static_cast<int64_t>(withinPage) * 8 + scaleColumn;
    int64_t const scaleOffset = static_cast<int64_t>(page) * p.scalePageStride
        + static_cast<int64_t>(plane) * p.scalePlaneStride + scaleWithinPage;
    uint8_t* scaleOut = laneId % 4 == 0 ? p.kvScale + scaleOffset : nullptr;
    uint16_t const packedFp4 = quantize_fp4x4(e, quantScale, scaleOut);
    int64_t const dataOffset = static_cast<int64_t>(page) * p.dataPageStride
        + static_cast<int64_t>(plane) * p.dataPlaneStride + static_cast<int64_t>(withinPage) * p.dataTokenStride
        + laneId * 2;
    *reinterpret_cast<uint16_t*>(p.kvData + dataOffset) = packedFp4;
#endif
}

template <int kHeadsQ, bool kEarly>
cudaError_t launch(Params const& p, int numTokens, bool pdl, cudaStream_t stream)
{
    constexpr int kTotalHeads = kHeadsQ + 2 * kHeadsKV + kHeadsIndex + 1;
    int const warps = numTokens * kTotalHeads;
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3((warps + kBlock / 32 - 1) / (kBlock / 32));
    cfg.blockDim = dim3(kBlock);
    cfg.stream = stream;
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = pdl ? 1 : 0;
    cfg.attrs = attr;
    cfg.numAttrs = 1;
    return cudaLaunchKernelEx(&cfg, m3_producer_kernel<kHeadsQ, kEarly>, p);
}

} // namespace

cudaError_t invokeProducerPw(Params const& p, int numTokens, int headsQ, bool pdl, bool early, cudaStream_t stream)
{
    if (numTokens <= 0)
        return cudaSuccess;
    if (headsQ == 16)
        return early ? launch<16, true>(p, numTokens, pdl, stream) : launch<16, false>(p, numTokens, pdl, stream);
    if (headsQ == 8)
        return early ? launch<8, true>(p, numTokens, pdl, stream) : launch<8, false>(p, numTokens, pdl, stream);
    return cudaErrorInvalidValue;
}

} // namespace producerPw
} // namespace minimaxM3
} // namespace kernels

TRTLLM_NAMESPACE_END
