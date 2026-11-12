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

#include "glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/csrc/indexerLayerRopeKvAppend.h"
#include "tensorrt_llm/modeling_v2/common/namespace.h"

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/kernels/decoderMaskedMultiheadAttentionUtils.h"

#include <cuda_fp8.h>

#include <algorithm>

TRTLLM_MV2_NAMESPACE_BEGIN

namespace indexer_layer_front
{

using rope_kv_append::kKvLoraRank;
using rope_kv_append::kLatentDim;
using rope_kv_append::kRopeDim;
using rope_kv_append::RopeAppendParams;

namespace
{

constexpr int32_t kRopeThreads = 256;
constexpr int32_t kEltsPerVec = 8;                           // bf16 elements per 16-byte load
constexpr int32_t kNopeVecs = kKvLoraRank / kEltsPerVec;     // 64
constexpr int32_t kRopeVecs = kRopeDim / kEltsPerVec;        // 8
constexpr int32_t kTotalVecsPerHead = kNopeVecs + kRopeVecs; // 72
constexpr int32_t kIdxBytesPerItem = 4;

__device__ __forceinline__ void pdlWait()
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
}

__device__ __forceinline__ void pdlLaunchDependents()
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
#endif
}

//! Same arithmetic as quantCopy<__nv_bfloat16, 8> in mlaKernels.cu: bf16 -> fp32, scale, saturating E4M3.
__device__ __forceinline__ uint2 quantizeBf16x8(uint4 const& src, float scale)
{
    __nv_bfloat162 const* in = reinterpret_cast<__nv_bfloat162 const*>(&src);
    uint2 out;
    __nv_fp8x2_e4m3* dst = reinterpret_cast<__nv_fp8x2_e4m3*>(&out);
#pragma unroll
    for (int32_t j = 0; j < 4; ++j)
    {
        float2 v = __bfloat1622float2(in[j]);
        v.x *= scale;
        v.y *= scale;
        dst[j] = __nv_fp8x2_e4m3(v);
    }
    return out;
}

//! GPT-J RoPE on 8 bf16 values (4 adjacent pairs), identical to the per-pair transform of the native kernel.
__device__ __forceinline__ void rotateBf16x8(uint4& data, float2 const (&coef)[4])
{
    __nv_bfloat162* pairs = reinterpret_cast<__nv_bfloat162*>(&data);
#pragma unroll
    for (int32_t j = 0; j < 4; ++j)
    {
        pairs[j] = kernels::mmha::rotary_embedding_transform(pairs[j], coef[j]);
    }
}

//! Native unravel of a flat byte index over the strided 4-D indexer cache view (indexerKCacheScatter.cu);
//! 32-bit arithmetic when the index fits, same result.
__device__ __forceinline__ int64_t idxFlatToOffset(int64_t flat, int32_t const (&dims)[4], int64_t const (&strides)[4])
{
    int32_t i0, i1, i2, i3;
    if (flat >= 0 && flat <= 0x7fffffff)
    {
        uint32_t f = static_cast<uint32_t>(flat);
        i3 = static_cast<int32_t>(f % static_cast<uint32_t>(dims[3]));
        f /= static_cast<uint32_t>(dims[3]);
        i2 = static_cast<int32_t>(f % static_cast<uint32_t>(dims[2]));
        f /= static_cast<uint32_t>(dims[2]);
        i1 = static_cast<int32_t>(f % static_cast<uint32_t>(dims[1]));
        f /= static_cast<uint32_t>(dims[1]);
        i0 = static_cast<int32_t>(f);
    }
    else
    {
        i3 = flat % dims[3];
        flat /= dims[3];
        i2 = flat % dims[2];
        flat /= dims[2];
        i1 = flat % dims[1];
        flat /= dims[1];
        i0 = flat;
    }
    return i0 * strides[0] + i1 * strides[1] + i2 * strides[2] + i3 * strides[3];
}

enum class RopeItem : int32_t
{
    kNone = 0,
    kQNope,
    kQRope,
    kKRope,
    kKNope,
    kIdxData,
    kIdxScale,
};

//! One thread per 16-byte work item of one token. Per token the item space is
//!   [0, H*64)            q_nope FP8 quantization      (head = i / 64, vec = i % 64)
//!   [H*64, H*72)         q_pe RoPE + FP8               (head, vec)
//!   [H*72, H*72+8)       k_pe RoPE + FP8 cache write   (vec)
//!   [H*72+8, H*72+72)    compressed-KV FP8 cache write (vec)
//!   then idxHeadBytes/4 indexer K data items (4 bytes each) and one indexer scale item.
//! Per-step metadata (cache lengths, KV block offsets, indexer slot mappings) and the item's source data are read only
//! after griddepcontrol.wait (see the header). Identical to ropeAppendScatterKernel except for the trigger position.
template <bool kEarlyTrigger>
__global__ void __launch_bounds__(kRopeThreads) ropeAppendScatterZ14Kernel(RopeAppendParams const p)
{
    int32_t const token = blockIdx.x;
    int32_t const item = blockIdx.y * kRopeThreads + threadIdx.x;
    int32_t const qNopeEnd = p.numHeads * kNopeVecs;
    int32_t const qRopeEnd = qNopeEnd + p.numHeads * kRopeVecs;
    int32_t const kRopeEnd = qRopeEnd + kRopeVecs;
    int32_t const kNopeEnd = kRopeEnd + kNopeVecs;
    int32_t const idxDataEnd = kNopeEnd + p.idxHeadBytes / kIdxBytesPerItem;

    bool const ropeToken = token < p.numTokens;
    bool const idxToken = token < p.idxNumTokens;

    RopeItem kind = RopeItem::kNone;
    int32_t head = 0;
    int32_t vec = 0;
    if (item < qNopeEnd)
    {
        kind = ropeToken ? RopeItem::kQNope : RopeItem::kNone;
        head = item / kNopeVecs;
        vec = item % kNopeVecs;
    }
    else if (item < qRopeEnd)
    {
        kind = ropeToken ? RopeItem::kQRope : RopeItem::kNone;
        head = (item - qNopeEnd) / kRopeVecs;
        vec = (item - qNopeEnd) % kRopeVecs;
    }
    else if (item < kRopeEnd)
    {
        kind = ropeToken ? RopeItem::kKRope : RopeItem::kNone;
        vec = item - qRopeEnd;
    }
    else if (item < kNopeEnd)
    {
        kind = ropeToken ? RopeItem::kKNope : RopeItem::kNone;
        vec = item - kRopeEnd;
    }
    else if (item < idxDataEnd)
    {
        kind = idxToken ? RopeItem::kIdxData : RopeItem::kNone;
        vec = item - kNopeEnd;
    }
    else if (item == idxDataEnd)
    {
        kind = idxToken ? RopeItem::kIdxScale : RopeItem::kNone;
    }

    int32_t const batchIdx = token / p.seqLen;
    int32_t const localIdx = token % p.seqLen;
    bool const needPos = kind == RopeItem::kQRope || kind == RopeItem::kKRope || kind == RopeItem::kKNope;
    bool const needCoef = kind == RopeItem::kQRope || kind == RopeItem::kKRope;
    bool const needBlock = kind == RopeItem::kKRope || kind == RopeItem::kKNope;
    bool const isIdx = kind == RopeItem::kIdxData || kind == RopeItem::kIdxScale;

    // Values that never change while kernels of the step run (the quantization scales) may be read before the wait;
    // per-step metadata and producer outputs may not.
    float scale = 1.F;
    if (needBlock)
    {
        scale = p.quantScaleKv ? p.quantScaleKv[0] : 1.F;
    }
    else if (kind == RopeItem::kQNope || kind == RopeItem::kQRope)
    {
        scale = p.quantScaleQ ? p.quantScaleQ[0] : 1.F;
    }
    auto loadCoef = [&](int32_t pos, float2(&c)[4])
    {
        float2 const* src = p.cosSin + static_cast<size_t>(kRopeDim) * pos + (vec * kEltsPerVec) / 2;
#pragma unroll
        for (int32_t j = 0; j < 4; ++j)
        {
            c[j] = src[j];
        }
    };
    auto idxOffset = [&](int64_t slotData, int64_t slotScale) -> int64_t
    {
        if (slotData < 0 || slotScale < 0)
        {
            return -1;
        }
        return kind == RopeItem::kIdxData ? idxFlatToOffset(slotData + vec * kIdxBytesPerItem, p.idxDims, p.idxStrides)
                                          : idxFlatToOffset(slotScale, p.idxDims, p.idxStrides);
    };
    // Source data of this item; written by producers, so it is read after the wait.
    uint4 vdata = make_uint4(0, 0, 0, 0);
    uint32_t wdata = 0;
    auto loadSource = [&]()
    {
        switch (kind)
        {
        case RopeItem::kQNope:
            vdata = *reinterpret_cast<uint4 const*>(
                p.fusedQ + (static_cast<size_t>(token) * p.numHeads + head) * kLatentDim + vec * kEltsPerVec);
            break;
        case RopeItem::kQRope:
            vdata = *reinterpret_cast<uint4 const*>(p.qPe + static_cast<size_t>(token) * p.qPeStride
                + static_cast<size_t>(p.qPeLd) * head + vec * kEltsPerVec);
            break;
        case RopeItem::kKRope:
            vdata = *reinterpret_cast<uint4 const*>(
                p.latent + static_cast<size_t>(token) * kLatentDim + kKvLoraRank + vec * kEltsPerVec);
            break;
        case RopeItem::kKNope:
            vdata = *reinterpret_cast<uint4 const*>(
                p.latent + static_cast<size_t>(token) * kLatentDim + vec * kEltsPerVec);
            break;
        case RopeItem::kIdxData:
            wdata = *reinterpret_cast<uint32_t const*>(
                p.idxK + static_cast<int64_t>(token) * p.idxHeadBytes + vec * kIdxBytesPerItem);
            break;
        case RopeItem::kIdxScale:
            wdata = *reinterpret_cast<uint32_t const*>(p.idxScale + static_cast<int64_t>(token) * kIdxBytesPerItem);
            break;
        default: break;
        }
    };

    int32_t kvPos = 0;
    float2 coef[4];
    __nv_fp8_e4m3* kBlock = nullptr;
    int64_t idxDst = -1;
    pdlWait();
    // Issue the source-data load first so it overlaps the metadata loads and the address math below.
    loadSource();
    if (needPos)
    {
        kvPos = p.kvLens[batchIdx] - p.seqLen + localIdx;
    }
    if (needCoef)
    {
        loadCoef(kvPos, coef);
    }
    if (needBlock)
    {
        kBlock = reinterpret_cast<__nv_fp8_e4m3*>(p.kvCache.getKBlockPtr(batchIdx, kvPos));
    }
    if (isIdx)
    {
        idxDst = idxOffset(p.idxSlotData[token], p.idxSlotScale[token]);
    }
    if constexpr (kEarlyTrigger)
    {
        // Every global read of this thread's item has been issued. The block counts as triggered at the first of
        // its threads to get here (later invocations in the block have no further effect), so a dependent may be
        // resident before this block's stores below and before block (0, 0)'s scheduler-buffer stores and their
        // cache-length reads. Dependents read this kernel's outputs only after their own griddepcontrol.wait, and
        // write no global memory before it, so neither the stores nor the reads below can race with them.
        pdlLaunchDependents();
    }

    switch (kind)
    {
    case RopeItem::kQNope:
    {
        size_t const off = (static_cast<size_t>(token) * p.numHeads + head) * kLatentDim + vec * kEltsPerVec;
        *reinterpret_cast<uint2*>(p.quantQ + off) = quantizeBf16x8(vdata, scale);
        break;
    }
    case RopeItem::kQRope:
    {
        uint4 data = vdata;
        rotateBf16x8(data, coef);
        size_t const dst
            = (static_cast<size_t>(token) * p.numHeads + head) * kLatentDim + kKvLoraRank + vec * kEltsPerVec;
        *reinterpret_cast<uint2*>(p.quantQ + dst) = quantizeBf16x8(data, scale);
        break;
    }
    case RopeItem::kKRope:
    {
        uint4 data = vdata;
        rotateBf16x8(data, coef);
        int32_t const inBlockIdx = p.kvCache.getKVLocalIdx(kvPos, 0, kTotalVecsPerHead, kNopeVecs + vec);
        *reinterpret_cast<uint2*>(kBlock + inBlockIdx * kEltsPerVec) = quantizeBf16x8(data, scale);
        break;
    }
    case RopeItem::kKNope:
    {
        int32_t const inBlockIdx = p.kvCache.getKVLocalIdx(kvPos, 0, kTotalVecsPerHead, vec);
        *reinterpret_cast<uint2*>(kBlock + inBlockIdx * kEltsPerVec) = quantizeBf16x8(vdata, scale);
        if (vec == 0 && localIdx == 0)
        {
            p.cuQSeqlens[batchIdx + 1] = p.numHeads * p.seqLen * (batchIdx + 1);
        }
        break;
    }
    case RopeItem::kIdxData:
    case RopeItem::kIdxScale:
    {
        if (idxDst >= 0)
        {
            *reinterpret_cast<uint32_t*>(p.idxCache + idxDst) = wdata;
        }
        break;
    }
    default: break;
    }

    if (blockIdx.x == 0 && blockIdx.y == 0)
    {
        if (threadIdx.x == 0)
        {
            p.fmhaTileCounter[0] = 0;
            p.cuQSeqlens[0] = 0;
            float const dequantQ = p.dequantScaleQ ? p.dequantScaleQ[0] : 1.F;
            float const dequantKv = p.dequantScaleKv ? p.dequantScaleKv[0] : 1.F;
            float const quantO = p.quantScaleO ? p.quantScaleO[0] : 1.F;
            if (p.bmm1Scale)
            {
                constexpr float kLog2e = 1.4426950408889634074F;
                float const bmm1 = dequantQ * dequantKv * p.hostBmm1Scale;
                p.bmm1Scale[0] = bmm1;
                p.bmm1Scale[1] = bmm1 * kLog2e;
            }
            if (p.bmm2Scale)
            {
                p.bmm2Scale[0] = quantO * dequantKv;
            }
        }
        if (threadIdx.x < 32)
        {
            // Exclusive prefix sum of the cache lengths -> cuKvSeqlens[0..batch].
            int32_t const lane = threadIdx.x;
            int32_t const batch = p.numTokens / p.seqLen;
            int32_t carry = 0;
            for (int32_t base = 0; base <= batch; base += 32)
            {
                int32_t const idx = base + lane;
                int32_t const len = idx < batch ? p.kvLens[idx] : 0;
                int32_t incl = len;
#pragma unroll
                for (int32_t off = 1; off < 32; off <<= 1)
                {
                    int32_t const n = __shfl_up_sync(0xffffffffU, incl, off);
                    if (lane >= off)
                    {
                        incl += n;
                    }
                }
                if (idx <= batch)
                {
                    p.cuKvSeqlens[idx] = incl - len + carry;
                }
                carry += __shfl_sync(0xffffffffU, incl, 31);
            }
        }
    }

    if constexpr (!kEarlyTrigger)
    {
        // ropeAppendScatterKernel's position.
        pdlLaunchDependents();
    }
}

template <typename Kernel, typename Params>
void launchWithOptionalPdl(Kernel kernel, dim3 grid, dim3 block, bool enablePdl, cudaStream_t stream, Params const& p)
{
    cudaLaunchConfig_t config{};
    config.gridDim = grid;
    config.blockDim = block;
    config.dynamicSmemBytes = 0;
    config.stream = stream;
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[0].val.programmaticStreamSerializationAllowed = enablePdl ? 1 : 0;
    config.numAttrs = 1;
    config.attrs = attrs;
    TLLM_CUDA_CHECK(cudaLaunchKernelEx(&config, kernel, p));
}

} // namespace

void invokeIndexerLayerRopeKvAppend(
    RopeAppendParams const& params, bool enablePdl, bool earlyTrigger, cudaStream_t stream)
{
    int32_t const gridTokens = std::max(params.numTokens, params.idxNumTokens);
    if (gridTokens == 0)
    {
        return;
    }
    TLLM_CHECK_WITH_INFO(params.seqLen > 0, "seqLen must be positive");
    TLLM_CHECK_WITH_INFO(params.cosSinRows > 0, "the RoPE table must have at least one position");
    TLLM_CHECK_WITH_INFO(params.idxHeadBytes % kIdxBytesPerItem == 0, "indexer head bytes must be a multiple of 4");
    int32_t const itemsPerToken
        = params.numHeads * kTotalVecsPerHead + kTotalVecsPerHead + params.idxHeadBytes / kIdxBytesPerItem + 1;
    dim3 const grid(gridTokens, (itemsPerToken + kRopeThreads - 1) / kRopeThreads);
    dim3 const block(kRopeThreads);
    if (earlyTrigger)
    {
        launchWithOptionalPdl(ropeAppendScatterZ14Kernel<true>, grid, block, enablePdl, stream, params);
    }
    else
    {
        launchWithOptionalPdl(ropeAppendScatterZ14Kernel<false>, grid, block, enablePdl, stream, params);
    }
}

} // namespace indexer_layer_front

TRTLLM_MV2_NAMESPACE_END
