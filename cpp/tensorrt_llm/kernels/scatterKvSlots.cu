/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/kernels/scatterKvSlots.h"

using namespace tensorrt_llm::common;

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

namespace
{

// One thread per (entry, head, vector): load one vector of K and one of V, store them to one
// or two slots. Strides are in units of Vec.
template <typename Vec>
__global__ void scatterKvSlotsKernel(Vec* __restrict__ pool, Vec const* __restrict__ k, Vec const* __restrict__ v,
    int64_t const* __restrict__ dst, int64_t const* __restrict__ dst2, int64_t const* __restrict__ src,
    int64_t numEntries, int64_t numHeads, int64_t vecsPerRow, int64_t tokensPerPage, int64_t poolPageStride,
    int64_t poolKvStride, int64_t poolHeadStride, int64_t poolSlotStride, int64_t kTokenStride, int64_t kHeadStride,
    int64_t vTokenStride, int64_t vHeadStride)
{
    int64_t const idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t const total = numEntries * numHeads * vecsPerRow;
    if (idx >= total)
    {
        return;
    }
    int64_t const vec = idx % vecsPerRow;
    int64_t const rest = idx / vecsPerRow;
    int64_t const head = rest % numHeads;
    int64_t const entry = rest / numHeads;

    int64_t const token = src ? src[entry] : entry;
    Vec const kv = k[token * kTokenStride + head * kHeadStride + vec];
    Vec const vv = v[token * vTokenStride + head * vHeadStride + vec];

    int64_t const headOffset = head * poolHeadStride + vec;
    int64_t slotId = dst[entry];
    int64_t base = (slotId / tokensPerPage) * poolPageStride + (slotId % tokensPerPage) * poolSlotStride + headOffset;
    pool[base] = kv;
    pool[base + poolKvStride] = vv;
    if (dst2)
    {
        slotId = dst2[entry];
        base = (slotId / tokensPerPage) * poolPageStride + (slotId % tokensPerPage) * poolSlotStride + headOffset;
        pool[base] = kv;
        pool[base + poolKvStride] = vv;
    }
}

int pickVecBytes(ScatterKvSlotsParams const& p, int elemSize)
{
    for (int bytes : {16, 8, 4, 2, 1})
    {
        if (bytes < elemSize)
        {
            break;
        }
        bool ok = p.headDim * elemSize % bytes == 0;
        for (void const* ptr : {static_cast<void const*>(p.pool), p.k, p.v})
        {
            ok = ok && reinterpret_cast<int64_t>(ptr) % bytes == 0;
        }
        for (int64_t stride : {p.poolPageStride, p.poolKvStride, p.poolHeadStride, p.poolSlotStride, p.kTokenStride,
                 p.kHeadStride, p.vTokenStride, p.vHeadStride})
        {
            ok = ok && stride * elemSize % bytes == 0;
        }
        if (ok)
        {
            return bytes;
        }
    }
    return elemSize;
}

template <typename Vec>
void launch(ScatterKvSlotsParams const& p, int elemSize, cudaStream_t stream)
{
    int64_t const ratio = static_cast<int64_t>(sizeof(Vec)) / elemSize; // elements per vector
    int64_t const vecsPerRow = p.headDim / ratio;
    int64_t const total = p.numEntries * p.numHeads * vecsPerRow;
    constexpr int kThreads = 256;
    unsigned const blocks = static_cast<unsigned>((total + kThreads - 1) / kThreads);
    scatterKvSlotsKernel<Vec><<<blocks, kThreads, 0, stream>>>(static_cast<Vec*>(p.pool), static_cast<Vec const*>(p.k),
        static_cast<Vec const*>(p.v), p.dst, p.dst2, p.src, p.numEntries, p.numHeads, vecsPerRow, p.tokensPerPage,
        p.poolPageStride / ratio, p.poolKvStride / ratio, p.poolHeadStride / ratio, p.poolSlotStride / ratio,
        p.kTokenStride / ratio, p.kHeadStride / ratio, p.vTokenStride / ratio, p.vHeadStride / ratio);
    check_cuda_error(cudaGetLastError());
}

} // namespace

void invokeScatterKvSlots(ScatterKvSlotsParams const& params, int elemSize, cudaStream_t stream)
{
    if (params.numEntries <= 0 || params.numHeads <= 0 || params.headDim <= 0)
    {
        return;
    }
    switch (pickVecBytes(params, elemSize))
    {
    case 16: launch<uint4>(params, elemSize, stream); break;
    case 8: launch<uint2>(params, elemSize, stream); break;
    case 4: launch<uint32_t>(params, elemSize, stream); break;
    case 2: launch<uint16_t>(params, elemSize, stream); break;
    case 1: launch<uint8_t>(params, elemSize, stream); break;
    default: TLLM_THROW("scatterKvSlots: unsupported element size %d bytes", elemSize);
    }
}

} // namespace kernels

TRTLLM_NAMESPACE_END
