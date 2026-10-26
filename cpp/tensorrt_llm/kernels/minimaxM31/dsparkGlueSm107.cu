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

// DSPARK_GLUE_CPP: C++ / CUDA replacements of two DSpark drafter Triton kernels (see dsparkGlueSm107.h for the exact
// contracts). The kernel bodies are the ones gated in SpecTail's DGLUE oracle / MB (857722 / 857723: bitwise vs the
// Triton kernels on every output; _advance 15.9 -> 7.1 us at the in-model block-table width).

#include "tensorrt_llm/kernels/minimaxM31/dsparkGlueSm107.h"

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{
namespace minimaxM31
{

namespace
{

constexpr int kAdvThreads = 1024;
constexpr int kAdvUnroll = 8;
constexpr int kIdxThreads = 256;

template <typename NaccT>
__global__ void __launch_bounds__(kAdvThreads) dsparkAdvanceKernel(int64_t const* __restrict__ slots,
    int64_t const* __restrict__ rows, NaccT const* __restrict__ nacc, int64_t* __restrict__ ctxLen,
    int64_t const* __restrict__ counts, int64_t const* __restrict__ basePages, int32_t const* __restrict__ tables,
    int32_t* __restrict__ readView, int64_t* __restrict__ out, int64_t width, int64_t tablesStride, int64_t readStride,
    int64_t maxCtx, int64_t dummySlot, int64_t page, int64_t blockSize)
{
    __shared__ int64_t sRebase;
    __shared__ int64_t sRow;
    int64_t const r = blockIdx.x;
    if (threadIdx.x == 0)
    {
        int64_t const slot = slots[r];
        int64_t const row = rows[r];
        bool const isDummy = slot == dummySlot;
        int64_t newLen = min(ctxLen[slot] + static_cast<int64_t>(nacc[r]), maxCtx);
        if (!isDummy)
        {
            ctxLen[slot] = newLen;
        }
        if (r == 0)
        {
            ctxLen[dummySlot] = 0;
        }
        newLen = isDummy ? 0 : newLen;
        int64_t const allocated = max(counts[row] * page - blockSize, int64_t{0});
        int64_t const numCtx = min(newLen, allocated);
        int64_t const base = basePages[slot];
        sRebase = min(base, numCtx / page);
        out[r] = max(numCtx - base * page, int64_t{0});
        sRow = row;
    }
    __syncthreads();
    int64_t const rebase = sRebase;
    int32_t const* src = tables + sRow * tablesStride;
    int32_t* dst = readView + sRow * readStride;
    int64_t const last = width - 1;
    int64_t j = threadIdx.x;
    for (; j + (kAdvUnroll - 1) * kAdvThreads < width; j += kAdvUnroll * kAdvThreads)
    {
        int32_t v[kAdvUnroll];
#pragma unroll
        for (int u = 0; u < kAdvUnroll; ++u)
        {
            v[u] = __ldg(src + min(j + u * kAdvThreads + rebase, last));
        }
#pragma unroll
        for (int u = 0; u < kAdvUnroll; ++u)
        {
            dst[j + u * kAdvThreads] = v[u];
        }
    }
    for (; j < width; j += kAdvThreads)
    {
        dst[j] = __ldg(src + min(j + rebase, last));
    }
}

template <typename CtxT, typename BidT, typename QT>
__global__ void __launch_bounds__(kIdxThreads) dsparkIndicesKernel(CtxT const* __restrict__ numCtx,
    BidT const* __restrict__ batchIdx, QT const* __restrict__ qpos, int64_t q0, int64_t q1,
    int32_t* __restrict__ seqlens, int32_t* __restrict__ batchOut, int32_t* __restrict__ after,
    int32_t* __restrict__ indptr, int32_t* __restrict__ last, int32_t* __restrict__ append, int32_t* __restrict__ qflat,
    int32_t B, int32_t block, int32_t pagesPerSlot, int32_t pageSize)
{
    for (int32_t b = threadIdx.x; b <= B; b += blockDim.x)
    {
        indptr[b] = b * pagesPerSlot;
        if (b < B)
        {
            int32_t const s = static_cast<int32_t>(numCtx[b]);
            seqlens[b] = s;
            batchOut[b] = static_cast<int32_t>(batchIdx[b]);
            int32_t const a = s + block;
            after[b] = a;
            int32_t m = (a - 1) % pageSize;
            m = m < 0 ? m + pageSize : m; // torch's floored modulo for a positive divisor
            last[b] = m + 1;
        }
    }
    for (int32_t t = threadIdx.x; t < B * block; t += blockDim.x)
    {
        int32_t const rb = t / block;
        int32_t const j = t - rb * block;
        append[t] = static_cast<int32_t>(numCtx[rb]) + j;
        qflat[t] = static_cast<int32_t>(qpos[rb * q0 + j * q1]);
    }
}

template <typename CtxT, typename BidT>
void launchIndices(CtxT const* numCtx, BidT const* batchIdx, void const* qpos, bool qposIs64, int64_t q0, int64_t q1,
    int32_t* seqlens, int32_t* batchOut, int32_t* after, int32_t* indptr, int32_t* last, int32_t* append,
    int32_t* qflat, int32_t B, int32_t block, int32_t pagesPerSlot, int32_t pageSize, cudaStream_t stream)
{
    if (qposIs64)
    {
        dsparkIndicesKernel<CtxT, BidT, int64_t><<<1, kIdxThreads, 0, stream>>>(numCtx, batchIdx,
            static_cast<int64_t const*>(qpos), q0, q1, seqlens, batchOut, after, indptr, last, append, qflat, B, block,
            pagesPerSlot, pageSize);
    }
    else
    {
        dsparkIndicesKernel<CtxT, BidT, int32_t><<<1, kIdxThreads, 0, stream>>>(numCtx, batchIdx,
            static_cast<int32_t const*>(qpos), q0, q1, seqlens, batchOut, after, indptr, last, append, qflat, B, block,
            pagesPerSlot, pageSize);
    }
}

template <typename CtxT>
void launchIndicesBid(CtxT const* numCtx, void const* batchIdx, bool batchIdxIs64, void const* qpos, bool qposIs64,
    int64_t q0, int64_t q1, int32_t* seqlens, int32_t* batchOut, int32_t* after, int32_t* indptr, int32_t* last,
    int32_t* append, int32_t* qflat, int32_t B, int32_t block, int32_t pagesPerSlot, int32_t pageSize,
    cudaStream_t stream)
{
    if (batchIdxIs64)
    {
        launchIndices(numCtx, static_cast<int64_t const*>(batchIdx), qpos, qposIs64, q0, q1, seqlens, batchOut, after,
            indptr, last, append, qflat, B, block, pagesPerSlot, pageSize, stream);
    }
    else
    {
        launchIndices(numCtx, static_cast<int32_t const*>(batchIdx), qpos, qposIs64, q0, q1, seqlens, batchOut, after,
            indptr, last, append, qflat, B, block, pagesPerSlot, pageSize, stream);
    }
}

} // namespace

void invokeDsparkAdvance(int64_t const* slots, int64_t const* rows, void const* nacc, bool naccIs64, int64_t* ctxLen,
    int64_t const* counts, int64_t const* basePages, int32_t const* tables, int32_t* readView, int64_t* out, int64_t n,
    int64_t width, int64_t tablesStride, int64_t readStride, int64_t maxCtx, int64_t dummySlot, int64_t page,
    int64_t blockSize, cudaStream_t stream)
{
    if (n <= 0)
    {
        return;
    }
    if (naccIs64)
    {
        dsparkAdvanceKernel<int64_t><<<static_cast<unsigned>(n), kAdvThreads, 0, stream>>>(slots, rows,
            static_cast<int64_t const*>(nacc), ctxLen, counts, basePages, tables, readView, out, width, tablesStride,
            readStride, maxCtx, dummySlot, page, blockSize);
    }
    else
    {
        dsparkAdvanceKernel<int32_t><<<static_cast<unsigned>(n), kAdvThreads, 0, stream>>>(slots, rows,
            static_cast<int32_t const*>(nacc), ctxLen, counts, basePages, tables, readView, out, width, tablesStride,
            readStride, maxCtx, dummySlot, page, blockSize);
    }
}

void invokeDsparkIndices(void const* numCtx, bool numCtxIs64, void const* batchIdx, bool batchIdxIs64, void const* qpos,
    bool qposIs64, int64_t q0, int64_t q1, int32_t* seqlens, int32_t* batchOut, int32_t* after, int32_t* indptr,
    int32_t* last, int32_t* append, int32_t* qflat, int32_t B, int32_t block, int32_t pagesPerSlot, int32_t pageSize,
    cudaStream_t stream)
{
    if (numCtxIs64)
    {
        launchIndicesBid(static_cast<int64_t const*>(numCtx), batchIdx, batchIdxIs64, qpos, qposIs64, q0, q1, seqlens,
            batchOut, after, indptr, last, append, qflat, B, block, pagesPerSlot, pageSize, stream);
    }
    else
    {
        launchIndicesBid(static_cast<int32_t const*>(numCtx), batchIdx, batchIdxIs64, qpos, qposIs64, q0, q1, seqlens,
            batchOut, after, indptr, last, append, qflat, B, block, pagesPerSlot, pageSize, stream);
    }
}

} // namespace minimaxM31
} // namespace kernels

TRTLLM_NAMESPACE_END
