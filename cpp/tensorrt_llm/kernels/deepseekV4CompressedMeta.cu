/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "tensorrt_llm/kernels/deepseekV4CompressedMeta.h"

#include <algorithm>
#include <cstdint>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

namespace
{
constexpr int32_t kThreadsPerBlock = 256;

// Binary search for the request owning a compact output index: the largest j
// with cu[j] <= value, i.e. searchsorted(cu[1:], value, right=True).
__device__ __forceinline__ int32_t upperBoundRequest(int32_t const* __restrict__ cu, int32_t numEntries, int32_t value)
{
    int32_t lo = 0;
    int32_t hi = numEntries;
    while (lo < hi)
    {
        int32_t const mid = lo + ((hi - lo) >> 1);
        if (cu[mid + 1] <= value)
        {
            lo = mid + 1;
        }
        else
        {
            hi = mid;
        }
    }
    return lo;
}

// ── Per-ratio compressed/past/new KV lens + padded exclusive scan of new_comp.
//
// One block per ratio. `batchSize` is the scheduler batch (a few hundred), so a
// single-block shared-memory scan keeps the step at one launch total instead of
// four ATen calls per ratio.
template <int kMaxBatch>
__global__ void computePerRatioKvLensKernel(int32_t const* __restrict__ kvLens,
    int32_t const* __restrict__ cachedTokens, PerRatioKvLensParams params, int32_t batchSize)
{
    // Ping-pong buffers so a scan pass never reads a slot another thread is
    // concurrently writing.
    __shared__ int32_t buffers[2][kMaxBatch];

    int32_t const ratioId = static_cast<int32_t>(blockIdx.x);
    int32_t const ratio = params.ratios[ratioId];
    int32_t* __restrict__ compressedOut = params.compressedKvLens[ratioId];
    int32_t* __restrict__ pastOut = params.pastKvLens[ratioId];
    int32_t* __restrict__ newCompOut = params.newCompKvLens[ratioId];
    int32_t* __restrict__ cuOut = params.cuNewCompKv[ratioId];

    // Thread-uniform trip count: batchSize and blockDim.x are both uniform, so
    // every thread reaches every __syncthreads() below.
    int32_t const stride = static_cast<int32_t>(blockDim.x);
    int32_t const rounded = ((batchSize + stride - 1) / stride) * stride;

    for (int32_t i = static_cast<int32_t>(threadIdx.x); i < rounded; i += stride)
    {
        if (i < batchSize)
        {
            int32_t const compressedKv = kvLens[i] / ratio;
            int32_t const pastKv = cachedTokens[i] / ratio;
            compressedOut[i] = compressedKv;
            pastOut[i] = pastKv;
            int32_t const newComp = compressedKv - pastKv;
            newCompOut[i] = newComp;
            buffers[0][i] = newComp;
        }
    }
    __syncthreads();

    // Inclusive Hillis-Steele scan over new_comp.
    int32_t src = 0;
    for (int32_t offset = 1; offset < batchSize; offset <<= 1)
    {
        int32_t const dst = src ^ 1;
        for (int32_t i = static_cast<int32_t>(threadIdx.x); i < rounded; i += stride)
        {
            if (i < batchSize)
            {
                buffers[dst][i] = buffers[src][i] + (i >= offset ? buffers[src][i - offset] : 0);
            }
        }
        __syncthreads();
        src = dst;
    }

    // Shifted by one on write-out: the padded exclusive scan that the python
    // code built with pad(cumsum(x), (1, 0)).
    if (threadIdx.x == 0)
    {
        cuOut[0] = 0;
    }
    for (int32_t i = static_cast<int32_t>(threadIdx.x); i < batchSize; i += stride)
    {
        cuOut[i + 1] = buffers[src][i];
    }
}

// ── Compressed mask: for each compact token, real token or decode padding?
__global__ void computeCompressedMaskKernel(CompressedMaskParams params, int32_t batchSize)
{
    int32_t const ratioId = static_cast<int32_t>(blockIdx.y);
    int32_t const total = params.totalTokens[ratioId];
    int32_t const* __restrict__ newComp = params.newCompKvLens[ratioId];
    int32_t const* __restrict__ cu = params.cuNewCompKv[ratioId];
    bool* __restrict__ out = params.mask[ratioId];

    for (int32_t t = blockIdx.x * blockDim.x + threadIdx.x; t < total; t += gridDim.x * blockDim.x)
    {
        int32_t seqIdx = upperBoundRequest(cu, batchSize, t);
        seqIdx = min(seqIdx, batchSize - 1);
        out[t] = (t - cu[seqIdx]) < newComp[seqIdx];
    }
}

// ── Context compressed position ids.
__global__ void computeCtxCompressedPositionIdsKernel(CompressedPositionIdsParams params, int32_t numContexts)
{
    int32_t const ratioId = static_cast<int32_t>(blockIdx.y);
    int32_t const total = params.counts[ratioId];
    int32_t const ratio = params.ratios[ratioId];
    int32_t const* __restrict__ pastKv = params.pastKvLens[ratioId];
    int32_t const* __restrict__ cu = params.cuNewCompKv[ratioId];
    int32_t* __restrict__ out = params.positionIds[ratioId];

    for (int32_t t = blockIdx.x * blockDim.x + threadIdx.x; t < total; t += gridDim.x * blockDim.x)
    {
        int32_t const reqIdx = upperBoundRequest(cu, numContexts, t);
        out[t] = (pastKv[reqIdx] + (t - cu[reqIdx])) * ratio;
    }
}

// ── Generation compressed position ids, in compact compressor output order.
__global__ void computeGenCompressedPositionIdsKernel(
    CompressedPositionIdsParams params, int32_t numContexts, int32_t batchSize)
{
    int32_t const ratioId = static_cast<int32_t>(blockIdx.y);
    int32_t const genComp = params.counts[ratioId];
    int32_t const ratio = params.ratios[ratioId];
    int32_t const outputOffset = params.offsets[ratioId];
    int32_t const* __restrict__ pastKv = params.pastKvLens[ratioId];
    int32_t const* __restrict__ cu = params.cuNewCompKv[ratioId];
    int32_t* __restrict__ out = params.positionIds[ratioId];

    for (int32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < genComp; i += gridDim.x * blockDim.x)
    {
        int32_t const outputIdx = i + outputOffset;
        int32_t reqIdx = upperBoundRequest(cu, batchSize, outputIdx);
        reqIdx = min(max(reqIdx, numContexts), batchSize - 1);
        out[outputIdx] = (pastKv[reqIdx] + (outputIdx - cu[reqIdx])) * ratio;
    }
}

int32_t gridX(int32_t work, int32_t cap)
{
    return std::min((work + kThreadsPerBlock - 1) / kThreadsPerBlock, cap);
}
} // namespace

void invokeDeepseekV4ComputePerRatioKvLens(int32_t const* kvLens, int32_t const* cachedTokens,
    PerRatioKvLensParams const& params, int32_t numRatios, int32_t batchSize, cudaStream_t stream)
{
    if (numRatios <= 0 || batchSize <= 0)
    {
        return;
    }
    dim3 const grid(static_cast<uint32_t>(numRatios));
    dim3 const block(static_cast<uint32_t>(kThreadsPerBlock));
    // Templated on a shared-memory batch bound; pick the smallest that fits.
    if (batchSize <= 512)
    {
        computePerRatioKvLensKernel<512><<<grid, block, 0, stream>>>(kvLens, cachedTokens, params, batchSize);
    }
    else if (batchSize <= 2048)
    {
        computePerRatioKvLensKernel<2048><<<grid, block, 0, stream>>>(kvLens, cachedTokens, params, batchSize);
    }
    else
    {
        computePerRatioKvLensKernel<kMaxScanBatch><<<grid, block, 0, stream>>>(kvLens, cachedTokens, params, batchSize);
    }
}

void invokeDeepseekV4ComputeCompressedMask(CompressedMaskParams const& params, int32_t maxTotalTokens,
    int32_t numRatios, int32_t batchSize, cudaStream_t stream)
{
    if (numRatios <= 0 || batchSize <= 0 || maxTotalTokens <= 0)
    {
        return;
    }
    dim3 const grid(static_cast<uint32_t>(gridX(maxTotalTokens, 1024)), static_cast<uint32_t>(numRatios));
    dim3 const block(static_cast<uint32_t>(kThreadsPerBlock));
    computeCompressedMaskKernel<<<grid, block, 0, stream>>>(params, batchSize);
}

void invokeDeepseekV4ComputeCtxCompressedPositionIds(CompressedPositionIdsParams const& params, int32_t maxCount,
    int32_t numRatios, int32_t numContexts, cudaStream_t stream)
{
    if (numRatios <= 0 || numContexts <= 0 || maxCount <= 0)
    {
        return;
    }
    dim3 const grid(static_cast<uint32_t>(gridX(maxCount, 2048)), static_cast<uint32_t>(numRatios));
    dim3 const block(static_cast<uint32_t>(kThreadsPerBlock));
    computeCtxCompressedPositionIdsKernel<<<grid, block, 0, stream>>>(params, numContexts);
}

void invokeDeepseekV4ComputeGenCompressedPositionIds(CompressedPositionIdsParams const& params, int32_t maxCount,
    int32_t numRatios, int32_t numContexts, int32_t batchSize, cudaStream_t stream)
{
    if (numRatios <= 0 || maxCount <= 0 || batchSize <= 0)
    {
        return;
    }
    dim3 const grid(static_cast<uint32_t>(gridX(maxCount, 1024)), static_cast<uint32_t>(numRatios));
    dim3 const block(static_cast<uint32_t>(kThreadsPerBlock));
    computeGenCompressedPositionIdsKernel<<<grid, block, 0, stream>>>(params, numContexts, batchSize);
}

} // namespace kernels

TRTLLM_NAMESPACE_END
