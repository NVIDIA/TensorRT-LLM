/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2024 by FlashInfer team.
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

// REJ_MC_FI: FlashInfer's chain speculative sampling on ctasPerRow CTAs per request, bitwise. See
// chainSpecSamplingMcFiSm107.h for the contract. Compiled with --use_fast_math (set on this source in
// kernels/CMakeLists.txt), as FlashInfer's JIT compiles it.

#include "minimax_m3/csrc/m31/chainSpecSamplingMcFiSm107.h"
#include "tensorrt_llm/modeling_v2/common/namespace.h"

#include <cuda/atomic>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <cstdint>

#include <cub/cub.cuh>

TRTLLM_MV2_NAMESPACE_BEGIN

namespace kernels
{
namespace minimaxM31
{

namespace
{

using tensorrt_llm::kernels::speculative_decoding::ChainSpeculativeSamplingParams;

constexpr int32_t kBlockSize = 1024;
constexpr int32_t kVecSize = 4;
constexpr int32_t kTileSize = kBlockSize * kVecSize;
constexpr int32_t kMaxTiles = kChainSpecMcMaxVocab / kTileSize;

//! One request's cross-CTA state: each chunk's residual sum and its last index with mass (-1 if none), and the
//! arrival counter (each launch adds kBlockSize in total, whatever its ctasPerRow, so it needs no reset).
struct RowWorkspace
{
    float tileSum[kMaxTiles];
    int32_t tileLast[kMaxTiles];
    unsigned arrive;
    int32_t pad[3];
};

__device__ __forceinline__ float const* draftRow(ChainSpeculativeSamplingParams const& params,
    ChainSpecMcDraftRows const& rows, int64_t row, int32_t i, int32_t numDraft, uint32_t vocabSize)
{
    if (rows.rowIds == nullptr)
    {
        return params.draftProbs + (row * numDraft + i) * static_cast<int64_t>(vocabSize);
    }
    return params.draftProbs + rows.rowIds[row] * rows.rowStride + i * rows.posStride;
}

//! A thread's 4 chunk values, max(target - draft, 0) as FlashInfer computes them (draft == nullptr: the bonus row,
//! FlashInfer's zero-filled draft vector). vocabSize % 4 == 0, so FlashInfer's per-vector guard is this one.
__device__ __forceinline__ void loadResidual(
    float const* target, float const* draft, uint32_t base, uint32_t vocabSize, float (&relu)[kVecSize])
{
    float4 q = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 d = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    if (base < vocabSize)
    {
        q = *reinterpret_cast<float4 const*>(target + base);
        if (draft != nullptr)
        {
            d = *reinterpret_cast<float4 const*>(draft + base);
        }
    }
    relu[0] = max(q.x - d.x, 0.0f);
    relu[1] = max(q.y - d.y, 0.0f);
    relu[2] = max(q.z - d.z, 0.0f);
    relu[3] = max(q.w - d.w, 0.0f);
}

// ---------------------------------------------------------------------------------------------------------------------
// FlashInfer's DeterministicInclusiveSum (include/flashinfer/sampling.cuh, FlashInfer v0.7.2). Copyright (c) 2024 by
// FlashInfer team, licensed under the Apache License, Version 2.0. Copied with its template parameters fixed
// (VEC_SIZE 4, BLOCK_THREADS 1024) and its shared array passed in; formatting kept. FlashInfer's headers are not on
// the TensorRT-LLM C++ include path.
// ---------------------------------------------------------------------------------------------------------------------
// clang-format off
__device__ __forceinline__ void deterministicInclusiveSum(const float* in_data, float* out_data,
                                                          float* smem_prefix_sum) {
  constexpr uint32_t VEC_SIZE = kVecSize;
  constexpr uint32_t BLOCK_THREADS = kBlockSize;
  float thread_data[VEC_SIZE];
  float thread_sum = 0;
#pragma unroll
  for (uint32_t i = 0; i < VEC_SIZE; ++i) {
    thread_sum += in_data[i];
    thread_data[i] = thread_sum;
  }

  float thread_exclusive_prefix_sum = thread_sum;

#pragma unroll
  for (uint32_t offset = 1; offset < 32; offset *= 2) {
    float tmp = __shfl_up_sync(0xffffffff, thread_exclusive_prefix_sum, offset);
    if ((threadIdx.x + 1) % (offset * 2) == 0) {
      thread_exclusive_prefix_sum += tmp;
    }
  }

  float warp_sum = __shfl_sync(0xffffffff, thread_exclusive_prefix_sum, threadIdx.x | 0xffffffff);
  if (threadIdx.x % 32 == 31) {
    thread_exclusive_prefix_sum = 0;
  }

#pragma unroll
  for (uint32_t offset = 16; offset >= 1; offset /= 2) {
    float tmp = __shfl_xor_sync(0xffffffff, thread_exclusive_prefix_sum, offset);
    if ((threadIdx.x + 1) % (offset * 2) == 0) {
      thread_exclusive_prefix_sum = tmp + thread_exclusive_prefix_sum;
    }
    if ((threadIdx.x + 1) % (offset * 2) == offset) {
      thread_exclusive_prefix_sum = tmp;
    }
  }

  smem_prefix_sum[threadIdx.x / 32] = warp_sum;
  __syncthreads();

  if (threadIdx.x < 32) {
    float warp_exclusive_prefix_sum =
        (threadIdx.x < BLOCK_THREADS / 32) ? smem_prefix_sum[threadIdx.x] : 0;

#pragma unroll
    for (uint32_t offset = 1; offset < 32; offset *= 2) {
      float tmp = __shfl_up_sync(0xffffffff, warp_exclusive_prefix_sum, offset);
      if ((threadIdx.x + 1) % (offset * 2) == 0) {
        warp_exclusive_prefix_sum += tmp;
      }
    }

    if (threadIdx.x % 32 == 31) {
      warp_exclusive_prefix_sum = 0;
    }

#pragma unroll
    for (uint32_t offset = 16; offset >= 1; offset /= 2) {
      float tmp = __shfl_xor_sync(0xffffffff, warp_exclusive_prefix_sum, offset);
      if ((threadIdx.x + 1) % (offset * 2) == 0) {
        warp_exclusive_prefix_sum = tmp + warp_exclusive_prefix_sum;
      }
      if ((threadIdx.x + 1) % (offset * 2) == offset) {
        warp_exclusive_prefix_sum = tmp;
      }
    }
    if (threadIdx.x < BLOCK_THREADS / 32) {
      smem_prefix_sum[threadIdx.x] = warp_exclusive_prefix_sum;
    }
  }
  __syncthreads();

#pragma unroll
  for (uint32_t i = 0; i < VEC_SIZE; ++i) {
    out_data[i] = smem_prefix_sum[threadIdx.x / 32] + thread_exclusive_prefix_sum + thread_data[i];
  }
}

// clang-format on
// ---------------------------------------------------------------------------------------------------------------------

__global__ void __launch_bounds__(kBlockSize, 1)
    chainSpecSamplingMcFiKernel(ChainSpeculativeSamplingParams params, RowWorkspace* ws, ChainSpecMcDraftRows rows)
{
    using BlockReduce = cub::BlockReduce<float, kBlockSize, cub::BLOCK_REDUCE_WARP_REDUCTIONS>;

    __shared__ union
    {
        typename BlockReduce::TempStorage reduce;
        float scan[kBlockSize / 32];
    } tempStorage;

    __shared__ float sTarget[kChainSpecMcMaxDraft];
    __shared__ float sDraft[kChainSpecMcMaxDraft];
    __shared__ int32_t sPos;
    __shared__ float sU;
    __shared__ int32_t sTileLast;
    __shared__ int32_t sIsLast;
    __shared__ float sTileSum[kMaxTiles];
    __shared__ int32_t sTileLastOf[kMaxTiles];
    __shared__ int32_t sChunk;
    __shared__ float sAggregate;
    __shared__ float sThreshold;
    __shared__ int32_t sLastValid;
    __shared__ int32_t sWinner;

    int64_t const row = blockIdx.y;
    int32_t const G = static_cast<int32_t>(gridDim.x);
    int32_t const cta = static_cast<int32_t>(blockIdx.x);
    int32_t const tid = static_cast<int32_t>(threadIdx.x);
    int32_t const numDraft = params.numDraftTokens;
    uint32_t const vocabSize = static_cast<uint32_t>(params.vocabSize);
    RowWorkspace* w = ws + row;

    // 1. FlashInfer's draws, identical in every CTA of the request.
    if (tid < numDraft)
    {
        int32_t const token = params.draftTokenIds[row * numDraft + tid];
        sTarget[tid] = params.targetProbs[(row * (numDraft + 1) + tid) * static_cast<int64_t>(vocabSize) + token];
        sDraft[tid] = draftRow(params, rows, row, tid, numDraft, vocabSize)[token];
    }
    __syncthreads();
    if (tid == 0)
    {
        curandStatePhilox4_32_10_t state;
        curand_init(static_cast<unsigned long long>(params.seed[row]), static_cast<unsigned long long>(row),
            static_cast<unsigned long long>(params.offset[row]), &state);
        int32_t pos = numDraft;
        for (int32_t i = 0; i < numDraft; ++i)
        {
            float const u = curand_uniform(&state);
            if (!(u * sDraft[i] < sTarget[i]))
            {
                pos = i;
                break;
            }
        }
        for (int32_t i = pos; i < numDraft; ++i)
        {
            curand_uniform(&state);
        }
        sPos = pos;
        sU = curand_uniform(&state);
    }
    __syncthreads();

    int32_t const pos = sPos;
    float const* target = params.targetProbs + (row * (numDraft + 1) + pos) * static_cast<int64_t>(vocabSize);
    float const* draft = pos < numDraft ? draftRow(params, rows, row, pos, numDraft, vocabSize) : nullptr;
    int32_t const numTiles = static_cast<int32_t>((vocabSize + kTileSize - 1) / kTileSize);

    // 2. This CTA's chunks: FlashInfer's per-chunk sum and the chunk's last index with mass.
    for (int32_t tile = cta; tile < numTiles; tile += G)
    {
        if (tid == 0)
        {
            sTileLast = -1;
        }
        __syncthreads();
        uint32_t const base = static_cast<uint32_t>(tile) * kTileSize + static_cast<uint32_t>(tid) * kVecSize;
        float relu[kVecSize];
        loadResidual(target, draft, base, vocabSize, relu);
        int32_t localLast = -1;
#pragma unroll
        for (int32_t j = 0; j < kVecSize; ++j)
        {
            if (relu[j] > 0.0f && base + j < vocabSize)
            {
                localLast = static_cast<int32_t>(base + j);
            }
        }
        if (localLast >= 0)
        {
            atomicMax(&sTileLast, localLast);
        }
        float const tileSum = BlockReduce(tempStorage.reduce).Sum(relu);
        __syncthreads();
        if (tid == 0)
        {
            w->tileSum[tile] = tileSum;
            w->tileLast[tile] = sTileLast;
        }
    }

    // 3. Arrival; the last CTA of the request runs FlashInfer's tail.
    if (tid == 0)
    {
        unsigned const add = static_cast<unsigned>(kBlockSize / G);
        unsigned const prev = cuda::atomic_ref<unsigned, cuda::thread_scope_device>(w->arrive).fetch_add(
            add, cuda::memory_order_acq_rel);
        sIsLast = ((prev + add) % static_cast<unsigned>(kBlockSize)) == 0;
    }
    __syncthreads();
    if (!sIsLast)
    {
        return;
    }
    if (tid < numTiles)
    {
        sTileSum[tid] = __ldcg(&w->tileSum[tid]);
        sTileLastOf[tid] = __ldcg(&w->tileLast[tid]);
    }
    __syncthreads();
    if (tid == 0)
    {
        float total = 0.0f;
        for (int32_t tile = 0; tile < numTiles; ++tile)
        {
            total += sTileSum[tile];
        }
        float const u = sU * total;
        float aggregate = 0.0f;
        int32_t chunk = -1;
        int32_t lastValid = -1;
        for (int32_t tile = 0; tile < numTiles; ++tile)
        {
            float const local = sTileSum[tile];
            if (aggregate + local > u)
            {
                chunk = tile;
            }
            if (sTileLastOf[tile] != -1)
            {
                lastValid = sTileLastOf[tile];
            }
            if (chunk >= 0)
            {
                break;
            }
            aggregate += local;
        }
        sChunk = chunk;
        sAggregate = aggregate;
        sThreshold = u;
        sLastValid = lastValid;
        sWinner = static_cast<int32_t>(vocabSize);
    }
    __syncthreads();
    if (sChunk >= 0)
    {
        uint32_t const base = static_cast<uint32_t>(sChunk) * kTileSize + static_cast<uint32_t>(tid) * kVecSize;
        float relu[kVecSize];
        loadResidual(target, draft, base, vocabSize, relu);
        float probGt[kVecSize];
        bool valid[kVecSize];
#pragma unroll
        for (int32_t j = 0; j < kVecSize; ++j)
        {
            probGt[j] = relu[j] > 0.0f ? relu[j] : 0;
            valid[j] = relu[j] > 0.0f && base + j < vocabSize;
        }
        float inclusive[kVecSize];
        deterministicInclusiveSum(probGt, inclusive, tempStorage.scan);
        float const aggregate = sAggregate;
        float const u = sThreshold;
#pragma unroll
        for (int32_t j = 0; j < kVecSize; ++j)
        {
            if ((inclusive[j] + aggregate > u) && valid[j])
            {
                atomicMin(&sWinner, static_cast<int32_t>(base + j));
                break;
            }
        }
    }
    __syncthreads();
    if (tid == 0)
    {
        int32_t token = sWinner;
        if (token == static_cast<int32_t>(vocabSize))
        {
            token = sLastValid >= 0 ? sLastValid : 0;
        }
        int32_t* out = params.outputTokenIds + row * (numDraft + 1);
        for (int32_t i = 0; i < pos; ++i)
        {
            out[i] = params.draftTokenIds[row * numDraft + i];
        }
        out[pos] = token;
        for (int32_t i = pos + 1; i <= numDraft; ++i)
        {
            out[i] = -1;
        }
        params.outputNumTokens[row] = pos + 1;
    }
}

} // namespace

cudaError_t invokeChainSpecSamplingMcFi(ChainSpeculativeSamplingParams const& params, int ctasPerRow, void* workspace,
    cudaStream_t stream, ChainSpecMcDraftRows const& rows)
{
    if (params.batchSize <= 0)
    {
        return cudaSuccess;
    }
    if (params.batchSize > 65535 || params.numDraftTokens < 0 || params.numDraftTokens > kChainSpecMcMaxDraft
        || params.vocabSize < 1 || params.vocabSize % 4 != 0 || params.vocabSize > kChainSpecMcMaxVocab
        || ctasPerRow < 1 || kBlockSize % ctasPerRow != 0 || workspace == nullptr
        || (rows.rowIds != nullptr && (rows.rowStride % 4 != 0 || rows.posStride % 4 != 0)))
    {
        return cudaErrorInvalidValue;
    }
    dim3 const grid(static_cast<unsigned>(ctasPerRow), static_cast<unsigned>(params.batchSize));
    chainSpecSamplingMcFiKernel<<<grid, kBlockSize, 0, stream>>>(params, static_cast<RowWorkspace*>(workspace), rows);
    return cudaGetLastError();
}

size_t chainSpecSamplingMcFiWorkspaceBytesPerRow()
{
    return sizeof(RowWorkspace);
}

} // namespace minimaxM31
} // namespace kernels

TRTLLM_MV2_NAMESPACE_END
