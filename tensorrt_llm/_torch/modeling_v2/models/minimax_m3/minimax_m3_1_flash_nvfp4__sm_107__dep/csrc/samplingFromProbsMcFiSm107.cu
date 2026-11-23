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
 *
 * deterministicInclusiveSum is FlashInfer's DeterministicInclusiveSum (include/flashinfer/sampling.cuh),
 * Copyright (c) 2024 by FlashInfer team, licensed under the Apache License, Version 2.0; copied with its template
 * parameters fixed (VEC_SIZE 4, BLOCK_THREADS 1024) and its shared array passed in.
 */

// SFP_MC_FI: FlashInfer's sampling from probabilities on ctasPerRow CTAs per row, bitwise. See
// samplingFromProbsMcFiSm107.h for the contract. Compiled with --use_fast_math (set on this source in
// kernels/CMakeLists.txt), as FlashInfer's JIT compiles it.

#include "minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__dep/csrc/samplingFromProbsMcFiSm107.h"
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

constexpr int32_t kBlockSize = 1024;
constexpr int32_t kVecSize = 4;
constexpr int32_t kTileSize = kBlockSize * kVecSize;
constexpr int32_t kMaxTiles = kSamplingFromProbsMcMaxVocab / kTileSize;

//! One row's cross-CTA state: each chunk's sum and its last index with p > 0 (-1 if none), and the arrival counter
//! (each launch adds kBlockSize in total, whatever its ctasPerRow, so it needs no reset).
struct RowWorkspace
{
    float tileSum[kMaxTiles];
    int32_t tileLast[kMaxTiles];
    unsigned arrive;
    int32_t pad[3];
};

struct Params
{
    float const* probs;
    int64_t rowStride;
    int64_t const* seed;
    int64_t const* offset;
    int32_t seedStride;
    int32_t offsetStride;
    int32_t* outputTokenIds;
    int32_t vocabSize;
};

//! A thread's 4 chunk values as FlashInfer loads them (vocabSize % 4 == 0, so its per-vector guard is this one).
__device__ __forceinline__ void loadProbs(float const* row, uint32_t base, uint32_t vocabSize, float (&v)[kVecSize])
{
    float4 q = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    if (base < vocabSize)
    {
        q = *reinterpret_cast<float4 const*>(row + base);
    }
    v[0] = q.x;
    v[1] = q.y;
    v[2] = q.z;
    v[3] = q.w;
}

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

__global__ void __launch_bounds__(kBlockSize, 1) samplingFromProbsMcFiKernel(Params p, RowWorkspace* ws)
{
    using BlockReduce = cub::BlockReduce<float, kBlockSize, cub::BLOCK_REDUCE_WARP_REDUCTIONS>;

    __shared__ union
    {
        typename BlockReduce::TempStorage reduce;
        float scan[kBlockSize / 32];
    } tempStorage;

    __shared__ int32_t sTileLast;
    __shared__ int32_t sIsLast;
    __shared__ float sTileSum[kMaxTiles];
    __shared__ int32_t sTileLastOf[kMaxTiles];
    __shared__ int32_t sChunk;
    __shared__ float sAggregate;
    __shared__ float sU;
    __shared__ int32_t sLastValid;
    __shared__ int32_t sWinner;

    int64_t const row = blockIdx.y;
    int32_t const G = static_cast<int32_t>(gridDim.x);
    int32_t const cta = static_cast<int32_t>(blockIdx.x);
    int32_t const tid = static_cast<int32_t>(threadIdx.x);
    uint32_t const vocabSize = static_cast<uint32_t>(p.vocabSize);
    float const* probs = p.probs + row * p.rowStride;
    RowWorkspace* w = ws + row;
    int32_t const numTiles = static_cast<int32_t>((vocabSize + kTileSize - 1) / kTileSize);

    // 1. This CTA's chunks: FlashInfer's per-chunk sum and the chunk's last index with p > 0.
    for (int32_t tile = cta; tile < numTiles; tile += G)
    {
        if (tid == 0)
        {
            sTileLast = -1;
        }
        __syncthreads();
        uint32_t const base = static_cast<uint32_t>(tile) * kTileSize + static_cast<uint32_t>(tid) * kVecSize;
        float v[kVecSize];
        loadProbs(probs, base, vocabSize, v);
        float positive[kVecSize];
        int32_t localLast = -1;
#pragma unroll
        for (int32_t j = 0; j < kVecSize; ++j)
        {
            positive[j] = v[j] > 0.0f ? v[j] : 0.0f;
            if (v[j] > 0.0f && base + j < vocabSize)
            {
                localLast = static_cast<int32_t>(base + j);
            }
        }
        if (localLast >= 0)
        {
            atomicMax(&sTileLast, localLast);
        }
        float const tileSum = BlockReduce(tempStorage.reduce).Sum(positive);
        __syncthreads();
        if (tid == 0)
        {
            w->tileSum[tile] = tileSum;
            w->tileLast[tile] = sTileLast;
        }
    }

    // 2. Arrival; the last CTA of the row runs FlashInfer's walk.
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
        curandStatePhilox4_32_10_t state;
        curand_init(static_cast<unsigned long long>(p.seed[row * p.seedStride]), static_cast<unsigned long long>(row),
            static_cast<unsigned long long>(p.offset[row * p.offsetStride]), &state);
        float const u = curand_uniform(&state);
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
        sU = u;
        sLastValid = lastValid;
        sWinner = static_cast<int32_t>(vocabSize);
    }
    __syncthreads();
    if (sChunk >= 0)
    {
        uint32_t const base = static_cast<uint32_t>(sChunk) * kTileSize + static_cast<uint32_t>(tid) * kVecSize;
        float v[kVecSize];
        loadProbs(probs, base, vocabSize, v);
        float positive[kVecSize];
        bool valid[kVecSize];
#pragma unroll
        for (int32_t j = 0; j < kVecSize; ++j)
        {
            positive[j] = v[j] > 0.0f ? v[j] : 0.0f;
            valid[j] = v[j] > 0.0f && base + j < vocabSize;
        }
        float inclusive[kVecSize];
        deterministicInclusiveSum(positive, inclusive, tempStorage.scan);
        float const aggregate = sAggregate;
        float const u = sU;
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
        p.outputTokenIds[row] = token;
    }
}

} // namespace

cudaError_t invokeSamplingFromProbsMcFi(float const* probs, int64_t rowStride, int64_t const* seed, int32_t seedStride,
    int64_t const* offset, int32_t offsetStride, int32_t* outputTokenIds, int32_t rows, int32_t vocabSize,
    int ctasPerRow, void* workspace, cudaStream_t stream)
{
    if (rows <= 0)
    {
        return cudaSuccess;
    }
    if (rows > 65535 || vocabSize < 1 || vocabSize % 4 != 0 || vocabSize > kSamplingFromProbsMcMaxVocab
        || ctasPerRow < 1 || kBlockSize % ctasPerRow != 0 || rowStride % 4 != 0 || workspace == nullptr
        || (seedStride != 0 && seedStride != 1) || (offsetStride != 0 && offsetStride != 1))
    {
        return cudaErrorInvalidValue;
    }
    Params const p{probs, rowStride, seed, offset, seedStride, offsetStride, outputTokenIds, vocabSize};
    dim3 const grid(static_cast<unsigned>(ctasPerRow), static_cast<unsigned>(rows));
    samplingFromProbsMcFiKernel<<<grid, kBlockSize, 0, stream>>>(p, static_cast<RowWorkspace*>(workspace));
    return cudaGetLastError();
}

size_t samplingFromProbsMcFiWorkspaceBytesPerRow()
{
    return sizeof(RowWorkspace);
}

} // namespace minimaxM31
} // namespace kernels

TRTLLM_MV2_NAMESPACE_END
