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

#include "tensorrt_llm/kernels/glmMoeDsaRubin/glmMnnvlHop.h"

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaTypeUtils.cuh"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/envUtils.h"
#include "tensorrt_llm/common/lamportUtils.cuh"
#include "tensorrt_llm/common/reduceKernelUtils.cuh"
#include "tensorrt_llm/kernels/glmMoeDsaRubin/glmLamport.cuh"

#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <cuda_pipeline.h>
#include <utility>

TRTLLM_NAMESPACE_BEGIN

namespace kernels::glm_moe_dsa
{
namespace
{

using common::cuda_cast;

// Eight BF16 elements, one 16-byte access.
constexpr int kEltsPerThread = static_cast<int>(sizeof(float4) / sizeof(__nv_bfloat16));
constexpr int kWarpSize = 32;
constexpr int kLog2WarpSize = 5;
constexpr int kLaneIdMask = kWarpSize - 1;
// The fast residual + RMSNorm epilogue takes CTAs of up to four whole warps in clusters of 2 to 8 CTAs.
constexpr int kMaxThreads = 128;
constexpr int kMaxWarps = kMaxThreads / kWarpSize;
constexpr int kMaxClusterSize = 8;
// Ranks summed from zero before their sum joins the total.
constexpr int kRankChunk = 8;
// Lamport data per rank up to which an all-reduce runs one-shot: 1 MiB (the MNNVL sweep's threshold), and on 16 ranks
// 1.5 MiB, which keeps an 8-row hidden-6144 BF16 step one-shot.
constexpr int64_t kOneShotBytes = 64 * 1024 * 8 * 2;
constexpr int64_t kOneShotBytes16Ranks = 3 * 512 * 1024;

union PackedBf16x8
{
    float4 packed;
    __nv_bfloat16 elements[kEltsPerThread];
};

struct HopKernelParams
{
    __nv_bfloat16* normOut;
    __nv_bfloat16* residualOut;
    __nv_bfloat16 const* shard;
    __nv_bfloat16 const* residualIn;
    __nv_bfloat16 const* gamma;
    __nv_bfloat16** bufferPtrs;
    __nv_bfloat16* mcastPtr;
    uint32_t* bufferFlags;
    int numTokens;
    int tokenDim;
    int rank;
    float epsilon;
};

struct MoeFinalizeKernelParams
{
    __nv_bfloat16 const* fc2Output;
    __nv_bfloat16 const* expertWeights;
    int32_t const* expandedIdxToPermutedIdx;
    __nv_bfloat16 const* addend;
    int hiddenDimPadded;
};

// Shared memory of the fast epilogue: each thread's residual and gamma vectors, copied before the Lamport exchange; the
// warp sums of squares of every CTA of the cluster, kMaxWarps per CTA; and the transaction barrier that completes once
// all of them have landed.
struct FastEpilogueStorage
{
    float4 residual[kMaxThreads];
    float4 gamma[kMaxThreads];
    float4 warpSums[kMaxClusterSize];
    uint64_t warpSumsBarrier;
};

// Only the kernels that call this function allocate the storage.
__device__ __forceinline__ FastEpilogueStorage& fastEpilogueStorage()
{
    __shared__ FastEpilogueStorage storage;
    return storage;
}

__device__ __forceinline__ void copyF4(float4* dstShared, __nv_bfloat16 const* src)
{
    __pipeline_memcpy_async(dstShared, reinterpret_cast<float4 const*>(src), sizeof(float4));
}

// -0.0 is the empty value of the Lamport buffers: a payload element of -0.0 is sent as +0.0.
__device__ __forceinline__ void sanitizeLamportPayload(PackedBf16x8& value)
{
#pragma unroll
    for (int i = 0; i < kEltsPerThread; i++)
    {
        if (common::isNegZero(value.elements[i]))
        {
            value.elements[i] = cuda_cast<__nv_bfloat16, float>(0.F);
        }
    }
}

__device__ __forceinline__ void accumulatePacked(float (&accum)[kEltsPerThread], PackedBf16x8 const& value)
{
#pragma unroll
    for (int i = 0; i < kEltsPerThread; i++)
    {
        accum[i] += cuda_cast<float, __nv_bfloat16>(value.elements[i]);
    }
}

__device__ __forceinline__ uint32_t loadArrivalsRelaxed(uint32_t const* arrivalCounter)
{
    uint32_t arrivals;
    asm volatile("ld.relaxed.gpu.global.u32 %0, [%1];" : "=r"(arrivals) : "l"(arrivalCounter) : "memory");
    return arrivals;
}

// Strong loads at GPU scope (L2, never a line that L1 kept from an earlier use), for reads before the grid dependency
// wait.
__device__ __forceinline__ int32_t loadRelaxedGpu(int32_t const* ptr)
{
    int32_t value;
    asm volatile("ld.relaxed.gpu.global.b32 %0, [%1];" : "=r"(value) : "l"(ptr));
    return value;
}

// The bits of a BF16 value.
__device__ __forceinline__ uint16_t loadRelaxedGpu(__nv_bfloat16 const* ptr)
{
    uint16_t bits;
    asm volatile("ld.relaxed.gpu.global.b16 %0, [%1];" : "=h"(bits) : "l"(ptr));
    return bits;
}

// The one-shot exchange in one poll loop: every iteration loads the slots of all WorldSize ranks together until none
// holds the Lamport empty value. With LocalValue the slot of localRank is not waited for: localValue, the bits this
// kernel stored there, stands in for it. With TrackArrivals a thread with a non-null arrivalCounter reads the counter
// (relaxed) at the start of every iteration, so that the last value it keeps in *arrivals is about as recent as the
// peers' data. The sum: the ranks of each chunk of kRankChunk summed in rank order from zero in FP32, the chunk sums
// added to zero in order, rounded to BF16. The WorldSize values in flight need the registers of a launch bound of
// kMaxThreads.
template <int WorldSize, bool LocalValue, bool TrackArrivals>
__device__ __forceinline__ PackedBf16x8 reduceOneshotPollAll(PackedBf16x8 const& localValue, int localRank,
    __nv_bfloat16 const* stagePtrLocal, int token, int tokenDim, int packedIdx, uint32_t const* arrivalCounter,
    uint32_t* arrivals)
{
    constexpr int kChunk = WorldSize < kRankChunk ? WorldSize : kRankChunk;
    static_assert(WorldSize % kChunk == 0, "The ranks must split into whole chunks");
    __nv_bfloat16 const* slots = &stagePtrLocal[token * tokenDim * WorldSize + packedIdx * kEltsPerThread];
    PackedBf16x8 values[WorldSize];
    while (true)
    {
        if constexpr (TrackArrivals)
        {
            if (arrivalCounter != nullptr)
            {
                *arrivals = loadArrivalsRelaxed(arrivalCounter);
            }
        }
        bool valid = true;
#pragma unroll
        for (int r = 0; r < WorldSize; r++)
        {
            auto const loaded = common::loadPackedVolatile<float4>(&slots[r * tokenDim]);
            values[r].packed = loaded.packed;
            valid &= (LocalValue && r == localRank) | !common::isLamportDirty(loaded);
        }
        if (valid)
        {
            break;
        }
    }
    float accum[kEltsPerThread];
#pragma unroll
    for (int i = 0; i < kEltsPerThread; i++)
    {
        accum[i] = 0.F;
    }
#pragma unroll
    for (int rankBase = 0; rankBase < WorldSize; rankBase += kChunk)
    {
        float chunkAccum[kEltsPerThread];
#pragma unroll
        for (int i = 0; i < kEltsPerThread; i++)
        {
            chunkAccum[i] = 0.F;
        }
#pragma unroll
        for (int rr = 0; rr < kChunk; rr++)
        {
            int const r = rankBase + rr;
            accumulatePacked(chunkAccum, LocalValue && r == localRank ? localValue : values[r]);
        }
#pragma unroll
        for (int i = 0; i < kEltsPerThread; i++)
        {
            accum[i] += chunkAccum[i];
        }
    }
    PackedBf16x8 packedAccum;
#pragma unroll
    for (int i = 0; i < kEltsPerThread; i++)
    {
        packedAccum.elements[i] = cuda_cast<__nv_bfloat16, float>(accum[i]);
    }
    return packedAccum;
}

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
// Called by one thread per CTA before every thread's relaxed arrive on the cluster barrier: the barrier expects one
// arrival (this one) and the bytes of the warp sums of every CTA of the cluster. That cluster barrier phase publishes
// the initialization; every thread waits for the phase before it sends.
__device__ __forceinline__ void initFastEpilogueBarrier()
{
    namespace cg = cooperative_groups;
    uint32_t const bytes
        = static_cast<uint32_t>(cg::this_cluster().num_blocks() * (blockDim.x >> kLog2WarpSize) * sizeof(float));
    uint32_t const barrier = static_cast<uint32_t>(__cvta_generic_to_shared(&fastEpilogueStorage().warpSumsBarrier));
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(barrier) : "memory");
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(barrier), "r"(bytes) : "memory");
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
}

// Sum of threadSum over all threads of the cluster. Each warp reduces its sums with the warpReduceSum butterfly (every
// lane ends with the same bits); every warp sends its sum to every CTA of the cluster with st.async, and each CTA waits
// on its transaction barrier until all warp sums have landed. Per CTA the warp sums are added as (w0 + w2) + (w1 + w3)
// without the absent terms, and the CTA sums in block-rank order from zero: the order of blockReduceSum<float, true>
// followed by the cluster loop of the one-shot all-reduce's default epilogue. Needs whole warps, at most kMaxWarps
// warps per CTA, 2 to kMaxClusterSize CTAs per cluster, the barrier initialized by initFastEpilogueBarrier and the
// cluster barrier phase that published it completed.
__device__ __forceinline__ float clusterSumOfSquaresFast(float threadSum)
{
    namespace cg = cooperative_groups;
    cg::cluster_group cluster = cg::this_cluster();
    auto& storage = fastEpilogueStorage();
    int const numBlocks = static_cast<int>(cluster.num_blocks());
    int const blockRank = static_cast<int>(cluster.block_rank());
    int const numWarps = static_cast<int>(blockDim.x >> kLog2WarpSize);
    int const warpId = static_cast<int>(threadIdx.x >> kLog2WarpSize);
    int const laneId = static_cast<int>(threadIdx.x & kLaneIdMask);

    float const warpSum = common::warpReduceSum(threadSum);
    uint32_t const barrier = static_cast<uint32_t>(__cvta_generic_to_shared(&storage.warpSumsBarrier));
    if (laneId < numBlocks)
    {
        // Lane l sends to CTA l.
        uint32_t const slot = static_cast<uint32_t>(
            __cvta_generic_to_shared(reinterpret_cast<float*>(storage.warpSums) + blockRank * kMaxWarps + warpId));
        uint32_t peerSlot;
        uint32_t peerBarrier;
        asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(peerSlot) : "r"(slot), "r"(laneId));
        asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(peerBarrier) : "r"(barrier), "r"(laneId));
        asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.b32 [%0], %1, [%2];" ::"r"(peerSlot),
                     "r"(__float_as_uint(warpSum)), "r"(peerBarrier)
                     : "memory");
    }
    uint32_t complete = 0;
    while (!complete)
    {
        asm volatile(
            "{\n"
            ".reg .pred done;\n"
            "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 done, [%1], 0;\n"
            "selp.u32 %0, 1, 0, done;\n"
            "}\n"
            : "=r"(complete)
            : "r"(barrier)
            : "memory");
    }

    float fullSum = 0.F;
#pragma unroll
    for (int block = 0; block < kMaxClusterSize; ++block)
    {
        if (block < numBlocks)
        {
            float4 const sums = storage.warpSums[block];
            float const even = numWarps > 2 ? sums.x + sums.z : sums.x;
            float const odd = numWarps > 3 ? sums.y + sums.w : sums.y;
            fullSum += numWarps > 1 ? even + odd : even;
        }
    }
    return fullSum;
}
#endif

// The rotation of the Lamport flags by the thread that GlmLamportFlags::waitAndUpdate elects. arrivals is the last
// value that thread read from the arrival counter with a relaxed load. Once it covers every CTA, the fence turns that
// read into an acquire of every CTA's release arrival and the flags rotate at once; otherwise the thread waits on the
// counter as waitAndUpdate does.
__device__ __forceinline__ void rotateLamportFlags(
    GlmLamportFlags<float4>& flag, uint32_t arrivals, uint4 bytesToClearPerStage)
{
    uint32_t const numCtas = gridDim.x * gridDim.y * gridDim.z;
    if (arrivals >= numCtas)
    {
        asm volatile("fence.acquire.gpu;" ::: "memory");
    }
    else
    {
        do
        {
            asm volatile("ld.acquire.gpu.global.u32 %0, [%1];"
                         : "=r"(arrivals)
                         : "l"(flag.arrivalCounter())
                         : "memory");
        } while (arrivals < numCtas);
    }
    flag.update(bytesToClearPerStage);
}

// After the poll of the rotating thread: rotate now if the last count it read covers every CTA; otherwise read the
// counter once more, while the epilogue runs, and leave the rotation to the end of the kernel.
__device__ __forceinline__ void rotateIfAllArrived(
    GlmLamportFlags<float4>& flag, bool& rotationPending, uint32_t& arrivals, uint4 bytesToClearPerStage)
{
    if (rotationPending)
    {
        if (arrivals >= gridDim.x * gridDim.y * gridDim.z)
        {
            rotateLamportFlags(flag, arrivals, bytesToClearPerStage);
            rotationPending = false;
        }
        else
        {
            arrivals = loadArrivalsRelaxed(flag.arrivalCounter());
        }
    }
}

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
// The fast residual + RMSNorm epilogue of both hops, after the copies of the residual and gamma have completed: the
// residual added in BF16, the BF16 squares summed in FP32 in element order and over the cluster, (x * rsqrt) * gamma
// rounded to BF16, the residual output stored after the exchange of the sums (the complete-tx of each st.async
// releases every earlier store of the thread), and the rotation of the flags if it is still pending.
__device__ __forceinline__ void hopEpilogue(HopKernelParams const& params, GlmLamportFlags<float4>& flag,
    PackedBf16x8 packedAccum, PackedBf16x8 const& residualIn, int threadOffset, bool rotationPending, uint32_t arrivals,
    uint4 bytesToClearPerStage)
{
#pragma unroll
    for (int i = 0; i < kEltsPerThread; i++)
    {
        packedAccum.elements[i] += residualIn.elements[i];
    }
    PackedBf16x8 gamma;
    gamma.packed = fastEpilogueStorage().gamma[threadIdx.x];
    float threadSum = 0.F;
#pragma unroll
    for (int i = 0; i < kEltsPerThread; i++)
    {
        threadSum += cuda_cast<float, __nv_bfloat16>(packedAccum.elements[i] * packedAccum.elements[i]);
    }
    // Every thread arrived on the cluster barrier before its grid dependency wait; waiting for that phase publishes
    // every CTA's transaction barrier before any thread sends.
    __cluster_barrier_wait();
    float const fullSum = clusterSumOfSquaresFast(threadSum);
    reinterpret_cast<float4*>(&params.residualOut[threadOffset])[0] = packedAccum.packed;
    float const rcpRms = rsqrtf(fullSum / params.tokenDim + params.epsilon);
#pragma unroll
    for (int i = 0; i < kEltsPerThread; i++)
    {
        packedAccum.elements[i]
            = cuda_cast<__nv_bfloat16, float>(cuda_cast<float, __nv_bfloat16>(packedAccum.elements[i]) * rcpRms
                * cuda_cast<float, __nv_bfloat16>(gamma.elements[i]));
    }
    reinterpret_cast<float4*>(&params.normOut[threadOffset])[0] = packedAccum.packed;
    if (rotationPending)
    {
        rotateLamportFlags(flag, arrivals, bytesToClearPerStage);
    }
}
#endif

// One row per cluster, eight elements per thread, the clusters' CTAs covering the row exactly. Each CTA counts its
// own arrival (GlmLamportFlags) and the thread that rotates the flags tracks the arrival counter in its poll. The
// transaction barrier is published with a relaxed cluster arrive before the grid dependency wait. The PDL dependents
// launch right after the grid dependency wait (at the start with a published shard) and must read the outputs only
// after their own wait.
// ShardPublished: the producer of the input has stored this rank's rows into the current Lamport buffer of every rank,
// sanitized, where this kernel would store them, so the kernel neither loads nor stores them and polls every rank's
// slot, its own included. It does not wait for its primary before the exchange: the producer releases it only after its
// own grid dependency wait, so every grid before the producer has completed, the previous hop (the flags, the residual)
// included. The flags and the residual are read with volatile loads, gamma (a weight) with the epilogue's copy. A CTA
// counts its arrival only once all of its slots are in: the rotation after the last arrival must not precede a producer
// CTA's read of the flags. The grid waits for its primary at its end, so that its completion still implies the
// producer's.
template <int WorldSize, bool ShardPublished>
__global__ void __launch_bounds__(kMaxThreads) mnnvlHopKernel(HopKernelParams params)
{
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    namespace cg = cooperative_groups;
    cg::cluster_group cluster = cg::this_cluster();
    int const packedIdx = static_cast<int>(cluster.thread_rank());
    int const token = static_cast<int>(blockIdx.x);
    int const threadOffset = token * params.tokenDim + packedIdx * kEltsPerThread;

    // The transaction barrier is this CTA's shared memory, which the previous grid does not touch: initialize it and
    // publish it to the cluster while the grid dependency wait is pending.
    if (threadIdx.x == 0)
    {
        initFastEpilogueBarrier();
    }
    asm volatile("barrier.cluster.arrive.relaxed;" ::: "memory");
    if constexpr (!ShardPublished)
    {
        cudaGridDependencySynchronize();
    }
    cudaTriggerProgrammaticLaunchCompletion();

    GlmLamportFlags<float4> flag(params.bufferFlags, /*numStages=*/1, /*volatileRead=*/ShardPublished);
    auto* stagePtrMcast = static_cast<__nv_bfloat16*>(flag.getCurLamportBuf(params.mcastPtr));
    auto* stagePtrLocal = static_cast<__nv_bfloat16*>(flag.getCurLamportBuf(params.bufferPtrs[params.rank]));
    auto& storage = fastEpilogueStorage();

    // ==================== Broadcast this rank's rows ====================
    PackedBf16x8 val{};
    [[maybe_unused]] PackedBf16x8 residualPublished;
    if constexpr (ShardPublished)
    {
        residualPublished.packed = common::loadPackedVolatile<float4>(&params.residualIn[threadOffset]).packed;
        copyF4(&storage.gamma[threadIdx.x], &params.gamma[packedIdx * kEltsPerThread]);
    }
    else
    {
        val.packed = *reinterpret_cast<float4 const*>(&params.shard[threadOffset]);
        // The residual and gamma do not depend on the exchange: copy them now, so that their latency overlaps the
        // broadcast and the poll instead of following the reduction.
        copyF4(&storage.residual[threadIdx.x], &params.residualIn[threadOffset]);
        copyF4(&storage.gamma[threadIdx.x], &params.gamma[packedIdx * kEltsPerThread]);
        sanitizeLamportPayload(val);
        reinterpret_cast<float4*>(
            &stagePtrMcast[token * params.tokenDim * WorldSize + params.rank * params.tokenDim])[packedIdx]
            = val.packed;
    }
    __pipeline_commit();
    if constexpr (!ShardPublished)
    {
        flag.ctaArrive();
    }
    // ==================== Empty the previous call's buffer ====================
    flag.clearDirtyLamportBuf(params.bufferPtrs[params.rank]);

    // ==================== Reduction ====================
    // The thread that waitAndUpdate would elect tracks the arrival counter and rotates the flags.
    bool rotationPending = threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0;
    uint32_t arrivals = 0;
    PackedBf16x8 packedAccum;
    if constexpr (ShardPublished)
    {
        packedAccum = reduceOneshotPollAll<WorldSize, false, false>(
            val, params.rank, stagePtrLocal, token, params.tokenDim, packedIdx, nullptr, nullptr);
        flag.ctaArrive();
    }
    else
    {
        packedAccum = reduceOneshotPollAll<WorldSize, true, true>(val, params.rank, stagePtrLocal, token,
            params.tokenDim, packedIdx, rotationPending ? flag.arrivalCounter() : nullptr, &arrivals);
    }
    uint4 const bytesToClear{
        static_cast<uint32_t>(params.numTokens * params.tokenDim * WorldSize * sizeof(__nv_bfloat16)), 0, 0, 0};
    rotateIfAllArrived(flag, rotationPending, arrivals, bytesToClear);

    // ==================== Residual + RMSNorm ====================
    // Each thread reads only the copies it issued, so their completion is the only wait needed.
    __pipeline_wait_prior(0);
    PackedBf16x8 residualIn;
    if constexpr (ShardPublished)
    {
        residualIn = residualPublished;
    }
    else
    {
        residualIn.packed = storage.residual[threadIdx.x];
    }
    hopEpilogue(params, flag, packedAccum, residualIn, threadOffset, rotationPending, arrivals, bytesToClear);
    if constexpr (ShardPublished)
    {
        // The completion of this grid implies its primary's, as when it waits at its start.
        cudaGridDependencySynchronize();
    }
#endif
}

// The routed experts' finalize step fused into the hop (BF16, top-8): each thread computes its eight elements of the
// finalize output as the TRTLLM-Gen finalize does (the token's eight FC2 rows weighted in k order with an FMA, -1 slots
// skipped, rounded to BF16; the addend added in FP32 and the sum rounded once more) and publishes them as its rows.
// Everything else is mnnvlHopKernel without a published shard. Eight threads per CTA read the token's FC2 row indices
// and routing weights before the grid dependency wait: the MoE's routing wrote them, and it has completed when this
// grid launches because FC1 and FC2 trigger their dependents only after their own waits. The launch bound lets the
// eight FC2 rows of a thread be in flight in registers.
template <int WorldSize>
__global__ void __launch_bounds__(kMaxThreads)
    mnnvlMoeFinalizeHopKernel(HopKernelParams params, MoeFinalizeKernelParams moe)
{
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    namespace cg = cooperative_groups;
    cg::cluster_group cluster = cg::this_cluster();
    int const packedIdx = static_cast<int>(cluster.thread_rank());
    int const token = static_cast<int>(blockIdx.x);
    int const threadOffset = token * params.tokenDim + packedIdx * kEltsPerThread;
    __shared__ int32_t sPermutedIdx[kMnnvlHopMoeTopK];
    __shared__ uint16_t sExpertWeightBits[kMnnvlHopMoeTopK];

    // The transaction barrier is this CTA's shared memory: initialize it and publish it to the cluster while the grid
    // dependency wait is pending.
    if (threadIdx.x == 0)
    {
        initFastEpilogueBarrier();
    }
    asm volatile("barrier.cluster.arrive.relaxed;" ::: "memory");
    bool const isLoader = threadIdx.x < kMnnvlHopMoeTopK;
    int32_t permutedIdxPart = -1;
    uint16_t expertWeightBitsPart = 0;
    if (isLoader)
    {
        int const expandedIdx = token * kMnnvlHopMoeTopK + static_cast<int>(threadIdx.x);
        permutedIdxPart = loadRelaxedGpu(&moe.expandedIdxToPermutedIdx[expandedIdx]);
        expertWeightBitsPart = loadRelaxedGpu(&moe.expertWeights[expandedIdx]);
    }
    cudaGridDependencySynchronize();
    cudaTriggerProgrammaticLaunchCompletion();

    GlmLamportFlags<float4> flag(params.bufferFlags, /*numStages=*/1, /*volatileRead=*/false);
    auto* stagePtrMcast = static_cast<__nv_bfloat16*>(flag.getCurLamportBuf(params.mcastPtr));
    auto* stagePtrLocal = static_cast<__nv_bfloat16*>(flag.getCurLamportBuf(params.bufferPtrs[params.rank]));
    auto& storage = fastEpilogueStorage();
    copyF4(&storage.residual[threadIdx.x], &params.residualIn[threadOffset]);
    copyF4(&storage.gamma[threadIdx.x], &params.gamma[packedIdx * kEltsPerThread]);
    __pipeline_commit();
    if (isLoader)
    {
        sPermutedIdx[threadIdx.x] = permutedIdxPart;
        sExpertWeightBits[threadIdx.x] = expertWeightBitsPart;
    }
    __syncthreads();

    // ==================== Finalize: this thread's elements of the routed sum plus the addend ====================
    int permutedIdx[kMnnvlHopMoeTopK];
#pragma unroll
    for (int k = 0; k < kMnnvlHopMoeTopK; k++)
    {
        permutedIdx[k] = sPermutedIdx[k];
    }
    PackedBf16x8 addend;
    addend.packed = *reinterpret_cast<float4 const*>(&moe.addend[threadOffset]);
    PackedBf16x8 rows[kMnnvlHopMoeTopK];
#pragma unroll
    for (int k = 0; k < kMnnvlHopMoeTopK; k++)
    {
        int64_t const rowOffset
            = static_cast<int64_t>(permutedIdx[k]) * moe.hiddenDimPadded + packedIdx * kEltsPerThread;
        rows[k].packed = permutedIdx[k] == -1 ? make_float4(0.F, 0.F, 0.F, 0.F)
                                              : *reinterpret_cast<float4 const*>(&moe.fc2Output[rowOffset]);
    }
    float routed[kEltsPerThread];
#pragma unroll
    for (int i = 0; i < kEltsPerThread; i++)
    {
        routed[i] = 0.F;
    }
#pragma unroll
    for (int k = 0; k < kMnnvlHopMoeTopK; k++)
    {
        if (permutedIdx[k] == -1)
        {
            continue;
        }
        float const weight = cuda_cast<float, __nv_bfloat16>(__ushort_as_bfloat16(sExpertWeightBits[k]));
#pragma unroll
        for (int i = 0; i < kEltsPerThread; i++)
        {
            routed[i] = __fmaf_rn(weight, cuda_cast<float, __nv_bfloat16>(rows[k].elements[i]), routed[i]);
        }
    }
    PackedBf16x8 val;
#pragma unroll
    for (int i = 0; i < kEltsPerThread; i++)
    {
        val.elements[i] = cuda_cast<__nv_bfloat16, float>(cuda_cast<float, __nv_bfloat16>(addend.elements[i])
            + cuda_cast<float, __nv_bfloat16>(cuda_cast<__nv_bfloat16, float>(routed[i])));
    }

    // ==================== Broadcast this rank's rows ====================
    sanitizeLamportPayload(val);
    reinterpret_cast<float4*>(
        &stagePtrMcast[token * params.tokenDim * WorldSize + params.rank * params.tokenDim])[packedIdx]
        = val.packed;
    flag.ctaArrive();
    flag.clearDirtyLamportBuf(params.bufferPtrs[params.rank]);

    // ==================== Reduction ====================
    bool rotationPending = threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0;
    uint32_t arrivals = 0;
    PackedBf16x8 const packedAccum = reduceOneshotPollAll<WorldSize, true, true>(val, params.rank, stagePtrLocal, token,
        params.tokenDim, packedIdx, rotationPending ? flag.arrivalCounter() : nullptr, &arrivals);
    uint4 const bytesToClear{
        static_cast<uint32_t>(params.numTokens * params.tokenDim * WorldSize * sizeof(__nv_bfloat16)), 0, 0, 0};
    rotateIfAllArrived(flag, rotationPending, arrivals, bytesToClear);

    // ==================== Residual + RMSNorm ====================
    __pipeline_wait_prior(0);
    PackedBf16x8 residualIn;
    residualIn.packed = storage.residual[threadIdx.x];
    hopEpilogue(params, flag, packedAccum, residualIn, threadOffset, rotationPending, arrivals, bytesToClear);
#endif
}

// (CTA size, cluster size) of the one-shot grid for rows of tokenDim BF16 elements, eight per thread: the grid of the
// one-shot MNNVL all-reduce (its grid configuration with clusters and one access per scale group). Starts from the
// widest cluster that splits the row evenly, trades cluster width for CTAs of at least 128 threads, narrows the
// clusters while numTokens of them exceed the SMs, and widens them again down to 64-thread CTAs while the SMs allow.
std::pair<int, int> hopGridConfig(int numTokens, int tokenDim, int smCount)
{
    int const threadsNeeded = (tokenDim + kEltsPerThread - 1) / kEltsPerThread;
    int clusterSize = kMaxClusterSize;
    while (threadsNeeded % clusterSize != 0 && clusterSize > 1)
    {
        clusterSize /= 2;
    }
    int const maxDivisibleClusterSize = clusterSize;
    int blockSize = (threadsNeeded + clusterSize - 1) / clusterSize;
    while (blockSize < 128 && clusterSize >= 2)
    {
        blockSize *= 2;
        clusterSize /= 2;
    }
    while (numTokens * clusterSize > smCount && clusterSize > 1 && blockSize <= 512)
    {
        blockSize *= 2;
        clusterSize /= 2;
    }
    while (clusterSize < maxDivisibleClusterSize)
    {
        int const candidateClusterSize = clusterSize * 2;
        int const candidateBlockSize = (threadsNeeded + candidateClusterSize - 1) / candidateClusterSize;
        if (candidateBlockSize < 64 || numTokens * candidateClusterSize > smCount)
        {
            break;
        }
        clusterSize = candidateClusterSize;
        blockSize = candidateBlockSize;
    }
    return {blockSize, clusterSize};
}

// The launch attributes of both hops: PDL when enabled, the cluster, and the largest shared-memory carveout, so that a
// dependent launched early whose CTAs need most of the shared memory (the fused-A GEMM) can place them on the SMs that
// run this kernel's CTAs instead of starting them only when these exit.
struct HopLaunch
{
    cudaLaunchAttribute attrs[3];
    cudaLaunchConfig_t config;
};

void initHopLaunch(HopLaunch& launch, MnnvlHopParams const& params)
{
    TLLM_CHECK_WITH_INFO(common::getSMVersion() >= 90, "[GLM MNNVL hop] requires SM 90 or newer.");
    TLLM_CHECK_WITH_INFO(mnnvlHopSupported(params.numTokens, params.tokenDim, params.nRanks),
        "[GLM MNNVL hop] unsupported shape: %d tokens, hidden %d, %d ranks.", params.numTokens, params.tokenDim,
        params.nRanks);
    auto const [blockSize, clusterSize]
        = hopGridConfig(params.numTokens, params.tokenDim, common::getMultiProcessorCount());
    launch.attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    launch.attrs[0].val.programmaticStreamSerializationAllowed = common::getEnvEnablePDL() ? 1 : 0;
    launch.attrs[1].id = cudaLaunchAttributeClusterDimension;
    launch.attrs[1].val.clusterDim.x = 1;
    launch.attrs[1].val.clusterDim.y = clusterSize;
    launch.attrs[1].val.clusterDim.z = 1;
    launch.attrs[2].id = cudaLaunchAttributePreferredSharedMemoryCarveout;
    launch.attrs[2].val.sharedMemCarveout = cudaSharedmemCarveoutMaxShared;
    launch.config = {};
    launch.config.gridDim = dim3(params.numTokens, clusterSize, 1);
    launch.config.blockDim = dim3(blockSize, 1, 1);
    launch.config.dynamicSmemBytes = 0;
    launch.config.stream = params.stream;
    launch.config.attrs = launch.attrs;
    launch.config.numAttrs = 3;
}

HopKernelParams makeHopKernelParams(MnnvlHopParams const& params)
{
    HopKernelParams kernelParams{};
    kernelParams.normOut = static_cast<__nv_bfloat16*>(params.normOut);
    kernelParams.residualOut = static_cast<__nv_bfloat16*>(params.residualOut);
    kernelParams.shard = static_cast<__nv_bfloat16 const*>(params.input);
    kernelParams.residualIn = static_cast<__nv_bfloat16 const*>(params.residualIn);
    kernelParams.gamma = static_cast<__nv_bfloat16 const*>(params.gamma);
    kernelParams.bufferPtrs = reinterpret_cast<__nv_bfloat16**>(params.bufferPtrsDev);
    kernelParams.mcastPtr = static_cast<__nv_bfloat16*>(params.multicastPtr);
    kernelParams.bufferFlags = params.bufferFlags;
    kernelParams.numTokens = params.numTokens;
    kernelParams.tokenDim = params.tokenDim;
    kernelParams.rank = params.rank;
    kernelParams.epsilon = params.epsilon;
    return kernelParams;
}

template <int WorldSize>
void launchHop(cudaLaunchConfig_t const& config, HopKernelParams const& kernelParams, bool shardPublished)
{
    if (shardPublished)
    {
        TLLM_CUDA_CHECK(cudaLaunchKernelEx(&config, &mnnvlHopKernel<WorldSize, true>, kernelParams));
    }
    else
    {
        TLLM_CUDA_CHECK(cudaLaunchKernelEx(&config, &mnnvlHopKernel<WorldSize, false>, kernelParams));
    }
}

template <int WorldSize>
void launchMoeFinalizeHop(
    cudaLaunchConfig_t const& config, HopKernelParams const& kernelParams, MoeFinalizeKernelParams const& moeParams)
{
    TLLM_CUDA_CHECK(cudaLaunchKernelEx(&config, &mnnvlMoeFinalizeHopKernel<WorldSize>, kernelParams, moeParams));
}

} // namespace

int64_t mnnvlHopBufferBytes(int numTokens, int tokenDim, int nRanks)
{
    return static_cast<int64_t>(numTokens) * tokenDim * nRanks * static_cast<int64_t>(sizeof(__nv_bfloat16));
}

bool mnnvlHopSupported(int numTokens, int tokenDim, int nRanks)
{
    if ((nRanks != 2 && nRanks != 4 && nRanks != 8 && nRanks != 16) || numTokens < 1 || tokenDim <= 0
        || tokenDim % kEltsPerThread != 0)
    {
        return false;
    }
    int64_t const bytes = mnnvlHopBufferBytes(numTokens, tokenDim, nRanks);
    if (bytes > kOneShotBytes && !(nRanks == 16 && bytes <= kOneShotBytes16Ranks))
    {
        return false;
    }
    auto const [blockSize, clusterSize] = hopGridConfig(numTokens, tokenDim, common::getMultiProcessorCount());
    return blockSize % kWarpSize == 0 && blockSize <= kMaxThreads && clusterSize > 1 && clusterSize <= kMaxClusterSize
        && blockSize * clusterSize * kEltsPerThread == tokenDim;
}

void mnnvlHop(MnnvlHopParams const& params)
{
    HopLaunch launch;
    initHopLaunch(launch, params);
    HopKernelParams const kernelParams = makeHopKernelParams(params);
    switch (params.nRanks)
    {
    case 2: launchHop<2>(launch.config, kernelParams, params.shardPublished); break;
    case 4: launchHop<4>(launch.config, kernelParams, params.shardPublished); break;
    case 8: launchHop<8>(launch.config, kernelParams, params.shardPublished); break;
    case 16: launchHop<16>(launch.config, kernelParams, params.shardPublished); break;
    default: TLLM_CHECK_WITH_INFO(false, "[GLM MNNVL hop] unsupported world size %d.", params.nRanks);
    }
}

void mnnvlMoeFinalizeHop(MnnvlHopParams const& params, MnnvlMoeFinalizeParams const& moe)
{
    HopLaunch launch;
    initHopLaunch(launch, params);
    HopKernelParams const kernelParams = makeHopKernelParams(params);
    MoeFinalizeKernelParams moeParams{};
    moeParams.fc2Output = static_cast<__nv_bfloat16 const*>(moe.fc2Output);
    moeParams.expertWeights = static_cast<__nv_bfloat16 const*>(moe.expertWeights);
    moeParams.expandedIdxToPermutedIdx = moe.expandedIdxToPermutedIdx;
    moeParams.addend = static_cast<__nv_bfloat16 const*>(moe.addend);
    moeParams.hiddenDimPadded = moe.hiddenDimPadded;
    switch (params.nRanks)
    {
    case 2: launchMoeFinalizeHop<2>(launch.config, kernelParams, moeParams); break;
    case 4: launchMoeFinalizeHop<4>(launch.config, kernelParams, moeParams); break;
    case 8: launchMoeFinalizeHop<8>(launch.config, kernelParams, moeParams); break;
    case 16: launchMoeFinalizeHop<16>(launch.config, kernelParams, moeParams); break;
    default: TLLM_CHECK_WITH_INFO(false, "[GLM MNNVL MoE-finalize hop] unsupported world size %d.", params.nRanks);
    }
}

} // namespace kernels::glm_moe_dsa

TRTLLM_NAMESPACE_END
