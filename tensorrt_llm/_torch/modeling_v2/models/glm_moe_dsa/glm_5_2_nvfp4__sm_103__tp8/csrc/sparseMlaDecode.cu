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

#include "glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/csrc/sparseMlaDecode.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/envUtils.h"
#include "tensorrt_llm/modeling_v2/common/namespace.h"

#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <cmath>

TRTLLM_MV2_NAMESPACE_BEGIN

namespace sparse_mla_decode
{
namespace
{

// Work split: one CTA handles one query row and 128 consecutive positions of its index list.
// The 16 CTAs of a row form one thread-block cluster. Each CTA pushes its unnormalized partial
// output and softmax statistics into the shared memory of the CTA that finalizes those output
// dims with asynchronous stores that complete on that CTA's mbarrier.
int constexpr kKeysPerCta = 128;
int constexpr kNumSplits = kTopK / kKeysPerCta;
int constexpr kNumWarps = 8;
int constexpr kThreads = kNumWarps * 32;
int constexpr kKeysPerWarp = kKeysPerCta / kNumWarps;
int constexpr kDimsPerWarp = kHeadDimV / kNumWarps;
int constexpr kDimsPerCtaOut = kHeadDimV / kNumSplits;
[[maybe_unused]] int constexpr kQkSteps = kHeadDimQk / 32;
[[maybe_unused]] int constexpr kPvSteps = kKeysPerCta / 32;
[[maybe_unused]] int constexpr kPvTilesPerWarp = kDimsPerWarp / 16;
[[maybe_unused]] int constexpr kQChunks = kNumHeads * kHeadDimQk / 16;
// Probabilities are scaled to the e4m3 range before rounding, as in the native kernel.
[[maybe_unused]] float constexpr kProbScale = 448.F;

// Shared-memory pitches, padded so that ldmatrix and fragment loads are bank-conflict free: K rows
// shift by 20 banks (eight ldmatrix rows hit disjoint 16-byte bank groups); the 64-bit query
// fragment loads of rows g = 0-3 cover banks 8g..8g+7 (pitch = 8 banks mod 32); the 32-bit
// probability fragment loads of rows g = 0-7 cover banks 4g..4g+3.
int constexpr kKRowBytes = kHeadDimQk + 16;
int constexpr kQRowHalves = kHeadDimQk + 16;
int constexpr kPRowHalves = kKeysPerCta + 8;

int constexpr kKTileOffset = 0;
int constexpr kQTileOffset = kKTileOffset + kKeysPerCta * kKRowBytes;
int constexpr kPTileOffset = kQTileOffset + kNumHeads * kQRowHalves * 2;
// [source split][head][dim within this CTA's 32 output dims] FP32 partial outputs.
int constexpr kRecvOffset = kPTileOffset + kNumHeads * kPRowHalves * 2;
// [max of heads 0-7, sum of heads 0-7][source split] FP32.
int constexpr kRecvStatsOffset = kRecvOffset + kNumSplits * kNumHeads * kDimsPerCtaOut * 4;
int constexpr kRedMaxOffset = kRecvStatsOffset + kNumSplits * 2 * kNumHeads * 4;
int constexpr kRedSumOffset = kRedMaxOffset + kNumWarps * kNumHeads * 4;
// mbarriers: one per warp for its row gather, one for the pushed partials of this CTA.
int constexpr kBarrierOffset = kRedSumOffset + kNumWarps * kNumHeads * 4;
int constexpr kSmemBytes = kBarrierOffset + (kNumWarps + 1) * 8;
// Bytes each CTA receives: 16 sources x (8 heads x 32 dims + 16 statistics) FP32 values.
[[maybe_unused]] uint32_t constexpr kRecvBytes = kNumSplits * (kNumHeads * kDimsPerCtaOut + 2 * kNumHeads) * 4;

static_assert(kTopK % kKeysPerCta == 0, "the index list must split evenly");
static_assert(kNumSplits <= 16, "the merge cluster holds at most 16 CTAs");
static_assert(kKeysPerWarp == 16, "one m16 tile of keys per warp");
static_assert(kNumHeads == 8, "the eight heads form the n8 side of the MMAs");
static_assert(kDimsPerCtaOut == 32 && kDimsPerWarp == 2 * kDimsPerCtaOut, "each warp feeds two merge CTAs");
static_assert(kDimsPerCtaOut * kNumHeads == kThreads, "one merged output element per thread");
static_assert(kKRowBytes % 16 == 0 && kQRowHalves % 8 == 0 && kPRowHalves % 8 == 0, "16-byte aligned shared rows");
static_assert(
    kKRowBytes / 4 % 32 == 20 && kQRowHalves / 2 % 32 == 8 && kPRowHalves / 2 % 32 == 4, "bank-conflict-free pitches");
static_assert(kBarrierOffset % 8 == 0, "mbarrier alignment");
static_assert(kQChunks <= 2 * kThreads, "two query chunks per thread at most");
static_assert(2 * kNumHeads * kNumSplits == kThreads, "one statistics store per thread");
static_assert(
    kNumWarps == kNumHeads && (kNumSplits & (kNumSplits - 1)) == 0, "merge: one warp per head, one lane per split");

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)

__device__ __forceinline__ uint32_t smemAddr(void const* ptr)
{
    return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

__device__ __forceinline__ void mbarrierInit(uint32_t bar, uint32_t count)
{
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(bar), "r"(count) : "memory");
}

__device__ __forceinline__ void fenceBarrierInit()
{
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
}

__device__ __forceinline__ void mbarrierArriveExpectTx(uint32_t bar, uint32_t bytes)
{
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(bar), "r"(bytes) : "memory");
}

__device__ __forceinline__ void mbarrierArrive(uint32_t bar)
{
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" ::"r"(bar) : "memory");
}

__device__ __forceinline__ void mbarrierWait(uint32_t bar, uint32_t parity)
{
    uint32_t done = 0;
    while (!done)
    {
        asm volatile(
            "{\n"
            ".reg .pred p;\n"
            "mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n"
            "selp.u32 %0, 1, 0, p;\n"
            "}\n"
            : "=r"(done)
            : "r"(bar), "r"(parity)
            : "memory");
    }
}

// The same wait with cluster-scope acquire: the arrivals and transaction bytes it waits for come
// from asynchronous stores issued by other CTAs of the cluster.
__device__ __forceinline__ void mbarrierWaitCluster(uint32_t bar, uint32_t parity)
{
    uint32_t done = 0;
    while (!done)
    {
        asm volatile(
            "{\n"
            ".reg .pred p;\n"
            "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%1], %2;\n"
            "selp.u32 %0, 1, 0, p;\n"
            "}\n"
            : "=r"(done)
            : "r"(bar), "r"(parity)
            : "memory");
    }
}

__device__ __forceinline__ void bulkCopyG2S(uint32_t dst, void const* src, uint32_t bytes, uint32_t bar)
{
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n" ::"r"(dst),
                 "l"(src), "r"(bytes), "r"(bar)
                 : "memory");
}

__device__ __forceinline__ void gridDependencyWait()
{
    asm volatile("griddepcontrol.wait;\n" ::: "memory");
}

__device__ __forceinline__ void gridDependencyLaunchDependents()
{
    asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory");
}

__device__ __forceinline__ void clusterArriveRelaxed()
{
    asm volatile("barrier.cluster.arrive.relaxed.aligned;\n" ::: "memory");
}

__device__ __forceinline__ void clusterWait()
{
    asm volatile("barrier.cluster.wait.acquire.aligned;\n" ::: "memory");
}

__device__ __forceinline__ uint32_t mapToRank(uint32_t localAddr, uint32_t rank)
{
    uint32_t remote;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;\n" : "=r"(remote) : "r"(localAddr), "r"(rank));
    return remote;
}

// Asynchronous stores into a peer CTA's shared memory that complete-tx on the peer's mbarrier.
__device__ __forceinline__ void storeAsync(uint32_t remoteAddr, float value, uint32_t remoteBar)
{
    asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.f32 [%0], %1, [%2];\n" ::"r"(remoteAddr),
                 "f"(value), "r"(remoteBar)
                 : "memory");
}

__device__ __forceinline__ void storeAsync2(uint32_t remoteAddr, float a, float b, uint32_t remoteBar)
{
    asm volatile(
        "st.async.shared::cluster.mbarrier::complete_tx::bytes.v2.f32 [%0], {%1, %2}, [%3];\n" ::"r"(remoteAddr),
        "f"(a), "f"(b), "r"(remoteBar)
        : "memory");
}

__device__ __forceinline__ void ldmatrixX4(uint32_t (&r)[4], uint32_t addr)
{
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(addr));
}

__device__ __forceinline__ void ldmatrixX4Trans(uint32_t (&r)[4], uint32_t addr)
{
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(addr));
}

// D += A * B with A 16x16 f16 (row), B 16x8 f16 (col), FP32 accumulation.
__device__ __forceinline__ void mmaF16(
    float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1)
{
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
        "{%0, %1, %2, %3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ float fastExp2(float x)
{
    float y;
    asm volatile("ex2.approx.ftz.f32 %0, %1;\n" : "=f"(y) : "f"(x));
    return y;
}

// Two e4m3 values (low byte first) to two FP16 values; exact.
__device__ __forceinline__ uint32_t e4m3x2ToHalf2(uint32_t pair)
{
    __half2_raw const h = __nv_cvt_fp8x2_to_halfraw2(static_cast<__nv_fp8x2_storage_t>(pair & 0xffffU), __NV_E4M3);
    return static_cast<uint32_t>(h.x) | (static_cast<uint32_t>(h.y) << 16);
}

// e4m3(448 * p) as an exact FP16 value.
__device__ __forceinline__ __half roundProbability(float p)
{
    __nv_fp8_storage_t const q = __nv_cvt_float_to_fp8(p * kProbScale, __NV_SATFINITE, __NV_E4M3);
    return __half(__nv_cvt_fp8_to_halfraw(q, __NV_E4M3));
}

#endif // defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)

#ifdef GLM_MOE_DSA_SPARSE_MLA_DECODE_TIMING
// Development timeline: thread 0 of each CTA records the SM clock at phase boundaries and the
// global timer at entry and exit.
__device__ unsigned long long gTimeline[kTimelineMaxCtas][kTimelineSlots];
#endif

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
#ifdef GLM_MOE_DSA_SPARSE_MLA_DECODE_TIMING
__device__ __forceinline__ void markTimeline(int slot)
{
    if (threadIdx.x == 0 && blockIdx.x < kTimelineMaxCtas)
    {
        gTimeline[blockIdx.x][slot] = clock64();
    }
}

__device__ __forceinline__ void markGlobalTime(int slot)
{
    if (threadIdx.x == 0 && blockIdx.x < kTimelineMaxCtas)
    {
        unsigned long long t;
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
        gTimeline[blockIdx.x][slot] = t;
    }
}
#else
__device__ __forceinline__ void markTimeline(int) {}

__device__ __forceinline__ void markGlobalTime(int) {}
#endif
#endif // defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)

__global__ void __launch_bounds__(kThreads) sparseMlaDecodeKernel(SparseMlaDecodeParams const params)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    extern __shared__ __align__(128) uint8_t smem[];

    int const tid = threadIdx.x;
    int const warp = tid >> 5;
    int const lane = tid & 31;
    int const groupId = lane >> 2;
    int const tig = lane & 3;
    int const row = blockIdx.x / kNumSplits;
    int const split = blockIdx.x % kNumSplits;

    uint8_t* kTile = smem + kKTileOffset;
    __half* qTile = reinterpret_cast<__half*>(smem + kQTileOffset);
    __half* pTile = reinterpret_cast<__half*>(smem + kPTileOffset);
    float* recv = reinterpret_cast<float*>(smem + kRecvOffset);
    float* recvStats = reinterpret_cast<float*>(smem + kRecvStatsOffset);
    float* redMax = reinterpret_cast<float*>(smem + kRedMaxOffset);
    float* redSum = reinterpret_cast<float*>(smem + kRedSumOffset);
    uint32_t const gatherBar = smemAddr(smem + kBarrierOffset + warp * 8);
    uint32_t const recvBar = smemAddr(smem + kBarrierOffset + kNumWarps * 8);

    markGlobalTime(0);
    markTimeline(1);
    if (lane == 0)
    {
        mbarrierInit(gatherBar, kKeysPerWarp);
    }
    if (tid == 0)
    {
        mbarrierInit(recvBar, 1);
    }
    // The initializations must be visible to the bulk-copy engine and to the peers' async stores.
    fenceBarrierInit();
    __syncthreads();
    clusterArriveRelaxed();

    // Everything below reads tensors written by preceding kernels in the stream. The kernel can be
    // resident before those kernels finish, so these tensors are read with coherent loads, not
    // through the non-coherent read-only path.
    gridDependencyWait();
    markTimeline(2);

    // The query does not depend on the index list: start its loads first.
    uint8_t const* qRow = static_cast<uint8_t const*>(params.q) + static_cast<int64_t>(row) * kNumHeads * kHeadDimQk;
    uint4 qRaw[2];
#pragma unroll
    for (int i = 0; i < 2; ++i)
    {
        int const chunk = tid + i * kThreads;
        if (chunk < kQChunks)
        {
            qRaw[i] = reinterpret_cast<uint4 const*>(qRow)[chunk];
        }
    }

    int32_t poolRow = -1;
    if (lane < kKeysPerWarp)
    {
        int const seq = row / params.seqLenQ;
        int const rowInSeq = row - seq * params.seqLenQ;
        int const key = split * kKeysPerCta + warp * kKeysPerWarp + lane;
        int32_t const listed = params.indices[static_cast<int64_t>(row) * params.indicesRowStride + key];
        int const usable = params.seqLensKv[seq] - (params.seqLenQ - 1 - rowInSeq);
        poolRow = key < usable ? listed : -1;
    }
    uint32_t const validBits = __ballot_sync(0xffffffffU, poolRow >= 0);
    markTimeline(3);

    // Gather this warp's 16 rows with one 576-byte bulk copy each; unusable rows are zeroed so that
    // zero probabilities never meet non-finite bytes.
    if (lane < kKeysPerWarp)
    {
        uint8_t* dst = kTile + (warp * kKeysPerWarp + lane) * kKRowBytes;
        if (poolRow >= 0)
        {
            mbarrierArriveExpectTx(gatherBar, kHeadDimQk);
            bulkCopyG2S(smemAddr(dst),
                static_cast<uint8_t const*>(params.kvPool) + static_cast<int64_t>(poolRow) * kHeadDimQk, kHeadDimQk,
                gatherBar);
        }
        else
        {
#pragma unroll 4
            for (int c = 0; c < kHeadDimQk / 16; ++c)
            {
                *reinterpret_cast<uint4*>(dst + c * 16) = make_uint4(0U, 0U, 0U, 0U);
            }
            mbarrierArrive(gatherBar);
        }
    }

    // FP16 copy of the query, shared by all warps.
#pragma unroll
    for (int i = 0; i < 2; ++i)
    {
        int const chunk = tid + i * kThreads;
        if (chunk < kQChunks)
        {
            int const head = chunk / (kHeadDimQk / 16);
            int const col = (chunk - head * (kHeadDimQk / 16)) * 16;
            uint4 lo;
            uint4 hi;
            lo.x = e4m3x2ToHalf2(qRaw[i].x);
            lo.y = e4m3x2ToHalf2(qRaw[i].x >> 16);
            lo.z = e4m3x2ToHalf2(qRaw[i].y);
            lo.w = e4m3x2ToHalf2(qRaw[i].y >> 16);
            hi.x = e4m3x2ToHalf2(qRaw[i].z);
            hi.y = e4m3x2ToHalf2(qRaw[i].z >> 16);
            hi.z = e4m3x2ToHalf2(qRaw[i].w);
            hi.w = e4m3x2ToHalf2(qRaw[i].w >> 16);
            uint4* dst = reinterpret_cast<uint4*>(qTile + head * kQRowHalves + col);
            dst[0] = lo;
            dst[1] = hi;
        }
    }
    float const scaleLog2 = params.bmm1Scale[1];
    float const outScale = params.bmm2Scale[0];

    gridDependencyLaunchDependents();
    __syncthreads();

    // Every peer has initialized its barriers once phase 0 of the cluster barrier completes; waiting
    // here overlaps that with the gather. Pushes into peers start after the softmax.
    clusterWait();
    mbarrierWait(gatherBar, 0);
    __syncwarp();
    markTimeline(4);

    // S^T = K * Q^T for this warp's 16 keys: keys on the M side, heads on the N side, FP16 MMAs on
    // converted e4m3 keys. Within each 16-value k chunk the MMA k-slot order is permuted so that
    // the two halves of an ldmatrix register map straight onto the A fragment: slots 2t, 2t+1 hold
    // k = 4t, 4t+1 and slots 2t+8, 2t+9 hold k = 4t+2, 4t+3. The FP16 query is read in that order.
    float sAcc[4];
    {
        float acc[2][4] = {};
        uint32_t const aAddr = smemAddr(kTile + (warp * kKeysPerWarp + (lane & 15)) * kKRowBytes + (lane >> 4) * 16);
        __half const* qFrag = qTile + groupId * kQRowHalves + 4 * tig;
#pragma unroll
        for (int kk = 0; kk < kQkSteps; ++kk)
        {
            uint32_t r[4];
            ldmatrixX4(r, aAddr + kk * 32);
            uint2 const b0 = *reinterpret_cast<uint2 const*>(qFrag + kk * 32);
            uint2 const b1 = *reinterpret_cast<uint2 const*>(qFrag + kk * 32 + 16);
            mmaF16(acc[0], e4m3x2ToHalf2(r[0]), e4m3x2ToHalf2(r[1]), e4m3x2ToHalf2(r[0] >> 16),
                e4m3x2ToHalf2(r[1] >> 16), b0.x, b0.y);
            mmaF16(acc[1], e4m3x2ToHalf2(r[2]), e4m3x2ToHalf2(r[3]), e4m3x2ToHalf2(r[2] >> 16),
                e4m3x2ToHalf2(r[3] >> 16), b1.x, b1.y);
        }
#pragma unroll
        for (int i = 0; i < 4; ++i)
        {
            sAcc[i] = acc[0][i] + acc[1][i];
        }
    }
    markTimeline(5);

    // Fragment element (key, head): 0 -> (groupId, 2*tig), 1 -> (groupId, 2*tig+1), 2/3 -> key + 8.
    bool const valid0 = (validBits >> groupId) & 1U;
    bool const valid1 = (validBits >> (groupId + 8)) & 1U;
    float s[4];
    s[0] = valid0 ? sAcc[0] * scaleLog2 : -INFINITY;
    s[1] = valid0 ? sAcc[1] * scaleLog2 : -INFINITY;
    s[2] = valid1 ? sAcc[2] * scaleLog2 : -INFINITY;
    s[3] = valid1 ? sAcc[3] * scaleLog2 : -INFINITY;

    float maxA = fmaxf(s[0], s[2]);
    float maxB = fmaxf(s[1], s[3]);
#pragma unroll
    for (int offset = 4; offset < 32; offset <<= 1)
    {
        maxA = fmaxf(maxA, __shfl_xor_sync(0xffffffffU, maxA, offset));
        maxB = fmaxf(maxB, __shfl_xor_sync(0xffffffffU, maxB, offset));
    }
    if (lane < 4)
    {
        redMax[warp * kNumHeads + 2 * tig] = maxA;
        redMax[warp * kNumHeads + 2 * tig + 1] = maxB;
    }
    __syncthreads();

    float headMaxA = -INFINITY;
    float headMaxB = -INFINITY;
#pragma unroll
    for (int w = 0; w < kNumWarps; ++w)
    {
        headMaxA = fmaxf(headMaxA, redMax[w * kNumHeads + 2 * tig]);
        headMaxB = fmaxf(headMaxB, redMax[w * kNumHeads + 2 * tig + 1]);
    }
    float p[4];
    p[0] = s[0] == -INFINITY ? 0.F : fastExp2(s[0] - headMaxA);
    p[1] = s[1] == -INFINITY ? 0.F : fastExp2(s[1] - headMaxB);
    p[2] = s[2] == -INFINITY ? 0.F : fastExp2(s[2] - headMaxA);
    p[3] = s[3] == -INFINITY ? 0.F : fastExp2(s[3] - headMaxB);

    float sumA = p[0] + p[2];
    float sumB = p[1] + p[3];
#pragma unroll
    for (int offset = 4; offset < 32; offset <<= 1)
    {
        sumA += __shfl_xor_sync(0xffffffffU, sumA, offset);
        sumB += __shfl_xor_sync(0xffffffffU, sumB, offset);
    }
    if (lane < 4)
    {
        redSum[warp * kNumHeads + 2 * tig] = sumA;
        redSum[warp * kNumHeads + 2 * tig + 1] = sumB;
    }
    {
        int const key = warp * kKeysPerWarp + groupId;
        pTile[(2 * tig) * kPRowHalves + key] = roundProbability(p[0]);
        pTile[(2 * tig + 1) * kPRowHalves + key] = roundProbability(p[1]);
        pTile[(2 * tig) * kPRowHalves + key + 8] = roundProbability(p[2]);
        pTile[(2 * tig + 1) * kPRowHalves + key + 8] = roundProbability(p[3]);
    }
    __syncthreads();

    // This CTA's softmax statistics to every CTA of the cluster: thread t sends value t / 16
    // (maxima of heads 0-7, then sums) to rank t % 16.
    {
        int const index = tid / kNumSplits;
        int const dest = tid % kNumSplits;
        int const head = index & (kNumHeads - 1);
        bool const isMax = index < kNumHeads;
        float value = isMax ? -INFINITY : 0.F;
#pragma unroll
        for (int w = 0; w < kNumWarps; ++w)
        {
            value = isMax ? fmaxf(value, redMax[w * kNumHeads + head]) : value + redSum[w * kNumHeads + head];
        }
        storeAsync(mapToRank(smemAddr(recvStats + index * kNumSplits + split), dest), value, mapToRank(recvBar, dest));
    }
    markTimeline(6);

    // O^T = V^T * P^T for this warp's 64 output dims, one 16-dim tile at a time; each finished tile
    // is pushed to the CTA that finalizes its dims while the next tile is computed. Transposing the
    // e4m3 V rows with b16 ldmatrix yields keys (2t, 2t+1) x dims (2g, 2g+1) per register; a byte
    // permute splits it into the two dims, so M row g of a tile is dim 2g and M row g+8 is dim 2g+1.
    {
        uint32_t pFrag[kPvSteps][2][2];
        __half const* pRow = pTile + groupId * kPRowHalves + 2 * tig;
#pragma unroll
        for (int ks = 0; ks < kPvSteps; ++ks)
        {
#pragma unroll
            for (int half = 0; half < 2; ++half)
            {
                pFrag[ks][half][0] = *reinterpret_cast<uint32_t const*>(pRow + ks * 32 + half * 16);
                pFrag[ks][half][1] = *reinterpret_cast<uint32_t const*>(pRow + ks * 32 + half * 16 + 8);
            }
        }
        uint32_t const vAddr = smemAddr(kTile + lane * kKRowBytes + warp * kDimsPerWarp);
        uint32_t const recvBase = smemAddr(recv);
        float const invProbScale = 1.F / kProbScale;
#pragma unroll
        for (int mt = 0; mt < kPvTilesPerWarp; ++mt)
        {
            // Independent accumulators per MMA keep the tensor pipe busy within one tile.
            float part[kPvSteps][2][4] = {};
#pragma unroll
            for (int ks = 0; ks < kPvSteps; ++ks)
            {
                uint32_t r[4];
                ldmatrixX4Trans(r, vAddr + ks * 32 * kKRowBytes + mt * 16);
                uint32_t dimEven[4];
                uint32_t dimOdd[4];
#pragma unroll
                for (int j = 0; j < 4; ++j)
                {
                    uint32_t const pairs = __byte_perm(r[j], 0, 0x3120);
                    dimEven[j] = e4m3x2ToHalf2(pairs);
                    dimOdd[j] = e4m3x2ToHalf2(pairs >> 16);
                }
                mmaF16(part[ks][0], dimEven[0], dimOdd[0], dimEven[1], dimOdd[1], pFrag[ks][0][0], pFrag[ks][0][1]);
                mmaF16(part[ks][1], dimEven[2], dimOdd[2], dimEven[3], dimOdd[3], pFrag[ks][1][0], pFrag[ks][1][1]);
            }
            float o[4];
#pragma unroll
            for (int i = 0; i < 4; ++i)
            {
                o[i] = ((part[0][0][i] + part[0][1][i]) + (part[1][0][i] + part[1][1][i]))
                    + ((part[2][0][i] + part[2][1][i]) + (part[3][0][i] + part[3][1][i]));
            }
            int const dim = warp * kDimsPerWarp + mt * 16 + 2 * groupId;
            uint32_t const dest = dim / kDimsPerCtaOut;
            int const localDim = dim - static_cast<int>(dest) * kDimsPerCtaOut;
            uint32_t const addrA
                = recvBase + ((split * kNumHeads + 2 * tig) * kDimsPerCtaOut + localDim) * sizeof(float);
            uint32_t const addrB = addrA + kDimsPerCtaOut * sizeof(float);
            uint32_t const remoteBar = mapToRank(recvBar, dest);
            storeAsync2(mapToRank(addrA, dest), o[0] * invProbScale, o[2] * invProbScale, remoteBar);
            storeAsync2(mapToRank(addrB, dest), o[1] * invProbScale, o[3] * invProbScale, remoteBar);
        }
    }
    markTimeline(7);
    markTimeline(8);
    if (tid == 0)
    {
        mbarrierArriveExpectTx(recvBar, kRecvBytes);
    }
    markTimeline(9);
    mbarrierWaitCluster(recvBar, 0);
    markTimeline(10);

    // Finalize 32 of the 512 output dims of all eight heads from the pushed partials: warp h
    // handles head h, lane i < 16 the statistics of source split i, every lane one output dim.
    {
        int const head = warp;
        float const peerMax = lane < kNumSplits ? recvStats[head * kNumSplits + lane] : -INFINITY;
        float const peerSum = lane < kNumSplits ? recvStats[(kNumHeads + head) * kNumSplits + lane] : 0.F;
        float globalMax = peerMax;
#pragma unroll
        for (int offset = kNumSplits / 2; offset > 0; offset >>= 1)
        {
            globalMax = fmaxf(globalMax, __shfl_xor_sync(0xffffffffU, globalMax, offset));
        }
        float const weight = peerMax == -INFINITY ? 0.F : fastExp2(peerMax - globalMax);
        float sum = weight * peerSum;
#pragma unroll
        for (int offset = kNumSplits / 2; offset > 0; offset >>= 1)
        {
            sum += __shfl_xor_sync(0xffffffffU, sum, offset);
        }
        sum = __shfl_sync(0xffffffffU, sum, 0);
        float acc = 0.F;
#pragma unroll
        for (int i = 0; i < kNumSplits; ++i)
        {
            acc += __shfl_sync(0xffffffffU, weight, i) * recv[(i * kNumHeads + head) * kDimsPerCtaOut + lane];
        }
        float const result = sum > 0.F ? acc * (outScale / sum) : 0.F;
        params.out[static_cast<int64_t>(row) * kNumHeads * kHeadDimV + head * kHeadDimV + split * kDimsPerCtaOut + lane]
            = __float2bfloat16_rn(result);
    }
    markTimeline(11);
    markGlobalTime(12);
#else
    __trap();
#endif
}

void prepareKernelAttributes()
{
    TLLM_CUDA_CHECK(
        cudaFuncSetAttribute(sparseMlaDecodeKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes));
    TLLM_CUDA_CHECK(cudaFuncSetAttribute(sparseMlaDecodeKernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
}

cudaLaunchAttribute clusterAttribute()
{
    cudaLaunchAttribute attr;
    attr.id = cudaLaunchAttributeClusterDimension;
    attr.val.clusterDim.x = kNumSplits;
    attr.val.clusterDim.y = 1;
    attr.val.clusterDim.z = 1;
    return attr;
}

} // namespace

#ifdef GLM_MOE_DSA_SPARSE_MLA_DECODE_TIMING
void copySparseMlaDecodeTimeline(unsigned long long* host)
{
    TLLM_CUDA_CHECK(cudaMemcpyFromSymbol(host, gTimeline, sizeof(gTimeline)));
}
#endif

size_t getSparseMlaDecodeSmemBytes()
{
    return static_cast<size_t>(kSmemBytes);
}

bool isSparseMlaDecodeDeviceSupported(int device)
{
    int major = 0;
    int maxSmem = 0;
    TLLM_CUDA_CHECK(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device));
    TLLM_CUDA_CHECK(cudaDeviceGetAttribute(&maxSmem, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
    int constexpr kBlackwellDatacenterMajor = 10;
    if (major != kBlackwellDatacenterMajor || maxSmem < kSmemBytes)
    {
        return false;
    }
    int currentDevice = 0;
    TLLM_CUDA_CHECK(cudaGetDevice(&currentDevice));
    TLLM_CUDA_CHECK(cudaSetDevice(device));
    prepareKernelAttributes();
    cudaLaunchConfig_t config{};
    config.gridDim = dim3(kNumSplits);
    config.blockDim = dim3(kThreads);
    config.dynamicSmemBytes = kSmemBytes;
    cudaLaunchAttribute attr = clusterAttribute();
    config.attrs = &attr;
    config.numAttrs = 1;
    int numClusters = 0;
    cudaError_t const status = cudaOccupancyMaxActiveClusters(&numClusters, sparseMlaDecodeKernel, &config);
    TLLM_CUDA_CHECK(cudaSetDevice(currentDevice));
    if (status != cudaSuccess)
    {
        // Clear the error recorded by the failed query.
        cudaGetLastError();
        return false;
    }
    return numClusters > 0;
}

void invokeSparseMlaDecode(SparseMlaDecodeParams const& params, cudaStream_t stream)
{
    if (params.numRows <= 0)
    {
        return;
    }
    prepareKernelAttributes();
    cudaLaunchConfig_t config{};
    config.gridDim = dim3(params.numRows * kNumSplits);
    config.blockDim = dim3(kThreads);
    config.dynamicSmemBytes = kSmemBytes;
    config.stream = stream;
    cudaLaunchAttribute attrs[2];
    attrs[0] = clusterAttribute();
    attrs[1].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[1].val.programmaticStreamSerializationAllowed = tensorrt_llm::common::getEnvEnablePDL() ? 1 : 0;
    config.attrs = attrs;
    config.numAttrs = 2;
    TLLM_CUDA_CHECK(cudaLaunchKernelEx(&config, sparseMlaDecodeKernel, params));
}

} // namespace sparse_mla_decode

TRTLLM_MV2_NAMESPACE_END
