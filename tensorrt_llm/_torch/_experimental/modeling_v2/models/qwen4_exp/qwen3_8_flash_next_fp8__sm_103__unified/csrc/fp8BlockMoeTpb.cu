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

// A block (one expert's 128 intermediate rows) closes its FC1 -> activation -> FC2 chain inside one cluster. Every CTA
// routes every row (top-10, ties to the lower id, fp32 softmax over the ten rounded to bf16). Cluster k takes blocks
// k, k + K, ...: a producer warp streams weights into a shared-memory ring, five consumer warps run the MMAs, and an
// epilogue warp sends each FC1 item to every cluster CTA by DSMEM bulk copy, completing that CTA's block barrier.
// Past one grid-wide counter every CTA sums its share of the K cluster partials in a fixed order and stores bf16.

#include "qwen4_exp/qwen3_8_flash_next_fp8__sm_103__unified/csrc/fp8BlockMoeTpb.h"

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <algorithm>
#include <mutex>
#include <stdexcept>
#include <string>
#include <type_traits>

TRTLLM_MV2_NAMESPACE_BEGIN

namespace kernels::fp8_block_moe_tpb
{
namespace
{

constexpr int kH = 2560;
constexpr int kE = 512;
constexpr int kK = 10;
constexpr int kRows = 8;
constexpr int kKb = kH / 128;          // 20 k-blocks of FC1
constexpr int kBlkRows = 256;          // FC1 rows of a block: 128 up, 128 gate
constexpr int kItemFc2Rows = 320;      // FC2 rows of one ring item
constexpr int kMt = kItemFc2Rows / 16; // 20 m-tiles of 16 FC2 rows an item
constexpr int kConsumerWarps = 5;
constexpr int kEpiWarp = 5;
constexpr int kProducerWarp = 6;
constexpr int kThreads = 7 * 32;
constexpr int kConsumerThreads = kConsumerWarps * 32;
constexpr int kMtPerWarp = kMt / kConsumerWarps; // 4
constexpr int kItemRowBytes = 16 * kH;           // 40960: 16 FC1 rows, or 320 FC2 rows x 128
constexpr int kStageBytes = kItemRowBytes + 128; // + the item's 20 fp32 scales
constexpr int kMaxStages = 8;
constexpr int kMaxSlots = kRows * kK;            // 80
constexpr int kMaxBlk = 2 * kMaxSlots + kShItems; // up to two blocks a routed expert, plus the shared items
constexpr int kActRow = 128 + 16;
constexpr int kMaxClusters = 40;
constexpr float kE4m3Max = 448.f;
constexpr float kAmaxFloor = 1.0e-10f;

// Workspace: the u64 grid counter at 0, which grows by kStride a call and never needs a reset, then the fp32 partials
// [kMaxClusters][8][2560] at 4096.
constexpr unsigned long long kStride = 1024;
constexpr size_t kWsPart = 4096;
constexpr size_t kWsBytes = kWsPart + size_t(kMaxClusters) * kRows * kH * 4;
constexpr int kDefaultStages = 4; // as fast as five, and leaves ~40 KB of shared memory an SM to side-stream kernels

struct alignas(16) Smem
{
    float4 red[2][kConsumerWarps][32];
    uint64_t fullBar[kMaxStages];
    uint64_t emptyBar[kMaxStages];
    uint64_t xbar[2];
    float topkW[kRows][kK];
    int topkId[kRows][kK];
    int expertOf[kMaxSlots];
    uint32_t bitmap[kE / 32];
    int wordBase[kE / 32];
    unsigned long long barTarget;
    int numDistinct;
    int numBlk;
    int8_t slotOf[kMaxSlots][kRows];
    uint16_t blkJ[kMaxBlk];
    uint8_t blkD[kMaxBlk];
    alignas(16) uint8_t actC[2][kRows][kActRow]; // 16-byte rows: the FC2 B fragments are 128-bit loads
    alignas(16) float stage[2][4][kRows][16];    // an FC1 item's 16 rows x T outputs, the DSMEM copy's source
    float actF[2][kRows];
};

struct alignas(16) SmemSh : Smem
{
    alignas(16) __nv_bfloat16 actS[2][kRows][32]; // the slice item's 32 activations of each row (FC2 B operand)
    float shScale[kRows];                         // sigmoid(bf16 gate logit) of each row
};

template <bool kSh>
using SmemT = std::conditional_t<kSh, SmemSh, Smem>;

template <bool kSh>
constexpr int kStaticBytes = (int(sizeof(SmemT<kSh>)) + 127) / 128 * 128;

constexpr int kShMark = 1024;       // block-list entries >= this are shared-slice items (kShMark + item)
constexpr int kShCols = 32;         // intermediate columns of a shared item
constexpr int kShRowBytes = kH * 2; // one bf16 weight / activation row

#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 900)
#define FP8_BLOCK_MOE_TPB_DEVICE 1
#endif

#ifdef FP8_BLOCK_MOE_TPB_DEVICE
__device__ __forceinline__ uint32_t smemAddr(void const* p)
{
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void mbarInit(uint64_t* bar, uint32_t count)
{
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(smemAddr(bar)), "r"(count) : "memory");
}

__device__ __forceinline__ void mbarArriveExpectTx(uint64_t* bar, uint32_t bytes)
{
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(smemAddr(bar)), "r"(bytes) : "memory");
}

__device__ __forceinline__ void mbarArrive(uint64_t* bar)
{
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(smemAddr(bar)) : "memory");
}

__device__ __forceinline__ void mbarWait(uint64_t* bar, uint32_t parity)
{
    asm volatile(
        "{\n"
        ".reg .pred P1;\n"
        "LAB_WAIT:\n"
        "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n"
        "@P1 bra DONE;\n"
        "bra LAB_WAIT;\n"
        "DONE:\n"
        "}\n" ::"r"(smemAddr(bar)),
        "r"(parity)
        : "memory");
}

// Wait with cluster-scope acquire: the peers' DSMEM stores released by their arrivals are visible after it.
__device__ __forceinline__ void mbarWaitCluster(uint64_t* bar, uint32_t parity)
{
    asm volatile(
        "{\n"
        ".reg .pred P1;\n"
        "LAB_WAIT:\n"
        "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 P1, [%0], %1;\n"
        "@P1 bra DONE;\n"
        "bra LAB_WAIT;\n"
        "DONE:\n"
        "}\n" ::"r"(smemAddr(bar)),
        "r"(parity)
        : "memory");
}

__device__ __forceinline__ uint32_t clusterCtaRank()
{
    uint32_t r;
    asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
    return r;
}

__device__ __forceinline__ uint32_t mapaShared(uint32_t addr, uint32_t rank)
{
    uint32_t r;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(r) : "r"(addr), "r"(rank));
    return r;
}

__device__ __forceinline__ void clusterSync()
{
    asm volatile("barrier.cluster.arrive.release.aligned;\nbarrier.cluster.wait.acquire.aligned;" ::: "memory");
}

__device__ __forceinline__ void fenceAcqRelGpu()
{
    asm volatile("fence.acq_rel.gpu;" ::: "memory");
}

__device__ __forceinline__ void fenceBarrierInit()
{
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
}

__device__ __forceinline__ void bulkG2S(void* dst, void const* src, uint32_t bytes, uint64_t* bar)
{
    asm volatile(
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(smemAddr(dst)),
        "l"(src), "r"(bytes), "r"(smemAddr(bar))
        : "memory");
}

// DSMEM bulk copy into another cluster CTA, completing `bytes` on its mbarrier (`dst`, `bar`: shared::cluster).
__device__ __forceinline__ void bulkS2Cluster(uint32_t dst, void const* src, uint32_t bytes, uint32_t bar)
{
    asm volatile(
        "cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(dst),
        "r"(smemAddr(src)), "r"(bytes), "r"(bar)
        : "memory");
}

__device__ __forceinline__ void bulkCommit()
{
    asm volatile("cp.async.bulk.commit_group;" ::: "memory");
}

template <int N>
__device__ __forceinline__ void bulkWaitRead()
{
    asm volatile("cp.async.bulk.wait_group.read %0;" ::"n"(N) : "memory");
}

__device__ __forceinline__ void fenceProxyAsyncShared()
{
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}

__device__ __forceinline__ void namedBarSync(int id, int count)
{
    asm volatile("bar.sync %0, %1;" ::"r"(id), "r"(count) : "memory");
}

__device__ __forceinline__ void namedBarArrive(int id, int count)
{
    asm volatile("bar.arrive %0, %1;" ::"r"(id), "r"(count) : "memory");
}

__device__ __forceinline__ void gridDepWait()
{
    asm volatile("griddepcontrol.wait;" ::: "memory");
}

__device__ __forceinline__ void gridDepLaunchDependents()
{
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
}

__device__ __forceinline__ unsigned long long ldAcquireGpu64(unsigned long long const* p)
{
    unsigned long long v;
    asm volatile("ld.acquire.gpu.global.u64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
    return v;
}

__device__ __forceinline__ void redAddRelease64(unsigned long long* p, unsigned long long v)
{
    asm volatile("red.release.gpu.global.add.u64 [%0], %1;" ::"l"(p), "l"(v) : "memory");
}

__device__ __forceinline__ uint4 lds128(uint8_t const* p)
{
    return *reinterpret_cast<uint4 const*>(p);
}

__device__ __forceinline__ uint4 ldcg128(void const* p)
{
    return __ldcg(reinterpret_cast<uint4 const*>(p));
}

__device__ __forceinline__ void mmaE4m3(
    float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1)
{
    asm("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
        "{%0,%1,%2,%3};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void mmaBf16(
    float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1)
{
    asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
        "{%0,%1,%2,%3};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// A 32-wide k step from the 8 bf16 a lane holds per operand, as two k16 MMAs that permute k alike in A and B.
__device__ __forceinline__ void mma32Bf16(float (&acc)[4], uint4 const& ag, uint4 const& ag8, uint4 const& b)
{
    mmaBf16(acc, ag.x, ag8.x, ag.y, ag8.y, b.x, b.y);
    mmaBf16(acc, ag.z, ag8.z, ag.w, ag8.w, b.z, b.w);
}

__device__ __forceinline__ float bf16Round(float x)
{
    return __bfloat162float(__float2bfloat16_rn(x));
}

__device__ __forceinline__ float warpMaxAbs(float v)
{
    uint32_t bits = __float_as_uint(fabsf(v));
    return __uint_as_float(__reduce_max_sync(0xffffffffu, bits));
}

__device__ __forceinline__ float e4m3RoundTrip(float x)
{
    __nv_fp8_storage_t q = __nv_cvt_float_to_fp8(x, __NV_SATFINITE, __NV_E4M3);
    return __half2float(__half(__nv_cvt_fp8_to_halfraw(q, __NV_E4M3)));
}

__device__ __forceinline__ uint32_t e4m3Code(float x)
{
    return static_cast<uint32_t>(__nv_cvt_float_to_fp8(x, __NV_SATFINITE, __NV_E4M3));
}

__device__ __forceinline__ float siluFast(float x)
{
    return __fdividef(x, 1.0f + __expf(-x));
}

// Row t's top-10 logits (ties to the lower id) and their softmax rounded to bf16; lane l holds experts 16l..16l+15.
__device__ __forceinline__ void routeRow(TpbParams const& p, int t, int lane, Smem& s)
{
    uint8_t const* row = static_cast<uint8_t const*>(p.logits) + size_t(t) * kE * 2 + lane * 32;
    uint4 r0 = ldcg128(row);
    uint4 r1 = ldcg128(row + 16);
    uint32_t words[8] = {r0.x, r0.y, r0.z, r0.w, r1.x, r1.y, r1.z, r1.w};
    uint32_t key[16];
#pragma unroll
    for (int i = 0; i < 16; ++i)
    {
        uint32_t b = (words[i >> 1] >> ((i & 1) * 16)) & 0xffffu;
        uint32_t ord = (b & 0x8000u) ? (~b & 0xffffu) : (b | 0x8000u);
        key[i] = (ord << 16) | uint32_t(kE - 1 - (lane * 16 + i));
    }
    uint32_t mine = 0;
#pragma unroll 1
    for (int k = 0; k < kK; ++k)
    {
        uint32_t m8[8];
#pragma unroll
        for (int i = 0; i < 8; ++i)
        {
            m8[i] = max(key[2 * i], key[2 * i + 1]);
        }
        uint32_t const m4a = max(m8[0], m8[1]), m4b = max(m8[2], m8[3]);
        uint32_t const m4c = max(m8[4], m8[5]), m4d = max(m8[6], m8[7]);
        uint32_t const best = max(max(m4a, m4b), max(m4c, m4d));
        uint32_t const top = __reduce_max_sync(0xffffffffu, best);
#pragma unroll
        for (int i = 0; i < 16; ++i)
        {
            key[i] = key[i] == top ? 0u : key[i];
        }
        mine = lane == k ? top : mine;
    }
    float v = -INFINITY;
    if (lane < kK)
    {
        uint32_t ord = mine >> 16;
        uint32_t b = (ord & 0x8000u) ? (ord & 0x7fffu) : (~ord & 0xffffu);
        v = __uint_as_float(b << 16);
        s.topkId[t][lane] = kE - 1 - int(mine & 0xffffu);
    }
    float const mx = __shfl_sync(0xffffffffu, v, 0);
    float e = lane < kK ? expf(v - mx) : 0.f;
    float sum = e;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1)
    {
        sum += __shfl_xor_sync(0xffffffffu, sum, o);
    }
    if (lane < kK)
    {
        s.topkW[t][lane] = __bfloat162float(__float2bfloat16_rn(__fdiv_rn(e, sum)));
    }
}

struct Ring
{
    int stage;
    uint32_t phase;
    int numStages;

    __device__ __forceinline__ void advance()
    {
        if (++stage == numStages)
        {
            stage = 0;
            phase ^= 1u;
        }
    }
};

#endif // FP8_BLOCK_MOE_TPB_DEVICE

// CS: CTAs a cluster (4 or 8); kSh: also fold in this rank's shared-expert slice.
template <int CS, bool kSh>
__global__ void __launch_bounds__(kThreads, 1) fp8BlockMoeTpbKernel(
    TpbParams const p, int const numStages, int const xbufOffset, int const ringOffset, int const shXOffset)
{
#ifdef FP8_BLOCK_MOE_TPB_DEVICE
    static_assert(!kSh || CS == 4, "the shared-expert fold takes 4-CTA clusters");
    constexpr int kCs = CS;
    constexpr int kUpRows = 128 / kCs; // a CTA's up rows (and gate rows) of a block
    constexpr int kNh = kUpRows / 16;  // items of each half, and FC2 items
    constexpr int kFc2Rows = kH / kCs; // a CTA's FC2 rows of a block
    extern __shared__ __align__(128) uint8_t smemRaw[];
    SmemT<kSh>& s = *reinterpret_cast<SmemT<kSh>*>(smemRaw);
    float* xbuf = reinterpret_cast<float*>(smemRaw + xbufOffset); // [2][T][256] raw FC1 outputs of a block
    uint8_t* ring = smemRaw + ringOffset;

    int const warp = threadIdx.x >> 5;
    int const lane = threadIdx.x & 31;
    int const T = p.numRows;
    int const G = gridDim.x;
    int const cta = blockIdx.x;
    int const nSlots = T * kK;
    int const crank = int(clusterCtaRank());
    int const cl = cta / kCs;
    int const NC = G / kCs;

    unsigned long long* counter = static_cast<unsigned long long*>(p.workspace);
    float* part = reinterpret_cast<float*>(static_cast<uint8_t*>(p.workspace) + kWsPart);

    if (threadIdx.x == 0)
    {
        for (int i = 0; i < numStages; ++i)
        {
            mbarInit(&s.fullBar[i], 1);
            mbarInit(&s.emptyBar[i], kConsumerWarps);
        }
        mbarInit(&s.xbar[0], 1); // the local arrive.expect_tx; the peers' copies complete the bytes
        mbarInit(&s.xbar[1], 1);
        fenceBarrierInit();
        // Read before arriving: the counter is a multiple of kStride plus fewer than all of this call's arrivals.
        s.barTarget = (ldAcquireGpu64(counter) / kStride + 1) * kStride;
    }
    for (int i = threadIdx.x; i < kMaxSlots * kRows; i += kThreads)
    {
        (&s.slotOf[0][0])[i] = -1;
    }
    if (threadIdx.x < kE / 32)
    {
        s.bitmap[threadIdx.x] = 0;
    }
    // The peers' barriers are initialized before any CTA of the cluster arrives on them.
    clusterSync();

    gridDepWait();

    for (int t = warp; t < T; t += kThreads / 32)
    {
        routeRow(p, t, lane, s);
    }
    __syncthreads();
    int myExpert = -1;
    if (threadIdx.x < nSlots)
    {
        int const t = threadIdx.x / kK;
        int const k = threadIdx.x % kK;
        myExpert = s.topkId[t][k];
        atomicOr(&s.bitmap[myExpert >> 5], 1u << (myExpert & 31));
    }
    __syncthreads();
    if (warp == 0)
    {
        int const c = lane < kE / 32 ? __popc(s.bitmap[lane]) : 0;
        int inc = c;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1)
        {
            int const v = __shfl_up_sync(0xffffffffu, inc, o);
            inc += lane >= o ? v : 0;
        }
        if (lane < kE / 32)
        {
            s.wordBase[lane] = inc - c;
        }
        if (lane == 31)
        {
            s.numDistinct = inc;
        }
    }
    __syncthreads();
    if (threadIdx.x < nSlots)
    {
        int const t = threadIdx.x / kK;
        int const wd = myExpert >> 5;
        int const bit = myExpert & 31;
        int const d = s.wordBase[wd] + __popc(s.bitmap[wd] & ((1u << bit) - 1u));
        s.expertOf[d] = myExpert;
        s.slotOf[d][t] = int8_t(threadIdx.x);
    }
    __syncthreads();
    int const D = s.numDistinct;
    // This rank's blocks: entry d is block (rank - e) % 4 of expert e = expertOf[d], then block-4 entries, then shared.
    if (warp == 0)
    {
        int base = 0;
        for (int d0 = 0; d0 < D; d0 += 32)
        {
            int const d = d0 + lane;
            int extra = 0;
            if (d < D)
            {
                int const e = s.expertOf[d];
                int const b0 = (p.rank - e) & (kTpbRanks - 1);
                s.blkJ[d] = uint16_t(e);
                s.blkD[d] = uint8_t(d);
                extra = b0 == 0;
            }
            uint32_t const bal = __ballot_sync(0xffffffffu, extra);
            if (extra)
            {
                int const pos = D + base + __popc(bal & ((1u << lane) - 1u));
                s.blkJ[pos] = uint16_t(kE + (s.expertOf[d] >> 2));
                s.blkD[pos] = uint8_t(d);
            }
            base += __popc(bal);
        }
        if constexpr (kSh)
        {
            if (lane < p.shBlocks)
            {
                s.blkJ[D + base + lane] = uint16_t(kShMark + lane);
                s.blkD[D + base + lane] = 0;
            }
        }
        if (lane == 0)
        {
            s.numBlk = D + base + (kSh ? p.shBlocks : 0);
        }
    }
    __syncthreads();
    // Every CTA is resident by now: the next kernel may start its prologue.
    gridDepLaunchDependents();
    int const nB = s.numBlk;
    if (nB == 0)
    {
        __nv_bfloat162* o = static_cast<__nv_bfloat162*>(p.out);
        for (int q = cta * kThreads + threadIdx.x; q < T * kH / 2; q += G * kThreads)
        {
            o[q] = __floats2bfloat162_rn(0.f, 0.f);
        }
        return;
    }
    int const NCa = min(NC, nB);
    int const nMy = cl < NCa ? (nB - cl + NC - 1) / NC : 0;

    if (warp == kProducerWarp)
    {
        // ---- producer: per block the up / gate items (16 FC1 rows + 20 scales), then FC2 items (320 rows + 20 scales)
        Ring rs{0, 0u, numStages};
        for (int n = 0; n < nMy; ++n)
        {
            int const j = s.blkJ[cl + n * NC];
            if (kSh && j >= kShMark)
            {
                // Shared item: 2 x 8 bf16 FC1 rows ([gate 4 ; up 4]), then this CTA's 640 down rows x 32 columns.
                int const sb = j - kShMark;
                for (int q = 0; q < 3; ++q)
                {
                    mbarWait(&s.emptyBar[rs.stage], rs.phase ^ 1u);
                    if (lane == 0)
                    {
                        uint8_t* dst = ring + rs.stage * kStageBytes;
                        mbarArriveExpectTx(&s.fullBar[rs.stage], kItemRowBytes);
                        uint8_t const* src = q < 2
                            ? static_cast<uint8_t const*>(p.shGu) + (size_t(sb) * 64 + crank * 16 + q * 8) * kShRowBytes
                            : static_cast<uint8_t const*>(p.shDn) + (size_t(sb) * kH + crank * kFc2Rows) * kShCols * 2;
                        bulkG2S(dst, src, kItemRowBytes, &s.fullBar[rs.stage]);
                    }
                    __syncwarp();
                    rs.advance();
                }
                continue;
            }
            for (int q = 0; q < 3 * kNh; ++q)
            {
                int const kind = q / kNh; // 0 up, 1 gate, 2 FC2
                int const u = q % kNh;
                mbarWait(&s.emptyBar[rs.stage], rs.phase ^ 1u);
                if (lane == 0)
                {
                    uint8_t* dst = ring + rs.stage * kStageBytes;
                    mbarArriveExpectTx(&s.fullBar[rs.stage], kItemRowBytes + kKb * 4);
                    if (kind < 2)
                    {
                        size_t const row = size_t(j) * kBlkRows + kind * 128 + crank * kUpRows + u * 16;
                        bulkG2S(dst, p.w1 + row * kH, kItemRowBytes, &s.fullBar[rs.stage]);
                        bulkG2S(
                            dst + kItemRowBytes, p.s1 + (size_t(j) * 2 + kind) * kKb, kKb * 4, &s.fullBar[rs.stage]);
                    }
                    else
                    {
                        size_t const row = size_t(j) * kH + crank * kFc2Rows + u * kItemFc2Rows;
                        bulkG2S(dst, p.w2 + row * 128, kItemRowBytes, &s.fullBar[rs.stage]);
                        bulkG2S(dst + kItemRowBytes, p.s2 + size_t(j) * kKb, kKb * 4, &s.fullBar[rs.stage]);
                    }
                }
                __syncwarp();
                rs.advance();
            }
        }
        __syncwarp();
    }
    else if (warp == kEpiWarp)
    {
        // ---- epilogue: per FC1 item, sum the consumer warps' partials in a fixed order and send them to every CTA
        int const g = lane >> 2;
        int const tig = lane & 3;
        int const t0 = 2 * tig, t1 = 2 * tig + 1;
        auto fc1Items = [&](int n) { return kSh && s.blkJ[cl + n * NC] >= kShMark ? 2 : 2 * kNh; };
        int nFc1 = 0;
        for (int n = 0; n < nMy; ++n)
        {
            nFc1 += fc1Items(n);
        }
        uint32_t const xbase = smemAddr(xbuf);
        uint32_t const xbarBase = smemAddr(&s.xbar[0]);
        int n = 0, q = 0, nq = nMy > 0 ? fc1Items(0) : 0, prevItems = 2 * kNh;
        for (int jf = 0; jf < nFc1; ++jf)
        {
            if (q == nq)
            {
                prevItems = nq;
                ++n;
                q = 0;
                nq = fc1Items(n);
            }
            int const buf = jf & 1;
            namedBarSync(1 + buf, kConsumerThreads + 32);
            float4 v = s.red[buf][0][lane];
#pragma unroll
            for (int w = 1; w < kConsumerWarps; ++w)
            {
                float4 const o = s.red[buf][w][lane];
                v.x += o.x;
                v.y += o.y;
                v.z += o.z;
                v.w += o.w;
            }
            if (jf + 2 < nFc1)
            {
                namedBarArrive(3 + buf, kConsumerThreads + 32);
            }
            // Stage 16 rows x T, then copy into each CTA's exchange buffer [CS][2 kNh][T][16] at (this CTA, item).
            if (q == 0 && n >= 2)
            {
                // Block n - 2's copies (this parity's slots) must have read their source; block n - 1's may pend.
                if (prevItems == 2)
                {
                    bulkWaitRead<2>();
                }
                else
                {
                    bulkWaitRead<2 * kNh>();
                }
            }
            float* stg = &s.stage[n & 1][q][0][0];
            if (t0 < T)
            {
                stg[t0 * 16 + g] = v.x;
                stg[t0 * 16 + g + 8] = v.z;
            }
            if (t1 < T)
            {
                stg[t1 * 16 + g] = v.y;
                stg[t1 * 16 + g + 8] = v.w;
            }
            fenceProxyAsyncShared();
            __syncwarp();
            if (lane < kCs)
            {
                uint32_t const off = uint32_t((n & 1) * T * kBlkRows + ((crank * nq + q) * T) * 16) * 4u;
                bulkS2Cluster(mapaShared(xbase + off, uint32_t(lane)), stg, uint32_t(T * 16 * 4),
                    mapaShared(xbarBase + uint32_t(n & 1) * 8u, uint32_t(lane)));
            }
            if (lane < kCs)
            {
                bulkCommit();
            }
            ++q;
        }
        bulkWaitRead<0>(); // the DSMEM copies have read this CTA's staging slots
    }
    else
    {
        // ---- consumers (warps 0..4)
        int const g = lane >> 2;
        int const tig = lane & 3;
        uint4 bx[4][2];
        float xs[4][2];
#pragma unroll
        for (int q = 0; q < 4; ++q)
        {
            int const kb = warp * 4 + q;
#pragma unroll
            for (int m = 0; m < 2; ++m)
            {
                int const kpos = kb * 128 + m * 64 + tig * 16;
                bx[q][m] = g < T ? ldcg128(p.xq + size_t(g) * kH + kpos) : make_uint4(0, 0, 0, 0);
            }
            xs[q][0] = 2 * tig < T ? __ldcg(p.xsf + kb * p.xsfStride + 2 * tig) : 0.f;
            xs[q][1] = 2 * tig + 1 < T ? __ldcg(p.xsf + kb * p.xsfStride + 2 * tig + 1) : 0.f;
        }
        uint8_t* xsh = smemRaw + shXOffset; // [T][2560] bf16 MoE-site rows (shared fold only)
        if constexpr (kSh)
        {
            bool anySh = false;
            for (int n = 0; n < nMy; ++n)
            {
                anySh |= s.blkJ[cl + n * NC] >= kShMark;
            }
            if (anySh)
            {
                for (int q = threadIdx.x; q < T * kH / 8; q += kConsumerThreads)
                {
                    *reinterpret_cast<uint4*>(xsh + q * 16) = ldcg128(static_cast<uint8_t const*>(p.shX) + q * 16);
                }
                // Gate logit of each row: bf16(x . w) as the unfused GEMM outputs it, then its sigmoid in fp32.
                for (int t = warp; t < T; t += kConsumerWarps)
                {
                    float acc = 0.f;
                    for (int c = lane; c < kH / 8; c += 32)
                    {
                        uint4 const xv = ldcg128(static_cast<uint8_t const*>(p.shX) + size_t(t) * kShRowBytes + c * 16);
                        uint4 const wv = ldcg128(static_cast<uint8_t const*>(p.shGl) + c * 16);
                        __nv_bfloat162 const* xa = reinterpret_cast<__nv_bfloat162 const*>(&xv);
                        __nv_bfloat162 const* wa = reinterpret_cast<__nv_bfloat162 const*>(&wv);
#pragma unroll
                        for (int i = 0; i < 4; ++i)
                        {
                            float2 const xf = __bfloat1622float2(xa[i]);
                            float2 const wf = __bfloat1622float2(wa[i]);
                            acc = fmaf(xf.x, wf.x, acc);
                            acc = fmaf(xf.y, wf.y, acc);
                        }
                    }
#pragma unroll
                    for (int o = 16; o > 0; o >>= 1)
                    {
                        acc += __shfl_xor_sync(0xffffffffu, acc, o);
                    }
                    if (lane == 0)
                    {
                        s.shScale[t] = 1.0f / (1.0f + expf(-bf16Round(acc)));
                    }
                }
                namedBarSync(9, kConsumerThreads);
            }
        }
        Ring rs{0, 0u, numStages};
        uint32_t xphase = 0; // bit p: the phase of buffer p's barrier
        float out[kNh][kMtPerWarp][4];
#pragma unroll
        for (int u = 0; u < kNh; ++u)
        {
#pragma unroll
            for (int mi = 0; mi < kMtPerWarp; ++mi)
            {
#pragma unroll
                for (int r = 0; r < 4; ++r)
                {
                    out[u][mi][r] = 0.f;
                }
            }
        }
        int jf = 0;
        for (int n = 0; n < nMy; ++n)
        {
            int const pb = n & 1;
            int const d = s.blkD[cl + n * NC];
            if constexpr (kSh)
            {
                if (s.blkJ[cl + n * NC] >= kShMark)
                {
                    // ---- shared item: bf16 FC1 of 2 x 8 rows, warp w over k [512 w, 512 w + 512)
                    for (int q = 0; q < 2; ++q, ++jf)
                    {
                        mbarWait(&s.fullBar[rs.stage], rs.phase);
                        uint8_t const* st = ring + rs.stage * kStageBytes;
                        float acc[4] = {0.f, 0.f, 0.f, 0.f};
                        float acc1[4] = {0.f, 0.f, 0.f, 0.f};
                        uint4 const zero = make_uint4(0, 0, 0, 0);
#pragma unroll 4
                        for (int m = 0; m < 16; m += 2)
                        {
                            int const k0 = warp * 512 + m * 32 + tig * 8;
                            uint4 const a0 = lds128(st + g * kShRowBytes + k0 * 2);
                            uint4 const a1 = lds128(st + g * kShRowBytes + (k0 + 32) * 2);
                            uint4 const b0 = g < T ? lds128(xsh + g * kShRowBytes + k0 * 2) : zero;
                            uint4 const b1 = g < T ? lds128(xsh + g * kShRowBytes + (k0 + 32) * 2) : zero;
                            mma32Bf16(acc, a0, zero, b0);
                            mma32Bf16(acc1, a1, zero, b1);
                        }
                        __syncwarp();
                        if (lane == 0)
                        {
                            mbarArrive(&s.emptyBar[rs.stage]);
                        }
                        rs.advance();
                        int const buf = jf & 1;
                        if (jf >= 2)
                        {
                            namedBarSync(3 + buf, kConsumerThreads + 32);
                        }
                        s.red[buf][warp][lane] = make_float4(acc[0] + acc1[0], acc[1] + acc1[1], 0.f, 0.f);
                        namedBarArrive(1 + buf, kConsumerThreads + 32);
                    }
                    if (warp == 0 && lane == 0)
                    {
                        mbarArriveExpectTx(&s.xbar[pb], uint32_t(kCs * 2 * T * 16 * 4));
                    }
                    mbarWaitCluster(&s.xbar[pb], (xphase >> pb) & 1u);
                    xphase ^= 1u << pb;
                    // Column c comes from CTA c / 8, item (c % 8) / 4, rows c % 4 (gate) and 4 + c % 4 (up).
                    float const* xb = xbuf + pb * T * kBlkRows;
                    for (int i = threadIdx.x; i < T * kShCols; i += kConsumerThreads)
                    {
                        int const t = i / kShCols;
                        int const c = i % kShCols;
                        float const* src = xb + (((c / 8) * 2 + (c % 8) / 4) * T + t) * 16 + (c % 4);
                        float const ga = bf16Round(src[0]);
                        float const up = bf16Round(src[4]);
                        s.actS[pb][t][c] = __float2bfloat16_rn(ga / (1.0f + expf(-ga)) * up);
                    }
                    namedBarSync(8, kConsumerThreads);
                    // FC2 of this CTA's 640 down rows, scaled by sigmoid(gate logit), into the routed accumulators.
                    uint4 const zero = make_uint4(0, 0, 0, 0);
                    uint4 const bsh = g < T ? lds128(reinterpret_cast<uint8_t const*>(&s.actS[pb][g][tig * 8])) : zero;
                    float const sc0 = 2 * tig < T ? s.shScale[2 * tig] : 0.f;
                    float const sc1 = 2 * tig + 1 < T ? s.shScale[2 * tig + 1] : 0.f;
                    mbarWait(&s.fullBar[rs.stage], rs.phase);
                    uint8_t const* st = ring + rs.stage * kStageBytes;
#pragma unroll
                    for (int u = 0; u < kNh; ++u)
                    {
#pragma unroll
                        for (int mi = 0; mi < kMtPerWarp; ++mi)
                        {
                            int const row = u * kItemFc2Rows + (warp + kConsumerWarps * mi) * 16 + g;
                            float acc[4] = {0.f, 0.f, 0.f, 0.f};
                            uint4 const ag = lds128(st + row * kShCols * 2 + tig * 16);
                            uint4 const ag8 = lds128(st + (row + 8) * kShCols * 2 + tig * 16);
                            mma32Bf16(acc, ag, ag8, bsh);
                            out[u][mi][0] = fmaf(acc[0], sc0, out[u][mi][0]);
                            out[u][mi][1] = fmaf(acc[1], sc1, out[u][mi][1]);
                            out[u][mi][2] = fmaf(acc[2], sc0, out[u][mi][2]);
                            out[u][mi][3] = fmaf(acc[3], sc1, out[u][mi][3]);
                        }
                    }
                    __syncwarp();
                    if (lane == 0)
                    {
                        mbarArrive(&s.emptyBar[rs.stage]);
                    }
                    rs.advance();
                    continue;
                }
            }
            // FC1: the up items, then the gate items
            for (int q = 0; q < 2 * kNh; ++q, ++jf)
            {
                mbarWait(&s.fullBar[rs.stage], rs.phase);
                uint8_t const* st = ring + rs.stage * kStageBytes;
                float const* sc = reinterpret_cast<float const*>(st + kItemRowBytes);
                float t0 = 0.f, t1 = 0.f, t2 = 0.f, t3 = 0.f;
#pragma unroll
                for (int qq = 0; qq < 4; ++qq)
                {
                    int const kb = warp * 4 + qq;
                    float acc[4] = {0.f, 0.f, 0.f, 0.f};
                    float acc1[4] = {0.f, 0.f, 0.f, 0.f};
                    int const koff = kb * 128 + tig * 16;
                    uint4 const a0 = lds128(st + g * kH + koff);
                    uint4 const a08 = lds128(st + (g + 8) * kH + koff);
                    uint4 const a1 = lds128(st + g * kH + koff + 64);
                    uint4 const a18 = lds128(st + (g + 8) * kH + koff + 64);
                    mmaE4m3(acc, a0.x, a08.x, a0.y, a08.y, bx[qq][0].x, bx[qq][0].y);
                    mmaE4m3(acc1, a1.x, a18.x, a1.y, a18.y, bx[qq][1].x, bx[qq][1].y);
                    mmaE4m3(acc, a0.z, a08.z, a0.w, a08.w, bx[qq][0].z, bx[qq][0].w);
                    mmaE4m3(acc1, a1.z, a18.z, a1.w, a18.w, bx[qq][1].z, bx[qq][1].w);
#pragma unroll
                    for (int r = 0; r < 4; ++r)
                    {
                        acc[r] += acc1[r];
                    }
                    float const sw = sc[kb];
                    t0 = fmaf(acc[0], sw * xs[qq][0], t0);
                    t1 = fmaf(acc[1], sw * xs[qq][1], t1);
                    t2 = fmaf(acc[2], sw * xs[qq][0], t2);
                    t3 = fmaf(acc[3], sw * xs[qq][1], t3);
                }
                __syncwarp();
                if (lane == 0)
                {
                    mbarArrive(&s.emptyBar[rs.stage]);
                }
                rs.advance();
                int const buf = jf & 1;
                if (jf >= 2)
                {
                    namedBarSync(3 + buf, kConsumerThreads + 32);
                }
                s.red[buf][warp][lane] = make_float4(t0, t1, t2, t3);
                namedBarArrive(1 + buf, kConsumerThreads + 32);
            }

            if (warp == 0 && lane == 0)
            {
                mbarArriveExpectTx(&s.xbar[pb], uint32_t(kCs * 2 * kNh * T * 16 * 4));
            }
            mbarWaitCluster(&s.xbar[pb], (xphase >> pb) & 1u);
            xphase ^= 1u << pb;
            // Activation, one routed token a warp: requantize up and gate (1x128), siluFast(gate) * up, requantize.
            float const* xb = xbuf + pb * T * kBlkRows;
            for (int t = warp; t < T; t += kConsumerWarps)
            {
                int const sl = s.slotOf[d][t];
                if (sl < 0)
                {
                    continue;
                }
                int const col = 4 * lane;
                int const c = col / kUpRows;
                int const u = (col % kUpRows) / 16;
                float const* up = xb + ((c * 2 * kNh + u) * T + t) * 16 + (col & 15);
                float const* gt = xb + ((c * 2 * kNh + kNh + u) * T + t) * 16 + (col & 15);
                float4 const upv = *reinterpret_cast<float4 const*>(up);
                float4 const gtv = *reinterpret_cast<float4 const*>(gt);
                float const ux[4] = {upv.x, upv.y, upv.z, upv.w};
                float const gx[4] = {gtv.x, gtv.y, gtv.z, gtv.w};
                float amU = 0.f, amG = 0.f;
#pragma unroll
                for (int i = 0; i < 4; ++i)
                {
                    amU = fmaxf(amU, fabsf(ux[i]));
                    amG = fmaxf(amG, fabsf(gx[i]));
                }
                float const sU = __fdiv_rn(fmaxf(warpMaxAbs(amU), kAmaxFloor), kE4m3Max);
                float const sG = __fdiv_rn(fmaxf(warpMaxAbs(amG), kAmaxFloor), kE4m3Max);
                float const rU = __frcp_rn(sU);
                float const rG = __frcp_rn(sG);
                float a[4];
                float amA = 0.f;
#pragma unroll
                for (int i = 0; i < 4; ++i)
                {
                    float const x1 = e4m3RoundTrip(ux[i] * rU) * sU;
                    float const x2 = e4m3RoundTrip(gx[i] * rG) * sG;
                    a[i] = siluFast(x2) * x1;
                    amA = fmaxf(amA, fabsf(a[i]));
                }
                float const sA = __fdiv_rn(fmaxf(warpMaxAbs(amA), kAmaxFloor), kE4m3Max);
                float const rA = __frcp_rn(sA);
                uint32_t packed = 0;
#pragma unroll
                for (int i = 0; i < 4; ++i)
                {
                    packed |= e4m3Code(a[i] * rA) << (8 * i);
                }
                *reinterpret_cast<uint32_t*>(&s.actC[pb][t][lane * 4]) = packed;
                if (lane == 0)
                {
                    s.actF[pb][t] = sA * s.topkW[t][sl % kK];
                }
            }
            namedBarSync(8, kConsumerThreads);

            int const sg = g < T ? s.slotOf[d][g] : -1;
            int const sa = 2 * tig < T ? s.slotOf[d][2 * tig] : -1;
            int const sb = 2 * tig + 1 < T ? s.slotOf[d][2 * tig + 1] : -1;
            float const fa = sa >= 0 ? s.actF[pb][2 * tig] : 0.f;
            float const fb = sb >= 0 ? s.actF[pb][2 * tig + 1] : 0.f;
            uint4 bm0 = make_uint4(0, 0, 0, 0), bm1 = make_uint4(0, 0, 0, 0);
            if (sg >= 0)
            {
                bm0 = lds128(&s.actC[pb][g][tig * 16]);
                bm1 = lds128(&s.actC[pb][g][64 + tig * 16]);
            }
#pragma unroll
            for (int u = 0; u < kNh; ++u)
            {
                mbarWait(&s.fullBar[rs.stage], rs.phase);
                uint8_t const* st = ring + rs.stage * kStageBytes;
                float const* sc2 = reinterpret_cast<float const*>(st + kItemRowBytes);
#pragma unroll
                for (int mi = 0; mi < kMtPerWarp; ++mi)
                {
                    int const mt = warp + kConsumerWarps * mi;
                    float acc[4] = {0.f, 0.f, 0.f, 0.f};
                    float acc1[4] = {0.f, 0.f, 0.f, 0.f};
                    uint8_t const* pb2 = st + mt * 16 * 128;
                    int const koff = tig * 16;
                    uint4 const a0 = lds128(pb2 + g * 128 + koff);
                    uint4 const a08 = lds128(pb2 + (g + 8) * 128 + koff);
                    uint4 const a1 = lds128(pb2 + g * 128 + koff + 64);
                    uint4 const a18 = lds128(pb2 + (g + 8) * 128 + koff + 64);
                    mmaE4m3(acc, a0.x, a08.x, a0.y, a08.y, bm0.x, bm0.y);
                    mmaE4m3(acc1, a1.x, a18.x, a1.y, a18.y, bm1.x, bm1.y);
                    mmaE4m3(acc, a0.z, a08.z, a0.w, a08.w, bm0.z, bm0.w);
                    mmaE4m3(acc1, a1.z, a18.z, a1.w, a18.w, bm1.z, bm1.w);
#pragma unroll
                    for (int r = 0; r < 4; ++r)
                    {
                        acc[r] += acc1[r];
                    }
                    float const s2 = sc2[(crank * kFc2Rows + u * kItemFc2Rows + mt * 16) >> 7];
                    float const f0 = s2 * fa;
                    float const f1 = s2 * fb;
                    out[u][mi][0] = fmaf(acc[0], f0, out[u][mi][0]);
                    out[u][mi][1] = fmaf(acc[1], f1, out[u][mi][1]);
                    out[u][mi][2] = fmaf(acc[2], f0, out[u][mi][2]);
                    out[u][mi][3] = fmaf(acc[3], f1, out[u][mi][3]);
                }
                __syncwarp();
                if (lane == 0)
                {
                    mbarArrive(&s.emptyBar[rs.stage]);
                }
                rs.advance();
            }
        }
        if (nMy > 0)
        {
            int const t0 = 2 * tig, t1 = 2 * tig + 1;
#pragma unroll
            for (int u = 0; u < kNh; ++u)
            {
#pragma unroll
                for (int mi = 0; mi < kMtPerWarp; ++mi)
                {
                    int const row = crank * kFc2Rows + u * kItemFc2Rows + (warp + kConsumerWarps * mi) * 16 + g;
                    float* pp = part + size_t(cl) * kRows * kH + row;
                    if (t0 < T)
                    {
                        __stcg(pp + t0 * kH, out[u][mi][0]);
                        __stcg(pp + t0 * kH + 8, out[u][mi][2]);
                    }
                    if (t1 < T)
                    {
                        __stcg(pp + t1 * kH, out[u][mi][1]);
                        __stcg(pp + t1 * kH + 8, out[u][mi][3]);
                    }
                }
            }
        }
    }

    // ---- past the grid counter, each CTA sums its share of the NCa cluster partials: 16 threads a (token, 4 rows)
    // unit, thread q summing partials q, q + 16, ... in order, then a fixed shuffle tree.
    __syncthreads();
    if (threadIdx.x == 0)
    {
        // Every CTA arrives, inactive ones too, so past the wait every CTA is past its prologue.
        fenceAcqRelGpu();
        redAddRelease64(counter, cta == 0 ? kStride - (unsigned long long) (G - 1) : 1ull);
        unsigned long long const target = s.barTarget;
        while (ldAcquireGpu64(counter) < target)
        {
        }
    }
    __syncthreads();
    {
        int const nUnits = T * (kH / 4);
        int const u0 = int((long long) cta * nUnits / G);
        int const u1 = int((long long) (cta + 1) * nUnits / G);
        int const grp = threadIdx.x >> 4;
        int const q = threadIdx.x & 15;
        constexpr int kGroups = kThreads / 16;
        __nv_bfloat16* o = static_cast<__nv_bfloat16*>(p.out);
        for (int ub = u0; ub < u1; ub += kGroups)
        {
            int const u = ub + grp;
            bool const live = u < u1;
            int const t = live ? u / (kH / 4) : 0;
            int const row = live ? (u % (kH / 4)) * 4 : 0;
            float const* src = part + size_t(t) * kH + row;
            float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
            if (live)
            {
                for (int k = q; k < NCa; k += 16)
                {
                    float4 const x = __ldcg(reinterpret_cast<float4 const*>(src + size_t(k) * kRows * kH));
                    v.x += x.x;
                    v.y += x.y;
                    v.z += x.z;
                    v.w += x.w;
                }
            }
#pragma unroll
            for (int off = 8; off > 0; off >>= 1)
            {
                v.x += __shfl_xor_sync(0xffffffffu, v.x, off);
                v.y += __shfl_xor_sync(0xffffffffu, v.y, off);
                v.z += __shfl_xor_sync(0xffffffffu, v.z, off);
                v.w += __shfl_xor_sync(0xffffffffu, v.w, off);
            }
            if (live && q == 0)
            {
                __nv_bfloat162 const lo = __floats2bfloat162_rn(v.x, v.y);
                __nv_bfloat162 const hi = __floats2bfloat162_rn(v.z, v.w);
                uint2 pk;
                pk.x = *reinterpret_cast<uint32_t const*>(&lo);
                pk.y = *reinterpret_cast<uint32_t const*>(&hi);
                *reinterpret_cast<uint2*>(o + size_t(t) * kH + row) = pk;
            }
        }
    }
#endif // FP8_BLOCK_MOE_TPB_DEVICE
}

int smemOptinTpb(int device)
{
    int v = 0;
    cudaDeviceGetAttribute(&v, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    return v;
}

template <int CS, bool kSh>
int activeClustersT(int device)
{
    static int cache[16] = {};
    static std::mutex mu;
    std::lock_guard<std::mutex> lock(mu);
    if (device >= 0 && device < 16 && cache[device] > 0)
    {
        return cache[device];
    }
    int const maxSmemBytes = smemOptinTpb(device);
    cudaError_t err = cudaFuncSetAttribute(
        fp8BlockMoeTpbKernel<CS, kSh>, cudaFuncAttributeMaxDynamicSharedMemorySize, maxSmemBytes);
    if (err != cudaSuccess)
    {
        throw std::runtime_error(std::string("fp8_block_moe_tpb: smem attribute: ") + cudaGetErrorString(err));
    }
    cudaLaunchConfig_t cfg{};
    cfg.gridDim = dim3(CS * kMaxClusters);
    cfg.blockDim = dim3(kThreads);
    cfg.dynamicSmemBytes = size_t(maxSmemBytes);
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeClusterDimension;
    attr[0].val.clusterDim.x = CS;
    attr[0].val.clusterDim.y = 1;
    attr[0].val.clusterDim.z = 1;
    cfg.attrs = attr;
    cfg.numAttrs = 1;
    int n = 0;
    err = cudaOccupancyMaxActiveClusters(&n, fp8BlockMoeTpbKernel<CS, kSh>, &cfg);
    if (err != cudaSuccess || n <= 0)
    {
        throw std::runtime_error(std::string("fp8_block_moe_tpb: occupancy: ") + cudaGetErrorString(err));
    }
    n = std::min(n, kMaxClusters);
    if (device >= 0 && device < 16)
    {
        cache[device] = n;
    }
    return n;
}

// Few rows have few blocks, so give each more CTAs; from 3 rows, 4-CTA clusters keep more clusters resident.
int clusterSize(int rows)
{
    return rows <= 2 ? 8 : 4;
}

template <int CS, bool kSh>
void launchTpbT(TpbParams const& params, bool pdl, cudaStream_t stream)
{
    int device = 0;
    cudaGetDevice(&device);
    int const clusters = activeClustersT<CS, kSh>(device);
    int const xbufOffset = kStaticBytes<kSh>;
    int const xbufBytes = (2 * params.numRows * kBlkRows * 4 + 127) / 128 * 128;
    int const shXOffset = xbufOffset + xbufBytes;
    int const shXBytes = kSh ? (params.numRows * kShRowBytes + 127) / 128 * 128 : 0;
    int const ringOffset = shXOffset + shXBytes;
    int const fit = (smemOptinTpb(device) - ringOffset) / kStageBytes;
    int const stages = std::min(std::min(kDefaultStages, fit), kMaxStages);
    if (stages < 3)
    {
        throw std::invalid_argument("fp8_block_moe_tpb: three ring stages do not fit in shared memory");
    }
    size_t const smem = size_t(ringOffset) + size_t(stages) * kStageBytes;
    cudaLaunchConfig_t cfg{};
    cfg.gridDim = dim3(clusters * CS);
    cfg.blockDim = dim3(kThreads);
    cfg.dynamicSmemBytes = smem;
    cfg.stream = stream;
    cudaLaunchAttribute attr[2];
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = pdl ? 1 : 0;
    attr[1].id = cudaLaunchAttributeClusterDimension;
    attr[1].val.clusterDim.x = CS;
    attr[1].val.clusterDim.y = 1;
    attr[1].val.clusterDim.z = 1;
    cfg.attrs = attr;
    cfg.numAttrs = 2;
    cudaError_t const err
        = cudaLaunchKernelEx(&cfg, fp8BlockMoeTpbKernel<CS, kSh>, params, stages, xbufOffset, ringOffset, shXOffset);
    if (err != cudaSuccess)
    {
        throw std::runtime_error(std::string("fp8_block_moe_tpb: launch: ") + cudaGetErrorString(err));
    }
}

} // namespace

size_t tpbWorkspaceBytes()
{
    return kWsBytes;
}

void launchTpb(TpbParams const& params, bool pdl, cudaStream_t stream)
{
    if (params.numRows < 1 || params.numRows > kRows)
    {
        throw std::invalid_argument("fp8_block_moe_tpb: 1..8 rows");
    }
    if (params.rank < 0 || params.rank >= kTpbRanks)
    {
        throw std::invalid_argument("fp8_block_moe_tpb: rank 0..3 of 4");
    }
    if (params.shBlocks > 0)
    {
        if (params.shBlocks != kShItems || clusterSize(params.numRows) != 4 || params.shX == nullptr
            || params.shGu == nullptr || params.shGl == nullptr || params.shDn == nullptr)
        {
            throw std::invalid_argument(
                "fp8_block_moe_tpb: the shared-expert fold takes 5 slice items and 4-CTA clusters (3..8 rows)");
        }
        launchTpbT<4, true>(params, pdl, stream);
    }
    else if (clusterSize(params.numRows) == 8)
    {
        launchTpbT<8, false>(params, pdl, stream);
    }
    else
    {
        launchTpbT<4, false>(params, pdl, stream);
    }
}

} // namespace kernels::fp8_block_moe_tpb

TRTLLM_MV2_NAMESPACE_END
