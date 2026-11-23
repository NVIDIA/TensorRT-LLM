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

#include "glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/csrc/postAttentionHopFromPartials.h"
#include "tensorrt_llm/modeling_v2/common/namespace.h"

#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <type_traits>

TRTLLM_MV2_NAMESPACE_BEGIN

namespace attention_output_into_hop
{
namespace
{
namespace cg = cooperative_groups;

// The post-attention hop (postAttentionHop.cu) with its input taken from the transport: the only differences
// are marked "Transport" below; the work decomposition, the arithmetic and the protocol are that hop's.
//
// Work decomposition.
//
// Every cluster has kSlices CTAs; CTA c owns slice [c * kSlice, (c + 1) * kSlice), the slice a CTA of the native
// one-shot kernel owns for numRows <= 8 (96 threads x 8 elements, 8 CTAs per row), so the RMSNorm reduction below
// reproduces the native tree exactly.
//
// Without router logits (dense layers) the layout is the native one: cluster y handles row y, and every CTA collects
// one row slice (8 ranks x 768 elements).
//
// With router logits there are kG x kRowSplit clusters (kG = 4, kRowSplit = min(R, 3): at most 12 clusters, which stay
// co-resident). Cluster (g, h) collects and normalizes the rows r with r % kRowSplit == h, so a CTA reads at most
// ceil(R / kRowSplit) row slices, and computes the logits of those rows for experts [64 g, 64 g + 64). Its router
// weights live in registers (loaded before the PDL wait). Each CTA multiplies its rows of slice c by the experts'
// weights of slice c with BF16 MMAs (experts on m16, rows on n8) and pushes the partial logits to the CTA that owns the
// expert; owners add the 8 slices in a fixed order, so the logits are identical on every rank. Clusters never
// synchronize with each other; inside a cluster, slice sums and partial logits travel as st.async stores that complete
// transactions on the consumer's mbarrier.
constexpr int kSlices = 8;
constexpr int kSlice = kHiddenDim / kSlices;
constexpr int kEltsPerVec = 8;
constexpr int kVecPerSlice = kSlice / kEltsPerVec;
constexpr int kNativeWarpsPerSlice = kVecPerSlice / 32;
// Row stride of the normalized-slice tile in 16-byte vectors; 100 = 96 + 4 keeps the 16-byte B-fragment loads of two
// rows in one quarter-warp phase on disjoint banks.
constexpr int kNormedStride = kVecPerSlice + 4;
constexpr int kMmaChains = 4;
constexpr uint32_t kLamportSentinel = 0x80000000U;

static_assert(kHiddenDim % (kSlices * kEltsPerVec * 32) == 0);
static_assert(kVecPerSlice == kNativeWarpsPerSlice * 32);
static_assert(kMaxRows == 8, "the router MMA maps rows onto the n8 dimension");

// kG = number of expert groups (0: hop without router logits); kRowSplit = number of row groups.
template <int kG, int kRowSplit>
struct HopConfig
{
    static constexpr int kThreads = 384;
    static constexpr int kWarps = kThreads / 32;
    static constexpr int kExpertsPerCluster = kNumExperts / kG;
    static constexpr int kExpertTiles = kExpertsPerCluster / 16;
    static constexpr int kKGroups = kWarps / kExpertTiles;
    static constexpr int kKPerGroup = kSlice / kKGroups;
    static constexpr int kPairsPerGroup = kKPerGroup / 32;
    static constexpr int kExpertsPerOwner = kExpertsPerCluster / kSlices;
    static_assert(kExpertsPerCluster % 16 == 0 && kWarps % kExpertTiles == 0 && kSlice % (kKGroups * 32) == 0);
    static_assert(kPairsPerGroup % (kMmaChains / 2) == 0);
};

template <int kRowSplit>
struct HopConfig<0, kRowSplit>
{
    static constexpr int kThreads = 128;
    static constexpr int kWarps = kThreads / 32;
    static constexpr int kMaxRowsPerCta = 1;
    static_assert(kMaxRowsPerCta * kVecPerSlice <= kThreads && kMaxRowsPerCta * kNativeWarpsPerSlice <= kWarps);
};

constexpr int kMaxRowsPerCtaAll = kMaxRows / 2;

struct alignas(16) HopStorage
{
    uint4 gamma[kVecPerSlice];
    float vecSq[kMaxRowsPerCtaAll][kVecPerSlice];
    float warpSq[kMaxRowsPerCtaAll][kNativeWarpsPerSlice];
    float blockSq[kMaxRowsPerCtaAll][kSlices];
    // Completes when all slice sums of this CTA's rows have arrived (st.async complete_tx).
    alignas(8) uint64_t normBar;
    // Completes when all partial logits of the experts this CTA owns have arrived.
    alignas(8) uint64_t routerBar;
};

template <int kG, int kRowSplit>
struct alignas(16) RouterStorage
{
    using Cfg = HopConfig<kG, kRowSplit>;
    uint4 normed[kMaxRows][kNormedStride]; // local row on the MMA n dimension; unused rows stay zero
    float partial[kSlices][Cfg::kKGroups][Cfg::kExpertsPerOwner][kMaxRows];
};

struct EmptyStorage
{
};

union BfVec
{
    uint4 packed;
    __nv_bfloat16 elts[kEltsPerVec];
};

__device__ __forceinline__ uint4 loadVolatile(void const* ptr)
{
    uint4 v;
    asm volatile("ld.volatile.global.v4.u32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                 : "l"(ptr)
                 : "memory");
    return v;
}

__device__ __forceinline__ uint4 loadConstant(void const* ptr)
{
    uint4 v;
    asm volatile("ld.global.nc.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(ptr));
    return v;
}

__device__ __forceinline__ void prefetchL2(void const* ptr)
{
    asm volatile("prefetch.global.L2 [%0];" ::"l"(ptr));
}

__device__ __forceinline__ bool isDirty(uint4 const& v)
{
    return v.x == kLamportSentinel || v.y == kLamportSentinel || v.z == kLamportSentinel || v.w == kLamportSentinel;
}

// Transport: the producer writes the Lamport payloads already sanitized (bf16 -0.0 mapped to +0.0), so they never
// contain the sentinel word.

__device__ __forceinline__ uint32_t sharedAddr(void const* ptr)
{
    return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

// Address of the same shared-memory object in CTA `rank` of the cluster.
__device__ __forceinline__ uint32_t clusterAddr(uint32_t addr, uint32_t rank)
{
    uint32_t out;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(out) : "r"(addr), "r"(rank));
    return out;
}

__device__ __forceinline__ void mbarrierInitExpectTx(uint64_t* bar, uint32_t bytes)
{
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(sharedAddr(bar)) : "memory");
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(sharedAddr(bar)), "r"(bytes)
                 : "memory");
}

__device__ __forceinline__ void mbarrierWaitPhase0(uint64_t* bar)
{
    asm volatile(
        "{\n"
        ".reg .pred done;\n"
        "HOP_MBAR_WAIT_%=:\n"
        "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 done, [%0], 0;\n"
        "@!done bra HOP_MBAR_WAIT_%=;\n"
        "}\n" ::"r"(sharedAddr(bar))
        : "memory");
}

// Remote shared-memory stores that complete transactions on the destination CTA's mbarrier.
__device__ __forceinline__ void storeAsync(uint32_t remoteAddr, float v, uint32_t remoteBar)
{
    asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.f32 [%0], %1, [%2];" ::"r"(remoteAddr), "f"(v),
                 "r"(remoteBar)
                 : "memory");
}

__device__ __forceinline__ void storeAsync(uint32_t remoteAddr, float a, float b, uint32_t remoteBar)
{
    asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v2.f32 [%0], {%1, %2}, [%3];" ::"r"(remoteAddr),
                 "f"(a), "f"(b), "r"(remoteBar)
                 : "memory");
}

__device__ __forceinline__ void mmaBf16(
    float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1)
{
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
        "{%0, %1, %2, %3};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// Transport: one warp's router-weight MMA fragments (the post-attention hop's register layout): experts
// tile * 16 + gid (+ 8) of this cluster's expert group, k pairs of this warp's k group within the CTA's slice.
template <class Cfg>
__device__ __forceinline__ void loadRouterFragments(uint4 (&wLo)[Cfg::kPairsPerGroup],
    uint4 (&wHi)[Cfg::kPairsPerGroup], void const* routerWeight, int expertGroup, int warp, int gid, int tig,
    int sliceBase)
{
    int const tile = warp % Cfg::kExpertTiles;
    int const kGroup = warp / Cfg::kExpertTiles;
    auto const* w = reinterpret_cast<__nv_bfloat16 const*>(routerWeight);
    int const expert = expertGroup * Cfg::kExpertsPerCluster + tile * 16 + gid;
#pragma unroll
    for (int p = 0; p < Cfg::kPairsPerGroup; ++p)
    {
        int const k = sliceBase + kGroup * Cfg::kKPerGroup + p * 32 + tig * kEltsPerVec;
        wLo[p] = loadConstant(w + static_cast<size_t>(expert) * kHiddenDim + k);
        wHi[p] = loadConstant(w + static_cast<size_t>(expert + 8) * kHiddenDim + k);
    }
}

// Transport: where the router-weight fragments are loaded (all orders give the same values and arithmetic):
//   kRouterAfterBarrier: after the barrier that ends the rank sum, overlapping the RMS tree and cluster exchange;
//   kRouterAfterSum:     right after the rank sum, before that barrier (the default; the loads go through
//                        loadRouterFragments so the rank sum is not sunk below them and the kernel stays at
//                        few enough registers to share an SM with a producer CTA);
//   kRouterBeforeMma:    right before the MMAs (fewest registers, the load latency follows the RMSNorm).
constexpr int kRouterAfterBarrier = 0;
constexpr int kRouterAfterSum = 1;
constexpr int kRouterBeforeMma = 2;

template <int R, int kG, int kRowSplit, int kRouterLoad>
__global__ void __launch_bounds__(HopConfig<kG, kRowSplit>::kThreads, 1)
    postAttnHopX1Kernel(PostAttentionHopFromPartialsParams const params)
{
    static_assert(R >= 1 && R <= kMaxRows);
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    constexpr bool kRouter = kG > 0;
    using Cfg = HopConfig<kG, kRowSplit>;
    constexpr int kClusters = kRouter ? kG * kRowSplit : R;
    constexpr int kGridCtas = kClusters * kSlices;
    // Rows one CTA collects, normalizes and multiplies.
    constexpr int kRowsPerCta = kRouter ? (R + kRowSplit - 1) / kRowSplit : 1;
    static_assert(kRowsPerCta <= kMaxRowsPerCtaAll);
    static_assert(kRowsPerCta * kVecPerSlice <= Cfg::kThreads && kRowsPerCta * kNativeWarpsPerSlice <= Cfg::kWarps);
    if constexpr (kRouter)
    {
        static_assert(
            HopConfig<(kG > 0 ? kG : 4), kRowSplit>::kExpertsPerOwner * kRowsPerCta * kSlices <= Cfg::kThreads);
    }

    __shared__ HopStorage hop;
    __shared__ std::conditional_t<kRouter, RouterStorage<(kG > 0 ? kG : 4), (kG > 0 ? kRowSplit : 2)>, EmptyStorage> rs;

    cg::cluster_group cluster = cg::this_cluster();
    int const slice = static_cast<int>(cluster.block_rank());
    int const clusterId = blockIdx.y;
    int const expertGroup = kRouter ? clusterId % (kG > 0 ? kG : 1) : 0;
    int const rowGroup = kRouter ? clusterId / (kG > 0 ? kG : 1) : clusterId;
    int const tid = threadIdx.x;
    int const lane = tid & 31;
    int const warp = tid >> 5;
    int const sliceBase = slice * kSlice;

    // Rows this CTA collects and normalizes: local row j is global row rowGroup + j * rowStride.
    constexpr int kRowStride = kRouter ? kRowSplit : 1;
    int const numRowsCta = kRouter ? (R - rowGroup + kRowSplit - 1) / kRowSplit : 1;
    // This thread's item: one 16-byte vector of one row slice.
    int const itemRow = tid / kVecPerSlice;
    int const itemVec = tid % kVecPerSlice;
    bool const hasItem = itemRow < numRowsCta;
    int const row = rowGroup + itemRow * kRowStride;
    size_t const rowOff = static_cast<size_t>(row) * kHiddenDim + sliceBase + itemVec * kEltsPerVec;
    // The cluster that writes the normalized rows and the residual.
    bool const writesOutputs = !kRouter || expertGroup == 0;

    // ---- Prologue: constant model weights only; may overlap the producer of `input` under PDL. ----
    if (tid < kVecPerSlice)
    {
        hop.gamma[tid] = loadConstant(reinterpret_cast<uint4 const*>(params.gamma) + slice * kVecPerSlice + tid);
    }
    [[maybe_unused]] int const gid = lane >> 2;
    [[maybe_unused]] int const tig = lane & 3;
    // Transport: the router-weight fragments are loaded after the poll (not here), so they are not live across it; with
    // fewer registers a CTA of this kernel can be resident next to a CTA of the producer. The prologue only asks L2
    // for the same lines (no registers held, no architectural effect).
    if constexpr (kRouter)
    {
        {
            int const tile = warp % Cfg::kExpertTiles;
            int const kGroup = warp / Cfg::kExpertTiles;
            auto const* w = reinterpret_cast<__nv_bfloat16 const*>(params.routerWeight);
            int const expert = expertGroup * Cfg::kExpertsPerCluster + tile * 16 + gid;
#pragma unroll
            for (int p = 0; p < Cfg::kPairsPerGroup; ++p)
            {
                int const k = sliceBase + kGroup * Cfg::kKPerGroup + p * 32 + tig * kEltsPerVec;
                prefetchL2(w + static_cast<size_t>(expert) * kHiddenDim + k);
                prefetchL2(w + static_cast<size_t>(expert + 8) * kHiddenDim + k);
            }
        }
        // MMA columns of rows this CTA does not own stay zero.
        for (int i = tid; i < kMaxRows * kNormedStride; i += Cfg::kThreads)
        {
            rs.normed[i / kNormedStride][i % kNormedStride] = make_uint4(0, 0, 0, 0);
        }
    }

    // Cluster-local completion barriers: every CTA of a cluster has the same rows, so each expects the 8 slice sums
    // of its rows and, with the router, the 8 x kKGroups partial tiles of the experts it owns. A split cluster
    // barrier publishes the initialization; its wait precedes the first remote store, long after all CTAs arrived.
    if (tid == 0)
    {
        mbarrierInitExpectTx(&hop.normBar, static_cast<uint32_t>(numRowsCta * kSlices * sizeof(float)));
        if constexpr (kRouter)
        {
            mbarrierInitExpectTx(&hop.routerBar, numRowsCta > 0 ? static_cast<uint32_t>(sizeof(rs.partial)) : 0U);
        }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");

    // Transport: with earlyPoll no wait here. The producer read the protocol state and wrote the transport after its
    // own griddepcontrol.wait and triggered this grid only after that wait, so every grid before the producer has
    // completed and the protocol state written by the previous hop is final (see postAttentionHopFromPartials.h).
    auto const* residualIn = reinterpret_cast<__nv_bfloat16 const*>(params.residualIn);
    uint4 res = make_uint4(0, 0, 0, 0);
    if (!params.earlyPoll)
    {
        asm volatile("griddepcontrol.wait;" ::: "memory");
        // As in the post-attention hop: dependents may start their prologue; they still wait for this grid to complete.
        asm volatile("griddepcontrol.launch_dependents;");
        // Transport: after the wait the residual is final; its load overlaps the poll instead of following it.
        if (hasItem)
        {
            res = *reinterpret_cast<uint4 const*>(residualIn + rowOff);
        }
    }
    uint4 const flags = loadVolatile(params.flags);
    uint32_t const cur = flags.x;
    uint32_t const clearBytes = flags.y;
    char* ucBase = reinterpret_cast<char*>(params.ucBuffer);
    auto* curUc = reinterpret_cast<__nv_bfloat16*>(ucBase + cur * kLamportBufferBytes);
    // Lamport layout of one buffer: [row][source rank][kHiddenDim].
    size_t const slotBase = static_cast<size_t>(row) * kNumRanks * kHiddenDim + sliceBase + itemVec * kEltsPerVec;

    // ---- Transport: collect every rank's partial, this rank's own included, from the transport. ----
    uint4 vals[kNumRanks];
    if (hasItem)
    {
#pragma unroll
        for (int s = 0; s < kNumRanks; ++s)
        {
            vals[s] = loadVolatile(curUc + slotBase + static_cast<size_t>(s) * kHiddenDim);
        }
    }

    // ---- Reset the previous call's buffer to the sentinel (reused two calls from now), off the poll's path. ----
    {
        uint32_t const dirty = (cur + kNumLamportBuffers - 1) % kNumLamportBuffers;
        auto* dirtyBuf = reinterpret_cast<uint4*>(ucBase + dirty * kLamportBufferBytes);
        uint32_t const numVec = clearBytes / sizeof(uint4);
        uint4 const sentinel = make_uint4(kLamportSentinel, kLamportSentinel, kLamportSentinel, kLamportSentinel);
        uint32_t const ctaLinear = clusterId * kSlices + slice;
        for (uint32_t j = ctaLinear * Cfg::kThreads + tid; j < numVec; j += kGridCtas * Cfg::kThreads)
        {
            dirtyBuf[j] = sentinel;
        }
    }

    if (hasItem)
    {
        bool pending = true;
        while (pending)
        {
            pending = false;
#pragma unroll
            for (int s = 0; s < kNumRanks; ++s)
            {
                if (isDirty(vals[s]))
                {
                    pending = true;
                    vals[s] = loadVolatile(curUc + slotBase + static_cast<size_t>(s) * kHiddenDim);
                }
            }
        }
    }

    // ---- Transport (earlyPoll): wait for the producer grid before reading the residual or writing any output. ----
    if (params.earlyPoll)
    {
        asm volatile("griddepcontrol.wait;" ::: "memory");
        // Dependents only start their own prologue; they still wait for this grid to complete.
        asm volatile("griddepcontrol.launch_dependents;");
        if (hasItem)
        {
            res = *reinterpret_cast<uint4 const*>(residualIn + rowOff);
        }
    }

    // Transport: router-weight fragments (see kRouterLoad).
    constexpr int kWPairs = kRouter ? HopConfig<(kG > 0 ? kG : 4), kRowSplit>::kPairsPerGroup : 1;
    [[maybe_unused]] uint4 wLo[kWPairs];
    [[maybe_unused]] uint4 wHi[kWPairs];

    // ---- Reduce in rank order, add the residual, and form per-vector sums of squares (native arithmetic). ----
    uint4 resOut = make_uint4(0, 0, 0, 0);
    if (hasItem)
    {
        float acc[kEltsPerVec];
#pragma unroll
        for (int e = 0; e < kEltsPerVec; ++e)
        {
            acc[e] = 0.F;
        }
#pragma unroll
        for (int s = 0; s < kNumRanks; ++s)
        {
            BfVec x;
            x.packed = vals[s];
#pragma unroll
            for (int e = 0; e < kEltsPerVec; ++e)
            {
                acc[e] += __bfloat162float(x.elts[e]);
            }
        }
        BfVec rv;
        rv.packed = res;
        BfVec y;
        float sq = 0.F;
#pragma unroll
        for (int e = 0; e < kEltsPerVec; ++e)
        {
            y.elts[e] = __hadd(__float2bfloat16(acc[e]), rv.elts[e]);
            sq += __bfloat162float(__hmul(y.elts[e], y.elts[e]));
        }
        resOut = y.packed;
        hop.vecSq[itemRow][itemVec] = sq;
    }
    if constexpr (kRouter && kRouterLoad == kRouterAfterSum)
    {
        if (numRowsCta > 0)
        {
            loadRouterFragments<Cfg>(wLo, wHi, params.routerWeight, expertGroup, warp, gid, tig, sliceBase);
        }
    }
    __syncthreads();
    if (tid == 0)
    {
        // Every thread of this CTA has read the protocol state; CTA 0 advances it only after all CTAs report.
        asm volatile("red.release.gpu.global.add.u32 [%0], %1;" ::"l"(params.flags + 2), "r"(1U) : "memory");
    }
    if constexpr (kRouter && kRouterLoad == kRouterAfterBarrier)
    {
        if (numRowsCta > 0)
        {
            loadRouterFragments<Cfg>(wLo, wHi, params.routerWeight, expertGroup, warp, gid, tig, sliceBase);
        }
    }

    // ---- RMS: warp butterfly over 32 vectors, then (w0 + w2) + w1 per slice, then the 8 slices in order. ----
    if (warp < numRowsCta * kNativeWarpsPerSlice)
    {
        int const j = warp / kNativeWarpsPerSlice;
        int const q = warp % kNativeWarpsPerSlice;
        float val = hop.vecSq[j][q * 32 + lane];
#pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1)
        {
            val = val + __shfl_xor_sync(0xffffffffU, val, mask, 32);
        }
        if (lane == 0)
        {
            hop.warpSq[j][q] = val;
        }
    }
    __syncthreads();
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    if (tid < numRowsCta * kSlices)
    {
        int const j = tid / kSlices;
        int const dst = tid % kSlices;
        float const blockSum = (hop.warpSq[j][0] + hop.warpSq[j][2]) + hop.warpSq[j][1];
        storeAsync(
            clusterAddr(sharedAddr(&hop.blockSq[j][slice]), dst), blockSum, clusterAddr(sharedAddr(&hop.normBar), dst));
    }
    mbarrierWaitPhase0(&hop.normBar);

    if (hasItem)
    {
        float fullSum = 0.F;
#pragma unroll
        for (int c = 0; c < kSlices; ++c)
        {
            fullSum += hop.blockSq[itemRow][c];
        }
        float const rcpRms = rsqrtf(fullSum / static_cast<float>(kHiddenDim) + params.eps);
        BfVec y;
        y.packed = resOut;
        BfVec g;
        g.packed = hop.gamma[itemVec];
        BfVec n;
#pragma unroll
        for (int e = 0; e < kEltsPerVec; ++e)
        {
            n.elts[e] = __float2bfloat16(__bfloat162float(y.elts[e]) * rcpRms * __bfloat162float(g.elts[e]));
        }
        if (writesOutputs)
        {
            *reinterpret_cast<uint4*>(reinterpret_cast<__nv_bfloat16*>(params.normOut) + rowOff) = n.packed;
            *reinterpret_cast<uint4*>(reinterpret_cast<__nv_bfloat16*>(params.residualOut) + rowOff) = y.packed;
        }
        if constexpr (kRouter)
        {
            rs.normed[itemRow][itemVec] = n.packed;
        }
    }

    if constexpr (kRouter)
    {
        if (numRowsCta > 0)
        {
            __syncthreads();
            int const tile = warp % Cfg::kExpertTiles;
            int const kGroup = warp / Cfg::kExpertTiles;
            if constexpr (kRouterLoad == kRouterBeforeMma)
            {
                loadRouterFragments<Cfg>(wLo, wHi, params.routerWeight, expertGroup, warp, gid, tig, sliceBase);
            }
            // Independent accumulator chains keep the MMA latency off a single dependency chain.
            float acc[kMmaChains][4] = {};
#pragma unroll
            for (int p = 0; p < Cfg::kPairsPerGroup; ++p)
            {
                // One 16-byte load covers two k16 steps; the k-permutation inside a step is the same for both operands.
                uint4 const b = rs.normed[gid][kGroup * (Cfg::kKPerGroup / kEltsPerVec) + p * 4 + tig];
                int const chain = (2 * p) % kMmaChains;
                mmaBf16(acc[chain], wLo[p].x, wHi[p].x, wLo[p].y, wHi[p].y, b.x, b.y);
                mmaBf16(acc[chain + 1], wLo[p].z, wHi[p].z, wLo[p].w, wHi[p].w, b.z, b.w);
            }
            float sum4[4];
#pragma unroll
            for (int i = 0; i < 4; ++i)
            {
                sum4[i] = (acc[0][i] + acc[1][i]) + (acc[2][i] + acc[3][i]);
            }
            // Expert e (cluster-local) is owned by CTA e / kExpertsPerOwner.
            int const e0 = tile * 16 + gid;
            int const e1 = e0 + 8;
            uint32_t const owner0 = e0 / Cfg::kExpertsPerOwner;
            uint32_t const owner1 = e1 / Cfg::kExpertsPerOwner;
            storeAsync(clusterAddr(sharedAddr(&rs.partial[slice][kGroup][e0 % Cfg::kExpertsPerOwner][tig * 2]), owner0),
                sum4[0], sum4[1], clusterAddr(sharedAddr(&hop.routerBar), owner0));
            storeAsync(clusterAddr(sharedAddr(&rs.partial[slice][kGroup][e1 % Cfg::kExpertsPerOwner][tig * 2]), owner1),
                sum4[2], sum4[3], clusterAddr(sharedAddr(&hop.routerBar), owner1));
            // Also keeps this CTA resident until every partial addressed to it has landed.
            mbarrierWaitPhase0(&hop.routerBar);

            // This CTA owns experts [slice * kExpertsPerOwner, (slice + 1) * kExpertsPerOwner) of the cluster. Lane
            // groups of 8 sum over the 8 slices.
            if (tid < Cfg::kExpertsPerOwner * numRowsCta * kSlices)
            {
                int const c = tid % kSlices;
                int const ij = tid / kSlices;
                int const ei = ij / numRowsCta;
                int const j = ij % numRowsCta;
                float sum = 0.F;
#pragma unroll
                for (int kg = 0; kg < Cfg::kKGroups; ++kg)
                {
                    sum += rs.partial[c][kg][ei][j];
                }
#pragma unroll
                for (int mask = kSlices / 2; mask > 0; mask >>= 1)
                {
                    sum = sum + __shfl_xor_sync(0xffffffffU, sum, mask, kSlices);
                }
                if (c == 0)
                {
                    int const r = rowGroup + j * kRowStride;
                    params.routerLogits[r * kNumExperts + expertGroup * Cfg::kExpertsPerCluster
                        + slice * Cfg::kExpertsPerOwner + ei]
                        = sum;
                }
            }
        }
    }

    // ---- CTA 0 advances the protocol state once every CTA has read it. ----
    if (tid == 0 && slice == 0 && clusterId == 0)
    {
        uint32_t arrived = 0;
        do
        {
            asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(arrived) : "l"(params.flags + 2) : "memory");
        } while (arrived < static_cast<uint32_t>(kGridCtas));
        uint4 const next = make_uint4((cur + 1) % kNumLamportBuffers,
            static_cast<uint32_t>(R * kNumRanks * kHiddenDim * sizeof(__nv_bfloat16)), 0U, 0U);
        asm volatile("st.relaxed.gpu.global.v4.u32 [%0], {%1, %2, %3, %4};" ::"l"(params.flags), "r"(next.x),
                     "r"(next.y), "r"(next.z), "r"(next.w)
                     : "memory");
    }
#else
    // Clusters, distributed shared memory and PDL need SM90+; the launcher is only reached on supported devices.
    __trap();
#endif
}

template <int R, int kG, int kRowSplit, int kRouterLoad = kRouterAfterSum>
cudaError_t launchTyped(PostAttentionHopFromPartialsParams const& params)
{
    using Cfg = HopConfig<kG, kRowSplit>;
    cudaLaunchAttribute attrs[2];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = kSlices;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;
    attrs[1].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[1].val.programmaticStreamSerializationAllowed = params.enablePdl ? 1 : 0;

    cudaLaunchConfig_t config{};
    config.gridDim = dim3(kSlices, kG > 0 ? kG * kRowSplit : R, 1);
    config.blockDim = dim3(Cfg::kThreads, 1, 1);
    config.dynamicSmemBytes = 0;
    config.stream = params.stream;
    config.attrs = attrs;
    config.numAttrs = 2;
    return cudaLaunchKernelEx(&config, postAttnHopX1Kernel<R, kG, kRowSplit, kRouterLoad>, params);
}

template <int kG, int kRowSplit>
cudaError_t launchRows(PostAttentionHopFromPartialsParams const& params)
{
    switch (params.numRows)
    {
    case 1: return launchTyped<1, kG, kRowSplit>(params);
    case 2: return launchTyped<2, kG, kRowSplit>(params);
    case 3: return launchTyped<3, kG, kRowSplit>(params);
    case 4: return launchTyped<4, kG, kRowSplit>(params);
    case 5: return launchTyped<5, kG, kRowSplit>(params);
    case 6: return launchTyped<6, kG, kRowSplit>(params);
    case 7: return launchTyped<7, kG, kRowSplit>(params);
    case 8: return launchTyped<8, kG, kRowSplit>(params);
    default: return cudaErrorInvalidValue;
    }
}

// Router hop: 4 expert groups x min(R, 3) row groups (at most 12 clusters of 8 CTAs, each CTA <= 3 row slices).
template <int kRouterLoad>
cudaError_t launchRouterAt(PostAttentionHopFromPartialsParams const& params)
{
    switch (params.numRows)
    {
    case 1: return launchTyped<1, 4, 1, kRouterLoad>(params);
    case 2: return launchTyped<2, 4, 2, kRouterLoad>(params);
    case 3: return launchTyped<3, 4, 3, kRouterLoad>(params);
    case 4: return launchTyped<4, 4, 3, kRouterLoad>(params);
    case 5: return launchTyped<5, 4, 3, kRouterLoad>(params);
    case 6: return launchTyped<6, 4, 3, kRouterLoad>(params);
    case 7: return launchTyped<7, 4, 3, kRouterLoad>(params);
    case 8: return launchTyped<8, 4, 3, kRouterLoad>(params);
    default: return cudaErrorInvalidValue;
    }
}

cudaError_t launchRouter(PostAttentionHopFromPartialsParams const& params)
{
    switch (params.routerLoad)
    {
    case kRouterAfterBarrier: return launchRouterAt<kRouterAfterBarrier>(params);
    case kRouterAfterSum: return launchRouterAt<kRouterAfterSum>(params);
    case kRouterBeforeMma: return launchRouterAt<kRouterBeforeMma>(params);
    default: return cudaErrorInvalidValue;
    }
}

} // namespace

cudaError_t invokePostAttentionHopFromPartials(PostAttentionHopFromPartialsParams const& params)
{
    if (params.rank < 0 || params.rank >= kNumRanks || params.flags == nullptr || params.ucBuffer == nullptr)
    {
        return cudaErrorInvalidValue;
    }
    if (params.routerWeight != nullptr)
    {
        if (params.routerLogits == nullptr)
        {
            return cudaErrorInvalidValue;
        }
        return launchRouter(params);
    }
    return launchRows<0, 1>(params);
}

} // namespace attention_output_into_hop

TRTLLM_MV2_NAMESPACE_END
