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

#include "glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/csrc/postAttentionHopRouting.h"
#include "tensorrt_llm/modeling_v2/common/namespace.h"

#include <cooperative_groups.h>
#include <cuda_bf16.h>

TRTLLM_MV2_NAMESPACE_BEGIN

namespace hop_routing
{
namespace
{
namespace cg = cooperative_groups;

// The partials hop (postAttentionHopFromPartials.cu, router variant) and top-8 routing. Everything up to and
// including the router-logit stores is the partials hop's code; the additions are marked "Routing".
//
// Work decomposition (the partials hop's). Every cluster has kSlices CTAs; CTA c owns hidden slice
// [c * kSlice, (c + 1) * kSlice). There are kG x kRowSplit clusters (kG = 4, kRowSplit = min(R, 3)). Cluster (g, h)
// collects and normalizes the rows r with r % kRowSplit == h and computes their logits for experts [64 g, 64 g + 64);
// CTA c of the cluster owns the 8 experts 64 g + 8 c + [0, 8) and writes their logits.
//
// Routing. The owner of logit (r, e) also writes the score bits of (r, e) into the gather buffer. Row
// r = h + j * kRowSplit (local row j of row group h) is finished by CTA 0 of cluster (g = j, h): its 256 threads poll
// the row's 256 scores, each warp ranks its 32 keys and keeps its top 8 (level 1), the 256 threads rank the 64
// candidates (level 2, four lanes per candidate), and lanes 0..7 write the row's ids and weights.
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
// The empty gather slot: a routing score is never -0.0f (see postAttentionHopRouting.h).
constexpr uint32_t kGatherSentinel = 0x80000000U;
constexpr int kG = 4;
constexpr int kExpertWarps = kNumExperts / 32;
constexpr int kCandidates = kExpertWarps * kTopK;

static_assert(kHiddenDim % (kSlices * kEltsPerVec * 32) == 0);
static_assert(kVecPerSlice == kNativeWarpsPerSlice * 32);
static_assert(kMaxRows == 8, "the router MMA maps rows onto the n8 dimension");
static_assert(kNumExperts == 256 && kTopK == 8, "the finisher maps one expert per thread and 8 candidates per warp");
static_assert(kCandidates * 4 == kNumExperts, "level 2 ranks each candidate on four lanes");

template <int kRowSplit>
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
    static_assert(kThreads >= kNumExperts);
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

template <int kRowSplit>
struct alignas(16) RouterStorage
{
    using Cfg = HopConfig<kRowSplit>;
    uint4 normed[kMaxRows][kNormedStride]; // local row on the MMA n dimension; unused rows stay zero
    float partial[kSlices][Cfg::kKGroups][Cfg::kExpertsPerOwner][kMaxRows];
};

// Routing: state of one CTA.
struct alignas(16) RouteStorage
{
    unsigned long long candKey[kCandidates]; // level-1 survivors: native 64-bit keys
    float candSg[kCandidates];               // their score - bias
    unsigned long long selKey[kTopK];        // level-2 result in rank order
    float selSg[kTopK];
    float biasOwn[8];                        // bias of the experts this CTA owns
    uint32_t parity;                         // gather parity of this call
    float biasAll[kNumExperts];              // finisher CTAs: bias of every expert
#if defined(TLLM_GLM_MOE_DSA_HOP_ROUTING_STAMPS)
    unsigned long long stamp[kNumStamps];
#endif
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

__device__ __forceinline__ uint32_t loadVolatileU32(void const* ptr)
{
    uint32_t v;
    asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(v) : "l"(ptr) : "memory");
    return v;
}

__device__ __forceinline__ uint4 loadConstant(void const* ptr)
{
    uint4 v;
    asm volatile("ld.global.nc.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(ptr));
    return v;
}

__device__ __forceinline__ float loadBiasConstant(void const* bias, int expert)
{
    unsigned short raw;
    asm volatile("ld.global.nc.u16 %0, [%1];" : "=h"(raw) : "l"(reinterpret_cast<__nv_bfloat16 const*>(bias) + expert));
    return __bfloat162float(__ushort_as_bfloat16(raw));
}

__device__ __forceinline__ void prefetchL2(void const* ptr)
{
    asm volatile("prefetch.global.L2 [%0];" ::"l"(ptr));
}

__device__ __forceinline__ void storeRelaxed(uint32_t* ptr, uint32_t v)
{
    asm volatile("st.relaxed.gpu.global.u32 [%0], %1;" ::"l"(ptr), "r"(v) : "memory");
}

__device__ __forceinline__ uint32_t loadRelaxed(uint32_t const* ptr)
{
    uint32_t v;
    asm volatile("ld.relaxed.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(ptr) : "memory");
    return v;
}

__device__ __forceinline__ bool isDirty(uint4 const& v)
{
    return v.x == kLamportSentinel || v.y == kLamportSentinel || v.z == kLamportSentinel || v.w == kLamportSentinel;
}

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
        "ROUTING_MBAR_WAIT_%=:\n"
        "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 done, [%0], 0;\n"
        "@!done bra ROUTING_MBAR_WAIT_%=;\n"
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

// One warp's router-weight MMA fragments (the post-attention hop's register layout): experts tile * 16 + gid (+ 8)
// of this cluster's expert group, k pairs of this warp's k group within the CTA's slice.
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

// Router-weight fragment load points, as in the partials hop (all orders give the same values and arithmetic).
constexpr int kRouterAfterBarrier = 0;
constexpr int kRouterAfterSum = 1;
constexpr int kRouterBeforeMma = 2;

// Routing: the native routing score (RoutingKernel.cuh sigmoid_accurate plus the SigmoidBias preprocess).
__device__ __forceinline__ float routingScore(float logit, float bias)
{
    float const s = 0.5f * tanhf(0.5f * logit) + 0.5f;
    return s + bias;
}

// Routing: cub::Traits<float>::TwiddleIn, the order-preserving map of FP32 bits onto unsigned integers.
__device__ __forceinline__ uint32_t twiddleIn(uint32_t bits)
{
    return bits ^ ((bits & 0x80000000U) != 0U ? 0xFFFFFFFFU : 0x80000000U);
}

#if defined(TLLM_GLM_MOE_DSA_HOP_ROUTING_STAMPS)
__device__ __forceinline__ unsigned long long globalTimer()
{
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

#define ROUTING_STAMP(field) atomicMax(&route.stamp[field], globalTimer())
#else
#define ROUTING_STAMP(field)                                                                                           \
    do                                                                                                                 \
    {                                                                                                                  \
    } while (0)
#endif

// Stamp fields (diagnostic builds).
constexpr int kStampEntry = 0;
constexpr int kStampWait = 1;
constexpr int kStampLogits = 2;
constexpr int kStampGather = 3;
constexpr int kStampPolled = 4;
constexpr int kStampLevel1 = 5;
constexpr int kStampRouted = 6;
constexpr int kStampEnd = 7;
static_assert(kStampEnd < kNumStamps);

// The register cap of the partials hop, under which one CTA of it fits on an SM next to a CTA of the publishing
// o_proj. The routing code after the MMA phase would raise the allocation above it; the cap keeps the partials hop's
// allocation without spills.
constexpr int kMaxRegisters = 112;

template <int R, int kRowSplit, int kRouterLoad, bool kLogitsIn>
__global__ void __maxnreg__(kMaxRegisters) postAttnHopZ2Kernel(PostAttentionHopRoutingParams const params)
{
    static_assert(R >= 1 && R <= kMaxRows);
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    using Cfg = HopConfig<kRowSplit>;
    constexpr int kClusters = kG * kRowSplit;
    constexpr int kGridCtas = kClusters * kSlices;
    static_assert(kGridCtas <= kMaxGridCtas);
    // Rows one CTA collects, normalizes and multiplies.
    constexpr int kRowsPerCta = (R + kRowSplit - 1) / kRowSplit;
    static_assert(kRowsPerCta <= kMaxRowsPerCtaAll && kRowsPerCta <= kG);
    static_assert(kRowsPerCta * kVecPerSlice <= Cfg::kThreads && kRowsPerCta * kNativeWarpsPerSlice <= Cfg::kWarps);
    static_assert(Cfg::kExpertsPerOwner * kRowsPerCta * kSlices <= Cfg::kThreads);

    __shared__ HopStorage hop;
    __shared__ RouterStorage<kRowSplit> rs;
    __shared__ RouteStorage route;

    cg::cluster_group cluster = cg::this_cluster();
    int const slice = static_cast<int>(cluster.block_rank());
    int const clusterId = blockIdx.y;
    int const expertGroup = clusterId % kG;
    int const rowGroup = clusterId / kG;
    int const tid = threadIdx.x;
    int const lane = tid & 31;
    int const warp = tid >> 5;
    int const sliceBase = slice * kSlice;
    uint32_t const ctaLinear = clusterId * kSlices + slice;

    // Rows this CTA collects and normalizes: local row j is global row rowGroup + j * kRowSplit.
    constexpr int kRowStride = kRowSplit;
    int const numRowsCta = (R - rowGroup + kRowSplit - 1) / kRowSplit;
    // This thread's item: one 16-byte vector of one row slice.
    int const itemRow = tid / kVecPerSlice;
    int const itemVec = tid % kVecPerSlice;
    bool const hasItem = itemRow < numRowsCta;
    int const row = rowGroup + itemRow * kRowStride;
    size_t const rowOff = static_cast<size_t>(row) * kHiddenDim + sliceBase + itemVec * kEltsPerVec;
    // The cluster that writes the normalized rows and the residual.
    bool const writesOutputs = expertGroup == 0;
    // Routing: this CTA finishes local row `expertGroup` of its row group, if that row exists.
    bool const finisher = slice == 0 && expertGroup < numRowsCta;

#if defined(TLLM_GLM_MOE_DSA_HOP_ROUTING_STAMPS)
    if (tid < kNumStamps)
    {
        route.stamp[tid] = 0ULL;
    }
    __syncthreads();
    if (tid == 0)
    {
        ROUTING_STAMP(kStampEntry);
    }
#endif

    // ---- Prologue: constant model weights only; may overlap the producer of `input` under PDL. ----
    if (tid < kVecPerSlice)
    {
        hop.gamma[tid] = loadConstant(reinterpret_cast<uint4 const*>(params.gamma) + slice * kVecPerSlice + tid);
    }
    // Routing: router bias of the owned experts and, on finishers, of every expert (a model parameter).
    if (tid < Cfg::kExpertsPerOwner)
    {
        route.biasOwn[tid] = loadBiasConstant(
            params.routingBias, expertGroup * Cfg::kExpertsPerCluster + slice * Cfg::kExpertsPerOwner + tid);
    }
    if (finisher && tid < kNumExperts)
    {
        route.biasAll[tid] = loadBiasConstant(params.routingBias, tid);
    }
    int const gid = lane >> 2;
    int const tig = lane & 3;
    // The router-weight fragments are loaded after the poll (not here), so they are not live across it; the prologue
    // only asks L2 for the same lines (no registers held, no architectural effect).
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

    // Cluster-local completion barriers: every CTA of a cluster has the same rows, so each expects the 8 slice sums
    // of its rows and the 8 x kKGroups partial tiles of the experts it owns. A split cluster barrier publishes the
    // initialization; its wait precedes the first remote store, long after all CTAs arrived.
    if (tid == 0)
    {
        mbarrierInitExpectTx(&hop.normBar, static_cast<uint32_t>(numRowsCta * kSlices * sizeof(float)));
        mbarrierInitExpectTx(&hop.routerBar, numRowsCta > 0 ? static_cast<uint32_t>(sizeof(rs.partial)) : 0U);
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");

    // With earlyPoll no wait here (the protocol state and the transport may be read early, see
    // postAttentionHopFromPartials.h).
    auto const* residualIn = reinterpret_cast<__nv_bfloat16 const*>(params.residualIn);
    uint4 res = make_uint4(0, 0, 0, 0);
    if (!params.earlyPoll)
    {
        asm volatile("griddepcontrol.wait;" ::: "memory");
        // Dependents may start their prologue; they still wait for this grid to complete.
        asm volatile("griddepcontrol.launch_dependents;");
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

    // ---- Collect every rank's partial, this rank's own included, from the transport. ----
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

    // ---- earlyPoll: wait for the producer grid before reading the residual or writing any output. ----
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
#if defined(TLLM_GLM_MOE_DSA_HOP_ROUTING_STAMPS)
    if (tid == 0)
    {
        ROUTING_STAMP(kStampWait);
    }
#endif

    // Router-weight fragments (see kRouterLoad).
    uint4 wLo[Cfg::kPairsPerGroup];
    uint4 wHi[Cfg::kPairsPerGroup];

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
    if constexpr (kRouterLoad == kRouterAfterSum)
    {
        if (numRowsCta > 0)
        {
            loadRouterFragments<Cfg>(wLo, wHi, params.routerWeight, expertGroup, warp, gid, tig, sliceBase);
        }
    }
    if (tid == 0)
    {
        // Routing: the gather parity of this call (the routing state is read only after the dependency wait). It is
        // kept in shared memory, published by the barrier below, so no register holds it across the MMA phase.
        route.parity = loadVolatileU32(params.zstate) & 1U;
    }
    __syncthreads();
    if (tid == 0)
    {
        // Every thread of this CTA has read the protocol state and thread 0 the gather parity; CTA 0 advances both only
        // after all CTAs report.
        asm volatile("red.release.gpu.global.add.u32 [%0], %1;" ::"l"(params.flags + 2), "r"(1U) : "memory");
    }
    // ---- Routing: reset the previous call's gather buffer (its readers belong to a completed grid), off the routing
    // path. ----
    {
        uint4* const other = reinterpret_cast<uint4*>(params.gather + (route.parity ^ 1U) * kGatherBufferWords);
        uint4 const sentinel = make_uint4(kGatherSentinel, kGatherSentinel, kGatherSentinel, kGatherSentinel);
        for (uint32_t j = ctaLinear * Cfg::kThreads + tid; j < kGatherBufferWords / 4; j += kGridCtas * Cfg::kThreads)
        {
            other[j] = sentinel;
        }
    }
    if constexpr (kRouterLoad == kRouterAfterBarrier)
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
        rs.normed[itemRow][itemVec] = n.packed;
    }

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
                int const e = expertGroup * Cfg::kExpertsPerCluster + slice * Cfg::kExpertsPerOwner + ei;
                params.routerLogits[r * kNumExperts + e] = sum;
                ROUTING_STAMP(kStampLogits);
                // Routing: the score of (r, e) into this call's gather buffer.
                float const logit = kLogitsIn ? params.logitsIn[r * kNumExperts + e] : sum;
                storeRelaxed(params.gather + route.parity * kGatherBufferWords + r * kNumExperts + e,
                    __float_as_uint(routingScore(logit, route.biasOwn[ei])));
                ROUTING_STAMP(kStampGather);
            }
        }
    }

    // ---- Routing: finisher CTAs select the top 8 of their row. ----
    if (finisher)
    {
        int const rf = rowGroup + expertGroup * kRowStride;
        if (tid < kNumExperts)
        {
            uint32_t const* slot = params.gather + route.parity * kGatherBufferWords + rf * kNumExperts + tid;
            uint32_t bits = loadRelaxed(slot);
            while (bits == kGatherSentinel)
            {
                bits = loadRelaxed(slot);
            }
#if defined(TLLM_GLM_MOE_DSA_HOP_ROUTING_STAMPS)
            __syncwarp();
            if (lane == 0)
            {
                ROUTING_STAMP(kStampPolled);
            }
#endif
            float const sg = __uint_as_float(bits) - route.biasAll[tid];
            uint32_t const hi = twiddleIn(bits);
            // Level 1: rank of this key among the warp's 32 (expert 32 * warp + i sits in lane i, so equal scores
            // rank the lower lane first, as the native key does).
            int rank = 0;
#pragma unroll
            for (int i = 0; i < 32; ++i)
            {
                uint32_t const other = __shfl_sync(0xffffffffU, hi, i);
                rank += (other > hi || (other == hi && i < lane)) ? 1 : 0;
            }
            if (rank < kTopK)
            {
                route.candKey[warp * kTopK + rank]
                    = (static_cast<unsigned long long>(hi) << 32) | static_cast<unsigned long long>(65535 - tid);
                route.candSg[warp * kTopK + rank] = sg;
            }
        }
        __syncthreads();
#if defined(TLLM_GLM_MOE_DSA_HOP_ROUTING_STAMPS)
        if (tid == 0)
        {
            ROUTING_STAMP(kStampLevel1);
        }
#endif
        if (tid < kNumExperts)
        {
            // Level 2: rank of the 64 candidates by the full native key (all keys are distinct). Four lanes share
            // candidate tid / 4; lane q of them counts the larger keys among candidates q, q + 4, ..., q + 60.
            int const ci = tid >> 2;
            int const q = tid & 3;
            unsigned long long const key = route.candKey[ci];
            int rank = 0;
#pragma unroll
            for (int i = 0; i < kCandidates / 4; ++i)
            {
                rank += route.candKey[i * 4 + q] > key ? 1 : 0;
            }
            rank += __shfl_xor_sync(0xffffffffU, rank, 1);
            rank += __shfl_xor_sync(0xffffffffU, rank, 2);
            if (q == 0 && rank < kTopK)
            {
                route.selKey[rank] = key;
                route.selSg[rank] = route.candSg[ci];
            }
        }
        __syncthreads();
        if (warp == 0)
        {
            if (lane < kTopK)
            {
                float s[kTopK];
#pragma unroll
                for (int k = 0; k < kTopK; ++k)
                {
                    s[k] = route.selSg[k];
                }
                // The native warp sum over lanes 0..7 (zeros above): butterfly offsets 16, 8, 4, 2, 1.
                float const total = ((s[0] + s[4]) + (s[2] + s[6])) + ((s[1] + s[5]) + (s[3] + s[7]));
                float const weight = route.selSg[lane] * params.routedScaling / total;
                int const expert = 65535 - static_cast<int>(route.selKey[lane] & 0xFFFFULL);
                params.topkIds[rf * kTopK + lane] = expert;
                params.topkWeights[rf * kTopK + lane] = __bfloat162float(__float2bfloat16(weight));
                ROUTING_STAMP(kStampRouted);
            }
        }
    }

    // ---- CTA 0 advances the protocol state and the gather parity once every CTA has read them. ----
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
        storeRelaxed(params.zstate, route.parity ^ 1U);
    }
#if defined(TLLM_GLM_MOE_DSA_HOP_ROUTING_STAMPS)
    __syncthreads();
    if (tid == 0)
    {
        ROUTING_STAMP(kStampEnd);
    }
    __syncthreads();
    if (tid < kNumStamps && params.stamps != nullptr)
    {
        params.stamps[ctaLinear * kNumStamps + tid] = route.stamp[tid];
    }
#endif
#else
    // Clusters, distributed shared memory and PDL need SM90+; the launcher is only reached on supported devices.
    __trap();
#endif
}

template <int R, int kRowSplit, int kRouterLoad, bool kLogitsIn>
cudaError_t launchTyped(PostAttentionHopRoutingParams const& params)
{
    using Cfg = HopConfig<kRowSplit>;
    cudaLaunchAttribute attrs[2];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = kSlices;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;
    attrs[1].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[1].val.programmaticStreamSerializationAllowed = params.enablePdl ? 1 : 0;

    cudaLaunchConfig_t config{};
    config.gridDim = dim3(kSlices, kG * kRowSplit, 1);
    config.blockDim = dim3(Cfg::kThreads, 1, 1);
    config.dynamicSmemBytes = 0;
    config.stream = params.stream;
    config.attrs = attrs;
    config.numAttrs = 2;
    return cudaLaunchKernelEx(&config, postAttnHopZ2Kernel<R, kRowSplit, kRouterLoad, kLogitsIn>, params);
}

// 4 expert groups x min(R, 3) row groups (at most 12 clusters of 8 CTAs, each CTA <= 3 row slices).
template <int kRouterLoad, bool kLogitsIn>
cudaError_t launchAt(PostAttentionHopRoutingParams const& params)
{
    switch (params.numRows)
    {
    case 1: return launchTyped<1, 1, kRouterLoad, kLogitsIn>(params);
    case 2: return launchTyped<2, 2, kRouterLoad, kLogitsIn>(params);
    case 3: return launchTyped<3, 3, kRouterLoad, kLogitsIn>(params);
    case 4: return launchTyped<4, 3, kRouterLoad, kLogitsIn>(params);
    case 5: return launchTyped<5, 3, kRouterLoad, kLogitsIn>(params);
    case 6: return launchTyped<6, 3, kRouterLoad, kLogitsIn>(params);
    case 7: return launchTyped<7, 3, kRouterLoad, kLogitsIn>(params);
    case 8: return launchTyped<8, 3, kRouterLoad, kLogitsIn>(params);
    default: return cudaErrorInvalidValue;
    }
}

} // namespace

bool stampsCompiled()
{
#if defined(TLLM_GLM_MOE_DSA_HOP_ROUTING_STAMPS)
    return true;
#else
    return false;
#endif
}

cudaError_t invokePostAttentionHopRouting(PostAttentionHopRoutingParams const& params)
{
    if (params.rank < 0 || params.rank >= kNumRanks || params.flags == nullptr || params.ucBuffer == nullptr
        || params.routerWeight == nullptr || params.routingBias == nullptr || params.routerLogits == nullptr
        || params.topkIds == nullptr || params.topkWeights == nullptr || params.gather == nullptr
        || params.zstate == nullptr)
    {
        return cudaErrorInvalidValue;
    }
    if (params.logitsIn != nullptr)
    {
        // Test instance: the default router-weight load point only.
        return params.routerLoad == kRouterAfterSum ? launchAt<kRouterAfterSum, true>(params) : cudaErrorInvalidValue;
    }
    switch (params.routerLoad)
    {
    case kRouterAfterBarrier: return launchAt<kRouterAfterBarrier, false>(params);
    case kRouterAfterSum: return launchAt<kRouterAfterSum, false>(params);
    case kRouterBeforeMma: return launchAt<kRouterBeforeMma, false>(params);
    default: return cudaErrorInvalidValue;
    }
}

} // namespace hop_routing

TRTLLM_MV2_NAMESPACE_END
