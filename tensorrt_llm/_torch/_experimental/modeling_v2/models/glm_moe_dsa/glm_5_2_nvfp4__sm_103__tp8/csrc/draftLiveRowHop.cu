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

#include "glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/csrc/draftLiveRowHop.h"
#include "tensorrt_llm/modeling_v2/common/namespace.h"

#include <cooperative_groups.h>
#include <cuda_bf16.h>

TRTLLM_MV2_NAMESPACE_BEGIN

namespace draft_live_rows
{
namespace
{
namespace cg = cooperative_groups;

// Work decomposition (the post-attention router hop's, on the compact rows).
//
// Every cluster has kSlices CTAs; CTA c owns slice [c * kSlice, (c + 1) * kSlice) of the hidden dimension (96 threads
// x 8 elements), the slice a CTA of the native one-shot kernel owns, so the RMSNorm reduction reproduces the native
// tree. There are kG x kRowSplit clusters (kG = 4 expert groups, kRowSplit = min(R, 3) row groups, R = live rows).
// Cluster (g, h) collects and normalizes the compact rows r with r % kRowSplit == h and computes their logits for
// experts [64 g, 64 g + 64). Its router weights live in registers (loaded before the PDL wait). Each CTA multiplies
// its rows of slice c by the experts' weights of slice c with BF16 MMAs (experts on m16, rows on n8) and pushes the
// partial logits to the CTA that owns the expert; owners add the 8 slices in a fixed order, so the logits are
// identical on every rank and do not depend on R or on the row's position. Clusters never synchronize with each
// other; inside a cluster, slice sums and partial logits travel as st.async stores that complete transactions on the
// consumer's mbarrier.
constexpr int kSlices = 8;
constexpr int kSlice = kHiddenDim / kSlices;
constexpr int kEltsPerVec = 8;
constexpr int kVecPerSlice = kSlice / kEltsPerVec;
constexpr int kNativeWarpsPerSlice = kVecPerSlice / 32;
// Row stride of the normalized-slice tile in 16-byte vectors; 100 = 96 + 4 keeps the 16-byte B-fragment loads of two
// rows in one quarter-warp phase on disjoint banks.
constexpr int kNormedStride = kVecPerSlice + 4;
constexpr int kMmaChains = 4;
constexpr int kExpertGroups = 4;
constexpr uint32_t kLamportSentinel = 0x80000000U;

static_assert(kHiddenDim % (kSlices * kEltsPerVec * 32) == 0);
static_assert(kVecPerSlice == kNativeWarpsPerSlice * 32);
static_assert(kMaxRows == 8, "the router MMA maps rows onto the n8 dimension");

struct HopConfig
{
    static constexpr int kThreads = 384;
    static constexpr int kWarps = kThreads / 32;
    static constexpr int kExpertsPerCluster = kNumExperts / kExpertGroups;
    static constexpr int kExpertTiles = kExpertsPerCluster / 16;
    static constexpr int kKGroups = kWarps / kExpertTiles;
    static constexpr int kKPerGroup = kSlice / kKGroups;
    static constexpr int kPairsPerGroup = kKPerGroup / 32;
    static constexpr int kExpertsPerOwner = kExpertsPerCluster / kSlices;
    static_assert(kExpertsPerCluster % 16 == 0 && kWarps % kExpertTiles == 0 && kSlice % (kKGroups * 32) == 0);
    static_assert(kPairsPerGroup % (kMmaChains / 2) == 0);
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

struct alignas(16) RouterStorage
{
    uint4 normed[kMaxRows][kNormedStride]; // local row on the MMA n dimension; unused rows stay zero
    float partial[kSlices][HopConfig::kKGroups][HopConfig::kExpertsPerOwner][kMaxRows];
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

__device__ __forceinline__ bool isDirty(uint4 const& v)
{
    return v.x == kLamportSentinel || v.y == kLamportSentinel || v.z == kLamportSentinel || v.w == kLamportSentinel;
}

// Lamport payloads must never contain the sentinel word: map bf16 -0.0 to +0.0 (the reduction starts from +0.0, so
// the sum is unchanged).
__device__ __forceinline__ uint4 sanitize(uint4 v)
{
    BfVec b;
    b.packed = v;
#pragma unroll
    for (int i = 0; i < kEltsPerVec; ++i)
    {
        if (__bfloat16_as_ushort(b.elts[i]) == 0x8000U)
        {
            b.elts[i] = __ushort_as_bfloat16(0);
        }
    }
    return b.packed;
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
        "LIVE_ROW_MBAR_WAIT_%=:\n"
        "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 done, [%0], 0;\n"
        "@!done bra LIVE_ROW_MBAR_WAIT_%=;\n"
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

// R = live (compact) rows; kRowSplit = row groups.
template <int R, int kRowSplit>
__global__ void __launch_bounds__(HopConfig::kThreads, 1) liveRowHopKernel(LiveRowHopParams const params)
{
    static_assert(R >= 1 && R <= kMaxRows);
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    using Cfg = HopConfig;
    constexpr int kClusters = kExpertGroups * kRowSplit;
    constexpr int kGridCtas = kClusters * kSlices;
    // Rows one CTA collects, normalizes and multiplies.
    constexpr int kRowsPerCta = (R + kRowSplit - 1) / kRowSplit;
    static_assert(kRowsPerCta <= kMaxRowsPerCtaAll);
    static_assert(kRowsPerCta * kVecPerSlice <= Cfg::kThreads && kRowsPerCta * kNativeWarpsPerSlice <= Cfg::kWarps);
    static_assert(Cfg::kExpertsPerOwner * kRowsPerCta * kSlices <= Cfg::kThreads);

    __shared__ HopStorage hop;
    __shared__ RouterStorage rs;

    cg::cluster_group cluster = cg::this_cluster();
    int const slice = static_cast<int>(cluster.block_rank());
    int const clusterId = blockIdx.y;
    int const expertGroup = clusterId % kExpertGroups;
    int const rowGroup = clusterId / kExpertGroups;
    int const tid = threadIdx.x;
    int const lane = tid & 31;
    int const warp = tid >> 5;
    int const sliceBase = slice * kSlice;

    // Compact rows this CTA collects and normalizes: local row j is compact row rowGroup + j * kRowSplit.
    int const numRowsCta = (R - rowGroup + kRowSplit - 1) / kRowSplit;
    // This thread's item: one 16-byte vector of one row slice.
    int const itemRow = tid / kVecPerSlice;
    int const itemVec = tid % kVecPerSlice;
    bool const hasItem = itemRow < numRowsCta;
    int const row = rowGroup + itemRow * kRowSplit;
    // The rank's own partial of a row slice is published by exactly one CTA of the grid.
    bool const sends = hasItem && itemRow % kExpertGroups == expertGroup;
    size_t const colOff = static_cast<size_t>(sliceBase) + itemVec * kEltsPerVec;
    size_t const rowOff = static_cast<size_t>(row) * kHiddenDim + colOff;
    // The cluster that writes the normalized rows and the residual.
    bool const writesOutputs = expertGroup == 0;

    // ---- Prologue: constant model weights only; may overlap the producer of `input` under PDL. ----
    if (tid < kVecPerSlice)
    {
        hop.gamma[tid] = loadConstant(reinterpret_cast<uint4 const*>(params.gamma) + slice * kVecPerSlice + tid);
    }
    int const gid = lane >> 2;
    int const tig = lane & 3;
    uint4 wLo[Cfg::kPairsPerGroup];
    uint4 wHi[Cfg::kPairsPerGroup];
    {
        int const tile = warp % Cfg::kExpertTiles;
        int const kGroup = warp / Cfg::kExpertTiles;
        auto const* w = reinterpret_cast<__nv_bfloat16 const*>(params.routerWeight);
        int const expert = expertGroup * Cfg::kExpertsPerCluster + tile * 16 + gid;
#pragma unroll
        for (int p = 0; p < Cfg::kPairsPerGroup; ++p)
        {
            int const k = sliceBase + kGroup * Cfg::kKPerGroup + p * 32 + tig * kEltsPerVec;
            wLo[p] = loadConstant(w + static_cast<size_t>(expert) * kHiddenDim + k);
            wHi[p] = loadConstant(w + static_cast<size_t>(expert + 8) * kHiddenDim + k);
        }
        // MMA columns of rows this CTA does not own stay zero.
        for (int i = tid; i < kMaxRows * kNormedStride; i += Cfg::kThreads)
        {
            rs.normed[i / kNormedStride][i % kNormedStride] = make_uint4(0, 0, 0, 0);
        }
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

    // ---- Wait for the producer of `input` (and transitively of `residualIn`, `rowIds` and the protocol state). ----
    asm volatile("griddepcontrol.wait;" ::: "memory");
    // Dependents only start their own prologue; they still wait for this grid to complete.
    asm volatile("griddepcontrol.launch_dependents;");

    // The live row's source row (written by an earlier kernel of the stream: a plain load after the wait).
    int64_t srcRow = 0;
    if (hasItem)
    {
        srcRow = params.rowIds[row];
    }
    uint4 const flags = loadVolatile(params.flags);
    srcRow = srcRow < 0 ? 0 : (srcRow >= params.numInputRows ? params.numInputRows - 1 : srcRow);
    size_t const srcOff = static_cast<size_t>(srcRow) * kHiddenDim + colOff;
    auto const* input = reinterpret_cast<__nv_bfloat16 const*>(params.input);
    auto const* residualIn = reinterpret_cast<__nv_bfloat16 const*>(params.residualIn);
    uint4 res = make_uint4(0, 0, 0, 0);
    // This rank's own partial, as published (sanitized); it is used from registers, not from the multicast copy.
    uint4 own = make_uint4(0, 0, 0, 0);
    if (hasItem)
    {
        res = *reinterpret_cast<uint4 const*>(residualIn + srcOff);
        own = sanitize(*reinterpret_cast<uint4 const*>(input + srcOff));
    }
    uint32_t const cur = flags.x;
    uint32_t const clearBytes = flags.y;
    char* ucBase = reinterpret_cast<char*>(params.ucBuffer);
    auto* curUc = reinterpret_cast<__nv_bfloat16*>(ucBase + cur * kLamportBufferBytes);
    auto* curMc
        = reinterpret_cast<__nv_bfloat16*>(reinterpret_cast<char*>(params.mcBuffer) + cur * kLamportBufferBytes);
    // Lamport layout of one buffer: [compact row][source rank][kHiddenDim].
    size_t const slotBase = static_cast<size_t>(row) * kNumRanks * kHiddenDim + colOff;

    // ---- Publish this rank's partial to every rank (one multicast store per vector). ----
    if (sends)
    {
        *reinterpret_cast<uint4*>(curMc + slotBase + static_cast<size_t>(params.rank) * kHiddenDim) = own;
    }

    // ---- Collect the other ranks' partials. ----
    uint4 vals[kNumRanks];
    if (hasItem)
    {
#pragma unroll
        for (int s = 0; s < kNumRanks; ++s)
        {
            vals[s] = s == params.rank ? own : loadVolatile(curUc + slotBase + static_cast<size_t>(s) * kHiddenDim);
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
    __syncthreads();
    if (tid == 0)
    {
        // Every thread of this CTA has read the protocol state; CTA 0 advances it only after all CTAs report.
        asm volatile("red.release.gpu.global.add.u32 [%0], %1;" ::"l"(params.flags + 2), "r"(1U) : "memory");
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

        // This CTA owns experts [slice * kExpertsPerOwner, (slice + 1) * kExpertsPerOwner) of the cluster. Lane groups
        // of 8 sum over the 8 slices.
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
                int const r = rowGroup + j * kRowSplit;
                params.routerLogits[r * kNumExperts + expertGroup * Cfg::kExpertsPerCluster
                    + slice * Cfg::kExpertsPerOwner + ei]
                    = sum;
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
        // The next call clears exactly the slots this call wrote: compact rows 0..R-1 of every rank.
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

template <int R, int kRowSplit>
cudaError_t launchTyped(LiveRowHopParams const& params)
{
    cudaLaunchAttribute attrs[2];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = kSlices;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;
    attrs[1].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[1].val.programmaticStreamSerializationAllowed = params.enablePdl ? 1 : 0;

    cudaLaunchConfig_t config{};
    config.gridDim = dim3(kSlices, kExpertGroups * kRowSplit, 1);
    config.blockDim = dim3(HopConfig::kThreads, 1, 1);
    config.dynamicSmemBytes = 0;
    config.stream = params.stream;
    config.attrs = attrs;
    config.numAttrs = 2;
    return cudaLaunchKernelEx(&config, liveRowHopKernel<R, kRowSplit>, params);
}

} // namespace

cudaError_t invokeDraftLiveRowHop(LiveRowHopParams const& params)
{
    if (params.rank < 0 || params.rank >= kNumRanks || params.flags == nullptr || params.ucBuffer == nullptr
        || params.mcBuffer == nullptr || params.rowIds == nullptr || params.routerWeight == nullptr
        || params.routerLogits == nullptr || params.numInputRows < 1 || params.numInputRows > kMaxRows
        || params.numRows < 1 || params.numRows > params.numInputRows)
    {
        return cudaErrorInvalidValue;
    }
    // 4 expert groups x min(R, 3) row groups (at most 12 clusters of 8 CTAs, each CTA <= 3 row slices).
    switch (params.numRows)
    {
    case 1: return launchTyped<1, 1>(params);
    case 2: return launchTyped<2, 2>(params);
    case 3: return launchTyped<3, 3>(params);
    case 4: return launchTyped<4, 3>(params);
    case 5: return launchTyped<5, 3>(params);
    case 6: return launchTyped<6, 3>(params);
    case 7: return launchTyped<7, 3>(params);
    case 8: return launchTyped<8, 3>(params);
    default: return cudaErrorInvalidValue;
    }
}

} // namespace draft_live_rows

TRTLLM_MV2_NAMESPACE_END
