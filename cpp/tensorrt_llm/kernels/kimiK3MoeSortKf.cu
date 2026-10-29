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

// The kernel comes from an automated kernel search against a PyTorch reference of the moe_sort layout contract; it is
// validated on Rubin against that reference and against moe_sort's own outputs
// (tests/unittest/_torch/thop/parallel/test_k3_moe_sort_kf.py).
// Here it gains PDL (wait before the first global load, trigger after the last store), the arch/shape guards and the
// restyling to the repository conventions.
//
// Contract (same as trtllm::moe_sort for top_k 16, 56 local experts, tile 128): ids [T, 16] int32, T a multiple of 16,
// the entries of local experts [offset, offset + 56) are grouped by expert, every expert padded to a multiple of 128
// rows; the rest of expanded_idx_to_permuted_idx is -1; tile tables, total_num_padded_tokens and num_non_exiting_tiles
// as in moe_sort. Padded rows of permuted_idx_to_expanded_idx and tile entries past num_non_exiting_tiles are not
// written.
//
// MoE routing sort for one expert-parallel rank (Kimi K3 DEP16, 56 local experts, 128-row tiles).
//
// One cluster of C <= 8 CTAs; CTA c owns a contiguous slice of the [T, 16] id buffer viewed as int4 vectors.
//   1. Load the ids (16 B, L2 only). Every local entry takes a rank from its warp's private counter of its expert (no
//      inter-warp race: the ranks are deterministic) and bumps the CTA counter; the first local entry of a vector is
//      the fast path, further ones (rare) take a branch. Vectors without a local entry get their -1 fill right away.
//   2. The CTA counts go to every CTA of the cluster with st.async (completion on the receiver's mbarrier); warps 1-2
//      turn the per-warp counters into per-warp offsets (one lane per expert) while warp 0 builds the tile layout and
//      the CTA's base row of every expert.
//   3. row = base[e] + warp offset[w][e] + rank: e2p vector store + p2e scatter; the tile tables are spread over the
//      CTAs.
// Permuted order inside an expert: (CTA, warp, intra-warp atomic order): deterministic, not stable.

#include "tensorrt_llm/common/config.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/envUtils.h"
#include "tensorrt_llm/kernels/kimiK3MoeSortKf.h"

#include <cstdint>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

namespace
{

constexpr int kTopK = 16;
[[maybe_unused]] constexpr int kNumLocal = 56;
[[maybe_unused]] constexpr int kCountStride = 57; // row stride of the per-warp counters (conflict-free column reads)
[[maybe_unused]] constexpr int kTile = 128;
constexpr int kMaxClusterCtas = 8;
[[maybe_unused]] constexpr int kMaxWarps = 32;
constexpr int kMinVecPerCta = 256; // int4 vectors (1024 ids) a CTA is worth
constexpr int kMaxVectorsPerThread = 4;

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000) && (__CUDA_ARCH__ < 1100)

__device__ __forceinline__ void pdlWait()
{
    asm volatile("griddepcontrol.wait;" ::: "memory");
}

__device__ __forceinline__ void pdlTrigger()
{
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
}

__device__ __forceinline__ int4 ldGlobalCg(int4 const* p)
{
    int4 v;
    asm volatile("ld.global.cg.v4.s32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
    return v;
}

__device__ __forceinline__ uint32_t clusterCtaRank()
{
    uint32_t r;
    asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
    return r;
}

__device__ __forceinline__ uint32_t clusterNumCtas()
{
    uint32_t n;
    asm volatile("mov.u32 %0, %%cluster_nctarank;" : "=r"(n));
    return n;
}

__device__ __forceinline__ void clusterArriveRelease()
{
    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
}

__device__ __forceinline__ void clusterWait()
{
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
}

__device__ __forceinline__ uint32_t smemAddr(void const* p)
{
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ uint32_t mapaShared(uint32_t addr, uint32_t cta)
{
    uint32_t r;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(r) : "r"(addr), "r"(cta));
    return r;
}

// The receiving mbarrier of a CTA expects `expectedBytes` of st.async payload from the cluster (one phase).
__device__ __forceinline__ void mbarrierInitExpectTx(uint32_t mbar, int expectedBytes)
{
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(mbar) : "memory");
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(mbar), "r"(expectedBytes) : "memory");
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
}

__device__ __forceinline__ void mbarrierWaitPhase0(uint32_t mbar)
{
    uint32_t done = 0;
    while (!done)
    {
        asm volatile(
            "{\n .reg .pred p;\n mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%1], 0;\n"
            " selp.u32 %0, 1, 0, p;\n}"
            : "=r"(done)
            : "r"(mbar)
            : "memory");
    }
}

__device__ __forceinline__ void stAsyncV4(uint32_t remoteAddr, int4 val, uint32_t remoteMbar)
{
    asm volatile(
        "st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.s32 [%0], {%1,%2,%3,%4}, [%5];" ::"r"(remoteAddr),
        "r"(val.x), "r"(val.y), "r"(val.z), "r"(val.w), "r"(remoteMbar)
        : "memory");
}

#endif // SM10x

// R: int4 vectors per thread; kStaticC / kStaticNw / kStaticVc: compile-time cluster size, warps per CTA and vectors
// per CTA of the shapes that matter (0: runtime values).
template <int R, int kStaticC = 0, int kStaticNw = 0, int kStaticVc = 0>
__global__ void __launch_bounds__(kStaticNw > 0 ? kStaticNw * 32 : 1024, 1) moeSortKernel(int4 const* __restrict__ ids,
    int localOffset, int vecPerCta, int* __restrict__ tileExpert, int* __restrict__ tileMn, int4* __restrict__ e2p,
    int* __restrict__ p2e, int* __restrict__ totalPadded, int* __restrict__ numTilesOut)
{
    static_assert(R >= 1 && R <= kMaxVectorsPerThread);
    static_assert(kStaticNw == 0 || (kStaticNw >= 3 && kStaticNw <= 32));
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000) && (__CUDA_ARCH__ < 1100)
    __shared__ __align__(16) int sCnt[kNumLocal];
    __shared__ __align__(16) int sAll[kMaxClusterCtas][kNumLocal];
    __shared__ int sWcnt[kMaxWarps * kCountStride]; // per-warp counters, then per-warp offsets
    __shared__ int sBase[kNumLocal];
    __shared__ __align__(8) uint64_t sMbar;

    int const tid = threadIdx.x;
    int const nthr = kStaticNw > 0 ? kStaticNw * 32 : static_cast<int>(blockDim.x);
    int const nwarps = kStaticNw > 0 ? kStaticNw : (nthr >> 5);
    int const lane = tid & 31;
    int const warp = tid >> 5;
    uint32_t const rank = clusterCtaRank();
    uint32_t const C = kStaticC > 0 ? kStaticC : clusterNumCtas();
    int const vecs = kStaticVc > 0 ? kStaticVc : vecPerCta;
    int const vbase = rank * vecs;

    // Shared-memory setup first: it touches no global memory, so under PDL it overlaps the tail of the preceding
    // kernel.
    if (tid < kNumLocal)
        sCnt[tid] = 0;
    for (int i = tid; i < nwarps * kCountStride; i += nthr)
        sWcnt[i] = 0;
    if (tid == 0 && C > 1)
        mbarrierInitExpectTx(smemAddr(&sMbar), static_cast<int>(C) * kNumLocal * 4); // the C x 56 counts of the cluster
    // publishes the mbarrier init to the cluster; waited on right before the first DSMEM access
    clusterArriveRelease();

    // The ids come from the preceding kernel of the stream (the dispatch of the MoE layer). A launch without the
    // programmatic-serialization attribute makes the wait a no-op.
    pdlWait();
    int4 v[R];
#pragma unroll
    for (int j = 0; j < R; ++j)
    {
        int const lv = tid + j * nthr;
        v[j] = make_int4(-1, -1, -1, -1);
        if (kStaticVc > 0 || lv < vecs)
            v[j] = ldGlobalCg(ids + vbase + lv);
    }
    __syncthreads();

    // phase 1: classify + rank inside the warp
    int* wcnt = sWcnt + warp * kCountStride;
    int prim[R];   // first local entry of vector j: component << 24 | e << 16, or -1
    int prk[R];    // its rank (first used in phase 3: the atomic's latency overlaps the exchange)
    int ext[R][4]; // further local entries (rare): e << 16 | rank, or -1
#pragma unroll
    for (int j = 0; j < R; ++j)
    {
        int const lv = tid + j * nthr;
        uint32_t const ex = static_cast<uint32_t>(v[j].x) - static_cast<uint32_t>(localOffset);
        uint32_t const ey = static_cast<uint32_t>(v[j].y) - static_cast<uint32_t>(localOffset);
        uint32_t const ez = static_cast<uint32_t>(v[j].z) - static_cast<uint32_t>(localOffset);
        uint32_t const ew = static_cast<uint32_t>(v[j].w) - static_cast<uint32_t>(localOffset);
        uint32_t const m = (ex < kNumLocal ? 1u : 0u) | (ey < kNumLocal ? 2u : 0u) | (ez < kNumLocal ? 4u : 0u)
            | (ew < kNumLocal ? 8u : 0u);
        prim[j] = -1;
        prk[j] = 0;
#pragma unroll
        for (int c = 0; c < 4; ++c)
            ext[j][c] = -1;
        if (kStaticVc > 0 || lv < vecs)
        {
            if (m)
            {
                int const f = __ffs(m) - 1;
                uint32_t const e = f == 0 ? ex : f == 1 ? ey : f == 2 ? ez : ew;
                prk[j] = atomicAdd(&wcnt[e], 1);
                atomicAdd(&sCnt[e], 1);
                prim[j] = (f << 24) | (static_cast<int>(e) << 16);
                if (m & (m - 1))
                {
                    uint32_t const ec[4] = {ex, ey, ez, ew};
#pragma unroll
                    for (int c = 1; c < 4; ++c)
                    {
                        if (c > f && ((m >> c) & 1u))
                        {
                            int const r2 = atomicAdd(&wcnt[ec[c]], 1);
                            atomicAdd(&sCnt[ec[c]], 1);
                            ext[j][c] = (static_cast<int>(ec[c]) << 16) | r2;
                        }
                    }
                }
            }
            else
            {
                e2p[vbase + lv] = make_int4(-1, -1, -1, -1);
            }
        }
    }
    __syncthreads();

    // phase 2: send the CTA counts
    clusterWait(); // every CTA of the cluster has started and initialised its mbarrier
    if (C > 1)
    {
        constexpr int kVecPerDst = kNumLocal / 4; // 14 x 16 B per destination
        if (tid < static_cast<int>(C) * kVecPerDst)
        {
            uint32_t const dst = tid / kVecPerDst;
            int const q = tid - dst * kVecPerDst;
            int4 const val = reinterpret_cast<int4 const*>(sCnt)[q];
            uint32_t const ra = mapaShared(smemAddr(&sAll[rank][q * 4]), dst);
            uint32_t const rb = mapaShared(smemAddr(&sMbar), dst);
            stAsyncV4(ra, val, rb);
        }
    }
    // warps 1-2 (one lane per expert): per-warp offsets = running sum over the warps, in parallel with warp 0
    if (warp == 1 || warp == 2)
    {
        int const e = (warp - 1) * 32 + lane;
        if (e < kNumLocal)
        {
            int vals[kMaxWarps];
#pragma unroll
            for (int w = 0; w < kMaxWarps; ++w)
                vals[w] = w < nwarps ? sWcnt[w * kCountStride + e] : 0;
            int run = 0;
#pragma unroll
            for (int w = 0; w < kMaxWarps; ++w)
            {
                if (w < nwarps)
                    sWcnt[w * kCountStride + e] = run;
                run += vals[w];
            }
        }
    }
    // warp 0: cluster totals, tile layout, base rows
    int tb0 = 0, tb1 = 0, tt0 = 0, tt1 = 0, tot0s = 0, tot1s = 0, ntiles = 0;
    if (tid < 32)
    {
        int const e0 = lane, e1 = lane + 32;
        bool const ok1 = e1 < kNumLocal;
        int const* cntRows = sCnt;
        if (C > 1)
        {
            mbarrierWaitPhase0(smemAddr(&sMbar));
            cntRows = &sAll[0][0];
        }
        int tot0 = 0, tot1 = 0, pre0 = 0, pre1 = 0;
#pragma unroll
        for (int c = 0; c < kMaxClusterCtas; ++c)
        {
            if (c < static_cast<int>(C))
            {
                int const a = cntRows[c * kNumLocal + e0];
                int const b = ok1 ? cntRows[c * kNumLocal + e1] : 0;
                tot0 += a;
                tot1 += b;
                if (c < static_cast<int>(rank))
                {
                    pre0 += a;
                    pre1 += b;
                }
            }
        }
        int const t0 = (tot0 + kTile - 1) / kTile, t1 = (tot1 + kTile - 1) / kTile;
        // exclusive prefix of the tile counts by bit planes (1-2 planes on typical routings)
        int const nb = 32 - __clz(__reduce_max_sync(0xffffffffu, max(t0, t1)));
        unsigned const lt = (1u << lane) - 1u;
        int ex0 = 0, ex1 = 0, sum0 = 0, sum1 = 0;
        for (int bit = 0; bit < nb; ++bit)
        {
            unsigned const m0 = __ballot_sync(0xffffffffu, (t0 >> bit) & 1);
            unsigned const m1 = __ballot_sync(0xffffffffu, (t1 >> bit) & 1);
            ex0 += __popc(m0 & lt) << bit;
            ex1 += __popc(m1 & lt) << bit;
            sum0 += __popc(m0) << bit;
            sum1 += __popc(m1) << bit;
        }
        tb0 = ex0;
        tb1 = sum0 + ex1;
        ntiles = sum0 + sum1;
        sBase[e0] = tb0 * kTile + pre0;
        if (ok1)
            sBase[e1] = tb1 * kTile + pre1;
        tt0 = t0;
        tt1 = t1;
        tot0s = tot0;
        tot1s = tot1;
    }
    __syncthreads();

    if (tid < 32)
    {
        // tile tables, tiles of an expert spread over the CTAs
        int const e0 = lane, e1 = lane + 32;
        for (int j = rank; j < tt0; j += C)
        {
            int const t = tb0 + j;
            tileExpert[t] = e0;
            tileMn[t] = min((t + 1) * kTile, tb0 * kTile + tot0s);
        }
        if (e1 < kNumLocal)
        {
            for (int j = rank; j < tt1; j += C)
            {
                int const t = tb1 + j;
                tileExpert[t] = e1;
                tileMn[t] = min((t + 1) * kTile, tb1 * kTile + tot1s);
            }
        }
        if (rank == 0 && lane == 0)
        {
            numTilesOut[0] = ntiles;
            totalPadded[0] = ntiles * kTile;
        }
    }

    // phase 3: placement
#pragma unroll
    for (int j = 0; j < R; ++j)
    {
        int const p = prim[j];
        if (p >= 0)
        {
            int const gv = vbase + tid + j * nthr;
            int const f = p >> 24, e = (p >> 16) & 0xff;
            int const row = sBase[e] + wcnt[e] + prk[j];
            p2e[row] = gv * 4 + f;
            int o[4] = {f == 0 ? row : -1, f == 1 ? row : -1, f == 2 ? row : -1, f == 3 ? row : -1};
            if (ext[j][1] >= 0 || ext[j][2] >= 0 || ext[j][3] >= 0)
            {
#pragma unroll
                for (int c = 1; c < 4; ++c)
                {
                    int const q = ext[j][c];
                    if (q >= 0)
                    {
                        int const e2 = q >> 16;
                        int const row2 = sBase[e2] + wcnt[e2] + (q & 0xffff);
                        p2e[row2] = gv * 4 + c;
                        o[c] = row2;
                    }
                }
            }
            e2p[gv] = make_int4(o[0], o[1], o[2], o[3]);
        }
    }
    // The trigger only lets the dependent kernel launch early. Its griddepcontrol.wait returns once this whole grid has
    // completed and its stores are visible, and every consumer of the outputs (the permute and the grouped GEMMs of the
    // K3 MoE layer) reads them behind that wait. A consumer that read an output before its wait would need a
    // __syncthreads() ahead of this trigger: warp 0 stores the tile tables on its own, before phase 3.
    pdlTrigger();
#endif // SM10x
}

template <int R, int kStaticC = 0, int kStaticNw = 0, int kStaticVc = 0>
void launchKernel(int C, int nthr, int4 const* ids, int off, int vecPerCta, int* te, int* tm, int4* e2p, int* p2e,
    int* tot, int* nt, bool pdl, cudaStream_t stream)
{
    int const launchC = kStaticC > 0 ? kStaticC : C;
    int const launchNthr = kStaticNw > 0 ? kStaticNw * 32 : nthr;
    int const launchVc = kStaticVc > 0 ? kStaticVc : vecPerCta;
    // Phase 2 sends the CTA's 56 counts (14 x 16 B) to every CTA of the cluster, one thread per (destination, vector)
    // pair: with fewer than 14 x C threads some pairs are never sent and the receivers' mbarriers never complete.
    TLLM_CHECK_WITH_INFO(launchNthr >= launchC * (kNumLocal / 4),
        "invokeKimiK3MoeSortKf: %d threads cannot send the counts of a %d-CTA cluster (need %d)", launchNthr, launchC,
        launchC * (kNumLocal / 4));
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(launchC, 1, 1);
    cfg.blockDim = dim3(launchNthr, 1, 1);
    cfg.dynamicSmemBytes = 0;
    cfg.stream = stream;
    cudaLaunchAttribute attr[2];
    attr[0].id = cudaLaunchAttributeClusterDimension;
    attr[0].val.clusterDim.x = launchC;
    attr[0].val.clusterDim.y = 1;
    attr[0].val.clusterDim.z = 1;
    attr[1].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[1].val.programmaticStreamSerializationAllowed = pdl ? 1 : 0;
    cfg.attrs = attr;
    cfg.numAttrs = 2;
    TLLM_CUDA_CHECK(cudaLaunchKernelEx(
        &cfg, moeSortKernel<R, kStaticC, kStaticNw, kStaticVc>, ids, off, launchVc, te, tm, e2p, p2e, tot, nt));
}

} // namespace

void invokeKimiK3MoeSortKf(int32_t const* tokenSelectedExperts, int32_t numTokens, int32_t localExpertOffset,
    int32_t* tileIdxToExpertIdx, int32_t* tileIdxToMnLimit, int32_t* expandedIdxToPermutedIdx,
    int32_t* permutedIdxToExpandedIdx, int32_t* totalNumPaddedTokens, int32_t* numNonExitingTiles, cudaStream_t stream)
{
    int const smVersion = tensorrt_llm::common::getSMVersion();
    TLLM_CHECK_WITH_INFO(smVersion == 107,
        "invokeKimiK3MoeSortKf requires an SM107 GPU: the kernel has only been run there (got SM%d)", smVersion);
    TLLM_CHECK_WITH_INFO(numTokens >= kKimiK3MoeSortKfTokenMultiple && numTokens <= kKimiK3MoeSortKfMaxTokens
            && numTokens % kKimiK3MoeSortKfTokenMultiple == 0,
        "invokeKimiK3MoeSortKf: numTokens %d must be a multiple of %d in [%d, %d]", numTokens,
        kKimiK3MoeSortKfTokenMultiple, kKimiK3MoeSortKfTokenMultiple, kKimiK3MoeSortKfMaxTokens);
    TLLM_CHECK_WITH_INFO(reinterpret_cast<uintptr_t>(tokenSelectedExperts) % 16 == 0
            && reinterpret_cast<uintptr_t>(expandedIdxToPermutedIdx) % 16 == 0,
        "invokeKimiK3MoeSortKf: token_selected_experts and expanded_idx_to_permuted_idx must be 16-byte aligned");

    bool const pdl = tensorrt_llm::common::getEnvEnablePDL();
    auto const* ids = reinterpret_cast<int4 const*>(tokenSelectedExperts);
    auto* e2p = reinterpret_cast<int4*>(expandedIdxToPermutedIdx);

#define KF_LAUNCH_FIXED(R, C, NW, VC)                                                                                  \
    launchKernel<R, C, NW, VC>(C, NW * 32, ids, localExpertOffset, VC, tileIdxToExpertIdx, tileIdxToMnLimit, e2p,      \
        permutedIdxToExpandedIdx, totalNumPaddedTokens, numNonExitingTiles, pdl, stream)
    // The shapes of the DEP16 decode buckets (T = 16 x tokens per rank): a full eight-CTA cluster, compile-time sizes.
    bool fixedShape = true;
    switch (numTokens)
    {
    case 768: KF_LAUNCH_FIXED(1, 8, 12, 384); break;
    case 1024: KF_LAUNCH_FIXED(1, 8, 16, 512); break;
    case 1536: KF_LAUNCH_FIXED(1, 8, 24, 768); break;
    case 2048: KF_LAUNCH_FIXED(1, 8, 32, 1024); break;
    case 3072: KF_LAUNCH_FIXED(2, 8, 24, 1536); break;
    case 4096: KF_LAUNCH_FIXED(2, 8, 32, 2048); break;
    case 6144: KF_LAUNCH_FIXED(3, 8, 32, 3072); break;
    default: fixedShape = false; break;
    }
#undef KF_LAUNCH_FIXED

    if (!fixedShape)
    {
        // Any other multiple of 16: clusters of 1, 2, 4 or 8 CTAs, at least 256 vectors per CTA (numVec % 64 == 0).
        int const numVec = numTokens * (kTopK / 4);
        int C = 1;
        while (C < kMaxClusterCtas && numVec / (C * 2) >= kMinVecPerCta)
            C *= 2;
        int const vecPerCta = numVec / C;
        int const R = (vecPerCta + 1023) / 1024;
        int nthr = (vecPerCta + R - 1) / R;
        nthr = (nthr + 31) / 32 * 32;
        if (nthr < 96) // warps 0-2 have roles in phase 2
            nthr = 96;

#define KF_LAUNCH_GENERIC(R)                                                                                           \
    launchKernel<R>(C, nthr, ids, localExpertOffset, vecPerCta, tileIdxToExpertIdx, tileIdxToMnLimit, e2p,             \
        permutedIdxToExpandedIdx, totalNumPaddedTokens, numNonExitingTiles, pdl, stream)
        switch (R)
        {
        case 1: KF_LAUNCH_GENERIC(1); break;
        case 2: KF_LAUNCH_GENERIC(2); break;
        case 3: KF_LAUNCH_GENERIC(3); break;
        default: KF_LAUNCH_GENERIC(4); break;
        }
#undef KF_LAUNCH_GENERIC
    }
    sync_check_cuda_error(stream);
}

} // namespace kernels

TRTLLM_NAMESPACE_END
