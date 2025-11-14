/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: NVIDIA TensorRT Source Code License Agreement
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include "defines.h"
#include "mha.h"
#if IS_MLA
#include "barriers.cuh"
#include "mhaUtils.cuh"
#include "mha_components.cuh"
#include "mha_stdheaders.cuh"
#include "mla_sm120.cuh"
#include "mma.cuh"
#include "tma.h"
#include "utils.cuh"
#include "utils.h"

#ifndef GENERATE_CUBIN
#include "hostUtils.h"
#include "tensorMap.h"
#include <cuda_runtime.h>
#endif

#define USE_REG_Q 1

__constant__ constexpr XQAKernelType kernelType = XQAKernelType::kSM120_MLA;

inline constexpr bool allowMultipleInputTokens = true;

inline constexpr uint32_t partElemsK = 64; // @fixme: change this to 128 to save L2 traffic
inline constexpr uint32_t nbKParts = exactDiv(validElemsPerKHead, partElemsK);
inline constexpr uint32_t nbQParts = nbKParts;

inline constexpr uint32_t tokensPerTile = 64;
inline constexpr uint32_t partElemsV = 128;
inline constexpr uint32_t nbVSplit = 2;
inline constexpr uint32_t gemm1V = exactDiv(validElemsPerVHead, nbVSplit);
inline constexpr uint32_t nbProducerCtasPerCga = nbVSplit;

inline constexpr uint32_t multiBlockMinNbTilesPerCta = 2;
inline constexpr uint32_t multiBlockMinNbTiles = multiBlockMinNbTilesPerCta * 2;

using MathElem = CacheElem;
inline constexpr uint32_t mathElemBytes = sizeof(MathElem);
inline constexpr uint32_t grainsPerPartK = exactDiv(partElemsK * mathElemBytes, grainBytes);

inline constexpr uint32_t grainElems = exactDiv(grainBytes, mathElemBytes);

inline constexpr float xScale = 1.f / kE4M3_MAX;
__constant__ constexpr float rcpXScale = kE4M3_MAX;

inline constexpr uint32_t nbRegsForIOWarps = 32;
inline constexpr uint32_t nbRegsForMathWarps = 232;

inline constexpr bool computeRowSumFromF8 = true;

struct KVTilePartLoader
{
#if USE_PAGED_KV_CACHE
    static_assert(tokensPerPage % tokensPerTile == 0 || tokensPerTile % tokensPerPage == 0);
    static inline constexpr uint32_t nbPagesPerTile
        = tokensPerTile >= tokensPerPage ? exactDiv(tokensPerTile, tokensPerPage) : 1;
#endif

    static inline constexpr uint32_t const nbKHeads = 1;
    KVCacheList<usePagedKVCache> const& cacheList;
    uint32_t const idxReq;
    static inline constexpr uint32_t const idxHeadGrp = 0;

    CUtensorMap const& tensorMap;
    // if greater than 1, then we need unrolling for the loading loop. Seems 1 is fine for latency.
    static inline constexpr uint32_t nbPageBuffers = 1;
#if USE_PAGED_KV_CACHE
    uint32_t const nbPages;    // for bound check
    Vec<KVCachePageIndex, nbPagesPerTile> pageBuffers[nbPageBuffers];
    uint32_t idxTileRef = ~0U; // idxTile used to load the pages
#endif
    uint32_t const baseOffset;

    __device__ KVTilePartLoader(
        KVCacheList<usePagedKVCache> const& cacheList, uint32_t idxReq, CUtensorMap const& tensorMap
#if USE_PAGED_KV_CACHE
        ,
        uint32_t nbPages
#endif
    );
    // tensorMap is for one whole page ([nbKHeads*tokensPerPage][headElems]) or whole cache
    template <uint32_t nbTokens, uint32_t grainsPerPart, bool alignedForSwizzle>
    __device__ void loadData(Array2D<LdGrain, nbTokens, grainsPerPart, alignedForSwizzle>& dst, uint32_t idxTile,
        uint32_t idxElemBeg, CtaBarrier& bar, uint32_t idxPageBuf);

    __device__ void loadPages(uint32_t idxTile, uint32_t idxPageBuf);
};

__device__ inline KVTilePartLoader::KVTilePartLoader(
    KVCacheList<usePagedKVCache> const& cacheList, uint32_t idxReq, CUtensorMap const& tensorMap
#if USE_PAGED_KV_CACHE
    ,
    uint32_t nbPages
#endif
    )
    : cacheList{cacheList}
    , idxReq{idxReq}
    , tensorMap{tensorMap}
#if USE_PAGED_KV_CACHE
    , nbPages{nbPages}
#if PAGED_KV_CACHE_LAYOUT == 1
    , baseOffset{idxReq * cacheList.maxNbPagesPerSeq}
#else
    , baseOffset{((idxReq * beamWidth) * 2) * cacheList.maxNbPagesPerSeq}
#endif
#else
    , baseOffset{(idxReq * beamWidth) * 2}
#endif
{
#pragma unroll
    for (auto& pageBuffer : pageBuffers)
    {
        pageBuffer.fill(kBAD_PAGE_INDEX);
    }
}

// tensorMap is for one whole page ([nbKHeads*tokensPerPage][headElems]) or whole cache
template <uint32_t nbTokens, uint32_t grainsPerPart, bool alignedForSwizzle>
__device__ inline void KVTilePartLoader::loadData(Array2D<LdGrain, nbTokens, grainsPerPart, alignedForSwizzle>& dst,
    uint32_t idxTile, uint32_t idxElemBeg, CtaBarrier& bar, uint32_t idxPageBuf)
{
    static_assert(nbTokens == tokensPerTile);
#if USE_PAGED_KV_CACHE
    assert(idxTile == idxTileRef);
    auto const& pages = pageBuffers[idxPageBuf];
    if constexpr (nbTokens < tokensPerPage)
    {
        assert(nbPagesPerTile == 1);
        uint32_t const offset = nbTokens * (idxTile % exactDiv(tokensPerPage, nbTokens));
        if (warpElectSync())
        {
#if PAGED_KV_CACHE_LAYOUT == 1
            tma::loadAsync(&dst, tensorMap, DimsLE<4>{idxElemBeg, idxHeadGrp, offset, (uint32_t) pages[0]}, bar);
#else
            tma::loadAsync(&dst, tensorMap, DimsLE<4>{idxElemBeg, offset, idxHeadGrp, (uint32_t) pages[0]}, bar);
#endif
        }
    }
    else
    {
#pragma unroll
        for (uint32_t i = 0; i < nbPagesPerTile; i++)
        {
            if (warpElectSync())
            {
#if PAGED_KV_CACHE_LAYOUT == 1
                tma::loadAsync(&dst(tokensPerPage * i, 0), tensorMap,
                    DimsLE<4>{idxElemBeg, idxHeadGrp, 0, (uint32_t) pages[i]}, bar);
#else
                tma::loadAsync(&dst(tokensPerPage * i, 0), tensorMap,
                    DimsLE<4>{idxElemBeg, 0, idxHeadGrp, (uint32_t) pages[i]}, bar);
#endif
            }
        }
    }
#else
    if (warpElectSync())
    {
        tma::loadAsync(&dst, tensorMap, DimsLE<4>{idxElemBeg, nbTokens * idxTile, idxHeadGrp, baseOffset}, bar);
    }
#endif
}

__device__ inline void KVTilePartLoader::loadPages(uint32_t idxTile, uint32_t idxPageBuf)
{
#if USE_PAGED_KV_CACHE
    uint32_t const idxPageBeg
        = tokensPerTile >= tokensPerPage ? nbPagesPerTile * idxTile : idxTile / exactDiv(tokensPerPage, tokensPerTile);
    auto& pages = pageBuffers[idxPageBuf];
#pragma unroll
    for (uint32_t i = 0; i < nbPagesPerTile; i++)
    {
        uint32_t const idxPage = idxPageBeg + i;
        pages[i] = idxPage < nbPages ? cacheList.kvCachePageList[baseOffset + idxPage] : kBAD_PAGE_INDEX;
    }
    idxTileRef = idxTile;
#endif
}

using Mat16x32 = Vec<uint32_t, 4>;

template <uint32_t srcRows, uint32_t srcCols>
class Mat16x32Loader
{
public:
    using Src = Array2D<LdGrain, srcRows, srcCols>;

    // default r and c are for mat A.
    __device__ inline Mat16x32Loader(
        Src const& src, uint32_t baseRow, uint32_t idxInstK, uint32_t r = laneId() % 16, uint32_t c = laneId() / 16)
        : src{src}
        , baseRow{baseRow}
        , idxInstK{idxInstK}
        , r{r}
        , c{c}
        , basePtr{getPtrRef(0)}
    {
        static_assert((grainBytes * srcCols * qmmaShape.m) % 1024 == 0);
    }

    __device__ inline Mat16x32 load(uint32_t idxInstM) const
    {
        return ldmatrix<false, 4>(getPtr(idxInstM));
    }

    template <uint32_t tileM>
    __device__ inline Vec<Mat16x32, exactDiv(tileM, qmmaShape.m)> loadWholeCol() const
    {
        uint32_t const nbInstM = exactDiv(tileM, qmmaShape.m);
        Vec<Mat16x32, nbInstM> ret;
#pragma unroll
        for (uint32_t i = 0; i < nbInstM; i++)
        {
            ret[i] = load(i);
        }
        return ret;
    }

    __device__ inline LdGrain const* getPtr(uint32_t idxInstM) const
    {
        return checkedVal(basePtr + idxInstM * qmmaShape.m * srcCols, getPtrRef(idxInstM));
    }

private:
    __device__ inline LdGrain const* getPtrRef(uint32_t idxInstM) const
    {
        return &src.template at<true>(
            baseRow + idxInstM * qmmaShape.m + r, idxInstK * exactDiv(qmmaShape.k, grainElems) + c);
    }

    Src const& src;
    uint32_t const baseRow;
    uint32_t const idxInstK;
    uint32_t const r;
    uint32_t const c;
    LdGrain const* const basePtr;
};

using InstAcc = Array2D<float, 2, 2>;

using XBuffer = Array2D<LdGrain, headGrpSize, exactDiv(mathElemBytes* tokensPerTile, grainBytes)>;

struct CgaXBuffer
{
    XBuffer x;
    Vec<float, headGrpSize> rowSum;
    Vec<float, headGrpSize> rowMaxLog2e;
};

struct PingPongMutex
{
    using ShmStorage = CtaBarrier[2];
    ShmStorage& barriers;
    uint32_t const idxGrp;
    bool skipWait = false;

    static __device__ inline void initStorage(ShmStorage& barriers, uint32_t thrdsPerGrp)
    {
        new (&barriers[0]) CtaBarrier(thrdsPerGrp);
        new (&barriers[1]) CtaBarrier(thrdsPerGrp);
        barriers[0].arrive(thrdsPerGrp);
    }

    __device__ inline PingPongMutex(ShmStorage& shmStorage, uint32_t idxGrp)
        : barriers{shmStorage}
        , idxGrp{idxGrp}
    {
    }

    __device__ inline void test_lock(uint32_t iter)
    {
        skipWait = barriers[idxGrp].test_wait_parity(toParity<1>(iter));
    }

    __device__ inline void lock(uint32_t iter)
    {
        if (!skipWait)
        {
            barriers[idxGrp].wait_parity(toParity<1>(iter));
        }
    }

    __device__ inline void unlock()
    {
        barriers[idxGrp ^ 1U].arrive();
        skipWait = false;
    }
};

struct PartialResult
{
    static constexpr uint32_t nbChunks = 4;
    static constexpr uint32_t nbRowsPerChunk = exactDiv(headGrpSize, nbChunks);

    struct Chunk
    {
        Vec<OutputHead, nbRowsPerChunk> data;
        Vec<float, nbRowsPerChunk> rowSum;
        Vec<float, nbRowsPerChunk> rowMaxLog2e;
    };

    Chunk chunks[nbChunks];
};

constexpr uint32_t nbMathWarpsA = 8;
constexpr uint32_t nbComputeWarpsB = 8;
constexpr uint32_t nbMathGrpsA = 2;
constexpr uint32_t nbMathWarpsB = 8;

constexpr uint32_t nbMultiBlockBufs = 2;
constexpr uint32_t multiBlockMathWarps = 8;

constexpr bool useRegQ = USE_REG_Q;

struct SharedMemA
{
    static inline constexpr uint32_t nbKBufs = 12;

    static inline constexpr uint32_t regQParts = (useRegQ ? 4 : 0);
    static inline constexpr uint32_t shmQParts = nbQParts - regQParts;

    using ShmQPart = Array2D<LdGrain, headGrpSize, grainsPerPartK>;
    using ShmKPart = Array2D<LdGrain, tokensPerTile, grainsPerPartK>;

    Vec<ShmQPart, shmQParts> q;
    ShmKPart k[nbKBufs];

    // single buffer reused by two groups. sendX() warp will arbitrate the order of x buffer access via two xBars.
    CgaXBuffer x;

    // scaled by log2e. Write by last CGA iteration (from the other producer CTA) and read by current producer CTA.
    Vec<float, headGrpSize> rowMaxLog2e;
    // sync rowMaxLog2e between two producer CTAs and .consumed means the buffer for next iteration (in next producer)
    // is ready. The 4 groups from 2 producers CTAs form a ring
    CgaBarrier rowMaxLog2eBar[nbMathGrpsA];

    PingPongMutex::ShmStorage tensorCoreMutex;

    CtaBarrierPair kBars[nbKBufs];
    static inline constexpr uint32_t nbXBars = nbMathGrpsA;
    CtaBarrierPair xBars[nbXBars];
#if USE_REG_Q
    CtaBarrierPair regQBar;
#endif
    CtaBarrier shmQBar;
    CgaBarrier cgaXBufConsumed; // for X

    CtaBarrierPair multiBlockBars[nbMultiBlockBufs];

    __device__ inline void invalidateBarriers(uint32_t thrdIdx)
    {
        constexpr uint32_t nbBars = (useRegQ ? 12 : 10) + 2 * (nbKBufs + nbXBars);
#ifndef __CUDACC_RTC__
        constexpr uint32_t nbBarsRef
            = exactDiv(offsetof(SharedMemA, qkScaleLog2e) - offsetof(SharedMemA, rowMaxLog2eBar), 8);
        static_assert(nbBars == nbBarsRef);
#endif
        if (thrdIdx < nbBars)
        {
            reinterpret_cast<CtaBarrier*>(&rowMaxLog2eBar[0])[thrdIdx].~CtaBarrier();
        }
    }

    __device__ inline Vec<PartialResult::Chunk, nbMultiBlockBufs>& getMultiBlockBufs()
    {
#ifndef __CUDACC_RTC__
        assert(sizeof(Vec<PartialResult::Chunk, nbMultiBlockBufs>) < offsetof(SharedMemA, rowMaxLog2eBar));
#endif
        return *reinterpret_cast<Vec<PartialResult::Chunk, nbMultiBlockBufs>*>(this);
    }

    float qkScaleLog2e;
    bool isLastSubSeq;
};

struct SharedMemB
{
    static inline constexpr uint32_t nbXVBufs = 2;
    static inline constexpr uint32_t nbXBufs = nbXVBufs;
    static inline constexpr uint32_t nbVBufs = nbXVBufs;

    using VBuffer
        = Vec<Array2D<LdGrain, tokensPerTile, exactDiv(partElemsV, grainElems)>, exactDiv(gemm1V, partElemsV)>;

    // x and v are using gemmK=128 per iteration. If we see high pressure on shared memory capacity, we can change to 64
    // in the future.
    struct XVBuffer
    {
        VBuffer v;
        CgaXBuffer x;
        uint8_t pad[headGrpSize * 128 * 2 - sizeof(VBuffer) - sizeof(CgaXBuffer)]; // for output swizzling
    };

    XVBuffer xv[nbXVBufs];

    __device__ inline XBuffer& x(uint32_t idx)
    {
        return xv[idx].x.x;
    }

    __device__ inline VBuffer& v(uint32_t idx)
    {
        return xv[idx].v;
    }

    __device__ inline Vec<float, headGrpSize>& xRowSum(uint32_t idx)
    {
        return xv[idx].x.rowSum;
    }

    __device__ inline Vec<float, headGrpSize>& xRowMaxLog2e(uint32_t idx)
    {
        return xv[idx].x.rowMaxLog2e;
    }

    static inline constexpr uint32_t nbAccRowMaxSumCopies = 2;
    Vec<float, headGrpSize> accRowMaxLog2e[nbAccRowMaxSumCopies];
    Vec<float, headGrpSize> accRowSum[nbAccRowMaxSumCopies];

    CtaBarrierPair xBars[nbXBufs];
    CtaBarrierPair vBars[nbVBufs];

    CgaBarrier cgaXBufProduced[nbProducerCtasPerCga];
    CtaBarrier mathWarpsBar;

    CtaBarrierPair multiBlockBars[nbMultiBlockBufs];

    __device__ inline void invalidateBarriers(uint32_t thrdIdx)
    {
        constexpr uint32_t nbBars = 15;
#ifndef __CUDACC_RTC__
        constexpr uint32_t nbBarsRef = exactDiv(offsetof(SharedMemB, isLastSubSeq) - offsetof(SharedMemB, xBars), 8);
        static_assert(nbBars == nbBarsRef);
#endif
        if (thrdIdx < nbBars)
        {
            reinterpret_cast<CtaBarrier*>(&xBars[0])[thrdIdx].~CtaBarrier();
        }
    }

    __device__ inline Vec<PartialResult::Chunk, nbMultiBlockBufs>& getMultiBlockBufs()
    {
#ifndef __CUDACC_RTC__
        static_assert(sizeof(Vec<PartialResult::Chunk, nbMultiBlockBufs>) < offsetof(SharedMemB, xBars));
#endif
        return *reinterpret_cast<Vec<PartialResult::Chunk, nbMultiBlockBufs>*>(this);
    }

    bool isLastSubSeq;
};

__device__ void mergePartialOutputs(uint32_t& semaphore, Vec<OutputHead, PartialResult::nbRowsPerChunk>& dst,
    PartialResult const* reqPartialResults, uint32_t nbSubSeq, uint32_t ctaRank, uint32_t warpRank, uint2 warpIdx,
    void* sharedMem);

struct KernelArgs
{
    CUtensorMap const& tensorMapQ; // MhaIOHead[nbQHeads * totalNbInputTokens]
    CUtensorMap const& tensorMapK;
    CUtensorMap const& tensorMapV;
    float const& qScale;
    OutputHead* __restrict__ const& output; // [totalNbIntputTokens][nbQHeads]
    KVCacheList<usePagedKVCache> const& cacheList;
    uint32_t const& batchSize;
    float const* __restrict__ const&
        kvCacheScale; // Device memory scalar. Same scale for K and V cache. Used only for int8/fp8 KV cache.
    Vec<CgaXBuffer, nbProducerCtasPerCga>* __restrict__ const& cgaXBuf; // [totalNbInputTokens][maxNbSubSeq]
    uint32_t* __restrict__ const& semaphores;                           // [totalNbInputTokens]
    PartialResult* __restrict__ const& partialResults;                  // [totalNbInputTokens][maxNbSubSeq]
};

struct Producer
{
    static inline constexpr uint32_t nbMathGrps = nbMathGrpsA;
    static inline constexpr uint32_t nbMathWarps = nbMathWarpsA;
    static inline constexpr uint32_t nbMathThrds = nbMathWarps * warp_size;
    static inline constexpr uint32_t warpsPerGrp = exactDiv(nbMathWarps, nbMathGrps);
    static inline constexpr uint32_t thrdsPerGrp = warpsPerGrp * warp_size;
    static inline constexpr uint2 warpTile = {tokensPerTile, exactDiv(headGrpSize, warpsPerGrp)};
    using WarpAcc = WarpAccT<warpTile.y, warpTile.x>;
    using ThrdRegRowMax = ThrdRegRowMaxT<warpTile.y>;
    using QuadRegRowMax = QuadRegRowMaxT<warpTile.y>;

    KernelArgs const& args;
    SharedMemA& smem;
    uint32_t const maxNbSubSeq;
    uint32_t const idxReq;
    uint32_t const idxInputTokenGlobal;
    uint32_t const nbSubSeq;
    uint32_t const idxSubSeq;
    uint32_t const seqLen;
    uint32_t const ctaRank;
    uint32_t const warpRank;
    uint2 const warpIdx;

    __device__ inline Producer(KernelArgs const& args, SharedMemA& smem, uint32_t const maxNbSubSeq,
        uint32_t const idxReq, uint32_t idxInputTokenGlobal, uint32_t const seqLen, uint32_t const nbSubSeq,
        uint32_t const idxSubSeq, uint32_t ctaRank, uint32_t const warpRank, uint2 const warpIdx)
        : args(args)
        , smem(smem)
        , maxNbSubSeq(maxNbSubSeq)
        , idxReq(idxReq)
        , idxInputTokenGlobal(idxInputTokenGlobal)
        , seqLen(seqLen)
        , nbSubSeq(nbSubSeq)
        , idxSubSeq(idxSubSeq)
        , ctaRank(ctaRank)
        , warpRank(warpRank)
        , warpIdx(warpIdx)
    {
#ifndef NDEBUG
        if (threadIdx.x == 0)
        {
            asm("st.bulk.weak [%0], %1, 0;\n" ::"l"(&smem), "n"(sizeof(SharedMemA)) : "memory");
        }
        __syncthreads();
#endif
        if (threadIdx.x == 0)
        {
            smem.qkScaleLog2e = args.qScale * args.kvCacheScale[0] * log2e;
        }

        if (threadIdx.x < headGrpSize)
        {
            smem.rowMaxLog2e[threadIdx.x] = safeInitRowMax;
        }
        if (warpElectSync())
        {
            if (warpRank < SharedMemA::nbKBufs)
            {
                auto& b = smem.kBars[warpRank];
                b.initialize(1, thrdsPerGrp);
                b.consumed.arrive<Scope::CTA, ArriveOrder::RELAXED>(thrdsPerGrp);
            }
            if (warpRank < SharedMemA::nbXBars)
            {
                auto& b = smem.xBars[warpRank];
                b.initialize(thrdsPerGrp, 1);
            }
#if USE_REG_Q
            if (warpRank == 0)
            {
                smem.regQBar.initialize(1, nbMathThrds);
                smem.regQBar.consumed.arrive<Scope::CTA, ArriveOrder::RELAXED>(nbMathThrds);
            }
#endif
            if (warpRank < nbMathGrpsA)
            {
                auto& b = smem.rowMaxLog2eBar[warpRank];
                init(&b, thrdsPerGrp);
            }
            if (ctaRank == 0 && warpRank == 0)
            {
                smem.rowMaxLog2eBar[0].arrive<Scope::CTA, ArriveOrder::RELAXED>(thrdsPerGrp);
            }
            if (warpRank == 0)
            {
                init(&smem.shmQBar, 1);
                init(&smem.cgaXBufConsumed, 1 * nbVSplit);
                smem.cgaXBufConsumed.arrive<Scope::CTA, ArriveOrder::RELAXED>(1 * nbVSplit);
                PingPongMutex::initStorage(smem.tensorCoreMutex, thrdsPerGrp);
            }
            if (nbSubSeq > 1 && warpRank < nbMultiBlockBufs)
            {
                auto& b = smem.multiBlockBars[warpRank];
                b.initialize(1, warp_size * multiBlockMathWarps);
                b.consumed.arrive<Scope::CTA, ArriveOrder::RELAXED>(warp_size * multiBlockMathWarps);
            }
        }
        clusterBarArrive();
        clusterBarWait();
    }

    __device__ inline ~Producer()
    {
        clusterBarArrive();
        clusterBarWait();
        smem.invalidateBarriers(threadIdx.x);
    }

    __device__ inline void run()
    {
        if (warpIdx.y == 2)
        { // IO warps
            asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" ::"n"(nbRegsForIOWarps));
            if (warpIdx.x == 0)
            { // q
                loadQ();
            }
            else if (warpIdx.x == 1)
            { // k
                loadK();
            }
            else if (warpIdx.x == 2)
            { // x
                sendX();
            }
        }
        else
        { // Compute warps
            asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" ::"n"(nbRegsForMathWarps));
            compute();
        }
        if (nbSubSeq > 1)
        {
            mergePartialOutputs(args.semaphores[idxInputTokenGlobal],
                reinterpret_cast<Vec<OutputHead, PartialResult::nbRowsPerChunk>&>(
                    args.output[headGrpSize * idxInputTokenGlobal + PartialResult::nbRowsPerChunk * ctaRank]),
                args.partialResults + maxNbSubSeq * idxInputTokenGlobal, nbSubSeq, ctaRank, warpRank, warpIdx, &smem);
        }
    }

private:
    __device__ inline uint32_t iterStride() const
    {
        return nbSubSeq * nbProducerCtasPerCga;
    }

    __device__ inline uint32_t idxTileBeg() const
    {
        return nbProducerCtasPerCga * idxSubSeq + ctaRank;
    }

    __device__ inline uint32_t nbTiles() const
    {
        return divUp(seqLen, tokensPerTile);
    }

    __device__ inline SharedMemB& getConsumerShm(uint32_t const idxConsumer)
    {
        return *mapa(reinterpret_cast<SharedMemB*>(&smem), nbProducerCtasPerCga + idxConsumer);
    };

    static constexpr uint32_t regQPartShmBeg = SharedMemA::shmQParts - SharedMemA::regQParts;

    __device__ inline void loadQ()
    {
#if USE_REG_Q
        static_assert(SharedMemA::regQParts <= SharedMemA::shmQParts);
        smem.regQBar.consumed.wait_parity(toParity<1>(0));
#pragma unroll 1
        for (uint32_t i = 0; i < SharedMemA::regQParts; i++)
        {
            if (warpElectSync())
            {
                tma::loadAsync(&smem.q[regQPartShmBeg + i], args.tensorMapQ,
                    DimsLE<2>{partElemsK * i, headGrpSize * idxInputTokenGlobal}, smem.regQBar.produced);
            }
        }
        if (warpElectSync())
        {
            smem.regQBar.produced.arrive_tx(sizeof(SharedMemA::ShmQPart) * SharedMemA::regQParts);
        }
#endif
#pragma unroll 1
        for (uint32_t i = 0; i < SharedMemA::shmQParts; i++)
        {
            uint32_t const idxPart = SharedMemA::regQParts + i;
#if USE_REG_Q
            if (i == regQPartShmBeg)
            {
                smem.regQBar.consumed.wait_parity(toParity<1>(1));
            }
#endif
            if (warpElectSync())
            {
                tma::loadAsync(&smem.q[i], args.tensorMapQ,
                    DimsLE<2>{partElemsK * idxPart, headGrpSize * idxInputTokenGlobal}, smem.shmQBar);
            }
        }
        if (warpElectSync())
        {
            smem.shmQBar.arrive_tx(sizeof(SharedMemA::ShmQPart) * SharedMemA::shmQParts);
        }
    }

    __device__ inline void loadK();

    __device__ inline void sendX();

    __device__ inline void compute()
    {
        uint32_t const grpIdx = warpIdx.y;
        uint32_t const tileBaseRow = warpTile.y * warpIdx.x;
        PingPongMutex tensorCoreMutex{smem.tensorCoreMutex, grpIdx};

        constexpr uint32_t partNbInstK = exactDiv(partElemsK, qmmaShape.k);
        using AtomA = Vec<uint32_t, 4>; // for 16x32 data, working as mat A of QMMA.16832
        using RegQPartCol = Vec<AtomA, exactDiv(warpTile.y, qmmaShape.m)>;
        using RegQPart = Vec<RegQPartCol, partNbInstK>;
        using RegQ = Vec<RegQPart, SharedMemA::regQParts>;
        constexpr uint32_t tileNbAtomBx2 = exactDiv(tokensPerTile, qmmaShape.n * 2);
        using AtomBx2 = Vec<uint32_t, 4>; // one AtomB is 8x32 and AtomBx2 is 16x32
        using RegKPartCol = Vec<AtomBx2, tileNbAtomBx2>;
        using RegKPart = Vec<RegKPartCol, partNbInstK>;

        uint32_t const lane = laneId();
        uint32_t const rA = lane % 16;
        uint32_t const cA = lane / 16;
        uint32_t const rB = (lane / 16) * 8 + lane % 8;
        uint32_t const cB = (lane % 16) / 8;
        auto loadRegQCol = [&](SharedMemA::ShmQPart const& q, uint32_t idxInstK) -> RegQPartCol
        {
            Mat16x32Loader const loaderQ(q, tileBaseRow, idxInstK, rA, cA);
            return loaderQ.loadWholeCol<warpTile.y>();
        };
        auto loadRegKCol = [&](SharedMemA::ShmKPart const& k, uint32_t idxInstK) -> RegKPartCol
        {
            Mat16x32Loader const loaderK(k, 0, idxInstK, rB, cB);
            return loaderK.loadWholeCol<warpTile.x>();
        };
        auto loadPart = [&](auto const& loadCol, auto const& shmPart)
        {
            mha::conditional_t<mha::is_same_v<SharedMemA::ShmQPart, mha::decay_t<decltype(shmPart)>>, RegQPart,
                RegKPart>
                regPart;
#pragma unroll
            for (uint32_t idxInstK = 0; idxInstK < partNbInstK; idxInstK++)
            {
                regPart[idxInstK] = loadCol(shmPart, idxInstK);
            }
            return regPart;
        };

#if USE_REG_Q
        // load regQ
        smem.regQBar.produced.wait_parity(toParity<1>(0));
        RegQ regQ;
#pragma unroll
        for (uint32_t idxPart = 0; idxPart < SharedMemA::regQParts; idxPart++)
        {
            uint32_t const idxBuf = regQPartShmBeg + idxPart;
            regQ[idxPart] = loadPart(loadRegQCol, smem.q[idxBuf]);
        }
        smem.regQBar.consumed.arrive();
#endif
// main loop
#pragma unroll 1
        for (uint32_t grpIter = 0; true; grpIter++)
        {
            uint32_t const ctaIter = grpIdx + grpIter * nbMathGrps;
            uint32_t const idxTile = idxTileBeg() + iterStride() * ctaIter;
            if (idxTile >= nbTiles())
            {
                break;
            }
            WarpAcc acc{};
            // wait until it's our turn
            tensorCoreMutex.lock(grpIter);
            BarWaiter kBarWaiter(smem.kBars, ctaIter * nbKParts + 0);
            kBarWaiter.testWait();
            RegQPart regQBuf;
#if USE_REG_Q
            static_assert(SharedMemA::regQParts > 0);
            regQBuf[0] = regQ[0][0];
#else
            regQBuf[0] = loadRegQCol(smem.q[0], 0);
#endif
            kBarWaiter.wait();
            RegKPart regKBuf;
            regKBuf[0] = loadRegKCol(smem.k[kBarWaiter.idxBuf], 0);

            auto shouldTestWait = [](uint32_t idxInstK, uint32_t idxAtomBx2)
            { return idxInstK == partNbInstK - 1 && idxAtomBx2 == tileNbAtomBx2 - 2; };
            BarWaiter kBarWaiterNext = kBarWaiter.next();
#if USE_REG_Q
#pragma unroll
            for (uint32_t idxPart = 0; idxPart < SharedMemA::regQParts; idxPart++)
            {
#pragma unroll
                for (uint32_t idxInstK = 0; idxInstK < partNbInstK; idxInstK++)
                {
                    bool const prefetchNextPart = (idxInstK == partNbInstK - 1);
                    uint32_t const idxPartPrefetch = prefetchNextPart ? idxPart + 1 : idxPart;
                    uint32_t const idxInstKPrefetch = prefetchNextPart ? 0 : idxInstK + 1;
                    bool const prefetch = (!prefetchNextPart || (idxPart < nbKParts - 1));

                    if (prefetchNextPart)
                    {
                        kBarWaiter = kBarWaiterNext;
                        kBarWaiterNext = kBarWaiter.next();
                        if (prefetch)
                        {
                            kBarWaiter.wait();
                        }
                    }

                    Mat16x32Loader const loaderK(smem.k[kBarWaiter.idxBuf], 0, idxInstKPrefetch, rB, cB);
#pragma unroll
                    for (uint32_t idxAtomBx2 = 0; idxAtomBx2 < tileNbAtomBx2; idxAtomBx2++)
                    {
                        if (idxAtomBx2 == 2 && prefetch)
                        {
                            if (idxPartPrefetch < SharedMemA::regQParts)
                            {
                                regQBuf[idxInstKPrefetch] = regQ[idxPartPrefetch][idxInstKPrefetch];
                            }
                            else
                            {
                                regQBuf[idxInstKPrefetch]
                                    = loadRegQCol(smem.q[idxPartPrefetch - SharedMemA::regQParts], idxInstKPrefetch);
                            }
                        }
                        AtomBx2 const& atomBx2 = regKBuf[idxInstK][idxAtomBx2];
                        regKBuf[idxInstKPrefetch][idxAtomBx2] = loaderK.load(idxAtomBx2);
                        if (shouldTestWait(idxInstKPrefetch, idxAtomBx2) && prefetch)
                        {
                            kBarWaiterNext.testWait();
                        }
#pragma unroll
                        for (uint32_t i = 0; i < WarpAcc::rows; i++)
                        {
#pragma unroll
                            for (uint32_t j = 0; j < 2; j++)
                            {
                                mma<__nv_fp8_e4m3>(reinterpret_cast<float(&)[2][2]>(acc(i, 2 * idxAtomBx2 + j)),
                                    reinterpret_cast<uint32_t const(&)[2][2]>(regQBuf[idxInstK][i]),
                                    reinterpret_cast<uint32_t const(&)[2][1]>(atomBx2[2 * j]));
                            }
                        }
                        if (prefetch)
                        {
                            regKBuf[idxInstKPrefetch][idxAtomBx2] = loaderK.load(idxAtomBx2);
                        }
                    }
                    if (idxInstKPrefetch == partNbInstK - 1)
                    {
                        assert(prefetch);
                        kBarWaiter.consumed();
                    }
                }
            }
#endif
            if (ctaIter == 0)
            {
                smem.shmQBar.wait_parity(false);
            }
#pragma unroll
            for (uint32_t idxPart = SharedMemA::regQParts; idxPart < nbQParts; idxPart++)
            {
#pragma unroll
                for (uint32_t idxInstK = 0; idxInstK < partNbInstK; idxInstK++)
                {
                    bool const prefetchNextPart = (idxInstK == partNbInstK - 1);
                    uint32_t const idxPartPrefetch = prefetchNextPart ? idxPart + 1 : idxPart;
                    uint32_t const idxInstKPrefetch = prefetchNextPart ? 0 : idxInstK + 1;
                    bool const prefetch = (!prefetchNextPart || (idxPart < nbKParts - 1));

                    if (prefetchNextPart)
                    {
                        kBarWaiter = kBarWaiterNext;
                        kBarWaiterNext = kBarWaiter.next();
                        if (prefetch)
                        {
                            kBarWaiter.wait();
                        }
                    }

                    Mat16x32Loader const loaderK(smem.k[kBarWaiter.idxBuf], 0, idxInstKPrefetch, rB, cB);
#pragma unroll
                    for (uint32_t idxAtomBx2 = 0; idxAtomBx2 < tileNbAtomBx2; idxAtomBx2++)
                    {
                        if (idxAtomBx2 == 2 && prefetch)
                        {
                            regQBuf[idxInstKPrefetch]
                                = loadRegQCol(smem.q[idxPartPrefetch - SharedMemA::regQParts], idxInstKPrefetch);
                        }
                        AtomBx2 const& atomBx2 = regKBuf[idxInstK][idxAtomBx2];
                        if (shouldTestWait(idxInstKPrefetch, idxAtomBx2) && prefetch)
                        {
                            kBarWaiterNext.testWait();
                        }
#pragma unroll
                        for (uint32_t i = 0; i < WarpAcc::rows; i++)
                        {
#pragma unroll
                            for (uint32_t j = 0; j < 2; j++)
                            {
                                mma<__nv_fp8_e4m3>(reinterpret_cast<float(&)[2][2]>(acc(i, 2 * idxAtomBx2 + j)),
                                    reinterpret_cast<uint32_t const(&)[2][2]>(regQBuf[idxInstK][i]),
                                    reinterpret_cast<uint32_t const(&)[2][1]>(atomBx2[2 * j]));
                            }
                        }
                        if (prefetch)
                        {
                            regKBuf[idxInstKPrefetch][idxAtomBx2] = loaderK.load(idxAtomBx2);
                        }
                    }
                    if (idxInstKPrefetch == partNbInstK - 1)
                    {
                        assert(prefetch);
                        kBarWaiter.consumed();
                        if (idxPartPrefetch == nbKParts - 1)
                        {
                            tensorCoreMutex.unlock(); // let the other group to use tensor cores
                        }
                    }
                }
            }
            uint32_t const validTokens = seqLen - tokensPerTile * idxTile;
            if (validTokens < tokensPerTile)
            {
                applyMask(this_warp(), acc, 0, validTokens);
            }
            ThrdRegRowMax rowMaxLog2e;
            WarpAcc const xF32 = scaleAndSoftmax(rowMaxLog2e, acc, grpIdx, grpIter, tileBaseRow);

            auto& xBar = smem.xBars[grpIdx];
            bool const skipXBarWait = xBar.consumed.test_wait_parity(toParity<1>(grpIter));
            // convert to fp8
            WarpAcc const xF32Quant = xF32 * rcpXScale;
            // 0, 1, 8, 9,  2, 3, 10, 11,  4, 5, 12, 13,  6, 7, 14, 15
            Array2D<Array2D<uint32_t, 2, 1>, WarpAcc::rows, exactDiv(WarpAcc::cols, 2)> xF8;
#pragma unroll
            for (uint32_t i = 0; i < WarpAcc::rows; i++)
            {
#pragma unroll
                for (uint32_t m = 0; m < exactDiv(qmmaShape.m, 8); m++)
                {
#pragma unroll
                    for (uint32_t j = 0; j < WarpAcc::cols; j += 2)
                    {
                        auto& dst = reinterpret_cast<__nv_fp8x2_e4m3(&)[2]>(xF8(i, j / 2)(m, 0));
                        dst[0] = __nv_fp8x2_e4m3(float2{xF32Quant(i, j)(m, 0), xF32Quant(i, j)(m, 1)});
                        dst[1] = __nv_fp8x2_e4m3(float2{xF32Quant(i, j + 1)(m, 0), xF32Quant(i, j + 1)(m, 1)});
                    }
                }
            }
            // use tensor core to compute rowSum
            ThrdRegRowMax const rowSum = computeRowSumFromF8
                ? computeRowSumF8<warpTile.y, warpTile.x>(this_warp(), xF8)
                : computeRowSumF32<warpTile.y, warpTile.x>(this_warp(), xF32);

            // store xF8 and rowSum into L2 scratch buffer
            if (!skipXBarWait)
            {
                xBar.consumed.wait_parity(toParity<1>(grpIter));
            }
            storeRowMax<warpTile.y>(smem.x.rowMaxLog2e, rowMaxLog2e, tileBaseRow, lane);
            storeRowMax<warpTile.y>(smem.x.rowSum, rowSum, tileBaseRow, lane);
            storeOrderedXToShm(smem.x.x, xF8, tileBaseRow, lane);
            xBar.produced.arrive();
        }
    }

    __device__ inline WarpAcc scaleAndSoftmax(
        ThrdRegRowMax& rowMaxLog2e, WarpAcc const& acc, uint32_t grpIdx, uint32_t grpIter, uint32_t tileBaseRow);

    __device__ inline void storeOrderedXToShm(XBuffer& dst,
        Array2D<Array2D<uint32_t, 2, 1>, WarpAcc::rows, exactDiv(WarpAcc::cols, 2)> const& src,
        uint32_t const tileBaseRow, uint32_t const lane = laneId());
};

__device__ inline void Producer::loadK()
{
    KVTilePartLoader loader
    {
        args.cacheList, idxReq, args.tensorMapK
#if USE_PAGED_KV_CACHE
            ,
            divUp(seqLen, tokensPerPage)
#endif
    };

#pragma unroll 1
    for (uint32_t iter = 0; true; iter++)
    {
        uint32_t const idxTile = idxTileBeg() + iterStride() * iter;
        if (idxTile >= nbTiles())
        {
            break;
        }
        uint32_t const idxPageBuf = iter % KVTilePartLoader::nbPageBuffers;
        loader.loadPages(idxTile, idxPageBuf);
#pragma unroll 1
        for (uint32_t idxPart = 0; idxPart < nbKParts; idxPart++)
        {
            uint32_t const idxPartGlobal = iter * nbKParts + idxPart;
            uint32_t const idxBuf = idxPartGlobal % SharedMemA::nbKBufs;
            auto& bar = smem.kBars[idxBuf];
            bar.consumed.wait_parity(toParity<SharedMemA::nbKBufs>(idxPartGlobal));
            loader.loadData(smem.k[idxBuf], idxTile, partElemsK * idxPart, bar.produced, idxPageBuf);
            if (warpElectSync())
            {
                bar.produced.arrive_tx(sizeof(SharedMemA::ShmKPart));
            }
        }
    }
}

__device__ inline void Producer::sendX()
{
    // let group 0 to produce first.
    if (warpElectSync())
    {
        smem.xBars[0].consumed.arrive();
    }
    for (uint32_t iter = 0; true; iter++)
    {
        uint32_t const idxTile = idxTileBeg() + iterStride() * iter;
        if (idxTile >= nbTiles())
        {
            break;
        }
        uint32_t const idxBar = iter % SharedMemA::nbXBars;
        auto& xBar = smem.xBars[idxBar];
        xBar.produced.wait_parity(toParity<SharedMemA::nbXBars>(iter));
        smem.cgaXBufConsumed.wait_parity(toParity<1>(iter));
        if (warpElectSync())
        {
            auto& dst = args.cgaXBuf[nbSubSeq * idxInputTokenGlobal + idxSubSeq][ctaRank];
            tma::store1DAsync(&dst, &smem.x, sizeof(CgaXBuffer));
            tma::commitGroup();
            tma::waitGroup<0>();
            // it's turn for the other math group to produce.
            uint32_t const idxBarNext = (iter + 1) % SharedMemA::nbXBars;
            auto& xBarNext = smem.xBars[idxBarNext];
            xBarNext.consumed.arrive();
            asm volatile("fence.release.cluster;\n");
#pragma unroll
            for (uint32_t i = 0; i < nbVSplit; i++)
            {
                auto& producedBar = getConsumerShm(i).cgaXBufProduced[ctaRank];
                producedBar.arrive<Scope::CGA, ArriveOrder::RELAXED>();
            }
        }
    }
}

__device__ inline Producer::WarpAcc Producer::scaleAndSoftmax(
    ThrdRegRowMax& rowMaxLog2e, WarpAcc const& acc, uint32_t grpIdx, uint32_t grpIter, uint32_t tileBaseRow)
{
    uint32_t const ctaIter = grpIdx + grpIter * nbMathGrps;
    uint32_t const cgaIter = ctaRank + ctaIter * nbProducerCtasPerCga;
    auto const warp = this_warp();
    uint32_t const lane = laneId();
    uint32_t const idxProducer = ctaRank;
    assert(ctaRank < nbProducerCtasPerCga);

    float const qkScaleLog2e = smem.qkScaleLog2e;
    bool const skipWaitLastShmRowMax = smem.rowMaxLog2eBar[grpIdx].test_wait_parity(toParity<1>(grpIter));
    QuadRegRowMax const tileRowMaxLog2e = computeRowMax<warpTile.y, warpTile.x>(acc) * qkScaleLog2e;
    // get max with previous CTA's rowMax
    if (!skipWaitLastShmRowMax)
    {
        smem.rowMaxLog2eBar[grpIdx].wait_parity(toParity<1>(grpIter));
    }
    auto const lastRowMaxLog2e = loadShmRowMax<warpTile.y>(smem.rowMaxLog2e, tileBaseRow, lane);

    auto const quadRowMaxLog2e = fmaxf(tileRowMaxLog2e, replicateForQuad(warp, lastRowMaxLog2e));

    // transfer new row max to the other producer CTA for next iteration
    SharedMemA& smemNext = mapa(smem, ctaRank ^ 1U);
    CgaBarrier& nextRowMaxLog2eBar
        = smemNext.rowMaxLog2eBar[(cgaIter + 1) % (nbMathGrps * nbProducerCtasPerCga) / nbMathGrps];
    rowMaxLog2e = dedupFromQuad(warp, quadRowMaxLog2e);
    storeRowMaxAsync<warpTile.y>(nextRowMaxLog2eBar, smemNext.rowMaxLog2e, rowMaxLog2e, tileBaseRow, lane);
    nextRowMaxLog2eBar.arrive_tx_relaxed(sizeof(rowMaxLog2e)); // notify that the next CTA can read rowMax now.

    WarpAcc x;
// apply softmax
#pragma unroll
    for (uint32_t m = 0; m < acc.rows; m++)
    {
#pragma unroll
        for (uint32_t i = 0; i < InstAcc::rows; i++)
        {
            float const maxVal = quadRowMaxLog2e[m * InstAcc::rows + i];
#pragma unroll
            for (uint32_t n = 0; n < acc.cols; n++)
            {
#pragma unroll
                for (uint32_t j = 0; j < InstAcc::cols; j++)
                {
                    float elem = acc(m, n)(i, j);
                    assert(maxVal >= elem * qkScaleLog2e);
                    x(m, n)(i, j) = exp2f(elem * qkScaleLog2e - maxVal);
                }
            }
        }
    }

    return x;
}

__device__ inline void Producer::storeOrderedXToShm(XBuffer& dst,
    Array2D<Array2D<uint32_t, 2, 1>, WarpAcc::rows, exactDiv(WarpAcc::cols, 2)> const& src, uint32_t const tileBaseRow,
    uint32_t const lane)
{
    uint32_t const r = lane % 16;
    uint32_t const c = lane / 16;
    using Src = mha::decay_t<decltype(src)>;
    LdGrain* ptrs[exactDiv(Src::cols, 2)][Src::rows];
#pragma unroll
    for (uint32_t idxInstK = 0; idxInstK < exactDiv(Src::cols, 2); idxInstK++)
    {
        Mat16x32Loader const loader(dst, tileBaseRow, idxInstK, r, c);
#pragma unroll
        for (uint32_t idxInstM = 0; idxInstM < Src::rows; idxInstM++)
        {
            auto const p = const_cast<LdGrain*>(loader.getPtr(idxInstM));
            stmatrix<false, 4>(p, reinterpret_cast<LdGrain const&>(src(idxInstM, idxInstK * 2)));
            ptrs[idxInstK][idxInstM] = p;
        }
    }
    // reorder from 0, 1, 8, 9,  2, 3, 10, 11,  4, 5, 12, 13,  6, 7, 14, 15
    // to 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
    __syncwarp();
#pragma unroll
    for (uint32_t idxInstK = 0; idxInstK < exactDiv(Src::cols, 2); idxInstK++)
    {
#pragma unroll
        for (uint32_t idxInstM = 0; idxInstM < Src::rows; idxInstM++)
        {
            auto const p = ptrs[idxInstK][idxInstM];
            auto const i = *p;
            LdGrain const o = {prmt(i[0], i[1], PermuteOrder{0, 1, 4, 5}), prmt(i[2], i[3], PermuteOrder{0, 1, 4, 5}),
                prmt(i[0], i[1], PermuteOrder{2, 3, 6, 7}), prmt(i[2], i[3], PermuteOrder{2, 3, 6, 7})};
            *p = o;
        }
    }
}

struct Consumer
{
    static inline constexpr uint32_t nbMathWarps = nbMathWarpsB;
    static inline constexpr uint32_t nbMathThrds = warp_size * nbMathWarps;
    static inline constexpr uint2 ctaShape = {2, 4};
    static_assert(SharedMemB::nbAccRowMaxSumCopies == ctaShape.x);
    static_assert(ctaShape.x * ctaShape.y == nbMathWarps);
    static inline constexpr uint2 warpTile = {exactDiv(gemm1V, ctaShape.x), exactDiv(headGrpSize, ctaShape.y)};

    static inline constexpr uint32_t nbWarpOutSwizzleBuf = nbMathWarps;
    using WarpOutSwizzleBuf = Array2D<LdGrain,
        exactDiv(exactDiv(sizeof(SharedMemB::XVBuffer), sizeof(OutputElem) * warpTile.x), nbMathWarps),
        exactDiv(sizeof(OutputElem) * warpTile.x, grainBytes)>;
    static_assert(WarpOutSwizzleBuf::rows % 8 == 0);

    using WarpAcc = WarpAccT<warpTile.y, warpTile.x>;
    using ThrdRegRowMax = ThrdRegRowMaxT<warpTile.y>;
    using UniformNeedRescaleMask = Vec<uint32_t, divUp(warpTile.y, warp_size)>;

    KernelArgs const& args;
    SharedMemB& smem;
    uint32_t const maxNbSubSeq;
    uint32_t const idxReq;
    uint32_t const idxInputTokenGlobal;
    uint32_t const nbSubSeq;
    uint32_t const idxSubSeq;
    uint32_t const seqLen;
    uint32_t const ctaRank;
    uint32_t const warpRank;
    uint2 const warpIdx;

    __device__ inline uint32_t iterStride() const
    {
        return nbSubSeq * nbProducerCtasPerCga;
    }

    __device__ inline uint32_t idxTileBeg() const
    {
        return nbProducerCtasPerCga * idxSubSeq;
    }

    __device__ inline uint32_t nbTiles() const
    {
        return divUp(seqLen, tokensPerTile);
    }

    __device__ inline uint32_t idxConsumer() const
    {
        return ctaRank - 2;
    }

    __device__ inline Consumer(KernelArgs const& args, SharedMemB& smem, uint32_t const maxNbSubSeq,
        uint32_t const idxReq, uint32_t const idxInputTokenGlobal, uint32_t const seqLen, uint32_t const nbSubSeq,
        uint32_t const idxSubSeq, uint32_t ctaRank, uint32_t const warpRank, uint2 const warpIdx)
        : args(args)
        , smem(smem)
        , maxNbSubSeq(maxNbSubSeq)
        , idxReq(idxReq)
        , idxInputTokenGlobal(idxInputTokenGlobal)
        , seqLen(seqLen)
        , nbSubSeq(nbSubSeq)
        , idxSubSeq(idxSubSeq)
        , ctaRank(ctaRank)
        , warpRank(warpRank)
        , warpIdx(warpIdx)
    {
#ifndef NDEBUG
        if (threadIdx.x == 0)
        {
            asm("st.bulk.weak [%0], %1, 0;\n" ::"l"(&smem), "n"(sizeof(SharedMemB)) : "memory");
        }
        __syncthreads();
#endif
        if (threadIdx.x < headGrpSize)
        {
            for (uint32_t i = 0; i < SharedMemB::nbAccRowMaxSumCopies; i++)
            {
                smem.accRowMaxLog2e[i][threadIdx.x] = safeInitRowMax;
                smem.accRowSum[i][threadIdx.x] = 0;
            }
        }
        if (warpElectSync())
        {
            if (warpRank < nbProducerCtasPerCga)
            {
                init(&smem.cgaXBufProduced[warpRank], 1);
            }
            if (warpRank < SharedMemB::nbXBufs)
            {
                auto& bar = smem.xBars[warpRank];
                bar.initialize(1, nbMathThrds);
                bar.consumed.arrive<Scope::CTA, ArriveOrder::RELAXED>(nbMathThrds);
            }
            if (warpRank < SharedMemB::nbVBufs)
            {
                auto& bar = smem.vBars[warpRank];
                bar.initialize(1, nbMathThrds);
                bar.consumed.arrive<Scope::CTA, ArriveOrder::RELAXED>(nbMathThrds);
            }
            if (warpRank == 0)
            {
                init(&smem.mathWarpsBar, warp_size * nbMathWarps);
            }
            if (nbSubSeq > 1 && warpRank < nbMultiBlockBufs)
            {
                auto& b = smem.multiBlockBars[warpRank];
                b.initialize(1, warp_size * multiBlockMathWarps);
                b.consumed.arrive<Scope::CTA, ArriveOrder::RELAXED>(warp_size * multiBlockMathWarps);
            }
        }
        clusterBarArrive();
        clusterBarWait();
    }

    __device__ inline ~Consumer()
    {
        clusterBarArrive();
        clusterBarWait();
        smem.invalidateBarriers(threadIdx.x);
    }

    __device__ inline void run()
    {
        if (warpIdx.y == 2)
        {
            asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" ::"n"(nbRegsForIOWarps));
            if (warpIdx.x == 0)
            {
                loadX();
            }
            else if (warpIdx.x == 1)
            {
                loadV();
            }
        }
        else
        {
            asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" ::"n"(nbRegsForMathWarps));
            compute();
        }
        if (nbSubSeq > 1)
        {
            mergePartialOutputs(args.semaphores[idxInputTokenGlobal],
                reinterpret_cast<Vec<OutputHead, PartialResult::nbRowsPerChunk>&>(
                    args.output[headGrpSize * idxInputTokenGlobal + PartialResult::nbRowsPerChunk * ctaRank]),
                args.partialResults + maxNbSubSeq * idxInputTokenGlobal, nbSubSeq, ctaRank, warpRank, warpIdx, &smem);
        }
    }

    __device__ inline void loadX();
    __device__ inline void loadV();
    __device__ inline void compute();

    __device__ inline uint32_t iterToTile(uint32_t iter) const
    {
        return idxTileBeg() + iterStride() * (iter / 2) + iter % 2;
    }

    __device__ inline SharedMemA& getProducerShm(uint32_t idxProducer) const
    {
        return mapa(reinterpret_cast<SharedMemA&>(smem), idxProducer);
    }

    using WarpOutputTile = Array2D<uint32_t, InstAcc::rows * Consumer::WarpAcc::rows, Consumer::WarpAcc::cols>;
    __device__ inline WarpOutputTile finalize(
        WarpAcc const& acc, ThrdRegRowMax const& accRowSum, float xvScale, uint32_t lane = laneId());
    __device__ inline void storeOutput(Vec<OutputHead, warpTile.y>& dst, uint32_t dstBaseCol,
        WarpOutputTile const& regTile, WarpOutSwizzleBuf& swizzleBuf, uint32_t lane = laneId());
};

__device__ inline void Consumer::compute()
{
    uint2 const tileIdx = {warpIdx.y, warpIdx.x};
    uint2 const tileBase = {tileIdx.x * warpTile.x, tileIdx.y * warpTile.y};

    constexpr uint32_t tileNbInstK = exactDiv(tokensPerTile, qmmaShape.k);
    constexpr uint32_t warpTileNbAtomBx2 = exactDiv(warpTile.x, qmmaShape.n * 2);

    uint32_t const lane = laneId();
    uint32_t const idxHalf = lane / 16;
    uint32_t const laneInHalf = lane % 16;
    uint32_t const rA = laneInHalf;
    uint32_t const cA = idxHalf;
    uint32_t const rB = lane;
    uint32_t const cB = 0;

    WarpAcc acc{};
    uint32_t idxXVBufLast{};
    for (uint32_t iter = 0; true; iter++)
    {
        uint32_t const idxTile = iterToTile(iter);
        if (idxTile >= nbTiles())
        {
            break;
        }

        ThrdRegRowMax accRowMaxLog2e = loadShmRowMax<warpTile.y>(smem.accRowMaxLog2e[tileIdx.x], tileBase.y, lane);
        ThrdRegRowMax accRowSum = loadShmRowMax<warpTile.y>(smem.accRowSum[tileIdx.x], tileBase.y, lane);

        uint32_t const idxXBuf = iter % SharedMemB::nbXBufs;
        uint32_t const idxVBuf = iter % SharedMemB::nbVBufs;
        auto& xBar = smem.xBars[idxXBuf];
        auto& vBar = smem.vBars[idxVBuf];
        // @fixme: merge these two barriers and use test_wait_parity() early to avoid latency.
        bool const skipVBarWait = vBar.produced.test_wait_parity(toParity<SharedMemB::nbVBufs>(iter));
        xBar.produced.wait_parity(toParity<SharedMemB::nbXBufs>(iter));

        ThrdRegRowMax const xRowMaxLog2e = loadShmRowMax<warpTile.y>(smem.xRowMaxLog2e(idxXBuf), tileBase.y, lane);
        assert(all(accRowMaxLog2e <= xRowMaxLog2e));

        auto const needRescaleVec = (xRowMaxLog2e > accRowMaxLog2e);
        UniformNeedRescaleMask rescaleMask{};
#pragma unroll
        for (uint32_t i = 0; i < rescaleMask.size; i++)
        {
            rescaleMask[i] = __ballot_sync(~0U, needRescaleVec[i]);
        }
        bool const anyNeedRescale = any(rescaleMask != UniformNeedRescaleMask::filled(0));
        if (anyNeedRescale)
        {
            auto const scaleVec = exp2f(accRowMaxLog2e - xRowMaxLog2e);
#pragma unroll
            for (uint32_t m = 0; m < WarpAcc::rows; m++)
            {
#pragma unroll
                for (uint32_t i = 0; i < InstAcc::rows; i++)
                {
                    uint8_t const mask = reinterpret_cast<uint8_t const(&)[2][2]>(rescaleMask[m / 2])[m % 2][i];
                    bool const needRescale = (mask != 0);
                    if (needRescale)
                    { // this branch is warp-uniform
                        float const scale = __shfl_sync(~0U, scaleVec[m / 2], 16 * (m % 2) + 8 * i + lane / 4);
#pragma unroll
                        for (uint32_t n = 0; n < WarpAcc::cols; n++)
                        {
#pragma unroll
                            for (uint32_t j = 0; j < InstAcc::cols; j++)
                            {
                                acc(m, n)(i, j) *= scale;
                            }
                        }
                    }
                }
            }
            accRowSum = accRowSum * scaleVec;
        }
        accRowMaxLog2e = xRowMaxLog2e;
        storeRowMax<warpTile.y>(smem.accRowMaxLog2e[tileIdx.x], accRowMaxLog2e, tileBase.y, lane);
        if (!skipVBarWait)
        {
            vBar.produced.wait_parity(toParity<SharedMemB::nbVBufs>(iter));
        }
        auto const& xBuf = smem.x(idxXBuf);
        auto const& vBuf = smem.v(idxVBuf)[tileIdx.x];
        auto const xRowSum = loadShmRowMax<warpTile.y>(smem.xRowSum(idxXBuf), tileBase.y, lane);
        accRowSum = accRowSum + xRowSum;
        storeRowMax<warpTile.y>(smem.accRowSum[tileIdx.x], accRowSum, tileBase.y, lane);

#pragma unroll
        for (uint32_t idxInstK = 0; idxInstK < tileNbInstK; idxInstK++)
        {
            Mat16x32Loader const loaderX(xBuf, tileBase.y, idxInstK, rA, cA);
            Vec<Mat16x32, exactDiv(warpTile.y, qmmaShape.m)> const x = loaderX.loadWholeCol<warpTile.y>();
            using AtomB = Vec<uint32_t, 2>;
#pragma unroll
            for (uint32_t idxAtomBx2 = 0; idxAtomBx2 < warpTileNbAtomBx2; idxAtomBx2++)
            {
                auto const data
                    = ldmatrix_16x16_trans<2>(&vBuf.template at<true>(qmmaShape.k * idxInstK + rB, idxAtomBx2 + cB));
                AtomB const v[2] = {data[0], data[2], data[1], data[3]};
#pragma unroll
                for (uint32_t i = 0; i < WarpAcc::rows; i++)
                {
#pragma unroll
                    for (uint32_t j = 0; j < 2; j++)
                    {
#if 1
                        mma<__nv_fp8_e4m3>(
#else
                        mmaF8_k32_2inst(
#endif
                            reinterpret_cast<float(&)[2][2]>(acc(i, 2 * idxAtomBx2 + j)),
                            reinterpret_cast<uint32_t const(&)[2][2]>(x[i]),
                            reinterpret_cast<uint32_t const(&)[2][1]>(v[j]));
                    }
                }
            }
        }
        bool const isLastIter = (iterToTile(iter + 1) >= nbTiles());
        if (isLastIter)
        {
            idxXVBufLast = idxXBuf;
            assert(idxXBuf == idxVBuf);
        }
        else
        {
            xBar.consumed.arrive();
            vBar.consumed.arrive();
        }
    }

    smem.mathWarpsBar.arrive();

    ThrdRegRowMax const accRowSum = loadShmRowMax<warpTile.y>(smem.accRowSum[tileIdx.x], tileBase.y, lane);
    float const xvScale = computeRowSumFromF8 ? args.kvCacheScale[0] : args.kvCacheScale[0] * xScale;
    WarpOutputTile const output = finalize(acc, accRowSum, xvScale, lane);

    bool const isMultiBlockMode = (nbSubSeq != 1);
    static_assert(PartialResult::nbRowsPerChunk == warpTile.y);
    auto& dst = isMultiBlockMode
        ? args.partialResults[maxNbSubSeq * idxInputTokenGlobal + idxSubSeq].chunks[tileIdx.y].data
        : reinterpret_cast<Vec<OutputHead, warpTile.y>&>(args.output[headGrpSize * idxInputTokenGlobal + tileBase.y]);

    assert(warpRank < nbMathWarps);
    WarpOutSwizzleBuf& swizzleBuf
        = reinterpret_cast<Vec<WarpOutSwizzleBuf, nbWarpOutSwizzleBuf>&>(smem.xv[idxXVBufLast])[warpRank];
    // make sure all math warps have finished using XVBuffer.
    smem.mathWarpsBar.wait_parity(false);

    storeOutput(dst, gemm1V * idxConsumer() + tileBase.x, output, swizzleBuf, lane);
    if (isMultiBlockMode && tileIdx.x == 0)
    {
        ThrdRegRowMax const accRowMaxLog2e
            = loadShmRowMax<warpTile.y>(smem.accRowMaxLog2e[tileIdx.x], tileBase.y, lane);
        auto& chunk = args.partialResults[maxNbSubSeq * idxInputTokenGlobal + idxSubSeq].chunks[tileIdx.y];
#pragma unroll
        for (uint32_t i = 0; i < ThrdRegRowMax::size; i++)
        {
            chunk.rowMaxLog2e[warp_size * i + lane] = accRowMaxLog2e[i];
            chunk.rowSum[warp_size * i + lane] = accRowSum[i];
        }
    }
    smem.xBars[idxXVBufLast].consumed.arrive();
    smem.vBars[idxXVBufLast].consumed.arrive();
}

__device__ inline void Consumer::loadX()
{
#pragma unroll 1
    for (uint32_t iter = 0; true; iter++)
    {
        uint32_t const idxTile = iterToTile(iter);
        if (idxTile >= nbTiles())
        {
            break;
        }
        // @todo: merge these two barriers.
        uint32_t const idxScratchXBuf = iter % nbProducerCtasPerCga;
        auto& srcProducedBar = smem.cgaXBufProduced[idxScratchXBuf];
        srcProducedBar.wait_parity(toParity<nbProducerCtasPerCga>(iter));
        uint32_t const idxXBuf = iter % SharedMemB::nbXBufs;
        auto& xBar = smem.xBars[idxXBuf];
        xBar.consumed.wait_parity(toParity<SharedMemB::nbXBufs>(iter));
        if (warpElectSync())
        {
            auto& src = args.cgaXBuf[nbSubSeq * idxInputTokenGlobal + idxSubSeq][idxScratchXBuf];
            auto& dst = smem.xv[idxXBuf].x;
            tma::loadLinearAsync(&dst, &src.x, sizeof(CgaXBuffer), xBar.produced);
            xBar.produced.arrive_tx(sizeof(CgaXBuffer));
            xBar.produced.wait_parity(toParity<SharedMemB::nbXBufs>(iter));
            uint32_t const idxProducer = idxScratchXBuf;
            // @fixme: check if this works. If it doesn't, randomly pick some data from dstX and dstRowSum and use
            // STAS + arrive_tx to avoid fence.
            getProducerShm(idxProducer).cgaXBufConsumed.arrive<Scope::CGA, ArriveOrder::RELAXED>();
        }
    }
}

__device__ inline void Consumer::loadV()
{
    KVTilePartLoader loader(args.cacheList, idxReq, args.tensorMapV
#if USE_PAGED_KV_CACHE
        ,
        divUp(seqLen, tokensPerPage)
#endif
    );
    for (uint32_t iter = 0; true; iter++)
    {
        uint32_t const idxTile = iterToTile(iter);
        if (idxTile >= nbTiles())
        {
            break;
        }
        uint32_t const idxPageBuf = iter % KVTilePartLoader::nbPageBuffers;
        loader.loadPages(idxTile, idxPageBuf);
        uint32_t const idxVBuf = iter % SharedMemB::nbVBufs;
        auto& vBar = smem.vBars[idxVBuf];
        vBar.consumed.wait_parity(toParity<SharedMemB::nbVBufs>(iter));
#pragma unroll
        for (uint32_t idxPart = 0; idxPart < SharedMemB::VBuffer::size; idxPart++)
        {
            loader.loadData(smem.v(idxVBuf)[idxPart], idxTile,
                gemm1V * idxConsumer() + exactDiv(gemm1V, SharedMemB::VBuffer::size) * idxPart, vBar.produced,
                idxPageBuf);
        }
        if (warpElectSync())
        {
            vBar.produced.arrive_tx(sizeof(SharedMemB::VBuffer));
        }
    }
}

__device__ inline Array2D<uint32_t, InstAcc::rows * Consumer::WarpAcc::rows, Consumer::WarpAcc::cols>
Consumer::finalize(WarpAcc const& acc, ThrdRegRowMax const& accRowSum, float const xvScale, uint32_t const lane)
{
    ThrdRegRowMax const scaleVec = 1.F / (accRowSum) *xvScale;
    WarpOutputTile ret;
#pragma unroll
    for (uint32_t m = 0; m < WarpAcc::rows; m++)
    {
#pragma unroll
        for (uint32_t i = 0; i < InstAcc::rows; i++)
        {
            uint32_t retRow = m * InstAcc::rows + i;
            float const scale = __shfl_sync(~0U, scaleVec[m / 2], 16 * (m % 2) + 8 * i + lane / 4);
#pragma unroll
            for (uint32_t n = 0; n < WarpAcc::cols; n++)
            {
                float data[InstAcc::cols];
#pragma unroll
                for (uint32_t j = 0; j < InstAcc::cols; j++)
                {
                    data[j] = acc(m, n)(i, j) * scale;
                }
                assert(InstAcc::cols == 2);
                reinterpret_cast<__nv_bfloat162&>(ret(retRow, n)) = __float22bfloat162_rn(float2{data[0], data[1]});
            }
        }
    }
    return ret;
}

__device__ inline void Consumer::storeOutput(Vec<OutputHead, warpTile.y>& dst, uint32_t dstBaseCol,
    WarpOutputTile const& src, WarpOutSwizzleBuf& swizzleBuf, uint32_t lane)
{
    using Dst = mha::decay_t<decltype(dst)>;
    static_assert(Dst::size == WarpOutputTile::rows * 8 && Dst::size % WarpOutSwizzleBuf::rows == 0);
    uint32_t const nbIters = exactDiv(Dst::size, WarpOutSwizzleBuf::rows);

    uint32_t const rS = lane % 8;
    uint32_t const cS = lane / 8;

    uint32_t const thrdsPerRow = exactDiv(sizeof(WarpOutSwizzleBuf::Elem) * WarpOutSwizzleBuf::cols, grainBytes);
    static_assert(thrdsPerRow <= 32);
    uint32_t const rL = lane / thrdsPerRow;
    uint32_t const cL = lane % thrdsPerRow;
#pragma unroll
    for (uint32_t iter = 0; iter < nbIters; iter++)
    {
#pragma unroll
        for (uint32_t j = 0; j < WarpOutputTile::cols; j += 4)
        {
            auto const baseSwzPtr = &swizzleBuf.template at<true>(rS, j + cS);
            constexpr uint32_t srcRowsPerIter = exactDiv(WarpOutputTile::rows, nbIters);
#pragma unroll
            for (uint32_t i = 0; i < srcRowsPerIter; i++)
            {
                static_assert(sizeof(WarpOutSwizzleBuf::Elem) * WarpOutSwizzleBuf::cols * 8 % 1024 == 0);
                auto const swzPtr = checkedVal(
                    baseSwzPtr + WarpOutputTile::cols * 8 * i, &swizzleBuf.template at<true>(8 * i + rS, j + cS));
                stmatrix<false, 4>(
                    swzPtr, reinterpret_cast<Vec<uint32_t, 4> const&>(src(srcRowsPerIter * iter + i, j)));
            }
        }
        __syncwarp();

        uint32_t const dstRowsPerIter = WarpOutSwizzleBuf::rows;
        uint32_t const rowsPerOp = exactDiv(warp_size, thrdsPerRow);
        LdGrain* const baseDstPtr = reinterpret_cast<LdGrain*>(
            &dst[dstRowsPerIter * iter + rL][dstBaseCol + exactDiv(grainBytes, sizeof(OutputElem)) * cL]);
#pragma unroll
        for (uint32_t i = 0; i < dstRowsPerIter; i += rowsPerOp)
        {
            LdGrain* const dstPtr = checkedVal(baseDstPtr + i * exactDiv(sizeof(OutputHead), grainBytes),
                reinterpret_cast<LdGrain*>(
                    &dst[dstRowsPerIter * iter + i + rL][dstBaseCol + exactDiv(grainBytes, sizeof(OutputElem)) * cL]));
            LdGrain* const srcPtr = &swizzleBuf.template at<true>(i + rL, cL);
            *dstPtr = *srcPtr;
        }
        __syncwarp();
    }
}

__device__ inline void mergePartialOutputs(uint32_t& semaphore, Vec<OutputHead, PartialResult::nbRowsPerChunk>& dst,
    PartialResult const* reqPartialResults, uint32_t nbSubSeq, uint32_t ctaRank, uint32_t warpRank, uint2 warpIdx,
    void* sharedMem)
{
    assert(nbSubSeq > 1);
    clusterBarArrive();
    clusterBarWait();
    bool const isProducer = (ctaRank < nbProducerCtasPerCga);

    bool& shmIsLastSubSeq = isProducer ? static_cast<SharedMemA*>(sharedMem)->isLastSubSeq
                                       : static_cast<SharedMemB*>(sharedMem)->isLastSubSeq;

    if (ctaRank == 3 && threadIdx.x == 0)
    {
        uint32_t old;
        uint32_t const lastOld = nbSubSeq - 1;
        asm volatile("atom.relaxed.gpu.global.inc.u32 %0, [%1], %2;\n" : "=r"(old) : "l"(&semaphore), "r"(lastOld));
        bool const isLastSubSeq = (old == lastOld);
#pragma unroll
        for (uint32_t i = 0; i < nbProducerCtasPerCga; i++)
        {
            static_cast<SharedMemA*>(mapa(sharedMem, i))->isLastSubSeq = isLastSubSeq;
        }
        mapa(shmIsLastSubSeq, 2) = isLastSubSeq;
        shmIsLastSubSeq = isLastSubSeq;
    }
    clusterBarArrive();
    clusterBarWait();
    bool const isLastCga = shmIsLastSubSeq;
    if (!isLastCga)
    {
        return;
    }

    CtaBarrierPair(&bars)[nbMultiBlockBufs] = isProducer ? static_cast<SharedMemA*>(sharedMem)->multiBlockBars
                                                         : static_cast<SharedMemB*>(sharedMem)->multiBlockBars;
    Vec<PartialResult::Chunk, nbMultiBlockBufs>& shmBufs = isProducer
        ? static_cast<SharedMemA*>(sharedMem)->getMultiBlockBufs()
        : static_cast<SharedMemB*>(sharedMem)->getMultiBlockBufs();

    constexpr uint32_t nbShmBufs = nbMultiBlockBufs;

    if (warpIdx.y == 2)
    {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" ::"n"(nbRegsForIOWarps));
        if (warpIdx.x == 0)
        {
#pragma unroll 1
            for (uint32_t idxSubSeq = 0; idxSubSeq < nbSubSeq; idxSubSeq++)
            {
                uint32_t const idxBuf = idxSubSeq % nbShmBufs;
                auto& bar = bars[idxBuf];
                bar.consumed.wait_parity(toParity<nbShmBufs>(idxSubSeq));
                if (warpElectSync())
                {
                    tma::loadLinearAsync(&shmBufs[idxBuf], &reqPartialResults[idxSubSeq].chunks[ctaRank],
                        sizeof(PartialResult::Chunk), bar.produced);
                    bar.produced.arrive_tx(sizeof(PartialResult::Chunk));
                }
            }
        }
    }
    else
    {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" ::"n"(nbRegsForMathWarps));
        constexpr uint32_t nbMathWarps = 8;
        constexpr uint32_t rowsPerWarp = exactDiv(PartialResult::nbRowsPerChunk, nbMathWarps);
        constexpr uint32_t regGrainsPerRow = exactDiv(sizeof(OutputHead), grainBytes * warp_size);
        constexpr uint32_t grainOutElems = exactDiv(grainBytes, sizeof(OutputElem));
        uint32_t const lane = laneId();

        uint32_t const tileRowBase = rowsPerWarp * warpRank;
        using RowWise = Vec<float, rowsPerWarp>;
        using RegChunk = Array2D<Vec<OutputElem, grainOutElems>, rowsPerWarp, regGrainsPerRow>;
        auto loadBuf = [&](RowWise& rowMaxLog2e, RowWise& rowSum, RegChunk& regChunk, PartialResult::Chunk const& chunk)
        {
            auto loadRowWise = [&](Vec<float, PartialResult::nbRowsPerChunk> const& src)
            { return reinterpret_cast<RowWise const&>(src[tileRowBase]); };
            rowMaxLog2e = loadRowWise(chunk.rowMaxLog2e);
            rowSum = loadRowWise(chunk.rowSum);
            regChunk;
#pragma unroll
            for (uint32_t i = 0; i < rowsPerWarp; i++)
            {
#pragma unroll
                for (uint32_t j = 0; j < regGrainsPerRow; j++)
                {
                    regChunk(i, j) = reinterpret_cast<Vec<OutputElem, grainOutElems> const&>(
                        chunk.data[tileRowBase + i][grainOutElems * (warp_size * j + lane)]);
                }
            }
        };

        uint32_t const idxSubSeqInit = 0;
        uint32_t const idxBufInit = idxSubSeqInit % nbShmBufs;
        bars[idxBufInit].produced.wait_parity(toParity<nbShmBufs>(idxSubSeqInit));
        RowWise accRowMaxLog2e;
        RowWise accRowSum;
        RegChunk chunk;
        loadBuf(accRowMaxLog2e, accRowSum, chunk, shmBufs[idxBufInit]);
        bars[idxBufInit].consumed.arrive();

        using Acc = Array2D<Vec<float, grainOutElems>, rowsPerWarp, regGrainsPerRow>;
        Acc acc;
#pragma unroll
        for (uint32_t i = 0; i < rowsPerWarp; i++)
        {
#pragma unroll
            for (uint32_t j = 0; j < regGrainsPerRow; j++)
            {
                acc(i, j) = convert<float>(chunk(i, j)) * accRowSum[i];
            }
        }

#pragma unroll 1
        for (uint32_t idxSubSeq = idxSubSeqInit + 1; idxSubSeq < nbSubSeq; idxSubSeq++)
        {
            uint32_t const idxBuf = idxSubSeq % nbShmBufs;
            auto& bar = bars[idxBuf];
            bar.produced.wait_parity(toParity<nbShmBufs>(idxSubSeq));
            RowWise chunkRowMaxLog2e;
            RowWise chunkRowSum;
            loadBuf(chunkRowMaxLog2e, chunkRowSum, chunk, shmBufs[idxBuf]);
            bar.consumed.arrive();
#pragma unroll
            for (uint32_t i = 0; i < rowsPerWarp; i++)
            {
                bool const newChunkGreater = (chunkRowMaxLog2e[i] > accRowMaxLog2e[i]);
                if (newChunkGreater)
                {
                    float const scale = exp2f(accRowMaxLog2e[i] - chunkRowMaxLog2e[i]);
#pragma unroll
                    for (uint32_t j = 0; j < regGrainsPerRow; j++)
                    {
                        acc(i, j) = acc(i, j) * scale + convert<float>(chunk(i, j)) * chunkRowSum[i];
                    }
                    accRowSum[i] = accRowSum[i] * scale + chunkRowSum[i];
                    accRowMaxLog2e[i] = chunkRowMaxLog2e[i];
                }
                else
                {
                    float const scale = exp2f(chunkRowMaxLog2e[i] - accRowMaxLog2e[i]);
                    float const fusedScale = scale * chunkRowSum[i];
#pragma unroll
                    for (uint32_t j = 0; j < regGrainsPerRow; j++)
                    {
                        acc(i, j) = acc(i, j) + convert<float>(chunk(i, j)) * fusedScale;
                    }
                    accRowSum[i] = accRowSum[i] + chunkRowSum[i] * scale;
                }
            }
        }

#pragma unroll
        for (uint32_t i = 0; i < rowsPerWarp; i++)
        {
            float const scale = 1.F / accRowSum[i];
            auto const dstHead = reinterpret_cast<Vec<OutputElem, grainOutElems>*>(&dst[tileRowBase + i]);
#pragma unroll
            for (uint32_t j = 0; j < regGrainsPerRow; j++)
            {
                dstHead[warp_size * j + lane] = convert<OutputElem>(acc(i, j) * scale);
            }
        }
    }
}

inline constexpr uint32_t cgaSize = nbProducerCtasPerCga + nbVSplit;

CUBIN_EXPORT __global__ __launch_bounds__(32 * 4 * 3, 1) __cluster_dims__(cgaSize, 1, 1) void kernel_mha(
    __grid_constant__ CUtensorMap const tensorMapQ, // MhaIOHead[nbQHeads * totalNbInputTokens],
    __grid_constant__ CUtensorMap const tensorMapK, // with box=64 for the least significant dim
    __grid_constant__ CUtensorMap const tensorMapV, // with box=128 for the least significant dim
    float const qScale,
    OutputHead* __restrict__ const output,          // [totalNbIntputTokens][nbQHeads]
    KVCacheList<usePagedKVCache> const cacheList, uint32_t const batchSize,
    float const* __restrict__ const kvCacheScale,   // Device memory scalar. Same scale for K and V cache. Used only for
                                                    // int8/fp8 KV cache.
    Vec<CgaXBuffer, nbProducerCtasPerCga>* __restrict__ const cgaXBuf, // [totalNbInputTokens][maxNbSubSeq]
    uint32_t* __restrict__ const semaphores = nullptr,                 // [totalNbInputTokens]
    PartialResult* __restrict__ const partialResults = nullptr)        // [totalNbInputTokens][maxNbSubSeq]
{
    assert(blockDim.x == 32 * 12 && blockDim.y == 1 && blockDim.z == 1);
    extern __shared__ char smemBuf[];
    uint32_t const warpRank = makeWarpUniform(this_warp(), threadIdx.x / warp_size);
    uint2 const warpIdx = {warpRank % 4, warpRank / 4};

    uint3 const& cgaId = clusterId();
    uint32_t const& idxReq = cgaId.z;
    uint32_t const& maxNbSubSeq = nbClusters().y;
    uint32_t const& idxSubSeq = cgaId.y;
    uint32_t const inputSeqLen
        = (allowMultipleInputTokens ? exactDiv(gridDim.x, cgaSize) : checkedVal(1U, exactDiv(gridDim.x, cgaSize)));
    uint32_t const reqIdxInputToken
        = (allowMultipleInputTokens ? blockIdx.x / cgaSize : checkedVal(0U, blockIdx.x / cgaSize));
    uint32_t const idxInputTokenGlobal = inputSeqLen * idxReq + reqIdxInputToken;
    uint32_t const cacheSeqLen = cacheList.seqLenList[idxReq] - (inputSeqLen - 1) + reqIdxInputToken;
    assert(beamWidth == 1);
    uint32_t const nbTiles = useKVCache ? divUp(cacheSeqLen, tokensPerTile) : 0;
    bool const isMultiBlockMode = (maxNbSubSeq > 1 && nbTiles >= multiBlockMinNbTiles);
    uint32_t const nbSubSeq = isMultiBlockMode ? mha::min(nbTiles / multiBlockMinNbTilesPerCta, maxNbSubSeq) : 1;
    static_assert(multiBlockMinNbTiles >= multiBlockMinNbTilesPerCta * 2);
    assert(isMultiBlockMode == (nbSubSeq > 1));
    if (idxSubSeq >= nbSubSeq)
    {
        return;
    }

    uint32_t const ctaRank = clusterCtaRank();
    bool const isProducer = (ctaRank < nbProducerCtasPerCga);

    KernelArgs const args{tensorMapQ, tensorMapK, tensorMapV, qScale, output, cacheList, batchSize, kvCacheScale,
        cgaXBuf, semaphores, partialResults};

    if (isProducer)
    {
        Producer{args, *reinterpret_cast<SharedMemA*>(smemBuf), maxNbSubSeq, idxReq, idxInputTokenGlobal, cacheSeqLen,
            nbSubSeq, idxSubSeq, ctaRank, warpRank, warpIdx}
            .run();
    }
    else
    {
        Consumer{args, *reinterpret_cast<SharedMemB*>(smemBuf), maxNbSubSeq, idxReq, idxInputTokenGlobal, cacheSeqLen,
            nbSubSeq, idxSubSeq, ctaRank, warpRank, warpIdx}
            .run();
    }
}

__constant__ constexpr uint32_t smemSize = mha::max(sizeof(SharedMemA), sizeof(SharedMemB));
static_assert(smemSize <= 99 * 1024, "Shared memory size exceeded");
#endif // is_MLA

#ifndef GENERATE_CUBIN
#if IS_MLA
CUtensorMap makeTensorMapForQ(
    void const* addr, CUtensorMapDataType_enum dataType, uint32_t headElems, uint32_t totalNbHeads, uint32_t partElems)
{
    CUtensorMap tensorMap{};
    uint64_t const globalDims[] = {headElems, totalNbHeads};
    uint32_t elemBytes = getElemBytes(dataType);
    uint32_t const headBytes = elemBytes * headElems;
    uint64_t const globalStrides[] = {headBytes};
    uint32_t const boxDims[] = {partElems, headGrpSize};
    uint32_t const elemStrides[] = {1, 1};
    auto const swizzle = CU_TENSOR_MAP_SWIZZLE_64B;

    checkCu(cuTensorMapEncodeTiled(&tensorMap, dataType, 2, const_cast<void*>(addr), globalDims, globalStrides, boxDims,
        elemStrides, CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle, CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    return tensorMap;
}
#endif // IS_MLA

void launchMLA(cudaDeviceProp const& prop,
    uint32_t inputSeqLen, // uniform for all requests and causal mask is assumed
    float qScale, OutputHead* output, InputHead const* q,
#if USE_PAGED_KV_CACHE
#if PAGED_KV_CACHE_LAYOUT == 1
    GMemCacheHead* kCacheVLLM, // K cache pool for VLLM layout
    GMemCacheHead* vCacheVLLM, // V cache pool for VLLM layout
#else
    GMemCacheHead* pool, // global pool of pages
#endif
    KVCachePageIndex const*
        kvCachePageList, // device pointer. shape: KVCachePage[batchSize][beamWidth][2][maxNbPagesPerSeq] (Layout 0) or
                         // [batchSize][maxNbPagesPerSeq] (Layout 1)
#else
    GMemKVCacheHead* kvCacheData,
#endif
    uint32_t maxSeqLen, uint32_t const* seqLen, uint32_t batchSize,
    float const* __restrict__ kvCacheScale, // Device memory scalar. Same scale for K and V cache. Used only for
                                            // int8/fp8 KV cache.
    uint32_t* semaphores, void* scratch, cudaStream_t stream)
{
#if IS_MLA
    static_assert(
        SLIDING_WINDOW == 0 && LOW_PREC_OUTPUT == 0 && USE_INPUT_KV == 0 && USE_BEAM_SEARCH == 0, "not implemented");
    if (beamWidth != 1)
    {
        throw std::runtime_error("not implemented");
    }
    static uint32_t const hostSmemSize = [&]()
    {
        // printf("smemSize = %u\n", smemSize);
        uint32_t size;
        checkCuda(cudaMemcpyFromSymbol(&size, smemSize, sizeof(smemSize)));
        checkCuda(cudaFuncSetAttribute(kernel_mha, cudaFuncAttributeMaxDynamicSharedMemorySize, size));
        return size;
    }();
    uint32_t const nbKHeads = 1;
    uint32_t const nbVHeads = nbKHeads;
    uint32_t const nbQHeads = nbKHeads * headGrpSize;
    uint32_t const nbQKVHeads = nbQHeads + nbKHeads + nbVHeads;
    uint32_t const nbSubSeqPerSeq = [&]() -> uint32_t
    {
        auto const env = std::getenv("XQA_NB_SUB_SEQ");
        if (env != nullptr)
        {
            int32_t const val = std::stoi(env);
            if (val > 0)
            {
                return val;
            }
        }
        float const factor = 4.f;
        return mha::min<uint32_t>(
            mha::max<uint32_t>(1U, (uint32_t) round(prop.multiProcessorCount / 4 / (batchSize * nbKHeads) * factor)),
            divUp(maxSeqLen, tokensPerTile * 2));
    }();
    // printf("nbSubSeqPerSeq = %u\n", nbSubSeqPerSeq);
    // gridDim.z == nbKHeads * batchSize && gridDim.y == nbSubSeqPerSeq && gridDim.x == nbInputSeqSplit
    dim3 const dimGrid{4 * inputSeqLen, nbSubSeqPerSeq, nbKHeads * batchSize};
    dim3 const dimCta{warp_size * 4 * 3, 1, 1};
    auto const launchCfg = makeLaunchConfig(dimGrid, dimCta, hostSmemSize, stream, ENABLE_PDL != 0);
#if USE_PAGED_KV_CACHE
    uint32_t const maxNbPagesPerSeq = exactDiv(maxSeqLen, tokensPerPage);
#if PAGED_KV_CACHE_LAYOUT == 1
    KVCacheList<true> const cacheList{kCacheVLLM, vCacheVLLM, kvCachePageList, seqLen, maxNbPagesPerSeq};
#else
    KVCacheList<true> const cacheList{pool, kvCachePageList, seqLen, maxNbPagesPerSeq};
#endif
    auto const dtype = []
    {
        if (std::is_same_v<CacheElem, half>)
        {
            return CU_TENSOR_MAP_DATA_TYPE_FLOAT16;
        }
        else if (std::is_same_v<CacheElem, __nv_bfloat16>)
        {
            return CU_TENSOR_MAP_DATA_TYPE_BFLOAT16;
        }
        else if (std::is_same_v<CacheElem, __nv_fp8_e4m3>)
        {
            return CU_TENSOR_MAP_DATA_TYPE_UINT8;
        }
        throw std::runtime_error("unsupported cache element type");
    }();

    auto const tensorMapQ
        = makeTensorMapForQ(q, dtype, validElemsPerHead, headGrpSize * inputSeqLen * batchSize, partElemsK);
#if PAGED_KV_CACHE_LAYOUT == 1
    auto const tensorMapK = makeTensorMapForPagedKVCache(
        kCacheVLLM, dtype, validElemsPerHead, nbKHeads, tokensPerPage, partElemsK, tokensPerTile);
    auto const tensorMapV = makeTensorMapForPagedKVCache(
        vCacheVLLM, dtype, validElemsPerHead, nbKHeads, tokensPerPage, partElemsV, tokensPerTile);
#else
    auto const tensorMapK = makeTensorMapForPagedKVCache(
        pool, dtype, validElemsPerHead, nbKHeads, tokensPerPage, partElemsK, tokensPerTile);
    auto const tensorMapV = makeTensorMapForPagedKVCache(
        pool, dtype, validElemsPerHead, nbKHeads, tokensPerPage, partElemsV, tokensPerTile);
#endif

    uint32_t const nbCgas = exactDiv(dimGrid.x, 4) * dimGrid.y * dimGrid.z;
    auto const cgaXBuf = static_cast<Vec<CgaXBuffer, nbProducerCtasPerCga>*>(scratch);
    auto const partialResults = reinterpret_cast<PartialResult*>(cgaXBuf + nbCgas);
    cudaError_t const err = cudaLaunchKernelEx(&launchCfg, &kernel_mha, tensorMapQ, tensorMapK, tensorMapV, qScale,
        output, cacheList, batchSize, kvCacheScale, cgaXBuf, semaphores, partialResults);
#else
    KVCacheList<false> const cacheList{kvCacheData, seqLen, maxSeqLen};
    static_assert(!usePagedKVCache);
    assert(gemm0CtaTileNbTokens == gemm1CtaTileNbTokens);
    auto const tensorMap = makeTensorMapForContiguousKVCache(kvCacheData, CU_TENSOR_MAP_DATA_TYPE_UINT8,
        validElemsPerHead, nbKHeads, maxSeqLen, beamWidth, batchSize, gemm0CtaTileNbTokens);
    cudaLaunchKernelEx(&launchCfg, kernel_mha, nbKHeads,
#if SLIDING_WINDOW
        slidingWinSize,
#endif
        qScale, output,
#if LOW_PREC_OUTPUT
        rcpOutScale,
#endif
#if USE_INPUT_KV
        qkv,
#if ROPE_STYLE != 0
        ropeCosSin,
#endif
#else
        q,
#endif
        cacheList,
#if USE_BEAM_SEARCH
        beamSearchParams,
#endif
        batchSize, kvCacheScale, tensorMap, semaphores, scratch);
#endif
    checkCuda(err);
#endif
}
#endif
