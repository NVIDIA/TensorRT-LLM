/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: NVIDIA TensorRT Source Code License Agreement
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#ifndef ENABLE_NVRTC
#define ENABLE_NVRTC 1
#endif

#include <gtest/gtest.h>
#if ENABLE_NVRTC
#include "generated/xqa_sources.h"
#include <nvrtc.h>
#endif
#include "../defines.h"
#include "../mha.h"
#include "../utils.h"
#include "refAttention.h"

#include <algorithm>
#include <array>
#include <fstream>
#include <future>
#include <limits>
#include <nvtx3/nvToolsExt.h>
#include <random>
#include <thread>

#ifdef NDEBUG
#define USE_SMALL_IO 0
#else
#define USE_SMALL_IO 1
#endif

void warmup(cudaDeviceProp const& prop, float ms, cudaStream_t stream = nullptr);
bool const isTracing = []()
{
    auto const v = std::getenv("XQA_IS_TRACING");
    if (!v)
    {
        return false;
    }
    return bool(std::stoi(v));
}();

template <typename T>
class ManagedMemBuf
{
public:
    ManagedMemBuf(size_t nbElems)
        : mSize{nbElems}
    {
        if (nbElems != 0)
        {
            void* p;
            checkCuda(cudaMallocManaged(&p, sizeof(T) * nbElems));
            mData.reset(reinterpret_cast<T*>(p));
        }
    }

    T* get() const
    {
        return mData.get();
    }

    size_t size() const
    {
        return mSize;
    }

    void prefetch(int dstDevice, cudaStream_t stream = nullptr) const
    {
        if (!isTracing)
        {
            checkCuda(cudaMemPrefetchAsync(get(), sizeof(T) * size(), dstDevice, stream));
        }
    }

    T& operator[](size_t i) const
    {
        return mData[i];
    };

private:
    struct CudaDeleter
    {
        void operator()(void* p) const
        {
            cudaFree(p);
        }
    };

    std::unique_ptr<T[], CudaDeleter> mData;
    size_t mSize;
};

template <typename D, typename S>
void save(char const* file, S const* src, size_t size)
{
    std::ofstream fout{file, std::ios::trunc};
    for (size_t i = 0; i < size; i++)
    {
        D data{src[i]};
        fout.write((char const*) &data, sizeof(D));
    }
    fout.close();
}

template <class T>
inline void hash_combine(std::size_t& seed, T const& v)
{
    std::hash<T> hasher;
    seed ^= hasher(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

#if IS_MLA
template <uint32_t nbKHeads, uint32_t runtimeHeadGrpSize = headGrpSize, uint32_t qSeqLen = 1>
#else
#if SPEC_DEC
template <uint32_t nbKHeads, uint32_t runtimeHeadGrpSize, uint32_t qSeqLen>
#else
template <uint32_t nbKHeads>
#endif
#endif
void runTest(uint32_t batchSize, uint32_t seqLen, bool testPerf, bool refCheck, bool verbose = false,
    bool saveData = false, uint32_t ctxLen = ~0U, uint32_t slidingWinSize = std::numeric_limits<uint32_t>::max())
{
#if IS_MLA
    if (nbKHeads != 1)
    {
        GTEST_SKIP() << "MLA only supports 1 K head";
    }
#endif
    constexpr uint32_t nbVHeads = nbKHeads;
#if SPEC_DEC
    assert(qSeqLen <= seqLen);
    constexpr uint32_t nbQHeads = nbKHeads * runtimeHeadGrpSize;
#if IS_MLA
    constexpr uint32_t nbBlocksPerGrp = qSeqLen;
#else
    constexpr uint32_t nbBlocksPerGrpMMA = divUp(qSeqLen * runtimeHeadGrpSize, rowsPerBlock);
    constexpr uint32_t nbBlocksPerGrpGMMA = divUp(qSeqLen, rowsPerBlock / runtimeHeadGrpSize);
    constexpr uint32_t nbBlocksPerGrp = std::max(nbBlocksPerGrpMMA, nbBlocksPerGrpGMMA);
#endif // IS_MLA
#else
    constexpr uint32_t nbQHeads = nbKHeads * headGrpSize;
#if !(IS_MLA)
    constexpr uint32_t qSeqLen = 1;
#endif
#endif

#if !(SLIDING_WINDOW)
    assert(slidingWinSize >= seqLen);
#endif

    checkCuda(cudaFree(nullptr));
    int device;
    checkCuda(cudaGetDevice(&device));
    cudaDeviceProp prop;
    checkCuda(cudaGetDeviceProperties(&prop, device));
    if (verbose)
    {
        printf("SM count: %d\n", prop.multiProcessorCount);
        if (!refCheck && (batchSize * nbKHeads) % prop.multiProcessorCount != 0)
        {
            printf("Tail effect will impact performance.\n");
        }
    }

    bool const useQGMMA = [&]() -> bool
    {
        if (std::getenv("XQA_USE_QGMMA"))
        {
            return std::stoi(std::getenv("XQA_USE_QGMMA")) != 0;
        }
        if (beamWidth != 1 || prop.minor != 0)
        {
            return false;
        }
        if (prop.major == 9)
        {
            return std::is_same_v<CacheElem, __nv_fp8_e4m3> || std::is_same_v<CacheElem, half>
                || std::is_same_v<CacheElem, __nv_bfloat16>;
        }
        else if (prop.major == 10)
        {
            return std::is_same_v<CacheElem, __nv_fp8_e4m3>;
        }
        return false;
    }();
    if (batchSize == 0)
    {
        batchSize = exactDiv(std::lcm((uint32_t) prop.multiProcessorCount * 6, nbKHeads), nbKHeads);
    }
    if (seqLen == 0)
    {
        seqLen = (16U << 20) / gmemCacheHeadBytes; // 32MB per K+V head.
    }
    ctxLen = std::min(ctxLen, seqLen);
    float const kScale = cacheElemSize == 2 ? 1.f : 1 / 4.f;
    float const vScale = kScale;
    float const qScale = 1.f;
    size_t const histLen = seqLen;
    if (verbose)
    {
        printf("batchSize=%u, nbKHeads=%u, seqLen=%u, histLen=%lu\n", batchSize, nbKHeads, seqLen, histLen);
    }
#if USE_PAGED_KV_CACHE
    size_t const maxSeqLen = roundUp(seqLen, tokensPerPage);
#else
    size_t const maxSeqLen = seqLen;
#endif
    uint32_t const totalNbCacheHeads = (nbKHeads + nbVHeads) * maxSeqLen * beamWidth * batchSize;
    size_t const totalNbCacheElems = validElemsPerKHead * size_t(totalNbCacheHeads);

#if USE_INPUT_KV
    size_t const qkvElems = validElemsPerKHead * (nbQHeads + nbKHeads * 2) * beamWidth * batchSize;
#endif

#if SPEC_DEC
    size_t const qElems = validElemsPerKHead * qSeqLen * nbQHeads * beamWidth * batchSize;
    size_t const outElems = validElemsPerVHead * qSeqLen * nbQHeads * beamWidth * batchSize;
#else
    size_t const qElems = validElemsPerKHead * nbQHeads * beamWidth * batchSize;
    size_t const outElems = validElemsPerVHead * nbQHeads * beamWidth * batchSize;
#endif
    size_t const inputElems
        = (useInputKV ? validElemsPerKHead * (nbQHeads + nbKHeads * 2) * beamWidth * batchSize : qElems);
    size_t const cacheBytes = cacheElemSize * totalNbCacheElems;
    size_t const inputBytes = inputElemSize * inputElems;
    size_t const outputBytes = outputElemSize * outElems;
    size_t const seqLenListBytes = sizeof(uint32_t) * beamWidth * batchSize;
    size_t const ctxLenListBytes = sizeof(uint32_t) * beamWidth * batchSize;
#if USE_PAGED_KV_CACHE
    uint32_t const nbPagesPerSeq = divUp<uint32_t>(maxSeqLen, tokensPerPage);
    size_t const totalNbPages = nbPagesPerSeq * 2 * beamWidth * batchSize;
    size_t const pageListBytes = sizeof(KVCachePageIndex) * totalNbPages;
#else
    size_t const pageListBytes = 0U;
#endif
    size_t const cacheIndirBytes = beamWidth == 1 ? 0 : sizeof(uint32_t) * maxSeqLen * beamWidth * batchSize;
    size_t const totalBytes
        = cacheBytes + inputBytes + outputBytes + seqLenListBytes + ctxLenListBytes + pageListBytes + cacheIndirBytes;
    size_t const nbSeq = nbKHeads * batchSize;
#if SPEC_DEC
    size_t const nbSemaphores = nbKHeads * nbBlocksPerGrp * batchSize;
#else
    size_t const nbSemaphores = roundUp<size_t>(nbSeq, 2) + 2 + nbSeq + 2;
#endif
    auto const semaphores = ManagedMemBuf<uint32_t>(nbSemaphores);
    size_t const scratchSize = (256u << 20);
    auto const scratchBuf = ManagedMemBuf<std::byte>(scratchSize);
    std::fill_n(scratchBuf.get(), scratchSize, std::byte(0));
    auto const kvCacheScale = ManagedMemBuf<float>(1);
    kvCacheScale[0] = kScale;
    cudaEvent_t tic, toc;
    checkCuda(cudaEventCreate(&tic));
    checkCuda(cudaEventCreate(&toc));
    std::unique_ptr<CUevent_st, cudaError (*)(cudaEvent_t)> const ticEv{tic, &cudaEventDestroy};
    std::unique_ptr<CUevent_st, cudaError (*)(cudaEvent_t)> const tocEv{toc, &cudaEventDestroy};

    auto const ropeCosSin = ManagedMemBuf<Vec<float, validElemsPerKHead>>(seqLen);
#if USE_INPUT_KV && defined(ROPE_STYLE) && ROPE_STYLE
    for (uint32_t m = 0; m < seqLen; m++)
    {
        auto& pairs = ropeCosSin[m];
        constexpr uint32_t nbPairs = exactDiv(validElemsPerKHead, 2);
        for (uint32_t i = 0; i < nbPairs; i++)
        {
            float const theta = m * std::pow(1E4F, (-1.F / nbPairs) * i);
            pairs[i * 2] = std::cos(theta);
            pairs[i * 2 + 1] = std::sin(theta);
        }
    }
#endif
    auto const cacheHeads = ManagedMemBuf<GMemCacheHead>(totalNbCacheHeads);
#if USE_INPUT_KV
    auto const qkvHeads = ManagedMemBuf<InputHead[beamWidth][nbQHeads + nbKHeads * 2]>(batchSize);
#endif
#if SPEC_DEC
    auto const qHeads = ManagedMemBuf<InputHead[beamWidth][qSeqLen][nbQHeads]>(batchSize);
    auto const output = ManagedMemBuf<OutputHead[beamWidth][qSeqLen][nbQHeads]>(batchSize);
#else
    auto const qHeads = ManagedMemBuf<InputHead[beamWidth][nbQHeads]>(batchSize);
    auto const output = ManagedMemBuf<OutputHead[beamWidth][nbQHeads]>(batchSize);
#endif
    auto const rcpOutScale = ManagedMemBuf<float>(1);
    auto const seqLenList = ManagedMemBuf<uint32_t[beamWidth]>(batchSize);
    auto const ctxLenList = ManagedMemBuf<uint32_t[beamWidth]>(batchSize);
#if USE_PAGED_KV_CACHE
    auto const pageListBuf = ManagedMemBuf<std::byte>(pageListBytes);
    auto const pageList = reinterpret_cast<KVCachePageIndex(*)[beamWidth][2][nbPagesPerSeq]>(pageListBuf.get());
    KVCachePageIndex const* const pageListArg = &pageList[0][0][0][0];
#endif
#if USE_PAGED_KV_CACHE
    for (uint32_t i = 0; i < totalNbPages; i++)
    {
        (&pageList[0][0][0][0])[i] = i;
    }
#endif
    std::fill_n(&seqLenList[0][0], beamWidth * batchSize, seqLen);
    std::fill_n(&ctxLenList[0][0], beamWidth * batchSize, ctxLen);

#if SPEC_DEC
    std::vector<uint32_t> qSeqLenList(batchSize, qSeqLen);
    std::vector<uint32_t> cuQSeqLen(batchSize + 1, 0u);
    for (size_t i = 1; i < batchSize + 1; i++)
    {
        cuQSeqLen[i] = qSeqLenList[i - 1] + cuQSeqLen[i - 1];
        printf("bi %lu cuQSeqLen %u \n", i, cuQSeqLen[i]);
    }
    void* deviceCuQSeqLen = nullptr;
    checkCuda(cudaMalloc(&deviceCuQSeqLen, sizeof(uint32_t) * (batchSize + 1)));
#endif

    if (verbose)
    {
        printf("cacheHeads= %p q= %p output= %p\n", cacheHeads.get(), qHeads.get(), output.get());
        printf("cacheBytes= %lu  qByte= %lu  outbytes= %lu  totalBytes= %lu\n", cacheElemSize * totalNbCacheElems,
            inputElemSize * qElems, inputElemSize * outElems, totalBytes);
        printf("generating input data\n");
    }
    uint64_t seed = std::getenv("SEED") ? std::stoi(std::getenv("SEED")) : 0;
    std::mt19937_64 rng{seed};
    auto const cacheIndir = ManagedMemBuf<uint32_t>(beamWidth == 1 ? 0 : batchSize * beamWidth * maxSeqLen);
    if (beamWidth > 1)
    {
        std::uniform_int_distribution<uint32_t> cacheIndirDist(0, beamWidth - 1);
        for (uint32_t req = 0; req < batchSize; req++)
        {
            for (uint32_t b = 0; b < beamWidth; b++)
            {
                auto indices = cacheIndir.get() + maxSeqLen * (b + req * beamWidth);
                std::fill_n(indices, ctxLen, 0);
                std::generate_n(indices + ctxLen, seqLen - ctxLen, [&]() { return cacheIndirDist(rng); });
                std::fill_n(indices + seqLen, maxSeqLen - seqLen, ~0U);
            }
        }
    }
#if SPEC_DEC
    // Packed mask (all 1s), MaskType (aligned with uint32_t)
    size_t const numBitsPerPackedMask = sizeof(MaskType) * 8; // 32 bits.
    size_t const numPackedMasksPerToken = divUp(size_t(qSeqLen), numBitsPerPackedMask);
    size_t const numPackedMasks = qSeqLen * numPackedMasksPerToken;
    MaskType* hostPackedMask = reinterpret_cast<MaskType*>(malloc(numPackedMasks * sizeof(MaskType)));
    bool* deviceMask;
    bool* hostMask = reinterpret_cast<bool*>(malloc(qSeqLen * qSeqLen * sizeof(bool)));
    MaskType* devicePackedMask;
    checkCuda(cudaMalloc((void**) &deviceMask, qSeqLen * qSeqLen * sizeof(bool)));
    checkCuda(cudaMalloc((void**) &devicePackedMask, batchSize * numPackedMasks * sizeof(MaskType)));
    std::bernoulli_distribution maskDist(0.5f);
    for (uint32_t tokenIdx = 0; tokenIdx < qSeqLen; tokenIdx++)
    {
        // Init random host uint32_t masks for reference codes.
        for (uint32_t kvPosIdx = 0; kvPosIdx < qSeqLen; kvPosIdx++)
        {
#if IS_MLA || SPEC_Q_SEQ_LEN
            hostMask[tokenIdx * qSeqLen + kvPosIdx] = (tokenIdx >= kvPosIdx);
#else
            hostMask[tokenIdx * qSeqLen + kvPosIdx] = maskDist(rng);
#endif
        }

        // Pack boolean masks into bits.
        for (uint32_t maskIdx = 0; maskIdx < numPackedMasksPerToken; maskIdx++)
        {
            MaskType packedMask = 0u;
            for (uint32_t posIdx = 0; posIdx < numBitsPerPackedMask; posIdx++)
            {
                uint32_t maskPosIdx = maskIdx * numBitsPerPackedMask + posIdx;
                uint32_t maskFlag = 0u;
                if (maskPosIdx < qSeqLen)
                {
                    maskFlag = hostMask[tokenIdx * qSeqLen + maskPosIdx];
                }

                packedMask |= maskFlag << posIdx;
            }
            hostPackedMask[tokenIdx * numPackedMasksPerToken + maskIdx] = packedMask;
        }
    }
#endif
    bool const zeroInput = !refCheck && std::getenv("XQA_ZERO_FILL") && std::stoi(std::getenv("XQA_ZERO_FILL"));
    if (!zeroInput)
    {
        auto genTokenElem = [&](auto&& generator)
        {
#if CACHE_ELEM_ENUM == 0
            return InputElem(generator());
#elif CACHE_ELEM_ENUM == 1
            return static_cast<int8_t>(std::clamp<float>(std::round(generator() / kScale), -127, 127));
#elif CACHE_ELEM_ENUM == 2
            return __nv_fp8_e4m3{generator() / kScale};
#endif
        };
        auto const nbThrds = std::thread::hardware_concurrency();
        std::vector<std::future<void>> futures;
        futures.reserve(nbThrds);
        uint32_t const headsPerThrd = divUp(totalNbCacheHeads, nbThrds);
        auto const threadTask = [&](uint32_t i)
        {
            std::mt19937_64 cacheRng{seed + (i + 3) * 1000639U};
            std::normal_distribution<float> cacheDist{0.f, 1.f};
            auto genCacheElem = [&]() { return genTokenElem([&]() { return cacheDist(cacheRng); }); };
            if (headsPerThrd * i >= totalNbCacheHeads)
            {
                return;
            }
            size_t const nbCacheElemsForThisThrd
                = validElemsPerKHead * std::min<size_t>(headsPerThrd, totalNbCacheHeads - headsPerThrd * i);
            std::generate_n(cacheHeads[headsPerThrd * i].data, nbCacheElemsForThisThrd, genCacheElem);
        };
        for (uint32_t i = 0; i < nbThrds; i++)
        {
            futures.emplace_back(std::async(std::launch::async, threadTask, i));
        }
        for (auto& f : futures)
        {
            f.wait();
        }
        futures.clear();
        std::normal_distribution<float> dist{0.f, 1.f};

#if USE_INPUT_KV
        std::generate_n(qkvHeads[0][0][0].data, qkvElems,
            [&] { return InputElem(genTokenElem([&]() { return dist(rng) * kScale; })); });
        for (uint32_t i = 0; i < batchSize; i++)
        {
            for (uint32_t j = 0; j < beamWidth; j++)
            {
                for (uint32_t k = 0; k < nbQHeads; k++)
                {
                    qHeads[i][j][k] = applyRoPE<ROPE_STYLE>(qkvHeads[i][j][k], ropeCosSin[seqLen - 1]);
                }
            }
        }
        std::fill_n(output[0][0][0].data, outElems, OutputElem(NAN));
#else
#if SPEC_DEC
        std::generate_n(
            qHeads[0][0][0][0].data, qElems, [&] { return InputElem(genTokenElem([&]() { return dist(rng); })); });
        std::fill_n(output[0][0][0][0].data, outElems, OutputElem(NAN));
#else
        std::generate_n(
            qHeads[0][0][0].data, qElems, [&] { return InputElem(genTokenElem([&]() { return dist(rng); })); });
        std::fill_n(output[0][0][0].data, outElems, OutputElem(NAN));
#endif
#endif
#if USE_PAGED_KV_CACHE
        std::shuffle(&pageList[0][0][0][0], &pageList[0][0][0][0] + totalNbPages, rng);
#endif
#if IS_MLA
#if USE_PAGED_KV_CACHE
        for (uint32_t idxReq = 0; idxReq < batchSize; idxReq++)
        {
            for (uint32_t idxBeam = 0; idxBeam < beamWidth; idxBeam++)
            {
                for (uint32_t idxPage = 0; idxPage < nbPagesPerSeq; idxPage++)
                {
                    pageList[idxReq][idxBeam][1][idxPage] = pageList[idxReq][idxBeam][0][idxPage];
                }
            }
        }
#else
        static_assert(false, "not implemented");
#endif
#endif
    }
    else
    {
#if CACHE_ELEM_ENUM == 0
        InputElem const cacheFillVal = InputElem(0.01f);
#elif CACHE_ELEM_ENUM == 1
        int8_t const cacheFillVal = 1;
#elif CACHE_ELEM_ENUM == 2
        __nv_fp8_e4m3 const cacheFillVal{0.01f};
#endif
        std::fill_n(&cacheHeads[0][0], totalNbCacheElems, cacheFillVal);
#if SPEC_DEC
        std::fill_n(qHeads[0][0][0][0].data, qElems, InputElem(0.01f));
        std::fill_n(output[0][0][0][0].data, outElems, OutputElem(NAN));
#else
        std::fill_n(qHeads[0][0][0].data, qElems, InputElem(0.01f));
        std::fill_n(output[0][0][0].data, outElems, OutputElem(NAN));
#endif
    }
    rcpOutScale[0] = lowPrecOutput ? 4.F : 1.F;
#if USE_INPUT_KV
    for (int i = 0; i < batchSize; i++)
    {
        uint32_t const pos = seqLen - 1;
        static_assert(beamWidth == 1);
        for (int kv = 0; kv < 2; kv++)
        {
            for (int j = 0; j < nbKHeads; j++)
            {
#if USE_PAGED_KV_CACHE
                uint32_t const pageIdx = pageList[i][0][kv][pos / tokensPerPage];
                uint32_t const idxHead = tokensPerPage * (nbKHeads * pageIdx + j) + pos % tokensPerPage;
#else
                uint32_t const idxHead = maxSeqLen * (nbKHeads * i + j) + pos;
#endif
                cacheHeads[idxHead].fill(CacheElem(128.F));
            }
        }
    }
#endif

    auto const cacheHeadAt = [&](uint32_t batch, bool isK, uint32_t idxKVHead, uint32_t pos) -> GMemCacheHead&
    {
        uint32_t const beam = 0;
        uint32_t const kv = isK ? 0 : 1;
#if USE_PAGED_KV_CACHE
        auto const pageList = reinterpret_cast<KVCachePageIndex(*)[beamWidth][2][nbPagesPerSeq]>(pageListBuf.get());
        uint32_t const pageIdx = pageList[batch][beam][kv][pos / tokensPerPage];
        uint32_t const idxHead = tokensPerPage * (nbKHeads * pageIdx + idxKVHead) + pos % tokensPerPage;
#else
        static_assert(beamWidth == 1);
        uint32_t const idxHead = maxSeqLen * (nbKHeads * (batch * 2 + kv) + idxKVHead) + pos;
#endif
        return cacheHeads[idxHead];
    };
    for (uint32_t batch = 0; batch < batchSize; batch++)
    {
        for (uint32_t kv = 0; kv < 2; kv++)
        {
            for (uint32_t idxKVHead = 0; idxKVHead < nbKHeads; idxKVHead++)
            {
                for (uint32_t pos = seqLen; pos < maxSeqLen; pos++)
                {
                    cacheHeadAt(batch, kv, idxKVHead, pos).fill(CacheElem(0.F));
                }
            }
        }
    }

    if (verbose)
    {
        printf("migrating data to gpu\n");
    }
    cudaStream_t const stream = nullptr;
    auto prefetchToDevice = [&](int dev)
    {
        semaphores.prefetch(dev, stream);
        scratchBuf.prefetch(dev, stream);
        kvCacheScale.prefetch(dev, stream);
        cacheHeads.prefetch(dev, stream);
        qHeads.prefetch(dev, stream);
        output.prefetch(dev, stream);
        rcpOutScale.prefetch(dev, stream);
        seqLenList.prefetch(dev, stream);
        ctxLenList.prefetch(dev, stream);
#if USE_PAGED_KV_CACHE
        pageListBuf.prefetch(dev, stream);
#endif
#if BEAM_WIDTH > 1
        cacheIndir.prefetch(dev, stream);
#endif
    };
    prefetchToDevice(device);
    checkCuda(cudaMemsetAsync(semaphores.get(), 0, 4 * nbSemaphores, stream));
#if SPEC_DEC
    for (size_t bi = 0; bi < batchSize; bi++)
    {
        checkCuda(cudaMemcpyAsync(devicePackedMask + bi * numPackedMasks, hostPackedMask,
            numPackedMasks * sizeof(MaskType), cudaMemcpyHostToDevice, stream));
    }
    checkCuda(cudaMemcpyAsync(deviceMask, hostMask, qSeqLen * qSeqLen * sizeof(bool), cudaMemcpyHostToDevice, stream));
    checkCuda(cudaMemcpyAsync(
        deviceCuQSeqLen, cuQSeqLen.data(), sizeof(uint32_t) * (batchSize + 1), cudaMemcpyHostToDevice, stream));
    checkCuda(cudaStreamSynchronize(stream));
#endif

#if BEAM_WIDTH > 1
    BeamSearchParams const beamSearchParams{
        .indices = cacheIndir.get(), .capacity = maxSeqLen, .ctxLenList = &ctxLenList[0][0]};
#endif

#if SPEC_DEC
    auto const scratch = reinterpret_cast<void*>(roundUp<uintptr_t>(reinterpret_cast<uintptr_t>(scratchBuf.get()),
        (useQGMMA ? ioHeadBytes : paddedInputHeadBytes) * runtimeHeadGrpSize
            * beamWidth)); // 8 is sufficient for qgmma kernel.
#else
    auto const scratch = reinterpret_cast<void*>(roundUp<uintptr_t>(reinterpret_cast<uintptr_t>(scratchBuf.get()),
        (useQGMMA ? ioHeadBytes : paddedInputHeadBytes) * headGrpSize
            * beamWidth)); // 8 is sufficient for qgmma kernel.
#endif

#if IS_MLA
    auto runKernel = [&]()
    {
        launchMLA(prop, qSeqLen, qScale,
#if SPEC_DEC
            &output[0][0][0][0], &qHeads[0][0][0][0],
#else
            &output[0][0][0], &qHeads[0][0][0],
#endif
            cacheHeads.get(),
#if USE_PAGED_KV_CACHE
            pageListArg,
#endif
            maxSeqLen, &seqLenList[0][0], batchSize, kvCacheScale.get(), semaphores.get(), scratch, stream);
    };
#else
    auto runKernel = [&]()
    {
        auto const launchFunc = useQGMMA ? &launchHopperF8MHA : &launchMHA;

#if SPEC_DEC
        SpecDecParams const specDecParams{.qSeqLen = qSeqLen,
            .qCuSeqLens = reinterpret_cast<uint32_t const*>(deviceCuQSeqLen),
            .mask = reinterpret_cast<MaskType const*>(devicePackedMask)};
#endif
        launchFunc(prop, nbKHeads,
#if SLIDING_WINDOW
            slidingWinSize,
#endif
            qScale,
#if SPEC_DEC
            &output[0][0][0][0],
#else
            &output[0][0][0],
#endif
#if LOW_PREC_OUTPUT
            rcpOutScale.get(),
#endif
#if USE_INPUT_KV
            &qkvHeads[0][0][0],
#if ROPE_STYLE != 0
            ropeCosSin.get(),
#endif
#else
#if SPEC_DEC
            &qHeads[0][0][0][0],
#else
            &qHeads[0][0][0],
#endif
#endif
            cacheHeads.get(),
#if USE_PAGED_KV_CACHE
            pageListArg,
#endif
            maxSeqLen, &seqLenList[0][0],
#if BEAM_WIDTH > 1
            beamSearchParams,
#endif
            batchSize, kvCacheScale.get(),
#if SPEC_DEC
            specDecParams,
#endif
            semaphores.get(), scratch, stream);
        checkCuda(cudaGetLastError());
    };
#endif
    if (testPerf && !isTracing)
    {
        if (verbose)
        {
            printf("warming up\n");
        }

        warmup(prop, 20.F, stream);
        for (int32_t i = 0; i < 20; i++)
        {
            runKernel();
        }
        if (verbose)
        {
            printf("testing\n");
        }
    }
    if (isTracing)
    {
        printf("Tracing is enabled\n");
    }
    checkCuda(cudaEventRecord(tic, stream));
    int32_t const nbIters = ((USE_SMALL_IO || isTracing || !testPerf) ? 1 : 100);
    nvtxRangePushA("test");
    for (int32_t i = 0; i < nbIters; i++)
    {
        runKernel();
    }
    nvtxRangePop();
    checkCuda(cudaEventRecord(toc, stream));
    prefetchToDevice(cudaCpuDeviceId);
    checkCuda(cudaStreamSynchronize(stream));
    if (testPerf)
    {
        float ms;
        checkCuda(cudaEventElapsedTime(&ms, tic, toc));
        ms /= nbIters;
        float const bandwidth = 2.f * prop.memoryBusWidth * prop.memoryClockRate * 1000 / 8;
#if BEAM_WIDTH == 1
        size_t nbLoadedCacheTokens = seqLen * beamWidth * batchSize;
#else
        size_t nbLoadedCacheTokens = 0;
        for (uint32_t req = 0; req < batchSize; req++)
        {
            nbLoadedCacheTokens += ctxLen;
            for (uint32_t s = ctxLen; s < seqLen; s++)
            {
                bool isUsed[beamWidth] = {};
                for (uint32_t b = 0; b < beamWidth; b++)
                {
                    uint32_t const idx = cacheIndir[s + maxSeqLen * (b + beamWidth * req)];
                    isUsed[idx] = true;
                }
                nbLoadedCacheTokens += std::count(std::begin(isUsed), std::end(isUsed), true);
            }
        }
#endif
        size_t const totalNbCacheLoadBytes = gmemCacheHeadBytes * (nbKHeads + nbVHeads) * nbLoadedCacheTokens;
        float const totalTraffic
            = totalNbCacheLoadBytes + inputBytes + outputBytes; // we ignore page indices and beam search indices.
        float const dramSolTime = totalTraffic / bandwidth * 1E3f;
        float const dramSolRatio = dramSolTime / ms;
        if (verbose)
        {
            printf("done\n");
            printf("time: %f ms\n", ms);
            printf("mem bus width = %d\nmem clock rate = %d\n", prop.memoryBusWidth, prop.memoryClockRate);
            printf("bandwidth = %e\n", (float) bandwidth);
            printf("traffic=%e\n", (float) totalTraffic);
        }
        float const tops = headGrpSize * qSeqLen * float(seqLen) * (validElemsPerKHead + validElemsPerVHead) * 2
            * nbKHeads * batchSize / (ms * 1E-3F) * 1E-12F;
        printf("dramSolRatio: %f%% (%f ms, TOPS = %f)\n", dramSolRatio * 100, ms, tops);
    }
    if (refCheck)
    {
        float const qScaleForRef = isMLA ? qScale * sqrtf(576.F) : qScale;
        if (saveData)
        {
            save<float>("kv.bin", &cacheHeads[0][0], validElemsPerKHead * cacheHeads.size());
#if SPEC_DEC
            save<float>(
                "q.bin", &qHeads[0][0][0][0][0], validElemsPerKHead * nbQHeads * qSeqLen * beamWidth * batchSize);
#else
            save<float>("q.bin", &qHeads[0][0][0][0], validElemsPerKHead * nbQHeads * beamWidth * batchSize);
#endif
        }

        size_t hash = 0;
        for (size_t i = 0; i < exactDiv(sizeof(output[0]) * output.size(), 8); i++)
        {
            hash_combine(hash, reinterpret_cast<uint64_t const*>(output.get())[i]);
        }
        printf("Output hash: %p\n", hash);

        for (size_t i = 0; i < semaphores.size(); i++)
        {
            EXPECT_EQ(semaphores[i], 0);
        }

        float maxAbsErr = 0.F;
        float maxRelErr = 0.F;
        uint32_t nbErrors = 0;
        float const allowedErr = ((useQGMMA || lowPrecOutput || isMLA) ? 0.15f : 0.05f);
        float const allowedRelErr = allowedErr;
        auto checkClose = [&](auto type, float val, float ref, float epsilon) mutable
        {
            EXPECT_TRUE(std::isfinite((val)));
            float const absErr = std::abs(val - ref);
            maxAbsErr = std::max(maxAbsErr, absErr);
            bool ok{true};
            if constexpr (std::is_same_v<std::decay_t<decltype(type)>, __nv_fp8_e4m3>)
            {
                auto const relErr = absErr / std::abs(ref);
                maxRelErr = std::max(maxRelErr, relErr);
                ok = (absErr <= epsilon || relErr <= allowedErr);
            }
            else
            {
                ok = (absErr < epsilon);
            }
            EXPECT_TRUE(ok);
            if (!ok)
            {
                printf("val=%f, ref=%f, epsilon=%f, absErr=%f\n", val, ref, epsilon, absErr);
                nbErrors++;
            }
        };

#if USE_INPUT_KV
        for (int i = 0; i < batchSize; i++)
        {
            uint32_t const pos = seqLen - 1;
            static_assert(beamWidth == 1);
            uint32_t const idxBeam = 0;
            for (int kv = 0; kv < 2; kv++)
            {
                for (int j = 0; j < nbKHeads; j++)
                {
#if USE_PAGED_KV_CACHE
                    uint32_t const pageIdx = pageList[i][0][kv][pos / tokensPerPage];
                    uint32_t const idxHead = tokensPerPage * (nbKHeads * pageIdx + j) + pos % tokensPerPage;
#else
                    uint32_t const idxHead = maxSeqLen * (nbKHeads * (i * 2 + kv) + j) + pos;
#endif
                    auto const& ch = cacheHeads[idxHead];
                    auto const& kvh = qkvHeads[i][idxBeam][nbQHeads + nbKHeads * kv + j];
#if defined(ROPE_STYLE)
                    auto const rh = (kv == 0 ? applyRoPE<ROPE_STYLE>(kvh, ropeCosSin[seqLen - 1]) : kvh);
#else
                    auto const rh = kvh;
#endif
                    Vec<CacheElem, validElemsPerKHead> ref;
                    std::transform(
                        rh.data, rh.data + rh.size, ref.data, [&](auto x) { return CacheElem{float(x) / kScale}; });
                    for (int e = 0; e < validElemsPerKHead; e++)
                    {
                        checkClose(CacheElem{}, float(ch[e]), float(ref[e]), allowedErr / kScale);
                    }
                }
            }
        }
#endif

#if SPEC_DEC
        std::vector<std::array<std::array<Vec<float, validElemsPerVHead>, nbQHeads * qSeqLen>, beamWidth>> outputF32(
            batchSize);
#else
        std::vector<std::array<std::array<Vec<float, validElemsPerVHead>, nbQHeads>, beamWidth>> outputF32(batchSize);
#endif
#pragma omp for
        for (uint32_t req = 0; req < batchSize; req++)
        {
            for (uint32_t b = 0; b < beamWidth; b++)
            {
#if SPEC_DEC
                for (uint32_t q_len = 0; q_len < qSeqLen; q_len++)
                {
                    for (uint32_t q = 0; q < nbQHeads; q++)
                    {
                        for (uint32_t i = 0; i < validElemsPerVHead; i++)
                        {
                            outputF32[req][b][q_len * nbQHeads + q][i] = float(output[req][b][q_len][q][i]);
                        }
                    }
                }
#else
                for (uint32_t q = 0; q < nbQHeads; q++)
                {
                    for (uint32_t i = 0; i < validElemsPerVHead; i++)
                    {
                        outputF32[req][b][q][i] = float(output[req][b][q][i]);
                    }
                }
#endif
            }
        }
        std::ofstream fout_refOutput;
        if (saveData)
        {
#if SPEC_DEC
            save<float>(
                "out.bin", &outputF32[0][0][0][0], validElemsPerVHead * nbQHeads * qSeqLen * beamWidth * batchSize);
#else
            save<float>("out.bin", &outputF32[0][0][0][0], validElemsPerVHead * nbQHeads * beamWidth * batchSize);
#endif
            fout_refOutput = std::ofstream("ref_cpp.bin", std::ios::binary | std::ios::trunc);
        }

        constexpr float kE4M3_MAX = 448.F;
        float const xScale = useQGMMA ? 1 / kE4M3_MAX : 1.f;
        for (uint32_t req = 0; req < batchSize; req++)
        {
            for (uint32_t b = 0; b < beamWidth; b++)
            {
#if SPEC_DEC
                for (uint32_t q_len = 0; q_len < qSeqLen; q_len++)
                {
#endif
                    for (uint32_t idxKHead = 0; idxKHead < nbKHeads; idxKHead++)
                    {

#if USE_PAGED_KV_CACHE
#if BEAM_WIDTH == 1
                        CacheSeq<true, false> const kCacheSeq{.pool = cacheHeads.get(),
                            .pageIndices = pageList[req][b][0],
                            .nbHeads = nbKHeads,
                            .idxHead = idxKHead};
                        CacheSeq<true, false> const vCacheSeq{.pool = cacheHeads.get(),
                            .pageIndices = pageList[req][b][1],
                            .nbHeads = nbKHeads,
                            .idxHead = idxKHead};

#else
                        CacheSeq<true, true> const kCacheSeq{.pool = cacheHeads.get(),
                            .pageIndices = pageList[req][0][0],
                            .maxNbPages = nbPagesPerSeq,
                            .nbHeads = nbKHeads,
                            .idxHead = idxKHead,
                            .cacheIndir = &cacheIndir[maxSeqLen * (b + beamWidth * req)]};
                        CacheSeq<true, true> const vCacheSeq{.pool = cacheHeads.get(),
                            .pageIndices = pageList[req][0][1],
                            .maxNbPages = nbPagesPerSeq,
                            .nbHeads = nbKHeads,
                            .idxHead = idxKHead,
                            .cacheIndir = &cacheIndir[maxSeqLen * (b + beamWidth * req)]};
#endif
#else
                    auto const kv
                        = reinterpret_cast<GMemCacheHead(*)[beamWidth][2][nbKHeads][maxSeqLen]>(cacheHeads.get());
#if BEAM_WIDTH == 1
                    CacheSeq<false, false> const kCacheSeq{kv[req][b][0][idxKHead]};
                    CacheSeq<false, false> const vCacheSeq{kv[req][b][1][idxKHead]};
#else
                    CacheSeq<false, true> const kCacheSeq{.nbKHeads = nbKHeads,
                        .data = kv[req][0][0][idxKHead],
                        .cacheIndir = &cacheIndir[maxSeqLen * (b + beamWidth * req)],
                        .maxSeqLen = maxSeqLen};
                    CacheSeq<false, true> const vCacheSeq{.nbKHeads = nbKHeads,
                        .data = kv[req][0][1][idxKHead],
                        .cacheIndir = &cacheIndir[maxSeqLen * (b + beamWidth * req)],
                        .maxSeqLen = maxSeqLen};

#endif
#endif
#if SPEC_DEC
                        Eigen::Matrix<float, runtimeHeadGrpSize, validElemsPerHead, Eigen::RowMajor> refOutput;
                        refOutput = refAttention<InputElem>(&qHeads[req][b][q_len][runtimeHeadGrpSize * idxKHead],
                            kCacheSeq, vCacheSeq, seqLen, qScaleForRef, kvCacheScale[0], xScale, slidingWinSize,
                            hostMask, qSeqLen, q_len);
#else
                    Eigen::Matrix<float, headGrpSize, validElemsPerHead, Eigen::RowMajor> refOutput;
                    if (useQGMMA)
                    {
                        refOutput = refFlashAttention<CacheElem, 64>(&qHeads[req][b][headGrpSize * idxKHead], kCacheSeq,
                            vCacheSeq, seqLen, qScaleForRef, kvCacheScale[0], xScale, slidingWinSize);
                        // refOutput = refAttention<CacheElem>(&qHeads[req][b][headGrpSize * idxKHead], kCacheSeq,
                        // vCacheSeq, seqLen, qScaleForRef, kvCacheScale[0], xScale, slidingWinSize);
                    }
                    else
                    {
                        // refOutput = refFlashAttention<InputElem, 64>(&qHeads[req][b][headGrpSize * idxKHead],
                        // kCacheSeq, vCacheSeq, seqLen, qScaleForRef, kvCacheScale[0], xScale);
                        refOutput = refAttention<InputElem>(&qHeads[req][b][headGrpSize * idxKHead], kCacheSeq,
                            vCacheSeq, seqLen, qScaleForRef, kvCacheScale[0], xScale, slidingWinSize);
                    }
#endif
                        if (lowPrecOutput)
                        {
                            refOutput = refOutput.unaryExpr(
                                [&](float e) { return float(__nv_fp8_e4m3(e * rcpOutScale[0])); });
                        }
                        if (saveData)
                        {
                            fout_refOutput.write(
                                (char const*) refOutput.data(), sizeof(refOutput[0]) * refOutput.size());
                        }
#if SPEC_DEC
                        for (uint32_t i = 0; i < runtimeHeadGrpSize; i++)
#else
                    for (uint32_t i = 0; i < headGrpSize; i++)
#endif
                        {
                            for (uint32_t j = 0; j < validElemsPerVHead; j++)
                            {
#if SPEC_DEC
                                float const val
                                    = outputF32[req][b][q_len * nbQHeads + runtimeHeadGrpSize * idxKHead + i][j];
#else
                            float const val = outputF32[req][b][headGrpSize * idxKHead + i][j];
#endif
                                float const ref = refOutput(i, j);
                                checkClose(OutputElem{}, val, ref, allowedErr * rcpOutScale[0]);
                            }
                        }
                    }
#ifdef MEUDSA
                }
#endif
            }
        }
        if (saveData)
        {
            fout_refOutput.close();
        }

        if (verbose)
        {
            printf("max absolute error: %f\n", maxAbsErr);
            printf("max relative error: %f\n", maxRelErr);
        }
        EXPECT_EQ(nbErrors, 0) << "number of wrong elements: " << nbErrors;
    }
#if SPEC_DEC
    free(hostMask);
    free(hostPackedMask);
}
#endif
}

#if SPEC_DEC
constexpr bool runPerfTest = false;
constexpr bool runCheckTest = true;

#if IS_MLA
TEST(RefCheck, mla)
{
    // runTest<1, headGrpSize, 2>(1, 2, runPerfTest, runCheckTest, true, true);
    runTest<1, headGrpSize, 1>(32, 200, runPerfTest, runCheckTest, true, true);
    runTest<1, headGrpSize, 2>(32, 200, runPerfTest, runCheckTest, true, true);
    runTest<1, headGrpSize, 2>(2, 1000, runPerfTest, runCheckTest, true, true);
    runTest<1, headGrpSize, 13>(2, 257, runPerfTest, runCheckTest, true, true);
}
#else
#define HEAD_GROUP_SIZE HEAD_GRP_SIZE
#ifdef SPEC_Q_SEQ_LEN
#define Q_SEQ_LEN SPEC_Q_SEQ_LEN
#else
#define Q_SEQ_LEN 62
#endif

TEST(RefCheck, llama_V2_70b_3)
{
    // runTest<2, headGrpSize, 12>(2, 97, false, true, true, true);
    if constexpr (Q_SEQ_LEN <= 13)
    {
        runTest<1, HEAD_GROUP_SIZE, Q_SEQ_LEN>(1, 13, runPerfTest, runCheckTest);
    }
    runTest<4, HEAD_GROUP_SIZE, Q_SEQ_LEN>(8, 1128, runPerfTest, runCheckTest);
    runTest<2, HEAD_GROUP_SIZE, Q_SEQ_LEN>(1, 1128, runPerfTest, runCheckTest);
    runTest<2, HEAD_GROUP_SIZE, Q_SEQ_LEN>(2, 1128, runPerfTest, runCheckTest);
    runTest<4, HEAD_GROUP_SIZE, Q_SEQ_LEN>(1, 1128, runPerfTest, runCheckTest);
    runTest<4, HEAD_GROUP_SIZE, Q_SEQ_LEN>(4, 1128, runPerfTest, runCheckTest);
    runTest<4, HEAD_GROUP_SIZE, Q_SEQ_LEN>(8, 1128, runPerfTest, runCheckTest);
    runTest<8, HEAD_GROUP_SIZE, Q_SEQ_LEN>(8, 256, runPerfTest, runCheckTest);
    runTest<8, HEAD_GROUP_SIZE, Q_SEQ_LEN>(8, 512, runPerfTest, runCheckTest);
    runTest<8, HEAD_GROUP_SIZE, Q_SEQ_LEN>(8, 1028, runPerfTest, runCheckTest);
    runTest<8, HEAD_GROUP_SIZE, Q_SEQ_LEN>(8, 2048, runPerfTest, runCheckTest);
    runTest<8, HEAD_GROUP_SIZE, Q_SEQ_LEN>(8, 4096, runPerfTest, runCheckTest);
}
#endif

#else

#if IS_MLA
TEST(RefCheck, mla)
{
    // runTest<1>(1, 2, false, true, true, true);
    runTest<1>(1, 2048, false, true, true, true);
    runTest<1>(2, 2, false, true);
    runTest<1>(2, 15, false, true);
    runTest<1>(2, 256, false, true);
    runTest<1>(2, 514, false, true);
    runTest<1>(1, 4096, false, true);
    runTest<1>(120, 367, false, true);
    runTest<1>(112, 2158, false, true);
}

TEST(Perf, mla)
{
#ifndef NDEBUG
    GTEST_SKIP() << "Skipping perf tests for debug build";
#endif
    // runTest<1>(38, 4096, true, false);
    runTest<1>(46, 4096, true, false);
}

TEST(Perf, mla_real)
{
#ifndef NDEBUG
    GTEST_SKIP() << "Skipping perf tests for debug build";
#endif
    runTest<1>(64, 4096, true, false);
}

TEST(Perf, mla_tracing)
{
#ifndef NDEBUG
    GTEST_SKIP() << "Skipping perf tests for debug build";
#endif
    runTest<1>(1, 64 * 4 * 4, true, false);
}
#else
TEST(RefCheck, llama_V2_70b)
{
    // runTest<1>(1, 2, false, true, true, true);
    runTest<2>(2, 2, false, true);
    runTest<2>(2, 15, false, true);
    runTest<2>(2, 256, false, true);
    runTest<2>(2, 514, false, true);
    runTest<1>(1, 4096, false, true);
#if SLIDING_WINDOW
    runTest<2>(2, 4096, false, true, false, false, ~0, 256);
    runTest<2>(2, 400, false, true, false, false, ~0U, 256);
#endif
    runTest<8>(120, 367, false, true);
    // runTest<8>(1792, 2048, false, true);
}

TEST(Perf, tracing_long)
{
#ifndef NDEBUG
    GTEST_SKIP() << "Skipping perf tests for debug build";
#endif
    runTest<1>(0, 4096, true, false);
}

TEST(Perf, tracing_short)
{
#ifndef NDEBUG
    GTEST_SKIP() << "Skipping perf tests for debug build";
#endif
    runTest<1>(0, 384, true, false);
}

TEST(Perf, llama_V2_70b_long_seq)
{
#ifndef NDEBUG
    GTEST_SKIP() << "Skipping perf tests for debug build";
#endif
    runTest<8>(0, 0, true, false);
}

TEST(Perf, llama_V2_70b_4096)
{
#ifndef NDEBUG
    GTEST_SKIP() << "Skipping perf tests for debug build";
#endif
    runTest<8>(0, 4096, true, false);
}

TEST(Perf, llama_V2_70b_2048)
{
#ifndef NDEBUG
    GTEST_SKIP() << "Skipping perf tests for debug build";
#endif
    runTest<8>(560, 2048, true, false);
}

TEST(Perf, llama_V2_70b_256)
{
#ifndef NDEBUG
    GTEST_SKIP() << "Skipping perf tests for debug build";
#endif
    runTest<8>(960, 256, true, false);
}

TEST(Perf, llama_V2_70b_512)
{
#ifndef NDEBUG
    GTEST_SKIP() << "Skipping perf tests for debug build";
#endif
    runTest<8>(960, 512, true, false);
}

TEST(Perf, mlperf_gptj)
{
#ifndef NDEBUG
    GTEST_SKIP() << "Skipping perf tests for debug build";
#endif
    runTest<32>(396, 800 + 224, true, false, false, false, 800);
}

TEST(Perf, mlperf_llama)
{
#ifndef NDEBUG
    GTEST_SKIP() << "Skipping perf tests for debug build";
#endif
    runTest<8>(1792, 367, true, false);
}

TEST(Perf, bs1)
{
#ifndef NDEBUG
    GTEST_SKIP() << "Skipping perf tests for debug build";
#endif
    runTest<16>(4, 64 * 16 * 16, true, false);
}

TEST(Perf, tmp)
{
#ifndef NDEBUG
    GTEST_SKIP() << "Skipping perf tests for debug build";
#endif
    runTest<4>(32, 100, true, false);
}
#endif

#if ENABLE_NVRTC
#define NVRTC_RUN(x) ASSERT_EQ(NVRTC_SUCCESS, (x))
#define CU_RUN(x) ASSERT_EQ(CUDA_SUCCESS, (x))

TEST(NVRTC, compile)
{
    checkCuda(cudaFree(nullptr));
    int device;
    checkCuda(cudaGetDevice(&device));
    cudaDeviceProp prop;
    checkCuda(cudaGetDeviceProperties(&prop, device));

    int const major = prop.major;
    int const minor = prop.minor;
    ASSERT_GT(major, 0);

    std::vector<char const*> headers_content = {
        tensorrt_llm::kernels::cuda_hint_cuh_content,
        tensorrt_llm::kernels::defines_h_content,
        tensorrt_llm::kernels::ldgsts_cuh_content,
        tensorrt_llm::kernels::mha_h_content,
        tensorrt_llm::kernels::mha_utils_cuh_content,
        tensorrt_llm::kernels::mma_cuh_content,
        tensorrt_llm::kernels::platform_h_content,
        tensorrt_llm::kernels::utils_cuh_content,
        tensorrt_llm::kernels::utils_h_content,
        tensorrt_llm::kernels::mha_stdheaders_cuh_content,
        tensorrt_llm::kernels::mha_components_cuh_content,
        tensorrt_llm::kernels::mla_sm120_cuh_content,
        tensorrt_llm::kernels::gmma_cuh_content,
        tensorrt_llm::kernels::gmma_impl_cuh_content,
        tensorrt_llm::kernels::barriers_cuh_content,
        tensorrt_llm::kernels::tma_h_content,
        tensorrt_llm::kernels::cuda_bf16_h_content,
        tensorrt_llm::kernels::cuda_bf16_hpp_content,
        tensorrt_llm::kernels::cuda_fp16_h_content,
        tensorrt_llm::kernels::cuda_fp16_hpp_content,
        tensorrt_llm::kernels::cuda_fp8_h_content,
        tensorrt_llm::kernels::cuda_fp8_hpp_content,
        tensorrt_llm::kernels::vector_types_h_content,
        tensorrt_llm::kernels::vector_functions_h_content,
        tensorrt_llm::kernels::device_types_h_content,
    };
    std::vector<char const*> headers_name = {"cuda_hint.cuh", "defines.h", "ldgsts.cuh", "mha.h", "mhaUtils.cuh",
        "mma.cuh", "platform.h", "utils.cuh", "utils.h", "mha_stdheaders.cuh", "mha_components.cuh", "mla_sm120.cuh",
        "gmma.cuh", "gmma_impl.cuh", "barriers.cuh", "tma.h", "cuda_bf16.h", "cuda_bf16.hpp", "cuda_fp16.h",
        "cuda_fp16.hpp", "cuda_fp8.h", "cuda_fp8.hpp", "vector_types.h", "vector_functions.h", "device_types.h"};
    assert(headers_content.size() == headers_name.size());
    auto test = [&](int input_fp16, int cache_enum, int head_dim, int head_grp_size, bool use_paged_kv_cache,
                    int beam_width, char const* source_file, int compileMajor, int compileMinor)
    {
        std::string arch_flag = "-arch=sm_" + std::to_string(compileMajor) + std::to_string(compileMinor);
        if ((compileMajor == 9 || compileMajor == 10 || compileMajor == 12) && compileMinor == 0)
        {
            arch_flag += "a";
        }
        std::vector<std::string> options = {
            "-dw",
            "-std=c++17",
            "--use_fast_math",
            arch_flag,
            "-default-device",
            "-DGENERATE_CUBIN=1",
            "-DNDEBUG",
            input_fp16 ? "-DDTYPE=__half" : "-DDTYPE=__nv_bfloat16",
            "-DINPUT_FP16=" + std::to_string(input_fp16),
            "-DHEAD_ELEMS=" + std::to_string(head_dim),
            "-DBEAM_WIDTH=" + std::to_string(beam_width),
            "-DCACHE_ELEM_ENUM=" + std::to_string(cache_enum),
            "-DTOKENS_PER_PAGE=" + std::to_string(use_paged_kv_cache ? 32 : 0),
            "-DHEAD_GRP_SIZE=" + std::to_string(head_grp_size),
            "-DM_TILESIZE=" + std::to_string(head_grp_size),
            "-DUSE_CUSTOM_BARRIER=1",
        };
        if (cache_enum == 2 && source_file == tensorrt_llm::kernels::mha_sm90_cu_content)
        {
            options.push_back("-DUSE_INPUT_KV=1");
            options.push_back("-DROPE_STYLE=1");
            options.push_back("-DSLIDING_WINDOW=1");
            options.push_back("-DLOW_PREC_OUTPUT=1");
        }
        std::vector<char const*> options_cstr;
        for (auto const& option : options)
        {
            options_cstr.push_back(option.c_str());
        }

        nvrtcProgram program;
        std::string log;

        NVRTC_RUN(nvrtcCreateProgram(
            &program, source_file, "program", headers_content.size(), headers_content.data(), headers_name.data()));
        auto status = nvrtcCompileProgram(program, options_cstr.size(), options_cstr.data());
        if (status != NVRTC_SUCCESS)
        {
            size_t log_size;
            NVRTC_RUN(nvrtcGetProgramLogSize(program, &log_size));
            log.resize(log_size);
            NVRTC_RUN(nvrtcGetProgramLog(program, const_cast<char*>(log.data())));
            FAIL() << log;
        }

        size_t cubinSize;
        NVRTC_RUN(nvrtcGetCUBINSize(program, &cubinSize));
        ASSERT_GT(cubinSize, 1000);
        std::string cubinContent(cubinSize, ' ');
        NVRTC_RUN(nvrtcGetCUBIN(program, const_cast<char*>(cubinContent.c_str())));

        NVRTC_RUN(nvrtcDestroyProgram(&program));

        if (compileMajor == major && compileMinor == minor)
        {
            CUmodule module;
            CU_RUN(cuModuleLoadData(&module, static_cast<void const*>(cubinContent.c_str())));
            CUfunction function;
            CU_RUN(cuModuleGetFunction(&function, module, "kernel_mha"));
            ASSERT_NE(function, nullptr);
            CUdeviceptr shmem_dev_ptr;
            CU_RUN(cuModuleGetGlobal(&shmem_dev_ptr, nullptr, module, "smemSize"));
            unsigned int shmem_bytes = 0;
            CU_RUN(cuMemcpyDtoH(&shmem_bytes, shmem_dev_ptr, sizeof(unsigned int)));
            ASSERT_GT(shmem_bytes, 1000);
        }
    };

    test(0, 2, 576, 128, true, 1, tensorrt_llm::kernels::mla_sm120_cu_content, 12, 0);

    std::pair<char const* const, std::function<bool(int, int)>> const sourceFileAndArchCond[] = {
        {tensorrt_llm::kernels::mha_cu_content, [](int major, int minor) { return major >= 8; }},
        {tensorrt_llm::kernels::mha_sm90_cu_content, [](int major, int minor) { return major == 9 && minor == 0; }}};
    for (int input_fp16 : {0, 1})
    {
        for (int cache_enum : {0, 1, 2})
        {
            for (int head_dim : {64, 128, 256})
            {
                for (bool use_paged_kv_cache : {false, true})
                {
                    for (int beam_width : {1, 4})
                    {
                        for (auto const& [source_file, archCond] : sourceFileAndArchCond)
                        {
                            if (!archCond(major, minor))
                            {
                                continue;
                            }
                            if ((source_file == tensorrt_llm::kernels::mha_sm90_cu_content)
                                && !(cache_enum == 2 && beam_width == 1))
                            {
                                continue;
                            }
                            test(input_fp16, cache_enum, head_dim, 8, use_paged_kv_cache, beam_width, source_file,
                                major, minor);
                        }
                    }
                }
            }
        }
    }
}
#endif
#endif
