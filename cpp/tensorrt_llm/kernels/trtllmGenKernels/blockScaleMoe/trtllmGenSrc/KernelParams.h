/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

//// FIX
#define TLLM_ENABLE_CUDA
// #include "BatchedGemmOptions.h"
#include "Dtype.h"
#include "SfLayoutDecl.h"

#include <cstdint>
#ifdef TLLM_ENABLE_CUDA
#include <cute/tensor.hpp>
#endif

#include "TmaDescriptor.h"

////////////////////////////////////////////////////////////////////////////////////////////////////

// TODO: Find a better header to put this in, that we can include from here.
template <typename T>
inline T ceilDiv(T m, T n)
{
    return (m + n - T(1)) / n;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace tg = trtllm::gen;

////////////////////////////////////////////////////////////////////////////////////////////////////

struct KernelParams
{
#ifdef TLLM_ENABLE_CUDA

    // Maximum number of batch
    static constexpr int MaxBatchSize = 256;
    // Maximum number of CTAs
    static constexpr int MaxNumCtas = 2048;

    // TMA descriptor for A.
    CUtensorMap tmaA[1];
    // TMA descriptor for B.
    CUtensorMap tmaB[1];
    // TMA descriptor for C, (when useTmaStore is true)
    CUtensorMap tmaC[1];

    // TMA descriptor for block scaling factors for A.
    CUtensorMap tmaSfA[1];
    // TMA descriptor for block scaling factors for B.
    CUtensorMap tmaSfB[1];

    // The output matrix C. The shape is m x n. Layout is row-major (contiguous in the n dimension).
    // (when useTmaStore is false)
    void* ptrC;

    // The device output scale for FP8 quantization. Only needed by trt-llm fp8 kernels as the sca-
    // les have to be on the device currently.
    float const* ptrScaleC;

    // in case we fuse activation, the output scale for FP8 gate quantization.
    float const* ptrScaleGate;

    // The K dimension. It is the hidden dimension of the input matrices.
    int32_t k;

    // The non-batched dimension
    int32_t nm;

    // tile Stride per batch for the non-batched dimension
    int32_t tileStridePerBatch;

    // A map from CTA index X/Y to batch index
    int32_t ctaIdxXyToBatchIdx[MaxNumCtas];
    // A map from CTA index X/Y to tile index **expanded** M/N  for batched dimension,
    // This is now an identity map and it is no longer needed.
    // int32_t ctaIdxXyToTileIdxMn[MaxNumCtas];

    // DeepSeek FP8 scaling factors for A
    float* ptrDqSfsA;
    // DeepSeek FP8 scaling factors for B
    float* ptrDqSfsB;
    // DeepSeek FP8 scaling factors for C
    float* ptrDqSfsC;
    // Total number of padded tokens - used as the stride for the activation and C scaling factors
    int32_t totalNumPaddedTokens;

    // The rank id.
    int rank;
    // The number of peer devices in tensor-parallel group.
    int tpGrpSize;

    // The barriers in global memory.
    //
    // The kernel arrives on (with release ordering) the multicast mapping of the barrier to broadcast
    // amongst peer devices. It then waits (with acquire ordering) on the unicast mapping of the
    // barrier.
    //
    // Flags in global memory that sync on "entrance" of reduce-scatter phase in two-shot all-reduce.
    void* ptrTileBars;
    void* multimemTileBars;

    // Flags in global memory that sync on "exit" after the all-reduce finishes.
    void* ptrCompletionBars;
    void* multimemCompletionBars;

    // Pointer for partial row max for DeepSeek computation.
    float* ptrPartialRowMax;
    // Flags in global memory that sync on "exit" for row max computation.
    uint32_t* ptrRowMaxCompletionBars;

    // The input matrix A. The shape is m x k. Layout is row-major (contiguous in the k dimension).
    void* ptrA;
    // The stride for matrix A in bytes.
    uint64_t strideInBytesA;

    // The input matrix B. The shape is k x n. Layout is column-major (contiguous in the k dimension).
    void* ptrB;
    // The stride for matrix A in bytes.
    uint64_t strideInBytesB;

    // The block scaling factors for A
    void* ptrSfA;
    // The block scaling factors for B
    void* ptrSfB;
    // The block scaling factors for C
    void* ptrSfC;

    // **Expanded** limits for the batched dimension:
    //   tile * ctaIdxXyToTileIdxMn[ctaIdxXy] -> ctaIdxXyToMnLimit[ctaIdxXy]
    int32_t ctaIdxXyToMnLimit[MaxNumCtas];

    // The input tokens (used when routeAct is enabled)
    void* ptrTokens;
    // The routeMap for the input tokens (used when routeAct is enabled)
    int32_t const* routeMap;
    // The scaling factors for DeepSeek FP8 for the tokens (used when routeAct is enabled)
    float* ptrDqSfsTokens;
    // The scaling factors for NVFP4 for the tokens (used when routeAct is enabled)
    void* ptrSfTokens;
    // Number of tokens
    int32_t numTokens;

    // In some cases, some CTAs must early-exit. E.g. when the grid size is set statically, but the
    // actual workload is decided at runtime. This element on the device contains the number of CTAs
    // that do not early-exit. The number corresponds to the X dim of the grid when the output is not
    // transposed (i.e. transposeMmaOutput is false). To the Y dim, otherwise.
    int32_t* ptrNumNonExitingCtas;
    // Pointer to total number of padded tokens
    int32_t* ptrTotalNumPaddedTokens;
    // Pointer to CTA index X/Y to batch index
    int32_t* ptrCtaIdxXyToBatchIdx;
    // Pointer to CTA index X/Y to tile index **expanded** M/N for batched dimension
    int32_t* ptrCtaIdxXyToMnLimit;

    enum class MatrixType
    {
        MatrixA = 0,
        MatrixB,
        MatrixC
    };

    // Create the TMA shape/stride for A/B/C.
    static auto makeTmaShapeStrideAbc(
        bool const transposeMmaOutput, bool const useFusedAct, int mM, int mN, int mK, MatrixType matrixType)
    {
        auto numTokens = (matrixType == MatrixType::MatrixA || matrixType == MatrixType::MatrixC) ? mM : mN;
        auto hiddenSize = (matrixType == MatrixType::MatrixC) ? mN : mK;
        if (matrixType == MatrixType::MatrixC && transposeMmaOutput)
        {
            numTokens = mN;
            hiddenSize = mM;
        }

        // For a fused activation kernel, the hidden size of output is halved. TODO: That's true for
        // gated activations but not regular activations.
        if (useFusedAct)
        {
            if (matrixType == MatrixType::MatrixC)
                hiddenSize /= 2;
        }

        // The cute tensor shape for A/B: (numTokens, hiddenSize).
        // Note that TMA descriptor expects the first dimension's stride to be
        // 1, so swap the first two dimension so that the hiddenSize dimension comes first.
        auto shape = std::vector<uint64_t>{static_cast<uint64_t>(hiddenSize), static_cast<uint64_t>(numTokens)};

        // Assemble the stride (strideTokens, 1).
        // Swap the first two dimension as mentioned before.
        auto stride = std::vector<uint64_t>{1, static_cast<uint64_t>(hiddenSize)};

        return std::make_tuple(shape, stride);
    }

    // Create the TMA shape/stride for A/B block scale factors.
    static auto makeTmaShapeStrideSfAb(
        int mM, int mN, int mK, MatrixType matrixType, int tileM, int tileN, int tileK, tg::SfLayout layout)
    {

        // The outer dimension.
        auto numTokens = matrixType == MatrixType::MatrixA ? mM : mN;
        // The inner dimension.
        auto hiddenSize = mK;
        // The outer tile dimension.
        auto numTokensPerTile = matrixType == MatrixType::MatrixA ? tileM : tileN;
        // The inner tile dimension.
        auto hiddenSizePerTile = tileK;
        // Number of elements per scaling factor.
        const int32_t numEltsPerSf = 16;

        switch (layout)
        {
        case tg::SfLayout::R128c4:
        {
            // The scale factor tensor packs 128x4 tiles into contiguous 512B blocks.
            // The 512B block maps to a 32x16B (32x128b) block in TMEM.
            // See https://nvbugspro.nvidia.com/bug/4165523
            //
            // Additionally, we have to meet constraints of TMA that the box dimensions are less
            // than 256 and boxDim[0] is a multiple of 16B.
            //
            // The "logical" tensor is:      [outer,       inner / numEltsPerSf]
            // The aforementioned format is: [outer / 128, inner / numEltsPerSf / 4,    512]
            // The shape we use for TMA is:  [outer / 128, inner / numEltsPerSf / 4, 2, 256]

            auto shape = std::vector<uint64_t>{256, 2, static_cast<uint64_t>(ceilDiv(hiddenSize, numEltsPerSf * 4)),
                static_cast<uint64_t>(ceilDiv(numTokens, 128))};

            std::vector<uint64_t> stride(shape.size());
            stride[0] = 1;
            for (size_t i = 1; i < shape.size(); i++)
            {
                stride[i] = shape[i - 1] * stride[i - 1];
            }

            auto tileShapes
                = std::vector<uint32_t>{256, 2, static_cast<uint32_t>(ceilDiv(hiddenSizePerTile, numEltsPerSf * 4)),
                    static_cast<uint32_t>(ceilDiv(numTokensPerTile, 128))};

            return std::make_tuple(shape, stride, tileShapes);
        }

        case tg::SfLayout::R8c4:
        {
            // The scale factor tensor packs 8x4 tiles into contiguous 32B blocks.
            //
            // As the inner dimension (k) is required to be a multiple of the tile size, we
            // can reshape to use fewer read requests, if the tile dimensions allow.
            // I.e., let's define repeats = min(hiddenSizePerTile / numEltsPerSf / 4, 8)
            //
            // The "logical" tensor is: [outer,     inner / numEltsPerSf]
            // The 8x4 SF layout is:    [outer / 8, inner / numEltsPerSf / 4, 32]
            // The TMA tensor shape is: [outer / 8, inner / numEltsPerSf / 4 / repeats, repeats * 32]

            int const repeats = std::min(ceilDiv(hiddenSizePerTile, numEltsPerSf * 4), 8);

            auto shape = std::vector<uint64_t>{static_cast<uint64_t>(repeats * 32),
                static_cast<uint64_t>(ceilDiv(hiddenSize, numEltsPerSf * 4 * repeats)),
                static_cast<uint64_t>(ceilDiv(numTokens, 8))};

            std::vector<uint64_t> stride(shape.size());
            stride[0] = 1;
            for (size_t i = 1; i < shape.size(); i++)
            {
                stride[i] = shape[i - 1] * stride[i - 1];
            }

            auto tileShapes = std::vector<uint32_t>{static_cast<uint32_t>(repeats * 32),
                static_cast<uint32_t>(ceilDiv(hiddenSizePerTile, numEltsPerSf * 4 * repeats)),
                static_cast<uint32_t>(ceilDiv(numTokensPerTile, 8))};

            return std::make_tuple(shape, stride, tileShapes);
        }

        default: TLLM_CHECK_ERROR(false, "Unsupported SF layout");
        }
        return std::make_tuple(std::vector<uint64_t>{}, std::vector<uint64_t>{}, std::vector<uint32_t>{});
    }

    static KernelParams setKernelParams(const int32_t numBatches, const int32_t numTokens,
        const int32_t permutedRowCount, bool const batchM, const int32_t m, const int32_t n, const int32_t k,
        std::vector<int32_t> batchedM, std::vector<int32_t> batchedN, int const tileM, int const tileN, int const tileK,
        int const epilogueTileM, int const epilogueTileN, bool const useDeepSeekFp8, bool const useTmaStore,
        bool const transposeMmaOutput, tg::SfLayout sfLayoutB, bool const useFusedAct, bool const allToAllRouteAct,
        tg::Dtype dtypeElt, tg::Dtype dtypeC, void* ptrA, void* ptrB, void* ptrC, float const* ptrScaleC,
        float* dDqSfsA, float* dDqSfsB, float* dDqSfsC, void* dSfA, void* dSfB, void* dSfTokens, void* dSfC,
        float const* ptrScaleGate, void* ptrTokens, int32_t const* routeMap, float* dDqSfsTokens, float* rowMax,
        uint32_t* rowMaxBars, bool isStaticBatch = true, int32_t* ptrNumNonExitingCtas = nullptr,
        int32_t* ptrTotalNumPaddedTokens = nullptr, int32_t* ptrCtaIdxXyToBatchIdx = nullptr,
        int32_t* ptrCtaIdxXyToMnLimit = nullptr)
    {

        // std::cout << "numBatches: " << numBatches << std::endl;
        // std::cout << "numTokens: " << numTokens << std::endl;
        // std::cout << "batchM: " << batchM << std::endl;
        // std::cout << "m: " << m << std::endl;
        // std::cout << "n: " << n << std::endl;
        // std::cout << "k: " << k << std::endl;
        // std::cout << "tileM: " << tileM << std::endl;
        // std::cout << "tileN: " << tileN << std::endl;
        // std::cout << "tileK: " << tileK << std::endl;
        // std::cout << "epilogueTileM: " << epilogueTileM << std::endl;
        // std::cout << "epilogueTileN: " << epilogueTileN << std::endl;
        // std::cout << "useDeepSeekFp8: " << useDeepSeekFp8 << std::endl;
        // std::cout << "useTmaStore: " << useTmaStore << std::endl;
        // std::cout << "transposeMmaOutput: " << transposeMmaOutput << std::endl;
        // std::cout << "sfLayoutB: " << (int) sfLayoutB << std::endl;
        // std::cout << "useFusedAct: " << useFusedAct << std::endl;
        // std::cout << "allToAllRouteAct: " << allToAllRouteAct << std::endl;
        // std::cout << "dtypeElt: " << (int) dtypeElt << std::endl;
        // std::cout << "dtypeC: " << (int) dtypeC << std::endl;
        // std::cout << "ptrA: " << ptrA << std::endl;
        // std::cout << "ptrB: " << ptrB << std::endl;
        // std::cout << "ptrC: " << ptrC << std::endl;
        // std::cout << "ptrScaleC: " << ptrScaleC << std::endl;
        // std::cout << "ptrScaleGate: " << ptrScaleGate << std::endl;
        // // std::cout << "dDqSfsA: " << dDqSfsA << std::endl;
        // // std::cout << "dDqSfsB: " << dDqSfsB << std::endl;
        // // std::cout << "dDqSfsC: " << dDqSfsC << std::endl;
        // std::cout << "dSfA: " << dSfA << std::endl;
        // std::cout << "dSfB: " << dSfB << std::endl;
        // std::cout << "dSfTokens: " << dSfTokens << std::endl;
        // std::cout << "dSfC: " << dSfC << std::endl;
        // std::cout << "ptrTokens: " << ptrTokens << std::endl;
        // std::cout << "routeMap: " << routeMap << std::endl;
        // std::cout << "isStaticBatch: " << isStaticBatch << std::endl;
        // std::cout << "ptrNumNonExitingCtas: " << ptrNumNonExitingCtas << std::endl;
        // std::cout << "ptrTotalNumPaddedTokens: " << ptrTotalNumPaddedTokens << std::endl;
        // std::cout << "ptrCtaIdxXyToBatchIdx: " << ptrCtaIdxXyToBatchIdx << std::endl;
        // std::cout << "ptrCtaIdxXyToMnLimit: " << ptrCtaIdxXyToMnLimit << std::endl;
        // std::cout << "batchedM.size(): " << batchedM.size() << std::endl;
        // std::cout << "batchedN.size(): " << batchedN.size() << std::endl;

        static_assert(sizeof(KernelParams) <= 32 * 1024, "sizeof(KernelParams) has to be less or equal than 32KB");

        // Create the return struct.
        KernelParams params;

        TLLM_CHECK_ERROR(numBatches <= KernelParams::MaxBatchSize, "GEMM batch limit reached.");

        params.ptrTokens = ptrTokens;
        params.routeMap = routeMap;
        params.ptrDqSfsTokens = dDqSfsTokens;
        params.numTokens = numTokens;
        params.ptrSfTokens = dSfTokens;

        params.ptrScaleC = ptrScaleC;
        params.ptrScaleGate = ptrScaleGate;

        int32_t ctaOffset = 0;

        // Compute totalNumPaddedTokens, ctaIdxXyToBatchIdx and ctaIdxXyToMnLimit if the batch dims are
        // known at kernel launch time. Otherwise, these parameters are defined in the device buffers:
        // ptrTotalNumPaddedTokens, ptrCtaIdxXyToBatchIdx and ptrCtaIdxXyToMnLimit respectively.

        if (isStaticBatch)
        {
            params.totalNumPaddedTokens = 0;
            for (int b = 0; b < numBatches; b++)
            {

                int mM = batchM ? batchedM[b] : n;
                int mN = batchM ? m : batchedN[b];

                // Skip Tma descriptor creation if expert isn't used
                if (mM == 0 || mN == 0)
                    continue;

                // The number of CTAs.
                int32_t numCtas = batchM ? (mM + tileM - 1) / tileM : (mN + tileN - 1) / tileN;
                // The size of the tile.
                int32_t tile = batchM ? tileM : tileN;
                // The problem size.
                int32_t mn = batchM ? mM : mN;
                // In case of allToAllRouteAct, we run each expert with all input tokens.
                int32_t tokensPerTile = allToAllRouteAct ? numTokens : mn;

                // Make sure we do not exceed the launch limit.
                TLLM_CHECK_ERROR(ctaOffset + numCtas <= MaxNumCtas, "Too many CTAs");

                for (int32_t cta = 0; cta < numCtas; cta++)
                {
                    params.ctaIdxXyToBatchIdx[ctaOffset + cta] = b;
                    // This is now an identity map and it is no longer needed.
                    // params.ctaIdxXyToTileIdxMn[ctaOffset + cta] = ctaOffset + cta;
                    params.ctaIdxXyToMnLimit[ctaOffset + cta]
                        = std::min((ctaOffset + cta + 1) * tile, ctaOffset * tile + tokensPerTile);
                }
                ctaOffset += numCtas;

                params.totalNumPaddedTokens += numCtas * tile;
            }
        }
        else
        {
            params.ptrTotalNumPaddedTokens = ptrTotalNumPaddedTokens;
            params.ptrCtaIdxXyToBatchIdx = ptrCtaIdxXyToBatchIdx;
            params.ptrCtaIdxXyToMnLimit = ptrCtaIdxXyToMnLimit;
            ctaOffset = MaxNumCtas;
        }

        if (useDeepSeekFp8 && dtypeElt == tg::Dtype::E4m3)
        {
            params.ptrDqSfsA = dDqSfsA;
            params.ptrDqSfsB = dDqSfsB;
        }

        if (useDeepSeekFp8 && dtypeC == tg::Dtype::E4m3)
        {
            params.ptrDqSfsC = dDqSfsC;
        }

        params.ptrA = ptrA;
        params.ptrB = ptrB;
        params.strideInBytesA = k * tg::dtypeGetNumBits(dtypeElt) / 8;
        params.strideInBytesB = k * tg::dtypeGetNumBits(dtypeElt) / 8;

        params.ptrSfA = dSfA;
        params.ptrSfB = dSfB;
        params.ptrSfC = dSfC;

        if (!batchM)
        {
            // A is the expert
            TLLM_CHECK_ERROR(0 == m % tileM, "0 == mM %% tileM");
            params.tileStridePerBatch = m / tileM;
            params.nm = m;
            // Shape/stride for gmem tensor A.
            auto [shapeA, strideA]
                = makeTmaShapeStrideAbc(transposeMmaOutput, useFusedAct, m * numBatches, n, k, MatrixType::MatrixA);
            // Build tma descriptor for A.
            params.tmaA[0] = gemm::buildNdTmaDescriptor(dtypeElt, shapeA, strideA, tileM, tileK, ptrA);

            // When the allToAllRouteAct, the input activations are packed [act0, act1, ...]
            // Otherwise, the input is padded:
            // [act0, padding, padding, ... TileN size .., act1, padding, padding, ...]
            auto const inputNumTokens = allToAllRouteAct ? numTokens : permutedRowCount;
            // B is the activation
            // Shape/stride for gmem tensor B.
            auto [shapeB, strideB]
                = makeTmaShapeStrideAbc(transposeMmaOutput, useFusedAct, m, inputNumTokens, k, MatrixType::MatrixB);
            // Load data from contiguous ptrToken buffer for allToAllRouteAct
            auto ptrTokensB = allToAllRouteAct ? ptrTokens : ptrB;
            // Build tma descriptor for B.
            params.tmaB[0] = gemm::buildNdTmaDescriptor(dtypeElt, shapeB, strideB, tileN, tileK, ptrTokensB);

            if (dtypeElt == tg::Dtype::E2m1)
            {
                const tg::Dtype dTypeSf = tg::Dtype::E4m3;

                // Build TMA descriptor for gmem A block scale factors.
                auto [shapeSfA, strideSfA, tileShapesSfA] = makeTmaShapeStrideSfAb(
                    m * numBatches, n, k, MatrixType::MatrixA, tileM, tileN, tileK, tg::SfLayout::R128c4);
                params.tmaSfA[0] = gemm::buildSfTmaDescriptor(dTypeSf, shapeSfA, strideSfA, tileShapesSfA, dSfA);

                // When the allToAllRouteAct, the input activations are packed [act0, act1, ...]
                // Otherwise, the input is padded:
                // [act0, padding, padding, ... TileN size .., act1, padding, padding, ...]
                //
                // Due to the TileN=128 restriction, we must set num tokens
                // per SfB to 128 for allToAllRouteAct.
                auto const inputNumTokensSfB = allToAllRouteAct ? tileN : permutedRowCount;

                // Load data from contiguous dSfTokens buffer for allToAllRouteAct
                auto ptrSfTokensB = allToAllRouteAct ? dSfTokens : dSfB;
                // Build TMA descriptor for gmem B block scale factors.
                auto [shapeSfB, strideSfB, tileShapesSfB] = makeTmaShapeStrideSfAb(
                    m, inputNumTokensSfB, k, MatrixType::MatrixB, tileM, tileN, tileK, sfLayoutB);
                params.tmaSfB[0]
                    = gemm::buildSfTmaDescriptor(dTypeSf, shapeSfB, strideSfB, tileShapesSfB, ptrSfTokensB);
            }

            // C is the output activation
            if (useTmaStore)
            {
                // Shape/stride for gmem tensor C.
                auto [shapeC, strideC] = makeTmaShapeStrideAbc(
                    transposeMmaOutput, useFusedAct, m, permutedRowCount, k, MatrixType::MatrixC);

                // Swap M and N tiles for the M-major epilogue.
                auto outputTileM = transposeMmaOutput ? epilogueTileN : epilogueTileM;
                auto outputTileN = transposeMmaOutput ? epilogueTileM : epilogueTileN;

                if (useFusedAct)
                {
                    // for a fused activation kernel, output tile `N` is halved
                    outputTileN /= 2;
                }
                // Build tma descriptor for C.
                params.tmaC[0] = gemm::buildNdTmaDescriptor(dtypeC, shapeC, strideC, outputTileM, outputTileN, ptrC);
            }
            else
            {
                params.ptrC = ptrC;
            }
        }
        else
        {
            // B is the expert
            TLLM_CHECK_ERROR(0 == n % tileN, "0 == mN %% tileN");
            params.tileStridePerBatch = n / tileN;
            params.nm = n;
            // Shape/stride for gmem tensor B.
            auto [shapeB, strideB]
                = makeTmaShapeStrideAbc(transposeMmaOutput, useFusedAct, m, n * numBatches, k, MatrixType::MatrixB);
            // Build tma descriptor for B.
            params.tmaB[0] = gemm::buildNdTmaDescriptor(dtypeElt, shapeB, strideB, tileN, tileK, ptrB);

            // A is the activation
            // Shape/stride for gmem tensor A.
            // When the allToAllRouteAct, the input activations are packed [act0, act1, ...]
            // Otherwise, the input is padded:
            // [act0, padding, padding, ... tileM size .., act1, padding, padding, ...]
            auto const inputNumTokens = allToAllRouteAct ? numTokens : permutedRowCount;
            auto [shapeA, strideA]
                = makeTmaShapeStrideAbc(transposeMmaOutput, useFusedAct, inputNumTokens, n, k, MatrixType::MatrixA);
            // Load data from contiguous ptrToken buffer for allToAllRouteAct
            auto ptrTokensA = allToAllRouteAct ? ptrTokens : ptrA;
            // Build tma descriptor for A.
            params.tmaA[0] = gemm::buildNdTmaDescriptor(dtypeElt, shapeA, strideA, tileM, tileK, ptrTokensA);

            if (dtypeElt == tg::Dtype::E2m1)
            {
                const tg::Dtype dTypeSf = tg::Dtype::E4m3;

                // When the allToAllRouteAct, the input activations are packed [act0, act1, ...]
                // Otherwise, the input is padded:
                // [act0, padding, padding, ... tileM size .., act1, padding, padding, ...]
                //
                // Due to the tileM=128 restriction, we must set num tokens
                // per SfA to 128 for allToAllRouteAct.
                auto const inputNumTokensSfA = allToAllRouteAct ? tileM : permutedRowCount;

                // Load data from contiguous dSfTokens buffer for allToAllRouteAct
                auto ptrSfTokensA = allToAllRouteAct ? dSfTokens : dSfA;

                // Build TMA descriptor for gmem A block scale factors.
                auto [shapeSfA, strideSfA, tileShapesSfA] = makeTmaShapeStrideSfAb(
                    inputNumTokensSfA, n, k, MatrixType::MatrixA, tileM, tileN, tileK, tg::SfLayout::R128c4);
                params.tmaSfA[0]
                    = gemm::buildSfTmaDescriptor(dTypeSf, shapeSfA, strideSfA, tileShapesSfA, ptrSfTokensA);

                // Build TMA descriptor for gmem B block scale factors.
                auto [shapeSfB, strideSfB, tileShapesSfB]
                    = makeTmaShapeStrideSfAb(m, n * numBatches, k, MatrixType::MatrixB, tileM, tileN, tileK, sfLayoutB);
                params.tmaSfB[0] = gemm::buildSfTmaDescriptor(dTypeSf, shapeSfB, strideSfB, tileShapesSfB, dSfB);
            }

            // C is the output activation
            if (useTmaStore)
            {
                // Shape/stride for gmem tensor C.
                auto [shapeC, strideC] = makeTmaShapeStrideAbc(
                    transposeMmaOutput, useFusedAct, permutedRowCount, n, k, MatrixType::MatrixC);

                // Swap M and N tiles for the M-major epilogue.
                auto outputTileM = transposeMmaOutput ? epilogueTileN : epilogueTileM;
                auto outputTileN = transposeMmaOutput ? epilogueTileM : epilogueTileN;

                if (useFusedAct)
                {
                    // for a fused activation kernel, output tile `N` is halved
                    outputTileN /= 2;
                }
                // Build tma descriptor for C.
                params.tmaC[0] = gemm::buildNdTmaDescriptor(dtypeC, shapeC, strideC, outputTileM, outputTileN, ptrC);
            }
            else
            {
                params.ptrC = ptrC;
            }
        }

        params.k = k;

        params.rank = 0;
        params.tpGrpSize = 1;

        params.ptrTileBars = nullptr;
        params.multimemTileBars = nullptr;

        params.ptrCompletionBars = nullptr;
        params.multimemCompletionBars = nullptr;

        params.ptrPartialRowMax = rowMax;
        params.ptrRowMaxCompletionBars = rowMaxBars;

        params.ptrNumNonExitingCtas = ptrNumNonExitingCtas;

        return params;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    template <class GemmOptions_>
    static KernelParams setKernelParams(GemmOptions_ const& options, void* ptrA, void* ptrB, void* ptrC,
        float const* ptrScaleC, float* dDqSfsA, float* dDqSfsB, float* dDqSfsC, void* dSfA, void* dSfB, void* dSfTokens,
        void* dSfC, float const* ptrScaleGate, void* ptrTokens, int32_t const* routeMap, float* dDqSfsTokens,
        float* rowMax, uint32_t* rowMaxBars, int32_t* ptrNumNonExitingCtas = nullptr,
        int32_t* ptrTotalNumPaddedTokens = nullptr, int32_t* ptrCtaIdxXyToBatchIdx = nullptr,
        int32_t* ptrCtaIdxXyToMnLimit = nullptr)
    {

        bool const useFusedAct = options.mUseFusedAct;
        // Taken from moe::getMaxPermutedPaddedCount
        int const expandedRowCount = options.mNumTokens * options.mTopK;
        int const maxPaddingRequired = ((options.mBatchM ? options.mTileM : options.mTileN) - 1) * options.mNumExperts;
        const int32_t permutedRowCount = expandedRowCount + maxPaddingRequired;

        return setKernelParams(options.mNumBatches, options.mNumTokens, permutedRowCount, options.mBatchM, options.mM,
            options.mN, options.mK, options.mBatchedM, options.mBatchedN, options.mTileM, options.mTileN,
            options.mTileK, options.mEpilogueTileM, options.mEpilogueTileN, options.mUseDeepSeekFp8,
            options.mUseTmaStore, options.mTransposeMmaOutput, options.mSfLayoutB, useFusedAct,
            options.mAllToAllRouteAct, options.mDtypeElt, options.mDtypeC, ptrA, ptrB, ptrC, ptrScaleC, dDqSfsA,
            dDqSfsB, dDqSfsC, dSfA, dSfB, dSfTokens, dSfC, ptrScaleGate, ptrTokens, routeMap, dDqSfsTokens, rowMax,
            rowMaxBars, options.mIsStaticBatch, ptrNumNonExitingCtas, ptrTotalNumPaddedTokens, ptrCtaIdxXyToBatchIdx,
            ptrCtaIdxXyToMnLimit);
    }
#endif
};

////////////////////////////////////////////////////////////////////////////////////////////////////
