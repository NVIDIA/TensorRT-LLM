/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
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

#include <cstdint>
#ifdef TLLM_ENABLE_CUDA
#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/half.h>
#endif

#include "TmaDescriptor.h"

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace tg = trtllm::gen;

////////////////////////////////////////////////////////////////////////////////////////////////////

struct KernelParams
{
#ifdef TLLM_ENABLE_CUDA
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

    // Maximum number of batch
    static constexpr int MaxBatchSize = 256;
    // Maximum number of CTAs
    static constexpr int MaxNumCtas = 2048;

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

    void* ptrSfA;
    void* ptrSfB;

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

        if (useFusedAct)
        {
            // for a fused activation kernel, hidden size of output is halved
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
    static auto makeTmaShapeStrideSfAb(int mM, int mN, int mK, MatrixType matrixType)
    {
        // Note: the scale factor tensor packs 128x4 tiles into contiguous 512B blocks.
        // The 512B block maps to a 32x16B (32x128b) block in TMEM.
        // See https://nvbugspro.nvidia.com/bug/4165523
        //
        // Additionally, we have to meet constraints of TMA that the box dimensions are less
        // than 256 and boxDim[0] is a multiple of 16B.
        //
        // The "logical" tensor is:      [outer,        inner / numEltsPerSf]
        // The aforementioned format is: [outer / 128,  inner / numEltsPerSf / 4,    512]
        // The shape we use for TMA is:  [outer / 128,  inner / numEltsPerSf / 4, 2, 256]

        // The outer dimension.
        auto numTokens = matrixType == MatrixType::MatrixA ? mM : mN;
        // The inner dimension.
        auto hiddenSize = mK;

        const int32_t numEltsPerSf = 16;

        auto shape = std::vector<uint64_t>{
            256, 2, static_cast<uint64_t>((hiddenSize / numEltsPerSf / 4)), static_cast<uint64_t>(numTokens / 128)};

        std::vector<uint64_t> stride(shape.size());
        stride[0] = 1;
        for (size_t i = 1; i < shape.size(); i++)
        {
            stride[i] = shape[i - 1] * stride[i - 1];
        }

        return std::make_tuple(shape, stride);
    }

    static KernelParams setKernelParams(const int32_t numBatches, const int32_t numTokens, bool const batchM,
        const int32_t m, const int32_t n, const int32_t k, std::vector<int32_t> batchedM, std::vector<int32_t> batchedN,
        int const tileM, int const tileN, int const tileK, int const epilogueTileM, int const epilogueTileN,
        bool const useDeepSeekFp8, bool const useTmaStore, bool const transposeMmaOutput, bool const useFusedAct,
        tg::Dtype dtypeElt, tg::Dtype dtypeC, void* ptrA, void* ptrB, void* ptrC, float const* ptrScaleC,
        float* dDqSfsA, float* dDqSfsB, float* dDqSfsC, void* dSfA, void* dSfB, void* dSfTokens,
        float const* ptrScaleGate, void* ptrTokens, int32_t const* routeMap, float* dDqSfsTokens, float* rowMax,
        uint32_t* rowMaxBars)
    {

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
        params.totalNumPaddedTokens = 0;
        for (int b = 0; b < numBatches; b++)
        {

            int mM = batchM ? batchedM[b] : n;
            int mN = batchM ? m : batchedN[b];

            // Skip Tma descriptor creation if expert isn't used
            if (mM == 0 || mN == 0)
                continue;

            int32_t numCta = batchM ? (mM + tileM - 1) / tileM : (mN + tileN - 1) / tileN;

            int32_t tile = batchM ? tileM : tileN;
            int32_t mn = batchM ? mM : mN;

            TLLM_CHECK_ERROR(ctaOffset + numCta <= MaxNumCtas, "Too many CTAs");

            for (int32_t cta = 0; cta < numCta; cta++)
            {
                params.ctaIdxXyToBatchIdx[ctaOffset + cta] = b;
                // This is now an identity map and it is no longer needed.
                // params.ctaIdxXyToTileIdxMn[ctaOffset + cta] = ctaOffset + cta;
                params.ctaIdxXyToMnLimit[ctaOffset + cta]
                    = std::min((ctaOffset + cta + 1) * tile, ctaOffset * tile + mn);
            }
            ctaOffset += numCta;

            params.totalNumPaddedTokens += numCta * tile;
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

            // B is the activation
            // Shape/stride for gmem tensor B.
            auto [shapeB, strideB]
                = makeTmaShapeStrideAbc(transposeMmaOutput, useFusedAct, m, ctaOffset * tileN, k, MatrixType::MatrixB);
            // Build tma descriptor for B.
            params.tmaB[0] = gemm::buildNdTmaDescriptor(dtypeElt, shapeB, strideB, tileN, tileK, ptrB);

            if (dtypeElt == tg::Dtype::E2m1)
            {
                const tg::Dtype dTypeSf = tg::Dtype::E4m3;

                const int32_t numEltsPerSf = 16;

                // Build TMA descriptor for gmem A block scale factors.
                auto [shapeSfA, strideSfA] = makeTmaShapeStrideSfAb(m * numBatches, n, k, MatrixType::MatrixA);
                auto tileShapesSfA = std::vector<uint32_t>{
                    256, 2, static_cast<uint32_t>(tileK / numEltsPerSf / 4), static_cast<uint32_t>(tileM / 128)};
                params.tmaSfA[0] = gemm::buildSfTmaDescriptor(dTypeSf, shapeSfA, strideSfA, tileShapesSfA, dSfA);

                // Build TMA descriptor for gmem B block scale factors.
                auto [shapeSfB, strideSfB] = makeTmaShapeStrideSfAb(m, ctaOffset * tileN, k, MatrixType::MatrixB);
                auto tileShapesSfB = std::vector<uint32_t>{
                    256, 2, static_cast<uint32_t>(tileK / numEltsPerSf / 4), static_cast<uint32_t>(tileN / 128)};
                params.tmaSfB[0] = gemm::buildSfTmaDescriptor(dTypeSf, shapeSfB, strideSfB, tileShapesSfB, dSfB);
            }

            // C is the output activation
            if (useTmaStore)
            {
                // Shape/stride for gmem tensor C.
                auto [shapeC, strideC] = makeTmaShapeStrideAbc(
                    transposeMmaOutput, useFusedAct, m, ctaOffset * tileN, k, MatrixType::MatrixC);

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
            auto [shapeA, strideA]
                = makeTmaShapeStrideAbc(transposeMmaOutput, useFusedAct, ctaOffset * tileM, n, k, MatrixType::MatrixA);
            // Build tma descriptor for A.
            params.tmaA[0] = gemm::buildNdTmaDescriptor(dtypeElt, shapeA, strideA, tileM, tileK, ptrA);

            if (dtypeElt == tg::Dtype::E2m1)
            {
                const tg::Dtype dTypeSf = tg::Dtype::E4m3;

                const int32_t numEltsPerSf = 16;

                // Build TMA descriptor for gmem A block scale factors.
                auto [shapeSfA, strideSfA] = makeTmaShapeStrideSfAb(ctaOffset * tileM, n, k, MatrixType::MatrixA);
                auto tileShapesSfA = std::vector<uint32_t>{
                    256, 2, static_cast<uint32_t>(tileK / numEltsPerSf / 4), static_cast<uint32_t>(tileM / 128)};
                params.tmaSfA[0] = gemm::buildSfTmaDescriptor(dTypeSf, shapeSfA, strideSfA, tileShapesSfA, dSfA);

                // Build TMA descriptor for gmem B block scale factors.
                auto [shapeSfB, strideSfB] = makeTmaShapeStrideSfAb(m, n * numBatches, k, MatrixType::MatrixB);
                auto tileShapesSfB = std::vector<uint32_t>{
                    256, 2, static_cast<uint32_t>(tileK / numEltsPerSf / 4), static_cast<uint32_t>(tileN / 128)};
                params.tmaSfB[0] = gemm::buildSfTmaDescriptor(dTypeSf, shapeSfB, strideSfB, tileShapesSfB, dSfB);
            }

            // C is the output activation
            if (useTmaStore)
            {
                // Shape/stride for gmem tensor C.
                auto [shapeC, strideC] = makeTmaShapeStrideAbc(
                    transposeMmaOutput, useFusedAct, ctaOffset * tileM, n, k, MatrixType::MatrixC);

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

        return params;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    template <class GemmOptions_>
    static KernelParams setKernelParams(GemmOptions_ const& options, void* ptrA, void* ptrB, void* ptrC,
        float const* ptrScaleC, float* dDqSfsA, float* dDqSfsB, float* dDqSfsC, void* dSfA, void* dSfB, void* dSfTokens,
        float const* ptrScaleGate, void* ptrTokens, int32_t const* routeMap, float* dDqSfsTokens, float* rowMax,
        uint32_t* rowMaxBars)
    {

        bool const useFusedAct = options.mUseFusedAct;

        return setKernelParams(options.mNumBatches, options.mNumTokens, options.mBatchM, options.mM, options.mN,
            options.mK, options.mBatchedM, options.mBatchedN, options.mTileM, options.mTileN, options.mTileK,
            options.mEpilogueTileM, options.mEpilogueTileN, options.mUseDeepSeekFp8, options.mUseTmaStore,
            options.mTransposeMmaOutput, useFusedAct, options.mDtypeElt, options.mDtypeC, ptrA, ptrB, ptrC, ptrScaleC,
            dDqSfsA, dDqSfsB, dDqSfsC, dSfA, dSfB, dSfTokens, ptrScaleGate, ptrTokens, routeMap, dDqSfsTokens, rowMax,
            rowMaxBars);
    }
#endif
};

////////////////////////////////////////////////////////////////////////////////////////////////////
