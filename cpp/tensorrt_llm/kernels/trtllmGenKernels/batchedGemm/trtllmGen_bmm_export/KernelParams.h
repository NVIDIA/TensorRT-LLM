/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION &
 * AFFILIATES. All rights reserved. SPDX-License-Identifier: Apache-2.0
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
#pragma once

#include "trtllm/gen/CommonUtils.h"
#include "trtllm/gen/SfLayoutDecl.h"
#include <stdexcept>

#include "BatchedGemmEnums.h"
#include "Enums.h"
#include "TmaDescriptor.h"

// NOTE: keep this code dependency free. It has to be included by the device code and has to be
// compilable with NVRTC.
#include "KernelParamsDecl.h"

namespace batchedGemm
{

namespace batchedGemm
{

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

namespace KernelParamsSetup
{
#ifdef TLLM_ENABLE_CUDA

enum class MatrixType
{
    MatrixA = 0,
    MatrixB,
    MatrixC
};

//////////////////////////////////////////////////////////////////////////////////////////////////
//
// Utility functions.
//
//////////////////////////////////////////////////////////////////////////////////////////////////

template <typename BatchedGemmOptions>
bool useTmaOobOptA(BatchedGemmOptions const& options)
{
    return options.mBatchMode == BatchedGemmOptions::BatchMode::BatchM && doesRouteImplUseNoRoute(options.mRouteImpl)
        && options.mUseTmaOobOpt;
}

//////////////////////////////////////////////////////////////////////////////////////////////////

template <typename BatchedGemmOptions>
bool useTmaOobOptB(BatchedGemmOptions const& options)
{
    return options.mBatchMode == BatchedGemmOptions::BatchMode::BatchN && doesRouteImplUseNoRoute(options.mRouteImpl)
        && options.mUseTmaOobOpt;
}

//////////////////////////////////////////////////////////////////////////////////////////////////

template <typename BatchedGemmOptions>
bool useTmaOobOptC(BatchedGemmOptions const& options)
{
    return options.mUseTmaStore && options.mUseTmaOobOpt;
}

//////////////////////////////////////////////////////////////////////////////////////////////////

// Create the TMA shape/stride for A/B/C.
template <class GemmOptions>
static auto makeTmaShapeStrideAbc(
    GemmOptions const& options, int mM, int mN, int mK, int tileM, int tileN, int tileK, MatrixType matrixType)
{
    // Weights matrix is A if we transpose the output of MMA (to have it M-major).
    // Otherwise, it is B, when the output of MMA is K-major.
    bool const isWeights = (matrixType == MatrixType::MatrixA && options.mTransposeMmaOutput)
        || (matrixType == MatrixType::MatrixB && !options.mTransposeMmaOutput);

    // Whether to use TMA OOB trick to block out padded dummy tokens and saving BW whenever no routing
    // is involved. It applies to batchM and matrixA, or batchN and matrixB, or any case for matrixC.
    bool const useTmaOobOpt = matrixType == MatrixType::MatrixA ? useTmaOobOptA(options)
        : matrixType == MatrixType::MatrixB                     ? useTmaOobOptB(options)
        : matrixType == MatrixType::MatrixC                     ? useTmaOobOptC(options)
                                                                : false;

    // The outer dimension.
    auto numTokens = (matrixType == MatrixType::MatrixA || matrixType == MatrixType::MatrixC) ? mM : mN;
    // The outer dimension tile size.
    auto ctaTileNumTokens = (matrixType == MatrixType::MatrixA || matrixType == MatrixType::MatrixC) ? tileM : tileN;
    // The outer dimension of TMA box shape.
    auto tileNumTokens = (matrixType == MatrixType::MatrixC) ? options.mEpilogueTileM : ctaTileNumTokens;

    // The inner dimension.
    auto hiddenSize = (matrixType == MatrixType::MatrixC) ? mN : mK;
    // The inner dimension tile size.
    auto ctaTileHiddenSize = (matrixType == MatrixType::MatrixC) ? tileN : tileK;
    // The inner dimension of TMA box shape.
    auto tileHiddenSize = (matrixType == MatrixType::MatrixC) ? options.mEpilogueTileN : ctaTileHiddenSize;

    // Swap matrix C sizes if output is transposed.
    if (matrixType == MatrixType::MatrixC && options.mTransposeMmaOutput)
    {
        std::swap(numTokens, hiddenSize);
        std::swap(ctaTileNumTokens, ctaTileHiddenSize);
        std::swap(tileNumTokens, tileHiddenSize);
    }

    // For a fused activation kernel, the hidden size of output is halved. TODO: That's true for
    // gated activations but not regular activations.
    if (options.mFusedAct && matrixType == MatrixType::MatrixC)
    {
        hiddenSize /= 2;
        tileHiddenSize /= 2;
        ctaTileHiddenSize /= 2;
    }

    // The cute tensor shape for A/B: (numTokens, hiddenSize).
    // Note that TMA descriptor expects the first dimension's stride to be
    // 1, so swap the first two dimension so that the hiddenSize dimension comes first.

    // Activations matrix is 2D (sum(divUpMul(M[bi], tileM) for bi in B), K).
    std::vector<uint64_t> shape = {static_cast<uint64_t>(hiddenSize), static_cast<uint64_t>(numTokens)};
    if (useTmaOobOpt /* also implies input/output activation */)
    {
        // If TMA OOB optimization is used:
        // Shape [hidden, tokens]                      Stride [1, hidden] becomes
        // Shape [hidden, tileN, TmaDimMax, TmaDimMax] Stride [1, hidden, XLargeN - hidden, hidden]
        shape = {static_cast<uint64_t>(hiddenSize), static_cast<uint64_t>(ctaTileNumTokens),
            static_cast<uint64_t>(tg::TmaDimMax), static_cast<uint64_t>(tg::TmaDimMax)};
    }
    else if (isWeights)
    {
        // If the matrix is a weights matrix, we use 3D logical shape (B, M, K) or (B, N, K).
        shape = {static_cast<uint64_t>(hiddenSize), static_cast<uint64_t>(numTokens),
            static_cast<uint64_t>(options.mNumBatches)};
    }

    // Assemble the stride (strideTokens, 1).
    // Swap the first two dimension as mentioned before.
    std::vector<uint64_t> stride = {1, static_cast<uint64_t>(hiddenSize)};
    if (useTmaOobOpt)
    {
        stride = {1, static_cast<uint64_t>(hiddenSize), static_cast<uint64_t>(tg::XLargeN - hiddenSize),
            static_cast<uint64_t>(hiddenSize)};
    }
    else if (isWeights)
    {
        stride = {
            1, static_cast<uint64_t>(hiddenSize), static_cast<uint64_t>(hiddenSize) * static_cast<uint64_t>(numTokens)};
    }

    // Assemble the box shape
    std::vector<int32_t> tileShape = {tileHiddenSize, tileNumTokens};

    // Alternate layouts (MajorMn and BlockMajorK) do not apply to matrixC
    if (matrixType != MatrixType::MatrixC)
    {
        // When using 2CTA MMA, we only need to load half of the tile in each CTA for B.
        if (matrixType == MatrixType::MatrixB && tileShape[1] > 1 && options.mClusterDimX == 2)
        {
            tileShape[1] /= 2;
        }
        gemm::MatrixLayout layout = (matrixType == MatrixType::MatrixA) ? options.mLayoutA : options.mLayoutB;
        // Note, only the weights support non MajorK layouts
        if (layout == gemm::MatrixLayout::MajorMn)
        {
            // Apply transpose if necessary
            std::swap(shape[0], shape[1]);
            stride[1] = numTokens;
            std::swap(tileShape[0], tileShape[1]);
        }
        else if (layout == gemm::MatrixLayout::BlockMajorK)
        {
            // Set shapes based on blocking layout
            shape = {static_cast<uint64_t>(options.mBlockK), static_cast<uint64_t>(numTokens),
                static_cast<uint64_t>(mK / options.mBlockK), static_cast<uint64_t>(options.mNumBatches)};
            stride = {1, static_cast<uint64_t>(options.mBlockK), static_cast<uint64_t>(numTokens * options.mBlockK),
                static_cast<uint64_t>(hiddenSize * numTokens)};

            // If blockK > tileK, then the inner most box size will be based on the tile
            int32_t const tileBlockK = std::min(options.mBlockK, tileHiddenSize);
            tileShape = {tileBlockK, tileNumTokens, tileHiddenSize / tileBlockK};
        }
    }

    return std::make_tuple(shape, stride, tileShape);
}

// Create the TMA shape/stride for A/B block scaling factors.
static auto makeTmaShapeStrideSfAb(int mM, int mN, int mK, MatrixType matrixType, int tileM, int tileN, int tileK,
    tg::SfLayout layout, int sfReshapeFactor, const int32_t numEltsPerSf)
{

    // The outer dimension.
    auto numTokens = matrixType == MatrixType::MatrixA ? mM : mN;
    // The inner dimension.
    auto hiddenSize = mK;
    // The outer tile dimension.
    auto numTokensPerTile = matrixType == MatrixType::MatrixA ? tileM : tileN;
    // The inner tile dimension.
    auto hiddenSizePerTile = tileK;

    switch (layout)
    {
    case tg::SfLayout::R128c4:
    {
        // The scaling factor tensor packs 128x4 tiles into contiguous 512B blocks.
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
        // The scaling factor tensor packs 8x4 tiles into contiguous 32B blocks.
        //
        // As the inner dimension (k) is often a multiple of the tile size, we can reshape to use
        // fewer read requests, if the tile dimensions allow. It does not reduce the number of
        // instructions.
        //
        // I.e., let's define r = min(⌈hiddenSizePerTile / (numEltsPerSf * 4)⌉, 8)
        //
        // The "logical" tensor is: [outer,      inner / numEltsPerSf]
        // The 8x4 SF layout is:    [⌈outer / 8⌉, inner / (4 * numEltsPerSf), 32]
        // The TMA tensor shape is: [⌈outer / 8⌉, inner / (4 * numEltsPerSf * r), r * 32]
        //
        // The caveat of NumRepeats>1 is we must pad the hidden dimension of SF to multiples of
        // NumRepeats * numEltsPerSf * 4.

        // Detect if the supplied factor is power of 2. E.g., 0b0100 and (0b0100 - 1) == 0b0000.
        int const r = sfReshapeFactor;
        if (r > 0 && (r & (r - 1)) != 0)
        {
            throw std::runtime_error("mSfReshapeFactor must be positive and a power of 2. Found " + std::to_string(r));
        }

        // Sanitize number of repeats so it doesn't exceed the dimension.
        int const repeats = std::min(ceilDiv(hiddenSizePerTile, numEltsPerSf * 4), r);

        // Detect if the input hidden size K is a multiple of the repeats.
        if (ceilDiv(hiddenSize, numEltsPerSf * 4) % repeats != 0)
        {
            throw std::runtime_error("SF hiddenSize K (" + std::to_string(ceilDiv(hiddenSize, numEltsPerSf * 4))
                + ") must be a multiple of repeats (" + std::to_string(repeats) + ")");
        }

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

    default: throw std::runtime_error("Unsupported SF layout");
    }
    return std::make_tuple(std::vector<uint64_t>{}, std::vector<uint64_t>{}, std::vector<uint32_t>{});
}

template <class GemmOptions_>
static KernelParams setKernelParams(GemmOptions_ const& options, bool const batchM, void const* ptrA, void const* ptrB,
    void* ptrC, void const* dSfA, void const* dSfB, void const* ptrPerTokenSfA, void const* ptrPerTokenSfB,
    void const* ptrBias, void* dSfC, float const* ptrScaleC, float const* ptrScaleGate, float const* ptrClampLimit,
    float const* ptrGatedActAlpha, float const* ptrGatedActBeta, int32_t const* routeMap, float* rowMax,
    uint32_t* rowMaxBars, int32_t const* ptrNumNonExitingCtas = nullptr,
    int32_t const* ptrTotalNumPaddedTokens = nullptr, int32_t const* ptrCtaIdxXyToBatchIdx = nullptr,
    int32_t const* ptrCtaIdxXyToMnLimit = nullptr, int32_t const maxNumCtas = KernelParams::MaxNumCtas)
{

    static_assert(sizeof(KernelParams) <= 32 * 1024, "sizeof(KernelParams) has to be less or equal than 32KB");

    // Create the return struct.
    KernelParams params;

    params.ptrRouteMap = routeMap;
    params.numTokens = options.mNumTokens;

    params.ptrScaleC = ptrScaleC;
    params.ptrScaleGate = ptrScaleGate;
    params.ptrClampLimit = ptrClampLimit;
    params.ptrGatedActAlpha = ptrGatedActAlpha;
    params.ptrGatedActBeta = ptrGatedActBeta;

    int32_t ctaOffset = 0;

    // Compute totalNumPaddedTokens, ctaIdxXyToBatchIdx and ctaIdxXyToMnLimit if the batch dims are
    // known at kernel launch time. Otherwise, these parameters are defined in the device buffers:
    // ptrTotalNumPaddedTokens, ptrCtaIdxXyToBatchIdx and ptrCtaIdxXyToMnLimit respectively.

    if (options.mIsStaticBatch)
    {
        params.totalNumPaddedTokens = 0;
        for (int b = 0; b < options.mNumBatches; b++)
        {

            int mM = batchM ? options.mBatchedM[b] : options.mM;
            int mN = batchM ? options.mN : options.mBatchedN[b];

            // Skip Tma descriptor creation if expert isn't used
            if (mM == 0 || mN == 0)
            {
                continue;
            }

            // The number of CTAs.
            int32_t numCtas
                = batchM ? (mM + options.mTileM - 1) / options.mTileM : (mN + options.mTileN - 1) / options.mTileN;
            // The size of the tile.
            int32_t tile = batchM ? options.mTileM : options.mTileN;
            // The problem size.
            int32_t mn = batchM ? mM : mN;
            int32_t tokensPerTile = mn;

            // Make sure we do not exceed the launch limit.
            if (ctaOffset + numCtas > KernelParams::MaxNumCtas)
            {
                throw std::runtime_error("Too many CTAs");
            }

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
        ctaOffset = maxNumCtas;
    }

    if (options.mUseDeepSeekFp8 && options.mDtypeC == tg::Dtype::E4m3)
    {
        params.ptrDqSfsC = reinterpret_cast<float*>(dSfC);
    }

    params.ptrA = ptrA;
    params.ptrB = ptrB;
    params.strideInBytesA = options.mK * tg::dtypeGetNumBits(options.mDtypeA) / 8;
    params.strideInBytesB = options.mK * tg::dtypeGetNumBits(options.mDtypeB) / 8;

    params.ptrSfA = dSfA;
    params.ptrSfB = dSfB;
    params.ptrSfC = dSfC;

    if (!batchM)
    {
        // A is the expert
        if (0 != options.mM % options.mTileM)
        {
            throw std::runtime_error("0 == mM %% tileM");
        }
        params.tileStridePerBatch = options.mM / options.mTileM;
        params.nm = options.mM;
        // Shape/stride for gmem tensor A.
        auto [shapeA, strideA, tileShapeA] = makeTmaShapeStrideAbc(options, options.mM, options.mN, options.mK,
            options.mTileM, options.mTileN, options.mTileK, MatrixType::MatrixA);
        // Build tma descriptor for A.
        params.tmaA[0] = gemm::buildNdTmaDescriptor(
            options.mDtypeA, options.mMmaKind, shapeA, strideA, tileShapeA, const_cast<void*>(ptrA));

        // The input is padded:
        // [act0, padding, padding, ... TileN size .., act1, padding, padding, ...]
        auto const inputNumTokens = ctaOffset * options.mTileN;

        if (!batchedGemm::doesRouteImplUseLdgsts(options.mRouteImpl))
        {
            bool useRouteAct = batchedGemm::doesRouteImplUseTma(options.mRouteImpl);
            // B is the activation
            // Shape/stride for gmem tensor B.
            auto [shapeB, strideB, tileShapeB] = makeTmaShapeStrideAbc(options, options.mM,
                useRouteAct ? options.mNumTokens : inputNumTokens, options.mK, options.mTileM,
                (useRouteAct ? 1 : options.mTileN), options.mTileK, MatrixType::MatrixB);
            // Build tma descriptor for B.
            params.tmaB[0] = gemm::buildNdTmaDescriptor(
                options.mDtypeB, options.mMmaKind, shapeB, strideB, tileShapeB, const_cast<void*>(ptrB));
        }

        if (options.mDtypeA == tg::Dtype::E2m1 || options.mDtypeA == tg::Dtype::MxE4m3
            || options.mDtypeA == tg::Dtype::MxE2m1)
        {
            tg::Dtype const dTypeSf = (options.mDtypeA == tg::Dtype::E2m1) ? tg::Dtype::E4m3 : tg::Dtype::UE8m0;

            // Build TMA descriptor for gmem A block scaling factors.
            auto [shapeSfA, strideSfA, tileShapesSfA]
                = makeTmaShapeStrideSfAb(options.mM * options.mNumBatches, options.mN, options.mK, MatrixType::MatrixA,
                    options.mTileM, options.mTileN, options.mTileK, tg::SfLayout::R128c4, options.mSfReshapeFactor,
                    options.mSfBlockSizeA.value_or(tg::dtypeNumEltsPerSf(options.mDtypeA)));
            params.tmaSfA[0]
                = gemm::buildSfTmaDescriptor(dTypeSf, shapeSfA, strideSfA, tileShapesSfA, const_cast<void*>(dSfA));
        }

        if (options.mDtypeB == tg::Dtype::E2m1 || options.mDtypeB == tg::Dtype::MxE4m3
            || options.mDtypeB == tg::Dtype::MxE2m1)
        {
            tg::Dtype const dTypeSf = (options.mDtypeB == tg::Dtype::E2m1) ? tg::Dtype::E4m3 : tg::Dtype::UE8m0;

            if (batchedGemm::doesRouteImplUseTma(options.mRouteSfsImpl.value()))
            {

                // The input is NOT padded:
                // [act0, act1, act2, ...]

                // Build TMA descriptor for gmem B block scaling factors.
                int32_t const numEltsPerSf = tg::dtypeNumEltsPerSf(options.mDtypeB);
                // Pad number of scaling factors to the nearest multiple of 16 because of the TMA 16B
                // alignment requirement.
                auto numSfsInK = options.mK / numEltsPerSf;
                numSfsInK = ceilDiv(numSfsInK, 16) * 16;

                auto [shapeSfB, strideSfB, tileShapesSfB]
                    = makeTmaShapeStrideAbc(options, options.mM, options.mNumTokens, numSfsInK, options.mTileM,
                        1 /* tileN */, options.mTileK / numEltsPerSf, MatrixType::MatrixB);
                params.tmaSfB[0] = gemm::buildNdTmaDescriptor(dTypeSf, options.mMmaKind, shapeSfB, strideSfB,
                    tileShapesSfB, const_cast<void*>(dSfB),
                    /*doSwizzle*/ true);
            }
            else if (batchedGemm::doesRouteImplUseNoRoute(options.mRouteSfsImpl.value()))
            {

                // The input is padded:
                // [act0, padding, padding, ... TileN size .., act1, padding, padding, ...]

                auto const inputNumTokensSfB = ctaOffset * options.mTileN;

                // Build TMA descriptor for gmem B block scaling factors.
                auto [shapeSfB, strideSfB, tileShapesSfB] = makeTmaShapeStrideSfAb(options.mM, inputNumTokensSfB,
                    options.mK, MatrixType::MatrixB, options.mTileM, options.mTileN, options.mTileK, options.mSfLayoutB,
                    options.mSfReshapeFactor, tg::dtypeNumEltsPerSf(options.mDtypeB));
                params.tmaSfB[0]
                    = gemm::buildSfTmaDescriptor(dTypeSf, shapeSfB, strideSfB, tileShapesSfB, const_cast<void*>(dSfB));
            }
        }

        // C is the output activation
        if (options.mUseTmaStore)
        {
            // Shape/stride for gmem tensor C.
            auto [shapeC, strideC, tileShapeC] = makeTmaShapeStrideAbc(options, options.mM, ctaOffset * options.mTileN,
                options.mK, options.mTileM, options.mTileN, options.mTileK, MatrixType::MatrixC);
            // Build tma descriptor for C.
            params.tmaC[0]
                = gemm::buildNdTmaDescriptor(options.mDtypeC, tg::MmaKind::Auto, shapeC, strideC, tileShapeC, ptrC);
        }
        else
        {
            params.ptrC = ptrC;
        }
    }
    else
    {
        // B is the expert
        if (0 != options.mN % options.mTileN)
        {
            throw std::runtime_error("0 == mN %% tileN");
        }
        params.tileStridePerBatch = options.mN / options.mTileN;
        params.nm = options.mN;
        // Shape/stride for gmem tensor B.
        auto [shapeB, strideB, tileShapeB] = makeTmaShapeStrideAbc(options, options.mM, options.mN, options.mK,
            options.mTileM, options.mTileN, options.mTileK, MatrixType::MatrixB);
        // Build tma descriptor for B.
        params.tmaB[0] = gemm::buildNdTmaDescriptor(
            options.mDtypeB, options.mMmaKind, shapeB, strideB, tileShapeB, const_cast<void*>(ptrB));

        if (options.mRouteImpl == batchedGemm::RouteImpl::NoRoute)
        {
            // A is the activation
            // Shape/stride for gmem tensor A.
            // The input is padded:
            // [act0, padding, padding, ... tileM size .., act1, padding, padding, ...]
            auto const inputNumTokens = ctaOffset * options.mTileM;
            auto [shapeA, strideA, tileShapeA] = makeTmaShapeStrideAbc(options, inputNumTokens, options.mN, options.mK,
                options.mTileM, options.mTileN, options.mTileK, MatrixType::MatrixA);
            // Build tma descriptor for A.
            params.tmaA[0] = gemm::buildNdTmaDescriptor(
                options.mDtypeA, options.mMmaKind, shapeA, strideA, tileShapeA, const_cast<void*>(ptrA));
        }

        if (options.mDtypeA == tg::Dtype::E2m1 || options.mDtypeA == tg::Dtype::MxE4m3
            || options.mDtypeA == tg::Dtype::MxE2m1)
        {
            tg::Dtype const dTypeSf = (options.mDtypeA == tg::Dtype::E2m1) ? tg::Dtype::E4m3 : tg::Dtype::UE8m0;

            if (options.mRouteSfsImpl.value() == batchedGemm::RouteImpl::NoRoute)
            {

                // The input is padded:
                // [act0, padding, padding, ... tileM size .., act1, padding, padding, ...]
                auto const inputNumTokensSfA = ctaOffset * options.mTileM;

                // Build TMA descriptor for gmem A block scaling factors.
                auto [shapeSfA, strideSfA, tileShapesSfA]
                    = makeTmaShapeStrideSfAb(inputNumTokensSfA, options.mN, options.mK, MatrixType::MatrixA,
                        options.mTileM, options.mTileN, options.mTileK, tg::SfLayout::R128c4, options.mSfReshapeFactor,
                        options.mSfBlockSizeA.value_or(tg::dtypeNumEltsPerSf(options.mDtypeA)));
                params.tmaSfA[0]
                    = gemm::buildSfTmaDescriptor(dTypeSf, shapeSfA, strideSfA, tileShapesSfA, const_cast<void*>(dSfA));
            }
        }

        if (options.mDtypeB == tg::Dtype::E2m1 || options.mDtypeB == tg::Dtype::MxE4m3
            || options.mDtypeB == tg::Dtype::MxE2m1)
        {
            tg::Dtype const dTypeSf = (options.mDtypeB == tg::Dtype::E2m1) ? tg::Dtype::E4m3 : tg::Dtype::UE8m0;

            // Build TMA descriptor for gmem B block scaling factors.
            auto [shapeSfB, strideSfB, tileShapesSfB] = makeTmaShapeStrideSfAb(options.mM,
                options.mN * options.mNumBatches, options.mK, MatrixType::MatrixB, options.mTileM, options.mTileN,
                options.mTileK, options.mSfLayoutB, options.mSfReshapeFactor, tg::dtypeNumEltsPerSf(options.mDtypeB));
            params.tmaSfB[0]
                = gemm::buildSfTmaDescriptor(dTypeSf, shapeSfB, strideSfB, tileShapesSfB, const_cast<void*>(dSfB));
        }

        // C is the output activation
        if (options.mUseTmaStore)
        {
            // Shape/stride for gmem tensor C.
            auto [shapeC, strideC, tileShapeC] = makeTmaShapeStrideAbc(options, ctaOffset * options.mTileM, options.mN,
                options.mK, options.mTileM, options.mTileN, options.mTileK, MatrixType::MatrixC);
            // Build tma descriptor for C.
            params.tmaC[0]
                = gemm::buildNdTmaDescriptor(options.mDtypeC, tg::MmaKind::Auto, shapeC, strideC, tileShapeC, ptrC);
        }
        else
        {
            params.ptrC = ptrC;
        }
    }

    params.k = options.mK;
    params.numBatches = options.mNumBatches;

    params.rank = 0;
    params.tpGrpSize = 1;

    params.ptrPartialRowMax = rowMax;
    params.ptrRowMaxCompletionBars = rowMaxBars;

    params.ptrNumNonExitingCtas = ptrNumNonExitingCtas;

    // Set the per-token scale factors for MetaFP8 or scale inputs
    params.ptrPerTokenSfA = ptrPerTokenSfA;
    params.ptrPerTokenSfB = ptrPerTokenSfB;
    params.ptrBias = ptrBias;

    return params;
}
#endif
}; // namespace KernelParamsSetup

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace batchedGemm

} // namespace batchedGemm
