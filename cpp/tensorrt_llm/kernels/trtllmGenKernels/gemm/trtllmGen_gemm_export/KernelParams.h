/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION &
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

#include <tuple>

#include "trtllm/gen/CommonUtils.h"
#include "trtllm/gen/SfLayoutDecl.h"
#include "trtllm/gen/SparsityDecl.h"

#include "Enums.h"
#include "TmaDescriptor.h"

// NOTE: keep this code dependency free. It has to be included by the device code and has to be
// compilable with NVRTC.
#include "KernelParamsDecl.h"

namespace gemm
{

namespace gemm
{

////////////////////////////////////////////////////////////////////////////////////////////////////
namespace tg = trtllm::gen;

namespace KernelParamsSetup
{
#ifdef TLLM_ENABLE_CUDA

using MatrixType = KernelParams::MatrixType;

// Create the TMA shape/stride for A/B.
template <class GemmOptions>
static auto makeTmaShapeStrideAb(GemmOptions const& options, MatrixType matrixType)
{
    // For sparse A, the k dimension in the TMA shapes is halved.
    int const isSparse = matrixType == MatrixType::MatrixA && tg::isSparse(options.mSparsityA);

    // The padded/valid dimensions.
    int const sizeM = options.mM;
    int const sizeN = options.mN;
    int const sizeK = options.mK >> isSparse;
    int const tileM = options.mTileM;
    int const tileN = options.mTileN;
    int const tileK = options.mTileK >> isSparse;
    int const validM = options.mValidM;
    int const validN = options.mValidN;
    int const validK = options.mValidK >> isSparse;

    // The outer dimension. Uses padded dimensions for strides and valid dimensions for shapes.
    auto numTokens = (matrixType == MatrixType::MatrixA) ? sizeM : sizeN;
    auto numTokensValid = (matrixType == MatrixType::MatrixA) ? validM : validN;
    // The outer dimension tile size.
    auto tileMn = (matrixType == MatrixType::MatrixA) ? tileM : tileN;
    // The inner dimension.
    auto hiddenSize = sizeK;
    auto hiddenSizeValid = validK;
    // The cute tensor shape for A/B: (numTokens, hiddenSize).
    // Note that TMA descriptor expects the first dimension's stride to be
    // 1, so swap the first two dimension so that the hiddenSize dimension comes first.
    // Use valid dimensions for shape, padded dimension for stride.
    auto shape = std::vector<uint64_t>{static_cast<uint64_t>(hiddenSizeValid), static_cast<uint64_t>(numTokensValid)};

    // Assemble the stride (strideTokens, 1).
    // Swap the first two dimension as mentioned before.
    auto stride = std::vector<uint64_t>{1, static_cast<uint64_t>(hiddenSize)};

    // Assemble the box shape
    std::vector<int32_t> tileShape = {tileK, tileMn};
    // When using 2CTA MMA, we only need to load half of the tile in each CTA for B.
    if (matrixType == MatrixType::MatrixB && options.mClusterDimX >= 2)
    {
        tileShape[1] /= 2;
    }

    MatrixLayout layout = (matrixType == MatrixType::MatrixA) ? options.mLayoutA : options.mLayoutB;
    if (layout == MatrixLayout::MajorMn)
    {
        // Apply transpose if necessary
        std::swap(shape[0], shape[1]);
        stride[1] = numTokens;
        std::swap(tileShape[0], tileShape[1]);
    }
    else if (layout == MatrixLayout::BlockMajorK)
    {
        // FIXME: fix for the 2CTA MMA case
        // Set shapes based on blocking layout.
        shape = {static_cast<uint64_t>(options.mBlockK), static_cast<uint64_t>(numTokens),
            static_cast<uint64_t>(sizeK / options.mBlockK)};
        stride = {1, static_cast<uint64_t>(options.mBlockK), static_cast<uint64_t>(numTokens * options.mBlockK)};

        // If blockK > tileK, then the inner most box size will be based on the tile
        int32_t const tileBlockK = std::min(options.mBlockK, tileK);
        tileShape = {tileBlockK, tileMn, tileK / tileBlockK};
        // When using 2CTA MMA, we only need to load half of the tile in each CTA for B.
        if (matrixType == MatrixType::MatrixB && options.mClusterDimX >= 2)
        {
            tileShape[1] /= 2;
        }
    }

    return std::make_tuple(shape, stride, tileShape);
}

// Create the TMA shape/stride for C.
template <class GemmOptions>
static auto makeTmaShapeStrideC(GemmOptions const& options)
{
    // The number of tokens.
    auto numTokens = options.mTransposeMmaOutput ? options.mN : options.mM;
    // The hidden dimension.
    auto hiddenSize = options.mTransposeMmaOutput ? options.mM : options.mN;
    // Note that TMA descriptor expects the first dimension's stride to be
    // 1, so swap the first two dimension so that the hiddenSize dimension comes first.
    auto shape = std::vector<uint64_t>{static_cast<uint64_t>(hiddenSize), static_cast<uint64_t>(numTokens)};

    // Assemble the stride (strideTokens, 1).
    // Swap the first two dimension as mentioned before.
    auto stride = std::vector<uint64_t>{1, static_cast<uint64_t>(hiddenSize)};

    return std::make_tuple(shape, stride);
}

// Create the TMA shape/stride for A/B block scaling factors.
template <class GemmOptions>
static auto makeTmaShapeStrideSfAb(
    GemmOptions const& options, MatrixType matrixType, tg::SfLayout layout, int32_t numEltsPerSf)
{
    // The outer dimension.
    auto numTokens = matrixType == MatrixType::MatrixA ? options.mM : options.mN;
    // The inner dimension.
    auto hiddenSize = options.mK;
    // The outer tile dimension.
    auto numTokensPerTile = matrixType == MatrixType::MatrixA ? options.mTileM : options.mTileN;
    // The inner tile dimension.
    auto hiddenSizePerTile = options.mMmaTileK;

    switch (layout)
    {
    case tg::SfLayout::R128c4:
    {
        // The scaling factor tensor packs 128x4 tiles into contiguous 512B blocks.
        // The 512B block maps to a 32x16B (32x128b) block in TMEM.
        //
        // Additionally, we have to meet constraints of TMA that the box dimensions are less
        // than 256 and boxDim[0] is a multiple of 16B.
        //
        // The "logical" tensor is:      [outer,        inner / numEltsPerSf]
        // The aforementioned format is: [⌈outer / 128⌉, inner / (4 * numEltsPerSf),    512]
        // The shape we use for TMA is:  [⌈outer / 128⌉, inner / (4 * numEltsPerSf), 2, 256]
        auto shape = std::vector<uint64_t>{256, 2, static_cast<uint64_t>(tg::ceilDiv(hiddenSize, numEltsPerSf * 4)),
            static_cast<uint64_t>(tg::ceilDiv(numTokens, 128))};

        std::vector<uint64_t> stride(shape.size());
        stride[0] = 1;
        for (size_t i = 1; i < shape.size(); i++)
        {
            stride[i] = shape[i - 1] * stride[i - 1];
        }

        auto tileShapes
            = std::vector<uint32_t>{256, 2, static_cast<uint32_t>(tg::ceilDiv(hiddenSizePerTile, numEltsPerSf * 4)),
                static_cast<uint32_t>(tg::ceilDiv(numTokensPerTile, 128))};

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
        int const r = options.mSfReshapeFactor;
        if (r > 0 && (r & (r - 1)) != 0)
        {
            throw std::runtime_error("mSfReshapeFactor must be positive and a power of 2. Found " + std::to_string(r));
        }

        // Sanitize number of repeats so it doesn't exceed the dimension.
        int const repeats = std::min(tg::ceilDiv(hiddenSizePerTile, numEltsPerSf * 4), r);

        // Detect if the input hidden size K is a multiple of the repeats.
        if (tg::ceilDiv(hiddenSize, numEltsPerSf * 4) % repeats != 0)
        {
            throw std::runtime_error("SF hiddenSize K (" + std::to_string(tg::ceilDiv(hiddenSize, numEltsPerSf * 4))
                + ") must be a multiple of repeats (" + std::to_string(repeats) + ")");
        }

        auto shape = std::vector<uint64_t>{static_cast<uint64_t>(repeats * 32),
            static_cast<uint64_t>(tg::ceilDiv(hiddenSize, numEltsPerSf * 4 * repeats)),
            static_cast<uint64_t>(tg::ceilDiv(numTokens, 8))};

        std::vector<uint64_t> stride(shape.size());
        stride[0] = 1;
        for (size_t i = 1; i < shape.size(); i++)
        {
            stride[i] = shape[i - 1] * stride[i - 1];
        }

        auto tileShapes = std::vector<uint32_t>{static_cast<uint32_t>(repeats * 32),
            static_cast<uint32_t>(tg::ceilDiv(hiddenSizePerTile, numEltsPerSf * 4 * repeats)),
            static_cast<uint32_t>(tg::ceilDiv(numTokensPerTile, 8))};

        return std::make_tuple(shape, stride, tileShapes);
    }

    default: throw std::runtime_error("Unsupported SF layout");
    }
    return std::make_tuple(std::vector<uint64_t>{}, std::vector<uint64_t>{}, std::vector<uint32_t>{});
}

// Create the TMA shape/stride for the sparsity information of A.
template <class GemmOptions>
static auto makeTmaShapeStrideSparsityInfoA(GemmOptions const& options)
{
    // Tensor dimensions.
    auto outerDim = options.mM;
    auto innerDim = tg::getNumBytesSparsityInfo(options.mSparsityA, options.mK);
    // Tile dimensions.
    auto tileOuterDim = options.mTileM;
    auto tileInnerDim = tg::getNumBytesSparsityInfo(options.mSparsityA, options.mTileK);

    auto shape = std::vector<uint64_t>{static_cast<uint64_t>(innerDim), static_cast<uint64_t>(outerDim)};

    std::vector<uint64_t> stride(shape.size());
    stride[0] = 1;
    for (size_t i = 1; i < shape.size(); i++)
    {
        stride[i] = shape[i - 1] * stride[i - 1];
    }

    auto tileShapes = std::vector<int32_t>{static_cast<int32_t>(tileInnerDim), static_cast<int32_t>(tileOuterDim)};

    return std::make_tuple(shape, stride, tileShapes);
}

// Setup the kernel parameters.
template <class GemmOptions_>
static KernelParams setKernelParams(GemmOptions_ const& options, void const* ptrA, void const* ptrSfA,
    void const* ptrPerTokenSfA, void const* ptrB, void const* ptrSfB, void const* ptrPerTokenSfB,
    void const* ptrSparsityInfoA, void const* ptrBias, void* ptrC, void* ptrSfC, void* multimemC, float* ptrScaleC,
    float* ptrScaleAct, void* ptrPartialSumsForSplitK, void* ptrTileBars, void* multimemTileBars,
    void* ptrCompletionBars, void* multimemCompletionBars, void* ptrSplitKCompletionBars, int32_t* ptrNumNonExitingCtas,
    int rank, int tpGrpSize)
{

    // Is one-shot all-reduce?
    bool const oneShotAr{options.mAllReduceAlgo == AllReduceAlgo::OneShot};
    // Is two-shot all-reduce?
    bool const twoShotAr{options.mAllReduceAlgo == AllReduceAlgo::TwoShot};
    // Are there peer devices?
    bool const multiDevice{tpGrpSize > 1};

    // Create the return struct.
    KernelParams params;

    // Is A using sparsity?
    int32_t const isSparseA = tg::isSparse(options.mSparsityA);
    // Do we pad A or B?
    bool doPadA = tg::dtypeNeedsPadding(options.mDtypeA, options.mMmaKind, options.mMmaK, isSparseA);
    bool doPadB = tg::dtypeNeedsPadding(options.mDtypeB, options.mMmaKind, options.mMmaK, isSparseA);

    void const* ptrForTmaA = ptrA;
    void const* ptrForTmaB = ptrB;

    // Shape/stride for gmem tensor A.
    std::vector<uint64_t> shapeA;
    std::vector<uint64_t> strideA;
    std::vector<int32_t> tileShapeA;
    std::tie(shapeA, strideA, tileShapeA) = makeTmaShapeStrideAb(options, MatrixType::MatrixA);
    params.tmaA = gemm::buildNdTmaDescriptor(options.mDtypeA, shapeA, strideA, tileShapeA,
        const_cast<void*>(ptrForTmaA), doPadA,
        /*doSwizzle=*/true);

    // Shape/stride for gmem tensor B.
    std::vector<uint64_t> shapeB;
    std::vector<uint64_t> strideB;
    std::vector<int32_t> tileShapeB;
    std::tie(shapeB, strideB, tileShapeB) = makeTmaShapeStrideAb(options, MatrixType::MatrixB);
    bool const doPadBFinal = doPadB;
    bool const doSwizzleB = true;
    params.tmaB = gemm::buildNdTmaDescriptor(
        options.mDtypeB, shapeB, strideB, tileShapeB, const_cast<void*>(ptrForTmaB), doPadBFinal, doSwizzleB);

    if (options.mDtypeA == tg::Dtype::E2m1 || options.mDtypeA == tg::Dtype::MxE2m1
        || options.mDtypeA == tg::Dtype::MxE4m3 || options.mDtypeA == tg::Dtype::MxInt4)
    {
        tg::Dtype dTypeSfA{};
        if (options.mDtypeA == tg::Dtype::E2m1)
        {
            dTypeSfA = tg::Dtype::E4m3;
        }
        else if (options.mDtypeA == tg::Dtype::MxInt4)
        {
            dTypeSfA = tg::Dtype::Bfloat16;
        }
        else
        {
            dTypeSfA = tg::Dtype::UE8m0;
        }

        int32_t const numEltsPerSfA = options.mSfBlockSizeA;

        // Build TMA descriptor for gmem A block scaling factors.
        auto [shapeSfA, strideSfA, tileShapesSfA]
            = makeTmaShapeStrideSfAb(options, MatrixType::MatrixA, options.mSfLayoutA, numEltsPerSfA);
        params.tmaSfA
            = gemm::buildSfTmaDescriptor(dTypeSfA, shapeSfA, strideSfA, tileShapesSfA, const_cast<void*>(ptrSfA));
    }

    if (options.mDtypeB == tg::Dtype::E2m1 || options.mDtypeB == tg::Dtype::MxE2m1
        || options.mDtypeB == tg::Dtype::MxE4m3)
    {
        tg::Dtype const dTypeSfB = (options.mDtypeB == tg::Dtype::E2m1) ? tg::Dtype::E4m3 : tg::Dtype::UE8m0;

        int32_t const numEltsPerSfB = options.mSfBlockSizeB;

        // Build TMA descriptor for gmem B block scaling factors.
        auto [shapeSfB, strideSfB, tileShapesSfB]
            = makeTmaShapeStrideSfAb(options, MatrixType::MatrixB, options.mSfLayoutB, numEltsPerSfB);
        params.tmaSfB
            = gemm::buildSfTmaDescriptor(dTypeSfB, shapeSfB, strideSfB, tileShapesSfB, const_cast<void*>(ptrSfB));
    }

    if (isSparseA)
    {
        // Build TMA descriptor for gmem A sparsity.
        auto [shapeSparsityInfoA, strideSparsityInfoA, tileShapesSparsityInfoA]
            = makeTmaShapeStrideSparsityInfoA(options);
        params.tmaSparsityInfoA = gemm::buildNdTmaDescriptor(tg::Dtype::UInt8, shapeSparsityInfoA, strideSparsityInfoA,
            tileShapesSparsityInfoA, const_cast<void*>(ptrSparsityInfoA),
            /*doPad=*/false,
            /*doSwizzle=*/true);
    }

    if (options.mUseTmaStore)
    {

        // Shape/stride for gmem tensor C.
        auto [shapeC, strideC] = makeTmaShapeStrideC(options);

        // Swap M and N tiles for the M-major epilogue.
        auto outputTileM = options.mTransposeMmaOutput ? options.mEpilogueTileN : options.mEpilogueTileM;
        auto outputTileN = options.mTransposeMmaOutput ? options.mEpilogueTileM : options.mEpilogueTileN;

        // One-shot performs TMA reduction on multicast mapping of the output buffer directly.
        // Two-shot performs TMA store on unicast mapping of the output buffer. The reduction happens
        // in the next phase.
        void* ptrTmaC{oneShotAr && multiDevice ? multimemC : ptrC};
        auto dtypeC{options.mDtypeC};
        // Regardless of output dtype, two-shot all-reduce store partial
        // accumulation results to global memory in float32 precision.
        if (twoShotAr && multiDevice)
        {
            dtypeC = options.mDtypeAcc;
        }

        // Build tma descriptor for C.
        params.tmaC = gemm::buildNdTmaDescriptor(dtypeC, shapeC, strideC,
            std::vector<int32_t>{outputTileN, outputTileM}, const_cast<void*>(ptrTmaC),
            /*doPad=*/false,
            /*doSwizzle=*/true);
    }

    params.ptrSfA = ptrSfA;
    params.ptrSfB = ptrSfB;

    // Set the per-token scale factors for MetaFP8 or scale inputs
    params.ptrPerTokenSfA = ptrPerTokenSfA;
    params.ptrPerTokenSfB = ptrPerTokenSfB;

    // Set the bias.
    params.ptrBias = ptrBias;

    // Also set ptrC (it may be used by the NCCL reduction code in "layers/Llama").
    params.ptrC = ptrC;

    // The scaling factors for the output tensor and the pre-activation scale.
    params.ptrScaleC = ptrScaleC;
    params.ptrScaleAct = ptrScaleAct;

    // The block scaling factors of C for MxFp{4,8} and NvFp4 formats.
    // (not to be confused with the tensor-level scaling factor stored in ptrScaleC)
    params.ptrSfC = ptrSfC;

    params.m = options.mM;
    params.n = options.mN;
    params.k = options.mK;

    params.rank = rank;
    params.tpGrpSize = tpGrpSize;

    params.multimemC = multimemC;
    params.ptrPartialSumsForSplitK = ptrPartialSumsForSplitK;
    params.ptrTileBars = ptrTileBars;
    params.multimemTileBars = multimemTileBars;
    params.ptrCompletionBars = ptrCompletionBars;
    params.multimemCompletionBars = multimemCompletionBars;

    params.ptrSplitKCompletionBars = ptrSplitKCompletionBars;
    params.ptrNumNonExitingCtas = ptrNumNonExitingCtas;
    return params;
}
#endif
}; // namespace KernelParamsSetup

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace gemm

} // namespace gemm
