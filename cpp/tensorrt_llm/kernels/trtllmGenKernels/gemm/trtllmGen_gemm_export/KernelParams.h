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

#include "Enums.h"
#include "TmaDescriptor.h"

namespace gemm
{

namespace gemm
{

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace tg = trtllm::gen;

////////////////////////////////////////////////////////////////////////////////////////////////////

struct KernelParams
{
#ifdef TLLM_ENABLE_CUDA
    //////////////////////////////////////////////////////////////////////////////////////////////////
    //
    // Gemm parameters.
    //
    //////////////////////////////////////////////////////////////////////////////////////////////////

    // TMA descriptor for A.
    // Must be setup using gemm::buildNdTmaDescriptor with shapes and strides from
    // makeTmaShapeStrideAb.
    //
    // If transposeMatrixA is false
    //   Logical shape is [M, K].
    //   Logical strides are [K, 1].
    //   Tile box shape is [tileM, tileK].
    //   Tile box strides are [tileK, 1].
    //   Dtype is set from options.mDtypeA.
    //
    // If transposeMatrixA is true
    //   Logical shape is [K, M].
    //   Logical strides are [M, 1].
    //   Tile box shape is [tileK, tileM].
    //   Tile box strides are [tileM, 1].
    //   Dtype is set from options.mDtypeA.
    CUtensorMap tmaA;

    // TMA descriptor for B.
    // Must be setup using gemm::buildNdTmaDescriptor with shapes and strides from
    // makeTmaShapeStrideAb.
    //
    // If transposeMatrixB is true
    //   Logical shape is [N, K].
    //   Logical strides are [K, 1].
    //   Tile box shape is [tileN, tileK].
    //   Tile box strides are [tileK, 1].
    //   Dtype is set from options.mDtypeB.
    //
    // If transposeMatrixB is false
    //   Logical shape is [K, N].
    //   Logical strides are [N, 1].
    //   Tile box shape is [tileK, tileN].
    //   Tile box strides are [tileN, 1].
    //   Dtype is set from options.mDtypeB.
    CUtensorMap tmaB;

    // TMA descriptor for C, (when useTmaStore is true)
    // Must be setup using gemm::buildNdTmaDescriptor with shapes and strides from
    // makeTmaShapeStrideC.
    //
    // If transposeMmaOutput is false,
    //    Logical shape is [M, N].
    //    Logical strides are [N, 1].
    //    Tile box shape is [epilogueTileM, epilogueTileN].
    //    Tile box strides are [epilogueTileN, 1].
    //    Dtype is set from options.mDtypeC.
    //
    // If transposeMmaOutput is true,
    //    Logical shape is [N, M].
    //    Logical strides are [M, 1].
    //    Tile box shape is [epilogueTileN, epilogueTileM].
    //    Tile box strides are [epilogueTileM, 1].
    //    Dtype is set from options.mDtypeC.
    CUtensorMap tmaC;

    // TMA descriptor for the block scaling factors for A, for MxFp{4,8} and NvFp4 formats.
    // Must be setup using gemm::buildSfTmaDescriptor with shapes and strides from
    // makeTmaShapeStrideSfAb.
    // The layout of scaling factors for A is always R128c4
    //
    // Let P be the number of elements per SF. P=16 for NvFp4, P=32 for Mx formats.
    // K must be a multiple of 4P.
    // The "logical" shape is: [M, K / P].
    // The R128c4 layout is: [⌈M / 128⌉, K / P / 4, 512].
    // The shape we use for TMA is: [⌈M / 128⌉, K / P / 4, 2, 256].
    //
    // Dtype is Dtype::E4m3 for NvFp4, Dtype::UE8m0 for Mx formats.
    CUtensorMap tmaSfA;

    // TMA descriptor for the block scaling factors for B, for MxFp{4,8} and NvFp4 formats.
    // Must be setup using gemm::buildSfTmaDescriptor with shapes and strides from
    // makeTmaShapeStrideSfAb.
    // The layout of scaling factors for B is controlled by options.mSfLayoutB.
    //
    // Let P be the number of elements per SF. P=16 for NvFp4, P=32 for Mx formats.
    // The "logical" shape is: [N, K / P]
    //
    // If the layout is R128c4,
    //    K must be a multiple of 4P.
    //    The R128c4 layout is: [⌈N / 128⌉, K / P / 4, 512]
    //    The shape we use for TMA is: [⌈N / 128⌉, K / P / 4, 2, 256]
    //
    // If the layout is R8c4,
    //    K must be a multiple of 4P.
    //    The R8c4 layout is: [⌈N / 8⌉, K / P / 4, 32]
    //    The shape we use for TMA is: [⌈N / 8⌉, K / P / 4 / r, r * 32]
    //    where r = min(tileK / P / 4, 8)
    //
    // Dtype is Dtype::E4m3 for NvFp4, Dtype::UE8m0 for Mx formats.
    CUtensorMap tmaSfB;

    // The output matrix C. The data type is controlled by options.mDtypeC.
    //
    // When transposeMmaOutput is true, the shape is [N, M].
    // Otherwise, the shape is [M, N].
    // Elements in a given row are stored contiguously in memory (row-major).
    void* ptrC;

    // The block scaling factors to dequantize A.
    //
    // If DeepSeek FP8 recipe is used:
    // If transposeMmaOutput is false, shape is [K / 128, M].
    // Otherwise, shape is [M / 128, K / 128].
    // The rightmost dimension is contiguous in memory.
    //
    // If DeepSeek FP8 recipe is not used, but for MxFp{4,8} and NvFp4 formats:
    // The layout and data type is the same as explained in tmaSfA.
    //
    // Otherwise should be set to nullptr.
    void const* ptrSfA;

    // The scaling factors to dequantize B.
    //
    // If DeepSeek FP8 recipe is used:
    //    If transposeMmaOutput is false, shape is [N / 128, K / 128].
    //    Otherwise, shape is [K / 128, N].
    //    The rightmost dimension is contiguous in memory.
    //
    // If DeepSeek FP8 recipe is not used, but for MxFp{4,8} and NvFp4 formats:
    //    The layout and data type is the same as explained in tmaSfB.
    //
    // Otherwise should be set to nullptr.
    void const* ptrSfB;

    // The per-token scaling factors from scale A.
    //
    // This is used for either:
    //   * Per-token scaling factor quantization schemes, such as MetaFP8. The dtype is Dtype::Float32
    //   * When the routing scales are applied to the input activations (only when output is not
    //   transposed). The dtype is Dtype::Bfloat16
    //
    // The shape is [M]
    void const* ptrPerTokenSfA;

    // The per-token scaling factors from scale B.
    //
    // This is used for either:
    //   * Per-token scaling factor quantization schemes, such as MetaFP8. The dtype is Dtype::Float32
    //   * When the routing scales are applied to the input activations (only when output is
    //   transposed). The dtype is Dtype::Bfloat16
    //
    // The shape is [N]
    void const* ptrPerTokenSfB;

    // The scaling factors calculated when quantizing C, for MxFp{4,8} and NvFp4 formats, also
    // used for the DeepSeek FP8 recipe.
    //
    // For DeepSeek FP8 recipe:
    //    If transposeMmaOutput is false, shape is [N / 128, M].
    //    Otherwise, shape is [M / 128, N].
    //    The rightmost dimension is contiguous in memory.
    //
    // For MxFp{4,8} and NvFp4 formats:
    //    If transposeMmaOutput is false, shape is [M, N / 16].
    //    Otherwise, shape is [N, M / 16].
    //    The layout is controlled by options.mSfLayoutC (either R128c4 or R8c4).
    void* ptrSfC;

    // The output tensor scaling factor for MxFp{4,8}, Fp8, NvFp4 and DeepSeek FP8 quantization.
    // TensorRT-LLM API requires a scaling factor on the device.
    // Shape is [1].
    float const* ptrScaleC;

    // The M dimension.
    // It is the total number of tokens if A is the activation matrix.
    // It is the total number of output channels if A is the weight matrix.
    int32_t m;
    // The N dimension.
    // It is the total number of tokens if B is the activation matrix.
    // It is the total number of output channels if B is the weight matrix.
    int32_t n;
    // The K dimension. It is the hidden dimension of the input matrices.
    int32_t k;

    //////////////////////////////////////////////////////////////////////////////////////////////////
    //
    // All-reduce parameters.
    //
    //////////////////////////////////////////////////////////////////////////////////////////////////

    // The rank id of the current device in the multi-gpu space.
    int rank;
    // The number of peer devices in tensor-parallel group.
    int tpGrpSize;
    // Pointer for output with multicast mapping. It is used by the "reduce" op (LDGMC.ADD) of the
    // two-shot reduce-scatter phase.
    // The shape is [M, N] and the dtype is float.
    void* multimemC;

    // The barriers in global memory.
    //
    // The kernel arrives at (with release ordering) the multicast mapping of the barrier to broadcast
    // amongst peer devices. It then waits (with acquire ordering) for the unicast mapping of the
    // barrier.
    //
    // Flags in global memory that sync on "entrance" of reduce-scatter phase in two-shot all-reduce.
    // The shape is [numTilesM * numTilesN] and the dtype is uint32_t.
    // The pointer to the unicast memory created with IpcNvlsHandle.
    // Must be set to 0 before the kernel launch.
    void* ptrTileBars;
    // The shape is [numTilesM * numTilesN] and the dtype is uint32_t.
    // The pointer to the multicast memory created with IpcNvlsHandle.
    void* multimemTileBars;

    // Flags in global memory that sync on "exit" after the all-reduce finishes.
    // The shape is [numTilesM * numTilesN] and the dtype is uint32_t.
    // The pointer to the unicast memory created with IpcNvlsHandle.
    // Must be set to 0 before the kernel launch.
    void* ptrCompletionBars;
    // The shape is [numTilesM * numTilesN] and the dtype is uint32_t.
    // The pointer to the multicast memory created with IpcNvlsHandle
    void* multimemCompletionBars;

    //////////////////////////////////////////////////////////////////////////////////////////////////
    //
    // Miscellaneous parameters.
    //
    //////////////////////////////////////////////////////////////////////////////////////////////////

    // The barriers in global memory for Split-k reduction with exchange in GMEM.
    // Each CTAs arrives at the barrier and blockIdx.z == gridDim.Z - 1 waits for the barrier to flip
    // to perform a reduction.
    // The shape is [numTilesM * numTilesN] and the dtype is uint32_t.
    // For DeepSeek FP8 recipe, the shape is [numTilesM * numTilesN * 2].
    // The memory must be set to 0 before the kernel launch.
    void* ptrSplitKCompletionBars;

    // Pointer to the memory holding the partial sums for split-K in GMEM.
    // The shape is [numSlicesForSplitK, numSlicesForSliceK, numTilesM * tileM, numTilesN * tileN].
    // The dtype is dtypeAcc, i.e. float.
    void* ptrPartialSumsForSplitK;

    // In some cases, some CTAs need to exit early. E.g. when the grid is statically set, but the
    // actual workload is decided at runtime. This device pointer maps to the number of non exiting
    // CTAs in the X dim of the grid when transposeMmaOutput is false. And the Y dim, otherwise.
    // The pointer points to a scalar and the dtype is int32_t. The pointed value must be >= 0.
    int32_t* ptrNumNonExitingCtas;

    //////////////////////////////////////////////////////////////////////////////////////////////////
    //
    // Miscellaneous parameters.
    //
    //////////////////////////////////////////////////////////////////////////////////////////////////

    enum class MatrixType
    {
        MatrixA = 0,
        MatrixB
    };

    // Create the TMA shape/stride for A/B.
    template <class GemmOptions>
    static auto makeTmaShapeStrideAb(GemmOptions const& options, MatrixType matrixType)
    {
        // The outer dimension.
        auto numTokens = (matrixType == MatrixType::MatrixA) ? options.mM : options.mN;
        // The inner dimension.
        auto hiddenSize = options.mK;
        // The cute tensor shape for A/B: (numTokens, hiddenSize).
        // Note that TMA descriptor expects the first dimension's stride to be
        // 1, so swap the first two dimension so that the hiddenSize dimension comes first.
        auto shape = std::vector<uint64_t>{static_cast<uint64_t>(hiddenSize), static_cast<uint64_t>(numTokens)};

        // Assemble the stride (strideTokens, 1).
        // Swap the first two dimension as mentioned before.
        auto stride = std::vector<uint64_t>{1, static_cast<uint64_t>(hiddenSize)};

        // Apply transpose if necessary
        if ((matrixType == MatrixType::MatrixA && options.mTransposeMatrixA)
            || (matrixType == MatrixType::MatrixB && !options.mTransposeMatrixB))
        {
            std::swap(shape[0], shape[1]);
            stride[1] = numTokens;
        }

        return std::make_tuple(shape, stride);
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
    static auto makeTmaShapeStrideSfAb(GemmOptions const& options, MatrixType matrixType, tg::SfLayout layout)
    {
        // The outer dimension.
        auto numTokens = matrixType == MatrixType::MatrixA ? options.mM : options.mN;
        // The inner dimension.
        auto hiddenSize = options.mK;
        // The outer tile dimension.
        auto numTokensPerTile = matrixType == MatrixType::MatrixA ? options.mTileM : options.mTileN;
        // The inner tile dimension.
        auto hiddenSizePerTile = options.mTileK;
        // The dtype of the matrix.
        tg::Dtype matrixDtype = matrixType == MatrixType::MatrixA ? options.mDtypeA : options.mDtypeB;
        // Number of elements per scaling factor.
        int32_t const numEltsPerSf = (matrixDtype == tg::Dtype::E2m1) ? 16 : 32;

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
            // As the inner dimension (k) is required to be a multiple of the tile size, we
            // can reshape to use fewer read requests, if the tile dimensions allow.
            // I.e., let's define r = min(⌈hiddenSizePerTile / (numEltsPerSf * 4)⌉, 8)
            //
            // The "logical" tensor is: [outer,        inner / numEltsPerSf]
            // The 8x4 SF layout is:    [⌈outer / 128⌉, inner / (4 * numEltsPerSf), 32]
            // The TMA tensor shape is: [⌈outer / 128⌉, inner / (4 * numEltsPerSf * r), r * 32]

            int const repeats = std::min(tg::ceilDiv(hiddenSizePerTile, numEltsPerSf * 4), 8);

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

    // Setup the kernel parameters.
    template <class GemmOptions_>
    static KernelParams setKernelParams(GemmOptions_ const& options, void const* ptrA, void const* ptrSfA,
        void const* ptrPerTokenSfA, void const* ptrB, void const* ptrSfB, void const* ptrPerTokenSfB, void* ptrC,
        void* ptrSfC, void* multimemC, float* ptrScaleC, void* ptrPartialSumsForSplitK, void* ptrTileBars,
        void* multimemTileBars, void* ptrCompletionBars, void* multimemCompletionBars, void* ptrSplitKCompletionBars,
        int32_t* ptrNumNonExitingCtas, int rank, int tpGrpSize)
    {

        // Is one-shot all-reduce?
        bool const oneShotAr{options.mAllReduceAlgo == AllReduceAlgo::OneShot};
        // Is two-shot all-reduce?
        bool const twoShotAr{options.mAllReduceAlgo == AllReduceAlgo::TwoShot};
        // Are there peer devices?
        bool const multiDevice{tpGrpSize > 1};

        // Create the return struct.
        KernelParams params;

        // Shape/stride for gmem tensor A.
        auto [shapeA, strideA] = makeTmaShapeStrideAb(options, MatrixType::MatrixA);
        // Build tma descriptor for A.
        params.tmaA = gemm::buildNdTmaDescriptor(options.mDtypeA, options.mMmaKind, shapeA, strideA,
            options.mTransposeMatrixA ? options.mTileK : options.mTileM,
            options.mTransposeMatrixA ? options.mTileM : options.mTileK, const_cast<void*>(ptrA));

        // Shape/stride for gmem tensor B.
        auto [shapeB, strideB] = makeTmaShapeStrideAb(options, MatrixType::MatrixB);
        // Build tma descriptor for B.
        params.tmaB = gemm::buildNdTmaDescriptor(options.mDtypeB, options.mMmaKind, shapeB, strideB,
            !options.mTransposeMatrixB ? options.mTileK : options.mTileN,
            !options.mTransposeMatrixB ? options.mTileN : options.mTileK, const_cast<void*>(ptrB),
            /* swizzle */ !options.mSliceK);

        if (options.mDtypeA == tg::Dtype::E2m1 || options.mDtypeA == tg::Dtype::MxE2m1
            || options.mDtypeA == tg::Dtype::MxE4m3)
        {
            tg::Dtype const dTypeSfA = (options.mDtypeA == tg::Dtype::E2m1) ? tg::Dtype::E4m3 : tg::Dtype::UE8m0;

            // Build TMA descriptor for gmem A block scaling factors.
            auto [shapeSfA, strideSfA, tileShapesSfA]
                = makeTmaShapeStrideSfAb(options, MatrixType::MatrixA, tg::SfLayout::R128c4);
            params.tmaSfA
                = gemm::buildSfTmaDescriptor(dTypeSfA, shapeSfA, strideSfA, tileShapesSfA, const_cast<void*>(ptrSfA));
        }

        if (options.mDtypeB == tg::Dtype::E2m1 || options.mDtypeB == tg::Dtype::MxE2m1
            || options.mDtypeB == tg::Dtype::MxE4m3)
        {
            tg::Dtype const dTypeSfB = (options.mDtypeB == tg::Dtype::E2m1) ? tg::Dtype::E4m3 : tg::Dtype::UE8m0;

            // Build TMA descriptor for gmem B block scaling factors.
            auto [shapeSfB, strideSfB, tileShapesSfB]
                = makeTmaShapeStrideSfAb(options, MatrixType::MatrixB, options.mSfLayoutB);
            params.tmaSfB
                = gemm::buildSfTmaDescriptor(dTypeSfB, shapeSfB, strideSfB, tileShapesSfB, const_cast<void*>(ptrSfB));
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
            params.tmaC = gemm::buildNdTmaDescriptor(
                dtypeC, tg::MmaKind::Auto, shapeC, strideC, outputTileM, outputTileN, const_cast<void*>(ptrTmaC));
        }

        // Set the dequantization factors for A and B when DeepSeek FP8 recipe is used.
        params.ptrSfA = ptrSfA;
        params.ptrSfB = ptrSfB;

        // Set the per-token scale factors for MetaFP8 or scale inputs
        params.ptrPerTokenSfA = ptrPerTokenSfA;
        params.ptrPerTokenSfB = ptrPerTokenSfB;

        // Also set ptrC (it may be used by the NCCL reduction code in "layers/Llama").
        params.ptrC = ptrC;
        params.ptrScaleC = ptrScaleC;

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

    // Setup the kernel parameters.
    template <class GemmOptions_>
    static KernelParams setKernelParams(GemmOptions_ const& options, void const* ptrA, void const* ptrB, void* ptrC,
        void* multimemC, float const* ptrScaleC, void* ptrTileBars, void* multimemTileBars, void* ptrCompletionBars,
        void* multimemCompletionBars, int rank, int tpGrpSize)
    {
        return setKernelParams(options, ptrA, nullptr, ptrB, nullptr, ptrC, multimemC, ptrScaleC, ptrTileBars,
            multimemTileBars, ptrCompletionBars, multimemCompletionBars, rank, tpGrpSize);
    }
#endif
};

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace gemm

} // namespace gemm
