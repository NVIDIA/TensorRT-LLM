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

#include "trtllm/gen/CommonUtils.h"
#include "trtllm/gen/DtypeDecl.h"
#include "trtllm/gen/SfLayoutDecl.h"
#include "trtllm/gen/SparsityDecl.h"

#include "Enums.h"
#include "TmaDescriptor.h"

namespace gemmGatedAct
{

namespace gemmGatedAct
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
    // If layoutA is MatrixLayout::MajorK
    //   Logical shape is [M, K].
    //   Logical strides are [K, 1].
    //   Tile box shape is [tileM, tileK].
    //   Tile box strides are [tileK, 1].
    //   Dtype is set from options.mDtypeA.
    //
    // If layoutA is MatrixLayout::MajorMn
    //   Logical shape is [K, M].
    //   Logical strides are [M, 1].
    //   Tile box shape is [tileK, tileM].
    //   Tile box strides are [tileM, 1].
    //   Dtype is set from options.mDtypeA.
    //
    // If layoutA is MatrixLayout::BlockMajorK
    //   Logical shape is [K / blockK, M, blockK].
    //   Logical strides are [M * blockK, blockK, 1].
    //   Tile box shape is [tileK / min(blockK, tileK), tileM, min(blockK, tileK)].
    //   Tile box strides are [tileM * min(blockK, tileK), min(blockK, tileK), 1].
    //   Dtype is set from options.mDtypeA, and blockK is 128B.
    CUtensorMap tmaA;

    // TMA descriptor for B.
    // Must be setup using gemm::buildNdTmaDescriptor with shapes and strides from
    // makeTmaShapeStrideAb.
    //
    // If layoutB is MatrixLayout::MajorK
    //   Logical shape is [N, K].
    //   Logical strides are [K, 1].
    //   Tile box shape is [tileN, tileK].
    //   Tile box strides are [tileK, 1].
    //   Dtype is set from options.mDtypeB.
    //
    // If layoutB is MatrixLayout::MajorMn
    //   Logical shape is [K, N].
    //   Logical strides are [N, 1].
    //   Tile box shape is [tileK, tileN].
    //   Tile box strides are [tileN, 1].
    //   Dtype is set from options.mDtypeB.
    //
    // If layoutB is MatrixLayout::BlockMajorK
    //   Logical shape is [K / blockK, N, blockK].
    //   Logical strides are [N * blockK, blockK, 1].
    //   Tile box shape is [tileK / min(blockK, tileK), tileN, min(blockK, tileK)].
    //   Tile box strides are [tileN * min(blockK, tileK), min(blockK, tileK), 1].
    //   Dtype is set from options.mDtypeB, and blockK is 128B.
    CUtensorMap tmaB;

    // TMA descriptor for C, (when useTmaStore is true)
    // Must be setup using gemm::buildNdTmaDescriptor with shapes and strides from
    // makeTmaShapeStrideC.
    //
    // If transposeMmaOutput is false,
    // Logical shape is [M, N / 2].
    // Logical strides are [N / 2, 1].
    // Tile box shape is [epilogueTileM, epilogueTileN / 2].
    // Tile box strides are [epilogueTileN / 2, 1].
    // Dtype is set from options.mDtypeC.
    //
    // If transposeMmaOutput is true,
    // Logical shape is [N, M / 2].
    // Logical strides are [M / 2, 1].
    // Tile box shape is [epilogueTileN, epilogueTileM / 2].
    // Tile box strides are [epilogueTileM / 2, 1].
    // Dtype is set from options.mDtypeC.
    CUtensorMap tmaC;

#ifdef TLLM_RUBIN_FEATURES
#ifdef TLLM_TEST
    // TMA descriptor for data invalidation, (when lamportProducer is true)
    // Must be setup using gemm::buildNdTmaDescriptor with shapes and strides from
    // makeTmaShapeStrideC.
    //
    // If transposeMmaOutput is false,
    // Logical shape is [M, N / 2].
    // Logical strides are [N / 2, 1].
    // Tile box shape is [epilogueTileM, epilogueTileN / 2].
    // Tile box strides are [epilogueTileN / 2, 1].
    // Dtype is set from options.mDtypeC.
    //
    // If transposeMmaOutput is true,
    // Logical shape is [N, M / 2].
    // Logical strides are [M / 2, 1].
    // Tile box shape is [epilogueTileN, epilogueTileM / 2].
    // Tile box strides are [epilogueTileM / 2, 1].
    // Dtype is set from options.mDtypeC.
    CUtensorMap tmaInvalidate;
#endif // TLLM_TEST
#endif // TLLM_RUBIN_FEATURES

    // TMA descriptor for the block scaling factors for A, for MxFp{4,8} and NvFp4 formats.
    // Must be setup using gemm::buildSfTmaDescriptor with shapes and strides from
    // makeTmaShapeStrideSfAb.
    // The layout of block scaling factors for A is always R128c4.
    // M must be a multiple of 128.
    // K must be a multiple of 64.
    // The "logical" shape is: [M, K / 16]
    // The R128c4 layout is: [M / 128, K / 16 / 4, 512].
    // The shape we use for TMA is: [M / 128, K / 16 / 4, 2, 256].
    // Dtype is Dtype::E4m3.
    CUtensorMap tmaSfA;

    // TMA descriptor for the block scaling factors for B, for MxFp{4,8} and NvFp4 formats.
    // Must be setup using gemm::buildSfTmaDescriptor with shapes and strides from
    // makeTmaShapeStrideSfAb.
    // The layout of block scaling factors for B is controlled by options.mSfLayoutB.
    //
    // The "logical" shape is: [N, K / 16]
    //
    // If the layout is R128c4,
    // N must be a multiple of 128.
    // K must be a multiple of 64.
    // The R128c4 layout is: [N / 128, K / 16 / 4, 512]
    // The shape we use for TMA is: [N / 128, K / 16 / 4, 2, 256]
    //
    // If the layout is R8c4,
    // N must be a multiple of 8.
    // K must be a multiple of 64.
    // The R8c4 layout is: [N / 8, K / 16 / 4, 32]
    // The shape we use for TMA is: [N / 8, K / 16 / 4 / repeats, repeats * 32]
    // where repeats = min(tileK / 16 / 4, 8)
    //
    // Dtype is Dtype::E4m3.
    CUtensorMap tmaSfB;

    // TMA descriptor for the sparsity information of A, if structured sparsity is used.
    // Must be setup using gemm::buildNdTmaDescriptor with shapes and strides from
    // makeTmaShapeStrideSparsityInfoA.
    //
    // When sparsityA is Any_2_4:
    //     2 elements are non-zero in any chunk of 4 elements.
    //     A 4-bit index indicates the position of the non-zero elements.
    //     The shape in UInt8 is: [M, K / 8] (two 4-bit indices packed into one UInt8)
    //
    // When sparsityA is Pairwise_4_8:
    //     4 elements are non-zero in any chunk of 8 elements.
    //     The zero and non-zero elements are grouped in pairs.
    //     A 4-bit index indicates the position of the non-zero pairs.
    //     The shape in UInt8 is: [M, K / 16] (two 4-bit indices packed into one UInt8)
    //
    // Dtype is Dtype::UInt8.
    CUtensorMap tmaSparsityInfoA;

    // The output matrix C. The data type is controlled by options.mDtypeC.
    //
    // When transposeMmaOutput is true, the shape is [N, M / 2].
    // Otherwise, the shape is [M, N / 2].
    // Elements in a given row are stored contiguously in memory (row-major).
    void* ptrC{nullptr};

    // The scaling factors to dequantize A.
    // It is used when the DeepSeek FP8 recipe is enabled. Otherwise should be set to nullptr.
    // If transposeMmaOutput is false, shape is [K / 128, M].
    // Otherwise, shape is [M / 128, K / 128].
    // The rightmost dimension is contiguous in memory.
    void const* ptrSfA{nullptr};

    // The scaling factors to dequantize B.
    // It is used when the DeepSeek FP8 recipe is enabled. Otherwise should be set to nullptr.
    // If transposeMmaOutput is false, shape is [N / 128, K / 128].
    // Otherwise, shape is [K / 128, N].
    // The rightmost dimension is contiguous in memory.
    void const* ptrSfB{nullptr};

    // The per-token scaling factors from scale A.
    //
    // This is used for either:
    //   * Per-token scaling factor quantization schemes, such as MetaFP8. The dtype is Dtype::Float32
    //   * When the routing scales are applied to the input activations (only when output is not
    //   transposed). The dtype is Dtype::Bfloat16
    //
    // The shape is [M]
    void const* ptrPerTokenSfA{nullptr};

    // The per-token scaling factors from scale B.
    //
    // This is used for either:
    //   * Per-token scaling factor quantization schemes, such as MetaFP8. The dtype is Dtype::Float32
    //   * When the routing scales are applied to the input activations (only when output is
    //   transposed). The dtype is Dtype::Bfloat16
    //
    // The shape is [N]
    void const* ptrPerTokenSfB{nullptr};

    // The bias applied after the GEMM and before the activation function.
    // The bias is applied before applying the global scaling factor. I.e.
    // C = act(A * B + bias') * scaleC
    // scaleC = dequantA * dequantB * quantC
    // Thus, the bias' = bias / (dequantA * dequantB), where the bias is the original bias.
    //
    // if BiasType is N, the shape is [N]
    // The bias is broadcasted along the M dimension.
    //
    // if BiasType is M, the shape is [M]
    // The bias is broadcasted along the N dimension.
    //
    // The dtype is float32.
    void const* ptrBias{nullptr};

    // The scaling factors calculated when quantizing C, for MxFp{4,8} and NvFp4 formats, also
    // used for the DeepSeek FP8 recipe.
    //
    // For DeepSeek FP8 recipe:
    // If transposeMmaOutput is false, shape is [N / 2 / 128, M].
    // Otherwise, shape is [M / 2 / 128, N].
    // The rightmost dimension is contiguous in memory.
    //
    // For MxFp{4,8} and NvFp4 formats:
    // If transposeMmaOutput is false, shape is [M, N / 2 / 16].
    // Otherwise, shape is [N, M / 2 / 16].
    // The layout is controlled by options.mSfLayoutC (either R128c4 or R8c4).
    void* ptrSfC{nullptr};

    // Output is equal to
    // y = act(ptrScaleGate[0] * y1) * (ptrScaleC[0] * y2)
    //
    // The output tensor scaling factor for MxFp{4,8}, NvFp4 and DeepSeek FP8 quantization.
    // TensorRT-LLM API requires a scaling factor on the device.
    // Shape is [1].
    float const* ptrScaleC{nullptr};
    // The output gate scale for MxFp{4,8}, NvFp4 and DeepSeek FP8 quantization.
    // Shape is [1].
    float const* ptrScaleGate{nullptr};

    // The clamp limit before the activation.
    // Shape is [1].
    // Clamp is INF if nullptr.
    // If applied on SwiGlu, it will be:
    //
    //   x_glu    = x_glu.clamp(min=None, max=limit)
    //   x_linear = x_linear.clamp(min=-limit, max=limit)
    float const* ptrClampLimit{nullptr};

    // The alpha and beta for SwiGlu or GeGlu.
    // Shape is [1]. One alpha and one beta per tensor in batch.
    // Alpha is 1.f if nullptr.
    // Beta is 0.f if nullptr.
    // The formula:
    //
    //   out_glu  = x_glu * torch.sigmoid(alpha * x_glu) * (x_linear + beta)
    float const* ptrGatedActAlpha{nullptr};
    float const* ptrGatedActBeta{nullptr};

    // The M dimension.
    // It is the total number of tokens if A is the activation matrix.
    // It is the total number of output channels multiplied by 2 if A is the weight matrix.
    int32_t m;
    // The N dimension.
    // It is the total number of tokens if B is the activation matrix.
    // It is the total number of output channels multiplied by 2 if B is the weight matrix.
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

    //////////////////////////////////////////////////////////////////////////////////////////////////
    //
    // Miscellaneous parameters.
    //
    //////////////////////////////////////////////////////////////////////////////////////////////////

    // Pointer for partial row max for DeepSeek FP8 recipe.
    // This is temporary storage for the row max results.
    // The shape is [2, M, N / 128] and the dtype is float.
    float* ptrPartialRowMax{nullptr};

    // Flags in global memory that sync on "exit" for row max computation.
    // The size is numTilesM * numTilesN / 2 and the dtype is uint32_t.
    // The memory must be set to 0 before the kernel launch.
    uint32_t* ptrRowMaxCompletionBars{nullptr};

    enum class MatrixType
    {
        MatrixA = 0,
        MatrixB,
        MatrixC
    };

    // Create the TMA shape/stride for A/B/C.
    template <class GemmOptions>
    static auto makeTmaShapeStrideAbc(GemmOptions const& options, MatrixType matrixType)
    {
        // For sparse A, the k dimension in the TMA shapes is halved.
        int const isSparse = matrixType == MatrixType::MatrixA && tg::isSparse(options.mSparsityA);

        // The padded/valid dimensions.
        int sizeM = options.mM;
        int sizeN = options.mN;
        int sizeK = options.mK >> isSparse;
        int const tileM = options.mTileM;
        int const tileN = options.mTileN;
        int const tileK = options.mTileK >> isSparse;
        int validM = options.mValidM;
        int validN = options.mValidN;
        int validK = options.mValidK >> isSparse;

        // The outer dimension.
        auto numTokens = (matrixType == MatrixType::MatrixA || matrixType == MatrixType::MatrixC) ? sizeM : sizeN;
        auto numTokensValid
            = (matrixType == MatrixType::MatrixA || matrixType == MatrixType::MatrixC) ? validM : validN;
        // The outer dimension tile size.
        auto tileNumTokens = (matrixType == MatrixType::MatrixC) ? options.mEpilogueTileM
            : (matrixType == MatrixType::MatrixA)                ? tileM
                                                                 : tileN;
        // The inner dimension.
        auto hiddenSize = (matrixType == MatrixType::MatrixC) ? sizeN / 2 : sizeK;
        auto hiddenSizeValid = (matrixType == MatrixType::MatrixC) ? validN / 2 : validK;
        // The inner dimension tile size.
        auto tileHiddenSize = (matrixType == MatrixType::MatrixC) ? options.mEpilogueTileN / 2 : tileK;
        // Swap variables if transpose output
        if (matrixType == MatrixType::MatrixC && options.mTransposeMmaOutput)
        {
            numTokens = sizeN;
            numTokensValid = validN;
            hiddenSize = sizeM / 2;
            hiddenSizeValid = validM / 2;
            tileNumTokens = options.mEpilogueTileN;
            tileHiddenSize = options.mEpilogueTileM / 2;
        }
        // The cute tensor shape for A/B: (numTokens, hiddenSize).
        // Note that TMA descriptor expects the first dimension's stride to be
        // 1, so swap the first two dimension so that the hiddenSize dimension comes first.
        auto shape
            = std::vector<uint64_t>{static_cast<uint64_t>(hiddenSizeValid), static_cast<uint64_t>(numTokensValid)};

        // Assemble the stride (strideTokens, 1).
        // Swap the first two dimension as mentioned before.
        auto stride = std::vector<uint64_t>{1, static_cast<uint64_t>(hiddenSize)};

        // Assemble the box shape
        std::vector<int32_t> tileShape = {tileHiddenSize, tileNumTokens};

        // Alternate layouts do not apply to matrixC
        if (matrixType != MatrixType::MatrixC)
        {
            gemm::MatrixLayout layout = (matrixType == MatrixType::MatrixA) ? options.mLayoutA : options.mLayoutB;
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
                    static_cast<uint64_t>(sizeK / options.mBlockK)};
                stride
                    = {1, static_cast<uint64_t>(options.mBlockK), static_cast<uint64_t>(numTokens * options.mBlockK)};

                // If blockK > tileK, then the inner most box size will be based on the tile
                int32_t const tileBlockK = std::min(options.mBlockK, tileHiddenSize);
                tileShape = {tileBlockK, tileNumTokens, tileHiddenSize / tileBlockK};
            }
        }

        return std::make_tuple(shape, stride, tileShape);
    }

    // Create the TMA shape/stride for A/B block scaling factors.
    template <class GemmOptions>
    static auto makeTmaShapeStrideSfAb(GemmOptions const& options, MatrixType matrixType, tg::SfLayout layout,
        int sfReshapeFactor, int32_t numEltsPerSf)
    {
        // The outer dimension.
        auto numTokens = matrixType == MatrixType::MatrixA ? options.mM : options.mN;
        // The inner dimension.
        auto hiddenSize = options.mK;
        // The outer tile dimension.
        auto numTokensPerTile = matrixType == MatrixType::MatrixA ? options.mTileM : options.mTileN;
        // The inner tile dimension.
        auto hiddenSizePerTile = options.mTileK;

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

#ifdef TLLM_RUBIN_FEATURES
        case tg::SfLayout::R128c16:
        {
            // The scaling factor tensor packs 128x16 tiles into contiguous 2048B blocks.
            // The 2048B block maps to a 128x16B (128x128b) block in TMEM.
            //
            // Additionally, we have to meet constraints of TMA that the box dimensions are less
            // than 256 and boxDim[0] is a multiple of 16B.
            //
            // The "logical" tensor is:      [outer,        inner / numEltsPerSf]
            // The aforementioned format is: [⌈outer / 128⌉, inner / 16 * numEltsPerSf),    2048]
            // The shape we use for TMA is:  [⌈outer / 128⌉, inner / 16 * numEltsPerSf), 8,  256]
            auto shape
                = std::vector<uint64_t>{256, 8, static_cast<uint64_t>(tg::ceilDiv(hiddenSize, numEltsPerSf * 16)),
                    static_cast<uint64_t>(tg::ceilDiv(numTokens, 128))};

            std::vector<uint64_t> stride(shape.size());
            stride[0] = 1;
            for (size_t i = 1; i < shape.size(); i++)
            {
                stride[i] = shape[i - 1] * stride[i - 1];
            }

            auto tileShapes = std::vector<uint32_t>{256, 8,
                static_cast<uint32_t>(tg::ceilDiv(hiddenSizePerTile, numEltsPerSf * 16)),
                static_cast<uint32_t>(tg::ceilDiv(numTokensPerTile, 128))};

            return std::make_tuple(shape, stride, tileShapes);
        }
#endif // TLLM_RUBIN_FEATURES

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
                throw std::runtime_error(
                    "mSfReshapeFactor must be positive and a power of 2. Found " + std::to_string(r));
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
        [[maybe_unused]] void const* ptrSparsityInfoA, void const* ptrBias, void* ptrC, float const* ptrScaleC,
        void* ptrSfC,
#ifdef TLLM_RUBIN_FEATURES
#ifdef TLLM_TEST
        void* ptrInvalidate,
#endif // TLLM_TEST
#endif // TLLM_RUBIN_FEATURES
        float const* ptrScaleGate, float const* ptrClampLimit, float const* ptrGatedActAlpha,
        float const* ptrGatedActBeta, float* rowMax, uint32_t* rowMaxBars)
    {

        // Create the return struct.
        KernelParams params;

        // Is A using sparsity?
        int32_t const isSparseA = tg::isSparse(options.mSparsityA);
        // Do we pad A or B?
        bool doPadA = tg::dtypeNeedsPadding(options.mDtypeA, options.mMmaKind, options.mMmaK, isSparseA);
        bool doPadB = tg::dtypeNeedsPadding(options.mDtypeB, options.mMmaKind, options.mMmaK, isSparseA);

        // Shape/stride for gmem tensor A.
        auto [shapeA, strideA, tileShapeA] = makeTmaShapeStrideAbc(options, MatrixType::MatrixA);
        // Build tma descriptor for A.
        params.tmaA
            = gemm::buildNdTmaDescriptor(options.mDtypeA, shapeA, strideA, tileShapeA, const_cast<void*>(ptrA), doPadA,
                /*doSwizzle=*/true);

        // Shape/stride for gmem tensor B.
        auto [shapeB, strideB, tileShapeB] = makeTmaShapeStrideAbc(options, MatrixType::MatrixB);
        // Build tma descriptor for B.
        params.tmaB
            = gemm::buildNdTmaDescriptor(options.mDtypeB, shapeB, strideB, tileShapeB, const_cast<void*>(ptrB), doPadB,
                /*doSwizzle=*/true);

        if (options.mDtypeA == tg::Dtype::E2m1 || options.mDtypeA == tg::Dtype::MxE4m3)
        {
            tg::Dtype const dTypeSf = tg::dtypeGetBlockSfType(options.mDtypeA);

            int32_t const numEltsPerSfA = options.mSfBlockSizeA;

            // Build TMA descriptor for gmem A block scaling factors.
            auto [shapeSfA, strideSfA, tileShapesSfA] = makeTmaShapeStrideSfAb(
                options, MatrixType::MatrixA, options.mSfLayoutA, options.mSfReshapeFactor, numEltsPerSfA);
            params.tmaSfA
                = gemm::buildSfTmaDescriptor(dTypeSf, shapeSfA, strideSfA, tileShapesSfA, const_cast<void*>(ptrSfA));
        }
        if (options.mDtypeB == tg::Dtype::E2m1 || options.mDtypeB == tg::Dtype::MxE4m3)
        {
            tg::Dtype const dTypeSf = tg::dtypeGetBlockSfType(options.mDtypeB);

            int32_t const numEltsPerSfB = options.mSfBlockSizeB;

            // Build TMA descriptor for gmem B block scaling factors.
            auto [shapeSfB, strideSfB, tileShapesSfB] = makeTmaShapeStrideSfAb(
                options, MatrixType::MatrixB, options.mSfLayoutB, options.mSfReshapeFactor, numEltsPerSfB);
            params.tmaSfB
                = gemm::buildSfTmaDescriptor(dTypeSf, shapeSfB, strideSfB, tileShapesSfB, const_cast<void*>(ptrSfB));
        }

        if (isSparseA)
        {
            // Build TMA descriptor for gmem A sparsity.
            auto [shapeSparsityInfoA, strideSparsityInfoA, tileShapesSparsityInfoA]
                = makeTmaShapeStrideSparsityInfoA(options);
            params.tmaSparsityInfoA = gemm::buildNdTmaDescriptor(tg::Dtype::UInt8, shapeSparsityInfoA,
                strideSparsityInfoA, tileShapesSparsityInfoA, const_cast<void*>(ptrSparsityInfoA),
                /*doPad=*/false,
                /*doSwizzle=*/true);
        }

        if (options.mUseTmaStore)
        {
            // Shape/stride for gmem tensor C.
            auto [shapeC, strideC, tileShapeC] = makeTmaShapeStrideAbc(options, MatrixType::MatrixC);
            // Build tma descriptor for C.
            params.tmaC
                = gemm::buildNdTmaDescriptor(options.mDtypeC, shapeC, strideC, tileShapeC, const_cast<void*>(ptrC),
                    /*doPad=*/false);
#ifdef TLLM_RUBIN_FEATURES
#ifdef TLLM_TEST
            if (options.mLamportProducer)
            {
                // Build tma descriptor for invalidate.
                params.tmaInvalidate = gemm::buildNdTmaDescriptor(options.mDtypeC, shapeC, strideC, tileShapeC,
                    const_cast<void*>(ptrInvalidate),
                    /*doPad=*/false);
            }
#endif // TLLM_TEST
#endif // TLLM_RUBIN_FEATURES
        }

        params.ptrC = ptrC;
        // Set the dequantization factors for A and B when DeepSeek FP8 recipe is used.
        params.ptrSfA = ptrSfA;
        params.ptrSfB = ptrSfB;
        params.ptrSfC = ptrSfC;

        // Set the per-token scale factors for MetaFP8 or scale inputs
        params.ptrPerTokenSfA = ptrPerTokenSfA;
        params.ptrPerTokenSfB = ptrPerTokenSfB;

        params.ptrBias = ptrBias;

        params.ptrScaleC = ptrScaleC;
        params.ptrScaleGate = ptrScaleGate;
        params.ptrClampLimit = ptrClampLimit;
        params.ptrGatedActAlpha = ptrGatedActAlpha;
        params.ptrGatedActBeta = ptrGatedActBeta;
        params.rank = 0;
        params.tpGrpSize = 1;

        params.ptrPartialRowMax = rowMax;
        params.ptrRowMaxCompletionBars = rowMaxBars;

        params.m = options.mM;
        params.n = options.mN;
        params.k = options.mK;

        return params;
    }
#endif
};

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace gemmGatedAct

} // namespace gemmGatedAct
