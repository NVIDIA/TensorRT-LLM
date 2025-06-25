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
#include "trtllm/gen/DtypeDecl.h"
#include "trtllm/gen/SfLayoutDecl.h"

#include "Enums.h"
#include "TmaDescriptor.h"

namespace gemmGatedAct
{

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace tg = trtllm::gen;

////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef TLLM_ENABLE_CUDA
CUtensorMap buildNdTmaDescriptor(tg::Dtype dtype, std::vector<uint64_t> const& shapes,
    std::vector<uint64_t> const& strides, int32_t tileSizeMn, int32_t tileSizeK, void* gmemAddr)
{
    CUtensorMap desc{};
    // The data type.
    CUtensorMapDataType tmaDataFormat{CU_TENSOR_MAP_DATA_TYPE_FLOAT32};
    if (dtype == tg::Dtype::E4m3)
    {
        tmaDataFormat = CU_TENSOR_MAP_DATA_TYPE_UINT8;
    }
    else if (dtype == tg::Dtype::Fp16)
    {
        tmaDataFormat = CU_TENSOR_MAP_DATA_TYPE_FLOAT16;
    }
    else if (dtype == tg::Dtype::Bfloat16)
    {
        tmaDataFormat = CU_TENSOR_MAP_DATA_TYPE_BFLOAT16;
    }
    else if (dtype == tg::Dtype::E2m1)
    {
        tmaDataFormat = CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B;
    }
    else if (dtype == tg::Dtype::Fp32)
    {
        tmaDataFormat = CU_TENSOR_MAP_DATA_TYPE_FLOAT32;
    }
    else
    {
        std::cerr << "Unexpected dtype " << static_cast<int32_t>(dtype) << std::endl;
        assert(false);
    }

    // The swizzle type.
    CUtensorMapSwizzle swizzleType{CU_TENSOR_MAP_SWIZZLE_NONE};
    int32_t tileKSizeInBytes = (tileSizeK * tg::dtypeGetNumBits(dtype)) / /* bits */ 8;
    if ((tileKSizeInBytes % 128) == 0)
    {
        swizzleType = CU_TENSOR_MAP_SWIZZLE_128B;
    }
    else if ((tileKSizeInBytes % 64) == 0)
    {
        swizzleType = CU_TENSOR_MAP_SWIZZLE_64B;
    }
    else if ((tileKSizeInBytes % 32) == 0)
    {
        swizzleType = CU_TENSOR_MAP_SWIZZLE_32B;
    }
    else
    {
        std::cerr << "Unexpected tileKSizeInBytes " << tileKSizeInBytes << std::endl;
        assert(false);
    }

    // Check gmem address must be 16B-aligned
    assert((reinterpret_cast<uint64_t>(gmemAddr) & 0b1111) == 0); //

    // Check shape must be in range [1, 2^32]
    int32_t dim = shapes.size();
    // Expect 2 dimensions.
    assert(dim == 2);
    // Check shape range.
    for (int32_t ii = 0; ii < dim; ++ii)
    {
        assert(shapes[ii] >= (uint64_t(1)));       // Size must be min 1
        assert(shapes[ii] <= (uint64_t(1) << 32)); // Size must be max 2^32
    }

    // TMA descriptor does not store the zeroth stride and assumes it is 1.
    assert(static_cast<int32_t>(strides.size()) == dim);
    assert(strides[0] == 1);

    // Build strides in bytes.
    // cuTensorMapEncodeTiled ignores the stride of the first dimension (implicitly 1).
    std::vector<uint64_t> stridesInBytes(dim - 1);
    for (int32_t ii = 0; ii < dim - 1; ++ii)
    {
        stridesInBytes[ii] = (strides[ii + 1] * tg::dtypeGetNumBits(dtype)) / /* bits */ 8;
    }

    // Set the number of elements in the packed uint32_t element.
    auto const numEltsPerUInt32 = 4 * /* bits */ 8 / tg::dtypeGetNumBits(dtype);
    // The number of elements in 128B.
    auto const numEltsIn128B = numEltsPerUInt32 /*4B*/ * 32;
    // The number of tile K hidden size (per token) in each block of shared memory.
    auto const numEltsInClampedTileKSize = std::min(numEltsIn128B, tileSizeK);

    // Build tile shapes.
    std::vector<uint32_t> tileShapes(dim, 1);
    tileShapes[0] = numEltsInClampedTileKSize; // tileSizeK
    tileShapes[1] = tileSizeMn;                // tileSizeMn

    // Set tile strides to 0;
    std::vector<uint32_t> tileStrides(dim, 1);

    // Build the descriptor.
    CUresult result = cuTensorMapEncodeTiled(&desc, tmaDataFormat,
        /*tensorRank=*/dim, gmemAddr, shapes.data(), stridesInBytes.data(), tileShapes.data(), tileStrides.data(),
        /*interleave=*/CU_TENSOR_MAP_INTERLEAVE_NONE, swizzleType,
        /*l2Promotion=*/CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        /*oobFill=*/CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

    if (result != CUDA_SUCCESS)
    {
        std::cerr << "Error: Failed to initialize the TMA descriptor " << result << std::endl;

        std::cerr << "tmaFormat: " << static_cast<int>(tmaDataFormat) << " dim: " << dim << " gmem: " << gmemAddr
                  << std::endl;
        std::cerr << "Shape: " << shapes[0] << " " << shapes[1] << std::endl;
        std::cerr << "Stride: " << stridesInBytes[0] << std::endl;
        std::cerr << "tileShapes: " << tileShapes[0] << " " << tileShapes[1] << std::endl;
        std::cerr << "tileStrides: " << tileStrides[0] << " " << tileStrides[1] << std::endl;
        std::cerr << "swizzleType: " << int(swizzleType) << std::endl;
        assert(false);
    }

    return desc;
}
#endif // defined TLLM_ENABLE_CUDA

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
    // makeTmaShapeStrideAb. Logical shape is [M, K]. Logical strides are [K, 1]. Tile box shape is
    // [tileM, tileK]. Tile box strides are [tileK, 1].
    // Dtype is set from options.mDtypeElt.
    CUtensorMap tmaA;

    // TMA descriptor for B.
    // Must be setup using gemm::buildNdTmaDescriptor with shapes and strides from
    // makeTmaShapeStrideAb. Logical shape is [N, K]. Logical strides are [K, 1]. Tile box shape is
    // [tileN, tileK]. Tile box strides are [tileK, 1].
    // Dtype is set from options.mDtypeElt.
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

    // The output matrix C. The data type is controlled by options.mDtypeC.
    //
    // When transposeMmaOutput is true, the shape is [N, M / 2].
    // Otherwise, the shape is [M, N / 2].
    // Elements in a given row are stored contiguously in memory (row-major).
    void* ptrC;

    // The scaling factors to dequantize A.
    // It is used when the DeepSeek FP8 recipe is enabled. Otherwise should be set to nullptr.
    // If transposeMmaOutput is false, shape is [K / 128, M].
    // Otherwise, shape is [M / 128, K / 128].
    // The rightmost dimension is contiguous in memory.
    void const* ptrSfA;

    // The scaling factors to dequantize B.
    // It is used when the DeepSeek FP8 recipe is enabled. Otherwise should be set to nullptr.
    // If transposeMmaOutput is false, shape is [N / 128, K / 128].
    // Otherwise, shape is [K / 128, N].
    // The rightmost dimension is contiguous in memory.
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
    // If transposeMmaOutput is false, shape is [N / 2 / 128, M].
    // Otherwise, shape is [M / 2 / 128, N].
    // The rightmost dimension is contiguous in memory.
    //
    // For MxFp{4,8} and NvFp4 formats:
    // If transposeMmaOutput is false, shape is [M, N / 2 / 16].
    // Otherwise, shape is [N, M / 2 / 16].
    // The layout is controlled by options.mSfLayoutC (either R128c4 or R8c4).
    void* ptrSfC;

    // Output is equal to
    // y = act(ptrScaleGate[0] * y1) * (ptrScaleC[0] * y2)
    //
    // The output tensor scaling factor for MxFp{4,8}, NvFp4 and DeepSeek FP8 quantization.
    // TensorRT-LLM API requires a scaling factor on the device.
    // Shape is [1].
    float const* ptrScaleC;
    // The output gate scale for MxFp{4,8}, NvFp4 and DeepSeek FP8 quantization.
    // Shape is [1].
    float const* ptrScaleGate;

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
    float* ptrPartialRowMax;

    // Flags in global memory that sync on "exit" for row max computation.
    // The size is numTilesM * numTilesN / 2 and the dtype is uint32_t.
    // The memory must be set to 0 before the kernel launch.
    uint32_t* ptrRowMaxCompletionBars;

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
        auto numTokens
            = (matrixType == MatrixType::MatrixA || matrixType == MatrixType::MatrixC) ? options.mM : options.mN;
        auto hiddenSize = (matrixType == MatrixType::MatrixC) ? options.mN / 2 : options.mK;
        if (matrixType == MatrixType::MatrixC && options.mTransposeMmaOutput)
        {
            numTokens = options.mN;
            hiddenSize = options.mM / 2;
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
        // Number of elements per scaling factor.
        int32_t const numEltsPerSf = (options.mDtypeElt == tg::Dtype::E2m1) ? 16 : 32;

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

        default: assert(false);
        }
        return std::make_tuple(std::vector<uint64_t>{}, std::vector<uint64_t>{}, std::vector<uint32_t>{});
    }

    // Setup the kernel parameters.
    template <class GemmOptions_>
    static KernelParams setKernelParams(GemmOptions_ const& options, void const* ptrA, void const* ptrSfA,
        void const* ptrPerTokenSfA, void const* ptrB, void const* ptrSfB, void const* ptrPerTokenSfB, void* ptrC,
        float const* ptrScaleC, void* ptrSfC, float const* ptrScaleGate, float* rowMax, uint32_t* rowMaxBars)
    {

        // Create the return struct.
        KernelParams params;

        // Shape/stride for gmem tensor A.
        auto [shapeA, strideA] = makeTmaShapeStrideAbc(options, MatrixType::MatrixA);
        // Build tma descriptor for A.
        params.tmaA = gemmGatedAct::buildNdTmaDescriptor(
            options.mDtypeElt, shapeA, strideA, options.mTileM, options.mTileK, const_cast<void*>(ptrA));

        // Shape/stride for gmem tensor B.
        auto [shapeB, strideB] = makeTmaShapeStrideAbc(options, MatrixType::MatrixB);
        // Build tma descriptor for B.
        params.tmaB = gemmGatedAct::buildNdTmaDescriptor(
            options.mDtypeElt, shapeB, strideB, options.mTileN, options.mTileK, const_cast<void*>(ptrB));

        if (options.mDtypeElt == tg::Dtype::E2m1 || options.mDtypeElt == tg::Dtype::MxE4m3)
        {
            tg::Dtype const dTypeSf = tg::dtypeGetBlockSfType(options.mDtypeElt);

            // Build TMA descriptor for gmem A block scaling factors.
            auto [shapeSfA, strideSfA, tileShapesSfA]
                = makeTmaShapeStrideSfAb(options, MatrixType::MatrixA, tg::SfLayout::R128c4);
            params.tmaSfA
                = gemm::buildSfTmaDescriptor(dTypeSf, shapeSfA, strideSfA, tileShapesSfA, const_cast<void*>(ptrSfA));

            // Build TMA descriptor for gmem B block scaling factors.
            auto [shapeSfB, strideSfB, tileShapesSfB]
                = makeTmaShapeStrideSfAb(options, MatrixType::MatrixB, options.mSfLayoutB);
            params.tmaSfB
                = gemm::buildSfTmaDescriptor(dTypeSf, shapeSfB, strideSfB, tileShapesSfB, const_cast<void*>(ptrSfB));
        }

        if (options.mUseTmaStore)
        {
            // Shape/stride for gmem tensor C.
            auto [shapeC, strideC] = makeTmaShapeStrideAbc(options, MatrixType::MatrixC);

            // Swap M and N tiles for the M-major epilogue.
            auto outputTileM = options.mTransposeMmaOutput ? options.mEpilogueTileN : options.mEpilogueTileM;
            auto outputTileN = options.mTransposeMmaOutput ? options.mEpilogueTileM : options.mEpilogueTileN;
            // Build tma descriptor for C.
            params.tmaC = gemmGatedAct::buildNdTmaDescriptor(
                options.mDtypeC, shapeC, strideC, outputTileM, outputTileN / 2, const_cast<void*>(ptrC));
        }

        params.ptrC = ptrC;
        // Set the dequantization factors for A and B when DeepSeek FP8 recipe is used.
        params.ptrSfA = ptrSfA;
        params.ptrSfB = ptrSfB;
        params.ptrSfC = ptrSfC;

        // Set the per-token scale factors for MetaFP8 or scale inputs
        params.ptrPerTokenSfA = ptrPerTokenSfA;
        params.ptrPerTokenSfB = ptrPerTokenSfB;

        params.ptrScaleC = ptrScaleC;
        params.ptrScaleGate = ptrScaleGate;
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
