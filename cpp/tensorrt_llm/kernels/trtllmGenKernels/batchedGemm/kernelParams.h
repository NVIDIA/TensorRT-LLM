/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION &
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

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/kernels/multiHeadAttentionCommon.h"

#include <cassert>
#include <cstdint>
#include <cuda.h>
#include <iostream>

namespace tensorrt_llm::kernels
{

#define TLLM_CHECK_ERROR(cond, info, ...)                                                                              \
    do                                                                                                                 \
    {                                                                                                                  \
        if (!(cond))                                                                                                   \
        {                                                                                                              \
            tensorrt_llm::common::throwRuntimeError(                                                                   \
                __FILE__, __LINE__, tensorrt_llm::common::fmtstr(info, ##__VA_ARGS__));                                \
        }                                                                                                              \
    } while (0)

inline CUtensorMap buildNdTmaDescriptor(Data_type dtype, std::vector<uint64_t> const& shapes,
    std::vector<uint64_t> const& strides, int32_t tileSizeMn, int32_t tileSizeK, void* gmemAddr, bool doSwizzle = true)
{
    CUtensorMap desc{};
    // The data type.
    CUtensorMapDataType tmaDataFormat{};
    if (dtype == Data_type::DATA_TYPE_E4M3)
    {
        tmaDataFormat = CU_TENSOR_MAP_DATA_TYPE_UINT8;
    }
    else if (dtype == Data_type::DATA_TYPE_FP16)
    {
        tmaDataFormat = CU_TENSOR_MAP_DATA_TYPE_FLOAT16;
    }
    else if (dtype == Data_type::DATA_TYPE_BF16)
    {
        tmaDataFormat = CU_TENSOR_MAP_DATA_TYPE_BFLOAT16;
    }
    else if (dtype == Data_type::DATA_TYPE_E2M1)
    {
        tmaDataFormat = CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B;
    }
    else if (dtype == Data_type::DATA_TYPE_FP32)
    {
        tmaDataFormat = CU_TENSOR_MAP_DATA_TYPE_FLOAT32;
    }
    else
    {
        std::cerr << "Unexpected dtype " << static_cast<int32_t>(dtype) << std::endl;
        assert(false);
    }

    // The swizzle type.
    CUtensorMapSwizzle swizzleType{};
    int32_t tileKSizeInBytes = (tileSizeK * get_size_in_bits(dtype)) / /* bits */ 8;
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
    int32_t const dim = shapes.size();
    // Expect 2 dimensions.
    assert(dim == 2 || dim == 3);
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
        stridesInBytes[ii] = (strides[ii + 1] * get_size_in_bits(dtype)) / /* bits */ 8;
    }

    // Set the number of elements in the packed uint32_t element.
    auto const numEltsPerUInt32 = 4 * /* bits */ 8 / get_size_in_bits(dtype);
    // The number of elements in 128B.
    auto const numEltsIn128B = numEltsPerUInt32 /*4B*/ * 32;
    // The number of tile K hidden size (per token) in each block of shared memory.
    auto const numEltsInClampedTileKSize = std::min((int32_t) numEltsIn128B, tileSizeK);

    // Build tile shapes.
    std::vector<uint32_t> tileShapes(dim, 1);
    tileShapes[0] = numEltsInClampedTileKSize; // tileSizeK
    tileShapes[1] = tileSizeMn;                // tileSizeMn

    // Set tile strides to 1;
    std::vector<uint32_t> tileStrides(dim, 1);

    // Build the descriptor.
    CUresult const result = cuTensorMapEncodeTiled(&desc, tmaDataFormat,
        /*tensorRank=*/dim, gmemAddr, shapes.data(), stridesInBytes.data(), tileShapes.data(), tileStrides.data(),
        /*interleave=*/CU_TENSOR_MAP_INTERLEAVE_NONE, swizzleType,
        /*l2Promotion=*/CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        /*oobFill=*/CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

    if (result != CUDA_SUCCESS)
    {
        std::cerr << "Error: Failed to initialize the TMA descriptor " << result << std::endl;

        std::cerr << "tmaFormat: " << static_cast<int>(tmaDataFormat) << " dim: " << dim << " gmem: " << gmemAddr
                  << std::endl;

        std::cerr << "Shape: ";
        for (int ii = 0; ii < dim; ++ii)
        {
            std::cerr << shapes[ii] << " ";
        }
        std::cerr << std::endl;

        std::cerr << "Stride: ";
        for (int ii = 0; ii < dim - 1; ++ii)
        {
            std::cerr << stridesInBytes[ii] << " ";
        }
        std::cerr << std::endl;

        std::cerr << "tileShapes: ";
        for (int ii = 0; ii < dim; ++ii)
        {
            std::cerr << tileShapes[ii] << " ";
        }
        std::cerr << std::endl;

        std::cerr << "tileStrides: ";
        for (int ii = 0; ii < dim; ++ii)
        {
            std::cerr << tileStrides[ii] << " ";
        }
        std::cerr << std::endl;
        std::cerr << "swizzleType: " << int(swizzleType) << std::endl;
        assert(false);
    }

    return desc;
}

struct TrtllmGenBatchedGemmKernelParams
{
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
    // The scaling factors for the original tokens
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

        // The cute tensor shape for A/B: (numTokens, hiddenSize).
        // Note that TMA descriptor expects the first dimension's stride to be
        // 1, so swap the first two dimension so that the hiddenSize dimension comes first.
        auto shape = std::vector<uint64_t>{static_cast<uint64_t>(hiddenSize), static_cast<uint64_t>(numTokens)};

        // Assemble the stride (strideTokens, 1).
        // Swap the first two dimension as mentioned before.
        auto stride = std::vector<uint64_t>{1, static_cast<uint64_t>(hiddenSize)};

        return std::make_tuple(shape, stride);
    }

    // Setup the kernel parameters.
    static TrtllmGenBatchedGemmKernelParams setKernelParams(int32_t const numBatches, int32_t const numTokens,
        bool const batchM, int32_t const m, int32_t const n, int32_t const k, std::vector<int32_t> batchedM,
        std::vector<int32_t> batchedN, int const tileM, int const tileN, int const tileK, int const epilogueTileM,
        int const epilogueTileN, bool const useDeepSeekFp8, bool const useTmaStore, bool const transposeMmaOutput,
        bool const useFusedAct, Data_type dtypeElt, Data_type dtypeC, void* ptrA, void* ptrB, void* ptrC,
        float const* ptrScaleC, float* dDqSfsA, float* dDqSfsB, float* dDqSfsC, void* dSfA, void* dSfB, void* dSfTokens,
        void* dSfC, float const* ptrScaleGate, void* ptrTokens, int32_t const* routeMap, float* dDqSfsTokens,
        float* rowMax, uint32_t* rowMaxBars)
    {

        static_assert(sizeof(TrtllmGenBatchedGemmKernelParams) <= 32 * 1024,
            "sizeof(KernelParams) has to be less or equal than 32KB");

        // Create the return struct.
        TrtllmGenBatchedGemmKernelParams params{};

        TLLM_CHECK_ERROR(numBatches <= TrtllmGenBatchedGemmKernelParams::MaxBatchSize, "GEMM batch limit reached.");

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

        if (useDeepSeekFp8 && dtypeElt == Data_type::DATA_TYPE_E4M3)
        {
            params.ptrDqSfsA = dDqSfsA;
            params.ptrDqSfsB = dDqSfsB;
        }

        if (useDeepSeekFp8 && dtypeC == Data_type::DATA_TYPE_E4M3)
        {
            params.ptrDqSfsC = dDqSfsC;
        }

        params.ptrA = ptrA;
        params.ptrB = ptrB;
        params.strideInBytesA = k * get_size_in_bits(dtypeElt) / 8;
        params.strideInBytesB = k * get_size_in_bits(dtypeElt) / 8;

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
            params.tmaA[0] = buildNdTmaDescriptor(dtypeElt, shapeA, strideA, tileM, tileK, ptrA);

            // B is the activation
            // Shape/stride for gmem tensor B.
            auto [shapeB, strideB]
                = makeTmaShapeStrideAbc(transposeMmaOutput, useFusedAct, m, ctaOffset * tileN, k, MatrixType::MatrixB);
            // Build tma descriptor for B.
            params.tmaB[0] = buildNdTmaDescriptor(dtypeElt, shapeB, strideB, tileN, tileK, ptrB);

            // C is the output activation
            if (useTmaStore)
            {
                // Shape/stride for gmem tensor C.
                auto [shapeC, strideC] = makeTmaShapeStrideAbc(
                    transposeMmaOutput, useFusedAct, m, ctaOffset * tileN, k, MatrixType::MatrixC);

                // Swap M and N tiles for the M-major epilogue.
                auto outputTileM = transposeMmaOutput ? epilogueTileN : epilogueTileM;
                auto outputTileN = transposeMmaOutput ? epilogueTileM : epilogueTileN;

                // Build tma descriptor for C.
                params.tmaC[0] = buildNdTmaDescriptor(dtypeC, shapeC, strideC, outputTileM, outputTileN, ptrC);
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
            params.tmaB[0] = buildNdTmaDescriptor(dtypeElt, shapeB, strideB, tileN, tileK, ptrB);

            // A is the activation
            // Shape/stride for gmem tensor A.
            auto [shapeA, strideA]
                = makeTmaShapeStrideAbc(transposeMmaOutput, useFusedAct, ctaOffset * tileM, n, k, MatrixType::MatrixA);
            // Build tma descriptor for A.
            params.tmaA[0] = buildNdTmaDescriptor(dtypeElt, shapeA, strideA, tileM, tileK, ptrA);

            // C is the output activation
            if (useTmaStore)
            {
                // Shape/stride for gmem tensor C.
                auto [shapeC, strideC] = makeTmaShapeStrideAbc(
                    transposeMmaOutput, useFusedAct, ctaOffset * tileM, n, k, MatrixType::MatrixC);

                // Swap M and N tiles for the M-major epilogue.
                auto outputTileM = transposeMmaOutput ? epilogueTileN : epilogueTileM;
                auto outputTileN = transposeMmaOutput ? epilogueTileM : epilogueTileN;

                // Build tma descriptor for C.
                params.tmaC[0] = buildNdTmaDescriptor(dtypeC, shapeC, strideC, outputTileM, outputTileN, ptrC);
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
};
} // namespace tensorrt_llm::kernels
