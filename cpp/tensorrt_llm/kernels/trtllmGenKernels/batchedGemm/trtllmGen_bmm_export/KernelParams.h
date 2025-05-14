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
    //////////////////////////////////////////////////////////////////////////////////////////////////
    //
    // BatchedGemm parameters.
    //
    //////////////////////////////////////////////////////////////////////////////////////////////////

    // Maximum number of batch
    static constexpr int MaxBatchSize = 256;
    // Maximum number of CTAs
    static constexpr int MaxNumCtas = 2048;

    // TMA descriptor for A.
    // Must be setup using gemm::buildNdTmaDescriptor with shapes and strides from
    // makeTmaShapeStrideAbc.
    //
    // If batchM:
    //    Logical shape is [sum(divUpMul(M[bi], tileM) for bi in B), K].
    //    Logical strides are [K, 1].
    //    Tile box shape is [tileM, tileK].
    //    Tile box strides are [tileK, 1].
    //
    // If batchN:
    //    Logical shape is [B, divUpMul(M, tileM), K].
    //    Logical strides are [divUpMul(M, tileM) * K, K, 1].
    //    Tile box shape is [1, tileM, tileK].
    //    Tile box strides are [0, tileK, 1].
    //
    // Dtype is set from options.mDtypeElt.
    CUtensorMap tmaA[1];

    // TMA descriptor for B.
    // Must be setup using gemm::buildNdTmaDescriptor with shapes and strides from
    // makeTmaShapeStrideAbc.
    //
    // If batchM:
    //    Logical shape is [B, divUpMul(N, tileN), K].
    //    Logical strides are [divUpMul(N, tileN) * K, K, 1].
    //    Tile box shape is [1, tileN, tileK].
    //    Tile box strides are [0, tileK, 1].
    //
    // If batchN:
    //    Logical shape is [sum(divUpMul(N[bi], tileN) for bi in B), K].
    //    Logical strides are [K, 1].
    //    Tile box shape is [tileN, tileK].
    //    Tile box strides are [tileK, 1].
    //
    // Dtype is set from options.mDtypeElt.
    CUtensorMap tmaB[1];

    // TMA descriptor for C, (when useTmaStore is true)
    // Must be setup using gemm::buildNdTmaDescriptor with shapes and strides from
    // makeTmaShapeStrideAbc.
    //
    // If batchM:
    //    Logical shape is [sum(divUpMul(M[bi], tileM) for bi in B), N].
    //    Logical strides are [N, 1].
    //    Tile box shape is [epilogueTileM, epilogueTileN].
    //    Tile box strides are [epilogueTileN, 1].
    //
    // If batchN:
    //    Logical shape is [sum(divUpMul(N[bi], tileN) for bi in B), M].
    //    Logical strides are [M, 1].
    //    Tile box shape is [epilogueTileN, epilogueTileM].
    //    Tile box strides are [epilogueTileM, 1].
    //
    // Dtype is set from options.mDtypeC.
    CUtensorMap tmaC[1];

    // TMA descriptor for the block scaling factors for A, for MxFp{4,8} and NvFp4 formats.
    // Must be setup using gemm::buildSfTmaDescriptor with shapes and strides from
    // makeTmaShapeStrideSfAb.
    // The layout of scaling factors for A is always R128c4.
    //
    // Let P be the number of elements per SF. P=16 for NvFp4, P=32 for Mx formats.
    // M must be a multiple of 128.
    // K must be a multiple of 4P.
    // The "logical" shape is: [paddedM, K / P], where paddedM is
    // sum(divUpMul(M[bi], tileM) for bi in B) if batchM,
    // otherwise divUpMul(M, TileM) * B.
    // The R128c4 layout is: [paddedM / 128, K / P / 4, 512].
    // The shape we use for TMA is: [paddedM / 128, K / P / 4, 2, 256].
    //
    // Dtype is Dtype::E4m3 for NvFp4, Dtype::UE8m0 for Mx formats.
    CUtensorMap tmaSfA[1];

    // TMA descriptor for the block scaling factors for B, for MxFp{4,8} and NvFp4 formats.
    // Must be setup using gemm::buildSfTmaDescriptor with shapes and strides from
    // makeTmaShapeStrideSfAb.
    // The layout of block scaling factors for B is controlled by options.mSfLayoutB.
    //
    // Let P be the number of elements per SF. P=16 for NvFp4, P=32 for Mx formats.
    // The "logical" shape is: [paddedN, K / 16]
    // where paddedN is sum(divUpMul(N[bi], tileN) for bi in B) if batchN,
    // otherwise divUpMul(N, TileN) * B.
    //
    // If the layout is R128c4,
    //    paddedN must be a multiple of 128.
    //    K must be a multiple of 4P.
    //    The R128c4 layout is: [paddedN / 128, K / P / 4, 512]
    //    The shape we use for TMA is: [paddedN / 128, K / P / 4, 2, 256]
    //
    // If the layout is R8c4,
    //    paddedN must be a multiple of 8.
    //    K must be a multiple of 4P.
    //    The R8c4 layout is: [paddedN / 8, K / P / 4, 32]
    //    The shape we use for TMA is: [paddedN / 8, K / P / 4 / repeats, repeats * 32]
    //    where repeats = min(tileK / P / 4, 8)
    //
    // Dtype is Dtype::E4m3 for NvFp4, Dtype::UE8m0 for Mx formats.
    CUtensorMap tmaSfB[1];

    // The input matrix A.
    // If (routeAct == true && batchM), the shape is [M, K]. tmaA is not used.
    // Otherwise, check layout of tmaA to see the shape and strides.
    void const* ptrA;

    // The stride for matrix A in bytes.
    // Equals to K * dtypeGetNumBits(dtypeElt) / 8.
    uint64_t strideInBytesA;

    // The input matrix B.
    // If (routeAct == true && batchN), the shape is [N, K]. tmaB is not used.
    // Otherwise, check layout of tmaB to see the shape and strides.
    void const* ptrB;
    // The stride for matrix B in bytes.
    // Equals to K * dtypeGetNumBits(dtypeElt) / 8.
    uint64_t strideInBytesB;

    // The output matrix C. Check "logical" layout of tmaC to see the shape and strides.
    void* ptrC;

    // Inputs and output are MxFp{4,8}, Fp8, NvFp4.
    // The scaling factors to apply to the output - can be used to incorporate input scaling factors
    // as described below: C = SEncC * act(SDecA * SDecB * A * Bl) . (SDecA * SDecB * A * Br)
    //  -> ScaleGate = SDecA * SDecB
    //     ScaleC    = SDecA * SDecB * SEncC
    //
    // Only the inputs are MxFp{4,8}, Fp8, NvFp4.
    // C = act(SDecA * SDecB * A * Bl) . (SDecA * SDecB * A * Br)
    //  -> ScaleGate = SDecA * SDecB
    //     ScaleC    = SDecA * SDecB
    //
    // Only the output is MxFp{4,8}, Fp8, NvFp4.
    // C = SEncC * act(A * Bl) . (A * Br)
    //  -> ScaleGate = 1
    //     ScaleC    = SEncC
    //
    // The output tensor scaling factor for MxFp{4,8}, Fp8, NvFp4 and DeepSeek FP8 quantization.
    // TensorRT-LLM API requires a scaling factor on the device.
    // Shape is [B]. One scaling factor per tensor in batch.
    float const* ptrScaleC;

    // The output gate scale for MxFp{4,8}, Fp8, NvFp4 and DeepSeek FP8 quantization.
    // TensorRT-LLM API requires a scaling factor on the device.
    // Shape is [B]. One scaling factor per tensor in batch.
    float const* ptrScaleGate;

    // The K dimension. It is the hidden dimension of the input matrices.
    int32_t k;

    // The non-batched dimension.
    // It is N if batchM, otherwise M.
    int32_t nm;

    // Tile stride per batch for the non-batched dimension.
    // It is N / TileN if batchM, otherwise M / TileM.
    int32_t tileStridePerBatch;

    // TODO get rid of that.
    // DeepSeek FP8 scaling factors for C
    float* ptrDqSfsC;

    // The block scaling factors for A.
    // The pointer must always be set regardless of the quantization recipe.
    // If (routeAct == true && batchM), the shape is [M, K / 16]. tmaSfA is not used.
    //    For the layout (r128c4), see below.
    // Otherwise,
    //    If MxFp{4,8} and NvFp4 formats are used,
    //      check the "logical" layout of tmaSfA to see the shape and strides.
    //      The dtype is Dtype::E4m3.
    //
    //    If DeepSeek FP8 quantization recipe is used,
    //      If batchM:
    //        The shape is [K / 128, paddedM],
    //        where paddedM is sum(divUpMul(M[bi], tileM) for bi in B).
    //      If batchN:
    //        The shape is [M / 128, K / 128],
    //      The rightmost dimension is contiguous in memory.
    //      The dtype is Dtype::Float32.
    void const* ptrSfA;

    // The block scaling factors for B.
    // The pointer must always be set regardless of the quantization recipe.
    // If (routeAct == true && batchN), the shape is [N, K / 16]. tmaSfB is not used.
    //    For the layout (r128c4, r8c4), see below.
    // Otherwise,
    //    If MxFp{4,8} and NvFp4 formats are used,
    //      check the layout of tmaSfB to see the shape and strides.
    //      The dtype is Dtype::E4m3.
    //
    //    If DeepSeek FP8 quantization recipe is used,
    //      If batchM:
    //        The shape is [N / 128, K / 128],
    //      If batchN:
    //        The shape is [K / 128, paddedN],
    //        where paddedN is sum(divUpMul(N[bi], tileN) for bi in B).
    //      The rightmost dimension is contiguous in memory.
    //      The dtype is Dtype::Float32.
    void const* ptrSfB;

    // The per-token scaling factors from scale A.
    //
    // This is used for either:
    //   * Per-token scaling factor quantization schemes, such as MetaFP8. The dtype is Dtype::Float32
    //   * When the routing scales are applied to the input activations (only when output is not
    //   transposed). The dtype is Dtype::Bfloat16
    //
    // if (batchM (A is activations)):
    //     Logical shape is [sum(divUpMul(M[bi], tileM) for bi in B)]
    //
    // if (batchN (A is weights)):
    //     Logical shape is [B, divUpMul(M, tileM)]
    //
    void const* ptrPerTokenSfA;

    // The per-token scaling factors from scale B.
    //
    // This is used for either:
    //   * Per-token scaling factor quantization schemes, such as MetaFP8. The dtype is Dtype::Float32
    //   * When the routing scales are applied to the input activations (only when output is
    //   transposed). The dtype is Dtype::Bfloat16
    //
    // if (batchM (B is weights)):
    //     Logical shape is [B, divUpMul(N, tileN)]
    //
    // if (batchN (B is activations)):
    //     Logical shape is [sum(divUpMul(N[bi], tileN) for bi in B)]
    void const* ptrPerTokenSfB;

    // The output block scaling factors for C.
    //
    // If MxFp{4,8} and NvFp4 formats are used,
    // The "logical" shape is:
    //    if batchM: [paddedM, N / 16]
    //    if batchN: [paddedN, M / 16]
    // where paddedM is sum(divUpMul(M[bi], tileM) for bi in B),
    // where paddedN is sum(divUpMul(N[bi], tileN) for bi in B).
    //
    // If the layout is R128c4,
    //    paddedOuter must be a multiple of 128.
    //    inner must be a multiple of 64.
    //    The R128c4 layout is: [paddedOuter / 128, inner / 16 / 4, 512]
    //    The shape we use for TMA is: [paddedOuter / 128, inner / 16 / 4, 2, 256]
    //    where inner = N if batchM, otherwise M.
    //    where paddedOuter = paddedM if batchM, otherwise paddedN.
    //
    // If the layout is R8c4,
    //    paddedOuter must be a multiple of 8.
    //    inner must be a multiple of 64.
    //    The R8c4 layout is: [paddedOuter / 8, inner / 16 / 4, 32]
    //    The shape we use for TMA is: [paddedOuter / 8, inner / 16 / 4 / repeats, repeats * 32]
    //    where repeats = min(tileInner / 16 / 4, 8),
    //    where tileInner = tileN if batchM, otherwise tileM,
    //    where paddedOuter = paddedM if batchM, otherwise paddedN.
    //    where inner = N if batchM, otherwise M.
    //
    // The dtype is Dtype::E4m3.
    //
    // If DeepSeek FP8 quantization recipe is used,
    // If batchM:
    //   The shape is [N / 128, paddedM],
    //   where paddedM is sum(divUpMul(M[bi], tileM) for bi in B).
    // If batchN:
    //   The shape is [M / 128, paddedN],
    //   where paddedN is sum(divUpMul(N[bi], tileN) for bi in B).
    // The rightmost dimension is contiguous in memory.
    // The dtype is Dtype::Float32.
    void* ptrSfC;

    //////////////////////////////////////////////////////////////////////////////////////////////////
    //
    // Routing activations parameters.
    //
    //////////////////////////////////////////////////////////////////////////////////////////////////
    // These params are used when the kernel is configured with -routeAct true.
    // The inputs are not padded, but the outputs are padded to divUpMul(M[bi], tileM) for batchM or
    // divUpMul(N[bi], tileN) for batchN.
    // If -routeAct is false, the params are not used and should be set to zero.

    // The routeMap for the input tokens.
    // Map of expanded token index (counting the previous padded tokens) to the batch index
    // the token belongs to.
    // The shape is
    // [sum(divUpMul(M[bi], tileM) for bi in B)] for batchM
    // [sum(divUpMul(N[bi], tileN) for bi in B)] for batchN
    // The dtype is int32_t.
    //
    // There are 3 tokens [0, 1, 2] such that [0, 1] belong to batch [B0] and [2] to batch [B1].
    // Let's assume that the padded size is 4.
    //
    // The expanded indices for tokens [0, 1, 2] are:
    // expandedIdx[0] = 0
    // expandedIdx[1] = 1
    // expandedIdx[2] = divUpMul(2, 4) + 0 = 4
    //
    // The route map is [B0, B0, X, X, B1, X, X, X] where X could be any value.
    int32_t const* ptrRouteMap;

    // Total number of unpadded inputs
    int32_t numTokens;

    //////////////////////////////////////////////////////////////////////////////////////////////////
    //
    // Batching information parameters.
    //
    //////////////////////////////////////////////////////////////////////////////////////////////////

    // In some cases, some CTAs must early-exit. E.g. when the grid size is set statically, but the
    // actual workload is decided at runtime. This element on the device contains the number of CTAs
    // that do not early-exit. The number corresponds to the X dim of the grid when the output is not
    // transposed (i.e. batchM). To the Y dim, otherwise.
    // The size is 1 and the dtype is int32_t.
    // Used if isStaticBatch == false, otherwise set to nullptr.
    // The pointer points to a scalar and the dtype is int32_t. The pointed value must be >= 0.
    int32_t const* ptrNumNonExitingCtas;

    // Pointer to total number of padded tokens.
    // Computed as
    // int32_t totalNumPaddedTokens{0};
    // for (int bi = 0; bi < options.mNumBatches; bi++) {
    //   totalNumPaddedTokens += batchM ? divUpMul(options.mBatchedM[bi], options.mTileM)
    //                                  : divUpMul(options.mBatchedN[bi], options.mTileN);
    // }
    // The size is 1 and the dtype is int32_t.
    // If isStaticBatch == true, ptrTotalNumPaddedTokens should be set to nullptr and
    // totalNumPaddedTokens is used.
    int32_t const* ptrTotalNumPaddedTokens;

    // Pointer to the map from the CTA index (in X/Y dim) to the batch index.
    // Maps CTA index in batch dim (i.e. blockDim.x if batchM, otherwise blockDim.y)
    // to batch index.
    // E.g. with listM = 128,255,32 and tileM = 128, should be equal to
    // ctaIdxXyToBatchIdx = [0, 1, 1, 2]
    // If isStaticBatch == true, ptrCtaIdxXyToBatchIdx should be set to nullptr and ctaIdxXyToBatchIdx
    // is used.
    int32_t const* ptrCtaIdxXyToBatchIdx;

    // Pointer from the CTA index X/Y to the expanded tile index where the expanded tile index is
    // computed as:
    //
    // int expandedIdx = 0;
    // for (int bi = 0; bi < batchIdx-1; ++bi) {
    //   expandIdx = divUpMul(numTokens[bi], TileM/N);
    // }
    // expandIdx += <index in the batch>
    // E.g. with numTokens = [128,255,32] and tileM = 128, should be equal to
    // ptrCtaIdxXyToMnLimit = [128, 256, 383, 416]
    int32_t const* ptrCtaIdxXyToMnLimit;

    // Total number of padded tokens - used as the stride for the activation and C scaling factors.
    // Check ptrTotalNumPaddedTokens to see how it is computed.
    // If isStaticBatch == true, totalNumPaddedTokens is used, otherwise ptrTotalNumPaddedTokens.
    int32_t totalNumPaddedTokens;

    // A map from CTA index X/Y to batch index.
    // Check ptrCtaIdxXyToBatchIdx to see how it is computed.
    // If isStaticBatch == true, ctaIdxXyToBatchIdx is used, otherwise ptrCtaIdxXyToBatchIdx.
    int32_t ctaIdxXyToBatchIdx[MaxNumCtas];

    // **Expanded** limits for the batched dimension:
    //   tile * ctaIdxXyToTileIdxMn[ctaIdxXy] -> ctaIdxXyToMnLimit[ctaIdxXy]
    // Check ptrCtaIdxXyToMnLimit to see how it is computed.
    // If isStaticBatch == true, ctaIdxXyToMnLimit is used, otherwise ptrCtaIdxXyToMnLimit.
    int32_t ctaIdxXyToMnLimit[MaxNumCtas];

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
    // GatedAct parameters.
    //
    //////////////////////////////////////////////////////////////////////////////////////////////////

    // Pointer for partial row max for DeepSeek FP8 recipe.
    // This is temporary storage for the row max results.
    // If batchM, the shape is [2, totalNumPaddedTokens, N / 128] and the dtype is float.
    // Otherwise, the shape is [2, totalNumPaddedTokens, M / 128] and the dtype is float.
    float* ptrPartialRowMax;

    // Flags in global memory that sync on "exit" for row max computation.
    // The shape is [numTilesM * numTilesN / 2] and the dtype is uint32_t, where
    // if batchM,
    // numTilesM = divUp(totalNumPaddedTokens, tileM).
    // numTilesN = divUp(N, tileN).
    // Otherwise,
    // numTilesM = divUp(M, tileM).
    // numTilesN = divUp(totalNumPaddedTokens, tileN).
    //
    // The memory must be set to 0 before the kernel launch.
    uint32_t* ptrRowMaxCompletionBars;

    //////////////////////////////////////////////////////////////////////////////////////////////////
    //
    // Member functions.
    //
    //////////////////////////////////////////////////////////////////////////////////////////////////
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

    // Create the TMA shape/stride for A/B block scaling factors.
    static auto makeTmaShapeStrideSfAb(int mM, int mN, int mK, MatrixType matrixType, int tileM, int tileN, int tileK,
        tg::Dtype dtypeElt, tg::SfLayout layout)
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
        const int32_t numEltsPerSf = (dtypeElt == tg::Dtype::E2m1) ? 16 : 32;

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

        default: assert(false && "Unsupported SF layout");
        }
        return std::make_tuple(std::vector<uint64_t>{}, std::vector<uint64_t>{}, std::vector<uint32_t>{});
    }

    static KernelParams setKernelParams(int32_t const numBatches, int32_t const numTokens, bool const batchM,
        int32_t const m, int32_t const n, int32_t const k, std::vector<int32_t> const& batchedM,
        std::vector<int32_t> const& batchedN, int32_t const tileM, int32_t const tileN, int32_t const tileK,
        int32_t const epilogueTileM, int32_t const epilogueTileN, bool const useDeepSeekFp8, bool const useTmaStore,
        bool const transposeMmaOutput, tg::SfLayout sfLayoutB, bool const useFusedAct, tg::Dtype dtypeElt,
        tg::Dtype dtypeC, void const* ptrA, void const* ptrB, void* ptrC, void const* dSfA, void const* dSfB,
        void const* ptrPerTokenSfA, void const* ptrPerTokenSfB, void* dSfC, float const* ptrScaleC,
        float const* ptrScaleGate, int32_t const* ptrRouteMap, float* rowMax, uint32_t* rowMaxBars,
        bool isStaticBatch = true, int32_t const* ptrNumNonExitingCtas = nullptr,
        int32_t const* ptrTotalNumPaddedTokens = nullptr, int32_t const* ptrCtaIdxXyToBatchIdx = nullptr,
        int32_t const* ptrCtaIdxXyToMnLimit = nullptr)
    {

        static_assert(sizeof(KernelParams) <= 32 * 1024, "sizeof(KernelParams) has to be less or equal than 32KB");

        // Create the return struct.
        KernelParams params;

        assert(numBatches <= KernelParams::MaxBatchSize && "GEMM batch limit reached.");

        params.ptrRouteMap = ptrRouteMap;
        params.numTokens = numTokens;

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
                {
                    continue;
                }

                // The number of CTAs.
                int32_t numCtas = batchM ? (mM + tileM - 1) / tileM : (mN + tileN - 1) / tileN;
                // The size of the tile.
                int32_t tile = batchM ? tileM : tileN;
                // The problem size.
                int32_t mn = batchM ? mM : mN;
                int32_t tokensPerTile = mn;

                // Make sure we do not exceed the launch limit.
                assert(ctaOffset + numCtas <= MaxNumCtas && "Too many CTAs");

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

        if (useDeepSeekFp8 && dtypeC == tg::Dtype::E4m3)
        {
            params.ptrDqSfsC = reinterpret_cast<float*>(dSfC);
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
            assert(0 == m % tileM && "0 == mM %% tileM");
            params.tileStridePerBatch = m / tileM;
            params.nm = m;
            // Shape/stride for gmem tensor A.
            auto [shapeA, strideA]
                = makeTmaShapeStrideAbc(transposeMmaOutput, useFusedAct, m * numBatches, n, k, MatrixType::MatrixA);
            // Build tma descriptor for A.
            params.tmaA[0]
                = gemm::buildNdTmaDescriptor(dtypeElt, shapeA, strideA, tileM, tileK, const_cast<void*>(ptrA));

            // The input is padded:
            // [act0, padding, padding, ... TileN size .., act1, padding, padding, ...]
            auto const inputNumTokens = ctaOffset * tileN;
            // B is the activation
            // Shape/stride for gmem tensor B.
            auto [shapeB, strideB]
                = makeTmaShapeStrideAbc(transposeMmaOutput, useFusedAct, m, inputNumTokens, k, MatrixType::MatrixB);
            // Build tma descriptor for B.
            params.tmaB[0]
                = gemm::buildNdTmaDescriptor(dtypeElt, shapeB, strideB, tileN, tileK, const_cast<void*>(ptrB));

            if (dtypeElt == tg::Dtype::E2m1 || dtypeElt == tg::Dtype::MxE4m3)
            {
                tg::Dtype const dTypeSf = (dtypeElt == tg::Dtype::E2m1) ? tg::Dtype::E4m3 : tg::Dtype::UE8m0;

                // Build TMA descriptor for gmem A block scaling factors.
                auto [shapeSfA, strideSfA, tileShapesSfA] = makeTmaShapeStrideSfAb(
                    m * numBatches, n, k, MatrixType::MatrixA, tileM, tileN, tileK, dtypeElt, tg::SfLayout::R128c4);
                params.tmaSfA[0]
                    = gemm::buildSfTmaDescriptor(dTypeSf, shapeSfA, strideSfA, tileShapesSfA, const_cast<void*>(dSfA));

                // The input is padded:
                // [act0, padding, padding, ... TileN size .., act1, padding, padding, ...]
                auto const inputNumTokensSfB = ctaOffset * tileN;

                // Build TMA descriptor for gmem B block scaling factors.
                auto [shapeSfB, strideSfB, tileShapesSfB] = makeTmaShapeStrideSfAb(
                    m, inputNumTokensSfB, k, MatrixType::MatrixB, tileM, tileN, tileK, dtypeElt, sfLayoutB);
                params.tmaSfB[0]
                    = gemm::buildSfTmaDescriptor(dTypeSf, shapeSfB, strideSfB, tileShapesSfB, const_cast<void*>(dSfB));
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
            assert(0 == n % tileN && "0 == mN %% tileN");
            params.tileStridePerBatch = n / tileN;
            params.nm = n;
            // Shape/stride for gmem tensor B.
            auto [shapeB, strideB]
                = makeTmaShapeStrideAbc(transposeMmaOutput, useFusedAct, m, n * numBatches, k, MatrixType::MatrixB);
            // Build tma descriptor for B.
            params.tmaB[0]
                = gemm::buildNdTmaDescriptor(dtypeElt, shapeB, strideB, tileN, tileK, const_cast<void*>(ptrB));

            // A is the activation
            // Shape/stride for gmem tensor A.
            // The input is padded:
            // [act0, padding, padding, ... tileM size .., act1, padding, padding, ...]
            auto const inputNumTokens = ctaOffset * tileM;
            auto [shapeA, strideA]
                = makeTmaShapeStrideAbc(transposeMmaOutput, useFusedAct, inputNumTokens, n, k, MatrixType::MatrixA);
            // Build tma descriptor for A.
            params.tmaA[0]
                = gemm::buildNdTmaDescriptor(dtypeElt, shapeA, strideA, tileM, tileK, const_cast<void*>(ptrA));

            if (dtypeElt == tg::Dtype::E2m1 || dtypeElt == tg::Dtype::MxE4m3)
            {
                tg::Dtype const dTypeSf = (dtypeElt == tg::Dtype::E2m1) ? tg::Dtype::E4m3 : tg::Dtype::UE8m0;

                // The input is padded:
                // [act0, padding, padding, ... tileM size .., act1, padding, padding, ...]
                auto const inputNumTokensSfA = ctaOffset * tileM;

                // Build TMA descriptor for gmem A block scaling factors.
                auto [shapeSfA, strideSfA, tileShapesSfA] = makeTmaShapeStrideSfAb(
                    inputNumTokensSfA, n, k, MatrixType::MatrixA, tileM, tileN, tileK, dtypeElt, tg::SfLayout::R128c4);
                params.tmaSfA[0]
                    = gemm::buildSfTmaDescriptor(dTypeSf, shapeSfA, strideSfA, tileShapesSfA, const_cast<void*>(dSfA));

                // Build TMA descriptor for gmem B block scaling factors.
                auto [shapeSfB, strideSfB, tileShapesSfB] = makeTmaShapeStrideSfAb(
                    m, n * numBatches, k, MatrixType::MatrixB, tileM, tileN, tileK, dtypeElt, sfLayoutB);
                params.tmaSfB[0]
                    = gemm::buildSfTmaDescriptor(dTypeSf, shapeSfB, strideSfB, tileShapesSfB, const_cast<void*>(dSfB));
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

        params.ptrPartialRowMax = rowMax;
        params.ptrRowMaxCompletionBars = rowMaxBars;

        params.ptrNumNonExitingCtas = ptrNumNonExitingCtas;

        // Set the per-token scale factors for MetaFP8 or scale inputs
        params.ptrPerTokenSfA = ptrPerTokenSfA;
        params.ptrPerTokenSfB = ptrPerTokenSfB;

        return params;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    template <class GemmOptions_>
    static KernelParams setKernelParams(GemmOptions_ const& options, bool const batchM, void const* ptrA,
        void const* ptrB, void* ptrC, void const* dSfA, void const* dSfB, void const* ptrPerTokenSfA,
        void const* ptrPerTokenSfB, void* dSfC, float const* ptrScaleC, float const* ptrScaleGate,
        int32_t const* routeMap, float* rowMax, uint32_t* rowMaxBars, int32_t const* ptrNumNonExitingCtas = nullptr,
        int32_t const* ptrTotalNumPaddedTokens = nullptr, int32_t const* ptrCtaIdxXyToBatchIdx = nullptr,
        int32_t const* ptrCtaIdxXyToMnLimit = nullptr)
    {

        bool const useFusedAct = options.mFusedAct;

        return setKernelParams(options.mNumBatches, options.mNumTokens, batchM, options.mM, options.mN, options.mK,
            options.mBatchedM, options.mBatchedN, options.mTileM, options.mTileN, options.mTileK,
            options.mEpilogueTileM, options.mEpilogueTileN, options.mUseDeepSeekFp8, options.mUseTmaStore,
            options.mTransposeMmaOutput, options.mSfLayoutB, useFusedAct, options.mDtypeElt, options.mDtypeC, ptrA,
            ptrB, ptrC, dSfA, dSfB, ptrPerTokenSfA, ptrPerTokenSfB, dSfC, ptrScaleC, ptrScaleGate, routeMap, rowMax,
            rowMaxBars, options.mIsStaticBatch, ptrNumNonExitingCtas, ptrTotalNumPaddedTokens, ptrCtaIdxXyToBatchIdx,
            ptrCtaIdxXyToMnLimit);
    }
#endif
};

////////////////////////////////////////////////////////////////////////////////////////////////////
