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

#include "BatchedGemmEnums.h"
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

namespace batchedGemm
{

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
    //    If layoutA is MatrixLayout::MajorK
    //       Logical shape is [B, divUpMul(M, tileM), K].
    //       Logical strides are [divUpMul(M, tileM) * K, K, 1].
    //       Tile box shape is [1, tileM, tileK].
    //       Tile box strides are [0, tileK, 1].
    //    If layoutA is MatrixLayout::Mn
    //       Logical shape is [B, K, divUpMul(M, tileM)].
    //       Logical strides are [K * divUpMul(M, tileM), divUpMul(M, tileM), 1].
    //       Tile box shape is [1, tileK, tileM].
    //       Tile box strides are [0, tileM, 1].
    //    If layoutA is MatrixLayout::BlockMajorK
    //       Logical shape is [B, K / blockK, divUpMul(M, tileM), blockK].
    //       Logical strides are [K * divUpMul(M, tileM),  divUpMul(M, tileM) * blockK, blockK, 1].
    //       Tile box shape is [1, tileK / min(blockK, tileK), tileM, min(blockK, tileK)].
    //       Tile box strides are [0, tileM * min(blockK, tileK), min(blockK, tileK), 1].
    //       where blockK is 128B.
    //
    // Dtype is set from options.mDtypeA.
    CUtensorMap tmaA[1];

    // TMA descriptor for B.
    // Must be setup using gemm::buildNdTmaDescriptor with shapes and strides from
    // makeTmaShapeStrideAbc.
    //
    // If batchM:
    //    If layoutB is MatrixLayout::MajorK
    //       Logical shape is [B, divUpMul(N, tileN), K].
    //       Logical strides are [divUpMul(N, tileN) * K, K, 1].
    //       Tile box shape is [1, tileN, tileK].
    //       Tile box strides are [0, tileK, 1].
    //    If layoutB is MatrixLayout::MajorMn
    //       Logical shape is [B, K, divUpMul(N, tileN)].
    //       Logical strides are [K * divUpMul(N, tileN), divUpMul(N, tileN), 1].
    //       Tile box shape is [1, tileK, tileN].
    //       Tile box strides are [0, tileN, 1].
    //    If layoutB is MatrixLayout::BlockMajorK
    //       Logical shape is [B, K / blockK, divUpMul(N, tileN), blockK].
    //       Logical strides are [K * divUpMul(N, tileN),  divUpMul(N, tileN) * blockK, blockK, 1].
    //       Tile box shape is [1, tileK / min(blockK, tileK), tileN, min(blockK, tileK)].
    //       Tile box strides are [0, tileN * min(blockK, tileK), min(blockK, tileK), 1].
    //       where blockK is 128B.
    //
    // If batchN:
    //    Logical shape is [sum(divUpMul(N[bi], tileN) for bi in B), K].
    //    Logical strides are [K, 1].
    //    Tile box shape is [tileN, tileK].
    //    Tile box strides are [tileK, 1].
    //
    // Dtype is set from options.mDtypeB.
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
    // Equals to K * dtypeGetNumBits(dtypeA) / 8.
    uint64_t strideInBytesA;

    // The input matrix B.
    // If (routeAct == true && batchN), the shape is [N, K]. tmaB is not used.
    // Otherwise, check layout of tmaB to see the shape and strides.
    void const* ptrB;
    // The stride for matrix B in bytes.
    // Equals to K * dtypeGetNumBits(dtypeB) / 8.
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

    // The alpha and beta for SwiGlu and Swish.
    // Shape is [B]. One alpha and one beta per tensor in batch.
    float const* ptrAlpha;
    float const* ptrBeta;

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

    // The bias applied after the GEMM and before the activation function.
    // The bias is applied before applying the global scaling factor. I.e.
    // C = act(A * B + bias') * scaleC
    // scaleC = dequantA * dequantB * quantC
    // Thus, the bias' = bias / (dequantA * dequantB), where the bias is the original bias.
    //
    // If batchM, BiasType must be N, and bias shape is [B, N].
    // The bias is broadcasted along the M dimension.
    //
    // If batchNm BiasType must be M, and bias shape is [B, M].
    // The bias is broadcasted along the N dimension.
    //
    // The dtype is float32.
    void const* ptrBias{nullptr};

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

    // Total number of batches
    int32_t numBatches;

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
    template <class GemmOptions>
    static auto makeTmaShapeStrideAbc(
        GemmOptions const& options, int mM, int mN, int mK, int tileM, int tileN, int tileK, MatrixType matrixType)
    {
        // Weights matrix is A if we transpose the output of MMA (to have it M-major).
        // Otherwise, it is B, when the output of MMA is K-major.
        bool const isWeights = (matrixType == MatrixType::MatrixA && options.mTransposeMmaOutput)
            || (matrixType == MatrixType::MatrixB && !options.mTransposeMmaOutput);

        // The outer dimension.
        auto numTokens = (matrixType == MatrixType::MatrixA || matrixType == MatrixType::MatrixC) ? mM : mN;
        // The outer dimension tile size.
        auto tileNumTokens = (matrixType == MatrixType::MatrixC) ? options.mEpilogueTileM
            : (matrixType == MatrixType::MatrixA)                ? tileM
                                                                 : tileN;
        // The inner dimension.
        auto hiddenSize = (matrixType == MatrixType::MatrixC) ? mN : mK;
        // The inner dimension tile size.
        auto tileHiddenSize = (matrixType == MatrixType::MatrixC) ? options.mEpilogueTileN : tileK;

        // Swap matrix C sizes if output is transpose
        if (matrixType == MatrixType::MatrixC && options.mTransposeMmaOutput)
        {
            numTokens = mN;
            hiddenSize = mM;
            tileNumTokens = options.mEpilogueTileN;
            tileHiddenSize = options.mEpilogueTileM;
        }

        // For a fused activation kernel, the hidden size of output is halved. TODO: That's true for
        // gated activations but not regular activations.
        if (options.mFusedAct)
        {
            if (matrixType == MatrixType::MatrixC)
            {
                hiddenSize /= 2;
                tileHiddenSize /= 2;
            }
        }

        // The cute tensor shape for A/B: (numTokens, hiddenSize).
        // Note that TMA descriptor expects the first dimension's stride to be
        // 1, so swap the first two dimension so that the hiddenSize dimension comes first.
        auto shape = std::vector<uint64_t>{static_cast<uint64_t>(hiddenSize), static_cast<uint64_t>(numTokens)};
        // If the matrix is a weights matrix, we use 3D logical shape for it (B, M, K) or (B, N, K).
        // Ativations matrix is 2D (sum(divUpMul(M[bi], tileM) for bi in B), K).
        if (isWeights)
        {
            shape.push_back(static_cast<uint64_t>(options.mNumBatches));
        }

        // Assemble the stride (strideTokens, 1).
        // Swap the first two dimension as mentioned before.
        auto stride = std::vector<uint64_t>{1, static_cast<uint64_t>(hiddenSize)};
        if (isWeights)
        {
            stride.push_back(static_cast<uint64_t>(hiddenSize * numTokens));
        }

        // Assemble the box shape
        std::vector<int32_t> tileShape = {tileHiddenSize, tileNumTokens};

        // Alternate layouts do not apply to matrixC
        if (matrixType != MatrixType::MatrixC)
        {
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
        tg::Dtype dtypeElt, tg::SfLayout layout, int sfReshapeFactor)
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
    static KernelParams setKernelParams(GemmOptions_ const& options, bool const batchM, void const* ptrA,
        void const* ptrB, void* ptrC, void const* dSfA, void const* dSfB, void const* ptrPerTokenSfA,
        void const* ptrPerTokenSfB, void const* ptrBias, void* dSfC, float const* ptrScaleC, float const* ptrScaleGate,
        float const* ptrAlpha, float const* ptrBeta, int32_t const* routeMap, float* rowMax, uint32_t* rowMaxBars,
        int32_t const* ptrNumNonExitingCtas = nullptr, int32_t const* ptrTotalNumPaddedTokens = nullptr,
        int32_t const* ptrCtaIdxXyToBatchIdx = nullptr, int32_t const* ptrCtaIdxXyToMnLimit = nullptr,
        int32_t const maxNumCtas = MaxNumCtas)
    {

        static_assert(sizeof(KernelParams) <= 32 * 1024, "sizeof(KernelParams) has to be less or equal than 32KB");

        // Create the return struct.
        KernelParams params;

        params.ptrRouteMap = routeMap;
        params.numTokens = options.mNumTokens;

        params.ptrScaleC = ptrScaleC;
        params.ptrScaleGate = ptrScaleGate;

        params.ptrAlpha = ptrAlpha;
        params.ptrBeta = ptrBeta;

        int32_t ctaOffset = 0;

        // Compute totalNumPaddedTokens, ctaIdxXyToBatchIdx and ctaIdxXyToMnLimit if the batch dims are
        // known at kernel launch time. Otherwise, these parameters are defined in the device buffers:
        // ptrTotalNumPaddedTokens, ptrCtaIdxXyToBatchIdx and ptrCtaIdxXyToMnLimit respectively.

        if (options.mIsStaticBatch)
        {
            params.totalNumPaddedTokens = 0;
            for (int b = 0; b < options.mNumBatches; b++)
            {

                int mM = batchM ? options.mBatchedM[b] : options.mN;
                int mN = batchM ? options.mM : options.mBatchedN[b];

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
                if (ctaOffset + numCtas > MaxNumCtas)
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
                auto [shapeSfA, strideSfA, tileShapesSfA] = makeTmaShapeStrideSfAb(options.mM * options.mNumBatches,
                    options.mN, options.mK, MatrixType::MatrixA, options.mTileM, options.mTileN, options.mTileK,
                    options.mDtypeA, tg::SfLayout::R128c4, options.mSfReshapeFactor);
                params.tmaSfA[0]
                    = gemm::buildSfTmaDescriptor(dTypeSf, shapeSfA, strideSfA, tileShapesSfA, const_cast<void*>(dSfA));
            }

            if (options.mDtypeB == tg::Dtype::E2m1 || options.mDtypeB == tg::Dtype::MxE4m3
                || options.mDtypeB == tg::Dtype::MxE2m1)
            {
                tg::Dtype const dTypeSf = (options.mDtypeB == tg::Dtype::E2m1) ? tg::Dtype::E4m3 : tg::Dtype::UE8m0;

                if (batchedGemm::doesRouteImplUseTma(options.mRouteImpl))
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
                else if (batchedGemm::doesRouteImplUseNoRoute(options.mRouteImpl))
                {

                    // The input is padded:
                    // [act0, padding, padding, ... TileN size .., act1, padding, padding, ...]

                    auto const inputNumTokensSfB = ctaOffset * options.mTileN;

                    // Build TMA descriptor for gmem B block scaling factors.
                    auto [shapeSfB, strideSfB, tileShapesSfB] = makeTmaShapeStrideSfAb(options.mM, inputNumTokensSfB,
                        options.mK, MatrixType::MatrixB, options.mTileM, options.mTileN, options.mTileK,
                        options.mDtypeB, options.mSfLayoutB, options.mSfReshapeFactor);
                    params.tmaSfB[0] = gemm::buildSfTmaDescriptor(
                        dTypeSf, shapeSfB, strideSfB, tileShapesSfB, const_cast<void*>(dSfB));
                }
            }

            // C is the output activation
            if (options.mUseTmaStore)
            {
                // Shape/stride for gmem tensor C.
                auto [shapeC, strideC, tileShapeC]
                    = makeTmaShapeStrideAbc(options, options.mM, ctaOffset * options.mTileN, options.mK, options.mTileM,
                        options.mTileN, options.mTileK, MatrixType::MatrixC);
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
                auto [shapeA, strideA, tileShapeA] = makeTmaShapeStrideAbc(options, inputNumTokens, options.mN,
                    options.mK, options.mTileM, options.mTileN, options.mTileK, MatrixType::MatrixA);
                // Build tma descriptor for A.
                params.tmaA[0] = gemm::buildNdTmaDescriptor(
                    options.mDtypeA, options.mMmaKind, shapeA, strideA, tileShapeA, const_cast<void*>(ptrA));
            }

            if (options.mDtypeA == tg::Dtype::E2m1 || options.mDtypeA == tg::Dtype::MxE4m3
                || options.mDtypeA == tg::Dtype::MxE2m1)
            {
                tg::Dtype const dTypeSf = (options.mDtypeA == tg::Dtype::E2m1) ? tg::Dtype::E4m3 : tg::Dtype::UE8m0;

                if (options.mRouteImpl == batchedGemm::RouteImpl::NoRoute)
                {

                    // The input is padded:
                    // [act0, padding, padding, ... tileM size .., act1, padding, padding, ...]
                    auto const inputNumTokensSfA = ctaOffset * options.mTileM;

                    // Build TMA descriptor for gmem A block scaling factors.
                    auto [shapeSfA, strideSfA, tileShapesSfA] = makeTmaShapeStrideSfAb(inputNumTokensSfA, options.mN,
                        options.mK, MatrixType::MatrixA, options.mTileM, options.mTileN, options.mTileK,
                        options.mDtypeA, tg::SfLayout::R128c4, options.mSfReshapeFactor);
                    params.tmaSfA[0] = gemm::buildSfTmaDescriptor(
                        dTypeSf, shapeSfA, strideSfA, tileShapesSfA, const_cast<void*>(dSfA));
                }
            }

            if (options.mDtypeB == tg::Dtype::E2m1 || options.mDtypeB == tg::Dtype::MxE4m3
                || options.mDtypeB == tg::Dtype::MxE2m1)
            {
                tg::Dtype const dTypeSf = (options.mDtypeB == tg::Dtype::E2m1) ? tg::Dtype::E4m3 : tg::Dtype::UE8m0;

                // Build TMA descriptor for gmem B block scaling factors.
                auto [shapeSfB, strideSfB, tileShapesSfB] = makeTmaShapeStrideSfAb(options.mM,
                    options.mN * options.mNumBatches, options.mK, MatrixType::MatrixB, options.mTileM, options.mTileN,
                    options.mTileK, options.mDtypeB, options.mSfLayoutB, options.mSfReshapeFactor);
                params.tmaSfB[0]
                    = gemm::buildSfTmaDescriptor(dTypeSf, shapeSfB, strideSfB, tileShapesSfB, const_cast<void*>(dSfB));
            }

            // C is the output activation
            if (options.mUseTmaStore)
            {
                // Shape/stride for gmem tensor C.
                auto [shapeC, strideC, tileShapeC] = makeTmaShapeStrideAbc(options, ctaOffset * options.mTileM,
                    options.mN, options.mK, options.mTileM, options.mTileN, options.mTileK, MatrixType::MatrixC);
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
};

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace batchedGemm
