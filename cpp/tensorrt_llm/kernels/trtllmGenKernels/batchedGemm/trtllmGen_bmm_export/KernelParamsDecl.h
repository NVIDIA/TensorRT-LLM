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

namespace batchedGemm
{

// This is device code

struct KernelParams
{
    //////////////////////////////////////////////////////////////////////////////////////////////////
    //
    // BatchedGemm parameters.
    //
    //////////////////////////////////////////////////////////////////////////////////////////////////

    // Maximum number of CTAs in the batch-token dimension.
    static constexpr int MaxNumCtas = 2048;

    //
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
    void const* ptrA{nullptr};

    // The stride for matrix A in bytes.
    // Equals to K * dtypeGetNumBits(dtypeA) / 8.
    uint64_t strideInBytesA;

    // The input matrix B.
    // If (routeAct == true && batchN), the shape is [N, K]. tmaB is not used.
    // Otherwise, check layout of tmaB to see the shape and strides.
    void const* ptrB{nullptr};
    // The stride for matrix B in bytes.
    // Equals to K * dtypeGetNumBits(dtypeB) / 8.
    uint64_t strideInBytesB;

    // The output matrix C. Check "logical" layout of tmaC to see the shape and strides.
    void* ptrC{nullptr};

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
    float const* ptrScaleC{nullptr};

    // The output gate scale for MxFp{4,8}, Fp8, NvFp4 and DeepSeek FP8 quantization.
    // TensorRT-LLM API requires a scaling factor on the device.
    // Shape is [B]. One scaling factor per tensor in batch.
    float const* ptrScaleGate{nullptr};

    // The clamp limit before the activation.
    // Shape is [B].
    // Clamp is INF if nullptr.
    // If applied on SwiGlu, it will be:
    //
    //   x_glu    = x_glu.clamp(min=None, max=limit)
    //   x_linear = x_linear.clamp(min=-limit, max=limit)
    float const* ptrClampLimit{nullptr};

    // The alpha and beta for SwiGlu or GeGlu.
    // Shape is [B]. One alpha and one beta per tensor in batch.
    // Alpha is 1.f if nullptr.
    // Beta is 0.f if nullptr.
    // The formula for SwiGlu (for GeGlu, replace sigmoid with phi):
    //
    //   out_glu  = x_glu * torch.sigmoid(alpha * x_glu) * (x_linear + beta)
    float const* ptrGatedActAlpha{nullptr};
    float const* ptrGatedActBeta{nullptr};

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
    float* ptrDqSfsC{nullptr};

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
    void const* ptrSfA{nullptr};

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
    void const* ptrSfB{nullptr};

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
    void const* ptrPerTokenSfA{nullptr};

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
    void const* ptrPerTokenSfB{nullptr};

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
    void* ptrSfC{nullptr};

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
    int32_t const* ptrRouteMap{nullptr};

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
    int32_t const* ptrNumNonExitingCtas{nullptr};

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
    int32_t const* ptrTotalNumPaddedTokens{nullptr};

    // Pointer to the map from the CTA index (in X/Y dim) to the batch index.
    // Maps CTA index in batch dim (i.e. blockDim.x if batchM, otherwise blockDim.y)
    // to batch index.
    // E.g. with listM = 128,255,32 and tileM = 128, should be equal to
    // ctaIdxXyToBatchIdx = [0, 1, 1, 2]
    // If isStaticBatch == true, ptrCtaIdxXyToBatchIdx should be set to nullptr and ctaIdxXyToBatchIdx
    // is used.
    int32_t const* ptrCtaIdxXyToBatchIdx{nullptr};

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
    int32_t const* ptrCtaIdxXyToMnLimit{nullptr};

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
    float* ptrPartialRowMax{nullptr};

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
    uint32_t* ptrRowMaxCompletionBars{nullptr};
};

///////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace batchedGemm
