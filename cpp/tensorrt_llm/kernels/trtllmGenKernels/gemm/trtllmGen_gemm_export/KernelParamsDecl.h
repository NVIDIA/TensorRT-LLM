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

// NOTE: keep this code dependency free. It has to be included by the device code and has to be
// compilable with NVRTC.

namespace gemm
{

namespace gemm
{

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

    // The bias applied after the GEMM.
    // The bias is applied before applying the global scaling factor. I.e.
    // C' = (A * B + bias') * scaleC
    // scaleC = dequantA * dequantB * quantC
    // Thus, the bias' = bias / (dequantA * dequantB), where the bias is the original bias.
    //
    // if BiasType is N, the shape is [N].
    // The bias is broadcasted along the M dimension.
    //
    // if BiasType is M, the shape is [M].
    // The bias is broadcasted along the N dimension.
    //
    // The dtype is float32.
    void const* ptrBias;

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
#endif
};

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace gemm

} // namespace gemm
