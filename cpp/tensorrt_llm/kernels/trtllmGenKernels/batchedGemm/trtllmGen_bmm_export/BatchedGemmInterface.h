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

#include <numeric>
#include <optional>

#include "BatchedGemmOptions.h"
#include "KernelParams.h"
#include "trtllm/gen/CudaKernelLauncher.h"

#ifdef TLLM_GEN_EXPORT_INTERFACE
#include "KernelMetaInfo.h"
#endif // TLLM_GEN_EXPORT_INTERFACE

namespace batchedGemm
{

namespace batchedGemm
{

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// BatchedGemmData
//
////////////////////////////////////////////////////////////////////////////////////////////////////

struct BatchedGemmData
{
    struct ProblemDimensions
    {
        // The number of batches.
        int32_t mNumBatches{0};
        // The number of tokens. Set to 0 if routeAct is false.
        int32_t mNumTokens{0};
        // Whether the batch is on the M dimension.
        bool mBatchM{true};
        // The maximum number of CTAs in the token dimension.
        // Need to be set if mNumTokens > 0 and the token per batch
        // distribution is not known at launch time.
        // In this case, the kernel will launch mMaxNumCtasInTokenDim CTAs in token dim and exit early
        // if the idx of CTAs is larger or equal to mPtrNumNonExitingCtas.
        int32_t mMaxNumCtasInTokenDim{0};

        // Either mBatchedM or mBatchedN must be set when mNumTokens == 0, otherwise not used.
        // The number of tokens in each batch on the M dimension if batchM,
        // otherwise not used.
        // The number of elements in the array is mNumBatches.
        // E.g. to implement a BMM with each batch having M tokens, one needs to set mBatchedM to
        // {M, M, M, .. mNumBatches times ..}
        std::vector<int32_t> mBatchedM{};
        // The number of tokens in each batch on the N dimension if batchN,
        // otherwise not used.
        // The number of elements in the array is mNumBatches.
        // E.g. to implement a BMM with each batch having N tokens, one needs to set mBatchedN to
        // {N, N, N, .. mNumBatches times ..}
        std::vector<int32_t> mBatchedN{};

        // The M dimension.
        // It is the total number of tokens if A is the activation matrix.
        // It is the total number of output channels if A is the weight matrix.
        int32_t mM{0};
        // The N dimension.
        // It is the total number of tokens if B is the activation matrix.
        // It is the total number of output channels if B is the weight matrix.
        int32_t mN{0};
        // The K dimension. It is the hidden dimension of the input matrices.
        int32_t mK{0};
        // The rank id of the current device in the multi-gpu space.
        int32_t mRank{0};
        // The number of devices in tensor-parallel group.
        int32_t mWorldSize{1};
    };

    struct InputBuffers
    {
        // The matrix A. The data type is controlled by options.mDtypeA.
        //
        // If (routeAct == true && batchM), the shape is [M, K]
        // Else
        //   If batchM:
        //      Logical shape is [sum(divUpMul(M[bi], tileM) for bi in B), K].
        //      Logical strides are [K, 1].
        //
        //   If batchN:
        //      If layoutA is MatrixLayout::MajorK
        //         Logical shape is [B, divUpMul(M, tileM), K].
        //         Logical strides are [divUpMul(M, tileM) * K, K, 1].
        //      If layoutA is MatrixLayout::MajorMn
        //         Logical shape is [B, K, divUpMul(M, tileM)].
        //         Logical strides are [K * divUpMul(M, tileM), divUpMul(M, tileM), 1].
        //      If layoutA is MatrixLayout::BlockMajorK
        //         Logical shape is [B, K / blockK, divUpMul(M, tileM), blockK].
        //         Logical strides are [K * divUpMul(M, tileM), divUpMul(M, tileM) * blockK, blockK, 1].
        //         where blockK is 128B.
        void const* mPtrA{nullptr};

        // The block scaling factors to dequantize A.
        //
        // If (routeAct == true && batchM), the shape is [M, K / 16]
        // Else
        //   If DeepSeek FP8 recipe is used:
        //      If transposeMmaOutput is false, shape is [K / 128, M].
        //      Otherwise, shape is [M / 128, K / 128].
        //    The rightmost dimension is contiguous in memory.
        //
        //   If DeepSeek FP8 recipe is not used, but for MxFp{4,8} and NvFp4 formats:
        //      The layout of scaling factors for A is always R128c4
        //      M must be a multiple of 128.
        //      K must be a multiple of 64.
        //      The "logical" shape is: [paddedM, K / 16].
        //      The R128c4 layout is: [paddedM / 128, K / 16 / 4, 512].
        //      The shape we use for TMA is: [paddedM / 128, K / 16 / 4, 2, 256].
        //  Where paddedM is M if (routeAct == true && batchM), or
        //  sum(divUpMul(M[bi], tileM) for bi in B) if batchM,
        //  otherwise divUpMul(M, tileM) * B.
        //  Dtype is Dtype::Fp32 if DeepSeek FP8 recipe is used, otherwise Dtype::E4m3.
        //
        // Otherwise should be set to nullptr.
        void const* mPtrSfA{nullptr};

        // The per-token scaling factors from scale A.
        //
        // This is used for either:
        //   * Per-token scaling factor quantization schemes, such as MetaFP8. The dtype is
        //   Dtype::Float32
        //   * When the routing scales are applied to the input activations (only when output is not
        //   transposed). The dtype is Dtype::Bfloat16
        //
        // if (batchM (A is activations)):
        //     Logical shape is [sum(divUpMul(M[bi], tileM) for bi in B)]
        //
        // if (batchN (A is weights)):
        //     Logical shape is [B, divUpMul(M, tileM)]
        //
        void const* mPtrPerTokenSfA{nullptr};

        // The matrix B. The data type is controlled by options.mDtypeB.
        //
        // If (routeAct == true && batchN), the shape is [N, K]
        //
        // Else
        //   If batchN:
        //      Logical shape is [sum(divUpMul(N[bi], tileN) for bi in B), K].
        //      Logical strides are [K, 1].
        //
        //   If batchM:
        //      If layoutB is MatrixLayout::MajorK
        //         Logical shape is [B, divUpMul(N, tileN), K].
        //         Logical strides are [divUpMul(N, tileN) * K, K, 1].
        //      If layoutB is MatrixLayout::MajorMn
        //         Logical shape is [B, K, divUpMul(N, tileN)].
        //         Logical strides are [K * divUpMul(N, tileN), divUpMul(N, tileN), 1].
        //      If layoutB is MatrixLayout::BlockMajorK
        //         Logical shape is [B, K / blockK, divUpMul(N, tileN), blockK].
        //         Logical strides are [K * divUpMul(N, tileN), divUpMul(N, tileN) * blockK, blockK, 1].
        //         where blockK is 128B.
        void const* mPtrB{nullptr};

        // The scaling factors to dequantize B.
        //
        //
        //
        // Else
        //   If DeepSeek FP8 recipe is used:
        //      If transposeMmaOutput is false, shape is [paddedN / 128, K / 128].
        //      Otherwise, shape is [K / 128, paddedN].
        //      The rightmost dimension is contiguous in memory.
        //
        //   If DeepSeek FP8 recipe is not used, but for MxFp{4,8} and NvFp4 formats:
        //    If the layout is R128c4,
        //       paddedN must be a multiple of 128.
        //       K must be a multiple of 64.
        //       The R128c4 layout is: [paddedN / 128, K / 16 / 4, 512]
        //       The shape we use for TMA is: [paddedN / 128, K / 16 / 4, 2, 256]
        //
        //    If the layout is R8c4,
        //       paddedN must be a multiple of 8.
        //       K must be a multiple of 64.
        //       The R8c4 layout is: [paddedN / 8, K / 16 / 4, 32]
        //       The shape we use for TMA is: [paddedN / 8, K / 16 / 4 / repeats, repeats * 32]
        //       where repeats = min(tileK / 16 / 4, 8)
        //
        // where paddedN is N if (routeAct == true && batchN),
        // or sum(divUpMul(N[bi], tileN) for bi in B) if batchN,
        // otherwise divUpMul(N, TileN) * B.
        //
        // Dtype is Dtype::Fp32 if DeepSeek FP8 recipe is used, otherwise Dtype::E4m3.
        //
        // Otherwise should be set to nullptr.
        void const* mPtrSfB{nullptr};

        // The per-token scaling factors from scale B.
        //
        // This is used for either:
        //   * Per-token scaling factor quantization schemes, such as MetaFP8. The dtype is
        //   Dtype::Float32
        //   * When the routing scales are applied to the input activations (only when output is
        //   transposed). The dtype is Dtype::Bfloat16
        //
        // if (batchM (B is weights)):
        //     Logical shape is [B, divUpMul(N, tileN)]
        //
        // if (batchN (B is activations)):
        //     Logical shape is [sum(divUpMul(N[bi], tileN) for bi in B)]
        void const* mPtrPerTokenSfB{nullptr};

        // The bias applied after the GEMM and before the activation function.
        // The bias is applied before applying the global scaling factor. I.e.
        // C = act(A * B + bias') * scaleC
        // scaleC = dequantA * dequantB * quantC
        // Thus, the bias' = bias / (dequantA * dequantB), where the bias is the original bias.
        //
        // If batchM, BiasType must be N, and bias shape is [B, N].
        // The bias is broadcasted along the M dimension.
        //
        // If batchN BiasType must be M, and bias shape is [B, M].
        // The bias is broadcasted along the N dimension.
        //
        // The dtype is float32.
        void const* mPtrBias{nullptr};

        // The output tensor scaling factor for Fp8 (not DeepSeek FP8) and NvFp4 quantization.
        // TensorRT-LLM API requires a scaling factor on the device.
        // scaleC = dequantA * dequantB * quantC,
        // where dequantA is global dequantization scaling factor of A
        //    if dtypeA is FP8, it transforms the range from [-448, 448] to [-amaxA, amaxA]
        //    if dtypeA is NvFp4, it transforms the range from [-448 * 6, 448 * 6] to [-amaxA, amaxA],
        //    otherwise it is 1.
        // dequantB is defined similarly to dequantA.
        // quantC is the quantization scaling factor of C.
        //    if dtypeC is FP8, it transforms the range from [-amaxC, amaxC] to [-448, 448]
        //    if dtypeC is NvFp4, it transforms the range from [-amaxC, amaxC] to [-448 * 6, 448 * 6],
        //    otherwise it is 1.
        // Shape is [B].
        float const* mPtrScaleC{nullptr};

        // The output gate scale for Fp8 (not DeepSeek FP8) and NvFp4 quantization.
        // TensorRT-LLM API requires a scaling factor on the device.
        // scaleGate = dequantA * dequantB,
        // where dequantA is global dequantization scaling factor of A
        //    if dtypeA is FP8, it transforms the range from [-448, 448] to [-amaxA, amaxA]
        //    if dtypeA is NvFp4, it transforms the range from [-448 * 6, 448 * 6] to [-amaxA, amaxA],
        //    otherwise it is 1.
        // dequantB is defined similarly to dequantA.
        // Shape is [B].
        float const* mPtrScaleGate{nullptr};

        // The clamp limit for the accumulator before applying the activation.
        // Shape is [B].
        // Clamp is INF if nullptr.
        // When the input is FP8 or NVFP4, the clamp has to be scaled by limit' = limit / dequantAb.
        // If applied on SwiGlu, it will be:
        //
        //   x_glu    = x_glu.clamp(min=None, max=limit)
        //   x_linear = x_linear.clamp(min=-limit, max=limit)
        //
        // The given clamp limit applies to the dequantized values, so the order of operations would
        // look something like this:
        //
        // x0 = x0 * dqAb
        // x0 = clamp(x0, none, limit)
        // x0 = x0 * sigmoid(alpha * x0)
        // x1 = dqAb * x1
        // x1 = clamp(x1, -limit, limit)
        // out = qC * (x1 + beta) * x0
        //
        // Given that the dqAb and qC are combined into scaleC, we can bring the dqAb into the clamp
        // limit and apply the clamping prior to dequantization:
        //
        // x0 = clamp(x0, none, limit / dqAb)
        // x0 = x0 * dqAb
        // x0 = x0 * sigmoid(alpha * x0)
        // x1 = clamp(x1, -limit / dqAb, limit / dqAb)
        // scaleC = dqAb * qC
        // beta' = beta / dqAb
        // out = scaleC * (x1 + beta') * x0
        //
        // Note this assumes that dequantScaleAb == scaleGate which is true in TRT-LLM MoE use-case
        //
        float const* mPtrClampLimit{nullptr};

        // The alpha and beta for SwiGlu or GeGlu.
        // gatedActivation <- (x0 + beta) * activation(x1, alpha)
        // Shape is [B].
        // Alpha is 1.f if nullptr.
        // Beta is 0.f if nullptr.
        // The formula for SwiGlu (for GeGlu, replace sigmoid with phi):
        //
        //   out_glu  = x_glu * torch.sigmoid(alpha * x_glu) * (x_linear + beta)
        //
        // The beta is added before applying the global scaling factor. I.e.
        // x_linear = (x_linear + beta') * scaleC
        // Thus, the beta' = beta / (dequantA * dequantB), where the beta is the original beta.
        float const* mPtrGatedActAlpha{nullptr};
        float const* mPtrGatedActBeta{nullptr};

        // Param is used when the kernel is configured with -routeAct true.
        // The inputs are not padded, but the outputs are padded to divUpMul(M[bi], tileM) for batchM or
        // divUpMul(N[bi], tileN) for batchN.
        // If -routeAct is false, the params are not used and should be set to zero.

        // The routeMap for the input tokens.
        // Map of expanded token index (counting the previous padded tokens) to the batch index
        // the token belongs to.
        // The shape is
        // [divUpMul(numTokens + numBatches * (tileM/N - 1), tileM/N)]
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
        int32_t const* mPtrRouteMap;

        //////////////////////////////////////////////////////////////////////////////////////////////////
        //
        // Batching information parameters.
        //
        //////////////////////////////////////////////////////////////////////////////////////////////////

        // In some cases, some CTAs must early-exit. E.g. when the grid size is set statically, but the
        // actual workload is decided at runtime. This element on the device contains the number of CTAs
        // that do not early-exit. The number corresponds to the X dim of the grid when the output is
        // not transposed (i.e. batchM). To the Y dim, otherwise. The size is 1 and the dtype is
        // int32_t. Used if isStaticBatch == false, otherwise set to nullptr. The pointer points to a
        // scalar and the dtype is int32_t. The pointed value must be >= 0.
        int32_t const* mPtrNumNonExitingCtas;

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
        int32_t const* mPtrTotalNumPaddedTokens;

        // Pointer to the map from the CTA index (in X/Y dim) to the batch index.
        // Maps CTA index in batch dim (i.e. blockDim.x if batchM, otherwise blockDim.y)
        // to batch index.
        // E.g. with listM = 128,255,32 and tileM = 128, should be equal to
        // ctaIdxXyToBatchIdx = [0, 1, 1, 2]
        // If isStaticBatch == true, ptrCtaIdxXyToBatchIdx should be set to nullptr and
        // ctaIdxXyToBatchIdx is used.
        // The shape is
        // [divUp(numTokens + numBatches * (tileM/N - 1), tileM/N)]
        int32_t const* mPtrCtaIdxXyToBatchIdx;

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
        // The shape is
        // [divUp(numTokens + numBatches * (tileM/N - 1), tileM/N)]
        int32_t const* mPtrCtaIdxXyToMnLimit;
    };

    struct OutputBuffers
    {
        // The output matrix C. The data type is controlled by options.mDtypeC.
        //
        // If batchM:
        //    Logical shape is [sum(divUpMul(M[bi], tileM) for bi in B), N].
        //    Logical strides are [N, 1].
        //
        // If batchN:
        //    Logical shape is [sum(divUpMul(N[bi], tileN) for bi in B), M].
        //    Logical strides are [M, 1].
        void* mPtrC{nullptr};

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
        void* mPtrSfC{nullptr};
    };

    ProblemDimensions mProblemDimensions;
    InputBuffers mInputBuffers;
    OutputBuffers mOutputBuffers;
};

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// BatchedGemmInterface
//
////////////////////////////////////////////////////////////////////////////////////////////////////

class BatchedGemmInterface
{
public:
    using ModuleCache = std::unordered_map<std::string, std::tuple<CUmodule, CUfunction>>;

    BatchedGemmInterface() {}

    // Launch the cubin from the provided config. It calls all necessary memsets for internal buffers.
    // Provided config must be validated with isValidConfig before the call.
    int32_t run(BatchedGemmConfig const& config, void* workspace, BatchedGemmData const& options, void* cudaStream,
        int32_t multiProcessorCount, bool usePdl = true,
        std::optional<std::reference_wrapper<ModuleCache>> moduleCache = std::nullopt);

    // Initializes the buffers before the world sync. Must be called before run.
    int32_t runInitBeforeWorldSync(
        BatchedGemmConfig const& /* config */, BatchedGemmData const& /* data */, void* /* cudaStream */) const
    {
        return 0;
    };

    size_t getWorkspaceSizeInBytes(BatchedGemmConfig const& /* config */, BatchedGemmData const& /* data */) const;

    // Returns the list of all available cubin configurations
    BatchedGemmConfig const* getBatchedGemmConfigs() const;

    // Returns the number of available cubin configurations
    size_t getNumBatchedGemmConfigs() const;

    // Returns the grid dimensions of the current kernel.
    std::tuple<int32_t, int32_t, int32_t> getGridDim(
        BatchedGemmOptions const& options, std::optional<int32_t> maxNumCtasInBatchDim = std::nullopt) const
    {
        bool const batchM = options.mBatchMode == BatchedGemmOptions::BatchMode::BatchM;

        int32_t numCtasBatch{0};
        // For normal BMM, mNumTokens == 0 and the number of CTAs is known to host.
        if (options.mIsStaticBatch)
        {
            for (int32_t bi = 0; bi < options.mNumBatches; ++bi)
            {
                numCtasBatch += batchM ? gemm::divUp(options.mBatchedM[bi], options.mTileM)
                                       : gemm::divUp(options.mBatchedN[bi], options.mTileN);
            }
        }
        // For MoE, mNumTokens != 0 and the number of CTAs is known only at runtime.
        // We launch maximally possible number of CTAs and use ptrNumNonExitingCtas to determine the
        // actual number of CTAs to run.
        else if ((options.mEnablesEarlyExit || options.mEnablesDelayedEarlyExit) && options.mNumTokens != 0)
        {
            assert(maxNumCtasInBatchDim.has_value()
                && "maxNumCtasInBatchDim must be provided when options.mNumTokens != 0");
            numCtasBatch = maxNumCtasInBatchDim.value();
        }
        else
        {
            throw std::invalid_argument("Invalid combination of options");
        }

        if (batchM)
        {
            numCtasBatch = gemm::divUpMul(numCtasBatch, options.mClusterDimX);
        }
        else
        {
            numCtasBatch = gemm::divUpMul(numCtasBatch, options.mClusterDimY);
        }

        int32_t numCtasTile
            = batchM ? gemm::divUp(options.mN, options.mTileN) : gemm::divUp(options.mM, options.mTileM);
        if (batchM)
        {
            numCtasTile = gemm::divUpMul(numCtasTile, options.mClusterDimY);
        }
        else
        {
            numCtasTile = gemm::divUpMul(numCtasTile, options.mClusterDimX);
        }
        int32_t const numCtasInner = options.mNumSlicesForSplitK;
        return std::make_tuple(numCtasBatch, numCtasTile, numCtasInner);
    }

    // Creates GemmOptions from kernel and data.
    BatchedGemmOptions getOptionsFromConfigAndData(BatchedGemmConfig const& config, BatchedGemmData const& data) const;

    // Returns the number of CTAs of the current kernel.
    int32_t getNumCtas(
        BatchedGemmOptions const& options, std::optional<int32_t> maxNumCtasInBatchDim = std::nullopt) const
    {
        auto [numCtasBatch, numCtasTile, numCtasInner] = getGridDim(options, maxNumCtasInBatchDim);
        return numCtasBatch * numCtasTile * numCtasInner;
    }

    // Returns true if the configuration of the cubin can be executed for the given params.
    bool isValidConfig(BatchedGemmConfig const& config, BatchedGemmData const& data) const;

private:
    // Aligns the pointer to the alignment
    template <typename Dtype>
    inline Dtype* alignPtr(Dtype* ptr, int64_t alignment) const;

    // Returns the size of the workspace buffers in bytes
    std::vector<size_t> getWorkspaceSizesInBytes(BatchedGemmConfig const& config, BatchedGemmData const& data) const;

    // Returns the size padded to the alignment
    size_t getSizePaddedToAlignment(size_t size, size_t alignment) const;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Dtype>
inline Dtype* BatchedGemmInterface::alignPtr(Dtype* ptr, int64_t alignment) const
{
    assert((alignment & (alignment - 1)) == 0 && "Alignment must be a power of 2");
    return reinterpret_cast<Dtype*>((reinterpret_cast<uintptr_t>(ptr) + alignment - 1) & ~(alignment - 1));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

BatchedGemmConfig const* BatchedGemmInterface::getBatchedGemmConfigs() const
{
#ifdef TLLM_GEN_EXPORT_INTERFACE
    return tensorrt_llm::kernels::tllmGenBatchedGemmList;
#else
    return nullptr;
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////

size_t BatchedGemmInterface::getNumBatchedGemmConfigs() const
{
#ifdef TLLM_GEN_EXPORT_INTERFACE
    return tensorrt_llm::kernels::tllmGenBatchedGemmListLen;
#else
    return 0;
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////

BatchedGemmOptions BatchedGemmInterface::getOptionsFromConfigAndData(
    BatchedGemmConfig const& config, BatchedGemmData const& data) const
{
    // Create options from config and data.
    BatchedGemmOptions options;
    options = config.mOptions;
    options.mM = data.mProblemDimensions.mM;
    options.mN = data.mProblemDimensions.mN;
    options.mK = data.mProblemDimensions.mK;
    options.mBatchedM = data.mProblemDimensions.mBatchedM;
    options.mBatchedN = data.mProblemDimensions.mBatchedN;
    options.mBatchMode = data.mProblemDimensions.mBatchM ? BatchedGemmOptions::BatchMode::BatchM
                                                         : BatchedGemmOptions::BatchMode::BatchN;
    options.mNumBatches = data.mProblemDimensions.mNumBatches;
    options.mNumTokens = data.mProblemDimensions.mNumTokens;
    return options;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool BatchedGemmInterface::isValidConfig(BatchedGemmConfig const& config, BatchedGemmData const& data) const
{
    // Get options from config and data.
    auto options = getOptionsFromConfigAndData(config, data);

    // Is Blackwell?
    bool isBlackwell = gemm::isSmVersionBlackwell(config.mSm);

    // Check options without modifications.
    return checkAndUpdateBatchedGemmOptions(options, isBlackwell,
        /* updateOptions */ false);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

size_t BatchedGemmInterface::getSizePaddedToAlignment(size_t size, size_t alignment) const
{
    assert((alignment & (alignment - 1)) == 0);
    return (size + alignment - 1) & ~(alignment - 1);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

size_t BatchedGemmInterface::getWorkspaceSizeInBytes(BatchedGemmConfig const& config, BatchedGemmData const& data) const
{
    auto workspaceSizes = getWorkspaceSizesInBytes(config, data);
    auto size = std::accumulate(workspaceSizes.begin(), workspaceSizes.end(), 0);
    // Additional 1023 bytes to align the pointer to 1024
    return size > 0 ? size + 1023 : 0;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::vector<size_t> BatchedGemmInterface::getWorkspaceSizesInBytes(
    BatchedGemmConfig const& config, BatchedGemmData const& data) const
{

    std::vector<size_t> workspaceSizes;

    // Get options from config and data.
    auto options = getOptionsFromConfigAndData(config, data);

    if (options.mUseDeepSeekFp8 && options.mFusedAct)
    {
        int32_t totalNumPaddedTokens = 0;
        auto const batchM = options.mBatchMode == BatchedGemmOptions::BatchMode::BatchM;
        if (!options.mEnablesEarlyExit || options.mNumTokens == 0)
        {
            for (int32_t bi = 0; bi < options.mNumBatches; ++bi)
            {
                totalNumPaddedTokens += batchM ? gemm::divUpMul(options.mBatchedM[bi], options.mTileM)
                                               : gemm::divUpMul(options.mBatchedN[bi], options.mTileN);
            }
        }
        else
        {
            // Get tile in token dim.
            auto tileTokensDim = batchM ? options.mTileM : options.mTileN;
            totalNumPaddedTokens = data.mProblemDimensions.mMaxNumCtasInTokenDim * tileTokensDim;
        }

        // Get options from config.
        auto& options = config.mOptions;

        int const tokenTile = batchM ? options.mTileM : options.mTileN;

        auto const numTokens = totalNumPaddedTokens;
        auto const intermediateDim = batchM ? options.mN : options.mM;
        auto const intermediateTile = batchM ? options.mTileN : options.mTileM;

        auto const numBytesRowMax = intermediateDim * totalNumPaddedTokens / 128 * sizeof(float);

        auto const numTilesToken = numTokens / tokenTile;
        auto const numTilesInt = intermediateDim / intermediateTile;
        auto const numBytesRowMaxBars = numTilesToken * numTilesInt / 2 * sizeof(uint32_t);

        // TODO: do we need to pad to 1024?
        workspaceSizes.push_back(getSizePaddedToAlignment(numBytesRowMax, 1024));
        workspaceSizes.push_back(getSizePaddedToAlignment(numBytesRowMaxBars, 1024));
    }

    return workspaceSizes;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
int32_t BatchedGemmInterface::run(BatchedGemmConfig const& config, void* workspace,
    BatchedGemmData const& batchedGemmData, void* cudaStream, int32_t /* multiProcessorCount */, bool usePdl,
    std::optional<std::reference_wrapper<ModuleCache>> moduleCache)
{
    // Might be used.
    (void) usePdl;
    (void) moduleCache;
    // Get options from config and data.
    auto options = getOptionsFromConfigAndData(config, batchedGemmData);

    bool const batchM = options.mBatchMode == BatchedGemmOptions::BatchMode::BatchM;
    bool const useDeepSeekFp8
        = options.mUseDeepSeekFp8 && options.mDtypeA == tg::Dtype::E4m3 && options.mDtypeB == tg::Dtype::E4m3;

    auto workspaceSizes = getWorkspaceSizesInBytes(config, batchedGemmData);
    float* dPtrRowMax{nullptr};
    uint32_t* dPtrRowMaxBars{nullptr};

    // Set the completion barriers to 0 if needed.
    if (useDeepSeekFp8 && options.mFusedAct)
    {
        dPtrRowMax = reinterpret_cast<float*>(alignPtr(reinterpret_cast<char*>(workspace), 1024));
        dPtrRowMaxBars
            = reinterpret_cast<uint32_t*>(alignPtr(reinterpret_cast<char*>(dPtrRowMax) + workspaceSizes[0], 1024));
        auto err = cudaMemsetAsync(
            (void*) dPtrRowMaxBars, 0x00, workspaceSizes[1], reinterpret_cast<cudaStream_t>(cudaStream));
        if (err != cudaSuccess)
        {
            return 1;
        }
    }

    auto [numCtaBatch, numCtaTile, numCtaInner]
        = getGridDim(options, batchedGemmData.mProblemDimensions.mMaxNumCtasInTokenDim);
    auto kernelParams = KernelParamsSetup::setKernelParams(options, batchM, batchedGemmData.mInputBuffers.mPtrA,
        batchedGemmData.mInputBuffers.mPtrB, batchedGemmData.mOutputBuffers.mPtrC,
        batchedGemmData.mInputBuffers.mPtrSfA, batchedGemmData.mInputBuffers.mPtrSfB,
        batchedGemmData.mInputBuffers.mPtrPerTokenSfA, batchedGemmData.mInputBuffers.mPtrPerTokenSfB,
        batchedGemmData.mInputBuffers.mPtrBias, batchedGemmData.mOutputBuffers.mPtrSfC,
        batchedGemmData.mInputBuffers.mPtrScaleC, batchedGemmData.mInputBuffers.mPtrScaleGate,
        batchedGemmData.mInputBuffers.mPtrClampLimit, batchedGemmData.mInputBuffers.mPtrGatedActAlpha,
        batchedGemmData.mInputBuffers.mPtrGatedActBeta, batchedGemmData.mInputBuffers.mPtrRouteMap, dPtrRowMax,
        dPtrRowMaxBars, batchedGemmData.mInputBuffers.mPtrNumNonExitingCtas,
        batchedGemmData.mInputBuffers.mPtrTotalNumPaddedTokens, batchedGemmData.mInputBuffers.mPtrCtaIdxXyToBatchIdx,
        batchedGemmData.mInputBuffers.mPtrCtaIdxXyToMnLimit, numCtaBatch);

    // The size of the grid.
    std::vector<int32_t> grid = batchM ? std::vector<int32_t>{numCtaBatch, numCtaTile, numCtaInner}
                                       : std::vector<int32_t>{numCtaTile, numCtaBatch, numCtaInner};

#ifdef TLLM_GEN_EXPORT_INTERFACE
    CUmodule cuModule;
    CUfunction cuFunction;

    if (moduleCache.has_value())
    {
        ModuleCache& moduleCacheRef = moduleCache.value().get();

        // Modules are associated with a specific context, so the context is included in the key
        CUcontext ctx;
        unsigned long long ctxId;
        cuCtxGetCurrent(&ctx);
        cuCtxGetId(ctx, &ctxId);

        // Reinterpret the ctxId as a string to avoid needing a custom hash or converting it to a
        // string in decimal representation.
        std::string const ctxName
            = std::string(reinterpret_cast<char*>(&ctxId), sizeof(unsigned long long) / sizeof(char));
        std::string const funcName = std::string(config.mFunctionName);
        auto const moduleKey = ctxName + funcName;
        auto module = moduleCacheRef.find(moduleKey);

        // Use cache if module is found, otherwise load and insert into cache
        if (module != moduleCacheRef.end())
        {
            cuFunction = std::get<1>(module->second);
        }
        else
        {
            cuModuleLoadData(&cuModule, config.mData);
            cuModuleGetFunction(&cuFunction, cuModule, config.mFunctionName);
            moduleCacheRef.insert(std::make_pair(moduleKey, std::make_tuple(cuModule, cuFunction)));
        }
    }
    else
    {
        cuModuleLoadData(&cuModule, config.mData);
        cuModuleGetFunction(&cuFunction, cuModule, config.mFunctionName);
    }

    // Prepare the grid/block.
    dim3 block3{static_cast<uint32_t>(config.mNumThreadsPerCTA), static_cast<uint32_t>(1), static_cast<uint32_t>(1)};
    dim3 grid3{(grid.size() > 0 ? static_cast<uint32_t>(grid[0]) : 1u),
        (grid.size() > 1 ? static_cast<uint32_t>(grid[1]) : 1u),
        (grid.size() > 2 ? static_cast<uint32_t>(grid[2]) : 1u)};
    // Prepare the cluster size.
    dim3 cluster3{static_cast<uint32_t>(options.mClusterDimX), static_cast<uint32_t>(options.mClusterDimY),
        static_cast<uint32_t>(options.mClusterDimZ)};

    // Run the kernel.
    auto result = trtllm::gen::launchKernel((void*) &kernelParams, cudaStream, config.mSharedMemSize, cuFunction,
        block3, grid3, cluster3,
        usePdl
            && (config.mOptions.mGridWaitForPrimaryEarlyExit | config.mOptions.mGridWaitForPrimaryA
                | config.mOptions.mGridWaitForPrimaryB));
    if (result != CUDA_SUCCESS)
    {
        return -1;
    }
    // If a module cache has not been given, unload the module to avoid leaking
    if (!moduleCache.has_value())
    {
        cuModuleUnload(cuModule);
    }
#else
    config.mCudaRunner->run((void*) &kernelParams, (void*) cudaStream, grid,
        /* cluster */ {},
        /* instanceId */ config.mInstanceIdx);
#endif

    return 0;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace batchedGemm

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace batchedGemm
