/*
 * Copyright (c) 2020-2026, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "fmhaReduction.h"
#include "fmhaRunnerParams.h"
#include "kernelUtils.h"
#include "tensorrt_llm/common/config.h"
#include "tensorrt_llm/common/envUtils.h"
#include <cuda_runtime_api.h>
#include <float.h>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

////////////////////////////////////////////////////////////////////////////////////////////////////

#define NumThreadsPerCta 512

// KvSplit > 1: the CTA's threads form KvSplit groups that each reduce every KvSplit-th partial (ctaIdxKv) of the same
// rows, then group 0 merges the KvSplit (max, sum, O) partial results through shared memory. Each CTA covers KvSplit
// times fewer rows, so the grid has KvSplit times more CTAs and each thread walks KvSplit times fewer partials. With
// KvSplit == 1 every thread walks all partials of its row.
template <int32_t TileSizePerCtaQ, int32_t HeadDimPerCta, typename DtypeO, typename DtypePartialO, int32_t KvSplit>
__global__ void __launch_bounds__(NumThreadsPerCta, 2) fmhaReductionKernel(fmha::KernelParams const params,
    bool isTokenSparse, bool groupsTokensHeadsQ, bool supportsVarSparseMlaTopKLens, float softmaxPScale,
    int32_t numCtasForReduction, int32_t numCtasForAllHeads, int32_t headDimV, int32_t numHeadDimCtasV)
{

    // clang-format off
  // The shape of partialO buffer: [batchSize, numHeadCtas, numCtasQ, numCtasKv, TileSizePerCtaQ, headDimPerCta].
  // The shape of final O buffer: [batchSize, numCtasQ, numHeadsQ, headDim].
  // The shape of attentionSinks buffer: [numHeadsQ].
  // The shape of partialStats buffer: [batchSize, numHeadCtas, numCtasQ, numCtasKv, TileSizePerCtaQ], where each element is a float2 (max/sum).
  // The shape of softmaxStats buffer: [batchSize, numCtasQ, numHeadsQ], where each element is a float2 (max/sum).
  // Note that numValidRows includes both numValidTokens and numHeadsQPerKv if grouping headsQ.
    // clang-format on

    // The batchIdx.
    int32_t const batchIdx{static_cast<int32_t>(blockIdx.z)};
    // The headCtaIdxO.
    int32_t const headCtaIdxO{static_cast<int32_t>(blockIdx.y)};
    // The headDimCtaIdxV.
    int32_t const headDimCtaIdxV{static_cast<int32_t>(blockIdx.y % numHeadDimCtasV)};
    // The headGrpIdxO.
    int32_t const headGrpIdxO{static_cast<int32_t>(blockIdx.y / numHeadDimCtasV)};
    // The ctaIdxQ.
    int32_t const ctaIdxQ{static_cast<int32_t>(blockIdx.x % params.mMaxNumCtasQ)};
    // The ctaIdx for the reduction work.
    int32_t const ctaIdxForReduction{static_cast<int32_t>(blockIdx.x / params.mMaxNumCtasQ)};
    // The numHeadsQPerKvCta.
    int32_t const numHeadsQPerKvCta{min(params.mNumHeadsQPerKv, TileSizePerCtaQ)};
    // The headIdxO.
    int32_t const headIdxO{headGrpIdxO * numHeadsQPerKvCta};
    // The threads per KV group, the KV group of this thread and its index inside the group.
    int32_t constexpr NumThreadsPerGrp{NumThreadsPerCta / KvSplit};
    int32_t const kvGrpIdx{static_cast<int32_t>(threadIdx.x) / NumThreadsPerGrp};
    // The warpGrpThreadIdx.
    int32_t const warpGrpThreadIdx{static_cast<int32_t>(threadIdx.x) % NumThreadsPerGrp};

    // The seqOffsetQ.
    int32_t const seqOffsetQ{
        params.ptrCumSeqLensQ == nullptr ? batchIdx * params.mMaxNumCtasQ : params.ptrCumSeqLensQ[batchIdx]};
    // The seqLenQ.
    int32_t const seqLenQ{
        params.ptrCumSeqLensQ == nullptr ? params.mMaxSeqLenQ : (params.ptrCumSeqLensQ[batchIdx + 1] - seqOffsetQ)};
    // The number of validTokens.
    int32_t const numValidTokens{min(seqLenQ - ctaIdxQ * params.mNumTokensPerCtaQ, params.mNumTokensPerCtaQ)};
    // The number of validRows.
    int32_t const numValidRows{numValidTokens * numHeadsQPerKvCta};
    // Early exit if there are no valid tokens.
    if (numValidTokens <= 0)
    {
        return;
    }

    // The actual number of seqLenKv.
    int32_t seqLenKv{params.ptrSeqLensKv[batchIdx]};
    // Consider the causal-mask speculative decoding.
    seqLenKv = seqLenKv - ((params.mMaxSeqLenQ - 1) - ctaIdxQ);
    // Consider sparseAttnTopK and variable sparse MLA topK lengths.
    if (supportsVarSparseMlaTopKLens)
    {
        seqLenKv = params.ptrSparseMlaTopKLens[seqOffsetQ + ctaIdxQ * params.mNumTokensPerCtaQ];
    }
    else if (isTokenSparse)
    {
        seqLenKv = min(seqLenKv, params.mSparseAttnTopK);
    }
    // The actual number of CtasKv (TileSizeKv is always 128 for now).
    int32_t numCtasKv{min((seqLenKv + 127) / 128, params.mMaxNumCtasKv)};

    // The tileIdx in the batch/head dimension.
    int64_t const batchHeadTileIdx{
        ((batchIdx * static_cast<int32_t>(gridDim.y) + headCtaIdxO) * params.mMaxNumCtasQ + ctaIdxQ)};

    // The offset of the partialStats buffer.
    int64_t const partialStatsOffset{batchHeadTileIdx * params.mMaxNumCtasKv * TileSizePerCtaQ};
    // The offset of the partialO buffer.
    int64_t const partialOOffset{partialStatsOffset * HeadDimPerCta};
    // The offset of the softmaxStats buffer.
    int64_t const softmaxStatsOffset{
        ((seqOffsetQ + ctaIdxQ * params.mNumTokensPerCtaQ) * numCtasForAllHeads + headGrpIdxO) * numHeadsQPerKvCta};
    // The offset of the O buffer.
    int64_t const oOffset{softmaxStatsOffset * headDimV + headDimCtaIdxV * HeadDimPerCta};

    // The partialStats pointer.
    float2* partialStatsPtr = reinterpret_cast<float2*>(params.ptrPartialStats) + partialStatsOffset;
    // The partialO pointer.
    DtypePartialO* partialOPtr = reinterpret_cast<DtypePartialO*>(params.ptrPartialO) + partialOOffset;
    // The softmaxStats pointer.
    float2* softmaxStatsPtr = reinterpret_cast<float2*>(params.ptrSoftmaxStats) + softmaxStatsOffset;
    // The O pointer.
    DtypeO* oPtr = reinterpret_cast<DtypeO*>(params.ptrO) + oOffset;
    // The attentionSinks pointer.
    float const* attentionSinksPtr
        = params.ptrAttentionSinks == nullptr ? nullptr : params.ptrAttentionSinks + headIdxO;

    // Whether to store the softmax stats.
    bool const storesSoftmaxStats{params.ptrSoftmaxStats != nullptr};

    // The softmaxScaleLog2. The maxima stored in partialStats are raw BMM1 outputs (see the
    // softmaxStats export below, which scales maxVal on the way out), so the correction factors
    // computed by the reduction must use exactly the softmax scale that the primary kernel applied.
    // On FP8 paths that scale carries the BMM1 dequant product (dequantQ * dequantKv) and lives on
    // the device only: ptrScaleSoftmaxLog2 points at bmm1_scale[kIdxScaleSoftmaxLog2Ptr], whereas
    // the host mirror mScaleSoftmaxLog2 never saw the dequant scales. Reading the host mirror is
    // therefore correct only when dequantQ * dequantKv == 1, and silently raises the per-CtaKv
    // weights to that product otherwise. Prefer the device scale whenever it is available; it is
    // written before the producer kernel triggers its programmatic launch completion, so it is
    // already visible here even though this read precedes cudaGridDependencySynchronize below.
    float const softmaxScaleLog2
        = params.ptrScaleSoftmaxLog2 != nullptr ? *params.ptrScaleSoftmaxLog2 : params.mScaleSoftmaxLog2;

    int32_t constexpr NumBytesPerPartialElt{sizeof(DtypePartialO)};
    static_assert(NumBytesPerPartialElt == 2, "The data type of partialO should be either fp16 or bf16.");

    // The threads in the warp-group should load different values from one partial output
    // [numValidRows, headDim], and then iterate over partial outputs from different CTAs.
    int32_t constexpr NumEltsPer16BVec{16 / NumBytesPerPartialElt};
    static_assert((HeadDimPerCta * NumBytesPerPartialElt) % 16 == 0, "Not implemented");

    // The number of unrolled iterations to issue multiple LDGs.
    int32_t constexpr UnrollSize{4};

    // The number of processed rows in one slice where each CTA will process one slice.
    int32_t constexpr NumBytesPerHeadDim{HeadDimPerCta * NumBytesPerPartialElt};
    int32_t constexpr NumBytePerSlice{NumThreadsPerGrp * 16};
    static_assert(NumBytePerSlice % NumBytesPerHeadDim == 0, "Not implemented");
    int32_t constexpr NumRowsPerSlice{NumBytePerSlice / NumBytesPerHeadDim};
    // The actual number of tensor slices for the reduction.
    int32_t numSlices{(numValidRows + NumRowsPerSlice - 1) / NumRowsPerSlice};

    // The number of slices that each CTA will process.
    int32_t numSlicesPerCta{(numSlices + numCtasForReduction - 1) / numCtasForReduction};
    // The start slice index for the current CTA.
    int32_t startSliceIdx{ctaIdxForReduction * numSlicesPerCta};
    // The end slice index for the current CTA.
    int32_t endSliceIdx{min(startSliceIdx + numSlicesPerCta, numSlices)};

    // The total number of rows in the partial buffers.
    int32_t numRowsInPartialBuffers{TileSizePerCtaQ};

    // Per-thread (max, sum, O) of every KV group, merged by group 0.
    __shared__ float smemStats[KvSplit > 1 ? 2 * NumThreadsPerCta : 1];
    __shared__ float4 smemO[KvSplit > 1 ? 2 * NumThreadsPerCta : 1];

    // Iterate over different slices.
    // Split the reduction work across multiple CtasKv to reduce the latency.
    for (int32_t sliceIdx = startSliceIdx; sliceIdx < endSliceIdx; ++sliceIdx)
    {

        // The base offset that each thread points to.
        int32_t const baseOffset{warpGrpThreadIdx * NumEltsPer16BVec};
        // The index in the row dimension.
        int32_t const rowIdx{sliceIdx * NumRowsPerSlice + (baseOffset / HeadDimPerCta)};
        // Does this thread point to a valid row ?
        bool const isValidRow{rowIdx < numValidRows};
        int32_t validRowIdx{min(rowIdx, numValidRows - 1)};
        int32_t loadRowIdx{validRowIdx};
        // The index in the headDim dimension.
        int32_t const headDimIdx{baseOffset % HeadDimPerCta};
        // The memory load offset.
        int64_t const destMemOffset{loadRowIdx * HeadDimPerCta + headDimIdx};
        // The memory store offset.
        int64_t gmemStoreOffset{validRowIdx * headDimV + headDimIdx};
        // The local headIdxO.
        int32_t localHeadIdxO{validRowIdx};
        // The rowIdx of softmaxStats.
        int32_t softmaxStatsRowIdx{validRowIdx};
        // If grouping both tokens and headsQ, map validRowIdx into tokenIdx and headIdxInGrp.
        if (groupsTokensHeadsQ)
        {
            int32_t tokenIdx{validRowIdx / params.mNumHeadsQPerKv};
            int32_t headIdxInGrp{validRowIdx % params.mNumHeadsQPerKv};
            localHeadIdxO = headIdxInGrp;
            softmaxStatsRowIdx = tokenIdx * params.mNumHeadsQ + headIdxInGrp;
            gmemStoreOffset = int64_t(softmaxStatsRowIdx) * headDimV + headDimIdx;
        }

// Wait for the primary kernel to complete.
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
        cudaGridDependencySynchronize();
#endif

        // Add offset to the pointers.
        float2* localPartialStatsPtr = partialStatsPtr + loadRowIdx;
        DtypePartialO* localPartialOPtr = partialOPtr + destMemOffset;

        // Reduce max, sum and partialO vectors from different CtasKv.
        float sumVal{0.f};
        float oldMaxVal{-FLT_MAX}, maxVal{-FLT_MAX};
        float outputVals[NumEltsPer16BVec] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
        for (int32_t ii = kvGrpIdx; ii < numCtasKv; ii += UnrollSize * KvSplit)
        {
            // The partialStats array and partialO array.
            float2 partialStatsArray[UnrollSize];
            uint4 partialOArray[UnrollSize];
#pragma unroll
            for (int32_t jj = 0; jj < UnrollSize; ++jj)
            {
                // An out-of-range slot re-reads a partial this group has already merged (ii), so the running max is
                // unchanged.
                int32_t ctaIdxKv = KvSplit == 1 ? min(ii + jj, numCtasKv - 1)
                                                : (ii + jj * KvSplit < numCtasKv ? ii + jj * KvSplit : ii);
                partialStatsArray[jj] = localPartialStatsPtr[ctaIdxKv * numRowsInPartialBuffers];
                partialOArray[jj] = *reinterpret_cast<uint4 const*>(
                    localPartialOPtr + ctaIdxKv * numRowsInPartialBuffers * HeadDimPerCta);
            }
#pragma unroll
            for (int32_t jj = 0; jj < UnrollSize; ++jj)
            {
                // Whether the ctaIdxKv is valid.
                bool const isValidCtaIdxKv = (ii + jj * KvSplit) < numCtasKv;
                // The local max and sum values.
                auto partialStats = partialStatsArray[jj];
                float localMax = partialStats.x;
                float localSum = partialStats.y;
                // Update the max value.
                maxVal = fmaxf(maxVal, localMax);
                // Compute the correction scales.
                float corrScale0 = isValidCtaIdxKv ? exp2f(softmaxScaleLog2 * (oldMaxVal - maxVal)) : 1.f;
                float corrScale1 = isValidCtaIdxKv ? exp2f(softmaxScaleLog2 * (localMax - maxVal)) : 0.f;
                // Update the old max value.
                oldMaxVal = maxVal;
                // The partialO value.
                uint4 vec = partialOArray[jj];
                // Reduce sum and finalO.
                sumVal = sumVal * corrScale0 + localSum * corrScale1;
                convertToFloatAndAccumulate<DtypePartialO>(outputVals, vec, corrScale0, corrScale1);
            }
        }

        if constexpr (KvSplit > 1)
        {
            // Make sure that the previous slice has been merged before overwriting the shared memory.
            __syncthreads();
            smemStats[2 * threadIdx.x] = maxVal;
            smemStats[2 * threadIdx.x + 1] = sumVal;
            smemO[2 * threadIdx.x] = make_float4(outputVals[0], outputVals[1], outputVals[2], outputVals[3]);
            smemO[2 * threadIdx.x + 1] = make_float4(outputVals[4], outputVals[5], outputVals[6], outputVals[7]);
            __syncthreads();
            if (kvGrpIdx == 0)
            {
                float grpMax{maxVal};
#pragma unroll
                for (int32_t gg = 1; gg < KvSplit; ++gg)
                {
                    grpMax = fmaxf(grpMax, smemStats[2 * (gg * NumThreadsPerGrp + warpGrpThreadIdx)]);
                }
                float const scale0{exp2f(softmaxScaleLog2 * (maxVal - grpMax))};
                sumVal *= scale0;
#pragma unroll
                for (int32_t kk = 0; kk < NumEltsPer16BVec; ++kk)
                {
                    outputVals[kk] *= scale0;
                }
#pragma unroll
                for (int32_t gg = 1; gg < KvSplit; ++gg)
                {
                    int32_t const tt{gg * NumThreadsPerGrp + warpGrpThreadIdx};
                    float const grpSum{smemStats[2 * tt + 1]};
                    if (grpSum == 0.f)
                    {
                        continue;
                    }
                    float const scale{exp2f(softmaxScaleLog2 * (smemStats[2 * tt] - grpMax))};
                    sumVal += grpSum * scale;
                    float4 const o0{smemO[2 * tt]};
                    float4 const o1{smemO[2 * tt + 1]};
                    outputVals[0] += o0.x * scale;
                    outputVals[1] += o0.y * scale;
                    outputVals[2] += o0.z * scale;
                    outputVals[3] += o0.w * scale;
                    outputVals[4] += o1.x * scale;
                    outputVals[5] += o1.y * scale;
                    outputVals[6] += o1.z * scale;
                    outputVals[7] += o1.w * scale;
                }
                maxVal = grpMax;
            }
        }
        if (KvSplit > 1 && kvGrpIdx != 0)
        {
            continue;
        }

        // Update the sums with the attention sink value.
        if (attentionSinksPtr != nullptr)
        {
            float attentionSinkVal = exp2f(attentionSinksPtr[localHeadIdxO] * M_LOG2E - maxVal * softmaxScaleLog2);
            // Multiply the attention sink value by the quantization scale included in the sum value.
            sumVal += attentionSinkVal * softmaxPScale;
        }

        // Stores the final softmax stats values to global memory if needed (Helix attention, which
        // splits seqLenKv across GPUs).
        if (storesSoftmaxStats && isValidRow && headDimIdx == 0)
        {
            // The softmaxScale.
            float softmaxScale = (softmaxScaleLog2 * (1.f / M_LOG2E));
            // The sumScale to unscale the softmaxPScale quantization scale from P.
            float sumScale = 1.f / softmaxPScale;
            // The final max and sum values.
            float2 stats{maxVal * softmaxScale, sumVal * sumScale};
            // Store the final max and sum values to global memory.
            reinterpret_cast<float2*>(softmaxStatsPtr)[softmaxStatsRowIdx] = stats;
        }

        // The final normalized scale.
        // Make sure that sumVal is divided by the quantization scale, so
        // 1.0f / (sumVal / softmaxPScale) = softmaxPScale / sumVal.
        float normalizedScale{softmaxPScale / sumVal};
        float2 normalizedScale2{normalizedScale, normalizedScale};

        // Apply the normalized scale to the reduced O values.
        for (int ii = 0; ii < NumEltsPer16BVec / 2; ++ii)
        {
            float2& f2 = reinterpret_cast<float2*>(outputVals)[ii];
            mul(f2, f2, normalizedScale2);
        }

        // Convert the float values to DtypeO, and Store it to global memory.
        if (isValidRow)
        {
            convertAndStoreToGmem<DtypeO>(reinterpret_cast<char*>(oPtr + gmemStoreOffset), outputVals);
        }
    }

// Trigger the secondary kernel.
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaTriggerProgrammaticLaunchCompletion();
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////

using FmhaReductionKernelFn
    = void (*)(fmha::KernelParams const, bool, bool, bool, float, int32_t, int32_t, int32_t, int32_t);

template <int32_t TileSizePerCtaQ, int32_t HeadDimPerCta, typename DtypeO, typename DtypePartialO>
FmhaReductionKernelFn selectFmhaReductionKvSplit(int32_t kvSplit)
{
    // A KV group needs at least one full row (NumThreadsPerCta / KvSplit * 16 bytes >= HeadDimPerCta * 2 bytes).
    if constexpr (NumThreadsPerCta / 16 * 16 >= HeadDimPerCta * 2)
    {
        if (kvSplit == 16)
        {
            return &fmhaReductionKernel<TileSizePerCtaQ, HeadDimPerCta, DtypeO, DtypePartialO, 16>;
        }
    }
    switch (kvSplit)
    {
    case 8: return &fmhaReductionKernel<TileSizePerCtaQ, HeadDimPerCta, DtypeO, DtypePartialO, 8>;
    case 4: return &fmhaReductionKernel<TileSizePerCtaQ, HeadDimPerCta, DtypeO, DtypePartialO, 4>;
    case 2: return &fmhaReductionKernel<TileSizePerCtaQ, HeadDimPerCta, DtypeO, DtypePartialO, 2>;
    default: return &fmhaReductionKernel<TileSizePerCtaQ, HeadDimPerCta, DtypeO, DtypePartialO, 1>;
    }
}

#define SELECT_FMHA_REDUCTION_KERNEL(TileSizePerCtaQ, HeadDimPerCta)                                                   \
    if (kernelMeta.mDataTypeQ == DATA_TYPE_E4M3)                                                                       \
    {                                                                                                                  \
        if (kernelMeta.mDataTypeO == DATA_TYPE_E4M3)                                                                   \
        {                                                                                                              \
            kernel = selectFmhaReductionKvSplit<TileSizePerCtaQ, HeadDimPerCta, __nv_fp8_e4m3, half>(kvSplit);         \
        }                                                                                                              \
        else if (kernelMeta.mDataTypeO == DATA_TYPE_FP16)                                                              \
        {                                                                                                              \
            kernel = selectFmhaReductionKvSplit<TileSizePerCtaQ, HeadDimPerCta, half, half>(kvSplit);                  \
        }                                                                                                              \
        else if (kernelMeta.mDataTypeO == DATA_TYPE_BF16)                                                              \
        {                                                                                                              \
            kernel                                                                                                     \
                = selectFmhaReductionKvSplit<TileSizePerCtaQ, HeadDimPerCta, __nv_bfloat16, __nv_bfloat16>(kvSplit);   \
        }                                                                                                              \
        else                                                                                                           \
        {                                                                                                              \
            TLLM_CHECK_WITH_INFO(false, "Not implemented");                                                            \
        }                                                                                                              \
    }                                                                                                                  \
    else                                                                                                               \
    {                                                                                                                  \
        TLLM_CHECK_WITH_INFO(kernelMeta.mDataTypeQ == kernelMeta.mDataTypeO, "Not implemented");                       \
        if (kernelMeta.mDataTypeQ == DATA_TYPE_FP16)                                                                   \
        {                                                                                                              \
            kernel = selectFmhaReductionKvSplit<TileSizePerCtaQ, HeadDimPerCta, half, half>(kvSplit);                  \
        }                                                                                                              \
        else if (kernelMeta.mDataTypeQ == DATA_TYPE_BF16)                                                              \
        {                                                                                                              \
            kernel                                                                                                     \
                = selectFmhaReductionKvSplit<TileSizePerCtaQ, HeadDimPerCta, __nv_bfloat16, __nv_bfloat16>(kvSplit);   \
        }                                                                                                              \
        else                                                                                                           \
        {                                                                                                              \
            TLLM_CHECK_WITH_INFO(false, "Not implemented");                                                            \
        }                                                                                                              \
    }

#define SELECT_FMHA_REDUCTION_KERNEL_WITH_HEAD_DIM_PER_CTA(HeadDimPerCta)                                              \
    if (kernelMeta.mTileSizeQ == 64)                                                                                   \
    {                                                                                                                  \
        SELECT_FMHA_REDUCTION_KERNEL(64, HeadDimPerCta);                                                               \
    }                                                                                                                  \
    else if (kernelMeta.mTileSizeQ == 128)                                                                             \
    {                                                                                                                  \
        SELECT_FMHA_REDUCTION_KERNEL(128, HeadDimPerCta);                                                              \
    }                                                                                                                  \
    else                                                                                                               \
    {                                                                                                                  \
        TLLM_CHECK_WITH_INFO(false, "Not implemented");                                                                \
    }

////////////////////////////////////////////////////////////////////////////////////////////////////

void runFmhaReduction(TllmGenFmhaKernelMetaInfo const& kernelMeta, fmha::KernelParams const& params,
    float softmaxPScale, int32_t multiProcessorCount, cudaStream_t stream)
{

    // Skip the kernel if not using the separate reduction kernel.
    if (!isGmemReductionWithSeparateKernel(static_cast<MultiCtasKvMode>(kernelMeta.mMultiCtasKvMode)))
    {
        return;
    }

    // This should only be enabled when using keepsMmaAbForGeneration kernel.
    TLLM_CHECK_WITH_INFO(
        isKeepsMmaAbForGenerationKernel(static_cast<FmhaKernelType>(kernelMeta.mKernelType)), "Not implemented");
    // The tileSizeQ should be 64 or 128 and tileSizeKv should be 128 for those kernels.
    TLLM_CHECK_WITH_INFO((kernelMeta.mTileSizeQ == 64 || kernelMeta.mTileSizeQ == 128) && kernelMeta.mTileSizeKv == 128,
        "Not implemented");

    // The headDimPerCtaV.
    int32_t const headDimPerCtaV = kernelMeta.m2CtaMma ? kernelMeta.mHeadDimPerCtaV * 2 : kernelMeta.mHeadDimPerCtaV;
    TLLM_CHECK_WITH_INFO(
        headDimPerCtaV == 64 || headDimPerCtaV == 128 || headDimPerCtaV == 256 || headDimPerCtaV == 512,
        "Not implemented");

    // The number of slices for the reduction work (one slice per CTA without the KV split).
    int32_t const numSlicesPerKvGrp
        = (headDimPerCtaV * /* bytesPerPartialElt */ 2 * kernelMeta.mTileSizeQ) / (NumThreadsPerCta * 16);
    // The number of heads computed by a single CTA.
    int numHeadsPerCta{1};
    if (kernelMeta.mGroupsHeadsQ)
    {
        numHeadsPerCta = std::min(params.mNumHeadsQPerKv, kernelMeta.mTileSizeQ);
    }
    // The number of Ctas for all heads.
    int32_t const numCtasForAllHeads{params.mNumHeadsQ / numHeadsPerCta};
    // The number of Ctas for headDim.
    int32_t const numHeadDimCtasV{kernelMeta.mHeadDimV / headDimPerCtaV};

    // The 512 threads will split the reduction work of TileSizePerCtaQ * HeadDimPerCta.
    dim3 blockDim(NumThreadsPerCta);
    dim3 gridDim;
    // Each CTA processes one tokenQ.
    gridDim.x = params.mMaxNumCtasQ;
    // The head dimension.
    gridDim.y = numCtasForAllHeads * numHeadDimCtasV;
    // The batch dimension.
    gridDim.z = params.mBatchSize;

    // The maximum number of Ctas for the reduction work.
    // This avoids having too many waves of CTAs which can have obvious launching overheads.
    int32_t const maxNumCtasForReduction{
        (multiProcessorCount * 2) / static_cast<int32_t>(gridDim.x * gridDim.y * gridDim.z)};
    // TLLM_FMHA_REDUCTION_KV_SPLIT=<2|4|8|16>: split the partials of each row across up to that many thread groups
    // (and that many times more CTAs) while the grid stays within the same CTA budget; unset or 1 keeps one group.
    static int32_t const maxKvSplit = []
    {
        int32_t const v = tensorrt_llm::common::getIntEnv("TLLM_FMHA_REDUCTION_KV_SPLIT").value_or(1);
        return v >= 16 ? 16 : v >= 8 ? 8 : v >= 4 ? 4 : v >= 2 ? 2 : 1;
    }();
    int32_t kvSplit{maxKvSplit};
    while (kvSplit > 1
        && ((NumThreadsPerCta / kvSplit) * 16 < headDimPerCtaV * 2
            || numSlicesPerKvGrp * kvSplit > maxNumCtasForReduction))
    {
        kvSplit /= 2;
    }
    int32_t const numSlices{numSlicesPerKvGrp * kvSplit};
    // The number of Ctas for the reduction work.
    int32_t const numCtasForReduction{std::min(maxNumCtasForReduction, numSlices)};
    // Launch more CTAs to split the reduction work if needed.
    gridDim.x *= numCtasForReduction;

    // The PDL attribute.
    cudaLaunchAttribute attribute[1];
    attribute[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attribute[0].val.programmaticStreamSerializationAllowed = tensorrt_llm::common::getEnvEnablePDL() ? 1 : 0;
    cudaLaunchConfig_t config;
    config.gridDim = gridDim;
    config.blockDim = blockDim;
    config.stream = stream;
    config.dynamicSmemBytes = 0;
    config.attrs = attribute;
    config.numAttrs = 1;

    // Select the kernel function pointer.
    FmhaReductionKernelFn kernel = nullptr;
    if (headDimPerCtaV == 64)
    {
        SELECT_FMHA_REDUCTION_KERNEL_WITH_HEAD_DIM_PER_CTA(64);
    }
    else if (headDimPerCtaV == 128)
    {
        SELECT_FMHA_REDUCTION_KERNEL_WITH_HEAD_DIM_PER_CTA(128);
    }
    else if (headDimPerCtaV == 256)
    {
        SELECT_FMHA_REDUCTION_KERNEL_WITH_HEAD_DIM_PER_CTA(256);
    }
    else if (headDimPerCtaV == 512)
    {
        SELECT_FMHA_REDUCTION_KERNEL_WITH_HEAD_DIM_PER_CTA(512);
    }

    // Launch the kernel.
    bool const hasVarSparseMlaTopKLens = isDynamicTokenSparse(static_cast<SparseType>(kernelMeta.mSparseAttn))
        && params.ptrSparseMlaTopKLens != nullptr;
    TLLM_CUDA_CHECK(
        cudaLaunchKernelEx(&config, kernel, params, isTokenSparse(static_cast<SparseType>(kernelMeta.mSparseAttn)),
            kernelMeta.mGroupsTokensHeadsQ, hasVarSparseMlaTopKLens, softmaxPScale, numCtasForReduction,
            numCtasForAllHeads, kernelMeta.mHeadDimV, numHeadDimCtasV));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace kernels

TRTLLM_NAMESPACE_END
