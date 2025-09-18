/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION. All rights reserved.
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
#include "kernelUtils.h"
#include "tensorrt_llm/common/envUtils.h"
#include <cuda_runtime_api.h>
#include <float.h>

namespace tensorrt_llm
{
namespace kernels
{

////////////////////////////////////////////////////////////////////////////////////////////////////

#define NumThreadsPerCta 512

template <int32_t TileSizePerCtaQ, int32_t HeadDim, int32_t HeadDimPerCta, bool IsE4m3Bmm, typename DtypeO,
    typename DtypePartialO>
__global__ void __launch_bounds__(NumThreadsPerCta, 2) fmhaReductionKernel(
    KernelParams const params, int32_t numCtasForReduction, int32_t numCtasForAllHeads, int32_t numHeadDimCtasV)
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
    // The headIdxO.
    int32_t const headIdxO{headGrpIdxO * TileSizePerCtaQ};
    // The warpGrpThreadIdx.
    int32_t const warpGrpThreadIdx{static_cast<int32_t>(threadIdx.x)};

    // The number of validRows.
    int32_t const numValidRows{TileSizePerCtaQ};
    // The actual number of seqLenKv.
    int32_t seqLenKv{params.ptrSeqLensKv[batchIdx]};
    // Consider the causal-mask speculative decoding.
    seqLenKv = seqLenKv - ((params.mMaxSeqLenQ - 1) - ctaIdxQ);
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
        ((batchIdx * params.mMaxNumCtasQ + ctaIdxQ) * numCtasForAllHeads + headGrpIdxO) * TileSizePerCtaQ};
    // The offset of the O buffer.
    int64_t const oOffset{softmaxStatsOffset * HeadDim + headDimCtaIdxV * HeadDimPerCta};

    // The partialStats pointer.
    float2* partialStatsPtr = reinterpret_cast<float2*>(params.ptrPartialStats) + partialStatsOffset;
    // The partialO pointer.
    DtypePartialO* partialOPtr = reinterpret_cast<DtypePartialO*>(params.ptrPartialO) + partialOOffset;
    // The softmaxStats pointer.
    float2* softmaxStatsPtr = reinterpret_cast<float2*>(params.ptrSoftmaxStats) + softmaxStatsOffset;
    // The O pointer.
    DtypeO* oPtr = reinterpret_cast<DtypeO*>(params.ptrO) + oOffset;
    // The attentionSinks pointer.
    float const* attentionSinksPtr = params.ptrAttentionSinks + headIdxO;

    // Whether to store the softmax stats.
    bool const storesSoftmaxStats{params.ptrSoftmaxStats != nullptr};

    // The softmaxScaleLog2.
    float const softmaxScaleLog2 = params.mScaleSoftmaxLog2;

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
    int32_t constexpr NumBytePerSlice{NumThreadsPerCta * 16};
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
        int64_t gmemStoreOffset{validRowIdx * HeadDim + headDimIdx};
        // The local headIdxO.
        int32_t localHeadIdxO{validRowIdx};

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
        for (int32_t ii = 0; ii < numCtasKv; ii += UnrollSize)
        {
            // The partialStats array and partialO array.
            float2 partialStatsArray[UnrollSize];
            uint4 partialOArray[UnrollSize];
#pragma unroll
            for (int32_t jj = 0; jj < UnrollSize; ++jj)
            {
                int32_t ctaIdxKv = min(ii + jj, numCtasKv - 1);
                partialStatsArray[jj] = localPartialStatsPtr[ctaIdxKv * numRowsInPartialBuffers];
                partialOArray[jj] = *reinterpret_cast<uint4 const*>(
                    localPartialOPtr + ctaIdxKv * numRowsInPartialBuffers * HeadDimPerCta);
            }
#pragma unroll
            for (int32_t jj = 0; jj < UnrollSize; ++jj)
            {
                // Whether the ctaIdxKv is valid.
                bool const isValidCtaIdxKv = (ii + jj) < numCtasKv;
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

        // Update the sums with the attention sink value.
        if (attentionSinksPtr != nullptr)
        {
            float attentionSinkVal = exp2f(attentionSinksPtr[localHeadIdxO] * M_LOG2E - maxVal * softmaxScaleLog2);
            // Multiply the attention sink value by 448.f if the MMA data type is e4m3 as the sum value
            // has also included the 448.f quantization scale.
            sumVal += IsE4m3Bmm ? attentionSinkVal * 448.f : attentionSinkVal;
        }

        // Stores the final softmax stats values to global memory if needed (Helix attention, which
        // splits seqLenKv across GPUs).
        if (storesSoftmaxStats && isValidRow && headDimIdx == 0)
        {
            // The softmaxScale.
            float softmaxScale = (softmaxScaleLog2 * (1.f / M_LOG2E));
            // The sumScale to unscale the 448.f quantization scale from P.
            float sumScale = IsE4m3Bmm ? (1.f / 448.f) : 1.f;
            // The final max and sum values.
            float2 stats{maxVal * softmaxScale, sumVal * sumScale};
            // Store the final max and sum values to global memory.
            reinterpret_cast<float2*>(softmaxStatsPtr)[validRowIdx] = stats;
        }

        // The final normalized scale.
        // If the output data type is e4m3, make sure that sumVal is divided by the quantization scale
        // (448.f), so 1.0f / (sumVal / 448.f) = 448.f / sumVal.
        float normalizedScale{IsE4m3Bmm ? (448.f / sumVal) : (1.0f / sumVal)};
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

#define SELECT_FMHA_REDUCTION_KERNEL(HeadDimPerCta)                                                                    \
    if (kernelMeta.mDataTypeQ == DATA_TYPE_E4M3)                                                                       \
    {                                                                                                                  \
        if (kernelMeta.mDataTypeO == DATA_TYPE_E4M3)                                                                   \
        {                                                                                                              \
            kernel = &fmhaReductionKernel<64, 512, HeadDimPerCta, true, __nv_fp8_e4m3, half>;                          \
        }                                                                                                              \
        else if (kernelMeta.mDataTypeO == DATA_TYPE_FP16)                                                              \
        {                                                                                                              \
            kernel = &fmhaReductionKernel<64, 512, HeadDimPerCta, true, half, half>;                                   \
        }                                                                                                              \
        else if (kernelMeta.mDataTypeO == DATA_TYPE_BF16)                                                              \
        {                                                                                                              \
            kernel = &fmhaReductionKernel<64, 512, HeadDimPerCta, true, __nv_bfloat16, __nv_bfloat16>;                 \
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
            kernel = &fmhaReductionKernel<64, 512, HeadDimPerCta, false, half, half>;                                  \
        }                                                                                                              \
        else if (kernelMeta.mDataTypeQ == DATA_TYPE_BF16)                                                              \
        {                                                                                                              \
            kernel = &fmhaReductionKernel<64, 512, HeadDimPerCta, false, __nv_bfloat16, __nv_bfloat16>;                \
        }                                                                                                              \
        else                                                                                                           \
        {                                                                                                              \
            TLLM_CHECK_WITH_INFO(false, "Not implemented");                                                            \
        }                                                                                                              \
    }

////////////////////////////////////////////////////////////////////////////////////////////////////

void runFmhaReduction(TllmGenFmhaKernelMetaInfo const& kernelMeta, KernelParams const& params,
    int32_t multiProcessorCount, cudaStream_t stream)
{

    // Skip the kernel if not using the separate reduction kernel.
    if (!isGmemReductionWithSeparateKernel(static_cast<MultiCtasKvMode>(kernelMeta.mMultiCtasKvMode)))
    {
        return;
    }

    // This should only be enabled when the keepsMmaAbForGeneration MLA kernel (either 1-CTA or 2-CTA)
    // is used.
    TLLM_CHECK_WITH_INFO(kernelMeta.mHeadDimQk == 576 && kernelMeta.mHeadDimV == 512
            && isKeepsMmaAbForGenerationKernel(static_cast<FmhaKernelType>(kernelMeta.mKernelType)),
        "Not implemented");
    // The tileSizeQ and tileSizeKv should be 64 and 128 for those kernels.
    TLLM_CHECK_WITH_INFO(kernelMeta.mTileSizeQ == 64 && kernelMeta.mTileSizeKv == 128, "Not implemented");

    // The headDimPerCtaV.
    int32_t const headDimPerCtaV = kernelMeta.m2CtaMma ? kernelMeta.mHeadDimPerCtaV * 2 : kernelMeta.mHeadDimPerCtaV;
    TLLM_CHECK_WITH_INFO(headDimPerCtaV == 128 || headDimPerCtaV == 256 || headDimPerCtaV == 512, "Not implemented");

    // The number of slices for the reduction work.
    int32_t const numSlices
        = (headDimPerCtaV * /* bytesPerPartialElt */ 2 * kernelMeta.mTileSizeQ) / (NumThreadsPerCta * 16);
    // The number of Ctas for all heads.
    int32_t const numCtasForAllHeads{params.mNumHeadsQ / kernelMeta.mTileSizeQ};
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
    void (*kernel)(KernelParams const, int32_t, int32_t, int32_t) = nullptr;
    if (headDimPerCtaV == 128)
    {
        SELECT_FMHA_REDUCTION_KERNEL(128);
    }
    else if (headDimPerCtaV == 256)
    {
        SELECT_FMHA_REDUCTION_KERNEL(256);
    }
    else if (headDimPerCtaV == 512)
    {
        SELECT_FMHA_REDUCTION_KERNEL(512);
    }

    // Launch the kernel.
    TLLM_CUDA_CHECK(
        cudaLaunchKernelEx(&config, kernel, params, numCtasForReduction, numCtasForAllHeads, numHeadDimCtasV));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace kernels
} // namespace tensorrt_llm
