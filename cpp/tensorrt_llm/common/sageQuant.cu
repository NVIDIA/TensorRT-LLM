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

#include "sageQuant.h"

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/logger.h"

#include <algorithm>
#include <cstdint>
#include <cute/tensor.hpp>
#include <cutlass/array.h>
#include <cutlass/numeric_conversion.h>
#include <type_traits>

namespace tensorrt_llm::common
{

/// @brief SageAttn quantization kernel for Q and K

// Quantization kernel for SageAttention doing 2 tasks per invocation:
// 1. Performs per-token-block quantization for Q **or** K, depending on the actual pointers passed in.
// 2. [Optional when gridDim.z>=2] Performs either per-channel Sfs gathering or per-channel quantization for V.
//
// NOTE: all tensors in this file are treated as column-major: [D, H, S].
template <typename Element, typename ElementQuantized, int TokenPerScale, int HeadDim, bool KSmooth, int VStage>
__global__ void sageQuantQkvKernel(int sumSeqLensQk, void const* ptrQk, void* ptrQkQuant, float* ptrQkScale,
    float* ptrKMean, int sumSeqLensV, int numHeadsV, void const* ptrV, void* ptrVQuant, float* ptrVScale)
{
    using namespace cute;
    using namespace cutlass;
    static_assert(!KSmooth, "K-smoothing not implemented yet");
#ifdef ENABLE_FP8
    static_assert(std::is_same_v<ElementQuantized, float_e4m3_t> || std::is_same_v<ElementQuantized, std::int8_t>,
        "Unrecognized target dtype for quantization");
    constexpr float TypeMax = cute::is_same_v<ElementQuantized, float_e4m3_t> ? 448.0f : static_cast<float>(126.9f);
#else
    static_assert(
        std::is_same_v<ElementQuantized, std::int8_t>, "Only int8 quantization is available without ENABLE_FP8.");
    constexpr float TypeMax = static_cast<float>(126.9f);
#endif
    constexpr int BestVL = 128 / sizeof_bits_v<Element>;
    using VL = Int<BestVL>;

    // Silence currently-unused argument until K-smoothing support is added.
    (void) ptrKMean;

    int const numHeads = gridDim.y;
    int const headIdx = blockIdx.y;
    int const numWarpsPerCta = blockDim.x / 32;
    int const numWarps = gridDim.x * numWarpsPerCta;
    int const warpId = blockIdx.x * numWarpsPerCta + threadIdx.x / 32;
    int const thrId = threadIdx.x % 32;

    if (blockIdx.z == 0)
    {
        // Qk task -- one-off per-token-block quantization.

        // IO tensors
        Tensor gQk
            = make_tensor(reinterpret_cast<Element const*>(ptrQk), make_shape(Int<HeadDim>{}, numHeads, sumSeqLensQk));
        Tensor gQkQuant = make_tensor(
            reinterpret_cast<ElementQuantized*>(ptrQkQuant), make_shape(Int<HeadDim>{}, numHeads, sumSeqLensQk));
        Tensor gQkScale = make_tensor(ptrQkScale, make_shape(ceil_div(sumSeqLensQk, TokenPerScale), numHeads));

        // This head
        Tensor gQkSeq = gQk(_, headIdx, _);
        Tensor gQkSeqQuant = gQkQuant(_, headIdx, _);
        Tensor gQkSeqScale = gQkScale(_, headIdx);

        // Tiling
        Tensor gQkVecs = tiled_divide(gQkSeq, Shape<VL, Int<TokenPerScale>>{});
        Tensor gQkVecsQuant = tiled_divide(gQkSeqQuant, Shape<VL, Int<TokenPerScale>>{});

        // Register buffers
        Tensor rQk = make_tensor<Element>(Shape<VL, Int<TokenPerScale>>{});
        Tensor rQkQuant = make_tensor<ElementQuantized>(Shape<VL, Int<TokenPerScale>>{});
        Tensor rQkCompute = make_tensor<float>(Shape<VL, Int<TokenPerScale>>{});

        // Compute tensors
        Tensor rQk_x2 = recast<Array<Element, 2>>(rQk);
        Tensor rQkCompute_x2 = recast<Array<float, 2>>(rQkCompute);
        // Conversion tensors
        Tensor rQk_x4 = recast<Array<Element, 4>>(rQk);
        Tensor rQkQuant_x4 = recast<Array<ElementQuantized, 4>>(rQkQuant);

        // Threads count per token block
        constexpr int threadsPerScale = size<1>(gQkVecs);
        static_assert(threadsPerScale <= 32, "One token block should never exceed warp scope");
        int const numScalesPerWarp = 32 / threadsPerScale;
        int const numScalesPerWave = numWarps * numScalesPerWarp;
        int const numWholeScales = sumSeqLensQk / TokenPerScale;

        // Thread coordinates
        int tokBlkIdx = warpId * numScalesPerWarp + thrId / threadsPerScale;
        int threadInScaleIdx = thrId % threadsPerScale;

        // Unpredicated iterations
        for (; tokBlkIdx < numWholeScales; tokBlkIdx += numScalesPerWave)
        {
            // Load input
            cute::copy(AutoVectorizingCopy{}, gQkVecs(_, threadInScaleIdx, tokBlkIdx), rQk);
            cute::transform(rQk_x2, rQkCompute_x2, NumericArrayConverter<float, Element, 2>::convert);

            // Intra-thread reduction
            float maxScale = 1e-3f;
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < size(rQk); ++i)
            {
                maxScale = ::fmaxf(maxScale, ::fabsf(rQkCompute(i)));
            }
            // Intra-warp reduction
            CUTLASS_PRAGMA_UNROLL
            for (int delta = 1; delta < threadsPerScale; delta <<= 1)
            {
                maxScale = ::fmaxf(maxScale, __shfl_xor_sync(0xffffffffu, maxScale, delta));
            }

            // Rescale to TypeMax
            maxScale = maxScale / TypeMax;
            // Store maxScale
            gQkSeqScale(tokBlkIdx) = maxScale;

            // 1/maxScale
            Array<Element, 2> scaleQuant
                = NumericArrayConverter<Element, float, 2>::convert(Array<float, 2>{maxScale, maxScale});
            scaleQuant = cutlass::reciprocal_approximate<Array<Element, 2>>{}(scaleQuant);
            cutlass::multiplies<Array<Element, 2>> scaleQuantOp;
            // Qk /= maxScale
            cute::transform(rQk_x2, rQk_x2, [&](auto& x) { return scaleQuantOp(x, scaleQuant); });
            // Convert to target quant type
            cute::transform(rQk_x4, rQkQuant_x4, NumericArrayConverter<ElementQuantized, Element, 4>::convert);
            // Store quantized output
            cute::copy(AutoVectorizingCopy{}, rQkQuant, gQkVecsQuant(_, threadInScaleIdx, tokBlkIdx));
        }

        // Predicated iteration
        int const lastIterTokenIdx = tokBlkIdx * TokenPerScale;
        if (lastIterTokenIdx < sumSeqLensQk)
        {
            // Load input
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < size<1>(rQk); ++i)
            {
                if (lastIterTokenIdx + i < sumSeqLensQk)
                {
                    cute::copy(
                        AutoVectorizingCopy{}, gQkVecs(make_tuple(_, i), threadInScaleIdx, tokBlkIdx), rQk(_, i));
                }
                else
                {
                    CUTLASS_PRAGMA_UNROLL
                    for (int j = 0; j < BestVL; ++j)
                    {
                        rQk(j, i) = static_cast<Element>(0);
                    }
                }
            }
            cute::transform(rQk_x2, rQkCompute_x2, NumericArrayConverter<float, Element, 2>::convert);

            // Intra-thread reduction
            float maxScale = 1e-3f;
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < size(rQk); ++i)
            {
                maxScale = ::fmaxf(maxScale, ::fabsf(rQkCompute(i)));
            }
            // Intra-warp reduction
            CUTLASS_PRAGMA_UNROLL
            for (int delta = 1; delta < threadsPerScale; delta <<= 1)
            {
                maxScale = ::fmaxf(maxScale, __shfl_xor_sync(0xffffffffu, maxScale, delta));
            }

            // Rescale to TypeMax
            maxScale = maxScale / TypeMax;
            // Store maxScale
            gQkSeqScale(tokBlkIdx) = maxScale;

            // 1/maxScale
            Array<Element, 2> scaleQuant
                = NumericArrayConverter<Element, float, 2>::convert(Array<float, 2>{maxScale, maxScale});
            scaleQuant = cutlass::reciprocal_approximate<Array<Element, 2>>{}(scaleQuant);
            cutlass::multiplies<Array<Element, 2>> scaleQuantOp;
            // Qk /= maxScale
            cute::transform(rQk_x2, rQk_x2, [&](auto& x) { return scaleQuantOp(x, scaleQuant); });
            // Convert to target quant type
            cute::transform(rQk_x4, rQkQuant_x4, NumericArrayConverter<ElementQuantized, Element, 4>::convert);

            // Store quantized output
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < size<1>(rQk); ++i)
            {
                if (lastIterTokenIdx + i < sumSeqLensQk)
                {
                    cute::copy(AutoVectorizingCopy{}, rQkQuant(_, i),
                        gQkVecsQuant(make_tuple(_, i), threadInScaleIdx, tokBlkIdx));
                }
            }
        }
    }
    else if (blockIdx.z == 1)
    {
        // V task -- per-channel (all tokens) 2-stage task
        using ElementQuantizedV = cutlass::float_e4m3_t;

        // IO tensors
        constexpr int threadsPerHead = HeadDim / BestVL;
        static_assert(HeadDim % BestVL == 0, "VL must divide HeadDim");
        static_assert(threadsPerHead <= 32, "One token block should never exceed warp scope");
        Tensor gV = make_tensor(
            reinterpret_cast<Element const*>(ptrV), make_shape(VL{}, Int<threadsPerHead>{}, numHeadsV, sumSeqLensV));
        Tensor gVQuant = make_tensor(reinterpret_cast<ElementQuantizedV*>(ptrVQuant),
            make_shape(VL{}, Int<threadsPerHead>{}, numHeadsV, sumSeqLensV));
        Tensor gVScale = make_tensor(ptrVScale, make_shape(VL{}, Int<threadsPerHead>{}, numHeadsV));

        // Register buffers
        Tensor rV = make_tensor<Element>(Shape<VL>{});
        Tensor rVMax = make_tensor<Element>(Shape<VL>{});
        Tensor rVQuant = make_tensor<ElementQuantizedV>(Shape<VL>{});
        Tensor rVScale = make_tensor<float>(Shape<VL>{});
        Tensor rVCompute = make_tensor<float>(Shape<VL>{});

        // Compute tensors
        Tensor rV_x2 = recast<Array<Element, 2>>(rV);
        Tensor rVMax_x2 = recast<Array<Element, 2>>(rVMax);
        Tensor rVScale_x2 = recast<Array<float, 2>>(rVScale);
        Tensor rVCompute_x2 = recast<Array<float, 2>>(rVCompute);

        // Conversion tensors
        Tensor rVCompute_x4 = recast<Array<float, 4>>(rVCompute);
        Tensor rVQuant_x4 = recast<Array<ElementQuantizedV, 4>>(rVQuant);

        // If the parallel on-going task is handling Q, numHeads inferred from gridDim.y could be larger than numHeadsKv
        if (headIdx < numHeadsV)
        {
            // Thread coordinates
            int const numToksPerWarp = 32 / threadsPerHead;
            int tokIdx = warpId * numToksPerWarp + thrId / threadsPerHead;
            int const threadInTokIdx = thrId % threadsPerHead;

            // Thread-local tensors
            Tensor gVSeq = gV(_, threadInTokIdx, headIdx, _);
            Tensor gVSeqQuant = gVQuant(_, threadInTokIdx, headIdx, _);
            Tensor gVSeqScale = gVScale(_, threadInTokIdx, headIdx);

            if constexpr (VStage == 1)
            {
                // Stage 1: reduction to obtain the Sfs

                // Avoid heavy atomics: limit the number of warps.
                int const numWarpsToUse = cutlass::fast_min(numWarps, 256);
                int const numToksPerWave = numWarpsToUse * numToksPerWarp;
                if (warpId >= numWarpsToUse)
                {
                    return;
                }

                // Initialize
                CUTLASS_PRAGMA_UNROLL
                for (int i = 0; i < size(rVScale); ++i)
                {
                    rVScale(i) = 1e-3f;
                }
                cute::transform(rVScale_x2, rVMax_x2, cutlass::NumericArrayConverter<Element, float, 2>::convert);

                // Loop over all tokens
                for (; tokIdx < sumSeqLensV; tokIdx += numToksPerWave)
                {
                    // Load inputs
                    cute::copy(AutoVectorizingCopy{}, gVSeq(_, tokIdx), rV);
                    // Compute abs-max
                    cute::transform(rV_x2, rV_x2, cutlass::absolute_value_op<Array<Element, 2>>{});
                    cute::transform(rV_x2, rVMax_x2, rVMax_x2, cutlass::maximum<Array<Element, 2>>{});
                }

                // Transform max to Sfs
                cute::transform(rVMax_x2, rVScale_x2, cutlass::NumericArrayConverter<float, Element, 2>::convert);
                cute::transform(rVScale_x2, rVScale_x2, cutlass::scale<Array<float, 2>>{1 / 448.0f});

                // Intra-warp reduction.
                for (int delta = threadsPerHead; delta < 32; delta <<= 1)
                {
                    cute::transform(rVScale, rVScale,
                        [&](auto const& x) { return ::fmaxf(x, __shfl_xor_sync(0xffffffffu, x, delta)); });
                }

                // Atomic reduction into global memory.
                if (threadInTokIdx == thrId)
                {
                    CUTLASS_PRAGMA_UNROLL
                    for (int i = 0; i < BestVL; ++i)
                    {
                        atomicMax(
                            reinterpret_cast<int32_t*>(&gVSeqScale(i)), *reinterpret_cast<int32_t const*>(&rVScale(i)));
                    }
                }
            }
            else if constexpr (VStage == 2)
            {
                // Stage 2: scale according to the Sfs

                // Full waves.
                int const numToksPerWave = numWarps * numToksPerWarp;

                // Load Sfs
                cute::copy(AutoVectorizingCopy{}, gVSeqScale, rVScale);
                // Take reciprocal
                cute::transform(rVScale_x2, rVScale_x2, cutlass::reciprocal_approximate<Array<float, 2>>{});

                // Loop over all tokens
                for (; tokIdx < sumSeqLensV; tokIdx += numToksPerWave)
                {
                    // Load inputs
                    cute::copy(AutoVectorizingCopy{}, gVSeq(_, tokIdx), rV);
                    // Convert up
                    cute::transform(rV_x2, rVCompute_x2, cutlass::NumericArrayConverter<float, Element, 2>::convert);
                    // Scale
                    cute::transform(rVCompute_x2, rVScale_x2, rVCompute_x2, cutlass::multiplies<Array<float, 2>>{});
                    // Convert (quantize)
                    cute::transform(
                        rVCompute_x4, rVQuant_x4, cutlass::NumericArrayConverter<ElementQuantizedV, float, 4>::convert);
                    // Write output
                    cute::copy(AutoVectorizingCopy{}, rVQuant, gVSeqQuant(_, tokIdx));
                }
            }
        }
    }
}

template <typename Element>
void invokeSageQuantQkvImpl(SageQuantParams const& params)
{
    using namespace cute;
    TLLM_CHECK_WITH_INFO(params.sumSeqLensQk > 0 && params.numHeads > 0 && params.headDim > 0
            && params.tokenBlockSize > 0 && params.ptrQk != nullptr && params.ptrQkQuant != nullptr
            && params.ptrQkScale != nullptr && params.smCount > 0,
        "Invalid SageQuantQk parameters.");
    TLLM_CHECK_WITH_INFO(params.vStage == 0
            || (params.sumSeqLensV > 0 && params.numHeadsV > 0 && params.ptrV != nullptr && params.ptrVQuant != nullptr
                && params.ptrVScale != nullptr),
        "Invalid SageQuantV parameters.");
    TLLM_CHECK_WITH_INFO(!params.kSmooth, "SageQuantQk K-smoothing is not supported yet.");

    auto invokeKernel = [&](auto headDimStatic, auto tokenBlockSizeStatic)
    {
        constexpr int HeadDim_ = headDimStatic;
        constexpr int TokenBlockSize_ = tokenBlockSizeStatic;

        SageQuantParams kernelParams = params;
        void* kernelArgs[] = {&kernelParams.sumSeqLensQk, &kernelParams.ptrQk, &kernelParams.ptrQkQuant,
            &kernelParams.ptrQkScale, &kernelParams.ptrKMean, &kernelParams.sumSeqLensV, &kernelParams.numHeadsV,
            &kernelParams.ptrV, &kernelParams.ptrVQuant, &kernelParams.ptrVScale};

        auto launchWithVStage = [&](auto vStageStatic)
        {
            constexpr int VStage_ = vStageStatic;
            void const* kernelFunc = nullptr;
            if (params.quantType == kernels::DATA_TYPE_E4M3)
            {
#ifdef ENABLE_FP8
                kernelFunc = reinterpret_cast<void const*>(
                    sageQuantQkvKernel<Element, cutlass::float_e4m3_t, TokenBlockSize_, HeadDim_, false, VStage_>);
#else
                TLLM_THROW("SageQuantQk FP8 quantization requires ENABLE_FP8.");
#endif
            }
            else if (params.quantType == kernels::DATA_TYPE_INT8)
            {
                kernelFunc = reinterpret_cast<void const*>(
                    sageQuantQkvKernel<Element, std::int8_t, TokenBlockSize_, HeadDim_, false, VStage_>);
            }
            else
            {
                TLLM_THROW("Unsupported SageQuantQk quantType: %d.", static_cast<int>(params.quantType));
            }

            uint32_t const gridX = static_cast<uint32_t>(std::max(1, (params.smCount * 32) / params.numHeads));
            uint32_t const gridY = static_cast<uint32_t>(params.numHeads);
            uint32_t const gridZ = VStage_ > 0 ? 2U : 1U;
            dim3 const launchGrid{gridX, gridY, gridZ};
            check_cuda_error(cudaLaunchKernel(kernelFunc, launchGrid, dim3{64U, 1U, 1U}, kernelArgs, 0, params.stream));
            check_cuda_error(cudaPeekAtLastError());
        };

        switch (params.vStage)
        {
        case 0: launchWithVStage(Int<0>{}); return;
        case 1: launchWithVStage(Int<1>{}); return;
        case 2: launchWithVStage(Int<2>{}); return;
        default: TLLM_THROW("Unsupported SageQuantV stage: %d.", params.vStage);
        }
    };

    // Dispatch
    if (params.headDim == 64)
    {
        switch (params.tokenBlockSize)
        {
        case 1: invokeKernel(Int<64>{}, Int<1>{}); return;
        case 4: invokeKernel(Int<64>{}, Int<4>{}); return;
        case 16: invokeKernel(Int<64>{}, Int<16>{}); return;
        default: break;
        }
    }
    if (params.headDim == 128)
    {
        switch (params.tokenBlockSize)
        {
        case 1: invokeKernel(Int<128>{}, Int<1>{}); return;
        case 4: invokeKernel(Int<128>{}, Int<4>{}); return;
        case 16: invokeKernel(Int<128>{}, Int<16>{}); return;
        default: break;
        }
    }
    if (params.headDim == 256)
    {
        switch (params.tokenBlockSize)
        {
        case 1: invokeKernel(Int<256>{}, Int<1>{}); return;
        case 4: invokeKernel(Int<256>{}, Int<4>{}); return;
        case 16: invokeKernel(Int<256>{}, Int<16>{}); return;
        default: break;
        }
    }
    TLLM_THROW(
        "Unsupported SageQuantQk dispatch config: headDim=%d tokenBlockSize=%d", params.headDim, params.tokenBlockSize);
}

void invokeSageQuant(SageQuantParams const& params)
{
    if (params.inputType == kernels::DATA_TYPE_FP16)
    {
        invokeSageQuantQkvImpl<cutlass::half_t>(params);
        return;
    }
#ifdef ENABLE_BF16
    if (params.inputType == kernels::DATA_TYPE_BF16)
    {
        invokeSageQuantQkvImpl<cutlass::bfloat16_t>(params);
        return;
    }
#endif
    TLLM_THROW("Unsupported SageQuantQk inputType: %d", static_cast<int>(params.inputType));
}

} // namespace tensorrt_llm::common
