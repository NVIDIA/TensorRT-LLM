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

#include "sageQuantQk.h"

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

// NOTE: all tensors in this file are treated as column-major: [D, H, S].
template <typename Element, typename ElementQuantized, int TokenPerScale, int HeadDim, bool KSmooth>
__global__ void sageQuantQkKernel(
    int sumSeqLensQk, void const* ptrQk, void* ptrQkQuant, float* ptrQkScale, float* ptrKMean)
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
                cute::copy(AutoVectorizingCopy{}, gQkVecs(make_tuple(_, i), threadInScaleIdx, tokBlkIdx), rQk(_, i));
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
                cute::copy(
                    AutoVectorizingCopy{}, rQkQuant(_, i), gQkVecsQuant(make_tuple(_, i), threadInScaleIdx, tokBlkIdx));
            }
        }
    }
}

template <typename Element>
void invokeSageQuantQkImpl(SageQuantQkParams const& params)
{
    using namespace cute;
    TLLM_CHECK_WITH_INFO(params.sumSeqLensQk > 0 && params.numHeads > 0 && params.headDim > 0 && params.tokenBlockSize > 0
            && params.ptrQk != nullptr && params.ptrQkQuant != nullptr && params.ptrQkScale != nullptr && params.smCount > 0,
        "Invalid SageQuantQk parameters.");
    TLLM_CHECK_WITH_INFO(!params.kSmooth, "SageQuantQk K-smoothing is not supported yet.");

    auto invokeKernel = [&](auto headDimStatic, auto tokenBlockSizeStatic)
    {
        constexpr int HeadDim_ = headDimStatic;
        constexpr int TokenBlockSize_ = tokenBlockSizeStatic;

        int sumSeqLensQk = params.sumSeqLensQk;
        void const* ptrQk = params.ptrQk;
        void* ptrQkQuant = params.ptrQkQuant;
        float* ptrQkScale = params.ptrQkScale;
        float* ptrKMean = params.ptrKMean;
        void* kernelArgs[] = {&sumSeqLensQk, &ptrQk, &ptrQkQuant, &ptrQkScale, &ptrKMean};
        void const* kernelFunc = nullptr;
        if (params.quantType == kernels::DATA_TYPE_E4M3)
        {
#ifdef ENABLE_FP8
            kernelFunc = reinterpret_cast<void const*>(
                sageQuantQkKernel<Element, cutlass::float_e4m3_t, TokenBlockSize_, HeadDim_, false>);
#else
            TLLM_THROW("SageQuantQk FP8 quantization requires ENABLE_FP8.");
#endif
        }
        else if (params.quantType == kernels::DATA_TYPE_INT8)
        {
            kernelFunc = reinterpret_cast<void const*>(
                sageQuantQkKernel<Element, std::int8_t, TokenBlockSize_, HeadDim_, false>);
        }
        else
        {
            TLLM_THROW("Unsupported SageQuantQk quantType: %d.", static_cast<int>(params.quantType));
        }

        uint32_t const gridX = static_cast<uint32_t>(std::max(1, (params.smCount * 32) / params.numHeads));
        dim3 const launchGrid{gridX, static_cast<uint32_t>(params.numHeads), 1U};
        check_cuda_error(cudaLaunchKernel(kernelFunc, launchGrid, dim3{64U, 1U, 1U}, kernelArgs, 0, params.stream));
        check_cuda_error(cudaPeekAtLastError());
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
    TLLM_THROW("Unsupported SageQuantQk dispatch config: headDim=%d tokenBlockSize=%d", params.headDim, params.tokenBlockSize);
}

void invokeSageQuantQk(SageQuantQkParams const& params)
{
    if (params.inputType == kernels::DATA_TYPE_FP16)
    {
        invokeSageQuantQkImpl<cutlass::half_t>(params);
        return;
    }
#ifdef ENABLE_BF16
    if (params.inputType == kernels::DATA_TYPE_BF16)
    {
        invokeSageQuantQkImpl<cutlass::bfloat16_t>(params);
        return;
    }
#endif
    TLLM_THROW("Unsupported SageQuantQk inputType: %d", static_cast<int>(params.inputType));
}

} // namespace tensorrt_llm::common