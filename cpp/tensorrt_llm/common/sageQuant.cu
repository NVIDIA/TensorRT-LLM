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

#include "sagePartition.h"
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

using tensorrt_llm::kernels::DATA_TYPE_BF16;
using tensorrt_llm::kernels::DATA_TYPE_E4M3;
using tensorrt_llm::kernels::DATA_TYPE_FP16;
using tensorrt_llm::kernels::DATA_TYPE_INT8;

// SageAttention quantization kernel. Tensors are interpreted as column-major [D, H, S], which is
// the same physical layout as contiguous PyTorch [S, H, D]. Each launch quantizes Q or K. It can
// simultaneously collect V scales (VStage=1) or quantize V using scales collected by an earlier
// launch (VStage=2). When SmoothK is enabled, the VStage=1 launch also collects a per-head-per-
// channel K mean and the VStage=2 launch uses it to perform K-smoothing before quantization.
//
// Which tokens share a scale is a template parameter: Partition is a CuTe layout
// ((intra...), (group...)) -> token offset inside a tile (see sagePartition.h). Contiguous blocks
// and swizzled per-thread groups are both instances, so this kernel serves both without knowing
// which it was given.
//
// Q/K scale layout: a sequence's scales start at PartitionTraits::scaleBase(cuSeqLens[b], b) and
// the head stride is scaleHeadStride(sumSeqLensQk, batchSize). Every sequence reserves a whole
// spare tile, so a trailing partial tile -- whose scales the consumer still indexes in full --
// cannot run into the next sequence.
template <typename Element, typename ElementQuantized, typename Partition, int HeadDim, bool SmoothK, int VStage>
__global__ void sageQuantQkvKernel(int sumSeqLensQk, int batchSize, int const* ptrCuSeqLensQk, void const* ptrQk,
    void* ptrQkQuant, float* ptrQkScale, void const* ptrKForMean, float* ptrKMean, int sumSeqLensV, int numHeadsV,
    void const* ptrV, void* ptrVQuant, float* ptrVScale)
{
    using namespace cute;
    using namespace cutlass;
    using Traits = PartitionTraits<Partition>;
    constexpr int TokensPerScale = Traits::TokensPerScale;
    constexpr int ScalesPerTile = Traits::ScalesPerTile;
    constexpr int TileTokens = Traits::TileTokens;
    static_assert(!SmoothK || VStage != 0, "K smoothing requires V staging");
    static_assert(std::is_same_v<ElementQuantized, float_e4m3_t> || std::is_same_v<ElementQuantized, std::int8_t>,
        "Unrecognized target dtype for quantization");
    constexpr float TypeMax = cute::is_same_v<ElementQuantized, float_e4m3_t> ? 448.0f : static_cast<float>(126.9f);
    constexpr int BestVL = 128 / sizeof_bits_v<Element>;
    using VL = Int<BestVL>;

    int const numWarpsPerCta = blockDim.x / 32;
    int const numWarps = gridDim.x * numWarpsPerCta;
    int const warpId = blockIdx.x * numWarpsPerCta + threadIdx.x / 32;
    int const thrId = threadIdx.x % 32;

    if (blockIdx.z == 0)
    {
        // blockIdx.y maps to (headIdx * batchSize + seqIdx).
        int const numHeads = gridDim.y / batchSize;
        int const headIdx = blockIdx.y / batchSize;
        int const seqIdx = blockIdx.y % batchSize;

        // threadsPerScale threads cooperate on one scale group: each owns a BestVL-wide slice of
        // the head dim and all TokensPerScale tokens of the group, and they reduce over the head
        // dim with a shuffle. This is independent of how the group's tokens are laid out, which is
        // why the partition only has to change the addresses, not the thread mapping.
        constexpr int threadsPerScale = HeadDim / BestVL;
        static_assert(HeadDim % BestVL == 0, "VL must divide HeadDim");
        static_assert(threadsPerScale <= 32, "One token block should never exceed warp scope");
        int const numScalesPerWarp = 32 / threadsPerScale;
        int const numScalesPerWave = numWarps * numScalesPerWarp;
        int const scaleIdxInWave = warpId * numScalesPerWarp + thrId / threadsPerScale;
        int const threadInScaleIdx = thrId % threadsPerScale;
        constexpr uint32_t scaleMask = threadsPerScale == 32 ? ~0u : ((1u << threadsPerScale) - 1u);
        uint32_t const laneMask = scaleMask << (thrId / threadsPerScale * threadsPerScale);

        int const seqBegin = ptrCuSeqLensQk[seqIdx];
        int const seqLen = ptrCuSeqLensQk[seqIdx + 1] - seqBegin;
        if (seqLen <= 0)
        {
            return;
        }

        float* ptrQkScaleHead
            = ptrQkScale + static_cast<int64_t>(headIdx) * Traits::scaleHeadStride(sumSeqLensQk, batchSize);
        int const tokenStride = numHeads * HeadDim;
        int64_t const seqHeadOffset = static_cast<int64_t>(seqBegin) * tokenStride + headIdx * HeadDim;
        // This thread's BestVL-wide channel slice of token 0 of the sequence. A token is reached by
        // adding tokenIdx * tokenStride, so the partition only contributes an index, never a
        // stride.
        Element const* ptrQkThread
            = reinterpret_cast<Element const*>(ptrQk) + seqHeadOffset + threadInScaleIdx * BestVL;
        ElementQuantized* ptrQkQuantThread
            = reinterpret_cast<ElementQuantized*>(ptrQkQuant) + seqHeadOffset + threadInScaleIdx * BestVL;
        float* ptrQkSeqScale = ptrQkScaleHead + Traits::scaleBase(seqBegin, seqIdx);
        int const numTiles = ceil_div(seqLen, TileTokens);

        // tileIdx/grpIdx identify a scale; isFullTile is a compile-time fast path for tiles that
        // lie entirely inside the sequence. In a partial tile validity has to be tested per token
        // rather than as a prefix count, because a group's tokens are strided in general -- for
        // SM90 Q the two rows of a group are 8 apart, so one can be in range while the other is
        // not.
        auto quantizeScale = [&](auto isFullTile, int tileIdx, int grpIdx)
        {
            constexpr bool IsFullTile = decltype(isFullTile)::value;
            int const tileBase = tileIdx * TileTokens;
            auto tokenOf = [&](int i) { return tileBase + Traits::tokenInTile(i, grpIdx); };

            Tensor rQk = make_tensor<Element>(Shape<VL, Int<TokensPerScale>>{});
            Tensor rQkQuant = make_tensor<ElementQuantized>(Shape<VL, Int<TokensPerScale>>{});
            Tensor rQkCompute = make_tensor<float>(Shape<VL, Int<TokensPerScale>>{});
            Tensor rQk_x2 = recast<Array<Element, 2>>(rQk);
            Tensor rQkCompute_x2 = recast<Array<float, 2>>(rQkCompute);
            Tensor rQk_x4 = recast<Array<Element, 4>>(rQk);
            Tensor rQkQuant_x4 = recast<Array<ElementQuantized, 4>>(rQkQuant);

            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < TokensPerScale; ++i)
            {
                int const tokenIdx = tokenOf(i);
                if (IsFullTile || tokenIdx < seqLen)
                {
                    Tensor gSrc = make_tensor(
                        make_gmem_ptr(ptrQkThread + static_cast<int64_t>(tokenIdx) * tokenStride), Shape<VL>{});
                    cute::copy(AutoVectorizingCopy{}, gSrc, rQk(_, i));
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

            if constexpr (SmoothK && VStage == 2)
            {
                float const* ptrKMeanHead = ptrKMean + headIdx * HeadDim;
                CUTLASS_PRAGMA_UNROLL
                for (int tokenIdx = 0; tokenIdx < TokensPerScale; ++tokenIdx)
                {
                    if (IsFullTile || tokenOf(tokenIdx) < seqLen)
                    {
                        CUTLASS_PRAGMA_UNROLL
                        for (int vecIdx = 0; vecIdx < BestVL; ++vecIdx)
                        {
                            rQkCompute(vecIdx, tokenIdx) -= ptrKMeanHead[threadInScaleIdx * BestVL + vecIdx];
                        }
                    }
                }
            }

            float maxScale = 1e-3f;
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < size(rQk); ++i)
            {
                maxScale = ::fmaxf(maxScale, ::fabsf(rQkCompute(i)));
            }
            CUTLASS_PRAGMA_UNROLL
            for (int delta = 1; delta < threadsPerScale; delta <<= 1)
            {
                maxScale = ::fmaxf(maxScale, __shfl_xor_sync(laneMask, maxScale, delta));
            }

            maxScale /= TypeMax;
            // Every group of every tile gets a scale, including groups of a partial tile whose
            // tokens are all out of range: the consumer indexes a partial tile in full, and masks
            // with a large negative sentinel that must stay negative after being multiplied by this
            // scale. The 1e-3 floor above is what guarantees that -- do not lower it to an epsilon.
            ptrQkSeqScale[tileIdx * ScalesPerTile + grpIdx] = maxScale;
            float const invScale = 1.0f / maxScale;
            if constexpr (SmoothK && VStage == 2)
            {
                // rQkCompute holds the result value
                Tensor rQkCompute_x4 = recast<Array<float, 4>>(rQkCompute);
                cute::transform(rQkCompute, rQkCompute, [&](float const& x) { return x * invScale; });
                cute::transform(rQkCompute_x4, rQkQuant_x4, NumericArrayConverter<ElementQuantized, float, 4>::convert);
            }
            else
            {
                // rQk holds the result value
                Array<Element, 2> scaleQuant
                    = NumericArrayConverter<Element, float, 2>::convert(Array<float, 2>{invScale, invScale});
                cutlass::multiplies<Array<Element, 2>> scaleQuantOp;
                cute::transform(rQk_x2, rQk_x2, [&](auto& x) { return scaleQuantOp(x, scaleQuant); });
                cute::transform(rQk_x4, rQkQuant_x4, NumericArrayConverter<ElementQuantized, Element, 4>::convert);
            }

            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < TokensPerScale; ++i)
            {
                int const tokenIdx = tokenOf(i);
                if (IsFullTile || tokenIdx < seqLen)
                {
                    Tensor gDst = make_tensor(
                        make_gmem_ptr(ptrQkQuantThread + static_cast<int64_t>(tokenIdx) * tokenStride), Shape<VL>{});
                    cute::copy(AutoVectorizingCopy{}, rQkQuant(_, i), gDst);
                }
            }
        };

        // Work items are (tile, group) pairs. Whole tiles take the fast path; only the last tile of
        // a sequence can be partial, and all of its groups still need a scale written.
        int const numWholeTiles = seqLen / TileTokens;
        int const numWholeScales = numWholeTiles * ScalesPerTile;
        for (int workIdx = scaleIdxInWave; workIdx < numWholeScales; workIdx += numScalesPerWave)
        {
            quantizeScale(cute::true_type{}, workIdx / ScalesPerTile, workIdx % ScalesPerTile);
        }
        if (numWholeTiles < numTiles)
        {
            for (int grpIdx = scaleIdxInWave; grpIdx < ScalesPerTile; grpIdx += numScalesPerWave)
            {
                quantizeScale(cute::false_type{}, numWholeTiles, grpIdx);
            }
        }
    }
    else if (blockIdx.z == 1)
    {
        int const headIdx = blockIdx.y;
        using ElementQuantizedV = cutlass::float_e4m3_t;
        constexpr int threadsPerHead = HeadDim / BestVL;
        static_assert(HeadDim % BestVL == 0, "VL must divide HeadDim");
        static_assert(threadsPerHead <= 32, "One token block should never exceed warp scope");
        Tensor gV = make_tensor(
            reinterpret_cast<Element const*>(ptrV), make_shape(VL{}, Int<threadsPerHead>{}, numHeadsV, sumSeqLensV));
        Tensor gVQuant = make_tensor(reinterpret_cast<ElementQuantizedV*>(ptrVQuant),
            make_shape(VL{}, Int<threadsPerHead>{}, numHeadsV, sumSeqLensV));
        Tensor gVScale = make_tensor(ptrVScale, make_shape(VL{}, Int<threadsPerHead>{}, numHeadsV));

        Tensor rV = make_tensor<Element>(Shape<VL>{});
        Tensor rVMax = make_tensor<Element>(Shape<VL>{});
        Tensor rVQuant = make_tensor<ElementQuantizedV>(Shape<VL>{});
        Tensor rVScale = make_tensor<float>(Shape<VL>{});
        Tensor rVCompute = make_tensor<float>(Shape<VL>{});
        Tensor rV_x2 = recast<Array<Element, 2>>(rV);
        Tensor rVMax_x2 = recast<Array<Element, 2>>(rVMax);
        Tensor rVScale_x2 = recast<Array<float, 2>>(rVScale);
        Tensor rVCompute_x2 = recast<Array<float, 2>>(rVCompute);
        Tensor rVCompute_x4 = recast<Array<float, 4>>(rVCompute);
        Tensor rVQuant_x4 = recast<Array<ElementQuantizedV, 4>>(rVQuant);

        if (headIdx < numHeadsV)
        {
            int const numToksPerWarp = 32 / threadsPerHead;
            int tokIdx = warpId * numToksPerWarp + thrId / threadsPerHead;
            int const threadInTokIdx = thrId % threadsPerHead;
            Tensor gVSeq = gV(_, threadInTokIdx, headIdx, _);
            Tensor gVSeqQuant = gVQuant(_, threadInTokIdx, headIdx, _);
            Tensor gVSeqScale = gVScale(_, threadInTokIdx, headIdx);

            if constexpr (VStage == 1)
            {
                int const numWarpsToUse = cutlass::fast_min(numWarps, 256);
                int const numToksPerWave = numWarpsToUse * numToksPerWarp;
                if (warpId >= numWarpsToUse)
                {
                    return;
                }
                CUTLASS_PRAGMA_UNROLL
                for (int i = 0; i < size(rVScale); ++i)
                {
                    rVScale(i) = 1e-3f;
                }
                cute::transform(rVScale_x2, rVMax_x2, cutlass::NumericArrayConverter<Element, float, 2>::convert);
                for (; tokIdx < sumSeqLensV; tokIdx += numToksPerWave)
                {
                    cute::copy(AutoVectorizingCopy{}, gVSeq(_, tokIdx), rV);
                    cute::transform(rV_x2, rV_x2, cutlass::absolute_value_op<Array<Element, 2>>{});
                    cute::transform(rV_x2, rVMax_x2, rVMax_x2, cutlass::maximum<Array<Element, 2>>{});
                }
                cute::transform(rVMax_x2, rVScale_x2, cutlass::NumericArrayConverter<float, Element, 2>::convert);
                cute::transform(rVScale_x2, rVScale_x2, cutlass::scale<Array<float, 2>>{1 / 448.0f});
                for (int delta = threadsPerHead; delta < 32; delta <<= 1)
                {
                    cute::transform(rVScale, rVScale,
                        [&](auto const& x) { return ::fmaxf(x, __shfl_xor_sync(0xffffffffu, x, delta)); });
                }
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
                int const numToksPerWave = numWarps * numToksPerWarp;
                cute::copy(AutoVectorizingCopy{}, gVSeqScale, rVScale);
                cute::transform(rVScale_x2, rVScale_x2, cutlass::reciprocal_approximate<Array<float, 2>>{});
                for (; tokIdx < sumSeqLensV; tokIdx += numToksPerWave)
                {
                    cute::copy(AutoVectorizingCopy{}, gVSeq(_, tokIdx), rV);
                    cute::transform(rV_x2, rVCompute_x2, cutlass::NumericArrayConverter<float, Element, 2>::convert);
                    cute::transform(rVCompute_x2, rVScale_x2, rVCompute_x2, cutlass::multiplies<Array<float, 2>>{});
                    cute::transform(
                        rVCompute_x4, rVQuant_x4, cutlass::NumericArrayConverter<ElementQuantizedV, float, 4>::convert);
                    cute::copy(AutoVectorizingCopy{}, rVQuant, gVSeqQuant(_, tokIdx));
                }
            }
        }
    }
    else if (blockIdx.z == 2)
    {
        if constexpr (SmoothK && VStage == 1)
        {
            int const headIdx = blockIdx.y;
            constexpr int threadsPerHead = HeadDim / BestVL;
            static_assert(HeadDim % BestVL == 0, "VL must divide HeadDim");
            static_assert(threadsPerHead <= 32, "One token block should never exceed warp scope");

            if (headIdx < numHeadsV)
            {
                Tensor gK = make_tensor(reinterpret_cast<Element const*>(ptrKForMean),
                    make_shape(VL{}, Int<threadsPerHead>{}, numHeadsV, sumSeqLensV));
                Tensor gKMean = make_tensor(ptrKMean, make_shape(VL{}, Int<threadsPerHead>{}, numHeadsV));
                Tensor rK = make_tensor<Element>(Shape<VL>{});
                Tensor rKCompute = make_tensor<float>(Shape<VL>{});
                Tensor rKSum = make_tensor<float>(Shape<VL>{});
                Tensor rK_x2 = recast<Array<Element, 2>>(rK);
                Tensor rKCompute_x2 = recast<Array<float, 2>>(rKCompute);

                int const numToksPerWarp = 32 / threadsPerHead;
                int const numWarpsToUse = cutlass::fast_min(numWarps, 256);
                int const numToksPerWave = numWarpsToUse * numToksPerWarp;
                if (warpId >= numWarpsToUse)
                {
                    return;
                }
                int tokIdx = warpId * numToksPerWarp + thrId / threadsPerHead;
                int const threadInTokIdx = thrId % threadsPerHead;
                Tensor gKSeq = gK(_, threadInTokIdx, headIdx, _);
                Tensor gKMeanHead = gKMean(_, threadInTokIdx, headIdx);

                clear(rKSum);
                for (; tokIdx < sumSeqLensV; tokIdx += numToksPerWave)
                {
                    cute::copy(AutoVectorizingCopy{}, gKSeq(_, tokIdx), rK);
                    cute::transform(rK_x2, rKCompute_x2, NumericArrayConverter<float, Element, 2>::convert);
                    CUTLASS_PRAGMA_UNROLL
                    for (int i = 0; i < BestVL; ++i)
                    {
                        rKSum(i) += rKCompute(i);
                    }
                }
                for (int delta = threadsPerHead; delta < 32; delta <<= 1)
                {
                    CUTLASS_PRAGMA_UNROLL
                    for (int i = 0; i < BestVL; ++i)
                    {
                        rKSum(i) += __shfl_xor_sync(0xffffffffu, rKSum(i), delta);
                    }
                }
                if (threadInTokIdx == thrId)
                {
                    float const invNumTokens = 1.0f / sumSeqLensV;
                    CUTLASS_PRAGMA_UNROLL
                    for (int i = 0; i < BestVL; ++i)
                    {
                        atomicAdd(&gKMeanHead(i), rKSum(i) * invNumTokens);
                    }
                }
            }
        }
    }
}

template <typename Element>
void invokeSageQuantQkvImpl(SageQuantParams const& params)
{
    using namespace cute;
    TLLM_CHECK_WITH_INFO(params.sumSeqLensQk > 0 && params.batchSize > 0 && params.ptrCuSeqLensQk != nullptr
            && params.numHeads > 0 && params.headDim > 0
            && (params.partition != SageScalePartition::Contiguous || params.tokenBlockSize > 0)
            && params.ptrQk != nullptr && params.ptrQkQuant != nullptr && params.ptrQkScale != nullptr
            && params.smCount > 0,
        "Invalid SageQuantQk parameters");
    TLLM_CHECK_WITH_INFO(params.vStage == 0
            || (params.sumSeqLensV > 0 && params.numHeadsV > 0 && params.ptrV != nullptr && params.ptrVQuant != nullptr
                && params.ptrVScale != nullptr),
        "Invalid SageQuantV parameters");
    TLLM_CHECK_WITH_INFO(!params.kSmooth || params.vStage != 0, "SageQuant K smoothing requires V staging");
    TLLM_CHECK_WITH_INFO(
        !params.kSmooth || (params.ptrKMean != nullptr && (params.vStage != 1 || params.ptrKForMean != nullptr)),
        "Invalid SageQuant K-smoothing parameters");

    auto invokeKernel = [&](auto headDimStatic, auto partitionStatic)
    {
        constexpr int HeadDim_ = headDimStatic;
        using Partition_ = decltype(partitionStatic);
        SageQuantParams kernelParams = params;
        void* kernelArgs[] = {&kernelParams.sumSeqLensQk, &kernelParams.batchSize, &kernelParams.ptrCuSeqLensQk,
            &kernelParams.ptrQk, &kernelParams.ptrQkQuant, &kernelParams.ptrQkScale, &kernelParams.ptrKForMean,
            &kernelParams.ptrKMean, &kernelParams.sumSeqLensV, &kernelParams.numHeadsV, &kernelParams.ptrV,
            &kernelParams.ptrVQuant, &kernelParams.ptrVScale};

        auto launchKernel = [&](auto smoothKStatic, auto vStageStatic)
        {
            constexpr bool SmoothK_ = decltype(smoothKStatic)::value;
            constexpr int VStage_ = vStageStatic;
            void const* kernelFunc = nullptr;
            if (params.quantType == DATA_TYPE_E4M3)
            {
                kernelFunc = reinterpret_cast<void const*>(
                    sageQuantQkvKernel<Element, cutlass::float_e4m3_t, Partition_, HeadDim_, SmoothK_, VStage_>);
            }
            else if (params.quantType == DATA_TYPE_INT8)
            {
                kernelFunc = reinterpret_cast<void const*>(
                    sageQuantQkvKernel<Element, std::int8_t, Partition_, HeadDim_, SmoothK_, VStage_>);
            }
            else
            {
                TLLM_THROW("SageQuant Q/K output must be INT8 or FP8 E4M3");
            }
            int const numHeadSeqs = params.numHeads * params.batchSize;
            uint32_t const gridX = static_cast<uint32_t>(std::max(1, (params.smCount * 32) / numHeadSeqs));
            constexpr uint32_t GridZ = VStage_ == 0 ? 1U : (SmoothK_ && VStage_ == 1 ? 3U : 2U);
            dim3 const launchGrid{gridX, static_cast<uint32_t>(numHeadSeqs), GridZ};
            auto status = cudaLaunchKernel(kernelFunc, launchGrid, dim3{64U, 1U, 1U}, kernelArgs, 0, params.stream);
            TLLM_CHECK_WITH_INFO(status == cudaSuccess, "%s", cudaGetErrorString(status));
            status = cudaPeekAtLastError();
            TLLM_CHECK_WITH_INFO(status == cudaSuccess, "%s", cudaGetErrorString(status));
        };

        switch (params.vStage)
        {
        case 0: launchKernel(cute::false_type{}, Int<0>{}); return;
        case 1:
            if (params.kSmooth)
            {
                launchKernel(cute::true_type{}, Int<1>{});
            }
            else
            {
                launchKernel(cute::false_type{}, Int<1>{});
            }
            return;
        case 2:
            if (params.kSmooth)
            {
                launchKernel(cute::true_type{}, Int<2>{});
            }
            else
            {
                launchKernel(cute::false_type{}, Int<2>{});
            }
            return;
        default: TLLM_THROW("Unsupported SageQuantV stage %d", params.vStage);
        }
    };

#define TLLM_SAGE_DISPATCH_HEAD_DIM(HEAD_DIM)                                                                          \
    if (params.headDim == HEAD_DIM)                                                                                    \
    {                                                                                                                  \
        switch (params.partition)                                                                                      \
        {                                                                                                              \
        case SageScalePartition::HopperQ: invokeKernel(Int<HEAD_DIM>{}, HopperQPartition{}); return;                   \
        case SageScalePartition::HopperK: invokeKernel(Int<HEAD_DIM>{}, HopperKPartition{}); return;                   \
        case SageScalePartition::Contiguous:                                                                           \
            switch (params.tokenBlockSize)                                                                             \
            {                                                                                                          \
            case 1: invokeKernel(Int<HEAD_DIM>{}, ContiguousPartition<1>{}); return;                                   \
            case 4: invokeKernel(Int<HEAD_DIM>{}, ContiguousPartition<4>{}); return;                                   \
            case 16: invokeKernel(Int<HEAD_DIM>{}, ContiguousPartition<16>{}); return;                                 \
            default: break;                                                                                            \
            }                                                                                                          \
            break;                                                                                                     \
        }                                                                                                              \
    }
    TLLM_SAGE_DISPATCH_HEAD_DIM(64)
    TLLM_SAGE_DISPATCH_HEAD_DIM(128)
    TLLM_SAGE_DISPATCH_HEAD_DIM(256)
#undef TLLM_SAGE_DISPATCH_HEAD_DIM
    TLLM_THROW(
        "Unsupported SageQuant dispatch config (head_dim must be 64, 128, or 256; contiguous token_block_size must "
        "be 1, 4, or 16): headDim=%d partition=%d tokenBlockSize=%d",
        params.headDim, static_cast<int>(params.partition), params.tokenBlockSize);
}

void invokeSageQuant(SageQuantParams const& params)
{
    if (params.inputType == DATA_TYPE_FP16)
    {
        invokeSageQuantQkvImpl<cutlass::half_t>(params);
        return;
    }
    if (params.inputType == DATA_TYPE_BF16)
    {
        invokeSageQuantQkvImpl<cutlass::bfloat16_t>(params);
        return;
    }
    TLLM_THROW("SageQuant input must be FP16 or BF16");
}

namespace
{

// Evaluate `fn` with the PartitionTraits the (partition, tokenBlockSize) pair selects, or return
// `fallback` when there is no such instantiation.
template <typename Fn>
int withPartitionTraits(SageScalePartition partition, int tokenBlockSize, int fallback, Fn&& fn)
{
    switch (partition)
    {
    case SageScalePartition::HopperQ: return fn(PartitionTraits<HopperQPartition>{});
    case SageScalePartition::HopperK: return fn(PartitionTraits<HopperKPartition>{});
    case SageScalePartition::Contiguous:
        switch (tokenBlockSize)
        {
        case 1: return fn(PartitionTraits<ContiguousPartition<1>>{});
        case 4: return fn(PartitionTraits<ContiguousPartition<4>>{});
        case 16: return fn(PartitionTraits<ContiguousPartition<16>>{});
        default: break;
        }
        break;
    }
    return fallback;
}

} // namespace

int getSageScaleHeadStride(SageScalePartition partition, int tokenBlockSize, int sumSeqLens, int batchSize)
{
    if (tokenBlockSize <= 0 || sumSeqLens <= 0 || batchSize <= 0)
    {
        return 0;
    }
    return withPartitionTraits(partition, tokenBlockSize, 0,
        [&](auto traits) { return decltype(traits)::scaleHeadStride(sumSeqLens, batchSize); });
}

} // namespace tensorrt_llm::common
