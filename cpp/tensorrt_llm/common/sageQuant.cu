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

// Most tokens of one Q/K scale group a thread holds in registers at once.
constexpr int MaxChunkTokens = 8;

// Largest FP8 E4M3 magnitude. A V scale maps the largest magnitude of its channel onto it, with a
// floor that keeps the reciprocal of a constant channel's scale finite; the smoothed and unsmoothed
// V scales share both.
constexpr float E4M3Max = 448.0f;
constexpr float VScaleFloor = 1e-3f;

// The V tokens one CTA row reduces for the V scales and the K mean, and the scale row they go to.
// blockIdx.y is (headIdx * batchSize + seqIdx) when the statistics are per sequence; otherwise
// rows below numHeadsV each cover one head over the whole batch.
struct SageStatRange
{
    int headIdx;
    int tokBegin;
    int tokEnd;
    int row;
};

__device__ inline SageStatRange getSageStatRange(
    int const* ptrCuSeqLensV, int batchSize, int numHeadsV, int sumSeqLensV)
{
    if (ptrCuSeqLensV == nullptr)
    {
        int const headIdx = blockIdx.y;
        return {headIdx, 0, sumSeqLensV, headIdx};
    }
    int const headIdx = blockIdx.y / batchSize;
    int const seqIdx = blockIdx.y % batchSize;
    return {headIdx, ptrCuSeqLensV[seqIdx], ptrCuSeqLensV[seqIdx + 1], seqIdx * numHeadsV + headIdx};
}

// Float atomic maximum and minimum through the integer orderings of IEEE-754 bit patterns: floats
// with a clear sign bit order like signed integers, floats with a set sign bit (-0.0 included)
// reversed like unsigned integers.
__device__ inline void sageAtomicMax(float* address, float value)
{
    if (__float_as_int(value) >= 0)
    {
        atomicMax(reinterpret_cast<int*>(address), __float_as_int(value));
    }
    else
    {
        atomicMin(reinterpret_cast<unsigned int*>(address), __float_as_uint(value));
    }
}

__device__ inline void sageAtomicMin(float* address, float value)
{
    if (__float_as_int(value) >= 0)
    {
        atomicMin(reinterpret_cast<int*>(address), __float_as_int(value));
    }
    else
    {
        atomicMax(reinterpret_cast<unsigned int*>(address), __float_as_uint(value));
    }
}

// Warps that reduce one row of the V statistics or the K mean.
constexpr int MaxStatWarps = 256;

// The smoothing means are summed in fixed point with 32 fraction bits: each warp adds its partial
// mean, rounded once, and integer additions are exact, so the mean does not depend on the order the
// atomics land in, as it would with float atomics. A partial mean is bounded by the largest
// |input|; clamping it to MaxPartialMean keeps MaxStatWarps of them within int64, and keeps the
// power-of-two scalings below exact. Any mean is exact for smoothing as long as every consumer uses
// the same one: softmax ignores a constant subtracted from all keys, and V's mean is added back.
constexpr float MeanFixedScale = static_cast<float>(1LL << 32);
constexpr float MaxPartialMean = static_cast<float>(1LL << 62) / MeanFixedScale / MaxStatWarps;

__device__ inline void sageMeanAdd(int64_t* sum, float partialMean)
{
    float const clamped = ::fminf(::fmaxf(partialMean, -MaxPartialMean), MaxPartialMean);
    long long const fixed = __float2ll_rn(clamped * MeanFixedScale);
    atomicAdd(reinterpret_cast<unsigned long long*>(sum), static_cast<unsigned long long>(fixed));
}

// Starts the reductions of a VStage=1 launch from their identities: zero for the mean sums and for
// the V scales, which collect absolute maxima; with V smoothing, -inf for the signed maxima the V
// scale buffer collects instead and +inf for the minima. A null sum is skipped.
__global__ void sageInitStatsKernel(int64_t* kSum, int64_t* vSum, float* vScale, float* vMin, bool smoothV, int numel)
{
    int const i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numel)
    {
        return;
    }
    if (kSum != nullptr)
    {
        kSum[i] = 0;
    }
    if (vSum != nullptr)
    {
        vSum[i] = 0;
    }
    vScale[i] = smoothV ? -INFINITY : 0.0f;
    if (smoothV)
    {
        vMin[i] = INFINITY;
    }
}

// Converts the summed means of a VStage=1 launch into the float means the VStage=2 launch and the
// consumers read; a null sum is skipped. With V smoothing, vScale holds the per-channel maximum of V
// and vMin the minimum; the maximum becomes the scale of V minus its mean, which covers the larger
// distance from the mean, with the floor of the unsmoothed scales.
__global__ void sageMeanFromSumKernel(
    int64_t const* kSum, float* kMean, int64_t const* vSum, float* vMean, float* vScale, float const* vMin, int numel)
{
    int const i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numel && kSum != nullptr)
    {
        kMean[i] = __ll2float_rn(kSum[i]) * (1.0f / MeanFixedScale);
    }
    if (i < numel && vSum != nullptr)
    {
        float const mean = __ll2float_rn(vSum[i]) * (1.0f / MeanFixedScale);
        vMean[i] = mean;
        vScale[i] = ::fmaxf(::fmaxf(vScale[i] - mean, mean - vMin[i]), VScaleFloor) / E4M3Max;
    }
}

namespace
{

constexpr int StatThreads = 256;

// The V scales and the means hold headDim values per sequence and head, or per head without
// ptrCuSeqLensV.
int getSageStatNumel(SageQuantParams const& params)
{
    return (params.ptrCuSeqLensV != nullptr ? params.batchSize : 1) * params.numHeadsV * params.headDim;
}

} // namespace

// SageAttention quantization kernel. Tensors are interpreted as column-major [D, H, S], which is
// the same physical layout as contiguous PyTorch [S, H, D]. Each launch quantizes Q or K. It can
// simultaneously collect V scales (VStage=1) or quantize V using scales collected by an earlier
// launch (VStage=2). When SmoothK is enabled, the VStage=1 launch also collects a per-head-per-
// channel K mean and the VStage=2 launch uses it to perform K-smoothing before quantization. When
// SmoothV is enabled, the VStage=1 launch collects the per-channel V mean and signed extrema, the
// maximum in ptrVScale itself, and sageMeanFromSumKernel turns them into the V scales; the VStage=2
// launch subtracts ptrVMean from V whenever it is given. The VStage=1 launch sums the means into
// ptrKMeanSum and ptrVMeanSum (see sageMeanAdd).
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
//
// V scales and the K mean are reduced per sequence when ptrCuSeqLensV is given (row
// seqIdx * numHeadsV + headIdx) and over the whole batch otherwise (row headIdx).
template <typename Element, typename ElementQuantized, typename Partition, int HeadDim, bool SmoothK, bool SmoothV,
    int VStage>
__global__ void sageQuantQkvKernel(int sumSeqLensQk, int batchSize, int const* ptrCuSeqLensQk, void const* ptrQk,
    void* ptrQkQuant, float* ptrQkScale, void const* ptrKForMean, int64_t* ptrKMeanSum, float* ptrKMean,
    int sumSeqLensV, int numHeadsV, void const* ptrV, void* ptrVQuant, float* ptrVScale, int const* ptrCuSeqLensV,
    int64_t* ptrVMeanSum, float* ptrVMean, float* ptrVMin)
{
    using namespace cute;
    using namespace cutlass;
    using Traits = PartitionTraits<Partition>;
    constexpr int TokensPerScale = Traits::TokensPerScale;
    constexpr int ScalesPerTile = Traits::ScalesPerTile;
    constexpr int TileTokens = Traits::TileTokens;
    static_assert(!SmoothK || VStage != 0, "K smoothing requires V staging");
    static_assert(!SmoothV || VStage == 1, "V smoothing only changes the statistics launch");
    static_assert(std::is_same_v<ElementQuantized, float_e4m3_t> || std::is_same_v<ElementQuantized, std::int8_t>,
        "Unrecognized target dtype for quantization");
    constexpr float TypeMax = cute::is_same_v<ElementQuantized, float_e4m3_t> ? E4M3Max : static_cast<float>(126.9f);
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

        // This thread's channel slice of the K mean of this sequence (or of the whole batch).
        Tensor rKMean = make_tensor<float>(Shape<VL>{});
        if constexpr (SmoothK && VStage == 2)
        {
            int const meanRow = ptrCuSeqLensV != nullptr ? seqIdx * numHeadsV + headIdx : headIdx;
            Tensor gKMean = make_tensor(
                make_gmem_ptr(ptrKMean + static_cast<int64_t>(meanRow) * HeadDim + threadInScaleIdx * BestVL),
                Shape<VL>{});
            cute::copy(AutoVectorizingCopy{}, gKMean, rKMean);
        }

        // A thread keeps at most ChunkTokens tokens of a group in registers, in the input type; FP32
        // values are recomputed from them when needed. Larger groups are read in chunks twice: once
        // to find the group maximum, once to quantize.
        constexpr int ChunkTokens = TokensPerScale < MaxChunkTokens ? TokensPerScale : MaxChunkTokens;
        constexpr int NumChunks = TokensPerScale / ChunkTokens;
        static_assert(TokensPerScale % ChunkTokens == 0, "a group must split into whole chunks");
        static_assert(BestVL % 4 == 0, "quantized values are converted four at a time");

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
            auto isValid = [&](int chunk, int i) { return IsFullTile || tokenOf(chunk * ChunkTokens + i) < seqLen; };

            Tensor rQk = make_tensor<Element>(Shape<VL, Int<ChunkTokens>>{});
            Tensor rQkQuant = make_tensor<ElementQuantized>(Shape<VL, Int<ChunkTokens>>{});
            Tensor rQkQuant_x4 = recast<Array<ElementQuantized, 4>>(rQkQuant);

            auto loadChunk = [&](int chunk)
            {
                CUTLASS_PRAGMA_UNROLL
                for (int i = 0; i < ChunkTokens; ++i)
                {
                    int const tokenIdx = tokenOf(chunk * ChunkTokens + i);
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
            };
            // FP32 value of channel j of token i of the loaded chunk, K-smoothed.
            auto valueAt = [&](int chunk, int i, int j)
            {
                float x = static_cast<float>(rQk(j, i));
                if constexpr (SmoothK && VStage == 2)
                {
                    if (isValid(chunk, i))
                    {
                        x -= rKMean(j);
                    }
                }
                return x;
            };
            // Scaling and conversion stay in FP32: converting a BF16/FP16 product to INT8 truncates
            // instead of rounding.
            auto quantizeChunk = [&](int chunk, float invScale)
            {
                CUTLASS_PRAGMA_UNROLL
                for (int i = 0; i < ChunkTokens; ++i)
                {
                    CUTLASS_PRAGMA_UNROLL
                    for (int j = 0; j < BestVL; j += 4)
                    {
                        Array<float, 4> scaled;
                        CUTLASS_PRAGMA_UNROLL
                        for (int e = 0; e < 4; ++e)
                        {
                            scaled[e] = valueAt(chunk, i, j + e) * invScale;
                        }
                        rQkQuant_x4((i * BestVL + j) / 4)
                            = NumericArrayConverter<ElementQuantized, float, 4>::convert(scaled);
                    }
                }
            };
            auto storeChunk = [&](int chunk)
            {
                CUTLASS_PRAGMA_UNROLL
                for (int i = 0; i < ChunkTokens; ++i)
                {
                    int const tokenIdx = tokenOf(chunk * ChunkTokens + i);
                    if (IsFullTile || tokenIdx < seqLen)
                    {
                        Tensor gDst = make_tensor(
                            make_gmem_ptr(ptrQkQuantThread + static_cast<int64_t>(tokenIdx) * tokenStride),
                            Shape<VL>{});
                        cute::copy(AutoVectorizingCopy{}, rQkQuant(_, i), gDst);
                    }
                }
            };

            float maxScale = 1e-3f;
            // Both chunk loops stay rolled: unrolling them multiplies the code of the large blocks by
            // the chunk count.
            CUTLASS_PRAGMA_NO_UNROLL
            for (int chunk = 0; chunk < NumChunks; ++chunk)
            {
                loadChunk(chunk);
                CUTLASS_PRAGMA_UNROLL
                for (int i = 0; i < ChunkTokens; ++i)
                {
                    CUTLASS_PRAGMA_UNROLL
                    for (int j = 0; j < BestVL; ++j)
                    {
                        maxScale = ::fmaxf(maxScale, ::fabsf(valueAt(chunk, i, j)));
                    }
                }
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
            CUTLASS_PRAGMA_NO_UNROLL
            for (int chunk = 0; chunk < NumChunks; ++chunk)
            {
                if constexpr (NumChunks > 1)
                {
                    // The single-chunk case still holds its tokens from the maximum pass.
                    loadChunk(chunk);
                }
                quantizeChunk(chunk, invScale);
                storeChunk(chunk);
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
        SageStatRange const range = getSageStatRange(ptrCuSeqLensV, batchSize, numHeadsV, sumSeqLensV);
        int const headIdx = range.headIdx;
        using ElementQuantizedV = cutlass::float_e4m3_t;
        constexpr int threadsPerHead = HeadDim / BestVL;
        static_assert(HeadDim % BestVL == 0, "VL must divide HeadDim");
        static_assert(threadsPerHead <= 32, "One token block should never exceed warp scope");
        Tensor gV = make_tensor(
            reinterpret_cast<Element const*>(ptrV), make_shape(VL{}, Int<threadsPerHead>{}, numHeadsV, sumSeqLensV));
        Tensor gVQuant = make_tensor(reinterpret_cast<ElementQuantizedV*>(ptrVQuant),
            make_shape(VL{}, Int<threadsPerHead>{}, numHeadsV, sumSeqLensV));
        Tensor gVScale = make_tensor(
            ptrVScale + static_cast<int64_t>(range.row) * HeadDim, make_shape(VL{}, Int<threadsPerHead>{}));

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
            int tokIdx = range.tokBegin + warpId * numToksPerWarp + thrId / threadsPerHead;
            int const threadInTokIdx = thrId % threadsPerHead;
            Tensor gVSeq = gV(_, threadInTokIdx, headIdx, _);
            Tensor gVSeqQuant = gVQuant(_, threadInTokIdx, headIdx, _);
            Tensor gVSeqScale = gVScale(_, threadInTokIdx);

            if constexpr (VStage == 1 && SmoothV)
            {
                int const numWarpsToUse = cutlass::fast_min(numWarps, MaxStatWarps);
                int const numToksPerWave = numWarpsToUse * numToksPerWarp;
                if (warpId >= numWarpsToUse)
                {
                    return;
                }
                // Per-channel sum and signed extrema, in FP32.
                float rSum[BestVL];
                float rMax[BestVL];
                float rMin[BestVL];
                CUTLASS_PRAGMA_UNROLL
                for (int i = 0; i < BestVL; ++i)
                {
                    rSum[i] = 0.0f;
                    rMax[i] = -INFINITY;
                    rMin[i] = INFINITY;
                }
                for (; tokIdx < range.tokEnd; tokIdx += numToksPerWave)
                {
                    cute::copy(AutoVectorizingCopy{}, gVSeq(_, tokIdx), rV);
                    cute::transform(rV_x2, rVCompute_x2, NumericArrayConverter<float, Element, 2>::convert);
                    CUTLASS_PRAGMA_UNROLL
                    for (int i = 0; i < BestVL; ++i)
                    {
                        rSum[i] += rVCompute(i);
                        rMax[i] = ::fmaxf(rMax[i], rVCompute(i));
                        rMin[i] = ::fminf(rMin[i], rVCompute(i));
                    }
                }
                for (int delta = threadsPerHead; delta < 32; delta <<= 1)
                {
                    CUTLASS_PRAGMA_UNROLL
                    for (int i = 0; i < BestVL; ++i)
                    {
                        rSum[i] += __shfl_xor_sync(0xffffffffu, rSum[i], delta);
                        rMax[i] = ::fmaxf(rMax[i], __shfl_xor_sync(0xffffffffu, rMax[i], delta));
                        rMin[i] = ::fminf(rMin[i], __shfl_xor_sync(0xffffffffu, rMin[i], delta));
                    }
                }
                // A smoothed V's scales cover V minus its mean, which is final only after this launch: the
                // maximum goes to the scale buffer, and sageMeanFromSumKernel turns it into the scale.
                if (threadInTokIdx == thrId && range.tokBegin < range.tokEnd)
                {
                    int64_t const statOffset = static_cast<int64_t>(range.row) * HeadDim + threadInTokIdx * BestVL;
                    float const invNumTokens = 1.0f / (range.tokEnd - range.tokBegin);
                    CUTLASS_PRAGMA_UNROLL
                    for (int i = 0; i < BestVL; ++i)
                    {
                        sageMeanAdd(ptrVMeanSum + statOffset + i, rSum[i] * invNumTokens);
                        sageAtomicMax(ptrVScale + statOffset + i, rMax[i]);
                        sageAtomicMin(ptrVMin + statOffset + i, rMin[i]);
                    }
                }
            }
            else if constexpr (VStage == 1)
            {
                int const numWarpsToUse = cutlass::fast_min(numWarps, MaxStatWarps);
                int const numToksPerWave = numWarpsToUse * numToksPerWarp;
                if (warpId >= numWarpsToUse)
                {
                    return;
                }
                CUTLASS_PRAGMA_UNROLL
                for (int i = 0; i < size(rVScale); ++i)
                {
                    rVScale(i) = VScaleFloor;
                }
                cute::transform(rVScale_x2, rVMax_x2, cutlass::NumericArrayConverter<Element, float, 2>::convert);
                for (; tokIdx < range.tokEnd; tokIdx += numToksPerWave)
                {
                    cute::copy(AutoVectorizingCopy{}, gVSeq(_, tokIdx), rV);
                    cute::transform(rV_x2, rV_x2, cutlass::absolute_value_op<Array<Element, 2>>{});
                    cute::transform(rV_x2, rVMax_x2, rVMax_x2, cutlass::maximum<Array<Element, 2>>{});
                }
                cute::transform(rVMax_x2, rVScale_x2, cutlass::NumericArrayConverter<float, Element, 2>::convert);
                cute::transform(rVScale_x2, rVScale_x2, cutlass::scale<Array<float, 2>>{1.0f / E4M3Max});
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
                int64_t const statOffset = static_cast<int64_t>(range.row) * HeadDim + threadInTokIdx * BestVL;
                // This thread's channel slice of the V mean, zero without V smoothing.
                Tensor rVMean = make_tensor<float>(Shape<VL>{});
                clear(rVMean);
                if (ptrVMean != nullptr)
                {
                    cute::copy(
                        AutoVectorizingCopy{}, make_tensor(make_gmem_ptr(ptrVMean + statOffset), Shape<VL>{}), rVMean);
                }
                cute::copy(AutoVectorizingCopy{}, gVSeqScale, rVScale);
                cute::transform(rVScale_x2, rVScale_x2, cutlass::reciprocal_approximate<Array<float, 2>>{});
                for (; tokIdx < range.tokEnd; tokIdx += numToksPerWave)
                {
                    cute::copy(AutoVectorizingCopy{}, gVSeq(_, tokIdx), rV);
                    cute::transform(rV_x2, rVCompute_x2, cutlass::NumericArrayConverter<float, Element, 2>::convert);
                    CUTLASS_PRAGMA_UNROLL
                    for (int i = 0; i < BestVL; ++i)
                    {
                        rVCompute(i) = (rVCompute(i) - rVMean(i)) * rVScale(i);
                    }
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
            SageStatRange const range = getSageStatRange(ptrCuSeqLensV, batchSize, numHeadsV, sumSeqLensV);
            int const headIdx = range.headIdx;
            constexpr int threadsPerHead = HeadDim / BestVL;
            static_assert(HeadDim % BestVL == 0, "VL must divide HeadDim");
            static_assert(threadsPerHead <= 32, "One token block should never exceed warp scope");

            if (headIdx < numHeadsV && range.tokBegin < range.tokEnd)
            {
                Tensor gK = make_tensor(reinterpret_cast<Element const*>(ptrKForMean),
                    make_shape(VL{}, Int<threadsPerHead>{}, numHeadsV, sumSeqLensV));
                Tensor rK = make_tensor<Element>(Shape<VL>{});
                Tensor rKCompute = make_tensor<float>(Shape<VL>{});
                Tensor rKSum = make_tensor<float>(Shape<VL>{});
                Tensor rK_x2 = recast<Array<Element, 2>>(rK);
                Tensor rKCompute_x2 = recast<Array<float, 2>>(rKCompute);

                int const numToksPerWarp = 32 / threadsPerHead;
                int const numWarpsToUse = cutlass::fast_min(numWarps, MaxStatWarps);
                int const numToksPerWave = numWarpsToUse * numToksPerWarp;
                if (warpId >= numWarpsToUse)
                {
                    return;
                }
                int tokIdx = range.tokBegin + warpId * numToksPerWarp + thrId / threadsPerHead;
                int const threadInTokIdx = thrId % threadsPerHead;
                Tensor gKSeq = gK(_, threadInTokIdx, headIdx, _);

                clear(rKSum);
                for (; tokIdx < range.tokEnd; tokIdx += numToksPerWave)
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
                    int64_t const statOffset = static_cast<int64_t>(range.row) * HeadDim + threadInTokIdx * BestVL;
                    float const invNumTokens = 1.0f / (range.tokEnd - range.tokBegin);
                    CUTLASS_PRAGMA_UNROLL
                    for (int i = 0; i < BestVL; ++i)
                    {
                        sageMeanAdd(ptrKMeanSum + statOffset + i, rKSum(i) * invNumTokens);
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
    TLLM_CHECK_WITH_INFO(!params.kSmooth
            || (params.ptrKMean != nullptr
                && (params.vStage != 1 || (params.ptrKForMean != nullptr && params.ptrKMeanSum != nullptr))),
        "Invalid SageQuant K-smoothing parameters");
    TLLM_CHECK_WITH_INFO(
        params.ptrCuSeqLensV == nullptr || params.vStage != 0, "SageQuant per-sequence V statistics require V staging");
    TLLM_CHECK_WITH_INFO(params.ptrVMean == nullptr
            || (params.vStage != 0
                && (params.vStage != 1 || (params.ptrVMeanSum != nullptr && params.ptrVMin != nullptr))),
        "Invalid SageQuant V-smoothing parameters");
    // The mean conversion after a VStage=1 launch writes the mean of every sum it is given, and for V
    // also reads the minimum.
    TLLM_CHECK_WITH_INFO(
        params.ptrKMeanSum == nullptr || params.ptrKMean != nullptr, "Invalid SageQuant K-smoothing parameters");
    TLLM_CHECK_WITH_INFO(params.ptrVMeanSum == nullptr || (params.ptrVMean != nullptr && params.ptrVMin != nullptr),
        "Invalid SageQuant V-smoothing parameters");

    if (params.vStage == 1)
    {
        // The VStage=1 launch reduces into the mean sums and the V statistics with atomics.
        int const numel = getSageStatNumel(params);
        sageInitStatsKernel<<<static_cast<unsigned>(divUp(numel, StatThreads)), StatThreads, 0, params.stream>>>(
            params.ptrKMeanSum, params.ptrVMeanSum, params.ptrVScale, params.ptrVMin, params.ptrVMean != nullptr,
            numel);
        TLLM_CUDA_CHECK(cudaPeekAtLastError());
    }

    auto invokeKernel = [&](auto headDimStatic, auto partitionStatic)
    {
        constexpr int HeadDim_ = headDimStatic;
        using Partition_ = decltype(partitionStatic);
        SageQuantParams kernelParams = params;
        void* kernelArgs[] = {&kernelParams.sumSeqLensQk, &kernelParams.batchSize, &kernelParams.ptrCuSeqLensQk,
            &kernelParams.ptrQk, &kernelParams.ptrQkQuant, &kernelParams.ptrQkScale, &kernelParams.ptrKForMean,
            &kernelParams.ptrKMeanSum, &kernelParams.ptrKMean, &kernelParams.sumSeqLensV, &kernelParams.numHeadsV,
            &kernelParams.ptrV, &kernelParams.ptrVQuant, &kernelParams.ptrVScale, &kernelParams.ptrCuSeqLensV,
            &kernelParams.ptrVMeanSum, &kernelParams.ptrVMean, &kernelParams.ptrVMin};

        auto launchKernel = [&](auto smoothKStatic, auto smoothVStatic, auto vStageStatic)
        {
            constexpr bool SmoothK_ = decltype(smoothKStatic)::value;
            constexpr bool SmoothV_ = decltype(smoothVStatic)::value;
            constexpr int VStage_ = vStageStatic;
            void const* kernelFunc = nullptr;
            if (params.quantType == DATA_TYPE_E4M3)
            {
                kernelFunc = reinterpret_cast<void const*>(sageQuantQkvKernel<Element, cutlass::float_e4m3_t,
                    Partition_, HeadDim_, SmoothK_, SmoothV_, VStage_>);
            }
            else if (params.quantType == DATA_TYPE_INT8)
            {
                kernelFunc = reinterpret_cast<void const*>(
                    sageQuantQkvKernel<Element, std::int8_t, Partition_, HeadDim_, SmoothK_, SmoothV_, VStage_>);
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

        // K smoothing needs V staging, and V smoothing only changes what the VStage==1 launch
        // collects, so only those combinations are instantiated.
        auto launchWithSmoothKV = [&](auto vStageStatic)
        {
            constexpr int VStage_ = vStageStatic;
            auto launchWithSmoothV = [&](auto smoothKStatic)
            {
                if constexpr (VStage_ == 1)
                {
                    if (params.ptrVMean != nullptr)
                    {
                        launchKernel(smoothKStatic, cute::true_type{}, vStageStatic);
                        return;
                    }
                }
                launchKernel(smoothKStatic, cute::false_type{}, vStageStatic);
            };
            if constexpr (VStage_ != 0)
            {
                if (params.kSmooth)
                {
                    launchWithSmoothV(cute::true_type{});
                    return;
                }
            }
            launchWithSmoothV(cute::false_type{});
        };

        switch (params.vStage)
        {
        case 0: launchWithSmoothKV(Int<0>{}); return;
        case 1: launchWithSmoothKV(Int<1>{}); return;
        case 2: launchWithSmoothKV(Int<2>{}); return;
        default: TLLM_THROW("Unsupported SageQuantV stage %d", params.vStage);
        }
    };

    // Contiguous blocks are instantiated for the sizes in use. The kernel takes any power of two up to
    // 256: add its case here and in withPartitionTraits.
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
            case 32: invokeKernel(Int<HEAD_DIM>{}, ContiguousPartition<32>{}); return;                                 \
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
        "be 1, 4, 16, or 32): headDim=%d partition=%d tokenBlockSize=%d",
        params.headDim, static_cast<int>(params.partition), params.tokenBlockSize);
}

void invokeSageQuant(SageQuantParams const& params)
{
    if (params.inputType == DATA_TYPE_FP16)
    {
        invokeSageQuantQkvImpl<cutlass::half_t>(params);
    }
    else if (params.inputType == DATA_TYPE_BF16)
    {
        invokeSageQuantQkvImpl<cutlass::bfloat16_t>(params);
    }
    else
    {
        TLLM_THROW("SageQuant input must be FP16 or BF16");
    }
    if (params.vStage == 1 && (params.ptrKMeanSum != nullptr || params.ptrVMeanSum != nullptr))
    {
        int const numel = getSageStatNumel(params);
        sageMeanFromSumKernel<<<static_cast<unsigned>(divUp(numel, StatThreads)), StatThreads, 0, params.stream>>>(
            params.ptrKMeanSum, params.ptrKMean, params.ptrVMeanSum, params.ptrVMean, params.ptrVScale, params.ptrVMin,
            numel);
        TLLM_CUDA_CHECK(cudaPeekAtLastError());
    }
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
        case 32: return fn(PartitionTraits<ContiguousPartition<32>>{});
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
