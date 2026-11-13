/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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

// The MiniMax-M3 sparse horizontal producers (kernels/fusedQKNormRopeKernel.cu's
// minimaxM3Nvfp4QKVIndexerNormRopeKVInsertKernel for an NVFP4 main KV cache, with the early PDL trigger, and
// minimaxM3Fp8QKVIndexerNormRopeKVInsertKernel for an FP8 one) with the index-K output modes of
// nvfp4QkvIndexerProducer.h.

#include "minimax_m3/csrc/m31/idxkNvfp4Quant.cuh"
#include "minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__dep/csrc/nvfp4QkvIndexerProducer.h"
#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaTypeUtils.cuh"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/envUtils.h"
#include "tensorrt_llm/common/mathUtils.h"
#include "tensorrt_llm/common/reduceKernelUtils.cuh"
#include "tensorrt_llm/kernels/quantization.cuh"
#include "tensorrt_llm/modeling_v2/common/namespace.h"

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

TRTLLM_MV2_NAMESPACE_BEGIN

namespace kernels
{

namespace
{

template <int numElemsPerThread>
__device__ __forceinline__ void storeFp8HeadElements64(
    __nv_fp8_e4m3* out, int64_t offsetThread, float const (&elements)[numElemsPerThread])
{
    static_assert(numElemsPerThread == 4, "MiniMax-M3 FP8 store expects four elements per thread");
    static_assert(sizeof(__nv_fp8x2_storage_t) == 2, "MiniMax-M3 FP8 pair storage must be 16 bits");
    // Form the final pointer with 64-bit arithmetic before one aligned 32-bit
    // store. Production coalesced paged-cache offsets can exceed INT32_MAX
    // FP8 elements even though each individual head row is small.
    auto* threadOut = out + offsetThread;
    __nv_fp8x2_e4m3 const low(make_float2(elements[0], elements[1]));
    __nv_fp8x2_e4m3 const high(make_float2(elements[2], elements[3]));
    uint32_t const packed = static_cast<uint32_t>(low.__x) | (static_cast<uint32_t>(high.__x) << 16);
    *reinterpret_cast<uint32_t*>(threadOut) = packed;
}

// Quantize the four BF16-rounded values owned by one lane into two packed
// E2M1 bytes.  Four adjacent lanes share one E4M3 block scale, matching the
// production fp4_quantize contract (SF_VEC_SIZE=16) without redistributing the
// horizontal producer's existing four-values-per-lane Q/K/V register layout.
__device__ __forceinline__ uint16_t quantizeMinimaxM3Fp4x4(
    float const (&elements)[4], float secondLevelScale, uint8_t* scaleOut)
{
    __nv_bfloat16 rounded[4];
    float localMax = 0.0F;
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        rounded[i] = __float2bfloat16_rn(elements[i]);
        localMax = fmaxf(localMax, fabsf(__bfloat162float(rounded[i])));
    }

    uint32_t const groupMask = tensorrt_llm::kernels::cvt_sf_group_shfl_mask<4>();
    localMax = fmaxf(localMax, __shfl_xor_sync(groupMask, localMax, 1));
    localMax = fmaxf(localMax, __shfl_xor_sync(groupMask, localMax, 2));

    float const scaleValue = secondLevelScale * (localMax * tensorrt_llm::kernels::reciprocal_approximate_ftz(6.0F));
    __nv_fp8_e4m3 const encodedScale(scaleValue);
    if (scaleOut != nullptr)
    {
        *scaleOut = encodedScale.__x;
    }
    float const narrowedScale = static_cast<float>(encodedScale);
    float const outputScale = localMax != 0.0F ? tensorrt_llm::kernels::reciprocal_approximate_ftz(
                                  narrowedScale * tensorrt_llm::kernels::reciprocal_approximate_ftz(secondLevelScale))
                                               : 0.0F;

    float quantValues[8] = {};
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        quantValues[i] = __bfloat162float(rounded[i]) * outputScale;
    }
    return static_cast<uint16_t>(tensorrt_llm::kernels::fp32_vec_to_e2m1(quantValues));
}

constexpr int kMinimaxM3HeadDim = 128;
constexpr int kMinimaxM3RotaryDim = 64;
constexpr int kMinimaxM3PageSize = 128;
static_assert((kMinimaxM3PageSize & (kMinimaxM3PageSize - 1)) == 0, "page size must be a power of two");
constexpr int kMinimaxM3ElemsPerThread = kMinimaxM3HeadDim / 32;
static_assert(kMinimaxM3ElemsPerThread == 4, "the NVFP4 index-K store takes four elements per lane");

// The horizontal producers' index-K store for kIdxkMode != kMinimaxM3IndexKFp8Sink (mode 0 keeps its inline E4M3
// store). indexKCache is the raw cache base; the mode names its element type and indexPageStride's unit.
//   kMinimaxM3IndexKNvfp4Pages: the index-K rounded to BF16, then quantized into the token's NVFP4 index-K page
//     (byte page stride; idxkNvfp4QuantStore, the writer's recipe). The whole warp must call it.
//   kMinimaxM3IndexKBf16Tap (tests): the same BF16-rounded values as a [pages, 1, 128, 128] BF16 row (element
//     strides), the rows the writer turns into the NVFP4 mode's bytes.
template <int kIdxkMode>
__device__ __forceinline__ void storeMinimaxM3IndexKDirect(__nv_fp8_e4m3* indexKCache, int page, int withinPage,
    int64_t indexPageStride, int64_t indexTokenStride, int laneId, float const (&elements)[kMinimaxM3ElemsPerThread])
{
    static_assert(
        kIdxkMode == kMinimaxM3IndexKNvfp4Pages || kIdxkMode == kMinimaxM3IndexKBf16Tap, "mode 0 stores E4M3 inline");
    float rounded[kMinimaxM3ElemsPerThread];
    kernels::minimaxM31::idxkNvfp4RoundToBf16(elements, rounded);
    if constexpr (kIdxkMode == kMinimaxM3IndexKNvfp4Pages)
    {
        uint8_t* pageBase = reinterpret_cast<uint8_t*>(indexKCache) + static_cast<int64_t>(page) * indexPageStride;
        kernels::minimaxM31::idxkNvfp4QuantStore(rounded, laneId, pageBase, withinPage);
    }
    else
    {
        int64_t const outputOffset = static_cast<int64_t>(page) * indexPageStride
            + static_cast<int64_t>(withinPage) * indexTokenStride + laneId * kMinimaxM3ElemsPerThread;
        __nv_bfloat162 const low = __floats2bfloat162_rn(rounded[0], rounded[1]);
        __nv_bfloat162 const high = __floats2bfloat162_rn(rounded[2], rounded[3]);
        uint2 packed;
        packed.x = reinterpret_cast<uint32_t const&>(low);
        packed.y = reinterpret_cast<uint32_t const&>(high);
        *reinterpret_cast<uint2*>(reinterpret_cast<__nv_bfloat16*>(indexKCache) + outputOffset) = packed;
    }
}

// The FP8 horizontal sparse producer (kernels/fusedQKNormRopeKernel.cu's minimaxM3Fp8QKVIndexerNormRopeKVInsertKernel)
// with the index-K output modes of nvfp4QkvIndexerProducer.h, for a packed [Q|K|V|index-Q|index-K] row.
// One warp owns one (token, head slot). All four norm/RoPE branches share the
// model's precomputed FP32 RoPE table, eliminating per-head powf/sincos work.
// kIdxkMode selects the index-K store (kMinimaxM3IndexK*); every other output
// is the same in all modes.
template <int kIdxkMode>
__global__ void minimaxM3Fp8QKVIndexerNormRopeKVInsertKernel(__nv_bfloat16 const* packedInput, __nv_fp8_e4m3* qOutput,
    __nv_fp8_e4m3* indexQOutput, __nv_fp8_e4m3* kvCache, __nv_fp8_e4m3* indexKCache, int const* outCacheLoc,
    int64_t kvPageStride, int64_t kvPlaneStride, int64_t kvHeadStride, int64_t kvTokenStride, int64_t indexPageStride,
    int64_t indexTokenStride, int64_t numPages, int numTokens, int numHeadsQ, int numHeadsKV, int numHeadsIndex,
    float eps, __nv_bfloat16 const* qWeight, __nv_bfloat16 const* kWeight, __nv_bfloat16 const* indexQWeight,
    __nv_bfloat16 const* indexKWeight, float const* rotaryCosSin, int const* positionIds)
{
    int const warpsPerBlock = blockDim.x / 32;
    int const warpId = threadIdx.x / 32;
    int const laneId = threadIdx.x % 32;
    int const globalWarp = blockIdx.x * warpsPerBlock + warpId;
    int const totalHeads = numHeadsQ + 2 * numHeadsKV + numHeadsIndex + 1;
    int const tokenIdx = globalWarp / totalHeads;
    int const localHead = globalWarp % totalHeads;
    if (tokenIdx >= numTokens)
    {
        return;
    }

    int const kBegin = numHeadsQ;
    int const vBegin = kBegin + numHeadsKV;
    int const indexQBegin = vBegin + numHeadsKV;
    int const indexKHead = indexQBegin + numHeadsIndex;
    bool const isQ = localHead < kBegin;
    bool const isK = localHead >= kBegin && localHead < vBegin;
    bool const isV = localHead >= vBegin && localHead < indexQBegin;
    bool const isIndexQ = localHead >= indexQBegin && localHead < indexKHead;
    bool const isIndexK = localHead == indexKHead;

    int64_t const inputOffset = (static_cast<int64_t>(tokenIdx) * totalHeads + localHead) * kMinimaxM3HeadDim
        + laneId * kMinimaxM3ElemsPerThread;
    constexpr int kVecSize = kMinimaxM3ElemsPerThread * sizeof(__nv_bfloat16) / 4;
    using VecT = typename tensorrt_llm::common::packed_as<uint, kVecSize>::type;
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    // packedInput is the fused QKV GEMM's output, so wait before the first load
    // of it rather than at entry.
    cudaGridDependencySynchronize();
#endif
    VecT const packed = *reinterpret_cast<VecT const*>(packedInput + inputOffset);

    float elements[kMinimaxM3ElemsPerThread];
    float sumSquares = 0.0F;
#pragma unroll
    for (int pair = 0; pair < kVecSize; ++pair)
    {
        float2 const values = __bfloat1622float2(
            *reinterpret_cast<__nv_bfloat162 const*>(reinterpret_cast<uint const*>(&packed) + pair));
        elements[2 * pair] = values.x;
        elements[2 * pair + 1] = values.y;
        if (!isV)
        {
            sumSquares += values.x * values.x + values.y * values.y;
        }
    }

    if (!isV)
    {
        auto const* normWeight = isQ ? qWeight : (isK ? kWeight : (isIndexQ ? indexQWeight : indexKWeight));
        sumSquares = tensorrt_llm::common::warpReduceSum(sumSquares);
        float const rmsReciprocal = rsqrtf(sumSquares / static_cast<float>(kMinimaxM3HeadDim) + eps);
#pragma unroll
        for (int i = 0; i < kMinimaxM3ElemsPerThread; ++i)
        {
            int const dim = laneId * kMinimaxM3ElemsPerThread + i;
            elements[i] *= rmsReciprocal * (1.0F + __bfloat162float(normWeight[dim]));
        }

        __syncwarp();
        constexpr int kPairOffset = (kMinimaxM3RotaryDim / 2) / kMinimaxM3ElemsPerThread;
        int positionId = laneId == 0 ? positionIds[tokenIdx] : 0;
        positionId = __shfl_sync(0xffffffff, positionId, 0);
        int64_t const ropeRow = static_cast<int64_t>(positionId) * kMinimaxM3RotaryDim;
#pragma unroll
        for (int i = 0; i < kMinimaxM3ElemsPerThread; ++i)
        {
            int const dim = laneId * kMinimaxM3ElemsPerThread + i;
            float paired = __shfl_xor_sync(0xffffffff, elements[i], kPairOffset);
            if (dim < kMinimaxM3RotaryDim)
            {
                bool const firstHalf = dim < kMinimaxM3RotaryDim / 2;
                if (firstHalf)
                {
                    paired = -paired;
                }
                int const coefficient = firstHalf ? dim : dim - kMinimaxM3RotaryDim / 2;
                float const cosine = rotaryCosSin[ropeRow + coefficient];
                float const sine = rotaryCosSin[ropeRow + kMinimaxM3RotaryDim / 2 + coefficient];
                elements[i] = elements[i] * cosine + paired * sine;
            }
        }
        __syncwarp();
    }

    if (isQ)
    {
        int const head = localHead;
        int64_t const outputOffset = (static_cast<int64_t>(tokenIdx) * numHeadsQ + head) * kMinimaxM3HeadDim
            + laneId * kMinimaxM3ElemsPerThread;
        storeFp8HeadElements64<kMinimaxM3ElemsPerThread>(qOutput, outputOffset, elements);
        return;
    }
    if (isIndexQ)
    {
        int const head = localHead - indexQBegin;
        int64_t const outputOffset = (static_cast<int64_t>(tokenIdx) * numHeadsIndex + head) * kMinimaxM3HeadDim
            + laneId * kMinimaxM3ElemsPerThread;
        // Match vLLM's CUDA path: normalized/RoPE FP32 registers convert
        // directly to saturating E4M3, without an intermediate BF16 round.
        storeFp8HeadElements64<kMinimaxM3ElemsPerThread>(indexQOutput, outputOffset, elements);
        return;
    }

    int slot = laneId == 0 ? outCacheLoc[tokenIdx] : 0;
    slot = __shfl_sync(0xffffffff, slot, 0);
    if (slot < 0)
    {
        return;
    }
    int const page = slot / kMinimaxM3PageSize;
    if (page >= numPages)
    {
        return;
    }
    int const withinPage = slot & (kMinimaxM3PageSize - 1);
    if (isIndexK)
    {
        if constexpr (kIdxkMode == kMinimaxM3IndexKFp8Sink)
        {
            int64_t const outputOffset = static_cast<int64_t>(page) * indexPageStride
                + static_cast<int64_t>(withinPage) * indexTokenStride + laneId * kMinimaxM3ElemsPerThread;
            storeFp8HeadElements64<kMinimaxM3ElemsPerThread>(indexKCache, outputOffset, elements);
        }
        else
        {
            storeMinimaxM3IndexKDirect<kIdxkMode>(
                indexKCache, page, withinPage, indexPageStride, indexTokenStride, laneId, elements);
        }
        return;
    }

    int const head = isK ? localHead - kBegin : localHead - vBegin;
    int const plane = isV ? 1 : 0;
    int64_t const outputOffset = static_cast<int64_t>(page) * kvPageStride + static_cast<int64_t>(plane) * kvPlaneStride
        + static_cast<int64_t>(head) * kvHeadStride + static_cast<int64_t>(withinPage) * kvTokenStride
        + laneId * kMinimaxM3ElemsPerThread;
    storeFp8HeadElements64<kMinimaxM3ElemsPerThread>(kvCache, outputOffset, elements);
}

// NVFP4-cache counterpart of the FP8 horizontal sparse producer
// (trtllm::minimax_m3_fp8_qkv_indexer_norm_rope_kv_insert).  Q, index-Q, and index-K retain the exact FP8 register
// epilogue.  Main K/V are rounded to BF16 (matching the existing materialized producer), quantized in four-lane SF16
// groups, and written directly to the packed-data and scale pools.  K scales are token-major; V scales use the
// runtime's 4x4 token-quad swizzle.  kIdxkMode selects the index-K store as in the FP8 producer.
template <int kIdxkMode>
__global__ void minimaxM3Nvfp4QKVIndexerNormRopeKVInsertKernel(__nv_bfloat16 const* packedInput, __nv_fp8_e4m3* qOutput,
    __nv_fp8_e4m3* indexQOutput, uint8_t* kvDataCache, uint8_t* kvScaleCache, __nv_fp8_e4m3* indexKCache,
    int const* outCacheLoc, float const* kvQuantScale, int64_t dataPageStride, int64_t dataPlaneStride,
    int64_t dataHeadStride, int64_t dataTokenStride, int64_t scalePageStride, int64_t scalePlaneStride,
    int64_t scaleHeadStride, int64_t indexPageStride, int64_t indexTokenStride, int64_t numPages, int numTokens,
    int numHeadsQ, int numHeadsKV, int numHeadsIndex, float eps, __nv_bfloat16 const* qWeight,
    __nv_bfloat16 const* kWeight, __nv_bfloat16 const* indexQWeight, __nv_bfloat16 const* indexKWeight,
    float const* rotaryCosSin, int const* positionIds, bool earlyTrigger)
{
    int const warpsPerBlock = blockDim.x / 32;
    int const warpId = threadIdx.x / 32;
    int const laneId = threadIdx.x % 32;
    int const globalWarp = blockIdx.x * warpsPerBlock + warpId;
    int const totalHeads = numHeadsQ + 2 * numHeadsKV + numHeadsIndex + 1;
    int const tokenIdx = globalWarp / totalHeads;
    int const localHead = globalWarp % totalHeads;
    if (tokenIdx >= numTokens)
    {
        return;
    }

    int const kBegin = numHeadsQ;
    int const vBegin = kBegin + numHeadsKV;
    int const indexQBegin = vBegin + numHeadsKV;
    int const indexKHead = indexQBegin + numHeadsIndex;
    bool const isQ = localHead < kBegin;
    bool const isK = localHead >= kBegin && localHead < vBegin;
    bool const isV = localHead >= vBegin && localHead < indexQBegin;
    bool const isIndexQ = localHead >= indexQBegin && localHead < indexKHead;
    bool const isIndexK = localHead == indexKHead;

    int64_t const inputOffset = (static_cast<int64_t>(tokenIdx) * totalHeads + localHead) * kMinimaxM3HeadDim
        + laneId * kMinimaxM3ElemsPerThread;
    constexpr int kVecSize = kMinimaxM3ElemsPerThread * sizeof(__nv_bfloat16) / 4;
    using VecT = typename tensorrt_llm::common::packed_as<uint, kVecSize>::type;
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaGridDependencySynchronize();
    // IDX-PDL: let the dependent grid (the index-K decode reader) launch now. Every thread that triggers has passed
    // its own wait, so the reader launches only once this grid's predecessors are complete; it reads this grid's
    // outputs (q, index-Q, the new index-K rows) only after its own griddepcontrol.wait, which waits for this whole
    // grid to complete and flush. Exits before the wait (tokenIdx >= numTokens) count as triggers.
    if (earlyTrigger)
    {
        cudaTriggerProgrammaticLaunchCompletion();
    }
#endif
    VecT const packed = *reinterpret_cast<VecT const*>(packedInput + inputOffset);

    float elements[kMinimaxM3ElemsPerThread];
    float sumSquares = 0.0F;
#pragma unroll
    for (int pair = 0; pair < kVecSize; ++pair)
    {
        float2 const values = __bfloat1622float2(
            *reinterpret_cast<__nv_bfloat162 const*>(reinterpret_cast<uint const*>(&packed) + pair));
        elements[2 * pair] = values.x;
        elements[2 * pair + 1] = values.y;
        if (!isV)
        {
            sumSquares += values.x * values.x + values.y * values.y;
        }
    }

    if (!isV)
    {
        auto const* normWeight = isQ ? qWeight : (isK ? kWeight : (isIndexQ ? indexQWeight : indexKWeight));
        sumSquares = tensorrt_llm::common::warpReduceSum(sumSquares);
        float const rmsReciprocal = rsqrtf(sumSquares / static_cast<float>(kMinimaxM3HeadDim) + eps);
#pragma unroll
        for (int i = 0; i < kMinimaxM3ElemsPerThread; ++i)
        {
            int const dim = laneId * kMinimaxM3ElemsPerThread + i;
            elements[i] *= rmsReciprocal * (1.0F + __bfloat162float(normWeight[dim]));
        }

        __syncwarp();
        constexpr int kPairOffset = (kMinimaxM3RotaryDim / 2) / kMinimaxM3ElemsPerThread;
        int positionId = laneId == 0 ? positionIds[tokenIdx] : 0;
        positionId = __shfl_sync(0xffffffff, positionId, 0);
        int64_t const ropeRow = static_cast<int64_t>(positionId) * kMinimaxM3RotaryDim;
#pragma unroll
        for (int i = 0; i < kMinimaxM3ElemsPerThread; ++i)
        {
            int const dim = laneId * kMinimaxM3ElemsPerThread + i;
            float paired = __shfl_xor_sync(0xffffffff, elements[i], kPairOffset);
            if (dim < kMinimaxM3RotaryDim)
            {
                bool const firstHalf = dim < kMinimaxM3RotaryDim / 2;
                if (firstHalf)
                {
                    paired = -paired;
                }
                int const coefficient = firstHalf ? dim : dim - kMinimaxM3RotaryDim / 2;
                float const cosine = rotaryCosSin[ropeRow + coefficient];
                float const sine = rotaryCosSin[ropeRow + kMinimaxM3RotaryDim / 2 + coefficient];
                elements[i] = elements[i] * cosine + paired * sine;
            }
        }
        __syncwarp();
    }

    if (isQ)
    {
        int const head = localHead;
        int64_t const outputOffset = (static_cast<int64_t>(tokenIdx) * numHeadsQ + head) * kMinimaxM3HeadDim
            + laneId * kMinimaxM3ElemsPerThread;
        storeFp8HeadElements64<kMinimaxM3ElemsPerThread>(qOutput, outputOffset, elements);
        return;
    }
    if (isIndexQ)
    {
        int const head = localHead - indexQBegin;
        int64_t const outputOffset = (static_cast<int64_t>(tokenIdx) * numHeadsIndex + head) * kMinimaxM3HeadDim
            + laneId * kMinimaxM3ElemsPerThread;
        storeFp8HeadElements64<kMinimaxM3ElemsPerThread>(indexQOutput, outputOffset, elements);
        return;
    }

    int slot = laneId == 0 ? outCacheLoc[tokenIdx] : 0;
    slot = __shfl_sync(0xffffffff, slot, 0);
    if (slot < 0)
    {
        return;
    }
    int const page = slot / kMinimaxM3PageSize;
    if (page >= numPages)
    {
        return;
    }
    int const withinPage = slot & (kMinimaxM3PageSize - 1);
    if (isIndexK)
    {
        if constexpr (kIdxkMode == kMinimaxM3IndexKFp8Sink)
        {
            int64_t const outputOffset = static_cast<int64_t>(page) * indexPageStride
                + static_cast<int64_t>(withinPage) * indexTokenStride + laneId * kMinimaxM3ElemsPerThread;
            storeFp8HeadElements64<kMinimaxM3ElemsPerThread>(indexKCache, outputOffset, elements);
        }
        else
        {
            storeMinimaxM3IndexKDirect<kIdxkMode>(
                indexKCache, page, withinPage, indexPageStride, indexTokenStride, laneId, elements);
        }
        return;
    }

    int const head = isK ? localHead - kBegin : localHead - vBegin;
    int const plane = isV ? 1 : 0;
    int const scaleColumn = laneId / 4;
    int64_t const scaleWithinPage = isV
        ? static_cast<int64_t>(withinPage / 4) * 32 + static_cast<int64_t>(scaleColumn) * 4 + withinPage % 4
        : static_cast<int64_t>(withinPage) * 8 + scaleColumn;
    int64_t const scaleOffset = static_cast<int64_t>(page) * scalePageStride
        + static_cast<int64_t>(plane) * scalePlaneStride + static_cast<int64_t>(head) * scaleHeadStride
        + scaleWithinPage;
    uint8_t* scaleOut = laneId % 4 == 0 ? kvScaleCache + scaleOffset : nullptr;
    uint16_t const packedFp4 = quantizeMinimaxM3Fp4x4(elements, kvQuantScale[isV ? 2 : 1], scaleOut);
    int64_t const dataOffset = static_cast<int64_t>(page) * dataPageStride
        + static_cast<int64_t>(plane) * dataPlaneStride + static_cast<int64_t>(head) * dataHeadStride
        + static_cast<int64_t>(withinPage) * dataTokenStride + laneId * 2;
    *reinterpret_cast<uint16_t*>(kvDataCache + dataOffset) = packedFp4;
}

} // namespace

void launchMinimaxM3Nvfp4QKVIndexerNormRopeKVInsert(void const* packed_input, void* q_output, void* index_q_output,
    void* kv_data_cache, void* kv_scale_cache, void* index_k_cache, int const* out_cache_loc,
    float const* kv_quant_scale, int64_t data_page_stride, int64_t data_plane_stride, int64_t data_head_stride,
    int64_t data_token_stride, int64_t scale_page_stride, int64_t scale_plane_stride, int64_t scale_head_stride,
    int64_t index_page_stride, int64_t index_token_stride, int64_t num_pages, int page_size, int num_tokens,
    int num_heads_q, int num_heads_kv, int num_heads_index, int head_dim, int rotary_dim, float eps,
    void const* q_weight, void const* k_weight, void const* index_q_weight, void const* index_k_weight,
    float const* rotary_cos_sin, int const* position_ids, int index_k_mode, cudaStream_t stream, bool early_trigger)
{
    TLLM_CHECK_WITH_INFO(head_dim == kMinimaxM3HeadDim, "MiniMax-M3 NVFP4 horizontal producer requires head_dim=128");
    TLLM_CHECK_WITH_INFO(
        rotary_dim == kMinimaxM3RotaryDim, "MiniMax-M3 NVFP4 horizontal producer requires rotary_dim=64");
    TLLM_CHECK_WITH_INFO(
        page_size == kMinimaxM3PageSize, "MiniMax-M3 NVFP4 horizontal producer requires page_size=128");
    TLLM_CHECK_WITH_INFO(num_heads_q > 0 && num_heads_kv > 0 && num_heads_index > 0,
        "MiniMax-M3 NVFP4 horizontal producer requires Q, KV, and index heads");
    TLLM_CHECK_WITH_INFO(
        num_heads_index == num_heads_kv, "MiniMax-M3 NVFP4 horizontal producer requires index heads to equal KV heads");

    constexpr int kBlockSize = 256;
    constexpr int kWarpsPerBlock = kBlockSize / 32;
    int const slotsPerToken = num_heads_q + 2 * num_heads_kv + num_heads_index + 1;
    int const totalWarps = num_tokens * slotsPerToken;
    int const gridSize = tensorrt_llm::common::divUp(totalWarps, kWarpsPerBlock);
    auto kernel = minimaxM3Nvfp4QKVIndexerNormRopeKVInsertKernel<kMinimaxM3IndexKFp8Sink>;
    if (index_k_mode == kMinimaxM3IndexKNvfp4Pages)
    {
        kernel = minimaxM3Nvfp4QKVIndexerNormRopeKVInsertKernel<kMinimaxM3IndexKNvfp4Pages>;
    }
    else if (index_k_mode == kMinimaxM3IndexKBf16Tap)
    {
        kernel = minimaxM3Nvfp4QKVIndexerNormRopeKVInsertKernel<kMinimaxM3IndexKBf16Tap>;
    }
    else
    {
        TLLM_CHECK_WITH_INFO(index_k_mode == kMinimaxM3IndexKFp8Sink,
            "MiniMax-M3 NVFP4 horizontal producer: unknown index_k_mode %d", index_k_mode);
    }
    cudaLaunchConfig_t config{};
    config.gridDim = gridSize;
    config.blockDim = kBlockSize;
    config.dynamicSmemBytes = 0;
    config.stream = stream;
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[0].val.programmaticStreamSerializationAllowed = tensorrt_llm::common::getEnvEnablePDL();
    config.numAttrs = 1;
    config.attrs = attrs;
    TLLM_CUDA_CHECK(cudaLaunchKernelEx(&config, kernel, static_cast<__nv_bfloat16 const*>(packed_input),
        static_cast<__nv_fp8_e4m3*>(q_output), static_cast<__nv_fp8_e4m3*>(index_q_output),
        static_cast<uint8_t*>(kv_data_cache), static_cast<uint8_t*>(kv_scale_cache),
        static_cast<__nv_fp8_e4m3*>(index_k_cache), out_cache_loc, kv_quant_scale, data_page_stride, data_plane_stride,
        data_head_stride, data_token_stride, scale_page_stride, scale_plane_stride, scale_head_stride,
        index_page_stride, index_token_stride, num_pages, num_tokens, num_heads_q, num_heads_kv, num_heads_index, eps,
        static_cast<__nv_bfloat16 const*>(q_weight), static_cast<__nv_bfloat16 const*>(k_weight),
        static_cast<__nv_bfloat16 const*>(index_q_weight), static_cast<__nv_bfloat16 const*>(index_k_weight),
        rotary_cos_sin, position_ids, early_trigger));
}

void launchMinimaxM3Fp8QKVIndexerNormRopeKVInsert(void const* packed_input, void* q_output, void* index_q_output,
    void* kv_cache, void* index_k_cache, int const* out_cache_loc, int64_t kv_page_stride, int64_t kv_plane_stride,
    int64_t kv_head_stride, int64_t kv_token_stride, int64_t index_page_stride, int64_t index_token_stride,
    int64_t num_pages, int page_size, int num_tokens, int num_heads_q, int num_heads_kv, int num_heads_index,
    int head_dim, int rotary_dim, float eps, void const* q_weight, void const* k_weight, void const* index_q_weight,
    void const* index_k_weight, float const* rotary_cos_sin, int const* position_ids, int index_k_mode,
    cudaStream_t stream)
{
    TLLM_CHECK_WITH_INFO(head_dim == kMinimaxM3HeadDim, "MiniMax-M3 horizontal producer requires head_dim=128");
    TLLM_CHECK_WITH_INFO(rotary_dim == kMinimaxM3RotaryDim, "MiniMax-M3 horizontal producer requires rotary_dim=64");
    TLLM_CHECK_WITH_INFO(page_size == kMinimaxM3PageSize, "MiniMax-M3 horizontal producer requires page_size=128");
    TLLM_CHECK_WITH_INFO(num_heads_q > 0 && num_heads_kv > 0 && num_heads_index > 0,
        "MiniMax-M3 horizontal producer requires Q, KV, and index heads");
    TLLM_CHECK_WITH_INFO(
        num_heads_index == num_heads_kv, "MiniMax-M3 horizontal producer requires index heads to equal KV heads");

    constexpr int kBlockSize = 256;
    constexpr int kWarpsPerBlock = kBlockSize / 32;
    int const slotsPerToken = num_heads_q + 2 * num_heads_kv + num_heads_index + 1;
    int const totalWarps = num_tokens * slotsPerToken;
    int const gridSize = tensorrt_llm::common::divUp(totalWarps, kWarpsPerBlock);
    auto kernel = minimaxM3Fp8QKVIndexerNormRopeKVInsertKernel<kMinimaxM3IndexKFp8Sink>;
    if (index_k_mode == kMinimaxM3IndexKNvfp4Pages)
    {
        kernel = minimaxM3Fp8QKVIndexerNormRopeKVInsertKernel<kMinimaxM3IndexKNvfp4Pages>;
    }
    else if (index_k_mode == kMinimaxM3IndexKBf16Tap)
    {
        kernel = minimaxM3Fp8QKVIndexerNormRopeKVInsertKernel<kMinimaxM3IndexKBf16Tap>;
    }
    else
    {
        TLLM_CHECK_WITH_INFO(index_k_mode == kMinimaxM3IndexKFp8Sink,
            "MiniMax-M3 horizontal producer: unknown index_k_mode %d", index_k_mode);
    }
    tensorrt_llm::common::launchWithPdlWhenEnabled("minimaxM3Fp8QKVIndexerNormRopeKVInsert", kernel, gridSize,
        kBlockSize, 0, stream, static_cast<__nv_bfloat16 const*>(packed_input), static_cast<__nv_fp8_e4m3*>(q_output),
        static_cast<__nv_fp8_e4m3*>(index_q_output), static_cast<__nv_fp8_e4m3*>(kv_cache),
        static_cast<__nv_fp8_e4m3*>(index_k_cache), out_cache_loc, kv_page_stride, kv_plane_stride, kv_head_stride,
        kv_token_stride, index_page_stride, index_token_stride, num_pages, num_tokens, num_heads_q, num_heads_kv,
        num_heads_index, eps, static_cast<__nv_bfloat16 const*>(q_weight), static_cast<__nv_bfloat16 const*>(k_weight),
        static_cast<__nv_bfloat16 const*>(index_q_weight), static_cast<__nv_bfloat16 const*>(index_k_weight),
        rotary_cos_sin, position_ids);
}

} // namespace kernels

TRTLLM_MV2_NAMESPACE_END
