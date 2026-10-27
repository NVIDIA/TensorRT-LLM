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

// MiniMax-M3.1 tp2ctx (#20629 L2 / carry rows 12 + 20): the long-chunk rows variant of the NVFP4 QKV + indexer
// producer, copied verbatim from #20629's fusedQKNormRopeKernel.cu (cf8a77e816) with the per-token kernel's helpers it
// uses, so the shared fusedQKNormRopeKernel.cu is not touched. Built with the library's flags, like the original.

#include "tensorrt_llm/kernels/minimaxM31/qkvIndexerProducerRowsSm103.h"

#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/envUtils.h"
#include "tensorrt_llm/common/mathUtils.h"
#include "tensorrt_llm/common/reduceKernelUtils.cuh"
#include "tensorrt_llm/kernels/quantization.cuh"

#include <cuda_bf16.h>
#include <cuda_fp8.h>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{
namespace minimaxM31
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

    uint32_t const groupMask = cvt_sf_group_shfl_mask<4>();
    localMax = fmaxf(localMax, __shfl_xor_sync(groupMask, localMax, 1));
    localMax = fmaxf(localMax, __shfl_xor_sync(groupMask, localMax, 2));

    float const scaleValue = secondLevelScale * (localMax * reciprocal_approximate_ftz(6.0F));
    __nv_fp8_e4m3 const encodedScale(scaleValue);
    if (scaleOut != nullptr)
    {
        *scaleOut = encodedScale.__x;
    }
    float const narrowedScale = static_cast<float>(encodedScale);
    float const outputScale = localMax != 0.0F
        ? reciprocal_approximate_ftz(narrowedScale * reciprocal_approximate_ftz(secondLevelScale))
        : 0.0F;

    float quantValues[8] = {};
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        quantValues[i] = __bfloat162float(rounded[i]) * outputScale;
    }
    return static_cast<uint16_t>(fp32_vec_to_e2m1(quantValues));
}

constexpr int kMinimaxM3HeadDim = 128;
constexpr int kMinimaxM3RotaryDim = 64;
constexpr int kMinimaxM3PageSize = 128;
static_assert((kMinimaxM3PageSize & (kMinimaxM3PageSize - 1)) == 0, "page size must be a power of two");
constexpr int kMinimaxM3ElemsPerThread = kMinimaxM3HeadDim / 32;

// minimaxM3Nvfp4QKVIndexerNormRopeKVInsertKernel for long chunks: one warp per (head slot, kTokens consecutive
// tokens) instead of one per (token, head slot). The norm weight is loaded once per warp and every token's input row,
// position id and cache slot are loaded before any token is processed; each token then runs the exact per-element
// code of the per-token kernel, so the outputs are bitwise the same.
template <int kTokens>
__global__ void minimaxM3Nvfp4QKVIndexerNormRopeKVInsertRowsKernel(__nv_bfloat16 const* packedInput,
    __nv_fp8_e4m3* qOutput, __nv_fp8_e4m3* indexQOutput, uint8_t* kvDataCache, uint8_t* kvScaleCache,
    __nv_fp8_e4m3* indexKCache, int const* outCacheLoc, float const* kvQuantScale, int64_t dataPageStride,
    int64_t dataPlaneStride, int64_t dataHeadStride, int64_t dataTokenStride, int64_t scalePageStride,
    int64_t scalePlaneStride, int64_t scaleHeadStride, int64_t indexPageStride, int64_t indexTokenStride,
    int64_t numPages, int numTokens, int numHeadsQ, int numHeadsKV, int numHeadsIndex, float eps,
    __nv_bfloat16 const* qWeight, __nv_bfloat16 const* kWeight, __nv_bfloat16 const* indexQWeight,
    __nv_bfloat16 const* indexKWeight, float const* rotaryCosSin, int const* positionIds)
{
    int const warpsPerBlock = blockDim.x / 32;
    int const warpId = threadIdx.x / 32;
    int const laneId = threadIdx.x % 32;
    int const globalWarp = blockIdx.x * warpsPerBlock + warpId;
    int const totalHeads = numHeadsQ + 2 * numHeadsKV + numHeadsIndex + 1;
    int const tokenBase = (globalWarp / totalHeads) * kTokens;
    int const localHead = globalWarp % totalHeads;
    if (tokenBase >= numTokens)
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
    bool const toCache = !isQ && !isIndexQ;

    constexpr int kVecSize = kMinimaxM3ElemsPerThread * sizeof(__nv_bfloat16) / 4;
    using VecT = typename tensorrt_llm::common::packed_as<uint, kVecSize>::type;
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaGridDependencySynchronize();
#endif
    float weights[kMinimaxM3ElemsPerThread];
    if (!isV)
    {
        auto const* normWeight = isQ ? qWeight : (isK ? kWeight : (isIndexQ ? indexQWeight : indexKWeight));
#pragma unroll
        for (int i = 0; i < kMinimaxM3ElemsPerThread; ++i)
        {
            weights[i] = __bfloat162float(normWeight[laneId * kMinimaxM3ElemsPerThread + i]);
        }
    }
    VecT packedRows[kTokens];
    int positions[kTokens];
    int slots[kTokens];
#pragma unroll
    for (int t = 0; t < kTokens; ++t)
    {
        int const tokenIdx = tokenBase + t;
        if (tokenIdx < numTokens)
        {
            int64_t const inputOffset = (static_cast<int64_t>(tokenIdx) * totalHeads + localHead) * kMinimaxM3HeadDim
                + laneId * kMinimaxM3ElemsPerThread;
            packedRows[t] = *reinterpret_cast<VecT const*>(packedInput + inputOffset);
            positions[t] = (!isV && laneId == 0) ? positionIds[tokenIdx] : 0;
            slots[t] = (toCache && laneId == 0) ? outCacheLoc[tokenIdx] : 0;
        }
    }

#pragma unroll
    for (int t = 0; t < kTokens; ++t)
    {
        int const tokenIdx = tokenBase + t;
        if (tokenIdx >= numTokens)
        {
            break;
        }
        float elements[kMinimaxM3ElemsPerThread];
        float sumSquares = 0.0F;
#pragma unroll
        for (int pair = 0; pair < kVecSize; ++pair)
        {
            float2 const values = __bfloat1622float2(
                *reinterpret_cast<__nv_bfloat162 const*>(reinterpret_cast<uint const*>(&packedRows[t]) + pair));
            elements[2 * pair] = values.x;
            elements[2 * pair + 1] = values.y;
            if (!isV)
            {
                sumSquares += values.x * values.x + values.y * values.y;
            }
        }

        if (!isV)
        {
            sumSquares = tensorrt_llm::common::warpReduceSum(sumSquares);
            float const rmsReciprocal = rsqrtf(sumSquares / static_cast<float>(kMinimaxM3HeadDim) + eps);
#pragma unroll
            for (int i = 0; i < kMinimaxM3ElemsPerThread; ++i)
            {
                elements[i] *= rmsReciprocal * (1.0F + weights[i]);
            }

            __syncwarp();
            constexpr int kPairOffset = (kMinimaxM3RotaryDim / 2) / kMinimaxM3ElemsPerThread;
            int const positionId = __shfl_sync(0xffffffff, positions[t], 0);
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
            int64_t const outputOffset = (static_cast<int64_t>(tokenIdx) * numHeadsQ + localHead) * kMinimaxM3HeadDim
                + laneId * kMinimaxM3ElemsPerThread;
            storeFp8HeadElements64<kMinimaxM3ElemsPerThread>(qOutput, outputOffset, elements);
            continue;
        }
        if (isIndexQ)
        {
            int const head = localHead - indexQBegin;
            int64_t const outputOffset = (static_cast<int64_t>(tokenIdx) * numHeadsIndex + head) * kMinimaxM3HeadDim
                + laneId * kMinimaxM3ElemsPerThread;
            storeFp8HeadElements64<kMinimaxM3ElemsPerThread>(indexQOutput, outputOffset, elements);
            continue;
        }

        int const slot = __shfl_sync(0xffffffff, slots[t], 0);
        if (slot < 0)
        {
            continue;
        }
        int const page = slot / kMinimaxM3PageSize;
        if (page >= numPages)
        {
            continue;
        }
        int const withinPage = slot & (kMinimaxM3PageSize - 1);
        if (isIndexK)
        {
            int64_t const outputOffset = static_cast<int64_t>(page) * indexPageStride
                + static_cast<int64_t>(withinPage) * indexTokenStride + laneId * kMinimaxM3ElemsPerThread;
            storeFp8HeadElements64<kMinimaxM3ElemsPerThread>(indexKCache, outputOffset, elements);
            continue;
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
}

} // namespace

void launchQkvIndexerProducerRows(void const* packed_input, void* q_output, void* index_q_output, void* kv_data_cache,
    void* kv_scale_cache, void* index_k_cache, int const* out_cache_loc, float const* kv_quant_scale,
    int64_t data_page_stride, int64_t data_plane_stride, int64_t data_head_stride, int64_t data_token_stride,
    int64_t scale_page_stride, int64_t scale_plane_stride, int64_t scale_head_stride, int64_t index_page_stride,
    int64_t index_token_stride, int64_t num_pages, int page_size, int num_tokens, int num_heads_q, int num_heads_kv,
    int num_heads_index, int head_dim, int rotary_dim, float eps, void const* q_weight, void const* k_weight,
    void const* index_q_weight, void const* index_k_weight, float const* rotary_cos_sin, int const* position_ids,
    cudaStream_t stream)
{
    TLLM_CHECK_WITH_INFO(head_dim == kMinimaxM3HeadDim, "MiniMax-M3.1 NVFP4 rows producer requires head_dim=128");
    TLLM_CHECK_WITH_INFO(rotary_dim == kMinimaxM3RotaryDim, "MiniMax-M3.1 NVFP4 rows producer requires rotary_dim=64");
    TLLM_CHECK_WITH_INFO(page_size == kMinimaxM3PageSize, "MiniMax-M3.1 NVFP4 rows producer requires page_size=128");
    TLLM_CHECK_WITH_INFO(num_heads_q > 0 && num_heads_kv > 0 && num_heads_index > 0,
        "MiniMax-M3.1 NVFP4 rows producer requires Q, KV, and index heads");
    TLLM_CHECK_WITH_INFO(
        num_heads_index == num_heads_kv, "MiniMax-M3.1 NVFP4 rows producer requires index heads to equal KV heads");

    constexpr int kBlockSize = 256;
    constexpr int kWarpsPerBlock = kBlockSize / 32;
    int const slotsPerToken = num_heads_q + 2 * num_heads_kv + num_heads_index + 1;
    int const totalWarps = common::divUp(num_tokens, kQkvIndexerProducerRowsTokens) * slotsPerToken;
    int const gridSize = common::divUp(totalWarps, kWarpsPerBlock);
    cudaLaunchConfig_t config{};
    config.gridDim = gridSize;
    config.blockDim = kBlockSize;
    config.dynamicSmemBytes = 0;
    config.stream = stream;
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[0].val.programmaticStreamSerializationAllowed = common::getEnvEnablePDL();
    config.numAttrs = 1;
    config.attrs = attrs;
    TLLM_CUDA_CHECK(
        cudaLaunchKernelEx(&config, minimaxM3Nvfp4QKVIndexerNormRopeKVInsertRowsKernel<kQkvIndexerProducerRowsTokens>,
            static_cast<__nv_bfloat16 const*>(packed_input), static_cast<__nv_fp8_e4m3*>(q_output),
            static_cast<__nv_fp8_e4m3*>(index_q_output), static_cast<uint8_t*>(kv_data_cache),
            static_cast<uint8_t*>(kv_scale_cache), static_cast<__nv_fp8_e4m3*>(index_k_cache), out_cache_loc,
            kv_quant_scale, data_page_stride, data_plane_stride, data_head_stride, data_token_stride, scale_page_stride,
            scale_plane_stride, scale_head_stride, index_page_stride, index_token_stride, num_pages, num_tokens,
            num_heads_q, num_heads_kv, num_heads_index, eps, static_cast<__nv_bfloat16 const*>(q_weight),
            static_cast<__nv_bfloat16 const*>(k_weight), static_cast<__nv_bfloat16 const*>(index_q_weight),
            static_cast<__nv_bfloat16 const*>(index_k_weight), rotary_cos_sin, position_ids));
}

} // namespace minimaxM31
} // namespace kernels

TRTLLM_NAMESPACE_END
