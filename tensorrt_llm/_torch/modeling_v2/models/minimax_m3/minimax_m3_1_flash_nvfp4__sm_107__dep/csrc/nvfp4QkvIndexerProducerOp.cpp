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

// The dep target's NVFP4 horizontal producer: trtllm::minimax_m3_nvfp4_qkv_indexer_norm_rope_kv_insert with the
// index-K output chosen by the index_k_cache dtype (FP8 rows, NVFP4 pages, or a bfloat16 tap) and the early PDL
// trigger.

#include "minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__dep/csrc/nvfp4QkvIndexerProducer.h"
#include "tensorrt_llm/modeling_v2/common/namespace.h"
#include "tensorrt_llm/modeling_v2/common/torchLibrary.h"
#include "tensorrt_llm/thop/thUtils.h"

#include <ATen/cuda/CUDAContext.h>
#include <limits>
#include <torch/extension.h>
#include <tuple>

TRTLLM_MV2_NAMESPACE_BEGIN

namespace torch_ext
{

namespace
{

void checkMinimaxM3Int32LaunchGeometry(int64_t numTokens, int64_t slotsPerToken)
{
    TORCH_CHECK(numTokens <= std::numeric_limits<int>::max(), "MiniMax-M3 producer num_tokens exceeds int32");
    TORCH_CHECK(slotsPerToken > 0 && slotsPerToken <= std::numeric_limits<int>::max(),
        "MiniMax-M3 producer head geometry exceeds int32");
    TORCH_CHECK(numTokens == 0 || slotsPerToken <= std::numeric_limits<int>::max() / numTokens,
        "MiniMax-M3 producer launch geometry exceeds int32");
}

void checkMinimaxM3Nvfp4HndKVPools(
    torch::Tensor const& dataCache, torch::Tensor const& scaleCache, int64_t numHeads, int64_t headDim)
{
    TORCH_CHECK(dataCache.is_cuda() && dataCache.scalar_type() == at::ScalarType::Byte,
        "NVFP4 data cache must be a CUDA uint8 tensor");
    TORCH_CHECK(scaleCache.is_cuda() && scaleCache.scalar_type() == at::ScalarType::Byte,
        "NVFP4 scale cache must be a CUDA uint8 tensor");
    TORCH_CHECK(dataCache.dim() == 5 && scaleCache.dim() == 5,
        "NVFP4 caches must be HND [num_pages, 2, num_heads, page_size, packed_width]");
    TORCH_CHECK(dataCache.size(0) > 0 && dataCache.size(1) == 2 && dataCache.size(2) == numHeads,
        "NVFP4 data cache page, plane, or head geometry mismatch");
    TORCH_CHECK(dataCache.size(3) == 128 && dataCache.size(4) * 2 == headDim,
        "NVFP4 data cache must use P128 with packed head_dim/2 bytes");
    TORCH_CHECK(scaleCache.size(0) == dataCache.size(0) && scaleCache.size(1) == dataCache.size(1)
            && scaleCache.size(2) == dataCache.size(2) && scaleCache.size(3) == dataCache.size(3)
            && scaleCache.size(4) * 16 == headDim,
        "NVFP4 scale cache must match the data-cache geometry with head_dim/16 bytes");
    TORCH_CHECK(dataCache.stride(4) == 1 && dataCache.stride(3) == dataCache.size(4),
        "NVFP4 data cache must have contiguous packed token rows");
    TORCH_CHECK(scaleCache.stride(4) == 1 && scaleCache.stride(3) == scaleCache.size(4),
        "NVFP4 scale cache must have contiguous scale token rows");
    TORCH_CHECK(dataCache.stride(2) == dataCache.size(3) * dataCache.stride(3),
        "NVFP4 data cache must have contiguous packed head blocks");
    TORCH_CHECK(scaleCache.stride(2) == scaleCache.size(3) * scaleCache.stride(3),
        "NVFP4 scale cache must have contiguous scale head blocks");
    TORCH_CHECK(dataCache.stride(1) >= dataCache.size(2) * dataCache.stride(2)
            && dataCache.stride(0) >= dataCache.size(1) * dataCache.stride(1),
        "NVFP4 data cache planes or pages overlap");
    TORCH_CHECK(scaleCache.stride(1) >= scaleCache.size(2) * scaleCache.stride(2)
            && scaleCache.stride(0) >= scaleCache.size(1) * scaleCache.stride(1),
        "NVFP4 scale cache planes or pages overlap");
    TORCH_CHECK(dataCache.stride(0) % 2 == 0 && dataCache.stride(1) % 2 == 0,
        "NVFP4 data cache page and plane strides must preserve uint16 store alignment");
}

// The index-K output of a horizontal producer, from the index_k_cache dtype (kMinimaxM3IndexK* in
// nvfp4QkvIndexerProducer.h):
//   float8_e4m3fn: E4M3 rows (the FP8 index-K cache, or the FP8 sink an NVFP4 index-K layer packs from); each
//     producer keeps its own checks of that layout.
//   uint8: NVFP4 index-K pages, [num_pages, 1, 128, 72] or [num_pages, 9216] with each page's bytes contiguous and a
//     16-byte-aligned base and page stride (the m31idxk writer's layout); written straight from the producer.
//   bfloat16 (tests): the BF16-rounded index-K rows the NVFP4 mode quantizes, [num_pages, 1, 128, 128].
// Checks the uint8 / bfloat16 layouts and returns the mode with the page and token strides (in the cache's
// elements) to pass the launcher.
int minimaxM3DirectIndexKMode(
    torch::Tensor const& indexKCache, int64_t numPages, int64_t& indexPageStride, int64_t& indexTokenStride)
{
    constexpr int64_t kPageSize = 128;
    constexpr int64_t kHeadDim = 128;
    constexpr int64_t kNvfp4PageBytes = kPageSize * (kHeadDim / 2 + kHeadDim / 16); // 9216
    TORCH_CHECK(indexKCache.is_cuda(), "Index-K cache must be a CUDA tensor");
    if (indexKCache.scalar_type() == at::ScalarType::Byte)
    {
        TORCH_CHECK(indexKCache.numel() > 0 && indexKCache.dim() >= 2, "NVFP4 index-K pages must be [num_pages, ...]");
        int64_t perPage = 1;
        for (int d = 1; d < indexKCache.dim(); ++d)
        {
            perPage *= indexKCache.size(d);
        }
        TORCH_CHECK(perPage == kNvfp4PageBytes, "an NVFP4 index-K page must hold 9216 bytes, got ", perPage);
        int64_t expect = 1;
        for (int d = indexKCache.dim() - 1; d >= 1; --d)
        {
            TORCH_CHECK(indexKCache.size(d) == 1 || indexKCache.stride(d) == expect,
                "an NVFP4 index-K page must be contiguous");
            expect *= indexKCache.size(d);
        }
        TORCH_CHECK(indexKCache.stride(0) >= kNvfp4PageBytes, "NVFP4 index-K pages must not overlap");
        TORCH_CHECK(indexKCache.stride(0) % 16 == 0 && reinterpret_cast<uintptr_t>(indexKCache.data_ptr()) % 16 == 0,
            "NVFP4 index-K pages must be 16-byte aligned");
        TORCH_CHECK(indexKCache.size(0) == numPages,
            "NVFP4 index-K pages and the main K/V cache must contain the same number of pages");
        indexPageStride = indexKCache.stride(0);
        indexTokenStride = 0;
        return kernels::kMinimaxM3IndexKNvfp4Pages;
    }
    TORCH_CHECK(indexKCache.scalar_type() == at::ScalarType::BFloat16,
        "Index-K cache must be CUDA torch.float8_e4m3fn (or uint8 NVFP4 index-K pages, or a bfloat16 tap)");
    TORCH_CHECK(indexKCache.dim() == 4 && indexKCache.size(1) == 1 && indexKCache.size(2) == kPageSize
            && indexKCache.size(3) == kHeadDim,
        "The bfloat16 index-K tap must be [num_pages, 1, 128, 128]");
    TORCH_CHECK(indexKCache.stride(3) == 1 && indexKCache.stride(2) == kHeadDim,
        "The bfloat16 index-K tap must have contiguous token rows");
    TORCH_CHECK(indexKCache.stride(0) >= kPageSize * kHeadDim, "The bfloat16 index-K tap pages must not overlap");
    TORCH_CHECK(indexKCache.stride(0) % 4 == 0 && reinterpret_cast<uintptr_t>(indexKCache.data_ptr()) % 8 == 0,
        "The bfloat16 index-K tap must keep 8-byte store alignment");
    TORCH_CHECK(indexKCache.size(0) == numPages,
        "The bfloat16 index-K tap and the main K/V cache must contain the same number of pages");
    indexPageStride = indexKCache.stride(0);
    indexTokenStride = indexKCache.stride(2);
    return kernels::kMinimaxM3IndexKBf16Tap;
}

} // namespace

std::tuple<torch::Tensor, torch::Tensor> minimaxM3Nvfp4QKVIndexerNormRopeKVInsert(torch::Tensor const& packed,
    torch::Tensor& kvDataCache, torch::Tensor& kvScaleCache, torch::Tensor& indexKCache,
    torch::Tensor const& outCacheLoc, torch::Tensor const& kvQuantScale, int64_t numHeadsQ, int64_t numHeadsKV,
    int64_t numHeadsIndex, int64_t headDim, int64_t rotaryDim, double eps, torch::Tensor const& qWeight,
    torch::Tensor const& kWeight, torch::Tensor const& indexQWeight, torch::Tensor const& indexKWeight,
    torch::Tensor const& rotaryCosSin, torch::Tensor const& positionIds, bool earlyTrigger)
{
    constexpr int64_t kHeadDim = 128;
    constexpr int64_t kRotaryDim = 64;
    constexpr int64_t kPageSize = 128;
    TORCH_CHECK(numHeadsQ > 0 && numHeadsKV > 0 && numHeadsIndex > 0,
        "MiniMax-M3 NVFP4 horizontal producer requires Q, KV, and index heads");
    TORCH_CHECK(
        numHeadsKV == numHeadsIndex, "MiniMax-M3 NVFP4 horizontal producer requires index heads to equal KV heads");
    // The model has a global 16:1 Q-to-KV head ratio, but that ratio is not
    // preserved on a rank once TP exceeds the four KV heads.  For example,
    // TP8 owns eight Q heads and one replicated KV head.  The kernel processes
    // every packed head independently and does not require a local ratio.
    TORCH_CHECK(headDim == kHeadDim, "MiniMax-M3 NVFP4 horizontal producer requires head_dim=128");
    TORCH_CHECK(rotaryDim == kRotaryDim, "MiniMax-M3 NVFP4 horizontal producer requires rotary_dim=64");
    TORCH_CHECK(eps >= 0.0, "MiniMax-M3 NVFP4 horizontal producer requires eps >= 0");

    TORCH_CHECK(packed.dim() == 2, "Packed QKV+index tensor must be two-dimensional");
    TORCH_CHECK(outCacheLoc.dim() == 1, "out_cache_loc must be one-dimensional");
    TORCH_CHECK(positionIds.dim() == 1, "position_ids must be one-dimensional");
    CHECK_INPUT(packed, torch::kBFloat16);
    CHECK_INPUT(outCacheLoc, torch::kInt32);
    CHECK_INPUT(positionIds, torch::kInt32);
    CHECK_INPUT(qWeight, torch::kBFloat16);
    CHECK_INPUT(kWeight, torch::kBFloat16);
    CHECK_INPUT(indexQWeight, torch::kBFloat16);
    CHECK_INPUT(indexKWeight, torch::kBFloat16);
    CHECK_INPUT(rotaryCosSin, torch::kFloat32);
    CHECK_INPUT(kvQuantScale, torch::kFloat32);
    checkMinimaxM3Nvfp4HndKVPools(kvDataCache, kvScaleCache, numHeadsKV, headDim);
    TORCH_CHECK(kvDataCache.size(3) == kPageSize, "MiniMax-M3 NVFP4 horizontal producer requires page_size=128");
    TORCH_CHECK(kvQuantScale.numel() >= 3, "NVFP4 K/V quantization scale must contain [Q, K, V]");
    int indexKMode = kernels::kMinimaxM3IndexKFp8Sink;
    int64_t indexPageStride = 0;
    int64_t indexTokenStride = 0;
    if (indexKCache.scalar_type() == at::ScalarType::Float8_e4m3fn)
    {
        TORCH_CHECK(indexKCache.is_cuda(), "Index-K cache must be CUDA torch.float8_e4m3fn");
        TORCH_CHECK(indexKCache.dim() == 4 && indexKCache.size(1) == 1 && indexKCache.size(2) == kPageSize
                && indexKCache.size(3) == kHeadDim,
            "Index-K cache must be HND [num_pages, 1, 128, 128]");
        TORCH_CHECK(indexKCache.size(0) == kvDataCache.size(0),
            "Index-K and main NVFP4 caches must contain the same number of pages");
        TORCH_CHECK(indexKCache.stride(3) == 1 && indexKCache.stride(2) == kHeadDim,
            "Index-K cache must have contiguous token rows");
        TORCH_CHECK(indexKCache.stride(0) >= indexKCache.size(2) * indexKCache.stride(2),
            "Index-K cache pages must not overlap");
        TORCH_CHECK(
            indexKCache.stride(0) % 4 == 0, "Index-K cache page stride must preserve 32-bit FP8 store alignment");
        indexPageStride = indexKCache.stride(0);
        indexTokenStride = indexKCache.stride(2);
    }
    else
    {
        indexKMode = minimaxM3DirectIndexKMode(indexKCache, kvDataCache.size(0), indexPageStride, indexTokenStride);
    }
    TORCH_CHECK(rotaryCosSin.dim() == 3 && rotaryCosSin.size(1) == 2 && rotaryCosSin.size(2) == kRotaryDim / 2,
        "rotary_cos_sin must be [max_positions, 2, rotary_dim/2]");

    int64_t const numTokens = packed.size(0);
    int64_t const totalHeads = numHeadsQ + 2 * numHeadsKV + numHeadsIndex + 1;
    checkMinimaxM3Int32LaunchGeometry(numTokens, totalHeads);
    TORCH_CHECK(
        packed.size(1) == totalHeads * headDim, "Packed tensor width must equal (Q + 2*KV + index-Q + 1) * head_dim");
    TORCH_CHECK(outCacheLoc.numel() >= numTokens, "out_cache_loc is shorter than num_tokens");
    TORCH_CHECK(positionIds.numel() == numTokens, "position_ids length must equal num_tokens");
    TORCH_CHECK(qWeight.numel() == headDim && kWeight.numel() == headDim && indexQWeight.numel() == headDim
            && indexKWeight.numel() == headDim,
        "All norm weights must contain head_dim elements");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(packed.data_ptr()) % 8 == 0,
        "Packed input must start at an 8-byte-aligned address for vectorized BF16 loads");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(kvDataCache.data_ptr()) % 2 == 0
            && reinterpret_cast<uintptr_t>(indexKCache.data_ptr()) % 4 == 0,
        "Paged caches must start at aligned addresses for packed stores");
    TORCH_CHECK(packed.get_device() == kvDataCache.get_device() && packed.get_device() == kvScaleCache.get_device()
            && packed.get_device() == indexKCache.get_device() && packed.get_device() == outCacheLoc.get_device()
            && packed.get_device() == kvQuantScale.get_device() && packed.get_device() == positionIds.get_device()
            && packed.get_device() == qWeight.get_device() && packed.get_device() == kWeight.get_device()
            && packed.get_device() == indexQWeight.get_device() && packed.get_device() == indexKWeight.get_device()
            && packed.get_device() == rotaryCosSin.get_device(),
        "All MiniMax-M3 NVFP4 horizontal producer tensors must be on the same CUDA device");

    auto qOut = torch::empty({numTokens, numHeadsQ, headDim}, packed.options().dtype(at::ScalarType::Float8_e4m3fn));
    auto indexQOut
        = torch::empty({numTokens, numHeadsIndex, headDim}, packed.options().dtype(at::ScalarType::Float8_e4m3fn));
    if (numTokens == 0)
    {
        return {qOut, indexQOut};
    }

    auto stream = at::cuda::getCurrentCUDAStream(packed.get_device());
    kernels::launchMinimaxM3Nvfp4QKVIndexerNormRopeKVInsert(packed.data_ptr(), qOut.data_ptr(), indexQOut.data_ptr(),
        kvDataCache.data_ptr(), kvScaleCache.data_ptr(), indexKCache.data_ptr(), outCacheLoc.data_ptr<int>(),
        kvQuantScale.data_ptr<float>(), kvDataCache.stride(0), kvDataCache.stride(1), kvDataCache.stride(2),
        kvDataCache.stride(3), kvScaleCache.stride(0), kvScaleCache.stride(1), kvScaleCache.stride(2), indexPageStride,
        indexTokenStride, kvDataCache.size(0), static_cast<int>(kvDataCache.size(3)), static_cast<int>(numTokens),
        static_cast<int>(numHeadsQ), static_cast<int>(numHeadsKV), static_cast<int>(numHeadsIndex),
        static_cast<int>(headDim), static_cast<int>(rotaryDim), static_cast<float>(eps), qWeight.data_ptr(),
        kWeight.data_ptr(), indexQWeight.data_ptr(), indexKWeight.data_ptr(), rotaryCosSin.data_ptr<float>(),
        positionIds.data_ptr<int>(), indexKMode, stream, earlyTrigger);
    return {qOut, indexQOut};
}

} // namespace torch_ext

TRTLLM_MV2_NAMESPACE_END

TRTLLM_MV2_LIBRARY_FRAGMENT(m)
{
    m.def(
        "minimax_m3_nvfp4_qkv_indexer_norm_rope_kv_insert(Tensor packed, Tensor(a!) kv_data_cache, Tensor(b!) "
        "kv_scale_cache, Tensor(c!) index_k_cache, Tensor out_cache_loc, Tensor kv_quant_scale, int num_heads_q, "
        "int num_heads_kv, int num_heads_index, int head_dim, int rotary_dim, float eps, Tensor q_weight, Tensor "
        "k_weight, Tensor index_q_weight, Tensor index_k_weight, Tensor rotary_cos_sin, Tensor position_ids, bool "
        "early_trigger=False) -> (Tensor, Tensor)");
}

TRTLLM_MV2_LIBRARY_IMPL(CUDA, m)
{
    m.impl("minimax_m3_nvfp4_qkv_indexer_norm_rope_kv_insert",
        &TRTLLM_MV2_NS::torch_ext::minimaxM3Nvfp4QKVIndexerNormRopeKVInsert);
}
