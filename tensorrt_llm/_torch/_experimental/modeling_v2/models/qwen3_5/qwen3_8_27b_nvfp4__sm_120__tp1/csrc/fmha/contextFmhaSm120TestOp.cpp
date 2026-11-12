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

// Test-only op for the target's SM120 context FMHA (contextFmhaSm120.h). The model does not call it; it fills the
// kernel parameters the way the attention op does for this target (Q_PAGED_KV, FP8 Q and paged KV, BF16 output,
// 24 query heads over 4 KV heads, head size 256, causal) so tests can check the kernel against a reference.

#include "qwen3_5/qwen3_8_27b_nvfp4__sm_120__tp1/csrc/fmha/contextFmhaSm120.h"
#include "tensorrt_llm/kernels/kvCacheUtils.h"
#include "tensorrt_llm/modeling_v2/common/namespace.h"
#include "tensorrt_llm/modeling_v2/common/torchLibrary.h"
#include "tensorrt_llm/thop/thUtils.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <torch/extension.h>

using torch::Tensor;

TRTLLM_MV2_NAMESPACE_BEGIN

namespace
{

constexpr int kNumHeads = 24;
constexpr int kNumKvHeads = 4;
constexpr int kHeadSize = 256;

bool isFp8Bytes(Tensor const& t)
{
    return t.scalar_type() == at::ScalarType::Byte || t.scalar_type() == at::ScalarType::Float8_e4m3fn;
}

// q:             [T, 24, 256] e4m3 (uint8 or float8_e4m3fn), contiguous
// kv_pool:       [num_blocks, 4, tokens_per_block, 256] e4m3, contiguous; a block holds the K (or V) of 4 heads x
//                tokens_per_block tokens (KVBlockArray layout, tokens_per_block a power of 2)
// block_offsets: [B, 2, max_blocks_per_seq] int32 block indices into kv_pool ([:, 0] K, [:, 1] V)
// cu_q_seqlens:  [B + 1] int32 prefix sums of the new (query) lengths
// cu_kv_seqlens: [B + 1] int32 prefix sums of the full KV lengths (cached prefix + new tokens)
// max_q_len:     the largest q length (params.s)
// bmm1_scale:    the softmax scale of Q K^T (q_deq * k_deq / sqrt(256)); bmm2_scale: v_deq (and the output scale)
// split:         pass the split workspace (the last q-tile KV split may run); skip_leading: skipLeadingQTiles
// Returns [T, 24, 256] BF16, preset to NaN so rows the kernel does not write are visible.
Tensor context_fmha_sm120_test(Tensor const& q, Tensor const& kv_pool, Tensor const& block_offsets,
    Tensor const& cu_q_seqlens, Tensor const& cu_kv_seqlens, int64_t max_q_len, double bmm1_scale, double bmm2_scale,
    bool split, bool skip_leading)
{
    CHECK_TH_CUDA(q);
    CHECK_TH_CUDA(kv_pool);
    CHECK_TH_CUDA(block_offsets);
    CHECK_TH_CUDA(cu_q_seqlens);
    CHECK_TH_CUDA(cu_kv_seqlens);
    TORCH_CHECK(q.dim() == 3 && q.size(1) == kNumHeads && q.size(2) == kHeadSize && isFp8Bytes(q) && q.is_contiguous(),
        "q must be a contiguous [T, 24, 256] e4m3 tensor, got ", q.sizes());
    TORCH_CHECK(kv_pool.dim() == 4 && kv_pool.size(1) == kNumKvHeads && kv_pool.size(3) == kHeadSize
            && isFp8Bytes(kv_pool) && kv_pool.is_contiguous(),
        "kv_pool must be a contiguous [num_blocks, 4, tokens_per_block, 256] e4m3 tensor, got ", kv_pool.sizes());
    int const tokensPerBlock = static_cast<int>(kv_pool.size(2));
    TORCH_CHECK(tokensPerBlock > 0 && (tokensPerBlock & (tokensPerBlock - 1)) == 0,
        "tokens_per_block must be a power of 2, got ", tokensPerBlock);
    TORCH_CHECK(block_offsets.dim() == 3 && block_offsets.size(1) == 2
            && block_offsets.scalar_type() == at::ScalarType::Int && block_offsets.is_contiguous(),
        "block_offsets must be a contiguous [B, 2, max_blocks_per_seq] int32 tensor");
    int const b = static_cast<int>(block_offsets.size(0));
    TORCH_CHECK(cu_q_seqlens.scalar_type() == at::ScalarType::Int && cu_q_seqlens.numel() == b + 1
            && cu_kv_seqlens.scalar_type() == at::ScalarType::Int && cu_kv_seqlens.numel() == b + 1
            && cu_q_seqlens.is_contiguous() && cu_kv_seqlens.is_contiguous(),
        "cu_q_seqlens / cu_kv_seqlens must be contiguous [B + 1] int32 tensors");
    TORCH_CHECK(max_q_len >= 1, "max_q_len must be positive");

    at::cuda::OptionalCUDAGuard const deviceGuard(device_of(q));
    auto const device = q.device();
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(q.get_device());
    int64_t const t = q.size(0);
    Tensor out = at::full({t, kNumHeads, kHeadSize}, std::numeric_limits<float>::quiet_NaN(),
        q.options().dtype(at::ScalarType::BFloat16));

    // The device scales as BuildDecoderInfo writes them: bmm1 = {scale, scale * log2(e)}, bmm2 = {scale}.
    float const bmm1 = static_cast<float>(bmm1_scale);
    float const bmm2 = static_cast<float>(bmm2_scale);
    Tensor scales
        = torch::tensor({bmm1, bmm1 * 1.4426950408889634f, bmm2}, torch::TensorOptions().dtype(torch::kFloat32))
              .to(device, /*non_blocking=*/false);
    Tensor splitWorkspace;
    if (split)
    {
        splitWorkspace = at::empty({static_cast<int64_t>(kernels::kContextFmhaSm120SplitWorkspaceBytes)},
            q.options().dtype(at::ScalarType::Byte));
    }

    auto floatBits = [](float x)
    {
        uint32_t u;
        std::memcpy(&u, &x, sizeof(u));
        return u;
    };

    bert::Fused_multihead_attention_params_v2 params{};
    params.b = b;
    params.s = static_cast<int>(max_q_len);
    params.d = kHeadSize;
    params.dv = kHeadSize;
    params.h = kNumHeads;
    params.h_kv = kNumKvHeads;
    params.h_q_per_kv = kNumHeads / kNumKvHeads;
    // The attention op sets max_seq_len; the causal mask (MASK_VERSION 3) does not read it.
    params.sliding_window_size = INT_MAX;
    params.log2_chunked_attention_size = 0;
    params.is_s_padded = false;
    params.num_grouped_heads = 1;

    params.q_ptr = q.data_ptr();
    params.q_stride_in_bytes = kNumHeads * kHeadSize;
    params.cu_q_seqlens = cu_q_seqlens.data_ptr<int>();
    params.cu_kv_seqlens = cu_kv_seqlens.data_ptr<int>();

    // KVBlockArray::copyKVBlockArrayForContextFMHA(): one token is 4 heads x 256 e4m3 bytes.
    int const maxBlocksPerSeq = static_cast<int>(block_offsets.size(2));
    tensorrt_llm::kernels::KVBlockArrayForContextFMHA const kvArray(b, maxBlocksPerSeq, tokensPerBlock,
        kNumKvHeads * kHeadSize, kv_pool.data_ptr(),
        reinterpret_cast<tensorrt_llm::kernels::KVCacheIndex const*>(block_offsets.data_ptr<int32_t>()));
    static_assert(sizeof(kvArray) == sizeof(params.paged_kv_cache), "KVBlockArrayForContextFMHA != Kv_block_array");
    std::memcpy(&params.paged_kv_cache, &kvArray, sizeof(kvArray));
    // One head's slab inside a page.
    params.k_stride_in_bytes = static_cast<int64_t>(tokensPerBlock) * kHeadSize;
    params.v_stride_in_bytes = params.k_stride_in_bytes;

    params.o_ptr = out.data_ptr();
    params.o_stride_in_bytes = kNumHeads * kHeadSize * 2;
    params.softmax_stats_ptr = nullptr;
    params.softmax_stats_stride_in_bytes = sizeof(float) * 2 * kNumHeads;

    params.scale_bmm1 = floatBits(bmm1);
    params.scale_softmax = floatBits(1.f);
    params.scale_bmm2 = floatBits(bmm2);
    params.softcapping_scale_bmm1 = 0;
    params.scale_bmm1_d = reinterpret_cast<uint32_t*>(scales.data_ptr<float>());
    params.scale_bmm2_d = reinterpret_cast<uint32_t*>(scales.data_ptr<float>() + 2);
    params.tile_id_counter_ptr = nullptr;
    params.has_alibi = false;
    params.attention_sinks = nullptr;

    int const multiProcessorCount = at::cuda::getDeviceProperties(q.get_device())->multiProcessorCount;
    kernels::launchContextFmhaSm120(
        params, multiProcessorCount, split ? splitWorkspace.data_ptr() : nullptr, skip_leading, stream);
    return out;
}

} // namespace

TRTLLM_MV2_NAMESPACE_END

TRTLLM_MV2_LIBRARY_FRAGMENT(m)
{
    m.def(
        "context_fmha_sm120_test(Tensor q, Tensor kv_pool, Tensor block_offsets, Tensor cu_q_seqlens,"
        " Tensor cu_kv_seqlens, int max_q_len, float bmm1_scale, float bmm2_scale, bool split,"
        " bool skip_leading) -> Tensor");
}

TRTLLM_MV2_LIBRARY_IMPL(CUDA, m)
{
    m.impl("context_fmha_sm120_test", &TRTLLM_MV2_NS::context_fmha_sm120_test);
}
