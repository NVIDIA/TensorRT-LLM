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

#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/quantization.h"
#include "tensorrt_llm/kernels/glmMoeDsaRubin/glmMlaKvHalfAppend.h"
#include "tensorrt_llm/runtime/torchUtils.h"
#include "tensorrt_llm/thop/attentionOp.h"

#include <cuda_bf16.h>
#include <torch/extension.h>

#include <cstdint>
#include <limits>
#include <optional>

namespace th = torch;
namespace tc = tensorrt_llm::common;
namespace tk = tensorrt_llm::kernels::glm_moe_dsa;

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{

namespace
{

constexpr int64_t kInt32Max = std::numeric_limits<int32_t>::max();
constexpr int64_t kLatentRowDim = tk::kGlmKvHalfLatentDim + tk::kGlmKvHalfRopeDim;
// The kernel walks the latent rows with 16-byte vector loads.
constexpr int64_t kVectorBytes = 16;
// Per position, the rope table holds 64 (cos, sin) float pairs.
constexpr int64_t kRopeTableFloatsPerPosition = 2 * tk::kGlmKvHalfRopeDim;

} // namespace

// The KV half of GLM-5.2's MLA generation step: kv_a_layernorm of the 512 latent columns of the raw kv_a_proj output,
// RoPE of its 64 K columns, E4M3 at unit scale, appended to layer layer_idx of the paged FP8 cache at the generation
// positions sequence_length[num_contexts + s] - seq_len_q + t, plus the generation attention's scales in
// mla_bmm1_scale / mla_bmm2_scale. The paged-cache arguments are those of mla_rope_generation.
void glmMlaKvHalfAppend(th::Tensor const& latent, th::Tensor const& kvNormWeight, double kvNormEps,
    th::Tensor const& rotaryCosSin, th::Tensor const& sequenceLength, int64_t numContexts, int64_t seqLenQ,
    th::Tensor const& kvCacheBlockOffsets, th::Tensor const& hostKvCachePoolPointers,
    th::Tensor const& hostKvCachePoolMapping, int64_t layerIdx, int64_t tokensPerBlock, int64_t attentionWindowSize,
    th::Tensor& mlaBmm1Scale, th::Tensor& mlaBmm2Scale, double softmaxScale, bool triggerAtEntry)
{
    TORCH_CHECK(latent.is_cuda() && kvNormWeight.is_cuda() && rotaryCosSin.is_cuda() && sequenceLength.is_cuda()
            && kvCacheBlockOffsets.is_cuda() && mlaBmm1Scale.is_cuda() && mlaBmm2Scale.is_cuda(),
        "glm_mla_kv_half_append takes CUDA tensors besides the host pool pointers and mapping");
    auto const device = latent.device();
    TORCH_CHECK(kvNormWeight.device() == device && rotaryCosSin.device() == device && sequenceLength.device() == device
            && kvCacheBlockOffsets.device() == device && mlaBmm1Scale.device() == device
            && mlaBmm2Scale.device() == device,
        "glm_mla_kv_half_append takes its CUDA tensors on one device");

    TORCH_CHECK(latent.scalar_type() == th::kBFloat16 && latent.dim() == 2 && latent.size(1) >= kLatentRowDim
            && latent.stride(1) == 1,
        "latent must be a BF16 [tokens, >= ", kLatentRowDim, "] view with unit column stride");
    TORCH_CHECK(latent.stride(0) % (kVectorBytes / 2) == 0 && latent.stride(0) <= kInt32Max
            && reinterpret_cast<uintptr_t>(latent.data_ptr()) % kVectorBytes == 0,
        "latent rows must be 16-byte aligned (row stride a multiple of 8 elements), got row stride ", latent.stride(0));
    int64_t const numTokens = latent.size(0);
    TORCH_CHECK(numTokens <= kInt32Max, "too many latent rows");
    TORCH_CHECK(seqLenQ > 0 && numTokens % seqLenQ == 0, "seq_len_q must be positive and divide the latent rows; got ",
        seqLenQ, " for ", numTokens, " rows");
    int64_t const numGenerations = numTokens / seqLenQ;

    TORCH_CHECK(kvNormWeight.scalar_type() == th::kBFloat16 && kvNormWeight.is_contiguous()
            && kvNormWeight.numel() == tk::kGlmKvHalfLatentDim
            && reinterpret_cast<uintptr_t>(kvNormWeight.data_ptr()) % kVectorBytes == 0,
        "kv_norm_weight must be a contiguous, 16-byte aligned BF16 tensor of ", tk::kGlmKvHalfLatentDim, " elements");
    TORCH_CHECK(rotaryCosSin.scalar_type() == th::kFloat32 && rotaryCosSin.is_contiguous()
            && rotaryCosSin.numel() % kRopeTableFloatsPerPosition == 0,
        "rotary_cos_sin must be a contiguous float32 table of ", kRopeTableFloatsPerPosition, " values per position");
    TORCH_CHECK(sequenceLength.scalar_type() == th::kInt32 && sequenceLength.dim() == 1
            && sequenceLength.is_contiguous() && numContexts >= 0
            && sequenceLength.size(0) >= numContexts + numGenerations,
        "sequence_length must be a contiguous int32 tensor covering the context and generation sequences");
    // KVBlockArray addresses sequence s of the pool's rows at s * 2 * max_blocks.
    TORCH_CHECK(kvCacheBlockOffsets.scalar_type() == th::kInt32 && kvCacheBlockOffsets.dim() == 4
            && kvCacheBlockOffsets.size(2) == 2 && kvCacheBlockOffsets.stride(3) == 1
            && kvCacheBlockOffsets.stride(2) == kvCacheBlockOffsets.size(3)
            && kvCacheBlockOffsets.stride(1) == 2 * kvCacheBlockOffsets.size(3)
            && kvCacheBlockOffsets.size(1) >= numContexts + numGenerations,
        "kv_cache_block_offsets must be an int32 [pools, sequences, 2, max_blocks] tensor, dense in its last three "
        "dimensions");
    TORCH_CHECK(hostKvCachePoolPointers.device().is_cpu() && hostKvCachePoolMapping.device().is_cpu(),
        "host_kv_cache_pool_pointers and host_kv_cache_pool_mapping must be CPU tensors");
    TORCH_CHECK(tokensPerBlock > 0 && (tokensPerBlock & (tokensPerBlock - 1)) == 0,
        "tokens_per_block must be a power of two, got ", tokensPerBlock);
    TORCH_CHECK(attentionWindowSize > 0 && attentionWindowSize <= kInt32Max, "invalid attention_window_size ",
        attentionWindowSize);
    TORCH_CHECK(mlaBmm1Scale.scalar_type() == th::kFloat32 && mlaBmm1Scale.is_contiguous() && mlaBmm1Scale.numel() >= 2,
        "mla_bmm1_scale must hold two contiguous float32 values");
    TORCH_CHECK(mlaBmm2Scale.scalar_type() == th::kFloat32 && mlaBmm2Scale.numel() >= 1,
        "mla_bmm2_scale must hold a float32 value");

    // The latent pool of the FP8 MLA cache: one 576-element head per token, one byte per element.
    auto const kvCacheBuffers = buildPagedKvCacheBuffers(std::optional<th::Tensor>(kvCacheBlockOffsets),
        std::optional<th::Tensor>(hostKvCachePoolPointers), std::optional<th::Tensor>(hostKvCachePoolMapping),
        tc::QuantMode::fp8KvCache(), layerIdx, numGenerations, tokensPerBlock, /*kv_head_num=*/1, kLatentRowDim,
        attentionWindowSize, attentionWindowSize, /*beam_width=*/1, numContexts, /*is_mla_enable=*/true,
        latent.element_size());
    TORCH_CHECK(kvCacheBuffers.kvCacheBuffer.data != nullptr, "glm_mla_kv_half_append found no paged KV cache");

    auto const stream = at::cuda::getCurrentCUDAStream(latent.get_device());
    tk::invokeGlmMlaKvHalfAppend(static_cast<__nv_bfloat16 const*>(latent.data_ptr()),
        static_cast<int32_t>(latent.stride(0)), kvCacheBuffers.kvCacheBuffer,
        reinterpret_cast<float2 const*>(rotaryCosSin.data_ptr<float>()),
        static_cast<__nv_bfloat16 const*>(kvNormWeight.data_ptr()), static_cast<float>(kvNormEps),
        static_cast<int32_t>(numTokens), static_cast<int32_t>(seqLenQ),
        sequenceLength.data_ptr<int32_t>() + numContexts, mlaBmm1Scale.data_ptr<float>(),
        mlaBmm2Scale.data_ptr<float>(), static_cast<float>(softmaxScale), triggerAtEntry, stream);
}

} // namespace torch_ext

TRTLLM_NAMESPACE_END

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "glm_mla_kv_half_append(Tensor latent, Tensor kv_norm_weight, float kv_norm_eps, Tensor rotary_cos_sin, "
        "Tensor sequence_length, int num_contexts, int seq_len_q, Tensor kv_cache_block_offsets, "
        "Tensor host_kv_cache_pool_pointers, Tensor host_kv_cache_pool_mapping, int layer_idx, int tokens_per_block, "
        "int attention_window_size, Tensor(a!) mla_bmm1_scale, Tensor(b!) mla_bmm2_scale, float softmax_scale, "
        "bool trigger_at_entry=False) -> ()");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("glm_mla_kv_half_append", &tensorrt_llm::torch_ext::glmMlaKvHalfAppend);
}
