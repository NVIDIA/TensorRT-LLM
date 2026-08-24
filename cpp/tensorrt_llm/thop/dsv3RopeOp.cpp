/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "tensorrt_llm/common/attentionOp.h"
#include "tensorrt_llm/common/dataType.h"
#include "tensorrt_llm/common/memoryUtils.h"
#include "tensorrt_llm/kernels/gptKernels.h"
#include "tensorrt_llm/kernels/mlaKernels.h"
#include "tensorrt_llm/kernels/unfusedAttentionKernels.h"
#include "tensorrt_llm/runtime/torchUtils.h"
#include "tensorrt_llm/runtime/utils/debugUtils.h"
#include "tensorrt_llm/thop/attentionOp.h"
#include "tensorrt_llm/thop/thUtils.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <torch/extension.h>

#include <cmath>
#include <cstdint>
#include <optional>
#include <vector>

namespace tk = tensorrt_llm::kernels;
namespace tc = tensorrt_llm::common;
namespace tr = tensorrt_llm::runtime;

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{

// Wrapper for MLA rope generation arguments
struct MlaRopeGenArgs
{
    int32_t q_pe_ld;
    int32_t q_pe_stride;
    float2 const* rotary_cos_sin_ptr;
    int32_t num_generations;
    int32_t num_gen_tokens;
    int32_t num_heads;
    tk::MlaMetaParams mla_meta_params;
    int32_t const* sequence_lengths_ptr;
    int32_t max_context_q_len;
    int const* block_ids_per_seq_ptr;
    tk::KvCacheDataType cache_type;
    int* cu_q_seqlens_ptr;
    int* cu_kv_seqlens_ptr;
    uint32_t* fmha_tile_counter_ptr;
    float* mla_bmm1_scale_ptr;
    float* mla_bmm2_scale_ptr;
    void* quant_q_buffer_ptr;
    // q_b_layernorm already wrote the FP8 q_nope segment, so the RoPE kernel drops
    // its quantize region and reuses this scale for the rope segment.
    float const* quant_scale_qkv_ptr;
    float const* quant_scale_o_ptr;
    float const* kv_scale_orig_quant_ptr;
    float const* kv_scale_quant_orig_ptr;
    float host_bmm1_scale;
    int32_t const* helix_position_offsets_ptr;
    bool const* helix_is_inactive_rank_ptr;
    // Per-token KV write slots for speculative verify groups (nullptr otherwise).
    int32_t const* helix_local_slots_ptr;
    // `kv_norm_weight_ptr` set: `invokeMLAKvNormRopeQuantGeneration` produces the KV
    // half (norm + rope + fp8 + paged write) and the RoPE kernel runs Q-only.
    void const* kv_norm_weight_ptr;
    float kv_norm_eps;
    int32_t latent_row_stride;
    // Filled once per iteration by the attention metadata; do not recompute per layer.
    bool precomputed_cu_seqlens;
    // The DSv4 sparse indices kernel already emits the FMHA scheduler prologue.
    bool precomputed_fmha_scheduler;
    // Split launch: the KV kernel needs only the raw kv_a_proj latent, so the caller
    // can run it early on its own stream and return for Q once q_pe exists.
    // `kv_only` launches the KV half, `kv_done_elsewhere` the Q half.
    bool kv_only;
    bool kv_done_elsewhere;
};

template <typename T, typename KVCacheBuffer>
void invokeMLARopeGenerationHelper(T const* latent_cache_ptr, T* q_pe_ptr, T* fused_q_ptr,
    KVCacheBuffer& kv_cache_buffer, MlaRopeGenArgs const& args, cudaStream_t stream)
{
    tk::MlaParams<T> mla_params{};
    mla_params.latent_cache = latent_cache_ptr;
    mla_params.q_pe = q_pe_ptr;
    mla_params.q_pe_ld = args.q_pe_ld;
    mla_params.q_pe_stride = args.q_pe_stride;
    mla_params.q_buf = fused_q_ptr;
    mla_params.cos_sin_cache = args.rotary_cos_sin_ptr;
    mla_params.batch_size = args.num_generations;
    mla_params.acc_q_len = args.num_gen_tokens;
    mla_params.head_num = args.num_heads;
    mla_params.meta = args.mla_meta_params;

    mla_params.cache_seq_lens = args.sequence_lengths_ptr;
    mla_params.max_input_seq_len = args.max_context_q_len;

    mla_params.block_ids_per_seq = args.block_ids_per_seq_ptr;

    mla_params.cache_type = args.cache_type;

    mla_params.seqQOffset = args.cu_q_seqlens_ptr;
    mla_params.cu_kv_seqlens = args.cu_kv_seqlens_ptr;
    mla_params.fmha_tile_counter = args.fmha_tile_counter_ptr;
    mla_params.bmm1_scale = args.mla_bmm1_scale_ptr;
    mla_params.bmm2_scale = args.mla_bmm2_scale_ptr;
    mla_params.quant_q_buf = args.quant_q_buffer_ptr;
    mla_params.quant_scale_qkv = args.quant_scale_qkv_ptr;
    mla_params.fuse_q_fp8_in_rope = args.quant_scale_qkv_ptr != nullptr;

    mla_params.quant_scale_o = args.quant_scale_o_ptr;
    mla_params.quant_scale_q = args.kv_scale_orig_quant_ptr;
    mla_params.quant_scale_kv = args.kv_scale_orig_quant_ptr;
    mla_params.dequant_scale_q = args.kv_scale_quant_orig_ptr;
    mla_params.dequant_scale_kv = args.kv_scale_quant_orig_ptr;
    mla_params.host_bmm1_scale = args.host_bmm1_scale;
    mla_params.helix_position_offsets = args.helix_position_offsets_ptr;
    mla_params.helix_is_inactive_rank = args.helix_is_inactive_rank_ptr;
    mla_params.helix_local_slots = args.helix_local_slots_ptr;

    mla_params.precomputed_cu_seqlens = args.precomputed_cu_seqlens;
    mla_params.precomputed_fmha_scheduler = args.precomputed_fmha_scheduler;
    // Set whether the KV kernel ran in this call or already ran on another stream.
    mla_params.fuse_kv_norm_in_rope = args.kv_norm_weight_ptr != nullptr || args.kv_done_elsewhere;
    mla_params.kv_norm_weight = args.kv_norm_weight_ptr;
    mla_params.kv_norm_eps = args.kv_norm_eps;
    mla_params.latent_row_stride = args.latent_row_stride;

    if (args.kv_norm_weight_ptr != nullptr)
    {
        // KV first: the RoPE kernel no longer writes the latent, and FMHA reads the cache.
        tk::invokeMLAKvNormRopeQuantGeneration<T>(mla_params, kv_cache_buffer, stream);
    }
    if (!args.kv_only)
    {
        tk::invokeMLARopeGeneration<T>(mla_params, kv_cache_buffer, stream);
    }
}

void MLARopeGeneration(std::optional<torch::Tensor> fused_q, // [tokens, num_heads, (nope_dim + rope+dim)]
    std::optional<torch::Tensor> q_pe,                       // [tokens, num_heads, rope_dim]
    torch::Tensor latent_cache,                              // [tokens, kv_lora_rank + rope_dim]
    std::optional<torch::Tensor> rotary_cos_sin, torch::Tensor cu_q_seqlens, torch::Tensor cu_kv_seqlens,
    torch::Tensor fmha_scheduler_counter, std::optional<torch::Tensor> mla_bmm1_scale,
    std::optional<torch::Tensor> mla_bmm2_scale, std::optional<torch::Tensor> quant_q_buffer,
    torch::Tensor sequence_length, torch::Tensor host_past_key_value_lengths, torch::Tensor host_context_lengths,
    int64_t const num_contexts, std::optional<torch::Tensor> kv_cache_block_offsets,
    std::optional<torch::Tensor> host_kv_cache_pool_pointers, std::optional<torch::Tensor> host_kv_cache_pool_mapping,
    torch::optional<torch::Tensor> kv_scale_orig_quant, // [1] q,k quant scale
    torch::optional<torch::Tensor> kv_scale_quant_orig, // [1] bmm quant scale
    torch::optional<torch::Tensor> out_scale,           // [1] output quant scale
    std::optional<torch::Tensor> block_ids_per_seq, std::vector<std::optional<torch::Tensor>> helix_tensor_params,
    int64_t const predicted_tokens_per_seq, int64_t const layer_idx, int64_t const num_heads,
    int64_t const num_kv_heads, int64_t const head_size,

    int64_t const tokens_per_block, int64_t const attention_window_size, int64_t const beam_width,
    int64_t const quant_mode, double const q_scaling, int64_t q_lora_rank, int64_t kv_lora_rank,
    int64_t qk_nope_head_dim, int64_t qk_rope_head_dim, int64_t v_head_dim, bool rope_append,
    std::optional<torch::Tensor> kv_norm_weight, double const kv_norm_eps, bool const precomputed_cu_seqlens,
    bool const precomputed_fmha_scheduler, bool const kv_only, bool const kv_done_elsewhere,
    std::optional<torch::Tensor> quant_scale_qkv)
{
    // `kv_only` runs before q_pe exists, so the Q tensors are absent.
    TORCH_CHECK(kv_only || (fused_q.has_value() && q_pe.has_value()),
        "mla_rope_generation needs fused_q and q_pe unless kv_only is set");
    TORCH_CHECK(!kv_only || kv_norm_weight.has_value(), "kv_only requires kv_norm_weight");
    TORCH_CHECK(!(kv_only && kv_done_elsewhere), "kv_only and kv_done_elsewhere are mutually exclusive");

    TLLM_CHECK_WITH_INFO(
        head_size == kv_lora_rank + qk_rope_head_dim, "head_size must = kv_lora_rank + qk_rope_head_dim");
    TLLM_CHECK_WITH_INFO(num_kv_heads == 1, "num_kv_heads must = 1");
    TORCH_CHECK(helix_tensor_params.size() == 2 || helix_tensor_params.size() == 3,
        "Expecting 2 or 3 tensors for helix_tensor_params: helix_position_offsets, helix_is_inactive_rank "
        "and optionally helix_local_slots (per-token KV write slots for speculative verify groups).");

    auto stream = at::cuda::getCurrentCUDAStream(latent_cache.get_device());
    auto const kv_cache_quant_mode = tc::QuantMode(uint32_t(quant_mode));
    bool const use_gen_flash_mla = tc::getSMVersion() == 90 && tokens_per_block == 64;
    TLLM_CHECK_WITH_INFO(!kv_cache_quant_mode.hasFp4KvCache(), "FP4 KV cache is not supported for MLA generation.");
    TLLM_CHECK_WITH_INFO(
        host_kv_cache_pool_mapping.has_value(), "KV cache pool mapping is required for MLA generation.");

    int32_t const num_seqs = host_context_lengths.size(0);

    int32_t const num_tokens = latent_cache.size(0);
    int32_t const num_generations = num_seqs - num_contexts;
    int32_t const num_gen_tokens = num_tokens;
    int32_t const seq_offset = num_contexts;
    auto const& helix_position_offsets = helix_tensor_params[0];
    auto const& helix_is_inactive_rank = helix_tensor_params[1];
    int32_t const layer_num = host_kv_cache_pool_mapping.value().size(0);

    tk::MlaMetaParams mla_meta_params = {static_cast<int>(q_lora_rank), static_cast<int>(kv_lora_rank),
        static_cast<int>(qk_nope_head_dim), static_cast<int>(qk_rope_head_dim), static_cast<int>(v_head_dim),
        static_cast<int>(predicted_tokens_per_seq), static_cast<int>(layer_num), static_cast<int>(rope_append)};

    int32_t const* helix_position_offsets_ptr
        = helix_position_offsets.has_value() ? helix_position_offsets->data_ptr<int32_t>() : nullptr;
    bool const* helix_is_inactive_rank_ptr
        = helix_is_inactive_rank.has_value() ? helix_is_inactive_rank->data_ptr<bool>() : nullptr;
    int32_t const* helix_local_slots_ptr = nullptr;
    if (helix_tensor_params.size() == 3 && helix_tensor_params[2].has_value())
    {
        helix_local_slots_ptr = helix_tensor_params[2]->data_ptr<int32_t>();
        TORCH_CHECK(!kv_norm_weight.has_value(),
            "helix_local_slots (speculative verify groups) is not supported on the fused "
            "KV-norm RoPE path: its KV append kernel has no per-token helix gate.");
    }

    int* cu_q_seqlens_ptr = reinterpret_cast<int*>(cu_q_seqlens.data_ptr());
    int* cu_kv_seqlens_ptr = reinterpret_cast<int*>(cu_kv_seqlens.data_ptr());
    uint32_t* fmha_tile_counter_ptr = reinterpret_cast<uint32_t*>(fmha_scheduler_counter.data_ptr());
    float* mla_bmm1_scale_ptr
        = mla_bmm1_scale.has_value() ? reinterpret_cast<float*>(mla_bmm1_scale.value().data_ptr()) : nullptr;
    float* mla_bmm2_scale_ptr
        = mla_bmm2_scale.has_value() ? reinterpret_cast<float*>(mla_bmm2_scale.value().data_ptr()) : nullptr;
    void* quant_q_buffer_ptr
        = quant_q_buffer.has_value() ? reinterpret_cast<void*>(quant_q_buffer.value().data_ptr()) : nullptr;

    float2 const* rotary_cos_sin_ptr = nullptr;
    if (rotary_cos_sin.has_value())
    {
        rotary_cos_sin_ptr = reinterpret_cast<float2 const*>(rotary_cos_sin.value().data_ptr());
    }

    int const* sequence_lengths_ptr = sequence_length.slice(0, seq_offset).data_ptr<int>();
    // Note we still need context length during generation for MMHA optimization.
    int32_t const max_context_q_len
        = host_context_lengths.slice(0, seq_offset, seq_offset + num_generations).max().item<int32_t>();

    int32_t q_pe_ld = 0;
    int32_t q_pe_stride = 0;
    if (!kv_only)
    {
        TORCH_CHECK(q_pe->defined());
        TORCH_CHECK(q_pe->dim() == 3);
        TORCH_CHECK(q_pe->strides()[2] == 1);
        q_pe_ld = q_pe->strides()[1];
        q_pe_stride = q_pe->strides()[0];
    }

    bool const fp8_context_fmha = kv_cache_quant_mode.hasFp8KvCache();
    int32_t const batch_beam = beam_width * num_generations;

    auto kv_cache_buffer = tensorrt_llm::torch_ext::buildPagedKvCacheBuffers(kv_cache_block_offsets,
        host_kv_cache_pool_pointers, host_kv_cache_pool_mapping, kv_cache_quant_mode, layer_idx, batch_beam,
        tokens_per_block, num_kv_heads, head_size, attention_window_size, attention_window_size, beam_width, seq_offset,
        true /*is_mla_enable*/, static_cast<size_t>(latent_cache.element_size()))
                               .kvCacheBuffer;

    tk::KvCacheDataType cache_type = tk::cacheTypeFromQuantMode(kv_cache_quant_mode);

    float const* kv_scale_orig_quant_ptr = nullptr;
    float const* kv_scale_quant_orig_ptr = nullptr;
    if (kv_cache_quant_mode.hasKvCacheQuant() && kv_scale_orig_quant.has_value() && kv_scale_quant_orig.has_value())
    {
        kv_scale_orig_quant_ptr = kv_scale_orig_quant.value().data_ptr<float>();
        kv_scale_quant_orig_ptr = kv_scale_quant_orig.value().data_ptr<float>();
    }

    float const* quant_scale_o_ptr
        = (fp8_context_fmha && out_scale.has_value()) ? out_scale.value().data_ptr<float>() : nullptr;
    float const host_bmm1_scale = 1.f / (q_scaling * sqrt(static_cast<float>(qk_nope_head_dim + qk_rope_head_dim)));

    if (use_gen_flash_mla)
    {
        TLLM_CHECK_WITH_INFO(block_ids_per_seq.has_value(), "block_ids_per_seq is required for gen flash mla");
    }
    int const* block_ids_per_seq_ptr = use_gen_flash_mla && block_ids_per_seq.has_value()
        ? static_cast<int*>(block_ids_per_seq->data_ptr())
        : nullptr;

    // Fused kv-norm: `latent_cache` is the RAW kv_a_proj slice, a last-dim view whose
    // row stride exceeds the row width, so read the stride off the tensor.
    float const* quant_scale_qkv_ptr = nullptr;
    if (quant_scale_qkv.has_value())
    {
        TORCH_CHECK(quant_q_buffer.has_value(), "quant_scale_qkv requires quant_q_buffer");
        quant_scale_qkv_ptr = quant_scale_qkv->data_ptr<float>();
    }

    void const* kv_norm_weight_ptr = nullptr;
    int32_t latent_row_stride = static_cast<int32_t>(latent_cache.stride(0));
    if (kv_norm_weight.has_value())
    {
        TORCH_CHECK(latent_cache.stride(-1) == 1, "fused kv-norm needs a unit-stride latent row");
        TORCH_CHECK(kv_norm_weight->scalar_type() == latent_cache.scalar_type(),
            "kv_norm_weight dtype must match latent_cache");
        TORCH_CHECK(kv_norm_weight->numel() == kv_lora_rank + qk_rope_head_dim,
            "kv_norm_weight must have kv_lora_rank + qk_rope_head_dim elements");
        // The kernel walks rows with 16-byte vector loads, so a row start that is not
        // 16B-aligned faults with a bare misaligned-address error far from here.
        auto const kEltsPer16B = 16 / latent_cache.element_size();
        TORCH_CHECK(latent_row_stride % kEltsPer16B == 0, "latent_cache row stride (", latent_row_stride,
            ") must be a multiple of ", kEltsPer16B, " for the fused kv-norm 16B vector loads");
        TORCH_CHECK(reinterpret_cast<uintptr_t>(latent_cache.data_ptr()) % 16 == 0,
            "latent_cache must be 16B-aligned for the fused kv-norm vector loads");
        kv_norm_weight_ptr = kv_norm_weight->data_ptr();
    }

    // Currently NVFP4 KV cache is not supported for MLA
    MlaRopeGenArgs args{q_pe_ld, q_pe_stride, rotary_cos_sin_ptr, num_generations, num_gen_tokens,
        static_cast<int32_t>(num_heads), mla_meta_params, sequence_lengths_ptr, max_context_q_len,
        block_ids_per_seq_ptr, cache_type, cu_q_seqlens_ptr, cu_kv_seqlens_ptr, fmha_tile_counter_ptr,
        mla_bmm1_scale_ptr, mla_bmm2_scale_ptr, quant_q_buffer_ptr, quant_scale_qkv_ptr, quant_scale_o_ptr,
        kv_scale_orig_quant_ptr, kv_scale_quant_orig_ptr, host_bmm1_scale, helix_position_offsets_ptr,
        helix_is_inactive_rank_ptr, helix_local_slots_ptr, kv_norm_weight_ptr, static_cast<float>(kv_norm_eps),
        latent_row_stride, precomputed_cu_seqlens, precomputed_fmha_scheduler, kv_only, kv_done_elsewhere};

    void* q_pe_ptr = kv_only ? nullptr : q_pe->data_ptr();
    void* fused_q_ptr = kv_only ? nullptr : fused_q->data_ptr();

    auto const input_dtype = latent_cache.scalar_type();
    if (input_dtype == torch::kFloat16)
    {
        invokeMLARopeGenerationHelper(static_cast<half const*>(latent_cache.data_ptr()), static_cast<half*>(q_pe_ptr),
            static_cast<half*>(fused_q_ptr), kv_cache_buffer, args, stream);
    }
    else if (input_dtype == torch::kBFloat16)
    {

        invokeMLARopeGenerationHelper(static_cast<__nv_bfloat16 const*>(latent_cache.data_ptr()),
            static_cast<__nv_bfloat16*>(q_pe_ptr), static_cast<__nv_bfloat16*>(fused_q_ptr), kv_cache_buffer, args,
            stream);
    }
    else if (input_dtype == torch::kFloat32)
    {
        invokeMLARopeGenerationHelper(static_cast<float const*>(latent_cache.data_ptr()), static_cast<float*>(q_pe_ptr),
            static_cast<float*>(fused_q_ptr), kv_cache_buffer, args, stream);
    }
    else
    {
        TLLM_CHECK_WITH_INFO(false, "Unsupported input dtype: %s", c10::toString(input_dtype));
    }
}

} // namespace torch_ext

TRTLLM_NAMESPACE_END

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "mla_rope_generation("
        "Tensor(a!)? fused_q"
        ", Tensor(a!)? q_pe"
        ", Tensor latent_cache"
        ", Tensor? rotary_cos_sin"
        ", Tensor cu_q_seqlens"
        ", Tensor cu_kv_seqlens"
        ", Tensor fmha_scheduler_counter"
        ", Tensor? mla_bmm1_scale"
        ", Tensor? mla_bmm2_scale"
        ", Tensor? quant_q_buffer"
        ", Tensor sequence_length"
        ", Tensor host_past_key_value_lengths"
        ", Tensor host_context_lengths"
        ", int num_contexts"
        ", Tensor? kv_cache_block_offsets"
        ", Tensor? host_kv_cache_pool_pointers"
        ", Tensor? host_kv_cache_pool_mapping"
        ", Tensor? kv_scale_orig_quant"
        ", Tensor? kv_scale_quant_orig"
        ", Tensor? out_scale"
        ", Tensor? block_ids_per_seq"
        ", Tensor?[] helix_tensor_params"
        ", int predicted_tokens_per_seq"
        ", int layer_idx"
        ", int num_heads"
        ", int num_kv_heads"
        ", int head_size"
        ", int tokens_per_block"
        ", int attention_window_size"
        ", int beam_width"
        ", int quant_mode"
        ", float q_scaling"
        ", int q_lora_rank"
        ", int kv_lora_rank"
        ", int qk_nope_head_dim"
        ", int qk_rope_head_dim"
        ", int v_head_dim"
        ", bool rope_append"
        ", Tensor? kv_norm_weight=None"
        ", float kv_norm_eps=1e-6"
        ", bool precomputed_cu_seqlens=False"
        ", bool precomputed_fmha_scheduler=False"
        ", bool kv_only=False"
        ", bool kv_done_elsewhere=False"
        ", Tensor? quant_scale_qkv=None"
        ") -> ()");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("mla_rope_generation", &tensorrt_llm::torch_ext::MLARopeGeneration);
}
