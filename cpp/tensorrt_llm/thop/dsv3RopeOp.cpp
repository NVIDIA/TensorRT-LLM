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
#include "tensorrt_llm/runtime/torchUtils.h"
#include "tensorrt_llm/runtime/utils/debugUtils.h"
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
    float const* quant_scale_o_ptr;
    float const* kv_scale_orig_quant_ptr;
    float const* kv_scale_quant_orig_ptr;
    float host_bmm1_scale;
    int32_t const* helix_position_offsets_ptr;
    bool const* helix_is_inactive_rank_ptr;
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

    mla_params.quant_scale_o = args.quant_scale_o_ptr;
    mla_params.quant_scale_q = args.kv_scale_orig_quant_ptr;
    mla_params.quant_scale_kv = args.kv_scale_orig_quant_ptr;
    mla_params.dequant_scale_q = args.kv_scale_quant_orig_ptr;
    mla_params.dequant_scale_kv = args.kv_scale_quant_orig_ptr;
    mla_params.host_bmm1_scale = args.host_bmm1_scale;
    mla_params.helix_position_offsets = args.helix_position_offsets_ptr;
    mla_params.helix_is_inactive_rank = args.helix_is_inactive_rank_ptr;

    tk::invokeMLARopeGeneration<T>(mla_params, kv_cache_buffer, stream);
}

void MLARopeGeneration(torch::Tensor fused_q, // [tokens, num_heads, (nope_dim + rope+dim)]
    torch::Tensor q_pe,                       // [tokens, num_heads, rope_dim]
    torch::Tensor latent_cache,               // [tokens, kv_lora_rank + rope_dim]
    std::optional<torch::Tensor> rotary_cos_sin, torch::Tensor cu_q_seqlens, torch::Tensor cu_kv_seqlens,
    torch::Tensor fmha_scheduler_counter, std::optional<torch::Tensor> mla_bmm1_scale,
    std::optional<torch::Tensor> mla_bmm2_scale, std::optional<torch::Tensor> quant_q_buffer,
    torch::Tensor sequence_length, torch::Tensor host_past_key_value_lengths, torch::Tensor host_context_lengths,
    int64_t const num_contexts, std::optional<torch::Tensor> kv_cache_block_offsets,
    std::optional<torch::Tensor> host_kv_cache_pool_pointers, std::optional<torch::Tensor> host_kv_cache_pool_mapping,
    torch::optional<torch::Tensor> kv_scale_orig_quant, // [1] q,k quant scale
    torch::optional<torch::Tensor> kv_scale_quant_orig, // [1] bmm quant scale
    torch::optional<torch::Tensor> out_scale,           // [1] output quant scale
    std::optional<torch::Tensor> block_ids_per_seq, std::vector<std::optional<torch::Tensor>> mla_tensor_params,
    int64_t const predicted_tokens_per_seq, int64_t const layer_idx, int64_t const num_heads,
    int64_t const num_kv_heads, int64_t const head_size,

    int64_t const tokens_per_block, int64_t const attention_window_size, int64_t const sink_token_length,
    int64_t const beam_width, int64_t const quant_mode, double const q_scaling, int64_t q_lora_rank,
    int64_t kv_lora_rank, int64_t qk_nope_head_dim, int64_t qk_rope_head_dim, int64_t v_head_dim)
{
    TLLM_CHECK_WITH_INFO(
        head_size == kv_lora_rank + qk_rope_head_dim, "head_size must = kv_lora_rank + qk_rope_head_dim");
    TLLM_CHECK_WITH_INFO(num_kv_heads == 1, "num_kv_heads must = 1");
    TORCH_CHECK(mla_tensor_params.size() == 2,
        "Expecting 2 tensors for custom MLA tensor params: helix_position_offsets and helix_is_inactive_rank.");

    auto stream = at::cuda::getCurrentCUDAStream(fused_q.get_device());
    auto const kv_cache_quant_mode = tc::QuantMode(uint32_t(quant_mode));
    bool const use_gen_flash_mla = tc::getSMVersion() == 90 && tokens_per_block == 64;
    TLLM_CHECK_WITH_INFO(!kv_cache_quant_mode.hasFp4KvCache(), "FP4 KV cache is not supported for MLA generation.");
    TLLM_CHECK_WITH_INFO(
        host_kv_cache_pool_mapping.has_value(), "KV cache pool mapping is required for MLA generation.");

    bool const use_kv_cache = kv_cache_block_offsets.has_value() && host_kv_cache_pool_pointers.has_value()
        && host_kv_cache_pool_mapping.has_value();

    int32_t const num_seqs = host_context_lengths.size(0);

    int32_t const num_tokens = fused_q.size(0);
    int32_t const num_generations = num_seqs - num_contexts;
    int32_t const num_gen_tokens = num_tokens;
    int32_t const seq_offset = num_contexts;
    auto& mla_helix_position_offsets = mla_tensor_params[0];
    auto& mla_helix_is_inactive_rank = mla_tensor_params[1];
    int32_t const layer_num = host_kv_cache_pool_mapping.value().size(0);

    tk::MlaMetaParams mla_meta_params = {static_cast<int>(q_lora_rank), static_cast<int>(kv_lora_rank),
        static_cast<int>(qk_nope_head_dim), static_cast<int>(qk_rope_head_dim), static_cast<int>(v_head_dim),
        static_cast<int>(predicted_tokens_per_seq), static_cast<int>(layer_num)};

    int32_t const* helix_position_offsets_ptr
        = mla_helix_position_offsets.has_value() ? mla_helix_position_offsets->data_ptr<int32_t>() : nullptr;
    bool const* helix_is_inactive_rank_ptr
        = mla_helix_is_inactive_rank.has_value() ? mla_helix_is_inactive_rank->data_ptr<bool>() : nullptr;

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

    TORCH_CHECK(q_pe.defined());
    TORCH_CHECK(q_pe.dim() == 3);
    TORCH_CHECK(q_pe.strides()[2] == 1);
    int32_t const q_pe_ld = q_pe.strides()[1];
    int32_t const q_pe_stride = q_pe.strides()[0];

    // kv cache related
    auto const block_size = tokens_per_block * num_kv_heads * head_size;
    int32_t const elem_bytes
        = kv_cache_quant_mode.hasFp8KvCache() ? sizeof(__nv_fp8_e4m3) : static_cast<int32_t>(fused_q.element_size());

    int32_t const bytes_per_token = num_kv_heads * head_size * elem_bytes;

    auto const bytes_per_block = block_size * elem_bytes;
    int32_t const kv_factor = 1; // 1 for mla, 2 for mha/gqa
    bool const fp8_context_fmha = kv_cache_quant_mode.hasFp8KvCache();
    // Commonly, cyclic_attention_window_size, and max_attention_window_size will be the same
    // unless each layer has different attention window sizes.
    // the kv_cache capacity.
    // The cyclic_attention_window_size will determine the cyclic kv cache position of new tokens.
    // Note that this cyclic_attention_window_size might be smaller than the actual kv cache capactity.
    int const cyclic_attention_window_size = attention_window_size;
    int const max_cyclic_attention_window_size = cyclic_attention_window_size;
    bool const can_use_one_more_block = beam_width > 1;

    // kv cache pool related
    int32_t const max_blocks_per_sequence
        = use_kv_cache && kv_cache_block_offsets.has_value() ? kv_cache_block_offsets.value().size(-1) : 0;
    int32_t const pool_index = use_kv_cache && host_kv_cache_pool_mapping.has_value()
        ? host_kv_cache_pool_mapping.value().index({layer_idx, 0}).item<int32_t>()
        : 0;
    int32_t const layer_idx_in_cache_pool = use_kv_cache && host_kv_cache_pool_mapping.has_value()
        ? host_kv_cache_pool_mapping.value().index({layer_idx, 1}).item<int32_t>()
        : 0;
    auto const intra_pool_offset = layer_idx_in_cache_pool * kv_factor * bytes_per_block;

    tk::KVBlockArray::DataType* block_offsets
        = static_cast<tk::KVBlockArray::DataType*>(use_kv_cache && kv_cache_block_offsets.has_value()
                ? kv_cache_block_offsets.value().index({pool_index, seq_offset}).data_ptr()
                : nullptr);

    void* host_primary_pool_pointer{nullptr};
    void* host_secondary_pool_pointer{nullptr};

    if (use_kv_cache)
    {
        TORCH_CHECK(host_kv_cache_pool_pointers.value().dim() == 2);
        host_primary_pool_pointer = reinterpret_cast<void*>(
            reinterpret_cast<char*>(host_kv_cache_pool_pointers.value().index({pool_index, 0}).item<int64_t>())
            + intra_pool_offset);
        host_secondary_pool_pointer = reinterpret_cast<void*>(
            reinterpret_cast<char*>(host_kv_cache_pool_pointers.value().index({pool_index, 1}).item<int64_t>())
            + intra_pool_offset);
    }

    float const* kv_scale_orig_quant_ptr = nullptr; // qk quant scale
    float const* kv_scale_quant_orig_ptr = nullptr; // bmm quant scale
    if (kv_cache_quant_mode.hasKvCacheQuant() && kv_scale_orig_quant.has_value() && kv_scale_quant_orig.has_value())
    {
        kv_scale_orig_quant_ptr = kv_scale_orig_quant.value().data_ptr<float>();
        kv_scale_quant_orig_ptr = kv_scale_quant_orig.value().data_ptr<float>();
    }

    // Prepare scalars for MLA params and wrapper
    // For FP8 output, out_scale represents the output scale.
    float const* quant_scale_o_ptr
        = (fp8_context_fmha && out_scale.has_value()) ? out_scale.value().data_ptr<float>() : nullptr;
    float const host_bmm1_scale = 1.f / (q_scaling * sqrt(static_cast<float>(qk_nope_head_dim + qk_rope_head_dim)));

    if (use_gen_flash_mla)
    {
        TLLM_CHECK_WITH_INFO(block_ids_per_seq.has_value(), "block_ids_per_seq is required for gen flash mla");
    }
    int const* block_ids_per_seq_ptr = use_gen_flash_mla && block_ids_per_seq.has_value()
        ? static_cast<int*>(block_ids_per_seq->data_ptr())
        : nullptr; // only used for flash mla

    int32_t const batch_beam = beam_width * num_generations;

    tk::KvCacheDataType cache_type
        = (kv_cache_quant_mode.hasFp8KvCache() ? tk::KvCacheDataType::FP8 : tk::KvCacheDataType::BASE);

    auto kv_cache_buffer = tk::KVBlockArray(batch_beam, max_blocks_per_sequence, tokens_per_block, bytes_per_token,
        cyclic_attention_window_size, max_cyclic_attention_window_size, sink_token_length, can_use_one_more_block,
        host_primary_pool_pointer, host_secondary_pool_pointer, block_offsets);

    // Currently NVFP4 KV cache is not supported for MLA
    MlaRopeGenArgs args{q_pe_ld, q_pe_stride, rotary_cos_sin_ptr, num_generations, num_gen_tokens,
        static_cast<int32_t>(num_heads), mla_meta_params, sequence_lengths_ptr, max_context_q_len,
        block_ids_per_seq_ptr, cache_type, cu_q_seqlens_ptr, cu_kv_seqlens_ptr, fmha_tile_counter_ptr,
        mla_bmm1_scale_ptr, mla_bmm2_scale_ptr, quant_q_buffer_ptr, quant_scale_o_ptr, kv_scale_orig_quant_ptr,
        kv_scale_quant_orig_ptr, host_bmm1_scale, helix_position_offsets_ptr, helix_is_inactive_rank_ptr};

    auto const input_dtype = fused_q.scalar_type();
    if (input_dtype == torch::kFloat16)
    {
        invokeMLARopeGenerationHelper(static_cast<half const*>(latent_cache.data_ptr()),
            static_cast<half*>(q_pe.data_ptr()), static_cast<half*>(fused_q.data_ptr()), kv_cache_buffer, args, stream);
    }
    else if (input_dtype == torch::kBFloat16)
    {

        invokeMLARopeGenerationHelper(static_cast<__nv_bfloat16 const*>(latent_cache.data_ptr()),
            static_cast<__nv_bfloat16*>(q_pe.data_ptr()), static_cast<__nv_bfloat16*>(fused_q.data_ptr()),
            kv_cache_buffer, args, stream);
    }
    else if (input_dtype == torch::kFloat32)
    {
        invokeMLARopeGenerationHelper(static_cast<float const*>(latent_cache.data_ptr()),
            static_cast<float*>(q_pe.data_ptr()), static_cast<float*>(fused_q.data_ptr()), kv_cache_buffer, args,
            stream);
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
        "Tensor(a!) fused_q"
        ", Tensor(a!) q_pe"
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
        ", Tensor?[] mla_tensor_params"
        ", int predicted_tokens_per_seq"
        ", int layer_idx"
        ", int num_heads"
        ", int num_kv_heads"
        ", int head_size"
        ", int tokens_per_block"
        ", int attention_window_size"
        ", int sink_token_length"
        ", int beam_width"
        ", int quant_mode"
        ", float q_scaling"
        ", int q_lora_rank"
        ", int kv_lora_rank"
        ", int qk_nope_head_dim"
        ", int qk_rope_head_dim"
        ", int v_head_dim"
        ") -> ()");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("mla_rope_generation", &tensorrt_llm::torch_ext::MLARopeGeneration);
}
