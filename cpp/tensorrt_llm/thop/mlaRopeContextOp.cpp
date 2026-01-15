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

/**
 * @file mlaRopeContextOp.cpp
 * @brief PyTorch bindings for MLA context-phase RoPE and FP8 quantization operations.
 *
 * This file exports the following functions to Python:
 * - torch.ops.trtllm.mla_rope_context: Applies RoPE to MLA Q/K and writes to KV cache
 * - torch.ops.trtllm.mla_context_fp8_quantize: Quantizes MLA QKV to FP8 for context attention
 */

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/quantization.h"
#include "tensorrt_llm/kernels/kvCacheUtils.h"
#include "tensorrt_llm/kernels/mlaKernels.h"
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

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{

namespace
{

/**
 * Creates a KVBlockArray object for managing KV cache
 */
tk::KVBlockArray createKVBlockArrayForMLA(int num_contexts, int max_blocks_per_sequence, int tokens_per_block,
    int head_size, int attention_window_size, int sink_token_length, int beam_width, tc::QuantMode kv_cache_quant_mode,
    torch::Dtype orig_dtype, torch::Tensor const& host_kv_cache_pool_pointers,
    torch::Tensor const& host_kv_cache_pool_mapping, torch::Tensor const& kv_cache_block_offsets, int layer_idx)
{
    auto const orig_elem_size = torch::elementSize(orig_dtype);
    auto const cache_elem_size = kv_cache_quant_mode.hasKvCacheQuant() ? sizeof(int8_t) : orig_elem_size;
    int const num_kv_heads = 1; // Always 1 for MLA
    auto const size_per_token = num_kv_heads * head_size * cache_elem_size;

    int const cyclic_attention_window_size = attention_window_size;
    int const max_cyclic_attention_window_size = attention_window_size;
    bool const can_use_one_more_block = beam_width > 1;

    auto const pool_index = host_kv_cache_pool_mapping.index({layer_idx, 0}).item<int32_t>();
    auto const layer_idx_in_cache_pool = host_kv_cache_pool_mapping.index({layer_idx, 1}).item<int32_t>();
    int32_t const seq_offset = 0;
    tk::KVBlockArray::DataType* block_offsets
        = static_cast<tk::KVBlockArray::DataType*>(kv_cache_block_offsets.index({pool_index, seq_offset}).data_ptr());

    auto const block_size = tokens_per_block * num_kv_heads * head_size;
    auto const bytes_per_block = block_size * cache_elem_size;
    int32_t const kv_factor = 1; // always 1 for MLA
    auto const intra_pool_offset = layer_idx_in_cache_pool * kv_factor * bytes_per_block;

    void* host_primary_pool_pointer = reinterpret_cast<void*>(
        reinterpret_cast<char*>(host_kv_cache_pool_pointers.index({pool_index, 0}).item<int64_t>())
        + intra_pool_offset);
    void* host_secondary_pool_pointer = reinterpret_cast<void*>(
        reinterpret_cast<char*>(host_kv_cache_pool_pointers.index({pool_index, 1}).item<int64_t>())
        + intra_pool_offset);

    return tk::KVBlockArray(num_contexts, max_blocks_per_sequence, tokens_per_block, size_per_token,
        cyclic_attention_window_size, max_cyclic_attention_window_size, sink_token_length, can_use_one_more_block,
        host_primary_pool_pointer, host_secondary_pool_pointer, block_offsets);
}

template <typename T, typename KVCacheBuffer>
void invokeMLARopeContextHelper(torch::Tensor& q, // [total_q_len, num_heads * (d_nope + d_rope)]
    torch::Tensor& q_pe,                          // [total_q_len, num_heads, d_rope]
    std::optional<torch::Tensor>& k,              // [total_kv_len, num_heads * (d_nope + d_rope)]
    torch::Tensor const& latent_cache,            // [total_kv_len, kv_lora_rank + d_rope]
    KVCacheBuffer& kv_cache_buffer, torch::Tensor const& cos_sin_cache, torch::Tensor const& cu_q_seqlens,
    torch::Tensor const& cache_seq_lens, int32_t batch_size, int32_t num_heads, int32_t max_input_seq_len,
    int32_t q_pe_ld, int32_t q_pe_stride, tk::MlaMetaParams const& mla_meta_params, tk::KvCacheDataType cache_type,
    float const* kv_scale_kv_ptr, int32_t const* helix_position_offsets_ptr, bool absorption_mode)
{
    auto stream = at::cuda::getCurrentCUDAStream(q.get_device());

    tk::MlaParams<T> mla_params{};
    mla_params.q_buf = static_cast<T*>(q.data_ptr());
    mla_params.q_pe = static_cast<T*>(q_pe.data_ptr());
    mla_params.k_buf = k.has_value() ? static_cast<T*>(k->data_ptr()) : nullptr;
    mla_params.latent_cache = static_cast<T const*>(latent_cache.data_ptr());
    mla_params.cos_sin_cache = static_cast<float2 const*>(cos_sin_cache.data_ptr());
    mla_params.batch_size = batch_size;
    mla_params.head_num = num_heads;
    mla_params.max_input_seq_len = max_input_seq_len;
    mla_params.q_pe_ld = q_pe_ld;
    mla_params.q_pe_stride = q_pe_stride;
    mla_params.meta = mla_meta_params;
    mla_params.cu_q_seqlens = const_cast<int*>(cu_q_seqlens.data_ptr<int>());
    mla_params.cache_seq_lens = cache_seq_lens.data_ptr<int32_t>();
    mla_params.cache_type = cache_type;
    mla_params.quant_scale_kv = kv_scale_kv_ptr;
    mla_params.helix_position_offsets = helix_position_offsets_ptr;
    mla_params.absorption_mode = absorption_mode;

    tk::invokeMLARopeContext<T>(mla_params, kv_cache_buffer, stream);
}

template <typename T>
void invokeMLAContextFp8QuantizeHelper(torch::Tensor& q, // Input Q buffer (modified in place)
    std::optional<torch::Tensor>& k,                     // Input K buffer
    std::optional<torch::Tensor>& v,                     // Input V buffer
    torch::Tensor& quant_q,                              // Output quantized Q
    std::optional<torch::Tensor>& quant_k,               // Output quantized K
    std::optional<torch::Tensor>& quant_v,               // Output quantized V
    torch::Tensor const& cu_q_seqlens, int32_t batch_size, int32_t num_heads, int32_t max_input_seq_len,
    int32_t total_kv_len, tk::MlaMetaParams const& mla_meta_params, float const* quant_scale_qkv_ptr,
    bool absorption_mode)
{
    auto stream = at::cuda::getCurrentCUDAStream(q.get_device());

    tk::MlaParams<T> mla_params{};
    mla_params.q_buf = static_cast<T*>(q.data_ptr());
    mla_params.k_buf = k.has_value() ? static_cast<T*>(k->data_ptr()) : nullptr;
    mla_params.v_buf = v.has_value() ? static_cast<T const*>(v->data_ptr()) : nullptr;
    mla_params.quant_q_buf = quant_q.data_ptr();
    mla_params.quant_k_buf = quant_k.has_value() ? quant_k->data_ptr() : nullptr;
    mla_params.quant_v_buf = quant_v.has_value() ? quant_v->data_ptr() : nullptr;
    mla_params.batch_size = batch_size;
    mla_params.head_num = num_heads;
    mla_params.max_input_seq_len = max_input_seq_len;
    mla_params.cu_q_seqlens = const_cast<int*>(cu_q_seqlens.data_ptr<int>());
    mla_params.meta = mla_meta_params;
    mla_params.quant_scale_qkv = quant_scale_qkv_ptr;
    mla_params.cache_type = tk::KvCacheDataType::FP8;
    mla_params.absorption_mode = absorption_mode;

    tk::invokeMLAContextFp8Quantize<T>(mla_params, total_kv_len, stream);
}

} // namespace

/**
 * @brief Apply RoPE to MLA Q/K tensors and write K to KV cache (context phase).
 *
 * This function performs:
 * 1. Applies rotary position embedding (RoPE) to Q and K tensors
 * 2. Writes the processed K (latent cache) to the paged KV cache
 *
 * @param q Query tensor [total_q_len, num_heads * (qk_nope_head_dim + qk_rope_head_dim)]
 * @param q_pe Query position embedding tensor [total_q_len, num_heads, qk_rope_head_dim]
 * @param k Optional key tensor [total_kv_len, num_heads * (qk_nope_head_dim + qk_rope_head_dim)]
 * @param latent_cache Latent cache tensor (compressed KV) [total_kv_len, kv_lora_rank + qk_rope_head_dim]
 * @param cos_sin_cache RoPE cos/sin cache [max_seq_len, qk_rope_head_dim]
 * @param cu_q_seqlens Cumulative query sequence lengths [batch_size + 1]
 * @param cache_seq_lens Cache sequence lengths per batch [batch_size]
 * @param kv_cache_block_offsets KV cache block offsets tensor
 * @param host_kv_cache_pool_pointers Host KV cache pool pointers tensor
 * @param host_kv_cache_pool_mapping Host KV cache pool mapping tensor
 * @param kv_scale_orig_quant Optional KV scale for quantization
 * @param helix_position_offsets Optional Helix position offsets for parallelism
 * @param batch_size Number of sequences in batch
 * @param num_heads Number of attention heads
 * @param max_input_seq_len Maximum input sequence length
 * @param layer_idx Layer index
 * @param tokens_per_block Tokens per KV cache block
 * @param attention_window_size Attention window size
 * @param sink_token_length Sink token length
 * @param beam_width Beam width
 * @param quant_mode Quantization mode
 * @param q_lora_rank Q LoRA rank (MLA parameter)
 * @param kv_lora_rank KV LoRA rank (MLA parameter)
 * @param qk_nope_head_dim QK nope head dimension (MLA parameter)
 * @param qk_rope_head_dim QK rope head dimension (MLA parameter)
 * @param v_head_dim V head dimension (MLA parameter)
 * @param absorption_mode Whether to use sparse MLA absorption mode
 */
void MLARopeContext(torch::Tensor& q, torch::Tensor& q_pe, std::optional<torch::Tensor> k,
    torch::Tensor const& latent_cache, torch::Tensor const& cos_sin_cache, torch::Tensor const& cu_q_seqlens,
    torch::Tensor const& cache_seq_lens, torch::Tensor const& kv_cache_block_offsets,
    torch::Tensor const& host_kv_cache_pool_pointers, torch::Tensor const& host_kv_cache_pool_mapping,
    std::optional<torch::Tensor> kv_scale_orig_quant, std::optional<torch::Tensor> helix_position_offsets,
    int64_t const batch_size, int64_t const num_heads, int64_t const max_input_seq_len, int64_t const layer_idx,
    int64_t const tokens_per_block, int64_t const attention_window_size, int64_t const sink_token_length,
    int64_t const beam_width, int64_t const quant_mode, int64_t const q_lora_rank, int64_t const kv_lora_rank,
    int64_t const qk_nope_head_dim, int64_t const qk_rope_head_dim, int64_t const v_head_dim,
    bool const absorption_mode)
{
    auto const input_dtype = q.scalar_type();
    TORCH_CHECK(input_dtype == torch::kFloat16 || input_dtype == torch::kFloat32 || input_dtype == torch::kBFloat16,
        "Input dtype must be float16, float32, or bfloat16");
    CHECK_TH_CUDA(q);
    CHECK_CONTIGUOUS(q);
    CHECK_TH_CUDA(q_pe);
    CHECK_TH_CUDA(latent_cache);
    CHECK_CONTIGUOUS(latent_cache);
    CHECK_TH_CUDA(cos_sin_cache);
    CHECK_TH_CUDA(cu_q_seqlens);

    auto const kv_cache_quant_mode = tc::QuantMode(static_cast<uint32_t>(quant_mode));
    int const head_size = kv_lora_rank + qk_rope_head_dim;
    int const max_blocks_per_sequence = kv_cache_block_offsets.size(-1);

    tk::KVBlockArray kv_cache_buffer = createKVBlockArrayForMLA(batch_size, max_blocks_per_sequence, tokens_per_block,
        head_size, attention_window_size, sink_token_length, beam_width, kv_cache_quant_mode, input_dtype,
        host_kv_cache_pool_pointers, host_kv_cache_pool_mapping, kv_cache_block_offsets, layer_idx);

    tk::MlaMetaParams mla_meta_params = {static_cast<int>(q_lora_rank), static_cast<int>(kv_lora_rank),
        static_cast<int>(qk_nope_head_dim), static_cast<int>(qk_rope_head_dim), static_cast<int>(v_head_dim), 1,
        static_cast<int>(host_kv_cache_pool_mapping.size(0))};

    float const* kv_scale_kv_ptr = kv_scale_orig_quant.has_value() ? kv_scale_orig_quant->data_ptr<float>() : nullptr;
    int32_t const* helix_position_offsets_ptr
        = helix_position_offsets.has_value() ? helix_position_offsets->data_ptr<int32_t>() : nullptr;

    tk::KvCacheDataType cache_type
        = kv_cache_quant_mode.hasFp8KvCache() ? tk::KvCacheDataType::FP8 : tk::KvCacheDataType::BASE;

    TORCH_CHECK(q_pe.dim() == 3, "q_pe must be 3D tensor");
    TORCH_CHECK(q_pe.strides()[2] == 1, "q_pe must be contiguous in last dimension");
    int32_t const q_pe_ld = q_pe.strides()[1];
    int32_t const q_pe_stride = q_pe.strides()[0];

    if (input_dtype == torch::kFloat16)
    {
        invokeMLARopeContextHelper<half>(q, q_pe, k, latent_cache, kv_cache_buffer, cos_sin_cache, cu_q_seqlens,
            cache_seq_lens, batch_size, num_heads, max_input_seq_len, q_pe_ld, q_pe_stride, mla_meta_params, cache_type,
            kv_scale_kv_ptr, helix_position_offsets_ptr, absorption_mode);
    }
    else if (input_dtype == torch::kBFloat16)
    {
        invokeMLARopeContextHelper<__nv_bfloat16>(q, q_pe, k, latent_cache, kv_cache_buffer, cos_sin_cache,
            cu_q_seqlens, cache_seq_lens, batch_size, num_heads, max_input_seq_len, q_pe_ld, q_pe_stride,
            mla_meta_params, cache_type, kv_scale_kv_ptr, helix_position_offsets_ptr, absorption_mode);
    }
    else if (input_dtype == torch::kFloat32)
    {
        invokeMLARopeContextHelper<float>(q, q_pe, k, latent_cache, kv_cache_buffer, cos_sin_cache, cu_q_seqlens,
            cache_seq_lens, batch_size, num_heads, max_input_seq_len, q_pe_ld, q_pe_stride, mla_meta_params, cache_type,
            kv_scale_kv_ptr, helix_position_offsets_ptr, absorption_mode);
    }
}

/**
 * @brief Quantize MLA Q/K/V tensors to FP8 for context phase attention.
 *
 * @param q Query tensor (input)
 * @param k Optional key tensor (input)
 * @param v Optional value tensor (input)
 * @param quant_q Output quantized Q tensor
 * @param quant_k Output quantized K tensor (optional)
 * @param quant_v Output quantized V tensor (optional)
 * @param cu_q_seqlens Cumulative query sequence lengths
 * @param quant_scale_qkv Quantization scale for QKV
 * @param batch_size Batch size
 * @param num_heads Number of attention heads
 * @param max_input_seq_len Maximum input sequence length
 * @param total_kv_len Total KV length
 * @param q_lora_rank Q LoRA rank
 * @param kv_lora_rank KV LoRA rank
 * @param qk_nope_head_dim QK nope head dimension
 * @param qk_rope_head_dim QK rope head dimension
 * @param v_head_dim V head dimension
 * @param absorption_mode Whether using absorption mode (sparse MLA)
 */
void MLAContextFp8Quantize(torch::Tensor& q, std::optional<torch::Tensor> k, std::optional<torch::Tensor> v,
    torch::Tensor& quant_q, std::optional<torch::Tensor> quant_k, std::optional<torch::Tensor> quant_v,
    torch::Tensor const& cu_q_seqlens, torch::Tensor const& quant_scale_qkv, int64_t const batch_size,
    int64_t const num_heads, int64_t const max_input_seq_len, int64_t const total_kv_len, int64_t const q_lora_rank,
    int64_t const kv_lora_rank, int64_t const qk_nope_head_dim, int64_t const qk_rope_head_dim,
    int64_t const v_head_dim, bool const absorption_mode)
{
    auto const input_dtype = q.scalar_type();
    TORCH_CHECK(input_dtype == torch::kFloat16 || input_dtype == torch::kFloat32 || input_dtype == torch::kBFloat16,
        "Input dtype must be float16, float32, or bfloat16");
    CHECK_TH_CUDA(q);
    CHECK_TH_CUDA(quant_q);
    CHECK_TH_CUDA(cu_q_seqlens);
    CHECK_TH_CUDA(quant_scale_qkv);

    tk::MlaMetaParams mla_meta_params = {static_cast<int>(q_lora_rank), static_cast<int>(kv_lora_rank),
        static_cast<int>(qk_nope_head_dim), static_cast<int>(qk_rope_head_dim), static_cast<int>(v_head_dim), 1, 0};

    float const* quant_scale_qkv_ptr = quant_scale_qkv.data_ptr<float>();

    if (input_dtype == torch::kFloat16)
    {
        invokeMLAContextFp8QuantizeHelper<half>(q, k, v, quant_q, quant_k, quant_v, cu_q_seqlens, batch_size, num_heads,
            max_input_seq_len, total_kv_len, mla_meta_params, quant_scale_qkv_ptr, absorption_mode);
    }
    else if (input_dtype == torch::kBFloat16)
    {
        invokeMLAContextFp8QuantizeHelper<__nv_bfloat16>(q, k, v, quant_q, quant_k, quant_v, cu_q_seqlens, batch_size,
            num_heads, max_input_seq_len, total_kv_len, mla_meta_params, quant_scale_qkv_ptr, absorption_mode);
    }
    else if (input_dtype == torch::kFloat32)
    {
        invokeMLAContextFp8QuantizeHelper<float>(q, k, v, quant_q, quant_k, quant_v, cu_q_seqlens, batch_size,
            num_heads, max_input_seq_len, total_kv_len, mla_meta_params, quant_scale_qkv_ptr, absorption_mode);
    }
}

} // namespace torch_ext

TRTLLM_NAMESPACE_END

// Register torch operations
TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "mla_rope_context("
        "Tensor(a!) q"
        ", Tensor(a!) q_pe"
        ", Tensor? k"
        ", Tensor latent_cache"
        ", Tensor cos_sin_cache"
        ", Tensor cu_q_seqlens"
        ", Tensor cache_seq_lens"
        ", Tensor kv_cache_block_offsets"
        ", Tensor host_kv_cache_pool_pointers"
        ", Tensor host_kv_cache_pool_mapping"
        ", Tensor? kv_scale_orig_quant"
        ", Tensor? helix_position_offsets"
        ", int batch_size"
        ", int num_heads"
        ", int max_input_seq_len"
        ", int layer_idx"
        ", int tokens_per_block"
        ", int attention_window_size"
        ", int sink_token_length"
        ", int beam_width"
        ", int quant_mode"
        ", int q_lora_rank"
        ", int kv_lora_rank"
        ", int qk_nope_head_dim"
        ", int qk_rope_head_dim"
        ", int v_head_dim"
        ", bool absorption_mode"
        ") -> ()");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("mla_rope_context", &tensorrt_llm::torch_ext::MLARopeContext);
}

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "mla_context_fp8_quantize("
        "Tensor(a!) q"
        ", Tensor? k"
        ", Tensor? v"
        ", Tensor(a!) quant_q"
        ", Tensor? quant_k"
        ", Tensor? quant_v"
        ", Tensor cu_q_seqlens"
        ", Tensor quant_scale_qkv"
        ", int batch_size"
        ", int num_heads"
        ", int max_input_seq_len"
        ", int total_kv_len"
        ", int q_lora_rank"
        ", int kv_lora_rank"
        ", int qk_nope_head_dim"
        ", int qk_rope_head_dim"
        ", int v_head_dim"
        ", bool absorption_mode"
        ") -> ()");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("mla_context_fp8_quantize", &tensorrt_llm::torch_ext::MLAContextFp8Quantize);
}
