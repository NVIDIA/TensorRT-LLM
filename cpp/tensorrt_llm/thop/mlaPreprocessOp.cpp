/*
 * Copyright (c) 2020-2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/quantization.h"
#include "tensorrt_llm/kernels/kvCacheUtils.h"
#include "tensorrt_llm/kernels/mlaChunkedPrefill.cuh"
#include "tensorrt_llm/kernels/mlaKernels.h"
#include "tensorrt_llm/thop/thUtils.h"
#include <cstdint>
#include <torch/extension.h>

namespace tk = tensorrt_llm::kernels;
namespace tc = tensorrt_llm::common;
using tk::KVBlockArray;

namespace torch_ext
{

namespace
{

template <typename T, typename TCache>
void loadPagedKVCacheForMLAHelper(torch::Tensor& compressed_kv, torch::Tensor& k_pe, KVBlockArray& kv_cache,
    int const num_contexts, torch::Tensor const& cu_ctx_cached_kv_lens, int const max_input_seq_len,
    int const lora_size, int const rope_size, float const* kv_scale_quant_orig_ptr)
{
    auto stream = at::cuda::getCurrentCUDAStream(compressed_kv.get_device());

    auto* compressed_kv_ptr = static_cast<T*>(compressed_kv.data_ptr());
    auto* k_pe_ptr = static_cast<T*>(k_pe.data_ptr());
    auto const* cu_ctx_cached_kv_lens_ptr = cu_ctx_cached_kv_lens.data_ptr<int64_t>();
    tensorrt_llm::kernels::invokeMLALoadPagedKV<T, TCache>(compressed_kv_ptr, k_pe_ptr, kv_cache, num_contexts,
        cu_ctx_cached_kv_lens_ptr, max_input_seq_len, lora_size, rope_size, kv_scale_quant_orig_ptr, stream);
}

template <typename T, typename TCache>
void loadChunkedKVCacheForMLAHelper(torch::Tensor& output_kv, torch::Tensor& output_k_pe, KVBlockArray& kv_cache,
    int const num_contexts, torch::Tensor const& cu_ctx_chunked_len, torch::Tensor const& chunked_ld_global_offset,
    int lora_size, int rope_size, int const max_seq_len, float const* kv_scale_quant_orig_ptr)
{
    auto stream = at::cuda::getCurrentCUDAStream(output_kv.get_device());

    T* output_kv_ptr = static_cast<T*>(output_kv.data_ptr());
    T* output_k_pe_ptr = static_cast<T*>(output_k_pe.data_ptr());
    tensorrt_llm::kernels::invokeMLALoadChunkedKV<T, TCache>(output_kv_ptr, output_k_pe_ptr, kv_cache, num_contexts,
        cu_ctx_chunked_len.data_ptr<int64_t>(), chunked_ld_global_offset.data_ptr<int64_t>(), lora_size, rope_size,
        max_seq_len, kv_scale_quant_orig_ptr, stream);
}

template <typename T, typename TCache>
void invokeMLARopeAppendPagedKVAssignQHelper(KVBlockArray& kv_cache, torch::Tensor& q, torch::Tensor& latent_cache,
    int const num_requests, torch::Tensor const& cu_ctx_cached_kv_lens, torch::Tensor const& cu_seq_lens,
    int const max_input_uncached_seq_len, torch::Tensor const& cos_sin_cache, int const head_num, int const nope_size,
    int const rope_size, int const lora_size, float const* kv_scale_orig_quant_ptr)
{
    auto stream = at::cuda::getCurrentCUDAStream(q.get_device());
    auto* q_ptr = static_cast<T*>(q.data_ptr());
    auto* latent_cache_ptr = static_cast<T*>(latent_cache.data_ptr());
    auto const* cu_ctx_cached_kv_lens_ptr = cu_ctx_cached_kv_lens.data_ptr<int64_t>();
    auto const* cu_seq_lens_ptr = cu_seq_lens.data_ptr<int64_t>();
    auto const* cos_sin_cache_ptr = static_cast<float2 const*>(cos_sin_cache.data_ptr());
    tensorrt_llm::kernels::invokeMLARopeAppendPagedKVAssignQ<T, TCache>(kv_cache, q_ptr, latent_cache_ptr, num_requests,
        cu_ctx_cached_kv_lens_ptr, cu_seq_lens_ptr, max_input_uncached_seq_len, cos_sin_cache_ptr, head_num, nope_size,
        rope_size, lora_size, kv_scale_orig_quant_ptr, stream);
}

template <typename T>
void mergeChunkedAttentionForMLAHelper(torch::Tensor& merged_attn, torch::Tensor const& temp_attn,
    torch::Tensor& merged_softmax_stats, torch::Tensor const& temp_softmax_stats, int64_t const num_requests,
    torch::Tensor const& cu_q_seq_lens, int64_t const max_q_seq_len, torch::Tensor const& merge_op,
    int64_t const num_heads, int64_t const head_size)
{
    auto stream = at::cuda::getCurrentCUDAStream(merged_attn.get_device());
    T* merged_attn_ptr = static_cast<T*>(merged_attn.data_ptr());
    T* temp_attn_ptr = static_cast<T*>(temp_attn.data_ptr());
    float* merged_softmax_stats_ptr = static_cast<float*>(merged_softmax_stats.data_ptr());
    float* temp_softmax_stats_ptr = static_cast<float*>(temp_softmax_stats.data_ptr());
    int64_t* const cu_q_seq_lens_ptr = cu_q_seq_lens.data_ptr<int64_t>();
    int64_t* const merge_op_ptr = merge_op.data_ptr<int64_t>();

    tensorrt_llm::kernels::invokeMergeAttnWithSoftmax(merged_attn_ptr, merged_softmax_stats_ptr, merged_attn_ptr,
        merged_softmax_stats_ptr, temp_attn_ptr, temp_softmax_stats_ptr, num_requests, cu_q_seq_lens_ptr, max_q_seq_len,
        merge_op_ptr, num_heads, head_size, stream);
}

/**
 * Creates a KVBlockArray object for managing KV cache
 *
 * @param num_contexts Number of contexts
 * @param max_blocks_per_sequence Maximum blocks per sequence
 * @param tokens_per_block Number of tokens per block
 * @param head_size Size of each head
 * @param num_kv_heads Number of KV heads (1 for MLA)
 * @param attention_window_size Attention window size
 * @param sink_token_length Sink token length
 * @param beam_width Beam width
 * @param kv_cache_quant_mode KV cache quantization mode
 * @param orig_dtype Original data type
 * @param host_kv_cache_pool_pointers Host KV cache pool pointers
 * @param host_kv_cache_pool_mapping Host KV cache pool mapping
 * @param kv_cache_block_offsets KV cache block offsets
 * @param layer_idx Layer index
 * @return Constructed KVBlockArray object
 */
KVBlockArray createKVBlockArray(int num_contexts, int max_blocks_per_sequence, int tokens_per_block, int head_size,
    int num_kv_heads, int attention_window_size, int sink_token_length, int beam_width,
    tc::QuantMode kv_cache_quant_mode, torch::Dtype orig_dtype, torch::Tensor const& host_kv_cache_pool_pointers,
    torch::Tensor const& host_kv_cache_pool_mapping, torch::Tensor const& kv_cache_block_offsets, int layer_idx)
{
    auto const orig_elem_size = torch::elementSize(orig_dtype);
    auto const cache_elem_size = kv_cache_quant_mode.hasKvCacheQuant() ? sizeof(int8_t) : orig_elem_size;
    auto const size_per_token = num_kv_heads * head_size * cache_elem_size;

    int const cyclic_attention_window_size = attention_window_size;
    int const max_cyclic_attention_window_size = attention_window_size;
    bool const can_use_one_more_block = beam_width > 1;

    auto const pool_index = host_kv_cache_pool_mapping.index({layer_idx, 0}).item<int32_t>();
    auto const layer_idx_in_cache_pool = host_kv_cache_pool_mapping.index({layer_idx, 1}).item<int32_t>();
    int32_t const seq_offset = 0;
    KVBlockArray::DataType* block_offsets
        = static_cast<KVBlockArray::DataType*>(kv_cache_block_offsets.index({pool_index, seq_offset}).data_ptr());

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

    return KVBlockArray(num_contexts, max_blocks_per_sequence, tokens_per_block, size_per_token,
        cyclic_attention_window_size, max_cyclic_attention_window_size, sink_token_length, can_use_one_more_block,
        host_primary_pool_pointer, host_secondary_pool_pointer, block_offsets);
}

} // namespace

std::vector<torch::Tensor> loadPagedKVCacheForMLA(torch::ScalarType out_dtype, int64_t const num_contexts,
    int64_t const num_ctx_cached_tokens, int64_t const max_ctx_cached_kv_len, torch::Tensor& cu_ctx_cached_kv_lens,
    torch::Tensor const& kv_cache_block_offsets, torch::Tensor const& host_kv_cache_block_offsets,
    torch::Tensor const& host_kv_cache_pool_pointers, torch::Tensor const& host_kv_cache_pool_mapping,
    torch::optional<torch::Tensor> kv_scale_orig_quant, torch::optional<torch::Tensor> kv_scale_quant_orig,
    int64_t const layer_idx, int64_t const lora_size, int64_t const rope_size, int64_t const tokens_per_block,
    int64_t const attention_window_size, int64_t const sink_token_length, int64_t const beam_width,
    int64_t const quant_mode)
{
    TORCH_CHECK(out_dtype == torch::kFloat16 || out_dtype == torch::kFloat32 || out_dtype == torch::kBFloat16,
        "out_dtype only support float16, float32, bfloat16");
    TLLM_CHECK(num_contexts > 0);
    TORCH_CHECK(num_ctx_cached_tokens > 0);
    TLLM_CHECK(max_ctx_cached_kv_len > 0);
    CHECK_INPUT(cu_ctx_cached_kv_lens, torch::kInt64);
    TORCH_CHECK(cu_ctx_cached_kv_lens.dim() == 1);
    TORCH_CHECK(cu_ctx_cached_kv_lens.size(0) >= num_contexts + 1);

    auto kv_cache_quant_mode = tc::QuantMode(static_cast<uint32_t>(quant_mode));
    int max_blocks_per_sequence = kv_cache_block_offsets.size(-1);
    int head_size = lora_size + rope_size;
    KVBlockArray kv_cache_buffer
        = createKVBlockArray(num_contexts, max_blocks_per_sequence, tokens_per_block, head_size,
            1, // num_kv_heads is always 1 for MLA
            attention_window_size, sink_token_length, beam_width, kv_cache_quant_mode, out_dtype,
            host_kv_cache_pool_pointers, host_kv_cache_pool_mapping, kv_cache_block_offsets, layer_idx);

    float const* kv_scale_orig_quant_ptr = nullptr;
    float const* kv_scale_quant_orig_ptr = nullptr;
    if (kv_cache_quant_mode.hasKvCacheQuant())
    {
        TLLM_CHECK_WITH_INFO(kv_cache_quant_mode.hasFp8KvCache(), "Only FP8 KV cache is supported for now");
        TORCH_CHECK(kv_scale_orig_quant.has_value());
        TORCH_CHECK(kv_scale_quant_orig.has_value());
        kv_scale_orig_quant_ptr = kv_scale_orig_quant.value().data_ptr<float>();
        kv_scale_quant_orig_ptr = kv_scale_quant_orig.value().data_ptr<float>();
        TLLM_CHECK(kv_scale_orig_quant_ptr != nullptr);
        TLLM_CHECK(kv_scale_quant_orig_ptr != nullptr);
    }

    std::vector<torch::Tensor> outputs;
    // compressed_kv {num_ctx_cached_tokens, lora_size}
    outputs.push_back(torch::empty(
        {num_ctx_cached_tokens, lora_size}, torch::dtype(out_dtype).device(torch::kCUDA).requires_grad(false)));
    // k_pe {num_ctx_cached_tokens, rope_size}
    outputs.push_back(torch::empty(
        {num_ctx_cached_tokens, rope_size}, torch::dtype(out_dtype).device(torch::kCUDA).requires_grad(false)));

    if (out_dtype == torch::kFloat16)
    {
        if (kv_cache_quant_mode.hasFp8KvCache())
        {
            loadPagedKVCacheForMLAHelper<half, __nv_fp8_e4m3>(outputs[0], outputs[1], kv_cache_buffer, num_contexts,
                cu_ctx_cached_kv_lens, max_ctx_cached_kv_len, lora_size, rope_size, kv_scale_quant_orig_ptr);
        }
        else
        {
            loadPagedKVCacheForMLAHelper<half, half>(outputs[0], outputs[1], kv_cache_buffer, num_contexts,
                cu_ctx_cached_kv_lens, max_ctx_cached_kv_len, lora_size, rope_size, kv_scale_quant_orig_ptr);
        }
    }
    else if (out_dtype == torch::kFloat32)
    {
        if (kv_cache_quant_mode.hasFp8KvCache())
        {
            loadPagedKVCacheForMLAHelper<float, __nv_fp8_e4m3>(outputs[0], outputs[1], kv_cache_buffer, num_contexts,
                cu_ctx_cached_kv_lens, max_ctx_cached_kv_len, lora_size, rope_size, kv_scale_quant_orig_ptr);
        }
        else
        {
            loadPagedKVCacheForMLAHelper<float, float>(outputs[0], outputs[1], kv_cache_buffer, num_contexts,
                cu_ctx_cached_kv_lens, max_ctx_cached_kv_len, lora_size, rope_size, kv_scale_quant_orig_ptr);
        }
    }
    else if (out_dtype == torch::kBFloat16)
    {
        if (kv_cache_quant_mode.hasFp8KvCache())
        {
            loadPagedKVCacheForMLAHelper<__nv_bfloat16, __nv_fp8_e4m3>(outputs[0], outputs[1], kv_cache_buffer,
                num_contexts, cu_ctx_cached_kv_lens, max_ctx_cached_kv_len, lora_size, rope_size,
                kv_scale_quant_orig_ptr);
        }
        else
        {
            loadPagedKVCacheForMLAHelper<__nv_bfloat16, __nv_bfloat16>(outputs[0], outputs[1], kv_cache_buffer,
                num_contexts, cu_ctx_cached_kv_lens, max_ctx_cached_kv_len, lora_size, rope_size,
                kv_scale_quant_orig_ptr);
        }
    }

    return outputs;
}

std::vector<torch::Tensor> loadChunkedKVCacheForMLA(torch::ScalarType out_dtype, int64_t const num_contexts,
    int64_t const num_ctx_cached_tokens, torch::Tensor const& cu_ctx_chunked_kv_lens,
    torch::Tensor const& chunked_ld_global_offset, torch::Tensor const& kv_cache_block_offsets,
    torch::Tensor const& host_kv_cache_pool_pointers, torch::Tensor const& host_kv_cache_pool_mapping,
    torch::optional<torch::Tensor> kv_scale_orig_quant, torch::optional<torch::Tensor> kv_scale_quant_orig,
    int64_t const layer_idx, int64_t const lora_size, int64_t const rope_size, int64_t const tokens_per_block,
    int64_t const max_seq_len, int64_t const attention_window_size, int64_t const sink_token_length,
    int64_t const beam_width, int64_t const quant_mode)
{
    TORCH_CHECK(out_dtype == torch::kFloat16 || out_dtype == torch::kFloat32 || out_dtype == torch::kBFloat16,
        "out_dtype only support float16, float32, bfloat16");
    TLLM_CHECK(num_contexts > 0);
    CHECK_INPUT(cu_ctx_chunked_kv_lens, torch::kInt64);
    TORCH_CHECK(cu_ctx_chunked_kv_lens.dim() == 1);
    TORCH_CHECK(cu_ctx_chunked_kv_lens.size(0) >= num_contexts + 1);
    int head_size = lora_size + rope_size;
    auto kv_cache_quant_mode = tc::QuantMode(static_cast<uint32_t>(quant_mode));
    int max_blocks_per_sequence = kv_cache_block_offsets.size(-1);
    KVBlockArray kv_cache_buffer
        = createKVBlockArray(num_contexts, max_blocks_per_sequence, tokens_per_block, head_size,
            1, // num_kv_heads is always 1 for MLA
            attention_window_size, sink_token_length, beam_width, kv_cache_quant_mode, out_dtype,
            host_kv_cache_pool_pointers, host_kv_cache_pool_mapping, kv_cache_block_offsets, layer_idx);

    float const* kv_scale_orig_quant_ptr = nullptr;
    float const* kv_scale_quant_orig_ptr = nullptr;
    if (kv_cache_quant_mode.hasKvCacheQuant())
    {
        TORCH_CHECK(kv_scale_orig_quant.has_value());
        TORCH_CHECK(kv_scale_quant_orig.has_value());
        kv_scale_orig_quant_ptr = kv_scale_orig_quant.value().data_ptr<float>();
        kv_scale_quant_orig_ptr = kv_scale_quant_orig.value().data_ptr<float>();
        TLLM_CHECK(kv_scale_orig_quant_ptr != nullptr);
        TLLM_CHECK(kv_scale_quant_orig_ptr != nullptr);
    }

    std::vector<torch::Tensor> outputs;

    // compressed_kv {num_ctx_cached_tokens, lora_size}
    outputs.push_back(torch::empty(
        {num_ctx_cached_tokens, lora_size}, torch::dtype(out_dtype).device(torch::kCUDA).requires_grad(false)));
    // k_pe {num_ctx_cached_tokens, rope_size}
    outputs.push_back(torch::empty(
        {num_ctx_cached_tokens, rope_size}, torch::dtype(out_dtype).device(torch::kCUDA).requires_grad(false)));

    if (out_dtype == torch::kFloat16)
    {
        if (kv_cache_quant_mode.hasFp8KvCache())
        {
            loadChunkedKVCacheForMLAHelper<half, __nv_fp8_e4m3>(outputs[0], outputs[1], kv_cache_buffer, num_contexts,
                cu_ctx_chunked_kv_lens, chunked_ld_global_offset, lora_size, rope_size, max_seq_len,
                kv_scale_quant_orig_ptr);
        }
        else
        {
            loadChunkedKVCacheForMLAHelper<half, half>(outputs[0], outputs[1], kv_cache_buffer, num_contexts,
                cu_ctx_chunked_kv_lens, chunked_ld_global_offset, lora_size, rope_size, max_seq_len,
                kv_scale_quant_orig_ptr);
        }
    }
    else if (out_dtype == torch::kFloat32)
    {
        if (kv_cache_quant_mode.hasFp8KvCache())
        {
            loadChunkedKVCacheForMLAHelper<float, __nv_fp8_e4m3>(outputs[0], outputs[1], kv_cache_buffer, num_contexts,
                cu_ctx_chunked_kv_lens, chunked_ld_global_offset, lora_size, rope_size, max_seq_len,
                kv_scale_quant_orig_ptr);
        }
        else
        {
            loadChunkedKVCacheForMLAHelper<float, float>(outputs[0], outputs[1], kv_cache_buffer, num_contexts,
                cu_ctx_chunked_kv_lens, chunked_ld_global_offset, lora_size, rope_size, max_seq_len,
                kv_scale_quant_orig_ptr);
        }
    }
    else if (out_dtype == torch::kBFloat16)
    {
        if (kv_cache_quant_mode.hasFp8KvCache())
        {
            loadChunkedKVCacheForMLAHelper<__nv_bfloat16, __nv_fp8_e4m3>(outputs[0], outputs[1], kv_cache_buffer,
                num_contexts, cu_ctx_chunked_kv_lens, chunked_ld_global_offset, lora_size, rope_size, max_seq_len,
                kv_scale_quant_orig_ptr);
        }
        else
        {
            loadChunkedKVCacheForMLAHelper<__nv_bfloat16, __nv_bfloat16>(outputs[0], outputs[1], kv_cache_buffer,
                num_contexts, cu_ctx_chunked_kv_lens, chunked_ld_global_offset, lora_size, rope_size, max_seq_len,
                kv_scale_quant_orig_ptr);
        }
    }

    return outputs;
}

void MLARopeAppendPagedKVAssignQ(torch::Tensor& q, torch::Tensor& latent_cache, int64_t const num_contexts,
    torch::Tensor const& cu_ctx_cached_kv_lens, torch::Tensor const& cu_seq_lens,
    int64_t const max_input_uncached_seq_len, torch::Tensor const& cos_sin_cache, int64_t const head_num,
    int64_t const nope_size, int64_t const rope_size, int64_t const lora_size,
    torch::Tensor const& kv_cache_block_offsets, torch::Tensor const& host_kv_cache_block_offsets,
    torch::Tensor const& host_kv_cache_pool_pointers, torch::Tensor const& host_kv_cache_pool_mapping,
    torch::optional<torch::Tensor> kv_scale_orig_quant, torch::optional<torch::Tensor> kv_scale_quant_orig,
    int64_t const layer_idx, int64_t const tokens_per_block, int64_t const attention_window_size,
    int64_t const sink_token_length, int64_t const beam_width, int64_t const quant_mode)
{
    auto input_dtype = q.scalar_type();
    TORCH_CHECK(input_dtype == torch::kFloat16 || input_dtype == torch::kFloat32 || input_dtype == torch::kBFloat16);
    TORCH_CHECK(q.numel() > 0);
    TORCH_CHECK(q.dim() == 2);
    CHECK_TH_CUDA(q);
    CHECK_CONTIGUOUS(q);
    CHECK_INPUT(latent_cache, input_dtype);
    TORCH_CHECK(latent_cache.dim() == 2);
    CHECK_INPUT(cu_seq_lens, torch::kInt64);
    TORCH_CHECK(cu_seq_lens.dim() == 1);
    TORCH_CHECK(cu_seq_lens.size(0) >= num_contexts + 1);
    CHECK_INPUT(cu_ctx_cached_kv_lens, torch::kInt64);
    TORCH_CHECK(cu_ctx_cached_kv_lens.dim() == 1);
    TORCH_CHECK(cu_ctx_cached_kv_lens.size(0) >= num_contexts + 1);
    TORCH_CHECK(max_input_uncached_seq_len > 0);

    auto kv_cache_quant_mode = tc::QuantMode(static_cast<uint32_t>(quant_mode));
    int max_blocks_per_sequence = kv_cache_block_offsets.size(-1);
    int head_size = lora_size + rope_size;
    KVBlockArray kv_cache_buffer
        = createKVBlockArray(num_contexts, max_blocks_per_sequence, tokens_per_block, head_size,
            1, // num_kv_heads is always 1 for MLA
            attention_window_size, sink_token_length, beam_width, kv_cache_quant_mode, input_dtype,
            host_kv_cache_pool_pointers, host_kv_cache_pool_mapping, kv_cache_block_offsets, layer_idx);

    float const* kv_scale_orig_quant_ptr = nullptr;
    float const* kv_scale_quant_orig_ptr = nullptr;
    if (kv_cache_quant_mode.hasKvCacheQuant())
    {
        TLLM_CHECK_WITH_INFO(kv_cache_quant_mode.hasFp8KvCache(), "Only FP8 KV cache is supported for now");
        TORCH_CHECK(kv_scale_orig_quant.has_value());
        TORCH_CHECK(kv_scale_quant_orig.has_value());
        kv_scale_orig_quant_ptr = kv_scale_orig_quant.value().data_ptr<float>();
        kv_scale_quant_orig_ptr = kv_scale_quant_orig.value().data_ptr<float>();
        TLLM_CHECK(kv_scale_orig_quant_ptr != nullptr);
        TLLM_CHECK(kv_scale_quant_orig_ptr != nullptr);
    }

    if (input_dtype == torch::kFloat16)
    {
        if (kv_cache_quant_mode.hasFp8KvCache())
        {
            invokeMLARopeAppendPagedKVAssignQHelper<half, __nv_fp8_e4m3>(kv_cache_buffer, q, latent_cache, num_contexts,
                cu_ctx_cached_kv_lens, cu_seq_lens, max_input_uncached_seq_len, cos_sin_cache, head_num, nope_size,
                rope_size, lora_size, kv_scale_orig_quant_ptr);
        }
        else
        {
            invokeMLARopeAppendPagedKVAssignQHelper<half, half>(kv_cache_buffer, q, latent_cache, num_contexts,
                cu_ctx_cached_kv_lens, cu_seq_lens, max_input_uncached_seq_len, cos_sin_cache, head_num, nope_size,
                rope_size, lora_size, kv_scale_orig_quant_ptr);
        }
    }
    else if (input_dtype == torch::kFloat32)
    {
        if (kv_cache_quant_mode.hasFp8KvCache())
        {
            invokeMLARopeAppendPagedKVAssignQHelper<float, __nv_fp8_e4m3>(kv_cache_buffer, q, latent_cache,
                num_contexts, cu_ctx_cached_kv_lens, cu_seq_lens, max_input_uncached_seq_len, cos_sin_cache, head_num,
                nope_size, rope_size, lora_size, kv_scale_orig_quant_ptr);
        }
        else
        {
            invokeMLARopeAppendPagedKVAssignQHelper<float, float>(kv_cache_buffer, q, latent_cache, num_contexts,
                cu_ctx_cached_kv_lens, cu_seq_lens, max_input_uncached_seq_len, cos_sin_cache, head_num, nope_size,
                rope_size, lora_size, kv_scale_orig_quant_ptr);
        }
    }
    else if (input_dtype == torch::kBFloat16)
    {
        if (kv_cache_quant_mode.hasFp8KvCache())
        {
            invokeMLARopeAppendPagedKVAssignQHelper<__nv_bfloat16, __nv_fp8_e4m3>(kv_cache_buffer, q, latent_cache,
                num_contexts, cu_ctx_cached_kv_lens, cu_seq_lens, max_input_uncached_seq_len, cos_sin_cache, head_num,
                nope_size, rope_size, lora_size, kv_scale_orig_quant_ptr);
        }
        else
        {
            invokeMLARopeAppendPagedKVAssignQHelper<__nv_bfloat16, __nv_bfloat16>(kv_cache_buffer, q, latent_cache,
                num_contexts, cu_ctx_cached_kv_lens, cu_seq_lens, max_input_uncached_seq_len, cos_sin_cache, head_num,
                nope_size, rope_size, lora_size, kv_scale_orig_quant_ptr);
        }
    }
}

void mergeChunkedAttentionForMLA(torch::Tensor& merged_attn, torch::Tensor const& temp_attn,
    torch::Tensor& merged_softmax_stats, torch::Tensor const& temp_softmax_stats, int64_t const num_requests,
    torch::Tensor const& cu_q_seq_lens, int64_t const max_q_seq_len, torch::Tensor const& merge_op,
    int64_t const num_heads, int64_t const head_size)
{
    TORCH_CHECK(merged_attn.numel() > 0);
    TORCH_CHECK(temp_attn.numel() > 0);
    TORCH_CHECK(merged_attn.scalar_type() == temp_attn.scalar_type());
    TORCH_CHECK(merged_attn.scalar_type() == torch::kFloat16 || merged_attn.scalar_type() == torch::kFloat32
        || merged_attn.scalar_type() == torch::kBFloat16);
    TORCH_CHECK(temp_softmax_stats.scalar_type() == merged_softmax_stats.scalar_type());
    TORCH_CHECK(merged_softmax_stats.scalar_type() == torch::kFloat32);

    if (merged_attn.scalar_type() == torch::kFloat16)
    {
        mergeChunkedAttentionForMLAHelper<half>(merged_attn, temp_attn, merged_softmax_stats, temp_softmax_stats,
            num_requests, cu_q_seq_lens, max_q_seq_len, merge_op, num_heads, head_size);
    }
    else if (merged_attn.scalar_type() == torch::kFloat32)
    {
        mergeChunkedAttentionForMLAHelper<float>(merged_attn, temp_attn, merged_softmax_stats, temp_softmax_stats,
            num_requests, cu_q_seq_lens, max_q_seq_len, merge_op, num_heads, head_size);
    }
    else if (merged_attn.scalar_type() == torch::kBFloat16)
    {
        mergeChunkedAttentionForMLAHelper<__nv_bfloat16>(merged_attn, temp_attn, merged_softmax_stats,
            temp_softmax_stats, num_requests, cu_q_seq_lens, max_q_seq_len, merge_op, num_heads, head_size);
    }
}
} // namespace torch_ext

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "load_paged_kv_cache_for_mla("
        "ScalarType out_dtype"
        ", int num_contexts"
        ", int num_ctx_cached_tokens"
        ", int max_ctx_cached_kv_len"
        ", Tensor cu_ctx_cached_kv_lens"
        ", Tensor kv_cache_block_offsets"
        ", Tensor host_kv_cache_block_offsets"
        ", Tensor host_kv_cache_pool_pointers"
        ", Tensor host_kv_cache_pool_mapping"
        ", Tensor? kv_scale_orig_quant"
        ", Tensor? kv_scale_quant_orig"
        ", int layer_idx"
        ", int lora_size"
        ", int rope_size"
        ", int tokens_per_block"
        ", int attention_window_size"
        ", int sink_token_length"
        ", int beam_width"
        ", int quant_mode"
        ") -> Tensor[]");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("load_paged_kv_cache_for_mla", &torch_ext::loadPagedKVCacheForMLA);
}

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "load_chunked_kv_cache_for_mla("
        "ScalarType out_dtype"
        ", int num_contexts"
        ", int num_ctx_cached_tokens"
        ", Tensor cu_ctx_chunked_kv_lens"
        ", Tensor chunked_ld_global_offset"
        ", Tensor kv_cache_block_offsets"
        ", Tensor host_kv_cache_pool_pointers"
        ", Tensor host_kv_cache_pool_mapping"
        ", Tensor? kv_scale_orig_quant"
        ", Tensor? kv_scale_quant_orig"
        ", int layer_idx"
        ", int lora_size"
        ", int rope_size"
        ", int tokens_per_block"
        ", int max_seq_len"
        ", int attention_window_size"
        ", int sink_token_length"
        ", int beam_width"
        ", int quant_mode"
        ") -> Tensor[]");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("load_chunked_kv_cache_for_mla", &torch_ext::loadChunkedKVCacheForMLA);
}

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "mla_rope_append_paged_kv_assign_q("
        "Tensor q"
        ", Tensor latent_cache"
        ", int num_contexts"
        ", Tensor cu_ctx_cached_kv_lens"
        ", Tensor cu_seq_lens"
        ", int max_input_uncached_seq_len"
        ", Tensor cos_sin_cache"
        ", int head_num"
        ", int nope_size"
        ", int rope_size"
        ", int lora_size"
        ", Tensor kv_cache_block_offsets"
        ", Tensor host_kv_cache_block_offsets"
        ", Tensor host_kv_cache_pool_pointers"
        ", Tensor host_kv_cache_pool_mapping"
        ", Tensor? kv_scale_orig_quant"
        ", Tensor? kv_scale_quant_orig"
        ", int layer_idx"
        ", int tokens_per_block"
        ", int attention_window_size"
        ", int sink_token_length"
        ", int beam_width"
        ", int quant_mode"
        ") -> ()");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("mla_rope_append_paged_kv_assign_q", &torch_ext::MLARopeAppendPagedKVAssignQ);
}

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "merge_chunked_attention_for_mla("
        "Tensor(a!) merged_attn"
        ", Tensor temp_attn"
        ", Tensor merged_softmax_stats"
        ", Tensor temp_softmax_stats"
        ", int num_requests"
        ", Tensor cu_q_seq_lens"
        ", int max_q_seq_len"
        ", Tensor merge_op"
        ", int num_heads"
        ", int head_size"
        ") -> ()");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("merge_chunked_attention_for_mla", &torch_ext::mergeChunkedAttentionForMLA);
}
