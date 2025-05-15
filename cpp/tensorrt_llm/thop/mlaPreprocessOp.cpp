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
#include "tensorrt_llm/common/quantization.h"
#include "tensorrt_llm/kernels/kvCacheUtils.h"
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

template <typename T>
void loadPagedKVCacheForMLAHelper(torch::Tensor& output, KVBlockArray& kv_cache, int const num_contexts,
    torch::Tensor const& cu_ctx_cached_kv_lens, int const max_input_seq_len, int head_dim)
{
    auto stream = at::cuda::getCurrentCUDAStream(output.get_device());

    T* output_ptr = static_cast<T*>(output.data_ptr());
    tensorrt_llm::kernels::invokeMLALoadPagedKV<T>(output_ptr, kv_cache, num_contexts,
        cu_ctx_cached_kv_lens.data_ptr<int64_t>(), max_input_seq_len, head_dim, stream);
}

template <typename T>
void setPagedKVCacheForMLAHelper(torch::Tensor& output, torch::Tensor const& k, torch::Tensor const& v,
    torch::Tensor const& k_pe, int const num_requests, torch::Tensor const& cu_seq_lens, int const max_input_seq_len,
    int num_heads, int kv_dim, int rope_dim, int kv_cache_tokens_per_block)
{
    auto stream = at::cuda::getCurrentCUDAStream(output.get_device());
    T* output_ptr = static_cast<T*>(output.data_ptr());
    T* k_ptr = static_cast<T*>(k.data_ptr());
    T* v_ptr = static_cast<T*>(v.data_ptr());
    T* k_pe_ptr = static_cast<T*>(k_pe.data_ptr());
    auto* cu_seq_lens_ptr = cu_seq_lens.data_ptr<int64_t>();

    tensorrt_llm::kernels::invokeMLASetPagedKV<T>(output_ptr, k_ptr, v_ptr, k_pe_ptr, num_requests, cu_seq_lens_ptr,
        max_input_seq_len, num_heads, kv_dim, rope_dim, kv_cache_tokens_per_block, stream);
}

template <typename T>
void setPagedKVCacheV2ForMLAHelper(torch::Tensor& output, torch::Tensor const& cached_k, torch::Tensor const& cached_v,
    torch::Tensor const& cached_k_pe, torch::Tensor const& new_k, torch::Tensor const& new_v,
    torch::Tensor const& new_k_pe, int const num_requests, torch::Tensor const& cu_ctx_cached_kv_lens,
    torch::Tensor const& cu_seq_lens, int const max_input_seq_len, int num_heads, int kv_dim, int rope_dim,
    int kv_cache_tokens_per_block)
{
    auto stream = at::cuda::getCurrentCUDAStream(output.get_device());
    T* output_ptr = static_cast<T*>(output.data_ptr());
    T* cached_k_ptr = static_cast<T*>(cached_k.data_ptr());
    T* cached_v_ptr = static_cast<T*>(cached_v.data_ptr());
    T* cached_k_pe_ptr = static_cast<T*>(cached_k_pe.data_ptr());
    T* new_k_ptr = static_cast<T*>(new_k.data_ptr());
    T* new_v_ptr = static_cast<T*>(new_v.data_ptr());
    T* new_k_pe_ptr = static_cast<T*>(new_k_pe.data_ptr());
    auto* cu_ctx_cached_kv_lens_ptr = cu_ctx_cached_kv_lens.data_ptr<int64_t>();
    auto* cu_seq_lens_ptr = cu_seq_lens.data_ptr<int64_t>();

    tensorrt_llm::kernels::invokeMLASetPagedKVV2<T>(output_ptr, cached_k_ptr, cached_v_ptr, cached_k_pe_ptr, new_k_ptr,
        new_v_ptr, new_k_pe_ptr, num_requests, cu_ctx_cached_kv_lens_ptr, cu_seq_lens_ptr, max_input_seq_len, num_heads,
        kv_dim, rope_dim, kv_cache_tokens_per_block, stream);
}

template <typename T>
void appendPagedKVCacheForMLAHelper(KVBlockArray& kv_cache, torch::Tensor const& compressed_kv,
    torch::Tensor const& k_pe, int const num_requests, torch::Tensor const& cu_ctx_cached_kv_lens,
    torch::Tensor const& cu_seq_lens, int const max_input_uncached_seq_len, int head_dim)
{
    auto stream = at::cuda::getCurrentCUDAStream(compressed_kv.get_device());
    auto* const compressed_kv_ptr = static_cast<T* const>(compressed_kv.data_ptr());
    auto* const k_pe_ptr = static_cast<T* const>(k_pe.data_ptr());
    auto* const cu_seq_lens_ptr = cu_seq_lens.data_ptr<int64_t>();
    auto* const cu_ctx_cached_kv_lens_ptr = cu_ctx_cached_kv_lens.data_ptr<int64_t>();
    tensorrt_llm::kernels::invokeMLAAppendPagedKV(kv_cache, compressed_kv_ptr, k_pe_ptr, num_requests,
        cu_ctx_cached_kv_lens_ptr, cu_seq_lens_ptr, max_input_uncached_seq_len, head_dim, stream);
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
 * @param out_dtype Output data type
 * @param host_kv_cache_pool_pointers Host KV cache pool pointers
 * @param host_kv_cache_pool_mapping Host KV cache pool mapping
 * @param kv_cache_block_offsets KV cache block offsets
 * @param layer_idx Layer index
 * @return Constructed KVBlockArray object
 */
KVBlockArray createKVBlockArray(int num_contexts, int max_blocks_per_sequence, int tokens_per_block, int head_size,
    int num_kv_heads, int attention_window_size, int sink_token_length, int beam_width,
    tc::QuantMode kv_cache_quant_mode, torch::Dtype out_dtype, torch::Tensor const& host_kv_cache_pool_pointers,
    torch::Tensor const& host_kv_cache_pool_mapping, torch::Tensor const& kv_cache_block_offsets, int layer_idx)
{
    auto const output_elem_size = torch::elementSize(out_dtype);
    auto const cache_elem_size = kv_cache_quant_mode.hasKvCacheQuant() ? sizeof(int8_t) : output_elem_size;
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

torch::Tensor loadPagedKVCacheForMLA(torch::ScalarType out_dtype, int64_t const num_contexts,
    int64_t const max_ctx_cached_kv_len, torch::Tensor& cu_ctx_cached_kv_lens,
    torch::Tensor const& kv_cache_block_offsets, torch::Tensor const& host_kv_cache_block_offsets,
    torch::Tensor const& host_kv_cache_pool_pointers, torch::Tensor const& host_kv_cache_pool_mapping,
    torch::optional<torch::Tensor> kv_scale_orig_quant, torch::optional<torch::Tensor> kv_scale_quant_orig,
    int64_t const layer_idx, int64_t const head_size, int64_t const tokens_per_block,
    int64_t const attention_window_size, int64_t const sink_token_length, int64_t const beam_width,
    int64_t const quant_mode)
{
    TORCH_CHECK(out_dtype == torch::kFloat16 || out_dtype == torch::kFloat32 || out_dtype == torch::kBFloat16,
        "out_dtype only support float16, float32, bfloat16");
    TLLM_CHECK(num_contexts > 0);
    TLLM_CHECK(max_ctx_cached_kv_len > 0);
    CHECK_INPUT(cu_ctx_cached_kv_lens, torch::kInt64);
    TORCH_CHECK(cu_ctx_cached_kv_lens.dim() == 1);
    TORCH_CHECK(cu_ctx_cached_kv_lens.size(0) >= num_contexts + 1);

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

    auto const num_ctx_cached_tokens = cu_ctx_cached_kv_lens.index({num_contexts}).item<int64_t>();
    TORCH_CHECK(num_ctx_cached_tokens > 0);
    auto output = torch::empty(
        {num_ctx_cached_tokens, 1 * head_size}, torch::dtype(out_dtype).device(torch::kCUDA).requires_grad(false));

    if (out_dtype == torch::kFloat16)
    {
        loadPagedKVCacheForMLAHelper<half>(
            output, kv_cache_buffer, num_contexts, cu_ctx_cached_kv_lens, max_ctx_cached_kv_len, head_size);
    }
    else if (out_dtype == torch::kFloat32)
    {
        loadPagedKVCacheForMLAHelper<float>(
            output, kv_cache_buffer, num_contexts, cu_ctx_cached_kv_lens, max_ctx_cached_kv_len, head_size);
    }
    else if (out_dtype == torch::kBFloat16)
    {
        loadPagedKVCacheForMLAHelper<__nv_bfloat16>(
            output, kv_cache_buffer, num_contexts, cu_ctx_cached_kv_lens, max_ctx_cached_kv_len, head_size);
    }

    return output;
}

torch::Tensor setPagedKVCacheForMLA(torch::Tensor& output, torch::Tensor const& k, torch::Tensor const& v,
    torch::Tensor const& k_pe, int64_t const num_requests, torch::Tensor const& cu_seq_lens,
    int64_t const max_input_seq_len, int64_t const num_heads, int64_t const kv_dim, int64_t const rope_dim,
    int64_t const kv_cache_tokens_per_block)
{
    TORCH_CHECK(output.numel() > 0);
    TORCH_CHECK(output.scalar_type() == torch::kFloat16 || output.scalar_type() == torch::kFloat32
        || output.scalar_type() == torch::kBFloat16);
    CHECK_TH_CUDA(output);
    CHECK_CONTIGUOUS(output);
    CHECK_INPUT(k, output.scalar_type());
    CHECK_INPUT(v, output.scalar_type());
    CHECK_INPUT(k_pe, output.scalar_type());
    CHECK_INPUT(cu_seq_lens, torch::kInt64);
    TORCH_CHECK(cu_seq_lens.dim() == 1);
    TORCH_CHECK(cu_seq_lens.size(0) >= num_requests + 1);

    if (output.scalar_type() == torch::kFloat16)
    {
        setPagedKVCacheForMLAHelper<half>(output, k, v, k_pe, num_requests, cu_seq_lens, max_input_seq_len, num_heads,
            kv_dim, rope_dim, kv_cache_tokens_per_block);
    }
    else if (output.scalar_type() == torch::kFloat32)
    {
        setPagedKVCacheForMLAHelper<float>(output, k, v, k_pe, num_requests, cu_seq_lens, max_input_seq_len, num_heads,
            kv_dim, rope_dim, kv_cache_tokens_per_block);
    }
    else if (output.scalar_type() == torch::kBFloat16)
    {
        setPagedKVCacheForMLAHelper<__nv_bfloat16>(output, k, v, k_pe, num_requests, cu_seq_lens, max_input_seq_len,
            num_heads, kv_dim, rope_dim, kv_cache_tokens_per_block);
    }

    int64_t max_block_num = (max_input_seq_len + kv_cache_tokens_per_block - 1) / kv_cache_tokens_per_block;

    torch::Tensor faked_kv_cache_block_offsets = torch::arange(
        0, num_requests * 2 * max_block_num, torch::TensorOptions().dtype(torch::kInt32).device(output.device()));

    faked_kv_cache_block_offsets = faked_kv_cache_block_offsets.view({num_requests, 2, max_block_num});

    return faked_kv_cache_block_offsets;
}

torch::Tensor setPagedKVCacheV2ForMLA(torch::Tensor& output, torch::Tensor const& cached_k,
    torch::Tensor const& cached_v, torch::Tensor const& cached_k_pe, torch::Tensor const& new_k,
    torch::Tensor const& new_v, torch::Tensor const& new_k_pe, int64_t const num_requests,
    torch::Tensor const& cu_ctx_cached_kv_lens, torch::Tensor const& cu_seq_lens, int64_t const max_input_seq_len,
    int64_t const num_heads, int64_t const kv_dim, int64_t const rope_dim, int64_t const kv_cache_tokens_per_block)
{
    TORCH_CHECK(output.numel() > 0);
    TORCH_CHECK(output.scalar_type() == torch::kFloat16 || output.scalar_type() == torch::kFloat32
        || output.scalar_type() == torch::kBFloat16);
    CHECK_TH_CUDA(output);
    CHECK_CONTIGUOUS(output);
    CHECK_INPUT(cached_k, output.scalar_type());
    CHECK_INPUT(cached_v, output.scalar_type());
    CHECK_INPUT(cached_k_pe, output.scalar_type());
    TORCH_CHECK(cached_k_pe.dim() == 2);
    CHECK_INPUT(new_k, output.scalar_type());
    CHECK_INPUT(new_v, output.scalar_type());
    CHECK_INPUT(new_k_pe, output.scalar_type());
    TORCH_CHECK(new_k_pe.dim() == 2);
    CHECK_INPUT(cu_ctx_cached_kv_lens, torch::kInt64);
    TORCH_CHECK(cu_ctx_cached_kv_lens.dim() == 1);
    TORCH_CHECK(cu_ctx_cached_kv_lens.size(0) >= num_requests + 1);
    CHECK_INPUT(cu_seq_lens, torch::kInt64);
    TORCH_CHECK(cu_seq_lens.dim() == 1);
    TORCH_CHECK(cu_seq_lens.size(0) >= num_requests + 1);

    if (output.scalar_type() == torch::kFloat16)
    {
        setPagedKVCacheV2ForMLAHelper<half>(output, cached_k, cached_v, cached_k_pe, new_k, new_v, new_k_pe,
            num_requests, cu_ctx_cached_kv_lens, cu_seq_lens, max_input_seq_len, num_heads, kv_dim, rope_dim,
            kv_cache_tokens_per_block);
    }
    else if (output.scalar_type() == torch::kFloat32)
    {
        setPagedKVCacheV2ForMLAHelper<float>(output, cached_k, cached_v, cached_k_pe, new_k, new_v, new_k_pe,
            num_requests, cu_ctx_cached_kv_lens, cu_seq_lens, max_input_seq_len, num_heads, kv_dim, rope_dim,
            kv_cache_tokens_per_block);
    }
    else if (output.scalar_type() == torch::kBFloat16)
    {
        setPagedKVCacheV2ForMLAHelper<__nv_bfloat16>(output, cached_k, cached_v, cached_k_pe, new_k, new_v, new_k_pe,
            num_requests, cu_ctx_cached_kv_lens, cu_seq_lens, max_input_seq_len, num_heads, kv_dim, rope_dim,
            kv_cache_tokens_per_block);
    }

    int64_t max_block_num = (max_input_seq_len + kv_cache_tokens_per_block - 1) / kv_cache_tokens_per_block;

    torch::Tensor faked_kv_cache_block_offsets = torch::arange(
        0, num_requests * 2 * max_block_num, torch::TensorOptions().dtype(torch::kInt32).device(output.device()));

    faked_kv_cache_block_offsets = faked_kv_cache_block_offsets.view({num_requests, 2, max_block_num});

    return faked_kv_cache_block_offsets;
}

void appendPagedKVCacheForMLA(torch::Tensor const& compressed_kv, torch::Tensor const& k_pe, int64_t const num_contexts,
    torch::Tensor const& cu_ctx_cached_kv_lens, torch::Tensor const& cu_seq_lens,
    int64_t const max_input_uncached_seq_len, torch::Tensor const& kv_cache_block_offsets,
    torch::Tensor const& host_kv_cache_block_offsets, torch::Tensor const& host_kv_cache_pool_pointers,
    torch::Tensor const& host_kv_cache_pool_mapping, torch::optional<torch::Tensor> kv_scale_orig_quant,
    torch::optional<torch::Tensor> kv_scale_quant_orig, int64_t const layer_idx, int64_t const head_size,
    int64_t const tokens_per_block, int64_t const attention_window_size, int64_t const sink_token_length,
    int64_t const beam_width, int64_t const quant_mode)
{
    TORCH_CHECK(compressed_kv.numel() > 0);
    TORCH_CHECK(compressed_kv.scalar_type() == torch::kFloat16 || compressed_kv.scalar_type() == torch::kFloat32
        || compressed_kv.scalar_type() == torch::kBFloat16);
    TORCH_CHECK(compressed_kv.dim() == 2);
    CHECK_TH_CUDA(compressed_kv);
    CHECK_CONTIGUOUS(compressed_kv);
    CHECK_INPUT(k_pe, compressed_kv.scalar_type());
    TORCH_CHECK(k_pe.dim() == 2);
    CHECK_INPUT(cu_seq_lens, torch::kInt64);
    TORCH_CHECK(cu_seq_lens.dim() == 1);
    TORCH_CHECK(cu_seq_lens.size(0) >= num_contexts + 1);
    CHECK_INPUT(cu_ctx_cached_kv_lens, torch::kInt64);
    TORCH_CHECK(cu_ctx_cached_kv_lens.dim() == 1);
    TORCH_CHECK(cu_ctx_cached_kv_lens.size(0) >= num_contexts + 1);
    TORCH_CHECK(max_input_uncached_seq_len > 0);

    auto kv_cache_quant_mode = tc::QuantMode(static_cast<uint32_t>(quant_mode));
    int max_blocks_per_sequence = kv_cache_block_offsets.size(-1);
    KVBlockArray kv_cache_buffer
        = createKVBlockArray(num_contexts, max_blocks_per_sequence, tokens_per_block, head_size,
            1,                           // num_kv_heads is always 1 for MLA
            attention_window_size, sink_token_length, beam_width, kv_cache_quant_mode,
            compressed_kv.scalar_type(), // TODO(zhhuang): support more output dtypes
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

    if (compressed_kv.scalar_type() == torch::kFloat16)
    {
        appendPagedKVCacheForMLAHelper<half>(kv_cache_buffer, compressed_kv, k_pe, num_contexts, cu_ctx_cached_kv_lens,
            cu_seq_lens, max_input_uncached_seq_len, head_size);
    }
    else if (compressed_kv.scalar_type() == torch::kFloat32)
    {
        appendPagedKVCacheForMLAHelper<float>(kv_cache_buffer, compressed_kv, k_pe, num_contexts, cu_ctx_cached_kv_lens,
            cu_seq_lens, max_input_uncached_seq_len, head_size);
    }
    else if (compressed_kv.scalar_type() == torch::kBFloat16)
    {
        appendPagedKVCacheForMLAHelper<__nv_bfloat16>(kv_cache_buffer, compressed_kv, k_pe, num_contexts,
            cu_ctx_cached_kv_lens, cu_seq_lens, max_input_uncached_seq_len, head_size);
    }
}

} // namespace torch_ext

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "load_paged_kv_cache_for_mla("
        "ScalarType out_dtype"
        ", int num_contexts"
        ", int max_ctx_cached_kv_len"
        ", Tensor cu_ctx_cached_kv_lens"
        ", Tensor kv_cache_block_offsets"
        ", Tensor host_kv_cache_block_offsets"
        ", Tensor host_kv_cache_pool_pointers"
        ", Tensor host_kv_cache_pool_mapping"
        ", Tensor? kv_scale_orig_quant"
        ", Tensor? kv_scale_quant_orig"
        ", int layer_idx"
        ", int head_size"
        ", int tokens_per_block"
        ", int attention_window_size"
        ", int sink_token_length"
        ", int beam_width"
        ", int quant_mode"
        ") -> Tensor");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("load_paged_kv_cache_for_mla", &torch_ext::loadPagedKVCacheForMLA);
}

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "set_paged_kv_cache_for_mla("
        "Tensor output"
        ", Tensor k"
        ", Tensor v"
        ", Tensor k_pe"
        ", int num_requests"
        ", Tensor cu_seq_lens"
        ", int max_input_seq_len"
        ", int num_heads"
        ", int kv_dim"
        ", int rope_dim"
        ", int kv_cache_tokens_per_block"
        ") -> Tensor");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("set_paged_kv_cache_for_mla", &torch_ext::setPagedKVCacheForMLA);
}

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "set_paged_kv_cache_v2_for_mla("
        "Tensor output"
        ", Tensor cached_k"
        ", Tensor cached_v"
        ", Tensor cached_k_pe"
        ", Tensor new_k"
        ", Tensor new_v"
        ", Tensor new_k_pe"
        ", int num_requests"
        ", Tensor cu_ctx_cached_kv_lens"
        ", Tensor cu_seq_lens"
        ", int max_input_seq_len"
        ", int num_heads"
        ", int kv_dim"
        ", int rope_dim"
        ", int kv_cache_tokens_per_block"
        ") -> Tensor");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("set_paged_kv_cache_v2_for_mla", &torch_ext::setPagedKVCacheV2ForMLA);
}

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "append_paged_kv_cache_for_mla("
        "Tensor compressed_kv"
        ", Tensor k_pe"
        ", int num_contexts"
        ", Tensor cu_ctx_cached_kv_lens"
        ", Tensor cu_seq_lens"
        ", int max_input_uncached_seq_len"
        ", Tensor kv_cache_block_offsets"
        ", Tensor host_kv_cache_block_offsets"
        ", Tensor host_kv_cache_pool_pointers"
        ", Tensor host_kv_cache_pool_mapping"
        ", Tensor? kv_scale_orig_quant"
        ", Tensor? kv_scale_quant_orig"
        ", int layer_idx"
        ", int head_size"
        ", int tokens_per_block"
        ", int attention_window_size"
        ", int sink_token_length"
        ", int beam_width"
        ", int quant_mode"
        ") -> ()");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("append_paged_kv_cache_for_mla", &torch_ext::appendPagedKVCacheForMLA);
}
