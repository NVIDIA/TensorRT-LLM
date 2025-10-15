/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION &
 * AFFILIATES. All rights reserved. SPDX-License-Identifier: Apache-2.0
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
#include "tensorrt_llm/kernels/gptKernels.h"
#include "tensorrt_llm/kernels/mlaKernels.h"
#include "tensorrt_llm/kernels/sparseAttentionKernels.h"
#include "tensorrt_llm/runtime/torchUtils.h"
#include "tensorrt_llm/runtime/utils/debugUtils.h"
#include "tensorrt_llm/thop/attentionOp.h"
#include "tensorrt_llm/thop/thUtils.h"
#include <cstdint>
#include <functional>
#include <torch/extension.h>
#include <unordered_set>

namespace torch_ext
{
using tensorrt_llm::common::op::AttentionOp;
using tensorrt_llm::common::op::hash;
using tensorrt_llm::runtime::RequestType;

namespace trtllm::attention
{
using tensorrt_llm::kernels::KVBlockArray;
using tensorrt_llm::kernels::MlaParams;
using tensorrt_llm::kernels::SparseAttentionParams;

enum class AttentionInputType : int8_t
{
    Mixed,
    ContextOnly,
    GenerationOnly,
};

class RunnerBase
{
public:
    int32_t beam_width;
    int32_t max_num_requests;
    int32_t attention_window_size;
    int32_t sink_token_length;

    auto data() const
    {
        return std::make_tuple(beam_width, max_num_requests, attention_window_size, sink_token_length);
    };

    virtual ~RunnerBase() = default;
    virtual void prepare(AttentionOp& op) const = 0;
    virtual int64_t getWorkspaceSize(AttentionOp const& op, int const num_tokens, int const max_attention_window_size,
        int const num_gen_tokens, int const max_blocks_per_sequence) const
        = 0;
    // typically, we use single qkv input, but for context MLA, we use separate qkv inputs
    virtual void run(AttentionOp& op, bool const is_context, int32_t const seq_offset, int32_t const num_seqs,
        int32_t const token_offset, int32_t const num_tokens, int32_t const predicted_tokens_per_seq,
        torch::Tensor workspace, torch::Tensor output, torch::optional<torch::Tensor> output_sf, torch::Tensor qkv_or_q,
        torch::optional<torch::Tensor> k, torch::optional<torch::Tensor> v, torch::Tensor sequence_length,
        torch::Tensor host_past_key_value_lengths, int32_t const total_kv_len, torch::Tensor context_lengths,
        torch::Tensor host_context_lengths, torch::optional<torch::Tensor> kv_cache_block_offsets,
        torch::optional<torch::Tensor> host_kv_cache_block_offsets,
        torch::optional<torch::Tensor> host_kv_cache_pool_pointers,
        torch::optional<torch::Tensor> host_kv_cache_pool_mapping, torch::optional<torch::Tensor> cache_indirection,
        torch::optional<torch::Tensor> kv_scale_orig_quant, torch::optional<torch::Tensor> kv_scale_quant_orig,
        torch::optional<torch::Tensor> out_scale, torch::optional<torch::Tensor> rotary_inv_freq,
        torch::optional<torch::Tensor> rotary_cos_sin, torch::optional<torch::Tensor> latent_cache,
        torch::optional<torch::Tensor> q_pe, torch::optional<torch::Tensor> block_ids_per_seq,
        torch::optional<torch::Tensor> mrope_rotary_cos_sin, torch::optional<torch::Tensor> mrope_position_deltas,
        std::vector<std::optional<torch::Tensor>> mla_tensor_params,
        torch::optional<torch::Tensor> softmax_stats_tensor,
        c10::ArrayRef<std::optional<torch::Tensor>> spec_decoding_tensor_params,
        torch::optional<torch::Tensor> attention_sinks, torch::optional<torch::Tensor> sparse_kv_indices,
        torch::optional<torch::Tensor> sparse_kv_offsets, torch::optional<torch::Tensor> sparse_attn_indices,
        torch::optional<torch::Tensor> sparse_attn_offsets) const
        = 0;
};

template <typename T, typename AttentionOutT = T>
class Runner : public RunnerBase
{
public:
    void prepare(AttentionOp& op) const override
    {
        AttentionOp::EnqueueGenerationParams<T> enqueueParams;
        enqueueParams.max_attention_window_size = attention_window_size;
        enqueueParams.cyclic_attention_window_size = attention_window_size;
        enqueueParams.max_cyclic_attention_window_size = attention_window_size;
        enqueueParams.sink_token_length = sink_token_length;
        enqueueParams.beam_width = beam_width;
        enqueueParams.num_requests = max_num_requests;

        op.prepareEnqueueGeneration<T, KVBlockArray>(enqueueParams);

        // Always reserve SemaphoreArray (for multi-block mode) as MMHA may enable multi-block mode when shared memory
        // is not enough.
        // The attention kernel might split the heads into multiple blocks, so we might need to reserve more semaphores.
        // Use mMultiProcessorCount as the lower-bound to make sure we reserve enough semaphores.
        op.reserveSemaphoreArray(std::max(op.mNumHeads * max_num_requests, op.getMultiProcessorCount()));
    }

    int64_t getWorkspaceSize(AttentionOp const& op, int const num_tokens, int const max_attention_window_size,
        int const num_gen_tokens, int const max_blocks_per_sequence) const override
    {
        size_t const context_workspace_size
            = op.getWorkspaceSizeForContext(op.mType, max_num_requests, op.mMaxContextLength, 0, num_tokens);
        size_t const generation_workspace_size = op.getWorkspaceSizeForGeneration(
            op.mType, max_num_requests, max_attention_window_size, num_gen_tokens, max_blocks_per_sequence);

        return std::max(context_workspace_size, generation_workspace_size);
    }

    void run(AttentionOp& op, bool const is_context, int32_t const seq_offset, int32_t const num_seqs,
        int32_t const token_offset, int32_t const num_tokens, int32_t const predicted_tokens_per_seq,
        torch::Tensor workspace, torch::Tensor output, torch::optional<torch::Tensor> output_sf, torch::Tensor qkv_or_q,
        torch::optional<torch::Tensor> k, torch::optional<torch::Tensor> v, torch::Tensor sequence_length,
        torch::Tensor host_past_key_value_lengths, int32_t const total_kv_len, torch::Tensor context_lengths,
        torch::Tensor host_context_lengths, torch::optional<torch::Tensor> kv_cache_block_offsets,
        torch::optional<torch::Tensor> host_kv_cache_block_offsets,
        torch::optional<torch::Tensor> host_kv_cache_pool_pointers,
        torch::optional<torch::Tensor> host_kv_cache_pool_mapping, torch::optional<torch::Tensor> cache_indirection,
        torch::optional<torch::Tensor> kv_scale_orig_quant, torch::optional<torch::Tensor> kv_scale_quant_orig,
        torch::optional<torch::Tensor> out_scale, torch::optional<torch::Tensor> rotary_inv_freq,
        torch::optional<torch::Tensor> rotary_cos_sin, torch::optional<torch::Tensor> latent_cache,
        torch::optional<torch::Tensor> q_pe, torch::optional<torch::Tensor> block_ids_per_seq,
        torch::optional<torch::Tensor> mrope_rotary_cos_sin, torch::optional<torch::Tensor> mrope_position_deltas,
        std::vector<std::optional<torch::Tensor>> mla_tensor_params,
        torch::optional<torch::Tensor> softmax_stats_tensor,
        c10::ArrayRef<std::optional<torch::Tensor>> spec_decoding_tensor_params,
        torch::optional<torch::Tensor> attention_sinks, torch::optional<torch::Tensor> sparse_kv_indices,
        torch::optional<torch::Tensor> sparse_kv_offsets, torch::optional<torch::Tensor> sparse_attn_indices,
        torch::optional<torch::Tensor> sparse_attn_offsets) const override
    {
        auto stream = at::cuda::getCurrentCUDAStream(qkv_or_q.get_device());
        T* attention_input = static_cast<T*>(qkv_or_q.slice(0, token_offset).data_ptr());
        T* k_ptr = nullptr;
        T* v_ptr = nullptr;
        AttentionOutT* context_buf = static_cast<AttentionOutT*>(output.slice(0, token_offset).data_ptr());
        TORCH_CHECK(!op.mFuseFp4Quant || output_sf.has_value());
        void* context_buf_sf = op.mFuseFp4Quant ? output_sf->data_ptr() : nullptr;

        // Rotary inv_freq, cos_sin cache to avoid re-computing.
        float const* rotary_inv_freq_ptr = nullptr;
        float2 const* rotary_cos_sin_ptr = nullptr;

        if (op.isRoPE())
        {
            if (rotary_inv_freq.has_value())
            {
                rotary_inv_freq_ptr = rotary_inv_freq.value().data_ptr<float>();
            }
            if (rotary_cos_sin.has_value())
            {
                rotary_cos_sin_ptr = static_cast<float2 const*>(rotary_cos_sin.value().data_ptr());
            }
        }

        void* workspace_ptr = workspace.data_ptr();
        [[maybe_unused]] MlaParams<T> mla_params;
        if (op.isMLAEnabled())
        {
            TORCH_CHECK(mla_tensor_params.size() == 1,
                "Expecting 1 tensor for custom MLA tensor params: helix_position_offsets.");
            if (is_context)
            {
                if (latent_cache.has_value())
                {
                    mla_params.latent_cache = static_cast<T const*>(latent_cache->data_ptr());
                }
                else
                {
                    // kv cache reuse / chunked context cases, latent_cache is not used
                    mla_params.latent_cache = nullptr;
                }
                TORCH_CHECK(k.has_value());
                TORCH_CHECK(v.has_value());
                TORCH_CHECK(k->dim() == 2);
                TORCH_CHECK(v->dim() == 2);
                TORCH_CHECK(k->strides()[1] == 1);
                TORCH_CHECK(v->strides()[1] == 1);

                k_ptr = static_cast<T*>(k->slice(0, token_offset).data_ptr());
                v_ptr = static_cast<T*>(v->slice(0, token_offset).data_ptr());
                mla_params.k_buf = k_ptr;
                mla_params.v_buf = v_ptr;
            }
            else
            {
                TORCH_CHECK(latent_cache.has_value());
                mla_params.latent_cache = static_cast<T const*>(latent_cache->data_ptr());
                TORCH_CHECK(q_pe.has_value());
                TORCH_CHECK(q_pe->dim() == 3);
                TORCH_CHECK(q_pe->strides()[2] == 1);

                mla_params.q_pe = static_cast<T*>(q_pe->data_ptr());
                mla_params.q_pe_ld = q_pe->strides()[1];
                mla_params.q_pe_stride = q_pe->strides()[0];
            }
            mla_params.q_buf = attention_input;
            mla_params.context_buf = reinterpret_cast<T*>(context_buf);

            mla_params.cos_sin_cache = rotary_cos_sin_ptr;
            mla_params.batch_size = num_seqs;
            mla_params.acc_q_len = num_tokens;
            mla_params.head_num = op.mNumHeads;
            mla_params.meta = op.mMLAParams;

            mla_params.workspace = workspace_ptr;
            auto& mla_helix_position_offsets = mla_tensor_params[0];
            if (mla_helix_position_offsets.has_value())
            {
                mla_params.helix_position_offsets = mla_helix_position_offsets->data_ptr<int32_t>();
            }
        }

        // Prepare sparse attention parameters
        if (is_context)
        {
            op.mRuntimeSparseAttentionParams.sparse_kv_indices
                = sparse_kv_indices.has_value() ? sparse_kv_indices.value().data_ptr<int32_t>() : nullptr;
            op.mRuntimeSparseAttentionParams.sparse_kv_offsets
                = sparse_kv_offsets.has_value() ? sparse_kv_offsets.value().data_ptr<int32_t>() : nullptr;
        }
        else
        {
            op.mRuntimeSparseAttentionParams.sparse_attn_indices
                = sparse_attn_indices.has_value() ? sparse_attn_indices.value().data_ptr<int32_t>() : nullptr;
            op.mRuntimeSparseAttentionParams.sparse_attn_offsets
                = sparse_attn_offsets.has_value() ? sparse_attn_offsets.value().data_ptr<int32_t>() : nullptr;
        }

        int const* context_lengths_ptr = context_lengths.slice(0, seq_offset).data_ptr<int>();
        int const* sequence_lengths_ptr = sequence_length.slice(0, seq_offset).data_ptr<int>();
        // Note we still need context length during generation for MMHA optimization.
        int32_t const max_context_q_len
            = host_context_lengths.slice(0, seq_offset, seq_offset + num_seqs).max().item<int32_t>();
        int32_t const max_past_kv_length
            = host_past_key_value_lengths.slice(0, seq_offset, seq_offset + num_seqs).max().item<int32_t>();

        // Commonly, cyclic_attention_window_size, and max_attention_window_size will be the same
        // unless each layer has different attention window sizes.
        // the kv_cache capacity.
        int const max_attention_window_size = beam_width == 1 ? attention_window_size
            : cache_indirection.has_value()                   ? cache_indirection.value().size(2)
                                                              : attention_window_size;
        // The cyclic_attention_window_size will determine the cyclic kv cache position of new tokens.
        // Note that this cyclic_attention_window_size might be smaller than the actual kv cache capactity.
        int const cyclic_attention_window_size = attention_window_size;
        bool const can_use_one_more_block = beam_width > 1;

        int max_blocks_per_sequence
            = op.useKVCache() && kv_cache_block_offsets.has_value() ? kv_cache_block_offsets.value().size(-1) : 0;
        int32_t const pool_index = op.useKVCache() && host_kv_cache_pool_mapping.has_value()
            ? host_kv_cache_pool_mapping.value().index({op.mLayerIdx, 0}).item<int32_t>()
            : 0;
        int32_t const layer_idx_in_cache_pool = op.useKVCache() && host_kv_cache_pool_mapping.has_value()
            ? host_kv_cache_pool_mapping.value().index({op.mLayerIdx, 1}).item<int32_t>()
            : 0;
        KVBlockArray::DataType* block_offsets
            = static_cast<KVBlockArray::DataType*>(op.useKVCache() && kv_cache_block_offsets.has_value()
                    ? kv_cache_block_offsets.value().index({pool_index, seq_offset}).data_ptr()
                    : nullptr);
        KVBlockArray::DataType* host_block_offsets
            = static_cast<KVBlockArray::DataType*>(op.useKVCache() && host_kv_cache_block_offsets.has_value()
                    ? host_kv_cache_block_offsets.value().index({pool_index, seq_offset}).data_ptr()
                    : nullptr);

        // The cache element size in bits.
        int cache_elem_bits = op.getKvCacheElemSizeInBits<T>();
        auto const block_size = op.mTokensPerBlock * op.mNumKVHeads * op.mHeadSize;
        auto const bytes_per_block = block_size * cache_elem_bits / 8 /*bits*/;
        int32_t const kv_factor = op.isMLAEnabled() ? 1 : 2;
        auto const intra_pool_offset = layer_idx_in_cache_pool * kv_factor * bytes_per_block;

        // Prepare block pool pointers for NVFP4 KV cache.
        void* host_primary_pool_pointer{nullptr};
        void* host_secondary_pool_pointer{nullptr};
        void* host_primary_block_scale_pool_pointer{nullptr};
        void* host_secondary_block_scale_pool_pointer{nullptr};

        // Whether NVFP4 KV cache is used.
        bool const use_kv_cache = op.useKVCache() && host_kv_cache_pool_pointers.has_value();
        bool const use_nvfp4_kv_cache = use_kv_cache && op.mKVCacheQuantMode.hasFp4KvCache();
        if (use_nvfp4_kv_cache)
        {
            // For NVFP4 KV cache, extra block scales are stored in separate pools.
            // The layout of host_kv_cache_pool_pointers is [num_pools, 2 (primary and secondary), 2 (data and scale)].
            TORCH_CHECK(host_kv_cache_pool_pointers.value().dim() == 3);
            host_primary_pool_pointer = reinterpret_cast<void*>(
                reinterpret_cast<char*>(host_kv_cache_pool_pointers.value().index({pool_index, 0, 0}).item<int64_t>())
                + intra_pool_offset);
            host_secondary_pool_pointer = reinterpret_cast<void*>(
                reinterpret_cast<char*>(host_kv_cache_pool_pointers.value().index({pool_index, 1, 0}).item<int64_t>())
                + intra_pool_offset);
            // Calculate the intra-pool offset for scaling factors.
            // Note that NVFP4 block scaling use a fixed vector size of 16.
            auto constexpr vector_size = 16;
            auto const bytes_per_block_sf = block_size / vector_size * 1 /*bytes per E4M3 sf*/;
            auto const intra_pool_offset_sf = layer_idx_in_cache_pool * kv_factor * bytes_per_block_sf;
            host_primary_block_scale_pool_pointer = reinterpret_cast<void*>(
                reinterpret_cast<char*>(host_kv_cache_pool_pointers.value().index({pool_index, 0, 1}).item<int64_t>())
                + intra_pool_offset_sf);
            host_secondary_block_scale_pool_pointer = reinterpret_cast<void*>(
                reinterpret_cast<char*>(host_kv_cache_pool_pointers.value().index({pool_index, 1, 1}).item<int64_t>())
                + intra_pool_offset_sf);
        }
        else if (use_kv_cache)
        {
            TORCH_CHECK(host_kv_cache_pool_pointers.value().dim() == 2);
            host_primary_pool_pointer = reinterpret_cast<void*>(
                reinterpret_cast<char*>(host_kv_cache_pool_pointers.value().index({pool_index, 0}).item<int64_t>())
                + intra_pool_offset);
            host_secondary_pool_pointer = reinterpret_cast<void*>(
                reinterpret_cast<char*>(host_kv_cache_pool_pointers.value().index({pool_index, 1}).item<int64_t>())
                + intra_pool_offset);
        }

        float const* kv_scale_orig_quant_ptr = nullptr;
        float const* kv_scale_quant_orig_ptr = nullptr;
        if (op.mKVCacheQuantMode.hasKvCacheQuant() && kv_scale_orig_quant.has_value()
            && kv_scale_quant_orig.has_value())
        {
            kv_scale_orig_quant_ptr = kv_scale_orig_quant.value().data_ptr<float>();
            kv_scale_quant_orig_ptr = kv_scale_quant_orig.value().data_ptr<float>();
            if (op.mKVCacheQuantMode.hasFp4KvCache())
            {
                TORCH_CHECK(kv_scale_orig_quant.value().size(0) == 3);
                TORCH_CHECK(kv_scale_quant_orig.value().size(0) == 3);
            }
        }
        // For FP8 output, out_scale represents the output scale.
        float const* out_scale_ptr = (op.mFP8ContextFMHA && !op.mFuseFp4Quant && out_scale.has_value())
            ? out_scale.value().data_ptr<float>()
            : nullptr;
        // For NVFP4 output, out_scale holds the global scale for scaling factors.
        float const* out_sf_scale_ptr
            = op.mFuseFp4Quant && out_scale.has_value() ? out_scale.value().data_ptr<float>() : nullptr;

        // The attention_sinks is a float tensor with shape [num_heads_q]
        float const* attention_sinks_ptr = nullptr;
        if (attention_sinks.has_value())
        {
            TORCH_CHECK(
                attention_sinks.value().dtype() == torch::kFloat32, "Expected attention_sinks to have float dtype");
            attention_sinks_ptr = attention_sinks.value().data_ptr<float>();
        }

        AttentionOp::EnqueueParams<T> common_enqueue_params;
        common_enqueue_params.attention_input = attention_input;
        common_enqueue_params.attention_sinks = attention_sinks_ptr;
        common_enqueue_params.rotary_inv_freq = rotary_inv_freq_ptr;
        common_enqueue_params.rotary_cos_sin = rotary_cos_sin_ptr;
        common_enqueue_params.max_past_kv_length = max_past_kv_length;
        common_enqueue_params.max_attention_window_size = max_attention_window_size;
        common_enqueue_params.cyclic_attention_window_size = cyclic_attention_window_size;
        common_enqueue_params.max_cyclic_attention_window_size = cyclic_attention_window_size;
        common_enqueue_params.can_use_one_more_block = can_use_one_more_block;
        common_enqueue_params.sink_token_length = sink_token_length;
        common_enqueue_params.kv_scale_orig_quant = kv_scale_orig_quant_ptr;
        common_enqueue_params.kv_scale_quant_orig = kv_scale_quant_orig_ptr;
        common_enqueue_params.attention_output_orig_quant = out_scale_ptr;
        common_enqueue_params.attention_output_sf_scale = out_sf_scale_ptr;
        common_enqueue_params.context_buf = context_buf;
        common_enqueue_params.context_buf_sf = context_buf_sf;
        common_enqueue_params.block_offsets = block_offsets;
        common_enqueue_params.host_primary_pool_pointer = host_primary_pool_pointer;
        common_enqueue_params.host_secondary_pool_pointer = host_secondary_pool_pointer;
        common_enqueue_params.host_primary_block_scale_pool_pointer = host_primary_block_scale_pool_pointer;
        common_enqueue_params.host_secondary_block_scale_pool_pointer = host_secondary_block_scale_pool_pointer;
        common_enqueue_params.num_tokens = num_tokens;
        common_enqueue_params.total_kv_len = total_kv_len;
        common_enqueue_params.max_blocks_per_sequence = max_blocks_per_sequence;
        common_enqueue_params.sequence_lengths = sequence_lengths_ptr;
        common_enqueue_params.context_lengths = context_lengths_ptr;
        common_enqueue_params.host_context_lengths = host_context_lengths.data_ptr<int32_t>();
        common_enqueue_params.workspace = workspace_ptr;
        if (softmax_stats_tensor.has_value())
        {
            TLLM_CHECK_WITH_INFO(softmax_stats_tensor.value().scalar_type() == at::ScalarType::Float,
                "softmax_stats_tensor must have float type");
            TLLM_CHECK_WITH_INFO(softmax_stats_tensor.value().size(0) >= num_tokens,
                "softmax_stats_tensor must have first dimension >= num_tokens");
            TLLM_CHECK_WITH_INFO(softmax_stats_tensor.value().size(1) >= op.mNumHeads,
                "softmax_stats_tensor must have second dimension >= num_heads");
            TLLM_CHECK_WITH_INFO(
                softmax_stats_tensor.value().size(2) == 2, "softmax_stats_tensor must have third dimension == 2");
            common_enqueue_params.softmax_stats = static_cast<float2*>(softmax_stats_tensor.value().data_ptr());
        }

        if (is_context) // context stage
        {
            common_enqueue_params.input_seq_length = max_context_q_len;
            AttentionOp::EnqueueContextParams<T> enqueue_params{common_enqueue_params};
            enqueue_params.host_block_offsets = host_block_offsets;
            enqueue_params.batch_size = num_seqs;
            enqueue_params.k_ptr = k_ptr;
            enqueue_params.v_ptr = v_ptr;

            if (op.isMLAEnabled())
            {
                mla_params.cache_seq_lens = sequence_lengths_ptr;
                mla_params.max_input_seq_len = max_context_q_len;
                enqueue_params.mla_param = &mla_params;
            }
            if (op.isMRoPE() && mrope_rotary_cos_sin.has_value())
            {
                enqueue_params.mrope_rotary_cos_sin
                    = static_cast<float2 const*>(mrope_rotary_cos_sin.value().data_ptr());
            }
            op.enqueueContext<T, KVBlockArray>(enqueue_params, stream);
        }
        else // generation stage
        {
            int32_t const batch_beam = num_seqs;
            TLLM_CHECK(batch_beam % beam_width == 0);
            int32_t const num_requests = batch_beam / beam_width;

            TLLM_CHECK_WITH_INFO(num_tokens % num_seqs == 0,
                "seq_len should be same for all generation requests, num_tokens=%d, num_seqs=%d", num_tokens, num_seqs);
            int32_t const input_seq_length = num_tokens / num_seqs;

            common_enqueue_params.input_seq_length = input_seq_length;
            AttentionOp::EnqueueGenerationParams<T> enqueue_params{common_enqueue_params};
            enqueue_params.beam_width = beam_width;
            enqueue_params.num_requests = num_requests;
            enqueue_params.cache_indir = beam_width == 1
                ? nullptr
                : (cache_indirection.has_value() ? cache_indirection.value().data_ptr<int32_t>() : nullptr);
            enqueue_params.semaphores = op.multiBlockSemaphores();
            enqueue_params.host_past_key_value_lengths = host_past_key_value_lengths.data_ptr<int32_t>();
            enqueue_params.start_token_idx_sf = token_offset;

            if (op.isMRoPE() && mrope_position_deltas.has_value())
            {
                enqueue_params.mrope_position_deltas = mrope_position_deltas.value().data_ptr<int32_t>();
            }
            if (op.mIsSpecDecodingEnabled && op.mUseSpecDecoding)
            {
                TORCH_CHECK(spec_decoding_tensor_params.size() == 3,
                    "Expecting 3 tensors for spec-dec mode, spec_decoding_generation_lengths, "
                    "spec_decoding_position_offsets and spec_decoding_packed_mask.");
                TORCH_CHECK(spec_decoding_tensor_params[0].has_value(),
                    "Expecting spec_decoding_generation_lengths spec-dec mode.");
                TORCH_CHECK(spec_decoding_tensor_params[1].has_value(),
                    "Expecting spec_decoding_position_offsets spec-dec mode.");
                TORCH_CHECK(
                    spec_decoding_tensor_params[2].has_value(), "Expecting spec_decoding_packed_mask spec-dec mode.");

                enqueue_params.spec_decoding_generation_lengths
                    = spec_decoding_tensor_params[0].value().data_ptr<int32_t>();
                enqueue_params.spec_decoding_position_offsets
                    = spec_decoding_tensor_params[1].value().data_ptr<int32_t>();
                enqueue_params.spec_decoding_packed_mask = spec_decoding_tensor_params[2].value().data_ptr<int32_t>();
                enqueue_params.spec_decoding_is_generation_length_variable = true;
                enqueue_params.spec_decoding_max_generation_length = input_seq_length + 1;
            }

            // Current mlaGeneration will using fmha to do attention, so we don't go into enqueueGeneration
            if (op.isMLAEnabled())
            {
                if (op.mUseGenFlashMLA == true)
                {
                    TORCH_CHECK(block_ids_per_seq.has_value());
                    int const* block_ids_per_seq_ptr = static_cast<int*>(block_ids_per_seq->data_ptr());
                    mla_params.block_ids_per_seq = block_ids_per_seq_ptr;
                }
                mla_params.cache_seq_lens = sequence_lengths_ptr;
                op.mlaGeneration<T>(mla_params, enqueue_params, stream);
            }
            else
            {
                op.enqueueGeneration<T, KVBlockArray>(enqueue_params, stream);
            }

            {
                std::string const afterGenStr = "gen attention at layer " + std::to_string(op.mLayerIdx);
                {
                    TLLM_CHECK_DEBUG_WITH_INFO(tensorrt_llm::runtime::utils::tensorHasInvalid(num_tokens,
                                                   output.size(1), op.mType, context_buf, stream, afterGenStr)
                            == false,
                        "Found invalid number (NaN or Inf) in " + afterGenStr);
                }
            }
        }
        sync_check_cuda_error(stream);
    }
};

template class Runner<float>;
template class Runner<half>;
template class Runner<half, __nv_fp8_e4m3>;
#ifdef ENABLE_BF16
template class Runner<__nv_bfloat16>;
template class Runner<__nv_bfloat16, __nv_fp8_e4m3>;
#endif

} // namespace trtllm::attention

using RunnerPtr = std::shared_ptr<torch_ext::trtllm::attention::RunnerBase>;
using torch_ext::trtllm::attention::Runner;
using torch_ext::trtllm::attention::AttentionInputType;

void attention(torch::Tensor q, std::optional<torch::Tensor> k, std::optional<torch::Tensor> v, torch::Tensor& output,
    std::optional<torch::Tensor> output_sf, std::optional<torch::ScalarType> out_dtype,
    std::optional<torch::Tensor> workspace_, torch::Tensor sequence_length, torch::Tensor host_past_key_value_lengths,
    torch::Tensor host_total_kv_lens, torch::Tensor context_lengths, torch::Tensor host_context_lengths,
    torch::Tensor host_request_types, std::optional<torch::Tensor> kv_cache_block_offsets,
    std::optional<torch::Tensor> host_kv_cache_block_offsets, std::optional<torch::Tensor> host_kv_cache_pool_pointers,
    std::optional<torch::Tensor> host_kv_cache_pool_mapping, std::optional<torch::Tensor> cache_indirection,
    std::optional<torch::Tensor> kv_scale_orig_quant, std::optional<torch::Tensor> kv_scale_quant_orig,
    std::optional<torch::Tensor> out_scale, std::optional<torch::Tensor> rotary_inv_freq,
    std::optional<torch::Tensor> rotary_cos_sin, std::optional<torch::Tensor> latent_cache,
    std::optional<torch::Tensor> q_pe, std::optional<torch::Tensor> block_ids_per_seq,
    std::optional<torch::Tensor> attention_sinks, bool const is_fused_qkv, bool const update_kv_cache,
    int64_t const predicted_tokens_per_seq, int64_t const layer_idx, int64_t const num_heads,
    int64_t const num_kv_heads, int64_t const head_size, std::optional<int64_t> const tokens_per_block,
    int64_t const max_num_requests, int64_t const max_context_length, int64_t const attention_window_size,
    int64_t const sink_token_length, int64_t const beam_width, int64_t const mask_type, int64_t const quant_mode,
    double const q_scaling, int64_t const position_embedding_type, int64_t const rotary_embedding_dim,
    double const rotary_embedding_base, int64_t const rotary_embedding_scale_type,
    std::vector<double> rotary_embedding_scales, std::vector<int64_t> rotary_embedding_max_position_info,
    bool const use_paged_context_fmha, std::optional<int64_t> attention_input_type, bool is_mla_enable,
    std::optional<int64_t> chunked_prefill_buffer_batch_size, std::optional<int64_t> q_lora_rank,
    std::optional<int64_t> kv_lora_rank, std::optional<int64_t> qk_nope_head_dim,
    std::optional<int64_t> qk_rope_head_dim, std::optional<int64_t> v_head_dim,
    std::optional<torch::Tensor> mrope_rotary_cos_sin, std::optional<torch::Tensor> mrope_position_deltas,
    std::vector<std::optional<torch::Tensor>> mla_tensor_params, std::optional<int64_t> attention_chunk_size,
    std::optional<torch::Tensor> softmax_stats_tensor, std::vector<bool> spec_decoding_bool_params,
    std::vector<std::optional<torch::Tensor>> spec_decoding_tensor_params,
    std::vector<std::optional<torch::Tensor>> sparse_attention_params)
{
    // Decompress sparse attention parameters
    TORCH_CHECK(sparse_attention_params.size() == 4, "Expected 4 sparse attention parameters");
    torch::optional<torch::Tensor> sparse_kv_indices = sparse_attention_params[0];
    torch::optional<torch::Tensor> sparse_kv_offsets = sparse_attention_params[1];
    torch::optional<torch::Tensor> sparse_attn_indices = sparse_attention_params[2];
    torch::optional<torch::Tensor> sparse_attn_offsets = sparse_attention_params[3];

    TLLM_LOG_TRACE("Attention op starts at layer %d", layer_idx);
    // Use these tensors to infer if the attention is using KV cache
    bool const use_kv_cache = kv_cache_block_offsets.has_value() && host_kv_cache_block_offsets.has_value()
        && host_kv_cache_pool_pointers.has_value() && host_kv_cache_pool_mapping.has_value();

    TLLM_CHECK_WITH_INFO(is_mla_enable || is_fused_qkv, "Only fused QKV is supported for non-MLA attention now");
    TLLM_CHECK_WITH_INFO(update_kv_cache, "KV cache update cannot be disabled now");
    auto qkv_or_q = q;
    if (is_fused_qkv)
    {
        TLLM_CHECK_WITH_INFO(!k.has_value(), "The k tensor should be null if using fused QKV");
        TLLM_CHECK_WITH_INFO(!v.has_value(), "The v tensor should be null if using fused QKV");
    }
    if (!is_fused_qkv && update_kv_cache)
    {
        TLLM_CHECK_WITH_INFO(k.has_value(), "The k tensor should be provided if updating KV cache with unfused K/V");
        TLLM_CHECK_WITH_INFO(v.has_value(), "The v tensor should be provided if updating KV cache with unfused K/V");
    }

    auto const dtype = tensorrt_llm::runtime::TorchUtils::dataType(qkv_or_q.scalar_type());
    bool const is_fp8_out = out_dtype.has_value() && out_dtype.value() == torch::kFloat8_e4m3fn;
    bool const is_fp4_out = out_dtype.has_value() && out_dtype.value() == torch::kUInt8;

    RunnerPtr runner;
    if (dtype == nvinfer1::DataType::kHALF)
    {
        if (is_fp8_out)
        {
            runner = std::make_shared<Runner<half, __nv_fp8_e4m3>>();
        }
        else if (is_fp4_out)
        {
            runner = std::make_shared<Runner<half, __nv_fp4_e2m1>>();
        }
        else
        {
            TLLM_CHECK(!out_dtype.has_value() || out_dtype.value() == torch::kFloat16);
            runner = std::make_shared<Runner<half>>();
        }
    }
    else if (dtype == nvinfer1::DataType::kFLOAT)
    {
        TLLM_CHECK(!out_dtype.has_value() || out_dtype.value() == torch::kFloat32);
        runner = std::make_shared<Runner<float>>();
    }
#ifdef ENABLE_BF16
    else if (dtype == nvinfer1::DataType::kBF16)
    {
        if (is_fp8_out)
        {
            runner = std::make_shared<Runner<__nv_bfloat16, __nv_fp8_e4m3>>();
        }
        else if (is_fp4_out)
        {
            runner = std::make_shared<Runner<__nv_bfloat16, __nv_fp4_e2m1>>();
        }
        else
        {
            TLLM_CHECK(!out_dtype.has_value() || out_dtype.value() == torch::kBFloat16);
            runner = std::make_shared<Runner<__nv_bfloat16>>();
        }
    }
#endif
    runner->beam_width = beam_width;
    runner->max_num_requests = max_num_requests;
    runner->attention_window_size = attention_window_size;
    runner->sink_token_length = sink_token_length;

    double const rotary_embedding_scale = rotary_embedding_scales[0];
    double const rotary_embedding_short_m_scale = rotary_embedding_scales[1];
    double const rotary_embedding_long_m_scale = rotary_embedding_scales[2];
    int64_t const rotary_embedding_max_positions = rotary_embedding_max_position_info[0];
    int64_t const rotary_embedding_original_max_positions = rotary_embedding_max_position_info[1];

    auto op = std::make_shared<AttentionOp>();
    op->mType = dtype;
    op->mFMHAForceFP32Acc = dtype == nvinfer1::DataType::kBF16;
    op->mLayerIdx = layer_idx;
    op->mNumHeads = num_heads;
    op->mNumKVHeads = num_kv_heads;
    op->mHeadSize = head_size;
    op->mMaskType = static_cast<tensorrt_llm::kernels::AttentionMaskType>(int32_t(mask_type));
    op->mKVCacheQuantMode = tensorrt_llm::common::QuantMode(uint32_t(quant_mode));
    op->mUseKVCache = use_kv_cache;
    op->mPagedKVCache = op->mPagedKVCache && use_kv_cache; // update mPagedKVCache based on use_kv_cache
    op->mTokensPerBlock = tokens_per_block.value_or(0);
    op->mFP8GenerationMLA = false;
    op->mFuseFp4Quant = is_fp4_out;
    op->mMaxContextLength = max_context_length;
    op->mQScaling = q_scaling;
    op->mPositionEmbeddingType
        = static_cast<tensorrt_llm::kernels::PositionEmbeddingType>(int8_t(position_embedding_type));
    op->mRotaryEmbeddingDim = rotary_embedding_dim;
    op->mRotaryEmbeddingBase = rotary_embedding_base;
    op->mRotaryEmbeddingScaleType
        = static_cast<tensorrt_llm::kernels::RotaryScalingType>(int8_t(rotary_embedding_scale_type));
    op->mRotaryEmbeddingScale = rotary_embedding_scale;
    op->mRotaryEmbeddingShortMscale = rotary_embedding_short_m_scale;
    op->mRotaryEmbeddingLongMscale = rotary_embedding_long_m_scale;
    op->mRotaryEmbeddingMaxPositions = rotary_embedding_max_positions;
    op->mRotaryEmbeddingOriginalMaxPositions = rotary_embedding_original_max_positions;
    op->mFP8ContextFMHA = is_fp8_out || is_fp4_out || (op->mKVCacheQuantMode.hasFp8KvCache() && use_paged_context_fmha);
    op->mFP8AttenOutput = is_fp8_out;
    op->mPagedContextFMHA = use_paged_context_fmha;

    op->mAttentionChunkSize = attention_chunk_size;

    TORCH_CHECK(spec_decoding_bool_params.size() == 3,
        "Expecting 3 bools for spec-dec mode, is_spec_decoding_enabled, use_spec_decoding, and is_spec_dec_tree.");
    op->mIsSpecDecodingEnabled = spec_decoding_bool_params[0]; // is_spec_decoding_enabled
    op->mUseSpecDecoding = spec_decoding_bool_params[1];       // use_spec_decoding
    op->mIsSpecDecTree = spec_decoding_bool_params[2];         // is_spec_dec_tree

    op->mUseSparseAttention = false;
    op->mUseTllmGenSparseAttention = false;
    if ((sparse_kv_indices.has_value() && sparse_kv_indices.value().numel() > 0)
        || (sparse_attn_indices.has_value() && sparse_attn_indices.value().numel() > 0))
    {
        op->mUseSparseAttention = true;
        if (sparse_attn_indices.has_value() && sparse_attn_indices.value().numel() > 0)
        {
            op->mUseTllmGenSparseAttention = true;
        }
    }

    if (is_mla_enable)
    {
        // MLA does not support NVFP4 output yet.
        TLLM_CHECK(!is_fp4_out);

        TLLM_CHECK(host_kv_cache_pool_mapping.has_value());
        int32_t const layer_num = host_kv_cache_pool_mapping.value().size(0);

        op->mIsMLAEnabled = true;
        op->mMLAParams = {static_cast<int>(q_lora_rank.value()), static_cast<int>(kv_lora_rank.value()),
            static_cast<int>(qk_nope_head_dim.value()), static_cast<int>(qk_rope_head_dim.value()),
            static_cast<int>(v_head_dim.value()), static_cast<int>(predicted_tokens_per_seq),
            static_cast<int>(layer_num)};

        op->mFP8ContextMLA
            = (tensorrt_llm::common::getSMVersion() == 90 || tensorrt_llm::common::getSMVersion() == 100
                  || tensorrt_llm::common::getSMVersion() == 103 || tensorrt_llm::common::getSMVersion() == 120)
            && op->mKVCacheQuantMode.hasFp8KvCache();
        op->mIsGenerationMLA = head_size == op->mMLAParams.kv_lora_rank + op->mMLAParams.qk_rope_head_dim;
        op->mFP8GenerationMLA = op->mKVCacheQuantMode.hasFp8KvCache();
        // only enable flash mla on sm90 and head_size == 576 and tokens_per_block == 64
        op->mUseGenFlashMLA = tensorrt_llm::common::getSMVersion() == 90 && tokens_per_block == 64;

        // The following two parameters are used to compute kvcache related parameters such as kvcache block_size. So
        // they need to be set to 1 and 512 + 64 for both context and generation. For MLA attention kernel configs,
        // mNumKVHeads/mHeadSize are overwritten in common/attentionOp.cpp.
        op->mNumKVHeads = 1;
        op->mHeadSize = op->mMLAParams.kv_lora_rank + op->mMLAParams.qk_rope_head_dim;

        // For chunked prefill MLA, we need larger buffer size for k and v
        op->mChunkPrefillBufferBatchSize
            = chunked_prefill_buffer_batch_size.has_value() ? chunked_prefill_buffer_batch_size.value() : 1;
    }

    auto cache_key = std::make_tuple(op->data(), runner->data());
    using CacheKey = decltype(cache_key);
    static std::unordered_map<CacheKey, std::shared_ptr<AttentionOp>, hash<CacheKey>> op_cache;
    if (auto it = op_cache.find(cache_key); it != op_cache.end())
    {
        TLLM_LOG_TRACE("Attention op for layer %d is cached", layer_idx);
        op = it->second;
    }
    else
    {
        TLLM_LOG_TRACE(
            "Preparing new attention op for layer %d with cache key: %s", layer_idx, to_string(cache_key).c_str());
        op->initialize();
        runner->prepare(*op);
        op_cache[cache_key] = op;
    }

    int32_t const num_seqs = host_context_lengths.size(0);
    RequestType const* request_types = static_cast<RequestType const*>(host_request_types.data_ptr());

    AttentionInputType attn_input_type = AttentionInputType::Mixed;
    if (attention_input_type.has_value())
    {
        attn_input_type = static_cast<AttentionInputType>(attention_input_type.value());
    }
    bool const is_gen_only = attn_input_type == AttentionInputType::GenerationOnly;

    int32_t num_contexts = 0;
    // count context requests
    for (int32_t idx = 0; idx < num_seqs; idx++)
    {
        if (request_types[idx] != RequestType::kCONTEXT)
        {
            break;
        }
        ++num_contexts;
    }
    int32_t const num_generations = num_seqs - num_contexts;
    int32_t const num_tokens = qkv_or_q.size(0);
    int32_t const num_ctx_tokens = host_context_lengths.slice(0, 0, num_contexts).sum().item<int32_t>();
    int32_t const num_gen_tokens = is_gen_only ? num_tokens : num_tokens - num_ctx_tokens;
    auto const ctx_total_kv_len = host_total_kv_lens.index({0}).item<int32_t>();
    auto const gen_total_kv_len = host_total_kv_lens.index({1}).item<int32_t>();

    for (int32_t idx = num_contexts; idx < num_seqs; idx++)
    {
        TLLM_CHECK(request_types[idx] == RequestType::kGENERATION);
    }

    int32_t const max_attention_window_size
        = beam_width == 1 ? attention_window_size : cache_indirection.value().size(2);
    int32_t const max_blocks_per_sequence
        = use_kv_cache && kv_cache_block_offsets.has_value() ? kv_cache_block_offsets.value().size(-1) : 0;
    int64_t const workspace_size
        = runner->getWorkspaceSize(*op, num_tokens, max_attention_window_size, num_gen_tokens, max_blocks_per_sequence);
    TLLM_LOG_TRACE("Expected workspace size is %ld bytes", workspace_size);

    if (workspace_size >= (16l << 30))
    {
        auto const [free_mem, total_mem] = tensorrt_llm::common::getDeviceMemoryInfo(false);
        if (workspace_size >= static_cast<int64_t const>(free_mem))
        {
            throw std::runtime_error("attention workspace size " + std::to_string(workspace_size)
                + " bytes, exceeds available CUDA memory " + std::to_string(free_mem) + " bytes");
        }
    }

    torch::Tensor workspace;
    if (workspace_.has_value())
    {
        if (workspace_.value().numel() < workspace_size)
        {
            TLLM_LOG_WARNING("Attention workspace size is not enough, increase the size from %ld bytes to %ld bytes",
                workspace_.value().numel(), workspace_size);
            workspace_.value().resize_({workspace_size});
        }
        workspace = workspace_.value();
    }
    else
    {
        workspace = torch::empty({workspace_size}, torch::dtype(torch::kByte).device(qkv_or_q.device()));
    }

    if ((num_contexts > 0) && (attn_input_type != AttentionInputType::GenerationOnly))
    {
        auto seq_offset = 0;
        auto token_offset = 0;
        runner->run(*op,
            /*is_context=*/true, seq_offset,
            /*num_seqs=*/num_contexts, token_offset,
            /*num_tokens=*/num_ctx_tokens, predicted_tokens_per_seq, workspace, output, output_sf, qkv_or_q, k, v,
            sequence_length, host_past_key_value_lengths, ctx_total_kv_len, context_lengths, host_context_lengths,
            kv_cache_block_offsets, host_kv_cache_block_offsets, host_kv_cache_pool_pointers,
            host_kv_cache_pool_mapping, cache_indirection, kv_scale_orig_quant, kv_scale_quant_orig, out_scale,
            rotary_inv_freq, rotary_cos_sin, latent_cache, q_pe, block_ids_per_seq, mrope_rotary_cos_sin,
            mrope_position_deltas, mla_tensor_params, softmax_stats_tensor, spec_decoding_tensor_params,
            attention_sinks, sparse_kv_indices, sparse_kv_offsets, sparse_attn_indices, sparse_attn_offsets);
    }

    if ((num_generations > 0) && (attn_input_type != AttentionInputType::ContextOnly))
    {

        auto seq_offset = num_contexts;
        auto token_offset = is_gen_only ? 0 : num_ctx_tokens;
        runner->run(*op,
            /*is_context=*/false, seq_offset,
            /*num_seqs=*/num_generations, token_offset,
            /*num_tokens=*/num_gen_tokens, predicted_tokens_per_seq, workspace, output, output_sf, qkv_or_q, k, v,
            sequence_length, host_past_key_value_lengths, gen_total_kv_len, context_lengths, host_context_lengths,
            kv_cache_block_offsets, host_kv_cache_block_offsets, host_kv_cache_pool_pointers,
            host_kv_cache_pool_mapping, cache_indirection, kv_scale_orig_quant, kv_scale_quant_orig, out_scale,
            rotary_inv_freq, rotary_cos_sin, latent_cache, q_pe, block_ids_per_seq, mrope_rotary_cos_sin,
            mrope_position_deltas, mla_tensor_params, softmax_stats_tensor, spec_decoding_tensor_params,
            attention_sinks, sparse_kv_indices, sparse_kv_offsets, sparse_attn_indices, sparse_attn_offsets);
    }

    TLLM_LOG_TRACE("Attention op stops at layer %d", layer_idx);
}

bool attention_supports_nvfp4_output(int64_t const num_heads, int64_t const num_kv_heads, int64_t const head_size,
    std::optional<int64_t> const tokens_per_block, int64_t const mask_type, int64_t const quant_mode,
    bool const use_paged_context_fmha, bool is_mla_enable)
{
    // Only Blackwell supports NVFP4 output.
    // SM 120 does not support NVFP4 output.
    if (tensorrt_llm::common::getSMVersion() < 100 || tensorrt_llm::common::getSMVersion() == 120)
    {
        return false;
    }

    // MLA is not supported.
    if (is_mla_enable)
    {
        return false;
    }

    auto op = std::make_shared<AttentionOp>();
    op->mType = nvinfer1::DataType::kHALF;
    op->mNumHeads = num_heads;
    op->mNumKVHeads = num_kv_heads;
    op->mHeadSize = head_size;
    op->mMaskType = static_cast<tensorrt_llm::kernels::AttentionMaskType>(int32_t(mask_type));
    op->mKVCacheQuantMode = tensorrt_llm::common::QuantMode(uint32_t(quant_mode));
    op->mFP8ContextFMHA = op->mKVCacheQuantMode.hasFp8KvCache() || op->mKVCacheQuantMode.hasFp4KvCache();
    op->mUseKVCache = true;
    op->mPagedKVCache = true;
    op->mTokensPerBlock = tokens_per_block.value_or(0);
    op->mFuseFp4Quant = true;
    op->mPagedContextFMHA = use_paged_context_fmha;

    auto cache_key = op->data();
    using CacheKey = decltype(cache_key);
    static std::unordered_map<CacheKey, bool, hash<CacheKey>> op_cache;
    if (auto it = op_cache.find(cache_key); it != op_cache.end())
    {
        TLLM_LOG_TRACE("Attention op runtime check is cached");
        return it->second;
    }
    else
    {
        TLLM_LOG_TRACE("Caching attention op runtime check with cache key: %s", to_string(cache_key).c_str());
        op->initialize();
        op_cache[cache_key] = op->supportsNvFp4Output();
    }

    return op->supportsNvFp4Output();
}

} // namespace torch_ext

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def("attention_supports_nvfp4_output", &torch_ext::attention_supports_nvfp4_output);
}
