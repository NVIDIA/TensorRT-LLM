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
#include "tensorrt_llm/runtime/torchUtils.h"
#include "tensorrt_llm/runtime/utils/debugUtils.h"
#include "tensorrt_llm/thop/thUtils.h"
#include <functional>
#include <torch/extension.h>
#include <unordered_set>

using tensorrt_llm::common::op::AttentionOp;
using tensorrt_llm::common::op::TupleHash;
using tensorrt_llm::kernels::KVBlockArray;
using tensorrt_llm::runtime::RequestType;
using tensorrt_llm::kernels::mlaParams;

namespace torch_ext
{

namespace trtllm::attention
{

class RunnerBase
{
public:
    int64_t beam_width;
    int64_t max_num_requests;
    int64_t attention_window_size;
    int64_t sink_token_length;

    virtual ~RunnerBase() = default;
    virtual void prepare(AttentionOp& op) const = 0;
    virtual int64_t getWorkspaceSize(
        AttentionOp const& op, int const num_tokens, int const max_attention_window_size) const
        = 0;
    virtual void run(AttentionOp& op, bool const is_context, int32_t const seq_offset, int32_t const num_seqs,
        int32_t const token_offset, int32_t const num_tokens, torch::Tensor workspace, torch::Tensor output,
        torch::Tensor qkv, torch::Tensor sequence_length, torch::Tensor host_past_key_value_lengths,
        torch::Tensor context_lengths, torch::Tensor host_context_lengths, torch::Tensor kv_cache_block_offsets,
        torch::Tensor host_kv_cache_block_offsets, torch::Tensor host_kv_cache_pool_pointers,
        torch::Tensor host_kv_cache_pool_mapping, torch::optional<torch::Tensor> cache_indirection,
        torch::optional<torch::Tensor> kv_scale_orig_quant, torch::optional<torch::Tensor> kv_scale_quant_orig,
        torch::optional<torch::Tensor> out_scale, torch::optional<torch::Tensor> rotary_inv_freq,
        torch::optional<torch::Tensor> rotary_cos_sin, torch::optional<torch::Tensor> q_b_proj,
        torch::optional<torch::Tensor> kv_b_proj, torch::optional<torch::Tensor> k_b_proj_trans,
        torch::optional<torch::Tensor> q_b_proj_scale, torch::optional<torch::Tensor> kv_b_proj_scale,
        torch::optional<torch::Tensor> k_b_proj_trans_scale) const
        = 0;
};

template <typename T, typename AttentionOutT = T>
class Runner : public RunnerBase
{
public:
    void prepare(AttentionOp& op) const override
    {
        AttentionOp::EnqueueGenerationParams<T> enqueueParams;
        enqueueParams.beam_width = beam_width;
        enqueueParams.max_attention_window = attention_window_size;
        enqueueParams.cyclic_attention_window_size = attention_window_size;
        enqueueParams.max_cyclic_attention_window_size = attention_window_size;
        enqueueParams.sink_token_length = sink_token_length;
        enqueueParams.num_requests = max_num_requests;

        op.prepareEnqueueGeneration<T, KVBlockArray>(enqueueParams);

        // Always reserve SemaphoreArray (for multi-block mode) as MMHA may enable multi-block mode when shared memory
        // is not enough.
        op.reserveSemaphoreArray(op.mNumHeads * max_num_requests);
    }

    int64_t getWorkspaceSize(
        AttentionOp const& op, int const num_tokens, int const max_attention_window_size) const override
    {
        size_t const context_workspace_size
            = op.getWorkspaceSizeForContext(op.mType, max_num_requests, op.mMaxContextLength, num_tokens);
        size_t const generation_workspace_size
            = op.getWorkspaceSizeForGeneration(op.mType, max_num_requests, max_attention_window_size, num_tokens);

        size_t attention_input_workspace_size = 0;
        size_t context_mla_fp8_quant_size = 0;
        if (op.isMLAEnabled())
        {
            int32_t const size_per_head
                = 2 * (op.mMLAParams.qk_nope_head_dim + op.mMLAParams.qk_rope_head_dim) + op.mMLAParams.v_head_dim;
            size_t const size = tensorrt_llm::runtime::BufferDataType(op.mType).getSize();
            size_t const attention_input_size = size * num_tokens * op.mNumHeads
                * std::max(size_per_head, op.mMLAParams.kv_lora_rank + op.mMLAParams.qk_rope_head_dim);
            size_t workspaces[1];
            workspaces[0] = attention_input_size;
            attention_input_workspace_size = tensorrt_llm::common::calculateTotalWorkspaceSize(workspaces, 1);
            if (op.mIsFP8BlockScalingEnabled)
            {
                size_t act_quant_size = op.mGemmRunner->getActWorkspaceSize(num_tokens,
                    op.mMLAParams.qk_nope_head_dim + op.mMLAParams.kv_lora_rank + op.mMLAParams.qk_rope_head_dim);
                size_t workspaces[2];
                workspaces[0] = act_quant_size;
                context_mla_fp8_quant_size = tensorrt_llm::common::calculateTotalWorkspaceSize(workspaces, 1);
            }
        }

        return std::max(context_mla_fp8_quant_size, std::max(context_workspace_size, generation_workspace_size))
            + attention_input_workspace_size;
    }

    void run(AttentionOp& op, bool const is_context, int32_t const seq_offset, int32_t const num_seqs,
        int32_t const token_offset, int32_t const num_tokens, torch::Tensor workspace, torch::Tensor output,
        torch::Tensor qkv, torch::Tensor sequence_length, torch::Tensor host_past_key_value_lengths,
        torch::Tensor context_lengths, torch::Tensor host_context_lengths, torch::Tensor kv_cache_block_offsets,
        torch::Tensor host_kv_cache_block_offsets, torch::Tensor host_kv_cache_pool_pointers,
        torch::Tensor host_kv_cache_pool_mapping, torch::optional<torch::Tensor> cache_indirection,
        torch::optional<torch::Tensor> kv_scale_orig_quant, torch::optional<torch::Tensor> kv_scale_quant_orig,
        torch::optional<torch::Tensor> out_scale, torch::optional<torch::Tensor> rotary_inv_freq,
        torch::optional<torch::Tensor> rotary_cos_sin, torch::optional<torch::Tensor> q_b_proj,
        torch::optional<torch::Tensor> kv_b_proj, torch::optional<torch::Tensor> k_b_proj_trans,
        torch::optional<torch::Tensor> q_b_proj_scale, torch::optional<torch::Tensor> kv_b_proj_scale,
        torch::optional<torch::Tensor> k_b_proj_trans_scale) const override
    {
        auto stream = at::cuda::getCurrentCUDAStream(qkv.get_device());
        T const* attention_input = static_cast<T const*>(qkv.slice(0, token_offset).data_ptr());
        AttentionOutT* context_buf = static_cast<AttentionOutT*>(output.slice(0, token_offset).data_ptr());

        // Rotary inv_freq, cos_sin cache to avoid re-computing.
        float const* rotary_inv_freq_ptr = nullptr;
        float2 const* rotary_cos_sin_ptr = nullptr;

        if (op.isRoPE())
        {
            rotary_inv_freq_ptr = rotary_inv_freq.value().data_ptr<float>();
        }

        if (op.isRoPE() || op.isMLAEnabled())
        {
            rotary_cos_sin_ptr = static_cast<float2 const*>(rotary_cos_sin.value().data_ptr());
        }

        void* workspace_ptr = workspace.data_ptr();
        [[maybe_unused]] mlaParams<T> mla_params;
        if (op.isMLAEnabled())
        {
            // In MLA, attention_input will be the ptr of workspace
            auto const* q_b_proj_ptr = static_cast<T const*>(q_b_proj.value().data_ptr());
            auto const* kv_b_proj_ptr = static_cast<T const*>(kv_b_proj.value().data_ptr());
            auto const* k_b_proj_trans_ptr = static_cast<T const*>(k_b_proj_trans.value().data_ptr());

            mla_params.fused_a_input = attention_input;
            mla_params.context_buf = reinterpret_cast<T*>(context_buf);
            mla_params.q_b_proj = q_b_proj_ptr;
            mla_params.kv_b_proj = kv_b_proj_ptr;
            mla_params.k_b_proj_trans = k_b_proj_trans_ptr;
            mla_params.cos_sin_cache = rotary_cos_sin_ptr;
            mla_params.batch_size = num_seqs;
            mla_params.acc_q_len = num_tokens;
            mla_params.head_num = op.mNumHeads;
            mla_params.meta = op.mMLAParams;
            if (op.mIsFP8BlockScalingEnabled)
            {
                mla_params.q_b_scale = static_cast<float const*>(q_b_proj_scale.value().data_ptr());
                mla_params.kv_b_scale = static_cast<float const*>(kv_b_proj_scale.value().data_ptr());
                mla_params.k_b_trans_scale = static_cast<float const*>(k_b_proj_trans_scale.value().data_ptr());
            }

            size_t const size_per_head = is_context
                ? (2 * (op.mMLAParams.qk_nope_head_dim + op.mMLAParams.qk_rope_head_dim) + op.mMLAParams.v_head_dim)
                : op.mMLAParams.kv_lora_rank + op.mMLAParams.qk_rope_head_dim;
            size_t const total_size = sizeof(T) * mla_params.acc_q_len * op.mNumHeads * size_per_head;
            int8_t* workspace_byte_ptr = reinterpret_cast<int8_t*>(workspace_ptr);
            size_t offset = 0;
            T* attention_input_qkv = reinterpret_cast<T*>(nextWorkspacePtr(workspace_byte_ptr, offset, total_size));
            workspace_ptr = reinterpret_cast<void*>(workspace_byte_ptr + offset);
            mla_params.attention_input_buf = attention_input_qkv;
            mla_params.workspace = workspace_ptr;
            attention_input = attention_input_qkv;
        }

        int const* q_seq_lengths = context_lengths.slice(0, seq_offset).data_ptr<int>();
        int const* kv_seq_lengths = sequence_length.slice(0, seq_offset).data_ptr<int>();
        // Note we still need context length during generation for MMHA optimization.
        int32_t const max_context_q_len
            = host_context_lengths.slice(0, seq_offset, seq_offset + num_seqs).max().item<int32_t>();
        int32_t const max_past_kv_len
            = host_past_key_value_lengths.slice(0, seq_offset, seq_offset + num_seqs).max().item<int32_t>();

        // Commonly, cyclic_attention_window_size, and max_attention_window_size will be the same
        // unless each layer has different attention window sizes.
        // the kv_cache capacity.
        int const max_attention_window_size
            = beam_width == 1 ? attention_window_size : cache_indirection.value().size(2);
        // The cyclic_attention_window_size will determine the cyclic kv cache position of new tokens.
        // Note that this cyclic_attention_window_size might be smaller than the actual kv cache capactity.
        int const cyclic_attention_window_size = attention_window_size;
        bool const can_use_one_more_block = beam_width > 1;

        int max_blocks_per_sequence = kv_cache_block_offsets.size(-1);
        int32_t const pool_index = host_kv_cache_pool_mapping.index({op.mLayerIdx}).item<int32_t>();
        int32_t const layer_idx_in_cache_pool
            = host_kv_cache_pool_mapping.slice(0, 0, op.mLayerIdx).eq(pool_index).sum().item<int32_t>();
        op.mLayerIdxInCachePool = layer_idx_in_cache_pool;
        KVBlockArray::DataType* block_offsets
            = static_cast<KVBlockArray::DataType*>(kv_cache_block_offsets.index({pool_index, seq_offset}).data_ptr());
        KVBlockArray::DataType* host_block_offsets = static_cast<KVBlockArray::DataType*>(
            host_kv_cache_block_offsets.index({pool_index, seq_offset}).data_ptr());

        auto const cache_elem_size = (op.mKVCacheQuantMode.hasKvCacheQuant() ? 1 : sizeof(T));
        auto const block_size = op.mTokensPerBlock * op.mNumKVHeads * op.mHeadSize;
        auto const bytes_per_block = block_size * cache_elem_size;
        auto const intra_pool_offset = op.mLayerIdxInCachePool * 2 * bytes_per_block;

        void* host_primary_pool_pointer = reinterpret_cast<void*>(
            reinterpret_cast<char*>(host_kv_cache_pool_pointers.index({pool_index, 0}).item<int64_t>())
            + intra_pool_offset);
        void* host_secondary_pool_pointer = reinterpret_cast<void*>(
            reinterpret_cast<char*>(host_kv_cache_pool_pointers.index({pool_index, 1}).item<int64_t>())
            + intra_pool_offset);

        float const* kv_scale_orig_quant_ptr = nullptr;
        float const* kv_scale_quant_orig_ptr = nullptr;
        if (op.mKVCacheQuantMode.hasKvCacheQuant())
        {
            kv_scale_orig_quant_ptr = kv_scale_orig_quant.value().data_ptr<float>();
            kv_scale_quant_orig_ptr = kv_scale_quant_orig.value().data_ptr<float>();
        }
        float const* out_scale_ptr = op.mFP8ContextFMHA ? out_scale.value().data_ptr<float>() : nullptr;

        if (is_context) // context stage
        {
            AttentionOp::EnqueueContextParams<T> enqueue_params;
            enqueue_params.attention_input = attention_input;
            enqueue_params.rotary_inv_freq = rotary_inv_freq_ptr;
            enqueue_params.rotary_cos_sin = rotary_cos_sin_ptr;
            enqueue_params.input_seq_length = max_context_q_len;
            enqueue_params.max_past_kv_len = max_past_kv_len;
            enqueue_params.max_attention_window = max_attention_window_size;
            enqueue_params.cyclic_attention_window_size = cyclic_attention_window_size;
            enqueue_params.max_cyclic_attention_window_size = cyclic_attention_window_size;
            enqueue_params.can_use_one_more_block = can_use_one_more_block;
            enqueue_params.sink_token_length = sink_token_length;
            enqueue_params.q_seq_lengths = q_seq_lengths;
            enqueue_params.kv_seq_lengths = kv_seq_lengths;
            enqueue_params.context_buf = context_buf;
            enqueue_params.block_offsets = block_offsets;
            enqueue_params.host_block_offsets = host_block_offsets;
            enqueue_params.host_primary_pool_pointer = host_primary_pool_pointer;
            enqueue_params.host_secondary_pool_pointer = host_secondary_pool_pointer;
            enqueue_params.batch_size = num_seqs;
            enqueue_params.num_tokens = num_tokens;
            enqueue_params.max_blocks_per_sequence = max_blocks_per_sequence;
            enqueue_params.host_context_lengths = host_context_lengths.data_ptr<int32_t>();
            enqueue_params.workspace = workspace_ptr;
            enqueue_params.kv_scale_orig_quant = kv_scale_orig_quant_ptr;
            enqueue_params.kv_scale_quant_orig = kv_scale_quant_orig_ptr;
            enqueue_params.attention_output_orig_quant = out_scale_ptr;

            if (op.isMLAEnabled())
            {
                mla_params.cache_seq_lens = kv_seq_lengths;
                mla_params.max_input_seq_len = max_context_q_len;
                op.mlaPreContext<T>(mla_params, stream);
                enqueue_params.mla_param = &mla_params;
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

            AttentionOp::EnqueueGenerationParams<T> enqueue_params;
            enqueue_params.attention_input = attention_input;
            enqueue_params.rotary_inv_freq = rotary_inv_freq_ptr;
            enqueue_params.rotary_cos_sin = rotary_cos_sin_ptr;
            enqueue_params.input_seq_length = input_seq_length;
            enqueue_params.sequence_lengths = kv_seq_lengths;
            enqueue_params.max_past_kv_length = max_past_kv_len;
            enqueue_params.beam_width = beam_width;
            enqueue_params.context_lengths = q_seq_lengths;
            enqueue_params.context_buf = context_buf;
            enqueue_params.block_offsets = block_offsets;
            enqueue_params.host_primary_pool_pointer = host_primary_pool_pointer;
            enqueue_params.host_secondary_pool_pointer = host_secondary_pool_pointer;
            enqueue_params.max_attention_window = max_attention_window_size;
            enqueue_params.cyclic_attention_window_size = cyclic_attention_window_size;
            enqueue_params.max_cyclic_attention_window_size = cyclic_attention_window_size;
            enqueue_params.can_use_one_more_block = can_use_one_more_block;
            enqueue_params.sink_token_length = sink_token_length;
            enqueue_params.num_requests = num_requests;
            enqueue_params.max_blocks_per_sequence = max_blocks_per_sequence;
            enqueue_params.cache_indir = beam_width == 1 ? nullptr : cache_indirection.value().data_ptr<int32_t>();
            enqueue_params.semaphores = op.mMultiBlockSemaphores.get();
            enqueue_params.workspace = workspace_ptr;
            enqueue_params.host_past_key_value_lengths = host_past_key_value_lengths.data_ptr<int32_t>();
            enqueue_params.host_context_lengths = host_context_lengths.data_ptr<int32_t>();
            enqueue_params.total_num_input_tokens = num_tokens;
            enqueue_params.kv_scale_orig_quant = kv_scale_orig_quant_ptr;
            enqueue_params.kv_scale_quant_orig = kv_scale_quant_orig_ptr;
            enqueue_params.attention_output_orig_quant = out_scale_ptr;

            // Current mlaGeneration will using fmha to do attention, so we don't go into enqueueGeneration
            if (op.isMLAEnabled())
            {
                mla_params.cache_seq_lens = kv_seq_lengths;
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

torch::Tensor attention(torch::Tensor q, torch::optional<torch::Tensor> k, torch::optional<torch::Tensor> v,
    std::optional<torch::ScalarType> out_dtype, torch::optional<torch::Tensor> workspace_,
    torch::Tensor sequence_length, torch::Tensor host_past_key_value_lengths, torch::Tensor context_lengths,
    torch::Tensor host_context_lengths, torch::Tensor host_request_types, torch::Tensor kv_cache_block_offsets,
    torch::Tensor host_kv_cache_block_offsets, torch::Tensor host_kv_cache_pool_pointers,
    torch::Tensor host_kv_cache_pool_mapping, torch::optional<torch::Tensor> cache_indirection,
    torch::optional<torch::Tensor> kv_scale_orig_quant, torch::optional<torch::Tensor> kv_scale_quant_orig,
    torch::optional<torch::Tensor> out_scale, torch::optional<torch::Tensor> rotary_inv_freq,
    torch::optional<torch::Tensor> rotary_cos_sin, torch::optional<torch::Tensor> q_b_proj,
    torch::optional<torch::Tensor> kv_b_proj, torch::optional<torch::Tensor> k_b_proj_trans,
    torch::optional<torch::Tensor> q_b_proj_scale, torch::optional<torch::Tensor> kv_b_proj_scale,
    torch::optional<torch::Tensor> k_b_proj_trans_scale, bool const is_fused_qkv, bool const update_kv_cache,
    int64_t const layer_idx, int64_t const num_heads, int64_t const num_kv_heads, int64_t const head_size,
    int64_t const tokens_per_block, int64_t const max_num_requests, int64_t const max_context_length,
    int64_t const attention_window_size, int64_t const sink_token_length, int64_t const beam_width,
    int64_t const mask_type, int64_t const quant_mode, int64_t const position_embedding_type,
    int64_t const rotary_embedding_dim, double const rotary_embedding_base, int64_t const rotary_embedding_scale_type,
    double const rotary_embedding_scale, double const rotary_embedding_short_m_scale,
    double const rotary_embedding_long_m_scale, int64_t const rotary_embedding_max_positions,
    int64_t const rotary_embedding_original_max_positions, bool const use_paged_context_fmha,
    std::optional<bool> is_mla_enable, std::optional<int64_t> q_lora_rank, std::optional<int64_t> kv_lora_rank,
    std::optional<int64_t> qk_nope_head_dim, std::optional<int64_t> qk_rope_head_dim, std::optional<int64_t> v_head_dim,
    bool is_fp8_block_scaling_enabled)
{
    TLLM_LOG_TRACE("Attention op starts at layer %d", layer_idx);

    TLLM_CHECK_WITH_INFO(is_fused_qkv, "Only fused QKV is supported now");
    TLLM_CHECK_WITH_INFO(update_kv_cache, "KV cache update cannot be disabled now");
    auto qkv = q;
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

    auto const dtype = tensorrt_llm::runtime::TorchUtils::dataType(qkv.scalar_type());
    bool const is_fp8_out = out_dtype.has_value() && out_dtype.value() == torch::kFloat8_e4m3fn;

    RunnerPtr runner;
    if (dtype == nvinfer1::DataType::kHALF)
    {
        if (is_fp8_out)
        {
            runner.reset(new Runner<half, __nv_fp8_e4m3>());
        }
        else
        {
            TLLM_CHECK(!out_dtype.has_value() || out_dtype.value() == torch::kFloat16);
            runner.reset(new Runner<half>());
        }
    }
    else if (dtype == nvinfer1::DataType::kFLOAT)
    {
        TLLM_CHECK(!out_dtype.has_value() || out_dtype.value() == torch::kFloat32);
        runner.reset(new Runner<float>());
    }
#ifdef ENABLE_BF16
    else if (dtype == nvinfer1::DataType::kBF16)
    {
        if (is_fp8_out)
        {
            runner.reset(new Runner<__nv_bfloat16, __nv_fp8_e4m3>());
        }
        else
        {
            TLLM_CHECK(!out_dtype.has_value() || out_dtype.value() == torch::kBFloat16);
            runner.reset(new Runner<__nv_bfloat16>());
        }
    }
#endif
    runner->beam_width = beam_width;
    runner->max_num_requests = max_num_requests;
    runner->attention_window_size = attention_window_size;
    runner->sink_token_length = sink_token_length;

    auto cache_key = std::make_tuple(dtype, out_dtype, layer_idx, num_heads, num_kv_heads, head_size, mask_type,
        quant_mode, tokens_per_block, max_context_length, position_embedding_type, rotary_embedding_dim,
        rotary_embedding_base, rotary_embedding_scale_type, rotary_embedding_scale, rotary_embedding_short_m_scale,
        rotary_embedding_long_m_scale, rotary_embedding_max_positions, rotary_embedding_original_max_positions,
        beam_width, max_num_requests, attention_window_size, sink_token_length, use_paged_context_fmha,
        is_fp8_block_scaling_enabled);
    using CacheKey = decltype(cache_key);
    static std::unordered_map<CacheKey, std::shared_ptr<AttentionOp>, TupleHash<CacheKey>> op_cache;
    std::shared_ptr<AttentionOp> op;
    if (auto it = op_cache.find(cache_key); it != op_cache.end())
    {
        TLLM_LOG_TRACE("Attention op for layer %d is cached", layer_idx);
        op = it->second;
    }
    else
    {
        TLLM_LOG_TRACE("Creating new attention op for layer %d", layer_idx);
        op = std::make_shared<AttentionOp>();

        op->mType = dtype;
        op->mFMHAForceFP32Acc = dtype == nvinfer1::DataType::kBF16;
        op->mFP8ContextFMHA = is_fp8_out;
        op->mLayerIdx = layer_idx;
        op->mNumHeads = num_heads;
        op->mNumKVHeads = num_kv_heads;
        op->mHeadSize = head_size;
        op->mMaskType = static_cast<tensorrt_llm::kernels::AttentionMaskType>(int32_t(mask_type));
        op->mKVCacheQuantMode = tensorrt_llm::common::QuantMode(uint32_t(quant_mode));
        op->mTokensPerBlock = tokens_per_block;
        op->mMaxContextLength = max_context_length;
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
        op->mPagedContextFMHA = use_paged_context_fmha;

        op->mIsFP8BlockScalingEnabled = is_fp8_block_scaling_enabled;

        if (is_mla_enable.has_value() && is_mla_enable.value())
        {
            op->mIsMLAEnabled = true;
            op->mMLAParams = {static_cast<int>(q_lora_rank.value()), static_cast<int>(kv_lora_rank.value()),
                static_cast<int>(qk_nope_head_dim.value()), static_cast<int>(qk_rope_head_dim.value()),
                static_cast<int>(v_head_dim.value())};
        }
        op->initialize();

        runner->prepare(*op);

        op_cache[cache_key] = op;
    }

    int32_t const num_seqs = host_context_lengths.size(0);
    RequestType const* request_types = static_cast<RequestType const*>(host_request_types.data_ptr());
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
    int32_t const num_tokens = qkv.size(0);
    int32_t const num_ctx_tokens = host_context_lengths.slice(0, 0, num_contexts).sum().item<int32_t>();
    int32_t const num_gen_tokens = num_tokens - num_ctx_tokens;
    for (int32_t idx = num_contexts; idx < num_seqs; idx++)
    {
        TLLM_CHECK(request_types[idx] == RequestType::kGENERATION);
    }

    int32_t const max_attention_window_size
        = beam_width == 1 ? attention_window_size : cache_indirection.value().size(2);
    int64_t const workspace_size = runner->getWorkspaceSize(*op, num_tokens, max_attention_window_size);
    TLLM_LOG_TRACE("Expected workspace size is %ld bytes", workspace_size);
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
        workspace = torch::empty({workspace_size}, torch::dtype(torch::kByte).device(qkv.device()));
    }
    auto output = torch::empty({num_tokens, op->mIsMLAEnabled ? num_heads * v_head_dim.value() : num_heads * head_size},
        qkv.options().dtype(out_dtype.value_or(qkv.scalar_type())));
    if (num_contexts > 0)
    {
        auto seq_offset = 0;
        auto token_offset = 0;
        runner->run(*op,
            /*is_context=*/true, seq_offset,
            /*num_seqs=*/num_contexts, token_offset,
            /*num_tokens=*/num_ctx_tokens, workspace, output, qkv, sequence_length, host_past_key_value_lengths,
            context_lengths, host_context_lengths, kv_cache_block_offsets, host_kv_cache_block_offsets,
            host_kv_cache_pool_pointers, host_kv_cache_pool_mapping, cache_indirection, kv_scale_orig_quant,
            kv_scale_quant_orig, out_scale, rotary_inv_freq, rotary_cos_sin, q_b_proj, kv_b_proj, k_b_proj_trans,
            q_b_proj_scale, kv_b_proj_scale, k_b_proj_trans_scale);
    }

    if (num_generations > 0)
    {
        auto seq_offset = num_contexts;
        auto token_offset = num_ctx_tokens;
        runner->run(*op,
            /*is_context=*/false, seq_offset,
            /*num_seqs=*/num_generations, token_offset,
            /*num_tokens=*/num_gen_tokens, workspace, output, qkv, sequence_length, host_past_key_value_lengths,
            context_lengths, host_context_lengths, kv_cache_block_offsets, host_kv_cache_block_offsets,
            host_kv_cache_pool_pointers, host_kv_cache_pool_mapping, cache_indirection, kv_scale_orig_quant,
            kv_scale_quant_orig, out_scale, rotary_inv_freq, rotary_cos_sin, q_b_proj, kv_b_proj, k_b_proj_trans,
            q_b_proj_scale, kv_b_proj_scale, k_b_proj_trans_scale);
    }

    sync_check_cuda_error();
    TLLM_LOG_TRACE("Attention op stops at layer %d", layer_idx);

    return output;
}

} // namespace torch_ext

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "attention("
        "Tensor q"
        ", Tensor? k"
        ", Tensor? v"
        ", ScalarType? out_dtype"
        ", Tensor? workspace"
        ", Tensor sequence_length"
        ", Tensor host_past_key_value_lengths"
        ", Tensor context_lengths"
        ", Tensor host_context_lengths"
        ", Tensor host_request_types"
        ", Tensor kv_cache_block_offsets"
        ", Tensor host_kv_cache_block_offsets"
        ", Tensor host_kv_cache_pool_pointers"
        ", Tensor host_kv_cache_pool_mapping"
        ", Tensor? cache_indirection"
        ", Tensor? kv_scale_orig_quant"
        ", Tensor? kv_scale_quant_orig"
        ", Tensor? out_scale"
        ", Tensor? rotary_inv_freq"
        ", Tensor? rotary_cos_sin"
        ", Tensor? q_b_proj"
        ", Tensor? kv_b_proj"
        ", Tensor? k_b_proj_trans"
        ", Tensor? q_b_proj_scale"
        ", Tensor? kv_b_proj_scale"
        ", Tensor? k_b_proj_trans_scale"
        ", bool is_fused_qkv"
        ", bool update_kv_cache"
        ", int layer_idx"
        ", int num_heads"
        ", int num_kv_heads"
        ", int head_size"
        ", int tokens_per_block"
        ", int max_num_requests"
        ", int max_context_length"
        ", int attention_window_size"
        ", int sink_token_length"
        ", int beam_width"
        ", int mask_type"
        ", int quant_mode"
        ", int position_embedding_type"
        ", int rotary_embedding_dim"
        ", float rotary_embedding_base"
        ", int rotary_embedding_scale_type"
        ", float rotary_embedding_scale"
        ", float rotary_embedding_short_m_scale"
        ", float rotary_embedding_long_m_scale"
        ", int rotary_embedding_max_positions"
        ", int rotary_embedding_original_max_positions"
        ", bool use_paged_context_fmha"
        ", bool? is_mla_enable"
        ", int? q_lora_rank"
        ", int? kv_lora_rank"
        ", int? qk_nope_head_dim"
        ", int? qk_rope_head_dim"
        ", int? v_head_dim"
        ", bool is_fp8_block_scaling_enabled"
        ") -> Tensor");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("attention", &torch_ext::attention);
}
