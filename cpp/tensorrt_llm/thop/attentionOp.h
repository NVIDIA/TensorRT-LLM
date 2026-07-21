/*
 * Copyright (c) 2025-2026, NVIDIA CORPORATION.  All rights reserved.
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

#pragma once

#include <climits>
#include <optional>
#include <torch/extension.h>
#include <tuple>

#include "tensorrt_llm/common/attentionOp.h"
#include "tensorrt_llm/common/config.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/quantization.h"
#include "tensorrt_llm/kernels/kvCacheUtils.h"
#include "tensorrt_llm/kernels/unfusedAttentionKernels.h"

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{

/**
 * @brief Attention operation for TensorRT-LLM
 *
 * This function performs multi-head attention computation in-place, supporting both
 * context and generation phases with various optimization features including:
 * - Fused QKV processing
 * - KV cache management
 * - Multiple position embedding types (RoPE, ALiBi, etc.)
 * - Quantization support (FP8, FP4, etc.)
 * - Multi-layer attention (MLA)
 * - Speculative decoding
 */
void attention(torch::Tensor q, std::optional<torch::Tensor> k, std::optional<torch::Tensor> v, torch::Tensor& output,
    std::optional<torch::Tensor> output_sf, std::optional<torch::Tensor> workspace_, torch::Tensor sequence_length,
    torch::Tensor host_past_key_value_lengths, torch::Tensor host_total_kv_lens, torch::Tensor context_lengths,
    torch::Tensor host_context_lengths, torch::Tensor host_request_types,
    std::optional<int64_t> max_context_q_len_override, std::optional<torch::Tensor> kv_cache_block_offsets,
    std::optional<torch::Tensor> host_kv_cache_pool_pointers, std::optional<torch::Tensor> host_kv_cache_pool_mapping,
    std::optional<torch::Tensor> cache_indirection, std::optional<torch::Tensor> kv_scale_orig_quant,
    std::optional<torch::Tensor> kv_scale_quant_orig, std::optional<torch::Tensor> out_scale,
    std::optional<torch::Tensor> rotary_inv_freq, std::optional<torch::Tensor> rotary_cos_sin,
    std::optional<torch::Tensor> latent_cache, std::optional<torch::Tensor> q_pe,
    std::optional<torch::Tensor> block_ids_per_seq, std::optional<torch::Tensor> attention_sinks,
    bool const is_fused_qkv, bool const update_kv_cache, int64_t const predicted_tokens_per_seq,
    int64_t const local_layer_idx, int64_t const num_heads, int64_t const num_kv_heads, int64_t const head_size,
    std::optional<int64_t> const tokens_per_block, int64_t const max_num_requests, int64_t const max_context_length,
    int64_t const max_seq_len, int64_t const attention_window_size, int64_t const beam_width, int64_t const mask_type,
    int64_t const quant_mode, double const q_scaling, int64_t const position_embedding_type, int64_t const rope_dim,
    double const rope_base, int64_t const rope_scale_type, double const rope_scale, double const rope_short_m_scale,
    double const rope_long_m_scale, int64_t const rope_max_positions, int64_t const rope_original_max_positions,
    bool const use_paged_context_fmha, std::optional<int64_t> attention_input_type, bool is_mla_enable,
    std::optional<int64_t> chunked_prefill_buffer_batch_size, std::optional<int64_t> q_lora_rank,
    std::optional<int64_t> kv_lora_rank, std::optional<int64_t> qk_nope_head_dim,
    std::optional<int64_t> qk_rope_head_dim, std::optional<int64_t> v_head_dim, std::optional<bool> rope_append,
    std::optional<torch::Tensor> mrope_rotary_cos_sin, std::optional<torch::Tensor> mrope_position_deltas,
    std::optional<torch::Tensor> helix_position_offsets, std::optional<torch::Tensor> helix_is_inactive_rank,
    std::optional<int64_t> attention_chunk_size, std::optional<torch::Tensor> softmax_stats_tensor,
    bool const is_spec_decoding_enabled, bool const use_spec_decoding, bool const is_spec_dec_tree,
    std::optional<torch::Tensor> spec_decoding_generation_lengths,
    std::optional<torch::Tensor> spec_decoding_position_offsets_for_cpp,
    std::optional<torch::Tensor> spec_decoding_packed_mask,
    std::optional<torch::Tensor> spec_decoding_bl_tree_mask_offset,
    std::optional<torch::Tensor> spec_decoding_bl_tree_mask,
    std::optional<torch::Tensor> spec_bl_tree_first_sparse_mask_offset_kv,
    std::optional<torch::Tensor> sparse_kv_indices, std::optional<torch::Tensor> sparse_kv_offsets,
    std::optional<torch::Tensor> sparse_attn_indices, std::optional<torch::Tensor> sparse_attn_offsets,
    int64_t const sparse_attn_indices_block_size, std::optional<int64_t> num_sparse_topk,
    std::optional<torch::Tensor> sparse_mla_topk_lens,
    std::optional<double> skip_softmax_threshold_scale_factor_prefill,
    std::optional<double> skip_softmax_threshold_scale_factor_decode, std::optional<torch::Tensor> skip_softmax_stat,
    std::optional<torch::Tensor> cu_q_seqlens, std::optional<torch::Tensor> cu_kv_seqlens,
    std::optional<torch::Tensor> fmha_scheduler_counter, std::optional<torch::Tensor> mla_bmm1_scale,
    std::optional<torch::Tensor> mla_bmm2_scale, std::optional<torch::Tensor> quant_q_buffer,
    std::optional<torch::Tensor> flash_mla_tile_scheduler_metadata = std::nullopt,
    std::optional<torch::Tensor> flash_mla_num_splits = std::nullopt, int64_t sage_attn_num_elts_per_blk_q = 0,
    int64_t sage_attn_num_elts_per_blk_k = 0, int64_t sage_attn_num_elts_per_blk_v = 0, bool sage_attn_qk_int8 = false,
    int64_t num_contexts = 0, int64_t num_ctx_tokens = 0, bool trtllm_gen_jit_warmup = false,
    std::optional<int64_t> compressed_kv_cache_pool_ptr = std::nullopt, bool const is_cross = false,
    std::optional<torch::Tensor> cross_kv = std::nullopt,
    std::optional<torch::Tensor> relative_attention_bias = std::nullopt, int64_t relative_attention_max_distance = 0,
    std::optional<int64_t> spec_decoding_target_max_draft_tokens = std::nullopt,
    std::optional<torch::Tensor> quant_scale_qkv = std::nullopt,
    std::optional<torch::Tensor> dsv4_inv_rope_cos_sin_cache = std::nullopt, bool enable_dsv4_epilogue_fusion = false);

struct KvCachePoolPointers
{
    void* primaryPoolPtr{nullptr};
    void* secondaryPoolPtr{nullptr};
    void* primaryBlockScalePoolPtr{nullptr};
    void* secondaryBlockScalePoolPtr{nullptr};
};

struct KvCachePoolMapping
{
    int32_t poolIndex{0};
    int32_t layerIdxInCachePool{0};
};

KvCachePoolMapping readKvCachePoolMapping(at::Tensor const& hostKvCachePoolMapping, int64_t layerIdx);

KvCachePoolPointers buildKvCachePoolPointers(at::Tensor const& hostKvCachePoolPointers, int32_t poolIndex,
    int64_t intraPoolOffset, int64_t blockSize, int32_t layerIdxInCachePool, int32_t kvFactor, bool isFp4KvCache);

common::op::KvCacheBuffers<kernels::KVBlockArray> buildPagedKvCacheBuffers(
    std::optional<torch::Tensor> const& kv_cache_block_offsets,
    std::optional<torch::Tensor> const& host_kv_cache_pool_pointers,
    std::optional<torch::Tensor> const& host_kv_cache_pool_mapping, common::QuantMode quantMode, int64_t layer_idx,
    int64_t batch_size, int64_t tokens_per_block, int64_t kv_head_num, int64_t size_per_head,
    int64_t cyclic_attention_window_size, int64_t max_attention_window_size, int64_t beam_width, int64_t seq_offset,
    bool is_mla_enable, size_t elem_size);

std::tuple<at::Tensor, std::optional<at::Tensor>> buildFlashinferTrtllmGenPagedKvCacheBuffers(
    at::Tensor host_kv_cache_pool_pointers, at::Tensor host_kv_cache_pool_mapping, int64_t layer_idx,
    int64_t num_kv_heads, int64_t tokens_per_block, int64_t head_dim, int64_t kv_factor, int64_t total_num_blocks,
    int64_t kv_cache_quant_mode, at::ScalarType dtype);

// Layout manager for the thop attention workspace slices used by trtllm-gen.
// Context follows AttentionOp::getWorkspaceSizeForContext() ordering. Generation
// follows the XQA workspace ordering used by AttentionOp generation.
struct TrtllmGenContextWorkspaceLayout
{
    int64_t trtllmGenWorkspaceOffset{};
    int64_t cuQSeqlensOffset{};
    int64_t cuKvSeqlensOffset{};
    int64_t cuMaskRowsOffset{};
    int64_t rotaryInvFreqOffset{};
    int64_t qBufOffset{};
    int64_t tokensInfoOffset{};
    int64_t fmhaTileCounterOffset{};
    int64_t fmhaBmm1ScaleOffset{};
    int64_t fmhaBmm2ScaleOffset{};
    int64_t trtllmGenWorkspaceSize{};
    int64_t cuSeqlensSize{};
    int64_t rotaryInvFreqSize{};
    int64_t qBufSize{};
    int64_t tokensInfoSize{};
    int64_t fmhaTileCounterSize{};
    int64_t fmhaBmm1ScaleSize{};
    int64_t fmhaBmm2ScaleSize{};
    int64_t totalSize{};
    at::ScalarType qBufScalarType{};
};

struct TrtllmGenGenerationWorkspaceLayout
{
    int64_t trtllmGenWorkspaceOffset{};
    int64_t cuSeqlensOffset{};
    int64_t cuKvSeqlensOffset{};
    int64_t rotaryInvFreqOffset{};
    int64_t tokensInfoOffset{};
    int64_t qBufOffset{};
    int64_t bmm1ScaleOffset{};
    int64_t bmm2ScaleOffset{};
    int64_t sparseAttnCacheOffset{};
    int64_t trtllmGenWorkspaceSize{};
    int64_t cuSeqlensSize{};
    int64_t cuKvSeqlensSize{};
    int64_t rotaryInvFreqSize{};
    int64_t tokensInfoSize{};
    int64_t qBufSize{};
    int64_t bmm1ScaleSize{};
    int64_t bmm2ScaleSize{};
    int64_t sparseAttnCacheSize{};
    int64_t totalSize{};
    at::ScalarType qBufScalarType{};
};

struct TrtllmGenContextWorkspaceViews
{
    at::Tensor trtllmGenWorkspace;
    at::Tensor cuQSeqlens;
    at::Tensor cuKvSeqlens;
    at::Tensor cuMaskRows;
    std::optional<at::Tensor> rotaryInvFreqBuf;
    std::optional<at::Tensor> qBuf;
    at::Tensor tokensInfo;
    at::Tensor fmhaTileCounter;
    std::optional<at::Tensor> fmhaBmm1Scale;
    std::optional<at::Tensor> fmhaBmm2Scale;
};

struct TrtllmGenGenerationWorkspaceViews
{
    at::Tensor trtllmGenWorkspace;
    at::Tensor cuSeqlens;
    at::Tensor cuKvSeqlens;
    std::optional<at::Tensor> rotaryInvFreqBuf;
    at::Tensor tokensInfo;
    at::Tensor qBuf;
    at::Tensor bmm1Scale;
    at::Tensor bmm2Scale;
    std::optional<at::Tensor> sparseAttnCache;
};

class TrtllmAttentionWorkspaceManager
{
public:
    static constexpr int64_t kWorkspaceAlignment = 256;
    static constexpr int64_t kTrtllmGenWorkspaceSize = CUBLAS_WORKSPACE_SIZE;

    static TrtllmGenContextWorkspaceLayout buildContextLayout(at::ScalarType qDtype, int64_t batchSize,
        int64_t numTokens, int64_t numHeads, int64_t headSize, int64_t rotaryEmbeddingDim, bool separateQKvInput,
        bool fp8ContextFmha);

    static TrtllmGenGenerationWorkspaceLayout buildGenerationLayout(at::ScalarType qDtype, int64_t batchBeam,
        int64_t numTokens, int64_t numHeads, int64_t headSize, int64_t rotaryEmbeddingDim, int64_t numKvHeads,
        int64_t maxBlocksPerSequence, bool useSparseAttention);

    static int64_t getContextWorkspaceSize(at::ScalarType qDtype, int64_t batchSize, int64_t numTokens,
        int64_t numHeads, int64_t headSize, int64_t rotaryEmbeddingDim, bool separateQKvInput, bool fp8ContextFmha);

    //! numKvHeads and maxBlocksPerSequence affect the size only when sparse attention is enabled.
    static int64_t getGenerationWorkspaceSize(at::ScalarType qDtype, int64_t batchBeam, int64_t numTokens,
        int64_t numHeads, int64_t headSize, int64_t rotaryEmbeddingDim, int64_t numKvHeads,
        int64_t maxBlocksPerSequence, bool useSparseAttention);

    static TrtllmGenContextWorkspaceViews materializeContextWorkspace(
        at::Tensor const& workspace, TrtllmGenContextWorkspaceLayout const& layout);

    static TrtllmGenContextWorkspaceViews materializeContextWorkspace(at::Tensor const& workspace,
        at::ScalarType qDtype, int64_t batchSize, int64_t numTokens, int64_t numHeads, int64_t headSize,
        int64_t rotaryEmbeddingDim, bool fp8ContextFmha);

    static TrtllmGenGenerationWorkspaceViews materializeGenerationWorkspace(
        at::Tensor const& workspace, TrtllmGenGenerationWorkspaceLayout const& layout);

    static TrtllmGenGenerationWorkspaceViews materializeGenerationWorkspace(at::Tensor const& workspace,
        at::ScalarType qDtype, int64_t batchBeam, int64_t numTokens, int64_t numHeads, int64_t headSize,
        int64_t rotaryEmbeddingDim, int64_t numKvHeads);

private:
    static std::optional<at::Tensor> makeWorkspaceView(
        at::Tensor const& workspace, int64_t offset, int64_t sizeBytes, at::ScalarType scalarType);
};

} // namespace torch_ext

/**
 * @brief Compute FlashMLA tile-scheduler metadata in-place.
 *
 * Call once per forward pass before the attention layers to pre-compute
 * get_mla_metadata and store the results in the provided tensors. Pass
 * these tensors to the attention op so all layers reuse the same metadata.
 */
void computeFlashMlaMetadata(torch::Tensor seqlens_k, torch::Tensor tile_scheduler_metadata, torch::Tensor num_splits,
    int64_t batch_size, int64_t s_q, int64_t num_q_heads, int64_t num_kv_heads, int64_t head_size_v);

TRTLLM_NAMESPACE_END
