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

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/config.h"
#include "tensorrt_llm/common/cublasMMWrapper.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/quantization.h"
#include "tensorrt_llm/kernels/fmhaDispatcher.h"
#include "tensorrt_llm/kernels/gptKernels.h"
#include "tensorrt_llm/kernels/kvCacheUtils.h"
#include "tensorrt_llm/kernels/mlaKernels.h"
#include "tensorrt_llm/kernels/sparseAttentionKernels.h"
#include "tensorrt_llm/kernels/xqaDispatcher.h"
#include "tensorrt_llm/runtime/torchUtils.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGraphsC10Utils.h>
#include <torch/extension.h>

#include <cassert>
#include <concepts>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <tuple>
#include <unordered_map>

#if ENABLE_MULTI_DEVICE
#include <nccl.h>
#endif // ENABLE_MULTI_DEVICE

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{

struct KvCachePoolPointers
{
    void* primaryPoolPtr{nullptr};
    void* secondaryPoolPtr{nullptr};
    void* primaryBlockScalePoolPtr{nullptr};
    void* secondaryBlockScalePoolPtr{nullptr};
};

#define TRTLLM_FMHA_PARAM_FIELD(name, cpp_type) cpp_type name{};

/// Rotary embedding parameters, mirrored from the Python RopeParams.
struct AttentionRopeParams
{
#include "tensorrt_llm/thop/attention_rope_params_fields.inc"
};

/// Layer state read from the Python TrtllmAttention. Members are the ones this file
/// reads as `attn.<name>`; the generator emits exactly that set.
struct AttentionLayerParams
{
#include "tensorrt_llm/thop/attention_layer_params_fields.inc"
};

/// Batch state read from the Python TrtllmAttentionMetadata, selected the same way.
struct AttentionMetadataParams
{
#include "tensorrt_llm/thop/attention_metadata_params_fields.inc"
};

/// Sparse inputs from the attention module to its backend.
struct SparseBackendForwardArgs
{
#include "tensorrt_llm/thop/sparse_backend_forward_args_accessors.inc"
#include "tensorrt_llm/thop/sparse_backend_forward_args_fields.inc"
};

/// Sparse inputs from the backend to the attention op.
struct SparseRuntimeParams
{
#include "tensorrt_llm/thop/sparse_runtime_params_accessors.inc"
#include "tensorrt_llm/thop/sparse_runtime_params_fields.inc"
};

/// Arguments that vary per forward pass.
struct AttentionForwardArgs
{
#include "tensorrt_llm/thop/attention_forward_args_accessors.inc"
#include "tensorrt_llm/thop/attention_forward_args_fields.inc"

    template <typename T>
    [[nodiscard]] T* getLatentCache() const
    {
        return latent_cache.has_value() ? static_cast<T*>(latent_cache.value().data_ptr()) : nullptr;
    }

    template <typename T>
    [[nodiscard]] T* getQPe() const
    {
        return q_pe.has_value() ? static_cast<T*>(q_pe.value().data_ptr()) : nullptr;
    }

    void* getOutputSf() const
    {
        return output_sf.has_value() ? output_sf.value().data_ptr() : nullptr;
    }

    void* getQuantQBuffer() const
    {
        return quant_q_buffer.has_value() ? quant_q_buffer.value().data_ptr() : nullptr;
    }

    [[nodiscard]] float2* getMropeRotaryCosSin() const
    {
        return mrope_rotary_cos_sin.has_value() ? static_cast<float2*>(mrope_rotary_cos_sin.value().data_ptr())
                                                : nullptr;
    }

    [[nodiscard]] float2* getSoftmaxStatsTensor() const
    {
        return softmax_stats_tensor.has_value() ? static_cast<float2*>(softmax_stats_tensor.value().data_ptr())
                                                : nullptr;
    }
};

class AttentionBase;

/// Everything that selects and configures a layer's kernel runners, and the key of the
/// per-thread op cache. Plain struct: copying a key is memberwise and `operator==` is
/// defaulted over every member; `std::hash<AttentionConfig>` below reads the same set.
/// Values derived from these members live on AttentionBase, not here.
struct AttentionConfig
{
    using RotaryScalingType = tensorrt_llm::kernels::RotaryScalingType;
    using PositionEmbeddingType = tensorrt_llm::kernels::PositionEmbeddingType;
    using AttentionMaskType = tensorrt_llm::kernels::AttentionMaskType;

    int device = 0;
    tensorrt_llm::DataType type = tensorrt_llm::DataType::kHALF;
    bool is_fp8_out = false;
    bool is_fp4_out = false;
    bool use_kv_cache = false;
    bool cross_attention = false;
    std::int64_t tokens_per_block = 0;
    AttentionMaskType mask_type = AttentionMaskType::CAUSAL;
    PositionEmbeddingType position_embedding_type = PositionEmbeddingType::kLEARNED_ABSOLUTE;
    std::int64_t num_heads = 0;
    std::int64_t num_kv_heads = 0;
    std::int64_t head_size = 0;
    double q_scaling = 1.0;
    std::int64_t rotary_embedding_dim = 0;
    tensorrt_llm::common::QuantMode quant_mode{};
    bool uses_spcompress = false;
    bool paged_context_fmha = false;
    bool is_spec_decoding_enabled = false;
    std::int64_t spec_decoding_target_max_gen_len = 0;
    bool fuses_dsv4_inv_rope_fp8_quant = false;
    bool use_sparse_attention = false;
    bool use_tllm_gen_sparse_attention = false;
    bool use_nvfp4_mla_kv_cache = false;
    bool is_mla_enable = false;
    std::int64_t q_lora_rank = 0;
    std::int64_t kv_lora_rank = 0;
    std::int64_t qk_nope_head_dim = 0;
    std::int64_t qk_rope_head_dim = 0;
    std::int64_t v_head_dim = 0;
    std::int64_t predicted_tokens_per_seq = 1;
    std::int64_t mla_layer_num = 0;
    bool rope_append = true;
    std::int64_t sage_attn_num_elts_per_blk_q = 0;
    std::int64_t sage_attn_num_elts_per_blk_k = 0;
    std::int64_t sage_attn_num_elts_per_blk_v = 0;
    bool sage_attn_qk_int8 = false;
    bool sage_attn_smooth_k = false;
    double skip_correction_threshold = 0.0;

    // Fixed for the phased path: inputs are never padded, the dense context FMHA is never
    // selected, and logit soft-capping is not exposed.
    static constexpr bool remove_padding = true;
    static constexpr bool dense_context_fmha = false;
    static constexpr double attn_logit_softcapping_scale = 0.0;

    bool operator==(AttentionConfig const&) const = default;

    [[nodiscard]] bool isRelativePosition() const
    {
        return position_embedding_type == PositionEmbeddingType::kRELATIVE;
    }

    [[nodiscard]] bool isALiBi() const
    {
        return position_embedding_type == PositionEmbeddingType::kALIBI
            || position_embedding_type == PositionEmbeddingType::kALIBI_WITH_SCALE;
    }

    [[nodiscard]] bool isAliBiWithScale() const
    {
        return position_embedding_type == PositionEmbeddingType::kALIBI_WITH_SCALE;
    }

    [[nodiscard]] bool isRoPE() const
    {
        return position_embedding_type == PositionEmbeddingType::kROPE_GPTJ
            || position_embedding_type == PositionEmbeddingType::kROPE_GPT_NEOX
            || position_embedding_type == PositionEmbeddingType::kLONG_ROPE
            || position_embedding_type == PositionEmbeddingType::kYARN
            || position_embedding_type == PositionEmbeddingType::kROPE_M;
    }

    [[nodiscard]] bool isLongRoPE() const
    {
        return position_embedding_type == PositionEmbeddingType::kLONG_ROPE;
    }

    [[nodiscard]] bool isMRoPE() const
    {
        return position_embedding_type == PositionEmbeddingType::kROPE_M;
    }

    [[nodiscard]] bool useCustomMask() const
    {
        return mask_type == AttentionMaskType::CUSTOM_MASK;
    }

    [[nodiscard]] bool isMLAEnabled() const
    {
        return is_mla_enable;
    }

    [[nodiscard]] bool sageAttnRequested() const
    {
        return sage_attn_num_elts_per_blk_q > 0 || sage_attn_num_elts_per_blk_k > 0 || sage_attn_num_elts_per_blk_v > 0;
    }

    /// The key fields, for the cache-miss log and for diffing two keys.
    [[nodiscard]] std::string toString() const;
};

/// Parameters for phased attention, generated from fmha/phased.py.
/// Device tensors are phase-local; host tensors, KV-cache block offsets and FP4
/// scales remain whole-batch. Their accessors apply seq_offset / token_offset.
///
/// One instance is cached per fallback library instance and host thread. The binding
/// refreshes the generated fields from the Python objects on every call and clears
/// every tensor reference afterwards.
struct FmhaParams
{
#include "tensorrt_llm/thop/fmha_params_fields.inc"
#undef TRTLLM_FMHA_PARAM_FIELD

    // Derived by prepare(): host reductions over the pinned length tensors, computed once
    // per call.
    std::int64_t total_kv_len = 0;

    [[nodiscard]] tensorrt_llm::DataType getType() const
    {
        return tensorrt_llm::runtime::TorchUtils::dataType(getQueryInput().scalar_type());
    }

    /// The query-bearing input: packed QKV, or the separate query tensor MLA hands over.
    [[nodiscard]] torch::Tensor const& getQueryInput() const
    {
        return qkv_input.has_value() ? qkv_input.value() : query_input.value();
    }

    [[nodiscard]] bool isFp8Out() const
    {
        return output->scalar_type() == torch::kFloat8_e4m3fn;
    }

    [[nodiscard]] bool isFp4Out() const
    {
        return output->scalar_type() == torch::kUInt8;
    }

    [[nodiscard]] bool hasSparseAttnIndices() const
    {
        return fwd.sparse_runtime_params.sparse_attn_indices.has_value()
            && fwd.sparse_runtime_params.sparse_attn_indices.value().numel() > 0;
    }

    [[nodiscard]] bool hasPagedSparseAttnIndices() const
    {
        return hasSparseAttnIndices() && fwd.sparse_runtime_params.sparse_attn_offsets.has_value()
            && fwd.sparse_runtime_params.sparse_attn_offsets.value().numel() > 0;
    }

    /// `out_scale` is the FP8 output scale, but the global scale of the scaling factors
    /// when the NVFP4 quant epilogue is fused; the two feed different kernel arguments.
    [[nodiscard]] float* getOutQuantScale() const
    {
        return isFp4Out() ? nullptr : getOutScale();
    }

    [[nodiscard]] float* getOutSfScale() const
    {
        return isFp4Out() ? getOutScale() : nullptr;
    }

    /// Cross attention addresses the encoder KV by the phase-local sequence lengths.
    [[nodiscard]] bool hasUnpagedSparseAttnIndices() const
    {
        return hasSparseAttnIndices() && !hasPagedSparseAttnIndices();
    }

#include "tensorrt_llm/thop/fmha_params_accessors.inc"
#include "tensorrt_llm/thop/fmha_params_forwarding.inc"

    template <typename T>
    [[nodiscard]] T* getQueryInput() const
    {
        return static_cast<T*>(getQueryInput().data_ptr());
    }

    template <typename T>
    [[nodiscard]] T* getK() const
    {
        return key_input.has_value() ? static_cast<T*>(key_input.value().data_ptr()) : nullptr;
    }

    template <typename T>
    [[nodiscard]] T* getV() const
    {
        return value_input.has_value() ? static_cast<T*>(value_input.value().data_ptr()) : nullptr;
    }

    template <typename T>
    [[nodiscard]] T* getLatentCache() const
    {
        return fwd.getLatentCache<T>();
    }

    template <typename T>
    [[nodiscard]] T* getQPe() const
    {
        return fwd.getQPe<T>();
    }

    void* getWorkspace() const
    {
        return workspace.data_ptr();
    }

    void* getOutput() const
    {
        return output.has_value() ? output->data_ptr() : nullptr;
    }

    void* getOutputSf() const
    {
        return fwd.getOutputSf();
    }

    void* getQuantQBuffer() const
    {
        return fwd.getQuantQBuffer();
    }

    // ---- layer tensors, read from attn ----
    [[nodiscard]] float2* getRotaryCosSin() const
    {
        return attn.rotary_cos_sin.has_value() ? static_cast<float2*>(attn.rotary_cos_sin.value().data_ptr()) : nullptr;
    }

    [[nodiscard]] float* getRotaryInvFreq() const
    {
        return attn.rotary_inv_freq.has_value() ? attn.rotary_inv_freq.value().data_ptr<float>() : nullptr;
    }

    [[nodiscard]] std::uint32_t* getSkipSoftmaxStat() const
    {
        return attn.skip_softmax_stat.defined() ? attn.skip_softmax_stat.data_ptr<std::uint32_t>() : nullptr;
    }

    // ---- batch tensors, read from meta ----
    [[nodiscard]] std::int32_t* getCacheIndirection() const
    {
        return meta.cache_indirection.has_value() ? meta.cache_indirection.value().data_ptr<std::int32_t>() : nullptr;
    }

    [[nodiscard]] std::int32_t* getBlockIdsPerSeq() const
    {
        return meta.block_ids_per_seq.has_value() ? meta.block_ids_per_seq.value().data_ptr<std::int32_t>() : nullptr;
    }

    [[nodiscard]] std::int32_t* getHelixPositionOffsets() const
    {
        return meta.helix_position_offsets.has_value() ? meta.helix_position_offsets.value().data_ptr<std::int32_t>()
                                                       : nullptr;
    }

    [[nodiscard]] bool* getHelixIsInactiveRank() const
    {
        return meta.helix_is_inactive_rank.has_value() ? meta.helix_is_inactive_rank.value().data_ptr<bool>() : nullptr;
    }

    [[nodiscard]] std::int32_t* getFlashMlaTileSchedulerMetadata() const
    {
        return meta.flash_mla_tile_scheduler_metadata.has_value()
            ? meta.flash_mla_tile_scheduler_metadata.value().data_ptr<std::int32_t>()
            : nullptr;
    }

    [[nodiscard]] std::int32_t* getFlashMlaNumSplits() const
    {
        return meta.flash_mla_num_splits.has_value() ? meta.flash_mla_num_splits.value().data_ptr<std::int32_t>()
                                                     : nullptr;
    }

    [[nodiscard]] float2* getMropeRotaryCosSin() const
    {
        return fwd.getMropeRotaryCosSin();
    }

    [[nodiscard]] float2* getSoftmaxStatsTensor() const
    {
        return fwd.getSoftmaxStatsTensor();
    }

    // ---- hand-written accessors: sizes, derived pointers, and anything taking an index ----
    // Sequence lengths, windows, and paged-KV metadata.
    /// Block offsets for this phase's sequences; `seq_offset` is applied here.
    [[nodiscard]] kernels::KVBlockArray::DataType* getKvCacheBlockOffsets(int32_t poolIndex) const
    {
        return meta.kv_cache_block_offsets.has_value() ? static_cast<kernels::KVBlockArray::DataType*>(
                   meta.kv_cache_block_offsets.value().index({poolIndex, seq_offset}).data_ptr())
                                                       : nullptr;
    }

    // Quantization scales and output quantization.
    // RoPE, ALiBi, and logn data.
    // MLA input/cache data.
    // MRoPE and Helix position data.
    // Context chunking, Helix reduction, and softmax statistics.
    // Speculative decoding masks and offsets.
    // Attention sinks.
    // Sparse attention, sparse MLA, and SageAttention runtime data.
    // Packed-varlen context boundaries and scheduler state.
    // MLA scales, quantized-Q buffers, and FlashMLA metadata.
    // Cross attention.
    // DeepSeek-V4 FP8-Q/epilogue fusion.
    /// Host past-KV lengths for this phase's sequences.
    ///
    /// `seq_offset` is applied here, so the returned array is phase-local and lines up
    /// with the device tensors, which the caller slices before handing them over.
    [[nodiscard]] int32_t* getHostPastKeyValueLengths() const
    {
        // A None source lowers to an undefined tensor rather than an error, so name the
        // field here instead of failing inside data_ptr with only a dtype to go on.
        TORCH_CHECK(meta.kv_lens_runtime.defined(), "FmhaParams.meta.kv_lens_runtime is unset.");
        return static_cast<int32_t*>(meta.kv_lens_runtime.data_ptr()) + seq_offset;
    }

    /// Host context lengths for this phase's sequences, offset as above.
    [[nodiscard]] int32_t* getHostContextLengths() const
    {
        TORCH_CHECK(meta.prompt_lens_cpu_runtime.defined(), "FmhaParams.meta.prompt_lens_cpu_runtime is unset.");
        return static_cast<int32_t*>(meta.prompt_lens_cpu_runtime.data_ptr()) + seq_offset;
    }

    /// A phase with no sequences has no extent; at::max() rejects an empty input, so the
    /// empty range yields 0, matching the sum-based reduction below.
    [[nodiscard]] int32_t getMaxHostPastKeyValueLength(int64_t seqOffset, int64_t numSeqs) const
    {
        if (numSeqs <= 0)
        {
            return 0;
        }
        return meta.kv_lens_runtime.slice(0, seqOffset, seqOffset + numSeqs).max().item<int32_t>();
    }

    [[nodiscard]] int32_t getMaxHostContextLength(int64_t seqOffset, int64_t numSeqs) const
    {
        if (numSeqs <= 0)
        {
            return 0;
        }
        return meta.prompt_lens_cpu_runtime.slice(0, seqOffset, seqOffset + numSeqs).max().item<int32_t>();
    }

    /// Total attended-KV length for this phase. Prefer explicit phase totals when
    /// supplied; otherwise sum the host KV lengths over the phase's sequences.
    [[nodiscard]] int32_t getTotalKvLen(int64_t seqOffset, int64_t numSeqs, bool isGen) const
    {
        if (numSeqs <= 0)
        {
            return 0;
        }
        if (host_total_kv_lens.has_value())
        {
            return host_total_kv_lens.value().select(0, isGen ? 1 : 0).item<int32_t>();
        }
        return meta.kv_lens_runtime.slice(0, seqOffset, seqOffset + numSeqs).sum().item<int32_t>();
    }

    [[nodiscard]] int getCacheIndirectionWindowSize(int defaultValue) const
    {
        return meta.cache_indirection.has_value() ? static_cast<int>(meta.cache_indirection.value().size(2))
                                                  : defaultValue;
    }

    [[nodiscard]] bool hasKvCache() const
    {
        return meta.kv_cache_block_offsets.has_value() && meta.host_kv_cache_pool_pointers.has_value()
            && meta.host_kv_cache_pool_mapping.has_value();
    }

    [[nodiscard]] torch::Tensor const& getHostKvCachePoolPointers() const
    {
        return meta.host_kv_cache_pool_pointers.value();
    }

    [[nodiscard]] int getMaxBlocksPerSequence() const
    {
        return meta.kv_cache_block_offsets.has_value() ? static_cast<int>(meta.kv_cache_block_offsets.value().size(-1))
                                                       : 0;
    }

    /// Sequences the generation workspace is sized for: the batch bound, not this
    /// call's row count, so a context call sizing a mixed batch and the following
    /// generation call agree.
    [[nodiscard]] int64_t getMaxNumSequences() const
    {
        return meta.max_num_sequences.value_or(meta.max_num_requests);
    }

    // NOTE: `host_kv_cache_pool_mapping` is indexed by the layer's index *within its own
    // KV-cache manager* (`local_layer_idx`), not by the model-global `layer_idx`. The two
    // differ for draft / MTP models, whose mapping holds only their own layers.
    [[nodiscard]] int32_t getKvCachePoolIndex(int64_t localLayerIdx) const
    {
        return meta.host_kv_cache_pool_mapping.has_value()
            ? checkedPoolMapping(localLayerIdx).index({localLayerIdx, 0}).item<int32_t>()
            : 0;
    }

    [[nodiscard]] int32_t getLayerIdxInCachePool(int64_t localLayerIdx) const
    {
        return meta.host_kv_cache_pool_mapping.has_value()
            ? checkedPoolMapping(localLayerIdx).index({localLayerIdx, 1}).item<int32_t>()
            : 0;
    }

    [[nodiscard]] torch::Tensor const& checkedPoolMapping(int64_t localLayerIdx) const
    {
        auto const& mapping = meta.host_kv_cache_pool_mapping.value();
        TORCH_CHECK(localLayerIdx >= 0 && localLayerIdx < mapping.size(0), "local_layer_idx ", localLayerIdx,
            " is out of range for host_kv_cache_pool_mapping with ", mapping.size(0),
            " layers. This index must be the layer's position within its own KV-cache manager, "
            "not the model-global layer index.");
        return mapping;
    }

    [[nodiscard]] int64_t getMlaLayerNum() const
    {
        return meta.host_kv_cache_pool_mapping.has_value() ? meta.host_kv_cache_pool_mapping.value().size(0) : 0;
    }

    [[nodiscard]] int64_t getSparseAttnIndicesStride() const
    {
        return fwd.sparse_runtime_params.sparse_attn_indices.has_value()
            ? fwd.sparse_runtime_params.sparse_attn_indices.value().size(-1)
            : 0;
    }

    [[nodiscard]] int32_t getCrossKvNumTokens() const
    {
        return fwd.cross_kv.has_value() ? static_cast<int32_t>(fwd.cross_kv.value().size(0)) : 0;
    }

    [[nodiscard]] char* getSparseKvCachePool(int32_t poolIndex) const
    {
        if (!meta.host_kv_cache_pool_pointers.has_value())
        {
            return nullptr;
        }
        auto const& poolPointers = meta.host_kv_cache_pool_pointers.value();
        return reinterpret_cast<char*>(poolPointers.dim() == 3 ? poolPointers.index({poolIndex, 0, 0}).item<int64_t>()
                                                               : poolPointers.index({poolIndex, 0}).item<int64_t>());
    }

    /// Kernel view of the sparse indices and the KV pools they address. The pool fields
    /// are only meaningful while sparse attention is active; callers gate on the op.
    [[nodiscard]] kernels::SparseAttentionParams getSparseParams() const
    {
        kernels::SparseAttentionParams sparse{};
        sparse.sparse_kv_indices = getSparseKvIndices();
        sparse.sparse_kv_offsets = getSparseKvOffsets();
        sparse.sparse_attn_indices = getSparseAttnIndices();
        sparse.sparse_attn_offsets = getSparseAttnOffsets();
        sparse.sparse_attn_indices_block_size = fwd.sparse_runtime_params.sparse_attn_indices_block_size;
        sparse.sparse_attn_indices_stride = getSparseAttnIndicesStride();
        sparse.num_sparse_topk = meta.num_sparse_topk;
        sparse.sparse_attn_kv_lens = getSparseAttnKvLens();
        if (!hasKvCache())
        {
            return sparse;
        }
        auto* kvCachePool = getSparseKvCachePool(getKvCachePoolIndex(local_layer_idx));
        if (kvCachePool == nullptr)
        {
            return sparse;
        }
        if (fwd.sparse_runtime_params.sparse_attn_kv_lens.has_value())
        {
            // Deepseek V4 dynamic sparse MLA always uses the SWA pool for now.
            sparse.sliding_window_kv_cache_pool = kvCachePool;
            if (fwd.sparse_runtime_params.aux_kv_cache_pool_ptr.has_value())
            {
                sparse.sparse_kv_cache_pool
                    = reinterpret_cast<void*>(fwd.sparse_runtime_params.aux_kv_cache_pool_ptr.value());
            }
        }
        else
        {
            sparse.sparse_kv_cache_pool = kvCachePool;
        }
        return sparse;
    }
};

/// Per-token byte cost of the context-MLA K/V dequant staging buffers, whose size scales with the
/// summed attended KV length. Only the fp8 context-MLA separate-Q/KV path stages these buffers; every
/// other path (incl. sparse MLA, which reads K/V straight from the paged cache) returns 0. Shared by
/// the op's context sizing and the KV-cache estimator so the two cannot drift.
[[nodiscard]] size_t contextMlaWorkspaceBytesPerToken(int32_t numAttnHeads, int32_t qkRopeHeadDim,
    int32_t qkNopeHeadDim, int32_t vHeadDim, bool fp8ContextMla, bool separateQAndKvInput, bool sparseMla) noexcept;

/// Fill the key from live parameters. Reads tensor metadata (dtype, device, presence and
/// sizes), so it runs in the binding while the tensors are still attached.
void fillAttentionConfig(AttentionConfig& config, FmhaParams const& p);

/// Abstract attention op: the key plus every value derived from it, and the entry points
/// the binding calls. The per-thread op cache stores these; the miss path chooses the
/// concrete class.
class AttentionBase : public AttentionConfig
{
public:
    /// Derives the device constants and the plain layer geometry. Does no CUDA work:
    /// runners and the cuBLAS handle are created on the first call of each phase, which
    /// keeps construction legal while a stream is capturing.
    explicit AttentionBase(AttentionConfig const& cfg);
    // No CUDA calls in destructors: the thread_local cache is destroyed at thread exit,
    // possibly after the CUDA context.
    virtual ~AttentionBase() = default;
    AttentionBase(AttentionBase const&) = delete;
    AttentionBase& operator=(AttentionBase const&) = delete;

    virtual void runContext(FmhaParams& p) = 0;
    virtual void runGeneration(FmhaParams& p) = 0;

    /// The op for `key`, built on first use and owned by the calling thread's cache.
    static AttentionBase* getCached(AttentionConfig const& key);

    [[nodiscard]] int getHeadSize() const;

    [[nodiscard]] int smVersion() const
    {
        return mSM;
    }

    [[nodiscard]] int getMultiProcessorCount() const
    {
        return mMultiProcessorCount;
    }

protected:
    // Derived from the device.
    int mSM = tensorrt_llm::common::getSMVersion();
    bool mUseTllmGen = (mSM >= 100) && (mSM != 120) && (mSM != 121);
    int mMultiProcessorCount = tensorrt_llm::common::getMultiProcessorCount();
    int mMaxSharedMemoryPerBlockOptin = tensorrt_llm::common::getMaxSharedMemoryPerBlockOptin();
    // Plain layer geometry from the key. An op whose KV layout differs (MLA) overrides
    // these in its own constructor; nothing MLA-specific lives here.
    int mNumKVHeads = -1;
    int mHeadSize = -1;
    // Equal to the full head counts: this op always runs on a single rank.
    int mNumAttnHeads = -1;
    int mNumAttnKVHeads = -1;
    bool mFMHAForceFP32Acc = false;
};

struct KvCachePoolMapping
{
    int32_t poolIndex{0};
    int32_t layerIdxInCachePool{0};
};

KvCachePoolMapping readKvCachePoolMapping(at::Tensor const& hostKvCachePoolMapping, int64_t layerIdx);

KvCachePoolPointers buildKvCachePoolPointers(at::Tensor const& hostKvCachePoolPointers, int32_t poolIndex,
    int64_t intraPoolOffset, int64_t blockSize, int32_t layerIdxInCachePool, int32_t kvFactor, bool isFp4KvCache);

template <typename KVCacheBuffer>
struct KvCacheBuffers
{
    KVCacheBuffer kvCacheBuffer;
    KVCacheBuffer kvScaleCacheBuffer;
};

template <typename KVCacheBuffer>
KvCacheBuffers<KVCacheBuffer> buildKvCacheBuffers(int32_t batchSize, int32_t maxBlocksPerSeq, int32_t tokensPerBlock,
    int32_t sizePerToken, int32_t cyclicAttentionWindowSize, int32_t maxCyclicAttentionWindowSize, int32_t sinkTokenLen,
    bool canUseOneMoreBlock, void* primaryPoolPtr, void* secondaryPoolPtr, void* primaryBlockScalePoolPtr,
    void* secondaryBlockScalePoolPtr, kernels::KVBlockArray::DataType* blockOffsets, bool hasFp4KvCache,
    int32_t maxAttentionWindowSize = 0, void* keyValueCache = nullptr);

KvCacheBuffers<kernels::KVBlockArray> buildPagedKvCacheBuffers(
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
// Context follows MHA::getWorkspaceSizeForContext() ordering. Generation
// follows the XQA workspace ordering used by MHA generation.
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
        bool fp8ContextFmha, bool skipFmhaWorkspace = false);

    static TrtllmGenGenerationWorkspaceLayout buildGenerationLayout(at::ScalarType qDtype, int64_t batchBeam,
        int64_t numTokens, int64_t numHeads, int64_t headSize, int64_t rotaryEmbeddingDim, int64_t numKvHeads,
        int64_t maxBlocksPerSequence, bool useSparseAttention, bool skipFmhaWorkspace = false);

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

/// Hashes exactly the members `AttentionConfig::operator==` compares.
template <>
struct std::hash<tensorrt_llm::torch_ext::AttentionConfig>
{
    std::size_t operator()(tensorrt_llm::torch_ext::AttentionConfig const& c) const noexcept;
};
