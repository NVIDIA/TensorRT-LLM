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
#include <torch/extension.h>

#include <cassert>
#include <cstdint>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <tuple>

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

/// An attention layer's fixed shape, handed to the op once at construction.
/// Generated from StaticAttentionConfig.
struct StaticAttentionConfig
{
#include "tensorrt_llm/thop/static_attention_config_fields.inc"
};

/// Sparse inputs an attention module hands to its backend. Generated from
/// SparseBackendForwardArgs; see FmhaParams for how a schema class is written.
struct SparseBackendForwardArgs
{
#include "tensorrt_llm/thop/sparse_backend_forward_args_accessors.inc"
#include "tensorrt_llm/thop/sparse_backend_forward_args_fields.inc"
};

/// Sparse inputs a backend hands to the attention op. Generated from
/// SparseRuntimeParams.
struct SparseRuntimeParams
{
#include "tensorrt_llm/thop/sparse_runtime_params_accessors.inc"
#include "tensorrt_llm/thop/sparse_runtime_params_fields.inc"
};

/// The arguments that vary per forward pass. Generated from AttentionForwardArgs and
/// held by value in FmhaParams, so the native layout mirrors the Python one.
struct AttentionForwardArgs
{
#include "tensorrt_llm/thop/attention_forward_args_accessors.inc"
#include "tensorrt_llm/thop/attention_forward_args_fields.inc"

    // The attention entry point's dtype dispatch supplies T.
    template <typename T>
    T* getLatentCache() const
    {
        return latent_cache.has_value() ? static_cast<T*>(latent_cache.value().data_ptr()) : nullptr;
    }

    template <typename T>
    T* getQPe() const
    {
        return q_pe.has_value() ? static_cast<T*>(q_pe.value().data_ptr()) : nullptr;
    }

    template <typename T>
    T* getCrossKv() const
    {
        return cross_kv.has_value() ? static_cast<T*>(cross_kv.value().data_ptr()) : nullptr;
    }

    template <typename T>
    T* getRelativeAttentionBias() const
    {
        return relative_attention_bias.has_value() ? static_cast<T*>(relative_attention_bias.value().data_ptr())
                                                   : nullptr;
    }

    // Handwritten for the same reason as in FmhaParams: the native view of these
    // buffers is not their dtype.
    void* getOutputSf() const
    {
        return output_sf.has_value() ? output_sf.value().data_ptr() : nullptr;
    }

    void* getQuantQBuffer() const
    {
        return quant_q_buffer.has_value() ? quant_q_buffer.value().data_ptr() : nullptr;
    }

    float2* getMropeRotaryCosSin() const
    {
        return mrope_rotary_cos_sin.has_value() ? static_cast<float2*>(mrope_rotary_cos_sin.value().data_ptr())
                                                : nullptr;
    }

    float2* getSoftmaxStatsTensor() const
    {
        return softmax_stats_tensor.has_value() ? static_cast<float2*>(softmax_stats_tensor.value().data_ptr())
                                                : nullptr;
    }
};

/// The unify attention parameter struct: every phased entry point and every enqueue path
/// consumes it directly. Data members are generated from the Python schema
/// (fmha/interface.py, via scripts/generate_fmha_params.py), which also states the offset
/// contract in full. In short: device tensors arrive pre-sliced for the phase, while host
/// tensors, KV-cache block offsets and FP4 scaling factors arrive whole-batch: a pointer
/// accessor applies `seq_offset` / `token_offset` itself, so call sites never pass one.
/// Fixed-dtype pointer accessors are generated. Runtime-dispatched types and semantic
/// views stay handwritten so dtype dispatch, validation and offsets remain explicit.
///
/// A C++-only field, for state the op derives rather than receives, is declared below the
/// generated block and filled during initialization; it stays invisible to Python. A field
/// visible to both is declared in the Python schema instead, and the build regenerates the
/// member and its binding.
class AttentionOp;

struct FmhaParams
{
#include "tensorrt_llm/thop/fmha_params_fields.inc"
#undef TRTLLM_FMHA_PARAM_FIELD

    // ---- handwritten derived state (deliberately outside the generated schema) ----
    // Both are filled by AttentionOp::prepare(): they are projections of the fields
    // above plus the cache-layout arithmetic the op alone can do.
    kernels::SparseAttentionParams sparse_params{};
    KvCachePoolPointers kv_cache_pool_pointers{};

    /// Read straight off the inputs rather than mirrored into a field: a mirror can
    /// disagree with the tensor it describes, and these are all cheap scalar queries.
    tensorrt_llm::DataType getType() const
    {
        return tensorrt_llm::runtime::TorchUtils::dataType(qkv_or_q.scalar_type());
    }

    bool isFp8Out() const
    {
        return output.scalar_type() == torch::kFloat8_e4m3fn;
    }

    bool isFp4Out() const
    {
        return output.scalar_type() == torch::kUInt8;
    }

    bool hasSparseAttnIndices() const
    {
        return fwd.sparse_runtime_params.sparse_attn_indices.has_value()
            && fwd.sparse_runtime_params.sparse_attn_indices.value().numel() > 0;
    }

    bool hasPagedSparseAttnIndices() const
    {
        return hasSparseAttnIndices() && fwd.sparse_runtime_params.sparse_attn_offsets.has_value()
            && fwd.sparse_runtime_params.sparse_attn_offsets.value().numel() > 0;
    }

    // Accessors for the members above. These mirror what the generator emitted while the
    // fields were part of the schema; nothing assigns them, so they resolve to nullptr.
    template <typename T>
    T* getQkvBias() const
    {
        return qkv_bias.has_value() ? static_cast<T*>(qkv_bias.value().data_ptr()) : nullptr;
    }

    template <typename T>
    T* getAlibiSlopes() const
    {
        return alibi_slopes.has_value() ? static_cast<T*>(alibi_slopes.value().data_ptr()) : nullptr;
    }

    bool* getAttentionMask() const
    {
        return attention_mask.has_value() ? attention_mask.value().data_ptr<bool>() : nullptr;
    }

    std::uint32_t* getAttentionPackedMask() const
    {
        return attention_packed_mask.has_value() ? attention_packed_mask.value().data_ptr<std::uint32_t>() : nullptr;
    }

    void* getKeyValueCache() const
    {
        return key_value_cache.has_value() ? key_value_cache.value().data_ptr() : nullptr;
    }

    float* getLognScalingPtr() const
    {
        return logn_scaling_ptr.has_value() ? logn_scaling_ptr.value().data_ptr<float>() : nullptr;
    }

    float* getOutSfScale() const
    {
        return out_sf_scale.has_value() ? out_sf_scale.value().data_ptr<float>() : nullptr;
    }

    std::int32_t* getEncoderInputLengths() const
    {
        return encoder_input_lengths.has_value() ? encoder_input_lengths.value().data_ptr<std::int32_t>() : nullptr;
    }

    std::int64_t* getRuntimePerfKnobs() const
    {
        return runtime_perf_knobs.has_value() ? runtime_perf_knobs.value().data_ptr<std::int64_t>() : nullptr;
    }

    bool hasUnpagedSparseAttnIndices() const
    {
        return hasSparseAttnIndices() && !hasPagedSparseAttnIndices();
    }

    std::int64_t attention_mask_stride = 0;
    tensorrt_llm::kernels::BlockSparseParams block_sparse_params{};
    bool can_use_one_more_block = false;
    std::int64_t cross_kv_length = 0;
    std::optional<torch::Tensor> encoder_input_lengths{};
    bool has_full_attention_mask = false;
    std::int64_t input_seq_length = 0;
    std::int64_t max_blocks_per_sequence = 0;
    std::int64_t max_cyclic_attention_window_size = 0;
    std::int64_t max_past_kv_length = 0;
    std::int64_t num_encoder_tokens = 0;
    std::optional<torch::Tensor> out_sf_scale{};
    std::optional<torch::Tensor> qkv_bias{};
    std::optional<torch::Tensor> alibi_slopes{};
    std::optional<torch::Tensor> key_value_cache{};
    std::optional<torch::Tensor> logn_scaling_ptr{};
    std::optional<torch::Tensor> attention_mask{};
    std::optional<torch::Tensor> attention_packed_mask{};
    bool pos_shift_enabled = false;
    bool qkv_bias_enabled = false;
    std::int64_t relative_attention_bias_stride = 0;
    std::optional<torch::Tensor> runtime_perf_knobs{};
    std::int64_t sink_token_length = 0;
    bool spec_decoding_is_generation_length_variable = false;
    std::int64_t spec_decoding_max_generation_length = 1;
    std::int64_t spec_decoding_target_max_gen_len = 0;
    std::int64_t total_kv_len = 0;
    bool unfuse_qkv_gemm = false;
    std::int64_t unidirectional = 1;
    bool use_logn_scaling = false;
    bool use_sparse_attention = false;
    std::int64_t vision_length = -1;
    std::int64_t vision_start = -1;

    /// Build the MlaParams the MLA kernels take. Returned by value: the kernels hold a
    /// pointer to it for the duration of the launch, so the caller owns the storage.
    /// Defined out of line because they reach for AttentionOp, which is declared below.
    template <typename T>
    kernels::MlaParams<T> buildContextMlaParams(AttentionOp const& op) const;

    template <typename T>
    kernels::MlaParams<T> buildGenerationMlaParams(AttentionOp const& op) const;

    /// Fills the fields flash-MLA generation adds on top of buildGenerationMlaParams().
    template <typename T>
    void addFlashMlaGenerationParams(kernels::MlaParams<T>& mla) const;

    /// The tail both builders share.
    template <typename T>
    /// Takes the op because the MLA dimensions, the head count and the position-embedding
    /// type are layer configuration, which FmhaParams deliberately does not carry.
    void finalizeMlaParams(kernels::MlaParams<T>& mla, AttentionOp const& op) const;

    // ---- generated accessors: fixed-dtype tensor views ----
#include "tensorrt_llm/thop/fmha_params_accessors.inc"

    // ---- generated forwarding: reaches through `fwd` so call sites stay flat ----
#include "tensorrt_llm/thop/fmha_params_forwarding.inc"

    // The attention entry point's dtype dispatch supplies T.
    template <typename T>
    T* getQkvOrQ() const
    {
        return static_cast<T*>(qkv_or_q.data_ptr());
    }

    template <typename T>
    T* getK() const
    {
        return k.has_value() ? static_cast<T*>(k.value().data_ptr()) : nullptr;
    }

    template <typename T>
    T* getV() const
    {
        return v.has_value() ? static_cast<T*>(v.value().data_ptr()) : nullptr;
    }

    template <typename T>
    T* getLatentCache() const
    {
        return fwd.getLatentCache<T>();
    }

    template <typename T>
    T* getQPe() const
    {
        return fwd.getQPe<T>();
    }

    template <typename T>
    T* getCrossKv() const
    {
        return fwd.getCrossKv<T>();
    }

    template <typename T>
    T* getRelativeAttentionBias() const
    {
        return fwd.getRelativeAttentionBias<T>();
    }

    // ---- hand-written accessors: the native view differs from the tensor's dtype ----
    // These read the buffer as something the schema cannot name: an opaque pointer, or
    // a float32 buffer consumed as pairs.
    void* getWorkspace() const
    {
        return workspace.data_ptr();
    }

    /// The multi-CTA-KV counter semaphores are an opaque byte-sized scratch buffer.
    void* getMultiCtasKvCounter() const
    {
        return multi_ctas_kv_counter.has_value() ? multi_ctas_kv_counter.value().data_ptr() : nullptr;
    }

    void* getOutput() const
    {
        return output.data_ptr();
    }

    void* getOutputSf() const
    {
        return fwd.getOutputSf();
    }

    void* getQuantQBuffer() const
    {
        return fwd.getQuantQBuffer();
    }

    float2* getRotaryCosSin() const
    {
        return rotary_cos_sin.has_value() ? static_cast<float2*>(rotary_cos_sin.value().data_ptr()) : nullptr;
    }

    float2* getMropeRotaryCosSin() const
    {
        return fwd.getMropeRotaryCosSin();
    }

    float2* getSoftmaxStatsTensor() const
    {
        return fwd.getSoftmaxStatsTensor();
    }

    // ---- hand-written accessors: sizes, derived pointers, and anything taking an index ----
    // Sequence lengths, windows, and paged-KV metadata.
    /// Block offsets for this phase's sequences; `seq_offset` is applied here.
    kernels::KVBlockArray::DataType* getKvCacheBlockOffsets(int32_t poolIndex) const
    {
        return kv_cache_block_offsets.has_value() ? static_cast<kernels::KVBlockArray::DataType*>(
                   kv_cache_block_offsets.value().index({poolIndex, seq_offset}).data_ptr())
                                                  : nullptr;
    }

    // The KV-cache pool base pointers are derived state, not inputs: they are the
    // int64 values stored in `host_kv_cache_pool_pointers` shifted by this layer's
    // intra-pool offset, resolved once the op knows the cache element size. Never
    // Python-facing.
    void* getHostPrimaryPoolPtr() const
    {
        return kv_cache_pool_pointers.primaryPoolPtr;
    }

    void* getHostSecondaryPoolPtr() const
    {
        return kv_cache_pool_pointers.secondaryPoolPtr;
    }

    void* getHostPrimaryBlockScalePoolPtr() const
    {
        return kv_cache_pool_pointers.primaryBlockScalePoolPtr;
    }

    void* getHostSecondaryBlockScalePoolPtr() const
    {
        return kv_cache_pool_pointers.secondaryBlockScalePoolPtr;
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
    int32_t* getHostPastKeyValueLengths() const
    {
        // A None source lowers to an undefined tensor rather than an error, so name the
        // field here instead of failing inside data_ptr with only a dtype to go on.
        TORCH_CHECK(host_past_key_value_lengths.defined(), "FmhaParams.host_past_key_value_lengths is unset.");
        return static_cast<int32_t*>(host_past_key_value_lengths.data_ptr()) + seq_offset;
    }

    /// Host context lengths for this phase's sequences, offset as above.
    int32_t* getHostContextLengths() const
    {
        TORCH_CHECK(host_context_lengths.defined(), "FmhaParams.host_context_lengths is unset.");
        return static_cast<int32_t*>(host_context_lengths.data_ptr()) + seq_offset;
    }

    /// A phase with no sequences has no extent; at::max() rejects an empty input, so the
    /// empty range yields 0, matching the sum-based reduction below.
    int32_t getMaxHostPastKeyValueLength(int64_t seqOffset, int64_t numSeqs) const
    {
        if (numSeqs <= 0)
        {
            return 0;
        }
        return host_past_key_value_lengths.slice(0, seqOffset, seqOffset + numSeqs).max().item<int32_t>();
    }

    int32_t getMaxHostContextLength(int64_t seqOffset, int64_t numSeqs) const
    {
        if (numSeqs <= 0)
        {
            return 0;
        }
        return host_context_lengths.slice(0, seqOffset, seqOffset + numSeqs).max().item<int32_t>();
    }

    /// Total attended-KV length for this phase. Prefer explicit phase totals when
    /// supplied; otherwise sum the host KV lengths over the phase's sequences.
    int32_t getTotalKvLen(int64_t seqOffset, int64_t numSeqs, bool isGen) const
    {
        if (numSeqs <= 0)
        {
            return 0;
        }
        if (host_total_kv_lens.has_value())
        {
            return host_total_kv_lens.value().select(0, isGen ? 1 : 0).item<int32_t>();
        }
        return host_past_key_value_lengths.slice(0, seqOffset, seqOffset + numSeqs).sum().item<int32_t>();
    }

    int getCacheIndirectionWindowSize(int defaultValue) const
    {
        return cache_indirection.has_value() ? static_cast<int>(cache_indirection.value().size(2)) : defaultValue;
    }

    bool hasKvCache() const
    {
        return kv_cache_block_offsets.has_value() && host_kv_cache_pool_pointers.has_value()
            && host_kv_cache_pool_mapping.has_value();
    }

    torch::Tensor const& getHostKvCachePoolPointers() const
    {
        return host_kv_cache_pool_pointers.value();
    }

    int getMaxBlocksPerSequence() const
    {
        return kv_cache_block_offsets.has_value() ? static_cast<int>(kv_cache_block_offsets.value().size(-1)) : 0;
    }

    // NOTE: `host_kv_cache_pool_mapping` is indexed by the layer's index *within its own
    // KV-cache manager* (`local_layer_idx`), not by the model-global `layer_idx`. The two
    // differ for draft / MTP models, whose mapping holds only their own layers.
    int32_t getKvCachePoolIndex(int64_t localLayerIdx) const
    {
        return host_kv_cache_pool_mapping.has_value()
            ? checkedPoolMapping(localLayerIdx).index({localLayerIdx, 0}).item<int32_t>()
            : 0;
    }

    int32_t getLayerIdxInCachePool(int64_t localLayerIdx) const
    {
        return host_kv_cache_pool_mapping.has_value()
            ? checkedPoolMapping(localLayerIdx).index({localLayerIdx, 1}).item<int32_t>()
            : 0;
    }

    torch::Tensor const& checkedPoolMapping(int64_t localLayerIdx) const
    {
        auto const& mapping = host_kv_cache_pool_mapping.value();
        TORCH_CHECK(localLayerIdx >= 0 && localLayerIdx < mapping.size(0), "local_layer_idx ", localLayerIdx,
            " is out of range for host_kv_cache_pool_mapping with ", mapping.size(0),
            " layers. This index must be the layer's position within its own KV-cache manager, "
            "not the model-global layer index.");
        return mapping;
    }

    int64_t getMlaLayerNum() const
    {
        return host_kv_cache_pool_mapping.has_value() ? host_kv_cache_pool_mapping.value().size(0) : 0;
    }

    int64_t getSparseAttnIndicesStride() const
    {
        return fwd.sparse_runtime_params.sparse_attn_indices.has_value()
            ? fwd.sparse_runtime_params.sparse_attn_indices.value().size(-1)
            : 0;
    }

    int32_t getCrossKvNumTokens() const
    {
        return fwd.cross_kv.has_value() ? static_cast<int32_t>(fwd.cross_kv.value().size(0)) : 0;
    }

    char* getSparseKvCachePool(int32_t poolIndex) const
    {
        return host_kv_cache_pool_pointers.has_value()
            ? reinterpret_cast<char*>(host_kv_cache_pool_pointers.value().index({poolIndex, 0}).item<int64_t>())
            : nullptr;
    }
};

class AttentionOp
{
public:
    /// Acquires the handles the op keeps for its lifetime and derives everything that
    /// follows from the layer's fixed shape. Deliberately not per call: cublasCreate()
    /// is illegal while a stream is capturing, so the op has to be built outside
    /// capture and reused for every call from its layer.
    explicit AttentionOp(StaticAttentionConfig const& cfg);

    using RotaryScalingType = tensorrt_llm::kernels::RotaryScalingType;
    using PositionEmbeddingType = tensorrt_llm::kernels::PositionEmbeddingType;
    using AttentionMaskType = tensorrt_llm::kernels::AttentionMaskType;

    /// Derives the per-call state into \p p. Kernel runners are fixed by the constructor;
    /// `mMultiBlockMode` remains a per-call runtime knob read from
    /// `p.runtime_perf_knobs` at generation time.
    int prepare(FmhaParams& p, bool isGen);

    /// One per entry point, instantiated per activation dtype. prepare() is called from
    /// here so the dtype-independent and dtype-dependent halves of the setup sit
    /// together, and so the MlaParams this scope owns outlives the launch.
    template <typename T>
    void runContextImpl(FmhaParams& p);
    template <typename T>
    void runGenerationImpl(FmhaParams& p);
    template <typename T>
    void runMlaGenerationImpl(FmhaParams& p);

    /// Phased attention entry points. Reuse one op per layer: it owns the cuBLAS handle,
    /// which is expensive to create and cannot be created while a stream is capturing.
    /// The caller's params are filled in place: Python builds a fresh holder for every
    /// call, so there is nothing to preserve and nothing to copy.
    void runContext(FmhaParams& params);
    void runGeneration(FmhaParams& params);
    void runMlaGeneration(FmhaParams& params);

    /// Max of the context and generation workspace byte requirements.
    int64_t getAttentionWorkspaceSize(FmhaParams const& params, int64_t numTokens, int64_t maxAttentionWindowSize,
        int64_t numGenTokens, int64_t maxBlocksPerSequence);

    [[nodiscard]] size_t getFmhaMultiCtasKvScratchSize(FmhaParams const& p) const noexcept;
    [[nodiscard]] int getHeadSize(bool checkInit = true) const;
    [[nodiscard]] int getMaxNumSeqLenTile(FmhaParams const& p, int batch_beam_size = 1) const;
    [[nodiscard]] size_t getWorkspaceSizeForContext(FmhaParams const& p, int32_t nbReq, int32_t max_input_length,
        int32_t cross_kv_length = 0, int32_t max_num_tokens = 0, int32_t total_kv_len = 0) const noexcept;
    // Per-token byte cost of the context-MLA K/V dequant staging buffers, whose size scales with the summed
    // attended KV length (`total_kv_len`). Only the fp8 context-MLA separate-Q/KV path stages these buffers;
    // every other path (incl. sparse MLA, which reads K/V straight from the paged cache) returns 0. Single
    // source of truth shared by getWorkspaceSizeForContext (runtime sizing) and the KV-cache estimator, so
    // the two cannot drift.
    [[nodiscard]] static size_t contextMlaWorkspaceBytesPerToken(int32_t numAttnHeads, int32_t qkRopeHeadDim,
        int32_t qkNopeHeadDim, int32_t vHeadDim, bool fp8ContextMla, bool separateQAndKvInput, bool sparseMla) noexcept;
    // total_num_seq is the sum of beam_width for multiple requests
    [[nodiscard]] size_t getWorkspaceSizeForGeneration(FmhaParams const& p, int32_t total_num_seq,
        int32_t max_attention_window_size, int32_t max_num_tokens, int32_t max_blocks_per_sequence) const noexcept;

    template <typename T, typename KVCacheBuffer>
    int enqueueContext(FmhaParams const& p, kernels::MlaParams<T>* mlaParam, cudaStream_t stream);

    template <typename T, typename KVCacheBuffer>
    int enqueueGeneration(FmhaParams const& p, cudaStream_t stream);

    template <typename T>
    int mlaGeneration(kernels::MlaParams<T>& params, FmhaParams const& p, cudaStream_t stream);

    int getFlashMlaNumSmParts(int s_q, int num_heads, int num_kv_heads, int head_size_v) const
    {
        static constexpr int block_size_m = 64;
        int num_heads_per_head_k = s_q * num_heads / num_kv_heads;
        int sm_cnt = mMultiProcessorCount;
        int num_sm_parts = sm_cnt / num_kv_heads / cutlass::ceil_div(num_heads_per_head_k, block_size_m);
        return num_sm_parts;
    }

    static int getFlashMlaNumSmPartsStatic(int s_q, int num_heads, int num_kv_heads, int head_size_v)
    {
        static constexpr int block_size_m = 64;
        int num_heads_per_head_k = s_q * num_heads / num_kv_heads;
        int device;
        cudaGetDevice(&device);
        int sm_cnt;
        cudaDeviceGetAttribute(&sm_cnt, cudaDevAttrMultiProcessorCount, device);
        int num_sm_parts = sm_cnt / num_kv_heads / cutlass::ceil_div(num_heads_per_head_k, block_size_m);
        return num_sm_parts;
    }

    template <typename T>
    int getKvCacheElemSizeInBits(FmhaParams const& p) const
    {
        return getKvCacheElemSizeInBits(mCfg.quant_mode, sizeof(T));
    }

    static int getKvCacheElemSizeInBits(tensorrt_llm::common::QuantMode quantMode, size_t dTypeSize)
    {
        if (quantMode.hasInt8KvCache() || quantMode.hasFp8KvCache())
        {
            return 8;
        }
        else if (quantMode.hasFp4KvCache())
        {
            return 4;
        }
        return dTypeSize * 8;
    }

    /// Defaulted so the dtype dispatch can name this directly; paged KV is the only
    /// layout the generation path prepares for.
    template <typename T, typename KVCacheBuffer = kernels::KVBlockArray>
    void prepareEnqueueGeneration(FmhaParams const& p);

    template <typename T, typename KVCacheBuffer>
    bool convertMMHAParamsToXQAParams(
        tensorrt_llm::kernels::XQAParams& xqaParams, FmhaParams const& p, bool forConfigurePlugin);

    /// The op's own derived state, for debugging a dispatch decision.
    [[nodiscard]] std::string toString() const;

    // ---------------------------------------------------------------------------
    // Predicates over the layer's configuration, read from mCfg, and over the per-call
    // state, which is passed explicitly.
    // ---------------------------------------------------------------------------
    [[nodiscard]] bool isRelativePosition() const
    {
        return mCfg.position_embedding_type == tensorrt_llm::kernels::PositionEmbeddingType::kRELATIVE;
    }

    [[nodiscard]] bool isALiBi() const
    {
        return mCfg.position_embedding_type == tensorrt_llm::kernels::PositionEmbeddingType::kALIBI
            || mCfg.position_embedding_type == tensorrt_llm::kernels::PositionEmbeddingType::kALIBI_WITH_SCALE;
    }

    [[nodiscard]] bool isAliBiWithScale() const
    {
        return mCfg.position_embedding_type == tensorrt_llm::kernels::PositionEmbeddingType::kALIBI_WITH_SCALE;
    }

    [[nodiscard]] bool isRoPE() const
    {
        return mCfg.position_embedding_type == tensorrt_llm::kernels::PositionEmbeddingType::kROPE_GPTJ
            || mCfg.position_embedding_type == tensorrt_llm::kernels::PositionEmbeddingType::kROPE_GPT_NEOX
            || mCfg.position_embedding_type == tensorrt_llm::kernels::PositionEmbeddingType::kLONG_ROPE
            || mCfg.position_embedding_type == tensorrt_llm::kernels::PositionEmbeddingType::kYARN
            || mCfg.position_embedding_type == tensorrt_llm::kernels::PositionEmbeddingType::kROPE_M;
    }

    [[nodiscard]] bool isLongRoPE() const
    {
        return mCfg.position_embedding_type == tensorrt_llm::kernels::PositionEmbeddingType::kLONG_ROPE;
    }

    [[nodiscard]] bool isUnfusedCrossAttention(FmhaParams const& p) const
    {
        return !mEnableContextFMHA && p.is_cross;
    }

    [[nodiscard]] bool isMRoPE() const
    {
        return mCfg.position_embedding_type == tensorrt_llm::kernels::PositionEmbeddingType::kROPE_M;
    }

    [[nodiscard]] bool isLognScaling(FmhaParams const& p) const
    {
        return p.use_logn_scaling;
    }

    [[nodiscard]] bool isCrossAttention(FmhaParams const& p) const
    {
        return p.is_cross;
    }

    [[nodiscard]] bool useKVCache(FmhaParams const& p) const
    {
        return p.hasKvCache();
    }

    [[nodiscard]] bool useCustomMask() const
    {
        return mCfg.mask_type == AttentionMaskType::CUSTOM_MASK;
    }

    [[nodiscard]] bool useFullCustomMask(FmhaParams const& p) const
    {
        return useCustomMask() && p.has_full_attention_mask;
    }

    [[nodiscard]] bool usePackedCustomMask(FmhaParams const& p) const
    {
        return useCustomMask() && mEnableContextFMHA;
    }

    [[nodiscard]] bool isMLAEnabled() const
    {
        return mCfg.is_mla_enable;
    }

    [[nodiscard]] bool useSparseAttention(FmhaParams const& p) const
    {
        return p.use_sparse_attention && mPagedKVCache && mEnableXQA;
    }

    [[nodiscard]] bool useTllmGenSparseAttentionPaged(FmhaParams const& p) const
    {
        return p.hasPagedSparseAttnIndices() && useSparseAttention(p);
    }

    // Config-only variants, for initialize(): it runs at construction and has no call to
    // read, so the same questions are answered from mCfg.
    [[nodiscard]] bool cfgUseSparseMLA() const
    {
        return mCfg.use_sparse_attention && mUseTllmGen && mCfg.is_mla_enable;
    }

    [[nodiscard]] bool cfgUseTllmGenSparseAttention() const
    {
        return cfgUseSparseMLA() || (mCfg.use_sparse_attention && mUseTllmGen && mCfg.use_tllm_gen_sparse_attention);
    }

    [[nodiscard]] bool useSparseMLA(FmhaParams const& p) const
    {
        return p.use_sparse_attention && mUseTllmGen && mCfg.is_mla_enable;
    }

    [[nodiscard]] bool useTllmGenSparseAttention(FmhaParams const& p) const
    {
        return useSparseMLA(p) || (p.use_sparse_attention && mUseTllmGen && p.hasUnpagedSparseAttnIndices());
    }

    [[nodiscard]] StaticAttentionConfig const& config() const
    {
        return mCfg;
    }

    [[nodiscard]] tensorrt_llm::kernels::MlaMetaParams const& mlaMeta() const
    {
        return mMLAParams;
    }

    [[nodiscard]] int smVersion() const
    {
        return mSM;
    }

    [[nodiscard]] bool supportsNvFp4Output() const
    {
        return mEnableContextFMHA && mEnableXQA;
    }

    [[nodiscard]] int getMultiProcessorCount() const
    {
        return mMultiProcessorCount;
    }

    // ---------------------------------------------------------------------------
    // Op state: the derived, hardware and dispatcher members. Per-call configuration
    // scalars are read off `FmhaParams` directly and are not mirrored here.
    // ---------------------------------------------------------------------------

    int mNumKVHeads = -1;
    int mHeadSize = -1;

    bool mPagedKVCache = true;
    bool mFP8ContextFMHA = false;
    bool mFP8AttenOutput = false;
    bool mFP8ContextMLA = false;
    bool mFP8GenerationMLA = false;
    bool mIsGenerationMLA = false;
    bool mUseGenFlashMLA = false;
    // Static sparse MLA reads a separately dequantized FP8 scratch pool, so an NVFP4
    // paged cache still runs the FP8 kernels.
    bool mUseNvfp4MlaKvCache = false;
    // Skip correction when the row-max increase is within this base-2 threshold.
    float mSkipCorrectionThreshold = 0.0F;

    // Equal to the full head counts: this op always runs on a single rank.
    int mNumAttnHeads = -1;
    // The layer's fixed configuration, captured at construction. Every value the
    // kernel-runner selection and the per-call paths read that does not vary between
    // calls lives here rather than being resent on each FmhaParams.
    StaticAttentionConfig mCfg{};
    tensorrt_llm::kernels::MlaMetaParams mMLAParams{};
    int mNumAttnKVHeads = -1;

    // fmha runner (enabled by default)
    // flag: disabled = 0, enabled = 1, enabled with fp32 accumulation = 2
    bool mEnableContextFMHA = true;
    bool mFMHAForceFP32Acc = false;
    bool mMultiBlockMode = true;
    bool mEnableXQA = true;

    bool mFuseFp4Quant = false;

#ifdef SKIP_SOFTMAX_STAT
    uint32_t* mSkipSoftmaxTotalBlocks;
    uint32_t* mSkipSoftmaxSkippedBlocks;
#endif

private:
    void initialize();
    int finishPrepare(FmhaParams& p, bool isGen);

    static constexpr int kReservedMaxSeqLenTilePerSeq = 64;

    int mSM = tensorrt_llm::common::getSMVersion();
    bool mUseTllmGen = (mSM >= 100) && (mSM != 120) && (mSM != 121);
    bool mForceMultiBlockWarned = false;
    int mMultiProcessorCount = tensorrt_llm::common::getMultiProcessorCount();
    int mMaxSharedMemoryPerBlockOptin = tensorrt_llm::common::getMaxSharedMemoryPerBlockOptin();
    std::shared_ptr<CUDADriverWrapper> mDriver;
    std::unique_ptr<tensorrt_llm::kernels::FusedMHARunnerV2> mDecoderFMHARunner;
    std::unique_ptr<tensorrt_llm::kernels::FmhaDispatcher> mFmhaDispatcher;
    std::unique_ptr<tensorrt_llm::kernels::XqaDispatcher> mXqaDispatcher;
    std::unique_ptr<tensorrt_llm::kernels::TllmGenFmhaRunner> mTllmGenFMHARunner;
    std::unique_ptr<tensorrt_llm::common::CublasMMWrapper> mCublasWrapper;
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
