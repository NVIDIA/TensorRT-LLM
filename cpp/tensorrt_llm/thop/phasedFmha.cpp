/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

#include "tensorrt_llm/thop/phasedFmha.h"

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/attentionWorkspace.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/envUtils.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/common/memoryUtils.h"
#include "tensorrt_llm/kernels/decoderMaskedMultiheadAttention.h"
#include "tensorrt_llm/kernels/decoderMaskedMultiheadAttention/cascadeAttentionKernel.h"
#include "tensorrt_llm/kernels/decoderMaskedMultiheadAttention/decoderXQARunnerUtils.h"
#include "tensorrt_llm/kernels/flashMLA/flash_mla.h"
#include "tensorrt_llm/kernels/gptKernels.h"
#include "tensorrt_llm/kernels/kvCacheUtils.h"
#include "tensorrt_llm/kernels/mlaKernels.h"
#include "tensorrt_llm/kernels/sparseAttentionKernels.h"
#include "tensorrt_llm/kernels/trtllmGenKernels/fmha/fmhaMaskSelection.h"
#include "tensorrt_llm/kernels/unfusedAttentionKernels.h"
#include "tensorrt_llm/thop/attentionVariant.h"

#include <algorithm>
#include <climits>
#include <cstring>
#include <limits>
#include <type_traits>
#include <vector>

using namespace tensorrt_llm::kernels;
namespace tc = tensorrt_llm::common;

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{
using tensorrt_llm::common::op::AttentionContextWorkspaceLayout;
using tensorrt_llm::common::op::AttentionFlashMlaWorkspaceSizes;
using tensorrt_llm::common::op::AttentionGenerationWorkspaceSizes;
using tensorrt_llm::common::op::AttentionWorkspaceManager;
using tensorrt_llm::common::op::AttentionXqaWorkspaceSizes;

template <typename T>
struct SATypeConverter
{
    using Type = T;
};

template <>
struct SATypeConverter<half>
{
    using Type = uint16_t;
};

template <typename T, typename KVCacheBuffer>
struct FusedQKVMaskedAttentionDispatchParams
{
    T const* qkv_buf;
    T const* qkv_bias;
    T const* relative_attention_bias;
    bool const* attention_mask;
    float const* attention_sinks;
    float const* logn_scaling_ptr;
    int const* cache_indir;
    void* context_buf;
    bool const* finished;
    int const* sequence_lengths;
    int max_batch_size;
    int inference_batch_size;
    int beam_width;
    int head_num;
    int kv_head_num;
    int size_per_head;
    int rotary_embedding_dim;
    float rotary_embedding_base;
    RotaryScalingType rotary_embedding_scale_type;
    float rotary_embedding_scale;
    float const* rotary_embedding_inv_freq_cache;
    float2 const* rotary_embedding_cos_sin_cache;
    float rotary_embedding_short_m_scale;
    float rotary_embedding_long_m_scale;
    int rotary_embedding_max_positions;
    int rotary_embedding_original_max_positions;
    int rotary_cogvlm_vision_start;
    int rotary_cogvlm_vision_length;
    PositionEmbeddingType position_embedding_type;
    bool position_shift_enabled;

    int chunked_attention_size;
    int attention_mask_stride;
    int max_attention_window_size;
    int cyclic_attention_window_size;
    int sink_token_length;
    int const* input_lengths;
    int timestep;
    float q_scaling;
    float attn_logit_softcapping_scale;
    int relative_attention_bias_stride;
    T const* linear_bias_slopes;
    int const* ia3_tasks;
    T const* ia3_key_weights;
    T const* ia3_value_weights;
    float const* qkv_scale_out;
    bool fp8_context_fmha;
    float const* attention_out_scale;
    bool mUnfuseQkvGemm;
    tc::QuantMode quant_option;
    bool multi_block_mode;
    int max_seq_len_tile;
    int min_seq_len_tile;
    T* partial_out;
    float* partial_sum;
    float* partial_max;
    int* block_counter;
    // Cascade attention prefix-side workspace (fp32).  Sliced from the same
    // generation workspace and forwarded into Multihead_attention_params, so the
    // cascade fast-path allocates nothing of its own.
    float* cascade_partial_out{};
    float* cascade_partial_max{};
    float* cascade_partial_sum{};
    float const* kv_scale_orig_quant;
    float const* kv_scale_quant_orig;
    tc::QuantMode kv_cache_quant_mode;
    int multi_processor_count;
    KVCacheBuffer kv_block_array;
    KVLinearBuffer shift_k_cache_buffer;
    bool cross_attention = false;
    int const* memory_length_per_sample = nullptr;
    int max_distance = 0;
    bool block_sparse_attention = false;
    BlockSparseParams block_sparse_params;
    int32_t const* mrope_position_deltas;
};

template <typename T, typename KVCacheBuffer>
void RopeFeature::apply(FusedQKVMaskedAttentionDispatchParams<T, KVCacheBuffer>& d) const
{
    d.rotary_embedding_dim = mDim;
    d.position_embedding_type = mType;
    d.rotary_embedding_base = mTheta;
    d.rotary_embedding_scale_type = mScaleType;
    d.rotary_embedding_scale = mScale;
    d.rotary_embedding_inv_freq_cache = mInvFreqCache;
    d.rotary_embedding_cos_sin_cache = mCosSin;
    d.rotary_embedding_short_m_scale = mShortMScale;
    d.rotary_embedding_long_m_scale = mLongMScale;
    d.rotary_embedding_max_positions = mMaxPositions;
    d.rotary_embedding_original_max_positions = mOriginalMaxPositions;
    d.rotary_cogvlm_vision_start = -1;
    d.rotary_cogvlm_vision_length = -1;
    d.mrope_position_deltas = mMropeDeltas;
}

template <typename T, typename KVCacheBuffer>
void RelativePositionFeature::apply(FusedQKVMaskedAttentionDispatchParams<T, KVCacheBuffer>& d) const
{
    d.relative_attention_bias = mRelative ? relativeAttentionBias<T>() : nullptr;
    d.relative_attention_bias_stride = mRelative ? mStride : 0;
    d.max_distance = mMaxDistance;
}

template <typename T, typename KVCacheBuffer>
void CrossAttentionFeature::apply(FusedQKVMaskedAttentionDispatchParams<T, KVCacheBuffer>& d) const
{
    d.cross_attention = mActive;
    d.memory_length_per_sample = mEncoderLengths;
}

template <typename T_MMHA, typename T, typename KVCacheBuffer, bool CROSS_ATTENTION>
void fusedQKV_masked_attention_dispatch(Multihead_attention_params<T_MMHA, CROSS_ATTENTION>& params,
    FusedQKVMaskedAttentionDispatchParams<T, KVCacheBuffer> const& input_params, cudaStream_t stream)
{
    using DataType = typename SATypeConverter<T>::Type;

    // Prepare the parameters.
    params = {};

    int hidden_units = input_params.head_num * input_params.size_per_head;
    int hidden_units_kv = input_params.kv_head_num * input_params.size_per_head;
    if (input_params.qkv_bias != nullptr)
    {
        params.q_bias = reinterpret_cast<DataType const*>(input_params.qkv_bias);
        params.k_bias = reinterpret_cast<DataType const*>(input_params.qkv_bias) + hidden_units;
        params.v_bias = reinterpret_cast<DataType const*>(input_params.qkv_bias) + hidden_units + hidden_units_kv;
    }
    else
    {
        params.q_bias = nullptr;
        params.k_bias = nullptr;
        params.v_bias = nullptr;
    }

    // Set the output buffer.
    params.out = input_params.context_buf;

    // Set the input buffers.
    params.q = reinterpret_cast<DataType const*>(input_params.qkv_buf);
    params.k = reinterpret_cast<DataType const*>(input_params.qkv_buf) + hidden_units;
    params.v = reinterpret_cast<DataType const*>(input_params.qkv_buf) + hidden_units + hidden_units_kv;

    params.int8_kv_cache = input_params.kv_cache_quant_mode.hasInt8KvCache();
    params.fp8_kv_cache = input_params.kv_cache_quant_mode.hasFp8KvCache();
    if (input_params.kv_cache_quant_mode.hasKvCacheQuant())
    {
        params.kv_scale_orig_quant = input_params.kv_scale_orig_quant;
        params.kv_scale_quant_orig = input_params.kv_scale_quant_orig;
    }

    params.stride = hidden_units + 2 * hidden_units_kv;
    params.finished = const_cast<bool*>(input_params.finished);

    params.cache_indir = input_params.cache_indir;
    params.batch_size = input_params.inference_batch_size;
    params.beam_width = input_params.beam_width;
    params.chunked_attention_size = input_params.chunked_attention_size;
    if (input_params.chunked_attention_size != INT_MAX && !tc::getEnvDisableChunkedAttentionInGenPhase())
    {
        TLLM_CHECK_WITH_INFO((input_params.chunked_attention_size & (input_params.chunked_attention_size - 1)) == 0,
            "Attention chunk size should be a power of 2.");
        params.chunked_attention_size_log2 = std::log2(input_params.chunked_attention_size);
    }
    else
    {
        params.chunked_attention_size_log2 = 0;
    }
    params.max_attention_window_size = input_params.max_attention_window_size;
    params.cyclic_attention_window_size = input_params.cyclic_attention_window_size;
    params.sink_token_length = input_params.sink_token_length;
    params.length_per_sample = input_params.sequence_lengths; // max_input_length + current output length
    // timestep for shared memory size calculation and rotary embedding computation
    params.timestep = input_params.timestep;
    params.num_heads = input_params.head_num;
    params.num_kv_heads = input_params.kv_head_num;
    params.hidden_size_per_head = input_params.size_per_head;
    params.rotary_embedding_dim = input_params.rotary_embedding_dim;
    params.rotary_embedding_base = input_params.rotary_embedding_base;
    params.rotary_embedding_scale_type = input_params.rotary_embedding_scale_type;
    params.rotary_embedding_scale = input_params.rotary_embedding_scale;
    params.rotary_embedding_inv_freq_cache = input_params.rotary_embedding_inv_freq_cache;
    params.rotary_embedding_cos_sin_cache = input_params.rotary_embedding_cos_sin_cache;
    params.rotary_embedding_short_m_scale = input_params.rotary_embedding_short_m_scale;
    params.rotary_embedding_long_m_scale = input_params.rotary_embedding_long_m_scale;
    params.rotary_embedding_max_positions = input_params.rotary_embedding_max_positions;
    params.rotary_embedding_original_max_positions = input_params.rotary_embedding_original_max_positions;
    params.rotary_cogvlm_vision_start = input_params.rotary_cogvlm_vision_start;
    params.rotary_cogvlm_vision_length = input_params.rotary_cogvlm_vision_length;
    params.position_embedding_type = input_params.position_embedding_type;
    params.position_shift_enabled = input_params.position_shift_enabled;
    // Note: keep norm factor (sqrt(K_dim)) when adopting megatron T5 structure (may adjust)
    params.inv_sqrt_dh = 1.F / (sqrtf((float) params.hidden_size_per_head) * input_params.q_scaling);
    params.attn_logit_softcapping_scale = input_params.attn_logit_softcapping_scale;
    params.attn_logit_softcapping_inverse_scale = 1.0f / input_params.attn_logit_softcapping_scale;

    params.logn_scaling_ptr = input_params.logn_scaling_ptr;
    params.relative_attention_bias = reinterpret_cast<DataType const*>(input_params.relative_attention_bias);
    params.relative_attention_bias_stride = input_params.relative_attention_bias_stride;
    params.max_distance = input_params.max_distance;
    params.block_sparse_attention = input_params.block_sparse_attention;
    params.block_sparse_params = input_params.block_sparse_params;

    // Attention mask input.
    params.attention_mask = input_params.attention_mask;
    params.attention_mask_stride = input_params.attention_mask_stride;

    // Attention sinks.
    params.attention_sinks = input_params.attention_sinks;

    // The slope of linear position bias per head, e.g., ALiBi.
    if (input_params.linear_bias_slopes != nullptr)
    {
        params.linear_bias_slopes = reinterpret_cast<DataType const*>(input_params.linear_bias_slopes);
    }
    params.input_lengths = input_params.input_lengths;

    params.ia3_tasks = input_params.ia3_tasks;
    params.ia3_key_weights = reinterpret_cast<DataType const*>(input_params.ia3_key_weights);
    params.ia3_value_weights = reinterpret_cast<DataType const*>(input_params.ia3_value_weights);

    if (input_params.quant_option.hasStaticActivationScaling() || input_params.fp8_context_fmha)
    {
        // qkv_scale_out is nullptr currently (no scale).
        params.qkv_scale_quant_orig = input_params.qkv_scale_out;
        TLLM_CHECK_WITH_INFO(!input_params.fp8_context_fmha || input_params.attention_out_scale != nullptr,
            "attention output scale should be provided.");
        params.attention_out_scale_orig_quant = input_params.attention_out_scale;
    }

    params.multi_block_mode = input_params.multi_block_mode;
    // Cascade-attention partials must be wired regardless of multi_block_mode.
    // Cascade decode runs with multi_block disabled (short-decode workloads have
    // max_num_seq_len_tiles == 1, so enable_multi_block is structurally false).
    // Gating these behind multi_block_mode leaves cascade_partial_* null and makes
    // launch_cascade_attention fall back with "cascade workspace not provisioned".
    params.cascade_partial_out = input_params.cascade_partial_out;
    params.cascade_partial_max = input_params.cascade_partial_max;
    params.cascade_partial_sum = input_params.cascade_partial_sum;
    if (input_params.multi_block_mode)
    {
        params.min_seq_len_tile = input_params.min_seq_len_tile;
        params.max_seq_len_tile = input_params.max_seq_len_tile;

        params.partial_out = reinterpret_cast<DataType*>(input_params.partial_out);
        params.partial_sum = input_params.partial_sum;
        params.partial_max = input_params.partial_max;

        params.block_counter = input_params.block_counter;
    }

    params.multi_processor_count = input_params.multi_processor_count;

    // cross attn
    params.memory_length_per_sample = input_params.memory_length_per_sample;

    params.mrope_position_deltas = input_params.mrope_position_deltas;
    sync_check_cuda_error(stream);

    masked_multihead_attention(params, input_params.kv_block_array, input_params.shift_k_cache_buffer, stream);
}

#define INSTANTIATE_MMHA_DISPATCH(T_MMHA, T)                                                                           \
    template void fusedQKV_masked_attention_dispatch(Multihead_attention_params<T_MMHA, false>&,                       \
        FusedQKVMaskedAttentionDispatchParams<T, KVLinearBuffer> const&, cudaStream_t stream);                         \
    template void fusedQKV_masked_attention_dispatch(Multihead_attention_params<T_MMHA, true>&,                        \
        FusedQKVMaskedAttentionDispatchParams<T, KVLinearBuffer> const&, cudaStream_t stream);                         \
    template void fusedQKV_masked_attention_dispatch(Multihead_attention_params<T_MMHA, false>&,                       \
        FusedQKVMaskedAttentionDispatchParams<T, KVBlockArray> const&, cudaStream_t stream);                           \
    template void fusedQKV_masked_attention_dispatch(Multihead_attention_params<T_MMHA, true>&,                        \
        FusedQKVMaskedAttentionDispatchParams<T, KVBlockArray> const&, cudaStream_t stream);
INSTANTIATE_MMHA_DISPATCH(float, float)
INSTANTIATE_MMHA_DISPATCH(uint16_t, half)
#ifdef ENABLE_BF16
INSTANTIATE_MMHA_DISPATCH(__nv_bfloat16, __nv_bfloat16)
#endif
#undef INSTANTIATE_MMHA_DISPATCH

namespace
{

/// Calls \p fn with a `std::type_identity` of the activation type of \p scalarType.
template <typename Fn>
void dispatchOnDtype(at::ScalarType scalarType, Fn&& fn)
{
    switch (scalarType)
    {
    case torch::kFloat32: fn(std::type_identity<float>{}); break;
    case torch::kFloat16: fn(std::type_identity<half>{}); break;
#ifdef ENABLE_BF16
    case torch::kBFloat16: fn(std::type_identity<__nv_bfloat16>{}); break;
#endif
    default: TLLM_THROW("Unsupported attention dtype");
    }
}

template <typename T>
constexpr inline T divUp(T a, T b)
{
    return (a + b - 1) / b;
}

template <typename T>
constexpr inline T roundUp(T a, T b)
{
    return divUp(a, b) * b;
}

// Persistent is the default for performance. TRTLLM_GEN_FMHA_CONTEXT_STATIC_TILE_SCHEDULER=1 selects the static
// variant instead: the sm107 persistent context kernel can hang in a tcgen05 TMEM-dealloc guardrail trap
// (observed with the MLA HQk192/HV128 SeparateQkv kernel on Kimi-K3 prefill). Kernel support check and launch
// must agree on the scheduler, so both go through this helper.
TileScheduler contextTileScheduler()
{
    return tensorrt_llm::common::getEnvUseStaticTileSchedulerForTrtllmGenContextFmha() ? TileScheduler::Static
                                                                                       : TileScheduler::Persistent;
}

QkvLayout AttentionInputLayoutToQkvLayout(AttentionInputLayout layout)
{
    if (layout == AttentionInputLayout::PACKED_QKV)
    {
        return QkvLayout::PackedQkv;
    }
    else if (layout == AttentionInputLayout::Q_CONTIGUOUS_KV)
    {
        return QkvLayout::ContiguousKv;
    }
    else if (layout == AttentionInputLayout::Q_PAGED_KV)
    {
        return QkvLayout::PagedKv;
    }
    else if (layout == AttentionInputLayout::SEPARATE_Q_K_V)
    {
        return QkvLayout::SeparateQkv;
    }
    TLLM_CHECK_WITH_INFO(false, "Unexpected AttentionInputLayout");
    return QkvLayout::SeparateQkv;
}

/// The trtllm-gen context runner for \p fixedParams.
std::unique_ptr<TllmGenFmhaRunner> makeTrtllmGenContextRunner(MHARunnerFixedParams const& fixedParams)
{
    auto [dataTypeK, dataTypeV] = unpack_kv_data_type(fixedParams.dataTypeKv);
    return std::make_unique<TllmGenFmhaRunner>(fixedParams.dataType, dataTypeK, dataTypeV, fixedParams.dataTypeOut,
        fixedParams.sageBlockSizeQ, fixedParams.sageBlockSizeK, 0, fixedParams.sageBlockSizeV,
        fixedParams.fusesDsv4InvRopeFp8Quant);
}

/// Context configurations fmha_v2 serves even on SM100f: head size 72, which trtllm-gen has no
/// kernels for, and FP16/BF16 full or causal packed-QKV head size 104. Mixed Q/KV precisions stay
/// on trtllm-gen: fmha_v2 requires matching types.
bool prefersFmhaV2Context(MHARunnerFixedParams const& fixedParams)
{
    if (fixedParams.headSize == 72)
    {
        return true;
    }
    return fixedParams.headSize == 104 && fixedParams.headSizeV == 104
        && fixedParams.attentionInputLayout == AttentionInputLayout::PACKED_QKV
        && fixedParams.dataType == fixedParams.dataTypeKv
        && (fixedParams.dataType == DATA_TYPE_FP16 || fixedParams.dataType == DATA_TYPE_BF16)
        && (fixedParams.attentionMaskType == ContextAttentionMaskType::PADDING
            || fixedParams.attentionMaskType == ContextAttentionMaskType::CAUSAL);
}

/// Whether \p runner has a context kernel for \p fixedParams.
bool isTrtllmGenContextSupported(TllmGenFmhaRunner const& runner, MHARunnerFixedParams const& fixedParams)
{
    // The (0, 0) sliding-window probe is rejected as Causal when the window covers the whole KV sequence
    // (leftSlidingWindow >= kvSeqLen - 1), so the probe needs a KV length of at least 2.
    constexpr int32_t kWindowProbeKvSeqLen = 2;

    if (fixedParams.attentionMaskType == ContextAttentionMaskType::CUSTOM_MASK)
    {
        TLLM_LOG_WARNING("TRTLLM-GEN does not support custom mask.");
        return false;
    }
    if (fixedParams.hasAlibi)
    {
        TLLM_LOG_WARNING("TRTLLM-GEN does not support ALiBi.");
        return false;
    }
    if (fixedParams.isSPadded)
    {
        TLLM_LOG_WARNING("TRTLLM-GEN does not support padded inputs.");
        return false;
    }

    auto qkvLayout = AttentionInputLayoutToQkvLayout(fixedParams.attentionInputLayout);
    // Create TllmGenFmhaRunnerParams based on MHARunnerFixedParams. Only fill necessary
    // attributes for kernel selection.
    TllmGenFmhaRunnerParams tllmRunnerParams{};
    tllmRunnerParams.mQkvLayout = qkvLayout;
    // A size of 1 becomes the (0, 0) inclusive sliding-window probe after converting size to reach.
    int32_t const probeSlidingWindowSize
        = fixedParams.attentionMaskType == ContextAttentionMaskType::SLIDING_OR_CHUNKED_CAUSAL ? 1 : 0;
    applyTrtllmGenMaskSelection(tllmRunnerParams,
        tensorrt_llm::kernels::detail::selectTrtllmGenMask(
            fixedParams.attentionMaskType, probeSlidingWindowSize, 0, kWindowProbeKvSeqLen));
    tllmRunnerParams.mKernelType = FmhaKernelType::Context;
    tllmRunnerParams.mTileScheduler = contextTileScheduler();
    tllmRunnerParams.mMultiCtasKvMode = false;
    tllmRunnerParams.mNumHeadsQ = fixedParams.numQHeads;
    tllmRunnerParams.mNumHeadsKv = fixedParams.numKvHeads;
    tllmRunnerParams.mHeadDimQkNope = fixedParams.headSizeQkNope;
    tllmRunnerParams.mBatchSize = 1;
    tllmRunnerParams.mMaxSeqLenQ = 1;
    tllmRunnerParams.mMaxSeqLenKv = kWindowProbeKvSeqLen;
    tllmRunnerParams.mMaxSeqLenCacheKv = kWindowProbeKvSeqLen;
    tllmRunnerParams.mSumOfSeqLensQ = 1;
    tllmRunnerParams.mSumOfSeqLensKv = kWindowProbeKvSeqLen;
    tllmRunnerParams.mMaxNumPagesPerSeqKv = 1;
    // Assume same headDim for Qk and V here.
    tllmRunnerParams.mHeadDimQk = fixedParams.headSize;
    tllmRunnerParams.mHeadDimV = fixedParams.headSizeV;
    tllmRunnerParams.mNumTokensPerPage = (qkvLayout == QkvLayout::PagedKv) ? fixedParams.numTokensPerBlock : 0;
    tllmRunnerParams.mNumHeadsQPerKv = fixedParams.numQHeads / fixedParams.numKvHeads;
    tllmRunnerParams.mMultiProcessorCount = tc::getMultiProcessorCount();
    // Sparse context attention uses a generation-style kernel with per-token sparse indices.
    if (fixedParams.useTllmGenSparseAttention)
    {
        // Sparse selection needs a positive TopK, aligned to four, even for this synthetic probe.
        tllmRunnerParams.mSparseTopK = 4;
        tllmRunnerParams.mSparseAttention = (fixedParams.useSparseMLA && fixedParams.headSizeV == fixedParams.headSize)
            ? SparseType::DynamicTokenSparse
            : SparseType::StaticTokenSparse;
        tllmRunnerParams.mDsv4EpilogueFusion.enabled = fixedParams.fusesDsv4InvRopeFp8Quant;
        tllmRunnerParams.mKernelType = FmhaKernelType::Generation;
        tllmRunnerParams.mMaskType = TrtllmGenAttentionMaskType::Causal;
        // Generation-style kernels on long KV can pick MultiCtasKv cubins
        tllmRunnerParams.mMultiCtasKvMode = true;
    }
    tllmRunnerParams.mUsesSpcompress = fixedParams.useSpcompress;

    return runner.isSupported(tllmRunnerParams);
}

/// The fmha_v2 context runner for \p fixedParams, or null when it has no kernel for them.
std::unique_ptr<FusedMHARunnerV2> makeFmhaV2ContextRunner(MHARunnerFixedParams fixedParams)
{
    // SageAttention sets different QK/V datatypes. Unlike trtllmGen, the fmha_v2 kernelMetaInfo
    // carries one input datatype. Collapse pair to the composite type where it's registered.
    if (fixedParams.dataType == DATA_TYPE_INT8 && fixedParams.dataTypeKv == DATA_TYPE_KV_INT8_E4M3)
    {
        fixedParams.dataType = DATA_TYPE_KV_INT8_E4M3;
    }

    TLLM_CHECK_WITH_INFO(fixedParams.dataType == fixedParams.dataTypeKv,
        "KV cache data type %s is not the same as input data type %s.",
        data_type_to_string(fixedParams.dataTypeKv).c_str(), data_type_to_string(fixedParams.dataType).c_str());

    if (fixedParams.sageBlockSizeQ > 0 || fixedParams.sageBlockSizeK > 0 || fixedParams.sageBlockSizeV > 0)
    {
        // This backend implements SageAttention on SM90 only, with one kernel, for block
        // sizes (2, 16, 1) and separate Q/K/V.
        int const sm = tensorrt_llm::common::getSMVersion();
        if (sm != kSM_90)
        {
            TLLM_LOG_WARNING("fmha_v2 implements SageAttention on SM90 only, got sm_%d.", sm);
            return nullptr;
        }
        if (fixedParams.sageBlockSizeQ != 2 || fixedParams.sageBlockSizeK != 16 || fixedParams.sageBlockSizeV != 1)
        {
            TLLM_LOG_WARNING(
                "SageAttention on SM90 supports exactly one block-size combination, (q, k, v) = (2, 16, 1), "
                "got (%d, %d, %d).",
                fixedParams.sageBlockSizeQ, fixedParams.sageBlockSizeK, fixedParams.sageBlockSizeV);
            return nullptr;
        }
    }
    auto runner = std::make_unique<FusedMHARunnerV2>(fixedParams);
    if (!runner->isFmhaSupported())
    {
        return nullptr;
    }
    return runner;
}

/// The trtllm-gen generation kernels' per-call limits: everything else falls back to MMHA.
bool isTrtllmGenGenerationUsable(
    FmhaParams const& p, bool crossAttention, bool pagedKvCache, int tokensPerBlock, int numQHeads, int numKvHeads)
{
    if (p.meta.effective_beam_width > 1)
    {
        TLLM_LOG_DEBUG("Fallback to MMHA as beam searching is not supported by TRTLLM-GEN kernels.");
        return false;
    }
    if (crossAttention && !pagedKvCache)
    {
        TLLM_LOG_DEBUG(
            "Fallback to MMHA as cross attention without paged KV Cache is not supported by TRTLLM-GEN kernels.");
        return false;
    }
    if (pagedKvCache && tokensPerBlock < 8)
    {
        TLLM_LOG_DEBUG("Fallback to MMHA as tokens_per_block < 8 is not supported by TRTLLM-GEN kernels.");
        return false;
    }
    if (p.cyclic_attention_window_size != p.max_attention_window_size)
    {
        TLLM_LOG_DEBUG("Fallback to MMHA as variable attention_window_size is not supported by TRTLLM-GEN kernels.");
        return false;
    }
    if ((static_cast<float>(numQHeads) / static_cast<float>(numKvHeads)) > 32)
    {
        TLLM_LOG_DEBUG("Fallback to MMHA as num_q_heads per kv_head > 32 is not supported by TRTLLM-GEN kernels.");
        return false;
    }
    return true;
}

/// The KV-cache element type of \p quantMode, with \p inputDataType for an unquantized cache.
Data_type kvCacheDataType(tc::QuantMode quantMode, Data_type inputDataType)
{
    if (quantMode.hasInt8KvCache())
    {
        return DATA_TYPE_INT8;
    }
    if (quantMode.hasFp8KvCache())
    {
        return DATA_TYPE_E4M3;
    }
    if (quantMode.hasFp4KvCache())
    {
        return DATA_TYPE_E2M1;
    }
    return inputDataType;
}

/// The trtllm-gen generation workspace for \p p's batch bound: the decoder info, the preprocessed Q
/// and the BMM scales. The multi-CTA scratch follows at `trtllmGenScratchOffset()`, sized for one
/// 256-row partial block per SM and at least the trtllm-gen workspace the other trtllm-gen callers use.
TrtllmGenGenerationWorkspaceLayout trtllmGenGenerationLayout(FmhaParams const& p, int64_t numHeads, int64_t numKvHeads,
    int64_t headSize, int64_t rotaryEmbeddingDim, bool useSparseAttention)
{
    return TrtllmAttentionWorkspaceManager::buildGenerationLayout(p.getQueryInput().scalar_type(),
        p.getMaxNumSequences(), p.num_gen_tokens, numHeads, headSize, rotaryEmbeddingDim, numKvHeads,
        p.getMaxBlocksPerSequence(), useSparseAttention, /*skipFmhaWorkspace=*/true);
}

std::size_t trtllmGenScratchOffset(TrtllmGenGenerationWorkspaceLayout const& layout)
{
    return tc::alignSize(static_cast<std::size_t>(layout.totalSize),
        static_cast<std::size_t>(TrtllmAttentionWorkspaceManager::kWorkspaceAlignment));
}

template <typename P>
P* workspacePtr(void* workspace, int64_t offset)
{
    return offset < 0 ? nullptr : reinterpret_cast<P*>(static_cast<int8_t*>(workspace) + offset);
}

/// Whether trtllm-gen is the GPU's kernel family: SM100f.
bool isTrtllmGenGpu()
{
    return tc::isSM100Family();
}

bool isHalfOrBf16(tensorrt_llm::DataType type)
{
    return type == tensorrt_llm::DataType::kHALF || type == tensorrt_llm::DataType::kBF16;
}

} // namespace

int xqaKernelWorkspaceAlignment(int headSize, int numQHeads, int numKvHeads, bool multiBlockMode)
{
    constexpr size_t kXQA_OUT_ELEM_SIZE = 2; // fp16 or bf16.
    constexpr int kMaxBeamWidth = 4;
    int group_size = numQHeads / numKvHeads;
    int32_t const multi_block_workspace_alignment
        = roundUp<int32_t>(kXQA_OUT_ELEM_SIZE * kMaxBeamWidth * group_size * headSize, 128);
    return multiBlockMode ? multi_block_workspace_alignment : 128;
}

std::size_t xqaKernelWorkspaceSize(
    int headSize, int numQHeads, int numKvHeads, bool isMLA, bool multiBlockMode, int max_num_tokens)
{
    // buffer for RoPE / output quantization.
    constexpr size_t kXQA_OUT_ELEM_SIZE = 2; // fp16 or bf16.
    constexpr int kMaxBeamWidth = 4;
    size_t workspace_size = roundUp<size_t>(kXQA_OUT_ELEM_SIZE * headSize * numQHeads * max_num_tokens, 128); // rope
    // output conversion.
    workspace_size = roundUp<size_t>(workspace_size + kXQA_OUT_ELEM_SIZE * headSize * numQHeads * max_num_tokens, 128);
    workspace_size = roundUp<size_t>(workspace_size, 128) + xqaMlaCgaXBufSize * max_num_tokens;
    if (multiBlockMode)
    {
        size_t workspaces[4];
        size_t const nbSubSeq = getXqaMaxNumSubSeq(isMLA);
        size_t const nbSeq = nbSubSeq / 2;
        int group_size = numQHeads / numKvHeads;
        workspaces[0] = sizeof(uint32_t) * nbSeq;                           // semaphores
        workspaces[1] = sizeof(float) * roundUp(group_size, 32) * nbSubSeq; // rowMax
        workspaces[2] = sizeof(float) * roundUp(group_size, 32) * nbSubSeq; // rowSum
        size_t const multi_block_workspace_alignment
            = roundUp<size_t>(kXQA_OUT_ELEM_SIZE * kMaxBeamWidth * group_size * headSize, 128);
        workspaces[3] = multi_block_workspace_alignment * nbSubSeq;
        workspace_size = roundUp<size_t>(workspace_size, multi_block_workspace_alignment)
            + roundUp(workspaces[0], multi_block_workspace_alignment)
            + roundUp(workspaces[1], multi_block_workspace_alignment)
            + roundUp(workspaces[2], multi_block_workspace_alignment)
            + roundUp(workspaces[3], multi_block_workspace_alignment)
            + multi_block_workspace_alignment; // extra space reserved for alignment
    }
    return workspace_size;
}

// ---------------------------------------------------------------------------
// TrtllmGenFmha
// ---------------------------------------------------------------------------

template <typename AttentionVariant>
bool TrtllmGenFmha<AttentionVariant>::isAvailable(AttentionBase const& /*key*/)
{
    return isTrtllmGenGpu();
}

template <typename AttentionVariant>
bool TrtllmGenFmha<AttentionVariant>::isContextSupported(MHARunnerFixedParams const& fixedParams)
{
    return isTrtllmGenContextSupported(*makeTrtllmGenContextRunner(fixedParams), fixedParams);
}

template <typename AttentionVariant>
std::size_t TrtllmGenFmha<AttentionVariant>::getMultiCtasKvScratchSize(
    std::size_t headDimV, int multiProcessorCount) noexcept
{
    static constexpr std::size_t kMultiCtasKvRowsPerCta = 256;
    static constexpr std::size_t kMultiCtasKvStatsPerRow = 2;
    static constexpr std::size_t kMultiCtasKvPartialOElementSize = 2;

    std::size_t const maxRows = kMultiCtasKvRowsPerCta * static_cast<std::size_t>(multiProcessorCount);
    std::size_t const partialStatsSize = sizeof(float) * kMultiCtasKvStatsPerRow * maxRows;
    std::size_t const partialOSize = kMultiCtasKvPartialOElementSize * maxRows * headDimV;

    return partialStatsSize + partialOSize;
}

template <typename AttentionVariant>
bool TrtllmGenFmha<AttentionVariant>::isSupported(FmhaParams const& p, bool isGen) const
{
    if (!isGen)
    {
        return mContextRunner != nullptr;
    }
    if (mGenerationRunner == nullptr)
    {
        return false;
    }
    if constexpr (std::is_same_v<AttentionVariant, MLA>)
    {
        return true;
    }
    else
    {
        auto const& key = this->attention;
        return isTrtllmGenGenerationUsable(p, key.cross_attention, key.use_kv_cache,
            static_cast<int>(key.tokens_per_block), key.mNumAttnHeads, key.mNumAttnKVHeads);
    }
}

template <typename AttentionVariant>
void TrtllmGenFmha<AttentionVariant>::initContext(AttentionVariant const& op)
{
    if (!isTrtllmGenGpu())
    {
        return;
    }
    if constexpr (std::is_same_v<AttentionVariant, MHA>)
    {
        if (!isHalfOrBf16(op.type) || op.isRelativePosition() || op.isQuantizedCrossCacheUnfused())
        {
            return;
        }
    }
    mContextParams = op.contextFixedParams();
    if (prefersFmhaV2Context(mContextParams))
    {
        return;
    }
    mContextRunner = makeTrtllmGenContextRunner(mContextParams);
    if (!isTrtllmGenContextSupported(*mContextRunner, mContextParams))
    {
        TLLM_LOG_WARNING("TRTLLM-GEN does not support the requested context kernels for %s in sm_%d.",
            mContextParams.convertToStrOutput().c_str(), op.smVersion());
        mContextRunner.reset();
    }
}

template <typename AttentionVariant>
void TrtllmGenFmha<AttentionVariant>::initGeneration(AttentionVariant const& op)
{
    if constexpr (std::is_same_v<AttentionVariant, MLA>)
    {
        if (!tc::isSM100Family(op.smVersion()))
        {
            return;
        }
        Data_type qDataType = DATA_TYPE_FP32;
        Data_type kvDataType = DATA_TYPE_FP32;
        Data_type outputDataType = DATA_TYPE_FP32;
        if (op.type == tensorrt_llm::DataType::kHALF)
        {
            qDataType = DATA_TYPE_FP16;
            kvDataType = DATA_TYPE_FP16;
            outputDataType = DATA_TYPE_FP16;
        }
        else if (op.type == tensorrt_llm::DataType::kBF16)
        {
            qDataType = DATA_TYPE_BF16;
            kvDataType = DATA_TYPE_BF16;
            outputDataType = DATA_TYPE_BF16;
        }
        else
        {
            TLLM_CHECK_WITH_INFO(false, "The data type is not supported.");
        }
        if (op.mFP8GenerationMLA)
        {
            qDataType = DATA_TYPE_E4M3;
            kvDataType = DATA_TYPE_E4M3;
        }
        if (op.fuses_dsv4_inv_rope_fp8_quant)
        {
            outputDataType = DATA_TYPE_E4M3;
        }
        mGenerationRunner = std::make_unique<TllmGenFmhaRunner>(
            qDataType, kvDataType, kvDataType, outputDataType, 0, 0, 0, 0, op.fuses_dsv4_inv_rope_fp8_quant);
    }
    else
    {
        if (!isTrtllmGenGpu() || !isHalfOrBf16(op.type) || !op.use_kv_cache)
        {
            return;
        }
        Data_type const inputDataType = op.type == tensorrt_llm::DataType::kHALF ? DATA_TYPE_FP16 : DATA_TYPE_BF16;
        Data_type const kvDataType = kvCacheDataType(op.quant_mode, inputDataType);
        // The BMMs run in FP8 over FP8 and NVFP4 caches.
        Data_type const mathDataType
            = (kvDataType == DATA_TYPE_E4M3 || kvDataType == DATA_TYPE_E2M1) ? DATA_TYPE_E4M3 : inputDataType;
        Data_type const outputDataType
            = op.is_fp4_out ? DATA_TYPE_E2M1 : (op.is_fp8_out ? DATA_TYPE_E4M3 : inputDataType);
        // The preprocessing kernel will convert Q from inputDataType to fp8 if the kv cache dtype e4m3 or e2m1,
        // as both the NVFP4 KV kernels and FP8 KV kernels uses FP8 input for Q.
        mGenerationQDataType = mathDataType;
        mGenerationRunner
            = std::make_unique<TllmGenFmhaRunner>(mGenerationQDataType, kvDataType, kvDataType, outputDataType);
        if (op.isALiBi())
        {
            TLLM_LOG_WARNING("TRTLLM-GEN does not support ALiBi.");
            mGenerationRunner.reset();
            return;
        }

        // Probe for a kernel; the launch uses the real sizes.
        bool const isSpecDecoding = op.is_spec_decoding_enabled;
        TllmGenFmhaRunnerParams tllmRunnerParams{};
        tllmRunnerParams.mQkvLayout = QkvLayout::PagedKv;
        tllmRunnerParams.mMaskType
            = isSpecDecoding ? TrtllmGenAttentionMaskType::Custom : TrtllmGenAttentionMaskType::Causal;
        tllmRunnerParams.mIsSpecDecTree = isSpecDecoding;
        tllmRunnerParams.mSpecDecodingTargetMaxGenLen = static_cast<int>(op.spec_decoding_target_max_gen_len);
        tllmRunnerParams.mKernelType = FmhaKernelType::Generation;
        tllmRunnerParams.mTileScheduler = TileScheduler::Static;
        tllmRunnerParams.mMultiCtasKvMode = true;
        // Assume same head size for Qk and V here.
        tllmRunnerParams.mHeadDimQk = op.mHeadSize;
        tllmRunnerParams.mHeadDimV = op.mHeadSize;
        tllmRunnerParams.mNumHeadsQ = op.mNumAttnHeads;
        tllmRunnerParams.mNumHeadsKv = op.mNumAttnKVHeads;
        tllmRunnerParams.mNumHeadsQPerKv = op.mNumAttnHeads / op.mNumAttnKVHeads;
        tllmRunnerParams.mNumTokensPerPage = static_cast<int>(op.tokens_per_block);
        tllmRunnerParams.mChunkedAttentionSize = 0;
        tllmRunnerParams.mLeftSlidingWindow = -1;
        tllmRunnerParams.mRightSlidingWindow = -1;
        // Problem size fields used by FMHA kernel selection (computeNumCtas). Must be non-zero to avoid
        // integer divide-by-zero when mMultiCtasKvMode is true. Actual launch uses real sizes.
        tllmRunnerParams.mBatchSize = 1;
        tllmRunnerParams.mMaxSeqLenQ = 1;
        tllmRunnerParams.mMaxSeqLenKv = 1;
        tllmRunnerParams.mMultiProcessorCount = op.getMultiProcessorCount();
        // Sparse MQA/GQA uses trtllm-gen sparse kernel.
        if (op.use_sparse_attention && op.use_tllm_gen_sparse_attention)
        {
            tllmRunnerParams.mSparseAttention = SparseType::StaticTokenSparse;
        }
        auto [isSupported, info] = mGenerationRunner->isSupportedWithInfo(tllmRunnerParams);
        if (!isSupported)
        {
            TLLM_LOG_WARNING("TRTLLLM-Gen kernels are not selected: " + info);
            mGenerationRunner.reset();
        }
    }
}

template <typename AttentionVariant>
std::size_t TrtllmGenFmha<AttentionVariant>::getWorkspaceSize(
    AttentionVariant const& op, FmhaParams const& p, bool isGen) const
{
    if (!isGen)
    {
        return mContextRunner == nullptr ? 0 : op.getWorkspaceSizeForContext(p);
    }
    if (mGenerationRunner == nullptr)
    {
        return 0;
    }
    if constexpr (std::is_same_v<AttentionVariant, MHA>)
    {
        if (p.num_gen_tokens == 0)
        {
            return 0;
        }
        auto const layout = trtllmGenGenerationLayout(p, op.mNumAttnHeads, op.mNumAttnKVHeads, op.mHeadSize,
            op.rotary_embedding_dim, op.useTllmGenSparseAttentionPaged(p));
        return trtllmGenScratchOffset(layout)
            + std::max(static_cast<std::size_t>(TrtllmAttentionWorkspaceManager::kTrtllmGenWorkspaceSize),
                getMultiCtasKvScratchSize(op.getHeadSize(), op.mMultiProcessorCount));
    }
    else
    {
        return op.getGenerationWorkspaceSize(p, /*kernelWorkspaceSize=*/0);
    }
}

template <typename AttentionVariant>
void TrtllmGenFmha<AttentionVariant>::runContext(AttentionVariant const& op, FmhaParams const& p, cudaStream_t stream)
{
    dispatchOnDtype(p.getQueryInput().scalar_type(),
        [&]<typename T>(std::type_identity<T>)
        {
            op.template enqueueFusedContext<T>(
                p, stream, [this](MHARunnerParams const& runnerParams) { launchContext(runnerParams); });
        });
}

template <typename AttentionVariant>
void TrtllmGenFmha<AttentionVariant>::launchContext(MHARunnerParams const& runnerParams)
{
    TLLM_LOG_DEBUG("Running TRTLLM-GEN context FMHA kernel.");
    // Convert from MHAFixedParams + MHARunnerParams to TllmGenFmhaRunnerParams
    void const* kvPoolPtr = nullptr;
    void const* kvSfPoolPtr = nullptr;
    void const* kvPageIdxPtr = nullptr;
    auto qkvLayout = kernels::QkvLayout::PackedQkv;
    int32_t maxBlocksPerSeq = 0;
    int32_t numTokensPerBlock = 0;
    if (mContextParams.attentionInputLayout == AttentionInputLayout::Q_PAGED_KV)
    {
        qkvLayout = kernels::QkvLayout::PagedKv;
        auto pagedKvCache = runnerParams.pagedKvCache.copyKVBlockArrayForContextFMHA();
        kvPoolPtr = pagedKvCache.mPrimaryPoolPtr;
        kvSfPoolPtr = runnerParams.pagedKvSfCache.mPrimaryPoolPtr;
        kvPageIdxPtr = reinterpret_cast<int const*>(pagedKvCache.data);
        maxBlocksPerSeq = pagedKvCache.mMaxBlocksPerSeq;
        numTokensPerBlock = pagedKvCache.mTokensPerBlock;
    }
    else if (mContextParams.attentionInputLayout == AttentionInputLayout::SEPARATE_Q_K_V)
    {
        qkvLayout = kernels::QkvLayout::SeparateQkv;
    }

    TllmGenFmhaRunnerParams tllmRunnerParams{};

    // Parameters to select kernels.
    tllmRunnerParams.mQkvLayout = qkvLayout;
    tllmRunnerParams.mKernelType = FmhaKernelType::Context;
    // Persistent scheduler by default for better performance; see contextTileScheduler().
    tllmRunnerParams.mTileScheduler = contextTileScheduler();
    tllmRunnerParams.mMultiCtasKvMode = false;

    tllmRunnerParams.qPtr = runnerParams.qPtr;
    tllmRunnerParams.kPtr = runnerParams.kPtr;
    tllmRunnerParams.vPtr = runnerParams.vPtr;
    tllmRunnerParams.mVTokenStrideInBytes = runnerParams.vStrideInBytes;
    tllmRunnerParams.kvPtr = kvPoolPtr;
    tllmRunnerParams.kvSfPtr = kvSfPoolPtr;
    tllmRunnerParams.qkvPtr = runnerParams.qkvPtr;
    tllmRunnerParams.attentionSinksPtr = runnerParams.attentionSinksPtr;
    tllmRunnerParams.cumSeqLensQPtr = reinterpret_cast<int const*>(runnerParams.cuQSeqLenPtr);
    tllmRunnerParams.cumSeqLensKvPtr = reinterpret_cast<int const*>(runnerParams.cuKvSeqLenPtr);
    tllmRunnerParams.variableWindowTokenStartsPtr = runnerParams.variableWindowTokenStartsPtr;
    tllmRunnerParams.variableWindowTokenEndsPtr = runnerParams.variableWindowTokenEndsPtr;
    // Attention scales device pointers (only fp8 kernels need to load scales from the device memory).
    tllmRunnerParams.outputScalePtr = reinterpret_cast<float const*>(runnerParams.scaleBmm2Ptr);
    // TRTLLM-GEN kernels always use the Log2 scale
    tllmRunnerParams.scaleSoftmaxLog2Ptr = runnerParams.scaleBmm1Ptr
        ? reinterpret_cast<float const*>(runnerParams.scaleBmm1Ptr + kIdxScaleSoftmaxLog2Ptr)
        : nullptr;
    tllmRunnerParams.kvPageIdxPtr = reinterpret_cast<int const*>(kvPageIdxPtr);
    tllmRunnerParams.oSfScalePtr = runnerParams.oSfScalePtr;
    tllmRunnerParams.oPtr = runnerParams.outputPtr;
    tllmRunnerParams.oSfPtr = runnerParams.outputSfPtr;
    if (runnerParams.dsv4EpilogueFusion.enabled)
    {
        tllmRunnerParams.mDsv4EpilogueFusion.enabled = true;
        tllmRunnerParams.mDsv4EpilogueFusion.cosSinCache = runnerParams.dsv4EpilogueFusion.cosSinCache;
        tllmRunnerParams.mDsv4EpilogueFusion.scaleBufM = runnerParams.dsv4EpilogueFusion.scaleBufM;
    }
    // The sequence lengths for K/V.
    tllmRunnerParams.seqLensKvPtr = reinterpret_cast<int const*>(runnerParams.kvSeqLenPtr);
    // Assume same headDim for Qk and V here.
    tllmRunnerParams.mHeadDimQk = mContextParams.headSize;
    tllmRunnerParams.mHeadDimV = mContextParams.headSizeV;
    tllmRunnerParams.mHeadDimQkNope = mContextParams.headSizeQkNope;
    tllmRunnerParams.mNumHeadsQ = mContextParams.numQHeads;
    tllmRunnerParams.mNumHeadsKv = mContextParams.numKvHeads;
    tllmRunnerParams.mNumHeadsQPerKv = tllmRunnerParams.mNumHeadsQ / tllmRunnerParams.mNumHeadsKv;
    tllmRunnerParams.mBatchSize = runnerParams.b;
    // It is used to construct contiguous kv cache TMA descriptors.
    tllmRunnerParams.mMaxSeqLenCacheKv = runnerParams.slidingWindowSize;
    tllmRunnerParams.mMaxSeqLenQ = runnerParams.qSeqLen;
    tllmRunnerParams.mMaxSeqLenKv = runnerParams.kvSeqLen;
    applyTrtllmGenMaskSelection(tllmRunnerParams,
        tensorrt_llm::kernels::detail::selectTrtllmGenMask(mContextParams.attentionMaskType,
            runnerParams.slidingWindowSize, runnerParams.chunkedAttentionSize, runnerParams.kvSeqLen));
    tllmRunnerParams.mSumOfSeqLensQ = runnerParams.totalQSeqLen;
    tllmRunnerParams.mSumOfSeqLensKv = runnerParams.totalKvSeqLen;
    tllmRunnerParams.mJITWarmup = runnerParams.trtllmGenJITWarmup;
    tllmRunnerParams.mJITWarmupMaxNumRequests = runnerParams.trtllmGenJITWarmupMaxNumRequests;
    tllmRunnerParams.mJITWarmupMaxSeqLenQ = runnerParams.trtllmGenJITWarmupMaxSeqLenQ;
    tllmRunnerParams.mJITWarmupMaxSeqLenKv = runnerParams.trtllmGenJITWarmupMaxSeqLenKv;
    tllmRunnerParams.mMaxNumPagesPerSeqKv = maxBlocksPerSeq;
    tllmRunnerParams.mNumTokensPerPage = (qkvLayout == QkvLayout::PagedKv) ? numTokensPerBlock : 0;
    tllmRunnerParams.mScaleQ = mContextParams.qScaling;
    // Set it to INT_MAX as the kv cache pageOffsets will ensure that there is no out-of-bounds access.
    tllmRunnerParams.mNumPagesInMemPool = INT_MAX;
    tllmRunnerParams.mMultiProcessorCount = this->attention.getMultiProcessorCount();
    tllmRunnerParams.mSfStartTokenIdx = 0;
    // SageAttention scaling factors.
    tllmRunnerParams.sageAttnSfsQPtr = runnerParams.qScalePtr;
    tllmRunnerParams.sageAttnSfsKPtr = runnerParams.kScalePtr;
    tllmRunnerParams.sageAttnSfsPPtr = nullptr;
    tllmRunnerParams.sageAttnSfsVPtr = runnerParams.vScalePtr;
    // For mla chunked prefill
    tllmRunnerParams.softmaxStatsPtr = reinterpret_cast<float2*>(runnerParams.softmaxStatsPtr);
    // For skip softmax
    tllmRunnerParams.mSkipSoftmaxThresholdScaleFactor = runnerParams.skipSoftmaxThresholdScaleFactor;
    tllmRunnerParams.mSkipCorrThreshold = runnerParams.skipCorrectionThreshold;

    tllmRunnerParams.stream = runnerParams.stream;
    // Sparse context attention: reuse the generation-style kernel with per-token sparse indices.
    // Same approach as sparse MLA: keep original batch structure, tileSizeQ only groups heads.
    // The kernel iterates over tokens via numCtasPerSeqQ when maxSeqLenQ > 1.
    if (mContextParams.useTllmGenSparseAttention)
    {
        bool const useDynamicSparseMLA = runnerParams.sparse_params.sparse_attn_kv_lens != nullptr;
        tllmRunnerParams.mSparseAttention
            = useDynamicSparseMLA ? SparseType::DynamicTokenSparse : SparseType::StaticTokenSparse;
        tllmRunnerParams.mSparseTopK = runnerParams.sparse_params.num_sparse_topk;
        tllmRunnerParams.ptrSparseMlaTopKLens = runnerParams.sparse_params.sparse_attn_kv_lens;
        tllmRunnerParams.mKernelType = FmhaKernelType::Generation;
        tllmRunnerParams.mUseGenKernelForPrefill = true;
        tllmRunnerParams.mMaskType = TrtllmGenAttentionMaskType::Causal;
        tllmRunnerParams.kvPageIdxPtr = reinterpret_cast<int const*>(runnerParams.sparse_params.sparse_attn_indices);
        if (useDynamicSparseMLA)
        {
            TLLM_CHECK_WITH_INFO(runnerParams.sparse_params.sliding_window_kv_cache_pool != nullptr,
                "SWA KV pool must be set for dynamic sparse MLA.");
            // Dynamic sparse MLA always has an SWA pool. The compressed pool is optional; when it
            // is absent (ratio == 1), use SWA as kvPtr only to keep TG's primary TMA descriptor valid.
            tllmRunnerParams.kvPtr = runnerParams.sparse_params.sparse_kv_cache_pool != nullptr
                ? runnerParams.sparse_params.sparse_kv_cache_pool
                : runnerParams.sparse_params.sliding_window_kv_cache_pool;
            tllmRunnerParams.slidingWindowKvPoolBasePtr = runnerParams.sparse_params.sliding_window_kv_cache_pool;
        }
        else
        {
            tllmRunnerParams.kvPtr = runnerParams.sparse_params.sparse_kv_cache_pool;
        }
        // Enable MultiCtasKv so the autotuner can select GmemReduction cubins
        tllmRunnerParams.mMultiCtasKvMode = true;
        tllmRunnerParams.multiCtasKvScratchPtr = runnerParams.multiCtasKvScratchPtr;
        tllmRunnerParams.multiCtasKvCounterPtr = runnerParams.multiCtasKvCounterPtr;
    }
    tllmRunnerParams.mUsesSpcompress = mContextParams.useSpcompress;

    mContextRunner->run(tllmRunnerParams);
}

template <typename AttentionVariant>
void TrtllmGenFmha<AttentionVariant>::runGeneration(
    AttentionVariant const& op, FmhaParams const& p, cudaStream_t stream)
{
    dispatchOnDtype(p.getQueryInput().scalar_type(),
        [&]<typename T>(std::type_identity<T>) { runGenerationImpl<T>(op, p, stream); });
}

template <>
template <typename T>
void TrtllmGenFmha<MHA>::runGenerationImpl(MHA const& op, FmhaParams const& p, cudaStream_t stream)
{
    TLLM_LOG_DEBUG("Running TRTLLM-GEN generation kernel.");
    auto const& specDecoding = op.template feature<torch_ext::SpecDecodingFeature>();
    bool const multiQueryTokens = specDecoding.active();
    // Speculative decoding (need to take new generated ids into consideration).
    TLLM_CHECK_WITH_INFO(!multiQueryTokens || p.getSpecDecodingPackedMask() != nullptr,
        "Speculative decoding mode needs a valid packed_mask input tensor.");
    int32_t const* specGenerationLengths = multiQueryTokens ? p.getSpecDecodingGenerationLengths() : nullptr;
    TLLM_CHECK_WITH_INFO(
        !multiQueryTokens || specGenerationLengths != nullptr, "Spec_decoding_generation_lengths must be provided.");

    int const num_q_heads = op.mNumAttnHeads;
    int const num_kv_heads = op.mNumAttnKVHeads;
    int const head_size = op.mHeadSize;
    TLLM_CHECK_WITH_INFO(num_q_heads % num_kv_heads == 0, "numQHeads should be multiple of numKVHeads.");
    unsigned int const num_q_heads_over_kv = num_q_heads / num_kv_heads;
    int const batch_size = static_cast<int>(p.num_requests);
    unsigned int const beam_width = p.meta.effective_beam_width;
    unsigned int const batch_beam_size = batch_size * beam_width;
    float const q_scaling = static_cast<float>(op.q_scaling);
    bool const cross_attention = op.cross_attention;
    int32_t const* sequence_lengths = p.getSequenceLengths();
    KvCacheDataType const cache_type = cacheTypeFromQuantMode(op.quant_mode);

    bool multi_block_mode = tc::getEnvForceDeterministicAttention() ? false : op.mMultiBlockMode;
    if (op.quant_mode.hasInt8KvCache() || (op.quant_mode.hasFp8KvCache() && op.mSM < kSM_90))
    {
        multi_block_mode = false;
    }
    int const chunked_attention_size = p.attn.attention_chunk_size && !tc::getEnvDisableChunkedAttentionInGenPhase()
        ? *p.attn.attention_chunk_size
        : INT_MAX;

    auto const kvBuffers = op.template getKvCacheBuffers<T>(p, /*isGen=*/true);
    auto const& kv_cache_buffer = kvBuffers.kvCacheBuffer;
    auto const& kv_cache_block_scales_buffer = kvBuffers.kvScaleCacheBuffer;

    bool const use_sparse_attention_gen_paged
        = p.hasPagedSparseAttnIndices() && op.use_sparse_attention && op.use_kv_cache;
    bool const useTllmGenSparseAttention
        = op.use_sparse_attention && op.use_tllm_gen_sparse_attention && isTllmGenDevice();
    auto const layout = trtllmGenGenerationLayout(
        p, num_q_heads, num_kv_heads, head_size, op.rotary_embedding_dim, op.useTllmGenSparseAttentionPaged(p));
    void* workspace = p.getWorkspace();
    auto* cu_seq_lens = workspacePtr<int>(workspace, layout.cuSeqlensOffset);
    auto* cu_kv_seq_lens = workspacePtr<int>(workspace, layout.cuKvSeqlensOffset);
    auto* rotary_inv_freq_workspace = workspacePtr<float>(workspace, layout.rotaryInvFreqOffset);
    auto* bmm1_scale_ptr = workspacePtr<float>(workspace, layout.bmm1ScaleOffset);
    auto* bmm2_scale_ptr = workspacePtr<float>(workspace, layout.bmm2ScaleOffset);
    auto* sparse_kv_block_offsets = workspacePtr<int32_t>(workspace, layout.sparseAttnCacheOffset);
    auto* sparse_seq_lengths = sparse_kv_block_offsets == nullptr ? nullptr
                                                                  : sparse_kv_block_offsets
            + static_cast<int64_t>(batch_beam_size) * 2 * p.getMaxBlocksPerSequence() * num_kv_heads;
    void* xqa_q_input_ptr = workspacePtr<void>(workspace, layout.qBufOffset);
    void* scratch = static_cast<int8_t*>(workspace) + trtllmGenScratchOffset(layout);

    // Build cu_seqlens, padding_offset, and rotary inv freq tensors
    BuildDecoderInfoParams<T> decoder_params{};
    decoder_params.seqQOffsets = cu_seq_lens;
    decoder_params.seqKVOffsets = cu_kv_seq_lens;
    decoder_params.seqQLengths = specGenerationLengths;
    decoder_params.seqKVLengths = sequence_lengths;
    decoder_params.batchSize = static_cast<int>(batch_beam_size);
    decoder_params.maxQSeqLength = static_cast<int>(p.input_seq_length);
    decoder_params.numTokens = static_cast<int>(p.num_tokens);
    decoder_params.removePadding = true;
    // Rotary embedding inv_freq buffer.
    decoder_params.rotaryEmbeddingScale = static_cast<float>(p.attn.rope_params.scale);
    decoder_params.rotaryEmbeddingBase = static_cast<float>(p.attn.rope_params.theta);
    decoder_params.rotaryEmbeddingDim = static_cast<int>(op.rotary_embedding_dim);
    decoder_params.rotaryScalingType = static_cast<RotaryScalingType>(p.attn.rope_params.scale_type);
    decoder_params.rotaryEmbeddingInvFreq = rotary_inv_freq_workspace;
    decoder_params.rotaryEmbeddingInvFreqCache = p.getRotaryInvFreq();
    decoder_params.rotaryEmbeddingMaxPositions = static_cast<int>(p.attn.rope_params.max_positions);

    // The rotary_embedding_inv_freq_cache for QKVPreprocessing.
    // Use the rotary_embedding_inv_freq_cache input when the buildDecoderInfoKernel is skipped.
    float const* rotary_inv_freq_buf = p.getRotaryInvFreq();
    // Use the nullptr for cu_seqlens when it is not computed.
    int const* cu_seqlens{nullptr};
    if (decoder_params.isBuildDecoderInfoKernelNeeded())
    {
        rotary_inv_freq_buf = rotary_inv_freq_workspace;
        cu_seqlens = cu_seq_lens;
        invokeBuildDecoderInfo(decoder_params, stream);
        sync_check_cuda_error(stream);
    }

    // IDEA: Store rotary_processed Q buffer to output buffer.
    // NOTE: MHA kernels should read kv cache that has already been appended with new tokens' kv cache.
    // The preprocessing kernel that applies RoPE and updates kv cache.
    QKVPreprocessingParams<T, KVBlockArray> preprocessingParms{};
    preprocessingParms.qkv_input = p.getQueryInput<T>();
    preprocessingParms.q_output = static_cast<T*>(xqa_q_input_ptr);
    preprocessingParms.kv_cache_buffer = kv_cache_buffer;
    preprocessingParms.kv_cache_block_scales_buffer = kv_cache_block_scales_buffer;
    preprocessingParms.qkv_bias = nullptr;
    // Prepare values for fmha.
    preprocessingParms.fmha_bmm1_scale = bmm1_scale_ptr;
    preprocessingParms.fmha_bmm2_scale = bmm2_scale_ptr;
    bool const is_fp8_q_input = (mGenerationQDataType == DATA_TYPE_E4M3);
    if (op.quant_mode.hasFp8KvCache() || op.quant_mode.hasFp4KvCache())
    {
        preprocessingParms.qkv_scale_quant_orig = p.getKvScaleQuantOrig();
    }
    if (op.is_fp8_out)
    {
        preprocessingParms.o_scale_orig_quant = p.getOutQuantScale();
    }
    // Buffers.
    preprocessingParms.logn_scaling = nullptr;
    preprocessingParms.seq_lens = specGenerationLengths;
    preprocessingParms.cache_seq_lens = sequence_lengths;
    preprocessingParms.cu_seq_lens = cu_seqlens;
    preprocessingParms.rotary_embedding_inv_freq = rotary_inv_freq_buf;
    preprocessingParms.rotary_coef_cache_buffer = p.getRotaryCosSin();
    preprocessingParms.qkv_scale_orig_quant = p.getKvScaleOrigQuant();
    preprocessingParms.spec_decoding_position_offsets
        = (cross_attention || !multiQueryTokens) ? nullptr : p.getSpecDecodingPositionOffsets();
    preprocessingParms.mrope_position_deltas = p.getMropePositionDeltas();
    preprocessingParms.helix_position_offsets = p.getHelixPositionOffsets();
    preprocessingParms.helix_is_inactive_rank = p.getHelixIsInactiveRank();
    // Scalar parameters.
    preprocessingParms.batch_size = int(batch_beam_size);
    preprocessingParms.max_input_seq_len = static_cast<int>(p.input_seq_length);
    preprocessingParms.max_kv_seq_len = static_cast<int>(p.max_past_kv_length);
    preprocessingParms.cyclic_kv_cache_len
        = static_cast<int>(cross_attention ? p.max_past_kv_length : p.cyclic_attention_window_size);
    preprocessingParms.sink_token_len = 0;
    preprocessingParms.token_num = static_cast<int>(p.num_tokens);
    preprocessingParms.remove_padding = true;
    preprocessingParms.cross_attention = cross_attention;
    preprocessingParms.head_num = num_q_heads;
    preprocessingParms.kv_head_num = num_kv_heads;
    preprocessingParms.qheads_per_kv_head = num_q_heads / num_kv_heads;
    preprocessingParms.size_per_head = head_size;
    preprocessingParms.fmha_host_bmm1_scale = 1.0f / (sqrtf(head_size * 1.0f) * q_scaling);
    preprocessingParms.rotary_embedding_dim = static_cast<int>(op.rotary_embedding_dim);
    preprocessingParms.rotary_embedding_base = static_cast<float>(p.attn.rope_params.theta);
    preprocessingParms.rotary_scale_type = static_cast<RotaryScalingType>(p.attn.rope_params.scale_type);
    preprocessingParms.rotary_embedding_scale = static_cast<float>(p.attn.rope_params.scale);
    preprocessingParms.rotary_embedding_max_positions = static_cast<int>(p.attn.rope_params.max_positions);
    preprocessingParms.position_embedding_type = op.position_embedding_type;
    preprocessingParms.position_shift_enabled = false;
    preprocessingParms.cache_type = cache_type;
    preprocessingParms.separate_q_kv_output = true;
    preprocessingParms.quantized_fp8_output = is_fp8_q_input;
    preprocessingParms.generation_phase = true;
    preprocessingParms.multi_processor_count = op.getMultiProcessorCount();
    preprocessingParms.rotary_vision_start = -1;
    preprocessingParms.rotary_vision_length = -1;
    // Cross-attention only.
    preprocessingParms.encoder_seq_lens = cross_attention ? sequence_lengths : nullptr;

    invokeQKVPreprocessing<T, KVBlockArray>(preprocessingParms, stream);
    sync_check_cuda_error(stream);

    // Build runner parameters.
    TllmGenFmhaRunnerParams tllmRunnerParams{};

    // Parameters to select kernels.
    tllmRunnerParams.mKernelType = FmhaKernelType::Generation;
    tllmRunnerParams.mMultiCtasKvMode = multi_block_mode;
    // Note that the tileScheduler and multiCtasKvMode will be automatically tuned when using multi_block mode.
    // Otherwise, always enable the persistent scheduler for better performance.
    tllmRunnerParams.mTileScheduler = multi_block_mode ? TileScheduler::Static : TileScheduler::Persistent;

    // The sequence lengths for K/V.
    tllmRunnerParams.seqLensKvPtr = sequence_lengths;

    // Q buffer.
    tllmRunnerParams.qPtr = xqa_q_input_ptr;

    // Use block sparse attention.
    tllmRunnerParams.mUseBlockSparseAttention = false;

    // Paged KV
    tllmRunnerParams.mQkvLayout = QkvLayout::PagedKv;
    tllmRunnerParams.kvPtr = kv_cache_buffer.mPrimaryPoolPtr;
    tllmRunnerParams.kvSfPtr = kv_cache_block_scales_buffer.mPrimaryPoolPtr;
    tllmRunnerParams.kvPageIdxPtr = reinterpret_cast<KVCacheIndex::UnderlyingType const*>(kv_cache_buffer.data);
    tllmRunnerParams.mMaxNumPagesPerSeqKv = kv_cache_buffer.mMaxBlocksPerSeq;
    tllmRunnerParams.mNumTokensPerPage = kv_cache_buffer.mTokensPerBlock;

    auto const sparse_params = p.getSparseParams();
    // Gather kv page offsets for block sparse attention (with offsets).
    if (use_sparse_attention_gen_paged)
    {
        invokeGatherKvPageOffsets(sparse_kv_block_offsets, sparse_seq_lengths,
            reinterpret_cast<int32_t const*>(kv_cache_buffer.data), sequence_lengths, sparse_params, batch_beam_size,
            num_kv_heads, kv_cache_buffer.mTokensPerBlock, kv_cache_buffer.mMaxBlocksPerSeq, stream);
        sync_check_cuda_error(stream);
        tllmRunnerParams.seqLensKvPtr = sparse_seq_lengths;
        tllmRunnerParams.kvPageIdxPtr = reinterpret_cast<KVCacheIndex::UnderlyingType const*>(sparse_kv_block_offsets);
        tllmRunnerParams.mUseBlockSparseAttention = true;
    }
    // Sparse MQA/GQA attention: use trtllm-gen sparse kernel.
    else if (useTllmGenSparseAttention)
    {
        tllmRunnerParams.mSparseAttention = SparseType::StaticTokenSparse;
        tllmRunnerParams.mSparseTopK = sparse_params.num_sparse_topk;
        tllmRunnerParams.kvPageIdxPtr = reinterpret_cast<int const*>(sparse_params.sparse_attn_indices);
        tllmRunnerParams.kvPtr = sparse_params.sparse_kv_cache_pool;
    }

    // The partial buffers' pointers when the multiCtasKv mode is enabled.
    tllmRunnerParams.multiCtasKvCounterPtr
        = op.getMultiCtasKvCounter(p.getMaxNumSequences(), p.getQueryInput().device());
    tllmRunnerParams.multiCtasKvScratchPtr = scratch;

    tllmRunnerParams.attentionSinksPtr = p.getAttentionSinks();
    tllmRunnerParams.cumSeqLensQPtr = cu_seqlens;
    tllmRunnerParams.cumSeqLensKvPtr = cu_kv_seq_lens;
    // Attention scales device pointers (only fp8 kernels need to load scales from the device memory).
    if (mGenerationQDataType == DATA_TYPE_E4M3)
    {
        tllmRunnerParams.outputScalePtr = bmm2_scale_ptr;
        tllmRunnerParams.scaleSoftmaxLog2Ptr = bmm1_scale_ptr ? bmm1_scale_ptr + kIdxScaleSoftmaxLog2Ptr : nullptr;
    }
    tllmRunnerParams.oSfScalePtr = p.getOutSfScale();

    tllmRunnerParams.oPtr = p.getOutput();
    tllmRunnerParams.oSfPtr = p.getOutputSf();
    tllmRunnerParams.mHeadDimQk = head_size;
    tllmRunnerParams.mHeadDimV = head_size;
    tllmRunnerParams.mNumHeadsQ = num_q_heads;
    tllmRunnerParams.mNumHeadsKv = num_kv_heads;
    tllmRunnerParams.mNumHeadsQPerKv = num_q_heads_over_kv;
    tllmRunnerParams.mBatchSize = batch_size;
    // It is used to construct contiguous kv cache TMA descriptors.
    tllmRunnerParams.mMaxSeqLenCacheKv = static_cast<int>(p.max_attention_window_size);
    tllmRunnerParams.mMaxSeqLenQ = static_cast<int>(p.input_seq_length);
    int const spec_decoding_max_generation_length = specDecoding.maxGenerationLength();
    bool const isSpecDecTree = p.meta.is_spec_dec_tree && multiQueryTokens;
    if (isSpecDecTree)
    {
        TLLM_CHECK_WITH_INFO(spec_decoding_max_generation_length > 0,
            "spec_decoding_max_generation_length must be positive for spec-dec tree.");
        tllmRunnerParams.mMaxSeqLenQ = std::min(tllmRunnerParams.mMaxSeqLenQ, spec_decoding_max_generation_length);
        tllmRunnerParams.mMaxSeqLenKv = static_cast<int>(p.max_past_kv_length) + tllmRunnerParams.mMaxSeqLenQ;
    }
    else
    {
        tllmRunnerParams.mMaxSeqLenKv = static_cast<int>(p.max_past_kv_length);
    }
    tllmRunnerParams.mJITWarmup = p.meta.trtllm_gen_jit_warmup;
    tllmRunnerParams.mJITWarmupMaxNumRequests = p.meta.max_num_requests;
    tllmRunnerParams.mJITWarmupMaxSeqLenQ = p.meta.max_context_length;
    tllmRunnerParams.mJITWarmupMaxSeqLenKv = p.meta.max_seq_len;
    tllmRunnerParams.mSumOfSeqLensQ = int(batch_size * beam_width * tllmRunnerParams.mMaxSeqLenQ);
    int const cyclic_attention_window_size = static_cast<int>(p.cyclic_attention_window_size);
    bool const usesSlidingWindow = cyclic_attention_window_size > 0 && cyclic_attention_window_size != INT_MAX
        && tllmRunnerParams.mMaxSeqLenKv > cyclic_attention_window_size;
    if (usesSlidingWindow)
    {
        tllmRunnerParams.mLeftSlidingWindow = cyclic_attention_window_size - 1;
        tllmRunnerParams.mRightSlidingWindow = 0;
    }
    else
    {
        tllmRunnerParams.mLeftSlidingWindow = -1;
        tllmRunnerParams.mRightSlidingWindow = -1;
    }
    // The chunked attention size.
    // The generation-phase chunked attention is disabled for now.
    bool const usesChunkedAttention = !isSpecDecTree && chunked_attention_size > 0 && chunked_attention_size != INT_MAX
        && chunked_attention_size < tllmRunnerParams.mMaxSeqLenKv;
    tllmRunnerParams.mChunkedAttentionSize = usesChunkedAttention ? chunked_attention_size : 0;
    TLLM_CHECK_WITH_INFO(!(usesChunkedAttention && usesSlidingWindow),
        "Chunked attention size and sliding window size should not be used together.");
    // Not used in the generation kernels as contiguous_kv or paged_kv layouts are used.
    tllmRunnerParams.mSumOfSeqLensKv = int(batch_size * beam_width * tllmRunnerParams.mMaxSeqLenKv);
    tllmRunnerParams.mScaleQ = q_scaling;
    // Set it to INT_MAX as the kv cache pageOffsets will ensure that there is no out-of-bounds access.
    tllmRunnerParams.mNumPagesInMemPool = INT_MAX;
    tllmRunnerParams.mMultiProcessorCount = op.getMultiProcessorCount();
    tllmRunnerParams.stream = stream;
    tllmRunnerParams.mSfStartTokenIdx = static_cast<int32_t>(p.token_offset);
    tllmRunnerParams.mIsSpecDecTree = isSpecDecTree;
    if (isSpecDecTree)
    {
        if (usesSlidingWindow)
        {
            tllmRunnerParams.mMaskType = TrtllmGenAttentionMaskType::SlidingWindowCustom;
        }
        else
        {
            tllmRunnerParams.mMaskType = TrtllmGenAttentionMaskType::Custom;
        }
    }
    else if (usesChunkedAttention)
    {
        tllmRunnerParams.mMaskType = TrtllmGenAttentionMaskType::SlidingOrChunkedCausal;
        tllmRunnerParams.mLeftSlidingWindow = -1;
        tllmRunnerParams.mRightSlidingWindow = -1;
    }
    else if (usesSlidingWindow)
    {
        tllmRunnerParams.mMaskType = TrtllmGenAttentionMaskType::SlidingOrChunkedCausal;
    }
    else
    {
        tllmRunnerParams.mMaskType = TrtllmGenAttentionMaskType::Causal;
    }
    tllmRunnerParams.mLayerIdx = static_cast<int>(p.local_layer_idx);
    tllmRunnerParams.seqLensQPtr = specGenerationLengths;
    if (multiQueryTokens)
    {
        tllmRunnerParams.generalPackedCustoMaskPtr = p.getSpecDecodingPackedMask();
        tllmRunnerParams.customMaskPtr = p.getSpecDecodingBlTreeMask();
        tllmRunnerParams.customMaskCapacity
            = p.spec_decoding_bl_tree_mask.has_value() ? p.spec_decoding_bl_tree_mask->numel() : 0;
        tllmRunnerParams.customMaskOffsetsPtr = p.getSpecDecodingBlTreeMaskOffset();
        tllmRunnerParams.firstSparseMaskOffsetsKvPtr = p.getSpecBlTreeFirstSparseMaskOffsetKv();
    }
    tllmRunnerParams.mPackedMaskMaxSeqLenQ = spec_decoding_max_generation_length;
    tllmRunnerParams.mSpecDecodingTargetMaxGenLen = static_cast<int>(op.spec_decoding_target_max_gen_len);
    tllmRunnerParams.mForcePrepareSpecDecTreeMask = p.meta.force_prepare_spec_dec_tree_mask;
    tllmRunnerParams.mSkipSoftmaxThresholdScaleFactor
        = static_cast<float>(p.fwd.sparse_runtime_params.threshold_scale_factor_decode);
    tllmRunnerParams.softmaxStatsPtr = p.getSoftmaxStatsTensor();
    mGenerationRunner->run(tllmRunnerParams);
}

template <>
template <typename T>
void TrtllmGenFmha<MLA>::runGenerationImpl(MLA const& op, FmhaParams const& p, cudaStream_t stream)
{
    auto params = op.template getGenerationParams<T>(p);
    auto const kv_cache_buffer = op.template getGenerationKvCacheBuffer<T>(p);
    int const num_kv_heads = 1;
    int const head_size = op.mMLAParams.kv_lora_rank + op.mMLAParams.qk_rope_head_dim;
    int const head_size_v = (op.useSparseMLA() && !op.mMLAParams.rope_append) ? head_size : op.mMLAParams.kv_lora_rank;
    int32_t const batch_beam = p.meta.effective_beam_width * p.num_requests;
    double const q_scaling = op.q_scaling;
    void* scratchPtr = params.workspace;

    TllmGenFmhaRunnerParams tllmRunnerParams{};

    // Parameters to select kernels.
    // MLA generation kernels use dense mask. For multi-token generation, TRTLLM-Gen applies causality by
    // shrinking each token's effective KV length.
    tllmRunnerParams.mMaskType = TrtllmGenAttentionMaskType::Dense;
    tllmRunnerParams.mKernelType = FmhaKernelType::Generation;
    tllmRunnerParams.mMultiCtasKvMode = op.mMultiBlockMode;
    // Note that the tileScheduler and multiCtasKvMode will be automatically tuned when using multi_block mode.
    // Otherwise, always enable the persistent scheduler for better performance.
    tllmRunnerParams.mTileScheduler = op.mMultiBlockMode ? TileScheduler::Static : TileScheduler::Persistent;

    // Q buffer.
    tllmRunnerParams.qPtr = op.mFP8GenerationMLA ? reinterpret_cast<void const*>(params.quant_q_buf)
                                                 : reinterpret_cast<void const*>(params.q_buf);

    // KV buffer
    // Paged KV
    tllmRunnerParams.mQkvLayout = QkvLayout::PagedKv;
    tllmRunnerParams.kvPtr = kv_cache_buffer.mPrimaryPoolPtr;
    tllmRunnerParams.kvPageIdxPtr = reinterpret_cast<KVCacheIndex::UnderlyingType const*>(kv_cache_buffer.data);
    tllmRunnerParams.mMaxNumPagesPerSeqKv = kv_cache_buffer.mMaxBlocksPerSeq;
    tllmRunnerParams.mNumTokensPerPage = kv_cache_buffer.mTokensPerBlock;

    // The partial buffers' pointers when the multiCtasKv mode is enabled.
    tllmRunnerParams.multiCtasKvCounterPtr
        = op.getMultiCtasKvCounter(p.getMaxNumSequences(), p.getQueryInput().device());
    tllmRunnerParams.multiCtasKvScratchPtr = scratchPtr;

    // The sequence lengths for K/V.
    tllmRunnerParams.seqLensKvPtr = params.cache_seq_lens;

    tllmRunnerParams.oPtr = reinterpret_cast<void*>(params.context_buf);
    tllmRunnerParams.oSfPtr = p.getOutputSf();
    if (params.dsv4_epilogue_fusion.enabled)
    {
        tllmRunnerParams.mDsv4EpilogueFusion.enabled = true;
        tllmRunnerParams.mDsv4EpilogueFusion.cosSinCache = params.dsv4_epilogue_fusion.cos_sin_cache;
        tllmRunnerParams.mDsv4EpilogueFusion.scaleBufM = params.dsv4_epilogue_fusion.scale_buf_m;
    }

    // softmax stats if needed
    tllmRunnerParams.softmaxStatsPtr = p.getSoftmaxStatsTensor();

    // Per-head attention sink added to the softmax denominator.
    tllmRunnerParams.attentionSinksPtr = p.getAttentionSinks();

    // MLA uses different head dimensions for Qk and V.
    tllmRunnerParams.mHeadDimQk = head_size;
    tllmRunnerParams.mHeadDimV = head_size_v;

    auto const num_q_heads = op.mNumAttnHeads;
    tllmRunnerParams.mNumHeadsQ = num_q_heads;
    tllmRunnerParams.mNumHeadsKv = num_kv_heads;
    tllmRunnerParams.mNumHeadsQPerKv = num_q_heads / num_kv_heads;

    tllmRunnerParams.mBatchSize = batch_beam;
    // It is used to construct contiguous kv cache TMA descriptors.
    tllmRunnerParams.mMaxSeqLenCacheKv = p.max_attention_window_size;
    // This should be set to numDraftTokens + 1.
    tllmRunnerParams.mMaxSeqLenQ = params.acc_q_len / batch_beam;
    tllmRunnerParams.mMaxSeqLenKv = p.max_past_kv_length;
    tllmRunnerParams.mJITWarmup = p.meta.trtllm_gen_jit_warmup;
    tllmRunnerParams.mJITWarmupMaxNumRequests = p.meta.max_num_requests;
    tllmRunnerParams.mJITWarmupMaxSeqLenQ = p.meta.max_context_length;
    tllmRunnerParams.mJITWarmupMaxSeqLenKv = p.meta.max_seq_len;
    tllmRunnerParams.mSumOfSeqLensQ = int(batch_beam * tllmRunnerParams.mMaxSeqLenQ);
    // Not used in the generation kernels as contiguous_kv or paged_kv layouts are used.
    tllmRunnerParams.mSumOfSeqLensKv = int(batch_beam * tllmRunnerParams.mMaxSeqLenKv);

    tllmRunnerParams.mLeftSlidingWindow = -1;
    tllmRunnerParams.mRightSlidingWindow = -1;
    tllmRunnerParams.mChunkedAttentionSize = 0;

    // The scaleQ that will be applied to the BMM1 output.
    tllmRunnerParams.mScaleQ = q_scaling
        * sqrt((float) (op.mMLAParams.qk_nope_head_dim + op.mMLAParams.qk_rope_head_dim))
        / sqrtf((float) (op.mMLAParams.kv_lora_rank + op.mMLAParams.qk_rope_head_dim));

    // Set it to INT_MAX as the kv cache pageOffsets will ensure that there is no out-of-bounds access.
    tllmRunnerParams.mNumPagesInMemPool = INT_MAX;
    tllmRunnerParams.mMultiProcessorCount = op.mMultiProcessorCount;
    tllmRunnerParams.stream = stream;
    tllmRunnerParams.mSfStartTokenIdx = p.token_offset;

    // Scales for quantization
    if (op.mFP8GenerationMLA)
    {
        static constexpr int bmm1_scale_offset = 1;
        tllmRunnerParams.outputScalePtr = reinterpret_cast<float const*>(params.bmm2_scale);
        tllmRunnerParams.scaleSoftmaxLog2Ptr = reinterpret_cast<float const*>(params.bmm1_scale) + bmm1_scale_offset;
    }

    // Set the following parameters if sparseAttention is used.
    if (op.useSparseMLA())
    {
        auto const& sparseParams = op.template feature<torch_ext::SparseMlaFeature>().params();
        bool const useDynamicSparseMLA = sparseParams.sparse_attn_kv_lens != nullptr;
        tllmRunnerParams.mSparseAttention
            = useDynamicSparseMLA ? SparseType::DynamicTokenSparse : SparseType::StaticTokenSparse;
        tllmRunnerParams.mSkipCorrThreshold = op.mSkipCorrectionThreshold;
        tllmRunnerParams.mSparseTopK = sparseParams.num_sparse_topk;
        tllmRunnerParams.ptrSparseMlaTopKLens = sparseParams.sparse_attn_kv_lens;
        tllmRunnerParams.kvPageIdxPtr
            = reinterpret_cast<KVCacheIndex::UnderlyingType const*>(sparseParams.sparse_attn_indices);
        if (useDynamicSparseMLA)
        {
            TLLM_CHECK_WITH_INFO(sparseParams.sliding_window_kv_cache_pool != nullptr,
                "SWA KV pool must be set for dynamic sparse MLA.");
            // Dynamic sparse MLA always has an SWA pool. The compressed pool is optional; when it
            // is absent (ratio == 1), use SWA as kvPtr only to keep TG's primary TMA descriptor valid.
            tllmRunnerParams.kvPtr = sparseParams.sparse_kv_cache_pool != nullptr
                ? sparseParams.sparse_kv_cache_pool
                : sparseParams.sliding_window_kv_cache_pool;
            tllmRunnerParams.slidingWindowKvPoolBasePtr = sparseParams.sliding_window_kv_cache_pool;
        }
        else
        {
            tllmRunnerParams.kvPtr = sparseParams.sparse_kv_cache_pool;
        }

        if (!useDynamicSparseMLA && op.mUseNvfp4MlaKvCache)
        {
            // Static sparse MLA indexes a compact KV pool containing at most
            // mSparseTopK rows per query. Do not let the original dense KV
            // length drive kernel selection or launch geometry: for long
            // sequences that can select a multi-CTA kernel which addresses
            // beyond the compact page table.
            TLLM_CHECK_WITH_INFO(tllmRunnerParams.mSparseTopK > 0, "Static sparse MLA requires a positive TopK, got %d",
                tllmRunnerParams.mSparseTopK);
            int32_t const originalMaxSeqLenKv = tllmRunnerParams.mMaxSeqLenKv;
            int32_t const effectiveMaxSeqLenKv = std::min(originalMaxSeqLenKv, tllmRunnerParams.mSparseTopK);
            tllmRunnerParams.mMaxSeqLenKv = effectiveMaxSeqLenKv;
            tllmRunnerParams.mJITWarmupMaxSeqLenKv
                = std::min(tllmRunnerParams.mJITWarmupMaxSeqLenKv, effectiveMaxSeqLenKv);
            int64_t const sumOfSeqLensKv = static_cast<int64_t>(tllmRunnerParams.mBatchSize) * effectiveMaxSeqLenKv;
            TLLM_CHECK_WITH_INFO(sumOfSeqLensKv <= std::numeric_limits<int32_t>::max(),
                "Static sparse MLA cumulative KV length exceeds int32 capacity: %ld", sumOfSeqLensKv);
            tllmRunnerParams.mSumOfSeqLensKv = static_cast<int32_t>(sumOfSeqLensKv);
            TLLM_LOG_DEBUG("Clamp static sparse MLA max KV length from %d to %d (TopK=%d)", originalMaxSeqLenKv,
                effectiveMaxSeqLenKv, tllmRunnerParams.mSparseTopK);
        }
    }

    mGenerationRunner->run(tllmRunnerParams);
    sync_check_cuda_error(stream);
}

// ---------------------------------------------------------------------------
// FmhaV2Fmha
// ---------------------------------------------------------------------------

template <typename AttentionVariant>
bool FmhaV2Fmha<AttentionVariant>::isAvailable(AttentionBase const& key)
{
    if constexpr (std::is_same_v<AttentionVariant, MHA>)
    {
        return !isTrtllmGenGpu() || key.head_size == 72 || key.head_size == 104;
    }
    else
    {
        return !isTrtllmGenGpu();
    }
}

template <typename AttentionVariant>
bool FmhaV2Fmha<AttentionVariant>::isContextSupported(MHARunnerFixedParams fixedParams)
{
    return makeFmhaV2ContextRunner(fixedParams) != nullptr;
}

template <typename AttentionVariant>
bool FmhaV2Fmha<AttentionVariant>::isSupported(FmhaParams const& p, bool isGen) const
{
    return isGen ? mGenerationRunner != nullptr : mContextRunner != nullptr;
}

template <typename AttentionVariant>
void FmhaV2Fmha<AttentionVariant>::initContext(AttentionVariant const& op)
{
    if constexpr (std::is_same_v<AttentionVariant, MHA>)
    {
        if (!isHalfOrBf16(op.type) || op.isRelativePosition() || op.isQuantizedCrossCacheUnfused())
        {
            return;
        }
    }
    auto fixedParams = op.contextFixedParams();
    // On SM100f fmha_v2 serves only what trtllm-gen leaves to it.
    if (isTrtllmGenGpu() && !prefersFmhaV2Context(fixedParams))
    {
        return;
    }
    mContextRunner = makeFmhaV2ContextRunner(fixedParams);
    if (mContextRunner == nullptr)
    {
        TLLM_LOG_WARNING(
            "Fall back to unfused MHA for %s in sm_%d.", fixedParams.convertToStrOutput().c_str(), op.smVersion());
    }
}

template <typename AttentionVariant>
void FmhaV2Fmha<AttentionVariant>::initGeneration(AttentionVariant const& op)
{
    if constexpr (std::is_same_v<AttentionVariant, MLA>)
    {
        if (!op.mIsGenerationMLA || tc::isSM100Family(op.smVersion()))
        {
            return;
        }
        // Construct the fmha runner for generation.
        Data_type data_type = op.type == tensorrt_llm::DataType::kBF16 ? DATA_TYPE_BF16 : DATA_TYPE_FP16;
        if (op.mFP8GenerationMLA)
        {
            data_type = DATA_TYPE_E4M3;
        }
        MHARunnerFixedParams fmhaParams{};
        fmhaParams.dataType = data_type;
        fmhaParams.dataTypeKv = data_type;
        fmhaParams.dataTypeOut = data_type;
        // For FP8 MLA generation, the output type is BF16, and the quantization before o_proj
        // is performed separately.
        if (op.mFP8GenerationMLA)
        {
            fmhaParams.dataTypeOut = DATA_TYPE_BF16;
        }
        // TODO: remove forceFp32Acc from MHARunnerFixedParams after adding
        // host_runtime_perf_knobs to bertAttentionPlugin input tensors, so that we can change
        // mLaunchParams.force_fp32_acc value in runtime.
        fmhaParams.forceFp32Acc = true;
        fmhaParams.attentionMaskType = ContextAttentionMaskType::PADDING;
        fmhaParams.attentionInputLayout = AttentionInputLayout::Q_PAGED_KV;
        fmhaParams.isSPadded = !op.remove_padding;
        fmhaParams.numQHeads = 1;
        fmhaParams.numKvHeads = 1;
        fmhaParams.headSize = op.mMLAParams.kv_lora_rank + op.mMLAParams.qk_rope_head_dim;
        fmhaParams.headSizeV = op.mMLAParams.kv_lora_rank;
        fmhaParams.qScaling = op.q_scaling
            * sqrt((float) (op.mMLAParams.qk_nope_head_dim + op.mMLAParams.qk_rope_head_dim))
            / sqrtf((float) (op.mMLAParams.kv_lora_rank + op.mMLAParams.qk_rope_head_dim));
        fmhaParams.attnLogitSoftcappingScale = op.attn_logit_softcapping_scale;
        fmhaParams.hasAlibi = false;
        fmhaParams.scaleAlibi = false;
        fmhaParams.tpSize = 1;
        fmhaParams.tpRank = 0;
        mGenerationRunner = std::make_unique<FusedMHARunnerV2>(fmhaParams);
        if (!mGenerationRunner->isFmhaSupported())
        {
            // FlashMLA serves the generation configs FMHA-v2 has no kernel for.
            TLLM_CHECK_WITH_INFO(
                FlashMlaFmha::isAvailable(op), "Deepseek should be supported by fmha in generation part.");
            mGenerationRunner.reset();
        }
    }
}

template <typename AttentionVariant>
std::size_t FmhaV2Fmha<AttentionVariant>::getWorkspaceSize(
    AttentionVariant const& op, FmhaParams const& p, bool isGen) const
{
    if (!isGen)
    {
        return mContextRunner == nullptr ? 0 : op.getWorkspaceSizeForContext(p);
    }
    if constexpr (std::is_same_v<AttentionVariant, MLA>)
    {
        if (mGenerationRunner != nullptr)
        {
            return op.getGenerationWorkspaceSize(p, /*kernelWorkspaceSize=*/0);
        }
    }
    return 0;
}

template <typename AttentionVariant>
void FmhaV2Fmha<AttentionVariant>::runContext(AttentionVariant const& op, FmhaParams const& p, cudaStream_t stream)
{
    dispatchOnDtype(p.getQueryInput().scalar_type(),
        [&]<typename T>(std::type_identity<T>)
        {
            op.template enqueueFusedContext<T>(
                p, stream, [this](MHARunnerParams const& runnerParams) { mContextRunner->run(runnerParams); });
        });
}

template <typename AttentionVariant>
void FmhaV2Fmha<AttentionVariant>::runGeneration(AttentionVariant const& op, FmhaParams const& p, cudaStream_t stream)
{
    if constexpr (std::is_same_v<AttentionVariant, MLA>)
    {
        dispatchOnDtype(p.getQueryInput().scalar_type(),
            [&]<typename T>(std::type_identity<T>) { runGenerationImpl<T>(op, p, stream); });
    }
    else
    {
        PhasedFmha<AttentionVariant>::runGeneration(op, p, stream);
    }
}

template <typename AttentionVariant>
template <typename T>
void FmhaV2Fmha<AttentionVariant>::runGenerationImpl(
    AttentionVariant const& op, FmhaParams const& p, cudaStream_t stream)
{
    if constexpr (std::is_same_v<AttentionVariant, MLA>)
    {
        auto params = op.template getGenerationParams<T>(p);
        auto const kv_cache_buffer = op.template getGenerationKvCacheBuffer<T>(p);
        int32_t const batch_beam = p.meta.effective_beam_width * p.num_requests;

        // Use FMHA otherwise.
        MHARunnerParams fmhaParams{};
        fmhaParams.b = batch_beam;
        fmhaParams.numGroupedHeads = params.head_num;
        fmhaParams.qSeqLen = params.head_num * (params.acc_q_len / batch_beam);
        fmhaParams.kvSeqLen = p.max_past_kv_length;
        // Disable sliding window attention when it is not needed.
        fmhaParams.slidingWindowSize = p.cyclic_attention_window_size;
        fmhaParams.totalQSeqLen = batch_beam * fmhaParams.qSeqLen;
        // TODO: set it correctly for contiguous kv buffer (cross-attention).
        // fmhaParams.totalKvSeqLen = params.num_tokens;
        // Device buffer pointers.
        // fmhaParams.qkvPtr = reinterpret_cast<void const*>(params.attention_input);
        fmhaParams.qPtr = op.mFP8GenerationMLA ? reinterpret_cast<void const*>(params.quant_q_buf)
                                               : reinterpret_cast<void const*>(params.q_buf);
        // TODO: add contiguous kv buffer (cross-attention).
        fmhaParams.kvPtr = nullptr;

        fmhaParams.outputPtr = reinterpret_cast<void*>(params.context_buf);

        // fmhaParams.packedMaskPtr = params.fmha_custom_mask;
        fmhaParams.pagedKvCache = kv_cache_buffer;
        fmhaParams.cuQSeqLenPtr = params.seqQOffset;
        fmhaParams.kvSeqLenPtr = params.cache_seq_lens;
        fmhaParams.cuKvSeqLenPtr = params.cu_kv_seqlens;
        fmhaParams.cuMaskRowsPtr = nullptr; // mla not support custorm mask right now
        fmhaParams.tileCounterPtr = params.fmha_tile_counter;
        fmhaParams.scaleBmm1Ptr = reinterpret_cast<float const*>(params.bmm1_scale);
        fmhaParams.scaleBmm2Ptr = reinterpret_cast<float const*>(params.bmm2_scale);
        fmhaParams.stream = stream;
        fmhaParams.forceFp32Acc = op.mFMHAForceFP32Acc;
        op.applyFeatures(fmhaParams);

        // MLA does not support skip-softmax attention right now

        // Run the fmha kernel
        mGenerationRunner->run(fmhaParams);
    }
}

// ---------------------------------------------------------------------------
// XqaFmha
// ---------------------------------------------------------------------------

template <typename AttentionVariant>
bool XqaFmha<AttentionVariant>::isAvailable(AttentionBase const& key)
{
    if (isTrtllmGenGpu() || !isHalfOrBf16(key.type) || !key.use_kv_cache)
    {
        return false;
    }
    if constexpr (std::is_same_v<AttentionVariant, MLA>)
    {
        bool const isGenerationMLA = key.head_size == key.kv_lora_rank + key.qk_rope_head_dim;
        return (tc::isSM120Family() && isGenerationMLA) || key.is_spec_decoding_enabled;
    }
    else
    {
        return true;
    }
}

template <typename AttentionVariant>
template <typename T>
bool XqaFmha<AttentionVariant>::fillParams(
    AttentionVariant const& op, FmhaParams const& p, XQAParams& xqaParams, bool forConfigurePlugin) const
{
    if constexpr (!(std::is_same_v<T, half> || std::is_same_v<T, __nv_bfloat16>) )
    {
        return false;
    }
    else
    {
        xqaParams = {};
        xqaParams.data_type = std::is_same_v<T, half> ? DATA_TYPE_FP16 : DATA_TYPE_BF16;

        xqaParams.num_q_heads = op.mNumAttnHeads;
        xqaParams.num_kv_heads = op.mNumAttnKVHeads;
        xqaParams.head_size = op.mHeadSize;
        xqaParams.unidirectional = 1;
        xqaParams.q_scaling = op.q_scaling;
        xqaParams.remove_padding = op.remove_padding;
        xqaParams.mask_type = op.mask_type;
        xqaParams.tp_size = 1;
        xqaParams.tp_rank = 0;
        xqaParams.multi_block_mode = tc::getEnvForceDeterministicAttention() ? false : op.mMultiBlockMode;
        xqaParams.layer_idx = static_cast<int>(p.local_layer_idx);

        op.applyFeatures(xqaParams);
        if (op.quant_mode.hasFp8KvCache() && xqaParams.isMLA())
        {
            // Inputs to MLA is FP8 instead of BF16/FP16 when using FP8 KV cache.
            xqaParams.data_type = DATA_TYPE_E4M3;
        }
        if (xqaParams.kv_cache_data_type == DATA_TYPE_INT8
            || (xqaParams.kv_cache_data_type == DATA_TYPE_E4M3 && op.mSM < kSM_90))
        {
            xqaParams.multi_block_mode = false;
        }

        xqaParams.output = p.getOutput();
        xqaParams.qkv = p.getQueryInput<T>();
        xqaParams.cache_indir = p.getCacheIndirection();
        xqaParams.attention_sinks = p.getAttentionSinks();
        xqaParams.kv_scale_orig_quant = p.getKvScaleOrigQuant();
        xqaParams.kv_scale_quant_orig = p.getKvScaleQuantOrig();
        xqaParams.host_past_key_value_lengths = p.getHostPastKeyValueLengths();
        xqaParams.host_context_lengths = p.getHostContextLengths();
        xqaParams.semaphores = op.getMultiCtasKvCounter(p.getMaxNumSequences(), p.getQueryInput().device());
        xqaParams.workspaces = p.getWorkspace();
        xqaParams.batch_size = p.num_requests;
        xqaParams.beam_width = p.meta.effective_beam_width;
        // Speculative decoding mode has generation input_length > 1.
        xqaParams.generation_input_length = p.input_seq_length;
        xqaParams.chunked_attention_size = p.attn.attention_chunk_size && !tc::getEnvDisableChunkedAttentionInGenPhase()
            ? *p.attn.attention_chunk_size
            : INT_MAX;
        xqaParams.max_attention_window_size = p.max_attention_window_size;
        xqaParams.cyclic_attention_window_size = p.cyclic_attention_window_size;
        xqaParams.max_past_kv_length = p.max_past_kv_length;
        xqaParams.sequence_lengths = p.getSequenceLengths();
        xqaParams.context_lengths = p.getContextLengths();
        if (!forConfigurePlugin)
        {
            // Speculative decoding (need to take new generated ids into consideration).
            TLLM_CHECK_WITH_INFO(!op.template feature<torch_ext::SpecDecodingFeature>().active()
                    || p.getSpecDecodingPackedMask() != nullptr,
                "Speculative decoding mode needs a valid packed_mask input tensor.");
        }
        xqaParams.helix_position_offsets = p.getHelixPositionOffsets();
        xqaParams.helix_is_inactive_rank = p.getHelixIsInactiveRank();
        xqaParams.softmax_stats = p.getSoftmaxStatsTensor();
        xqaParams.trtllm_gen_jit_warmup = p.meta.trtllm_gen_jit_warmup;
        xqaParams.trtllm_gen_jit_warmup_max_num_requests = p.meta.max_num_requests;
        xqaParams.trtllm_gen_jit_warmup_max_seq_len_q = p.meta.max_context_length;
        xqaParams.trtllm_gen_jit_warmup_max_seq_len_kv = p.meta.max_seq_len;

        xqaParams.total_num_input_tokens = p.num_tokens;
        // Skip softmax threshold.
        xqaParams.skip_softmax_threshold_scale_factor
            = static_cast<float>(p.fwd.sparse_runtime_params.threshold_scale_factor_decode);
#ifdef SKIP_SOFTMAX_STAT
        // Statistics of skip-softmax, pointers of device memory for output
        xqaParams.skip_softmax_total_blocks = p.getSkipSoftmaxStat();
        xqaParams.skip_softmax_skipped_blocks = p.getSkipSoftmaxStat() + 1;
#endif
        if constexpr (std::is_same_v<AttentionVariant, MLA>)
        {
            if (!forConfigurePlugin)
            {
                xqaParams.quant_q_buffer_ptr = p.getQuantQBuffer();
                xqaParams.q_scaling = 1
                    / (op.q_scaling * sqrtf((float) (op.mMLAParams.qk_nope_head_dim + op.mMLAParams.qk_rope_head_dim)));
            }
        }
        return true;
    }
}

template <typename AttentionVariant>
bool XqaFmha<AttentionVariant>::isSupported(FmhaParams const& p, bool isGen) const
{
    if (!isGen || mRunner == nullptr)
    {
        return false;
    }
    auto const& op = static_cast<AttentionVariant const&>(this->attention);
    bool usable = false;
    dispatchOnDtype(p.getQueryInput().scalar_type(),
        [&]<typename T>(std::type_identity<T>)
        {
            XQAParams params{};
            usable = fillParams<T>(op, p, params, /*forConfigurePlugin=*/false)
                && mRunner->supportConfig(params, /*forConfigurePlugin=*/false);
        });
    return usable;
}

template <typename AttentionVariant>
bool XqaFmha<AttentionVariant>::hasPerfGain(FmhaParams const& p) const
{
    if (mRunner == nullptr)
    {
        return false;
    }
    auto const& op = static_cast<AttentionVariant const&>(this->attention);
    bool gain = false;
    dispatchOnDtype(p.getQueryInput().scalar_type(),
        [&]<typename T>(std::type_identity<T>)
        {
            XQAParams params{};
            gain = fillParams<T>(op, p, params, /*forConfigurePlugin=*/false) && mRunner->hasPerfGain(params);
        });
    return gain;
}

template <typename AttentionVariant>
void XqaFmha<AttentionVariant>::initGeneration(AttentionVariant const& op)
{
    TLLM_LOG_DEBUG("Enabling XQA kernels.");
    TLLM_CHECK_WITH_INFO(!op.is_spec_decoding_enabled || op.mNumAttnHeads % op.mNumAttnKVHeads == 0,
        "num_heads should be multiples of mNumKVHeads.");
    Data_type const dataType = op.type == tensorrt_llm::DataType::kHALF ? DATA_TYPE_FP16 : DATA_TYPE_BF16;
    mRunner = std::make_unique<DecoderXQARunner>(
        dataType, op.mNumAttnHeads, op.mNumAttnKVHeads, op.mHeadSize, op.mMultiBlockMode);
}

template <typename AttentionVariant>
std::size_t XqaFmha<AttentionVariant>::getWorkspaceSize(
    AttentionVariant const& op, FmhaParams const& p, bool isGen) const
{
    if (!isGen || mRunner == nullptr)
    {
        return 0;
    }
    // Sized per sequence (max_num_requests * beam_width) from the batch bound, not from this
    // call's rows, so a context call sizing a mixed batch and the following generation call
    // compute the same number.
    auto const max_num_seq = static_cast<int32_t>(p.getMaxNumSequences());
    auto const max_num_tokens = static_cast<int32_t>(p.num_gen_tokens);
    auto const max_blocks_per_sequence = static_cast<int32_t>(p.getMaxBlocksPerSequence());
    if (max_num_tokens == 0)
    {
        return 0;
    }
    int const batch_beam = max_num_seq;

    bool isMLA = false;
    bool pagedSparseAttention = false;
    if constexpr (std::is_same_v<AttentionVariant, MHA>)
    {
        pagedSparseAttention = op.useTllmGenSparseAttentionPaged(p);
    }
    else
    {
        isMLA = op.mIsGenerationMLA;
        pagedSparseAttention
            = op.template feature<torch_ext::SparseMlaFeature>().pagedIndices() && op.use_sparse_attention;
    }

    size_t const cu_seqlens_size = sizeof(int) * (batch_beam + 1);
    size_t const cu_kv_seqlens_size = sizeof(int) * (batch_beam + 1);
    size_t const rotary_inv_freq_size = sizeof(float) * batch_beam * op.rotary_embedding_dim / 2;
    // Two workspaces for sparse attention. One for the sequence lengths, and one for kv block offsets.
    size_t const sparse_attn_cache_size = pagedSparseAttention
        ? sizeof(int) * (batch_beam + batch_beam * 2 * max_blocks_per_sequence) * op.mNumKVHeads
        : 0;
    AttentionXqaWorkspaceSizes xqaWorkspaceSizes{};
    xqaWorkspaceSizes.cuSeqlens = cu_seqlens_size;
    xqaWorkspaceSizes.cuKvSeqlens = cu_kv_seqlens_size;
    xqaWorkspaceSizes.rotaryInvFreq = rotary_inv_freq_size;
    xqaWorkspaceSizes.tokensInfo = max_num_tokens * sizeof(int2);
    xqaWorkspaceSizes.bmm1Scale = sizeof(float) * 2;
    xqaWorkspaceSizes.bmm2Scale = sizeof(float);
    xqaWorkspaceSizes.sparseAttnCache = sparse_attn_cache_size;
    xqaWorkspaceSizes.kernelWorkspace = xqaKernelWorkspaceSize(op.mHeadSize, op.mNumAttnHeads, op.mNumAttnKVHeads,
        isMLA, op.mMultiBlockMode,
        std::min<uint32_t>(
            op.template feature<torch_ext::SpecDecodingFeature>().maxGenerationLength() * max_num_seq, max_num_tokens));
    return AttentionWorkspaceManager::buildXqaLayout(xqaWorkspaceSizes,
        xqaKernelWorkspaceAlignment(op.mHeadSize, op.mNumAttnHeads, op.mNumAttnKVHeads, op.mMultiBlockMode))
        .totalSize;
}

template <typename AttentionVariant>
template <typename T>
void XqaFmha<AttentionVariant>::prepare(AttentionVariant const& op, FmhaParams const& p)
{
    // Batch size and window extents do not specialize XQA cubins. Only prepare
    // again for a new execution mode or a beam width beyond the covered range.
    bool const multiQueryTokens = op.template feature<torch_ext::SpecDecodingFeature>().active();
    bool const smallQueryTile = multiQueryTokens && p.input_seq_length * op.mNumAttnHeads / op.mNumAttnKVHeads < 16;
    auto const profile = std::make_tuple(multiQueryTokens, p.meta.is_spec_dec_tree, smallQueryTile,
        p.attn.attention_chunk_size.value_or(INT_MAX),
        p.fwd.sparse_runtime_params.threshold_scale_factor_decode != 0.0);
    auto& preparedBeamWidth = mPreparedBeamWidths[profile];
    if (p.meta.effective_beam_width > preparedBeamWidth)
    {
        TLLM_LOG_TRACE("Preparing XQA kernels.");
        XQAParams params{};
        if (fillParams<T>(op, p, params, /*forConfigurePlugin=*/true)
            && mRunner->shouldUse(params, /*forConfigurePlugin=*/true))
        {
            mRunner->prepare(params);
        }
        preparedBeamWidth = p.meta.effective_beam_width;
    }
}

template <typename AttentionVariant>
void XqaFmha<AttentionVariant>::runGeneration(AttentionVariant const& op, FmhaParams const& p, cudaStream_t stream)
{
    dispatchOnDtype(p.getQueryInput().scalar_type(),
        [&]<typename T>(std::type_identity<T>) { runGenerationImpl<T>(op, p, stream); });
}

template <typename AttentionVariant>
template <typename T>
void XqaFmha<AttentionVariant>::runGenerationImpl(AttentionVariant const& op, FmhaParams const& p, cudaStream_t stream)
{
    prepare<T>(op, p);
    XQAParams params{};
    TLLM_CHECK_WITH_INFO(
        fillParams<T>(op, p, params, /*forConfigurePlugin=*/false), "XQA kernels need FP16 or BF16 activations.");
    params.stream = stream;
    TLLM_LOG_DEBUG("XQA kernels are selected in the generation phase.");
    if constexpr (std::is_same_v<AttentionVariant, MHA>)
    {
        auto const kvBuffers = op.template getKvCacheBuffers<T>(p, /*isGen=*/true);
        mRunner->template dispatch<KVBlockArray>(params, kvBuffers.kvCacheBuffer, stream);
    }
    else
    {
        auto const kvCacheBuffer = op.template getGenerationKvCacheBuffer<T>(p);
        mRunner->template dispatch<KVBlockArray>(params, kvCacheBuffer, stream);
    }
}

// ---------------------------------------------------------------------------
// FlashMlaFmha
// ---------------------------------------------------------------------------

bool FlashMlaFmha::isAvailable(AttentionBase const& key)
{
    return tc::getSMVersion() == 90 && key.tokens_per_block == 64 && key.head_size == 576;
}

int FlashMlaFmha::getNumSmParts(int s_q, int numHeads, int numKvHeads, int multiProcessorCount)
{
    static constexpr int block_size_m = 64;
    int const num_heads_per_head_k = s_q * numHeads / numKvHeads;
    return multiProcessorCount / numKvHeads / divUp(num_heads_per_head_k, block_size_m);
}

bool FlashMlaFmha::isSupported(FmhaParams const& p, bool isGen) const
{
    return isGen;
}

std::size_t FlashMlaFmha::getWorkspaceSize(MLA const& op, FmhaParams const& p, bool isGen) const
{
    if (!isGen || p.num_gen_tokens == 0)
    {
        return 0;
    }
    static constexpr int TileSchedulerMetaDataSize = 8;

    int const batch_beam = static_cast<int32_t>(p.getMaxNumSequences());
    int const s_q = op.mMLAParams.predicted_tokens_per_seq;
    int const num_q_heads = op.num_heads;
    int const num_kv_heads = op.mNumKVHeads;
    int const head_size_v = (op.use_sparse_attention && !op.mMLAParams.rope_append)
        ? op.mMLAParams.kv_lora_rank + op.mMLAParams.qk_rope_head_dim
        : op.mMLAParams.kv_lora_rank;
    int const num_sm_parts = getNumSmParts(s_q, num_q_heads, num_kv_heads, op.mMultiProcessorCount);

    AttentionFlashMlaWorkspaceSizes flashMlaWorkspaceSizes{};
    flashMlaWorkspaceSizes.tileSchedulerMetadata = sizeof(int) * (num_sm_parts * TileSchedulerMetaDataSize);
    flashMlaWorkspaceSizes.numSplits = sizeof(int) * (batch_beam + 1);
    flashMlaWorkspaceSizes.softmaxLse = sizeof(float) * (batch_beam * s_q * num_q_heads);
    flashMlaWorkspaceSizes.softmaxLseAccum = sizeof(float) * ((batch_beam + num_sm_parts) * num_q_heads * s_q);
    flashMlaWorkspaceSizes.outAccum = sizeof(float) * ((batch_beam + num_sm_parts) * num_q_heads * s_q * head_size_v);
    return op.getGenerationWorkspaceSize(
        p, AttentionWorkspaceManager::buildFlashMlaLayout(flashMlaWorkspaceSizes).totalSize);
}

void FlashMlaFmha::runGeneration(MLA const& op, FmhaParams const& p, cudaStream_t stream)
{
    TORCH_CHECK(p.meta.block_ids_per_seq.has_value());
    TORCH_CHECK(!p.meta.flash_mla_tile_scheduler_metadata.has_value() || p.meta.flash_mla_num_splits.has_value(),
        "flash_mla_num_splits must be provided when flash_mla_tile_scheduler_metadata is set.");
    dispatchOnDtype(p.getQueryInput().scalar_type(),
        [&]<typename T>(std::type_identity<T>) { runGenerationImpl<T>(op, p, stream); });
}

template <typename T>
void FlashMlaFmha::runGenerationImpl(MLA const& op, FmhaParams const& p, cudaStream_t stream)
{
    auto params = op.template getGenerationParams<T>(p);
    params.block_ids_per_seq = p.getBlockIdsPerSeq();
    if (p.meta.flash_mla_tile_scheduler_metadata.has_value())
    {
        params.flash_mla_tile_scheduler_metadata = p.getFlashMlaTileSchedulerMetadata();
        params.flash_mla_num_splits = p.getFlashMlaNumSplits();
    }
    auto const kv_cache_buffer = op.template getGenerationKvCacheBuffer<T>(p);
    int const num_kv_heads = 1;
    int const head_size = op.mMLAParams.kv_lora_rank + op.mMLAParams.qk_rope_head_dim;
    int32_t const batch_beam = p.meta.effective_beam_width * p.num_requests;
    double const q_scaling = op.q_scaling;

    static constexpr int TileSchedulerMetaDataSize = 8;

    int const num_q_heads = op.num_heads;
    int const ngroups = num_q_heads / num_kv_heads;

    int const s_q = params.acc_q_len / batch_beam;
    assert(s_q == op.mMLAParams.predicted_tokens_per_seq);
    int const head_size_v = op.mMLAParams.kv_lora_rank;
    int const num_sm_parts = getNumSmParts(s_q, num_q_heads, num_kv_heads, op.mMultiProcessorCount);

    size_t const num_splits_size = sizeof(int) * (batch_beam + 1);
    size_t const tile_scheduler_metadata_size = sizeof(int) * (num_sm_parts * TileSchedulerMetaDataSize);
    size_t const softmax_lse_size = sizeof(float) * (batch_beam * s_q * num_q_heads * num_kv_heads); // softmax_lse
    size_t const softmax_lse_accum_size = sizeof(float) * ((batch_beam + num_sm_parts) * num_q_heads * s_q);
    size_t const out_accum_size = sizeof(float) * ((batch_beam + num_sm_parts) * num_q_heads * s_q * head_size_v);

    AttentionFlashMlaWorkspaceSizes flashMlaWorkspaceSizes{};
    flashMlaWorkspaceSizes.tileSchedulerMetadata = tile_scheduler_metadata_size;
    flashMlaWorkspaceSizes.numSplits = num_splits_size;
    flashMlaWorkspaceSizes.softmaxLse = softmax_lse_size;
    flashMlaWorkspaceSizes.softmaxLseAccum = softmax_lse_accum_size;
    flashMlaWorkspaceSizes.outAccum = out_accum_size;
    auto const flashMlaWorkspaceLayout = AttentionWorkspaceManager::buildFlashMlaLayout(flashMlaWorkspaceSizes);
    float* softmax_lse_ptr
        = AttentionWorkspaceManager::ptr<float>(params.workspace, flashMlaWorkspaceLayout.softmaxLse);
    float* softmax_lse_accum_ptr
        = AttentionWorkspaceManager::ptr<float>(params.workspace, flashMlaWorkspaceLayout.softmaxLseAccum);
    float* out_accum_ptr = AttentionWorkspaceManager::ptr<float>(params.workspace, flashMlaWorkspaceLayout.outAccum);

    // Metadata must always be pre-computed by Python (compute_flash_mla_metadata) and passed in.
    TLLM_CHECK_WITH_INFO(params.flash_mla_tile_scheduler_metadata != nullptr,
        "FlashMLA tile-scheduler metadata must be pre-computed by Python.");
    TLLM_CHECK_WITH_INFO(params.flash_mla_num_splits != nullptr, "FlashMLA num_splits must be pre-computed by Python.");
    int* tile_scheduler_metadata_ptr = const_cast<int*>(params.flash_mla_tile_scheduler_metadata);
    int* num_splits_ptr = const_cast<int*>(params.flash_mla_num_splits);

    Flash_fwd_mla_params flashMlaParams{};
    flashMlaParams.b = batch_beam;
    flashMlaParams.seqlen_q = ngroups * s_q;
    flashMlaParams.cu_seqlens_k = const_cast<int*>(params.cache_seq_lens);
    flashMlaParams.h = 1;
    flashMlaParams.h_h_k_ratio = 1;

    float softmax_scale
        = 1.0f / (q_scaling * sqrtf((op.mMLAParams.qk_nope_head_dim + op.mMLAParams.qk_rope_head_dim) * 1.0f));

    flashMlaParams.ngroups = ngroups;
    flashMlaParams.is_causal = !(s_q == 1);
    flashMlaParams.d = head_size;
    flashMlaParams.d_v = head_size_v;
    flashMlaParams.scale_softmax = softmax_scale;
    flashMlaParams.scale_softmax_log2 = float(softmax_scale * M_LOG2E);

    flashMlaParams.q_ptr = op.mFP8GenerationMLA ? const_cast<void*>(reinterpret_cast<void const*>(params.quant_q_buf))
                                                : const_cast<void*>(reinterpret_cast<void const*>(params.q_buf));
    flashMlaParams.k_ptr = kv_cache_buffer.mPrimaryPoolPtr;
    flashMlaParams.v_ptr = flashMlaParams.k_ptr;
    flashMlaParams.o_ptr = reinterpret_cast<void*>(params.context_buf);
    flashMlaParams.softmax_lse_ptr = softmax_lse_ptr;

    // since head_num_kv = 1
    flashMlaParams.q_batch_stride = head_size * params.head_num * s_q;
    flashMlaParams.k_batch_stride = op.tokens_per_block * num_kv_heads * head_size * op.mMLAParams.num_layers;
    flashMlaParams.o_batch_stride = s_q * num_q_heads * head_size_v;
    flashMlaParams.q_row_stride = head_size;
    flashMlaParams.k_row_stride = head_size;
    flashMlaParams.o_row_stride = head_size_v;
    flashMlaParams.q_head_stride = head_size;
    flashMlaParams.k_head_stride = head_size;
    flashMlaParams.o_head_stride = head_size_v;

    flashMlaParams.v_batch_stride = flashMlaParams.k_batch_stride;
    flashMlaParams.v_row_stride = flashMlaParams.k_row_stride;
    flashMlaParams.v_head_stride = flashMlaParams.k_head_stride;

    flashMlaParams.block_table = const_cast<int*>(params.block_ids_per_seq);
    flashMlaParams.block_table_batch_stride = p.getMaxBlocksPerSequence();
    flashMlaParams.page_block_size = op.tokens_per_block;

    flashMlaParams.descale_q_ptr = const_cast<float*>(params.dequant_scale_q);
    flashMlaParams.descale_k_ptr = const_cast<float*>(params.dequant_scale_kv);

    flashMlaParams.tile_scheduler_metadata_ptr = tile_scheduler_metadata_ptr;
    flashMlaParams.num_sm_parts = num_sm_parts;
    flashMlaParams.num_splits_ptr = num_splits_ptr;

    flashMlaParams.softmax_lseaccum_ptr = softmax_lse_accum_ptr;
    flashMlaParams.oaccum_ptr = out_accum_ptr;

    if constexpr (std::is_same_v<T, half>)
    {
        if (op.mFP8GenerationMLA)
        {
            TLLM_THROW("FP8 KV cache MLA is only supported for bf16 output");
        }
        else
        {
            run_mha_fwd_splitkv_mla<cutlass::half_t, cutlass::half_t, 576>(flashMlaParams, stream);
        }
    }
    else if constexpr (std::is_same_v<T, __nv_bfloat16>)
    {
        if (op.mFP8GenerationMLA)
        {
            run_mha_fwd_splitkv_mla<cutlass::float_e4m3_t, cutlass::bfloat16_t, 576>(flashMlaParams, stream);
        }
        else
        {
            run_mha_fwd_splitkv_mla<cutlass::bfloat16_t, cutlass::bfloat16_t, 576>(flashMlaParams, stream);
        }
    }
    else
    {
        TLLM_THROW("Unsupported data type for FlashMLA");
    }
}

// ---------------------------------------------------------------------------
// MmhaFmha
// ---------------------------------------------------------------------------

bool MmhaFmha::isAvailable(AttentionBase const& key)
{
    return mmha_supported(static_cast<int>(key.head_size));
}

bool MmhaFmha::isSupported(FmhaParams const& p, bool isGen) const
{
    if (!isGen)
    {
        return false;
    }
    auto const& op = static_cast<MHA const&>(this->attention);
    TLLM_CHECK_WITH_INFO(!op.template feature<torch_ext::SpecDecodingFeature>().active(),
        "No available XQA kernels are found for speculative decoding mode.");
    TLLM_CHECK_WITH_INFO(!op.is_fp4_out, "No available kernels are found for FP4 output.");
    TLLM_CHECK_WITH_INFO(!op.quant_mode.hasFp4KvCache(), "No available kernels are found for FP4 KV cache.");
    return true;
}

void MmhaFmha::initGeneration(MHA const& op)
{
    // Reading these caches the parsed values, keeping getenv() off the per-call path.
    tc::getEnvMmhaMultiblockDebug();
    tc::getEnvMmhaBlocksPerSequence();
}

int MmhaFmha::getMaxNumSeqLenTile(MHA const& op, int batchBeamSize) const
{
    if (op.mMultiBlockMode)
    {
        // And we allocate the buffer based on the maximum number of blocks per sequence (batchBeamSize = 1).
        // Assume we can only have 1 block (large block size like 1024) in SM, and we only want one wave of blocks.
        return tc::getEnvMmhaMultiblockDebug()
            ? std::max(kReservedMaxSeqLenTilePerSeq, tc::getEnvMmhaBlocksPerSequence())
            : tc::divUp(op.mMultiProcessorCount, batchBeamSize * op.num_heads);
    }
    return 0;
}

std::size_t MmhaFmha::getWorkspaceSize(MHA const& op, FmhaParams const& p, bool isGen) const
{
    if (!isGen)
    {
        return 0;
    }
    // Sized per sequence (max_num_requests * beam_width) from the batch bound, not from this
    // call's rows, so a context call sizing a mixed batch and the following generation call
    // compute the same number.
    auto const max_num_seq = static_cast<int32_t>(p.getMaxNumSequences());
    auto const max_attention_window_size = static_cast<int32_t>(p.max_attention_window_size);
    auto const max_num_tokens = static_cast<int32_t>(p.num_gen_tokens);
    if (max_num_tokens == 0)
    {
        return 0;
    }

    auto const size = tensorrt_llm::runtime::BufferDataType(p.getType()).getSize();
    int const batch_beam = max_num_seq;

    size_t generation_workspace_size = 0;
    // The minimum number of sequence length tiles (limited by the shared memory size).
    int minSeqLenTile
        = estimate_min_multi_block_count(max_attention_window_size, op.mMaxSharedMemoryPerBlockOptin - 2048, size);
    int32_t const maxSeqLenTile = std::max(
        {minSeqLenTile, getMaxNumSeqLenTile(op, batch_beam), (int) tc::divUp(op.mMultiProcessorCount, op.num_heads)});

    size_t const partial_out_size = size * batch_beam * op.num_heads * op.mHeadSize * maxSeqLenTile;
    size_t const partial_sum_size = sizeof(float) * batch_beam * op.num_heads * maxSeqLenTile;
    size_t const partial_max_size = sizeof(float) * batch_beam * op.num_heads * maxSeqLenTile;
    AttentionGenerationWorkspaceSizes generationWorkspaceSizes{};
    generationWorkspaceSizes.partialOut = partial_out_size;
    generationWorkspaceSizes.partialSum = partial_sum_size;
    generationWorkspaceSizes.partialMax = partial_max_size;
    {
        auto const cascadeSizes
            = tensorrt_llm::kernels::mmha::cascade::getCascadeWorkspaceSizes(batch_beam, op.num_heads, op.mHeadSize);
        generationWorkspaceSizes.cascadeOut = cascadeSizes.out;
        generationWorkspaceSizes.cascadeMax = cascadeSizes.mMax;
        generationWorkspaceSizes.cascadeSum = cascadeSizes.lSum;
    }
    generation_workspace_size = AttentionWorkspaceManager::buildGenerationLayout(generationWorkspaceSizes).totalSize;
    return generation_workspace_size;
}

void MmhaFmha::runGeneration(MHA const& op, FmhaParams const& p, cudaStream_t stream)
{
    dispatchOnDtype(p.getQueryInput().scalar_type(),
        [&]<typename T>(std::type_identity<T>) { runGenerationImpl<T>(op, p, stream); });
}

template <typename T>
void MmhaFmha::runGenerationImpl(MHA const& op, FmhaParams const& p, cudaStream_t stream)
{
    float const q_scaling = static_cast<float>(op.q_scaling);
    bool const* finished = nullptr;

    auto const quant_option = tc::QuantMode{};
    float const* qkv_scale_out = nullptr;

    int const* ia3_tasks = nullptr;
    T const* ia3_key_weights = nullptr;
    T const* ia3_value_weights = nullptr;

    int32_t const batch_beam = p.meta.effective_beam_width * p.num_requests;
    auto const kvBuffers = op.template getKvCacheBuffers<T>(p, /*isGen=*/true);
    auto const& kv_cache_buffer = kvBuffers.kvCacheBuffer;
    using KVCacheBuffer = KVBlockArray;
    T* attention_input = p.getQueryInput<T>();

    // This is the number of kv tokens that q needs to visit, but excluding one as it will be processed before the kv
    // loop.
    int timestep = p.max_past_kv_length;
    int const max_timesteps = std::min(timestep, static_cast<int>(p.cyclic_attention_window_size));
    int estimated_min_multi_block_count
        = estimate_min_multi_block_count(max_timesteps, op.mMaxSharedMemoryPerBlockOptin - 2048, sizeof(T));

    if (!op.mMultiBlockMode && !mForceMultiBlockWarned && estimated_min_multi_block_count > 1)
    {
        mForceMultiBlockWarned = true;
        TLLM_LOG_WARNING(
            "Force using MultiBlockMode in MMHA as shared memory is not enough, "
            "MultiBlockMode may have different accuracy compared to non-MultiBlockMode.");
    }

    // estimate min block count to satisfy shared memory requirement to run kernel.
    // Runtime check to see the actual number of blocks per sequence we need.
    int32_t const max_num_seq_len_tiles
        = std::max(getMaxNumSeqLenTile(op, batch_beam), estimated_min_multi_block_count);
    int32_t const min_num_seq_len_tiles = std::max(1, estimated_min_multi_block_count);
    bool const enable_multi_block
        = (op.mMultiBlockMode && max_num_seq_len_tiles > 1) || estimated_min_multi_block_count > 1;
    size_t const partial_out_size
        = enable_multi_block ? sizeof(T) * batch_beam * op.num_heads * op.mHeadSize * max_num_seq_len_tiles : 0;
    size_t const partial_sum_size
        = enable_multi_block ? sizeof(float) * batch_beam * op.num_heads * max_num_seq_len_tiles : 0;
    size_t const partial_max_size
        = enable_multi_block ? sizeof(float) * batch_beam * op.num_heads * max_num_seq_len_tiles : 0;

    AttentionGenerationWorkspaceSizes workspaceSizes{};
    workspaceSizes.partialOut = partial_out_size;
    workspaceSizes.partialSum = partial_sum_size;
    workspaceSizes.partialMax = partial_max_size;
    {
        auto const cascadeSizes
            = tensorrt_llm::kernels::mmha::cascade::getCascadeWorkspaceSizes(batch_beam, op.num_heads, op.mHeadSize);
        workspaceSizes.cascadeOut = cascadeSizes.out;
        workspaceSizes.cascadeMax = cascadeSizes.mMax;
        workspaceSizes.cascadeSum = cascadeSizes.lSum;
    }
    auto const workspaceLayout = AttentionWorkspaceManager::buildGenerationLayout(workspaceSizes);
    auto const workspaceViews = AttentionWorkspaceManager::materializeGeneration<T>(p.getWorkspace(), workspaceLayout);

    FusedQKVMaskedAttentionDispatchParams<T, KVCacheBuffer> dispatch_params{};
    dispatch_params.qkv_buf = attention_input;
    dispatch_params.attention_sinks = p.getAttentionSinks();
    dispatch_params.cache_indir = p.getCacheIndirection();
    dispatch_params.context_buf = p.getOutput(); //
    dispatch_params.finished = finished;
    dispatch_params.sequence_lengths
        = p.getSequenceLengths(); // NOTE: current seq len including padding (fixed after meeting the finished id)
    dispatch_params.max_batch_size = batch_beam;
    dispatch_params.inference_batch_size = batch_beam;
    dispatch_params.beam_width = p.meta.effective_beam_width;
    dispatch_params.head_num = op.mNumAttnHeads;
    dispatch_params.kv_head_num = op.mNumAttnKVHeads;
    dispatch_params.size_per_head = op.getHeadSize();
    dispatch_params.chunked_attention_size = p.attn.attention_chunk_size ? *p.attn.attention_chunk_size : INT_MAX;
    dispatch_params.max_attention_window_size = p.max_attention_window_size;
    dispatch_params.cyclic_attention_window_size = p.cyclic_attention_window_size;
    dispatch_params.input_lengths = p.getContextLengths();
    dispatch_params.timestep = timestep;
    dispatch_params.q_scaling = q_scaling;
    dispatch_params.attn_logit_softcapping_scale = op.attn_logit_softcapping_scale;
    dispatch_params.ia3_tasks = ia3_tasks;
    dispatch_params.ia3_key_weights = ia3_key_weights;
    dispatch_params.ia3_value_weights = ia3_value_weights;
    dispatch_params.qkv_scale_out = qkv_scale_out;
    dispatch_params.fp8_context_fmha = op.mFP8ContextFMHA;
    dispatch_params.attention_out_scale = op.getOutQuantScale(p);
    dispatch_params.quant_option = quant_option;
    dispatch_params.multi_block_mode = enable_multi_block;
    dispatch_params.max_seq_len_tile = max_num_seq_len_tiles;
    dispatch_params.min_seq_len_tile = min_num_seq_len_tiles;
    dispatch_params.partial_out = workspaceViews.partialOut;
    dispatch_params.partial_sum = workspaceViews.partialSum;
    dispatch_params.partial_max = workspaceViews.partialMax;
    dispatch_params.cascade_partial_out = workspaceViews.cascadeOut;
    dispatch_params.cascade_partial_max = workspaceViews.cascadeMax;
    dispatch_params.cascade_partial_sum = workspaceViews.cascadeSum;
    dispatch_params.block_counter = op.getMultiCtasKvCounter(p.getMaxNumSequences(), p.getQueryInput().device());
    dispatch_params.kv_cache_quant_mode = op.quant_mode;
    dispatch_params.kv_scale_orig_quant = p.getKvScaleOrigQuant();
    dispatch_params.kv_scale_quant_orig = p.getKvScaleQuantOrig();
    dispatch_params.kv_block_array = kv_cache_buffer;
    dispatch_params.multi_processor_count = op.mMultiProcessorCount;
    dispatch_params.block_sparse_attention = op.mask_type == AttentionMaskType::BLOCKSPARSE;
    op.applyFeatures(dispatch_params);

    using DataType = typename SATypeConverter<T>::Type;
    {
        if (!op.isCrossAttention(p))
        {
            // self attn
            Masked_multihead_attention_params<DataType> mmha_params;
            fusedQKV_masked_attention_dispatch(mmha_params, dispatch_params, stream);
        }
        else
        {
            // cross attn
            Cross_multihead_attention_params<DataType> mmhca_params;
            fusedQKV_masked_attention_dispatch(mmhca_params, dispatch_params, stream);
        }
        sync_check_cuda_error(stream);
    }
}

// ---------------------------------------------------------------------------
// UnfusedFmha
// ---------------------------------------------------------------------------

bool UnfusedFmha::isAvailable(AttentionBase const& key)
{
    return true;
}

bool UnfusedFmha::isSupported(FmhaParams const& p, bool isGen) const
{
    if (isGen)
    {
        return false;
    }
    // This library comes last, so the call has no fused context kernel. Refuse what the
    // unfused path would compute wrongly: variable-window bounds it ignores, and a cached
    // prefix that unfused self-attention cannot read.
    auto const& op = this->attention;
    TLLM_CHECK_WITH_INFO(
        !op.use_variable_window, "Variable-window attention requires a supported context FMHA kernel configuration.");
    bool const needsFusedPagedContext = op.paged_context_fmha && op.use_kv_cache && !op.cross_attention;
    TLLM_CHECK_WITH_INFO(!needsFusedPagedContext,
        "Paged-context attention requires a fused context FMHA kernel, and this build has none for this "
        "configuration. The unfused fallback cannot attend to cached KV. If the device's SM is not named in the "
        "build's --cuda_architectures then the build carries no kernels for it at all; check that first. Otherwise "
        "use another attention backend, or disable chunked prefill, KV cache reuse and speculative decoding.");
    return true;
}

void UnfusedFmha::initContext(MHA const& op)
{
    mCublasWrapper = std::make_unique<tc::CublasMMWrapper>(getCublasHandle(), getCublasLtHandle(), nullptr, nullptr);
}

std::size_t UnfusedFmha::getWorkspaceSize(MHA const& op, FmhaParams const& p, bool isGen) const
{
    return isGen || p.num_tokens == 0 ? 0 : contextWorkspaceLayout(op, p).totalSize;
}

AttentionContextWorkspaceLayout UnfusedFmha::contextWorkspaceLayout(MHA const& op, FmhaParams const& p) const
{
    auto const input_seq_length = static_cast<int32_t>(p.input_seq_length);
    int32_t const cross_kv_length = op.feature<torch_ext::CrossAttentionFeature>().crossKvLength();

    int const local_hidden_units_qo = op.mNumAttnHeads * op.getHeadSize();
    int const local_hidden_units_kv = op.mNumAttnKVHeads * op.getHeadSize();

    auto const size = tensorrt_llm::runtime::BufferDataType(p.getType()).getSize();

    auto const batch_size = static_cast<size_t>(p.batch_size);
    auto const kv_seq_length = (op.isCrossAttention(p) ? cross_kv_length : input_seq_length);

    auto sizes = op.contextWorkspaceSizes(p);
    // These buffers must upper-bound the carve in runContextImpl, which sizes them by
    // batch_size * input_seq_length (not num_tokens): with padding removal the actual token count can be
    // smaller than batch_size * max(context q length), so sizing by num_tokens underestimates.
    sizes.attentionMask = size * batch_size * input_seq_length * kv_seq_length;
    sizes.qBuf = size * batch_size * input_seq_length * local_hidden_units_qo;
    sizes.kBuf = size * batch_size * kv_seq_length * local_hidden_units_kv;
    sizes.vBuf = size * batch_size * kv_seq_length * local_hidden_units_kv;
    sizes.qkBuf = size * batch_size * op.num_heads * input_seq_length * kv_seq_length;
    sizes.qkvBuf = size * batch_size * input_seq_length * local_hidden_units_qo;
    sizes.qkFloatBuf = sizeof(float) * batch_size * op.num_heads * input_seq_length * kv_seq_length;
    sizes.fp8QkvBuf = 0;
    sizes.fp8QBuf = 0;
    sizes.fp8KBuf = 0;
    sizes.fp8VBuf = 0;
    sizes.paddingOffset = sizeof(int) * batch_size * input_seq_length;
    sizes.encoderPaddingOffset = sizeof(int) * batch_size * cross_kv_length;
    sizes.fmhaTileCounter = 0;
    return AttentionWorkspaceManager::buildContextLayout(sizes);
}

void UnfusedFmha::runContext(MHA const& op, FmhaParams const& p, cudaStream_t stream)
{
    // Without encoder K/V input, cross attention must read the cached encoder KV.
    TLLM_CHECK_WITH_INFO(p.fwd.cross_kv.has_value() || !p.is_cross,
        "Cross attention without encoder K/V input requires a fused context FMHA kernel to read the "
        "cached cross KV, and this build has none for this configuration. Disable chunked prefill for "
        "this model, or use a build whose --cuda_architectures includes this device's SM.");
    TLLM_CHECK_WITH_INFO(!op.sageAttnRequested() || op.useSageAttn(),
        "SageAttention requires the FP8 context FMHA path without KV caching and does not apply to MLA.");
    TLLM_CHECK_WITH_INFO(!op.useSageAttn(),
        "Sage Attention requires contextFMHA with no unfused fallback, but the supplied configuration is unsupported.");
    TLLM_CHECK_WITH_INFO(!op.useCustomMask(), "Only Context FMHA supports custom mask input currently.");
    dispatchOnDtype(
        p.getQueryInput().scalar_type(), [&]<typename T>(std::type_identity<T>) { runContextImpl<T>(op, p, stream); });
    sync_check_cuda_error(stream);
}

template <typename T>
void UnfusedFmha::runContextImpl(MHA const& op, FmhaParams const& p, cudaStream_t stream)
{
    PositionEmbeddingType const position_embedding_type = op.position_embedding_type;
    float const q_scaling = static_cast<float>(op.q_scaling);
    auto kvBuffers = op.template getKvCacheBuffers<T>(p, /*isGen=*/false);
    auto& kv_cache_buffer = kvBuffers.kvCacheBuffer;

    auto cublasHandle = mCublasWrapper->getCublasHandle();
    TLLM_CUDA_CHECK(cublasSetStream(cublasHandle, stream));
    mCublasWrapper->setStream(stream);
    mCublasWrapper->setWorkspace(p.getWorkspace());
    if constexpr (std::is_same_v<T, half>)
    {
        mCublasWrapper->setFP16GemmConfig();
    }
    else if constexpr (std::is_same_v<T, float>)
    {
        mCublasWrapper->setFP32GemmConfig();
    }
#ifdef ENABLE_BF16
    else if constexpr (std::is_same_v<T, __nv_bfloat16>)
    {
        mCublasWrapper->setBF16GemmConfig();
    }
#endif

    auto const& cross = op.template feature<torch_ext::CrossAttentionFeature>();
    auto const& relativePosition = op.template feature<torch_ext::RelativePositionFeature>();
    auto const workspaceLayout = contextWorkspaceLayout(op, p);
    bool const is_qk_buf_float_ = true;
    auto const workspaceViews = AttentionWorkspaceManager::materializeContext<T>(p.getWorkspace(), workspaceLayout);
    op.template buildContextDecoderInfo<T>(p, workspaceViews, stream);

    // In cross attention context phase, the attention mask should be a matrix of all ones.
    // Override the attention mask produced by invokeBuildDecoderInfo().
    // also, invokeBuildDecoderInfo can only handle square mask, not cross B x q_len x kv_len mask
    // TODO: put this logic in the kernel above. currently not much concern because q_len is mostly = 1
    if (p.is_cross)
    {
        int32_t const crossKvLength = cross.crossKvLength();
        std::vector<T> h_attention_mask(p.batch_size * p.input_seq_length * crossKvLength, 1.);
        std::vector<int32_t> h_encoder_input_lengths(p.batch_size);
        tensorrt_llm::common::cudaMemcpyAsyncSanitized(h_encoder_input_lengths.data(), cross.encoderInputLengths(),
            sizeof(int32_t) * p.batch_size, cudaMemcpyDeviceToHost, stream);
        sync_check_cuda_error(stream);

        for (int bi = 0; bi < p.batch_size; bi++)
        {
            int b_offset = bi * p.input_seq_length * crossKvLength;
            for (int qi = 0; qi < p.input_seq_length; qi++)
            {
                int q_offset = b_offset + qi * crossKvLength;
                if (h_encoder_input_lengths[bi] < crossKvLength)
                {
                    std::fill(h_attention_mask.begin() + q_offset + h_encoder_input_lengths[bi],
                        h_attention_mask.begin() + q_offset + crossKvLength, 0.f);
                }
            }
        }
        cudaMemcpyAsync(workspaceViews.attentionMask, h_attention_mask.data(),
            sizeof(T) * p.batch_size * crossKvLength * p.input_seq_length, cudaMemcpyHostToDevice, stream);
        sync_check_cuda_error(stream);
    }

    KvCacheDataType cache_type = cacheTypeFromQuantMode(op.quant_mode);

    cudaDataType_t const gemm_data_type = tc::CudaDataType<T>::value;
    int const attention_seq_len_1 = p.input_seq_length;                                          // q length
    int const attention_seq_len_2 = cross.active() ? cross.crossKvLength() : p.input_seq_length; // kv length

    // If the model has relative attentiona bias, q scaling should be applied in QK gemm stage and use 1 in
    // softamax stage (because to get softmax[scale(Q*K) + rel pos bias] here, q_scaling can't be applied during
    // softmax phase by qk_scale); otherwise, use 1 in gemm stage and apply scaling in softmax stage
    float const qk_scale
        = 1.0f / (sqrtf(op.getHeadSize() * 1.0f) * q_scaling); // q_scaling in denominator. by default q_scaling =1.0f
    float const qk_scale_gemm = op.isRelativePosition() ? qk_scale : 1.0f;
    T const qk_scale_softmax = static_cast<T>(op.isRelativePosition() ? 1.0f : qk_scale);

    TLLM_CHECK_WITH_INFO(p.attn.attention_chunk_size == std::nullopt, "Unfused MHA does not support chunked attention");
    // FIXME: a temporary solution to make sure the padding part of key/value buffer is 0
    // NOTE: pointer subtraction is used below since there could be some extra gap due to alignment.
    //  Otherwise, we could do cudaMemsetAsync(workspaceViews.kBuf, 0, k_buf_2_size + v_buf_2_size, stream).
    // cudaMemsetAsync(workspaceViews.kBuf, 0,
    //     reinterpret_cast<int8_t*>(workspaceViews.qkBuf) - reinterpret_cast<int8_t*>(workspaceViews.kBuf),
    //     stream);
    cudaMemsetAsync(workspaceViews.kBuf, 0,
        reinterpret_cast<int8_t*>(workspaceViews.vBuf) - reinterpret_cast<int8_t*>(workspaceViews.kBuf)
            + workspaceLayout.vBuf.size,
        stream);

    if (!op.isCrossAttention(p))
    {
        // self attention, write to from QKV to Q/K/V
        invokeAddFusedQKVBiasTranspose(workspaceViews.qBuf, workspaceViews.kBuf, workspaceViews.vBuf,
            p.getQueryInput<T>(), (T const*) nullptr, p.getContextLengths(),
            op.remove_padding ? workspaceViews.paddingOffset : nullptr, p.batch_size, p.input_seq_length, p.num_tokens,
            op.num_heads, op.mNumKVHeads, op.getHeadSize(), op.rotary_embedding_dim, p.attn.rope_params.theta,
            p.attn.rope_params.scale_type, p.attn.rope_params.scale, p.attn.rope_params.max_positions,
            position_embedding_type, (float*) nullptr, 0, stream);
        sync_check_cuda_error(stream);
    }
    else
    {
        // cross attention, write from self QKV [*, head_num * head_size + 2 * kv_head_num * head_size]to Q, write
        // from cross KV [*, 2 * kv_head_num * head_size] to K/V kernel modified accordingly to handle nullptr
        // buffer
        invokeAddFusedQKVBiasTranspose(workspaceViews.qBuf, (T*) nullptr, (T*) nullptr, p.getQueryInput<T>(),
            (T const*) nullptr, p.getContextLengths(), op.remove_padding ? workspaceViews.paddingOffset : nullptr,
            p.batch_size, p.input_seq_length, p.num_tokens, op.num_heads, op.mNumKVHeads, op.getHeadSize(),
            op.rotary_embedding_dim, p.attn.rope_params.theta, p.attn.rope_params.scale_type, p.attn.rope_params.scale,
            p.attn.rope_params.max_positions, position_embedding_type, (float*) nullptr, 0, stream);
        sync_check_cuda_error(stream);

        invokeAddFusedQKVBiasTranspose((T*) nullptr, workspaceViews.kBuf, workspaceViews.vBuf, cross.encoderKv<T>(),
            (T const*) nullptr, cross.encoderInputLengths(),
            op.remove_padding ? workspaceViews.encoderPaddingOffset : nullptr, p.batch_size, cross.crossKvLength(),
            cross.numEncoderTokens(), /*op.num_heads*/ 0, op.mNumKVHeads, op.getHeadSize(), op.rotary_embedding_dim,
            p.attn.rope_params.theta, p.attn.rope_params.scale_type, p.attn.rope_params.scale,
            p.attn.rope_params.max_positions, position_embedding_type, (float*) nullptr, 0, stream);
        sync_check_cuda_error(stream);
    }

    // write KV to cache
    if (op.useKVCache(p))
    {
        invokeTranspose4dBatchMajor(workspaceViews.kBuf, workspaceViews.vBuf, kv_cache_buffer, p.batch_size,
            cross.active() ? cross.crossKvLength() : p.input_seq_length,
            cross.active() ? cross.crossKvLength() : p.cyclic_attention_window_size, op.getHeadSize(), op.mNumKVHeads,
            cache_type, p.getKvScaleOrigQuant(), cross.active() ? cross.encoderInputLengths() : p.getContextLengths(),
            stream);
    }
    sync_check_cuda_error(stream);

    T const* linear_bias_slopes = nullptr;
    T const* relative_attention_bias = relativePosition.relativeAttentionBias<T>();
    int const relative_attention_bias_stride = relativePosition.biasStride();
    int const max_distance = relativePosition.maxDistance();
    cudaDataType_t gemm_out_data_type = is_qk_buf_float_ ? CUDA_R_32F : gemm_data_type;
    void* gemm_out_buf_
        = is_qk_buf_float_ ? static_cast<void*>(workspaceViews.qkFloatBuf) : static_cast<void*>(workspaceViews.qkBuf);
    if (op.mNumKVHeads == 1) // MQA
    {
        // Attn_weight[b, h*s_q, s_k] = Q[b, h*s_q, d] * K'[b, d, s_k]
        // Attn_weight'[b, s_k, h*s_q] = K[b, s_k, d] * Q'[b, d, h*s_q]
        mCublasWrapper->stridedBatchedGemm(CUBLAS_OP_T, CUBLAS_OP_N,
            attention_seq_len_2,                                                            // n
            attention_seq_len_1 * op.num_heads,                                             // m
            op.getHeadSize(),                                                               // k
            qk_scale_gemm, workspaceViews.kBuf, gemm_data_type,
            op.getHeadSize(),                                                               // k
            static_cast<int64_t>(attention_seq_len_2) * op.getHeadSize(),                   // n * k
            workspaceViews.qBuf, gemm_data_type,
            op.getHeadSize(),                                                               // k
            static_cast<int64_t>(attention_seq_len_1) * op.num_heads * op.getHeadSize(),    // m * k
            0.0f, gemm_out_buf_, gemm_out_data_type,
            attention_seq_len_2,                                                            // n
            static_cast<int64_t>(attention_seq_len_1) * op.num_heads * attention_seq_len_2, // m * n
            p.batch_size,                                                                   // global batch size
            CUDA_R_32F);
    }
    else if (op.mNumKVHeads == op.num_heads) // MHA
    {
        // Attn_weight[b*h, s_q, s_k] = Q[b*h, s_q, d] * K'[b*h, d, s_k]
        // Attn_weight'[b*h, s_k, s_q] = K[b*h, s_k, d] * Q'[b*h, d, s_q]
        mCublasWrapper->stridedBatchedGemm(CUBLAS_OP_T, CUBLAS_OP_N,
            attention_seq_len_2,                                          // n
            attention_seq_len_1,                                          // m
            op.getHeadSize(),                                             // k
            qk_scale_gemm, workspaceViews.kBuf, gemm_data_type,
            op.getHeadSize(),                                             // k
            static_cast<int64_t>(attention_seq_len_2) * op.getHeadSize(), // n * k
            workspaceViews.qBuf, gemm_data_type,
            op.getHeadSize(),                                             // k
            static_cast<int64_t>(attention_seq_len_1) * op.getHeadSize(), // m * k
            0.0f, gemm_out_buf_, gemm_out_data_type,
            attention_seq_len_2,                                          // n
            static_cast<int64_t>(attention_seq_len_2) * attention_seq_len_1,
            p.batch_size * op.num_heads,                                  // global batch size
            CUDA_R_32F);
    }
    else // GQA
    {
        // Some number of contiguous Q heads will share the same K/V head
        // Since the KV stride is NOT fixed for all Q, we have 2 options:
        //  1. Loop over stridedBatchedGemm for each KV head. (multiple API calls/cuda kernels)
        //  2. Calculate the pointers and use batchedGemm() (extra device memory) ::TODO::
        int const num_qheads_per_kv_head = op.num_heads / op.mNumKVHeads;
        for (int ki = 0; ki < op.mNumKVHeads; ++ki)
        {
            int64_t const q_offset
                = static_cast<int64_t>(ki) * num_qheads_per_kv_head * attention_seq_len_1 * op.getHeadSize();
            int64_t const k_offset = static_cast<int64_t>(ki) * attention_seq_len_2 * op.getHeadSize();
            int64_t const qk_offset
                = static_cast<int64_t>(ki) * attention_seq_len_1 * num_qheads_per_kv_head * attention_seq_len_2;
            T* qptr = workspaceViews.qBuf + q_offset;
            T* kptr = workspaceViews.kBuf + k_offset;
            void* qkptr = is_qk_buf_float_ ? static_cast<void*>(workspaceViews.qkFloatBuf + qk_offset)
                                           : static_cast<void*>(workspaceViews.qkBuf + qk_offset);
            mCublasWrapper->stridedBatchedGemm(CUBLAS_OP_T, CUBLAS_OP_N,
                attention_seq_len_2,                                                            // n
                attention_seq_len_1 * num_qheads_per_kv_head,                                   // m
                op.getHeadSize(),                                                               // k
                qk_scale_gemm, kptr, gemm_data_type,
                op.getHeadSize(),                                                               // k
                static_cast<int64_t>(op.mNumKVHeads) * attention_seq_len_2 * op.getHeadSize(),  // n * k
                qptr, gemm_data_type,
                op.getHeadSize(),                                                               // k
                static_cast<int64_t>(attention_seq_len_1) * op.num_heads * op.getHeadSize(),    // m * k
                0.0f, qkptr, gemm_out_data_type,
                attention_seq_len_2,                                                            // n
                static_cast<int64_t>(attention_seq_len_1) * op.num_heads * attention_seq_len_2, // m * n
                p.batch_size,                                                                   // global batch size
                CUDA_R_32F);
        }
    }

    if (is_qk_buf_float_ == true)
    {
        // add relative position bias
        if (op.isRelativePosition())
        {
            // Add relative_attention_bias
            // QK is (batch_size, local_head_num, q_length, k_length), relative_attention_bias is (1,
            // local_head_num, max_output_len + 1, max_output_len + 1). broadcast along 1st dim. max_seq_len is
            // already max_output_len + 1. In implicit mode, relative_attention_bias is relative_attention_table
            // [num_heads, num_buckets], with necessary p (max_distance, num_buckets) passed at the end
            invokeAddRelativeAttentionBiasUnaligned(workspaceViews.qkFloatBuf, relative_attention_bias, p.batch_size,
                op.num_heads, attention_seq_len_1,
                cross.active() ? cross.crossKvLength() : p.cyclic_attention_window_size, stream, max_distance > 0,
                relative_attention_bias_stride, max_distance, false /* bidirectional */);
        }

        MaskedSoftmaxParam<T, float> param;
        param.attention_score = workspaceViews.qkBuf;        // (batch_size, head_num, q_length, k_length)
        param.qk = workspaceViews.qkFloatBuf;                // (batch_size, head_num, q_length, k_length)
        param.attention_mask = workspaceViews.attentionMask; // (batch_size, q_length, k_length)
        param.batch_size = p.batch_size;
        param.q_length = attention_seq_len_1;
        param.k_length = attention_seq_len_2;
        param.num_heads = op.num_heads;
        param.qk_scale = qk_scale_softmax;
        param.attn_logit_softcapping_scale = op.attn_logit_softcapping_scale;
        param.attn_logit_softcapping_inverse_scale = 1.0f / op.attn_logit_softcapping_scale;
        param.linear_bias_slopes = const_cast<T*>(linear_bias_slopes); // (head_num,), optional
        param.block_sparse_attn = op.mask_type == AttentionMaskType::BLOCKSPARSE;
        param.q_seq_lengths = p.getContextLengths();
        invokeMaskedSoftmax(param, stream);
    }
    else
    {
        // add relative position bias
        if (op.isRelativePosition())
        {
            // Add relative_attention_bias
            // QK is (batch_size, local_head_num, q_length, k_length), relative_attention_bias is (1,
            // local_head_num, max_output_len + 1, max_output_len + 1). broadcast along 1st dim. max_seq_len is
            // already max_output_len + 1. In implicit mode, relative_attention_bias is relative_attention_table
            // [num_heads, num_buckets], with necessary p (max_distance, num_buckets) passed at the end
            invokeAddRelativeAttentionBiasUnaligned(workspaceViews.qkBuf, relative_attention_bias, p.batch_size,
                op.num_heads, attention_seq_len_1,
                cross.active() ? cross.crossKvLength() : p.cyclic_attention_window_size, stream, max_distance > 0,
                relative_attention_bias_stride, max_distance, false /* bidirectional */);
        }

        MaskedSoftmaxParam<T, T> param;
        param.attention_score = workspaceViews.qkBuf;        // (batch_size, head_num, q_length, k_length)
        param.qk = workspaceViews.qkBuf;                     // (batch_size, head_num, q_length, k_length)
        param.attention_mask = workspaceViews.attentionMask; // (batch_size, q_length, k_length)
        param.batch_size = p.batch_size;
        param.q_length = attention_seq_len_1;
        param.k_length = attention_seq_len_2;
        param.num_heads = op.num_heads;
        param.qk_scale = qk_scale_softmax;
        param.attn_logit_softcapping_scale = op.attn_logit_softcapping_scale;
        param.attn_logit_softcapping_inverse_scale = 1.0f / op.attn_logit_softcapping_scale;
        param.linear_bias_slopes = const_cast<T*>(linear_bias_slopes); // (head_num,), optional
        param.block_sparse_attn = op.mask_type == AttentionMaskType::BLOCKSPARSE;
        param.q_seq_lengths = p.getContextLengths();
        invokeMaskedSoftmax(param, stream);
    }

    if (op.mNumKVHeads == 1)
    {
        // Attn_weight[b, h*s_q, s_k]
        // O[b, h*s_q, d] = Attn_weight[b, h*s_q, s_k] * V[b, s_k, d]
        // O'[b, d, h*s_q] = V'[b, d, s_k] * Attn_weight'[b, s_k, h*s_q]
        mCublasWrapper->stridedBatchedGemm(CUBLAS_OP_N, CUBLAS_OP_N,
            op.getHeadSize(),                                                               // n
            op.num_heads * attention_seq_len_1,                                             // m
            attention_seq_len_2,                                                            // k
            workspaceViews.vBuf,
            op.getHeadSize(),                                                               // n
            static_cast<int64_t>(op.getHeadSize()) * attention_seq_len_2,                   // n * k
            workspaceViews.qkBuf,
            attention_seq_len_2,                                                            // k
            static_cast<int64_t>(attention_seq_len_2) * op.num_heads * attention_seq_len_1, // m * k
            workspaceViews.qkvBuf,
            op.getHeadSize(),                                                               // n
            static_cast<int64_t>(op.getHeadSize()) * op.num_heads * attention_seq_len_1,    // n * m
            p.batch_size                                                                    // global batch size
        );
    }
    else if (op.mNumKVHeads == op.num_heads) // MHA
    {
        // O[b*h, s_q, d] = Attn_weight[b*h, s_q, s_k] * V[b*h, s_k, d]
        // O'[b*h, d, s_q] = V'[b*h, d, s_k] * Attn_weight'[b*h, s_k, s_q]
        mCublasWrapper->stridedBatchedGemm(CUBLAS_OP_N, CUBLAS_OP_N, op.getHeadSize(), attention_seq_len_1,
            attention_seq_len_2, workspaceViews.vBuf, op.getHeadSize(),
            static_cast<int64_t>(attention_seq_len_2) * op.getHeadSize(), workspaceViews.qkBuf, attention_seq_len_2,
            static_cast<int64_t>(attention_seq_len_1) * attention_seq_len_2, workspaceViews.qkvBuf, op.getHeadSize(),
            static_cast<int64_t>(attention_seq_len_1) * op.getHeadSize(), p.batch_size * op.num_heads);
    }
    else // GQA
    {
        // Attn_weight[b, h*s_q, s_k]
        // O[b, h*s_q, d] = Attn_weight[b, h*s_q, s_k] * V[b, s_k, d]
        // O'[b, d, h*s_q] = V'[b, d, s_k] * Attn_weight'[b, s_k, h*s_q]
        int const num_qheads_per_kv_head = op.num_heads / op.mNumKVHeads;
        for (int ki = 0; ki < op.mNumKVHeads; ++ki)
        {
            int64_t const qk_offset
                = static_cast<int64_t>(ki) * num_qheads_per_kv_head * attention_seq_len_1 * attention_seq_len_2;
            int64_t const v_offset = static_cast<int64_t>(ki) * attention_seq_len_2 * op.getHeadSize();
            int64_t const qkv_offset
                = static_cast<int64_t>(ki) * attention_seq_len_1 * num_qheads_per_kv_head * op.getHeadSize();
            T* qkptr = workspaceViews.qkBuf + qk_offset;
            T* vptr = workspaceViews.vBuf + v_offset;
            T* qkvptr = workspaceViews.qkvBuf + qkv_offset;
            mCublasWrapper->stridedBatchedGemm(CUBLAS_OP_N, CUBLAS_OP_N,
                op.getHeadSize(),                                                               // n
                num_qheads_per_kv_head * attention_seq_len_1,                                   // m
                attention_seq_len_2,                                                            // k
                vptr,
                op.getHeadSize(),                                                               // n
                static_cast<int64_t>(op.mNumKVHeads) * op.getHeadSize() * attention_seq_len_2,  // n * k
                qkptr,
                attention_seq_len_2,                                                            // k
                static_cast<int64_t>(attention_seq_len_2) * op.num_heads * attention_seq_len_1, // m * k
                qkvptr,
                op.getHeadSize(),                                                               // n
                static_cast<int64_t>(op.getHeadSize()) * op.num_heads * attention_seq_len_1,    // n * m
                p.batch_size                                                                    // global batch size
            );
        }
    }

    if (!op.remove_padding)
    {
        invokeTransposeQKV(static_cast<T*>(p.getOutput()), workspaceViews.qkvBuf, p.batch_size, attention_seq_len_1,
            op.num_heads, op.getHeadSize(), (float*) nullptr, 0, stream);
    }
    else
    {
        invokeTransposeAttentionOutRemovePadding(workspaceViews.qkvBuf, static_cast<T*>(p.getOutput()), p.num_tokens,
            p.batch_size, attention_seq_len_1, op.num_heads, op.getHeadSize(), workspaceViews.paddingOffset,
            (float*) nullptr, 0, stream);
    }
}

template class TrtllmGenFmha<MHA>;
template class TrtllmGenFmha<MLA>;
template class FmhaV2Fmha<MHA>;
template class FmhaV2Fmha<MLA>;
template class XqaFmha<MHA>;
template class XqaFmha<MLA>;

} // namespace torch_ext

TRTLLM_NAMESPACE_END
