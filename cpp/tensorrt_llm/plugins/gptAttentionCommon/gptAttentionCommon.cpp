/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION &
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
#include "gptAttentionCommon.h"
#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/envUtils.h"
#include "tensorrt_llm/kernels/decoderMaskedMultiheadAttention.h"
#include "tensorrt_llm/kernels/decoderMaskedMultiheadAttention/decoderXQARunner.h"
#include "tensorrt_llm/kernels/gptKernels.h"
#include "tensorrt_llm/kernels/unfusedAttentionKernels.h"
#include "tensorrt_llm/plugins/common/checkMacrosPlugin.h"
#include "tensorrt_llm/runtime/iBuffer.h"
#include <NvInferRuntimePlugin.h>
#include <algorithm>
#include <cstdint>
#include <type_traits>

using namespace nvinfer1;
using namespace tensorrt_llm::kernels;
namespace tc = tensorrt_llm::common;
using tensorrt_llm::plugins::GPTAttentionPluginCreatorCommon;
using tensorrt_llm::plugins::GPTAttentionPluginCommon;

template <typename KVCacheBuffer>
struct KVCacheBufferDataType
{
};

template <>
struct KVCacheBufferDataType<KVLinearBuffer>
{
    using Type = int8_t;
};

template <>
struct KVCacheBufferDataType<KVBlockArray>
{
    using Type = int64_t;
};

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
    const T* qkv_buf;
    const T* qkv_bias;
    const T* relative_attention_bias;
    const int* cache_indir;
    T* context_buf;
    const bool* finished;
    const int* sequence_lengths;
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
    int rotary_embedding_max_positions;
    PositionEmbeddingType position_embedding_type;
    bool position_shift_enabled;
    int max_attention_window;
    int cyclic_attention_window_size;
    int sink_token_length;
    const int* input_lengths;
    int step;
    float q_scaling;
    int relative_attention_bias_stride;
    const T* linear_bias_slopes;
    const int* ia3_tasks;
    const T* ia3_key_weights;
    const T* ia3_value_weights;
    const float* qkv_scale_out;
    const float* attention_out_scale;
    bool mUnfuseQkvGemm;
    tc::QuantMode quant_option;
    bool multi_block_mode;
    int max_seq_len_tile;
    int min_seq_len_tile;
    T* partial_out;
    float* partial_sum;
    float* partial_max;
    int* block_counter;
    const float* kv_scale_orig_quant;
    const float* kv_scale_quant_orig;
    tc::QuantMode kv_cache_quant_mode;
    int multi_processor_count;
    KVCacheBuffer kv_block_array;
    KVLinearBuffer shift_k_cache_buffer;
    bool cross_attention = false;
    const int* memory_length_per_sample = nullptr;
    int max_distance = 0;
};

template <typename T, typename KVCacheBuffer>
struct ConvertMMHAToXQAParamsHelper
{
    static constexpr Data_type data_type = DATA_TYPE_FP16;
    static constexpr bool supported = false;
};

template <>
struct ConvertMMHAToXQAParamsHelper<__half, KVLinearBuffer>
{
    static constexpr Data_type data_type = DATA_TYPE_FP16;
    static constexpr bool supported = true;
};

template <>
struct ConvertMMHAToXQAParamsHelper<__half, KVBlockArray>
{
    static constexpr Data_type data_type = DATA_TYPE_FP16;
    static constexpr bool supported = true;
};

#ifdef ENABLE_BF16
template <>
struct ConvertMMHAToXQAParamsHelper<__nv_bfloat16, KVLinearBuffer>
{
    static constexpr Data_type data_type = DATA_TYPE_BF16;
    static constexpr bool supported = true;
};

template <>
struct ConvertMMHAToXQAParamsHelper<__nv_bfloat16, KVBlockArray>
{
    static constexpr Data_type data_type = DATA_TYPE_BF16;
    static constexpr bool supported = true;
};
#endif

template <typename T, typename KVCacheBuffer>
bool GPTAttentionPluginCommon::convertMMHAParamsToXQAParams(
    tensorrt_llm::kernels::XQAParams& xqaParams, const EnqueueGenerationParams<T, KVCacheBuffer>& generationsParams)
{
    bool retval = ConvertMMHAToXQAParamsHelper<T, KVCacheBuffer>::supported;
    if (!retval)
    {
        return false;
    }
    memset(&xqaParams, 0, sizeof(XQAParams));
    xqaParams.data_type = ConvertMMHAToXQAParamsHelper<T, KVCacheBuffer>::data_type;

    xqaParams.num_q_heads = mNumHeads;
    xqaParams.num_kv_heads = mNumKVHeads;
    xqaParams.head_size = mHeadSize;
    xqaParams.unidirectional = mUnidirectional;
    xqaParams.q_scaling = mQScaling;
    xqaParams.rotary_embedding_dim = mRotaryEmbeddingDim;
    xqaParams.rotary_embedding_base = mRotaryEmbeddingBase;
    xqaParams.rotary_embedding_scale_type = mRotaryEmbeddingScaleType;
    xqaParams.rotary_embedding_scale = mRotaryEmbeddingScale;
    xqaParams.rotary_embedding_max_positions = mRotaryEmbeddingMaxPositions;
    xqaParams.position_embedding_type = mPositionEmbeddingType;
    xqaParams.position_shift_enabled = mPosShiftEnabled;
    xqaParams.remove_padding = mRemovePadding;
    xqaParams.mask_type = mMaskType;
    xqaParams.paged_kv_cache = mPagedKVCache;
    xqaParams.tokens_per_block = mTokensPerBlock;
    xqaParams.kv_cache_quant_mode = mKVCacheQuantMode;
    xqaParams.tp_size = mTpSize;
    xqaParams.tp_rank = mTpRank;
    xqaParams.qkv_bias_enabled = mQKVBiasEnabled;
    xqaParams.cross_attention = mCrossAttention;
    xqaParams.max_distance = mMaxDistance;
    xqaParams.multi_block_mode = mMultiBlockMode;
    // Medusa mode will have multiple query tokens.
    xqaParams.multi_query_tokens = mIsMedusaEnabled;

    if (mKVCacheQuantMode.hasInt8KvCache())
    {
        xqaParams.kv_cache_data_type = DATA_TYPE_INT8;
    }
    else if (mKVCacheQuantMode.hasFp8KvCache())
    {
        xqaParams.kv_cache_data_type = DATA_TYPE_E4M3;
    }
    else
    {
        xqaParams.kv_cache_data_type = xqaParams.data_type;
    }

    xqaParams.output = generationsParams.context_buf;
    xqaParams.qkv = generationsParams.attention_input;
    xqaParams.cache_indir = generationsParams.cache_indir;
    xqaParams.kv_scale_orig_quant = generationsParams.kv_scale_orig_quant;
    xqaParams.kv_scale_quant_orig = generationsParams.kv_scale_quant_orig;
    xqaParams.host_past_key_value_lengths = generationsParams.host_past_key_value_lengths;
    xqaParams.host_context_lengths = generationsParams.host_context_lengths;
    xqaParams.workspaces = generationsParams.workspace;
    xqaParams.batch_size = generationsParams.num_requests;
    xqaParams.beam_width = generationsParams.beam_width;
    // Medusa mode has generation input_length > 1.
    xqaParams.generation_input_length = generationsParams.input_seq_length;
    xqaParams.max_attention_window_size = generationsParams.max_attention_window;
    xqaParams.cyclic_attention_window_size = generationsParams.cyclic_attention_window_size;
    xqaParams.max_blocks_per_sequence = generationsParams.max_blocks_per_sequence;
    xqaParams.sink_token_length = generationsParams.sink_token_length;
    xqaParams.timestep = generationsParams.past_kv_length;
    xqaParams.qkv_bias = generationsParams.qkv_bias;
    xqaParams.sequence_lengths = generationsParams.sequence_lengths;
    xqaParams.context_lengths = generationsParams.context_lengths;
    xqaParams.alibi_slopes = generationsParams.alibi_slopes;
    // Medusa (need to take new generated ids into consideration).
    TLLM_CHECK_WITH_INFO(!mIsMedusaEnabled || generationsParams.medusa_packed_mask != nullptr,
        "Medusa mode needs a valid packed_mask input tensor.");
    xqaParams.medusa_packed_mask = generationsParams.medusa_packed_mask;
    xqaParams.medusa_position_offsets = generationsParams.medusa_position_offsets;
    return true;
}

template <typename T_MMHA, typename T, typename KVCacheBuffer, bool CROSS_ATTENTION>
void fusedQKV_masked_attention_dispatch(Multihead_attention_params<T_MMHA, CROSS_ATTENTION>& params,
    const FusedQKVMaskedAttentionDispatchParams<T, KVCacheBuffer>& input_params, cudaStream_t stream)
{
    using DataType = typename SATypeConverter<T>::Type;

    // Prepare the parameters.
    memset(&params, 0, sizeof(params));

    int hidden_units = input_params.head_num * input_params.size_per_head;
    int hidden_units_kv = input_params.kv_head_num * input_params.size_per_head;
    if (input_params.qkv_bias != nullptr)
    {
        params.q_bias = reinterpret_cast<const DataType*>(input_params.qkv_bias);
        params.k_bias = reinterpret_cast<const DataType*>(input_params.qkv_bias) + hidden_units;
        params.v_bias = reinterpret_cast<const DataType*>(input_params.qkv_bias) + hidden_units + hidden_units_kv;
    }
    else
    {
        params.q_bias = nullptr;
        params.k_bias = nullptr;
        params.v_bias = nullptr;
    }

    // Set the output buffer.
    params.out = reinterpret_cast<DataType*>(input_params.context_buf);

    // Set the input buffers.
    params.q = reinterpret_cast<const DataType*>(input_params.qkv_buf);
    params.k = reinterpret_cast<const DataType*>(input_params.qkv_buf) + hidden_units;
    params.v = reinterpret_cast<const DataType*>(input_params.qkv_buf) + hidden_units + hidden_units_kv;

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
    params.max_attention_window_size = input_params.max_attention_window;
    params.cyclic_attention_window_size = input_params.cyclic_attention_window_size;
    params.sink_token_length = input_params.sink_token_length;
    params.length_per_sample = input_params.sequence_lengths; // max_input_length + current output length
    // timestep for shared memory size calculation and rotary embedding computation
    params.timestep = input_params.step - 1;
    params.num_heads = input_params.head_num;
    params.num_kv_heads = input_params.kv_head_num;
    params.hidden_size_per_head = input_params.size_per_head;
    params.rotary_embedding_dim = input_params.rotary_embedding_dim;
    params.rotary_embedding_base = input_params.rotary_embedding_base;
    params.rotary_embedding_scale_type = input_params.rotary_embedding_scale_type;
    params.rotary_embedding_scale = input_params.rotary_embedding_scale;
    params.rotary_embedding_max_positions = input_params.rotary_embedding_max_positions;
    params.position_embedding_type = input_params.position_embedding_type;
    params.position_shift_enabled = input_params.position_shift_enabled;
    // Note: keep norm factor (sqrt(K_dim)) when adopting megatron T5 structure (may adjust)
    params.inv_sqrt_dh = 1.F / (sqrtf((float) params.hidden_size_per_head) * input_params.q_scaling);

    params.relative_attention_bias = reinterpret_cast<const DataType*>(input_params.relative_attention_bias);
    params.relative_attention_bias_stride = input_params.relative_attention_bias_stride;
    params.max_distance = input_params.max_distance;

    // The slope of linear position bias per head, e.g., ALiBi.
    if (input_params.linear_bias_slopes != nullptr)
    {
        params.linear_bias_slopes = reinterpret_cast<const DataType*>(input_params.linear_bias_slopes);
    }
    params.input_lengths = input_params.input_lengths;

    params.ia3_tasks = input_params.ia3_tasks;
    params.ia3_key_weights = reinterpret_cast<const DataType*>(input_params.ia3_key_weights);
    params.ia3_value_weights = reinterpret_cast<const DataType*>(input_params.ia3_value_weights);

    if (input_params.quant_option.hasStaticActivationScaling())
    {
        params.qkv_scale_quant_orig = input_params.qkv_scale_out;
        params.attention_out_scale_orig_quant = input_params.attention_out_scale;
    }

    params.multi_block_mode = input_params.multi_block_mode;
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
    sync_check_cuda_error();

    masked_multihead_attention(params, input_params.kv_block_array, input_params.shift_k_cache_buffer, stream);
}

#define INSTANTIATE_MMHA_DISPATCH(T_MMHA, T)                                                                           \
    template void fusedQKV_masked_attention_dispatch(Multihead_attention_params<T_MMHA, false>&,                       \
        const FusedQKVMaskedAttentionDispatchParams<T, KVLinearBuffer>&, cudaStream_t stream);                         \
    template void fusedQKV_masked_attention_dispatch(Multihead_attention_params<T_MMHA, true>&,                        \
        const FusedQKVMaskedAttentionDispatchParams<T, KVLinearBuffer>&, cudaStream_t stream);                         \
    template void fusedQKV_masked_attention_dispatch(Multihead_attention_params<T_MMHA, false>&,                       \
        const FusedQKVMaskedAttentionDispatchParams<T, KVBlockArray>&, cudaStream_t stream);                           \
    template void fusedQKV_masked_attention_dispatch(Multihead_attention_params<T_MMHA, true>&,                        \
        const FusedQKVMaskedAttentionDispatchParams<T, KVBlockArray>&, cudaStream_t stream);
INSTANTIATE_MMHA_DISPATCH(float, float)
INSTANTIATE_MMHA_DISPATCH(uint16_t, half)
#ifdef ENABLE_BF16
INSTANTIATE_MMHA_DISPATCH(__nv_bfloat16, __nv_bfloat16)
#endif
#undef INSTANTIATE_MMHA_DISPATCH

GPTAttentionPluginCommon::GPTAttentionPluginCommon(int num_heads, int num_kv_heads, int head_size, int unidirectional,
    float q_scaling, tensorrt_llm::kernels::PositionEmbeddingType position_embedding_type,
    int rotary_embedding_dim, // for RoPE. Use 0 for non-RoPE
    float rotary_embedding_base, tensorrt_llm::kernels::RotaryScalingType rotary_embedding_scale_type,
    float rotary_embedding_scale, int rotary_embedding_max_positions, int tp_size, int tp_rank, // for ALiBi
    bool unfuse_qkv_gemm,                                                                       // for AutoPP
    tensorrt_llm::kernels::ContextFMHAType context_fmha_type, bool multi_block_mode, bool enable_xqa,
    int kv_cache_quant_mode, bool remove_input_padding, tensorrt_llm::kernels::AttentionMaskType mask_type,
    bool paged_kv_cache, int tokens_per_block, nvinfer1::DataType type, int32_t max_context_length,
    bool qkv_bias_enabled, bool cross_attention, int max_distance, bool pos_shift_enabled, bool dense_context_fmha,
    bool use_paged_context_fmha, bool use_cache, bool is_medusa_enabled)
    : mNumHeads(num_heads)
    , mNumKVHeads(num_kv_heads)
    , mHeadSize(head_size)
    , mUnidirectional(unidirectional)
    , mQScaling(q_scaling)
    , mRotaryEmbeddingDim(rotary_embedding_dim)
    , mRotaryEmbeddingBase(rotary_embedding_base)
    , mRotaryEmbeddingScaleType(rotary_embedding_scale_type)
    , mRotaryEmbeddingScale(rotary_embedding_scale)
    , mRotaryEmbeddingMaxPositions(rotary_embedding_max_positions)
    , mPositionEmbeddingType(position_embedding_type)
    , mEnableContextFMHA(context_fmha_type != ContextFMHAType::DISABLED)
    , mFMHAForceFP32Acc(
          context_fmha_type == ContextFMHAType::ENABLED_WITH_FP32_ACC || type == nvinfer1::DataType::kBF16)
    , mMaskType(mask_type)
    , mType(type)
    , mMultiBlockMode(multi_block_mode)
    , mEnableXQA(enable_xqa)
    , mKVCacheQuantMode(kv_cache_quant_mode)
    , mRemovePadding(remove_input_padding)
    , mPagedKVCache(paged_kv_cache)
    , mTokensPerBlock(tokens_per_block)
    , mTpSize(tp_size)
    , mTpRank(tp_rank)
    , mUnfuseQkvGemm(unfuse_qkv_gemm)
    , mMaxContextLength(max_context_length)
    , mQKVBiasEnabled(qkv_bias_enabled)
    , mCrossAttention(cross_attention)
    , mMaxDistance(max_distance)
    , mPosShiftEnabled(pos_shift_enabled)
    , mDenseContextFMHA(dense_context_fmha)
    , mPagedContextFMHA(use_paged_context_fmha)
    , mUseKVCache(use_cache)
    , mIsMedusaEnabled(is_medusa_enabled)
{
    // pre-check whether FMHA is supported in order to save memory allocation
    mEnableContextFMHA = mEnableContextFMHA
        && (mType == nvinfer1::DataType::kHALF || mType == nvinfer1::DataType::kBF16)
        && MHARunner::fmha_supported(getHeadSize(), mSM) && !mCrossAttention
        && mPositionEmbeddingType != tensorrt_llm::kernels::PositionEmbeddingType::kRELATIVE;

    TLLM_CHECK(isRoPE() == (rotary_embedding_dim != 0));
    TLLM_CHECK_WITH_INFO((mSM >= 80) || (mType != nvinfer1::DataType::kBF16),
        "Unsupported data type, pre SM 80 GPUs do not support bfloat16");
}

const int GPTAttentionPluginCommon::getHeadSize(bool checkInit) const
{
    if (checkInit)
    {
        TLLM_CHECK_WITH_INFO(mHeadSize > 0, "Trying to read mHeadSize before it's been initialized");
    }
    return mHeadSize;
}

// Parameterized constructor
GPTAttentionPluginCommon::GPTAttentionPluginCommon(const void* data, size_t length)
{
    const char *d = reinterpret_cast<const char*>(data), *a = d;
    unsigned int kvCacheQuantMode;

    read(d, mNumHeads);
    read(d, mNumKVHeads);
    read(d, mHeadSize);
    read(d, mUnidirectional);
    read(d, mQScaling);
    read(d, mPositionEmbeddingType);
    read(d, mRotaryEmbeddingDim);
    read(d, mRotaryEmbeddingBase);
    read(d, mRotaryEmbeddingScaleType);
    read(d, mRotaryEmbeddingScale);
    read(d, mRotaryEmbeddingMaxPositions);
    read(d, mTpSize);
    read(d, mTpRank);
    read(d, mUnfuseQkvGemm);
    read(d, mEnableContextFMHA);
    read(d, mFMHAForceFP32Acc);
    read(d, mMultiBlockMode);
    read(d, mEnableXQA);
    read(d, kvCacheQuantMode);
    read(d, mRemovePadding);
    read(d, mMaskType);
    read(d, mPagedKVCache);
    read(d, mTokensPerBlock);
    read(d, mType);
    read(d, mMaxContextLength);
    read(d, mQKVBiasEnabled);
    read(d, mCrossAttention);
    read(d, mMaxDistance);
    read(d, mPosShiftEnabled);
    read(d, mDenseContextFMHA);
    read(d, mPagedContextFMHA);
    read(d, mUseKVCache);
    read(d, mIsMedusaEnabled);

    mKVCacheQuantMode = tc::QuantMode(kvCacheQuantMode);

    TLLM_CHECK(d == a + length);
    TLLM_CHECK_WITH_INFO((mSM >= 80) || (mType != nvinfer1::DataType::kBF16),
        "Unsupported data type, pre SM 80 GPUs do not support bfloat16");
}

size_t GPTAttentionPluginCommon::getWorkspaceSizeForContext(nvinfer1::DataType type, int32_t nbReq,
    int32_t input_seq_length, int32_t max_attention_window, int32_t cross_qkv_length) const noexcept
{
    const int local_hidden_units_qo = mNumHeads * getHeadSize();
    const int local_hidden_units_kv = mNumKVHeads * getHeadSize();

    auto const size = tensorrt_llm::runtime::BufferDataType(type).getSize();

    size_t context_workspace_size = 0;

    const int batch_size = nbReq;
    const size_t attention_mask_size = mEnableContextFMHA
        ? 0
        : size * batch_size * input_seq_length * (isCrossAttention() ? cross_qkv_length : input_seq_length);
    const size_t cu_seqlens_size = sizeof(int) * (batch_size + 1);
    const size_t q_buf_2_size = (mEnableContextFMHA && mPagedKVCache && mPagedContextFMHA) || !mEnableContextFMHA
        ? size * batch_size * input_seq_length * local_hidden_units_qo
        : 0;
    const size_t k_buf_2_size = mEnableContextFMHA
        ? 0
        : size * batch_size * (isCrossAttention() ? cross_qkv_length : input_seq_length) * local_hidden_units_kv;
    const size_t v_buf_2_size = mEnableContextFMHA
        ? 0
        : size * batch_size * (isCrossAttention() ? cross_qkv_length : input_seq_length) * local_hidden_units_kv;
    const size_t qk_buf_size = mEnableContextFMHA
        ? 0
        : size * batch_size * mNumHeads * input_seq_length * (isCrossAttention() ? cross_qkv_length : input_seq_length);
    const size_t qkv_buf_2_size = mEnableContextFMHA ? 0 : size * batch_size * input_seq_length * local_hidden_units_qo;
    const size_t qk_buf_float_size = mEnableContextFMHA ? 0
                                                        : sizeof(float) * batch_size * mNumHeads * input_seq_length
            * (isCrossAttention() ? cross_qkv_length : input_seq_length);
    const size_t padding_offset_size = sizeof(int) * batch_size * input_seq_length;
    // It is assumed that the number of tokens per paged kv block should be >= 128.
    const size_t paged_kv_tma_desc_size = mPagedKVCache && mPagedContextFMHA
        ? batch_size * 2 * TMA_DESC_SIZE_IN_BYTE * tc::divUp(max_attention_window, mTokensPerBlock)
        : 0;

    const int NUM_BUFFERS = 12;
    size_t workspaces[NUM_BUFFERS];
    workspaces[0] = CUBLAS_WORKSPACE_SIZE;
    workspaces[1] = attention_mask_size;
    workspaces[2] = cu_seqlens_size; // cu_seqlen_q
    workspaces[3] = cu_seqlens_size; // cu_seqlen_kv
    workspaces[4] = q_buf_2_size;
    workspaces[5] = k_buf_2_size;
    workspaces[6] = v_buf_2_size;
    workspaces[7] = qk_buf_size;
    workspaces[8] = qkv_buf_2_size;
    workspaces[9] = qk_buf_float_size;
    workspaces[10] = padding_offset_size;
    workspaces[11] = paged_kv_tma_desc_size;
    context_workspace_size = tc::calculateTotalWorkspaceSize(workspaces, NUM_BUFFERS);
    return context_workspace_size;
}

size_t GPTAttentionPluginCommon::getWorkspaceSizeForGeneration(
    nvinfer1::DataType type, int32_t total_num_seq, int32_t max_attention_window) const noexcept
{
    const int local_hidden_units_qo = mNumHeads * getHeadSize();
    const int local_hidden_units_kv = mNumKVHeads * getHeadSize();

    auto const size = tensorrt_llm::runtime::BufferDataType(type).getSize();

    size_t context_workspace_size = 0;
    size_t generation_workspace_size = 0;

    const int batch_beam = total_num_seq;
    int32_t const maxSeqLenTile
        = std::max(getMaxNumSeqLenTile(batch_beam), (int) tc::divUp(mMultiProcessorCount, mNumHeads));

    const size_t partial_out_size = size * batch_beam * mNumHeads * mHeadSize * maxSeqLenTile;
    const size_t partial_sum_size = sizeof(float) * batch_beam * mNumHeads * maxSeqLenTile;
    const size_t partial_max_size = sizeof(float) * batch_beam * mNumHeads * maxSeqLenTile;
    const size_t block_counter_size = sizeof(int) * batch_beam * mNumHeads;
    const size_t shift_k_cache_size = (!mPosShiftEnabled || isCrossAttention())
        ? 0
        : size * batch_beam * mNumHeads * mHeadSize * max_attention_window;

    const int NUM_BUFFERS = 5;
    size_t workspaces[NUM_BUFFERS];
    workspaces[0] = partial_out_size;
    workspaces[1] = partial_sum_size;
    workspaces[2] = partial_max_size;
    workspaces[3] = block_counter_size;
    workspaces[4] = shift_k_cache_size;
    generation_workspace_size = tc::calculateTotalWorkspaceSize(workspaces, NUM_BUFFERS);

    size_t mqa_workspace_size = 0;
    if (mDecoderXQARunner.get())
    {
        size_t mqa_workspaces[1];
        mqa_workspaces[0] = mDecoderXQARunner->getWorkspaceSize(batch_beam);
        mqa_workspace_size = tc::calculateTotalWorkspaceSize(mqa_workspaces, 1);
    }

    return std::max(generation_workspace_size, mqa_workspace_size);
}

int GPTAttentionPluginCommon::getMaxNumSeqLenTile(int batch_beam_size) const
{
    if (mMultiBlockMode)
    {
        // And we allocate the buffer based on the maximum number of blocks per sequence (batch_beam_size = 1).
        // Assume we can only have 1 block (large block size like 1024) in SM, and we only want one wave of blocks.
        return tc::getEnvMmhaMultiblockDebug() ? std::max(kReservedMaxSeqLenTilePerSeq, getEnvMmhaBlocksPerSequence())
                                               : tc::divUp(mMultiProcessorCount, batch_beam_size * mNumHeads);
    }
    return 0;
}

template <typename T, typename KVCacheBuffer>
int GPTAttentionPluginCommon::enqueueContext(const EnqueueContextParams<T, KVCacheBuffer>& params, cudaStream_t stream)
{
    const int num_heads = mNumHeads;
    const int num_kv_heads = mNumKVHeads;
    const int head_size = getHeadSize();
    const int local_hidden_units_qo = num_heads * head_size;
    const int local_hidden_units_kv = num_kv_heads * head_size;
    const PositionEmbeddingType position_embedding_type = mPositionEmbeddingType;
    const float q_scaling = mQScaling;
    const bool* finished = nullptr;
    const bool has_ia3 = false;

    KVCacheBuffer kv_cache_buffer;
    const auto elem_size = mKVCacheQuantMode.hasKvCacheQuant() ? sizeof(int8_t) : sizeof(T);
    int64_t* host_kv_cache_block_ptrs = nullptr;
    if (mPagedKVCache)
    {
        using BufferDataType = typename KVCacheBufferDataType<KVCacheBuffer>::Type;
        kv_cache_buffer = KVCacheBuffer(params.batch_size, params.max_blocks_per_sequence, mTokensPerBlock,
            num_kv_heads * head_size * elem_size, params.cyclic_attention_window_size, params.sink_token_length, false);
        kv_cache_buffer.data = reinterpret_cast<BufferDataType*>(params.block_pointers);
        host_kv_cache_block_ptrs = reinterpret_cast<int64_t*>(params.host_block_pointers);
    }
    else
    {
        using BufferDataType = typename KVCacheBufferDataType<KVCacheBuffer>::Type;
        kv_cache_buffer = KVCacheBuffer(params.batch_size, 1,
            isCrossAttention() ? params.cross_qkv_length : params.max_attention_window,
            num_kv_heads * head_size * elem_size, params.cyclic_attention_window_size, params.sink_token_length, false);
        kv_cache_buffer.data = reinterpret_cast<BufferDataType*>(params.key_value_cache);
    }

    const auto quant_option = tc::QuantMode::fromDescription();
    const float* qkv_scale_out = nullptr;
    const float* attention_out_scale = nullptr;

    const int* ia3_tasks = nullptr;
    const T* ia3_key_weights = nullptr;
    const T* ia3_value_weights = nullptr;

    const bool multi_block_mode = false;
    const int max_seq_len_tile = 0;
    T* partial_out = nullptr;
    float* partial_sum = nullptr;
    float* partial_max = nullptr;
    int* block_counter = nullptr;

    auto cublasHandle = mCublasWrapper->getCublasHandle();
    TLLM_CUDA_CHECK(cublasSetStream(cublasHandle, stream));
    mCublasWrapper->setStream(stream);
    mCublasWrapper->setWorkspace(params.workspace);
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

    const size_t attention_mask_size = mEnableContextFMHA ? 0
                                                          : sizeof(T) * params.batch_size * params.input_seq_length
            * (isCrossAttention() ? params.cross_qkv_length : params.input_seq_length);
    const size_t cu_seqlens_size = sizeof(int) * (params.batch_size + 1);
    const size_t q_buf_2_size = (mEnableContextFMHA && mPagedKVCache && mPagedContextFMHA) || !mEnableContextFMHA
        ? sizeof(T) * params.batch_size * params.input_seq_length * local_hidden_units_qo
        : 0;
    const size_t k_buf_2_size = mEnableContextFMHA ? 0
                                                   : sizeof(T) * params.batch_size
            * (isCrossAttention() ? params.cross_qkv_length : params.input_seq_length) * local_hidden_units_kv;
    const size_t v_buf_2_size = mEnableContextFMHA ? 0
                                                   : sizeof(T) * params.batch_size
            * (isCrossAttention() ? params.cross_qkv_length : params.input_seq_length) * local_hidden_units_kv;
    const size_t qk_buf_size = mEnableContextFMHA ? 0
                                                  : sizeof(T) * params.batch_size * mNumHeads * params.input_seq_length
            * (isCrossAttention() ? params.cross_qkv_length : params.input_seq_length);
    const size_t qkv_buf_2_size
        = mEnableContextFMHA ? 0 : sizeof(T) * params.batch_size * params.input_seq_length * local_hidden_units_qo;
    const size_t qk_buf_float_size = mEnableContextFMHA ? 0
                                                        : sizeof(float) * params.batch_size * mNumHeads
            * params.input_seq_length * (isCrossAttention() ? params.cross_qkv_length : params.input_seq_length);
    const size_t padding_offset_size
        = sizeof(int) * params.batch_size * (isCrossAttention() ? params.cross_qkv_length : params.input_seq_length);
    // It is assumed that the number of tokens per paged kv block should be >= 128.
    const size_t blocks_per_context_sequence = mPagedKVCache ? tc::divUp(params.input_seq_length, mTokensPerBlock) : 0;
    const size_t paged_kv_tma_desc_size = mPagedKVCache && mPagedContextFMHA
        ? params.batch_size * 2 * TMA_DESC_SIZE_IN_BYTE * blocks_per_context_sequence
        : 0;

    const bool is_qk_buf_float_ = true;

    // Workspace pointer shift
    int8_t* workspace_byte_ptr = reinterpret_cast<int8_t*>(params.workspace);
    size_t offset = CUBLAS_WORKSPACE_SIZE;

    T* attention_mask = reinterpret_cast<T*>(nextWorkspacePtr(workspace_byte_ptr, offset, attention_mask_size));
    int* cu_q_seqlens = reinterpret_cast<int*>(nextWorkspacePtr(workspace_byte_ptr, offset, cu_seqlens_size));
    int* cu_kv_seqlens = reinterpret_cast<int*>(nextWorkspacePtr(workspace_byte_ptr, offset, cu_seqlens_size));
    T* q_buf_2_ = reinterpret_cast<T*>(nextWorkspacePtr(workspace_byte_ptr, offset, q_buf_2_size));
    T* k_buf_2_ = reinterpret_cast<T*>(nextWorkspacePtr(workspace_byte_ptr, offset, k_buf_2_size));
    T* v_buf_2_ = reinterpret_cast<T*>(nextWorkspacePtr(workspace_byte_ptr, offset, v_buf_2_size));
    T* qk_buf_ = reinterpret_cast<T*>(nextWorkspacePtr(workspace_byte_ptr, offset, qk_buf_size));
    T* qkv_buf_2_ = reinterpret_cast<T*>(nextWorkspacePtr(workspace_byte_ptr, offset, qkv_buf_2_size));
    float* qk_buf_float_ = reinterpret_cast<float*>(nextWorkspacePtr(workspace_byte_ptr, offset, qk_buf_float_size));
    int* padding_offset = reinterpret_cast<int*>(nextWorkspacePtr(workspace_byte_ptr, offset, padding_offset_size));
    void* paged_kv_tma_desc
        = reinterpret_cast<int*>(nextWorkspacePtr(workspace_byte_ptr, offset, paged_kv_tma_desc_size));

    // build attention_mask, cu_seqlens, and padding_offset tensors
    // Note: self attn and cross attn should use different params
    // cross attn's seqlen info is from encoder input lengths, not decoder input lengths!
    // moreover, attn mask for cross attn should be set separately (see below)
    BuildDecoderInfoParams<T> decoder_params;
    memset(&decoder_params, 0, sizeof(decoder_params));
    decoder_params.seqQOffsets = cu_q_seqlens;
    decoder_params.seqKVOffsets = cu_kv_seqlens;
    decoder_params.paddingOffsets = padding_offset;
    decoder_params.attentionMask = isCrossAttention() ? nullptr : attention_mask; // manually set for cross attn
    decoder_params.seqQLengths = isCrossAttention() ? params.encoder_input_lengths : params.q_seq_lengths;
    decoder_params.seqKVLengths = isCrossAttention() ? params.encoder_input_lengths : params.kv_seq_lengths;
    decoder_params.batchSize = params.batch_size;
    decoder_params.maxSeqLength = isCrossAttention() ? params.cross_qkv_length : params.input_seq_length;
    decoder_params.attentionWindowSize = params.cyclic_attention_window_size;
    decoder_params.sinkTokenLength = params.sink_token_length;
    decoder_params.numTokens = params.num_tokens;
    decoder_params.attentionMaskType = mMaskType;
    invokeBuildDecoderInfo(decoder_params, stream);
    sync_check_cuda_error();

    // In cross attention context phase, the attention mask should be a matrix of all ones.
    // We reassign attention_mask to override what previous invokeBuildDecoderInfo() does
    // also, invokeBuildDecoderInfo can only handle square mask, not cross B x q_len x kv_len mask
    // TODO: put this logic in the kernel above. currently not much concern because q_len is mostly = 1
    if (isCrossAttention())
    {
        std::vector<T> h_attention_mask(params.batch_size * params.input_seq_length * params.cross_qkv_length, 1.);
        std::vector<int32_t> h_encoder_input_lengths(params.batch_size);
        cudaMemcpyAsync(h_encoder_input_lengths.data(), params.encoder_input_lengths,
            sizeof(int32_t) * params.batch_size, cudaMemcpyDeviceToHost, stream);
        for (int bi = 0; bi < params.batch_size; bi++)
        {
            int b_offset = bi * params.input_seq_length * params.cross_qkv_length;
            for (int qi = 0; qi < params.input_seq_length; qi++)
            {
                int q_offset = b_offset + qi * params.cross_qkv_length;
                if (h_encoder_input_lengths[bi] < params.cross_qkv_length)
                {
                    std::fill(h_attention_mask.begin() + q_offset + h_encoder_input_lengths[bi],
                        h_attention_mask.begin() + q_offset + params.cross_qkv_length, 0.f);
                }
            }
        }
        cudaMemcpyAsync(attention_mask, h_attention_mask.data(),
            sizeof(T) * params.batch_size * params.cross_qkv_length * params.input_seq_length, cudaMemcpyHostToDevice,
            stream);
    }

    // write KV to cache
    const KvCacheDataType cache_type = mKVCacheQuantMode.hasInt8KvCache()
        ? KvCacheDataType::INT8
        : (mKVCacheQuantMode.hasFp8KvCache() ? KvCacheDataType::FP8 : KvCacheDataType::BASE);

    const cudaDataType_t gemm_data_type = tc::CudaDataType<T>::value;
    const int attention_seq_len_1 = params.input_seq_length;                                                // q length
    const int attention_seq_len_2 = isCrossAttention() ? params.cross_qkv_length : params.input_seq_length; // kv length

    // If the model has relative attentiona bias, q scaling should be applied in QK gemm stage and use 1 in
    // softamax stage (because to get softmax[scale(Q*K) + rel pos bias] here, q_scaling can't be applied during
    // softmax phase by qk_scale); otherwise, use 1 in gemm stage and apply scaling in softmax stage
    const float qk_scale
        = 1.0f / (sqrtf(getHeadSize() * 1.0f) * q_scaling); // q_scaling in denominator. by default q_scaling =1.0f
    const float qk_scale_gemm = isRelativePosition() ? qk_scale : 1.0f;
    const T qk_scale_softmax = static_cast<T>(isRelativePosition() ? 1.0f : qk_scale);

    // in context phase, currently FMHA runner has two restrictions:
    // 1. only apply to self attention. If want fused multi-head cross attention, FMHCA kernels and runner is needed
    // 2. doesn't apply to MHA with relative attention bias, i.e. softmax(QK + bias) * V
    // We update mEnableContextFMHA in constructor to check these conditions
    if (mEnableContextFMHA)
    {
        const bool enablePagedKVContextFMHA = mPagedKVCache && mPagedContextFMHA;
        invokeApplyBiasRopeUpdateKVCache(const_cast<T*>(params.attention_input), q_buf_2_, kv_cache_buffer,
            const_cast<T*>(params.qkv_bias), params.q_seq_lengths, params.kv_seq_lengths,
            mRemovePadding ? padding_offset : nullptr, params.batch_size, params.input_seq_length,
            params.cyclic_attention_window_size, params.sink_token_length, params.num_tokens, mNumHeads, mNumKVHeads,
            getHeadSize(), mRotaryEmbeddingDim, mRotaryEmbeddingBase, mRotaryEmbeddingScaleType, mRotaryEmbeddingScale,
            mRotaryEmbeddingMaxPositions, position_embedding_type, (int*) nullptr, mPosShiftEnabled, (float*) nullptr,
            0, cache_type, params.kv_scale_orig_quant, enablePagedKVContextFMHA, 1, mLaunchGridBlockCache, stream);
        sync_check_cuda_error();

        //  It is not needed with packed QKV input.
        if (enablePagedKVContextFMHA)
        {
            // to enable chunked attention,
            // 1. make sure you call setup_paged_kv(batch_size, max_query_length, max_kv_length, ....)
            // 2. make sure you call run_paged_kv(q_ptr, kv_tma_desc_device_ptr, kv_cache_block_ptrs_on_host,
            //                                    kv_cache_buffer, cu_q_seqlens, cu_kv_seqlens, ...)
            //    - q_ptr: [B, S, H, D], which supports variable sequence length
            //    - kv_tma_desc_device_ptr: allocated on device based on the number of context kv blocks.
            //    - kv_cache_block_ptrs_on_host: tma descriptors need the paged kv cache device ptrs to be in host.
            //    - kv_cache_buffer: paged kv buffer
            //    - cu_q_seqlens: the cumulative query sequence lengths, needed for variable sequence length.
            //    - cu_kv_seqlens: the cumulative kv sequence lengths, needed for variable sequence length.

            // the token will pay attention to previous tokens while starting from max(0, rowIdx -
            // cyclic_attention_window_size);
            if (params.sink_token_length > 0)
            {
                TLLM_LOG_ERROR("Cannot support StreamingLLM now when enabling paged KV context FMHA.");
            }
            mFMHARunner->setup_paged_kv(params.batch_size, params.input_seq_length, params.max_past_kv_len,
                blocks_per_context_sequence, mTokensPerBlock, params.cyclic_attention_window_size, params.num_tokens,
                isALiBi(), isAliBiWithScale(), mTpSize, mTpRank);
            mFMHARunner->run_paged_kv(q_buf_2_, paged_kv_tma_desc, host_kv_cache_block_ptrs,
                reinterpret_cast<KVBlockArray&>(kv_cache_buffer), cu_q_seqlens, cu_kv_seqlens, params.context_buf,
                stream);
        }
        else
        {
            // the token will pay attention to previous tokens while starting from max(0, rowIdx -
            // cyclic_attention_window_size);
            const int attention_window_size
                = mDenseContextFMHA ? params.num_tokens : params.cyclic_attention_window_size;
            mFMHARunner->setup(params.batch_size, params.input_seq_length, attention_window_size, params.num_tokens,
                isALiBi(), isAliBiWithScale(), mTpSize, mTpRank);
            mFMHARunner->run(const_cast<T*>(params.attention_input), cu_q_seqlens, params.context_buf, stream);
        }
        sync_check_cuda_error();
    }
    else
    {
        // FIXME: a temporary solution to make sure the padding part of key/value buffer is 0
        // NOTE: pointer subtraction is used below since there could be some extra gap due to alignment.
        //  Otherwise, we could do cudaMemsetAsync(k_buf_2_, 0, k_buf_2_size + v_buf_2_size, stream);
        // cudaMemsetAsync(k_buf_2_, 0, reinterpret_cast<int8_t*>(qk_buf_) - reinterpret_cast<int8_t*>(k_buf_2_),
        // stream);
        cudaMemsetAsync(k_buf_2_, 0,
            reinterpret_cast<int8_t*>(v_buf_2_) - reinterpret_cast<int8_t*>(k_buf_2_) + v_buf_2_size, stream);

        if (!isCrossAttention())
        {
            // self attention, write to Q/K/V
            invokeAddFusedQKVBiasTranspose(q_buf_2_, k_buf_2_, v_buf_2_, const_cast<T*>(params.attention_input),
                const_cast<T*>(params.qkv_bias), params.q_seq_lengths, mRemovePadding ? padding_offset : nullptr,
                params.batch_size, params.input_seq_length, params.num_tokens, mNumHeads, mNumKVHeads, getHeadSize(),
                mRotaryEmbeddingDim, mRotaryEmbeddingBase, mRotaryEmbeddingScaleType, mRotaryEmbeddingScale,
                mRotaryEmbeddingMaxPositions, position_embedding_type, (float*) nullptr, 0, stream);
        }
        else
        {
            // cross attention, write Q from self QKV, write KV from cross QKV
            // kernel modified accordingly to handle nullptr buffer
            invokeAddFusedQKVBiasTranspose(q_buf_2_, (T*) nullptr, (T*) nullptr, const_cast<T*>(params.attention_input),
                const_cast<T*>(params.qkv_bias), params.q_seq_lengths, mRemovePadding ? padding_offset : nullptr,
                params.batch_size, params.input_seq_length, params.num_tokens, mNumHeads, mNumKVHeads, getHeadSize(),
                mRotaryEmbeddingDim, mRotaryEmbeddingBase, mRotaryEmbeddingScaleType, mRotaryEmbeddingScale,
                mRotaryEmbeddingMaxPositions, position_embedding_type, (float*) nullptr, 0, stream);
            invokeAddFusedQKVBiasTranspose((T*) nullptr, k_buf_2_, v_buf_2_, const_cast<T*>(params.cross_qkv),
                const_cast<T*>(params.qkv_bias), params.encoder_input_lengths,
                mRemovePadding ? padding_offset : nullptr, params.batch_size, params.cross_qkv_length,
                params.num_encoder_tokens, mNumHeads, mNumKVHeads, getHeadSize(), mRotaryEmbeddingDim,
                mRotaryEmbeddingBase, mRotaryEmbeddingScaleType, mRotaryEmbeddingScale, mRotaryEmbeddingMaxPositions,
                position_embedding_type, (float*) nullptr, 0, stream);
        }
        sync_check_cuda_error();

        // write KV to cache
        if (useKVCache())
        {
            invokeTranspose4dBatchMajor(k_buf_2_, v_buf_2_, kv_cache_buffer, params.batch_size,
                isCrossAttention() ? params.cross_qkv_length : params.input_seq_length,
                isCrossAttention() ? params.cross_qkv_length : params.cyclic_attention_window_size, getHeadSize(),
                mNumKVHeads, cache_type, params.kv_scale_orig_quant,
                isCrossAttention() ? params.encoder_input_lengths : params.q_seq_lengths, stream);
        }
        sync_check_cuda_error();

        const T* linear_bias_slopes = isALiBi() ? params.alibi_slopes : nullptr;
        const T* relative_attention_bias = isRelativePosition() ? params.relative_attention_bias : nullptr;
        const int relative_attention_bias_stride = isRelativePosition() ? params.relative_attention_bias_stride : 0;
        const int max_distance = mMaxDistance;
        cudaDataType_t gemm_out_data_type = is_qk_buf_float_ ? CUDA_R_32F : gemm_data_type;
        void* gemm_out_buf_ = is_qk_buf_float_ ? static_cast<void*>(qk_buf_float_) : static_cast<void*>(qk_buf_);
        if (mNumKVHeads == 1) // MQA
        {
            // Attn_weight[b, h*s_q, s_k] = Q[b, h*s_q, d] * K'[b, d, s_k]
            // Attn_weight'[b, s_k, h*s_q] = K[b, s_k, d] * Q'[b, d, h*s_q]
            mCublasWrapper->stridedBatchedGemm(CUBLAS_OP_T, CUBLAS_OP_N,
                attention_seq_len_2,                                   // n
                attention_seq_len_1 * mNumHeads,                       // m
                getHeadSize(),                                         // k
                qk_scale_gemm, k_buf_2_, gemm_data_type,
                getHeadSize(),                                         // k
                attention_seq_len_2 * getHeadSize(),                   // n * k
                q_buf_2_, gemm_data_type,
                getHeadSize(),                                         // k
                attention_seq_len_1 * mNumHeads * getHeadSize(),       // m * k
                0.0f, gemm_out_buf_, gemm_out_data_type,
                attention_seq_len_2,                                   // n
                attention_seq_len_1 * mNumHeads * attention_seq_len_2, // m * n
                params.batch_size,                                     // global batch size
                CUDA_R_32F);
        }
        else if (mNumKVHeads == mNumHeads) // MHA
        {
            // Attn_weight[b*h, s_q, s_k] = Q[b*h, s_q, d] * K'[b*h, d, s_k]
            // Attn_weight'[b*h, s_k, s_q] = K[b*h, s_k, d] * Q'[b*h, d, s_q]
            mCublasWrapper->stridedBatchedGemm(CUBLAS_OP_T, CUBLAS_OP_N,
                attention_seq_len_2,                 // n
                attention_seq_len_1,                 // m
                getHeadSize(),                       // k
                qk_scale_gemm, k_buf_2_, gemm_data_type,
                getHeadSize(),                       // k
                attention_seq_len_2 * getHeadSize(), // n * k
                q_buf_2_, gemm_data_type,
                getHeadSize(),                       // k
                attention_seq_len_1 * getHeadSize(), // m * k
                0.0f, gemm_out_buf_, gemm_out_data_type,
                attention_seq_len_2,                 // n
                attention_seq_len_2 * attention_seq_len_1,
                params.batch_size * mNumHeads,       // global batch size
                CUDA_R_32F);
        }
        else // GQA
        {
            // Some number of contiguous Q heads will share the same K/V head
            // Since the KV stride is NOT fixed for all Q, we have 2 options:
            //  1. Loop over stridedBatchedGemm for each KV head. (multiple API calls/cuda kernels)
            //  2. Calculate the pointers and use batchedGemm() (extra device memory) ::TODO::
            const int num_qheads_per_kv_head = mNumHeads / mNumKVHeads;
            for (int ki = 0; ki < mNumKVHeads; ++ki)
            {
                T* qptr = q_buf_2_ + (ki * num_qheads_per_kv_head * attention_seq_len_1 * getHeadSize());
                T* kptr = k_buf_2_ + (ki * attention_seq_len_2 * getHeadSize());
                const int qk_offset = ki * attention_seq_len_1 * num_qheads_per_kv_head * attention_seq_len_2;
                void* qkptr = is_qk_buf_float_ ? static_cast<void*>(qk_buf_float_ + qk_offset)
                                               : static_cast<void*>(qk_buf_ + qk_offset);
                mCublasWrapper->stridedBatchedGemm(CUBLAS_OP_T, CUBLAS_OP_N,
                    attention_seq_len_2,                                   // n
                    attention_seq_len_1 * num_qheads_per_kv_head,          // m
                    getHeadSize(),                                         // k
                    qk_scale_gemm, kptr, gemm_data_type,
                    getHeadSize(),                                         // k
                    mNumKVHeads * attention_seq_len_2 * getHeadSize(),     // n * k
                    qptr, gemm_data_type,
                    getHeadSize(),                                         // k
                    attention_seq_len_1 * mNumHeads * getHeadSize(),       // m * k
                    0.0f, qkptr, gemm_out_data_type,
                    attention_seq_len_2,                                   // n
                    attention_seq_len_1 * mNumHeads * attention_seq_len_2, // m * n
                    params.batch_size,                                     // global batch size
                    CUDA_R_32F);
            }
        }

        if (is_qk_buf_float_ == true)
        {
            // add relative position bias
            if (isRelativePosition())
            {
                // Add relative_attention_bias
                // QK is (batch_size, local_head_num, q_length, k_length), relative_attention_bias is (1,
                // local_head_num, max_output_len + 1, max_output_len + 1). broadcast along 1st dim. max_seq_len is
                // already max_output_len + 1. In implicit mode, relative_attention_bias is relative_attention_table
                // [num_heads, num_buckets], with necessary params (max_distance, num_buckets) passed at the end
                invokeAddRelativeAttentionBiasUnaligned(qk_buf_float_, relative_attention_bias, params.batch_size,
                    mNumHeads, attention_seq_len_1,
                    isCrossAttention() ? params.cross_qkv_length : params.cyclic_attention_window_size, stream,
                    max_distance > 0, relative_attention_bias_stride, max_distance, true /* bidirectional */);
            }

            MaskedSoftmaxParam<T, float> param;
            param.attention_score = qk_buf_;       // (batch_size, head_num, q_length, k_length)
            param.qk = qk_buf_float_;              // (batch_size, head_num, q_length, k_length)
            param.attention_mask = attention_mask; // (batch_size, q_length, k_length)
            param.batch_size = params.batch_size;
            param.q_length = attention_seq_len_1;
            param.k_length = attention_seq_len_2;
            param.num_heads = mNumHeads;
            param.qk_scale = qk_scale_softmax;
            param.linear_bias_slopes = const_cast<T*>(linear_bias_slopes); // (head_num,), optional
            invokeMaskedSoftmax(param, stream);
        }
        else
        {
            // add relative position bias
            if (isRelativePosition())
            {
                // Add relative_attention_bias
                // QK is (batch_size, local_head_num, q_length, k_length), relative_attention_bias is (1,
                // local_head_num, max_output_len + 1, max_output_len + 1). broadcast along 1st dim. max_seq_len is
                // already max_output_len + 1. In implicit mode, relative_attention_bias is relative_attention_table
                // [num_heads, num_buckets], with necessary params (max_distance, num_buckets) passed at the end
                invokeAddRelativeAttentionBiasUnaligned(qk_buf_, relative_attention_bias, params.batch_size, mNumHeads,
                    attention_seq_len_1,
                    isCrossAttention() ? params.cross_qkv_length : params.cyclic_attention_window_size, stream,
                    max_distance > 0, relative_attention_bias_stride, max_distance, true /* bidirectional */);
            }

            MaskedSoftmaxParam<T, T> param;
            param.attention_score = qk_buf_;       // (batch_size, head_num, q_length, k_length)
            param.qk = qk_buf_;                    // (batch_size, head_num, q_length, k_length)
            param.attention_mask = attention_mask; // (batch_size, q_length, k_length)
            param.batch_size = params.batch_size;
            param.q_length = attention_seq_len_1;
            param.k_length = attention_seq_len_2;
            param.num_heads = mNumHeads;
            param.qk_scale = qk_scale_softmax;
            param.linear_bias_slopes = const_cast<T*>(linear_bias_slopes); // (head_num,), optional
            invokeMaskedSoftmax(param, stream);
        }

        if (mNumKVHeads == 1)
        {
            // Attn_weight[b, h*s_q, s_k]
            // O[b, h*s_q, d] = Attn_weight[b, h*s_q, s_k] * V[b, s_k, d]
            // O'[b, d, h*s_q] = V'[b, d, s_k] * Attn_weight'[b, s_k, h*s_q]
            mCublasWrapper->stridedBatchedGemm(CUBLAS_OP_N, CUBLAS_OP_N,
                getHeadSize(),                                         // n
                mNumHeads * attention_seq_len_1,                       // m
                attention_seq_len_2,                                   // k
                v_buf_2_,
                getHeadSize(),                                         // n
                getHeadSize() * attention_seq_len_2,                   // n * k
                qk_buf_,
                attention_seq_len_2,                                   // k
                attention_seq_len_2 * mNumHeads * attention_seq_len_1, // m * k
                qkv_buf_2_,
                getHeadSize(),                                         // n
                getHeadSize() * mNumHeads * attention_seq_len_1,       // n * m
                params.batch_size                                      // global batch size
            );
        }
        else if (mNumKVHeads == mNumHeads) // MHA
        {
            // O[b*h, s_q, d] = Attn_weight[b*h, s_q, s_k] * V[b*h, s_k, d]
            // O'[b*h, d, s_q] = V'[b*h, d, s_k] * Attn_weight'[b*h, s_k, s_q]
            mCublasWrapper->stridedBatchedGemm(CUBLAS_OP_N, CUBLAS_OP_N, getHeadSize(), attention_seq_len_1,
                attention_seq_len_2, v_buf_2_, getHeadSize(), attention_seq_len_2 * getHeadSize(), qk_buf_,
                attention_seq_len_2, attention_seq_len_1 * attention_seq_len_2, qkv_buf_2_, getHeadSize(),
                attention_seq_len_1 * getHeadSize(), params.batch_size * mNumHeads);
        }
        else // GQA
        {
            // Attn_weight[b, h*s_q, s_k]
            // O[b, h*s_q, d] = Attn_weight[b, h*s_q, s_k] * V[b, s_k, d]
            // O'[b, d, h*s_q] = V'[b, d, s_k] * Attn_weight'[b, s_k, h*s_q]
            const int num_qheads_per_kv_head = mNumHeads / mNumKVHeads;
            for (int ki = 0; ki < mNumKVHeads; ++ki)
            {
                T* qkptr = qk_buf_ + (ki * num_qheads_per_kv_head * attention_seq_len_1 * attention_seq_len_2);
                T* vptr = v_buf_2_ + (ki * attention_seq_len_2 * getHeadSize());
                T* qkvptr = qkv_buf_2_ + (ki * attention_seq_len_1 * num_qheads_per_kv_head * getHeadSize());
                mCublasWrapper->stridedBatchedGemm(CUBLAS_OP_N, CUBLAS_OP_N,
                    getHeadSize(),                                         // n
                    num_qheads_per_kv_head * attention_seq_len_1,          // m
                    attention_seq_len_2,                                   // k
                    vptr,
                    getHeadSize(),                                         // n
                    mNumKVHeads * getHeadSize() * attention_seq_len_2,     // n * k
                    qkptr,
                    attention_seq_len_2,                                   // k
                    attention_seq_len_2 * mNumHeads * attention_seq_len_1, // m * k
                    qkvptr,
                    getHeadSize(),                                         // n
                    getHeadSize() * mNumHeads * attention_seq_len_1,       // n * m
                    params.batch_size                                      // global batch size
                );
            }
        }

        if (!mRemovePadding)
        {
            invokeTransposeQKV(params.context_buf, qkv_buf_2_, params.batch_size, attention_seq_len_1, mNumHeads,
                getHeadSize(), (float*) nullptr, 0, stream);
        }
        else
        {
            invokeTransposeAttentionOutRemovePadding(qkv_buf_2_, params.context_buf, params.num_tokens,
                params.batch_size, attention_seq_len_1, mNumHeads, getHeadSize(), padding_offset, (float*) nullptr, 0,
                stream);
        }
    }

    return 0;
}

template int GPTAttentionPluginCommon::enqueueContext<half, KVLinearBuffer>(
    const EnqueueContextParams<half, KVLinearBuffer>& params, cudaStream_t stream);

template int GPTAttentionPluginCommon::enqueueContext<float, KVLinearBuffer>(
    const EnqueueContextParams<float, KVLinearBuffer>& params, cudaStream_t stream);

#ifdef ENABLE_BF16
template int GPTAttentionPluginCommon::enqueueContext<__nv_bfloat16, KVLinearBuffer>(
    const EnqueueContextParams<__nv_bfloat16, KVLinearBuffer>& params, cudaStream_t stream);
#endif

template int GPTAttentionPluginCommon::enqueueContext<half, KVBlockArray>(
    const EnqueueContextParams<half, KVBlockArray>& params, cudaStream_t stream);

template int GPTAttentionPluginCommon::enqueueContext<float, KVBlockArray>(
    const EnqueueContextParams<float, KVBlockArray>& params, cudaStream_t stream);

#ifdef ENABLE_BF16
template int GPTAttentionPluginCommon::enqueueContext<__nv_bfloat16, KVBlockArray>(
    const EnqueueContextParams<__nv_bfloat16, KVBlockArray>& params, cudaStream_t stream);
#endif

bool GPTAttentionPluginCommon::mForceMultiBlockWarned = false;

template <typename T, typename KVCacheBuffer>
int GPTAttentionPluginCommon::enqueueGeneration(
    const EnqueueGenerationParams<T, KVCacheBuffer>& params, cudaStream_t stream)
{
    const int step = params.past_kv_length + 1;

    const int num_heads = mNumHeads;
    const int num_kv_heads = mNumKVHeads;
    const int head_size = getHeadSize();
    const int local_hidden_units_qo = num_heads * head_size;
    const int local_hidden_units_kv = num_kv_heads * head_size;
    const PositionEmbeddingType position_embedding_type = mPositionEmbeddingType;
    const float q_scaling = mQScaling;
    const T* relative_attention_bias = isRelativePosition() ? params.relative_attention_bias : nullptr;
    const int relative_attention_bias_stride = isRelativePosition() ? params.relative_attention_bias_stride : 0;
    const int max_distance = mMaxDistance;
    const bool* finished = nullptr;
    const bool has_ia3 = false;

    const auto quant_option = tc::QuantMode::fromDescription();
    const float* qkv_scale_out = nullptr;
    const float* attention_out_scale = nullptr;

    const int* ia3_tasks = nullptr;
    const T* ia3_key_weights = nullptr;
    const T* ia3_value_weights = nullptr;

    int32_t const batch_beam = params.beam_width * params.num_requests;

    KVCacheBuffer kv_cache_buffer;
    const auto elem_size = mKVCacheQuantMode.hasKvCacheQuant() ? sizeof(int8_t) : sizeof(T);
    if (useKVCache())
    {
        if (mPagedKVCache)
        {
            using BufferDataType = typename KVCacheBufferDataType<KVCacheBuffer>::Type;
            kv_cache_buffer = KVCacheBuffer(batch_beam, params.max_blocks_per_sequence, mTokensPerBlock,
                num_kv_heads * head_size * elem_size, params.cyclic_attention_window_size, params.sink_token_length,
                false);
            kv_cache_buffer.data = reinterpret_cast<BufferDataType*>(params.block_pointers);
        }
        else
        {
            using BufferDataType = typename KVCacheBufferDataType<KVCacheBuffer>::Type;
            kv_cache_buffer
                = KVCacheBuffer(batch_beam, 1, params.max_attention_window, num_kv_heads * head_size * elem_size,
                    params.cyclic_attention_window_size, params.sink_token_length, false);
            kv_cache_buffer.data = reinterpret_cast<BufferDataType*>(params.key_value_cache);
        }
    }
    sync_check_cuda_error();

    // Try XQA optimization first.
    {
        // NOTE: input_seq_length = num_medusa_tokens + 1 (new generated one from the original LM head)
        // self attn
        XQAParams xqaParams{};
        if (tensorrt_llm::kernels::XQADispatchHelper<T, KVCacheBuffer>::CanSupport && mDecoderXQARunner.get() != nullptr
            && this->template convertMMHAParamsToXQAParams<T, KVCacheBuffer>(xqaParams, params)
            && mDecoderXQARunner->template shouldUse<T>(xqaParams))
        {
            TLLM_LOG_DEBUG("XQA kernels are selected in the generation phase.");
            mDecoderXQARunner->template dispatch<KVCacheBuffer>(xqaParams, kv_cache_buffer, stream);
            return 0;
        }
        else if (mIsMedusaEnabled)
        {
            TLLM_CHECK_WITH_INFO(false, "No available XQA kernels are found for medusa mode.");
        }
    }

    int timestep = params.past_kv_length;
    const int max_timesteps = mCrossAttention ? params.cyclic_attention_window_size
                                              : std::min(timestep, params.cyclic_attention_window_size);
    int estimated_min_multi_block_count
        = estimate_min_multi_block_count<T>(max_timesteps, mMaxSharedMemoryPerBlockOptin - 2048);

    if (!mMultiBlockMode && !mForceMultiBlockWarned && estimated_min_multi_block_count > 1)
    {
        mForceMultiBlockWarned = true;
        TLLM_LOG_WARNING(
            "Force using MultiBlockMode in MMHA as shared memory is not enough, "
            "MultiBlockMode may have different accuracy compared to non-MultiBlockMode.");
    }

    int8_t* workspace_byte_ptr = reinterpret_cast<int8_t*>(params.workspace);
    size_t offset = 0;
    // estimate min block count to satisfy shared memory requirement to run kernel.
    // Runtime check to see the actual number of blocks per sequence we need.
    int32_t const max_num_seq_len_tiles = std::max(getMaxNumSeqLenTile(batch_beam), estimated_min_multi_block_count);
    int32_t const min_num_seq_len_tiles = std::max(1, estimated_min_multi_block_count);
    const bool enable_multi_block
        = (mMultiBlockMode && max_num_seq_len_tiles > 1) || estimated_min_multi_block_count > 1;
    const size_t partial_out_size
        = enable_multi_block ? sizeof(T) * batch_beam * mNumHeads * mHeadSize * max_num_seq_len_tiles : 0;
    const size_t partial_sum_size
        = enable_multi_block ? sizeof(float) * batch_beam * mNumHeads * max_num_seq_len_tiles : 0;
    const size_t partial_max_size
        = enable_multi_block ? sizeof(float) * batch_beam * mNumHeads * max_num_seq_len_tiles : 0;
    const size_t block_counter_size = enable_multi_block ? sizeof(int) * batch_beam * mNumHeads : 0;
    const size_t shift_k_cache_size = (!mPosShiftEnabled || isCrossAttention())
        ? 0
        : sizeof(T) * batch_beam * mNumHeads * mHeadSize * params.max_attention_window;

    // Workspace pointer shift
    T* partial_out = reinterpret_cast<T*>(nextWorkspacePtr(workspace_byte_ptr, offset, partial_out_size));
    float* partial_sum = reinterpret_cast<float*>(nextWorkspacePtr(workspace_byte_ptr, offset, partial_sum_size));
    float* partial_max = reinterpret_cast<float*>(nextWorkspacePtr(workspace_byte_ptr, offset, partial_max_size));
    int* block_counter = reinterpret_cast<int*>(nextWorkspacePtr(workspace_byte_ptr, offset, block_counter_size));
    T* shift_k_cache = reinterpret_cast<T*>(nextWorkspacePtr(workspace_byte_ptr, offset, shift_k_cache_size));

    if (enable_multi_block)
    {
        TLLM_CUDA_CHECK(cudaMemsetAsync(block_counter, 0, block_counter_size, stream));
    }

    // Apply position embedding to the keys in the K cache
    KVLinearBuffer shift_k_cache_buffer;
    if (mPosShiftEnabled && !isCrossAttention())
    {
        shift_k_cache_buffer = KVLinearBuffer(batch_beam, 1, params.max_attention_window,
            num_kv_heads * head_size * elem_size, params.cyclic_attention_window_size, params.sink_token_length, true);
        shift_k_cache_buffer.data = reinterpret_cast<int8_t*>(shift_k_cache);
        sync_check_cuda_error();
        // KV cache type
        const KvCacheDataType kv_cache_type = KvCacheDataType::BASE;
        using DataType = typename SATypeConverter<T>::Type;
        invokeShiftKCache<DataType, KVCacheBuffer>(kv_cache_buffer, shift_k_cache_buffer, kv_cache_type, getHeadSize(),
            step - 1, batch_beam, mNumKVHeads, params.beam_width, params.cyclic_attention_window_size,
            params.sink_token_length, params.kv_scale_quant_orig, params.sequence_lengths, params.context_lengths,
            mRotaryEmbeddingDim, mRotaryEmbeddingBase, mRotaryEmbeddingScaleType, mRotaryEmbeddingScale,
            mRotaryEmbeddingMaxPositions, mPositionEmbeddingType, stream);
    }

    FusedQKVMaskedAttentionDispatchParams<T, KVCacheBuffer> dispatch_params;
    memset(&dispatch_params, 0, sizeof(dispatch_params));
    dispatch_params.mUnfuseQkvGemm = mUnfuseQkvGemm;
    dispatch_params.qkv_buf = params.attention_input;
    dispatch_params.qkv_bias = params.qkv_bias;
    dispatch_params.relative_attention_bias = relative_attention_bias;
    dispatch_params.relative_attention_bias_stride = relative_attention_bias_stride;
    dispatch_params.max_distance = max_distance;
    dispatch_params.cache_indir = params.cache_indir;
    dispatch_params.context_buf = params.context_buf;
    dispatch_params.finished = finished;
    dispatch_params.sequence_lengths
        = params.sequence_lengths; // NOTE: current seq len including padding (fixed after meeting the finished id)
    dispatch_params.max_batch_size = batch_beam;
    dispatch_params.inference_batch_size = batch_beam;
    dispatch_params.beam_width = params.beam_width;
    dispatch_params.head_num = mNumHeads;
    dispatch_params.kv_head_num = mNumKVHeads;
    dispatch_params.size_per_head = getHeadSize();
    dispatch_params.rotary_embedding_dim = mRotaryEmbeddingDim;
    dispatch_params.position_embedding_type = mPositionEmbeddingType;
    dispatch_params.max_attention_window = params.max_attention_window;
    dispatch_params.cyclic_attention_window_size = params.cyclic_attention_window_size;
    dispatch_params.sink_token_length = isCrossAttention() ? 0 : params.sink_token_length;
    dispatch_params.input_lengths = params.context_lengths;
    dispatch_params.step = step;
    dispatch_params.q_scaling = q_scaling;
    dispatch_params.linear_bias_slopes = isALiBi() ? params.alibi_slopes : nullptr;
    dispatch_params.ia3_tasks = ia3_tasks;
    dispatch_params.ia3_key_weights = ia3_key_weights;
    dispatch_params.ia3_value_weights = ia3_value_weights;
    dispatch_params.qkv_scale_out = qkv_scale_out;
    dispatch_params.attention_out_scale = attention_out_scale;
    dispatch_params.quant_option = quant_option;
    dispatch_params.multi_block_mode = enable_multi_block;
    dispatch_params.max_seq_len_tile = max_num_seq_len_tiles;
    dispatch_params.min_seq_len_tile = min_num_seq_len_tiles;
    dispatch_params.partial_out = partial_out;
    dispatch_params.partial_sum = partial_sum;
    dispatch_params.partial_max = partial_max;
    dispatch_params.block_counter = block_counter;
    dispatch_params.kv_cache_quant_mode = mKVCacheQuantMode;
    dispatch_params.kv_scale_orig_quant = params.kv_scale_orig_quant;
    dispatch_params.kv_scale_quant_orig = params.kv_scale_quant_orig;
    dispatch_params.kv_block_array = kv_cache_buffer;
    dispatch_params.shift_k_cache_buffer = shift_k_cache_buffer;
    dispatch_params.multi_processor_count = mMultiProcessorCount;
    dispatch_params.rotary_embedding_base = mRotaryEmbeddingBase;
    dispatch_params.rotary_embedding_scale_type = mRotaryEmbeddingScaleType;
    dispatch_params.rotary_embedding_scale = mRotaryEmbeddingScale;
    dispatch_params.rotary_embedding_max_positions = mRotaryEmbeddingMaxPositions;
    dispatch_params.position_shift_enabled = mPosShiftEnabled;
    dispatch_params.cross_attention = mCrossAttention;
    dispatch_params.memory_length_per_sample = params.encoder_input_lengths;

    using DataType = typename SATypeConverter<T>::Type;
    if (!mCrossAttention)
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

    return 0;
}

template int GPTAttentionPluginCommon::enqueueGeneration<half, KVLinearBuffer>(
    const EnqueueGenerationParams<half, KVLinearBuffer>& params, cudaStream_t stream);

template int GPTAttentionPluginCommon::enqueueGeneration<float, KVLinearBuffer>(
    const EnqueueGenerationParams<float, KVLinearBuffer>& params, cudaStream_t stream);

#ifdef ENABLE_BF16
template int GPTAttentionPluginCommon::enqueueGeneration<__nv_bfloat16, KVLinearBuffer>(
    const EnqueueGenerationParams<__nv_bfloat16, KVLinearBuffer>& params, cudaStream_t stream);
#endif

template int GPTAttentionPluginCommon::enqueueGeneration<half, KVBlockArray>(
    const EnqueueGenerationParams<half, KVBlockArray>& params, cudaStream_t stream);

template int GPTAttentionPluginCommon::enqueueGeneration<float, KVBlockArray>(
    const EnqueueGenerationParams<float, KVBlockArray>& params, cudaStream_t stream);

#ifdef ENABLE_BF16
template int GPTAttentionPluginCommon::enqueueGeneration<__nv_bfloat16, KVBlockArray>(
    const EnqueueGenerationParams<__nv_bfloat16, KVBlockArray>& params, cudaStream_t stream);
#endif

int GPTAttentionPluginCommon::initialize() noexcept
{
    auto cublasHandle = getCublasHandle();
    auto cublasLtHandle = getCublasLtHandle();

    // Pre-warm getting environment variables
    getEnvMmhaMultiblockDebug();
    getEnvMmhaBlocksPerSequence();

    mCublasWrapper.reset(new tc::CublasMMWrapper(cublasHandle, cublasLtHandle, nullptr, nullptr));
    if (mEnableContextFMHA)
    {
        // Pre-checked during constructing.
        Data_type data_type;
        if (mType == nvinfer1::DataType::kHALF)
        {
            data_type = DATA_TYPE_FP16;
        }
        else if (mType == nvinfer1::DataType::kBF16)
        {
            data_type = DATA_TYPE_BF16;
        }
        else
        {
            TLLM_CHECK_WITH_INFO(false, "GPTAttentionPlugin received wrong data type.");
        }

        // Load kernels for contiguous cache and paged kv cache at the same time.
        mFMHARunner.reset(new FusedMHARunnerV2(data_type, mNumHeads, getHeadSize(false), mQScaling));
        // Set flags: force_fp32_acc, is_s_padded, causal_mask, num_kv_heads.
        mFMHARunner->setup_flags(mFMHAForceFP32Acc, !mRemovePadding, true, mNumKVHeads);
    }

    bool useXQAKernels = (mEnableXQA || mIsMedusaEnabled) && !mCrossAttention
        && (mType == nvinfer1::DataType::kHALF || mType == nvinfer1::DataType::kBF16);

    if (useXQAKernels)
    {
        Data_type xqa_runner_data_type;
        if (mType == nvinfer1::DataType::kHALF)
        {
            xqa_runner_data_type = DATA_TYPE_FP16;
        }
        else if (mType == nvinfer1::DataType::kBF16)
        {
            xqa_runner_data_type = DATA_TYPE_BF16;
        }
        TLLM_LOG_DEBUG("Enabling XQA kernels for GPTAttention.");
        if (mIsMedusaEnabled)
        {
            TLLM_CHECK_WITH_INFO(mNumHeads % mNumKVHeads == 0, "mNumHeads should be multiples of mNumKVHeads.");
            int numQHeadsPerKV = mNumHeads / mNumKVHeads;
            bool isPowerOfTwo = ((numQHeadsPerKV & (numQHeadsPerKV - 1)) == 0);
            TLLM_CHECK_WITH_INFO(isPowerOfTwo,
                "numQHeadsPerKV should be power of 2 for Medusa, mNumHeads=%d, mNumKVHeads=%d.", mNumHeads,
                mNumKVHeads);
        }

        mDecoderXQARunner.reset(
            new DecoderXQARunner(xqa_runner_data_type, mNumHeads, mNumKVHeads, mHeadSize, mMultiBlockMode));
    }
    else if (mIsMedusaEnabled)
    {
        TLLM_CHECK_WITH_INFO(false, "Medusa mode doesn't support the data type or cross attention.");
    }

    return 0;
}

void GPTAttentionPluginCommon::destroy() noexcept
{
    delete this;
}

size_t GPTAttentionPluginCommon::getCommonSerializationSize() noexcept
{
    return sizeof(mNumHeads) + sizeof(mNumKVHeads) + sizeof(mHeadSize) + sizeof(mUnidirectional) + sizeof(mQScaling)
        + sizeof(mPositionEmbeddingType) + sizeof(mRotaryEmbeddingDim) + sizeof(mRotaryEmbeddingBase)
        + sizeof(mRotaryEmbeddingScaleType) + sizeof(mRotaryEmbeddingScale) + sizeof(mRotaryEmbeddingMaxPositions)
        + sizeof(mTpSize) + sizeof(mTpRank) + sizeof(mEnableContextFMHA) + sizeof(mFMHAForceFP32Acc)
        + sizeof(mMultiBlockMode) + sizeof(mEnableXQA) + sizeof(unsigned int) // mKVCacheQuantMode
        + sizeof(mRemovePadding) + sizeof(mMaskType) + sizeof(mPagedKVCache) + sizeof(mTokensPerBlock) + sizeof(mType)
        + sizeof(mMaxContextLength) + sizeof(mQKVBiasEnabled) + sizeof(mCrossAttention) + sizeof(mMaxDistance)
        + sizeof(mPosShiftEnabled) + sizeof(mDenseContextFMHA) + sizeof(mPagedContextFMHA) + sizeof(mUseKVCache)
        + sizeof(mUnfuseQkvGemm) + sizeof(mIsMedusaEnabled);
}

void GPTAttentionPluginCommon::serializeCommon(void* buffer) const noexcept
{
    char *d = static_cast<char*>(buffer), *a = d;
    write(d, mNumHeads);
    write(d, mNumKVHeads);
    write(d, mHeadSize);
    write(d, mUnidirectional);
    write(d, mQScaling);
    write(d, mPositionEmbeddingType);
    write(d, mRotaryEmbeddingDim);
    write(d, mRotaryEmbeddingBase);
    write(d, mRotaryEmbeddingScaleType);
    write(d, mRotaryEmbeddingScale);
    write(d, mRotaryEmbeddingMaxPositions);
    write(d, mTpSize);
    write(d, mTpRank);
    write(d, mUnfuseQkvGemm);
    write(d, mEnableContextFMHA);
    write(d, mFMHAForceFP32Acc);
    write(d, mMultiBlockMode);
    write(d, mEnableXQA);
    write(d, mKVCacheQuantMode.value());
    write(d, mRemovePadding);
    write(d, mMaskType);
    write(d, mPagedKVCache);
    write(d, mTokensPerBlock);
    write(d, mType);
    write(d, mMaxContextLength);
    write(d, mQKVBiasEnabled);
    write(d, mCrossAttention);
    write(d, mMaxDistance);
    write(d, mPosShiftEnabled);
    write(d, mDenseContextFMHA);
    write(d, mPagedContextFMHA);
    write(d, mUseKVCache);
    write(d, mIsMedusaEnabled);
    assert(d == a + getCommonSerializationSize());
}

void GPTAttentionPluginCommon::terminate() noexcept
{
    // Do nothing, destroy will always be called, so release the resources there.
}

///////////////

GPTAttentionPluginCreatorCommon::GPTAttentionPluginCreatorCommon()
{
    // Fill PluginFieldCollection with PluginField arguments metadata
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("num_heads", nullptr, PluginFieldType::kINT32, -1));
    mPluginAttributes.emplace_back(PluginField("num_kv_heads", nullptr, PluginFieldType::kINT32, 0));
    mPluginAttributes.emplace_back(PluginField("head_size", nullptr, PluginFieldType::kINT32, 0));
    mPluginAttributes.emplace_back(PluginField("unidirectional", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("q_scaling", nullptr, PluginFieldType::kFLOAT32, 1.0));
    mPluginAttributes.emplace_back(PluginField("position_embedding_type", nullptr, PluginFieldType::kINT8, 0));
    mPluginAttributes.emplace_back(PluginField("rotary_embedding_dim", nullptr, PluginFieldType::kINT32, 0));
    mPluginAttributes.emplace_back(PluginField("rotary_embedding_base", nullptr, PluginFieldType::kFLOAT32, 0));
    mPluginAttributes.emplace_back(PluginField("rotary_embedding_scale_type", nullptr, PluginFieldType::kINT8, 0));
    mPluginAttributes.emplace_back(PluginField("rotary_embedding_scale", nullptr, PluginFieldType::kFLOAT32, 0));
    mPluginAttributes.emplace_back(PluginField("rotary_embedding_max_positions", nullptr, PluginFieldType::kINT32, 0));
    mPluginAttributes.emplace_back(PluginField("tp_size", nullptr, PluginFieldType::kINT32, 0));
    mPluginAttributes.emplace_back(PluginField("tp_rank", nullptr, PluginFieldType::kINT32, 0));
    mPluginAttributes.emplace_back(PluginField("unfuse_qkv_gemm", nullptr, PluginFieldType::kINT8, 0));
    mPluginAttributes.emplace_back(PluginField("context_fmha_type", nullptr, PluginFieldType::kINT8, 0));
    mPluginAttributes.emplace_back(PluginField("multi_block_mode", nullptr, PluginFieldType::kINT8, 0));
    mPluginAttributes.emplace_back(PluginField("enable_xqa", nullptr, PluginFieldType::kINT8, 0));
    mPluginAttributes.emplace_back(PluginField("kv_cache_quant_mode", nullptr, PluginFieldType::kINT32, 0));
    mPluginAttributes.emplace_back(PluginField("remove_input_padding", nullptr, PluginFieldType::kINT8, 0));
    mPluginAttributes.emplace_back(PluginField("mask_type", nullptr, PluginFieldType::kINT32, 0));
    mPluginAttributes.emplace_back(PluginField("paged_kv_cache", nullptr, PluginFieldType::kINT32, 0));
    mPluginAttributes.emplace_back(PluginField("tokens_per_block", nullptr, PluginFieldType::kINT32, 0));
    mPluginAttributes.emplace_back(PluginField("type_id", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("max_context_length", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("qkv_bias_enabled", nullptr, PluginFieldType::kINT8, 0));
    mPluginAttributes.emplace_back(PluginField("do_cross_attention", nullptr, PluginFieldType::kINT8, 0));
    mPluginAttributes.emplace_back(PluginField("max_distance", nullptr, PluginFieldType::kINT32, 0));
    mPluginAttributes.emplace_back(PluginField("pos_shift_enabled", nullptr, PluginFieldType::kINT8, 0));
    mPluginAttributes.emplace_back(PluginField("dense_context_fmha", nullptr, PluginFieldType::kINT8, 0));
    mPluginAttributes.emplace_back(PluginField("use_paged_context_fmha", nullptr, PluginFieldType::kINT8, 0));
    mPluginAttributes.emplace_back(PluginField("use_cache", nullptr, PluginFieldType::kINT32, 0));
    mPluginAttributes.emplace_back(PluginField("is_medusa_enabled", nullptr, PluginFieldType::kINT8, 0));
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const PluginFieldCollection* GPTAttentionPluginCreatorCommon::getFieldNames() noexcept
{
    return &mFC;
}
