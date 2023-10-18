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
#include "tensorrt_llm/kernels/decoderMaskedMultiheadAttention.h"
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
    int max_seq_len;
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
    tc::QuantMode quant_option;
    bool multi_block_mode;
    int max_seq_len_tile;
    T* partial_out;
    float* partial_sum;
    float* partial_max;
    int* block_counter;
    const float* kv_scale_orig_quant;
    const float* kv_scale_quant_orig;
    tc::QuantMode kv_cache_quant_mode;
    int multi_processor_count;
    KVCacheBuffer kv_block_array;
    bool cross_attention = false;
    const int* memory_length_per_sample = nullptr;
    int max_distance = 0;
};

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
    params.memory_max_len = input_params.max_seq_len;
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

    masked_multihead_attention(params, input_params.kv_block_array, stream);
}

#define INSTANTIATE_MMHA_DISPATH(T_MMHA, T)                                                                            \
    template void fusedQKV_masked_attention_dispatch(Multihead_attention_params<T_MMHA, false>&,                       \
        const FusedQKVMaskedAttentionDispatchParams<T, KVLinearBuffer>&, cudaStream_t stream);                         \
    template void fusedQKV_masked_attention_dispatch(Multihead_attention_params<T_MMHA, true>&,                        \
        const FusedQKVMaskedAttentionDispatchParams<T, KVLinearBuffer>&, cudaStream_t stream);                         \
    template void fusedQKV_masked_attention_dispatch(Multihead_attention_params<T_MMHA, false>&,                       \
        const FusedQKVMaskedAttentionDispatchParams<T, KVBlockArray>&, cudaStream_t stream);                           \
    template void fusedQKV_masked_attention_dispatch(Multihead_attention_params<T_MMHA, true>&,                        \
        const FusedQKVMaskedAttentionDispatchParams<T, KVBlockArray>&, cudaStream_t stream);
INSTANTIATE_MMHA_DISPATH(float, float)
INSTANTIATE_MMHA_DISPATH(uint16_t, half)
#ifdef ENABLE_BF16
INSTANTIATE_MMHA_DISPATH(__nv_bfloat16, __nv_bfloat16)
#endif
#undef INSTANTIATE_MMHA_DISPATH

GPTAttentionPluginCommon::GPTAttentionPluginCommon(int num_heads, int num_kv_heads, int head_size, int unidirectional,
    float q_scaling, tensorrt_llm::kernels::PositionEmbeddingType position_embedding_type,
    int rotary_embedding_dim, // for RoPE. Use 0 for non-RoPE
    float rotary_embedding_base, tensorrt_llm::kernels::RotaryScalingType rotary_embedding_scale_type,
    float rotary_embedding_scale, int rotary_embedding_max_positions, int tp_size, int tp_rank, // for ALiBi
    tensorrt_llm::kernels::ContextFMHAType context_fmha_type, bool multi_block_mode, int kv_cache_quant_mode,
    bool remove_input_padding, tensorrt_llm::kernels::AttentionMaskType mask_type, bool paged_kv_cache,
    int tokens_per_block, nvinfer1::DataType type, int32_t max_context_length, bool qkv_bias_enabled,
    bool cross_attention, int max_distance)
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
    , mFMHAForceFP32Acc(context_fmha_type == ContextFMHAType::ENABLED_WITH_FP32_ACC || type == DataType::kBF16)
    , mMaskType(mask_type)
    , mType(type)
    , mMultiBlockMode(multi_block_mode)
    , mKVCacheQuantMode(kv_cache_quant_mode)
    , mRemovePadding(remove_input_padding)
    , mPagedKVCache(paged_kv_cache)
    , mTokensPerBlock(tokens_per_block)
    , mTpSize(tp_size)
    , mTpRank(tp_rank)
    , mMaxContextLength(max_context_length)
    , mQKVBiasEnabled(qkv_bias_enabled)
    , mCrossAttention(cross_attention)
    , mMaxDistance(max_distance)
{
    mEnableContextFMHA = mEnableContextFMHA && (mType == DataType::kHALF || mType == DataType::kBF16);
    TLLM_CHECK(isRoPE() == (rotary_embedding_dim != 0));
    TLLM_CHECK_WITH_INFO((tc::getSMVersion() >= 80) || (mType != DataType::kBF16),
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
    read(d, mEnableContextFMHA);
    read(d, mFMHAForceFP32Acc);
    read(d, mMultiBlockMode);
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

    mKVCacheQuantMode = tc::QuantMode(kvCacheQuantMode);

    TLLM_CHECK(d == a + length);
    TLLM_CHECK_WITH_INFO((tc::getSMVersion() >= 80) || (mType != DataType::kBF16),
        "Unsupported data type, pre SM 80 GPUs do not support bfloat16");
}

size_t GPTAttentionPluginCommon::getWorkspaceSizeForContext(
    DataType type, int32_t nbReq, int32_t input_seq_length, int32_t cross_qkv_length) const noexcept
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
    const size_t q_buf_2_size = mEnableContextFMHA ? 0 : size * batch_size * input_seq_length * local_hidden_units_qo;
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

    const int NUM_BUFFERS = 10;
    size_t workspaces[NUM_BUFFERS];
    workspaces[0] = CUBLAS_WORKSPACE_SIZE;
    workspaces[1] = attention_mask_size;
    workspaces[2] = cu_seqlens_size;
    workspaces[3] = q_buf_2_size;
    workspaces[4] = k_buf_2_size;
    workspaces[5] = v_buf_2_size;
    workspaces[6] = qk_buf_size;
    workspaces[7] = qkv_buf_2_size;
    workspaces[8] = qk_buf_float_size;
    workspaces[9] = padding_offset_size;
    context_workspace_size = tensorrt_llm::plugins::calculateTotalWorkspaceSize(workspaces, NUM_BUFFERS);
    return context_workspace_size;
}

size_t GPTAttentionPluginCommon::getWorkspaceSizeForGeneration(DataType type, int32_t total_num_seq) const noexcept
{
    const int local_hidden_units_qo = mNumHeads * getHeadSize();
    const int local_hidden_units_kv = mNumKVHeads * getHeadSize();

    auto const size = tensorrt_llm::runtime::BufferDataType(type).getSize();

    size_t context_workspace_size = 0;
    size_t generation_workspace_size = 0;

    const int batch_beam = total_num_seq;
    int32_t const maxSeqLenTile = getMaxSeqLenTile(size);

    const size_t partial_out_size = mMultiBlockMode ? size * batch_beam * mNumHeads * mHeadSize * maxSeqLenTile : 0;
    const size_t partial_sum_size = mMultiBlockMode ? sizeof(float) * batch_beam * mNumHeads * maxSeqLenTile : 0;
    const size_t partial_max_size = mMultiBlockMode ? sizeof(float) * batch_beam * mNumHeads * maxSeqLenTile : 0;
    const size_t block_counter_size = mMultiBlockMode ? sizeof(int) * batch_beam * mNumHeads : 0;

    const int NUM_BUFFERS = 4;
    size_t workspaces[NUM_BUFFERS];
    workspaces[0] = partial_out_size;
    workspaces[1] = partial_sum_size;
    workspaces[2] = partial_max_size;
    workspaces[3] = block_counter_size;
    generation_workspace_size = tensorrt_llm::plugins::calculateTotalWorkspaceSize(workspaces, NUM_BUFFERS);
    return generation_workspace_size;
}

int GPTAttentionPluginCommon::getMaxSeqLenTile(int elemSize) const
{
    if (mMultiBlockMode)
    {
        const int threads_per_value = pow2roundup(getHeadSize()) * elemSize / 16;

        // max_seq_len_tile to make sure: seq_len_tile * threads_per_value <= threads_per_block (for
        // multi_block_mode)
        const int max_seq_len_tile
            = 256 / threads_per_value; // for allocate partial output results memory. Regardless to THDS_PER_BLOCK
                                       // (which may be smaller than 256 like being 128)
        return max_seq_len_tile;
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
    if (mPagedKVCache)
    {
        using BufferDataType = typename KVCacheBufferDataType<KVCacheBuffer>::Type;
        kv_cache_buffer = KVCacheBuffer(
            params.batch_size, params.max_blocks_per_sequence, mTokensPerBlock, num_kv_heads * head_size * elem_size);
        kv_cache_buffer.data = reinterpret_cast<BufferDataType*>(params.block_pointers);
    }
    else
    {
        using BufferDataType = typename KVCacheBufferDataType<KVCacheBuffer>::Type;
        kv_cache_buffer = KVCacheBuffer(params.batch_size, 1,
            isCrossAttention() ? params.cross_qkv_length : params.max_seq_length, num_kv_heads * head_size * elem_size);
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
    const size_t q_buf_2_size
        = mEnableContextFMHA ? 0 : sizeof(T) * params.batch_size * params.input_seq_length * local_hidden_units_qo;
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
    const size_t padding_offset_size = sizeof(int) * params.batch_size * params.input_seq_length;

    const bool is_qk_buf_float_ = true;

    // Workspace pointer shift
    int8_t* workspace_byte_ptr = reinterpret_cast<int8_t*>(params.workspace);
    size_t offset = CUBLAS_WORKSPACE_SIZE;

    T* attention_mask = reinterpret_cast<T*>(nextWorkspacePtr(workspace_byte_ptr, offset, attention_mask_size));
    int* cu_seqlens = reinterpret_cast<int*>(nextWorkspacePtr(workspace_byte_ptr, offset, cu_seqlens_size));
    T* q_buf_2_ = reinterpret_cast<T*>(nextWorkspacePtr(workspace_byte_ptr, offset, q_buf_2_size));
    T* k_buf_2_ = reinterpret_cast<T*>(nextWorkspacePtr(workspace_byte_ptr, offset, k_buf_2_size));
    T* v_buf_2_ = reinterpret_cast<T*>(nextWorkspacePtr(workspace_byte_ptr, offset, v_buf_2_size));
    T* qk_buf_ = reinterpret_cast<T*>(nextWorkspacePtr(workspace_byte_ptr, offset, qk_buf_size));
    T* qkv_buf_2_ = reinterpret_cast<T*>(nextWorkspacePtr(workspace_byte_ptr, offset, qkv_buf_2_size));
    float* qk_buf_float_ = reinterpret_cast<float*>(nextWorkspacePtr(workspace_byte_ptr, offset, qk_buf_float_size));
    int* padding_offset = reinterpret_cast<int*>(nextWorkspacePtr(workspace_byte_ptr, offset, padding_offset_size));

    // build attention_mask, cu_seqlens, and padding_offset tensors
    BuildDecoderInfoParams<T> decoder_params;
    memset(&decoder_params, 0, sizeof(decoder_params));
    decoder_params.seqOffsets = cu_seqlens;
    decoder_params.paddingOffsets = padding_offset;
    decoder_params.attentionMask = attention_mask;
    decoder_params.seqLengths = params.context_lengths;
    decoder_params.batchSize = params.batch_size;
    decoder_params.maxSeqLength = params.input_seq_length;
    decoder_params.numTokens = params.num_tokens;
    decoder_params.attentionMaskType = mMaskType;
    invokeBuildDecoderInfo(decoder_params, stream);
    sync_check_cuda_error();

    // In cross attention context phase, the attention mask should be a matrix of all ones.
    // We reassign attention_mask to override what previous invokeBuildDecoderInfo() does
    if (isCrossAttention())
    {
        std::vector<T> h_attention_mask(params.batch_size * params.cross_qkv_length * params.input_seq_length, 1.);
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
    if (mEnableContextFMHA && !isCrossAttention() && !isRelativePosition())
    {
        invokeApplyBiasRopeUpdateKVCache(const_cast<T*>(params.attention_input), kv_cache_buffer,
            const_cast<T*>(params.qkv_bias), params.context_lengths, mRemovePadding ? padding_offset : nullptr,
            params.batch_size, params.input_seq_length, params.num_tokens, mNumHeads, mNumKVHeads, getHeadSize(),
            mRotaryEmbeddingDim, mRotaryEmbeddingBase, mRotaryEmbeddingScaleType, mRotaryEmbeddingScale,
            mRotaryEmbeddingMaxPositions, position_embedding_type, (float*) nullptr, 0, cache_type,
            params.kv_scale_orig_quant, stream);
        mFMHARunner->setup(params.batch_size, params.input_seq_length, params.num_tokens, isALiBi(), isAliBiWithScale(),
            mTpSize, mTpRank);
        mFMHARunner->run(const_cast<T*>(params.attention_input), cu_seqlens, params.context_buf, stream);
    }
    else
    {
        // FIXME: a temporary solution to make sure the padding part of key/value buffer is 0
        // NOTE: pointer subtraction is used below since there could be some extra gap due to alignment.
        //  Otherwise, we could do cudaMemsetAsync(k_buf_2_, 0, k_buf_2_size + v_buf_2_size, stream);
        cudaMemsetAsync(k_buf_2_, 0,
            reinterpret_cast<int8_t*>(v_buf_2_) - reinterpret_cast<int8_t*>(k_buf_2_) + v_buf_2_size, stream);

        if (!isCrossAttention())
        {
            // self attention, write to Q/K/V
            invokeAddFusedQKVBiasTranspose(q_buf_2_, k_buf_2_, v_buf_2_, const_cast<T*>(params.attention_input),
                const_cast<T*>(params.qkv_bias), params.context_lengths, mRemovePadding ? padding_offset : nullptr,
                params.batch_size, params.input_seq_length, params.num_tokens, mNumHeads, mNumKVHeads, getHeadSize(),
                mEnableContextFMHA, mRotaryEmbeddingDim, mRotaryEmbeddingBase, mRotaryEmbeddingScaleType,
                mRotaryEmbeddingScale, mRotaryEmbeddingMaxPositions, position_embedding_type, (float*) nullptr, 0,
                stream);
        }
        else
        {
            // cross attention, write Q from self QKV, write KV from cross QKV
            // kernel modifed accordingly to handle nullptr buffer
            invokeAddFusedQKVBiasTranspose(q_buf_2_, (T*) nullptr, (T*) nullptr, const_cast<T*>(params.attention_input),
                const_cast<T*>(params.qkv_bias), params.context_lengths, mRemovePadding ? padding_offset : nullptr,
                params.batch_size, params.input_seq_length, params.num_tokens, mNumHeads, mNumKVHeads, getHeadSize(),
                mEnableContextFMHA, mRotaryEmbeddingDim, mRotaryEmbeddingBase, mRotaryEmbeddingScaleType,
                mRotaryEmbeddingScale, mRotaryEmbeddingMaxPositions, position_embedding_type, (float*) nullptr, 0,
                stream);
            invokeAddFusedQKVBiasTranspose((T*) nullptr, k_buf_2_, v_buf_2_, const_cast<T*>(params.cross_qkv),
                const_cast<T*>(params.qkv_bias), params.encoder_input_lengths,
                mRemovePadding ? padding_offset : nullptr, params.batch_size, params.cross_qkv_length,
                params.num_encoder_tokens, mNumHeads, mNumKVHeads, getHeadSize(), mEnableContextFMHA,
                mRotaryEmbeddingDim, mRotaryEmbeddingBase, mRotaryEmbeddingScaleType, mRotaryEmbeddingScale,
                mRotaryEmbeddingMaxPositions, position_embedding_type, (float*) nullptr, 0, stream);
        }
        sync_check_cuda_error();

        // write KV to cach
        invokeTranspose4dBatchMajor(k_buf_2_, v_buf_2_, kv_cache_buffer, params.batch_size,
            isCrossAttention() ? params.cross_qkv_length : params.input_seq_length,
            isCrossAttention() ? params.cross_qkv_length : params.max_seq_length, getHeadSize(), mNumKVHeads,
            cache_type, params.kv_scale_orig_quant,
            isCrossAttention() ? params.encoder_input_lengths : params.context_lengths, stream);
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

        // add relative position bias
        if (isRelativePosition())
        {
            // Add relative_attention_bias
            // QK is (batch_size, local_head_num, q_length, k_length), relative_attention_bias is (1, local_head_num,
            // max_output_len + 1, max_output_len + 1).
            // broadcast along 1st dim. max_seq_len is already max_output_len + 1.
            // In implicit mode, relative_attention_bias is relative_attention_table [num_heads, num_buckets], with
            // necessary params (max_distance, num_buckets) passed at the end
            invokeAddRelativeAttentionBiasUnaligned(qk_buf_float_, relative_attention_bias, params.batch_size,
                mNumHeads, attention_seq_len_1, isCrossAttention() ? params.cross_qkv_length : params.max_seq_length,
                stream, max_distance > 0, relative_attention_bias_stride, max_distance, true /* bidirectional */);
        }

        if (is_qk_buf_float_ == true)
        {
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

    const bool multi_block_mode = false;
    int32_t const batch_beam = params.beam_width * params.num_requests;

    int8_t* workspace_byte_ptr = reinterpret_cast<int8_t*>(params.workspace);
    size_t offset = 0;
    int32_t const maxSeqLenTile = getMaxSeqLenTile(sizeof(T));
    const size_t partial_out_size
        = mMultiBlockMode ? sizeof(T) * batch_beam * mNumHeads * mHeadSize * maxSeqLenTile : 0;
    const size_t partial_sum_size = mMultiBlockMode ? sizeof(float) * batch_beam * mNumHeads * maxSeqLenTile : 0;
    const size_t partial_max_size = mMultiBlockMode ? sizeof(float) * batch_beam * mNumHeads * maxSeqLenTile : 0;
    const size_t block_counter_size = mMultiBlockMode ? sizeof(int) * batch_beam * mNumHeads : 0;

    // Workspace pointer shift
    T* partial_out = reinterpret_cast<T*>(nextWorkspacePtr(workspace_byte_ptr, offset, partial_out_size));
    float* partial_sum = reinterpret_cast<float*>(nextWorkspacePtr(workspace_byte_ptr, offset, partial_sum_size));
    float* partial_max = reinterpret_cast<float*>(nextWorkspacePtr(workspace_byte_ptr, offset, partial_max_size));
    int* block_counter = reinterpret_cast<int*>(nextWorkspacePtr(workspace_byte_ptr, offset, block_counter_size));
    if (mMultiBlockMode)
    {
        TLLM_CUDA_CHECK(cudaMemsetAsync(block_counter, 0, block_counter_size, stream));
    }

    KVCacheBuffer kv_cache_buffer;
    const auto elem_size = mKVCacheQuantMode.hasKvCacheQuant() ? sizeof(int8_t) : sizeof(T);
    if (mPagedKVCache)
    {
        using BufferDataType = typename KVCacheBufferDataType<KVCacheBuffer>::Type;
        kv_cache_buffer = KVCacheBuffer(
            batch_beam, params.max_blocks_per_sequence, mTokensPerBlock, num_kv_heads * head_size * elem_size);
        kv_cache_buffer.data = reinterpret_cast<BufferDataType*>(params.block_pointers);
    }
    else
    {
        using BufferDataType = typename KVCacheBufferDataType<KVCacheBuffer>::Type;
        kv_cache_buffer = KVCacheBuffer(batch_beam, 1, params.max_seq_length, num_kv_heads * head_size * elem_size);
        kv_cache_buffer.data = reinterpret_cast<BufferDataType*>(params.key_value_cache);
    }
    sync_check_cuda_error();

    FusedQKVMaskedAttentionDispatchParams<T, KVCacheBuffer> dispatch_params;
    memset(&dispatch_params, 0, sizeof(dispatch_params));
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
    dispatch_params.max_seq_len = params.max_seq_length; // difference between max_seq_lengths and max_seq_length?
    dispatch_params.input_lengths = params.context_lengths;
    dispatch_params.step = step;
    dispatch_params.q_scaling = q_scaling;
    dispatch_params.relative_attention_bias_stride = relative_attention_bias_stride;
    dispatch_params.linear_bias_slopes = isALiBi() ? params.alibi_slopes : nullptr;
    dispatch_params.ia3_tasks = ia3_tasks;
    dispatch_params.ia3_key_weights = ia3_key_weights;
    dispatch_params.ia3_value_weights = ia3_value_weights;
    dispatch_params.qkv_scale_out = qkv_scale_out;
    dispatch_params.attention_out_scale = attention_out_scale;
    dispatch_params.quant_option = quant_option;
    dispatch_params.multi_block_mode = mMultiBlockMode;
    dispatch_params.max_seq_len_tile = getMaxSeqLenTile(sizeof(T));
    dispatch_params.partial_out = partial_out;
    dispatch_params.partial_sum = partial_sum;
    dispatch_params.partial_max = partial_max;
    dispatch_params.block_counter = block_counter;
    dispatch_params.kv_cache_quant_mode = mKVCacheQuantMode;
    dispatch_params.kv_scale_orig_quant = params.kv_scale_orig_quant;
    dispatch_params.kv_scale_quant_orig = params.kv_scale_quant_orig;
    dispatch_params.kv_block_array = kv_cache_buffer;
    dispatch_params.multi_processor_count = mMultiProcessorCount;
    dispatch_params.rotary_embedding_base = mRotaryEmbeddingBase;
    dispatch_params.rotary_embedding_scale_type = mRotaryEmbeddingScaleType;
    dispatch_params.rotary_embedding_scale = mRotaryEmbeddingScale;
    dispatch_params.rotary_embedding_max_positions = mRotaryEmbeddingMaxPositions;
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

    mCublasWrapper.reset(new tc::CublasMMWrapper(cublasHandle, cublasLtHandle, nullptr, nullptr));
    if (mEnableContextFMHA)
    {
        // Pre-checked during constructing.
        Data_type data_type;
        if (mType == DataType::kHALF)
        {
            data_type = DATA_TYPE_FP16;
        }
        else if (mType == DataType::kBF16)
        {
            data_type = DATA_TYPE_BF16;
        }
        else
        {
            TLLM_CHECK_WITH_INFO(false, "GPTAttentionPlugin received wrong data type.");
        }

        mFMHARunner.reset(new FusedMHARunnerV2(data_type, mNumHeads, getHeadSize(false), mQScaling));
        // set flags: force_fp32_acc, is_s_padded, causal_mask, num_kv_heads.
        mFMHARunner->setup_flags(mFMHAForceFP32Acc, !mRemovePadding, true, mNumKVHeads);
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
        + sizeof(mMultiBlockMode) + sizeof(unsigned int) // mKVCacheQuantMode
        + sizeof(mRemovePadding) + sizeof(mMaskType) + sizeof(mPagedKVCache) + sizeof(mTokensPerBlock) + sizeof(mType)
        + sizeof(mMaxContextLength) + sizeof(mQKVBiasEnabled) + sizeof(mCrossAttention) + sizeof(mMaxDistance);
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
    write(d, mEnableContextFMHA);
    write(d, mFMHAForceFP32Acc);
    write(d, mMultiBlockMode);
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
    mPluginAttributes.emplace_back(PluginField("context_fmha_type", nullptr, PluginFieldType::kINT8, 0));
    mPluginAttributes.emplace_back(PluginField("multi_block_mode", nullptr, PluginFieldType::kINT8, 0));
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
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const PluginFieldCollection* GPTAttentionPluginCreatorCommon::getFieldNames() noexcept
{
    return &mFC;
}
