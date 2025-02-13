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
#include "attentionOp.h"
#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/dataType.h"
#include "tensorrt_llm/common/envUtils.h"
#include "tensorrt_llm/common/memoryUtils.h"
#include "tensorrt_llm/common/mpiUtils.h"
#include "tensorrt_llm/kernels/decoderMaskedMultiheadAttention.h"
#include "tensorrt_llm/kernels/decoderMaskedMultiheadAttention/decoderXQARunner.h"
#include "tensorrt_llm/kernels/gptKernels.h"
#include "tensorrt_llm/kernels/kvCacheUtils.h"
#include "tensorrt_llm/kernels/rmsnormKernels.h"
#include "tensorrt_llm/kernels/unfusedAttentionKernels.h"
#include "tensorrt_llm/plugins/common/checkMacrosPlugin.h"
#include "tensorrt_llm/runtime/iBuffer.h"
#include "tensorrt_llm/runtime/utils/debugUtils.h"
#include <algorithm>
#include <cstdint>
#include <type_traits>

using namespace tensorrt_llm::kernels;
namespace tc = tensorrt_llm::common;
using tensorrt_llm::common::op::AttentionOp;

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
    float rotary_embedding_short_m_scale;
    float rotary_embedding_long_m_scale;
    int rotary_embedding_max_positions;
    int rotary_embedding_original_max_positions;
    int rotary_cogvlm_vision_start;
    int rotary_cogvlm_vision_length;
    PositionEmbeddingType position_embedding_type;
    bool position_shift_enabled;

    int attention_mask_stride;
    int max_attention_window;
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
bool AttentionOp::convertMMHAParamsToXQAParams(tensorrt_llm::kernels::XQAParams& xqaParams,
    EnqueueGenerationParams<T> const& generationsParams, bool forConfigurePlugin)
{
    bool retval = ConvertMMHAToXQAParamsHelper<T, KVCacheBuffer>::supported;
    if (!retval)
    {
        return false;
    }
    xqaParams = {};
    xqaParams.data_type = ConvertMMHAToXQAParamsHelper<T, KVCacheBuffer>::data_type;

    xqaParams.layer_idx = mLayerIdxInCachePool;
    // TODO(ziqingc): A better description for these parameters affected by CP size
    xqaParams.num_q_heads = mNumHeads / mCpSize;    // when we use CP, the MHA part is spilt like TP
    xqaParams.num_kv_heads = mNumKVHeads / mCpSize; // when we use CP, the MHA part is spilt like TP
    xqaParams.head_size = mHeadSize;
    xqaParams.unidirectional = mUnidirectional;
    xqaParams.q_scaling = mQScaling;
    xqaParams.rotary_embedding_dim = mRotaryEmbeddingDim;
    xqaParams.rotary_embedding_base = mRotaryEmbeddingBase;
    xqaParams.rotary_embedding_scale_type = mRotaryEmbeddingScaleType;
    xqaParams.rotary_embedding_scale = mRotaryEmbeddingScale;
    xqaParams.rotary_embedding_max_positions = mRotaryEmbeddingMaxPositions;
    xqaParams.rotary_vision_start = mVisionStart;
    xqaParams.rotary_vision_length = mVisionLength;
    xqaParams.rotary_cos_sin = generationsParams.rotary_cos_sin;
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
    xqaParams.multi_block_mode = common::getEnvForceDeterministicAttention() ? false : mMultiBlockMode;
    // Medusa mode will have multiple query tokens.
    xqaParams.multi_query_tokens = mIsSpecDecodingEnabled && mUseSpecDecoding;

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
    if (xqaParams.kv_cache_data_type == DATA_TYPE_INT8
        || (xqaParams.kv_cache_data_type == DATA_TYPE_E4M3 && mSM < kSM_90))
    {
        xqaParams.multi_block_mode = false;
    }

    xqaParams.output = generationsParams.context_buf;
    xqaParams.qkv = generationsParams.attention_input;
    xqaParams.cache_indir = generationsParams.cache_indir;
    xqaParams.kv_scale_orig_quant = generationsParams.kv_scale_orig_quant;
    xqaParams.kv_scale_quant_orig = generationsParams.kv_scale_quant_orig;
    xqaParams.host_past_key_value_lengths = generationsParams.host_past_key_value_lengths;
    xqaParams.host_context_lengths = generationsParams.host_context_lengths;
    xqaParams.semaphores = generationsParams.semaphores;
    xqaParams.workspaces = generationsParams.workspace;
    xqaParams.batch_size = generationsParams.num_requests;
    xqaParams.beam_width = generationsParams.beam_width;
    // Speculative decoding mode has generation input_length > 1.
    xqaParams.generation_input_length = generationsParams.input_seq_length;
    xqaParams.max_attention_window_size = generationsParams.max_attention_window;
    xqaParams.cyclic_attention_window_size = generationsParams.cyclic_attention_window_size;
    xqaParams.max_blocks_per_sequence = generationsParams.max_blocks_per_sequence;
    xqaParams.sink_token_length = generationsParams.sink_token_length;
    xqaParams.max_past_kv_length = generationsParams.max_past_kv_length;
    xqaParams.qkv_bias = generationsParams.qkv_bias;
    xqaParams.sequence_lengths = generationsParams.sequence_lengths;
    xqaParams.context_lengths = generationsParams.context_lengths;
    xqaParams.alibi_slopes = generationsParams.alibi_slopes;
    // Pre-computed rotary inv freq when building the engines.
    xqaParams.rotary_embedding_inv_freq_cache = generationsParams.rotary_inv_freq;
    if (!forConfigurePlugin)
    {
        // Speculative decoding (need to take new generated ids into consideration).
        TLLM_CHECK_WITH_INFO(
            !(mIsSpecDecodingEnabled && mUseSpecDecoding) || generationsParams.spec_decoding_packed_mask != nullptr,
            "Speculative decoding mode needs a valid packed_mask input tensor.");
    }
    xqaParams.spec_decoding_packed_mask = generationsParams.spec_decoding_packed_mask;
    xqaParams.spec_decoding_position_offsets = generationsParams.spec_decoding_position_offsets;
    xqaParams.spec_decoding_generation_lengths = generationsParams.spec_decoding_generation_lengths;
    xqaParams.spec_decoding_is_generation_length_variable
        = generationsParams.spec_decoding_is_generation_length_variable;
    xqaParams.spec_decoding_max_generation_length = generationsParams.spec_decoding_max_generation_length;
    xqaParams.mrope_position_deltas = generationsParams.mrope_position_deltas;

    xqaParams.logn_scaling_ptr = generationsParams.logn_scaling_ptr;
    xqaParams.total_num_input_tokens = generationsParams.total_num_input_tokens;
    xqaParams.is_fp8_output = mFP8ContextFMHA;
    xqaParams.fp8_out_scale = (mFP8ContextFMHA ? generationsParams.attention_output_orig_quant : nullptr);
    // Parameters required for FP4 output.
    xqaParams.output_sf = generationsParams.context_buf_sf;
    xqaParams.fp4_out_sf_scale = generationsParams.attention_output_sf_scale;
    xqaParams.start_token_idx_sf = generationsParams.start_token_idx_sf;

    return true;
}

template <typename T>
int AttentionOp::ulyssesGenerationPreprocess(
    int32_t batch_beam, T* mhaInput, T* mhaOutput, T*& input, cudaStream_t stream)
{
    if (mCpSize <= 1)
        return 0;

    auto const totalHeads = mNumHeads + 2 * mNumKVHeads;
    auto const partialHeads = totalHeads / mCpSize;
    auto const partialQHeads = mNumHeads / mCpSize;
    auto const partialKVHeads = mNumKVHeads / mCpSize;
    auto const partialTokenNum = (batch_beam + mCpSize - 1) / mCpSize;

    // attention_input shape: [partialTokenNum, numHeads, headSize]
    // view_1: [partialTokenNum, cpSize_Head, partialHeads, headSize]
    // transpose_1: [cpSize_Head, partialTokenNum, partialHeads, headSize]
    // all-to-all to get [cpSize_Length, partialTokenNum, partialHead, headSize]
    // view_2 as [tokens, partialHead, headSize]

    // do transpose_1
    // [1, mNumHeads + 2*mNumKVHeads, headSize]
    // -> (view) [1, cpSize * partialQHeads + cpSize * partialKVHeads + cpSize * partilKVHeads,
    // headSize]
    // -> (transpose) [cpSize, 1, partialQHeads + partialKvHeads + partialKVHeads, headSize]
    invokeCpTranspose(mhaOutput, mhaInput, input, partialTokenNum, mCpSize, partialQHeads, partialKVHeads, mHeadSize,
        mCpRank, stream);
    sync_check_cuda_error();

    // Do all to all
#if ENABLE_MULTI_DEVICE
    ncclGroupStart();
    for (int cpIdx = 0; cpIdx < mCpSize; cpIdx++)
    {
        if (cpIdx != mCpRank)
        {
            NCCLCHECK(ncclSend(mhaOutput + cpIdx * (partialTokenNum * getHeadSize() * partialHeads),
                (partialTokenNum * getHeadSize() * partialHeads), (*getDtypeMap())[mType], cpIdx, *mCpNcclComm,
                stream));
            NCCLCHECK(ncclRecv(mhaInput + cpIdx * (partialTokenNum * getHeadSize() * partialHeads),
                (partialTokenNum * getHeadSize() * partialHeads), (*getDtypeMap())[mType], cpIdx, *mCpNcclComm,
                stream));
        }
    }
    ncclGroupEnd();
    sync_check_cuda_error();
#endif // ENABLE_MULTI_DEVICE

    input = mhaInput;
    return 0;
}

template <typename T>
int AttentionOp::ulyssesGenerationPostprocess(
    int32_t batch_beam, T* mhaInput, T* mhaOutput, void* output, cudaStream_t stream)
{
    if (mCpSize <= 1)
        return 0;

    // mmha output shape: [tokens, partialHead, headSize]
    // view: [cpSize_Length, partialTokens, partialHead, headSize]
    // all-to-all: [cpSize_Head, partialTokens, partialHead, headSize]
    // transpose_1_reverse: [partialTokens, cpSize_Head, partialHead, headSize]
    // view: [partialTokens, head, headSize]

    auto partialHeads = mNumHeads / mCpSize;
    auto const partialTokenNum = (batch_beam + mCpSize - 1) / mCpSize;

    // do all-to-all
#if ENABLE_MULTI_DEVICE
    size_t const elementNum = partialTokenNum * getHeadSize() * partialHeads;
    ncclGroupStart();
    for (int cpIdx = 0; cpIdx < mCpSize; cpIdx++)
    {
        if (cpIdx != mCpRank)
        {
            if (mFP8ContextFMHA)
            {
                NCCLCHECK(ncclSend(reinterpret_cast<__nv_fp8_e4m3*>(mhaOutput) + cpIdx * elementNum, elementNum,
                    ncclInt8, cpIdx, *mCpNcclComm, stream));
                NCCLCHECK(ncclRecv(reinterpret_cast<__nv_fp8_e4m3*>(mhaInput) + cpIdx * elementNum, elementNum,
                    ncclInt8, cpIdx, *mCpNcclComm, stream));
            }
            else
            {
                NCCLCHECK(ncclSend(
                    mhaOutput + cpIdx * elementNum, elementNum, (*getDtypeMap())[mType], cpIdx, *mCpNcclComm, stream));
                NCCLCHECK(ncclRecv(
                    mhaInput + cpIdx * elementNum, elementNum, (*getDtypeMap())[mType], cpIdx, *mCpNcclComm, stream));
            }
        }
    }
    ncclGroupEnd();
#endif // ENABLE_MULTI_DEVICE

    // do transpose_1_reverse
    if (mFP8ContextFMHA)
    {
        invokeCpTransposeToSeqMajor<__nv_fp8_e4m3>(reinterpret_cast<__nv_fp8_e4m3*>(output),
            reinterpret_cast<__nv_fp8_e4m3 const*>(mhaOutput), reinterpret_cast<__nv_fp8_e4m3 const*>(mhaInput),
            partialTokenNum, mCpSize, partialHeads, getHeadSize(), mCpRank, stream);
    }
    else
    {
        invokeCpTransposeToSeqMajor<T>(
            (T*) output, mhaOutput, mhaInput, partialTokenNum, mCpSize, partialHeads, getHeadSize(), mCpRank, stream);
    }
    sync_check_cuda_error();
    return 0;
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
    params.max_attention_window_size = input_params.max_attention_window;
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
    sync_check_cuda_error();

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

int AttentionOp::getHeadSize(bool checkInit) const
{
    if (checkInit)
    {
        TLLM_CHECK_WITH_INFO(mHeadSize > 0, "Trying to read mHeadSize before it's been initialized");
    }
    return mHeadSize;
}

size_t AttentionOp::getWorkspaceSizeForContext(nvinfer1::DataType type, int32_t max_num_seq, int32_t input_seq_length,
    int32_t cross_kv_length, int32_t max_num_tokens) const noexcept
{
    int const local_hidden_units_qo = mNumHeads / mCpSize * getHeadSize();
    int const local_hidden_units_kv = mNumKVHeads / mCpSize * getHeadSize();

    auto const size = tensorrt_llm::runtime::BufferDataType(type).getSize();

    size_t context_workspace_size = 0;

    auto const batch_size = static_cast<size_t>(max_num_seq);
    auto const kv_seq_length = (isCrossAttention() ? cross_kv_length : input_seq_length);
    size_t const attention_mask_size = mEnableContextFMHA ? 0 : size * max_num_tokens * kv_seq_length;
    size_t const cu_seqlens_size = sizeof(int) * (batch_size + 1);
    size_t const rotary_inv_freq_size = sizeof(float) * batch_size * mRotaryEmbeddingDim / 2;

    size_t q_buf_2_size = 0;
    if (!mEnableContextFMHA)
    {
        // Unfused mha
        q_buf_2_size = size * batch_size * input_seq_length * local_hidden_units_qo;
    }
    else if (mFmhaDispatcher->isSeparateQAndKvInput())
    {
        // Paged context fmha
        q_buf_2_size = (mFP8ContextFMHA ? 1 : size) * max_num_tokens * local_hidden_units_qo;
    }

    size_t const k_buf_2_size = mEnableContextFMHA ? 0 : size * batch_size * kv_seq_length * local_hidden_units_kv;
    size_t const v_buf_2_size = mEnableContextFMHA ? 0 : size * batch_size * kv_seq_length * local_hidden_units_kv;
    size_t const qk_buf_size
        = mEnableContextFMHA ? 0 : size * batch_size * mNumHeads * input_seq_length * kv_seq_length;
    size_t const qkv_buf_2_size = mEnableContextFMHA ? 0 : size * max_num_tokens * local_hidden_units_qo;
    size_t const qk_buf_float_size
        = mEnableContextFMHA ? 0 : sizeof(float) * batch_size * mNumHeads * input_seq_length * kv_seq_length;
    size_t const fp8_qkv_buffer_size
        = mFP8ContextFMHA && mEnableContextFMHA && !mFmhaDispatcher->isSeparateQAndKvInput()
        ? max_num_tokens * size_t(local_hidden_units_qo + 2 * local_hidden_units_kv)
        : 0;
    size_t const padding_offset_size = mEnableContextFMHA ? 0 : sizeof(int) * max_num_tokens;
    size_t const encoder_padding_offset_size = mEnableContextFMHA ? 0 : sizeof(int) * max_num_tokens;
    // Each token holds (batch_idx, token_idx_in_seq) int2.
    size_t const tokens_info_size = sizeof(int2) * max_num_tokens;
    size_t const fmha_scheduler_counter = mEnableContextFMHA ? sizeof(uint32_t) : 0;
    size_t const fmha_bmm1_scale_size = mFP8ContextFMHA ? sizeof(float) * 2 : 0;
    size_t const fmha_bmm2_scale_size = mFP8ContextFMHA ? sizeof(float) : 0;

    // cp workspace size upper bound
    size_t const cpMaxPaddedSequenceLength = max_num_tokens + batch_size * (mCpSize - 1);
    size_t const cpWorkspaceSize = mCpSize == 1
        ? 0
        : (2 * size * cpMaxPaddedSequenceLength * getHeadSize() * (mNumHeads + 2 * mNumKVHeads) + cu_seqlens_size);

    int const NUM_BUFFERS = 20;
    size_t workspaces[NUM_BUFFERS];
    workspaces[0] = CUBLAS_WORKSPACE_SIZE;
    workspaces[1] = attention_mask_size;
    workspaces[2] = cu_seqlens_size; // cu_seqlen_q
    workspaces[3] = cu_seqlens_size; // cu_seqlen_kv
    workspaces[4] = cu_seqlens_size; // cu_mask_rows
    workspaces[5] = rotary_inv_freq_size;
    workspaces[6] = q_buf_2_size;
    workspaces[7] = k_buf_2_size;
    workspaces[8] = v_buf_2_size;
    workspaces[9] = qk_buf_size;
    workspaces[10] = qkv_buf_2_size;
    workspaces[11] = qk_buf_float_size;
    workspaces[12] = fp8_qkv_buffer_size;
    workspaces[13] = padding_offset_size;
    workspaces[14] = encoder_padding_offset_size;
    workspaces[15] = tokens_info_size;
    workspaces[16] = fmha_scheduler_counter;
    workspaces[17] = fmha_bmm1_scale_size;
    workspaces[18] = fmha_bmm2_scale_size;
    workspaces[19] = cpWorkspaceSize;
    context_workspace_size = tc::calculateTotalWorkspaceSize(workspaces, NUM_BUFFERS);

    return context_workspace_size;
}

size_t AttentionOp::getWorkspaceSizeForGeneration(
    nvinfer1::DataType type, int32_t max_num_seq, int32_t max_attention_window, int32_t max_num_tokens) const noexcept
{
    auto const size = tensorrt_llm::runtime::BufferDataType(type).getSize();

    size_t generation_workspace_size = 0;

    // MLA use fmha instead of XQA in generation
    if (mIsMLAEnabled)
    {
        size_t cu_seqlens_size = sizeof(int) * (max_num_seq + 1);
        size_t fmha_scheduler_counter = sizeof(uint32_t);
        size_t q_buffer_size = size * max_num_seq * mNumHeads * (mMLAParams.q_lora_rank + mMLAParams.qk_rope_head_dim);
        size_t o_buffer_size = size * max_num_seq * mNumHeads * mMLAParams.kv_lora_rank;
        size_t act_buffer_size = 0;
        size_t weight_buffer_size = 0;
        if (mIsFP8BlockScalingEnabled)
        {
            act_buffer_size
                = std::max(mGemmRunner->getActWorkspaceSize(max_num_seq,
                               mMLAParams.q_lora_rank + mMLAParams.kv_lora_rank + mMLAParams.qk_rope_head_dim),
                    mGemmRunner->getActWorkspaceSize(
                        max_num_seq, mNumHeads * (mMLAParams.kv_lora_rank + mMLAParams.qk_rope_head_dim)));
        }

        int const NUM_BUFFERS = 8;
        size_t workspaces[NUM_BUFFERS];
        workspaces[0] = CUBLAS_WORKSPACE_SIZE;
        workspaces[1] = cu_seqlens_size; // cu_q_len
        workspaces[2] = cu_seqlens_size; // cu_kv_len
        workspaces[3] = fmha_scheduler_counter;
        workspaces[4] = q_buffer_size;
        workspaces[5] = o_buffer_size;
        workspaces[6] = act_buffer_size;
        workspaces[7] = weight_buffer_size;
        generation_workspace_size = tc::calculateTotalWorkspaceSize(workspaces, NUM_BUFFERS);
        return generation_workspace_size;
    }

    int const batch_beam = max_num_seq;
    int32_t const maxSeqLenTile
        = std::max(getMaxNumSeqLenTile(batch_beam), (int) tc::divUp(mMultiProcessorCount, mNumHeads));

    size_t const partial_out_size = size * batch_beam * mNumHeads * mHeadSize * maxSeqLenTile;
    size_t const partial_sum_size = sizeof(float) * batch_beam * mNumHeads * maxSeqLenTile;
    size_t const partial_max_size = sizeof(float) * batch_beam * mNumHeads * maxSeqLenTile;
    size_t const shift_k_cache_size = (!mPosShiftEnabled || isCrossAttention())
        ? 0
        : size * batch_beam * mNumHeads * mHeadSize * max_attention_window;
    size_t const cpMaxPaddedSequenceLength = (batch_beam + mCpSize - 1) / mCpSize * mCpSize;
    size_t const cpWorkspaceSize
        = mCpSize == 1 ? 0 : (2 * size * cpMaxPaddedSequenceLength * getHeadSize() * (mNumHeads + 2 * mNumKVHeads));

    int const NUM_BUFFERS = 5;
    size_t workspaces[NUM_BUFFERS];
    workspaces[0] = partial_out_size;
    workspaces[1] = partial_sum_size;
    workspaces[2] = partial_max_size;
    workspaces[3] = shift_k_cache_size;
    workspaces[4] = cpWorkspaceSize;
    generation_workspace_size = tc::calculateTotalWorkspaceSize(workspaces, NUM_BUFFERS);

    size_t mqa_workspace_size = 0;
    if (mEnableXQA)
    {
        int const XQA_NUM_BUFFERS = 7;
        size_t mqa_workspaces[XQA_NUM_BUFFERS];
        size_t const cu_seqlens_size = sizeof(int) * (batch_beam + 1);
        size_t const cu_kv_seqlens_size = sizeof(int) * (batch_beam + 1);
        size_t const rotary_inv_freq_size = sizeof(float) * batch_beam * mRotaryEmbeddingDim / 2;
        mqa_workspaces[0] = cu_seqlens_size;
        mqa_workspaces[1] = cu_kv_seqlens_size;
        mqa_workspaces[2] = rotary_inv_freq_size;
        // The tokensInfo.
        mqa_workspaces[3] = max_num_tokens * sizeof(int2);
        // Scales used for trtllm-gen kernels.
        mqa_workspaces[4] = sizeof(float) * 2;
        mqa_workspaces[5] = sizeof(float);
        mqa_workspaces[6] = mXqaDispatcher->getWorkspaceSize(max_num_tokens);
        mqa_workspace_size
            = tc::calculateTotalWorkspaceSize(mqa_workspaces, XQA_NUM_BUFFERS, mXqaDispatcher->getWorkspaceAlignment());
    }

    return std::max(generation_workspace_size, mqa_workspace_size);
}

int AttentionOp::getMaxNumSeqLenTile(int batch_beam_size) const
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

template <typename T>
int AttentionOp::mlaPreContext(mlaParams<T>& params, cudaStream_t stream)
{
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

    auto rope_dim = mMLAParams.qk_rope_head_dim;
    auto c_q = mMLAParams.q_lora_rank;
    auto c_k = mMLAParams.kv_lora_rank;
    auto context_head_size = mMLAParams.qk_nope_head_dim + mMLAParams.qk_rope_head_dim;
    auto v_head_dim = mMLAParams.v_head_dim;

    if (!mIsFP8BlockScalingEnabled)
    {
        {
            auto transa = CUBLAS_OP_T;
            auto transb = CUBLAS_OP_N;
            int m = params.head_num * context_head_size;
            int n = params.acc_q_len;
            int k = c_q;
            int lda = k;
            int ldb = c_q + c_k + rope_dim;
            int ldc = params.head_num
                * (2 * context_head_size + v_head_dim); // output shape: [(b * s), (3 * h * (d_h + rope)]
            mCublasWrapper->createDescriptors(transa, transb, m, n, k, lda, ldb, ldc);
            mCublasWrapper->Gemm(transa, transb, m, n, k, params.q_b_proj, lda, params.fused_a_input, ldb,
                params.attention_input_buf, ldc);
            mCublasWrapper->destroyDescriptors();
        }

        {
            auto transa = CUBLAS_OP_T;
            auto transb = CUBLAS_OP_N;
            int m = mMLAParams.qk_nope_head_dim;
            int n = params.acc_q_len;
            int k = c_k;
            int lda = k;
            int ldb = c_q + c_k + rope_dim;
            int ldc = params.head_num * (2 * context_head_size + v_head_dim);
            mCublasWrapper->createDescriptors(transa, transb, m, n, k, lda, ldb, ldc);
            mCublasWrapper->stridedBatchedGemm(transa, transb, m, n, k, params.kv_b_proj, lda,
                mMLAParams.qk_nope_head_dim * c_k, params.fused_a_input + c_q, ldb, 0,
                params.attention_input_buf + static_cast<size_t>(params.head_num) * context_head_size,
                params.head_num * (context_head_size * 2 + v_head_dim), context_head_size, params.head_num, 1.0f, 0.0);

            mCublasWrapper->Gemm(transa, transb, params.head_num * m, n, k,
                params.kv_b_proj + mMLAParams.qk_nope_head_dim * c_k * params.head_num, lda, params.fused_a_input + c_q,
                ldb, params.attention_input_buf + 2 * static_cast<size_t>(params.head_num) * context_head_size, ldc);
            mCublasWrapper->destroyDescriptors();
        }
    }
    else
    {
        int8_t* workspace_byte_ptr = reinterpret_cast<int8_t*>(params.workspace);
        size_t offset = CUBLAS_WORKSPACE_SIZE;

        size_t act_size = mGemmRunner->getFP8DataSize(params.acc_q_len, c_q + c_k + rope_dim, true);
        size_t act_scale_size = mGemmRunner->getActScaleSize(params.acc_q_len, c_q + c_k + rope_dim);
        __nv_fp8_e4m3* act_buffer
            = reinterpret_cast<__nv_fp8_e4m3*>(nextWorkspacePtr(workspace_byte_ptr, offset, act_size));
        float* act_scale_buffer
            = reinterpret_cast<float*>(nextWorkspacePtr(workspace_byte_ptr, offset, act_scale_size));
        size_t act_offset = offset;

        mGemmRunner->fp8CS1x128(act_buffer, act_scale_buffer,
            reinterpret_cast<__nv_bfloat16 const*>(params.fused_a_input), c_q + c_k + rope_dim, params.acc_q_len,
            stream);
        sync_check_cuda_error();

        // mat_a: [params.acc_q_len, c_q], mat_b: [c_q, params.head_num * context_head_size]
        {
            int m = params.head_num * context_head_size;
            int n = params.acc_q_len;
            int k = c_q;
            int lda = k, ldb = c_q + c_k + rope_dim;
            int ldc = params.head_num
                * (2 * context_head_size + v_head_dim); // output shape: [(b * s), (3 * h * (d_h + rope)]

            __nv_fp8_e4m3 const* q_gemm_weight_ptr = reinterpret_cast<__nv_fp8_e4m3 const*>(params.q_b_proj);
            float const* q_gemm_scale_ptr = reinterpret_cast<float const*>(params.q_b_scale);

            mGemmRunner->gemm(act_buffer, ldb, const_cast<__nv_fp8_e4m3*>(q_gemm_weight_ptr), lda,
                reinterpret_cast<__nv_bfloat16*>(params.attention_input_buf), ldc, n, m, k, act_scale_buffer,
                const_cast<float*>(q_gemm_scale_ptr), stream);
            sync_check_cuda_error();
        }

        {
            offset = act_offset;
            __nv_fp8_e4m3 const* kv_gemm_weight_ptr = reinterpret_cast<__nv_fp8_e4m3 const*>(params.kv_b_proj);
            float const* kv_gemm_scale_ptr = reinterpret_cast<float const*>(params.kv_b_scale);

            int m = mMLAParams.qk_nope_head_dim;
            int n = params.acc_q_len;
            int k = c_k;

            int shape_n_4_align = ((params.acc_q_len - 1) / 4 + 1) * 4;
            int lda = k, ldb = c_q + c_k + rope_dim;
            int ldc = params.head_num * (2 * context_head_size + v_head_dim);

            mGemmRunner->strideBatchGemm(reinterpret_cast<__nv_bfloat16*>(params.attention_input_buf)
                    + static_cast<size_t>(params.head_num) * context_head_size,
                ldc, context_head_size, act_buffer + c_q, ldb, 0, const_cast<__nv_fp8_e4m3*>(kv_gemm_weight_ptr), lda,
                m * k, params.head_num, shape_n_4_align, m, k, stream, act_scale_buffer + shape_n_4_align * (c_q / 128),
                0, const_cast<float*>(kv_gemm_scale_ptr));
            sync_check_cuda_error();
            mGemmRunner->gemm(act_buffer + c_q, ldb,
                const_cast<__nv_fp8_e4m3*>(kv_gemm_weight_ptr) + params.head_num * m * k, lda,
                reinterpret_cast<__nv_bfloat16*>(
                    params.attention_input_buf + 2 * static_cast<size_t>(params.head_num) * context_head_size),
                ldc, n, params.head_num * m, k, act_scale_buffer + shape_n_4_align * (c_q / 128),
                const_cast<float*>(kv_gemm_scale_ptr) + params.head_num * (k / 128) * (m / 128), stream);
            sync_check_cuda_error();
            mCublasWrapper->destroyDescriptors();
        }
    }

    return 0;
}

template <typename T>
int AttentionOp::mlaGeneration(
    mlaParams<T>& params, EnqueueGenerationParams<T> const& generation_params, cudaStream_t stream)
{
    int const num_kv_heads = 1;
    int const head_size = mMLAParams.kv_lora_rank + mMLAParams.qk_rope_head_dim;
    int32_t const batch_beam = generation_params.beam_width * generation_params.num_requests;

    auto const sizePerToken = num_kv_heads * head_size * sizeof(T);

    auto kv_cache_buffer = KVBlockArray(batch_beam, generation_params.max_blocks_per_sequence, mTokensPerBlock,
        sizePerToken, generation_params.cyclic_attention_window_size,
        generation_params.max_cyclic_attention_window_size, generation_params.sink_token_length,
        generation_params.can_use_one_more_block, generation_params.host_primary_pool_pointer,
        generation_params.host_secondary_pool_pointer, generation_params.block_offsets);

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

    // Workspace pointer shift
    auto const size = sizeof(T);
    int8_t* workspace_byte_ptr = reinterpret_cast<int8_t*>(params.workspace);
    size_t offset = CUBLAS_WORKSPACE_SIZE;

    size_t const cu_seqlens_size = sizeof(int) * (params.batch_size + 1);
    size_t const fmha_scheduler_counter = sizeof(uint32_t);
    size_t q_buffer_size = size * batch_beam * mNumHeads * (mMLAParams.qk_nope_head_dim + mMLAParams.qk_rope_head_dim);
    size_t o_buffer_size = size * batch_beam * mNumHeads * mMLAParams.kv_lora_rank;
    int* cu_q_seqlens = reinterpret_cast<int*>(nextWorkspacePtr(workspace_byte_ptr, offset, cu_seqlens_size));
    int* cu_kv_seqlens = reinterpret_cast<int*>(nextWorkspacePtr(workspace_byte_ptr, offset, cu_seqlens_size));
    uint32_t* fmha_tile_counter_ptr
        = reinterpret_cast<uint32_t*>(nextWorkspacePtr(workspace_byte_ptr, offset, fmha_scheduler_counter));
    T* q_buffer = reinterpret_cast<T*>(nextWorkspacePtr(workspace_byte_ptr, offset, q_buffer_size));
    T* o_buffer = reinterpret_cast<T*>(nextWorkspacePtr(workspace_byte_ptr, offset, o_buffer_size));
    size_t quant_offset = offset;

    if (!mIsFP8BlockScalingEnabled)
    {
        {
            auto transa = CUBLAS_OP_T;
            auto transb = CUBLAS_OP_N;
            int m = params.head_num * (mMLAParams.qk_nope_head_dim + mMLAParams.qk_rope_head_dim);
            int n = params.acc_q_len;
            int k = mMLAParams.q_lora_rank;
            int lda = k;
            int ldb = mMLAParams.q_lora_rank + mMLAParams.kv_lora_rank + mMLAParams.qk_rope_head_dim;
            int ldc = m;
            mCublasWrapper->createDescriptors(transa, transb, m, n, k, lda, ldb, ldc);
            mCublasWrapper->Gemm(
                transa, transb, m, n, k, params.q_b_proj, lda, params.fused_a_input, ldb, q_buffer, ldc);
            mCublasWrapper->destroyDescriptors();
        }

        {
            auto transa = CUBLAS_OP_T;
            auto transb = CUBLAS_OP_N;
            int m = mMLAParams.kv_lora_rank;
            int n = params.acc_q_len;
            int k = mMLAParams.qk_nope_head_dim;
            int lda = k, ldb = params.head_num * (mMLAParams.qk_nope_head_dim + mMLAParams.qk_rope_head_dim);
            int ldc = params.head_num * (mMLAParams.kv_lora_rank + mMLAParams.qk_rope_head_dim);
            mCublasWrapper->createDescriptors(transa, transb, m, n, k, lda, ldb, ldc);
            mCublasWrapper->stridedBatchedGemm(transa, transb, m, n, k, params.k_b_proj_trans, lda,
                mMLAParams.qk_nope_head_dim * mMLAParams.kv_lora_rank, q_buffer, ldb,
                mMLAParams.qk_nope_head_dim + mMLAParams.qk_rope_head_dim, params.attention_input_buf, ldc,
                mMLAParams.kv_lora_rank + mMLAParams.qk_rope_head_dim, params.head_num, 1.0f, 0.0);
            mCublasWrapper->destroyDescriptors();
        }
    }
    else
    {
        {
            int m = params.head_num * (mMLAParams.qk_nope_head_dim + mMLAParams.qk_rope_head_dim);
            int n = params.acc_q_len;
            int k = mMLAParams.q_lora_rank;
            int lda = k, ldb = mMLAParams.q_lora_rank + mMLAParams.kv_lora_rank + mMLAParams.qk_rope_head_dim;
            int ldc = m;
            size_t act_size = mGemmRunner->getFP8DataSize(n, ldb, true);
            size_t act_scale_size = mGemmRunner->getActScaleSize(n, ldb);
            __nv_fp8_e4m3* act_buffer
                = reinterpret_cast<__nv_fp8_e4m3*>(nextWorkspacePtr(workspace_byte_ptr, offset, act_size));
            float* act_scale_buffer
                = reinterpret_cast<float*>(nextWorkspacePtr(workspace_byte_ptr, offset, act_scale_size));

            __nv_fp8_e4m3 const* q_gemm_weight_ptr = reinterpret_cast<__nv_fp8_e4m3 const*>(params.q_b_proj);
            float const* q_gemm_scale_ptr = reinterpret_cast<float const*>(params.q_b_scale);

            mGemmRunner->fp8CS1x128(act_buffer, act_scale_buffer,
                reinterpret_cast<__nv_bfloat16 const*>(params.fused_a_input), ldb, n, stream);
            mGemmRunner->gemm(act_buffer, ldb, const_cast<__nv_fp8_e4m3*>(q_gemm_weight_ptr), lda,
                reinterpret_cast<__nv_bfloat16*>(q_buffer), ldc, n, m, k, act_scale_buffer,
                const_cast<float*>(q_gemm_scale_ptr), stream);
        }
        {
            offset = quant_offset;
            int m = mMLAParams.kv_lora_rank;
            int n = params.acc_q_len;
            int k = mMLAParams.qk_nope_head_dim;
            int lda = k, ldb = mMLAParams.qk_nope_head_dim;
            int ldc = params.head_num * (mMLAParams.kv_lora_rank + mMLAParams.qk_rope_head_dim);
            size_t act_size = mGemmRunner->getFP8DataSize(n, params.head_num * mMLAParams.qk_nope_head_dim, true);
            size_t act_scale_size = mGemmRunner->getActScaleSize(n, params.head_num * mMLAParams.qk_nope_head_dim);
            __nv_fp8_e4m3* act_buffer
                = reinterpret_cast<__nv_fp8_e4m3*>(nextWorkspacePtr(workspace_byte_ptr, offset, act_size));
            float* act_scale_buffer
                = reinterpret_cast<float*>(nextWorkspacePtr(workspace_byte_ptr, offset, act_scale_size));
            __nv_fp8_e4m3 const* k_gemm_weight_ptr = reinterpret_cast<__nv_fp8_e4m3 const*>(params.k_b_proj_trans);
            float const* k_gemm_scale_ptr = reinterpret_cast<float const*>(params.k_b_trans_scale);

            int shape_n_4_align = ((n - 1) / 4 + 1) * 4;
            mGemmRunner->fp8CS1x128Reshape(act_buffer, act_scale_buffer,
                reinterpret_cast<__nv_bfloat16 const*>(q_buffer), mMLAParams.qk_nope_head_dim, params.head_num, n,
                mMLAParams.qk_nope_head_dim + mMLAParams.qk_rope_head_dim, stream);

            mGemmRunner->strideBatchGemm(reinterpret_cast<__nv_bfloat16*>(params.attention_input_buf), ldc,
                mMLAParams.kv_lora_rank + mMLAParams.qk_rope_head_dim, act_buffer, ldb, n * k,
                const_cast<__nv_fp8_e4m3*>(k_gemm_weight_ptr), lda, m * k, params.head_num, n, m, k, stream,
                act_scale_buffer, shape_n_4_align * (mMLAParams.qk_nope_head_dim / 128),
                const_cast<float*>(k_gemm_scale_ptr));
        }
    }

    params.seqQOffset = cu_q_seqlens;
    params.cu_kv_seqlens = cu_kv_seqlens;
    params.fmha_tile_counter = fmha_tile_counter_ptr;
    params.q_buf = q_buffer;

    invokeMLARopeGeneration<T>(params, kv_cache_buffer, stream);

    if (generation_params.runtime_perf_knobs)
    {
        int64_t enable_context_fmha_fp32_acc_val = generation_params.runtime_perf_knobs[1];
        mFMHAForceFP32Acc = mFMHAForceFP32Acc || enable_context_fmha_fp32_acc_val == 1;
    }

    MHARunnerParams fmhaParams{};
    fmhaParams.b = batch_beam;
    fmhaParams.qSeqLen = params.head_num;
    fmhaParams.kvSeqLen = generation_params.max_past_kv_length;
    // Disable sliding window attention when it is not needed.
    fmhaParams.slidingWindowSize = generation_params.cyclic_attention_window_size;
    fmhaParams.totalQSeqLen = batch_beam * params.head_num;
    // TODO: set it correctly for contiguous kv buffer (cross-attention).
    // fmhaParams.totalKvSeqLen = params.num_tokens;
    // Device buffer pointers.
    // fmhaParams.qkvPtr = reinterpret_cast<void const*>(params.attention_input);
    fmhaParams.qPtr = reinterpret_cast<void const*>(params.attention_input_buf);
    // TODO: add contiguous kv buffer (cross-attention).
    fmhaParams.kvPtr = nullptr;
    fmhaParams.outputPtr = o_buffer;
    // fmhaParams.packedMaskPtr = params.fmha_custom_mask;
    fmhaParams.pagedKvCache = kv_cache_buffer;
    fmhaParams.cuQSeqLenPtr = cu_q_seqlens;
    fmhaParams.cuKvSeqLenPtr = cu_kv_seqlens;
    fmhaParams.cuMaskRowsPtr = nullptr; // mla not support custorm mask right now
    fmhaParams.tileCounterPtr = fmha_tile_counter_ptr;
    fmhaParams.scaleBmm1Ptr = nullptr;
    fmhaParams.scaleBmm2Ptr = nullptr;
    fmhaParams.stream = stream;
    fmhaParams.forceFp32Acc = mFMHAForceFP32Acc;

    // Run the fmha kernel
    mDecoderFMHARunner->run(fmhaParams);
    if (!mIsFP8BlockScalingEnabled)
    {
        {
            auto transa = CUBLAS_OP_T;
            auto transb = CUBLAS_OP_N;
            int m = mMLAParams.v_head_dim;
            int n = params.batch_size;
            int k = mMLAParams.kv_lora_rank;
            int lda = k;
            int ldb = k;
            int ldc = m;
            mCublasWrapper->createDescriptors(transa, transb, m, n, k, lda, ldb, ldc);

            mCublasWrapper->stridedBatchedGemm(transa, transb, m, n, k,
                params.kv_b_proj + params.head_num * mMLAParams.kv_lora_rank * mMLAParams.qk_nope_head_dim, lda,
                mMLAParams.v_head_dim * mMLAParams.kv_lora_rank, o_buffer, ldb * params.head_num, ldb,
                params.context_buf, ldc * params.head_num, ldc, params.head_num, 1.0f, 0.0);
            mCublasWrapper->destroyDescriptors();
        }
    }
    else
    {
        offset = quant_offset;
        int m = mMLAParams.v_head_dim;
        int n = params.batch_size;
        int k = mMLAParams.kv_lora_rank;
        int h = params.head_num;
        int lda = k, ldb = k;
        int ldc = h * m;
        size_t act_size = mGemmRunner->getFP8DataSize(n, h * k, true);
        size_t act_scale_size = mGemmRunner->getActScaleSize(n, h * k);
        __nv_fp8_e4m3* act_buffer
            = reinterpret_cast<__nv_fp8_e4m3*>(nextWorkspacePtr(workspace_byte_ptr, offset, act_size));
        float* act_scale_buffer
            = reinterpret_cast<float*>(nextWorkspacePtr(workspace_byte_ptr, offset, act_scale_size));
        __nv_fp8_e4m3 const* v_gemm_weight_ptr = reinterpret_cast<__nv_fp8_e4m3 const*>(params.kv_b_proj) + h * k * m;
        float const* v_gemm_scale_ptr = reinterpret_cast<float const*>(params.kv_b_scale) + h * (k / 128) * (m / 128);
        int shape_n_4_align = ((n - 1) / 4 + 1) * 4;
        mGemmRunner->fp8CS1x128Reshape(
            act_buffer, act_scale_buffer, reinterpret_cast<__nv_bfloat16 const*>(o_buffer), k, h, n, k, stream);
        mGemmRunner->strideBatchGemm(reinterpret_cast<__nv_bfloat16*>(params.context_buf), ldc, m, act_buffer, ldb,
            n * k, const_cast<__nv_fp8_e4m3*>(v_gemm_weight_ptr), lda, m * k, params.head_num, n, m, k, stream,
            act_scale_buffer, shape_n_4_align * (k / 128), const_cast<float*>(v_gemm_scale_ptr));
    }

    sync_check_cuda_error();
    return 0;
}

#define MLA_FUNC_DEFINE(T)                                                                                             \
    template int AttentionOp::mlaPreContext<T>(mlaParams<T> & params, cudaStream_t stream);                            \
    template int AttentionOp::mlaGeneration<T>(                                                                        \
        mlaParams<T> & params, EnqueueGenerationParams<T> const& generation_params, cudaStream_t stream);

MLA_FUNC_DEFINE(float)
MLA_FUNC_DEFINE(half)
#ifdef ENABLE_BF16
MLA_FUNC_DEFINE(__nv_bfloat16)
#endif

template <typename T, typename KVCacheBuffer>
int AttentionOp::enqueueContext(EnqueueContextParams<T> const& params, cudaStream_t stream)
{
    int const headSize = getHeadSize();

    int const local_hidden_units_qo = mNumHeads * headSize;
    int const local_hidden_units_kv = mNumKVHeads / mCpSize * headSize;
    PositionEmbeddingType const position_embedding_type = mPositionEmbeddingType;
    float const q_scaling = mQScaling;

    KVCacheBuffer kv_cache_buffer;
    auto const elemSize = mKVCacheQuantMode.hasKvCacheQuant() ? sizeof(int8_t) : sizeof(T);
    auto sizePerToken = mNumKVHeads / mCpSize * headSize * elemSize;
    if (useKVCache())
    {
        if constexpr (std::is_same_v<KVCacheBuffer, KVBlockArray>)
        {
            kv_cache_buffer = KVBlockArray(params.batch_size, params.max_blocks_per_sequence, mTokensPerBlock,
                sizePerToken, params.cyclic_attention_window_size, params.max_cyclic_attention_window_size,
                params.sink_token_length, params.can_use_one_more_block, params.host_primary_pool_pointer,
                params.host_secondary_pool_pointer, params.block_offsets);
        }
        else if constexpr (std::is_same_v<KVCacheBuffer, KVLinearBuffer>)
        {
            using BufferDataType = typename KVCacheBuffer::DataType;
            kv_cache_buffer = KVLinearBuffer(params.batch_size,
                isCrossAttention() ? params.cross_kv_length : params.max_attention_window, sizePerToken,
                params.cyclic_attention_window_size, params.sink_token_length, false,
                reinterpret_cast<BufferDataType*>(params.key_value_cache));
        }
    }

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

    size_t const kv_seq_length = (isCrossAttention() ? params.cross_kv_length : params.input_seq_length);
    size_t const attention_mask_size
        = mEnableContextFMHA ? 0 : sizeof(T) * params.batch_size * params.input_seq_length * kv_seq_length;
    size_t const cu_seqlens_size = sizeof(int) * (params.batch_size + 1);
    size_t const rotary_inv_freq_size = sizeof(float) * params.batch_size * mRotaryEmbeddingDim / 2;
    size_t q_buf_2_size = 0;
    if (!mEnableContextFMHA)
    {
        // Unfused mha
        q_buf_2_size = sizeof(T) * params.batch_size * params.input_seq_length * local_hidden_units_qo;
    }
    else if (mFmhaDispatcher->isSeparateQAndKvInput())
    {
        // Paged context fmha
        q_buf_2_size = (mFP8ContextFMHA ? 1 : sizeof(T)) * params.num_tokens * local_hidden_units_qo;
    }

    size_t const k_buf_2_size
        = mEnableContextFMHA ? 0 : sizeof(T) * params.batch_size * kv_seq_length * local_hidden_units_kv;
    size_t const v_buf_2_size
        = mEnableContextFMHA ? 0 : sizeof(T) * params.batch_size * kv_seq_length * local_hidden_units_kv;
    size_t const qk_buf_size
        = mEnableContextFMHA ? 0 : sizeof(T) * params.batch_size * mNumHeads * params.input_seq_length * kv_seq_length;
    size_t const qkv_buf_2_size
        = mEnableContextFMHA ? 0 : sizeof(T) * params.batch_size * params.input_seq_length * local_hidden_units_qo;
    size_t const qk_buf_float_size = mEnableContextFMHA
        ? 0
        : sizeof(float) * params.batch_size * mNumHeads * params.input_seq_length * kv_seq_length;
    size_t const fp8_qkv_buffer_size
        = mEnableContextFMHA && mFP8ContextFMHA && !mFmhaDispatcher->isSeparateQAndKvInput()
        ? params.num_tokens * (local_hidden_units_qo + 2 * local_hidden_units_kv)
        : 0;
    size_t const padding_offset_size
        = mEnableContextFMHA ? 0 : sizeof(int) * params.batch_size * params.input_seq_length;
    size_t const encoder_padding_offset_size
        = mEnableContextFMHA ? 0 : sizeof(int) * params.batch_size * params.cross_kv_length;
    // Each token holds (batch_idx, token_idx_in_seq) int2.
    size_t const tokens_info_size = sizeof(int2) * params.num_tokens;
    size_t const fmha_scheduler_counter = mEnableContextFMHA ? sizeof(uint32_t) : 0;
    size_t const fmha_bmm1_scale_size = mFP8ContextFMHA ? sizeof(float) * 2 : 0;
    size_t const fmha_bmm2_scale_size = mFP8ContextFMHA ? sizeof(float) : 0;

    // cp workspace size upper bound
    size_t const cpMaxPadedSequenceLength = params.num_tokens + params.batch_size * (mCpSize - 1);
    size_t const cpWorkspaceSize
        = mCpSize == 1 ? 0 : 2 * sizeof(T) * cpMaxPadedSequenceLength * getHeadSize() * (mNumHeads + 2 * mNumKVHeads);

    bool const is_qk_buf_float_ = true;

    // Workspace pointer shift
    int8_t* workspace_byte_ptr = reinterpret_cast<int8_t*>(params.workspace);
    size_t offset = CUBLAS_WORKSPACE_SIZE;

    T* attention_mask = reinterpret_cast<T*>(nextWorkspacePtr(workspace_byte_ptr, offset, attention_mask_size));
    int* cu_q_seqlens = reinterpret_cast<int*>(nextWorkspacePtr(workspace_byte_ptr, offset, cu_seqlens_size));
    int* cu_kv_seqlens = reinterpret_cast<int*>(nextWorkspacePtr(workspace_byte_ptr, offset, cu_seqlens_size));
    int* cu_mask_rows = reinterpret_cast<int*>(nextWorkspacePtr(workspace_byte_ptr, offset, cu_seqlens_size));
    float* rotary_inv_freq_buf
        = reinterpret_cast<float*>(nextWorkspacePtr(workspace_byte_ptr, offset, rotary_inv_freq_size));
    T* q_buf_2_ = reinterpret_cast<T*>(nextWorkspacePtr(workspace_byte_ptr, offset, q_buf_2_size));
    T* k_buf_2_ = reinterpret_cast<T*>(nextWorkspacePtr(workspace_byte_ptr, offset, k_buf_2_size));
    T* v_buf_2_ = reinterpret_cast<T*>(nextWorkspacePtr(workspace_byte_ptr, offset, v_buf_2_size));
    T* qk_buf_ = reinterpret_cast<T*>(nextWorkspacePtr(workspace_byte_ptr, offset, qk_buf_size));
    T* qkv_buf_2_ = reinterpret_cast<T*>(nextWorkspacePtr(workspace_byte_ptr, offset, qkv_buf_2_size));
    float* qk_buf_float_ = reinterpret_cast<float*>(nextWorkspacePtr(workspace_byte_ptr, offset, qk_buf_float_size));
    __nv_fp8_e4m3* fp8_qkv_buffer
        = reinterpret_cast<__nv_fp8_e4m3*>(nextWorkspacePtr(workspace_byte_ptr, offset, fp8_qkv_buffer_size));
    int* padding_offset = mEnableContextFMHA
        ? nullptr
        : reinterpret_cast<int*>(nextWorkspacePtr(workspace_byte_ptr, offset, padding_offset_size));
    int* encoder_padding_offset = (mEnableContextFMHA && !isCrossAttention())
        ? nullptr
        : reinterpret_cast<int*>(nextWorkspacePtr(workspace_byte_ptr, offset, encoder_padding_offset_size));
    int2* tokens_info = reinterpret_cast<int2*>(nextWorkspacePtr(workspace_byte_ptr, offset, tokens_info_size));
    uint32_t* fmha_tile_counter_ptr
        = reinterpret_cast<uint32_t*>(nextWorkspacePtr(workspace_byte_ptr, offset, fmha_scheduler_counter));
    float* fmha_bmm1_scale_ptr
        = reinterpret_cast<float*>(nextWorkspacePtr(workspace_byte_ptr, offset, fmha_bmm1_scale_size));
    float* fmha_bmm2_scale_ptr
        = reinterpret_cast<float*>(nextWorkspacePtr(workspace_byte_ptr, offset, fmha_bmm2_scale_size));

    T* gatherInBuffer = reinterpret_cast<T*>(nextWorkspacePtr(workspace_byte_ptr, offset, cpWorkspaceSize));
    T* gatherOutBuffer = gatherInBuffer + cpMaxPadedSequenceLength * getHeadSize() * (mNumHeads + 2 * mNumKVHeads);
    int* cu_cp_partial_seqlens = reinterpret_cast<int*>(
        gatherOutBuffer + cpMaxPadedSequenceLength * getHeadSize() * (mNumHeads + 2 * mNumKVHeads));

    // build attention_mask, cu_seqlens, and padding_offset tensors
    // Note: self attn and cross attn should use different params
    // cross attn's seqlen info is from encoder input lengths, not decoder input lengths!
    // moreover, attn mask for cross attn should be set separately (see below)
    BuildDecoderInfoParams<T> decoder_params{};
    decoder_params.seqQOffsets = cu_q_seqlens;
    decoder_params.seqKVOffsets = cu_kv_seqlens;
    decoder_params.seqCpPartialOffsets = cu_cp_partial_seqlens;
    decoder_params.cpSize = mCpSize;
    decoder_params.packedMaskRowOffsets = cu_mask_rows;
    decoder_params.paddingOffsets = padding_offset;
    decoder_params.tokensInfo = tokens_info;
    decoder_params.encoderPaddingOffsets
        = isCrossAttention() ? encoder_padding_offset : nullptr; // cross attention takes offsets from encoder inputs
    decoder_params.attentionMask = isCrossAttention() ? nullptr : attention_mask; // manually set for unfused cross attn
    // Fixed sequence length offset if not removing the padding (cu_q_seqlens[i] = i * seq_length).
    decoder_params.seqQLengths = params.q_seq_lengths;
    decoder_params.seqKVLengths = isCrossAttention() ? params.encoder_input_lengths : params.kv_seq_lengths;
    decoder_params.batchSize = params.batch_size;
    decoder_params.maxQSeqLength = params.input_seq_length;
    decoder_params.maxEncoderQSeqLength
        = isCrossAttention() ? params.cross_kv_length : 0; // cross attention uses encoder seq length
    decoder_params.attentionWindowSize = params.cyclic_attention_window_size;
    decoder_params.sinkTokenLength = params.sink_token_length;
    decoder_params.numTokens = params.num_tokens;
    decoder_params.removePadding = mRemovePadding;
    decoder_params.attentionMaskType = mMaskType;
    decoder_params.blockSparseParams = mBlockSparseParams;
    decoder_params.fmhaTileCounter = fmha_tile_counter_ptr;
    decoder_params.quantScaleO = params.attention_output_orig_quant;
    decoder_params.dequantScaleQ = params.kv_scale_quant_orig;
    decoder_params.dequantScaleKv = params.kv_scale_quant_orig;
    decoder_params.fmhaHostBmm1Scale = 1.0f / (sqrtf(getHeadSize() * 1.0f) * q_scaling);
    decoder_params.fmhaBmm1Scale = fmha_bmm1_scale_ptr;
    decoder_params.fmhaBmm2Scale = fmha_bmm2_scale_ptr;
    // Rotary embedding inv_freq buffer.
    decoder_params.rotaryEmbeddingScale = mRotaryEmbeddingScale;
    decoder_params.rotaryEmbeddingBase = mRotaryEmbeddingBase;
    decoder_params.rotaryEmbeddingDim = mRotaryEmbeddingDim;
    decoder_params.rotaryScalingType = mRotaryEmbeddingScaleType;
    // The inv freq might be updated during runtime with dynamic scaling type.
    decoder_params.rotaryEmbeddingInvFreq = rotary_inv_freq_buf;
    // This is pre-computed when building the engines.
    decoder_params.rotaryEmbeddingInvFreqCache = params.rotary_inv_freq;
    decoder_params.rotaryEmbeddingMaxPositions = mRotaryEmbeddingMaxPositions;

    invokeBuildDecoderInfo(decoder_params, stream);
    sync_check_cuda_error();

    // In cross attention context phase, the attention mask should be a matrix of all ones.
    // We reassign attention_mask to override what previous invokeBuildDecoderInfo() does
    // also, invokeBuildDecoderInfo can only handle square mask, not cross B x q_len x kv_len mask
    // TODO: put this logic in the kernel above. currently not much concern because q_len is mostly = 1
    if (isUnfusedCrossAttention())
    {
        {
            std::vector<T> h_attention_mask(params.batch_size * params.input_seq_length * params.cross_kv_length, 1.);
            std::vector<int32_t> h_encoder_input_lengths(params.batch_size);
            cudaMemcpyAsync(h_encoder_input_lengths.data(), params.encoder_input_lengths,
                sizeof(int32_t) * params.batch_size, cudaMemcpyDeviceToHost, stream);
            sync_check_cuda_error();

            for (int bi = 0; bi < params.batch_size; bi++)
            {
                int b_offset = bi * params.input_seq_length * params.cross_kv_length;
                for (int qi = 0; qi < params.input_seq_length; qi++)
                {
                    int q_offset = b_offset + qi * params.cross_kv_length;
                    if (h_encoder_input_lengths[bi] < params.cross_kv_length)
                    {
                        std::fill(h_attention_mask.begin() + q_offset + h_encoder_input_lengths[bi],
                            h_attention_mask.begin() + q_offset + params.cross_kv_length, 0.f);
                    }
                }
            }
            cudaMemcpyAsync(attention_mask, h_attention_mask.data(),
                sizeof(T) * params.batch_size * params.cross_kv_length * params.input_seq_length,
                cudaMemcpyHostToDevice, stream);
            sync_check_cuda_error();
        }
    }

    // FIXME: a temporary solution to make sure the padding part is 0.
    if (!mRemovePadding)
    {
        cudaMemsetAsync(params.context_buf, 0, params.num_tokens * local_hidden_units_qo * sizeof(T), stream);
        sync_check_cuda_error();
    }

    KvCacheDataType const cache_type = mKVCacheQuantMode.hasInt8KvCache()
        ? KvCacheDataType::INT8
        : (mKVCacheQuantMode.hasFp8KvCache() ? KvCacheDataType::FP8 : KvCacheDataType::BASE);

    cudaDataType_t const gemm_data_type = tc::CudaDataType<T>::value;
    int const attention_seq_len_1 = params.input_seq_length;                                               // q length
    int const attention_seq_len_2 = isCrossAttention() ? params.cross_kv_length : params.input_seq_length; // kv length

    // If the model has relative attentiona bias, q scaling should be applied in QK gemm stage and use 1 in
    // softamax stage (because to get softmax[scale(Q*K) + rel pos bias] here, q_scaling can't be applied during
    // softmax phase by qk_scale); otherwise, use 1 in gemm stage and apply scaling in softmax stage
    float const qk_scale
        = 1.0f / (sqrtf(getHeadSize() * 1.0f) * q_scaling); // q_scaling in denominator. by default q_scaling =1.0f
    float const qk_scale_gemm = isRelativePosition() ? qk_scale : 1.0f;
    T const qk_scale_softmax = static_cast<T>(isRelativePosition() ? 1.0f : qk_scale);

    // in context phase, currently FMHA runner has two restrictions:
    // 1. only apply to self attention. If want fused multi-head cross attention, FMHCA kernels and runner is needed
    // 2. doesn't apply to MHA with relative attention bias, i.e. softmax(QK + bias) * V
    // We update mEnableContextFMHA in constructor to check these conditions
    if (mEnableContextFMHA)
    {
        // do all-to-all for params.attention_input, need to split on kv head
        // [token_num // cp_size, kv_heads, head_size] -> [token_num, kv_heads // cp_size, head_size]
        T* attention_input = const_cast<T*>(params.attention_input);
        if (mCpSize > 1)
        {
            int32_t partialTokenNum = 0;
            int32_t maxPartialLength = 0;
            for (size_t batchIdx = 0; batchIdx < params.batch_size; ++batchIdx)
            {
                int32_t partialLength = (params.host_context_lengths[batchIdx] + mCpSize - 1) / mCpSize;
                maxPartialLength = std::max(maxPartialLength, partialLength);
                partialTokenNum += partialLength;
            }
            auto const totalHeads = mNumHeads + 2 * mNumKVHeads;
            auto const partialHeads = totalHeads / mCpSize;
            auto const partialQHeads = mNumHeads / mCpSize;
            auto const partialKVHeads = mNumKVHeads / mCpSize;

            // full request: [bs, seqlen, head, headSize]
            //
            // input of cp: [bs, partialLength, head, headSize]
            // view_1 as [bs, partialLength, cpSize_Head, partialHead, headSize]
            // transpose_1 as [cpSize_Head, bs, partialLenth, partialHead, headSize]
            // all-to-all to get [cpSize_Length, bs, partialLength, partialHead, headSize]
            // transpose_2 to [bs, cpSize_Length, partialLength, partialHead, headSize]
            // view_2 as [bs, totalLength, partialHead, headSize]
            // and this is same to the input under TP.
            //
            // when we use remove_input_padding, bs and length are fused into numTokens. So, we need to
            // insert the cpSize_Length dimension of transpose_2 into numTokens directly like
            // input of cp: [partialNumTokens, head, headSize]
            // view_1 as [partialNumTokens, cpSize_Head, partialHead, headSize]
            // transpose_1 as [cpSize_Head, partialNumTokens, partialHead, headSize]
            // all-to-all to get [cpSize_Length, partialNumTokens, partialHead, headSize]
            // transpose_2 as [NumTokens, partialHead, headSize]
            // and this is same to the input under TP.

            // view_1 + transpose_1
            invokeCpTranspose(gatherInBuffer, gatherOutBuffer, params.attention_input, partialTokenNum, mCpSize,
                partialQHeads, partialKVHeads, getHeadSize(), mCpRank, stream);
            sync_check_cuda_error();

            // Do all to all
#if ENABLE_MULTI_DEVICE
            ncclGroupStart();
            for (int cpIdx = 0; cpIdx < mCpSize; cpIdx++)
            {
                if (cpIdx != mCpRank)
                {
                    NCCLCHECK(ncclSend(gatherInBuffer + cpIdx * (partialTokenNum * getHeadSize() * partialHeads),
                        (partialTokenNum * getHeadSize() * partialHeads), (*getDtypeMap())[mType], cpIdx, *mCpNcclComm,
                        stream));
                    NCCLCHECK(ncclRecv(gatherOutBuffer + cpIdx * (partialTokenNum * getHeadSize() * partialHeads),
                        (partialTokenNum * getHeadSize() * partialHeads), (*getDtypeMap())[mType], cpIdx, *mCpNcclComm,
                        stream));
                }
            }
            ncclGroupEnd();
            sync_check_cuda_error();
#endif // ENABLE_MULTI_DEVICE

            // transpose_2 + view_2
            invokeCpTranspose2(gatherInBuffer, gatherOutBuffer, params.q_seq_lengths, cu_q_seqlens,
                cu_cp_partial_seqlens, mCpSize, maxPartialLength, params.batch_size, partialHeads, getHeadSize(),
                stream);

            attention_input = gatherInBuffer;
        }
        sync_check_cuda_error();

        bool const enablePagedKVContextFMHA = mPagedKVCache && mPagedContextFMHA;
        TLLM_CHECK_WITH_INFO(!(mKVCacheQuantMode.hasInt8KvCache() && enablePagedKVContextFMHA),
            "Paged Context FMHA doesn't work with int8 kv cache currently.");
        TLLM_CHECK_WITH_INFO(!(params.sink_token_length > 0 && enablePagedKVContextFMHA),
            "Cannot support StreamingLLM now when enabling paged KV context FMHA.");

        // The max_kv_seq_len comes from the encoder seqlen when cross attention is used.
        int const max_kv_seq_len = isCrossAttention() ? params.cross_kv_length : params.max_past_kv_len;

        // Prepare QKV preprocessing parameters.
        QKVPreprocessingParams<T, KVCacheBuffer> preprocessingParams;

        // Buffers.
        preprocessingParams.qkv_input = const_cast<T*>(attention_input);
        preprocessingParams.cross_kv_input = const_cast<T*>(params.cross_kv);
        preprocessingParams.quantized_qkv_output = fp8_qkv_buffer;
        preprocessingParams.q_output = q_buf_2_;
        preprocessingParams.kv_cache_buffer = kv_cache_buffer;
        preprocessingParams.qkv_bias = params.qkv_bias;
        preprocessingParams.tokens_info = decoder_params.tokensInfo;
        preprocessingParams.seq_lens = params.q_seq_lengths;
        // Indicate if chunked-context is used (i.e. q_seqlen > kv_seqlen).
        preprocessingParams.cache_seq_lens = params.kv_seq_lengths;
        preprocessingParams.encoder_seq_lens = params.encoder_input_lengths;
        preprocessingParams.cu_seq_lens = cu_q_seqlens;
        // Cross-attention only.
        preprocessingParams.cu_kv_seq_lens = cu_kv_seqlens;
        preprocessingParams.rotary_embedding_inv_freq = rotary_inv_freq_buf;
        preprocessingParams.rotary_coef_cache_buffer = params.rotary_cos_sin;
        preprocessingParams.mrope_rotary_cos_sin = params.mrope_rotary_cos_sin;
        preprocessingParams.kvScaleOrigQuant = params.kv_scale_orig_quant;
        preprocessingParams.spec_decoding_position_offsets = nullptr;
        preprocessingParams.logn_scaling = params.logn_scaling_ptr;

        // Scalars
        preprocessingParams.batch_size = params.batch_size;
        preprocessingParams.max_input_seq_len = params.input_seq_length;
        preprocessingParams.max_kv_seq_len = max_kv_seq_len;
        preprocessingParams.cyclic_kv_cache_len
            = isCrossAttention() ? params.cross_kv_length : params.cyclic_attention_window_size;
        preprocessingParams.sink_token_len = params.sink_token_length;
        preprocessingParams.token_num = params.num_tokens;
        preprocessingParams.remove_padding = mRemovePadding;
        preprocessingParams.cross_attention = isCrossAttention();
        preprocessingParams.head_num = mNumHeads / mCpSize;
        preprocessingParams.kv_head_num = mNumKVHeads / mCpSize;
        preprocessingParams.qheads_per_kv_head = mNumHeads / mNumKVHeads;
        preprocessingParams.size_per_head = getHeadSize();
        preprocessingParams.rotary_embedding_dim = mRotaryEmbeddingDim;
        preprocessingParams.rotary_embedding_base = mRotaryEmbeddingBase;
        preprocessingParams.rotary_scale_type = mRotaryEmbeddingScaleType;
        preprocessingParams.rotary_embedding_scale = mRotaryEmbeddingScale;
        preprocessingParams.rotary_embedding_max_positions = mRotaryEmbeddingMaxPositions;
        preprocessingParams.position_embedding_type = position_embedding_type;
        preprocessingParams.position_shift_enabled = mPosShiftEnabled;
        preprocessingParams.cache_type = cache_type;
        preprocessingParams.separate_q_kv_output = enablePagedKVContextFMHA || isCrossAttention();
        preprocessingParams.quantized_fp8_output = mFP8ContextFMHA;
        preprocessingParams.generation_phase = false;
        preprocessingParams.multi_processor_count = mMultiProcessorCount;

        preprocessingParams.rotary_vision_start = mVisionStart;
        preprocessingParams.rotary_vision_length = mVisionLength;

        {
            std::string const beforeRopeStr = "ctx attention before RoPE at layer " + std::to_string(mLayerIdx);
            TLLM_CHECK_DEBUG_WITH_INFO(tensorrt_llm::runtime::utils::tensorHasInvalid(params.num_tokens,
                                           (local_hidden_units_qo + 2 * local_hidden_units_kv), mType,
                                           const_cast<T*>(attention_input), stream, beforeRopeStr)
                    == false,
                "Found invalid number (NaN or Inf) in " + beforeRopeStr);
        }
        if (mIsMLAEnabled)
        {
            params.mla_param->cu_q_seqlens = cu_q_seqlens;
            invokeMLARopeContext<T, KVCacheBuffer>(*params.mla_param, kv_cache_buffer, stream);
        }
        else
        {
            invokeQKVPreprocessing(preprocessingParams, stream);
        }
        sync_check_cuda_error();
        {
            std::string const afterRopeStr = "ctx attention after RoPE at layer " + std::to_string(mLayerIdx);
            TLLM_CHECK_DEBUG_WITH_INFO(tensorrt_llm::runtime::utils::tensorHasInvalid(params.num_tokens,
                                           (local_hidden_units_qo + 2 * local_hidden_units_kv), mType,
                                           const_cast<T*>(attention_input), stream, afterRopeStr)
                    == false,
                "Found invalid number (NaN or Inf) in " + afterRopeStr);
            sync_check_cuda_error();
        }

        if (params.runtime_perf_knobs)
        {
            int64_t enable_context_fmha_fp32_acc_val = params.runtime_perf_knobs[1];
            mFMHAForceFP32Acc = mFMHAForceFP32Acc || enable_context_fmha_fp32_acc_val == 1;
        }

        // Unified FMHA runner interface for both packed QKV FMHA, contiguous Q_KV and paged KV FMHA.
        // Page KV input layout:
        //    - q_ptr: [B, S, H, D], which supports variable sequence length
        //    - paged_kv_cache: paged kv buffer
        //    - cu_q_seqlens: the cumulative query sequence lengths, needed for variable sequence length.
        //    - cu_kv_seqlens: the cumulative kv sequence lengths, needed for variable sequence length.
        //
        // Contiguous KV input layout:
        //    - q_ptr: [B, S, H, D], which supports variable sequence length
        //    - kv_ptr: [B, S, 2, H, D], which supports variable sequence length
        //    - cu_q_seqlens: the cumulative query sequence lengths, needed for variable sequence length.
        //    - cu_kv_seqlens: the cumulative kv sequence lengths, needed for variable sequence length.

        // Construct the fmha params for running kernels.
        MHARunnerParams fmhaParams{};
        fmhaParams.b = params.batch_size;
        fmhaParams.qSeqLen = params.input_seq_length;
        fmhaParams.kvSeqLen = max_kv_seq_len;
        // Disable sliding window attention when it is not needed.
        fmhaParams.slidingWindowSize
            = (mDenseContextFMHA || isCrossAttention()) ? max_kv_seq_len : params.cyclic_attention_window_size;
        fmhaParams.totalQSeqLen = params.num_tokens;
        // TODO: set it correctly for contiguous kv buffer (cross-attention).
        fmhaParams.totalKvSeqLen = params.num_tokens;
        // Device buffer pointers.
        fmhaParams.qkvPtr = mFP8ContextFMHA ? reinterpret_cast<void const*>(fp8_qkv_buffer)
                                            : reinterpret_cast<void const*>(attention_input);
        fmhaParams.qPtr = reinterpret_cast<void const*>(q_buf_2_);
        // TODO: add contiguous kv buffer (cross-attention).
        fmhaParams.kvPtr = nullptr;
        fmhaParams.outputPtr
            = mCpSize > 1 ? gatherOutBuffer : params.context_buf; // only use [totalLength, h / cpSize, Dh]
        fmhaParams.outputSfPtr = params.context_buf_sf;
        fmhaParams.packedMaskPtr = params.attention_packed_mask;
        if constexpr (std::is_same_v<KVCacheBuffer, KVBlockArray>)
        {
            fmhaParams.pagedKvCache = kv_cache_buffer;
        }
        fmhaParams.cuQSeqLenPtr = cu_q_seqlens;
        fmhaParams.cuKvSeqLenPtr = cu_kv_seqlens;
        fmhaParams.cuMaskRowsPtr = cu_mask_rows;
        fmhaParams.tileCounterPtr = fmha_tile_counter_ptr;
        fmhaParams.scaleBmm1Ptr = fmha_bmm1_scale_ptr;
        fmhaParams.scaleBmm2Ptr = fmha_bmm2_scale_ptr;
        fmhaParams.oSfScalePtr = params.attention_output_sf_scale;
        fmhaParams.stream = stream;
        fmhaParams.forceFp32Acc = mFMHAForceFP32Acc;

        // Run the fmha kernel.
        mFmhaDispatcher->run(fmhaParams);
        sync_check_cuda_error();

        // The kv cache might need to be updated after FMHA (only when sliding window attention + chunked context is
        // used together). Reuse the preprocessingParams.
        invokeKvCachePostprocessing(preprocessingParams, stream);
        sync_check_cuda_error();

        if (mCpSize > 1)
        {
            // After FMHA, we get result [numTokens(bs, cp, paritalLength), partialHead, headSize]
            // transpose_2_reverse: [cpSize_Length, partialTokens(bs, partialLength), partialHead, headSize]
            // all-to-all: [cpSize_Head, partialTokens, partialHead, headSize]
            // transpose_1_reverse: [partialTokens, cpSize_Head, partialHead, headSize]
            // view: [partialTokens, head, headSize]

            int32_t maxPartialLength = 0;
            int32_t partialTokenNum = 0;
            for (size_t batchIdx = 0; batchIdx < params.batch_size; ++batchIdx)
            {
                int32_t partialLength = (params.host_context_lengths[batchIdx] + mCpSize - 1) / mCpSize;
                maxPartialLength = std::max(maxPartialLength, partialLength);
                partialTokenNum += partialLength;
            }
            auto partialHeads = mNumHeads / mCpSize;

            // transpose_2_reverse
            if (mFP8ContextFMHA)
            {
                invokeCpTransposeToSeqMajor2(reinterpret_cast<__nv_fp8_e4m3*>(gatherInBuffer),
                    reinterpret_cast<__nv_fp8_e4m3 const*>(gatherOutBuffer), params.q_seq_lengths, cu_q_seqlens,
                    cu_cp_partial_seqlens, mCpSize, maxPartialLength, params.batch_size, partialHeads, getHeadSize(),
                    stream);
            }
            else
            {
                invokeCpTransposeToSeqMajor2(gatherInBuffer, gatherOutBuffer, params.q_seq_lengths, cu_q_seqlens,
                    cu_cp_partial_seqlens, mCpSize, maxPartialLength, params.batch_size, partialHeads, getHeadSize(),
                    stream);
            }

            // all-to-all
#if ENABLE_MULTI_DEVICE
            size_t const elementNum = partialTokenNum * getHeadSize() * partialHeads;
            ncclGroupStart();
            for (int cpIdx = 0; cpIdx < mCpSize; cpIdx++)
            {
                if (cpIdx != mCpRank)
                {
                    if (mFP8ContextFMHA)
                    {
                        NCCLCHECK(ncclSend(reinterpret_cast<__nv_fp8_e4m3*>(gatherInBuffer) + cpIdx * elementNum,
                            elementNum, ncclInt8, cpIdx, *mCpNcclComm, stream));
                        NCCLCHECK(ncclRecv(reinterpret_cast<__nv_fp8_e4m3*>(gatherOutBuffer) + cpIdx * elementNum,
                            elementNum, ncclInt8, cpIdx, *mCpNcclComm, stream));
                    }
                    else
                    {
                        NCCLCHECK(ncclSend(gatherInBuffer + cpIdx * elementNum, elementNum, (*getDtypeMap())[mType],
                            cpIdx, *mCpNcclComm, stream));
                        NCCLCHECK(ncclRecv(gatherOutBuffer + cpIdx * elementNum, elementNum, (*getDtypeMap())[mType],
                            cpIdx, *mCpNcclComm, stream));
                    }
                }
            }
            ncclGroupEnd();
#endif // ENABLE_MULTI_DEVICE

            // transpose_1_reverse + view
            if (mFP8ContextFMHA)
            {
                invokeCpTransposeToSeqMajor<__nv_fp8_e4m3>(reinterpret_cast<__nv_fp8_e4m3*>(params.context_buf),
                    reinterpret_cast<__nv_fp8_e4m3 const*>(gatherInBuffer),
                    reinterpret_cast<__nv_fp8_e4m3 const*>(gatherOutBuffer), partialTokenNum, mCpSize, partialHeads,
                    getHeadSize(), mCpRank, stream);
            }
            else
            {
                invokeCpTransposeToSeqMajor<T>((T*) params.context_buf, gatherInBuffer, gatherOutBuffer,
                    partialTokenNum, mCpSize, partialHeads, getHeadSize(), mCpRank, stream);
            }
            sync_check_cuda_error();
        }
    }
    else
    {
        TLLM_CHECK_DEBUG_WITH_INFO(params.logn_scaling_ptr == nullptr, "Unfused MHA does not support logn scaling");
        // FIXME: a temporary solution to make sure the padding part of key/value buffer is 0
        // NOTE: pointer subtraction is used below since there could be some extra gap due to alignment.
        //  Otherwise, we could do cudaMemsetAsync(k_buf_2_, 0, k_buf_2_size + v_buf_2_size, stream);
        // cudaMemsetAsync(k_buf_2_, 0, reinterpret_cast<int8_t*>(qk_buf_) - reinterpret_cast<int8_t*>(k_buf_2_),
        // stream);
        cudaMemsetAsync(k_buf_2_, 0,
            reinterpret_cast<int8_t*>(v_buf_2_) - reinterpret_cast<int8_t*>(k_buf_2_) + v_buf_2_size, stream);

        if (!isCrossAttention())
        {
            // self attention, write to from QKV to Q/K/V
            invokeAddFusedQKVBiasTranspose(q_buf_2_, k_buf_2_, v_buf_2_, const_cast<T*>(params.attention_input),
                const_cast<T*>(params.qkv_bias), params.q_seq_lengths, mRemovePadding ? padding_offset : nullptr,
                params.batch_size, params.input_seq_length, params.num_tokens, mNumHeads, mNumKVHeads, getHeadSize(),
                mRotaryEmbeddingDim, mRotaryEmbeddingBase, mRotaryEmbeddingScaleType, mRotaryEmbeddingScale,
                mRotaryEmbeddingMaxPositions, position_embedding_type, (float*) nullptr, 0, stream);
            sync_check_cuda_error();
        }
        else
        {
            // cross attention, write from self QKV [*, head_num * head_size + 2 * kv_head_num * head_size]to Q, write
            // from cross KV [*, 2 * kv_head_num * head_size] to K/V kernel modified accordingly to handle nullptr
            // buffer
            invokeAddFusedQKVBiasTranspose(q_buf_2_, (T*) nullptr, (T*) nullptr, const_cast<T*>(params.attention_input),
                const_cast<T*>(params.qkv_bias), params.q_seq_lengths, mRemovePadding ? padding_offset : nullptr,
                params.batch_size, params.input_seq_length, params.num_tokens, mNumHeads, mNumKVHeads, getHeadSize(),
                mRotaryEmbeddingDim, mRotaryEmbeddingBase, mRotaryEmbeddingScaleType, mRotaryEmbeddingScale,
                mRotaryEmbeddingMaxPositions, position_embedding_type, (float*) nullptr, 0, stream);
            sync_check_cuda_error();

            invokeAddFusedQKVBiasTranspose((T*) nullptr, k_buf_2_, v_buf_2_, const_cast<T*>(params.cross_kv),
                const_cast<T*>(params.qkv_bias), params.encoder_input_lengths,
                mRemovePadding ? encoder_padding_offset : nullptr, params.batch_size, params.cross_kv_length,
                params.num_encoder_tokens, /*mNumHeads*/ 0, mNumKVHeads, getHeadSize(), mRotaryEmbeddingDim,
                mRotaryEmbeddingBase, mRotaryEmbeddingScaleType, mRotaryEmbeddingScale, mRotaryEmbeddingMaxPositions,
                position_embedding_type, (float*) nullptr, 0, stream);
            sync_check_cuda_error();
        }

        // write KV to cache
        if (useKVCache())
        {
            invokeTranspose4dBatchMajor(k_buf_2_, v_buf_2_, kv_cache_buffer, params.batch_size,
                isCrossAttention() ? params.cross_kv_length : params.input_seq_length,
                isCrossAttention() ? params.cross_kv_length : params.cyclic_attention_window_size, getHeadSize(),
                mNumKVHeads, cache_type, params.kv_scale_orig_quant,
                isCrossAttention() ? params.encoder_input_lengths : params.q_seq_lengths, stream);
        }
        sync_check_cuda_error();

        T const* linear_bias_slopes = isALiBi() ? params.alibi_slopes : nullptr;
        T const* relative_attention_bias = isRelativePosition() ? params.relative_attention_bias : nullptr;
        int const relative_attention_bias_stride = isRelativePosition() ? params.relative_attention_bias_stride : 0;
        int const max_distance = mMaxDistance;
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
            int const num_qheads_per_kv_head = mNumHeads / mNumKVHeads;
            for (int ki = 0; ki < mNumKVHeads; ++ki)
            {
                T* qptr = q_buf_2_ + (ki * num_qheads_per_kv_head * attention_seq_len_1 * getHeadSize());
                T* kptr = k_buf_2_ + (ki * attention_seq_len_2 * getHeadSize());
                int const qk_offset = ki * attention_seq_len_1 * num_qheads_per_kv_head * attention_seq_len_2;
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
                    isCrossAttention() ? params.cross_kv_length : params.cyclic_attention_window_size, stream,
                    max_distance > 0, relative_attention_bias_stride, max_distance, false /* bidirectional */);
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
            param.attn_logit_softcapping_scale = mAttnLogitSoftcappingScale;
            param.attn_logit_softcapping_inverse_scale = 1.0f / mAttnLogitSoftcappingScale;
            param.linear_bias_slopes = const_cast<T*>(linear_bias_slopes); // (head_num,), optional
            param.block_sparse_attn = mMaskType == AttentionMaskType::BLOCKSPARSE;
            param.block_sparse_params = mBlockSparseParams;
            param.q_seq_lengths = params.q_seq_lengths;
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
                    isCrossAttention() ? params.cross_kv_length : params.cyclic_attention_window_size, stream,
                    max_distance > 0, relative_attention_bias_stride, max_distance, false /* bidirectional */);
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
            param.attn_logit_softcapping_scale = mAttnLogitSoftcappingScale;
            param.attn_logit_softcapping_inverse_scale = 1.0f / mAttnLogitSoftcappingScale;
            param.linear_bias_slopes = const_cast<T*>(linear_bias_slopes); // (head_num,), optional
            param.block_sparse_attn = mMaskType == AttentionMaskType::BLOCKSPARSE;
            param.block_sparse_params = mBlockSparseParams;
            param.q_seq_lengths = params.q_seq_lengths;
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
            int const num_qheads_per_kv_head = mNumHeads / mNumKVHeads;
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
            invokeTransposeQKV(static_cast<T*>(params.context_buf), qkv_buf_2_, params.batch_size, attention_seq_len_1,
                mNumHeads, getHeadSize(), (float*) nullptr, 0, stream);
        }
        else
        {
            invokeTransposeAttentionOutRemovePadding(qkv_buf_2_, static_cast<T*>(params.context_buf), params.num_tokens,
                params.batch_size, attention_seq_len_1, mNumHeads, getHeadSize(), padding_offset, (float*) nullptr, 0,
                stream);
        }
    }
    return 0;
}

template int AttentionOp::enqueueContext<half, KVLinearBuffer>(
    EnqueueContextParams<half> const& params, cudaStream_t stream);

template int AttentionOp::enqueueContext<float, KVLinearBuffer>(
    EnqueueContextParams<float> const& params, cudaStream_t stream);

#ifdef ENABLE_BF16
template int AttentionOp::enqueueContext<__nv_bfloat16, KVLinearBuffer>(
    EnqueueContextParams<__nv_bfloat16> const& params, cudaStream_t stream);
#endif

template int AttentionOp::enqueueContext<half, KVBlockArray>(
    EnqueueContextParams<half> const& params, cudaStream_t stream);

template int AttentionOp::enqueueContext<float, KVBlockArray>(
    EnqueueContextParams<float> const& params, cudaStream_t stream);

#ifdef ENABLE_BF16
template int AttentionOp::enqueueContext<__nv_bfloat16, KVBlockArray>(
    EnqueueContextParams<__nv_bfloat16> const& params, cudaStream_t stream);
#endif

bool AttentionOp::mForceMultiBlockWarned = false;

template <typename T, typename KVCacheBuffer>
int AttentionOp::enqueueGeneration(EnqueueGenerationParams<T> const& params, cudaStream_t stream)
{
    int const headSize = getHeadSize();
    float const q_scaling = mQScaling;
    float const* logn_scaling_ptr = isLognScaling() ? params.logn_scaling_ptr : nullptr;
    T const* relative_attention_bias = isRelativePosition() ? params.relative_attention_bias : nullptr;
    int const relative_attention_bias_stride = isRelativePosition() ? params.relative_attention_bias_stride : 0;
    int const max_distance = mMaxDistance;
    bool const* finished = nullptr;

    auto const quant_option = tc::QuantMode::fromDescription();
    float const* qkv_scale_out = nullptr;

    int const* ia3_tasks = nullptr;
    T const* ia3_key_weights = nullptr;
    T const* ia3_value_weights = nullptr;

    int32_t const batch_beam = params.beam_width * params.num_requests;

    KVCacheBuffer kv_cache_buffer;
    auto const elemSize = mKVCacheQuantMode.hasKvCacheQuant() ? sizeof(int8_t) : sizeof(T);
    auto const sizePerToken = mNumKVHeads / mCpSize * headSize * elemSize;
    if (useKVCache())
    {
        if constexpr (std::is_same_v<KVCacheBuffer, KVBlockArray>)
        {
            using BufferDataType = typename KVCacheBuffer::DataType;
            kv_cache_buffer = KVBlockArray(batch_beam, params.max_blocks_per_sequence, mTokensPerBlock, sizePerToken,
                params.cyclic_attention_window_size, params.max_cyclic_attention_window_size, params.sink_token_length,
                params.can_use_one_more_block, params.host_primary_pool_pointer, params.host_secondary_pool_pointer,
                reinterpret_cast<BufferDataType*>(params.block_offsets));
        }
        else if constexpr (std::is_same_v<KVCacheBuffer, KVLinearBuffer>)
        {
            using BufferDataType = typename KVCacheBuffer::DataType;
            kv_cache_buffer = KVLinearBuffer(batch_beam, params.max_attention_window, sizePerToken,
                params.cyclic_attention_window_size, params.sink_token_length, false,
                reinterpret_cast<BufferDataType*>(params.key_value_cache));
        }
    }
    sync_check_cuda_error();

#ifndef NDEBUG
    debugCheckSemaphores(stream);
#endif

    // Medusa doesn't support multi-block mode.
    if (!(mIsSpecDecodingEnabled && mUseSpecDecoding))
    {
        if (params.runtime_perf_knobs)
        {
            int64_t multi_block_mode_val = params.runtime_perf_knobs[0];
            mMultiBlockMode = multi_block_mode_val == 1;
            if (common::getEnvForceDeterministicAttention())
            {
                mMultiBlockMode = false;
            }
        }

        if (common::getEnvForceDeterministicAttention())
        {
            mMultiBlockMode = false;
        }

        // TODO only for debug usage
        if (!mMultiBlockMode)
        {
            char* isForceMultiBlockModeChar = std::getenv("FORCE_MULTI_BLOCK_MODE");
            bool isForceMultiBlockMode
                = (isForceMultiBlockModeChar != nullptr && std::string(isForceMultiBlockModeChar) == "ON");
            TLLM_CHECK_WITH_INFO(!(common::getEnvForceDeterministicAttention() && isForceMultiBlockMode),
                "FORCE_MULTI_BLOCK_MODE and FORCE_DETERMINISTIC/FORCE_ATTENTION_KERNEL_DETERMINISTIC can not be set at "
                "the same time.");
            mMultiBlockMode = isForceMultiBlockMode;
        }
    }

    int8_t* workspace_byte_ptr = reinterpret_cast<int8_t*>(params.workspace);
    size_t offset = 0;
    size_t const cpMaxPaddedSequenceLength = (batch_beam + mCpSize - 1) / mCpSize * mCpSize;
    size_t const cpWorkspaceSize
        = mCpSize == 1 ? 0 : 2 * sizeof(T) * cpMaxPaddedSequenceLength * (mNumHeads + 2 * mNumKVHeads) * mHeadSize;
    T* mhaOutput = reinterpret_cast<T*>(nextWorkspacePtr(workspace_byte_ptr, offset, cpWorkspaceSize));
    T* mhaInput = mhaOutput + cpMaxPaddedSequenceLength * (mNumHeads + 2 * mNumKVHeads) * mHeadSize;

    T* attention_input = const_cast<T*>(params.attention_input);
    this->template ulyssesGenerationPreprocess<T>(batch_beam, mhaInput, mhaOutput, attention_input, stream);

    // Try XQA optimization first.
    {
        // NOTE: input_seq_length = num_medusa_tokens + 1 (new generated one from the original LM head)
        // self attn
        XQAParams xqaParams{};
        this->template convertMMHAParamsToXQAParams<T, KVCacheBuffer>(xqaParams, params, /*forConfigurePlugin=*/false);
        if (mEnableXQA && mXqaDispatcher->shouldUse(xqaParams))
        {
            TLLM_LOG_DEBUG("XQA kernels are selected in the generation phase.");
            xqaParams.stream = stream;
            if (mCpSize > 1)
            {
                xqaParams.output = mhaOutput;
                xqaParams.qkv = attention_input;
            }
            mXqaDispatcher->run(xqaParams, kv_cache_buffer);
            this->template ulyssesGenerationPostprocess<T>(batch_beam, mhaInput, mhaOutput, params.context_buf, stream);
            return 0;
        }
        else if (mIsSpecDecodingEnabled && mUseSpecDecoding)
        {
            TLLM_CHECK_WITH_INFO(false, "No available XQA kernels are found for speculative decoding mode.");
        }
        else if (mFuseFp4Quant)
        {
            TLLM_CHECK_WITH_INFO(false, "No available kernels are found for FP4 output.");
        }
    }

    // This is the number of kv tokens that q needs to visit, but excluding one as it will be processed before the kv
    // loop.
    int timestep = params.max_past_kv_length;
    int const max_timesteps = std::min(timestep, params.cyclic_attention_window_size);
    int estimated_min_multi_block_count
        = estimate_min_multi_block_count<T>(max_timesteps, mMaxSharedMemoryPerBlockOptin - 2048);

    if (!mMultiBlockMode && !mForceMultiBlockWarned && estimated_min_multi_block_count > 1)
    {
        mForceMultiBlockWarned = true;
        TLLM_LOG_WARNING(
            "Force using MultiBlockMode in MMHA as shared memory is not enough, "
            "MultiBlockMode may have different accuracy compared to non-MultiBlockMode.");
    }

    // estimate min block count to satisfy shared memory requirement to run kernel.
    // Runtime check to see the actual number of blocks per sequence we need.
    int32_t const max_num_seq_len_tiles = std::max(getMaxNumSeqLenTile(batch_beam), estimated_min_multi_block_count);
    int32_t const min_num_seq_len_tiles = std::max(1, estimated_min_multi_block_count);
    bool const enable_multi_block
        = (mMultiBlockMode && max_num_seq_len_tiles > 1) || estimated_min_multi_block_count > 1;
    size_t const partial_out_size
        = enable_multi_block ? sizeof(T) * batch_beam * mNumHeads * mHeadSize * max_num_seq_len_tiles : 0;
    size_t const partial_sum_size
        = enable_multi_block ? sizeof(float) * batch_beam * mNumHeads * max_num_seq_len_tiles : 0;
    size_t const partial_max_size
        = enable_multi_block ? sizeof(float) * batch_beam * mNumHeads * max_num_seq_len_tiles : 0;
    size_t const shift_k_cache_size = (!mPosShiftEnabled || isCrossAttention())
        ? 0
        : sizeof(T) * batch_beam * mNumHeads * mHeadSize * params.max_attention_window;

    // Workspace pointer shift
    T* partial_out = reinterpret_cast<T*>(nextWorkspacePtr(workspace_byte_ptr, offset, partial_out_size));
    float* partial_sum = reinterpret_cast<float*>(nextWorkspacePtr(workspace_byte_ptr, offset, partial_sum_size));
    float* partial_max = reinterpret_cast<float*>(nextWorkspacePtr(workspace_byte_ptr, offset, partial_max_size));
    T* shift_k_cache = reinterpret_cast<T*>(nextWorkspacePtr(workspace_byte_ptr, offset, shift_k_cache_size));

    // Apply position embedding to the keys in the K cache
    KVLinearBuffer shift_k_cache_buffer;
    if (useKVCache() && mPosShiftEnabled && !isCrossAttention())
    {
        shift_k_cache_buffer
            = KVLinearBuffer(batch_beam, params.max_attention_window, sizePerToken, params.cyclic_attention_window_size,
                params.sink_token_length, true, reinterpret_cast<int8_t*>(shift_k_cache));
        sync_check_cuda_error();
        // KV cache type
        KvCacheDataType const kv_cache_type = KvCacheDataType::BASE;
        using DataType = typename SATypeConverter<T>::Type;
        invokeShiftKCache<DataType, KVCacheBuffer>(kv_cache_buffer, shift_k_cache_buffer, kv_cache_type, getHeadSize(),
            timestep, batch_beam, mNumKVHeads, params.beam_width, params.cyclic_attention_window_size,
            params.sink_token_length, params.kv_scale_quant_orig, params.sequence_lengths, params.context_lengths,
            mRotaryEmbeddingDim, mRotaryEmbeddingBase, mRotaryEmbeddingScaleType, mRotaryEmbeddingScale,
            mRotaryEmbeddingMaxPositions, mPositionEmbeddingType, stream);
    }

    FusedQKVMaskedAttentionDispatchParams<T, KVCacheBuffer> dispatch_params{};
    dispatch_params.mUnfuseQkvGemm = mUnfuseQkvGemm;
    dispatch_params.qkv_buf = attention_input;
    dispatch_params.qkv_bias = params.qkv_bias;
    dispatch_params.logn_scaling_ptr = logn_scaling_ptr;
    dispatch_params.relative_attention_bias = relative_attention_bias;
    dispatch_params.relative_attention_bias_stride = relative_attention_bias_stride;
    dispatch_params.attention_mask = params.attention_mask;
    dispatch_params.attention_mask_stride = params.attention_mask_stride;
    dispatch_params.max_distance = max_distance;
    dispatch_params.cache_indir = params.cache_indir;
    dispatch_params.context_buf = mCpSize > 1 ? mhaOutput : params.context_buf; //
    dispatch_params.finished = finished;
    dispatch_params.sequence_lengths
        = params.sequence_lengths; // NOTE: current seq len including padding (fixed after meeting the finished id)
    dispatch_params.max_batch_size = batch_beam;
    dispatch_params.inference_batch_size = batch_beam;
    dispatch_params.beam_width = params.beam_width;
    dispatch_params.head_num = mNumHeads / mCpSize;
    dispatch_params.kv_head_num = mNumKVHeads / mCpSize;
    dispatch_params.size_per_head = getHeadSize();
    dispatch_params.rotary_embedding_dim = mRotaryEmbeddingDim;
    dispatch_params.position_embedding_type = mPositionEmbeddingType;
    dispatch_params.max_attention_window = params.max_attention_window;
    dispatch_params.cyclic_attention_window_size = params.cyclic_attention_window_size;
    dispatch_params.sink_token_length = isCrossAttention() ? 0 : params.sink_token_length;
    dispatch_params.input_lengths = params.context_lengths;
    dispatch_params.timestep = timestep;
    dispatch_params.q_scaling = q_scaling;
    dispatch_params.attn_logit_softcapping_scale = mAttnLogitSoftcappingScale;
    dispatch_params.linear_bias_slopes = isALiBi() ? params.alibi_slopes : nullptr;
    dispatch_params.ia3_tasks = ia3_tasks;
    dispatch_params.ia3_key_weights = ia3_key_weights;
    dispatch_params.ia3_value_weights = ia3_value_weights;
    dispatch_params.qkv_scale_out = qkv_scale_out;
    dispatch_params.fp8_context_fmha = mFP8ContextFMHA;
    dispatch_params.attention_out_scale = params.attention_output_orig_quant;
    dispatch_params.quant_option = quant_option;
    dispatch_params.multi_block_mode = enable_multi_block;
    dispatch_params.max_seq_len_tile = max_num_seq_len_tiles;
    dispatch_params.min_seq_len_tile = min_num_seq_len_tiles;
    dispatch_params.partial_out = partial_out;
    dispatch_params.partial_sum = partial_sum;
    dispatch_params.partial_max = partial_max;
    dispatch_params.block_counter = mMultiBlockSemaphores.get();
    dispatch_params.kv_cache_quant_mode = mKVCacheQuantMode;
    dispatch_params.kv_scale_orig_quant = params.kv_scale_orig_quant;
    dispatch_params.kv_scale_quant_orig = params.kv_scale_quant_orig;
    dispatch_params.kv_block_array = kv_cache_buffer;
    dispatch_params.shift_k_cache_buffer = shift_k_cache_buffer;
    dispatch_params.multi_processor_count = mMultiProcessorCount;
    dispatch_params.rotary_embedding_base = mRotaryEmbeddingBase;
    dispatch_params.rotary_embedding_scale_type = mRotaryEmbeddingScaleType;
    dispatch_params.rotary_embedding_scale = mRotaryEmbeddingScale;
    dispatch_params.rotary_embedding_inv_freq_cache = params.rotary_inv_freq;
    dispatch_params.rotary_embedding_short_m_scale = mRotaryEmbeddingShortMscale;
    dispatch_params.rotary_embedding_long_m_scale = mRotaryEmbeddingLongMscale;
    dispatch_params.rotary_embedding_max_positions = mRotaryEmbeddingMaxPositions;
    dispatch_params.rotary_embedding_original_max_positions = mRotaryEmbeddingOriginalMaxPositions;
    dispatch_params.position_shift_enabled = mPosShiftEnabled;
    dispatch_params.rotary_cogvlm_vision_start = mVisionStart;
    dispatch_params.rotary_cogvlm_vision_length = mVisionLength;
    dispatch_params.cross_attention = isCrossAttention();
    dispatch_params.memory_length_per_sample = params.encoder_input_lengths;
    dispatch_params.block_sparse_attention = mMaskType == AttentionMaskType::BLOCKSPARSE;
    dispatch_params.block_sparse_params = mBlockSparseParams;
    dispatch_params.mrope_position_deltas = params.mrope_position_deltas;

    using DataType = typename SATypeConverter<T>::Type;
    if (!isCrossAttention())
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

    this->template ulyssesGenerationPostprocess<T>(batch_beam, mhaInput, mhaOutput, params.context_buf, stream);
    return 0;
}

template int AttentionOp::enqueueGeneration<half, KVLinearBuffer>(
    EnqueueGenerationParams<half> const& params, cudaStream_t stream);

template int AttentionOp::enqueueGeneration<float, KVLinearBuffer>(
    EnqueueGenerationParams<float> const& params, cudaStream_t stream);

#ifdef ENABLE_BF16
template int AttentionOp::enqueueGeneration<__nv_bfloat16, KVLinearBuffer>(
    EnqueueGenerationParams<__nv_bfloat16> const& params, cudaStream_t stream);
#endif

template int AttentionOp::enqueueGeneration<half, KVBlockArray>(
    EnqueueGenerationParams<half> const& params, cudaStream_t stream);

template int AttentionOp::enqueueGeneration<float, KVBlockArray>(
    EnqueueGenerationParams<float> const& params, cudaStream_t stream);

#ifdef ENABLE_BF16
template int AttentionOp::enqueueGeneration<__nv_bfloat16, KVBlockArray>(
    EnqueueGenerationParams<__nv_bfloat16> const& params, cudaStream_t stream);
#endif

template <typename T, typename KVCacheBuffer>
void AttentionOp::prepareEnqueueGeneration(EnqueueGenerationParams<T> const& params)
{
    // self attn
    if (mXqaDispatcher.get() != nullptr)
    {
        TLLM_LOG_TRACE("Preparing XQA kernels in prepareEnqueueGeneration.");
        XQAParams xqaParams{};
        this->template convertMMHAParamsToXQAParams<T, KVCacheBuffer>(xqaParams, params, /*forConfigurePlugin=*/true);
        mXqaDispatcher->prepare(xqaParams);
    }
}

template void AttentionOp::prepareEnqueueGeneration<half, KVLinearBuffer>(EnqueueGenerationParams<half> const& params);

template void AttentionOp::prepareEnqueueGeneration<float, KVLinearBuffer>(
    EnqueueGenerationParams<float> const& params);

#ifdef ENABLE_BF16
template void AttentionOp::prepareEnqueueGeneration<__nv_bfloat16, KVLinearBuffer>(
    EnqueueGenerationParams<__nv_bfloat16> const& params);
#endif

template void AttentionOp::prepareEnqueueGeneration<half, KVBlockArray>(EnqueueGenerationParams<half> const& params);

template void AttentionOp::prepareEnqueueGeneration<float, KVBlockArray>(EnqueueGenerationParams<float> const& params);

#ifdef ENABLE_BF16
template void AttentionOp::prepareEnqueueGeneration<__nv_bfloat16, KVBlockArray>(
    EnqueueGenerationParams<__nv_bfloat16> const& params);
#endif

int AttentionOp::initialize() noexcept
{
    // Pre-check whether FMHA is supported in order to save memory allocation.
    if (mEnableContextFMHA)
    {
        mEnableContextFMHA = false;
        if (!(mType == nvinfer1::DataType::kHALF || mType == nvinfer1::DataType::kBF16))
        {
            TLLM_LOG_WARNING("Fall back to unfused MHA because of unsupported data type.");
        }
        else if (mPositionEmbeddingType == tensorrt_llm::kernels::PositionEmbeddingType::kRELATIVE)
        {
            TLLM_LOG_WARNING("Fall back to unfused MHA because of relative position embedding.");
        }
        else if (isCrossAttention() && !mPagedKVCache)
        {
            // TODO: add the support for cross attention + contiguous kv cache.
            TLLM_LOG_WARNING("Fall back to unfused MHA because of cross attention + contiguous kv cache.");
        }
        else
        {
            mEnableContextFMHA = true;
        }
    }

    // Pre-Check of FP8 Context FMHA.
    if (mFP8ContextFMHA)
    {
        TLLM_CHECK_WITH_INFO(mEnableContextFMHA, "FP8 FMHA cannot be enabled because Context FMHA is not supported.");
        TLLM_CHECK_WITH_INFO(mSM == 89 || mSM == 90 || mSM == 100,
            "FP8 FMHA cannot be enabled except on Ada or Hopper or Blackwell Arch.");
    }

    // Check requirements for FP4 output.
    TLLM_CHECK_WITH_INFO(!mFuseFp4Quant || mEnableContextFMHA, "Context FMHA must enable if fuse_fp4_quant is enabled");
    TLLM_CHECK_WITH_INFO(!mFuseFp4Quant || (mSM >= 100), "fuse_fp4_quant only supports SM100 and later devices.");

    TLLM_CHECK(isRoPE() == (mRotaryEmbeddingDim != 0));
    TLLM_CHECK_WITH_INFO((mSM >= 80) || (mType != nvinfer1::DataType::kBF16),
        "Unsupported data type, pre SM 80 GPUs do not support bfloat16");

    // Pre-check whether the head size is supported by MMHA.
    if (!mmha_supported(getHeadSize()) && !mIsMLAEnabled)
    {
        TLLM_CHECK_WITH_INFO(false, "Head size %d is not supported by MMHA.", getHeadSize());
    }

    if (mIsMLAEnabled)
    {
        TLLM_CHECK_WITH_INFO(mEnableContextFMHA, "MLA(Deepseek v2) only support fmha");
        TLLM_CHECK_WITH_INFO(
            !mFP8ContextFMHA && !mDenseContextFMHA, "MLA(Deepseek v2) currently not support FP8 and dense fmha");
        TLLM_CHECK_WITH_INFO(
            mPagedKVCache && mUseKVCache && mRemovePadding, "MLA(Deepseek v2) only support paged kv cache");
        TLLM_CHECK_WITH_INFO(!mCrossAttention, "MLA(Deepseek v2) do not support cross attention right now");
        TLLM_CHECK_WITH_INFO(mMaskType != tensorrt_llm::kernels::AttentionMaskType::CUSTOM_MASK,
            "MLA(Deepseek v2) do not support custom mask right now");
        TLLM_CHECK_WITH_INFO(mMLAParams.qk_rope_head_dim == 64 && mMLAParams.kv_lora_rank == 512,
            "MLA(Deepseek v2) only support fixed kv_lora_rank(512) and fixed qk_rope_head_dim(64) right now.");
    }

    if (mIsFP8BlockScalingEnabled)
    {
        TLLM_CHECK_WITH_INFO(mIsMLAEnabled, "FP8 block scaling can only be used with MLA");
        mGemmRunner
            = std::make_shared<tensorrt_llm::kernels::small_m_gemm::CutlassFp8BlockScaleGemmRunner<__nv_bfloat16,
                __nv_fp8_e4m3, __nv_bfloat16>>();
    }

    mDriver = CUDADriverWrapper::getInstance();

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

        // FP8 FMHA should be used with fp8 workflow together.
        if (mFP8ContextFMHA)
        {
            data_type = DATA_TYPE_E4M3;
        }

        // Construct the fmha runner.
        MHARunnerFixedParams fmhaParams{};
        // The input dtype.
        fmhaParams.dataType = data_type;
        // The KV input data type. The default is same as dataType.
        fmhaParams.dataTypeKv = fmhaParams.dataType;
        // If the kernel must read from KV cache, set the dtype correctly.
        if (mPagedKVCache && mPagedContextFMHA)
        {
            if (mKVCacheQuantMode.hasFp8KvCache())
            {
                fmhaParams.dataTypeKv = DATA_TYPE_E4M3;
            }
            // TODO(tizheng): add FP4 KV cache support.
        }
        // The output dtype.
        fmhaParams.dataTypeOut = data_type;
        if (mFuseFp4Quant)
        {
            // If FP4 quantization workflow is enabled, set output type to FP4.
            fmhaParams.dataTypeOut = DATA_TYPE_E2M1;
        }
        // TODO(yibinl): remove forceFp32Acc from MHARunnerFixedParams after adding host_runtime_perf_knobs to
        // bertAttentionPlugin input tensors, so that we can change mLaunchParams.force_fp32_acc value in runtime.
        fmhaParams.forceFp32Acc = false;
        if (isCrossAttention())
        {
            fmhaParams.attentionMaskType = ContextAttentionMaskType::PADDING;
            // always use paged-kv-fmha if paged_kv cache is used.
            fmhaParams.attentionInputLayout
                = mPagedKVCache ? AttentionInputLayout::Q_PAGED_KV : AttentionInputLayout::Q_CONTIGUOUS_KV;
        }
        else
        {
            fmhaParams.attentionMaskType = ContextAttentionMaskType::CAUSAL;
            fmhaParams.attentionInputLayout = (mPagedKVCache && mPagedContextFMHA && !mIsMLAEnabled)
                ? AttentionInputLayout::Q_PAGED_KV
                : AttentionInputLayout::PACKED_QKV;
        }
        if (useCustomMask())
        {
            fmhaParams.attentionMaskType = ContextAttentionMaskType::CUSTOM_MASK;
        }
        fmhaParams.isSPadded = !mRemovePadding;
        fmhaParams.numQHeads = mNumHeads / mCpSize;
        fmhaParams.numKvHeads = mNumKVHeads / mCpSize;
        fmhaParams.numTokensPerBlock = mTokensPerBlock;
        fmhaParams.headSize = mHeadSize;
        if (mIsMLAEnabled)
        {
            // Context attention of MLA is different
            fmhaParams.numKvHeads = mNumHeads;
            fmhaParams.headSize = mMLAParams.qk_nope_head_dim + mMLAParams.qk_rope_head_dim;
            fmhaParams.headSizeV = mMLAParams.v_head_dim;
        }
        fmhaParams.qScaling = mQScaling;
        fmhaParams.attnLogitSoftcappingScale = mAttnLogitSoftcappingScale;
        fmhaParams.hasAlibi = isALiBi();
        fmhaParams.scaleAlibi = isAliBiWithScale();
        if (mTpSize > 1)
        {
            fmhaParams.tpSize = mTpSize;
            fmhaParams.tpRank = mTpRank;
        }
        else if (mCpSize > 1)
        {
            fmhaParams.tpSize = mCpSize;
            fmhaParams.tpRank = mCpRank;
        }

        // Load kernels from the pre-compiled cubins.
        mFmhaDispatcher.reset(new FmhaDispatcher(fmhaParams));

        // Deepseek-V2 Generation needs a differ fmha with different argumments
        if (mIsMLAEnabled)
        {
            mEnableXQA = false;
            // Construct the fmha runner.
            MHARunnerFixedParams fmhaParams{};
            fmhaParams.dataType = data_type;
            // TODO(yibinl): remove forceFp32Acc from MHARunnerFixedParams after adding host_runtime_perf_knobs to
            // bertAttentionPlugin input tensors, so that we can change mLaunchParams.force_fp32_acc value in runtime.
            fmhaParams.forceFp32Acc = false;
            fmhaParams.attentionMaskType
                = useCustomMask() ? ContextAttentionMaskType::CUSTOM_MASK : ContextAttentionMaskType::PADDING;
            // TODO: set it to Q_CONTIGUOUS_KV layout for cross-attention.
            fmhaParams.attentionInputLayout = AttentionInputLayout::Q_PAGED_KV;
            fmhaParams.isSPadded = !mRemovePadding;
            fmhaParams.numQHeads = 1;
            fmhaParams.numKvHeads = 1;
            fmhaParams.headSize = mMLAParams.kv_lora_rank + mMLAParams.qk_rope_head_dim;
            fmhaParams.headSizeV = mMLAParams.kv_lora_rank;
            fmhaParams.qScaling = mQScaling * sqrt((float) (mMLAParams.qk_nope_head_dim + mMLAParams.qk_rope_head_dim))
                / sqrtf((float) (mMLAParams.kv_lora_rank + mMLAParams.qk_rope_head_dim));
            fmhaParams.attnLogitSoftcappingScale = mAttnLogitSoftcappingScale;
            fmhaParams.hasAlibi = isALiBi();
            fmhaParams.scaleAlibi = isAliBiWithScale();
            fmhaParams.tpSize = mTpSize;
            fmhaParams.tpRank = mTpRank;
            mDecoderFMHARunner.reset(new FusedMHARunnerV2(fmhaParams));

            // Only deepseek must using fmha.
            TLLM_CHECK_WITH_INFO(mFmhaDispatcher->isSupported() && mDecoderFMHARunner->isFmhaSupported(),
                "Deepseek should be supported by fmha in context and generation part.");
        }

        // Fall back to unfused MHA kernels if not supported.
        mEnableContextFMHA = mFmhaDispatcher->isSupported();

        // Only FMHA supports custom mask currently.
        TLLM_CHECK_WITH_INFO(
            !useCustomMask() || mEnableContextFMHA, "Only Context FMHA supports custom mask input currently.");
    }

    mEnableXQA = (mEnableXQA || mIsSpecDecodingEnabled) && !mCrossAttention
        && (mType == nvinfer1::DataType::kHALF || mType == nvinfer1::DataType::kBF16);

    if (mEnableXQA)
    {
        TLLM_LOG_DEBUG("Enabling XQA kernels for GPTAttention.");

        XqaFixedParams fixedParams{};
        // TODO(tizheng): support more combinations.
        // Update Q and O dtype.
        if (mType == nvinfer1::DataType::kHALF)
        {
            fixedParams.inputDataType = DATA_TYPE_FP16;
            fixedParams.outputDataType = DATA_TYPE_FP16;
        }
        else if (mType == nvinfer1::DataType::kBF16)
        {
            fixedParams.inputDataType = DATA_TYPE_BF16;
            fixedParams.outputDataType = DATA_TYPE_BF16;
        }
        // Update KV cache and math dtype.
        if (mKVCacheQuantMode.hasInt8KvCache())
        {
            fixedParams.kvDataType = DATA_TYPE_INT8;
            fixedParams.mathDataType = fixedParams.inputDataType;
        }
        else if (mKVCacheQuantMode.hasFp8KvCache())
        {
            fixedParams.kvDataType = DATA_TYPE_E4M3;
            fixedParams.mathDataType = DATA_TYPE_E4M3;
        }
        else
        {
            fixedParams.kvDataType = fixedParams.inputDataType;
            fixedParams.mathDataType = fixedParams.inputDataType;
        }
        // If fuse_fp4_quant is enabled, set output data type to FP4.
        if (mFuseFp4Quant)
        {
            fixedParams.outputDataType = DATA_TYPE_E2M1;
        }
        // If FP8 context FMHA is enable, FP8 output is expected.
        else if (mFP8ContextFMHA)
        {
            fixedParams.outputDataType = DATA_TYPE_E4M3;
        }
        if (mIsSpecDecodingEnabled)
        {
            fixedParams.outputDataType = DATA_TYPE_E4M3;
            TLLM_CHECK_WITH_INFO(mNumHeads % mNumKVHeads == 0, "mNumHeads should be multiples of mNumKVHeads.");
            TLLM_CHECK_WITH_INFO(!mMultiBlockMode, "Medusa doesn't support multi-block mode.");
        }
        fixedParams.numQHeads = mNumHeads / mCpSize;
        fixedParams.numKvHeads = mNumKVHeads / mCpSize;
        fixedParams.numTokensPerBlock = mTokensPerBlock;
        fixedParams.headSize = mHeadSize;
        fixedParams.qScaling = mQScaling;
        fixedParams.multiBlockMode = mMultiBlockMode;
        fixedParams.isPagedKv = mPagedKVCache;
        fixedParams.isSpecDecoding = mIsSpecDecodingEnabled;
        fixedParams.hasAlibi = isALiBi();

        mXqaDispatcher.reset(new XqaDispatcher(fixedParams));

        // Fall back to unfused MHA kernels if not supported.
        mEnableXQA = mXqaDispatcher->isSupported();
    }
    else if (mIsSpecDecodingEnabled)
    {
        TLLM_CHECK_WITH_INFO(false, "Speculative decoding mode doesn't support the data type or cross attention.");
    }

    if (mNbMultiBlockSemaphores != 0)
    {
        reserveSemaphoreArray(mNbMultiBlockSemaphores);
    }

    if (isBuilding())
    {
        return 0;
    }
#if ENABLE_MULTI_DEVICE
    if (mCpSize > 1 && COMM_SESSION.getSize() > 1)
    {
        TLLM_LOG_TRACE("%s start for rank %d", __PRETTY_FUNCTION__, COMM_SESSION.getRank());
        mCpNcclComm = getComm(mCpGroup);
        TLLM_LOG_TRACE("%s stop for rank %d", __PRETTY_FUNCTION__, COMM_SESSION.getRank());
    }
#endif // ENABLE_MULTI_DEVICE
    return 0;
}

void AttentionOp::reserveSemaphoreArray(int32_t size)
{
    if (size == 0 || (size <= mNbMultiBlockSemaphores && mMultiBlockSemaphores != nullptr))
    {
        return;
    }
    int32_t* ptr;
    deviceMalloc(&ptr, size, false);
    deviceMemSetZero(ptr, size);
    mMultiBlockSemaphores.reset(ptr);
    mNbMultiBlockSemaphores = size;
}

void AttentionOp::debugCheckSemaphores(cudaStream_t stream)
{
#ifdef NDEBUG
    TLLM_CHECK_WITH_INFO(false, "debugCheckSemaphores should not be called in release build");
#endif
    if (mNbMultiBlockSemaphores == 0)
    {
        return;
    }
    std::vector<uint32_t> hostBuf(mNbMultiBlockSemaphores);
    TLLM_CUDA_CHECK(cudaMemcpyAsync(hostBuf.data(), mMultiBlockSemaphores.get(),
        sizeof(uint32_t) * mNbMultiBlockSemaphores, cudaMemcpyDeviceToHost, stream));
    TLLM_CUDA_CHECK(cudaStreamSynchronize(stream));
    TLLM_CHECK(std::count(hostBuf.begin(), hostBuf.end(), 0U) == mNbMultiBlockSemaphores);
}
