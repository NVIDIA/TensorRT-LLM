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
#include "gptAttentionPlugin.h"
#include "tensorrt_llm/kernels/decoderMaskedMultiheadAttention.h"
#include "tensorrt_llm/kernels/gptKernels.h"
#include "tensorrt_llm/kernels/unfusedAttentionKernels.h"
#include "tensorrt_llm/plugins/common/checkMacrosPlugin.h"
#include "tensorrt_llm/plugins/common/plugin.h"
#include "tensorrt_llm/plugins/gptAttentionCommon/gptAttentionCommon.h"
#include "tensorrt_llm/plugins/gptAttentionCommon/gptAttentionCommonImpl.h"
#include "tensorrt_llm/runtime/bufferManager.h"
#include "tensorrt_llm/runtime/iBuffer.h"
#include "tensorrt_llm/runtime/iTensor.h"
#include "tensorrt_llm/runtime/utils/debugUtils.h"
#include <algorithm>
#include <cstdint>
#include <functional>
#include <numeric>

using namespace nvinfer1;
using namespace tensorrt_llm::kernels;
using namespace tensorrt_llm::common;
using tensorrt_llm::plugins::GPTAttentionPluginCreator;
using tensorrt_llm::plugins::GPTAttentionPlugin;

static char const* GPT_ATTENTION_PLUGIN_VERSION{"1"};
static char const* GPT_ATTENTION_PLUGIN_NAME{"GPTAttention"};

GPTAttentionPlugin::GPTAttentionPlugin(int layer_idx, int num_heads, int vision_start, int vision_length,
    int num_kv_heads, int head_size, int unidirectional, float q_scaling, float qk_tanh_scale,
    tensorrt_llm::kernels::PositionEmbeddingType position_embedding_type,
    int rotary_embedding_dim, // for RoPE. 0 for non-RoPE
    float rotary_embedding_base, tensorrt_llm::kernels::RotaryScalingType rotary_embedding_scale_type,
    float rotary_embedding_scale, float rotary_embedding_short_m_scale,
    float rotary_embedding_long_m_scale, // magnitude scaling factors for Phi-3 long RoPE
    int rotary_embedding_max_positions, int rotary_embedding_original_max_positions, int tp_size,
    int tp_rank,                         // for ALiBi
    bool unfuse_qkv_gemm,                // for AutoPP
    tensorrt_llm::kernels::ContextFMHAType context_fmha_type, bool multi_block_mode, bool enable_xqa,
    int kv_cache_quant_mode, bool remove_input_padding, tensorrt_llm::kernels::AttentionMaskType mask_type,
    tensorrt_llm::kernels::BlockSparseParams block_sparse_params, bool paged_kv_cache, int tokens_per_block,
    nvinfer1::DataType type, int32_t max_context_length, bool qkv_bias_enabled, bool cross_attention, int max_distance,
    bool pos_shift_enabled, bool dense_context_fmha, bool use_paged_context_fmha, bool use_fp8_context_fmha,
    bool use_cache, bool is_spec_decoding_enabled)
    : GPTAttentionPluginCommon(layer_idx, num_heads, vision_start, vision_length, num_kv_heads, head_size,
        unidirectional, q_scaling, qk_tanh_scale, position_embedding_type, rotary_embedding_dim, rotary_embedding_base,
        rotary_embedding_scale_type, rotary_embedding_scale, rotary_embedding_short_m_scale,
        rotary_embedding_long_m_scale, rotary_embedding_max_positions, rotary_embedding_original_max_positions, tp_size,
        tp_rank, unfuse_qkv_gemm, context_fmha_type, multi_block_mode, enable_xqa, kv_cache_quant_mode,
        remove_input_padding, mask_type, block_sparse_params, paged_kv_cache, tokens_per_block, type,
        max_context_length, qkv_bias_enabled, cross_attention, max_distance, pos_shift_enabled, dense_context_fmha,
        use_paged_context_fmha, use_fp8_context_fmha, use_cache, is_spec_decoding_enabled)
{
    initEntryIdx();
}

GPTAttentionPlugin::GPTAttentionPlugin(void const* data, size_t length)
    : GPTAttentionPluginCommon(data, length)
{
    initEntryIdx();
}

bool GPTAttentionPlugin::isEntryUsed(IdxEntry const& entry) const
{
    switch (entry)
    {
    case IdxEntry::QKV_TENSOR: return true;
    case IdxEntry::K_TENSOR: return mUnfuseQkvGemm;
    case IdxEntry::V_TENSOR: return mUnfuseQkvGemm;
    case IdxEntry::SEQUENCE_LENGTH: return useKVCache();
    case IdxEntry::HOST_PAST_KEY_VALUE_LENGTHS: return useKVCache();
    case IdxEntry::HOST_MAX_ATTENTION_WINDOW: return true;
    case IdxEntry::HOST_SINK_TOKEN_LENGTH: return true;
    case IdxEntry::CONTEXT_LENGTHS: return true;
    case IdxEntry::CACHE_INDIR: return useKVCache();
    case IdxEntry::REQUEST_TYPES: return true;
    case IdxEntry::KV_CACHE_BLOCK_OFFSETS: return useKVCache() && mPagedKVCache;
    case IdxEntry::HOST_KV_CACHE_BLOCK_OFFSETS: return useKVCache() && mPagedKVCache;
    case IdxEntry::HOST_KV_CACHE_POOL_POINTERS: return useKVCache() && mPagedKVCache;
    case IdxEntry::PAST_KEY_VALUE: return useKVCache() && !mPagedKVCache;
    case IdxEntry::KV_CACHE_QUANTIZATION_SCALE: return useKVCache() && mKVCacheQuantMode.hasKvCacheQuant();
    case IdxEntry::KV_CACHE_DEQUANTIZATION_SCALE: return useKVCache() && mKVCacheQuantMode.hasKvCacheQuant();
    case IdxEntry::ATTENTION_OUTPUT_QUANTIZATION_SCALE: return mFP8ContextFMHA && mKVCacheQuantMode.hasFp8Qdq();
    case IdxEntry::ROTARY_COS_SIN: return isRoPE();
    case IdxEntry::ROTARY_EMBEDDING_SCALING_FACTORS: return isLongRoPE();
    case IdxEntry::ALIBI_SLOPES: return isALiBi();
    case IdxEntry::RELATIVE_ATTENTION_BIAS: return isRelativePosition();
    case IdxEntry::CROSS_QKV: return isCrossAttention();
    case IdxEntry::CROSS_QKV_LENGTH: return isCrossAttention();
    case IdxEntry::ENCODER_INPUT_LENGTH: return isCrossAttention();
    case IdxEntry::HOST_CONTEXT_LENGTH: return mRemovePadding;
    case IdxEntry::QKV_BIAS_TENSOR: return mQKVBiasEnabled;
    case IdxEntry::SPEC_DECODING_GENERATION_LENGTHS: return mIsSpecDecodingEnabled;
    case IdxEntry::SPEC_DECODING_PACKED_MASK: return mIsSpecDecodingEnabled;
    case IdxEntry::SPEC_DECODING_POSITION_OFFSETS: return mIsSpecDecodingEnabled;
    default: return false;
    }
}

void GPTAttentionPlugin::initEntryIdx()
{
    mEntryIdx.resize(static_cast<size_t>(IdxEntry::ENUM_SIZE));
    size_t entryIdx = 0;
    for (int i = 0; i < static_cast<size_t>(IdxEntry::ENUM_SIZE); i++)
    {
        mEntryIdx[i] = entryIdx;
        entryIdx += isEntryUsed(static_cast<IdxEntry>(i));
    }
}

GPTAttentionPlugin::IndexType GPTAttentionPlugin::getIdx(IdxEntry const& entry) const
{
    TLLM_CHECK_WITH_INFO(
        isEntryUsed(entry), common::fmtstr("getIdx() should not be used with entry %lu\n", static_cast<size_t>(entry)));
    return mEntryIdx[static_cast<size_t>(entry)];
}

// IPluginV2DynamicExt Methods
GPTAttentionPlugin* GPTAttentionPlugin::clone() const noexcept
{
    return dynamic_cast<GPTAttentionPlugin*>(this->cloneImpl<GPTAttentionPlugin>());
}

static int getPackedTensorHiddenDimIndex(bool removePadding)
{
    return removePadding ? 1 : 2;
}

// NOTE: generation input length might be larger than one in the spec decoding mode.
int GPTAttentionPlugin::getGenerationInputSequenceLength(
    nvinfer1::PluginTensorDesc const* inputDesc, int32_t localNbSeq, int32_t localNbTokens) const
{
    if (mRemovePadding)
    {
        // Speculative decoding mode might need variable generation input sequence length.
        if (mIsSpecDecodingEnabled)
        {
            // SPEC_DECODING_POSITION_OFFSETS: [batch_size, max_generation_input_length].
            return inputDesc[getIdx(IdxEntry::SPEC_DECODING_POSITION_OFFSETS)].dims.d[1];
        }
        else
        {
            // [num_tokens, local_hidden_size] where num_tokens = batch_size * generation_input_length
            TLLM_CHECK_WITH_INFO(localNbTokens % localNbSeq == 0,
                "seq_len should be same for all generation requests, localNbTokens=%d, localNbSeq=%d", localNbTokens,
                localNbSeq);
            return localNbTokens / localNbSeq;
        }
    }
    else
    {
        // We don't have IFB without mRemovePadding, so just take it out from inputDesc
        // [batch_size, seq_len, local_hidden_size]
        return inputDesc[getIdx(IdxEntry::QKV_TENSOR)].dims.d[1];
    }
}

// outputs
//     output_tensor [batch_size, seq_len, local_hidden_size] or [num_tokens, local_hidden_size]
//     present_key_value_pool (optional if mPagedKVCache is false) [batch_size, 2, local_num_kv_heads, max_seq_len,
//     head_size]
nvinfer1::DimsExprs GPTAttentionPlugin::getOutputDimensions(
    int outputIndex, nvinfer1::DimsExprs const* inputs, int nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept
{
    TLLM_CHECK(outputIndex == 0 || (!mPagedKVCache && useKVCache() && outputIndex == 1));
    if (outputIndex == 0)
    {
        auto ret = inputs[getIdx(IdxEntry::QKV_TENSOR)];
        ret.d[getPackedTensorHiddenDimIndex(mRemovePadding)] = exprBuilder.operation(
            DimensionOperation::kPROD, *exprBuilder.constant(mHeadSize), *exprBuilder.constant(mNumHeads));
        return ret;
    }
    return inputs[getIdx(IdxEntry::PAST_KEY_VALUE)];
}

bool GPTAttentionPlugin::supportsFormatCombination(
    int pos, nvinfer1::PluginTensorDesc const* inOut, int nbInputs, int nbOutputs) noexcept
{
    if (pos == getIdx(IdxEntry::CONTEXT_LENGTHS) || pos == getIdx(IdxEntry::REQUEST_TYPES)
        || pos == getIdx(IdxEntry::HOST_MAX_ATTENTION_WINDOW) || pos == getIdx(IdxEntry::HOST_SINK_TOKEN_LENGTH)
        || (isEntryUsed(IdxEntry::SPEC_DECODING_PACKED_MASK) && pos == getIdx(IdxEntry::SPEC_DECODING_PACKED_MASK))
        || (isEntryUsed(IdxEntry::SPEC_DECODING_POSITION_OFFSETS)
            && pos == getIdx(IdxEntry::SPEC_DECODING_POSITION_OFFSETS))
        || (isEntryUsed(IdxEntry::SPEC_DECODING_GENERATION_LENGTHS)
            && pos == getIdx(IdxEntry::SPEC_DECODING_GENERATION_LENGTHS)))
    {
        return inOut[pos].type == nvinfer1::DataType::kINT32;
    }
    else if (useKVCache()
        && (pos == getIdx(IdxEntry::SEQUENCE_LENGTH) || pos == getIdx(IdxEntry::HOST_PAST_KEY_VALUE_LENGTHS)
            || pos == getIdx(IdxEntry::CACHE_INDIR)))
    {
        return inOut[pos].type == nvinfer1::DataType::kINT32;
    }
    else if (isRoPE() && (pos == getIdx(IdxEntry::ROTARY_COS_SIN)))
    {
        return inOut[pos].type == nvinfer1::DataType::kFLOAT;
    }
    else if (isLongRoPE() && (pos == getIdx(IdxEntry::ROTARY_EMBEDDING_SCALING_FACTORS)))
    {
        return inOut[pos].type == nvinfer1::DataType::kFLOAT;
    }
    else if (useKVCache() && mKVCacheQuantMode.hasKvCacheQuant()
        && (pos == getIdx(IdxEntry::KV_CACHE_DEQUANTIZATION_SCALE)
            || pos == getIdx(IdxEntry::KV_CACHE_QUANTIZATION_SCALE)))
    {
        // kv_scale for mType->int8/fp8 and int8/fp8->mType conversion
        return inOut[pos].type == nvinfer1::DataType::kFLOAT && inOut[pos].format == TensorFormat::kLINEAR;
    }
    else if (mFP8ContextFMHA && pos == getIdx(IdxEntry::ATTENTION_OUTPUT_QUANTIZATION_SCALE))
    {
        return inOut[pos].type == nvinfer1::DataType::kFLOAT && inOut[pos].format == TensorFormat::kLINEAR;
    }
    else if (mPagedKVCache
        && (pos == getIdx(IdxEntry::KV_CACHE_BLOCK_OFFSETS) || pos == getIdx(IdxEntry::HOST_KV_CACHE_BLOCK_OFFSETS)))
    {
        // kv cache block offsets
        return inOut[pos].type == nvinfer1::DataType::kINT32 && inOut[pos].format == TensorFormat::kLINEAR;
    }
    else if (mPagedKVCache && (pos == getIdx(IdxEntry::HOST_KV_CACHE_POOL_POINTERS)))
    {
        // kv cache pool pointers
        return inOut[pos].type == nvinfer1::DataType::kINT64 && inOut[pos].format == TensorFormat::kLINEAR;
    }
    else if (mKVCacheQuantMode.hasInt8KvCache()
        && (!mPagedKVCache && (pos == getIdx(IdxEntry::PAST_KEY_VALUE) || pos == nbInputs + 1)))
    {
        // If use Int8 K/V cache we require I/O KV values to int8
        return (inOut[pos].type == nvinfer1::DataType::kINT8) && (inOut[pos].format == TensorFormat::kLINEAR);
    }
    else if (mKVCacheQuantMode.hasFp8KvCache()
        && (!mPagedKVCache && (pos == getIdx(IdxEntry::PAST_KEY_VALUE) || pos == nbInputs + 1)))
    {
        // If use FP8 K/V cache we require I/O KV values to FP8
        return (inOut[pos].type == nvinfer1::DataType::kFP8) && (inOut[pos].format == TensorFormat::kLINEAR);
    }
    else if (mRemovePadding && (pos == getIdx(IdxEntry::HOST_CONTEXT_LENGTH)))
    {
        return inOut[pos].type == nvinfer1::DataType::kINT32 && inOut[pos].format == TensorFormat::kLINEAR;
    }
    else if (mCrossAttention
        && (pos == getIdx(IdxEntry::CROSS_QKV_LENGTH) || pos == getIdx(IdxEntry::ENCODER_INPUT_LENGTH)))
    {
        return inOut[pos].type == nvinfer1::DataType::kINT32;
    }
    else if (pos == nbInputs && mFP8ContextFMHA)
    {
        // Output tensor now supports fp8 data type.
        return (inOut[pos].type == nvinfer1::DataType::kFP8) && (inOut[pos].format == TensorFormat::kLINEAR);
    }
    else
    {
        return (inOut[pos].type == mType) && (inOut[pos].format == TensorFormat::kLINEAR);
    }
    return false;
}

template <typename T, typename KVCacheBuffer>
void GPTAttentionPlugin::configurePluginImpl(nvinfer1::DynamicPluginTensorDesc const* in, int nbInputs,
    nvinfer1::DynamicPluginTensorDesc const* out, int nbOutputs) noexcept
{
    TLLM_CHECK(mHeadSize > 0);

    int const beamWidth
        = isCrossAttention() ? 1 : (useKVCache() ? in[getIdx(IdxEntry::CACHE_INDIR)].desc.dims.d[1] : 1);
    // Commonly, cyclic_attention_window_size, and max_attention_window_size will be the same
    // unless each layer has different attention window sizes.
    // the kv_cache capacity.
    int max_encoder_context_len = isCrossAttention() ? in[getIdx(IdxEntry::CROSS_QKV_LENGTH)].desc.dims.d[0] : 0;
    int const max_attention_window_size = isCrossAttention()
        ? max_encoder_context_len
        : (useKVCache() ? in[getIdx(IdxEntry::CACHE_INDIR)].desc.dims.d[2] : 0);
    int const cyclic_attention_window_size = max_attention_window_size;

    int const num_requests = 256;
    int const sink_token_length = 0;

    EnqueueGenerationParams<T, KVCacheBuffer> enqueueParams{/*attention_input=*/nullptr,
        /*qkv_bias=*/nullptr,
        /*input_seq_length=*/0,
        /*sequence_lengths=*/nullptr,
        /*past_kv_length=*/0, beamWidth,
        /*context_lengths=*/nullptr,
        /*kv_scale_orig_quant=*/nullptr,
        /*kv_scale_quant_orig=*/nullptr,
        /*attention_out_orig_quant=*/nullptr,
        /*rotary_embedding_scaling_factors*/ nullptr,
        /*alibi_slopes=*/nullptr,
        /*context_buf_=*/nullptr,
        /*key_value_cache=*/nullptr,
        /*block_offsets=*/nullptr,
        /*host_primary_pool_pointer=*/nullptr,
        /*host_secondary_pool_pointer=*/nullptr, max_attention_window_size, cyclic_attention_window_size,
        sink_token_length, num_requests,
        /*max_blocks_per_sequence=*/0,
        /*cache_indir=*/nullptr,
        /*workspace=*/nullptr,
        /*max_context_kv_len_list=*/nullptr};

    prepareEnqueueGeneration(enqueueParams);

    // Always reserve SemaphoreArray (for multi-block mode) as MMHA may enable multi-block mode when shared memory is
    // not enough.
    auto const& ctxLenTensor = in[getIdx(IdxEntry::CONTEXT_LENGTHS)];
    TLLM_CHECK_DEBUG(ctxLenTensor.max.nbDims == 1);
    int32_t const max_batch_beam = in[getIdx(IdxEntry::CONTEXT_LENGTHS)].max.d[0];
    reserveSemaphoreArray(mNumHeads * max_batch_beam);
}

template <typename T>
void GPTAttentionPlugin::configurePluginDispatchKVCacheType(nvinfer1::DynamicPluginTensorDesc const* in, int nbInputs,
    nvinfer1::DynamicPluginTensorDesc const* out, int nbOutputs) noexcept
{
    if (mPagedKVCache)
    {
        configurePluginImpl<T, KVBlockArray>(in, nbInputs, out, nbOutputs);
    }
    else
    {
        configurePluginImpl<T, KVLinearBuffer>(in, nbInputs, out, nbOutputs);
    }
}

void GPTAttentionPlugin::configurePlugin(nvinfer1::DynamicPluginTensorDesc const* in, int nbInputs,
    nvinfer1::DynamicPluginTensorDesc const* out, int nbOutputs) noexcept
{
    if (mType == nvinfer1::DataType::kHALF)
    {
        configurePluginDispatchKVCacheType<half>(in, nbInputs, out, nbOutputs);
    }
    else if (mType == nvinfer1::DataType::kFLOAT)
    {
        configurePluginDispatchKVCacheType<float>(in, nbInputs, out, nbOutputs);
    }
#ifdef ENABLE_BF16
    else if (mType == nvinfer1::DataType::kBF16)
    {
        configurePluginDispatchKVCacheType<__nv_bfloat16>(in, nbInputs, out, nbOutputs);
    }
#endif
}

size_t GPTAttentionPlugin::getWorkspaceSize(nvinfer1::PluginTensorDesc const* inputs, int nbInputs,
    nvinfer1::PluginTensorDesc const* outputs, int nbOutputs) const noexcept
{
    int const max_context_length = mMaxContextLength;
    int const cross_qkv_length = isCrossAttention() ? inputs[getIdx(IdxEntry::CROSS_QKV_LENGTH)].dims.d[0] : 0;
    int const max_num_seq = inputs[getIdx(IdxEntry::CONTEXT_LENGTHS)].dims.d[0];
    auto const type = inputs[getIdx(IdxEntry::QKV_TENSOR)].type;
    int const max_kv_cache_length
        = isCrossAttention() ? cross_qkv_length : (useKVCache() ? inputs[getIdx(IdxEntry::CACHE_INDIR)].dims.d[2] : 0);
    int const max_num_tokens
        = mRemovePadding ? inputs[getIdx(IdxEntry::QKV_TENSOR)].dims.d[0] : max_num_seq * max_context_length;
    size_t const context_workspace_size
        = getWorkspaceSizeForContext(type, max_num_seq, max_context_length, cross_qkv_length, max_num_tokens);

    int32_t const num_spec_dec_tokens
        = mIsSpecDecodingEnabled ? inputs[getIdx(IdxEntry::SPEC_DECODING_POSITION_OFFSETS)].dims.d[1] : 1;
    int32_t const max_batch_beam = inputs[getIdx(IdxEntry::CONTEXT_LENGTHS)].dims.d[0];
    int32_t const max_num_gen_tokens = std::min(max_num_tokens, num_spec_dec_tokens * max_batch_beam);
    size_t const generation_workspace_size
        = getWorkspaceSizeForGeneration(type, max_num_seq, max_kv_cache_length, max_num_tokens);

    size_t attention_input_workspace_size = 0;
    if (mUnfuseQkvGemm)
    {
        int const local_hidden_units_q
            = inputs[getIdx(IdxEntry::QKV_TENSOR)].dims.d[getPackedTensorHiddenDimIndex(mRemovePadding)];
        int const local_hidden_units_kv
            = inputs[getIdx(IdxEntry::K_TENSOR)].dims.d[getPackedTensorHiddenDimIndex(mRemovePadding)];
        size_t const size = tensorrt_llm::runtime::BufferDataType(type).getSize();
        size_t const attention_input_size = size * max_num_tokens * (local_hidden_units_q + 2 * local_hidden_units_kv);
        size_t workspaces[1];
        workspaces[0] = attention_input_size;
        attention_input_workspace_size = tensorrt_llm::common::calculateTotalWorkspaceSize(workspaces, 1);
    }
    return std::max(context_workspace_size, generation_workspace_size) + attention_input_workspace_size;
}

static size_t getStride(nvinfer1::Dims const& dims, int n)
{
    TLLM_CHECK(n >= 0 && n < dims.nbDims);
    return std::accumulate(dims.d + n + 1, dims.d + dims.nbDims, 1, std::multiplies<size_t>{});
}

template <typename T, typename AttentionOutT, typename KVCacheBuffer>
int GPTAttentionPlugin::enqueueImpl(nvinfer1::PluginTensorDesc const* inputDesc,
    nvinfer1::PluginTensorDesc const* outputDesc, void const* const* inputs, void* const* outputs, void* workspace,
    cudaStream_t stream)
{
    TLLM_LOG_TRACE("Attention plugin start at layer %d", mLayerIdx);
    int32_t const nbSeq = inputDesc[getIdx(IdxEntry::CONTEXT_LENGTHS)].dims.d[0];
    int32_t const beam_width = useKVCache() ? inputDesc[getIdx(IdxEntry::CACHE_INDIR)].dims.d[1] : 1;
    RequestType const* reqTypes = static_cast<RequestType const*>(inputs[getIdx(IdxEntry::REQUEST_TYPES)]);

    int32_t nbContextRequests = 0;
    int32_t contextTokenIdxEnd = 0;
    // count context requests
    for (int32_t seqIdx = 0; seqIdx < nbSeq; seqIdx++)
    {
        if (reqTypes[seqIdx] != RequestType::kCONTEXT)
        {
            break;
        }
        ++nbContextRequests;
        contextTokenIdxEnd += mRemovePadding
            ? static_cast<int32_t const*>(inputs[getIdx(IdxEntry::HOST_CONTEXT_LENGTH)])[seqIdx]
            : inputDesc[getIdx(IdxEntry::QKV_TENSOR)].dims.d[1];
    }
    for (int32_t seqIdx = nbContextRequests; seqIdx < nbSeq; seqIdx++)
    {
        TLLM_CHECK(reqTypes[seqIdx] == RequestType::kGENERATION);
    }

    // mixed requests require mRemovePadding and mPagedKVCache
    if (nbContextRequests != 0 && nbContextRequests != nbSeq)
    {
        TLLM_CHECK(mRemovePadding && mPagedKVCache);
    }

    if (nbContextRequests > 0)
    {
        auto seqIdxBeg = 0;
        auto tokenIdxBeg = 0;
        auto localNbTokens = contextTokenIdxEnd;
        enqueueSome<T, AttentionOutT, KVCacheBuffer>(seqIdxBeg, nbContextRequests, tokenIdxBeg, localNbTokens,
            inputDesc, outputDesc, inputs, outputs, workspace, stream);
    }

    if (auto nbGenerationSeq = nbSeq - nbContextRequests; nbGenerationSeq > 0)
    {
        auto seqIdxBeg = nbContextRequests;
        auto tokenIdxBeg = contextTokenIdxEnd;
        // if mRemovePadding is true, we may have IFB, and need to remove context tokens.
        // if mRemovePadding is false, it is only generation requests, so just multiply batch_beam and seq_len (May not
        // 1 for Parallel Decoding)
        auto localNbTokens = mRemovePadding
            ? inputDesc[getIdx(IdxEntry::QKV_TENSOR)].dims.d[0] - contextTokenIdxEnd
            : inputDesc[getIdx(IdxEntry::QKV_TENSOR)].dims.d[0] * inputDesc[getIdx(IdxEntry::QKV_TENSOR)].dims.d[1];
        enqueueSome<T, AttentionOutT, KVCacheBuffer>(seqIdxBeg, nbGenerationSeq, tokenIdxBeg, localNbTokens, inputDesc,
            outputDesc, inputs, outputs, workspace, stream);
    }

    TLLM_LOG_TRACE("Attention plugin stop at layer %d", mLayerIdx);

    return 0;
}

template <typename T, typename AttentionOutT, typename KVCacheBuffer>
int GPTAttentionPlugin::enqueueSome(int32_t seqIdxBeg, int32_t localNbSeq, int32_t tokenIdxBeg, int32_t localNbTokens,
    nvinfer1::PluginTensorDesc const* inputDesc, nvinfer1::PluginTensorDesc const* outputDesc,
    void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream)
{
    //     relative_attention_bias [head_num, max_seq_len, max_seq_len] (optional in relative position)
    //                          or [head_num, num_buckets] (optional in implicit relative attention)
    //     cross_qkv [batch_size, seq_len, 3 * local_hidden_size] or [num_tokens, 3 * local_hidden_size]
    //               when enable remove_input_padding (optional in cross attention mode)
    //     cross_qkv_length [int] max encoder input context length (optional in cross attention mode)
    //     encoder_input_lengths [batch_size] raw sequence lengths (optional in cross attention mode)

    T const* attention_input = static_cast<T const*>(inputs[getIdx(IdxEntry::QKV_TENSOR)])
        + inputDesc[getIdx(IdxEntry::QKV_TENSOR)].dims.d[getPackedTensorHiddenDimIndex(mRemovePadding)]
            * size_t(tokenIdxBeg);
    T const* qkv_bias = nullptr;
    if (mQKVBiasEnabled)
    {
        qkv_bias = reinterpret_cast<T const*>(inputs[getIdx(IdxEntry::QKV_BIAS_TENSOR)]);
    }
    // Rotary cos_sin cache to avoid re-computing.
    float2 const* rotary_cos_sin = nullptr;
    if (isRoPE())
    {
        rotary_cos_sin = reinterpret_cast<float2 const*>(inputs[getIdx(IdxEntry::ROTARY_COS_SIN)]);
    }

    float const* rotary_embedding_scaling_factors = nullptr;
    if (isLongRoPE())
    {
        rotary_embedding_scaling_factors
            = reinterpret_cast<float const*>(inputs[getIdx(IdxEntry::ROTARY_EMBEDDING_SCALING_FACTORS)]);
    }

    auto const reqTypeInBatchPtr = static_cast<RequestType const*>(inputs[getIdx(IdxEntry::REQUEST_TYPES)]) + seqIdxBeg;
    bool const is_context = (reqTypeInBatchPtr[0] == RequestType::kCONTEXT);

    if (mUnfuseQkvGemm)
    {
        int const max_seqlen = inputDesc[getIdx(IdxEntry::QKV_TENSOR)].dims.d[mRemovePadding ? 0 : 1];
        int const batch_size = mRemovePadding ? 1 : inputDesc[getIdx(IdxEntry::QKV_TENSOR)].dims.d[0];

        T const* attention_input_q = static_cast<T const*>(inputs[getIdx(IdxEntry::QKV_TENSOR)]);
        T const* attention_input_k = static_cast<T const*>(inputs[getIdx(IdxEntry::K_TENSOR)]);
        T const* attention_input_v = static_cast<T const*>(inputs[getIdx(IdxEntry::V_TENSOR)]);
        size_t const hidden_units_q
            = inputDesc[getIdx(IdxEntry::QKV_TENSOR)].dims.d[getPackedTensorHiddenDimIndex(mRemovePadding)];
        size_t const hidden_units_kv
            = inputDesc[getIdx(IdxEntry::K_TENSOR)].dims.d[getPackedTensorHiddenDimIndex(mRemovePadding)];
        size_t const hidden_units = hidden_units_q + 2 * hidden_units_kv;
        size_t const size_qkv = sizeof(T) * hidden_units;
        size_t const size_q = sizeof(T) * hidden_units_q;
        size_t const size_kv = sizeof(T) * hidden_units_kv;
        size_t const total_size = size_qkv * batch_size * max_seqlen;
        int8_t* workspace_byte_ptr = reinterpret_cast<int8_t*>(workspace);
        size_t offset = 0;
        T* attention_input_qkv = reinterpret_cast<T*>(nextWorkspacePtr(workspace_byte_ptr, offset, total_size));
        workspace = reinterpret_cast<void*>(workspace_byte_ptr + offset);

        cudaMemcpy2DAsync(attention_input_qkv, size_qkv, attention_input_q, size_q, size_q, batch_size * max_seqlen,
            cudaMemcpyDeviceToDevice, stream);
        cudaMemcpy2DAsync(attention_input_qkv + hidden_units_q, size_qkv, attention_input_k, size_kv, size_kv,
            batch_size * max_seqlen, cudaMemcpyDeviceToDevice, stream);
        cudaMemcpy2DAsync(attention_input_qkv + hidden_units_q + hidden_units_kv, size_qkv, attention_input_v, size_kv,
            size_kv, batch_size * max_seqlen, cudaMemcpyDeviceToDevice, stream);

        attention_input = attention_input_qkv + hidden_units * tokenIdxBeg;
    }

    int const* context_q_lengths = reinterpret_cast<int const*>(inputs[getIdx(IdxEntry::CONTEXT_LENGTHS)]) + seqIdxBeg;
    int const* sequence_kv_length = useKVCache()
        ? static_cast<int const*>(inputs[getIdx(IdxEntry::SEQUENCE_LENGTH)]) + seqIdxBeg
        : context_q_lengths;
    // Note we still need context length during generation for MMHA optimization.
    int32_t const max_context_q_len = [&]()
    {
        if (!mRemovePadding)
        {
            return static_cast<int>(inputDesc[getIdx(IdxEntry::QKV_TENSOR)].dims.d[1]);
        }
        auto const host_context_lengths
            = static_cast<int32_t const*>(inputs[getIdx(IdxEntry::HOST_CONTEXT_LENGTH)]) + seqIdxBeg;
        return *std::max_element(host_context_lengths, host_context_lengths + localNbSeq);
    }();

    int max_encoder_context_len = isCrossAttention() ? inputDesc[getIdx(IdxEntry::CROSS_QKV_LENGTH)].dims.d[0] : 0;
    // for enc-dec model, since decoder_input_ids could be longer than 1,
    // such model has an encoder context (for cross attn) and an decoder context (for self attn)
    // clarify 3 lens:
    // -- max_context_q_len: len of decoder input. No "max" concept, it's what it is given.
    //                     Also called (decoder_)input_seq_length, normally 1 for encoder-decoder start token
    // -- max_seq_len: max allowed len of decoder output, i.e. final results
    // -- max_encoder_context_len: len of encoder input (in cross attn). Also called encoder_input_seq_length

    int const beamWidth
        = isCrossAttention() ? 1 : (useKVCache() ? inputDesc[getIdx(IdxEntry::CACHE_INDIR)].dims.d[1] : 1);

    // Commonly, cyclic_attention_window_size, and max_attention_window_size will be the same
    // unless each layer has different attention window sizes.
    // the kv_cache capacity.
    int const max_attention_window_size = isCrossAttention()
        ? max_encoder_context_len
        : (useKVCache() ? inputDesc[getIdx(IdxEntry::CACHE_INDIR)].dims.d[2] : 0);
    // The cyclic_attention_window_size will determine the cyclic kv cache position of new tokens.
    // Note that this cyclic_attention_window_size might be smaller than the actual kv cache capactity.
    int const cyclic_attention_window_size = isCrossAttention()
        ? max_encoder_context_len
        : reinterpret_cast<int const*>(inputs[getIdx(IdxEntry::HOST_MAX_ATTENTION_WINDOW)])[mLayerIdx];
    int const sink_token_length = reinterpret_cast<int const*>(inputs[getIdx(IdxEntry::HOST_SINK_TOKEN_LENGTH)])[0];

    float const* kv_scale_orig_quant = nullptr;
    float const* kv_scale_quant_orig = nullptr;
    if (useKVCache() && mKVCacheQuantMode.hasKvCacheQuant())
    {
        assert(inputDesc[getIdx(IdxEntry::KV_CACHE_QUANTIZATION_SCALE)].type == nvinfer1::DataType::kFLOAT);
        assert(inputDesc[getIdx(IdxEntry::KV_CACHE_DEQUANTIZATION_SCALE)].type == nvinfer1::DataType::kFLOAT);
        kv_scale_orig_quant = reinterpret_cast<float const*>(inputs[getIdx(IdxEntry::KV_CACHE_QUANTIZATION_SCALE)]);
        kv_scale_quant_orig = reinterpret_cast<float const*>(inputs[getIdx(IdxEntry::KV_CACHE_DEQUANTIZATION_SCALE)]);
    }

    float const* attention_output_orig_quant = nullptr;
    if (mFP8ContextFMHA)
    {
        assert(inputDesc[getIdx(IdxEntry::ATTENTION_OUTPUT_QUANTIZATION_SCALE)].type == nvinfer1::DataType::kFLOAT);
        attention_output_orig_quant
            = reinterpret_cast<float const*>(inputs[getIdx(IdxEntry::ATTENTION_OUTPUT_QUANTIZATION_SCALE)]);
    }

    int max_blocks_per_sequence = 0;
    kernels::KVBlockArray::DataType* block_offsets = nullptr;
    kernels::KVBlockArray::DataType* host_block_offsets = nullptr;
    void* host_primary_pool_pointer = nullptr;
    void* host_secondary_pool_pointer = nullptr;
    if (useKVCache() && mPagedKVCache)
    {
        auto const& kvCacheBlockOffsets = inputDesc[getIdx(IdxEntry::KV_CACHE_BLOCK_OFFSETS)];
        auto const& kvCacheBlockOffsetsShape = inputDesc[getIdx(IdxEntry::KV_CACHE_BLOCK_OFFSETS)].dims;
        max_blocks_per_sequence = kvCacheBlockOffsetsShape.d[kvCacheBlockOffsetsShape.nbDims - 1];
        auto const seqStride = getStride(kvCacheBlockOffsetsShape, 0);
        auto const seqOffset = seqIdxBeg * seqStride;

        block_offsets
            = reinterpret_cast<kernels::KVBlockArray::DataType*>(inputs[getIdx(IdxEntry::KV_CACHE_BLOCK_OFFSETS)])
            + seqOffset;

        host_block_offsets
            = reinterpret_cast<kernels::KVBlockArray::DataType*>(inputs[getIdx(IdxEntry::HOST_KV_CACHE_BLOCK_OFFSETS)])
            + seqOffset;

        auto const* const typed_host_pool_pointers
            = static_cast<char* const*>(inputs[getIdx(IdxEntry::HOST_KV_CACHE_POOL_POINTERS)]);

        auto const cacheElemSize = (mKVCacheQuantMode.hasKvCacheQuant() ? 1 : sizeof(T));
        auto const blockSize = mTokensPerBlock * mNumKVHeads * mHeadSize;
        auto const bytesPerBlock = blockSize * cacheElemSize;
        auto const layerOffset = mLayerIdx * 2 * bytesPerBlock;

        host_primary_pool_pointer = reinterpret_cast<void*>(typed_host_pool_pointers[0] + layerOffset);
        host_secondary_pool_pointer = reinterpret_cast<void*>(typed_host_pool_pointers[1] + layerOffset);
    }

    AttentionOutT* context_buf_ = static_cast<AttentionOutT*>(outputs[0])
        + outputDesc[0].dims.d[getPackedTensorHiddenDimIndex(mRemovePadding)] * tokenIdxBeg;
    void* key_value_cache = nullptr;
    if (useKVCache() && !mPagedKVCache)
    {
        auto const cacheElemSize = (mKVCacheQuantMode.hasKvCacheQuant() ? 1 : sizeof(T));
        key_value_cache
            = static_cast<std::byte*>(outputs[1]) + cacheElemSize * getStride(outputDesc[1].dims, 0) * seqIdxBeg;
        void const* past_key_value_cache = inputs[getIdx(IdxEntry::PAST_KEY_VALUE)];
        if (past_key_value_cache != outputs[1])
        {
            auto shape = outputDesc[1].dims;
            auto const size = std::accumulate(shape.d, shape.d + shape.nbDims, 1, std::multiplies<size_t>{});
            cudaMemcpyAsync(outputs[1], past_key_value_cache, size, cudaMemcpyDeviceToDevice, stream);
        }
    }

    T const* alibi_slopes = isALiBi() ? static_cast<T const*>(inputs[getIdx(IdxEntry::ALIBI_SLOPES)]) : nullptr;

    int const* spec_decoding_packed_mask = nullptr;
    int const* spec_decoding_position_offsets = nullptr;
    int const* spec_decoding_generation_lengths = nullptr;
    int num_decoding_draft_tokens = 0;
    if (mIsSpecDecodingEnabled)
    {
        // Second dimension of spec_decoding_position_offsets is num_decoding_draft_tokens + 1.
        // [batch_size, num_decoding_draft_tokens + 1]
        num_decoding_draft_tokens = inputDesc[getIdx(IdxEntry::SPEC_DECODING_POSITION_OFFSETS)].dims.d[1] - 1;
        if (num_decoding_draft_tokens > 0)
        {
            // spec_decoding_* tensors are not filled for context requests. Hence, always strting from 0th index
            int32_t constexpr genSeqIdx = 0;
            spec_decoding_packed_mask = static_cast<int const*>(inputs[getIdx(IdxEntry::SPEC_DECODING_PACKED_MASK)])
                + genSeqIdx * getStride(inputDesc[getIdx(IdxEntry::SPEC_DECODING_PACKED_MASK)].dims, 0);
            // Packed as [num_tokens, packed_mask_size]
            // Use seqIdxBeg * (num_decoding_draft_tokens + 1) here as only generation tokens have the packed_mask
            // buffer.
            // TODO: support variable sequence length based on generationTokenIdxBeg.
            spec_decoding_packed_mask = static_cast<int const*>(inputs[getIdx(IdxEntry::SPEC_DECODING_PACKED_MASK)])
                + genSeqIdx * (num_decoding_draft_tokens + 1)
                    * getStride(inputDesc[getIdx(IdxEntry::SPEC_DECODING_PACKED_MASK)].dims, 0);
            spec_decoding_position_offsets
                = static_cast<int const*>(inputs[getIdx(IdxEntry::SPEC_DECODING_POSITION_OFFSETS)])
                + genSeqIdx * getStride(inputDesc[getIdx(IdxEntry::SPEC_DECODING_POSITION_OFFSETS)].dims, 0);
            spec_decoding_generation_lengths
                = static_cast<int const*>(inputs[getIdx(IdxEntry::SPEC_DECODING_GENERATION_LENGTHS)]) + genSeqIdx;
        }
    }

    int32_t const* max_context_kv_len_list = useKVCache()
        ? static_cast<int const*>(inputs[getIdx(IdxEntry::HOST_PAST_KEY_VALUE_LENGTHS)]) + seqIdxBeg
        : nullptr;
    int32_t const max_context_kv_len = useKVCache()
        ? *std::max_element(max_context_kv_len_list, max_context_kv_len_list + localNbSeq)
        : max_context_q_len;

    if (is_context) // context stage
    {
        int const batch_size = localNbSeq;
        int const request_batch_size = batch_size;
        // num of total tokens (without paddings when remove paddings).
        int num_encoder_tokens = 0;
        if (isCrossAttention())
        {
            if (!mRemovePadding)
            {
                num_encoder_tokens = request_batch_size * max_encoder_context_len;
            }
            else
            {
                num_encoder_tokens = inputDesc[getIdx(IdxEntry::CROSS_QKV)].dims.d[0];
            }
        }

        EnqueueContextParams<T, KVCacheBuffer> enqueue_params{attention_input, qkv_bias, rotary_cos_sin,
            max_context_q_len, max_context_kv_len, max_attention_window_size, cyclic_attention_window_size,
            sink_token_length, context_q_lengths, sequence_kv_length, kv_scale_orig_quant, kv_scale_quant_orig,
            attention_output_orig_quant, alibi_slopes, context_buf_, key_value_cache, block_offsets, host_block_offsets,
            host_primary_pool_pointer, host_secondary_pool_pointer, batch_size, localNbTokens, max_blocks_per_sequence,
            workspace};
        if (isRelativePosition())
        {
            enqueue_params.relative_attention_bias
                = static_cast<T const*>(inputs[getIdx(IdxEntry::RELATIVE_ATTENTION_BIAS)]);
            enqueue_params.relative_attention_bias_stride
                = inputDesc[getIdx(IdxEntry::RELATIVE_ATTENTION_BIAS)].dims.d[1]; // max_seq_len or num_buckets
        }
        if (isCrossAttention())
        {
            enqueue_params.cross_qkv = static_cast<T const*>(inputs[getIdx(IdxEntry::CROSS_QKV)]);
            enqueue_params.cross_qkv_length = max_encoder_context_len;
            enqueue_params.encoder_input_lengths
                = reinterpret_cast<int const*>(inputs[getIdx(IdxEntry::ENCODER_INPUT_LENGTH)]) + seqIdxBeg;
            enqueue_params.num_encoder_tokens = num_encoder_tokens;
        }

        enqueueContext<T, KVCacheBuffer>(enqueue_params, stream);

        {
            std::string const afterContexStr = "ctx attention at layer " + std::to_string(mLayerIdx);
            TLLM_CHECK_DEBUG_WITH_INFO(
                tensorrt_llm::runtime::utils::tensorHasNan(localNbTokens,
                    outputDesc[0].dims.d[getPackedTensorHiddenDimIndex(mRemovePadding)],
                    mFP8ContextFMHA ? nvinfer1::DataType::kFP8 : mType, context_buf_, stream, afterContexStr)
                    == false,
                "Found Nan in " + afterContexStr);
        }
    }
    else // generation stage; max_context_q_len == input_seq_len == 1
    {
        TLLM_CHECK_WITH_INFO(useKVCache(), "KV-cache-less is only supported for context");
        int batch_beam = localNbSeq;
        TLLM_CHECK(batch_beam % beamWidth == 0);
        int32_t const num_requests = batch_beam / beamWidth;

        int const* cache_indir
            = beamWidth == 1 ? nullptr : reinterpret_cast<int const*>(inputs[getIdx(IdxEntry::CACHE_INDIR)]);
        int const* host_context_lengths
            = mRemovePadding ? reinterpret_cast<int const*>(inputs[getIdx(IdxEntry::HOST_CONTEXT_LENGTH)]) : nullptr;

        // Medusa: the max input sequence length if variable sequence length is needed.
        int const input_seq_length = getGenerationInputSequenceLength(inputDesc, localNbSeq, localNbTokens);
        auto qkvDims = inputDesc[getIdx(IdxEntry::QKV_TENSOR)].dims;
        TLLM_CHECK_WITH_INFO(input_seq_length == 1 || mIsSpecDecodingEnabled,
            "Only speculative decoding mode supports input length > 1 in the generation phase, input_seq_length=%d, "
            "mIsSpecDecodingEnabled=%s, nDims=%d, (" FMT_DIM ", " FMT_DIM ", " FMT_DIM ")",
            input_seq_length, mIsSpecDecodingEnabled ? "true" : "false", qkvDims.nbDims, qkvDims.d[0], qkvDims.d[1],
            qkvDims.d[2]);
        TLLM_CHECK_WITH_INFO(
            input_seq_length == num_decoding_draft_tokens + 1, "The generation input length is not expected.");
        EnqueueGenerationParams<T, KVCacheBuffer> enqueue_params{attention_input, qkv_bias, input_seq_length,
            sequence_kv_length, max_context_kv_len, beamWidth, context_q_lengths, kv_scale_orig_quant,
            kv_scale_quant_orig, attention_output_orig_quant, rotary_embedding_scaling_factors, alibi_slopes,
            context_buf_, key_value_cache, block_offsets, host_primary_pool_pointer, host_secondary_pool_pointer,
            max_attention_window_size, cyclic_attention_window_size, sink_token_length, num_requests,
            max_blocks_per_sequence, cache_indir, mMultiBlockSemaphores.get(), workspace, max_context_kv_len_list};
        enqueue_params.host_context_lengths = host_context_lengths;
        if (isRelativePosition())
        {
            enqueue_params.relative_attention_bias
                = static_cast<T const*>(inputs[getIdx(IdxEntry::RELATIVE_ATTENTION_BIAS)]);
            enqueue_params.relative_attention_bias_stride
                = inputDesc[getIdx(IdxEntry::RELATIVE_ATTENTION_BIAS)].dims.d[1]; // max_seq_len or num_buckets
        }
        if (isCrossAttention())
        {
            enqueue_params.encoder_input_lengths
                = reinterpret_cast<int const*>(inputs[getIdx(IdxEntry::ENCODER_INPUT_LENGTH)]) + seqIdxBeg;
        }
        if (mIsSpecDecodingEnabled)
        {
            enqueue_params.spec_decoding_packed_mask = spec_decoding_packed_mask;
            enqueue_params.spec_decoding_position_offsets = spec_decoding_position_offsets;
            enqueue_params.spec_decoding_generation_lengths = spec_decoding_generation_lengths;
        }
        enqueue_params.total_num_input_tokens = localNbTokens;

        enqueueGeneration<T, KVCacheBuffer>(enqueue_params, stream);

        {
            std::string const afterGenStr = "gen attention at layer " + std::to_string(mLayerIdx);
            {
                TLLM_CHECK_DEBUG_WITH_INFO(
                    tensorrt_llm::runtime::utils::tensorHasNan(localNbTokens,
                        outputDesc[0].dims.d[getPackedTensorHiddenDimIndex(mRemovePadding)],
                        mFP8ContextFMHA ? nvinfer1::DataType::kFP8 : mType, context_buf_, stream, afterGenStr)
                        == false,
                    "Found Nan in " + afterGenStr);
            }
        }
    }

    return 0;
}

template <typename T, typename AttentionOutT>
int GPTAttentionPlugin::enqueueDispatchKVCacheType(nvinfer1::PluginTensorDesc const* inputDesc,
    nvinfer1::PluginTensorDesc const* outputDesc, void const* const* inputs, void* const* outputs, void* workspace,
    cudaStream_t stream)
{
    if (mPagedKVCache)
    {
        return enqueueImpl<T, AttentionOutT, KVBlockArray>(inputDesc, outputDesc, inputs, outputs, workspace, stream);
    }
    else
    {
        return enqueueImpl<T, AttentionOutT, KVLinearBuffer>(inputDesc, outputDesc, inputs, outputs, workspace, stream);
    }
    return 0;
}

int GPTAttentionPlugin::enqueue(nvinfer1::PluginTensorDesc const* inputDesc,
    nvinfer1::PluginTensorDesc const* outputDesc, void const* const* inputs, void* const* outputs, void* workspace,
    cudaStream_t stream) noexcept
{
    if (isBuilding())
    {
        return 0;
    }
    if (mType == nvinfer1::DataType::kHALF)
    {
#ifdef ENABLE_FP8
        if (mFP8ContextFMHA)
        {
            return enqueueDispatchKVCacheType<half, __nv_fp8_e4m3>(
                inputDesc, outputDesc, inputs, outputs, workspace, stream);
        }
#endif
        return enqueueDispatchKVCacheType<half>(inputDesc, outputDesc, inputs, outputs, workspace, stream);
    }
    else if (mType == nvinfer1::DataType::kFLOAT)
    {
        return enqueueDispatchKVCacheType<float>(inputDesc, outputDesc, inputs, outputs, workspace, stream);
    }
#ifdef ENABLE_BF16
    else if (mType == nvinfer1::DataType::kBF16)
    {
#ifdef ENABLE_FP8
        if (mFP8ContextFMHA)
        {
            return enqueueDispatchKVCacheType<__nv_bfloat16, __nv_fp8_e4m3>(
                inputDesc, outputDesc, inputs, outputs, workspace, stream);
        }
#endif
        return enqueueDispatchKVCacheType<__nv_bfloat16>(inputDesc, outputDesc, inputs, outputs, workspace, stream);
    }
#endif
    return 0;
}

// IPluginV2Ext Methods
nvinfer1::DataType GPTAttentionPlugin::getOutputDataType(
    int index, nvinfer1::DataType const* inputTypes, int nbInputs) const noexcept
{
    TLLM_CHECK(index == 0 || (!mPagedKVCache && index == 1));
    if (index == 0)
    {
        return mFP8ContextFMHA && mEnableContextFMHA ? nvinfer1::DataType::kFP8
                                                     : inputTypes[getIdx(IdxEntry::QKV_TENSOR)];
    }
    else
    {
        return inputTypes[getIdx(IdxEntry::PAST_KEY_VALUE)];
    }
}

// IPluginV2 Methods

char const* GPTAttentionPlugin::getPluginType() const noexcept
{
    return GPT_ATTENTION_PLUGIN_NAME;
}

char const* GPTAttentionPlugin::getPluginVersion() const noexcept
{
    return GPT_ATTENTION_PLUGIN_VERSION;
}

int GPTAttentionPlugin::getNbOutputs() const noexcept
{
    return (mPagedKVCache || !useKVCache()) ? 1 : 2;
}

size_t GPTAttentionPlugin::getSerializationSize() const noexcept
{
    return GPTAttentionPluginCommon::getCommonSerializationSize();
}

void GPTAttentionPlugin::serialize(void* buffer) const noexcept
{
    GPTAttentionPluginCommon::serializeCommon(buffer);
}

///////////////

GPTAttentionPluginCreator::GPTAttentionPluginCreator()
    : GPTAttentionPluginCreatorCommon()
{

    mPluginAttributes.emplace_back(PluginField("in_flight_batching", nullptr, PluginFieldType::kINT8, 0));
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

char const* GPTAttentionPluginCreator::getPluginName() const noexcept
{
    return GPT_ATTENTION_PLUGIN_NAME;
}

char const* GPTAttentionPluginCreator::getPluginVersion() const noexcept
{
    return GPT_ATTENTION_PLUGIN_VERSION;
}

PluginFieldCollection const* GPTAttentionPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2* GPTAttentionPluginCreator::createPlugin(char const* name, PluginFieldCollection const* fc) noexcept
{
    PluginFieldParser p{fc->nbFields, fc->fields};

    try
    {
        auto* obj = new GPTAttentionPlugin(p.getScalar<int32_t>("layer_idx").value(),
            p.getScalar<int32_t>("num_heads").value(), p.getScalar<int32_t>("vision_start").value(),
            p.getScalar<int32_t>("vision_length").value(), p.getScalar<int32_t>("num_kv_heads").value(),
            p.getScalar<int32_t>("head_size").value(), p.getScalar<int32_t>("unidirectional").value(),
            p.getScalar<float>("q_scaling").value(), p.getScalar<float>("qk_tanh_scale").value(),
            static_cast<PositionEmbeddingType>(p.getScalar<int8_t>("position_embedding_type").value()),
            p.getScalar<int32_t>("rotary_embedding_dim").value(), p.getScalar<float>("rotary_embedding_base").value(),
            static_cast<RotaryScalingType>(p.getScalar<int8_t>("rotary_embedding_scale_type").value()),
            p.getScalar<float>("rotary_embedding_scale").value(),
            p.getScalar<float>("rotary_embedding_short_m_scale").value(),
            p.getScalar<float>("rotary_embedding_long_m_scale").value(),
            p.getScalar<int32_t>("rotary_embedding_max_positions").value(),
            p.getScalar<int32_t>("rotary_embedding_original_max_positions").value(),
            static_cast<int32_t>(p.getScalar<int32_t>("tp_size").value()),
            static_cast<int32_t>(p.getScalar<int32_t>("tp_rank").value()),
            static_cast<bool>(p.getScalar<int8_t>("unfuse_qkv_gemm").value()),
            static_cast<ContextFMHAType>(p.getScalar<int8_t>("context_fmha_type").value()),
            static_cast<bool>(p.getScalar<int8_t>("multi_block_mode").value()),
            static_cast<bool>(p.getScalar<int8_t>("enable_xqa").value()),
            p.getScalar<int32_t>("kv_cache_quant_mode").value(),
            static_cast<bool>(p.getScalar<int8_t>("remove_input_padding").value()),
            static_cast<AttentionMaskType>(p.getScalar<int32_t>("mask_type").value()),
            BlockSparseParams{p.getScalar<int32_t>("block_sparse_block_size").value(),
                static_cast<bool>(p.getScalar<int8_t>("block_sparse_homo_head_pattern").value()),
                p.getScalar<int32_t>("block_sparse_num_local_blocks").value(),
                p.getScalar<int32_t>("block_sparse_vertical_stride").value()},
            static_cast<bool>(p.getScalar<int32_t>("paged_kv_cache").value()),
            p.getScalar<int32_t>("tokens_per_block").value(),
            static_cast<nvinfer1::DataType>(p.getScalar<int32_t>("type_id").value()),
            p.getScalar<int32_t>("max_context_length").value(),
            static_cast<bool>(p.getScalar<int8_t>("qkv_bias_enabled").value()),
            static_cast<bool>(p.getScalar<int8_t>("do_cross_attention").value()),
            static_cast<int32_t>(p.getScalar<int32_t>("max_distance").value()),
            static_cast<bool>(p.getScalar<int8_t>("pos_shift_enabled").value()),
            static_cast<bool>(p.getScalar<int8_t>("dense_context_fmha").value()),
            static_cast<bool>(p.getScalar<int8_t>("use_paged_context_fmha").value()),
            static_cast<bool>(p.getScalar<int8_t>("use_fp8_context_fmha").value()),
            static_cast<bool>(p.getScalar<int32_t>("use_cache").value()),
            static_cast<bool>(p.getScalar<int8_t>("is_spec_decoding_enabled").value()));
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2* GPTAttentionPluginCreator::deserializePlugin(
    char const* name, void const* serialData, size_t serialLength) noexcept
{
    // This object will be deleted when the network is destroyed, which will
    // call GPTAttentionPlugin::destroy()
    try
    {
        auto* obj = new GPTAttentionPlugin(serialData, serialLength);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}
