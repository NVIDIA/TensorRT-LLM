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
#include "tensorrt_llm/kernels/decoderMaskedMultiheadAttention/decoderXQARunner.h"
#include "tensorrt_llm/kernels/gptKernels.h"
#include <NvInferRuntimePlugin.h>
#include <cstdint>

using namespace nvinfer1;
using namespace tensorrt_llm::kernels;
namespace tc = tensorrt_llm::common;
using tensorrt_llm::plugins::GPTAttentionPluginCreatorCommon;
using tensorrt_llm::plugins::GPTAttentionPluginCommon;

GPTAttentionPluginCommon::GPTAttentionPluginCommon(int layer_idx, int num_heads, int vision_start, int vision_length,
    int num_kv_heads, int num_kv_heads_origin, int head_size, int unidirectional, float q_scaling,
    float attn_logit_softcapping_scale, tensorrt_llm::kernels::PositionEmbeddingType position_embedding_type,
    int rotary_embedding_dim, // for RoPE. Use 0 for non-RoPE
    float rotary_embedding_base, tensorrt_llm::kernels::RotaryScalingType rotary_embedding_scale_type,
    float rotary_embedding_scale, float rotary_embedding_short_m_scale, float rotary_embedding_long_m_scale,
    int rotary_embedding_max_positions, int rotary_embedding_original_max_positions, int tp_size,
    int tp_rank,           // for ALiBi
    bool unfuse_qkv_gemm,  // for AutoPP
    bool use_logn_scaling, // for LognScaling
    tensorrt_llm::kernels::ContextFMHAType context_fmha_type, int kv_cache_quant_mode, bool remove_input_padding,
    tensorrt_llm::kernels::AttentionMaskType mask_type, tensorrt_llm::kernels::BlockSparseParams block_sparse_params,
    bool paged_kv_cache, int tokens_per_block, nvinfer1::DataType type, int32_t max_context_length,
    bool qkv_bias_enabled, bool cross_attention, int max_distance, bool pos_shift_enabled, bool dense_context_fmha,
    bool use_paged_context_fmha, bool use_fp8_context_fmha, bool has_full_attention_mask, bool use_cache,
    bool is_spec_decoding_enabled, bool spec_decoding_is_generation_length_variable,
    int32_t spec_decoding_max_generation_length, bool is_mla_enabled, int q_lora_rank, int kv_lora_rank,
    int qk_nope_head_dim, int qk_rope_head_dim, int v_head_dim, bool fuse_fp4_quant, bool skip_attn, int cp_size,
    int cp_rank, std::set<int32_t> cp_group)
    : mResource{DecoderXQARunner::getResourceGlobal()}
{
    mLayerIdx = layer_idx;
    mNumHeads = num_heads;
    mVisionStart = vision_start;
    mVisionLength = vision_length;
    mNumKVHeads = num_kv_heads;
    mNumKVHeadsOrigin = num_kv_heads_origin;
    mHeadSize = head_size;
    mUnidirectional = unidirectional;
    mQScaling = q_scaling;
    mAttnLogitSoftcappingScale = attn_logit_softcapping_scale;
    mRotaryEmbeddingDim = rotary_embedding_dim;
    mRotaryEmbeddingBase = rotary_embedding_base;
    mRotaryEmbeddingScaleType = rotary_embedding_scale_type;
    mRotaryEmbeddingScale = rotary_embedding_scale;
    mRotaryEmbeddingShortMscale = rotary_embedding_short_m_scale;
    mRotaryEmbeddingLongMscale = rotary_embedding_long_m_scale;
    mRotaryEmbeddingMaxPositions = rotary_embedding_max_positions;
    mRotaryEmbeddingOriginalMaxPositions = rotary_embedding_original_max_positions;
    mPositionEmbeddingType = position_embedding_type;
    mEnableContextFMHA = context_fmha_type != ContextFMHAType::DISABLED;
    mFMHAForceFP32Acc = type == nvinfer1::DataType::kBF16;
    mMaskType = mask_type;
    mBlockSparseParams = block_sparse_params;
    mType = type;
    mMultiBlockMode = true;
    mEnableXQA = true;
    mKVCacheQuantMode = tc::QuantMode(kv_cache_quant_mode);
    mRemovePadding = remove_input_padding;
    mPagedKVCache = paged_kv_cache;
    mTokensPerBlock = tokens_per_block;
    mTpSize = tp_size;
    mTpRank = tp_rank;
    mUnfuseQkvGemm = unfuse_qkv_gemm;
    mUseLognScaling = use_logn_scaling;
    mMaxContextLength = max_context_length;
    mQKVBiasEnabled = qkv_bias_enabled;
    mCrossAttention = cross_attention;
    mMaxDistance = max_distance;
    mPosShiftEnabled = pos_shift_enabled;
    mDenseContextFMHA = dense_context_fmha;
    mPagedContextFMHA = use_paged_context_fmha;
    mFP8ContextFMHA = use_fp8_context_fmha;
    mFP8AttenOutput = use_fp8_context_fmha;
    mHasFullAttentionMask = has_full_attention_mask;
    mUseKVCache = use_cache;
    mIsSpecDecodingEnabled = is_spec_decoding_enabled;
    mSpecDecodingIsGenerationLengthVariable = spec_decoding_is_generation_length_variable;
    mSpecDecodingMaxGenerationLength = spec_decoding_max_generation_length;
    mIsMLAEnabled = is_mla_enabled;
    mMLAParams = {q_lora_rank, kv_lora_rank, qk_nope_head_dim, qk_rope_head_dim, v_head_dim};
    mCpSize = cp_size;
    mCpRank = cp_rank;
    mCpGroup = std::move(cp_group);
    mFuseFp4Quant = fuse_fp4_quant;
    mSkipAttn = skip_attn;
}

// Parameterized constructor
GPTAttentionPluginCommon::GPTAttentionPluginCommon(void const* data, size_t length)
    : mResource{DecoderXQARunner::getResourceGlobal()}
{
    char const *d = reinterpret_cast<char const*>(data), *a = d;
    unsigned int kvCacheQuantMode;

    read(d, mLayerIdx);
    read(d, mNumHeads);
    read(d, mVisionStart);
    read(d, mVisionLength);
    read(d, mNumKVHeads);
    read(d, mNumKVHeadsOrigin);
    read(d, mHeadSize);
    read(d, mUnidirectional);
    read(d, mQScaling);
    read(d, mAttnLogitSoftcappingScale);
    read(d, mPositionEmbeddingType);
    read(d, mRotaryEmbeddingDim);
    read(d, mRotaryEmbeddingBase);
    read(d, mRotaryEmbeddingScaleType);
    read(d, mRotaryEmbeddingScale);
    read(d, mRotaryEmbeddingShortMscale);
    read(d, mRotaryEmbeddingLongMscale);
    read(d, mRotaryEmbeddingMaxPositions);
    read(d, mRotaryEmbeddingOriginalMaxPositions);
    read(d, mTpSize);
    read(d, mTpRank);
    read(d, mUnfuseQkvGemm);
    read(d, mUseLognScaling);
    read(d, mEnableContextFMHA);
    read(d, mFMHAForceFP32Acc);
    read(d, mMultiBlockMode);
    read(d, mEnableXQA);
    read(d, kvCacheQuantMode);
    read(d, mRemovePadding);
    read(d, mMaskType);
    read(d, mBlockSparseParams);
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
    read(d, mFP8ContextFMHA);
    read(d, mFP8AttenOutput);
    read(d, mHasFullAttentionMask);
    read(d, mUseKVCache);
    read(d, mIsSpecDecodingEnabled);
    read(d, mUseSpecDecoding);
    read(d, mSpecDecodingIsGenerationLengthVariable);
    read(d, mSpecDecodingMaxGenerationLength);
    read(d, mIsMLAEnabled);
    read(d, mMLAParams);
    read(d, mNbMultiBlockSemaphores);
    read(d, mFuseFp4Quant);
    read(d, mSkipAttn);
    read(d, mCpSize);
    read(d, mCpRank);

    mKVCacheQuantMode = tc::QuantMode(kvCacheQuantMode);

    uint32_t decoderXQARunnerResourceSerializedSize;
    read(d, decoderXQARunnerResourceSerializedSize);
    mResource->merge(DecoderXQARunnerResource(d, decoderXQARunnerResourceSerializedSize), /*initialize=*/true);
    d += decoderXQARunnerResourceSerializedSize;

    mCpGroup.clear();
    int32_t groupItem = 0;
    while (d != a + length)
    {
        read(d, groupItem);
        mCpGroup.insert(groupItem);
    }
    TLLM_CHECK_WITH_INFO(d == a + length,
        "Expected length (%d) != real length (%d). This is often "
        "caused by using different TensorRT LLM version to build "
        "engine and run engine.",
        (int) length, (int) (d - a));
    TLLM_CHECK_WITH_INFO((smVersion() >= 80) || (mType != nvinfer1::DataType::kBF16),
        "Unsupported data type, pre SM 80 GPUs do not support bfloat16");
}

int GPTAttentionPluginCommon::initialize() noexcept
{
    return AttentionOp::initialize();
}

void GPTAttentionPluginCommon::destroy() noexcept
{
    delete this;
}

size_t GPTAttentionPluginCommon::getCommonSerializationSize() const noexcept
{
    return sizeof(mLayerIdx) + sizeof(mNumHeads) + +sizeof(mVisionStart) + sizeof(mVisionLength) + sizeof(mNumKVHeads)
        + sizeof(mNumKVHeadsOrigin) + sizeof(mHeadSize) + sizeof(mUnidirectional) + sizeof(mQScaling)
        + sizeof(mAttnLogitSoftcappingScale) + sizeof(mPositionEmbeddingType) + sizeof(mRotaryEmbeddingDim)
        + sizeof(mRotaryEmbeddingBase) + sizeof(mRotaryEmbeddingScaleType) + sizeof(mRotaryEmbeddingScale)
        + sizeof(mRotaryEmbeddingShortMscale) + sizeof(mRotaryEmbeddingLongMscale)
        + sizeof(mRotaryEmbeddingMaxPositions) + sizeof(mRotaryEmbeddingOriginalMaxPositions) + sizeof(mTpSize)
        + sizeof(mTpRank) + sizeof(mEnableContextFMHA) + sizeof(mFMHAForceFP32Acc) + sizeof(mMultiBlockMode)
        + sizeof(mEnableXQA) + sizeof(unsigned int) // mKVCacheQuantMode
        + sizeof(mRemovePadding) + sizeof(mMaskType) + sizeof(mBlockSparseParams) + sizeof(mPagedKVCache)
        + sizeof(mTokensPerBlock) + sizeof(mType) + sizeof(mMaxContextLength) + sizeof(mQKVBiasEnabled)
        + sizeof(mCrossAttention) + sizeof(mMaxDistance) + sizeof(mPosShiftEnabled) + sizeof(mDenseContextFMHA)
        + sizeof(mPagedContextFMHA) + sizeof(mFP8ContextFMHA) + sizeof(mFP8AttenOutput) + sizeof(mHasFullAttentionMask)
        + sizeof(mUseKVCache) + sizeof(mUnfuseQkvGemm) + sizeof(mUseLognScaling) + sizeof(mIsSpecDecodingEnabled)
        + sizeof(mUseSpecDecoding) + sizeof(mSpecDecodingIsGenerationLengthVariable)
        + sizeof(mSpecDecodingMaxGenerationLength) + sizeof(mNbMultiBlockSemaphores) + sizeof(mIsMLAEnabled)
        + sizeof(mMLAParams) + sizeof(mFuseFp4Quant) + sizeof(mSkipAttn)
        + sizeof(uint32_t) // size of DecoderXQARunnerResource buffer.
        + sizeof(mCpSize) + sizeof(mCpRank) + sizeof(int32_t) * mCpGroup.size() + mResource->getSerializationSize();
}

void GPTAttentionPluginCommon::serializeCommon(void* buffer) const noexcept
{
    char *d = static_cast<char*>(buffer), *a = d;
    write(d, mLayerIdx);
    write(d, mNumHeads);
    write(d, mVisionStart);
    write(d, mVisionLength);
    write(d, mNumKVHeads);
    write(d, mNumKVHeadsOrigin);
    write(d, mHeadSize);
    write(d, mUnidirectional);
    write(d, mQScaling);
    write(d, mAttnLogitSoftcappingScale);
    write(d, mPositionEmbeddingType);
    write(d, mRotaryEmbeddingDim);
    write(d, mRotaryEmbeddingBase);
    write(d, mRotaryEmbeddingScaleType);
    write(d, mRotaryEmbeddingScale);
    write(d, mRotaryEmbeddingShortMscale);
    write(d, mRotaryEmbeddingLongMscale);
    write(d, mRotaryEmbeddingMaxPositions);
    write(d, mRotaryEmbeddingOriginalMaxPositions);
    write(d, mTpSize);
    write(d, mTpRank);
    write(d, mUnfuseQkvGemm);
    write(d, mUseLognScaling);
    write(d, mEnableContextFMHA);
    write(d, mFMHAForceFP32Acc);
    write(d, mMultiBlockMode);
    write(d, mEnableXQA);
    write(d, mKVCacheQuantMode.value());
    write(d, mRemovePadding);
    write(d, mMaskType);
    write(d, mBlockSparseParams);
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
    write(d, mFP8ContextFMHA);
    write(d, mFP8AttenOutput);
    write(d, mHasFullAttentionMask);
    write(d, mUseKVCache);
    write(d, mIsSpecDecodingEnabled);
    write(d, mUseSpecDecoding);
    write(d, mSpecDecodingIsGenerationLengthVariable);
    write(d, mSpecDecodingMaxGenerationLength);
    write(d, mIsMLAEnabled);
    write(d, mMLAParams);
    write(d, mNbMultiBlockSemaphores);
    write(d, mFuseFp4Quant);
    write(d, mSkipAttn);
    write(d, mCpSize);
    write(d, mCpRank);

    // An uint32_t that specifies the size of the serialized buffer, followed by the actual content.
    uint32_t decoderXQARunnerResourceSerializedSize = mResource->getSerializationSize();
    write(d, decoderXQARunnerResourceSerializedSize);
    mResource->serialize(d, decoderXQARunnerResourceSerializedSize);
    d += decoderXQARunnerResourceSerializedSize;

    for (auto it = mCpGroup.begin(); it != mCpGroup.end(); ++it)
    {
        write(d, *it);
    }
    TLLM_CHECK(d == a + getCommonSerializationSize());
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
    mPluginAttributes.emplace_back(PluginField("layer_idx", nullptr, PluginFieldType::kINT32));
    mPluginAttributes.emplace_back(PluginField("num_heads", nullptr, PluginFieldType::kINT32));
    mPluginAttributes.emplace_back(PluginField("vision_start", nullptr, PluginFieldType::kINT32));
    mPluginAttributes.emplace_back(PluginField("vision_length", nullptr, PluginFieldType::kINT32));
    mPluginAttributes.emplace_back(PluginField("num_kv_heads", nullptr, PluginFieldType::kINT32));
    mPluginAttributes.emplace_back(PluginField("num_kv_heads_origin", nullptr, PluginFieldType::kINT32));
    mPluginAttributes.emplace_back(PluginField("layer_idx_in_cache_pool", nullptr, PluginFieldType::kINT32));
    mPluginAttributes.emplace_back(PluginField("head_size", nullptr, PluginFieldType::kINT32));
    mPluginAttributes.emplace_back(PluginField("unidirectional", nullptr, PluginFieldType::kINT32));
    mPluginAttributes.emplace_back(PluginField("q_scaling", nullptr, PluginFieldType::kFLOAT32));
    mPluginAttributes.emplace_back(PluginField("attn_logit_softcapping_scale", nullptr, PluginFieldType::kFLOAT32));
    mPluginAttributes.emplace_back(PluginField("position_embedding_type", nullptr, PluginFieldType::kINT8));
    mPluginAttributes.emplace_back(PluginField("rotary_embedding_dim", nullptr, PluginFieldType::kINT32));
    mPluginAttributes.emplace_back(PluginField("rotary_embedding_base", nullptr, PluginFieldType::kFLOAT32));
    mPluginAttributes.emplace_back(PluginField("rotary_embedding_scale_type", nullptr, PluginFieldType::kINT8));
    mPluginAttributes.emplace_back(PluginField("rotary_embedding_scale", nullptr, PluginFieldType::kFLOAT32));
    mPluginAttributes.emplace_back(PluginField("rotary_embedding_short_m_scale", nullptr, PluginFieldType::kFLOAT32));
    mPluginAttributes.emplace_back(PluginField("rotary_embedding_long_m_scale", nullptr, PluginFieldType::kFLOAT32));
    mPluginAttributes.emplace_back(PluginField("rotary_embedding_max_positions", nullptr, PluginFieldType::kINT32));
    mPluginAttributes.emplace_back(
        PluginField("rotary_embedding_original_max_positions", nullptr, PluginFieldType::kINT32));
    mPluginAttributes.emplace_back(PluginField("tp_size", nullptr, PluginFieldType::kINT32));
    mPluginAttributes.emplace_back(PluginField("tp_rank", nullptr, PluginFieldType::kINT32));
    mPluginAttributes.emplace_back(PluginField("unfuse_qkv_gemm", nullptr, PluginFieldType::kINT8));
    mPluginAttributes.emplace_back(PluginField("use_logn_scaling", nullptr, PluginFieldType::kINT8));
    mPluginAttributes.emplace_back(PluginField("context_fmha_type", nullptr, PluginFieldType::kINT8));
    mPluginAttributes.emplace_back(PluginField("kv_cache_quant_mode", nullptr, PluginFieldType::kINT32));
    mPluginAttributes.emplace_back(PluginField("remove_input_padding", nullptr, PluginFieldType::kINT8));
    mPluginAttributes.emplace_back(PluginField("mask_type", nullptr, PluginFieldType::kINT32));
    mPluginAttributes.emplace_back(PluginField("block_sparse_block_size", nullptr, PluginFieldType::kINT32));
    mPluginAttributes.emplace_back(PluginField("block_sparse_homo_head_pattern", nullptr, PluginFieldType::kINT32));
    mPluginAttributes.emplace_back(PluginField("block_sparse_num_local_blocks", nullptr, PluginFieldType::kINT32));
    mPluginAttributes.emplace_back(PluginField("block_sparse_vertical_stride", nullptr, PluginFieldType::kINT32));
    mPluginAttributes.emplace_back(PluginField("paged_kv_cache", nullptr, PluginFieldType::kINT32));
    mPluginAttributes.emplace_back(PluginField("tokens_per_block", nullptr, PluginFieldType::kINT32));
    mPluginAttributes.emplace_back(PluginField("type_id", nullptr, PluginFieldType::kINT32));
    mPluginAttributes.emplace_back(PluginField("max_context_length", nullptr, PluginFieldType::kINT32));
    mPluginAttributes.emplace_back(PluginField("qkv_bias_enabled", nullptr, PluginFieldType::kINT8));
    mPluginAttributes.emplace_back(PluginField("do_cross_attention", nullptr, PluginFieldType::kINT8));
    mPluginAttributes.emplace_back(PluginField("max_distance", nullptr, PluginFieldType::kINT32));
    mPluginAttributes.emplace_back(PluginField("pos_shift_enabled", nullptr, PluginFieldType::kINT8));
    mPluginAttributes.emplace_back(PluginField("dense_context_fmha", nullptr, PluginFieldType::kINT8));
    mPluginAttributes.emplace_back(PluginField("use_paged_context_fmha", nullptr, PluginFieldType::kINT8));
    mPluginAttributes.emplace_back(PluginField("use_fp8_context_fmha", nullptr, PluginFieldType::kINT8));
    mPluginAttributes.emplace_back(PluginField("has_full_attention_mask", nullptr, PluginFieldType::kINT8));
    mPluginAttributes.emplace_back(PluginField("use_cache", nullptr, PluginFieldType::kINT32));
    mPluginAttributes.emplace_back(PluginField("is_spec_decoding_enabled", nullptr, PluginFieldType::kINT8));
    mPluginAttributes.emplace_back(
        PluginField("spec_decoding_is_generation_length_variable", nullptr, PluginFieldType::kINT8));
    mPluginAttributes.emplace_back(
        PluginField("spec_decoding_max_generation_length", nullptr, PluginFieldType::kINT32));
    mPluginAttributes.emplace_back(PluginField("is_mla_enabled", nullptr, PluginFieldType::kINT8));
    mPluginAttributes.emplace_back(PluginField("q_lora_rank", nullptr, PluginFieldType::kINT32));
    mPluginAttributes.emplace_back(PluginField("kv_lora_rank", nullptr, PluginFieldType::kINT32));
    mPluginAttributes.emplace_back(PluginField("qk_nope_head_dim", nullptr, PluginFieldType::kINT32));
    mPluginAttributes.emplace_back(PluginField("qk_rope_head_dim", nullptr, PluginFieldType::kINT32));
    mPluginAttributes.emplace_back(PluginField("v_head_dim", nullptr, PluginFieldType::kINT32));
    mPluginAttributes.emplace_back(PluginField("fuse_fp4_quant", nullptr, PluginFieldType::kINT8));
    mPluginAttributes.emplace_back(PluginField("skip_attn", nullptr, PluginFieldType::kINT8));
    mPluginAttributes.emplace_back(PluginField("cp_size", nullptr, PluginFieldType::kINT32));
    mPluginAttributes.emplace_back(PluginField("cp_rank", nullptr, PluginFieldType::kINT32));
    mPluginAttributes.emplace_back(PluginField("cp_group", nullptr, PluginFieldType::kINT32));
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

PluginFieldCollection const* GPTAttentionPluginCreatorCommon::getFieldNames() noexcept
{
    return &mFC;
}
