/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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
#pragma once

#include "tensorrt_llm/common/cublasMMWrapper.h"
#include "tensorrt_llm/common/quantization.h"
#include "tensorrt_llm/kernels/contextFusedMultiHeadAttention/fmhaRunner.h"
#include "tensorrt_llm/kernels/contextFusedMultiHeadAttention/fused_multihead_attention_common.h"
#include "tensorrt_llm/kernels/decoderMaskedMultiheadAttention/decoderXQARunner.h"
#include "tensorrt_llm/kernels/gptKernels.h"
#include "tensorrt_llm/kernels/kvCacheUtils.h"
#include "tensorrt_llm/plugins/common/plugin.h"
#include <cassert>
#include <set>
#include <string>
#include <vector>

namespace tensorrt_llm::plugins
{

class GPTAttentionPluginCommon : public BasePlugin
{
public:
    GPTAttentionPluginCommon() = delete;

    GPTAttentionPluginCommon(int layer_idx, int num_heads, int vision_start, int vision_length, int num_kv_heads,
        int head_size, int unidirectional, float q_scaling, float qk_tanh_scale,
        tensorrt_llm::kernels::PositionEmbeddingType position_embedding_type,
        int rotary_embedding_dim, // for RoPE. Use 0 for non-RoPE
        float rotary_embedding_base, tensorrt_llm::kernels::RotaryScalingType rotary_embedding_scale_type,
        float rotary_embedding_scale, float rotary_embedding_short_m_scale, float rotary_embedding_long_m_scale,
        int rotary_embedding_max_positions, int rotary_embedding_original_max_positions, int tp_size,
        int tp_rank,          // for ALiBi
        bool unfuse_qkv_gemm, // for AutoPP
        tensorrt_llm::kernels::ContextFMHAType context_fmha_type, bool multi_block_mode, bool enable_xqa,
        int kv_cache_quant_mode, bool remove_input_padding, tensorrt_llm::kernels::AttentionMaskType mask_type,
        tensorrt_llm::kernels::BlockSparseParams block_sparse_params, bool paged_kv_cache, int tokens_per_block,
        nvinfer1::DataType type, int32_t max_context_length, bool qkv_bias_enabled, bool cross_attention = false,
        int max_distance = 0, bool pos_shift_enabled = false, bool dense_context_fmha = false,
        bool use_paged_context_fmha = false, bool use_fp8_context_fmha = false, bool use_cache = true,
        bool is_spec_decoding_enabled = false);

    GPTAttentionPluginCommon(void const* data, size_t length);

    ~GPTAttentionPluginCommon() override = default;

    template <typename T>
    int enqueueImpl(nvinfer1::PluginTensorDesc const* inputDesc, nvinfer1::PluginTensorDesc const* outputDesc,
        void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream);

    //! This is called on every trt Engine creation
    int initialize() noexcept override;
    //! This is called on every trt Engine destroy
    void terminate() noexcept override;

    //! This is called on every trt ExecutionContext creation by TRT
    //! Note TRT does not call the initialize on cloned plugin, so clone internally should do initialization.
    template <typename T>
    T* cloneImpl() const noexcept;

    //! This is called on evert trt Engine or ExecutionContext destroy.
    //! None-cloned plugins will call terminate and then call destroy, while the cloned plugins will call destroy only
    //! So plugin should put the resource release inside destroy.
    void destroy() noexcept override;

    size_t getCommonSerializationSize() const noexcept;
    void serializeCommon(void* buffer) const noexcept;
    int getHeadSize(bool checkInit = true) const;

protected:
    int getMaxNumSeqLenTile(int batch_beam_size = 1) const;
    size_t getWorkspaceSizeForContext(nvinfer1::DataType type, int32_t nbReq, int32_t max_input_length,
        int32_t cross_qkv_length = 0, int32_t max_num_tokens = 0) const noexcept;
    // total_num_seq is the sum of beam_width for multiple requests
    size_t getWorkspaceSizeForGeneration(nvinfer1::DataType type, int32_t total_num_seq, int32_t max_kv_cache_length,
        int32_t max_num_tokens) const noexcept;

    template <typename T, typename KVCacheBuffer>
    struct EnqueueContextParams
    {
        T const* attention_input;
        T const* qkv_bias;
        // Rotary cos sin cache buffer to avoid re-computing.
        float2 const* rotary_cos_sin;
        int32_t input_seq_length; // padded input length
        int32_t max_past_kv_len;
        // By default, max_attention_window == cyclic_attention_window_size
        // unless each layer has different cyclic kv cache length.
        // Max cache capacity (used to allocate KV cache)
        int32_t max_attention_window;
        // Cyclic kv cache capacity (used to get the cyclic kv cache position for new tokens)
        int32_t cyclic_attention_window_size;
        int32_t sink_token_length;
        int32_t const* q_seq_lengths;
        int32_t const* kv_seq_lengths;
        float const* kv_scale_orig_quant;
        float const* kv_scale_quant_orig;
        float const* attention_output_orig_quant;
        T const* alibi_slopes;
        void* context_buf;
        void* key_value_cache;
        kernels::KVBlockArray::DataType* block_offsets;
        kernels::KVBlockArray::DataType* host_block_offsets;
        void* host_primary_pool_pointer;
        void* host_secondary_pool_pointer;
        int32_t batch_size;
        int32_t num_tokens;
        int32_t max_blocks_per_sequence;
        void* workspace;
        // optional when relative position
        T const* relative_attention_bias = nullptr;
        int relative_attention_bias_stride = 0;
        // optional when cross attention
        T const* cross_qkv = nullptr;
        int32_t cross_qkv_length = 0;
        int32_t const* encoder_input_lengths = nullptr;
        int32_t num_encoder_tokens = 0;

        std::string enqueueContextParamsToString() const
        {
            // variables from the params coming from the runtime
            std::stringstream ss;
            ss << "EnqueueContextParams ====================" << std::endl;

            ss << "attention_input: " << attention_input << std::endl;
            ss << "qkv_bias: " << qkv_bias << std::endl;
            ss << "rotary_cos_sin: " << rotary_cos_sin << std::endl;
            ss << "input_seq_length: " << input_seq_length << std::endl;
            ss << "max_past_kv_len: " << max_past_kv_len << std::endl;
            ss << "max_attention_window: " << max_attention_window << std::endl;
            ss << "cyclic_attention_window_size: " << cyclic_attention_window_size << std::endl;
            ss << "sink_token_length: " << sink_token_length << std::endl;
            ss << "q_seq_lengths: "
               << *(runtime::ITensor::wrap(
                      (void*) q_seq_lengths, nvinfer1::DataType::kINT32, runtime::ITensor::makeShape({batch_size})))
               << std::endl;
            ss << "kv_seq_lengths: "
               << *(runtime::ITensor::wrap(
                      (void*) kv_seq_lengths, nvinfer1::DataType::kINT32, runtime::ITensor::makeShape({batch_size})))
               << std::endl;
            ss << "kv_scale_orig_quant: " << kv_scale_orig_quant << std::endl;
            ss << "kv_scale_quant_orig: " << kv_scale_quant_orig << std::endl;
            ss << "attention_output_orig_quant: " << attention_output_orig_quant << std::endl;
            ss << "alibi_slopes: " << alibi_slopes << std::endl;
            ss << "context_buf: " << context_buf << std::endl;
            ss << "key_value_cache: " << (half*) key_value_cache << std::endl;
            ss << "block_offsets: " << block_offsets << std::endl;
            ss << "host_block_offsets: " << host_block_offsets << std::endl;
            ss << "host_primary_pool_pointer: " << host_primary_pool_pointer << std::endl;
            ss << "host_secondary_pool_pointer: " << host_secondary_pool_pointer << std::endl;
            ss << "batch_size: " << batch_size << std::endl;
            ss << "num_tokens: " << num_tokens << std::endl;
            ss << "max_blocks_per_sequence: " << max_blocks_per_sequence << std::endl;
            ss << "workspace: " << workspace << std::endl;
            ss << "relative_attention_bias: " << relative_attention_bias << std::endl;
            ss << "relative_attention_bias_stride: " << relative_attention_bias_stride << std::endl;
            ss << "cross_qkv: " << cross_qkv << std::endl;
            ss << "cross_qkv_length: " << cross_qkv_length << std::endl;
            ss << "encoder_input_lengths: " << encoder_input_lengths << std::endl;
            ss << "num_encoder_tokens: " << num_encoder_tokens << std::endl;
            return ss.str();
        }
    };

    template <typename T, typename KVCacheBuffer>
    int enqueueContext(EnqueueContextParams<T, KVCacheBuffer> const& params, cudaStream_t stream);

    template <typename T, typename KVCacheBuffer>
    struct EnqueueGenerationParams
    {
        T const* attention_input;
        T const* qkv_bias;
        // NOTE: input_seq_length might be larger than one in the medusa mode.
        int32_t input_seq_length;
        int32_t const* sequence_lengths;
        int32_t max_past_kv_length;
        int32_t beam_width;
        int32_t const* context_lengths;
        float const* kv_scale_orig_quant;
        float const* kv_scale_quant_orig;
        float const* attention_output_orig_quant;
        float const* rotary_embedding_scaling_factors;
        T const* alibi_slopes;
        void* context_buf;
        void* key_value_cache;
        kernels::KVBlockArray::DataType* block_offsets;
        void* host_primary_pool_pointer;
        void* host_secondary_pool_pointer;
        // By default, max_attention_window == cyclic_attention_window_size
        // unless each layer has different cyclic kv cache length.
        // Max cache capacity (used to allocate KV cache)
        int32_t max_attention_window;
        // Cyclic kv cache capacity (used to get the cyclic kv cache position for new tokens)
        int32_t cyclic_attention_window_size;
        int32_t sink_token_length;
        int32_t num_requests;
        int32_t max_blocks_per_sequence;
        int32_t const* cache_indir;
        int32_t* semaphores;
        void* workspace;
        int32_t const* host_past_key_value_lengths;
        // optional when relative position
        T const* relative_attention_bias = nullptr;
        int relative_attention_bias_stride = 0;
        // optional when cross attention
        int32_t const* encoder_input_lengths = nullptr;
        int32_t const* host_context_lengths = nullptr;
        // optional when speculative decoding is used.
        bool const* spec_decoding_mask = nullptr;
        int32_t const* spec_decoding_packed_mask = nullptr;
        int32_t const* spec_decoding_position_offsets = nullptr;
        int32_t const* spec_decoding_generation_lengths = nullptr;
        int32_t total_num_input_tokens;
    };

    template <typename T, typename KVCacheBuffer>
    int enqueueGeneration(EnqueueGenerationParams<T, KVCacheBuffer> const& params, cudaStream_t stream);

    // Called in configurePlugin().
    template <typename T, typename KVCacheBuffer>
    void prepareEnqueueGeneration(EnqueueGenerationParams<T, KVCacheBuffer> const& params);

    template <typename T, typename KVCacheBuffer>
    bool convertMMHAParamsToXQAParams(tensorrt_llm::kernels::XQAParams& xqaParams,
        EnqueueGenerationParams<T, KVCacheBuffer> const& generationsParams, bool forConfigurePlugin);

    bool isRelativePosition() const
    {
        return mPositionEmbeddingType == tensorrt_llm::kernels::PositionEmbeddingType::kRELATIVE;
    }

    bool isALiBi() const
    {
        return mPositionEmbeddingType == tensorrt_llm::kernels::PositionEmbeddingType::kALIBI
            || mPositionEmbeddingType == tensorrt_llm::kernels::PositionEmbeddingType::kALIBI_WITH_SCALE;
    }

    bool isAliBiWithScale() const
    {
        return mPositionEmbeddingType == tensorrt_llm::kernels::PositionEmbeddingType::kALIBI_WITH_SCALE;
    }

    bool isRoPE() const
    {
        return mPositionEmbeddingType == tensorrt_llm::kernels::PositionEmbeddingType::kROPE_GPTJ
            || mPositionEmbeddingType == tensorrt_llm::kernels::PositionEmbeddingType::kROPE_GPT_NEOX
            || mPositionEmbeddingType == tensorrt_llm::kernels::PositionEmbeddingType::kLONG_ROPE;
    }

    bool isLongRoPE() const
    {
        return mPositionEmbeddingType == tensorrt_llm::kernels::PositionEmbeddingType::kLONG_ROPE;
    }

    bool isCrossAttention() const
    {
        return mCrossAttention;
    }

    bool useKVCache() const
    {
        return mUseKVCache;
    }

    void reserveSemaphoreArray(int32_t size);

    void debugCheckSemaphores(cudaStream_t stream);

protected:
    static constexpr int kReservedMaxSeqLenTilePerSeq = 64;

    const std::string mLayerName;

    int mLayerIdx;
    int mNumHeads;
    int mVisionStart;
    int mVisionLength;
    int mNumKVHeads;
    int mHeadSize;
    int mUnidirectional;
    float mQScaling;
    float mQKTanhScale;
    int mRotaryEmbeddingDim;
    float mRotaryEmbeddingBase;
    tensorrt_llm::kernels::RotaryScalingType mRotaryEmbeddingScaleType;
    float mRotaryEmbeddingScale;
    float mRotaryEmbeddingShortMscale;
    float mRotaryEmbeddingLongMscale;
    int mRotaryEmbeddingMaxPositions;
    int mRotaryEmbeddingOriginalMaxPositions;
    tensorrt_llm::kernels::PositionEmbeddingType mPositionEmbeddingType;
    bool mRemovePadding = false;
    tensorrt_llm::kernels::AttentionMaskType mMaskType;
    tensorrt_llm::kernels::BlockSparseParams mBlockSparseParams;

    // NOTE: default values for paged kv cache.
    bool mPagedKVCache = false;
    int mTokensPerBlock = 0;
    tensorrt_llm::common::QuantMode mKVCacheQuantMode;
    int mTpSize = 1;
    int mTpRank = 0;
    bool mUnfuseQkvGemm = false;
    nvinfer1::DataType mType;
    int32_t mMaxContextLength;
    bool mQKVBiasEnabled;
    bool mCrossAttention = false;
    int mMaxDistance = 0;
    bool mPosShiftEnabled = false;
    bool mPagedContextFMHA = false;
    bool mFP8ContextFMHA = false;
    bool mDenseContextFMHA = false;
    bool mIsSpecDecodingEnabled = false;

    // Speculative decoding packed mask.
    uint4* mSpecDecodingPackedMask;
    uint4* mSpecDecodingPackedHostMask;

    // fmha runner (disable by default)
    // flag: disabled = 0, enabled = 1, enabled with fp32 accumulation = 2
    bool mEnableContextFMHA = false;
    bool mFMHAForceFP32Acc = false;
    int mSM = tensorrt_llm::common::getSMVersion();
    int mMultiProcessorCount = tensorrt_llm::common::getMultiProcessorCount();
    int mMaxSharedMemoryPerBlockOptin = tensorrt_llm::common::getMaxSharedMemoryPerBlockOptin();
    // The default copy constructor will leave it as nullptr. clone() shall initialize it.
    std::shared_ptr<CUDADriverWrapper> mDriver;
    UniqPtrWNullCopy<tensorrt_llm::kernels::MHARunner> mFMHARunner;
    UniqPtrWNullCopy<tensorrt_llm::kernels::DecoderXQARunner> mDecoderXQARunner;

    bool mMultiBlockMode;
    bool mEnableXQA;
    int mDeviceId = -1;
    static bool mForceMultiBlockWarned;
    // The default copy constructor will leave it as nullptr. clone() shall initialize it.
    UniqPtrWNullCopy<tensorrt_llm::common::CublasMMWrapper> mCublasWrapper;
    bool mUseKVCache = true;

    // This is implementation details which we want to save when serializing, but not expose as
    // a plugin field or a constructor parameter
    int32_t mNbMultiBlockSemaphores = 0;

    struct Deleter
    {
        void operator()(void* ptr)
        {
            cudaFree(ptr);
        }
    };

    UniqPtrWNullCopy<int32_t[], Deleter> mMultiBlockSemaphores = {};

    std::string toString() const
    {
        // member variables
        std::stringstream ss;
        ss << "gptAttentionCommon members ====================" << std::endl;
        ss << "mNumHeads: " << mNumHeads << std::endl;
        ss << "mNumKVHeads: " << mNumKVHeads << std::endl;
        ss << "mHeadSize: " << mHeadSize << std::endl;
        ss << "mUnidirectional: " << mUnidirectional << std::endl;
        ss << "mQScaling: " << mQScaling << std::endl;
        ss << "mRotaryEmbeddingDim: " << mRotaryEmbeddingDim << std::endl;
        ss << "mRotaryEmbeddingBase: " << mRotaryEmbeddingBase << std::endl;
        ss << "mRotaryEmbeddingScaleType: " << static_cast<int>(mRotaryEmbeddingScaleType) << std::endl;
        ss << "mRotaryEmbeddingScale: " << mRotaryEmbeddingScale << std::endl;
        ss << "mRotaryEmbeddingMaxPositions: " << mRotaryEmbeddingMaxPositions << std::endl;
        ss << "mPositionEmbeddingType: " << static_cast<int>(mPositionEmbeddingType) << std::endl;
        ss << "mRemovePadding: " << std::boolalpha << mRemovePadding << std::endl;
        ss << "mMaskType: " << static_cast<int>(mMaskType) << std::endl;
        ss << "mPagedKVCache: " << std::boolalpha << mPagedKVCache << std::endl;
        ss << "mTokensPerBlock: " << mTokensPerBlock << std::endl;
        ss << "mKVCacheQuantMode: " << static_cast<int>(mKVCacheQuantMode.value()) << std::endl;
        ss << "mTpSize: " << mTpSize << std::endl;
        ss << "mTpRank: " << mTpRank << std::endl;
        ss << "mUnfuseQkvGemm: " << std::boolalpha << mUnfuseQkvGemm << std::endl;
        ss << "mType: " << static_cast<int>(mType) << std::endl;
        ss << "mMaxContextLength: " << mMaxContextLength << std::endl;
        ss << "mQKVBiasEnabled: " << std::boolalpha << mQKVBiasEnabled << std::endl;
        ss << "mCrossAttention: " << std::boolalpha << mCrossAttention << std::endl;
        ss << "mMaxDistance: " << mMaxDistance << std::endl;
        ss << "mPosShiftEnabled: " << std::boolalpha << mPosShiftEnabled << std::endl;
        ss << "mPagedContextFMHA: " << std::boolalpha << mPagedContextFMHA << std::endl;
        ss << "mFP8ContextFMHA: " << std::boolalpha << mFP8ContextFMHA << std::endl;
        ss << "mDenseContextFMHA: " << std::boolalpha << mDenseContextFMHA << std::endl;
        ss << "mEnableContextFMHA: " << std::boolalpha << mEnableContextFMHA << std::endl;
        ss << "mFMHAForceFP32Acc: " << std::boolalpha << mFMHAForceFP32Acc << std::endl;
        ss << "mSM: " << mSM << std::endl;
        ss << "mMultiProcessorCount: " << mMultiProcessorCount << std::endl;
        ss << "mMaxSharedMemoryPerBlockOptin: " << mMaxSharedMemoryPerBlockOptin << std::endl;
        ss << "mMultiBlockMode: " << std::boolalpha << mMultiBlockMode << std::endl;
        ss << "mEnableXQA: " << std::boolalpha << mEnableXQA << std::endl;
        ss << "mDeviceId: " << mDeviceId << std::endl;
        ss << "mUseKVCache: " << std::boolalpha << mUseKVCache << std::endl;
        ss << "mForceMultiBlockWarned: " << mForceMultiBlockWarned << std::endl;

        return ss.str();
    }
};

class GPTAttentionPluginCreatorCommon : public BaseCreator
{
public:
    GPTAttentionPluginCreatorCommon();

    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override;

    template <typename T>
    T* deserializePluginImpl(char const* name, void const* serialData, size_t serialLength) noexcept;

protected:
    std::vector<nvinfer1::PluginField> mPluginAttributes;
    nvinfer1::PluginFieldCollection mFC{};
};

} // namespace tensorrt_llm::plugins
