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
#include "tensorrt_llm/common/opUtils.h"
#include "tensorrt_llm/common/quantization.h"
#include "tensorrt_llm/kernels/contextFusedMultiHeadAttention/fused_multihead_attention_common.h"
#include "tensorrt_llm/kernels/decoderMaskedMultiheadAttention/decoderXQARunner.h"
#include "tensorrt_llm/kernels/fmhaDispatcher.h"
#include "tensorrt_llm/kernels/gptKernels.h"
#include "tensorrt_llm/kernels/kvCacheUtils.h"
#include "tensorrt_llm/kernels/mlaKernels.h"
#include "tensorrt_llm/kernels/xqaDispatcher.h"
#include <cassert>
#include <set>
#include <string>
#include <vector>
#if ENABLE_MULTI_DEVICE
#include <nccl.h>
#endif // ENABLE_MULTI_DEVICE

namespace tensorrt_llm::common::op
{
using tensorrt_llm::kernels::RotaryScalingType;
using tensorrt_llm::kernels::PositionEmbeddingType;
using tensorrt_llm::kernels::AttentionMaskType;

class AttentionOp
{
public:
    AttentionOp(){};
    AttentionOp(AttentionOp const&) = default;
    AttentionOp& operator=(AttentionOp const&) = default;
    AttentionOp(AttentionOp&&) = default;
    AttentionOp& operator=(AttentionOp&&) = default;

    ~AttentionOp() = default;

    int initialize() noexcept;
    int getHeadSize(bool checkInit = true) const;
    int getMaxNumSeqLenTile(int batch_beam_size = 1) const;
    size_t getWorkspaceSizeForContext(nvinfer1::DataType type, int32_t nbReq, int32_t max_input_length,
        int32_t cross_kv_length = 0, int32_t max_num_tokens = 0) const noexcept;
    // total_num_seq is the sum of beam_width for multiple requests
    size_t getWorkspaceSizeForGeneration(nvinfer1::DataType type, int32_t total_num_seq, int32_t max_kv_cache_length,
        int32_t max_num_tokens) const noexcept;

    size_t getWorkspaceSizeForMLAPreProcess(
        nvinfer1::DataType type, size_t& remaining_size, int32_t total_token_length, int32_t rope_dim) const noexcept;

    template <typename T>
    struct EnqueueContextParams
    {
        T const* attention_input = nullptr;
        T const* qkv_bias = nullptr;
        // Attention mask input.
        bool const* attention_mask = nullptr;
        // Attention packed mask input (used by context FMHA).
        uint32_t const* attention_packed_mask = nullptr;
        // Rotary inv_freq cache buffer to avoid re-computing.
        float const* rotary_inv_freq = nullptr;
        // Rotary cos sin cache buffer to avoid re-computing.
        float2 const* rotary_cos_sin = nullptr;
        int32_t input_seq_length = 0; // padded input length
        int32_t max_past_kv_len = 0;
        // By default, max_attention_window == cyclic_attention_window_size
        // unless each layer has different cyclic kv cache length.
        // Max cache capacity (used to allocate KV cache)
        int32_t max_attention_window = 0;
        // Cyclic kv cache capacity (used to get the cyclic kv cache position for new tokens)
        int32_t cyclic_attention_window_size = 0;
        int32_t max_cyclic_attention_window_size = 0;
        bool can_use_one_more_block = false;
        int32_t sink_token_length = 0;
        int32_t const* q_seq_lengths = nullptr;
        int32_t const* kv_seq_lengths = nullptr;
        float const* kv_scale_orig_quant = nullptr;
        float const* kv_scale_quant_orig = nullptr;
        float const* attention_output_orig_quant = nullptr;
        float const* attention_output_sf_scale = nullptr;
        T const* alibi_slopes = nullptr;
        void* context_buf = nullptr;
        void* context_buf_sf = nullptr;
        void* key_value_cache = nullptr;
        kernels::KVBlockArray::DataType* block_offsets = nullptr;
        kernels::KVBlockArray::DataType* host_block_offsets = nullptr;
        void* host_primary_pool_pointer = nullptr;
        void* host_secondary_pool_pointer = nullptr;
        int32_t batch_size = 0;
        int32_t num_tokens = 0;
        int32_t max_blocks_per_sequence = 0;
        int32_t const* host_context_lengths = nullptr;
        void* workspace = nullptr;
        float2 const* mrope_rotary_cos_sin = nullptr;

        // optional when logn scaling
        float const* logn_scaling_ptr = nullptr;
        // optional when relative position
        T const* relative_attention_bias = nullptr;
        int relative_attention_bias_stride = 0;
        // optional when cross attention
        T const* cross_kv = nullptr;
        int32_t cross_kv_length = 0;
        int32_t const* encoder_input_lengths = nullptr;
        int32_t num_encoder_tokens = 0;
        int64_t const* runtime_perf_knobs = nullptr;
        kernels::mlaParams<T>* mla_param = nullptr;

        std::string enqueueContextParamsToString() const
        {
            // variables from the params coming from the runtime
            std::stringstream ss;
            ss << "EnqueueContextParams ====================" << std::endl;

            ss << "attention_input: " << attention_input << std::endl;
            ss << "qkv_bias: " << qkv_bias << std::endl;
            ss << "attention_mask: " << attention_mask << std::endl;
            ss << "attention_packed_mask: " << attention_packed_mask << std::endl;
            ss << "rotary_inv_freq: " << rotary_inv_freq << std::endl;
            ss << "rotary_cos_sin: " << rotary_cos_sin << std::endl;
            ss << "input_seq_length: " << input_seq_length << std::endl;
            ss << "max_past_kv_len: " << max_past_kv_len << std::endl;
            ss << "max_attention_window: " << max_attention_window << std::endl;
            ss << "cyclic_attention_window_size: " << cyclic_attention_window_size << std::endl;
            ss << "max_cyclic_attention_window_size: " << max_cyclic_attention_window_size << std::endl;
            ss << "can_use_one_more_block: " << (can_use_one_more_block ? "true" : "false") << std::endl;
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
            ss << "context_buf_sf: " << context_buf_sf << std::endl;
            ss << "key_value_cache: " << (half*) key_value_cache << std::endl;
            ss << "block_offsets: " << block_offsets << std::endl;
            ss << "host_block_offsets: " << host_block_offsets << std::endl;
            ss << "host_primary_pool_pointer: " << host_primary_pool_pointer << std::endl;
            ss << "host_secondary_pool_pointer: " << host_secondary_pool_pointer << std::endl;
            ss << "batch_size: " << batch_size << std::endl;
            ss << "num_tokens: " << num_tokens << std::endl;
            ss << "max_blocks_per_sequence: " << max_blocks_per_sequence << std::endl;
            ss << "workspace: " << workspace << std::endl;
            ss << "logn_scaling_ptr: " << logn_scaling_ptr << std::endl;
            ss << "relative_attention_bias: " << relative_attention_bias << std::endl;
            ss << "relative_attention_bias_stride: " << relative_attention_bias_stride << std::endl;
            ss << "cross_kv: " << cross_kv << std::endl;
            ss << "cross_kv_length: " << cross_kv_length << std::endl;
            ss << "encoder_input_lengths: " << encoder_input_lengths << std::endl;
            ss << "num_encoder_tokens: " << num_encoder_tokens << std::endl;
            return ss.str();
        }
    };

    template <typename T, typename KVCacheBuffer>
    int enqueueContext(EnqueueContextParams<T> const& params, cudaStream_t stream);

    template <typename T>
    struct EnqueueGenerationParams
    {
        T const* attention_input = nullptr;
        T const* qkv_bias = nullptr;
        // Attention mask input, which has shape of [batch_size, attention_mask_stride].
        bool const* attention_mask = nullptr;
        // Rotary inv_freq cache buffer to avoid re-computing.
        float const* rotary_inv_freq = nullptr;
        // Rotary cos sin cache buffer to avoid re-computing.
        float2 const* rotary_cos_sin = nullptr;
        // NOTE: input_seq_length might be larger than one in the medusa mode.
        int32_t input_seq_length = 0;
        int32_t const* sequence_lengths = nullptr;
        int32_t max_past_kv_length = 0;
        int32_t beam_width = 1;
        int32_t const* context_lengths = nullptr;
        float const* kv_scale_orig_quant = nullptr;
        float const* kv_scale_quant_orig = nullptr;
        float const* attention_output_orig_quant = nullptr;
        float const* attention_output_sf_scale = nullptr;
        T const* alibi_slopes = nullptr;
        void* context_buf = nullptr;
        void* context_buf_sf = nullptr;
        void* key_value_cache = nullptr;
        kernels::KVBlockArray::DataType* block_offsets = nullptr;
        void* host_primary_pool_pointer = nullptr;
        void* host_secondary_pool_pointer = nullptr;
        // Attention mask has shape of [batch_size, attention_mask_stride].
        int32_t attention_mask_stride = 0;
        // By default, max_attention_window == cyclic_attention_window_size
        // unless each layer has different cyclic kv cache length.
        // Max cache capacity (used to allocate KV cache)
        int32_t max_attention_window = 0;
        // Cyclic kv cache capacity (used to get the cyclic kv cache position for new tokens)
        int32_t cyclic_attention_window_size = 0;
        int32_t max_cyclic_attention_window_size = 0;
        bool can_use_one_more_block = false;
        int32_t sink_token_length = 0;
        int32_t num_requests = 0;
        int32_t max_blocks_per_sequence = 0;
        int32_t const* cache_indir = nullptr;
        int32_t* semaphores = nullptr;
        void* workspace = nullptr;
        int32_t const* host_past_key_value_lengths = nullptr;
        int32_t const* mrope_position_deltas = nullptr;

        // optional when logn scaling
        float const* logn_scaling_ptr = nullptr;
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
        bool spec_decoding_is_generation_length_variable = false;
        int32_t spec_decoding_max_generation_length = 1;
        int32_t total_num_input_tokens = 0;
        int64_t const* runtime_perf_knobs = nullptr;
        // optional when fuse_fp4_quant is enabled
        int32_t start_token_idx_sf = 0;
    };

    template <typename T, typename KVCacheBuffer>
    int enqueueGeneration(EnqueueGenerationParams<T> const& params, cudaStream_t stream);

    template <typename T>
    int mlaPreContext(kernels::mlaParams<T>& params, cudaStream_t stream);

    template <typename T>
    int mlaGeneration(
        kernels::mlaParams<T>& params, EnqueueGenerationParams<T> const& generation_params, cudaStream_t stream);

    // Called in configurePlugin().
    template <typename T, typename KVCacheBuffer>
    void prepareEnqueueGeneration(EnqueueGenerationParams<T> const& params);

    template <typename T, typename KVCacheBuffer>
    bool convertMMHAParamsToXQAParams(tensorrt_llm::kernels::XQAParams& xqaParams,
        EnqueueGenerationParams<T> const& generationsParams, bool forConfigurePlugin);

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
            || mPositionEmbeddingType == tensorrt_llm::kernels::PositionEmbeddingType::kLONG_ROPE
            || mPositionEmbeddingType == tensorrt_llm::kernels::PositionEmbeddingType::kROPE_M;
    }

    bool isLongRoPE() const
    {
        return mPositionEmbeddingType == tensorrt_llm::kernels::PositionEmbeddingType::kLONG_ROPE;
    }

    bool isUnfusedCrossAttention() const
    {
        return !mEnableContextFMHA && mCrossAttention;
    }

    bool isMRoPE() const
    {
        return mPositionEmbeddingType == tensorrt_llm::kernels::PositionEmbeddingType::kROPE_M;
    }

    bool isLognScaling() const
    {
        return mUseLognScaling;
    }

    bool isCrossAttention() const
    {
        return mCrossAttention;
    }

    bool useKVCache() const
    {
        return mUseKVCache;
    }

    bool useCustomMask() const
    {
        return mMaskType == tensorrt_llm::kernels::AttentionMaskType::CUSTOM_MASK;
    }

    bool useFullCustomMask() const
    {
        return useCustomMask() && mHasFullAttentionMask;
    }

    bool usePackedCustomMask() const
    {
        return useCustomMask() && mEnableContextFMHA;
    }

    void reserveSemaphoreArray(int32_t size);

    void debugCheckSemaphores(cudaStream_t stream);

    static constexpr int kReservedMaxSeqLenTilePerSeq = 64;

    int mLayerIdx = -1;
    int mNumHeads = -1;
    int mVisionStart = -1;
    int mVisionLength = -1;
    int mNumKVHeads = -1;
    int mLayerIdxInCachePool = -1;
    int mHeadSize = -1;
    int mUnidirectional = 1;
    float mQScaling = 1.0;
    float mAttnLogitSoftcappingScale = 0.0;
    int mRotaryEmbeddingDim = 0;
    float mRotaryEmbeddingBase = 10000.0;
    RotaryScalingType mRotaryEmbeddingScaleType = RotaryScalingType::kNONE;
    float mRotaryEmbeddingScale = 1.0;
    float mRotaryEmbeddingShortMscale = 1.0;
    float mRotaryEmbeddingLongMscale = 1.0;
    int mRotaryEmbeddingMaxPositions = 1024;
    int mRotaryEmbeddingOriginalMaxPositions = 1024;
    PositionEmbeddingType mPositionEmbeddingType = PositionEmbeddingType::kLEARNED_ABSOLUTE;
    bool mUseLognScaling = false;
    bool mRemovePadding = true;
    AttentionMaskType mMaskType = AttentionMaskType::CAUSAL;
    tensorrt_llm::kernels::BlockSparseParams mBlockSparseParams;

    // NOTE: default values for paged kv cache.
    bool mPagedKVCache = true;
    int mTokensPerBlock = 0;
    tensorrt_llm::common::QuantMode mKVCacheQuantMode;
    int mTpSize = 1;
    int mTpRank = 0;
    bool mUnfuseQkvGemm = false;
    nvinfer1::DataType mType;
    int32_t mMaxContextLength = 0;
    bool mQKVBiasEnabled = false;
    bool mCrossAttention = false;
    int mMaxDistance = 0;
    bool mPosShiftEnabled = false;
    bool mPagedContextFMHA = false;
    bool mFP8ContextFMHA = false;
    bool mDenseContextFMHA = false;
    bool mHasFullAttentionMask = false;
    bool mIsSpecDecodingEnabled = false;
    bool mUseSpecDecoding = false;
    bool mSpecDecodingIsGenerationLengthVariable = false;
    int32_t mSpecDecodingMaxGenerationLength = 1;
    bool mIsMLAEnabled = false;
    tensorrt_llm::kernels::mlaMetaParams mMLAParams;
    int mCpSize = 1;
    int mCpRank = 0;
    std::set<int32_t> mCpGroup = {};
#if ENABLE_MULTI_DEVICE
    std::shared_ptr<ncclComm_t> mCpNcclComm;
#endif // ENABLE_MULTI_DEVICE

    // Speculative decoding packed mask.
    uint4* mSpecDecodingPackedMask = nullptr;
    uint4* mSpecDecodingPackedHostMask = nullptr;

    // fmha runner (enabled by default)
    // flag: disabled = 0, enabled = 1, enabled with fp32 accumulation = 2
    bool mEnableContextFMHA = true;
    bool mFMHAForceFP32Acc = false;
    int mSM = tensorrt_llm::common::getSMVersion();
    bool mUseTllmGen = (mSM >= 100);
    int mMultiProcessorCount = tensorrt_llm::common::getMultiProcessorCount();
    int mMaxSharedMemoryPerBlockOptin = tensorrt_llm::common::getMaxSharedMemoryPerBlockOptin();
    // The default copy constructor will leave it as nullptr. clone() shall initialize it.
    std::shared_ptr<CUDADriverWrapper> mDriver;
    UniqPtrWNullCopy<tensorrt_llm::kernels::FusedMHARunnerV2> mDecoderFMHARunner;
    UniqPtrWNullCopy<tensorrt_llm::kernels::FmhaDispatcher> mFmhaDispatcher;
    UniqPtrWNullCopy<tensorrt_llm::kernels::XqaDispatcher> mXqaDispatcher;

    bool mMultiBlockMode = true;
    bool mEnableXQA = true;
    int mDeviceId = -1;
    static bool mForceMultiBlockWarned;
    // The default copy constructor will leave it as nullptr. clone() shall initialize it.
    UniqPtrWNullCopy<tensorrt_llm::common::CublasMMWrapper> mCublasWrapper;
    bool mUseKVCache = true;

    // This is implementation details which we want to save when serializing, but not expose as
    // a plugin field or a constructor parameter
    int32_t mNbMultiBlockSemaphores = 0;

    // Whether to fuse FP4 quant into attention kernel.
    bool mFuseFp4Quant = false;

    bool mSkipAttn = false;

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
        ss << "mLayerIdxInCachePool: " << mLayerIdxInCachePool << std::endl;
        ss << "mHeadSize: " << mHeadSize << std::endl;
        ss << "mUnidirectional: " << mUnidirectional << std::endl;
        ss << "mQScaling: " << mQScaling << std::endl;
        ss << "mRotaryEmbeddingDim: " << mRotaryEmbeddingDim << std::endl;
        ss << "mRotaryEmbeddingBase: " << mRotaryEmbeddingBase << std::endl;
        ss << "mRotaryEmbeddingScaleType: " << static_cast<int>(mRotaryEmbeddingScaleType) << std::endl;
        ss << "mRotaryEmbeddingScale: " << mRotaryEmbeddingScale << std::endl;
        ss << "mRotaryEmbeddingMaxPositions: " << mRotaryEmbeddingMaxPositions << std::endl;
        ss << "mPositionEmbeddingType: " << static_cast<int>(mPositionEmbeddingType) << std::endl;
        ss << "mUseLognScaling: " << std::boolalpha << mUseLognScaling << std::endl;
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
        ss << "mUseTllmGen: " << mUseTllmGen << std::endl;
        ss << "mMultiProcessorCount: " << mMultiProcessorCount << std::endl;
        ss << "mMaxSharedMemoryPerBlockOptin: " << mMaxSharedMemoryPerBlockOptin << std::endl;
        ss << "mMultiBlockMode: " << std::boolalpha << mMultiBlockMode << std::endl;
        ss << "mEnableXQA: " << std::boolalpha << mEnableXQA << std::endl;
        ss << "mDeviceId: " << mDeviceId << std::endl;
        ss << "mUseKVCache: " << std::boolalpha << mUseKVCache << std::endl;
        ss << "mForceMultiBlockWarned: " << mForceMultiBlockWarned << std::endl;
        ss << "mFuseFp4Quant: " << std::boolalpha << mFuseFp4Quant << std::endl;
        ss << "mSkipAttn: " << std::boolalpha << mSkipAttn << std::endl;
        ss << "mCpSize: " << mCpSize << std::endl;
        ss << "mCpRank: " << mCpRank << std::endl;
        ss << "mCpGroup: [";
        for (auto it = mCpGroup.begin(); it != mCpGroup.end(); it++)
        {
            if (it != mCpGroup.begin())
            {
                ss << ", ";
            }
            ss << *it;
        }
        ss << "]" << std::endl;

        return ss.str();
    }
};

} // namespace tensorrt_llm::common::op
