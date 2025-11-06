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
#include "tensorrt_llm/kernels/cutlass_kernels/fp8_blockscale_gemm/fp8_blockscale_gemm.h"
#include "tensorrt_llm/kernels/decoderMaskedMultiheadAttention/decoderXQARunner.h"
#include "tensorrt_llm/kernels/fmhaDispatcher.h"
#include "tensorrt_llm/kernels/gptKernels.h"
#include "tensorrt_llm/kernels/kvCacheUtils.h"
#include "tensorrt_llm/kernels/mlaKernels.h"
#include "tensorrt_llm/kernels/sparseAttentionKernels.h"
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

class AttentionOp
{
public:
    using RotaryScalingType = tensorrt_llm::kernels::RotaryScalingType;
    using PositionEmbeddingType = tensorrt_llm::kernels::PositionEmbeddingType;
    using AttentionMaskType = tensorrt_llm::kernels::AttentionMaskType;

    AttentionOp(){};
    ~AttentionOp() = default;

    int initialize() noexcept;
    [[nodiscard]] int getHeadSize(bool checkInit = true) const;
    [[nodiscard]] int getMaxNumSeqLenTile(int batch_beam_size = 1) const;
    [[nodiscard]] size_t getWorkspaceSizeForContext(nvinfer1::DataType type, int32_t nbReq, int32_t max_input_length,
        int32_t cross_kv_length = 0, int32_t max_num_tokens = 0) const noexcept;
    // total_num_seq is the sum of beam_width for multiple requests
    [[nodiscard]] size_t getWorkspaceSizeForGeneration(nvinfer1::DataType type, int32_t total_num_seq,
        int32_t max_attention_window_size, int32_t max_num_tokens, int32_t max_blocks_per_sequence) const noexcept;

    template <typename T>
    class EnqueueParams
    {
    public:
        T const* attention_input = nullptr;
        T const* qkv_bias = nullptr;
        // Attention mask input, which has shape of [batch_size, attention_mask_stride].
        bool const* attention_mask = nullptr;
        // Attention sinks with shape of [num_heads_q] float.
        float const* attention_sinks = nullptr;
        // Rotary inv_freq cache buffer to avoid re-computing.
        float const* rotary_inv_freq = nullptr;
        // Rotary cos sin cache buffer to avoid re-computing.
        float2 const* rotary_cos_sin = nullptr;
        // NOTE: input_seq_length might be larger than one in the medusa mode.
        int32_t input_seq_length = 0;
        int32_t max_past_kv_length = 0;
        // By default, max_attention_window_size == cyclic_attention_window_size
        // unless each layer has different cyclic kv cache length.
        // Max cache capacity (used to allocate KV cache)
        int32_t max_attention_window_size = 0;
        // Cyclic kv cache capacity (used to get the cyclic kv cache position for new tokens)
        int32_t cyclic_attention_window_size = 0;
        int32_t max_cyclic_attention_window_size = 0;
        bool can_use_one_more_block = false;
        int32_t sink_token_length = 0;
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
        void* host_primary_block_scale_pool_pointer = nullptr;
        void* host_secondary_block_scale_pool_pointer = nullptr;
        int32_t num_tokens = 0;
        int32_t total_kv_len = 0;
        int32_t max_blocks_per_sequence = 0;
        int32_t const* sequence_lengths = nullptr;
        int32_t const* context_lengths = nullptr;
        int32_t const* host_context_lengths = nullptr;
        void* workspace = nullptr;
        // optional when logn scaling
        float const* logn_scaling_ptr = nullptr;
        // optional when relative position
        T const* relative_attention_bias = nullptr;
        int relative_attention_bias_stride = 0;
        // optional when cross attention
        int32_t const* encoder_input_lengths = nullptr;
        int64_t const* runtime_perf_knobs = nullptr;
        // optional when compute attention stats (MLA chunked prefill or Helix parallelism)
        // this is a buffer of size [num_tokens, num_heads_q] with each element
        // representing the max and LSE/denominator of the softmax values
        float2* softmax_stats = nullptr;
    };

    template <typename T>
    class EnqueueContextParams : public EnqueueParams<T>
    {
    public:
        // Attention packed mask input (used by context FMHA).
        uint32_t const* attention_packed_mask = nullptr;
        kernels::KVBlockArray::DataType* host_block_offsets = nullptr;
        int32_t batch_size = 0;
        float2 const* mrope_rotary_cos_sin = nullptr;

        // optional when cross attention
        T const* cross_kv = nullptr;
        int32_t cross_kv_length = 0;
        int32_t num_encoder_tokens = 0;
        kernels::MlaParams<T>* mla_param = nullptr;

        // optional for separate QKV input, currently only used for context MLA
        T const* k_ptr = nullptr;
        T const* v_ptr = nullptr;

        std::string enqueueContextParamsToString() const
        {
            // variables from the params coming from the runtime
            std::stringstream ss;
            ss << "EnqueueContextParams ====================" << std::endl;

            ss << "attention_input: " << this->attention_input << std::endl;
            ss << "qkv_bias: " << this->qkv_bias << std::endl;
            ss << "attention_mask: " << this->attention_mask << std::endl;
            ss << "attention_packed_mask: " << this->attention_packed_mask << std::endl;
            ss << "rotary_inv_freq: " << this->rotary_inv_freq << std::endl;
            ss << "rotary_cos_sin: " << this->rotary_cos_sin << std::endl;
            ss << "input_seq_length: " << this->input_seq_length << std::endl;
            ss << "max_past_kv_length: " << this->max_past_kv_length << std::endl;
            ss << "max_attention_window_size: " << this->max_attention_window_size << std::endl;
            ss << "cyclic_attention_window_size: " << this->cyclic_attention_window_size << std::endl;
            ss << "max_cyclic_attention_window_size: " << this->max_cyclic_attention_window_size << std::endl;
            ss << "can_use_one_more_block: " << (this->can_use_one_more_block ? "true" : "false") << std::endl;
            ss << "sink_token_length: " << this->sink_token_length << std::endl;
            if (this->context_lengths && batch_size > 0)
            {
                ss << "context_lengths: "
                   << *(runtime::ITensor::wrap((void*) this->context_lengths, nvinfer1::DataType::kINT32,
                          runtime::ITensor::makeShape({batch_size})))
                   << std::endl;
            }
            if (this->sequence_lengths && batch_size > 0)
            {
                ss << "sequence_lengths: "
                   << *(runtime::ITensor::wrap((void*) this->sequence_lengths, nvinfer1::DataType::kINT32,
                          runtime::ITensor::makeShape({batch_size})))
                   << std::endl;
            }
            ss << "kv_scale_orig_quant: " << this->kv_scale_orig_quant << std::endl;
            ss << "kv_scale_quant_orig: " << this->kv_scale_quant_orig << std::endl;
            ss << "attention_output_orig_quant: " << this->attention_output_orig_quant << std::endl;
            ss << "alibi_slopes: " << this->alibi_slopes << std::endl;
            ss << "context_buf: " << this->context_buf << std::endl;
            ss << "context_buf_sf: " << this->context_buf_sf << std::endl;
            ss << "key_value_cache: " << (half*) this->key_value_cache << std::endl;
            ss << "block_offsets: " << this->block_offsets << std::endl;
            ss << "host_block_offsets: " << this->host_block_offsets << std::endl;
            ss << "host_primary_pool_pointer: " << this->host_primary_pool_pointer << std::endl;
            ss << "host_secondary_pool_pointer: " << this->host_secondary_pool_pointer << std::endl;
            ss << "batch_size: " << this->batch_size << std::endl;
            ss << "num_tokens: " << this->num_tokens << std::endl;
            ss << "total_kv_len: " << this->total_kv_len << std::endl;
            ss << "max_blocks_per_sequence: " << this->max_blocks_per_sequence << std::endl;
            ss << "workspace: " << this->workspace << std::endl;
            ss << "logn_scaling_ptr: " << this->logn_scaling_ptr << std::endl;
            ss << "relative_attention_bias: " << this->relative_attention_bias << std::endl;
            ss << "relative_attention_bias_stride: " << this->relative_attention_bias_stride << std::endl;
            ss << "cross_kv: " << this->cross_kv << std::endl;
            ss << "cross_kv_length: " << this->cross_kv_length << std::endl;
            ss << "encoder_input_lengths: " << this->encoder_input_lengths << std::endl;
            ss << "num_encoder_tokens: " << this->num_encoder_tokens << std::endl;
            ss << "softmaxStatsPtr: " << this->softmax_stats << std::endl;
            ss << "k_ptr: " << this->k_ptr << std::endl;
            ss << "v_ptr: " << this->v_ptr << std::endl;
            return ss.str();
        }
    };

    template <typename T, typename KVCacheBuffer>
    int enqueueContext(EnqueueContextParams<T> const& params, cudaStream_t stream);

    template <typename T>
    class EnqueueGenerationParams : public EnqueueParams<T>
    {
    public:
        int32_t beam_width = 1;
        // Attention mask has shape of [batch_size, attention_mask_stride].
        int32_t attention_mask_stride = 0;
        int32_t num_requests = 0;
        int32_t const* cache_indir = nullptr;
        int32_t* semaphores = nullptr;
        int32_t const* host_past_key_value_lengths = nullptr;
        int32_t const* mrope_position_deltas = nullptr;

        // optional when speculative decoding is used.
        bool const* spec_decoding_mask = nullptr;
        int32_t const* spec_decoding_packed_mask = nullptr;
        int32_t const* spec_decoding_position_offsets = nullptr;
        int32_t const* spec_decoding_generation_lengths = nullptr;
        bool spec_decoding_is_generation_length_variable = false;
        int32_t spec_decoding_max_generation_length = 1;
        // optional when fuse_fp4_quant is enabled
        int32_t start_token_idx_sf = 0;
    };

    template <typename T, typename KVCacheBuffer>
    int enqueueGeneration(EnqueueGenerationParams<T> const& params, cudaStream_t stream);

    template <typename T>
    int mlaGeneration(
        kernels::MlaParams<T>& params, EnqueueGenerationParams<T> const& generation_params, cudaStream_t stream);

    int getFlashMlaNumSmParts(int s_q, int num_heads, int num_kv_heads, int head_size_v) const
    {
        static constexpr int block_size_m = 64;
        int num_heads_per_head_k = s_q * num_heads / num_kv_heads;
        int sm_cnt = mMultiProcessorCount;
        int num_sm_parts = sm_cnt / num_kv_heads / cutlass::ceil_div(num_heads_per_head_k, block_size_m);
        return num_sm_parts;
    }

    template <typename T>
    int getKvCacheElemSizeInBits() const
    {
        if (mKVCacheQuantMode.hasInt8KvCache() || mKVCacheQuantMode.hasFp8KvCache())
        {
            return 8;
        }
        else if (mKVCacheQuantMode.hasFp4KvCache())
        {
            return 4;
        }
        return sizeof(T) * 8;
    }

    // Called in configurePlugin().
    template <typename T, typename KVCacheBuffer>
    void prepareEnqueueGeneration(EnqueueGenerationParams<T> const& params);

    template <typename T, typename KVCacheBuffer>
    bool convertMMHAParamsToXQAParams(tensorrt_llm::kernels::XQAParams& xqaParams,
        EnqueueGenerationParams<T> const& generationsParams, bool forConfigurePlugin);

    template <typename T>
    int ulyssesContextPreprocess(T const* input, T* output, T* buffer, EnqueueContextParams<T> const& params,
        int const* cu_q_seqlens, int const* cu_cp_partial_seqlens, cudaStream_t stream);

    template <typename T>
    int ulyssesContextPostprocess(T* input, T* output, T* buffer, EnqueueContextParams<T> const& params,
        int const* cu_q_seqlens, int const* cu_cp_partial_seqlens, cudaStream_t stream);

    template <typename T>
    int ulyssesGenerationPreprocess(T const* input, T* output, T* buffer, int32_t batch_beam, cudaStream_t stream);

    template <typename T>
    int ulyssesGenerationPostprocess(T* input, T* output, T* buffer, int32_t batch_beam, cudaStream_t stream);

    [[nodiscard]] bool isRelativePosition() const
    {
        return mPositionEmbeddingType == tensorrt_llm::kernels::PositionEmbeddingType::kRELATIVE;
    }

    [[nodiscard]] bool isALiBi() const
    {
        return mPositionEmbeddingType == tensorrt_llm::kernels::PositionEmbeddingType::kALIBI
            || mPositionEmbeddingType == tensorrt_llm::kernels::PositionEmbeddingType::kALIBI_WITH_SCALE;
    }

    [[nodiscard]] bool isAliBiWithScale() const
    {
        return mPositionEmbeddingType == tensorrt_llm::kernels::PositionEmbeddingType::kALIBI_WITH_SCALE;
    }

    [[nodiscard]] bool isRoPE() const
    {
        return mPositionEmbeddingType == tensorrt_llm::kernels::PositionEmbeddingType::kROPE_GPTJ
            || mPositionEmbeddingType == tensorrt_llm::kernels::PositionEmbeddingType::kROPE_GPT_NEOX
            || mPositionEmbeddingType == tensorrt_llm::kernels::PositionEmbeddingType::kLONG_ROPE
            || mPositionEmbeddingType == tensorrt_llm::kernels::PositionEmbeddingType::kYARN
            || mPositionEmbeddingType == tensorrt_llm::kernels::PositionEmbeddingType::kROPE_M;
    }

    [[nodiscard]] bool isLongRoPE() const
    {
        return mPositionEmbeddingType == tensorrt_llm::kernels::PositionEmbeddingType::kLONG_ROPE;
    }

    [[nodiscard]] bool isUnfusedCrossAttention() const
    {
        return !mEnableContextFMHA && mCrossAttention;
    }

    [[nodiscard]] bool isMRoPE() const
    {
        return mPositionEmbeddingType == tensorrt_llm::kernels::PositionEmbeddingType::kROPE_M;
    }

    [[nodiscard]] bool isLognScaling() const
    {
        return mUseLognScaling;
    }

    [[nodiscard]] bool isCrossAttention() const
    {
        return mCrossAttention;
    }

    [[nodiscard]] bool useKVCache() const
    {
        return mUseKVCache;
    }

    [[nodiscard]] bool useCustomMask() const
    {
        return mMaskType == AttentionMaskType::CUSTOM_MASK;
    }

    [[nodiscard]] bool useFullCustomMask() const
    {
        return useCustomMask() && mHasFullAttentionMask;
    }

    [[nodiscard]] bool usePackedCustomMask() const
    {
        return useCustomMask() && mEnableContextFMHA;
    }

    [[nodiscard]] bool isMLAEnabled() const
    {
        return mIsMLAEnabled;
    }

    [[nodiscard]] bool useSparseAttention() const
    {
        return mUseSparseAttention && mPagedKVCache && mEnableXQA;
    }

    [[nodiscard]] bool useTllmGenSparseAttention() const
    {
        return mUseTllmGenSparseAttention && useSparseAttention();
    }

    [[nodiscard]] bool useSparseMLA() const
    {
        return mUseSparseAttention && mUseTllmGen && mIsMLAEnabled;
    }

    [[nodiscard]] int smVersion() const
    {
        return mSM;
    }

    [[nodiscard]] bool supportsNvFp4Output() const
    {
        bool needsUlyssesPostprocess = mCpSize > 1 && mAttnTpSize > 1 && mAttnCpSize == 1;
        return mEnableContextFMHA && mEnableXQA && !needsUlyssesPostprocess;
    }

    [[nodiscard]] int32_t* multiBlockSemaphores() const
    {
        return mMultiBlockSemaphores.get();
    }

    void reserveSemaphoreArray(int32_t size);

    void debugCheckSemaphores(cudaStream_t stream);

    [[nodiscard]] int getMultiProcessorCount() const
    {
        return mMultiProcessorCount;
    }

    [[nodiscard]] std::string toString() const;

    int mLayerIdx = -1;
    int mNumHeads = -1;
    int mVisionStart = -1;
    int mVisionLength = -1;
    int mNumKVHeads = -1;
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
    bool mFP8AttenOutput = false;
    bool mFP8ContextMLA = false;
    bool mFP8GenerationMLA = false;
    size_t mChunkPrefillBufferBatchSize = 1;
    bool mDenseContextFMHA = false;
    bool mHasFullAttentionMask = false;
    bool mIsSpecDecodingEnabled = false;
    bool mUseSpecDecoding = false;
    bool mIsSpecDecTree = true;
    bool mSpecDecodingIsGenerationLengthVariable = false;
    int32_t mSpecDecodingMaxGenerationLength = 1;
    bool mIsMLAEnabled = false;
    bool mIsGenerationMLA = false;
    bool mUseGenFlashMLA = false;
    bool mUseSparseAttention = false;
    bool mUseTllmGenSparseAttention = false;
    tensorrt_llm::kernels::MlaMetaParams mMLAParams;
    int mCpSize = 1;
    int mCpRank = 0;
    std::set<int32_t> mCpGroup = {};
    // These parameters are used to specifically configure the attention attributes when cp/tp_size are different
    // between Attention and FFN(such as Ulysses)
    int mNumAttnHeads = -1;
    int mNumAttnKVHeads = -1;
    int mNumKVHeadsOrigin = -1;
    int mAttnTpSize = -1;
    int mAttnTpRank = 0;
    int mAttnCpSize = -1;
    int mAttnCpRank = 0;
    int mUlyssesMQABroadcast = 1;

    // fmha runner (enabled by default)
    // flag: disabled = 0, enabled = 1, enabled with fp32 accumulation = 2
    bool mEnableContextFMHA = true;
    bool mFMHAForceFP32Acc = false;
    bool mMultiBlockMode = true;
    bool mEnableXQA = true;
    bool mUseKVCache = true;
    bool mSkipAttn = false;

    // Whether to fuse FP4 quant into attention kernel.
    bool mFuseFp4Quant = false;

    kernels::SparseAttentionParams mRuntimeSparseAttentionParams;

    // This is implementation details which we want to save when serializing, but not expose as
    // a plugin field or a constructor parameter
    int32_t mNbMultiBlockSemaphores = 0;

    // See [Chunked Attention] in _torch/modules/attention.py
    std::optional<int64_t> mAttentionChunkSize = std::nullopt;

    [[nodiscard]] auto data() const
    {
        return std::make_tuple(mLayerIdx, mNumHeads, mVisionStart, mVisionLength, mNumKVHeads, mHeadSize,
            mUnidirectional, mQScaling, mAttnLogitSoftcappingScale, mRotaryEmbeddingDim, mRotaryEmbeddingBase,
            (int8_t) mRotaryEmbeddingScaleType, mRotaryEmbeddingScale, mRotaryEmbeddingShortMscale,
            mRotaryEmbeddingLongMscale, mRotaryEmbeddingMaxPositions, mRotaryEmbeddingOriginalMaxPositions,
            (int8_t) mPositionEmbeddingType, mUseLognScaling, mRemovePadding, (int32_t) mMaskType,
            mBlockSparseParams.data(), mPagedKVCache, mTokensPerBlock, mKVCacheQuantMode.value(), mTpSize, mTpRank,
            mUnfuseQkvGemm, (int32_t) mType, mMaxContextLength, mQKVBiasEnabled, mCrossAttention, mMaxDistance,
            mPosShiftEnabled, mPagedContextFMHA, mFP8ContextFMHA, mFP8AttenOutput, mFP8ContextMLA, mFP8GenerationMLA,
            mChunkPrefillBufferBatchSize, mDenseContextFMHA, mHasFullAttentionMask, mIsSpecDecodingEnabled,
            mUseSpecDecoding, mIsSpecDecTree, mSpecDecodingIsGenerationLengthVariable, mSpecDecodingMaxGenerationLength,
            mIsMLAEnabled, mIsGenerationMLA, mUseGenFlashMLA, mUseSparseAttention, mUseTllmGenSparseAttention,
            mMLAParams.data(), mCpSize, mCpRank, mCpGroup, mNumAttnHeads, mNumAttnKVHeads, mNumKVHeadsOrigin,
            mAttnTpSize, mAttnTpRank, mAttnCpSize, mAttnCpRank, mUlyssesMQABroadcast, mEnableContextFMHA,
            mFMHAForceFP32Acc, mMultiBlockMode, mEnableXQA, mUseKVCache, mSkipAttn, mFuseFp4Quant,
            mRuntimeSparseAttentionParams.data(), mNbMultiBlockSemaphores, mAttentionChunkSize.value_or(-1));
    };

private:
    static constexpr int kReservedMaxSeqLenTilePerSeq = 64;

    int mSM = tensorrt_llm::common::getSMVersion();
    bool mUseTllmGen = (mSM >= 100) && (mSM != 120);
    bool mForceMultiBlockWarned = false;
    int mMultiProcessorCount = tensorrt_llm::common::getMultiProcessorCount();
    int mMaxSharedMemoryPerBlockOptin = tensorrt_llm::common::getMaxSharedMemoryPerBlockOptin();
    // The default copy constructor will leave it as nullptr. clone() shall initialize it.
    std::shared_ptr<CUDADriverWrapper> mDriver;
    UniqPtrWNullCopy<tensorrt_llm::kernels::FusedMHARunnerV2> mDecoderFMHARunner;
    UniqPtrWNullCopy<tensorrt_llm::kernels::FmhaDispatcher> mFmhaDispatcher;
    UniqPtrWNullCopy<tensorrt_llm::kernels::XqaDispatcher> mXqaDispatcher;
    UniqPtrWNullCopy<tensorrt_llm::kernels::TllmGenFmhaRunner> mTllmGenFMHARunner;

    // The default copy constructor will leave it as nullptr. clone() shall initialize it.
    UniqPtrWNullCopy<tensorrt_llm::common::CublasMMWrapper> mCublasWrapper;

#if ENABLE_MULTI_DEVICE
    std::shared_ptr<ncclComm_t> mCpNcclComm;
#endif // ENABLE_MULTI_DEVICE

    struct Deleter
    {
        void operator()(void* ptr)
        {
            cudaFree(ptr);
        }
    };

    UniqPtrWNullCopy<int32_t[], Deleter> mMultiBlockSemaphores = {};
};

} // namespace tensorrt_llm::common::op
