/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "tensorrt_llm/common/config.h"
#include "tensorrt_llm/common/cublasMMWrapper.h"
#include "tensorrt_llm/common/opUtils.h"
#include "tensorrt_llm/common/quantization.h"
#include "tensorrt_llm/common/tllmDataType.h"
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
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>
#if ENABLE_MULTI_DEVICE
#include <nccl.h>
#endif // ENABLE_MULTI_DEVICE

TRTLLM_NAMESPACE_BEGIN

namespace common::op
{

struct AttentionStaticConfig
{
#define TRTLLM_ATTENTION_STATIC_CONFIG_FIELD(cpp_type, name, cpp_default) cpp_type name cpp_default;
#include "tensorrt_llm/thop/fmha_params_fields.inc"
#undef TRTLLM_ATTENTION_STATIC_CONFIG_FIELD
};

class AttentionOp
{
public:
    using RotaryScalingType = tensorrt_llm::kernels::RotaryScalingType;
    using PositionEmbeddingType = tensorrt_llm::kernels::PositionEmbeddingType;
    using AttentionMaskType = tensorrt_llm::kernels::AttentionMaskType;

    AttentionOp(){};
    ~AttentionOp() = default;

    int initialize() noexcept;
    [[nodiscard]] size_t getFmhaMultiCtasKvScratchSize() const noexcept;
    [[nodiscard]] int getHeadSize(bool checkInit = true) const;
    [[nodiscard]] int getMaxNumSeqLenTile(int batch_beam_size = 1) const;
    [[nodiscard]] size_t getWorkspaceSizeForContext(tensorrt_llm::DataType type, int32_t nbReq,
        int32_t max_input_length, int32_t cross_kv_length = 0, int32_t max_num_tokens = 0,
        int32_t total_kv_len = 0) const noexcept;
    // Per-token byte cost of the context-MLA K/V dequant staging buffers, whose size scales with the summed
    // attended KV length (`total_kv_len`). Only the fp8 context-MLA separate-Q/KV path stages these buffers;
    // every other path (incl. sparse MLA, which reads K/V straight from the paged cache) returns 0. Single
    // source of truth shared by getWorkspaceSizeForContext (runtime sizing) and the KV-cache estimator, so
    // the two cannot drift.
    [[nodiscard]] static size_t contextMlaWorkspaceBytesPerToken(int32_t numAttnHeads, int32_t qkRopeHeadDim,
        int32_t qkNopeHeadDim, int32_t vHeadDim, bool fp8ContextMla, bool separateQAndKvInput, bool sparseMla) noexcept;
    // total_num_seq is the sum of beam_width for multiple requests
    [[nodiscard]] size_t getWorkspaceSizeForGeneration(tensorrt_llm::DataType type, int32_t total_num_seq,
        int32_t max_attention_window_size, int32_t max_num_tokens, int32_t max_blocks_per_sequence) const noexcept;

    template <typename T>
    class EnqueueParams
    {
    public:
#define TRTLLM_FMHA_ENQUEUE_PARAM_FIELD(name, cpp_type, cpp_default) cpp_type name cpp_default;
#include "tensorrt_llm/thop/fmha_params_fields.inc"
#undef TRTLLM_FMHA_ENQUEUE_PARAM_FIELD
    };

    template <typename T>
    class EnqueueContextParams : public EnqueueParams<T>
    {
    public:
#define TRTLLM_FMHA_ENQUEUE_CONTEXT_PARAM_FIELD(name, cpp_type, cpp_default) cpp_type name cpp_default;
#include "tensorrt_llm/thop/fmha_params_fields.inc"
#undef TRTLLM_FMHA_ENQUEUE_CONTEXT_PARAM_FIELD

        std::string enqueueContextParamsToString() const
        {
            std::stringstream ss;
            ss << "EnqueueContextParams ====================" << std::endl;
            auto appendParam = [this, &ss](char const* name, auto const& value)
            {
                if constexpr (std::is_same_v<std::decay_t<decltype(value)>, int32_t const*>)
                {
                    if (std::string_view{name} == "context_lengths" || std::string_view{name} == "sequence_length")
                    {
                        if (value && this->num_seqs > 0)
                        {
                            ss << name << ": "
                               << *(runtime::ITensor::wrap((void*) value, tensorrt_llm::DataType::kINT32,
                                      runtime::ITensor::makeShape({this->num_seqs})))
                               << std::endl;
                        }
                        return;
                    }
                }
                ss << name << ": " << value << std::endl;
            };

#define TRTLLM_FMHA_ENQUEUE_PARAM_FIELD(name, cpp_type, cpp_default) appendParam(#name, this->name);
#include "tensorrt_llm/thop/fmha_params_fields.inc"
#undef TRTLLM_FMHA_ENQUEUE_PARAM_FIELD
#define TRTLLM_FMHA_ENQUEUE_CONTEXT_PARAM_FIELD(name, cpp_type, cpp_default) appendParam(#name, this->name);
#include "tensorrt_llm/thop/fmha_params_fields.inc"
#undef TRTLLM_FMHA_ENQUEUE_CONTEXT_PARAM_FIELD
            return ss.str();
        }
    };

    template <typename T, typename KVCacheBuffer>
    int enqueueContext(EnqueueContextParams<T> const& params, cudaStream_t stream);

    template <typename T>
    class EnqueueGenerationParams : public EnqueueParams<T>
    {
    public:
#define TRTLLM_FMHA_ENQUEUE_GENERATION_PARAM_FIELD(name, cpp_type, cpp_default) cpp_type name cpp_default;
#include "tensorrt_llm/thop/fmha_params_fields.inc"
#undef TRTLLM_FMHA_ENQUEUE_GENERATION_PARAM_FIELD
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
    int getKvCacheElemSizeInBits() const
    {
        return getKvCacheElemSizeInBits(mConfig.quant_mode, sizeof(T));
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
        return mConfig.position_embedding_type == tensorrt_llm::kernels::PositionEmbeddingType::kRELATIVE;
    }

    [[nodiscard]] bool isALiBi() const
    {
        return mConfig.position_embedding_type == tensorrt_llm::kernels::PositionEmbeddingType::kALIBI
            || mConfig.position_embedding_type == tensorrt_llm::kernels::PositionEmbeddingType::kALIBI_WITH_SCALE;
    }

    [[nodiscard]] bool isAliBiWithScale() const
    {
        return mConfig.position_embedding_type == tensorrt_llm::kernels::PositionEmbeddingType::kALIBI_WITH_SCALE;
    }

    [[nodiscard]] bool isRoPE() const
    {
        return mConfig.position_embedding_type == tensorrt_llm::kernels::PositionEmbeddingType::kROPE_GPTJ
            || mConfig.position_embedding_type == tensorrt_llm::kernels::PositionEmbeddingType::kROPE_GPT_NEOX
            || mConfig.position_embedding_type == tensorrt_llm::kernels::PositionEmbeddingType::kLONG_ROPE
            || mConfig.position_embedding_type == tensorrt_llm::kernels::PositionEmbeddingType::kYARN
            || mConfig.position_embedding_type == tensorrt_llm::kernels::PositionEmbeddingType::kROPE_M;
    }

    [[nodiscard]] bool isLongRoPE() const
    {
        return mConfig.position_embedding_type == tensorrt_llm::kernels::PositionEmbeddingType::kLONG_ROPE;
    }

    [[nodiscard]] bool isUnfusedCrossAttention() const
    {
        return !mEnableContextFMHA && mConfig.cross_attention;
    }

    [[nodiscard]] bool isMRoPE() const
    {
        return mConfig.position_embedding_type == tensorrt_llm::kernels::PositionEmbeddingType::kROPE_M;
    }

    [[nodiscard]] bool isLognScaling() const
    {
        return mConfig.use_logn_scaling;
    }

    [[nodiscard]] bool isCrossAttention() const
    {
        return mConfig.cross_attention;
    }

    [[nodiscard]] bool useKVCache() const
    {
        return mConfig.use_kv_cache;
    }

    [[nodiscard]] bool useCustomMask() const
    {
        return mConfig.mask_type == AttentionMaskType::CUSTOM_MASK;
    }

    [[nodiscard]] bool useFullCustomMask() const
    {
        return useCustomMask() && mConfig.has_full_attention_mask;
    }

    [[nodiscard]] bool usePackedCustomMask() const
    {
        return useCustomMask() && mEnableContextFMHA;
    }

    [[nodiscard]] bool isMLAEnabled() const
    {
        return mConfig.is_mla_enable;
    }

    [[nodiscard]] bool useSparseAttention() const
    {
        return mConfig.use_sparse_attention && mPagedKVCache && mEnableXQA;
    }

    [[nodiscard]] bool useTllmGenSparseAttentionPaged() const
    {
        return mConfig.use_tllm_gen_sparse_attention_paged && useSparseAttention();
    }

    [[nodiscard]] bool useSparseMLA() const
    {
        return mConfig.use_sparse_attention && mUseTllmGen && mConfig.is_mla_enable;
    }

    [[nodiscard]] bool useTllmGenSparseAttention() const
    {
        return useSparseMLA() || (mConfig.use_sparse_attention && mUseTllmGen && mConfig.use_tllm_gen_sparse_attention);
    }

    [[nodiscard]] int smVersion() const
    {
        return mSM;
    }

    [[nodiscard]] bool supportsNvFp4Output() const
    {
        bool needsUlyssesPostprocess = mConfig.cp_size > 1 && mAttnTpSize > 1 && mAttnCpSize == 1;
        return mEnableContextFMHA && mEnableXQA && !needsUlyssesPostprocess;
    }

    [[nodiscard]] int getMultiProcessorCount() const
    {
        return mMultiProcessorCount;
    }

    [[nodiscard]] std::string toString() const;

    AttentionStaticConfig mConfig{};

    int mNumKVHeads = -1;
    int mHeadSize = -1;

    bool mPagedKVCache = true;
    bool mFP8ContextFMHA = false;
    bool mFP8AttenOutput = false;
    bool mFP8ContextMLA = false;
    bool mFP8GenerationMLA = false;
    bool mIsGenerationMLA = false;
    bool mUseGenFlashMLA = false;

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

    bool mFuseFp4Quant = false;

    kernels::SparseAttentionParams mRuntimeSparseAttentionParams;

#ifdef SKIP_SOFTMAX_STAT
    uint32_t* mSkipSoftmaxTotalBlocks;
    uint32_t* mSkipSoftmaxSkippedBlocks;
#endif

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
};

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

} // namespace common::op

TRTLLM_NAMESPACE_END
