/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
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
#include "tensorrt_llm/kernels/decoderMaskedMultiheadAttention/decoderXQAImplPrecompiled.h"

#include "tensorrt_llm/common/cudaDriverWrapper.h"
#include "tensorrt_llm/common/envUtils.h"
#include "tensorrt_llm/common/workspace.h"
#include "tensorrt_llm/kernels/decoderMaskedMultiheadAttention/cubin/xqa_kernel_cubin.h"
#include "tensorrt_llm/kernels/decoderMaskedMultiheadAttention/decoderXQAConstants.h"
#include "tensorrt_llm/kernels/decoderMaskedMultiheadAttention/decoderXQAImplCommon.h"
#include "tensorrt_llm/kernels/decoderMaskedMultiheadAttention/decoderXQARunner.h"
#include "tensorrt_llm/kernels/decoderMaskedMultiheadAttention/tensorMapUtils.h"
#include "tensorrt_llm/kernels/kvCacheUtils.h"
#include "tensorrt_llm/kernels/unfusedAttentionKernels.h"
#include <cassert>
#include <cuda.h>
#include <functional>
#include <memory>
#include <mutex>

using namespace tensorrt_llm::common;

namespace tensorrt_llm::kernels
{

class XQAKernelList
{
public:
    using TKernelMeta = XQAKernelMetaInfo;

    XQAKernelList(Data_type type, unsigned int sm)
        : mDriver(tensorrt_llm::common::CUDADriverWrapper::getInstance())
        , mDataType(type)
        , mKernelMetaCount(sizeof(sXqaKernelMetaInfo) / sizeof(sXqaKernelMetaInfo[0]))
        , mKernelMeta(&sXqaKernelMetaInfo[0])
        , mSM(sm)
    {
        mForceXQA = forceXQAKernels();
    }

    void loadXQAKernels()
    {
        if (!mFunctions.empty())
        {
            return;
        }
        for (unsigned int i = 0; i < mKernelMetaCount; ++i)
        {
            auto const& kernelMeta = mKernelMeta[i];
            if (kernelMeta.mSM != mSM || kernelMeta.mDataType != mDataType)
                continue;

            // Cubins for kernels that would take the JIT path are removed from kernelMeta.
            if (kernelMeta.mCubin == nullptr)
                continue;

            CUlibrary hlib{0};
            auto findLibIter = mCuLibs.find(kernelMeta.mCubin);
            if (findLibIter != mCuLibs.end())
            {
                hlib = findLibIter->second;
            }
            else
            {
                TLLM_CU_CHECK(
                    mDriver->cuLibraryLoadData(&hlib, kernelMeta.mCubin, nullptr, nullptr, 0, nullptr, nullptr, 0));
                mCuLibs.insert(std::make_pair(kernelMeta.mCubin, hlib));
            }

            XQAKernelFuncInfo funcInfo{};
            funcInfo.mMetaInfoIndex = i;
            TLLM_CU_CHECK(mDriver->cuLibraryGetKernel(&funcInfo.mDeviceFunction, hlib, kernelMeta.mFuncName));
            funcInfo.mSharedMemBytes = getGlobalVar<uint32_t>(mDriver, hlib, "smemSize", true).value();
            funcInfo.mKernelType = getGlobalVar<XQAKernelType>(mDriver, hlib, "kernelType", false)
                                       .value_or(XQAKernelType::kAMPERE_WARP_SPECIALIZED);

            /* Set 46KB threshold here because we have to take static/driver shared memory into consideration. */
            if (funcInfo.mSharedMemBytes >= 46 * 1024)
            {
                CUdevice dev;
                TLLM_CU_CHECK(mDriver->cuCtxGetDevice(&dev));
                TLLM_CU_CHECK(mDriver->cuKernelSetAttribute(CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                    funcInfo.mSharedMemBytes, funcInfo.mDeviceFunction, dev));
            }
            XQAKernelRuntimeHashKey hash_key{kernelMeta.mKVDataType, kernelMeta.mHeadDim, kernelMeta.mBeamWidth,
                kernelMeta.mNumQHeadsOverKV, kernelMeta.mMTileSize, kernelMeta.mTokensPerPage, kernelMeta.mPagedKVCache,
                kernelMeta.mMultiQueryTokens, false, std::nullopt};

            mFunctions.insert(std::make_pair(hash_key, funcInfo));
        }
    }

    bool supportConfig(XQAParams const& xqaParams) const
    {
        unsigned int head_size = xqaParams.head_size;
        int num_q_heads = xqaParams.num_q_heads;
        int num_kv_heads = xqaParams.num_kv_heads;
        TLLM_CHECK_WITH_INFO(num_q_heads % num_kv_heads == 0, "numQHeads should be multiple of numKVHeads.");
        unsigned int num_q_heads_over_kv = num_q_heads / num_kv_heads;
        unsigned int beam_width = xqaParams.beam_width;
        // MultiQueryToken kernels can support any num_q_heads_over_kv that is power of 2.
        unsigned int kernel_num_q_heads_over_kv = xqaParams.multi_query_tokens ? 0 : num_q_heads_over_kv;
        unsigned int m_tilesize;
        if (xqaParams.multi_query_tokens)
        {
            // MultiQueryToken kernels can handle either 16/32 for M direction per CTA.
            m_tilesize = xqaParams.generation_input_length <= 16 ? 16 : 32;
        }
        else
        {
            m_tilesize = num_q_heads_over_kv;
        }

        // precompiled XQA does not support param is_fp8_output in hash key
        XQAKernelRuntimeHashKey hash_key
            = {xqaParams.kv_cache_data_type, head_size, beam_width, kernel_num_q_heads_over_kv, m_tilesize,
                xqaParams.paged_kv_cache ? static_cast<unsigned int>(xqaParams.tokens_per_block) : 0,
                xqaParams.paged_kv_cache, xqaParams.multi_query_tokens, 0, /* xqa jit param is_fp8_output */
                std::nullopt};
        auto const findIter = mFunctions.find(hash_key);
        return findIter != mFunctions.end();
    }

    bool mayHavePerfGain(XQAParams const& xqaParams, int multiprocessor_count) const
    {
        // NOTE: only XQA supports multi_query_tokens (Medusa mode).
        if (mForceXQA || xqaParams.multi_query_tokens)
        {
            return true;
        }
        int num_kv_heads = xqaParams.num_kv_heads;
        int batch_size = static_cast<int>(xqaParams.batch_size);
        int multi_block_count = 1;
        if (xqaParams.multi_block_mode)
        {
            int history_length = xqaParams.max_past_kv_length;
            multi_block_count = history_length / kMinHistoryTokensPerBlock;
        }
        int block_count = num_kv_heads * batch_size * multi_block_count;
        return static_cast<float>(block_count) * kEnableMinBlockFactor >= static_cast<float>(multiprocessor_count);
    }

    template <typename T, typename KVCacheBuffer>
    void run(XQAParams const& xqaParams, KVCacheBuffer const& kv_cache_buffer, int multiprocessor_count,
        cudaStream_t const& stream) const
    {
        unsigned int head_size = xqaParams.head_size;
        int num_q_heads = xqaParams.num_q_heads;
        int num_kv_heads = xqaParams.num_kv_heads;
        TLLM_CHECK_WITH_INFO(num_q_heads % num_kv_heads == 0, "numQHeads should be multiple of numKVHeads.");
        unsigned int num_q_heads_over_kv = num_q_heads / num_kv_heads;
        unsigned int beam_width = xqaParams.beam_width;
        unsigned int batch_beam_size = xqaParams.batch_size * beam_width;

        const KvCacheDataType cache_type = xqaParams.kv_cache_quant_mode.hasInt8KvCache()
            ? KvCacheDataType::INT8
            : (xqaParams.kv_cache_quant_mode.hasFp8KvCache() ? KvCacheDataType::FP8 : KvCacheDataType::BASE);

        XQALaunchParam<KVCacheBuffer> launchParams;
        bool const needOutputCvt = (xqaParams.fp8_out_scale != nullptr);
        void* inputScratch = nullptr;
        buildXQALaunchParams(launchParams, inputScratch, needOutputCvt, xqaParams, kv_cache_buffer);

        // Build cu_seqlens, padding_offset, and rotary inv freq tensors
        BuildDecoderInfoParams<T> decoder_params{};
        decoder_params.seqQOffsets = launchParams.cu_seq_lens;
        decoder_params.seqQLengths = xqaParams.spec_decoding_generation_lengths;
        decoder_params.seqKVLengths = xqaParams.sequence_lengths;
        decoder_params.tokensInfo = launchParams.tokens_info;
        decoder_params.batchSize = int(batch_beam_size);
        decoder_params.maxQSeqLength = xqaParams.generation_input_length;
        decoder_params.numTokens = xqaParams.total_num_input_tokens;
        decoder_params.removePadding = true;
        TLLM_CHECK_WITH_INFO(!xqaParams.multi_query_tokens || xqaParams.spec_decoding_generation_lengths != nullptr,
            "Spec_decoding_generation_lengths must be provided.");
        // Rotary embedding inv_freq buffer.
        decoder_params.rotaryEmbeddingScale = xqaParams.rotary_embedding_scale;
        decoder_params.rotaryEmbeddingBase = xqaParams.rotary_embedding_base;
        decoder_params.rotaryEmbeddingDim = xqaParams.rotary_embedding_dim;
        decoder_params.rotaryScalingType = xqaParams.rotary_embedding_scale_type;
        decoder_params.rotaryEmbeddingInvFreq = launchParams.rotary_inv_freq_buf;
        decoder_params.rotaryEmbeddingInvFreqCache = xqaParams.rotary_embedding_inv_freq_cache;
        decoder_params.rotaryEmbeddingMaxPositions = xqaParams.rotary_embedding_max_positions;
        // The rotary_embedding_inv_freq_cache for QKVPreprocessing.
        // Use the xqaParams.rotary_embedding_inv_freq_cache input when the buildDecoderInfoKernel is skipped.
        float const* rotary_inv_freq_buf = xqaParams.rotary_embedding_inv_freq_cache;
        if (decoder_params.isBuildDecoderInfoKernelNeeded() || xqaParams.multi_query_tokens)
        {
            rotary_inv_freq_buf = launchParams.rotary_inv_freq_buf;
            invokeBuildDecoderInfo(decoder_params, stream);
        }
        sync_check_cuda_error(stream);

        // IDEA: Store rotary_processed Q buffer to output buffer.
        // NOTE: MHA kernels should read kv cache that has already been appended with new tokens' kv cache.
        void* xqa_q_input_ptr = inputScratch;
        // The preprocessing kernel that applies RoPE and updates kv cache.
        QKVPreprocessingParams<T, KVCacheBuffer> preprocessingParams;
        memset(&preprocessingParams, 0, sizeof(preprocessingParams));
        // Set parameters.
        preprocessingParams.qkv_input = static_cast<T*>(const_cast<void*>(xqaParams.qkv));
        preprocessingParams.q_output = static_cast<T*>(xqa_q_input_ptr);
        preprocessingParams.kv_cache_buffer = kv_cache_buffer;
        preprocessingParams.kv_cache_block_scales_buffer = {};
        preprocessingParams.qkv_bias = static_cast<T const*>(xqaParams.qkv_bias);
        // Buffers.
        preprocessingParams.logn_scaling = xqaParams.logn_scaling_ptr;
        preprocessingParams.tokens_info = launchParams.tokens_info;
        preprocessingParams.seq_lens = xqaParams.spec_decoding_generation_lengths;
        preprocessingParams.cache_seq_lens = xqaParams.sequence_lengths;
        preprocessingParams.cu_seq_lens = xqaParams.multi_query_tokens ? launchParams.cu_seq_lens : nullptr;
        preprocessingParams.rotary_embedding_inv_freq = rotary_inv_freq_buf;
        preprocessingParams.rotary_coef_cache_buffer = xqaParams.rotary_cos_sin;
        preprocessingParams.qkv_scale_orig_quant = xqaParams.kv_scale_orig_quant;
        preprocessingParams.spec_decoding_position_offsets = xqaParams.spec_decoding_position_offsets;
        preprocessingParams.mrope_position_deltas = xqaParams.mrope_position_deltas;
        // Scalar parameters.
        preprocessingParams.batch_size = int(batch_beam_size);
        preprocessingParams.max_input_seq_len = xqaParams.generation_input_length;
        preprocessingParams.max_kv_seq_len = xqaParams.max_past_kv_length;
        preprocessingParams.cyclic_kv_cache_len = xqaParams.cyclic_attention_window_size;
        preprocessingParams.sink_token_len = xqaParams.sink_token_length;
        preprocessingParams.token_num = xqaParams.total_num_input_tokens;
        preprocessingParams.remove_padding = true;
        preprocessingParams.cross_attention = false;
        preprocessingParams.head_num = xqaParams.num_q_heads;
        preprocessingParams.kv_head_num = xqaParams.num_kv_heads;
        preprocessingParams.qheads_per_kv_head = xqaParams.num_q_heads / xqaParams.num_kv_heads;
        preprocessingParams.size_per_head = xqaParams.head_size;
        preprocessingParams.rotary_embedding_dim = xqaParams.rotary_embedding_dim;
        preprocessingParams.rotary_embedding_base = xqaParams.rotary_embedding_base;
        preprocessingParams.rotary_scale_type = xqaParams.rotary_embedding_scale_type;
        preprocessingParams.rotary_embedding_scale = xqaParams.rotary_embedding_scale;
        preprocessingParams.rotary_embedding_max_positions = xqaParams.rotary_embedding_max_positions;
        preprocessingParams.position_embedding_type = xqaParams.position_embedding_type;
        preprocessingParams.position_shift_enabled = xqaParams.position_shift_enabled;
        preprocessingParams.cache_type = cache_type;
        preprocessingParams.separate_q_kv_output = true;
        preprocessingParams.quantized_fp8_output = false;
        preprocessingParams.generation_phase = true;
        preprocessingParams.multi_processor_count = multiprocessor_count;
        preprocessingParams.rotary_vision_start = xqaParams.rotary_vision_start;
        preprocessingParams.rotary_vision_length = xqaParams.rotary_vision_length;

        invokeQKVPreprocessing<T, KVCacheBuffer>(preprocessingParams, stream);
        sync_check_cuda_error(stream);

        XQAKernelRuntimeHashKey hash_key = getRuntimeHashKeyFromXQAParams(xqaParams, false, mSM);
        auto const findIter = mFunctions.find(hash_key);

        TLLM_CHECK_WITH_INFO(findIter != mFunctions.end(), "XQAKernelFunc not found.");

        auto const& kernelMeta = mKernelMeta[findIter->second.mMetaInfoIndex];
        const CUfunction func = reinterpret_cast<CUfunction>(findIter->second.mDeviceFunction);
        unsigned int const shared_mem_bytes = findIter->second.mSharedMemBytes;
        auto const kernelType = findIter->second.mKernelType;

        if (xqaParams.multi_query_tokens)
        {
            // MultiQueryTokens (generation_input_length > 1) need extra parameters (like qSeqLen, headGrpSize, and
            // mask). Input parameters for MultiQueryTokens kernels.
            unsigned int headGrpSize = num_q_heads_over_kv;
            // Use mTileSize = 16 kernels when qSeqLen <= 16.
            unsigned int qSeqLen = static_cast<unsigned int>(xqaParams.generation_input_length);
            unsigned int mTileSize = qSeqLen <= 16 ? 16 : 32;
            unsigned int nbTokenBlocksPerGrp = divUp(qSeqLen * headGrpSize, mTileSize);
            int const* maskPtr = xqaParams.spec_decoding_packed_mask;
            int const* cuQSeqLens = launchParams.cu_seq_lens;
            unsigned int maxQSeqLen = xqaParams.spec_decoding_is_generation_length_variable ? // true for ReDrafter
                xqaParams.spec_decoding_max_generation_length
                                                                                            : qSeqLen;
            // TODO: merge SingleQueryToken params and MultiQueryTokens params into one kernelParams.
            void* kernelParams[] = {&maxQSeqLen, &launchParams.num_k_heads, &headGrpSize, &cuQSeqLens,
                &launchParams.output, &xqa_q_input_ptr, &maskPtr, &launchParams.kvCacheParams, &launchParams.batch_size,
                &launchParams.kv_scale_quant_orig, &launchParams.scratch};
            // precompiled XQA Spec-dec kernel does not support multi-block mode
            int multi_block = 1;
            TLLM_CU_CHECK(mDriver->cuLaunchKernel(func, multi_block, xqaParams.num_kv_heads * nbTokenBlocksPerGrp,
                xqaParams.batch_size, 128, 1, 2, shared_mem_bytes, stream, kernelParams, nullptr));
        }
        else
        {
            bool const isGmmaKernel = (kernelType == XQAKernelType::kHOPPER_WARP_SPECIALIZED);
            TLLM_CHECK(isGmmaKernel
                == (mSM == kSM_90 && xqaParams.kv_cache_data_type == XQADataType::DATA_TYPE_E4M3
                    && xqaParams.beam_width == 1));
            constexpr uint32_t kMAX_NB_KERNEL_PARAMS = 11;
            uint32_t const maxNbKernelParams = (isGmmaKernel ? 11 : 10);
            uint32_t idxNextParam = 0;
            void* kernelParams[kMAX_NB_KERNEL_PARAMS];
            auto appendParam = [&](auto* p) mutable
            {
                TLLM_CHECK(idxNextParam < maxNbKernelParams);
                kernelParams[idxNextParam++] = p;
            };
            appendParam(&launchParams.num_k_heads);
            appendParam(&launchParams.output);
            appendParam(&xqa_q_input_ptr);
            appendParam(&launchParams.kvCacheParams);
            if (xqaParams.beam_width > 1)
            {
                appendParam(&launchParams.beamSearchParams.value());
            }
            appendParam(&launchParams.batch_size);
            appendParam(&launchParams.kv_scale_quant_orig);
            CUtensorMap tensorMap{};
            if (isGmmaKernel)
            {
                tensorMap = makeTensorMapForHopperXqaKVCache(mDriver, xqaParams, kv_cache_buffer);
                appendParam(&tensorMap);
            }
            appendParam(&launchParams.semaphores);
            appendParam(&launchParams.scratch);
            kernelParams[idxNextParam] = nullptr; // one extra nullptr at end as guard.
            int multi_block = 1;
            if (xqaParams.multi_block_mode)
            {
                multi_block = computeMultiBlockCount(xqaParams, xqaParams.batch_size, multiprocessor_count);
            }
            TLLM_CU_CHECK(mDriver->cuLaunchKernel(func, multi_block, xqaParams.num_kv_heads, xqaParams.batch_size, 128,
                1, isGmmaKernel ? 3 : 2, shared_mem_bytes, stream, kernelParams, nullptr));
        }

        sync_check_cuda_error(stream);

        if (needOutputCvt)
        {
            tensorrt_llm::kernels::invokeConversion<__nv_fp8_e4m3, T>(static_cast<__nv_fp8_e4m3*>(xqaParams.output),
                static_cast<T const*>(launchParams.output),
                xqaParams.head_size * xqaParams.num_q_heads * xqaParams.total_num_input_tokens, xqaParams.fp8_out_scale,
                stream);
            sync_check_cuda_error(stream);
        }
    }

protected:
    std::shared_ptr<tensorrt_llm::common::CUDADriverWrapper> mDriver;

    Data_type mDataType;
    TKernelMeta const* mKernelMeta;
    unsigned int mKernelMetaCount;
    unsigned int mSM;
    std::unordered_map<unsigned long long const*, CUlibrary> mCuLibs;

    bool mForceXQA = false;

    struct XQAKernelFuncInfo
    {
        unsigned int mMetaInfoIndex;
        unsigned int mSharedMemBytes;
        CUkernel mDeviceFunction;
        XQAKernelType mKernelType;
    };

    std::unordered_map<XQAKernelRuntimeHashKey, XQAKernelFuncInfo, XQAKernelRuntimeHasher> mFunctions;
};

class XQAKernelLoader
{
public:
    XQAKernelList const* getXQAKernels(Data_type type, unsigned int sm)
    {
        static std::mutex s_mutex;
        std::lock_guard<std::mutex> lg(s_mutex);

        XQAKernelLoadHashKey hash_key{type, sm};

        auto const findIter = mKernels.find(hash_key);
        if (findIter == mKernels.end())
        {
            XQAKernelList* newKernel = new XQAKernelList{type, sm};
            newKernel->loadXQAKernels();
            mKernels.insert(std::make_pair(hash_key, std::unique_ptr<XQAKernelList>(newKernel)));
            return newKernel;
        }
        return findIter->second.get();
    }

    static XQAKernelLoader& Get()
    {
        int device_id = tensorrt_llm::common::getDevice();
        static std::unique_ptr<XQAKernelLoader> s_factory[32] = {nullptr};
        if (s_factory[device_id] == nullptr)
        {
            assert(device_id <= 32);
            s_factory[device_id] = std::make_unique<XQAKernelLoader>(XQAKernelLoader());
        }

        return *(s_factory[device_id]);
    }

private:
    XQAKernelLoader() = default;

    std::unordered_map<XQAKernelLoadHashKey, const std::unique_ptr<XQAKernelList>, XQAKernelLoadHasher> mKernels;
};

inline XQAKernelList const* getXQAKernels(Data_type type, unsigned int sm)
{
    if (sm == kSM_121)
    {
        sm = kSM_120;
    }
    return XQAKernelLoader::Get().getXQAKernels(type, sm);
}

#define XQA_KERNEL_RUN(DATA_TYPE)                                                                                      \
    xqa_kernel->template run<DATA_TYPE, KVCacheBuffer>(xqa_params, kv_cache_buffer, multi_processor_count, stream);

template <typename KVCacheBuffer>
void DecoderXQAImplPrecompiled::runDispatchBuffer(
    XQAParams const& xqa_params, KVCacheBuffer const& kv_cache_buffer, cudaStream_t const& stream)
{
    XQAKernelList const* xqa_kernel = getXQAKernels(mRunner->mDataType, tensorrt_llm::common::getSMVersion());
    int multi_processor_count = mRunner->mMultiProcessorCount;
    if (mRunner->mDataType == DATA_TYPE_FP16)
    {
        XQA_KERNEL_RUN(__half);
    }
    else
    {
        XQA_KERNEL_RUN(__nv_bfloat16);
    }
}

#undef XQA_KERNEL_RUN

#define SUPPORT_RETURN_FALSE(X)                                                                                        \
    {                                                                                                                  \
        TLLM_LOG_DEBUG("XQA is not used. Reason: %s", X);                                                              \
        return false;                                                                                                  \
    }

bool DecoderXQAImplPrecompiled::shouldUse(XQAParams const& xqaParams, bool forConfigurePlugin)
{
    if (!(xqaParams.data_type == DATA_TYPE_FP16 || xqaParams.data_type == DATA_TYPE_BF16))
    {
        SUPPORT_RETURN_FALSE("data type");
    }
    bool const isGPTJBeam4Kernel = (xqaParams.head_size == 256 && xqaParams.beam_width == 4 && xqaParams.paged_kv_cache
        && (xqaParams.tokens_per_block == 64 || xqaParams.tokens_per_block == 128));
    if (xqaParams.head_size != 64 && xqaParams.head_size != 128 && xqaParams.head_size != 256 && !isGPTJBeam4Kernel)
    {
        SUPPORT_RETURN_FALSE("head_size");
    }
    if (xqaParams.unidirectional != 1)
    {
        SUPPORT_RETURN_FALSE("unidirectional");
    }
    if (xqaParams.q_scaling != 1.0f)
    {
        SUPPORT_RETURN_FALSE("q_scaling");
    }
    if (xqaParams.mask_type != tensorrt_llm::kernels::AttentionMaskType::CAUSAL)
    {
        SUPPORT_RETURN_FALSE("mask_type");
    }
    if (xqaParams.cross_attention)
    {
        SUPPORT_RETURN_FALSE("cross_attention");
    }
    // Only support 32/64/128 tokens per block.
    if (xqaParams.paged_kv_cache && xqaParams.tokens_per_block != 32 && xqaParams.tokens_per_block != 64
        && xqaParams.tokens_per_block != 128)
    {
        SUPPORT_RETURN_FALSE("paged_kv_cache");
    }
    if (xqaParams.beam_width != 1 && !isGPTJBeam4Kernel)
    {
        SUPPORT_RETURN_FALSE("beam_width");
    }
    if (xqaParams.cyclic_attention_window_size != xqaParams.max_attention_window_size)
    {
        SUPPORT_RETURN_FALSE("cyclic_attention_window_size != max_attention_window_size");
    }
    if (xqaParams.position_shift_enabled || xqaParams.sink_token_length > 0)
    {
        SUPPORT_RETURN_FALSE("streaming-llm");
    }

    // OPTIMIZE: For the standard generation-phase MHA, there are still extra limitations.
    // NOTE: Medusa mode = Multi_query_tokens > 1.
    int const nbQHeads = xqaParams.num_q_heads;
    int const nbKVHeads = xqaParams.num_kv_heads;
    int const nbQHeadsPerKV = nbQHeads / nbKVHeads;
    // MultiQueryTokens mode (Medusa mode) can support any nbQHeadsPerKV.
    if (!xqaParams.multi_query_tokens)
    {
        if (nbQHeadsPerKV != 16 && nbQHeadsPerKV != 8 && nbQHeadsPerKV != 1)
        {
            SUPPORT_RETURN_FALSE("nbHeads");
        }
    }

    if (!forConfigurePlugin)
    {
        // Inference time checks.
        if (xqaParams.host_past_key_value_lengths == nullptr)
        {
            SUPPORT_RETURN_FALSE("host_past_key_value_lengths");
        }
        for (int i = 0; i < xqaParams.batch_size; ++i)
        {
            // Only checks for non-medusa case, because medusa may not accept all tokens in host_past_key_value_lengths.
            // FIXME(perkzz): medusa should check for sliding-window attention.
            if (!xqaParams.multi_query_tokens
                && xqaParams.host_past_key_value_lengths[i] + 1 > xqaParams.max_attention_window_size)
            {
                SUPPORT_RETURN_FALSE("sliding window attention");
            }
        }
    }

    XQAKernelList const* xqa_kernel = getXQAKernels(mRunner->mDataType, tensorrt_llm::common::getSMVersion());
    bool supportConfig = xqa_kernel->supportConfig(xqaParams);
    if (!supportConfig)
    {
        SUPPORT_RETURN_FALSE("supportConfig");
    }
    bool mayHavePerfGain = xqa_kernel->mayHavePerfGain(xqaParams, mRunner->mMultiProcessorCount);
    if (!mayHavePerfGain)
    {
        SUPPORT_RETURN_FALSE("mayHavePerfGain");
    }
    return true;
}

#undef SUPPORT_RETURN_FALSE

void DecoderXQAImplPrecompiled::prepare(XQAParams const&)
{
    // Intentionally do nothing.
}

void DecoderXQAImplPrecompiled::runWithKVLinearBuffer(
    XQAParams const& xqa_params, KVLinearBuffer const& kv_linear_buffer, cudaStream_t const& stream)
{
    runDispatchBuffer<KVLinearBuffer>(xqa_params, kv_linear_buffer, stream);
}

void DecoderXQAImplPrecompiled::runWithKVBlockArray(
    XQAParams const& xqa_params, KVBlockArray const& kv_block_array, cudaStream_t const& stream)
{
    runDispatchBuffer<KVBlockArray>(xqa_params, kv_block_array, stream);
}

} // namespace tensorrt_llm::kernels
