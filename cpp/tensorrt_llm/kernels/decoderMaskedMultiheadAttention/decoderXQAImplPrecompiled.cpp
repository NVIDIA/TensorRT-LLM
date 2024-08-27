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

namespace tensorrt_llm
{
namespace kernels
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

            CUmodule hmod{0};
            auto findModuleIter = mModules.find(kernelMeta.mCubin);
            if (findModuleIter != mModules.end())
            {
                hmod = findModuleIter->second;
            }
            else
            {
                cuErrCheck(mDriver->cuModuleLoadData(&hmod, kernelMeta.mCubin), mDriver);
                mModules.insert(std::make_pair(kernelMeta.mCubin, hmod));
            }

            XQAKernelFuncInfo funcInfo{};
            funcInfo.mMetaInfoIndex = i;
            cuErrCheck(mDriver->cuModuleGetFunction(&funcInfo.mDeviceFunction, hmod, kernelMeta.mFuncName), mDriver);
            funcInfo.mSharedMemBytes = getGlobalVar<uint32_t>(mDriver, hmod, "smemSize", true).value();
            funcInfo.mKernelType = getGlobalVar<XQAKernelType>(mDriver, hmod, "kernelType", false)
                                       .value_or(XQAKernelType::kAMPERE_WARP_SPECIALIZED);

            /* Set 46KB threshold here because we have to take static/driver shared memory into consideration. */
            if (funcInfo.mSharedMemBytes >= 46 * 1024)
            {
                cuErrCheck(mDriver->cuFuncSetAttribute(funcInfo.mDeviceFunction,
                               CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, funcInfo.mSharedMemBytes),
                    mDriver);
            }
            XQAKernelRuntimeHashKey hash_key{kernelMeta.mKVDataType, kernelMeta.mHeadDim, kernelMeta.mBeamWidth,
                kernelMeta.mNumQHeadsOverKV, kernelMeta.mMTileSize, kernelMeta.mTokensPerPage, kernelMeta.mPagedKVCache,
                kernelMeta.mMultiQueryTokens};

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

        XQAKernelRuntimeHashKey hash_key
            = {xqaParams.kv_cache_data_type, head_size, beam_width, kernel_num_q_heads_over_kv, m_tilesize,
                xqaParams.paged_kv_cache ? static_cast<unsigned int>(xqaParams.tokens_per_block) : 0,
                xqaParams.paged_kv_cache, xqaParams.multi_query_tokens};
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
            int history_length = xqaParams.timestep;
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
        void* ioScratch = nullptr;
        buildXQALaunchParams(launchParams, ioScratch, xqaParams, kv_cache_buffer);
        bool const needOutputCvt = (xqaParams.fp8_out_scale != nullptr);
        if (needOutputCvt)
        {
            launchParams.output = ioScratch;
        }

        // Build cu_seqlens, padding_offset, and rotary inv freq tensors
        BuildDecoderInfoParams<T> decoder_params;
        memset(&decoder_params, 0, sizeof(decoder_params));
        decoder_params.seqQOffsets = launchParams.cu_seq_lens;
        decoder_params.seqQLengths = xqaParams.spec_decoding_generation_lengths;
        decoder_params.seqKVLengths = xqaParams.sequence_lengths;
        decoder_params.batchSize = int(batch_beam_size);
        decoder_params.maxQSeqLength = xqaParams.generation_input_length;
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

        invokeBuildDecoderInfo(decoder_params, stream);
        sync_check_cuda_error();

        // IDEA: Store rotary_processed Q buffer to output buffer.
        // NOTE: MHA kernels should read kv cache that has already been appended with new tokens' kv cache.
        void* xqa_q_input_ptr = ioScratch;
        QKVPreprocessingParams<T, KVCacheBuffer> preprocessingParms{static_cast<T*>(const_cast<void*>(xqaParams.qkv)),
            nullptr, static_cast<T*>(xqa_q_input_ptr), kv_cache_buffer, static_cast<T const*>(xqaParams.qkv_bias),
            xqaParams.spec_decoding_generation_lengths, xqaParams.sequence_lengths,
            xqaParams.multi_query_tokens ? launchParams.cu_seq_lens : nullptr, launchParams.rotary_inv_freq_buf,
            (float2 const*) nullptr, xqaParams.kv_scale_orig_quant, xqaParams.spec_decoding_position_offsets,
            int(batch_beam_size), xqaParams.generation_input_length, xqaParams.timestep,
            xqaParams.cyclic_attention_window_size, xqaParams.sink_token_length,
            int(xqaParams.batch_size * beam_width * xqaParams.generation_input_length),
            /*remove_padding*/ true, xqaParams.num_q_heads, xqaParams.num_kv_heads,
            xqaParams.num_q_heads / xqaParams.num_kv_heads, xqaParams.head_size, xqaParams.rotary_embedding_dim,
            xqaParams.rotary_embedding_base, xqaParams.rotary_embedding_scale_type, xqaParams.rotary_embedding_scale,
            xqaParams.rotary_embedding_max_positions, xqaParams.position_embedding_type,
            xqaParams.position_shift_enabled, cache_type, true, false, multiprocessor_count,
            xqaParams.rotary_vision_start, xqaParams.rotary_vision_length};

        invokeQKVPreprocessing<T, KVCacheBuffer>(preprocessingParms, stream);
        sync_check_cuda_error();

        XQAKernelRuntimeHashKey hash_key = getRuntimeHashKeyFromXQAParams(xqaParams);
        auto const findIter = mFunctions.find(hash_key);

        TLLM_CHECK_WITH_INFO(findIter != mFunctions.end(), "XQAKernelFunc not found.");

        auto const& kernelMeta = mKernelMeta[findIter->second.mMetaInfoIndex];
        const CUfunction func = findIter->second.mDeviceFunction;
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
            int multi_block = 1;
            if (xqaParams.multi_block_mode)
            {
                multi_block = computeMultiBlockCount(xqaParams, xqaParams.batch_size, multiprocessor_count);
                check_cuda_error(cudaMemsetAsync(xqaParams.workspaces, 0,
                    sizeof(int) * xqaParams.batch_size * qSeqLen * xqaParams.num_kv_heads, stream));
                sync_check_cuda_error();
            }
            cuErrCheck(mDriver->cuLaunchKernel(func, multi_block, xqaParams.num_kv_heads * nbTokenBlocksPerGrp,
                           xqaParams.batch_size, 128, 1, 2, shared_mem_bytes, stream, kernelParams, nullptr),
                mDriver);
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
                tensorMap = makeTensorMapForKVCache(mDriver, xqaParams, kv_cache_buffer);
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
            cuErrCheck(mDriver->cuLaunchKernel(func, multi_block, xqaParams.num_kv_heads, xqaParams.batch_size, 128, 1,
                           isGmmaKernel ? 3 : 2, shared_mem_bytes, stream, kernelParams, nullptr),
                mDriver);
        }

        sync_check_cuda_error();

        if (needOutputCvt)
        {
            tensorrt_llm::kernels::invokeConversion<__nv_fp8_e4m3, T>(static_cast<__nv_fp8_e4m3*>(xqaParams.output),
                static_cast<T const*>(launchParams.output),
                xqaParams.head_size * xqaParams.num_q_heads * xqaParams.total_num_input_tokens, xqaParams.fp8_out_scale,
                stream);
            sync_check_cuda_error();
        }
    }

protected:
    std::shared_ptr<tensorrt_llm::common::CUDADriverWrapper> mDriver;

    Data_type mDataType;
    TKernelMeta const* mKernelMeta;
    unsigned int mKernelMetaCount;
    unsigned int mSM;
    std::unordered_map<unsigned long long const*, CUmodule> mModules;

    bool mForceXQA = false;

    struct XQAKernelFuncInfo
    {
        unsigned int mMetaInfoIndex;
        unsigned int mSharedMemBytes;
        CUfunction mDeviceFunction;
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
    if (xqaParams.head_size != 128 && xqaParams.head_size != 256 && !isGPTJBeam4Kernel)
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
    // Only support 64/128 tokens per block.
    if (xqaParams.paged_kv_cache && xqaParams.tokens_per_block != 64 && xqaParams.tokens_per_block != 128)
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
    return xqa_kernel->supportConfig(xqaParams)
        && xqa_kernel->mayHavePerfGain(xqaParams, mRunner->mMultiProcessorCount);
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

} // namespace kernels
} // namespace tensorrt_llm
