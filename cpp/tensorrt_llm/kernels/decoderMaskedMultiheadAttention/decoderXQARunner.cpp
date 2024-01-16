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

#include "decoderXQARunner.h"

#include <assert.h>
#include <string.h>

#include <mutex>
#include <unordered_map>

#include "tensorrt_llm/common/cudaDriverWrapper.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/envUtils.h"
#include "tensorrt_llm/kernels/decoderMaskedMultiheadAttention/cubin/xqa_kernel_cubin.h"
#include "tensorrt_llm/kernels/kvCacheUtils.h"
#include "tensorrt_llm/kernels/unfusedAttentionKernels.h"

namespace tensorrt_llm
{
namespace kernels
{

struct XQAKernelLoadHashKey
{
    Data_type data_type;
    unsigned int sm;

    bool operator==(const XQAKernelLoadHashKey other) const
    {
        return data_type == other.data_type && sm == other.sm;
    }
};

struct XQAKernelLoadHasher
{
    size_t operator()(const XQAKernelLoadHashKey& s) const
    {
        size_t key = s.data_type;
        key <<= 16;
        key ^= s.sm;
        return key;
    }
};

struct XQAKernelRuntimeHashKey
{
    Data_type kv_data_type;
    unsigned int head_size;
    unsigned int beam_size;
    unsigned int num_q_heads_per_kv;
    unsigned int m_tilesize;
    unsigned int tokens_per_page;
    bool paged_kv_cache;
    bool multi_query_tokens;

    bool operator==(const XQAKernelRuntimeHashKey other) const
    {
        return kv_data_type == other.kv_data_type && head_size == other.head_size
            && num_q_heads_per_kv == other.num_q_heads_per_kv && beam_size == other.beam_size
            && multi_query_tokens == other.multi_query_tokens && m_tilesize == other.m_tilesize
            && tokens_per_page == other.tokens_per_page && paged_kv_cache == other.paged_kv_cache;
    }
};

struct XQAKernelRuntimeHasher
{
    size_t operator()(const XQAKernelRuntimeHashKey& s) const
    {
        size_t key = s.kv_data_type;
        key <<= 16;
        key ^= s.head_size;
        key <<= 8;
        key ^= s.num_q_heads_per_kv;
        key <<= 8;
        key ^= s.beam_size;
        key <<= 6;
        key ^= s.m_tilesize;
        key <<= 10;
        key ^= s.tokens_per_page;
        key <<= 1;
        key ^= s.paged_kv_cache;
        key <<= 1;
        key ^= s.multi_query_tokens;
        return key;
    }
};

// NOTE: we use int32_t sequence lengths as gpt attention plugins use int32_t for that.
// XQA kernels assume all length should use uint32_t.
template <bool HasBeam>
struct KVCache
{
    void* data;
    int const* sequence_lengths;
    uint32_t capacity;
    const int32_t* cacheInDir;
};

// NOTE: Linear KV cache and paged KV cache uses the same structure.
template <>
struct KVCache<false>
{
    void* data;
    int const* sequence_lengths;
    // NOTE: max_num_blocks_per_sequence for paged kv cache, max_sequence_length for linear kv cache.
    uint32_t capacity;
};

// XQA kernels assume all integer values should use uint32_t.
template <bool HasBeam>
struct XQALaunchParam
{
    uint32_t num_k_heads;
    void* output;
    const void* qkv;
    KVCache<HasBeam> kvCacheParams;
    uint32_t batch_size;
    const float* kv_scale_quant_orig = nullptr;
    void* scratch = nullptr;
};

// Setup launch params.
template <typename KVCacheBuffer, bool HasBeam>
void buildXQALaunchParams(XQALaunchParam<HasBeam>& launchParams, const XQAParams& params, KVCacheBuffer kv_cache_buffer)
{
    TLLM_CHECK_WITH_INFO(
        params.data_type == DATA_TYPE_FP16 || params.data_type == DATA_TYPE_BF16, "Only fp16 or bf16 supported now.");
    memset(&launchParams, 0, sizeof(XQALaunchParam<HasBeam>));
    launchParams.num_k_heads = params.num_kv_heads;
    launchParams.output = static_cast<uint8_t*>(params.output);
    launchParams.qkv = static_cast<const uint8_t*>(params.qkv);
    launchParams.batch_size = params.batch_size;
    launchParams.scratch = params.workspaces;
    launchParams.kv_scale_quant_orig = params.kv_scale_quant_orig;
    launchParams.kvCacheParams.data = kv_cache_buffer.data;
    launchParams.kvCacheParams.sequence_lengths = params.sequence_lengths;
    // TODO: beam searching has not been implemented yet.
    if constexpr (HasBeam)
    {
        launchParams.kvCacheParams.cacheInDir = params.cache_indir;
    }
    launchParams.kvCacheParams.capacity
        = params.paged_kv_cache ? params.max_blocks_per_sequence : params.max_attention_window_size;
}

// max number of CTAs for each KV head, multiple CTAs for one KV head is multi-block mode.
// this number defines the maximum number when reaches both max_batch_size and max_beam_width.
// If batch_size or beam_width doesn't reach maximum value, it is possible to have more CTAs per KV head than this
// value.
static constexpr int kMaxNbCtaPerKVHeadFactor = 8;
static constexpr int kMinHistoryTokensPerBlock = 512;

class XQAKernelList
{
public:
    using TKernelMeta = XQAKernelMetaInfo;

    XQAKernelList(Data_type type, unsigned int sm)
        : mDataType(type)
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
            const auto& kernelMeta = mKernelMeta[i];
            if (kernelMeta.mSM != mSM || kernelMeta.mDataType != mDataType)
                continue;

            CUmodule hmod{0};
            auto findModuleIter = mModules.find(kernelMeta.mCubin);
            if (findModuleIter != mModules.end())
            {
                hmod = findModuleIter->second;
            }
            else
            {
                cuErrCheck(mDriver.cuModuleLoadData(&hmod, kernelMeta.mCubin), mDriver);
                mModules.insert(std::make_pair(kernelMeta.mCubin, hmod));
            }

            XQAKernelFuncInfo funcInfo{};
            funcInfo.mMetaInfoIndex = i;
            cuErrCheck(mDriver.cuModuleGetFunction(&funcInfo.mDeviceFunction, hmod, kernelMeta.mFuncName), mDriver);
            unsigned int* shmem_dev_ptr = nullptr;
            cuErrCheck(
                mDriver.cuModuleGetGlobal(reinterpret_cast<CUdeviceptr*>(&shmem_dev_ptr), nullptr, hmod, "smemSize"),
                mDriver);
            check_cuda_error(
                cudaMemcpy(&funcInfo.mSharedMemBytes, shmem_dev_ptr, sizeof(unsigned int), cudaMemcpyDeviceToHost));

            /* Set 46KB threshold here because we have to take static/driver shared memory into consideration. */
            if (funcInfo.mSharedMemBytes >= 46 * 1024)
            {
                cuErrCheck(mDriver.cuFuncSetAttribute(funcInfo.mDeviceFunction,
                               CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, funcInfo.mSharedMemBytes),
                    mDriver);
            }
            XQAKernelRuntimeHashKey hash_key{kernelMeta.mKVDataType, kernelMeta.mHeadDim, kernelMeta.mBeamWidth,
                kernelMeta.mNumQHeadsOverKV, kernelMeta.mMTileSize, kernelMeta.mTokensPerPage, kernelMeta.mPagedKVCache,
                kernelMeta.mMultiQueryTokens};

            mFunctions.insert(std::make_pair(hash_key, funcInfo));
        }
    }

    bool supportConfig(const XQAParams& xqaParams) const
    {
        unsigned int head_size = xqaParams.head_size;
        int num_q_heads = xqaParams.num_q_heads;
        int num_kv_heads = xqaParams.num_kv_heads;
        TLLM_CHECK_WITH_INFO(num_q_heads % num_kv_heads == 0, "numQHeads should be multiple of numKVHeads.");
        unsigned int num_q_heads_over_kv = num_q_heads / num_kv_heads;
        unsigned int beam_width = xqaParams.beam_width;
        // MultiQueryToken kernels can support any num_q_heads_over_kv that is power of 2.
        unsigned int kernel_num_q_heads_over_kv = xqaParams.multi_query_tokens ? 0 : num_q_heads_over_kv;
        // MultiQueryToken kernels can handle either 16/32 for M direction per CTA.
        unsigned int m_tilesize = xqaParams.multi_query_tokens ? 16 : num_q_heads_over_kv;

        XQAKernelRuntimeHashKey hash_key = {xqaParams.kv_cache_data_type, head_size, beam_width,
            kernel_num_q_heads_over_kv, m_tilesize, static_cast<unsigned int>(xqaParams.tokens_per_block),
            xqaParams.paged_kv_cache, xqaParams.multi_query_tokens};
        const auto findIter = mFunctions.find(hash_key);
        return findIter != mFunctions.end();
    }

    bool mayHavePerfGain(const XQAParams& xqaParams, int multiprocessor_count) const
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
        static constexpr float kEnableMinBlockFactor = 4.0;
        return static_cast<float>(block_count) * kEnableMinBlockFactor >= static_cast<float>(multiprocessor_count);
    }

    template <typename T, typename KVCacheBuffer, bool HAS_BEAM>
    void run(const XQAParams& xqaParams, KVCacheBuffer& kv_cache_buffer, int2& rotary_kernel_launch_cache,
        int multiprocessor_count, const cudaStream_t& stream) const
    {
        unsigned int head_size = xqaParams.head_size;
        int num_q_heads = xqaParams.num_q_heads;
        int num_kv_heads = xqaParams.num_kv_heads;
        TLLM_CHECK_WITH_INFO(num_q_heads % num_kv_heads == 0, "numQHeads should be multiple of numKVHeads.");
        unsigned int num_q_heads_over_kv = num_q_heads / num_kv_heads;
        unsigned int beam_width = xqaParams.beam_width;

        const KvCacheDataType cache_type = xqaParams.kv_cache_quant_mode.hasInt8KvCache()
            ? KvCacheDataType::INT8
            : (xqaParams.kv_cache_quant_mode.hasFp8KvCache() ? KvCacheDataType::FP8 : KvCacheDataType::BASE);

        // IDEA: Store rotary_processed Q buffer to output buffer.
        // NOTE: MHA kernels should read kv cache that has already been appended with new tokens' kv cache.
        void const* xqa_q_input_ptr = xqaParams.output;
        invokeApplyBiasRopeUpdateKVCache<T, KVCacheBuffer, true>(static_cast<T*>(const_cast<void*>(xqaParams.qkv)),
            static_cast<T*>(const_cast<void*>(xqaParams.output)), kv_cache_buffer,
            static_cast<const T*>(xqaParams.qkv_bias), xqaParams.sequence_lengths, nullptr, nullptr,
            xqaParams.batch_size, xqaParams.generation_input_length, xqaParams.cyclic_attention_window_size,
            xqaParams.sink_token_length, xqaParams.batch_size * beam_width * xqaParams.generation_input_length,
            xqaParams.num_q_heads, xqaParams.num_kv_heads, xqaParams.head_size, xqaParams.rotary_embedding_dim,
            xqaParams.rotary_embedding_base, xqaParams.rotary_embedding_scale_type, xqaParams.rotary_embedding_scale,
            xqaParams.rotary_embedding_max_positions, xqaParams.position_embedding_type,
            xqaParams.medusa_position_offsets, xqaParams.position_shift_enabled, (float*) nullptr, 0, cache_type,
            xqaParams.kv_scale_orig_quant, true, beam_width, rotary_kernel_launch_cache, stream);

        // Use mTileSize = 16 kernels when qSeqLen <= 16.
        unsigned int qSeqLen = static_cast<unsigned int>(xqaParams.generation_input_length);
        unsigned int mTileSize = qSeqLen <= 16 ? 16 : 32;
        // MultiQueryToken kernels can support any num_q_heads_over_kv that is power of 2.
        unsigned int kernel_num_q_heads_over_kv = xqaParams.multi_query_tokens ? 0 : num_q_heads_over_kv;
        // MultiQueryToken kernels can handle either 16/32 for M direction per CTA.
        unsigned int kernel_m_tilesize = xqaParams.multi_query_tokens ? mTileSize : num_q_heads_over_kv;
        XQAKernelRuntimeHashKey hash_key{xqaParams.kv_cache_data_type, head_size, beam_width,
            kernel_num_q_heads_over_kv, kernel_m_tilesize, static_cast<unsigned int>(xqaParams.tokens_per_block),
            xqaParams.paged_kv_cache, xqaParams.multi_query_tokens};
        const auto findIter = mFunctions.find(hash_key);

        TLLM_CHECK_WITH_INFO(findIter != mFunctions.end(), "XQAKernelFunc not found.");

        const auto& kernelMeta = mKernelMeta[findIter->second.mMetaInfoIndex];
        const CUfunction func = findIter->second.mDeviceFunction;
        const unsigned int shared_mem_bytes = findIter->second.mSharedMemBytes;

        XQALaunchParam<HAS_BEAM> launchParams;
        buildXQALaunchParams(launchParams, xqaParams, kv_cache_buffer);
        if (xqaParams.multi_query_tokens)
        {
            // MultiQueryTokens (generation_input_length > 1) need extra parameters (like qSeqLen, log2HeadGrpSize, and
            // mask). Input parameters for MultiQueryTokens kernels.
            unsigned int log2HeadGrpSize = log2(num_q_heads_over_kv);
            unsigned int nbTokenBlocksPerGrp = divUp(qSeqLen << log2HeadGrpSize, mTileSize);
            int const* maskPtr = xqaParams.medusa_packed_mask;
            // TODO: add fp8/int8 kv cache kernels.
            float kvCacheQuantOrig = 1.0f;
            // TODO: merge SingleQueryToken params and MultiQueryTokens params into one kernelParams.
            void* kernelParams[] = {&qSeqLen, &launchParams.num_k_heads, &log2HeadGrpSize, &launchParams.output,
                &xqa_q_input_ptr, &maskPtr, &launchParams.kvCacheParams, &launchParams.batch_size, &kvCacheQuantOrig,
                &launchParams.scratch};
            int multi_block = 1;
            if (xqaParams.multi_block_mode)
            {
                multi_block = computeMultiBlockCount(xqaParams, xqaParams.batch_size, multiprocessor_count);
                cudaMemsetAsync(
                    xqaParams.workspaces, 0, sizeof(int) * xqaParams.batch_size * xqaParams.num_kv_heads, stream);
            }
            cuErrCheck(mDriver.cuLaunchKernel(func, multi_block, xqaParams.num_kv_heads * nbTokenBlocksPerGrp,
                           xqaParams.batch_size, 128, 1, 2, shared_mem_bytes, stream, kernelParams, nullptr),
                mDriver);
        }
        else
        {
            void* kernelParams[]
                = {&launchParams.num_k_heads, &launchParams.output, &xqa_q_input_ptr, &launchParams.kvCacheParams,
                    &launchParams.batch_size, &launchParams.kv_scale_quant_orig, &launchParams.scratch, nullptr};
            int multi_block = 1;
            if (xqaParams.multi_block_mode)
            {
                multi_block = computeMultiBlockCount(xqaParams, xqaParams.batch_size, multiprocessor_count);
                cudaMemsetAsync(
                    xqaParams.workspaces, 0, sizeof(int) * xqaParams.batch_size * xqaParams.num_kv_heads, stream);
            }
            cuErrCheck(mDriver.cuLaunchKernel(func, multi_block, xqaParams.num_kv_heads, xqaParams.batch_size, 128, 1,
                           2, shared_mem_bytes, stream, kernelParams, nullptr),
                mDriver);
        }

        sync_check_cuda_error();
    }

    static int computeMultiBlockCount(const XQAParams& xqaParams, int batch_size, int multiprocessor_count)
    {
        static constexpr int kTargetWaveFactor = 8;
        int multi_block_count = 1;
        int num_kv_heads = xqaParams.num_kv_heads;
        int history_length = xqaParams.timestep;

        multi_block_count = history_length / kMinHistoryTokensPerBlock;
        multi_block_count = std::max(multi_block_count, 1);
        // adjust to kTargetWaveFactor, as already initialized using kMinHistoryTokensPerBlock, only need to decrease.
        double wave_count = (double) batch_size * num_kv_heads * multi_block_count / (double) multiprocessor_count;
        double adj_factor = wave_count / (double) kTargetWaveFactor;
        if (adj_factor > 1.0)
        {
            multi_block_count = floor(multi_block_count / adj_factor);
        }
        multi_block_count = std::max(multi_block_count, 1);

        // add limitation on upper bound.
        multi_block_count = std::min(kMaxNbCtaPerKVHeadFactor, multi_block_count);

        TLLM_CHECK_WITH_INFO(multi_block_count >= 1, "MultiBlock count should be larger than 1");
        return multi_block_count;
    }

protected:
    tensorrt_llm::common::CUDADriverWrapper mDriver;

    Data_type mDataType;
    const TKernelMeta* mKernelMeta;
    unsigned int mKernelMetaCount;
    unsigned int mSM;
    std::unordered_map<const unsigned long long*, CUmodule> mModules;

    bool mForceXQA = false;

    struct XQAKernelFuncInfo
    {
        unsigned int mMetaInfoIndex;
        unsigned int mSharedMemBytes;
        CUfunction mDeviceFunction;
    };

    std::unordered_map<XQAKernelRuntimeHashKey, XQAKernelFuncInfo, XQAKernelRuntimeHasher> mFunctions;
};

class XQAKernelLoader
{
public:
    const XQAKernelList* getXQAKernels(Data_type type, unsigned int sm)
    {
        static std::mutex s_mutex;
        std::lock_guard<std::mutex> lg(s_mutex);

        XQAKernelLoadHashKey hash_key{type, sm};

        const auto findIter = mKernels.find(hash_key);
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

inline const XQAKernelList* getXQAKernels(Data_type type, unsigned int sm)
{
    return XQAKernelLoader::Get().getXQAKernels(type, sm);
}

class DecoderXQARunner::xqaImpl
{
public:
    xqaImpl(const XQADataType data_type, int sm_)
        : mDataType(data_type)
        , sm(sm_)
        , xqaKernel(getXQAKernels(data_type, sm_))
    {
        mMultiProcessorCount = tensorrt_llm::common::getMultiProcessorCount();
    }

    bool shouldUse(const XQAParams& xqaParams)
    {
        return xqaKernel->supportConfig(xqaParams) && xqaKernel->mayHavePerfGain(xqaParams, mMultiProcessorCount);
    }

    template <typename KVCacheBuffer>
    void run(const XQAParams& xqa_params, KVCacheBuffer& kv_cache_buffer, int2& rotary_kernel_launch_cache,
        const cudaStream_t& stream);

private:
    const XQAKernelList* xqaKernel;
    int sm;
    const XQADataType mDataType;
    int mMultiProcessorCount;
};

#define XQA_KERNEL_RUN(DATA_TYPE, HAS_BEAM)                                                                            \
    xqaKernel->template run<DATA_TYPE, KVCacheBuffer, HAS_BEAM>(                                                       \
        xqa_params, kv_cache_buffer, rotary_kernel_launch_cache, mMultiProcessorCount, stream);

template <typename KVCacheBuffer>
void DecoderXQARunner::xqaImpl::run(const XQAParams& xqa_params, KVCacheBuffer& kv_cache_buffer,
    int2& rotary_kernel_launch_cache, const cudaStream_t& stream)
{
    if (xqa_params.beam_width > 1)
    {
        if (mDataType == DATA_TYPE_FP16)
        {
            XQA_KERNEL_RUN(__half, true);
        }
        else
        {
            XQA_KERNEL_RUN(__nv_bfloat16, true);
        }
    }
    else
    {
        if (mDataType == DATA_TYPE_FP16)
        {
            XQA_KERNEL_RUN(__half, false);
        }
        else
        {
            XQA_KERNEL_RUN(__nv_bfloat16, false);
        }
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////

DecoderXQARunner::DecoderXQARunner(
    const XQADataType data_type, int num_heads, int num_kv_heads, int head_size, bool multi_block_mode)
    : pimpl(new xqaImpl(data_type, tensorrt_llm::common::getSMVersion()))
    , mNumHeads(num_heads)
    , mNumKVHeads(num_kv_heads)
    , mHeadSize(head_size)
    , mMultiBlockMode(multi_block_mode)
{
    mMultiProcessorCount = tensorrt_llm::common::getMultiProcessorCount();
}

DecoderXQARunner::~DecoderXQARunner() = default;

namespace
{

template <typename T>
constexpr inline T divUp(T a, T b)
{
    return (a + b - 1) / b;
}

template <typename T>
constexpr inline T roundUp(T a, T b)
{
    return divUp(a, b) * b;
}

} // namespace

size_t DecoderXQARunner::getWorkspaceSize(int max_batch_beam_size)
{
    size_t workspace_size = 0;
    if (mMultiBlockMode)
    {
        int workspaces[4];
        const int max_num_request = max_batch_beam_size;
        uint32_t const nbSeq = mNumKVHeads * max_num_request;
        uint32_t const nbSubSeq = kMaxNbCtaPerKVHeadFactor * nbSeq;
        int group_size = mNumHeads / mNumKVHeads;
        workspaces[0] = sizeof(uint32_t) * nbSeq;
        workspaces[1] = sizeof(float) * roundUp(group_size, 32) * nbSubSeq;
        workspaces[2] = sizeof(float) * roundUp(group_size, 32) * nbSubSeq;
        workspaces[3] = sizeof(__half) * group_size * mHeadSize * nbSubSeq;
        workspace_size = roundUp(workspaces[0], 128) + roundUp(workspaces[1], 128) + roundUp(workspaces[2], 128)
            + roundUp(workspaces[3], 128);
    }
    return workspace_size;
}

bool DecoderXQARunner::shouldUseImpl(const XQAParams& xqaParams)
{
    return pimpl->shouldUse(xqaParams);
}

template <typename KVCacheBuffer>
void DecoderXQARunner::run(const XQAParams& xqa_params, KVCacheBuffer& kv_cache_buffer, const cudaStream_t& stream)
{
    return pimpl->run(xqa_params, kv_cache_buffer, mLaunchGridBlockCache, stream);
}

template void DecoderXQARunner::run(
    const XQAParams& xqa_params, KVLinearBuffer& kv_cache_buffer, const cudaStream_t& stream);

template void DecoderXQARunner::run(
    const XQAParams& xqa_params, KVBlockArray& kv_cache_buffer, const cudaStream_t& stream);

} // namespace kernels

} // namespace tensorrt_llm
