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
    unsigned int num_q_heads_per_kv;
    unsigned int beam_size;

    bool operator==(const XQAKernelRuntimeHashKey other) const
    {
        return kv_data_type == other.kv_data_type && head_size == other.head_size
            && num_q_heads_per_kv == other.num_q_heads_per_kv && beam_size == other.beam_size;
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
        return key;
    }
};

template <bool HasBeam>
struct KVCache
{
    void* data;
    uint32_t capacity;
    uint32_t size;
    std::conditional_t<HasBeam, const int32_t*, int32_t[]> cacheInDir;
};

template <bool HasBeam>
struct XQALaunchParam
{
private:
    static constexpr int kMaxBatchSizePerWave = 144;

public:
    static inline int GetMaxBatchSizePerWave(int sm_count)
    {
        return std::min(kMaxBatchSizePerWave, sm_count);
    }

    uint32_t num_k_heads;
    void* output;
    const void* qkv;
#ifdef USE_KV_SCALE
    const float* kv_scale_orig_quant = nullptr;
#endif
    // Max 3K size
    KVCache<HasBeam> cacheList[kMaxBatchSizePerWave];
    int batch_size;
    const float* kv_scale_quant_orig = nullptr;
    void* scratch = nullptr;
};

// Returns actual micro_batch_size built
template <bool HasBeam>
int buildXQALaunchParams(XQALaunchParam<HasBeam>& launchParams, const XQAParams& params,
    KVLinearBuffer kv_linear_buffer, int start_batch_idx, int sm_count)
{
    if (start_batch_idx >= params.batch_size)
        return 0;
    TLLM_CHECK_WITH_INFO(params.data_type == DATA_TYPE_FP16, "Only fp16 supported now.");
    size_t elt_size = 0;
    if (params.data_type == DATA_TYPE_FP16)
    {
        elt_size = sizeof(__half);
    }
    memset(&launchParams, 0, sizeof(launchParams));
    int micro_batch_size
        = std::min<int>(XQALaunchParam<HasBeam>::GetMaxBatchSizePerWave(sm_count), params.batch_size - start_batch_idx);
    int hidden_units = params.num_q_heads * params.head_size;
    int hidden_units_kv = params.num_kv_heads * params.head_size;
    size_t qkv_stride = hidden_units + 2 * hidden_units_kv;
    size_t out_stride = hidden_units;
    int batch_beam_start_idx = start_batch_idx * params.beam_width;
    launchParams.output = static_cast<uint8_t*>(params.output) + out_stride * elt_size * batch_beam_start_idx;
    launchParams.qkv = static_cast<const uint8_t*>(params.qkv) + qkv_stride * elt_size * batch_beam_start_idx;
    launchParams.num_k_heads = params.num_kv_heads;
#ifdef USE_KV_SCALE
    launchParams.kv_scale_orig_quant = params.kv_scale_orig_quant;
#endif
    launchParams.kv_scale_quant_orig = params.kv_scale_quant_orig;
    launchParams.batch_size = micro_batch_size;
    launchParams.scratch = params.workspaces;

    int max_context_length = 0;
    int max_past_kv_length = 0;
    if (params.host_context_lengths)
    {
        // TODO: remove this logic, maybe use xqaParams.sequence_lengths inside kernel.
        max_context_length
            = *std::max_element(params.host_context_lengths, params.host_context_lengths + params.batch_size);
        max_past_kv_length = *std::max_element(
            params.host_past_key_value_lengths, params.host_past_key_value_lengths + params.batch_size);
    }
    for (int i = 0; i < micro_batch_size; i++)
    {
        int batch_idx = start_batch_idx + i;
        launchParams.cacheList[i].data = kv_linear_buffer.getKBlockPtr(batch_idx * params.beam_width, 0);
        int current_len = 0;
        // TODO: remove this logic, maybe use xqaParams.sequence_lengths inside kernel.
        if (params.host_context_lengths)
        {
            // the kernel_mha use KV from KVCache, so need plus 1 here.
            current_len = params.host_context_lengths[batch_idx] + max_past_kv_length - max_context_length + 1;
        }
        else
        {
            current_len = params.host_past_key_value_lengths[batch_idx] + 1;
        }
        launchParams.cacheList[i].size = current_len;
        launchParams.cacheList[i].capacity = params.max_attention_window_size;
        if constexpr (HasBeam)
        {
            launchParams.cacheList[i].cacheInDir
                = params.cache_indir + batch_idx * params.beam_width * params.max_attention_window_size;
        }
    }
    return micro_batch_size;
}

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
        const char* enable_xqa_env_var = getenv("TRTLLM_FORCE_XQA");
        if (enable_xqa_env_var != nullptr)
        {
            if (enable_xqa_env_var[0] == '1' && enable_xqa_env_var[1] == '\0')
            {
                mForceXQA = true;
            }
        }
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
            XQAKernelRuntimeHashKey hash_key{
                kernelMeta.mKVDataType, kernelMeta.mHeadDim, kernelMeta.mNumQHeadsOverKV, kernelMeta.mBeamWidth};
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

        XQAKernelRuntimeHashKey hash_key{xqaParams.kv_cache_data_type, head_size, num_q_heads_over_kv, beam_width};
        const auto findIter = mFunctions.find(hash_key);
        return findIter != mFunctions.end();
    }

    bool mayHavePerfGain(const XQAParams& xqaParams, int multiprocessor_count) const
    {
        if (mForceXQA)
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

    template <typename T, bool HAS_BEAM>
    void run(const XQAParams& xqaParams, KVLinearBuffer& kv_linear_buffer, const cudaStream_t& stream,
        int multiprocessor_count, int max_multi_block_slots) const
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

        invokeApplyBiasRopeUpdateKVCache<T, KVLinearBuffer, true>(static_cast<T*>(const_cast<void*>(xqaParams.qkv)),
            nullptr, kv_linear_buffer, static_cast<const T*>(xqaParams.qkv_bias), xqaParams.sequence_lengths, nullptr,
            nullptr, xqaParams.batch_size, 1, xqaParams.cyclic_attention_window_size, xqaParams.batch_size * beam_width,
            xqaParams.num_q_heads, xqaParams.num_kv_heads, xqaParams.head_size, xqaParams.rotary_embedding_dim,
            xqaParams.rotary_embedding_base, xqaParams.rotary_embedding_scale_type, xqaParams.rotary_embedding_scale,
            xqaParams.rotary_embedding_max_positions, xqaParams.position_embedding_type, (float*) nullptr, 0,
            cache_type, xqaParams.kv_scale_orig_quant, false, stream, beam_width);

        XQAKernelRuntimeHashKey hash_key{xqaParams.kv_cache_data_type, head_size, num_q_heads_over_kv, beam_width};
        const auto findIter = mFunctions.find(hash_key);

        TLLM_CHECK_WITH_INFO(findIter != mFunctions.end(), "XQAKernelFunc not found.");

        const auto& kernelMeta = mKernelMeta[findIter->second.mMetaInfoIndex];
        const CUfunction func = findIter->second.mDeviceFunction;
        const unsigned int shared_mem_bytes = findIter->second.mSharedMemBytes;

        int start_batch_idx = 0;
        while (start_batch_idx < xqaParams.batch_size)
        {
            XQALaunchParam<HAS_BEAM> launchParams;
            int micro_batch_size = buildXQALaunchParams(
                launchParams, xqaParams, kv_linear_buffer, start_batch_idx, multiprocessor_count);
            void* kernelParams[]
                = {&launchParams.num_k_heads, &launchParams.output, &launchParams.qkv, &launchParams.cacheList,
                    &launchParams.batch_size, &launchParams.kv_scale_quant_orig, &launchParams.scratch, nullptr};
            int multi_block = 1;
            if (xqaParams.multi_block_mode)
            {
                multi_block
                    = computeMultiBlockCount(xqaParams, micro_batch_size, multiprocessor_count, max_multi_block_slots);
                cudaMemsetAsync(
                    launchParams.scratch, 0, sizeof(int) * micro_batch_size * xqaParams.num_kv_heads, stream);
            }
            cuErrCheck(mDriver.cuLaunchKernel(func, multi_block, xqaParams.num_kv_heads, micro_batch_size, 128, 1, 2,
                           shared_mem_bytes, stream, kernelParams, nullptr),
                mDriver);
            start_batch_idx += micro_batch_size;
        }
    }

    static constexpr int kMinHistoryTokensPerBlock = 512;

    static int computeMultiBlockCount(
        const XQAParams& xqaParams, int micro_batchsize, int multiprocessor_count, int max_multi_block_slots)
    {
        static constexpr int kTargetWaveFactor = 8;
        int multi_block_count = 1;
        int num_kv_heads = xqaParams.num_kv_heads;
        int history_length = xqaParams.timestep;

        multi_block_count = history_length / kMinHistoryTokensPerBlock;
        multi_block_count = std::max(multi_block_count, 1);
        // adjust to kTargetWaveFactor, as already initialized using kMinHistoryTokensPerBlock, only need to decrease.
        double wave_count = (double) micro_batchsize * num_kv_heads * multi_block_count / (double) multiprocessor_count;
        double adj_factor = wave_count / (double) kTargetWaveFactor;
        if (adj_factor > 1.0)
        {
            multi_block_count = floor(multi_block_count / adj_factor);
        }
        multi_block_count = std::max(multi_block_count, 1);

        // apply workspace limitation
        int beam_width = xqaParams.beam_width;
        int slots_needed_now = micro_batchsize * beam_width * num_kv_heads * multi_block_count;
        if (slots_needed_now > max_multi_block_slots)
        {
            multi_block_count = multi_block_count * max_multi_block_slots / slots_needed_now;
        }
        // add limitation on upper bound.
        multi_block_count = std::min(multiprocessor_count, multi_block_count);

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

    void run(const XQAParams& xqa_params, KVLinearBuffer& kv_linear_buffer, const cudaStream_t& stream,
        int max_multi_block_slots);

private:
    const XQAKernelList* xqaKernel;
    int sm;
    const XQADataType mDataType;
    int mMultiProcessorCount;
};

void DecoderXQARunner::xqaImpl::run(const XQAParams& xqa_params, KVLinearBuffer& kv_linear_buffer,
    const cudaStream_t& stream, int max_multi_block_slots)
{
    if (xqa_params.beam_width > 1)
    {
        xqaKernel->template run<__half, true>(
            xqa_params, kv_linear_buffer, stream, mMultiProcessorCount, max_multi_block_slots);
    }
    else
    {
        xqaKernel->template run<__half, false>(
            xqa_params, kv_linear_buffer, stream, mMultiProcessorCount, max_multi_block_slots);
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

size_t DecoderXQARunner::getWorkspaceSize()
{
    size_t workspace_size = 0;
    if (mMultiBlockMode)
    {
        int workspaces[4];
        const int max_num_request = kMaxBeamWidth * XQALaunchParam<true>::GetMaxBatchSizePerWave(mMultiProcessorCount);
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

void DecoderXQARunner::run(const XQAParams& xqa_params, KVLinearBuffer& kv_linear_buffer, const cudaStream_t& stream)
{
    int max_multi_block_slots = kMaxBeamWidth * XQALaunchParam<true>::GetMaxBatchSizePerWave(mMultiProcessorCount)
        * kMaxNbCtaPerKVHeadFactor * mNumKVHeads;
    return pimpl->run(xqa_params, kv_linear_buffer, stream, max_multi_block_slots);
}

} // namespace kernels

} // namespace tensorrt_llm
