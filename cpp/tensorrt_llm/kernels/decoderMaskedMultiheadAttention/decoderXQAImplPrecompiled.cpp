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
#include "tensorrt_llm/kernels/decoderMaskedMultiheadAttention/decoderXQARunner.h"
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
    size_t operator()(XQAKernelLoadHashKey const& s) const
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
    size_t operator()(XQAKernelRuntimeHashKey const& s) const
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
// NOTE: Linear KV cache and paged KV cache uses the same structure.

template <typename KVCacheBuffer>
struct KVCache
{
};

template <>
struct KVCache<KVBlockArray>
{
    // Start address of the paged kv block pool.
    void* poolPtr = nullptr;
    // Block indices in the memory pool.
    int32_t const* blockIndices = nullptr;
    int32_t const* sequence_lengths = nullptr;
    // NOTE: max_num_blocks_per_sequence for paged kv cache.
    uint32_t capacity = 0;

    KVCache(KVBlockArray& kv_cache_buffer)
    {
        poolPtr = kv_cache_buffer.mPrimaryPoolPtr;
        blockIndices = reinterpret_cast<KVCacheIndex::UnderlyingType const*>(kv_cache_buffer.data);
    }

    KVCache() = default;
};

template <>
struct KVCache<KVLinearBuffer>
{
    // Buffer address.
    void* data = nullptr;
    int32_t const* sequence_lengths = nullptr;
    // NOTE: max_sequence_length for linear kv cache.
    uint32_t capacity = 0;

    KVCache(KVLinearBuffer& kv_cache_buffer)
    {
        data = kv_cache_buffer.data;
    }

    KVCache() = default;
};

struct BeamSearchParams
{
    int32_t const* indices;    // cacheIndir with shape: [batchSize][beamWidth][capacity]
    int32_t capacity;
    int32_t const* ctxLenList; // shape: [batchSize][beamWidth]. Should be [batchSize] but we have to match trt-llm API.
};

// XQA kernels assume all integer values should use uint32_t.
template <typename KVCacheBuffer>
struct XQALaunchParam
{
    uint32_t num_k_heads;
    void* output;
    void const* qkv;
    KVCache<KVCacheBuffer> kvCacheParams;
    std::optional<BeamSearchParams> beamSearchParams;
    uint32_t batch_size;
    float const* kv_scale_quant_orig = nullptr;
    int* cu_seq_lens = nullptr;
    float* rotary_inv_freq_buf = nullptr;
    void* scratch = nullptr;
};

// Setup launch params.
template <typename KVCacheBuffer>
void buildXQALaunchParams(
    XQALaunchParam<KVCacheBuffer>& launchParams, XQAParams const& params, KVCacheBuffer kv_cache_buffer)
{
    TLLM_CHECK_WITH_INFO(
        params.data_type == DATA_TYPE_FP16 || params.data_type == DATA_TYPE_BF16, "Only fp16 or bf16 supported now.");
    memset(&launchParams, 0, sizeof(XQALaunchParam<KVCacheBuffer>));
    launchParams.num_k_heads = params.num_kv_heads;
    launchParams.output = static_cast<uint8_t*>(params.output);
    launchParams.qkv = static_cast<uint8_t const*>(params.qkv);
    launchParams.batch_size = params.batch_size;
    launchParams.kv_scale_quant_orig = params.kv_scale_quant_orig;

    // Workspace.
    size_t offset = 0;
    int8_t* workspace = reinterpret_cast<int8_t*>(params.workspaces);
    unsigned int batch_beam_size = params.batch_size * params.beam_width;
    const size_t cu_seqlens_size = sizeof(int) * (batch_beam_size + 1);
    const size_t rotary_inv_freq_size = sizeof(float) * batch_beam_size * params.rotary_embedding_dim / 2;
    launchParams.cu_seq_lens = reinterpret_cast<int*>(workspace);
    workspace = nextWorkspacePtrWithAlignment(workspace, cu_seqlens_size);
    launchParams.rotary_inv_freq_buf = reinterpret_cast<float*>(workspace);
    auto const multi_block_workspace_alignment = roundUp(
        sizeof(half) * params.head_size * (params.num_q_heads / params.num_kv_heads) * params.beam_width, 128);
    workspace = nextWorkspacePtrWithAlignment(workspace, rotary_inv_freq_size, multi_block_workspace_alignment);
    launchParams.scratch = reinterpret_cast<void*>(workspace);

    launchParams.kvCacheParams = KVCache<KVCacheBuffer>(kv_cache_buffer);
    launchParams.kvCacheParams.sequence_lengths = params.sequence_lengths;
    launchParams.kvCacheParams.capacity
        = params.paged_kv_cache ? params.max_blocks_per_sequence : params.max_attention_window_size;
    // TODO: beam searching has not been implemented yet.
    if (params.beam_width > 1)
    {
        launchParams.beamSearchParams
            = BeamSearchParams{params.cache_indir, params.max_attention_window_size, params.context_lengths};
    }
    else
    {
        launchParams.beamSearchParams = std::nullopt;
    }
}

namespace
{
template <typename T>
std::optional<T> getGlobalVar(
    tensorrt_llm::common::CUDADriverWrapper const& driver, CUmodule hmod, char const* const name, bool required = false)
{
    T* pVar = nullptr;
    size_t size = 0;
    auto const error = driver.cuModuleGetGlobal(reinterpret_cast<CUdeviceptr*>(&pVar), &size, hmod, name);
    T ret;
    switch (error)
    {
    case CUDA_SUCCESS:
        TLLM_CHECK(size == sizeof(T));
        check_cuda_error(cudaMemcpy(&ret, pVar, size, cudaMemcpyDeviceToHost));
        break;
    case CUDA_ERROR_NOT_FOUND:
        if (!required)
        {
            return std::nullopt;
        }
        [[fallthrough]];
    default: cuErrCheck(("Failed to retrieve global variable from cubin.", error), driver);
    }
    return std::optional<T>{std::move(ret)};
}
} // namespace

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
            auto const& kernelMeta = mKernelMeta[i];
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
            funcInfo.mSharedMemBytes = getGlobalVar<uint32_t>(mDriver, hmod, "smemSize", true).value();
            funcInfo.mKernelType = getGlobalVar<XQAKernelType>(mDriver, hmod, "kernelType", false)
                                       .value_or(XQAKernelType::kAMPERE_WARP_SPECIALIZED);

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
        // MultiQueryToken kernels can handle either 16/32 for M direction per CTA.
        unsigned int m_tilesize = xqaParams.multi_query_tokens ? 16 : num_q_heads_over_kv;

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
        buildXQALaunchParams(launchParams, xqaParams, kv_cache_buffer);

        // Build cu_seqlens, padding_offset, and rotary inv freq tensors
        BuildDecoderInfoParams<T> decoder_params;
        memset(&decoder_params, 0, sizeof(decoder_params));
        decoder_params.seqQOffsets = launchParams.cu_seq_lens;
        decoder_params.seqKVLengths = xqaParams.sequence_lengths;
        decoder_params.batchSize = int(batch_beam_size);
        decoder_params.maxQSeqLength = xqaParams.generation_input_length;
        // Rotary embedding inv_freq buffer.
        decoder_params.rotaryEmbeddingScale = xqaParams.rotary_embedding_scale;
        decoder_params.rotaryEmbeddingBase = xqaParams.rotary_embedding_base;
        decoder_params.rotaryEmbeddingDim = xqaParams.rotary_embedding_dim;
        decoder_params.rotaryScalingType = xqaParams.rotary_embedding_scale_type;
        decoder_params.rotaryEmbeddingInvFreq = launchParams.rotary_inv_freq_buf;
        decoder_params.rotaryEmbeddingMaxPositions = xqaParams.rotary_embedding_max_positions;

        invokeBuildDecoderInfo(decoder_params, stream);
        sync_check_cuda_error();

        // IDEA: Store rotary_processed Q buffer to output buffer.
        // NOTE: MHA kernels should read kv cache that has already been appended with new tokens' kv cache.
        void const* xqa_q_input_ptr = xqaParams.output;
        QKVPreprocessingParams<T, KVCacheBuffer> preprocessingParms{static_cast<T*>(const_cast<void*>(xqaParams.qkv)),
            nullptr, static_cast<T*>(const_cast<void*>(xqaParams.output)), kv_cache_buffer,
            static_cast<T const*>(xqaParams.qkv_bias), nullptr, xqaParams.sequence_lengths, nullptr,
            launchParams.rotary_inv_freq_buf, (float2 const*) nullptr, xqaParams.kv_scale_orig_quant,
            xqaParams.medusa_position_offsets, int(batch_beam_size), xqaParams.generation_input_length,
            xqaParams.timestep, xqaParams.cyclic_attention_window_size, xqaParams.sink_token_length,
            int(xqaParams.batch_size * beam_width * xqaParams.generation_input_length), xqaParams.num_q_heads,
            xqaParams.num_kv_heads, xqaParams.num_q_heads / xqaParams.num_kv_heads, xqaParams.head_size,
            xqaParams.rotary_embedding_dim, xqaParams.rotary_embedding_base, xqaParams.rotary_embedding_scale_type,
            xqaParams.rotary_embedding_scale, xqaParams.rotary_embedding_max_positions,
            xqaParams.position_embedding_type, xqaParams.position_shift_enabled, cache_type, true, false,
            multiprocessor_count};

        invokeQKVPreprocessing<T, KVCacheBuffer>(preprocessingParms, stream);
        sync_check_cuda_error();

        // Use mTileSize = 16 kernels when qSeqLen <= 16.
        unsigned int qSeqLen = static_cast<unsigned int>(xqaParams.generation_input_length);
        unsigned int mTileSize = qSeqLen <= 16 ? 16 : 32;
        // MultiQueryToken kernels can support any num_q_heads_over_kv that is power of 2.
        unsigned int kernel_num_q_heads_over_kv = xqaParams.multi_query_tokens ? 0 : num_q_heads_over_kv;
        // MultiQueryToken kernels can handle either 16/32 for M direction per CTA.
        unsigned int kernel_m_tilesize = xqaParams.multi_query_tokens ? mTileSize : num_q_heads_over_kv;
        XQAKernelRuntimeHashKey hash_key{xqaParams.kv_cache_data_type, head_size, beam_width,
            kernel_num_q_heads_over_kv, kernel_m_tilesize,
            xqaParams.paged_kv_cache ? static_cast<unsigned int>(xqaParams.tokens_per_block) : 0,
            xqaParams.paged_kv_cache, xqaParams.multi_query_tokens};
        auto const findIter = mFunctions.find(hash_key);

        TLLM_CHECK_WITH_INFO(findIter != mFunctions.end(), "XQAKernelFunc not found.");

        auto const& kernelMeta = mKernelMeta[findIter->second.mMetaInfoIndex];
        const CUfunction func = findIter->second.mDeviceFunction;
        unsigned int const shared_mem_bytes = findIter->second.mSharedMemBytes;
        auto const kernelType = findIter->second.mKernelType;

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
                check_cuda_error(cudaMemsetAsync(
                    launchParams.scratch, 0, sizeof(int) * xqaParams.batch_size * xqaParams.num_kv_heads, stream));
                sync_check_cuda_error();
            }
            cuErrCheck(mDriver.cuLaunchKernel(func, multi_block, xqaParams.num_kv_heads * nbTokenBlocksPerGrp,
                           xqaParams.batch_size, 128, 1, 2, shared_mem_bytes, stream, kernelParams, nullptr),
                mDriver);
        }
        else
        {
            bool const isGmmaKernel = (kernelType == XQAKernelType::kHOPPER_WARP_SPECIALIZED);
            TLLM_CHECK(isGmmaKernel
                == (mSM == kSM_90 && xqaParams.kv_cache_data_type == XQADataType::DATA_TYPE_E4M3
                    && xqaParams.beam_width == 1));
            constexpr uint32_t kMAX_NB_KERNEL_PARAMS = 10;
            uint32_t const maxNbKernelParams = (isGmmaKernel ? 10 : 9);
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
                tensorMap = makeTensorMapForKVCache(xqaParams, kv_cache_buffer);
                appendParam(&tensorMap);
            }
            appendParam(&launchParams.scratch);
            kernelParams[idxNextParam] = nullptr; // one extra nullptr at end as guard.
            int multi_block = 1;
            if (xqaParams.multi_block_mode)
            {
                multi_block = computeMultiBlockCount(xqaParams, xqaParams.batch_size, multiprocessor_count);
                check_cuda_error(cudaMemsetAsync(
                    launchParams.scratch, 0, sizeof(int) * xqaParams.batch_size * xqaParams.num_kv_heads, stream));
                sync_check_cuda_error();
            }
            cuErrCheck(mDriver.cuLaunchKernel(func, multi_block, xqaParams.num_kv_heads, xqaParams.batch_size, 128, 1,
                           isGmmaKernel ? 3 : 2, shared_mem_bytes, stream, kernelParams, nullptr),
                mDriver);
        }

        sync_check_cuda_error();
    }

    static int computeMultiBlockCount(XQAParams const& xqaParams, int batch_size, int multiprocessor_count)
    {
        if (envXqaNbCtaPerKVHead().has_value())
        {
            return envXqaNbCtaPerKVHead().value();
        }
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
        multi_block_count = std::min(xqaMaxNbCtaPerKVHeadFactor(), multi_block_count);

        TLLM_CHECK_WITH_INFO(multi_block_count >= 1, "MultiBlock count should be larger than 1");
        return multi_block_count;
    }

private:
    static uint32_t getElemBytes(CUtensorMapDataType_enum dataType)
    {
        switch (dataType)
        {
        case CU_TENSOR_MAP_DATA_TYPE_UINT8: return 1;
        case CU_TENSOR_MAP_DATA_TYPE_UINT16: return 2;
        case CU_TENSOR_MAP_DATA_TYPE_UINT32: return 4;
        case CU_TENSOR_MAP_DATA_TYPE_INT32: return 4;
        case CU_TENSOR_MAP_DATA_TYPE_UINT64: return 8;
        case CU_TENSOR_MAP_DATA_TYPE_INT64: return 8;
        case CU_TENSOR_MAP_DATA_TYPE_FLOAT16: return 2;
        case CU_TENSOR_MAP_DATA_TYPE_FLOAT32: return 4;
        case CU_TENSOR_MAP_DATA_TYPE_FLOAT64: return 8;
        case CU_TENSOR_MAP_DATA_TYPE_BFLOAT16: return 2;
        case CU_TENSOR_MAP_DATA_TYPE_FLOAT32_FTZ: return 4;
        case CU_TENSOR_MAP_DATA_TYPE_TFLOAT32: return 4;
        case CU_TENSOR_MAP_DATA_TYPE_TFLOAT32_FTZ: return 4;
        }
        throw std::runtime_error("unsupported data type");
    }

    CUtensorMap makeTensorMapForContiguousKVCache(void const* addr, CUtensorMapDataType_enum dataType,
        uint32_t headElems, uint32_t nbKHeads, uint32_t maxCacheLen, uint32_t beamWidth, uint32_t batchSize,
        uint32_t nbTokensPerTile = 64) const
    {
        CUtensorMap tensorMap{};
        uint64_t const globalDims[] = {headElems, maxCacheLen, nbKHeads, 2 * beamWidth * batchSize};
        uint32_t elemBytes = getElemBytes(dataType);
        uint32_t const headBytes = elemBytes * headElems;
        uint64_t const globalStrides[] = {headBytes, headBytes * maxCacheLen, headBytes * maxCacheLen * nbKHeads};
        uint32_t const partElems = std::min(headBytes, 128U) / elemBytes;
        uint32_t const boxDims[] = {partElems, nbTokensPerTile, 1, 1};
        uint32_t const elemStrides[] = {1, 1, 1, 1};

        auto const swizzle = [&]
        {
            switch (partElems)
            {
            case 128: return CU_TENSOR_MAP_SWIZZLE_128B;
            case 64: return CU_TENSOR_MAP_SWIZZLE_64B;
            default: throw std::runtime_error("unsupported cache head size");
            }
        }();

        cuErrCheck(mDriver.cuTensorMapEncodeTiled(&tensorMap, dataType, 4, const_cast<void*>(addr), globalDims,
                       globalStrides, boxDims, elemStrides, CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle,
                       CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE),
            mDriver);
        return tensorMap;
    }

    CUtensorMap makeTensorMapForPagedKVCache(void const* addr, CUtensorMapDataType_enum dataType, uint32_t headElems,
        uint32_t nbKHeads, uint32_t tokensPerPage, uint32_t nbTokensPerTile = 64) const
    {
        CUtensorMap tensorMap{};
        uint32_t elemBytes = getElemBytes(dataType);
        uint64_t const globalDims[] = {headElems, tokensPerPage, nbKHeads, 1U << 31};
        uint32_t const headBytes = elemBytes * headElems;
        uint64_t const globalStrides[] = {headBytes, headBytes * tokensPerPage, headBytes * tokensPerPage * nbKHeads};
        uint32_t const partElems = std::min(headBytes, 128U) / elemBytes;
        uint32_t const boxDims[] = {partElems, std::min(tokensPerPage, nbTokensPerTile), 1, 1};
        uint32_t const elemStrides[] = {1, 1, 1, 1};

        auto const swizzle = [&]
        {
            switch (partElems)
            {
            case 128: return CU_TENSOR_MAP_SWIZZLE_128B;
            case 64: return CU_TENSOR_MAP_SWIZZLE_64B;
            default: throw std::runtime_error("unsupported cache head size");
            }
        }();

        cuErrCheck(mDriver.cuTensorMapEncodeTiled(&tensorMap, dataType, 4, const_cast<void*>(addr), globalDims,
                       globalStrides, boxDims, elemStrides, CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle,
                       CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE),
            mDriver);
        return tensorMap;
    }

    template <typename KVCacheBuffer>
    CUtensorMap makeTensorMapForKVCache(XQAParams const& xqaParams, KVCacheBuffer const& kv_cache_buffer) const
    {
        if constexpr (std::is_same_v<KVCacheBuffer, KVBlockArray>)
        {
            return makeTensorMapForPagedKVCache(kv_cache_buffer.mPrimaryPoolPtr, CU_TENSOR_MAP_DATA_TYPE_UINT8,
                xqaParams.head_size, xqaParams.num_kv_heads, xqaParams.tokens_per_block);
        }
        else
        {
            static_assert(std::is_same_v<KVCacheBuffer, KVLinearBuffer>);
            return makeTensorMapForContiguousKVCache(kv_cache_buffer.data, CU_TENSOR_MAP_DATA_TYPE_UINT8,
                xqaParams.head_size, xqaParams.num_kv_heads, xqaParams.max_attention_window_size, xqaParams.beam_width,
                xqaParams.batch_size);
        }
    }

protected:
    tensorrt_llm::common::CUDADriverWrapper mDriver;

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

bool DecoderXQAImplPrecompiled::shouldUse(XQAParams const& xqaParams)
{
    XQAKernelList const* xqa_kernel = getXQAKernels(mRunner->mDataType, tensorrt_llm::common::getSMVersion());
    return xqa_kernel->supportConfig(xqaParams)
        && xqa_kernel->mayHavePerfGain(xqaParams, mRunner->mMultiProcessorCount);
}

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
