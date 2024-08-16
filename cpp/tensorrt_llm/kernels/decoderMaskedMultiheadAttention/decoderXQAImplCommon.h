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
 *
 * Common utils to be shared between Precompiled and JIT implementation.
 */
#pragma once
#include "decoderXQAConstants.h"
#include "tensorrt_llm/common/cudaDriverWrapper.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/envUtils.h"
#include "tensorrt_llm/common/workspace.h"
#include "tensorrt_llm/kernels/kvCacheUtils.h"
#include "tensorrt_llm/kernels/multiHeadAttentionCommon.h"
#include "xqaParams.h"
#include <cstddef>
#include <utility>

namespace tensorrt_llm
{
namespace kernels
{

struct XQAKernelLoadHashKey
{
    Data_type data_type;
    unsigned int sm;

    bool operator==(XQAKernelLoadHashKey const& other) const
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

    bool operator==(XQAKernelRuntimeHashKey const& other) const
    {
        return kv_data_type == other.kv_data_type && head_size == other.head_size
            && num_q_heads_per_kv == other.num_q_heads_per_kv && beam_size == other.beam_size
            && multi_query_tokens == other.multi_query_tokens && m_tilesize == other.m_tilesize
            && tokens_per_page == other.tokens_per_page && paged_kv_cache == other.paged_kv_cache;
    }
};

XQAKernelRuntimeHashKey getRuntimeHashKeyFromXQAParams(XQAParams const& xqaParams);

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

// XQA kernel can be uniquely identified by (LoadHashKey, RuntimeHashKey).
struct XQAKernelFullHashKey
{
    XQAKernelLoadHashKey load_key;
    XQAKernelRuntimeHashKey runtime_key;

    XQAKernelFullHashKey() = default;

    XQAKernelFullHashKey(XQAKernelLoadHashKey const& load_key, XQAKernelRuntimeHashKey const& runtime_key)
        : load_key(load_key)
        , runtime_key(runtime_key)
    {
    }

    XQAKernelFullHashKey(void const* buffer, size_t buffer_size)
    {
        TLLM_CHECK(sizeof(*this) <= buffer_size);
        memcpy(this, buffer, sizeof(*this));
    }

    bool operator==(XQAKernelFullHashKey const& other) const
    {
        return load_key == other.load_key && runtime_key == other.runtime_key;
    }

    size_t getSerializationSize() const
    {
        return sizeof(*this);
    }

    void serialize(void* buffer, size_t buffer_size) const
    {
        TLLM_CHECK(sizeof(*this) <= buffer_size);
        memcpy(buffer, this, sizeof(*this));
    }
};

struct XQAKernelFullHasher
{
    size_t operator()(XQAKernelFullHashKey const& s) const
    {
        return XQAKernelLoadHasher()(s.load_key) ^ XQAKernelRuntimeHasher()(s.runtime_key);
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
    int32_t* semaphores = nullptr;
    void* scratch = nullptr;
};

// Setup launch params and ioScratch. ioScratch is for RoPE and output type conversion.
template <typename KVCacheBuffer>
void buildXQALaunchParams(XQALaunchParam<KVCacheBuffer>& launchParams, void*& ioScratch, XQAParams const& params,
    KVCacheBuffer kv_cache_buffer)
{
    TLLM_CHECK_WITH_INFO(
        params.data_type == DATA_TYPE_FP16 || params.data_type == DATA_TYPE_BF16, "Only fp16 or bf16 supported now.");
    memset(&launchParams, 0, sizeof(XQALaunchParam<KVCacheBuffer>));
    launchParams.num_k_heads = params.num_kv_heads;
    launchParams.output = static_cast<uint8_t*>(params.output);
    launchParams.qkv = static_cast<uint8_t const*>(params.qkv);
    launchParams.batch_size = params.batch_size;
    launchParams.kv_scale_quant_orig = params.kv_scale_quant_orig;
    launchParams.semaphores = params.semaphores;

    // Workspace.
    size_t offset = 0;
    int8_t* workspace = reinterpret_cast<int8_t*>(params.workspaces);
    ioScratch = workspace;
    workspace = tensorrt_llm::common::nextWorkspacePtrWithAlignment(
        workspace, 2 * params.head_size * params.num_q_heads * params.total_num_input_tokens);
    unsigned int batch_beam_size = params.batch_size * params.beam_width;
    const size_t cu_seqlens_size = sizeof(int) * (batch_beam_size + 1);
    const size_t rotary_inv_freq_size = sizeof(float) * batch_beam_size * params.rotary_embedding_dim / 2;
    launchParams.cu_seq_lens = reinterpret_cast<int*>(workspace);
    workspace = tensorrt_llm::common::nextWorkspacePtrWithAlignment(workspace, cu_seqlens_size);
    launchParams.rotary_inv_freq_buf = reinterpret_cast<float*>(workspace);
    auto const multi_block_workspace_alignment = tensorrt_llm::common::roundUp(
        sizeof(half) * params.head_size * (params.num_q_heads / params.num_kv_heads) * params.beam_width, 128);
    workspace = tensorrt_llm::common::nextWorkspacePtrWithAlignment(
        workspace, rotary_inv_freq_size, multi_block_workspace_alignment);
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

template <typename T>
std::optional<T> getGlobalVar(std::shared_ptr<tensorrt_llm::common::CUDADriverWrapper> const& driver, CUmodule hmod,
    char const* const name, bool required = false)
{
    T* pVar = nullptr;
    size_t size = 0;
    auto const error = driver->cuModuleGetGlobal(reinterpret_cast<CUdeviceptr*>(&pVar), &size, hmod, name);
    T ret;
    switch (error)
    {
    case CUDA_SUCCESS:
        TLLM_CHECK(size == sizeof(T));
        tensorrt_llm::common::check_cuda_error(cudaMemcpy(&ret, pVar, size, cudaMemcpyDeviceToHost));
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

inline int computeMultiBlockCount(XQAParams const& xqaParams, int batch_size, int multiprocessor_count)
{
    int multi_block_count = 1;
    int num_kv_heads = xqaParams.num_kv_heads;
    int history_length = xqaParams.timestep;

    int32_t const maxNbSubSeq = kXQA_MAX_NUM_SUB_SEQ;

    multi_block_count = history_length / kMinHistoryTokensPerBlock;
    // avoid using too many blocks for one sequence, otherwise the final reduction may dominate.
    multi_block_count = std::min(multi_block_count, static_cast<int>(std::round(std::sqrt(multi_block_count * 8.F))));
    multi_block_count = std::max(multi_block_count, 1);
    // adjust to kTargetWaveFactor, as already initialized using kMinHistoryTokensPerBlock, only need to decrease.
    double wave_count = (double) batch_size * num_kv_heads * multi_block_count / (double) multiprocessor_count;
    double adj_factor = wave_count / (double) kTargetWaveFactor;
    if (adj_factor > 1.0)
    {
        multi_block_count = floor(multi_block_count / adj_factor);
    }
    multi_block_count = std::max(multi_block_count, 1);

    // Add limitation due to reserved workspace size.
    // When batch_size is large, multi-block is useless anyway. So large workspace is not useful and we can set a hard
    // limit for workspace size (computed from maxNbSubSeq).
    multi_block_count = std::max(std::min(multi_block_count, maxNbSubSeq / batch_size), 1);

    TLLM_CHECK_WITH_INFO(multi_block_count >= 1, "MultiBlock count should be larger than 1");
    TLLM_CHECK_WITH_INFO(
        multi_block_count == 1 || batch_size * multi_block_count <= maxNbSubSeq, "Insufficient workspace");
    return multi_block_count;
}

} // namespace kernels
} // namespace tensorrt_llm
