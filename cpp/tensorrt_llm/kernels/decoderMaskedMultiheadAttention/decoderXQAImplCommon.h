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
#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaDriverWrapper.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/envUtils.h"
#include "tensorrt_llm/common/workspace.h"
#include "tensorrt_llm/kernels/decoderMaskedMultiheadAttention/decoderXQAImplJIT/kernelUtils.h"
#include "tensorrt_llm/kernels/kvCacheUtils.h"
#include "tensorrt_llm/kernels/multiHeadAttentionCommon.h"
#include "xqaParams.h"
#include <cstddef>
#include <cstdint>
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
    bool is_fp8_output;
    std::optional<PositionEmbeddingType> position_embedding_type;

    bool operator==(XQAKernelRuntimeHashKey const& other) const
    {
        return kv_data_type == other.kv_data_type && head_size == other.head_size
            && num_q_heads_per_kv == other.num_q_heads_per_kv && beam_size == other.beam_size
            && multi_query_tokens == other.multi_query_tokens && m_tilesize == other.m_tilesize
            && tokens_per_page == other.tokens_per_page && paged_kv_cache == other.paged_kv_cache
            && is_fp8_output == other.is_fp8_output && position_embedding_type == other.position_embedding_type;
    }
};

uint32_t getKernelMTileSize(
    uint32_t headGrpSize, bool isSpecDec, uint32_t qSeqLen, bool isXqaJit, bool supportQGMMA, bool supportMLA);

XQAKernelRuntimeHashKey getRuntimeHashKeyFromXQAParams(XQAParams const& xqaParams, bool isXqaJit, int SM);

struct XQAKernelRuntimeHasher
{
    size_t operator()(XQAKernelRuntimeHashKey const& s) const
    {
        size_t key = s.kv_data_type;
        key <<= 16; // 16
        key ^= s.head_size;
        key <<= 8;  // 24
        key ^= s.num_q_heads_per_kv;
        key <<= 8;  // 32
        key ^= s.beam_size;
        key <<= 6;  // 38
        key ^= s.m_tilesize;
        key <<= 10; // 48
        key ^= s.tokens_per_page;
        key <<= 1;  // 49
        key ^= s.paged_kv_cache;
        key <<= 1;  // 50
        key ^= s.multi_query_tokens;
        key <<= 1;  // 51
        key ^= s.is_fp8_output;
        key <<= 8;
        key ^= static_cast<int8_t>(s.position_embedding_type.value_or(static_cast<PositionEmbeddingType>(-1)));
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
    uint32_t slidingWindowSize;
    float qScale;
    void* output;
    float const* rcpOutScale;
    void const* qkv;
    float2 const* ropeCosSin;
    KVCache<KVCacheBuffer> kvCacheParams;
    std::optional<BeamSearchParams> beamSearchParams;
    uint32_t batch_size;
    float const* kv_scale_quant_orig = nullptr;
    int* cu_seq_lens = nullptr;
    int* cu_kv_seq_lens = nullptr;
    float* rotary_inv_freq_buf = nullptr;
    int2* tokens_info = nullptr;
    float* bmm1_scale_ptr = nullptr;
    float* bmm2_scale_ptr = nullptr;
    int32_t* semaphores = nullptr;
    void* scratch = nullptr;
    void* sparse_kv_block_offsets = nullptr;
    int32_t* sparse_seq_lengths = nullptr;
};

// Setup launch params and ioScratch. ioScratch is for RoPE and output type conversion.
template <typename KVCacheBuffer>
void buildXQALaunchParams(XQALaunchParam<KVCacheBuffer>& launchParams, void*& inputScratch, bool hasOutputScratch,
    XQAParams const& params, KVCacheBuffer kv_cache_buffer)
{
    TLLM_CHECK_WITH_INFO(
        params.data_type == DATA_TYPE_FP16 || params.data_type == DATA_TYPE_BF16 || params.data_type == DATA_TYPE_E4M3,
        "Only fp16 or bf16 supported now.");
    launchParams = {};
    launchParams.num_k_heads = params.num_kv_heads;
    launchParams.slidingWindowSize = params.cyclic_attention_window_size;
    launchParams.qScale = params.q_scaling;
    launchParams.output = static_cast<uint8_t*>(params.output);
    launchParams.rcpOutScale = params.fp8_out_scale;
    launchParams.qkv = static_cast<uint8_t const*>(params.qkv);
    launchParams.ropeCosSin = params.rotary_cos_sin;
    launchParams.batch_size = params.batch_size;
    launchParams.kv_scale_quant_orig = params.kv_scale_quant_orig;
    launchParams.semaphores = params.semaphores;

    // The workspace alignment.
    auto const multi_block_workspace_alignment = tensorrt_llm::common::roundUp(
        sizeof(half) * params.head_size * (params.num_q_heads / params.num_kv_heads) * params.beam_width, 128);

    // Workspace.
    int8_t* workspace = reinterpret_cast<int8_t*>(params.workspaces);
    unsigned int batch_beam_size = params.batch_size * params.beam_width;
    const size_t cu_seqlens_size = sizeof(int) * (batch_beam_size + 1);
    const size_t cu_kv_seqlens_size = sizeof(int) * (batch_beam_size + 1);
    const size_t rotary_inv_freq_size = sizeof(float) * batch_beam_size * params.rotary_embedding_dim / 2;
    const size_t tokens_info_size = sizeof(int2) * params.total_num_input_tokens;
    const size_t kv_block_offsets_size
        = sizeof(int) * batch_beam_size * 2 * params.max_blocks_per_sequence * params.num_kv_heads;
    const size_t seq_lengths_size = sizeof(int) * batch_beam_size * params.num_kv_heads;
    launchParams.cu_seq_lens = reinterpret_cast<int*>(workspace);
    workspace = tensorrt_llm::common::nextWorkspacePtrWithAlignment(workspace, cu_seqlens_size);
    launchParams.cu_kv_seq_lens = reinterpret_cast<int*>(workspace);
    workspace = tensorrt_llm::common::nextWorkspacePtrWithAlignment(workspace, cu_kv_seqlens_size);
    launchParams.rotary_inv_freq_buf = reinterpret_cast<float*>(workspace);
    workspace = tensorrt_llm::common::nextWorkspacePtrWithAlignment(workspace, rotary_inv_freq_size);
    launchParams.tokens_info = reinterpret_cast<int2*>(workspace);
    workspace = tensorrt_llm::common::nextWorkspacePtrWithAlignment(workspace, tokens_info_size);
    // Only used for trtllm-gen kernels.
    size_t const bmm1_scale_size = sizeof(float) * 2;
    size_t const bmm2_scale_size = sizeof(float);
    launchParams.bmm1_scale_ptr = reinterpret_cast<float*>(workspace);
    workspace = tensorrt_llm::common::nextWorkspacePtrWithAlignment(workspace, bmm1_scale_size);
    launchParams.bmm2_scale_ptr = reinterpret_cast<float*>(workspace);
    workspace = tensorrt_llm::common::nextWorkspacePtrWithAlignment(workspace, bmm2_scale_size);
    // Used for block sparse attention
    if (params.use_sparse_attention)
    {
        launchParams.sparse_kv_block_offsets = reinterpret_cast<void*>(workspace);
        workspace = tensorrt_llm::common::nextWorkspacePtrWithAlignment(workspace, kv_block_offsets_size);
        launchParams.sparse_seq_lengths = reinterpret_cast<int*>(workspace);
        workspace = tensorrt_llm::common::nextWorkspacePtrWithAlignment(workspace, seq_lengths_size);
    }
    inputScratch = workspace;
    if (hasOutputScratch)
    {
        workspace = tensorrt_llm::common::nextWorkspacePtrWithAlignment(
            workspace, 2 * params.head_size * params.num_q_heads * params.total_num_input_tokens);
        // Only used for output conversion.
        launchParams.output = workspace;
        workspace = tensorrt_llm::common::nextWorkspacePtrWithAlignment(workspace,
            2 * params.head_size * params.num_q_heads * params.total_num_input_tokens, multi_block_workspace_alignment);
    }
    else
    {
        workspace = tensorrt_llm::common::nextWorkspacePtrWithAlignment(workspace,
            2 * params.head_size * params.num_q_heads * params.total_num_input_tokens, multi_block_workspace_alignment);
    }
    workspace = tensorrt_llm::common::nextWorkspacePtrWithAlignment(workspace,
        2 * params.head_size * params.num_q_heads * params.total_num_input_tokens, multi_block_workspace_alignment);
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
std::optional<T> getGlobalVar(std::shared_ptr<tensorrt_llm::common::CUDADriverWrapper> const& driver, CUlibrary lib,
    char const* const name, bool required = false)
{
    T* pVar = nullptr;
    size_t size = 0;
    auto const error = driver->cuLibraryGetGlobal(reinterpret_cast<CUdeviceptr*>(&pVar), &size, lib, name);
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
    default: TLLM_THROW("Failed to retrieve global variable from cubin: error code %i.", static_cast<int32_t>(error));
    }
    return std::optional<T>{std::move(ret)};
}

inline int computeMultiBlockCount(XQAParams const& xqaParams, int batch_size, int multiprocessor_count)
{
    auto const userSpecified = tensorrt_llm::common::getEnvXqaBlocksPerSequence();
    if (userSpecified.has_value())
    {
        return userSpecified.value();
    }
    int multi_block_count = 1;
    int num_kv_heads = xqaParams.num_kv_heads;
    int history_length = xqaParams.max_past_kv_length;

    int32_t const maxNbSubSeq = getXqaMaxNumSubSeq(xqaParams.isMLA());

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

inline int computeMultiBlockCountForMLA(XQAParams const& xqaParams, int multiprocessor_count)
{
    return 1; // disable multi-block for MLA kernel for now.
}

inline int computeMultiBlockCountSpecDecGMMA(
    XQAParams const& xqaParams, int batch_size, int multiprocessor_count, int specDecBlocks)
{
    auto const userSpecified = tensorrt_llm::common::getEnvXqaBlocksPerSequence();
    if (userSpecified.has_value())
    {
        return userSpecified.value();
    }
    int multi_block_count = 1;

    int num_kv_heads = xqaParams.num_kv_heads;
    int history_length = xqaParams.max_past_kv_length;

    // skip tuning for large BS or short ISL case.
    if (batch_size > 32 || history_length < 2048)
    {
        return multi_block_count;
    }

    // gridDim = dim3{specDecBlocks, multi_block, nbKVHeads * xqaParams.batch_size}
    int single_block_count = specDecBlocks * num_kv_heads * batch_size;
    double wave_count = (double) single_block_count / (double) multiprocessor_count;

    // Multi block tuning for low CTA: populating CTAs to at most 1 wave of SMs
    if (wave_count < 1)
    {
        auto highestPowerof2 = [](int x)
        {
            x |= x >> 1;
            x |= x >> 2;
            x |= x >> 4;
            x |= x >> 8;
            x |= x >> 16;
            return x ^ (x >> 1);
        };

        // calculate the maximum blocks to be populated at most 1 wave
        multi_block_count = floor(multiprocessor_count / single_block_count);
        // make multi_block_count a power of 2 for tuning convenience.
        multi_block_count = highestPowerof2(multi_block_count);
        // make multi_block_count at most 64 and at least 1.
        multi_block_count = std::min(multi_block_count, 64);
        multi_block_count = std::max(multi_block_count, 1);

        // tune only when original CTA is too small, multi_block_count is too big, and history length < 2^16
        // For Hopper, most cases there are 114, 132, 144 SMs. For H20 about 78.
        // single_block_count = [1..8]
        // multi_block_count = [16,32,64,128]
        // history_length = [1024..65536]
        if (single_block_count <= 8 && multi_block_count >= 16 && history_length < 65536)
        {
            if (history_length < 2048)
            {
                // for history length < 2048 and low CTA, scaling is not effective, so we set a hard limit to
                // multi_block_count = 4
                multi_block_count = std::min(multi_block_count, 4);
            }
            else if (history_length < 65536)
            {
                // at single_block == 8, multi_block_count can only be 16. (SM / 8 ~= 16)
                // tune only 2048 <= kvlen < 8192
                if (single_block_count == 8 && history_length <= 8192)
                {
                    multi_block_count >>= 1;
                }
                else
                {
                    auto getLog2 = [](int x) { return x ? 31 - __builtin_clz(x) : -1; };
                    auto history_length_log2 = getLog2(history_length);
                    // Adjust multi_block_count based on history length using formula:
                    // shift_amount = 3 - (log2(history_length) - 10) / 2
                    // This gives us:
                    // - history_length in [2^11, 2^12): shift by 3
                    // - history_length in [2^13, 2^14): shift by 2
                    // - history_length in [2^15, 2^16): shift by 1
                    multi_block_count >>= 3 - (history_length_log2 - 10) / 2;
                }
            }
        }
        TLLM_CHECK_WITH_INFO((multi_block_count * single_block_count) <= multiprocessor_count,
            "The adjusted MultiBlock exceed number of SMs, adding additional wave may result to perf drop.");
    }
    return multi_block_count;
}

} // namespace kernels
} // namespace tensorrt_llm
