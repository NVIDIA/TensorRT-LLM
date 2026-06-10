/*
 * Copyright (c) 2020-2026, NVIDIA CORPORATION.  All rights reserved.
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
 * Common utils for the JIT XQA implementation.
 */
#pragma once
#include "decoderXQAConstants.h"
#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/config.h"
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

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

enum class XQAKernelType : int32_t
{
    kAMPERE_WARP_SPECIALIZED = 0,
    kHOPPER_WARP_SPECIALIZED = 1,
    kSM120_MLA = 2
};

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
    PositionEmbeddingType position_embedding_type;
    // Number of head elements RoPE is applied to. A value of 0 means rotary dim is not part of this key.
    int rotary_embedding_dim;

    bool operator==(XQAKernelRuntimeHashKey const& other) const
    {
        return kv_data_type == other.kv_data_type && head_size == other.head_size
            && num_q_heads_per_kv == other.num_q_heads_per_kv && beam_size == other.beam_size
            && multi_query_tokens == other.multi_query_tokens && m_tilesize == other.m_tilesize
            && tokens_per_page == other.tokens_per_page && paged_kv_cache == other.paged_kv_cache
            && is_fp8_output == other.is_fp8_output && position_embedding_type == other.position_embedding_type
            && rotary_embedding_dim == other.rotary_embedding_dim;
    }
};

inline uint32_t getKernelMTileSize(
    uint32_t headGrpSize, bool isSpecDec, uint32_t qSeqLen, bool supportQGMMA, bool supportMLA)
{
    if (!isSpecDec)
    {
        return headGrpSize;
    }
    if (supportQGMMA || supportMLA) // HMMA (mha.cu) goes to the heuristic below
    {
        return 64;
    }
    uint32_t const gemmM = qSeqLen * headGrpSize;
    return gemmM < 16 ? 16 : 32;
};

inline XQAKernelRuntimeHashKey getRuntimeHashKeyFromXQAParams(XQAParams const& xqaParams, int SM)
{
    unsigned int headSize = xqaParams.head_size;
    unsigned int numQHeads = xqaParams.num_q_heads;
    unsigned int numKVHeads = xqaParams.num_kv_heads;
    TLLM_CHECK_WITH_INFO(numQHeads % numKVHeads == 0, "numQHeads should be multiple of numKVHeads.");
    unsigned int numQHeadsOverKV = numQHeads / numKVHeads;
    unsigned int beamWidth = xqaParams.beam_width;

    unsigned int qSeqLen = static_cast<unsigned int>(xqaParams.generation_input_length);
    // MultiQueryToken kernels can support any numQHeadsOverKV that is power of 2.
    unsigned int kernelNumQHeadsOverKV = xqaParams.multi_query_tokens ? 0 : numQHeadsOverKV;
    bool supportQGMMA = jit::supportConfigQGMMA(xqaParams, SM, true);
    bool supportMLA = jit::supportConfigMLA(xqaParams, SM, true);
    unsigned int kernelMTilesize
        = getKernelMTileSize(numQHeadsOverKV, xqaParams.multi_query_tokens, qSeqLen, supportQGMMA, supportMLA);

    bool const includesRotaryDim = jit::appliesRoPEInXqaKernel(xqaParams, supportQGMMA);
    return {xqaParams.kv_cache_data_type, headSize, beamWidth, kernelNumQHeadsOverKV, kernelMTilesize,
        xqaParams.paged_kv_cache ? static_cast<unsigned int>(xqaParams.tokens_per_block) : 0, xqaParams.paged_kv_cache,
        xqaParams.multi_query_tokens, xqaParams.is_fp8_output, xqaParams.position_embedding_type,
        includesRotaryDim ? xqaParams.rotary_embedding_dim : 0};
}

struct XQAKernelRuntimeHasher
{
    size_t operator()(XQAKernelRuntimeHashKey const& s) const
    {
        size_t key = s.kv_data_type; // 6
        key <<= 6;                   // 12
        key ^= s.head_size / 16;
        key <<= 8;                   // 20
        key ^= s.num_q_heads_per_kv;
        key <<= 8;                   // 28
        key ^= s.beam_size;
        key <<= 6;                   // 34
        key ^= s.m_tilesize;
        key <<= 10;                  // 44
        key ^= s.tokens_per_page;
        key <<= 1;                   // 45
        key ^= s.paged_kv_cache;
        key <<= 1;                   // 46
        key ^= s.multi_query_tokens;
        key <<= 1;                   // 47
        key ^= s.is_fp8_output;
        key <<= 6;                   // 53
        key ^= static_cast<int8_t>(s.position_embedding_type);
        key <<= 9;                   // 62; rotary dims are <= 256, 0 means not used.
        key ^= static_cast<size_t>(s.rotary_embedding_dim);
        return key;
    }
};

// XQA kernel can be uniquely identified by (LoadHashKey, RuntimeHashKey).
struct XQAKernelFullHashKey
{
    XQAKernelLoadHashKey load_key;
    XQAKernelRuntimeHashKey runtime_key;

    XQAKernelFullHashKey() = default;

    XQAKernelFullHashKey(XQAKernelLoadHashKey const& loadKey, XQAKernelRuntimeHashKey const& runtimeKey)
        : load_key(loadKey)
        , runtime_key(runtimeKey)
    {
    }

    XQAKernelFullHashKey(void const* buffer, size_t bufferSize)
    {
        TLLM_CHECK(sizeof(*this) <= bufferSize);
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

    void serialize(void* buffer, size_t bufferSize) const
    {
        TLLM_CHECK(sizeof(*this) <= bufferSize);
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

    KVCache(KVBlockArray& kvCacheBuffer)
    {
        poolPtr = kvCacheBuffer.mPrimaryPoolPtr;
        blockIndices = reinterpret_cast<KVCacheIndex::UnderlyingType const*>(kvCacheBuffer.data);
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

    KVCache(KVLinearBuffer& kvCacheBuffer)
    {
        data = kvCacheBuffer.data;
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
    XQAParams const& params, KVCacheBuffer kvCacheBuffer)
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
    auto const multiBlockWorkspaceAlignment = tensorrt_llm::common::roundUp(
        sizeof(half) * params.head_size * (params.num_q_heads / params.num_kv_heads) * params.beam_width, 128);

    // Workspace.
    int8_t* workspace = reinterpret_cast<int8_t*>(params.workspaces);
    unsigned int batchBeamSize = params.batch_size * params.beam_width;
    const size_t cuSeqlensSize = sizeof(int) * (batchBeamSize + 1);
    const size_t cuKvSeqlensSize = sizeof(int) * (batchBeamSize + 1);
    const size_t rotaryInvFreqSize = sizeof(float) * batchBeamSize * params.rotary_embedding_dim / 2;
    const size_t tokensInfoSize = sizeof(int2) * params.total_num_input_tokens;
    const size_t kvBlockOffsetsSize
        = sizeof(int) * batchBeamSize * 2 * params.max_blocks_per_sequence * params.num_kv_heads;
    const size_t seqLengthsSize = sizeof(int) * batchBeamSize * params.num_kv_heads;
    launchParams.cu_seq_lens = reinterpret_cast<int*>(workspace);
    workspace = tensorrt_llm::common::nextWorkspacePtrWithAlignment(workspace, cuSeqlensSize);
    launchParams.cu_kv_seq_lens = reinterpret_cast<int*>(workspace);
    workspace = tensorrt_llm::common::nextWorkspacePtrWithAlignment(workspace, cuKvSeqlensSize);
    launchParams.rotary_inv_freq_buf = reinterpret_cast<float*>(workspace);
    workspace = tensorrt_llm::common::nextWorkspacePtrWithAlignment(workspace, rotaryInvFreqSize);
    launchParams.tokens_info = reinterpret_cast<int2*>(workspace);
    workspace = tensorrt_llm::common::nextWorkspacePtrWithAlignment(workspace, tokensInfoSize);
    // Only used for trtllm-gen kernels.
    size_t const bmm1ScaleSize = sizeof(float) * 2;
    size_t const bmm2ScaleSize = sizeof(float);
    launchParams.bmm1_scale_ptr = reinterpret_cast<float*>(workspace);
    workspace = tensorrt_llm::common::nextWorkspacePtrWithAlignment(workspace, bmm1ScaleSize);
    launchParams.bmm2_scale_ptr = reinterpret_cast<float*>(workspace);
    workspace = tensorrt_llm::common::nextWorkspacePtrWithAlignment(workspace, bmm2ScaleSize);
    // Used for block sparse attention
    if (params.use_sparse_attention_gen_paged)
    {
        launchParams.sparse_kv_block_offsets = reinterpret_cast<void*>(workspace);
        workspace = tensorrt_llm::common::nextWorkspacePtrWithAlignment(workspace, kvBlockOffsetsSize);
        launchParams.sparse_seq_lengths = reinterpret_cast<int*>(workspace);
        workspace = tensorrt_llm::common::nextWorkspacePtrWithAlignment(workspace, seqLengthsSize);
    }
    inputScratch = workspace;
    if (hasOutputScratch)
    {
        workspace = tensorrt_llm::common::nextWorkspacePtrWithAlignment(
            workspace, 2 * params.head_size * params.num_q_heads * params.total_num_input_tokens);
        // Only used for output conversion.
        launchParams.output = workspace;
        workspace = tensorrt_llm::common::nextWorkspacePtrWithAlignment(workspace,
            2 * params.head_size * params.num_q_heads * params.total_num_input_tokens, multiBlockWorkspaceAlignment);
    }
    else
    {
        workspace = tensorrt_llm::common::nextWorkspacePtrWithAlignment(workspace,
            2 * params.head_size * params.num_q_heads * params.total_num_input_tokens, multiBlockWorkspaceAlignment);
    }
    workspace = tensorrt_llm::common::nextWorkspacePtrWithAlignment(workspace,
        2 * params.head_size * params.num_q_heads * params.total_num_input_tokens, multiBlockWorkspaceAlignment);
    launchParams.scratch = reinterpret_cast<void*>(workspace);

    launchParams.kvCacheParams = KVCache<KVCacheBuffer>(kvCacheBuffer);
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

inline int computeMultiBlockCount(XQAParams const& xqaParams, int batchSize, int multiprocessorCount)
{
    auto const userSpecified = tensorrt_llm::common::getEnvXqaBlocksPerSequence();
    if (userSpecified.has_value())
    {
        return userSpecified.value();
    }
    int multiBlockCount = 1;
    int numKVHeads = xqaParams.num_kv_heads;
    int historyLength = xqaParams.max_past_kv_length;

    int32_t const maxNbSubSeq = getXqaMaxNumSubSeq(xqaParams.isMLA());

    multiBlockCount = historyLength / kMinHistoryTokensPerBlock;
    // avoid using too many blocks for one sequence, otherwise the final reduction may dominate.
    multiBlockCount = std::min(multiBlockCount, static_cast<int>(std::round(std::sqrt(multiBlockCount * 8.F))));
    multiBlockCount = std::max(multiBlockCount, 1);
    // adjust to kTargetWaveFactor, as already initialized using kMinHistoryTokensPerBlock, only need to decrease.
    double waveCount = (double) batchSize * numKVHeads * multiBlockCount / (double) multiprocessorCount;
    double adjFactor = waveCount / (double) kTargetWaveFactor;
    if (adjFactor > 1.0)
    {
        multiBlockCount = floor(multiBlockCount / adjFactor);
    }
    multiBlockCount = std::max(multiBlockCount, 1);

    // Add limitation due to reserved workspace size.
    // When batch_size is large, multi-block is useless anyway. So large workspace is not useful and we can set a hard
    // limit for workspace size (computed from maxNbSubSeq).
    multiBlockCount = std::max(std::min(multiBlockCount, maxNbSubSeq / batchSize), 1);

    TLLM_CHECK_WITH_INFO(multiBlockCount >= 1, "MultiBlock count should be larger than 1");
    TLLM_CHECK_WITH_INFO(multiBlockCount == 1 || batchSize * multiBlockCount <= maxNbSubSeq, "Insufficient workspace");
    return multiBlockCount;
}

inline int computeMultiBlockCountForMLA(XQAParams const& xqaParams, int multiprocessorCount)
{
    return 1; // disable multi-block for MLA kernel for now.
}

inline int computeMultiBlockCountSpecDecGMMA(
    XQAParams const& xqaParams, int batchSize, int multiprocessorCount, int specDecBlocks)
{
    auto const userSpecified = tensorrt_llm::common::getEnvXqaBlocksPerSequence();
    if (userSpecified.has_value())
    {
        return userSpecified.value();
    }
    int multiBlockCount = 1;

    int numKVHeads = xqaParams.num_kv_heads;
    int historyLength = xqaParams.max_past_kv_length;

    // skip tuning for large BS or short ISL case.
    if (batchSize > 32 || historyLength < 2048)
    {
        return multiBlockCount;
    }

    // gridDim = dim3{specDecBlocks, multiBlock, nbKVHeads * xqaParams.batch_size}
    int singleBlockCount = specDecBlocks * numKVHeads * batchSize;
    double waveCount = (double) singleBlockCount / (double) multiprocessorCount;

    // Multi block tuning for low CTA: populating CTAs to at most 1 wave of SMs
    if (waveCount < 1)
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
        multiBlockCount = floor(multiprocessorCount / singleBlockCount);
        // make multiBlockCount a power of 2 for tuning convenience.
        multiBlockCount = highestPowerof2(multiBlockCount);
        // make multiBlockCount at most 64 and at least 1.
        multiBlockCount = std::min(multiBlockCount, 64);
        multiBlockCount = std::max(multiBlockCount, 1);

        // tune only when original CTA is too small, multiBlockCount is too big, and history length < 2^16
        // For Hopper, most cases there are 114, 132, 144 SMs. For H20 about 78.
        // singleBlockCount = [1..8]
        // multiBlockCount = [16,32,64,128]
        // historyLength = [1024..65536]
        if (singleBlockCount <= 8 && multiBlockCount >= 16 && historyLength < 65536)
        {
            if (historyLength < 2048)
            {
                // for history length < 2048 and low CTA, scaling is not effective, so we set a hard limit to
                // multiBlockCount = 4
                multiBlockCount = std::min(multiBlockCount, 4);
            }
            else if (historyLength < 65536)
            {
                // at singleBlock == 8, multiBlockCount can only be 16. (SM / 8 ~= 16)
                // tune only 2048 <= kvlen < 8192
                if (singleBlockCount == 8 && historyLength <= 8192)
                {
                    multiBlockCount >>= 1;
                }
                else
                {
                    auto getLog2 = [](int x) { return x ? 31 - __builtin_clz(x) : -1; };
                    auto historyLengthLog2 = getLog2(historyLength);
                    // Adjust multiBlockCount based on history length using formula:
                    // shift_amount = 3 - (log2(historyLength) - 10) / 2
                    // This gives us:
                    // - historyLength in [2^11, 2^12): shift by 3
                    // - historyLength in [2^13, 2^14): shift by 2
                    // - historyLength in [2^15, 2^16): shift by 1
                    multiBlockCount >>= 3 - (historyLengthLog2 - 10) / 2;
                }
            }
        }
        TLLM_CHECK_WITH_INFO((multiBlockCount * singleBlockCount) <= multiprocessorCount,
            "The adjusted MultiBlock exceed number of SMs, adding additional wave may result to perf drop.");
    }
    return multiBlockCount;
}

} // namespace kernels

TRTLLM_NAMESPACE_END
