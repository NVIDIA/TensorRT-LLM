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
 */

#include "decoderXQARunner.h"

#include <mutex>
#include <string.h>

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/envUtils.h"
#include "tensorrt_llm/kernels/decoderMaskedMultiheadAttention/decoderXQAConstants.h"
#include "tensorrt_llm/kernels/decoderMaskedMultiheadAttention/decoderXQAImplJIT/compileEngine.h"
#include "tensorrt_llm/kernels/decoderMaskedMultiheadAttention/decoderXQAImplJIT/kernelUtils.h"
#include "tensorrt_llm/kernels/decoderMaskedMultiheadAttention/tensorMapUtils.h"
#include "tensorrt_llm/kernels/kvCacheUtils.h"
#include "tensorrt_llm/kernels/unfusedAttentionKernels.h"
#include "tensorrt_llm/kernels/xqaDispatcher.h"

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

DecoderXQARunner::DecoderXQARunner(
    XQADataType const dataType, int numHeads, int numKVHeads, int headSize, bool multiBlockMode)
    : mDataType(dataType)
    , mNumHeads(numHeads)
    , mNumKVHeads(numKVHeads)
    , mHeadSize(headSize)
    , mMultiBlockMode(multiBlockMode)
    , mMultiProcessorCount(tensorrt_llm::common::getMultiProcessorCount())
    , mDriver(tensorrt_llm::common::CUDADriverWrapper::getInstance())
    , mResource(DecoderXQARunner::getResourceGlobal())
    , mForceXQA(tensorrt_llm::common::forceXQAKernels())
    , mSM(tensorrt_llm::common::getSMVersion())
{
}

DecoderXQARunner::~DecoderXQARunner() = default;

bool DecoderXQARunner::shouldUse(XQAParams const& umbrellaXQAParams, bool forConfigurePlugin)
{
    if (forConfigurePlugin)
    {
        for (int beamWidth = 1; beamWidth <= umbrellaXQAParams.beam_width; ++beamWidth)
        {
            XQAParams actualXQAParams = umbrellaXQAParams;
            actualXQAParams.beam_width = beamWidth;
            if (supportConfig(actualXQAParams, forConfigurePlugin))
            {
                return true;
            }
        }
        TLLM_LOG_DEBUG("JIT XQA is not used: no supported configuration found for any beam_width");
        return false;
    }
    else
    {
        auto const& xqaParams = umbrellaXQAParams;
        bool isConfigSupported = supportConfig(xqaParams, forConfigurePlugin);
        if (!isConfigSupported)
        {
            TLLM_LOG_DEBUG("JIT XQA is not used: unsupported configuration");
            return false;
        }
        bool hasPerfGain = mayHavePerfGain(xqaParams);
        if (!hasPerfGain)
        {
            if (!xqaParams.is_fp8_output && xqaParams.kv_cache_data_type == DATA_TYPE_E4M3
                && (xqaParams.data_type == DATA_TYPE_BF16 || xqaParams.data_type == DATA_TYPE_FP16))
            {
                TLLM_LOG_DEBUG(
                    "JIT XQA is selected in the generation phase for fp16/bf16 input and e4m3 kv cache because MMHA "
                    "does not support this combination.");
                return true;
            }
            TLLM_LOG_DEBUG("JIT XQA is not used: maybe no performance gain");
            return false;
        }
        return true;
    }
}

void DecoderXQARunner::prepare(XQAParams const& umbrellaXQAParams)
{
    for (int beamWidth = 1; beamWidth <= umbrellaXQAParams.beam_width; ++beamWidth)
    {
        XQAParams actualXQAParams = umbrellaXQAParams;
        actualXQAParams.beam_width = beamWidth;
        prepareForActualXQAParams(actualXQAParams);
        if (needHMMASpecDec(umbrellaXQAParams, true))
        {
            actualXQAParams.generation_input_length = 16; // a WAR to generate tileSize=32 JIT cubin
            prepareForActualXQAParams(actualXQAParams);
        }
    }
}

template <>
void DecoderXQARunner::run(XQAParams const& xqaParams, KVLinearBuffer const& kvLinearBuffer, cudaStream_t const& stream)
{
    runDispatchKVCacheBuffer<KVLinearBuffer>(xqaParams, kvLinearBuffer, stream);
}

template <>
void DecoderXQARunner::run(XQAParams const& xqaParams, KVBlockArray const& kvBlockArray, cudaStream_t const& stream)
{
    runDispatchKVCacheBuffer<KVBlockArray>(xqaParams, kvBlockArray, stream);
}

std::shared_ptr<DecoderXQARunnerResource> DecoderXQARunner::getResourceGlobal()
{
    static std::mutex sMutex;
    static std::weak_ptr<DecoderXQARunnerResource> sResource;
    std::lock_guard<std::mutex> lock(sMutex);
    auto ret = sResource.lock();
    if (ret != nullptr)
    {
        return ret;
    }
    ret = std::make_shared<DecoderXQARunnerResource>();
    sResource = ret;
    return ret;
}

bool DecoderXQARunner::needHMMASpecDec(XQAParams const& xqaParams, bool forConfigurePlugin) const
{

    return xqaParams.multi_query_tokens && !jit::supportConfigQGMMA(xqaParams, mSM, forConfigurePlugin)
        && jit::supportConfigHMMA(xqaParams, mSM, forConfigurePlugin)
        && !jit::supportConfigMLA(xqaParams, mSM, forConfigurePlugin);
}

bool DecoderXQARunner::supportConfig(XQAParams const& xqaParams, bool forConfigurePlugin) const
{

    return jit::supportConfigQGMMA(xqaParams, mSM, forConfigurePlugin)
        || jit::supportConfigHMMA(xqaParams, mSM, forConfigurePlugin)
        || jit::supportConfigMLA(xqaParams, mSM, forConfigurePlugin);
}

bool DecoderXQARunner::mayHavePerfGain(XQAParams const& xqaParams) const
{
    // NOTE: only XQA supports multi_query_tokens (Medusa mode).
    if (mForceXQA || xqaParams.multi_query_tokens)
    {
        return true;
    }
    // Always prefer XQA-based MLA over FMHA-base MLA for now.
    if (xqaParams.isMLA())
    {
        return true;
    }
    int numKVHeads = xqaParams.num_kv_heads;
    int batchSize = static_cast<int>(xqaParams.batch_size);
    int multiBlockCount = 1;
    if (xqaParams.multi_block_mode)
    {
        int historyLength = xqaParams.max_past_kv_length;
        // Always use at least 1 block regardless of history length
        multiBlockCount = std::max(1, historyLength / kMinHistoryTokensPerBlock);
    }
    int blockCount = numKVHeads * batchSize * multiBlockCount;
    return static_cast<float>(blockCount) * kEnableMinBlockFactor >= static_cast<float>(mMultiProcessorCount);
}

jit::CubinObjKey DecoderXQARunner::getCubinObjKeyFromXQAParams(XQAParams const& xqaParams) const
{
    XQAKernelLoadHashKey loadKey;
    loadKey.data_type = xqaParams.data_type;
    loadKey.sm = mSM;

    XQAKernelRuntimeHashKey runtimeKey = getRuntimeHashKeyFromXQAParams(xqaParams, mSM);
    return {loadKey, runtimeKey};
}

void DecoderXQARunner::prepareForActualXQAParams(XQAParams const& xqaParams)
{
    jit::CubinObjKey currentKey = getCubinObjKeyFromXQAParams(xqaParams);

    jit::CompileEngine compileEngine(mSM, xqaParams);

    auto registryGlobal = mResource->getCubinObjRegistry();

    if (supportConfig(xqaParams, true))
    {
        jit::CubinObjKey key = getCubinObjKeyFromXQAParams(xqaParams);
        registryGlobal->insertCubinIfNotExists(key, &compileEngine, /*initialize=*/true);
    }
}

#define XQA_KERNEL_RUN(DATA_TYPE)                                                                                      \
    runImpl<DATA_TYPE, KVCacheBuffer>(xqaParams, kvCacheBuffer, mMultiProcessorCount, stream)

template <typename KVCacheBuffer>
void DecoderXQARunner::runDispatchKVCacheBuffer(
    XQAParams const& xqaParams, KVCacheBuffer const& kvCacheBuffer, cudaStream_t const& stream)
{
    if (mDataType == DATA_TYPE_FP16)
    {
        XQA_KERNEL_RUN(__half);
    }
    else
    {
        XQA_KERNEL_RUN(__nv_bfloat16);
    }
}

#undef XQA_KERNEL_RUN

namespace
{
struct SpecDecParams
{
    uint32_t qSeqLen;
    uint32_t const* qCuSeqLens; // [nbReq + 1]
    using MaskType = uint32_t;
    MaskType const* mask;       // [nbReq][qSeqLen][divUp(qSeqLen, 32)] or [qCuSeqLen[nbReq]][divUp(qSeqLen, 32)]
};
} // namespace

template <typename T, typename KVCacheBuffer>
void DecoderXQARunner::runImpl(XQAParams const& xqaParams, KVCacheBuffer const& kvCacheBuffer, int multiprocessorCount,
    cudaStream_t const& stream) const
{
    jit::CubinObjKey const key = getCubinObjKeyFromXQAParams(xqaParams);
    jit::CubinObj const* const cubinObj = mResource->getCubinObjRegistry()->getCubin(key);
    TLLM_CHECK(cubinObj != nullptr && cubinObj->isInitialized());
    bool const isSpecDec = xqaParams.multi_query_tokens;
    bool const isSkipSoftmax = xqaParams.skip_softmax_threshold_scale_factor != 0;
    bool const isHMMAKernel = (cubinObj->getKernelType() == XQAKernelType::kAMPERE_WARP_SPECIALIZED);
    bool const isGMMAKernel = (cubinObj->getKernelType() == XQAKernelType::kHOPPER_WARP_SPECIALIZED);
    bool const isMLAKernel = (cubinObj->getKernelType() == XQAKernelType::kSM120_MLA);
    TLLM_CHECK_WITH_INFO(!isSpecDec || isGMMAKernel || isHMMAKernel
            || (isMLAKernel && !xqaParams.spec_decoding_is_generation_length_variable),
        "speculative decoding is available for GMMA/MLA kernel only in JIT path for now. For MLA, the input sequence "
        "length must be uniform and draft tokens must be linear.");
    TLLM_CHECK_DEBUG(isGMMAKernel == jit::supportConfigQGMMA(xqaParams, mSM, false));
    // @fixme: also embed these compile-time flags in cubin directly
    // Whether RoPE is fused into the XQA kernel.
    //  * If applyRoPEInXqaKernel is true, XQA kernel applies RoPE AND performs SDPA.
    //  * If applyRoPEInXqaKernel is false, a separate kernel applies RoPE (see invokeQKVPreprocessing), then XQA kernel
    //  performs SDPA.
    //    In this case, xqaQInputPtr (see below) serves as the scratch space to store intermediate RoPE output.

    bool const applyRoPEInXqaKernel = jit::appliesRoPEInXqaKernel(xqaParams, isGMMAKernel);

    int numQHeads = xqaParams.num_q_heads;
    int numKVHeads = xqaParams.num_kv_heads;
    TLLM_CHECK_WITH_INFO(numQHeads % numKVHeads == 0, "numQHeads should be multiple of numKVHeads.");
    unsigned int numQHeadsOverKV = numQHeads / numKVHeads;
    unsigned int beamWidth = xqaParams.beam_width;
    unsigned int batchBeamSize = xqaParams.batch_size * beamWidth;

    const KvCacheDataType cacheType = xqaParams.kv_cache_quant_mode.hasInt8KvCache()
        ? KvCacheDataType::INT8
        : (xqaParams.kv_cache_quant_mode.hasFp8KvCache() ? KvCacheDataType::FP8 : KvCacheDataType::BASE);

    XQALaunchParam<KVCacheBuffer> launchParams;
    bool const isFp8Out = xqaParams.is_fp8_output;
    bool const needOutputCvt = false;
    void* inputScratch = nullptr;
    buildXQALaunchParams(launchParams, inputScratch, needOutputCvt, xqaParams, kvCacheBuffer);
    if (needOutputCvt)
    {
        launchParams.output = inputScratch;
    }

    // NOTE: MHA kernels should read kv cache that has already been appended with new tokens' kv cache.
    void* xqaQInputPtr = (applyRoPEInXqaKernel ? nullptr : inputScratch);
    if (!applyRoPEInXqaKernel)
    {
        if (!xqaParams.isMLA())
        {
            // Build cu_seqlens, padding_offset, and rotary inv freq tensors
            BuildDecoderInfoParams<T> decoderParams{};
            decoderParams.seqQOffsets = launchParams.cu_seq_lens;
            decoderParams.seqQLengths = xqaParams.spec_decoding_generation_lengths;
            decoderParams.seqKVLengths = xqaParams.sequence_lengths;
            decoderParams.tokensInfo = launchParams.tokens_info;
            decoderParams.batchSize = int(batchBeamSize);
            decoderParams.maxQSeqLength = xqaParams.generation_input_length;
            decoderParams.numTokens = xqaParams.total_num_input_tokens;
            decoderParams.removePadding = true;
            TLLM_CHECK_WITH_INFO(!xqaParams.multi_query_tokens || xqaParams.spec_decoding_generation_lengths != nullptr,
                "Spec_decoding_generation_lengths must be provided.");
            // Rotary embedding inv_freq buffer.
            decoderParams.rotaryEmbeddingScale = xqaParams.rotary_embedding_scale;
            decoderParams.rotaryEmbeddingBase = xqaParams.rotary_embedding_base;
            decoderParams.rotaryEmbeddingDim = xqaParams.rotary_embedding_dim;
            decoderParams.rotaryScalingType = xqaParams.rotary_embedding_scale_type;
            decoderParams.rotaryEmbeddingInvFreq = launchParams.rotary_inv_freq_buf;
            decoderParams.rotaryEmbeddingInvFreqCache = xqaParams.rotary_embedding_inv_freq_cache;
            decoderParams.rotaryEmbeddingMaxPositions = xqaParams.rotary_embedding_max_positions;
            // The rotary_embedding_inv_freq_cache for QKVPreprocessing.
            // Use the xqaParams.rotary_embedding_inv_freq_cache input when the buildDecoderInfoKernel is skipped.
            float const* rotaryInvFreqBuf = xqaParams.rotary_embedding_inv_freq_cache;
            if (decoderParams.isBuildDecoderInfoKernelNeeded() || xqaParams.multi_query_tokens)
            {
                rotaryInvFreqBuf = launchParams.rotary_inv_freq_buf;
                invokeBuildDecoderInfo(decoderParams, stream);
            }
            sync_check_cuda_error(stream);

            // The preprocessing kernel that applies RoPE and updates kv cache.
            QKVPreprocessingParams<T, KVCacheBuffer> preprocessingParams;
            memset(&preprocessingParams, 0, sizeof(preprocessingParams));
            // Set parameters.
            preprocessingParams.qkv_input = static_cast<T*>(const_cast<void*>(xqaParams.qkv));
            preprocessingParams.q_output = static_cast<T*>(xqaQInputPtr);
            preprocessingParams.kv_cache_buffer = kvCacheBuffer;
            preprocessingParams.kv_cache_block_scales_buffer = {};
            preprocessingParams.qkv_bias = static_cast<T const*>(xqaParams.qkv_bias);
            // Buffers.
            preprocessingParams.logn_scaling = xqaParams.logn_scaling_ptr;
            preprocessingParams.tokens_info = launchParams.tokens_info;
            preprocessingParams.seq_lens = xqaParams.spec_decoding_generation_lengths;
            preprocessingParams.cache_seq_lens = xqaParams.sequence_lengths;
            preprocessingParams.cu_seq_lens = xqaParams.multi_query_tokens ? launchParams.cu_seq_lens : nullptr;
            preprocessingParams.rotary_embedding_inv_freq = rotaryInvFreqBuf;
            preprocessingParams.rotary_coef_cache_buffer = xqaParams.rotary_cos_sin;
            preprocessingParams.qkv_scale_orig_quant = xqaParams.kv_scale_orig_quant;
            preprocessingParams.spec_decoding_position_offsets = xqaParams.spec_decoding_position_offsets;
            preprocessingParams.mrope_position_deltas = xqaParams.mrope_position_deltas;
            // Scalar parameters.
            preprocessingParams.batch_size = int(batchBeamSize);
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
            preprocessingParams.cache_type = cacheType;
            preprocessingParams.separate_q_kv_output = true;
            preprocessingParams.quantized_fp8_output = false;
            preprocessingParams.generation_phase = true;
            preprocessingParams.multi_processor_count = multiprocessorCount;
            preprocessingParams.rotary_vision_start = xqaParams.rotary_vision_start;
            preprocessingParams.rotary_vision_length = xqaParams.rotary_vision_length;

            invokeQKVPreprocessing<T, KVCacheBuffer>(preprocessingParams, stream);
            sync_check_cuda_error(stream);
        }
        else
        {
            xqaQInputPtr = xqaParams.quant_q_buffer_ptr;
        }
    }

    auto const makeSpecDecParams = [&]() -> SpecDecParams
    {
        auto const qSeqLen = static_cast<uint32_t>(xqaParams.generation_input_length);
        uint32_t maxQSeqLen = xqaParams.spec_decoding_is_generation_length_variable
            ? xqaParams.spec_decoding_max_generation_length
            : qSeqLen;
        return {.qSeqLen = maxQSeqLen,
            .qCuSeqLens = reinterpret_cast<uint32_t const*>(launchParams.cu_seq_lens),
            .mask = reinterpret_cast<SpecDecParams::MaskType const*>(xqaParams.spec_decoding_packed_mask)};
    };

    constexpr uint32_t kMAX_NB_KERNEL_PARAMS = 19;
    uint32_t idxNextParam = 0;
    void* kernelParams[kMAX_NB_KERNEL_PARAMS];
    auto appendParam = [&](auto* p) mutable
    {
        TLLM_CHECK(idxNextParam < kMAX_NB_KERNEL_PARAMS);
        kernelParams[idxNextParam++] = const_cast<void*>(static_cast<void const*>(p));
    };
    void const* const kernelInputTokens = (applyRoPEInXqaKernel ? launchParams.qkv : xqaQInputPtr);
    if (isMLAKernel)
    {
        CUtensorMap const tensorMapQ = makeTensorMapForXqaMlaQ(mDriver, xqaParams, kernelInputTokens);
        appendParam(&tensorMapQ);
        CUtensorMap const tensorMapK = makeTensorMapForXqaMlaKVCache(mDriver, xqaParams, kvCacheBuffer, true);
        appendParam(&tensorMapK);
        CUtensorMap const tensorMapV = makeTensorMapForXqaMlaKVCache(mDriver, xqaParams, kvCacheBuffer, false);
        appendParam(&tensorMapV);
        appendParam(&launchParams.qScale);
        appendParam(&launchParams.output);
        appendParam(&launchParams.kvCacheParams);
        appendParam(&launchParams.batch_size);
        appendParam(&launchParams.kv_scale_quant_orig);
        appendParam(&launchParams.scratch);
        appendParam(&launchParams.semaphores);
        uint32_t const multiBlock = computeMultiBlockCountForMLA(xqaParams, multiprocessorCount);
        std::byte* const partialResults = static_cast<std::byte*>(launchParams.scratch)
            + xqaMlaCgaXBufSize * multiBlock * xqaParams.total_num_input_tokens;
        appendParam(&partialResults);
        kernelParams[idxNextParam] = nullptr; // one extra nullptr at end as guard.
        uint32_t const inputSeqLen = (xqaParams.multi_query_tokens || xqaParams.isMLA())
            ? static_cast<uint32_t>(xqaParams.generation_input_length)
            : 1U;
        dim3 const dimGrid{4 * inputSeqLen, multiBlock, xqaParams.batch_size};
        dim3 const blockDim(128 * 3, 1, 1);
        cubinObj->launch(dimGrid, blockDim, stream, kernelParams);
    }
    else if (isSpecDec && isHMMAKernel)
    {
        // MultiQueryTokens (generation_input_length > 1) need extra parameters (like qSeqLen, headGrpSize, and
        // mask). Input parameters for MultiQueryTokens kernels.
        unsigned int headGrpSize = numQHeadsOverKV;
        // Use mTileSize = 16 kernels when qSeqLen <= 16.
        unsigned int qSeqLen = static_cast<unsigned int>(xqaParams.generation_input_length);
        unsigned int mTileSize = qSeqLen <= 16 ? 16 : 32;
        unsigned int nbTokenBlocksPerGrp = divUp(qSeqLen * headGrpSize, mTileSize);
        unsigned int maxQSeqLen = xqaParams.spec_decoding_is_generation_length_variable ? // true for ReDrafter
            xqaParams.spec_decoding_max_generation_length
                                                                                        : qSeqLen;

        appendParam(&maxQSeqLen);
        appendParam(&launchParams.num_k_heads);
        appendParam(&headGrpSize);
        appendParam(&launchParams.cu_seq_lens);
        bool const allowSlidingWindow
            = !(isSpecDec && xqaParams.is_spec_dec_tree); // sliding windows does not support spec dec with tree-based
                                                          // token, only chained tokens
        if (allowSlidingWindow)
        {
            appendParam(&launchParams.slidingWindowSize);
        }
        appendParam(&launchParams.qScale);
        appendParam(&launchParams.output);
        if (isFp8Out && !needOutputCvt)
        {
            appendParam(&launchParams.rcpOutScale);
        }
        appendParam(&kernelInputTokens);
        appendParam(&xqaParams.spec_decoding_packed_mask);
        appendParam(&xqaParams.attention_sinks);
        appendParam(&launchParams.kvCacheParams);
        if (xqaParams.beam_width > 1)
        {
            appendParam(&launchParams.beamSearchParams.value());
        }
        appendParam(&launchParams.batch_size);
        appendParam(&launchParams.kv_scale_quant_orig);
        appendParam(&launchParams.semaphores);
        appendParam(&launchParams.scratch);

        uint32_t multiBlock = 1;
        // if (xqaParams.multi_block_mode)
        // {
        //     multiBlock = computeMultiBlockCount(xqaParams, xqaParams.batch_size, multiprocessorCount);
        // }
        auto const gridDim = (dim3{multiBlock, xqaParams.num_kv_heads * nbTokenBlocksPerGrp, xqaParams.batch_size});
        dim3 const blockDim(128, 1, 2);

        cubinObj->launch(gridDim, blockDim, stream, kernelParams);
    }
    else
    {
        appendParam(&launchParams.num_k_heads);
        bool const allowSlidingWindow
            = !(isSpecDec && xqaParams.is_spec_dec_tree); // sliding windows does not support spec dec with tree-based
                                                          // token, only chained tokens
        if (allowSlidingWindow)
        {
            appendParam(&launchParams.slidingWindowSize);
        }
        appendParam(&launchParams.qScale);
        appendParam(&launchParams.output);
        if (isFp8Out && !needOutputCvt)
        {
            appendParam(&launchParams.rcpOutScale);
        }
        appendParam(&kernelInputTokens);
        if (applyRoPEInXqaKernel)
        {
            appendParam(&launchParams.ropeCosSin);
        }
        appendParam(&xqaParams.attention_sinks);
        appendParam(&launchParams.kvCacheParams);
        if (xqaParams.beam_width > 1)
        {
            appendParam(&launchParams.beamSearchParams.value());
        }
        appendParam(&launchParams.batch_size);
        appendParam(&launchParams.kv_scale_quant_orig);
        CUtensorMap tensorMap{};
        if (isGMMAKernel)
        {
            tensorMap = makeTensorMapForHopperXqaKVCache(mDriver, xqaParams, kvCacheBuffer);
            appendParam(&tensorMap);
        }
        uint32_t specDecBlocks = 1;
        SpecDecParams specDecParams{};
        if (isSpecDec)
        {
            TLLM_CHECK_WITH_INFO(
                isGMMAKernel, "speculative decoding is available for GMMA kernel only in JIT path for now.");
            TLLM_CHECK_DEBUG_WITH_INFO(xqaParams.max_past_kv_length + 1 <= xqaParams.cyclic_attention_window_size,
                "SWA and speculative decoding cannot be used at the same time for now.");
            specDecParams = makeSpecDecParams();
            appendParam(&specDecParams);
            specDecBlocks = divUp(specDecParams.qSeqLen, 64 / numQHeadsOverKV);
        }
        if (isSkipSoftmax)
        {
            TLLM_CHECK_WITH_INFO(isGMMAKernel, "skip softmax is only supported for GMMA kernel for now.");
            TLLM_CHECK_WITH_INFO(!isSpecDec, "skip softmax is not supported with spec dec for now.");
            appendParam(&xqaParams.skip_softmax_threshold_scale_factor);
#ifdef SKIP_SOFTMAX_STAT
            appendParam(&xqaParams.skip_softmax_total_blocks);
            appendParam(&xqaParams.skip_softmax_skipped_blocks);
#endif
        }
        appendParam(&launchParams.semaphores);
        appendParam(&launchParams.scratch);
        kernelParams[idxNextParam] = nullptr; // one extra nullptr at end as guard.
        uint32_t multiBlock = 1;
        if (xqaParams.multi_block_mode)
        {
            if (isSpecDec && isGMMAKernel)
            {
                multiBlock = computeMultiBlockCountSpecDecGMMA(
                    xqaParams, xqaParams.batch_size, multiprocessorCount, specDecBlocks);
            }
            else if (!isSpecDec)
            {
                multiBlock = computeMultiBlockCount(xqaParams, xqaParams.batch_size, multiprocessorCount);
            }
        }
        uint32_t const nbKVHeads = xqaParams.num_kv_heads;
        auto const gridDim = (isGMMAKernel ? dim3{specDecBlocks, multiBlock, nbKVHeads * xqaParams.batch_size}
                                           : dim3{multiBlock, nbKVHeads, xqaParams.batch_size});
        dim3 const blockDim(128, 1, isGMMAKernel ? 3 : 2);
        cubinObj->launch(gridDim, blockDim, stream, kernelParams);
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

} // namespace kernels

TRTLLM_NAMESPACE_END
