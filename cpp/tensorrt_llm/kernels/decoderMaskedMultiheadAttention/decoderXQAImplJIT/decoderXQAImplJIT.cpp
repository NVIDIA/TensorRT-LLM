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

#include "tensorrt_llm/kernels/decoderMaskedMultiheadAttention/decoderXQAImplJIT/decoderXQAImplJIT.h"
#include "compileEngine.h"
#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/envUtils.h"
#include "tensorrt_llm/common/utils.h"
#include "tensorrt_llm/kernels/decoderMaskedMultiheadAttention/cubin/xqa_kernel_cubin.h"
#include "tensorrt_llm/kernels/decoderMaskedMultiheadAttention/decoderXQAConstants.h"
#include "tensorrt_llm/kernels/decoderMaskedMultiheadAttention/decoderXQAImplJIT/kernelUtils.h"
#include "tensorrt_llm/kernels/decoderMaskedMultiheadAttention/decoderXQARunner.h"
#include "tensorrt_llm/kernels/decoderMaskedMultiheadAttention/tensorMapUtils.h"
#include "tensorrt_llm/kernels/unfusedAttentionKernels.h"
#include "tensorrt_llm/kernels/xqaDispatcher.h"

namespace
{

using ::tensorrt_llm::kernels::XQAKernelRuntimeHashKey;
using ::tensorrt_llm::kernels::XQAParams;
using ::tensorrt_llm::kernels::XQAKernelMetaInfo;

XQAKernelRuntimeHashKey getRuntimeHashKeyFromKernelMeta(XQAKernelMetaInfo const& kernelMeta)
{
    return {kernelMeta.mKVDataType, kernelMeta.mHeadDim, kernelMeta.mBeamWidth, kernelMeta.mNumQHeadsOverKV,
        kernelMeta.mMTileSize, kernelMeta.mTokensPerPage, kernelMeta.mPagedKVCache, kernelMeta.mMultiQueryTokens, false,
        std::nullopt};
}

} // anonymous namespace

namespace tensorrt_llm::kernels
{

DecoderXQAImplJIT::DecoderXQAImplJIT(DecoderXQARunner* runner)
    : DecoderXQAImpl(runner)
    , mDriver(tensorrt_llm::common::CUDADriverWrapper::getInstance())
    , mResource(DecoderXQARunner::getResourceGlobal())
    , mForceXQA(tensorrt_llm::common::forceXQAKernels())
    , mSM(tensorrt_llm::common::getSMVersion())
{
}

bool DecoderXQAImplJIT::needHMMASpecDec(XQAParams const& xqaParams, bool forConfigurePlugin) const
{

    return xqaParams.multi_query_tokens && !jit::supportConfigQGMMA(xqaParams, mSM, forConfigurePlugin)
        && jit::supportConfigHMMA(xqaParams, mSM, forConfigurePlugin)
        && !jit::supportConfigMLA(xqaParams, mSM, forConfigurePlugin);
}

bool DecoderXQAImplJIT::supportConfig(XQAParams const& xqaParams, bool forConfigurePlugin) const
{

    return jit::supportConfigQGMMA(xqaParams, mSM, forConfigurePlugin)
        || jit::supportConfigHMMA(xqaParams, mSM, forConfigurePlugin)
        || jit::supportConfigMLA(xqaParams, mSM, forConfigurePlugin);
}

bool DecoderXQAImplJIT::mayHavePerfGain(XQAParams const& xqaParams) const
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
    int num_kv_heads = xqaParams.num_kv_heads;
    int batch_size = static_cast<int>(xqaParams.batch_size);
    int multi_block_count = 1;
    if (xqaParams.multi_block_mode)
    {
        int history_length = xqaParams.max_past_kv_length;
        // Always use at least 1 block regardless of history length
        multi_block_count = std::max(1, history_length / kMinHistoryTokensPerBlock);
    }
    int block_count = num_kv_heads * batch_size * multi_block_count;
    return static_cast<float>(block_count) * kEnableMinBlockFactor >= static_cast<float>(mRunner->mMultiProcessorCount);
}

bool DecoderXQAImplJIT::shouldUse(XQAParams const& umbrellaXQAParams, bool forConfigurePlugin)
{
    if (forConfigurePlugin)
    {
        for (int beam_width = 1; beam_width <= umbrellaXQAParams.beam_width; ++beam_width)
        {
            XQAParams actualXQAParams = umbrellaXQAParams;
            actualXQAParams.beam_width = beam_width;
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
            TLLM_LOG_DEBUG("JIT XQA is not used: maybe no performance gain");
            return false;
        }
        return true;
    }
}

jit::CubinObjKey DecoderXQAImplJIT::getCubinObjKeyFromXQAParams(XQAParams const& xqaParams) const
{
    XQAKernelLoadHashKey loadKey;
    loadKey.data_type = xqaParams.data_type;
    loadKey.sm = mSM;

    XQAKernelRuntimeHashKey runtimeKey = getRuntimeHashKeyFromXQAParams(xqaParams, true, mSM);
    return {loadKey, runtimeKey};
}

void DecoderXQAImplJIT::prepareForActualXQAParams(XQAParams const& xqaParams)
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

void DecoderXQAImplJIT::prepare(XQAParams const& umbrellaXQAParams)
{
    for (int beam_width = 1; beam_width <= umbrellaXQAParams.beam_width; ++beam_width)
    {
        XQAParams actualXQAParams = umbrellaXQAParams;
        actualXQAParams.beam_width = beam_width;
        prepareForActualXQAParams(actualXQAParams);
        if (needHMMASpecDec(umbrellaXQAParams, true))
        {
            actualXQAParams.generation_input_length = 16; // a WAR to generate tileSize=32 JIT cubin
            prepareForActualXQAParams(actualXQAParams);
        }
    }
}

void DecoderXQAImplJIT::runWithKVLinearBuffer(
    XQAParams const& xqaParams, KVLinearBuffer const& kv_linear_buffer, cudaStream_t const& stream)
{
    runDispatchKVCacheBuffer<KVLinearBuffer>(xqaParams, kv_linear_buffer, stream);
}

void DecoderXQAImplJIT::runWithKVBlockArray(
    XQAParams const& xqaParams, KVBlockArray const& kv_block_array, cudaStream_t const& stream)
{
    runDispatchKVCacheBuffer<KVBlockArray>(xqaParams, kv_block_array, stream);
}

#define XQA_KERNEL_RUN(DATA_TYPE)                                                                                      \
    runImpl<DATA_TYPE, KVCacheBuffer>(xqa_params, kv_cache_buffer, mRunner->mMultiProcessorCount, stream)

template <typename KVCacheBuffer>
void DecoderXQAImplJIT::runDispatchKVCacheBuffer(
    XQAParams const& xqa_params, KVCacheBuffer const& kv_cache_buffer, cudaStream_t const& stream)
{
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
void DecoderXQAImplJIT::runImpl(XQAParams const& xqaParams, KVCacheBuffer const& kv_cache_buffer,
    int multiprocessor_count, cudaStream_t const& stream) const
{
    jit::CubinObjKey const key = getCubinObjKeyFromXQAParams(xqaParams);
    jit::CubinObj const* const cubinObj = mResource->getCubinObjRegistry()->getCubin(key);
    TLLM_CHECK(cubinObj != nullptr && cubinObj->isInitialized());
    bool const isSpecDec = xqaParams.multi_query_tokens;
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
    //    In this case, xqa_q_input_ptr (see below) serves as the scratch space to store intermediate RoPE output.
    bool const applyRoPEInXqaKernel = isGMMAKernel && !isSpecDec
        && tensorrt_llm::common::contains({PositionEmbeddingType::kLONG_ROPE, PositionEmbeddingType::kROPE_GPT_NEOX,
                                              PositionEmbeddingType::kROPE_GPTJ},
            xqaParams.position_embedding_type)
        && !xqaParams.isMLA();

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
    bool const isFp8Out = xqaParams.is_fp8_output;
    bool const needOutputCvt = false;
    void* inputScratch = nullptr;
    buildXQALaunchParams(launchParams, inputScratch, needOutputCvt, xqaParams, kv_cache_buffer);
    if (needOutputCvt)
    {
        launchParams.output = inputScratch;
    }

    // NOTE: MHA kernels should read kv cache that has already been appended with new tokens' kv cache.
    void* xqa_q_input_ptr = (applyRoPEInXqaKernel ? nullptr : inputScratch);
    if (!applyRoPEInXqaKernel)
    {
        if (!xqaParams.isMLA())
        {
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
        }
        else
        {
            xqa_q_input_ptr = xqaParams.quant_q_buffer_ptr;
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

    constexpr uint32_t kMAX_NB_KERNEL_PARAMS = 15;
    uint32_t idxNextParam = 0;
    void* kernelParams[kMAX_NB_KERNEL_PARAMS];
    auto appendParam = [&](auto* p) mutable
    {
        TLLM_CHECK(idxNextParam < kMAX_NB_KERNEL_PARAMS);
        kernelParams[idxNextParam++] = const_cast<void*>(static_cast<void const*>(p));
    };
    void const* const kernel_input_tokens = (applyRoPEInXqaKernel ? launchParams.qkv : xqa_q_input_ptr);
    if (isMLAKernel)
    {
        CUtensorMap const tensorMapQ = makeTensorMapForXqaMlaQ(mDriver, xqaParams, kernel_input_tokens);
        appendParam(&tensorMapQ);
        CUtensorMap const tensorMapK = makeTensorMapForXqaMlaKVCache(mDriver, xqaParams, kv_cache_buffer, true);
        appendParam(&tensorMapK);
        CUtensorMap const tensorMapV = makeTensorMapForXqaMlaKVCache(mDriver, xqaParams, kv_cache_buffer, false);
        appendParam(&tensorMapV);
        appendParam(&launchParams.qScale);
        appendParam(&launchParams.output);
        appendParam(&launchParams.kvCacheParams);
        appendParam(&launchParams.batch_size);
        appendParam(&launchParams.kv_scale_quant_orig);
        appendParam(&launchParams.scratch);
        appendParam(&launchParams.semaphores);
        uint32_t const multi_block = computeMultiBlockCountForMLA(xqaParams, multiprocessor_count);
        std::byte* const partialResults = static_cast<std::byte*>(launchParams.scratch)
            + xqaMlaCgaXBufSize * multi_block * xqaParams.total_num_input_tokens;
        appendParam(&partialResults);
        kernelParams[idxNextParam] = nullptr; // one extra nullptr at end as guard.
        uint32_t const inputSeqLen = (xqaParams.multi_query_tokens || xqaParams.isMLA())
            ? static_cast<uint32_t>(xqaParams.generation_input_length)
            : 1U;
        dim3 const dimGrid{4 * inputSeqLen, multi_block, xqaParams.batch_size};
        dim3 const blockDim(128 * 3, 1, 1);
        cubinObj->launch(dimGrid, blockDim, stream, kernelParams);
    }
    else if (isSpecDec && isHMMAKernel)
    {
        // MultiQueryTokens (generation_input_length > 1) need extra parameters (like qSeqLen, headGrpSize, and
        // mask). Input parameters for MultiQueryTokens kernels.
        unsigned int headGrpSize = num_q_heads_over_kv;
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
        appendParam(&kernel_input_tokens);
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

        uint32_t multi_block = 1;
        // if (xqaParams.multi_block_mode)
        // {
        //     multi_block = computeMultiBlockCount(xqaParams, xqaParams.batch_size, multiprocessor_count);
        // }
        auto const gridDim = (dim3{multi_block, xqaParams.num_kv_heads * nbTokenBlocksPerGrp, xqaParams.batch_size});
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
        appendParam(&kernel_input_tokens);
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
            tensorMap = makeTensorMapForHopperXqaKVCache(mDriver, xqaParams, kv_cache_buffer);
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
            specDecBlocks = divUp(specDecParams.qSeqLen, 64 / num_q_heads_over_kv);
        }
        appendParam(&launchParams.semaphores);
        appendParam(&launchParams.scratch);
        kernelParams[idxNextParam] = nullptr; // one extra nullptr at end as guard.
        uint32_t multi_block = 1;
        if (xqaParams.multi_block_mode)
        {
            if (isSpecDec && isGMMAKernel)
            {
                multi_block = computeMultiBlockCountSpecDecGMMA(
                    xqaParams, xqaParams.batch_size, multiprocessor_count, specDecBlocks);
            }
            else if (!isSpecDec)
            {
                multi_block = computeMultiBlockCount(xqaParams, xqaParams.batch_size, multiprocessor_count);
            }
        }
        uint32_t const nbKVHeads = xqaParams.num_kv_heads;
        auto const gridDim = (isGMMAKernel ? dim3{specDecBlocks, multi_block, nbKVHeads * xqaParams.batch_size}
                                           : dim3{multi_block, nbKVHeads, xqaParams.batch_size});
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

} // namespace tensorrt_llm::kernels
