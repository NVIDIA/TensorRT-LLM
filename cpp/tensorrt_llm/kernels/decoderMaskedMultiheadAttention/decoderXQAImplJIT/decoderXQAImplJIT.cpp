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
#include "tensorrt_llm/common/envUtils.h"
#include "tensorrt_llm/common/utils.h"
#include "tensorrt_llm/kernels/decoderMaskedMultiheadAttention/cubin/xqa_kernel_cubin.h"
#include "tensorrt_llm/kernels/decoderMaskedMultiheadAttention/decoderXQAConstants.h"
#include "tensorrt_llm/kernels/decoderMaskedMultiheadAttention/decoderXQAImplJIT/decoderXQAImplJIT.h"
#include "tensorrt_llm/kernels/decoderMaskedMultiheadAttention/decoderXQAImplJIT/kernelUtils.h"
#include "tensorrt_llm/kernels/decoderMaskedMultiheadAttention/decoderXQARunner.h"
#include "tensorrt_llm/kernels/decoderMaskedMultiheadAttention/tensorMapUtils.h"
#include "tensorrt_llm/kernels/unfusedAttentionKernels.h"

namespace
{

using ::tensorrt_llm::kernels::XQAKernelRuntimeHashKey;
using ::tensorrt_llm::kernels::XQAParams;
using ::tensorrt_llm::kernels::XQAKernelMetaInfo;

XQAKernelRuntimeHashKey getRuntimeHashKeyFromKernelMeta(XQAKernelMetaInfo const& kernelMeta)
{
    return {kernelMeta.mKVDataType, kernelMeta.mHeadDim, kernelMeta.mBeamWidth, kernelMeta.mNumQHeadsOverKV,
        kernelMeta.mMTileSize, kernelMeta.mTokensPerPage, kernelMeta.mPagedKVCache, kernelMeta.mMultiQueryTokens};
}

} // anonymous namespace

namespace tensorrt_llm
{
namespace kernels
{

DecoderXQAImplJIT::DecoderXQAImplJIT(DecoderXQARunner* runner)
    : DecoderXQAImpl(runner)
    , mDriver(tensorrt_llm::common::CUDADriverWrapper::getInstance())
    , mForceXQA(tensorrt_llm::common::forceXQAKernels())
    , mSM(tensorrt_llm::common::getSMVersion())
{
}

bool DecoderXQAImplJIT::supportConfig(XQAParams const& xqaParams, bool forConfigurePlugin) const
{

    return jit::supportConfigQGMMA(xqaParams, mSM, forConfigurePlugin)
        || jit::supportConfigHMMA(xqaParams, mSM, forConfigurePlugin);
}

bool DecoderXQAImplJIT::mayHavePerfGain(XQAParams const& xqaParams) const
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
        return false;
    }
    else
    {
        auto const& xqaParams = umbrellaXQAParams;
        return supportConfig(xqaParams, forConfigurePlugin) && mayHavePerfGain(xqaParams);
    }
}

jit::CubinObjKey DecoderXQAImplJIT::getCubinObjKeyFromXQAParams(XQAParams const& xqaParams) const
{
    XQAKernelLoadHashKey loadKey;
    loadKey.data_type = xqaParams.data_type;
    loadKey.sm = mSM;

    XQAKernelRuntimeHashKey runtimeKey = getRuntimeHashKeyFromXQAParams(xqaParams);
    return {loadKey, runtimeKey};
}

void DecoderXQAImplJIT::prepareForActualXQAParams(XQAParams const& xqaParams)
{
    jit::CubinObjKey currentKey = getCubinObjKeyFromXQAParams(xqaParams);

    jit::CompileEngine compileEngine(mSM, xqaParams);

    auto registryGlobal = DecoderXQARunner::getResourceGlobal()->getCubinObjRegistry();

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

template <typename T, typename KVCacheBuffer>
void DecoderXQAImplJIT::runImpl(XQAParams const& xqaParams, KVCacheBuffer const& kv_cache_buffer,
    int multiprocessor_count, cudaStream_t const& stream)
{
    jit::CubinObjKey const key = getCubinObjKeyFromXQAParams(xqaParams);
    jit::CubinObj const* const cubinObj = DecoderXQARunner::getResourceGlobal()->getCubinObjRegistry()->getCubin(key);
    TLLM_CHECK(cubinObj != nullptr && cubinObj->isInitialized());
    TLLM_CHECK_WITH_INFO(!xqaParams.multi_query_tokens, "Medusa should take XQA Precompiled codepath.");
    bool const isGMMAKernel = (cubinObj->getKernelType() == XQAKernelType::kHOPPER_WARP_SPECIALIZED);
    TLLM_CHECK_DEBUG(isGMMAKernel == jit::supportConfigQGMMA(xqaParams, mSM, false));
    // @fixme: also embed these compile-time flags in cubin directly
    // Whether RoPE is fused into the XQA kernel.
    //  * If applyRoPEInXqaKernel is true, XQA kernel applies RoPE AND performs SDPA.
    //  * If applyRoPEInXqaKernel is false, a separate kernel applies RoPE (see invokeQKVPreprocessing), then XQA kernel
    //  performs SDPA.
    //    In this case, xqa_q_input_ptr (see below) serves as the scratch space to store intermediate RoPE output.
    bool const applyRoPEInXqaKernel = isGMMAKernel
        && tensorrt_llm::common::contains({PositionEmbeddingType::kLONG_ROPE, PositionEmbeddingType::kROPE_GPT_NEOX,
                                              PositionEmbeddingType::kROPE_GPTJ},
            xqaParams.position_embedding_type);

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
        // Build cu_seqlens, padding_offset, and rotary inv freq tensors
        BuildDecoderInfoParams<T> decoder_params;
        memset(&decoder_params, 0, sizeof(decoder_params));
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
        if (decoder_params.isBuildDecoderInfoKernelNeeded())
        {
            rotary_inv_freq_buf = launchParams.rotary_inv_freq_buf;
            invokeBuildDecoderInfo(decoder_params, stream);
        }
        sync_check_cuda_error();

        QKVPreprocessingParams<T, KVCacheBuffer> preprocessingParams{static_cast<T*>(const_cast<void*>(xqaParams.qkv)),
            nullptr, nullptr, static_cast<T*>(xqa_q_input_ptr), kv_cache_buffer,
            /* kv_cache_block_scales_buffer */ KVCacheBuffer{}, static_cast<T const*>(xqaParams.qkv_bias),
            xqaParams.logn_scaling_ptr, launchParams.tokens_info, xqaParams.spec_decoding_generation_lengths,
            xqaParams.sequence_lengths,
            /* encoder_seqlens */ nullptr, xqaParams.multi_query_tokens ? launchParams.cu_seq_lens : nullptr,
            /* cu_kv_seqlens */ nullptr, rotary_inv_freq_buf, xqaParams.rotary_cos_sin, xqaParams.kv_scale_orig_quant,
            /* kv_cache_scale_factors */ nullptr, xqaParams.spec_decoding_position_offsets, (float2 const*) nullptr,
            xqaParams.mrope_position_deltas, int(batch_beam_size), xqaParams.generation_input_length,
            xqaParams.max_past_kv_length, xqaParams.cyclic_attention_window_size, xqaParams.sink_token_length,
            xqaParams.total_num_input_tokens,
            /*remove_padding*/ true, /*cross_attention*/ false, xqaParams.num_q_heads, xqaParams.num_kv_heads,
            xqaParams.num_q_heads / xqaParams.num_kv_heads, xqaParams.head_size, xqaParams.rotary_embedding_dim,
            xqaParams.rotary_embedding_base, xqaParams.rotary_embedding_scale_type, xqaParams.rotary_embedding_scale,
            xqaParams.rotary_embedding_max_positions, xqaParams.position_embedding_type,
            xqaParams.position_shift_enabled, cache_type, true, false, /* generation_phase */ true,
            multiprocessor_count, xqaParams.rotary_vision_start, xqaParams.rotary_vision_length};

        invokeQKVPreprocessing<T, KVCacheBuffer>(preprocessingParams, stream);
        sync_check_cuda_error();
    }

    // Use mTileSize = 16 kernels when qSeqLen <= 16.
    unsigned int qSeqLen = static_cast<unsigned int>(xqaParams.generation_input_length);
    unsigned int mTileSize = qSeqLen <= 16 ? 16 : 32;
    // MultiQueryToken kernels can support any num_q_heads_over_kv that is power of 2.
    unsigned int kernel_num_q_heads_over_kv = xqaParams.multi_query_tokens ? 0 : num_q_heads_over_kv;
    // MultiQueryToken kernels can handle either 16/32 for M direction per CTA.
    unsigned int kernel_m_tilesize = xqaParams.multi_query_tokens ? mTileSize : num_q_heads_over_kv;

    constexpr uint32_t kMAX_NB_KERNEL_PARAMS = 14;
    uint32_t idxNextParam = 0;
    void* kernelParams[kMAX_NB_KERNEL_PARAMS];
    auto appendParam = [&](auto* p) mutable
    {
        TLLM_CHECK(idxNextParam < kMAX_NB_KERNEL_PARAMS);
        kernelParams[idxNextParam++] = const_cast<void*>(static_cast<void const*>(p));
    };
    appendParam(&launchParams.num_k_heads);
    appendParam(&launchParams.slidingWindowSize);
    appendParam(&launchParams.qScale);
    appendParam(&launchParams.output);
    if (isFp8Out && !needOutputCvt)
    {
        appendParam(&launchParams.rcpOutScale);
    }
    appendParam(applyRoPEInXqaKernel ? &launchParams.qkv : const_cast<void const**>(&xqa_q_input_ptr));
    if (applyRoPEInXqaKernel)
    {
        appendParam(&launchParams.ropeCosSin);
    }
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

    dim3 gridDim(multi_block, xqaParams.num_kv_heads, xqaParams.batch_size);
    dim3 blockDim(128, 1, isGMMAKernel ? 3 : 2);
    cubinObj->launch(gridDim, blockDim, stream, kernelParams);
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

} // namespace kernels
} // namespace tensorrt_llm
