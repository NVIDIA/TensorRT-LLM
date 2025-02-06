/*
 * Copyright (c) 2020-2024, NVIDIA CORPORATION.  All rights reserved.
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

#include "xqaDispatcher.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/kernels/unfusedAttentionKernels.h"

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

namespace tensorrt_llm::kernels
{

////////////////////////////////////////////////////////////////////////////////////////////////////

XqaDispatcher::XqaDispatcher(XqaFixedParams fixedParams)
    : mFixedParams(fixedParams)
    , mQDataType(mFixedParams.inputDataType)
    , mUseTllmGen(tensorrt_llm::common::getSMVersion() == 100)
    , mMultiProcessorCount(getMultiProcessorCount())
{
    if (mUseTllmGen)
    {
        // The preprocessing kernel will convert Q from inputDataType to fp8 if the kv cache dtype is also e4m3.
        mQDataType = (mFixedParams.kvDataType == DATA_TYPE_E4M3) ? DATA_TYPE_E4M3 : mFixedParams.inputDataType;
        mTllmGenFMHARunner.reset(
            new TllmGenFmhaRunner(mQDataType, mFixedParams.kvDataType, mFixedParams.outputDataType));
    }
    else
    {
        mDecoderXqaRunner.reset(new DecoderXQARunner(mFixedParams.inputDataType, mFixedParams.numQHeads,
            mFixedParams.numKvHeads, mFixedParams.headSize, mFixedParams.multiBlockMode));
    }
}

void XqaDispatcher::prepare(XQAParams const& params)
{
    if (!mUseTllmGen)
    {
        if (mDecoderXqaRunner->shouldUse(params, /*forConfigurePlugin=*/true))
        {
            mDecoderXqaRunner->prepare(params);
        }
    }
}

int XqaDispatcher::getWorkspaceAlignment()
{
    constexpr size_t kXQA_OUT_ELEM_SIZE = 2; // fp16 or bf16.
    constexpr int kMaxBeamWidth = 4;
    int group_size = mFixedParams.numQHeads / mFixedParams.numKvHeads;
    int32_t const multi_block_workspace_alignment
        = roundUp<int32_t>(kXQA_OUT_ELEM_SIZE * kMaxBeamWidth * group_size * mFixedParams.headSize, 128);
    return mFixedParams.multiBlockMode ? multi_block_workspace_alignment : 128;
}

size_t XqaDispatcher::getWorkspaceSize(int max_num_tokens)
{
    // buffer for RoPE / output quantization.
    constexpr size_t kXQA_OUT_ELEM_SIZE = 2; // fp16 or bf16.
    constexpr int kMaxBeamWidth = 4;
    size_t workspace_size = roundUp<size_t>(
        kXQA_OUT_ELEM_SIZE * mFixedParams.headSize * mFixedParams.numQHeads * max_num_tokens, 128); // rope
    // output conversion.
    workspace_size = roundUp<size_t>(
        workspace_size + kXQA_OUT_ELEM_SIZE * mFixedParams.headSize * mFixedParams.numQHeads * max_num_tokens, 128);
    if (mFixedParams.multiBlockMode)
    {
        int workspaces[4];
        uint32_t const nbSubSeq = kXQA_MAX_NUM_SUB_SEQ;
        uint32_t const nbSeq = nbSubSeq / 2;
        int group_size = mFixedParams.numQHeads / mFixedParams.numKvHeads;
        workspaces[0] = sizeof(uint32_t) * nbSeq;                           // semaphores
        workspaces[1] = sizeof(float) * roundUp(group_size, 32) * nbSubSeq; // rowMax
        workspaces[2] = sizeof(float) * roundUp(group_size, 32) * nbSubSeq; // rowSum
        int32_t const multi_block_workspace_alignment
            = roundUp<int32_t>(kXQA_OUT_ELEM_SIZE * kMaxBeamWidth * group_size * mFixedParams.headSize, 128);
        workspaces[3] = multi_block_workspace_alignment * nbSubSeq;
        workspace_size = roundUp<size_t>(workspace_size, multi_block_workspace_alignment)
            + roundUp(workspaces[0], multi_block_workspace_alignment)
            + roundUp(workspaces[1], multi_block_workspace_alignment)
            + roundUp(workspaces[2], multi_block_workspace_alignment)
            + roundUp(workspaces[3], multi_block_workspace_alignment)
            + multi_block_workspace_alignment; // extra space reserved for alignment
    }
    return workspace_size;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

#define SHOULD_NOT_USE(message)                                                                                        \
    TLLM_LOG_DEBUG(message);                                                                                           \
    return false;

bool XqaDispatcher::shouldUse(XQAParams const& params)
{
    if (mUseTllmGen)
    {
        // Fall-back to MMHA for some unsupported cases.
        if (params.beam_width > 1)
        {
            SHOULD_NOT_USE("Fallback to MMHA as beam searching is not supported by TRTLLM-GEN kernels.")
        }
        if (params.position_shift_enabled || params.sink_token_length > 0)
        {
            SHOULD_NOT_USE("Fallback to MMHA as streamingLLM is not supported by TRTLLM-GEN kernels.")
        }
        if (params.unidirectional != 1)
        {
            SHOULD_NOT_USE("Fallback to MMHA as unidirectional is not supported by TRTLLM-GEN kernels.");
        }
        if (params.cross_attention)
        {
            SHOULD_NOT_USE("Fallback to MMHA as cross attention is not supported by TRTLLM-GEN kernels.");
        }
        if (params.paged_kv_cache && params.tokens_per_block < 8)
        {
            SHOULD_NOT_USE("Fallback to MMHA as tokens_per_block < 8 is not supported by TRTLLM-GEN kernels.");
        }
        if (params.cyclic_attention_window_size != params.max_attention_window_size)
        {
            SHOULD_NOT_USE(
                "Fallback to MMHA as variable attention_window_size is not supported by TRTLLM-GEN kernels.");
        }

        return true;
    }
    return mDecoderXqaRunner->shouldUse(params, /*forConfigurePlugin=*/false);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool XqaDispatcher::isSupported()
{
    if (mUseTllmGen)
    {
        // TODO (perkzz): add the support of fp8-kv fp16/bf16-mma fmha.
        if ((mFixedParams.kvDataType != mFixedParams.mathDataType) || (mQDataType != mFixedParams.mathDataType))
        {
            TLLM_LOG_WARNING("Unsupported data type combination.");
            return false;
        }
        if (mFixedParams.isSpecDecoding)
        {
            TLLM_LOG_WARNING("TRTLLM-GEN does not support Speculative decoding.");
            return false;
        }
        if (mFixedParams.hasAlibi)
        {
            TLLM_LOG_WARNING("TRTLLM-GEN does not support ALiBi.");
            return false;
        }
        if (mFixedParams.cpSize > 1)
        {
            TLLM_LOG_WARNING("TRTLLM-GEN does not support CP.");
            return false;
        }

        // Create TllmGenFmhaRunnerParams.
        TllmGenFmhaRunnerParams tllmRunnerParams;
        memset(&tllmRunnerParams, 0, sizeof(tllmRunnerParams));
        tllmRunnerParams.mQkvLayout = mFixedParams.isPagedKv ? QkvLayout::PagedKv : QkvLayout::ContiguousKv;
        tllmRunnerParams.mMaskType = TrtllmGenAttentionMaskType::Dense;
        tllmRunnerParams.mKernelType = FmhaKernelType::Generation;
        tllmRunnerParams.mTileScheduler = TileScheduler::Static;
        tllmRunnerParams.mMultiCtasKvMode = true;
        tllmRunnerParams.mHeadDim = mFixedParams.headSize;
        tllmRunnerParams.mNumHeadsQ = mFixedParams.numQHeads;
        tllmRunnerParams.mNumHeadsKv = mFixedParams.numKvHeads;
        tllmRunnerParams.mNumHeadsQPerKv = mFixedParams.numQHeads / mFixedParams.numKvHeads;
        tllmRunnerParams.mNumTokensPerPage = mFixedParams.numTokensPerBlock;

        // Check if it is supported or not.
        auto [isSupported, info] = mTllmGenFMHARunner->isSupportedWithInfo(tllmRunnerParams);
        if (!isSupported)
        {
            TLLM_LOG_WARNING("TRTLLLM-Gen kernels are not selected: " + info);
            return false;
        }
    }
    else
    {
        if (!(mFixedParams.inputDataType == DATA_TYPE_FP16 || mFixedParams.inputDataType == DATA_TYPE_BF16))
        {
            TLLM_LOG_WARNING("Unsupported datatype.");
            return false;
        }
    }
    return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T, typename KVCacheBuffer>
void XqaDispatcher::runImpl(XQAParams params, KVCacheBuffer const& kv_cache_buffer)
{
    if (mUseTllmGen)
    {
        TLLM_LOG_DEBUG("Running TRTLLM-GEN generation kernel.");

        int num_q_heads = params.num_q_heads;
        int num_kv_heads = params.num_kv_heads;
        TLLM_CHECK_WITH_INFO(num_q_heads % num_kv_heads == 0, "numQHeads should be multiple of numKVHeads.");
        unsigned int num_q_heads_over_kv = num_q_heads / num_kv_heads;
        unsigned int beam_width = params.beam_width;
        unsigned int batch_beam_size = params.batch_size * beam_width;

        const KvCacheDataType cache_type = params.kv_cache_quant_mode.hasInt8KvCache()
            ? KvCacheDataType::INT8
            : (params.kv_cache_quant_mode.hasFp8KvCache() ? KvCacheDataType::FP8 : KvCacheDataType::BASE);

        XQALaunchParam<KVCacheBuffer> launchParams;
        void* inputScratch = nullptr;
        buildXQALaunchParams(launchParams, inputScratch, /*needOutputCvt*/ false, params, kv_cache_buffer);

        // Build cu_seqlens, padding_offset, and rotary inv freq tensors
        BuildDecoderInfoParams<T> decoder_params;
        memset(&decoder_params, 0, sizeof(decoder_params));
        decoder_params.seqQOffsets = launchParams.cu_seq_lens;
        decoder_params.seqKVOffsets = launchParams.cu_kv_seq_lens;
        decoder_params.seqQLengths = params.spec_decoding_generation_lengths;
        decoder_params.seqKVLengths = params.sequence_lengths;
        decoder_params.batchSize = int(batch_beam_size);
        decoder_params.maxQSeqLength = params.generation_input_length;
        decoder_params.numTokens = params.total_num_input_tokens;
        decoder_params.removePadding = true;
        TLLM_CHECK_WITH_INFO(!params.multi_query_tokens || params.spec_decoding_generation_lengths != nullptr,
            "Spec_decoding_generation_lengths must be provided.");
        // Scales for FP8 FMHA.
        decoder_params.fmhaBmm1Scale = launchParams.bmm1_scale_ptr;
        decoder_params.fmhaBmm2Scale = launchParams.bmm2_scale_ptr;
        decoder_params.fmhaHostBmm1Scale = 1.0f / (sqrtf(params.head_size * 1.0f) * params.q_scaling);
        bool const is_fp8_q_input = (mQDataType == DATA_TYPE_E4M3);
        if (params.kv_cache_quant_mode.hasFp8KvCache())
        {
            decoder_params.dequantScaleQ = params.kv_scale_quant_orig;
            decoder_params.dequantScaleKv = params.kv_scale_quant_orig;
        }
        if (params.is_fp8_output)
        {
            decoder_params.quantScaleO = params.fp8_out_scale;
        }
        // Rotary embedding inv_freq buffer.
        decoder_params.rotaryEmbeddingScale = params.rotary_embedding_scale;
        decoder_params.rotaryEmbeddingBase = params.rotary_embedding_base;
        decoder_params.rotaryEmbeddingDim = params.rotary_embedding_dim;
        decoder_params.rotaryScalingType = params.rotary_embedding_scale_type;
        decoder_params.rotaryEmbeddingInvFreq = launchParams.rotary_inv_freq_buf;
        decoder_params.rotaryEmbeddingInvFreqCache = params.rotary_embedding_inv_freq_cache;
        decoder_params.rotaryEmbeddingMaxPositions = params.rotary_embedding_max_positions;

        // FIXME(perkzz): currently, this kernel is always needed as cu_kv_seq_lens is needed by trtllm-gen kernels.
        invokeBuildDecoderInfo(decoder_params, params.stream);
        sync_check_cuda_error();

        // IDEA: Store rotary_processed Q buffer to output buffer.
        // NOTE: MHA kernels should read kv cache that has already been appended with new tokens' kv cache.
        void* xqa_q_input_ptr = inputScratch;
        QKVPreprocessingParams<T, KVCacheBuffer> preprocessingParms{static_cast<T*>(const_cast<void*>(params.qkv)),
            nullptr, nullptr, static_cast<T*>(xqa_q_input_ptr), kv_cache_buffer,
            /* kv_cache_block_scales_buffer*/ KVCacheBuffer{}, static_cast<T const*>(params.qkv_bias),
            params.logn_scaling_ptr, /* tokens_info*/ nullptr, params.spec_decoding_generation_lengths,
            params.sequence_lengths,
            /* encoder_seqlens */ nullptr, params.multi_query_tokens ? launchParams.cu_seq_lens : nullptr,
            /* cu_kv_seqlens */ nullptr, launchParams.rotary_inv_freq_buf, nullptr, params.kv_scale_orig_quant,
            /* kv_cache_scale_factors */ nullptr, params.spec_decoding_position_offsets, (float2 const*) nullptr,
            params.mrope_position_deltas, int(batch_beam_size), params.generation_input_length,
            params.max_past_kv_length, params.cyclic_attention_window_size, params.sink_token_length,
            int(params.batch_size * beam_width * params.generation_input_length),
            /*remove_padding*/ true, /*cross_attention*/ false, params.num_q_heads, params.num_kv_heads,
            params.num_q_heads / params.num_kv_heads, params.head_size, params.rotary_embedding_dim,
            params.rotary_embedding_base, params.rotary_embedding_scale_type, params.rotary_embedding_scale,
            params.rotary_embedding_max_positions, params.position_embedding_type, params.position_shift_enabled,
            cache_type, /* separate_q_kv_output */ true,
            /* quantized_fp8_output */ is_fp8_q_input, /* generation_phase */ true, mMultiProcessorCount,
            params.rotary_vision_start, params.rotary_vision_length};

        invokeQKVPreprocessing<T, KVCacheBuffer>(preprocessingParms, params.stream);
        sync_check_cuda_error();

        // Build runner parameters.
        TllmGenFmhaRunnerParams tllmRunnerParams;
        memset(&tllmRunnerParams, 0, sizeof(tllmRunnerParams));

        // Parameters to select kernels.
        tllmRunnerParams.mMaskType = TrtllmGenAttentionMaskType::Dense;
        tllmRunnerParams.mKernelType = FmhaKernelType::Generation;
        // Note that the tileScheduler and multiCtasKvMode will be automatically tuned when launching the kernels.
        tllmRunnerParams.mTileScheduler = TileScheduler::Static;
        tllmRunnerParams.mMultiCtasKvMode = params.multi_block_mode;

        // Q buffer.
        tllmRunnerParams.qPtr = xqa_q_input_ptr;
        // KV buffer
        if constexpr (std::is_same_v<KVCacheBuffer, KVBlockArray>)
        {
            // Paged KV
            tllmRunnerParams.mQkvLayout = QkvLayout::PagedKv;
            tllmRunnerParams.kvPtr = kv_cache_buffer.mPrimaryPoolPtr;
            tllmRunnerParams.kvPageIdxPtr = reinterpret_cast<KVCacheIndex::UnderlyingType const*>(kv_cache_buffer.data);
            tllmRunnerParams.mMaxNumPagesPerSeqKv = kv_cache_buffer.mMaxBlocksPerSeq;
            tllmRunnerParams.mNumTokensPerPage = kv_cache_buffer.mTokensPerBlock;
        }
        else
        {
            static_assert(std::is_same_v<KVCacheBuffer, KVLinearBuffer>);
            // Contiguous KV
            tllmRunnerParams.mQkvLayout = QkvLayout::ContiguousKv;
            tllmRunnerParams.kvPtr = kv_cache_buffer.data;
        }

        // The partial buffers' pointers when the multiCtasKv mode is enabled.
        tllmRunnerParams.multiCtasKvCounterPtr = launchParams.semaphores;
        tllmRunnerParams.multiCtasKvScratchPtr = launchParams.scratch;

        tllmRunnerParams.cumSeqLensKvPtr = reinterpret_cast<int const*>(launchParams.cu_kv_seq_lens);
        tllmRunnerParams.outputScalePtr = reinterpret_cast<float const*>(launchParams.bmm2_scale_ptr);
        // TRTLLM-GEN kernels always use the Log2 scale
        tllmRunnerParams.scaleSoftmaxLog2Ptr
            = reinterpret_cast<float const*>(launchParams.bmm1_scale_ptr + kIdxScaleSoftmaxLog2Ptr);
        tllmRunnerParams.oSfScalePtr = params.fp4_out_sf_scale;

        tllmRunnerParams.oPtr = params.output;
        tllmRunnerParams.oSfPtr = params.output_sf;
        tllmRunnerParams.mHeadDim = params.head_size;
        tllmRunnerParams.mNumHeadsQ = params.num_q_heads;
        tllmRunnerParams.mNumHeadsKv = params.num_kv_heads;
        tllmRunnerParams.mNumHeadsQPerKv = num_q_heads_over_kv;
        tllmRunnerParams.mBatchSize = params.batch_size;
        // It is used to construct contiguous kv cache TMA descriptors.
        tllmRunnerParams.mMaxSeqLenCacheKv = params.max_attention_window_size;
        tllmRunnerParams.mMaxSeqLenQ = 1;
        tllmRunnerParams.mMaxSeqLenKv = std::min(params.cyclic_attention_window_size, params.max_past_kv_length);
        tllmRunnerParams.mSumOfSeqLensQ = int(params.batch_size * beam_width * tllmRunnerParams.mMaxSeqLenQ);
        // Not used in the generation kernels as contiguous_kv or paged_kv layouts are used.
        tllmRunnerParams.mSumOfSeqLensKv = int(params.batch_size * beam_width * tllmRunnerParams.mMaxSeqLenKv);
        tllmRunnerParams.mScaleQ = params.q_scaling;
        if constexpr (std::is_same_v<KVCacheBuffer, KVBlockArray>)
        {
            auto const [freeMemory, totalMemory] = tensorrt_llm::common::getDeviceMemoryInfo(false);
            tllmRunnerParams.mNumPagesInMemPool = totalMemory
                / (tllmRunnerParams.mNumHeadsKv * tllmRunnerParams.mNumTokensPerPage * tllmRunnerParams.mHeadDim
                    * get_size_in_bytes(mFixedParams.kvDataType));
        }
        tllmRunnerParams.mMaxNumCtas = mMultiProcessorCount;
        tllmRunnerParams.stream = params.stream;
        tllmRunnerParams.mSfStartTokenIdx = params.start_token_idx_sf;

        TLLM_CHECK_WITH_INFO(mTllmGenFMHARunner.get(), "mTllmGenFMHARunner not initialized.");
        mTllmGenFMHARunner->run(tllmRunnerParams);
    }
    else
    {
        mDecoderXqaRunner->template dispatch<KVCacheBuffer>(params, kv_cache_buffer, params.stream);
    }
}

void XqaDispatcher::run(XQAParams const& params, KVLinearBuffer const& kv_cache_buffer)
{
    TLLM_CHECK_WITH_INFO((mFixedParams.inputDataType == DATA_TYPE_FP16 || mFixedParams.inputDataType == DATA_TYPE_BF16),
        "The input Qkv tensor must be fp16/bf16.");
    if (mFixedParams.inputDataType == DATA_TYPE_FP16)
    {
        this->runImpl<__half, KVLinearBuffer>(params, kv_cache_buffer);
    }
    else
    {
        this->runImpl<__nv_bfloat16, KVLinearBuffer>(params, kv_cache_buffer);
    }
}

void XqaDispatcher::run(XQAParams const& params, KVBlockArray const& kv_cache_buffer)
{
    TLLM_CHECK_WITH_INFO((mFixedParams.inputDataType == DATA_TYPE_FP16 || mFixedParams.inputDataType == DATA_TYPE_BF16),
        "The input Qkv tensor must be fp16/bf16.");
    if (mFixedParams.inputDataType == DATA_TYPE_FP16)
    {
        this->runImpl<__half, KVBlockArray>(params, kv_cache_buffer);
    }
    else
    {
        this->runImpl<__nv_bfloat16, KVBlockArray>(params, kv_cache_buffer);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace tensorrt_llm::kernels
