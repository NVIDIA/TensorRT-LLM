/*
 * Copyright (c) 2020-2025, NVIDIA CORPORATION.  All rights reserved.
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

#include "fmhaRunner.h"
#include "tensorrt_llm/common/envUtils.h"
#include "tensorrt_llm/common/mathUtils.h"
#include <cassert>
#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>
#include <iostream>
#include <math.h>
#include <tuple>
#include <vector>

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace tensorrt_llm
{
namespace kernels
{

////////////////////////////////////////////////////////////////////////////////////////////////////

union __half2_uint32_t_union
{
    half2 fp162;
    uint32_t u32;
};

union __float_uint32_t_union
{
    float fp32;
    uint32_t u32;
};

static inline void set_alpha(uint32_t& alpha, float norm, Data_type dtype)
{
    if (dtype == DATA_TYPE_FP16)
    {
        __half2_uint32_t_union temp;
        temp.fp162 = __float2half2_rn(norm);
        alpha = temp.u32;
    }
    else if (dtype == DATA_TYPE_FP32)
    {
        __float_uint32_t_union temp;
        temp.fp32 = norm;
        alpha = temp.u32;
    }
    else if (dtype == DATA_TYPE_INT32)
    {
        int32_t inorm = static_cast<int32_t>(norm);
        alpha = reinterpret_cast<uint32_t const&>(inorm);
    }
    else if (dtype == DATA_TYPE_BF16)
    {
        // TODO HACK!! BF16 Outputs are computed in FP32 for FP8.
        // This is because cublas does not allow current FP32 output.
        alpha = reinterpret_cast<uint32_t const&>(norm);
    }
    else
    {
        assert(false);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

FusedMHARunnerV2::FusedMHARunnerV2(MHARunnerFixedParams fixedParams)
    : mFixedParams(fixedParams)
{
    TLLM_CHECK_WITH_INFO((mSM == kSM_80 || mSM == kSM_86 || mSM == kSM_89 || mSM == kSM_90 || mSM == kSM_100
                             || mSM == kSM_103 || mSM == kSM_120 || mSM == kSM_121),
        "Unsupported architecture");
    TLLM_CHECK_WITH_INFO((mFixedParams.dataType == DATA_TYPE_FP16 || mFixedParams.dataType == DATA_TYPE_BF16
                             || mFixedParams.dataType == DATA_TYPE_E4M3),
        "Unsupported data type");
    xmmaKernel = getXMMAKernelsV2(mFixedParams.dataType, mFixedParams.dataTypeOut, mSM);

    if (mFixedParams.headSizeV == 0)
    {
        mFixedParams.headSizeV = mFixedParams.headSize;
    }
    // Get device attributes.
    int device_id;
    cudaGetDevice(&device_id);
    cudaDeviceGetAttribute(&mMultiProcessorCount, cudaDevAttrMultiProcessorCount, device_id);
    cudaDeviceGetAttribute(&mDeviceL2CacheSize, cudaDevAttrL2CacheSize, device_id);
    auto const [free_memory, total_memory] = tensorrt_llm::common::getDeviceMemoryInfo(false);
    mTotalDeviceMemory = total_memory;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Shared setup function.
void FusedMHARunnerV2::setupKernelParams(MHARunnerParams runnerParams)
{
    // Reinit kernel params.
    mKernelParams = {};

    // Set the batch size, and sequence length.
    mKernelParams.b = runnerParams.b;
    mKernelParams.s = runnerParams.qSeqLen;
    mKernelParams.sliding_window_size = runnerParams.slidingWindowSize;
    // Set the log chunked attention size if the chunked attention is used.
    if (mLaunchParams.attention_mask_type == ContextAttentionMaskType::SLIDING_OR_CHUNKED_CAUSAL
        && runnerParams.kvSeqLen > runnerParams.chunkedAttentionSize)
    {
        TLLM_CHECK_WITH_INFO((runnerParams.chunkedAttentionSize & (runnerParams.chunkedAttentionSize - 1)) == 0,
            "Chunked attention size should be a power of 2.");
        mKernelParams.log2_chunked_attention_size = std::log2(runnerParams.chunkedAttentionSize);
    }
    // Set the head size and number of heads.
    mKernelParams.d = mFixedParams.headSize;
    mKernelParams.dv = mFixedParams.headSizeV;
    // The number of grouped heads (only used by generation-phase MLA kernels) currently.
    mKernelParams.num_grouped_heads = runnerParams.numGroupedHeads;
    TLLM_CHECK_WITH_INFO(mFixedParams.numQHeads % mFixedParams.numKvHeads == 0,
        "number of Query heads should be multiple of KV heads !");
    mKernelParams.h = mFixedParams.numQHeads;
    mKernelParams.h_kv = mFixedParams.numKvHeads;
    mKernelParams.h_q_per_kv = mFixedParams.numQHeads / mFixedParams.numKvHeads;
    // Are the input sequences padded ?
    mKernelParams.is_s_padded = mFixedParams.isSPadded;

    // [total_q, h, 2] (max/sum)
    mKernelParams.softmax_stats_ptr = runnerParams.softmaxStatsPtr;
    mKernelParams.softmax_stats_stride_in_bytes = sizeof(float) * 2 * mFixedParams.numQHeads;

    if (mFixedParams.attentionInputLayout == AttentionInputLayout::PACKED_QKV)
    {
        // Packed QKV input layout, [B, S, H * D + H_kv * D + H_kv * Dv].
        mKernelParams.qkv_ptr = runnerParams.qkvPtr;
        mKernelParams.q_stride_in_bytes = mKernelParams.k_stride_in_bytes = mKernelParams.v_stride_in_bytes
            = get_size_in_bytes(mFixedParams.numQHeads * mFixedParams.headSize
                    + mFixedParams.numKvHeads * mFixedParams.headSize
                    + mFixedParams.numKvHeads * mFixedParams.headSizeV,
                mFixedParams.dataType);
    }
    else
    {
        // Contiguous Q input layout, [B, S, H, D].
        mKernelParams.q_ptr = runnerParams.qPtr;
        mKernelParams.q_stride_in_bytes
            = get_size_in_bytes(mFixedParams.numQHeads * mFixedParams.headSize, mFixedParams.dataType);

        // Separate q and kv buffers may have different q and kv sequence lengths.
        mKernelParams.cu_kv_seqlens = reinterpret_cast<int const*>(runnerParams.cuKvSeqLenPtr);

        if (mFixedParams.attentionInputLayout == AttentionInputLayout::Q_CONTIGUOUS_KV)
        {
            // Contiguous kv input layout, [B, S, H_kv * D + H_kv * Dv].
            mKernelParams.kv_ptr = runnerParams.kvPtr;
            mKernelParams.k_stride_in_bytes = mKernelParams.v_stride_in_bytes = get_size_in_bytes(
                mFixedParams.numKvHeads * (mFixedParams.headSize + mFixedParams.headSizeV), mFixedParams.dataType);
        }
        else if (mFixedParams.attentionInputLayout == AttentionInputLayout::Q_PAGED_KV)
        {
            // Paged kv cache layout.
            mKernelParams.paged_kv_cache = runnerParams.pagedKvCache.copyKVBlockArrayForContextFMHA();
            mKernelParams.k_stride_in_bytes = get_size_in_bytes(
                runnerParams.pagedKvCache.mTokensPerBlock * mFixedParams.headSize, mFixedParams.dataType);
            // If d == dv, then v_stride_in_bytes == k_stride_in_bytes.
            // For DeepSeek MLA, which is the only case where d != dv, V is padded to the sizeof K.
            // Thus, v_stride_in_bytes always equals to k_stride_in_bytes so far.
            mKernelParams.v_stride_in_bytes = mKernelParams.k_stride_in_bytes;
        }
        else if (mFixedParams.attentionInputLayout == AttentionInputLayout::SEPARATE_Q_K_V)
        {
            // Separate QKV input layout, [total_kv_seqlen, H_KV, D] + [total_kv_seqlen, H_KV, DV]
            TLLM_CHECK_WITH_INFO(runnerParams.kPtr != nullptr && runnerParams.vPtr != nullptr,
                "SEPARATE_Q_K_V requires valid K and V pointers.");
            mKernelParams.k_ptr = runnerParams.kPtr;
            mKernelParams.v_ptr = runnerParams.vPtr;
            // Tensor K is contiguous.
            mKernelParams.k_stride_in_bytes
                = get_size_in_bytes(mFixedParams.numKvHeads * mFixedParams.headSize, mFixedParams.dataType);
            if (mFixedParams.headSizeQkNope > 0 && mFixedParams.dataType != DATA_TYPE_E4M3)
            {
                // Non-FP8 context MLA: tensor V is not contiguous. The token stride is numKvHeads * (headSizeQkNope +
                // headSizeV).
                mKernelParams.v_stride_in_bytes = get_size_in_bytes(
                    mFixedParams.numKvHeads * (mFixedParams.headSizeQkNope + mFixedParams.headSizeV),
                    mFixedParams.dataType);
            }
            else
            {
                // Tensor V is contiguous for other cases.
                mKernelParams.v_stride_in_bytes
                    = get_size_in_bytes(mFixedParams.numKvHeads * mFixedParams.headSizeV, mFixedParams.dataType);
            }
        }
    }

    mKernelParams.o_ptr = runnerParams.outputPtr;
    // Set the output buffer stride in bytes.
    mKernelParams.o_stride_in_bytes
        = get_size_in_bytes(mFixedParams.numQHeads * mFixedParams.headSizeV, mFixedParams.dataTypeOut);
    // Set the packed_mask_stride_in_bytes.
    if (mFixedParams.attentionMaskType == ContextAttentionMaskType::CUSTOM_MASK)
    {
        // The packed mask col (n) dimension has to be padded to multiple of 256.
        mKernelParams.packed_mask_stride_in_bytes
            = (tensorrt_llm::common::divUp(int64_t(runnerParams.kvSeqLen), int64_t(FLASH_ATTEN_PACKED_MASK_N_ALIGNMENT))
                  * FLASH_ATTEN_PACKED_MASK_N_ALIGNMENT)
            / 8;
    }

    float const inv_sqrt_scale = (1.f / (sqrtf(mFixedParams.headSize) * mFixedParams.qScaling));
    // Note that we apply scales and bias in the order of
    // (bmm1_output * scale_bmm1 + alibi) * scale_after_alibi
    float const scale_after_alibi = mFixedParams.scaleAlibi ? inv_sqrt_scale : 1.0f;
    float scale_bmm1 = mFixedParams.scaleAlibi ? 1.0f : inv_sqrt_scale;
    // Fuse 1.0f / attn_logit_softcapping_scale into scale_bmm1.
    scale_bmm1 = mFixedParams.attnLogitSoftcappingScale != 0.f ? scale_bmm1 / mFixedParams.attnLogitSoftcappingScale
                                                               : scale_bmm1;
    // The softmax output scale (not used).
    float const scale_softmax = 1.f;
    // FP8 FMHA kernels load the scale_bmm2 from the device memory.
    float const scale_bmm2 = 1.f;

    Data_type scale_type = mLaunchParams.force_fp32_acc ? DATA_TYPE_FP32 : mFixedParams.dataType;
    // Use exp2f optimization for warp-specialized ws kernels on Hopper.
    if (mLaunchParams.useBase2ExpTrick)
    {
        // The kernel adopts the log2f optimization.
        constexpr float kLog2e = 1.4426950408889634074; // log_2(e) = M_LOG2E
        set_alpha(mKernelParams.scale_bmm1, scale_bmm1 * float(kLog2e), DATA_TYPE_FP32);
    }
    else
    {
        set_alpha(mKernelParams.scale_bmm1, scale_bmm1, scale_type);
    }
    set_alpha(mKernelParams.scale_softmax, scale_softmax, scale_type);
    // Host scale_bmm2 will not be used.
    set_alpha(mKernelParams.scale_bmm2, scale_bmm2, scale_type);
    // The attention logit softcapping scale after bmm1 (always float32).
    mKernelParams.softcapping_scale_bmm1 = mFixedParams.attnLogitSoftcappingScale;

    // alibi.
    if (mFixedParams.hasAlibi && mSM > kSM_70)
    {
        mKernelParams.has_alibi = true;
        mKernelParams.alibi_params = AlibiParams(
            mFixedParams.numQHeads, runnerParams.kvSeqLen, mFixedParams.tpSize, mFixedParams.tpRank, scale_after_alibi);
    }

    if (mFixedParams.attentionMaskType == ContextAttentionMaskType::CUSTOM_MASK)
    {
        mKernelParams.packed_mask_ptr = runnerParams.packedMaskPtr;
        mKernelParams.cu_mask_rows = reinterpret_cast<int const*>(runnerParams.cuMaskRowsPtr);
    }
    mKernelParams.attention_sinks_ptr = runnerParams.attentionSinksPtr;
    mKernelParams.cu_q_seqlens = reinterpret_cast<int const*>(runnerParams.cuQSeqLenPtr);
    mKernelParams.tile_id_counter_ptr = reinterpret_cast<uint32_t*>(runnerParams.tileCounterPtr);
    // TRT doesn't support host scales. Use device scales instead.
    // The scaleBmm1Ptr offset.
    // 2 scales prepared for scaleBmm1 in the device memory: float scale, float (scale with log2e).
    int64_t scaleBmm1PtrOffset = (mLaunchParams.useBase2ExpTrick ? kIdxScaleSoftmaxLog2Ptr : kIdxScaleSoftmaxPtr);
    // Only fp8 kernels need to load scales from the device memory.
    if (mFixedParams.dataType == DATA_TYPE_E4M3)
    {
        mKernelParams.scale_bmm1_d = reinterpret_cast<uint32_t const*>(runnerParams.scaleBmm1Ptr + scaleBmm1PtrOffset);
        mKernelParams.scale_bmm2_d = reinterpret_cast<uint32_t const*>(runnerParams.scaleBmm2Ptr);
    }

    // for sage attention
    mKernelParams.sage.q.scales = runnerParams.qScalePtr;
    mKernelParams.sage.k.scales = runnerParams.kScalePtr;
    mKernelParams.sage.v.scales = runnerParams.vScalePtr;
    mKernelParams.sage.q.max_nblock = runnerParams.qMaxNBlock;
    mKernelParams.sage.k.max_nblock = runnerParams.kMaxNBlock;
    mKernelParams.sage.v.max_nblock = runnerParams.vMaxNBlock;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Set the launch params to select kernels.
void FusedMHARunnerV2::setupLaunchParams(MHARunnerParams runnerParams)
{

    // Determine launch parameters.
    // Reset launch params to default.
    mLaunchParams = {};

    // Device properties.
    mLaunchParams.multi_processor_count = mMultiProcessorCount;
    mLaunchParams.device_l2_cache_size = mDeviceL2CacheSize;
    mLaunchParams.total_device_memory = mTotalDeviceMemory;

    // Do we use attnLogitSoftcappingScale ?
    TLLM_CHECK_WITH_INFO(
        (mFixedParams.headSize == 128 || mFixedParams.headSize == 256) || !mFixedParams.attnLogitSoftcappingScale,
        "FMHA only supports head_size = 128 or 256 with attention logit softcapping scale currently.");
    mLaunchParams.enableAttnLogitSoftcapping = mFixedParams.attnLogitSoftcappingScale != 0.f;
    // BF16 FMHA only accumulates on FP32.
    // E4M3 FMHA only supports fp32 accumulation currently.
    mLaunchParams.force_fp32_acc = mFixedParams.dataType == DATA_TYPE_BF16 || mFixedParams.dataType == DATA_TYPE_E4M3
        || mFixedParams.forceFp32Acc || runnerParams.forceFp32Acc;
    // The attention mask type.
    mLaunchParams.attention_mask_type = mFixedParams.attentionMaskType;
    // The input layout type.
    mLaunchParams.attention_input_layout = mFixedParams.attentionInputLayout;

    // The total sequence length used to set the tma descriptors.
    mLaunchParams.total_q_seqlen
        = mFixedParams.isSPadded ? runnerParams.b * runnerParams.qSeqLen : runnerParams.totalQSeqLen;
    mLaunchParams.total_kv_seqlen
        = mFixedParams.isSPadded ? runnerParams.b * runnerParams.kvSeqLen : runnerParams.totalKvSeqLen;
    // Workaround for nvbug 5412456: total_kv_seqlen fallbacks to total_q_seqlen if it's zero.
    if (mLaunchParams.total_kv_seqlen == 0)
    {
        mLaunchParams.total_kv_seqlen = mLaunchParams.total_q_seqlen;
    }

    TLLM_CHECK_WITH_INFO(mFixedParams.headSize > 0, "Head size should be greater than 0.");
    // Pad head size to next power of 2.
    int padded_d_next_power_of_2 = (mFixedParams.headSize & (mFixedParams.headSize - 1)) == 0
        ? mFixedParams.headSize
        : pow(2, int(log2(mFixedParams.headSize)) + 1);
    // In fact, due to 128B swizzle mode of TMA, only 128 bytes alignment is required,
    // so we pad head size to next multiply of 128B.
    int d_per_group = 128 / get_size_in_bytes(mFixedParams.dataType);
    int d_groups = (mFixedParams.headSize + d_per_group - 1) / d_per_group;
    int padded_d_next_multiply_of_128byte = d_groups * d_per_group;
    // Choose the smaller one to save SMEM.
    mLaunchParams.padded_d = std::min(padded_d_next_power_of_2, padded_d_next_multiply_of_128byte);

    bool const isSm70 = (mSM == kSM_70);
    bool const isSm90 = (mSM == kSM_90);
    bool const isSm8x = (mSM == kSM_86 || mSM == kSM_89);
    bool const isSm80 = (mSM == kSM_80);
    bool const isSm89 = (mSM == kSM_89);
    bool const isSm100f = (mSM == kSM_100 || mSM == kSM_103);
    bool const isSm120f = (mSM == kSM_120 || mSM == kSM_121);

    // Sliding_or_chunked_causal mask.
    if ((runnerParams.kvSeqLen > runnerParams.slidingWindowSize
            || runnerParams.kvSeqLen > runnerParams.chunkedAttentionSize)
        && mLaunchParams.attention_mask_type == ContextAttentionMaskType::CAUSAL)
    {
        TLLM_CHECK_WITH_INFO(!(runnerParams.kvSeqLen > runnerParams.chunkedAttentionSize
                                 && runnerParams.kvSeqLen > runnerParams.slidingWindowSize),
            "Chunked attention size and sliding window size should not be used together.");
        TLLM_CHECK_WITH_INFO(isSm90 || runnerParams.kvSeqLen <= runnerParams.chunkedAttentionSize,
            "Chunked attention is only supported on Sm90.");
        mLaunchParams.attention_mask_type = ContextAttentionMaskType::SLIDING_OR_CHUNKED_CAUSAL;
    }

    // Is the input layout separate q + kv input ?
    bool const separateQKvInput = mFixedParams.attentionInputLayout != AttentionInputLayout::PACKED_QKV;
    // Is the mask type padding or causal mask ?
    bool const paddingOrCausalMask = mFixedParams.attentionMaskType == ContextAttentionMaskType::PADDING
        || mFixedParams.attentionMaskType == ContextAttentionMaskType::CAUSAL;

    // Only warp-specialized FMHA kernels support FP8 on Hopper.
    // Separate Q + KV input layout: enable warp-specialization kernels when s > 512, otherwise use ampere-style flash
    // attention kernels.
    if (isSm90 && (mFixedParams.dataType == DATA_TYPE_E4M3 || (separateQKvInput && runnerParams.kvSeqLen > 512)))
    {
        mLaunchParams.flash_attention = true;
        mLaunchParams.force_unroll = true;
    }
    else if (isSm70)
    {
        TLLM_CHECK_WITH_INFO(false, "Unsupported architecture");
    }
    // Hopper: fallback to original fmha_v2 when head_size <= 64 and seq_len <= 256
    // Only supports packed_qkv input + padding/causal mask.
    else if (isSm90 && !separateQKvInput && paddingOrCausalMask
        && (mFixedParams.headSize == 32 || mFixedParams.headSize == 64) && runnerParams.qSeqLen <= 256
        && !common::getEnvForceDeterministicAttention())
    {
        mLaunchParams.flash_attention = false;
        // get max sequence length for non-flash-attention.
        // this doesn't support different q and kv sequence lengths.
        mLaunchParams.kernel_s = getSFromMaxSeqLen(runnerParams.qSeqLen);
    }
    else
    { // always use flash attention kernels for Ampere/Ada
        mLaunchParams.flash_attention = true;
        // flash attention kernles s = 0 (support any seq length)
        mLaunchParams.kernel_s = 0;
        mLaunchParams.force_unroll = true;
        // enable tiled kernels on Ampere/Ada
        if ((isSm89 || isSm120f) && mFixedParams.dataType == DATA_TYPE_E4M3)
        {
            // so far Ada QMMA only supports non-tiled kernels.
            mLaunchParams.granular_tiling = false;
        }
        else if (mLaunchParams.flash_attention && runnerParams.kvSeqLen <= 64)
        {
            // flash attention tiled kernels allows larger free dim tile size (M, N) with flexibility
            // in unroll dimension tile size (K). for short sequence length (s<=128), tiled kernels
            // can suffer from tile quantization loss therefore use flash attention non-tiled instead
            mLaunchParams.granular_tiling = false;
        }
        else if ((isSm8x || isSm120f) && mFixedParams.headSize < 256)
        {
            // flash attention tiled kernel is faster on Ada and Ampere derivatives when head_size>=256
            mLaunchParams.granular_tiling = false;
        }
        else if (isSm80 || isSm8x || isSm100f || isSm120f)
        {
            // otherwise, choose tiled kernel for Ampere/Ada/Gb20x
            mLaunchParams.granular_tiling = true;
        }
    }

    // when flash attention is enabled on Hopper, we need to set the tma descriptors
    if (isSm90 && mLaunchParams.flash_attention)
    {
        mLaunchParams.warp_specialization = true;
        mLaunchParams.use_tma = true;
        // Enable dynamic tile scheduling for hopper ws kernel.
        mLaunchParams.dynamic_scheduler = true;
    }

    // Use specialized ws kernels on Hopper for cases without alibi.
    if (mLaunchParams.warp_specialization && !mFixedParams.hasAlibi)
    {
        // Use specialized ws kernels for cases without alibi.
        mLaunchParams.useKernelWithoutAlibi = true;
        // Enable exp2f optimization (which helps improve performance).
        //    - note that this is not compatible with alibi bias due to the accuracy issues.
        //    - only hopper warp-specialized kernels have this optimization.
        //    - it doesn't work with attention logit softcapping.
        mLaunchParams.useBase2ExpTrick = !mLaunchParams.enableAttnLogitSoftcapping;
    }

    // TODO: Refactor these dirty hacks.
    // For Deepseek-v2(MLA), all of SM80, SM89 and SM90 kernels use tiled flash attention
    // in both context (192/128 dimensions) and generation (576/512 dimensions)
    if (mFixedParams.headSize == mFixedParams.headSizeV + 64)
    {
        mLaunchParams.flash_attention = true;
        mLaunchParams.force_unroll = true;
        mLaunchParams.kernel_s = 0;

        // Now we have SM90 context and FP8 generation MLA kernels
        bool isHopperContextMLA = isSm90 && mFixedParams.headSizeV == 128;
        bool isHopperFP8GenerationMLA
            = isSm90 && mFixedParams.dataType == DATA_TYPE_E4M3 && mFixedParams.headSizeV == 512;

        // These treatments are only for other MLA cases
        if (!isHopperContextMLA && !isHopperFP8GenerationMLA)
        {
            mLaunchParams.granular_tiling = true;
            // Even on SM90, we use ampere-style kernel, will be optimized later
            mLaunchParams.warp_specialization = false;
            mLaunchParams.useKernelWithoutAlibi = false;
            // Deepseek-V2 kernel is not hooper style right now.
            mLaunchParams.useBase2ExpTrick = false;
            mLaunchParams.use_tma = false;
            mLaunchParams.dynamic_scheduler = false;
        }
    }

    mLaunchParams.sage_block_size_q = mFixedParams.sageBlockSizeQ;
    mLaunchParams.sage_block_size_k = mFixedParams.sageBlockSizeK;
    mLaunchParams.sage_block_size_v = mFixedParams.sageBlockSizeV;
    // for not (sm90 + warp_specialization + flash attention kernel) kernel:
    //   all kernels enable saving softmaxStatsPtr, just let softmaxStatsPtr != null
    // for (sm90 + warp_specialization + flash attention) kernel:
    //   we need to explicitly set supportReturnSoftmaxStats to true when
    //  satisfying the following constrains
    if (!isSm90)
    {
        mLaunchParams.supportReturnSoftmaxStats = true;
    }
    else
    {
        bool isHopperContextMLA = (mFixedParams.headSize == mFixedParams.headSizeV + 64) && isSm90
            && (mFixedParams.dataType == DATA_TYPE_BF16 || mFixedParams.dataType == DATA_TYPE_E4M3)
            && mFixedParams.headSizeV == 128;
        mLaunchParams.supportReturnSoftmaxStats = (runnerParams.softmaxStatsPtr != nullptr
            && mLaunchParams.flash_attention && mLaunchParams.warp_specialization
            && ((!isHopperContextMLA && mLaunchParams.attention_input_layout == AttentionInputLayout::Q_CONTIGUOUS_KV)
                || (isHopperContextMLA
                    && (mLaunchParams.attention_input_layout == AttentionInputLayout::SEPARATE_Q_K_V))));
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// TMA descriptors are used as grid_constant parameters (remove MemCpyH2D operations)
void FusedMHARunnerV2::setTmaDescriptors(MHARunnerParams runnerParams)
{
    const uint32_t d = mKernelParams.d;
    const uint32_t dv = mKernelParams.dv;
    const uint32_t h = mKernelParams.h;
    const uint32_t h_kv = mKernelParams.h_kv;
    const uint32_t total_q_seqlen = mLaunchParams.total_q_seqlen;
    const uint32_t total_kv_seqlen = mLaunchParams.total_kv_seqlen;

    uint64_t const d_in_bytes = get_size_in_bytes(d, mFixedParams.dataType);
    uint64_t const dv_in_bytes = get_size_in_bytes(dv, mFixedParams.dataType);

    // split D into multiple groups in order to match the TMA swizzle mode (128B)
    uint32_t const padded_d_in_bytes = get_size_in_bytes(mLaunchParams.padded_d, mFixedParams.dataType);
    uint32_t const d_groups = padded_d_in_bytes > 128 ? padded_d_in_bytes / 128 : 1;
    uint32_t const d_bytes_per_group = padded_d_in_bytes / d_groups;
    uint32_t const d_per_group = mLaunchParams.padded_d / d_groups;

    uint32_t q_step = 0, kv_step = 0;
    xmmaKernel->getStepSize(q_step, kv_step, mKernelParams, mLaunchParams);

    auto const layout = mFixedParams.attentionInputLayout;

    // Q Layout: [total_seqlen, H, D]
    const uint32_t tensor_size_q[3] = {d, h, total_q_seqlen};

    // Stride size in bytes. Assumes least significant dim is 1
    const uint64_t tensor_stride_q[2] = {d_in_bytes, uint64_t(mKernelParams.q_stride_in_bytes)};

    // Starting memory address
    char const* q_ptr = reinterpret_cast<char const*>(
        layout == AttentionInputLayout::PACKED_QKV ? mKernelParams.qkv_ptr : mKernelParams.q_ptr);

    // Box size of TMA
    const uint32_t box_size_q[3] = {d_per_group, 1, q_step};

    // Traversal stride.
    const uint32_t traversal_stride[3] = {1, 1, 1};

    // OOB fill zeros.
    const uint32_t oob_fill = 0;

    // FP32 to TF32 conversion disabled.
    const uint32_t fp32_to_tf32 = 0;

    // GMMA descriptor mode.
    cudaTmaDescSwizzle const swizzle_mode = (d_bytes_per_group > 64
            ? cudaTmaDescSwizzle::SWIZZLE_128B
            : (d_bytes_per_group > 32 ? cudaTmaDescSwizzle::SWIZZLE_64B : cudaTmaDescSwizzle::SWIZZLE_32B));

    // Desc Format (data type).
    cudaTmaDescFormat const desc_format
        = (get_size_in_bytes(mFixedParams.dataType) == 1) ? cudaTmaDescFormat::U8 : cudaTmaDescFormat::F16_RN;

    Multiple_tma_descriptor<3> qo_tma_descriptor;

    // Q
    qo_tma_descriptor.set_tma_desctriptor(q_ptr, desc_format, cudaTmaDescInterleave::INTERLEAVE_DISABLED, swizzle_mode,
        cudaTmaDescPromotion::PROMOTION_DISABLED, tensor_size_q, tensor_stride_q, traversal_stride, box_size_q,
        oob_fill, fp32_to_tf32, &mKernelParams.tma_desc_q);

    // O
    if ((get_size_in_bytes(mFixedParams.dataTypeOut) == 1)
        && mLaunchParams.attention_mask_type != ContextAttentionMaskType::SLIDING_OR_CHUNKED_CAUSAL)
    {
        // O Layout: [total_seqlen, H, DV]
        const uint32_t tensor_size_o[3] = {dv, h, total_q_seqlen};

        const uint64_t tensor_stride_o[2]
            = {get_size_in_bytes(dv, mFixedParams.dataTypeOut), uint64_t(mKernelParams.o_stride_in_bytes)};

        char* o_ptr = reinterpret_cast<char*>(mKernelParams.o_ptr);

        // Box size of TMA
        const uint32_t box_size_o[3] = {d_per_group, 1, 16};

        // dataTypeOut may be different with dataType, so desc_format and swizzle_mode
        // may be incorrect. For example, QKV are in bf16 while O is in fp8.
        // Luckily, this case doesn't exist so far. But we should keep one eye on it.
        qo_tma_descriptor.set_tma_desctriptor(o_ptr, desc_format, cudaTmaDescInterleave::INTERLEAVE_DISABLED,
            swizzle_mode, cudaTmaDescPromotion::PROMOTION_DISABLED, tensor_size_o, tensor_stride_o, traversal_stride,
            box_size_o, oob_fill, fp32_to_tf32, &mKernelParams.tma_desc_o);
    }

    if (layout == AttentionInputLayout::Q_PAGED_KV)
    {
        // KV in q_paged_kv uses 4D tensor
        // Layout: [INT32_MAX, H_KV, TokensPerBlock, D]
        const uint32_t tokens_per_block = mKernelParams.paged_kv_cache.mTokensPerBlock;
        const uint32_t tensor_size_k[4] = {d, tokens_per_block, h_kv, INT_MAX};
        const uint32_t tensor_size_v[4] = {dv, tokens_per_block, h_kv, INT_MAX};

        const uint64_t tensor_stride_k[3] = {uint64_t(mKernelParams.k_stride_in_bytes / tokens_per_block), // d
            uint64_t(mKernelParams.k_stride_in_bytes),                                                     // d * 64
            uint64_t(mKernelParams.paged_kv_cache.mBytesPerBlock)};
        const uint64_t tensor_stride_v[3]
            = {// we cannot use dv * Kernel_traits::ELEMENT_BYTES because V may be padded (MLA)
                uint64_t(mKernelParams.v_stride_in_bytes / tokens_per_block), // dv
                uint64_t(mKernelParams.v_stride_in_bytes),                    // dv * 64
                uint64_t(mKernelParams.paged_kv_cache.mBytesPerBlock)};

        char const* kv_ptr = reinterpret_cast<char*>(runnerParams.pagedKvCache.mPrimaryPoolPtr);

        const uint32_t box_size_kv[4] = {d_per_group, std::min(tokens_per_block, kv_step), 1, 1};

        TLLM_CHECK(kv_step % tokens_per_block == 0 || tokens_per_block % kv_step == 0);
        mKernelParams.blocks_per_tma_load = std::max<uint32_t>(1, kv_step / tokens_per_block);
        mKernelParams.blocks_per_tma_load_log2 = log2(mKernelParams.blocks_per_tma_load);

        const uint32_t traversal_stride[4] = {1, 1, 1, 1};

        Multiple_tma_descriptor<4> kv_tma_descriptor;
        // K
        kv_tma_descriptor.set_tma_desctriptor(kv_ptr, desc_format, cudaTmaDescInterleave::INTERLEAVE_DISABLED,
            swizzle_mode, cudaTmaDescPromotion::PROMOTION_DISABLED, tensor_size_k, tensor_stride_k, traversal_stride,
            box_size_kv, oob_fill, fp32_to_tf32, &mKernelParams.tma_desc_k);
        // V
        kv_tma_descriptor.set_tma_desctriptor(kv_ptr, desc_format, cudaTmaDescInterleave::INTERLEAVE_DISABLED,
            swizzle_mode, cudaTmaDescPromotion::PROMOTION_DISABLED, tensor_size_v, tensor_stride_v, traversal_stride,
            box_size_kv, oob_fill, fp32_to_tf32, &mKernelParams.tma_desc_v);
    }
    else
    {
        // Otherwise KV uses 3D tensor
        const uint32_t tensor_size_k[3] = {d, h_kv, total_kv_seqlen};
        const uint32_t tensor_size_v[3] = {dv, h_kv, total_kv_seqlen};

        const uint64_t tensor_stride_k[2] = {d_in_bytes, uint64_t(mKernelParams.k_stride_in_bytes)};
        const uint64_t tensor_stride_v[2] = {dv_in_bytes, uint64_t(mKernelParams.v_stride_in_bytes)};

        const uint32_t box_size_kv[3] = {d_per_group, 1, kv_step};

        char const *k_ptr, *v_ptr;

        if (layout == AttentionInputLayout::PACKED_QKV)
        {
            // Layout: [total_seqlen, (H, D) + (H_KV, D) + (H_KV, DV)]
            k_ptr = q_ptr + h * d_in_bytes;
            v_ptr = k_ptr + h_kv * d_in_bytes;
        }
        else if (layout == AttentionInputLayout::Q_CONTIGUOUS_KV)
        {
            // Layout, [B, S, H_kv * D + H_kv * Dv].
            k_ptr = reinterpret_cast<char const*>(mKernelParams.kv_ptr);
            v_ptr = k_ptr + h_kv * d_in_bytes;
        }
        else if (layout == AttentionInputLayout::SEPARATE_Q_K_V)
        {
            // Layout: [total_kv_seqlen, H_KV, D] + [total_kv_seqlen, H_KV, DV]
            k_ptr = reinterpret_cast<char const*>(mKernelParams.k_ptr);
            v_ptr = reinterpret_cast<char const*>(mKernelParams.v_ptr);
        }

        Multiple_tma_descriptor<3> kv_tma_descriptor;
        // K
        kv_tma_descriptor.set_tma_desctriptor(k_ptr, desc_format, cudaTmaDescInterleave::INTERLEAVE_DISABLED,
            swizzle_mode, cudaTmaDescPromotion::PROMOTION_DISABLED, tensor_size_k, tensor_stride_k, traversal_stride,
            box_size_kv, oob_fill, fp32_to_tf32, &mKernelParams.tma_desc_k);
        // V
        kv_tma_descriptor.set_tma_desctriptor(v_ptr, desc_format, cudaTmaDescInterleave::INTERLEAVE_DISABLED,
            swizzle_mode, cudaTmaDescPromotion::PROMOTION_DISABLED, tensor_size_v, tensor_stride_v, traversal_stride,
            box_size_kv, oob_fill, fp32_to_tf32, &mKernelParams.tma_desc_v);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
void FusedMHARunnerV2::run(MHARunnerParams runnerParams)
{
    // Note that we must set the launch params first.
    // Set the launch params.
    setupLaunchParams(runnerParams);
    // Set the kernel params.
    setupKernelParams(runnerParams);
    // Need to set tma descriptors additionally.
    if (mSM == kSM_90 && mLaunchParams.use_tma)
    {
        setTmaDescriptors(runnerParams);
    }
    // Select the kernel and run it.
    xmmaKernel->run(mKernelParams, mLaunchParams, runnerParams.stream);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool FusedMHARunnerV2::isValidS(int s) const
{
    return xmmaKernel->isValid(s);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

int FusedMHARunnerV2::getSFromMaxSeqLen(int const max_seq_len) const
{
    int S = 1024;

    if (max_seq_len <= 64)
    {
        S = 64;
    }
    else if (max_seq_len <= 128)
    {
        S = 128;
    }
    else if (max_seq_len <= 256)
    {
        S = 256;
    }
    else if (max_seq_len <= 384)
    {
        S = 384;
    }
    else if (max_seq_len <= 512)
    {
        S = 512;
    }
    // for bert and vit, use flash attention when s >= 512
    else if (max_seq_len > 512)
    {
        S = max_seq_len;
    }

    return S;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Function to check if fmha is supported when building plugins.
// If any kernel in the map meets the requirements, then return true.
bool FusedMHARunnerV2::isFmhaSupported()
{
    bool is_supported = xmmaKernel->checkIfKernelExist(mFixedParams);
    if (!is_supported)
    {
        std::string msg = "FMHA Kernel doesn't exist for mFixedParams:\n" + mFixedParams.convertToStrOutput();
        TLLM_LOG_WARNING("%s\n", msg.c_str());
    }
    return is_supported;
}

} // namespace kernels
} // namespace tensorrt_llm
