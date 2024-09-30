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

#include "fmhaRunner.h"
#include "tensorrt_llm/common/mathUtils.h"
#include <cassert>
#include <cstring>
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
    TLLM_CHECK_WITH_INFO((mSM == kSM_70 || mSM == kSM_80 || mSM == kSM_86 || mSM == kSM_89 || mSM == kSM_90),
        "Unsupported architecture");
    TLLM_CHECK_WITH_INFO((mFixedParams.dataType == DATA_TYPE_FP16 || mFixedParams.dataType == DATA_TYPE_BF16
                             || mFixedParams.dataType == DATA_TYPE_E4M3),
        "Unsupported data type");
    xmmaKernel = getXMMAKernelsV2(mFixedParams.dataType, mSM);

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
    // Memset kernel params.
    memset(&mKernelParams, 0, sizeof(mKernelParams));

    // Set the batch size, and sequence length.
    mKernelParams.b = runnerParams.b;
    mKernelParams.s = runnerParams.qSeqLen;
    mKernelParams.sliding_window_size = runnerParams.slidingWindowSize;
    // Set the head size and number of heads.
    mKernelParams.d = mFixedParams.headSize;
    TLLM_CHECK_WITH_INFO(mFixedParams.numQHeads % mFixedParams.numKvHeads == 0,
        "number of Query heads should be multiple of KV heads !");
    mKernelParams.h = mFixedParams.numQHeads;
    mKernelParams.h_kv = mFixedParams.numKvHeads;
    mKernelParams.h_q_per_kv = mFixedParams.numQHeads / mFixedParams.numKvHeads;
    // Are the input sequences padded ?
    mKernelParams.is_s_padded = mFixedParams.isSPadded;

    // Packed QKV input layout.
    mKernelParams.qkv_stride_in_bytes = get_size_in_bytes(
        (mFixedParams.numQHeads + 2 * mFixedParams.numKvHeads) * mFixedParams.headSize, mFixedParams.dataType);
    // Contiguous Q input layout.
    mKernelParams.q_stride_in_bytes
        = get_size_in_bytes(mFixedParams.numQHeads * mFixedParams.headSize, mFixedParams.dataType);
    // Set the kv_stride_in_bytes when separate kv buffer is used.
    if (mFixedParams.attentionInputLayout == AttentionInputLayout::Q_PAGED_KV)
    {
        // Paged kv cache layout.
        mKernelParams.kv_stride_in_bytes = get_size_in_bytes(
            runnerParams.pagedKvCache.mTokensPerBlock * mFixedParams.headSize, mFixedParams.dataType);
    }
    else if (mFixedParams.attentionInputLayout == AttentionInputLayout::Q_CONTIGUOUS_KV)
    {
        // Contiguous kv input layout.
        mKernelParams.kv_stride_in_bytes = get_size_in_bytes(mFixedParams.headSize, mFixedParams.dataType);
    }
    // Set the output buffer stride in bytes.
    mKernelParams.o_stride_in_bytes
        = get_size_in_bytes(mFixedParams.numQHeads * mFixedParams.headSize, mFixedParams.dataType);
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
    // Fuse 1.0f / qk_tanh_scale into scale_bmm1.
    scale_bmm1 = mFixedParams.qkTanhScale != 0.f ? scale_bmm1 / mFixedParams.qkTanhScale : scale_bmm1;
    // The softmax output scale (not used).
    float const scale_softmax = 1.f;
    // FP8 FMHA kernels load the scale_bmm2 from the device memory.
    float const scale_bmm2 = 1.f;

    Data_type scale_type = mLaunchParams.force_fp32_acc ? DATA_TYPE_FP32 : mFixedParams.dataType;
    // Use exp2f optimization for warp-specialized ws kernels on Hopper.
    if (mLaunchParams.useBase2ExpTrick)
    {
        // The kernel adopts the log2f optimziation.
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
    // The tanh scale after bmm1 (always float32).
    mKernelParams.tanh_scale_bmm1 = mFixedParams.qkTanhScale;

    // alibi.
    if (mFixedParams.hasAlibi && mSM > kSM_70)
    {
        mKernelParams.has_alibi = true;
        mKernelParams.alibi_params = AlibiParams(
            mFixedParams.numQHeads, runnerParams.kvSeqLen, mFixedParams.tpSize, mFixedParams.tpRank, scale_after_alibi);
    }

    // Set device pointers.
    mKernelParams.qkv_ptr = runnerParams.qkvPtr;
    mKernelParams.q_ptr = runnerParams.qPtr;
    mKernelParams.kv_ptr = runnerParams.kvPtr;
    mKernelParams.o_ptr = runnerParams.outputPtr;
    if (mFixedParams.attentionMaskType == ContextAttentionMaskType::CUSTOM_MASK)
    {
        mKernelParams.packed_mask_ptr = runnerParams.packedMaskPtr;
        mKernelParams.cu_mask_rows = reinterpret_cast<int const*>(runnerParams.cuMaskRowsPtr);
    }
    mKernelParams.cu_q_seqlens = reinterpret_cast<int const*>(runnerParams.cuQSeqLenPtr);
    mKernelParams.tile_id_counter_ptr = reinterpret_cast<uint32_t*>(runnerParams.tileCounterPtr);
    // TRT doesn't support host scales. Use device scales instead.
    // The scaleBmm1Ptr offset.
    // 2 scales prepared for scaleBmm1 in the device memory: float scale, float (scale with log2e).
    int64_t scaleBmm1PtrOffset = (mLaunchParams.useBase2ExpTrick ? 1 : 0);
    // Only fp8 kernels need to load scales from the device memory.
    if (mFixedParams.dataType == DATA_TYPE_E4M3)
    {
        mKernelParams.scale_bmm1_d = reinterpret_cast<uint32_t const*>(runnerParams.scaleBmm1Ptr + scaleBmm1PtrOffset);
        mKernelParams.scale_bmm2_d = reinterpret_cast<uint32_t const*>(runnerParams.scaleBmm2Ptr);
    }

    // Separate q and kv buffers may have different q and kv sequence lengths.
    if (mFixedParams.attentionInputLayout != AttentionInputLayout::PACKED_QKV)
    {
        mKernelParams.cu_kv_seqlens = reinterpret_cast<int const*>(runnerParams.cuKvSeqLenPtr);
    }

    // Paged kv fmha.
    if (mFixedParams.attentionInputLayout == AttentionInputLayout::Q_PAGED_KV)
    {
        mKernelParams.paged_kv_cache = runnerParams.pagedKvCache.copyKVBlockArrayForContextFMHA();
    }
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

    // Do we use qkTanhScale ?
    TLLM_CHECK_WITH_INFO((mFixedParams.headSize == 128 || mFixedParams.headSize == 256) || !mFixedParams.qkTanhScale,
        "FMHA only supports head_size = 128 or 256 with QK Tanh Scale currently.");
    mLaunchParams.enableQKTanhScale = mFixedParams.qkTanhScale != 0.f;
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

    // Next power of 2 head size.
    TLLM_CHECK_WITH_INFO(mFixedParams.headSize > 0, "Head size should be greater than 0.");
    mLaunchParams.padded_d = (mFixedParams.headSize & (mFixedParams.headSize - 1)) == 0
        ? mFixedParams.headSize
        : pow(2, int(log2(mFixedParams.headSize)) + 1);

    bool const isSm70 = (mSM == kSM_70);
    bool const isSm90 = (mSM == kSM_90);
    bool const isSm8x = (mSM == kSM_86 || mSM == kSM_89);
    bool const isSm80 = (mSM == kSM_80);
    bool const isSm89 = (mSM == kSM_89);

    // Sliding_window_causal mask.
    if (runnerParams.kvSeqLen > runnerParams.slidingWindowSize
        && mLaunchParams.attention_mask_type == ContextAttentionMaskType::CAUSAL)
    {
        mLaunchParams.attention_mask_type = ContextAttentionMaskType::SLIDING_WINDOW_CAUSAL;
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
        mLaunchParams.flash_attention = true;
        mLaunchParams.force_unroll = true; // need more profile
    }
    // Hopper: fallback to original fmha_v2 when head_size <= 64 and seq_len <= 256
    // Only supports packed_qkv input + padding/causal mask.
    else if (isSm90 && !separateQKvInput && paddingOrCausalMask
        && (mFixedParams.headSize == 32 || mFixedParams.headSize == 64) && runnerParams.qSeqLen <= 256)
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
        if (isSm89 && mFixedParams.dataType == DATA_TYPE_E4M3)
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
        else if (isSm8x && mFixedParams.headSize < 256)
        {
            // flash attention tiled kernel is faster on Ada and Ampere derivatives when head_size>=256
            mLaunchParams.granular_tiling = false;
        }
        else if (isSm80 || isSm8x)
        {
            // otherwise, choose tiled kernel for Ampere/Ada
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
        //    - it doesn't work with scale * tanh(qk / scale) operation (from Grok).
        mLaunchParams.useBase2ExpTrick = !mLaunchParams.enableQKTanhScale;
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// TMA descriptors are used as grid_constant parameters (remove MemCpyH2D operations)
void FusedMHARunnerV2::setPackedQkvTmaDescriptors(MHARunnerParams runnerParams)
{
    // split D into multiple groups in order to match the TMA swizzle mode (128B)
    const uint32_t d_in_bytes = get_size_in_bytes(mLaunchParams.padded_d, mFixedParams.dataType);
    const uint32_t d_groups = d_in_bytes > 128 ? d_in_bytes / 128 : 1;

    // separate q, k, v and o tma descriptors
    Multiple_tma_descriptor<4> qkv_tma_descriptor;

    // tensor size
    uint32_t tensor_size_qkv[4];
    if (mKernelParams.h_kv < mKernelParams.h)
    {
        // if multi-query or grouped-query
        tensor_size_qkv[2] = 1;
        tensor_size_qkv[1] = (mKernelParams.h + 2 * mKernelParams.h_kv);
        tensor_size_qkv[0] = mKernelParams.d; // mKernelParams.d;
    }
    else
    {
        tensor_size_qkv[2] = 3;
        tensor_size_qkv[1] = mKernelParams.h;
        tensor_size_qkv[0] = mKernelParams.d; // mKernelParams.d;
    }

    // O : [TOTAL, 1, h, d]
    uint32_t tensor_size_o[4];
    tensor_size_o[0] = mKernelParams.d;
    tensor_size_o[1] = mKernelParams.h;
    tensor_size_o[2] = 1;

    // box size for k and v
    uint32_t box_size[4];
    // Update this on device?
    box_size[2] = 1;
    box_size[1] = 1;
    box_size[0] = mLaunchParams.padded_d / d_groups;

    // stride size in bytes. Assumes least significant dim is 1 (?)
    uint64_t tensor_stride_qkv[3];
    tensor_stride_qkv[0] = get_size_in_bytes(tensor_size_qkv[0], mFixedParams.dataType); // d
    tensor_stride_qkv[1] = tensor_size_qkv[1] * tensor_stride_qkv[0];                    // d*h
    tensor_stride_qkv[2] = tensor_size_qkv[2] * tensor_stride_qkv[1];                    // d*h*3

    uint64_t tensor_stride_o[3];
    tensor_stride_o[0] = get_size_in_bytes(tensor_size_o[0], mFixedParams.dataType); // d
    tensor_stride_o[1] = tensor_size_o[1] * tensor_stride_o[0];                      // d*h
    tensor_stride_o[2] = tensor_size_o[2] * tensor_stride_o[1];                      // d*h*1

    // traversal stride
    uint32_t traversal_stride_qkv[4] = {1, 1, 1, 1};
    uint32_t traversal_stride_o[4] = {1, 1, 1, 1};

    // OOB fill zeros
    uint32_t oob_fill = 0;

    // FP32 to TF32 conversion disabled
    uint32_t fp32_to_tf32 = 0;

    // gmma descriptor mode
    const uint32_t d_bytes_per_group = d_in_bytes / d_groups;
    const cudaTmaDescSwizzle swizzle_mode = (d_bytes_per_group > 64
            ? cudaTmaDescSwizzle::SWIZZLE_128B
            : (d_bytes_per_group > 32 ? cudaTmaDescSwizzle::SWIZZLE_64B : cudaTmaDescSwizzle::SWIZZLE_32B));

    uint32_t q_step = 0, kv_step = 0;
    xmmaKernel->getStepSize(q_step, kv_step, mKernelParams, mLaunchParams);

    // QKV [TOTAL, 3, h, d]
    // NOTE: we may need to use actual seqlen to set oob_value
    auto const* qkv_ptr = static_cast<char const*>(mKernelParams.qkv_ptr);
    tensor_size_qkv[3] = mLaunchParams.total_q_seqlen;
    // O [TOTAL, 1, h, d]
    auto* o_ptr = static_cast<char*>(mKernelParams.o_ptr);
    tensor_size_o[3] = mLaunchParams.total_q_seqlen;

    // Q: STEP_Q
    box_size[3] = q_step;
    // Desc Format (data type).
    cudaTmaDescFormat const desc_format
        = (get_size_in_bytes(mFixedParams.dataType) == 1) ? cudaTmaDescFormat::U8 : cudaTmaDescFormat::F16_RN;
    qkv_tma_descriptor.set_tma_desctriptor(qkv_ptr, desc_format, cudaTmaDescInterleave::INTERLEAVE_DISABLED,
        swizzle_mode, cudaTmaDescPromotion::PROMOTION_DISABLED, tensor_size_qkv, tensor_stride_qkv,
        traversal_stride_qkv, box_size, oob_fill, fp32_to_tf32, &mKernelParams.tma_desc_q);

    // K/V: STEP_KV
    box_size[3] = kv_step;
    qkv_tma_descriptor.set_tma_desctriptor(qkv_ptr, desc_format, cudaTmaDescInterleave::INTERLEAVE_DISABLED,
        swizzle_mode, cudaTmaDescPromotion::PROMOTION_DISABLED, tensor_size_qkv, tensor_stride_qkv,
        traversal_stride_qkv, box_size, oob_fill, fp32_to_tf32, &mKernelParams.tma_desc_kv);

    // O: 16
    // Note: sliding window causal kernel currently has reg spill when TMA store is enabled
    box_size[3] = 16;
    if ((get_size_in_bytes(mFixedParams.dataType) == 1)
        && mLaunchParams.attention_mask_type != ContextAttentionMaskType::SLIDING_WINDOW_CAUSAL)
    {
        qkv_tma_descriptor.set_tma_desctriptor(o_ptr, desc_format, cudaTmaDescInterleave::INTERLEAVE_DISABLED,
            swizzle_mode, cudaTmaDescPromotion::PROMOTION_DISABLED, tensor_size_o, tensor_stride_o, traversal_stride_o,
            box_size, oob_fill, fp32_to_tf32, &mKernelParams.tma_desc_o);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Contiguous in the shape of [B, S, H, D].
// Contiguous KV in the shape of [B, S, 2, H, D].
// Paged KV has [B, 2, NumBlocksPerSequence] buffers,
//  and each points to the contiguous buffer with shape [H, TokensPerBlock, D]
// TMA descriptors need cudaMemcpyAsync since we need multiple tma descriptors in device memory.
void FusedMHARunnerV2::setSeparateQKvTmaDescriptors(MHARunnerParams runnerParams)
{
    // split D into multiple groups in order to match the TMA swizzle mode (128B)
    const uint32_t d_in_bytes = get_size_in_bytes(mLaunchParams.padded_d, mFixedParams.dataType);
    const uint32_t d_groups = d_in_bytes > 128 ? d_in_bytes / 128 : 1;

    uint32_t q_step = 0, kv_step = 0;
    xmmaKernel->getStepSize(q_step, kv_step, mKernelParams, mLaunchParams);

    // Separate q, and paged kv tma descriptors.
    Multiple_tma_descriptor<4> qo_tma_descriptor;
    Multiple_tma_descriptor<4> kv_tma_descriptor;
    // Contiguous Q
    // query tensor size [B x S, 1, H, D]
    uint32_t tensor_size_qo[4];
    tensor_size_qo[3] = mLaunchParams.total_q_seqlen;
    tensor_size_qo[2] = 1;
    tensor_size_qo[1] = mKernelParams.h;
    tensor_size_qo[0] = mKernelParams.d;

    // box size for q and o
    uint32_t box_size_qo[4];
    box_size_qo[3] = q_step;
    box_size_qo[2] = 1;
    box_size_qo[1] = 1;
    box_size_qo[0] = mLaunchParams.padded_d / d_groups;

    // stride size in bytes.
    uint64_t tensor_stride_qo[3];
    tensor_stride_qo[0] = get_size_in_bytes(tensor_size_qo[0], mFixedParams.dataType);
    tensor_stride_qo[1] = tensor_size_qo[1] * tensor_stride_qo[0];
    tensor_stride_qo[2] = tensor_size_qo[2] * tensor_stride_qo[1];

    // traversal stride
    uint32_t traversal_stride[4] = {1, 1, 1, 1};

    // OOB fill zeros
    uint32_t oob_fill = 0;

    // FP32 to TF32 conversion disabled
    uint32_t fp32_to_tf32 = 0;

    // Desc Format (data type).
    cudaTmaDescFormat const desc_format
        = (get_size_in_bytes(mFixedParams.dataType) == 1) ? cudaTmaDescFormat::U8 : cudaTmaDescFormat::F16_RN;

    // gmma descriptor mode
    const uint32_t d_bytes_per_group = d_in_bytes / d_groups;
    cudaTmaDescSwizzle const swizzle_mode = (d_bytes_per_group > 64
            ? cudaTmaDescSwizzle::SWIZZLE_128B
            : (d_bytes_per_group > 32 ? cudaTmaDescSwizzle::SWIZZLE_64B : cudaTmaDescSwizzle::SWIZZLE_32B));

    // Q ptr.
    auto const* q_ptr = static_cast<char const*>(mKernelParams.q_ptr);

    // Q: STEP_Q.
    qo_tma_descriptor.set_tma_desctriptor(q_ptr, desc_format, cudaTmaDescInterleave::INTERLEAVE_DISABLED, swizzle_mode,
        cudaTmaDescPromotion::PROMOTION_DISABLED, tensor_size_qo, tensor_stride_qo, traversal_stride, box_size_qo,
        oob_fill, fp32_to_tf32, &mKernelParams.tma_desc_q);

    // O ptr.
    auto const* o_ptr = static_cast<char const*>(mKernelParams.o_ptr);

    // O: 16. Reuse
    box_size_qo[3] = 16;
    if ((get_size_in_bytes(mFixedParams.dataType) == 1)
        && mLaunchParams.attention_mask_type != ContextAttentionMaskType::SLIDING_WINDOW_CAUSAL)
    {
        qo_tma_descriptor.set_tma_desctriptor(o_ptr, desc_format, cudaTmaDescInterleave::INTERLEAVE_DISABLED,
            swizzle_mode, cudaTmaDescPromotion::PROMOTION_DISABLED, tensor_size_qo, tensor_stride_qo, traversal_stride,
            box_size_qo, oob_fill, fp32_to_tf32, &mKernelParams.tma_desc_o);
    }

    // Contiguous KV layout [B, S, 2, H, D].
    if (mFixedParams.attentionInputLayout == AttentionInputLayout::Q_CONTIGUOUS_KV)
    {
        // Per batch tensor size.
        uint32_t tensor_size_kv[4];
        // Maximum number of blocks in this device.
        tensor_size_kv[3] = mLaunchParams.total_kv_seqlen;
        tensor_size_kv[2] = 2;
        tensor_size_kv[1] = mKernelParams.h_kv;
        tensor_size_kv[0] = mKernelParams.d;

        // Box size for k and v.
        uint32_t box_size_kv[4];
        box_size_kv[3] = kv_step;
        box_size_kv[2] = 1;
        box_size_kv[1] = 1;
        box_size_kv[0] = mLaunchParams.padded_d / d_groups;

        // Stride size in bytes.
        uint64_t tensor_stride_kv[3];
        tensor_stride_kv[0] = get_size_in_bytes(tensor_size_kv[0], mFixedParams.dataType);
        tensor_stride_kv[1] = tensor_size_kv[1] * tensor_stride_kv[0];
        tensor_stride_kv[2] = tensor_size_kv[2] * tensor_stride_kv[1];

        // Set the paged_kv tma descriptor.
        kv_tma_descriptor.set_tma_desctriptor(runnerParams.kvPtr, desc_format,
            cudaTmaDescInterleave::INTERLEAVE_DISABLED, swizzle_mode, cudaTmaDescPromotion::PROMOTION_DISABLED,
            tensor_size_kv, tensor_stride_kv, traversal_stride, box_size_kv, oob_fill, fp32_to_tf32,
            &mKernelParams.tma_desc_kv);
    }
    else if (mFixedParams.attentionInputLayout == AttentionInputLayout::Q_PAGED_KV)
    {
        // Paged KV
        // Per batch tensor size.
        uint32_t tokens_per_block = uint32_t(mKernelParams.paged_kv_cache.mTokensPerBlock);
        uint32_t tensor_size_kv[4];
        // Maximum number of blocks in this device.
        tensor_size_kv[3] = mLaunchParams.total_device_memory / mKernelParams.paged_kv_cache.mBytesPerBlock;
        tensor_size_kv[2] = mKernelParams.h_kv;
        tensor_size_kv[1] = tokens_per_block;
        tensor_size_kv[0] = mKernelParams.d;

        // Box size for k and v.
        uint32_t box_size_kv[4];
        box_size_kv[3] = 1;
        box_size_kv[2] = 1;
        box_size_kv[1] = std::min(tokens_per_block, kv_step);
        box_size_kv[0] = mLaunchParams.padded_d / d_groups;

        TLLM_CHECK_WITH_INFO(
            tokens_per_block % 2 == 0, "FMHA with paged kv cache needs tokens_per_block to be power of 2 !");
        mKernelParams.blocks_per_tma_load = std::max(1, int32_t(kv_step / tokens_per_block));
        mKernelParams.blocks_per_tma_load_log2 = log2(mKernelParams.blocks_per_tma_load);

        // Stride size in bytes.
        uint64_t tensor_stride_kv[3];
        tensor_stride_kv[0] = get_size_in_bytes(tensor_size_kv[0], mFixedParams.dataType);
        tensor_stride_kv[1] = tensor_size_kv[1] * tensor_stride_kv[0];
        tensor_stride_kv[2] = tensor_size_kv[2] * tensor_stride_kv[1];

        // Set the paged_kv tma descriptor.
        kv_tma_descriptor.set_tma_desctriptor(runnerParams.pagedKvCache.mPrimaryPoolPtr, desc_format,
            cudaTmaDescInterleave::INTERLEAVE_DISABLED, swizzle_mode, cudaTmaDescPromotion::PROMOTION_DISABLED,
            tensor_size_kv, tensor_stride_kv, traversal_stride, box_size_kv, oob_fill, fp32_to_tf32,
            &mKernelParams.tma_desc_kv);
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
        switch (mFixedParams.attentionInputLayout)
        {
        case AttentionInputLayout::PACKED_QKV: setPackedQkvTmaDescriptors(runnerParams); break;
        case AttentionInputLayout::Q_CONTIGUOUS_KV:
        case AttentionInputLayout::Q_PAGED_KV: setSeparateQKvTmaDescriptors(runnerParams); break;
        default: TLLM_CHECK_WITH_INFO(false, "Unsupported attention input layout.");
        }
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
    bool foundKernels = xmmaKernel->checkIfKernelExist(mFixedParams);

    if (!foundKernels)
    {
        TLLM_LOG_WARNING("Fall back to unfused MHA for %s in sm_%d.", mFixedParams.convertToStrOutput().c_str(), mSM);
    }

    return foundKernels;
}

} // namespace kernels
} // namespace tensorrt_llm
