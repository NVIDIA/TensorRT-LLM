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
#include "fused_multihead_attention_v2.h"

#include <cassert>
#include <cstring>
#include <iostream>
#include <math.h>
#include <tuple>
#include <vector>

namespace tensorrt_llm
{
namespace kernels
{

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

class FusedMHARunnerV2::mhaImpl
{
public:
    mhaImpl(const Data_type data_type, bool const pagedKVFMHA, int const numHeads, int const headSize,
        float const qScaling, float const qkTanhScale, int sm_)
        : mDataType(data_type)
        , mPagedKVFMHA(pagedKVFMHA)
        , mNumHeads(numHeads)
        , mHeadSize(headSize)
        , mQScaling(qScaling)
        , mQKTanhScale(qkTanhScale)
        , sm(sm_)
    {
        TLLM_CHECK_WITH_INFO(
            (sm == kSM_70 || sm == kSM_80 || sm == kSM_86 || sm == kSM_89 || sm == kSM_90), "Unsupported architecture");
        TLLM_CHECK_WITH_INFO(
            (mDataType == DATA_TYPE_FP16 || mDataType == DATA_TYPE_BF16 || mDataType == DATA_TYPE_E4M3),
            "Unsupported data type");
        TLLM_CHECK_WITH_INFO(
            mHeadSize == 128 || !mQKTanhScale, "FMHA only supports head_size = 128 with QK Tanh Scale currently.");

        xmmaKernel = getXMMAKernelsV2(mDataType, sm);

        mParams.clear();
        mPagedKVParams.clear();

        // get device attributes
        int device_id;
        cudaGetDevice(&device_id);
        cudaDeviceGetAttribute(&mLaunchParams.multi_processor_count, cudaDevAttrMultiProcessorCount, device_id);
        cudaDeviceGetAttribute(&mLaunchParams.device_l2_cache_size, cudaDevAttrL2CacheSize, device_id);
        auto const [free_memory, total_memory] = tensorrt_llm::common::getDeviceMemoryInfo(false);
        mLaunchParams.total_device_memory = total_memory;
    }

    ~mhaImpl() {}

    // Whether use paged kv fmha or not.
    bool use_paged_kv_fmha()
    {
        return mPagedKVFMHA;
    }

    // Shared setup function.
    template <typename Params>
    void setup_params(Params& params, int const b, int const s_q, int const s_kv, int const sliding_window_size,
        int const total_seqlen, bool const has_alibi, bool const scale_alibi, int const tp_size, int const tp_rank)
    {

        float const inv_sqrt_scale = (1.f / (sqrtf(mHeadSize) * mQScaling));
        // Note that we apply scales and bias in the order of
        // (bmm1_output * scale_bmm1 + alibi) * scale_after_alibi
        float const scale_after_alibi = scale_alibi ? inv_sqrt_scale : 1.0f;
        float const scale_bmm1 = scale_alibi ? 1.0f : inv_sqrt_scale;
        float const scale_softmax = 1.f; // Seems to be only required for int8
        float const scale_bmm2 = 1.f;

        Data_type scale_type = mLaunchParams.force_fp32_acc ? DATA_TYPE_FP32 : mDataType;
        // Use exp2f optimization for warp-specialized ws kernels on Hopper.
        if (mLaunchParams.useBase2ExpTrick)
        {
            // The kernel adopts the log2f optimziation.
            constexpr float kLog2e = 1.4426950408889634074; // log_2(e) = M_LOG2E
            set_alpha(params.scale_bmm1, scale_bmm1 * float(kLog2e), DATA_TYPE_FP32);
        }
        else
        {
            set_alpha(params.scale_bmm1, scale_bmm1, scale_type);
        }
        set_alpha(params.scale_softmax, scale_softmax, scale_type);
        // Host scale_bmm2 will not be used.
        set_alpha(params.scale_bmm2, scale_bmm2, scale_type);

        params.b = b;
        params.h = mNumHeads;
        params.s = s_q;
        params.d = mHeadSize;
        params.sliding_window_size = sliding_window_size;

        params.o_stride_in_bytes = get_size_in_bytes(mNumHeads * mHeadSize, mDataType);

        // Total sequence length needed by TMA descriptor
        // it should be actual total seq length if non-padded input is given.
        mTotalSeqLen = total_seqlen;

        // alibi.
        if (has_alibi && sm > kSM_70)
        {
            params.has_alibi = true;
            params.alibi_params = AlibiParams(mNumHeads, s_kv, tp_size, tp_rank, scale_after_alibi);
        }
    }

    // Support packed QKV.
    void setup(int const b, int const s, int const sliding_window_size, int const total_seqlen, bool const has_alibi,
        bool const scale_alibi, int const tp_size, int const tp_rank)
    {

        // Determine launch parameters.
        // Hopper: fallback to original fmha_v2 when head_size <= 64 and seq_len <= 256
        mLaunchParams.set_default_kernel_selection_params();

        // Grok tanh scale.
        // FIXME: mQKTanhScale value (30.f) is fixed in fmha kernels.
        mLaunchParams.enableQKTanhScale = mQKTanhScale > 0.f;

        // Next power of 2 head size.
        TLLM_CHECK_WITH_INFO(mHeadSize > 0, "Head size should be greater than 0.");
        mLaunchParams.padded_d = (mHeadSize & (mHeadSize - 1)) == 0 ? mHeadSize : pow(2, int(log2(mHeadSize)) + 1);

        bool const isSm70 = (sm == kSM_70);
        bool const isSm90 = (sm == kSM_90);
        bool const isSm8x = (sm == kSM_86 || sm == kSM_89);
        bool const isSm80 = (sm == kSM_80);

        // Only warp-specialized FMHA kernels support FP8 on Hopper.
        if (isSm90 && mDataType == DATA_TYPE_E4M3)
        {
            mLaunchParams.flash_attention = true;
            mLaunchParams.force_unroll = true;
        }
        else if (isSm70)
        {
            mLaunchParams.flash_attention = true;
            mLaunchParams.force_unroll = true; // need more profile
        }
        else if (isSm90 && (mHeadSize == 32 || mHeadSize == 64) && s <= 256)
        {
            mLaunchParams.flash_attention = false;
            // get max sequence length for non-flash-attentio
            mLaunchParams.kernel_s = getSFromMaxSeqLen(s);
        }
        else
        { // always use flash attention kernels for Ampere/Ada
            mLaunchParams.flash_attention = true;
            // flash attention kernles s = 0 (support any seq length)
            mLaunchParams.kernel_s = 0;
            mLaunchParams.force_unroll = true;
            // enable tiled kernels on Ampere/Ada
            if (mLaunchParams.flash_attention && s <= 64)
            {
                // flash attention tiled kernels allows larger free dim tile size (M, N) with flexibility
                // in unroll dimension tile size (K). for short sequence length (s<=128), tiled kernels
                // can suffer from tile quantization loss therefore use flash attention non-tiled instead
                mLaunchParams.granular_tiling = false;
            }
            else if (isSm8x && mHeadSize < 256)
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
        if (mLaunchParams.warp_specialization && !has_alibi)
        {
            // Use specialized ws kernels for cases without alibi.
            mLaunchParams.useKernelWithoutAlibi = true;
            // Enable exp2f optimization (which helps improve performance).
            //    - note that this is not compatible with alibi bias due to the accuracy issues.
            //    - only hopper warp-specialized kernels have this optimization.
            //    - it doesn't work with scale * tanh(qk / scale) operation (from Grok).
            mLaunchParams.useBase2ExpTrick = !mLaunchParams.enableQKTanhScale;
        }

        // Sliding_window_causal mask.
        if (s > sliding_window_size && mLaunchParams.attention_mask_type == ContextAttentionMaskType::CAUSAL)
        {
            mLaunchParams.attention_mask_type = ContextAttentionMaskType::SLIDING_WINDOW_CAUSAL;
        }

        // Set kernel parameters.
        setup_params(mParams, b, s, s, sliding_window_size, total_seqlen, has_alibi, scale_alibi, tp_size, tp_rank);
        mParams.qkv_stride_in_bytes = get_size_in_bytes((mNumHeads + 2 * mParams.h_kv) * mHeadSize, mDataType);
    }

    // Support paged_kv_cache and chunked_attention.
    void setup_paged_kv(int const b, int const s_q, int const s_kv, int const blocks_per_context_sequence,
        int const tokens_per_kv_block, int const sliding_window_size, int const total_seqlen, bool const has_alibi,
        bool const scale_alibi, int const tp_size, int const tp_rank)
    {

        // Determine launch parameters.
        mLaunchParams.set_default_kernel_selection_params();

        // Grok tanh scale.
        // FIXME: mQKTanhScale value (30.f) is fixed in fmha kernels.
        mLaunchParams.enableQKTanhScale = mQKTanhScale > 0.f;
        TLLM_CHECK_WITH_INFO(
            !mLaunchParams.enableQKTanhScale, "Paged KV FMHA doesn't support qk_tanh_scale operation.");

        // Needed by TMA descriptors.
        mLaunchParams.blocks_per_context_sequence = blocks_per_context_sequence;
        // Next power of 2 head size.
        TLLM_CHECK_WITH_INFO(mHeadSize > 0, "Head size should be greater than 0.");
        mLaunchParams.padded_d = (mHeadSize & (mHeadSize - 1)) == 0 ? mHeadSize : pow(2, int(log2(mHeadSize)) + 1);

        // Hopper: fallback to original fmha_v2 when head_size <= 64 and seq_len <= 256
        bool const isSm90 = (sm == kSM_90);
        bool const isSm70 = (sm == kSM_70);
        bool const isSm8x = (sm == kSM_86 || sm == kSM_89);
        bool const isSm80 = (sm == kSM_80);

        // always use flash attention kernels.
        mLaunchParams.flash_attention = true;
        // flash attention kernles s = 0 (support any seq length)
        mLaunchParams.kernel_s = 0;
        mLaunchParams.kernel_kv_s = s_kv;
        mLaunchParams.force_unroll = true;

        // only hopper warp-specialized FMHA kernels support FP8.
        // enable warp-specialization kernels when s > 512, otherwise use ampere-style flash attention kernels.
        if (isSm90 && (mDataType == DATA_TYPE_E4M3 || s_kv > 512))
        {
            mLaunchParams.warp_specialization = true;
            // Enable dynamic tile scheduling for hopper ws kernel.
            mLaunchParams.dynamic_scheduler = true;
            mLaunchParams.use_tma = true;
        }
        else if (isSm70)
        {
            mLaunchParams.flash_attention = true;
            mLaunchParams.force_unroll = true; // need more profile
        }
        else
        {
            // enable tiled kernels on Ampere/Ada
            if (mLaunchParams.flash_attention && s_kv <= 64)
            {
                // flash attention tiled kernels allows larger free dim tile size (M, N) with flexibility
                // in unroll dimension tile size (K). for short sequence length (s<=128), tiled kernels
                // can suffer from tile quantization loss therefore use flash attention non-tiled instead
                mLaunchParams.granular_tiling = false;
            }
            else if (isSm8x && mParams.d < 256)
            {
                // flash attention tiled kernel is faster on Ada and Ampere derivatives when head_size>=256
                mLaunchParams.granular_tiling = false;
            }
            else if (isSm90 || isSm80 || isSm8x)
            {
                // otherwise, choose tiled kernel for Ampere/Ada
                mLaunchParams.granular_tiling = true;
            }
        }

        // Use specialized ws kernels on Hopper for cases without alibi.
        if (mLaunchParams.warp_specialization && !has_alibi)
        {
            // Use specialized ws kernels for cases without alibi.
            mLaunchParams.useKernelWithoutAlibi = true;
            // Enable exp2f optimization (which helps improve performance).
            //    - note that this is not compatible with alibi bias due to the accuracy issues.
            //    - only hopper warp-specialized kernels have this optimization.
            mLaunchParams.useBase2ExpTrick = true;
        }

        // Sliding_window_causal mask.
        if (s_kv > sliding_window_size && mLaunchParams.attention_mask_type == ContextAttentionMaskType::CAUSAL)
        {
            mLaunchParams.attention_mask_type = ContextAttentionMaskType::SLIDING_WINDOW_CAUSAL;
        }

        // TODO: add paged kv FP8 FMHA.
        setup_params(
            mPagedKVParams, b, s_q, s_kv, sliding_window_size, total_seqlen, has_alibi, scale_alibi, tp_size, tp_rank);
        mPagedKVParams.q_stride_in_bytes = get_size_in_bytes(mNumHeads * mHeadSize, mDataType);
        mPagedKVParams.kv_stride_in_bytes = get_size_in_bytes(tokens_per_kv_block * mHeadSize, mDataType);
    }

    // NOTE: assume that heads_interleaved = false (b, s, 3, h, d), and sequences are padded/non-padded
    // TMA descriptors are used as grid_constant parameters (remove MemCpyH2D operations)
    void set_tma_descriptors()
    {
        // split D into multiple groups in order to match the TMA swizzle mode (128B)
        const uint32_t d_in_bytes = get_size_in_bytes(mLaunchParams.padded_d, mDataType);
        const uint32_t d_groups = d_in_bytes > 128 ? d_in_bytes / 128 : 1;

        // separate q, k, and v tma descriptors
        Multiple_tma_descriptor<4> qkv_tma_descriptor;

        // tensor size
        uint32_t tensor_size_qkv[4];
        if (mParams.h_kv < mParams.h)
        {
            // if multi-query or grouped-query
            tensor_size_qkv[2] = 1;
            tensor_size_qkv[1] = (mParams.h + 2 * mParams.h_kv);
            tensor_size_qkv[0] = mParams.d; // mParams.d;
        }
        else
        {
            tensor_size_qkv[2] = 3;
            tensor_size_qkv[1] = mParams.h;
            tensor_size_qkv[0] = mParams.d; // mParams.d;
        }

        // O : [TOTAL, 1, h, d]
        uint32_t tensor_size_o[4];
        tensor_size_o[0] = mParams.d;
        tensor_size_o[1] = mParams.h;
        tensor_size_o[2] = 1;

        // box size for k and v
        uint32_t box_size[4];
        // Update this on device?
        box_size[2] = 1;
        box_size[1] = 1;
        box_size[0] = mLaunchParams.padded_d / d_groups;

        // stride size in bytes. Assumes least significant dim is 1 (?)
        uint64_t tensor_stride_qkv[3];
        tensor_stride_qkv[0] = get_size_in_bytes(tensor_size_qkv[0], mDataType); // d
        tensor_stride_qkv[1] = tensor_size_qkv[1] * tensor_stride_qkv[0];        // d*h
        tensor_stride_qkv[2] = tensor_size_qkv[2] * tensor_stride_qkv[1];        // d*h*3

        uint64_t tensor_stride_o[3];
        tensor_stride_o[0] = get_size_in_bytes(tensor_size_o[0], mDataType); // d
        tensor_stride_o[1] = tensor_size_o[1] * tensor_stride_o[0];          // d*h
        tensor_stride_o[2] = tensor_size_o[2] * tensor_stride_o[1];          // d*h*1

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
        xmmaKernel->getStepSize(q_step, kv_step, mParams, mLaunchParams);

        // QKV [TOTAL, 3, h, d]
        // NOTE: we may need to use actual seqlen to set oob_value
        auto const* qkv_ptr = static_cast<char const*>(mParams.qkv_ptr);
        tensor_size_qkv[3] = mTotalSeqLen;
        // O [TOTAL, 1, h, d]
        auto* o_ptr = static_cast<char*>(mParams.o_ptr);
        tensor_size_o[3] = mTotalSeqLen;

        // Q: STEP_Q
        box_size[3] = q_step;
        // Desc Format (data type).
        cudaTmaDescFormat const desc_format
            = (get_size_in_bytes(mDataType) == 1) ? cudaTmaDescFormat::U8 : cudaTmaDescFormat::F16_RN;
        qkv_tma_descriptor.set_tma_desctriptor(qkv_ptr, desc_format, cudaTmaDescInterleave::INTERLEAVE_DISABLED,
            swizzle_mode, cudaTmaDescPromotion::PROMOTION_DISABLED, tensor_size_qkv, tensor_stride_qkv,
            traversal_stride_qkv, box_size, oob_fill, fp32_to_tf32, &mParams.tma_desc_q);

        // K/V: STEP_KV
        box_size[3] = kv_step;
        qkv_tma_descriptor.set_tma_desctriptor(qkv_ptr, desc_format, cudaTmaDescInterleave::INTERLEAVE_DISABLED,
            swizzle_mode, cudaTmaDescPromotion::PROMOTION_DISABLED, tensor_size_qkv, tensor_stride_qkv,
            traversal_stride_qkv, box_size, oob_fill, fp32_to_tf32, &mParams.tma_desc_k);
        qkv_tma_descriptor.set_tma_desctriptor(qkv_ptr, desc_format, cudaTmaDescInterleave::INTERLEAVE_DISABLED,
            swizzle_mode, cudaTmaDescPromotion::PROMOTION_DISABLED, tensor_size_qkv, tensor_stride_qkv,
            traversal_stride_qkv, box_size, oob_fill, fp32_to_tf32, &mParams.tma_desc_v);

        // O: 16
        // Note: sliding window causal kernel currently has reg spill when TMA store is enabled
        box_size[3] = 16;
        if ((get_size_in_bytes(mDataType) == 1)
            && mLaunchParams.attention_mask_type != ContextAttentionMaskType::SLIDING_WINDOW_CAUSAL)
        {
            qkv_tma_descriptor.set_tma_desctriptor(o_ptr, desc_format, cudaTmaDescInterleave::INTERLEAVE_DISABLED,
                swizzle_mode, cudaTmaDescPromotion::PROMOTION_DISABLED, tensor_size_o, tensor_stride_o,
                traversal_stride_o, box_size, oob_fill, fp32_to_tf32, &mParams.tma_desc_o);
        }
    }

    // Q are contiguous in the shape of [B, S, H, D]
    // Paged KV has [B, 2, NumBlocksPerSequence] buffers,
    //  and each points to the contiguous buffer with shape [H, TokensPerBlock, D]
    // TMA descriptors need cudaMemcpyAsync since we need multiple tma descriptors in device memory.
    void set_paged_kv_tma_descriptors(cudaStream_t stream)
    {
        // split D into multiple groups in order to match the TMA swizzle mode (128B)
        const uint32_t d_in_bytes = get_size_in_bytes(mLaunchParams.padded_d, mDataType);
        const uint32_t d_groups = d_in_bytes > 128 ? d_in_bytes / 128 : 1;

        uint32_t q_step = 0, kv_step = 0;
        xmmaKernel->getStepSize(q_step, kv_step, mPagedKVParams, mLaunchParams);

        // Separate q, and paged kv tma descriptors.
        Multiple_tma_descriptor<4> qo_tma_descriptor;
        Multiple_tma_descriptor<4> paged_kv_tma_descriptor;
        // mPagedKVParams.b * 2 * mLaunchParams.blocks_per_context_sequence
        // Contiguous Q
        // query tensor size [B x S, 1, H, D]
        uint32_t tensor_size_qo[4];
        tensor_size_qo[3] = mTotalSeqLen;
        tensor_size_qo[2] = 1;
        tensor_size_qo[1] = mPagedKVParams.h;
        tensor_size_qo[0] = mPagedKVParams.d;

        // box size for q and o
        uint32_t box_size_qo[4];
        box_size_qo[3] = q_step;
        box_size_qo[2] = 1;
        box_size_qo[1] = 1;
        box_size_qo[0] = mLaunchParams.padded_d / d_groups;

        // stride size in bytes.
        uint64_t tensor_stride_qo[3];
        tensor_stride_qo[0] = get_size_in_bytes(tensor_size_qo[0], mDataType);
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
            = (get_size_in_bytes(mDataType) == 1) ? cudaTmaDescFormat::U8 : cudaTmaDescFormat::F16_RN;

        // gmma descriptor mode
        const uint32_t d_bytes_per_group = d_in_bytes / d_groups;
        cudaTmaDescSwizzle const swizzle_mode = (d_bytes_per_group > 64
                ? cudaTmaDescSwizzle::SWIZZLE_128B
                : (d_bytes_per_group > 32 ? cudaTmaDescSwizzle::SWIZZLE_64B : cudaTmaDescSwizzle::SWIZZLE_32B));

        // Q ptr.
        auto const* q_ptr = static_cast<char const*>(mPagedKVParams.q_ptr);

        // Q: STEP_Q.
        qo_tma_descriptor.set_tma_desctriptor(q_ptr, desc_format, cudaTmaDescInterleave::INTERLEAVE_DISABLED,
            swizzle_mode, cudaTmaDescPromotion::PROMOTION_DISABLED, tensor_size_qo, tensor_stride_qo, traversal_stride,
            box_size_qo, oob_fill, fp32_to_tf32, &mPagedKVParams.tma_desc_q);

        // O ptr.
        auto const* o_ptr = static_cast<char const*>(mPagedKVParams.o_ptr);

        // O: 16. Reuse
        box_size_qo[3] = 16;
        if ((get_size_in_bytes(mDataType) == 1)
            && mLaunchParams.attention_mask_type != ContextAttentionMaskType::SLIDING_WINDOW_CAUSAL)
        {
            qo_tma_descriptor.set_tma_desctriptor(o_ptr, desc_format, cudaTmaDescInterleave::INTERLEAVE_DISABLED,
                swizzle_mode, cudaTmaDescPromotion::PROMOTION_DISABLED, tensor_size_qo, tensor_stride_qo,
                traversal_stride, box_size_qo, oob_fill, fp32_to_tf32, &mPagedKVParams.tma_desc_o);
        }

        // Paged KV
        // Per batch tensor size.
        uint32_t tokens_per_block = uint32_t(mPagedKVParams.paged_kv_cache.mTokensPerBlock);
        uint32_t tensor_size_kv[4];
        // Maximum number of blocks in this device.
        tensor_size_kv[3] = mLaunchParams.total_device_memory / mPagedKVParams.paged_kv_cache.mBytesPerBlock;
        tensor_size_kv[2] = mPagedKVParams.h_kv;
        tensor_size_kv[1] = tokens_per_block;
        tensor_size_kv[0] = mPagedKVParams.d;

        // Box size for k and v.
        uint32_t box_size_kv[4];
        box_size_kv[3] = 1;
        box_size_kv[2] = 1;
        box_size_kv[1] = std::min(tokens_per_block, kv_step);
        box_size_kv[0] = mLaunchParams.padded_d / d_groups;

        TLLM_CHECK_WITH_INFO(
            tokens_per_block % 2 == 0, "FMHA with paged kv cache needs tokens_per_block to be power of 2 !");
        mPagedKVParams.blocks_per_tma_load = std::max(1, int32_t(kv_step / tokens_per_block));
        mPagedKVParams.blocks_per_tma_load_log2 = log2(mPagedKVParams.blocks_per_tma_load);

        // Stride size in bytes.
        uint64_t tensor_stride_kv[3];
        tensor_stride_kv[0] = get_size_in_bytes(tensor_size_kv[0], mDataType);
        tensor_stride_kv[1] = tensor_size_kv[1] * tensor_stride_kv[0];
        tensor_stride_kv[2] = tensor_size_kv[2] * tensor_stride_kv[1];

        // 2 stands for k, and v blocks.
        TLLM_CHECK_WITH_INFO(
            mPagedKVParams.paged_kv_cache.mMaxBlocksPerSeq == mLaunchParams.blocks_per_context_sequence,
            "Mismatching blocks_per_sequence for the paged kv FMHA.");

        paged_kv_tma_descriptor.set_tma_desctriptor(mLaunchParams.paged_kv_pool_ptr, desc_format,
            cudaTmaDescInterleave::INTERLEAVE_DISABLED, swizzle_mode, cudaTmaDescPromotion::PROMOTION_DISABLED,
            tensor_size_kv, tensor_stride_kv, traversal_stride, box_size_kv, oob_fill, fp32_to_tf32,
            &mPagedKVParams.tma_desc_paged_kv);
    }

    void setup_flags(bool const force_fp32_acc, bool const is_s_padded, bool const causal_mask, int const num_kv_heads)
    {
        // BF16 FMHA only accumulates on FP32.
        // E4M3 FMHA only supports fp32 accumulation currently.
        mLaunchParams.force_fp32_acc = mDataType == DATA_TYPE_BF16 || mDataType == DATA_TYPE_E4M3 || force_fp32_acc;
        mLaunchParams.attention_mask_type
            = causal_mask ? ContextAttentionMaskType::CAUSAL : ContextAttentionMaskType::PADDING;

        // Paged KV Cache.
        mPagedKVParams.h_kv = num_kv_heads;
        TLLM_CHECK_WITH_INFO(mNumHeads % num_kv_heads == 0, "number of Query heads should be multiple of KV heads !");
        mPagedKVParams.h_q_per_kv = mNumHeads / num_kv_heads;
        mPagedKVParams.is_s_padded = is_s_padded;

        // Contiguous Cache.
        mParams.h_kv = num_kv_heads;
        mParams.h_q_per_kv = mNumHeads / num_kv_heads;
        mParams.is_s_padded = is_s_padded;
    }

    bool fmha_supported()
    {
        return MHARunner::fmha_supported(mHeadSize, sm);
    }

    void run(void const* qkvPtr, void const* cuSeqlenPtr, uint32_t* tileCounterPtr, float const* scaleBmm2Ptr,
        void* outputPtr, cudaStream_t stream)
    {
        mParams.qkv_ptr = qkvPtr;
        mParams.o_ptr = outputPtr;
        mParams.cu_seqlens = reinterpret_cast<int const*>(cuSeqlenPtr);
        mParams.tile_id_counter_ptr = tileCounterPtr;
        // TRT doesn't support host scales. Use device scales instead.
        mParams.scale_bmm2_d = reinterpret_cast<uint32_t const*>(scaleBmm2Ptr);
        mLaunchParams.paged_kv_input = false;

        if (sm == kSM_90 && mLaunchParams.use_tma)
        {
            set_tma_descriptors();
        }

        xmmaKernel->run(mParams, mLaunchParams, stream);
    }

    void run_paged_kv(void const* qPtr, void const* pagedKVBlockOffsetsOnHost, KVBlockArray const& pagedKVCache,
        void const* cuQSeqlenPtr, void const* cuKVSeqlenPtr, uint32_t* tileCounterPtr, float const* scaleBmm2Ptr,
        void* outputPtr, cudaStream_t stream)
    {
        mPagedKVParams.q_ptr = qPtr;
        mPagedKVParams.paged_kv_cache = pagedKVCache.copyKVBlockArrayForContextFMHA();
        mPagedKVParams.o_ptr = outputPtr;
        mPagedKVParams.cu_q_seqlens = reinterpret_cast<int const*>(cuQSeqlenPtr);
        mPagedKVParams.cu_seqlens = reinterpret_cast<int const*>(cuKVSeqlenPtr);
        mPagedKVParams.tile_id_counter_ptr = tileCounterPtr;
        // TRT doesn't support host scales. Use device scales instead.
        mPagedKVParams.scale_bmm2_d = reinterpret_cast<uint32_t const*>(scaleBmm2Ptr);
        // paged kv block device ptrs on host (used by tma descriptors).
        mLaunchParams.paged_kv_input = true;
        mLaunchParams.paged_kv_pool_ptr = pagedKVCache.mPrimaryPoolPtr;
        mLaunchParams.paged_kv_block_offsets
            = reinterpret_cast<decltype(mLaunchParams.paged_kv_block_offsets)>(pagedKVBlockOffsetsOnHost);

        if (sm == kSM_90 && mLaunchParams.use_tma)
        {
            set_paged_kv_tma_descriptors(stream);
        }

        xmmaKernel->run(mPagedKVParams, mLaunchParams, stream);
    }

    bool isValid(int s) const
    {
        return xmmaKernel->isValid(s);
    }

    int getSFromMaxSeqLen(int const max_seq_len)
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

private:
    Fused_multihead_attention_params_v2 mParams;
    Fused_multihead_attention_paged_kv_params_v2 mPagedKVParams;
    Launch_params mLaunchParams;
    int sm;
    FusedMultiHeadAttentionXMMAKernelV2 const* xmmaKernel;
    bool use_flash_attention = false;
    const Data_type mDataType;
    bool const mPagedKVFMHA;
    int const mNumHeads;
    int const mHeadSize;
    float const mQScaling;
    float const mQKTanhScale;
    int mTotalSeqLen;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

FusedMHARunnerV2::FusedMHARunnerV2(const Data_type data_type, bool const pagedKVFMHA, int const numHeads,
    int const headSize, float const qScaling, float const qkTanhScale)
    : pimpl(new mhaImpl(
        data_type, pagedKVFMHA, numHeads, headSize, qScaling, qkTanhScale, tensorrt_llm::common::getSMVersion()))
{
}

FusedMHARunnerV2::~FusedMHARunnerV2() = default;

void FusedMHARunnerV2::setup(int const b, int const s_q, int const s_kv, int const blocks_per_context_sequence,
    int const tokens_per_kv_block, int const sliding_window_size, int const total_seqlen, bool const has_alibi,
    bool const scale_alibi, int const tp_size, int const tp_rank)
{
    if (pimpl->use_paged_kv_fmha())
    {
        pimpl->setup_paged_kv(b, s_q, s_kv, blocks_per_context_sequence, tokens_per_kv_block, sliding_window_size,
            total_seqlen, has_alibi, scale_alibi, tp_size, tp_rank);
    }
    else
    {
        pimpl->setup(b, s_q, sliding_window_size, total_seqlen, has_alibi, scale_alibi, tp_size, tp_rank);
    }
}

bool FusedMHARunnerV2::fmha_supported()
{
    return pimpl->fmha_supported();
}

void FusedMHARunnerV2::setup_flags(
    bool const force_fp32_acc, bool const is_s_padded, bool const causal_mask, int const num_kv_heads)
{
    pimpl->setup_flags(force_fp32_acc, is_s_padded, causal_mask, num_kv_heads);
}

void FusedMHARunnerV2::run(void const* qPtr, void const* pagedKVBlockOffsetsOnHost, KVBlockArray const& pagedKVCache,
    void const* cuQSeqlenPtr, void const* cuKVSeqlenPtr, uint32_t* tileCounterPtr, float const* scaleBmm2Ptr,
    void* outputPtr, cudaStream_t stream)
{
    if (pimpl->use_paged_kv_fmha())
    {
        pimpl->run_paged_kv(qPtr, pagedKVBlockOffsetsOnHost, pagedKVCache, cuQSeqlenPtr, cuKVSeqlenPtr, tileCounterPtr,
            scaleBmm2Ptr, outputPtr, stream);
    }
    else
    {
        pimpl->run(qPtr, cuQSeqlenPtr, tileCounterPtr, scaleBmm2Ptr, outputPtr, stream);
    }
}

bool FusedMHARunnerV2::isValid(int s) const
{
    return pimpl->isValid(s);
}

// static function to check if fmha is supported when building plugins
bool MHARunner::fmha_supported(int const headSize, int const sm)
{
    // Check if the gpu architecture is supported or not.
    if (sm == 70 || sm == 80 || sm == 86 || sm == 89 || sm == 90)
    {
        // Check if the head size is supported or not.
        return (headSize == 32 || headSize == 40 || headSize == 64 || headSize == 80 || headSize == 96
            || headSize == 104 || headSize == 128 || headSize == 160 || headSize == 192 || headSize == 256);
    }
    // The gpu architecture is not supported.
    return false;
}

} // namespace kernels
} // namespace tensorrt_llm
