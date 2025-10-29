/*
 * SPDX-FileCopyrightText: Copyright (c) 2011-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: NVIDIA TensorRT Source Code License Agreement
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include <algorithm>
#include <float.h>
#include <fmha/hopper/tma_types.h>
#include <fmha/paged_kv_cache.h>
#include <fstream>
#include <fused_multihead_attention_api.h>
#include <iostream>
#include <math.h>
#include <numeric>
#include <string>
#include <vector>

using Launch_params = bert::Fused_multihead_attention_launch_params;
using Attention_mask_type = fmha::Attention_mask_type;
using Attention_input_layout = fmha::Attention_input_layout;
using Kv_block_array = fmha::Kv_block_array;

////////////////////////////////////////////////////////////////////////////////////////////////////

void run_softmax_fp32(void* dst, void const* src, void const* mask, void const* attention_sinks, void* softmax_sum_d,
    void* cu_q_seqlens_d, int s_inner, int s_outer, int b, int h, float softcapping_scale_bmm1, int warps_n,
    bool has_alibi);

////////////////////////////////////////////////////////////////////////////////////////////////////

void run_softmax_e4m3(void* dst, void const* src, void const* mask, void const* attention_sinks, void* softmax_sum_d,
    void* cu_q_seqlens_d, int s_inner, int s_outer, int b, int h, float scale_softmax, float softcapping_scale_bmm1,
    int warps_n, bool has_alibi);

////////////////////////////////////////////////////////////////////////////////////////////////////

void run_softmax_fp16(void* dst, void const* src, void const* mask, void const* attention_sinks, void* softmax_sum_d,
    void* cu_q_seqlens_d, int s_inner, int s_outer, int b, int h, float softcapping_scale_bmm1, int warps_n,
    bool has_alibi);

////////////////////////////////////////////////////////////////////////////////////////////////////

void run_softmax_bf16(void* dst, void const* src, void const* mask, void const* attention_sinks, void* softmax_sum_d,
    void* cu_q_seqlens_d, int s_inner, int s_outer, int b, int h, float softcapping_scale_bmm1, int warps_n,
    bool has_alibi);

////////////////////////////////////////////////////////////////////////////////////////////////////

void run_softmax_int8(void* dst, void const* src, void const* mask, void const* attention_sinks, void* softmax_sum_d,
    void* cu_q_seqlens_d, int s_inner, int s_outer, int b, int h, float scale_i2f, float scale_f2i,
    float softcapping_scale_bmm1, int warps_n, bool has_alibi);

////////////////////////////////////////////////////////////////////////////////////////////////////

void run_conversion_int32_to_int8(void* dst, void const* src, int s, int b, int h, int d, float scale);

////////////////////////////////////////////////////////////////////////////////////////////////////

void run_conversion_fp32_to_fp16(void* dst, void const* src, int s, int b, int h, int d);

////////////////////////////////////////////////////////////////////////////////////////////////////

void run_conversion_fp32_to_bf16(void* dst, void const* src, int s, int b, int h, int d);

////////////////////////////////////////////////////////////////////////////////////////////////////

void run_conversion_fp32_to_e4m3(void* dst, void const* src, int s, int b, int h, int d, float scale_o);

////////////////////////////////////////////////////////////////////////////////////////////////////

void run_sage_quant(unsigned int batch_size, unsigned int head_num, unsigned int head_size, unsigned int max_seq_len,
    // device var
    void const* q, void const* k, void const* v, int stride_q, int stride_k, int stride_v, int const* cu_seqlens_q,
    int const* cu_seqlens_kv, int block_size_q, int block_size_k, int block_size_v,
    // output
    void* quant_q, void* quant_k, void* quant_v, float* scales_q, float* scales_k, float* scales_v);

////////////////////////////////////////////////////////////////////////////////////////////////////

void ground_truth(RefBMM& bmm1, RefBMM& bmm2, const Data_type data_type, const Data_type acc_type,
    float const scale_bmm1, float const scale_softmax, float const scale_bmm2, float const softcapping_scale_bmm1,
    void* qkv_d, void* vt_d, void* mask_d, void* attention_sinks_d, void* p_d, void* s_d, void* tmp_d, void* o_d,
    void* softmax_sum_d, void* cu_q_seqlens_d, const size_t b, const size_t s, const size_t h, const size_t d,
    const size_t dv, int const runs, int const warps_m, int const warps_n, bool const has_alibi)
{

    cudaStream_t stream = 0;
    // The stride between rows of the QKV matrix.
    size_t qkv_stride = get_size_in_bytes(d, data_type);

    // 1st GEMMd.
    uint32_t alpha, beta = 0u;

    for (int ii = 0; ii < runs; ++ii)
    {

        // If we run the INT8 kernel, defer the scaling of P to softmax.
        set_alpha(alpha, data_type == DATA_TYPE_INT8 ? 1.f : scale_bmm1, acc_type);
        // P = Q x K'
        bmm1(static_cast<char*>(qkv_d) + 0 * qkv_stride, static_cast<char*>(qkv_d) + 1 * qkv_stride, p_d, &alpha, &beta,
            stream);
        // Softmax.
        if (data_type == DATA_TYPE_FP16 && acc_type == DATA_TYPE_FP16)
        {
            run_softmax_fp16(s_d, p_d, mask_d, attention_sinks_d, softmax_sum_d, cu_q_seqlens_d, s, s, b, h,
                softcapping_scale_bmm1, warps_n, has_alibi);
        }
        else if (data_type == DATA_TYPE_BF16 && acc_type == DATA_TYPE_FP32)
        {
            run_softmax_bf16(s_d, p_d, mask_d, attention_sinks_d, softmax_sum_d, cu_q_seqlens_d, s, s, b, h,
                softcapping_scale_bmm1, warps_n, has_alibi);
        }
        else if (data_type == DATA_TYPE_FP16 && acc_type == DATA_TYPE_FP32)
        {
            run_softmax_fp32(s_d, p_d, mask_d, attention_sinks_d, softmax_sum_d, cu_q_seqlens_d, s, s, b, h,
                softcapping_scale_bmm1, warps_n, has_alibi);
        }
        else if (data_type == DATA_TYPE_E4M3 && acc_type == DATA_TYPE_FP32)
        {
            run_softmax_e4m3(s_d, p_d, mask_d, attention_sinks_d, softmax_sum_d, cu_q_seqlens_d, s, s, b, h,
                scale_softmax, softcapping_scale_bmm1, warps_n, has_alibi);
        }
        else if (data_type == DATA_TYPE_INT8 && acc_type == DATA_TYPE_INT32)
        {
            run_softmax_int8(s_d, p_d, mask_d, attention_sinks_d, softmax_sum_d, cu_q_seqlens_d, s, s, b, h, scale_bmm1,
                scale_softmax, softcapping_scale_bmm1, warps_n, has_alibi);
        }
        else
        {
            assert(false && "Reference Softmax: Unsupported type config");
        }

        // 2nd GEMM.
        set_alpha(alpha, 1.f, acc_type);

        void* out_d = o_d;

        // We may have to do a final conversion.
        if (data_type != acc_type)
        {
            out_d = tmp_d;
        }
        // O = S x V
        bmm2(static_cast<char*>(s_d),
            static_cast<char*>(vt_d), // static_cast<char *>(qkv_d) + 2 * qkv_stride,
            out_d, &alpha, &beta, stream);
        // Conversion to output type.
        if (data_type == DATA_TYPE_FP16 && acc_type == DATA_TYPE_FP16)
        {
            // Noop.
        }
        else if (data_type == DATA_TYPE_FP16 && acc_type == DATA_TYPE_FP32)
        {
            run_conversion_fp32_to_fp16(o_d, out_d, s, b, h, dv);
        }
        else if (data_type == DATA_TYPE_BF16 && acc_type == DATA_TYPE_FP32)
        {
            run_conversion_fp32_to_bf16(o_d, out_d, s, b, h, dv);
        }
        else if (data_type == DATA_TYPE_E4M3 && acc_type == DATA_TYPE_FP32)
        {
            run_conversion_fp32_to_e4m3(o_d, out_d, s, b, h, dv, scale_bmm2);
        }
        else if (data_type == DATA_TYPE_INT8 && acc_type == DATA_TYPE_INT32)
        {
            // quantize output in second step
            run_conversion_int32_to_int8(o_d, out_d, s, b, h, dv, scale_bmm2);
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

static inline void set_params(bert::Fused_multihead_attention_params_v1& params,
    // types
    Data_type data_type, Data_type acc_type,
    // sizes
    const size_t b, const size_t s, const size_t h, const size_t d, const size_t packed_mask_stride,
    // device pointers
    void* qkv_d, void* packed_mask_d, void* o_d, void* p_d, void* s_d,
    // scale factors
    float const scale_bmm1, float const scale_softmax, float const scale_bmm2,
    // flags
    bool const has_alibi)
{
    memset(&params, 0, sizeof(params));

    // Set the pointers.
    params.qkv_ptr = qkv_d;
    params.qkv_stride_in_bytes = get_size_in_bytes(b * h * 3 * d, data_type);
    // params.qkv_stride_in_bytes = get_size_in_bytes(h * 3 * d, data_type);
    params.packed_mask_ptr = packed_mask_d;
    // params.packed_mask_stride_in_bytes = mmas_m * threads_per_cta * sizeof(uint32_t);
    params.packed_mask_stride_in_bytes = packed_mask_stride * sizeof(uint32_t);
    params.o_ptr = o_d;
    params.o_stride_in_bytes = get_size_in_bytes(b * h * d, data_type);
    params.has_alibi = has_alibi;
    params.alibi_params = fmha::AlibiParams(h);

#if defined(STORE_P)
    params.p_ptr = p_d;
    params.p_stride_in_bytes = get_size_in_bytes(b * h * s, acc_type);
#endif // defined(STORE_P)

#if defined(STORE_S)
    params.s_ptr = s_d;
    params.s_stride_in_bytes = get_size_in_bytes(b * h * s, data_type);
#endif // defined(STORE_S)

    // Set the dimensions.
    params.b = b;
    params.h = h;
    params.s = s;
    params.d = d;

    // Set the different scale values.
    Data_type scale_type1 = (data_type == DATA_TYPE_FP16) || (data_type == DATA_TYPE_BF16) ? acc_type : DATA_TYPE_FP32;
    Data_type scale_type2 = (data_type == DATA_TYPE_FP16) || (data_type == DATA_TYPE_BF16) ? data_type : DATA_TYPE_FP32;

    set_alpha(params.scale_bmm1, scale_bmm1, scale_type1);
    set_alpha(params.scale_softmax, scale_softmax, scale_type1);
    set_alpha(params.scale_bmm2, scale_bmm2, scale_type2);

    // Do we enable the trick to replace I2F with FP math in the 2nd GEMM?
    if (data_type == DATA_TYPE_INT8)
    {
        params.enable_i2f_trick
            = -double(1 << 22) * double(scale_bmm2) <= -128.f && double(1 << 22) * double(scale_bmm2) >= 127.f;
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

static inline void set_params(bert::Fused_multihead_attention_params_v2& params, const Launch_params launch_params,
    // types
    Data_type data_type, Data_type acc_type, Data_type output_dtype,
    // attention input layout
    Attention_input_layout input_layout,
    // sizes
    const size_t b, const size_t s_q, const size_t s_kv, const size_t h, const size_t h_kv, const size_t d,
    const size_t dv, const size_t total, const size_t num_grouped_heads, const size_t sliding_window_size,
    const size_t chunked_attention_size,
    // paged kv cache block size.
    const size_t tokens_per_block,
    // device pointers
    void* qkv_packed_d,
    // contiguous q.
    void* q_d,
    // separate k.
    void* k_d,
    // separate v.
    void* v_d,
    // contiguous kv.
    void* kv_d,
    // start address of the paged kv pool.
    void* paged_kv_pool_ptr,
    // offsets for different blocks in terms of the start address.
    int32_t* paged_block_offsets,
    // mask input.
    void* packed_mask_d, void* cu_mask_rows_d,
    // attention sinks.
    void* attention_sinks_d, void* cu_kv_seqlens_d, void* cu_q_seqlens_d, void* o_packed_d, void* p_d, void* s_d,
    void* softmax_stats_d, void* scale_bmm2_d,
    // scale factors
    float const scale_bmm1, float const scale_softmax, float const scale_bmm2, float const softcapping_scale_bmm1,
    // flags
    bool const use_int8_scale_max, bool const interleaved, bool const is_s_padded, bool const has_alibi)
{

    memset(&params, 0, sizeof(params));

    params.o_ptr = o_packed_d;
    params.o_stride_in_bytes = get_size_in_bytes(h * dv, output_dtype);

    if (interleaved)
    {
        params.q_stride_in_bytes = total;
        params.o_stride_in_bytes = total;
    }

    if (input_layout == Attention_input_layout::PACKED_QKV)
    {
        // For grouped- or multi-query attention (h denotes num_q_heads; h' denotes h_kv):
        //   qkv_layout = [b, s, [q_hd, k_h'd, v_h'd]]
        //   qkv_stride = (h+2*h')d * bytes_per_elt
        // Otherwise:
        //   qkv_layout = [b, s, 3, h, d] or [b, s, h, 3, d]
        //   qkv_stride = 3hd * bytes_per_elt
        params.qkv_ptr = qkv_packed_d;
        params.q_stride_in_bytes = params.k_stride_in_bytes = params.v_stride_in_bytes
            = get_size_in_bytes(h * d + h_kv * d + h_kv * dv, data_type);
    }
    else
    {
        // Layout [B, S, H, D].
        params.q_ptr = q_d;
        params.q_stride_in_bytes = get_size_in_bytes(h * d, data_type);

        if (input_layout == Attention_input_layout::CONTIGUOUS_Q_KV)
        {
            // Layout [B, S, 2, H, D].
            params.kv_ptr = kv_d;
            params.k_stride_in_bytes = params.v_stride_in_bytes = get_size_in_bytes(h_kv * (d + dv), data_type);
        }
        else if (input_layout == Attention_input_layout::Q_PAGED_KV)
        {
            int max_blocks_per_sequence = (s_kv + tokens_per_block - 1) / tokens_per_block;
            params.paged_kv_cache = Kv_block_array(b, max_blocks_per_sequence, tokens_per_block,
                get_size_in_bytes(tokens_per_block * h_kv * std::gcd(d, dv), data_type), paged_kv_pool_ptr);
            params.paged_kv_cache.mBlockOffsets = paged_block_offsets;
            params.k_stride_in_bytes = get_size_in_bytes(tokens_per_block * d, data_type);
            params.v_stride_in_bytes = get_size_in_bytes(tokens_per_block * dv, data_type);
        }
        else if (input_layout == Attention_input_layout::SEPARATE_Q_K_V)
        {
            // Layout [B, S, H_kv, D].
            params.k_ptr = k_d;
            // Layout [B, S, H_kv, Dv].
            params.v_ptr = v_d;
            params.k_stride_in_bytes = get_size_in_bytes(h_kv * d, data_type);
            params.v_stride_in_bytes = get_size_in_bytes(h_kv * dv, data_type);
        }
    }

    // Packed mask.
    params.packed_mask_ptr = packed_mask_d;
    // The N dimension has to be aligned.
    params.packed_mask_stride_in_bytes = (align_to(int64_t(s_kv), int64_t(fmha::FLASH_ATTEN_MASK_N_ALIGNMENT))) / 8;

    // Attention sinks.
    params.attention_sinks = reinterpret_cast<float*>(attention_sinks_d);

#if defined(STORE_P)
    params.p_ptr = p_d;
    params.p_stride_in_bytes = get_size_in_bytes(b * h * s_kv, acc_type);
#endif // defined(STORE_P)

#if defined(STORE_S)
    params.s_ptr = s_d;
    params.s_stride_in_bytes = get_size_in_bytes(b * h * s_kv, data_type);
#endif // defined(STORE_S)

    params.softmax_stats_ptr = softmax_stats_d;
    params.softmax_stats_stride_in_bytes = get_size_in_bytes(h * 2, DATA_TYPE_FP32);

    // Set the dimensions.
    params.b = b;
    params.h = h;
    params.s = s_q;
    params.d = d;
    params.dv = dv;
    params.num_grouped_heads = num_grouped_heads;
    params.sliding_window_size = sliding_window_size;
    assert((chunked_attention_size == 0 || (chunked_attention_size & (chunked_attention_size - 1)) == 0)
        && "chunked_attention_size has to be a power of 2");
    params.log2_chunked_attention_size = chunked_attention_size > 0 ? std::log2(chunked_attention_size) : 0;

    // cumulative q or kv sequence lengths.
    params.cu_q_seqlens = static_cast<int*>(cu_q_seqlens_d);
    params.cu_kv_seqlens = static_cast<int*>(cu_kv_seqlens_d);
    // cumulative mask sequence lengths.
    params.cu_mask_rows = static_cast<int*>(cu_mask_rows_d);

    // Set the different scale values.
    Data_type scale_type1 = (data_type == DATA_TYPE_FP16) || (data_type == DATA_TYPE_BF16) ? acc_type : DATA_TYPE_FP32;
    Data_type scale_softmax_type = scale_type1;
    Data_type scale_type2 = (data_type == DATA_TYPE_FP16) || (data_type == DATA_TYPE_BF16) ? data_type : DATA_TYPE_FP32;
    if (data_type == DATA_TYPE_E4M3)
    {
        scale_type1 = acc_type;
        scale_type2 = acc_type;
    }

    // Fuse 1.0f / softcapping_scale into scale_bmm1.
    bool const enable_attn_logit_softcapping = softcapping_scale_bmm1 != 0.f;
    float fused_scale_bmm1 = enable_attn_logit_softcapping ? scale_bmm1 / softcapping_scale_bmm1 : scale_bmm1;

    // use specialized hopper kernels without alibi support.
    // alibi or softcapping_scale cannot utilize the exp2f with fused_scale optimization.
    if (launch_params.warp_specialization && !has_alibi && !enable_attn_logit_softcapping)
    {
        set_alpha(params.scale_bmm1, fused_scale_bmm1 * float(M_LOG2E), DATA_TYPE_FP32);
    }
    else
    {
        set_alpha(params.scale_bmm1, fused_scale_bmm1, scale_type1);
    }
    set_alpha(params.scale_softmax, scale_softmax, scale_softmax_type);
    set_alpha(params.scale_bmm2, scale_bmm2, scale_type2);
    params.scale_bmm2_d = reinterpret_cast<uint32_t*>(scale_bmm2_d);
    params.softcapping_scale_bmm1 = softcapping_scale_bmm1;

    FMHA_CHECK_CUDA(cudaMemcpy(params.scale_bmm2_d, &params.scale_bmm2, sizeof(uint32_t), cudaMemcpyHostToDevice));

    // attention type, h_kv < h if MQA or GQA
    params.h_kv = h_kv;
    assert(h % h_kv == 0 && "MQA/GQA needs h to be divisible by h_kv!");
    params.h_q_per_kv = h / h_kv;
    params.has_alibi = has_alibi;
    params.alibi_params = fmha::AlibiParams(h);

    // Set flags
    params.is_s_padded = is_s_padded;
    params.use_int8_scale_max = use_int8_scale_max;

    // Do we enable the trick to replace I2F with FP math in the 2nd GEMM?
    if (data_type == DATA_TYPE_INT8)
    {
        params.enable_i2f_trick
            = -double(1 << 22) * double(scale_bmm2) <= -128.f && double(1 << 22) * double(scale_bmm2) >= 127.f;
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

static inline void determine_launch_params(Launch_params& launch_params, Data_type data_type, int sm, const size_t s,
    const size_t d, const Attention_mask_type attention_mask_type, const Attention_input_layout input_layout,
    bool const interleaved, bool const ignore_b1opt, bool const force_unroll, bool const use_tma,
    bool const force_non_flash_attention, bool const force_non_warp_specialization,
    bool const force_non_granular_tiling, bool const force_fp32_acc,
    // device props
    const cudaDeviceProp props)
{

    // Set launch params to choose kernels
    launch_params.ignore_b1opt = ignore_b1opt;
    launch_params.force_unroll = force_unroll;
    launch_params.force_fp32_acc = force_fp32_acc;
    launch_params.interleaved = interleaved;
    launch_params.attention_mask_type = attention_mask_type;
    launch_params.attention_input_layout = input_layout;

    // Set SM count and L2 cache size (used to determine launch blocks/grids to maximum performance)
    launch_params.multi_processor_count = props.multiProcessorCount;
    launch_params.device_l2_cache_size = props.l2CacheSize;

    // threshold for adopting flash attention or warp_specialized kernels.
    launch_params.flash_attention
        = (data_type == DATA_TYPE_FP16 || data_type == DATA_TYPE_BF16 || data_type == DATA_TYPE_E4M3)
        && (s >= 16 && d >= 16) && !force_non_flash_attention;

    // enable warp_speialized kernels when s >= 512 on hopper
    // note that warp_speialized kernels need flash attention + tma
    launch_params.warp_specialization
        = (data_type == DATA_TYPE_FP16 || data_type == DATA_TYPE_BF16 || data_type == DATA_TYPE_E4M3) && sm == 90
        && launch_params.flash_attention && !force_non_warp_specialization;
    // warp specialization kernels on hopper need tma
    launch_params.use_tma = use_tma || launch_params.warp_specialization;

    // use granular tiling on Ampere-style flash attention
    launch_params.use_granular_tiling
        = !force_non_granular_tiling && launch_params.flash_attention && !launch_params.warp_specialization && sm >= 80;

    if (launch_params.use_granular_tiling && (data_type == DATA_TYPE_E4M3 && sm == 80))
    {
        printf(
            "Fallback to non-granular-tiling kernels as tiled e4m3 kernels"
            "are not supported on Ada currently.\n");
        launch_params.use_granular_tiling = false;
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char** argv)
{

    // The device. Reset on destruction
    CudaDevice device;
    int sm = device.sm;
    cudaDeviceProp props = device.props;

    GpuTimer timer;

    // The batch size.
    size_t b = 128;
    // The number of heads.
    size_t h = 16;
    // The dimension of the Q, K and V vectors.
    size_t d = 64;
    // The dimension of V if set to non-zero, otherwise dimension of V equals to that of Q
    size_t dv = 0;
    // The length of the sequence.
    size_t s = 384;
    // Number of grouped heads in the seqlen dimension.
    size_t num_grouped_heads = 1;
    // Sliding Window Attention
    // Only pay attention to [max(0, query_idx - sliding_window_size), query_idx].
    size_t sliding_window_size = size_t(INT_MAX);
    // The chunked-attention size.
    size_t chunked_attention_size = 0;

    // The data type of the kernel.
    Data_type data_type = DATA_TYPE_FP16;
    // The type of the intermediate P matrix.
    Data_type acc_type = DATA_TYPE_FP16;
    // The type of the output.
    Data_type output_dtype = DATA_TYPE_FP16;
    // Is the output type set ?
    bool is_output_dtype_set = false;

    // The scaling factors.
    float scale_bmm1 = 0.f, scale_softmax = 0.f, scale_bmm2 = 0.25f;
    // The number of runs.
    int runs = 1, warm_up_runs = 0;
    // Do we use 1s for Q, K, V.
    bool use_1s_q = false, use_1s_k = false, use_1s_v = false;
    // The range of the different inputs.
    int range_q = 5, range_k = 3, range_v = 5;
    // The scale.
    float scale_q = 0.f, scale_k = 0.f, scale_v = 0.f;
    // The threshold for dropout. By default, drop 10%.
    float dropout = 0.1f;
    // Do we skip the checks.
    bool skip_checks = false;
    // The tolerance when checking results.
    float epsilon = -1.f; // data_type == DATA_TYPE_FP16 ? 0.015f : 0.f;
    // Use causal mask / padding_mask / sliding_or_chunked_causal mask / custom_mask input.
    Attention_mask_type attention_mask_type = Attention_mask_type::PADDING;
    // Use padded format for input QKV tensor & output O tensor.
    // Instead of variable lengths [total, h, 3, d]  where total = b1*s1 + b2*s2 + ... bn*sn,
    // use padded length [b, max_s, h, 3, d]         where max_s is the maximum expected seq len
    bool is_s_padded = false;

    // minimum sequence length for sampling variable seqlens
    uint32_t min_s = -1;

    // run interleaved kernels and transpose input and output accordingly
    bool interleaved = false;
    bool ignore_b1opt = false;
    bool force_unroll = false;
    // used by kernels that have different acc data types (like hmma, qmma)
    bool force_fp32_acc = false;
    bool force_non_flash_attention = false;
    // enable warp specialization kernels on sm 90
    bool force_non_warp_specialization = (sm != 90);
    bool use_int8_scale_max = false;
    bool verbose = true;
    bool save_softmax = false;

    // use granular tiling
    // supported only by Ampere-based Flash Attention at this moment
    bool force_non_granular_tiling = false;

    // set all sequence lengths to min(s, min_s)
    bool fix_s = false;

    bool v1 = false;

    // use TMA or not. ignored if not in SM90
    bool use_tma = false;

    // use alibi.
    bool has_alibi = false;

    // Use softcapping_scale_bmm1 (scale * __tanhf(x / scale)).
    float softcapping_scale_bmm1 = 0.f;

    // In multi-query or grouped-query attention (MQA/GQA), several Q heads are associated with one KV head
    bool multi_query_attention = false;
    size_t h_kv = 0;

    // The attention input layout.
    Attention_input_layout input_layout = Attention_input_layout::PACKED_QKV;

    // TRTLLM uses 64 by default in paged kv cache.
    size_t tokens_per_block = 64;

    // Attention that has different q and kv lengths.
    size_t s_q = 0;
    // different q and kv sequence lengths.
    bool different_q_kv_lengths = false;

    // SageAttention block sizes
    int sage_block_size_q = 0, sage_block_size_k = 0, sage_block_size_v = 0;

    // Use attention sinks (added to the denominator of softmax)
    bool use_attention_sinks = false;

    // Read the parameters from the command-line.
    for (int ii = 1; ii < argc; ++ii)
    {
        if (!strcmp(argv[ii], "-1s"))
        {
            use_1s_k = use_1s_q = use_1s_v = true;
        }
        else if (!strcmp(argv[ii], "-1s-k"))
        {
            use_1s_k = true;
        }
        else if (!strcmp(argv[ii], "-1s-q"))
        {
            use_1s_q = true;
        }
        else if (!strcmp(argv[ii], "-1s-v"))
        {
            use_1s_v = true;
        }
        else if (!strcmp(argv[ii], "-b") && ++ii < argc)
        {
            b = strtol(argv[ii], nullptr, 10);
        }
        else if (!strcmp(argv[ii], "-d") && ++ii < argc)
        {
            d = strtol(argv[ii], nullptr, 10);
        }
        else if (!strcmp(argv[ii], "-dv") && ++ii < argc)
        {
            dv = strtol(argv[ii], nullptr, 10);
        }
        else if (!strcmp(argv[ii], "-s-q") && ++ii < argc)
        {
            s_q = strtol(argv[ii], nullptr, 10);
            different_q_kv_lengths = true;
        }
        else if (!strcmp(argv[ii], "-dropout") && ++ii < argc)
        {
            dropout = (float) strtod(argv[ii], nullptr);
        }
        else if (!strcmp(argv[ii], "-epsilon") && ++ii < argc)
        {
            epsilon = (float) strtod(argv[ii], nullptr);
        }
        else if (!strcmp(argv[ii], "-h") && ++ii < argc)
        {
            h = strtol(argv[ii], nullptr, 10);
        }
        else if (!strcmp(argv[ii], "-int8"))
        {
            data_type = DATA_TYPE_INT8;
            acc_type = DATA_TYPE_INT32;
        }
        else if (!strcmp(argv[ii], "-fp16"))
        {
            data_type = DATA_TYPE_FP16;
            acc_type = DATA_TYPE_FP16;
        }
        else if (!strcmp(argv[ii], "-fp16-fp32"))
        {
            data_type = DATA_TYPE_FP16;
            acc_type = DATA_TYPE_FP32;
            force_fp32_acc = true;
        }
        else if (!strcmp(argv[ii], "-bf16"))
        {
            data_type = DATA_TYPE_BF16;
            acc_type = DATA_TYPE_FP32;
            force_fp32_acc = true;
        }
        else if (!strcmp(argv[ii], "-e4m3"))
        {
            data_type = DATA_TYPE_E4M3;
            // Technically not the acc type.
            acc_type = DATA_TYPE_FP32;
            force_fp32_acc = true;
        }
        else if (!strcmp(argv[ii], "-e4m3-fp16"))
        { // Ada QMMA only
            data_type = DATA_TYPE_E4M3;
            // Technically not the acc type.
            acc_type = DATA_TYPE_FP16;
        }
        else if (!strcmp(argv[ii], "-e4m3-fp32"))
        {
            data_type = DATA_TYPE_E4M3;
            // Technically not the acc type.
            acc_type = DATA_TYPE_FP32;
            force_fp32_acc = true;
        }
        else if (!strcmp(argv[ii], "-fp16-output"))
        {
            output_dtype = DATA_TYPE_FP16;
            is_output_dtype_set = true;
        }
        else if (!strcmp(argv[ii], "-bf16-output"))
        {
            output_dtype = DATA_TYPE_BF16;
            is_output_dtype_set = true;
        }
        else if (!strcmp(argv[ii], "-num-grouped-heads") && ++ii < argc)
        {
            num_grouped_heads = strtol(argv[ii], nullptr, 10);
        }
        else if (!strcmp(argv[ii], "-range-k") && ++ii < argc)
        {
            range_k = strtol(argv[ii], nullptr, 10);
        }
        else if (!strcmp(argv[ii], "-range-q") && ++ii < argc)
        {
            range_q = strtol(argv[ii], nullptr, 10);
        }
        else if (!strcmp(argv[ii], "-range-v") && ++ii < argc)
        {
            range_v = strtol(argv[ii], nullptr, 10);
        }
        else if (!strcmp(argv[ii], "-runs") && ++ii < argc)
        {
            runs = strtol(argv[ii], nullptr, 10);
        }
        else if (!strcmp(argv[ii], "-s") && ++ii < argc)
        {
            s = strtol(argv[ii], nullptr, 10);
        }
        else if (!strcmp(argv[ii], "-sliding-window-size") && ++ii < argc)
        {
            sliding_window_size = strtol(argv[ii], nullptr, 10);
        }
        else if (!strcmp(argv[ii], "-chunked-attention-size") && ++ii < argc)
        {
            chunked_attention_size = strtol(argv[ii], nullptr, 10);
        }
        else if (!strcmp(argv[ii], "-scale-bmm1") && ++ii < argc)
        {
            scale_bmm1 = (float) strtod(argv[ii], nullptr);
        }
        else if (!strcmp(argv[ii], "-scale-bmm2") && ++ii < argc)
        {
            scale_bmm2 = (float) strtod(argv[ii], nullptr);
        }
        else if (!strcmp(argv[ii], "-scale-k") && ++ii < argc)
        {
            scale_k = (float) strtod(argv[ii], nullptr);
        }
        else if (!strcmp(argv[ii], "-scale-softmax") && ++ii < argc)
        {
            scale_softmax = (float) strtod(argv[ii], nullptr);
        }
        else if (!strcmp(argv[ii], "-scale-q") && ++ii < argc)
        {
            scale_q = (float) strtod(argv[ii], nullptr);
        }
        else if (!strcmp(argv[ii], "-scale-v") && ++ii < argc)
        {
            scale_v = (float) strtod(argv[ii], nullptr);
        }
        else if (!strcmp(argv[ii], "-skip-checks"))
        {
            skip_checks = true;
        }
        else if (!strcmp(argv[ii], "-warm-up-runs") && ++ii < argc)
        {
            warm_up_runs = strtol(argv[ii], nullptr, 10);
        }
        else if (!strcmp(argv[ii], "-min-s") && ++ii < argc)
        {
            min_s = strtol(argv[ii], nullptr, 10);
        }
        else if (!strcmp(argv[ii], "-il"))
        {
            interleaved = true;
        }
        else if (!strcmp(argv[ii], "-causal-mask"))
        {
            attention_mask_type = Attention_mask_type::CAUSAL;
        }
        else if (!strcmp(argv[ii], "-sliding-or-chunked-causal-mask"))
        {
            attention_mask_type = Attention_mask_type::SLIDING_OR_CHUNKED_CAUSAL;
        }
        else if (!strcmp(argv[ii], "-custom-mask"))
        {
            attention_mask_type = Attention_mask_type::CUSTOM_MASK;
        }
        else if (!strcmp(argv[ii], "-multi-query-attention") || !strcmp(argv[ii], "-mqa"))
        {
            h_kv = 1;
            multi_query_attention = true; // subset of GQA
        }
        else if ((!strcmp(argv[ii], "-grouped-query-attention") || !strcmp(argv[ii], "-gqa")) && ++ii < argc)
        {
            h_kv = strtol(argv[ii], nullptr, 10);
            multi_query_attention = true;
        }
        else if (!strcmp(argv[ii], "-contiguous-q-kv"))
        {
            input_layout = Attention_input_layout::CONTIGUOUS_Q_KV;
        }
        else if (!strcmp(argv[ii], "-paged-kv"))
        {
            input_layout = Attention_input_layout::Q_PAGED_KV;
        }
        else if (!strcmp(argv[ii], "-separate-q-k-v"))
        {
            input_layout = Attention_input_layout::SEPARATE_Q_K_V;
        }
        else if (!strcmp(argv[ii], "-tokens-per-block") && ++ii < argc)
        {
            tokens_per_block = strtol(argv[ii], nullptr, 10);
        }
        else if (!strcmp(argv[ii], "-pad-s"))
        {
            is_s_padded = true;
        }
        else if (!strcmp(argv[ii], "-ignore-b1opt"))
        {
            ignore_b1opt = true;
        }
        else if (!strcmp(argv[ii], "-force-unroll"))
        {
            force_unroll = true;
        }
        else if (!strcmp(argv[ii], "-force-non-flash-attention"))
        {
            force_non_flash_attention = true;
            force_non_warp_specialization = true;
        }
        else if (!strcmp(argv[ii], "-force-flash-attention"))
        {
            fprintf(stderr,
                "Deprecation warning: -force-flash-attention is no longer valid; use "
                "-force-non-flash-attention instead, as Flash Attention is enabled by default.\n");
        }
        else if (!strcmp(argv[ii], "-force-non-warp-specialization"))
        {
            force_non_warp_specialization = true;
        }
        else if (!strcmp(argv[ii], "-force-non-granular-tiling") || !strcmp(argv[ii], "-force-non-tiled"))
        {
            force_non_granular_tiling = true;
        }
        else if (!strcmp(argv[ii], "-fix-s"))
        {
            fix_s = true;
        }
        else if (!strcmp(argv[ii], "-scale-max"))
        {
            use_int8_scale_max = true;
        }
        else if (!strcmp(argv[ii], "-v") && ++ii < argc)
        {
            int v = strtol(argv[ii], nullptr, 10);
            verbose = v != 0;
        }
        else if (!strcmp(argv[ii], "-v1"))
        {
            v1 = true;
        }
        else if (!strcmp(argv[ii], "-use-tma"))
        {
            use_tma = true;
            // flash attention + tma + non_warp_specialized kernels are not supported
            // use non_flash_attention + tma + non_warp_specialized instead
            if (force_non_warp_specialization)
            {
                force_non_flash_attention = true;
            }
        }
        else if (!strcmp(argv[ii], "-alibi"))
        {
            has_alibi = true;
        }
        else if (!strcmp(argv[ii], "-softcapping-scale-bmm1") && ++ii < argc)
        {
            softcapping_scale_bmm1 = (float) strtod(argv[ii], nullptr);
        }
        else if (!strcmp(argv[ii], "-save-softmax"))
        {
            save_softmax = true;
        }
        else if (!strcmp(argv[ii], "-sage-block-q") && ++ii < argc)
        {
            sage_block_size_q = strtol(argv[ii], nullptr, 10);
        }
        else if (!strcmp(argv[ii], "-sage-block-k") && ++ii < argc)
        {
            sage_block_size_k = strtol(argv[ii], nullptr, 10);
        }
        else if (!strcmp(argv[ii], "-sage-block-v") && ++ii < argc)
        {
            sage_block_size_v = strtol(argv[ii], nullptr, 10);
        }
        else if (!strcmp(argv[ii], "-use-attention-sinks"))
        {
            use_attention_sinks = true;
        }
        else
        {
            fprintf(stderr, "Unrecognized option: %s. Aborting!\n", argv[ii]);
            return -1;
        }
    }
    if (save_softmax == true)
    {
        bool is_MLA = (d == 192 && dv == 128);
        if (((!is_MLA) && input_layout != Attention_input_layout::CONTIGUOUS_Q_KV)
            || (is_MLA && input_layout != Attention_input_layout::SEPARATE_Q_K_V))
        {
            fprintf(stderr,
                "For normal attention, Only '--contiguous-q-kv' layout supports "
                "'-save-softmax'. For MLA only '-separate-q-k-v' layout supports "
                "'-save-softmax'.\n");
            exit(1);
        }
    }
    // Sanitize
    if (min_s == -1)
        min_s = s;
    min_s = std::min<uint32_t>(s, min_s);
    h_kv = multi_query_attention ? h_kv : h;

    // Check if the options are valid.
    if (different_q_kv_lengths)
    {
        assert(input_layout != Attention_input_layout::PACKED_QKV
            && "Packed QKV input layout is not supported with different q and kv lengths.");
        assert(s >= s_q && "q seqlen has to be smaller than or equal to the kv seqlen !");
    }
    else
    {
        s_q = s;
    }

    // Sliding window attention (only pay attention to sliding-window-size long previous tokens).
    if (sliding_window_size < s)
    {
        assert(
            chunked_attention_size == 0 && "chunked_attention_size should not be used when sliding_window_size is set");
        attention_mask_type = Attention_mask_type::SLIDING_OR_CHUNKED_CAUSAL;
    }
    // Chunked attention.
    if (chunked_attention_size > 0)
    {
        assert((chunked_attention_size & (chunked_attention_size - 1)) == 0
            && "chunked_attention_size has to be a power of 2");
        attention_mask_type = Attention_mask_type::SLIDING_OR_CHUNKED_CAUSAL;
    }

    // Set the norm.
    if (scale_bmm1 == 0.f)
    {
        scale_bmm1 = 1.f / sqrtf((float) d);
    }

    // Set the output type if not set by user.
    if (!is_output_dtype_set)
    {
        output_dtype = data_type;
    }

    // Force the softmax scale to 1.f for the FP16 kernel.
    if (data_type == DATA_TYPE_FP16)
    {
        scale_softmax = 1.f;
    }
    else if (data_type == DATA_TYPE_INT8 && scale_softmax == 0.f)
    {
        scale_softmax = std::max(512.f, (float) s);
    }
    else if (data_type == DATA_TYPE_E4M3 && scale_softmax == 0.f)
    {
        scale_softmax = 1.f; // For E4M3 this is hardcoded as the largest power-of-2 below E4M3_MAX
    }

    // Sage Attention uses the e4m3 data type
    if (sage_block_size_q > 0 || sage_block_size_k > 0 || sage_block_size_v > 0)
    {
        scale_softmax = 1.f;
        scale_bmm2 = 1.f;
        force_fp32_acc = true;
        acc_type = DATA_TYPE_FP32;
    }

    // Define the scaling factor for the different inputs.
    if (scale_q == 0.f)
    {
        scale_q = 1.f;
    }
    if (scale_k == 0.f)
    {
        scale_k = 1.f;
    }
    if (scale_v == 0.f)
    {
        // BF16 here just for debug.
        scale_v = (data_type == DATA_TYPE_FP16 || data_type == DATA_TYPE_BF16) ? 0.125f : 1.f;
    }
    if (has_alibi && attention_mask_type == Attention_mask_type::PADDING)
    {
        attention_mask_type = Attention_mask_type::CAUSAL;
    }

    // BF16 only support FP32 acc_type.
    if (data_type == DATA_TYPE_BF16 && acc_type != DATA_TYPE_FP32)
    {
        fprintf(stderr, "Only FP32 accumulation is supported for BF16 I/O\n");
        exit(1);
    }

    // Set the tolerance if not already set by the user.
    if (epsilon < 0.f)
    {
        switch (data_type)
        {
        case DATA_TYPE_FP16: epsilon = 0.015f; break;
        case DATA_TYPE_BF16: epsilon = 0.025f; break;
        case DATA_TYPE_E4M3: epsilon = 0.15f; break;
        default: epsilon = 0.f;
        }
        // the accuracy of SageAttention may be between fp8 and fp16/bf16 ?
        if (sage_block_size_q > 0 || sage_block_size_k > 0 || sage_block_size_v > 0)
        {
            epsilon = 0.05f;
        }
    }

    // let the dimension of V equal to that of Q if not set by user
    if (dv == 0)
    {
        dv = d;
    }

    // Debug info -- only in verbose mode.
    if (verbose)
    {
        // Running the following command.
        printf("Command.......: %s", argv[0]);
        for (int ii = 1; ii < argc; ++ii)
        {
            printf(" %s", argv[ii]);
        }
        printf("\n");

        // Device info.
        printf("Device........: %s\n", props.name);
        printf("Arch.(sm).....: %d\n", sm);
        printf("#.of.SMs......: %d\n", props.multiProcessorCount);

        // Problem info.
        printf("Batch ........: %lu\n", b);
        printf("Heads ........: %lu\n", h);
        printf("Dimension ....: %lu\n", d);
        printf("Dimension of V ....: %lu\n", dv);
        printf("Seq length ...: %lu\n", s);
        printf("Warm-up runs .: %d\n", warm_up_runs);
        printf("Runs..........: %d\n\n", runs);

        // The scaling factors for the 3 operations.
        printf("Scale bmm1 ...: %.6f\n", scale_bmm1);
        printf("Scale softmax.: %.6f\n", scale_softmax);
        printf("Scale bmm2 ...: %.6f\n", scale_bmm2);
        printf("\n");
    }

    // determine the launch params to select kernels
    Launch_params launch_params;
    determine_launch_params(launch_params, data_type, sm, s, d, attention_mask_type, input_layout, interleaved,
        ignore_b1opt, force_unroll, use_tma, force_non_flash_attention, force_non_warp_specialization,
        force_non_granular_tiling, force_fp32_acc, props);

    // The Q, K and V matrices are packed into one big matrix of size S x B x H x 3 x D.
    const size_t qkv_size = s * b * h * (2 * d + dv);
    // Allocate on the host.
    float* qkv_h = (float*) malloc(qkv_size * sizeof(float));
    // The size in bytes.
    const size_t qkv_size_in_bytes = get_size_in_bytes(qkv_size, data_type);
    // Allocate on the device.
    void *qkv_sbh3d_d = nullptr, *qkv_bsh3d_d = nullptr;
    FMHA_CHECK_CUDA(cudaMalloc(&qkv_sbh3d_d, qkv_size_in_bytes));
    FMHA_CHECK_CUDA(cudaMalloc(&qkv_bsh3d_d, qkv_size_in_bytes));

    // Contiguous KV cache buffer.
    // The shape is [B, 2, S, H, D].
    const size_t kv_size = b * s * h_kv * (d + dv);
    // The size in bytes.
    const size_t kv_size_in_bytes = get_size_in_bytes(kv_size, data_type);
    // Allocate on the host.
    void* contiguous_kv_h = malloc(kv_size_in_bytes);
    // Memset the buffer.
    memset(contiguous_kv_h, 0, kv_size_in_bytes);
    // Allocate on the device.
    void* contiguous_kv_d;
    FMHA_CHECK_CUDA(cudaMalloc(&contiguous_kv_d, kv_size_in_bytes));

    // Paged KV Cache buffer.
    // The shape is [B, 2, Blocks_per_sequence], and each block's buffer shape is [H, Tokens_per_block, Dh].
    void** kv_cache_ptrs_h = nullptr;
    void* kv_cache_pool_ptr = nullptr;
    int32_t *kv_cache_block_offsets_h, *kv_cache_block_offsets_d = nullptr;
    const size_t max_blocks_per_seq = (s + tokens_per_block - 1) / tokens_per_block;
    const size_t num_total_blocks = b * 2 * max_blocks_per_seq;
    kv_cache_ptrs_h = (void**) malloc(num_total_blocks * sizeof(void*));
    kv_cache_block_offsets_h = (int32_t*) malloc(num_total_blocks * sizeof(int32_t));
    const size_t paged_kv_block_size_in_bytes = get_size_in_bytes(tokens_per_block * h_kv * std::gcd(d, dv), data_type);
    FMHA_CHECK_CUDA(cudaMalloc((void**) (&kv_cache_block_offsets_d), num_total_blocks * sizeof(int32_t)));
    const size_t kv_cache_pool_sz
        = get_size_in_bytes(num_total_blocks * tokens_per_block * h_kv * (d + dv) / 2, data_type);
    FMHA_CHECK_CUDA(cudaMalloc((void**) (&kv_cache_pool_ptr), kv_cache_pool_sz));
    size_t ptr_index = 0;
    size_t abs_offset = 0;
    for (size_t bi = 0; bi < b; bi++)
    {
        for (int kv_offset = 0; kv_offset < 2; kv_offset++)
        {
            size_t block_size = get_size_in_bytes(tokens_per_block * h_kv * (kv_offset == 0 ? d : dv), data_type);
            for (size_t block_i = 0; block_i < max_blocks_per_seq; block_i++)
            {
                kv_cache_ptrs_h[ptr_index]
                    = reinterpret_cast<void*>(reinterpret_cast<char*>(kv_cache_pool_ptr) + abs_offset);
                assert(abs_offset % paged_kv_block_size_in_bytes == 0);
                kv_cache_block_offsets_h[ptr_index] = abs_offset / paged_kv_block_size_in_bytes;
                ptr_index++;
                abs_offset += block_size;
            }
        }
    }
    assert(ptr_index == num_total_blocks && abs_offset == kv_cache_pool_sz);
    FMHA_CHECK_CUDA(cudaMemcpy(
        kv_cache_block_offsets_d, kv_cache_block_offsets_h, num_total_blocks * sizeof(int32_t), cudaMemcpyDefault));

    // Q will always be [B, S, H, Dh] with paged kv cache.
    void* q_d;
    const size_t q_size = s * b * h * d;
    FMHA_CHECK_CUDA(cudaMalloc(&q_d, get_size_in_bytes(q_size, data_type)));

    // K has [B, S, H_kv, D] with separate kv cache.
    void* k_d;
    const size_t k_size = s * b * h_kv * d;
    FMHA_CHECK_CUDA(cudaMalloc(&k_d, get_size_in_bytes(k_size, data_type)));

    // V has [B, S, H_kv, Dv] with separate kv cache.
    void* v_d;
    const size_t v_size = s * b * h_kv * dv;
    FMHA_CHECK_CUDA(cudaMalloc(&v_d, get_size_in_bytes(v_size, data_type)));

    // Scale bmm2 (per-tensor).
    void* scale_bmm2_d;
    FMHA_CHECK_CUDA(cudaMalloc(&scale_bmm2_d, sizeof(uint32_t)));

    // The mask for dropout or any mask patterns.
    const size_t mask_size = s * b * s;
    // Allocate on the host.
    float* mask_h = (float*) malloc(mask_size * sizeof(float));
    // The size in bytes.
    const size_t mask_size_in_bytes = get_size_in_bytes(mask_size, DATA_TYPE_INT8);
    // Allocate on the device.
    void* mask_d = nullptr;
    if (!skip_checks)
    {
        FMHA_CHECK_CUDA(cudaMalloc(&mask_d, mask_size_in_bytes));
    }

    // The decomposition of threads and warps for BMM1.
    size_t warps_m, warps_n, warps_k;
    std::tie(warps_m, warps_n, warps_k) = get_warps(launch_params, sm, data_type, s, b, d, v1 ? 1 : 2);

    // print launch configuration
    printf(
        "v1=%d il=%d s_q=%lu, s=%lu b=%lu h=%lu/%lu d=%lu/%lu dtype=%s, output_dtype=%s, "
        "flash_attn=%s, "
        "warp_spec=%s, mask=%s, "
        "alibi=%s, attn=%s, qkv_layout=%s, wm=%lu wn=%lu\n",
        v1, interleaved, s_q, s, b, h, h_kv, d, dv, data_type_to_name(data_type).c_str(),
        data_type_to_name(output_dtype).c_str(),
        launch_params.flash_attention ? (launch_params.use_granular_tiling ? "true_tiled" : "true") : "false",
        launch_params.warp_specialization ? "true" : "false", mask_type_to_string(attention_mask_type).c_str(),
        has_alibi ? "true" : "false", h_kv == 1 ? "mqa" : (h_kv == h ? "mha" : "gqa"),
        attention_input_layout_to_string(input_layout).c_str(), warps_m, warps_n);

    // For multi-CTA cases, determine the size of the CTA wave.
    int heads_per_wave, ctas_per_head;
    get_grid_size(heads_per_wave, ctas_per_head, sm, data_type, b, s, h, d,
        false, // disable multi-cta kernels by default
        v1 ? 1 : 2);

    // The number of threads per CTA.
    const size_t threads_per_cta = warps_m * warps_n * warps_k * 32;
    // The number of mmas in the M dimension. We use one uint32_t per MMA in the M dimension.
    size_t mmas_m = (s + 16 * warps_m - 1) / (16 * warps_m);
    // The number of mmas in the N dimension.
    size_t mmas_n = (s + 16 * warps_n - 1) / (16 * warps_n);
    // We do not support more than 4 MMAS in the N dimension (as each MMA needs 8 bits in the mask).
    assert(!v1 || mmas_n <= 4);
    // The packed mask for dropout (in the fused kernel). Layout is B * MMAS_M * THREADS_PER_CTA.
    size_t packed_mask_size = b * mmas_m * threads_per_cta;
    // Flash attention on Ampere and Hopper, which supports multiple mmas_n
    if (!v1 && !force_non_flash_attention && attention_mask_type == Attention_mask_type::CUSTOM_MASK)
    {

        // We need to align q and k sequence lengths.
        size_t rounded_q_s = align_to(s, size_t(fmha::FLASH_ATTEN_MASK_M_ALIGNMENT));
        size_t rounded_k_s = align_to(s, size_t(fmha::FLASH_ATTEN_MASK_N_ALIGNMENT));
        // The number of mmas in the M dimension (MMA_M = 64).
        mmas_m = rounded_q_s / fmha::FLASH_ATTEN_MASK_MMA_M;
        // The number of mmas in the N dimension (MMA_N = 64).
        mmas_n = rounded_k_s / fmha::FLASH_ATTEN_MASK_MMA_N;
        // Each thread holds 32 bit (2 rows, 16 cols -> 8 core MMAs) in one MMA here.
        packed_mask_size = b * mmas_m * mmas_n * threads_per_cta;
    }
    // The size in bytes.
    const size_t packed_mask_size_in_bytes = packed_mask_size * sizeof(uint32_t);
    // Allocate on the host.
    uint32_t* packed_mask_h = (uint32_t*) malloc(packed_mask_size_in_bytes);
    // Set it to 0 (indicates that all elements are valid).
    memset(packed_mask_h, 0, packed_mask_size_in_bytes);
    // Allocate on the device.
    void* packed_mask_d = nullptr;

    // The size of the attention sinks.
    const size_t attention_sinks_size_in_bytes = h * sizeof(float);

    // The attention sinks.
    void* attention_sinks_d = nullptr;
    if (use_attention_sinks)
    {
        // Allocate on the host.
        float* attention_sinks_h = (float*) malloc(attention_sinks_size_in_bytes);
        // Randomly initialize the attention sinks.
        random_init("attention_sinks", attention_sinks_h, 1, h, 1, false, 5.f, 1.f, verbose);
        // Allocate on the device.
        FMHA_CHECK_CUDA(cudaMalloc(&attention_sinks_d, attention_sinks_size_in_bytes));
        // Copy from the host to the device.
        FMHA_CHECK_CUDA(
            cudaMemcpy(attention_sinks_d, attention_sinks_h, attention_sinks_size_in_bytes, cudaMemcpyDefault));
    }

    // The O matrix is packed as S * B * H * D.
    const size_t o_size = s * b * h * dv;
    // Allocate on the host.
    float* o_h = (float*) malloc(o_size * sizeof(float));
    // The size in bytes.
    const size_t o_size_in_bytes = get_size_in_bytes(o_size, data_type);
    // Allocate on the device.
    void* o_d = nullptr;
    FMHA_CHECK_CUDA(cudaMalloc(&o_d, o_size_in_bytes));

    // The softmax_stats_d vector is used to store the max/sum of the softmax per token
    void* softmax_stats_d;
    FMHA_CHECK_CUDA(cudaMalloc(&softmax_stats_d, 2 * sizeof(float) * b * s * h));
    FMHA_CHECK_CUDA(cudaMemset(softmax_stats_d, 0x00, 2 * sizeof(float) * b * s * h));

    // The size in bytes.
    const size_t tmp_size_in_bytes = get_size_in_bytes(o_size, acc_type);
    // Allocate on the device.
    void* tmp_d = nullptr;
    if (data_type != acc_type)
    {
        FMHA_CHECK_CUDA(cudaMalloc(&tmp_d, tmp_size_in_bytes));
    }

    // Allocate the reference on the host.
    float* o_ref_h = (float*) malloc(o_size * sizeof(float));
    float* softmax_stats_ref_h = (float*) malloc(2 * b * s * h * sizeof(float));
    float* softmax_stats_h = (float*) malloc(2 * b * s * h * sizeof(float));

    // The P matrix is stored as one big matrix of size S x B x H x S.
    const size_t p_size = s * b * h * s;
    // The size in bytes.
    const size_t p_size_in_bytes = get_size_in_bytes(p_size, acc_type);
    // Allocate on the device.
    void* p_d = nullptr;
    if (!skip_checks)
    {
        FMHA_CHECK_CUDA(cudaMalloc(&p_d, p_size_in_bytes));
    }

    // Allocate the reference on the host.
    float* p_ref_h = (float*) malloc(p_size * sizeof(float));
#if defined(STORE_P)
    // Allocate on the host.
    float* p_h = (float*) malloc(p_size * sizeof(float));
#endif // defined(STORE_P)

    // The size in bytes of the S matrix (the data type may be different from P for int8).
    const size_t s_size_in_bytes = get_size_in_bytes(p_size, data_type);
    // Allocate on the device.
    void* s_d = nullptr;
    if (!skip_checks)
    {
        FMHA_CHECK_CUDA(cudaMalloc(&s_d, s_size_in_bytes));
    }

    // Allocate the reference on the host.
    float* s_ref_h = (float*) malloc(p_size * sizeof(float));

    // Allocate on the host.
    float* s_h = (float*) malloc(p_size * sizeof(float));
    // Make sure we set the seed for reproducible results.
    srand(1234UL);

    // Set the Q, K and V matrices.
    random_init("Q", qkv_h + 0 * d, d, s * b * h, 2 * d + dv, use_1s_q, range_q, scale_q, verbose);
    random_init("K", qkv_h + 1 * d, d, s * b * h, 2 * d + dv, use_1s_k, range_k, scale_k, verbose);
    random_init("V", qkv_h + 2 * d, dv, s * b * h, 2 * d + dv, use_1s_v, range_v, scale_v, verbose);
    // iota_init("Q", qkv_h + 0 * d, d, s * b * h, 3 * d, use_1s_q, range_q, scale_q, verbose, true, 0);
    // iota_init("K", qkv_h + 1 * d, d, s * b * h, 3 * d, use_1s_k, range_k, scale_k, verbose, true, 128);
    // iota_init("V", qkv_h + 2 * d, d, s * b * h, 3 * d, use_1s_v, range_v, scale_v, verbose, true, 256);

    // Multi-query or grouped-query attention for reference input
    if (multi_query_attention)
    {
        for (size_t sbi = 0; sbi < s * b; sbi++)
        {
            for (size_t hi = 0; hi < h; hi++)
            {
                for (size_t di = 0; di < d; di++)
                {
                    // E.g., h=8, h_kv=4
                    //            hi: 0, 1, 2, 3, 4, 5, 6, 7
                    // hi_kv_scatter: 0, 0, 2, 2, 4, 4, 6, 6
                    int const h_per_group = h / h_kv;
                    int const hi_kv_scatter = (hi / h_per_group) * h_per_group;
                    size_t src_offset = sbi * h * 3 * d + hi_kv_scatter * 3 * d + di; // [sbi, hi_kv_scatter, 0, di]
                    size_t dst_offset = sbi * h * 3 * d + hi * 3 * d + di;            // [sbi, hi, 0, di]

                    // make sure all heads of kv in a group share the same d
                    qkv_h[dst_offset + 1 * d]
                        = qkv_h[src_offset + 1 * d]; // qkv[sbi, hi, 1, di] = qkv[sbi, hi_kv_scatter, 1, di]
                    qkv_h[dst_offset + 2 * d]
                        = qkv_h[src_offset + 2 * d]; // qkv[sbi, hi, 2, di] = qkv[sbi, hi_kv_scatter, 2, di]
                }
            }
        }
    }

    //   WAR fOR MISSING CUBLAS FP8 NN SUPPORT.
    //   Transpose V, so that we can do a TN BMM2, i.e. O = S x V'  instead of O = S x V.
    float* vt_h = (float*) malloc(o_size * sizeof(float));
    void* vt_d = nullptr;
    FMHA_CHECK_CUDA(cudaMalloc(&vt_d, o_size_in_bytes));
    for (size_t it = 0; it < o_size; it++)
    {
        // vt is B x H x D x S
        size_t si = it % s;
        size_t di = (it / s) % dv;
        size_t hi = ((it / s) / dv) % h;
        size_t bi = (((it / s) / dv) / h) % b;
        // qkv is S x B x H x 3 x D
        size_t qkv_idx = si * b * h * (2 * d + dv) + bi * h * (2 * d + dv) + hi * (2 * d + dv) + 2 * d // index V here
            + di;
        vt_h[it] = qkv_h[qkv_idx];
    }
    FMHA_CHECK_CUDA(cuda_memcpy_h2d(vt_d, vt_h, o_size, data_type));

    // // DEBUG.
    // float sum = 0.f;
    // for( size_t si = 0; si < s; ++si ) {
    //   float v = qkv_h[si*b*h*3*d + 2*d];
    //   printf("V[%3d]=%8.3f\n", si, v);
    //   sum += v;
    // }
    // printf("Sum of V = %8.3f\n", sum);
    // // END OF DEBUG.

    // Copy from the host to the device.
    FMHA_CHECK_CUDA(cuda_memcpy_h2d(qkv_sbh3d_d, qkv_h, qkv_size, data_type));

    // Create the buffer of mask.
    // if(verbose) {printf("Init .........: mask\n"); }
    // random_init_with_zeroes_or_ones(mask_h, b*s, false, 1.f - dropout, verbose);

    std::vector<uint32_t> seqlens(b, 0); // randomly draw a batch of sequence lengths >= min_s
    std::transform(seqlens.begin(), seqlens.end(), seqlens.begin(),
        [=](const uint32_t)
        {
            if (fix_s)
            {
                return std::min(uint32_t(s), min_s);
            }
            if (s == min_s)
            {
                return min_s;
            }
            uint32_t s_ = s - min_s + 1;
            uint32_t ret = min_s + (rand() % s_);
            assert(ret <= s);
            return ret;
        });

    // Compute the prefix sum of the sequence lengths.
    std::vector<int> cu_seqlens(b + 1, 0);
    for (int it = 0; it < b; it++)
    {
        cu_seqlens[it + 1] = cu_seqlens[it] + seqlens[it];
    }
    int total = cu_seqlens.back();
    seqlens.emplace_back(total);

    // Different q and kv sequence lengths.
    std::vector<uint32_t> q_seqlens = seqlens;
    std::vector<int> cu_q_seqlens = cu_seqlens;
    if (different_q_kv_lengths)
    {
        for (int it = 0; it < b; it++)
        {
            q_seqlens[it] = s_q;
            cu_q_seqlens[it + 1] = cu_q_seqlens[it] + q_seqlens[it];
        }
    }

    // Compute the prefix sum of the mask sequence lengths.
    std::vector<int> cu_mask_rows(b + 1, 0);
    // The mask_h row offset in each sequence to support s_q < s_kv.
    // we only need the last s_q rows in the [s, s] mask_h.
    std::vector<int> mask_h_row_offsets(b);
    for (int it = 0; it < b; it++)
    {
        // The actual q sequence length.
        int actual_q_seqlen = q_seqlens[it];
        // The mask_h row offset.
        mask_h_row_offsets[it] = seqlens[it] - q_seqlens[it];
        // Round up the sequence length to multiple of 128.
        int mask_seqlen = align_to(actual_q_seqlen, fmha::FLASH_ATTEN_MASK_M_ALIGNMENT);
        cu_mask_rows[it + 1] = cu_mask_rows[it] + mask_seqlen;
    }

    // transfer to device
    void *cu_seqlens_d, *cu_q_seqlens_d, *cu_mask_rows_d;
    FMHA_CHECK_CUDA(cudaMalloc(&cu_seqlens_d, sizeof(int) * cu_seqlens.size()));
    FMHA_CHECK_CUDA(cudaMalloc(&cu_q_seqlens_d, sizeof(int) * cu_q_seqlens.size()));
    FMHA_CHECK_CUDA(cudaMalloc(&cu_mask_rows_d, sizeof(int) * cu_mask_rows.size()));
    FMHA_CHECK_CUDA(
        cudaMemcpy(cu_seqlens_d, cu_seqlens.data(), sizeof(int) * cu_seqlens.size(), cudaMemcpyHostToDevice));
    FMHA_CHECK_CUDA(
        cudaMemcpy(cu_q_seqlens_d, cu_q_seqlens.data(), sizeof(int) * cu_q_seqlens.size(), cudaMemcpyHostToDevice));
    FMHA_CHECK_CUDA(
        cudaMemcpy(cu_mask_rows_d, cu_mask_rows.data(), sizeof(int) * cu_mask_rows.size(), cudaMemcpyHostToDevice));

    size_t qkv_packed_size = cu_seqlens.back() * h * (2 * d + dv);
    size_t qkv_packed_size_in_bytes = get_size_in_bytes(qkv_packed_size, data_type);
    void* qkv_packed_d = nullptr;
    FMHA_CHECK_CUDA(cudaMalloc(&qkv_packed_d, qkv_packed_size_in_bytes));

    // Specify device buffers for multi-query attention or grouped-query attention
    // TODO: Use the same buffer for all cases, and allow to set name to aid tracing/debugging
    // e.g.,
    //   Buffer<float> qkv_buf(size);
    //   if( packed ) { qkv_buf.set_name("QKV_packed[total, h, 3, d]"); }
    //   else { qkv_buf.set_name("QKV_padded[b, s, h, 3, d]"); }
    //   qkv_buf.copy_to_device();
    //   float *qkv_buf_d = qkv_buf.get_device_buf();
    // Or, more aggressively, use torch::Tensor from PyTorch ATen
    size_t mqa_qkv_packed_size = cu_seqlens.back() * (h + 2 * h_kv) * d;
    size_t mqa_qkv_packed_size_in_bytes = get_size_in_bytes(mqa_qkv_packed_size, data_type);
    size_t mqa_qkv_size = b * s * (h + 2 * h_kv) * d; // original padded tensor
    size_t mqa_qkv_size_in_bytes = get_size_in_bytes(mqa_qkv_size, data_type);
    void* mqa_qkv_packed_d = nullptr;
    void* mqa_qkv_d = nullptr;
    FMHA_CHECK_CUDA(cudaMalloc(&mqa_qkv_packed_d, mqa_qkv_packed_size_in_bytes));
    FMHA_CHECK_CUDA(cudaMalloc(&mqa_qkv_d, mqa_qkv_size_in_bytes));

    const size_t o_packed_size = cu_seqlens.back() * h * dv;
    // Allocate on the host.
    float* o_packed_h = (float*) malloc(o_packed_size * sizeof(float));
    void* o_packed_d = nullptr;

    size_t o_packed_size_in_bytes = get_size_in_bytes(o_packed_size, output_dtype);
    FMHA_CHECK_CUDA(cudaMalloc(&o_packed_d, o_packed_size_in_bytes));

    // qkv_packed_h is TotalH3D
    std::vector<float> qkv_packed_h(qkv_packed_size);
    extract_and_transpose_input<float>(qkv_packed_h.data(), qkv_h, seqlens, s, b, h, d, dv, 3, false);
    if (interleaved)
    {
        x_vec32(true, qkv_packed_h.data(), h, total, 3);
    }

    // qkv_h is SBH3D
    // qkv_bsh3d_h is BSH3D
    std::vector<float> qkv_bsh3d_h(qkv_size);
    extract_and_transpose_input<float>(qkv_bsh3d_h.data(), qkv_h, seqlens, s, b, h, d, dv, 3, is_s_padded);
    if (interleaved)
    {
        x_vec32(true, qkv_bsh3d_h.data(), h, b * h, 3);
    }

    std::vector<float> mqa_qkv_packed_h(mqa_qkv_packed_size);
    std::vector<float> mqa_qkv_h(mqa_qkv_size);
    // for now MLA doesn't use MQA, may enable it in the future
    if (d == dv)
    {
        // from qkv[s, h, 3, d] to mqa_qkv[s, h + 2*h_kv, d]
        // where
        //  Q is qkv[s, h, 0, d],
        //  K is qkv[s, h, 1, d],
        //  V is qkv[s, h, 2, d]
        // and
        //  MQA_Q is mqa_qkv[s, h, [       0 :          h - 1], d],
        //  MQA_K is mqa_qkv[s, h, [       h :   h + h_kv - 1], d],
        //  MQA_V is mqa_qkv[s, h, [h + h_kv : h + 2*h_kv - 1], d]
        for (size_t si = 0; si < cu_seqlens.back(); si++)
        {
            for (size_t hi = 0; hi < h; hi++)
            {
                for (size_t di = 0; di < d; di++)
                {
                    // Q: [si, hi, di] <- [si, hi, 0, di]
                    mqa_qkv_packed_h[si * (h + 2 * h_kv) * d + hi * d + di]
                        = qkv_packed_h[si * h * 3 * d + hi * 3 * d + 0 * d + di];
                    if (hi < h_kv)
                    {
                        // E.g., h=8, h_kv=4
                        //     src kv id: 0, 0, 1, 1, 2, 2, 3, 3
                        //            hi: 0, 1, 2, 3, 4, 5, 6, 7
                        // hi_kv_scatter: 0, 2, 4, 6, x, x, x, x
                        int const h_per_group = h / h_kv;
                        int const hi_kv_scatter = hi * h_per_group;
                        // K: [si, h + hi, di] <- [si, hi_kv_scatter, 1, di]
                        mqa_qkv_packed_h[si * (h + 2 * h_kv) * d + (h + hi) * d + di]
                            = qkv_packed_h[si * 3 * h * d + hi_kv_scatter * 3 * d + 1 * d + di];
                        // V: [si, h + h_kv + hi, di] <- [si, hi_kv_scatter, 2, di]
                        mqa_qkv_packed_h[si * (h + 2 * h_kv) * d + (h + h_kv + hi) * d + di]
                            = qkv_packed_h[si * 3 * h * d + hi_kv_scatter * 3 * d + 2 * d + di];
                    }
                }
            }
        }

        // from qkv_bsh3d_h[b, s, h, 3, d] to mqa_qkv[b, s, h + 2*h_kv, d]
        for (size_t bi = 0; bi < b; bi++)
        {
            int actual_s = seqlens[bi];
            for (size_t si = 0; si < actual_s; si++)
            {
                for (size_t hi = 0; hi < h; hi++)
                {
                    for (size_t di = 0; di < d; di++)
                    {
                        mqa_qkv_h[bi * s * (h + 2 * h_kv) * d + si * (h + 2 * h_kv) * d + hi * d + di]
                            = qkv_bsh3d_h[bi * s * h * 3 * d + si * h * 3 * d + hi * 3 * d + 0 * d + di];
                        if (hi < h_kv)
                        {
                            // E.g., h=8, h_kv=4
                            //     src kv id: 0, 0, 1, 1, 2, 2, 3, 3
                            //            hi: 0, 1, 2, 3, 4, 5, 6, 7
                            // hi_kv_scatter: 0, 2, 4, 6, x, x, x, x
                            int const h_per_group = h / h_kv;
                            int const hi_kv_scatter = hi * h_per_group;
                            mqa_qkv_h[bi * s * (h + 2 * h_kv) * d + si * (h + 2 * h_kv) * d + (h + hi) * d + di]
                                = qkv_bsh3d_h[bi * s * h * 3 * d + si * h * 3 * d + hi_kv_scatter * 3 * d + 1 * d + di];
                            mqa_qkv_h[bi * s * (h + 2 * h_kv) * d + si * (h + 2 * h_kv) * d + (h + h_kv + hi) * d + di]
                                = qkv_bsh3d_h[bi * s * h * 3 * d + si * h * 3 * d + hi_kv_scatter * 3 * d + 2 * d + di];
                        }
                    }
                }
            }
        }
    }
    // if( verbose ) {
    //     print_tensor(qkv_packed_h.data() + 0 * d, d, total * h, 3 * d, "Packed Q[bs, h, d]");
    //     print_tensor(qkv_packed_h.data() + 1 * d, d, total * h, 3 * d, "Packed K[bs, h, d]");
    //     print_tensor(qkv_packed_h.data() + 2 * d, d, total * h, 3 * d, "Packed V[bs, h, d]");

    //     print_tensor(mqa_qkv_packed_h.data() + 0 * d,            h * d,    total, (h + 2 * h_kv) * d, "Packed MQA
    //     Q[bs, h*d]"); print_tensor(mqa_qkv_packed_h.data() + h * d,            h_kv * d, total, (h + 2 * h_kv) * d,
    //     "Packed MQA K[bs, h_kv*d]"); print_tensor(mqa_qkv_packed_h.data() + h * d + h_kv * d, h_kv * d, total, (h + 2
    //     * h_kv) * d, "Packed MQA V[bs, h_kv*d]");

    //     print_tensor(qkv_bsh3d_h.data() + 0 * d, d, b * h * s, 3 * d, "Padded Q[b, s, h, d]");
    //     print_tensor(qkv_bsh3d_h.data() + 1 * d, d, b * h * s, 3 * d, "Padded K[b, s, h, d]");
    //     print_tensor(qkv_bsh3d_h.data() + 2 * d, d, b * h * s, 3 * d, "Padded V[b, s, h, d]");

    //     print_tensor(mqa_qkv_h.data() + 0 * d,            h * d,    b * s, (h + 2 * h_kv) * d, "Padded MQA Q[b, s,
    //     h*d]"); print_tensor(mqa_qkv_h.data() + h * d,            h_kv * d, b * s, (h + 2 * h_kv) * d, "Padded MQA
    //     K[b, s, h_kv*d]"); print_tensor(mqa_qkv_h.data() + h * d + h_kv * d, h_kv * d, b * s, (h + 2 * h_kv) * d,
    //     "Padded MQA V[b, s, h_kv*d]");
    // }

    // Contiguous KV Cache and Separate KV Cache.
    store_q_and_contiguous_kv_cache(q_d, k_d, v_d, contiguous_kv_h, contiguous_kv_d,
        reinterpret_cast<float const*>(qkv_packed_h.data()), reinterpret_cast<int const*>(cu_seqlens.data()),
        reinterpret_cast<int const*>(cu_q_seqlens.data()), b, s, h, h_kv, d, dv, data_type);

    // Paged KV Cache.
    store_paged_kv_cache(kv_cache_ptrs_h, reinterpret_cast<float const*>(qkv_packed_h.data()),
        reinterpret_cast<int const*>(cu_seqlens.data()), max_blocks_per_seq, tokens_per_block, b, h, h_kv, d, dv,
        data_type);

    // Copy packed, padded, mqa packed, mqa padded data buffers
    // TODO: use the same buffer for all cases
    FMHA_CHECK_CUDA(cuda_memcpy_h2d(qkv_packed_d, qkv_packed_h.data(), qkv_packed_size, data_type));
    FMHA_CHECK_CUDA(cuda_memcpy_h2d(mqa_qkv_packed_d, mqa_qkv_packed_h.data(), mqa_qkv_packed_size, data_type));
    FMHA_CHECK_CUDA(cuda_memcpy_h2d(mqa_qkv_d, mqa_qkv_h.data(), mqa_qkv_size, data_type));
    FMHA_CHECK_CUDA(cuda_memcpy_h2d(qkv_bsh3d_d, qkv_bsh3d_h.data(), qkv_size, data_type));

    // Is MTP used?
    bool is_mtp = (d == 576 && dv == 512);

    for (size_t so = 0; so < s; ++so)
    { // s_q
        for (size_t bi = 0; bi < b; ++bi)
        {
            int actual_seqlen = seqlens[bi];
            for (size_t si = 0; si < s; ++si)
            { // s_kv
                // Are both the query and the key inside the sequence?
                bool valid = (si < actual_seqlen) && (so < actual_seqlen);
                // FIXME: add random mask generator.
                //  attention_mask_type == Attention_mask_type::CUSTOM_MASK
                if (attention_mask_type == Attention_mask_type::CUSTOM_MASK
                    || attention_mask_type == Attention_mask_type::CAUSAL
                    || attention_mask_type == Attention_mask_type::SLIDING_OR_CHUNKED_CAUSAL)
                {
                    valid = valid && (so >= si);
                }
                if (attention_mask_type == Attention_mask_type::SLIDING_OR_CHUNKED_CAUSAL)
                {
                    if (chunked_attention_size > 0)
                    {
                        int chunk_idx = so / chunked_attention_size;
                        valid = valid && (si >= (chunk_idx * chunked_attention_size));
                    }
                    else
                    {
                        valid = valid && (si >= std::max(int(so + 1 - sliding_window_size), 0));
                    }
                }
                if (is_mtp)
                {
                    // Only the last s_q tokens are used for verifying the results.
                    size_t idx = so - (actual_seqlen - s_q);
                    size_t num_mtp_tokens = s_q / num_grouped_heads;
                    size_t mtp_token_idx = idx / num_grouped_heads;
                    valid
                        = idx >= 0 && si < (actual_seqlen - num_mtp_tokens + 1 + mtp_token_idx) && (so < actual_seqlen);
                }
                if (!skip_checks)
                {
                    // The mask is stored as floats.
                    mask_h[so * b * s + bi * s + si] = valid ? 1.f : 0.f; // mask dims [s_q, b, s_kv]
                }
            }
        }
    }

    if (verbose)
    {
        printf("Sequence lengths (first 10 batches): ");
        for (int bi = 0; bi < seqlens.size() && bi < 10; bi++)
        {
            printf("%d, ", seqlens[bi]);
        }
        printf("\n");
    }

    if (v1)
    {
        assert(!interleaved && "Interleaved not supported in v1");
        assert(mmas_n <= 4 && "Not supported");

        FMHA_CHECK_CUDA(cudaMalloc(&packed_mask_d, packed_mask_size_in_bytes));
        if (sm == 70)
        {
            pack_mask_sm70(packed_mask_h, mask_h, s, b, mmas_m, mmas_n, warps_m, warps_n, threads_per_cta);
        }
        else
        {
            pack_mask(packed_mask_h, mask_h, s, b, mmas_m, mmas_n, warps_m, warps_n, threads_per_cta);
        }

        // Copy the packed mask to the device.
        if (!skip_checks)
        {
            FMHA_CHECK_CUDA(
                cudaMemcpy(packed_mask_d, packed_mask_h, packed_mask_size_in_bytes, cudaMemcpyHostToDevice));
        }
    }
    else if (attention_mask_type == Attention_mask_type::CUSTOM_MASK)
    {
        FMHA_CHECK_CUDA(cudaMalloc(&packed_mask_d, packed_mask_size_in_bytes));
        assert(fmha::FLASH_ATTEN_MASK_MMA_M == warps_m * 16 && "Not supported");
        assert(fmha::FLASH_ATTEN_MASK_MMA_N / 8 == 8 && "Not supported");
        pack_flash_attention_mask(packed_mask_h, mask_h, b, s, warps_m, warps_n, threads_per_cta, mmas_n,
            fmha::FLASH_ATTEN_MASK_MMA_N / 8, mask_h_row_offsets.data(), cu_mask_rows.data());

        // Copy the packed mask to the device.
        FMHA_CHECK_CUDA(cudaMemcpy(packed_mask_d, packed_mask_h, packed_mask_size_in_bytes, cudaMemcpyHostToDevice));
    }

    // Copy the mask to the device.
    if (!skip_checks)
    {
        FMHA_CHECK_CUDA(cuda_memcpy_h2d(mask_d, mask_h, mask_size, DATA_TYPE_INT8));
    }

    // non-owning pointer to the IO buffer
    void* qkv_d_view = nullptr;
    void* o_d_view = nullptr;
    int o_view_size = 0;
    if (is_s_padded)
    {
        qkv_d_view = multi_query_attention ? mqa_qkv_d : qkv_bsh3d_d;
        o_d_view = o_d;
        o_view_size = o_size;
    }
    else
    {
        qkv_d_view = multi_query_attention ? mqa_qkv_packed_d : qkv_packed_d;
        o_d_view = o_packed_d;
        o_view_size = o_packed_size;
    }
    void* softmax_stats_ptr = save_softmax ? softmax_stats_d : nullptr;
    // Set the params.
    bert::Fused_multihead_attention_params_v1 params_v1;
    set_params(params_v1, data_type, acc_type, b, s, h, d, mmas_m * threads_per_cta, qkv_sbh3d_d, packed_mask_d, o_d,
        p_d, s_d, scale_bmm1, scale_softmax, scale_bmm2, has_alibi);

    bert::Fused_multihead_attention_params_v2 params_v2;
    set_params(params_v2, launch_params, data_type, acc_type, output_dtype, input_layout, b, s_q, s, h, h_kv, d, dv,
        total, num_grouped_heads, sliding_window_size, chunked_attention_size,
        // Paged kv cache.
        tokens_per_block, qkv_d_view, q_d, k_d, v_d, contiguous_kv_d, kv_cache_pool_ptr, kv_cache_block_offsets_d,
        packed_mask_d, cu_mask_rows_d, attention_sinks_d, cu_seqlens_d, cu_q_seqlens_d, o_d_view, p_d, s_d,
        softmax_stats_ptr, scale_bmm2_d, scale_bmm1, scale_softmax, scale_bmm2, softcapping_scale_bmm1,
        use_int8_scale_max, interleaved, is_s_padded, has_alibi);

    // total number of tokens is needed to set TMA desc on the host.
    launch_params.total_q_seqlen = q_seqlens[b];
    launch_params.total_kv_seqlen = seqlens[b];
    // set enable_attn_logit_softcapping to select the right kernel.
    launch_params.enable_attn_logit_softcapping = softcapping_scale_bmm1 != 0.f;

    // Allocate barriers and locks.
    void* counters_d = nullptr;
    if (ctas_per_head > 1)
    {
        size_t sz = heads_per_wave * sizeof(int);
        FMHA_CHECK_CUDA(cudaMalloc((void**) &counters_d, 3 * sz));
    }

    // Allocate scratch storage for softmax.
    void *max_scratch_d = nullptr, *sum_scratch_d = nullptr;
    if (ctas_per_head > 1)
    {
        size_t sz = heads_per_wave * ctas_per_head * threads_per_cta * sizeof(float);
        FMHA_CHECK_CUDA(cudaMalloc((void**) &max_scratch_d, sz));
        FMHA_CHECK_CUDA(cudaMalloc((void**) &sum_scratch_d, sz));
    }

    // Allocate temporary storage for the parallel reduction.
    void* o_scratch_d = nullptr;
    if (ctas_per_head > 1 && data_type != DATA_TYPE_FP16)
    {
        size_t sz = heads_per_wave * threads_per_cta * MAX_STGS_PER_LOOP * sizeof(uint4);
        FMHA_CHECK_CUDA(cudaMalloc((void**) &o_scratch_d, sz));
    }

    // Allocate tile id for dynamic scheduling
    void* tile_id_counter_d = nullptr;
    FMHA_CHECK_CUDA(cudaMalloc((void**) &tile_id_counter_d, sizeof(uint32_t)));

    // The number of heads computed per wave.
    params_v1.heads_per_wave = heads_per_wave;
    params_v2.heads_per_wave = heads_per_wave;

    // Barriers for the global sync in the multi-CTA kernel(s).
    params_v1.counters = (int*) counters_d + 0 * heads_per_wave;
    params_v2.counters = (int*) counters_d + 0 * heads_per_wave;
    params_v1.max_barriers = (int*) counters_d + 0 * heads_per_wave;
    params_v2.max_barriers = (int*) counters_d + 0 * heads_per_wave;
    params_v1.sum_barriers = (int*) counters_d + 1 * heads_per_wave;
    params_v2.sum_barriers = (int*) counters_d + 1 * heads_per_wave;
    params_v1.locks = (int*) counters_d + 2 * heads_per_wave;
    params_v2.locks = (int*) counters_d + 2 * heads_per_wave;

    // Scratch storage for softmax.
    params_v1.max_scratch_ptr = (float*) max_scratch_d;
    params_v2.max_scratch_ptr = (float*) max_scratch_d;
    params_v1.sum_scratch_ptr = (float*) sum_scratch_d;
    params_v2.sum_scratch_ptr = (float*) sum_scratch_d;

    // Scratch storage for output.
    params_v1.o_scratch_ptr = (int*) o_scratch_d;
    params_v2.o_scratch_ptr = (int*) o_scratch_d;

    // Tile id counter for dynamic scheduling
    params_v2.tile_id_counter_ptr = (uint32_t*) tile_id_counter_d;
    // params_paged_v2.tile_id_counter_ptr = (uint32_t*) tile_id_counter_d;

    if (sage_block_size_q > 0 || sage_block_size_k > 0 || sage_block_size_v > 0)
    {
        assert(input_layout == Attention_input_layout::PACKED_QKV && "for now this test only supports PACKED_QKV");
        assert(d == dv && "for now SageAttention doesn't support different QKV dims");
        assert(((sm == 90 && !force_non_warp_specialization) || (sm == 89))
            && "only hopper and ada kernels support SageAttention");
        fmha::e4m3_t* quant_qkv;
        FMHA_CHECK_CUDA(cudaMalloc((void**) &quant_qkv, qkv_packed_size));
        params_v2.sage.q.block_size = sage_block_size_q;
        params_v2.sage.q.max_nblock = (s + sage_block_size_q - 1) / sage_block_size_q;
        FMHA_CHECK_CUDA(
            cudaMalloc((void**) &params_v2.sage.q.scales, params_v2.sage.q.max_nblock * h * b * sizeof(float)));
        params_v2.sage.k.block_size = sage_block_size_k;
        params_v2.sage.k.max_nblock = (s + sage_block_size_k - 1) / sage_block_size_k;
        FMHA_CHECK_CUDA(
            cudaMalloc((void**) &params_v2.sage.k.scales, params_v2.sage.k.max_nblock * h * b * sizeof(float)));
        params_v2.sage.v.block_size = sage_block_size_v;
        params_v2.sage.v.max_nblock = (s + sage_block_size_v - 1) / sage_block_size_v;
        FMHA_CHECK_CUDA(
            cudaMalloc((void**) &params_v2.sage.v.scales, params_v2.sage.v.max_nblock * h * b * sizeof(float)));
#if 1
        {
            // simple test, all scales are the same
            constexpr float const_scale = 0.618f;
            fmha::e4m3_t* quant_qkv_h = (fmha::e4m3_t*) malloc(qkv_packed_size);
            for (size_t i = 0; i < qkv_packed_size; i++)
            {
                quant_qkv_h[i] = fmha::e4m3_t(qkv_packed_h[i] / const_scale);
            }
            FMHA_CHECK_CUDA(cudaMemcpy(quant_qkv, quant_qkv_h, qkv_packed_size, cudaMemcpyHostToDevice));
            free(quant_qkv_h);
            auto init_scales = [&](bert::Fused_multihead_attention_params_v2::SageAttention::Scales& x)
            {
                std::vector<float> scales(x.max_nblock * h * b, const_scale);
                FMHA_CHECK_CUDA(
                    cudaMemcpy(x.scales, scales.data(), sizeof(float) * scales.size(), cudaMemcpyHostToDevice));
            };
            init_scales(params_v2.sage.q);
            init_scales(params_v2.sage.k);
            init_scales(params_v2.sage.v);
        }
#else
        {
            // use external quant kernel
            run_sage_quant(b, h, d, s, params_v2.qkv_ptr,
                (char*) params_v2.qkv_ptr + get_size_in_bytes(h * d, data_type),
                (char*) params_v2.qkv_ptr + get_size_in_bytes(2 * h * d, data_type,
                params_v2.q_stride_in_bytes,
                params_v2.k_stride_in_bytes,
                params_v2.v_stride_in_bytes,
                params_v2.cu_q_seqlens, params_v2.cu_kv_seqlens, sage_block_size_q, sage_block_size_k,
                sage_block_size_v, quant_qkv, quant_qkv + h * d, quant_qkv + 2 * h * d, params_v2.sage.q.scales,
                params_v2.sage.k.scales, params_v2.sage.v.scales);
        }
#endif
        // no need to free old params_v2.qkv_ptr, it will be released in the end
        params_v2.qkv_ptr = quant_qkv;
        params_v2.q_stride_in_bytes = params_v2.k_stride_in_bytes = params_v2.v_stride_in_bytes
            = get_size_in_bytes((h + 2 * h_kv) * d, DATA_TYPE_E4M3);
    }

#if defined(DEBUG_HAS_PRINT_BUFFER)
    auto& params = params_v2;
    constexpr size_t bytes = 32 * 1024;
    void* print_ptr = nullptr;
    FMHA_CHECK_CUDA(cudaMalloc(&params.print_ptr, bytes));
    std::vector<float> print_buffer(bytes / sizeof(float));
#endif
    // Run a few warm-up kernels.
    for (int ii = 0; ii < warm_up_runs; ++ii)
    {
        if (v1)
        {
            run_fmha_v1(params_v1, launch_params, data_type, output_dtype, sm, 0);
        }
        else
        {
            run_fmha_v2(params_v2, launch_params, data_type, output_dtype, sm, 0);
        }
    }
    FMHA_CHECK_CUDA(cudaPeekAtLastError());

    float non_fused_elapsed = INFINITY;
    if (!skip_checks)
    {
        // Run cuBLAS.

        RefBMM bmm1(data_type_to_cuda(data_type), // a
            data_type_to_cuda(data_type),         // b
            data_type_to_cuda(acc_type),          // d
            data_type_to_cublas(acc_type),        // compute
            data_type_to_cuda(acc_type),          // scale
            false,                                // Q
            true,                                 // K'
            s,                                    // m
            s,                                    // n
            d,                                    // k
            b * h * (2 * d + dv),                 // ld Q
            b * h * (2 * d + dv),                 // ld K
            b * h * s,                            // ld P
            (2 * d + dv),                         // stride Q
            (2 * d + dv),                         // stride K
            s,                                    // stride P
            b * h                                 // batch count
        );

        /*
        RefBMM bmm2(data_type_to_cuda(data_type), // a
                    data_type_to_cuda(data_type), // b
                    data_type_to_cuda(acc_type), // d
                    data_type_to_cublas(acc_type), //compute
                    data_type_to_cuda(acc_type), // scale
                    false, // S
                    false, // V
                    s, // m
                    d, // n
                    s, // k
                    b * h * s, // ld S
                    b * h * 3 * d, // ld V
                    b * h * d, // ld O
                    s, // stride S
                    3 * d, // stride V
                    d, // stride O
                    b * h // batch count
                   );
        */

        // WAR fOR MISSING CUBLAS FP8 NN SUPPORT.
        // Transpose V, so that we can do a TN BMM2, i.e. O = S x V'  instead of O = S x V.
        RefBMM bmm2(data_type_to_cuda(data_type), // a
            data_type_to_cuda(data_type),         // b
            data_type_to_cuda(acc_type),          // d
            data_type_to_cublas(acc_type),        // compute
            data_type_to_cuda(acc_type),          // scale
            false,                                // S
            true,                                 // V'
            s,                                    // m
            dv,                                   // n
            s,                                    // k
            b * h * s,                            // ld S
            s,                                    // ld V
            b * h * dv,                           // ld O
            s,                                    // stride S
            s * dv,                               // stride V
            dv,                                   // stride O
            b * h                                 // batch count
        );
        timer.start();
        ground_truth(bmm1, bmm2, data_type, acc_type, scale_bmm1, scale_softmax, scale_bmm2, softcapping_scale_bmm1,
            qkv_sbh3d_d,
            vt_d, // WAR pass in V'
            mask_d, attention_sinks_d, p_d, s_d, tmp_d, o_d, softmax_stats_d, cu_seqlens_d, b, s, h, d, dv, runs,
            warps_m, warps_n, has_alibi);
        timer.stop();
        FMHA_CHECK_CUDA(cudaPeekAtLastError());
        FMHA_CHECK_CUDA(cudaDeviceSynchronize());
        non_fused_elapsed = timer.millis();

#if defined(STORE_P)
        FMHA_CHECK_CUDA(cuda_memcpy_d2h(p_ref_h, p_d, p_size, acc_type));
#endif // defined(STORE_P)

#if defined(STORE_S)
        FMHA_CHECK_CUDA(cuda_memcpy_d2h(s_ref_h, s_d, p_size, data_type));
#endif // defined(STORE_S)

        // Read the results.
        FMHA_CHECK_CUDA(cuda_memcpy_d2h(o_ref_h, o_d, o_size, data_type));
        FMHA_CHECK_CUDA(cuda_memcpy_d2h(softmax_stats_ref_h, softmax_stats_d, 2 * b * s * h, DATA_TYPE_FP32));
    }

    // Fill-in p/s/o with garbage data.
    // WAR: if sequence is padded, we zero-fill the output buffer as kernel will not write to the
    // padded area, and the host expects to check the padded area
    if (!skip_checks)
    {
        FMHA_CHECK_CUDA(cudaMemset(p_d, 0xdc, p_size_in_bytes));
        FMHA_CHECK_CUDA(cudaMemset(s_d, 0xdc, s_size_in_bytes));
    }
    FMHA_CHECK_CUDA(cudaMemset(o_d, 0x00, o_size_in_bytes));
    FMHA_CHECK_CUDA(cudaMemset(softmax_stats_d, 0x00, 2 * b * s * h * sizeof(float)));

    // Run the kernel.
    timer.start();
    for (int ii = 0; ii < runs; ++ii)
    {
        if (v1)
        {
            run_fmha_v1(params_v1, launch_params, data_type, output_dtype, sm, 0);
        }
        else
        {
            run_fmha_v2(params_v2, launch_params, data_type, output_dtype, sm, 0);
        }
    }
    timer.stop();
    FMHA_CHECK_CUDA(cudaPeekAtLastError());

    FMHA_CHECK_CUDA(cudaDeviceSynchronize());
    float fused_elapsed = timer.millis();

#if defined(STORE_P)
    FMHA_CHECK_CUDA(cuda_memcpy_d2h(p_h, p_d, p_size, acc_type));
    printf("\nChecking .....: P = norm * K^T * Q\n");

    // DEBUG.
    printf("seqlens[0]=%d\n", seqlens[0]);
    // END OF DEBUG.

    // Clear the invalid region of P.
    set_mat<float>(p_ref_h, seqlens, s, b, h, s, 0.f, true);
    set_mat<float>(p_h, seqlens, s, b, h, s, 0.f, true);

    // Do the check.
    check_results(p_h, p_ref_h, s, s * b * h, s, 0.f, true, true);
#endif // defined(STORE_P)

#if defined(STORE_S)
    FMHA_CHECK_CUDA(cuda_memcpy_d2h(s_h, s_d, p_size, data_type));
    printf("\nChecking .....: S = softmax(P)\n");
#if defined(USE_SAME_SUM_ORDER_IN_SOFTMAX_AS_REF_CODE)
    float softmax_epsilon = data_type == DATA_TYPE_FP16 ? 1e-3f : 0.f;
#else
    float softmax_epsilon = 1.e-3f;
#endif // defined(USE_SAME_SUM_ORDER_IN_SOFTMAX_AS_REF_CODE)

    // Clear the invalid region of S.
    set_mat<float>(s_ref_h, seqlens, s, b, h, s, 0.f);
    set_mat<float>(s_h, seqlens, s, b, h, s, 0.f);

    // Do the check.
    check_results(s_h, s_ref_h, s, s * b * h, s, softmax_epsilon, true, true);
#endif // defined(STORE_S)

    // Check the final results.
    int status = -1;
    if (skip_checks)
    {
        status = 0;
        printf("\n");
        print_results(true, false);
    }
    else
    {
        if (v1)
        {
            FMHA_CHECK_CUDA(cuda_memcpy_d2h(o_h, o_d, o_size, output_dtype));
            status = check_results(o_h, o_ref_h, d, s * b * h, d, epsilon, verbose, true);
        }
        else
        {
            std::vector<float> o_ref_trans_h(o_size);

            FMHA_CHECK_CUDA(cuda_memcpy_d2h(o_h, o_d_view, o_view_size, output_dtype));
            FMHA_CHECK_CUDA(cuda_memcpy_d2h(softmax_stats_h, softmax_stats_d, 2 * b * s * h, DATA_TYPE_FP32));

            if (interleaved)
            {
                // revert batch-interleaved format: 3 x h/32 x total x d x 32 => total x
                // h x 3 x d
                x_vec32(false, o_h, h, is_s_padded ? b * h : total, 1);
            }

            // Extract the last s_q tokens from the output.
            extract_and_transpose_output<float>(
                o_ref_trans_h.data(), o_ref_h, seqlens, q_seqlens, s, s_q, b, h, dv, is_s_padded);
            if (verbose)
            {
                printf("\nChecking .....: O = V * S\n");
            }
            status = check_results(o_h, o_ref_trans_h.data(), dv, is_s_padded ? s_q * b * h : cu_q_seqlens.back() * h,
                dv, epsilon, verbose, true);
            if (save_softmax)
            {
                auto errors = check_softmax_results(softmax_stats_h, softmax_stats_ref_h, b, s, h, seqlens, cu_seqlens);
                status = status | ((errors.first + errors.second) > 0);
            }
        }
        if (status != 0)
        { // if there was an error, print the config of the run
            printf("v1=%d il=%d s=%lu b=%lu h=%lu dv=%lu dtype=%s\n", v1, interleaved, s, b, h, dv,
                data_type_to_name(data_type).c_str());
        }
        if (!verbose)
        { // this just prints the SUCCESS/ERROR line
            print_results(true, true, status == 0);
        }
    }

    // accounts for tensor core flops only; excludes flops spent in softmax
    size_t total_flops = 0;
    // remove last seqlen(total_seqlen)
    seqlens.pop_back();
    for (auto& s_ : seqlens)
    {
        size_t s_size = size_t(s_);
        total_flops += 2ull * h * (s_q * s_size * d + s_q * dv * s_size); // 1st BMM + 2nd BMM
    }
    total_flops = attention_mask_type == Attention_mask_type::CAUSAL ? total_flops / 2 : total_flops;

    size_t total_bytes = o_packed_size_in_bytes + qkv_packed_size_in_bytes;
    if (verbose)
    {
        // Runtimes.
        printf("\n");
        if (!skip_checks)
        {
            printf("Non-fused time: %.6f ms\n", non_fused_elapsed / float(runs));
        }
        printf("Fused time ...: %.6f us\n", fused_elapsed * 1000 / float(runs));
        printf("Tensor core ..: %.2f Tflop/s\n", total_flops / (fused_elapsed / float(runs) / 1e-9));
        printf("Bandwidth ....: %.2f GB/s\n", total_bytes / (fused_elapsed / float(runs) / 1e-6));
        if (!skip_checks)
        {
            printf("Ratio ........: %.2fx\n", non_fused_elapsed / fused_elapsed);
        }
    }
    else
    {
        printf("Elapsed ......: %.6f us (%.2fx), %.2f Tflop/s, %.2f GB/s\n", fused_elapsed * 1000 / float(runs),
            non_fused_elapsed / fused_elapsed, total_flops / (fused_elapsed / float(runs) / 1e-9),
            total_bytes / (fused_elapsed / float(runs) / 1e-6));
    }
#if defined(DEBUG_HAS_PRINT_BUFFER)
    FMHA_CHECK_CUDA(cuda_memcpy_d2h(print_buffer.data(), params.print_ptr, print_buffer.size(), DATA_TYPE_FP32));

    printf("\n====================\n");
    for (int it = 0; it < 16; it++)
    {
        printf("% .4f ", print_buffer[it]);
    }
    printf("\n====================\n");

    FMHA_CHECK_CUDA(cudaFree(params.print_ptr));

#endif
    // Release memory.
    FMHA_CHECK_CUDA(cudaFree(qkv_sbh3d_d));
    FMHA_CHECK_CUDA(cudaFree(qkv_packed_d));
    FMHA_CHECK_CUDA(cudaFree(scale_bmm2_d));
    FMHA_CHECK_CUDA(cudaFree(mqa_qkv_d));
    FMHA_CHECK_CUDA(cudaFree(mqa_qkv_packed_d));
    FMHA_CHECK_CUDA(cudaFree(qkv_bsh3d_d));
    FMHA_CHECK_CUDA(cudaFree(mask_d));
    FMHA_CHECK_CUDA(cudaFree(packed_mask_d));
    FMHA_CHECK_CUDA(cudaFree(q_d));
    FMHA_CHECK_CUDA(cudaFree(k_d));
    FMHA_CHECK_CUDA(cudaFree(v_d));
    FMHA_CHECK_CUDA(cudaFree(p_d));
    FMHA_CHECK_CUDA(cudaFree(s_d));
    FMHA_CHECK_CUDA(cudaFree(o_d));
    FMHA_CHECK_CUDA(cudaFree(tmp_d));
    FMHA_CHECK_CUDA(cudaFree(cu_seqlens_d));
    FMHA_CHECK_CUDA(cudaFree(cu_mask_rows_d));
    FMHA_CHECK_CUDA(cudaFree(max_scratch_d));
    FMHA_CHECK_CUDA(cudaFree(sum_scratch_d));
    FMHA_CHECK_CUDA(cudaFree(o_scratch_d));
    FMHA_CHECK_CUDA(cudaFree(counters_d));
    FMHA_CHECK_CUDA(cudaFree(tile_id_counter_d));
    FMHA_CHECK_CUDA(cudaFree(kv_cache_pool_ptr));
    FMHA_CHECK_CUDA(cudaFree(kv_cache_block_offsets_d));
    FMHA_CHECK_CUDA(cudaFree(contiguous_kv_d));
    FMHA_CHECK_CUDA(cudaFree(softmax_stats_d));

    free(qkv_h);
    free(mask_h);
    free(packed_mask_h);
    free(s_h);
    free(o_h);
    free(o_ref_h);
    free(softmax_stats_h);
    free(softmax_stats_ref_h);
    free(contiguous_kv_h);
    free(kv_cache_ptrs_h);
    free(kv_cache_block_offsets_h);

    free(p_ref_h);
#if defined(STORE_P)
    free(p_h);
#endif // defined(STORE_P)
    free(s_ref_h);

    return status;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
