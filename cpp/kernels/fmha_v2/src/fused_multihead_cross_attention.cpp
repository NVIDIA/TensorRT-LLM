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
#include <fused_multihead_cross_attention_api.h>
#include <iostream>
#include <math.h>
#include <numeric>
#include <string>
#include <vector>

using Launch_params = bert::Fused_multihead_attention_launch_params;

////////////////////////////////////////////////////////////////////////////////////////////////////

void run_softmax_fp32(void* dst, void const* src, void const* mask, void const* attention_sinks, void* softmax_sum_d,
    void* cu_seqlens_q_d, int s_inner, int s_outer, int b, int h, float softcapping_scale_bmm1, int warps_n,
    bool has_alibi);

////////////////////////////////////////////////////////////////////////////////////////////////////

void run_softmax_e4m3(void* dst, void const* src, void const* mask, void const* attention_sinks, void* softmax_sum_d,
    void* cu_seqlens_q_d, int s_inner, int s_outer, int b, int h, float scale_softmax, float softcapping_scale_bmm1,
    int warps_n, bool has_alibi);

////////////////////////////////////////////////////////////////////////////////////////////////////

void run_softmax_fp16(void* dst, void const* src, void const* mask, void const* attention_sinks, void* softmax_sum_d,
    void* cu_seqlens_q_d, int s_inner, int s_outer, int b, int h, float softcapping_scale_bmm1, int warps_n,
    bool has_alibi);

////////////////////////////////////////////////////////////////////////////////////////////////////

void run_softmax_int8(void* dst, void const* src, void const* mask, void const* attention_sinks, void* softmax_sum_d,
    void* cu_seqlens_q_d, int s_inner, int s_outer, int b, int h, float scale_i2f, float scale_f2i,
    float softcapping_scale_bmm1, int warps_n, bool has_alibi);

////////////////////////////////////////////////////////////////////////////////////////////////////

void run_conversion_int32_to_int8(void* dst, void const* src, int s, int b, int h, int d, float scale);

////////////////////////////////////////////////////////////////////////////////////////////////////

void run_conversion_fp32_to_fp16(void* dst, void const* src, int s, int b, int h, int d);

////////////////////////////////////////////////////////////////////////////////////////////////////

void run_conversion_fp32_to_e4m3(void* dst, void const* src, int s, int b, int h, int d, float scale_o);

////////////////////////////////////////////////////////////////////////////////////////////////////

void ground_truth(RefBMM& bmm1, RefBMM& bmm2, const Data_type data_type, const Data_type acc_type,
    float const scale_bmm1, float const scale_softmax, float const scale_bmm2, void* q_d, void* kv_d, void* vt_d,
    void* mask_d, void* p_d, void* s_d, void* tmp_d, void* o_d, void* softmax_sum_d, void* cu_seqlens_q_d,
    const size_t b, const size_t s_q, const size_t s_kv, const size_t h, const size_t d, int const runs,
    int const warps_m, int const warps_n, bool has_alibi)
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
        bmm1(static_cast<char*>(q_d) + 0 * qkv_stride, static_cast<char*>(kv_d) + 0 * qkv_stride, p_d, &alpha, &beta,
            stream);

        // Softmax.
        if (data_type == DATA_TYPE_FP16 && acc_type == DATA_TYPE_FP16)
        {
            run_softmax_fp16(
                s_d, p_d, mask_d, nullptr, softmax_sum_d, cu_seqlens_q_d, s_kv, s_q, b, h, 0.f, warps_n, has_alibi);
        }
        else if (data_type == DATA_TYPE_FP16 && acc_type == DATA_TYPE_FP32)
        {
            run_softmax_fp32(
                s_d, p_d, mask_d, nullptr, softmax_sum_d, cu_seqlens_q_d, s_kv, s_q, b, h, 0.f, warps_n, has_alibi);
        }
        else if (data_type == DATA_TYPE_E4M3 && acc_type == DATA_TYPE_FP32)
        {
            run_softmax_e4m3(s_d, p_d, mask_d, nullptr, softmax_sum_d, cu_seqlens_q_d, s_kv, s_q, b, h, scale_softmax,
                0.f, warps_n, has_alibi);
        }
        else if (data_type == DATA_TYPE_INT8 && acc_type == DATA_TYPE_INT32)
        {
            run_softmax_int8(s_d, p_d, mask_d, nullptr, softmax_sum_d, cu_seqlens_q_d, s_kv, s_q, b, h, scale_bmm1,
                scale_softmax, 0.f, warps_n, has_alibi);
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
            run_conversion_fp32_to_fp16(o_d, out_d, s_q, b, h, d);
        }
        else if (data_type == DATA_TYPE_E4M3 && acc_type == DATA_TYPE_FP32)
        {
            run_conversion_fp32_to_e4m3(o_d, out_d, s_q, b, h, d, scale_bmm2);
        }
        else if (data_type == DATA_TYPE_INT8 && acc_type == DATA_TYPE_INT32)
        {
            // quantize output in second step
            run_conversion_int32_to_int8(o_d, out_d, s_q, b, h, d, scale_bmm2);
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

static inline void set_params(bert::Fused_multihead_attention_params_mhca& params,
    // types
    Data_type data_type, Data_type acc_type,
    // sizes
    const size_t b, const size_t s_q, const size_t s_kv, const size_t h, const size_t d, const size_t d_padded,
    const size_t total,
    // device pointers
    void* q_packed_d, void* kv_packed_d, void* cu_seqlens_q_d, void* cu_seqlens_kv_d, void* o_packed_d, void* p_d,
    void* s_d,
    // scale factors
    float const scale_bmm1, float const scale_softmax, float const scale_bmm2,
    // flags
    bool const use_int8_scale_max)
{
    memset(&params, 0, sizeof(params));

    // Set the pointers.
    params.o_ptr = o_packed_d;
    params.o_stride_in_bytes = get_size_in_bytes(h * d, data_type);

    // if( interleaved ) {
    //     params.qkv_stride_in_bytes = total;
    //     params.o_stride_in_bytes = total;
    // }

#if defined(STORE_P)
    params.p_ptr = p_d;
    params.p_stride_in_bytes = get_size_in_bytes(b * h * s_kv, acc_type);
#endif // defined(STORE_P)

#if defined(STORE_S)
    params.s_ptr = s_d;
    params.s_stride_in_bytes = get_size_in_bytes(b * h * s_kv, data_type);
#endif // defined(STORE_S)

    // Set the dimensions.
    params.b = b;
    params.h = h;
    params.s_q = s_q;
    params.s = s_kv;
    params.d = d;
    params.d_padded = d_padded;

    // Set the different scale values.
    Data_type scale_type1 = data_type == DATA_TYPE_FP16 ? acc_type : DATA_TYPE_FP32;
    Data_type scale_type2 = data_type == DATA_TYPE_FP16 ? DATA_TYPE_FP16 : DATA_TYPE_FP32;

    set_alpha(params.scale_bmm1, scale_bmm1, scale_type1);
    set_alpha(params.scale_softmax, scale_softmax, scale_type1);
    set_alpha(params.scale_bmm2, scale_bmm2, scale_type2);

    // Set the pointers.
    params.gmem_q_params.ptr = q_packed_d;
    params.gmem_q_params.stride_in_bytes = get_size_in_bytes(h * d, data_type);
    params.gmem_q_params.h = h;
    params.gmem_q_params.d = d;
    params.gmem_q_params.cu_seqlens = static_cast<int*>(cu_seqlens_q_d);

    params.gmem_kv_params.ptr = kv_packed_d;
    params.gmem_kv_params.stride_in_bytes = get_size_in_bytes(h * 2 * d, data_type);
    params.gmem_kv_params.h = h;
    params.gmem_kv_params.d = d;
    params.gmem_kv_params.cu_seqlens = static_cast<int*>(cu_seqlens_kv_d);

    // Set flags
    params.use_int8_scale_max = use_int8_scale_max;

    // Do we enable the trick to replace I2F with FP math in the 2nd GEMM?
    if (data_type == DATA_TYPE_INT8)
    {
        params.enable_i2f_trick
            = -double(1 << 22) * double(scale_bmm2) <= -128.f && double(1 << 22) * double(scale_bmm2) >= 127.f;
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
    // The length of the sequence for query tokens
    size_t s_q = 4096;
    // The length of the sequence for K/V cross attention tokens
    size_t s_kv = 77;

    // The data type of the kernel.
    Data_type data_type = DATA_TYPE_FP16;
    // The type of the intermediate P matrix.
    Data_type acc_type = DATA_TYPE_FP16;
    // The scaling factors.
    float scale_bmm1 = 0.f, scale_softmax = 0.f, scale_bmm2 = 0.25f;
    // The number of runs.
    int runs = 1, warm_up_runs = 0;
    // Do we use 1s for Q, K, V.
    bool use_1s_q = false, use_1s_k = false, use_1s_v = false, use_1s_mask = false;
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

    // minimum sequence length for sampling variable seqlens
    uint32_t min_s = s_q;

    // run interleaved kernels and transpose input and output accordingly
    bool interleaved = false;
    bool ignore_b1opt = false;
    bool force_unroll = true;
    bool use_int8_scale_max = false;
    bool verbose = true;

    // set all sequence lengths to min(s, min_s)
    bool fix_s = true;

    bool v1 = false;

    // use TMA or not. ignored if not in SM90
    bool use_tma = false;

    // Read the parameters from the command-line.
    for (int ii = 1; ii < argc; ++ii)
    {
        if (!strcmp(argv[ii], "-1s"))
        {
            use_1s_k = use_1s_q = use_1s_v = use_1s_mask = true;
        }
        else if (!strcmp(argv[ii], "-1s-k"))
        {
            use_1s_k = true;
        }
        else if (!strcmp(argv[ii], "-1s-mask"))
        {
            use_1s_mask = true;
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
        else if (!strcmp(argv[ii], "-e4m3"))
        {
            data_type = DATA_TYPE_E4M3;
            // Technically not the acc type.
            acc_type = DATA_TYPE_FP32;
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
        else if (!strcmp(argv[ii], "-s-q") && ++ii < argc)
        {
            s_q = strtol(argv[ii], nullptr, 10);
        }
        else if (!strcmp(argv[ii], "-s-kv") && ++ii < argc)
        {
            s_kv = strtol(argv[ii], nullptr, 10);
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
        else if (!strcmp(argv[ii], "-ignore-b1opt"))
        {
            ignore_b1opt = true;
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
        else if (!strcmp(argv[ii], "-use-tma"))
        {
            use_tma = true;
        }
        else
        {
            fprintf(stderr, "Unrecognized option: %s. Aborting!\n", argv[ii]);
            return -1;
        }
    }

    if (interleaved)
    {
        throw std::runtime_error("Interleaved layout is not supported!");
    }

    min_s = std::min<uint32_t>(s_q, min_s);

    // The padded sizes.
    int const s_kv_padded = std::pow(2, std::ceil(std::log(s_kv) / std::log(2)));
    int const d_padded = std::pow(2, std::ceil(std::log(d) / std::log(2)));

    // Set the norm.
    if (scale_bmm1 == 0.f)
    {
        scale_bmm1 = 1.f / sqrtf((float) d);
    }

    // Force the softmax scale to 1.f for the FP16 kernel.
    if (data_type == DATA_TYPE_FP16)
    {
        scale_softmax = 1.f;
    }
    else if (data_type == DATA_TYPE_INT8 && scale_softmax == 0.f)
    {
        scale_softmax = std::max(512.f, (float) s_kv);
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
        scale_v = data_type == DATA_TYPE_FP16 ? 0.125f : 1.f;
    }

    // Set the tolerance if not already set by the user.
    if (epsilon < 0.f)
    {
        epsilon = data_type == DATA_TYPE_FP16 ? 0.015f : 0.f;
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
        printf("Seq len Q ....: %lu\n", s_q);
        printf("Seq len KV ...: %lu\n", s_kv);
        printf("Warm-up runs .: %d\n", warm_up_runs);
        printf("Runs..........: %d\n\n", runs);

        // The scaling factors for the 3 operations.
        printf("Scale bmm1 ...: %.6f\n", scale_bmm1);
        printf("Scale softmax.: %.6f\n", scale_softmax);
        printf("Scale bmm2 ...: %.6f\n", scale_bmm2);
        printf("\n");
    }

    Launch_params launch_params;
    // Set launch params to choose kernels
    launch_params.interleaved = interleaved;
    launch_params.ignore_b1opt = ignore_b1opt;
    launch_params.force_unroll = force_unroll;
    launch_params.use_tma = use_tma;

    // The Q matrix of size S_Q x B x H x D.
    const size_t q_size = s_q * b * h * d;
    // The K and V matrices are packed into one big matrix of size S_KV x B x H x 2 x D.
    const size_t kv_size = s_kv_padded * b * h * 2 * d;
    // Allocate on the host.
    float* q_h = (float*) malloc(q_size * sizeof(float));
    // Allocate on the host.
    float* kv_h = (float*) malloc(kv_size * sizeof(float));
    // The size in bytes.
    const size_t q_size_in_bytes = get_size_in_bytes(q_size, data_type);
    // The size in bytes.
    const size_t kv_size_in_bytes = get_size_in_bytes(kv_size, data_type);
    // Allocate on the device.
    void* q_d = nullptr;
    FMHA_CHECK_CUDA(cudaMalloc(&q_d, q_size_in_bytes));
    // Allocate on the device.
    void* kv_d = nullptr;
    FMHA_CHECK_CUDA(cudaMalloc(&kv_d, kv_size_in_bytes));

    // The mask for dropout.
    const size_t mask_size = s_q * b * s_kv_padded;
    // Allocate on the host.
    float* mask_h = (float*) malloc(mask_size * sizeof(float));
    // The size in bytes.
    const size_t mask_size_in_bytes = get_size_in_bytes(mask_size, DATA_TYPE_INT8);
    // Allocate on the device.
    void* mask_d = nullptr;
    FMHA_CHECK_CUDA(cudaMalloc(&mask_d, mask_size_in_bytes));

    // The decomposition of threads and warps for BMM1.
    size_t warps_m, warps_n, warps_k;
    std::tie(warps_m, warps_n, warps_k) = get_warps(launch_params, sm, data_type, s_kv_padded, b, d_padded, v1 ? 1 : 2);

    // For multi-CTA cases, determine the size of the CTA wave.
    int heads_per_wave, ctas_per_head;
    get_grid_size(heads_per_wave, ctas_per_head, sm, data_type, b, s_kv_padded, h, d,
        false, // disable multi-cta kernels by default
        v1 ? 1 : 2);

    // The number of threads per CTA.
    const size_t threads_per_cta = warps_m * warps_n * warps_k * 32;
    // The number of mmas in the M dimension. We use one uint32_t per MMA in the M dimension.
    const size_t mmas_m = (s_q + 16 * warps_m - 1) / (16 * warps_m);
    // The number of mmas in the N dimension.
    const size_t mmas_n = (s_kv_padded + 16 * warps_n - 1) / (16 * warps_n);
    // We do not support more than 4 MMAS in the N dimension (as each MMA needs 8 bits in the mask).
    assert(!v1 || mmas_n <= 4);
    // The packed mask for dropout (in the fused kernel). Layout is B * MMAS_M * THREADS_PER_CTA.
    const size_t packed_mask_size = b * mmas_m * threads_per_cta;
    // The size in bytes.
    const size_t packed_mask_size_in_bytes = packed_mask_size * sizeof(uint32_t);
    // Allocate on the host.
    uint32_t* packed_mask_h = (uint32_t*) malloc(packed_mask_size_in_bytes);
    // Allocate on the device.
    void* packed_mask_d = nullptr;

    // The O matrix is packed as S_Q * B * H * D.
    const size_t o_size = s_q * b * h * d;
    // Allocate on the host.
    float* o_h = (float*) malloc(o_size * sizeof(float));
    // The size in bytes.
    const size_t o_size_in_bytes = get_size_in_bytes(o_size, data_type);
    // Allocate on the device.
    void* o_d = nullptr;
    FMHA_CHECK_CUDA(cudaMalloc(&o_d, o_size_in_bytes));
    void* softmax_sum_d;
    FMHA_CHECK_CUDA(cudaMalloc(&softmax_sum_d, sizeof(float) * b * s_q * h));
    FMHA_CHECK_CUDA(cudaMemset(softmax_sum_d, 0x00, sizeof(float) * b * s_q * h));
    void* softmax_max_d;
    FMHA_CHECK_CUDA(cudaMalloc(&softmax_max_d, sizeof(float) * b * s_q * h));
    FMHA_CHECK_CUDA(cudaMemset(softmax_max_d, 0x00, sizeof(float) * b * s_q * h));

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

    // The P matrix is stored as one big matrix of size S_Q x B x H x S_KV.
    const size_t p_size = s_q * b * h * s_kv_padded;
    // The size in bytes.
    const size_t p_size_in_bytes = get_size_in_bytes(p_size, acc_type);
    // Allocate on the device.
    void* p_d = nullptr;
    FMHA_CHECK_CUDA(cudaMalloc(&p_d, p_size_in_bytes));

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
    FMHA_CHECK_CUDA(cudaMalloc(&s_d, s_size_in_bytes));

    // Allocate the reference on the host.
    float* s_ref_h = (float*) malloc(p_size * sizeof(float));

    // Allocate on the host.
    float* s_h = (float*) malloc(p_size * sizeof(float));
    // Make sure we set the seed for reproducible results.
    srand(1234UL);

    // Set the Q, K and V matrices.
    random_init("Q", q_h, d, s_q * b * h, d, use_1s_q, range_q, scale_q, verbose);
    random_init("K", kv_h + 0 * d, d, s_kv_padded * b * h, 2 * d, use_1s_k, range_k, scale_k, verbose);
    random_init("V", kv_h + 1 * d, d, s_kv_padded * b * h, 2 * d, use_1s_v, range_v, scale_v, verbose);

    //   WAR fOR MISSING CUBLAS FP8 NN SUPPORT.
    //   Transpose V, so that we can do a TN BMM2, i.e. O = S x V'  instead of O = S x V.
    const size_t v_size = s_kv_padded * b * h * d;
    // The size in bytes.
    const size_t v_size_in_bytes = get_size_in_bytes(v_size, data_type);
    float* vt_h = (float*) malloc(v_size * sizeof(float));
    void* vt_d = nullptr;
    FMHA_CHECK_CUDA(cudaMalloc(&vt_d, v_size_in_bytes));
    for (size_t it = 0; it < v_size; it++)
    {
        // vt is B x H x D x S_KV
        size_t si = it % s_kv_padded;
        size_t di = (it / s_kv_padded) % d;
        size_t hi = ((it / s_kv_padded) / d) % h;
        size_t bi = (((it / s_kv_padded) / d) / h) % b;
        // kv is S_KV x B x H x 2 x D
        size_t kv_idx = si * b * h * 2 * d + bi * h * 2 * d + hi * 2 * d + 1 * d // index V here
            + di;
        vt_h[it] = kv_h[kv_idx];
    }
    FMHA_CHECK_CUDA(cuda_memcpy_h2d(vt_d, vt_h, v_size, data_type));

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
    FMHA_CHECK_CUDA(cuda_memcpy_h2d(q_d, q_h, q_size, data_type));
    FMHA_CHECK_CUDA(cuda_memcpy_h2d(kv_d, kv_h, kv_size, data_type));

    // Create the buffer of mask.
    // if(verbose) {printf("Init .........: mask\n"); }
    // random_init_with_zeroes_or_ones(mask_h, b*s, use_1s_mask, 1.f - dropout, verbose);

    auto const create_seqlen
        = [min_s, fix_s, b](int s, std::vector<uint32_t>& seqlens, std::vector<int>& cu_seqlens, void** cu_seqlens_d)
    {
        std::transform(seqlens.begin(), seqlens.end(), seqlens.begin(),
            [=](const uint32_t)
            {
                if (fix_s)
                {
                    return std::min(uint32_t(s), min_s);
                }
                throw std::runtime_error("Not supported");
                // if( s_q == min_s ) {
                //     return min_s;
                // }
                // uint32_t s_ = s_q - min_s + 1;
                // uint32_t ret = min_s + (rand() % s_);
                // assert(ret <= s_q);
                // return ret;
            });

        // Compute the prefix sum of the sequence lengths.
        for (int it = 0; it < b; it++)
        {
            cu_seqlens[it + 1] = cu_seqlens[it] + seqlens[it];
        }

        FMHA_CHECK_CUDA(cudaMalloc(cu_seqlens_d, sizeof(int) * cu_seqlens.size()));
        FMHA_CHECK_CUDA(
            cudaMemcpy(*cu_seqlens_d, cu_seqlens.data(), sizeof(int) * cu_seqlens.size(), cudaMemcpyHostToDevice));
    };

    std::vector<uint32_t> seqlens_q(b, 0); // randomly draw a batch of sequence lengths >= min_s
    std::vector<int> cu_seqlens_q(b + 1, 0);
    // transfer to device
    void* cu_seqlens_q_d;

    std::vector<uint32_t> seqlens_kv(b, 0); // randomly draw a batch of sequence lengths >= min_s
    std::vector<int> cu_seqlens_kv(b + 1, 0);
    // transfer to device
    void* cu_seqlens_kv_d;

    create_seqlen(s_q, seqlens_q, cu_seqlens_q, &cu_seqlens_q_d);
    int total_q = cu_seqlens_q.back();
    create_seqlen(s_kv, seqlens_kv, cu_seqlens_kv, &cu_seqlens_kv_d);
    int total_kv = cu_seqlens_kv.back();

    size_t q_packed_size = cu_seqlens_q.back() * h * d;
    size_t kv_packed_size = cu_seqlens_kv.back() * h * 2 * d;
    size_t q_packed_size_in_bytes = get_size_in_bytes(q_packed_size, data_type);
    size_t kv_packed_size_in_bytes = get_size_in_bytes(kv_packed_size, data_type);
    void* q_packed_d = nullptr;
    FMHA_CHECK_CUDA(cudaMalloc(&q_packed_d, q_packed_size_in_bytes));

    void* kv_packed_d = nullptr;
    FMHA_CHECK_CUDA(cudaMalloc(&kv_packed_d, kv_packed_size_in_bytes));

    const size_t o_packed_size = cu_seqlens_q.back() * h * d;
    // Allocate on the host.
    float* o_packed_h = (float*) malloc(o_packed_size * sizeof(float));
    float* o_ref_packed_h = (float*) malloc(o_packed_size * sizeof(float));
    void* o_packed_d = nullptr;

    size_t o_packed_size_in_bytes = get_size_in_bytes(o_packed_size, data_type);
    FMHA_CHECK_CUDA(cudaMalloc(&o_packed_d, o_packed_size_in_bytes));

    std::vector<float> kv_packed_h(kv_packed_size);
    extract_and_transpose_input<float>(kv_packed_h.data(), kv_h, seqlens_kv, s_kv_padded, b, h, d, 2);
    if (interleaved)
    {
        x_vec32(true, kv_packed_h.data(), h, total_kv, 2);
    }

    std::vector<float> q_packed_h(q_packed_size);
    extract_and_transpose_input<float>(q_packed_h.data(), q_h, seqlens_q, s_q, b, h, d, 1);
    if (interleaved)
    {
        x_vec32(true, q_packed_h.data(), h, total_q, 1);
    }

    // printf("%f %f\n", qkv_packed_h[0], qkv_h[0]);
    FMHA_CHECK_CUDA(cuda_memcpy_h2d(q_packed_d, q_packed_h.data(), q_packed_size, data_type));
    FMHA_CHECK_CUDA(cuda_memcpy_h2d(kv_packed_d, kv_packed_h.data(), kv_packed_size, data_type));

    for (size_t so = 0; so < s_q; ++so)
    {
        for (size_t bi = 0; bi < b; ++bi)
        {
            int actual_seqlen_q = seqlens_q[bi];
            int actual_seqlen_kv = seqlens_kv[bi];
            for (size_t si = 0; si < s_kv_padded; ++si)
            {
                // Are both the query and the key inside the sequence?
                bool valid = si < actual_seqlen_kv && so < actual_seqlen_q;
                // The mask is stored as floats.
                mask_h[so * b * s_kv_padded + bi * s_kv_padded + si] = valid ? 1.f : 0.f;
            }
        }
    }

    // Copy the mask to the device.
    FMHA_CHECK_CUDA(cuda_memcpy_h2d(mask_d, mask_h, mask_size, DATA_TYPE_INT8));

    // Set the params.
    bert::Fused_multihead_attention_params_mhca params;
    set_params(params, data_type, acc_type, b, s_q, s_kv_padded, h, d, d_padded, total_kv, q_packed_d, kv_packed_d,
        cu_seqlens_q_d, cu_seqlens_kv_d, o_packed_d, p_d, s_d, scale_bmm1, scale_softmax, scale_bmm2,
        use_int8_scale_max);

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

    // The number of heads computed per wave.
    params.heads_per_wave = heads_per_wave;

    // Barriers for the global sync in the multi-CTA kernel(s).
    params.counters = (int*) counters_d + 0 * heads_per_wave;
    params.max_barriers = (int*) counters_d + 0 * heads_per_wave;
    params.sum_barriers = (int*) counters_d + 1 * heads_per_wave;
    params.locks = (int*) counters_d + 2 * heads_per_wave;

    // Scratch storage for softmax.
    params.max_scratch_ptr = (float*) max_scratch_d;
    params.sum_scratch_ptr = (float*) sum_scratch_d;

    // Scratch storage for output.
    params.o_scratch_ptr = (int*) o_scratch_d;

    // Run a few warm-up kernels.
    for (int ii = 0; ii < warm_up_runs; ++ii)
    {
        run_fmhca(params, launch_params, data_type, sm, 0);
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
            s_q,                                  // m
            s_kv_padded,                          // n
            d,                                    // k
            b * h * d,                            // ld Q
            b * h * 2 * d,                        // ld K
            b * h * s_kv_padded,                  // ld P
            d,                                    // stride Q
            2 * d,                                // stride K
            s_kv_padded,                          // stride P
            b * h                                 // batch count
        );

        // WAR fOR MISSING CUBLAS FP8 NN SUPPORT.
        // Transpose V, so that we can do a TN BMM2, i.e. O = S x V'  instead of O = S x V.
        RefBMM bmm2(data_type_to_cuda(data_type), // a
            data_type_to_cuda(data_type),         // b
            data_type_to_cuda(acc_type),          // d
            data_type_to_cublas(acc_type),        // compute
            data_type_to_cuda(acc_type),          // scale
            false,                                // S
            true,                                 // V'
            s_q,                                  // m
            d,                                    // n
            s_kv_padded,                          // k
            b * h * s_kv_padded,                  // ld S
            s_kv_padded,                          // ld V
            b * h * d,                            // ld O
            s_kv_padded,                          // stride S
            s_kv_padded * d,                      // stride V
            d,                                    // stride O
            b * h                                 // batch count
        );

        timer.start();
        ground_truth(bmm1, bmm2, data_type, acc_type, scale_bmm1, scale_softmax, scale_bmm2, q_d, kv_d,
            vt_d, // WAR pass in V'
            mask_d, p_d, s_d, tmp_d, o_d, softmax_sum_d, cu_seqlens_q_d, b, s_q, s_kv_padded, h, d, runs, warps_m,
            warps_n, false);
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
    }

    // Fill-in p/s/o with garbage data.
    FMHA_CHECK_CUDA(cudaMemset(p_d, 0xdc, p_size_in_bytes));
    FMHA_CHECK_CUDA(cudaMemset(s_d, 0xdc, s_size_in_bytes));
    FMHA_CHECK_CUDA(cudaMemset(o_d, 0xdc, o_size_in_bytes));

    // Run the kernel.
    timer.start();
    for (int ii = 0; ii < runs; ++ii)
    {
        run_fmhca(params, launch_params, data_type, sm, 0);
    }
    timer.stop();
    FMHA_CHECK_CUDA(cudaPeekAtLastError());

    FMHA_CHECK_CUDA(cudaDeviceSynchronize());
    float fused_elapsed = timer.millis();

#if defined(STORE_P)
    FMHA_CHECK_CUDA(cuda_memcpy_d2h(p_h, p_d, p_size, acc_type));
    printf("\nChecking .....: P = norm * K^T * Q\n");

    // Clear the invalid region of P.
    set_mat<float>(p_ref_h, seqlens_q, seqlens_kv, s_q, b, h, s_kv_padded, 0.f, true);
    set_mat<float>(p_h, seqlens_q, seqlens_kv, s_q, b, h, s_kv_padded, 0.f, true);

    // Do the check.
    check_results(p_h, p_ref_h, s_kv_padded, cu_seqlens_q.back() /*not needed: * b -- already counted */ * h,
        s_kv_padded, 0.f, true, true);
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
    set_mat<float>(s_ref_h, seqlens_q, s_q, b, h, s_kv_padded, 0.f);
    set_mat<float>(s_h, seqlens_q, s_q, b, h, s_kv_padded, 0.f);

    // Do the check.
    check_results(s_h, s_ref_h, s_kv_padded, cu_seqlens_q.back() * h, s_kv_padded, softmax_epsilon, true, true);
#endif // defined(STORE_S)

    // Check the final results.
    int status = -1;
    if (skip_checks)
    {
        printf("\n");
        print_results(true, false);
        status = 0;
    }
    else
    {
        FMHA_CHECK_CUDA(cuda_memcpy_d2h(o_packed_h, o_packed_d, o_packed_size, data_type));

        if (interleaved)
        {
            // revert batch-interleaved format: 3 x h/32 x total x d x 32 => total x
            // h x 3 x d
            x_vec32(false, o_packed_h, h, total_q, 1);
        }

        extract_and_transpose_output<float>(o_ref_packed_h, o_ref_h, seqlens_q, s_q, b, h, d);

        if (verbose)
        {
            printf("\nChecking .....: O = V * S\n");
        }

        status = check_results(o_packed_h, o_ref_packed_h, d, cu_seqlens_q.back() * h, d, epsilon, verbose, true);

        expand_and_transpose_output<float>(o_h, o_packed_h, seqlens_q, s_q, b, h, d);
        eval(o_ref_h, o_h, seqlens_q, b, s_q, h, d, verbose);
        // printf("%f %f\n", o_packed_h[0], o_ref_h[0]);

        if (status != 0)
        { // if there was an error, print the config of the run
            printf("v1=%d il=%d s_q=%lu s_kv=%lu b=%lu h=%lu d=%lu dtype=%s\n", v1, interleaved, s_q, s_kv, b, h, d,
                data_type_to_name(data_type).c_str());
        }

        if (!verbose)
        { // this just prints the SUCCESS/ERROR line
            print_results(true, true, status == 0);
        }
    }

    if (verbose)
    {
        // Runtimes.
        printf("\n");
        if (skip_checks)
        {
            printf("Non-fused time: %.6fms\n", non_fused_elapsed / float(runs));
        }
        printf("Fused time ...: %.6fms\n", fused_elapsed / float(runs));
        if (!skip_checks)
        {
            printf("Ratio ........: %.2fx\n", non_fused_elapsed / fused_elapsed);
        }
    }
    else
    {
        printf("Elapsed ......: %.6f (%.2fx)\n", fused_elapsed, non_fused_elapsed / fused_elapsed);
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
    FMHA_CHECK_CUDA(cudaFree(q_d));
    FMHA_CHECK_CUDA(cudaFree(kv_d));
    FMHA_CHECK_CUDA(cudaFree(mask_d));
    FMHA_CHECK_CUDA(cudaFree(packed_mask_d));
    FMHA_CHECK_CUDA(cudaFree(p_d));
    FMHA_CHECK_CUDA(cudaFree(s_d));
    FMHA_CHECK_CUDA(cudaFree(o_d));
    FMHA_CHECK_CUDA(cudaFree(tmp_d));
    FMHA_CHECK_CUDA(cudaFree(cu_seqlens_q_d));
    FMHA_CHECK_CUDA(cudaFree(cu_seqlens_kv_d));
    FMHA_CHECK_CUDA(cudaFree(max_scratch_d));
    FMHA_CHECK_CUDA(cudaFree(sum_scratch_d));
    FMHA_CHECK_CUDA(cudaFree(o_scratch_d));
    FMHA_CHECK_CUDA(cudaFree(counters_d));
    FMHA_CHECK_CUDA(cudaFree(softmax_sum_d));
    FMHA_CHECK_CUDA(cudaFree(softmax_max_d));

    free(q_h);
    free(kv_h);
    free(mask_h);
    free(packed_mask_h);
    free(s_h);
    free(o_h);
    free(o_ref_h);

    free(p_ref_h);
#if defined(STORE_P)
    free(p_h);
#endif // defined(STORE_P)
    free(s_ref_h);

    return status;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
