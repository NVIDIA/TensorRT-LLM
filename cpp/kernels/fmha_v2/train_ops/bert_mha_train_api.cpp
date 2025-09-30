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

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include "fused_multihead_attention_fprop.h"

#include "fmha/alibi_params.h"
#include "fmha/numeric_types.h"
#include "fused_multihead_attention_utils.h"

#include "hopper/fmha_dgrad.h"

void run_conversion_int32_to_int8(void* dst, void const* src, int s, int b, int h, int d, float scale);

////////////////////////////////////////////////////////////////////////////////////////////////////
// disable the non-flash-attention fmha by default
// void run_fmha_v2_fp16_128_64_sm80(Launch_params<Fused_multihead_attention_fprop_params> &launch_params, const bool
// configure); void run_fmha_v2_fp16_256_64_sm80(Launch_params<Fused_multihead_attention_fprop_params> &launch_params,
// const bool configure); void run_fmha_v2_fp16_384_64_sm80(Launch_params<Fused_multihead_attention_fprop_params>
// &launch_params, const bool configure); void
// run_fmha_v2_fp16_512_64_sm80(Launch_params<Fused_multihead_attention_fprop_params> &launch_params, const bool
// configure);

// [FP16] flash attention fprop: support any sequence length
void run_fmha_v2_flash_attention_fp16_S_40_sm80(
    Launch_params<Fused_multihead_attention_fprop_params>& launch_params, bool const configure);
void run_fmha_v2_flash_attention_fp16_S_64_sm80(
    Launch_params<Fused_multihead_attention_fprop_params>& launch_params, bool const configure);
void run_fmha_v2_flash_attention_fp16_S_80_sm80(
    Launch_params<Fused_multihead_attention_fprop_params>& launch_params, bool const configure);
void run_fmha_v2_flash_attention_fp16_S_96_sm80(
    Launch_params<Fused_multihead_attention_fprop_params>& launch_params, bool const configure);
void run_fmha_v2_flash_attention_fp16_S_128_sm80(
    Launch_params<Fused_multihead_attention_fprop_params>& launch_params, bool const configure);

// [BF16] flash attention fprop: support any sequence length
void run_fmha_v2_flash_attention_bf16_S_40_sm80(
    Launch_params<Fused_multihead_attention_fprop_params>& launch_params, bool const configure);
void run_fmha_v2_flash_attention_bf16_S_64_sm80(
    Launch_params<Fused_multihead_attention_fprop_params>& launch_params, bool const configure);
void run_fmha_v2_flash_attention_bf16_S_80_sm80(
    Launch_params<Fused_multihead_attention_fprop_params>& launch_params, bool const configure);
void run_fmha_v2_flash_attention_bf16_S_96_sm80(
    Launch_params<Fused_multihead_attention_fprop_params>& launch_params, bool const configure);
void run_fmha_v2_flash_attention_bf16_S_128_sm80(
    Launch_params<Fused_multihead_attention_fprop_params>& launch_params, bool const configure);

// [FP16] flash attention backwards: support any sequence length
void run_fmha_dgrad_v2_flash_attention_fp16_S_40_sm80(
    Fused_multihead_attention_fprop_params const& params, cudaStream_t stream);
void run_fmha_dgrad_v2_flash_attention_fp16_S_64_sm80(
    Fused_multihead_attention_fprop_params const& params, cudaStream_t stream);
void run_fmha_dgrad_v2_flash_attention_fp16_S_80_sm80(
    Fused_multihead_attention_fprop_params const& params, cudaStream_t stream);
void run_fmha_dgrad_v2_flash_attention_fp16_S_96_sm80(
    Fused_multihead_attention_fprop_params const& params, cudaStream_t stream);
void run_fmha_dgrad_v2_flash_attention_fp16_S_128_sm80(
    Fused_multihead_attention_fprop_params const& params, cudaStream_t stream);

// [BF16] flash attention backwards: support any sequence length
void run_fmha_dgrad_v2_flash_attention_bf16_S_40_sm80(
    Fused_multihead_attention_fprop_params const& params, cudaStream_t stream);
void run_fmha_dgrad_v2_flash_attention_bf16_S_64_sm80(
    Fused_multihead_attention_fprop_params const& params, cudaStream_t stream);
void run_fmha_dgrad_v2_flash_attention_bf16_S_80_sm80(
    Fused_multihead_attention_fprop_params const& params, cudaStream_t stream);
void run_fmha_dgrad_v2_flash_attention_bf16_S_96_sm80(
    Fused_multihead_attention_fprop_params const& params, cudaStream_t stream);
void run_fmha_dgrad_v2_flash_attention_bf16_S_128_sm80(
    Fused_multihead_attention_fprop_params const& params, cudaStream_t stream);

void fmha_run_noloop_reduce(void* out, void const* in, int const* cu_seqlens, int const hidden_size,
    int const batch_size, int const total, int const num_chunks, cudaStream_t stream);

// enable this flag when heads are interleaved (need to refactor here)
// #define HEADS_INTERLEAVED

#ifdef HEADS_INTERLEAVED
// Expecting interleaved heads: total x h x 3 x d
constexpr int TOTAL_DIM = 0;
constexpr int THREE_DIM = 2;
constexpr int H_DIM = 1;
constexpr int D_DIM = 3;
#else
// Expecting non-interleaved heads: total x 3 x h x d
constexpr int TOTAL_DIM = 0;
constexpr int THREE_DIM = 1;
constexpr int H_DIM = 2;
constexpr int D_DIM = 3;
#endif

static int Next_power_of_two(int head_size)
{
    if (head_size == 40 || head_size == 64)
    {
        return 64;
    }
    else if (head_size == 80 || head_size == 96 || head_size == 128)
    {
        return 128;
    }
    else if (head_size == 160 || head_size == 256)
    {
        return 256;
    }
    else
    {
        TORCH_CHECK(false);
    }
}

void set_params(Fused_multihead_attention_fprop_params& params,
    // sizes
    size_t const b, size_t const s, size_t const h, size_t const d, size_t const total_s,
    // device pointers
    void* qkv_packed_d, void* cu_seqlens_d, void* o_packed_d, void* do_packed_d, void* s_d, void* softmax_lse_d,
    void* softmax_sum_d, void* dq_acc_d, void* seed_d, float p_dropout,
    // different layouts: sequences_interleaved --> [s, b]
    bool sequences_interleaved, bool is_bf16, bool has_alibi, bool is_causal)
{

    Data_type acc_type = DATA_TYPE_FP32;
    Data_type data_type = is_bf16 ? DATA_TYPE_BF16 : DATA_TYPE_FP16;

    // Reset the parameters
    memset(&params, 0, sizeof(params));

    // Set the pointers and strides.
    params.qkv_ptr = qkv_packed_d;
    params.qkv_stride_in_bytes = sequences_interleaved
        ? get_size_in_bytes(h * 3 * d * b, data_type)
        : get_size_in_bytes(h * 3 * d, data_type); // [s, b, 3, h, d] : [b, s, 3, h, d]
    params.o_ptr = o_packed_d;
    params.do_ptr = do_packed_d;
    params.lse_ptr = softmax_lse_d;
    params.softmax_sum_ptr = softmax_sum_d;
    params.dq_acc_ptr = dq_acc_d;
    params.seed_ptr = seed_d;
    params.o_stride_in_bytes = sequences_interleaved
        ? get_size_in_bytes(h * d * b, data_type)
        : get_size_in_bytes(h * d, data_type); // [s, b, h, d] : [b, s, h, d]
    params.is_bf16 = is_bf16;
    params.is_causal = is_causal;
    params.has_alibi = has_alibi;
    params.alibi_params = fmha::AlibiParams(h);

    params.cu_seqlens = static_cast<int*>(cu_seqlens_d);

    // S = softmax(P)
    params.s_ptr = s_d;
    params.s_stride_in_bytes = get_size_in_bytes(b * h * s, data_type);

    params.lse_stride_in_bytes = get_size_in_bytes(s, acc_type); // fp32, [b, h, s]
    params.sum_stride_in_bytes = get_size_in_bytes(1, acc_type); // [b, h, s], TODO

    // Set the dimensions.
    params.b = b;
    params.h = h;
    params.s = s;
    params.d = d;
    params.total_s = total_s;

    // Set the different scale values.
    float const scale_bmm1 = 1.f / sqrtf(d);
    constexpr float scale_softmax = 1.f;
    constexpr float scale_bmm2 = 1.f;

    params.fscale_bmm1 = scale_bmm1;

    // set_alpha(params.scale_bmm1, scale_bmm1, acc_type);
    set_alpha(params.scale_bmm1, scale_bmm1, acc_type);
    set_alpha(params.scale_softmax, scale_softmax, acc_type);
    // set_alpha(params.scale_bmm2, scale_bmm2, data_type);

    // Set this to probability of keeping an element to simplify things.
    params.p_dropout = 1.f - p_dropout;
    // scale by 2^16
    params.p_dropout_16bit = (uint16_t) (65536.0 * (1.f - p_dropout));
    params.rp_dropout = 1.f / params.p_dropout;
    if (p_dropout == 0.f)
    {
        params.has_dropout = false;
    }
    else
    {
        params.has_dropout = true;
    }
    TORCH_CHECK(p_dropout < 1.f && p_dropout >= 0.f);
    set_alpha(params.scale_dropout, params.rp_dropout, data_type);
}

std::vector<at::Tensor> mha_fwd(at::Tensor const& qkv, // total x num_heads x 3 x head_size, total := \sum_{i=0}^{b} s_i
    at::Tensor const& cu_seqlens,                      // b+1
    float const p_dropout, int const max_seq_len, bool const is_training, bool const is_nl,
    bool is_sequences_interleaved, bool is_causal, bool has_alibi, c10::optional<at::Generator> gen_)
{

    auto props = at::cuda::getCurrentDeviceProperties();
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    Launch_params<Fused_multihead_attention_fprop_params> launch_params(props, stream, is_training, is_causal, is_nl);

    TORCH_CHECK(qkv.is_cuda())
    TORCH_CHECK(cu_seqlens.is_cuda())

    TORCH_CHECK(qkv.is_contiguous())
    TORCH_CHECK(cu_seqlens.is_contiguous())

    TORCH_CHECK(cu_seqlens.dim() == 1);
    TORCH_CHECK(qkv.dim() == 4);

    TORCH_CHECK(qkv.dtype() == torch::kFloat16 || qkv.dtype() == torch::kBFloat16);

    auto const sizes = qkv.sizes();

    TORCH_CHECK(sizes[THREE_DIM] == 3);

    int const batch_size = cu_seqlens.numel() - 1;
    int const total = sizes[TOTAL_DIM];
    int const num_heads = sizes[H_DIM];
    int const head_size = sizes[D_DIM];
    TORCH_CHECK(batch_size > 0);
    TORCH_CHECK(head_size == 40 || head_size == 64 || head_size == 80 || head_size == 96 || head_size == 128);

    int seq_len = max_seq_len;

    auto opts = qkv.options();

    auto ctx = torch::empty({total, num_heads, head_size}, opts);

    auto s = torch::empty({batch_size, num_heads, seq_len, seq_len}, opts);

    // flash attention m + log(l) (float32)
    auto lse = torch::empty({batch_size, num_heads, seq_len}, opts.dtype(torch::kFloat32));

    // to save seed and philox_offset_per_thread for bwd pass (int uint64_t).
    auto random_seeds = torch::empty({2}, opts.dtype(torch::kInt64));

    auto gen = at::get_generator_or_default<at::CUDAGeneratorImpl>(gen_, at::cuda::detail::getDefaultCUDAGenerator());

    auto is_bf16 = qkv.dtype() == torch::kBFloat16;

    auto launcher = &run_fmha_v2_flash_attention_fp16_S_64_sm80;

    // use flash attention by default
    if (is_bf16)
    {
        switch (head_size)
        {
        case 40: launcher = &run_fmha_v2_flash_attention_bf16_S_40_sm80; break;
        case 64: launcher = &run_fmha_v2_flash_attention_bf16_S_64_sm80; break;
        case 80: launcher = &run_fmha_v2_flash_attention_bf16_S_80_sm80; break;
        case 96: launcher = &run_fmha_v2_flash_attention_bf16_S_96_sm80; break;
        case 128: launcher = &run_fmha_v2_flash_attention_bf16_S_128_sm80; break;
        default:
            std::string const error_msg
                = "bf16, the d = " + std::to_string(head_size) + " kernel hasn't been generated!";
            TORCH_CHECK(false, error_msg);
            break;
        }
    }
    else
    {
        switch (head_size)
        {
        case 40: launcher = &run_fmha_v2_flash_attention_fp16_S_40_sm80; break;
        case 64: launcher = &run_fmha_v2_flash_attention_fp16_S_64_sm80; break;
        case 80: launcher = &run_fmha_v2_flash_attention_fp16_S_80_sm80; break;
        case 96: launcher = &run_fmha_v2_flash_attention_fp16_S_96_sm80; break;
        case 128: launcher = &run_fmha_v2_flash_attention_fp16_S_128_sm80; break;
        default:
            std::string const error_msg = "the d = " + std::to_string(head_size) + " kernel hasn't been generated!";
            TORCH_CHECK(false, error_msg);
            break;
        }
    }

    set_params(launch_params.params, batch_size, seq_len, num_heads, head_size, total, qkv.data_ptr(),
        cu_seqlens.data_ptr(), ctx.data_ptr(),
        nullptr,        // do
        s.data_ptr(),
        lse.data_ptr(), // softmax_lse
        nullptr,        // softmax_sum
        nullptr,        // dq_acc
        random_seeds.data_ptr(), p_dropout, is_sequences_interleaved, is_bf16, has_alibi, is_causal);

    launcher(launch_params, /*configure=*/true);
    // number of times random will be generated per thread, to offset philox counter in thc random
    // state
    int64_t counter_offset = launch_params.params.b * launch_params.params.h * 32;

    if (is_training)
    {
        // See Note [Acquire lock when using random generators]
        std::lock_guard<std::mutex> lock(gen->mutex_);
        launch_params.params.philox_args = gen->philox_cuda_state(counter_offset);
    }

    launcher(launch_params, /*configure=*/false);

    return {ctx, lse, random_seeds, s};
}

std::vector<at::Tensor> mha_bwd(at::Tensor const& d_out, // total x num_heads, x head_size
    at::Tensor const& out,                               // total x num_head, x hea_szie
    at::Tensor const& qkv, // total x num_heads x 3 x head_size, total := \sum_{i=0}^{b} s_i
    // at::Tensor &softmax,           // b x h x s x s softmax and dmask - will be overwritten with dP
    at::Tensor const& softmax_lse, // b x h x s
    at::Tensor const& random_seed, // 2 elements, seed and offset
    at::Tensor const& cu_seqlens,  // b+1
    float const p_dropout,         // probability to drop
    int const max_seq_len,         // max sequence length to choose the kernel
    bool save_dp, bool is_sequences_interleaved, bool is_causal, bool has_alibi, c10::optional<at::Generator> gen_)
{

    auto stream = at::cuda::getCurrentCUDAStream().stream();

    TORCH_CHECK(qkv.is_cuda())
    TORCH_CHECK(cu_seqlens.is_cuda())

    TORCH_CHECK(qkv.is_contiguous())
    TORCH_CHECK(cu_seqlens.is_contiguous())

    TORCH_CHECK(softmax_lse.dim() == 3);
    TORCH_CHECK(cu_seqlens.dim() == 1);
    TORCH_CHECK(qkv.dim() == 4);

    TORCH_CHECK(qkv.dtype() == torch::kFloat16 || qkv.dtype() == torch::kBFloat16);

    auto const sizes = qkv.sizes();

    {
        auto const size_dout = d_out.sizes();
        auto const size_out = out.sizes();
        TORCH_CHECK(d_out.dim() == 4);
        for (int i = 0; i < 4; i++)
        {
            TORCH_CHECK(size_dout[i] == size_out[i]);
        }
    }

    TORCH_CHECK(sizes[THREE_DIM] == 3);

    int const batch_size = cu_seqlens.numel() - 1;
    int const total = sizes[TOTAL_DIM];
    int const num_heads = sizes[H_DIM];
    int const head_size = sizes[D_DIM];
    TORCH_CHECK(batch_size > 0);
    TORCH_CHECK(head_size == 64 || head_size == 40 || head_size == 128 || head_size == 80 || head_size == 96);

    int seq_len = max_seq_len;
    auto opts = qkv.options();

    // auto dqkv = torch::empty_like(qkv); // TODO: need to check the impact to performance
    auto dqkv = torch::zeros_like(qkv);
    auto softmax_sum = torch::empty({batch_size, num_heads, seq_len}, opts.dtype(at::kFloat));

    // The array to store the reductions for dQ (using FP32 atomics).
    auto dq_acc = torch::empty({num_heads, total, Next_power_of_two(head_size)}, opts.dtype(at::kFloat));
    dq_acc.zero_();
    assert(dq_acc.data_ptr() != nullptr);

    Fused_multihead_attention_fprop_params params;

    // auto gen = at::get_generator_or_default<at::CUDAGeneratorImpl>(gen_,
    // at::cuda::detail::getDefaultCUDAGenerator());

    auto is_bf16 = qkv.dtype() == torch::kBFloat16;

    auto launcher = &run_fmha_dgrad_v2_flash_attention_fp16_S_40_sm80;
    // use flash attention by default
    if (is_bf16)
    {
        switch (head_size)
        {
        case 40: launcher = &run_fmha_dgrad_v2_flash_attention_bf16_S_40_sm80; break;
        case 64: launcher = &run_fmha_dgrad_v2_flash_attention_bf16_S_64_sm80; break;
        case 80: launcher = &run_fmha_dgrad_v2_flash_attention_bf16_S_80_sm80; break;
        case 96: launcher = &run_fmha_dgrad_v2_flash_attention_bf16_S_96_sm80; break;
        case 128: launcher = &run_fmha_dgrad_v2_flash_attention_bf16_S_128_sm80; break;
        default:
            std::string const error_msg
                = "bf16, the d = " + std::to_string(head_size) + " kernel hasn't been generated!";
            TORCH_CHECK(false, error_msg);
            break;
        }
    }
    else
    {
        switch (head_size)
        {
        case 40: launcher = &run_fmha_dgrad_v2_flash_attention_fp16_S_40_sm80; break;
        case 64: launcher = &run_fmha_dgrad_v2_flash_attention_fp16_S_64_sm80; break;
        case 80: launcher = &run_fmha_dgrad_v2_flash_attention_fp16_S_80_sm80; break;
        case 96: launcher = &run_fmha_dgrad_v2_flash_attention_fp16_S_96_sm80; break;
        case 128: launcher = &run_fmha_dgrad_v2_flash_attention_fp16_S_128_sm80; break;
        default:
            std::string const error_msg = "the d = " + std::to_string(head_size) + " kernel hasn't been generated!";
            TORCH_CHECK(false, error_msg);
            break;
        }
    }

    if (save_dp)
    {
        auto softmax = torch::empty({batch_size, num_heads, seq_len, seq_len}, opts);

        set_params(params, batch_size, seq_len, num_heads, head_size, total, qkv.data_ptr(), cu_seqlens.data_ptr(),
            out.data_ptr(),     // o_ptr = out
            d_out.data_ptr(),   // do_ptr = d_out
            softmax.data_ptr(), // softmax gets overwritten by dP!
            softmax_lse.data_ptr(), softmax_sum.data_ptr(), dq_acc.data_ptr(), random_seed.data_ptr(), p_dropout,
            is_sequences_interleaved, is_bf16, has_alibi, is_causal);

        Data_type acc_type = DATA_TYPE_FP32;
        set_alpha(params.scale_bmm1, 1.f, acc_type);
        set_alpha(params.scale_softmax, 1.f / sqrtf(head_size), acc_type);
        set_alpha(params.scale_bmm2, 1.f, DATA_TYPE_FP16);
        params.dqkv_ptr = dqkv.data_ptr();
        params.save_dp = true;

        launcher(params, stream);

        return {dqkv, softmax};
    }
    else
    {
        auto softmax = torch::empty({1}, opts);
        set_params(params, batch_size, seq_len, num_heads, head_size, total, qkv.data_ptr(), cu_seqlens.data_ptr(),
            out.data_ptr(),   // o_ptr = out
            d_out.data_ptr(), // do_ptr = d_out
            nullptr,          // softmax gets overwritten by dP!
            softmax_lse.data_ptr(), softmax_sum.data_ptr(), dq_acc.data_ptr(), random_seed.data_ptr(), p_dropout,
            is_sequences_interleaved, is_bf16, has_alibi, is_causal);

        Data_type acc_type = DATA_TYPE_FP32;
        set_alpha(params.scale_bmm1, 1.f, acc_type);
        set_alpha(params.scale_softmax, 1.f / sqrtf(head_size), acc_type);
        set_alpha(params.scale_bmm2, 1.f, DATA_TYPE_FP16);
        params.dqkv_ptr = dqkv.data_ptr();
        params.save_dp = false;

        // WAR: just use the seed + offset passed from fprop.
        // int64_t counter_offset = params.b * params.h * 32;
        // if ( params.has_dropout ) {
        //     // See Note [Acquire lock when using random generators]
        //     std::lock_guard<std::mutex> lock(gen->mutex_);
        //     params.philox_args = gen->philox_cuda_state(counter_offset);
        // }

        // TODO: Make that more robust.
        // params.fscale_bmm1 *= float(M_LOG2E);

        // Clear the accumulation buffer for dQ. TODO: Should we do that somewhere else?
        // dq_acc.zero_();

        launcher(params, stream);

        return {dqkv, softmax};
    }
}

std::vector<at::Tensor> mha_bwd_noloop(at::Tensor const& d_out, // total x num_heads, x head_size
    at::Tensor const& qkv,               // total x num_heads x 3 x head_size, total := \sum_{i=0}^{b} s_i
    at::Tensor& softmax,                 // b x h x s x s softmax and dmask - will be overwritten with dP
    at::Tensor const& cu_seqlens,        // b+1
    float const p_dropout,               // probability to drop
    int const max_seq_len,               // max sequence length to choose the kernel
    bool const is_sequences_interleaved, // layout sequence interleaved ([s, b]) or not
    bool const is_causal)
{

    auto stream = at::cuda::getCurrentCUDAStream().stream();

    TORCH_CHECK(qkv.is_cuda())
    TORCH_CHECK(cu_seqlens.is_cuda())

    TORCH_CHECK(qkv.is_contiguous())
    TORCH_CHECK(cu_seqlens.is_contiguous())

    TORCH_CHECK(cu_seqlens.dim() == 1);

    TORCH_CHECK(qkv.dim() == 4);

    auto const sizes = qkv.sizes();

    TORCH_CHECK(sizes[THREE_DIM] == 3);

    int const batch_size = cu_seqlens.numel() - 1;

    int const total = sizes[TOTAL_DIM];
    int const num_heads = sizes[H_DIM];
    int const head_size = sizes[D_DIM];
    TORCH_CHECK(batch_size > 0);
    TORCH_CHECK(head_size == 64);

    int seq_len = 512;
    // auto launch = &run_fmha_dgrad_v2_fp16_512_64_sm80_noloop;

    auto opts = qkv.options();

    auto dqkv = torch::empty_like(qkv);

    // The code below requires num_chunks > 1
    // TODO num_chunks should be based on the number of SMs
    int num_chunks = 2;
    if (batch_size == 1)
    {
        num_chunks = 4;
    }
    else if (batch_size == 2)
    {
        num_chunks = 3;
    }
    auto dkv = torch::empty({total, num_chunks, 2, num_heads, head_size}, opts);

    Fused_multihead_attention_fprop_params params;

    auto is_bf16 = qkv.dtype() == torch::kBFloat16;

    set_params(params, batch_size, seq_len, num_heads, head_size, total, qkv.data_ptr(), cu_seqlens.data_ptr(),
        nullptr,            // o_ptr = out
        d_out.data_ptr(),   // do_ptr = d_out
        softmax.data_ptr(), // softmax gets overwritten by dP!
        nullptr,            // softmax lse
        nullptr,            // softmax sum reduce_sum (dO * O)
        nullptr,            // dq_acc_d
        nullptr,            // random_seed
        p_dropout, is_sequences_interleaved, is_bf16,
        false,              // has_alibi
        is_causal);

    params.dkv_ptr = dkv.data_ptr();

    Data_type acc_type = DATA_TYPE_FP32;
    set_alpha(params.scale_bmm1, 1.f, acc_type);
    set_alpha(params.scale_softmax, 1.f / sqrtf(head_size), acc_type);
    set_alpha(params.scale_bmm2, 1.f, DATA_TYPE_FP16);
    params.dqkv_ptr = dqkv.data_ptr();

    // launch(params, num_chunks, stream);

    // SPLIT-K reduction of num_chunks dK, dV parts

    // The equivalent of the following Pytorch code:
    // using namespace torch::indexing;
    // at::Tensor view_out = dqkv.index({Slice(), Slice(1, None, None)});
    // torch::sum_out(view_out, dkv, 1);

    int const hidden_size = num_heads * head_size;
    fmha_run_noloop_reduce(dqkv.data_ptr(), dkv.data_ptr(), cu_seqlens.data_ptr<int>(), hidden_size, batch_size, total,
        num_chunks, stream);

    return {dqkv, softmax, dkv};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{

    m.doc() = "CUDA fused Multihead-Attention for BERT";

    m.def("fwd", &mha_fwd, "Forward pass");
    m.def("bwd", &mha_bwd, "Backward pass");
    m.def("bwd_nl", &mha_bwd_noloop, "Backward pass (small-batch)");
}
