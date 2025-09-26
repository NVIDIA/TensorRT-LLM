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

#include "fmha/numeric_types.h"
#include "fused_multihead_attention_utils.h"

#include "hopper/fmha_dgrad.h"
#include "hopper/fmha_fprop.h"

static inline bool check_tensor(at::Tensor const& x)
{
    return x.is_cuda() && x.is_contiguous();
}

static inline void check_qkv(at::Tensor const& QKV8)
{
    TORCH_CHECK(check_tensor(QKV8));
    // This should be the FP8 encoding used by TE.
    TORCH_CHECK(QKV8.scalar_type() == torch::kByte);
    TORCH_CHECK(QKV8.dim() == 4); // total x 3 x h x d
    TORCH_CHECK(QKV8.size(1) == 3);
    TORCH_CHECK(QKV8.size(3) == 64);
}

static inline void check_o(at::Tensor const& o)
{
    TORCH_CHECK(check_tensor(o));
    // This should be the FP8 encoding used by TE.
    TORCH_CHECK(o.scalar_type() == torch::kByte);
    TORCH_CHECK(o.dim() == 3); // total x h x d
    TORCH_CHECK(o.size(2) == 64);
}

static inline void check_stats(at::Tensor const& stats, int const b, int const h, int const s)
{
    TORCH_CHECK(check_tensor(stats));
    TORCH_CHECK(stats.scalar_type() == torch::kFloat32);
    TORCH_CHECK(stats.dim() == 4); // b,h,s,1
    TORCH_CHECK(stats.size(0) == b);
    TORCH_CHECK(stats.size(1) == h);
    TORCH_CHECK(stats.size(2) == s);
    TORCH_CHECK(stats.size(3) == 1);
}

static inline void check_cu_seqlens(at::Tensor const& cu_seqlens)
{
    TORCH_CHECK(check_tensor(cu_seqlens));
    TORCH_CHECK(cu_seqlens.scalar_type() == torch::kInt32);
    TORCH_CHECK(cu_seqlens.dim() == 1);
}

static inline void check_scalar(at::Tensor const& scalar)
{
    TORCH_CHECK(check_tensor(scalar));
    TORCH_CHECK(scalar.dim() <= 1);
    TORCH_CHECK(scalar.numel() == 1);
    TORCH_CHECK(scalar.scalar_type() == torch::kFloat32);
}

static inline void check_seed(at::Tensor const& philox_unpacked)
{
    TORCH_CHECK(check_tensor(philox_unpacked));
    TORCH_CHECK(philox_unpacked.numel() == 2);
    TORCH_CHECK(philox_unpacked.scalar_type() == torch::kInt64);
}

std::vector<at::Tensor> mha_fwd(
    at::Tensor const& QKV8,       // total x num_heads x 3 x head_size, total := \sum_{i=0}^{b} s_i
    at::Tensor const& cu_seqlens, // b+1
    at::Tensor const& d_scale_qkv, at::Tensor const& q_scale_s, at::Tensor const& q_scale_o, at::Tensor& amax_s,
    at::Tensor& amax_o,
    at::Tensor& d_scale_s, // we have to produce this
    at::Tensor& d_scale_o, // we have to produce this
    float const p_dropout, int const max_seq_len, bool const is_training, bool const set_zero,
    c10::optional<at::Generator> gen_)
{

    using namespace fmha::hopper;
    using namespace torch::indexing;

    auto props = at::cuda::getCurrentDeviceProperties();
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    auto gen = at::get_generator_or_default<at::CUDAGeneratorImpl>(gen_, at::cuda::detail::getDefaultCUDAGenerator());

    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);

    check_qkv(QKV8);
    check_cu_seqlens(cu_seqlens);
    check_scalar(d_scale_qkv);
    check_scalar(q_scale_o);
    check_scalar(amax_s);
    check_scalar(amax_o);

    TORCH_CHECK(max_seq_len <= 512);
    int const s = max_seq_len;
    int const b = cu_seqlens.numel() - 1;
    TORCH_CHECK(b <= QKV8.size(0));

    int const total = QKV8.size(0);
    int const h = QKV8.size(2);
    int const d = QKV8.size(3);
    float const scale_q_k = 1.f / sqrtf(d);

    auto Mfprop = torch::empty({b, h, s, 1}, options);
    auto Zfprop = torch::empty({b, h, s, 1}, options);
    auto O8 = torch::empty({total, h, d}, options.dtype(torch::kByte));

    // Stores the seed and offset as returned by at::cuda::philox::unpacked.
    // Note: torch does not have an uint64_t.
    auto philox_unpacked = torch::empty({2}, options.dtype(torch::kInt64));

    if (set_zero)
    {
        O8.zero_();
    }

    typename fprop::Launch_params::Params params_(QKV8.data_ptr(), O8.data_ptr(), Mfprop.data_ptr<float>(),
        Zfprop.data_ptr<float>(), cu_seqlens.data_ptr<int>(), amax_s.data_ptr<float>(), amax_o.data_ptr<float>(), b, s,
        h, d, p_dropout, scale_q_k, d_scale_qkv.data_ptr<float>(), d_scale_s.data_ptr<float>(),
        d_scale_o.data_ptr<float>(), q_scale_s.data_ptr<float>(), q_scale_o.data_ptr<float>(),
        reinterpret_cast<uint64_t*>(philox_unpacked.data_ptr()));

    auto launch = &fprop::run_fmha_fprop_fp8_512_64_sm90;
    fprop::Launch_params launch_params(props, stream, params_, is_training);

    launch(launch_params, /*configure=*/true);

    // TODO need to make sure seeds are set correctly
    launch_params.init_philox_state(gen);
    auto& params = launch_params.params;

    TORCH_CHECK(params.qkv_stride_in_bytes == h * 3 * d);
    TORCH_CHECK(params.o_stride_in_bytes == h * d);
    TORCH_CHECK(params.s_stride_in_bytes == b * h * s * 4);

#if defined(FMHA_TRAIN_OPS_DEBUG_HOPPER_FPROP)
    // debug tensors.
    auto print_buf = torch::zeros({1024}, options);
    auto P = torch::zeros({s, b, h, s}, options);
    auto D = torch::zeros({s, b, h, s}, options);

    params.print_buf = print_buf.data_ptr();
    params.p_ptr = P.data_ptr();
    params.s_ptr = D.data_ptr();

    launch(launch_params, /*configure=*/false);

    return {O8, Mfprop, Zfprop, philox_unpacked, P, D, print_buf};
#else
    launch(launch_params, /*configure=*/false);

    return {O8, Mfprop, Zfprop, philox_unpacked};
#endif
}

std::vector<at::Tensor> mha_bwd(at::Tensor const& dO8, // total x num_heads, x head_size
    at::Tensor const& QKV8,                            // total x num_heads x 3 x head_size, total := \sum_{i=0}^{b} s_i
    at::Tensor const& O8,                              // total x num_heads, x head_size
    at::Tensor const& Mfprop,                          // b x h x s x 1 softmax stats: max
    at::Tensor const& Zfprop,                          // b x h x s x 1 softmax stats: normalizer
    at::Tensor const& cu_seqlens,                      // b+1
    at::Tensor const& d_scale_qkv,                     // From fwd - OK
    at::Tensor const& d_scale_s,                       // From fwd - NEW
    at::Tensor const& d_scale_o,                       // From fwd - OK
    at::Tensor const& d_scale_do,                      // Set by predecessor.
    at::Tensor const& q_scale_s,                       // From fwd - NEW
    at::Tensor const& q_scale_dp,                      // From framework - NEW.
    at::Tensor const& q_scale_dqkv,                    // From framework.
    at::Tensor& amax_dp,                               // update inplace
    at::Tensor& amax_dqkv,                             // update inplace
    at::Tensor& d_scale_dp,                            // We have to produce these for the next layer.
    at::Tensor& d_scale_dqkv,                          // We have to produce these for the next layer.
    float const p_dropout,                             // probability to drop
    int const max_seq_len,                             // max sequence length to choose the kernel
    bool const set_zero, bool const all_e5m2, at::Tensor const& philox_unpacked)
{

    using namespace fmha::hopper;
    using namespace torch::indexing;

    auto props = at::cuda::getCurrentDeviceProperties();
    auto stream = at::cuda::getCurrentCUDAStream().stream();

    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);

    check_o(O8);
    check_o(dO8);
    check_qkv(QKV8);

    TORCH_CHECK(max_seq_len <= 512);
    int const s = max_seq_len;
    int const b = cu_seqlens.numel() - 1;
    TORCH_CHECK(b <= QKV8.size(0));

    int const total = QKV8.size(0);
    int const h = QKV8.size(2);
    int const d = QKV8.size(3);
    float const scale_q_k = 1.f / sqrtf(d);

    check_stats(Mfprop, b, h, s);
    check_stats(Zfprop, b, h, s);
    check_seed(philox_unpacked);

    auto dQtmp = torch::empty({b, h, s, d}, options);
    auto dQKV8 = torch::empty_like(QKV8);

    if (set_zero)
    {
        dQKV8.zero_();
    }

    typename dgrad::Launch_params::Params params_(dQKV8.data_ptr(), QKV8.data_ptr(), dO8.data_ptr(), O8.data_ptr(),
        Mfprop.data_ptr<float>(), Zfprop.data_ptr<float>(), cu_seqlens.data_ptr<int>(), amax_dp.data_ptr<float>(),
        amax_dqkv.data_ptr<float>(), dQtmp.data_ptr(), b, s, h, d, p_dropout, scale_q_k, d_scale_qkv.data_ptr<float>(),
        d_scale_s.data_ptr<float>(), d_scale_o.data_ptr<float>(), d_scale_do.data_ptr<float>(),
        d_scale_dp.data_ptr<float>(), d_scale_dqkv.data_ptr<float>(), q_scale_s.data_ptr<float>(),
        q_scale_dp.data_ptr<float>(), q_scale_dqkv.data_ptr<float>(),
        reinterpret_cast<uint64_t*>(philox_unpacked.data_ptr()));

    dgrad::Launch_params launch_params(props, stream, params_, all_e5m2);

    auto launch = &dgrad::run_fmha_dgrad_fp8_512_64_sm90;

    launch(launch_params, /*configure=*/true);

    auto& params = launch_params.params;

    TORCH_CHECK(params.qkv_stride_in_bytes == 3 * h * d);
    TORCH_CHECK(params.o_stride_in_bytes == h * d);
    TORCH_CHECK(params.ds_stride_in_bytes == b * h * s * 4);

#if defined(FMHA_TRAIN_OPS_DEBUG_HOPPER_DGRAD)
    auto print_buf = torch::zeros({1024}, options);
    auto S = torch::zeros({s, b, h, s}, options);
    auto dS = torch::zeros({s, b, h, s}, options);
    auto dP = torch::zeros({s, b, h, s}, options);
    params.print_buf = print_buf.data_ptr();
    params.s_ptr = S.data_ptr();
    params.ds_ptr = dS.data_ptr();
    params.dp_ptr = dP.data_ptr();

    launch(launch_params, /*configure=*/false);

    return {dQKV8, S, dS, dP, print_buf};
#else
    launch(launch_params, /*configure=*/false);

    return {dQKV8};
#endif
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{

    m.doc() = "CUDA fused Multihead-Attention for BERT (FP8)";

    m.def("fwd", &mha_fwd, "Forward pass");
    m.def("bwd", &mha_bwd, "Backward pass");
}
