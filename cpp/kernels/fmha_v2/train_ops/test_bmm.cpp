/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: NVIDIA TensorRT Source Code License Agreement
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include <torch/torch.h>
// #include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include "fused_multihead_attention_fprop.h"

#include "fmha/numeric_types.h"
#include "fused_multihead_attention_utils.h"

#include "hopper/fmha_dgrad.h"
#include "test.h"

void run_conversion_fp32_to_e4m3(void* dst, void const* src, size_t n, float scale_o);
void run_conversion_e4m3_to_fp32(void* dst, void const* src, size_t n, float scale_o);
void run_conversion_fp32_to_e5m2(void* dst, void const* src, size_t n, float scale_o);
void run_conversion_e5m2_to_fp32(void* dst, void const* src, size_t n, float scale_o);

at::Tensor bmm_nt(
    at::Tensor const& A, at::Tensor const& B, cudaDataType_t A_type, cudaDataType_t B_type, float const alpha)
{
    TORCH_CHECK(A.dim() == 3); // BxMxK
    TORCH_CHECK(B.dim() == 3); // BxNxK
    auto sizes_A = A.sizes();
    auto sizes_B = B.sizes();
    int b = sizes_A[0];
    int m = sizes_A[1];
    int k = sizes_A[2];
    int n = sizes_B[1];

    auto strides_A = A.strides();
    auto strides_B = B.strides();
    TORCH_CHECK(b == sizes_B[0]);
    TORCH_CHECK(k == sizes_B[2]);
    TORCH_CHECK(strides_A[2] == 1);
    TORCH_CHECK(strides_B[2] == 1);
    TORCH_CHECK(A.scalar_type() == B.scalar_type());
    TORCH_CHECK(A.is_cuda() && B.is_cuda());

    // We represent FP8 as generic bytes.
    TORCH_CHECK(A.scalar_type() == torch::kByte);

    auto opts = A.options();
    at::Tensor C = torch::empty({b, m, n}, opts.dtype(torch::kFloat32));

    RefBMM bmm(A_type, // CUDA_R_8F_E4M3,
        B_type,        // CUDA_R_8F_E4M3,
        CUDA_R_32F, CUBLAS_COMPUTE_32F, CUDA_R_32F, false, true, m, n, k, strides_A[1], strides_B[1], n, strides_A[0],
        strides_B[0], m * n, b);

    float const beta = 0.f;
    bmm(A.data_ptr(), B.data_ptr(), C.data_ptr(), &alpha, &beta, 0);

    return C;
}

at::Tensor matmul_nt(
    at::Tensor const& A, at::Tensor const& B, cudaDataType_t A_type, cudaDataType_t B_type, float const alpha)
{
    auto dim = A.dim();
    TORCH_CHECK(dim == B.dim());
    TORCH_CHECK(dim > 1);

    auto sizes_A = A.sizes();
    auto sizes_B = B.sizes();

    auto Av = A.view({-1, sizes_A[dim - 2], sizes_A[dim - 1]});
    auto Bv = B.view({-1, sizes_B[dim - 2], sizes_B[dim - 1]});

    auto Cv = bmm_nt(Av, Bv, A_type, B_type, alpha);

    std::vector<int64_t> sizes_C(A.sizes().begin(), A.sizes().end());
    sizes_C[dim - 2] = Cv.sizes()[Cv.dim() - 2];
    sizes_C[dim - 1] = Cv.sizes()[Cv.dim() - 1];
    c10::IntArrayRef tmp(sizes_C);
    return Cv.view(tmp);
}

at::Tensor convert_fp32_to_e4m3(at::Tensor const& src, float scale)
{
    TORCH_CHECK(src.scalar_type() == torch::kFloat32);
    auto options = src.options();
    auto dst = torch::empty(src.sizes(), options.dtype(torch::kByte));
    run_conversion_fp32_to_e4m3(dst.data_ptr(), src.data_ptr(), src.numel(), scale);
    return dst;
}

at::Tensor convert_e4m3_to_fp32(at::Tensor const& src, float scale)
{
    TORCH_CHECK(src.scalar_type() == torch::kByte);
    auto options = src.options();
    auto dst = torch::empty(src.sizes(), options.dtype(torch::kFloat32));
    run_conversion_e4m3_to_fp32(dst.data_ptr(), src.data_ptr(), src.numel(), scale);
    return dst;
}

at::Tensor convert_fp32_to_e5m2(at::Tensor const& src, float scale)
{
    TORCH_CHECK(src.scalar_type() == torch::kFloat32);
    auto options = src.options();
    auto dst = torch::empty(src.sizes(), options.dtype(torch::kByte));
    run_conversion_fp32_to_e5m2(dst.data_ptr(), src.data_ptr(), src.numel(), scale);
    return dst;
}

at::Tensor convert_e5m2_to_fp32(at::Tensor const& src, float scale)
{
    TORCH_CHECK(src.scalar_type() == torch::kByte);
    auto options = src.options();
    auto dst = torch::empty(src.sizes(), options.dtype(torch::kFloat32));
    run_conversion_e5m2_to_fp32(dst.data_ptr(), src.data_ptr(), src.numel(), scale);
    return dst;
}

at::Tensor convert_fp32_to_fp8(at::Tensor const& src, float scale, cudaDataType_t dst_type)
{
    TORCH_CHECK(dst_type == CUDA_R_8F_E5M2 || dst_type == CUDA_R_8F_E4M3);
    if (dst_type == CUDA_R_8F_E5M2)
    {
        return convert_fp32_to_e5m2(src, scale);
    }
    else
    {
        return convert_fp32_to_e4m3(src, scale);
    }
}

at::Tensor convert_fp8_to_fp32(at::Tensor const& src, float scale, cudaDataType_t src_type)
{
    TORCH_CHECK(src_type == CUDA_R_8F_E5M2 || src_type == CUDA_R_8F_E4M3);
    if (src_type == CUDA_R_8F_E5M2)
    {
        return convert_e5m2_to_fp32(src, scale);
    }
    else
    {
        return convert_e4m3_to_fp32(src, scale);
    }
}

at::Tensor seqlens2mask(at::Tensor const& cu_seqlens, int const s, at::TensorOptions const& options)
{
    using namespace torch::indexing;
    int b = cu_seqlens.numel() - 1;
    // [b, 1, s, s]
    auto amask = torch::zeros({b, 1, s, s}, options);
    for (int bi = 0; bi < b; bi++)
    {
        int begin = cu_seqlens[bi].item<int>();
        int end = cu_seqlens[bi + 1].item<int>();
        int si = end - begin;
        amask.index({bi, 0, Slice(None, si), Slice(None, si)}) = 1.f;

        TORCH_CHECK(amask.index({bi, 0, Slice(), Slice()}).sum().item<int>() == si * si);
        TORCH_CHECK(amask.index({bi, 0, 0, Slice()}).sum().item<int>() == si);
    }
    return amask;
}

at::Tensor pad(at::Tensor const& tensor, at::Tensor const& cu_seqlens, int const s)
{
    // pad dim 0 of tensor from [total, ...] => [b, s, ...]

    using namespace torch::indexing;
    auto sizes = tensor.sizes();
    int b = cu_seqlens.numel() - 1;
    TORCH_CHECK(sizes[0] == cu_seqlens[-1].item<int64_t>());
    std::vector<int64_t> new_size = {b, s};
    for (auto d : tensor.index({0}).sizes())
    {
        new_size.push_back(d);
    }

    auto options = tensor.options();
    auto dst = torch::zeros(torch::makeArrayRef(new_size), options);

    for (int bi = 0; bi < b; bi++)
    {
        int begin = cu_seqlens[bi].item<int>();
        int end = cu_seqlens[bi + 1].item<int>();
        int si = end - begin;
        dst.index({bi, Slice(0, si), "..."}) = tensor.index({Slice(begin, end), "..."});
    }

    return dst;
}

at::Tensor unpad(at::Tensor const& tensor, at::Tensor const& cu_seqlens)
{
    // unpad dim 0 of tensor from [b, s, ...] => [total, ...]

    using namespace torch::indexing;
    auto sizes = tensor.sizes();
    int b = cu_seqlens.numel() - 1;
    TORCH_CHECK(b == sizes[0]);
    int s = sizes[1];
    int total = cu_seqlens[-1].item<int>();
    std::vector<int64_t> new_size = {total};
    for (auto d : tensor.index({0, 0}).sizes())
    {
        new_size.push_back(d);
    }

    auto options = tensor.options();
    auto dst = torch::zeros(torch::makeArrayRef(new_size), options);
    for (int bi = 0; bi < b; bi++)
    {
        int begin = cu_seqlens[bi].item<int>();
        int end = cu_seqlens[bi + 1].item<int>();
        int si = end - begin;
        dst.index({Slice(begin, end), "..."}) = tensor.index({bi, Slice(0, si), "..."});
    }

    return dst;
}

std::tuple<at::Tensor, at::Tensor> full_mask(int const b, int const s, torch::TensorOptions const& options)
{
    // Get a mask that represents b full sequences of length s.
    using namespace torch::indexing;

    auto cu_seqlens = torch::arange({b + 1}, options.dtype(torch::kInt32)) * s;
    auto amask = seqlens2mask(cu_seqlens, s, options);
    return {cu_seqlens, amask};
}

std::tuple<at::Tensor, at::Tensor> rand_mask(int const b, int const s, torch::TensorOptions const& options)
{
    // Get a mask that represents b sequences of length randomly drawn from [1, s)
    using namespace torch::indexing;

    auto seqlens = torch::randint(1, s, {b}, options.dtype(torch::kInt32));
    TORCH_CHECK(seqlens.numel() == b);
    TORCH_CHECK(seqlens.min().item<int>() > 0);
    TORCH_CHECK(seqlens.max().item<int>() <= s);
    auto cu_seqlens = torch::zeros({b + 1}, seqlens.options());
    cu_seqlens.index({Slice(1, None)}) = torch::cumsum(seqlens, 0);
    auto amask = seqlens2mask(cu_seqlens, s, options);
    return {cu_seqlens, amask};
}

std::tuple<at::Tensor, at::Tensor> lin_mask(int const b, int const s, torch::TensorOptions const& options)
{
    // Get a mask that represents b sequences of length randomly drawn from [1, s)
    using namespace torch::indexing;

    auto seqlens = torch::linspace(1, s, b, options.dtype(torch::kInt32));
    TORCH_CHECK(seqlens.numel() == b);
    TORCH_CHECK(seqlens.min().item<int>() > 0);
    TORCH_CHECK(seqlens.max().item<int>() <= s);
    auto cu_seqlens = torch::zeros({b + 1}, seqlens.options());
    cu_seqlens.index({Slice(1, None)}) = torch::cumsum(seqlens, 0);
    auto amask = seqlens2mask(cu_seqlens, s, options);
    return {cu_seqlens, amask};
}

std::tuple<at::Tensor, at::Tensor> make_mask(
    int const b, int const s, torch::TensorOptions const& options, Maskgen const maskgen)
{
    at::Tensor cu_seqlens, amask;
    switch (maskgen)
    {
    case RANDOM: std::tie(cu_seqlens, amask) = rand_mask(b, s, options); break;
    case LINEAR: std::tie(cu_seqlens, amask) = lin_mask(b, s, options); break;
    default: std::tie(cu_seqlens, amask) = full_mask(b, s, options); break;
    }
    return {cu_seqlens, amask};
}

at::Tensor draw_tensor(at::IntArrayRef const dims, torch::TensorOptions const& options, Datagen const datagen)
{
    switch (datagen)
    {
    case SMALLINT: return torch::randint(-2, 3, dims, options);
    case ONES: return torch::ones(dims, options);
    }
    // case NORMAL:
    return torch::randn(dims, options);
}

int check_results(at::Tensor const& out, at::Tensor const& ref, float epsilon, bool verbose, bool with_colors)
{

    int m = out.size(-1);
    TORCH_CHECK(m == ref.size(-1));
    auto out_h = out.detach().contiguous().cpu().view({-1, m});
    auto ref_h = ref.detach().contiguous().cpu().view({-1, m});

    TORCH_CHECK(out_h.dim() == 2);
    TORCH_CHECK(ref_h.dim() == 2);

    size_t n = out_h.size(0);

    TORCH_CHECK(n == ref_h.size(0));

    TORCH_CHECK(out_h.scalar_type() == torch::kFloat32);
    TORCH_CHECK(ref_h.scalar_type() == torch::kFloat32);

    TORCH_CHECK(out_h.stride(1) == 1);
    TORCH_CHECK(ref_h.stride(1) == 1);

    TORCH_CHECK(out_h.stride(0) == m);
    TORCH_CHECK(ref_h.stride(0) == m);

    return check_results(
        out_h.data_ptr<float>(), ref_h.data_ptr<float>(), m, n, out_h.stride(0), epsilon, verbose, with_colors);
}
