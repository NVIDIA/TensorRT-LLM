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

#pragma once

enum Maskgen
{
    FULL = 0,
    RANDOM = 1,
    LINEAR = 2,
};

enum Datagen
{
    NORMAL = 0,
    SMALLINT = 1,
    ONES = 2,
};

at::Tensor bmm_nt(
    at::Tensor const& A, at::Tensor const& B, cudaDataType_t A_type, cudaDataType_t B_type, float const alpha);
at::Tensor matmul_nt(
    at::Tensor const& A, at::Tensor const&, cudaDataType_t A_type, cudaDataType_t B_typeB, float const alpha);

at::Tensor convert_fp32_to_fp8(at::Tensor const& src, float q_scale, cudaDataType_t src_type);
at::Tensor convert_fp8_to_fp32(at::Tensor const& src, float d_scale, cudaDataType_t dst_type);

at::Tensor pad(at::Tensor const& tensor, at::Tensor const& cu_seqlens, int const s);
at::Tensor unpad(at::Tensor const& tensor, at::Tensor const& cu_seqlens);

std::tuple<at::Tensor, at::Tensor> full_mask(int const b, int const s, torch::TensorOptions const& options);
std::tuple<at::Tensor, at::Tensor> rand_mask(int const b, int const s, torch::TensorOptions const& options);
std::tuple<at::Tensor, at::Tensor> lin_mask(int const b, int const s, torch::TensorOptions const& options);
std::tuple<at::Tensor, at::Tensor> make_mask(
    int const b, int const s, torch::TensorOptions const& options, Maskgen const maskgen);

at::Tensor seqlens2mask(at::Tensor const& seqlens, int const s, at::TensorOptions const& options);

at::Tensor draw_tensor(at::IntArrayRef const dims, torch::TensorOptions const& options, Datagen const datagen);

int check_results(at::Tensor const& out, at::Tensor const& ref, float epsilon, bool verbose, bool with_colors);

struct FP8FpropMeta
{
    FP8FpropMeta(float const d_scale_qkv_, float const d_scale_s_, float const q_scale_s_, float const q_scale_o_,
        at::TensorOptions& options)
        : d_scale_qkv(torch::full({1}, d_scale_qkv_, options.dtype(torch::kFloat32)))
        , d_scale_s(torch::full({1}, d_scale_s_, options.dtype(torch::kFloat32)))
        , q_scale_s(torch::full({1}, q_scale_s_, options.dtype(torch::kFloat32)))
        , q_scale_o(torch::full({1}, q_scale_o_, options.dtype(torch::kFloat32)))
    {
    }

    at::Tensor d_scale_qkv;
    at::Tensor d_scale_s;
    at::Tensor q_scale_s;
    at::Tensor q_scale_o;
};

struct FP8DgradMeta
{
    FP8DgradMeta(float const d_scale_qkv_, float const d_scale_s_, float const d_scale_o_, float const d_scale_do_,
        float const d_scale_dp_, float const q_scale_s_, float const q_scale_dp_, float const q_scale_dqkv_,
        at::TensorOptions& options)
        : d_scale_qkv(torch::full({1}, d_scale_qkv_, options.dtype(torch::kFloat32)))
        , d_scale_s(torch::full({1}, d_scale_s_, options.dtype(torch::kFloat32)))
        , d_scale_o(torch::full({1}, d_scale_o_, options.dtype(torch::kFloat32)))
        , d_scale_do(torch::full({1}, d_scale_do_, options.dtype(torch::kFloat32)))
        , d_scale_dp(torch::full({1}, d_scale_dp_, options.dtype(torch::kFloat32)))
        , q_scale_s(torch::full({1}, q_scale_s_, options.dtype(torch::kFloat32)))
        , q_scale_dp(torch::full({1}, q_scale_dp_, options.dtype(torch::kFloat32)))
        , q_scale_dqkv(torch::full({1}, q_scale_dqkv_, options.dtype(torch::kFloat32)))
    {
    }

    at::Tensor d_scale_qkv;
    at::Tensor d_scale_s;
    at::Tensor d_scale_o;
    at::Tensor d_scale_do;
    at::Tensor d_scale_dp;

    at::Tensor q_scale_s;
    at::Tensor q_scale_dp;
    at::Tensor q_scale_dqkv;
};

struct FP8TensorMeta
{
    FP8TensorMeta(cudaDataType_t dtype_)
        : dtype(dtype_)
        , fp8_max(dtype_ == CUDA_R_8F_E4M3 ? fmha::MAX_E4M3 : fmha::MAX_E5M2)
    {
    }

    float q_scale(float amax) const
    {
        float q_scale = pow(2, (int) log2f(fp8_max / amax));
        return q_scale;
        // return fp8_max / amax;
    }

    cudaDataType_t dtype;
    float fp8_max;
};

using MetaMap = std::map<std::string, FP8TensorMeta>;

static MetaMap get_recipe(bool all_e5m2)
{

    if (all_e5m2)
    {
        return {
            {"QKV", FP8TensorMeta(CUDA_R_8F_E4M3)},
            {"P", FP8TensorMeta(CUDA_R_8F_E4M3)},
            {"expP", FP8TensorMeta(CUDA_R_8F_E4M3)},
            {"S", FP8TensorMeta(CUDA_R_8F_E4M3)},
            {"D", FP8TensorMeta(CUDA_R_8F_E4M3)},
            {"O", FP8TensorMeta(CUDA_R_8F_E4M3)},
            {"dO", FP8TensorMeta(CUDA_R_8F_E5M2)},
            {"dD", FP8TensorMeta(CUDA_R_8F_E5M2)},
            {"dS", FP8TensorMeta(CUDA_R_8F_E5M2)},
            {"dP", FP8TensorMeta(CUDA_R_8F_E5M2)},
            {"dQKV", FP8TensorMeta(CUDA_R_8F_E5M2)},
        };
    }

    return {
        {"QKV", FP8TensorMeta(CUDA_R_8F_E4M3)},
        {"P", FP8TensorMeta(CUDA_R_8F_E4M3)},
        {"expP", FP8TensorMeta(CUDA_R_8F_E4M3)},
        {"S", FP8TensorMeta(CUDA_R_8F_E4M3)},
        {"D", FP8TensorMeta(CUDA_R_8F_E4M3)},
        {"O", FP8TensorMeta(CUDA_R_8F_E4M3)},
        {"dO", FP8TensorMeta(CUDA_R_8F_E4M3)},
        {"dD", FP8TensorMeta(CUDA_R_8F_E4M3)},
        {"dS", FP8TensorMeta(CUDA_R_8F_E4M3)},
        {"dP", FP8TensorMeta(CUDA_R_8F_E5M2)}, // <-
        {"dQKV", FP8TensorMeta(CUDA_R_8F_E4M3)},
    };
}
