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

enum Maskgen {
    FULL = 0,
    RANDOM = 1,
    LINEAR = 2,
};

enum Datagen {
    NORMAL = 0,
    SMALLINT = 1,
    ONES = 2,
};

at::Tensor bmm_nt(const at::Tensor &A,
                  const at::Tensor &B,
                  cudaDataType_t A_type,
                  cudaDataType_t B_type,
                  const float alpha);
at::Tensor matmul_nt(const at::Tensor &A,
                     const at::Tensor &,
                     cudaDataType_t A_type,
                     cudaDataType_t B_typeB,
                     const float alpha);

at::Tensor convert_fp32_to_fp8(const at::Tensor &src, float q_scale, cudaDataType_t src_type);
at::Tensor convert_fp8_to_fp32(const at::Tensor &src, float d_scale, cudaDataType_t dst_type);

at::Tensor pad(const at::Tensor &tensor, const at::Tensor &cu_seqlens, const int s);
at::Tensor unpad(const at::Tensor &tensor, const at::Tensor &cu_seqlens);

std::tuple<at::Tensor, at::Tensor>
full_mask(const int b, const int s, const torch::TensorOptions &options);
std::tuple<at::Tensor, at::Tensor>
rand_mask(const int b, const int s, const torch::TensorOptions &options);
std::tuple<at::Tensor, at::Tensor>
lin_mask(const int b, const int s, const torch::TensorOptions &options);
std::tuple<at::Tensor, at::Tensor>
make_mask(const int b, const int s, const torch::TensorOptions &options, const Maskgen maskgen);

at::Tensor seqlens2mask(const at::Tensor &seqlens, const int s, const at::TensorOptions &options);

at::Tensor
draw_tensor(const at::IntArrayRef dims, const torch::TensorOptions &options, const Datagen datagen);

int check_results(const at::Tensor &out,
                  const at::Tensor &ref,
                  float epsilon,
                  bool verbose,
                  bool with_colors);

struct FP8FpropMeta {
    FP8FpropMeta(const float d_scale_qkv_,
                 const float d_scale_s_,
                 const float q_scale_s_,
                 const float q_scale_o_,
                 at::TensorOptions &options)
        : d_scale_qkv(torch::full({ 1 }, d_scale_qkv_, options.dtype(torch::kFloat32))),
          d_scale_s(torch::full({ 1 }, d_scale_s_, options.dtype(torch::kFloat32))),
          q_scale_s(torch::full({ 1 }, q_scale_s_, options.dtype(torch::kFloat32))),
          q_scale_o(torch::full({ 1 }, q_scale_o_, options.dtype(torch::kFloat32))) {
    }

    at::Tensor d_scale_qkv;
    at::Tensor d_scale_s;
    at::Tensor q_scale_s;
    at::Tensor q_scale_o;
};

struct FP8DgradMeta {
    FP8DgradMeta(const float d_scale_qkv_,
                 const float d_scale_s_,
                 const float d_scale_o_,
                 const float d_scale_do_,
                 const float d_scale_dp_,
                 const float q_scale_s_,
                 const float q_scale_dp_,
                 const float q_scale_dqkv_,
                 at::TensorOptions &options)
        : d_scale_qkv(torch::full({ 1 }, d_scale_qkv_, options.dtype(torch::kFloat32))),
          d_scale_s(torch::full({ 1 }, d_scale_s_, options.dtype(torch::kFloat32))),
          d_scale_o(torch::full({ 1 }, d_scale_o_, options.dtype(torch::kFloat32))),
          d_scale_do(torch::full({ 1 }, d_scale_do_, options.dtype(torch::kFloat32))),
          d_scale_dp(torch::full({ 1 }, d_scale_dp_, options.dtype(torch::kFloat32))),
          q_scale_s(torch::full({ 1 }, q_scale_s_, options.dtype(torch::kFloat32))),
          q_scale_dp(torch::full({ 1 }, q_scale_dp_, options.dtype(torch::kFloat32))),
          q_scale_dqkv(torch::full({ 1 }, q_scale_dqkv_, options.dtype(torch::kFloat32))) {
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

struct FP8TensorMeta {
    FP8TensorMeta(cudaDataType_t dtype_)
        : dtype(dtype_), fp8_max(dtype_ == CUDA_R_8F_E4M3 ? fmha::MAX_E4M3 : fmha::MAX_E5M2) {
    }

    float q_scale(float amax) const {
        float q_scale = pow(2, (int)log2f(fp8_max / amax));
        return q_scale;
        //return fp8_max / amax;
    }

    cudaDataType_t dtype;
    float fp8_max;
};

using MetaMap = std::map<std::string, FP8TensorMeta>;

static MetaMap get_recipe(bool all_e5m2) {

    if( all_e5m2 ) {
        return {
            { "QKV", FP8TensorMeta(CUDA_R_8F_E4M3) },  { "P", FP8TensorMeta(CUDA_R_8F_E4M3) },
            { "expP", FP8TensorMeta(CUDA_R_8F_E4M3) }, { "S", FP8TensorMeta(CUDA_R_8F_E4M3) },
            { "D", FP8TensorMeta(CUDA_R_8F_E4M3) },    { "O", FP8TensorMeta(CUDA_R_8F_E4M3) },
            { "dO", FP8TensorMeta(CUDA_R_8F_E5M2) },   { "dD", FP8TensorMeta(CUDA_R_8F_E5M2) },
            { "dS", FP8TensorMeta(CUDA_R_8F_E5M2) },   { "dP", FP8TensorMeta(CUDA_R_8F_E5M2) },
            { "dQKV", FP8TensorMeta(CUDA_R_8F_E5M2) },
        };
    }

    return {
        { "QKV", FP8TensorMeta(CUDA_R_8F_E4M3) },  { "P", FP8TensorMeta(CUDA_R_8F_E4M3) },
        { "expP", FP8TensorMeta(CUDA_R_8F_E4M3) }, { "S", FP8TensorMeta(CUDA_R_8F_E4M3) },
        { "D", FP8TensorMeta(CUDA_R_8F_E4M3) },    { "O", FP8TensorMeta(CUDA_R_8F_E4M3) },
        { "dO", FP8TensorMeta(CUDA_R_8F_E4M3) },   { "dD", FP8TensorMeta(CUDA_R_8F_E4M3) },
        { "dS", FP8TensorMeta(CUDA_R_8F_E4M3) },   { "dP", FP8TensorMeta(CUDA_R_8F_E5M2) },  // <-
        { "dQKV", FP8TensorMeta(CUDA_R_8F_E4M3) },
    };
}
