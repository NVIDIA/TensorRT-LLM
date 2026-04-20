/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
#pragma once
#include "tensorrt_llm/common/assert.h"
#include <NvInferRuntime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "tensorrt_llm/common/config.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/kernels/quantization.h"
#include "tensorrt_llm/runtime/ipcUtils.h"

TRTLLM_NAMESPACE_BEGIN

namespace kernels::minimax_ar
{
template <typename DType>
struct ElemsPerAccess;

template <>
struct ElemsPerAccess<half>
{
    static constexpr int value = 8;
    using norm_weight_type = common::__nv_bfloat168;
};

template <>
struct ElemsPerAccess<nv_bfloat16>
{
    static constexpr int value = 8;
    using norm_weight_type = common::__nv_bfloat168;
};

template <>
struct ElemsPerAccess<float>
{
    static constexpr int value = 4;
    using norm_weight_type = common::__nv_bfloat164;
};

template <typename DType>
static constexpr int kElemsPerAccess = ElemsPerAccess<DType>::value;

struct MiniMaxReduceRMSParams
{
    int nranks{};
    int rank{};
    nvinfer1::DataType dtype;
    int size_q{};           // numel of Q (num_token * head_dim_q)
    int hidden_dim{};       // head_dim_q
    int size_k{};           // numel of K (num_token * head_dim_k)
    int hidden_dim_k{};     // head_dim_k; must have head_dim_q >= head_dim_k
    void** workspace{};
    void* allreduce_in{};   // Q input
    void* rms_norm_out{};   // Q output
    void* rms_gamma{};      // Q norm weight
    void* allreduce_in_k{}; // K input (nullptr for single-matrix path)
    void* rms_norm_out_k{}; // K output
    void* rms_gamma_k{};    // K norm weight
    float rms_eps{};
    cudaStream_t stream{};
    bool trigger_completion_at_end = true;
};

void minimax_reduce_rms_op(MiniMaxReduceRMSParams const& params);

} // namespace kernels::minimax_ar

TRTLLM_NAMESPACE_END
