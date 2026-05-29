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

#include "tensorrt_llm/common/config.h"
#include "tensorrt_llm/runtime/common.h"

#include <NvInferRuntime.h>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{
namespace cuda_core_gemm_w4a16_nvfp4
{
using SizeType32 = tensorrt_llm::runtime::SizeType32;

struct Params
{
    void const* act;
    void const* weight;
    void const* weightScale;
    float const* weightGlobalScale;
    void* output;
    SizeType32 m, n, k;
    cudaDataType_t inputType;
    cudaDataType_t outputType;

    Params(void const* act_, void const* weight_, void const* weightScale_, float const* weightGlobalScale_,
        void* output_, SizeType32 m_, SizeType32 n_, SizeType32 k_, cudaDataType_t inputType_,
        cudaDataType_t outputType_)
        : act(act_)
        , weight(weight_)
        , weightScale(weightScale_)
        , weightGlobalScale(weightGlobalScale_)
        , output(output_)
        , m(m_)
        , n(n_)
        , k(k_)
        , inputType(inputType_)
        , outputType(outputType_)
    {
    }
};

bool cudaCoreGemmDispatcher(Params const& params, cudaStream_t stream);

} // namespace cuda_core_gemm_w4a16_nvfp4
} // namespace kernels

TRTLLM_NAMESPACE_END
