/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.  All rights reserved.
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
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/envUtils.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/common/quantization.h"
#include "tensorrt_llm/kernels/cutlass_kernels/cutlass_type_conversion.h"
#include "tensorrt_llm/runtime/common.h"

#include <NvInferRuntime.h>

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <iostream>

namespace tensorrt_llm
{
namespace kernels
{
namespace cuda_core_gemm
{
using SizeType32 = tensorrt_llm::runtime::SizeType32;

struct Params
{
    void const* act;
    void const* weight;
    void* output;
    SizeType32 m, n, k;
    cudaDataType_t inputType;
    cudaDataType_t outputType;
    // gemmPlugin
    float alpha;
    // torch flow
    float const* scale_a;
    float const* scale_b;

    // used by gemmPlugin
    Params(void const* _act, void const* _weight, float _alpha, void* _output, SizeType32 _m, SizeType32 _n,
        SizeType32 _k, cudaDataType_t _inputType, cudaDataType_t _outputType)
        : act(_act)
        , weight(_weight)
        , output(_output)
        , m(_m)
        , n(_n)
        , k(_k)
        , inputType(_inputType)
        , outputType(_outputType)
        , alpha(_alpha)
        , scale_a(nullptr)
        , scale_b(nullptr)
    {
    }

    // used by torch flow
    Params(void const* _act, void const* _weight, void* _output, SizeType32 _m, SizeType32 _n, SizeType32 _k,
        float const* _scale_a, float const* _scale_b, cudaDataType_t _inputType, cudaDataType_t _outputType)
        : act(_act)
        , weight(_weight)
        , output(_output)
        , m(_m)
        , n(_n)
        , k(_k)
        , inputType(_inputType)
        , outputType(_outputType)
        , alpha(1.f)
        , scale_a(_scale_a)
        , scale_b(_scale_b)
    {
    }
};

bool cudaCoreGemmDispatcher(Params const& params, cudaStream_t stream);
} // namespace cuda_core_gemm
} // namespace kernels
} // namespace tensorrt_llm
