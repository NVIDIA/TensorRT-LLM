/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.
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

#include "tensorrt_llm/common/int8Utils.cuh"
#include "tensorrt_llm/kernels/cutlass_kernels/cutlass_preprocessors.h"
#include "tensorrt_llm/kernels/cutlass_kernels/fpA_intB_gemm/fpA_intB_gemm.h"
#include <assert.h>
#include <cmath>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

namespace tensorrt_llm
{
namespace kernels
{

using cutlass_kernels::QuantType;
using cutlass_kernels::ActivationType;

template <typename WT, typename AT>
void weight_only_gemv_launcher(const AT* input, const WT* weight, const AT* scale_list, const AT* bias, AT* output,
    const int k, const int n, ActivationType activation, QuantType qtype, cudaStream_t stream)
{
    assert(false);
}

template <>
void weight_only_gemv_launcher(const half* input, const int8_t* weight, const half* scale_list, const half* bias,
    half* output, const int k, const int n, ActivationType activation, QuantType qtype, cudaStream_t stream);
} // namespace kernels
} // namespace tensorrt_llm
