/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
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

TRTLLM_NAMESPACE_BEGIN

namespace kernels::cutlass_kernels
{

// IMPORTANT: Keep the same order of activation functions in this enum and the activation functions in
// cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_kernels.cu::doActivationKernel().
enum class ActivationType
{
    InvalidType = 0,
    Identity = 1,
    Gelu = 2,
    Relu = 3,
    Silu = 4,
    Swiglu = 5,
    Geglu = 6,
    SwigluBias = 7,
    Relu2 = 8,
};

} // namespace kernels::cutlass_kernels

TRTLLM_NAMESPACE_END
