/*
 * Copyright (c) 2022-2025, NVIDIA CORPORATION.  All rights reserved.
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
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaUtils.h"

namespace tensorrt_llm::kernels::group_rms_norm
{

template <typename DType>
void GroupRMSNormKernel(DType** inputs, DType** weights, DType** outputs, uint32_t const* input_dims,
    uint32_t const* input_strides, uint32_t const* output_strides, const uint32_t batch_size, const uint32_t num_inputs,
    float const eps, float const weight_bias, bool enable_weights, cudaStream_t stream);

} // namespace tensorrt_llm::kernels::group_rms_norm
