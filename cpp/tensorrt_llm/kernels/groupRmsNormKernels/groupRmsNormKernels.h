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
#include <NvInferRuntime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaUtils.h"

namespace tensorrt_llm::kernels::group_rms_norm
{

template </*number of inputs*/ int n>
struct GroupRMSParams
{
    float4 const* inputs[n];
    float4* outputs[n];
    float4 const* weights[n];
    int input_last_dims[n];
    int input_strides[n];
    int output_strides[n];
    int warp_input_idx[32];
    int warp_prefix_sum[n + 1];
    int batch_size;
    int num_inputs;
    float eps;
    float weight_bias;
    bool enable_weights;
    nvinfer1::DataType dtype;
    cudaStream_t stream;
};

template <int n>
void GroupRMSNormKernelLauncher(GroupRMSParams<n> params);

template <int n>
void GroupRMSNormKernelLargeBatchLauncher(GroupRMSParams<n> params);

} // namespace tensorrt_llm::kernels::group_rms_norm
