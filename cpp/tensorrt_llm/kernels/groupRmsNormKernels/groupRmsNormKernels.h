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
#include <map>

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

// Logistic regression model for predicting when the large batch kernel is faster
struct Model
{
    float batch_size;
    // Number of warps to launch for the base kernel
    float base_warps;
    // Ratio of the concurrent_block_per_sm for the large batch kernel vs the base kernel
    float scheduling_efficiency_ratio;
    // Intercept of the logistic regression model
    float intercept;
};

// Trained parameters for the logistic regression model
// For each major version of the Compute Capability
inline std::map<int, Model> gpu_models = {
    {10, {-0.004011f, -0.180179f, -0.396733f, 6.714080f}},
    {9, {-0.006522f, -0.178540f, -0.558174f, 8.210834f}},
};

template <int n>
void GroupRMSNormBaseKernelLauncher(GroupRMSParams<n>& params);

template <int n>
void GroupRMSNormKernelLargeBatchLauncher(GroupRMSParams<n>& params);

template <int n>
void GroupRMSNormKernelLauncherWithHeuristic(GroupRMSParams<n>& params);

} // namespace tensorrt_llm::kernels::group_rms_norm
