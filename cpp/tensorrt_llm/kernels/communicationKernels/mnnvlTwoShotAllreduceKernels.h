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

#include <NvInferRuntime.h>
#include <array>
#include <cstddef>
#include <cstdint>

namespace tensorrt_llm::kernels::mnnvl
{
struct AllReduceParams
{
    int nranks;
    int rank;
    nvinfer1::DataType dtype;
    int buffer_M;
    int num_tokens;
    int token_dim;
    uint32_t buffer_size;
    void** buffer_ptrs_dev;
    void* multicast_ptr;
    void* buffer_flags;
    bool wait_for_results;

    void* input;
    void* output;
    cudaStream_t stream;
};

void twoshot_allreduce_op(AllReduceParams const& params);

struct RMSNormParams
{
    void* residual_output;
    void* output;
    void const* input;
    void const* gamma;
    double epsilon;
    void* residual;
    uint32_t buffer_size;
    uint32_t* buffer_flags;
    int batch;
    int hidden_dim;
    cudaStream_t stream;
    nvinfer1::DataType dtype;
};

void twoshot_rmsnorm_op(RMSNormParams const& params);
} // namespace tensorrt_llm::kernels::mnnvl
