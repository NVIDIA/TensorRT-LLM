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
#include "cuda_runtime.h"
#include "tensorrt_llm/common/config.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/dataType.h"
#include "ub_allocator.h"

namespace tensorrt_llm::runtime::ub
{
TRTLLM_API void ub_initialize(tensorrt_llm::runtime::WorldConfig const& world_config);
TRTLLM_API void ub_initialize(int tp_size);
TRTLLM_API bool ub_is_initialized();
TRTLLM_API UBBuffer ub_allocate(size_t bytes);
TRTLLM_API void ub_deallocate(void* addr);
TRTLLM_API UBBuffer ub_get(int idx);
TRTLLM_API communicator* ub_comm();
TRTLLM_API bool ub_supported();
}; // namespace tensorrt_llm::runtime::ub

namespace tensorrt_llm::kernels::ub
{
using namespace tensorrt_llm::runtime::ub;
TRTLLM_API void allreduce2_userbuff_inplace_launcher(int const handler, size_t const offset, size_t const elements,
    nvinfer1::DataType dataType, communicator* comm, cudaStream_t stream = 0);

TRTLLM_API int allgather2_userbuff_residual_launcher(int const handler, size_t const offset, size_t const elements,
    int const hidden_size, void* residual, nvinfer1::DataType dataType, communicator* comm, cudaStream_t stream,
    bool force_enable = false);

TRTLLM_API int allreduce2_userbuff_rmsnorm_launcher(int const handler, size_t const offset, int const out_handler,
    size_t const out_offset, size_t const elements, int const hidden_size, void* beta, void* gamma, float eps,
    void* residual_in, void* residual_out, nvinfer1::DataType dataType, communicator* comm, cudaStream_t stream);

TRTLLM_API int allreduce2_userbuff_inplace_rmsnorm_quant_launcher(int const handler, size_t const offset,
    int const out_handler, size_t const out_offset, size_t const elements, int const hidden_size, void* beta,
    void* gamma, float eps, float* scalefactor, void* residual_in, void* residual_out, nvinfer1::DataType dataType,
    communicator* comm, cudaStream_t stream);
TRTLLM_API int allreduce2_userbuff_inplace_rmsnorm_quant_fp4_launcher(int const handler, size_t const offset,
    int const out_handler, size_t const out_offset, int const scale_handler, size_t const scale_offset,
    size_t const elements, int const hidden_size, void* beta, void* gamma, float eps, float* scalefactor,
    void* residual_in, void* residual_out, nvinfer1::DataType dataType, communicator* comm, cudaStream_t stream);
} // namespace tensorrt_llm::kernels::ub
