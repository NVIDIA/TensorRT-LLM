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

#include <cuda_runtime_api.h>

#include <cub/device/device_scan.cuh>

#include "cumsumLastDim.h"

namespace tensorrt_llm
{
namespace kernels
{
template <typename input_t>
size_t invokeComputeCumsumLastDimWorkspaceSize(int input_length)
{
    input_t* iodata = nullptr;
    size_t temp_storage_bytes;
    cub::DeviceScan::InclusiveSum(nullptr, temp_storage_bytes, iodata, iodata, input_length);
    return temp_storage_bytes;
}

#define INSTANTIATE_COMPUTE_CUMSUM_LastDim_WORKSPACE_SIZE_DATA_TYPE(input_t)                                           \
    template size_t invokeComputeCumsumLastDimWorkspaceSize<input_t>(int input_length)

INSTANTIATE_COMPUTE_CUMSUM_LastDim_WORKSPACE_SIZE_DATA_TYPE(int);
INSTANTIATE_COMPUTE_CUMSUM_LastDim_WORKSPACE_SIZE_DATA_TYPE(float);
INSTANTIATE_COMPUTE_CUMSUM_LastDim_WORKSPACE_SIZE_DATA_TYPE(half);
#ifdef ENABLE_BF16
INSTANTIATE_COMPUTE_CUMSUM_LastDim_WORKSPACE_SIZE_DATA_TYPE(__nv_bfloat16);
#endif
#undef INSTANTIATE_COMPUTE_CUMSUM_LastDim_WORKSPACE_SIZE_DATA_TYPE

///////////////

template <typename input_t>
void invokeCumsumLastDim(int batch_size, int input_length, void const* __restrict__ input, void* __restrict__ output,
    void* d_temp_storage, size_t temp_storage_bytes, cudaStream_t stream)
{
    for (int i = 0; i < batch_size; i++)
    {
        input_t const* input_ptr = reinterpret_cast<input_t const*>(input) + i * input_length;
        input_t* output_ptr = reinterpret_cast<input_t*>(output) + i * input_length;
        cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, input_ptr, output_ptr, input_length, stream);
    }
}

#define INSTANTIATE_CUMSUM_LastDim_DATA_TYPE(input_t)                                                                  \
    template void invokeCumsumLastDim<input_t>(int batch_size, int input_length, const void* __restrict__ input,       \
        void* __restrict__ output, void* workspace, size_t temp_storage_bytes, cudaStream_t stream)

INSTANTIATE_CUMSUM_LastDim_DATA_TYPE(int);
INSTANTIATE_CUMSUM_LastDim_DATA_TYPE(float);
INSTANTIATE_CUMSUM_LastDim_DATA_TYPE(half);
#ifdef ENABLE_BF16
INSTANTIATE_CUMSUM_LastDim_DATA_TYPE(__nv_bfloat16);
#endif
#undef INSTANTIATE_CUMSUM_LastDim_DATA_TYPE

} // namespace kernels
} // namespace tensorrt_llm
