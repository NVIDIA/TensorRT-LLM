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

#include "moe_gemm_mixed_utils.h"

namespace tensorrt_llm::kernels::cutlass_kernels
{

__global__ void interleave_fp4_for_Hopper_mixed_gemm_kernel(
    uint8_t* weight, uint8_t* weight_interleaved, int const rows, int const cols)
{
    for (int block_id = blockIdx.x; block_id < rows / 2; block_id += gridDim.x)
    {
        for (int col_id = threadIdx.x; col_id < cols / 2; col_id += blockDim.x)
        {
            int row_id = block_id / 8 * 16 + block_id % 8;

            int index_a = row_id * cols / 2 + col_id;
            int index_b = (row_id + 8) * cols / 2 + col_id;

            uint8_t fp4x2_a = weight[index_a];
            uint8_t fp4x2_b = weight[index_b];

            uint8_t fp4_temp_a = (fp4x2_a & 0xF0U) >> 4;
            uint8_t fp4_temp_b = (fp4x2_b & 0x0FU) << 4;

            fp4x2_a = (fp4x2_a & 0x0FU) | fp4_temp_b;
            fp4x2_b = (fp4x2_b & 0xF0U) | fp4_temp_a;

            weight_interleaved[index_a] = fp4x2_a;
            weight_interleaved[index_b] = fp4x2_b;
        }
    }
}

void interleave_fp4_for_Hopper_mixed_gemm(uint8_t* weight, uint8_t* weight_interleaved, int const rows, int const cols)
{
    // column-major input
    interleave_fp4_for_Hopper_mixed_gemm_kernel<<<1024, 1024>>>(weight, weight_interleaved, rows, cols);
}

} // namespace tensorrt_llm::kernels::cutlass_kernels
