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

TRTLLM_NAMESPACE_BEGIN

namespace kernels::cutlass_kernels
{

/////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void interleave_fp4_weights_for_Hopper_mixed_gemm_kernel(
    uint8_t* fp4_weight, uint8_t* fp4_weight_interleaved, int const rows, int const cols)
{
    for (int block_id = blockIdx.x; block_id < rows / 2; block_id += gridDim.x)
    {
        for (int partition_id = threadIdx.y; partition_id < cols / 64; partition_id += blockDim.y)
        {
            int lane_id = threadIdx.x;
            int row_id = block_id / 8 * 16 + block_id % 8;

            int mma_id = lane_id / 8;
            int dst_row_id = row_id + (mma_id % 2) * 8;

            int interleaved_lane_id = lane_id / 16 * 16 + (lane_id % 4) * 4 + (lane_id % 8) / 4 * 2;

            int col_id = partition_id * 32 + lane_id;
            int dst_col_id = partition_id * 32 + interleaved_lane_id;

            int index_a = row_id * cols / 2 + col_id;
            int index_b = (row_id + 8) * cols / 2 + col_id;

            uint8_t fp4x2_a = fp4_weight[index_a];
            uint8_t fp4x2_b = fp4_weight[index_b];

            uint8_t fp4_temp_a = (fp4x2_a & 0xF0U) >> 4;
            uint8_t fp4_temp_b = (fp4x2_b & 0x0FU) << 4;

            fp4x2_a = (fp4x2_a & 0x0FU) | fp4_temp_b;
            fp4x2_b = (fp4x2_b & 0xF0U) | fp4_temp_a;

            int dst_id = dst_row_id * cols / 2 + dst_col_id;

            fp4_weight_interleaved[dst_id] = fp4x2_a;
            fp4_weight_interleaved[dst_id + 1] = fp4x2_b;
        }
    }
}

__global__ void interleave_int4_weights_for_Hopper_mixed_gemm_kernel(
    uint8_t* int4_weight, uint8_t* int4_weight_interleaved, int const rows, int const cols)
{
    uint16_t* uint16_ptr = reinterpret_cast<uint16_t*>(int4_weight);
    uint16_t* uint16_interleaved_ptr = reinterpret_cast<uint16_t*>(int4_weight_interleaved);

    for (int block_id = blockIdx.x; block_id < rows / 2; block_id += gridDim.x)
    {
        for (int partition_id = threadIdx.y; partition_id < cols / 64; partition_id += blockDim.y)
        {
            int lane_id = threadIdx.x;

            int row_id = block_id / 8 * 16 + block_id % 8;
            int dst_row_id = row_id + (lane_id % 8) / 4 * 8;

            int mma_id = lane_id / 8;
            int interleaved_lane_id = mma_id * 8 + lane_id % 4 * 2;

            int col_id = partition_id * 16 + lane_id;
            int dst_col_id = partition_id * 16 + interleaved_lane_id;

            int src_id_a = row_id * cols / 4 + col_id;
            int src_id_b = (row_id + 8) * cols / 4 + col_id;

            uint16_t int4x2_a = uint16_ptr[src_id_a];
            uint16_t int4x2_b = uint16_ptr[src_id_b];

            int dst_id = dst_row_id * cols / 4 + dst_col_id;

            uint16_interleaved_ptr[dst_id] = int4x2_a;
            uint16_interleaved_ptr[dst_id + 1] = int4x2_b;
        }
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////

void interleave_fp4_weights_for_Hopper_mixed_gemm(
    uint8_t* fp4_weight, uint8_t* fp4_weight_interleaved, int const rows, int const cols)
{
    dim3 block(32, 32);
    interleave_fp4_weights_for_Hopper_mixed_gemm_kernel<<<1024, block>>>(
        fp4_weight, fp4_weight_interleaved, rows, cols);
}

void interleave_int4_weights_for_Hopper_mixed_gemm(
    uint8_t* int4_weight, uint8_t* int4_weight_interleaved, int const rows, int const cols)
{
    dim3 block(16, 32);
    interleave_int4_weights_for_Hopper_mixed_gemm_kernel<<<1024, block>>>(
        int4_weight, int4_weight_interleaved, rows, cols);
}

} // namespace kernels::cutlass_kernels

TRTLLM_NAMESPACE_END
