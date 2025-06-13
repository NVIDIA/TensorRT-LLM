/*
 * MIT License
 *
 * Copyright (c) 2025 DeepSeek
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
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
 *
 * reference: https://github.com/deepseek-ai/FlashMLA
 */

#include "flash_fwd_mla_kernel.h"

[[maybe_unused]] static constexpr int MaxBatchSize = 4096;

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
__launch_bounds__(256, 1, 1)
#endif // __CUDA_ARCH__ >= 900
    __global__ void get_mla_metadata_kernel(__grid_constant__ const Mla_metadata_params params)
{
    int* seqlens_k_ptr = params.seqlens_k_ptr;
    int* tile_scheduler_metadata_ptr = params.tile_scheduler_metadata_ptr;
    int* num_splits_ptr = params.num_splits_ptr;
    int batch_size = params.batch_size;
    int block_size_n = params.block_size_n;
    int fixed_overhead_num_blocks = params.fixed_overhead_num_blocks;
    int num_sm_parts = params.num_sm_parts;

    __shared__ int num_blocks_shared[MaxBatchSize];
    __shared__ int num_splits_shared[MaxBatchSize];

    int total_num_blocks = 0;
    for (int i = threadIdx.x; i < batch_size; i += 32)
    {
        int num_blocks = cutlass::ceil_div(seqlens_k_ptr[i], block_size_n);
        total_num_blocks += num_blocks + fixed_overhead_num_blocks;
        num_blocks_shared[i] = num_blocks;
    }
    for (int offset = 16; offset >= 1; offset /= 2)
    {
        total_num_blocks += __shfl_xor_sync(uint32_t(-1), total_num_blocks, offset);
    }
    __syncwarp();

    if (threadIdx.x == 0)
    {
        int payload = cutlass::ceil_div(total_num_blocks, num_sm_parts) + fixed_overhead_num_blocks;

        int now_idx = 0, now_block = 0, now_n_split_idx = 0, cum_num_splits = 0;
        num_splits_shared[0] = 0;
        for (int i = 0; i < num_sm_parts; ++i)
        {
            int tile_scheduler_metadata0[4], tile_scheduler_metadata1;
            tile_scheduler_metadata0[0] = now_idx;
            tile_scheduler_metadata0[1] = now_block * block_size_n;
            tile_scheduler_metadata1 = now_n_split_idx;
            int remain_payload = payload;
            while (now_idx < batch_size)
            {
                int num_blocks = num_blocks_shared[now_idx];
                int now_remain_blocks = num_blocks - now_block;
                if (remain_payload >= now_remain_blocks + fixed_overhead_num_blocks)
                {
                    cum_num_splits += now_n_split_idx + 1;
                    num_splits_shared[now_idx + 1] = cum_num_splits;
                    remain_payload -= now_remain_blocks + fixed_overhead_num_blocks;
                    ++now_idx;
                    now_block = 0;
                    now_n_split_idx = 0;
                }
                else
                {
                    if (remain_payload - fixed_overhead_num_blocks > 0)
                    {
                        now_block += remain_payload - fixed_overhead_num_blocks;
                        ++now_n_split_idx;
                        remain_payload = 0;
                    }
                    break;
                }
            }
            tile_scheduler_metadata0[2] = now_block > 0 ? now_idx : now_idx - 1;
            tile_scheduler_metadata0[3] = now_block > 0 ? now_block * block_size_n : seqlens_k_ptr[now_idx - 1];
            *reinterpret_cast<int4*>(tile_scheduler_metadata_ptr + i * TileSchedulerMetaDataSize)
                = *reinterpret_cast<int4*>(tile_scheduler_metadata0);
            tile_scheduler_metadata_ptr[i * TileSchedulerMetaDataSize + 4] = tile_scheduler_metadata1;
        }
        FLASH_DEVICE_ASSERT(now_idx == batch_size && now_block == 0 && now_n_split_idx == 0);
    }
    __syncwarp();

    for (int i = threadIdx.x; i <= batch_size; i += 32)
    {
        num_splits_ptr[i] = num_splits_shared[i];
    }
}

void get_mla_metadata_func(Mla_metadata_params& params, cudaStream_t stream)
{
    FLASH_ASSERT(params.batch_size < MaxBatchSize);
    get_mla_metadata_kernel<<<1, 32, 0, stream>>>(params);
    CHECK_CUDA_KERNEL_LAUNCH();
}
