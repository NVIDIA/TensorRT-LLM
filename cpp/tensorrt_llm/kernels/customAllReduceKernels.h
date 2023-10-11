/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.
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

#include <assert.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <iostream>

#include "tensorrt_llm/common/cudaUtils.h"

#define CUSTOM_AR_SIZE_THRESHOLD 50331648
#define MAX_ALL_REDUCE_BLOCKS 24
#define FLAG(a) ((uint32_t) ((a) % 0x146))
#define MAX_RANKS_PER_NODE 8
#define WARP_SIZE 32
#define DEFAULT_BLOCK_SIZE 1024
#define DEFAULT_ALGO_AR_SIZE_THRESHOLD 393216

namespace tensorrt_llm::kernels
{

#ifdef ENABLE_BF16
typedef struct bf168
{
    __nv_bfloat162 x;
    __nv_bfloat162 y;
    __nv_bfloat162 z;
    __nv_bfloat162 w;
} bf168;
#endif

struct AllReduceIpcMemHandles
{
    cudaIpcMemHandle_t peer_barrier_ipc_handles[MAX_RANKS_PER_NODE];
    cudaIpcMemHandle_t peer_comm_buffer_ipc_handles[MAX_RANKS_PER_NODE];
};

struct AllReduceParams
{
    size_t elts_total;
    size_t elts_per_rank;
    size_t elts_per_block;
    size_t rank_offset;
    size_t ranks_per_node, rank, local_rank, node_id;
    uint32_t barrier_flag;
    uint32_t* peer_barrier_ptrs[MAX_RANKS_PER_NODE];
    void* peer_comm_buffer_ptrs[MAX_RANKS_PER_NODE];
    void* local_output_buffer_ptr;
    AllReduceIpcMemHandles ipc_mem_handles;
};

template <typename T>
void invokeOneOrTwoShotAllReduceKernel(AllReduceParams& param, cudaStream_t stream);

void kernelLaunchConfig(int& blocks_per_grid, int& threads_per_block, size_t elts, int kernel_algo);

} // namespace tensorrt_llm::kernels
