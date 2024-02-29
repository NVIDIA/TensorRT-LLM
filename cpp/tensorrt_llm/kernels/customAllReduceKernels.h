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

#include <assert.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <iostream>

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/tensor.h"

namespace tensorrt_llm::kernels
{

constexpr size_t WARP_SIZE = 32;
constexpr size_t MAX_ALL_REDUCE_BLOCKS = 24;
constexpr size_t MAX_RANKS_PER_NODE = 8;
constexpr size_t DEFAULT_BLOCK_SIZE = 1024;

// Warning: python definition is in tensorrt_llm/functional.py
// they must be kept in sync
enum class AllReduceStrategyType : int8_t
{
    RING = 0,
    ONESHOT = 1,
    TWOSHOT = 2,
    AUTO = 3,
};

struct AllReduceParams
{
    size_t elts_total;
    size_t elts_per_rank;
    size_t elts_per_block;
    size_t rank_offset;
    size_t ranks_per_node, rank, local_rank;
    uint32_t barrier_flag;
    uint32_t* peer_barrier_ptrs_in[MAX_RANKS_PER_NODE];
    uint32_t* peer_barrier_ptrs_out[MAX_RANKS_PER_NODE];
    void* peer_comm_buffer_ptrs[MAX_RANKS_PER_NODE];
    void* local_output_buffer_ptr;

    static AllReduceParams deserialize(const int32_t* buffer, size_t tpSize, size_t tpRank, uint32_t flag_value);
};

template <typename T>
void invokeOneOrTwoShotAllReduceKernel(AllReduceParams& param, AllReduceStrategyType strat, cudaStream_t stream);

void invokeMultiGpuBarrier(AllReduceParams& param, cudaStream_t stream);

void customAllReduce(kernels::AllReduceParams& params, void* data, size_t elts, size_t size_per_elem,
    common::datatype_enum dataType, AllReduceStrategyType strat, cudaStream_t stream);

} // namespace tensorrt_llm::kernels
