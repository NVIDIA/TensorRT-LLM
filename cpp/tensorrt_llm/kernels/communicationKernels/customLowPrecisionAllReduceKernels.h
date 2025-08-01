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

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/kernels/customAllReduceKernels.h"
#include <NvInferRuntime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <vector>

namespace tensorrt_llm::kernels
{

constexpr int LP_ALLREDUCE_MAX_BLOCKS = 8;
constexpr int LP_ALLREDUCE_WARPSIZE = 32;
constexpr int LP_ALLREDUCE_DEFAULT_BLOCK_SIZE = 512;
constexpr int LP_ALLREDUCE_WARP_NUM_PER_BLOCK = 16;
constexpr int LP_ALLREDUCE_BYTES_PER_LOAD = 16;
constexpr int LP_ALLREDUCE_NUMA_NUM = 2;
constexpr int LP_ALLREDUCE_MAX_RANKS_PER_NUMA = 4;
constexpr int LP_ALLREDUCE_BUFFER_DUPLICATE = 16;
constexpr int LP_ALLREDUCE_BUFFER_CHUNKS = 8;
constexpr int LP_ALLREDUCE_HIER_STAGE_NUM = 3;
constexpr int LP_ALLREDUCE_RANKS_PER_NUMA = 4;
constexpr int LP_ALLREDUCE_MAX_ELTS_IN_WORKSPACE = 32 * 1024 * 1024;
constexpr int LP_ALLREDUCE_MIN_ELTS_THRESHOLD = 8 * 1024 * 1024;
constexpr int LP_ALLREDUCE_MAX_TP_SIZE = 8;
constexpr int LP_ALLREDUCE_MAX_RANKS_PER_NODE = 16;

struct StaticLowPrecisionBuffers
{
    void* peer_comm_buffer_ptrs[LP_ALLREDUCE_MAX_TP_SIZE * 2];
    uint64_t* peer_barrier_ptrs_in[LP_ALLREDUCE_MAX_TP_SIZE];
    uint64_t* peer_barrier_ptrs_out[LP_ALLREDUCE_MAX_TP_SIZE];
    int64_t* flag_ptr;
    bool initialized = false;
    size_t tpSize = 0;
};

void initialize_static_lowprecision_buffers(int64_t* buffer, size_t tpSize);

std::vector<size_t> splitNumber(size_t number);

struct LowPrecisionAllReduceParams
{
    size_t elts_total;
    size_t elts_per_rank;
    size_t elts_per_block;
    size_t rank_offset;
    int32_t ranks_per_node, rank, local_rank;
    uint64_t barrier_flag;
    uint64_t* peer_barrier_ptrs_in[LP_ALLREDUCE_MAX_RANKS_PER_NODE];
    uint64_t* peer_barrier_ptrs_out[LP_ALLREDUCE_MAX_RANKS_PER_NODE];
    void* peer_comm_buffer_ptrs[LP_ALLREDUCE_MAX_RANKS_PER_NODE];
    void* local_output_buffer_ptr;
    void const* local_input_buffer_ptr;

    // for low precision
    size_t buffer_elts_per_rank;
    size_t buffer_offset;

    // for low precision hier
    uint32_t num_rounds = 0;
    uint32_t num_rounds_fence = 0;
    uint32_t block_num = 0;
    int32_t numa_rank = -1;

    void* inputs_inside_numa[4];

    void* rs_buffers[LP_ALLREDUCE_MAX_BLOCKS];
    void* ar_buffers[LP_ALLREDUCE_MAX_BLOCKS];
    void* ar_peer_buffers_cross_numa[LP_ALLREDUCE_MAX_BLOCKS];
    void* ag_peer_buffers_inside_numa[LP_ALLREDUCE_MAX_BLOCKS * 4];

    // for low precision hier handshake rs stage
    uint64_t* rs_send_flags[LP_ALLREDUCE_MAX_BLOCKS];
    uint64_t* rs_ack_flags[LP_ALLREDUCE_MAX_BLOCKS]; // 2*flags
    uint64_t* rs_notify_local_flags[LP_ALLREDUCE_MAX_BLOCKS];
    uint64_t* rs_notify_remote_flags[LP_ALLREDUCE_MAX_BLOCKS];

    // for low precision hier handshake ar stage
    uint64_t* ar_send_flags[LP_ALLREDUCE_MAX_BLOCKS];
    uint64_t* ar_ack_peer_rs_flags[LP_ALLREDUCE_MAX_BLOCKS];
    uint64_t* ar_ack_flags[LP_ALLREDUCE_MAX_BLOCKS];
    uint64_t* ar_notify_rs_local_flags[LP_ALLREDUCE_MAX_BLOCKS];
    uint64_t* ar_notify_rs_remote_flags[LP_ALLREDUCE_MAX_BLOCKS];
    uint64_t* ar_notify_ag_flags[LP_ALLREDUCE_MAX_BLOCKS];

    // for low precision hier handshake ag stage
    uint64_t* ag_send_flags[LP_ALLREDUCE_MAX_BLOCKS];
    uint64_t* ag_ack_peer_inside_numa_flags[LP_ALLREDUCE_MAX_BLOCKS];        // 3*flags , 3 is other rank inside numa
    uint64_t* ag_notify_peer_inside_numa_flags[LP_ALLREDUCE_MAX_BLOCKS * 4]; // 3*flags , 3 is other rank inside numa

    static LowPrecisionAllReduceParams deserialize(
        size_t tpSize, size_t tpRank, nvinfer1::DataType dataType, int token_num, int hidden_size);
    static LowPrecisionAllReduceParams deserialize_hier(
        size_t tpSize, size_t tpRank, nvinfer1::DataType dataType, int token_num, int hidden_size);
};

bool lowPrecisionConfigurationSupported(size_t msg_size, size_t n_ranks);

void customLowPrecisionAllReduce(
    kernels::LowPrecisionAllReduceParams& params, nvinfer1::DataType dataType, cudaStream_t stream);

int32_t max_workspace_size_lowprecision(int32_t tp_size);
} // namespace tensorrt_llm::kernels
