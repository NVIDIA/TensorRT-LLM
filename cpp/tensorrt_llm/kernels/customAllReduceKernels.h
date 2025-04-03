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
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaUtils.h"

namespace tensorrt_llm::kernels
{

constexpr size_t WARP_SIZE = 32;
constexpr size_t MAX_ALL_REDUCE_BLOCKS = 24;
constexpr size_t MAX_RANKS_PER_NODE = 16;
constexpr size_t DEFAULT_BLOCK_SIZE = 512;

namespace reduce_fusion::details
{
static constexpr int kBytesPerAccess = 16;
static constexpr int kWarpSize = 32;
static constexpr int kMaxCtaSize = 1024;
static constexpr int kClusterMaxSize = 8;
static constexpr int kLamportTokenNumThreshold = 16;
static constexpr int kLamportHiddenSizeThreshold = 256;
}; // namespace reduce_fusion::details

// Warning: python definition is in tensorrt_llm/functional.py
// they must be kept in sync
enum class AllReduceStrategyType : int8_t
{
    NCCL = 0,
    ONESHOT = 1,
    TWOSHOT = 2,
    UB = 3,
    AUTO = 4,
};

enum class AllReduceStrategyConfig : int8_t
{
    USE_MEMCPY = 1 << 0,
    PUSH_MODE = 1 << 1,
};

enum class AllReduceFusionOp : int8_t
{
    NONE = 0,
    RESIDUAL_RMS_NORM = 1,
    LAST_PROCESS_FOR_UB = 2,
    RESIDUAL_RMS_PREPOST_NORM = 3,
    RESIDUAL_RMS_NORM_QUANT_FP8 = 4,
    RESIDUAL_RMS_NORM_QUANT_NVFP4 = 5,
    MOE_ALLREDUCE_RESIDUAL_RMS_NORM = 6,
    RESIDUAL_RMS_NORM_AND_QUANT_NVFP4 = 7,
};

struct AllReduceFusionParams
{
    AllReduceFusionParams()
        : bias_buffer(nullptr)
        , residual_buffer(nullptr)
        , weight_buffer(nullptr)
        , weight_buffer_pre_residual_norm(nullptr)
        , intermediate_buffer(nullptr)
    {
    }

    // gemm bias
    void const* bias_buffer;
    // residuial add
    void const* residual_buffer;
    // rms norm
    int hidden_size;                             // equal to normalized_shape
    void const* weight_buffer;                   // norm elem-wise affine gamma
    void const* weight_buffer_pre_residual_norm; // for gemma norm before residual
    float eps;
    // new residual
    void* intermediate_buffer;
    void* lamport_peer_comm_buffer_ptrs[MAX_RANKS_PER_NODE * 3];
};

struct AllReduceParams
{
    size_t elts_total;
    size_t elts_per_rank;
    size_t elts_per_block;
    size_t rank_offset;
    size_t ranks_per_node;
    size_t local_rank;
    uint32_t barrier_flag;
    uint32_t* peer_barrier_ptrs_in[MAX_RANKS_PER_NODE];
    uint32_t* peer_barrier_ptrs_out[MAX_RANKS_PER_NODE];
    void* peer_comm_buffer_ptrs[MAX_RANKS_PER_NODE];
    void* local_output_buffer_ptr;
    void const* local_input_buffer_ptr;

    AllReduceFusionParams fusion_params;

    static AllReduceParams deserialize(int64_t* buffer, size_t tpSize, size_t tpRank, nvinfer1::DataType dataType,
        int token_num, int hidden_size, AllReduceFusionOp op);
};

bool configurationSupported(AllReduceStrategyType algo, size_t msg_size, size_t n_ranks, nvinfer1::DataType type);

void customAllReduce(kernels::AllReduceParams& params, nvinfer1::DataType dataType, AllReduceStrategyType strat,
    AllReduceStrategyConfig config, AllReduceFusionOp fusionOp, cudaStream_t stream);

void residualRmsNorm(
    kernels::AllReduceParams& params, nvinfer1::DataType dataType, cudaStream_t stream, AllReduceFusionOp fusionOp);

void lamportInitialize(void* buffer, size_t size, nvinfer1::DataType dataType, cudaStream_t stream);

} // namespace tensorrt_llm::kernels
