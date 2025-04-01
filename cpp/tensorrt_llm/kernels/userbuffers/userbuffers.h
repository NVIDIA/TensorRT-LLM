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
#include "tensorrt_llm/runtime/utils/mpiUtils.h"
#include "tensorrt_llm/runtime/worldConfig.h"

#include <cuda.h>
#if defined(__aarch64__) || defined(_M_ARM64)
#define MNNVL
#endif

#define MAX_REGIONS 16
#define MAX_SMS 32
#define MAX_OPS 32
#define MAX_PEERS 8192
#define MAX_REQUESTS 1024
#define LAUNCH_GPU 1
#define LAUNCH_CPU 2
#define MAX_NVLINK 32

#define UB_MEM_UC_CONTIG 1
#define UB_MEM_MC_CREATED 2
#define UB_MEM_ALLOCATED 4

// region 0 flag offsets
#define REG0_OPFLAGS (MAX_PEERS * 2)
#define REG0_RECV (REG0_OPFLAGS * userbuffers_op_types)
#define REG0_SINGLENODE (2 * MAX_NVLINK * MAX_SMS + MAX_OPS)
#define REG0_OFFSET(comm) ((2 * MAX_REGIONS) * MAX_NVLINK + REG0_SINGLENODE * 2 + MAX_PEERS)
#define REG0_ONESHOT_MAX 32 * 1024
#define REG0_ONESHOT_BUFFER (MAX_NVLINK * REG0_ONESHOT_MAX)
#define REG0_COMMBUFFER (REG0_ONESHOT_BUFFER * 2)
#define REG0_FLAGS (REG0_RECV + MAX_PEERS * MAX_REGIONS * 3)

namespace tensorrt_llm::runtime::ub
{
enum req_type
{
    userbuffers_allreduceop_sharp,
    userbuffers_sendop,
    userbuffers_allreduceop_nonsharp,
    userbuffers_allreduceop_nonsharp2,
    userbuffers_alltoall,
    userbuffers_op_types
};

struct communicator
{
    int myrank, nranks; // global job communicator
    int nvrank, nvsize; // single node comm_intra
    int free_region;

    void* gpu_ptrs;
    int sms, threads;
    int use_rr_kernel; // Whether to use RR (or RW) for NVLink-only kernel
    int cga_size;

    void* mem_ptr[MAX_REGIONS];
    void** peer_ptr[MAX_REGIONS];

    int memflags[MAX_REGIONS]; // UC,MC, user/lib allocated

    CUmemGenericAllocationHandle* uchandles[MAX_REGIONS];
    void* ucbase_ptr[MAX_REGIONS]; // only for cuMem allocated memory
    size_t mem_size[MAX_REGIONS];

    void* mc_ptr[MAX_REGIONS];
    void* mc_baseptr;
    CUmemGenericAllocationHandle mc_handle;
    size_t mc_offset, mc_maxsize;
    int use_mc;                          // 1: use MC if available, 0: override not to use MC

    int tp_size, tp_first_rank, tp_rank; // with ar_nvsize as a step
    int sm_arch;
    int oneshot, pdl_launch;
    int oneshot_force_enable_threshold;

    MPI_Comm comm_world, // clone of MPI_COMM_WORLD
        comm_inter,      // reduction group communicator (subset of the nodes) along GPU rail
        comm_intra;      // full intranode (all ndev GPUS)

    int *send_id, *recv_id;
    int mydev;
};
using communicator = struct communicator;

int create_communicator_grouped2(communicator** comm, tensorrt_llm::runtime::WorldConfig const& world_config);
/*  creates communicator with
    allreduce1 to happen in datagpus x datanodes groups,
    allreduce2 to happen in tensorgpus x tensor nodes,
        where num_nodes = pipenodes x tensornodes x datanodes
            nvlink_size = pipegpus x tensorgpus x datagpus
 */

int register_user_buffer_collective(void** gpubuff, size_t bytes, communicator* comm);
/*  returns handler and registers buffers. assumed to be collective i.e. you use same groups and dont mix buffers for
   different operations returns -1 if can't register (too many preregistered regions already) if alloc==true will
   allocate memory and fill the pointers (required for NVL SHARP and NSO/MNNVL)
*/

void destroy_communicator(communicator* comm);
} // namespace tensorrt_llm::runtime::ub

namespace tensorrt_llm::kernels::ub
{
using namespace tensorrt_llm::runtime::ub;
void allreduce2_userbuff_inplace_impl(int const handler, size_t const offset, size_t const elements,
    nvinfer1::DataType dataType, communicator* comm, cudaStream_t stream = 0);
// for TP-parallelism, only single node is implemented

int allgather2_userbuff_residual_impl(int const handler, size_t const offset, size_t const elements,
    int const hidden_size, void* residual, nvinfer1::DataType dataType, communicator* comm, cudaStream_t stream,
    bool force_enable);

int allreduce2_userbuff_rmsnorm_impl(int const handler, size_t const offset, int const out_handler,
    size_t const out_offset, size_t const elements, int const hidden_size, void* beta, void* gamma, float eps,
    void* residual_in, void* residual_out, nvinfer1::DataType dataType, communicator* comm, cudaStream_t stream);

int allreduce2_userbuff_inplace_rmsnorm_quant_impl(int const handler, size_t const offset, int const out_handler,
    size_t const out_offset, size_t const elements, int const hidden_size, void* beta, void* gamma, float eps,
    float* scalefactor, void* residual_in, void* residual_out, nvinfer1::DataType dataType, communicator* comm,
    cudaStream_t stream);
int allreduce2_userbuff_inplace_rmsnorm_quant_fp4_impl(int const handler, size_t const offset, int const out_handler,
    size_t const out_offset, int const scale_handler, size_t const scale_offset, size_t const elements,
    int const hidden_size, void* beta, void* gamma, float eps, float* scalefactor, void* residual_in,
    void* residual_out, nvinfer1::DataType dataType, communicator* comm, cudaStream_t stream);
} // namespace tensorrt_llm::kernels::ub
