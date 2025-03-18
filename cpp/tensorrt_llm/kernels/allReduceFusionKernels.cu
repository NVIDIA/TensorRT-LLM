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
#include "tensorrt_llm/common/envUtils.h"
#include "tensorrt_llm/common/reduceKernelUtils.cuh"
#include "tensorrt_llm/kernels/allReduceFusionKernels.h"
#include "tensorrt_llm/kernels/quantization.cuh"
#include <cooperative_groups.h>

namespace tensorrt_llm::kernels::ar_fusion
{
template <int NRanks>
struct SyncComm
{
    __device__ __forceinline__ SyncComm(void** workspace)
    {
        counter_ptr = &reinterpret_cast<int*>(workspace[NRanks * 3])[0];
        flag_ptr = &reinterpret_cast<int*>(workspace[NRanks * 3])[1];
        flag_value = *flag_ptr;
        for (int r = 0; r < NRanks; ++r)
        {
            comm_bufs[r] = workspace[r];
            barrier_flags[r] = workspace[NRanks + r];
        }
        __syncthreads();
        if (threadIdx.x == 0)
        {
            atomicAdd(counter_ptr, 1);
        }
    }

    __device__ __forceinline__ void update(int new_flag_value)
    {
        if (blockIdx.x == 0 && threadIdx.x == 0)
        {
            while (*reinterpret_cast<int volatile*>(counter_ptr) != gridDim.x)
            {
            }
            *flag_ptr = new_flag_value;
            *counter_ptr = 0;
        }
    }

    int* counter_ptr;
    int* flag_ptr;
    void* comm_bufs[NRanks];
    void* barrier_flags[NRanks];
    int flag_value;
};

template <int NRanks>
struct LamportComm
{
    __device__ __forceinline__ LamportComm(void** workspace, int rank)
    {
        counter_ptr = &reinterpret_cast<int*>(workspace[NRanks * 3])[0];
        flag_ptr = &reinterpret_cast<int*>(workspace[NRanks * 3])[2];
        clear_ptr = &reinterpret_cast<int*>(workspace[NRanks * 3])[4];
        flag_value = *flag_ptr;
        int comm_size = reinterpret_cast<int*>(workspace[NRanks * 3])[3];
        clear_size = *clear_ptr;
        int data_offset = flag_value % 3;
        int clear_offset = (flag_value + 2) % 3;
        for (int r = 0; r < NRanks; ++r)
        {
            data_bufs[r] = reinterpret_cast<uint8_t*>(workspace[2 * NRanks + r]) + data_offset * comm_size;
        }
        clear_buf = reinterpret_cast<uint8_t*>(workspace[2 * NRanks + rank]) + clear_offset * comm_size;
        __syncthreads();
        if (threadIdx.x == 0)
        {
            atomicAdd(counter_ptr, 1);
        }
    }

    __device__ __forceinline__ void update(int new_clear_size)
    {
        if (blockIdx.x == 0 && threadIdx.x == 0)
        {
            while (*reinterpret_cast<int volatile*>(counter_ptr) != gridDim.x)
            {
            }
            *flag_ptr = (flag_value + 1) % 3;
            *clear_ptr = new_clear_size;
            *counter_ptr = 0;
        }
    }

    int* counter_ptr;
    int* flag_ptr;
    int* clear_ptr;
    uint8_t* data_bufs[NRanks];
    uint8_t* clear_buf;
    int clear_size;
    int flag_value;
};

template <int NRanks>
class Barrier
{
public:
    __device__ __forceinline__ Barrier(int rank, SyncComm<NRanks> const& comm)
    {
        if (threadIdx.x < NRanks)
        {
            m_flag_value = comm.flag_value;
            int current_rank = rank;
            int target_rank = threadIdx.x;
            m_target_flag
                = reinterpret_cast<int*>(comm.barrier_flags[target_rank]) + blockIdx.x * NRanks + current_rank;
            m_current_flag
                = reinterpret_cast<int*>(comm.barrier_flags[current_rank]) + blockIdx.x * NRanks + target_rank;
        }
    }

    __device__ __forceinline__ void sync()
    {
        __syncthreads();
        if (threadIdx.x < NRanks)
        {
            m_flag_value = next_flag(m_flag_value);
            st_flag(m_target_flag, m_flag_value);
            while (ld_flag(m_current_flag) == prev_flag(m_flag_value))
            {
            }
        }
        __syncthreads();
    }

protected:
    __device__ __forceinline__ void st_flag(int* addr, int flag)
    {
        asm volatile("st.global.release.sys.b32 [%1], %0;" ::"r"(flag), "l"(addr));
    }

    __device__ __forceinline__ int ld_flag(int* addr)
    {
        int flag;
        asm volatile("ld.global.acquire.sys.b32 %0, [%1];" : "=r"(flag) : "l"(addr));
        return flag;
    }

    __device__ __forceinline__ int next_flag(int flag)
    {
        return flag == 2 ? 0 : flag + 1;
    }

    __device__ __forceinline__ int prev_flag(int flag)
    {
        return flag == 0 ? 2 : flag - 1;
    }

public:
    int m_flag_value;

private:
    int* m_target_flag;
    int* m_current_flag;
};

template <typename DType, typename PackedType>
__device__ __forceinline__ PackedType add128(PackedType const& a, PackedType const& b)
{
    static constexpr int kMathCount = sizeof(PackedType) / sizeof(DType);
    PackedType c;
#pragma unroll
    for (int i = 0; i < kMathCount; ++i)
    {
        reinterpret_cast<DType*>(&c)[i] = reinterpret_cast<DType const*>(&a)[i] + reinterpret_cast<DType const*>(&b)[i];
    }
    return c;
}

template <typename DType, typename PackedType>
__device__ __forceinline__ PackedType rms_norm(
    PackedType const& residual, PackedType const& gamma, float const eps, int hidden_dim)
{
    static constexpr int kMathCount = sizeof(PackedType) / sizeof(DType);
    __shared__ float s_val;
    PackedType norm_out;
    cg::cluster_group cluster = cg::this_cluster();
    float acc = 0.f;
#pragma unroll
    for (int i = 0; i < kMathCount; ++i)
    {
        float v = static_cast<float>(reinterpret_cast<DType const*>(&residual)[i]);
        acc += v * v;
    }
    tensorrt_llm::common::blockReduceSumV2<float, 1>(&acc);
    if (cluster.num_blocks() > 1)
    {
        if (threadIdx.x == 0)
        {
            s_val = acc;
            acc = 0.f;
        }
        cluster.sync();
        if (threadIdx.x == 0)
        {
            for (int i = 0; i < cluster.num_blocks(); ++i)
            {
                acc += *cluster.map_shared_rank(&s_val, i);
            }
        }
        cluster.sync();
    }
    if (threadIdx.x == 0)
    {
        s_val = rsqrtf(acc / hidden_dim + eps);
    }
    __syncthreads();
#pragma unroll
    for (int i = 0; i < kMathCount; ++i)
    {
        reinterpret_cast<DType*>(&norm_out)[i]
            = static_cast<DType>(static_cast<float>(reinterpret_cast<DType const*>(&residual)[i]) * s_val
                * static_cast<float>(reinterpret_cast<DType const*>(&gamma)[i]));
    }
    return norm_out;
}

template <bool ResidualOut, bool NormOut, bool QuantOut, typename DType, typename PackedType>
__device__ __forceinline__ void fused_op(
    PackedType const& val, int access_id, int token_id, int access_id_in_token, AllReduceFusionParams& params)
{
    float4 residual_val = reinterpret_cast<float4*>(params.residual_in)[access_id];
    float4 gamma_val = reinterpret_cast<float4*>(params.rms_gamma)[access_id_in_token];
    residual_val = add128<DType>(val, residual_val);
    if constexpr (ResidualOut)
    {
        reinterpret_cast<float4*>(params.residual_out)[access_id] = residual_val;
    }
    float4 norm_val = rms_norm<DType>(residual_val, gamma_val, params.rms_eps, params.hidden_dim);
    if constexpr (NormOut)
    {
        reinterpret_cast<float4*>(params.norm_out)[access_id] = norm_val;
    }
    if constexpr (QuantOut)
    {
        PackedVec<DType> pack_val = *reinterpret_cast<PackedVec<DType> const*>(&norm_val);
        auto sf_out
            = cvt_quant_to_fp4_get_sf_out_offset<uint32_t, 2>(std::nullopt /* batchIdx */, token_id, access_id_in_token,
                std::nullopt /* numRows */, params.hidden_dim, reinterpret_cast<uint32_t*>(params.scale_out));
        reinterpret_cast<uint32_t*>(params.quant_out)[access_id]
            = cvt_warp_fp16_to_fp4(pack_val, *params.scale_factor, sf_out);
    }
}

__device__ __forceinline__ bool is_neg_zero(float v)
{
    return *reinterpret_cast<uint32_t*>(&v) == 0x80000000;
}

__device__ __forceinline__ bool is_neg_zero(float4 v)
{
    return is_neg_zero(v.x) || is_neg_zero(v.y) || is_neg_zero(v.z) || is_neg_zero(v.w);
}

__device__ __forceinline__ float4 get_neg_zero()
{
    float4 vec;
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        reinterpret_cast<uint32_t*>(&vec)[i] = 0x80000000;
    }
    return vec;
}

__device__ __forceinline__ float4 ld_global_volatile(float4* addr)
{
    float4 val;
    asm volatile("ld.volatile.global.v4.f32 {%0, %1, %2, %3}, [%4];"
                 : "=f"(val.x), "=f"(val.y), "=f"(val.z), "=f"(val.w)
                 : "l"(addr));
    return val;
}

template <typename DType, int NRanks, bool ResidualOut, bool NormOut, bool QuantOut>
__global__ void allreduce_fusion_kernel_oneshot_lamport(AllReduceFusionParams params)
{
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    namespace cg = cooperative_groups;
    cg::cluster_group cluster = cg::this_cluster();
    cg::grid_group grid = cg::this_grid();
    int token_id = grid.cluster_rank();
    int access_id_in_token = cluster.thread_rank();
    int access_id = token_id * params.hidden_dim / kElemsPerAccess + access_id_in_token;
    int token_stride = grid.num_clusters();
    int access_stride = token_stride * params.hidden_dim / kElemsPerAccess;
    int tot_access = params.size / kElemsPerAccess;
    float4 clear_vec = get_neg_zero();
    cudaGridDependencySynchronize();
    LamportComm<NRanks> comm(params.workspace, params.rank);
    int clear_access = comm.clear_size / kElemsPerAccess;

    // Persistent Kernel
    // Each cluster iterate through all token it need to handle
    for (int idx = access_id; idx < tot_access; idx += access_stride)
    {
        // LDG.128
        float val[4];
        *reinterpret_cast<float4*>(val) = reinterpret_cast<float4*>(params.allreduce_in)[idx];
#pragma unroll
        for (int i = 0; i < 4; ++i)
        {
            if (is_neg_zero(val[i]))
            {
                val[i] = 0.f;
            }
        }
#pragma unroll
        for (int r = 0; r < NRanks; ++r)
        {
            // STG.128 to remote rank
            reinterpret_cast<float4*>(comm.data_bufs[r])[params.rank * tot_access + idx]
                = *reinterpret_cast<float4*>(val);
        }

        if (idx < clear_access)
        {
            // STG.128
            reinterpret_cast<float4*>(comm.clear_buf)[idx] = clear_vec;
        }
    }

    // Persistent Kernel
    // Each cluster iterate through all token it need to handle
    for (int idx = access_id, tidx = token_id; idx < tot_access; idx += access_stride, tidx += token_stride)
    {
        float4 vals[NRanks];
        bool done = false;
        while (!done)
        {
            done = true;
#pragma unroll
            for (int r = 0; r < NRanks; ++r)
            {
                // LDG.128 from local rank
                vals[r]
                    = ld_global_volatile(&reinterpret_cast<float4*>(comm.data_bufs[params.rank])[r * tot_access + idx]);
                done &= !is_neg_zero(vals[r]);
            }
        }
        float4 sum_val = vals[0];
#pragma unroll
        for (int r = 1; r < NRanks; ++r)
        {
            // FFMA
            sum_val = add128<DType>(sum_val, vals[r]);
        }

        // Fused Norm
        fused_op<ResidualOut, NormOut, QuantOut, DType>(sum_val, idx, tidx, access_id_in_token, params);
    }
    comm.update(params.size * NRanks);
    cudaTriggerProgrammaticLaunchCompletion();
#endif
}

template <typename DType, int NRanks, bool ResidualOut, bool NormOut, bool QuantOut>
__global__ void allreduce_fusion_kernel_oneshot_sync(AllReduceFusionParams params)
{
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    namespace cg = cooperative_groups;
    cg::cluster_group cluster = cg::this_cluster();

    int token_id = blockIdx.x / cluster.num_blocks();
    int access_id_in_token = cluster.block_rank() * blockDim.x + threadIdx.x;
    int access_id = token_id * params.hidden_dim / kElemsPerAccess + access_id_in_token;
    int token_stride = gridDim.x / cluster.num_blocks();
    int access_stride = token_stride * params.hidden_dim / kElemsPerAccess;
    int tot_access = params.size / kElemsPerAccess;
    cudaGridDependencySynchronize();
    SyncComm<NRanks> comm(params.workspace);
    for (int idx = access_id; idx < tot_access; idx += access_stride)
    {
        float4 val;
        val = reinterpret_cast<float4*>(params.allreduce_in)[idx];
#pragma unroll
        for (int r = 0; r < NRanks; ++r)
        {
            reinterpret_cast<float4*>(comm.comm_bufs[r])[params.rank * tot_access + idx] = val;
        }
    }
    Barrier<NRanks> barrier(params.rank, comm);
    barrier.sync();
    for (int idx = access_id, tidx = token_id; idx < tot_access; idx += access_stride, tidx += token_stride)
    {
        float4 vals[NRanks];
#pragma unroll
        for (int r = 0; r < NRanks; ++r)
        {
            vals[r] = reinterpret_cast<float4*>(comm.comm_bufs[params.rank])[r * tot_access + idx];
        }
        float4 sum_val = vals[0];
#pragma unroll
        for (int r = 1; r < NRanks; ++r)
        {
            sum_val = add128<DType>(sum_val, vals[r]);
        }
        fused_op<ResidualOut, NormOut, QuantOut, DType>(sum_val, idx, tidx, access_id_in_token, params);
    }
    comm.update(barrier.m_flag_value);
    cudaTriggerProgrammaticLaunchCompletion();
#endif
}

template <typename DType, int NRanks, bool ResidualOut, bool NormOut, bool QuantOut>
__global__ void allreduce_fusion_kernel_twoshot_sync(AllReduceFusionParams params, int begin_token, int token_num)
{
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    namespace cg = cooperative_groups;
    cg::cluster_group cluster = cg::this_cluster();

    int token_id = blockIdx.x / cluster.num_blocks();
    int access_id_in_token = cluster.block_rank() * blockDim.x + threadIdx.x;
    int access_id = token_id * params.hidden_dim / kElemsPerAccess + access_id_in_token;
    int token_stride = gridDim.x / cluster.num_blocks();
    int access_stride = token_stride * params.hidden_dim / kElemsPerAccess;
    int tot_access = params.size / kElemsPerAccess;
    cudaGridDependencySynchronize();
    SyncComm<NRanks> comm(params.workspace);
    for (int idx = access_id; idx < tot_access; idx += access_stride)
    {
        float4 val;
        val = reinterpret_cast<float4*>(params.allreduce_in)[idx];
        reinterpret_cast<float4*>(comm.comm_bufs[params.rank])[idx] = val;
    }
    Barrier<NRanks> barrier(params.rank, comm);
    barrier.sync();
    int comm_access_id = access_id + begin_token * params.hidden_dim / kElemsPerAccess;
    int comm_tot_access = (begin_token + token_num) * params.hidden_dim / kElemsPerAccess;
    for (int idx = comm_access_id; idx < comm_tot_access; idx += access_stride)
    {
        float4 vals[NRanks];
#pragma unroll
        for (int r = 0; r < NRanks; ++r)
        {
            vals[r] = reinterpret_cast<float4*>(comm.comm_bufs[r])[idx];
        }
        float4 sum_val = vals[0];
#pragma unroll
        for (int r = 1; r < NRanks; ++r)
        {
            sum_val = add128<DType>(sum_val, vals[r]);
        }
#pragma unroll
        for (int r = 0; r < NRanks; ++r)
        {
            reinterpret_cast<float4*>(comm.comm_bufs[r])[tot_access + idx] = sum_val;
        }
    }
    barrier.sync();
    for (int idx = access_id, tidx = token_id; idx < tot_access; idx += access_stride, tidx += token_stride)
    {
        float4 sum_val = reinterpret_cast<float4*>(comm.comm_bufs[params.rank])[tot_access + idx];
        fused_op<ResidualOut, NormOut, QuantOut, DType>(sum_val, idx, tidx, access_id_in_token, params);
    }
    comm.update(barrier.m_flag_value);
    cudaTriggerProgrammaticLaunchCompletion();
#endif
}

int get_sm_count()
{
    static int sm_count = 0;
    if (sm_count == 0)
    {
        int device_id;
        TLLM_CUDA_CHECK(cudaGetDevice(&device_id));
        cudaDeviceProp device_prop;
        cudaGetDeviceProperties(&device_prop, device_id);
        sm_count = device_prop.multiProcessorCount;
    }
    return sm_count;
}

template <typename DType, int NRanks, bool ResidualOut, bool NormOut, bool QuantOut>
void launch_oneshot_lamport(AllReduceFusionParams const& params, cudaLaunchConfig_t& cfg)
{
    TLLM_CUDA_CHECK(cudaLaunchKernelEx(
        &cfg, allreduce_fusion_kernel_oneshot_lamport<DType, NRanks, ResidualOut, NormOut, QuantOut>, params));
}

template <typename DType, int NRanks, bool ResidualOut, bool NormOut, bool QuantOut>
void launch_oneshot_sync(AllReduceFusionParams const& params, cudaLaunchConfig_t& cfg)
{
    TLLM_CUDA_CHECK(cudaLaunchKernelEx(
        &cfg, allreduce_fusion_kernel_oneshot_sync<DType, NRanks, ResidualOut, NormOut, QuantOut>, params));
}

template <typename DType, int NRanks, bool ResidualOut, bool NormOut, bool QuantOut>
void launch_twoshot_sync(AllReduceFusionParams const& params, cudaLaunchConfig_t& cfg, int begin_token, int token_num)
{
    TLLM_CUDA_CHECK(
        cudaLaunchKernelEx(&cfg, allreduce_fusion_kernel_twoshot_sync<DType, NRanks, ResidualOut, NormOut, QuantOut>,
            params, begin_token, token_num));
}

bool use_oneshot(int token_num)
{
    return token_num <= kOneShotMaxToken;
}

template <typename DType, int NRanks, bool ResidualOut, bool NormOut, bool QuantOut>
void allreduce_fusion_kernel_launcher(AllReduceFusionParams const& params)
{
    int token_num = params.size / params.hidden_dim;
    bool oneshot = use_oneshot(token_num);
    int cluster_num = token_num;
    int begin_token = 0;
    if (!oneshot)
    {
        int remaining_token = token_num % NRanks;
        token_num = token_num / NRanks;
        cluster_num = token_num;
        if (remaining_token)
        {
            cluster_num++;
        }
        begin_token = params.rank * token_num;
        begin_token += remaining_token > params.rank ? params.rank : remaining_token;
        if (remaining_token > params.rank)
        {
            ++token_num;
        }
    }
    int threads_per_token = params.hidden_dim / kElemsPerAccess;
    int warps_per_token = (threads_per_token + 31) / 32;
    int cluster_size = 8;
    while (warps_per_token % cluster_size != 0)
    {
        cluster_size /= 2;
    }
    int block_size = warps_per_token / cluster_size * 32;
    TLLM_CHECK(block_size <= 1024 && cluster_size > 0);
    int sm_count = get_sm_count();
    int grid_size = (std::min(sm_count, cluster_num * cluster_size) / cluster_size) * cluster_size;
    cudaLaunchConfig_t cfg;
    cudaLaunchAttribute attribute[2];
    cfg.gridDim = grid_size;
    cfg.blockDim = block_size;
    cfg.dynamicSmemBytes = 0;
    cfg.stream = params.stream;
    attribute[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attribute[0].val.programmaticStreamSerializationAllowed = tensorrt_llm::common::getEnvEnablePDL() ? 1 : 0;
    attribute[1].id = cudaLaunchAttributeClusterDimension;
    attribute[1].val.clusterDim.x = cluster_size;
    attribute[1].val.clusterDim.y = 1;
    attribute[1].val.clusterDim.z = 1;
    cfg.attrs = attribute;
    cfg.numAttrs = 2;
    if (oneshot)
    {
        launch_oneshot_lamport<DType, NRanks, ResidualOut, NormOut, QuantOut>(params, cfg);
        // launch_oneshot_sync<DType, NRanks, ResidualOut, NormOut, QuantOut>(params, cfg);
    }
    else
    {
        launch_twoshot_sync<DType, NRanks, ResidualOut, NormOut, QuantOut>(params, cfg, begin_token, token_num);
    }
}

void allreduce_fusion_op(AllReduceFusionParams const& params)
{
#define DISPATCH1(DType, NRanks, ResidualOut, NormOut, QuantOut)                                                       \
    return allreduce_fusion_kernel_launcher<DType, NRanks, ResidualOut, NormOut, QuantOut>(params);
#define DISPATCH0(NRanks, ResidualOut, NormOut, QuantOut)                                                              \
    if (params.nranks == NRanks && params.dtype == nvinfer1::DataType::kHALF)                                          \
    {                                                                                                                  \
        DISPATCH1(half, NRanks, ResidualOut, NormOut, QuantOut);                                                       \
    }                                                                                                                  \
    else if (params.nranks == NRanks && params.dtype == nvinfer1::DataType::kBF16)                                     \
    {                                                                                                                  \
        DISPATCH1(__nv_bfloat16, NRanks, ResidualOut, NormOut, QuantOut);                                              \
    }

    TLLM_CHECK(params.allreduce_in && params.residual_in && params.rms_gamma);
    TLLM_CHECK(params.size % params.hidden_dim == 0);
    if (params.residual_out && !params.norm_out && params.quant_out)
    {
        // pattern1: AR+Add_RMS+Quant
        // [m, 7168] bf16 allreduce_in, [m, 7168] bf16 residual_in
        // [m, 7168] bf16 residual_out, [m, 7168] fp4 quant_out
        DISPATCH0(2, true, false, true);
        DISPATCH0(4, true, false, true);
        DISPATCH0(8, true, false, true);
        DISPATCH0(16, true, false, true);
    }
    else if (!params.residual_out && params.norm_out && !params.quant_out)
    {
        // pattern2: AR+AddRMS
        // [m, 7168] bf16 allreduce_in, [m, 7168] bf16 residual_in
        // [m, 7168] bf16 norm_out
        DISPATCH0(2, false, true, false);
        DISPATCH0(4, false, true, false);
        DISPATCH0(8, false, true, false);
        DISPATCH0(16, false, true, false);
    }
    else if (params.residual_out && params.norm_out && !params.quant_out)
    {
        DISPATCH0(2, true, true, false);
        DISPATCH0(4, true, true, false);
        DISPATCH0(8, true, true, false);
        DISPATCH0(16, true, true, false);
    }
    else if (params.residual_out && params.norm_out && params.quant_out)
    {
        // for test
        DISPATCH0(2, true, true, true);
        DISPATCH0(4, true, true, true);
        DISPATCH0(8, true, true, true);
        DISPATCH0(16, true, true, true);
    }
    TLLM_CHECK_WITH_INFO(false, "allreduce_fusion_kernel: unsupported pattern!");
}

__global__ void lamport_initialize_kernel(float* ptr, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size)
        return;
    ptr[idx] = -0.f;
}

void lamport_initialize(void* ptr, int bytes, cudaStream_t stream)
{
    lamport_initialize_kernel<<<bytes / 128, 128, 0, stream>>>(reinterpret_cast<float*>(ptr), bytes / sizeof(float));
}

Workspace::Workspace(int rank, int tp_size, int max_token_num, int hidden_dim,
    std::shared_ptr<tensorrt_llm::runtime::CudaStream> stream_ptr)
    : m_world_config(tp_size, 1, 1, rank, tp_size)
    , m_cuda_stream(stream_ptr)
{
    bool p2p_supported = tensorrt_llm::runtime::canAccessPeer(m_world_config);
    TLLM_CHECK(p2p_supported);
    int device_id;
    TLLM_CUDA_CHECK(cudaGetDevice(&device_id));
    m_buffer_mgr = std::make_shared<tensorrt_llm::runtime::BufferManager>(m_cuda_stream);
    int buffer_size = tp_size * max_token_num * hidden_dim * sizeof(half);
    int flag_size = tp_size * kBarrierFlagCount * sizeof(int);
    int lamport_comm_size = tp_size * std::max(kOneShotMaxToken, max_token_num) * hidden_dim * sizeof(half);
    int lamport_buffer_size = 3 * lamport_comm_size;
    for (auto size : {buffer_size, flag_size, lamport_buffer_size})
    {
        m_ipc_mem_handles.emplace_back(size, *m_buffer_mgr, m_world_config, p2p_supported);
    }
    std::vector<void*> workspace;
    for (auto& ipc_mem_handle : m_ipc_mem_handles)
    {
        for (int r = 0; r < tp_size; ++r)
        {
            workspace.push_back(ipc_mem_handle.getCommPtrs()[r]);
        }
    }
    // atomic flag read counter
    // kernel_flag_ptr[0] = 0;
    // non-lamport flag
    // kernel_flag_ptr[1] = 0;
    // lamport flag
    // kernel_flag_ptr[2] = 0;
    // lamport triple buffer offset
    // kernel_flag_ptr[3] = lamport_comm_size;
    // lamport clear size
    // kernel_flag_ptr[4] = 0;
    TLLM_CUDA_CHECK(cudaMalloc(&m_flag_d_ptr, 5 * sizeof(int)));
    std::vector<int> h_data{0, 0, 0, lamport_comm_size, 0};
    TLLM_CUDA_CHECK(cudaMemcpy(m_flag_d_ptr, h_data.data(), 5 * sizeof(int), cudaMemcpyHostToDevice));
    workspace.push_back(m_flag_d_ptr);
    TLLM_CUDA_CHECK(cudaMalloc(&m_workspace, workspace.size() * sizeof(void*)));
    TLLM_CUDA_CHECK(
        cudaMemcpy(m_workspace, workspace.data(), workspace.size() * sizeof(void*), cudaMemcpyHostToDevice));
    lamport_initialize(m_ipc_mem_handles[2].getCommPtrs()[rank], lamport_buffer_size, 0);
}

Workspace::~Workspace()
{
    if (m_flag_d_ptr)
    {
        TLLM_CUDA_CHECK(cudaFree(m_flag_d_ptr));
    }
    if (m_workspace)
    {
        TLLM_CUDA_CHECK(cudaFree(m_workspace));
    }
}

void** Workspace::get_workspace()
{
    return reinterpret_cast<void**>(m_workspace);
}
}; // namespace tensorrt_llm::kernels::ar_fusion
