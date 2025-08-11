/*
 * Copyright (c) 2022-2025, NVIDIA CORPORATION.  All rights reserved.
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
#include "tensorrt_llm/kernels/communicationKernels/allReduceFusionKernels.h"
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
            m_target_flag = reinterpret_cast<int*>(comm.barrier_flags[target_rank]) + current_rank;
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
            // To avoid the ABA problem, we need to synchronize the correct flag value to all barrier_flags, even if the
            // corresponding CTA has not been launched.
            for (int flag_idx = blockIdx.x; flag_idx < kBarrierFlagCount; flag_idx += gridDim.x)
            {
                st_flag(m_target_flag + flag_idx * NRanks, m_flag_value);
            }
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

template <AllReduceFusionPattern Pattern, typename DType>
class FusedOp
{
    static constexpr int kMathCount = sizeof(float4) / sizeof(DType);

public:
    __device__ __forceinline__ FusedOp(AllReduceFusionParams const& params, int access_id, int access_id_in_token)
        : m_params(params)
        , m_access_id(access_id)
        , m_access_id_in_token(access_id_in_token)
    {
        if constexpr (HasRMSNorm<Pattern>)
        {
            m_gamma_val = reinterpret_cast<float4*>(params.rms_gamma)[m_access_id_in_token];
        }
        if constexpr (HasResidual<Pattern>)
        {
            m_residual_val = reinterpret_cast<float4*>(params.residual_in)[m_access_id];
        }
        if constexpr (GetQuantType<Pattern> == QuantType::kFP8)
        {
            m_scale_factor = 1.f / *params.scale_factor;
        }
        else if constexpr (GetQuantType<Pattern> == QuantType::kFP4)
        {
            m_scale_factor = *params.scale_factor;
        }
    }

    __device__ __forceinline__ void update(int access_id)
    {

        if (m_access_id != access_id)
        {
            m_access_id = access_id;
            if constexpr (HasResidual<Pattern>)
            {
                m_residual_val = reinterpret_cast<float4*>(m_params.residual_in)[m_access_id];
            }
        }
    }

    __device__ __forceinline__ void operator()(float4 val, int token_id)
    {
        if constexpr (HasAllReduceOut<Pattern>)
        {
            reinterpret_cast<float4*>(m_params.allreduce_out)[m_access_id] = val;
        }
        if constexpr (HasResidual<Pattern>)
        {
            val = add128<DType>(val, m_residual_val);
            if constexpr (HasResidualOut<Pattern>)
            {
                reinterpret_cast<float4*>(m_params.residual_out)[m_access_id] = val;
            }
        }
        if constexpr (HasRMSNorm<Pattern>)
        {
            val = rms_norm(val, m_gamma_val);
            if constexpr (HasNormOut<Pattern>)
            {
                reinterpret_cast<float4*>(m_params.norm_out)[m_access_id] = val;
            }
        }
        if constexpr (GetQuantType<Pattern> == QuantType::kFP4)
        {
            constexpr int SF_VEC_SIZE = 16;
            using PackedVec = PackedVec<DType>;
            PackedVec pack_val = *reinterpret_cast<PackedVec const*>(&val);
            auto sf_out = cvt_quant_get_sf_out_offset<uint32_t, 2>(std::nullopt, token_id, m_access_id_in_token,
                std::nullopt, m_params.hidden_dim / SF_VEC_SIZE, reinterpret_cast<uint32_t*>(m_params.scale_out),
                m_params.layout);
            reinterpret_cast<uint32_t*>(m_params.quant_out)[m_access_id]
                = cvt_warp_fp16_to_fp4<DType, SF_VEC_SIZE, false>(pack_val, m_scale_factor, sf_out);
        }
        else if constexpr (GetQuantType<Pattern> == QuantType::kFP8)
        {
            using PackedQuantizedType = std::conditional_t<std::is_same_v<DType, float>, float, float2>;
            PackedQuantizedType ret;
#pragma unroll
            for (int i = 0; i < kMathCount; ++i)
            {
                reinterpret_cast<__nv_fp8_e4m3*>(&ret)[i] = static_cast<__nv_fp8_e4m3>(
                    static_cast<float>(reinterpret_cast<DType*>(&val)[i]) * m_scale_factor);
            }
            reinterpret_cast<PackedQuantizedType*>(m_params.quant_out)[m_access_id] = ret;
        }
        else
        {
            static_assert(GetQuantType<Pattern> == QuantType::kNone, "Invalid quant type");
        }
    }

protected:
    __device__ __forceinline__ float4 rms_norm(float4 const& residual, float4 const& gamma)
    {
        __shared__ float s_val;
        float4 norm_out;
        float acc = 0.f;
#pragma unroll
        for (int i = 0; i < kMathCount; ++i)
        {
            float v = static_cast<float>(reinterpret_cast<DType const*>(&residual)[i]);
            acc += v * v;
        }
        tensorrt_llm::common::blockReduceSumV2<float, 1>(&acc);
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
        cg::cluster_group cluster = cg::this_cluster();
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
#endif
        if (threadIdx.x == 0)
        {
            s_val = rsqrtf(acc / m_params.hidden_dim + m_params.rms_eps);
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

private:
    AllReduceFusionParams const& m_params;
    int m_access_id;
    int m_access_id_in_token;
    float m_scale_factor;
    float4 m_residual_val;
    float4 m_gamma_val;
};

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

template <typename DType, int NRanks, bool Fp32Acc>
__device__ __forceinline__ float4 allreduce_sum(float4* vals)
{
    if constexpr (Fp32Acc)
    {
        static_assert(!std::is_same_v<DType, float>);
        float acc_f32[kElemsPerAccess<DType>];
#pragma unroll
        for (int i = 0; i < kElemsPerAccess<DType>; ++i)
        {
            acc_f32[i] = static_cast<float>(reinterpret_cast<DType*>(&vals[0])[i]);
        }
#pragma unroll
        for (int r = 1; r < NRanks; ++r)
        {
#pragma unroll
            for (int i = 0; i < kElemsPerAccess<DType>; ++i)
            {
                acc_f32[i] += static_cast<float>(reinterpret_cast<DType*>(&vals[r])[i]);
            }
        }
        float4 acc;
#pragma unroll
        for (int i = 0; i < kElemsPerAccess<DType>; ++i)
        {
            reinterpret_cast<DType*>(&acc)[i] = static_cast<DType>(acc_f32[i]);
        }
        return acc;
    }
    else
    {
        float4 acc = vals[0];
#pragma unroll
        for (int r = 1; r < NRanks; ++r)
        {
            acc = add128<DType>(acc, vals[r]);
        }
        return acc;
    }
}

template <typename DType>
class IndexHelper
{
public:
    __device__ __forceinline__ IndexHelper(AllReduceFusionParams const& params)
    {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
        namespace cg = cooperative_groups;
        cg::cluster_group cluster = cg::this_cluster();
        cg::grid_group grid = cg::this_grid();
        token_id = grid.cluster_rank();
        access_id_in_token = cluster.thread_rank();
        token_stride = grid.num_clusters();
#else
        token_id = blockIdx.x;
        access_id_in_token = threadIdx.x;
        token_stride = gridDim.x;
#endif
        access_id = token_id * params.hidden_dim / kElemsPerAccess<DType> + access_id_in_token;
        access_stride = token_stride * params.hidden_dim / kElemsPerAccess<DType>;
        tot_access = params.size / kElemsPerAccess<DType>;
    }

    int token_id;
    int access_id_in_token;
    int token_stride;
    int access_id;
    int access_stride;
    int tot_access;
};

template <AllReduceFusionPattern Pattern, typename DType, int NRanks, bool Fp32Acc, bool TriggerCompletionAtEnd = true>
__global__ void __launch_bounds__(1024) allreduce_fusion_kernel_oneshot_lamport(AllReduceFusionParams params)
{
    IndexHelper<DType> index_helper(params);
    int token_id = index_helper.token_id;
    int access_id_in_token = index_helper.access_id_in_token;
    int token_stride = index_helper.token_stride;
    int access_id = index_helper.access_id;
    int access_stride = index_helper.access_stride;
    int tot_access = index_helper.tot_access;
    float4 clear_vec = get_neg_zero();
    FusedOp<Pattern, DType> fused_op(params, access_id, access_id_in_token);

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaGridDependencySynchronize();
    if constexpr (!TriggerCompletionAtEnd)
    {
        cudaTriggerProgrammaticLaunchCompletion();
    }
#endif
    LamportComm<NRanks> comm(params.workspace, params.rank);
    int clear_access = comm.clear_size / kElemsPerAccess<DType>;

    for (int idx = access_id; idx < tot_access; idx += access_stride)
    {
        alignas(16) float val[4];
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
            // Push data to other ranks
            reinterpret_cast<float4*>(comm.data_bufs[r])[params.rank * tot_access + idx]
                = *reinterpret_cast<float4*>(val);
        }
    }
    for (int idx = access_id; idx < clear_access; idx += access_stride)
    {
        // Clear comm buffer that previous kernel used
        reinterpret_cast<float4*>(comm.clear_buf)[idx] = clear_vec;
    }

    for (int idx = access_id, tidx = token_id; idx < tot_access; idx += access_stride, tidx += token_stride)
    {
        fused_op.update(idx);
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
        float4 sum_val = allreduce_sum<DType, NRanks, Fp32Acc>(vals);
        fused_op(sum_val, tidx);
    }

    comm.update(params.size * NRanks);

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    if constexpr (TriggerCompletionAtEnd)
    {
        cudaTriggerProgrammaticLaunchCompletion();
    }
#endif
}

template <AllReduceFusionPattern Pattern, typename DType, int NRanks, bool Fp32Acc>
__global__ void __launch_bounds__(1024) allreduce_fusion_kernel_twoshot_sync(
    AllReduceFusionParams params, std::array<int, NRanks> begin_tokens, std::array<int, NRanks> token_num_per_ranks)
{
    IndexHelper<DType> index_helper(params);
    int token_id = index_helper.token_id;
    int access_id_in_token = index_helper.access_id_in_token;
    int token_stride = index_helper.token_stride;
    int access_id = index_helper.access_id;
    int access_stride = index_helper.access_stride;
    int tot_access = index_helper.tot_access;
    FusedOp<Pattern, DType> fused_op(params, access_id, access_id_in_token);
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaGridDependencySynchronize();
#endif
    SyncComm<NRanks> comm(params.workspace);
#pragma unroll
    for (int r = 0; r < NRanks; ++r)
    {
        int comm_access_id = access_id + begin_tokens[r] * params.hidden_dim / kElemsPerAccess<DType>;
        int comm_tot_access = (begin_tokens[r] + token_num_per_ranks[r]) * params.hidden_dim / kElemsPerAccess<DType>;
        for (int idx = comm_access_id; idx < comm_tot_access; idx += access_stride)
        {
            reinterpret_cast<float4*>(comm.comm_bufs[params.rank])[idx]
                = reinterpret_cast<float4*>(params.allreduce_in)[idx];
        }
    }
    Barrier<NRanks> barrier(params.rank, comm);
    barrier.sync();
    int comm_access_id = access_id + begin_tokens[params.rank] * params.hidden_dim / kElemsPerAccess<DType>;
    int comm_tot_access
        = (begin_tokens[params.rank] + token_num_per_ranks[params.rank]) * params.hidden_dim / kElemsPerAccess<DType>;
    for (int idx = comm_access_id; idx < comm_tot_access; idx += access_stride)
    {
        float4 vals[NRanks];
#pragma unroll
        for (int r = 0; r < NRanks; ++r)
        {
            vals[r] = reinterpret_cast<float4*>(comm.comm_bufs[r])[idx];
        }
        float4 sum_val = allreduce_sum<DType, NRanks, Fp32Acc>(vals);
#pragma unroll
        for (int r = 0; r < NRanks; ++r)
        {
            reinterpret_cast<float4*>(comm.comm_bufs[r])[tot_access + idx] = sum_val;
        }
    }
    barrier.sync();
#pragma unroll
    for (int r = 0; r < NRanks; ++r)
    {
        int comm_access_id = access_id + begin_tokens[r] * params.hidden_dim / kElemsPerAccess<DType>;
        int comm_token_id = token_id + begin_tokens[r];
        int comm_tot_access = (begin_tokens[r] + token_num_per_ranks[r]) * params.hidden_dim / kElemsPerAccess<DType>;
        for (int idx = comm_access_id, tidx = comm_token_id; idx < comm_tot_access;
             idx += access_stride, tidx += token_stride)
        {
            fused_op.update(idx);
            float4 sum_val = reinterpret_cast<float4*>(comm.comm_bufs[params.rank])[tot_access + idx];
            fused_op(sum_val, tidx);
        }
    }
    comm.update(barrier.m_flag_value);
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
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

template <AllReduceFusionPattern Pattern, typename DType, int NRanks, bool Fp32Acc, bool TriggerCompletionAtEnd = true>
void launch_oneshot_lamport(AllReduceFusionParams const& params, cudaLaunchConfig_t& cfg)
{
    TLLM_CUDA_CHECK(cudaLaunchKernelEx(&cfg,
        allreduce_fusion_kernel_oneshot_lamport<Pattern, DType, NRanks, Fp32Acc, TriggerCompletionAtEnd>, params));
}

template <AllReduceFusionPattern Pattern, typename DType, int NRanks, bool Fp32Acc>
void launch_twoshot_sync(AllReduceFusionParams const& params, cudaLaunchConfig_t& cfg,
    std::array<int, NRanks> begin_tokens, std::array<int, NRanks> token_num_per_ranks)
{
    TLLM_CUDA_CHECK(cudaLaunchKernelEx(&cfg, allreduce_fusion_kernel_twoshot_sync<Pattern, DType, NRanks, Fp32Acc>,
        params, begin_tokens, token_num_per_ranks));
}

bool use_oneshot(int token_num)
{
    return token_num <= kOneShotMaxToken;
}

template <AllReduceFusionPattern Pattern, typename DType, int NRanks, bool Fp32Acc>
void allreduce_fusion_kernel_launcher(AllReduceFusionParams const& params)
{
    TLLM_CHECK(params.size % params.hidden_dim == 0);
    TLLM_CHECK(params.hidden_dim % kElemsPerAccess<DType> == 0);
    static int SM = tensorrt_llm::common::getSMVersion();
    int token_num = params.size / params.hidden_dim;
    bool oneshot = params.use_oneshot;
    int cluster_num = token_num;
    std::array<int, NRanks> begin_tokens, token_num_per_ranks;
    if (!oneshot)
    {
        int remaining_token = token_num % NRanks;
        int token_num_per_rank = token_num / NRanks;
        cluster_num = token_num_per_rank;
        if (remaining_token)
        {
            cluster_num++;
        }
        for (int r = 0; r < NRanks; ++r)
        {
            begin_tokens[r] = r * token_num_per_rank + (remaining_token > r ? r : remaining_token);
            token_num_per_ranks[r] = token_num_per_rank + (remaining_token > r ? 1 : 0);
        }
    }
    int threads_per_token = params.hidden_dim / kElemsPerAccess<DType>;
    int cluster_size;
    if (SM >= 90)
    {
        cluster_size = 8;
    }
    else
    {
        cluster_size = 1;
    }
    while (threads_per_token % cluster_size != 0 && cluster_size > 1)
    {
        cluster_size /= 2;
    }
    int threads_per_block = threads_per_token / cluster_size;
    while (threads_per_block < 128 && cluster_size >= 2)
    {
        threads_per_block *= 2;
        cluster_size /= 2;
    }
    int sm_count = get_sm_count();
    while (cluster_num * cluster_size > sm_count && cluster_size > 1 && threads_per_block <= 512)
    {
        threads_per_block *= 2;
        cluster_size /= 2;
    }
    TLLM_CHECK(oneshot || threads_per_block >= params.nranks);
    int block_size = threads_per_block;
    TLLM_CHECK(block_size <= 1024 && cluster_size > 0);

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
    cfg.numAttrs = SM >= 90 ? 2 : 0;
    if (oneshot)
    {
        bool trigger_completion_at_end = params.trigger_completion_at_end;
        if (trigger_completion_at_end)
        {
            launch_oneshot_lamport<Pattern, DType, NRanks, Fp32Acc, true>(params, cfg);
        }
        else
        {
            launch_oneshot_lamport<Pattern, DType, NRanks, Fp32Acc, false>(params, cfg);
        }
    }
    else
    {
        launch_twoshot_sync<Pattern, DType, NRanks, Fp32Acc>(params, cfg, begin_tokens, token_num_per_ranks);
    }
}

bool use_fp32_acc()
{
    // we use fp16 acc type by default due to keep align with nccl
    static char* fp32_acc = std::getenv("ALL_REDUCE_FUSION_KERNEL_ACC_FP32");
    return fp32_acc != nullptr;
}

void allreduce_fusion_op(AllReduceFusionParams const& params)
{
#define DISPATCH_ACC_TYPE(DType, Pattern, NRanks)                                                                      \
    if constexpr (std::is_same_v<DType, float>)                                                                        \
    {                                                                                                                  \
        return allreduce_fusion_kernel_launcher<Pattern, DType, NRanks, false>(params);                                \
    }                                                                                                                  \
    else                                                                                                               \
    {                                                                                                                  \
        if (fp32_acc)                                                                                                  \
        {                                                                                                              \
            return allreduce_fusion_kernel_launcher<Pattern, DType, NRanks, true>(params);                             \
        }                                                                                                              \
        else                                                                                                           \
        {                                                                                                              \
            return allreduce_fusion_kernel_launcher<Pattern, DType, NRanks, false>(params);                            \
        }                                                                                                              \
    }

#define DISPATCH_PATTERN(DType, NRanks)                                                                                \
    if (params.pattern == AllReduceFusionPattern::kAllReduce)                                                          \
    {                                                                                                                  \
        DISPATCH_ACC_TYPE(DType, AllReduceFusionPattern::kAllReduce, NRanks);                                          \
    }                                                                                                                  \
    else if (params.pattern == AllReduceFusionPattern::kARResidualRMSNorm)                                             \
    {                                                                                                                  \
        DISPATCH_ACC_TYPE(DType, AllReduceFusionPattern::kARResidualRMSNorm, NRanks);                                  \
    }                                                                                                                  \
    else if (params.pattern == AllReduceFusionPattern::kARResidualRMSNormFP8Quant)                                     \
    {                                                                                                                  \
        DISPATCH_ACC_TYPE(DType, AllReduceFusionPattern::kARResidualRMSNormFP8Quant, NRanks);                          \
    }                                                                                                                  \
    else if (params.pattern == AllReduceFusionPattern::kARResidualRMSNormFP4Quant)                                     \
    {                                                                                                                  \
        if constexpr (!std::is_same_v<DType, float>)                                                                   \
        {                                                                                                              \
            DISPATCH_ACC_TYPE(DType, AllReduceFusionPattern::kARResidualRMSNormFP4Quant, NRanks);                      \
        }                                                                                                              \
        else                                                                                                           \
        {                                                                                                              \
            TLLM_CHECK_WITH_INFO(false,                                                                                \
                "allreduce_fusion_kernel: AllReduceFusionPattern=kARResidualRMSNormFP4Quant can not work with "        \
                "DType=float!");                                                                                       \
        }                                                                                                              \
    }                                                                                                                  \
    else if (params.pattern == AllReduceFusionPattern::kARResidualRMSNormOutFP8Quant)                                  \
    {                                                                                                                  \
        DISPATCH_ACC_TYPE(DType, AllReduceFusionPattern::kARResidualRMSNormOutFP8Quant, NRanks);                       \
    }                                                                                                                  \
    else if (params.pattern == AllReduceFusionPattern::kARResidualRMSNormOutFP4Quant)                                  \
    {                                                                                                                  \
        if constexpr (!std::is_same_v<DType, float>)                                                                   \
        {                                                                                                              \
            DISPATCH_ACC_TYPE(DType, AllReduceFusionPattern::kARResidualRMSNormOutFP4Quant, NRanks);                   \
        }                                                                                                              \
        else                                                                                                           \
        {                                                                                                              \
            TLLM_CHECK_WITH_INFO(false,                                                                                \
                "allreduce_fusion_kernel: AllReduceFusionPattern=kARResidualRMSNormOutFP4Quant can not work with "     \
                "DType=float!");                                                                                       \
        }                                                                                                              \
    }                                                                                                                  \
    else                                                                                                               \
    {                                                                                                                  \
        TLLM_CHECK_WITH_INFO(false, "allreduce_fusion_kernel: unsupported pattern!");                                  \
    }

#define DISPATCH_DTYPE(NRanks)                                                                                         \
    if (params.dtype == nvinfer1::DataType::kHALF)                                                                     \
    {                                                                                                                  \
        DISPATCH_PATTERN(half, NRanks);                                                                                \
    }                                                                                                                  \
    else if (params.dtype == nvinfer1::DataType::kBF16)                                                                \
    {                                                                                                                  \
        DISPATCH_PATTERN(__nv_bfloat16, NRanks);                                                                       \
    }                                                                                                                  \
    else if (params.dtype == nvinfer1::DataType::kFLOAT)                                                               \
    {                                                                                                                  \
        DISPATCH_PATTERN(float, NRanks);                                                                               \
    }                                                                                                                  \
    else                                                                                                               \
    {                                                                                                                  \
        TLLM_CHECK_WITH_INFO(false, "allreduce_fusion_kernel: unsupported dtype!");                                    \
    }

#define DISPATCH_RANKS(NRanks)                                                                                         \
    if (params.nranks == NRanks)                                                                                       \
    {                                                                                                                  \
        DISPATCH_DTYPE(NRanks);                                                                                        \
    }

    bool fp32_acc = use_fp32_acc();
    DISPATCH_RANKS(2);
    DISPATCH_RANKS(4);
    DISPATCH_RANKS(8);
    DISPATCH_RANKS(16);
    TLLM_CHECK_WITH_INFO(false, "allreduce_fusion_kernel: unsupported ranks number!");
}
}; // namespace tensorrt_llm::kernels::ar_fusion
