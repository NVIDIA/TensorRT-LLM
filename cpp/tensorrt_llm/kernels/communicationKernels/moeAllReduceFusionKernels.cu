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
#include "tensorrt_llm/kernels/communicationKernels/moeAllReduceFusionKernels.h"
#include "tensorrt_llm/kernels/quantization.cuh"
#include <cooperative_groups.h>

namespace tensorrt_llm::kernels::ar_fusion::moe
{
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
        constexpr int SF_VEC_SIZE = 16;
        using PackedVec = PackedVec<DType>;
        PackedVec pack_val = *reinterpret_cast<PackedVec const*>(&norm_val);
        auto sf_out = cvt_quant_get_sf_out_offset<uint32_t, 2>(std::nullopt /* batchIdx */, token_id,
            access_id_in_token, std::nullopt /* numRows */, params.hidden_dim / SF_VEC_SIZE,
            reinterpret_cast<uint32_t*>(params.scale_out), params.layout);
        reinterpret_cast<uint32_t*>(params.quant_out)[access_id]
            = cvt_warp_fp16_to_fp4<DType, SF_VEC_SIZE, false>(pack_val, *params.scale_factor, sf_out);
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

bool use_oneshot(int token_num)
{
    return token_num <= kOneShotMaxToken;
}

/////////////////////////////////////////////////////////////////
//                  * MoE Reduction Fusion *                   //
/////////////////////////////////////////////////////////////////

template <typename DType, int NRanks, bool ResidualOut, bool NormOut, bool QuantOut>
__global__ void moereduce_allreduce_fusion_kernel_oneshot_lamport(MoeReductionAllReduceFusionParams params)
{
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    namespace cg = cooperative_groups;
    cg::cluster_group cluster = cg::this_cluster();
    cg::grid_group grid = cg::this_grid();

    // Each token is handled by one cluster
    // which token is handled by current cluster
    int token_id = grid.cluster_rank();
    // total number of token
    int num_token = params.size / params.hidden_dim;
    // Each thread handle kElemsPerAccess num elem in token. Total cluster.num_threads() to handle one token
    // For current token, which kElemsPerAccess is handled by current thread (in unit of kElemsPerAccess)
    int access_id_in_token = cluster.thread_rank();
    // Across all token, which kElemsPerAccess is handled by current thread (in unit of kElemsPerAccess)
    int access_id = token_id * params.hidden_dim / kElemsPerAccess + access_id_in_token;
    // Persistent kernel
    // stride to next token handled by current cta
    int token_stride = grid.num_clusters();
    // stride in unit of kElemsPerAccess
    int access_stride = token_stride * params.hidden_dim / kElemsPerAccess;
    // Total number of access in unit of kElemsPerAccess to handle (token_num * hidden_dim)
    // This is within one rank
    int tot_access = params.size / kElemsPerAccess;
    float4 clear_vec = get_neg_zero();

    cudaGridDependencySynchronize();
    cudaTriggerProgrammaticLaunchCompletion();
    LamportComm<NRanks> comm(params.workspace, params.rank);
    int clear_access = comm.clear_size / kElemsPerAccess;

    // * MoE related
    int threadid_in_cluster = cluster.thread_rank();
    // Start Offset within one token's hidden_size of element
    // Current thread handle token[thread_offset_within_token : thread_offset_within_token + kElemsPerAccess]
    int thread_offset_within_token = threadid_in_cluster * kElemsPerAccess;

    union ACC_TYPE
    {
        float4 packed;
        DType unpacked[kElemsPerAccess];
    };

    // Persistent Kernel
    // Each cluster iterate through all token it need to handle
    for (int token_id = grid.cluster_rank(); token_id < num_token; token_id += grid.num_clusters())
    {
        if (thread_offset_within_token >= params.hidden_dim)
        {
            break;
        }

        // * MoE Reduce
        // Offset within (num_token, hidden_size) in unit of element
        int thread_offset_across_token = token_id * params.hidden_dim + thread_offset_within_token;

        ACC_TYPE accumulator;
#pragma unroll
        for (int i = 0; i < kElemsPerAccess; ++i)
        {
            accumulator.unpacked[i] = static_cast<DType>(0);
        }

        // * Iterate through all active expert
        int num_actexp = *(params.moe_reduction_device_num_experts);
        for (int actexp_i = 0; actexp_i < num_actexp; ++actexp_i)
        {
            // * Load active expert i's token j's partial data
            // Offset within (num_act_exp, num_token, hidden_size) in unit of element
            int thread_offset_across_actexp_token
                = actexp_i * (params.hidden_dim * num_token) + thread_offset_across_token;
            ACC_TYPE actexp_i_data;
            actexp_i_data.packed = reinterpret_cast<float4 const*>(
                params.moe_reduction_active_experts_token_input)[thread_offset_across_actexp_token / kElemsPerAccess];

            // * Load active expert i's token j's scale
            int thread_offset_scale = actexp_i * num_token + token_id;
            float actexp_i_token_j_scale
                = reinterpret_cast<float const*>(params.moe_reduction_scale_input)[thread_offset_scale];

            // * acc += scale(data)
#pragma unroll
            for (int i = 0; i < kElemsPerAccess; ++i)
            {
                // assume computation is done in ScaleType
                accumulator.unpacked[i]
                    += static_cast<DType>((static_cast<float>(actexp_i_data.unpacked[i]) * actexp_i_token_j_scale));
            }
        }

        // * FC2 + reduced(gGEMM2)
        ACC_TYPE fc2_data;
        fc2_data.packed = reinterpret_cast<float4 const*>(
            params.moe_reduction_token_input)[thread_offset_across_token / kElemsPerAccess];
#pragma unroll
        for (int i = 0; i < kElemsPerAccess; ++i)
        {
            accumulator.unpacked[i] += fc2_data.unpacked[i];
        }

        // * AR Store
        int access_id = token_id * params.hidden_dim / kElemsPerAccess + access_id_in_token;
        int idx = access_id;
        alignas(16) float val[4]
            = {accumulator.packed.x, accumulator.packed.y, accumulator.packed.z, accumulator.packed.w};

#pragma unroll
        for (int i = 0; i < 4; ++i)
        {
            // Handle two bf16/fp16 at one time
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
    }

    // * Clear previous buffer
    for (int idx = access_id; idx < clear_access; idx += access_stride)
    {
        reinterpret_cast<float4*>(comm.clear_buf)[idx] = clear_vec;
    }

    // * AR Load + Fusion
    for (int idx = access_id, tidx = token_id; idx < tot_access; idx += access_stride, tidx += token_stride)
    {
        // * AR Load
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
            sum_val = add128<DType>(sum_val, vals[r]);
        }

        // * Fuse
        fused_op<ResidualOut, NormOut, QuantOut, DType>(sum_val, idx, tidx, access_id_in_token, params);
    }
    comm.update(params.size * NRanks);
#endif
}

template <typename DType, int NRanks, bool ResidualOut, bool NormOut, bool QuantOut>
void launch_oneshot_moereduce_lamport(MoeReductionAllReduceFusionParams const& params, cudaLaunchConfig_t& cfg)
{
    TLLM_CUDA_CHECK(cudaLaunchKernelEx(&cfg,
        moereduce_allreduce_fusion_kernel_oneshot_lamport<DType, NRanks, ResidualOut, NormOut, QuantOut>, params));
}

template <typename DType, int NRanks, bool ResidualOut, bool NormOut, bool QuantOut>
void moereduction_allreduce_fusion_kernel_launcher(MoeReductionAllReduceFusionParams const& params)
{
    int token_num = params.size / params.hidden_dim;
    bool oneshot = use_oneshot(token_num);
    // Only support one shot
    TLLM_CHECK(oneshot);
    // Each token is handled by one cluster
    int cluster_num = token_num;
    // Total number of threads (within one cluster) that's need to handle one token
    // given that each thread handle kElemsPerAccess
    int threads_per_token = params.hidden_dim / kElemsPerAccess;
    // Total number of warp (within one cluster) that's need to handle one token
    // given that each thread handle kElemsPerAccess
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
        launch_oneshot_moereduce_lamport<DType, NRanks, ResidualOut, NormOut, QuantOut>(params, cfg);
    }
}

void moereduction_allreduce_fusion_op(MoeReductionAllReduceFusionParams const& params)
{
#define MOE_DISPATCH1(DTYPE, NRANKS, RESIDUAL_OUT, NORM_OUT, QUANT_OUT)                                                \
    return moereduction_allreduce_fusion_kernel_launcher<DTYPE, NRANKS, RESIDUAL_OUT, NORM_OUT, QUANT_OUT>(params);
#define MOE_DISPATCH0(NRANKS, RESIDUAL_OUT, NORM_OUT, QUANT_OUT)                                                       \
    if (params.nranks == NRANKS && params.dtype == nvinfer1::DataType::kHALF)                                          \
    {                                                                                                                  \
        MOE_DISPATCH1(half, NRANKS, RESIDUAL_OUT, NORM_OUT, QUANT_OUT);                                                \
    }                                                                                                                  \
    else if (params.nranks == NRANKS && params.dtype == nvinfer1::DataType::kBF16)                                     \
    {                                                                                                                  \
        MOE_DISPATCH1(__nv_bfloat16, NRANKS, RESIDUAL_OUT, NORM_OUT, QUANT_OUT);                                       \
    }

    TLLM_CHECK(params.residual_in && params.rms_gamma);
    TLLM_CHECK(params.moe_reduction_scale_input && params.moe_reduction_active_experts_token_input
        && params.moe_reduction_token_input);
    TLLM_CHECK(params.size % params.hidden_dim == 0);
    TLLM_CHECK(params.hidden_dim % kElemsPerAccess == 0);
    if (params.residual_out && not params.norm_out && params.quant_out)
    {
        // pattern1: AR+Add_RMS+Quant
        // [m, 7168] bf16 allreduce_in, [m, 7168] bf16 residual_in
        // [m, 7168] bf16 residual_out, [m, 7168] fp4 quant_out
        MOE_DISPATCH0(2, true, false, true);
        MOE_DISPATCH0(4, true, false, true);
        MOE_DISPATCH0(8, true, false, true);
        MOE_DISPATCH0(16, true, false, true);
    }
    else if (not params.residual_out && params.norm_out && not params.quant_out)
    {
        // pattern2: AR+AddRMS
        // [m, 7168] bf16 allreduce_in, [m, 7168] bf16 residual_in
        // [m, 7168] bf16 norm_out
        MOE_DISPATCH0(2, false, true, false);
        MOE_DISPATCH0(4, false, true, false);
        MOE_DISPATCH0(8, false, true, false);
        MOE_DISPATCH0(16, false, true, false);
    }
    else if (params.residual_out && params.norm_out && not params.quant_out)
    {
        MOE_DISPATCH0(2, true, true, false);
        MOE_DISPATCH0(4, true, true, false);
        MOE_DISPATCH0(8, true, true, false);
        MOE_DISPATCH0(16, true, true, false);
    }
    else if (params.residual_out && params.norm_out && params.quant_out)
    {
        // for test
        MOE_DISPATCH0(2, true, true, true);
        MOE_DISPATCH0(4, true, true, true);
        MOE_DISPATCH0(8, true, true, true);
        MOE_DISPATCH0(16, true, true, true);
    }
    TLLM_CHECK_WITH_INFO(false, "allreduce_fusion_kernel: unsupported pattern!");
}

/////////////////////////////////////////////////////////////////
//                  * MoE Finalize Allreduce Fusion *                   //
/////////////////////////////////////////////////////////////////

template <typename DType, int NRanks, bool ResidualOut, bool NormOut, bool QuantOut, typename ScaleType = DType>
__global__ void moefinalize_allreduce_fusion_kernel_oneshot_lamport(MoeFinalizeAllReduceFusionParams params)
{
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    namespace cg = cooperative_groups;
    cg::cluster_group cluster = cg::this_cluster();
    cg::grid_group grid = cg::this_grid();

    // Each token is handled by one cluster
    // which token is handled by current cluster
    int token_id = grid.cluster_rank();
    // total number of token
    int num_token = params.size / params.hidden_dim;
    // Each thread handle kElemsPerAccess num elem in token. Total cluster.num_threads() to handle one token
    // For current token, which kElemsPerAccess is handled by current thread (in unit of kElemsPerAccess)
    int access_id_in_token = cluster.thread_rank();
    // Across all token, which kElemsPerAccess is handled by current thread (in unit of kElemsPerAccess)
    int access_id = token_id * params.hidden_dim / kElemsPerAccess + access_id_in_token;
    // Persistent kernel
    // stride to next token handled by current cta
    int token_stride = grid.num_clusters();
    // stride in unit of kElemsPerAccess
    int access_stride = token_stride * params.hidden_dim / kElemsPerAccess;
    // Total number of access in unit of kElemsPerAccess to handle (token_num * hidden_dim)
    // This is within one rank
    int tot_access = params.size / kElemsPerAccess;
    float4 clear_vec = get_neg_zero();

    cudaGridDependencySynchronize();
    cudaTriggerProgrammaticLaunchCompletion();
    LamportComm<NRanks> comm(params.workspace, params.rank);
    int clear_access = comm.clear_size / kElemsPerAccess;

    // * MoE related
    int threadid_in_cluster = cluster.thread_rank();
    // Start Offset within one token's hidden_size of element
    // Current thread handle token[thread_offset_within_token : thread_offset_within_token + kElemsPerAccess]
    int thread_offset_within_token = threadid_in_cluster * kElemsPerAccess;

    union ACC_TYPE
    {
        float4 packed;
        DType unpacked[kElemsPerAccess];
    };

    int top_k = params.top_k;
    bool use_scale_factor = params.expert_scale_factor != nullptr;

    // Persistent Kernel
    // Each cluster iterate through all token it need to handle
    for (int token_id = grid.cluster_rank(); token_id < num_token; token_id += grid.num_clusters())
    {
        if (thread_offset_within_token >= params.hidden_dim)
        {
            break;
        }

        // * MoE finalize
        ACC_TYPE accumulator;
#pragma unroll
        for (int i = 0; i < kElemsPerAccess; ++i)
        {
            accumulator.unpacked[i] = static_cast<DType>(0);
        }

        for (int k = 0; k < top_k; k++)
        {
            int const expanded_idx = token_id * top_k + k;
            int32_t const permuted_idx = params.expanded_idx_to_permuted_idx[expanded_idx];

            if (permuted_idx == -1)
                continue;

            int thread_offset_across_token = permuted_idx * params.hidden_dim + thread_offset_within_token;
            float block_scale = 1.0;
            if (use_scale_factor)
            {
                block_scale = static_cast<float>(static_cast<ScaleType*>(params.expert_scale_factor)[expanded_idx]);
            }

            ACC_TYPE permuted_data;
            permuted_data.packed
                = reinterpret_cast<float4 const*>(params.allreduce_in)[thread_offset_across_token / kElemsPerAccess];

            // * acc += scale(data)
#pragma unroll
            for (int i = 0; i < kElemsPerAccess; ++i)
            {
                // assume computation is done in ScaleType
                accumulator.unpacked[i]
                    += static_cast<DType>((static_cast<float>(permuted_data.unpacked[i]) * block_scale));
            }
        }

        // * Add shared expert output
        if (params.shared_expert_output)
        {
            // * Load shared expert output
            int thread_offset_across_token = token_id * params.hidden_dim + thread_offset_within_token;
            ACC_TYPE shared_expert_output;
            shared_expert_output.packed = reinterpret_cast<float4 const*>(
                params.shared_expert_output)[thread_offset_across_token / kElemsPerAccess];
#pragma unroll
            for (int i = 0; i < kElemsPerAccess; ++i)
            {
                accumulator.unpacked[i] += shared_expert_output.unpacked[i];
            }
        }

        // * AR Store
        int access_id = token_id * params.hidden_dim / kElemsPerAccess + access_id_in_token;
        int idx = access_id;
        alignas(16) float val[4]
            = {accumulator.packed.x, accumulator.packed.y, accumulator.packed.z, accumulator.packed.w};

#pragma unroll
        for (int i = 0; i < 4; ++i)
        {
            // Handle two bf16/fp16 at one time
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
    }

    // * Clear previous buffer
    for (int idx = access_id; idx < clear_access; idx += access_stride)
    {
        reinterpret_cast<float4*>(comm.clear_buf)[idx] = clear_vec;
    }

    // * AR Load + Fusion
    for (int idx = access_id, tidx = token_id; idx < tot_access; idx += access_stride, tidx += token_stride)
    {
        // * AR Load
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
            sum_val = add128<DType>(sum_val, vals[r]);
        }

        // * Fuse
        fused_op<ResidualOut, NormOut, QuantOut, DType>(sum_val, idx, tidx, access_id_in_token, params);
    }
    comm.update(params.size * NRanks);

#endif
}

template <typename DType, int NRanks, bool ResidualOut, bool NormOut, bool QuantOut, typename ScaleType = DType>
void launch_oneshot_moefinalize_lamport(MoeFinalizeAllReduceFusionParams const& params, cudaLaunchConfig_t& cfg)
{
    TLLM_CUDA_CHECK(cudaLaunchKernelEx(&cfg,
        moefinalize_allreduce_fusion_kernel_oneshot_lamport<DType, NRanks, ResidualOut, NormOut, QuantOut, ScaleType>,
        params));
}

template <typename DType, int NRanks, bool ResidualOut, bool NormOut, bool QuantOut, typename ScaleType = DType>
void moefinalize_allreduce_fusion_kernel_launcher(MoeFinalizeAllReduceFusionParams const& params)
{
    int token_num = params.size / params.hidden_dim;
    bool oneshot = use_oneshot(token_num);
    // Only support one shot
    TLLM_CHECK(oneshot);
    // Each token is handled by one cluster
    int cluster_num = token_num;
    // Total number of threads (within one cluster) that's need to handle one token
    // given that each thread handle kElemsPerAccess
    int threads_per_token = params.hidden_dim / kElemsPerAccess;
    // Total number of warp (within one cluster) that's need to handle one token
    // given that each thread handle kElemsPerAccess
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
        launch_oneshot_moefinalize_lamport<DType, NRanks, ResidualOut, NormOut, QuantOut, ScaleType>(params, cfg);
    }
}

void moefinalize_allreduce_fusion_op(MoeFinalizeAllReduceFusionParams const& params)
{
#define MOE_FINALIZE_DISPATCH1(DTYPE, NRANKS, RESIDUAL_OUT, NORM_OUT, QUANT_OUT)                                       \
    return moefinalize_allreduce_fusion_kernel_launcher<DTYPE, NRANKS, RESIDUAL_OUT, NORM_OUT, QUANT_OUT>(params);
#define MOE_FINALIZE_DISPATCH0(NRANKS, RESIDUAL_OUT, NORM_OUT, QUANT_OUT)                                              \
    if (params.nranks == NRANKS && params.dtype == nvinfer1::DataType::kHALF                                           \
        && params.scale_dtype == nvinfer1::DataType::kHALF)                                                            \
    {                                                                                                                  \
        MOE_FINALIZE_DISPATCH1(half, NRANKS, RESIDUAL_OUT, NORM_OUT, QUANT_OUT);                                       \
    }                                                                                                                  \
    else if (params.nranks == NRANKS && params.dtype == nvinfer1::DataType::kBF16                                      \
        && params.scale_dtype == nvinfer1::DataType::kBF16)                                                            \
    {                                                                                                                  \
        MOE_FINALIZE_DISPATCH1(__nv_bfloat16, NRANKS, RESIDUAL_OUT, NORM_OUT, QUANT_OUT);                              \
    }

    TLLM_CHECK(params.allreduce_in && params.expanded_idx_to_permuted_idx && params.top_k);
    TLLM_CHECK(params.size % params.hidden_dim == 0);
    TLLM_CHECK(params.hidden_dim % kElemsPerAccess == 0);
    if (params.residual_out && not params.norm_out && params.quant_out)
    {
        // pattern1: AR+Add_RMS+Quant
        // [m, 7168] bf16 allreduce_in, [m, 7168] bf16 residual_in
        // [m, 7168] bf16 residual_out, [m, 7168] fp4 quant_out
        MOE_FINALIZE_DISPATCH0(2, true, false, true);
        MOE_FINALIZE_DISPATCH0(4, true, false, true);
        MOE_FINALIZE_DISPATCH0(8, true, false, true);
        MOE_FINALIZE_DISPATCH0(16, true, false, true);
    }
    else if (not params.residual_out && params.norm_out && not params.quant_out)
    {
        // pattern2: AR+AddRMS
        // [m, 7168] bf16 allreduce_in, [m, 7168] bf16 residual_in
        // [m, 7168] bf16 norm_out
        MOE_FINALIZE_DISPATCH0(2, false, true, false);
        MOE_FINALIZE_DISPATCH0(4, false, true, false);
        MOE_FINALIZE_DISPATCH0(8, false, true, false);
        MOE_FINALIZE_DISPATCH0(16, false, true, false);
    }
    else if (params.residual_out && params.norm_out && not params.quant_out)
    {
        MOE_FINALIZE_DISPATCH0(2, true, true, false);
        MOE_FINALIZE_DISPATCH0(4, true, true, false);
        MOE_FINALIZE_DISPATCH0(8, true, true, false);
        MOE_FINALIZE_DISPATCH0(16, true, true, false);
    }
    else if (params.residual_out && params.norm_out && params.quant_out)
    {
        // for test
        MOE_FINALIZE_DISPATCH0(2, true, true, true);
        MOE_FINALIZE_DISPATCH0(4, true, true, true);
        MOE_FINALIZE_DISPATCH0(8, true, true, true);
        MOE_FINALIZE_DISPATCH0(16, true, true, true);
    }
    TLLM_CHECK_WITH_INFO(false, "moefinalize_allreduce_fusion_op: unsupported pattern!");
#undef MOE_FINALIZE_DISPATCH0
#undef MOE_FINALIZE_DISPATCH1
}

}; // namespace tensorrt_llm::kernels::ar_fusion::moe
