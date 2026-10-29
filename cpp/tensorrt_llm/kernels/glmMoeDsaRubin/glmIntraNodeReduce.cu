/*
 * Copyright (c) 2022-2026, NVIDIA CORPORATION.  All rights reserved.
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
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/envUtils.h"
#include "tensorrt_llm/common/reduceKernelUtils.cuh"
#include "tensorrt_llm/kernels/glmMoeDsaRubin/glmIntraNodeLamport.cuh"
#include "tensorrt_llm/kernels/glmMoeDsaRubin/glmIntraNodeReduce.h"
#include <cooperative_groups.h>
#include <cstdlib>
#include <cuda_bf16.h>
#include <map>
#include <tuple>
#include <type_traits>
TRTLLM_NAMESPACE_BEGIN

namespace kernels::glm_moe_dsa
{
namespace
{
using DType = __nv_bfloat16;
namespace cg = cooperative_groups;
constexpr int kFinalizeFastMaxTopK = 8;
constexpr int kFinalizeFastMaxThreads = 768;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
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

// FastNorm: a cluster's partials are exchanged with ipc::exchangeNormPartials (the kernel called
// ipc::initNormExchange before its grid dependency wait) instead of two cluster barriers; same sum.
template <typename DType, typename PackedType, bool FastNorm = false>
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
        if constexpr (FastNorm)
        {
            float const clusterSum = ipc::exchangeNormPartials(acc);
            if (threadIdx.x == 0)
            {
                acc = clusterSum;
            }
        }
        else
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
        float acc_f32[8];
#pragma unroll
        for (int i = 0; i < 8; ++i)
        {
            acc_f32[i] = static_cast<float>(reinterpret_cast<DType*>(&vals[0])[i]);
        }
#pragma unroll
        for (int r = 1; r < NRanks; ++r)
        {
#pragma unroll
            for (int i = 0; i < 8; ++i)
            {
                acc_f32[i] += static_cast<float>(reinterpret_cast<DType*>(&vals[r])[i]);
            }
        }
        float4 acc;
#pragma unroll
        for (int i = 0; i < 8; ++i)
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

#endif
template <int NRanks, bool Fp32Acc, bool TriggerCompletionAtEnd>
__global__ void __launch_bounds__(1024) reduceKernel(IntraNodeReduceParams params)
{
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    auto cluster = cg::this_cluster();
    auto grid = cg::this_grid();
    int token_id = grid.cluster_rank();
    int access_id_in_token = cluster.thread_rank();
    int access_id = token_id * params.hidden_dim / 8 + access_id_in_token;
    int access_stride = grid.num_clusters() * params.hidden_dim / 8;
    int tot_access = params.size / 8;
    float4 clear_vec = get_neg_zero();
    float4 const gamma = reinterpret_cast<float4 const*>(params.rms_gamma)[access_id_in_token];
    float4 const residual = reinterpret_cast<float4 const*>(params.residual_in)[access_id];
    ipc::LamportCommSplit<NRanks> comm(params.workspace, params.rank, params.flags, params.layout);
    if (cg::this_cluster().num_blocks() > 1)
    {
        ipc::initNormExchange();
    }

    cudaGridDependencySynchronize();
    if constexpr (!TriggerCompletionAtEnd)
    {
        cudaTriggerProgrammaticLaunchCompletion();
    }
    alignas(16) float val[4];
    *reinterpret_cast<float4*>(val) = reinterpret_cast<float4 const*>(params.allreduce_in)[access_id];
    comm.loadFlags();
    comm.arrive();
    int clear_access = comm.clear_size / 8;

#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        if (is_neg_zero(val[i]))
        {
            val[i] = 0.f;
        }
    }
    float4 const own = *reinterpret_cast<float4*>(val);
#pragma unroll
    for (int r = 0; r < NRanks; ++r)
    {
        reinterpret_cast<float4*>(comm.data_bufs[r])[params.rank * tot_access + access_id] = own;
    }
    for (int idx = access_id; idx < clear_access; idx += access_stride)
    {
        reinterpret_cast<float4*>(comm.clear_buf)[idx] = clear_vec;
    }

    bool rotating = blockIdx.x == 0 && threadIdx.x == 0;
    int arrivals = rotating ? comm.loadArrivals() : 0;
    float4 vals[NRanks];
    bool done = false;
    while (!done)
    {
        done = true;
#pragma unroll
        for (int r = 0; r < NRanks; ++r)
        {
            if (r != params.rank)
            {
                vals[r] = ld_global_volatile(&reinterpret_cast<float4*>(comm.local_data)[r * tot_access + access_id]);
                done &= !is_neg_zero(vals[r]);
            }
        }
    }
#pragma unroll
    for (int r = 0; r < NRanks; ++r)
    {
        if (r == params.rank)
        {
            vals[r] = own;
        }
    }
    if (rotating && arrivals == static_cast<int>(gridDim.x))
    {
        comm.rotate(arrivals, params.size * NRanks);
        rotating = false;
    }
    float4 sum_val = allreduce_sum<DType, NRanks, Fp32Acc>(vals);
    float4 const added = add128<DType>(sum_val, residual);
    reinterpret_cast<float4*>(params.residual_out)[access_id] = added;
    reinterpret_cast<float4*>(params.norm_out)[access_id]
        = rms_norm<DType, float4, true>(added, gamma, params.rms_eps, params.hidden_dim);
    if (rotating)
    {
        comm.rotate(arrivals, params.size * NRanks);
    }

    if constexpr (TriggerCompletionAtEnd)
    {
        cudaTriggerProgrammaticLaunchCompletion();
    }
#endif
}

template <int NRanks>
__global__ void __launch_bounds__(kFinalizeFastMaxThreads) finalizeKernel(IntraNodeReduceParams params)
{
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    namespace cg = cooperative_groups;
    cg::cluster_group cluster = cg::this_cluster();
    cg::grid_group grid = cg::this_grid();

    int token_id = grid.cluster_rank();
    int access_id_in_token = cluster.thread_rank();
    int access_id = token_id * params.hidden_dim / 8 + access_id_in_token;
    int access_stride = grid.num_clusters() * params.hidden_dim / 8;
    int tot_access = params.size / 8;
    int thread_offset_within_token = access_id_in_token * 8;
    float4 clear_vec = get_neg_zero();

    union ACC_TYPE
    {
        float4 packed;
        DType unpacked[8];
    };

    float4 const gamma_val = reinterpret_cast<float4 const*>(params.rms_gamma)[access_id_in_token];
    ipc::LamportCommSplit<NRanks> comm(params.workspace, params.rank, params.flags, params.layout);
    if (cluster.num_blocks() > 1)
    {
        ipc::initNormExchange();
    }
    // The lines read right after the wait, brought into L2 while it is pending: the token's routing (thread 0), and
    // the residual and shared-expert rows (one thread per 128-byte line).
    if (threadIdx.x == 0)
    {
        ipc::prefetchL2(params.expanded_idx_to_permuted_idx + token_id * params.top_k);
        if (params.expert_scale_factor)
        {
            ipc::prefetchL2(static_cast<DType const*>(params.expert_scale_factor) + token_id * params.top_k);
        }
    }
    if (access_id_in_token % (128 / sizeof(float4)) == 0)
    {
        if (params.residual_in)
        {
            ipc::prefetchL2(reinterpret_cast<float4 const*>(params.residual_in) + access_id);
        }
        if (params.shared_expert_output)
        {
            ipc::prefetchL2(reinterpret_cast<float4 const*>(params.shared_expert_output) + access_id);
        }
    }

    cudaGridDependencySynchronize();
    cudaTriggerProgrammaticLaunchCompletion();

    int const top_k = params.top_k;
    bool const use_scale_factor = params.expert_scale_factor != nullptr;
    int32_t permuted_idx[kFinalizeFastMaxTopK];
    float block_scale[kFinalizeFastMaxTopK];
#pragma unroll
    for (int k = 0; k < kFinalizeFastMaxTopK; ++k)
    {
        permuted_idx[k] = -1;
        block_scale[k] = 1.0;
        if (k < top_k)
        {
            int const expanded_idx = token_id * top_k + k;
            permuted_idx[k] = params.expanded_idx_to_permuted_idx[expanded_idx];
            if (use_scale_factor)
            {
                block_scale[k]
                    = static_cast<float>(static_cast<DType const*>(params.expert_scale_factor)[expanded_idx]);
            }
        }
    }
    ACC_TYPE shared_expert_output;
    shared_expert_output.packed = make_float4(0.f, 0.f, 0.f, 0.f);
    if (params.shared_expert_output)
    {
        shared_expert_output.packed = reinterpret_cast<float4 const*>(params.shared_expert_output)[access_id];
    }
    float4 residual_val = make_float4(0.f, 0.f, 0.f, 0.f);
    if (params.residual_in)
    {
        residual_val = reinterpret_cast<float4 const*>(params.residual_in)[access_id];
    }
    comm.loadFlags();

    ACC_TYPE permuted_data[kFinalizeFastMaxTopK];
#pragma unroll
    for (int k = 0; k < kFinalizeFastMaxTopK; ++k)
    {
        if (permuted_idx[k] != -1)
        {
            int thread_offset_across_token = permuted_idx[k] * params.hidden_dim + thread_offset_within_token;
            permuted_data[k].packed
                = reinterpret_cast<float4 const*>(params.allreduce_in)[thread_offset_across_token / 8];
        }
    }
    comm.arrive();
    int clear_access = comm.clear_size / 8;

    // The finalize of the default kernel: top-k slots in order, -1 slots skipped, then the shared expert's output, in
    // fp32, rounded to DType once.
    float facc[8];
#pragma unroll
    for (int i = 0; i < 8; ++i)
    {
        facc[i] = 0.f;
    }
#pragma unroll
    for (int k = 0; k < kFinalizeFastMaxTopK; ++k)
    {
        if (permuted_idx[k] == -1)
            continue;
#pragma unroll
        for (int i = 0; i < 8; ++i)
        {
            facc[i] += static_cast<float>(permuted_data[k].unpacked[i]) * block_scale[k];
        }
    }
    if (params.shared_expert_output)
    {
#pragma unroll
        for (int i = 0; i < 8; ++i)
        {
            facc[i] += static_cast<float>(shared_expert_output.unpacked[i]);
        }
    }
    ACC_TYPE accumulator;
#pragma unroll
    for (int i = 0; i < 8; ++i)
    {
        accumulator.unpacked[i] = static_cast<DType>(facc[i]);
    }
    alignas(16) float val[4] = {accumulator.packed.x, accumulator.packed.y, accumulator.packed.z, accumulator.packed.w};
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        if (is_neg_zero(val[i]))
        {
            val[i] = 0.f;
        }
    }
    float4 const own = *reinterpret_cast<float4*>(val);
#pragma unroll
    for (int r = 0; r < NRanks; ++r)
    {
        reinterpret_cast<float4*>(comm.data_bufs[r])[params.rank * tot_access + access_id] = own;
    }
    for (int idx = access_id; idx < clear_access; idx += access_stride)
    {
        reinterpret_cast<float4*>(comm.clear_buf)[idx] = clear_vec;
    }

    bool rotating = blockIdx.x == 0 && threadIdx.x == 0;
    int arrivals = rotating ? comm.loadArrivals() : 0;
    float4 vals[NRanks];
    bool done = false;
    while (!done)
    {
        done = true;
#pragma unroll
        for (int r = 0; r < NRanks; ++r)
        {
            if (r != params.rank)
            {
                vals[r] = ld_global_volatile(&reinterpret_cast<float4*>(comm.local_data)[r * tot_access + access_id]);
                done &= !is_neg_zero(vals[r]);
            }
        }
    }
#pragma unroll
    for (int r = 0; r < NRanks; ++r)
    {
        if (r == params.rank)
        {
            vals[r] = own;
        }
    }
    if (rotating && arrivals == static_cast<int>(gridDim.x))
    {
        comm.rotate(arrivals, params.size * NRanks);
        rotating = false;
    }
    float4 sum_val = vals[0];
#pragma unroll
    for (int r = 1; r < NRanks; ++r)
    {
        sum_val = add128<DType>(sum_val, vals[r]);
    }

    // The epilogue of fused_op, with the residual and gamma already loaded.
    float4 norm_input;
    if (params.residual_in)
    {
        norm_input = add128<DType>(sum_val, residual_val);
        reinterpret_cast<float4*>(params.residual_out)[access_id] = norm_input;
    }
    else
    {
        norm_input = sum_val;
    }
    float4 norm_val = rms_norm<DType, float4, true>(norm_input, gamma_val, params.rms_eps, params.hidden_dim);
    reinterpret_cast<float4*>(params.norm_out)[access_id] = norm_val;
    if (rotating)
    {
        comm.rotate(arrivals, params.size * NRanks);
    }
#endif
}

struct LaunchShape
{
    int cluster{};
    int block{};
    bool valid{};
};

bool fp32Acc()
{
    static char const* value = std::getenv("ALL_REDUCE_FUSION_KERNEL_ACC_FP32");
    return value == nullptr || value[0] != '0';
}

LaunchShape launchShape(int rows, int hidden, int ranks, bool finalize)
{
    if (rows < 1 || rows > 24 || hidden != 6144 || ranks != 4)
        return {};
    int device;
    TLLM_CUDA_CHECK(cudaGetDevice(&device));
    using Key = std::tuple<int, int, bool>;
    static thread_local std::map<Key, LaunchShape> cache;
    Key key{device, rows, finalize};
    if (auto it = cache.find(key); it != cache.end())
        return it->second;
    cudaDeviceProp prop;
    TLLM_CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    LaunchShape shape{};
    if (prop.major == 10 && prop.minor == 7)
    {
        int cluster = 8;
        if (finalize)
        {
            if (common::getBoolEnv("TRTLLM_MOE_FINALIZE_CTA1"))
            {
                int active = 0;
                TLLM_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active, finalizeKernel<4>, 768, 0));
                if (active > 0)
                    cluster = 1;
            }
        }
        else
        {
            while (768 / cluster < 128 || (rows * cluster > prop.multiProcessorCount && cluster > 1))
                cluster /= 2;
        }
        int block = 768 / cluster;
        int active = 0;
        if (finalize)
            TLLM_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active, finalizeKernel<4>, block, 0));
        else if (fp32Acc())
            TLLM_CUDA_CHECK(
                cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active, reduceKernel<4, true, true>, block, 0));
        else
            TLLM_CUDA_CHECK(
                cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active, reduceKernel<4, false, true>, block, 0));
        shape = {cluster, block, active > 0 && rows * cluster <= prop.multiProcessorCount};
    }
    cache.emplace(key, shape);
    return shape;
}

template <typename Kernel>
void launch(Kernel kernel, IntraNodeReduceParams const& p, LaunchShape shape)
{
    cudaLaunchAttribute attrs[2]{};
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[0].val.programmaticStreamSerializationAllowed = common::getEnvEnablePDL() ? 1 : 0;
    attrs[1].id = cudaLaunchAttributeClusterDimension;
    attrs[1].val.clusterDim = {static_cast<unsigned>(shape.cluster), 1, 1};
    cudaLaunchConfig_t cfg{};
    cfg.gridDim = (p.size / p.hidden_dim) * shape.cluster;
    cfg.blockDim = shape.block;
    cfg.stream = p.stream;
    cfg.attrs = attrs;
    cfg.numAttrs = 2;
    TLLM_CUDA_CHECK(cudaLaunchKernelEx(&cfg, kernel, p));
}
} // namespace

bool intraNodeReduceSupported(int rows, int hidden, int nranks, bool finalize)
{
    return launchShape(rows, hidden, nranks, finalize).valid;
}

void intraNodeReduce(IntraNodeReduceParams const& p)
{
    auto shape = launchShape(p.size / p.hidden_dim, p.hidden_dim, p.nranks, false);
    TLLM_CHECK_WITH_INFO(shape.valid, "Unsupported GLM Rubin IPC reduce launch");
    if (fp32Acc())
    {
        if (p.trigger_completion_at_end)
            launch(reduceKernel<4, true, true>, p, shape);
        else
            launch(reduceKernel<4, true, false>, p, shape);
    }
    else if (p.trigger_completion_at_end)
        launch(reduceKernel<4, false, true>, p, shape);
    else
        launch(reduceKernel<4, false, false>, p, shape);
}

void intraNodeMoeFinalize(IntraNodeReduceParams const& p)
{
    auto shape = launchShape(p.size / p.hidden_dim, p.hidden_dim, p.nranks, true);
    TLLM_CHECK_WITH_INFO(shape.valid && p.top_k >= 1 && p.top_k <= 8, "Unsupported GLM Rubin IPC finalize launch");
    launch(finalizeKernel<4>, p, shape);
}
} // namespace kernels::glm_moe_dsa

TRTLLM_NAMESPACE_END
