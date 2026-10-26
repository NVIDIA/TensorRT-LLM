/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * MiniMax-M3.1 TP2 decode MoE tail in one launch, bit for bit the chain
 *   m3_moe_finalize (fp32 fma over the top-k slots in order, round to bf16, + shared in fp32, round)
 *   -> allreduce fusion, one-shot Lamport, pattern kARResidualRMSNorm (residual_out + norm_out), Fp32Acc
 *   -> mxfp8_quantize(norm_out, swizzled) (cvt_warp_fp16_to_mxfp8, K32 UE8M0 scales, pad rows' scales 0).
 * The kernel is allreduce_fusion_kernel_oneshot_lamport (communicationKernels/allReduceFusionKernels.cu) with (1) the
 * allreduce_in load replaced by the finalize of the same 8 columns and (2) the MXFP8 quantize of the normed 8 columns
 * in the epilogue; the launch configuration (cluster, block, grid) is that file's launcher, so the RMSNorm block /
 * cluster reduction order is unchanged.
 * Op: trtllm::minimax_m31_s2_moe_tail(fc2, scales, perm, shared, residual, norm_weight, workspace, rank, nranks, eps,
 *     trigger_completion_at_end) -> (norm_out bf16 [T, H], residual_out bf16 [T, H], mxfp8 [T, H], sf u8 swizzled)
 * workspace = the AllReduce module's lamport workspace (the stock op's).
 */

#include "tensorrt_llm/kernels/minimaxM31/s2MoeTailSm107.h"

#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/envUtils.h"
#include "tensorrt_llm/common/reduceKernelUtils.cuh"
#include "tensorrt_llm/kernels/quantization.cuh"

#include <cooperative_groups.h>
#include <cuda_bf16.h>

#include <algorithm>
#include <cstdlib>

namespace cg = cooperative_groups;

TRTLLM_NAMESPACE_BEGIN

namespace kernels::minimaxM31Sm107::s2tail
{

using tensorrt_llm::kernels::PackedVec;
using tensorrt_llm::QuantizationSFLayout;

template <int NRanks>
struct LamportComm
{
    __device__ __forceinline__ LamportComm(void** workspace, int rank)
    {
        counter_ptr = &reinterpret_cast<int*>(workspace[NRanks * 3])[0];
        flag_ptr = &reinterpret_cast<int*>(workspace[NRanks * 3])[2];
        clear_ptr = &reinterpret_cast<int64_t*>(workspace[NRanks * 3 + 1])[0];
        flag_value = *flag_ptr;
        auto comm_size = reinterpret_cast<int64_t*>(workspace[NRanks * 3 + 1])[1];
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

    __device__ __forceinline__ void update(int64_t new_clear_size)
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
    int64_t* clear_ptr;
    uint8_t* data_bufs[NRanks];
    uint8_t* clear_buf;
    int64_t clear_size;
    int flag_value;
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

__device__ __forceinline__ float4 add128(float4 const& a, float4 const& b)
{
    float4 c;
#pragma unroll
    for (int i = 0; i < kElems; ++i)
    {
        reinterpret_cast<__nv_bfloat16*>(&c)[i]
            = reinterpret_cast<__nv_bfloat16 const*>(&a)[i] + reinterpret_cast<__nv_bfloat16 const*>(&b)[i];
    }
    return c;
}

template <int NRanks, bool Fp32Acc>
__device__ __forceinline__ float4 allreduce_sum(float4* vals)
{
    if constexpr (Fp32Acc)
    {
        float acc_f32[kElems];
#pragma unroll
        for (int i = 0; i < kElems; ++i)
        {
            acc_f32[i] = static_cast<float>(reinterpret_cast<__nv_bfloat16*>(&vals[0])[i]);
        }
#pragma unroll
        for (int r = 1; r < NRanks; ++r)
        {
#pragma unroll
            for (int i = 0; i < kElems; ++i)
            {
                acc_f32[i] += static_cast<float>(reinterpret_cast<__nv_bfloat16*>(&vals[r])[i]);
            }
        }
        float4 acc;
#pragma unroll
        for (int i = 0; i < kElems; ++i)
        {
            reinterpret_cast<__nv_bfloat16*>(&acc)[i] = static_cast<__nv_bfloat16>(acc_f32[i]);
        }
        return acc;
    }
    else
    {
        float4 acc = vals[0];
#pragma unroll
        for (int r = 1; r < NRanks; ++r)
        {
            acc = add128(acc, vals[r]);
        }
        return acc;
    }
}

// The finalize of 8 columns of one token: m3_moe_finalize's per-element math (fp32 fma over the slots in order, a
// dropped slot leaves the accumulator, bf16 round, + shared in fp32, bf16 round).
template <typename ScaleT>
__device__ __forceinline__ float4 finalize8(Params const& p, int token, int col8)
{
    float acc[kElems];
#pragma unroll
    for (int i = 0; i < kElems; ++i)
    {
        acc[i] = 0.f;
    }
    for (int k = 0; k < p.top_k; ++k)
    {
        int const slot = p.perm[token * p.top_k + k];
        float const w = static_cast<float>(reinterpret_cast<ScaleT const*>(p.scales)[token * p.top_k + k]);
        if (slot >= 0)
        {
            float4 x4 = reinterpret_cast<float4 const*>(p.fc2 + static_cast<int64_t>(slot) * p.ld_fc2)[col8];
#pragma unroll
            for (int i = 0; i < kElems; ++i)
            {
                acc[i] = __fmaf_rn(w, static_cast<float>(reinterpret_cast<__nv_bfloat16*>(&x4)[i]), acc[i]);
            }
        }
    }
    float4 s4 = reinterpret_cast<float4 const*>(p.shared + static_cast<int64_t>(token) * p.hidden_dim)[col8];
    float4 out;
#pragma unroll
    for (int i = 0; i < kElems; ++i)
    {
        float const r = static_cast<float>(__float2bfloat16_rn(acc[i]));
        float const s = static_cast<float>(reinterpret_cast<__nv_bfloat16*>(&s4)[i]);
        reinterpret_cast<__nv_bfloat16*>(&out)[i] = __float2bfloat16_rn(s + r);
    }
    return out;
}

// allreduce_fusion's FusedOp::rms_norm verbatim (block reduce, then the cluster's partial sums in rank order).
__device__ __forceinline__ float4 rms_norm(float4 const& residual, float4 const& gamma, int hidden_dim, float eps)
{
    __shared__ float s_val;
    float4 norm_out;
    float acc = 0.f;
#pragma unroll
    for (int i = 0; i < kElems; ++i)
    {
        float v = static_cast<float>(reinterpret_cast<__nv_bfloat16 const*>(&residual)[i]);
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
        s_val = rsqrtf(acc / hidden_dim + eps);
    }
    __syncthreads();
#pragma unroll
    for (int i = 0; i < kElems; ++i)
    {
        reinterpret_cast<__nv_bfloat16*>(&norm_out)[i]
            = static_cast<__nv_bfloat16>(static_cast<float>(reinterpret_cast<__nv_bfloat16 const*>(&residual)[i])
                * s_val * static_cast<float>(reinterpret_cast<__nv_bfloat16 const*>(&gamma)[i]));
    }
    return norm_out;
}

template <typename ScaleT, int NRanks, bool Fp32Acc, bool TriggerCompletionAtEnd>
__global__ void __launch_bounds__(1024) s2_moe_tail_kernel(Params params)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    cg::cluster_group cluster = cg::this_cluster();
    cg::grid_group grid = cg::this_grid();
    int const token_id = grid.cluster_rank();
    int const access_id_in_token = cluster.thread_rank();
    int const token_stride = grid.num_clusters();
    int const access_per_token = params.hidden_dim / kElems;
    int const access_id = token_id * access_per_token + access_id_in_token;
    int const access_stride = token_stride * access_per_token;
    int const tot_access = params.size / kElems;
    int const tokens = params.size / params.hidden_dim;
    float4 const clear_vec = get_neg_zero();
    float4 const gamma = reinterpret_cast<float4 const*>(params.rms_gamma)[access_id_in_token];

    cudaGridDependencySynchronize();
    if constexpr (!TriggerCompletionAtEnd)
    {
        cudaTriggerProgrammaticLaunchCompletion();
    }
    LamportComm<NRanks> comm(params.workspace, params.rank);
    int const clear_access = comm.clear_size / kElems;

    for (int idx = access_id, tidx = token_id; idx < tot_access; idx += access_stride, tidx += token_stride)
    {
        alignas(16) float val[4];
        *reinterpret_cast<float4*>(val) = finalize8<ScaleT>(params, tidx, access_id_in_token);
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
            reinterpret_cast<float4*>(comm.data_bufs[r])[params.rank * tot_access + idx]
                = *reinterpret_cast<float4*>(val);
        }
    }
    for (int idx = access_id; idx < clear_access; idx += access_stride)
    {
        reinterpret_cast<float4*>(comm.clear_buf)[idx] = clear_vec;
    }

    for (int idx = access_id, tidx = token_id; idx < tot_access; idx += access_stride, tidx += token_stride)
    {
        float4 const residual = reinterpret_cast<float4 const*>(params.residual_in)[idx];
        float4 vals[NRanks];
        bool done = false;
        while (!done)
        {
            done = true;
#pragma unroll
            for (int r = 0; r < NRanks; ++r)
            {
                vals[r]
                    = ld_global_volatile(&reinterpret_cast<float4*>(comm.data_bufs[params.rank])[r * tot_access + idx]);
                done &= !is_neg_zero(vals[r]);
            }
        }
        float4 v = allreduce_sum<NRanks, Fp32Acc>(vals);
        v = add128(v, residual);
        reinterpret_cast<float4*>(params.residual_out)[idx] = v;
        v = rms_norm(v, gamma, params.hidden_dim, params.rms_eps);
        reinterpret_cast<float4*>(params.norm_out)[idx] = v;
        // MXFP8 of the normed 8 columns: mxfp8_quantize's per-element math (4 lanes share a K32 scale).
        PackedVec<__nv_bfloat16> pack = *reinterpret_cast<PackedVec<__nv_bfloat16>*>(&v);
        auto sf_out
            = tensorrt_llm::kernels::cvt_quant_get_sf_out_offset<uint32_t, 4>(std::nullopt, tidx, access_id_in_token,
                std::optional<int>(tokens), params.hidden_dim / 32, params.scale_out, QuantizationSFLayout::SWIZZLED);
        params.quant_out[idx] = tensorrt_llm::kernels::cvt_warp_fp16_to_mxfp8<__nv_bfloat16, 32>(pack, sf_out);
    }
    // The swizzled scales of the pad rows up to the 128-row tile are 0 (mxfp8_quantize's).
    int const padded = (tokens + 127) / 128 * 128;
    int const sf_cols = params.hidden_dim / 32;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (padded - tokens) * sf_cols; i += gridDim.x * blockDim.x)
    {
        int const row = tokens + i / sf_cols;
        int const col = i % sf_cols;
        int64_t const off = tensorrt_llm::kernels::get_sf_out_offset_128x4(
            std::nullopt, row, col, std::optional<int>(tokens), sf_cols);
        reinterpret_cast<uint8_t*>(params.scale_out)[off] = 0;
    }

    comm.update(static_cast<int64_t>(params.size) * NRanks);
    if constexpr (TriggerCompletionAtEnd)
    {
        __threadfence();
        cudaTriggerProgrammaticLaunchCompletion();
    }
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

bool use_fp32_acc()
{
    static char* fp32_acc = std::getenv("ALL_REDUCE_FUSION_KERNEL_ACC_FP32");
    return fp32_acc == nullptr || fp32_acc[0] != '0';
}

template <typename ScaleT, int NRanks, bool Fp32Acc, bool Trigger>
void launch(Params const& params, cudaStream_t stream)
{
    // allreduce_fusion_kernel_launcher (oneshot) verbatim.
    int token_num = params.size / params.hidden_dim;
    int cluster_num = token_num;
    int threads_per_token = params.hidden_dim / kElems;
    int cluster_size = 8;
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
    TLLM_CHECK(threads_per_block <= 1024 && cluster_size > 0);
    int grid_size = (std::min(sm_count, cluster_num * cluster_size) / cluster_size) * cluster_size;
    cudaLaunchConfig_t cfg;
    cudaLaunchAttribute attribute[2];
    cfg.gridDim = grid_size;
    cfg.blockDim = threads_per_block;
    cfg.dynamicSmemBytes = 0;
    cfg.stream = stream;
    attribute[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attribute[0].val.programmaticStreamSerializationAllowed = tensorrt_llm::common::getEnvEnablePDL() ? 1 : 0;
    attribute[1].id = cudaLaunchAttributeClusterDimension;
    attribute[1].val.clusterDim.x = cluster_size;
    attribute[1].val.clusterDim.y = 1;
    attribute[1].val.clusterDim.z = 1;
    cfg.attrs = attribute;
    cfg.numAttrs = 2;
    TLLM_CUDA_CHECK(cudaLaunchKernelEx(&cfg, s2_moe_tail_kernel<ScaleT, NRanks, Fp32Acc, Trigger>, params));
}

template <typename ScaleT, int NRanks>
void dispatch_acc(Params const& p, bool trigger, cudaStream_t s)
{
    bool const fp32 = use_fp32_acc();
    if (fp32)
    {
        trigger ? launch<ScaleT, NRanks, true, true>(p, s) : launch<ScaleT, NRanks, true, false>(p, s);
    }
    else
    {
        trigger ? launch<ScaleT, NRanks, false, true>(p, s) : launch<ScaleT, NRanks, false, false>(p, s);
    }
}

void invokeS2MoeTail(Params const& params, bool scalesBf16, bool triggerCompletionAtEnd, cudaStream_t stream)
{
    TLLM_CHECK_WITH_INFO(params.nranks == 2, "TP2 only");
    if (scalesBf16)
    {
        dispatch_acc<__nv_bfloat16, 2>(params, triggerCompletionAtEnd, stream);
    }
    else
    {
        dispatch_acc<float, 2>(params, triggerCompletionAtEnd, stream);
    }
}

} // namespace kernels::minimaxM31Sm107::s2tail

TRTLLM_NAMESPACE_END
