/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Device code copied from TRT-LLM c972d6f162 cpp/tensorrt_llm/kernels/communicationKernels/allReduceFusionKernels.cu
// (md5 546e8c75) and common/reduceKernelUtils.cuh: LamportComm, FusedOp<kARResidualRMSNorm, bf16>, allreduce_sum
// (Fp32Acc), IndexHelper, blockReduceSumV2, with the push loop removed (the producer pre-pushed) and the MXFP8 output
// of the norm (quantization.cuh cvt_quant_get_sf_out_offset<uint32_t, 4> SWIZZLED + cvt_warp_fp16_to_mxfp8<bf16, 32>,
// as quantize_mxfp8_rows). Same device code as the shipped libm31_f3_ar.so.
#include "tensorrt_llm/kernels/minimaxM31/f3ArPrepushedSm107.h"
#include "tensorrt_llm/kernels/quantization.cuh"

#include <algorithm>
#include <cooperative_groups.h>
#include <cuda_bf16.h>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{
namespace minimaxM31
{
namespace
{
namespace cg = cooperative_groups;
#define FINAL_MASK 0xffffffff

namespace f3
{
using DType = __nv_bfloat16;
constexpr int NRanks = 2;
constexpr int kElemsPerAccess = 8;

using Params = F3ArParams;

template <typename T, int NUM>
__inline__ __device__ T warpReduceSumV2(T* val)
{
#pragma unroll
    for (int i = 0; i < NUM; i++)
    {
#pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1)
            val[i] += __shfl_xor_sync(FINAL_MASK, val[i], mask, 32);
    }
    return (T) (0.0f);
}

template <typename T, int NUM>
__inline__ __device__ T blockReduceSumV2(T* val)
{
    static __shared__ T shared[NUM][33];
    int lane = threadIdx.x & 0x1f;
    int wid = threadIdx.x >> 5;

    warpReduceSumV2<T, NUM>(val);

    if (lane == 0)
    {
#pragma unroll
        for (int i = 0; i < NUM; i++)
        {
            shared[i][wid] = val[i];
        }
    }

    __syncthreads();

    bool is_mask = threadIdx.x < (blockDim.x / 32.f);
#pragma unroll
    for (int i = 0; i < NUM; i++)
    {
        val[i] = is_mask ? shared[i][lane] : (T) (0.0f);
    }
    warpReduceSumV2<T, NUM>(val);
    return (T) 0.0f;
}

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

template <typename PackedType>
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

// FusedOp<kARResidualRMSNorm, bf16>: residual add -> residual_out -> rms_norm -> norm_out.
class FusedOp
{
    static constexpr int kMathCount = sizeof(float4) / sizeof(DType);

public:
    __device__ __forceinline__ FusedOp(Params const& params, int access_id, int access_id_in_token)
        : m_params(params)
        , m_access_id(access_id)
        , m_access_id_in_token(access_id_in_token)
    {
        m_gamma_val = reinterpret_cast<float4*>(params.rms_gamma)[m_access_id_in_token];
        m_residual_val = reinterpret_cast<float4*>(params.residual_in)[m_access_id];
    }

    __device__ __forceinline__ void update(int access_id)
    {
        if (m_access_id != access_id)
        {
            m_access_id = access_id;
            m_residual_val = reinterpret_cast<float4*>(m_params.residual_in)[m_access_id];
        }
    }

    __device__ __forceinline__ void operator()(float4 val, int token_id)
    {
        val = add128(val, m_residual_val);
        reinterpret_cast<float4*>(m_params.residual_out)[m_access_id] = val;
        val = rms_norm(val, m_gamma_val);
        reinterpret_cast<float4*>(m_params.norm_out)[m_access_id] = val;
        if (m_params.mx_out != nullptr)
        {
            using PV = tensorrt_llm::kernels::PackedVec<DType>;
            PV pv = *reinterpret_cast<PV const*>(&val);
            auto sf_out = tensorrt_llm::kernels::cvt_quant_get_sf_out_offset<uint32_t, 4>(std::optional<int>(0),
                token_id, m_access_id_in_token, std::optional<int>(m_params.size / m_params.hidden_dim),
                m_params.hidden_dim / 32, reinterpret_cast<uint32_t*>(m_params.mx_sf),
                tensorrt_llm::QuantizationSFLayout::SWIZZLED);
            reinterpret_cast<uint64_t*>(m_params.mx_out)[m_access_id]
                = tensorrt_llm::kernels::cvt_warp_fp16_to_mxfp8<DType, 32>(pv, sf_out);
        }
    }

    // The swizzled scale rows past the last token, up to the 128-row tile, are 0 (as quantize_mxfp8_rows writes).
    __device__ __forceinline__ void pad_sf(int token_id)
    {
        auto sf_out = tensorrt_llm::kernels::cvt_quant_get_sf_out_offset<uint32_t, 4>(std::optional<int>(0), token_id,
            m_access_id_in_token, std::optional<int>(m_params.size / m_params.hidden_dim), m_params.hidden_dim / 32,
            reinterpret_cast<uint32_t*>(m_params.mx_sf), tensorrt_llm::QuantizationSFLayout::SWIZZLED);
        tensorrt_llm::kernels::cvt_store_replicated_sf<1>(sf_out, 0x00);
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
        blockReduceSumV2<float, 1>(&acc);
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
    Params const& m_params;
    int m_access_id;
    int m_access_id_in_token;
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

__device__ __forceinline__ float4 allreduce_sum(float4* vals)
{
    float acc_f32[kElemsPerAccess];
#pragma unroll
    for (int i = 0; i < kElemsPerAccess; ++i)
    {
        acc_f32[i] = static_cast<float>(reinterpret_cast<DType*>(&vals[0])[i]);
    }
#pragma unroll
    for (int r = 1; r < NRanks; ++r)
    {
#pragma unroll
        for (int i = 0; i < kElemsPerAccess; ++i)
        {
            acc_f32[i] += static_cast<float>(reinterpret_cast<DType*>(&vals[r])[i]);
        }
    }
    float4 acc;
#pragma unroll
    for (int i = 0; i < kElemsPerAccess; ++i)
    {
        reinterpret_cast<DType*>(&acc)[i] = static_cast<DType>(acc_f32[i]);
    }
    return acc;
}

struct IndexHelper
{
    __device__ __forceinline__ IndexHelper(Params const& params)
    {
        cg::cluster_group cluster = cg::this_cluster();
        cg::grid_group grid = cg::this_grid();
        token_id = grid.cluster_rank();
        access_id_in_token = cluster.thread_rank();
        token_stride = grid.num_clusters();
        access_id = token_id * params.hidden_dim / kElemsPerAccess + access_id_in_token;
        access_stride = token_stride * params.hidden_dim / kElemsPerAccess;
        tot_access = params.size / kElemsPerAccess;
    }

    int token_id;
    int access_id_in_token;
    int token_stride;
    int access_id;
    int access_stride;
    int tot_access;
};

// allreduce_fusion_kernel_oneshot_lamport<kARResidualRMSNorm, bf16, 2, Fp32Acc true, TriggerCompletionAtEnd> without
// the push loop: the data slots were filled by the producer.
template <bool TriggerCompletionAtEnd>
__global__ void __launch_bounds__(1024) f3_ar_prepushed(Params params)
{
    IndexHelper index_helper(params);
    int token_id = index_helper.token_id;
    int access_id_in_token = index_helper.access_id_in_token;
    int token_stride = index_helper.token_stride;
    int access_id = index_helper.access_id;
    int access_stride = index_helper.access_stride;
    int tot_access = index_helper.tot_access;
    float4 clear_vec = get_neg_zero();
    FusedOp fused_op(params, access_id, access_id_in_token);

    cudaGridDependencySynchronize();
    if constexpr (!TriggerCompletionAtEnd)
    {
        cudaTriggerProgrammaticLaunchCompletion();
    }
    LamportComm comm(params.workspace, params.rank);
    int clear_access = comm.clear_size / kElemsPerAccess;

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
                vals[r]
                    = ld_global_volatile(&reinterpret_cast<float4*>(comm.data_bufs[params.rank])[r * tot_access + idx]);
                done &= !is_neg_zero(vals[r]);
            }
        }
        float4 sum_val = allreduce_sum(vals);
        fused_op(sum_val, tidx);
    }
    if (params.mx_out != nullptr)
    {
        int const token_num = params.size / params.hidden_dim;
        int const padded = (token_num + 127) / 128 * 128;
        for (int t = token_num + token_id; t < padded; t += token_stride)
        {
            fused_op.pad_sf(t);
        }
    }

    comm.update(params.size * NRanks);

    if constexpr (TriggerCompletionAtEnd)
    {
        cudaTriggerProgrammaticLaunchCompletion();
    }
}

int smCount()
{
    static int n = 0;
    if (n == 0)
    {
        int dev;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, dev);
    }
    return n;
}

} // namespace f3
} // namespace

cudaError_t invokeF3ArPrepushed(
    F3ArParams const& params, bool triggerCompletionAtEnd, bool enablePdl, cudaStream_t stream)
{
    using namespace f3;
    Params const p = params;
    int token_num = p.size / p.hidden_dim;
    int cluster_num = token_num;
    int threads_per_token = p.hidden_dim / kElemsPerAccess;
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
    int sms = smCount();
    while (cluster_num * cluster_size > sms && cluster_size > 1 && threads_per_block <= 512)
    {
        threads_per_block *= 2;
        cluster_size /= 2;
    }
    if (threads_per_block > 1024 || cluster_size <= 0)
    {
        return cudaErrorInvalidConfiguration;
    }
    int grid_size = (std::min(sms, cluster_num * cluster_size) / cluster_size) * cluster_size;
    cudaLaunchConfig_t cfg = {};
    cudaLaunchAttribute attr[2];
    cfg.gridDim = grid_size;
    cfg.blockDim = threads_per_block;
    cfg.dynamicSmemBytes = 0;
    cfg.stream = stream;
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = enablePdl ? 1 : 0;
    attr[1].id = cudaLaunchAttributeClusterDimension;
    attr[1].val.clusterDim.x = cluster_size;
    attr[1].val.clusterDim.y = 1;
    attr[1].val.clusterDim.z = 1;
    cfg.attrs = attr;
    cfg.numAttrs = 2;
    return triggerCompletionAtEnd ? cudaLaunchKernelEx(&cfg, f3_ar_prepushed<true>, p)
                                  : cudaLaunchKernelEx(&cfg, f3_ar_prepushed<false>, p);
}

} // namespace minimaxM31
} // namespace kernels

TRTLLM_NAMESPACE_END
