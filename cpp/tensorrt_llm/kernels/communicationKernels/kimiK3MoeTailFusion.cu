/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
#include "tensorrt_llm/common/config.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/envUtils.h"
#include "tensorrt_llm/common/reduceKernelUtils.cuh"
#include "tensorrt_llm/kernels/communicationKernels/kimiK3MoeTailFusion.h"
#include <algorithm>
#include <cuda_bf16.h>

TRTLLM_NAMESPACE_BEGIN

namespace kernels::kimi_k3_moe
{
namespace
{

using bf16_t = __nv_bfloat16;

// 8 bf16 elements per 16B access, matching ar_fusion::kElemsPerAccess<bf16>.
constexpr int kElemsPerAccess = 8;
constexpr int kMaxToken = 64;

// Same lamport workspace protocol as ar_fusion (allReduceFusionKernels.cu) /
// minimax_ar: 3-phase rotating buffers, device-side flag rotation, safe to
// interleave with allreduce_fusion_op calls on one stream.
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

//! MoE-tail head: oneshot lamport AR(latent)+RMSNorm concurrent with
//! RS(shared). One CTA per token (latent_dim/8 threads). Both cross-node
//! landings (AR slots + RS segment contributions) overlap inside one kernel,
//! replacing the baseline two-landing twoshot AR over the cat buffer.
//! Phase-buffer layout per rank: [NRanks AR slots of M*L][NRanks RS slots of
//! M*S], S = H/NRanks.
template <int NRanks, bool TriggerCompletionAtEnd>
__global__ void __launch_bounds__(512) kimi_k3_moe_tail_kernel(KimiK3MoeTailParams params)
{
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    int const t = blockIdx.x; // token id (grid = token_num)
    int const tid = threadIdx.x;
    int const L = params.latent_dim;
    int const H = params.hidden_dim;
    int const S = H / NRanks;
    int const M = params.token_num;
    int const l_acc = L / kElemsPerAccess; // accesses per latent row (= blockDim)
    int const h_acc = H / kElemsPerAccess; // accesses per shared row
    int const s_acc = S / kElemsPerAccess; // accesses per segment
    int const ar_size = M * L;             // elements per AR slot
    int const rs_size = M * S;             // elements per RS slot
    int const ar_total = NRanks * ar_size;
    float4 const clear_vec = get_neg_zero();

    cudaGridDependencySynchronize();
    if constexpr (!TriggerCompletionAtEnd)
    {
        // Early launch-completion: the dependent kernel becomes resident (and
        // can prefetch weights) while this one runs; its own grid sync still
        // fences the z/s_shard HBM outputs. Safe because the dependent blocks
        // in the grid sync rather than spinning.
        cudaTriggerProgrammaticLaunchCompletion();
    }
    LamportComm<NRanks> comm(params.workspace, params.rank);
    int const clear_access = comm.clear_size / kElemsPerAccess;

    // Hoisted per-slice norm gamma.
    float norm_w[kElemsPerAccess];
    {
        float4 const nw = reinterpret_cast<float4 const*>(params.norm_weight)[tid];
#pragma unroll
        for (int i = 0; i < kElemsPerAccess; ++i)
        {
            norm_w[i] = static_cast<float>(reinterpret_cast<bf16_t const*>(&nw)[i]);
        }
    }

    // Push 1: this token's latent slice to every peer's AR slot[rank].
    {
        alignas(16) float val[4];
        *reinterpret_cast<float4*>(val) = reinterpret_cast<float4 const*>(params.latent_in)[t * l_acc + tid];
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
            reinterpret_cast<float4*>(comm.data_bufs[r])[(params.rank * ar_size + t * L) / kElemsPerAccess + tid]
                = *reinterpret_cast<float4*>(val);
        }
    }
    // Push 2: this token's shared row, segment j to owner j's RS slot[rank].
    for (int a = tid; a < h_acc; a += blockDim.x)
    {
        alignas(16) float val[4];
        *reinterpret_cast<float4*>(val) = reinterpret_cast<float4 const*>(params.shared_in)[t * h_acc + a];
#pragma unroll
        for (int i = 0; i < 4; ++i)
        {
            if (is_neg_zero(val[i]))
            {
                val[i] = 0.f;
            }
        }
        int const owner = a / s_acc;
        int const c = a % s_acc;
        reinterpret_cast<float4*>(
            comm.data_bufs[owner])[(ar_total + params.rank * rs_size + t * S) / kElemsPerAccess + c]
            = *reinterpret_cast<float4*>(val);
    }
    // Legacy-protocol compatibility: clear whatever the op two rounds ago
    // asked for (zero iterations when every op consumer-clears).
    for (int idx = t * blockDim.x + tid; idx < clear_access; idx += gridDim.x * blockDim.x)
    {
        reinterpret_cast<float4*>(comm.clear_buf)[idx] = clear_vec;
    }

    // Poll AR: fp32 sum of NRanks latent slots, bf16 round, then RMSNorm
    // (fp32 variance over the rounded values -- baseline rmsnorm contract).
    // Consumer-side clear: restore the sentinel on every fragment read while
    // the line is still hot, so this round's region needs no clear pass.
    bf16_t mixed[kElemsPerAccess];
    float sq = 0.f;
    {
        float4 vals[NRanks];
        bool done = false;
        while (!done)
        {
            done = true;
#pragma unroll
            for (int r = 0; r < NRanks; ++r)
            {
                vals[r] = ld_global_volatile(&reinterpret_cast<float4*>(
                    comm.data_bufs[params.rank])[(r * ar_size + t * L) / kElemsPerAccess + tid]);
                done &= !is_neg_zero(vals[r]);
            }
        }
#pragma unroll
        for (int r = 0; r < NRanks; ++r)
        {
            reinterpret_cast<float4*>(comm.data_bufs[params.rank])[(r * ar_size + t * L) / kElemsPerAccess + tid]
                = clear_vec;
        }
#pragma unroll
        for (int i = 0; i < kElemsPerAccess; ++i)
        {
            float acc = static_cast<float>(reinterpret_cast<bf16_t*>(&vals[0])[i]);
#pragma unroll
            for (int r = 1; r < NRanks; ++r)
            {
                acc += static_cast<float>(reinterpret_cast<bf16_t*>(&vals[r])[i]);
            }
            mixed[i] = __float2bfloat16_rn(acc);
            float const m = static_cast<float>(mixed[i]);
            sq = fmaf(m, m, sq);
        }
    }
    // Poll RS: this rank's fully-reduced shared segment for token t.
    for (int c = tid; c < s_acc; c += blockDim.x)
    {
        float4 vals[NRanks];
        bool done = false;
        while (!done)
        {
            done = true;
#pragma unroll
            for (int r = 0; r < NRanks; ++r)
            {
                vals[r] = ld_global_volatile(&reinterpret_cast<float4*>(
                    comm.data_bufs[params.rank])[(ar_total + r * rs_size + t * S) / kElemsPerAccess + c]);
                done &= !is_neg_zero(vals[r]);
            }
        }
#pragma unroll
        for (int r = 0; r < NRanks; ++r)
        {
            reinterpret_cast<float4*>(
                comm.data_bufs[params.rank])[(ar_total + r * rs_size + t * S) / kElemsPerAccess + c]
                = clear_vec;
        }
        float4 packed;
#pragma unroll
        for (int i = 0; i < kElemsPerAccess; ++i)
        {
            float acc = static_cast<float>(reinterpret_cast<bf16_t*>(&vals[0])[i]);
#pragma unroll
            for (int r = 1; r < NRanks; ++r)
            {
                acc += static_cast<float>(reinterpret_cast<bf16_t*>(&vals[r])[i]);
            }
            reinterpret_cast<bf16_t*>(&packed)[i] = __float2bfloat16_rn(acc);
        }
        reinterpret_cast<float4*>(params.sshard_out)[t * s_acc + c] = packed;
    }

    // Norm reduce + z write.
    tensorrt_llm::common::blockReduceSumV2<float, 1>(&sq);
    __shared__ float s_sq;
    if (tid == 0)
    {
        s_sq = sq;
    }
    __syncthreads();
    float const rsig = rsqrtf(s_sq / static_cast<float>(L) + params.rms_eps);
    float4 z_packed;
#pragma unroll
    for (int i = 0; i < kElemsPerAccess; ++i)
    {
        reinterpret_cast<bf16_t*>(&z_packed)[i] = __float2bfloat16_rn(static_cast<float>(mixed[i]) * rsig * norm_w[i]);
    }
    reinterpret_cast<float4*>(params.z_out)[t * l_acc + tid] = z_packed;

    // Consumer-side clear above leaves this round's region pristine.
    comm.update(0);

    if constexpr (TriggerCompletionAtEnd)
    {
        cudaTriggerProgrammaticLaunchCompletion();
    }
#endif
}

template <int NRanks>
void launch_moe_tail(KimiK3MoeTailParams const& params)
{
    TLLM_CHECK(params.latent_dim % kElemsPerAccess == 0);
    TLLM_CHECK(params.hidden_dim % (NRanks * kElemsPerAccess) == 0);
    int const threads = params.latent_dim / kElemsPerAccess;
    TLLM_CHECK_WITH_INFO(threads >= 128 && threads <= 512, "kimi_k3_moe_tail: latent/8 must be in [128, 512]");
    TLLM_CHECK(params.token_num >= 1 && params.token_num <= kMaxToken);
    static int const SM = tensorrt_llm::common::getSMVersion();
    TLLM_CHECK_WITH_INFO(SM >= 90, "kimi_k3_moe_tail requires SM90+");

    cudaLaunchConfig_t cfg;
    cudaLaunchAttribute attribute[1];
    cfg.gridDim = params.token_num;
    cfg.blockDim = threads;
    cfg.dynamicSmemBytes = 0;
    cfg.stream = params.stream;
    attribute[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attribute[0].val.programmaticStreamSerializationAllowed = tensorrt_llm::common::getEnvEnablePDL() ? 1 : 0;
    cfg.attrs = attribute;
    cfg.numAttrs = 1;

    if (params.trigger_completion_at_end)
    {
        TLLM_CUDA_CHECK(cudaLaunchKernelEx(&cfg, kimi_k3_moe_tail_kernel<NRanks, true>, params));
    }
    else
    {
        TLLM_CUDA_CHECK(cudaLaunchKernelEx(&cfg, kimi_k3_moe_tail_kernel<NRanks, false>, params));
    }
}

//! Oneshot-lamport AllGather of per-rank up_proj output stripes, with the
//! RS'd shared segment folded into the push (shard_add) and/or a full-width
//! add on the poll side (shared_in). Flat grid, pure data movement.
template <int NRanks, bool TriggerCompletionAtEnd>
__global__ void __launch_bounds__(512) kimi_k3_stripe_ag_add_kernel(KimiK3StripeAgParams params)
{
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    int const tid = blockIdx.x * blockDim.x + threadIdx.x;
    int const nthreads = gridDim.x * blockDim.x;
    int const stripe_acc = params.stripe_dim / kElemsPerAccess;
    int const hidden_acc = params.hidden_dim / kElemsPerAccess;
    int const shard_acc_tot = params.size / kElemsPerAccess;
    int const token_num = params.size / params.stripe_dim;
    int const shared_ld_acc = params.shared_ld / kElemsPerAccess;
    float4 const clear_vec = get_neg_zero();

    cudaGridDependencySynchronize();
    if constexpr (!TriggerCompletionAtEnd)
    {
        cudaTriggerProgrammaticLaunchCompletion();
    }
    LamportComm<NRanks> comm(params.workspace, params.rank);
    int const clear_access = comm.clear_size / kElemsPerAccess;

    // Push this rank's stripe (with the optional folded addend) to
    // slot[rank] of every peer, then honor the legacy clear contract.
    for (int idx = tid; idx < shard_acc_tot; idx += nthreads)
    {
        alignas(16) float val[4];
        *reinterpret_cast<float4*>(val) = reinterpret_cast<float4 const*>(params.shard_in)[idx];
        if (params.shard_add != nullptr)
        {
            float4 const av = reinterpret_cast<float4 const*>(params.shard_add)[idx];
            float4 summed;
#pragma unroll
            for (int i = 0; i < kElemsPerAccess; ++i)
            {
                reinterpret_cast<bf16_t*>(&summed)[i]
                    = __float2bfloat16_rn(static_cast<float>(reinterpret_cast<bf16_t const*>(val)[i])
                        + static_cast<float>(reinterpret_cast<bf16_t const*>(&av)[i]));
            }
            *reinterpret_cast<float4*>(val) = summed;
        }
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
            reinterpret_cast<float4*>(comm.data_bufs[r])[params.rank * shard_acc_tot + idx]
                = *reinterpret_cast<float4*>(val);
        }
    }
    for (int idx = tid; idx < clear_access; idx += nthreads)
    {
        reinterpret_cast<float4*>(comm.clear_buf)[idx] = clear_vec;
    }

    // Poll each incoming stripe slice, restore the sentinel while the line
    // is hot (consumer-side clear), optionally add the full-width shared
    // operand (fp32 accumulate + one bf16 rounding, torch's bf16 opmath).
    int const out_acc_tot = token_num * hidden_acc;
    for (int idx = tid; idx < out_acc_tot; idx += nthreads)
    {
        int const t = idx / hidden_acc;
        int const a = idx % hidden_acc;
        int const src_rank = a / stripe_acc;
        int const c = a % stripe_acc;
        float4* slot
            = &reinterpret_cast<float4*>(comm.data_bufs[params.rank])[src_rank * shard_acc_tot + t * stripe_acc + c];
        float4 v = ld_global_volatile(slot);
        while (is_neg_zero(v))
        {
            v = ld_global_volatile(slot);
        }
        *slot = clear_vec;
        float4 out_packed = v;
        if (params.shared_in != nullptr)
        {
            float4 const sh
                = reinterpret_cast<float4 const*>(params.shared_in)[static_cast<int64_t>(t) * shared_ld_acc + a];
#pragma unroll
            for (int i = 0; i < kElemsPerAccess; ++i)
            {
                reinterpret_cast<bf16_t*>(&out_packed)[i]
                    = __float2bfloat16_rn(static_cast<float>(reinterpret_cast<bf16_t*>(&v)[i])
                        + static_cast<float>(reinterpret_cast<bf16_t const*>(&sh)[i]));
            }
        }
        reinterpret_cast<float4*>(params.out)[static_cast<int64_t>(t) * hidden_acc + a] = out_packed;
    }

    comm.update(0);

    if constexpr (TriggerCompletionAtEnd)
    {
        cudaTriggerProgrammaticLaunchCompletion();
    }
#endif
}

template <int NRanks>
void launch_stripe_ag(KimiK3StripeAgParams const& params)
{
    TLLM_CHECK(params.stripe_dim % kElemsPerAccess == 0);
    TLLM_CHECK(params.hidden_dim == params.stripe_dim * params.nranks);
    TLLM_CHECK(params.shared_in == nullptr || params.shared_ld % kElemsPerAccess == 0);
    TLLM_CHECK(params.size % params.stripe_dim == 0);
    int const token_num = params.size / params.stripe_dim;
    TLLM_CHECK(token_num >= 1 && token_num <= kMaxToken);
    static int const SM = tensorrt_llm::common::getSMVersion();
    TLLM_CHECK_WITH_INFO(SM >= 90, "kimi_k3_stripe_ag_add requires SM90+");

    int const out_acc_tot = token_num * (params.hidden_dim / kElemsPerAccess);
    int const grid_size = std::min(get_sm_count(), (out_acc_tot + 511) / 512);

    cudaLaunchConfig_t cfg;
    cudaLaunchAttribute attribute[1];
    cfg.gridDim = grid_size;
    cfg.blockDim = 512;
    cfg.dynamicSmemBytes = 0;
    cfg.stream = params.stream;
    attribute[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attribute[0].val.programmaticStreamSerializationAllowed = tensorrt_llm::common::getEnvEnablePDL() ? 1 : 0;
    cfg.attrs = attribute;
    cfg.numAttrs = 1;

    if (params.trigger_completion_at_end)
    {
        TLLM_CUDA_CHECK(cudaLaunchKernelEx(&cfg, kimi_k3_stripe_ag_add_kernel<NRanks, true>, params));
    }
    else
    {
        TLLM_CUDA_CHECK(cudaLaunchKernelEx(&cfg, kimi_k3_stripe_ag_add_kernel<NRanks, false>, params));
    }
}

} // namespace

void kimi_k3_moe_tail_op(KimiK3MoeTailParams const& params)
{
    switch (params.nranks)
    {
    case 2: return launch_moe_tail<2>(params);
    case 4: return launch_moe_tail<4>(params);
    case 8: return launch_moe_tail<8>(params);
    default: TLLM_CHECK_WITH_INFO(false, "kimi_k3_moe_tail: unsupported nranks (2/4/8 only)");
    }
}

void kimi_k3_stripe_ag_add_op(KimiK3StripeAgParams const& params)
{
    switch (params.nranks)
    {
    case 2: return launch_stripe_ag<2>(params);
    case 4: return launch_stripe_ag<4>(params);
    case 8: return launch_stripe_ag<8>(params);
    default: TLLM_CHECK_WITH_INFO(false, "kimi_k3_stripe_ag_add: unsupported nranks (2/4/8 only)");
    }
}

} // namespace kernels::kimi_k3_moe

TRTLLM_NAMESPACE_END
