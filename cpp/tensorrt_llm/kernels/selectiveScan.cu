/*
 * Adapted from https://github.com/state-spaces/mamba/blob/main/csrc/selective_scan/selective_scan_fwd_kernel.cuh
 * Copyright (c) 2023, Tri Dao.
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
 *
 * Not a contribution
 * Changes made by NVIDIA CORPORATION & AFFILIATES or otherwise documented as
 * NVIDIA-proprietary are not a contribution and subject to the following terms and conditions:
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include <cuda_runtime_api.h>
#ifdef ENABLE_FP8
#include <cuda_fp8.h>
#endif

#include <cub/block/block_load.cuh>
#include <cub/block/block_scan.cuh>
#include <cub/block/block_store.cuh>

#include "selectiveScan.h"
#include "selectiveScanCommon.h"

namespace tensorrt_llm
{
namespace kernels
{

template <int kNThreads_, int kNItems_, int kNRows_, bool kIsEvenLen_, bool kIsVariableB_, bool kIsVariableC_,
    bool kHasZ_, typename input_t_, typename weight_t_>
struct Selective_Scan_fwd_kernel_traits
{
    static_assert(kNItems_ % 4 == 0);
    using input_t = input_t_;
    using weight_t = weight_t_;
    static constexpr int kNThreads = kNThreads_;
    // Setting MinBlocksPerMP to be 3 (instead of 2) for 128 threads improves occupancy.
    static constexpr int kMinBlocks = kNThreads < 128 ? 5 : 3;
    static constexpr int kNItems = kNItems_;
    static constexpr int kNRows = kNRows_;
    static constexpr int kNBytes = sizeof(input_t);
    static_assert(kNBytes == 2 || kNBytes == 4);
    static constexpr int kNElts = kNBytes == 4 ? 4 : std::min(8, kNItems);
    static_assert(kNItems % kNElts == 0);
    static constexpr int kNLoads = kNItems / kNElts;
    static constexpr bool kIsEvenLen = kIsEvenLen_;
    static constexpr bool kIsVariableB = kIsVariableB_;
    static constexpr bool kIsVariableC = kIsVariableC_;
    static constexpr bool kHasZ = kHasZ_;

    static constexpr bool kDirectIO = kIsEvenLen && kNLoads == 1;

    using vec_t = typename BytesToType<kNBytes * kNElts>::Type;
    using scan_t = float2;
    using scan_t_s = float;
    using BlockLoadT = cub::BlockLoad<input_t, kNThreads, kNItems, cub::BLOCK_LOAD_WARP_TRANSPOSE>;
    using BlockLoadVecT = cub::BlockLoad<vec_t, kNThreads, kNLoads,
        !kDirectIO ? cub::BLOCK_LOAD_WARP_TRANSPOSE : cub::BLOCK_LOAD_DIRECT>;
    using BlockLoadWeightT = cub::BlockLoad<input_t, kNThreads, kNItems, cub::BLOCK_LOAD_WARP_TRANSPOSE>;
    using BlockLoadWeightVecT = cub::BlockLoad<vec_t, kNThreads, kNLoads,
        !kDirectIO ? cub::BLOCK_LOAD_WARP_TRANSPOSE : cub::BLOCK_LOAD_DIRECT>;
    using BlockStoreT = cub::BlockStore<input_t, kNThreads, kNItems, cub::BLOCK_STORE_WARP_TRANSPOSE>;
    using BlockStoreVecT = cub::BlockStore<vec_t, kNThreads, kNLoads,
        !kDirectIO ? cub::BLOCK_STORE_WARP_TRANSPOSE : cub::BLOCK_STORE_DIRECT>;
    // using BlockScanT = cub::BlockScan<scan_t, kNThreads, cub::BLOCK_SCAN_RAKING_MEMOIZE>;
    // using BlockScanT = cub::BlockScan<scan_t, kNThreads, cub::BLOCK_SCAN_RAKING>;
    using BlockScanT = cub::BlockScan<scan_t, kNThreads, cub::BLOCK_SCAN_WARP_SCANS>;
    static constexpr int kSmemIOSize
        = std::max({sizeof(typename BlockLoadT::TempStorage), sizeof(typename BlockLoadVecT::TempStorage),
            (int(kIsVariableB) + int(kIsVariableC)) * sizeof(typename BlockLoadWeightT::TempStorage),
            (int(kIsVariableB) + int(kIsVariableC)) * sizeof(typename BlockLoadWeightVecT::TempStorage),
            sizeof(typename BlockStoreT::TempStorage), sizeof(typename BlockStoreVecT::TempStorage)});
    static constexpr int kSmemSize = kSmemIOSize + sizeof(typename BlockScanT::TempStorage);
};

template <typename Ktraits>
__global__ __launch_bounds__(Ktraits::kNThreads, Ktraits::kMinBlocks) void selective_scan_fwd_kernel(
    SSMParamsBase params)
{
    constexpr bool kIsVariableB = Ktraits::kIsVariableB;
    constexpr bool kIsVariableC = Ktraits::kIsVariableC;
    constexpr bool kHasZ = Ktraits::kHasZ;
    constexpr int kNThreads = Ktraits::kNThreads;
    constexpr int kNItems = Ktraits::kNItems;
    constexpr int kNRows = Ktraits::kNRows;
    constexpr bool kDirectIO = Ktraits::kDirectIO;
    using input_t = typename Ktraits::input_t;
    using weight_t = typename Ktraits::weight_t;
    using scan_t = typename Ktraits::scan_t;
    using scan_t_s = typename Ktraits::scan_t_s;

    // Shared memory.
    extern __shared__ char smem_[];
    // cast to lvalue reference of expected type
    // char *smem_loadstorescan = smem_ + 2 * MAX_DSTATE * sizeof(weight_t);
    // auto& smem_load = reinterpret_cast<typename BlockLoadT::TempStorage&>(smem_ + 2 * MAX_DSTATE * sizeof(weight_t));
    // auto& smem_load = reinterpret_cast<typename BlockLoadT::TempStorage&>(smem_loadstorescan);
    auto& smem_load = reinterpret_cast<typename Ktraits::BlockLoadT::TempStorage&>(smem_);
    auto& smem_load_weight = reinterpret_cast<typename Ktraits::BlockLoadWeightT::TempStorage&>(smem_);
    auto& smem_load_weight1 = *reinterpret_cast<typename Ktraits::BlockLoadWeightT::TempStorage*>(
        smem_ + sizeof(typename Ktraits::BlockLoadWeightT::TempStorage));
    auto& smem_store = reinterpret_cast<typename Ktraits::BlockStoreT::TempStorage&>(smem_);
    auto& smem_scan = *reinterpret_cast<typename Ktraits::BlockScanT::TempStorage*>(smem_ + Ktraits::kSmemIOSize);
    // weight_t *smem_a = reinterpret_cast<weight_t *>(smem_ + smem_loadstorescan_size);
    // weight_t *smem_bc = reinterpret_cast<weight_t *>(smem_a + MAX_DSTATE);
    scan_t* smem_running_prefix = reinterpret_cast<scan_t*>(smem_ + Ktraits::kSmemSize);

    const int batch_id = blockIdx.x;
    const int dim_id = blockIdx.y;
    const int group_id = dim_id / (params.dim_ngroups_ratio);
    input_t* u = reinterpret_cast<input_t*>(params.u_ptr) + batch_id * params.u_batch_stride
        + dim_id * kNRows * params.u_d_stride;
    input_t* delta = reinterpret_cast<input_t*>(params.delta_ptr) + batch_id * params.delta_batch_stride
        + dim_id * kNRows * params.delta_d_stride;
    weight_t* A = reinterpret_cast<weight_t*>(params.A_ptr) + dim_id * kNRows * params.A_d_stride;
    weight_t* B = reinterpret_cast<weight_t*>(params.B_ptr) + dim_id * kNRows * params.B_d_stride;
    input_t* Bvar = reinterpret_cast<input_t*>(params.B_ptr) + batch_id * params.B_batch_stride
        + group_id * params.B_group_stride;
    weight_t* C = reinterpret_cast<weight_t*>(params.C_ptr) + dim_id * kNRows * params.C_d_stride;
    input_t* Cvar = reinterpret_cast<input_t*>(params.C_ptr) + batch_id * params.C_batch_stride
        + group_id * params.C_group_stride;
    scan_t_s* x = reinterpret_cast<scan_t_s*>(params.x_ptr) + (batch_id * params.dim + dim_id * kNRows) * params.dstate;

    float D_val[kNRows] = {0};
    if (params.D_ptr != nullptr)
    {
#pragma unroll
        for (int r = 0; r < kNRows; ++r)
        {
            D_val[r] = reinterpret_cast<float*>(params.D_ptr)[dim_id * kNRows + r];
        }
    }
    float delta_bias[kNRows] = {0};
    if (params.delta_bias_ptr != nullptr)
    {
#pragma unroll
        for (int r = 0; r < kNRows; ++r)
        {
            delta_bias[r] = reinterpret_cast<float*>(params.delta_bias_ptr)[dim_id * kNRows + r];
        }
    }

    // for (int state_idx = threadIdx.x; state_idx < params.dstate; state_idx += blockDim.x) {
    //     smem_a[state_idx] = A[state_idx * params.A_dstate_stride];
    //     smem_bc[state_idx] = B[state_idx * params.B_dstate_stride] * C[state_idx * params.C_dstate_stride];
    // }

    constexpr int kChunkSize = kNThreads * kNItems;
    for (int chunk = 0; chunk < params.n_chunks; ++chunk)
    {
        input_t u_vals[kNRows][kNItems], delta_vals_load[kNRows][kNItems];
        __syncthreads();
#pragma unroll
        for (int r = 0; r < kNRows; ++r)
        {
            if constexpr (!kDirectIO)
            {
                if (r > 0)
                {
                    __syncthreads();
                }
            }
            load_input<Ktraits>(u + r * params.u_d_stride, u_vals[r], smem_load, params.seqlen - chunk * kChunkSize);
            if constexpr (!kDirectIO)
            {
                __syncthreads();
            }
            load_input<Ktraits>(
                delta + r * params.delta_d_stride, delta_vals_load[r], smem_load, params.seqlen - chunk * kChunkSize);
        }
        u += kChunkSize;
        delta += kChunkSize;

        float delta_vals[kNRows][kNItems], delta_u_vals[kNRows][kNItems], out_vals[kNRows][kNItems];
#pragma unroll
        for (int r = 0; r < kNRows; ++r)
        {
#pragma unroll
            for (int i = 0; i < kNItems; ++i)
            {
                float u_val = float(u_vals[r][i]);
                delta_vals[r][i] = float(delta_vals_load[r][i]) + delta_bias[r];
                if (params.delta_softplus)
                {
                    delta_vals[r][i] = delta_vals[r][i] <= 20.f ? log1pf(expf(delta_vals[r][i])) : delta_vals[r][i];
                }
                delta_u_vals[r][i] = delta_vals[r][i] * u_val;
                out_vals[r][i] = D_val[r] * u_val;
            }
        }

        __syncthreads();
        for (int state_idx = 0; state_idx < params.dstate; ++state_idx)
        {
            weight_t A_val[kNRows];
#pragma unroll
            for (int r = 0; r < kNRows; ++r)
            {
                A_val[r] = A[state_idx * params.A_dstate_stride + r * params.A_d_stride];
                // Multiply the real part of A with LOG2E so we can use exp2f instead of expf.
                constexpr float kLog2e = M_LOG2E;
                A_val[r] *= kLog2e;
            }
            // This variable holds B * C if both B and C are constant across seqlen. If only B varies
            // across seqlen, this holds C. If only C varies across seqlen, this holds B.
            // If both B and C vary, this is unused.
            weight_t BC_val[kNRows];
            weight_t B_vals[kNItems], C_vals[kNItems];
            if constexpr (kIsVariableB)
            {
                load_weight<Ktraits>(Bvar + state_idx * params.B_dstate_stride, B_vals, smem_load_weight,
                    params.seqlen - chunk * kChunkSize);
                if constexpr (!kIsVariableC)
                {
#pragma unroll
                    for (int r = 0; r < kNRows; ++r)
                    {
                        BC_val[r] = C[state_idx * params.C_dstate_stride + r * params.C_d_stride];
                    }
                }
            }
            if constexpr (kIsVariableC)
            {
                auto& smem_load_weight_C = !kIsVariableB ? smem_load_weight : smem_load_weight1;
                load_weight<Ktraits>(Cvar + state_idx * params.C_dstate_stride, C_vals, smem_load_weight_C,
                    params.seqlen - chunk * kChunkSize);
                if constexpr (!kIsVariableB)
                {
#pragma unroll
                    for (int r = 0; r < kNRows; ++r)
                    {
                        BC_val[r] = B[state_idx * params.B_dstate_stride + r * params.B_d_stride];
                    }
                }
            }
            if constexpr (!kIsVariableB && !kIsVariableC)
            {
#pragma unroll
                for (int r = 0; r < kNRows; ++r)
                {
                    BC_val[r] = B[state_idx * params.B_dstate_stride + r * params.B_d_stride]
                        * C[state_idx * params.C_dstate_stride + r * params.C_d_stride];
                }
            }

#pragma unroll
            for (int r = 0; r < kNRows; ++r)
            {
                if (r > 0)
                {
                    __syncthreads();
                } // Scan could be using the same smem
                scan_t thread_data[kNItems];
#pragma unroll
                for (int i = 0; i < kNItems; ++i)
                {
                    thread_data[i] = make_float2(exp2f(delta_vals[r][i] * A_val[r]),
                        !kIsVariableB ? delta_u_vals[r][i] : B_vals[i] * delta_u_vals[r][i]);
                    if constexpr (!Ktraits::kIsEvenLen)
                    { // So that the last state is correct
                        if (threadIdx.x * kNItems + i >= params.seqlen - chunk * kChunkSize)
                        {
                            thread_data[i] = make_float2(1.f, 0.f);
                        }
                    }
                }
                // Initialize running total
                scan_t running_prefix;
                // If we use WARP_SCAN then all lane 0 of all warps (not just thread 0) needs to read
                running_prefix = chunk > 0 && threadIdx.x % 32 == 0 ? smem_running_prefix[state_idx + r * MAX_DSTATE]
                                                                    : make_float2(1.f, 0.f);
                // running_prefix = chunk > 0 && threadIdx.x == 0 ? smem_running_prefix[state_idx] :
                // make_float2(1.f, 0.f);
                SSMScanPrefixCallbackOp<weight_t> prefix_op(running_prefix);
                Ktraits::BlockScanT(smem_scan).InclusiveScan(
                    thread_data, thread_data, SSMScanOp<weight_t>(), prefix_op);
                // There's a syncthreads in the scan op, so we don't need to sync here.
                // Unless there's only 1 warp, but then it's the same thread (0) reading and writing.
                if (threadIdx.x == 0)
                {
                    smem_running_prefix[state_idx] = prefix_op.running_prefix;
                    if (chunk == params.n_chunks - 1)
                    {
                        x[r * params.dstate + state_idx] = prefix_op.running_prefix.y;
                    }
                }
#pragma unroll
                for (int i = 0; i < kNItems; ++i)
                {
                    const weight_t C_val
                        = !kIsVariableC ? BC_val[r] : (!kIsVariableB ? BC_val[r] * C_vals[i] : C_vals[i]);
                    out_vals[r][i] += thread_data[i].y * C_val;
                }
            }
        }

        input_t* out = reinterpret_cast<input_t*>(params.out_ptr) + batch_id * params.out_batch_stride
            + dim_id * kNRows * params.out_d_stride + chunk * kChunkSize;
        if constexpr (kHasZ)
        {
            input_t* z = reinterpret_cast<input_t*>(params.z_ptr) + batch_id * params.z_batch_stride
                + dim_id * kNRows * params.z_d_stride + chunk * kChunkSize;
#pragma unroll
            for (int r = 0; r < kNRows; ++r)
            {
                input_t z_vals[kNItems];
                __syncthreads();
                load_input<Ktraits>(z + r * params.z_d_stride, z_vals, smem_load, params.seqlen - chunk * kChunkSize);
#pragma unroll
                for (int i = 0; i < kNItems; ++i)
                {
                    float z_val = z_vals[i];
                    out_vals[r][i] *= z_val / (1 + expf(-z_val));
                }
                __syncthreads();
                store_output<Ktraits>(
                    out + r * params.out_d_stride, out_vals[r], smem_store, params.seqlen - chunk * kChunkSize);
            }
        }
        else
        {
            __syncthreads();
#pragma unroll
            for (int r = 0; r < kNRows; ++r)
            {
                if constexpr (!kDirectIO)
                {
                    if (r > 0)
                    {
                        __syncthreads();
                    }
                }
                store_output<Ktraits>(
                    out + r * params.out_d_stride, out_vals[r], smem_store, params.seqlen - chunk * kChunkSize);
            }
        }

        Bvar += kChunkSize;
        Cvar += kChunkSize;
    }
}

template <int kNThreads, int kNItems, typename input_t, typename weight_t>
void selective_scan_fwd_launch(SSMParamsBase& params, cudaStream_t stream)
{
    // Only kNRows == 1 is tested for now, which ofc doesn't differ from previously when we had each block
    // processing 1 row.
    constexpr int kNRows = 1;
    BOOL_SWITCH(params.seqlen % (kNThreads * kNItems) == 0, kIsEvenLen,
        [&]
        {
            BOOL_SWITCH(params.is_variable_B, kIsVariableB,
                [&]
                {
                    BOOL_SWITCH(params.is_variable_C, kIsVariableC,
                        [&]
                        {
                            BOOL_SWITCH(params.z_ptr != nullptr, kHasZ,
                                [&]
                                {
                                    using Ktraits = Selective_Scan_fwd_kernel_traits<kNThreads, kNItems, kNRows,
                                        kIsEvenLen, kIsVariableB, kIsVariableC, kHasZ, input_t, weight_t>;
                                    // constexpr int kSmemSize = Ktraits::kSmemSize;
                                    constexpr int kSmemSize
                                        = Ktraits::kSmemSize + kNRows * MAX_DSTATE * sizeof(typename Ktraits::scan_t);
                                    // printf("smem_size = %d\n", kSmemSize);
                                    dim3 grid(params.batch, params.dim / kNRows);
                                    auto kernel = &selective_scan_fwd_kernel<Ktraits>;
                                    if (kSmemSize >= 48 * 1024)
                                    {
                                        TLLM_CUDA_CHECK(cudaFuncSetAttribute(
                                            kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemSize));
                                    }
                                    kernel<<<grid, Ktraits::kNThreads, kSmemSize, stream>>>(params);
                                });
                        });
                });
        });
}

template <typename input_t, typename weight_t>
void invokeSelectiveScan(SSMParamsBase& params, cudaStream_t stream)
{
    if (params.seqlen <= 128)
    {
        selective_scan_fwd_launch<32, 4, input_t, weight_t>(params, stream);
    }
    else if (params.seqlen <= 256)
    {
        selective_scan_fwd_launch<32, 8, input_t, weight_t>(params, stream);
    }
    else if (params.seqlen <= 512)
    {
        selective_scan_fwd_launch<32, 16, input_t, weight_t>(params, stream);
    }
    else if (params.seqlen <= 1024)
    {
        selective_scan_fwd_launch<64, 16, input_t, weight_t>(params, stream);
    }
    else
    {
        selective_scan_fwd_launch<128, 16, input_t, weight_t>(params, stream);
    }
}

#define INSTANTIATE_SELECTIVE_SCAN_DATA_TYPE(input_t, weight_t)                                                        \
    template void invokeSelectiveScan<input_t, weight_t>(SSMParamsBase & params, cudaStream_t stream);

INSTANTIATE_SELECTIVE_SCAN_DATA_TYPE(float, float);
INSTANTIATE_SELECTIVE_SCAN_DATA_TYPE(half, float);
#ifdef ENABLE_BF16
INSTANTIATE_SELECTIVE_SCAN_DATA_TYPE(__nv_bfloat16, float);
#endif
#undef INSTANTIATE_SELECTIVE_SCAN_DATA_TYPE

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename input_t, typename weight_t, bool dt_softplus, bool has_dt_bias, bool has_d, bool has_z>
__global__ void selectiveScanUpdate(SSMParamsBase params)
{
    // Shared memory.
    extern __shared__ char smem_[];

    input_t* smem_b = reinterpret_cast<input_t*>(smem_);
    input_t* smem_c = reinterpret_cast<input_t*>(smem_ + sizeof(input_t) * params.dstate);

    const int batch_id = blockIdx.x;
    const int dim_id = blockIdx.y * blockDim.x + threadIdx.x;

    const input_t x = reinterpret_cast<const input_t*>(params.u_ptr)[batch_id * params.u_batch_stride + dim_id];
    const weight_t* A = reinterpret_cast<const weight_t*>(params.A_ptr) + dim_id * params.A_d_stride;
    const input_t* B = reinterpret_cast<const input_t*>(params.B_ptr) + batch_id * params.B_batch_stride;
    const input_t* C = reinterpret_cast<const input_t*>(params.C_ptr) + batch_id * params.C_batch_stride;
    const float* D_ptr = reinterpret_cast<const float*>(params.D_ptr);
    const input_t* z_ptr = reinterpret_cast<const input_t*>(params.z_ptr);
    weight_t* state = reinterpret_cast<weight_t*>(params.x_ptr) + batch_id * params.state_batch_stride
        + dim_id * params.state_d_stride;
    const input_t dt
        = reinterpret_cast<const input_t*>(params.delta_ptr)[batch_id * params.delta_batch_stride + dim_id];
    const float* dt_bias_ptr = reinterpret_cast<const float*>(params.delta_bias_ptr);
    input_t* out = reinterpret_cast<input_t*>(params.out_ptr) + batch_id * params.out_batch_stride;
    float out_tmp = 0.0f;

    // get delta bias
    float dt_bias = 0.0f;
    if (has_dt_bias)
    {
        dt_bias = dt_bias_ptr[dim_id];
    }

    // get D
    float D = 0.0f;
    if (has_d)
    {
        D = D_ptr[dim_id];
    }

    // dt = softplus(dt + dt_bias)
    float dt_val = float(dt) + dt_bias;
    if (dt_softplus)
    {
        dt_val = dt_val <= 20.f ? log1pf(expf(dt_val)) : dt_val;
    }

    out_tmp = D * float(x);

    // read B, C
    if (threadIdx.x == 0)
    {
#pragma unroll
        for (int i = 0; i < params.dstate; ++i)
        {
            smem_b[i] = B[i];
            smem_c[i] = C[i];
        }
    }
    __syncthreads();

    for (int state_idx = 0; state_idx < params.dstate; ++state_idx)
    {
        // read A
        weight_t A_val = A[state_idx];

        // Multiply the real part of A with LOG2E so we can use exp2f instead of expf.
        constexpr float kLog2e = M_LOG2E;
        A_val *= kLog2e;

        // dtA = exp(dt * A), dtB = dt * B
        float dt_A = exp2f(dt_val * A_val);
        float dt_B = dt_val * float(smem_b[state_idx]);

        // update state
        float state_new = float(state[state_idx]) * dt_A + float(x) * dt_B;
        state[state_idx] = weight_t(state_new);

        // y = C * state + D * x
        out_tmp += state_new * float(smem_c[state_idx]);
    }

    // y = y * silu(z)
    if (has_z)
    {
        float z = z_ptr[batch_id * params.z_batch_stride + dim_id];
        out_tmp *= z / (1 + expf(-z));
    }

    // save out
    out[dim_id] = input_t(out_tmp);
}

template <typename input_t, typename weight_t>
void invokeSelectiveScanUpdate(SSMParamsBase& params, cudaStream_t stream)
{
    const int kNThreads = 32;
    dim3 block(kNThreads);
    dim3 grid(params.batch, (params.dim + kNThreads - 1) / kNThreads);
    // only save B and C to shared mem for reuse
    size_t smem_size = params.dstate * sizeof(input_t) * 2;

    BOOL_SWITCH(params.delta_softplus, kDtSoftplus,
        [&]
        {
            BOOL_SWITCH(params.delta_bias_ptr != nullptr, kHasDtBias,
                [&]
                {
                    BOOL_SWITCH(params.D_ptr != nullptr, kHasD,
                        [&]
                        {
                            BOOL_SWITCH(params.z_ptr != nullptr, kHasZ,
                                [&]
                                {
                                    selectiveScanUpdate<input_t, weight_t, kDtSoftplus, kHasDtBias, kHasD, kHasZ>
                                        <<<grid, block, smem_size, stream>>>(params);
                                });
                        });
                });
        });
}

#define INSTANTIATE_SELECTIVE_SCAN_UPDATE_DATA_TYPE(input_t, weight_t)                                                 \
    template void invokeSelectiveScanUpdate<input_t, weight_t>(SSMParamsBase & params, cudaStream_t stream)

INSTANTIATE_SELECTIVE_SCAN_UPDATE_DATA_TYPE(float, float);
INSTANTIATE_SELECTIVE_SCAN_UPDATE_DATA_TYPE(half, float);
#ifdef ENABLE_BF16
INSTANTIATE_SELECTIVE_SCAN_UPDATE_DATA_TYPE(__nv_bfloat16, float);
#endif
#undef INSTANTIATE_SELECTIVE_SCAN_UPDATE_DATA_TYPE

} // namespace kernels
} // namespace tensorrt_llm
