/*
 * Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

#ifndef TRTLLM_NCCL_DEVICE_KERNELS_CUH
#define TRTLLM_NCCL_DEVICE_KERNELS_CUH

#include "constants.h"
#include "multimem.h"
#include "nccl.h"
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 28, 0)
#include "nccl_device.h"
#endif
#include "tensorrt_llm/common/assert.h"
#include "vector_types.h"
#include <cassert>
#include <cooperative_groups.h>
#include <cub/cub.cuh>
#include <cuda/std/cmath>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace tensorrt_llm::kernels::nccl_device
{

#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 28, 0)

template <typename T, int NUM>
__inline__ __device__ T warpReduceSumV2(T* val)
{
    constexpr unsigned int kFinalMask = 0xffffffff;
#pragma unroll
    for (int i = 0; i < NUM; i++)
    {
#pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1)
            val[i] += __shfl_xor_sync(kFinalMask, val[i], mask, kWarpSize);
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

// AllReduce deterministic multimem unrolled kernel with template parameters
template <typename T, typename TN, int Nunroll, bool useResidual, bool useBias, bool oneShot>
__global__ void fusedAllReduceRMSNormKernel(ncclWindow_t input_win, ncclWindow_t output_win, const TN* residual,
    ncclWindow_t residual_out_win, const TN* weight, const TN* bias, int const startToken, int const hidden_size,
    int const tokensPerRank, ncclDevComm devComm, float const eps)
{

    using accType = typename VectorType<T>::accType;
    ncclLsaBarrierSession<ncclCoopCta> bar{
        ncclCoopCta(), devComm, ncclTeamLsa(devComm), devComm.lsaBarrier, blockIdx.x, true, devComm.lsaMultimem};
    bar.sync(ncclCoopCta(), cuda::memory_order_relaxed);

    // Calculate which token this block should process
#pragma unroll 1
    for (int token_offset = blockIdx.x; token_offset < tokensPerRank; token_offset += gridDim.x)
    {
        int const token_id = token_offset + startToken;
        // Calculate elements per vector type
        constexpr int elems_per_vec = sizeof(TN) / sizeof(T);

        int const token_base_offset = token_id * hidden_size; // Base offset for this token in T elements

        // Calculate warp and lane within this block
        int const warp_id = threadIdx.x / kWarpSize;
        int const lane_id = threadIdx.x % kWarpSize;

        // Ensure warp striding through memory within the token
        // Scale offsets by elements per vector since each thread handles more data with vectors
        int const warp_offset = (warp_id * kWarpSize * Nunroll) * elems_per_vec;
        int const lane_offset = lane_id * elems_per_vec;

        int const base_offset_T = warp_offset + lane_offset + token_base_offset;

        // Get aligned pointers for vector types
        TN* send_ptr = reinterpret_cast<TN*>(ncclGetMultimemPointer(input_win, 0, devComm.lsaMultimem));
        TN* recv_ptr = reinterpret_cast<TN*>(
            oneShot ? ncclGetLocalPointer(output_win, 0) : ncclGetMultimemPointer(output_win, 0, devComm.lsaMultimem));

        assert(send_ptr != nullptr);
        assert(recv_ptr != nullptr);
        TN* residual_out = nullptr;
        if constexpr (useResidual)
        {
            residual_out = oneShot
                ? reinterpret_cast<TN*>(ncclGetLocalPointer(residual_out_win, 0))
                : reinterpret_cast<TN*>(ncclGetMultimemPointer(residual_out_win, 0, devComm.lsaMultimem));

            assert(residual != nullptr);
            assert(residual_out != nullptr);
        }
        if constexpr (useBias)
        {
            assert(bias != nullptr);
        }

        // Process exactly the elements assigned to this thread
        TN v[Nunroll];
        accType local_sum_squares = accType{0}; // For RMS calculation
#pragma unroll Nunroll
        for (int i = 0; i < Nunroll; i++)
        {
            int const stride_offset = i * kWarpSize * elems_per_vec; // Scale stride by elements per vector
            size_t const offset_T = base_offset_T + stride_offset;
            size_t const offset_TN = offset_T / elems_per_vec;       // Convert to vector offset

            v[i] = multimemLoadSum<T, TN>(reinterpret_cast<T*>(send_ptr + offset_TN));
        }

#pragma unroll Nunroll
        for (int i = 0; i < Nunroll; i++)
        {
            int const stride_offset = i * kWarpSize * elems_per_vec; // Scale stride by elements per vector
            size_t const offset_T = base_offset_T + stride_offset;
            size_t const offset_TN = offset_T / elems_per_vec;       // Convert to vector offset
            // The residual is the allreduced result (v) plus the input residual
            T const* residual_elem = useResidual ? reinterpret_cast<T const*>(residual + offset_TN) : nullptr;
            T* v_elem = reinterpret_cast<T*>(&v[i]);

#pragma unroll elems_per_vec
            for (int j = 0; j < elems_per_vec; ++j)
            {
                if constexpr (useResidual)
                {
                    // Residual = allreduced_result + input_residual
                    v_elem[j]
                        = static_cast<T>(static_cast<accType>(v_elem[j]) + static_cast<accType>(residual_elem[j]));
                }

                // Calculate sum of squares using residual values
                accType value = static_cast<accType>(v_elem[j]);
                local_sum_squares += value * value;
            }
        }

#pragma unroll Nunroll
        for (int i = 0; i < Nunroll; i++)
        {
            int const stride_offset = i * kWarpSize * elems_per_vec; // Scale stride by elements per vector
            size_t const offset_T = base_offset_T + stride_offset;
            size_t const offset_TN = offset_T / elems_per_vec;       // Convert to vector offset
            if (!oneShot)
                multimemStore<T, TN>(reinterpret_cast<T*>(residual_out + offset_TN), v[i]);
            else
                residual_out[offset_TN] = v[i];
        }

        // RMS normalization: each block processes exactly one token
        __shared__ accType rms;
        blockReduceSumV2<accType, 1>(&local_sum_squares);
        if (threadIdx.x == 0)
        {
            accType const block_sum_squares = local_sum_squares;
            rms = rsqrtf((block_sum_squares / static_cast<accType>(hidden_size)) + eps);
        }
        // Synchronize again to ensure RMS is computed before using it
        __syncthreads();

        // Apply RMS normalization with per-token weight and bias
#pragma unroll Nunroll
        for (int i = 0; i < Nunroll; i++)
        {
            // Get the position within the hidden dimension for this thread
            // Since each block processes one token, we just need the position within that token
            int const hidden_dim_pos = warp_offset + lane_offset + i * kWarpSize * elems_per_vec;

            // Index into weight and bias arrays: just the position within hidden dimension
            TN weight_vec = weight[hidden_dim_pos / elems_per_vec];
            TN bias_vec = useBias ? bias[hidden_dim_pos / elems_per_vec] : TN{0};

            // Apply RMS normalization: v = (v / rms) * weight + bias
            // Unroll vector types and handle each element individually with proper type promotion
            T* v_elem = reinterpret_cast<T*>(&v[i]);
            T* weight_elem = reinterpret_cast<T*>(&weight_vec);
            T* bias_elem = reinterpret_cast<T*>(&bias_vec);

#pragma unroll elems_per_vec
            for (int j = 0; j < elems_per_vec; ++j)
            {
                // Promote to accType for intermediate calculations
                accType v_acc = static_cast<accType>(v_elem[j]);
                accType weight_acc = static_cast<accType>(weight_elem[j]);
                accType bias_acc = static_cast<accType>(bias_elem[j]);

                // Apply RMS normalization: v = (v / rms) * weight + bias
                accType normalized = v_acc * rms;
                accType weighted = normalized * weight_acc;
                accType result = weighted + bias_acc;

                // Cast back to T
                v_elem[j] = static_cast<T>(result);
            }
        }
#pragma unroll Nunroll
        for (int i = 0; i < Nunroll; i++)
        {
            int const stride_offset = i * kWarpSize * elems_per_vec; // Scale stride by elements per vector
            size_t const offset_T = base_offset_T + stride_offset;
            size_t const offset_TN = offset_T / elems_per_vec;       // Convert to vector offset
            if (!oneShot)
                multimemStore<T, TN>(reinterpret_cast<T*>(recv_ptr + offset_TN), v[i]);
            else
                recv_ptr[offset_TN] = v[i];
        }
    }
    bar.sync(ncclCoopCta(), cuda::memory_order_release);
}

#endif // NCCL_VERSION_CODE >= NCCL_VERSION(2,28,0)

} // namespace tensorrt_llm::kernels::nccl_device

#endif // TRTLLM_NCCL_DEVICE_KERNELS_CUH
