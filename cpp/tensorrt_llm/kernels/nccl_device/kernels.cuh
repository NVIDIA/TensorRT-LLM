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

#pragma once

#include "constants.h"
#include "multimem.cuh"
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

    bool isMask = threadIdx.x < (blockDim.x / 32.f);
#pragma unroll
    for (int i = 0; i < NUM; i++)
    {
        val[i] = isMask ? shared[i][lane] : (T) (0.0f);
    }
    warpReduceSumV2<T, NUM>(val);
    return (T) 0.0f;
}

// AllReduce deterministic multimem unrolled kernel with template parameters
template <typename T, typename TN, int Nunroll, bool useResidual, bool useBias, bool oneShot>
__global__ void fusedAllReduceRMSNormKernel(ncclWindow_t inputWin, ncclWindow_t outputWin, const TN* residual,
    ncclWindow_t residualOutWin, const TN* weight, const TN* bias, int const startToken, int const hiddenSize,
    int const tokensPerRank, ncclDevComm devComm, float const eps)
{

    using accType = typename VectorType<T>::accType;
    ncclLsaBarrierSession<ncclCoopCta> bar{
        ncclCoopCta(), devComm, ncclTeamLsa(devComm), devComm.lsaBarrier, blockIdx.x, true, devComm.lsaMultimem};
    bar.sync(ncclCoopCta(), cuda::memory_order_relaxed);

    // Calculate which token this block should process
#pragma unroll 1
    for (int tokenOffset = blockIdx.x; tokenOffset < tokensPerRank; tokenOffset += gridDim.x)
    {
        int const tokenId = tokenOffset + startToken;
        // Calculate elements per vector type
        constexpr int elemsPerVec = sizeof(TN) / sizeof(T);

        int const tokenBaseOffset = tokenId * hiddenSize; // Base offset for this token in T elements

        // Calculate warp and lane within this block
        int const warpId = threadIdx.x / kWarpSize;
        int const laneId = threadIdx.x % kWarpSize;

        // Ensure warp striding through memory within the token
        // Scale offsets by elements per vector since each thread handles more data with vectors
        int const warpOffset = (warpId * kWarpSize * Nunroll) * elemsPerVec;
        int const laneOffset = laneId * elemsPerVec;

        int const baseOffsetT = warpOffset + laneOffset + tokenBaseOffset;

        // Get aligned pointers for vector types
        TN* sendPtr = reinterpret_cast<TN*>(ncclGetMultimemPointer(inputWin, 0, devComm.lsaMultimem));
        TN* recvPtr = reinterpret_cast<TN*>(
            oneShot ? ncclGetLocalPointer(outputWin, 0) : ncclGetMultimemPointer(outputWin, 0, devComm.lsaMultimem));

        assert(sendPtr != nullptr);
        assert(recvPtr != nullptr);
        TN* residualOut = nullptr;
        if constexpr (useResidual)
        {
            residualOut = oneShot
                ? reinterpret_cast<TN*>(ncclGetLocalPointer(residualOutWin, 0))
                : reinterpret_cast<TN*>(ncclGetMultimemPointer(residualOutWin, 0, devComm.lsaMultimem));

            assert(residual != nullptr);
            assert(residualOut != nullptr);
        }
        if constexpr (useBias)
        {
            assert(bias != nullptr);
        }

        // Process exactly the elements assigned to this thread
        TN v[Nunroll];
        accType localSumSquares = accType{0}; // For RMS calculation
#pragma unroll Nunroll
        for (int i = 0; i < Nunroll; i++)
        {
            int const strideOffset = i * kWarpSize * elemsPerVec; // Scale stride by elements per vector
            size_t const offsetT = baseOffsetT + strideOffset;
            size_t const offsetTN = offsetT / elemsPerVec;        // Convert to vector offset

            v[i] = multimemLoadSum<T, TN>(reinterpret_cast<T*>(sendPtr + offsetTN));
        }

#pragma unroll Nunroll
        for (int i = 0; i < Nunroll; i++)
        {
            int const strideOffset = i * kWarpSize * elemsPerVec; // Scale stride by elements per vector
            size_t const offsetT = baseOffsetT + strideOffset;
            size_t const offsetTN = offsetT / elemsPerVec;        // Convert to vector offset
            // The residual is the allreduced result (v) plus the input residual
            T const* residualElem = useResidual ? reinterpret_cast<T const*>(residual + offsetTN) : nullptr;
            T* vElem = reinterpret_cast<T*>(&v[i]);

#pragma unroll elemsPerVec
            for (int j = 0; j < elemsPerVec; ++j)
            {
                if constexpr (useResidual)
                {
                    // Residual = allreduced_result + input_residual
                    vElem[j] = static_cast<T>(static_cast<accType>(vElem[j]) + static_cast<accType>(residualElem[j]));
                }

                // Calculate sum of squares using residual values
                accType value = static_cast<accType>(vElem[j]);
                localSumSquares += value * value;
            }
        }

#pragma unroll Nunroll
        for (int i = 0; i < Nunroll; i++)
        {
            int const strideOffset = i * kWarpSize * elemsPerVec; // Scale stride by elements per vector
            size_t const offsetT = baseOffsetT + strideOffset;
            size_t const offsetTN = offsetT / elemsPerVec;        // Convert to vector offset
            if (!oneShot)
                multimemStore<T, TN>(reinterpret_cast<T*>(residualOut + offsetTN), v[i]);
            else
                residualOut[offsetTN] = v[i];
        }

        // RMS normalization: each block processes exactly one token
        __shared__ accType rms;
        blockReduceSumV2<accType, 1>(&localSumSquares);
        if (threadIdx.x == 0)
        {
            accType const blockSumSquares = localSumSquares;
            rms = rsqrtf((blockSumSquares / static_cast<accType>(hiddenSize)) + eps);
        }
        // Synchronize again to ensure RMS is computed before using it
        __syncthreads();

        // Apply RMS normalization with per-token weight and bias
#pragma unroll Nunroll
        for (int i = 0; i < Nunroll; i++)
        {
            // Get the position within the hidden dimension for this thread
            // Since each block processes one token, we just need the position within that token
            int const hiddenDimPos = warpOffset + laneOffset + i * kWarpSize * elemsPerVec;

            // Index into weight and bias arrays: just the position within hidden dimension
            TN weightVec = weight[hiddenDimPos / elemsPerVec];
            TN biasVec = useBias ? bias[hiddenDimPos / elemsPerVec] : TN{0};

            // Apply RMS normalization: v = (v / rms) * weight + bias
            // Unroll vector types and handle each element individually with proper type promotion
            T* vElem = reinterpret_cast<T*>(&v[i]);
            T* weightElem = reinterpret_cast<T*>(&weightVec);
            T* biasElem = reinterpret_cast<T*>(&biasVec);

#pragma unroll elemsPerVec
            for (int j = 0; j < elemsPerVec; ++j)
            {
                // Promote to accType for intermediate calculations
                accType vAcc = static_cast<accType>(vElem[j]);
                accType weightAcc = static_cast<accType>(weightElem[j]);
                accType biasAcc = static_cast<accType>(biasElem[j]);

                // Apply RMS normalization: v = (v / rms) * weight + bias
                accType normalized = vAcc * rms;
                accType weighted = normalized * weightAcc;
                accType result = weighted + biasAcc;

                // Cast back to T
                vElem[j] = static_cast<T>(result);
            }
        }
#pragma unroll Nunroll
        for (int i = 0; i < Nunroll; i++)
        {
            int const strideOffset = i * kWarpSize * elemsPerVec; // Scale stride by elements per vector
            size_t const offsetT = baseOffsetT + strideOffset;
            size_t const offsetTN = offsetT / elemsPerVec;        // Convert to vector offset
            if (!oneShot)
                multimemStore<T, TN>(reinterpret_cast<T*>(recvPtr + offsetTN), v[i]);
            else
                recvPtr[offsetTN] = v[i];
        }
    }
    bar.sync(ncclCoopCta(), cuda::memory_order_release);
}

#endif // NCCL_VERSION_CODE >= NCCL_VERSION(2,28,0)

} // namespace tensorrt_llm::kernels::nccl_device
