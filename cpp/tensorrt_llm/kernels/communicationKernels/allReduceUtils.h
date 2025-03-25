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
#pragma once
#include "tensorrt_llm/kernels/communicationKernels/allReduceFusionKernels.h"
#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace tensorrt_llm::kernels::ar_fusion
{
template <typename DType>
struct ElemsPerAccess;

template <>
struct ElemsPerAccess<half>
{
    static constexpr int value = 8;
};

template <>
struct ElemsPerAccess<nv_bfloat16>
{
    static constexpr int value = 8;
};

template <>
struct ElemsPerAccess<float>
{
    static constexpr int value = 4;
};

template <typename DType>
static constexpr int kElemsPerAccess = ElemsPerAccess<DType>::value;

template <int NRanks>
struct SyncComm
{
    __device__ __forceinline__ SyncComm(void** workspace)
    {
        counter_ptr = &reinterpret_cast<uint32_t*>(workspace[NRanks * 4])[0];
        flag_ptr = &reinterpret_cast<uint32_t*>(workspace[NRanks * 4])[1];
        flag_value = *flag_ptr;
        auto comm_offset = flag_value % 2 ? 0 : NRanks;
        for (int r = 0; r < NRanks; ++r)
        {
            comm_bufs[r] = workspace[r + comm_offset];
            barrier_flags[r] = workspace[NRanks * 2 + r];
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

    uint32_t* counter_ptr;
    uint32_t* flag_ptr;
    void* comm_bufs[NRanks];
    void* barrier_flags[NRanks];
    int flag_value;
};

template <int NRanks>
struct LamportComm
{
    __device__ __forceinline__ LamportComm(void** workspace, int rank)
    {
        counter_ptr = &reinterpret_cast<uint32_t*>(workspace[NRanks * 4])[0];
        flag_ptr = &reinterpret_cast<uint32_t*>(workspace[NRanks * 4])[2];
        clear_ptr = &reinterpret_cast<uint32_t*>(workspace[NRanks * 4])[4];
        flag_value = *flag_ptr;
        int comm_size = reinterpret_cast<uint32_t*>(workspace[NRanks * 4])[3];
        clear_size = *clear_ptr;
        int data_offset = flag_value % 3;
        int clear_offset = (flag_value + 2) % 3;
        for (int r = 0; r < NRanks; ++r)
        {
            data_bufs[r] = reinterpret_cast<uint8_t*>(workspace[3 * NRanks + r]) + data_offset * comm_size;
        }
        clear_buf = reinterpret_cast<uint8_t*>(workspace[3 * NRanks + rank]) + clear_offset * comm_size;
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

    uint32_t* counter_ptr;
    uint32_t* flag_ptr;
    uint32_t* clear_ptr;
    uint8_t* data_bufs[NRanks];
    uint8_t* clear_buf;
    uint32_t clear_size;
    uint32_t flag_value;
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
        return (flag + 1) % 4;
    }

    __device__ __forceinline__ int prev_flag(int flag)
    {
        return (flag + 3) % 4;
    }

public:
    int m_flag_value;

private:
    int* m_target_flag;
    int* m_current_flag;
};
} // namespace tensorrt_llm::kernels::ar_fusion
