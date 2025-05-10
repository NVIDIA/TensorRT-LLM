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

#include "tensorrt_llm/common/cudaBf16Fallbacks.cuh"
#include "tensorrt_llm/common/cudaTypeUtils.cuh"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/customAllReduceUtils.h"
#include "tensorrt_llm/common/dataType.h"
#include "tensorrt_llm/common/envUtils.h"
#include "tensorrt_llm/kernels/communicationKernels/customLowPrecisionAllReduceKernels.h"
#include <cooperative_groups.h>
#include <tuple>
#include <type_traits>

namespace tensorrt_llm::kernels
{

using tensorrt_llm::common::divUp;
using tensorrt_llm::common::roundUp;
using tensorrt_llm::common::cuda_max;
using tensorrt_llm::common::cuda_abs;

static StaticLowPrecisionBuffers static_tp2_buffers;
static StaticLowPrecisionBuffers static_tp4_buffers;
static StaticLowPrecisionBuffers static_tp8_buffers;

StaticLowPrecisionBuffers* getBufferForTpSize(size_t tpSize)
{
    if (tpSize == 2)
    {
        return &static_tp2_buffers;
    }
    else if (tpSize == 4)
    {
        return &static_tp4_buffers;
    }
    else if (tpSize == 8)
    {
        return &static_tp8_buffers;
    }
    else
    {
        TLLM_THROW("Unsupported tpSize for LowPrecisionCustomAllReduce");
    }
}

void initialize_static_lowprecision_buffers(int64_t* buffer, size_t tpSize)
{
    void* const* buffer_ptrs = reinterpret_cast<void* const*>(buffer);

    StaticLowPrecisionBuffers* static_buffers = getBufferForTpSize(tpSize);

    // Store pointers in static structure
    for (int i = 0; i < tpSize; ++i)
    {
        static_buffers->peer_comm_buffer_ptrs[i] = buffer_ptrs[i];
        static_buffers->peer_comm_buffer_ptrs[tpSize + i] = buffer_ptrs[tpSize + i];
        static_buffers->peer_barrier_ptrs_in[i] = reinterpret_cast<uint64_t*>(buffer_ptrs[2 * tpSize + i]);
        static_buffers->peer_barrier_ptrs_out[i] = reinterpret_cast<uint64_t*>(buffer_ptrs[3 * tpSize + i]);
    }

    constexpr int LOW_PRECISION_NUM_POINTERS_PER_RANK = 4;
    // Store the flag pointer
    int flag_offset = 1;
    static_buffers->flag_ptr = &buffer[LOW_PRECISION_NUM_POINTERS_PER_RANK * tpSize + flag_offset];

    static_buffers->initialized = true;
    static_buffers->tpSize = tpSize;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

static inline __device__ void lp_allreduce_st_flag_release(uint64_t const& flag, uint64_t* flag_addr)
{
#if __CUDA_ARCH__ >= 700
    asm volatile("st.global.release.sys.b64 [%1], %0;" ::"l"(flag), "l"(flag_addr));
#else
    __threadfence_system();
    asm volatile("st.global.volatile.b64 [%1], %0;" ::"l"(flag), "l"(flag_addr));
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////

static inline __device__ void lp_allreduce_ld_flag_acquire(uint64_t& flag, uint64_t* flag_addr)
{
#if __CUDA_ARCH__ >= 700
    asm volatile("ld.global.acquire.sys.b64 %0, [%1];" : "=l"(flag) : "l"(flag_addr));
#else
    asm volatile("ld.global.volatile.b64 %0, [%1];" : "=l"(flag) : "l"(flag_addr));
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Type Converter that packs data format to 128 bits data type
//
using PackedFloat = union
{
    int4 packed;
    float unpacked[4];
};

using PackedHalf = union
{
    int4 packed;
    // half2 unpacked[4];
    __half unpacked[8];
};

template <typename T>
struct PackedOn16Bytes
{
};

template <typename T, int Num>
struct PackedOnNum
{
};

template <>
struct PackedOn16Bytes<float>
{
    using Type = PackedFloat;
};

template <>
struct PackedOn16Bytes<half>
{
    using Type = PackedHalf;
};

using PackedINT8 = union
{
    int4 packed;
    int8_t unpacked[16];
};

using PackedINT8_8Bytes = union
{
    int2 packed;
    int8_t unpacked[8];
};

using PackedINT8_4Bytes = union
{
    int packed;
    int8_t unpacked[4];
};

template <>
struct PackedOn16Bytes<int8_t>
{
    using Type = PackedINT8;
};

template <>
struct PackedOnNum<int8_t, 8>
{
    using Type = PackedINT8_8Bytes;
};

template <>
struct PackedOnNum<int8_t, 4>
{
    using Type = PackedINT8_4Bytes;
};

#ifdef ENABLE_BF16
using PackedBFloat16 = union
{
    int4 packed;
    //__nv_bfloat162 unpacked[4];
    __nv_bfloat16 unpacked[8];
};

template <>
struct PackedOn16Bytes<__nv_bfloat16>
{
    using Type = PackedBFloat16;
};
#endif

#ifdef ENABLE_FP8
using PackedFloat8E4m3 = union
{
    int4 packed;
    __nv_fp8_e4m3 unpacked[16];
};

using PackedFloat8E4m3_8Bytes = union
{
    int2 packed;
    __nv_fp8_e4m3 unpacked[8];
};

using PackedFloat8E4m3_4Bytes = union
{
    int packed;
    __nv_fp8_e4m3 unpacked[4];
};

template <>
struct PackedOn16Bytes<__nv_fp8_e4m3>
{
    using Type = PackedFloat8E4m3;
};

template <>
struct PackedOnNum<__nv_fp8_e4m3, 8>
{
    using Type = PackedFloat8E4m3_8Bytes;
};

template <>
struct PackedOnNum<__nv_fp8_e4m3, 4>
{
    using Type = PackedFloat8E4m3_4Bytes;
};
#endif

template <int num>
struct LowPrecisionIntPack
{
};

template <>
struct LowPrecisionIntPack<4>
{
    using Type = int;
};

template <>
struct LowPrecisionIntPack<8>
{
    using Type = int2;
};

template <>
struct LowPrecisionIntPack<16>
{
    using Type = int4;
};

__inline__ __device__ void multi_gpu_barrier(
    uint64_t** signals, const uint64_t flag, const size_t rank, const size_t world_size, int const tidx, int const bidx)
{
    // At the end of the function, we now that has least block 0 from all others GPUs have reached that point.
    uint64_t volatile* my_signals = signals[rank];
    if (tidx < world_size)
    {
        // The 1st block notifies the other ranks.
        if (bidx == 0)
        {
            signals[tidx][rank] = flag;
        }

        // Busy-wait until all ranks are ready.
        while (my_signals[tidx] != flag)
        {
        }
    }

    // Make sure we can move on...
    __syncthreads();
}

__device__ __forceinline__ void st_global_release(int4 const& val, int4* addr)
{
    asm volatile("st.release.global.sys.v4.b32 [%4], {%0, %1, %2, %3};" ::"r"(val.x), "r"(val.y), "r"(val.z),
        "r"(val.w), "l"(addr));
}

__device__ __forceinline__ int4 ld_global_acquire(int4* addr)
{
    int4 val;
    asm volatile("ld.acquire.global.sys.v4.b32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(val.x), "=r"(val.y), "=r"(val.z), "=r"(val.w)
                 : "l"(addr));
    return val;
}

__device__ __forceinline__ void st_global_volatile(int4 const& val, int4* addr)
{
    asm volatile("st.volatile.global.v4.b32 [%4], {%0, %1, %2, %3};" ::"r"(val.x), "r"(val.y), "r"(val.z), "r"(val.w),
        "l"(addr));
}

__device__ __forceinline__ int4 ld_global_volatile(int4* addr)
{
    int4 val;
    asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(val.x), "=r"(val.y), "=r"(val.z), "=r"(val.w)
                 : "l"(addr));
    return val;
}

__device__ __forceinline__ void fence_acq_rel_sys()
{
    asm volatile("fence.acq_rel.sys;" ::: "memory");
}

template <typename T>
__device__ __forceinline__ uintptr_t cvta_to_global(T* ptr)
{
    return (uintptr_t) __cvta_generic_to_global(ptr);
}

__device__ __forceinline__ uint64_t ld_volatile_global(uint64_t* ptr)
{
    uint64_t ans;
    asm("ld.volatile.global.u64 %0, [%1];" : "=l"(ans) : "l"(cvta_to_global(ptr)));
    return ans;
}

__device__ __forceinline__ void wait_send_peer(uint64_t local_flag, uint64_t* peer_flag_ptr)
{
    uint64_t peer_flag = ld_volatile_global(peer_flag_ptr);
    while (local_flag - peer_flag >= LP_ALLREDUCE_BUFFER_CHUNKS)
    {
        peer_flag = ld_volatile_global(peer_flag_ptr);
    }
    return;
}

__device__ __forceinline__ void wait_recv_peer(uint64_t local_flag, uint64_t* peer_flag_ptr)
{
    uint64_t peer_flag = ld_volatile_global(peer_flag_ptr);

    while (local_flag >= peer_flag)
    {
        peer_flag = ld_volatile_global(peer_flag_ptr);
    }
    return;
}

__device__ __forceinline__ void notify_peer(uint64_t* peer_flag_ptr)
{
    asm volatile("st.relaxed.sys.global.u64 [%0], %1;" ::"l"(cvta_to_global(peer_flag_ptr)), "l"(uint64_t(1))
                 : "memory");
    return;
}

__device__ __forceinline__ void notify_peer_with_value_relax(uint64_t* peer_flag_ptr, uint64_t value)
{
    asm volatile("st.relaxed.sys.global.u64 [%0], %1;" ::"l"(cvta_to_global(peer_flag_ptr)), "l"(value) : "memory");
    return;
}

__device__ __forceinline__ void notify_peer_with_value(uint64_t* peer_flag_ptr, uint64_t value)
{
    *peer_flag_ptr = value;
    return;
}

__device__ float warp_reduce_max(float val)
{
    val = cuda_max(__shfl_xor_sync(~0, val, 16), val);
    val = cuda_max(__shfl_xor_sync(~0, val, 8), val);
    val = cuda_max(__shfl_xor_sync(~0, val, 4), val);
    val = cuda_max(__shfl_xor_sync(~0, val, 2), val);
    val = cuda_max(__shfl_xor_sync(~0, val, 1), val);
    return val;
}

template <typename QUANTIZE_T>
struct QuantMaxValue;

template <>
struct QuantMaxValue<int8_t>
{
    static constexpr float value = 127.0f;
};

template <>
struct QuantMaxValue<__nv_fp8_e4m3>
{
    static constexpr float value = 448.0f;
};

template <int32_t RANKS_PER_NODE, typename T_IN, typename T_OUT>
__global__ void lowPrecisionPreprocessKernel(
    const T_IN* __restrict__ input, size_t elts_per_rank_in, size_t elts_per_rank_out, T_OUT* __restrict__ output)
{
    constexpr float QUANT_MAX = QuantMaxValue<T_OUT>::value;
    constexpr int32_t output_rounds = sizeof(T_IN) / sizeof(T_OUT);
    constexpr int32_t elts_per_thread = sizeof(int4) / sizeof(T_OUT);
    constexpr int32_t elts_per_round = sizeof(int4) / sizeof(T_IN);
    constexpr int32_t elts_per_warp_per_round = elts_per_round * WARP_SIZE;
    constexpr int32_t NUM_ELTS_PER_WARP_IN = (WARP_SIZE - 1) * elts_per_thread;
    constexpr int32_t NUM_ELTS_PER_WARP_OUT = WARP_SIZE * elts_per_thread;
    using PackedInputType = typename PackedOn16Bytes<T_IN>::Type;
    using PackedOutputType = typename PackedOnNum<T_OUT, elts_per_round>::Type;

    using PackedInputIntType = typename LowPrecisionIntPack<sizeof(int4)>::Type;
    using PackedOutputIntType = typename LowPrecisionIntPack<elts_per_round>::Type;

    const int32_t target_rank = blockIdx.x / (gridDim.x / RANKS_PER_NODE);
    const int32_t local_bid = blockIdx.x % (gridDim.x / RANKS_PER_NODE);

    input += elts_per_rank_in * target_rank;
    output += elts_per_rank_out * target_rank;

    const int32_t lane_id = threadIdx.x % WARP_SIZE;
    const int32_t wid = threadIdx.x / WARP_SIZE;

    PackedInputType vals[output_rounds];
    size_t start_in = NUM_ELTS_PER_WARP_IN * LP_ALLREDUCE_WARP_NUM_PER_BLOCK * local_bid + wid * NUM_ELTS_PER_WARP_IN;
    size_t start_out
        = NUM_ELTS_PER_WARP_OUT * LP_ALLREDUCE_WARP_NUM_PER_BLOCK * local_bid + wid * NUM_ELTS_PER_WARP_OUT;

#pragma unroll
    for (int32_t i = 0; i < output_rounds; ++i)
    {
        int32_t local_offset = lane_id * elts_per_round + elts_per_warp_per_round * i;
        int32_t global_offset = start_in + local_offset;
        if (local_offset < NUM_ELTS_PER_WARP_IN && global_offset < elts_per_rank_in)
        {
            vals[i].packed = *reinterpret_cast<PackedInputIntType const*>(input + start_in + local_offset);
        }
        else
        {
#pragma unroll
            for (int j = 0; j < elts_per_round; j++)
            {
                vals[i].unpacked[j] = 0.0f;
            }
        }
    }

    // Calculate scaling factor
    float scalar = 0;
    for (int32_t i = 0; i < output_rounds; ++i)
    {
#pragma unroll
        for (int32_t j = 0; j < elts_per_round; ++j)
        {
            scalar = cuda_max(cuda_abs((float) (vals[i].unpacked[j])), scalar);
        }
    }

    scalar = warp_reduce_max(scalar);
    if (scalar != 0.0f)
    {
        scalar = QUANT_MAX / scalar;
    }

    // Quantize and write output
    PackedOutputType output_vals[output_rounds];
    for (int32_t i = 0; i < output_rounds; ++i)
    {
        int32_t local_write_offset = lane_id * elts_per_round + elts_per_warp_per_round * i;
        if (local_write_offset < NUM_ELTS_PER_WARP_IN)
        {
#pragma unroll
            for (int32_t j = 0; j < elts_per_round; ++j)
            {
                float out_val = vals[i].unpacked[j];
                if (scalar != 0.0f)
                {
                    out_val *= scalar;
                }
                output_vals[i].unpacked[j] = static_cast<T_OUT>(out_val);
            }
        }
        else if (local_write_offset == NUM_ELTS_PER_WARP_IN)
        {
            *(reinterpret_cast<float*>(&output_vals[i])) = scalar;
        }
    }

#pragma unroll
    for (int32_t i = 0; i < output_rounds; ++i)
    {

        int32_t local_write_offset = lane_id * elts_per_round + elts_per_warp_per_round * i;
        *reinterpret_cast<PackedOutputIntType*>(output + start_out + local_write_offset) = output_vals[i].packed;
    }
}

template <int32_t RANKS_PER_NODE, typename T_IN>
__device__ void lowPrecisionTwoShotFirstStageKernel(int32_t myrank, size_t elts_per_rank, T_IN** input, float* smem)
{
    constexpr float QUANT_MAX = QuantMaxValue<T_IN>::value;
    constexpr int32_t elts_per_thread = sizeof(int4) / sizeof(T_IN);
    constexpr int32_t NUM_ELTS_PER_WARP_IN = WARP_SIZE * elts_per_thread;

    const int32_t lane_id = threadIdx.x % WARP_SIZE;
    const int32_t bid = blockIdx.x;
    const int32_t wid = threadIdx.x / WARP_SIZE;
    const size_t in_start
        = (bid * LP_ALLREDUCE_WARP_NUM_PER_BLOCK + wid) * NUM_ELTS_PER_WARP_IN + lane_id * elts_per_thread;

    // Packed data type for comms
    using PackedType = typename PackedOn16Bytes<T_IN>::Type;
    float* smem_scalar_ptr = &smem[RANKS_PER_NODE * wid];
    const size_t rank_offset = elts_per_rank * myrank;

    for (size_t local_offset = in_start; local_offset < elts_per_rank;
         local_offset += gridDim.x * blockDim.x * elts_per_thread)
    {
        float sums[elts_per_thread];
#pragma unroll
        for (int32_t ii = 0; ii < elts_per_thread; ++ii)
        {
            sums[ii] = 0;
        }

        // Read, dequantize and reduce sum
        {
            PackedType vals[RANKS_PER_NODE];
#pragma unroll
            for (int32_t ii = 0; ii < RANKS_PER_NODE; ++ii)
            {
                vals[ii].packed = *reinterpret_cast<int4 const*>(&input[ii][local_offset + rank_offset]);
            }

            if (lane_id == (WARP_SIZE - 1))
            {
#pragma unroll
                for (int32_t ii = 0; ii < RANKS_PER_NODE; ++ii)
                {
                    float* tmp_scalar = (float*) (&(vals[ii]));
                    smem_scalar_ptr[ii] = tmp_scalar[0];
                }
            }
            __syncwarp();

            if (lane_id < (WARP_SIZE - 1))
            {
                // Sum the values from the different ranks
                for (int32_t ii = 0; ii < RANKS_PER_NODE; ++ii)
                {
#pragma unroll
                    for (int32_t jj = 0; jj < elts_per_thread; ++jj)
                    {
                        if (smem_scalar_ptr[ii] != 0)
                        {
                            sums[jj] += (float) (vals[ii].unpacked[jj]) / smem_scalar_ptr[ii];
                        }
                        else
                        {
                            sums[jj] += (float) (vals[ii].unpacked[jj]);
                        }
                    }
                }
            }
        }

        // Quantize and write back results
        {
            float scalar = 0;

            if (lane_id < (WARP_SIZE - 1))
            {
#pragma unroll
                for (int32_t ii = 0; ii < elts_per_thread; ++ii)
                {
                    scalar = cuda_max(cuda_abs(sums[ii]), scalar);
                }
            }

            scalar = warp_reduce_max(scalar);

            if (scalar != 0.0f)
            {
                scalar = (QUANT_MAX) / scalar;
            }

            PackedType tmp_val;
            if (lane_id < (WARP_SIZE - 1))
            {
#pragma unroll
                for (int32_t ii = 0; ii < elts_per_thread; ++ii)
                {
                    float tmp = sums[ii];
                    if (scalar != 0.0f)
                    {
                        tmp *= scalar;
                    }
                    tmp_val.unpacked[ii] = static_cast<T_IN>(tmp);
                }
            }
            else
            {
                ((float*) (&tmp_val))[0] = scalar;
            }

            *reinterpret_cast<int4*>(input[0] + local_offset + rank_offset) = tmp_val.packed;
        }
    }
}

template <int32_t RANKS_PER_NODE, typename T_IN, typename T_OUT>
__device__ void lowPrecisionTwoShotSecondStageKernel(size_t input_elts_per_rank, size_t output_elts_per_rank,
    T_IN** input, T_OUT* output, float* smem, int32_t* dst_rank)
{
    constexpr int32_t elts_per_thread = sizeof(int4) / sizeof(T_IN);
    constexpr int32_t output_rounds = sizeof(T_OUT) / sizeof(T_IN);
    constexpr int32_t depack_num = elts_per_thread / output_rounds;

    constexpr int32_t NUM_ELTS_PER_WARP_IN = WARP_SIZE * elts_per_thread;
    constexpr int32_t NUM_ELTS_PER_WARP_OUT = (WARP_SIZE - 1) * elts_per_thread;

    const int32_t lane_id = threadIdx.x % WARP_SIZE;
    const int32_t bid = blockIdx.x;
    const int32_t wid = threadIdx.x / WARP_SIZE;

    const size_t in_start
        = (bid * LP_ALLREDUCE_WARP_NUM_PER_BLOCK + wid) * NUM_ELTS_PER_WARP_IN + lane_id * elts_per_thread;
    const size_t out_start
        = (bid * LP_ALLREDUCE_WARP_NUM_PER_BLOCK + wid) * NUM_ELTS_PER_WARP_OUT + lane_id * elts_per_thread;

    float* smem_scalar_ptr = &smem[RANKS_PER_NODE * wid];

    using PackedInType = typename PackedOn16Bytes<T_IN>::Type;
    using PackedOutType = typename PackedOn16Bytes<T_OUT>::Type;

    PackedInType vals[RANKS_PER_NODE];

    for (size_t input_offset = in_start, output_offset = out_start; input_offset < input_elts_per_rank;
         input_offset += gridDim.x * LP_ALLREDUCE_WARP_NUM_PER_BLOCK * NUM_ELTS_PER_WARP_IN,
                output_offset += gridDim.x * LP_ALLREDUCE_WARP_NUM_PER_BLOCK * NUM_ELTS_PER_WARP_OUT)
    {
#pragma unroll
        for (int32_t ii = 0; ii < RANKS_PER_NODE; ++ii)
        {
            size_t tmp_offset = dst_rank[ii] * input_elts_per_rank + input_offset;
            if (input_offset < input_elts_per_rank)
            {
                vals[ii].packed = *reinterpret_cast<int4 const*>(&input[ii][tmp_offset]);
            }
        }

        if (lane_id == (WARP_SIZE - 1))
        {
#pragma unroll
            for (int32_t ii = 0; ii < RANKS_PER_NODE; ++ii)
            {
                float* tmp_scalar = (float*) (&(vals[ii]));
                smem_scalar_ptr[ii] = tmp_scalar[0];
            }
        }
        __syncwarp();

        for (int32_t ii = 0; ii < RANKS_PER_NODE; ++ii)
        {
            float scale = smem_scalar_ptr[ii];
            size_t tmp_output_offset = dst_rank[ii] * output_elts_per_rank + output_offset;

            if (output_offset < output_elts_per_rank)
            {
                if (lane_id < (WARP_SIZE - 1))
                {
                    for (int32_t jj = 0; jj < output_rounds; ++jj)
                    {
                        PackedOutType tmp_output;

#pragma unroll
                        for (int32_t kk = 0; kk < depack_num; kk++)
                        {
                            float tmp = (float) (vals[ii].unpacked[kk + jj * depack_num]);
                            if (scale != 0.0f)
                            {
                                tmp /= scale;
                            }
                            tmp_output.unpacked[kk] = static_cast<T_OUT>(tmp);
                        }

                        *reinterpret_cast<PackedOutType*>(output + tmp_output_offset + jj * depack_num) = tmp_output;
                    }
                }
            }
        }
    }
}

template <typename T, typename QUANT_T, int32_t RANKS_PER_NODE>
static __global__ void lowPrecisionTwoShotAllReduceKernel(LowPrecisionAllReduceParams params)
{
    const int32_t bidx = blockIdx.x;
    const int32_t tidx = threadIdx.x;

    extern __shared__ float smem[];

    multi_gpu_barrier(params.peer_barrier_ptrs_in, params.barrier_flag, params.local_rank, RANKS_PER_NODE, tidx, bidx);

    // The source pointers. Distributed round-robin for the different warps.
    QUANT_T* src_d[RANKS_PER_NODE];
    // The destination ranks for round-robin gathering
    int32_t dst_rank[RANKS_PER_NODE];

#pragma unroll
    for (int32_t ii = 0; ii < RANKS_PER_NODE; ++ii)
    {
        int32_t rank = (params.local_rank + ii) % RANKS_PER_NODE;
        src_d[ii] = reinterpret_cast<QUANT_T*>(params.peer_comm_buffer_ptrs[rank]);
        dst_rank[ii] = rank;
    }

    lowPrecisionTwoShotFirstStageKernel<RANKS_PER_NODE, QUANT_T>(
        params.local_rank, params.buffer_elts_per_rank, src_d, smem);

    // Sync threads to make sure all block threads have the sums
    __syncthreads();

    // Barriers among the blocks with the same idx (release-acquire semantics)
    if (tidx < RANKS_PER_NODE)
    {
        // The all blocks notifies the other ranks.
        uint32_t flag_block_offset = RANKS_PER_NODE + bidx * RANKS_PER_NODE;
        lp_allreduce_st_flag_release(
            params.barrier_flag, params.peer_barrier_ptrs_in[tidx] + flag_block_offset + params.local_rank);

        // Busy-wait until all ranks are ready.
        uint64_t rank_barrier = 0;
        uint64_t* peer_barrier_d = params.peer_barrier_ptrs_in[params.local_rank] + flag_block_offset + tidx;
        do
        {
            lp_allreduce_ld_flag_acquire(rank_barrier, peer_barrier_d);
        } while (rank_barrier != params.barrier_flag);
    }

    __syncthreads();

    // Do allgather and dequantize
    float* smem_allgather = smem + (RANKS_PER_NODE * LP_ALLREDUCE_WARP_NUM_PER_BLOCK);
    lowPrecisionTwoShotSecondStageKernel<RANKS_PER_NODE, QUANT_T, T>(params.buffer_elts_per_rank, params.elts_per_rank,
        src_d, reinterpret_cast<T*>(params.local_output_buffer_ptr), smem_allgather, dst_rank);
}

template <typename T_IN, typename T_OUT>
__global__ void lowPrecisionHierPreprocessKernel(
    const T_IN* __restrict__ input, size_t n_in, T_OUT* __restrict__ output)
{
    constexpr float QUANT_MAX = QuantMaxValue<T_OUT>::value;
    constexpr int32_t output_rounds = sizeof(T_IN) / sizeof(T_OUT);
    constexpr int32_t elts_per_thread = sizeof(int4) / sizeof(T_OUT);
    constexpr int32_t elts_per_round = sizeof(int4) / sizeof(T_IN);
    constexpr int32_t elts_per_warp_per_round = elts_per_round * WARP_SIZE;
    constexpr int32_t NUM_ELTS_PER_WARP_IN = (WARP_SIZE - 1) * elts_per_thread;
    constexpr int32_t NUM_ELTS_PER_WARP_OUT = WARP_SIZE * elts_per_thread;

    using PackedInputType = typename PackedOn16Bytes<T_IN>::Type;
    using PackedOutputType = typename PackedOnNum<T_OUT, elts_per_round>::Type;
    using PackedInputIntType = typename LowPrecisionIntPack<16>::Type;
    using PackedOutputIntType = typename LowPrecisionIntPack<elts_per_round>::Type;

    const int32_t lane_id = threadIdx.x % WARP_SIZE;
    const int32_t wid = threadIdx.x / WARP_SIZE;
    PackedInputType vals[output_rounds];

    for (size_t start = blockIdx.x * LP_ALLREDUCE_WARP_NUM_PER_BLOCK + wid; start * NUM_ELTS_PER_WARP_IN < n_in;
         start += LP_ALLREDUCE_WARP_NUM_PER_BLOCK * gridDim.x)
    {
        int32_t read_rounds = 0;
        int32_t local_n_in = (n_in - start * NUM_ELTS_PER_WARP_IN) > NUM_ELTS_PER_WARP_IN
            ? NUM_ELTS_PER_WARP_IN
            : (n_in - start * NUM_ELTS_PER_WARP_IN);
        if (local_n_in <= 0)
        {
            return;
        }

#pragma unroll
        for (int32_t i = 0; i < output_rounds; ++i)
        {
            int32_t local_offset = lane_id * elts_per_round + elts_per_warp_per_round * i;
            if (local_offset < local_n_in)
            {
                vals[i].packed
                    = *reinterpret_cast<PackedInputIntType const*>(input + start * NUM_ELTS_PER_WARP_IN + local_offset);
                read_rounds++;
            }
            else
            {
#pragma unroll
                for (int j = 0; j < elts_per_round; j++)
                {
                    vals[i].unpacked[j] = 0.0f;
                }
            }
        }

        // Calculate scaling factor
        float scalar = 0;
        for (int32_t i = 0; i < read_rounds; ++i)
        {
#pragma unroll
            for (int32_t j = 0; j < elts_per_round; ++j)
            {
                scalar = cuda_max(cuda_abs((float) (vals[i].unpacked[j])), scalar);
            }
        }

        scalar = warp_reduce_max(scalar);
        if (scalar != 0.0f)
        {
            scalar = QUANT_MAX / scalar;
        }

        // Quantize and write output
        PackedOutputType output_vals[output_rounds];
        for (int32_t i = 0; i < output_rounds; ++i)
        {
            int32_t local_write_offset = lane_id * elts_per_round + elts_per_warp_per_round * i;
            if (local_write_offset < NUM_ELTS_PER_WARP_IN)
            {
#pragma unroll
                for (int32_t j = 0; j < elts_per_round; ++j)
                {
                    float out_val = vals[i].unpacked[j];
                    if (scalar != 0.0f)
                    {
                        out_val *= scalar;
                    }
                    output_vals[i].unpacked[j] = static_cast<T_OUT>(out_val);
                }
            }
            else if (local_write_offset == NUM_ELTS_PER_WARP_IN)
            {
                *(reinterpret_cast<float*>(&output_vals[i])) = scalar;
            }
        }

#pragma unroll
        for (int32_t i = 0; i < output_rounds; ++i)
        {
            int32_t local_write_offset = lane_id * elts_per_round + elts_per_warp_per_round * i;
            *reinterpret_cast<PackedOutputIntType*>(output + start * NUM_ELTS_PER_WARP_OUT + local_write_offset)
                = output_vals[i].packed;
        }
    }
}

template <int32_t RANKS_PER_NODE, typename T>
__device__ void hierReduceWithQdq(
    LowPrecisionAllReduceParams params, T** input, T* output, int64_t start_offset, int64_t length, float* smem)
{
    // Constants
    constexpr float QUANT_MAX = QuantMaxValue<T>::value;
    constexpr int32_t elts_per_thread = sizeof(int4) / sizeof(T);

    // Thread indices
    const int32_t lane_id = threadIdx.x % WARP_SIZE;
    const int32_t wid = threadIdx.x / WARP_SIZE;
    const size_t start = threadIdx.x * elts_per_thread;

    // Packed data type for comms
    using PackedType = typename PackedOn16Bytes<T>::Type;
    float* smem_scalar_ptr = &smem[RANKS_PER_NODE * wid];

    for (size_t index = start; index < length; index += LP_ALLREDUCE_DEFAULT_BLOCK_SIZE * elts_per_thread)
    {
        // Initialize sum array
        float sums[elts_per_thread];
#pragma unroll
        for (int32_t ii = 0; ii < elts_per_thread; ++ii)
        {
            sums[ii] = 0;
        }

        // Load values from different ranks and dequantize
        {
            PackedType vals[RANKS_PER_NODE];

#pragma unroll
            for (int32_t ii = 0; ii < RANKS_PER_NODE; ++ii)
            {
                vals[ii].packed = *reinterpret_cast<int4 const*>(&input[ii][start_offset + index]);
            }

            if (lane_id == (WARP_SIZE - 1))
            {
#pragma unroll
                for (int32_t ii = 0; ii < RANKS_PER_NODE; ++ii)
                {
                    float* tmp_scalar = (float*) (&(vals[ii]));
                    smem_scalar_ptr[ii] = tmp_scalar[0];
                }
            }
            __syncwarp();

            if (lane_id < (WARP_SIZE - 1))
            {
                for (int32_t ii = 0; ii < RANKS_PER_NODE; ++ii)
                {
#pragma unroll
                    for (int32_t jj = 0; jj < elts_per_thread; ++jj)
                    {
                        if (smem_scalar_ptr[ii] != 0)
                        {
                            sums[jj] += (float) (vals[ii].unpacked[jj]) / smem_scalar_ptr[ii];
                        }
                        else
                        {
                            sums[jj] += (float) (vals[ii].unpacked[jj]);
                        }
                    }
                }
            }
        }

        // Quantize results and write output
        {
            float scalar = 0;

            if (lane_id < (WARP_SIZE - 1))
            {
#pragma unroll
                for (int32_t ii = 0; ii < elts_per_thread; ++ii)
                {
                    scalar = cuda_max(cuda_abs(sums[ii]), scalar);
                }
            }

            scalar = warp_reduce_max(scalar);

            if (scalar != 0.0f)
            {
                scalar = QUANT_MAX / scalar;
            }

            PackedType tmp_val;

            if (lane_id < (WARP_SIZE - 1))
            {
#pragma unroll
                for (int32_t ii = 0; ii < elts_per_thread; ++ii)
                {
                    float tmp = sums[ii];
                    if (scalar != 0.0f)
                    {
                        tmp *= scalar;
                    }
                    tmp_val.unpacked[ii] = (T) tmp;
                }
            }
            else
            {
                ((float*) (&tmp_val))[0] = scalar;
            }

            *reinterpret_cast<int4*>(&output[threadIdx.x * elts_per_thread]) = tmp_val.packed;
        }
    }
}

template <int32_t RANKS_PER_NODE, typename T_IN, typename T_OUT>
__device__ void hierAllgatherWithDq(LowPrecisionAllReduceParams params, T_IN** input, T_OUT* output,
    size_t input_offset, int32_t global_iter, int32_t length, int32_t blocks_per_stage, float* smem)
{
    // Constants and thread indices
    constexpr int32_t elts_per_thread = sizeof(int4) / sizeof(T_IN);
    constexpr int32_t output_rounds = sizeof(T_OUT) / sizeof(T_IN);
    constexpr int32_t depack_num = elts_per_thread / output_rounds;

    const int32_t bidx = blockIdx.x;
    const int32_t tidx = threadIdx.x;
    const int32_t lane_id = tidx % WARP_SIZE;
    const int32_t wid = tidx / WARP_SIZE;
    const int32_t start = tidx * elts_per_thread;

    const int32_t OUTPUT_ELEMENT_PER_WARP = (WARP_SIZE - 1) * elts_per_thread;
    const int32_t OUTPUT_ELEMENT_PER_BLOCK = OUTPUT_ELEMENT_PER_WARP * LP_ALLREDUCE_WARP_NUM_PER_BLOCK;

    using PackedType = typename PackedOn16Bytes<T_IN>::Type;
    using PackedOutputType = typename PackedOn16Bytes<T_OUT>::Type;
    const int32_t numa_rank = params.numa_rank;

    PackedType vals[RANKS_PER_NODE];
    float* smem_scalar_ptr = &smem[RANKS_PER_NODE * wid];

    for (size_t index = start; index < length; index += LP_ALLREDUCE_DEFAULT_BLOCK_SIZE * elts_per_thread)
    {
#pragma unroll
        for (int32_t ii = 0; ii < RANKS_PER_NODE; ++ii)
        {
            vals[ii].packed = *reinterpret_cast<int4 const*>(&input[ii][input_offset + index]);
        }

#pragma unroll
        for (int32_t ii = 0; ii < RANKS_PER_NODE; ++ii)
        {
            if (lane_id == WARP_SIZE - 1)
            {
                float* tmp_scalar = (float*) (&(vals[ii]));
                smem_scalar_ptr[ii] = tmp_scalar[0];
            }
        }
        __syncwarp();

        const size_t elts_total = params.elts_total;

        for (int32_t ii = 0; ii < RANKS_PER_NODE; ++ii)
        {
            float scale = smem_scalar_ptr[ii];
            size_t offset_global = global_iter * blocks_per_stage * RANKS_PER_NODE * OUTPUT_ELEMENT_PER_BLOCK;

            int32_t tmp_rank = (numa_rank + ii) % RANKS_PER_NODE;
            size_t offset_local = offset_global + (bidx % blocks_per_stage) * RANKS_PER_NODE * OUTPUT_ELEMENT_PER_BLOCK
                + tmp_rank * OUTPUT_ELEMENT_PER_BLOCK + wid * OUTPUT_ELEMENT_PER_WARP + lane_id * elts_per_thread;
            bool need_write = elts_total > offset_local;

            if (lane_id < WARP_SIZE - 1 && need_write)
            {
                for (int32_t jj = 0; jj < output_rounds; ++jj)
                {
                    PackedOutputType tmp_output;

#pragma unroll
                    for (int32_t kk = 0; kk < depack_num; kk++)
                    {
                        float tmp = (float) (vals[ii].unpacked[kk + jj * depack_num]);
                        if (scale != 0)
                        {
                            tmp /= scale;
                        }
                        ((T_OUT*) (&tmp_output))[kk] = (T_OUT) tmp;
                    }

                    *reinterpret_cast<int4*>(&reinterpret_cast<T_OUT*>(output)[offset_local + jj * depack_num])
                        = *reinterpret_cast<int4*>(&tmp_output);
                }
            }
        }
    }
}

template <typename T, typename QUANT_T, int RANKS_PER_NODE>
static __global__ __launch_bounds__(512, 1) void lowPrecisionTwoShotHierAllReduceKernel(
    LowPrecisionAllReduceParams params)
{

    // The block index.
    int const bidx = blockIdx.x;
    // The thread index with the block.
    int const tidx = threadIdx.x;
    // The block num
    int const block_num = gridDim.x;
    int const duplicate = LP_ALLREDUCE_BUFFER_DUPLICATE;
    // this algorithm have 3 stages , so for one stage, have 1/3's block num
    int const block_num_per_stage = block_num / LP_ALLREDUCE_HIER_STAGE_NUM;

    // The number of elements packed into one for comms
    constexpr int elts_per_thread = sizeof(int4) / sizeof(QUANT_T);
    constexpr int ELTS_PER_BLOCK = elts_per_thread * LP_ALLREDUCE_DEFAULT_BLOCK_SIZE;

    extern __shared__ float smem[];

    multi_gpu_barrier(params.peer_barrier_ptrs_in, params.barrier_flag, params.local_rank, RANKS_PER_NODE, tidx, bidx);
    // Packed data type for comms
    using PackedType = typename PackedOn16Bytes<QUANT_T>::Type;

    if (bidx < block_num_per_stage)
    {
        // reduce-scatter inside NUMA
        int local_bid = bidx % block_num_per_stage;
        uint64_t send_flag = *params.rs_send_flags[local_bid];
        QUANT_T* src_d[LP_ALLREDUCE_RANKS_PER_NUMA];
        QUANT_T* dst = reinterpret_cast<QUANT_T*>(params.rs_buffers[local_bid]);

        // The destination ranks for round-robin gathering
#pragma unroll
        for (int ii = 0; ii < LP_ALLREDUCE_RANKS_PER_NUMA; ++ii)
        {
            int numa_rank = (params.numa_rank + ii) % LP_ALLREDUCE_RANKS_PER_NUMA;
            src_d[ii] = reinterpret_cast<QUANT_T*>(params.inputs_inside_numa[numa_rank]);
        }

        int32_t index = 0;
        while (index < params.num_rounds_fence)
        {
            if (tidx < LP_ALLREDUCE_NUMA_NUM)
            {
                wait_send_peer(send_flag, params.rs_ack_flags[local_bid] + tidx);
            }
            __syncthreads();
            int const processed = index * duplicate;
            int const remaining = params.num_rounds - processed;
            int const transfer_times = min(duplicate, remaining);

            for (int i = 0; i < transfer_times; ++i)
            {
                int const global_iter = index * duplicate + i;

                int const chunk_idx = send_flag % LP_ALLREDUCE_BUFFER_CHUNKS;
                int const dst_offset = chunk_idx * ELTS_PER_BLOCK * duplicate + ELTS_PER_BLOCK * i;
                int const global_per_tier = block_num_per_stage * LP_ALLREDUCE_RANKS_PER_NUMA * ELTS_PER_BLOCK;
                int const rank_offset = LP_ALLREDUCE_RANKS_PER_NUMA * ELTS_PER_BLOCK;
                const size_t global_offset
                    = global_iter * global_per_tier + local_bid * rank_offset + params.numa_rank * ELTS_PER_BLOCK;
                hierReduceWithQdq<LP_ALLREDUCE_RANKS_PER_NUMA, QUANT_T>(
                    params, src_d, dst + dst_offset, global_offset, ELTS_PER_BLOCK, smem);
            }

            __syncthreads();
            send_flag++;
            if (tidx == 0)
            {
                __threadfence_system();
                notify_peer_with_value(params.rs_notify_remote_flags[local_bid], send_flag);
                notify_peer_with_value(params.rs_notify_local_flags[local_bid], send_flag);
            }
            index++;
        }
        if (tidx == 0)
        {
            *params.rs_send_flags[local_bid] = send_flag;
        }
        return;
    }

    else if (bidx >= block_num_per_stage && bidx < block_num_per_stage * 2)
    {
        // partial allreduce cross NUMA
        int local_bid = bidx % block_num_per_stage;
        uint64_t send_flag = *params.ar_send_flags[local_bid];
        // 2 is all
        QUANT_T* src_d[LP_ALLREDUCE_NUMA_NUM];
        QUANT_T* dst = reinterpret_cast<QUANT_T*>(params.ar_buffers[local_bid]);
        src_d[0] = reinterpret_cast<QUANT_T*>(params.rs_buffers[local_bid]);
        src_d[1] = reinterpret_cast<QUANT_T*>(params.ar_peer_buffers_cross_numa[local_bid]);

        int32_t index = 0;
        while (index < params.num_rounds_fence)
        {
            if (tidx == 0)
            {
                wait_recv_peer(send_flag, params.rs_notify_local_flags[local_bid]);
                wait_recv_peer(send_flag, params.ar_ack_peer_rs_flags[local_bid]);
                wait_send_peer(send_flag, params.ar_ack_flags[local_bid]);
            }
            __syncthreads();

            int const processed = index * duplicate;
            int const remaining = params.num_rounds - processed;
            int const transfer_times = min(duplicate, remaining);

            int const chunk_idx = send_flag % LP_ALLREDUCE_BUFFER_CHUNKS;
            int const base_offset = chunk_idx * ELTS_PER_BLOCK * duplicate;

            for (int i = 0; i < transfer_times; ++i)
            {
                int const offset = base_offset + i * ELTS_PER_BLOCK;
                hierReduceWithQdq<LP_ALLREDUCE_NUMA_NUM, QUANT_T>(
                    params, src_d, dst + offset, offset, ELTS_PER_BLOCK, smem);
            }
            __syncthreads();

            send_flag++;
            if (tidx == 0)
            {
                __threadfence_system();
                notify_peer_with_value(params.ar_notify_rs_remote_flags[local_bid], send_flag);
                notify_peer_with_value(params.ar_notify_rs_local_flags[local_bid], send_flag);
                notify_peer_with_value(params.ar_notify_ag_flags[local_bid], send_flag);
            }
            index++;
        }
        if (tidx == 0)
        {
            *params.ar_send_flags[local_bid] = send_flag;
        }
        return;
    }
    else if (bidx >= block_num_per_stage * 2 && bidx < block_num_per_stage * 3)
    {
        // allgather inside NUMA
        int local_bid = bidx % block_num_per_stage;
        uint64_t send_flag = *params.ag_send_flags[local_bid];
        QUANT_T* src_d[LP_ALLREDUCE_RANKS_PER_NUMA];
        T* dst = reinterpret_cast<T*>(params.local_output_buffer_ptr);
#pragma unroll
        for (int ii = 0; ii < LP_ALLREDUCE_RANKS_PER_NUMA; ++ii)
        {
            int numa_rank = (params.numa_rank + ii) % LP_ALLREDUCE_RANKS_PER_NUMA;

            src_d[ii] = reinterpret_cast<QUANT_T*>(params.ag_peer_buffers_inside_numa[local_bid * 4 + numa_rank]);
        }

        int32_t index = 0;
        while (index < params.num_rounds_fence)
        {
            if (tidx == 0)
            {
                wait_recv_peer(send_flag, params.ar_notify_ag_flags[local_bid]);
            }

            __syncthreads();
            if (tidx < LP_ALLREDUCE_RANKS_PER_NUMA)
            {

                notify_peer_with_value_relax(
                    params.ag_notify_peer_inside_numa_flags[local_bid * LP_ALLREDUCE_RANKS_PER_NUMA + tidx],
                    send_flag + 1);
                wait_recv_peer(send_flag, params.ag_ack_peer_inside_numa_flags[local_bid] + tidx);
            }
            __syncthreads();

            int const processed = index * duplicate;
            int const remaining = params.num_rounds - processed;
            int const transfer_times = min(duplicate, remaining);

            int const chunk_idx = send_flag % LP_ALLREDUCE_BUFFER_CHUNKS;
            int const base_offset = chunk_idx * ELTS_PER_BLOCK * duplicate;

            for (int i = 0; i < transfer_times; ++i)
            {

                int const global_iter = processed + i;
                const size_t curr_offset = base_offset + i * ELTS_PER_BLOCK;

                hierAllgatherWithDq<LP_ALLREDUCE_RANKS_PER_NUMA, QUANT_T, T>(
                    params, src_d, dst, curr_offset, global_iter, ELTS_PER_BLOCK, block_num_per_stage, smem);
            }

            __syncthreads();

            send_flag++;
            if (tidx == 0)
            {
                notify_peer_with_value_relax(params.ar_ack_flags[local_bid], send_flag);
            }
            index++;
        }
        if (tidx == 0)
        {
            *params.ag_send_flags[local_bid] = send_flag;
        }
    }
    else
    {
        return;
    }
}

template <typename T, typename QUANT_T, int RANKS_PER_NODE>
void lowPrecisionAllReduceDispatchRanksPerNode(kernels::LowPrecisionAllReduceParams& params, cudaStream_t stream)
{
    constexpr int qtype_elts_per_load = LP_ALLREDUCE_BYTES_PER_LOAD / sizeof(QUANT_T);
    constexpr int elts_per_block = qtype_elts_per_load * (LP_ALLREDUCE_WARPSIZE - 1) * LP_ALLREDUCE_WARP_NUM_PER_BLOCK;
    constexpr int elts_per_block_with_scale = qtype_elts_per_load * LP_ALLREDUCE_DEFAULT_BLOCK_SIZE;
    if (RANKS_PER_NODE <= 4)
    {

        int blocks_per_grid = LP_ALLREDUCE_MAX_BLOCKS * 2, threads_per_block = LP_ALLREDUCE_DEFAULT_BLOCK_SIZE;

        params.elts_per_rank = params.elts_total / RANKS_PER_NODE;
        params.rank_offset = params.rank * params.elts_per_rank;
        params.elts_per_block = elts_per_block;

        size_t num_rounds_per_rank = (params.elts_per_rank - 1) / elts_per_block + 1;
        size_t my_rank = params.local_rank;

        params.buffer_offset = my_rank * elts_per_block_with_scale * num_rounds_per_rank;
        params.buffer_elts_per_rank = elts_per_block_with_scale * num_rounds_per_rank;
        lowPrecisionPreprocessKernel<RANKS_PER_NODE, T, QUANT_T>
            <<<num_rounds_per_rank * RANKS_PER_NODE, threads_per_block, 0, stream>>>(
                (T const*) params.local_input_buffer_ptr, params.elts_per_rank, params.buffer_elts_per_rank,
                (QUANT_T*) params.peer_comm_buffer_ptrs[my_rank]);
        lowPrecisionTwoShotAllReduceKernel<T, QUANT_T, RANKS_PER_NODE><<<blocks_per_grid, threads_per_block,
            (LP_ALLREDUCE_WARP_NUM_PER_BLOCK * RANKS_PER_NODE) * sizeof(float) * 2, stream>>>(params);
    }
    else
    {
        int blocks_per_grid = LP_ALLREDUCE_MAX_BLOCKS, threads_per_block = LP_ALLREDUCE_DEFAULT_BLOCK_SIZE;
        params.num_rounds = (((params.elts_total - 1) / elts_per_block + 1) - 1) / LP_ALLREDUCE_MAX_RANKS_PER_NUMA
                / LP_ALLREDUCE_MAX_BLOCKS
            + 1;
        params.num_rounds_fence = (params.num_rounds - 1) / LP_ALLREDUCE_BUFFER_DUPLICATE + 1;
        blocks_per_grid = params.num_rounds < LP_ALLREDUCE_MAX_BLOCKS ? params.num_rounds : blocks_per_grid;

        size_t preprocess_blocks_per_grid = params.num_rounds * LP_ALLREDUCE_MAX_RANKS_PER_NUMA * blocks_per_grid;
        size_t my_rank = params.local_rank;
        blocks_per_grid *= LP_ALLREDUCE_HIER_STAGE_NUM; // 3 stages need more block

        lowPrecisionHierPreprocessKernel<T, QUANT_T><<<preprocess_blocks_per_grid, LP_ALLREDUCE_DEFAULT_BLOCK_SIZE,
            (LP_ALLREDUCE_WARP_NUM_PER_BLOCK) * sizeof(float), stream>>>((T const*) params.local_input_buffer_ptr,
            params.elts_total, (QUANT_T*) params.peer_comm_buffer_ptrs[my_rank]);
        lowPrecisionTwoShotHierAllReduceKernel<T, QUANT_T, RANKS_PER_NODE><<<blocks_per_grid, threads_per_block,
            (LP_ALLREDUCE_WARP_NUM_PER_BLOCK * RANKS_PER_NODE) * sizeof(float), stream>>>(params);
    }
}

template <typename T>
void lowPrecisionAllReduceDispatchType(kernels::LowPrecisionAllReduceParams& param, cudaStream_t stream)
{
#ifdef ENABLE_FP8
    switch (param.ranks_per_node)
    {
    case 2: lowPrecisionAllReduceDispatchRanksPerNode<T, __nv_fp8_e4m3, 2>(param, stream); break;
    case 4: lowPrecisionAllReduceDispatchRanksPerNode<T, __nv_fp8_e4m3, 4>(param, stream); break;
    case 8: lowPrecisionAllReduceDispatchRanksPerNode<T, __nv_fp8_e4m3, 8>(param, stream); break;
    default: TLLM_THROW("Custom LowPrecision all reduce only supported on {2, 4, 8} GPUs per node.");
    }
#else
    TLLM_THROW("Can't Use Low Precision Allreduce When Compile Without ENABLE_FP8");
#endif
}

std::vector<size_t> splitNumber(size_t number)
{
    std::vector<size_t> parts;
    size_t parts_num = number / LP_ALLREDUCE_MAX_ELTS_IN_WORKSPACE;
    size_t remain = number % LP_ALLREDUCE_MAX_ELTS_IN_WORKSPACE;
    if (parts_num == 0)
    {
        parts.push_back(remain);
    }
    else
    {
        if (remain == 0)
        {
            for (size_t i = 0; i < parts_num; ++i)
            {
                parts.push_back(LP_ALLREDUCE_MAX_ELTS_IN_WORKSPACE);
            }
        }
        else
        {
            for (size_t i = 0; i < parts_num - 1; ++i)
            {
                parts.push_back(LP_ALLREDUCE_MAX_ELTS_IN_WORKSPACE);
            }
            // if last remain part is small, will split a normal part, and fuse remain part to half normal
            // part
            if (remain < LP_ALLREDUCE_MIN_ELTS_THRESHOLD)
            {
                parts.push_back(LP_ALLREDUCE_MAX_ELTS_IN_WORKSPACE / 2 + remain);
                parts.push_back(LP_ALLREDUCE_MAX_ELTS_IN_WORKSPACE / 2);
            }
            else
            {
                parts.push_back(LP_ALLREDUCE_MAX_ELTS_IN_WORKSPACE);
                parts.push_back(remain);
            }
        }
    }
    return parts;
}

LowPrecisionAllReduceParams LowPrecisionAllReduceParams::deserialize(
    size_t tpSize, size_t tpRank, nvinfer1::DataType dataType, int token_num, int hidden_size)
{

    // Get appropriate static buffer
    StaticLowPrecisionBuffers* static_buffers = getBufferForTpSize(tpSize);

    // Check initialization
    if (!static_buffers->initialized || static_buffers->tpSize != tpSize)
    {
        TLLM_THROW("Static buffers for TP size %zu not initialized", tpSize);
    }

    // Use the stored flag pointer
    *(static_buffers->flag_ptr) += 1;

    TLLM_LOG_TRACE("AllReduceParams's flag value is %d", *(static_buffers->flag_ptr));
    uint64_t flag_value = *(static_buffers->flag_ptr);
    LowPrecisionAllReduceParams params;
    // Even plugins use ping buffers, odd plugins use pong.
    // That way, we don't need to wait for other GPUs to be done
    // before copying input tensor to workspace.
    auto const buffer_offset = (flag_value % 2 == 0) ? 0 : tpSize;

    for (int i = 0; i < tpSize; ++i)
    {
        params.peer_comm_buffer_ptrs[i] = static_buffers->peer_comm_buffer_ptrs[buffer_offset + i];
    }
    for (int i = 0; i < tpSize; ++i)
    {
        params.peer_barrier_ptrs_in[i] = static_buffers->peer_barrier_ptrs_in[i];
    }
    for (int i = 0; i < tpSize; ++i)
    {
        params.peer_barrier_ptrs_out[i] = static_buffers->peer_barrier_ptrs_out[i];
    }
    // Assume that a single allreduce will not be divided into more than 64 allreduces of 64MB each,it is not very safe
    params.barrier_flag = flag_value;
    params.ranks_per_node = tpSize;
    params.local_rank = tpRank;

    return params;
}

LowPrecisionAllReduceParams LowPrecisionAllReduceParams::deserialize_hier(
    size_t tpSize, size_t tpRank, nvinfer1::DataType dataType, int token_num, int hidden_size)
{

    // Get appropriate static buffer
    StaticLowPrecisionBuffers* static_buffers = getBufferForTpSize(tpSize);

    // Check initialization
    if (!static_buffers->initialized || static_buffers->tpSize != tpSize)
    {
        TLLM_THROW("Static buffers for TP size %zu not initialized", tpSize);
    }

    // Use the stored flag pointer
    *(static_buffers->flag_ptr) += 1;

    TLLM_LOG_TRACE("AllReduceParams's flag value is %d", *(static_buffers->flag_ptr));
    uint64_t flag_value = *(static_buffers->flag_ptr);
    LowPrecisionAllReduceParams params;
    // Even plugins use ping buffers, odd plugins use pong.
    // That way, we don't need to wait for other GPUs to be done
    // before copying input tensor to workspace.
    auto const buffer_offset = (flag_value % 2 == 0) ? 0 : tpSize;

    for (int i = 0; i < tpSize; ++i)
    {
        params.peer_comm_buffer_ptrs[i] = static_buffers->peer_comm_buffer_ptrs[buffer_offset + i];
    }
    for (int i = 0; i < tpSize; ++i)
    {
        params.peer_barrier_ptrs_in[i] = static_buffers->peer_barrier_ptrs_in[i];
    }
    for (int i = 0; i < tpSize; ++i)
    {
        params.peer_barrier_ptrs_out[i] = static_buffers->peer_barrier_ptrs_out[i];
    }
    // Assume that a single allreduce will not be divided into more than 64 allreduces of 64MB each,it is not very safe
    params.barrier_flag = flag_value;
    params.ranks_per_node = tpSize;
    params.local_rank = tpRank;

    params.numa_rank = tpRank % LP_ALLREDUCE_MAX_RANKS_PER_NUMA;

    // assume quant_type is 1 bytes , so we can transfer LP_ALLREDUCE_BYTES_PER_LOAD elts once
    int REAL_ELTS_PER_BLOCK
        = (LP_ALLREDUCE_WARPSIZE - 1) * LP_ALLREDUCE_BYTES_PER_LOAD * LP_ALLREDUCE_WARP_NUM_PER_BLOCK;
    int QUANT_ELTS_PER_BLOCK = LP_ALLREDUCE_DEFAULT_BLOCK_SIZE * LP_ALLREDUCE_BYTES_PER_LOAD;

    int max_rounds = (((LP_ALLREDUCE_MAX_ELTS_IN_WORKSPACE - 1) / REAL_ELTS_PER_BLOCK + 1) - 1)
            / LP_ALLREDUCE_MAX_RANKS_PER_NUMA / LP_ALLREDUCE_MAX_BLOCKS
        + 1;
    int max_fence_rounds = (max_rounds - 1) / LP_ALLREDUCE_BUFFER_DUPLICATE + 1;

    uint64_t quantize_offset = max_fence_rounds * LP_ALLREDUCE_MAX_RANKS_PER_NUMA * LP_ALLREDUCE_MAX_BLOCKS
        * LP_ALLREDUCE_BUFFER_DUPLICATE * QUANT_ELTS_PER_BLOCK;
    for (int i = 0; i < LP_ALLREDUCE_MAX_RANKS_PER_NUMA; ++i)
    {
        params.inputs_inside_numa[i]
            = params.peer_comm_buffer_ptrs[(tpRank / LP_ALLREDUCE_MAX_RANKS_PER_NUMA) * LP_ALLREDUCE_MAX_RANKS_PER_NUMA
                + i];
    }

    for (int i = 0; i < LP_ALLREDUCE_MAX_BLOCKS; ++i)
    {

        const size_t block_buffer_size
            = QUANT_ELTS_PER_BLOCK * LP_ALLREDUCE_BUFFER_CHUNKS * LP_ALLREDUCE_BUFFER_DUPLICATE;
        char* base_ptr = reinterpret_cast<char*>(params.peer_comm_buffer_ptrs[tpRank]);

        params.rs_buffers[i] = base_ptr + quantize_offset + block_buffer_size * i;

        const size_t ar_buffer_offset = quantize_offset + block_buffer_size * LP_ALLREDUCE_MAX_BLOCKS;

        params.ar_buffers[i] = base_ptr + ar_buffer_offset + block_buffer_size * i;

        int const cross_numa_rank = (tpRank + LP_ALLREDUCE_MAX_RANKS_PER_NUMA) % tpSize;
        params.ar_peer_buffers_cross_numa[i] = reinterpret_cast<char*>(params.peer_comm_buffer_ptrs[cross_numa_rank])
            + quantize_offset + block_buffer_size * i;
        int const numa_group_base = (tpRank / LP_ALLREDUCE_MAX_RANKS_PER_NUMA) * LP_ALLREDUCE_MAX_RANKS_PER_NUMA;
        for (int j = 0; j < LP_ALLREDUCE_MAX_RANKS_PER_NUMA; ++j)
        {
            int const rank_in_numa = numa_group_base + j;
            params.ag_peer_buffers_inside_numa[i * LP_ALLREDUCE_MAX_RANKS_PER_NUMA + j]
                = reinterpret_cast<uint64_t*>(reinterpret_cast<char*>(params.peer_comm_buffer_ptrs[rank_in_numa])
                    + ar_buffer_offset + block_buffer_size * i);
        }

        const size_t rs_send_flags_offset = ar_buffer_offset + block_buffer_size * LP_ALLREDUCE_MAX_BLOCKS;
        params.rs_send_flags[i] = reinterpret_cast<uint64_t*>(base_ptr + rs_send_flags_offset + i * sizeof(uint64_t));

        uint64_t rs_ack_flags_offset = rs_send_flags_offset + LP_ALLREDUCE_MAX_BLOCKS * sizeof(uint64_t);
        params.rs_ack_flags[i]
            = reinterpret_cast<uint64_t*>(reinterpret_cast<char*>(params.peer_comm_buffer_ptrs[tpRank])
                + rs_ack_flags_offset + i * sizeof(uint64_t) * 2);

        uint64_t rs_notify_local_flags_offset = rs_ack_flags_offset + LP_ALLREDUCE_MAX_BLOCKS * sizeof(uint64_t) * 2;
        params.rs_notify_local_flags[i]
            = reinterpret_cast<uint64_t*>(reinterpret_cast<char*>(params.peer_comm_buffer_ptrs[tpRank])
                + rs_notify_local_flags_offset + i * sizeof(uint64_t));

        uint64_t rs_notify_remote_flags_offset
            = rs_notify_local_flags_offset + LP_ALLREDUCE_MAX_BLOCKS * sizeof(uint64_t);

        // now only 8gpus can use hier , so %8 is a magic num
        params.rs_notify_remote_flags[i] = reinterpret_cast<uint64_t*>(
            reinterpret_cast<char*>(params.peer_comm_buffer_ptrs[(tpRank + LP_ALLREDUCE_MAX_RANKS_PER_NUMA) % tpSize])
            + rs_notify_remote_flags_offset + i * sizeof(uint64_t));

        // special flag for ar stage
        params.ar_ack_peer_rs_flags[i]
            = reinterpret_cast<uint64_t*>(reinterpret_cast<char*>(params.peer_comm_buffer_ptrs[tpRank])
                + rs_notify_remote_flags_offset + i * sizeof(uint64_t));

        // rs stage handshake done

        // for partial ar stage handshake
        uint64_t ar_send_flags_offset = rs_notify_remote_flags_offset + LP_ALLREDUCE_MAX_BLOCKS * sizeof(uint64_t);
        params.ar_send_flags[i]
            = reinterpret_cast<uint64_t*>(reinterpret_cast<char*>(params.peer_comm_buffer_ptrs[tpRank])
                + ar_send_flags_offset + i * sizeof(uint64_t));

        // 2 flag in numa,so use fix *2
        // for ar notify , it is rs_ack_flags
        params.ar_notify_rs_local_flags[i]
            = reinterpret_cast<uint64_t*>(reinterpret_cast<char*>(params.peer_comm_buffer_ptrs[tpRank])
                + rs_ack_flags_offset + i * sizeof(uint64_t) * 2);
        // now only 8gpus can use hier , so %8 is a magic num
        params.ar_notify_rs_remote_flags[i] = reinterpret_cast<uint64_t*>(
            reinterpret_cast<char*>(params.peer_comm_buffer_ptrs[(tpRank + LP_ALLREDUCE_MAX_RANKS_PER_NUMA) % tpSize])
            + rs_ack_flags_offset + i * sizeof(uint64_t) * 2 + sizeof(uint64_t));

        uint64_t ar_ack_flags_offset = ar_send_flags_offset + LP_ALLREDUCE_MAX_BLOCKS * sizeof(uint64_t);

        params.ar_ack_flags[i] = reinterpret_cast<uint64_t*>(
            reinterpret_cast<char*>(params.peer_comm_buffer_ptrs[tpRank]) + ar_ack_flags_offset + i * sizeof(uint64_t));

        uint64_t ar_notify_ag_flags_offset = ar_ack_flags_offset + LP_ALLREDUCE_MAX_BLOCKS * sizeof(uint64_t);
        params.ar_notify_ag_flags[i]
            = reinterpret_cast<uint64_t*>(reinterpret_cast<char*>(params.peer_comm_buffer_ptrs[tpRank])
                + ar_notify_ag_flags_offset + i * sizeof(uint64_t));

        // partial ar stage done

        // for ag stage
        uint64_t ag_send_flags_offset = ar_notify_ag_flags_offset + LP_ALLREDUCE_MAX_BLOCKS * sizeof(uint64_t);
        params.ag_send_flags[i]
            = reinterpret_cast<uint64_t*>(reinterpret_cast<char*>(params.peer_comm_buffer_ptrs[tpRank])
                + ag_send_flags_offset + i * sizeof(uint64_t));

        // 4 flag in numa,so use fix *4
        uint64_t ag_ack_peer_inside_numa_flags_offset
            = ag_send_flags_offset + LP_ALLREDUCE_MAX_BLOCKS * sizeof(uint64_t);
        params.ag_ack_peer_inside_numa_flags[i]
            = reinterpret_cast<uint64_t*>(reinterpret_cast<char*>(params.peer_comm_buffer_ptrs[tpRank])
                + ag_ack_peer_inside_numa_flags_offset + i * sizeof(uint64_t) * 4);

        for (int j = 0; j < LP_ALLREDUCE_MAX_RANKS_PER_NUMA; ++j)
        {
            params.ag_notify_peer_inside_numa_flags[i * LP_ALLREDUCE_MAX_RANKS_PER_NUMA + j]
                = reinterpret_cast<uint64_t*>(
                    reinterpret_cast<char*>(params.peer_comm_buffer_ptrs[(tpRank / LP_ALLREDUCE_MAX_RANKS_PER_NUMA)
                            * LP_ALLREDUCE_MAX_RANKS_PER_NUMA
                        + j])
                    + ag_ack_peer_inside_numa_flags_offset + i * sizeof(uint64_t) * 4
                    + (tpRank % LP_ALLREDUCE_MAX_RANKS_PER_NUMA) * sizeof(uint64_t));
        }
        // ag stage done
    }

    return params;
}

bool lowPrecisionConfigurationSupported(size_t n_ranks, size_t msg_size)
{
    size_t elts_per_thread = LP_ALLREDUCE_BYTES_PER_LOAD; // assume quant_type size is 1 bytes
    int msg_align = elts_per_thread;
    if (n_ranks <= 4)
    {
        msg_align *= n_ranks;
    }
    return msg_size % msg_align == 0;
}

int32_t max_workspace_size_lowprecision(int32_t tp_size)
{
    // assume quant_type is 1 byte , so we can transfer LP_ALLREDUCE_BYTES_PER_LOAD elts once
    constexpr int32_t REAL_ELTS_PER_BLOCK
        = (LP_ALLREDUCE_WARPSIZE - 1) * LP_ALLREDUCE_BYTES_PER_LOAD * LP_ALLREDUCE_WARP_NUM_PER_BLOCK;
    constexpr int32_t QUANT_ELTS_PER_BLOCK = LP_ALLREDUCE_DEFAULT_BLOCK_SIZE * LP_ALLREDUCE_BYTES_PER_LOAD;

    int32_t buffer_bytes;
    if (tp_size == 8)
    {
        int32_t max_rounds = ((((LP_ALLREDUCE_MAX_ELTS_IN_WORKSPACE - 1) / REAL_ELTS_PER_BLOCK + 1) - 1)
                                 / LP_ALLREDUCE_MAX_RANKS_PER_NUMA / LP_ALLREDUCE_MAX_BLOCKS)
            + 1;
        int32_t max_fence_rounds = ((max_rounds - 1) / LP_ALLREDUCE_BUFFER_DUPLICATE) + 1;
        int32_t quantize_buffer_bytes = max_fence_rounds * LP_ALLREDUCE_MAX_RANKS_PER_NUMA * LP_ALLREDUCE_MAX_BLOCKS
            * LP_ALLREDUCE_BUFFER_DUPLICATE * QUANT_ELTS_PER_BLOCK;
        int32_t comm_buffer_bytes = LP_ALLREDUCE_BUFFER_CHUNKS * LP_ALLREDUCE_BUFFER_DUPLICATE * LP_ALLREDUCE_MAX_BLOCKS
            * LP_ALLREDUCE_HIER_STAGE_NUM * QUANT_ELTS_PER_BLOCK;
        buffer_bytes = quantize_buffer_bytes + comm_buffer_bytes;
    }
    else
    {
        buffer_bytes = (((LP_ALLREDUCE_MAX_ELTS_IN_WORKSPACE / tp_size - 1) / REAL_ELTS_PER_BLOCK) + 1)
            * QUANT_ELTS_PER_BLOCK * tp_size;
    }

    constexpr int32_t HANDSHAKE_FLAG_NUM = 32;
    int32_t flag_bytes = LP_ALLREDUCE_MAX_BLOCKS * HANDSHAKE_FLAG_NUM * sizeof(uint64_t);

    return buffer_bytes + flag_bytes;
}

void customLowPrecisionAllReduce(
    kernels::LowPrecisionAllReduceParams& params, nvinfer1::DataType dataType, cudaStream_t stream)
{
    TLLM_CHECK_WITH_INFO(lowPrecisionConfigurationSupported(params.ranks_per_node, params.elts_total),
        "Low Precision Custom all-reduce configuration unsupported");

    sync_check_cuda_error(stream);

    switch (dataType)
    {
    case nvinfer1::DataType::kFLOAT: lowPrecisionAllReduceDispatchType<float>(params, stream); break;
    case nvinfer1::DataType::kHALF: lowPrecisionAllReduceDispatchType<half>(params, stream); break;
#ifdef ENABLE_BF16
    case nvinfer1::DataType::kBF16: lowPrecisionAllReduceDispatchType<__nv_bfloat16>(params, stream); break;
#endif
    default: TLLM_THROW("Unsupported dataType for customAllReduce");
    }
    sync_check_cuda_error(stream);
}

} // namespace tensorrt_llm::kernels
