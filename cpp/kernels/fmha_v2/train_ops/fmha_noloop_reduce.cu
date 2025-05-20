/*
 * SPDX-FileCopyrightText: Copyright (c) 2011-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: NVIDIA TensorRT Source Code License Agreement
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include "fused_multihead_attention_fprop.h"

inline __device__ float4 ldg128(void const* ptr)
{
    return *static_cast<float4 const*>(ptr);
}

inline __device__ void stg128(void* ptr, float4 const& data)
{
    *static_cast<float4*>(ptr) = data;
}

template <typename T, int THREADS, int HIDDEN_SIZE, int CHUNKS>
__global__ __launch_bounds__(THREADS) void fmha_noloop_reduce_kernel(
    void* __restrict__ out, void const* __restrict__ in, int const* __restrict__ cu_seqlens, int const batch_size)
{

    enum
    {
        BYTES_PER_LDG = 16
    };

    enum
    {
        NUM_ELTS = BYTES_PER_LDG / sizeof(T)
    };

    // One CTA hidden vector for K and V
    enum
    {
        BYTES_PER_ROW = HIDDEN_SIZE * sizeof(T) * 2
    };

    // The stride in bytes in dQKV
    enum
    {
        OUT_STRIDE_BYTES = 3 * HIDDEN_SIZE * sizeof(T)
    };

    // The offset in bytes in dQKV to the dKV part for non-interleaved heads
    enum
    {
        OUT_OFFSET_KV_BYTES = HIDDEN_SIZE * sizeof(T)
    };

    static_assert(BYTES_PER_ROW == HIDDEN_SIZE * 2 * sizeof(T));

    // Size in bytes of the input tile
    enum
    {
        BYTES_PER_TILE = CHUNKS * BYTES_PER_ROW
    };

    enum
    {
        BYTES_PER_CTA = THREADS * BYTES_PER_LDG
    };

    enum
    {
        LDGS = BYTES_PER_ROW / BYTES_PER_CTA
    };

    static_assert(BYTES_PER_CTA * LDGS == BYTES_PER_ROW);

    union Vec_t
    {
        float4 raw;
        T elt[NUM_ELTS];
    };

    // ZERO-OUT invalid positions in dQKV
    int const total = cu_seqlens[batch_size];
    if (blockIdx.x >= total)
    {
        enum
        {
            BYTES_PER_QKV_ROW = 3 * HIDDEN_SIZE * sizeof(T)
        };

        enum
        {
            STGS = BYTES_PER_QKV_ROW / BYTES_PER_LDG
        };

        float4 const zeros = make_float4(0.f, 0.f, 0.f, 0.f);

        char* base_ptr = static_cast<char*>(out) + blockIdx.x * OUT_STRIDE_BYTES;

        for (int tidx = threadIdx.x; tidx < STGS; tidx += THREADS)
        {
            stg128(base_ptr + tidx * BYTES_PER_LDG, zeros);
        }

        return;
    }

    // SETUP
    int const offset_in = blockIdx.x * BYTES_PER_TILE + threadIdx.x * BYTES_PER_LDG;
    char const* ptr_in = static_cast<char const*>(in) + offset_in;

    int const offset_out = blockIdx.x * OUT_STRIDE_BYTES + threadIdx.x * BYTES_PER_LDG;
    char* ptr_out = static_cast<char*>(out) + OUT_OFFSET_KV_BYTES + offset_out;

    // LOAD

    Vec_t local_in[CHUNKS][LDGS];

#pragma unroll
    for (int c = 0; c < CHUNKS; c++)
    {
#pragma unroll
        for (int l = 0; l < LDGS; l++)
        {
            int offset = c * BYTES_PER_ROW + l * BYTES_PER_CTA;
            local_in[c][l].raw = ldg128(ptr_in + offset);
        }
    }

    // UNPACK
    float acc[LDGS][NUM_ELTS];

#pragma unroll
    for (int l = 0; l < LDGS; l++)
    {
#pragma unroll
        for (int e = 0; e < NUM_ELTS; e++)
        {
            acc[l][e] = float(local_in[0][l].elt[e]);
        }
    }

// COMPUTE
#pragma unroll
    for (int c = 1; c < CHUNKS; c++)
    {
#pragma unroll
        for (int l = 0; l < LDGS; l++)
        {
#pragma unroll
            for (int e = 0; e < NUM_ELTS; e++)
            {
                acc[l][e] += float(local_in[c][l].elt[e]);
            }
        }
    }

    // PACK
    Vec_t local_out[LDGS];

#pragma unroll
    for (int l = 0; l < LDGS; l++)
    {
#pragma unroll
        for (int e = 0; e < NUM_ELTS; e++)
        {
            local_out[l].elt[e] = T(acc[l][e]);
        }
    }

// STORE
#pragma unroll
    for (int l = 0; l < LDGS; l++)
    {
        int const offset = l * BYTES_PER_CTA;
        stg128(ptr_out + offset, local_out[l].raw);
    }
}

void fmha_run_noloop_reduce(void* out, void const* in, int const* cu_seqlens, int const hidden_size,
    int const batch_size, int const total, int const num_chunks, cudaStream_t stream)
{

    int const blocks = total;

    if (hidden_size == 1024)
    {

        constexpr int HIDDEN_SIZE = 1024;
        constexpr int THREADS = 256;

        if (num_chunks == 2)
        {
            fmha_noloop_reduce_kernel<half, THREADS, HIDDEN_SIZE, 2>
                <<<blocks, THREADS, 0, stream>>>(out, in, cu_seqlens, batch_size);
        }
        else if (num_chunks == 3)
        {
            fmha_noloop_reduce_kernel<half, THREADS, HIDDEN_SIZE, 3>
                <<<blocks, THREADS, 0, stream>>>(out, in, cu_seqlens, batch_size);
        }
        else
        {
            assert(false && "Unsupported num_chunks");
        }
    }
    else
    {
        assert(false && "Unsupported hidden_size");
    }

    FMHA_CHECK_CUDA(cudaPeekAtLastError());
}
