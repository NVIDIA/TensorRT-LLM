/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "marlin.cuh"

namespace marlin
{

template <int const num_threads, int const num_bits, bool const has_perm, bool is_a_8bit>
__global__ void gptq_marlin_repack_kernel(uint32_t const* __restrict__ b_q_weight_ptr,
    uint32_t const* __restrict__ perm_ptr, uint32_t* __restrict__ out_ptr, int size_k, int size_n)
{
    constexpr int pack_factor = 32 / num_bits;

    constexpr int target_tile_n_size = tile_n_size / (is_a_8bit ? 2 : 1);
    constexpr int target_tile_k_size = tile_k_size * (is_a_8bit ? 2 : 1);
    int k_tiles = size_k / target_tile_k_size;
    int n_tiles = size_n / target_tile_n_size;
    int block_k_tiles = div_ceil(k_tiles, gridDim.x);

    auto start_k_tile = blockIdx.x * block_k_tiles;
    if (start_k_tile >= k_tiles)
    {
        return;
    }

    int finish_k_tile = min(start_k_tile + block_k_tiles, k_tiles);

    // Wait until the next thread tile has been loaded to shared memory.
    auto wait_for_stage = [&]()
    {
        // We only have `stages - 2` active fetches since we are double buffering
        // and can only issue the next fetch when it is guaranteed that the previous
        // shared memory load is fully complete (as it may otherwise be
        // overwritten).
        cp_async_wait<repack_stages - 2>();
        __syncthreads();
    };

    extern __shared__ int4 sh[];

    constexpr int perm_size = target_tile_k_size / 4;

    int4* sh_perm_ptr = sh;
    int4* sh_pipe_ptr = sh_perm_ptr;
    if constexpr (has_perm)
    {
        sh_pipe_ptr += perm_size;
    }

    constexpr int tile_ints = target_tile_k_size / pack_factor;

    constexpr int stage_n_threads = target_tile_n_size / 4;
    constexpr int stage_k_threads = has_perm ? target_tile_k_size : tile_ints;
    constexpr int stage_size = stage_k_threads * stage_n_threads;

    auto load_perm_to_shared = [&](int k_tile_id)
    {
        int first_k_int4 = (k_tile_id * target_tile_k_size) / 4;

        int4 const* perm_int4_ptr = reinterpret_cast<int4 const*>(perm_ptr);

        if (threadIdx.x < perm_size)
        {
            sh_perm_ptr[threadIdx.x] = perm_int4_ptr[first_k_int4 + threadIdx.x];
        }
        __syncthreads();
    };

    auto fetch_to_shared = [&](int pipe, int k_tile_id, int n_tile_id)
    {
        if (n_tile_id >= n_tiles)
        {
            cp_async_fence();
            return;
        }

        int first_n = n_tile_id * target_tile_n_size;

        int4* sh_ptr = sh_pipe_ptr + stage_size * pipe;

        if constexpr (has_perm)
        {
            if (threadIdx.x < stage_size)
            {
                auto k_id = threadIdx.x / stage_n_threads;
                auto n_id = threadIdx.x % stage_n_threads;

                uint32_t const* sh_perm_int_ptr = reinterpret_cast<uint32_t const*>(sh_perm_ptr);

                int src_k = sh_perm_int_ptr[k_id];
                int src_k_packed = src_k / pack_factor;

                cp_async4(&sh_ptr[k_id * stage_n_threads + n_id],
                    reinterpret_cast<int4 const*>(&(b_q_weight_ptr[src_k_packed * size_n + first_n + (n_id * 4)])));
            }
        }
        else
        {
            if (threadIdx.x < stage_size)
            {
                auto k_id = threadIdx.x / stage_n_threads;
                auto n_id = threadIdx.x % stage_n_threads;

                int first_k = k_tile_id * target_tile_k_size;
                int first_k_packed = first_k / pack_factor;

                cp_async4(&sh_ptr[k_id * stage_n_threads + n_id],
                    reinterpret_cast<int4 const*>(
                        &(b_q_weight_ptr[(first_k_packed + k_id) * size_n + first_n + (n_id * 4)])));
            }
        }

        cp_async_fence();
    };

    auto repack_tile = [&](int pipe, int k_tile_id, int n_tile_id)
    {
        if (n_tile_id >= n_tiles)
        {
            return;
        }

        auto warp_id = threadIdx.x / 32;
        auto th_id = threadIdx.x % 32;

        if (warp_id >= 4)
        {
            return;
        }

        int tc_col = th_id / 4;
        int tc_row = (th_id % 4) * (is_a_8bit ? 4 : 2);

        constexpr int tc_offsets[4] = {0, 1, 8, 9};

        int cur_n = (warp_id / (is_a_8bit ? 2 : 1)) * 16 + tc_col;

        constexpr int sh_stride = target_tile_n_size;
        constexpr uint32_t mask = (1 << num_bits) - 1;

        int4* sh_stage_ptr = sh_pipe_ptr + stage_size * pipe;
        uint32_t* sh_stage_int_ptr = reinterpret_cast<uint32_t*>(sh_stage_ptr);

        uint32_t* sh_perm_int_ptr = reinterpret_cast<uint32_t*>(sh_perm_ptr);

        uint32_t vals[8];

        if constexpr (has_perm)
        {
            static_assert(!is_a_8bit);
            for (int i = 0; i < 4; i++)
            {
                int k_idx = tc_row + tc_offsets[i];

                uint32_t src_k = sh_perm_int_ptr[k_idx];
                uint32_t src_k_pos = src_k % pack_factor;

                uint32_t b1_val = sh_stage_int_ptr[k_idx * sh_stride + cur_n];
                uint32_t b1_cur_val = (b1_val >> (src_k_pos * num_bits)) & mask;

                uint32_t b2_val = sh_stage_int_ptr[k_idx * sh_stride + cur_n + 8];
                uint32_t b2_cur_val = (b2_val >> (src_k_pos * num_bits)) & mask;

                vals[i] = b1_cur_val;
                vals[4 + i] = b2_cur_val;
            }
        }
        else
        {
            uint32_t b1_vals[tile_ints];
            uint32_t b2_vals[tile_ints];

#pragma unroll
            for (int i = 0; i < tile_ints; i++)
            {
                if constexpr (is_a_8bit)
                {
                    b1_vals[i] = sh_stage_int_ptr[cur_n + sh_stride * i + (warp_id % 2) * 8];
                }
                else
                {
                    b1_vals[i] = sh_stage_int_ptr[cur_n + sh_stride * i];
                    b2_vals[i] = sh_stage_int_ptr[cur_n + 8 + sh_stride * i];
                }
            }

#pragma unroll
            for (int i = 0; i < 4; i++)
            {
                int cur_elem = tc_row + (is_a_8bit ? i : tc_offsets[i]);
                int cur_int = cur_elem / pack_factor;
                int cur_pos = cur_elem % pack_factor;

                vals[i] = (b1_vals[cur_int] >> (cur_pos * num_bits)) & mask;
                if constexpr (is_a_8bit)
                    vals[4 + i] = (b1_vals[cur_int + tile_ints / 2] >> (cur_pos * num_bits)) & mask;
                else
                    vals[4 + i] = (b2_vals[cur_int] >> (cur_pos * num_bits)) & mask;
            }
        }

        constexpr int tile_size = target_tile_k_size * target_tile_n_size / pack_factor;
        int out_offset = (k_tile_id * n_tiles + n_tile_id) * tile_size;

        // Result of:
        // https://github.com/NVIDIA/FasterTransformer/blob/main/src/fastertransformer/cutlass_extensions/include/cutlass_extensions/interleaved_numeric_conversion.h
        if constexpr (!is_a_8bit && num_bits == 4)
        {
            int pack_idx[8] = {0, 2, 4, 6, 1, 3, 5, 7};

            uint32_t res = 0;
#pragma unroll
            for (int i = 0; i < 8; i++)
            {
                res |= vals[pack_idx[i]] << (i * 4);
            }

            out_ptr[out_offset + th_id * 4 + warp_id] = res;
        }
        else if constexpr (is_a_8bit && num_bits == 4)
        {
            int pack_idx[8] = {0, 4, 1, 5, 2, 6, 3, 7};

            uint32_t res = 0;
#pragma unroll
            for (int i = 0; i < 8; i++)
            {
                res |= vals[pack_idx[i]] << (i * 4);
            }

            out_ptr[out_offset + th_id * 4 + warp_id] = res;
        }
        else
        {
            constexpr int pack_idx[4] = {0, 2, 1, 3};

            uint32_t res1 = 0;
            uint32_t res2 = 0;
#pragma unroll
            for (int i = 0; i < 4; i++)
            {
                const int ii = is_a_8bit ? i : pack_idx[i];
                res1 |= vals[ii] << (i * 8);
                res2 |= vals[4 + ii] << (i * 8);
            }

            out_ptr[out_offset + th_id * 8 + (warp_id * 2) + 0] = res1;
            out_ptr[out_offset + th_id * 8 + (warp_id * 2) + 1] = res2;
        }
    };

    auto start_pipes = [&](int k_tile_id, int n_tile_id)
    {
#pragma unroll
        for (int pipe = 0; pipe < repack_stages - 1; pipe++)
        {
            fetch_to_shared(pipe, k_tile_id, n_tile_id + pipe);
        }

        wait_for_stage();
    };
#pragma unroll
    for (int k_tile_id = start_k_tile; k_tile_id < finish_k_tile; k_tile_id++)
    {
        int n_tile_id = 0;

        if constexpr (has_perm)
        {
            load_perm_to_shared(k_tile_id);
        }

        start_pipes(k_tile_id, n_tile_id);

        while (n_tile_id < n_tiles)
        {
#pragma unroll
            for (int pipe = 0; pipe < repack_stages; pipe++)
            {
                fetch_to_shared(
                    (pipe + repack_stages - 1) % repack_stages, k_tile_id, n_tile_id + pipe + repack_stages - 1);
                repack_tile(pipe, k_tile_id, n_tile_id + pipe);
                wait_for_stage();
            }
            n_tile_id += repack_stages;
        }
    }
}

} // namespace marlin

#define CALL_IF(NUM_BITS, HAS_PERM, IS_A_8BIT)                                                                         \
    else if (num_bits == NUM_BITS && has_perm == HAS_PERM && is_a_8bit == IS_A_8BIT)                                   \
    {                                                                                                                  \
        cudaFuncSetAttribute(marlin::gptq_marlin_repack_kernel<marlin::repack_threads, NUM_BITS, HAS_PERM, IS_A_8BIT>, \
            cudaFuncAttributeMaxDynamicSharedMemorySize, max_shared_mem);                                              \
        marlin::gptq_marlin_repack_kernel<marlin::repack_threads, NUM_BITS, HAS_PERM, IS_A_8BIT>                       \
            <<<blocks, marlin::repack_threads, max_shared_mem, stream>>>(                                              \
                b_q_weight_ptr, perm_ptr, out_ptr, size_k, size_n);                                                    \
    }

namespace marlin_nvfp4
{

void gptq_marlin_repack_dispatch(uint32_t const* b_q_weight_ptr, uint32_t const* perm_ptr, uint32_t* out_ptr,
    int size_k, int size_n, int num_bits, bool has_perm, bool is_a_8bit, cudaStream_t stream)
{

    int blocks;
    int dev;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&blocks, cudaDevAttrMultiProcessorCount, dev);

    int max_shared_mem = 0;
    cudaDeviceGetAttribute(&max_shared_mem, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);

    if (false)
    {
    }
    CALL_IF(4, false, false)
    CALL_IF(4, true, false)
    CALL_IF(8, false, false)
    CALL_IF(8, true, false)
    CALL_IF(4, false, true)
    CALL_IF(8, false, true)
}

} // namespace marlin_nvfp4

#undef CALL_IF
