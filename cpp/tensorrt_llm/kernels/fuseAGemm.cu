/*
 * Copyright (c) 2019-2024, NVIDIA CORPORATION.  All rights reserved.
 * Copyright (c) 2021, NAVER Corp.  Authored by CLOVA.
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

#include <cstdio>
#include <tuple>

#include "cuda.h"
#include "cuda_bf16.h"
#include "cuda_runtime.h"
#include "tensorrt_llm/common/envUtils.h"
#include "tensorrt_llm/kernels/fuseAGemm.h"
using namespace tensorrt_llm::common;
using bf16_t = __nv_bfloat16;

namespace tensorrt_llm::kernels
{
__device__ void hmma_16_8_16_f32acc_bf16ab(
    float (&d_reg)[4], const bf16_t (&a_reg)[8], const bf16_t (&b_reg)[4], float const (&c_reg)[4])
{
    uint32_t a0 = *reinterpret_cast<uint32_t const*>(a_reg + 0);
    uint32_t a1 = *reinterpret_cast<uint32_t const*>(a_reg + 2);
    uint32_t a2 = *reinterpret_cast<uint32_t const*>(a_reg + 4);
    uint32_t a3 = *reinterpret_cast<uint32_t const*>(a_reg + 6);
    uint32_t b0 = *reinterpret_cast<uint32_t const*>(b_reg + 0);
    uint32_t b1 = *reinterpret_cast<uint32_t const*>(b_reg + 2);
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,  %1,  %2,  %3},"
        "{%4,  %5,  %6,  %7},"
        "{%8,  %9},"
        "{%10, %11, %12, %13};\n"
        : "=f"(d_reg[0]), "=f"(d_reg[1]), "=f"(d_reg[2]), "=f"(d_reg[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "f"(d_reg[0]), "f"(d_reg[1]), "f"(d_reg[2]),
        "f"(d_reg[3]));
}

extern "C"
{
    __device__ uint32_t __nvvm_get_smem_pointer(void*);
}

__device__ void ldgsts_128(void const* gPtr, void* sPtr, uint32_t pred)
{
    if (pred)
    {
        uint32_t smemPtrAsUint32 = __nvvm_get_smem_pointer(sPtr);
        asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(smemPtrAsUint32), "l"(gPtr), "n"(16));
    }
}

__device__ void ldgsts_commit()
{
    asm volatile("cp.async.commit_group;\n" ::);
}

template <int allowedInFlightBunchCnt>
__device__ void ldgsts_wait()
{
    asm volatile("cp.async.wait_group %0;\n" ::"n"(allowedInFlightBunchCnt));
}

__device__ std::tuple<int, int> hmma_c_tv2mn(int t_idx, int v_idx)
{
    int linear_idx = (t_idx % 4) * 32 + (t_idx / 4) * 1 + (v_idx % 2) * 16 + (v_idx / 2) * 8;
    int m_idx = linear_idx % 16;
    int n_idx = linear_idx / 16;
    return std::make_tuple(m_idx, n_idx);
}

__device__ void ldsm_x4(void* smem_ptr, uint32_t* reg_ptr)
{
    asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(reg_ptr[0]), "=r"(reg_ptr[1]), "=r"(reg_ptr[2]), "=r"(reg_ptr[3])
                 : "r"(__nvvm_get_smem_pointer(smem_ptr)));
}

template <class T>
__device__ T* apply_swizzle_343_on_ptr(T* addr)
{
    uint64_t mask = reinterpret_cast<uint64_t>(addr) & 0x00000000000000380;
    mask = mask >> 3;
    return reinterpret_cast<T*>(reinterpret_cast<uint64_t>(addr) ^ mask);
    // return addr;
}

template <class Type, class T>
__device__ T apply_swizzle_343_on_elem(T elem_offset)
{
    T byte_offset = elem_offset / sizeof(Type);
    T mask = byte_offset & 0x380;
    mask = mask >> 3;
    return (byte_offset ^ mask) * sizeof(Type);
}

template <class Type>
__device__ int apply_swizzle_343_on_elem_row_col(int row_idx_, int col_idx_)
{
    // return col_idx;
    uint32_t row_idx = *reinterpret_cast<uint32_t*>(&row_idx_);
    uint32_t col_idx = *reinterpret_cast<uint32_t*>(&col_idx_);
    // uint32_t row_idx_original = row_idx;
    // uint32_t col_idx_original = col_idx;
    row_idx = row_idx % 8;
    row_idx = row_idx * (16 / sizeof(Type));
    col_idx = col_idx ^ row_idx;
    // printf("tid: %d, [%d, %d] -> [%d, %d]\n", threadIdx.x, row_idx_original, col_idx_original, row_idx, col_idx);
    return *reinterpret_cast<int*>(&col_idx);
}

__device__ uint32_t elect_one_sync()
{
    uint32_t pred = 0;
    uint32_t laneid = 0;
    asm volatile(
        "{\n"
        ".reg .b32 %%rx;\n"
        ".reg .pred %%px;\n"
        "     elect.sync %%rx|%%px, %2;\n"
        "@%%px mov.s32 %1, 1;\n"
        "     mov.s32 %0, %%rx;\n"
        "}\n"
        : "+r"(laneid), "+r"(pred)
        : "r"(0xFFFFFFFF));
    return pred;
}

__device__ void initialize_barrier(uint64_t* smem_barrier, // 64 bits user-manged barrier in smem
    int thread_count = 1)                                  // Thread count expected to arrive/wait on this barrier
{
    uint32_t smem_int_ptr = __nvvm_get_smem_pointer(smem_barrier);
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(smem_int_ptr), "r"(thread_count));
}

// Barrier wait
__device__ void wait_barrier(uint64_t* smem_barrier, // 64 bits user-manged barrier in smem
    int phase_bit)                                   // Current phase bit the barrier waiting to flip
{
    asm volatile(".pragma \"set knob DontInsertYield\";\n" : : : "memory"); // {$nv-internal-release}

    uint32_t smem_int_ptr = __nvvm_get_smem_pointer(smem_barrier);
    asm volatile(
        "{\n"
        ".reg .pred                P1;\n"
        "LAB_WAIT:\n"
        "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n"
        "@P1                       bra DONE;\n"
        "bra                   LAB_WAIT;\n"
        "DONE:\n"
        "}\n" ::"r"(smem_int_ptr),
        "r"(phase_bit));

    asm volatile(".pragma \"reset knob DontInsertYield\";\n" : : : "memory"); // {$nv-internal-release}
}

__device__ bool try_wait_barrier(uint64_t* smem_ptr, int phase_bit)
{
    uint32_t wait_complete;
    uint32_t smem_int_ptr = __nvvm_get_smem_pointer(smem_ptr);
    asm volatile(
        "{\n\t"
        ".reg .pred P1; \n\t"
        "mbarrier.try_wait.parity.shared::cta.b64 P1, [%1], %2; \n\t"
        "selp.b32 %0, 1, 0, P1; \n\t"
        "}"
        : "=r"(wait_complete)
        : "r"(smem_int_ptr), "r"(phase_bit));
    return static_cast<bool>(wait_complete);
}

// Barrier arrive
__device__ void arrive_barrier(uint64_t* smem_barrier) // 64 bits user-manged barrier in smem
{
    uint32_t smem_int_ptr = __nvvm_get_smem_pointer(smem_barrier);
    asm volatile(
        "{\n"
        ".reg .b64 state; \n"
        "mbarrier.arrive.shared::cta.b64   state, [%0];\n"
        "}\n" ::"r"(smem_int_ptr));
}

__device__ void ldgsts_arrive(uint64_t* smem_barrier)
{
    uint32_t smem_int_ptr = __nvvm_get_smem_pointer(smem_barrier);
    asm volatile("cp.async.mbarrier.arrive.noinc.shared.b64 [%0];" : : "r"(smem_int_ptr));
}

template <int gemm_k, int tile_m, int tile_n, int tile_k, int stage_cnt, int gemm_m>
struct GmemLoaderA
{
    static constexpr int elem_bytes = 2;
    static constexpr int vec_bytes = 16;
    static constexpr int vec_elems = vec_bytes / elem_bytes;
    static constexpr int thread_cnt = 64;
    static_assert((tile_m * tile_k) % (vec_elems * thread_cnt) == 0);
    static constexpr int a_inst_cnt_per_iter = (tile_m * tile_k) / (vec_elems * thread_cnt);
    static_assert(gemm_k % tile_k == 0);
    static constexpr int k_iter_cnt = gemm_k / tile_k;

    __device__ GmemLoaderA(bf16_t const* gmem_a_local_, bf16_t* smem_a_, uint64_t* smem_barrier_)
        : gmem_a(gmem_a_local_)
        , smem_a(smem_a_)
        , smem_barrier(smem_barrier_)
        , local_tid(threadIdx.x % thread_cnt)
    {
    }

    __device__ void prepare()
    {
// swizzle, that's what we want.
#pragma unroll
        for (int i = 0; i < a_inst_cnt_per_iter; i++)
        {
            int linear_idx = local_tid * vec_elems + i * thread_cnt * vec_elems;
            int m_idx = linear_idx / tile_k;
            int k_idx = linear_idx % tile_k;
            k_idx = apply_swizzle_343_on_elem_row_col<bf16_t>(m_idx, k_idx);
            a_smem_offsets[i] = m_idx * tile_k + k_idx;
        }
    }

    __device__ void issue_mainloop()
    {
#pragma unroll 1
        for (int loop_idx = 0; loop_idx < k_iter_cnt; loop_idx++)
        {
            if (need_wait)
            {
                wait_barrier(smem_barrier + 1 + stage_idx * 2, phase_bit);
            }
            int next_stage_idx = stage_idx + 1;
            int next_phase_bit = next_stage_idx == stage_cnt ? phase_bit ^ 1 : phase_bit;
            next_stage_idx = next_stage_idx == stage_cnt ? 0 : next_stage_idx;
            if (loop_idx != k_iter_cnt - 1)
            {
                need_wait = !try_wait_barrier(smem_barrier + 1 + next_stage_idx * 2, next_phase_bit);
            }

#pragma unroll
            for (int i = 0; i < a_inst_cnt_per_iter; i++)
            {
                int smem_offset = a_smem_offsets[i];
                bf16_t* smem_ptr_this_iter = smem_a + stage_idx * tile_m * tile_k + smem_offset;
                int linear_idx = local_tid * vec_elems + i * thread_cnt * vec_elems;
                int m_idx = linear_idx / tile_k;
                int k_idx = linear_idx % tile_k;
                if (m_idx < gemm_m && k_idx < gemm_k)
                {
                    int gmem_offset = m_idx * gemm_k + k_idx;
                    bf16_t const* gmem_ptr_this_iter = gmem_a + gmem_offset;
                    ldgsts_128(gmem_ptr_this_iter, smem_ptr_this_iter, true);
                }
            }
            ldgsts_arrive(smem_barrier + stage_idx * 2);
            asm volatile(".pragma \"next knob FenceInterference\";\n" : : : "memory");

            stage_idx = next_stage_idx;
            phase_bit = next_phase_bit;
            gmem_a += tile_k;
        }
    }

    bf16_t const* gmem_a;
    bf16_t* smem_a;
    uint64_t* smem_barrier;
    int local_tid;
    int stage_idx = 0;
    int phase_bit = 1;
    bool need_wait = true;

    // per smem_stage, store with swizzle information
    int a_smem_offsets[a_inst_cnt_per_iter];
};

template <int gemm_k, int tile_m, int tile_n, int tile_k, int stage_cnt, int gemm_n>
struct GmemLoaderB
{
    static constexpr int elem_bytes = 2;
    static constexpr int vec_bytes = 16;
    static constexpr int vec_elems = vec_bytes / elem_bytes;
    static constexpr int thread_cnt = 64;
    static_assert((tile_n * tile_k) % (vec_elems * thread_cnt) == 0);
    static constexpr int b_inst_cnt_per_iter = (tile_n * tile_k) / (vec_elems * thread_cnt);
    static_assert(gemm_k % tile_k == 0);
    static constexpr int k_iter_cnt = gemm_k / tile_k;

    __device__ GmemLoaderB(bf16_t const* gmem_b_local_, bf16_t* smem_b_, uint64_t* smem_barrier_)
        : gmem_b(gmem_b_local_)
        , smem_b(smem_b_)
        , smem_barrier(smem_barrier_)
        , local_tid(threadIdx.x % thread_cnt)
    {
    }

    __device__ void prepare()
    {
// swizzle, that's what we want.
#pragma unroll
        for (int i = 0; i < b_inst_cnt_per_iter; i++)
        {
            int linear_idx = local_tid * vec_elems + i * thread_cnt * vec_elems;
            int n_idx = linear_idx / tile_k;
            int k_idx = linear_idx % tile_k;
            k_idx = apply_swizzle_343_on_elem_row_col<bf16_t>(n_idx, k_idx);
            b_smem_offsets[i] = n_idx * tile_k + k_idx;
        }
    }

    __device__ void issue_mainloop()
    {
        asm volatile("griddepcontrol.wait;");
#pragma unroll 1
        for (int loop_idx = 0; loop_idx < k_iter_cnt; loop_idx++)
        {
            if (need_wait)
            {
                wait_barrier(smem_barrier + 1 + stage_idx * 2, phase_bit);
            }
            int next_stage_idx = stage_idx + 1;
            int next_phase_bit = next_stage_idx == stage_cnt ? phase_bit ^ 1 : phase_bit;
            next_stage_idx = next_stage_idx == stage_cnt ? 0 : next_stage_idx;
            if (loop_idx != k_iter_cnt - 1)
            {
                need_wait = !try_wait_barrier(smem_barrier + 1 + next_stage_idx * 2, next_phase_bit);
            }
#pragma unroll
            for (int i = 0; i < b_inst_cnt_per_iter; i++)
            {
                int smem_offset = b_smem_offsets[i];
                bf16_t* smem_ptr_this_iter = smem_b + stage_idx * tile_n * tile_k + smem_offset;
                int linear_idx = local_tid * vec_elems + i * thread_cnt * vec_elems;
                int n_idx = linear_idx / tile_k;
                int k_idx = linear_idx % tile_k;
                if (n_idx < gemm_n && k_idx < gemm_k)
                {
                    int gmem_offset = n_idx * gemm_k + k_idx;
                    bf16_t const* gmem_ptr_this_iter = gmem_b + gmem_offset;
                    ldgsts_128(gmem_ptr_this_iter, smem_ptr_this_iter, true);
                }
            }
            ldgsts_arrive(smem_barrier + stage_idx * 2);
            asm volatile(".pragma \"next knob FenceInterference\";\n" : : : "memory"); // internal

            stage_idx = next_stage_idx;
            phase_bit = next_phase_bit;
            gmem_b += tile_k;
        }
    }

    bf16_t const* gmem_b;
    bf16_t* smem_b;
    uint64_t* smem_barrier;
    int local_tid;
    int stage_idx = 0;
    int phase_bit = 1;
    bool need_wait = true;

    // per smem_stage, store with swizzle information
    int b_smem_offsets[b_inst_cnt_per_iter];
};

template <int gemm_m, int gemm_n, int gemm_k, int tile_m, int tile_n, int tile_k, int stage_cnt>
struct MmaComputer
{
    static constexpr int elem_bytes = 2;
    static constexpr int thread_cnt = 128;
    static_assert(gemm_k % tile_k == 0);
    static_assert(tile_k % (thread_cnt / 32) == 0);
    static constexpr int per_warp_tile_k = tile_k / (thread_cnt / 32);
    static constexpr int k_iter_cnt = gemm_k / tile_k;
    static constexpr int k_phase_cnt = per_warp_tile_k / 16;
    static constexpr int m_iter_cnt = (tile_m + 15) / 16;
    static constexpr int n_iter_cnt = (tile_n + 7) / 8; // Possible to have non-1 n_iter_cnt for ab_swap m16 case.
    static_assert(m_iter_cnt == 1);
    static_assert(n_iter_cnt == 1 || n_iter_cnt == 2);

    __device__ MmaComputer(bf16_t* gmem_c_, bf16_t* smem_a_, bf16_t* smem_b_, uint64_t* smem_barrier_, int warp_idx_,
        int block_m_, int block_n_)
        : gmem_c(gmem_c_)
        , smem_a(smem_a_)
        , smem_b(smem_b_)
        , smem_barrier(smem_barrier_)
        , warp_idx(warp_idx_ - (thread_cnt / 32))
        , block_m(block_m_)
        , block_n(block_n_)
    {
    }

private:
    __device__ constexpr int internal_b_atom_func(int tid)
    {
        if constexpr (tile_n < 8)
        {
            return (tid % tile_n) + ((tid % 8) / tile_n * 0) + tid / 8 * 8 * tile_n;
        }
        else
        {
            return (tid % 8)
                + ((tid % 32) / 8
                    * (tile_n * 8)) /* + (tid / 32 * 0) handled by outer code... This is why we need CuTe.*/;
        }
    }

public:
    __device__ void prepare()
    {
        /*
        A: (t) -> (16, 16) -> (tile_m, tile_k) which tile_m === 16, tile_k % 64 == 0
                  (16,2):(1,128) on (16,16)
           (t, k_iter) -> (tile_m, tile_k) will be: (16,2,k_iter):(1,128,256)

        // This is awful!!
        B: (t) -> (8, 32) -> (tile_n, tile_k) which tile_n could be [1,2,4,8,16] and tile_k % 64 == 0
                  (8,4):(1,64) on (8,32)
               or ((tile_n,8/tile_n),4):((1,0),tile_n*8) on (tile_n,32) when tile_n < 8
               or (8,4,2):((1,tile_n*8,0) on (tile_n,32) when tile_n > 8      Awful Awful Awful
           (t, iter) -> (tile_n, tile_k) will be: (atom_32_shape,k_iter/2,m_iter):(atom_32_stride,tile_n*32,8)
        */

#pragma unroll
        for (int i = 0; i < k_phase_cnt; i++)
        {
            int linear_idx = (lane_idx % 16) + (lane_idx / 16) * 128 + i * 256;
            int m_idx = linear_idx % tile_m;
            int k_idx = linear_idx / tile_m + warp_k_offset_in_tile_k;
            // printf("tid: %d ldsm a idx: [%d, %d]\n", threadIdx.x % 128, m_idx, k_idx);
            k_idx = apply_swizzle_343_on_elem_row_col<bf16_t>(m_idx, k_idx);
            a_smem_offsets[0][i] = m_idx * tile_k + k_idx;
        }
#pragma unroll
        for (int n_iter_idx = 0; n_iter_idx < n_iter_cnt; n_iter_idx++)
        {
#pragma unroll
            for (int i = 0; i < k_phase_cnt; i += 2)
            { // Special i+=2 for B.
                int linear_idx = internal_b_atom_func(lane_idx) + i * tile_n * 16 + n_iter_idx * 8;
                int n_idx = linear_idx % tile_n;
                int k_idx = linear_idx / tile_n + warp_k_offset_in_tile_k;
                // printf("tid: %d ldsm b idx: [%d, %d]\n", threadIdx.x % 128, n_idx, k_idx);
                k_idx = apply_swizzle_343_on_elem_row_col<bf16_t>(n_idx, k_idx);
                b_smem_offsets[n_iter_idx][i] = n_idx * tile_k + k_idx;
            }
        }
    }

    __device__ void issue_mainloop()
    {
#pragma unroll 1
        for (int loop_idx = 0; loop_idx < k_iter_cnt; loop_idx++)
        {
            wait_barrier(smem_barrier + 0 + stage_idx * 2, phase_bit);

#pragma unroll
            for (int i = 0; i < k_phase_cnt; i++)
            {
                int smem_offset = a_smem_offsets[0][i];
                bf16_t* smem_ptr_this_iter = smem_a + stage_idx * tile_m * tile_k + smem_offset;
                ldsm_x4(smem_ptr_this_iter, reinterpret_cast<uint32_t*>(a_reg[0][i]));
            }

#pragma unroll
            for (int n_iter_idx = 0; n_iter_idx < n_iter_cnt; n_iter_idx++)
            {
#pragma unroll
                for (int i = 0; i < k_phase_cnt; i += 2)
                {
                    int smem_offset = b_smem_offsets[n_iter_idx][i];
                    bf16_t* smem_ptr_this_iter = smem_b + stage_idx * tile_n * tile_k + smem_offset;
                    ldsm_x4(smem_ptr_this_iter, reinterpret_cast<uint32_t*>(b_reg[n_iter_idx][i]));
                }
            }

#pragma unroll
            for (int k_iter_idx = 0; k_iter_idx < k_phase_cnt; k_iter_idx++)
            {
#pragma unroll
                for (int n_iter_idx = 0; n_iter_idx < n_iter_cnt; n_iter_idx++)
                {
                    hmma_16_8_16_f32acc_bf16ab(acc_reg[0][n_iter_idx], a_reg[0][k_iter_idx],
                        b_reg[n_iter_idx][k_iter_idx], acc_reg[0][n_iter_idx]);
                }
            }
            asm volatile(".pragma \"next knob FenceInterference\";\n" : : : "memory"); // internal
            arrive_barrier(smem_barrier + 1 + stage_idx * 2);
            stage_idx += 1;
            phase_bit = stage_idx == stage_cnt ? phase_bit ^ 1 : phase_bit;
            stage_idx = stage_idx == stage_cnt ? 0 : stage_idx;
        }
    }

    __device__ void epi()
    {
        asm volatile("bar.sync %0, %1;" : : "r"(1), "r"(thread_cnt));
        // reorganize the acc_reg
        constexpr int thread_m = 2;
        constexpr int thread_n = 2 * n_iter_cnt;
        constexpr int cta_mma_n = n_iter_cnt * 8;
        float acc_reg_reorg[thread_m][thread_n];

        // (m,n) -> (v)
        for (int i = 0; i < thread_m; i++)
        {
            for (int j = 0; j < thread_n; j++)
            {
                acc_reg_reorg[i][j] = acc_reg[0][j / 2][(j % 2) + (i * 2)];
            }
        }
        /*
        # //                 4 or 2        2       8 or 16
        # smem_c layout: ((group_rows, group_cnt), cta_mma_n) : ((cta_mma_n, 32+group_cnt), 1)
        def func_3(cta_mma_n, m, n):
            group_rows = 32 // cta_mma_n
            group_cnt = 2
            return ((m % group_rows * cta_mma_n) + (m // group_rows * (32 + group_cnt)) + n) % 32
        */

        // 4 x cosize(smem_c_layout)
        float* smem_c = reinterpret_cast<float*>(smem_a);
        // coord -> index
        auto smem_c_index_func = [&](int m_idx, int n_idx)
        {
            int group_rows = 32 / cta_mma_n;
            int group_cnt = 2;
            return (m_idx % group_rows * cta_mma_n) + (m_idx / group_rows * (32 + group_cnt)) + n_idx;
        };
        constexpr int cosize_smem_c = ((tile_m * cta_mma_n) / 32) * (32 + 2);

// This should be optimized to STS.64 but can not be STS.128 due to the bank index.
#pragma unroll
        for (int m_idx_thread = 0; m_idx_thread < thread_m; m_idx_thread++)
        {
#pragma unroll
            for (int n_idx_thread = 0; n_idx_thread < thread_n; n_idx_thread++)
            {
                int m_idx = (lane_idx / 4) + m_idx_thread * 8;
                int n_idx = ((lane_idx % 4) * 2) + (n_idx_thread % 2) + (n_idx_thread / 2) * 8;
                // printf("tid: %d, [%d, %d]\n", threadIdx.x % 128, m_idx, n_idx);
                smem_c[cosize_smem_c * warp_idx + smem_c_index_func(m_idx, n_idx)]
                    = acc_reg_reorg[m_idx_thread][n_idx_thread];
            }
        }
        asm volatile("bar.sync %0, %1;" : : "r"(1), "r"(thread_cnt));

        if (warp_idx == 0)
        {
            constexpr int final_acc_reg_cnt = (tile_m * tile_n + 31) / 32;
            float acc_final[final_acc_reg_cnt]{};

#pragma unroll
            for (int reg_idx = 0; reg_idx < final_acc_reg_cnt; reg_idx++)
            {
                int linear_idx = reg_idx * 32 + lane_idx;
                int m_idx = linear_idx % tile_m;
                int n_idx = linear_idx / tile_m;
                acc_final[reg_idx] += smem_c[smem_c_index_func(m_idx, n_idx) + 0 * cosize_smem_c]
                    + smem_c[smem_c_index_func(m_idx, n_idx) + 1 * cosize_smem_c]
                    + smem_c[smem_c_index_func(m_idx, n_idx) + 2 * cosize_smem_c]
                    + smem_c[smem_c_index_func(m_idx, n_idx) + 3 * cosize_smem_c];
            }

#pragma unroll
            for (int reg_idx = 0; reg_idx < final_acc_reg_cnt; reg_idx++)
            {
                int linear_idx = reg_idx * 32 + lane_idx;
                int m_idx = linear_idx % tile_m;
                int n_idx = linear_idx / tile_m;
                // printf("lane_id: %d, [%d, %d]\n", lane_idx, m_idx, n_idx);
                // if (m)
                // if (m_idx < tile_m && n_idx < tile_n)
                int g_m_idx = block_m * tile_m + m_idx;
                int g_n_idx = block_n * tile_n + n_idx;
                if (g_m_idx < gemm_m && g_n_idx < gemm_n)
                {
                    gmem_c[g_n_idx * gemm_m + g_m_idx] = acc_final[reg_idx];
                }
            }
        }
    }

    bf16_t* gmem_c;
    bf16_t* smem_a;
    bf16_t* smem_b;
    uint64_t* smem_barrier;
    int warp_idx;
    int block_m;
    int block_n;
    int stage_idx = 0;
    int phase_bit = 0;
    int lane_idx = threadIdx.x % 32;
    int warp_k_offset_in_tile_k = warp_idx * per_warp_tile_k;

    int a_smem_offsets[m_iter_cnt][k_phase_cnt];
    int b_smem_offsets[n_iter_cnt][k_phase_cnt];

    bf16_t a_reg[m_iter_cnt][k_phase_cnt][8];
    bf16_t b_reg[n_iter_cnt][k_phase_cnt][4];
    float acc_reg[m_iter_cnt][n_iter_cnt][4]{};
};

// AB swapped, kernel is k-major, k-major, m-major
template <int batch_size, int gemm_m, int gemm_n, int gemm_k, int tile_m, int tile_n, int tile_k, int stage_cnt>
__global__ __launch_bounds__(256, 1) void fuse_a_gemm_kernel(bf16_t* output, bf16_t const* mat_a, bf16_t const* mat_b)
{
    asm volatile(".pragma \"global knob DisableImplicitMemDesc\";\n" // internal
                 ::
                     : "memory");
    asm volatile(".pragma \"global knob sectorpromotion=128\";\n" // internal
                 ::
                     : "memory");
    asm volatile(" .pragma \"global knob DisableWar_SW2549067\";\n");                     // internal
    asm volatile(".pragma \"set knob SchedMemNoAlias=LDS+LDSM+LDGSTS\";\n" ::: "memory"); // internal

    constexpr int load_thread_cnt = 128;
    constexpr int compute_thread_cnt = 128;
    constexpr int thread_cnt = load_thread_cnt + compute_thread_cnt;
    (void) thread_cnt;
    static_assert(gemm_m % 16 == 0);
    static_assert(gemm_n <= 16);
    static_assert(gemm_k % tile_k == 0);
    static_assert(gemm_m % tile_m == 0);
    // static_assert(gemm_n % tile_n == 0);
    static_assert(tile_k == 128 || tile_k == 256 || tile_k == 512
        || tile_k == 1024); // tile_k must be larger than 64 since 4 warp splitK.
    static_assert(tile_m == 16);
    constexpr int g2s_vec_bytes = 16;
    constexpr int a_elem_bytes = 2;
    constexpr int b_elem_bytes = 2;
    // constexpr int c_elem_bytes = 2;
    static_assert((tile_m * a_elem_bytes + tile_n * b_elem_bytes) * tile_k * stage_cnt <= 225 * 1024);
    static_assert((tile_m * tile_k * a_elem_bytes) % (load_thread_cnt * g2s_vec_bytes) == 0);
    static_assert((tile_n * tile_k * b_elem_bytes) % (load_thread_cnt * g2s_vec_bytes) == 0);

    extern __shared__ char smem[];
    uint64_t* smem_barrier = reinterpret_cast<uint64_t*>(smem); // producer,consumer; producer,consumer; ...
    bf16_t* smem_a = reinterpret_cast<bf16_t*>(smem + (stage_cnt * 8 * 2 + 1024) / 1024 * 1024);
    bf16_t* smem_b = smem_a + tile_m * tile_k * stage_cnt;
    // bf16_t* smem_c = smem_a;

    int cta_m_idx = tile_m * blockIdx.x;
    int cta_n_idx = tile_n * blockIdx.y;
    bf16_t const* gmem_a_local = mat_a + cta_m_idx * gemm_k;
    bf16_t const* gmem_b_local = mat_b + cta_n_idx * gemm_k;

    int warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
    if (warp_idx == 4)
    {
        for (int i = 0; i < stage_cnt; i++)
        {
            initialize_barrier(smem_barrier + i * 2 + 0, load_thread_cnt);    // producer
            initialize_barrier(smem_barrier + i * 2 + 1, compute_thread_cnt); // consumer
        }
    }
    __syncthreads();

    if (warp_idx < 2)
    {
        GmemLoaderA<gemm_k, tile_m, tile_n, tile_k, stage_cnt, gemm_m> a_loader(gmem_a_local, smem_a, smem_barrier);
        a_loader.prepare();
        a_loader.issue_mainloop();
    }
    else if (warp_idx < 4)
    {
        GmemLoaderB<gemm_k, tile_m, tile_n, tile_k, stage_cnt, gemm_n> b_loader(gmem_b_local, smem_b, smem_barrier);
        b_loader.prepare();
        b_loader.issue_mainloop();
    }
    else
    {
        MmaComputer<gemm_m, gemm_n, gemm_k, tile_m, tile_n, tile_k, stage_cnt> mma_computer(
            output, smem_a, smem_b, smem_barrier, warp_idx, blockIdx.x, blockIdx.y);
        mma_computer.prepare();
        mma_computer.issue_mainloop();
        mma_computer.epi();
    }
    // asm volatile("griddepcontrol.launch_dependents;");
}

constexpr int button_pow_of_two(int x)
{ // 移除参数前的constexpr
    if (x <= 0)
    {
        return -1;
    }
    int n = x - 1; // 移除内部变量前的constexpr
    int n1 = n | (n >> 1);
    int n2 = n1 | (n1 >> 2);
    int n3 = n2 | (n2 >> 4);
    int n4 = n3 | (n3 >> 8);
    int n5 = n4 | (n4 >> 16);
    return n5 + 1;
}

template <typename T, int kNumTokens, int kHdIn, int kHdOut>
void invokeFuseAGemm(T* output, T const* mat_a, T const* mat_b, cudaStream_t const stream)
{
    constexpr int gemm_m = kHdOut;     // 2112
    constexpr int gemm_n = kNumTokens; // 4
    constexpr int gemm_k = kHdIn;      // 7168
    constexpr int batch_size = 1;
    std::swap(mat_a, mat_b);
    constexpr int tile_m = 16;
    constexpr int tile_n = gemm_n & (gemm_n - 1) == 0 ? gemm_n : button_pow_of_two(gemm_n);
    constexpr int tile_k = std::max(256, 1024 / tile_n);           // 256
    constexpr int max_stage_cnt = 1024 * 225 / ((tile_m + tile_n) * tile_k * sizeof(bf16_t));
    constexpr int k_iter_cnt = gemm_k / tile_k;                    // 7168 / 256 = 28
    constexpr int stage_cnt
        = k_iter_cnt > max_stage_cnt ? max_stage_cnt : k_iter_cnt; // possible tunable for smallK > 1 wave n. // 22
    int cta_m_cnt = (gemm_m + tile_m - 1) / tile_m;
    int cta_n_cnt = (gemm_n + tile_n - 1) / tile_n;
    constexpr int barrier_bytes = (stage_cnt * 16 + 1023) / 1024 * 1024; // 4096
    constexpr int smem_bytes = ((tile_m * 2 + tile_n * 2) * tile_k * stage_cnt + barrier_bytes + 1023) / 1024
        * 1024; // ((16 * 2 + 4 * 2) * 256 * 22 + 4096 + 1023) / 1024 * 1024 = 230399

    dim3 grid(cta_m_cnt, cta_n_cnt, 1);
    dim3 block_size(256);
    cudaLaunchConfig_t config;
    config.gridDim = grid;
    config.blockDim = block_size;
    config.dynamicSmemBytes = smem_bytes;
    config.stream = stream;
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[0].val.programmaticStreamSerializationAllowed = tensorrt_llm::common::getEnvEnablePDL();
    config.numAttrs = 1;
    config.attrs = attrs;
    if (smem_bytes >= (48 * 1024))
    {
        TLLM_CUDA_CHECK(cudaFuncSetAttribute(
            fuse_a_gemm_kernel<batch_size, gemm_m, gemm_n, gemm_k, tile_m, tile_n, tile_k, stage_cnt>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));
    }
    TLLM_CUDA_CHECK(cudaLaunchKernelEx(&config,
        fuse_a_gemm_kernel<batch_size, gemm_m, gemm_n, gemm_k, tile_m, tile_n, tile_k, stage_cnt>, output, mat_a,
        mat_b));
}

template void invokeFuseAGemm<__nv_bfloat16, 1, 7168, 2112>(
    __nv_bfloat16*, __nv_bfloat16 const*, __nv_bfloat16 const*, cudaStream_t);

template void invokeFuseAGemm<__nv_bfloat16, 2, 7168, 2112>(
    __nv_bfloat16*, __nv_bfloat16 const*, __nv_bfloat16 const*, cudaStream_t);

template void invokeFuseAGemm<__nv_bfloat16, 3, 7168, 2112>(
    __nv_bfloat16*, __nv_bfloat16 const*, __nv_bfloat16 const*, cudaStream_t);

template void invokeFuseAGemm<__nv_bfloat16, 4, 7168, 2112>(
    __nv_bfloat16*, __nv_bfloat16 const*, __nv_bfloat16 const*, cudaStream_t);

template void invokeFuseAGemm<__nv_bfloat16, 5, 7168, 2112>(
    __nv_bfloat16*, __nv_bfloat16 const*, __nv_bfloat16 const*, cudaStream_t);

template void invokeFuseAGemm<__nv_bfloat16, 6, 7168, 2112>(
    __nv_bfloat16*, __nv_bfloat16 const*, __nv_bfloat16 const*, cudaStream_t);

template void invokeFuseAGemm<__nv_bfloat16, 7, 7168, 2112>(
    __nv_bfloat16*, __nv_bfloat16 const*, __nv_bfloat16 const*, cudaStream_t);

template void invokeFuseAGemm<__nv_bfloat16, 8, 7168, 2112>(
    __nv_bfloat16*, __nv_bfloat16 const*, __nv_bfloat16 const*, cudaStream_t);

template void invokeFuseAGemm<__nv_bfloat16, 9, 7168, 2112>(
    __nv_bfloat16*, __nv_bfloat16 const*, __nv_bfloat16 const*, cudaStream_t);

template void invokeFuseAGemm<__nv_bfloat16, 10, 7168, 2112>(
    __nv_bfloat16*, __nv_bfloat16 const*, __nv_bfloat16 const*, cudaStream_t);

template void invokeFuseAGemm<__nv_bfloat16, 11, 7168, 2112>(
    __nv_bfloat16*, __nv_bfloat16 const*, __nv_bfloat16 const*, cudaStream_t);

template void invokeFuseAGemm<__nv_bfloat16, 12, 7168, 2112>(
    __nv_bfloat16*, __nv_bfloat16 const*, __nv_bfloat16 const*, cudaStream_t);

template void invokeFuseAGemm<__nv_bfloat16, 13, 7168, 2112>(
    __nv_bfloat16*, __nv_bfloat16 const*, __nv_bfloat16 const*, cudaStream_t);

template void invokeFuseAGemm<__nv_bfloat16, 14, 7168, 2112>(
    __nv_bfloat16*, __nv_bfloat16 const*, __nv_bfloat16 const*, cudaStream_t);

template void invokeFuseAGemm<__nv_bfloat16, 15, 7168, 2112>(
    __nv_bfloat16*, __nv_bfloat16 const*, __nv_bfloat16 const*, cudaStream_t);

template void invokeFuseAGemm<__nv_bfloat16, 16, 7168, 2112>(
    __nv_bfloat16*, __nv_bfloat16 const*, __nv_bfloat16 const*, cudaStream_t);

} // namespace tensorrt_llm::kernels
